"""FastAPI application exposing the on-demand anomaly inference API."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import Depends, FastAPI, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from .cache import ModelCache, ModelLease
from .config import ConfigurationError, Settings, load_settings
from .errors import (
    InferenceError,
    ModelBusyError,
    ModelCacheError,
    ModelLoadError,
    ModelTooLargeError,
    UnknownModelError,
)
from .features import FEATURE_NAMES, extract_features
from .models import Manifest, load_manifest

logger = logging.getLogger(__name__)

MAX_SAMPLES_PER_SIGNAL = 10_000
_DISCONNECT_POLL_SECONDS = 0.02


class MeasurementRequest(BaseModel):
    """One measurement window for a specific device model."""

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model_id": "pump",
                    "vibration": [2.1, -1.9, 3.4, -3.0, 2.5, -2.2],
                    "current": [12.4, 12.8, 11.9, 12.6, 12.1, 12.9],
                }
            ]
        }
    }

    model_id: str = Field(
        ...,
        description="Device model id, one of the ids returned by GET /v1/models",
        examples=["pump"],
        min_length=1,
        max_length=64,
    )
    vibration: list[float] = Field(
        ...,
        description="Vibration samples in mm/s, ordered by time",
        min_length=1,
        max_length=MAX_SAMPLES_PER_SIGNAL,
    )
    current: list[float] = Field(
        ...,
        description="Drive current samples in A, aligned with the vibration window",
        min_length=1,
        max_length=MAX_SAMPLES_PER_SIGNAL,
    )

    @field_validator("vibration", "current")
    @classmethod
    def _validate_samples(cls, values: list[float]) -> list[float]:
        for value in values:
            if not np.isfinite(value):
                raise ValueError("measurements must be finite numbers")
        return values


class InferenceResponse(BaseModel):
    model_id: str
    model_name: str
    model_version: str
    anomaly_score: float = Field(..., ge=0.0, le=1.0)
    is_anomaly: bool
    decision_threshold: float
    features: dict[str, float] = Field(
        default_factory=dict,
        description="Extracted feature vector used by the model",
    )


class ModelInfo(BaseModel):
    model_id: str
    name: str
    device_type: str
    version: str
    weights_bytes: int


class ModelListResponse(BaseModel):
    models: list[ModelInfo]


class CacheSnapshot(BaseModel):
    budget_bytes: int
    resident_bytes: int
    reserved_bytes: int
    used_bytes: int
    free_bytes: int
    models: list[dict[str, Any]]
    counters: dict[str, int]
    last_eviction: str | None


def _error_payload(code: str, message: str, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"code": code, "message": message}
    payload.update(extra)
    return {"error": payload}


def _cache_error_response(exc: ModelCacheError) -> JSONResponse:
    headers: dict[str, str] = {}
    if isinstance(exc, UnknownModelError):
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=_error_payload(exc.code, str(exc), model_id=exc.model_id),
        )
    if isinstance(exc, ModelBusyError):
        headers["Retry-After"] = "1"
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            headers=headers,
            content=_error_payload(
                exc.code,
                str(exc),
                model_id=exc.model_id,
                budget_bytes=exc.budget_bytes,
                used_bytes=exc.used_bytes,
                required_bytes=exc.required_bytes,
            ),
        )
    if isinstance(exc, ModelTooLargeError):
        return JSONResponse(
            status_code=status.HTTP_507_INSUFFICIENT_STORAGE,
            content=_error_payload(
                exc.code,
                str(exc),
                model_id=exc.model_id,
                budget_bytes=exc.budget_bytes,
                required_bytes=exc.required_bytes,
            ),
        )
    if isinstance(exc, ModelLoadError):
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_error_payload(exc.code, str(exc), model_id=exc.model_id),
        )
    if isinstance(exc, InferenceError):
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_error_payload(exc.code, str(exc)),
        )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_error_payload(exc.code, str(exc)),
    )


async def _wait_for_disconnect(request: Request) -> None:
    while True:
        if await request.is_disconnected():
            return
        await asyncio.sleep(_DISCONNECT_POLL_SECONDS)


async def _acquire_lease(
    cache: ModelCache, model_id: str, request: Request
) -> ModelLease | None:
    """Acquire a lease, giving up cleanly if the client goes away first.

    The load runs in the cache's own task.  Cancelling ``acquire_task`` only
    removes this waiter; the shared load continues for every other request.
    """

    acquire_task = asyncio.create_task(cache.acquire(model_id), name=f"acquire:{model_id}")
    disconnect_task = asyncio.create_task(
        _wait_for_disconnect(request), name=f"disconnect:{model_id}"
    )
    # Retrieve the result/error of an abandoned acquire so asyncio never warns
    # about an unobserved exception.
    acquire_task.add_done_callback(_consume_task_result)

    def abandon() -> None:
        """Drop this waiter; release a lease if it won the race anyway.

        ``cache.acquire`` creates a lease and returns it without any await in
        between, so a task that is not done yet when we cancel it can never
        produce a lease afterwards.  A task that is already done is consumed
        synchronously.  This keeps cleanup safe even when the cleanup itself
        runs while the caller is being cancelled.
        """

        if acquire_task.done():
            if not acquire_task.cancelled():
                try:
                    lease = acquire_task.result()
                except ModelCacheError:
                    lease = None
                except Exception:  # noqa: BLE001 - cleanup must not mask cancel
                    logger.exception("acquire failed while abandoning model %s", model_id)
                    lease = None
                else:
                    lease.release()
            return
        acquire_task.cancel()

    lease: ModelLease | None = None
    try:
        done, _ = await asyncio.wait(
            {acquire_task, disconnect_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if acquire_task in done:
            # Surface cache errors even if the client also disconnected.
            lease = acquire_task.result()
            return lease

        abandon()
        logger.info("client disconnected while waiting for model %s", model_id)
        return None
    except asyncio.CancelledError:
        # Server-side cancellation (shutdown, middleware) rather than a client
        # disconnect: same cleanup, then propagate.
        abandon()
        raise
    finally:
        disconnect_task.cancel()
        try:
            await disconnect_task
        except asyncio.CancelledError:
            # Awaiting a task we just cancelled normally raises with only the
            # child cancelled.  If *this* task was cancelled too, a late
            # cancellation arrived while tidying up; a lease that was already
            # won and is about to be returned would otherwise leak, so return
            # it and let the cancellation surface.
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                if lease is not None:
                    lease.release()
                    lease = None
                raise
        except BaseException:
            # Any other failure here would discard the pending return value,
            # taking an already-won lease with it.
            if lease is not None:
                lease.release()
                lease = None
            raise


def _compute(lease: ModelLease, features: np.ndarray, delay_seconds: float) -> tuple[float, bool]:
    """Synchronous scoring body.

    The simulated latency lives inside this function (rather than around it)
    so that "computation is still running" is an explicit, testable window.
    """

    if delay_seconds > 0:
        # Simulated device-side model latency.  Blocking here would stall the
        # event loop, so the API runs this whole function in a worker thread.
        time.sleep(delay_seconds)
    model = lease.model
    score = model.score(features)
    return score, model.is_anomalous(score)


class _LeaseHandoff:
    """Arbitrates ownership of a lease between the loop and a worker thread.

    Cancelling the future returned by ``run_in_executor`` propagates to the
    underlying ``concurrent.futures.Future``.  If the work item has not started
    yet, that cancel succeeds and the worker body never runs - so "the worker
    will return the lease in its ``finally``" is not universally true.

    This handshake resolves the race exactly once:

    * the worker calls :meth:`try_start_worker` *before* scoring;
    * whoever cancels/fails before the worker starts calls
      :meth:`return_if_not_handed_off`.

    Exactly one side wins the lock.  The winner owns the single
    ``inflight`` decrement; the loser must not touch it.  ``ModelLease.release``
    is idempotent as a second line of defence.
    """

    __slots__ = ("lease", "_lock", "_handed_off", "_abandoned")

    def __init__(self, lease: ModelLease) -> None:
        self.lease = lease
        self._lock = threading.Lock()
        self._handed_off = False
        self._abandoned = False

    @property
    def handed_off(self) -> bool:
        return self._handed_off

    def try_start_worker(self) -> bool:
        """Claim the lease for this worker; ``False`` means it was abandoned."""

        with self._lock:
            if self._abandoned:
                return False
            self._handed_off = True
            return True

    def return_if_not_handed_off(self) -> None:
        """Release the lease iff no worker has claimed it.

        Must run on the event-loop thread (``ModelLease.release`` mutates
        loop-owned state).
        """

        with self._lock:
            if self._handed_off:
                return
            self._abandoned = True
        self.lease.release()


class _AbandonedInference(Exception):
    """Raised by a worker that lost the lease handoff race."""


def _compute_and_hand_back(
    handoff: _LeaseHandoff,
    features: np.ndarray,
    delay_seconds: float,
    loop: asyncio.AbstractEventLoop,
) -> tuple[float, bool]:
    """Worker-thread body that owns the lease until the thread really ends.

    The lease is returned from this function's ``finally`` - i.e. only after
    ``_compute`` has finished - and the return is marshalled onto the loop.
    Cancelling the awaiting coroutine therefore cannot make the cache treat the
    model as idle while this thread is still scoring.
    """

    if not handoff.try_start_worker():
        # The loop already returned the lease because this work item was
        # cancelled before it started; there is nobody left to score for.
        raise _AbandonedInference
    try:
        return _compute(handoff.lease, features, delay_seconds)
    finally:
        # ``ModelLease.release`` performs the decrement synchronously, so
        # scheduling the plain method (not a coroutine) is enough.  A closed
        # loop (interpreter shutdown) has no occupancy left to observe.
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(handoff.lease.release)


async def _execute_inference(
    handoff: _LeaseHandoff, features: np.ndarray, delay_seconds: float
) -> tuple[float, bool]:
    """Run the computation with the worker thread holding the lease.

    There is no unconditional ``finally: release`` here: cancelling the
    executor future returns immediately while the worker thread may still be
    scoring.  ``_compute_and_hand_back`` returns the lease when the thread
    really ends; if the work item never starts, the cancellation path returns
    it instead.
    """

    loop = asyncio.get_running_loop()
    try:
        future = loop.run_in_executor(
            None,
            _compute_and_hand_back,
            handoff,
            features,
            delay_seconds,
            loop,
        )
    except BaseException:
        # Submission failed (e.g. the default executor is already shut down):
        # no thread will ever run, so nobody else can return the lease.
        handoff.return_if_not_handed_off()
        raise
    # If this coroutine is cancelled we stop waiting, but the executor future
    # may still finish with an error (e.g. the worker lost the handoff race).
    # Retrieve it so asyncio does not warn about an unobserved exception.
    future.add_done_callback(_consume_task_result)
    try:
        return await future
    except asyncio.CancelledError:
        # If the work item never started, no worker will ever return the
        # lease.  If it did start, the worker owns it and this is a no-op.
        handoff.return_if_not_handed_off()
        raise
    except _AbandonedInference:
        # Lost the race to the cancellation path, which already released.
        raise asyncio.CancelledError from None


def _consume_task_result(task: asyncio.Future[Any]) -> None:
    # A cancelled request/computation may leave a task or executor future whose
    # result nobody awaits.  Retrieve it so asyncio does not warn about an
    # unobserved exception.
    if not task.cancelled():
        task.exception()


async def _run_inference(
    lease: ModelLease,
    features: np.ndarray,
    delay_seconds: float,
    pending: set[asyncio.Task[Any]],
) -> tuple[float, bool]:
    """Await scoring without letting a cancellation abort or shorten it.

    ``shield`` protects the computation task from the request's cancellation;
    the task hands the lease to a worker which releases it once the thread
    finishes.  The task is also registered in ``pending`` so the app's lifespan
    can wait for it.
    """

    handoff = _LeaseHandoff(lease)
    task = asyncio.create_task(
        _execute_inference(handoff, features, delay_seconds),
        name="inference",
    )
    pending.add(task)

    def _done(finished: asyncio.Task[Any]) -> None:
        pending.discard(finished)
        _consume_task_result(finished)
        # Covers a task cancelled before its coroutine ever ran *and* a work
        # item cancelled while still queued in the executor (the worker never
        # starts, so nobody else can return the lease).  If a worker did start,
        # it owns the lease and this is a no-op.
        handoff.return_if_not_handed_off()

    task.add_done_callback(_done)
    # Awaiting a shield means a cancellation of the request detaches from the
    # computation instead of cancelling it; the task releases the lease itself.
    return await asyncio.shield(task)


def create_app(
    settings: Settings | None = None,
    *,
    manifest: Manifest | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        active_settings = settings or load_settings()
        active_manifest = manifest or load_manifest(active_settings.model_dir)
        cache = ModelCache(
            active_manifest,
            active_settings.model_budget_bytes,
            active_settings.model_load_delay_seconds,
        )
        app.state.settings = active_settings
        app.state.cache = cache
        app.state.feature_names = active_manifest.feature_names
        # Detached computations keep running after their request is gone; the
        # lifespan waits for them so no lease outlives this app instance.
        pending: set[asyncio.Task[Any]] = set()
        app.state.pending_inferences = pending
        logger.info(
            "inference service ready: %d models, budget=%d bytes, inference_delay=%dms",
            len(active_manifest.descriptors),
            active_settings.model_budget_bytes,
            active_settings.inference_delay_ms,
        )
        try:
            yield
        finally:
            if pending:
                await asyncio.gather(*list(pending), return_exceptions=True)
            await cache.aclose()

    app = FastAPI(
        title="Device Anomaly Inference API",
        description=(
            "按设备型号提交振动与电流测值，按需装载固定权重逻辑回归模型并返回异常分数。\n\n"
            "模型权重按需驻留，驻留与装载预留合计不超预算；空间不足且无空闲模型时返回 503。"
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    def get_cache(request: Request) -> ModelCache:
        return request.app.state.cache

    @app.get("/healthz", summary="Liveness probe", tags=["ops"])
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get(
        "/v1/models",
        response_model=ModelListResponse,
        summary="List available device models",
        tags=["inference"],
    )
    async def list_models(cache: ModelCache = Depends(get_cache)) -> ModelListResponse:
        return ModelListResponse(models=[ModelInfo(**item) for item in cache.list_models()])

    @app.post(
        "/v1/infer",
        response_model=InferenceResponse,
        summary="Score one measurement window",
        responses={
            404: {"description": "Unknown device model"},
            422: {"description": "Invalid measurement payload"},
            500: {"description": "Model load or inference failure"},
            503: {"description": "Model cache busy, retry later"},
            507: {"description": "Model cannot fit into the configured budget"},
        },
        tags=["inference"],
    )
    async def infer(
        payload: MeasurementRequest,
        request: Request,
        cache: ModelCache = Depends(get_cache),
    ) -> InferenceResponse | JSONResponse:
        try:
            features = extract_features(payload.vibration, payload.current)
        except ValueError as exc:
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content=_error_payload("invalid_measurement", str(exc)),
            )

        try:
            lease = await _acquire_lease(cache, payload.model_id, request)
        except ModelCacheError as exc:
            return _cache_error_response(exc)

        if lease is None:
            # Client is gone; nothing to return.  A 499-style status is used
            # because the response will not be delivered anyway.
            return JSONResponse(
                status_code=499,
                content=_error_payload(
                    "client_disconnected",
                    "client disconnected while waiting for the model",
                ),
            )

        delay = request.app.state.settings.inference_delay_seconds
        try:
            # Lease release is owned by _execute_inference: doing it here would
            # release the model early when the request is cancelled mid-compute.
            score, is_anomaly = await _run_inference(
                lease, features, delay, request.app.state.pending_inferences
            )
            model = lease.model
        except ModelCacheError as exc:
            return _cache_error_response(exc)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("inference failed for model %s", payload.model_id)
            return _cache_error_response(InferenceError(f"inference failed: {exc}"))

        return InferenceResponse(
            model_id=model.model_id,
            model_name=model.name,
            model_version=model.version,
            anomaly_score=score,
            is_anomaly=is_anomaly,
            decision_threshold=model.decision_threshold,
            features={
                name: float(value)
                for name, value in zip(FEATURE_NAMES, features, strict=True)
            },
        )

    @app.get(
        "/v1/cache",
        response_model=CacheSnapshot,
        summary="Inspect model cache usage",
        tags=["ops"],
    )
    async def cache_snapshot(cache: ModelCache = Depends(get_cache)) -> CacheSnapshot:
        return CacheSnapshot(**cache.snapshot())

    app.state.settings = None
    app.state.cache = None
    return app


try:
    app = create_app()
except ConfigurationError:
    logger.exception("invalid configuration")
    raise
