"""End-to-end server-side cancellation interleavings through the ASGI app.

Client disconnects are covered in ``test_disconnect.py``.  These tests cancel
the *request task* the way a server shutdown or timeout middleware would, and
then verify the budget is actually usable again - not merely that ``inflight``
looks zero.
"""

from __future__ import annotations

import asyncio

import pytest

from app.api import create_app
from app.errors import ModelBusyError
from app.models import Manifest
from tests.conftest import MODEL_BYTES
from tests.helpers import RawASGIRequest


def payload(model_id: str) -> dict:
    return {
        "model_id": model_id,
        "vibration": [1.0 + (index % 4) for index in range(16)],
        "current": [5.0 + (index % 2) for index in range(16)],
    }


async def test_cancelled_inference_pins_model_until_thread_ends_then_frees_budget(
    manifest: Manifest, settings
):
    """Server-side cancellation of the computation task, not a client drop.

    A shutdown/timeout handler cancels the detached inference task itself.  The
    worker thread keeps scoring, so the model must stay pinned and its budget
    must not be handed to another model until the thread has really ended.
    """

    tight = settings.__class__(
        **{**settings.__dict__, "model_budget_bytes": MODEL_BYTES, "inference_delay_ms": 400}
    )
    app = create_app(tight, manifest=manifest)

    async with app.router.lifespan_context(app):
        cache = app.state.cache
        request = RawASGIRequest(app, "/v1/infer", payload("alpha"))
        request_task = asyncio.create_task(request.run())
        await asyncio.sleep(0.15)
        assert _inflight(cache, "alpha") == 1

        # Cancel the computation the way server-side cancellation does.
        pending = list(app.state.pending_inferences)
        assert pending, "the inference task should be registered while it runs"
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        await asyncio.gather(request_task, return_exceptions=True)

        # The request is gone, but the worker thread is still scoring.  The
        # model must stay pinned and cannot be evicted to make room for beta.
        assert _inflight(cache, "alpha") == 1
        with pytest.raises(ModelBusyError):
            await cache.acquire("beta")

        # Only once the thread has really finished does the occupancy drop,
        # exactly once.
        deadline = asyncio.get_running_loop().time() + 2.0
        while _inflight(cache, "alpha") > 0:  # noqa: ASYNC110
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0.02)
        assert cache.snapshot()["used_bytes"] == MODEL_BYTES

        beta = await cache.acquire("beta")
        assert beta.model.model_id == "beta"
        await beta.release()


async def test_cancelled_wait_for_cold_load_then_retry_succeeds(
    manifest: Manifest, settings
):
    slow = settings.__class__(
        **{**settings.__dict__, "model_load_delay_ms": 250, "model_budget_bytes": MODEL_BYTES}
    )
    app = create_app(slow, manifest=manifest)

    async with app.router.lifespan_context(app):
        cache = app.state.cache
        request = RawASGIRequest(app, "/v1/infer", payload("alpha"))
        request_task = asyncio.create_task(request.run())
        await asyncio.sleep(0.05)

        request_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request_task

        snapshot = cache.snapshot()
        # The reservation belongs to the cache, not to the cancelled waiter.
        assert snapshot["counters"]["load_attempts"] == 1
        assert snapshot["reserved_bytes"] == MODEL_BYTES

        # A fresh request joins (or picks up) the same load and succeeds.
        lease = await cache.acquire("alpha")
        assert lease.model.model_id == "alpha"
        await lease.release()

        snapshot = cache.snapshot()
        assert snapshot["counters"]["load_attempts"] == 1
        assert snapshot["counters"]["load_succeeded"] == 1
        assert snapshot["reserved_bytes"] == 0
        assert snapshot["resident_bytes"] == MODEL_BYTES


async def test_late_cancellation_after_lease_is_won_returns_it(
    manifest: Manifest, settings, monkeypatch
):
    """A cancellation delivered during post-acquire cleanup must not leak.

    ``_acquire_lease`` wins the lease, then tidies up its disconnect watcher.
    If the request task is cancelled exactly while it awaits that cleanup, the
    lease would be dropped with the discarded return value.  The poller is
    slowed down here to hold that window open deterministically.
    """

    import app.api as api

    class _IdleRequest:
        async def is_disconnected(self) -> bool:
            return False

    async def slow_poller(request) -> None:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            await asyncio.sleep(0.15)

    monkeypatch.setattr(api, "_wait_for_disconnect", slow_poller)

    cache = api.ModelCache(manifest, MODEL_BYTES)
    task = asyncio.create_task(api._acquire_lease(cache, "alpha", _IdleRequest()))
    await asyncio.sleep(0.05)  # lease won; task is now blocked in cleanup
    assert _inflight(cache, "alpha") == 1

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.sleep(0.2)
    # Exactly one decrement, and the budget is usable again.
    assert _inflight(cache, "alpha") == 0
    beta = await cache.acquire("beta")
    assert beta.model.model_id == "beta"
    await beta.release()
    await cache.aclose()


def _inflight(cache, model_id: str) -> int:
    for entry in cache.snapshot()["models"]:
        if entry["model_id"] == model_id:
            return int(entry["inflight"])
    return 0
