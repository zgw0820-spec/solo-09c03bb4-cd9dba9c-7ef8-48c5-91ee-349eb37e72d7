"""Interleaving tests for lease ownership and release.

These target the two races that used to leak budget:

* a cancelled request returning the lease while the worker thread was still
  scoring (``asyncio.to_thread`` cancellation returns before the thread ends);
* a lease release that is cancelled while it waits for the cache lock, which
  used to leave ``inflight`` pinned forever because the "released" flag had
  already been set.
"""

from __future__ import annotations

import asyncio
import contextlib
import time

import numpy as np
import pytest

from app.api import _execute_inference, _LeaseHandoff
from app.cache import ModelCache
from app.errors import ModelBusyError
from app.models import Manifest
from tests.conftest import MODEL_BYTES

FEATURES = np.zeros(12)


def inflight(cache: ModelCache, model_id: str) -> int:
    for entry in cache.snapshot()["models"]:
        if entry["model_id"] == model_id:
            return int(entry["inflight"])
    return 0


async def test_lease_release_decrements_synchronously_and_only_once(
    settings, manifest: Manifest
):
    cache = ModelCache(manifest, MODEL_BYTES * 2)
    lease = await cache.acquire("alpha")
    assert inflight(cache, "alpha") == 1

    # Calling release schedules nothing: the decrement has already happened
    # before the returned awaitable is awaited.  Cancelling that awaitable
    # therefore cannot strand the occupancy.
    pending_release = lease.release()
    assert inflight(cache, "alpha") == 0
    assert lease.released is True

    task = asyncio.ensure_future(pending_release)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Repeated cleanup is harmless and must not decrement below zero.
    await lease.release()
    await lease.release()
    assert inflight(cache, "alpha") == 0
    assert cache.snapshot()["used_bytes"] == MODEL_BYTES

    # The freed budget is real: a second model now fits.
    other = await cache.acquire("beta")
    assert other.model.model_id == "beta"
    await other.release()
    await cache.aclose()


async def test_release_cancelled_while_cache_lock_is_held_does_not_leak(
    settings, manifest: Manifest
):
    cache = ModelCache(manifest, MODEL_BYTES)
    lease = await cache.acquire("alpha")

    started = asyncio.Event()
    release_lock = asyncio.Event()

    async def hold_lock() -> None:
        async with cache._lock:
            started.set()
            await release_lock.wait()

    holder = asyncio.create_task(hold_lock())
    await started.wait()

    # The decrement happens inside ``release()`` before anything is scheduled,
    # so holding the cache lock cannot delay it.  Cancelling the awaiter (or
    # never awaiting at all) cannot swallow it either.
    releasing = asyncio.ensure_future(lease.release())
    await asyncio.sleep(0)
    releasing.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await releasing

    assert inflight(cache, "alpha") == 0
    release_lock.set()
    await holder

    # Repeated cleanup after the cancellation is a no-op, not a second
    # decrement.
    await lease.release()
    assert inflight(cache, "alpha") == 0
    assert cache.snapshot()["used_bytes"] == MODEL_BYTES

    beta = await cache.acquire("beta")
    assert beta.model.model_id == "beta"
    await beta.release()
    await cache.aclose()


async def test_cancelled_execute_inference_holds_model_until_thread_ends(
    settings, manifest: Manifest
):
    cache = ModelCache(manifest, MODEL_BYTES)
    lease = await cache.acquire("alpha")

    task = asyncio.create_task(_execute_inference(_LeaseHandoff(lease), FEATURES, 0.4))
    await asyncio.sleep(0.05)
    started = asyncio.get_running_loop().time()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert asyncio.get_running_loop().time() - started < 0.1

    # The coroutine is cancelled but the worker thread is still scoring: the
    # model must stay pinned and its budget must not be handed to anyone else.
    assert inflight(cache, "alpha") == 1
    with pytest.raises(ModelBusyError):
        await cache.acquire("beta")

    while inflight(cache, "alpha") > 0:  # noqa: ASYNC110
        await asyncio.sleep(0.01)

    # Exactly one decrement, and the budget is fully available again.
    assert cache.snapshot()["used_bytes"] == MODEL_BYTES
    await lease.release()
    assert inflight(cache, "alpha") == 0

    beta = await cache.acquire("beta")
    assert beta.model.model_id == "beta"
    await beta.release()
    await cache.aclose()


async def test_concurrent_duplicate_cleanup_decrements_once(
    settings, manifest: Manifest
):
    cache = ModelCache(manifest, MODEL_BYTES)
    lease = await cache.acquire("alpha")

    # Simulate two cleanup paths racing (request cancellation plus shutdown).
    await asyncio.gather(lease.release(), lease.release(), lease.release())
    assert inflight(cache, "alpha") == 0

    beta = await cache.acquire("beta")
    assert beta.model.model_id == "beta"
    await beta.release()
    await cache.aclose()


async def test_task_cancelled_before_submission_returns_lease_once(
    settings, manifest: Manifest
):
    """A task cancelled before its coroutine body runs never hands off.

    No worker will ever exist, so the done-callback fallback must return the
    lease - exactly once, even though the cancellation path also runs.
    """

    cache = ModelCache(manifest, MODEL_BYTES)
    lease = await cache.acquire("alpha")
    handoff = _LeaseHandoff(lease)

    task = asyncio.create_task(_execute_inference(handoff, FEATURES, 0.2))
    task.cancel()  # before the first scheduling step
    with pytest.raises(asyncio.CancelledError):
        await task

    # Mirror the production done-callback.
    handoff.return_if_not_handed_off()
    handoff.return_if_not_handed_off()
    assert handoff.handed_off is False
    assert inflight(cache, "alpha") == 0

    await lease.release()
    beta = await cache.acquire("beta")
    assert beta.model.model_id == "beta"
    await beta.release()
    await cache.aclose()


async def test_work_item_cancelled_while_queued_returns_lease(
    settings, manifest: Manifest
):
    """Cancelling an executor future only stops the job if it has not started.

    With a single worker already blocked, the inference work item sits in the
    queue; the cancellation succeeds, the worker body never runs, and the lease
    must be returned by the loop side rather than leaking.
    """

    from concurrent.futures import ThreadPoolExecutor

    pool = ThreadPoolExecutor(max_workers=1)
    loop = asyncio.get_running_loop()
    # pytest-asyncio gives every test its own loop, so no executor restore is
    # needed; the worker pool is shut down below.
    loop.set_default_executor(pool)
    try:
        cache = ModelCache(manifest, MODEL_BYTES)
        lease = await cache.acquire("alpha")
        handoff = _LeaseHandoff(lease)

        release_blocker = asyncio.Event()

        def block() -> None:
            # Occupy the only worker until the test says otherwise.
            while not release_blocker.is_set():
                time.sleep(0.005)

        blocker = loop.run_in_executor(None, block)
        task = asyncio.create_task(_execute_inference(handoff, FEATURES, 0.0))
        await asyncio.sleep(0.05)  # let the inference job queue up
        assert handoff.handed_off is False

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # No manual cleanup: ``_execute_inference``'s own cancellation path must
        # notice that the work item never started and return the lease.

        assert inflight(cache, "alpha") == 0
        assert cache.snapshot()["used_bytes"] == MODEL_BYTES

        release_blocker.set()
        await blocker

        # The freed budget is usable by another model.  (Do this before the
        # pool is shut down: model loading also uses the default executor.)
        beta = await cache.acquire("beta")
        assert beta.model.model_id == "beta"
        await beta.release()
        await cache.aclose()
    finally:
        pool.shutdown(wait=True)
