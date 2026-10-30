from __future__ import annotations
import asyncio
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Coroutine

from ..common.exceptions import MCPLifecycleError, MCPTimeoutError


@dataclass
class _Job:
    coroutine: Coroutine[Any, Any, Any] | None
    result: Future[Any] | None
    timeout: float = 0.0


class AsyncRuntime:
    """One private event loop, kept alive for every synchronous operation."""
    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._queue: asyncio.Queue[_Job] | None = None
        self._worker: Future[Any] | None = None
        self._thread = threading.Thread(target=self._run, name="mcp-wrapper", daemon=True)
        self._thread.start(); self._ready.wait()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._queue = asyncio.Queue()
        self._worker = asyncio.run_coroutine_threadsafe(self._work(), self._loop)
        self._ready.set(); self._loop.run_forever()

    async def _work(self) -> None:
        """SDK contexts must all be used in the same async Task."""
        assert self._queue is not None
        while True:
            job = await self._queue.get()
            if job.coroutine is None:
                return
            if job.result is None or job.result.cancelled():
                job.coroutine.close()
                continue

            try:
                # Keep all SDK context entry, use, and exit in this one task.
                # ``asyncio.timeout`` cancels and then restores this task, so a
                # timed-out operation cannot strand the serialized worker.
                async with asyncio.timeout(job.timeout):
                    result = await job.coroutine
            except TimeoutError:
                if job.result is not None and not job.result.cancelled():
                    job.result.set_exception(MCPTimeoutError(f"operation exceeded {job.timeout:g}s"))
            except BaseException as exc:
                if job.result is not None and not job.result.cancelled():
                    job.result.set_exception(exc)
            else:
                if job.result is not None and not job.result.cancelled():
                    job.result.set_result(result)

    def call(self, coro: Coroutine[Any, Any, Any], timeout: float) -> Any:
        if threading.current_thread() is self._thread:
            coro.close()
            raise MCPLifecycleError("synchronous client methods cannot run on the wrapper runtime thread")
        future: Future[Any] = Future()
        job = _Job(coro, future, timeout)
        assert self._queue is not None
        self._loop.call_soon_threadsafe(self._queue.put_nowait, job)
        try:
            # The runtime enforces the requested timeout.  A small scheduling
            # allowance prevents a caller-side race from abandoning the job
            # immediately before the runtime reports its normalized timeout.
            return future.result(timeout + 0.5)
        except TimeoutError as exc:
            # This means the event loop itself is unresponsive, not merely that
            # the operation exceeded its normal deadline.
            future.cancel()
            raise MCPTimeoutError(f"operation exceeded {timeout:g}s") from exc

    def close(self) -> None:
        if self._queue is not None and self._worker is not None:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, _Job(None, None))
            try: self._worker.result(2)
            except Exception: pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2)
        if not self._thread.is_alive(): self._loop.close()
