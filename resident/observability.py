from __future__ import annotations

import asyncio
import time
from contextvars import ContextVar
from typing import Any, Callable, Generic, Iterable, TypeVar


TimelineReporter = Callable[[dict[str, Any]], None]
timeline_reporter: ContextVar[TimelineReporter | None] = ContextVar(
    "resident_timeline_reporter", default=None)

T = TypeVar("T")


class ObservedQueue(asyncio.Queue[T], Generic[T]):
    """An asyncio queue whose successful puts are observed at the boundary."""

    def __init__(self, on_put: Callable[[T, int], None]):
        super().__init__()
        self._on_put = on_put

    def put_nowait(self, item: T) -> None:
        super().put_nowait(item)
        self._on_put(item, self.qsize())


class EventLoopLagProbe:
    """Sample loop scheduling lag and support an explicit sampling checkpoint."""

    def __init__(self, observers: Iterable[Callable[[float], None]],
                 interval: float = 0.5):
        self._observers = tuple(observers)
        self._interval = interval
        self._wake = asyncio.Event()
        self._ready = asyncio.Event()
        self._waiters: list[asyncio.Future[None]] = []
        self._task: asyncio.Task[None] | None = None
        self._stopping = False

    async def start(self) -> None:
        if self._task is not None:
            raise RuntimeError("Event-loop lag probe is already started")
        self._task = asyncio.create_task(self._run())
        await self._ready.wait()

    async def checkpoint(self) -> None:
        """Wait until the probe has observed any currently overdue sample."""
        task = self._task
        if task is None or task.done():
            return
        waiter = asyncio.get_running_loop().create_future()
        self._waiters.append(waiter)
        self._wake.set()
        await waiter

    async def stop(self) -> None:
        task = self._task
        if task is None:
            return
        self._stopping = True
        self._wake.set()
        await task
        self._task = None

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        expected = loop.time() + self._interval
        self._ready.set()
        try:
            while not self._stopping:
                try:
                    await asyncio.wait_for(
                        self._wake.wait(),
                        timeout=max(0.0, expected - loop.time()),
                    )
                except TimeoutError:
                    pass
                self._wake.clear()
                now = loop.time()
                if now >= expected:
                    lag = max(0.0, now - expected)
                    for observer in self._observers:
                        observer(lag)
                    # Observation is outside the next interval so the probe does
                    # not attribute its own bookkeeping to loop lag.
                    expected = loop.time() + self._interval
                waiters, self._waiters = self._waiters, []
                for waiter in waiters:
                    if not waiter.done():
                        waiter.set_result(None)
        finally:
            waiters, self._waiters = self._waiters, []
            for waiter in waiters:
                if not waiter.done():
                    waiter.set_result(None)


def emit_timeline(operation: str, moment: str, **fields: Any) -> None:
    reporter = timeline_reporter.get()
    if reporter is not None:
        reporter({"operation": operation, "moment": moment, **fields})


async def to_thread_timed(operation: str, function: Callable[..., Any], *args: Any,
                          timeline_before_finished: Callable[[], None] | None = None,
                          **fields: Any) -> Any:
    """Run blocking work while separating executor queue and execution time."""
    queued_at = time.monotonic()
    timing: dict[str, float] = {}
    emit_timeline(
        operation, "started", started_monotonic_seconds=queued_at, **fields)

    def invoke() -> Any:
        timing["started"] = time.monotonic()
        try:
            return function(*args)
        finally:
            timing["ended"] = time.monotonic()

    outcome = "ok"
    try:
        return await asyncio.to_thread(invoke)
    except BaseException:
        outcome = "error"
        raise
    finally:
        resumed_at = time.monotonic()
        worker_started = timing.get("started", resumed_at)
        worker_ended = timing.get("ended", resumed_at)
        if timeline_before_finished is not None:
            timeline_before_finished()
        emit_timeline(
            operation, "finished", outcome=outcome,
            duration_seconds=resumed_at - queued_at,
            started_monotonic_seconds=queued_at,
            finished_monotonic_seconds=resumed_at,
            executor_queue_seconds=max(0.0, worker_started - queued_at),
            worker_seconds=max(0.0, worker_ended - worker_started),
            **fields,
        )
