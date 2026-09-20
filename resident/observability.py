from __future__ import annotations

import asyncio
import time
from contextvars import ContextVar
from typing import Any, Callable, Generic, TypeVar


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
