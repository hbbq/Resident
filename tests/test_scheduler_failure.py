from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from resident.config import Config
from resident.runtime import ResidentRuntime
from resident.store import utc_now


class FailingProvider:
    async def respond(self, context, tools, results, previous_response_id=None):
        raise RuntimeError("simulated provider failure")


class BlockingProvider:
    def __init__(self):
        self.started = asyncio.Event()

    async def respond(self, context, tools, results, previous_response_id=None):
        self.started.set()
        await asyncio.Future()


class ScheduledWakeFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_scheduled_wakeup_is_preserved_without_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = ResidentRuntime(
                Config(Path(temporary)), FailingProvider(),
                owner_output=lambda _: None, diagnostic_output=lambda _: None)
            schedule_id = runtime.store.schedule(utc_now(), "retry me", {})
            queue = asyncio.Queue()
            await runtime.enqueue_due_wakeups(queue)
            event = await queue.get()

            with self.assertRaisesRegex(RuntimeError, "simulated provider failure"):
                await runtime.process(event)

            status = runtime.store.connection.execute(
                "SELECT status FROM scheduled_wakeups WHERE id=?", (schedule_id,)).fetchone()[0]
            self.assertEqual("failed", status)
            run_status = runtime.store.connection.execute(
                "SELECT status FROM wake_runs").fetchone()[0]
            self.assertEqual("failed", run_status)

            await runtime.enqueue_due_wakeups(queue)
            self.assertTrue(queue.empty())
            runtime.close()

    async def test_cancelled_scheduled_wakeup_is_failed_not_completed(self):
        with tempfile.TemporaryDirectory() as temporary:
            provider = BlockingProvider()
            runtime = ResidentRuntime(
                Config(Path(temporary)), provider,
                owner_output=lambda _: None, diagnostic_output=lambda _: None)
            schedule_id = runtime.store.schedule(utc_now(), "cancel me", {})
            queue = asyncio.Queue()
            await runtime.enqueue_due_wakeups(queue)
            event = await queue.get()

            task = asyncio.create_task(runtime.process(event))
            await provider.started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

            schedule_status = runtime.store.connection.execute(
                "SELECT status FROM scheduled_wakeups WHERE id=?", (schedule_id,)).fetchone()[0]
            self.assertEqual("failed", schedule_status)
            run_status = runtime.store.connection.execute(
                "SELECT status FROM wake_runs").fetchone()[0]
            self.assertEqual("failed", run_status)
            runtime.close()


if __name__ == "__main__":
    unittest.main()
