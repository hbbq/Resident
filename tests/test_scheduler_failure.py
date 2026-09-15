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


class ScheduledWakeFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_scheduled_wakeup_returns_to_pending(self):
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
            self.assertEqual("pending", status)
            runtime.close()


if __name__ == "__main__":
    unittest.main()
