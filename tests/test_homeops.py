from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from resident.__main__ import TerminalDiagnostics
from resident.config import Config
from resident.homeops import HomeOpsConnector
from resident.runtime import ResidentRuntime
from resident.readiness import ReadinessItem, ReadinessResult


def measurement(point_id: str, value: object, timestamp: str, **extra: object) -> dict:
    return {
        "pointId": point_id, "pointKey": f"key-{point_id}",
        "pointName": f"Point {point_id}", "kind": "temperature", "unit": "C",
        "deviceId": "device-1", "deviceName": "Kitchen", "value": value,
        "timestamp": timestamp, **extra,
    }


class FakeHomeOpsConnector(HomeOpsConnector):
    def __init__(self, responses):
        super().__init__("http://homeops.test")
        self.responses = iter(responses)
        self.requests = []

    async def _request(self, path, query=None):
        self.requests.append((path, query))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


class HomeOpsConnectorTests(unittest.IsolatedAsyncioTestCase):
    async def test_first_poll_is_silent_and_timestamp_only_update_is_suppressed(self):
        connector = FakeHomeOpsConnector([
            [measurement("1", 20.0, "2026-09-15T08:00:00Z")],
            [measurement("1", 20.0, "2026-09-15T08:01:00Z")],
        ])
        queue = asyncio.Queue()

        await connector.poll_once(queue)
        await connector.poll_once(queue)

        self.assertTrue(queue.empty())
        self.assertEqual("2026-09-15T08:01:00Z", connector._snapshot["1"]["timestamp"])

    async def test_changed_and_new_points_are_batched_in_one_wake(self):
        connector = FakeHomeOpsConnector([
            [measurement("1", 20, "old")],
            [measurement("1", 21, "new"), measurement("2", True, "new")],
        ])
        queue = asyncio.Queue()
        await connector.poll_once(queue)
        await connector.poll_once(queue)

        event = queue.get_nowait()
        self.assertEqual(("homeops", "measurement_changed"), (event.source, event.reason))
        self.assertEqual(2, event.payload["change_count"])
        self.assertEqual(20, event.payload["changes"][0]["old_value"])
        self.assertEqual(21, event.payload["changes"][0]["new_value"])
        self.assertIsNone(event.payload["changes"][1]["old_value"])
        self.assertTrue(queue.empty())

    async def test_failed_poll_preserves_baseline_for_next_comparison(self):
        connector = FakeHomeOpsConnector([
            [measurement("1", 20, "old")], RuntimeError("offline"),
            [measurement("1", 22, "new")],
        ])
        queue = asyncio.Queue()
        await connector.poll_once(queue)
        with self.assertRaisesRegex(RuntimeError, "offline"):
            await connector.poll_once(queue)
        await connector.poll_once(queue)

        self.assertEqual(20, queue.get_nowait().payload["changes"][0]["old_value"])

    async def test_read_capabilities_use_expected_endpoints_and_arguments(self):
        latest = [measurement("a/b", 20, "now")]
        history = {"items": [measurement("a/b", 19, "before")]}
        connector = FakeHomeOpsConnector([latest, history])

        current = await connector.get_current_measurements({})
        result = await connector.get_measurement_history({
            "point_id": "a/b", "from_time": "start", "to_time": "end", "limit": 5,
        })

        self.assertEqual(latest, current["measurements"])
        self.assertEqual(history["items"], result["measurements"])
        self.assertEqual(("/api/measurement-points/a%2Fb/history", {
            "from": "start", "to": "end", "limit": 5,
        }), connector.requests[1])

    async def test_run_reports_failure_without_waking_and_stops(self):
        diagnostics = []
        connector = FakeHomeOpsConnector([RuntimeError("offline")])
        connector.diagnostic_output = diagnostics.append
        stop = asyncio.Event()
        connector.poll_seconds = 10
        queue = asyncio.Queue()
        task = asyncio.create_task(connector.run(queue, stop))
        while not diagnostics:
            await asyncio.sleep(0)
        stop.set()
        await task

        self.assertIn("poll failed: RuntimeError: offline", diagnostics)
        self.assertTrue(queue.empty())

    async def test_first_failed_poll_reports_failed_readiness_and_keeps_retrying(self):
        connector = FakeHomeOpsConnector([
            RuntimeError("offline"), [measurement("1", 20, "now")],
        ])
        connector.poll_seconds = 0.01
        stop = asyncio.Event()
        readiness = asyncio.Queue()
        task = asyncio.create_task(connector.run(asyncio.Queue(), stop, readiness))

        self.assertFalse((await readiness.get()).ok)
        while connector._snapshot is None:
            await asyncio.sleep(0)
        stop.set()
        await task

        self.assertEqual(2, len(connector.requests))


class HomeOpsConfigTests(unittest.TestCase):
    def test_homeops_is_disabled_by_default_and_cli_can_enable_it(self):
        with patch.dict(os.environ, {"RESIDENT_HOMEOPS_URL": ""}):
            disabled = Config.from_env_and_args(["--data-dir", ".resident"])
        enabled = Config.from_env_and_args([
            "--data-dir", ".resident", "--homeops-url", "http://homeops.test/",
            "--homeops-poll-seconds", "2.5", "--homeops-request-timeout-seconds", "3",
        ])

        self.assertIsNone(disabled.homeops_url)
        self.assertEqual("http://homeops.test", enabled.homeops_url)
        self.assertEqual(2.5, enabled.homeops_poll_seconds)
        self.assertEqual(3, enabled.homeops_request_timeout_seconds)

    def test_verbose_can_be_enabled_by_environment_or_cli(self):
        with patch.dict(os.environ, {"RESIDENT_VERBOSE": "true"}):
            from_environment = Config.from_env_and_args(["--data-dir", ".resident"])
        with patch.dict(os.environ, {"RESIDENT_VERBOSE": ""}):
            from_cli = Config.from_env_and_args(["--data-dir", ".resident", "--verbose"])

        self.assertTrue(from_environment.verbose)
        self.assertTrue(from_cli.verbose)

    def test_terminal_renderer_suppresses_connector_retries_unless_verbose(self):
        with patch("builtins.print") as output:
            TerminalDiagnostics(verbose=False).homeops("poll failed: offline")
            output.assert_not_called()

            TerminalDiagnostics(verbose=True).homeops("poll failed: offline")
            output.assert_called_once_with("[homeops] poll failed: offline")


class IdleProvider:
    async def respond(self, context, tools, results, previous_response_id=None):
        raise AssertionError("No wake should be processed")


class RecordingProducer:
    def __init__(self):
        self.started = False
        self.stopped = False

    async def run(self, queue, stop):
        self.started = True
        try:
            await stop.wait()
        finally:
            self.stopped = True


class ReadinessProducer(RecordingProducer):
    def __init__(self, key, label, result=None):
        super().__init__()
        self.readiness_items = (ReadinessItem(key, label),)
        self.result = result

    async def run(self, queue, stop, readiness):
        self.started = True
        if self.result is not None:
            readiness.put_nowait(ReadinessResult(self.readiness_items[0].key, self.result))
            await stop.wait()
        self.stopped = True


class EventProducerLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_interactive_runtime_starts_and_stops_event_producers(self):
        with tempfile.TemporaryDirectory() as temporary:
            producer = RecordingProducer()
            runtime = ResidentRuntime(
                Config(Path(temporary)), IdleProvider(), event_producers=[producer],
                owner_output=lambda _: None, diagnostic_output=lambda _: None,
            )
            with patch("resident.runtime.asyncio.to_thread", AsyncMock(return_value="/quit")):
                await runtime.run_interactive()

            self.assertTrue(producer.started)
            self.assertTrue(producer.stopped)
            runtime.close()

    async def test_interactive_runtime_prints_ordered_readiness_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = []
            runtime = ResidentRuntime(
                Config(Path(temporary)), IdleProvider(), event_producers=[
                    ReadinessProducer("first", "First", True),
                    ReadinessProducer("second", "Second", False),
                ], owner_output=lambda _: None, diagnostic_output=output.append,
            )
            with patch("resident.runtime.asyncio.to_thread", AsyncMock(return_value="/quit")):
                await runtime.run_interactive()

            self.assertEqual("First............... OK", output[0])
            self.assertEqual("Second.............. FAILED", output[1])
            self.assertEqual("Startup completed with connector errors.", output[2])
            self.assertIn("is sleeping", output[3])
            runtime.close()

    async def test_producer_exit_before_readiness_is_reported_failed(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = []
            runtime = ResidentRuntime(
                Config(Path(temporary)), IdleProvider(),
                event_producers=[ReadinessProducer("early", "Early")],
                owner_output=lambda _: None, diagnostic_output=output.append,
            )
            with patch("resident.runtime.asyncio.to_thread", AsyncMock(return_value="/quit")):
                await runtime.run_interactive()

            self.assertEqual("Early............... FAILED", output[0])
            runtime.close()


if __name__ == "__main__":
    unittest.main()
