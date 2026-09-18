from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from resident.camera import CameraConnector
from resident.config import CameraConfig, Config
from resident.domain import ModelTurn, ToolCall, WakeEvent
from resident.runtime import ResidentRuntime
from resident.store import utc_now


JPEG = b"\xff\xd8frame-data\xff\xd9"
SECRET_URL = "rtsp://secret-user:secret-password@camera.test/live"


class FakeStream:
    def __init__(self, data: bytes):
        self.data = data

    async def read(self, _: int) -> bytes:
        data, self.data = self.data, b""
        return data


class BlockingStream:
    async def read(self, _: int) -> bytes:
        await asyncio.Event().wait()
        return b""


class FakeProcess:
    def __init__(self, data: bytes, returncode: int = 0, *, stderr: bytes = b"", waits: bool = False):
        self.stdout = FakeStream(data)
        self.stderr = FakeStream(stderr)
        self.returncode = None
        self._exit_code = returncode
        self.waits = waits
        self.killed = False

    async def wait(self) -> int:
        if self.waits and not self.killed:
            await asyncio.Event().wait()
        self.returncode = -9 if self.killed else self._exit_code
        return self.returncode

    def kill(self) -> None:
        self.killed = True


class CameraConnectorTests(unittest.IsolatedAsyncioTestCase):
    def connector(self, process: FakeProcess, **options) -> tuple[CameraConnector, list]:
        calls = []

        async def factory(*args, **kwargs):
            calls.append((args, kwargs))
            return process

        camera = CameraConfig("entry", "Entry", SECRET_URL, "Front entry")
        return CameraConnector([camera], process_factory=factory, **options), calls

    async def test_list_returns_safe_metadata_without_starting_ffmpeg(self):
        connector, calls = self.connector(FakeProcess(JPEG))

        result = await connector.list_cameras({})

        self.assertEqual({"cameras": [{"id": "entry", "name": "Entry", "description": "Front entry"}]}, result)
        self.assertEqual([], calls)
        self.assertNotIn(SECRET_URL, json.dumps(result))

    async def test_capture_returns_ephemeral_jpeg_attachment(self):
        connector, calls = self.connector(FakeProcess(JPEG))

        result = await connector.capture_frame({"camera_id": "entry"})

        self.assertEqual("captured", result.output["status"])
        self.assertEqual(JPEG, result.attachments[0].data)
        self.assertEqual("image/jpeg", result.attachments[0].mime_type)
        self.assertIn(SECRET_URL, calls[0][0])
        self.assertNotIn(SECRET_URL, json.dumps(result.output))

    async def test_camera_stream_failure_is_unavailable(self):
        connector, _ = self.connector(FakeProcess(
            b"diagnostic text", returncode=1, stderr=b"Connection refused"))

        result = await connector.capture_frame({"camera_id": "entry"})

        self.assertEqual("unavailable", result.output["status"])
        self.assertEqual((), result.attachments)
        self.assertNotIn("secret", json.dumps(result.output))

    async def test_local_ffmpeg_configuration_failure_is_error(self):
        connector, _ = self.connector(FakeProcess(
            b"diagnostic text", returncode=1, stderr=b"Unknown encoder 'mjpeg'"))

        result = await connector.capture_frame({"camera_id": "entry"})

        self.assertEqual("error", result.output["status"])
        self.assertEqual("Frame capture failed locally.", result.output["description"])
        self.assertNotIn("mjpeg", json.dumps(result.output))

    async def test_process_start_failure_is_sanitized(self):
        async def failed_factory(*args, **kwargs):
            raise RuntimeError(f"could not open {SECRET_URL}")

        connector = CameraConnector(
            [CameraConfig("entry", "Entry", SECRET_URL)], process_factory=failed_factory)

        result = await connector.capture_frame({"camera_id": "entry"})

        self.assertEqual("error", result.output["status"])
        self.assertNotIn(SECRET_URL, json.dumps(result.output))

    async def test_timeout_kills_ffmpeg_and_is_a_normal_outcome(self):
        process = FakeProcess(b"", waits=True)
        connector, _ = self.connector(process, timeout_seconds=0.01)

        result = await connector.capture_frame({"camera_id": "entry"})

        self.assertEqual("timeout", result.output["status"])
        self.assertTrue(process.killed)

    async def test_launch_timeout_is_a_normal_outcome(self):
        async def stalled_factory(*args, **kwargs):
            await asyncio.Event().wait()

        connector = CameraConnector(
            [CameraConfig("entry", "Entry", SECRET_URL)],
            process_factory=stalled_factory,
            timeout_seconds=0.01,
        )

        result = await connector.capture_frame({"camera_id": "entry"})

        self.assertEqual("timeout", result.output["status"])

    async def test_cancellation_kills_ffmpeg_and_reraises(self):
        process = FakeProcess(b"")
        process.stdout = BlockingStream()
        connector, _ = self.connector(process)

        capture = asyncio.create_task(connector.capture_frame({"camera_id": "entry"}))
        await asyncio.sleep(0)
        capture.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await capture
        self.assertTrue(process.killed)

    async def test_oversized_frame_is_rejected(self):
        connector, _ = self.connector(FakeProcess(JPEG + b"x" * 100), max_bytes=len(JPEG))

        result = await connector.capture_frame({"camera_id": "entry"})

        self.assertEqual("error", result.output["status"])
        self.assertEqual((), result.attachments)

    async def test_refresh_emits_one_safe_batched_camera_change(self):
        connector, calls = self.connector(FakeProcess(JPEG))
        queue = asyncio.Queue()
        stop = asyncio.Event()
        producer = asyncio.create_task(connector.run(queue, stop))
        await asyncio.sleep(0)
        replacement_url = "rtsp://different-secret@camera.test/live"

        event = connector.replace_cameras([
            CameraConfig("entry", "Renamed", replacement_url, "New description"),
            CameraConfig("yard", "Yard", "rtsp://yard-secret@camera.test/live"),
        ])

        self.assertIs(event, queue.get_nowait())
        self.assertEqual(("camera", "cameras_changed"), (event.source, event.reason))
        self.assertEqual([{"id": "yard", "name": "Yard"}], event.payload["added"])
        self.assertEqual([], event.payload["removed"])
        self.assertEqual([{
            "id": "entry", "name": "Renamed", "description": "New description",
        }], event.payload["changed"])
        self.assertNotIn("rtsp", json.dumps(event.payload))
        self.assertEqual([], calls)
        stop.set()
        await producer

    async def test_camera_refresh_suppresses_noop_and_detects_secret_endpoint_change(self):
        connector, calls = self.connector(FakeProcess(JPEG))
        same = CameraConfig("entry", "Entry", SECRET_URL, "Front entry")

        self.assertIsNone(connector.replace_cameras([same]))
        event = connector.replace_cameras([
            CameraConfig("entry", "Entry", "rtsp://new-secret@camera.test/live", "Front entry")
        ])

        self.assertEqual([{"id": "entry", "name": "Entry", "description": "Front entry"}],
                         event.payload["changed"])
        self.assertNotIn("new-secret", json.dumps(event.payload))
        self.assertEqual([], calls)


class CameraConfigTests(unittest.TestCase):
    def test_camera_configuration_is_opt_in_and_redacts_repr(self):
        with patch.dict(os.environ, {"RESIDENT_CAMERAS": ""}):
            disabled = Config.from_env_and_args(["--data-dir", ".resident"])
        with patch.dict(os.environ, {"RESIDENT_CAMERAS": json.dumps([{
            "id": "entry", "name": "Entry", "url": SECRET_URL, "description": "Door",
        }])}):
            enabled = Config.from_env_and_args(["--data-dir", ".resident"])

        self.assertEqual((), disabled.cameras)
        self.assertEqual("entry", enabled.cameras[0].id)
        self.assertNotIn(SECRET_URL, repr(enabled.cameras[0]))

    def test_duplicate_camera_ids_are_rejected(self):
        cameras = [{"id": "same", "name": "One", "url": SECRET_URL},
                   {"id": "same", "name": "Two", "url": SECRET_URL}]
        with patch.dict(os.environ, {"RESIDENT_CAMERAS": json.dumps(cameras)}):
            with self.assertRaisesRegex(ValueError, "Duplicate camera id"):
                Config.from_env_and_args(["--data-dir", ".resident"])

    def test_onvif_configuration_is_explicit_and_secrets_are_redacted(self):
        configured = [{
            "id": "entry", "name": "Entry", "url": SECRET_URL,
            "onvif": {
                "endpoint": "http://camera.test/onvif/device_service",
                "username": "onvif-user", "password": "onvif-password",
            },
        }]
        with patch.dict(os.environ, {"RESIDENT_CAMERAS": json.dumps(configured)}):
            config = Config.from_env_and_args(["--data-dir", ".resident"])

        camera = config.cameras[0]
        self.assertEqual("http://camera.test/onvif/device_service", camera.onvif.endpoint)
        rendered = repr(camera)
        self.assertNotIn("camera.test", rendered)
        self.assertNotIn("onvif-user", rendered)
        self.assertNotIn("onvif-password", rendered)

    def test_partial_or_non_http_onvif_configuration_is_rejected(self):
        for onvif in (
            {"endpoint": "http://camera.test/onvif", "username": "user"},
            {"endpoint": "rtsp://camera.test/onvif", "username": "user", "password": "pass"},
        ):
            configured = [{"id": "entry", "name": "Entry", "url": SECRET_URL, "onvif": onvif}]
            with self.subTest(onvif=onvif), patch.dict(
                    os.environ, {"RESIDENT_CAMERAS": json.dumps(configured)}):
                with self.assertRaisesRegex(ValueError, "onvif|ONVIF"):
                    Config.from_env_and_args(["--data-dir", ".resident"])

    def test_onvif_configuration_rejects_invalid_effective_ports(self):
        for endpoint in ("http://camera.test:99999/onvif", "https://camera.test:0/onvif"):
            configured = [{
                "id": "entry", "name": "Entry", "url": SECRET_URL,
                "onvif": {"endpoint": endpoint, "username": "user", "password": "pass"},
            }]
            with self.subTest(endpoint=endpoint), patch.dict(
                    os.environ, {"RESIDENT_CAMERAS": json.dumps(configured)}):
                with self.assertRaisesRegex(ValueError, "ONVIF endpoint.*valid port"):
                    Config.from_env_and_args(["--data-dir", ".resident"])


class CaptureProvider:
    def __init__(self):
        self.results = None

    async def respond(self, context, tools, results, previous_response_id=None):
        if previous_response_id is None:
            return ModelTurn("capture", tool_calls=(ToolCall(
                "call", "camera_capture_frame", {"camera_id": "entry"}),))
        self.results = results
        return ModelTurn("done", message="done")


class RecoveringCaptureProvider(CaptureProvider):
    def bind_action_store(self, begin, complete):
        self._begin_action = begin
        self._complete_action = complete

    def prepare_tool_call(self, call):
        action = self._begin_action(
            "openai_agents", "session", "capture", call.id, call.name, call.arguments)
        if not action["claimed"] and action.get("attachments_ephemeral"):
            action["ephemeral_result"] = None
        return action

    def record_tool_result(self, result):
        self._complete_action(
            "openai_agents", "session", result.call_id, result.output,
            bool(result.attachments))


class CameraRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_journal_contains_attachment_metadata_but_no_image_or_url(self):
        connector = CameraConnector([CameraConfig("entry", "Entry", SECRET_URL)])

        async def capture(_):
            from resident.domain import ImageAttachment, ToolOutput
            return ToolOutput({"status": "captured", "camera_id": "entry"}, (ImageAttachment(JPEG),))

        connector.capabilities[1] = connector.capabilities[1].__class__(
            **{**connector.capabilities[1].__dict__, "handler": capture})
        provider = CaptureProvider()
        with tempfile.TemporaryDirectory() as temporary:
            runtime = ResidentRuntime(
                Config(Path(temporary)), provider, capabilities=connector.capabilities,
                owner_output=lambda _: None, diagnostic_output=lambda _: None,
            )
            await runtime.process(WakeEvent("event", "test", "capture", utc_now(), {}))
            rows = [row[0] for row in runtime.store.connection.execute("SELECT data_json FROM journal")]
            journal = "\n".join(rows)
            runtime.close()

        self.assertEqual(JPEG, provider.results[0].attachments[0].data)
        self.assertTrue(provider.results[0].output["ok"])
        self.assertIn('"ephemeral":true', journal)
        self.assertNotIn(SECRET_URL, journal)
        self.assertNotIn("frame-data", journal)

    async def test_completed_camera_action_is_reacquired_after_ephemeral_frame_is_lost(self):
        captures = 0
        connector = CameraConnector([CameraConfig("entry", "Entry", SECRET_URL)])

        async def capture(_):
            nonlocal captures
            captures += 1
            from resident.domain import ImageAttachment, ToolOutput
            return ToolOutput(
                {"status": "captured", "camera_id": "entry"},
                (ImageAttachment(JPEG),))

        connector.capabilities[1] = connector.capabilities[1].__class__(
            **{**connector.capabilities[1].__dict__, "handler": capture})
        provider = RecoveringCaptureProvider()
        with tempfile.TemporaryDirectory() as temporary:
            runtime = ResidentRuntime(
                Config(Path(temporary)), provider, capabilities=connector.capabilities,
                owner_output=lambda _: None, diagnostic_output=lambda _: None,
            )
            runtime.store.begin_agent_tool_action(
                "openai_agents", "session", "capture", "call",
                "camera_capture_frame", {"camera_id": "entry"})
            runtime.store.complete_agent_tool_action(
                "openai_agents", "session", "call",
                {"ok": True, "status": "captured", "camera_id": "entry"}, True)

            await runtime.process(WakeEvent("event", "test", "capture", utc_now(), {}))
            runtime.close()

        self.assertEqual(1, captures)
        self.assertEqual(JPEG, provider.results[0].attachments[0].data)


if __name__ == "__main__":
    unittest.main()
