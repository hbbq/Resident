from __future__ import annotations

import asyncio
from contextlib import suppress
import time
import uuid
from typing import Awaitable, Callable, Sequence

from .capabilities import Capability
from .config import CameraConfig
from .domain import ImageAttachment, ToolOutput, WakeEvent
from .onvif import OnvifClient
from .store import utc_now


ProcessFactory = Callable[..., Awaitable[asyncio.subprocess.Process]]
OnvifClientFactory = Callable[..., OnvifClient]


class CameraConnector:
    """Read-only, on-demand capture for configured RTSP cameras."""

    def __init__(self, cameras: Sequence[CameraConfig], *, timeout_seconds: float = 8.0,
                 max_width: int = 1280, max_height: int = 720, max_bytes: int = 2_000_000,
                 rtsp_transport: str = "tcp", ffmpeg_executable: str = "ffmpeg",
                 process_factory: ProcessFactory = asyncio.create_subprocess_exec,
                 onvif_request_timeout_seconds: float = 10.0,
                 onvif_pull_timeout_seconds: float = 30.0,
                 onvif_retry_seconds: float = 30.0,
                 onvif_client_factory: OnvifClientFactory = OnvifClient,
                 diagnostic_output: Callable[[str], None] | None = None):
        self._cameras = {camera.id: camera for camera in cameras}
        if len(self._cameras) != len(cameras):
            raise ValueError("Duplicate camera id")
        self.timeout_seconds = timeout_seconds
        self.max_width, self.max_height, self.max_bytes = max_width, max_height, max_bytes
        self.rtsp_transport, self.ffmpeg_executable = rtsp_transport, ffmpeg_executable
        self._process_factory = process_factory
        self.onvif_request_timeout_seconds = max(0.1, onvif_request_timeout_seconds)
        self.onvif_pull_timeout_seconds = max(1.0, onvif_pull_timeout_seconds)
        self.onvif_retry_seconds = max(0.1, onvif_retry_seconds)
        self._onvif_client_factory = onvif_client_factory
        self.diagnostic_output = diagnostic_output or (lambda _: None)
        self._runtime_queue: asyncio.Queue[WakeEvent] | None = None
        self._pending_events: list[WakeEvent] = []
        self.capabilities = [
            Capability(
                connector_id="camera", connector_description="Configured read-only cameras",
                name="camera_list", description="List configured cameras without probing their availability.",
                input_schema={"type": "object", "properties": {}, "additionalProperties": False},
                handler=self.list_cameras,
            ),
            Capability(
                connector_id="camera", connector_description="Configured read-only cameras",
                name="camera_capture_frame",
                description="Attempt an on-demand capture of one current frame from a configured camera.",
                input_schema={
                    "type": "object", "properties": {"camera_id": {"type": "string"}},
                    "required": ["camera_id"], "additionalProperties": False,
                },
                handler=self.capture_frame,
            ),
        ]

    def replace_cameras(self, cameras: Sequence[CameraConfig]) -> WakeEvent | None:
        """Replace a refreshable camera source and emit one safe domain wake."""
        replacement = {camera.id: camera for camera in cameras}
        if len(replacement) != len(cameras):
            raise ValueError("Duplicate camera id")
        previous = self._cameras
        previous_ids, current_ids = set(previous), set(replacement)
        changed_ids = sorted(
            camera_id for camera_id in previous_ids & current_ids
            if previous[camera_id] != replacement[camera_id]
        )
        if previous_ids == current_ids and not changed_ids:
            return None
        self._cameras = replacement
        event = WakeEvent(str(uuid.uuid4()), "camera", "cameras_changed", utc_now(), {
            "added": [self._safe_metadata(replacement[camera_id])
                      for camera_id in sorted(current_ids - previous_ids)],
            "removed": [self._safe_metadata(previous[camera_id])
                        for camera_id in sorted(previous_ids - current_ids)],
            "changed": [self._safe_metadata(replacement[camera_id])
                        for camera_id in changed_ids],
        })
        if self._runtime_queue is None:
            self._pending_events.append(event)
        else:
            self._runtime_queue.put_nowait(event)
        return event

    async def run(self, queue: asyncio.Queue[WakeEvent], stop: asyncio.Event) -> None:
        self._runtime_queue = queue
        for event in self._pending_events:
            queue.put_nowait(event)
        self._pending_events.clear()
        workers = [asyncio.create_task(self._run_onvif(camera, stop))
                   for camera in self._cameras.values() if camera.onvif is not None]
        try:
            await stop.wait()
        finally:
            for worker in workers:
                worker.cancel()
            for worker in workers:
                with suppress(asyncio.CancelledError):
                    await worker
            self._runtime_queue = None

    async def _run_onvif(self, camera: CameraConfig, stop: asyncio.Event) -> None:
        assert camera.onvif is not None
        while not stop.is_set():
            pullpoint = None
            client = None
            recreate = False
            try:
                client = self._onvif_client_factory(
                    camera.onvif, request_timeout=self.onvif_request_timeout_seconds,
                    pull_timeout=self.onvif_pull_timeout_seconds)
                service, topics = await client.discover()
                topic_text = ", ".join(topics) if topics else "none advertised"
                self.diagnostic_output(
                    f"{camera.id}: ONVIF event topics ({len(topics)}): {topic_text}")
                pullpoint = await client.subscribe(service)
                self.diagnostic_output(f"{camera.id}: ONVIF PullPoint subscription active")
                while not stop.is_set():
                    if pullpoint.expires_within(
                            self.onvif_pull_timeout_seconds + self.onvif_request_timeout_seconds):
                        recreate = True
                        break
                    notifications = await client.pull(pullpoint)
                    for notification in notifications:
                        self.diagnostic_output(
                            f"{camera.id}: observed ONVIF event shape: "
                            f"topic={notification['topic']!r}, fields={notification['fields']!r}")
                    if not notifications:
                        try:
                            await asyncio.wait_for(stop.wait(), min(1.0, self.onvif_retry_seconds))
                        except TimeoutError:
                            pass
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.diagnostic_output(
                    f"{camera.id}: ONVIF probe/pull failed ({type(exc).__name__}); retrying")
            finally:
                if client is not None and pullpoint is not None:
                    with suppress(Exception):
                        await asyncio.shield(client.unsubscribe(pullpoint))
            if recreate:
                continue
            try:
                await asyncio.wait_for(stop.wait(), self.onvif_retry_seconds)
            except TimeoutError:
                pass

    async def list_cameras(self, _: dict) -> dict:
        return {"cameras": [self._safe_metadata(camera) for camera in self._cameras.values()]}

    async def capture_frame(self, arguments: dict) -> ToolOutput:
        camera_id = arguments["camera_id"]
        camera = self._cameras.get(camera_id)
        if camera is None:
            return ToolOutput({
                "status": "error", "camera_id": camera_id,
                "description": "No configured camera has that ID.",
            })
        started = time.monotonic()
        process = None

        async def launch_and_read() -> tuple[bytes, int, bytes]:
            nonlocal process
            process = await self._process_factory(
                self.ffmpeg_executable, "-hide_banner", "-loglevel", "error",
                "-rtsp_transport", self.rtsp_transport, "-i", camera.rtsp_url,
                "-frames:v", "1", "-vf",
                f"scale={self.max_width}:{self.max_height}:force_original_aspect_ratio=decrease",
                "-q:v", "4", "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            return await self._read_frame(process)

        try:
            data, return_code, diagnostics = await asyncio.wait_for(launch_and_read(), self.timeout_seconds)
        except TimeoutError:
            if process is not None:
                await self._stop(process)
            return self._outcome(camera, "timeout", started, "No frame was obtained before the capture timeout.")
        except asyncio.CancelledError:
            if process is not None:
                await asyncio.shield(self._stop(process))
            raise
        except Exception:
            if process is None:
                return self._outcome(camera, "error", started, "Frame capture is not available on this host.")
            await self._stop(process)
            return self._outcome(camera, "error", started, "Frame capture failed locally.")

        if len(data) > self.max_bytes:
            return self._outcome(camera, "error", started, "The captured frame exceeded the configured size limit.")
        if return_code != 0:
            status = "error" if self._is_local_ffmpeg_failure(diagnostics) else "unavailable"
            description = "Frame capture failed locally." if status == "error" else "No usable frame was available."
            return self._outcome(camera, status, started, description)
        if not self._is_jpeg(data):
            return self._outcome(camera, "unavailable", started, "No usable frame was available.")
        output = self._outcome(camera, "captured", started, "A current frame was captured.")
        return ToolOutput(output.output, (ImageAttachment(data),))

    async def _read_frame(self, process: asyncio.subprocess.Process) -> tuple[bytes, int, bytes]:
        data = bytearray()
        assert process.stdout is not None
        stderr = getattr(process, "stderr", None)
        diagnostics_task = asyncio.create_task(stderr.read(64 * 1024)) if stderr is not None else None
        try:
            while chunk := await process.stdout.read(64 * 1024):
                data.extend(chunk)
                if len(data) > self.max_bytes:
                    await self._stop(process)
                    diagnostics = await diagnostics_task if diagnostics_task is not None else b""
                    return bytes(data), process.returncode if process.returncode is not None else -1, diagnostics
            return bytes(data), await process.wait(), await diagnostics_task if diagnostics_task is not None else b""
        finally:
            if diagnostics_task is not None and not diagnostics_task.done():
                diagnostics_task.cancel()
                with suppress(asyncio.CancelledError):
                    await diagnostics_task

    @staticmethod
    def _is_local_ffmpeg_failure(diagnostics: bytes) -> bool:
        text = diagnostics.decode("utf-8", errors="replace").lower()
        return any(marker in text for marker in (
            "unknown encoder", "no such filter", "error opening output",
            "invalid argument", "unrecognized option", "option not found",
        ))

    @staticmethod
    async def _stop(process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        process.kill()
        try:
            await asyncio.wait_for(process.wait(), 2)
        except TimeoutError:
            pass

    @staticmethod
    def _is_jpeg(data: bytes) -> bool:
        return len(data) >= 4 and data.startswith(b"\xff\xd8") and data.endswith(b"\xff\xd9")

    @staticmethod
    def _safe_metadata(camera: CameraConfig) -> dict:
        metadata = {"id": camera.id, "name": camera.name}
        if camera.description:
            metadata["description"] = camera.description
        return metadata

    @staticmethod
    def _outcome(camera: CameraConfig, status: str, started: float, description: str) -> ToolOutput:
        return ToolOutput({
            "status": status, "camera_id": camera.id, "camera_name": camera.name,
            "description": description,
            "duration_seconds": round(time.monotonic() - started, 3),
        })
