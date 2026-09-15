from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable, Sequence

from .capabilities import Capability
from .config import CameraConfig
from .domain import ImageAttachment, ToolOutput


ProcessFactory = Callable[..., Awaitable[asyncio.subprocess.Process]]


class CameraConnector:
    """Read-only, on-demand capture for configured RTSP cameras."""

    def __init__(self, cameras: Sequence[CameraConfig], *, timeout_seconds: float = 8.0,
                 max_width: int = 1280, max_height: int = 720, max_bytes: int = 2_000_000,
                 rtsp_transport: str = "tcp", ffmpeg_executable: str = "ffmpeg",
                 process_factory: ProcessFactory = asyncio.create_subprocess_exec):
        self._cameras = {camera.id: camera for camera in cameras}
        self.timeout_seconds = timeout_seconds
        self.max_width, self.max_height, self.max_bytes = max_width, max_height, max_bytes
        self.rtsp_transport, self.ffmpeg_executable = rtsp_transport, ffmpeg_executable
        self._process_factory = process_factory
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
        try:
            process = await self._process_factory(
                self.ffmpeg_executable, "-hide_banner", "-loglevel", "error",
                "-rtsp_transport", self.rtsp_transport, "-i", camera.rtsp_url,
                "-frames:v", "1", "-vf",
                f"scale={self.max_width}:{self.max_height}:force_original_aspect_ratio=decrease",
                "-q:v", "4", "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            )
        except Exception:
            return self._outcome(camera, "error", started, "Frame capture is not available on this host.")

        try:
            data, return_code = await asyncio.wait_for(self._read_frame(process), self.timeout_seconds)
        except TimeoutError:
            await self._stop(process)
            return self._outcome(camera, "timeout", started, "No frame was obtained before the capture timeout.")
        except Exception:
            await self._stop(process)
            return self._outcome(camera, "error", started, "Frame capture failed locally.")

        if len(data) > self.max_bytes:
            return self._outcome(camera, "error", started, "The captured frame exceeded the configured size limit.")
        if return_code != 0 or not self._is_jpeg(data):
            return self._outcome(camera, "unavailable", started, "No usable frame was available.")
        output = self._outcome(camera, "captured", started, "A current frame was captured.")
        return ToolOutput(output.output, (ImageAttachment(data),))

    async def _read_frame(self, process: asyncio.subprocess.Process) -> tuple[bytes, int]:
        data = bytearray()
        assert process.stdout is not None
        while chunk := await process.stdout.read(64 * 1024):
            data.extend(chunk)
            if len(data) > self.max_bytes:
                await self._stop(process)
                return bytes(data), process.returncode if process.returncode is not None else -1
        return bytes(data), await process.wait()

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
