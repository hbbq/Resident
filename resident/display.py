from __future__ import annotations

import asyncio
import json
from typing import Any, Sequence
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

from .capabilities import Capability
from .config import DisplayConfig
from .observability import to_thread_timed
from .outputs import OutputCapability


class DisplayConnector:
    """Action-only text output for configured HomeOps display queues."""

    def __init__(self, base_url: str, displays: Sequence[DisplayConfig], *,
                 request_timeout_seconds: float = 10.0):
        parsed = urlsplit(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("HomeOps base URL must be an absolute HTTP(S) URL")
        self.base_url = base_url.rstrip("/")
        self.request_timeout_seconds = max(0.1, request_timeout_seconds)
        display_ids = [display.id for display in displays]
        if len(set(display_ids)) != len(display_ids):
            raise ValueError("Duplicate display id")
        self.capabilities = [self._capability(display_id) for display_id in display_ids]
        self.output_capabilities = [self._output_capability(display) for display in displays]

    def _output_capability(self, display: DisplayConfig) -> OutputCapability:
        async def deliver(payload: dict[str, Any]) -> dict[str, Any]:
            return await self.show_text(display.id, payload["content"])

        content_schema: dict[str, Any] = {"type": "string", "minLength": 1}
        if display.max_length is not None:
            content_schema["maxLength"] = display.max_length
        return OutputCapability(
            output_type="display", target=display.id,
            description=f"Enqueue plain text for the configured {display.id} display.",
            payload_schema={
                "type": "object", "properties": {"content": content_schema},
                "required": ["content"], "additionalProperties": False,
            },
            route_identity=f"homeops-display:{display.id}", handler=deliver,
            legacy_tool_name=f"{display.id}_show_text",
        )

    def _capability(self, display_id: str) -> Capability:
        async def show_text(arguments: dict[str, Any]) -> dict[str, Any]:
            return await self.show_text(display_id, arguments["text"])

        return Capability(
            connector_id="display",
            connector_description="Configured text displays backed by HomeOps queues",
            name=f"{display_id}_show_text",
            description=f"Enqueue plain text for the configured {display_id} display.",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
            handler=show_text,
        )

    def _post_text(self, display_id: str, text: str) -> None:
        path_id = quote(display_id, safe="")
        request = Request(
            f"{self.base_url}/api/displays/{path_id}/messages",
            data=json.dumps({"text": text}).encode("utf-8"),
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.request_timeout_seconds) as response:
            status = response.status
        if status != 204:
            raise RuntimeError(
                f"HomeOps display {display_id!r} returned HTTP status {status}; expected 204")

    async def show_text(self, display_id: str, text: str) -> dict[str, Any]:
        await to_thread_timed(
            "homeops.display_request", self._post_text, display_id, text,
            display_id=display_id, request_timeout_seconds=self.request_timeout_seconds)
        return {"display_id": display_id, "status": "queued"}
