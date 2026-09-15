from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Callable
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen

from .capabilities import Capability
from .domain import WakeEvent
from .store import utc_now


class HomeOpsConnector:
    """A small, read-only adapter for the HomeOps measurements API."""

    def __init__(self, base_url: str, *, poll_seconds: float = 30.0,
                 request_timeout_seconds: float = 10.0,
                 diagnostic_output: Callable[[str], None] | None = None):
        parsed = urlsplit(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("HomeOps base URL must be an absolute HTTP(S) URL")
        self.base_url = base_url.rstrip("/")
        self.poll_seconds = max(0.1, poll_seconds)
        self.request_timeout_seconds = max(0.1, request_timeout_seconds)
        self.diagnostic_output = diagnostic_output or (lambda _: None)
        self._snapshot: dict[str, dict[str, Any]] | None = None

    @property
    def capabilities(self) -> list[Capability]:
        description = "Read-only access to current and historical HomeOps measurements"
        return [
            Capability(
                connector_id="homeops", connector_description=description,
                name="homeops_get_current_measurements",
                description="Get the latest value, timestamp, point metadata, and device metadata for all HomeOps measurements.",
                input_schema={"type": "object", "properties": {}, "additionalProperties": False},
                handler=self.get_current_measurements,
            ),
            Capability(
                connector_id="homeops", connector_description=description,
                name="homeops_get_measurement_history",
                description="Get newest-first HomeOps history for one measurement point, optionally bounded by time and count.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "point_id": {"type": "string"},
                        "from_time": {"type": "string"},
                        "to_time": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 5000},
                    },
                    "required": ["point_id"],
                    "additionalProperties": False,
                },
                handler=self.get_measurement_history,
            ),
        ]

    def _get_json(self, path: str, query: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        if query:
            url = f"{url}?{urlencode(query)}"
        request = Request(url, headers={"Accept": "application/json"}, method="GET")
        with urlopen(request, timeout=self.request_timeout_seconds) as response:
            return json.load(response)

    async def _request(self, path: str, query: dict[str, Any] | None = None) -> Any:
        return await asyncio.to_thread(self._get_json, path, query)

    @staticmethod
    def _measurement_list(payload: Any) -> list[dict[str, Any]]:
        measurements = payload.get("measurements", payload.get("items")) if isinstance(payload, dict) else payload
        if not isinstance(measurements, list) or not all(isinstance(item, dict) for item in measurements):
            raise ValueError("HomeOps response must be a measurement array")
        return measurements

    async def get_current_measurements(self, _: dict[str, Any]) -> dict[str, Any]:
        measurements = self._measurement_list(await self._request("/api/measurements/latest"))
        return {"measurements": measurements}

    async def get_measurement_history(self, arguments: dict[str, Any]) -> dict[str, Any]:
        point_id = arguments["point_id"]
        query = {
            api_name: arguments[argument_name]
            for argument_name, api_name in (("from_time", "from"), ("to_time", "to"), ("limit", "limit"))
            if argument_name in arguments
        }
        path = f"/api/measurement-points/{quote(point_id, safe='')}/history"
        measurements = self._measurement_list(await self._request(path, query))
        return {"measurements": measurements}

    @staticmethod
    def _index(measurements: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        indexed: dict[str, dict[str, Any]] = {}
        for measurement in measurements:
            point_id = measurement.get("pointId")
            if point_id is None:
                raise ValueError("HomeOps latest measurement is missing pointId")
            indexed[str(point_id)] = measurement
        return indexed

    @staticmethod
    def _change(previous: dict[str, Any] | None, current: dict[str, Any]) -> dict[str, Any]:
        return {
            "point_id": current["pointId"],
            "point_key": current.get("pointKey"),
            "point_name": current.get("pointName"),
            "kind": current.get("kind"),
            "unit": current.get("unit"),
            "device_id": current.get("deviceId"),
            "device_name": current.get("deviceName"),
            "old_value": previous.get("value") if previous is not None else None,
            "new_value": current.get("value"),
            "old_timestamp": previous.get("timestamp") if previous is not None else None,
            "new_timestamp": current.get("timestamp"),
        }

    async def poll_once(self, queue: asyncio.Queue[WakeEvent]) -> None:
        measurements = self._measurement_list(await self._request("/api/measurements/latest"))
        current = self._index(measurements)
        previous = self._snapshot
        self._snapshot = current
        if previous is None:
            return
        changes = [
            self._change(previous.get(point_id), measurement)
            for point_id, measurement in current.items()
            if point_id not in previous or previous[point_id].get("value") != measurement.get("value")
        ]
        if changes:
            await queue.put(WakeEvent(
                id=str(uuid.uuid4()), source="homeops", reason="measurement_changed",
                occurred_at=utc_now(), payload={"change_count": len(changes), "changes": changes},
            ))

    async def run(self, queue: asyncio.Queue[WakeEvent], stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await self.poll_once(queue)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.diagnostic_output(f"poll failed: {type(exc).__name__}: {exc}")
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.poll_seconds)
            except TimeoutError:
                pass
