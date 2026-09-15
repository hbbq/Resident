from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable

from .domain import ToolSpec


Handler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class Capability:
    connector_id: str
    connector_description: str
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Handler

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.input_schema)


async def current_time(_: dict[str, Any]) -> dict[str, Any]:
    return {"current_time": datetime.now(UTC).isoformat(), "timezone": "UTC"}


def diagnostic_capabilities() -> list[Capability]:
    return [Capability(
        connector_id="diagnostics", connector_description="Local, read-only runtime diagnostics",
        name="diagnostics_current_time", description="Read the runtime's current UTC time.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler=current_time,
    )]

