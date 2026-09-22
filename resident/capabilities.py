from __future__ import annotations

import json
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable

from .domain import ToolOutput, ToolSpec


Handler = Callable[[dict[str, Any]], Awaitable[dict[str, Any] | ToolOutput]]
_INVOCATION_ID: ContextVar[str | None] = ContextVar("capability_invocation_id", default=None)


def current_invocation_id() -> str | None:
    """Return the durable provider call id while a capability handler is running."""
    return _INVOCATION_ID.get()


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

    def public_descriptor(self) -> dict[str, Any]:
        """Return the stable, non-executable capability data visible to Resident."""
        descriptor = {
            "connector_id": self.connector_id,
            "connector_description": self.connector_description,
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
        # Round-tripping also rejects non-serializable schemas and prevents later
        # mutation of a schema from changing an already-taken snapshot.
        return json.loads(json.dumps(descriptor, sort_keys=True, separators=(",", ":")))


async def current_time(_: dict[str, Any]) -> dict[str, Any]:
    return {"current_time": datetime.now(UTC).isoformat(), "timezone": "UTC"}


def diagnostic_capabilities() -> list[Capability]:
    return [Capability(
        connector_id="diagnostics", connector_description="Local, read-only runtime diagnostics",
        name="diagnostics_current_time", description="Read the runtime's current UTC time.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler=current_time,
    )]

