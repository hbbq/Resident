from __future__ import annotations

import inspect
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Awaitable, Callable, Sequence

from .capabilities import Capability
from .domain import ToolSpec
from .store import Store


@dataclass(frozen=True)
class Tool:
    spec: ToolSpec
    handler: Callable[[dict[str, Any]], Awaitable[dict[str, Any]] | dict[str, Any]]


def _schema(required: Sequence[str] = (), **properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}


class ToolRegistry:
    def __init__(self, store: Store, capabilities: Sequence[Capability], send_message: Callable[[str], dict[str, Any]],
                 emit: Callable[[str, dict[str, Any]], None]):
        self.store, self.emit = store, emit
        self.tools: dict[str, Tool] = {
            "remember": Tool(ToolSpec("remember", "Persist something for your future self.",
                _schema(("content",), content={"type": "string"})), self._remember),
            "recall": Tool(ToolSpec("recall", "Search your persistent memories by words or list recent memories.",
                _schema(("query",), query={"type": "string"}, limit={"type": "integer", "minimum": 1, "maximum": 20})), self._recall),
            "update_memory": Tool(ToolSpec("update_memory", "Replace an existing memory's content.",
                _schema(("id", "content"), id={"type": "string"}, content={"type": "string"})), self._update_memory),
            "forget": Tool(ToolSpec("forget", "Delete a memory by id.",
                _schema(("id",), id={"type": "string"})), self._forget),
            "create_intention": Tool(ToolSpec("create_intention", "Persist a small pending intention for a future wake.",
                _schema(("content",), content={"type": "string"})), self._create_intention),
            "update_intention": Tool(ToolSpec("update_intention", "Change an intention's content or status.",
                _schema(("id",), id={"type": "string"}, content={"type": ["string", "null"]},
                        status={"type": ["string", "null"], "enum": ["pending", "completed", "cancelled", None]})), self._update_intention),
            "send_owner_message": Tool(ToolSpec("send_owner_message",
                "Send intentional communication to your owner. This is the only Owner-facing output path, "
                "including for replies to Owner-initiated wakes.",
                _schema(("content",), content={"type": "string"})), lambda a: send_message(a["content"])),
            "schedule_wakeup": Tool(ToolSpec("schedule_wakeup", "Request a persistent future wakeup after a delay.",
                _schema(("delay_seconds", "reason"), delay_seconds={"type": "integer", "minimum": 1, "maximum": 31536000},
                        reason={"type": "string"}, context={"type": "object"})), self._schedule),
        }
        for capability in capabilities:
            if capability.name in self.tools:
                raise ValueError(f"Duplicate tool name: {capability.name}")
            self.tools[capability.name] = Tool(capability.spec, capability.handler)

    @property
    def specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self.tools.values()]

    async def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        tool = self.tools.get(name)
        if tool is None:
            return {"ok": False, "error": f"Unknown or unavailable tool: {name}"}
        error = self._validate(tool.spec.input_schema, arguments)
        if error:
            return {"ok": False, "error": error}
        try:
            result = tool.handler(arguments)
            if inspect.isawaitable(result):
                result = await result
            return {"ok": True, **result}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    @staticmethod
    def _validate(schema: dict[str, Any], arguments: Any) -> str | None:
        if not isinstance(arguments, dict):
            return "Arguments must be an object"
        properties = schema.get("properties", {})
        unknown = set(arguments) - set(properties)
        missing = set(schema.get("required", [])) - set(arguments)
        if unknown:
            return f"Unknown arguments: {', '.join(sorted(unknown))}"
        if missing:
            return f"Missing arguments: {', '.join(sorted(missing))}"
        for key, value in arguments.items():
            allowed = properties[key].get("type")
            allowed = [allowed] if isinstance(allowed, str) else allowed
            matches = (value is None and "null" in allowed) or ("string" in allowed and isinstance(value, str)) or \
                ("integer" in allowed and isinstance(value, int) and not isinstance(value, bool)) or \
                ("object" in allowed and isinstance(value, dict))
            if not matches:
                return f"Argument {key!r} has the wrong type"
            if isinstance(value, int) and (value < properties[key].get("minimum", value) or value > properties[key].get("maximum", value)):
                return f"Argument {key!r} is outside the allowed range"
            if "enum" in properties[key] and value not in properties[key]["enum"]:
                return f"Argument {key!r} is not an allowed value"
        return None

    def _remember(self, a: dict[str, Any]) -> dict[str, Any]:
        item_id = self.store.remember(a["content"], "resident")
        self.emit("memory.created", {"memory_id": item_id})
        return {"memory_id": item_id}

    def _recall(self, a: dict[str, Any]) -> dict[str, Any]:
        return {"memories": self.store.recall(a["query"], a.get("limit", 10))}

    def _update_memory(self, a: dict[str, Any]) -> dict[str, Any]:
        updated = self.store.update_memory(a["id"], a["content"])
        if updated: self.emit("memory.updated", {"memory_id": a["id"]})
        return {"updated": updated}

    def _forget(self, a: dict[str, Any]) -> dict[str, Any]:
        forgotten = self.store.forget(a["id"])
        if forgotten: self.emit("memory.forgotten", {"memory_id": a["id"]})
        return {"forgotten": forgotten}

    def _create_intention(self, a: dict[str, Any]) -> dict[str, Any]:
        item_id = self.store.create_intention(a["content"])
        self.emit("intention.created", {"intention_id": item_id})
        return {"intention_id": item_id}

    def _update_intention(self, a: dict[str, Any]) -> dict[str, Any]:
        updated = self.store.update_intention(a["id"], content=a.get("content"), status=a.get("status"))
        if updated: self.emit("intention.updated", {"intention_id": a["id"], "status": a.get("status")})
        return {"updated": updated}

    def _schedule(self, a: dict[str, Any]) -> dict[str, Any]:
        due = datetime.now(UTC) + timedelta(seconds=a["delay_seconds"])
        schedule_id = self.store.schedule(due.isoformat(), a["reason"], a.get("context", {}))
        self.emit("wakeup.scheduled", {"schedule_id": schedule_id, "due_at": due.isoformat(), "reason": a["reason"]})
        return {"schedule_id": schedule_id, "due_at": due.isoformat()}

