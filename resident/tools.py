from __future__ import annotations

import inspect
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Awaitable, Callable, Sequence

from .capabilities import Capability
from .domain import ToolOutput, ToolSpec
from .store import Store


CORE_TOOL_NAMES = frozenset({
    "remember", "recall", "update_memory", "forget", "create_intention",
    "update_intention", "send_owner_message", "schedule_wakeup",
    "search_communication", "list_wake_history",
})

MEMORY_KINDS = ["fact", "preference", "rule", "hypothesis", "experience", "unknown"]
MEMORY_LEVELS = ["low", "medium", "high"]
MEMORY_PROVENANCES = ["owner", "resident", "connector", "unknown"]


@dataclass(frozen=True)
class Tool:
    spec: ToolSpec
    handler: Callable[[dict[str, Any]], Awaitable[dict[str, Any] | ToolOutput] | dict[str, Any] | ToolOutput]


def _schema(required: Sequence[str] = (), **properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}


class ToolRegistry:
    def __init__(self, store: Store, capabilities: Sequence[Capability],
                 send_message: Callable[[str], Awaitable[dict[str, Any]] | dict[str, Any]],
                 emit: Callable[[str, dict[str, Any]], None], *,
                 current_run_id: str | None = None, memory_enabled: bool = True,
                 owner_communication_enabled: bool = True):
        self.store, self.emit, self.current_run_id = store, emit, current_run_id
        self.tools: dict[str, Tool] = {
            "remember": Tool(ToolSpec("remember",
                "Persist a selective autobiographical memory with your own descriptive assessment. "
                "Provenance describes the underlying information source; it does not enact policy. "
                "Set standing true only for an Owner-derived preference or rule that should be supplied "
                "as durable guidance across future wakes, even without textual relevance.",
                _schema(("content", "kind", "importance", "confidence", "provenance", "standing"),
                        content={"type": "string"}, kind={"type": "string", "enum": MEMORY_KINDS},
                        importance={"type": "string", "enum": MEMORY_LEVELS},
                        confidence={"type": "string", "enum": MEMORY_LEVELS},
                        provenance={"type": "string", "enum": MEMORY_PROVENANCES},
                        standing={"type": "boolean"})), self._remember),
            "recall": Tool(ToolSpec("recall", "Search persistent memories by words, ranked by relevance and semantic metadata, or list the most durable memories.",
                _schema(("query",), query={"type": "string"}, limit={"type": "integer", "minimum": 1, "maximum": 20})), self._recall),
            "update_memory": Tool(ToolSpec("update_memory",
                "Refine an existing memory's content or assessment instead of accumulating avoidable "
                "contradictions, especially when the Owner changes standing guidance.",
                _schema(("id",), id={"type": "string"}, content={"type": "string"},
                        kind={"type": "string", "enum": MEMORY_KINDS},
                        importance={"type": "string", "enum": MEMORY_LEVELS},
                        confidence={"type": "string", "enum": MEMORY_LEVELS},
                        provenance={"type": "string", "enum": MEMORY_PROVENANCES},
                        standing={"type": "boolean"}) |
                {"minProperties": 2}), self._update_memory),
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
            "search_communication": Tool(ToolSpec("search_communication",
                "Search or page through persisted Owner communication, newest first. This is read-only.",
                _schema(query={"type": "string"},
                        direction={"type": ["string", "null"], "enum": ["inbound", "outbound", None]},
                        from_time={"type": ["string", "null"]}, to_time={"type": ["string", "null"]},
                        limit={"type": "integer", "minimum": 1, "maximum": 50},
                        offset={"type": "integer", "minimum": 0, "maximum": 10000})),
                self._search_communication),
            "list_wake_history": Tool(ToolSpec("list_wake_history",
                "Search or page through prior wake runs and safe observable event summaries, newest first. "
                "Raw payloads, tool arguments and results, model content, and attachments are excluded. This is read-only.",
                _schema(query={"type": "string"}, source={"type": ["string", "null"]},
                        status={"type": ["string", "null"],
                                "enum": ["running", "completed", "failed", None]},
                        from_time={"type": ["string", "null"]}, to_time={"type": ["string", "null"]},
                        limit={"type": "integer", "minimum": 1, "maximum": 20},
                        offset={"type": "integer", "minimum": 0, "maximum": 1000})),
                self._list_wake_history),
        }
        if not memory_enabled:
            for name in ("remember", "recall", "update_memory", "forget"):
                self.tools.pop(name)
        if not owner_communication_enabled:
            self.tools.pop("send_owner_message")
        for capability in capabilities:
            if capability.name in self.tools:
                raise ValueError(f"Duplicate tool name: {capability.name}")
            self.tools[capability.name] = Tool(capability.spec, capability.handler)

    @property
    def specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self.tools.values()]

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolOutput:
        tool = self.tools.get(name)
        if tool is None:
            return ToolOutput({"ok": False, "error": f"Unknown or unavailable tool: {name}"})
        error = self._validate(tool.spec.input_schema, arguments)
        if error:
            return ToolOutput({"ok": False, "error": error})
        try:
            result = tool.handler(arguments)
            if inspect.isawaitable(result):
                result = await result
            if isinstance(result, ToolOutput):
                return ToolOutput({"ok": True, **result.output}, result.attachments)
            return ToolOutput({"ok": True, **result})
        except Exception as exc:
            return ToolOutput({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

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
        if len(arguments) < schema.get("minProperties", 0):
            return f"At least {schema['minProperties']} arguments are required"
        for key, value in arguments.items():
            allowed = properties[key].get("type")
            allowed = [allowed] if isinstance(allowed, str) else allowed
            matches = (value is None and "null" in allowed) or ("string" in allowed and isinstance(value, str)) or \
                ("integer" in allowed and isinstance(value, int) and not isinstance(value, bool)) or \
                ("boolean" in allowed and isinstance(value, bool)) or \
                ("object" in allowed and isinstance(value, dict))
            if not matches:
                return f"Argument {key!r} has the wrong type"
            if isinstance(value, int) and (value < properties[key].get("minimum", value) or value > properties[key].get("maximum", value)):
                return f"Argument {key!r} is outside the allowed range"
            if "enum" in properties[key] and value not in properties[key]["enum"]:
                return f"Argument {key!r} is not an allowed value"
        return None

    def _remember(self, a: dict[str, Any]) -> dict[str, Any]:
        item_id = self.store.remember(
            a["content"], "resident", kind=a["kind"], importance=a["importance"],
            confidence=a["confidence"], provenance=a["provenance"], standing=a["standing"])
        self.emit("memory.created", {"memory_id": item_id})
        return {"memory_id": item_id, "memory": self.store.memory(item_id)}

    def _recall(self, a: dict[str, Any]) -> dict[str, Any]:
        return {"memories": self.store.recall(a["query"], a.get("limit", 10))}

    def _update_memory(self, a: dict[str, Any]) -> dict[str, Any]:
        fields = {name: a[name] for name in (
            "content", "kind", "importance", "confidence", "provenance", "standing") if name in a}
        updated = self.store.update_memory(a["id"], **fields)
        if updated: self.emit("memory.updated", {"memory_id": a["id"]})
        return {"updated": updated, "memory": self.store.memory(a["id"]) if updated else None}

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

    @staticmethod
    def _validated_time(value: str | None) -> str | None:
        if value is None:
            return None
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("History times must include a UTC offset")
        return parsed.astimezone(UTC).isoformat()

    def _search_communication(self, a: dict[str, Any]) -> dict[str, Any]:
        messages = self.store.search_messages(
            query=a.get("query", ""), direction=a.get("direction"),
            from_time=self._validated_time(a.get("from_time")),
            to_time=self._validated_time(a.get("to_time")),
            limit=a.get("limit", 20), offset=a.get("offset", 0),
        )
        return {"messages": messages}

    def _list_wake_history(self, a: dict[str, Any]) -> dict[str, Any]:
        runs = self.store.wake_history(
            query=a.get("query", ""), source=a.get("source"), status=a.get("status"),
            from_time=self._validated_time(a.get("from_time")),
            to_time=self._validated_time(a.get("to_time")),
            limit=a.get("limit", 10), offset=a.get("offset", 0),
            exclude_run_id=self.current_run_id,
        )
        return {"wake_runs": runs}

