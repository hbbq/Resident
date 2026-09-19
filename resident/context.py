from __future__ import annotations

import json
from typing import Any, Sequence

from .capabilities import Capability
from .domain import Identity, WakeEvent
from .store import Store, utc_now


class ContextBuilder:
    def __init__(self, store: Store, *, message_limit: int = 8, role: str = ""):
        self.store = store
        self.message_limit = message_limit
        self.role = role

    def build(self, resident: Identity, owner: Identity, event: WakeEvent,
              capabilities: Sequence[Capability]) -> str:
        """Build the full continuity document used by stateless providers."""
        document = {
            "resident": {"stable_id": resident.id, "address_name": resident.address_name,
                         "personality": resident.personality, "role": self.role},
            "owner": {"stable_id": owner.id, "address_name": owner.address_name},
            "current_time": utc_now(),
            "wake_event": {"id": event.id, "source": event.source, "reason": event.reason,
                           "occurred_at": event.occurred_at, "payload": event.payload},
            "available_connectors": [{
                "id": c.connector_id, "description": c.connector_description, "status": "available",
                "capability": {"name": c.name, "description": c.description,
                               "input_schema": c.input_schema},
            } for c in capabilities],
            "pending_intentions": self.store.pending_intentions(),
            "recent_communication": self.store.recent_messages(self.message_limit),
            "standing_owner_guidance": self.store.active_owner_guidance(),
        }
        return json.dumps(document, ensure_ascii=False, indent=2)

    @staticmethod
    def _wake_event(event: WakeEvent) -> dict[str, Any]:
        return {"id": event.id, "source": event.source, "reason": event.reason,
                "occurred_at": event.occurred_at, "payload": event.payload}

    @staticmethod
    def _capabilities(capabilities: Sequence[Capability]) -> list[dict[str, Any]]:
        return [{
            "id": capability.connector_id,
            "description": capability.connector_description,
            "status": "available",
            "capability": {
                "name": capability.name,
                "description": capability.description,
                "input_schema": capability.input_schema,
            },
        } for capability in capabilities]

    def authoritative_state(self, resident: Identity, owner: Identity,
                            capabilities: Sequence[Capability]) -> dict[str, Any]:
        """Return locally authoritative state that can change outside a session."""
        return {
            "resident": {
                "stable_id": resident.id,
                "address_name": resident.address_name,
                "personality": resident.personality,
                "role": self.role,
            },
            "owner": {"stable_id": owner.id, "address_name": owner.address_name},
            "available_connectors": self._capabilities(capabilities),
            "standing_owner_guidance": self.store.active_owner_guidance(),
        }

    def build_managed_wake(self, event: WakeEvent, *,
                           authoritative_update: dict[str, Any] | None = None) -> str:
        """Build new input for an existing long-lived managed session."""
        document: dict[str, Any] = {"wake_event": self._wake_event(event)}
        if authoritative_update:
            document["authoritative_state_update"] = authoritative_update
        return json.dumps(document, ensure_ascii=False, indent=2)

    def build_managed_bootstrap(self, resident: Identity, owner: Identity,
                                event: WakeEvent, capabilities: Sequence[Capability], *,
                                handover: str | None) -> str:
        """Build the one-time continuity input for a new managed session."""
        bootstrap = {
            **self.authoritative_state(resident, owner, capabilities),
            "current_time": utc_now(),
            "pending_intentions": self.store.pending_intentions(),
            "durable_memory_awareness": self.store.memory_awareness(limit=8),
            "handover": handover,
            "note": "Long-term memory is selectively available through memory tools.",
        }
        return json.dumps({
            "wake_event": self._wake_event(event),
            "new_session_bootstrap": bootstrap,
        }, ensure_ascii=False, indent=2)
