from __future__ import annotations

import json
from typing import Sequence

from .capabilities import Capability
from .domain import Identity, WakeEvent
from .store import Store, utc_now


class ContextBuilder:
    def __init__(self, store: Store, *, memory_limit: int = 8, message_limit: int = 8,
                 owner_guidance_limit: int = 4, role: str = ""):
        self.store = store
        self.memory_limit, self.message_limit = memory_limit, message_limit
        self.owner_guidance_limit = owner_guidance_limit
        self.role = role
        self.memory_enabled = memory_limit > 0

    def build(self, resident: Identity, owner: Identity, event: WakeEvent,
              capabilities: Sequence[Capability]) -> str:
        # Put human/source context before metadata so the deliberately small keyword
        # selector does not spend its term budget on ids and field names.
        primary = event.payload.get("content") or event.payload.get("context") or ""
        query = f"{primary} {event.reason}"
        owner_guidance = (self.store.recall_standing_owner_guidance(self.owner_guidance_limit)
                          if self.memory_enabled else [])
        guidance_ids = {memory["id"] for memory in owner_guidance}
        retrieved_memories = [
            memory for memory in (self.store.recall(query, self.memory_limit)
                                  if self.memory_enabled else [])
            if memory["id"] not in guidance_ids
        ]
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
            "owner_guidance": owner_guidance,
            "retrieved_memories": retrieved_memories,
            "recent_communication": self.store.recent_messages(self.message_limit),
        }
        return json.dumps(document, ensure_ascii=False, indent=2)
