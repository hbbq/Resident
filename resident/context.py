from __future__ import annotations

import json
from typing import Sequence

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
        }
        return json.dumps(document, ensure_ascii=False, indent=2)
