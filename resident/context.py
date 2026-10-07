from __future__ import annotations

import json
from typing import Sequence

from .capabilities import Capability
from .domain import Identity, WakeEvent
from .provider import RESIDENT_INSTRUCTIONS
from .store import Store, utc_now


class ContextBuilder:
    def __init__(self, store: Store, *, role: str = ''):
        self.store, self.role = store, role

    def instructions(self, resident: Identity, owner: Identity,
                     capabilities: Sequence[Capability]) -> str:
        current = {
            'resident': {'stable_id': resident.id, 'address_name': resident.address_name,
                         'personality': resident.personality, 'role': self.role},
            'owner': {'stable_id': owner.id, 'address_name': owner.address_name},
            'standing_owner_guidance': self.store.active_owner_guidance(),
            'available_connectors': sorted({c.connector_id for c in capabilities}),
        }
        return RESIDENT_INSTRUCTIONS + '\nCurrent authoritative configuration:\n' + json.dumps(
            current, ensure_ascii=False, sort_keys=True)

    def build(self, event: WakeEvent) -> str:
        return json.dumps({
            'current_time': utc_now(),
            'wake_event': {'id': event.id, 'source': event.source, 'reason': event.reason,
                           'occurred_at': event.occurred_at, 'payload': event.payload},
            'pending_intentions': self.store.pending_intentions(),
        }, ensure_ascii=False)
