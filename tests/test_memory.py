from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from resident.memory import MemoryCurator, SessionItemPage
from resident.store import Store


class FakeSource:
    session_id = "session-1"

    def __init__(self):
        self.cursors = []

    async def session_items(self, cursor, limit):
        self.cursors.append(cursor)
        if cursor is not None:
            return SessionItemPage((), cursor, False)
        return SessionItemPage(({
            "id": "item-1", "type": "message", "role": "user",
            "created_at": "2026-01-01T00:00:00+00:00",
            "content": [{"type": "input_text", "text": "I prefer tea. token=secret"}],
        },), "item-1", False)


class FakeModel:
    def __init__(self):
        self.items = []

    async def curate(self, session_id, items, existing):
        self.items.extend(items)
        return {"mutations": [{
            "operation": "create", "kind": "preference", "content": "Owner prefers tea",
            "confidence": .9, "provenance": [{"item_id": "item-1", "source_type": "message",
                                                "excerpt": "token=secret; prefers tea"}],
        }], "handover": "Continue discussing tea."}


class MemoryStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_curator_checkpoint_and_idempotent_memory_with_redacted_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            source, model = FakeSource(), FakeModel()
            curator = MemoryCurator(store, source, model)

            handover = await curator.catch_up()
            await curator.catch_up()

            memories = store.search_memories("tea")
            self.assertEqual(1, len(memories))
            self.assertEqual("Continue discussing tea.", handover)
            self.assertEqual("item-1", store.curator_checkpoint(
                "openai_agents", "session-1")["cursor"])
            detail = store.memory(memories[0]["id"])
            self.assertIn("[REDACTED]", detail["provenance"][0]["excerpt"])
            self.assertNotIn("secret", str(model.items))
            store.close()

    async def test_memory_updates_invalidate_without_destroying_revision_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            store.apply_curator_batch("openai_agents", "s", "one", "one", "batch-1", [{
                "memory_id": "m", "operation": "create", "kind": "fact", "content": "old"}])
            store.apply_curator_batch("openai_agents", "s", "two", "two", "batch-2", [{
                "memory_id": "m", "operation": "update", "content": "new"}])
            store.apply_curator_batch("openai_agents", "s", "three", "three", "batch-3", [{
                "memory_id": "m", "operation": "invalidate", "content": "incorrect"}])

            self.assertEqual([], store.search_memories())
            self.assertEqual(3, store.connection.execute(
                "SELECT count(*) FROM memory_revisions WHERE memory_id='m'").fetchone()[0])
            self.assertEqual("invalidated", store.memory("m")["status"])
            store.close()

    async def test_owner_guidance_is_revisioned_and_removable(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            guidance_id = store.set_owner_guidance("Always ask first")
            store.set_owner_guidance("Ask before purchases", guidance_id=guidance_id)
            self.assertEqual("Ask before purchases", store.active_owner_guidance()[0]["content"])
            self.assertTrue(store.remove_owner_guidance(guidance_id))
            self.assertEqual([], store.active_owner_guidance())
            self.assertEqual(3, store.connection.execute(
                "SELECT count(*) FROM owner_guidance_revisions WHERE guidance_id=?",
                (guidance_id,)).fetchone()[0])
            store.close()


if __name__ == "__main__":
    unittest.main()
