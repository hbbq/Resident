from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from resident.context import ContextBuilder
from resident.domain import WakeEvent
from resident.store import Store, utc_now
from resident.tools import ToolRegistry


class MemoryStoreTests(unittest.TestCase):
    def test_legacy_memories_receive_conservative_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "resident.sqlite3"
            connection = sqlite3.connect(path)
            connection.executescript("""
                CREATE TABLE schema_version(version INTEGER NOT NULL);
                INSERT INTO schema_version VALUES(6);
                CREATE TABLE memories(
                  id TEXT PRIMARY KEY, content TEXT NOT NULL, source TEXT NOT NULL,
                  created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
            """)
            now = utc_now()
            connection.execute(
                "INSERT INTO memories VALUES(?,?,?,?,?)",
                ("legacy", "An old memory", "resident", now, now),
            )
            connection.commit()
            connection.close()

            store = Store(path)

            self.assertEqual(10, store.connection.execute(
                "SELECT version FROM schema_version").fetchone()[0])
            memory = store.memory("legacy")
            self.assertEqual("resident", memory["source"])
            self.assertEqual(
                ("unknown", "low", "low", "unknown", False),
                tuple(memory[name] for name in (
                    "kind", "importance", "confidence", "provenance", "standing")),
            )
            store.close()

    def test_standing_owner_guidance_is_bounded_and_not_text_gated(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            expected = store.remember(
                "Use everyday non-technical language with the Owner", "resident",
                kind="preference", importance="high", confidence="high",
                provenance="owner", standing=True,
            )
            store.remember(
                "An important but narrow preference about tea", "resident",
                kind="preference", importance="high", confidence="high",
                provenance="owner", standing=False,
            )
            store.remember(
                "Resident inference must not become Owner guidance", "resident",
                kind="rule", importance="high", confidence="high",
                provenance="resident", standing=True,
            )

            guidance = store.recall_standing_owner_guidance(limit=1)

            self.assertEqual([expected], [memory["id"] for memory in guidance])
            self.assertTrue(guidance[0]["standing"])
            store.close()

    def test_context_supplies_and_refines_standing_guidance_on_unrelated_wakes(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            resident, owner = store.provision("Resident", "Owner", "")
            guidance_id = store.remember(
                "Do not notify the Owner about small ordinary changes", "resident",
                kind="preference", importance="high", confidence="high",
                provenance="owner", standing=True,
            )
            display_guidance_id = store.remember(
                "Always summarize events briefly on display1, even when no Owner message is warranted",
                "resident", kind="rule", importance="high", confidence="high",
                provenance="owner", standing=True,
            )
            event = WakeEvent(
                "event", "camera", "capabilities_changed", utc_now(),
                {"added": ["display1"]},
            )
            builder = ContextBuilder(store)

            first = json.loads(builder.build(resident, owner, event, []))
            self.assertEqual(
                {guidance_id, display_guidance_id},
                {item["id"] for item in first["owner_guidance"]},
            )
            self.assertNotIn(guidance_id, [item["id"] for item in first["retrieved_memories"]])

            store.update_memory(
                guidance_id,
                content="Only notify the Owner about small changes when safety is involved",
            )
            second = json.loads(builder.build(resident, owner, event, []))
            self.assertEqual(
                "Only notify the Owner about small changes when safety is involved",
                next(item["content"] for item in second["owner_guidance"]
                     if item["id"] == guidance_id),
            )
            store.close()

    def test_relevant_durable_owner_preference_outranks_new_observations(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            preference_id = store.remember(
                "Routine environmental changes generally do not need reporting", "resident",
                kind="preference", importance="high", confidence="high", provenance="owner",
            )
            for number in range(12):
                store.remember(
                    f"Routine environmental observation {number}", "resident",
                    kind="experience", importance="low", confidence="medium",
                    provenance="resident",
                )

            recalled = store.recall("routine environmental activity", limit=3)

            self.assertEqual(preference_id, recalled[0]["id"])
            self.assertEqual(3, len(recalled))
            self.assertEqual("owner", recalled[0]["provenance"])
            store.close()

    def test_textual_relevance_remains_primary_and_gates_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            more_relevant = store.remember(
                "Detector transitions can be noisy transitions", "resident",
                kind="hypothesis", importance="low", confidence="low", provenance="resident",
            )
            store.remember(
                "Detector behavior is notable", "resident", kind="rule", importance="high",
                confidence="high", provenance="owner",
            )
            unrelated = store.remember(
                "Always preserve this unrelated lesson", "resident", kind="rule",
                importance="high", confidence="high", provenance="owner",
            )

            recalled = store.recall("detector transitions", limit=5)

            self.assertEqual(more_relevant, recalled[0]["id"])
            self.assertNotIn(unrelated, [memory["id"] for memory in recalled])
            self.assertEqual("hypothesis", recalled[0]["kind"])
            store.close()


class MemoryToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_tools_require_metadata_and_support_refinement(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            registry = ToolRegistry(store, [], lambda _: {}, lambda *_: None)

            rejected = await registry.execute("remember", {"content": "Unclassified"})
            self.assertFalse(rejected.output["ok"])

            created = await registry.execute("remember", {
                "content": "Transitions may be noisy", "kind": "hypothesis",
                "importance": "medium", "confidence": "low", "provenance": "resident",
                "standing": False,
            })
            memory = created.output["memory"]
            self.assertEqual("hypothesis", memory["kind"])
            self.assertEqual("resident", memory["source"])

            rejected = await registry.execute("update_memory", {"id": memory["id"]})
            self.assertFalse(rejected.output["ok"])
            self.assertEqual("At least 2 arguments are required", rejected.output["error"])

            updated = await registry.execute("update_memory", {
                "id": memory["id"], "confidence": "high",
            })
            self.assertTrue(updated.output["updated"])
            self.assertEqual("hypothesis", updated.output["memory"]["kind"])
            self.assertEqual("high", updated.output["memory"]["confidence"])
            self.assertEqual("medium", updated.output["memory"]["importance"])
            self.assertFalse(updated.output["memory"]["standing"])
            self.assertEqual("Transitions may be noisy", updated.output["memory"]["content"])
            store.close()
