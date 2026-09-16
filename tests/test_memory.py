from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

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

            self.assertEqual(7, store.connection.execute(
                "SELECT version FROM schema_version").fetchone()[0])
            memory = store.memory("legacy")
            self.assertEqual("resident", memory["source"])
            self.assertEqual(
                ("unknown", "low", "low", "unknown"),
                tuple(memory[name] for name in (
                    "kind", "importance", "confidence", "provenance")),
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
            self.assertEqual("Transitions may be noisy", updated.output["memory"]["content"])
            store.close()
