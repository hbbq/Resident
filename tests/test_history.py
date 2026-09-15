from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from resident.domain import WakeEvent
from resident.store import Store
from resident.tools import CORE_TOOL_NAMES, ToolRegistry


class HistoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temporary.name) / "resident.sqlite3")
        self.resident, self.owner = self.store.provision("Resident", "Owner", "")

    def tearDown(self):
        self.store.close()
        self.temporary.cleanup()

    def test_message_search_filters_pages_and_exposes_only_safe_fields(self):
        first = self.store.add_message("inbound", "private-sender", "literal 100% earlier")
        second = self.store.add_message("outbound", self.resident.id, "literal 100% later")
        third = self.store.add_message("inbound", self.owner.id, "unrelated")
        with self.store.connection:
            self.store.connection.execute(
                "UPDATE messages SET created_at='2026-01-01T00:00:00+00:00' WHERE id=?", (first,))
            self.store.connection.execute(
                "UPDATE messages SET created_at='2026-01-02T00:00:00+00:00' WHERE id=?", (second,))
            self.store.connection.execute(
                "UPDATE messages SET created_at='2026-01-03T00:00:00+00:00' WHERE id=?", (third,))

        result = self.store.search_messages(
            query="100%", from_time="2026-01-01T00:00:00+00:00",
            to_time="2026-01-02T00:00:00+00:00", limit=1, offset=1)

        self.assertEqual([first], [message["id"] for message in result])
        self.assertEqual(
            {"id", "direction", "content", "delivery_status", "created_at"}, set(result[0]))
        self.assertNotIn("sender_id", result[0])

    def test_wake_history_filters_and_redacts_each_journal_event_at_read_time(self):
        run_id = self.store.start_run(WakeEvent(
            "event", "scheduler", "check kitchen", "2026-01-01T00:00:00+00:00", {}))
        with self.store.connection:
            self.store.connection.execute(
                "UPDATE wake_runs SET started_at='2026-01-02T00:00:00+00:00' WHERE id=?", (run_id,))
        self.store.journal("wake.started", {"payload": {"credential": "secret"}}, run_id)
        self.store.journal("tool.called", {
            "call_id": "private", "name": "camera_capture_frame",
            "arguments": {"url": "rtsp://user:secret@example"},
        }, run_id)
        self.store.journal("tool.completed", {
            "name": "camera_capture_frame", "result": {"credential": "secret"},
            "attachments": [{"bytes": "secret"}],
        }, run_id)
        self.store.journal("model.message", {"content": "secret response"}, run_id)
        self.store.journal("wake.failed", {
            "error_type": "RuntimeError", "error": "credential=secret"}, run_id)

        result = self.store.wake_history(
            query="kitchen", source="scheduler", status="running",
            from_time="2026-01-01T00:00:00+00:00", limit=1)

        self.assertEqual(run_id, result[0]["id"])
        self.assertEqual(
            ["tool.called", "tool.completed", "wake.failed"],
            [event["type"] for event in result[0]["events"]])
        self.assertEqual("camera_capture_frame", result[0]["events"][0]["name"])
        self.assertFalse(result[0]["events_truncated"])
        self.assertNotIn("arguments", result[0]["events"][0])
        self.assertNotIn("result", result[0]["events"][1])
        self.assertNotIn("error", result[0]["events"][2])
        self.assertNotIn("secret", str(result))

    def test_wake_event_summaries_have_an_explicit_bound(self):
        run_id = self.store.start_run(WakeEvent("event", "owner", "busy", "", {}))
        for number in range(51):
            self.store.journal("tool.called", {"name": f"tool-{number}"}, run_id)

        result = self.store.wake_history(limit=1)

        self.assertEqual(50, len(result[0]["events"]))
        self.assertTrue(result[0]["events_truncated"])
        self.assertEqual("tool-1", result[0]["events"][0]["name"])
        self.assertEqual("tool-50", result[0]["events"][-1]["name"])


class HistoryToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_tools_are_bounded_reserved_and_read_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            resident, owner = store.provision("Resident", "Owner", "")
            message_id = store.add_message("inbound", owner.id, "older message")
            emitted = []
            registry = ToolRegistry(store, [], lambda _: {}, lambda *event: emitted.append(event))
            changes_before = store.connection.total_changes

            result = await registry.execute("search_communication", {"limit": 1})
            invalid = await registry.execute("list_wake_history", {"limit": 21})

            self.assertEqual(message_id, result.output["messages"][0]["id"])
            self.assertFalse(invalid.output["ok"])
            self.assertEqual(changes_before, store.connection.total_changes)
            self.assertEqual([], emitted)
            self.assertIn("search_communication", CORE_TOOL_NAMES)
            self.assertIn("list_wake_history", CORE_TOOL_NAMES)
            store.close()

    async def test_wake_history_excludes_the_active_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            store.provision("Resident", "Owner", "")
            prior = store.start_run(WakeEvent("prior-event", "owner", "prior", "", {}))
            active = store.start_run(WakeEvent("active-event", "owner", "active", "", {}))
            registry = ToolRegistry(
                store, [], lambda _: {}, lambda *_: None, current_run_id=active)

            result = await registry.execute("list_wake_history", {})

            self.assertEqual([prior], [run["id"] for run in result.output["wake_runs"]])
            store.close()

    async def test_invalid_time_is_a_safe_tool_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            store.provision("Resident", "Owner", "")
            registry = ToolRegistry(store, [], lambda _: {}, lambda *_: None)

            result = await registry.execute("search_communication", {"from_time": "not-a-time"})

            self.assertFalse(result.output["ok"])
            self.assertIn("ValueError", result.output["error"])
            store.close()
