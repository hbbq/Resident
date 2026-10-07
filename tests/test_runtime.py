import asyncio
import json
import tempfile
import unittest
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from resident.capabilities import Capability, diagnostic_capabilities
from resident.config import Config
from resident.domain import ModelTurn, ToolCall, WakeEvent
from resident.runtime import ResidentRuntime
from resident.store import Store, utc_now
from runtime_support import RecordingProvider

class MessageOnlyProvider(RecordingProvider):
    pass

class SingleToolProvider(RecordingProvider):
    async def create_conversation(self):
        return 'conversation-test'

    def __init__(self, name, arguments):
        super().__init__([ModelTurn('tool', tool_calls=(ToolCall('call', name, arguments),)), '{"outputs":[]}'])
        self.results = []
    async def respond(self, context, tools, results, **request):
        self.results.extend(results)
        return await super().respond(context, tools, results, **request)

class FailingProvider(RecordingProvider):
    async def create_conversation(self):
        return 'conversation-test'

    async def respond(self, *args, **request):
        raise RuntimeError('provider unavailable')

class LifecycleProvider(RecordingProvider):
    pass

class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def capability(name: str, *, description: str = "Test capability") -> Capability:
        async def handler(_):
            return {}
        return Capability(
            "test", "Test connector", name, description,
            {"type": "object", "properties": {}, "additionalProperties": False}, handler)

    async def test_first_capability_baseline_is_silent_and_restart_change_wakes(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Config(Path(temporary))
            original = [self.capability("one")]
            first = ResidentRuntime(
                config, MessageOnlyProvider(), capabilities=original,
                owner_output=lambda _: None, diagnostic_output=lambda _: None)
            first_queue = asyncio.Queue()
            await first.enqueue_startup_wakeups(first_queue)
            self.assertTrue(first_queue.empty())
            first.close()

            second = ResidentRuntime(
                config, MessageOnlyProvider(), capabilities=[*original, self.capability("two")],
                owner_output=lambda _: None, diagnostic_output=lambda _: None)
            second_queue = asyncio.Queue()
            await second.enqueue_startup_wakeups(second_queue)
            event = second_queue.get_nowait()
            self.assertEqual(("runtime", "capabilities_changed"), (event.source, event.reason))
            self.assertEqual({"added": ["two"], "removed": [], "changed": []}, event.payload)
            second.close()

    async def test_explicit_capability_changes_are_classified_and_suppress_noops(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = ResidentRuntime(
                Config(Path(temporary)), MessageOnlyProvider(),
                capabilities=[self.capability("one"), self.capability("removed")],
                owner_output=lambda _: None, diagnostic_output=lambda _: None)

            self.assertIsNone(runtime.replace_capabilities(runtime.capabilities))
            event = runtime.replace_capabilities([
                self.capability("one", description="Changed"), self.capability("added")])

            self.assertEqual({
                "added": ["added"], "removed": ["removed"], "changed": ["one"],
            }, event.payload)
            runtime.close()

    async def test_capability_wake_context_and_tools_use_same_current_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            provider = LifecycleProvider()
            runtime = ResidentRuntime(
                Config(Path(temporary)), provider, capabilities=[self.capability("old")],
                owner_output=lambda _: None, diagnostic_output=lambda _: None)
            event = runtime.replace_capabilities([self.capability("new")])

            await runtime.process(event)

            self.assertIn('"available_connectors": ["test"]', provider.requests[0]["instructions"])
            self.assertNotIn("old", provider.tool_sets[0])
            self.assertIn("new", provider.tool_sets[0])
            runtime.close()

    async def test_duplicate_dynamic_capabilities_are_rejected_before_exposure(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = ResidentRuntime(
                Config(Path(temporary)), MessageOnlyProvider(), capabilities=[self.capability("one")],
                owner_output=lambda _: None, diagnostic_output=lambda _: None)
            with self.assertRaisesRegex(ValueError, "Duplicate capability name: one"):
                runtime.register_capabilities([self.capability("one")])
            self.assertEqual(["one"], [capability.name for capability in runtime.capabilities])
            runtime.close()

    async def test_dynamic_capability_cannot_shadow_core_tool(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "conflicts with a core tool: create_intention"):
                ResidentRuntime(
                    Config(Path(temporary)), MessageOnlyProvider(),
                    capabilities=[self.capability("create_intention")],
                    owner_output=lambda _: None, diagnostic_output=lambda _: None)

    async def test_default_diagnostics_hide_successful_spontaneous_wake_but_keep_journal(self):
        with tempfile.TemporaryDirectory() as temporary:
            diagnostics: list[str] = []
            runtime = ResidentRuntime(
                Config(Path(temporary)), MessageOnlyProvider(),
                owner_output=lambda _: self.fail("Resident should not contact the owner"),
                diagnostic_output=diagnostics.append,
            )
            event = WakeEvent("event", "homeops", "measurement_changed", utc_now(), {})

            await runtime.process(event)

            self.assertEqual([], diagnostics)
            journal = {row[0] for row in runtime.store.connection.execute(
                "SELECT event_type FROM journal")}
            self.assertIn("wake.started", journal)
            self.assertIn("disposition.generated", journal)
            self.assertIn("wake.finished", journal)
            runtime.close()

    async def test_verbose_diagnostics_include_lifecycle_and_model_events(self):
        with tempfile.TemporaryDirectory() as temporary:
            diagnostics: list[str] = []
            runtime = ResidentRuntime(
                Config(Path(temporary), verbose=True), MessageOnlyProvider(),
                owner_output=lambda _: None, diagnostic_output=diagnostics.append,
            )

            await runtime.process(WakeEvent(
                "event", "homeops", "measurement_changed", utc_now(), {}))

            self.assertTrue(any(line.startswith("wake.started ") for line in diagnostics))
            self.assertTrue(any(line.startswith("disposition.generated ") for line in diagnostics))
            self.assertTrue(any(line.startswith("wake.finished ") for line in diagnostics))
            runtime.close()

    async def test_timeline_is_opt_in_and_omits_tool_arguments_and_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = ResidentRuntime(
                Config(Path(temporary), timeline=True),
                SingleToolProvider("diagnostics_current_time", {}),
                owner_output=lambda _: None, diagnostic_output=lambda _: None,
            )

            await runtime.process(WakeEvent(
                "event", "test", "timeline", utc_now(), {}))

            rows = runtime.store.connection.execute(
                "SELECT data_json FROM journal WHERE event_type='timeline' ORDER BY sequence"
            ).fetchall()
            events = [json.loads(row[0]) for row in rows]
            operations = {event["operation"] for event in events}
            self.assertIn("wake.process", operations)
            self.assertIn("provider.turn", operations)
            self.assertIn("tool.execute", operations)
            tool_events = [event for event in events
                           if event["operation"] == "tool.execute"]
            self.assertEqual(["started", "finished"], [
                event["moment"] for event in tool_events])
            self.assertEqual("ok", tool_events[1]["outcome"])
            self.assertEqual(
                {"call_id": "call", "tool_name": "diagnostics_current_time", "round": 0},
                {key: tool_events[1][key]
                 for key in ("call_id", "tool_name", "round")})
            self.assertGreaterEqual(tool_events[1]["duration_seconds"], 0.0)
            serialized = json.dumps(events)
            runtime.close()
            self.assertTrue(all("arguments" not in event and "result" not in event
                                for event in events))

    async def test_default_diagnostics_show_actionable_runtime_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            diagnostics: list[str] = []
            runtime = ResidentRuntime(
                Config(Path(temporary)), FailingProvider(),
                owner_output=lambda _: None, diagnostic_output=diagnostics.append,
            )

            with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
                await runtime.process(WakeEvent("event", "scheduler", "scheduled", utc_now(), {}))

            self.assertEqual(1, len(diagnostics))
            self.assertTrue(diagnostics[0].startswith("wake.failed "))
            runtime.close()

    async def test_scheduled_wakeup_survives_restart_and_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Config(Path(temporary))
            scheduler_provider = SingleToolProvider("schedule_wakeup", {
                "delay_seconds": 1, "reason": "check continuity", "context": {"origin": "test"}})
            first = ResidentRuntime(config, scheduler_provider, owner_output=lambda _: None, diagnostic_output=lambda _: None)
            await first.process(first.owner_message_event("schedule it"))
            schedule = first.store.connection.execute("SELECT id FROM scheduled_wakeups").fetchone()[0]
            first.store.connection.execute(
                "UPDATE scheduled_wakeups SET due_at=? WHERE id=?",
                ((datetime.now(UTC) - timedelta(seconds=1)).isoformat(), schedule))
            first.store.connection.commit()
            first.close()

            second_provider = LifecycleProvider()
            second = ResidentRuntime(config, second_provider, owner_output=lambda _: None, diagnostic_output=lambda _: None)
            queue = asyncio.Queue()
            await second.enqueue_due_wakeups(queue)
            event = await queue.get()
            self.assertEqual("scheduler", event.source)
            self.assertEqual({"origin": "test"}, event.payload["context"])
            await second.process(event)
            status = second.store.connection.execute(
                "SELECT status FROM scheduled_wakeups WHERE id=?", (schedule,)).fetchone()[0]
            self.assertEqual("completed", status)
            second.close()

    def test_direct_config_and_cli_window_defaults_match(self):
        direct = Config(Path("."))
        parsed = Config.from_env_and_args(["--data-dir", "."])

        self.assertEqual(180, direct.spontaneous_message_window_seconds)
        self.assertEqual(direct.spontaneous_message_window_seconds,
                         parsed.spontaneous_message_window_seconds)

    def test_claimed_schedule_is_recovered_when_store_reopens(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "resident.sqlite3"
            store = Store(path)
            schedule = store.schedule(utc_now(), "recover", {})
            self.assertEqual(1, len(store.claim_due_wakeups(utc_now())))
            store.close()
            reopened = Store(path)
            recovered = reopened.claim_due_wakeups(utc_now())
            self.assertEqual(schedule, recovered[0]["id"])
            reopened.close()

    def test_run_and_schedule_finalization_roll_back_together(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "resident.sqlite3"
            store = Store(path)
            schedule = store.schedule(utc_now(), "recover interrupted run", {})
            self.assertEqual(1, len(store.claim_due_wakeups(utc_now())))
            event = WakeEvent("event", "scheduler", "recover interrupted run", utc_now(), {})
            run_id = store.start_run(event)
            store.connection.execute("""
                CREATE TRIGGER interrupt_schedule_finalization
                BEFORE UPDATE OF status ON scheduled_wakeups
                WHEN NEW.status IN ('completed', 'failed')
                BEGIN
                  SELECT RAISE(ABORT, 'simulated crash');
                END
            """)

            with self.assertRaisesRegex(sqlite3.IntegrityError, "simulated crash"):
                store.finish_run(run_id, "completed", 1.0, 1, schedule)

            run = store.connection.execute(
                "SELECT status,finished_at FROM wake_runs WHERE id=?", (run_id,)).fetchone()
            self.assertEqual(("running", None), tuple(run))
            self.assertEqual("claimed", store.connection.execute(
                "SELECT status FROM scheduled_wakeups WHERE id=?", (schedule,)).fetchone()[0])
            store.close()

            reopened = Store(path)
            self.assertEqual("pending", reopened.connection.execute(
                "SELECT status FROM scheduled_wakeups WHERE id=?", (schedule,)).fetchone()[0])
            self.assertEqual(schedule, reopened.claim_due_wakeups(utc_now())[0]["id"])
            reopened.close()

    def test_startup_recovery_does_not_requeue_terminal_schedules(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "resident.sqlite3"
            store = Store(path)
            schedules = {
                status: store.schedule(utc_now(), status, {})
                for status in ("claimed", "completed", "failed")
            }
            for status, schedule in schedules.items():
                store.connection.execute(
                    "UPDATE scheduled_wakeups SET status=? WHERE id=?", (status, schedule))
            store.connection.commit()
            store.close()

            reopened = Store(path)
            statuses = dict(reopened.connection.execute(
                "SELECT reason,status FROM scheduled_wakeups"))
            self.assertEqual({
                "claimed": "pending", "completed": "completed", "failed": "failed",
            }, statuses)
            reopened.close()
