from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
import threading
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from resident.config import Config
from resident.capabilities import Capability
from resident.domain import ModelTurn, ToolCall, WakeEvent
from resident.domain import ImageAttachment, ToolResult, ToolSpec
from resident.provider import (OpenAIAgentsProvider, OpenAIResponsesProvider,
                               RemoteSessionUnavailable, RolloverRecoveryRequired)
from resident.runtime import ResidentRuntime
from resident.store import Store, utc_now


class LifecycleProvider:
    """Deterministic provider that exercises capability and communication tools."""

    def __init__(self):
        self.contexts: list[dict] = []
        self.round = 0

    async def respond(self, context, tools, results, previous_response_id=None):
        if previous_response_id is None:
            document = json.loads(context)
            self.contexts.append(document)
            self.round += 1
            content = document["wake_event"]["payload"].get("content", document["wake_event"]["reason"])
            return ModelTurn(f"response-{self.round}", tool_calls=(
                ToolCall(f"time-{self.round}", "diagnostics_current_time", {}),
                ToolCall(f"message-{self.round}", "send_owner_message", {"content": f"I observed {content}"}),
            ))
        return ModelTurn(f"final-{self.round}", message="Finished observable work.")


class SingleToolProvider:
    def __init__(self, name, arguments):
        self.name, self.arguments, self.called = name, arguments, False

    async def respond(self, context, tools, results, previous_response_id=None):
        if not self.called:
            self.called = True
            return ModelTurn("tool-response", tool_calls=(ToolCall("call", self.name, self.arguments),))
        return ModelTurn("done", message="done")


class MessageOnlyProvider:
    async def respond(self, context, tools, results, previous_response_id=None):
        return ModelTurn("done", message="No owner communication is needed.")


class FailingProvider:
    async def respond(self, context, tools, results, previous_response_id=None):
        raise RuntimeError("provider unavailable")


class ThreadRecordingStore(Store):
    def __init__(self, path):
        self.owner_thread_id = threading.get_ident()
        self.binding_save_threads: list[int] = []
        super().__init__(path)

    def save_agent_session_binding(self, provider, session_id, agent_id, last_turn_id):
        self.binding_save_threads.append(threading.get_ident())
        return super().save_agent_session_binding(
            provider, session_id, agent_id, last_turn_id)

    def bind_initial_agent_session(self, provider, session_id, agent_id, create_request,
                                   protocol_descriptor, mutable_settings):
        self.binding_save_threads.append(threading.get_ident())
        return super().bind_initial_agent_session(
            provider, session_id, agent_id, create_request,
            protocol_descriptor, mutable_settings)

    def bind_session_rollover(self, rollover_id, new_session_id, agent_id,
                              finalization_status):
        self.binding_save_threads.append(threading.get_ident())
        return super().bind_session_rollover(
            rollover_id, new_session_id, agent_id, finalization_status)


class ContinuationLifecycleProvider:
    def __init__(self, *, cancel=False):
        self.cancel = cancel
        self.discarded: list[str] = []
        self.release = asyncio.Event()

    async def respond(self, context, tools, results, previous_response_id=None):
        if previous_response_id is None:
            return ModelTurn("continuation", tool_calls=(ToolCall(
                "call", "create_intention", {"content": "work"}),))
        if self.cancel:
            await self.release.wait()
        raise RuntimeError("continuation failed")

    def discard_continuation(self, continuation_id):
        self.discarded.append(continuation_id)


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

            available = provider.contexts[0]["available_connectors"]
            self.assertEqual(["new"], [item["capability"]["name"] for item in available])
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
            self.assertIn("model.message", journal)
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
            self.assertTrue(any(line.startswith("model.message ") for line in diagnostics))
            self.assertTrue(any(line.startswith("wake.finished ") for line in diagnostics))
            runtime.close()

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

    async def test_failed_continuation_is_discarded(self):
        with tempfile.TemporaryDirectory() as temporary:
            provider = ContinuationLifecycleProvider()
            runtime = ResidentRuntime(
                Config(Path(temporary)), provider,
                owner_output=lambda _: None, diagnostic_output=lambda _: None,
            )

            with self.assertRaisesRegex(RuntimeError, "continuation failed"):
                await runtime.process(WakeEvent("event", "test", "failure", utc_now(), {}))

            self.assertEqual(["continuation"], provider.discarded)
            runtime.close()

    async def test_cancelled_continuation_is_discarded_and_reraises(self):
        with tempfile.TemporaryDirectory() as temporary:
            provider = ContinuationLifecycleProvider(cancel=True)
            runtime = ResidentRuntime(
                Config(Path(temporary)), provider,
                owner_output=lambda _: None, diagnostic_output=lambda _: None,
            )
            processing = asyncio.create_task(runtime.process(
                WakeEvent("event", "test", "cancellation", utc_now(), {})))
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            processing.cancel()

            with self.assertRaises(asyncio.CancelledError):
                await processing
            self.assertEqual(["continuation"], provider.discarded)
            runtime.close()

    async def test_owner_communication_remains_visible_in_default_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            delivered: list[str] = []
            diagnostics: list[str] = []
            runtime = ResidentRuntime(
                Config(Path(temporary)),
                SingleToolProvider("send_owner_message", {"content": "Worth your attention"}),
                owner_output=delivered.append, diagnostic_output=diagnostics.append,
            )

            await runtime.process(runtime.owner_message_event("Any news?"))

            self.assertEqual(["Worth your attention"], delivered)
            self.assertEqual([], diagnostics)
            runtime.close()

    async def test_final_model_message_does_not_duplicate_owner_communication(self):
        with tempfile.TemporaryDirectory() as temporary:
            delivered: list[str] = []
            runtime = ResidentRuntime(
                Config(Path(temporary)),
                SingleToolProvider("send_owner_message", {"content": "The intentional reply"}),
                owner_output=delivered.append, diagnostic_output=lambda _: None,
            )

            await runtime.process(runtime.owner_message_event("Please reply"))

            self.assertEqual(["The intentional reply"], delivered)
            model_messages = runtime.store.connection.execute(
                "SELECT data_json FROM journal WHERE event_type='model.message'").fetchall()
            self.assertEqual([{"content": "done"}], [json.loads(row[0]) for row in model_messages])
            runtime.close()

    async def test_final_model_message_alone_is_not_owner_communication(self):
        with tempfile.TemporaryDirectory() as temporary:
            delivered: list[str] = []
            diagnostics: list[str] = []
            runtime = ResidentRuntime(
                Config(Path(temporary)), MessageOnlyProvider(),
                owner_output=delivered.append, diagnostic_output=diagnostics.append,
            )

            await runtime.process(runtime.owner_message_event("Are you there?"))

            self.assertEqual([], delivered)
            self.assertEqual([], diagnostics)
            self.assertEqual(0, runtime.store.connection.execute(
                "SELECT count(*) FROM messages WHERE direction='outbound'").fetchone()[0])
            runtime.close()

    async def test_transport_failure_is_persisted_without_model_message_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            diagnostics: list[str] = []

            def failed_transport(_: str) -> None:
                raise RuntimeError("transport offline")

            runtime = ResidentRuntime(
                Config(Path(temporary)),
                SingleToolProvider("send_owner_message", {"content": "The intended reply"}),
                owner_output=failed_transport, diagnostic_output=diagnostics.append,
            )

            await runtime.process(runtime.owner_message_event("Please reply"))

            message = runtime.store.connection.execute(
                "SELECT content,delivery_status FROM messages WHERE direction='outbound'").fetchone()
            self.assertEqual(("The intended reply", "transport_failed"), tuple(message))
            self.assertEqual(1, len(diagnostics))
            self.assertTrue(diagnostics[0].startswith("communication.failed "))
            event_types = [row[0] for row in runtime.store.connection.execute(
                "SELECT event_type FROM journal")]
            self.assertIn("communication.failed", event_types)
            self.assertIn("model.message", event_types)
            runtime.close()

    async def test_full_lifecycle_retains_identity_and_communication_across_restart(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Config(Path(temporary))
            delivered: list[str] = []
            first_provider = LifecycleProvider()
            first = ResidentRuntime(config, first_provider, owner_output=delivered.append, diagnostic_output=lambda _: None)
            identity = first.resident.id
            await first.process(first.owner_message_event("hello home"))
            first.close()

            second_provider = LifecycleProvider()
            second = ResidentRuntime(config, second_provider, owner_output=delivered.append, diagnostic_output=lambda _: None)
            self.assertEqual(identity, second.resident.id)
            await second.process(second.owner_message_event("what did I say before?"))

            context = second_provider.contexts[0]
            self.assertTrue(any(m["content"] == "hello home" for m in context["recent_communication"]))
            self.assertEqual(2, len(delivered))
            statuses = [r[0] for r in second.store.connection.execute("SELECT status FROM wake_runs")]
            self.assertEqual(["completed", "completed"], statuses)
            event_types = {r[0] for r in second.store.connection.execute("SELECT event_type FROM journal")}
            self.assertTrue({"wake.started", "context.assembled", "tool.called", "tool.completed",
                             "communication.delivered", "wake.sleeping"}.issubset(event_types))
            second.close()

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

    async def test_spontaneous_attention_budget_rejects_without_queueing(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Config(Path(temporary), spontaneous_message_limit=0)
            provider = SingleToolProvider("send_owner_message", {"content": "unsolicited"})
            delivered: list[str] = []
            runtime = ResidentRuntime(config, provider, owner_output=delivered.append, diagnostic_output=lambda _: None)
            from resident.domain import WakeEvent
            event = WakeEvent("event", "scheduler", "scheduled", utc_now(), {})
            await runtime.process(event)
            row = runtime.store.connection.execute(
                "SELECT delivery_status,spontaneous FROM messages WHERE direction='outbound'").fetchone()
            self.assertEqual(("rejected_attention_budget", 1), tuple(row))
            self.assertEqual([], delivered)
            self.assertEqual(0, runtime.store.connection.execute(
                "SELECT count(*) FROM messages WHERE delivery_status='queued'").fetchone()[0])
            runtime.close()

    async def test_context_is_bounded_and_trigger_is_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Config(Path(temporary), context_messages=2)
            provider = LifecycleProvider()
            runtime = ResidentRuntime(config, provider, owner_output=lambda _: None, diagnostic_output=lambda _: None)
            for number in range(6):
                runtime.store.add_message("inbound", runtime.owner.id, f"older {number}")
            event = runtime.owner_message_event("bounded topic with full payload")
            await runtime.process(event)
            context = provider.contexts[0]
            self.assertEqual("bounded topic with full payload", context["wake_event"]["payload"]["content"])
            self.assertLessEqual(len(context["recent_communication"]), 2)
            runtime.close()


class AttentionBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_spontaneous_limit_counts_delivered_messages_and_rejects_next(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = ResidentRuntime(
                Config(Path(temporary), spontaneous_message_limit=2),
                MessageOnlyProvider(), owner_output=lambda _: None, diagnostic_output=lambda _: None)

            first = await runtime._send_owner_message("first")
            second = await runtime._send_owner_message("second")
            third = await runtime._send_owner_message("third")

            self.assertTrue(first["delivered"])
            self.assertTrue(second["delivered"])
            self.assertEqual("Spontaneous owner-message attention budget exceeded", third["reason"])
            self.assertEqual(2, runtime.store.spontaneous_count_since(
                (datetime.now(UTC) - timedelta(seconds=60)).isoformat()))
            runtime.close()

    async def test_rejected_attempt_does_not_consume_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = ResidentRuntime(
                Config(Path(temporary), spontaneous_message_limit=1),
                MessageOnlyProvider(), owner_output=lambda _: None, diagnostic_output=lambda _: None)

            delivered = await runtime._send_owner_message("first")
            rejected = await runtime._send_owner_message("rejected")
            count = runtime.store.spontaneous_count_since(
                (datetime.now(UTC) - timedelta(seconds=60)).isoformat())

            self.assertTrue(delivered["delivered"])
            self.assertFalse(rejected["delivered"])
            self.assertEqual(1, count)
            self.assertEqual(1, runtime.store.connection.execute(
                "SELECT count(*) FROM messages WHERE delivery_status='rejected_attention_budget'").fetchone()[0])
            runtime.close()

    async def test_owner_reply_bypasses_spontaneous_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = ResidentRuntime(
                Config(Path(temporary), spontaneous_message_limit=0),
                MessageOnlyProvider(), owner_output=lambda _: None, diagnostic_output=lambda _: None)
            runtime._active_event = WakeEvent("event", "owner", "owner_message", utc_now(), {})

            result = await runtime._send_owner_message("direct reply")

            self.assertTrue(result["delivered"])
            self.assertFalse(result["spontaneous"])
            self.assertEqual(0, runtime.store.spontaneous_count_since(
                (datetime.now(UTC) - timedelta(seconds=60)).isoformat()))
            runtime.close()

    async def test_delivered_spontaneous_messages_survive_restart(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            first = ResidentRuntime(
                Config(path, spontaneous_message_limit=1),
                MessageOnlyProvider(), owner_output=lambda _: None, diagnostic_output=lambda _: None)
            self.assertTrue((await first._send_owner_message("before restart"))["delivered"])
            first.close()

            second = ResidentRuntime(
                Config(path, spontaneous_message_limit=1),
                MessageOnlyProvider(), owner_output=lambda _: None, diagnostic_output=lambda _: None)
            result = await second._send_owner_message("after restart")

            self.assertEqual("Spontaneous owner-message attention budget exceeded", result["reason"])
            second.close()

    async def test_messages_older_than_window_leave_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = ResidentRuntime(
                Config(Path(temporary), spontaneous_message_limit=1, spontaneous_message_window_seconds=60),
                MessageOnlyProvider(), owner_output=lambda _: None, diagnostic_output=lambda _: None)
            await runtime._send_owner_message("old")
            old_timestamp = (datetime.now(UTC) - timedelta(seconds=61)).isoformat()
            runtime.store.connection.execute(
                "UPDATE messages SET created_at=? WHERE direction='outbound'", (old_timestamp,))
            runtime.store.connection.commit()

            result = await runtime._send_owner_message("new")

            self.assertTrue(result["delivered"])
            runtime.close()

    def test_window_boundary_is_inclusive(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            resident, _ = store.provision("Resident", "Owner", "")
            boundary = (datetime.now(UTC) - timedelta(seconds=60)).isoformat()
            store.connection.execute(
                "INSERT INTO messages VALUES(?,?,?,?,?,?,?)",
                ("boundary", "outbound", resident.id, "boundary", 1, "delivered", boundary))
            store.connection.commit()

            self.assertEqual(1, store.spontaneous_count_since(boundary))
            store.close()

    def test_direct_config_and_cli_window_defaults_match(self):
        direct = Config(Path("."))
        parsed = Config.from_env_and_args(["--data-dir", "."])

        self.assertEqual(180, direct.spontaneous_message_window_seconds)
        self.assertEqual(direct.spontaneous_message_window_seconds,
                         parsed.spontaneous_message_window_seconds)


class StoreTests(unittest.TestCase):
    def test_existing_agent_actions_gain_ephemeral_attachment_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "resident.sqlite3"
            connection = sqlite3.connect(path)
            connection.executescript("""
                CREATE TABLE schema_version(version INTEGER NOT NULL);
                INSERT INTO schema_version VALUES(10);
                CREATE TABLE agent_tool_actions(
                  provider TEXT NOT NULL, session_id TEXT NOT NULL, turn_id TEXT NOT NULL,
                  call_id TEXT NOT NULL, name TEXT NOT NULL, arguments_json TEXT NOT NULL,
                  status TEXT NOT NULL CHECK(status IN ('pending','completed')),
                  output_json TEXT, created_at TEXT NOT NULL, completed_at TEXT,
                  PRIMARY KEY(provider,session_id,call_id));
            """)
            connection.execute(
                "INSERT INTO agent_tool_actions VALUES(?,?,?,?,?,?,?,?,?,?)",
                ("openai_agents", "session", "turn", "call", "clock", "{}",
                 "completed", '{"ok":true}', utc_now(), utc_now()))
            connection.commit()
            connection.close()

            store = Store(path)
            action = store.begin_agent_tool_action(
                "openai_agents", "session", "turn", "call", "clock", {})
            self.assertEqual(15, store.connection.execute(
                "SELECT version FROM schema_version").fetchone()[0])
            self.assertFalse(action["attachments_ephemeral"])
            self.assertEqual({"ok": True}, action["output"])
            store.close()

    def test_existing_schedule_table_is_migrated_to_support_failed_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "resident.sqlite3"
            connection = sqlite3.connect(path)
            connection.executescript("""
                CREATE TABLE schema_version(version INTEGER NOT NULL);
                INSERT INTO schema_version VALUES(1);
                CREATE TABLE scheduled_wakeups(
                  id TEXT PRIMARY KEY, due_at TEXT NOT NULL, reason TEXT NOT NULL,
                  context_json TEXT NOT NULL,
                  status TEXT NOT NULL CHECK(status IN ('pending','claimed','completed')),
                  created_at TEXT NOT NULL);
            """)
            connection.execute(
                "INSERT INTO scheduled_wakeups VALUES(?,?,?,?,?,?)",
                ("schedule", utc_now(), "migrate", "{}", "pending", utc_now()))
            connection.commit()
            connection.close()

            store = Store(path)
            self.assertEqual(15, store.connection.execute(
                "SELECT version FROM schema_version").fetchone()[0])
            self.assertEqual(1, len(store.claim_due_wakeups(utc_now())))
            event = WakeEvent("event", "scheduler", "migrate", utc_now(), {})
            run_id = store.start_run(event)
            store.finish_run(run_id, "failed", 0, 0, "schedule")
            self.assertEqual("failed", store.connection.execute(
                "SELECT status FROM scheduled_wakeups WHERE id='schedule'").fetchone()[0])
            store.close()

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

    def test_legacy_pending_rollover_is_migrated_as_uncertain_not_retried(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "resident.sqlite3"
            store = Store(path)
            with store.connection:
                store.connection.execute("""
                    INSERT INTO session_rollovers(
                      id,provider,old_session_id,reason,requested_by,
                      finalization_status,status,created_at)
                    VALUES('legacy','openai_agents','session-old','change','runtime',
                      'pending','pending',?)
                """, (utc_now(),))
                store.connection.execute(
                    "UPDATE schema_version SET version=13")
            store.close()

            reopened = Store(path)
            rollover = reopened.pending_session_rollover("openai_agents")
            self.assertEqual("create_uncertain", rollover["creation_state"])
            self.assertIsNone(rollover.get("create_request"))
            reopened.close()


class OpenAIAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_agents_runtime_persists_session_and_turn_on_sqlite_owner_thread(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "resident.sqlite3"
            state = {"status": "idle", "turns": {}, "items": []}
            request_threads = []
            requests = []

            def fake_request(method, request_path, body=None, **_):
                request_threads.append(threading.get_ident())
                requests.append((method, request_path, body))
                if request_path == "/agents/sessions" and method == "POST":
                    state["turns"]["turn-1"] = "waiting"
                    state["status"] = "requires_action"
                    state["items"] = [{
                        "id": "input-turn-1", "type": "message", "role": "user",
                        "turn_id": "turn-1", "content": [
                            {"type": "input_text", "text": body["input"]}],
                    }]
                    return {"id": "session-1", "status": "requires_action",
                            "agent": {"id": "agent-1"}}
                if request_path == "/agents/sessions/session-1" and method == "GET":
                    session = {"id": "session-1", "status": state["status"],
                               "agent": {"id": "agent-1"}}
                    if state["status"] == "requires_action":
                        session["required_actions"] = [{
                            "type": "function_call", "turn_id": "turn-1",
                            "call_id": "call-1", "name": "clock", "arguments": {},
                        }]
                    return session
                if request_path == "/agents/sessions/session-1" and method == "POST":
                    return {"id": "session-1", "status": state["status"],
                            "agent": {"id": "agent-1"}}
                if request_path.endswith("/events"):
                    event = body["events"][0]
                    if event["type"] == "agent.session.input.message":
                        turn_id = "turn-1" if not state["turns"] else "turn-2"
                        state["turns"][turn_id] = (
                            "waiting" if turn_id == "turn-1" else "completed")
                        state["status"] = (
                            "requires_action" if turn_id == "turn-1" else "idle")
                        state["items"].insert(0, {
                            "id": f"input-{turn_id}", "type": "message", "role": "user",
                            "turn_id": turn_id, "content": event["input"][0]["content"],
                        })
                    else:
                        state["turns"]["turn-1"] = "completed"
                        state["status"] = "idle"
                        state["items"].insert(0, {
                            "id": "output-turn-1", "type": "message", "role": "assistant",
                            "turn_id": "turn-1", "content": [
                                {"type": "output_text", "text": "first complete"}],
                        })
                    return {}
                if "/turns?" in request_path:
                    if not state["turns"]:
                        return {"data": []}
                    turn_id = next(reversed(state["turns"]))
                    return {"data": [{"id": turn_id, "status": state["turns"][turn_id]}]}
                if "/turns/" in request_path:
                    turn_id = request_path.rsplit("/", 1)[-1]
                    return {"id": turn_id, "status": state["turns"][turn_id]}
                if "/items?" in request_path:
                    return {"data": state["items"]}
                raise AssertionError((method, request_path, body))

            store = ThreadRecordingStore(path)
            provider = OpenAIAgentsProvider("test-key", "gpt-5.6-luna", poll_seconds=0)
            provider._request = fake_request
            runtime = ResidentRuntime(
                Config(Path(temporary)), provider, store=store, capabilities=[],
                owner_output=lambda _: None, diagnostic_output=lambda _: None)
            first_context = json.dumps({
                "wake_event": {"id": "wake-1", "source": "connector", "payload": {}}})

            first = await provider.respond(first_context, [], [])

            self.assertEqual("turn-1", first.response_id)
            self.assertEqual("session-1", store.agent_session_binding("openai_agents")["session_id"])
            self.assertIsNone(store.agent_session_binding("openai_agents")["agent_id"])
            self.assertIsNone(store.agent_session_binding("openai_agents")["last_turn_id"])
            completed = await provider.respond(
                first_context, [], [ToolResult("call-1", {"ok": True})], first.response_id)
            self.assertEqual("first complete", completed.message)
            self.assertEqual(
                "turn-1", store.agent_session_binding("openai_agents")["last_turn_id"])
            self.assertEqual(
                [store.owner_thread_id, store.owner_thread_id], store.binding_save_threads)
            self.assertTrue(request_threads)
            self.assertTrue(all(thread_id != store.owner_thread_id for thread_id in request_threads))
            runtime.close()

            reopened_store = ThreadRecordingStore(path)
            recovered = OpenAIAgentsProvider("test-key", "gpt-5.6-luna", poll_seconds=0)
            recovered._request = fake_request
            reopened_runtime = ResidentRuntime(
                Config(Path(temporary)), recovered, store=reopened_store, capabilities=[],
                owner_output=lambda _: None, diagnostic_output=lambda _: None)
            self.assertEqual("session-1", recovered._session_id)
            self.assertEqual("turn-1", recovered._last_turn_id)

            second_context = json.dumps({
                "wake_event": {"id": "wake-2", "source": "connector", "payload": {}}})
            second = await recovered.respond(second_context, [], [])

            self.assertEqual("turn-2", second.response_id)
            creates = [body for method, request_path, body in requests
                       if method == "POST" and request_path == "/agents/sessions"]
            message_events = [body for method, request_path, body in requests
                              if method == "POST" and request_path.endswith("/events")
                              and body["events"][0]["type"] == "agent.session.input.message"]
            self.assertEqual(1, len(creates))
            self.assertIn('"id":"wake-1"', creates[0]["input"])
            self.assertEqual(1, len(message_events))
            self.assertIn(
                '"id":"wake-2"',
                message_events[0]["events"][0]["input"][0]["content"][0]["text"])
            self.assertEqual(
                "turn-2", reopened_store.agent_session_binding("openai_agents")["last_turn_id"])
            self.assertEqual(
                [reopened_store.owner_thread_id], reopened_store.binding_save_threads)
            reopened_runtime.close()

    async def test_agents_restart_replaces_missing_session_without_reusing_session_agent_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "resident.sqlite3"
            sessions = {}
            creates = []

            def fake_request(method, request_path, body=None, **_):
                if request_path == "/agents/sessions" and method == "POST":
                    if "agent_id" in body:
                        raise RuntimeError(
                            f"No persisted agent found: {body['agent_id']}. "
                            "Session-local agent IDs cannot be reused")
                    number = len(creates) + 1
                    session_id, turn_id = f"session-{number}", f"turn-{number}"
                    creates.append(body)
                    sessions[session_id] = {
                        "turn_id": turn_id,
                        "agent_id": f"agent-session-{number}",
                        "input": body["input"],
                    }
                    return {
                        "id": session_id, "status": "idle",
                        "agent": {"id": f"agent-session-{number}"},
                    }
                if method == "GET" and request_path.startswith("/agents/sessions/"):
                    parts = request_path.split("/")
                    session_id = parts[3].split("?", 1)[0]
                    if session_id not in sessions:
                        raise RuntimeError("OpenAI Agents API returned HTTP 404: gone")
                    state = sessions[session_id]
                    if "/items?" in request_path:
                        return {"data": [{
                            "id": f"input-{state['turn_id']}", "type": "message",
                            "role": "user", "turn_id": state["turn_id"],
                            "content": [{"type": "input_text", "text": state["input"]}],
                        }]}
                    if "/turns/" in request_path:
                        return {"id": state["turn_id"], "status": "completed"}
                    return {
                        "id": session_id, "status": "idle",
                        "agent": {"id": state["agent_id"]},
                    }
                raise AssertionError((method, request_path, body))

            store = Store(path)
            first_provider = OpenAIAgentsProvider(
                "test-key", "gpt-5.6-luna", poll_seconds=0)
            first_provider._request = fake_request
            first_runtime = ResidentRuntime(
                Config(Path(temporary)), first_provider, store=store, capabilities=[],
                owner_output=lambda _: None, diagnostic_output=lambda _: None)
            first = await first_provider.respond(json.dumps({
                "wake_event": {"id": "wake-1", "source": "connector", "payload": {}},
            }), [], [])
            self.assertEqual("turn-1", first.response_id)

            # Reproduce a binding written by the broken implementation, then
            # make the original remote session unavailable before restart.
            store.save_agent_session_binding(
                "openai_agents", "session-1", "agent-session-1", "turn-1")
            first_runtime.close()
            sessions.pop("session-1")

            reopened_store = Store(path)
            restarted_provider = OpenAIAgentsProvider(
                "test-key", "gpt-5.6-luna", poll_seconds=0)
            restarted_provider._request = fake_request
            restarted_runtime = ResidentRuntime(
                Config(Path(temporary)), restarted_provider, store=reopened_store,
                capabilities=[], owner_output=lambda _: None,
                diagnostic_output=lambda _: None)

            await restarted_runtime.process(WakeEvent(
                "wake-2", "connector", "changed", utc_now(), {}))

            self.assertEqual(2, len(creates))
            self.assertTrue(all("agent_id" not in body for body in creates))
            binding = reopened_store.agent_session_binding("openai_agents")
            self.assertEqual("session-2", binding["session_id"])
            self.assertIsNone(binding["agent_id"])
            self.assertEqual("turn-2", binding["last_turn_id"])
            restarted_runtime.close()

    def test_agents_binding_migrates_legacy_session_local_agent_id(self):
        provider = OpenAIAgentsProvider("test-key", "gpt-5.6-luna")
        binding = {
            "session_id": "session-persisted",
            "agent_id": "agent-persisted",
            "last_turn_id": "turn-persisted",
        }

        provider.bind_session_store(
            lambda: dict(binding),
            lambda session_id, agent_id, last_turn_id: binding.update(
                session_id=session_id, agent_id=agent_id, last_turn_id=last_turn_id),
        )

        self.assertEqual("session-persisted", provider._session_id)
        self.assertEqual("turn-persisted", provider._last_turn_id)
        self.assertIsNone(binding["agent_id"])

    def test_agents_restored_matching_session_needs_no_configuration_update(self):
        provider = OpenAIAgentsProvider("test-key", "gpt-5.6-luna")
        provider._session_id = "session-1"
        spec = ToolSpec("clock", "Read clock", {"type": "object"})
        remote_agent = provider._agent_config([spec])
        remote_agent.update(id="agent-1", name="ignored response name")
        remote_agent["tools"][0]["defer_loading"] = False
        requests = []

        def fake_request(method, path, body=None, **_):
            requests.append((method, path, body))
            if path == "/agents/sessions/session-1" and method == "GET":
                return {
                    "id": "session-1", "status": "idle",
                    "agent": remote_agent,
                }
            raise AssertionError((method, path, body))

        provider._request = fake_request
        session, created = provider._ensure_session([spec], initial_input="wake")

        self.assertEqual("session-1", session["id"])
        self.assertFalse(created)
        self.assertEqual([("GET", "/agents/sessions/session-1", None)], requests)
        self.assertIsNotNone(provider._tool_fingerprint)

    def test_agents_mutable_settings_can_be_cleared(self):
        provider = OpenAIAgentsProvider("test-key", "gpt-5.6-luna")
        provider._session_id = "session-1"
        remote_agent = provider._agent_config([])
        remote_agent.update(
            reasoning={"effort": "high"}, service_tier="priority")
        requests = []

        def fake_request(method, path, body=None, **_):
            requests.append((method, path, body))
            if method == "GET":
                return {"id": "session-1", "status": "idle", "agent": remote_agent}
            if method == "PATCH":
                return {"id": "session-1", "status": "idle"}
            raise AssertionError((method, path, body))

        provider._request = fake_request

        session, created = provider._ensure_session([], initial_input="wake")

        self.assertFalse(created)
        self.assertEqual("session-1", session["id"])
        self.assertEqual({"agent": {"reasoning": None, "service_tier": None}},
                         requests[-1][2])

    def test_agents_mutable_settings_cover_all_transitions(self):
        cases = (
            ("unset to set", {}, "high", "priority",
             {"reasoning": {"effort": "high"}, "service_tier": "priority"}),
            ("set to different", {"reasoning": {"effort": "low"},
                                  "service_tier": "default"}, "high", "priority",
             {"reasoning": {"effort": "high"}, "service_tier": "priority"}),
            ("set to unset", {"reasoning": {"effort": "high"},
                              "service_tier": "priority"}, None, None,
             {"reasoning": None, "service_tier": None}),
            ("unchanged", {"reasoning": {"effort": "high"},
                           "service_tier": "priority"}, "high", "priority", {}),
            ("unset unchanged", {}, None, None, {}),
        )
        for name, remote_settings, reasoning, service_tier, expected in cases:
            with self.subTest(name=name):
                provider = OpenAIAgentsProvider(
                    "test-key", "gpt-5.6-luna", reasoning_effort=reasoning,
                    service_tier=service_tier)
                remote = {"model": "gpt-5.6-luna", **remote_settings}
                self.assertEqual(expected, provider._mutable_patch(remote))

    def test_agents_missing_mutable_settings_are_patched_without_rollover(self):
        provider = OpenAIAgentsProvider(
            "test-key", "gpt-5.6-luna", reasoning_effort="high",
            service_tier="priority")
        provider._session_id = "session-1"
        remote_agent = provider._agent_config([])
        remote_agent.pop("reasoning")
        remote_agent.pop("service_tier")
        requests = []

        def fake_request(method, path, body=None, **_):
            requests.append((method, path, body))
            if method == "GET":
                return {"id": "session-1", "status": "idle", "agent": remote_agent}
            if method == "PATCH":
                return {"id": "session-1", "status": "idle"}
            raise AssertionError((method, path, body))

        provider._request = fake_request
        session, created = provider._ensure_session([], initial_input="wake")

        self.assertFalse(created)
        self.assertEqual("session-1", session["id"])
        self.assertEqual(["GET", "PATCH"], [request[0] for request in requests])
        self.assertEqual({"agent": {
            "reasoning": {"effort": "high"}, "service_tier": "priority",
        }}, requests[-1][2])

    def test_restored_partial_remote_uses_durable_mutable_settings_for_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "resident.sqlite3"
            store = Store(path)
            seed = OpenAIAgentsProvider("test-key", "model-old")
            store.save_agent_session_binding("openai_agents", "session-1", None, None)
            store.save_session_protocol(
                "openai_agents", "session-1",
                seed._agent_protocol(seed._agent_config([])))
            store.save_session_mutable_settings("openai_agents", "session-1", {
                "model": "model-old", "reasoning": {"effort": "low"},
                "service_tier": "default",
            })
            provider = OpenAIAgentsProvider(
                "test-key", "model-new", reasoning_effort="high",
                service_tier="priority")
            requests = []

            def fake_request(method, path, body=None, **_):
                requests.append((method, path, body))
                if method == "GET":
                    return {"id": "session-1", "status": "idle", "agent": {}}
                if method == "PATCH":
                    return {"id": "session-1", "status": "idle"}
                raise AssertionError((method, path, body))

            provider._request = fake_request
            runtime = ResidentRuntime(
                Config(Path(temporary)), provider, store=store, capabilities=[],
                owner_output=lambda _: None, diagnostic_output=lambda _: None)
            session, created = provider._ensure_session([], initial_input="wake")

            self.assertFalse(created)
            self.assertEqual("session-1", session["id"])
            self.assertEqual({"agent": {
                "model": "model-new", "reasoning": {"effort": "high"},
                "service_tier": "priority",
            }}, requests[-1][2])
            self.assertEqual({
                "model": "model-new", "reasoning": {"effort": "high"},
                "service_tier": "priority",
            }, store.session_mutable_settings("openai_agents", "session-1"))
            runtime.close()

            reopened = Store(path)
            clearing = OpenAIAgentsProvider("test-key", "model-new")
            clear_requests = []

            def clear_request(method, path, body=None, **_):
                clear_requests.append((method, path, body))
                if method == "GET":
                    return {"id": "session-1", "status": "idle", "agent": {}}
                if method == "PATCH":
                    return {"id": "session-1", "status": "idle"}
                raise AssertionError((method, path, body))

            clearing._request = clear_request
            restarted = ResidentRuntime(
                Config(Path(temporary)), clearing, store=reopened, capabilities=[],
                owner_output=lambda _: None, diagnostic_output=lambda _: None)
            clearing._ensure_session([], initial_input="wake")
            self.assertEqual({"agent": {
                "reasoning": None, "service_tier": None,
            }}, clear_requests[-1][2])
            self.assertEqual({
                "model": "model-new", "reasoning": None, "service_tier": None,
            }, reopened.session_mutable_settings("openai_agents", "session-1"))
            restarted.close()

    def test_initial_create_atomically_binds_exact_applied_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "resident.sqlite3"
            store = Store(path)
            provider = OpenAIAgentsProvider(
                "test-key", "model-a", reasoning_effort="low", service_tier="flex")
            create_bodies = []

            def create_request(method, request_path, body=None, **_):
                if method == "POST" and request_path == "/agents/sessions":
                    create_bodies.append(body)
                    return {"id": "session-1", "status": "idle"}
                raise AssertionError((method, request_path, body))

            provider._request = create_request
            runtime = ResidentRuntime(
                Config(Path(temporary)), provider, store=store, capabilities=[],
                owner_output=lambda _: None, diagnostic_output=lambda _: None)
            provider._ensure_session([], initial_input="initial bootstrap")

            expected_protocol = provider._agent_protocol(create_bodies[0]["agent"])
            expected_mutable = provider._desired_mutable_settings()
            self.assertEqual("session-1", store.agent_session_binding(
                "openai_agents")["session_id"])
            self.assertEqual(expected_protocol, store.session_protocol(
                "openai_agents", "session-1"))
            self.assertEqual(expected_mutable, store.session_mutable_settings(
                "openai_agents", "session-1"))
            runtime.close()

            reopened = Store(path)
            changed = OpenAIAgentsProvider(
                "test-key", "model-b", reasoning_effort="high", service_tier="priority")
            requests = []

            def partial_request(method, request_path, body=None, **_):
                requests.append((method, request_path, body))
                if method == "GET":
                    return {"id": "session-1", "status": "idle", "agent": {}}
                if method == "PATCH":
                    return {"id": "session-1", "status": "idle"}
                raise AssertionError((method, request_path, body))

            changed._request = partial_request
            restarted = ResidentRuntime(
                Config(Path(temporary)), changed, store=reopened, capabilities=[],
                owner_output=lambda _: None, diagnostic_output=lambda _: None)
            self.assertEqual(expected_protocol, changed._protocol_descriptor)
            self.assertEqual(expected_mutable, changed._mutable_settings_descriptor)
            changed._ensure_session([], initial_input="wake after restart")
            self.assertEqual(["GET", "PATCH"], [method for method, _, _ in requests])
            self.assertEqual({"agent": changed._desired_mutable_settings()}, requests[-1][2])
            self.assertEqual(expected_protocol, reopened.session_protocol(
                "openai_agents", "session-1"))
            restarted.close()

    def test_initial_binding_transaction_rolls_back_on_descriptor_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            provider = OpenAIAgentsProvider("test-key", "model")
            agent = provider._agent_config([])
            request = {
                "environment": {"type": "none"}, "agent": agent,
                "input": "initial bootstrap", "metadata": {"managed_by": "resident"},
            }
            with store.connection:
                store.connection.execute("""
                    CREATE TRIGGER fail_initial_protocol BEFORE INSERT
                    ON session_protocol_descriptors
                    BEGIN SELECT RAISE(ABORT, 'simulated descriptor failure'); END
                """)

            with self.assertRaisesRegex(sqlite3.IntegrityError, "simulated descriptor failure"):
                store.bind_initial_agent_session(
                    "openai_agents", "session-1", None, request,
                    provider._agent_protocol(agent), provider._desired_mutable_settings())

            self.assertIsNone(store.agent_session_binding("openai_agents"))
            self.assertIsNone(store.session_protocol("openai_agents", "session-1"))
            self.assertIsNone(store.session_mutable_settings("openai_agents", "session-1"))
            store.close()

    def test_new_chapter_without_binding_is_satisfied_by_one_initial_create(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            provider = OpenAIAgentsProvider("test-key", "model")
            requests = []

            def fake_request(method, path, body=None, **_):
                requests.append((method, path, body))
                if method == "POST":
                    return {"id": "session-initial", "status": "idle"}
                if method == "GET":
                    return {"id": "session-initial", "status": "idle",
                            "agent": provider._agent_config([])}
                raise AssertionError((method, path, body))

            provider._request = fake_request
            runtime = ResidentRuntime(
                Config(Path(temporary), new_chapter=True), provider, store=store,
                capabilities=[], owner_output=lambda _: None,
                diagnostic_output=lambda _: None)

            first, created = provider._ensure_session(
                [], initial_input="initial bootstrap")
            second, reused_create = provider._ensure_session(
                [], initial_input="following wake")

            self.assertTrue(created)
            self.assertFalse(reused_create)
            self.assertEqual("session-initial", first["id"])
            self.assertEqual("session-initial", second["id"])
            self.assertEqual(1, sum(method == "POST" for method, _, _ in requests))
            self.assertIsNone(provider._requested_rollover_reason)
            attempt = store.connection.execute("""
                SELECT old_session_id,reason,creation_state,status
                FROM session_rollovers
            """).fetchone()
            self.assertEqual(
                (None, "explicit_new_chapter", "bound", "completed"), tuple(attempt))
            runtime.close()

    def test_initial_create_intent_is_durable_before_remote_post(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            provider = OpenAIAgentsProvider("test-key", "model")
            observed = []

            def fake_request(method, path, body=None, **_):
                row = store.connection.execute("""
                    SELECT old_session_id,creation_state,create_request_json,
                           protocol_descriptor_json,mutable_settings_json
                    FROM session_rollovers WHERE status='pending'
                """).fetchone()
                observed.append((method, path, body, dict(row) if row else None))
                return {"id": "session-initial", "status": "idle"}

            provider._request = fake_request
            runtime = ResidentRuntime(
                Config(Path(temporary)), provider, store=store, capabilities=[],
                owner_output=lambda _: None, diagnostic_output=lambda _: None)
            provider._ensure_session([], initial_input="bootstrap-identity")

            _, _, posted, durable = observed[0]
            self.assertIsNone(durable["old_session_id"])
            self.assertEqual("create_uncertain", durable["creation_state"])
            self.assertEqual(posted, json.loads(durable["create_request_json"]))
            self.assertIsNotNone(durable["protocol_descriptor_json"])
            self.assertIsNotNone(durable["mutable_settings_json"])
            runtime.close()

    def test_initial_create_failure_before_post_preserves_exact_request_for_restart(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "resident.sqlite3"
            store = Store(path)
            seed = OpenAIAgentsProvider(
                "test-key", "model-a", reasoning_effort="low")

            def fail_before_post(_):
                raise sqlite3.OperationalError("simulated pre-POST failure")

            store.mark_session_rollover_create_started = fail_before_post
            seed._request = lambda *_args, **_kwargs: self.fail("POST must not run")
            runtime = ResidentRuntime(
                Config(Path(temporary), new_chapter=True), seed, store=store,
                capabilities=[], owner_output=lambda _: None,
                diagnostic_output=lambda _: None)
            with self.assertRaisesRegex(sqlite3.OperationalError, "pre-POST"):
                seed._ensure_session([], initial_input="bootstrap-a")
            pending = store.pending_session_rollover("openai_agents")
            self.assertEqual("not_attempted", pending["creation_state"])
            self.assertEqual("bootstrap-a", pending["create_request"]["input"])
            runtime.close()

            reopened = Store(path)
            restarted = OpenAIAgentsProvider(
                "test-key", "model-b", reasoning_effort="high")
            requests = []

            def fake_request(method, request_path, body=None, **_):
                requests.append((method, request_path, body))
                if method == "POST":
                    return {"id": "session-initial", "status": "idle"}
                if method == "GET":
                    return {"id": "session-initial", "status": "idle", "agent": {}}
                if method == "PATCH":
                    return {"id": "session-initial", "status": "idle"}
                raise AssertionError((method, request_path, body))

            restarted._request = fake_request
            resumed_runtime = ResidentRuntime(
                Config(Path(temporary), new_chapter=True), restarted, store=reopened,
                capabilities=[], owner_output=lambda _: None,
                diagnostic_output=lambda _: None)
            restarted._ensure_session([], initial_input="bootstrap-b")
            self.assertEqual("bootstrap-a", requests[0][2]["input"])
            self.assertEqual("model-a", requests[0][2]["agent"]["model"])
            self.assertIsNone(restarted._requested_rollover_reason)

            restarted._ensure_session([], initial_input="following wake")
            self.assertEqual({"agent": {
                "model": "model-b", "reasoning": {"effort": "high"},
            }}, requests[-1][2])
            self.assertEqual(1, sum(method == "POST" for method, _, _ in requests))
            resumed_runtime.close()

    def test_definitive_initial_create_rejection_is_retryable_and_keeps_new_chapter(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            provider = OpenAIAgentsProvider("test-key", "model")
            attempts = 0

            def fake_request(method, path, body=None, **_):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise RuntimeError("OpenAI Agents API returned HTTP 400: rejected")
                return {"id": "session-retried", "status": "idle"}

            provider._request = fake_request
            runtime = ResidentRuntime(
                Config(Path(temporary), new_chapter=True), provider, store=store,
                capabilities=[], owner_output=lambda _: None,
                diagnostic_output=lambda _: None)
            with self.assertRaisesRegex(RuntimeError, "HTTP 400"):
                provider._ensure_session([], initial_input="bootstrap")
            failed = store.connection.execute("""
                SELECT creation_state,status FROM session_rollovers
            """).fetchone()
            self.assertEqual(("rejected", "failed"), tuple(failed))
            self.assertEqual("explicit_new_chapter", provider._requested_rollover_reason)

            provider._ensure_session([], initial_input="bootstrap")
            self.assertEqual(2, attempts)
            self.assertIsNone(provider._requested_rollover_reason)
            runtime.close()

    def test_uncertain_initial_create_blocks_restart_and_does_not_duplicate_bootstrap(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "resident.sqlite3"
            store = Store(path)
            original_bind = store.bind_session_rollover

            def fail_binding(*_):
                raise sqlite3.OperationalError("simulated post-POST crash")

            store.bind_session_rollover = fail_binding
            provider = OpenAIAgentsProvider("test-key", "model")
            posted = []

            def create(method, request_path, body=None, **_):
                posted.append(body)
                return {"id": "unknown-to-store", "status": "idle"}

            provider._request = create
            runtime = ResidentRuntime(
                Config(Path(temporary), new_chapter=True), provider, store=store,
                capabilities=[], owner_output=lambda _: None,
                diagnostic_output=lambda _: None)
            with self.assertRaisesRegex(sqlite3.OperationalError, "post-POST"):
                provider._ensure_session([], initial_input="one bootstrap")
            self.assertEqual(1, len(posted))
            self.assertEqual("create_uncertain", store.pending_session_rollover(
                "openai_agents")["creation_state"])
            store.bind_session_rollover = original_bind
            runtime.close()

            reopened = Store(path)
            restarted = OpenAIAgentsProvider("test-key", "model")
            restarted._request = lambda *_args, **_kwargs: self.fail(
                "uncertain create must exclude another POST")
            resumed_runtime = ResidentRuntime(
                Config(Path(temporary), new_chapter=True), restarted, store=reopened,
                capabilities=[], owner_output=lambda _: None,
                diagnostic_output=lambda _: None)
            with self.assertRaisesRegex(RolloverRecoveryRequired,
                                       "Initial session creation may have succeeded"):
                restarted._ensure_session([], initial_input="different bootstrap")
            durable = reopened.pending_session_rollover("openai_agents")
            self.assertEqual("one bootstrap", durable["create_request"]["input"])
            self.assertEqual("explicit_new_chapter", durable["reason"])
            resumed_runtime.close()

    def test_recovered_initial_create_keeps_immutable_snapshot_then_rolls_over(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "resident.sqlite3"
            store = Store(path)
            seed = OpenAIAgentsProvider("test-key", "model")
            original_tool = ToolSpec(
                "original_contract", "Original contract", {"type": "object"})
            original_agent = seed._agent_config([original_tool])
            original_protocol = seed._agent_protocol(original_agent)
            attempt = store.begin_session_rollover(
                "openai_agents", None, "initial_session", "runtime", {
                    "environment": {"type": "none"}, "agent": original_agent,
                    "input": "bootstrap-a", "metadata": {"managed_by": "resident"},
                }, original_protocol, seed._desired_mutable_settings())
            store.close()

            reopened = Store(path)
            restarted = OpenAIAgentsProvider("test-key", "model")
            changed_tool = ToolSpec(
                "changed_contract", "Changed contract", {"type": "object"})
            creates = []

            def fake_request(method, request_path, body=None, **_):
                if method == "POST":
                    creates.append(body)
                    return {"id": f"session-{len(creates)}", "status": "idle"}
                if method == "GET":
                    session_id = request_path.rsplit("/", 1)[-1]
                    return {"id": session_id, "status": "idle", "agent": {}}
                raise AssertionError((method, request_path, body))

            restarted._request = fake_request
            runtime = ResidentRuntime(
                Config(Path(temporary)), restarted, store=reopened, capabilities=[],
                owner_output=lambda _: None, diagnostic_output=lambda _: None)
            restarted._ensure_session([changed_tool], initial_input="ignored-bootstrap-b")

            self.assertEqual("bootstrap-a", creates[0]["input"])
            self.assertEqual("original_contract", creates[0]["agent"]["tools"][0]["name"])
            self.assertEqual(original_protocol, reopened.session_protocol(
                "openai_agents", "session-1"))
            self.assertTrue(restarted.protocol_change_requires_rollover([changed_tool]))

            restarted._ensure_session([changed_tool], initial_input="bootstrap-b")
            self.assertEqual(2, len(creates))
            self.assertEqual("changed_contract", creates[1]["agent"]["tools"][0]["name"])
            self.assertNotEqual(attempt["id"], reopened.connection.execute("""
                SELECT id FROM session_rollovers ORDER BY created_at DESC LIMIT 1
            """).fetchone()[0])
            runtime.close()

    def test_legacy_binding_with_partial_remote_protocol_rolls_over_conservatively(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            store.save_agent_session_binding(
                "openai_agents", "legacy-session", None, None)
            provider = OpenAIAgentsProvider("test-key", "model")
            new_tool = ToolSpec("new_contract", "New contract", {"type": "object"})
            requests = []

            def fake_request(method, request_path, body=None, **_):
                requests.append((method, request_path, body))
                if method == "GET":
                    return {"id": "legacy-session", "status": "idle", "agent": {}}
                if method == "POST":
                    return {"id": "replacement-session", "status": "idle"}
                raise AssertionError((method, request_path, body))

            provider._request = fake_request
            runtime = ResidentRuntime(
                Config(Path(temporary)), provider, store=store, capabilities=[],
                owner_output=lambda _: None, diagnostic_output=lambda _: None)
            provider._ensure_session([new_tool], initial_input="safe replacement")

            self.assertEqual(["GET", "POST"], [method for method, _, _ in requests])
            self.assertIsNone(store.session_protocol("openai_agents", "legacy-session"))
            replacement = store.session_protocol(
                "openai_agents", "replacement-session")
            self.assertEqual("new_contract", replacement["tools"][0]["name"])
            runtime.close()

    def test_uncertain_rollover_blocks_recreate_on_repeated_restart(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "resident.sqlite3"
            store = Store(path)
            seed = OpenAIAgentsProvider("test-key", "model")
            store.save_agent_session_binding("openai_agents", "session-old", None, None)
            store.save_session_protocol(
                "openai_agents", "session-old",
                seed._agent_protocol(seed._agent_config([])))
            rollover = store.begin_session_rollover(
                "openai_agents", "session-old", "explicit_new_chapter", "runtime", {
                    "environment": {"type": "none"}, "agent": seed._agent_config([]),
                    "input": "durable bootstrap", "metadata": {"managed_by": "resident"},
                }, seed._agent_protocol(seed._agent_config([])),
                seed._desired_mutable_settings())
            store.mark_session_rollover_create_started(rollover["id"])
            store.close()

            creates = []
            for _ in range(2):
                reopened = Store(path)
                provider = OpenAIAgentsProvider("test-key", "model")

                def fake_request(method, request_path, body=None, **_):
                    if method == "GET":
                        return {"id": "session-old", "status": "idle",
                                "agent": provider._agent_config([])}
                    if method == "POST":
                        creates.append(body)
                        return {"id": "should-not-exist", "status": "idle"}
                    raise AssertionError((method, request_path, body))

                provider._request = fake_request
                runtime = ResidentRuntime(
                    Config(Path(temporary)), provider, store=reopened, capabilities=[],
                    owner_output=lambda _: None, diagnostic_output=lambda _: None)
                with self.assertRaisesRegex(
                        RolloverRecoveryRequired, "automatic re-creation is blocked"):
                    provider._ensure_session([], initial_input="new bootstrap")
                runtime.close()
            self.assertEqual([], creates)

    def test_uncertain_rollover_excludes_competing_reasons_but_not_other_scopes(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            original = OpenAIAgentsProvider("test-key", "model-a")
            original_request = {
                "environment": {"type": "none"},
                "agent": original._agent_config([]),
                "input": "original bootstrap",
                "metadata": {"managed_by": "resident"},
            }
            first = store.begin_session_rollover(
                "openai_agents", "session-old", "reason-a", "runtime",
                original_request,
                original._agent_protocol(original._agent_config([])),
                original._desired_mutable_settings())
            store.mark_session_rollover_create_started(first["id"])

            newer = OpenAIAgentsProvider(
                "test-key", "model-b", reasoning_effort="high", agent_id="agent-b")
            newer_request = {
                "environment": {"type": "none"},
                "agent": newer._agent_config([]), "agent_id": "agent-b",
                "input": "new bootstrap", "metadata": {"managed_by": "resident"},
            }
            for reason in (
                    "explicit_new_chapter", "saved_agent_id_changed",
                    "function_or_immutable_protocol_changed", "remote_session_missing"):
                recovered = store.begin_session_rollover(
                    "openai_agents", "session-old", reason, "runtime", newer_request,
                    newer._agent_protocol(newer._agent_config([])),
                    newer._desired_mutable_settings())
                self.assertEqual(first["id"], recovered["id"])
                self.assertEqual("reason-a", recovered["reason"])
                self.assertEqual("original bootstrap", recovered["create_request"]["input"])
                self.assertEqual(first["create_token"], recovered["create_token"])
                self.assertEqual(first["protocol_descriptor"],
                                 recovered["protocol_descriptor"])
                self.assertEqual(first["mutable_settings"], recovered["mutable_settings"])

            for provider_name, old_session in (
                    ("openai_agents", "unrelated-session"),
                    ("another_provider", "session-old")):
                unrelated = store.begin_session_rollover(
                    provider_name, old_session, "reason-b", "runtime", newer_request,
                    newer._agent_protocol(newer._agent_config([])),
                    newer._desired_mutable_settings())
                self.assertNotEqual(first["id"], unrelated["id"])
            self.assertEqual(3, store.connection.execute(
                "SELECT COUNT(*) FROM session_rollovers").fetchone()[0])
            store.close()

    def test_rollover_triggers_cannot_bypass_uncertain_rollover(self):
        for trigger in ("new_chapter", "saved_agent", "remote_404", "remote_failed"):
            with self.subTest(trigger=trigger), tempfile.TemporaryDirectory() as temporary:
                store = Store(Path(temporary) / "resident.sqlite3")
                seed = OpenAIAgentsProvider(
                    "test-key", "model", agent_id="agent-a" if trigger == "saved_agent" else None)
                descriptor = seed._agent_protocol(seed._agent_config([]))
                store.save_agent_session_binding(
                    "openai_agents", "session-old", seed.agent_id, None)
                store.save_session_protocol("openai_agents", "session-old", descriptor)
                rollover = store.begin_session_rollover(
                    "openai_agents", "session-old", "reason-a", "runtime", {
                        "environment": {"type": "none"},
                        "agent": seed._agent_config([]), "agent_id": seed.agent_id,
                        "input": "original bootstrap",
                        "metadata": {"managed_by": "resident"},
                    }, descriptor, seed._desired_mutable_settings())
                store.mark_session_rollover_create_started(rollover["id"])
                provider = OpenAIAgentsProvider(
                    "test-key", "model", agent_id="agent-b" if trigger == "saved_agent" else None)
                requests = []

                def fake_request(method, request_path, body=None, **_):
                    requests.append((method, request_path, body))
                    if method == "GET":
                        return {"id": "session-old", "status": "idle",
                                "agent": ({"id": "agent-a"} if trigger == "saved_agent"
                                          else provider._agent_config([]))}
                    raise AssertionError("uncertain rollover must exclude another POST")

                provider._request = fake_request
                runtime = ResidentRuntime(
                    Config(Path(temporary), new_chapter=trigger == "new_chapter"),
                    provider, store=store, capabilities=[], owner_output=lambda _: None,
                    diagnostic_output=lambda _: None)
                if trigger == "remote_404":
                    provider._unavailable_session_id = "session-old"
                    provider._unavailable_session_reason = "remote_session_missing"
                elif trigger == "remote_failed":
                    provider._unavailable_session_id = "session-old"
                    provider._unavailable_session_reason = "remote_session_failed"
                with self.assertRaises(RolloverRecoveryRequired):
                    provider._ensure_session([], initial_input="new bootstrap")
                self.assertFalse(any(method == "POST" for method, _, _ in requests))
                authoritative = store.pending_session_rollover("openai_agents")
                self.assertEqual(rollover["id"], authoritative["id"])
                self.assertEqual("reason-a", authoritative["reason"])
                runtime.close()

    def test_resolved_uncertain_rollover_reconciles_new_mutable_and_chapter_requests(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "resident.sqlite3"
            store = Store(path)
            seed = OpenAIAgentsProvider("test-key", "model-a")
            descriptor = seed._agent_protocol(seed._agent_config([]))
            store.save_agent_session_binding("openai_agents", "session-old", None, None)
            rollover = store.begin_session_rollover(
                "openai_agents", "session-old", "reason-a", "runtime", {
                    "environment": {"type": "none"}, "agent": seed._agent_config([]),
                    "input": "original bootstrap", "metadata": {"managed_by": "resident"},
                }, descriptor, seed._desired_mutable_settings())
            store.mark_session_rollover_create_started(rollover["id"])
            store.bind_session_rollover(
                rollover["id"], "session-recovered", None, "operator_reconciled")
            store.complete_session_rollover(rollover["id"])
            store.close()

            reopened = Store(path)
            provider = OpenAIAgentsProvider("test-key", "model-b")
            requests = []

            def fake_request(method, request_path, body=None, **_):
                requests.append((method, request_path, body))
                if method == "GET":
                    return {"id": "session-recovered", "status": "idle", "agent": {}}
                if method == "PATCH":
                    return {"id": "session-recovered", "status": "idle"}
                if method == "POST":
                    return {"id": "session-chapter", "status": "idle"}
                raise AssertionError((method, request_path, body))

            provider._request = fake_request
            runtime = ResidentRuntime(
                Config(Path(temporary)), provider, store=reopened, capabilities=[],
                owner_output=lambda _: None, diagnostic_output=lambda _: None)
            provider._ensure_session([], initial_input="ordinary wake")
            self.assertEqual({"agent": {"model": "model-b"}},
                             next(body for method, _, body in requests if method == "PATCH"))

            provider.request_rollover("explicit_new_chapter")
            provider._ensure_session([], initial_input="chapter bootstrap")
            self.assertEqual(1, sum(method == "POST" for method, _, _ in requests))
            self.assertEqual("session-chapter", provider.session_id)
            runtime.close()

    def test_rollover_create_success_with_binding_failure_stays_uncertain(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "resident.sqlite3"
            store = Store(path)
            provider = OpenAIAgentsProvider("test-key", "model")
            store.save_agent_session_binding("openai_agents", "session-old", None, None)
            descriptor = provider._agent_protocol(provider._agent_config([]))
            store.save_session_protocol("openai_agents", "session-old", descriptor)
            original_bind = store.bind_session_rollover

            def fail_binding(*_):
                raise sqlite3.OperationalError("simulated binding crash")

            store.bind_session_rollover = fail_binding
            provider._request = lambda method, path, body=None, **_: (
                {"id": "session-new", "status": "idle"} if method == "POST"
                else {"id": "session-old", "status": "idle",
                      "agent": provider._agent_config([])})
            runtime = ResidentRuntime(
                Config(Path(temporary), new_chapter=True), provider, store=store,
                capabilities=[], owner_output=lambda _: None,
                diagnostic_output=lambda _: None)

            with self.assertRaisesRegex(sqlite3.OperationalError, "simulated binding crash"):
                provider._ensure_session([], initial_input="bootstrap")

            pending = store.pending_session_rollover("openai_agents")
            self.assertEqual("create_uncertain", pending["creation_state"])
            self.assertEqual(
                "session-old", store.agent_session_binding("openai_agents")["session_id"])
            store.bind_session_rollover = original_bind
            runtime.close()

    def test_restart_completes_durably_bound_rollover_without_new_create(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "resident.sqlite3"
            store = Store(path)
            provider = OpenAIAgentsProvider("test-key", "model")
            descriptor = provider._agent_protocol(provider._agent_config([]))
            store.save_agent_session_binding("openai_agents", "session-old", None, None)
            handover_id = store.create_handover(
                "session-old", "durable handover",
                (datetime.now(UTC) + timedelta(hours=1)).isoformat())
            rollover = store.begin_session_rollover(
                "openai_agents", "session-old", "change", "runtime", {
                    "environment": {"type": "none"}, "agent": provider._agent_config([]),
                    "input": json.dumps({"new_session_bootstrap": {
                        "handover": "durable handover"}}),
                    "metadata": {"managed_by": "resident"},
                }, descriptor, provider._desired_mutable_settings())
            store.mark_session_rollover_create_started(rollover["id"])
            store.bind_session_rollover(
                rollover["id"], "session-new", None, "completed")
            store.close()

            reopened = Store(path)
            restarted_provider = OpenAIAgentsProvider("test-key", "model")
            creates = []

            def fake_request(method, request_path, body=None, **_):
                if method == "GET":
                    return {"id": "session-new", "status": "idle",
                            "agent": restarted_provider._agent_config([])}
                creates.append(body)
                raise AssertionError("replacement must not be created again")

            restarted_provider._request = fake_request
            runtime = ResidentRuntime(
                Config(Path(temporary)), restarted_provider, store=reopened,
                capabilities=[], owner_output=lambda _: None,
                diagnostic_output=lambda _: None)
            self.assertEqual("session-new", restarted_provider.session_id)
            row = reopened.connection.execute(
                "SELECT creation_state,status FROM session_rollovers WHERE id=?",
                (rollover["id"],)).fetchone()
            self.assertEqual(("bound", "completed"), tuple(row))
            consumed = reopened.connection.execute(
                "SELECT new_session_id,consumed_at FROM session_handovers WHERE id=?",
                (handover_id,)).fetchone()
            self.assertEqual("session-new", consumed["new_session_id"])
            self.assertIsNotNone(consumed["consumed_at"])
            restarted_provider._ensure_session([], initial_input="wake")
            self.assertEqual([], creates)
            runtime.close()

    def test_unattempted_rollover_restarts_with_exact_durable_create_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "resident.sqlite3"
            store = Store(path)
            seed = OpenAIAgentsProvider(
                "test-key", "model-a", reasoning_effort="low", service_tier="flex")
            descriptor = seed._agent_protocol(seed._agent_config([]))
            store.save_agent_session_binding("openai_agents", "session-old", None, None)
            store.save_session_protocol("openai_agents", "session-old", descriptor)
            rollover = store.begin_session_rollover(
                "openai_agents", "session-old", "explicit_new_chapter", "runtime", {
                    "environment": {"type": "none"}, "agent": seed._agent_config([]),
                    "input": "bootstrap-before-crash",
                    "metadata": {"managed_by": "resident"},
                }, descriptor, seed._desired_mutable_settings())
            self.assertEqual("not_attempted", rollover["creation_state"])
            store.close()

            reopened = Store(path)
            provider = OpenAIAgentsProvider(
                "test-key", "model-b", reasoning_effort="high", service_tier="priority")
            creates = []
            patches = []

            def fake_request(method, request_path, body=None, **_):
                if method == "GET":
                    session_id = request_path.rsplit("/", 1)[-1]
                    return {"id": session_id, "status": "idle", "agent": {}}
                if method == "POST":
                    creates.append(body)
                    return {"id": "session-new", "status": "idle"}
                if method == "PATCH":
                    patches.append(body)
                    return {"id": "session-new", "status": "idle"}
                raise AssertionError((method, request_path, body))

            provider._request = fake_request
            runtime = ResidentRuntime(
                Config(Path(temporary)), provider, store=reopened, capabilities=[],
                owner_output=lambda _: None, diagnostic_output=lambda _: None)
            provider._ensure_session([], initial_input="different-after-restart")

            self.assertEqual(1, len(creates))
            self.assertEqual("bootstrap-before-crash", creates[0]["input"])
            self.assertEqual("model-a", creates[0]["agent"]["model"])
            self.assertEqual(rollover["create_token"],
                             creates[0]["metadata"]["rollover_token"])
            self.assertEqual(descriptor, reopened.session_protocol(
                "openai_agents", "session-new"))
            self.assertEqual(seed._desired_mutable_settings(),
                             reopened.session_mutable_settings(
                                 "openai_agents", "session-new"))

            provider._ensure_session([], initial_input="reconcile-current-config")

            self.assertEqual([{"agent": provider._desired_mutable_settings()}], patches)
            self.assertEqual(provider._desired_mutable_settings(),
                             reopened.session_mutable_settings(
                                 "openai_agents", "session-new"))
            runtime.close()

    def test_recovered_rollover_keeps_immutable_snapshot_then_rolls_again(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "resident.sqlite3"
            store = Store(path)
            seed = OpenAIAgentsProvider("test-key", "model", agent_id="agent-a")
            old_descriptor = seed._agent_protocol(seed._agent_config([]))
            store.save_agent_session_binding(
                "openai_agents", "session-old", "agent-a", None)
            store.save_session_protocol("openai_agents", "session-old", old_descriptor)
            first = store.begin_session_rollover(
                "openai_agents", "session-old", "explicit_new_chapter", "runtime", {
                    "environment": {"type": "none"}, "agent": seed._agent_config([]),
                    "agent_id": "agent-a",
                    "input": "bootstrap-a", "metadata": {"managed_by": "resident"},
                }, old_descriptor, seed._desired_mutable_settings())
            store.close()

            reopened = Store(path)
            provider = OpenAIAgentsProvider("test-key", "model", agent_id="agent-b")
            new_tool = ToolSpec("new_contract", "New contract", {"type": "object"})
            creates = []

            def fake_request(method, request_path, body=None, **_):
                if method == "GET":
                    session_id = request_path.rsplit("/", 1)[-1]
                    return {"id": session_id, "status": "idle", "agent": {}}
                if method == "POST":
                    creates.append(body)
                    return {"id": f"session-new-{len(creates)}", "status": "idle"}
                raise AssertionError((method, request_path, body))

            provider._request = fake_request
            runtime = ResidentRuntime(
                Config(Path(temporary)), provider, store=reopened, capabilities=[],
                owner_output=lambda _: None, diagnostic_output=lambda _: None)
            provider._ensure_session([new_tool], initial_input="current-b")

            self.assertEqual("bootstrap-a", creates[0]["input"])
            self.assertEqual([], creates[0]["agent"]["tools"])
            self.assertEqual(old_descriptor, reopened.session_protocol(
                "openai_agents", "session-new-1"))
            self.assertEqual("agent-a", reopened.agent_session_binding(
                "openai_agents")["agent_id"])
            self.assertTrue(provider.protocol_change_requires_rollover([new_tool]))

            provider._ensure_session([new_tool], initial_input="bootstrap-b")

            self.assertEqual(2, len(creates))
            self.assertEqual("agent-b", creates[1]["agent_id"])
            self.assertEqual("new_contract", creates[1]["agent"]["tools"][0]["name"])
            self.assertNotEqual(first["id"], reopened.connection.execute(
                "SELECT id FROM session_rollovers ORDER BY created_at DESC LIMIT 1"
            ).fetchone()[0])
            runtime.close()

    def test_recovered_rollover_with_unchanged_config_needs_no_reconciliation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "resident.sqlite3"
            store = Store(path)
            seed = OpenAIAgentsProvider("test-key", "model")
            descriptor = seed._agent_protocol(seed._agent_config([]))
            store.save_agent_session_binding("openai_agents", "session-old", None, None)
            store.save_session_protocol("openai_agents", "session-old", descriptor)
            store.begin_session_rollover(
                "openai_agents", "session-old", "explicit_new_chapter", "runtime", {
                    "environment": {"type": "none"}, "agent": seed._agent_config([]),
                    "input": "bootstrap-a", "metadata": {"managed_by": "resident"},
                }, descriptor, seed._desired_mutable_settings())
            store.close()

            reopened = Store(path)
            provider = OpenAIAgentsProvider("test-key", "model")
            requests = []

            def fake_request(method, request_path, body=None, **_):
                requests.append((method, request_path, body))
                if method == "GET":
                    session_id = request_path.rsplit("/", 1)[-1]
                    return {"id": session_id, "status": "idle",
                            "agent": provider._agent_config([])}
                if method == "POST":
                    return {"id": "session-new", "status": "idle"}
                raise AssertionError((method, request_path, body))

            provider._request = fake_request
            runtime = ResidentRuntime(
                Config(Path(temporary)), provider, store=reopened, capabilities=[],
                owner_output=lambda _: None, diagnostic_output=lambda _: None)
            provider._ensure_session([], initial_input="ignored-current-input")
            provider._ensure_session([], initial_input="ordinary-next-wake")

            self.assertEqual(1, sum(
                method == "POST" and request_path == "/agents/sessions"
                for method, request_path, _ in requests))
            self.assertFalse(any(method == "PATCH" for method, _, _ in requests))
            self.assertFalse(provider.protocol_change_requires_rollover([]))
            runtime.close()

    def test_definitive_rollover_create_rejection_is_audited_as_failed(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            provider = OpenAIAgentsProvider("test-key", "model")
            descriptor = provider._agent_protocol(provider._agent_config([]))
            store.save_agent_session_binding("openai_agents", "session-old", None, None)
            store.save_session_protocol("openai_agents", "session-old", descriptor)

            def fake_request(method, request_path, body=None, **_):
                if method == "GET":
                    return {"id": "session-old", "status": "idle",
                            "agent": provider._agent_config([])}
                raise RuntimeError("OpenAI Agents API returned HTTP 400: rejected")

            provider._request = fake_request
            runtime = ResidentRuntime(
                Config(Path(temporary), new_chapter=True), provider, store=store,
                capabilities=[], owner_output=lambda _: None,
                diagnostic_output=lambda _: None)
            with self.assertRaisesRegex(RuntimeError, "HTTP 400"):
                provider._ensure_session([], initial_input="bootstrap")
            row = store.connection.execute(
                "SELECT creation_state,finalization_status,status FROM session_rollovers"
            ).fetchone()
            self.assertEqual(("rejected", "create_rejected", "failed"), tuple(row))
            runtime.close()

    async def test_deferred_rollover_handover_survives_restart_and_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "resident.sqlite3"
            store = Store(path)
            seed = OpenAIAgentsProvider("test-key", "model", poll_seconds=0)
            store.save_agent_session_binding("openai_agents", "session-old", None, None)
            store.save_session_protocol(
                "openai_agents", "session-old",
                seed._agent_protocol(seed._agent_config([])))
            state = {"old_status": "in_progress", "items": {}}
            creates = []
            submitted_contexts = []

            def fake_request(method, request_path, body=None, **_):
                if request_path == "/agents/sessions/session-old" and method == "GET":
                    return {"id": "session-old", "status": state["old_status"],
                            "agent": seed._agent_config([])}
                if request_path == "/agents/sessions/session-new" and method == "GET":
                    return {"id": "session-new", "status": "idle",
                            "agent": seed._agent_config([])}
                if request_path == "/agents/sessions" and method == "POST":
                    creates.append(body)
                    state["items"]["session-new"] = [{
                        "id": "new-input", "type": "message", "role": "user",
                        "turn_id": "turn-new", "content": [{
                            "type": "input_text", "text": body["input"]}],
                    }]
                    return {"id": "session-new", "status": "idle"}
                if request_path.endswith("/events"):
                    session_id = request_path.split("/")[3]
                    event = body["events"][0]
                    if event["type"] == "agent.session.input.message":
                        submitted_contexts.append(
                            event["input"][0]["content"][0]["text"])
                    turn_id = "turn-old-wake" if session_id == "session-old" else "turn-new"
                    state["items"][session_id] = [{
                        "id": f"{session_id}-input", "type": "message", "role": "user",
                        "turn_id": turn_id, "content": event["input"][0]["content"],
                    }]
                    return {}
                if "/items?" in request_path:
                    session_id = request_path.split("/")[3]
                    return {"data": state["items"].get(session_id, [])}
                if "/turns?" in request_path:
                    return {"data": [{"id": "turn-new", "status": "completed"}]}
                if "/turns/" in request_path:
                    return {"id": request_path.rsplit("/", 1)[-1], "status": "completed"}
                raise AssertionError((method, request_path, body))

            first_provider = OpenAIAgentsProvider("test-key", "model", poll_seconds=0)
            first_provider._request = fake_request
            first_runtime = ResidentRuntime(
                Config(Path(temporary), new_chapter=True), first_provider, store=store,
                capabilities=[], owner_output=lambda _: None,
                diagnostic_output=lambda _: None)

            class Curator:
                def __init__(self):
                    self.calls = []

                async def catch_up(self, final=False):
                    self.calls.append(final)
                    if final:
                        raise AssertionError("handover must not finalize while old session is active")
                    return "draft after additional old-session activity"

            first_curator = Curator()
            first_runtime.bind_curator(first_curator)
            await first_runtime.process(WakeEvent(
                "wake-old", "owner", "message", utc_now(), {"message_id": "message-old"}))

            handover = store.pending_handover("session-old")
            self.assertIsNone(handover)
            self.assertEqual("session-old", first_provider.session_id)
            self.assertEqual(0, len(creates))
            self.assertEqual([False], first_curator.calls)
            self.assertNotIn("new_session_bootstrap", json.loads(submitted_contexts[0]))
            stale_handover_id = store.create_handover(
                "session-old", "stale handover prepared before deferred activity",
                (datetime.now(UTC) + timedelta(hours=1)).isoformat())
            first_runtime.close()

            state["old_status"] = "idle"
            reopened = Store(path)
            replacement = OpenAIAgentsProvider("test-key", "model", poll_seconds=0)
            replacement._request = fake_request
            restarted = ResidentRuntime(
                Config(Path(temporary), new_chapter=True), replacement, store=reopened,
                capabilities=[], owner_output=lambda _: None,
                diagnostic_output=lambda _: None)

            class FinalCurator:
                def __init__(self):
                    self.calls = []

                async def catch_up(self, final=False):
                    self.calls.append(final)
                    if final:
                        self.assert_old_activity_curated()
                    return ("final handover including additional old-session activity"
                            if final else None)

                @staticmethod
                def assert_old_activity_curated():
                    if not any(item["id"] == "session-old-input"
                               for item in state["items"].get("session-old", [])):
                        raise AssertionError("final catch-up missed deferred old-session activity")

            final_curator = FinalCurator()
            restarted.bind_curator(final_curator)
            await restarted.process(WakeEvent(
                "wake-new", "connector", "changed", utc_now(), {}))

            self.assertEqual("session-new", replacement.session_id)
            self.assertEqual(1, len(creates))
            self.assertEqual(
                "final handover including additional old-session activity",
                json.loads(creates[0]["input"])["new_session_bootstrap"]["handover"])
            self.assertTrue(final_curator.calls[0])
            handover = reopened.connection.execute(
                "SELECT id,new_session_id,consumed_at FROM session_handovers "
                "WHERE old_session_id='session-old'").fetchone()
            handover_id = handover["id"]
            self.assertEqual(stale_handover_id, handover_id)
            consumed = reopened.connection.execute(
                "SELECT new_session_id,consumed_at FROM session_handovers WHERE id=?",
                (handover_id,)).fetchone()
            self.assertEqual("session-new", consumed["new_session_id"])
            self.assertIsNotNone(consumed["consumed_at"])
            await restarted.process(WakeEvent(
                "wake-after", "connector", "changed", utc_now(), {}))
            self.assertEqual(1, len(creates))
            self.assertEqual(1, reopened.connection.execute(
                "SELECT COUNT(*) FROM session_handovers WHERE id=? AND consumed_at IS NOT NULL",
                (handover_id,)).fetchone()[0])
            restarted.close()

    async def test_remote_404_replacement_receives_degraded_new_session_bootstrap(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            provider = OpenAIAgentsProvider("test-key", "gpt-5.6-luna", poll_seconds=0)
            store.save_agent_session_binding(
                "openai_agents", "session-missing", None, "turn-old")
            descriptor = provider._agent_protocol(provider._agent_config([]))
            store.save_session_protocol("openai_agents", "session-missing", descriptor)
            store.set_owner_guidance("Always preserve the garden schedule.")
            store.create_intention("Finish checking the greenhouse")
            store.apply_curator_batch(
                "openai_agents", "session-missing", "item-1", "item-1", "memory-1",
                [{"memory_id": "memory-1", "operation": "create", "kind": "place",
                  "content": "The greenhouse has a north bed.",
                  "provenance": [{"item_id": "item-1"}]}])
            creates = []

            def fake_request(method, path, body=None, **_):
                if method == "GET" and path == "/agents/sessions/session-missing":
                    raise RuntimeError("OpenAI Agents API returned HTTP 404: gone")
                if method == "POST" and path == "/agents/sessions":
                    creates.append(body)
                    return {"id": "session-replacement", "status": "idle"}
                raise AssertionError((method, path, body))

            provider._request = fake_request
            provider._wait_for_submitted_wake = lambda *_: ModelTurn(
                "turn-new", "replacement ready")
            runtime = ResidentRuntime(
                Config(Path(temporary)), provider, store=store, capabilities=[],
                owner_output=lambda _: None, diagnostic_output=lambda _: None)
            resident_id = runtime.resident.id

            await runtime.process(WakeEvent(
                "wake-404", "scheduler", "due", utc_now(), {}))

            self.assertEqual(1, len(creates))
            created_context = json.loads(creates[0]["input"])
            bootstrap = created_context["new_session_bootstrap"]
            self.assertEqual("place", bootstrap["durable_memory_awareness"][0]["kind"])
            self.assertIn("previous remote session was unavailable", bootstrap["handover"])
            self.assertEqual(resident_id, created_context["resident"]["stable_id"])
            self.assertEqual(
                "Always preserve the garden schedule.",
                created_context["standing_owner_guidance"][0]["content"])
            self.assertEqual(
                "Finish checking the greenhouse",
                created_context["pending_intentions"][0]["content"])
            rollover = store.connection.execute(
                "SELECT old_session_id,new_session_id,reason,finalization_status,status "
                "FROM session_rollovers").fetchone()
            self.assertEqual(
                ("session-missing", "session-replacement", "remote_session_missing",
                 "unavailable", "completed"), tuple(rollover))
            runtime.close()

    async def test_restored_session_statuses_are_classified_explicitly(self):
        for status, expected in (("idle", None), ("in_progress", None),
                                 ("requires_action", None)):
            with self.subTest(status=status):
                provider = OpenAIAgentsProvider("test-key", "model")
                provider._session_id = "session-old"
                provider._request = lambda *_args, **_kwargs: {
                    "id": "session-old", "status": status}
                self.assertEqual(expected, await provider.preflight_session())
                self.assertEqual(status == "idle", provider.rollover_ready)
                self.assertIsNone(provider._requested_rollover_reason)

        provider = OpenAIAgentsProvider("test-key", "model")
        provider._session_id = "session-old"
        provider._request = lambda *_args, **_kwargs: {
            "id": "session-old", "status": "future_state"}
        with self.assertRaisesRegex(RuntimeError, "unsupported status 'future_state'"):
            await provider.preflight_session()
        self.assertIsNone(provider._requested_rollover_reason)

    async def test_terminal_restored_session_rolls_over_with_final_handover(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            provider = OpenAIAgentsProvider("test-key", "model", poll_seconds=0)
            store.save_agent_session_binding(
                "openai_agents", "session-failed", None, "turn-old")
            store.save_session_protocol(
                "openai_agents", "session-failed",
                provider._agent_protocol(provider._agent_config([])))
            creates = []

            def fake_request(method, path, body=None, **_):
                if method == "GET" and path == "/agents/sessions/session-failed":
                    return {"id": "session-failed", "status": "failed",
                            "error": {"message": "terminal remote failure"}}
                if method == "POST" and path == "/agents/sessions":
                    creates.append(body)
                    return {"id": "session-replacement", "status": "idle"}
                raise AssertionError((method, path, body))

            class FinalCurator:
                def __init__(self):
                    self.calls = []

                async def catch_up(self, final=False):
                    self.calls.append(final)
                    return "authoritative final handover" if final else None

            provider._request = fake_request
            provider._wait_for_submitted_wake = lambda *_: ModelTurn("turn-new", "ready")
            runtime = ResidentRuntime(
                Config(Path(temporary)), provider, store=store, capabilities=[],
                owner_output=lambda _: None, diagnostic_output=lambda _: None)
            curator = FinalCurator()
            runtime.curator = curator

            await runtime.process(WakeEvent(
                "wake-terminal", "scheduler", "due", utc_now(), {}))

            self.assertEqual([True, False], curator.calls)
            self.assertEqual(1, len(creates))
            bootstrap = json.loads(creates[0]["input"])["new_session_bootstrap"]
            self.assertEqual("authoritative final handover", bootstrap["handover"])
            rollover = store.connection.execute(
                "SELECT old_session_id,new_session_id,reason,finalization_status,status "
                "FROM session_rollovers").fetchone()
            self.assertEqual(
                ("session-failed", "session-replacement", "remote_session_failed",
                 "unavailable", "completed"), tuple(rollover))
            history = store.connection.execute(
                "SELECT content,new_session_id,consumed_at FROM session_handovers"
            ).fetchone()
            self.assertEqual("authoritative final handover", history["content"])
            self.assertEqual("session-replacement", history["new_session_id"])
            self.assertIsNotNone(history["consumed_at"])
            runtime.close()

    async def test_terminal_session_with_unavailable_source_uses_degraded_handover(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            provider = OpenAIAgentsProvider("test-key", "model", poll_seconds=0)
            store.save_agent_session_binding(
                "openai_agents", "session-failed", None, "turn-old")
            store.save_session_protocol(
                "openai_agents", "session-failed",
                provider._agent_protocol(provider._agent_config([])))
            creates = []

            def fake_request(method, path, body=None, **_):
                if method == "GET" and path == "/agents/sessions/session-failed":
                    return {"id": "session-failed", "status": "failed"}
                if method == "POST" and path == "/agents/sessions":
                    creates.append(body)
                    return {"id": "session-replacement", "status": "idle"}
                raise AssertionError((method, path, body))

            class UnavailableCurator:
                async def catch_up(self, final=False):
                    if final:
                        raise RuntimeError("old session items unavailable")
                    return None

            provider._request = fake_request
            provider._wait_for_submitted_wake = lambda *_: ModelTurn("turn-new", "ready")
            runtime = ResidentRuntime(
                Config(Path(temporary)), provider, store=store, capabilities=[],
                owner_output=lambda _: None, diagnostic_output=lambda _: None)
            runtime.curator = UnavailableCurator()

            await runtime.process(WakeEvent(
                "wake-terminal-degraded", "scheduler", "due", utc_now(), {}))

            bootstrap = json.loads(creates[0]["input"])["new_session_bootstrap"]
            self.assertIn("previous remote session was unavailable", bootstrap["handover"])
            self.assertEqual(1, len(creates))
            runtime.close()

    def test_agents_missing_session_ignores_legacy_agent_id_and_saves_replacement(self):
        provider = OpenAIAgentsProvider("test-key", "gpt-5.6-luna")
        binding = {
            "session_id": "session-missing",
            "agent_id": "agent-persisted",
            "last_turn_id": "turn-old",
        }
        provider.bind_session_store(
            lambda: dict(binding),
            lambda session_id, agent_id, last_turn_id: binding.update(
                session_id=session_id, agent_id=agent_id, last_turn_id=last_turn_id),
        )
        requests = []

        def fake_request(method, path, body=None, **_):
            requests.append((method, path, body))
            if path == "/agents/sessions/session-missing" and method == "GET":
                raise RuntimeError("OpenAI Agents API returned HTTP 404: gone")
            if path == "/agents/sessions" and method == "POST":
                return {"id": "session-replacement", "status": "idle"}
            raise AssertionError((method, path, body))

        provider._request = fake_request

        with self.assertRaises(RemoteSessionUnavailable):
            provider._ensure_session([], initial_input="ordinary wake")
        session, created = provider._ensure_session([], initial_input="replacement bootstrap")

        self.assertEqual("session-replacement", session["id"])
        self.assertTrue(created)
        create_body = requests[-1][2]
        self.assertNotIn("agent_id", create_body)
        self.assertEqual("gpt-5.6-luna", create_body["agent"]["model"])
        self.assertEqual("replacement bootstrap", create_body["input"])
        self.assertNotIn("name", create_body["agent"])
        self.assertEqual({
            "session_id": "session-replacement",
            "agent_id": None,
            "last_turn_id": None,
        }, binding)

    def test_agents_explicit_agent_override_replaces_conflicting_persisted_binding(self):
        provider = OpenAIAgentsProvider(
            "test-key", "gpt-5.6-luna", agent_id="agent-configured")
        binding = {
            "session_id": "session-stale",
            "agent_id": "agent-persisted",
            "last_turn_id": "turn-stale",
        }
        provider.bind_session_store(
            lambda: dict(binding),
            lambda session_id, agent_id, last_turn_id: binding.update(
                session_id=session_id, agent_id=agent_id, last_turn_id=last_turn_id),
        )
        requests = []

        def fake_request(method, path, body=None, **_):
            requests.append((method, path, body))
            if path == "/agents/sessions" and method == "POST":
                return {"id": "session-configured", "status": "idle",
                        "agent": {"id": "agent-configured"}}
            raise AssertionError((method, path, body))

        provider._request = fake_request

        self.assertIsNone(provider._session_id)
        session, created = provider._ensure_session([], initial_input="override wake")

        self.assertEqual("session-configured", session["id"])
        self.assertTrue(created)
        self.assertEqual([("POST", "/agents/sessions")], [
            (method, path) for method, path, _ in requests])
        self.assertEqual("agent-configured", requests[0][2]["agent_id"])
        self.assertEqual({
            "session_id": "session-configured",
            "agent_id": "agent-configured",
            "last_turn_id": None,
        }, binding)

    async def test_agents_binding_failure_is_propagated_and_retried_before_reuse(self):
        provider = OpenAIAgentsProvider("test-key", "gpt-5.6-luna", poll_seconds=0)
        owner_thread_id = threading.get_ident()
        save_calls = []
        create_calls = []
        state = {"items": [], "turn_created": False}

        def save(session_id, agent_id, last_turn_id):
            save_calls.append((threading.get_ident(), session_id, agent_id, last_turn_id))
            if len(save_calls) == 1:
                raise sqlite3.OperationalError("simulated persistence failure")

        provider.bind_session_store(lambda: None, save)

        def fake_request(method, request_path, body=None, **_):
            if request_path == "/agents/sessions" and method == "POST":
                create_calls.append(request_path)
                state["turn_created"] = True
                state["items"] = [{
                    "id": "input-1", "type": "message", "role": "user",
                    "turn_id": "turn-1", "content": [
                        {"type": "input_text", "text": body["input"]}],
                }]
                return {"id": "session-1", "status": "idle", "agent": {"id": "agent-1"}}
            if request_path == "/agents/sessions/session-1" and method == "GET":
                return {"id": "session-1", "status": "idle", "agent": {"id": "agent-1"}}
            if request_path.endswith("/events"):
                event = body["events"][0]
                state["turn_created"] = True
                state["items"] = [{
                    "id": "input-1", "type": "message", "role": "user",
                    "turn_id": "turn-1", "content": event["input"][0]["content"],
                }]
                return {}
            if "/turns?" in request_path:
                return {"data": []}
            if request_path.endswith("/turns/turn-1"):
                return {"id": "turn-1", "status": "completed"}
            if "/items?" in request_path:
                return {"data": state["items"]}
            raise AssertionError((method, request_path, body))

        provider._request = fake_request
        context = json.dumps({
            "wake_event": {"id": "wake-1", "source": "connector", "payload": {}}})

        with self.assertRaisesRegex(sqlite3.OperationalError, "simulated persistence failure"):
            await provider.respond(context, [], [])
        self.assertTrue(state["turn_created"])

        turn = await provider.respond(context, [], [])

        self.assertEqual("turn-1", turn.response_id)
        self.assertEqual(1, len(create_calls))
        self.assertEqual([owner_thread_id] * 3, [call[0] for call in save_calls])
        self.assertEqual(
            [("session-1", None, None), ("session-1", None, None),
             ("session-1", None, "turn-1")],
            [call[1:] for call in save_calls])

    def test_agents_configuration_change_is_deferred_until_session_is_idle(self):
        provider = OpenAIAgentsProvider("test-key", "gpt-5.6-luna")
        status = {"value": "idle"}
        creates = []

        def fake_request(method, path, body=None, **_):
            if path == "/agents/sessions" and method == "POST":
                creates.append(body)
                return {"id": f"session-{len(creates)}", "status": "idle"}
            if path.startswith("/agents/sessions/session-") and method == "GET":
                return {"id": path.rsplit("/", 1)[-1], "status": status["value"]}
            raise AssertionError((method, path, body))

        provider._request = fake_request
        original = ToolSpec("clock", "Read clock", {"type": "object"})
        changed = ToolSpec("clock", "Read the local clock", {"type": "object"})

        provider._ensure_session([original], initial_input="original wake")
        applied_fingerprint = provider._tool_fingerprint
        status["value"] = "in_progress"
        provider._ensure_session([changed], initial_input="changed wake")

        self.assertEqual(applied_fingerprint, provider._tool_fingerprint)
        self.assertEqual(1, len(creates))

        status["value"] = "idle"
        provider._ensure_session([changed], initial_input="changed wake")
        changed_fingerprint = json.dumps(
            provider._agent_config([changed]), sort_keys=True, separators=(",", ":"))
        self.assertEqual(changed_fingerprint, provider._tool_fingerprint)
        self.assertEqual(
            "Read the local clock", creates[1]["agent"]["tools"][0]["description"])

        provider._ensure_session([changed], initial_input="changed wake")
        self.assertEqual(2, len(creates))

    def test_agents_configuration_fingerprint_advances_only_after_successful_replacement(self):
        provider = OpenAIAgentsProvider("test-key", "gpt-5.6-luna")
        fail_replacement = {"value": True}
        creates = []

        def fake_request(method, path, body=None, **_):
            if path == "/agents/sessions" and method == "POST":
                creates.append(body)
                if len(creates) > 1 and fail_replacement["value"]:
                    raise RuntimeError("configuration replacement failed")
                return {"id": f"session-{len(creates)}", "status": "idle"}
            if path == "/agents/sessions/session-1" and method == "GET":
                return {"id": "session-1", "status": "idle"}
            raise AssertionError((method, path, body))

        provider._request = fake_request
        original = ToolSpec("clock", "Read clock", {"type": "object"})
        changed = ToolSpec("clock", "Read the local clock", {"type": "object"})
        provider._ensure_session([original], initial_input="original wake")
        applied_fingerprint = provider._tool_fingerprint

        with self.assertRaisesRegex(RuntimeError, "configuration replacement failed"):
            provider._ensure_session([changed], initial_input="changed wake")
        self.assertEqual(applied_fingerprint, provider._tool_fingerprint)

        fail_replacement["value"] = False
        provider._ensure_session([changed], initial_input="changed wake")
        self.assertNotEqual(applied_fingerprint, provider._tool_fingerprint)
        self.assertEqual(3, len(creates))

    async def test_agents_session_is_persisted_and_reused_for_tool_continuation(self):
        provider = OpenAIAgentsProvider("test-key", "gpt-5.6-luna", poll_seconds=0)
        binding = {}
        provider.bind_session_store(
            lambda: binding or None,
            lambda session_id, agent_id, last_turn_id: binding.update(
                session_id=session_id, agent_id=agent_id, last_turn_id=last_turn_id),
        )
        requests = []
        state = {"status": "idle", "turn_status": None, "items": []}

        def fake_request(method, path, body=None, **_):
            requests.append((method, path, body))
            if path == "/agents/sessions" and method == "POST":
                state["status"] = "requires_action"
                state["turn_status"] = "waiting"
                state["items"] = [{
                    "id": "input-1", "type": "message", "role": "user",
                    "turn_id": "turn-1", "content": [
                        {"type": "input_text", "text": body["input"]}],
                }]
                return {"id": "session-1", "status": "requires_action",
                        "agent": {"id": "agent-1"}}
            if path == "/agents/sessions/session-1" and method == "GET":
                session = {"id": "session-1", "status": state["status"],
                           "agent": {"id": "agent-1"}}
                if state["status"] == "requires_action":
                    session["required_actions"] = [{
                        "type": "function_call", "turn_id": "turn-1", "call_id": "call-1",
                        "name": "clock", "arguments": {},
                    }]
                return session
            if path.endswith("/events"):
                event = body["events"][0]
                if event["type"] == "agent.session.input.message":
                    state["status"] = "requires_action"
                    state["turn_status"] = "waiting"
                    state["items"] = [{
                        "id": "input-1", "type": "message", "role": "user",
                        "turn_id": "turn-1", "content": event["input"][0]["content"],
                    }]
                else:
                    state["status"] = "idle"
                    state["turn_status"] = "completed"
                    state["items"].insert(0, {
                        "id": "output-1", "type": "message", "role": "assistant",
                        "turn_id": "turn-1",
                        "content": [{"type": "output_text", "text": "done"}],
                    })
                return {}
            if "/turns?" in path:
                data = [] if state["turn_status"] is None else [{
                    "id": "turn-1", "status": state["turn_status"],
                    "usage": {"input_tokens": 3},
                }]
                return {"data": data}
            if path.endswith("/turns/turn-1"):
                return {"id": "turn-1", "status": state["turn_status"],
                        "usage": {"input_tokens": 3}}
            if "/items?" in path:
                return {"data": state["items"]}
            if path == "/agents/sessions/session-1" and method == "POST":
                return {"id": "session-1", "status": "idle", "agent": {"id": "agent-1"}}
            raise AssertionError((method, path, body))

        provider._request = fake_request
        context = json.dumps({"wake_event": {"id": "wake-1", "payload": {}}})
        spec = ToolSpec("clock", "Read clock", {
            "type": "object", "properties": {}, "additionalProperties": False})
        first = await provider.respond(context, [spec], [])
        second = await provider.respond(
            context, [spec], [ToolResult("call-1", {"ok": True})], first.response_id)

        self.assertEqual("turn-1", first.response_id)
        self.assertEqual("done", second.message)
        self.assertEqual("session-1", binding["session_id"])
        event_bodies = [body for method, path, body in requests if path.endswith("/events")]
        create_body = next(body for method, path, body in requests
                           if method == "POST" and path == "/agents/sessions")
        self.assertIn('"id":"wake-1"', create_body["input"])
        self.assertEqual("agent.session.input.tool_result", event_bodies[0]["events"][0]["type"])
        self.assertEqual("turn-1", event_bodies[0]["events"][0]["turn_id"])
        self.assertEqual(1, len(event_bodies))

    def test_agents_event_idempotency_is_an_http_header_not_a_body_field(self):
        provider = OpenAIAgentsProvider("test-key", "gpt-5.6-luna")
        requests = []

        def fake_request(method, path, body=None, **kwargs):
            requests.append((method, path, body, kwargs))
            return {}

        provider._request = fake_request
        event = {
            "type": "agent.session.input.tool_result",
            "turn_id": "turn-1", "call_id": "call-1",
            "success": True, "output": '{"ok":true}',
        }

        provider._submit_events("session-1", [event], "resident-tool:turn-1:call-1")

        method, path, body, kwargs = requests[0]
        self.assertEqual(("POST", "/agents/sessions/session-1/events"), (method, path))
        self.assertEqual({"events": [event]}, body)
        self.assertNotIn("idempotency_key", body)
        self.assertEqual(
            {"Idempotency-Key": "resident-tool:turn-1:call-1"},
            kwargs["extra_headers"])
        self.assertTrue(kwargs["allow_empty"])

    async def test_recovered_requires_action_finishes_before_new_ordinary_wake(self):
        provider = OpenAIAgentsProvider("test-key", "gpt-5.6-luna", poll_seconds=0)
        provider._session_id = "session-1"
        provider._last_turn_id = "turn-before-recovery"
        state = {"status": "requires_action", "turns": {
            "turn-recovered": "waiting"}, "items": []}
        submitted = []

        def fake_request(method, path, body=None, **_):
            if path == "/agents/sessions/session-1" and method == "GET":
                session = {"id": "session-1", "status": state["status"]}
                if state["status"] == "requires_action":
                    session["required_actions"] = [{
                        "type": "function_call", "turn_id": "turn-recovered",
                        "call_id": "recovered-call", "name": "clock", "arguments": {},
                    }]
                return session
            if path == "/agents/sessions/session-1" and method == "POST":
                return {"id": "session-1", "status": state["status"]}
            if path.endswith("/events"):
                event = body["events"][0]
                submitted.append(event["type"])
                if event["type"] == "agent.session.input.tool_result":
                    state["status"] = "idle"
                    state["turns"]["turn-recovered"] = "completed"
                    state["items"].append({
                        "id": "recovered-output", "type": "message", "role": "assistant",
                        "turn_id": "turn-recovered",
                        "content": [{"type": "output_text", "text": "recovered done"}],
                    })
                else:
                    state["turns"]["turn-wake"] = "completed"
                    state["items"] = [{
                        "id": "wake-output", "type": "message", "role": "assistant",
                        "turn_id": "turn-wake",
                        "content": [{"type": "output_text", "text": "new wake done"}],
                    }, {
                        "id": "wake-input", "type": "message", "role": "user",
                        "turn_id": "turn-wake", "content": event["input"][0]["content"],
                    }, *state["items"]]
                return {}
            if "/turns?" in path:
                latest = next(reversed(state["turns"]))
                return {"data": [{"id": latest, "status": state["turns"][latest]}]}
            if "/turns/" in path:
                turn_id = path.rsplit("/", 1)[-1]
                return {"id": turn_id, "status": state["turns"][turn_id]}
            if "/items?" in path:
                return {"data": state["items"]}
            raise AssertionError((method, path, body))

        provider._request = fake_request
        context = json.dumps({
            "wake_event": {"id": "wake-new", "source": "homeops", "payload": {}}})
        first = await provider.respond(context, [], [])
        self.assertEqual("turn-recovered", first.response_id)
        self.assertEqual([], submitted)

        completed = await provider.respond(
            context, [], [ToolResult("recovered-call", {"ok": True})], first.response_id)

        self.assertEqual("turn-wake", completed.response_id)
        self.assertEqual("new wake done", completed.message)
        self.assertEqual([
            "agent.session.input.tool_result", "agent.session.input.message"], submitted)

    async def test_stale_persisted_turn_cannot_satisfy_new_wake(self):
        provider = OpenAIAgentsProvider("test-key", "gpt-5.6-luna", poll_seconds=0)
        provider._session_id = "session-1"
        provider._last_turn_id = "turn-stale-binding"
        state = {"items": [{
            "id": "old-output", "type": "message", "role": "assistant",
            "turn_id": "turn-old", "content": [
                {"type": "output_text", "text": "unrelated old completion"}],
        }]}

        def fake_request(method, path, body=None, **_):
            if path == "/agents/sessions/session-1" and method == "GET":
                return {"id": "session-1", "status": "idle"}
            if path == "/agents/sessions/session-1" and method == "POST":
                return {"id": "session-1", "status": "idle"}
            if path.endswith("/events"):
                event = body["events"][0]
                state["items"] = [{
                    "id": "new-output", "type": "message", "role": "assistant",
                    "turn_id": "turn-new", "content": [
                        {"type": "output_text", "text": "correlated completion"}],
                }, {
                    "id": "new-input", "type": "message", "role": "user",
                    "turn_id": "turn-new", "content": event["input"][0]["content"],
                }, *state["items"]]
                return {}
            if "/turns?" in path:
                return {"data": [{"id": "turn-old", "status": "completed"}]}
            if path.endswith("/turns/turn-new"):
                return {"id": "turn-new", "status": "completed"}
            if "/items?" in path:
                return {"data": state["items"]}
            raise AssertionError((method, path, body))

        provider._request = fake_request
        context = json.dumps({
            "wake_event": {"id": "wake-new", "source": "homeops", "payload": {}}})
        turn = await provider.respond(context, [], [])

        self.assertEqual("turn-new", turn.response_id)
        self.assertEqual("correlated completion", turn.message)

    async def test_ordinary_wake_waits_for_active_turn_before_submission(self):
        provider = OpenAIAgentsProvider("test-key", "gpt-5.6-luna", poll_seconds=0)
        provider._session_id = "session-1"
        state = {"status": "in_progress", "items": []}
        operations = []

        def fake_request(method, path, body=None, **_):
            operations.append((method, path))
            if path == "/agents/sessions/session-1" and method == "GET":
                return {"id": "session-1", "status": state["status"]}
            if path == "/agents/sessions/session-1" and method == "POST":
                return {"id": "session-1", "status": state["status"]}
            if "/turns?" in path:
                return {"data": [{"id": "turn-active", "status": "in_progress"}]}
            if path.endswith("/turns/turn-active"):
                state["status"] = "idle"
                return {"id": "turn-active", "status": "completed"}
            if path.endswith("/events"):
                event = body["events"][0]
                state["items"] = [{
                    "id": "input-new", "type": "message", "role": "user",
                    "turn_id": "turn-new", "content": event["input"][0]["content"],
                }]
                return {}
            if path.endswith("/turns/turn-new"):
                return {"id": "turn-new", "status": "completed"}
            if "/items?" in path:
                return {"data": state["items"]}
            raise AssertionError((method, path, body))

        provider._request = fake_request
        context = json.dumps({
            "wake_event": {"id": "ordinary", "source": "connector", "payload": {}}})
        turn = await provider.respond(context, [], [])

        active_done = operations.index(("GET", "/agents/sessions/session-1/turns/turn-active"))
        wake_submitted = operations.index(("POST", "/agents/sessions/session-1/events"))
        self.assertLess(active_done, wake_submitted)
        self.assertEqual("turn-new", turn.response_id)

    async def test_deferred_configuration_is_applied_before_reconciled_wake(self):
        provider = OpenAIAgentsProvider("test-key", "gpt-5.6-luna", poll_seconds=0)
        provider._session_id = "session-1"
        original = ToolSpec("clock", "Read clock", {"type": "object"})
        changed = ToolSpec("clock", "Read the local clock", {"type": "object"})
        provider._tool_fingerprint = json.dumps(
            provider._agent_config([original]), sort_keys=True, separators=(",", ":"))
        changed_fingerprint = json.dumps(
            provider._agent_config([changed]), sort_keys=True, separators=(",", ":"))
        state = {"status": "in_progress", "items": []}
        operations = []

        def fake_request(method, path, body=None, **_):
            if path == "/agents/sessions/session-1" and method == "GET":
                return {"id": "session-1", "status": state["status"]}
            if path == "/agents/sessions" and method == "POST":
                operations.append("configuration_with_wake")
                self.assertEqual(
                    "Read the local clock", body["agent"]["tools"][0]["description"])
                self.assertNotIn("name", body["agent"])
                state["items"] = [{
                    "id": "input-new", "type": "message", "role": "user",
                    "turn_id": "turn-new", "content": [
                        {"type": "input_text", "text": body["input"]}],
                }]
                return {"id": "session-2", "status": "idle"}
            if path == "/agents/sessions/session-2" and method == "GET":
                return {"id": "session-2", "status": "idle"}
            if "/turns?" in path:
                return {"data": [{"id": "turn-active", "status": "in_progress"}]}
            if path.endswith("/turns/turn-active"):
                state["status"] = "idle"
                return {"id": "turn-active", "status": "completed"}
            if path.endswith("/events"):
                operations.append("wake")
                event = body["events"][0]
                state["items"] = [{
                    "id": "input-new", "type": "message", "role": "user",
                    "turn_id": "turn-new", "content": event["input"][0]["content"],
                }]
                return {}
            if path.endswith("/turns/turn-new"):
                return {"id": "turn-new", "status": "completed"}
            if "/items?" in path:
                return {"data": state["items"]}
            raise AssertionError((method, path, body))

        provider._request = fake_request
        context = json.dumps({
            "wake_event": {"id": "capability-change", "source": "connector", "payload": {}}})

        turn = await provider.respond(context, [changed], [])

        self.assertEqual("turn-new", turn.response_id)
        self.assertEqual(["configuration_with_wake"], operations)
        self.assertEqual(changed_fingerprint, provider._tool_fingerprint)

    async def test_failed_deferred_configuration_replacement_does_not_submit_wake(self):
        provider = OpenAIAgentsProvider("test-key", "gpt-5.6-luna", poll_seconds=0)
        provider._session_id = "session-1"
        original = ToolSpec("clock", "Read clock", {"type": "object"})
        changed = ToolSpec("clock", "Read the local clock", {"type": "object"})
        applied_fingerprint = json.dumps(
            provider._agent_config([original]), sort_keys=True, separators=(",", ":"))
        provider._tool_fingerprint = applied_fingerprint
        state = {"status": "in_progress"}
        submitted = []

        def fake_request(method, path, body=None, **_):
            if path == "/agents/sessions/session-1" and method == "GET":
                return {"id": "session-1", "status": state["status"]}
            if path == "/agents/sessions" and method == "POST":
                raise RuntimeError("configuration replacement failed")
            if "/turns?" in path:
                return {"data": [{"id": "turn-active", "status": "in_progress"}]}
            if path.endswith("/turns/turn-active"):
                state["status"] = "idle"
                return {"id": "turn-active", "status": "completed"}
            if path.endswith("/events"):
                submitted.append(body)
                return {}
            if "/items?" in path:
                return {"data": []}
            raise AssertionError((method, path, body))

        provider._request = fake_request
        context = json.dumps({
            "wake_event": {"id": "capability-change", "source": "connector", "payload": {}}})

        with self.assertRaisesRegex(RuntimeError, "configuration replacement failed"):
            await provider.respond(context, [changed], [])

        self.assertEqual([], submitted)
        self.assertEqual(applied_fingerprint, provider._tool_fingerprint)

    async def test_reconciled_wake_does_not_reapply_unchanged_configuration(self):
        provider = OpenAIAgentsProvider("test-key", "gpt-5.6-luna", poll_seconds=0)
        provider._session_id = "session-1"
        spec = ToolSpec("clock", "Read clock", {"type": "object"})
        provider._tool_fingerprint = json.dumps(
            provider._agent_config([spec]), sort_keys=True, separators=(",", ":"))
        state = {"status": "in_progress", "items": []}
        updates = []

        def fake_request(method, path, body=None, **_):
            if path == "/agents/sessions/session-1" and method == "GET":
                return {"id": "session-1", "status": state["status"]}
            if path == "/agents/sessions/session-1" and method == "POST":
                updates.append(body)
                return {"id": "session-1", "status": "idle"}
            if "/turns?" in path:
                return {"data": [{"id": "turn-active", "status": "in_progress"}]}
            if path.endswith("/turns/turn-active"):
                state["status"] = "idle"
                return {"id": "turn-active", "status": "completed"}
            if path.endswith("/events"):
                event = body["events"][0]
                state["items"] = [{
                    "id": "input-new", "type": "message", "role": "user",
                    "turn_id": "turn-new", "content": event["input"][0]["content"],
                }]
                return {}
            if path.endswith("/turns/turn-new"):
                return {"id": "turn-new", "status": "completed"}
            if "/items?" in path:
                return {"data": state["items"]}
            raise AssertionError((method, path, body))

        provider._request = fake_request
        context = json.dumps({
            "wake_event": {"id": "ordinary", "source": "connector", "payload": {}}})

        turn = await provider.respond(context, [spec], [])

        self.assertEqual("turn-new", turn.response_id)
        self.assertEqual([], updates)

    async def test_owner_wake_can_steer_active_turn_and_is_correlated_to_it(self):
        provider = OpenAIAgentsProvider("test-key", "gpt-5.6-luna", poll_seconds=0)
        provider._session_id = "session-1"
        state = {"items": []}
        operations = []

        def fake_request(method, path, body=None, **_):
            operations.append((method, path))
            if path == "/agents/sessions/session-1" and method == "GET":
                return {"id": "session-1", "status": "in_progress"}
            if path == "/agents/sessions/session-1" and method == "POST":
                return {"id": "session-1", "status": "in_progress"}
            if path.endswith("/events"):
                event = body["events"][0]
                state["items"] = [{
                    "id": "owner-input", "type": "message", "role": "user",
                    "turn_id": "turn-active", "content": event["input"][0]["content"],
                }]
                return {}
            if path.endswith("/turns/turn-active"):
                return {"id": "turn-active", "status": "completed"}
            if "/items?" in path:
                return {"data": state["items"]}
            raise AssertionError((method, path, body))

        provider._request = fake_request
        context = json.dumps({
            "wake_event": {"id": "owner", "source": "owner", "payload": {
                "message_id": "message-1"}}})
        turn = await provider.respond(context, [], [])

        self.assertEqual("turn-active", turn.response_id)
        self.assertNotIn(("GET", "/agents/sessions/session-1/turns?order=desc&limit=1"),
                         operations)
        self.assertIn(("POST", "/agents/sessions/session-1/events"), operations)

    def test_agent_session_binding_can_be_rolled_over(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            store.save_agent_session_binding("openai_agents", "session-1", "agent-1", "turn-1")
            store.save_agent_session_binding("openai_agents", "session-2", "agent-1", None)

            binding = store.agent_session_binding("openai_agents")
            self.assertEqual("session-2", binding["session_id"])
            self.assertIsNone(binding["last_turn_id"])
            store.close()

    def test_agent_tool_action_is_claimed_once_and_replays_completed_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            first = store.begin_agent_tool_action(
                "openai_agents", "session", "turn", "call", "display1_show_text",
                {"text": "hello"})
            store.complete_agent_tool_action(
                "openai_agents", "session", "call", {"ok": True, "queued": True})
            replay = store.begin_agent_tool_action(
                "openai_agents", "session", "turn", "call", "display1_show_text",
                {"text": "hello"})

            self.assertTrue(first["claimed"])
            self.assertFalse(replay["claimed"])
            self.assertEqual({"ok": True, "queued": True}, replay["output"])
            self.assertFalse(replay["attachments_ephemeral"])
            store.close()

    def test_legacy_memories_table_is_removed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "resident.sqlite3"
            connection = sqlite3.connect(path)
            connection.executescript("""
                CREATE TABLE schema_version(version INTEGER NOT NULL);
                INSERT INTO schema_version VALUES(11);
                CREATE TABLE memories(
                  id TEXT PRIMARY KEY, content TEXT NOT NULL, source TEXT NOT NULL,
                  created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
                INSERT INTO memories VALUES('legacy','obsolete','resident','now','now');
            """)
            connection.commit()
            connection.close()

            store = Store(path)

            self.assertEqual(15, store.connection.execute(
                "SELECT version FROM schema_version").fetchone()[0])
            self.assertIsNone(store.connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memories'").fetchone())
            store.close()

    def test_restart_restores_session_and_replays_completed_action_without_reclaiming(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "resident.sqlite3"
            store = Store(path)
            store.save_agent_session_binding(
                "openai_agents", "session-1", "agent-1", "turn-1")
            claimed = store.begin_agent_tool_action(
                "openai_agents", "session-1", "turn-1", "call-1",
                "display1_show_text", {"text": "hello"})
            store.complete_agent_tool_action(
                "openai_agents", "session-1", "call-1",
                {"ok": True, "queued": True})
            self.assertTrue(claimed["claimed"])
            store.close()

            reopened = Store(path)
            provider = OpenAIAgentsProvider("test-key", "gpt-5.6-luna")
            provider.bind_session_store(
                lambda: reopened.agent_session_binding("openai_agents"),
                lambda session_id, agent_id, last_turn_id:
                    reopened.save_agent_session_binding(
                        "openai_agents", session_id, agent_id, last_turn_id),
            )
            provider.bind_action_store(
                reopened.begin_agent_tool_action, reopened.complete_agent_tool_action)
            provider._active_turn_id = "turn-1"

            replay = provider.prepare_tool_call(ToolCall(
                "call-1", "display1_show_text", {"text": "hello"}))

            self.assertEqual("session-1", provider._session_id)
            self.assertEqual("turn-1", provider._last_turn_id)
            self.assertFalse(replay["claimed"])
            self.assertEqual({"ok": True, "queued": True}, replay["output"])
            reopened.close()

    def test_attachment_action_reuses_memory_only_and_marks_durable_reacquisition(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            provider = OpenAIAgentsProvider("test-key", "vision-model")
            provider._session_id = "session"
            provider._active_turn_id = "turn"
            provider.bind_action_store(
                store.begin_agent_tool_action, store.complete_agent_tool_action)
            call = ToolCall("capture", "camera_capture_frame", {"camera_id": "entry"})
            self.assertTrue(provider.prepare_tool_call(call)["claimed"])

            result = ToolResult(
                "capture", {"ok": True, "status": "captured"},
                (ImageAttachment(b"private-frame"),))
            provider.record_tool_result(result)
            same_process = provider.prepare_tool_call(call)
            self.assertIs(result, same_process["ephemeral_result"])
            self.assertTrue(same_process["attachments_ephemeral"])

            restarted = OpenAIAgentsProvider("test-key", "vision-model")
            restarted._session_id = "session"
            restarted._active_turn_id = "turn"
            restarted.bind_action_store(
                store.begin_agent_tool_action, store.complete_agent_tool_action)
            after_restart = restarted.prepare_tool_call(call)
            self.assertFalse(after_restart["claimed"])
            self.assertTrue(after_restart["attachments_ephemeral"])
            self.assertIsNone(after_restart["ephemeral_result"])
            database_text = " ".join(
                str(value) for row in store.connection.execute(
                    "SELECT output_json,attachments_ephemeral FROM agent_tool_actions")
                for value in row)
            self.assertNotIn("private-frame", database_text)
            store.close()

    def test_agent_turn_message_uses_descending_cursor_pagination(self):
        provider = OpenAIAgentsProvider("test-key", "gpt-5.6-luna")
        paths = []

        def fake_request(method, path, body=None, **_):
            paths.append(path)
            if "after=" not in path:
                return {
                    "data": [
                        {"id": "newest", "turn_id": "turn-2", "type": "message",
                         "role": "assistant", "content": [
                             {"type": "output_text", "text": "second"}]},
                        {"id": "cursor", "turn_id": "turn-2", "type": "reasoning"},
                    ],
                    "has_more": True,
                    "last_id": "cursor",
                }
            return {
                "data": [
                    {"id": "older-in-turn", "turn_id": "turn-2", "type": "message",
                     "role": "assistant", "content": [
                         {"type": "output_text", "text": "first"}]},
                    {"id": "old-turn", "turn_id": "turn-1", "type": "message",
                     "role": "assistant", "content": [
                         {"type": "output_text", "text": "ignore"}]},
                ],
                "has_more": True,
                "last_id": "old-turn",
            }

        provider._request = fake_request
        self.assertEqual("first\nsecond", provider._turn_message("session", "turn-2"))
        self.assertEqual(
            "/agents/sessions/session/items?order=desc&limit=100", paths[0])
        self.assertEqual(
            "/agents/sessions/session/items?order=desc&limit=100&after=cursor", paths[1])
        self.assertEqual(2, len(paths))

    async def test_responses_wire_contract_is_confined_to_adapter(self):
        provider = OpenAIResponsesProvider("test-key", "gpt-5.6-luna")
        requests = []
        responses = iter((
            {"id": "resp-1", "status": "completed", "output": [{
                "type": "function_call", "call_id": "call-1", "name": "clock", "arguments": "{}"}]},
            {"id": "resp-2", "status": "completed", "output": [{
                "type": "message", "content": [{"type": "output_text", "text": "done"}]}]},
        ))

        def fake_post(body):
            requests.append(body)
            return next(responses)

        provider._post = fake_post
        spec = ToolSpec("clock", "Read clock", {"type": "object", "properties": {}, "additionalProperties": False})
        first = await provider.respond("context", [spec], [])
        second = await provider.respond("context", [spec], [ToolResult("call-1", {"ok": True})], first.response_id)

        self.assertEqual("gpt-5.6-luna", requests[0]["model"])
        self.assertEqual("function", requests[0]["tools"][0]["type"])
        self.assertNotIn("previous_response_id", requests[1])
        self.assertEqual("user", requests[1]["input"][0]["role"])
        self.assertEqual("function_call", requests[1]["input"][1]["type"])
        self.assertEqual("function_call_output", requests[1]["input"][2]["type"])
        self.assertEqual("call-1", requests[1]["input"][2]["call_id"])
        self.assertEqual("done", second.message)
        self.assertFalse(requests[0]["store"])
        self.assertFalse(requests[1]["store"])
        self.assertEqual(["reasoning.encrypted_content"], requests[0]["include"])
        self.assertIn("all intentional communication", requests[0]["instructions"])
        self.assertIn("final response message is wake-result diagnostic text only",
                      requests[0]["instructions"])
        self.assertIn("Local events are factual observations", requests[0]["instructions"])

    async def test_image_tool_result_is_sent_as_multimodal_ephemeral_content(self):
        provider = OpenAIResponsesProvider("test-key", "vision-model")
        requests = []
        responses = iter((
            {"id": "previous", "status": "completed", "output": [{
                "type": "function_call", "call_id": "capture", "name": "camera_capture_frame",
                "arguments": '{"camera_id":"entry"}',
            }]},
            {"id": "response", "status": "completed", "output": [{
                "type": "function_call", "call_id": "follow-up", "name": "clock", "arguments": "{}",
            }]},
            {"id": "final", "status": "completed", "output": []},
        ))
        provider._post = lambda body: requests.append(body) or next(responses)

        first = await provider.respond("context", [], [])
        second = await provider.respond("context", [], [ToolResult(
            "capture", {"status": "captured"},
            (ImageAttachment(b"\xff\xd8image\xff\xd9", detail="low"),),
        )], first.response_id)
        await provider.respond("context", [], [ToolResult("follow-up", {"time": "12:00"})], second.response_id)

        output = requests[1]["input"][-1]["output"]
        self.assertEqual({"type": "input_text", "text": '{"status":"captured"}'}, output[0])
        self.assertEqual("input_image", output[1]["type"])
        self.assertEqual("low", output[1]["detail"])
        self.assertTrue(output[1]["image_url"].startswith("data:image/jpeg;base64,"))
        self.assertEqual("function_call_output", requests[2]["input"][2]["type"])
        self.assertEqual("capture", requests[2]["input"][2]["call_id"])
        self.assertEqual("function_call_output", requests[2]["input"][4]["type"])
        self.assertEqual("follow-up", requests[2]["input"][4]["call_id"])
        self.assertEqual(output, requests[2]["input"][2]["output"])

    async def test_discard_removes_image_bearing_continuation_history(self):
        provider = OpenAIResponsesProvider("test-key", "vision-model")
        provider._post = lambda body: {
            "id": "image-continuation", "status": "completed", "output": [{
                "type": "function_call", "call_id": "capture", "name": "camera_capture_frame",
                "arguments": "{}",
            }],
        }

        await provider.respond("context", [], [])
        await provider.respond("context", [], [ToolResult(
            "capture", {"status": "captured"}, (ImageAttachment(b"frame"),),
        )], "image-continuation")

        self.assertIn("image-continuation", provider._histories)
        self.assertTrue(any("base64" in json.dumps(item) for item in provider._histories["image-continuation"]))
        provider.discard_continuation("image-continuation")
        self.assertNotIn("image-continuation", provider._histories)


if __name__ == "__main__":
    unittest.main()
