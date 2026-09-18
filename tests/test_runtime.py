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
from resident.provider import OpenAIAgentsProvider, OpenAIResponsesProvider
from resident.runtime import ResidentRuntime
from resident.store import Store, utc_now


class LifecycleProvider:
    """Deterministic provider that exercises durable state, capability, and communication tools."""

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
                ToolCall(f"remember-{self.round}", "remember", {
                    "content": f"Wake observed: {content}", "kind": "experience",
                    "importance": "low", "confidence": "high", "provenance": "resident",
                    "standing": False,
                }),
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


class ContinuationLifecycleProvider:
    def __init__(self, *, cancel=False):
        self.cancel = cancel
        self.discarded: list[str] = []
        self.release = asyncio.Event()

    async def respond(self, context, tools, results, previous_response_id=None):
        if previous_response_id is None:
            return ModelTurn("continuation", tool_calls=(ToolCall("call", "remember", {
                "content": "work", "kind": "experience", "importance": "low",
                "confidence": "medium", "provenance": "resident",
                "standing": False,
            }),))
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
            with self.assertRaisesRegex(ValueError, "conflicts with a core tool: remember"):
                ResidentRuntime(
                    Config(Path(temporary)), MessageOnlyProvider(),
                    capabilities=[self.capability("remember")],
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

    async def test_full_lifecycle_retains_identity_and_memory_across_restart(self):
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
            await second.process(second.owner_message_event("what do you remember about hello home?"))

            context = second_provider.contexts[0]
            self.assertTrue(any("hello home" in m["content"] for m in context["retrieved_memories"]))
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
            config = Config(Path(temporary), context_memories=2, context_messages=2)
            provider = LifecycleProvider()
            runtime = ResidentRuntime(config, provider, owner_output=lambda _: None, diagnostic_output=lambda _: None)
            for number in range(6):
                runtime.store.remember(f"bounded topic {number}", "test")
                runtime.store.add_message("inbound", runtime.owner.id, f"older {number}")
            event = runtime.owner_message_event("bounded topic with full payload")
            await runtime.process(event)
            context = provider.contexts[0]
            self.assertEqual("bounded topic with full payload", context["wake_event"]["payload"]["content"])
            self.assertLessEqual(len(context["retrieved_memories"]), 2)
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
            self.assertEqual(11, store.connection.execute(
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
            self.assertEqual(11, store.connection.execute(
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


class OpenAIAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_agents_runtime_persists_session_and_turn_on_sqlite_owner_thread(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "resident.sqlite3"
            state = {"status": "idle", "turns": {}, "items": []}
            request_threads = []

            def fake_request(method, request_path, body=None, **_):
                request_threads.append(threading.get_ident())
                if request_path == "/agents/sessions" and method == "POST":
                    return {"id": "session-1", "status": "idle",
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
            self.assertEqual("agent-1", store.agent_session_binding("openai_agents")["agent_id"])
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
            self.assertEqual("agent-1", recovered._bound_agent_id)

            second_context = json.dumps({
                "wake_event": {"id": "wake-2", "source": "connector", "payload": {}}})
            second = await recovered.respond(second_context, [], [])

            self.assertEqual("turn-2", second.response_id)
            self.assertEqual(
                "turn-2", reopened_store.agent_session_binding("openai_agents")["last_turn_id"])
            self.assertEqual(
                [reopened_store.owner_thread_id], reopened_store.binding_save_threads)
            reopened_runtime.close()

    def test_agents_binding_restores_persisted_agent_identity_without_override(self):
        provider = OpenAIAgentsProvider("test-key", "gpt-5.6-luna")
        binding = {
            "session_id": "session-persisted",
            "agent_id": "agent-persisted",
            "last_turn_id": "turn-persisted",
        }

        provider.bind_session_store(lambda: binding, lambda *_: None)

        self.assertEqual("session-persisted", provider._session_id)
        self.assertEqual("agent-persisted", provider._bound_agent_id)
        self.assertEqual("turn-persisted", provider._last_turn_id)

    def test_agents_missing_session_reuses_persisted_agent_and_saves_replacement(self):
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

        session = provider._ensure_session([])

        self.assertEqual("session-replacement", session["id"])
        create_body = requests[-1][2]
        self.assertEqual("agent-persisted", create_body["agent_id"])
        self.assertEqual("gpt-5.6-luna", create_body["agent"]["model"])
        self.assertEqual({
            "session_id": "session-replacement",
            "agent_id": "agent-persisted",
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
        session = provider._ensure_session([])

        self.assertEqual("session-configured", session["id"])
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
        self.assertFalse(state["turn_created"])

        turn = await provider.respond(context, [], [])

        self.assertEqual("turn-1", turn.response_id)
        self.assertEqual(1, len(create_calls))
        self.assertEqual([owner_thread_id] * 3, [call[0] for call in save_calls])
        self.assertEqual(
            [("session-1", "agent-1", None), ("session-1", "agent-1", None),
             ("session-1", "agent-1", "turn-1")],
            [call[1:] for call in save_calls])

    def test_agents_configuration_change_is_deferred_until_session_is_idle(self):
        provider = OpenAIAgentsProvider("test-key", "gpt-5.6-luna")
        status = {"value": "idle"}
        updates = []

        def fake_request(method, path, body=None, **_):
            if path == "/agents/sessions" and method == "POST":
                return {"id": "session-1", "status": "idle"}
            if path == "/agents/sessions/session-1" and method == "GET":
                return {"id": "session-1", "status": status["value"]}
            if path == "/agents/sessions/session-1" and method == "POST":
                updates.append(body)
                return {"id": "session-1", "status": "idle"}
            raise AssertionError((method, path, body))

        provider._request = fake_request
        original = ToolSpec("clock", "Read clock", {"type": "object"})
        changed = ToolSpec("clock", "Read the local clock", {"type": "object"})

        provider._ensure_session([original])
        applied_fingerprint = provider._tool_fingerprint
        status["value"] = "in_progress"
        provider._ensure_session([changed])

        self.assertEqual(applied_fingerprint, provider._tool_fingerprint)
        self.assertEqual([], updates)

        status["value"] = "idle"
        provider._ensure_session([changed])
        changed_fingerprint = json.dumps(
            provider._agent_config([changed]), sort_keys=True, separators=(",", ":"))
        self.assertEqual(changed_fingerprint, provider._tool_fingerprint)
        self.assertEqual("Read the local clock", updates[0]["agent"]["tools"][0]["description"])

        provider._ensure_session([changed])
        self.assertEqual(1, len(updates))

    def test_agents_configuration_fingerprint_advances_only_after_successful_update(self):
        provider = OpenAIAgentsProvider("test-key", "gpt-5.6-luna")
        fail_update = {"value": True}
        updates = []

        def fake_request(method, path, body=None, **_):
            if path == "/agents/sessions" and method == "POST":
                return {"id": "session-1", "status": "idle"}
            if path == "/agents/sessions/session-1" and method == "GET":
                return {"id": "session-1", "status": "idle"}
            if path == "/agents/sessions/session-1" and method == "POST":
                updates.append(body)
                if fail_update["value"]:
                    raise RuntimeError("configuration update failed")
                return {"id": "session-1", "status": "idle"}
            raise AssertionError((method, path, body))

        provider._request = fake_request
        original = ToolSpec("clock", "Read clock", {"type": "object"})
        changed = ToolSpec("clock", "Read the local clock", {"type": "object"})
        provider._ensure_session([original])
        applied_fingerprint = provider._tool_fingerprint

        with self.assertRaisesRegex(RuntimeError, "configuration update failed"):
            provider._ensure_session([changed])
        self.assertEqual(applied_fingerprint, provider._tool_fingerprint)

        fail_update["value"] = False
        provider._ensure_session([changed])
        self.assertNotEqual(applied_fingerprint, provider._tool_fingerprint)
        self.assertEqual(2, len(updates))

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
                return {"id": "session-1", "status": "idle", "agent": {"id": "agent-1"}}
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
        self.assertEqual("resident-wake:wake-1", event_bodies[0]["idempotency_key"])
        self.assertEqual("agent.session.input.tool_result", event_bodies[1]["events"][0]["type"])
        self.assertEqual("turn-1", event_bodies[1]["events"][0]["turn_id"])
        self.assertEqual(2, len(event_bodies))

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
            if path == "/agents/sessions/session-1" and method == "POST":
                operations.append("configuration")
                self.assertEqual(
                    "Read the local clock", body["agent"]["tools"][0]["description"])
                return {"id": "session-1", "status": "idle"}
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
        self.assertEqual(["configuration", "wake"], operations)
        self.assertEqual(changed_fingerprint, provider._tool_fingerprint)

    async def test_failed_deferred_configuration_update_does_not_submit_wake(self):
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
            if path == "/agents/sessions/session-1" and method == "POST":
                raise RuntimeError("configuration update failed")
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

        with self.assertRaisesRegex(RuntimeError, "configuration update failed"):
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
        self.assertEqual(("POST", "/agents/sessions/session-1/events"), operations[1])

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
        self.assertIn("supplied owner_guidance", requests[0]["instructions"])

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
