from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from resident.config import Config
from resident.capabilities import Capability
from resident.domain import ModelTurn, ToolCall, WakeEvent
from resident.domain import ImageAttachment, ToolResult, ToolSpec
from resident.provider import OpenAIResponsesProvider
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
                ToolCall(f"remember-{self.round}", "remember", {"content": f"Wake observed: {content}"}),
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


class ContinuationLifecycleProvider:
    def __init__(self, *, cancel=False):
        self.cancel = cancel
        self.discarded: list[str] = []
        self.release = asyncio.Event()

    async def respond(self, context, tools, results, previous_response_id=None):
        if previous_response_id is None:
            return ModelTurn("continuation", tool_calls=(ToolCall("call", "remember", {"content": "work"}),))
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

    def test_direct_config_default_window_differs_from_cli_default(self):
        direct = Config(Path("."))
        parsed = Config.from_env_and_args(["--data-dir", "."])

        self.assertEqual(180, direct.spontaneous_message_window_seconds)
        self.assertEqual(3600, parsed.spontaneous_message_window_seconds)


class StoreTests(unittest.TestCase):
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
            self.assertEqual(6, store.connection.execute(
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
