from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from resident.config import Config
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


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
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
            import asyncio
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
            self.assertEqual(2, store.connection.execute(
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
            {"id": "response", "status": "completed", "output": []},
        ))
        provider._post = lambda body: requests.append(body) or next(responses)

        first = await provider.respond("context", [], [])
        await provider.respond("context", [], [ToolResult(
            "capture", {"status": "captured"},
            (ImageAttachment(b"\xff\xd8image\xff\xd9", detail="low"),),
        )], first.response_id)

        output = requests[1]["input"][-1]["output"]
        self.assertEqual({"type": "input_text", "text": '{"status":"captured"}'}, output[0])
        self.assertEqual("input_image", output[1]["type"])
        self.assertEqual("low", output[1]["detail"])
        self.assertTrue(output[1]["image_url"].startswith("data:image/jpeg;base64,"))


if __name__ == "__main__":
    unittest.main()
