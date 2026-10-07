from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from resident.config import Config
from resident.domain import ModelTurn, ToolCall, WakeEvent
from resident.outputs import (DeliveryPolicy, OutputCapability, output_schema,
                              schema_fingerprint, validate_disposition)
from runtime_support import RecordingProvider as StructuredProvider
from resident.runtime import ResidentRuntime
from resident.store import Store, utc_now


def display_capability(target, delivered, *, max_length=40, handler=None,
                       max_attempts=3):
    async def deliver(payload):
        delivered.append((target, payload["content"]))
        if handler is not None:
            return await handler(payload)
        return {"status": "accepted_by_homeops"}

    return OutputCapability(
        "display", f"Write to {target}", {
            "type": "object",
            "properties": {"content": {
                "type": "string", "minLength": 1, "maxLength": max_length}},
            "required": ["content"], "additionalProperties": False,
        }, f"homeops-display:{target}", deliver, target=target,
        delivery_policy=DeliveryPolicy(max_attempts=max_attempts))


def notify_owner_capability():
    async def deliver(_payload):
        return {"status": "accepted_by_transport"}

    return OutputCapability(
        "notify_owner", "Send a statement or question to the Owner after this turn completes.",
        {
            "type": "object",
            "properties": {"content": {
                "type": "string", "minLength": 1, "maxLength": 4096}},
            "required": ["content"], "additionalProperties": False,
        }, "owner", deliver)


class OwnerTransport:
    def __init__(self, outcomes=()):
        self.outcomes = list(outcomes)
        self.messages = []

    async def send_text(self, content):
        self.messages.append(content)
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome


class OutputCapabilityTests(unittest.IsolatedAsyncioTestCase):
    def runtime(self, temporary, provider, *, outputs=(), transport=None, **config):
        return ResidentRuntime(
            Config(Path(temporary), **config), provider, capabilities=[],
            output_capabilities=outputs, owner_transport=transport,
            owner_output=lambda _: None, diagnostic_output=lambda _: None)

    def test_zero_output_capabilities_produce_valid_silent_only_schema(self):
        schema = output_schema([])
        outputs = schema["properties"]["outputs"]
        self.assertEqual(0, outputs["maxItems"])
        self.assertEqual([{
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }], outputs["items"]["anyOf"])
        self.assertIsNone(validate_disposition({"outputs": []}, schema))
        self.assertIsNotNone(validate_disposition(
            {"outputs": [{"type": "display", "content": "not authorized"}]},
            schema))

    def test_output_schema_fingerprint_is_deterministic(self):
        schema = output_schema([])
        reordered = {
            "additionalProperties": schema["additionalProperties"],
            "required": schema["required"],
            "properties": schema["properties"],
            "type": schema["type"],
        }
        self.assertEqual(schema_fingerprint(schema), schema_fingerprint(reordered))

    def test_nonempty_output_schema_shape_and_behavior_are_unchanged(self):
        capability = display_capability("display1", [], max_length=40)
        schema = output_schema([capability])
        outputs = schema["properties"]["outputs"]
        self.assertEqual(8, outputs["maxItems"])
        self.assertEqual([capability.schema_branch()], outputs["items"]["anyOf"])
        self.assertIsNone(validate_disposition({"outputs": [{
            "type": "display", "target": "display1", "content": "hello",
        }]}, schema))
        self.assertIsNotNone(validate_disposition({"outputs": [{
            "type": "display", "target": "display1", "content": "x" * 41,
        }]}, schema))

    async def test_silent_disposition_is_durable_and_dispatches_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            provider = StructuredProvider(['{"outputs":[]}'])
            runtime = self.runtime(
                temporary, provider, outputs=(), owner_communication_enabled=False)
            await runtime.process(WakeEvent("wake", "runtime", "test", utc_now(), {}))
            row = runtime.store.connection.execute(
                "SELECT validation_state,normalized_disposition_json FROM final_dispositions"
            ).fetchone()
            self.assertEqual("valid", row["validation_state"])
            self.assertEqual({"outputs": []}, json.loads(row["normalized_disposition_json"]))
            self.assertFalse(await runtime.dispatch_outputs_once())
            runtime.close()

    async def test_notify_owner_question_is_queued_then_later_reply_is_a_new_wake(self):
        with tempfile.TemporaryDirectory() as temporary:
            provider = StructuredProvider([
                '{"outputs":[{"type":"notify_owner","content":"Open. Close it?"}]}',
                '{"outputs":[]}',
            ])
            transport = OwnerTransport()
            runtime = self.runtime(temporary, provider, outputs=(), transport=transport)
            await runtime.process(WakeEvent("wake-1", "sensor", "changed", utc_now(), {}))
            self.assertEqual([], transport.messages)
            self.assertTrue(await runtime.dispatch_outputs_once())
            self.assertEqual(["Open. Close it?"], transport.messages)
            await runtime.process(runtime.owner_message_event("Yes"))
            self.assertEqual("Yes", provider.contexts[-1]["wake_event"]["payload"]["content"])
            runtime.close()

    async def test_display_and_multiple_outputs_preserve_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            delivered = []
            outputs = (display_capability("display1", delivered),
                       display_capability("display2", delivered, max_length=120))
            provider = StructuredProvider([json.dumps({"outputs": [
                {"type": "display", "target": "display1", "content": "one"},
                {"type": "display", "target": "display2", "content": "two"},
            ]})])
            runtime = self.runtime(
                temporary, provider, outputs=outputs, owner_communication_enabled=False)
            await runtime.process(WakeEvent("wake", "sensor", "changed", utc_now(), {}))
            self.assertTrue(await runtime.dispatch_outputs_once())
            self.assertTrue(await runtime.dispatch_outputs_once())
            self.assertEqual([("display1", "one"), ("display2", "two")], delivered)
            self.assertEqual(0, runtime.store.connection.execute(
                "SELECT count(*) FROM scheduled_wakeups").fetchone()[0])
            runtime.close()


    async def test_target_specific_max_length_and_invalid_json_are_receipted(self):
        for raw, expected in (
                ('{"outputs":[{"type":"display","target":"display1","content":"toolong"}]}',
                 "schema_invalid"),
                ("not-json", "invalid_json")):
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as temporary:
                runtime = self.runtime(
                    temporary, StructuredProvider(),
                    outputs=(display_capability("display1", [], max_length=3),),
                    owner_communication_enabled=False)
                with self.assertRaises(RuntimeError):
                    runtime._persist_disposition(
                        raw, "session-1", "turn-1", run_id=None, wake=None)
                state = runtime.store.connection.execute(
                    "SELECT validation_state FROM final_dispositions").fetchone()[0]
                self.assertEqual(expected, state)
                runtime.close()

    async def test_maximum_eight_outputs_is_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = self.runtime(
                temporary, StructuredProvider(),
                outputs=(display_capability("display1", []),),
                owner_communication_enabled=False)
            raw = json.dumps({"outputs": [
                {"type": "display", "target": "display1", "content": str(index)}
                for index in range(9)]})
            with self.assertRaises(RuntimeError):
                runtime._persist_disposition(
                    raw, "session", "turn", run_id=None, wake=None)
            self.assertEqual("schema_invalid", runtime.store.connection.execute(
                "SELECT validation_state FROM final_dispositions").fetchone()[0])
            runtime.close()

    async def test_revoked_output_is_rejected_by_current_authorization(self):
        with tempfile.TemporaryDirectory() as temporary:
            old = display_capability("retired", [], max_length=20)
            old_schema = output_schema([old])
            runtime = self.runtime(
                temporary, StructuredProvider(), outputs=(),
                owner_communication_enabled=False)
            runtime._persist_disposition(
                '{"outputs":[{"type":"display","target":"retired","content":"x"}]}',
                "old-session", "turn", run_id=None, wake=None,
                schema=old_schema, fingerprint=schema_fingerprint(old_schema))
            row = runtime.store.connection.execute(
                "SELECT delivery_state,last_failure_classification FROM output_requests"
            ).fetchone()
            self.assertEqual(("rejected_unavailable", "capability_revoked"), tuple(row))
            runtime.close()

    async def test_duplicate_disposition_creates_one_job(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = display_capability("display1", [])
            runtime = self.runtime(
                temporary, StructuredProvider(), outputs=(output,),
                owner_communication_enabled=False)
            raw = '{"outputs":[{"type":"display","target":"display1","content":"x"}]}'
            first = runtime._persist_disposition(raw, "session", "turn", run_id=None, wake=None)
            replay = runtime._persist_disposition(raw, "session", "turn", run_id=None, wake=None)
            self.assertTrue(first["created"])
            self.assertFalse(replay["created"])
            self.assertEqual(1, runtime.store.connection.execute(
                "SELECT count(*) FROM output_requests").fetchone()[0])
            runtime.close()


    async def test_retry_then_success_and_dispatcher_restart_recovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            attempts = 0

            async def flaky(_):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise OSError("temporary")
                return {}

            output = display_capability("display1", [], handler=flaky)
            runtime = self.runtime(
                temporary, StructuredProvider(), outputs=(output,),
                owner_communication_enabled=False)
            raw = '{"outputs":[{"type":"display","target":"display1","content":"x"}]}'
            runtime._persist_disposition(raw, "session", "turn", run_id=None, wake=None)
            await runtime.dispatch_outputs_once()
            runtime.store.connection.execute(
                "UPDATE output_requests SET next_attempt_at=?", (utc_now(),))
            runtime.store.connection.commit()
            await runtime.dispatch_outputs_once()
            self.assertEqual("accepted_by_transport", runtime.store.connection.execute(
                "SELECT delivery_state FROM output_requests").fetchone()[0])

            runtime._persist_disposition(raw, "session", "turn-2", run_id=None, wake=None)
            claimed = runtime.store.claim_output_request()
            self.assertEqual("attempting", runtime.store.output_request(claimed["id"])["delivery_state"])
            runtime.close()
            reopened = Store(Path(temporary) / "resident.sqlite3")
            self.assertEqual("retry_wait", reopened.output_request(claimed["id"])["delivery_state"])
            reopened.close()

    async def test_interrupted_attempts_retry_only_below_persisted_maximum(self):
        for interrupted_attempt, max_attempts, expected_state in (
                (1, 3, "retry_wait"), (2, 3, "retry_wait"),
                (3, 3, "failed_permanent"), (2, 2, "failed_permanent")):
            with self.subTest(attempt=interrupted_attempt, maximum=max_attempts), \
                    tempfile.TemporaryDirectory() as temporary:
                output = display_capability(
                    "display1", [], max_attempts=max_attempts)
                runtime = self.runtime(
                    temporary, StructuredProvider(), outputs=(output,),
                    owner_communication_enabled=False)
                receipt = runtime._persist_disposition(
                    ('{"outputs":[{"type":"display","target":"display1",'
                     '"content":"x"}]}'),
                    "session", "turn", run_id=None, wake=None)
                output_id = receipt["requests"][0]["id"]
                for attempt in range(1, interrupted_attempt + 1):
                    claimed = runtime.store.claim_output_request()
                    self.assertEqual(attempt, claimed["attempt_count"])
                    if attempt < interrupted_attempt:
                        runtime.store.finish_output_attempt(
                            output_id, attempt, "retry_wait",
                            classification="OSError", retry_delay_seconds=0)
                runtime.close()

                reopened = Store(Path(temporary) / "resident.sqlite3")
                request = reopened.output_request(output_id)
                self.assertEqual(max_attempts, request["max_attempts"])
                self.assertEqual(expected_state, request["delivery_state"])
                if interrupted_attempt < max_attempts:
                    retry = reopened.claim_output_request()
                    self.assertEqual(interrupted_attempt + 1, retry["attempt_count"])
                else:
                    self.assertIsNone(reopened.claim_output_request())
                    self.assertEqual("retries_exhausted",
                                     request["last_failure_classification"])
                reopened.close()

    async def test_maxed_interrupted_attempt_terminalizes_once_without_fourth_call(self):
        with tempfile.TemporaryDirectory() as temporary:
            calls = 0

            async def handler(_):
                nonlocal calls
                calls += 1
                return {}

            output = display_capability(
                "display1", [], handler=handler, max_attempts=3)
            runtime = self.runtime(
                temporary, StructuredProvider(), outputs=(output,),
                owner_communication_enabled=False)
            receipt = runtime._persist_disposition(
                ('{"outputs":[{"type":"display","target":"display1",'
                 '"content":"private"}]}'),
                "session", "turn", run_id=None, wake=None)
            output_id = receipt["requests"][0]["id"]
            for attempt in range(1, 4):
                runtime.store.claim_output_request()
                if attempt < 3:
                    runtime.store.finish_output_attempt(
                        output_id, attempt, "retry_wait",
                        classification="OSError", retry_delay_seconds=0)
            runtime.close()

            for _ in range(2):
                reopened = Store(Path(temporary) / "resident.sqlite3")
                self.assertEqual("failed_permanent",
                                 reopened.output_request(output_id)["delivery_state"])
                schedules = reopened.connection.execute(
                    "SELECT reason,context_json FROM scheduled_wakeups").fetchall()
                self.assertEqual(1, len(schedules))
                self.assertEqual("output_delivery_failed", schedules[0]["reason"])
                self.assertNotIn("private", schedules[0]["context_json"])
                reopened.close()

            recovered = self.runtime(
                temporary, StructuredProvider(), outputs=(output,),
                owner_communication_enabled=False)
            self.assertFalse(await recovered.dispatch_outputs_once())
            self.assertEqual(0, calls)
            recovered.close()

    async def test_permanent_failure_generates_exactly_one_safe_wake_and_success_none(self):
        with tempfile.TemporaryDirectory() as temporary:
            async def rejected(_):
                raise ValueError("secret raw body")

            output = display_capability("display1", [], handler=rejected)
            runtime = self.runtime(
                temporary, StructuredProvider(), outputs=(output,),
                owner_communication_enabled=False)
            raw = '{"outputs":[{"type":"display","target":"display1","content":"private"}]}'
            receipt = runtime._persist_disposition(
                raw, "session", "turn", run_id=None, wake=None)
            await runtime.dispatch_outputs_once()
            runtime._record_terminal_output_failure(
                {"id": receipt["requests"][0]["id"], "output_type": "display",
                 "target": "display1"}, "ValueError", 1)
            schedules = runtime.store.connection.execute(
                "SELECT reason,context_json FROM scheduled_wakeups").fetchall()
            self.assertEqual(1, len(schedules))
            self.assertEqual("output_delivery_failed", schedules[0]["reason"])
            self.assertNotIn("private", schedules[0]["context_json"])
            self.assertNotIn("secret", schedules[0]["context_json"])
            operational = " ".join(row[0] for row in runtime.store.connection.execute("""
                SELECT data_json FROM journal WHERE event_type LIKE 'output.%'
                   OR event_type='disposition.generated'
            """))
            self.assertNotIn("private", operational)
            self.assertNotIn("secret", operational)
            runtime.close()

    async def test_attention_budget_rejection_is_persisted_without_dispatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = self.runtime(
                temporary, StructuredProvider(), outputs=(), transport=OwnerTransport(),
                spontaneous_message_limit=0)
            raw = '{"outputs":[{"type":"notify_owner","content":"unsolicited"}]}'
            runtime._persist_disposition(
                raw, "session", "turn", run_id=None,
                wake=WakeEvent("wake", "sensor", "changed", utc_now(), {}))
            self.assertEqual("rejected_policy", runtime.store.connection.execute(
                "SELECT delivery_state FROM output_requests").fetchone()[0])
            self.assertEqual("rejected_attention_budget", runtime.store.connection.execute(
                "SELECT delivery_status FROM messages").fetchone()[0])
            self.assertFalse(await runtime.dispatch_outputs_once())
            runtime.close()

    async def test_interactive_tool_round_precedes_final_disposition(self):
        with tempfile.TemporaryDirectory() as temporary:
            provider = StructuredProvider([
                ModelTurn("turn", tool_calls=(ToolCall("call", "diagnostics_current_time", {}),)),
                '{"outputs":[]}',
            ])
            runtime = ResidentRuntime(
                Config(Path(temporary), owner_communication_enabled=False), provider,
                output_capabilities=(), owner_output=lambda _: None,
                diagnostic_output=lambda _: None)
            await runtime.process(WakeEvent("wake", "runtime", "test", utc_now(), {}))
            self.assertEqual(2, provider.round)
            self.assertEqual(1, runtime.store.connection.execute(
                "SELECT count(*) FROM final_dispositions").fetchone()[0])
            runtime.close()


if __name__ == "__main__":
    unittest.main()
