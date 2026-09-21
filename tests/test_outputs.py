from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from resident.config import Config
from resident.domain import ModelTurn, ToolCall, WakeEvent
from resident.outputs import DeliveryPolicy, OutputCapability, output_schema, schema_fingerprint
from resident.runtime import ResidentRuntime
from resident.store import Store, utc_now


class StructuredProvider:
    uses_managed_session = True
    supports_output_capabilities = True
    session_protocol_known = True
    rollover_ready = False

    def __init__(self, messages=(), *, session_id="session-1", recovery=None):
        self.session_id = session_id
        self.messages = list(messages)
        self.recovery = recovery
        self.contexts = []
        self.tool_sets = []
        self.schema = None
        self.fingerprint = None
        self.descriptors = None
        self.round = 0
        self.applied_output_protocol = True

    @property
    def session_uses_output_capabilities(self):
        return self.schema is not None and self.applied_output_protocol

    @property
    def active_output_protocol(self):
        return {"schema": self.schema, "fingerprint": self.fingerprint,
                "capabilities": self.descriptors}

    def configure_output_protocol(self, schema, descriptors, fingerprint):
        self.schema, self.descriptors, self.fingerprint = schema, descriptors, fingerprint

    async def preflight_session(self):
        return None

    async def respond(self, context, tools, results, previous_response_id=None):
        self.contexts.append(json.loads(context))
        self.tool_sets.append([tool.name for tool in tools])
        self.round += 1
        value = self.messages.pop(0)
        if isinstance(value, ModelTurn):
            return value
        return ModelTurn(f"turn-{self.round}", value)

    async def recover_final_output(self, session_id, turn_id):
        self.recovery = (session_id, turn_id, self.recovery)
        return self.recovery[2]


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
        delivery_policy=DeliveryPolicy(max_attempts=max_attempts),
        legacy_tool_name=f"{target}_show_text")


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

    async def test_restart_recovers_bound_completed_turn_without_duplicate(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "resident.sqlite3"
            output = display_capability("display1", [])
            schema = output_schema([output])
            fingerprint = schema_fingerprint(schema)
            store = Store(path)
            store.provision("Resident", "Owner", "")
            store.save_agent_session_binding("openai_agents", "session", None, "turn")
            store.mark_agent_wake_submission_attempted(
                "openai_agents", "session", "wake", "correlation",
                wake_id="wake", wake_source="sensor", wake_reason="changed")
            store.correlate_agent_wake_submission(
                "openai_agents", "session", "wake", "turn")
            store.settle_agent_wake_submission("openai_agents", "session", "turn")
            store.save_session_protocol("openai_agents", "session", {
                "version": 2, "output_schema": schema,
                "output_schema_fingerprint": fingerprint,
                "output_capabilities": [output.semantic_descriptor()]})
            store.close()
            raw = '{"outputs":[{"type":"display","target":"display1","content":"x"}]}'
            provider = StructuredProvider(session_id="session", recovery=raw)
            runtime = self.runtime(
                temporary, provider, outputs=(output,), owner_communication_enabled=False)
            self.assertTrue(await runtime.recover_missing_disposition())
            self.assertFalse(await runtime.recover_missing_disposition())
            self.assertEqual(1, runtime.store.connection.execute(
                "SELECT count(*) FROM output_requests").fetchone()[0])
            runtime.close()

    async def test_recovered_owner_wake_retains_immediate_reply_semantics(self):
        with tempfile.TemporaryDirectory() as temporary:
            initial = self.runtime(
                temporary, StructuredProvider(), outputs=(), transport=OwnerTransport(),
                spontaneous_message_limit=0)
            initial.store.save_agent_session_binding(
                "openai_agents", "session", None, "turn")
            initial.store.save_session_protocol("openai_agents", "session", {
                "version": 2, "output_schema": initial._output_schema,
                "output_schema_fingerprint": initial._output_schema_fingerprint,
                "output_capabilities": [
                    output.semantic_descriptor() for output in initial._output_capabilities],
            })
            initial.store.mark_agent_wake_submission_attempted(
                "openai_agents", "session", "message", "correlation",
                wake_id="owner-wake", wake_source="owner", wake_reason="owner_message")
            initial.store.correlate_agent_wake_submission(
                "openai_agents", "session", "message", "turn")
            initial.store.settle_agent_wake_submission(
                "openai_agents", "session", "turn")
            initial.close()

            transport = OwnerTransport()
            provider = StructuredProvider(
                session_id="session",
                recovery='{"outputs":[{"type":"notify_owner","content":"reply"}]}')
            recovered = self.runtime(
                temporary, provider, outputs=(), transport=transport,
                spontaneous_message_limit=0)
            self.assertTrue(await recovered.recover_missing_disposition())
            request = recovered.store.connection.execute(
                "SELECT delivery_state FROM output_requests").fetchone()
            message = recovered.store.connection.execute(
                "SELECT spontaneous,delivery_status FROM messages").fetchone()
            self.assertEqual("queued", request["delivery_state"])
            self.assertEqual((0, "pending_delivery"), tuple(message))
            self.assertTrue(await recovered.dispatch_outputs_once())
            self.assertEqual(["reply"], transport.messages)
            recovered.close()

    async def test_recovered_failure_wake_suppresses_recursive_failure_event(self):
        with tempfile.TemporaryDirectory() as temporary:
            async def rejected(_):
                raise ValueError("still unavailable")

            output = display_capability("display1", [], handler=rejected)
            initial = self.runtime(
                temporary, StructuredProvider(), outputs=(output,),
                owner_communication_enabled=False)
            initial.store.save_agent_session_binding(
                "openai_agents", "session", None, "turn")
            initial.store.save_session_protocol("openai_agents", "session", {
                "version": 2, "output_schema": initial._output_schema,
                "output_schema_fingerprint": initial._output_schema_fingerprint,
                "output_capabilities": [output.semantic_descriptor()],
            })
            initial.store.mark_agent_wake_submission_attempted(
                "openai_agents", "session", "failure-schedule", "correlation",
                wake_id="failure-wake", wake_source="scheduler",
                wake_reason="output_delivery_failed")
            initial.store.correlate_agent_wake_submission(
                "openai_agents", "session", "failure-schedule", "turn")
            initial.store.settle_agent_wake_submission(
                "openai_agents", "session", "turn")
            initial.close()

            recovered = self.runtime(
                temporary, StructuredProvider(
                    session_id="session",
                    recovery=('{"outputs":[{"type":"display","target":"display1",'
                              '"content":"fallback"}]}')),
                outputs=(output,), owner_communication_enabled=False)
            self.assertTrue(await recovered.recover_missing_disposition())
            self.assertEqual(1, recovered.store.connection.execute(
                "SELECT failure_event_generated FROM output_requests").fetchone()[0])
            self.assertTrue(await recovered.dispatch_outputs_once())
            self.assertEqual("failed_permanent", recovered.store.connection.execute(
                "SELECT delivery_state FROM output_requests").fetchone()[0])
            self.assertEqual(0, recovered.store.connection.execute(
                "SELECT count(*) FROM scheduled_wakeups").fetchone()[0])
            recovered.close()

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

    def test_managed_agent_uses_validated_text_format_and_protocol_fingerprint(self):
        from resident.provider import OpenAIAgentsProvider
        from resident.store import _create_request_configuration
        provider = OpenAIAgentsProvider("key", "gpt-5.6-luna")
        cap = display_capability("display1", [], max_length=40)
        schema = output_schema([cap])
        fingerprint = schema_fingerprint(schema)
        provider.configure_output_protocol(schema, [cap.semantic_descriptor()], fingerprint)
        agent = provider._agent_config([])
        self.assertEqual({
            "type": "json_schema", "name": "resident_final_disposition",
            "schema": schema, "strict": True,
        }, agent["text"]["format"])
        protocol = provider._agent_protocol(agent)
        self.assertEqual(fingerprint, protocol["output_schema_fingerprint"])
        reconstructed, _ = _create_request_configuration({
            "environment": {"type": "none"}, "agent": agent})
        self.assertEqual(protocol, reconstructed)
        empty_schema = output_schema([])
        provider.configure_output_protocol(
            empty_schema, [], schema_fingerprint(empty_schema))
        empty_agent = provider._agent_config([])
        empty_reconstructed, _ = _create_request_configuration({
            "environment": {"type": "none"}, "agent": empty_agent})
        self.assertEqual(provider._agent_protocol(empty_agent), empty_reconstructed)

    def test_output_schema_change_requires_rollover(self):
        from resident.provider import OpenAIAgentsProvider
        provider = OpenAIAgentsProvider("key", "gpt-5.6-luna")
        first = display_capability("display1", [], max_length=40)
        first_schema = output_schema([first])
        provider.configure_output_protocol(
            first_schema, [first.semantic_descriptor()], schema_fingerprint(first_schema))
        provider._session_id = "session"
        provider._protocol_descriptor = provider._agent_protocol(provider._agent_config([]))
        second = display_capability("display1", [], max_length=120)
        second_schema = output_schema([second])
        provider.configure_output_protocol(
            second_schema, [second.semantic_descriptor()], schema_fingerprint(second_schema))
        self.assertTrue(provider.protocol_change_requires_rollover([]))

    def test_busy_old_session_keeps_legacy_paths_for_recovery(self):
        from resident.capabilities import Capability

        async def legacy(_):
            return {}

        with tempfile.TemporaryDirectory() as temporary:
            provider = StructuredProvider()
            provider.applied_output_protocol = False
            legacy_display = Capability(
                "display", "legacy", "display1_show_text", "legacy",
                {"type": "object", "properties": {"text": {"type": "string"}},
                 "required": ["text"], "additionalProperties": False}, legacy)
            runtime = ResidentRuntime(
                Config(Path(temporary)), provider, capabilities=[legacy_display],
                output_capabilities=(display_capability("display1", []),),
                owner_output=lambda _: None, diagnostic_output=lambda _: None)
            self.assertFalse(runtime._uses_structured_output_protocol())
            names = [item.name for item in runtime._tool_capabilities_for_protocol(False)]
            self.assertIn("display1_show_text", names)
            runtime.close()


if __name__ == "__main__":
    unittest.main()
