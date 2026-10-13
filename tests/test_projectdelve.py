import asyncio
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from resident.capabilities import Capability
from resident.config import Config
from resident.domain import ModelTurn, ToolCall, WakeEvent
from resident.instances import load_resident_definition, load_resident_catalog
from resident.outputs import validate_disposition
from resident.projectdelve import ProjectDelveConnector
from resident.runtime import ResidentRuntime
from resident.provider import ResponseInvalid, ResponseRejected, ResponseUnavailable
from resident.store import Store, utc_now
from runtime_support import RecordingProvider


def observation():
    return {"schemaVersion": 3, "binding": {"sessionId": "session", "bindingId": "gambit",
        "sideId": "blue", "decisionId": "decision", "revision": 7, "attemptId": "attempt",
        "cursorStart": 2, "cursorEnd": 8}, "objective": {"sideId": "blue", "objective": "EliminateEnemies"},
        "status": "waiting-external", "completion": None,
        "request": {"kind": "Activation", "typeId": "Shaman", "unitId": "shaman",
                    "candidates": [{"key": "move-a"}, {"key": "end-turn"}], "allowsNone": True},
        "defaultBehaviorPreviews": [],
        "events": [{"type": "Moved"}], "observation": {"state": {"round": 1},
        "activation": {"currentUnitId": "other"}, "units": [
            {"id": "shaman", "typeId": "Shaman", "sideId": "blue", "agency": "Direct"},
            {"id": "shaman-two", "typeId": "Shaman", "sideId": "blue", "agency": "Direct"},
            {"id": "goblin", "typeId": "Goblin", "sideId": "blue", "agency": "Allied"},
            {"id": "grunt", "typeId": "Grunt", "sideId": "red", "agency": "Hostile"},
        ]}}


class FakeConnector(ProjectDelveConnector):
    def __init__(self):
        super().__init__("http://127.0.0.1:5080", "secret-marker")
        self.observation = observation()
        self.requests = []
        self.receipts = {}
        self.available = True
        self.admission = True
        self.lost_ack = False
        self.failed_receipt_read = False
        self.before_admission = None

    async def _http(self, route, payload=None):
        self.requests.append((route, copy.deepcopy(payload)))
        if not self.available:
            raise OSError("unavailable")
        if route == "/status":
            completed = self.observation["completion"] is not None
            return {"schemaVersion": 3, "sessionId": self.observation["binding"]["sessionId"],
                    "bindingId": "gambit", "sideId": self.observation["binding"]["sideId"],
                    "status": "completed" if completed else "waiting-external",
                    "completion": self.observation["completion"],
                    "request": None if completed else self.observation["binding"],
                    "objective": self.observation["objective"]}
        if route == "/request":
            return copy.deepcopy(self.observation)
        if route == "/admit":
            if self.before_admission:
                self.before_admission()
            return {"admitted": self.admission, "binding": payload}
        if route.startswith("/receipts/"):
            if self.failed_receipt_read:
                self.failed_receipt_read = False
                raise OSError("receipt unavailable")
            output_id = route.split("/")[-1]
            return self.receipts.get(output_id, {"submissionId": output_id, "status": "unknown", "error": None})
        if route == "/selection":
            output_id = payload["submissionId"]
            value = {"submissionId": output_id, "status": "applied", "error": None}
            self.receipts[output_id] = value
            self.observation["completion"] = {"achievedSideIds": ["blue"]}
            if self.lost_ack:
                raise OSError("lost acknowledgement")
            return value
        if route == "/error":
            return {"accepted": True}
        raise AssertionError(route)


def choice(key="move-a", commentary=None, activation_control="Direct"):
    return json.dumps({"outputs": [{"type": "projectdelve_decision", "candidateKey": key,
                                   "activationControl": activation_control,
                                   "commentary": commentary}]})


class ProjectDelveTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.connector = FakeConnector()
        self.provider = RecordingProvider([choice()])
        self.runtime = ResidentRuntime(Config(Path(self.temp.name)), self.provider,
            projectdelve_client=self.connector, output_capabilities=self.connector.output_capabilities,
            owner_output_enabled=False, diagnostic_output=lambda _: None)

    def tearDown(self):
        self.runtime.close()
        self.temp.cleanup()

    async def event(self):
        queue = asyncio.Queue()
        await self.connector.poll_once(queue)
        return queue.get_nowait()

    def output(self):
        row = self.runtime.store.connection.execute("SELECT * FROM output_requests").fetchone()
        return dict(row) if row else None

    async def force_dispatch(self):
        self.runtime.store.connection.execute("UPDATE output_requests SET next_attempt_at=NULL")
        self.runtime.store.connection.commit()
        return await self.runtime.dispatch_outputs_once()

    async def test_winning_final_output_has_no_tools_or_continuation_and_immutable_identity(self):
        event = await self.event()
        original = copy.deepcopy(event.payload["observation"]["binding"])
        await self.runtime.process(event)
        output = self.output()
        self.assertEqual(original, json.loads(output["payload_json"])["binding"])
        self.assertEqual([], self.provider.tool_sets[0])
        self.assertEqual([], self.provider.contexts[0]["pending_intentions"])
        self.assertTrue(self.provider.contexts[0]["wake_event"]["payload"]["newSession"])
        self.assertNotIn("secret-marker", self.provider.requests[0]["context"])
        self.assertNotIn(self.connector.base_url, self.provider.requests[0]["context"])
        self.assertNotIn("transportFingerprint", self.provider.requests[0]["context"])
        self.assertNotIn("transportIdentity", self.provider.requests[0]["context"])
        self.connector.observation["binding"]["attemptId"] = "new-attempt"
        await self.force_dispatch()
        self.assertEqual("applied", self.output()["delivery_state"])
        sent = next(payload for route, payload in self.connector.requests if route == "/selection")
        self.assertEqual(original, sent["binding"])
        self.assertEqual("Direct", sent["activationControl"])
        self.assertEqual(1, len(self.provider.requests))
        queue = asyncio.Queue()
        await self.connector.poll_once(queue)
        reflection = queue.get_nowait()
        self.assertEqual("reflection", reflection.reason)

    async def test_completion_queues_one_stable_reflection_with_authoritative_result(self):
        self.connector.observation["completion"] = {"achievedSideIds": ["blue"]}
        queue = asyncio.Queue()

        await self.connector.poll_once(queue)
        event = queue.get_nowait()

        self.assertEqual("projectdelve", event.source)
        self.assertEqual("reflection", event.reason)
        self.assertTrue(event.id.startswith("projectdelve:reflection:"))
        self.assertEqual({
            "schemaVersion": 3,
            "sessionId": "session",
            "sideId": "blue",
            "objective": {"sideId": "blue", "objective": "EliminateEnemies"},
            "result": {"achievedSideIds": ["blue"]},
        }, event.payload)
        await self.connector.poll_once(queue)
        self.assertTrue(queue.empty())

    async def test_human_only_completion_without_gambit_side_is_ignored(self):
        status = {"schemaVersion": 3, "sessionId": "human-session", "bindingId": "gambit",
                  "sideId": None, "status": "completed", "completion": {"achievedSideIds": ["blue"]},
                  "request": None, "objective": None}
        self.connector._http = AsyncMock(return_value=status)
        queue = asyncio.Queue()

        await self.connector.poll_once(queue)

        self.assertTrue(queue.empty())
        self.connector._http.assert_awaited_once_with("/status")

    async def test_reflection_is_ordinary_tool_enabled_wake_and_deduplicates_after_restart(self):
        sent = []

        async def messaging_send(arguments):
            sent.append(arguments)
            return {"message_id": "report", "status": "pending", "expires_at": None}

        messaging = Capability(
            "messaging", "mailbox", "messaging_send", "send", {
                "type": "object", "properties": {
                    "recipient": {"type": "string", "enum": ["watcher"]},
                    "content": {"type": "string"},
                    "ttl_seconds": {"type": ["integer", "null"]},
                }, "required": ["recipient", "content"], "additionalProperties": False,
            }, messaging_send)
        self.runtime.replace_capabilities([messaging])
        self.connector.observation["completion"] = {"achievedSideIds": ["red"]}
        self.provider.messages = [
            ModelTurn("reflection-tool", tool_calls=(ToolCall("send-report", "messaging_send", {
                "recipient": "watcher", "content": "I lost; my lesson is uncertain.",
                "ttl_seconds": None,
            }),)),
            '{"outputs":[]}',
        ]
        event = await self.event()

        await self.runtime.process(event)

        self.assertTrue(all("messaging_send" in tools for tools in self.provider.tool_sets))
        self.assertEqual([{"recipient": "watcher", "content": "I lost; my lesson is uncertain.",
                           "ttl_seconds": None}], sent)
        self.assertEqual(0, self.provider.requests[0]["output_schema"]["properties"]["outputs"]["maxItems"])
        replacement = FakeConnector()
        replacement.observation["completion"] = {"achievedSideIds": ["red"]}
        replacement.bind_store(self.runtime.store)
        queue = asyncio.Queue()
        await replacement.poll_once(queue)
        self.assertTrue(queue.empty())

    async def test_model_context_uses_compact_json_without_changing_observation(self):
        event = await self.event()
        expected = copy.deepcopy(event.payload["observation"])

        await self.runtime.process(event)

        context = self.provider.requests[0]["context"]
        decoded = json.loads(context)
        self.assertEqual(
            json.dumps(decoded, ensure_ascii=False, separators=(",", ":")),
            context)
        self.assertEqual(
            expected,
            decoded["wake_event"]["payload"]["observation"])

    async def test_exactly_one_nullable_choice_and_commentary_are_validated(self):
        event = await self.event()
        schema = self.connector.decision_schema(event)
        key_schema = schema["properties"]["outputs"]["items"]["anyOf"][0]["properties"]["candidateKey"]
        control_schema = schema["properties"]["outputs"]["items"]["anyOf"][0]["properties"]["activationControl"]
        self.assertEqual(["move-a", "end-turn", None], key_schema["enum"])
        self.assertEqual(["Direct"], control_schema["enum"])
        for invalid in ({"outputs": []}, {"outputs": [json.loads(choice())["outputs"][0]] * 2},
                        json.loads(choice(123)),
                        json.loads(choice("move-a", {})), json.loads(choice("move-a", "x" * 281)),
                        json.loads(choice("move-a", activation_control="Delegate"))):
            self.assertIsNotNone(validate_disposition(invalid, schema))
        self.assertIsNotNone(self.connector.validate_choice(event, json.loads(choice("invented"))))
        self.assertIsNone(self.connector.validate_choice(event, json.loads(choice("move-a"))))
        self.assertIsNone(validate_disposition(json.loads(choice(None)), schema))
        event.payload["observation"]["request"]["allowsNone"] = False
        required_schema = self.connector.decision_schema(event)
        required_key_schema = required_schema["properties"]["outputs"]["items"]["anyOf"][0]["properties"]["candidateKey"]
        self.assertEqual(["move-a", "end-turn"], required_key_schema["enum"])
        self.assertIsNotNone(validate_disposition(json.loads(choice(None)), required_schema))
        self.assertIsNone(validate_disposition(json.loads(choice("move-a", "optional note")), required_schema))

    async def test_select_unit_allows_delegation_and_validates_advisory_previews(self):
        self.connector.observation["request"] = {
            "kind": "SelectUnit", "typeId": "Shaman", "unitId": None,
            "candidates": [{"key": "shaman"}, {"key": "shaman-two"}], "allowsNone": False,
        }
        preview = {"unitId": "shaman", "advisory": True, "steps": [
            {"decisionKind": "Activation", "candidateKey": "stay",
             "intendedAction": "Stay in place"},
            {"decisionKind": "Activation", "candidateKey": "attack:grunt",
             "intendedAction": "Attack grunt"},
        ], "stopReason": "DiceRequired"}
        second_preview = {"unitId": "shaman-two", "advisory": True, "steps": [],
                          "stopReason": "NoDecision"}
        self.connector.observation["defaultBehaviorPreviews"] = [preview, second_preview]
        event = await self.event()
        schema = self.connector.decision_schema(event)
        branches = schema["properties"]["outputs"]["items"]["anyOf"]
        self.assertEqual([["Direct"], ["Delegate"]], [
            branch["properties"]["activationControl"]["enum"] for branch in branches])
        self.assertEqual(["shaman", "shaman-two"],
                         branches[1]["properties"]["candidateKey"]["enum"])
        delegated = json.loads(choice("shaman", "I trust its instincts here.", "Delegate"))
        self.assertIsNone(validate_disposition(delegated, schema))
        self.assertIsNone(self.connector.validate_choice(event, delegated))
        self.assertIsNotNone(validate_disposition(
            json.loads(choice(None, activation_control="Delegate")), schema))
        self.assertEqual("Delegation requires selecting a Unit", self.connector.validate_choice(
            event, json.loads(choice(None, activation_control="Delegate"))))

        valid = copy.deepcopy(self.connector.observation)
        for mutate in (
            lambda value: value.pop("defaultBehaviorPreviews"),
            lambda value: value["defaultBehaviorPreviews"][0].update(advisory=False),
            lambda value: value["defaultBehaviorPreviews"][0].update(stopReason="Invented"),
            lambda value: value["defaultBehaviorPreviews"][0].update(stopReason=[]),
            lambda value: value["defaultBehaviorPreviews"][0]["steps"][0].update(decisionKind={}),
            lambda value: value["defaultBehaviorPreviews"][0]["steps"][0].update(intendedAction=""),
            lambda value: value["defaultBehaviorPreviews"][0].update(unitId="other"),
        ):
            with self.subTest(mutate=mutate):
                malformed = copy.deepcopy(valid)
                mutate(malformed)
                with self.assertRaisesRegex(ValueError, "Behavior previews"):
                    self.connector.decision_schema(WakeEvent(
                        "malformed", "projectdelve", "decision", utc_now(),
                        {"observation": malformed}))

    async def test_delegation_is_forwarded_once_with_the_trusted_binding(self):
        self.connector.observation["request"] = {
            "kind": "SelectUnit", "typeId": "Shaman", "unitId": None,
            "candidates": [{"key": "shaman"}, {"key": "shaman-two"}], "allowsNone": False,
        }
        self.connector.observation["defaultBehaviorPreviews"] = [
            {"unitId": unit_id, "advisory": True, "steps": [], "stopReason": "NoDecision"}
            for unit_id in ("shaman", "shaman-two")]
        original = copy.deepcopy(self.connector.observation["binding"])
        self.provider.messages = [choice("shaman", "I delegate this activation.", "Delegate")]

        await self.runtime.process(await self.event())
        await self.force_dispatch()

        sent = next(payload for route, payload in self.connector.requests if route == "/selection")
        self.assertEqual("Delegate", sent["activationControl"])
        self.assertEqual("shaman", sent["candidateKey"])
        self.assertEqual(original, sent["binding"])
        self.assertEqual(1, len(self.provider.requests))
        self.assertEqual(1, sum(route == "/selection" for route, _ in self.connector.requests))

    async def test_active_unit_multi_choice_allows_only_null_key_delegation(self):
        self.connector.observation["observation"]["activation"]["currentUnitId"] = "shaman"
        self.connector.observation["request"]["allowsNone"] = False
        self.connector.observation["defaultBehaviorPreviews"] = [{
            "unitId": "shaman", "advisory": True, "steps": [{
                "decisionKind": "Activation", "candidateKey": "move-a",
                "intendedAction": "Move toward the enemy",
            }], "stopReason": "StepLimit",
        }]
        event = await self.event()
        schema = self.connector.decision_schema(event)
        branches = schema["properties"]["outputs"]["items"]["anyOf"]
        self.assertEqual([["Direct"], ["Delegate"]], [
            branch["properties"]["activationControl"]["enum"] for branch in branches])
        self.assertEqual(["move-a", "end-turn"],
                         branches[0]["properties"]["candidateKey"]["enum"])
        self.assertEqual([None], branches[1]["properties"]["candidateKey"]["enum"])
        delegated = json.loads(choice(None, "I leave the rest to its instincts.", "Delegate"))
        self.assertIsNone(validate_disposition(delegated, schema))
        self.assertIsNone(self.connector.validate_choice(event, delegated))
        self.assertIsNotNone(validate_disposition(
            json.loads(choice("move-a", activation_control="Delegate")), schema))
        self.assertIsNotNone(validate_disposition(
            json.loads(choice(None, activation_control="Direct")), schema))
        self.assertEqual("Mid-activation delegation requires a null candidate key",
                         self.connector.validate_choice(
                             event, json.loads(choice("move-a", activation_control="Delegate"))))
        invalid_control = json.loads(choice("move-a"))
        invalid_control["outputs"][0]["activationControl"] = []
        self.assertEqual("Activation control is not valid",
                         self.connector.validate_choice(event, invalid_control))
        self.assertEqual("Decline is not legal", self.connector.validate_choice(
            event, json.loads(choice(None, activation_control="Direct"))))
        self.assertIsNone(self.connector.validate_choice(
            event, json.loads(choice("move-a", activation_control="Direct"))))
        event.payload["observation"]["request"]["candidates"] = [
            {"key": "candidate-" + str(index)} for index in range(1500)]
        large_schema = self.connector.decision_schema(event)
        large_key = large_schema["properties"]["outputs"]["items"]["anyOf"][0]["properties"]["candidateKey"]
        self.assertNotIn("enum", large_key)
        self.assertEqual("string", large_key["type"])

    async def test_conversation_and_strategic_identity_survive_new_session(self):
        await self.runtime.process(await self.event())
        first_conversation = self.runtime.conversation_id
        self.connector.observation = observation()
        self.connector.observation["binding"].update(sessionId="second-session", attemptId="second-attempt")
        self.provider.messages = [choice("end-turn")]
        await self.runtime.process(await self.event())
        self.assertEqual(1, self.provider.created)
        self.assertTrue(all(request["conversation_id"] == first_conversation for request in self.provider.requests))
        self.assertTrue(self.provider.contexts[1]["wake_event"]["payload"]["newSession"])

    async def test_crash_after_admission_before_checkpoint_preserves_new_session_marker(self):
        self.runtime.store.save_observed_snapshot("projectdelve.session", "previous-session")
        event = await self.event()
        self.assertTrue(await self.connector.admit(event))
        self.assertTrue(event.payload["newSession"])
        self.assertEqual("previous-session", self.runtime.store.observed_snapshot("projectdelve.session"))
        # No response checkpoint exists: a restarted producer must rediscover and
        # still tell the first model request that this is a new match.
        replacement = FakeConnector()
        replacement.bind_store(self.runtime.store)
        queue = asyncio.Queue()
        await replacement.poll_once(queue)
        recovered = queue.get_nowait()
        self.runtime.projectdelve_client = replacement
        await self.runtime.process(recovered)
        self.assertTrue(self.provider.contexts[0]["wake_event"]["payload"]["newSession"])
        self.assertEqual("session", self.runtime.store.observed_snapshot("projectdelve.session"))

    async def test_malformed_status_fails_closed(self):
        for malformed in (None, {}, {"schemaVersion": True, "status": "waiting-external", "request": None, "completion": None},
                          {"schemaVersion": 3, "status": "waiting-external", "request": None}):
            with self.assertRaisesRegex(ValueError, "Malformed ProjectDelve status"):
                self.connector.validate_status(malformed)
        self.assertEqual([], self.provider.requests)

    async def test_malformed_objective_or_decline_fails_closed_before_queueing(self):
        self.connector.observation["objective"]["sideId"] = "opponent"
        with self.assertRaisesRegex(ValueError, "player observation"):
            await self.event()
        self.assertEqual([], self.provider.requests)
        self.connector.observation = observation()
        self.connector.observation["request"]["allowsNone"] = "true"
        with self.assertRaisesRegex(ValueError, "decline permission"):
            await self.event()

    async def test_v3_requires_complete_and_consistent_relative_unit_agency(self):
        for mutate, message in (
            (lambda units: units[0].pop("agency"), "unit agency"),
            (lambda units: units[0].update(agency="Human"), "unit agency"),
            (lambda units: units[2].update(sideId="red"), "Inconsistent"),
            (lambda units: units[3].update(agency="Allied"), "Inconsistent"),
            (lambda units: units[0].update(agency="Allied"), "request agency"),
        ):
            with self.subTest(message=message):
                self.connector.observation = observation()
                mutate(self.connector.observation["observation"]["units"])
                with self.assertRaisesRegex(ValueError, message):
                    await self.event()
        self.assertEqual([], self.provider.requests)

    async def test_v1_decisions_and_status_are_rejected(self):
        self.connector.observation["schemaVersion"] = 1
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            await self.event()
        status = {"schemaVersion": 1, "sessionId": "session", "bindingId": "gambit",
                  "sideId": "blue", "status": "waiting-external", "completion": None,
                  "request": self.connector.observation["binding"], "objective": self.connector.observation["objective"]}
        with self.assertRaisesRegex(ValueError, "Malformed ProjectDelve status"):
            self.connector.validate_status(status)

    async def test_large_legal_candidate_set_is_preserved_without_provider_enum_limits(self):
        self.connector.observation["request"]["candidates"] = [
            {"key": "candidate-" + str(index), "path": [{"x": index, "y": 0}]}
            for index in range(1500)]
        self.provider.messages = [choice("candidate-1499")]
        await self.runtime.process(await self.event())
        schema = self.provider.requests[0]["output_schema"]
        key_schema = schema["properties"]["outputs"]["items"]["anyOf"][0]["properties"]["candidateKey"]
        self.assertNotIn("enum", key_schema)
        observed = self.provider.contexts[0]["wake_event"]["payload"]["observation"]["request"]["candidates"]
        self.assertEqual(1500, len(observed))
        self.assertEqual("candidate-1499", json.loads(self.output()["payload_json"])["candidateKey"])

    async def test_long_candidate_keys_use_narrow_schema_fallback(self):
        self.connector.observation["request"]["candidates"] = [
            {"key": str(index) + "-" + "x" * 100} for index in range(150)]
        event = await self.event()
        key_schema = self.connector.decision_schema(event)["properties"]["outputs"]["items"]["anyOf"][0]["properties"]["candidateKey"]
        self.assertNotIn("enum", key_schema)
        self.assertIsNone(self.connector.validate_choice(event, json.loads(choice("149-" + "x" * 100))))

    async def test_commentary_matches_server_utf16_limit_before_finalization(self):
        event = await self.event()
        self.assertIsNone(self.connector.validate_choice(event, json.loads(choice(commentary="\U0001f600" * 140))))
        self.assertIsNotNone(self.connector.validate_choice(event, json.loads(choice(commentary="\U0001f600" * 141))))
        self.assertIsNotNone(self.connector.validate_choice(event, json.loads(choice(commentary="\ud800"))))
        self.provider.messages = [choice(commentary="\U0001f600" * 141)]
        with self.assertRaisesRegex(RuntimeError, "schema_invalid"):
            await self.runtime.process(event)
        self.assertIsNone(self.output())
        self.assertFalse(any(route == "/selection" for route, _ in self.connector.requests))

    async def test_invalid_final_output_pauses_and_does_not_reinfer_after_restart(self):
        self.provider.messages = [choice("invalid")]
        event = await self.event()
        with self.assertRaisesRegex(RuntimeError, "schema_invalid"):
            await self.runtime.process(event)
        self.assertIsNone(self.output())
        error = next(payload for route, payload in self.connector.requests if route == "/error")
        self.assertEqual("invalid-decision", error["error"])
        await self.runtime.process(event)
        self.assertEqual(1, len(self.provider.requests))
        replacement = FakeConnector()
        replacement.bind_store(self.runtime.store)
        queue = asyncio.Queue()
        await replacement.poll_once(queue)
        self.assertTrue(queue.empty())

    async def test_game_failure_reports_stable_safe_error_classes(self):
        cases = [
            (ResponseRejected("rate limited", 429), "model-unavailable"),
            (ResponseInvalid("private invalid detail", "response-bad"), "invalid-decision"),
            (ResponseRejected("bad request", 400), "controller-failure"),
            (ResponseUnavailable("private provider detail"), "model-unavailable"),
        ]
        for index, (failure, expected) in enumerate(cases):
            with self.subTest(expected=expected, failure=type(failure).__name__), tempfile.TemporaryDirectory() as directory:
                connector = FakeConnector()
                connector.observation["binding"].update(attemptId=f"attempt-{index}")
                provider = RecordingProvider([failure])
                runtime = ResidentRuntime(Config(Path(directory)), provider,
                    projectdelve_client=connector, output_capabilities=connector.output_capabilities,
                    owner_output_enabled=False, diagnostic_output=lambda _: None)
                queue = asyncio.Queue()
                await connector.poll_once(queue)
                try:
                    with self.assertRaises(type(failure)):
                        await runtime.process(queue.get_nowait())
                    payload = [payload for route, payload in connector.requests if route == "/error"][-1]
                    self.assertEqual(expected, payload["error"])
                    self.assertNotIn("private", json.dumps(payload))
                finally:
                    runtime.close()

    async def test_controller_transport_failure_reports_controller_failure(self):
        event = await self.event()
        self.connector.available = False
        with self.assertRaises(OSError):
            await self.runtime.process(event)
        self.connector.available = True
        payload = [payload for route, payload in self.connector.requests if route == "/error"][-1]
        self.assertEqual("controller-failure", payload["error"])

    async def test_live_reset_completion_unavailable_and_denied_admission_gate(self):
        event = await self.event()
        for mode in ("stale", "completed", "denied"):
            self.connector.observation = observation()
            self.connector.admission = mode != "denied"
            if mode == "stale":
                self.connector.observation["binding"]["revision"] += 1
            if mode == "completed":
                self.connector.observation["completion"] = {"achievedSideIds": ["blue"]}
            changed = WakeEvent(event.id + mode, event.source, event.reason, event.occurred_at, copy.deepcopy(event.payload))
            await self.runtime.process(changed)
        self.connector.available = False
        with self.assertRaises(OSError):
            await self.runtime.process(WakeEvent("unavailable", event.source, event.reason, event.occurred_at, event.payload))
        self.assertEqual([], self.provider.requests)

    async def test_startup_rediscovery_and_queue_dedup(self):
        first = await self.event()
        queue = asyncio.Queue()
        await self.connector.poll_once(queue)
        self.assertTrue(queue.empty())
        restarted = FakeConnector()
        restarted.bind_store(self.runtime.store)
        await restarted.poll_once(queue)
        self.assertEqual(first.id, queue.get_nowait().id)

    async def test_uncertain_delivery_reconciles_original_id_without_resending_or_waking_model(self):
        await self.runtime.process(await self.event())
        self.connector.lost_ack = True
        await self.force_dispatch()
        original = self.output()["id"]
        self.assertEqual("retry_wait", self.output()["delivery_state"])
        await self.force_dispatch()
        self.assertEqual("applied", self.output()["delivery_state"])
        self.assertEqual(1, sum(route == "/selection" for route, _ in self.connector.requests))
        self.assertEqual(original, self.output()["id"])
        self.assertEqual(0, self.runtime.store.connection.execute("SELECT count(*) FROM scheduled_wakeups").fetchone()[0])
        self.assertEqual(1, len(self.provider.requests))

    async def test_failed_presend_receipt_read_can_recover_without_inference(self):
        await self.runtime.process(await self.event())
        self.connector.failed_receipt_read = True
        await self.force_dispatch()
        self.assertFalse(any(route == "/selection" for route, _ in self.connector.requests))
        await self.force_dispatch()
        self.assertEqual("applied", self.output()["delivery_state"])
        self.assertEqual(1, len(self.provider.requests))

    async def test_expired_receipt_remains_unknown_and_never_blindly_resends(self):
        await self.runtime.process(await self.event())
        self.connector.lost_ack = True
        await self.force_dispatch()
        self.connector.receipts.clear()
        await self.force_dispatch()
        self.assertEqual("unknown", self.output()["last_failure_classification"])
        self.assertEqual(1, sum(route == "/selection" for route, _ in self.connector.requests))

    async def test_interrupted_delivery_restart_recovers_without_generic_failure_wake(self):
        await self.runtime.process(await self.event())
        request = self.runtime.store.claim_output_request()
        self.runtime.store.save_observed_snapshot("projectdelve.send-started." + request["id"], True)
        self.connector.receipts[request["id"]] = {"submissionId": request["id"], "status": "applied", "error": None}
        self.runtime.close()
        self.runtime = ResidentRuntime(Config(Path(self.temp.name)), self.provider,
            projectdelve_client=self.connector, output_capabilities=self.connector.output_capabilities,
            owner_output_enabled=False, diagnostic_output=lambda _: None)
        await self.force_dispatch()
        self.assertEqual("applied", self.output()["delivery_state"])
        self.assertEqual(0, self.runtime.store.connection.execute("SELECT count(*) FROM scheduled_wakeups").fetchone()[0])

    async def test_recorded_final_response_recovers_exact_bound_selection_without_inference(self):
        event = await self.event()
        await self.runtime.initialize()
        run_id = self.runtime.store.start_run(event)
        schema = self.connector.decision_schema(event)
        step = self.runtime.store.begin_response_step(run_id, self.runtime.conversation_id, 0, event, schema)
        self.runtime.store.record_response(step, ModelTurn("recorded-response", choice(None)))
        await self.runtime.recover_pending_responses()
        self.assertEqual("session", self.runtime.store.observed_snapshot("projectdelve.session"))
        payload = json.loads(self.output()["payload_json"])
        self.assertEqual(event.payload["observation"]["binding"], payload["binding"])
        self.assertIsNone(payload["candidateKey"])
        self.assertEqual([], self.provider.requests)
        await self.force_dispatch()
        self.assertEqual("applied", self.output()["delivery_state"])

    async def test_ambiguous_inference_remains_operator_blocked(self):
        event = await self.event()
        await self.runtime.initialize()
        run_id = self.runtime.store.start_run(event)
        self.runtime.store.begin_response_step(run_id, self.runtime.conversation_id, 0, event, self.connector.decision_schema(event))
        with self.assertRaisesRegex(RuntimeError, "Unfinished inference"):
            await self.runtime.process(event)
        self.assertEqual([], self.provider.requests)

    async def test_recorded_invalid_response_recovery_reports_safe_error_without_reinference(self):
        event = await self.event()
        await self.runtime.initialize()
        run_id = self.runtime.store.start_run(event)
        step = self.runtime.store.begin_response_step(run_id, self.runtime.conversation_id, 0, event,
                                                      self.connector.decision_schema(event))
        self.runtime.store.record_response(step, ModelTurn("invalid-recorded", choice("invalid")))
        with self.assertRaisesRegex(RuntimeError, "schema_invalid"):
            await self.runtime.recover_pending_responses()
        error = next(payload for route, payload in self.connector.requests if route == "/error")
        self.assertEqual("invalid-decision", error["error"])
        self.assertIsNone(self.output())
        self.assertEqual([], self.provider.requests)

    async def test_transport_change_before_recorded_response_recovery_rejects_original_choice(self):
        event = await self.event()
        await self.runtime.initialize()
        run_id = self.runtime.store.start_run(event)
        step = self.runtime.store.begin_response_step(run_id, self.runtime.conversation_id, 0, event,
                                                      self.connector.decision_schema(event))
        self.runtime.store.record_response(step, ModelTurn("old-transport-response", choice()))
        self.connector.base_url = "http://127.0.0.1:9999"
        await self.runtime.recover_pending_responses()
        self.assertEqual("rejected_unavailable", self.output()["delivery_state"])
        self.assertEqual("controller_binding_changed", self.output()["last_failure_classification"])
        self.assertFalse(await self.force_dispatch())
        self.assertEqual([], self.provider.requests)

    async def test_model_tool_call_never_executes_or_continues(self):
        self.provider.messages = [ModelTurn("tool-response", tool_calls=(ToolCall("call", "schedule_wakeup", {}),))]
        with self.assertRaisesRegex(RuntimeError, "do not permit"):
            await self.runtime.process(await self.event())
        self.assertEqual(1, len(self.provider.requests))
        with self.assertRaisesRegex(RuntimeError, "Unfinished inference"):
            await self.runtime.recover_pending_responses()

    async def test_same_session_opposing_side_and_scheduler_are_forbidden(self):
        await self.runtime.process(await self.event())
        self.connector.observation["binding"].update(sideId="red", attemptId="opponent")
        self.connector.observation["objective"]["sideId"] = "red"
        self.connector.observation["request"].update(typeId="Grunt", unitId="grunt")
        for unit in self.connector.observation["observation"]["units"]:
            unit["agency"] = "Direct" if unit["sideId"] == "red" else "Hostile"
        event = await self.event()
        with self.assertRaises(PermissionError):
            await self.runtime.process(event)
        await self.runtime.process(WakeEvent("scheduled", "scheduler", "game", utc_now(), {"context": "old plan"}))
        self.assertEqual(1, len(self.provider.requests))

    async def test_owner_wake_cannot_submit_game_output(self):
        self.provider.messages = [choice()]
        event = self.runtime.owner_message_event("Please think about the game")
        with self.assertRaisesRegex(RuntimeError, "schema_invalid"):
            await self.runtime.process(event)
        self.assertEqual(0, self.provider.requests[0]["output_schema"]["properties"]["outputs"]["maxItems"])
        self.assertIsNone(self.output())
        self.assertFalse(any(route == "/selection" for route, _ in self.connector.requests))

    async def test_cancelled_inference_stays_ambiguous_and_error_contains_no_private_text(self):
        class PendingProvider(RecordingProvider):
            def __init__(self):
                super().__init__()
                self.started = asyncio.Event()
            async def respond(self, *args, **kwargs):
                self.started.set()
                await asyncio.Event().wait()
        pending = PendingProvider()
        self.runtime.provider = pending
        task = asyncio.create_task(self.runtime.process(await self.event()))
        await pending.started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        with self.assertRaisesRegex(RuntimeError, "Unfinished inference"):
            await self.runtime.recover_pending_responses()
        errors = [payload for route, payload in self.connector.requests if route == "/error"]
        self.assertEqual("controller-failure", errors[0]["error"])
        self.assertIsNone(self.output())

    async def test_controller_configuration_change_rejects_persisted_output(self):
        await self.runtime.process(await self.event())
        self.connector.base_url = "http://127.0.0.1:9999"
        await self.force_dispatch()
        self.assertEqual("rejected_unavailable", self.output()["delivery_state"])
        self.assertFalse(any(route == "/selection" for route, _ in self.connector.requests))


class ProjectDelveDefinitionTests(unittest.TestCase):
    def test_gambit_is_enabled_in_normal_catalog_and_independently_configurable(self):
        root = Path(__file__).resolve().parent.parent
        definition = load_resident_definition(root / "residents/gambit.yaml", root / "prompts")
        self.assertEqual(("gambit", "Gambit", True), (definition.id, definition.name, definition.enabled))
        self.assertEqual(("gpt-5.6-luna", "medium"), (definition.agent.model, definition.agent.reasoning_effort))
        self.assertEqual(80000, definition.agent.compact_threshold)
        self.assertEqual(("messaging",), definition.capabilities)
        self.assertEqual(("messaging",), definition.subscriptions)
        self.assertIn("messaging_send", definition.personality)
        self.assertIn("ttl_seconds: null", definition.personality)
        self.assertIn("Direct Units are allied Units you control", definition.personality)
        self.assertIn("Allied Units share your Side but belong", definition.personality)
        self.assertIn("cooperate with them tactically but never choose or command", definition.personality)
        self.assertIn("summoner", definition.personality)
        self.assertIn("activationControl to Direct", definition.personality)
        self.assertIn("Delegate", definition.personality)
        self.assertIn("read-only advisory", definition.personality)
        self.assertIn("recomputes legal choices", definition.personality)
        self.assertIn("or delegate an activation", definition.personality)
        normal = load_resident_catalog(root / "residents", prompt_root=root / "prompts", default_id="watcher")
        self.assertEqual("watcher", normal.default_id)
        self.assertIn("gambit", [entry.id for entry in normal.residents])
        self.assertIn("watcher", [entry.id for entry in normal.residents])
        standalone = load_resident_catalog(root / "examples/gambit", prompt_root=root / "prompts", default_id="gambit")
        self.assertEqual("gambit", standalone.default_id)
        self.assertEqual(["gambit"], [definition.id for definition in standalone.residents])

    def test_gambit_rejects_non_messaging_subscriptions(self):
        root = Path(__file__).resolve().parent.parent
        source = (root / "residents/gambit.yaml").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as temporary:
            definition_path = Path(temporary) / "gambit.yaml"
            for selector in ("*", "homeops", "messaging.message_received"):
                with self.subTest(selector=selector):
                    definition_path.write_text(
                        source.replace("subscriptions: [messaging]",
                                       f"subscriptions: ['{selector}']"),
                        encoding="utf-8")
                    with self.assertRaisesRegex(
                            ValueError,
                            "only the messaging subscription and no other connectors"):
                        load_resident_definition(definition_path, root / "prompts")


class ProjectDelveHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from aiohttp import web
        self.web = web
        self.hits = []
        self.receipts = {}
        self.current = observation()
        app = web.Application()
        app.router.add_route("*", "/api/controllers/gambit/{tail:.*}", self.handle)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.url = "http://127.0.0.1:" + str(site._server.sockets[0].getsockname()[1])
        self.temp = tempfile.TemporaryDirectory()
        self.connector = ProjectDelveConnector(self.url, "http-secret-marker")
        self.provider = RecordingProvider([choice()])
        self.runtime = ResidentRuntime(Config(Path(self.temp.name)), self.provider,
            projectdelve_client=self.connector, output_capabilities=self.connector.output_capabilities,
            owner_output_enabled=False, diagnostic_output=lambda _: None)
        self.mode = None

    async def asyncTearDown(self):
        self.runtime.close()
        await self.runner.cleanup()
        self.temp.cleanup()

    async def handle(self, request):
        route = "/" + request.match_info["tail"]
        self.hits.append((route, request.headers.get("Authorization")))
        if request.headers.get("Authorization") != "Bearer http-secret-marker":
            return self.web.Response(status=403)
        if self.mode == "redirect" and route == "/status":
            raise self.web.HTTPFound(self.url + "/api/controllers/gambit/leak")
        if self.mode == "oversize" and route == "/status":
            return self.web.Response(body=b"x" * 1025)
        if self.mode == "slow" and route == "/status":
            await asyncio.sleep(0.2)
        if route == "/status":
            completed = self.current["completion"] is not None
            return self.web.json_response({"schemaVersion": 3,
                "sessionId": self.current["binding"]["sessionId"], "bindingId": "gambit",
                "sideId": self.current["binding"]["sideId"],
                "status": "completed" if completed else "waiting-external",
                "completion": self.current["completion"],
                "request": None if completed else self.current["binding"],
                "objective": self.current["objective"]})
        if route == "/request":
            return self.web.json_response(self.current)
        if route == "/admit":
            binding = await request.json()
            return self.web.json_response({"admitted": binding == self.current["binding"], "binding": binding})
        if route.startswith("/receipts/"):
            output_id = route.split("/")[-1]
            return self.web.json_response(self.receipts.get(output_id,
                {"submissionId": output_id, "status": "unknown", "error": None}))
        if route == "/selection":
            payload = await request.json()
            output_id = payload["submissionId"]
            self.receipts[output_id] = {"submissionId": output_id, "status": "applied", "error": None}
            self.current["completion"] = {"achievedSideIds": ["blue"]}
            return self.web.json_response(self.receipts[output_id])
        return self.web.json_response({})

    async def test_real_http_poll_admit_finalize_deliver_and_terminal_status(self):
        queue = asyncio.Queue()
        await self.connector.poll_once(queue)
        await self.runtime.process(queue.get_nowait())
        await self.runtime.dispatch_outputs_once()
        await self.connector.poll_once(queue)
        self.assertEqual("reflection", queue.get_nowait().reason)
        self.assertEqual(1, len(self.provider.requests))
        self.assertNotIn("http-secret-marker", json.dumps(self.provider.contexts))
        self.assertTrue(all(header == "Bearer http-secret-marker" for _, header in self.hits))
        self.assertEqual(1, sum(route == "/selection" for route, _ in self.hits))

    async def test_redirect_is_rejected_without_forwarding_controller_secret(self):
        self.mode = "redirect"
        with self.assertRaises(ValueError):
            await self.connector._http("/status")
        self.assertEqual(["/status"], [route for route, _ in self.hits])

    async def test_http_observation_overflow_and_deadline_fail_closed(self):
        self.mode = "oversize"
        with patch("resident.projectdelve.MAX_OBSERVATION_BYTES", 1024):
            with self.assertRaisesRegex(ValueError, "size limit"):
                await self.connector._http("/status")
        self.mode = "slow"
        self.connector.timeout = 0.03
        with self.assertRaises(TimeoutError):
            await self.connector._http("/status")
