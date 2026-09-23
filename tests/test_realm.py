import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from resident.realm import RealmClient, RealmHTTPError
from resident.config import Config
from resident.domain import ModelTurn, ToolResult
from resident.instances import load_resident_definition
from resident.provider import OpenAIAgentsProvider
from resident.runtime import ResidentRuntime
from resident.store import Store
from resident.tools import ToolRegistry


class RecordingProvider:
    uses_managed_session = True

    def __init__(self):
        self.session_id = None
        self.contexts = []

    async def respond(self, context, tools, results, continuation_id=None):
        import json
        self.contexts.append(json.loads(context))
        self.session_id = "new-session"
        return ModelTurn("done", message="done")


class RealmClientTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.client = RealmClient("http://127.0.0.1:3000", "game", "hero")
        self.revision = 1
        self.world_time = 0
        self.calls = []

        async def request(method, path, body=None):
            self.calls.append((method, path, body))
            if method == "GET" and path.endswith("/state?actor_id=hero"):
                return {"game": {"world_time_minutes": self.world_time},
                        "entities": [{"id": "hero", "name": "Hero"}], "facts": []}
            if method == "GET":
                return {"game": {"current_revision": self.revision,
                                 "world_time_minutes": self.world_time},
                        "entities": [{"id": "hero", "kind": "creature", "name": "Hero"},
                                     {"id": "secret", "kind": "item", "name": "Secret"}],
                        "facts": []}
            if body["expected_revision"] != self.revision:
                raise RealmHTTPError(409)
            self.revision += 1
            if path.endswith("advance-time"):
                self.world_time += body["minutes"]
            return {"revision": self.revision, "events": [], "idempotent": False}

        self.client._request = request

    async def test_snapshot_and_replacement_client_read_current_realm_state(self):
        first = await self.client.snapshot()
        self.assertNotIn("secret", str(first["player_state"]))
        self.assertIn("secret", str(first["trusted_state"]))
        await self.client.mutate("advance-time", {"minutes": 15})
        replacement = RealmClient("http://127.0.0.1:3000", "game", "hero")
        replacement._request = self.client._request
        state = await replacement.snapshot()
        self.assertEqual(2, state["trusted_state"]["game"]["current_revision"])
        self.assertEqual(15, state["player_state"]["game"]["world_time_minutes"])

    async def test_replacement_managed_session_gets_fresh_realm_state(self):
        with tempfile.TemporaryDirectory() as directory:
            first_provider = RecordingProvider()
            first = ResidentRuntime(Config(Path(directory)), first_provider,
                                    capabilities=self.client.capabilities,
                                    realm_client=self.client,
                                    diagnostic_output=lambda _: None)
            await first.process(first.owner_message_event("look around"))
            self.assertEqual(1, first_provider.contexts[0]["realm_state"]["trusted_state"]["game"]["current_revision"])
            first.close()

            await self.client.mutate("advance-time", {"minutes": 15})
            second_provider = RecordingProvider()
            second = ResidentRuntime(Config(Path(directory)), second_provider,
                                     capabilities=self.client.capabilities,
                                     realm_client=self.client,
                                     diagnostic_output=lambda _: None)
            await second.process(second.owner_message_event("continue"))
            second_state = second_provider.contexts[0]["realm_state"]
            self.assertEqual(2, second_state["trusted_state"]["game"]["current_revision"])
            self.assertEqual(15, second_state["player_state"]["game"]["world_time_minutes"])
            second.close()

    async def test_mutations_bind_actor_revision_and_call_identity(self):
        from resident.capabilities import _INVOCATION_ID
        token = _INVOCATION_ID.set("durable-call")
        try:
            result = await self.client.mutate("reveal-fact", {"fact_id": "fact"})
        finally:
            _INVOCATION_ID.reset(token)
        post = next(call for call in self.calls if call[0] == "POST")
        self.assertEqual({"fact_id": "fact", "actor_id": "hero", "expected_revision": 1,
                          "idempotency_key": "durable-call"}, post[2])
        self.assertEqual(2, result["mutation"]["revision"])
        self.assertEqual(2, result["trusted_state"]["game"]["current_revision"])

    async def test_world_patch_schema_and_combined_creation(self):
        schema = next(cap.spec.input_schema for cap in self.client.capabilities
                      if cap.name == "realm_world_patch")
        patch_body = {
            "entities": [{"id": "case", "kind": "item", "name": "Leather case",
                          "description": "A small case", "properties": {"closed": True},
                          "player": {"name": "Case", "properties": {"hint": [1]}}}],
            "entity_updates": [{"entity_id": "hero", "player_visible": True,
                                "player": {"description": "Visible"}}],
            "containment": [{"child_id": "case", "parent_id": "hero"}],
            "connections": [{"ref": "exit", "from_place_id": "here",
                             "to_place_id": "there", "typical_travel_minutes": 0}],
            "facts": [{"ref": "fact", "text": "The case is closed",
                       "subject_entity_id": "case", "metadata": {"source": ["hero"]}}],
            "knowledge": [{"actor_id": "hero", "fact_id": "fact"}],
            "observations": [{"actor_id": "hero", "entity_id": "case"}],
        }
        self.assertIsNone(ToolRegistry._validate(schema, patch_body))
        result = await self.client.mutate("world-patch", patch_body)
        post = next(call for call in self.calls if call[0] == "POST")
        self.assertEqual("/games/game/world-patches", post[1])
        self.assertEqual({**patch_body, "expected_revision": 1,
                          "idempotency_key": result["idempotency_key"]}, post[2])
        self.assertEqual(2, result["mutation"]["revision"])

        invalid = {
            "entities": {"kind": "tool", "name": "Case"},
            "entity_updates": {"entity_id": "hero", "name": ""},
            "containment": {"child_entity_id": "case", "parent_entity_id": "hero"},
            "connections": {"from_place_id": "here", "to_place_id": "there",
                            "typical_travel_minutes": -1},
            "facts": {"text": "", "metadata": {}},
            "knowledge": {"actor_entity_id": "hero", "fact_id": "fact"},
            "observations": {"actor_entity_id": "hero", "entity_id": "case"},
        }
        for section, item in invalid.items():
            with self.subTest(section=section):
                self.assertIsNotNone(ToolRegistry._validate(schema, {section: [item]}))
        for item in ({"child_entity_id": "case", "parent_id": "hero"},
                     {"child_id": "case", "parent_entity_id": "hero"}):
            with self.subTest(containment=item):
                self.assertIsNotNone(ToolRegistry._validate(schema, {"containment": [item]}))

    async def test_realm_validation_detail_is_available_to_managed_tool_error(self):
        schema = next(cap.spec.input_schema for cap in self.client.capabilities
                      if cap.name == "realm_world_patch")
        self.assertIn("child_entity_id", ToolRegistry._validate(
            schema, {"containment": [{"child_entity_id": "case", "parent_id": "hero"}]}))
        original = self.client._request

        async def reject(method, path, body=None):
            if method == "POST":
                raise RealmHTTPError(400, "INVALID_REQUEST: body/containment/0 must have required property 'child_id'")
            return await original(method, path, body)

        self.client._request = reject
        result = await self.client.mutate("world-patch", {"containment": [
            {"child_id": "case", "parent_id": "hero"}]})
        self.assertEqual("rejected", result["error_code"])
        self.assertEqual(400, result["http_status"])
        self.assertIn("body/containment/0", result["error"])
        self.assertIn("reread Realm before continuing", result["error"])
        event = OpenAIAgentsProvider._tool_result_event(ToolResult("call", result), "turn")
        self.assertFalse(event["success"])
        self.assertIn("body/containment/0", event["error"])
        self.assertEqual(1, self.revision)

    async def test_http_error_parsing_is_bounded_and_structured(self):
        response = MagicMock()
        response.status = 400
        response.content.read = AsyncMock()
        session = MagicMock()
        session.request.return_value.__aenter__ = AsyncMock(return_value=response)
        session.request.return_value.__aexit__ = AsyncMock(return_value=None)
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        with patch("resident.realm.aiohttp.ClientSession", return_value=session):
            response.content.read.return_value = json.dumps({
                "error": "INVALID_REQUEST", "message": "body/containment/0 requires child_id"
            }).encode()
            with self.assertRaises(RealmHTTPError) as caught:
                await self.client_request("POST")
            self.assertEqual("INVALID_REQUEST: body/containment/0 requires child_id",
                             caught.exception.detail)
            for raw in (b"<html>bad</html>", b"x" * 4097,
                        json.dumps({"error": "INVALID_REQUEST", "message": "bad\ninput"}).encode()):
                response.content.read.return_value = raw
                with self.assertRaises(RealmHTTPError) as caught:
                    await self.client_request("POST")
                self.assertIsNone(caught.exception.detail)
            response.content.read.side_effect = OSError("connection closed")
            with self.assertRaises(RealmHTTPError) as caught:
                await self.client_request("POST")
            self.assertEqual(400, caught.exception.status)
            self.assertIsNone(caught.exception.detail)
            response.content.read.assert_awaited()

    async def client_request(self, method):
        return await RealmClient._request(self.client, method, "/games/game/world-patches", {})

    async def test_lost_response_retry_reuses_durable_request_after_restart(self):
        from resident.capabilities import _INVOCATION_ID

        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "resident.sqlite3")
            self.client.bind_mutation_store(store.realm_mutation_request)
            original = self.client._request
            posts = []

            async def lost_response(method, path, body=None):
                if method == "POST":
                    posts.append((path, dict(body)))
                    self.revision += 1
                    self.world_time += body["minutes"]
                    raise asyncio.TimeoutError
                return await original(method, path, body)

            self.client._request = lost_response
            token = _INVOCATION_ID.set("durable-call")
            try:
                first = await self.client.mutate("advance-time", {"minutes": 15})
            finally:
                _INVOCATION_ID.reset(token)
            self.assertEqual("unknown_outcome", first["error_code"])
            store.close()

            reopened = Store(Path(directory) / "resident.sqlite3")
            replacement = RealmClient("http://127.0.0.1:3000", "game", "hero")
            replacement.bind_mutation_store(reopened.realm_mutation_request)

            async def idempotent_replay(method, path, body=None):
                if method == "POST":
                    posts.append((path, dict(body)))
                    self.assertEqual(1, body["expected_revision"])
                    return {"revision": self.revision, "events": [], "idempotent": True}
                return await original(method, path, body)

            replacement._request = idempotent_replay
            token = _INVOCATION_ID.set("durable-call")
            try:
                second = await replacement.mutate("advance-time", {"minutes": 99})
            finally:
                _INVOCATION_ID.reset(token)
                reopened.close()
            self.assertEqual(posts[0], posts[1])
            self.assertEqual(15, second["player_state"]["game"]["world_time_minutes"])
            self.assertTrue(second["mutation"]["idempotent"])

    async def test_snapshot_retries_mutation_between_player_and_trusted_reads(self):
        original = self.client._request
        player_reads = 0

        async def concurrent_mutation(method, path, body=None):
            nonlocal player_reads
            result = await original(method, path, body)
            if method == "GET" and path.endswith("/state?actor_id=hero"):
                player_reads += 1
                if player_reads == 1:
                    self.revision += 1
                    self.world_time += 15
            return result

        self.client._request = concurrent_mutation
        snapshot = await self.client.snapshot()
        self.assertEqual(2, player_reads)
        self.assertEqual(2, snapshot["trusted_state"]["game"]["current_revision"])
        self.assertEqual(15, snapshot["player_state"]["game"]["world_time_minutes"])

    async def test_conflict_does_not_auto_retry(self):
        async def reject(method, path, body=None):
            if method == "POST":
                raise RealmHTTPError(409)
            return await original(method, path, body)
        original = self.client._request
        self.client._request = reject
        result = await self.client.mutate("advance-time", {"minutes": 1})
        self.assertEqual("conflict", result["error_code"])
        self.assertEqual(1, self.revision)

    async def test_unknown_outcome_is_reported_without_retry(self):
        async def timeout(method, path, body=None):
            if method == "POST":
                raise asyncio.TimeoutError
            return await original(method, path, body)
        original = self.client._request
        self.client._request = timeout
        result = await self.client.mutate("advance-time", {"minutes": 1})
        self.assertEqual("unknown", result["outcome"])
        self.assertEqual("unknown_outcome", result["error_code"])

    async def test_keeper_definition_grants_native_realm(self):
        root = Path(__file__).resolve().parents[1]
        definition = load_resident_definition(root / "residents" / "keeper.yaml",
                                              root / "prompts")
        self.assertEqual(("realm",), definition.capabilities)
        self.assertEqual("KEEPER_REALM_GAME_ID", definition.realm.game_id_env)
        self.assertEqual("KEEPER_REALM_ACTOR_ID", definition.realm.actor_id_env)


if __name__ == "__main__":
    unittest.main()
