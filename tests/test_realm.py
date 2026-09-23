import asyncio
import tempfile
import unittest
from pathlib import Path

from resident.realm import RealmClient, RealmHTTPError
from resident.config import Config
from resident.domain import ModelTurn
from resident.instances import load_resident_definition
from resident.runtime import ResidentRuntime


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
