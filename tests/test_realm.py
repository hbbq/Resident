import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from resident.realm import RealmClient, RealmHTTPError
from resident.config import Config
from resident.domain import ModelTurn, ToolCall, ToolResult, WakeEvent
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
            first = ResidentRuntime(Config(Path(directory), keeper_history=True), first_provider,
                                    capabilities=self.client.capabilities,
                                    realm_client=self.client,
                                    diagnostic_output=lambda _: None)
            await first.process(first.owner_message_event("look around"))
            self.assertEqual(1, first_provider.contexts[0]["realm_state"]["trusted_state"]["game"]["current_revision"])
            first.close()

            await self.client.mutate("advance-time", {"minutes": 15})
            second_provider = RecordingProvider()
            second = ResidentRuntime(Config(Path(directory), keeper_history=True), second_provider,
                                     capabilities=self.client.capabilities,
                                     realm_client=self.client,
                                     diagnostic_output=lambda _: None)
            await second.process(second.owner_message_event("continue"))
            second_state = second_provider.contexts[0]["realm_state"]
            self.assertEqual(2, second_state["trusted_state"]["game"]["current_revision"])
            self.assertEqual(15, second_state["player_state"]["game"]["world_time_minutes"])
            history = second_provider.contexts[0]["new_session_bootstrap"]["keeper_recent_interactions"]
            self.assertEqual(1, len(history))
            self.assertEqual("done", history[0]["activity"][0]["text"])
            self.assertNotIn("realm_state", str(history))
            self.assertNotIn("secret", str(history))
            second.close()

    async def test_rollover_history_is_scoped_to_configured_game_and_actor(self):
        for game_id, actor_id in (("new-game", "hero"), ("game", "new-actor")):
            with self.subTest(game_id=game_id, actor_id=actor_id):
                with tempfile.TemporaryDirectory() as directory:
                    store = Store(Path(directory) / "resident.sqlite3")
                    event = WakeEvent("old-wake", "owner", "play", "2026-01-01T00:00:00Z", {})
                    run_id = store.start_run(event)
                    store.start_keeper_interaction(run_id, event, "game", "hero")
                    store.add_keeper_activity(run_id, "model_turn", {
                        "message": "Old game and actor narrative", "tool_calls": []})
                    store.finish_keeper_interaction(run_id, "completed")
                    store.close()

                    client = RealmClient("http://127.0.0.1:3000", game_id, actor_id)
                    client.read = AsyncMock(return_value={"player_state": {}, "trusted_state": {}})
                    provider = RecordingProvider()
                    runtime = ResidentRuntime(Config(Path(directory), keeper_history=True),
                                              provider, capabilities=client.capabilities,
                                              realm_client=client,
                                              diagnostic_output=lambda _: None)
                    try:
                        await runtime.process(runtime.owner_message_event("continue"))
                        self.assertEqual([], provider.contexts[0]["new_session_bootstrap"][
                            "keeper_recent_interactions"])
                    finally:
                        runtime.close()

    async def test_rollover_history_filters_before_interaction_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "resident.sqlite3")
            for index, (game_id, actor_id) in enumerate((
                    ("game", "hero"), ("other-game", "hero"), ("game", "other-actor"))):
                event = WakeEvent(f"wake-{index}", "owner", "play",
                                  "2026-01-01T00:00:00Z", {})
                run_id = store.start_run(event)
                store.start_keeper_interaction(run_id, event, game_id, actor_id)
                store.add_keeper_activity(run_id, "model_turn", {
                    "message": f"narrative-{index}", "tool_calls": []})
                store.finish_keeper_interaction(run_id, "completed")
            for game_id, actor_id, expected in (("game", "hero", "narrative-0"),
                                                ("other-game", "hero", "narrative-1"),
                                                ("game", "other-actor", "narrative-2")):
                with self.subTest(game_id=game_id, actor_id=actor_id):
                    history = store.keeper_recent_context(
                        1, 16384, game_id=game_id, actor_id=actor_id)
                    self.assertEqual([expected], [entry["activity"][0]["text"]
                                                  for entry in history])
            store.close()

    async def test_interaction_persists_exact_input_and_trusted_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = RecordingProvider()
            runtime = ResidentRuntime(Config(Path(directory), keeper_history=True), provider,
                                      capabilities=self.client.capabilities,
                                      realm_client=self.client,
                                      diagnostic_output=lambda _: None)
            run_id = await runtime.process(runtime.owner_message_event("open the door"))
            row = runtime.store.connection.execute(
                "SELECT * FROM keeper_interactions WHERE run_id=?", (run_id,)).fetchone()
            self.assertEqual("completed", row["status"])
            self.assertEqual("new-session", row["session_id"])
            self.assertEqual(1, row["realm_revision"])
            self.assertEqual(provider.contexts[0], json.loads(row["input_text"]))
            self.assertIn("secret", row["realm_snapshot_json"])
            self.assertEqual(["model_turn"], [item[0] for item in
                runtime.store.connection.execute(
                    "SELECT kind FROM keeper_activity WHERE run_id=? ORDER BY sequence", (run_id,))])
            runtime.close()

    async def test_realm_managed_resident_without_keeper_opt_in_has_no_history(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = RecordingProvider()
            runtime = ResidentRuntime(Config(Path(directory)), provider,
                                      capabilities=self.client.capabilities,
                                      realm_client=self.client,
                                      diagnostic_output=lambda _: None)
            await runtime.process(runtime.owner_message_event("look"))
            self.assertNotIn("keeper_recent_interactions",
                             provider.contexts[0]["new_session_bootstrap"])
            self.assertEqual(0, runtime.store.connection.execute(
                "SELECT COUNT(*) FROM keeper_interactions").fetchone()[0])
            runtime.close()

    async def test_rollover_projects_player_call_content_and_actual_encoding_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "resident.sqlite3")
            event = WakeEvent("wake", "owner", "play", "2026-01-01T00:00:00Z", {})
            run_id = store.start_run(event)
            store.start_keeper_interaction(run_id, event, "game", "hero")
            store.add_keeper_activity(run_id, "model_turn", {
                "message": None, "tool_calls": [
                    {"id": "message", "name": "send_owner_message", "arguments": {"content": "The door opens."}},
                    {"id": "patch", "name": "realm_world_patch", "arguments": {"entities": [
                        {"name": "secret", "description": "trusted secret",
                         "player": {"description": "A dark doorway."}}],
                        "facts": [{"text": "hidden fact"}]}},
                ]}, turn_id="turn")
            for call_id, name, result in (
                    ("message", "send_owner_message", {"ok": True, "delivered": True}),
                    ("patch", "realm_world_patch", {"ok": True, "mutation": {"revision": 2}})):
                store.add_keeper_activity(run_id, "tool_result", {
                    "call_id": call_id, "name": name, "result": result,
                }, turn_id="turn", call_id=call_id)
            store.finish_keeper_interaction(run_id, "completed")
            entry = store.keeper_recent_context(1, 16384, game_id="game", actor_id="hero")
            self.assertEqual("The door opens.", entry[0]["activity"][0]["content"])
            self.assertEqual([{"description": "A dark doorway."}],
                             entry[0]["activity"][1]["player_views"])
            self.assertNotIn("secret", str(entry))
            self.assertNotIn("hidden fact", str(entry))
            wrapper = {"new_session_bootstrap": {
                "preceding_field": None, "keeper_recent_interactions": entry}}
            baseline = {"new_session_bootstrap": {"preceding_field": None}}
            size = len(json.dumps(wrapper, ensure_ascii=False, indent=2).encode()) - len(
                json.dumps(baseline, ensure_ascii=False, indent=2).encode())
            self.assertEqual(entry, store.keeper_recent_context(
                1, size, game_id="game", actor_id="hero"))
            self.assertEqual([], store.keeper_recent_context(
                1, size - 1, game_id="game", actor_id="hero"))
            store.close()

    async def test_bootstrap_skips_non_list_historical_world_patch_sections(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "resident.sqlite3")
            event = WakeEvent("prior-wake", "owner", "play", "2026-01-01T00:00:00Z", {})
            run_id = store.start_run(event)
            store.start_keeper_interaction(run_id, event, "game", "hero")
            store.add_keeper_activity(run_id, "model_turn", {
                "message": None, "tool_calls": [
                    {"id": "patch-1", "name": "realm_world_patch", "arguments": {
                        "entities": None, "entity_updates": [
                            {"player": {"description": "A doorway appears."}}]}},
                    {"id": "patch-2", "name": "realm_world_patch", "arguments": {
                        "entities": {"player": {"description": "not a list"}},
                        "entity_updates": {"player": {"description": "also not a list"}}}},
                    {"id": "patch-3", "name": "realm_world_patch", "arguments": {
                        "entities": [None, {"player": {"name": "The doorway"}}],
                        "entity_updates": None}},
                ]}, turn_id="turn")
            for call_id in ("patch-1", "patch-2", "patch-3"):
                store.add_keeper_activity(run_id, "tool_result", {
                    "call_id": call_id, "name": "realm_world_patch",
                    "result": {"ok": True, "mutation": {"revision": 2}},
                }, turn_id="turn", call_id=call_id)
            store.finish_keeper_interaction(run_id, "completed")
            store.close()

            provider = RecordingProvider()
            runtime = ResidentRuntime(Config(Path(directory), keeper_history=True), provider,
                                      capabilities=self.client.capabilities,
                                      realm_client=self.client,
                                      diagnostic_output=lambda _: None)
            try:
                await runtime.process(runtime.owner_message_event("continue"))
                history = provider.contexts[0]["new_session_bootstrap"]["keeper_recent_interactions"]
                self.assertEqual([
                    {"kind": "player_facing_call", "name": "realm_world_patch",
                     "player_views": [{"description": "A doorway appears."}]},
                    {"kind": "player_facing_call", "name": "realm_world_patch",
                     "player_views": [{"name": "The doorway"}]},
                    {"kind": "action_outcome", "name": "realm_world_patch",
                     "ok": True, "error_code": None, "outcome": None},
                    {"kind": "action_outcome", "name": "realm_world_patch",
                     "ok": True, "error_code": None, "outcome": None},
                    {"kind": "action_outcome", "name": "realm_world_patch",
                     "ok": True, "error_code": None, "outcome": None},
                ], history[0]["activity"])
                self.assertNotIn("not a list", str(history))
            finally:
                runtime.close()

    async def test_rollover_only_projects_confirmed_player_facing_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "resident.sqlite3")
            event = WakeEvent("wake", "owner", "play", "2026-01-01T00:00:00Z", {})
            run_id = store.start_run(event)
            store.start_keeper_interaction(run_id, event, "game", "hero")
            calls = [
                {"id": "rejected-patch", "name": "realm_world_patch", "arguments": {
                    "entities": [{"player": {"description": "A rejected doorway."}}]}},
                {"id": "unknown-patch", "name": "realm_world_patch", "arguments": {
                    "entities": [{"player": {"description": "An uncertain doorway."}}]}},
                {"id": "undelivered", "name": "send_owner_message", "arguments": {
                    "content": "An undelivered message."}},
                {"id": "missing-result", "name": "send_owner_message", "arguments": {
                    "content": "An unconfirmed message."}},
                {"id": "accepted-patch", "name": "realm_world_patch", "arguments": {
                    "entities": [{"player": {"description": "An accepted doorway."}}]}},
                {"id": "delivered", "name": "send_owner_message", "arguments": {
                    "content": "A delivered message."}},
            ]
            store.add_keeper_activity(run_id, "model_turn", {
                "message": None, "tool_calls": calls,
            }, turn_id="turn")
            for call_id, name, result in (
                    ("rejected-patch", "realm_world_patch",
                     {"ok": False, "error_code": "rejected"}),
                    ("unknown-patch", "realm_world_patch",
                     {"ok": False, "error_code": "unknown_outcome", "outcome": "unknown"}),
                    ("undelivered", "send_owner_message", {"ok": True, "delivered": False}),
                    ("accepted-patch", "realm_world_patch", {"ok": True}),
                    ("delivered", "send_owner_message", {"ok": True, "delivered": True})):
                store.add_keeper_activity(run_id, "tool_result", {
                    "call_id": call_id, "name": name, "result": result,
                }, turn_id="turn", call_id=call_id)
            store.finish_keeper_interaction(run_id, "completed")
            activity = store.keeper_recent_context(
                1, 16384, game_id="game", actor_id="hero")[0]["activity"]
            self.assertEqual([
                {"kind": "player_facing_call", "name": "realm_world_patch",
                 "player_views": [{"description": "An accepted doorway."}]},
                {"kind": "player_facing_call", "name": "send_owner_message",
                 "content": "A delivered message."},
            ], [item for item in activity if item["kind"] == "player_facing_call"])
            self.assertEqual([False, False, False, True, True], [
                item["ok"] for item in activity if item["kind"] == "action_outcome"])
            self.assertNotIn("rejected doorway", str(activity))
            self.assertNotIn("uncertain doorway", str(activity))
            self.assertNotIn("undelivered message", str(activity))
            self.assertNotIn("unconfirmed message", str(activity))
            store.close()

    async def test_failed_interaction_is_not_replayed(self):
        class FailingProvider(RecordingProvider):
            async def respond(self, context, tools, results, continuation_id=None):
                raise RuntimeError("unavailable")

        with tempfile.TemporaryDirectory() as directory:
            runtime = ResidentRuntime(Config(Path(directory), keeper_history=True), FailingProvider(),
                                      capabilities=self.client.capabilities,
                                      realm_client=self.client,
                                      diagnostic_output=lambda _: None)
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                await runtime.process(runtime.owner_message_event("look"))
            row = runtime.store.connection.execute(
                "SELECT status,input_text FROM keeper_interactions").fetchone()
            self.assertEqual("failed", row["status"])
            self.assertIsNotNone(row["input_text"])
            self.assertEqual([], runtime.store.keeper_recent_context(
                8, 16384, game_id="game", actor_id="hero"))
            runtime.close()

    async def test_tool_activity_is_ordered_and_rollover_omits_realm_views(self):
        class ToolProvider(RecordingProvider):
            async def respond(self, context, tools, results, continuation_id=None):
                self.contexts.append(json.loads(context))
                self.session_id = "session-with-tool"
                if not results:
                    return ModelTurn("first", tool_calls=(
                        ToolCall("read-1", "realm_read", {}),))
                return ModelTurn("final", message="A door remains mysterious")

        with tempfile.TemporaryDirectory() as directory:
            runtime = ResidentRuntime(Config(Path(directory), keeper_history=True), ToolProvider(),
                                      capabilities=self.client.capabilities,
                                      realm_client=self.client,
                                      diagnostic_output=lambda _: None)
            run_id = await runtime.process(runtime.owner_message_event("inspect"))
            rows = runtime.store.connection.execute(
                "SELECT kind,content_json FROM keeper_activity WHERE run_id=? ORDER BY sequence",
                (run_id,)).fetchall()
            self.assertEqual(["model_turn", "tool_result", "model_turn"],
                             [row["kind"] for row in rows])
            self.assertIn("secret", rows[1]["content_json"])
            recent = runtime.store.keeper_recent_context(
                1, 16384, game_id="game", actor_id="hero")
            self.assertIn("A door remains mysterious", str(recent))
            self.assertNotIn("secret", str(recent))
            self.assertEqual([], runtime.store.keeper_recent_context(
                1, 10, game_id="game", actor_id="hero"))
            runtime.close()

    async def test_retrying_model_turn_activity_keeps_one_ordered_history_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "resident.sqlite3")
            event = WakeEvent("wake", "owner", "play", "2026-01-01T00:00:00Z", {})
            run_id = store.start_run(event)
            store.start_keeper_interaction(run_id, event, "game", "hero")
            first = {"message": "First turn", "tool_calls": []}
            second = {"message": "Second turn", "tool_calls": []}
            store.add_keeper_activity(run_id, "model_turn", first,
                                      session_id="session", turn_id="turn-1")
            store.add_keeper_activity(run_id, "tool_result", {"name": "check", "result": {"ok": True}},
                                      session_id="session", turn_id="turn-1", call_id="call-1")
            store.add_keeper_activity(run_id, "model_turn", first,
                                      session_id="session", turn_id="turn-1")
            store.add_keeper_activity(run_id, "model_turn", second,
                                      session_id="session", turn_id="turn-2")
            store.add_keeper_activity(run_id, "model_turn", second,
                                      session_id="session", turn_id="turn-2")
            store.finish_keeper_interaction(run_id, "completed")

            rows = store.connection.execute(
                "SELECT kind,turn_id FROM keeper_activity WHERE run_id=? ORDER BY sequence",
                (run_id,)).fetchall()
            self.assertEqual([("model_turn", "turn-1"), ("tool_result", "turn-1"),
                              ("model_turn", "turn-2")], [tuple(row) for row in rows])
            history = store.keeper_recent_context(1, 16384, game_id="game", actor_id="hero")
            self.assertEqual(["First turn", "Second turn"],
                             [entry["text"] for entry in history[0]["activity"]
                              if entry["kind"] == "keeper_output"])
            store.close()

    async def test_existing_duplicate_model_turns_keep_first_sequence_on_upgrade(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resident.sqlite3"
            store = Store(path)
            event = WakeEvent("wake", "owner", "play", "2026-01-01T00:00:00Z", {})
            run_id = store.start_run(event)
            store.start_keeper_interaction(run_id, event, "game", "hero")
            with store.connection:
                store.connection.execute("DROP INDEX idx_keeper_activity_identity")
            store.add_keeper_activity(run_id, "model_turn", {"message": "Original"},
                                      session_id="session", turn_id="turn")
            store.add_keeper_activity(run_id, "model_turn", {"message": "Retry"},
                                      session_id="session", turn_id="turn")
            store.close()

            reopened = Store(path)
            rows = reopened.connection.execute(
                "SELECT sequence,content_json FROM keeper_activity WHERE run_id=?", (run_id,)
            ).fetchall()
            self.assertEqual(1, len(rows))
            self.assertEqual(1, rows[0]["sequence"])
            self.assertEqual("Original", json.loads(rows[0]["content_json"])["message"])
            reopened.close()

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
                          "description": "A small case", "appearance": "Scuffed brown leather",
                          "properties": {"closed": True},
                          "player": {"name": "Case", "properties": {"hint": [1]}}}],
            "entity_updates": [{"entity_id": "hero", "player_visible": True,
                                "appearance": "A red cloak",
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

        for section, item in (
                ("entities", {"kind": "item", "name": "Unadorned case"}),
                ("entity_updates", {"entity_id": "hero", "description": "A traveler"})):
            with self.subTest(section=section):
                item_schema = schema["properties"][section]["items"]
                self.assertIn("appearance", item_schema["properties"])
                self.assertEqual({"type": "string"}, item_schema["properties"]["appearance"])
                self.assertNotIn("appearance", item_schema["required"])
                self.assertIsNone(ToolRegistry._validate(schema, {section: [item]}))
                self.assertIsNone(ToolRegistry._validate(
                    schema, {section: [{**item, "appearance": "A plain outline"}]}))

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

    async def test_keeper_guidance_explains_appearance(self):
        root = Path(__file__).resolve().parents[1]
        guidance = load_resident_definition(root / "residents" / "keeper.yaml",
                                            root / "prompts").personality
        guidance = " ".join(guidance.split())
        for phrase in ("optional `appearance`", "observable visual characteristics",
                       "secrets, hidden motives", "Do not invent filler appearance",
                       "separate from player-facing projections", "ordinary world",
                       "Do not generate or manage illustrations", "Realm's independent",
                       "does not necessarily regenerate"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, guidance)


if __name__ == "__main__":
    unittest.main()
