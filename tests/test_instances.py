import asyncio
import json
import os
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from resident.__main__ import build_host
from resident.config import Config, DisplayConfig, SUPPORTED_REASONING_EFFORTS
from resident.domain import ModelTurn, WakeEvent
from resident.host import InstancePolicy, RuntimeHost, messaging_capability
from resident.instances import load_resident_catalog, migrate_legacy_state
from resident.mailbox import Mailbox
from resident.observability import EventLoopLagProbe
from resident.readiness import ReadinessItem, ReadinessResult
from resident.runtime import ResidentRuntime
from resident.store import utc_now


class IdleProvider:
    async def respond(self, context, tools, results, continuation_id=None):
        return ModelTurn("turn", None, ())


class OutputProvider(IdleProvider):
    supports_output_capabilities = True

    def configure_output_protocol(self, schema, descriptors, fingerprint):
        self.schema = schema
        self.descriptors = descriptors
        self.fingerprint = fingerprint


class InstanceDefinitionTests(unittest.TestCase):
    def test_keeper_history_requires_explicit_valid_opt_in(self):
        with tempfile.TemporaryDirectory() as temporary:
            definitions = Path(temporary) / "residents"
            definitions.mkdir()
            path = definitions / "resident.yaml"
            base = "id: resident\nname: Resident\npersonality: Test.\nrole: Test.\n"
            realm = ("realm:\n  base_url: http://realm.test\n"
                     "  game_id_env: GAME_ID\n  actor_id_env: ACTOR_ID\n")
            path.write_text(base + realm, encoding="utf-8")
            self.assertFalse(load_resident_catalog(definitions).residents[0].keeper_history)
            path.write_text(base + realm + "keeper_history: true\n", encoding="utf-8")
            self.assertTrue(load_resident_catalog(definitions).residents[0].keeper_history)
            with patch.dict(os.environ, {
                    "OPENAI_API_KEY": "test-key", "GAME_ID": "game", "ACTOR_ID": "hero"}):
                host = build_host(Config(Path(temporary) / "data", residents_dir=definitions))
            try:
                self.assertTrue(host.runtimes["resident"].config.keeper_history)
            finally:
                host.close()
            for content, error in (
                    (base + realm + "keeper_history: yes-please\n", "must be boolean"),
                    (base + "keeper_history: true\n", "requires realm and openai-agents"),
                    (base + realm + "agent:\n  provider: openai-responses\n"
                     "keeper_history: true\n", "requires realm and openai-agents")):
                with self.subTest(error=error):
                    path.write_text(content, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, error):
                        load_resident_catalog(definitions)

    def test_legacy_curator_environment_and_cli_configuration_is_preserved(self):
        with patch.dict(os.environ, {
                "RESIDENT_CURATOR_API_KEY": "legacy-key",
                "RESIDENT_CURATOR_BASE_URL": "https://legacy.example/v1/",
        }, clear=True):
            config = Config.from_env_and_args([
                "--curator-model", "legacy-curator",
                "--curator-batch-size", "17",
                "--curator-max-batches", "3",
            ])

        self.assertEqual("legacy-curator", config.curator_model)
        self.assertEqual("legacy-key", config.curator_api_key)
        self.assertEqual("https://legacy.example/v1", config.curator_base_url)
        self.assertEqual(17, config.curator_batch_size)
        self.assertEqual(3, config.curator_max_batches)

    def test_declarative_reasoning_effort_matches_cli_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            definitions = Path(temporary) / "residents"
            definitions.mkdir()
            definition = definitions / "resident.yaml"
            base = "id: resident\nname: Resident\npersonality: Test.\nrole: Test.\n"

            definition.write_text(base, encoding="utf-8")
            self.assertIsNone(
                load_resident_catalog(definitions).residents[0].agent.reasoning_effort)

            for effort in SUPPORTED_REASONING_EFFORTS:
                with self.subTest(effort=effort):
                    definition.write_text(
                        f"{base}agent:\n  reasoning_effort: {effort}\n", encoding="utf-8")
                    loaded = load_resident_catalog(definitions).residents[0]
                    self.assertEqual(effort, loaded.agent.reasoning_effort)
                    self.assertEqual(
                        effort,
                        Config.from_env_and_args(
                            ["--reasoning-effort", effort]).reasoning_effort)

            definition.write_text(
                f"{base}agent:\n  reasoning_effort: medum\n", encoding="utf-8")
            with self.assertRaisesRegex(
                    ValueError, "agent.reasoning_effort must be one of"):
                load_resident_catalog(definitions)
            with self.assertRaises(SystemExit):
                Config.from_env_and_args(["--reasoning-effort", "medum"])

    def test_reasoning_effort_typo_fails_before_provider_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            definitions = root / "residents"
            definitions.mkdir()
            (definitions / "resident.yaml").write_text(
                "id: resident\nname: Resident\npersonality: Test.\nrole: Test.\n"
                "agent:\n  reasoning_effort: almost_high\n", encoding="utf-8")

            with patch("resident.__main__._provider") as provider:
                with self.assertRaisesRegex(
                        ValueError, "agent.reasoning_effort must be one of"):
                    build_host(Config(root / "data", residents_dir=definitions))
                provider.assert_not_called()

    def test_catalog_runtimes_use_per_resident_curator_and_explicit_new_chapter(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            definitions = root / "residents"
            definitions.mkdir()
            (definitions / "resident.yaml").write_text(
                "id: resident\nname: Resident\npersonality: Test.\nrole: Test.\n"
                "curator:\n  model: resident-curator\n  api_key_env: CURATOR_KEY\n"
                "  base_url_env: CURATOR_URL\n  batch_size: 17\n  max_batches: 3\n",
                encoding="utf-8")
            (definitions / "helper.yaml").write_text(
                "id: helper\nname: Helper\npersonality: Test.\nrole: Test.\n",
                encoding="utf-8")
            config = Config(
                root / "data", residents_dir=definitions, new_chapter=True,
                curator_model="legacy-curator", curator_api_key="legacy-key",
                curator_base_url="https://curator.example/v1",
                curator_batch_size=17, curator_max_batches=3)

            with patch.dict(os.environ, {
                    "OPENAI_API_KEY": "resident-key", "CURATOR_KEY": "curator-key",
                    "CURATOR_URL": "https://per-resident.example/v1/"}):
                host = build_host(config)
            try:
                self.assertEqual({"resident", "helper"}, set(host.runtimes))
                resident = host.runtimes["resident"]
                self.assertEqual("resident-curator", resident.config.curator_model)
                self.assertEqual("curator-key", resident.config.curator_api_key)
                self.assertEqual(
                    "https://per-resident.example/v1", resident.config.curator_base_url)
                self.assertEqual(17, resident.curator.batch_size)
                self.assertEqual(3, resident.curator.max_batches)
                self.assertEqual("resident-curator", resident.curator.model.model)
                helper = host.runtimes["helper"]
                self.assertIsNone(helper.config.curator_model)
                self.assertIsNone(helper.config.curator_api_key)
                self.assertIsNone(helper.curator)
                for runtime in host.runtimes.values():
                    self.assertEqual(
                        "explicit_new_chapter",
                        runtime.provider._requested_rollover_reason)
            finally:
                host.close()

    def test_loads_prompt_files_and_separates_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "residents").mkdir()
            (root / "prompts").mkdir()
            (root / "prompts" / "oracle.md").write_text("Be precise.", encoding="utf-8")
            (root / "residents" / "oracle.yaml").write_text("""
version: 1
id: oracle
name: Oracle
personality_prompt: oracle.md
role: Answer narrow questions.
capabilities: [messaging]
outputs: [notify_owner, display/display1]
subscriptions: [homeops]
""", encoding="utf-8")
            catalog = load_resident_catalog(root / "residents", default_id="oracle")
            oracle = catalog.residents[0]
            self.assertEqual("Be precise.", oracle.personality)
            self.assertEqual("Answer narrow questions.", oracle.role)
            self.assertEqual(("messaging",), oracle.capabilities)
            self.assertEqual(("notify_owner", "display/display1"), oracle.outputs)
            self.assertEqual(("homeops",), oracle.subscriptions)

    def test_outputs_must_be_a_unique_string_list(self):
        with tempfile.TemporaryDirectory() as temporary:
            definitions = Path(temporary) / "residents"
            definitions.mkdir()
            definition = definitions / "resident.yaml"
            base = "id: resident\nname: Resident\npersonality: Test.\nrole: Test.\n"
            for value in ("{notify_owner: true}", "[notify_owner, notify_owner]", "[1]"):
                with self.subTest(value=value):
                    definition.write_text(f"{base}outputs: {value}\n", encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "outputs"):
                        load_resident_catalog(definitions)

    def test_per_resident_output_grants_resolve_against_shared_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            definitions = root / "residents"
            definitions.mkdir()
            (definitions / "resident.yaml").write_text(
                "id: resident\nname: Resident\npersonality: Test.\nrole: Test.\n"
                "capabilities: [display]\noutputs: [notify_owner, display/display1]\n",
                encoding="utf-8")
            (definitions / "helper.yaml").write_text(
                "id: helper\nname: Helper\npersonality: Test.\nrole: Test.\n"
                "outputs: [display/display2]\n", encoding="utf-8")
            config = Config(
                root / "data", residents_dir=definitions,
                homeops_url="http://homeops.test",
                displays=(DisplayConfig("display1"), DisplayConfig("display2")))

            with patch("resident.__main__._provider", side_effect=lambda _: OutputProvider()):
                host = build_host(config)
            try:
                resident = host.runtimes["resident"]
                helper = host.runtimes["helper"]
                self.assertEqual(
                    ["notify_owner", "display/display1"],
                    [output.grant_id for output in resident.output_capabilities])
                self.assertEqual(
                    ["display/display2"],
                    [output.grant_id for output in helper.output_capabilities])
                self.assertNotIn("display2", json.dumps(resident._output_schema))
                self.assertNotIn("display1", json.dumps(helper._output_schema))
                self.assertNotEqual(
                    resident._output_schema_fingerprint,
                    helper._output_schema_fingerprint)
                # The compatibility function is present for Responses/old-session
                # protocols even though the new structured protocol filters it.
                self.assertIn("display2_show_text", [item.name for item in helper.capabilities])
                self.assertNotIn(
                    "display2_show_text",
                    [item.name for item in helper._tool_capabilities_for_protocol(True)])
            finally:
                host.close()

    def test_available_owner_route_is_not_an_implicit_grant_and_empty_outputs_is_silent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            definitions = root / "residents"
            definitions.mkdir()
            (definitions / "resident.yaml").write_text(
                "id: resident\nname: Resident\npersonality: Test.\nrole: Test.\n"
                "outputs: []\n", encoding="utf-8")
            with patch("resident.__main__._provider", return_value=OutputProvider()):
                host = build_host(Config(root / "data", residents_dir=definitions))
            try:
                runtime = host.runtimes["resident"]
                self.assertEqual((), runtime.output_capabilities)
                self.assertEqual(0, runtime._output_schema["properties"]["outputs"]["maxItems"])
                self.assertEqual([], runtime.provider.descriptors)
            finally:
                host.close()

    def test_configured_owner_transport_is_available_but_not_granted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            definitions = root / "residents"
            definitions.mkdir()
            (definitions / "resident.yaml").write_text(
                "id: resident\nname: Resident\npersonality: Test.\nrole: Test.\n"
                "outputs: []\nowner_transport:\n  type: telegram\n"
                "  token_env: TEST_BOT_TOKEN\n  owner_user_id_env: TEST_OWNER_USER\n"
                "  owner_chat_id_env: TEST_OWNER_CHAT\n", encoding="utf-8")
            environment = {
                "TEST_BOT_TOKEN": "token", "TEST_OWNER_USER": "1", "TEST_OWNER_CHAT": "1",
            }
            with (patch.dict(os.environ, environment, clear=True),
                  patch("resident.__main__._provider", return_value=OutputProvider())):
                host = build_host(Config(root / "data", residents_dir=definitions))
            try:
                runtime = host.runtimes["resident"]
                self.assertTrue(runtime._remote_owner_transport)
                self.assertEqual((), runtime.output_capabilities)
                self.assertFalse(runtime.config.owner_communication_enabled)
            finally:
                host.close()

    def test_output_grants_do_not_come_from_callable_capabilities(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            definitions = root / "residents"
            definitions.mkdir()
            (definitions / "resident.yaml").write_text(
                "id: resident\nname: Resident\npersonality: Test.\nrole: Test.\n"
                "capabilities: [display]\noutputs: []\n", encoding="utf-8")
            config = Config(
                root / "data", residents_dir=definitions,
                homeops_url="http://homeops.test", displays=(DisplayConfig("display1"),))
            with patch("resident.__main__._provider", return_value=OutputProvider()):
                host = build_host(config)
            try:
                runtime = host.runtimes["resident"]
                self.assertEqual((), runtime.output_capabilities)
                self.assertIn("display1_show_text", [item.name for item in runtime.capabilities])
            finally:
                host.close()

    def test_unavailable_output_grants_fail_before_provider_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            definitions = root / "residents"
            definitions.mkdir()
            (definitions / "resident.yaml").write_text(
                "id: resident\nname: Resident\npersonality: Test.\nrole: Test.\noutputs: [display/missing]\n",
                encoding="utf-8")
            with patch("resident.__main__._provider") as provider:
                with self.assertRaisesRegex(
                        ValueError, "Unknown or unavailable output grants for resident: display/missing"):
                    build_host(Config(root / "data", residents_dir=definitions))
                provider.assert_not_called()

    def test_notify_owner_grant_requires_an_owner_route(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            definitions = root / "residents"
            definitions.mkdir()
            (definitions / "resident.yaml").write_text(
                "id: resident\nname: Resident\npersonality: Test.\nrole: Test.\noutputs: []\n",
                encoding="utf-8")
            (definitions / "helper.yaml").write_text(
                "id: helper\nname: Helper\npersonality: Test.\nrole: Test.\n"
                "outputs: [notify_owner]\n", encoding="utf-8")
            with patch("resident.__main__._provider", return_value=OutputProvider()):
                with self.assertRaisesRegex(
                        ValueError, "Unknown or unavailable output grants for helper: notify_owner"):
                    build_host(Config(root / "data", residents_dir=definitions))

    def test_rejects_inline_secrets_and_prompt_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "residents").mkdir()
            definition = root / "residents" / "resident.yaml"
            definition.write_text("id: resident\nname: Resident\nagent:\n  api_key: nope\n",
                                  encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Inline secret"):
                load_resident_catalog(root / "residents")
            definition.write_text(
                "id: resident\nname: Resident\npersonality_prompt: ../outside.md\nrole: Test.\n",
                encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prompt root"):
                load_resident_catalog(root / "residents")

    def test_rejects_removed_memory_policy_field(self):
        with tempfile.TemporaryDirectory() as temporary:
            definitions = Path(temporary) / "residents"
            definitions.mkdir()
            definition = definitions / "resident.yaml"
            definition.write_text(
                "id: resident\nname: Resident\npersonality: Test.\nrole: Test.\n"
                "memory: {}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unknown fields"):
                load_resident_catalog(definitions)

    def test_validates_declarative_curator_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            definitions = Path(temporary) / "residents"
            definitions.mkdir()
            definition = definitions / "resident.yaml"
            base = "id: resident\nname: Resident\npersonality: Test.\nrole: Test.\ncurator:\n"
            cases = (
                ("  api_key: inline\n", "Inline secret"),
                ("  api_key_env: not-an-env\n", "must name an environment variable"),
                ("  batch_size: 0\n", "must be a positive integer"),
                ("  batch_size: 101\n", "must be at most 100"),
                ("  max_batches: false\n", "must be a positive integer"),
                ("  surprise: true\n", "Unknown curator fields"),
            )
            for body, message in cases:
                with self.subTest(body=body):
                    definition.write_text(base + body, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        load_resident_catalog(definitions)

    def test_curator_without_model_is_disabled_and_does_not_resolve_secret(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            definitions = root / "residents"
            definitions.mkdir()
            (definitions / "resident.yaml").write_text(
                "id: resident\nname: Resident\npersonality: Test.\nrole: Test.\n"
                "curator:\n  api_key_env: MISSING_CURATOR_KEY\n",
                encoding="utf-8")

            with patch.dict(os.environ, {"OPENAI_API_KEY": "resident-key"}, clear=True):
                host = build_host(Config(root / "data", residents_dir=definitions))
            try:
                runtime = host.runtimes["resident"]
                self.assertIsNone(runtime.config.curator_model)
                self.assertIsNone(runtime.config.curator_api_key)
                self.assertIsNone(runtime.curator)
            finally:
                host.close()

    def test_rejects_unknown_and_malformed_subscriptions(self):
        with tempfile.TemporaryDirectory() as temporary:
            definitions = Path(temporary) / "residents"
            definitions.mkdir()
            definition = definitions / "resident.yaml"
            for selector, message in (("homeops.typo", "Unknown subscription"),
                                      ("homeops.*", "Malformed subscription")):
                with self.subTest(selector=selector):
                    definition.write_text(
                        "id: resident\nname: Resident\npersonality: Test.\nrole: Test.\n"
                        f"subscriptions: [{selector}]\n", encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        load_resident_catalog(definitions)

    def test_explicit_legacy_migration_preserves_database(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            source = data / "resident.sqlite3"
            connection = sqlite3.connect(source)
            connection.execute("CREATE TABLE marker(value TEXT)")
            connection.execute("INSERT INTO marker VALUES('kept')")
            connection.commit()
            connection.close()
            target = migrate_legacy_state(data)
            self.assertFalse(source.exists())
            connection = sqlite3.connect(target)
            self.assertEqual("kept", connection.execute("SELECT value FROM marker").fetchone()[0])
            connection.close()


class RuntimeHostTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.mailbox = Mailbox(root / "runtime" / "mailbox.sqlite3")
        self.a = ResidentRuntime(Config(root / "instances" / "a", instance_id="a"), IdleProvider(),
                                 capabilities=[])
        self.b = ResidentRuntime(Config(root / "instances" / "b", instance_id="b"), IdleProvider(),
                                 capabilities=[])
        self.host = RuntimeHost(
            {"a": self.a, "b": self.b},
            {"a": InstancePolicy(frozenset({"homeops", "messaging"})),
             "b": InstancePolicy(frozenset({"homeops.changed", "messaging"}))},
            self.mailbox, default_id="a")

    async def asyncTearDown(self):
        self.host.close()
        self.temporary.cleanup()

    async def test_routing_fans_out_only_to_subscribers(self):
        changed = WakeEvent("event", "homeops", "changed", utc_now(), {})
        self.assertEqual(("a", "b"), await self.host.route(changed))
        self.assertEqual("event", (await self.host.queues["a"].get()).id)
        self.assertEqual("event", (await self.host.queues["b"].get()).id)
        ignored = WakeEvent("ignored", "camera", "changed", utc_now(), {})
        self.assertEqual((), await self.host.route(ignored))

    async def test_instance_producer_queue_boundary_records_queue_wait(self):
        class DirectProducer:
            def __init__(self):
                self.produced = asyncio.Event()

            async def run(self, queue, stop):
                await queue.put(WakeEvent(
                    "direct-event", "telegram", "message", utc_now(), {}))
                self.produced.set()
                await stop.wait()

        root = Path(self.temporary.name)
        runtime = ResidentRuntime(
            Config(root / "direct", instance_id="direct", timeline=True),
            IdleProvider(), capabilities=[], owner_output=lambda _: None,
            diagnostic_output=lambda _: None)
        mailbox = Mailbox(root / "direct-mailbox.sqlite3")
        producer = DirectProducer()
        host = RuntimeHost(
            {"direct": runtime}, {"direct": InstancePolicy(frozenset({"*"}))},
            mailbox, default_id="direct", instance_producers={"direct": (producer,)})
        stop = asyncio.Event()
        tasks = []
        try:
            tasks, _ = await host._collect_startup_readiness(asyncio.Queue(), stop)
            await producer.produced.wait()
            event = await host.queues["direct"].get()
            runtime.observe_dequeue(event, host.queues["direct"].qsize())
            await runtime.process(event)

            rows = runtime.store.connection.execute(
                "SELECT data_json FROM journal WHERE event_type='timeline' "
                "ORDER BY sequence").fetchall()
            events = [json.loads(row[0]) for row in rows]
            enqueue = next(item for item in events
                           if item["operation"] == "host.enqueue")
            dequeue = next(item for item in events
                           if item["operation"] == "host.dequeue")
            self.assertEqual("direct-event", enqueue["event_id"])
            self.assertIsNotNone(dequeue["queue_wait_seconds"])
            self.assertGreaterEqual(dequeue["queue_wait_seconds"], 0.0)
        finally:
            stop.set()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            host.close()

    async def test_event_loop_lag_is_idle_silent_and_aggregated_per_wake(self):
        class WaitingProvider:
            async def respond(self, context, tools, results, continuation_id=None):
                await asyncio.sleep(0.02)
                return ModelTurn("turn", None, ())

        root = Path(self.temporary.name)
        runtime = ResidentRuntime(
            Config(root / "lag", instance_id="lag", timeline=True),
            WaitingProvider(), capabilities=[], owner_output=lambda _: None,
            diagnostic_output=lambda _: None)
        mailbox = Mailbox(root / "lag-mailbox.sqlite3")
        host = RuntimeHost(
            {"lag": runtime}, {"lag": InstancePolicy(frozenset({"*"}))},
            mailbox, default_id="lag")
        probe = EventLoopLagProbe((runtime.observe_event_loop_lag,), interval=0.001)
        await probe.start()
        runtime.bind_event_loop_lag_checkpoint(probe.checkpoint)
        try:
            await asyncio.sleep(0.005)
            idle_count = runtime.store.connection.execute(
                "SELECT count(*) FROM journal WHERE event_type='timeline'"
            ).fetchone()[0]
            self.assertEqual(0, idle_count)

            await runtime.process(WakeEvent(
                "lag-event", "test", "lag", utc_now(), {}))
            lag_rows = runtime.store.connection.execute(
                "SELECT data_json FROM journal WHERE event_type='timeline'"
            ).fetchall()
            lag_events = [event for row in lag_rows
                          if (event := json.loads(row[0])).get("operation")
                          == "event_loop.lag"]
            self.assertEqual(1, len(lag_events))
            self.assertEqual("summary", lag_events[0]["moment"])
            self.assertGreater(lag_events[0]["sample_count"], 0)
            self.assertGreaterEqual(lag_events[0]["max_event_loop_lag_seconds"], 0.0)
        finally:
            runtime.bind_event_loop_lag_checkpoint(None)
            await probe.stop()
            host.close()

    async def test_end_of_wake_blocking_is_included_in_lag_summary(self):
        root = Path(self.temporary.name)
        runtime = ResidentRuntime(
            Config(root / "final-lag", instance_id="final-lag", timeline=True),
            IdleProvider(), capabilities=[], owner_output=lambda _: None,
            diagnostic_output=lambda _: None)
        mailbox = Mailbox(root / "final-lag-mailbox.sqlite3")
        host = RuntimeHost(
            {"final-lag": runtime},
            {"final-lag": InstancePolicy(frozenset({"*"}))},
            mailbox, default_id="final-lag")
        original_finish_run = runtime.store.finish_run
        host_task = None

        def blocking_finish_run(*args, **kwargs):
            time.sleep(0.02)
            return original_finish_run(*args, **kwargs)

        try:
            fast_probe = lambda observers: EventLoopLagProbe(observers, interval=0.001)
            with (patch.object(runtime.store, "finish_run", side_effect=blocking_finish_run),
                  patch("resident.host.EventLoopLagProbe", fast_probe)):
                host_task = asyncio.create_task(host.run(interactive=False))
                await host.queues["final-lag"].put(WakeEvent(
                    "final-lag-event", "test", "lag", utc_now(), {}))
                for _ in range(100):
                    row = runtime.store.connection.execute(
                        "SELECT data_json FROM journal WHERE event_type='timeline' "
                        "AND json_extract(data_json, '$.operation')='event_loop.lag'"
                    ).fetchone()
                    if row is not None:
                        break
                    await asyncio.sleep(0.005)
                else:
                    self.fail("Hosted wake did not emit a lag summary")
            row = runtime.store.connection.execute(
                "SELECT data_json FROM journal WHERE event_type='timeline' "
                "AND json_extract(data_json, '$.operation')='event_loop.lag'"
            ).fetchone()
            summary = json.loads(row[0])
            self.assertGreater(summary["sample_count"], 0)
            self.assertGreater(summary["max_event_loop_lag_seconds"], 0.005)
        finally:
            if host_task is not None:
                host_task.cancel()
                await asyncio.gather(host_task, return_exceptions=True)
            host.close()

    async def test_durable_message_handoff_and_reply_are_independent(self):
        send_a = messaging_capability(self.mailbox, "a", lambda: self.host.recipients)
        self.assertEqual(["a", "b"],
                         send_a.input_schema["properties"]["recipient"]["enum"])
        result = await send_a.handler({"recipient": "b", "content": "hello"})
        self.assertEqual("pending", result["status"])
        await self.host.deliver_mailbox()
        event = await self.host.queues["b"].get()
        self.assertEqual("a", event.payload["sender"])
        self.assertEqual("delivered", self.mailbox.get(result["message_id"])["status"])

        send_b = messaging_capability(self.mailbox, "b", lambda: self.host.recipients)
        reply = await send_b.handler({"recipient": "a", "content": "reply"})
        await self.host.deliver_mailbox()
        reply_event = await self.host.queues["a"].get()
        self.assertEqual(reply["message_id"], reply_event.payload["message_id"])

    async def test_generic_messaging_does_not_address_owner(self):
        self.assertEqual(frozenset({"a", "b"}), self.host.recipients)
        send_a = messaging_capability(self.mailbox, "a", lambda: self.host.recipients)

        with self.assertRaisesRegex(ValueError, "Unknown message recipient: owner"):
            await send_a.handler({"recipient": "owner", "content": "hello"})

    async def test_mailbox_does_not_deliver_or_wake_a_non_subscriber(self):
        self.host.policies["b"] = InstancePolicy(frozenset({"homeops"}))
        send_a = messaging_capability(self.mailbox, "a", lambda: self.host.recipients)
        result = await send_a.handler({"recipient": "b", "content": "private"})

        self.assertEqual(0, await self.host.deliver_mailbox())
        self.assertTrue(self.host.queues["b"].empty())
        self.assertEqual("pending", self.mailbox.get(result["message_id"])["status"])

    async def test_host_waits_for_connector_readiness_before_terminal(self):
        class GatedProducer:
            readiness_items = (ReadinessItem("connector", "Connector"),)

            def __init__(self):
                self.started = asyncio.Event()
                self.release = asyncio.Event()

            async def run(self, queue, stop, readiness):
                self.started.set()
                await self.release.wait()
                readiness.put_nowait(ReadinessResult("connector", False, "initial attempt"))
                await stop.wait()

        producer = GatedProducer()
        output = []
        self.host.event_producers = (producer,)
        self.host.diagnostic_output = output.append
        terminal = AsyncMock(return_value="/quit")
        with patch("resident.host.asyncio.to_thread", new=terminal):
            run = asyncio.create_task(self.host.run())
            await producer.started.wait()
            await asyncio.sleep(0)
            self.assertEqual(0, terminal.await_count)
            self.assertEqual([], output)
            producer.release.set()
            await run

        self.assertEqual("Connector........... FAILED (initial attempt)", output[0])
        self.assertEqual("Startup completed with connector errors.", output[1])
        self.assertTrue(output[2].startswith("Runtime host started"))

    async def test_instance_state_and_identity_are_isolated_and_stable(self):
        self.a.store.create_intention("only a")
        self.assertEqual([], self.b.store.pending_intentions())
        identity = self.a.resident.id
        self.a.close()
        reopened = ResidentRuntime(
            Config(Path(self.temporary.name) / "instances" / "a", instance_id="a",
                   personality="Changed without becoming someone else"), IdleProvider(),
            capabilities=[])
        self.host.runtimes["a"] = reopened
        self.a = reopened
        self.assertEqual(identity, reopened.resident.id)
        self.assertEqual("Changed without becoming someone else", reopened.resident.personality)
