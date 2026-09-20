import asyncio
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from resident.__main__ import build_host
from resident.config import Config, SUPPORTED_REASONING_EFFORTS
from resident.domain import ModelTurn, WakeEvent
from resident.host import InstancePolicy, RuntimeHost, messaging_capability
from resident.instances import load_resident_catalog, migrate_legacy_state
from resident.mailbox import Mailbox
from resident.readiness import ReadinessItem, ReadinessResult
from resident.runtime import ResidentRuntime
from resident.store import utc_now


class IdleProvider:
    async def respond(self, context, tools, results, continuation_id=None):
        return ModelTurn("turn", None, ())


class InstanceDefinitionTests(unittest.TestCase):
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

    def test_catalog_runtimes_inherit_curator_and_explicit_new_chapter(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            definitions = root / "residents"
            definitions.mkdir()
            for resident_id in ("resident", "helper"):
                (definitions / f"{resident_id}.yaml").write_text(
                    f"id: {resident_id}\nname: {resident_id.title()}\n"
                    "personality: Test.\nrole: Test.\n", encoding="utf-8")
            config = Config(
                root / "data", residents_dir=definitions, new_chapter=True,
                curator_model="curator-model", curator_api_key="curator-key",
                curator_base_url="https://curator.example/v1",
                curator_batch_size=17, curator_max_batches=3)

            with patch.dict(os.environ, {"OPENAI_API_KEY": "resident-key"}):
                host = build_host(config)
            try:
                self.assertEqual({"resident", "helper"}, set(host.runtimes))
                for runtime in host.runtimes.values():
                    self.assertEqual("curator-model", runtime.config.curator_model)
                    self.assertEqual("curator-key", runtime.config.curator_api_key)
                    self.assertEqual("https://curator.example/v1", runtime.config.curator_base_url)
                    self.assertEqual(17, runtime.curator.batch_size)
                    self.assertEqual(3, runtime.curator.max_batches)
                    self.assertEqual("curator-model", runtime.curator.model.model)
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
subscriptions: [homeops]
""", encoding="utf-8")
            catalog = load_resident_catalog(root / "residents", default_id="oracle")
            oracle = catalog.residents[0]
            self.assertEqual("Be precise.", oracle.personality)
            self.assertEqual("Answer narrow questions.", oracle.role)
            self.assertEqual(("messaging",), oracle.capabilities)
            self.assertEqual(("homeops",), oracle.subscriptions)

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

    def test_rejects_removed_memory_and_curator_policy_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            definitions = Path(temporary) / "residents"
            definitions.mkdir()
            definition = definitions / "resident.yaml"
            for removed_field in ("memory", "curator"):
                with self.subTest(field=removed_field):
                    definition.write_text(
                        "id: resident\nname: Resident\npersonality: Test.\nrole: Test.\n"
                        f"{removed_field}: {{}}\n", encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "Unknown fields"):
                        load_resident_catalog(definitions)

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
        stop = asyncio.Event()
        probe = asyncio.create_task(host._probe_event_loop_lag(stop, interval=0.001))
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
            stop.set()
            await probe
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
