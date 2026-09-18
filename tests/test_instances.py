import asyncio
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from resident.config import Config
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
