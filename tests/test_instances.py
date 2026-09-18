import asyncio
import sqlite3
import tempfile
import unittest
from pathlib import Path

from resident.config import Config
from resident.domain import ModelTurn, WakeEvent
from resident.host import InstancePolicy, RuntimeHost, messaging_capability
from resident.instances import load_resident_catalog, migrate_legacy_state
from resident.mailbox import Mailbox
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
memory:
  enabled: false
capabilities: [messaging]
subscriptions: [messaging]
""", encoding="utf-8")
            catalog = load_resident_catalog(root / "residents", default_id="oracle")
            oracle = catalog.residents[0]
            self.assertEqual("Be precise.", oracle.personality)
            self.assertEqual("Answer narrow questions.", oracle.role)
            self.assertFalse(oracle.memory["enabled"])
            self.assertEqual(("messaging",), oracle.capabilities)

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
            {"a": InstancePolicy(frozenset({"homeops"})),
             "b": InstancePolicy(frozenset({"homeops.changed"}))},
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

    async def test_instance_state_and_identity_are_isolated_and_stable(self):
        self.a.store.remember("only a", "resident")
        self.assertEqual([], self.b.store.recall("only a"))
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
