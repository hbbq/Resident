from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from resident.config import Config
from resident.runtime import ResidentRuntime
from resident.telegram import TelegramTransport, TelegramTransportError


class FakeTelegramTransport(TelegramTransport):
    def __init__(self, responses=()):
        super().__init__("secret-token", 101, 202)
        self.responses = iter(responses)
        self.requests = []

    async def _request(self, method, parameters):
        self.requests.append((method, parameters))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


class TelegramTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_authorized_private_text_uses_canonical_owner_message_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = ResidentRuntime(
                Config(Path(temporary)), object(), owner_output=lambda _: None,
                diagnostic_output=lambda _: None,
            )
            transport = FakeTelegramTransport([{"ok": True, "result": [{
                "update_id": 7,
                "message": {"chat": {"id": 202, "type": "private"},
                            "from": {"id": 101}, "text": "hello from Telegram"},
            }]}])
            offsets = []
            transport.bind_owner_message(runtime.owner_message_event)
            transport.bind_offset_checkpoint(lambda: None, offsets.append)
            queue = asyncio.Queue()

            polling = asyncio.create_task(transport.poll_once(queue, None))
            event = await queue.get()

            self.assertEqual(("owner", "owner_message"), (event.source, event.reason))
            self.assertEqual("hello from Telegram", event.payload["content"])
            transport.acknowledge_owner_message(event.id, True)
            offset = await polling
            stored = runtime.store.connection.execute(
                "SELECT direction,content FROM messages").fetchone()
            self.assertEqual(("inbound", "hello from Telegram"), tuple(stored))
            self.assertEqual(8, offset)
            self.assertEqual([8], offsets)
            runtime.close()

    async def test_owner_update_is_replayed_when_process_stops_before_processing(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Config(Path(temporary))
            first_runtime = ResidentRuntime(
                config, object(), owner_output=lambda _: None,
                diagnostic_output=lambda _: None,
            )
            response = {"ok": True, "result": [{
                "update_id": 7,
                "message": {"chat": {"id": 202, "type": "private"},
                            "from": {"id": 101}, "text": "recover me"},
            }]}
            offsets = []
            first_transport = FakeTelegramTransport([response])
            first_transport.bind_owner_message(first_runtime.owner_message_event)
            first_transport.bind_offset_checkpoint(lambda: None, offsets.append)
            first_queue = asyncio.Queue()

            polling = asyncio.create_task(first_transport.poll_once(first_queue, None))
            event = await first_queue.get()
            self.assertEqual("recover me", event.payload["content"])
            polling.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await polling
            self.assertEqual([], offsets)
            first_runtime.close()

            second_runtime = ResidentRuntime(
                config, object(), owner_output=lambda _: None,
                diagnostic_output=lambda _: None,
            )
            second_transport = FakeTelegramTransport([response])
            second_transport.bind_owner_message(second_runtime.owner_message_event)
            second_transport.bind_offset_checkpoint(lambda: offsets[-1] if offsets else None, offsets.append)
            second_queue = asyncio.Queue()

            replay = asyncio.create_task(second_transport.poll_once(second_queue, None))
            replayed_event = await second_queue.get()
            second_transport.acknowledge_owner_message(replayed_event.id, True)

            self.assertEqual(8, await replay)
            self.assertEqual([8], offsets)
            second_runtime.close()

    async def test_unauthorized_group_and_sender_updates_are_discarded_and_acked(self):
        updates = [
            {"update_id": 1, "message": {"chat": {"id": 202, "type": "group"},
                                          "from": {"id": 101}, "text": "group"}},
            {"update_id": 2, "message": {"chat": {"id": 202, "type": "private"},
                                          "from": {"id": 999}, "text": "stranger"}},
            {"update_id": 3, "message": {"chat": {"id": 999, "type": "private"},
                                          "from": {"id": 101}, "text": "other chat"}},
        ]
        transport = FakeTelegramTransport([{"ok": True, "result": updates}])
        transport.bind_owner_message(lambda _: self.fail("Unauthorized update was accepted"))
        queue = asyncio.Queue()

        self.assertEqual(4, await transport.poll_once(queue, None))
        self.assertTrue(queue.empty())

    async def test_send_chunks_without_changing_content(self):
        content = "a" * 4096 + "b" * 4096 + "tail"
        transport = FakeTelegramTransport([
            {"ok": True, "result": {}}, {"ok": True, "result": {}}, {"ok": True, "result": {}},
        ])

        await transport.send_text(content)

        chunks = [parameters["text"] for method, parameters in transport.requests]
        self.assertTrue(all(method == "sendMessage" for method, _ in transport.requests))
        self.assertEqual(content, "".join(chunks))
        self.assertEqual([4096, 4096, 4], list(map(len, chunks)))

    async def test_raw_http_failure_is_replaced_with_safe_error(self):
        transport = TelegramTransport("super-secret", 101, 202)
        with patch("resident.telegram.urlopen", side_effect=RuntimeError("leaked super-secret URL")):
            with self.assertRaisesRegex(TelegramTransportError, "Telegram sendMessage request failed") as raised:
                await transport.send_text("hello")
        self.assertNotIn("super-secret", str(raised.exception))

    async def test_configured_webhook_prevents_long_polling(self):
        transport = FakeTelegramTransport([{"ok": True, "result": {"url": "https://example.invalid/hook"}}])

        with self.assertRaisesRegex(TelegramTransportError, "webhook is configured"):
            await transport._check_webhook()


class TelegramRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_telegram_is_authoritative_and_terminal_is_only_a_successful_mirror(self):
        with tempfile.TemporaryDirectory() as temporary:
            mirror = []
            transport = FakeTelegramTransport([{"ok": True, "result": {}}])
            runtime = ResidentRuntime(
                Config(Path(temporary)), object(), owner_transport=transport,
                owner_output=mirror.append, diagnostic_output=lambda _: None,
            )

            result = await runtime._send_owner_message("remote message")

            self.assertTrue(result["delivered"])
            self.assertEqual(["remote message"], mirror)
            runtime.close()

    async def test_telegram_failure_is_persisted_while_terminal_remains_a_local_mirror(self):
        with tempfile.TemporaryDirectory() as temporary:
            mirror = []
            transport = FakeTelegramTransport([TelegramTransportError("Telegram sendMessage request failed")])
            runtime = ResidentRuntime(
                Config(Path(temporary)), object(), owner_transport=transport,
                owner_output=mirror.append, diagnostic_output=lambda _: None,
            )

            result = await runtime._send_owner_message("remote message")

            self.assertFalse(result["delivered"])
            self.assertEqual(["remote message"], mirror)
            status = runtime.store.connection.execute(
                "SELECT delivery_status FROM messages WHERE direction='outbound'").fetchone()[0]
            self.assertEqual("transport_failed", status)
            runtime.close()

    async def test_closed_terminal_does_not_stop_remote_runtime(self):
        terminal_closed = asyncio.Event()
        remained_running = []

        class StopAfterTerminalCloses:
            async def run(self, queue, stop):
                await terminal_closed.wait()
                await asyncio.sleep(0)
                remained_running.append(not stop.is_set())
                stop.set()
                await queue.put(None)

        async def closed_terminal(*_):
            terminal_closed.set()
            raise EOFError

        with tempfile.TemporaryDirectory() as temporary:
            transport = FakeTelegramTransport()
            runtime = ResidentRuntime(
                Config(Path(temporary)), object(), owner_transport=transport,
                event_producers=[StopAfterTerminalCloses()], owner_output=lambda _: None,
                diagnostic_output=lambda _: None,
            )
            with patch("resident.runtime.asyncio.to_thread", new=closed_terminal):
                await runtime.run_interactive()

            self.assertEqual([True], remained_running)
            runtime.close()


class TelegramConfigTests(unittest.TestCase):
    def test_telegram_is_optional_and_complete_environment_configuration_is_secret_safe(self):
        names = {
            "RESIDENT_TELEGRAM_BOT_TOKEN": "",
            "RESIDENT_TELEGRAM_OWNER_USER_ID": "",
            "RESIDENT_TELEGRAM_OWNER_CHAT_ID": "",
        }
        with patch.dict(os.environ, names):
            disabled = Config.from_env_and_args(["--data-dir", ".resident"])
        names = {
            "RESIDENT_TELEGRAM_BOT_TOKEN": "secret-token",
            "RESIDENT_TELEGRAM_OWNER_USER_ID": "101",
            "RESIDENT_TELEGRAM_OWNER_CHAT_ID": "202",
        }
        with patch.dict(os.environ, names):
            enabled = Config.from_env_and_args(["--data-dir", ".resident"])

        self.assertIsNone(disabled.telegram_bot_token)
        self.assertEqual((101, 202), (enabled.telegram_owner_user_id, enabled.telegram_owner_chat_id))
        self.assertNotIn("secret-token", repr(enabled))
        self.assertNotIn("101", repr(enabled))
        self.assertNotIn("202", repr(enabled))

    def test_partial_or_nonnumeric_binding_is_rejected(self):
        base = {
            "RESIDENT_TELEGRAM_BOT_TOKEN": "secret-token",
            "RESIDENT_TELEGRAM_OWNER_USER_ID": "101",
            "RESIDENT_TELEGRAM_OWNER_CHAT_ID": "",
        }
        with patch.dict(os.environ, base):
            with self.assertRaisesRegex(ValueError, "Telegram requires"):
                Config.from_env_and_args(["--data-dir", ".resident"])
        base["RESIDENT_TELEGRAM_OWNER_CHAT_ID"] = "not-a-number"
        with patch.dict(os.environ, base):
            with self.assertRaisesRegex(ValueError, "must be integers"):
                Config.from_env_and_args(["--data-dir", ".resident"])
        base["RESIDENT_TELEGRAM_OWNER_CHAT_ID"] = "0"
        with patch.dict(os.environ, base):
            with self.assertRaisesRegex(ValueError, "positive integers"):
                Config.from_env_and_args(["--data-dir", ".resident"])


if __name__ == "__main__":
    unittest.main()
