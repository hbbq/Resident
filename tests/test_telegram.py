from __future__ import annotations

import asyncio
import os
import sqlite3
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from resident.__main__ import TerminalDiagnostics
from resident.config import Config
from resident.domain import ModelTurn
from resident.runtime import ResidentRuntime
from resident.store import Store
from resident.telegram import (
    TelegramAuthenticationError, TelegramTransport, TelegramTransportError,
    TelegramWebhookConflictError,
)


class FakeTelegramTransport(TelegramTransport):
    def __init__(self, responses=(), *, bot_token="secret-token"):
        super().__init__(bot_token, 101, 202)
        self.responses = iter(responses)
        self.requests = []

    async def _request(self, method, parameters):
        self.requests.append((method, parameters))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


class TelegramTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_ingest_is_durable_and_offset_advances_before_wake_processing(self):
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
            transport.bind_owner_message(runtime.telegram_owner_message_event)
            transport.bind_offset_checkpoint(lambda: None, offsets.append)
            queue = asyncio.Queue()

            offset = await transport.poll_once(queue, None)
            event = await queue.get()

            self.assertEqual(("owner", "owner_message"), (event.source, event.reason))
            self.assertEqual("hello from Telegram", event.payload["content"])
            self.assertNotIn("update_id", event.payload)
            stored = runtime.store.connection.execute(
                "SELECT direction,content FROM messages").fetchone()
            self.assertEqual(("inbound", "hello from Telegram"), tuple(stored))
            self.assertEqual(8, offset)
            self.assertEqual([8], offsets)
            runtime.close()

    async def test_refetched_pending_update_recreates_wake_without_duplicate_message(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Config(Path(temporary))
            runtime = ResidentRuntime(
                config, object(), owner_output=lambda _: None,
                diagnostic_output=lambda _: None,
            )
            response = {"ok": True, "result": [{
                "update_id": 7,
                "message": {"chat": {"id": 202, "type": "private"},
                            "from": {"id": 101}, "text": "recover me"},
            }]}
            offsets = []
            transport = FakeTelegramTransport([response, response])
            transport.bind_owner_message(runtime.telegram_owner_message_event)
            transport.bind_offset_checkpoint(lambda: None, offsets.append)
            queue = asyncio.Queue()

            self.assertEqual(8, await transport.poll_once(queue, None))
            self.assertEqual(8, await transport.poll_once(queue, None))

            first = queue.get_nowait()
            retried = queue.get_nowait()
            self.assertEqual(first.payload["message_id"], retried.payload["message_id"])
            self.assertEqual("recover me", retried.payload["content"])
            self.assertEqual(1, runtime.store.connection.execute(
                "SELECT count(*) FROM messages WHERE direction='inbound'").fetchone()[0])
            self.assertEqual(1, runtime.store.connection.execute(
                "SELECT count(*) FROM telegram_owner_updates WHERE update_id=7").fetchone()[0])
            runtime.close()

    async def test_failed_wake_does_not_block_later_updates(self):
        with tempfile.TemporaryDirectory() as temporary:
            class FailingProvider:
                async def respond(self, *_):
                    raise RuntimeError("provider failed")

            runtime = ResidentRuntime(
                Config(Path(temporary)), FailingProvider(), owner_output=lambda _: None,
                diagnostic_output=lambda _: None,
            )
            responses = [
                {"ok": True, "result": [{
                    "update_id": 7,
                    "message": {"chat": {"id": 202, "type": "private"},
                                "from": {"id": 101}, "text": "first"},
                }]},
                {"ok": True, "result": [{
                    "update_id": 8,
                    "message": {"chat": {"id": 202, "type": "private"},
                                "from": {"id": 101}, "text": "second"},
                }]},
            ]
            offsets = []
            transport = FakeTelegramTransport(responses)
            transport.bind_owner_message(runtime.telegram_owner_message_event)
            transport.bind_offset_checkpoint(lambda: offsets[-1] if offsets else None, offsets.append)
            queue = asyncio.Queue()

            offset = await transport.poll_once(queue, None)
            with self.assertRaisesRegex(RuntimeError, "provider failed"):
                await runtime.process(await queue.get())
            offset = await transport.poll_once(queue, offset)

            self.assertEqual(9, offset)
            self.assertEqual("second", (await queue.get()).payload["content"])
            self.assertEqual([8, 9], offsets)
            runtime.close()

    async def test_retry_after_offset_checkpoint_failure_recreates_pending_wake(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = ResidentRuntime(
                Config(Path(temporary)), object(), owner_output=lambda _: None,
                diagnostic_output=lambda _: None)
            response = {"ok": True, "result": [{
                "update_id": 7,
                "message": {"chat": {"id": 202, "type": "private"},
                            "from": {"id": 101}, "text": "recover me"},
            }]}
            later_response = {"ok": True, "result": [{
                "update_id": 8,
                "message": {"chat": {"id": 202, "type": "private"},
                            "from": {"id": 101}, "text": "later"},
            }]}
            transport = FakeTelegramTransport([response, response, later_response])
            transport.bind_owner_message(runtime.telegram_owner_message_event)
            offsets = []

            def save_offset(offset):
                offsets.append(offset)
                if len(offsets) == 1:
                    raise RuntimeError("crash before checkpoint")

            transport.bind_offset_checkpoint(lambda: None, save_offset)
            queue = asyncio.Queue()

            with self.assertRaisesRegex(RuntimeError, "crash before checkpoint"):
                await transport.poll_once(queue, None)
            self.assertTrue(queue.empty())
            self.assertEqual(1, runtime.store.connection.execute(
                "SELECT count(*) FROM messages WHERE direction='inbound'").fetchone()[0])
            self.assertEqual(1, runtime.store.connection.execute(
                "SELECT count(*) FROM telegram_owner_updates WHERE update_id=7").fetchone()[0])

            offset = await transport.poll_once(queue, None)
            recovered = queue.get_nowait()
            stored_id = runtime.store.connection.execute(
                "SELECT id FROM messages WHERE content='recover me'").fetchone()["id"]
            self.assertEqual(stored_id, recovered.payload["message_id"])
            self.assertEqual("recover me", recovered.payload["content"])
            self.assertEqual([8, 8], offsets)
            self.assertEqual(1, runtime.store.connection.execute(
                "SELECT count(*) FROM messages WHERE direction='inbound'").fetchone()[0])
            self.assertEqual(1, runtime.store.connection.execute(
                "SELECT count(*) FROM telegram_owner_updates WHERE update_id=7").fetchone()[0])

            self.assertEqual(9, await transport.poll_once(queue, offset))
            self.assertEqual("later", queue.get_nowait().payload["content"])
            self.assertEqual([8, 8, 9], offsets)
            self.assertEqual(2, runtime.store.connection.execute(
                "SELECT count(*) FROM messages WHERE direction='inbound'").fetchone()[0])
            runtime.close()

    async def test_multi_update_retry_reloads_checkpoint_without_duplicate_wakes(self):
        stop = asyncio.Event()
        updates = [
            {
                "update_id": update_id,
                "message": {"chat": {"id": 202, "type": "private"},
                            "from": {"id": 101}, "text": text},
            }
            for update_id, text in ((7, "first"), (8, "second"), (9, "recover me"))
        ]

        class CheckpointAwareTransport(TelegramTransport):
            def __init__(self):
                super().__init__("secret-token", 101, 202)
                self.request_offsets = []

            async def _check_webhook(self):
                pass

            async def _request(self, method, parameters):
                self.request_offsets.append(parameters.get("offset"))
                requested = parameters.get("offset")
                return {"ok": True, "result": [
                    update for update in updates
                    if requested is None or update["update_id"] >= requested
                ]}

            async def poll_once(self, queue, offset):
                result = await super().poll_once(queue, offset)
                stop.set()
                return result

        class RecordingProvider:
            def __init__(self):
                self.calls = 0

            async def respond(self, *_):
                self.calls += 1
                return ModelTurn("done")

        with tempfile.TemporaryDirectory() as temporary:
            provider = RecordingProvider()
            runtime = ResidentRuntime(
                Config(Path(temporary)), provider, owner_output=lambda _: None,
                diagnostic_output=lambda _: None)
            transport = CheckpointAwareTransport()
            ingestion_attempts = []

            def owner_message_event(bot_identity, update_id, content):
                ingestion_attempts.append(update_id)
                return runtime.telegram_owner_message_event(bot_identity, update_id, content)

            persisted_offset = None
            failed_checkpoint = False

            def load_offset():
                return persisted_offset

            def save_offset(offset):
                nonlocal persisted_offset, failed_checkpoint
                if offset == 10 and not failed_checkpoint:
                    failed_checkpoint = True
                    raise RuntimeError("crash before later checkpoint")
                persisted_offset = offset

            transport.bind_owner_message(owner_message_event)
            transport.bind_offset_checkpoint(load_offset, save_offset)
            queue = asyncio.Queue()

            async def immediate_timeout(awaitable, *, timeout):
                awaitable.close()
                raise TimeoutError

            with patch("resident.telegram.asyncio.wait_for", new=immediate_timeout):
                await transport.run(queue, stop)

            self.assertEqual([None, 9], transport.request_offsets)
            self.assertEqual([7, 8, 9, 9], ingestion_attempts)
            self.assertEqual(10, persisted_offset)
            events = [queue.get_nowait() for _ in range(queue.qsize())]
            self.assertEqual(["first", "second", "recover me"], [
                event.payload["content"] for event in events])
            self.assertEqual(3, len({event.payload["message_id"] for event in events}))

            for event in events:
                await runtime.process(event)

            self.assertEqual(3, provider.calls)
            self.assertEqual(3, runtime.store.connection.execute(
                "SELECT count(*) FROM messages WHERE direction='inbound'").fetchone()[0])
            self.assertEqual(3, runtime.store.connection.execute(
                "SELECT count(*) FROM telegram_owner_updates").fetchone()[0])
            self.assertEqual((3, 3), tuple(runtime.store.connection.execute("""
                SELECT count(*), sum(status='completed') FROM owner_message_processing
            """).fetchone()))
            runtime.close()

    async def test_refetched_completed_update_does_not_recreate_wake(self):
        class CompleteProvider:
            async def respond(self, *_):
                return ModelTurn("done")

        with tempfile.TemporaryDirectory() as temporary:
            runtime = ResidentRuntime(
                Config(Path(temporary)), CompleteProvider(), owner_output=lambda _: None,
                diagnostic_output=lambda _: None)
            response = {"ok": True, "result": [{
                "update_id": 7,
                "message": {"chat": {"id": 202, "type": "private"},
                            "from": {"id": 101}, "text": "completed"},
            }]}
            transport = FakeTelegramTransport([response, response])
            transport.bind_owner_message(runtime.telegram_owner_message_event)
            queue = asyncio.Queue()

            await transport.poll_once(queue, None)
            await runtime.process(queue.get_nowait())
            await transport.poll_once(queue, None)

            self.assertTrue(queue.empty())
            self.assertEqual(1, runtime.store.connection.execute(
                "SELECT count(*) FROM messages WHERE direction='inbound'").fetchone()[0])
            self.assertEqual(1, runtime.store.connection.execute(
                "SELECT count(*) FROM telegram_owner_updates WHERE update_id=7").fetchone()[0])
            runtime.close()

    async def test_restart_after_ack_before_processing_recreates_owner_wake(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Config(Path(temporary))
            first_runtime = ResidentRuntime(
                config, object(), owner_output=lambda _: None, diagnostic_output=lambda _: None)
            transport = FakeTelegramTransport([{"ok": True, "result": [{
                "update_id": 7,
                "message": {"chat": {"id": 202, "type": "private"},
                            "from": {"id": 101}, "text": "recover after ack"},
            }]}])
            offsets = []
            transport.bind_owner_message(first_runtime.telegram_owner_message_event)
            transport.bind_offset_checkpoint(lambda: None, offsets.append)

            await transport.poll_once(asyncio.Queue(), None)
            self.assertEqual([8], offsets)
            first_runtime.close()

            restarted = ResidentRuntime(
                config, object(), owner_output=lambda _: None, diagnostic_output=lambda _: None)
            queue = asyncio.Queue()
            await restarted.enqueue_startup_wakeups(queue)

            event = queue.get_nowait()
            self.assertEqual(("owner", "owner_message"), (event.source, event.reason))
            self.assertEqual("recover after ack", event.payload["content"])
            self.assertNotIn("update_id", event.payload)
            restarted.close()

    async def test_successful_owner_wake_is_not_recreated_after_restart(self):
        class CompleteProvider:
            async def respond(self, *_):
                return ModelTurn("done")

        with tempfile.TemporaryDirectory() as temporary:
            config = Config(Path(temporary))
            runtime = ResidentRuntime(
                config, CompleteProvider(), owner_output=lambda _: None,
                diagnostic_output=lambda _: None)
            await runtime.process(runtime.owner_message_event("completed"))
            runtime.close()

            restarted = ResidentRuntime(
                config, CompleteProvider(), owner_output=lambda _: None,
                diagnostic_output=lambda _: None)
            queue = asyncio.Queue()
            await restarted.enqueue_startup_wakeups(queue)

            self.assertTrue(queue.empty())
            restarted.close()

    async def test_failed_owner_wake_is_recoverable_after_restart(self):
        class FailingProvider:
            async def respond(self, *_):
                raise RuntimeError("interrupted")

        with tempfile.TemporaryDirectory() as temporary:
            config = Config(Path(temporary))
            runtime = ResidentRuntime(
                config, FailingProvider(), owner_output=lambda _: None,
                diagnostic_output=lambda _: None)
            event = runtime.owner_message_event("try again")
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                await runtime.process(event)
            runtime.close()

            restarted = ResidentRuntime(
                config, object(), owner_output=lambda _: None, diagnostic_output=lambda _: None)
            queue = asyncio.Queue()
            await restarted.enqueue_startup_wakeups(queue)

            recovered = queue.get_nowait()
            self.assertEqual(event.payload["message_id"], recovered.payload["message_id"])
            self.assertEqual("try again", recovered.payload["content"])
            restarted.close()

    async def test_interrupted_owner_wake_is_recoverable_after_restart(self):
        class InterruptedProvider:
            async def respond(self, *_):
                raise asyncio.CancelledError

        with tempfile.TemporaryDirectory() as temporary:
            config = Config(Path(temporary))
            runtime = ResidentRuntime(
                config, InterruptedProvider(), owner_output=lambda _: None,
                diagnostic_output=lambda _: None)
            event = runtime.owner_message_event("resume me")
            with self.assertRaises(asyncio.CancelledError):
                await runtime.process(event)
            runtime.close()

            restarted = ResidentRuntime(
                config, object(), owner_output=lambda _: None, diagnostic_output=lambda _: None)
            queue = asyncio.Queue()
            await restarted.enqueue_startup_wakeups(queue)
            self.assertEqual("resume me", queue.get_nowait().payload["content"])
            restarted.close()

    async def test_same_update_id_from_different_bots_is_independent(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = ResidentRuntime(
                Config(Path(temporary)), object(), owner_output=lambda _: None,
                diagnostic_output=lambda _: None)
            update = {"ok": True, "result": [{
                "update_id": 7,
                "message": {"chat": {"id": 202, "type": "private"},
                            "from": {"id": 101}, "text": "bot message"},
            }]}
            first = FakeTelegramTransport([update])
            second = FakeTelegramTransport([update], bot_token="different-secret-token")
            first.bind_owner_message(runtime.telegram_owner_message_event)
            second.bind_owner_message(runtime.telegram_owner_message_event)
            queue = asyncio.Queue()

            await first.poll_once(queue, None)
            await second.poll_once(queue, None)

            self.assertEqual(2, queue.qsize())
            self.assertEqual(2, runtime.store.connection.execute(
                "SELECT count(*) FROM messages WHERE direction='inbound'").fetchone()[0])
            runtime.close()

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
        transport.bind_owner_message(lambda *_: self.fail("Unauthorized update was accepted"))
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

    async def test_invalid_credentials_are_a_safe_permanent_failure(self):
        token = "123:super-secret"
        response = HTTPError(
            f"https://api.telegram.org/bot{token}/getWebhookInfo",
            401,
            f"Unauthorized: revoked {token}",
            {},
            BytesIO(f'{{"description":"revoked {token}"}}'.encode()),
        )
        transport = TelegramTransport(token, 101, 202)

        with patch("resident.telegram.urlopen", side_effect=response):
            with self.assertRaises(TelegramAuthenticationError) as raised:
                await transport._check_webhook()

        diagnostic = str(raised.exception)
        self.assertIn("check RESIDENT_TELEGRAM_BOT_TOKEN", diagnostic)
        self.assertNotIn(token, diagnostic)
        self.assertNotIn("super-secret", diagnostic)
        self.assertNotIn("revoked 123", diagnostic)

    async def test_authentication_failure_stops_polling_without_retry(self):
        diagnostics = []
        transport = FakeTelegramTransport([
            TelegramAuthenticationError(
                "Telegram bot authentication failed; check RESIDENT_TELEGRAM_BOT_TOKEN "
                "and replace it if the token was revoked"),
        ], bot_token="123:super-secret")
        transport.diagnostic_output = diagnostics.append

        await transport.run(asyncio.Queue(), asyncio.Event())

        self.assertEqual([("getWebhookInfo", {})], transport.requests)
        self.assertEqual(1, len(diagnostics))
        self.assertIn("permanent failure", diagnostics[0])
        self.assertIn("check RESIDENT_TELEGRAM_BOT_TOKEN", diagnostics[0])
        self.assertNotIn("123:super-secret", diagnostics[0])
        self.assertNotIn("Unauthorized", diagnostics[0])

    async def test_configured_webhook_prevents_long_polling(self):
        transport = FakeTelegramTransport([{"ok": True, "result": {"url": "https://example.invalid/hook"}}])

        with self.assertRaisesRegex(TelegramWebhookConflictError, "webhook is configured"):
            await transport._check_webhook()

    async def test_webhook_conflict_stops_polling_without_retry_and_redacts_credentials(self):
        token = "123:super-secret"
        diagnostics = []
        transport = FakeTelegramTransport(
            [{"ok": True, "result": {"url": f"https://example.invalid/{token}"}}],
            bot_token=token,
        )
        transport.diagnostic_output = diagnostics.append

        await transport.run(asyncio.Queue(), asyncio.Event())

        self.assertEqual([("getWebhookInfo", {})], transport.requests)
        self.assertEqual(1, len(diagnostics))
        self.assertIn("permanent failure", diagnostics[0])
        self.assertIn("webhook is configured", diagnostics[0])
        self.assertNotIn(token, diagnostics[0])
        self.assertNotIn("super-secret", diagnostics[0])

    async def test_transient_polling_failure_still_retries(self):
        stop = asyncio.Event()

        class RetryTransport(TelegramTransport):
            def __init__(self):
                super().__init__("secret-token", 101, 202, diagnostic_output=diagnostics.append)
                self.webhook_checks = 0
                self.polls = 0

            async def _check_webhook(self):
                self.webhook_checks += 1

            async def poll_once(self, queue, offset):
                self.polls += 1
                if self.polls == 1:
                    raise TelegramTransportError("Telegram getUpdates request failed")
                stop.set()
                return offset

        diagnostics = []
        transport = RetryTransport()

        async def immediate_timeout(awaitable, *, timeout):
            awaitable.close()
            raise TimeoutError

        with patch("resident.telegram.asyncio.wait_for", new=immediate_timeout):
            await transport.run(asyncio.Queue(), stop)

        self.assertEqual(1, transport.webhook_checks)
        self.assertEqual(2, transport.polls)
        self.assertEqual(
            ["poll failed: TelegramTransportError: Telegram getUpdates request failed"],
            diagnostics,
        )

    def test_permanent_webhook_failure_is_visible_without_verbose_diagnostics(self):
        with patch("builtins.print") as output:
            TerminalDiagnostics(verbose=False).telegram(
                "permanent failure: Telegram long polling is unavailable while a webhook is configured")

        output.assert_called_once_with(
            "[telegram] permanent failure: Telegram long polling is unavailable while a webhook is configured")

    def test_permanent_authentication_failure_is_visible_without_verbose_diagnostics(self):
        message = (
            "permanent failure: Telegram bot authentication failed; "
            "check RESIDENT_TELEGRAM_BOT_TOKEN and replace it if the token was revoked")
        with patch("builtins.print") as output:
            TerminalDiagnostics(verbose=False).telegram(message)

        output.assert_called_once_with(f"[telegram] {message}")

    def test_bot_scoped_offsets_are_independent_and_secret_safe(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            first = TelegramTransport("123:first-super-secret", 101, 202)
            second = TelegramTransport("456:second-super-secret", 101, 202)

            self.assertNotEqual(first.offset_checkpoint_scope, second.offset_checkpoint_scope)
            store.save_observed_snapshot(first.offset_checkpoint_scope, 8)
            store.save_observed_snapshot(second.offset_checkpoint_scope, 42)

            self.assertEqual(8, store.observed_snapshot(first.offset_checkpoint_scope))
            self.assertEqual(42, store.observed_snapshot(second.offset_checkpoint_scope))
            dump = "\n".join(store.connection.iterdump())
            self.assertNotIn("first-super-secret", dump)
            self.assertNotIn("second-super-secret", dump)
            self.assertNotIn("123:first-super-secret", first.offset_checkpoint_scope)
            self.assertNotIn("456:second-super-secret", second.offset_checkpoint_scope)
            store.close()

    def test_legacy_global_update_mapping_migrates_without_cross_bot_collision(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "resident.sqlite3"
            connection = sqlite3.connect(path)
            connection.executescript("""
                CREATE TABLE schema_version(version INTEGER NOT NULL);
                INSERT INTO schema_version VALUES(4);
                CREATE TABLE messages(
                  id TEXT PRIMARY KEY, direction TEXT NOT NULL,
                  sender_id TEXT NOT NULL, content TEXT NOT NULL,
                  spontaneous INTEGER NOT NULL DEFAULT 0,
                  delivery_status TEXT NOT NULL, created_at TEXT NOT NULL);
                CREATE TABLE telegram_owner_updates(
                  update_id INTEGER PRIMARY KEY,
                  message_id TEXT NOT NULL UNIQUE REFERENCES messages(id));
                INSERT INTO messages VALUES(
                  'old-message','inbound','owner','old',0,'delivered','2026-01-01T00:00:00+00:00');
                INSERT INTO telegram_owner_updates VALUES(7,'old-message');
            """)
            connection.close()

            store = Store(path)
            new_message = store.ingest_telegram_owner_message(
                "new-bot-identity", 7, "owner", "new")

            self.assertIsNotNone(new_message)
            self.assertEqual(2, store.connection.execute(
                "SELECT count(*) FROM telegram_owner_updates WHERE update_id=7").fetchone()[0])
            store.close()


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
