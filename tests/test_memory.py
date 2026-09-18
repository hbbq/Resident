from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from resident.memory import MemoryCurator, SessionItemPage
from resident.store import Store


class FakeSource:
    session_id = "session-1"

    def __init__(self):
        self.cursors = []

    async def session_items(self, cursor, limit):
        self.cursors.append(cursor)
        if cursor is not None:
            return SessionItemPage((), cursor, False)
        return SessionItemPage(({
            "id": "item-1", "type": "message", "role": "user",
            "created_at": "2026-01-01T00:00:00+00:00",
            "content": [{"type": "input_text", "text": "I prefer tea. token=secret"}],
        },), "item-1", False)


class FakeModel:
    def __init__(self):
        self.items = []

    async def curate(self, session_id, items, existing):
        self.items.extend(items)
        return {"mutations": [{
            "operation": "create", "kind": "preference", "content": "Owner prefers tea",
            "confidence": .9, "provenance": [{"item_id": "item-1", "source_type": "message",
                                                "excerpt": "token=secret; prefers tea"}],
        }], "handover": "Continue discussing tea."}


class EmptyModel:
    def __init__(self):
        self.calls = 0

    async def curate(self, session_id, items, existing):
        self.calls += 1
        return {"mutations": []}


class RecordingModel(EmptyModel):
    def __init__(self):
        super().__init__()
        self.pages = []

    async def curate(self, session_id, items, existing):
        self.calls += 1
        self.pages.append(tuple(items))
        return {"mutations": []}


class PageSource:
    session_id = "session-pages"

    def __init__(self, pages):
        self.pages = list(pages)
        self.calls = 0

    async def session_items(self, cursor, limit):
        self.calls += 1
        return self.pages.pop(0)


class MemoryStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_curator_checkpoint_and_idempotent_memory_with_redacted_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            source, model = FakeSource(), FakeModel()
            curator = MemoryCurator(store, source, model)

            handover = await curator.catch_up()
            await curator.catch_up()

            memories = store.search_memories("tea")
            self.assertEqual(1, len(memories))
            self.assertEqual("Continue discussing tea.", handover)
            self.assertEqual("item-1", store.curator_checkpoint(
                "openai_agents", "session-1")["cursor"])
            detail = store.memory(memories[0]["id"])
            self.assertIn("[REDACTED]", detail["provenance"][0]["excerpt"])
            self.assertNotIn("secret", str(model.items))
            store.close()

    async def test_unsupported_tool_fields_never_reach_curator_or_provenance(self):
        raw_secrets = (
            "sk_live_UNSUPPORTED123456", "AKIA1234567890ABCDEF",
            "structured-api-key", "structured-password", "output-token", "cookie-value",
        )

        class CredentialSource:
            session_id = "real-session"

            async def session_items(self, cursor, limit):
                if cursor is not None:
                    return SessionItemPage((), cursor, False)
                return SessionItemPage((
                    {
                        "id": "call-item", "type": "function_call", "name": "clock",
                        "call_id": "call-1", "arguments": {
                            "api_key": "structured-api-key",
                            "nested": {"password": "structured-password"},
                            "unknown": raw_secrets[1],
                        },
                    },
                    {
                        "id": "credential-item", "type": "function_call_output",
                        "created_at": "2026-02-03T04:05:06+00:00", "call_id": "call-1",
                        "content": [{"type": "input_text", "text": raw_secrets[0]}],
                        "output": {"token": "output-token", "cookie": "cookie-value",
                                   "unknown": {"nested": raw_secrets[0]}},
                        "error": raw_secrets[1],
                        "unsupported": raw_secrets[0],
                    },
                ), "credential-item", False)

        class HostileModel:
            def __init__(self):
                self.items = []

            async def curate(self, session_id, items, existing):
                self.items.extend(items)
                return {"mutations": [{
                    "operation": "create", "kind": "fact",
                    "content": f"Keep {raw_secrets[0]}",
                    "rationale": f"Authentication: {raw_secrets[1]}",
                    "provenance": [{
                        "item_id": "credential-item", "session_id": "invented-session",
                        "source_type": "invented-type", "timestamp": "invented-time",
                        "excerpt": "invented excerpt", "content_hash": "invented-hash",
                    }],
                }], "handover": f"Bearer {raw_secrets[4]}"}

        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            model = HostileModel()
            handover = await MemoryCurator(store, CredentialSource(), model).catch_up()

            model_input = json.dumps(model.items)
            for secret in raw_secrets:
                self.assertNotIn(secret, model_input)
            self.assertEqual([
                {"type": "function_call", "id": "call-item", "call_id": "call-1",
                 "name": "clock"},
                {"type": "function_call_output", "id": "credential-item",
                 "created_at": "2026-02-03T04:05:06+00:00", "call_id": "call-1"},
            ], model.items)
            memory = store.memory(store.search_memories()[0]["id"])
            persisted = json.dumps(memory)
            for secret in raw_secrets:
                self.assertNotIn(secret, persisted + str(handover))
            self.assertIn("[REDACTED]", memory["content"])
            self.assertIn("[REDACTED]", memory["rationale"])
            self.assertEqual("Bearer [REDACTED]", handover)
            evidence = memory["provenance"][0]
            self.assertEqual("real-session", evidence["source_session_id"])
            self.assertEqual("function_call_output", evidence["source_type"])
            self.assertEqual("2026-02-03T04:05:06+00:00", evidence["source_timestamp"])
            self.assertNotIn("invented", json.dumps(evidence))
            canonical = json.dumps(model.items[1], ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":"))
            self.assertEqual(hashlib.sha256(canonical.encode()).hexdigest(),
                             evidence["content_hash"])
            self.assertEqual(canonical[:500], evidence["excerpt"])
            dump = "\n".join(store.connection.iterdump())
            for secret in raw_secrets:
                self.assertNotIn(secret, dump)
            store.close()

    async def test_allowed_text_is_preserved_while_common_credentials_are_redacted(self):
        secrets = (
            "sk_live_1234567890abcdef", "rk_test_abcdef1234567890",
            "AKIAIOSFODNN7EXAMPLE", "ASIAIOSFODNN7EXAMPLE",
            "github_pat_1234567890abcdef", "glpat-1234567890abcdef",
            "npm_1234567890abcdef", "hf_1234567890abcdef",
            "AIzaSyD1234567890abcdefghijklmnop",
        )

        class TextSource:
            session_id = "text-session"

            async def session_items(self, cursor, limit):
                text = ("Owner prefers jasmine tea. " + " ".join(secrets) +
                        "\nAuthentication: opaque-auth-value"
                        "\nBearer opaque-bearer-value"
                        "\nAPI token=opaque-api-token")
                return SessionItemPage(({
                    "id": "message-1", "type": "message", "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                    "unsupported": {"safe_looking": "must not cross"},
                },), "message-1", False)

        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            model = RecordingModel()
            await MemoryCurator(store, TextSource(), model).catch_up()

            model_input = json.dumps(model.pages)
            self.assertIn("Owner prefers jasmine tea.", model_input)
            self.assertNotIn("must not cross", model_input)
            for secret in (*secrets, "opaque-auth-value", "opaque-bearer-value",
                           "opaque-api-token"):
                self.assertNotIn(secret, model_input)
            self.assertGreaterEqual(model_input.count("[REDACTED]"), len(secrets) + 3)
            store.close()

    async def test_repeated_cursor_fails_without_looping(self):
        pages = [
            SessionItemPage(({"id": "one", "type": "message"},), "cursor-one", True),
            SessionItemPage(({"id": "two", "type": "message"},), "cursor-one", True),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            source, model = PageSource(pages), EmptyModel()
            with self.assertRaisesRegex(RuntimeError, "cursor did not advance"):
                await MemoryCurator(store, source, model, max_batches=10).catch_up()
            self.assertEqual(2, source.calls)
            self.assertEqual("cursor-one", store.curator_checkpoint(
                "openai_agents", source.session_id)["cursor"])
            store.close()

    async def test_has_more_empty_page_fails_without_looping(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            source = PageSource([SessionItemPage((), None, True)])
            with self.assertRaisesRegex(RuntimeError, "empty page"):
                await MemoryCurator(store, source, EmptyModel()).catch_up()
            self.assertEqual(1, source.calls)
            store.close()

    async def test_repeated_page_fails_even_if_cursor_changes(self):
        repeated = ({"id": "same", "type": "message"},)
        pages = [SessionItemPage(repeated, "one", True),
                 SessionItemPage(repeated, "two", True)]
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            source = PageSource(pages)
            with self.assertRaisesRegex(RuntimeError, "repeated a page"):
                await MemoryCurator(store, source, EmptyModel(), max_batches=10).catch_up()
            self.assertEqual(2, source.calls)
            store.close()

    async def test_distinct_filtered_pages_advance_checkpoint_without_curator_input(self):
        pages = [
            SessionItemPage(({"id": "reasoning-1", "type": "reasoning",
                              "content": "first private payload"},), "one", True),
            SessionItemPage(({"id": "reasoning-2", "type": "encrypted_reasoning",
                              "encrypted_content": "second private payload"},), "two", False),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            source, model = PageSource(pages), RecordingModel()
            await MemoryCurator(store, source, model, max_batches=10).catch_up()

            self.assertEqual(2, source.calls)
            self.assertEqual(0, model.calls)
            checkpoint = store.curator_checkpoint("openai_agents", source.session_id)
            self.assertEqual("two", checkpoint["cursor"])
            self.assertEqual("reasoning-2", checkpoint["last_item_id"])
            store.close()

    async def test_filtered_pages_are_followed_by_visible_curator_input(self):
        pages = [
            SessionItemPage(({"id": "reasoning-1", "type": "reasoning",
                              "content": "private payload"},), "one", True),
            SessionItemPage(({"id": "message-2", "type": "message", "role": "assistant",
                              "content": [{"type": "output_text", "text": "Useful summary"}]},),
                            "two", False),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            source, model = PageSource(pages), RecordingModel()
            await MemoryCurator(store, source, model, max_batches=10).catch_up()

            self.assertEqual(1, model.calls)
            self.assertIn("Useful summary", json.dumps(model.pages))
            self.assertNotIn("private payload", json.dumps(model.pages))
            self.assertEqual("two", store.curator_checkpoint(
                "openai_agents", source.session_id)["cursor"])
            store.close()

    async def test_final_catch_up_is_bounded_and_reports_incomplete(self):
        class AdvancingSource:
            session_id = "session-final"

            def __init__(self):
                self.calls = 0

            async def session_items(self, cursor, limit):
                self.calls += 1
                item_id = f"item-{self.calls}"
                return SessionItemPage(({"id": item_id, "type": "message"},),
                                       item_id, True)

        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            source, model = AdvancingSource(), EmptyModel()
            with self.assertRaisesRegex(RuntimeError, "consolidation incomplete"):
                await MemoryCurator(store, source, model, max_batches=2).catch_up(final=True)
            self.assertEqual(2, source.calls)
            self.assertEqual(2, model.calls)
            self.assertEqual("item-2", store.curator_checkpoint(
                "openai_agents", source.session_id)["cursor"])
            store.close()

    async def test_memory_updates_invalidate_without_destroying_revision_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            store.apply_curator_batch("openai_agents", "s", "one", "one", "batch-1", [{
                "memory_id": "m", "operation": "create", "kind": "fact", "content": "old"}])
            store.apply_curator_batch("openai_agents", "s", "two", "two", "batch-2", [{
                "memory_id": "m", "operation": "update", "content": "new"}])
            store.apply_curator_batch("openai_agents", "s", "three", "three", "batch-3", [{
                "memory_id": "m", "operation": "invalidate", "content": "incorrect"}])

            self.assertEqual([], store.search_memories())
            self.assertEqual(3, store.connection.execute(
                "SELECT count(*) FROM memory_revisions WHERE memory_id='m'").fetchone()[0])
            self.assertEqual("invalidated", store.memory("m")["status"])
            store.close()

    async def test_owner_guidance_is_revisioned_and_removable(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            guidance_id = store.set_owner_guidance("Always ask first")
            store.set_owner_guidance("Ask before purchases", guidance_id=guidance_id)
            self.assertEqual("Ask before purchases", store.active_owner_guidance()[0]["content"])
            self.assertTrue(store.remove_owner_guidance(guidance_id))
            self.assertEqual([], store.active_owner_guidance())
            self.assertEqual(3, store.connection.execute(
                "SELECT count(*) FROM owner_guidance_revisions WHERE guidance_id=?",
                (guidance_id,)).fetchone()[0])
            store.close()


if __name__ == "__main__":
    unittest.main()
