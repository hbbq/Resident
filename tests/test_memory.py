from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from resident.context import ContextBuilder
from resident.domain import Identity, WakeEvent
from resident.memory import MemoryCurator, OpenAICuratorModel, SessionItemPage
from resident.observability import timeline_reporter
from resident.store import (MAX_ACTIVE_OWNER_GUIDANCE_BYTES,
                            MAX_ACTIVE_OWNER_GUIDANCE_COUNT,
                            MAX_OWNER_GUIDANCE_ENTRY_BYTES, Store, utc_now)
from resident.tools import ToolRegistry


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

    async def curate(self, session_id, items, existing, current_handover):
        self.items.extend(items)
        return {"mutations": [{
            "operation": "create", "kind": "preference", "content": "Owner prefers tea",
            "confidence": .9, "provenance": [{"item_id": "item-1", "source_type": "message",
                                                "excerpt": "token=secret; prefers tea"}],
        }], "handover": {"operation": "replace", "content": "Continue discussing tea."}}


class EmptyModel:
    def __init__(self):
        self.calls = 0

    async def curate(self, session_id, items, existing, current_handover):
        self.calls += 1
        return {"mutations": []}


class RecordingModel(EmptyModel):
    def __init__(self):
        super().__init__()
        self.pages = []

    async def curate(self, session_id, items, existing, current_handover):
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
    def test_openai_curator_json_mode_mentions_json_in_input(self):
        document = {"session_id": "session-1", "new_session_items": []}
        response = io.BytesIO(json.dumps({
            "output_text": json.dumps({"mutations": [], "handover": {"operation": "keep"}}),
        }).encode())

        with patch("resident.memory.urllib.request.urlopen", return_value=response) as urlopen:
            result = OpenAICuratorModel("key", "model")._post(document)

        request = urlopen.call_args.args[0]
        body = json.loads(request.data)
        self.assertIn("json", body["input"].lower())
        self.assertEqual(document, json.loads(body["input"].split("\n\n", 1)[1]))
        self.assertEqual([], result["mutations"])

    async def test_curator_batch_timeline_finishes_successfully(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            events = []
            token = timeline_reporter.set(events.append)
            try:
                await MemoryCurator(store, FakeSource(), FakeModel()).catch_up()
            finally:
                timeline_reporter.reset(token)

            batch_events = [event for event in events
                            if event["operation"] == "curator.batch"]
            self.assertEqual(["started", "finished"], [
                event["moment"] for event in batch_events])
            self.assertEqual("ok", batch_events[1]["outcome"])
            self.assertEqual(
                {"phase": "incremental", "round": 1},
                {key: batch_events[1][key] for key in ("phase", "round")})
            self.assertGreaterEqual(batch_events[1]["duration_seconds"], 0.0)
            store.close()

    async def test_curator_batch_timeline_finishes_when_model_raises(self):
        failure = RuntimeError("prompt=curator-secret")

        class FailingModel:
            async def curate(self, session_id, items, existing, current_handover):
                raise failure

        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            events = []
            token = timeline_reporter.set(events.append)
            try:
                with self.assertRaises(RuntimeError) as raised:
                    await MemoryCurator(store, FakeSource(), FailingModel()).catch_up()
            finally:
                timeline_reporter.reset(token)

            self.assertIs(failure, raised.exception)
            batch_events = [event for event in events
                            if event["operation"] == "curator.batch"]
            self.assertEqual(["started", "finished"], [
                event["moment"] for event in batch_events])
            self.assertEqual("error", batch_events[1]["outcome"])
            self.assertEqual(
                {"phase": "incremental", "round": 1},
                {key: batch_events[1][key] for key in ("phase", "round")})
            self.assertGreaterEqual(batch_events[1]["duration_seconds"], 0.0)
            self.assertNotIn("curator-secret", json.dumps(batch_events))
            store.close()

    async def test_resident_memory_tools_expose_only_active_knowledge(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            for memory_id, content in (("active", "current fact"),
                                       ("superseded", "retired fact"),
                                       ("invalidated", "incorrect fact")):
                store.apply_curator_batch(
                    "openai_agents", "s", memory_id, memory_id, f"create-{memory_id}", [{
                        "memory_id": memory_id, "operation": "create", "kind": "fact",
                        "content": content, "provenance": [{"item_id": memory_id}],
                    }])
            store.apply_curator_batch(
                "openai_agents", "s", "supersede", "supersede", "supersede", [{
                    "memory_id": "superseded", "operation": "supersede",
                    "content": "retired fact", "provenance": [{"item_id": "supersede"}],
                }])
            store.apply_curator_batch(
                "openai_agents", "s", "invalidate", "invalidate", "invalidate", [{
                    "memory_id": "invalidated", "operation": "invalidate",
                    "content": "incorrect fact", "provenance": [{"item_id": "invalidate"}],
                }])
            registry = ToolRegistry(store, [], lambda _: {}, lambda *_: None)

            active = await registry.execute("get_long_term_memory", {"id": "active"})
            superseded = await registry.execute(
                "get_long_term_memory", {"id": "superseded"})
            invalidated = await registry.execute(
                "get_long_term_memory", {"id": "invalidated"})
            unknown = await registry.execute("get_long_term_memory", {"id": "unknown"})
            search = await registry.execute("search_long_term_memory", {})

            self.assertEqual("current fact", active.output["memory"]["content"])
            unavailable = {"ok": True, "memory": None}
            self.assertEqual(unavailable, superseded.output)
            self.assertEqual(unavailable, invalidated.output)
            self.assertEqual(unavailable, unknown.output)
            self.assertEqual(
                ["active"], [memory["id"] for memory in search.output["memories"]])
            self.assertEqual("retired fact", store.memory("superseded")["content"])
            self.assertEqual("superseded", store.memory("superseded")["status"])
            self.assertEqual("incorrect fact", store.memory("invalidated")["content"])
            self.assertEqual("invalidated", store.memory("invalidated")["status"])
            store.close()

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

            async def curate(self, session_id, items, existing, current_handover):
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
                }], "handover": {
                    "operation": "replace", "content": f"Bearer {raw_secrets[4]}"}}

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
                        "\nUseful link: https://example.com/gardening?topic=tea"
                        "\nDatabase: postgresql://alice:correct-horse@db.example/app"
                        "\nAuthentication: opaque-auth-value"
                        "\nBearer opaque-bearer-value"
                        "\nAPI token=opaque-api-token"
                        "\nAWS_SECRET_ACCESS_KEY=cloud-secret-value"
                        "\nConnection: Server=db.internal;User Id=resident;Password=connection-secret"
                        "\nConfig: {\"client_secret\":\"json-secret\",\"region\":\"north\"}"
                        "\n-----BEGIN PRIVATE KEY-----\nprivate-key-material"
                        "\n-----END PRIVATE KEY-----")
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
            self.assertIn("https://example.com/gardening?topic=tea", model_input)
            self.assertNotIn("must not cross", model_input)
            for secret in (*secrets, "opaque-auth-value", "opaque-bearer-value",
                           "opaque-api-token", "correct-horse", "alice", "db.example",
                           "cloud-secret-value", "connection-secret", "db.internal",
                           "json-secret", "private-key-material"):
                self.assertNotIn(secret, model_input)
            self.assertIn("[REDACTED CREDENTIAL URL]", model_input)
            self.assertIn("[REDACTED CREDENTIAL STRUCTURE]", model_input)
            self.assertIn("[REDACTED PRIVATE KEY]", model_input)
            self.assertIn("region", model_input)
            self.assertGreaterEqual(model_input.count("[REDACTED]"), len(secrets) + 4)
            store.close()

    async def test_curator_output_credentials_are_removed_before_durable_persistence(self):
        private_key = ("-----BEGIN PRIVATE KEY-----\ncurator-private-material\n"
                       "-----END PRIVATE KEY-----")
        secrets = ("db-password", "bearer-output", "cloud-output", "api-output",
                   "curator-private-material")

        class OutputSource:
            session_id = "output-session"

            async def session_items(self, cursor, limit):
                return SessionItemPage(({
                    "id": "observed-1", "type": "message", "role": "user",
                    "content": [{"type": "input_text", "text": "The greenhouse needs water."}],
                },), "observed-1", False)

        class CredentialOutputModel:
            async def curate(self, session_id, items, existing, current_handover):
                return {"mutations": [{
                    "operation": "create", "kind": "fact",
                    "content": ("Water plants; postgresql://resident:db-password@db.local/home; "
                                "Authorization: Bearer bearer-output; "
                                "AWS_SECRET_ACCESS_KEY=cloud-output; "
                                "api_key=api-output; " + private_key),
                    "rationale": "Authorization: Bearer bearer-output",
                    "provenance": [{"item_id": "observed-1"}],
                }], "handover": {"operation": "replace", "content":
                                  "Continue safely. " + private_key + " api_key=api-output"}}

        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            handover = await MemoryCurator(
                store, OutputSource(), CredentialOutputModel()).catch_up()
            persisted = "\n".join(store.connection.iterdump()) + str(handover)
            for secret in secrets:
                self.assertNotIn(secret, persisted)
            self.assertIn("The greenhouse needs water.",
                          store.memory(store.search_memories()[0]["id"])["provenance"][0]["excerpt"])
            store.close()

    async def test_only_fully_grounded_mutations_are_persisted_and_checkpointed(self):
        class GroundingSource:
            session_id = "grounding-session"

            async def session_items(self, cursor, limit):
                if cursor is not None:
                    return SessionItemPage((), cursor, False)
                return SessionItemPage(({
                    "id": "real-1", "type": "message", "role": "user",
                    "content": [{"type": "input_text", "text": "I grow basil."}],
                },), "real-1", False)

        class GroundingModel:
            async def curate(self, session_id, items, existing, current_handover):
                base = {"operation": "create", "kind": "fact"}
                return {"mutations": [
                    {**base, "memory_id": "missing", "content": "missing"},
                    {**base, "memory_id": "nonexistent", "content": "nonexistent",
                     "provenance": [{"item_id": "invented"}]},
                    {**base, "memory_id": "mixed", "content": "mixed",
                     "provenance": [{"item_id": "real-1"}, {"item_id": "invented"}]},
                    {**base, "memory_id": "grounded", "content": "Owner grows basil",
                     "provenance": [{"item_id": "real-1"}]},
                ]}

        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            curator = MemoryCurator(store, GroundingSource(), GroundingModel())
            await curator.catch_up()
            await curator.catch_up()

            self.assertEqual(["grounded"], [memory["id"] for memory in store.search_memories()])
            self.assertEqual("real-1", store.memory("grounded")["provenance"][0][
                "source_item_id"])
            self.assertEqual("real-1", store.curator_checkpoint(
                "openai_agents", "grounding-session")["cursor"])
            self.assertEqual(1, store.connection.execute(
                "SELECT count(*) FROM curator_operations").fetchone()[0])
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

    async def test_final_catch_up_retry_resumes_checkpoint_without_duplicate_mutation(self):
        class TwoPageSource:
            session_id = "session-retry"

            def __init__(self):
                self.cursors = []

            async def session_items(self, cursor, limit):
                self.cursors.append(cursor)
                if cursor is None:
                    return SessionItemPage(({
                        "id": "one", "type": "message", "role": "user",
                        "content": [{"type": "input_text", "text": "remember this"}],
                    },), "one", True)
                return SessionItemPage(({
                    "id": "two", "type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": "acknowledged"}],
                },), "two", False)

        class OneMutationModel:
            async def curate(self, session_id, items, existing, current_handover):
                mutations = []
                if items[0]["id"] == "one":
                    mutations.append({
                        "memory_id": "durable-one", "operation": "create",
                        "kind": "fact", "content": "remember this",
                        "provenance": [{"item_id": "one"}],
                    })
                return {"mutations": mutations,
                        "handover": {"operation": "keep"}}

        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            source = TwoPageSource()
            curator = MemoryCurator(
                store, source, OneMutationModel(), max_batches=1)

            with self.assertRaisesRegex(RuntimeError, "consolidation incomplete"):
                await curator.catch_up(final=True)
            self.assertEqual("one", store.curator_checkpoint(
                "openai_agents", source.session_id)["cursor"])
            self.assertEqual(1, store.connection.execute("""
                SELECT COUNT(*) FROM memory_revisions WHERE memory_id='durable-one'
            """).fetchone()[0])

            await curator.catch_up(final=True)
            self.assertEqual([None, "one"], source.cursors)
            self.assertEqual("two", store.curator_checkpoint(
                "openai_agents", source.session_id)["cursor"])
            self.assertEqual(1, store.connection.execute("""
                SELECT COUNT(*) FROM memory_revisions WHERE memory_id='durable-one'
            """).fetchone()[0])
            store.close()

    async def test_handover_operations_track_the_curators_current_view(self):
        class AdvancingSource:
            session_id = "handover-session"

            async def session_items(self, cursor, limit):
                index = 1 if cursor is None else int(cursor.rsplit("-", 1)[1]) + 1
                if index > 6:
                    return SessionItemPage((), cursor, False)
                item_id = f"item-{index}"
                return SessionItemPage(({
                    "id": item_id, "type": "message", "role": "user",
                    "content": [{"type": "input_text", "text": f"event {index}"}],
                },), item_id, False)

        class HandoverModel:
            def __init__(self):
                self.current = []
                self.operations = [
                    {"operation": "keep"},
                    {"operation": "replace", "content": "first draft"},
                    {"operation": "keep"},
                    {"operation": "replace", "content": "authoritative second draft"},
                    {"operation": "clear"},
                    {"operation": "keep"},
                ]

            async def curate(self, session_id, items, existing, current_handover):
                self.current.append(current_handover)
                return {"mutations": [], "handover": self.operations.pop(0)}

        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            model = HandoverModel()
            curator = MemoryCurator(store, AdvancingSource(), model)

            observed = [await curator.catch_up() for _ in range(6)]

            self.assertEqual([
                None, "first draft", "first draft", "authoritative second draft",
                None, None,
            ], observed)
            self.assertEqual([
                None, None, "first draft", "first draft",
                "authoritative second draft", None,
            ], model.current)
            self.assertIsNone(store.curator_checkpoint(
                "openai_agents", "handover-session")["handover_draft"])
            store.close()

    async def test_cleared_handover_survives_retry_without_touching_finalized_history(self):
        class ClearSource:
            session_id = "session-old"

            async def session_items(self, cursor, limit):
                if cursor == "seed":
                    return SessionItemPage(({
                        "id": "clear-item", "type": "message", "role": "user",
                        "content": [{"type": "input_text", "text": "Work is resolved."}],
                    },), "clear-item", False)
                return SessionItemPage((), cursor, False)

        class ClearModel:
            def __init__(self):
                self.current = []

            async def curate(self, session_id, items, existing, current_handover):
                self.current.append(current_handover)
                return {"mutations": [], "handover": {"operation": "clear"}}

        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            store.apply_curator_batch(
                "openai_agents", "session-old", "seed", "seed", "seed-operation", [],
                "replace", "stale mutable draft")
            finalized_id = store.create_handover(
                "session-old", "finalized historical bridge", "9999-12-31T23:59:59+00:00")
            store.consume_handover(finalized_id, "session-new")
            model = ClearModel()
            curator = MemoryCurator(store, ClearSource(), model)
            finish = store.finish_curator_job
            failed_once = False

            def crash_after_batch(job_id):
                nonlocal failed_once
                if not failed_once:
                    failed_once = True
                    raise RuntimeError("simulated checkpoint-adjacent crash")
                finish(job_id)

            store.finish_curator_job = crash_after_batch
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                await curator.catch_up()
            store.finish_curator_job = finish

            self.assertIsNone(await curator.catch_up())
            self.assertEqual(["stale mutable draft"], model.current)
            self.assertIsNone(store.curator_checkpoint(
                "openai_agents", "session-old")["handover_draft"])
            finalized = store.connection.execute(
                "SELECT content,new_session_id,consumed_at FROM session_handovers WHERE id=?",
                (finalized_id,)).fetchone()
            self.assertEqual("finalized historical bridge", finalized["content"])
            self.assertEqual("session-new", finalized["new_session_id"])
            self.assertIsNotNone(finalized["consumed_at"])
            store.close()

    async def test_duplicate_create_is_rejected_for_every_existing_lifecycle_state(self):
        for status_operation in (None, "supersede", "invalidate"):
            with self.subTest(status_operation=status_operation), tempfile.TemporaryDirectory() as temporary:
                store = Store(Path(temporary) / "resident.sqlite3")
                store.apply_curator_batch("openai_agents", "s", "one", "one", "create", [{
                    "memory_id": "existing", "operation": "create", "kind": "fact",
                    "content": "original", "provenance": [{"item_id": "one"}],
                }])
                if status_operation is not None:
                    store.apply_curator_batch(
                        "openai_agents", "s", "retire", "retire", "retire", [{
                            "memory_id": "existing", "operation": status_operation,
                            "content": "retired", "provenance": [{"item_id": "retire"}],
                        }])
                before = store.memory("existing")
                revision_count = store.connection.execute(
                    "SELECT count(*) FROM memory_revisions WHERE memory_id='existing'"
                ).fetchone()[0]

                with self.assertRaisesRegex(ValueError, "create targets an existing record"):
                    store.apply_curator_batch(
                        "openai_agents", "s", "duplicate", "duplicate", "duplicate", [{
                            "memory_id": "existing", "operation": "create", "kind": "fact",
                            "content": "must not reactivate",
                            "provenance": [{"item_id": "duplicate"}],
                        }])

                after = store.memory("existing")
                self.assertEqual(before["status"], after["status"])
                self.assertEqual(before["content"], after["content"])
                self.assertEqual(revision_count, store.connection.execute(
                    "SELECT count(*) FROM memory_revisions WHERE memory_id='existing'"
                ).fetchone()[0])
                self.assertEqual(0, store.connection.execute(
                    "SELECT count(*) FROM curator_operations WHERE operation_key='duplicate'"
                ).fetchone()[0])
                store.close()

    async def test_duplicate_create_rolls_back_the_entire_curator_batch(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            store.apply_curator_batch("openai_agents", "s", "seed", "seed", "seed", [{
                "memory_id": "existing", "operation": "create", "kind": "fact",
                "content": "original", "provenance": [{"item_id": "seed"}],
            }])

            with self.assertRaisesRegex(ValueError, "create targets an existing record"):
                store.apply_curator_batch(
                    "openai_agents", "s", "bad", "bad", "bad-batch", [{
                        "memory_id": "new-first", "operation": "create", "kind": "fact",
                        "content": "must roll back", "provenance": [{"item_id": "bad"}],
                    }, {
                        "memory_id": "existing", "operation": "create", "kind": "fact",
                        "content": "duplicate", "provenance": [{"item_id": "bad"}],
                    }], "replace", "must also roll back")

            self.assertIsNone(store.memory("new-first"))
            self.assertEqual("seed", store.curator_checkpoint(
                "openai_agents", "s")["cursor"])
            self.assertIsNone(store.curator_checkpoint(
                "openai_agents", "s")["handover_draft"])
            self.assertEqual(0, store.connection.execute(
                "SELECT count(*) FROM curator_operations WHERE operation_key='bad-batch'"
            ).fetchone()[0])
            store.close()

    async def test_memory_updates_invalidate_without_destroying_revision_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            store.apply_curator_batch("openai_agents", "s", "one", "one", "batch-1", [{
                "memory_id": "m", "operation": "create", "kind": "fact", "content": "old",
                "provenance": [{"item_id": "one"}]}])
            store.apply_curator_batch("openai_agents", "s", "two", "two", "batch-2", [{
                "memory_id": "m", "operation": "update", "content": "new",
                "provenance": [{"item_id": "two"}]}])
            store.apply_curator_batch("openai_agents", "s", "three", "three", "batch-3", [{
                "memory_id": "m", "operation": "invalidate", "content": "incorrect",
                "provenance": [{"item_id": "three"}]}])

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

    async def test_owner_guidance_enforces_entry_and_active_set_bounds(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            with self.assertRaisesRegex(ValueError, "exceeds"):
                store.set_owner_guidance("x" * (MAX_OWNER_GUIDANCE_ENTRY_BYTES + 1))
            self.assertEqual(0, store.connection.execute(
                "SELECT count(*) FROM owner_guidance_revisions").fetchone()[0])

            ids = [store.set_owner_guidance(f"guidance-{index}")
                   for index in range(MAX_ACTIVE_OWNER_GUIDANCE_COUNT)]
            with self.assertRaisesRegex(ValueError, "replace or remove"):
                store.set_owner_guidance("one too many")

            store.set_owner_guidance("replacement", guidance_id=ids[0])
            self.assertEqual(2, store.connection.execute(
                "SELECT count(*) FROM owner_guidance_revisions WHERE guidance_id=?",
                (ids[0],)).fetchone()[0])
            self.assertTrue(store.remove_owner_guidance(ids[1]))
            replacement_id = store.set_owner_guidance("replacement after removal")
            self.assertEqual(MAX_ACTIVE_OWNER_GUIDANCE_COUNT,
                             len(store.active_owner_guidance()))
            self.assertEqual(2, store.connection.execute(
                "SELECT count(*) FROM owner_guidance_revisions WHERE guidance_id=?",
                (ids[1],)).fetchone()[0])
            self.assertIn(replacement_id,
                          {entry["id"] for entry in store.active_owner_guidance()})

            context = json.loads(ContextBuilder(store).build(
                Identity("resident", "Resident"), Identity("owner", "Owner"),
                WakeEvent("wake", "test", "bounded", utc_now(), {}), []))
            projection = context["standing_owner_guidance"]
            self.assertEqual(MAX_ACTIVE_OWNER_GUIDANCE_COUNT, len(projection))
            self.assertLessEqual(len(json.dumps(
                projection, ensure_ascii=False, sort_keys=True,
                separators=(",", ":")).encode("utf-8")),
                MAX_ACTIVE_OWNER_GUIDANCE_BYTES)
            store.close()

    async def test_owner_guidance_enforces_total_serialized_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "resident.sqlite3")
            content = "å" * 1800
            created = 0
            with self.assertRaisesRegex(ValueError, "serialized UTF-8 bytes"):
                for _ in range(MAX_ACTIVE_OWNER_GUIDANCE_COUNT):
                    store.set_owner_guidance(content)
                    created += 1
            self.assertGreater(created, 1)
            self.assertLess(created, MAX_ACTIVE_OWNER_GUIDANCE_COUNT)
            projection = store.active_owner_guidance()
            self.assertLessEqual(len(json.dumps(
                projection, ensure_ascii=False, sort_keys=True,
                separators=(",", ":")).encode("utf-8")),
                MAX_ACTIVE_OWNER_GUIDANCE_BYTES)
            store.close()


if __name__ == "__main__":
    unittest.main()
