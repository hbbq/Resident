from __future__ import annotations

import asyncio
import hashlib
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from .store import Store


_SECRET = re.compile(
    r"(?i)([\"']?(?:authorization|api[-_ ]?key|password|token|secret)[\"']?"
    r"\s*[:=]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s,;}\]]+)"
)


@dataclass(frozen=True)
class SessionItemPage:
    items: tuple[dict[str, Any], ...]
    cursor: str | None
    has_more: bool = False


class SessionItemSource(Protocol):
    @property
    def session_id(self) -> str | None: ...

    async def session_items(self, cursor: str | None, limit: int) -> SessionItemPage: ...


class CuratorModel(Protocol):
    async def curate(self, session_id: str, items: Sequence[dict[str, Any]],
                     existing: Sequence[dict[str, Any]]) -> dict[str, Any]: ...


def _safe_item(item: dict[str, Any]) -> dict[str, Any] | None:
    """Allow model-visible material while excluding reasoning and attachment payloads."""
    if item.get("type") in {"reasoning", "encrypted_reasoning"}:
        return None
    safe = {key: item.get(key) for key in ("id", "turn_id", "type", "role", "created_at")
            if item.get(key) is not None}
    content: list[dict[str, str]] = []
    for part in item.get("content") or []:
        if not isinstance(part, dict) or part.get("type") not in {"input_text", "output_text"}:
            continue
        text = part.get("text")
        if isinstance(text, str):
            content.append({"type": part["type"], "text": _SECRET.sub(r"\1[REDACTED]", text)})
    if content:
        safe["content"] = content
    for key in ("name", "call_id", "arguments", "output", "error"):
        value = item.get(key)
        if isinstance(value, (str, int, float, bool, dict, list)):
            encoded = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
            safe[key] = _SECRET.sub(r"\1[REDACTED]", encoded)[:4000]
    return safe if len(safe) > 1 else None


class OpenAICuratorModel:
    """Independent, stateless curator using structured JSON output."""

    def __init__(self, api_key: str, model: str,
                 base_url: str = "https://api.openai.com/v1", timeout_seconds: float = 60):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for the Curator")
        self.api_key, self.model = api_key, model
        self.base_url, self.timeout_seconds = base_url.rstrip("/"), timeout_seconds

    async def curate(self, session_id: str, items: Sequence[dict[str, Any]],
                     existing: Sequence[dict[str, Any]]) -> dict[str, Any]:
        document = {"session_id": session_id, "existing_memories": list(existing),
                    "new_session_items": list(items)}
        return await asyncio.to_thread(self._post, document)

    def _post(self, document: dict[str, Any]) -> dict[str, Any]:
        body = {
            "model": self.model,
            "store": False,
            "instructions": (
                "Curate only durable, useful experience into memory. Temporary work belongs in handover, "
                "not memory. Never retain credentials, authentication material, private reasoning, or "
                "attachment payloads. Return JSON with mutations and optional handover. Each mutation has "
                "operation (create/update/supersede/invalidate), optional memory_id, kind, content, rationale, "
                "confidence, and provenance entries referencing supplied item_id values."
            ),
            "input": json.dumps(document, ensure_ascii=False),
            "text": {"format": {"type": "json_object"}},
        }
        request = urllib.request.Request(
            f"{self.base_url}/responses", data=json.dumps(body).encode(), method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:2000]
            raise RuntimeError(f"OpenAI Curator returned HTTP {exc.code}: {detail}") from exc
        text = raw.get("output_text")
        if not text:
            for item in raw.get("output", []):
                for part in item.get("content", []):
                    if part.get("type") == "output_text":
                        text = part.get("text")
                        break
        try:
            result = json.loads(text or "{}")
        except json.JSONDecodeError as exc:
            raise RuntimeError("Curator returned invalid JSON") from exc
        if not isinstance(result, dict):
            raise RuntimeError("Curator output must be an object")
        return result


class MemoryCurator:
    def __init__(self, store: Store, source: SessionItemSource, model: CuratorModel, *,
                 batch_size: int = 50, max_batches: int = 4):
        self.store, self.source, self.model = store, source, model
        self.batch_size, self.max_batches = max(1, min(batch_size, 100)), max(1, max_batches)
        self._lock = asyncio.Lock()

    async def catch_up(self, *, final: bool = False) -> str | None:
        """Consolidate bounded pages; checkpoints advance only with durable decisions."""
        async with self._lock:
            session_id = self.source.session_id
            if not session_id:
                return None
            checkpoint = self.store.curator_checkpoint("openai_agents", session_id)
            cursor = checkpoint["cursor"] if checkpoint else None
            handover: str | None = checkpoint.get("handover_draft") if checkpoint else None
            batches = 0
            while batches < self.max_batches or final:
                page = await self.source.session_items(cursor, self.batch_size)
                safe_items = tuple(item for raw in page.items
                                   if (item := _safe_item(raw)) is not None)
                if not page.items:
                    break
                last_item_id = page.items[-1].get("id")
                next_cursor = page.cursor or last_item_id
                key_material = f"{session_id}:{cursor or ''}:{next_cursor or ''}"
                operation_key = hashlib.sha256(key_material.encode()).hexdigest()
                job_id = hashlib.sha256(f"job:{session_id}:{cursor or ''}".encode()).hexdigest()
                if not self.store.claim_curator_job(job_id, session_id, cursor):
                    cursor = next_cursor
                    if not page.has_more:
                        break
                    continue
                try:
                    existing = self.store.search_memories(limit=20)
                    decision = await self.model.curate(session_id, safe_items, existing)
                    mutations = self._validate_mutations(decision.get("mutations", []), safe_items)
                    proposed_handover = decision.get("handover")
                    if isinstance(proposed_handover, str) and proposed_handover.strip():
                        handover = proposed_handover.strip()[:8000]
                    self.store.apply_curator_batch(
                        "openai_agents", session_id, next_cursor, last_item_id,
                        operation_key, mutations, handover)
                    self.store.finish_curator_job(job_id)
                except BaseException as exc:
                    self.store.fail_curator_job(job_id, type(exc).__name__)
                    raise
                cursor = next_cursor
                batches += 1
                if not page.has_more:
                    break
            return handover

    @staticmethod
    def _validate_mutations(value: object,
                            items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            raise RuntimeError("Curator mutations must be an array")
        known_ids = {item.get("id") for item in items}
        accepted: list[dict[str, Any]] = []
        for mutation in value[:50]:
            if not isinstance(mutation, dict):
                raise RuntimeError("Curator mutation must be an object")
            operation = mutation.get("operation")
            content = mutation.get("content", "")
            if operation not in {"create", "update", "supersede", "invalidate"}:
                raise RuntimeError("Curator returned an unsupported operation")
            if operation != "invalidate" and (not isinstance(content, str) or not content.strip()):
                raise RuntimeError("Curator memory content must be nonempty")
            clean = dict(mutation)
            clean["content"] = _SECRET.sub(r"\1[REDACTED]", str(content))[:12000]
            provenance = []
            for evidence in mutation.get("provenance", []):
                if not isinstance(evidence, dict) or evidence.get("item_id") not in known_ids:
                    continue
                evidence = dict(evidence)
                if isinstance(evidence.get("excerpt"), str):
                    evidence["excerpt"] = _SECRET.sub(
                        r"\1[REDACTED]", evidence["excerpt"][:500])
                provenance.append(evidence)
            clean["provenance"] = provenance
            accepted.append(clean)
        return accepted
