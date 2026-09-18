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


_CREDENTIAL_FIELD = re.compile(
    r"(?i)(?:^|[_\-\s])(?:authorization|api[_\-\s]?key|access[_\-\s]?token|"
    r"refresh[_\-\s]?token|auth[_\-\s]?token|token|password|passwd|secret|"
    r"client[_\-\s]?secret|api[_\-\s]?secret|secret[_\-\s]?key|"
    r"private[_\-\s]?key|cookie|credentials?)(?:$|[_\-\s])"
)
_AUTHORIZATION = re.compile(
    r"(?im)(\bauthorization\s*[:=]\s*)(?:[^\r\n]+)"
)
_BEARER = re.compile(r"(?i)(\bbearer\s+)[A-Za-z0-9._~+/=-]+")
_LABELED_SECRET = re.compile(
    r"(?i)(\b(?:api[-_ ]?key|access[-_ ]?token|refresh[-_ ]?token|auth[-_ ]?token|"
    r"token|password|passwd|client[-_ ]?secret|secret|cookie|credentials?)"
    r"\s*(?::|=|\bis\b|\bare\b)\s*)"
    r"(?:\"[^\"\r\n]*\"|'[^'\r\n]*'|[^\s,;\r\n}\]]+)"
)
_COMMON_TOKEN = re.compile(
    r"(?<![A-Za-z0-9])(?:sk-[A-Za-z0-9_-]{8,}|gh[pousr]_[A-Za-z0-9_]{8,}|"
    r"xox[a-z]-[A-Za-z0-9-]{8,}|eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\."
    r"[A-Za-z0-9_-]+)"
)
_REDACTED = "[REDACTED]"


def _redact_text(value: str) -> str:
    """Redact common complete credentials without retaining their values."""
    value = _AUTHORIZATION.sub(r"\1[REDACTED]", value)
    value = _BEARER.sub(r"\1[REDACTED]", value)
    value = _LABELED_SECRET.sub(r"\1[REDACTED]", value)
    return _COMMON_TOKEN.sub(_REDACTED, value)


def _is_credential_field(key: object) -> bool:
    return isinstance(key, str) and _CREDENTIAL_FIELD.search(f"_{key}_") is not None


def _redact_structured(value: Any) -> Any:
    """Return a redacted copy, preferring field-aware handling for structured data."""
    if isinstance(value, dict):
        return {
            key: _REDACTED if _is_credential_field(key) else _redact_structured(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_structured(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_structured(item) for item in value)
    if isinstance(value, str):
        return _redact_text(value)
    return value


def _canonical_item(item: dict[str, Any]) -> str:
    return json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _source_evidence(session_id: str, item: dict[str, Any]) -> dict[str, Any]:
    canonical = _canonical_item(item)
    return {
        "session_id": session_id,
        "item_id": item.get("id"),
        "source_type": item.get("type") or "session_item",
        "timestamp": item.get("created_at"),
        "excerpt": canonical[:500],
        "content_hash": hashlib.sha256(canonical.encode()).hexdigest(),
    }


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
            content.append({"type": part["type"], "text": _redact_text(text)})
    if content:
        safe["content"] = content
    for key in ("name", "call_id", "arguments", "output", "error"):
        value = item.get(key)
        if isinstance(value, (str, int, float, bool, dict, list)):
            redacted = _redact_structured(value)
            encoded = (json.dumps(redacted, ensure_ascii=False)
                       if not isinstance(redacted, str) else redacted)
            safe[key] = encoded[:4000]
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
        document = {"session_id": session_id,
                    "existing_memories": _redact_structured(list(existing)),
                    "new_session_items": _redact_structured(list(items))}
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
            saved_handover = checkpoint.get("handover_draft") if checkpoint else None
            handover = _redact_text(saved_handover) if isinstance(saved_handover, str) else None
            pages = 0
            seen_cursors = {cursor}
            seen_pages: set[str] = set()
            more_pages = False
            while pages < self.max_batches:
                page = await self.source.session_items(cursor, self.batch_size)
                more_pages = page.has_more
                safe_items = tuple(item for raw in page.items
                                   if (item := _safe_item(raw)) is not None)
                page_fingerprint = hashlib.sha256(
                    json.dumps(safe_items, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":")).encode()
                ).hexdigest()
                if not page.items:
                    if page.has_more:
                        raise RuntimeError("Curator pagination stalled on an empty page")
                    break
                last_item_id = page.items[-1].get("id")
                next_cursor = page.cursor or last_item_id
                if next_cursor is None or next_cursor == cursor or next_cursor in seen_cursors:
                    raise RuntimeError("Curator pagination cursor did not advance")
                if page_fingerprint in seen_pages:
                    raise RuntimeError("Curator pagination repeated a page")
                seen_cursors.add(next_cursor)
                seen_pages.add(page_fingerprint)
                pages += 1
                key_material = f"{session_id}:{cursor or ''}:{next_cursor or ''}"
                operation_key = hashlib.sha256(key_material.encode()).hexdigest()
                job_id = hashlib.sha256(f"job:{session_id}:{cursor or ''}".encode()).hexdigest()
                if not self.store.claim_curator_job(job_id, session_id, cursor):
                    cursor = next_cursor
                    if not page.has_more:
                        break
                    continue
                try:
                    existing = _redact_structured(self.store.search_memories(limit=20))
                    decision = await self.model.curate(session_id, safe_items, existing)
                    mutations = self._validate_mutations(
                        decision.get("mutations", []), session_id, safe_items)
                    proposed_handover = decision.get("handover")
                    if isinstance(proposed_handover, str) and proposed_handover.strip():
                        handover = _redact_text(proposed_handover.strip())[:8000]
                    self.store.apply_curator_batch(
                        "openai_agents", session_id, next_cursor, last_item_id,
                        operation_key, mutations, handover)
                    self.store.finish_curator_job(job_id)
                except BaseException as exc:
                    self.store.fail_curator_job(job_id, type(exc).__name__)
                    raise
                cursor = next_cursor
                if not page.has_more:
                    break
            if final and more_pages:
                raise RuntimeError(
                    f"Final Curator consolidation incomplete after {self.max_batches} pages")
            return handover

    @staticmethod
    def _validate_mutations(value: object,
                            session_id: str,
                            items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            raise RuntimeError("Curator mutations must be an array")
        known_items = {item.get("id"): item for item in items
                       if isinstance(item.get("id"), str)}
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
            clean = {"operation": operation,
                     "content": _redact_text(str(content))[:12000]}
            for key in ("memory_id", "kind"):
                if isinstance(mutation.get(key), str):
                    clean[key] = _redact_text(mutation[key])
            if isinstance(mutation.get("rationale"), str):
                clean["rationale"] = _redact_text(mutation["rationale"])[:4000]
            if isinstance(mutation.get("confidence"), (int, float)):
                clean["confidence"] = mutation["confidence"]
            provenance = []
            proposed_provenance = mutation.get("provenance", [])
            if not isinstance(proposed_provenance, list):
                proposed_provenance = []
            referenced_ids: set[str] = set()
            for evidence in proposed_provenance:
                if not isinstance(evidence, dict):
                    continue
                item_id = evidence.get("item_id")
                if item_id not in known_items or item_id in referenced_ids:
                    continue
                referenced_ids.add(item_id)
                provenance.append(_source_evidence(session_id, known_items[item_id]))
            clean["provenance"] = provenance
            accepted.append(clean)
        return accepted
