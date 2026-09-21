from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from .observability import emit_timeline
from .store import Store


class FinalCatchUpIncomplete(RuntimeError):
    """Final consolidation reached its operational page limit and must resume later."""


class SessionHistoryUnavailable(RuntimeError):
    """The old session's item history is definitively unavailable."""


_AUTHORIZATION = re.compile(
    r"(?im)(\b(?:authorization|authentication|proxy-authorization)\s*[:=]\s*)"
    r"(?:[^\r\n]+)"
)
_BEARER = re.compile(r"(?i)(\bbearer\s+)[A-Za-z0-9._~+/=-]+")
_PRIVATE_KEY_BLOCK = re.compile(
    r"-----BEGIN(?: [A-Z0-9]+)? PRIVATE KEY-----.*?"
    r"-----END(?: [A-Z0-9]+)? PRIVATE KEY-----",
    re.DOTALL,
)
_URL = re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.-]*://[^\s<>\"']+")
_LABELED_SECRET = re.compile(
    r"(?i)((?<![A-Za-z0-9])(?:aws[-_ ]?|azure[-_ ]?|gcp[-_ ]?|google[-_ ]?)?"
    r"(?:api[-_ ]?key|access[-_ ]?key(?:[-_ ]?id)?|secret[-_ ]?access[-_ ]?key|"
    r"access[-_ ]?token|refresh[-_ ]?token|auth[-_ ]?token|token|password|passwd|pwd|"
    r"client[-_ ]?secret|private[-_ ]?key|secret|cookie|credentials?|account[-_ ]?key|"
    r"shared[-_ ]?access[-_ ]?signature|sas[-_ ]?token|connection[-_ ]?string)"
    r"[\"']?\s*(?::|=|\bis\b|\bare\b)\s*)"
    r"(?:\"[^\"\r\n]*\"|'[^'\r\n]*'|[^\s,;\r\n}\]]+)"
)
_COMMON_TOKEN = re.compile(
    r"(?<![A-Za-z0-9])(?:sk-[A-Za-z0-9_-]{8,}|(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{8,}|"
    r"(?:AKIA|ASIA)[A-Z0-9]{16}|gh[pousr]_[A-Za-z0-9_]{8,}|"
    r"github_pat_[A-Za-z0-9_]{12,}|glpat-[A-Za-z0-9_-]{8,}|"
    r"xox[a-z]-[A-Za-z0-9-]{8,}|npm_[A-Za-z0-9]{8,}|ya29\.[A-Za-z0-9_-]{8,}|"
    r"pypi-[A-Za-z0-9_-]{12,}|hf_[A-Za-z0-9]{12,}|"
    r"AIza[A-Za-z0-9_-]{20,}|eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\."
    r"[A-Za-z0-9_-]+)"
)
_REDACTED = "[REDACTED]"
_CREDENTIAL_FIELD_NAMES = {
    "accountkey", "apikey", "authentication", "authorization", "clientsecret",
    "connectionstring", "cookie", "credential", "credentials", "password", "passwd",
    "privatekey", "pwd", "refreshtoken", "sas", "sastoken", "secret",
    "secretaccesskey", "sharedaccesssignature", "token",
}


def _redact_connection_structures(value: str) -> str:
    """Drop complete semicolon-delimited credential-bearing structures."""
    protected: list[str] = []
    for line in value.splitlines(keepends=True):
        ending = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
        body = line[:-len(ending)] if ending else line
        assignments = []
        for field in body.split(";"):
            if "=" not in field:
                continue
            key, _ = field.split("=", 1)
            # A display label may precede the first connection-string key.
            key = key.rsplit(":", 1)[-1]
            assignments.append("".join(character for character in key.lower()
                                       if character.isalnum()))
        if len(assignments) >= 2 and any(
                key in _CREDENTIAL_FIELD_NAMES or key.endswith("password")
                or key.endswith("secret") or key.endswith("token")
                or key.endswith("apikey") for key in assignments):
            protected.append("[REDACTED CREDENTIAL STRUCTURE]" + ending)
        else:
            protected.append(line)
    return "".join(protected)


def _redact_text(value: str) -> str:
    """Protect recognizable credential structures while retaining benign prose."""
    value = _PRIVATE_KEY_BLOCK.sub("[REDACTED PRIVATE KEY]", value)
    value = _redact_connection_structures(value)

    def redact_credential_url(match: re.Match[str]) -> str:
        candidate = match.group(0)
        try:
            parsed = urllib.parse.urlsplit(candidate)
        except ValueError:
            return "[REDACTED CREDENTIAL URL]" if "@" in candidate else candidate
        # A password-bearing userinfo component is itself a credential. Drop the
        # complete URL so neither the password nor a potentially sensitive user,
        # host, path, or query survives in excerpts or hashes.
        return "[REDACTED CREDENTIAL URL]" if parsed.password is not None else candidate

    value = _URL.sub(redact_credential_url, value)
    value = _AUTHORIZATION.sub(r"\1[REDACTED]", value)
    value = _BEARER.sub(r"\1[REDACTED]", value)
    value = _LABELED_SECRET.sub(r"\1[REDACTED]", value)
    return _COMMON_TOKEN.sub(_REDACTED, value)


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
                     existing: Sequence[dict[str, Any]],
                     current_handover: str | None) -> dict[str, Any]: ...


def _safe_item(item: dict[str, Any]) -> dict[str, Any] | None:
    """Project a supported source item onto the Curator's explicit input schema."""
    item_type = item.get("type")
    if item_type not in {"message", "function_call", "function_call_output"}:
        return None

    safe: dict[str, Any] = {"type": item_type}
    for key in ("id", "turn_id", "created_at"):
        value = item.get(key)
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            safe[key] = value

    if item_type == "message":
        role = item.get("role")
        if role in {"user", "assistant", "system", "developer"}:
            safe["role"] = role
    else:
        call_id = item.get("call_id")
        if isinstance(call_id, str):
            safe["call_id"] = call_id[:500]

    if item_type == "function_call":
        name = item.get("name")
        if isinstance(name, str):
            safe["name"] = _redact_text(name)[:500]
        # Tool arguments are intentionally outside the Curator boundary. Their
        # schemas vary by tool and may contain credentials in unknown fields.
        return safe

    if item_type == "function_call_output":
        # Result bodies and errors are similarly excluded. Assistant messages retain
        # the safe semantic account of what was learned from a tool invocation.
        success = item.get("success")
        if isinstance(success, bool):
            safe["success"] = success
        return safe

    content: list[dict[str, str]] = []
    for part in item.get("content") or []:
        if not isinstance(part, dict) or part.get("type") not in {"input_text", "output_text"}:
            continue
        text = part.get("text")
        if isinstance(text, str):
            content.append({"type": part["type"], "text": _redact_text(text)})
    if content:
        safe["content"] = content
    return safe


def _safe_existing_memory(memory: dict[str, Any]) -> dict[str, Any]:
    """Project durable memory onto the fields needed for consolidation."""
    safe: dict[str, Any] = {}
    for key in ("id", "updated_at"):
        value = memory.get(key)
        if isinstance(value, str):
            safe[key] = value
    for key in ("kind", "content"):
        value = memory.get(key)
        if isinstance(value, str):
            safe[key] = _redact_text(value)
    confidence = memory.get("confidence")
    if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
        safe["confidence"] = confidence
    return safe


def _source_page_identity(page: SessionItemPage) -> tuple[str, tuple[str, ...] | str]:
    """Identify a remote page from source metadata, never from filtered payloads."""
    item_ids = tuple(item.get("id") for item in page.items)
    if item_ids and all(isinstance(item_id, str) and item_id for item_id in item_ids):
        return "item_ids", item_ids
    if isinstance(page.cursor, str) and page.cursor:
        return "cursor", page.cursor
    raise RuntimeError("Curator page lacks stable source identity")


class OpenAICuratorModel:
    """Independent, stateless curator using structured JSON output."""

    def __init__(self, api_key: str, model: str,
                 base_url: str = "https://api.openai.com/v1", timeout_seconds: float = 60):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for the Curator")
        self.api_key, self.model = api_key, model
        self.base_url, self.timeout_seconds = base_url.rstrip("/"), timeout_seconds

    async def curate(self, session_id: str, items: Sequence[dict[str, Any]],
                     existing: Sequence[dict[str, Any]],
                     current_handover: str | None) -> dict[str, Any]:
        document = {"session_id": session_id,
                    "existing_memories": list(existing),
                    "current_handover_draft": current_handover,
                    "new_session_items": list(items)}
        return await asyncio.to_thread(self._post, document)

    def _post(self, document: dict[str, Any]) -> dict[str, Any]:
        body = {
            "model": self.model,
            "store": False,
            "instructions": (
                "Curate only durable, useful experience into memory. Temporary work belongs in handover, "
                "not memory. Never retain credentials, authentication material, private reasoning, or "
                "attachment payloads. Return JSON with mutations and a handover object whose operation is "
                "keep, replace, or clear. Replace also has nonempty content and makes it authoritative; keep "
                "retains the supplied current_handover_draft; clear removes it. Each mutation has "
                "operation (create/update/supersede/invalidate), optional memory_id, kind, content, rationale, "
                "confidence, and provenance entries referencing supplied item_id values."
            ),
            # JSON mode requires the input itself to mention JSON; instructions are
            # not considered when the Responses API validates this requirement.
            "input": "Return the requested result as JSON.\n\n"
                     + json.dumps(document, ensure_ascii=False),
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

    async def catch_up(self, *, final: bool = False,
                       through_turn_id: str | None = None,
                       session_id: str | None = None) -> str | None:
        """Consolidate bounded pages; checkpoints advance only with durable decisions."""
        async with self._lock:
            source_session_id = self.source.session_id
            if session_id is not None and source_session_id != session_id:
                raise SessionHistoryUnavailable("Requested Curator session is no longer bound")
            session_id = source_session_id
            if not session_id:
                return None
            checkpoint = self.store.curator_checkpoint("openai_agents", session_id)
            cursor = checkpoint["cursor"] if checkpoint else None
            saved_handover = checkpoint.get("handover_draft") if checkpoint else None
            handover = _redact_text(saved_handover) if isinstance(saved_handover, str) else None
            pages = 0
            seen_cursors = {cursor}
            seen_pages: set[tuple[str, tuple[str, ...] | str]] = set()
            more_pages = False
            reached_boundary = bool(
                through_turn_id is not None and checkpoint is not None
                and checkpoint.get("last_turn_id") == through_turn_id)
            while pages < self.max_batches:
                batch_round = pages + 1
                batch_started = time.monotonic()
                emit_timeline("curator.batch", "started", phase="final" if final else "incremental",
                              round=batch_round)
                batch_outcome = "error"
                try:
                    page = await self.source.session_items(cursor, self.batch_size)
                    page_items = page.items
                    if through_turn_id is not None:
                        boundary_indexes = [
                            index for index, item in enumerate(page_items)
                            if item.get("turn_id") == through_turn_id]
                        if reached_boundary and not boundary_indexes:
                            batch_outcome = "ok"
                            more_pages = False
                            break
                        if boundary_indexes:
                            reached_boundary = True
                            boundary_end = len(page_items)
                            for index in range(boundary_indexes[0] + 1, len(page_items)):
                                if page_items[index].get("turn_id") != through_turn_id:
                                    boundary_end = index
                                    break
                            if boundary_end < len(page_items):
                                page_items = page_items[:boundary_end]
                                more_pages = False
                            else:
                                more_pages = page.has_more
                        else:
                            more_pages = page.has_more
                    else:
                        more_pages = page.has_more
                    safe_items = tuple(item for raw in page_items
                                       if (item := _safe_item(raw)) is not None)
                    if not page_items:
                        if page.has_more:
                            raise RuntimeError("Curator pagination stalled on an empty page")
                        if through_turn_id is not None and not reached_boundary:
                            raise RuntimeError("Completed-turn Curator boundary was not found")
                        batch_outcome = "ok"
                        break
                    last_item_id = page_items[-1].get("id")
                    next_cursor = (last_item_id if len(page_items) != len(page.items)
                                   else page.cursor or last_item_id)
                    if next_cursor is None or next_cursor == cursor or next_cursor in seen_cursors:
                        raise RuntimeError("Curator pagination cursor did not advance")
                    bounded_page = SessionItemPage(tuple(page_items), next_cursor, more_pages)
                    page_identity = _source_page_identity(bounded_page)
                    if page_identity in seen_pages:
                        raise RuntimeError("Curator pagination repeated a page")
                    seen_cursors.add(next_cursor)
                    seen_pages.add(page_identity)
                    pages += 1
                    key_material = f"{session_id}:{cursor or ''}:{next_cursor or ''}"
                    operation_key = hashlib.sha256(key_material.encode()).hexdigest()
                    job_id = hashlib.sha256(f"job:{session_id}:{cursor or ''}".encode()).hexdigest()
                    if not self.store.claim_curator_job(job_id, session_id, cursor):
                        cursor = next_cursor
                        batch_outcome = "ok"
                        if not page.has_more:
                            break
                        continue
                    try:
                        mutations: list[dict[str, Any]] = []
                        handover_operation = "keep"
                        proposed_handover = None
                        if safe_items:
                            existing = tuple(_safe_existing_memory(memory) for memory in
                                             self.store.search_memories(limit=20))
                            decision = await self.model.curate(
                                session_id, safe_items, existing, handover)
                            mutations = self._validate_mutations(
                                decision.get("mutations", []), session_id, safe_items)
                            handover_operation, proposed_handover = self._validate_handover(
                                decision.get("handover", {"operation": "keep"}))
                            if handover_operation == "replace":
                                handover = proposed_handover
                            elif handover_operation == "clear":
                                handover = None
                        self.store.apply_curator_batch(
                            "openai_agents", session_id, next_cursor, last_item_id,
                            operation_key, mutations, handover_operation, proposed_handover,
                            page_items[-1].get("turn_id"))
                        self.store.finish_curator_job(job_id)
                    except BaseException as exc:
                        self.store.fail_curator_job(job_id, type(exc).__name__)
                        raise
                    cursor = next_cursor
                    batch_outcome = "ok"
                finally:
                    emit_timeline(
                        "curator.batch", "finished",
                        phase="final" if final else "incremental",
                        round=batch_round, outcome=batch_outcome,
                        duration_seconds=time.monotonic() - batch_started)
                if not more_pages:
                    break
            if through_turn_id is not None and (not reached_boundary or more_pages):
                raise FinalCatchUpIncomplete("Completed-turn Curator boundary was not reached")
            if final and more_pages:
                raise FinalCatchUpIncomplete(
                    f"Final Curator consolidation incomplete after {self.max_batches} pages")
            return handover

    @staticmethod
    def _validate_handover(value: object) -> tuple[str, str | None]:
        if not isinstance(value, dict):
            raise RuntimeError("Curator handover must be an operation object")
        operation = value.get("operation")
        if operation not in {"keep", "replace", "clear"}:
            raise RuntimeError("Curator returned an unsupported handover operation")
        if operation == "replace":
            content = value.get("content")
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("Curator replacement handover must be nonempty")
            return operation, _redact_text(content.strip())[:8000]
        if "content" in value and value.get("content") not in (None, ""):
            raise RuntimeError(f"Curator handover {operation} cannot include content")
        return operation, None

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
            if not isinstance(proposed_provenance, list) or not proposed_provenance:
                continue
            referenced_ids: set[str] = set()
            valid = True
            for evidence in proposed_provenance:
                if not isinstance(evidence, dict):
                    valid = False
                    break
                item_id = evidence.get("item_id")
                if item_id not in known_items:
                    valid = False
                    break
                if item_id in referenced_ids:
                    continue
                referenced_ids.add(item_id)
                provenance.append(_source_evidence(session_id, known_items[item_id]))
            if not valid or not provenance:
                continue
            clean["provenance"] = provenance
            accepted.append(clean)
        return accepted
