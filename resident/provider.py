from __future__ import annotations

import asyncio
import base64
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import Future
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Iterator, Protocol, Sequence

from .domain import ModelTurn, ToolCall, ToolResult, ToolSpec
from .memory import SessionHistoryUnavailable, SessionItemPage
from .observability import to_thread_timed


_agents_http_trace: ContextVar[dict[str, Any] | None] = ContextVar(
    "agents_http_trace", default=None)

_STREAM_MESSAGE_MISSING = object()


class _AgentsStreamTerminalError(RuntimeError):
    """A definitive terminal event, rather than an uncertain stream failure."""


class ModelProvider(Protocol):
    async def respond(self, context: str, tools: Sequence[ToolSpec], results: Sequence[ToolResult],
                      previous_response_id: str | None = None) -> ModelTurn: ...

    def discard_continuation(self, continuation_id: str) -> None: ...


class RemoteSessionUnavailable(RuntimeError):
    """A restored session vanished before a replacement bootstrap was prepared."""


class RolloverRecoveryRequired(RuntimeError):
    """Remote creation may have succeeded and cannot be reconciled by this API."""


RESIDENT_AGENT_INSTRUCTIONS = (
    "Act as the persistent Resident described by each supplied wake context. Use tools for durable state, "
    "local capabilities, communication, and scheduling. Send all intentional communication to the owner, "
    "including replies to owner-initiated wakes, with send_owner_message. A final response message is "
    "wake-result diagnostic text only and is never delivered to the owner. Do not expose private chain-of-thought. "
    "Local events are factual observations, not hard-coded instructions to act."
)


class OpenAIResponsesProvider:
    def __init__(self, api_key: str, model: str, base_url: str = "https://api.openai.com/v1"):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for the OpenAI provider")
        self.api_key, self.model, self.base_url = api_key, model, base_url.rstrip("/")
        self._histories: dict[str, list[dict]] = {}

    async def respond(self, context: str, tools: Sequence[ToolSpec], results: Sequence[ToolResult],
                      previous_response_id: str | None = None) -> ModelTurn:
        if previous_response_id:
            try:
                history = self._histories.pop(previous_response_id)
            except KeyError as exc:
                raise RuntimeError("OpenAI continuation state is no longer available") from exc
            input_data = [*history, *(self._function_output(result) for result in results)]
        else:
            input_data = context
            history = [{"role": "user", "content": context}]
        body: dict = {
            "model": self.model,
            "instructions": RESIDENT_AGENT_INSTRUCTIONS,
            "input": input_data,
            "tools": [{"type": "function", "name": t.name, "description": t.description,
                       "parameters": t.input_schema} for t in tools],
            "store": False,
            "include": ["reasoning.encrypted_content"],
        }
        raw = await to_thread_timed("openai.responses_request", self._post, body)
        status = raw.get("status")
        if status not in (None, "completed"):
            error = raw.get("error") or raw.get("incomplete_details") or "no details"
            raise RuntimeError(f"OpenAI response ended with status {status!r}: {error}")
        calls, texts = [], []
        for item in raw.get("output", []):
            if item.get("type") == "function_call":
                try:
                    arguments = json.loads(item.get("arguments") or "{}")
                except json.JSONDecodeError as exc:
                    arguments = {"_invalid_json": str(exc)}
                calls.append(ToolCall(item.get("call_id", item.get("id", "")), item["name"], arguments))
            elif item.get("type") == "message":
                for part in item.get("content", []):
                    if part.get("type") == "output_text":
                        texts.append(part.get("text", ""))
        usage = raw.get("usage") or {}
        if calls and raw.get("id"):
            current_input = input_data if isinstance(input_data, list) else history
            self._histories[raw["id"]] = [*current_input, *raw.get("output", [])]
        return ModelTurn(raw.get("id"), "\n".join(texts) or None, tuple(calls),
                         usage.get("input_tokens"), usage.get("output_tokens"))

    def discard_continuation(self, continuation_id: str) -> None:
        self._histories.pop(continuation_id, None)

    @staticmethod
    def _function_output(result: ToolResult) -> dict:
        metadata = json.dumps(result.output, separators=(",", ":"))
        if not result.attachments:
            output: str | list[dict] = metadata
        else:
            output = [{"type": "input_text", "text": metadata}]
            for attachment in result.attachments:
                encoded = base64.b64encode(attachment.data).decode("ascii")
                output.append({
                    "type": "input_image", "detail": attachment.detail,
                    "image_url": f"data:{attachment.mime_type};base64,{encoded}",
                })
        return {"type": "function_call_output", "call_id": result.call_id, "output": output}

    def _post(self, body: dict) -> dict:
        request = urllib.request.Request(
            f"{self.base_url}/responses", data=json.dumps(body).encode(), method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:2000]
            raise RuntimeError(f"OpenAI Responses API returned HTTP {exc.code}: {detail}") from exc


class OpenAIAgentsProvider:
    """Adapter for one long-lived managed Agents session per Resident.

    The beta wire contract is deliberately confined here. Resident still owns
    wake selection, local policy, and function execution; OpenAI owns the
    durable conversational session and agent turn loop.
    """

    uses_managed_session = True

    def __init__(self, api_key: str, model: str, base_url: str = "https://api.openai.com/v1", *,
                 agent_id: str | None = None, poll_seconds: float = 0.25,
                 timeout_seconds: float = 120.0, reasoning_effort: str | None = None,
                 service_tier: str | None = None):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for the OpenAI provider")
        self.api_key, self.model, self.base_url = api_key, model, base_url.rstrip("/")
        # Only an operator-supplied ID is known to name a saved reusable Agent.
        # The ID nested in an inline-created session describes that session's
        # resolved agent, but is not valid as agent_id on a later session create.
        self.agent_id = agent_id
        self.reasoning_effort, self.service_tier = reasoning_effort, service_tier
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds
        self._session_id: str | None = None
        self._last_turn_id: str | None = None
        self._tool_fingerprint: str | None = None
        self._protocol_descriptor: dict[str, Any] | None = None
        self._mutable_settings_descriptor: dict[str, Any] | None = None
        self._active_turn_id: str | None = None
        self._submitted_call_ids: dict[str, set[str]] = {}
        self._pending_wakes: dict[str, tuple[str, str, str]] = {}
        self._ephemeral_tool_results: dict[str, ToolResult] = {}
        self._save_binding: Callable[[str, str | None, str | None], None] = lambda *_: None
        self._binding_writer: ContextVar[
            Callable[[str, str | None, str | None], None] | None
        ] = ContextVar("agents_binding_writer", default=None)
        self._lifecycle_writer: ContextVar[Callable[[Callable, tuple], Any] | None] = (
            ContextVar("agents_lifecycle_writer", default=None))
        self._pending_binding: tuple[str, str | None, str | None] | None = None
        self._begin_action: Callable[..., dict] | None = None
        self._complete_action: Callable[..., None] | None = None
        self._load_protocol: Callable[[str], dict | None] = lambda _session_id: None
        self._save_protocol: Callable[[str, dict], None] = lambda *_: None
        self._load_mutable: Callable[[str], dict | None] = lambda _session_id: None
        self._save_mutable: Callable[[str, dict], None] = lambda *_: None
        self._bind_initial_session: Callable[..., None] = lambda *_: None
        self._load_pending_rollover: Callable[[], dict | None] = lambda: None
        self._begin_rollover: Callable[..., dict] = (
            lambda old, reason, requested_by, request, protocol, mutable: {
                "id": "", "old_session_id": old, "reason": reason,
                "creation_state": "not_attempted", "create_request": request,
                "protocol_descriptor": protocol, "mutable_settings": mutable})
        self._mark_rollover_create_started: Callable[[str], None] = lambda *_: None
        self._bind_rollover: Callable[..., None] = lambda *_: None
        self._complete_rollover: Callable[[str], None] = lambda *_: None
        self._fail_rollover: Callable[[str, str], None] = lambda *_: None
        self._requested_rollover_reason: str | None = None
        self._unavailable_session_id: str | None = None
        self._unavailable_session_reason: str | None = None
        self._preflight_session_status: str | None = None
        self._confirmed_rollover_session: dict[str, Any] | None = None
        self._rollover_deferred_while_busy = False
        self._lifecycle_bound = False

    def bind_session_store(self, load: Callable[[], dict | None],
                           save: Callable[[str, str | None, str | None], None]) -> None:
        self._save_binding = save
        binding = load()
        if binding is not None:
            persisted_agent_id = binding.get("agent_id")
            # Configuration drift does not make the old session disappear.  Keep
            # it bound so the common idle/final-curation/handover lifecycle owns
            # the saved-Agent transition just like every other immutable change.
            self._session_id = binding["session_id"]
            self._last_turn_id = binding.get("last_turn_id")
            if self.agent_id is None and persisted_agent_id is not None:
                # Older versions persisted session-local agent IDs. Keep the
                # recoverable session and turn, but migrate away from ever
                # presenting that ID as a saved Agent resource.
                save(self._session_id, None, self._last_turn_id)

    def bind_action_store(self, begin: Callable[..., dict], complete: Callable[..., None]) -> None:
        self._begin_action, self._complete_action = begin, complete

    def bind_lifecycle_store(self, load_protocol: Callable[[str], dict | None],
                             save_protocol: Callable[[str, dict], None],
                             load_mutable: Callable[[str], dict | None],
                             save_mutable: Callable[[str, dict], None],
                             load_pending_rollover: Callable[[], dict | None],
                             begin_rollover: Callable[[str | None, str, str, dict], dict],
                             mark_rollover_create_started: Callable[[str], None],
                             bind_rollover: Callable[..., None],
                             complete_rollover: Callable[[str], None],
                             fail_rollover: Callable[[str, str], None],
                             bind_initial_session: Callable[..., None]) -> None:
        self._lifecycle_bound = True
        self._load_protocol, self._save_protocol = load_protocol, save_protocol
        self._load_mutable, self._save_mutable = load_mutable, save_mutable
        self._load_pending_rollover = load_pending_rollover
        self._begin_rollover = begin_rollover
        self._mark_rollover_create_started = mark_rollover_create_started
        self._bind_rollover = bind_rollover
        self._complete_rollover, self._fail_rollover = complete_rollover, fail_rollover
        self._bind_initial_session = bind_initial_session
        if self._session_id is not None:
            self._protocol_descriptor = load_protocol(self._session_id)
            self._mutable_settings_descriptor = load_mutable(self._session_id)
        pending = load_pending_rollover()
        if (pending is not None and (self._session_id is None
                                    or pending.get("old_session_id") == self._session_id)):
            if self._session_id is None and pending.get("old_session_id") is not None:
                # A durable create attempt takes precedence over a newly desired
                # saved Agent ID until its recorded configuration is replayed.
                self._session_id = pending["old_session_id"]
                self._last_turn_id = None
                self._protocol_descriptor = load_protocol(self._session_id)
                self._mutable_settings_descriptor = load_mutable(self._session_id)
            self._requested_rollover_reason = pending["reason"]

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def session_protocol_known(self) -> bool:
        return self._session_id is None or self._protocol_descriptor is not None

    @property
    def unavailable_session_reason(self) -> str | None:
        return self._unavailable_session_reason

    def request_rollover(self, reason: str = "explicit_new_chapter") -> None:
        if not reason.strip():
            raise ValueError("Session rollover reason must be nonempty")
        self._requested_rollover_reason = reason.strip()

    @property
    def rollover_ready(self) -> bool:
        """Whether the most recent preflight found the old session idle."""
        return (self._session_id is None or self._unavailable_session_id == self._session_id
                or self._preflight_session_status == "idle")

    @property
    def requested_rollover_reason(self) -> str | None:
        return self._requested_rollover_reason

    @staticmethod
    def _session_usability(session: dict[str, Any]) -> str:
        """Classify only session states represented by the Agents adapter contract."""
        status = session.get("status")
        if status == "idle":
            return "usable"
        if status in {"in_progress", "requires_action"}:
            return "busy"
        if status == "failed":
            return "terminal"
        raise RuntimeError(
            f"OpenAI Agents session returned unsupported status {status!r}")

    @staticmethod
    def _definitive_session_unavailable(exc: Exception) -> bool:
        """Recognize only provider responses that definitively retire a session."""
        message = str(exc)
        return any(f"HTTP {status}:" in message for status in (404, 410))

    def _mark_session_unavailable(self, session_id: str, reason: str) -> None:
        self._unavailable_session_id = session_id
        self._unavailable_session_reason = reason
        self._preflight_session_status = None
        self._confirmed_rollover_session = None
        if self._requested_rollover_reason is None:
            self._requested_rollover_reason = reason

    async def preflight_session(self) -> str | None:
        """Detect an unavailable restored session before wake context is built."""
        session_id = self._session_id
        if session_id is None:
            self._preflight_session_status = None
            return None
        try:
            session = await asyncio.to_thread(
                self._request, "GET", f"/agents/sessions/{session_id}")
        except RuntimeError as exc:
            if not self._definitive_session_unavailable(exc):
                raise
            self._mark_session_unavailable(session_id, "remote_session_missing")
            return "remote_session_missing"
        self._preflight_session_status = session.get("status")
        usability = self._session_usability(session)
        if usability == "terminal":
            self._mark_session_unavailable(session_id, "remote_session_failed")
            return "remote_session_failed"
        if usability == "usable":
            # A fresh idle preflight gives Runtime the opportunity to prepare
            # final curation and bootstrap input for this wake.
            self._rollover_deferred_while_busy = False
        remote_agent = session.get("agent")
        remote_agent_id = (remote_agent.get("id")
                           if isinstance(remote_agent, dict) else None)
        if (self.agent_id is not None and remote_agent_id is not None
                and remote_agent_id != self.agent_id
                and self._requested_rollover_reason is None):
            # Make the immutable mismatch visible before Runtime decides whether
            # final curation and a handover can safely be prepared.  A busy
            # session remains bound and the request is merely carried forward.
            self._requested_rollover_reason = "saved_agent_id_changed"
        return None

    async def confirm_rollover_ready(self) -> bool:
        """Recheck idleness and preserve that exact observation for creation."""
        session_id = self._session_id
        if session_id is None or self._unavailable_session_id == session_id:
            return True
        try:
            session = await asyncio.to_thread(
                self._request, "GET", f"/agents/sessions/{session_id}")
        except RuntimeError as exc:
            if not self._definitive_session_unavailable(exc):
                raise
            self._mark_session_unavailable(session_id, "remote_session_missing")
            return True
        self._preflight_session_status = session.get("status")
        usability = self._session_usability(session)
        if usability == "terminal":
            self._mark_session_unavailable(session_id, "remote_session_failed")
            return True
        self._confirmed_rollover_session = session if usability == "usable" else None
        return self._confirmed_rollover_session is not None

    async def session_items(self, cursor: str | None, limit: int = 50) -> SessionItemPage:
        if self._session_id is None:
            return SessionItemPage((), cursor, False)
        return await asyncio.to_thread(self._session_items_sync, self._session_id, cursor, limit)

    def _session_items_sync(self, session_id: str, cursor: str | None,
                            limit: int) -> SessionItemPage:
        from urllib.parse import quote
        path = f"/agents/sessions/{session_id}/items?order=asc&limit={max(1, min(limit, 100))}"
        if cursor:
            path += f"&after={quote(cursor, safe='')}"
        try:
            page = self._request("GET", path)
        except RuntimeError as exc:
            if self._definitive_session_unavailable(exc):
                raise SessionHistoryUnavailable(
                    f"Session item history for {session_id} is unavailable") from exc
            raise
        data = tuple(item for item in (page.get("data") or []) if isinstance(item, dict))
        next_cursor = page.get("last_id") or (data[-1].get("id") if data else cursor)
        return SessionItemPage(data, next_cursor, bool(page.get("has_more")))

    def prepare_tool_call(self, call: ToolCall) -> dict | None:
        if self._begin_action is None or self._session_id is None or self._active_turn_id is None:
            return None
        action = self._begin_action(
            "openai_agents", self._session_id, self._active_turn_id,
            call.id, call.name, call.arguments)
        if not action["claimed"] and action.get("attachments_ephemeral"):
            action["ephemeral_result"] = self._ephemeral_tool_results.get(call.id)
        return action

    def record_tool_result(self, result: ToolResult) -> None:
        if self._complete_action is not None and self._session_id is not None:
            if result.attachments:
                self._ephemeral_tool_results[result.call_id] = result
            self._complete_action(
                "openai_agents", self._session_id, result.call_id, result.output,
                bool(result.attachments))

    async def respond(self, context: str, tools: Sequence[ToolSpec], results: Sequence[ToolResult],
                      previous_response_id: str | None = None) -> ModelTurn:
        loop = asyncio.get_running_loop()

        def save_on_event_loop(session_id: str, agent_id: str | None,
                               last_turn_id: str | None) -> None:
            completed: Future[None] = Future()

            def save() -> None:
                try:
                    self._save_binding(session_id, agent_id, last_turn_id)
                except BaseException as exc:
                    completed.set_exception(exc)
                else:
                    completed.set_result(None)

            # The HTTP/session state machine stays in the worker, but Store's
            # SQLite connection remains owned by the runtime event-loop thread.
            # Wait for the checkpoint so remote work cannot outrun durability.
            loop.call_soon_threadsafe(save)
            completed.result()

        # asyncio.to_thread propagates this context into only this response's
        # worker, avoiding a process-wide or connection-wide thread escape.
        token = self._binding_writer.set(save_on_event_loop)
        def lifecycle_on_event_loop(function: Callable, arguments: tuple) -> Any:
            completed: Future[Any] = Future()

            def invoke() -> None:
                try:
                    completed.set_result(function(*arguments))
                except BaseException as exc:
                    completed.set_exception(exc)

            loop.call_soon_threadsafe(invoke)
            return completed.result()

        lifecycle_token = self._lifecycle_writer.set(lifecycle_on_event_loop)
        http_trace: dict[str, Any] = {
            "lifecycle_started": time.monotonic(), "events": []}
        trace_token = _agents_http_trace.set(http_trace)

        def flush_http_trace() -> None:
            from .observability import emit_timeline
            previous_finished = http_trace["lifecycle_started"]
            for event in sorted(
                    http_trace["events"],
                    key=lambda entry: entry["started_monotonic_seconds"]):
                started = event["started_monotonic_seconds"]
                operation = event.pop("timeline_operation", "openai.agents_http")
                emit_timeline(
                    operation, "finished",
                    gap_since_previous_seconds=max(0.0, started - previous_finished),
                    lifecycle_offset_seconds=max(
                        0.0, started - http_trace["lifecycle_started"]),
                    **event)
                previous_finished = event["finished_monotonic_seconds"]

        try:
            return await to_thread_timed(
                "openai.agents_lifecycle", self._respond_sync,
                context, tools, results, previous_response_id,
                timeline_before_finished=flush_http_trace,
                request=("tool_results" if results else "wake"),
                tool_result_count=len(results), turn_id=previous_response_id)
        finally:
            _agents_http_trace.reset(trace_token)
            self._lifecycle_writer.reset(lifecycle_token)
            self._binding_writer.reset(token)

    def discard_continuation(self, continuation_id: str) -> None:
        # Managed sessions retain turn state. A failed local wake is reconciled
        # from the session on the next attempt rather than erased locally.
        return None

    def _respond_sync(self, context: str, tools: Sequence[ToolSpec], results: Sequence[ToolResult],
                      previous_response_id: str | None) -> ModelTurn:
        if results:
            session, _ = self._ensure_session(tools, allow_create=False)
            session_id = session["id"]
            turn_id = previous_response_id
            if not turn_id:
                raise RuntimeError("Agents tool results require the requesting turn id")
            events = [self._tool_result_event(result, turn_id) for result in results]
            call_key = ":".join(sorted(result.call_id for result in results))
            def submit_results() -> None:
                self._submit_events(
                    session_id, events, f"resident-tool:{turn_id}:{call_key}"[:256])
                self._submitted_call_ids.setdefault(turn_id, set()).update(
                    result.call_id for result in results)

            turn = self._submit_and_stream(
                session_id, submit_results, expected_turn_id=turn_id)
            if turn.tool_calls or turn_id not in self._pending_wakes:
                return turn
            pending_context, wake_key, correlation = self._pending_wakes[turn_id]
            completed = self._submit_ordinary_wake(
                tools, pending_context, wake_key, correlation)
            self._pending_wakes.pop(turn_id, None)
            return completed

        wake_key = self._wake_idempotency_key(context)
        correlated_context, correlation = self._correlated_context(context, wake_key)
        session, created_with_input = self._ensure_session(
            tools, initial_input=correlated_context)
        session_id = session["id"]
        if created_with_input:
            return self._wait_for_submitted_wake(session_id, correlation, wake_key)

        # Creation has no event idempotency key. If the local binding checkpoint
        # failed, or the process restarted after accepting the initial input,
        # recover that correlated turn instead of submitting the wake again.
        existing_turn_id = self._correlated_turn_id(session_id, correlation, wake_key)
        if existing_turn_id:
            return self._wait_for_turn(session_id, existing_turn_id)
        if not self._is_owner_wake(context):
            recovered = self._reconcile_before_wake(session_id, session)
            if recovered is not None and recovered.tool_calls:
                if not recovered.response_id:
                    raise RuntimeError("Recovered Agents tool calls have no turn id")
                self._pending_wakes[recovered.response_id] = (
                    correlated_context, wake_key, correlation)
                return recovered
            return self._submit_ordinary_wake(
                tools, correlated_context, wake_key, correlation)
        return self._submit_wake(session_id, correlated_context, wake_key, correlation)

    def _submit_ordinary_wake(self, tools: Sequence[ToolSpec], context: str,
                              wake_key: str, correlation: str) -> ModelTurn:
        """Submit only after an idle session has accepted current configuration."""
        session, created_with_input = self._ensure_session(
            tools, initial_input=context)
        if created_with_input:
            return self._wait_for_submitted_wake(session["id"], correlation, wake_key)
        if session.get("status") != "idle":
            raise RuntimeError(
                "OpenAI Agents session was not idle after wake reconciliation")
        return self._submit_wake(session["id"], context, wake_key, correlation)

    def _submit_wake(self, session_id: str, context: str, wake_key: str,
                     correlation: str) -> ModelTurn:
        event = {
            "type": "agent.session.input.message",
            "input": [{"role": "user", "content": [
                {"type": "input_text", "text": context}
            ]}],
        }
        return self._submit_and_stream(
            session_id,
            lambda: self._submit_events(
                session_id, [event], f"resident-wake:{wake_key}"[:256]),
            correlation=correlation, wake_key=wake_key)

    def _wait_for_submitted_wake(self, session_id: str, correlation: str,
                                 wake_key: str) -> ModelTurn:
        # Session creation can accept its initial input before a stream can be
        # opened. Reconcile exactly rather than assuming that a live stream can
        # replay the already-created turn.
        turn_id = self._wait_for_correlated_turn(session_id, correlation, wake_key)
        return self._wait_for_turn(session_id, turn_id)

    def _submit_and_stream(self, session_id: str, submit: Callable[[], None], *,
                           expected_turn_id: str | None = None,
                           correlation: str | None = None,
                           wake_key: str | None = None) -> ModelTurn:
        """Open a live stream before submission, falling back on uncertainty."""
        if "_request" in self.__dict__ and "_open_event_stream" not in self.__dict__:
            # Existing request-level test doubles model the reconciliation path.
            submit()
            return self._fallback_wait(
                session_id, expected_turn_id, correlation, wake_key, "stream_unavailable")
        submission_attempted = False
        try:
            with self._open_event_stream(session_id) as events:
                submission_attempted = True
                submit()
                return self._consume_event_stream(
                    session_id, events, expected_turn_id=expected_turn_id,
                    correlation=correlation, wake_key=wake_key)
        except _AgentsStreamTerminalError:
            raise
        except Exception as exc:
            if submission_attempted:
                return self._fallback_wait(
                    session_id, expected_turn_id, correlation, wake_key,
                    self._stream_fallback_reason(exc))
            submit()
            return self._fallback_wait(
                session_id, expected_turn_id, correlation, wake_key,
                "stream_connect_error")

    def _fallback_wait(self, session_id: str, expected_turn_id: str | None,
                       correlation: str | None, wake_key: str | None,
                       reason: str) -> ModelTurn:
        self._trace_instant("openai.agents_fallback", reason=reason)
        if expected_turn_id:
            return self._wait_for_turn(session_id, expected_turn_id)
        if correlation is None or wake_key is None:
            raise RuntimeError("Agents stream fallback lacks wake correlation")
        return self._wait_for_submitted_wake(session_id, correlation, wake_key)

    def _consume_event_stream(self, session_id: str, events: Iterator[dict], *,
                              expected_turn_id: str | None,
                              correlation: str | None,
                              wake_key: str | None) -> ModelTurn:
        turn_id = expected_turn_id
        messages: dict[int, list[str]] = {}
        saw_complete_message = False
        for event in events:
            if event.get("session_id", session_id) != session_id:
                raise ValueError("Agents stream event belongs to another session")
            event_type = event.get("type")
            event_turn_id = event.get("turn_id")
            if event_type == "agent.session.turn.item.added" and turn_id is None:
                item = event.get("item")
                if (isinstance(item, dict) and correlation is not None
                        and wake_key is not None
                        and self._item_matches_wake(item, correlation, wake_key)):
                    candidate = event_turn_id or item.get("turn_id")
                    if not candidate:
                        raise ValueError("Correlated stream item has no turn id")
                    turn_id = candidate
                continue
            if event_type == "agent.session.requires_action":
                session = event.get("session")
                if not isinstance(session, dict) or session.get("id") != session_id:
                    raise ValueError("Malformed Agents required-action event")
                action_turn_ids = {
                    action.get("turn_id")
                    for action in session.get("required_actions", [])
                    if isinstance(action, dict) and action.get("type") == "function_call"
                }
                if action_turn_ids != {turn_id}:
                    raise ValueError("Agents required actions belong to another turn")
                turn = self._required_actions_turn(session)
                if turn.tool_calls:
                    return turn
                continue
            if event_type == "agent.session.turn.item.done" and event_turn_id == turn_id:
                item = event.get("item")
                output_index = event.get("output_index")
                if (not isinstance(item, dict) or not isinstance(output_index, int)
                        or item.get("turn_id", turn_id) != turn_id):
                    raise ValueError("Malformed Agents completed-item event")
                if item.get("type") == "message" and item.get("role") == "assistant":
                    content = item.get("content")
                    if not isinstance(content, list):
                        raise ValueError("Malformed streamed assistant message")
                    texts = []
                    for part in content:
                        if not isinstance(part, dict):
                            raise ValueError("Malformed streamed assistant content")
                        if part.get("type") == "output_text":
                            text = part.get("text")
                            if not isinstance(text, str):
                                raise ValueError("Malformed streamed output text")
                            if text:
                                texts.append(text)
                    messages[output_index] = texts
                    saw_complete_message = True
                continue
            if event_type == "agent.session.turn.completed" and event_turn_id == turn_id:
                turn = event.get("turn")
                if not isinstance(turn, dict) or turn.get("id") != turn_id:
                    raise ValueError("Malformed Agents completed-turn event")
                if event.get("usage") is not None:
                    turn = {**turn, "usage": event["usage"]}
                message: object = _STREAM_MESSAGE_MISSING
                if saw_complete_message:
                    message = "\n".join(
                        text for index in sorted(messages) for text in messages[index]) or None
                return self._completed_turn(
                    session_id, {}, turn, streamed_message=message)
            if event_type in ("agent.session.turn.failed", "agent.session.turn.cancelled"):
                if event_turn_id != turn_id:
                    continue
                turn = event.get("turn") if isinstance(event.get("turn"), dict) else {}
                if event_type.endswith("failed"):
                    raise _AgentsStreamTerminalError(
                        f"OpenAI Agents turn failed: {turn.get('error') or 'no details'}")
                raise _AgentsStreamTerminalError("OpenAI Agents turn was cancelled")
            if event_type in ("agent.session.failed", "error"):
                raise _AgentsStreamTerminalError(
                    f"OpenAI Agents session failed: "
                    f"{event.get('error') or event.get('session', {}).get('error') or 'no details'}")
        raise EOFError("OpenAI Agents event stream ended before the turn settled")

    @staticmethod
    def _stream_fallback_reason(exc: Exception) -> str:
        if isinstance(exc, (TimeoutError, urllib.error.URLError)):
            return "stream_timeout_or_disconnect"
        if isinstance(exc, (ValueError, json.JSONDecodeError)):
            return "stream_malformed"
        if isinstance(exc, EOFError):
            return "stream_eof"
        return "stream_error"

    def _reconcile_before_wake(self, session_id: str, session: dict) -> ModelTurn | None:
        """Finish remote work without attributing it to the next ordinary wake."""
        status = session.get("status")
        usability = self._session_usability(session)
        if usability == "terminal":
            raise RuntimeError(
                f"OpenAI Agents session failed: {session.get('error') or 'no details'}")
        if usability == "usable":
            latest = self._latest_turn(session_id)
            if latest is not None and latest.get("id") != self._last_turn_id:
                self._completed_turn(session_id, session, latest)
            return None
        if status == "requires_action":
            turn = self._required_actions_turn(session)
            if turn.tool_calls:
                return turn
            return self._wait_for_turn(session_id, turn.response_id)
        latest = self._latest_turn(session_id)
        if latest is None or not latest.get("id"):
            raise RuntimeError("Active Agents session has no recoverable turn")
        return self._wait_for_turn(session_id, latest["id"])

    def _ensure_session(self, tools: Sequence[ToolSpec], *,
                        initial_input: str | None = None,
                        allow_create: bool = True) -> tuple[dict, bool]:
        self._flush_pending_binding()
        agent = self._agent_config(tools)
        desired_protocol = self._agent_protocol(agent)
        fingerprint = json.dumps(agent, sort_keys=True, separators=(",", ":"))
        if self._session_id is not None:
            if self._unavailable_session_id == self._session_id:
                pending = self._pending_rollover_for_bound_session()
                requested_reason = self._requested_rollover_reason
                reason = (pending["reason"] if pending is not None else
                          self._unavailable_session_reason or "remote_session_missing")
                result = self._intentional_rollover(
                    reason, agent, desired_protocol, initial_input, allow_create,
                    finalization_status="unavailable")
                if (pending is not None and requested_reason is not None
                        and requested_reason != reason):
                    self._requested_rollover_reason = requested_reason
                return result
            try:
                confirmed = self._confirmed_rollover_session
                self._confirmed_rollover_session = None
                session = (confirmed if confirmed is not None
                           and confirmed.get("id") == self._session_id else
                           self._request("GET", f"/agents/sessions/{self._session_id}"))
            except RuntimeError as exc:
                if not self._definitive_session_unavailable(exc):
                    raise
                self._mark_session_unavailable(
                    self._session_id, "remote_session_missing")
                raise RemoteSessionUnavailable(str(exc)) from exc
            else:
                if self._session_usability(session) == "terminal":
                    self._mark_session_unavailable(
                        self._session_id, "remote_session_failed")
                    raise RemoteSessionUnavailable(
                        "OpenAI Agents session is terminal and cannot accept another wake")
                remote_agent = session.get("agent")
                applied_protocol = self._protocol_descriptor
                if (applied_protocol is None and isinstance(remote_agent, dict)
                        and {"model", "instructions", "tools"} <= remote_agent.keys()):
                    applied_protocol = self._agent_protocol(remote_agent)
                if applied_protocol is None and self._tool_fingerprint is not None:
                    try:
                        applied_protocol = self._agent_protocol(
                            json.loads(self._tool_fingerprint))
                    except (TypeError, json.JSONDecodeError):
                        pass
                compatibility = self._protocol_compatibility(applied_protocol, desired_protocol)
                if applied_protocol is None and self._lifecycle_bound:
                    # A legacy binding without an applied descriptor cannot prove
                    # that today's immutable protocol was used.
                    compatibility = "rollover"

                # Transition precedence is deliberate: an unresolved durable
                # attempt owns this old session before any newly inferred reason.
                # Newer desired state is retained and reconciled after that exact
                # historical create request has completed.
                pending = self._pending_rollover_for_bound_session()
                remote_agent_id = (remote_agent.get("id")
                                   if isinstance(remote_agent, dict) else None)
                inferred_reason = None
                if (self.agent_id is not None and remote_agent_id is not None
                        and remote_agent_id != self.agent_id):
                    inferred_reason = "saved_agent_id_changed"
                elif compatibility == "rollover":
                    inferred_reason = "function_or_immutable_protocol_changed"

                requested_reason = self._requested_rollover_reason
                if pending is not None:
                    rollover_reason = pending["reason"]
                    future_reason = (
                        requested_reason if requested_reason != rollover_reason else None)
                    if (future_reason is None and inferred_reason is not None
                            and pending.get("protocol_descriptor") != desired_protocol):
                        future_reason = inferred_reason
                else:
                    rollover_reason = requested_reason or inferred_reason
                    future_reason = None

                if rollover_reason is not None:
                    self._requested_rollover_reason = rollover_reason
                    if session.get("status") != "idle":
                        # Detection records intent only.  Active and requires-action
                        # sessions finish under their existing immutable contract.
                        self._rollover_deferred_while_busy = self._lifecycle_bound
                        return session, False
                    if self._lifecycle_bound and self._rollover_deferred_while_busy:
                        # This input was assembled before the active turn became
                        # idle, so it has no final catch-up/handover bootstrap.
                        # Let it use the old session and leave the request pending
                        # for the next preflight-prepared wake boundary.
                        self._rollover_deferred_while_busy = False
                        return session, False
                    result = self._intentional_rollover(
                        rollover_reason, agent, desired_protocol,
                        initial_input, allow_create)
                    if pending is not None:
                        self._requested_rollover_reason = future_reason
                    return result

                if self._tool_fingerprint is None:
                    if (not isinstance(remote_agent, dict)
                            or not {"model", "instructions", "tools"} <= remote_agent.keys()
                            or self._agent_config_matches(remote_agent, agent)):
                        # A restored provider has no in-memory fingerprint.
                        # Recognize a matching persisted session from its returned
                        # config. A partial legacy response cannot prove a mismatch.
                        self._tool_fingerprint = fingerprint

                # Mutable settings are an independent reconciliation dimension.
                # A compatible local revocation retains the old immutable remote
                # descriptor, but must not suppress a safe mutable update.
                mutable_patch = self._mutable_patch(remote_agent)
                if mutable_patch:
                    if session.get("status") != "idle":
                        return session, False
                    patched = self._request(
                        "POST", f"/agents/sessions/{self._session_id}",
                        {"agent": mutable_patch})
                    session = {**session, **patched}
                    desired_mutable = self._desired_mutable_settings()
                    self._lifecycle_call(
                        self._save_mutable, self._session_id, desired_mutable)
                    self._mutable_settings_descriptor = desired_mutable
                if compatibility != "compatible_revocation":
                    self._protocol_descriptor = desired_protocol
                    self._lifecycle_call(
                        self._save_protocol, self._session_id, desired_protocol)
                    self._tool_fingerprint = fingerprint
                return session, False
        if not allow_create:
            raise RuntimeError(
                "OpenAI Agents session disappeared while continuing a turn")
        if initial_input is None:
            raise RuntimeError(
                "Conversation-only OpenAI Agents sessions require initial input")
        body: dict = {
            "environment": {"type": "none"},
            "agent": agent,
            "input": initial_input,
            "metadata": {"managed_by": "resident"},
        }
        if self.agent_id:
            body["agent_id"] = self.agent_id
        mutable_settings = self._desired_mutable_settings()
        if self._lifecycle_bound:
            reason = self._requested_rollover_reason or "initial_session"
            return self._create_initial_session(
                reason, body, desired_protocol, mutable_settings, fingerprint)
        else:
            session = self._request("POST", "/agents/sessions", body)
            session_id = session["id"]
            # Keep the standalone adapter's retry checkpoint behavior: if its
            # binding callback fails, the known remote ID is retried before use.
            self._session_id = session_id
            self._persist_binding(session_id, self.agent_id, None)
            self._save_protocol(session_id, desired_protocol)
            self._save_mutable(session_id, mutable_settings)
        self._session_id = session_id
        self._last_turn_id = None
        self._tool_fingerprint = fingerprint
        self._protocol_descriptor = desired_protocol
        self._mutable_settings_descriptor = mutable_settings
        return session, True

    def _create_initial_session(self, reason: str, body: dict[str, Any],
                                descriptor: dict[str, Any],
                                mutable_settings: dict[str, Any],
                                fingerprint: str) -> tuple[dict, bool]:
        """Create the first session only after its exact request is durable."""
        attempt = self._lifecycle_call(
            self._begin_rollover, None, reason, "runtime", body,
            descriptor, mutable_settings)
        attempt_id = attempt["id"]
        if attempt.get("creation_state") == "create_uncertain":
            raise RolloverRecoveryRequired(
                "Initial session creation may have succeeded before Resident could "
                "record its ID. The Agents API exposes no supported create-idempotency "
                "or reconciliation contract, so automatic re-creation is blocked; "
                f"create attempt {attempt_id} requires operator reconciliation.")
        if attempt.get("creation_state") != "not_attempted":
            raise RuntimeError(
                f"Initial create attempt {attempt_id} is not in a creatable state: "
                f"{attempt.get('creation_state')}")
        persisted_body = attempt.get("create_request")
        persisted_descriptor = attempt.get("protocol_descriptor")
        persisted_mutable = attempt.get("mutable_settings")
        if not isinstance(persisted_body, dict):
            raise RuntimeError(
                f"Initial create attempt {attempt_id} has no durable create request")
        if (not isinstance(persisted_descriptor, dict)
                or not isinstance(persisted_mutable, dict)):
            raise RuntimeError(
                f"Initial create attempt {attempt_id} has no durable configuration snapshot")
        self._lifecycle_call(self._mark_rollover_create_started, attempt_id)
        try:
            session = self._request("POST", "/agents/sessions", persisted_body)
            session_id = session["id"]
            self._lifecycle_call(
                self._bind_rollover, attempt_id, session_id,
                persisted_descriptor.get("saved_agent_id"), "not_applicable")
            # The binding and exact create-time descriptors are now one durable
            # transaction. That is the point at which a new-chapter request for
            # a previously sessionless Resident has been satisfied.
            self._requested_rollover_reason = None
            self._session_id, self._last_turn_id = session_id, None
            self._pending_binding = None
            self._protocol_descriptor = persisted_descriptor
            self._mutable_settings_descriptor = persisted_mutable
            self._tool_fingerprint = json.dumps(
                persisted_body.get("agent", {}), sort_keys=True, separators=(",", ":"))
            self._lifecycle_call(self._complete_rollover, attempt_id)
            return session, True
        except Exception as exc:
            if self._definitive_create_rejection(exc):
                self._lifecycle_call(
                    self._fail_rollover, attempt_id, "create_rejected")
            raise

    def _pending_rollover_for_bound_session(self) -> dict[str, Any] | None:
        """Return only unresolved durable work owned by the current binding."""
        pending = self._lifecycle_call(self._load_pending_rollover)
        if (pending is None or pending.get("old_session_id") != self._session_id
                or pending.get("creation_state") not in {
                    "not_attempted", "create_uncertain"}):
            return None
        return pending

    def _intentional_rollover(self, reason: str, agent: dict,
                              descriptor: dict[str, Any], initial_input: str | None,
                              allow_create: bool, *,
                              finalization_status: str = "completed") -> tuple[dict, bool]:
        if not allow_create:
            raise RuntimeError("OpenAI Agents session requires rollover while continuing a turn")
        if initial_input is None:
            raise RuntimeError("Intentional session rollover requires bootstrap input")
        old_session_id = self._session_id
        body: dict = {
            "environment": {"type": "none"}, "agent": agent, "input": initial_input,
            "metadata": {"managed_by": "resident", "rollover_reason": reason},
        }
        if self.agent_id:
            body["agent_id"] = self.agent_id
        mutable_settings = self._desired_mutable_settings()
        rollover = self._lifecycle_call(
            self._begin_rollover, old_session_id, reason, "runtime", body,
            descriptor, mutable_settings)
        rollover_id = rollover["id"]
        if rollover.get("creation_state") == "create_uncertain":
            raise RolloverRecoveryRequired(
                "Replacement session creation may have succeeded before Resident could "
                "record its ID. The Agents API exposes no supported create-idempotency or "
                "lookup-by-rollover-token contract, so automatic re-creation is blocked; "
                f"rollover {rollover_id} requires operator reconciliation.")
        if rollover.get("creation_state") != "not_attempted":
            raise RuntimeError(
                f"Rollover {rollover_id} is not in a creatable state: "
                f"{rollover.get('creation_state')}")
        persisted_body = rollover.get("create_request")
        if not isinstance(persisted_body, dict):
            raise RuntimeError(f"Rollover {rollover_id} has no durable create request")
        persisted_descriptor = rollover.get("protocol_descriptor")
        persisted_mutable = rollover.get("mutable_settings")
        if not isinstance(persisted_descriptor, dict) or not isinstance(persisted_mutable, dict):
            raise RuntimeError(f"Rollover {rollover_id} has no durable configuration snapshot")
        self._lifecycle_call(self._mark_rollover_create_started, rollover_id)
        try:
            session = self._request("POST", "/agents/sessions", persisted_body)
            new_session_id = session["id"]
            if self._lifecycle_bound:
                self._lifecycle_call(
                    self._bind_rollover, rollover_id, new_session_id,
                    persisted_descriptor.get("saved_agent_id"),
                    finalization_status)
            self._session_id, self._last_turn_id = new_session_id, None
            self._unavailable_session_id = None
            self._unavailable_session_reason = None
            self._requested_rollover_reason = (
                reason if rollover.get("reason") != reason else None)
            self._protocol_descriptor = persisted_descriptor
            self._mutable_settings_descriptor = persisted_mutable
            self._tool_fingerprint = json.dumps(
                persisted_body.get("agent", {}), sort_keys=True, separators=(",", ":"))
            if self._lifecycle_bound:
                self._pending_binding = None
                self._lifecycle_call(self._complete_rollover, rollover_id)
            else:
                self._persist_binding(
                    new_session_id, persisted_descriptor.get("saved_agent_id"), None)
                self._save_protocol(new_session_id, persisted_descriptor)
                self._save_mutable(new_session_id, persisted_mutable)
            return session, True
        except Exception as exc:
            if self._definitive_create_rejection(exc):
                self._lifecycle_call(self._fail_rollover, rollover_id, "create_rejected")
            raise

    def _persist_binding(self, session_id: str, agent_id: str | None,
                         last_turn_id: str | None) -> None:
        binding = (session_id, agent_id, last_turn_id)
        self._pending_binding = binding
        writer = self._binding_writer.get() or self._save_binding
        writer(*binding)
        if self._pending_binding == binding:
            self._pending_binding = None

    def _flush_pending_binding(self) -> None:
        if self._pending_binding is not None:
            self._persist_binding(*self._pending_binding)

    def _agent_config(self, tools: Sequence[ToolSpec]) -> dict:
        agent = {
            "model": self.model,
            "instructions": RESIDENT_AGENT_INSTRUCTIONS,
            "tools": [{
                "type": "function", "name": tool.name,
                "description": tool.description, "parameters": tool.input_schema,
            } for tool in tools],
        }
        if self.reasoning_effort is not None:
            agent["reasoning"] = {"effort": self.reasoning_effort}
        if self.service_tier is not None:
            agent["service_tier"] = self.service_tier
        return agent

    def _desired_mutable_settings(self) -> dict[str, Any]:
        settings: dict[str, Any] = {"model": self.model}
        if self.reasoning_effort is not None:
            settings["reasoning"] = {"effort": self.reasoning_effort}
        if self.service_tier is not None:
            settings["service_tier"] = self.service_tier
        return settings

    def _mutable_patch(self, remote_agent: object) -> dict[str, Any]:
        remote = remote_agent if isinstance(remote_agent, dict) else {}
        desired = self._desired_mutable_settings()
        applied = self._mutable_settings_descriptor or {}
        patch: dict[str, Any] = {}
        for key, desired_value in desired.items():
            if key in remote:
                known, current = True, remote.get(key)
            elif key in applied:
                known, current = True, applied.get(key)
            else:
                known, current = False, None
            if (known and current != desired_value) or (
                    not known and desired_value is not None
                    and (key != "model" or self._lifecycle_bound)):
                patch[key] = desired_value
        return patch

    @staticmethod
    def _definitive_create_rejection(exc: Exception) -> bool:
        message = str(exc)
        return any(f"HTTP {status}" in message for status in (
            400, 401, 403, 404, 405, 406, 410, 411, 413, 414, 415, 422))

    def protocol_change_requires_rollover(self, tools: Sequence[ToolSpec]) -> bool:
        if self._session_id is None:
            return False
        if self._requested_rollover_reason:
            return True
        applied = self._protocol_descriptor
        desired = self._agent_protocol(self._agent_config(tools))
        return self._protocol_compatibility(applied, desired) == "rollover"

    def _lifecycle_call(self, function: Callable, *arguments: Any) -> Any:
        writer = self._lifecycle_writer.get()
        return function(*arguments) if writer is None else writer(function, arguments)

    def _agent_protocol(self, agent: dict) -> dict[str, Any]:
        return {
            "version": 1,
            "instructions": agent.get("instructions"),
            "tools": [{key: tool.get(key) for key in
                       ("type", "name", "description", "parameters")}
                      for tool in agent.get("tools", []) if isinstance(tool, dict)],
            "saved_agent_id": self.agent_id,
            "environment": {"type": "none"},
            "security_policy_revision": 1,
        }

    @staticmethod
    def _protocol_compatibility(applied: dict[str, Any] | None,
                                desired: dict[str, Any]) -> str:
        if applied is None:
            return "unchanged"
        if any(applied.get(key) != desired.get(key) for key in (
                "version", "instructions", "saved_agent_id", "environment",
                "security_policy_revision")):
            return "rollover"
        old_tools = {tool.get("name"): tool for tool in applied.get("tools", [])}
        new_tools = {tool.get("name"): tool for tool in desired.get("tools", [])}
        # A local revocation can safely preserve the old advertised contract:
        # ToolRegistry returns a deterministic unavailable response if it is called.
        if set(new_tools) < set(old_tools) and all(
                old_tools[name] == tool for name, tool in new_tools.items()):
            return "compatible_revocation"
        return "unchanged" if old_tools == new_tools else "rollover"

    @staticmethod
    def _agent_config_matches(remote: object, expected: dict) -> bool:
        if not isinstance(remote, dict):
            return False
        remote_tools = [{key: tool.get(key) for key in (
            "type", "name", "description", "parameters")}
            for tool in remote.get("tools", []) if isinstance(tool, dict)]
        comparable = {
            "model": remote.get("model"),
            "instructions": remote.get("instructions"),
            "tools": remote_tools,
        }
        return comparable == {key: expected.get(key) for key in comparable}

    def _wait_for_turn(self, session_id: str, expected_turn_id: str | None) -> ModelTurn:
        if not expected_turn_id:
            raise RuntimeError("Agents turn correlation requires an exact turn id")
        # time.monotonic is imported lazily to keep the adapter's dependencies small.
        import time
        expires = time.monotonic() + self.timeout_seconds
        while time.monotonic() < expires:
            session = self._request("GET", f"/agents/sessions/{session_id}")
            status = session.get("status")
            if self._session_usability(session) == "terminal":
                raise RuntimeError(f"OpenAI Agents session failed: {session.get('error') or 'no details'}")
            if status == "requires_action":
                turn = self._required_actions_turn(session)
                if turn.response_id == expected_turn_id and turn.tool_calls:
                    return turn
            turn = self._request(
                "GET", f"/agents/sessions/{session_id}/turns/{expected_turn_id}")
            turn_status = turn.get("status")
            if turn_status == "failed":
                raise RuntimeError(
                    f"OpenAI Agents turn failed: {turn.get('error') or 'no details'}")
            if turn_status == "cancelled":
                raise RuntimeError("OpenAI Agents turn was cancelled")
            if turn_status == "completed":
                return self._completed_turn(session_id, session, turn)
            time.sleep(self.poll_seconds)
        raise TimeoutError("Timed out waiting for the OpenAI Agents session")

    def _completed_turn(self, session_id: str, session: dict, turn: dict, *,
                        streamed_message: object = _STREAM_MESSAGE_MISSING) -> ModelTurn:
        turn_id = turn.get("id")
        if not turn_id:
            raise RuntimeError("Completed Agents turn has no id")
        message = (self._turn_message(session_id, turn_id)
                   if streamed_message is _STREAM_MESSAGE_MISSING
                   else streamed_message)
        submitted = self._submitted_call_ids.pop(turn_id, set())
        for call_id in submitted:
            self._ephemeral_tool_results.pop(call_id, None)
        if self._active_turn_id == turn_id:
            self._active_turn_id = None
        self._last_turn_id = turn_id
        self._persist_binding(session_id, self.agent_id, turn_id)
        usage = turn.get("usage") or session.get("usage") or {}
        return ModelTurn(turn_id, message=message if isinstance(message, str) else None,
                         input_tokens=usage.get("input_tokens"),
                         output_tokens=usage.get("output_tokens"))

    def _required_actions_turn(self, session: dict) -> ModelTurn:
        actions = [action for action in session.get("required_actions", [])
                   if action.get("type") == "function_call"]
        turn_ids = {action.get("turn_id") for action in actions}
        if len(turn_ids) != 1 or None in turn_ids:
            raise RuntimeError("Agents session returned function actions without one turn id")
        turn_id = next(iter(turn_ids))
        self._active_turn_id = turn_id
        submitted = self._submitted_call_ids.get(turn_id, set())
        calls = tuple(ToolCall(
            action.get("call_id", ""), action.get("name", ""),
            self._arguments(action.get("arguments")),
        ) for action in actions if action.get("call_id", "") not in submitted)
        usage = session.get("usage") or {}
        return ModelTurn(turn_id, tool_calls=calls,
                         input_tokens=usage.get("input_tokens"),
                         output_tokens=usage.get("output_tokens"))

    def _latest_turn(self, session_id: str) -> dict | None:
        page = self._request("GET", f"/agents/sessions/{session_id}/turns?order=desc&limit=1")
        data = page.get("data") or []
        return data[0] if data else None

    def _wait_for_correlated_turn(self, session_id: str, correlation: str,
                                  wake_key: str) -> str:
        import time
        expires = time.monotonic() + self.timeout_seconds
        while time.monotonic() < expires:
            session = self._request("GET", f"/agents/sessions/{session_id}")
            if session.get("status") == "failed":
                raise RuntimeError(
                    f"OpenAI Agents session failed: {session.get('error') or 'no details'}")
            turn_id = self._correlated_turn_id(session_id, correlation, wake_key)
            if turn_id:
                return turn_id
            time.sleep(self.poll_seconds)
        raise TimeoutError("Timed out waiting for the submitted Agents wake")

    def _correlated_turn_id(self, session_id: str, correlation: str,
                            wake_key: str) -> str | None:
        after: str | None = None
        while True:
            path = f"/agents/sessions/{session_id}/items?order=desc&limit=100"
            if after:
                from urllib.parse import quote
                path += f"&after={quote(after, safe='')}"
            page = self._request("GET", path)
            data = page.get("data") or []
            for item in data:
                if self._item_matches_wake(item, correlation, wake_key):
                    turn_id = item.get("turn_id")
                    if turn_id:
                        return turn_id
            if not page.get("has_more") or not data:
                return None
            after = page.get("last_id") or data[-1].get("id")
            if not after:
                raise RuntimeError("Agents item page has_more without a pagination cursor")

    @classmethod
    def _item_matches_wake(cls, item: dict, correlation: str, wake_key: str) -> bool:
        if item.get("type") != "message" or item.get("role") != "user":
            return False
        for part in item.get("content") or []:
            if not isinstance(part, dict) or part.get("type") != "input_text":
                continue
            text = part.get("text")
            if not isinstance(text, str) or not text:
                continue
            try:
                document = json.loads(text)
            except json.JSONDecodeError:
                document = None
            if ((isinstance(document, dict)
                 and document.get("resident_wake_correlation") == correlation)
                    or cls._wake_idempotency_key(text) == wake_key):
                return True
        return False

    def _turn_message(self, session_id: str, turn_id: str) -> str | None:
        # Session items are cursor-paginated. Read newest-first so a long-lived
        # session reaches the just-completed turn immediately, then continue
        # until all of that turn's items have been consumed.
        after: str | None = None
        found_turn = False
        messages: list[list[str]] = []
        while True:
            path = f"/agents/sessions/{session_id}/items?order=desc&limit=100"
            if after:
                from urllib.parse import quote
                path += f"&after={quote(after, safe='')}"
            page = self._request("GET", path)
            data = page.get("data") or []
            for item in data:
                if item.get("turn_id") != turn_id:
                    if found_turn:
                        return self._join_turn_messages(messages)
                    continue
                found_turn = True
                if item.get("type") != "message" or item.get("role") != "assistant":
                    continue
                messages.append([
                    part["text"] for part in item.get("content") or []
                    if part.get("type") == "output_text" and part.get("text")
                ])
            if not page.get("has_more") or not data:
                return self._join_turn_messages(messages)
            after = page.get("last_id") or data[-1].get("id")
            if not after:
                raise RuntimeError("Agents item page has_more without a pagination cursor")

    @staticmethod
    def _join_turn_messages(messages_descending: list[list[str]]) -> str | None:
        texts = [text for message in reversed(messages_descending) for text in message]
        return "\n".join(texts) or None

    def _submit_events(self, session_id: str, events: list[dict], idempotency_key: str) -> None:
        self._request(
            "POST", f"/agents/sessions/{session_id}/events", {"events": events},
            allow_empty=True, extra_headers={"Idempotency-Key": idempotency_key})

    @contextmanager
    def _open_event_stream(self, session_id: str) -> Iterator[Iterator[dict]]:
        """Open and parse the live session SSE stream without assuming replay."""
        request = urllib.request.Request(
            f"{self.base_url}/agents/sessions/{session_id}/events", method="GET",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "text/event-stream",
                "OpenAI-Beta": "agents=v1",
            })
        started = time.monotonic()
        outcome = "ok"
        event_count = 0
        response = None
        try:
            response = urllib.request.urlopen(request, timeout=self.timeout_seconds)

            def events() -> Iterator[dict]:
                nonlocal event_count
                data_lines: list[str] = []
                expires = started + self.timeout_seconds
                for raw_line in response:
                    if time.monotonic() >= expires:
                        raise TimeoutError("Timed out waiting for the OpenAI Agents stream")
                    line = raw_line.decode("utf-8").rstrip("\r\n")
                    if line == "":
                        if not data_lines:
                            continue
                        payload = "\n".join(data_lines)
                        data_lines.clear()
                        if payload == "[DONE]":
                            return
                        event = json.loads(payload)
                        if not isinstance(event, dict):
                            raise ValueError("Agents stream data is not an object")
                        event_count += 1
                        yield event
                    elif line.startswith("data:"):
                        data_lines.append(line[5:].lstrip(" "))
                    elif line.startswith(":") or line.startswith("event:") or line.startswith("id:"):
                        continue
                if data_lines:
                    payload = "\n".join(data_lines)
                    if payload != "[DONE]":
                        event = json.loads(payload)
                        if not isinstance(event, dict):
                            raise ValueError("Agents stream data is not an object")
                        event_count += 1
                        yield event

            yield events()
        except urllib.error.HTTPError as exc:
            outcome = "error"
            detail = exc.read().decode(errors="replace")[:2000]
            raise RuntimeError(
                f"OpenAI Agents stream returned HTTP {exc.code}: {detail}") from exc
        except BaseException:
            outcome = "error"
            raise
        finally:
            if response is not None:
                response.close()
            self._trace_span(
                "openai.agents_stream", started, outcome,
                event_count=event_count,
                request_timeout_seconds=self.timeout_seconds)

    def _trace_span(self, operation: str, started: float, outcome: str, **details: Any) -> None:
        trace = _agents_http_trace.get()
        if trace is not None:
            finished = time.monotonic()
            trace["events"].append({
                "timeline_operation": operation,
                "outcome": outcome,
                "duration_seconds": finished - started,
                "started_monotonic_seconds": started,
                "finished_monotonic_seconds": finished,
                **details,
            })

    def _trace_instant(self, operation: str, **details: Any) -> None:
        now = time.monotonic()
        self._trace_span(operation, now, "ok", **details)

    @staticmethod
    def _arguments(value: object) -> dict:
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value or "{}")
            except json.JSONDecodeError as exc:
                return {"_invalid_json": str(exc)}
            return parsed if isinstance(parsed, dict) else {"_invalid_json": "Arguments are not an object"}
        return {"_invalid_json": "Arguments are not an object"}

    @staticmethod
    def _wake_idempotency_key(context: str) -> str:
        try:
            wake = json.loads(context)["wake_event"]
            payload = wake.get("payload") or {}
            return str(payload.get("message_id") or payload.get("schedule_id") or wake["id"])
        except (KeyError, TypeError, json.JSONDecodeError):
            import hashlib
            return hashlib.sha256(context.encode()).hexdigest()

    @staticmethod
    def _is_owner_wake(context: str) -> bool:
        try:
            return json.loads(context)["wake_event"]["source"] == "owner"
        except (KeyError, TypeError, json.JSONDecodeError):
            return False

    @staticmethod
    def _correlated_context(context: str, wake_key: str) -> tuple[str, str]:
        import hashlib
        correlation = hashlib.sha256(wake_key.encode()).hexdigest()
        try:
            document = json.loads(context)
        except json.JSONDecodeError:
            document = {"wake_context": context}
        if not isinstance(document, dict):
            document = {"wake_context": document}
        document["resident_wake_correlation"] = correlation
        return json.dumps(document, separators=(",", ":")), correlation

    @staticmethod
    def _tool_result_event(result: ToolResult, turn_id: str) -> dict:
        metadata = json.dumps(result.output, separators=(",", ":"))
        if result.attachments:
            output: str | list[dict] = [{"type": "input_text", "text": metadata}]
            output.extend({
                "type": "input_image",
                "image_url": f"data:{attachment.mime_type};base64,"
                             f"{base64.b64encode(attachment.data).decode('ascii')}",
            } for attachment in result.attachments)
        else:
            output = metadata
        success = result.output.get("ok") is not False
        event = {
            "type": "agent.session.input.tool_result", "turn_id": turn_id,
            "call_id": result.call_id, "success": success,
        }
        if success:
            event["output"] = output
        else:
            event["error"] = str(result.output.get("error") or metadata)
        return event

    def _request(self, method: str, path: str, body: dict | None = None, *,
                 allow_empty: bool = False,
                 extra_headers: dict[str, str] | None = None) -> dict:
        data = None if body is None else json.dumps(body).encode()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "OpenAI-Beta": "agents=v1",
        }
        if extra_headers:
            headers.update(extra_headers)
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=data, method=method,
            headers=headers)
        started = time.monotonic()
        outcome = "ok"
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = response.read()
                if not payload and allow_empty:
                    return {}
                return json.loads(payload)
        except urllib.error.HTTPError as exc:
            outcome = "error"
            detail = exc.read().decode(errors="replace")[:2000]
            raise RuntimeError(f"OpenAI Agents API returned HTTP {exc.code}: {detail}") from exc
        except BaseException:
            outcome = "error"
            raise
        finally:
            trace = _agents_http_trace.get()
            if trace is not None:
                finished = time.monotonic()
                trace["events"].append({
                    "request": self._request_operation(method, path, body),
                    "outcome": outcome,
                    "duration_seconds": finished - started,
                    "started_monotonic_seconds": started,
                    "finished_monotonic_seconds": finished,
                    "request_timeout_seconds": self.timeout_seconds,
                })

    @staticmethod
    def _request_operation(method: str, path: str, body: dict | None) -> str:
        """Classify an Agents request without retaining its URL or content."""
        resource = path.split("?", 1)[0].rstrip("/")
        if method == "POST" and resource == "/agents/sessions":
            return "create_session"
        if resource.endswith("/events"):
            event_types = {
                event.get("type") for event in (body or {}).get("events", [])
                if isinstance(event, dict)
            }
            if event_types and event_types <= {"agent.session.input.tool_result"}:
                return "submit_tool_results"
            return "submit_wake"
        if "/items" in resource:
            return "retrieve_items"
        if "/turns/" in resource:
            return "poll_turn"
        if resource.endswith("/turns"):
            return "reconcile_turns"
        if method == "GET":
            return "poll_session"
        if method == "PATCH":
            return "update_session"
        return "session_request"
