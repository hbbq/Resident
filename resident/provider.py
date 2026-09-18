from __future__ import annotations

import asyncio
import base64
import json
import urllib.error
import urllib.request
from concurrent.futures import Future
from contextvars import ContextVar
from typing import Any, Callable, Protocol, Sequence

from .domain import ModelTurn, ToolCall, ToolResult, ToolSpec
from .memory import SessionItemPage


class ModelProvider(Protocol):
    async def respond(self, context: str, tools: Sequence[ToolSpec], results: Sequence[ToolResult],
                      previous_response_id: str | None = None) -> ModelTurn: ...

    def discard_continuation(self, continuation_id: str) -> None: ...


class RemoteSessionUnavailable(RuntimeError):
    """A restored session vanished before a replacement bootstrap was prepared."""


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
        raw = await asyncio.to_thread(self._post, body)
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
        self._begin_rollover: Callable[[str | None, str, str], str] = (
            lambda _old, _reason, _requested_by: "")
        self._complete_rollover: Callable[[str, str, str], None] = lambda *_: None
        self._fail_rollover: Callable[[str, str], None] = lambda *_: None
        self._requested_rollover_reason: str | None = None
        self._unavailable_session_id: str | None = None

    def bind_session_store(self, load: Callable[[], dict | None],
                           save: Callable[[str, str | None, str | None], None]) -> None:
        self._save_binding = save
        binding = load()
        if binding is not None:
            persisted_agent_id = binding.get("agent_id")
            if self.agent_id is None or self.agent_id == persisted_agent_id:
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
                             begin_rollover: Callable[[str | None, str, str], str],
                             complete_rollover: Callable[[str, str, str], None],
                             fail_rollover: Callable[[str, str], None]) -> None:
        self._load_protocol, self._save_protocol = load_protocol, save_protocol
        self._begin_rollover = begin_rollover
        self._complete_rollover, self._fail_rollover = complete_rollover, fail_rollover
        if self._session_id is not None:
            self._protocol_descriptor = load_protocol(self._session_id)

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def session_protocol_known(self) -> bool:
        return self._session_id is None or self._protocol_descriptor is not None

    def request_rollover(self, reason: str = "explicit_new_chapter") -> None:
        if not reason.strip():
            raise ValueError("Session rollover reason must be nonempty")
        self._requested_rollover_reason = reason.strip()

    async def preflight_session(self) -> str | None:
        """Detect an unavailable restored session before wake context is built."""
        session_id = self._session_id
        if session_id is None:
            return None
        try:
            await asyncio.to_thread(
                self._request, "GET", f"/agents/sessions/{session_id}")
        except RuntimeError as exc:
            if "HTTP 404" not in str(exc):
                raise
            self._unavailable_session_id = session_id
            self._requested_rollover_reason = "remote_session_missing"
            return "remote_session_missing"
        return None

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
        page = self._request("GET", path)
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
        try:
            return await asyncio.to_thread(
                self._respond_sync, context, tools, results, previous_response_id)
        finally:
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
            self._submit_events(session_id, events, f"resident-tool:{turn_id}:{call_key}"[:256])
            self._submitted_call_ids.setdefault(turn_id, set()).update(
                result.call_id for result in results)
            turn = self._wait_for_turn(session_id, turn_id)
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
        self._submit_events(session_id, [event], f"resident-wake:{wake_key}"[:256])
        return self._wait_for_submitted_wake(session_id, correlation, wake_key)

    def _wait_for_submitted_wake(self, session_id: str, correlation: str,
                                 wake_key: str) -> ModelTurn:
        turn_id = self._wait_for_correlated_turn(session_id, correlation, wake_key)
        return self._wait_for_turn(session_id, turn_id)

    def _reconcile_before_wake(self, session_id: str, session: dict) -> ModelTurn | None:
        """Finish remote work without attributing it to the next ordinary wake."""
        status = session.get("status")
        if status == "failed":
            raise RuntimeError(
                f"OpenAI Agents session failed: {session.get('error') or 'no details'}")
        if status == "idle":
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
                return self._intentional_rollover(
                    "remote_session_missing", agent, desired_protocol,
                    initial_input, allow_create, finalization_status="unavailable")
            try:
                session = self._request("GET", f"/agents/sessions/{self._session_id}")
            except RuntimeError as exc:
                if "HTTP 404" not in str(exc):
                    raise
                self._unavailable_session_id = self._session_id
                self._requested_rollover_reason = "remote_session_missing"
                raise RemoteSessionUnavailable(str(exc)) from exc
            else:
                remote_agent_id = (session.get("agent") or {}).get("id")
                if (self.agent_id is not None and remote_agent_id is not None
                        and remote_agent_id != self.agent_id):
                    # A current operator override must not inherit a session
                    # attached to a different saved Agent resource.
                    return self._intentional_rollover(
                        "saved_agent_id_changed", agent, desired_protocol,
                        initial_input, allow_create)
                else:
                    remote_agent = session.get("agent")
                    applied_protocol = self._protocol_descriptor
                    if applied_protocol is None and isinstance(remote_agent, dict):
                        applied_protocol = self._agent_protocol(remote_agent)
                    if applied_protocol is None and self._tool_fingerprint is not None:
                        try:
                            applied_protocol = self._agent_protocol(
                                json.loads(self._tool_fingerprint))
                        except (TypeError, json.JSONDecodeError):
                            pass
                    compatibility = self._protocol_compatibility(applied_protocol, desired_protocol)
                    if self._requested_rollover_reason and session.get("status") == "idle":
                        reason = self._requested_rollover_reason
                        result = self._intentional_rollover(
                            reason, agent, desired_protocol, initial_input, allow_create)
                        self._requested_rollover_reason = None
                        return result
                    if self._tool_fingerprint is None:
                        if (not isinstance(remote_agent, dict)
                                or not {"model", "instructions", "tools"} <= remote_agent.keys()
                                or self._agent_config_matches(remote_agent, agent)):
                            # A restored provider has no in-memory fingerprint.
                            # Recognize a matching persisted session from its
                            # returned config. A partial legacy response cannot
                            # prove a mismatch, so avoid replacing it eagerly.
                            self._tool_fingerprint = fingerprint
                    if compatibility == "rollover" and session.get("status") == "idle":
                        return self._intentional_rollover(
                            "function_or_immutable_protocol_changed", agent, desired_protocol,
                            initial_input, allow_create)
                    if compatibility == "rollover":
                        # Finish/recover the active turn under its existing contract;
                        # the next idle reconciliation performs the recorded rollover.
                        return session, False
                    if compatibility == "compatible_revocation":
                        return session, False
                    mutable_patch = self._mutable_patch(remote_agent)
                    if mutable_patch:
                        if session.get("status") != "idle":
                            return session, False
                        patched = self._request(
                            "PATCH", f"/agents/sessions/{self._session_id}",
                            {"agent": mutable_patch})
                        session = {**session, **patched}
                    self._protocol_descriptor = desired_protocol
                    self._lifecycle_call(self._save_protocol, self._session_id, desired_protocol)
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
        session = self._request("POST", "/agents/sessions", body)
        self._session_id = session["id"]
        self._last_turn_id = None
        self._tool_fingerprint = fingerprint
        self._protocol_descriptor = desired_protocol
        self._persist_binding(self._session_id, self.agent_id, None)
        self._lifecycle_call(self._save_protocol, self._session_id, desired_protocol)
        return session, True

    def _intentional_rollover(self, reason: str, agent: dict,
                              descriptor: dict[str, Any], initial_input: str | None,
                              allow_create: bool, *,
                              finalization_status: str = "completed") -> tuple[dict, bool]:
        if not allow_create:
            raise RuntimeError("OpenAI Agents session requires rollover while continuing a turn")
        if initial_input is None:
            raise RuntimeError("Intentional session rollover requires bootstrap input")
        old_session_id = self._session_id
        rollover_id = self._lifecycle_call(
            self._begin_rollover, old_session_id, reason, "runtime")
        body: dict = {
            "environment": {"type": "none"}, "agent": agent, "input": initial_input,
            "metadata": {"managed_by": "resident", "rollover_reason": reason},
        }
        if self.agent_id:
            body["agent_id"] = self.agent_id
        try:
            session = self._request("POST", "/agents/sessions", body)
            new_session_id = session["id"]
            self._session_id, self._last_turn_id = new_session_id, None
            self._unavailable_session_id = None
            self._requested_rollover_reason = None
            self._protocol_descriptor = descriptor
            self._tool_fingerprint = json.dumps(agent, sort_keys=True, separators=(",", ":"))
            self._persist_binding(new_session_id, self.agent_id, None)
            self._lifecycle_call(self._save_protocol, new_session_id, descriptor)
            self._lifecycle_call(
                self._complete_rollover, rollover_id, new_session_id, finalization_status)
            return session, True
        except Exception:
            self._lifecycle_call(self._fail_rollover, rollover_id, "unavailable")
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
        if self.reasoning_effort:
            agent["reasoning"] = {"effort": self.reasoning_effort}
        if self.service_tier:
            agent["service_tier"] = self.service_tier
        return agent

    def _mutable_patch(self, remote_agent: object) -> dict[str, Any]:
        if not isinstance(remote_agent, dict):
            return {}
        desired = self._agent_config(())
        patch: dict[str, Any] = {}
        for key in ("model", "reasoning", "service_tier"):
            # A partial legacy session response cannot prove mutable drift.
            # The session update contract clears reasoning/service tier with
            # JSON null, so absence from the desired config remains meaningful.
            if key in remote_agent and remote_agent.get(key) != desired.get(key):
                patch[key] = desired.get(key)
        return patch

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
            if status == "failed":
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

    def _completed_turn(self, session_id: str, session: dict, turn: dict) -> ModelTurn:
        turn_id = turn.get("id")
        if not turn_id:
            raise RuntimeError("Completed Agents turn has no id")
        message = self._turn_message(session_id, turn_id)
        submitted = self._submitted_call_ids.pop(turn_id, set())
        for call_id in submitted:
            self._ephemeral_tool_results.pop(call_id, None)
        if self._active_turn_id == turn_id:
            self._active_turn_id = None
        self._last_turn_id = turn_id
        self._persist_binding(session_id, self.agent_id, turn_id)
        usage = turn.get("usage") or session.get("usage") or {}
        return ModelTurn(turn_id, message=message,
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
                if item.get("type") != "message" or item.get("role") != "user":
                    continue
                for part in item.get("content") or []:
                    if part.get("type") != "input_text" or not part.get("text"):
                        continue
                    text = part["text"]
                    try:
                        document = json.loads(text)
                    except (TypeError, json.JSONDecodeError):
                        document = None
                    matches_marker = (
                        isinstance(document, dict) and
                        document.get("resident_wake_correlation") == correlation)
                    if matches_marker or self._wake_idempotency_key(text) == wake_key:
                        turn_id = item.get("turn_id")
                        if turn_id:
                            return turn_id
            if not page.get("has_more") or not data:
                return None
            after = page.get("last_id") or data[-1].get("id")
            if not after:
                raise RuntimeError("Agents item page has_more without a pagination cursor")

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
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = response.read()
                if not payload and allow_empty:
                    return {}
                return json.loads(payload)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:2000]
            raise RuntimeError(f"OpenAI Agents API returned HTTP {exc.code}: {detail}") from exc
