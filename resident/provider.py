from __future__ import annotations

import asyncio
import base64
import json
import urllib.error
import urllib.request
from concurrent.futures import Future
from contextvars import ContextVar
from typing import Callable, Protocol, Sequence

from .domain import ModelTurn, ToolCall, ToolResult, ToolSpec


class ModelProvider(Protocol):
    async def respond(self, context: str, tools: Sequence[ToolSpec], results: Sequence[ToolResult],
                      previous_response_id: str | None = None) -> ModelTurn: ...

    def discard_continuation(self, continuation_id: str) -> None: ...


RESIDENT_AGENT_INSTRUCTIONS = (
    "Act as the persistent Resident described by each supplied wake context. Use tools for durable state, "
    "local capabilities, communication, and scheduling. Send all intentional communication to the owner, "
    "including replies to owner-initiated wakes, with send_owner_message. A final response message is "
    "wake-result diagnostic text only and is never delivered to the owner. Do not expose private chain-of-thought. "
    "Choose memories selectively, preserve uncertainty, and treat supplied owner_guidance as durable guidance where "
    "it applies. Local events are factual observations, not hard-coded instructions to act."
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
                 timeout_seconds: float = 120.0):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for the OpenAI provider")
        self.api_key, self.model, self.base_url = api_key, model, base_url.rstrip("/")
        # Keep the operator-supplied identity separate from the identity learned
        # from the durable session binding. The former always wins; the latter
        # supplies continuity when no current override is configured.
        self.agent_id = agent_id
        self._bound_agent_id: str | None = None
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds
        self._session_id: str | None = None
        self._last_turn_id: str | None = None
        self._tool_fingerprint: str | None = None
        self._active_turn_id: str | None = None
        self._submitted_call_ids: dict[str, set[str]] = {}
        self._pending_wakes: dict[str, tuple[str, str, str]] = {}
        self._ephemeral_tool_results: dict[str, ToolResult] = {}
        self._save_binding: Callable[[str, str | None, str | None], None] = lambda *_: None
        self._binding_writer: ContextVar[
            Callable[[str, str | None, str | None], None] | None
        ] = ContextVar("agents_binding_writer", default=None)
        self._pending_binding: tuple[str, str | None, str | None] | None = None
        self._begin_action: Callable[..., dict] | None = None
        self._complete_action: Callable[..., None] | None = None

    def bind_session_store(self, load: Callable[[], dict | None],
                           save: Callable[[str, str | None, str | None], None]) -> None:
        self._save_binding = save
        binding = load()
        if binding is not None:
            persisted_agent_id = binding.get("agent_id")
            if self.agent_id is None or self.agent_id == persisted_agent_id:
                self._session_id = binding["session_id"]
                self._last_turn_id = binding.get("last_turn_id")
                self._bound_agent_id = persisted_agent_id

    def bind_action_store(self, begin: Callable[..., dict], complete: Callable[..., None]) -> None:
        self._begin_action, self._complete_action = begin, complete

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
        try:
            return await asyncio.to_thread(
                self._respond_sync, context, tools, results, previous_response_id)
        finally:
            self._binding_writer.reset(token)

    def discard_continuation(self, continuation_id: str) -> None:
        # Managed sessions retain turn state. A failed local wake is reconciled
        # from the session on the next attempt rather than erased locally.
        return None

    def _respond_sync(self, context: str, tools: Sequence[ToolSpec], results: Sequence[ToolResult],
                      previous_response_id: str | None) -> ModelTurn:
        session = self._ensure_session(tools)
        session_id = session["id"]
        if results:
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
        session = self._ensure_session(tools)
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

    def _ensure_session(self, tools: Sequence[ToolSpec]) -> dict:
        self._flush_pending_binding()
        agent = self._agent_config(tools)
        fingerprint = json.dumps(agent, sort_keys=True, separators=(",", ":"))
        if self._session_id is not None:
            try:
                session = self._request("GET", f"/agents/sessions/{self._session_id}")
            except RuntimeError as exc:
                if "HTTP 404" not in str(exc):
                    raise
                self._session_id = None
                self._last_turn_id = None
            else:
                remote_agent_id = (session.get("agent") or {}).get("id")
                if (self.agent_id is not None and remote_agent_id is not None
                        and remote_agent_id != self.agent_id):
                    # A current operator override must not inherit a session
                    # attached to a different saved Agent resource.
                    self._session_id = None
                    self._last_turn_id = None
                else:
                    resolved_agent_id = self.agent_id or remote_agent_id or self._bound_agent_id
                    if resolved_agent_id != self._bound_agent_id:
                        self._bound_agent_id = resolved_agent_id
                        self._persist_binding(
                            self._session_id, resolved_agent_id, self._last_turn_id)
                    remote_agent = session.get("agent")
                    if self._tool_fingerprint is None:
                        if (not isinstance(remote_agent, dict)
                                or not {"model", "instructions", "tools"} <= remote_agent.keys()
                                or self._agent_config_matches(remote_agent, agent)):
                            # A restored provider has no in-memory fingerprint.
                            # Recognize a matching persisted session from its
                            # returned config. A partial legacy response cannot
                            # prove a mismatch, so avoid replacing it eagerly.
                            self._tool_fingerprint = fingerprint
                    if self._tool_fingerprint != fingerprint and session.get("status") == "idle":
                        # Session updates only accept model, reasoning, and
                        # service-tier overrides. Instructions and tools are part
                        # of session creation, so replace an idle session when the
                        # Resident configuration actually changes.
                        self._session_id = None
                        self._last_turn_id = None
                    else:
                        # If the session is active, leave the mismatch pending and
                        # retry once it becomes idle.
                        return session
        body: dict = {
            "environment": {"type": "none"},
            "agent": agent,
            "metadata": {"managed_by": "resident"},
        }
        intended_agent_id = self.agent_id or self._bound_agent_id
        if intended_agent_id:
            body["agent_id"] = intended_agent_id
        session = self._request("POST", "/agents/sessions", body)
        self._session_id = session["id"]
        resolved_agent_id = (
            self.agent_id or (session.get("agent") or {}).get("id") or self._bound_agent_id)
        self._bound_agent_id = resolved_agent_id
        self._last_turn_id = None
        self._tool_fingerprint = fingerprint
        self._persist_binding(self._session_id, resolved_agent_id, None)
        return session

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
        return {
            "model": self.model,
            "instructions": RESIDENT_AGENT_INSTRUCTIONS,
            "tools": [{
                "type": "function", "name": tool.name,
                "description": tool.description, "parameters": tool.input_schema,
            } for tool in tools],
        }

    @staticmethod
    def _agent_config_matches(remote: object, expected: dict) -> bool:
        if not isinstance(remote, dict):
            return False
        remote_tools = [{key: tool.get(key) for key in (
            "type", "name", "description", "parameters")}
            for tool in remote.get("tools", []) if isinstance(tool, dict)]
        return {
            "model": remote.get("model"),
            "instructions": remote.get("instructions"),
            "tools": remote_tools,
        } == expected

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
        agent_id = self.agent_id or (session.get("agent") or {}).get("id") or self._bound_agent_id
        self._bound_agent_id = agent_id
        self._persist_binding(session_id, agent_id, turn_id)
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
        self._request("POST", f"/agents/sessions/{session_id}/events", {
            "events": events, "idempotency_key": idempotency_key,
        }, allow_empty=True)

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
                 allow_empty: bool = False) -> dict:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=data, method=method,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "OpenAI-Beta": "agents=v1",
            })
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = response.read()
                if not payload and allow_empty:
                    return {}
                return json.loads(payload)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:2000]
            raise RuntimeError(f"OpenAI Agents API returned HTTP {exc.code}: {detail}") from exc
