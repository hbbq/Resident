from __future__ import annotations

import asyncio
import base64
import json
import urllib.error
import urllib.request
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
        self.agent_id = agent_id
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds
        self._session_id: str | None = None
        self._last_turn_id: str | None = None
        self._tool_fingerprint: str | None = None
        self._active_turn_id: str | None = None
        self._submitted_call_ids: dict[str, set[str]] = {}
        self._ephemeral_tool_results: dict[str, ToolResult] = {}
        self._save_binding: Callable[[str, str | None, str | None], None] = lambda *_: None
        self._begin_action: Callable[..., dict] | None = None
        self._complete_action: Callable[..., None] | None = None

    def bind_session_store(self, load: Callable[[], dict | None],
                           save: Callable[[str, str | None, str | None], None]) -> None:
        self._save_binding = save
        binding = load()
        if binding is not None:
            self._session_id = binding["session_id"]
            self._last_turn_id = binding.get("last_turn_id")

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
        return await asyncio.to_thread(
            self._respond_sync, context, tools, results, previous_response_id)

    def discard_continuation(self, continuation_id: str) -> None:
        # Managed sessions retain turn state. A failed local wake is reconciled
        # from the session on the next attempt rather than erased locally.
        return None

    def _respond_sync(self, context: str, tools: Sequence[ToolSpec], results: Sequence[ToolResult],
                      previous_response_id: str | None) -> ModelTurn:
        session = self._ensure_session(tools)
        session_id = session["id"]
        if not results and session.get("status") == "requires_action":
            turn = self._required_actions_turn(session)
            if turn.tool_calls:
                return turn
            return self._wait_for_turn(session_id, turn.response_id)
        if results:
            turn_id = previous_response_id
            if not turn_id:
                raise RuntimeError("Agents tool results require the requesting turn id")
            events = [self._tool_result_event(result, turn_id) for result in results]
            call_key = ":".join(sorted(result.call_id for result in results))
            self._submit_events(session_id, events, f"resident-tool:{turn_id}:{call_key}"[:256])
            self._submitted_call_ids.setdefault(turn_id, set()).update(
                result.call_id for result in results)
            expected_turn_id = turn_id
        else:
            wake_key = self._wake_idempotency_key(context)
            event = {
                "type": "agent.session.input.message",
                "input": [{"role": "user", "content": [
                    {"type": "input_text", "text": context}
                ]}],
            }
            self._submit_events(session_id, [event], f"resident-wake:{wake_key}"[:256])
            expected_turn_id = None
        return self._wait_for_turn(session_id, expected_turn_id)

    def _ensure_session(self, tools: Sequence[ToolSpec]) -> dict:
        agent = self._agent_config(tools)
        fingerprint = json.dumps(agent, sort_keys=True, separators=(",", ":"))
        if self._session_id is not None:
            try:
                session = self._request("GET", f"/agents/sessions/{self._session_id}")
            except RuntimeError as exc:
                if "HTTP 404" not in str(exc):
                    raise
                self._session_id = None
            else:
                if self._tool_fingerprint != fingerprint and session.get("status") == "idle":
                    session = self._request(
                        "POST", f"/agents/sessions/{self._session_id}", {"agent": agent})
                self._tool_fingerprint = fingerprint
                return session
        body: dict = {
            "environment": {"type": "none"},
            "agent": agent,
            "metadata": {"managed_by": "resident"},
        }
        if self.agent_id:
            body["agent_id"] = self.agent_id
        session = self._request("POST", "/agents/sessions", body)
        self._session_id = session["id"]
        resolved_agent_id = (session.get("agent") or {}).get("id") or self.agent_id
        self._save_binding(self._session_id, resolved_agent_id, None)
        self._last_turn_id = None
        self._tool_fingerprint = fingerprint
        return session

    def _agent_config(self, tools: Sequence[ToolSpec]) -> dict:
        return {
            "model": self.model,
            "name": "Resident",
            "instructions": RESIDENT_AGENT_INSTRUCTIONS,
            "tools": [{
                "type": "function", "name": tool.name,
                "description": tool.description, "parameters": tool.input_schema,
            } for tool in tools],
        }

    def _wait_for_turn(self, session_id: str, expected_turn_id: str | None) -> ModelTurn:
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
                if turn.tool_calls:
                    return turn
            if status == "idle":
                turn = self._latest_turn(session_id)
                if turn is not None:
                    turn_id = turn.get("id")
                    if ((expected_turn_id and turn_id == expected_turn_id) or
                            (expected_turn_id is None and turn_id != self._last_turn_id)):
                        message = self._turn_message(session_id, turn_id)
                        submitted = self._submitted_call_ids.pop(turn_id, set())
                        for call_id in submitted:
                            self._ephemeral_tool_results.pop(call_id, None)
                        self._last_turn_id = turn_id
                        agent_id = (session.get("agent") or {}).get("id") or self.agent_id
                        self._save_binding(session_id, agent_id, turn_id)
                        usage = turn.get("usage") or session.get("usage") or {}
                        return ModelTurn(turn_id, message=message,
                                         input_tokens=usage.get("input_tokens"),
                                         output_tokens=usage.get("output_tokens"))
            time.sleep(self.poll_seconds)
        raise TimeoutError("Timed out waiting for the OpenAI Agents session")

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
