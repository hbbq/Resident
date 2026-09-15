from __future__ import annotations

import asyncio
import base64
import json
import urllib.error
import urllib.request
from typing import Protocol, Sequence

from .domain import ModelTurn, ToolCall, ToolResult, ToolSpec


class ModelProvider(Protocol):
    async def respond(self, context: str, tools: Sequence[ToolSpec], results: Sequence[ToolResult],
                      previous_response_id: str | None = None) -> ModelTurn: ...


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
            "instructions": (
                "Act as the persistent Resident described by the supplied context. Use tools for durable state, "
                "capabilities, communication, and scheduling. Send all intentional communication to the owner, "
                "including replies to owner-initiated wakes, with send_owner_message. A final response message is "
                "wake-result diagnostic text only and is never delivered to the owner. Do not expose private "
                "chain-of-thought. When useful, provide concise observable rationale in tool arguments or the "
                "final wake result."
            ),
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
            self._histories[raw["id"]] = [*history, *raw.get("output", [])]
        return ModelTurn(raw.get("id"), "\n".join(texts) or None, tuple(calls),
                         usage.get("input_tokens"), usage.get("output_tokens"))

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
