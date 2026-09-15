from __future__ import annotations

import asyncio
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

    async def respond(self, context: str, tools: Sequence[ToolSpec], results: Sequence[ToolResult],
                      previous_response_id: str | None = None) -> ModelTurn:
        if previous_response_id:
            input_data = [{"type": "function_call_output", "call_id": r.call_id,
                           "output": json.dumps(r.output, separators=(",", ":"))} for r in results]
        else:
            input_data = context
        body: dict = {
            "model": self.model,
            "instructions": (
                "Act as the persistent Resident described by the supplied context. Use tools for durable state, "
                "capabilities, communication, and scheduling. Do not expose private chain-of-thought. "
                "When useful, provide concise observable rationale in tool arguments or your final message."
            ),
            "input": input_data,
            "tools": [{"type": "function", "name": t.name, "description": t.description,
                       "parameters": t.input_schema} for t in tools],
        }
        if previous_response_id:
            body["previous_response_id"] = previous_response_id
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
        return ModelTurn(raw.get("id"), "\n".join(texts) or None, tuple(calls),
                         usage.get("input_tokens"), usage.get("output_tokens"))

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
