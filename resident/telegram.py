from __future__ import annotations

import asyncio
import json
from typing import Any, Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .domain import WakeEvent


class TelegramTransportError(RuntimeError):
    """A credential-safe Telegram transport failure."""


class TelegramTransport:
    """A text-only Telegram Owner transport using Bot API long polling."""

    _MAX_TEXT_LENGTH = 4096

    def __init__(self, bot_token: str, owner_user_id: int, owner_chat_id: int, *,
                 poll_seconds: float = 30.0, request_timeout_seconds: float = 40.0,
                 diagnostic_output: Callable[[str], None] | None = None,
                 load_offset: Callable[[], int | None] | None = None,
                 save_offset: Callable[[int], None] | None = None):
        self._bot_token = bot_token
        self._owner_user_id = owner_user_id
        self._owner_chat_id = owner_chat_id
        self.poll_seconds = poll_seconds
        self.request_timeout_seconds = request_timeout_seconds
        self.diagnostic_output = diagnostic_output or (lambda _: None)
        self._load_offset = load_offset or (lambda: None)
        self._save_offset = save_offset or (lambda _: None)
        self._owner_message_event: Callable[[str], WakeEvent] | None = None

    def __repr__(self) -> str:
        return "TelegramTransport(configured=True)"

    def bind_owner_message(self, factory: Callable[[str], WakeEvent]) -> None:
        self._owner_message_event = factory

    def bind_offset_checkpoint(self, load: Callable[[], int | None], save: Callable[[int], None]) -> None:
        self._load_offset = load
        self._save_offset = save

    def _post_json(self, method: str, parameters: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            f"https://api.telegram.org/bot{self._bot_token}/{method}",
            data=urlencode(parameters).encode("utf-8"),
            headers={"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.request_timeout_seconds) as response:
                payload = json.load(response)
        except Exception:
            raise TelegramTransportError(f"Telegram {method} request failed") from None
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            raise TelegramTransportError(f"Telegram {method} request was rejected")
        return payload

    async def _request(self, method: str, parameters: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._post_json, method, parameters)

    async def send_text(self, content: str) -> None:
        for start in range(0, len(content), self._MAX_TEXT_LENGTH):
            await self._request("sendMessage", {
                "chat_id": self._owner_chat_id,
                "text": content[start:start + self._MAX_TEXT_LENGTH],
            })

    async def _check_webhook(self) -> None:
        payload = await self._request("getWebhookInfo", {})
        result = payload.get("result")
        if not isinstance(result, dict):
            raise TelegramTransportError("Telegram getWebhookInfo returned an invalid response")
        if result.get("url"):
            raise TelegramTransportError("Telegram long polling is unavailable while a webhook is configured")

    async def poll_once(self, queue: asyncio.Queue[WakeEvent], offset: int | None) -> int | None:
        parameters: dict[str, Any] = {
            "timeout": int(self.poll_seconds),
            "allowed_updates": json.dumps(["message"]),
        }
        if offset is not None:
            parameters["offset"] = offset
        payload = await self._request("getUpdates", parameters)
        updates = payload.get("result")
        if not isinstance(updates, list):
            raise TelegramTransportError("Telegram getUpdates returned an invalid response")
        for update in updates:
            if not isinstance(update, dict) or not isinstance(update.get("update_id"), int):
                continue
            message = update.get("message")
            if isinstance(message, dict):
                chat, sender, text = message.get("chat"), message.get("from"), message.get("text")
                authorized = (
                    isinstance(chat, dict) and chat.get("type") == "private"
                    and chat.get("id") == self._owner_chat_id
                    and isinstance(sender, dict) and sender.get("id") == self._owner_user_id
                    and isinstance(text, str) and bool(text.strip())
                )
                if authorized:
                    if self._owner_message_event is None:
                        raise RuntimeError("Telegram Owner message handler is not bound")
                    await queue.put(self._owner_message_event(text))
            offset = update["update_id"] + 1
            self._save_offset(offset)
        return offset

    async def run(self, queue: asyncio.Queue[WakeEvent], stop: asyncio.Event) -> None:
        offset = self._load_offset()
        webhook_checked = False
        backoff = 1.0
        while not stop.is_set():
            try:
                if not webhook_checked:
                    await self._check_webhook()
                    webhook_checked = True
                offset = await self.poll_once(queue, offset)
                backoff = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.diagnostic_output(f"poll failed: {type(exc).__name__}: {exc}")
                try:
                    await asyncio.wait_for(stop.wait(), timeout=backoff)
                except TimeoutError:
                    pass
                backoff = min(backoff * 2, 30.0)
