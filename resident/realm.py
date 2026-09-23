"""Instance-bound client for Realm's v1 game API."""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any
from urllib.parse import quote, urlsplit

import aiohttp

from .capabilities import Capability, current_invocation_id


_LIMIT = 1024 * 1024
_ID = {"type": "string", "minLength": 1, "maxLength": 200}
_OBJECT = {"type": "object", "additionalProperties": True}


def _schema(required: tuple[str, ...] = (), **properties: Any) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(required),
            "additionalProperties": False}


class RealmClient:
    def __init__(self, base_url: str, game_id: str, actor_id: str,
                 timeout_seconds: float = 10.0):
        parsed = urlsplit(base_url)
        if (parsed.scheme not in ("http", "https") or not parsed.netloc or
                parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("Realm base URL must be an absolute HTTP(S) URL without credentials")
        if not game_id or not actor_id or len(game_id) > 200 or len(actor_id) > 200:
            raise ValueError("Realm game and actor IDs must contain 1 to 200 characters")
        self.base_url = base_url.rstrip("/")
        self.game_id = game_id
        self.actor_id = actor_id
        self.timeout_seconds = timeout_seconds
        self.path = f"/games/{quote(game_id, safe='')}"

    async def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = None if body is None else json.dumps(body, allow_nan=False, separators=(",", ":")).encode()
        if payload is not None and len(payload) > 256 * 1024:
            raise ValueError("Realm request is too large")
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout, auto_decompress=False) as session:
            async with session.request(method, self.base_url + path, data=payload,
                                       headers={"Accept": "application/json", "Content-Type": "application/json"},
                                       allow_redirects=False) as response:
                if response.status >= 300:
                    raise RealmHTTPError(response.status)
                raw = await response.content.read(_LIMIT + 1)
                if len(raw) > _LIMIT:
                    raise ValueError("Realm response is too large")
                result = json.loads(raw)
                if not isinstance(result, dict):
                    raise ValueError("Realm response must be an object")
                return result

    async def snapshot(self) -> dict[str, Any]:
        player = await self._request("GET", f"{self.path}/state?actor_id={quote(self.actor_id, safe='')}")
        trusted = await self._request("GET", f"{self.path}/authoritative-state")
        game = trusted.get("game")
        entities = trusted.get("entities")
        if (not isinstance(game, dict) or not isinstance(game.get("current_revision"), int)
                or not isinstance(entities, list) or not any(
                    isinstance(item, dict) and item.get("id") == self.actor_id
                    and item.get("kind") == "creature" for item in entities)):
            raise ValueError("Realm game or configured creature actor is unavailable")
        return {"player_state": player, "trusted_state": trusted}

    @staticmethod
    def _error(exc: Exception, *, mutating: bool = False) -> dict[str, Any]:
        if isinstance(exc, RealmHTTPError):
            code = "conflict" if exc.status == 409 else "rejected" if 400 <= exc.status < 500 else "unknown_outcome" if mutating else "unavailable"
        elif mutating:
            code = "unknown_outcome"
        else:
            code = "unavailable"
        result: dict[str, Any] = {"ok": False, "error_code": code,
                                  "error": f"Realm {code.replace('_', ' ')}; reread Realm before continuing"}
        if code == "unknown_outcome":
            result["outcome"] = "unknown"
        return result

    async def read(self, _: dict[str, Any]) -> dict[str, Any]:
        try:
            return await self.snapshot()
        except (aiohttp.ClientError, TimeoutError, OSError, ValueError, RealmHTTPError) as exc:
            return self._error(exc)

    async def mutate(self, route: str, arguments: dict[str, Any]) -> dict[str, Any]:
        key = current_invocation_id() or str(uuid.uuid4())
        try:
            before = await self.snapshot()
            revision = before["trusted_state"]["game"]["current_revision"]
            body = {**arguments, "expected_revision": revision, "idempotency_key": key}
            if route in ("reveal-fact", "observe-entity"):
                body["actor_id"] = self.actor_id
            path = f"{self.path}/world-patches" if route == "world-patch" else f"{self.path}/operations/{route}"
            result = await self._request("POST", path, body)
        except (aiohttp.ClientError, TimeoutError, OSError, ValueError, RealmHTTPError) as exc:
            failure = self._error(exc, mutating="body" in locals())
            failure["idempotency_key"] = key
            return failure
        try:
            after = await self.snapshot()
        except (aiohttp.ClientError, TimeoutError, OSError, ValueError, RealmHTTPError) as exc:
            return {"mutation": result, "idempotency_key": key,
                    "state_refresh": self._error(exc)}
        return {"mutation": result, "idempotency_key": key, **after}

    @property
    def capabilities(self) -> list[Capability]:
        specs = [
            ("realm_read", "Read the current actor projection and trusted canonical game state.",
             _schema(), None),
            ("realm_world_patch", "Atomically materialize missing canonical world details. Realm validates the patch.",
             _schema(entities={"type": "array", "items": _OBJECT},
                     entity_updates={"type": "array", "items": _OBJECT},
                     containment={"type": "array", "items": _OBJECT},
                     connections={"type": "array", "items": _OBJECT},
                     facts={"type": "array", "items": _OBJECT},
                     knowledge={"type": "array", "items": _OBJECT},
                     observations={"type": "array", "items": _OBJECT}), "world-patch"),
            ("realm_reveal_fact", "Reveal an existing fact to the configured player actor.",
             _schema(("fact_id",), fact_id=_ID), "reveal-fact"),
            ("realm_observe_entity", "Make an existing entity visible to the configured player actor.",
             _schema(("entity_id",), entity_id=_ID), "observe-entity"),
            ("realm_move", "Move an entity to an existing destination.",
             _schema(("entity_id", "destination_id"), entity_id=_ID, destination_id=_ID), "move"),
            ("realm_establish_fact", "Add a canonical fact. Reveal it separately when the player learns it.",
             _schema(("text",), id=_ID, text={"type": "string", "minLength": 1},
                     subject_entity_id=_ID, metadata=_OBJECT), "establish-fact"),
            ("realm_advance_time", "Advance canonical world time by positive minutes.",
             _schema(("minutes",), minutes={"type": "integer", "minimum": 1}), "advance-time"),
        ]
        return [Capability("realm", "Authoritative persistent Realm game", name, description,
                           schema, self.read if route is None else
                           (lambda args, route=route: self.mutate(route, args)))
                for name, description, schema, route in specs]


class RealmHTTPError(Exception):
    def __init__(self, status: int):
        self.status = status
