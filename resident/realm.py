"""Instance-bound client for Realm's v1 game API."""
from __future__ import annotations

import asyncio
import json
import re
import uuid
from typing import Any, Callable
from urllib.parse import quote, urlsplit

import aiohttp

from .capabilities import Capability, current_invocation_id


_LIMIT = 1024 * 1024
_ERROR_LIMIT = 4096
_ID = {"type": "string", "minLength": 1, "maxLength": 200}
_OBJECT = {"type": "object", "additionalProperties": True}
_NAME = {"type": "string", "minLength": 1, "maxLength": 500}
_TEXT = {"type": "string", "minLength": 1}
_STRING = {"type": "string"}
_BOOLEAN = {"type": "boolean"}


def _schema(required: tuple[str, ...] = (), **properties: Any) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(required),
            "additionalProperties": False}


_PLAYER = _schema(name=_STRING, description=_STRING, properties=_OBJECT)
_WORLD_PATCH = _schema(
    entities={"type": "array", "items": _schema(
        ("kind", "name"), id=_ID, ref=_ID,
        kind={"type": "string", "enum": ["place", "creature", "item"]},
        name=_NAME, description=_STRING, appearance=_STRING, properties=_OBJECT,
        player=_PLAYER, player_visible=_BOOLEAN)},
    entity_updates={"type": "array", "items": _schema(
        ("entity_id",), entity_id=_ID, name=_TEXT, description=_STRING,
        appearance=_STRING,
        properties=_OBJECT, player=_PLAYER, player_visible=_BOOLEAN)},
    containment={"type": "array", "items": _schema(
        ("child_id", "parent_id"), child_id=_ID, parent_id=_ID)},
    connections={"type": "array", "items": _schema(
        ("from_place_id", "to_place_id"), id=_ID, ref=_ID,
        from_place_id=_ID, to_place_id=_ID, bidirectional=_BOOLEAN,
        typical_travel_minutes={"type": "integer", "minimum": 0},
        player_visible=_BOOLEAN)},
    facts={"type": "array", "items": _schema(
        ("text",), id=_ID, ref=_ID, text=_TEXT,
        subject_entity_id=_ID, metadata=_OBJECT)},
    knowledge={"type": "array", "items": _schema(
        ("actor_id", "fact_id"), actor_id=_ID, fact_id=_ID)},
    observations={"type": "array", "items": _schema(
        ("actor_id", "entity_id"), actor_id=_ID, entity_id=_ID)},
)


def _realm_error_detail(raw: bytes) -> str | None:
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    code, message = payload.get("error"), payload.get("message")
    if not isinstance(code, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", code):
        return None
    if not isinstance(message, str) or not message or len(message) > 300:
        return None
    if not message.isprintable():
        return None
    return f"{code}: {message}"


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
        self._mutation_requests: dict[str, tuple[str, dict[str, Any]]] = {}
        self._mutation_store: Callable[[str, str | None, dict[str, Any] | None],
                                       tuple[str, dict[str, Any]] | None] | None = None

    def bind_mutation_store(self, store: Callable[
            [str, str | None, dict[str, Any] | None],
            tuple[str, dict[str, Any]] | None] | None) -> None:
        self._mutation_store = store

    def _mutation_request(self, key: str, path: str | None = None,
                          body: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]] | None:
        if self._mutation_store is not None:
            return self._mutation_store(key, path, body)
        if key not in self._mutation_requests and path is not None and body is not None:
            self._mutation_requests[key] = (path, json.loads(json.dumps(body, allow_nan=False)))
        return self._mutation_requests.get(key)

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
                    detail = None
                    if 400 <= response.status < 500:
                        try:
                            raw_error = await response.content.read(_ERROR_LIMIT + 1)
                            if len(raw_error) <= _ERROR_LIMIT:
                                detail = _realm_error_detail(raw_error)
                        except (aiohttp.ClientError, TimeoutError, OSError):
                            pass
                    raise RealmHTTPError(response.status, detail)
                raw = await response.content.read(_LIMIT + 1)
                if len(raw) > _LIMIT:
                    raise ValueError("Realm response is too large")
                result = json.loads(raw)
                if not isinstance(result, dict):
                    raise ValueError("Realm response must be an object")
                return result

    async def snapshot(self) -> dict[str, Any]:
        for _ in range(3):
            before = await self._request("GET", f"{self.path}/authoritative-state")
            player = await self._request("GET", f"{self.path}/state?actor_id={quote(self.actor_id, safe='')}")
            trusted = await self._request("GET", f"{self.path}/authoritative-state")
            before_game = before.get("game")
            trusted_game = trusted.get("game")
            if (isinstance(before_game, dict) and isinstance(trusted_game, dict)
                    and before_game.get("current_revision") == trusted_game.get("current_revision")):
                break
        else:
            raise ValueError("Realm changed during snapshot acquisition")
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
        if isinstance(exc, RealmHTTPError):
            result["http_status"] = exc.status
            if exc.detail:
                result["error"] += f" (HTTP {exc.status}, {exc.detail})"
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
            request = self._mutation_request(key)
            if request is None:
                before = await self.snapshot()
                revision = before["trusted_state"]["game"]["current_revision"]
                body = {**arguments, "expected_revision": revision, "idempotency_key": key}
                if route in ("reveal-fact", "observe-entity"):
                    body["actor_id"] = self.actor_id
                path = f"{self.path}/world-patches" if route == "world-patch" else f"{self.path}/operations/{route}"
                request = self._mutation_request(key, path, body)
            if request is None:
                raise ValueError("Realm mutation request could not be recorded")
            path, body = request
            result = await self._request("POST", path, body)
        except (aiohttp.ClientError, TimeoutError, OSError, ValueError, RealmHTTPError) as exc:
            failure = self._error(exc, mutating="request" in locals() and request is not None)
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
             _WORLD_PATCH, "world-patch"),
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
    def __init__(self, status: int, detail: str | None = None):
        self.status = status
        self.detail = detail
