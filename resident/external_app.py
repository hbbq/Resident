from __future__ import annotations

import asyncio
import json
import socket
import uuid
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .capabilities import Capability, current_invocation_id
from .instances import ExternalApplicationDefinition, ExternalOperationDefinition
from .observability import to_thread_timed


_MAX_REQUEST_BYTES = 256 * 1024
_MAX_RESPONSE_BYTES = 1024 * 1024


class ExternalApplicationConnector:
    """Adapt a locally pinned capability catalog to a narrow HTTP invocation API."""

    def __init__(self, definition: ExternalApplicationDefinition,
                 bearer_token: str | None = None):
        parsed = urlsplit(definition.base_url)
        if (parsed.scheme not in ("http", "https") or not parsed.netloc or
                parsed.username is not None or parsed.password is not None or
                parsed.query or parsed.fragment):
            raise ValueError(
                f"External application {definition.id} base URL must be an absolute "
                "HTTP(S) URL without credentials, query, or fragment")
        self.definition = definition
        self.base_url = definition.base_url.rstrip("/")
        self.bearer_token = bearer_token

    @property
    def capabilities(self) -> list[Capability]:
        return [Capability(
            connector_id=self.definition.id,
            connector_description=self.definition.description,
            name=operation.name,
            description=operation.description,
            input_schema=operation.input_schema,
            handler=lambda arguments, operation=operation: self.invoke(operation, arguments),
        ) for operation in self.definition.operations]

    def _post(self, payload: bytes) -> Any:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.bearer_token is not None:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        request = Request(
            f"{self.base_url}/api/capabilities/invoke", data=payload,
            headers=headers, method="POST")
        with urlopen(request, timeout=self.definition.request_timeout_seconds) as response:
            raw = response.read(_MAX_RESPONSE_BYTES + 1)
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise ValueError("response_too_large")
        try:
            decoded = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid_json_response") from exc
        if not isinstance(decoded, (dict, list)):
            raise ValueError("invalid_response_shape")
        return decoded

    @staticmethod
    def _failure(code: str, message: str, request_id: str, *,
                 unknown: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "ok": False, "error": message, "error_code": code, "request_id": request_id,
        }
        if unknown:
            result["outcome"] = "unknown"
        return result

    async def invoke(self, operation: ExternalOperationDefinition,
                     arguments: dict[str, Any]) -> dict[str, Any]:
        request_id = current_invocation_id() or str(uuid.uuid4())
        payload = json.dumps({
            "operation": operation.operation,
            "request_id": request_id,
            "arguments": arguments,
            "bindings": self.definition.bindings,
        }, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(payload) > _MAX_REQUEST_BYTES:
            return self._failure(
                "request_too_large", "External operation request is too large", request_id)
        try:
            result = await to_thread_timed(
                "external_application.request", self._post, payload,
                provider=self.definition.id, external_operation=operation.operation,
                mutating=operation.mutating,
                request_timeout_seconds=self.definition.request_timeout_seconds)
        except asyncio.CancelledError:
            raise
        except HTTPError as exc:
            if exc.code in (401, 403):
                code, message = "authentication_failed", "External application rejected authentication"
            elif exc.code == 409:
                code, message = "conflict", "External operation was rejected due to a conflict"
            elif operation.mutating and (exc.code == 408 or exc.code >= 500):
                return self._failure(
                    "unknown_outcome",
                    "External operation outcome is unknown after an upstream failure; "
                    "reconcile by request_id",
                    request_id, unknown=True)
            elif exc.code in (408, 429) or exc.code >= 500:
                code, message = "unavailable", "External application is unavailable"
            elif 400 <= exc.code < 500:
                code, message = "rejected", "External application rejected the operation"
            else:
                code, message = "unavailable", "External application is unavailable"
            return self._failure(code, message, request_id)
        except (TimeoutError, socket.timeout) as exc:
            del exc
            if operation.mutating:
                return self._failure(
                    "unknown_outcome",
                    "External operation outcome is unknown after a timeout; reconcile by request_id",
                    request_id, unknown=True)
            return self._failure("timeout", "External operation timed out", request_id)
        except URLError as exc:
            is_timeout = isinstance(exc.reason, (TimeoutError, socket.timeout))
            if operation.mutating:
                return self._failure(
                    "unknown_outcome",
                    "External operation outcome is unknown after a transport failure; "
                    "reconcile by request_id",
                    request_id, unknown=True)
            return self._failure(
                "timeout" if is_timeout else "unavailable",
                "External operation timed out" if is_timeout else "External application is unavailable",
                request_id)
        except ValueError:
            return self._failure(
                "invalid_response", "External application returned an invalid response", request_id)
        except OSError:
            if operation.mutating:
                return self._failure(
                    "unknown_outcome",
                    "External operation outcome is unknown after a transport failure; "
                    "reconcile by request_id",
                    request_id, unknown=True)
            return self._failure(
                "unavailable", "External application is unavailable", request_id)
        return {"request_id": request_id, "result": result}
