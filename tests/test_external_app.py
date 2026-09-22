import json
import socket
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from http.client import BadStatusLine, IncompleteRead
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from unittest.mock import patch

from resident.__main__ import build_host
from resident.capabilities import Capability
from resident.config import Config
from resident.domain import ModelTurn
from resident.external_app import ExternalApplicationConnector
from resident.instances import load_resident_catalog
from resident.store import Store
from resident.tools import ToolRegistry


DEFINITION = """
id: resident
name: Resident
personality: Test.
role: Test.
capabilities: [realm]
external_applications:
  - id: realm
    description: Persistent game state
    base_url: http://realm.local
    bearer_token_env: REALM_TOKEN
    request_timeout_seconds: 3
    bindings:
      game_id: game-7
    operations:
      - name: realm_apply_damage
        operation: apply_damage
        description: Apply validated damage to a character.
        mutating: true
        input_schema:
          type: object
          properties:
            character_id: {type: string}
            amount: {type: integer, minimum: 1, maximum: 100}
          required: [character_id, amount]
          additionalProperties: false
      - name: realm_get_operation
        operation: get_operation
        description: Reconcile an operation by request identifier.
        input_schema:
          type: object
          properties:
            request_id: {type: string}
          required: [request_id]
          additionalProperties: false
"""


class IdleProvider:
    async def respond(self, context, tools, results, continuation_id=None):
        return ModelTurn("turn", None, ())


class ExternalApplicationDefinitionTests(unittest.TestCase):
    def load(self, text=DEFINITION):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        definitions = Path(temporary.name) / "residents"
        definitions.mkdir()
        (definitions / "resident.yaml").write_text(text, encoding="utf-8")
        return load_resident_catalog(definitions).residents[0]

    def test_loads_locally_pinned_provider_catalog_and_bindings(self):
        resident = self.load()
        provider = resident.external_applications[0]
        self.assertEqual("realm", provider.id)
        self.assertEqual({"game_id": "game-7"}, provider.bindings)
        self.assertEqual("REALM_TOKEN", provider.bearer_token_env)
        self.assertTrue(provider.operations[0].mutating)
        self.assertEqual("realm_apply_damage", provider.operations[0].name)

    def test_rejects_un_namespaced_tools_and_unsupported_schema(self):
        with self.assertRaisesRegex(ValueError, "namespaced tool name"):
            self.load(DEFINITION.replace("realm_apply_damage", "apply_damage", 1))
        with self.assertRaisesRegex(ValueError, "Unsupported JSON Schema"):
            self.load(DEFINITION.replace(
                "additionalProperties: false\n      - name: realm_get_operation",
                "additionalProperties: false\n          oneOf: []\n      - name: realm_get_operation"))

    def test_rejects_non_finite_request_timeout(self):
        for value in (".nan", ".inf", "-.inf"):
            with self.subTest(value=value), self.assertRaisesRegex(
                    ValueError, "request_timeout_seconds must be a positive number"):
                self.load(DEFINITION.replace("request_timeout_seconds: 3",
                                             f"request_timeout_seconds: {value}"))

    def test_rejects_external_messaging_provider_even_without_a_messaging_grant(self):
        for grants in ("[]", "[messaging]"):
            with self.subTest(grants=grants), self.assertRaisesRegex(
                    ValueError, "reserved for built-in messaging"):
                self.load(DEFINITION.replace("capabilities: [realm]", f"capabilities: {grants}")
                          .replace("id: realm\n", "id: messaging\n")
                          .replace("realm_apply_damage", "messaging_send")
                          .replace("realm_get_operation", "messaging_get_operation"))

    def test_special_messaging_capability_participates_in_collision_check(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            definitions = root / "residents"
            definitions.mkdir()
            (definitions / "resident.yaml").write_text(
                "id: resident\nname: Resident\npersonality: Test.\nrole: Test.\n"
                "capabilities: [messaging]\n", encoding="utf-8")
            impostor = Capability(
                "other", "Other", "messaging_send", "Other send", {"type": "object"},
                lambda _: None)
            with patch("resident.__main__._shared_resources", return_value=([], [impostor], [])):
                with self.assertRaisesRegex(ValueError, "Duplicate available capability names.*messaging_send"):
                    build_host(Config(root / "data", residents_dir=definitions))

    def test_external_inventory_is_private_to_its_resident(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            definitions = root / "residents"
            definitions.mkdir()
            (definitions / "resident.yaml").write_text(DEFINITION, encoding="utf-8")
            (definitions / "helper.yaml").write_text(
                "id: helper\nname: Helper\npersonality: Test.\nrole: Test.\n",
                encoding="utf-8")
            with patch.dict("os.environ", {"REALM_TOKEN": "private-token"}), patch(
                    "resident.__main__._provider", side_effect=lambda _: IdleProvider()):
                host = build_host(Config(root / "data", residents_dir=definitions))
            try:
                self.assertEqual(
                    {"realm_apply_damage", "realm_get_operation"},
                    {item.name for item in host.runtimes["resident"].capabilities})
                self.assertEqual((), host.runtimes["helper"].capabilities)
            finally:
                host.close()


class ExternalApplicationConnectorTests(unittest.IsolatedAsyncioTestCase):
    def connector(self):
        with tempfile.TemporaryDirectory() as temporary:
            definitions = Path(temporary) / "residents"
            definitions.mkdir()
            (definitions / "resident.yaml").write_text(DEFINITION, encoding="utf-8")
            definition = load_resident_catalog(definitions).residents[0].external_applications[0]
        return ExternalApplicationConnector(definition, "private-token")

    def test_optional_bearer_token_is_transport_only(self):
        connector = self.connector()
        captured = {}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return None

            def read(self, _):
                return b'{"ready":true}'

        def open_request(request, timeout):
            captured["authorization"] = request.get_header("Authorization")
            captured["timeout"] = timeout
            return Response()

        class Opener:
            open = staticmethod(open_request)

        with patch("resident.external_app.build_opener", return_value=Opener()):
            result = connector._post(b"{}")
        self.assertEqual({"ready": True}, result)
        self.assertEqual("Bearer private-token", captured["authorization"])
        self.assertEqual(3, captured["timeout"])

    async def test_redirect_does_not_forward_bearer_token_to_another_origin(self):
        received = []

        class Destination(BaseHTTPRequestHandler):
            def do_POST(self):
                received.append(self.headers.get("Authorization"))
                self.send_response(200)
                self.end_headers()

            def log_message(self, *_):
                pass

        destination = ThreadingHTTPServer(("127.0.0.1", 0), Destination)

        class Redirect(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(307)
                self.send_header(
                    "Location", f"http://127.0.0.1:{destination.server_port}/stolen")
                self.end_headers()

            def log_message(self, *_):
                pass

        redirect = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
        threads = [threading.Thread(target=server.serve_forever, daemon=True)
                   for server in (destination, redirect)]
        for thread in threads:
            thread.start()
        try:
            connector = self.connector()
            connector.base_url = f"http://127.0.0.1:{redirect.server_port}"
            with self.assertRaises(HTTPError) as error:
                connector._post(b"{}")
            self.assertEqual(307, error.exception.code)
            result = await connector.invoke(connector.definition.operations[0], {})
            self.assertEqual("unknown_outcome", result["error_code"])
            self.assertEqual([], received)
        finally:
            for server in (redirect, destination):
                server.shutdown()
                server.server_close()
            for thread in threads:
                thread.join()

    async def test_invalid_responses_preserve_mutating_request_id(self):
        connector = self.connector()

        class Response:
            def __init__(self, raw):
                self.raw = raw

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return None

            def read(self, _):
                return self.raw

        class Opener:
            def __init__(self, raw):
                self.raw = raw
                self.request_id = None

            def open(self, request, timeout):
                self.request_id = json.loads(request.data)["request_id"]
                return Response(self.raw)

        for raw in (b"not json", b"true", b" " * (1024 * 1024 + 1)):
            for operation, expected in ((connector.definition.operations[0], "unknown_outcome"),
                                        (connector.definition.operations[1], "invalid_response")):
                with self.subTest(raw=raw[:16], mutating=operation.mutating):
                    opener = Opener(raw)
                    with patch("resident.external_app.build_opener", return_value=opener), patch(
                            "resident.external_app.current_invocation_id", return_value="managed-call-42"):
                        result = await connector.invoke(operation, {})
                    self.assertEqual(expected, result["error_code"])
                    self.assertEqual("managed-call-42", opener.request_id)
                    self.assertEqual(opener.request_id, result["request_id"])
                    self.assertEqual(operation.mutating, result.get("outcome") == "unknown")

    async def test_invocation_uses_stable_call_id_and_server_side_bindings(self):
        connector = self.connector()
        captured = {}

        def post(payload):
            captured.update(json.loads(payload))
            return {"character_id": "c1", "hp": 4}

        connector._post = post
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "state.sqlite3")
            try:
                registry = ToolRegistry(
                    store, connector.capabilities, lambda _: {}, lambda *_: None)
                result = await registry.execute(
                    "realm_apply_damage", {"character_id": "c1", "amount": 3},
                    invocation_id="managed-call-42")
            finally:
                store.close()

        self.assertTrue(result.output["ok"])
        self.assertEqual("managed-call-42", result.output["request_id"])
        self.assertEqual("managed-call-42", captured["request_id"])
        self.assertEqual({"game_id": "game-7"}, captured["bindings"])
        self.assertNotIn("bindings", captured["arguments"])

    async def test_mutating_timeout_is_unknown_and_never_retried(self):
        connector = self.connector()
        calls = 0

        def timeout(_):
            nonlocal calls
            calls += 1
            raise socket.timeout()

        connector._post = timeout
        result = await connector.invoke(
            connector.definition.operations[0], {"character_id": "c1", "amount": 3})
        self.assertEqual(1, calls)
        self.assertFalse(result["ok"])
        self.assertEqual("unknown_outcome", result["error_code"])
        self.assertEqual("unknown", result["outcome"])
        self.assertIn("request_id", result)
        self.assertNotIn("realm.local", json.dumps(result))

    async def test_incremental_response_cannot_extend_overall_deadline(self):
        class Trickle(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", "100")
                self.end_headers()
                for _ in range(100):
                    try:
                        self.wfile.write(b" ")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        break
                    time.sleep(0.02)

            def log_message(self, *_):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Trickle)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            connector = self.connector()
            connector.base_url = f"http://127.0.0.1:{server.server_port}"
            connector.definition = replace(connector.definition, request_timeout_seconds=0.15)
            for operation, expected in (
                    (connector.definition.operations[0], "unknown_outcome"),
                    (connector.definition.operations[1], "timeout")):
                with self.subTest(mutating=operation.mutating), patch(
                        "resident.external_app.current_invocation_id",
                        return_value="managed-call-42"):
                    started = time.monotonic()
                    result = await connector.invoke(operation, {})
                    self.assertLess(time.monotonic() - started, 0.5)
                    self.assertEqual(expected, result["error_code"])
                    self.assertEqual("managed-call-42", result["request_id"])
                    self.assertEqual(operation.mutating, result.get("outcome") == "unknown")
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    async def test_protocol_failures_are_sanitized_with_stable_request_id(self):
        connector = self.connector()
        for exception in (BadStatusLine("secret upstream status"),
                          IncompleteRead(b"secret upstream body", 10)):
            for operation, expected in (
                    (connector.definition.operations[0], "unknown_outcome"),
                    (connector.definition.operations[1], "invalid_response")):
                with self.subTest(exception=type(exception).__name__,
                                  mutating=operation.mutating), patch(
                        "resident.external_app.current_invocation_id",
                        return_value="managed-call-42"):
                    def fail(payload):
                        self.assertEqual("managed-call-42", json.loads(payload)["request_id"])
                        raise exception

                    connector._post = fail
                    result = await connector.invoke(operation, {})
                    self.assertEqual(expected, result["error_code"])
                    self.assertEqual("managed-call-42", result["request_id"])
                    self.assertEqual(operation.mutating, result.get("outcome") == "unknown")
                    self.assertNotIn("secret upstream", json.dumps(result))

    async def test_nested_and_array_arguments_are_validated_locally(self):
        connector = self.connector()
        capability = connector.capabilities[0]
        schema = {
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": {"type": "integer", "minimum": 1}},
            },
            "required": ["items"],
            "additionalProperties": False,
        }
        capability = type(capability)(
            capability.connector_id, capability.connector_description,
            capability.name, capability.description, schema, capability.handler)
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "state.sqlite3")
            try:
                registry = ToolRegistry(store, [capability], lambda _: {}, lambda *_: None)
                result = await registry.execute(capability.name, {"items": [1, 0]})
            finally:
                store.close()
        self.assertFalse(result.output["ok"])
        self.assertIn("outside the allowed range", result.output["error"])
