import asyncio
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

    def build(self, text=DEFINITION, *, shared=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        definitions = root / "residents"
        definitions.mkdir()
        (definitions / "resident.yaml").write_text(text, encoding="utf-8")
        config = Config(root / "data", residents_dir=definitions)
        if shared is not None:
            with patch("resident.__main__._shared_resources", return_value=([], shared, [])):
                return build_host(config)
        return build_host(config)

    @staticmethod
    def second_provider(provider_id, tool_name):
        return (f"  - id: {provider_id}\n"
                "    description: Second provider\n"
                "    base_url: http://second.local\n"
                "    operations:\n"
                f"      - name: {tool_name}\n"
                "        description: Second operation\n"
                "        input_schema: {type: object, additionalProperties: false}\n")

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

    def test_rejects_provider_id_matching_another_providers_tool(self):
        text = DEFINITION + self.second_provider("realm_apply_damage", "realm_apply_damage_ping")
        with self.assertRaisesRegex(ValueError, "realm_apply_damage.*external provider id.*rename"):
            self.build(text)

    def test_rejects_tool_matching_another_provider_id_in_reverse_order(self):
        text = DEFINITION.replace("realm_apply_damage", "realm_other", 1)
        text = text.replace("external_applications:\n", "external_applications:\n" +
                            self.second_provider("realm_other", "realm_other_ping"), 1)
        with self.assertRaisesRegex(ValueError, "realm_other.*external provider id.*rename"):
            self.build(text)

    def test_rejects_provider_id_matching_builtin_capability_name(self):
        text = DEFINITION.replace("realm", "diagnostics_current_time")
        with self.assertRaisesRegex(ValueError, "diagnostics_current_time.*built-in capability name.*rename"):
            self.build(text)

    def test_rejects_external_tool_matching_shared_provider_id(self):
        shared = [Capability("realm_apply_damage", "Shared provider", "shared_ping",
                             "Shared tool", {"type": "object"}, lambda _: None)]
        with self.assertRaisesRegex(ValueError, "realm_apply_damage.*built-in provider id.*rename"):
            self.build(shared=shared)

    def test_multiple_providers_keep_provider_and_individual_tool_grants(self):
        text = DEFINITION + self.second_provider("other", "other_ping")
        for grants, expected in (
                ("[realm, other_ping]",
                 {"realm_apply_damage", "realm_get_operation", "other_ping"}),
                ("[other, realm_apply_damage]", {"other_ping", "realm_apply_damage"})):
            with self.subTest(grants=grants), patch.dict("os.environ", {"REALM_TOKEN": "token"}), patch(
                    "resident.__main__._provider", side_effect=lambda _: IdleProvider()):
                host = self.build(text.replace("capabilities: [realm]", f"capabilities: {grants}"))
                try:
                    self.assertEqual(expected, {item.name for item in host.runtimes["resident"].capabilities})
                finally:
                    host.close()

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

    async def test_optional_bearer_token_is_transport_only(self):
        connector = self.connector()
        captured = {}
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                captured["authorization"] = self.headers.get("Authorization")
                self.send_response(200)
                self.send_header("Content-Length", "14")
                self.end_headers()
                self.wfile.write(b'{"ready":true}')

            def log_message(self, *_):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            connector.base_url = f"http://127.0.0.1:{server.server_port}"
            result = await connector._post(b"{}")
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        self.assertEqual({"ready": True}, result)
        self.assertEqual("Bearer private-token", captured["authorization"])

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
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(307)
                self.send_header(
                    "Location", f"http://127.0.0.1:{destination.server_port}/stolen")
                self.send_header("Content-Length", "0")
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
            result = await connector.invoke(connector.definition.operations[1], {})
            self.assertEqual("invalid_response", result["error_code"])
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

        for raw in (b"not json", b"true", b" " * (1024 * 1024 + 1),
                    b'{"value":NaN}', b'{"value":Infinity}', b'{"value":-Infinity}',
                    b'[1, {"value": NaN}]', b'{"value":1e400}',
                    b'{"value":-1e400}'):
            captured = {}
            class Handler(BaseHTTPRequestHandler):
                def do_POST(self):
                    length = int(self.headers["Content-Length"])
                    captured["request_id"] = json.loads(
                        self.rfile.read(length))["request_id"]
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(raw)

                def log_message(self, *_):
                    pass

            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            connector.base_url = f"http://127.0.0.1:{server.server_port}"
            for operation, expected in ((connector.definition.operations[0], "unknown_outcome"),
                                        (connector.definition.operations[1], "invalid_response")):
                with self.subTest(raw=raw[:16], mutating=operation.mutating):
                    with patch("resident.external_app.current_invocation_id",
                               return_value="managed-call-42"):
                        result = await connector.invoke(operation, {})
                    self.assertEqual(expected, result["error_code"])
                    self.assertEqual("managed-call-42", captured["request_id"])
                    self.assertEqual(captured["request_id"], result["request_id"])
                    self.assertEqual(operation.mutating, result.get("outcome") == "unknown")
            server.shutdown()
            server.server_close()
            thread.join()

    async def test_deep_json_response_is_sanitized_for_both_operation_types(self):
        connector = self.connector()
        raw = b"[" * 2000 + b"0" + b"]" * 2000
        captured = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                captured.append(request["request_id"])
                self.send_response(200)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *_):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            connector.base_url = f"http://127.0.0.1:{server.server_port}"
            for operation, expected_code, expected_message in (
                    (connector.definition.operations[1], "invalid_response",
                     "External application returned an invalid response"),
                    (connector.definition.operations[0], "unknown_outcome",
                     "External operation outcome is unknown after an invalid response; "
                     "reconcile by request_id")):
                with self.subTest(mutating=operation.mutating), patch(
                        "resident.external_app.current_invocation_id",
                        return_value="managed-call-42"):
                    result = await connector.invoke(operation, {})
                    self.assertEqual(expected_code, result["error_code"])
                    self.assertEqual(expected_message, result["error"])
                    self.assertEqual("managed-call-42", result["request_id"])
                    self.assertEqual(operation.mutating, result.get("outcome") == "unknown")
            self.assertEqual(["managed-call-42"] * 2, captured)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    async def test_invocation_uses_stable_call_id_and_server_side_bindings(self):
        connector = self.connector()
        captured = {}

        async def post(payload):
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

        async def timeout(_):
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
                self.rfile.read(int(self.headers["Content-Length"]))
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
            original_post = connector._post
            worker_done = threading.Event()

            async def tracked_post(payload):
                worker_done.clear()
                try:
                    return await original_post(payload)
                finally:
                    worker_done.set()

            connector._post = tracked_post
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
                    self.assertTrue(worker_done.is_set(), "HTTP worker remained active after invoke")
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    async def test_stalled_dns_resolution_is_cancelled_at_deadline(self):
        connector = self.connector()
        connector.definition = replace(connector.definition, request_timeout_seconds=0.15)
        active = 0
        cancelled = asyncio.Event()

        async def stalled_resolution(resolver, host, port=0, family=socket.AF_INET):
            nonlocal active
            active += 1
            try:
                await asyncio.Event().wait()
            finally:
                active -= 1
                cancelled.set()

        with patch("resident.external_app.AsyncResolver.resolve", stalled_resolution), patch(
                "resident.external_app.current_invocation_id",
                return_value="managed-call-42"):
            for operation, expected in (
                    (connector.definition.operations[0], "unknown_outcome"),
                    (connector.definition.operations[1], "timeout")):
                cancelled.clear()
                started = time.monotonic()
                result = await connector.invoke(operation, {})
                self.assertLess(time.monotonic() - started, 0.5)
                self.assertEqual(expected, result["error_code"])
                self.assertEqual("managed-call-42", result["request_id"])
                self.assertTrue(cancelled.is_set(), "DNS resolution remained active")
                self.assertEqual(0, active)

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
                    async def fail(payload):
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
