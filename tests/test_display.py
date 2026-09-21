from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from resident.config import Config, DisplayConfig
from resident.display import DisplayConnector
from resident.tools import ToolRegistry


class FakeResponse:
    def __init__(self, status: int):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class DisplayConnectorTests(unittest.IsolatedAsyncioTestCase):
    async def test_each_display_exposes_a_named_show_text_capability(self):
        connector = DisplayConnector("http://homeops.test", [
            DisplayConfig("display1"), DisplayConfig("hall-panel"),
        ])

        self.assertEqual(
            ["display1_show_text", "hall-panel_show_text"],
            [capability.name for capability in connector.capabilities],
        )
        self.assertEqual(["text"], connector.capabilities[0].input_schema["required"])

    async def test_each_display_exposes_a_target_specific_output_capability(self):
        connector = DisplayConnector("http://homeops.test", [
            DisplayConfig("display1", 40), DisplayConfig("display2", 120),
        ])

        first, second = connector.output_capabilities
        self.assertEqual(("display", "display1"), (first.output_type, first.target))
        self.assertEqual(40, first.payload_schema["properties"]["content"]["maxLength"])
        self.assertEqual(120, second.payload_schema["properties"]["content"]["maxLength"])

    async def test_show_text_posts_json_to_escaped_display_endpoint_and_accepts_204(self):
        connector = DisplayConnector(
            "http://homeops.test/root/", [DisplayConfig("display1")],
            request_timeout_seconds=3,
        )
        calls = []

        def open_request(request, timeout):
            calls.append((request, timeout))
            return FakeResponse(204)

        with patch("resident.display.urlopen", open_request):
            result = await connector.show_text("display/one", "Hej ☃")

        request, timeout = calls[0]
        self.assertEqual(
            "http://homeops.test/root/api/displays/display%2Fone/messages",
            request.full_url,
        )
        self.assertEqual("POST", request.get_method())
        self.assertEqual({"text": "Hej ☃"}, json.loads(request.data.decode("utf-8")))
        self.assertEqual("application/json", request.get_header("Content-type"))
        self.assertEqual(3, timeout)
        self.assertEqual({"display_id": "display/one", "status": "queued"}, result)

    async def test_unexpected_success_status_becomes_a_clear_tool_failure(self):
        connector = DisplayConnector("http://homeops.test", [DisplayConfig("display1")])
        registry = ToolRegistry(
            None, connector.capabilities, lambda _: None, lambda *_: None)

        with patch("resident.display.urlopen", return_value=FakeResponse(200)):
            result = await registry.execute("display1_show_text", {"text": "hello"})

        self.assertFalse(result.output["ok"])
        self.assertIn("HTTP status 200; expected 204", result.output["error"])


class DisplayConfigTests(unittest.TestCase):
    def test_display_configuration_is_opt_in_and_reuses_homeops_settings(self):
        with patch.dict(os.environ, {"RESIDENT_DISPLAYS": "", "RESIDENT_HOMEOPS_URL": ""}):
            disabled = Config.from_env_and_args(["--data-dir", ".resident"])
        with patch.dict(os.environ, {
            "RESIDENT_DISPLAYS": '[{"id":"display1"}]',
            "RESIDENT_HOMEOPS_URL": "http://homeops.test/",
            "RESIDENT_HOMEOPS_REQUEST_TIMEOUT_SECONDS": "4",
        }):
            enabled = Config.from_env_and_args(["--data-dir", ".resident"])

        self.assertEqual((), disabled.displays)
        self.assertEqual((DisplayConfig("display1"),), enabled.displays)
        self.assertEqual("http://homeops.test", enabled.homeops_url)
        self.assertEqual(4, enabled.homeops_request_timeout_seconds)

    def test_display_configuration_accepts_target_specific_max_length(self):
        with patch.dict(os.environ, {
            "RESIDENT_DISPLAYS": '[{"id":"display1","max_length":40}]',
            "RESIDENT_HOMEOPS_URL": "http://homeops.test",
        }):
            config = Config.from_env_and_args(["--data-dir", ".resident"])
        self.assertEqual((DisplayConfig("display1", 40),), config.displays)

    def test_displays_require_homeops_url(self):
        with patch.dict(os.environ, {
            "RESIDENT_DISPLAYS": '[{"id":"display1"}]', "RESIDENT_HOMEOPS_URL": "",
        }):
            with self.assertRaisesRegex(ValueError, "RESIDENT_DISPLAYS requires"):
                Config.from_env_and_args(["--data-dir", ".resident"])

    def test_invalid_and_duplicate_display_ids_are_rejected(self):
        invalid_configs = (
            '[{"id":"not valid"}]',
            '[{"id":"same"},{"id":"same"}]',
            '[{"id":"one","name":"extra"}]',
        )
        for configured in invalid_configs:
            with self.subTest(configured=configured), patch.dict(os.environ, {
                "RESIDENT_DISPLAYS": configured,
                "RESIDENT_HOMEOPS_URL": "http://homeops.test",
            }):
                with self.assertRaises(ValueError):
                    Config.from_env_and_args(["--data-dir", ".resident"])


if __name__ == "__main__":
    unittest.main()
