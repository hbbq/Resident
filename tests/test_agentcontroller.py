from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from resident.__main__ import TerminalDiagnostics
from resident.agentcontroller import AgentControllerConnector
from resident.config import Config


def workflow_item(number: int, *, repository: str = "owner/repo",
                  state: str = "queued_investigation", updated_at: str = "old",
                  title: str | None = None, complexity: str | None = "medium",
                  url: str | None = None, kind: str = "issue") -> dict:
    return {
        "repository": repository,
        "kind": kind,
        "number": number,
        "title": title or f"Item {number}",
        "state": state,
        "complexity": complexity,
        "url": url,
        "updated_at": updated_at,
    }


def snapshot(*items: dict, refreshed_at: str = "2026-09-16T06:00:00Z") -> dict:
    repositories = {}
    for item in items:
        repository = item["repository"]
        repositories.setdefault(repository, {"name": repository.split("/")[-1], "items": []})
        repositories[repository]["items"].append(item)
    return {"schema_version": 1, "refreshed_at": refreshed_at, "repositories": repositories}


class AgentControllerConnectorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "dashboard.json"

    def tearDown(self):
        self.temporary.cleanup()

    def write_snapshot(self, payload: object) -> None:
        self.path.write_text(json.dumps(payload), encoding="utf-8")

    async def test_first_poll_is_silent_and_refreshed_at_is_not_activity(self):
        self.write_snapshot(snapshot(workflow_item(1)))
        connector = AgentControllerConnector(self.path)
        queue = asyncio.Queue()

        await connector.poll_once(queue)
        self.write_snapshot(snapshot(workflow_item(1), refreshed_at="later"))
        await connector.poll_once(queue)

        self.assertTrue(queue.empty())

    async def test_poll_batches_added_removed_and_changed_items_factually(self):
        self.write_snapshot(snapshot(
            workflow_item(1, state="queued_investigation"),
            workflow_item(2),
        ))
        connector = AgentControllerConnector(self.path)
        queue = asyncio.Queue()
        await connector.poll_once(queue)

        self.write_snapshot(snapshot(
            workflow_item(1, state="needs_input", updated_at="new"),
            workflow_item(3, kind="pull_request", state="open_pull_request"),
        ))
        await connector.poll_once(queue)

        event = queue.get_nowait()
        self.assertEqual(("agentcontroller", "workflow_changed"), (event.source, event.reason))
        self.assertEqual((1, 1, 1), (
            event.payload["added_count"], event.payload["removed_count"],
            event.payload["changed_count"],
        ))
        self.assertEqual(3, event.payload["added"][0]["number"])
        self.assertEqual(2, event.payload["removed"][0]["number"])
        self.assertEqual({
            "state": {"old": "queued_investigation", "new": "needs_input"},
            "updated_at": {"old": "old", "new": "new"},
        }, event.payload["changed"][0]["changes"])
        self.assertTrue(queue.empty())

    async def test_durable_baseline_is_reused_after_restart(self):
        checkpoint = []
        self.write_snapshot(snapshot(workflow_item(1)))
        first = AgentControllerConnector(self.path)
        first.bind_checkpoint(
            lambda: checkpoint[-1] if checkpoint else None,
            checkpoint.append,
        )
        await first.poll_once(asyncio.Queue())

        self.write_snapshot(snapshot(workflow_item(1, state="needs_input", updated_at="new")))
        second = AgentControllerConnector(self.path)
        second.bind_checkpoint(lambda: checkpoint[-1], checkpoint.append)
        queue = asyncio.Queue()
        await second.poll_once(queue)

        self.assertEqual("needs_input", queue.get_nowait().payload["changed"][0]["changes"]["state"]["new"])

    async def test_invalid_snapshot_preserves_last_successful_baseline(self):
        self.write_snapshot(snapshot(workflow_item(1)))
        connector = AgentControllerConnector(self.path)
        queue = asyncio.Queue()
        await connector.poll_once(queue)

        self.write_snapshot({"schema_version": 2, "repositories": {}})
        with self.assertRaisesRegex(ValueError, "schema_version 1"):
            await connector.poll_once(queue)
        self.write_snapshot(snapshot(workflow_item(1, state="needs_input")))
        await connector.poll_once(queue)

        change = queue.get_nowait().payload["changed"][0]["changes"]["state"]
        self.assertEqual({"old": "queued_investigation", "new": "needs_input"}, change)

    async def test_list_capability_filters_and_bounds_current_snapshot(self):
        items = [workflow_item(number) for number in range(1, 4)]
        items.append(workflow_item(4, repository="other/repo"))
        self.write_snapshot(snapshot(*items, refreshed_at="fresh"))
        connector = AgentControllerConnector(self.path)

        result = await connector.list_workflow_items({"repository": "owner/repo", "limit": 2})

        self.assertEqual("fresh", result["snapshot_refreshed_at"])
        self.assertEqual(3, result["total_count"])
        self.assertEqual(2, result["returned_count"])
        self.assertTrue(result["truncated"])
        self.assertEqual("repo", result["items"][0]["repository_name"])
        self.assertEqual(["agentcontroller_list_workflow_items"], [
            capability.name for capability in connector.capabilities
        ])

    async def test_missing_snapshot_is_reported_without_exposing_its_path(self):
        connector = AgentControllerConnector(self.path)
        with self.assertRaisesRegex(RuntimeError, "snapshot is unavailable") as raised:
            await connector.poll_once(asyncio.Queue())
        self.assertNotIn(str(self.path), str(raised.exception))

    async def test_run_reports_recoverable_failure_and_stops(self):
        diagnostics = []
        connector = AgentControllerConnector(
            self.path, poll_seconds=10, diagnostic_output=diagnostics.append)
        stop = asyncio.Event()
        task = asyncio.create_task(connector.run(asyncio.Queue(), stop))
        while not diagnostics:
            await asyncio.sleep(0)
        stop.set()
        await task

        self.assertEqual(
            "poll failed: RuntimeError: AgentController snapshot is unavailable",
            diagnostics[0],
        )


class AgentControllerConfigTests(unittest.TestCase):
    def test_connector_is_disabled_by_default_and_cli_can_enable_it(self):
        with patch.dict(os.environ, {"RESIDENT_AGENTCONTROLLER_SNAPSHOT_PATH": ""}):
            disabled = Config.from_env_and_args(["--data-dir", ".resident"])
        enabled = Config.from_env_and_args([
            "--data-dir", ".resident",
            "--agentcontroller-snapshot-path", "controller/output/dashboard.json",
            "--agentcontroller-poll-seconds", "2.5",
        ])

        self.assertIsNone(disabled.agentcontroller_snapshot_path)
        self.assertEqual(
            Path("controller/output/dashboard.json"), enabled.agentcontroller_snapshot_path)
        self.assertEqual(2.5, enabled.agentcontroller_poll_seconds)

    def test_terminal_renderer_suppresses_retries_unless_verbose(self):
        with patch("builtins.print") as output:
            TerminalDiagnostics(verbose=False).agentcontroller("poll failed: unavailable")
            output.assert_not_called()

            TerminalDiagnostics(verbose=True).agentcontroller("poll failed: unavailable")
            output.assert_called_once_with("[agentcontroller] poll failed: unavailable")


if __name__ == "__main__":
    unittest.main()
