"""Opt-in cross-repository acceptance against an isolated compiled game host.

PROJECTDELVE_TEST_HOST_DLL points to a freshly built ProjectDelve.Web.dll.
Only the process started by this fixture is stopped; no real Resident/OpenAI
provider, shared database, or managed development service is used.
"""
import asyncio
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import unittest
import uuid
from pathlib import Path

import aiohttp

from resident.config import Config
from resident.domain import ModelTurn
from resident.projectdelve import ProjectDelveConnector
from resident.runtime import ResidentRuntime


class LegalProvider:
    def __init__(self, *, delayed=False):
        self.conversation = "isolated-gambit-" + uuid.uuid4().hex
        self.passive = True
        self.contexts = []
        self.tools = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        if not delayed:
            self.release.set()

    async def create_conversation(self):
        return self.conversation

    async def respond(self, context, tools, results, **kwargs):
        value = json.loads(context)
        self.contexts.append(value)
        self.tools.append(list(tools))
        self.entered.set()
        await self.release.wait()
        candidates = value["wake_event"]["payload"]["observation"]["request"]["candidates"]
        # First activation is passive so both players can publish private commentary.
        if self.passive:
            selected = next((c for c in candidates if c["kind"] == "EndTurn"), None)
            selected = selected or next((c for c in candidates if c["kind"] == "Stay"), candidates[0])
        else:
            selected = next((c for c in candidates if c.get("targetId") or c.get("targetIds")), None)
            selected = selected or next((c for c in candidates if c["kind"] == "Action"), None)
            selected = selected or next((c for c in candidates if c["kind"] == "EndTurn"), None)
            selected = selected or next((c for c in candidates if c["kind"] == "Stay"), candidates[0])
        return ModelTurn(self.conversation + "-response-" + str(len(self.contexts)), json.dumps({"outputs": [{
            "type": "projectdelve_decision", "candidateKey": selected["key"],
            "commentary": "GAMBIT_PRIVATE_HTTP_MARKER"}]}))


@unittest.skipUnless(os.environ.get("PROJECTDELVE_TEST_HOST_DLL"),
                     "Set PROJECTDELVE_TEST_HOST_DLL to opt into isolated game-host acceptance")
class ProjectDelveHttpAcceptance(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gambit-http-")
        self.runtimes = []
        self.process = None
        self.log = None
        self.player = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5))
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        self.url = "http://127.0.0.1:" + str(port)
        self.token = "isolated-controller-token"
        dll = Path(os.environ["PROJECTDELVE_TEST_HOST_DLL"]).resolve()
        self.assertTrue(dll.is_file(), str(dll))
        env = dict(os.environ, PROJECTDELVE_GAMBIT_TOKEN=self.token,
                   DATA_DIR=self.temp.name,
                   ASPNETCORE_ENVIRONMENT="Production", DOTNET_ENVIRONMENT="Production",
                   Logging__EventLog__LogLevel__Default="None")
        self.log = open(Path(self.temp.name) / "host.log", "w+", encoding="utf-8")
        self.process = subprocess.Popen([shutil.which("dotnet") or "dotnet", str(dll),
            "--urls", self.url], cwd=self.temp.name, env=env, stdin=subprocess.DEVNULL,
            stdout=self.log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            deadline = time.monotonic() + 15
            while True:
                try:
                    await self.http("/api/game")
                    break
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    if time.monotonic() >= deadline or self.process.poll() is not None:
                        self.log.seek(0)
                        self.fail("Isolated host startup failed: " + self.log.read())
                    await asyncio.sleep(.05)
        except BaseException:
            await self.cleanup()
            raise

    async def cleanup(self):
        for runtime in self.runtimes:
            runtime.close()
        self.runtimes.clear()
        await self.player.close()
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                await asyncio.to_thread(self.process.wait, 5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                await asyncio.to_thread(self.process.wait, 5)
        if self.log is not None:
            self.log.close()
        self.temp.cleanup()

    async def asyncTearDown(self):
        await self.cleanup()

    async def http(self, route, body=None):
        async with self.player.request(
                "POST" if body is not None else "GET", self.url + route, json=body) as response:
            raw = await response.text()
            self.assertLess(response.status, 300, (route, response.status, raw))
            return json.loads(raw)

    def runtime(self, name, *, delayed=False):
        connector = ProjectDelveConnector(self.url, self.token)
        provider = LegalProvider(delayed=delayed)
        runtime = ResidentRuntime(Config(Path(self.temp.name) / name), provider,
            projectdelve_client=connector, output_capabilities=connector.output_capabilities,
            owner_output_enabled=False, diagnostic_output=lambda _: None)
        self.runtimes.append(runtime)
        return runtime, connector, provider

    async def setup_match(self, gambit_side):
        catalog = await self.http("/api/designer/catalog")
        kinds = {unit["displayName"]: unit["unitTypeId"] for unit in catalog["unitTypes"]}
        barbarian, grunt = kinds["Barbarian"], kinds["Grunt"]
        draft = {"board": {"width": 3, "height": 2, "defaultTerrain": "StoneFloor",
                            "cells": [], "edges": []}, "unitTypeIds": [barbarian, grunt],
                 "units": [{"unitTypeId": kind, "sideId": side, "anchor": {"x": x, "y": 0},
                            "posture": "Upright", "initialHp": 1}
                           for kind, side, x in [(barbarian, "violet", 0), (grunt, "copper", 1)]],
                 "agency": [{"unitTypeId": kind, "sideId": side, "controller": "Human"}
                            for kind, side in [(barbarian, "violet"), (grunt, "copper")]]}
        exported = await self.http("/api/designer/export", draft)
        current = await self.http("/api/game")
        configured = await self.http("/api/game/scenario/import", {
            "expectedRevision": current["revision"], "transport": exported["transport"],
            "controllers": {side: "Gambit" if side == gambit_side else "Human"
                            for side in ("violet", "copper")}})
        self.assertEqual(0, configured["result"]["state"]["round"])
        return configured

    async def next_event(self, connector, *, limit=60):
        queue = asyncio.Queue()
        for _ in range(limit):
            await connector.poll_once(queue)
            if not queue.empty():
                return queue.get_nowait()
            current = await self.http("/api/game")
            state = current["result"]["state"]
            if state.get("completion"):
                self.fail("Match completed before expected Gambit decision")
            if state["round"] == 0 or state["roundComplete"]:
                await self.http("/api/game/round", {"expectedRevision": current["revision"]})
            elif current["presentation"].get("decision"):
                await self.human_pass(current)
            else:
                await asyncio.sleep(.01)
        self.fail("No Gambit decision within bounded legal progression")

    async def human_pass(self, current, *, passive=True):
        decision = current["presentation"]["decision"]
        # Browser projections expose supplied opaque choices: do not reconstruct legality.
        candidates = decision["candidates"]
        selected = next((c for c in candidates if c.get("key", "").endswith("EndTurn")), None)
        # Use engine candidate metadata for a passive legal choice when available.
        raw = current["result"].get("nextInput") or {}
        end = next((c for c in raw.get("candidates", []) if c["kind"] == "EndTurn"), None)
        end = end or next((c for c in raw.get("candidates", []) if c["kind"] == "Stay"), None)
        if not passive:
            end = next((c for c in raw.get("candidates", []) if c.get("targetId") or c.get("targetIds")), None) or end
        key = end["key"] if end else (selected or candidates[0])["key"]
        return await self.http("/api/game/decision", {"expectedRevision": current["revision"],
            "candidateKey": key, "commentary": "HUMAN_PRIVATE_HTTP_MARKER"})

    async def reconcile(self, runtime):
        for _ in range(100):
            runtime.store.connection.execute("UPDATE output_requests SET next_attempt_at=NULL")
            runtime.store.connection.commit()
            await runtime.dispatch_outputs_once()
            row = runtime.store.connection.execute(
                "SELECT delivery_state FROM output_requests ORDER BY rowid DESC LIMIT 1").fetchone()
            if row and row[0] in ("applied", "rejected"):
                return row[0]
            await asyncio.sleep(.01)
        self.fail("Output receipt did not settle")

    async def replay(self, session_id):
        values = []
        route = "/api/game/updates"
        session = self.player
        async with session.get(self.url + route,
                               headers={"Last-Event-ID": session_id + ":0"}) as response:
            self.assertEqual(200, response.status)
            try:
                async with asyncio.timeout(.2):
                    async for line in response.content:
                        if line.startswith(b"data: "):
                            values.append(json.loads(line[6:]))
            except TimeoutError:
                pass
        return json.dumps(values)

    async def test_both_side_assignments_final_disposition_receipts_privacy_and_manual_rounds(self):
        for side in ("violet", "copper"):
            with self.subTest(gambit_side=side):
                configured = await self.setup_match(side)
                session_id = configured["sessionId"]
                runtime, connector, provider = self.runtime(side)
                first = await self.next_event(connector)
                observed = first.payload["observation"]
                self.assertEqual(side, observed["binding"]["sideId"])
                self.assertEqual(side, observed["objective"]["sideId"])
                self.assertNotIn("HUMAN_PRIVATE_HTTP_MARKER", json.dumps(observed))
                self.assertNotIn("drawPile", json.dumps(observed))
                self.assertGreater(len(observed["request"]["candidates"]) + int(observed["request"]["allowsNone"]), 1)
                await runtime.process(first)
                self.assertEqual("applied", await self.reconcile(runtime))
                queue = asyncio.Queue()
                saw_manual_boundary = False
                for _ in range(100):
                    await connector.poll_once(queue)
                    if not queue.empty():
                        await runtime.process(queue.get_nowait())
                        self.assertEqual("applied", await self.reconcile(runtime))
                        continue
                    current = await self.http("/api/game")
                    state = current["result"]["state"]
                    if state.get("completion"):
                        break
                    if state["roundComplete"]:
                        previous_round = state["round"]
                        await asyncio.sleep(.02)
                        same = await self.http("/api/game")
                        self.assertEqual(previous_round, same["result"]["state"]["round"])
                        self.assertTrue(same["result"]["state"]["roundComplete"])
                        saw_manual_boundary = True
                        provider.passive = False
                        await self.http("/api/game/round", {"expectedRevision": same["revision"]})
                    elif current["presentation"].get("decision"):
                        await self.human_pass(current, passive=not saw_manual_boundary)
                    else:
                        await asyncio.sleep(.01)
                else:
                    self.fail("Bounded canonical match did not complete")
                self.assertTrue(saw_manual_boundary)
                calls = len(provider.contexts)
                await connector.poll_once(queue)
                self.assertTrue(queue.empty())
                status = await connector._http("/status")
                self.assertIsNotNone(status["completion"])
                self.assertIsNone(status["request"])
                self.assertIsNone(await connector._http("/request"))
                self.assertEqual(calls, len(provider.contexts))
                self.assertTrue(all(not tools for tools in provider.tools))
                self.assertNotIn("HUMAN_PRIVATE_HTTP_MARKER", json.dumps(provider.contexts))
                player_replay = await self.replay(session_id)
                self.assertIn("HUMAN_PRIVATE_HTTP_MARKER", player_replay)
                self.assertIn("GAMBIT_PRIVATE_HTTP_MARKER", player_replay)
                row = runtime.store.connection.execute("SELECT id FROM output_requests LIMIT 1").fetchone()
                receipt = await connector._http("/receipts/" + row[0])
                self.assertEqual("applied", receipt["status"])

    async def test_reset_during_admitted_inference_rejects_stale_output_without_new_inference(self):
        await self.setup_match("copper")
        runtime, connector, provider = self.runtime("delayed", delayed=True)
        event = await self.next_event(connector)
        task = asyncio.create_task(runtime.process(event))
        try:
            await asyncio.wait_for(provider.entered.wait(), 5)
            current = await asyncio.wait_for(self.http("/api/game"), 2)
            old_session = current["sessionId"]
            reset = await self.http("/api/game/restart", {"expectedRevision": current["revision"]})
            self.assertNotEqual(old_session, reset["sessionId"])
            provider.release.set()
            await asyncio.wait_for(task, 5)
            self.assertEqual("rejected", await self.reconcile(runtime))
            stable = await self.http("/api/game")
            self.assertEqual(reset["revision"], stable["revision"])
            self.assertEqual(0, stable["result"]["state"]["round"])
            queue = asyncio.Queue()
            await connector.poll_once(queue)
            self.assertTrue(queue.empty())
            self.assertEqual(1, len(provider.contexts))
            self.assertEqual(provider.conversation, runtime.conversation_id)
        finally:
            provider.release.set()
            await task
