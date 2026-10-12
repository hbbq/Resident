"""Instance-private ProjectDelve decisions; no model-visible transport tools."""
from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any
from urllib.parse import quote, urlsplit
import aiohttp
from aiohttp.resolver import AsyncResolver

from .domain import WakeEvent
from .outputs import DeliveryPolicy, OutputCapability, output_schema
from .store import utc_now

BINDING_FIELDS = frozenset({"sessionId", "bindingId", "sideId", "decisionId", "revision",
                            "attemptId", "cursorStart", "cursorEnd"})
MAX_OBSERVATION_BYTES = 8 * 1024 * 1024


class ProjectDelveConnector:
    def __init__(self, base_url: str, bearer_token: str, *, binding_id: str = "gambit",
                 poll_seconds: float = 1, request_timeout_seconds: float = 10):
        url = urlsplit(base_url)
        if (url.scheme not in {"http", "https"} or not url.netloc or url.username
                or url.password or url.query or url.fragment):
            raise ValueError("ProjectDelve requires an absolute HTTP(S) service URL")
        if binding_id != "gambit":
            raise ValueError("ProjectDelve v1 supports only the gambit binding")
        if not bearer_token:
            raise ValueError("ProjectDelve controller token is required")
        self.base_url = base_url.rstrip("/")
        self._token = bearer_token
        self.binding_id = binding_id
        self.poll_seconds = poll_seconds
        self.timeout = request_timeout_seconds
        self.store = None
        self._queued: set[str] = set()

    def bind_store(self, store):
        self.store = store

    async def _http(self, route: str, payload: Any = None):
        headers = {"Authorization": "Bearer " + self._token, "Content-Type": "application/json"}
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        connector = aiohttp.TCPConnector(resolver=AsyncResolver())
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            async with session.request(
                    "GET" if payload is None else "POST",
                    self.base_url + "/api/controllers/" + self.binding_id + route,
                    json=payload, headers=headers, allow_redirects=False) as response:
                if response.status != 200:
                    raise ValueError("ProjectDelve controller transport unavailable")
                chunks, size = [], 0
                async for chunk in response.content.iter_chunked(65536):
                    size += len(chunk)
                    if size > MAX_OBSERVATION_BYTES:
                        raise ValueError("ProjectDelve observation exceeds size limit")
                    chunks.append(chunk)
                raw = b"".join(chunks)
                return json.loads(raw) if raw else None

    @property
    def output_capabilities(self):
        async def forbidden(_):
            raise PermissionError("Game output requires immutable dispatcher identity")
        return [OutputCapability(
            output_type="projectdelve_decision",
            description="Choose exactly one supplied legal ProjectDelve candidate or authorized decline.",
            payload_schema={"type": "object", "properties": {
                "candidateKey": {"type": ["string", "null"]},
                "commentary": {"type": ["string", "null"], "maxLength": 280}},
                "required": ["candidateKey", "commentary"], "additionalProperties": False},
            route_identity=self.base_url + "/controllers/" + self.binding_id,
            handler=forbidden, delivery_policy=DeliveryPolicy(max_attempts=2147483647))]

    def validate_binding(self, binding):
        if not isinstance(binding, dict) or set(binding) != BINDING_FIELDS:
            raise ValueError("Malformed ProjectDelve binding")
        if binding["bindingId"] != self.binding_id:
            raise ValueError("Unauthorized ProjectDelve binding")
        for field in ("sessionId", "sideId", "decisionId", "attemptId"):
            if not isinstance(binding[field], str) or not binding[field]:
                raise ValueError("Malformed ProjectDelve identity")
        for field in ("revision", "cursorStart", "cursorEnd"):
            if not isinstance(binding[field], int) or isinstance(binding[field], bool) or binding[field] < 0:
                raise ValueError("Malformed ProjectDelve revision/cursor")
        if binding["cursorStart"] > binding["cursorEnd"]:
            raise ValueError("Malformed ProjectDelve cursor interval")

    def decision_schema(self, event):
        observation = event.payload["observation"]
        self.validate_binding(observation["binding"])
        if (type(observation.get("schemaVersion")) is not int or observation["schemaVersion"] != 1
                or observation.get("completion") is not None or observation.get("status") != "waiting-external"):
            raise ValueError("Unsupported or completed ProjectDelve decision")
        objective = observation.get("objective")
        if (not isinstance(objective, dict) or objective.get("sideId") != observation["binding"]["sideId"]
                or not isinstance(objective.get("objective"), str)
                or not isinstance(observation.get("events"), list)
                or not isinstance(observation.get("observation"), dict)):
            raise ValueError("Malformed ProjectDelve player observation")
        request = observation["request"]
        candidates = request["candidates"]
        keys = [candidate["key"] for candidate in candidates]
        if not all(isinstance(key, str) and key for key in keys) or len(keys) != len(set(keys)):
            raise ValueError("Malformed ProjectDelve candidate keys")
        if not isinstance(request["allowsNone"], bool):
            raise ValueError("Malformed ProjectDelve decline permission")
        if request["allowsNone"]:
            keys.append(None)
        if not keys:
            raise ValueError("ProjectDelve decision has no legal choice")
        schema = output_schema(self.output_capabilities)
        schema["properties"]["outputs"].update(minItems=1, maxItems=1)
        if not request["allowsNone"]:
            schema["properties"]["outputs"]["items"]["anyOf"][0]["properties"]["candidateKey"]["type"] = "string"
        return schema

    def validate_choice(self, event, disposition):
        # Legality is checked against the immutable observation without enlarging
        # the provider's schema with arbitrarily many exact candidate strings.
        request = event.payload["observation"]["request"]
        output = disposition["outputs"][0]
        commentary = output["commentary"]
        if commentary is not None:
            try:
                length = len(commentary.strip().encode("utf-16-le")) // 2
            except UnicodeEncodeError:
                return "Commentary contains malformed Unicode"
            if length > 280:
                return "Commentary exceeds ProjectDelve's UTF-16 length limit"
        key = output["candidateKey"]
        if key is None:
            return None if request["allowsNone"] else "Decline is not legal"
        if key not in {candidate["key"] for candidate in request["candidates"]}:
            return "Candidate key is not legal"
        return None

    @staticmethod
    def wake_id(binding):
        encoded = json.dumps(binding, sort_keys=True, separators=(",", ":"))
        return "projectdelve:" + hashlib.sha256(encoded.encode()).hexdigest()

    def validate_status(self, status):
        if (not isinstance(status, dict) or type(status.get("schemaVersion")) is not int
                or status["schemaVersion"] != 1 or "completion" not in status or "request" not in status
                or not isinstance(status.get("status"), str)):
            raise ValueError("Malformed ProjectDelve status")
        if status["request"] is not None:
            self.validate_binding(status["request"])

    def release_queued(self, event):
        self._queued.discard(event.id)

    async def poll_once(self, queue):
        status = await self._http("/status")
        self.store.save_observed_snapshot("projectdelve.status", status)
        self.validate_status(status)
        if status.get("completion") is not None or status.get("request") is None:
            return
        observation = await self._http("/request")
        if observation is None:
            return
        binding = observation["binding"]
        self.validate_binding(binding)
        event_id = self.wake_id(binding)
        if (event_id in self._queued or self.store.completed_event_run(event_id)
                or self.store.projectdelve_finalized(event_id)):
            return
        event = WakeEvent(event_id, "projectdelve", "decision", utc_now(),
                          {"observation": observation,
                           "transportIdentity": self.output_capabilities[0].route_identity,
                           "transportFingerprint": self.output_capabilities[0].fingerprint})
        self.decision_schema(event)
        self._queued.add(event_id)
        await queue.put(event)

    def matches_transport(self, event):
        capability = self.output_capabilities[0]
        return (event.payload.get("transportIdentity") == capability.route_identity
                and event.payload.get("transportFingerprint") == capability.fingerprint)

    async def admit(self, event):
        self.decision_schema(event)
        if not self.matches_transport(event):
            raise PermissionError("ProjectDelve request belongs to a different configured transport")
        binding = event.payload["observation"]["binding"]
        status = await self._http("/status")
        self.validate_status(status)
        if status["completion"] is not None or status["request"] != binding:
            return False
        # Persist Side custody across all matches in this Conversation: one session
        # can never later expose the opposing Side's knowledge to the same player.
        scope = "projectdelve.side." + binding["sessionId"]
        previous = self.store.observed_snapshot(scope)
        if previous is not None and previous != binding["sideId"]:
            raise PermissionError("Opposing same-session Side binding is forbidden")
        response = await self._http("/admit", binding)
        if not isinstance(response, dict) or response.get("admitted") is not True or response.get("binding") != binding:
            return False
        self.store.save_observed_snapshot(scope, binding["sideId"])
        old = self.store.observed_snapshot("projectdelve.session")
        event.payload["newSession"] = old != binding["sessionId"]
        return True

    def record_seen_session(self, event):
        # Admission alone does not prove this match reached the Conversation.
        # Advance continuity only after recording a returned model response.
        self.store.save_observed_snapshot("projectdelve.session",
                                          event.payload["observation"]["binding"]["sessionId"])

    async def report_error(self, event):
        # Never forward model text, exception bodies or private reasoning to the game.
        try:
            await self._http("/error", {"binding": event.payload["observation"]["binding"],
                                        "error": "resident-decision-error"})
        except Exception:
            pass

    async def deliver(self, output):
        payload = output["payload"]
        self.validate_binding(payload["binding"])
        submission_id = output["id"]
        receipt = await self._http("/receipts/" + quote(submission_id, safe=""))
        self._validate_receipt(receipt, submission_id)
        send_scope = "projectdelve.send-started." + submission_id
        if receipt["status"] == "unknown" and self.store.observed_snapshot(send_scope) is None:
            # Persist the transmission boundary BEFORE POST. A failed receipt
            # read is safely retryable; crash/uncertainty after this boundary only
            # reconciles the ORIGINAL ID and never proves no effect from absence.
            self.store.save_observed_snapshot(send_scope, {"startedAt": utc_now()})
            receipt = await self._http("/selection", {"submissionId": submission_id, **payload})
            self._validate_receipt(receipt, submission_id)
        self.store.save_observed_snapshot("projectdelve.delivery." + submission_id,
                                          {"status": receipt["status"], "binding": payload["binding"]})
        if receipt["status"] in {"applied", "rejected"}:
            return receipt["status"]
        return "unknown" if receipt["status"] == "unknown" else "queued"

    @staticmethod
    def _validate_receipt(receipt, submission_id):
        if (not isinstance(receipt, dict) or receipt.get("submissionId") != submission_id
                or receipt.get("status") not in {"unknown", "queued", "applied", "rejected"}):
            raise ValueError("Malformed ProjectDelve receipt")

    async def run(self, queue, stop):
        while not stop.is_set():
            try:
                await self.poll_once(queue)
            except Exception:
                # Connector failures are local status, never model wakeups.
                self.store.save_observed_snapshot("projectdelve.status", {"available": False})
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.poll_seconds)
            except TimeoutError:
                pass
