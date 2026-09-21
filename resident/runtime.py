from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import Any, Awaitable, Callable, Protocol, Sequence

from .capabilities import Capability, diagnostic_capabilities
from .config import Config
from .context import ContextBuilder
from .domain import ToolResult, WakeEvent
from .provider import ModelProvider, RemoteSessionUnavailable
from .memory import MemoryCurator, SessionHistoryUnavailable
from .observability import EventLoopLagProbe, ObservedQueue, emit_timeline, timeline_reporter
from .outputs import (OutputCapability, capability_for_output, output_schema,
                      schema_fingerprint, validate_disposition)
from .readiness import ReadinessItem, ReadinessResult
from .store import Store, utc_now
from .tools import CORE_TOOL_NAMES, OwnerGuidanceAuthorization, ToolRegistry


_DEGRADED_HANDOVER = (
    "The previous remote session was unavailable, so its final working context could not be "
    "curated. Continue from the durable Resident identity, standing Owner guidance, pending "
    "intentions, and long-term-memory index in this bootstrap. Older communication remains "
    "available through bounded communication search."
)


class EventProducer(Protocol):
    async def run(self, queue: asyncio.Queue[WakeEvent], stop: asyncio.Event) -> None: ...


class OwnerTransport(Protocol):
    async def send_text(self, content: str) -> None: ...


class CallbackOwnerTransport:
    def __init__(self, callback: Callable[[str], None]):
        self.callback = callback

    async def send_text(self, content: str) -> None:
        self.callback(content)


class CuratorCoordinator:
    """One durable, coalescing routine-curation stream for a Resident."""

    def __init__(self, runtime: "ResidentRuntime"):
        self.runtime = runtime
        self._signal = asyncio.Event()
        self._stop = False
        self._task: asyncio.Task[None] | None = None
        self._run_lock = asyncio.Lock()

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop = False
            self._task = asyncio.create_task(self._run(), name="resident-curator")

    def signal(self) -> None:
        self._signal.set()

    async def barrier(self, *, final: bool = False) -> str | None:
        curator = self.runtime.curator
        if curator is None:
            return None
        async with self._run_lock:
            session_id = getattr(getattr(curator, "source", None), "session_id", None)
            handover = await curator.catch_up(final=final)
            if session_id:
                self.runtime.store.complete_curator_request("openai_agents", session_id)
            return handover

    async def stop(self, grace_seconds: float = 2.0) -> None:
        self._stop = True
        self._signal.set()
        task = self._task
        if task is None:
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=grace_seconds)
        except TimeoutError:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        finally:
            self._task = None

    def cancel(self) -> None:
        if self._task is not None:
            self._task.cancel()

    async def _run(self) -> None:
        while not self._stop:
            curator = self.runtime.curator
            session_id = (getattr(getattr(curator, "source", None), "session_id", None)
                          if curator else None)
            request = (self.runtime.store.curator_request("openai_agents", session_id)
                       if session_id else None)
            if request is None:
                self._signal.clear()
                await self._signal.wait()
                continue
            retry_at = request.get("next_retry_at")
            if retry_at:
                delay = max(0.0, (datetime.fromisoformat(retry_at) - datetime.now(UTC)).total_seconds())
                if delay:
                    self._signal.clear()
                    try:
                        await asyncio.wait_for(self._signal.wait(), timeout=delay)
                        continue
                    except TimeoutError:
                        pass
            target = request["target_turn_id"]
            attempt = self.runtime.store.start_curator_request(
                "openai_agents", session_id, target)
            if attempt is None:
                continue
            self.runtime._emit_background("curator.started", {"attempt": attempt})
            try:
                async with self._run_lock:
                    await curator.catch_up(
                        through_turn_id=target, session_id=session_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                retry_seconds = min(60.0, float(2 ** min(max(attempt - 1, 0), 6)))
                if self.runtime.store.retry_curator_request(
                        "openai_agents", session_id, target,
                        type(exc).__name__, retry_seconds):
                    event_type = "curator.degraded" if attempt > 1 else "curator.retry_scheduled"
                    self.runtime._emit_background(event_type, {
                        "attempt": attempt, "error_type": type(exc).__name__,
                        "retry_seconds": retry_seconds})
            else:
                if self.runtime.store.complete_curator_request(
                        "openai_agents", session_id, target):
                    self.runtime._emit_background("curator.caught_up", {"attempt": attempt})


class ResidentRuntime:
    _NORMAL_DIAGNOSTIC_EVENTS = frozenset({"communication.failed", "wake.failed"})

    def __init__(self, config: Config, provider: ModelProvider, *, store: Store | None = None,
                 capabilities: Sequence[Capability] | None = None,
                 output_capabilities: Sequence[OutputCapability] | None = None,
                 event_producers: list[EventProducer] | None = None,
                 owner_transport: OwnerTransport | None = None,
                 owner_output_enabled: bool | None = None,
                 owner_output: Callable[[str], None] | None = None,
                 diagnostic_output: Callable[[str], None] | None = None):
        self.config, self.provider = config, provider
        initial_capabilities = capabilities if capabilities is not None else diagnostic_capabilities()
        self._capabilities = self._validated_capabilities(initial_capabilities)
        self.store = store or Store(config.data_dir / "resident.sqlite3")
        self.resident, self.owner = self.store.provision(
            config.resident_name, config.owner_name, config.personality)
        bind_session_store = getattr(provider, "bind_session_store", None)
        if bind_session_store is not None:
            bind_session_store(
                lambda: self.store.agent_session_binding("openai_agents"),
                lambda session_id, agent_id, last_turn_id: self.store.save_agent_session_binding(
                    "openai_agents", session_id, agent_id, last_turn_id),
            )
        bind_action_store = getattr(provider, "bind_action_store", None)
        if bind_action_store is not None:
            bind_action_store(
                self.store.begin_agent_tool_action,
                self.store.complete_agent_tool_action,
            )
        bind_wake_store = getattr(provider, "bind_wake_submission_store", None)
        if bind_wake_store is not None:
            bind_wake_store(
                lambda session_id, wake_key: self.store.agent_wake_submission(
                    "openai_agents", session_id, wake_key),
                lambda session_id, wake_key, correlation:
                    self.store.mark_agent_wake_submission_attempted(
                        "openai_agents", session_id, wake_key, correlation,
                        wake_id=self._active_event.id if self._active_event else None,
                        wake_source=self._active_event.source if self._active_event else None,
                        wake_reason=self._active_event.reason if self._active_event else None),
                lambda session_id, wake_key, turn_id:
                    self.store.correlate_agent_wake_submission(
                        "openai_agents", session_id, wake_key, turn_id),
                lambda session_id, turn_id: self.store.settle_agent_wake_submission(
                    "openai_agents", session_id, turn_id),
                lambda session_id, wake_key: self.store.clear_agent_wake_submission(
                    "openai_agents", session_id, wake_key),
            )
        bind_lifecycle_store = getattr(provider, "bind_lifecycle_store", None)
        if bind_lifecycle_store is not None:
            self.store.recover_session_rollovers("openai_agents")
            bind_lifecycle_store(
                lambda session_id: self.store.session_protocol("openai_agents", session_id),
                lambda session_id, descriptor: self.store.save_session_protocol(
                    "openai_agents", session_id, descriptor),
                lambda session_id: self.store.session_mutable_settings(
                    "openai_agents", session_id),
                lambda session_id, settings: self.store.save_session_mutable_settings(
                    "openai_agents", session_id, settings),
                lambda: self.store.pending_session_rollover("openai_agents"),
                lambda old, reason, requested_by, request, protocol, mutable:
                    self.store.begin_session_rollover(
                        "openai_agents", old, reason, requested_by, request,
                        protocol, mutable),
                self.store.mark_session_rollover_create_started,
                self.store.bind_session_rollover,
                self.store.complete_session_rollover,
                self.store.fail_session_rollover,
                lambda session_id, agent_id, request, protocol, mutable:
                    self.store.bind_initial_agent_session(
                        "openai_agents", session_id, agent_id, request,
                        protocol, mutable),
            )
        deferred_request = self.store.pending_session_rollover_request("openai_agents")
        restored_deferred_request = False
        if deferred_request is not None and getattr(provider, "session_id", None) is not None:
            if deferred_request["old_session_id"] != provider.session_id:
                self.store.clear_session_rollover_request(
                    "openai_agents", deferred_request["old_session_id"])
            elif self.store.pending_session_rollover("openai_agents") is None:
                request_rollover = getattr(provider, "request_rollover", None)
                if request_rollover is not None:
                    request_rollover(deferred_request["reason"])
                    restored_deferred_request = True
        if config.new_chapter and not restored_deferred_request:
            request_rollover = getattr(provider, "request_rollover", None)
            if request_rollover is not None:
                request_rollover("explicit_new_chapter")
        self._event_queue: asyncio.Queue[WakeEvent | None] | None = None
        self._capability_event_states: dict[
            str, tuple[tuple[Capability, ...], dict[str, dict]]
        ] = {}
        current_snapshot = self._capability_snapshot(self._capabilities)
        persisted_snapshot = self.store.observed_snapshot("runtime.capabilities")
        if persisted_snapshot is None:
            self.store.save_observed_snapshot("runtime.capabilities", current_snapshot)
            self._observed_capability_snapshot = current_snapshot
            self._pending_capability_event = None
        else:
            self._observed_capability_snapshot = persisted_snapshot
            self._pending_capability_event = self._record_capability_change(self._capabilities)
        self.event_producers = event_producers or []
        default_output = lambda message: print(f"\n[{self.resident.address_name} -> {self.owner.address_name}] {message}")
        self.owner_output = owner_output or default_output
        self.owner_transport = owner_transport or CallbackOwnerTransport(self.owner_output)
        self._remote_owner_transport = owner_transport is not None
        self._mirror_owner_output = self.owner_output if owner_transport is not None else None
        self.diagnostic_output = diagnostic_output or (lambda message: print(f"[runtime] {message}"))
        self._output_protocol_enabled = (
            output_capabilities is not None or owner_transport is not None)
        configured_outputs = list(output_capabilities or ())
        if owner_output_enabled is None:
            owner_output_enabled = config.owner_communication_enabled
        if (self._output_protocol_enabled
                and getattr(provider, "supports_output_capabilities", False)
                and owner_output_enabled):
            async def notify_owner(payload: dict[str, Any]) -> dict[str, Any]:
                if self._mirror_owner_output is not None:
                    try:
                        self._mirror_owner_output(payload["content"])
                    except Exception:
                        pass
                await self.owner_transport.send_text(payload["content"])
                return {"status": "accepted_by_transport"}

            configured_outputs.insert(0, OutputCapability(
                output_type="notify_owner",
                description="Send a statement or question to the Owner after this turn completes.",
                payload_schema={
                    "type": "object",
                    "properties": {"content": {
                        "type": "string", "minLength": 1, "maxLength": 4096}},
                    "required": ["content"], "additionalProperties": False,
                },
                route_identity="owner", handler=notify_owner,
                legacy_tool_name="send_owner_message",
            ))
        self._output_capabilities = self._validated_output_capabilities(configured_outputs)
        self._output_schema = output_schema(self._output_capabilities)
        self._output_schema_fingerprint = schema_fingerprint(self._output_schema)
        configure_outputs = getattr(provider, "configure_output_protocol", None)
        if configure_outputs is not None and self._output_protocol_enabled:
            configure_outputs(
                self._output_schema,
                [capability.semantic_descriptor() for capability in self._output_capabilities],
                self._output_schema_fingerprint)
        self.context_builder = ContextBuilder(
            self.store, message_limit=config.context_messages,
            role=config.role)
        self._active_run_id: str | None = None
        self._active_event: WakeEvent | None = None
        self._owner_event_authorizations: dict[str, str] = {}
        self.curator: MemoryCurator | None = None
        self._curator_coordinator = CuratorCoordinator(self)
        self._enqueue_times: dict[str, float] = {}
        self._dequeue_observations: dict[str, tuple[float | None, int]] = {}
        self._active_max_loop_lag = 0.0
        self._active_total_loop_lag = 0.0
        self._active_loop_lag_samples = 0
        self._event_loop_lag_checkpoint: Callable[[], Awaitable[None]] | None = None

    def bind_curator(self, curator: MemoryCurator) -> None:
        self.curator = curator

    @property
    def capabilities(self) -> tuple[Capability, ...]:
        return self._capabilities

    @property
    def output_capabilities(self) -> tuple[OutputCapability, ...]:
        return self._output_capabilities

    @staticmethod
    def _validated_output_capabilities(
            capabilities: Sequence[OutputCapability]) -> tuple[OutputCapability, ...]:
        snapshot = tuple(capabilities)
        identities = [(item.output_type, item.target) for item in snapshot]
        if len(identities) != len(set(identities)):
            raise ValueError("Duplicate output capability type and target")
        for capability in snapshot:
            capability.semantic_descriptor()
            if capability.delivery_policy.max_attempts < 1:
                raise ValueError("Output delivery max_attempts must be positive")
        return snapshot

    def _uses_structured_output_protocol(self) -> bool:
        return bool(
            self._output_protocol_enabled
            and getattr(self.provider, "supports_output_capabilities", False)
            and (getattr(self.provider, "session_id", None) is None
                 or getattr(self.provider, "session_uses_output_capabilities", False)
                 or getattr(self.provider, "rollover_ready", False)))

    def _tool_capabilities_for_protocol(self, structured: bool) -> tuple[Capability, ...]:
        if not structured:
            return self._capabilities
        replaced = {item.legacy_tool_name for item in self._output_capabilities
                    if item.legacy_tool_name}
        return tuple(item for item in self._capabilities if item.name not in replaced)

    @staticmethod
    def _validated_capabilities(capabilities: Sequence[Capability]) -> tuple[Capability, ...]:
        snapshot = tuple(capabilities)
        names = [capability.name for capability in snapshot]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Duplicate capability name: {', '.join(duplicates)}")
        reserved = sorted(set(names) & CORE_TOOL_NAMES)
        if reserved:
            raise ValueError(f"Capability name conflicts with a core tool: {', '.join(reserved)}")
        for capability in snapshot:
            capability.public_descriptor()
        return snapshot

    @staticmethod
    def _capability_snapshot(capabilities: Sequence[Capability]) -> dict[str, dict]:
        return {capability.name: capability.public_descriptor() for capability in capabilities}

    def _record_capability_change(self, capabilities: Sequence[Capability]) -> WakeEvent | None:
        capability_view = tuple(capabilities)
        current = self._capability_snapshot(capability_view)
        previous = self._observed_capability_snapshot
        if previous == current:
            return None
        previous_names, current_names = set(previous), set(current)
        payload = {
            "added": sorted(current_names - previous_names),
            "removed": sorted(previous_names - current_names),
            "changed": sorted(
                name for name in previous_names & current_names
                if previous[name] != current[name]
            ),
        }
        event = WakeEvent(str(uuid.uuid4()), "runtime", "capabilities_changed", utc_now(), payload)
        self._observed_capability_snapshot = current
        self._capability_event_states[event.id] = (capability_view, current)
        return event

    def replace_capabilities(self, capabilities: Sequence[Capability]) -> WakeEvent | None:
        """Atomically replace visible capabilities and notify a running Resident."""
        replacement = self._validated_capabilities(capabilities)
        event = self._record_capability_change(replacement)
        self._capabilities = replacement
        if event is not None and self._event_queue is not None:
            self._event_queue.put_nowait(event)
        return event

    def register_capabilities(self, capabilities: Sequence[Capability]) -> WakeEvent | None:
        return self.replace_capabilities((*self._capabilities, *capabilities))

    def unregister_capabilities(self, names: Sequence[str]) -> WakeEvent | None:
        removed = set(names)
        return self.replace_capabilities(
            tuple(capability for capability in self._capabilities if capability.name not in removed))

    async def enqueue_startup_wakeups(self, queue: asyncio.Queue[WakeEvent]) -> None:
        try:
            await self.recover_missing_disposition()
        except Exception as exc:
            self._emit("wake.failed", {
                "error_type": type(exc).__name__, "phase": "disposition_recovery"})
        if self.curator is not None:
            token = timeline_reporter.set(self._timeline) if self.config.timeline else None
            try:
                await self.curator.catch_up()
                session_id = getattr(getattr(self.curator, "source", None), "session_id", None)
                if session_id:
                    self.store.complete_curator_request("openai_agents", session_id)
            except Exception as exc:
                self._emit("curator.failed", {"phase": "startup", "error_type": type(exc).__name__})
                binding = self.store.agent_session_binding("openai_agents")
                if binding is not None and binding.get("last_turn_id"):
                    self.store.request_curator_catch_up(
                        "openai_agents", binding["session_id"], binding["last_turn_id"])
            finally:
                if token is not None:
                    timeline_reporter.reset(token)
            self._curator_coordinator.start()
        for message in self.store.pending_owner_messages():
            event = self._owner_message_wake(
                message["id"], message["content"], message["created_at"])
            await queue.put(event)
        if self._pending_capability_event is not None:
            await queue.put(self._pending_capability_event)
            self._pending_capability_event = None

    def close(self) -> None:
        self._curator_coordinator.cancel()
        self.store.close()

    async def stop_background_services(self) -> None:
        await self._curator_coordinator.stop()

    def owner_message_event(self, content: str) -> WakeEvent:
        message_id = self.store.ingest_owner_message(self.owner.id, content)
        return self._owner_message_wake(message_id, content)

    def _owner_message_wake(self, message_id: str, content: str,
                            occurred_at: str | None = None) -> WakeEvent:
        event = WakeEvent(str(uuid.uuid4()), "owner", "owner_message",
                          occurred_at or utc_now(),
                          {"message_id": message_id, "content": content})
        self._owner_event_authorizations[event.id] = message_id
        return event

    def telegram_owner_message_event(self, bot_identity: str, update_id: int,
                                     content: str) -> WakeEvent | None:
        message_id = self.store.ingest_telegram_owner_message(
            bot_identity, update_id, self.owner.id, content)
        if message_id is None:
            return None
        return self._owner_message_wake(message_id, content)

    def _emit(self, event_type: str, data: dict) -> None:
        self.store.journal(event_type, data, self._active_run_id)
        if self.config.verbose or event_type in self._NORMAL_DIAGNOSTIC_EVENTS:
            details = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
            self.diagnostic_output(f"{event_type} {details}")

    def _emit_background(self, event_type: str, data: dict) -> None:
        self.store.journal(event_type, data)
        if self.config.verbose or event_type in self._NORMAL_DIAGNOSTIC_EVENTS:
            details = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
            self.diagnostic_output(f"{event_type} {details}")

    def observe_enqueue(self, event: WakeEvent, queue_depth: int) -> None:
        if not self.config.timeline:
            return
        self._enqueue_times[event.id] = time.monotonic()
        self.store.journal("timeline", {
            "operation": "host.enqueue", "moment": "finished",
            "event_id": event.id, "queue_depth": queue_depth,
        })

    def observe_dequeue(self, event: WakeEvent, queue_depth: int) -> None:
        if self.config.timeline:
            self._dequeue_observations[event.id] = (
                self._enqueue_times.pop(event.id, None), queue_depth)

    def observe_event_loop_lag(self, lag_seconds: float) -> None:
        if self.config.timeline and self._active_run_id is not None:
            self._active_max_loop_lag = max(self._active_max_loop_lag, lag_seconds)
            self._active_total_loop_lag += lag_seconds
            self._active_loop_lag_samples += 1

    def bind_event_loop_lag_checkpoint(
            self, checkpoint: Callable[[], Awaitable[None]] | None) -> None:
        self._event_loop_lag_checkpoint = checkpoint

    def observe_external_timeline(self, data: dict) -> None:
        if not self.config.timeline:
            return
        self.store.journal("timeline", data)
        if self.config.verbose:
            details = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
            self.diagnostic_output(f"timeline {details}")

    def _timeline(self, data: dict) -> None:
        if self.config.timeline:
            self._emit("timeline", data)

    async def _send_owner_message(self, content: str) -> dict:
        if not content.strip():
            return {"delivered": False, "reason": "Message content is empty"}
        immediate_response = self._active_event is not None and self._active_event.source == "owner"
        spontaneous = not immediate_response
        allowed = True
        if spontaneous:
            since = (datetime.now(UTC) - timedelta(
                seconds=self.config.spontaneous_message_window_seconds)).isoformat()
            allowed = self.store.spontaneous_count_since(since) < self.config.spontaneous_message_limit
        status = "pending_delivery" if allowed else "rejected_attention_budget"
        message_id = self.store.add_message(
            "outbound", self.resident.id, content, spontaneous=spontaneous, delivery_status=status)
        result = {"message_id": message_id, "delivered": False, "spontaneous": spontaneous}
        if allowed:
            if self._mirror_owner_output is not None:
                try:
                    self._mirror_owner_output(content)
                except Exception:
                    pass
            try:
                await self.owner_transport.send_text(content)
            except Exception as exc:
                self.store.update_message_delivery_status(message_id, "transport_failed")
                result.update({
                    "delivered": False,
                    "reason": f"Owner transport failed: {type(exc).__name__}: {exc}",
                })
                self._emit("communication.failed", result)
                return result
            self.store.update_message_delivery_status(message_id, "delivered")
            result["delivered"] = True
            self._emit("communication.delivered", result)
        else:
            result["reason"] = "Spontaneous owner-message attention budget exceeded"
            self._emit("communication.rejected", result)
        return result

    def _active_output_protocol(self) -> tuple[dict[str, Any], str]:
        protocol = getattr(self.provider, "active_output_protocol", None)
        if isinstance(protocol, dict):
            schema = protocol.get("schema")
            fingerprint = protocol.get("fingerprint")
            if isinstance(schema, dict) and isinstance(fingerprint, str):
                return schema, fingerprint
        return self._output_schema, self._output_schema_fingerprint

    def _persist_disposition(
            self, raw: str | None, session_id: str, turn_id: str, *,
            run_id: str | None, wake: WakeEvent | None,
            schema: dict[str, Any] | None = None,
            fingerprint: str | None = None) -> dict[str, Any]:
        schema = schema or self._output_schema
        fingerprint = fingerprint or schema_fingerprint(schema)
        normalized: dict[str, Any] | None = None
        validation_state = "valid"
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else None
        except json.JSONDecodeError:
            parsed = None
            validation_state = "invalid_json"
        if validation_state == "valid":
            error = validate_disposition(parsed, schema)
            if error:
                validation_state = "schema_invalid"
            else:
                normalized = parsed

        jobs: list[dict[str, Any]] = []
        spontaneous = not (wake is not None and wake.source == "owner")
        attention_used = 0
        if normalized is not None:
            if spontaneous:
                since = (datetime.now(UTC) - timedelta(
                    seconds=self.config.spontaneous_message_window_seconds)).isoformat()
                attention_used = self.store.spontaneous_attention_count_since(since)
            for output in normalized["outputs"]:
                capability = capability_for_output(output, self._output_capabilities)
                current_error = (None if capability is None else
                                 validate_disposition(
                                     {"outputs": [output]}, output_schema([capability])))
                state = "queued"
                classification = None
                if capability is None:
                    state, classification = "rejected_unavailable", "capability_revoked"
                elif current_error:
                    state, classification = "rejected_policy", "current_policy_rejected"
                elif output["type"] == "notify_owner" and spontaneous:
                    if attention_used >= self.config.spontaneous_message_limit:
                        state, classification = "rejected_policy", "attention_budget"
                    else:
                        attention_used += 1
                payload = {key: value for key, value in output.items()
                           if key not in {"type", "target"}}
                jobs.append({
                    "output_type": output["type"], "target": output.get("target"),
                    "payload": payload,
                    "route_identity": capability.route_identity if capability else None,
                    "capability_fingerprint": capability.fingerprint if capability else None,
                    "max_attempts": (
                        capability.delivery_policy.max_attempts if capability else 1),
                    "delivery_state": state, "failure_classification": classification,
                    "sender_id": self.resident.id, "spontaneous": spontaneous,
                    "message_status": (
                        "rejected_attention_budget" if classification == "attention_budget"
                        else "pending_delivery" if state == "queued"
                        else state),
                    "suppress_failure_event": bool(
                        wake is not None and wake.reason == "output_delivery_failed"),
                })
        receipt = self.store.persist_final_disposition(
            "openai_agents", session_id, turn_id, fingerprint, raw, normalized,
            validation_state, jobs, run_id=run_id, wake_id=wake.id if wake else None)
        if receipt["created"]:
            self._emit("disposition.generated", {
                "disposition_id": receipt["id"], "turn_id": turn_id,
                "validation_state": validation_state,
                "output_count": len(normalized["outputs"]) if normalized else 0,
            })
            for request in receipt["requests"]:
                if request["delivery_state"] == "queued":
                    self._emit("output.queued", {
                        "output_id": request["id"], "output_type": request["output_type"],
                        "target": request.get("target")})
                else:
                    self._emit("output.rejected", {
                        "output_id": request["id"], "output_type": request["output_type"],
                        "target": request.get("target"),
                        "classification": request.get("failure_classification")})
        if validation_state != "valid":
            raise RuntimeError(f"Managed Agents final disposition is {validation_state}")
        return receipt

    async def recover_missing_disposition(self) -> bool:
        if not getattr(self.provider, "supports_output_capabilities", False):
            return False
        binding = self.store.agent_session_binding("openai_agents")
        if binding is None or not binding.get("last_turn_id"):
            return False
        session_id, turn_id = binding["session_id"], binding["last_turn_id"]
        protocol = self.store.session_protocol("openai_agents", session_id)
        if not protocol or not protocol.get("output_schema_fingerprint"):
            return False
        if self.store.has_final_disposition("openai_agents", session_id, turn_id):
            return False
        recover = getattr(self.provider, "recover_final_output", None)
        if recover is None:
            return False
        raw = await recover(session_id, turn_id)
        wake_context = self.store.disposition_wake_context(
            "openai_agents", session_id, turn_id)
        wake = (None if wake_context is None else WakeEvent(
            wake_context.get("wake_id") or f"recovered:{turn_id}",
            wake_context["wake_source"], wake_context["wake_reason"], utc_now(), {}))
        self._persist_disposition(
            raw, session_id, turn_id, run_id=None, wake=wake,
            schema=protocol["output_schema"],
            fingerprint=protocol["output_schema_fingerprint"])
        return True

    async def dispatch_outputs_once(self) -> bool:
        request = self.store.claim_output_request()
        if request is None:
            return False
        output_id, attempt = request["id"], request["attempt_count"]
        self._emit("output.delivery_attempted", {
            "output_id": output_id, "output_type": request["output_type"],
            "target": request.get("target"), "attempt": attempt})
        capability = next((item for item in self._output_capabilities
                           if item.route_identity == request.get("route_identity")
                           and item.fingerprint == request.get("capability_fingerprint")), None)
        if capability is None:
            self.store.finish_output_attempt(
                output_id, attempt, "rejected_unavailable",
                classification="capability_unavailable")
            self._record_terminal_output_failure(request, "capability_unavailable", attempt)
            return True
        try:
            result = await capability.handler(request["payload"])
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            from .telegram import TelegramPermanentTransportError
            permanent = isinstance(exc, (TelegramPermanentTransportError,
                                         ValueError, PermissionError))
            if not permanent and attempt < capability.delivery_policy.max_attempts:
                delays = capability.delivery_policy.retry_delays_seconds
                delay = delays[min(attempt - 1, len(delays) - 1)] if delays else 1.0
                self.store.finish_output_attempt(
                    output_id, attempt, "retry_wait",
                    classification=type(exc).__name__, retry_delay_seconds=delay)
                self._emit("output.delivery_failed", {
                    "output_id": output_id, "output_type": request["output_type"],
                    "target": request.get("target"), "attempt": attempt,
                    "classification": type(exc).__name__})
            else:
                classification = (type(exc).__name__ if permanent else "retries_exhausted")
                self.store.finish_output_attempt(
                    output_id, attempt, "failed_permanent", classification=classification)
                self._record_terminal_output_failure(request, classification, attempt)
            return True
        external_id = result.get("external_message_id") if isinstance(result, dict) else None
        self.store.finish_output_attempt(
            output_id, attempt, "accepted_by_transport", external_message_id=external_id)
        self._emit("output.delivery_succeeded", {
            "output_id": output_id, "output_type": request["output_type"],
            "target": request.get("target"), "attempt": attempt})
        return True

    def _record_terminal_output_failure(
            self, request: dict[str, Any], classification: str, attempt: int) -> None:
        self._emit("output.delivery_failed", {
            "output_id": request["id"], "output_type": request["output_type"],
            "target": request.get("target"), "attempt": attempt,
            "classification": classification})
        if self.store.generate_output_failure_event(request["id"]):
            self._emit("output.failure_event_generated", {
                "output_id": request["id"], "output_type": request["output_type"],
                "target": request.get("target"), "classification": classification,
                "attempt_count": attempt})

    async def output_dispatcher_loop(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            worked = await self.dispatch_outputs_once()
            if worked:
                continue
            try:
                await asyncio.wait_for(stop.wait(), timeout=0.25)
            except TimeoutError:
                pass

    def _discard_continuation(self, continuation_id: str | None) -> None:
        if not continuation_id:
            return
        discard = getattr(self.provider, "discard_continuation", None)
        if discard is None:
            return
        try:
            discard(continuation_id)
        except Exception:
            pass

    def _authoritative_state_update(self, session_id: str,
                                    current: dict) -> dict | None:
        """Describe only externally owned state not yet synchronized to this session."""
        previous = self.store.session_authoritative_state("openai_agents", session_id)
        if previous is None:
            return {"mode": "replace", **current}

        update: dict = {"mode": "delta"}
        for key in ("resident", "owner", "available_connectors"):
            if previous.get(key) != current.get(key):
                update[key] = current[key]

        old_guidance = {
            entry["id"]: entry for entry in previous.get("standing_owner_guidance", [])
        }
        new_guidance = {
            entry["id"]: entry for entry in current.get("standing_owner_guidance", [])
        }
        changed = [entry for guidance_id, entry in new_guidance.items()
                   if old_guidance.get(guidance_id) != entry]
        removed = [{
            "id": guidance_id,
            "revision": self.store.owner_guidance_revision(guidance_id),
        } for guidance_id in old_guidance.keys() - new_guidance.keys()]
        if changed or removed:
            update["standing_owner_guidance"] = {
                "set": changed,
                "removed": removed,
            }
        return update if len(update) > 1 else None

    async def process(self, event: WakeEvent) -> str:
        started = time.monotonic()
        run_id = self.store.start_run(event)
        self._active_run_id, self._active_event = run_id, event
        self._active_max_loop_lag = 0.0
        self._active_total_loop_lag = 0.0
        self._active_loop_lag_samples = 0
        calls = 0
        status = "failed"
        continuation_id: str | None = None
        capability_event_state = self._capability_event_states.get(event.id)
        timeline_token = None
        if self.config.timeline:
            timeline_token = timeline_reporter.set(self._timeline)
            queued_at, queue_depth = self._dequeue_observations.pop(
                event.id, (None, 0))
            self._timeline({
                "operation": "host.dequeue", "moment": "finished",
                "event_id": event.id, "queue_depth": queue_depth,
                "queue_wait_seconds": (None if queued_at is None else started - queued_at),
            })
            self._timeline({"operation": "wake.process", "moment": "started"})
        try:
            self._emit("wake.started", {
                "event_id": event.id, "source": event.source, "reason": event.reason,
                "occurred_at": event.occurred_at, "payload": event.payload,
            })
            configured_capabilities = (
                capability_event_state[0] if capability_event_state else self._capabilities)
            preflight_session = getattr(self.provider, "preflight_session", None)
            emit_timeline("provider.preflight", "started")
            preflight_started = time.monotonic()
            try:
                unavailable_reason = (
                    await preflight_session() if preflight_session is not None else None)
            finally:
                emit_timeline("provider.preflight", "finished",
                              duration_seconds=time.monotonic() - preflight_started)
            managed_session = bool(getattr(self.provider, "uses_managed_session", False))
            structured_outputs = self._uses_structured_output_protocol()
            capabilities = (self._tool_capabilities_for_protocol(structured_outputs)
                            if configured_capabilities is self._capabilities else
                            tuple(item for item in configured_capabilities
                                  if not structured_outputs or item.name not in {
                                      output.legacy_tool_name
                                      for output in self._output_capabilities
                                      if output.legacy_tool_name}))
            authoritative_state = self.context_builder.authoritative_state(
                self.resident, self.owner, capabilities)
            existing_session_id = getattr(self.provider, "session_id", None)
            if managed_session:
                authoritative_update = (
                    self._authoritative_state_update(existing_session_id, authoritative_state)
                    if existing_session_id is not None else None)
                context = self.context_builder.build_managed_wake(
                    event, authoritative_update=authoritative_update)
            else:
                context = self.context_builder.build(
                    self.resident, self.owner, event, capabilities)
            self._emit("context.assembled", {
                "characters": len(context),
                "pending_intentions": (
                    0 if managed_session and existing_session_id is not None
                    else len(self.store.pending_intentions())),
                "recent_messages": (
                    0 if managed_session
                    else len(self.store.recent_messages(self.config.context_messages))),
            })
            owner_guidance_authorization = None
            if event.source == "owner" and event.reason == "owner_message":
                message_id = event.payload.get("message_id")
                if (self._owner_event_authorizations.get(event.id) == message_id
                        and self.store.is_pending_owner_message(message_id, self.owner.id)):
                    owner_guidance_authorization = OwnerGuidanceAuthorization(
                        message_id, getattr(self.provider, "session_id", None))
            registry = ToolRegistry(
                self.store, capabilities, self._send_owner_message, self._emit,
                current_run_id=run_id,
                owner_communication_enabled=(
                    self.config.owner_communication_enabled and not structured_outputs),
                owner_guidance_authorization=owner_guidance_authorization)
            protocol_rollover = getattr(self.provider, "protocol_change_requires_rollover", None)
            new_session = getattr(self.provider, "session_id", None) is None
            handover = None
            handover_id = None
            unknown_restored_protocol = (
                getattr(self.provider, "session_id", None) is not None and
                not getattr(self.provider, "session_protocol_known", True))
            needs_rollover = (
                protocol_rollover is not None and protocol_rollover(registry.specs)
            ) or unknown_restored_protocol
            rollover_ready = bool(getattr(self.provider, "rollover_ready", True))
            if needs_rollover and (rollover_ready or unavailable_reason is not None):
                old_session_id = getattr(self.provider, "session_id", None)
                pending_rollover = self.store.pending_session_rollover("openai_agents")
                if old_session_id:
                    rollover_reason = (
                        pending_rollover["reason"] if pending_rollover is not None else
                        getattr(self.provider, "requested_rollover_reason", None) or
                        "function_or_immutable_protocol_changed")
                    self.store.request_session_rollover(
                        "openai_agents", old_session_id, rollover_reason)
                pending_handover = self.store.pending_handover(old_session_id) if (
                    old_session_id and pending_rollover is not None) else None
                if pending_rollover is not None and pending_handover is not None:
                    # This handover is already final for a durable create snapshot.
                    handover = pending_handover["content"]
                    handover_id = pending_handover["id"]
                elif self.curator is not None and unavailable_reason != "remote_session_missing":
                    try:
                        handover = await self._curator_coordinator.barrier(final=True)
                    except SessionHistoryUnavailable as exc:
                        self._emit("curator.failed", {
                            "phase": "final", "error_type": type(exc).__name__,
                            "history_status": "unavailable"})
                        handover = _DEGRADED_HANDOVER
                    except Exception as exc:
                        self._emit("curator.failed", {
                            "phase": "final", "error_type": type(exc).__name__,
                            "history_status": "reachable"})
                        raise
                elif unavailable_reason is not None:
                    handover = _DEGRADED_HANDOVER
                confirm_rollover = getattr(self.provider, "confirm_rollover_ready", None)
                confirmed = (unavailable_reason is not None or confirm_rollover is None
                             or await confirm_rollover())
                if confirmed:
                    new_session = True
                    if old_session_id and handover and handover_id is None:
                        handover_id = self.store.create_handover(
                            old_session_id, handover,
                            (datetime.now(UTC) + timedelta(hours=24)).isoformat())
                else:
                    handover = None
            if new_session:
                if managed_session:
                    context = self.context_builder.build_managed_bootstrap(
                        self.resident, self.owner, event, capabilities, handover=handover)
                else:
                    context_document = json.loads(context)
                    context_document["new_session_bootstrap"] = {
                        "durable_memory_awareness": self.store.memory_awareness(limit=8),
                        "handover": handover,
                        "note": "Long-term memory is selectively available through memory tools.",
                    }
                    context = json.dumps(context_document, ensure_ascii=False, indent=2)
            results: list[ToolResult] = []
            for round_number in range(self.config.max_tool_rounds + 1):
                calls += 1
                response_session_id = getattr(self.provider, "session_id", None)
                provider_operation = ("provider.tool_result_continuation" if results
                                      else "provider.turn")
                provider_started = time.monotonic()
                emit_timeline(provider_operation, "started", round=round_number,
                              tool_result_count=len(results), turn_id=continuation_id)
                try:
                    turn = await self.provider.respond(
                        context, registry.specs, results, continuation_id)
                except RemoteSessionUnavailable:
                    if results or continuation_id is not None:
                        raise
                    old_session_id = getattr(self.provider, "session_id", None)
                    unavailable_reason = getattr(
                        self.provider, "unavailable_session_reason", None)
                    if old_session_id:
                        self.store.request_session_rollover(
                            "openai_agents", old_session_id,
                            unavailable_reason or "remote_session_unavailable")
                    if (self.curator is not None
                            and unavailable_reason != "remote_session_missing"):
                        try:
                            handover = await self._curator_coordinator.barrier(final=True)
                        except SessionHistoryUnavailable as exc:
                            self._emit("curator.failed", {
                                "phase": "final", "error_type": type(exc).__name__,
                                "history_status": "unavailable"})
                            handover = _DEGRADED_HANDOVER
                        except Exception as exc:
                            self._emit("curator.failed", {
                                "phase": "final", "error_type": type(exc).__name__,
                                "history_status": "reachable"})
                            raise
                    else:
                        handover = _DEGRADED_HANDOVER
                    if old_session_id:
                        handover_id = self.store.create_handover(
                            old_session_id, handover,
                            (datetime.now(UTC) + timedelta(hours=24)).isoformat())
                    context = self.context_builder.build_managed_bootstrap(
                        self.resident, self.owner, event, capabilities, handover=handover)
                    turn = await self.provider.respond(
                        context, registry.specs, results, continuation_id)
                finally:
                    emit_timeline(
                        provider_operation, "finished", round=round_number,
                        tool_result_count=len(results), turn_id=continuation_id,
                        outcome="error" if sys.exc_info()[0] is not None else "ok",
                        duration_seconds=time.monotonic() - provider_started)
                if handover_id is not None:
                    replacement_id = getattr(self.provider, "session_id", None)
                    if replacement_id and replacement_id != response_session_id:
                        self.store.consume_handover(handover_id, replacement_id)
                        handover_id = None
                if (response_session_id is not None
                        and getattr(self.provider, "session_id", None) != response_session_id):
                    self.store.clear_session_rollover_request(
                        "openai_agents", response_session_id)
                self._emit("model.responded", {
                    "response_id": turn.response_id, "tool_call_count": len(turn.tool_calls),
                    "has_message": bool(turn.message), "input_tokens": turn.input_tokens,
                    "output_tokens": turn.output_tokens,
                })
                active_structured_outputs = bool(
                    getattr(self.provider, "session_uses_output_capabilities", False))
                if turn.message and not active_structured_outputs:
                    self._emit("model.message", {"content": turn.message})
                if not turn.tool_calls:
                    if active_structured_outputs:
                        session_id = getattr(self.provider, "session_id", None)
                        if not session_id or not turn.response_id:
                            raise RuntimeError(
                                "Structured final disposition lacks session or turn identity")
                        schema, fingerprint = self._active_output_protocol()
                        self._persist_disposition(
                            turn.message, session_id, turn.response_id,
                            run_id=run_id, wake=event, schema=schema,
                            fingerprint=fingerprint)
                    continuation_id = None
                    break
                continuation_id = turn.response_id
                if round_number >= self.config.max_tool_rounds:
                    self._discard_continuation(continuation_id)
                    raise RuntimeError("Model exceeded the configured tool-round limit")
                results = []
                for call in turn.tool_calls:
                    self._emit("tool.called", {"call_id": call.id, "name": call.name, "arguments": call.arguments})
                    tool_started = time.monotonic()
                    emit_timeline("tool.execute", "started", call_id=call.id,
                                  tool_name=call.name, round=round_number)
                    tool_outcome = "error"
                    try:
                        prepare = getattr(self.provider, "prepare_tool_call", None)
                        action = prepare(call) if prepare is not None else None
                        if action is not None and not action["claimed"]:
                            ephemeral_result = action.get("ephemeral_result")
                            if ephemeral_result is not None:
                                result = ephemeral_result
                            elif action.get("attachments_ephemeral"):
                                # Attachment payloads (for example camera frames) are
                                # intentionally not persisted. Reacquire them after a
                                # restart instead of submitting an incomplete replay.
                                execution = await registry.execute(call.name, call.arguments)
                                result = ToolResult(call.id, execution.output, execution.attachments)
                                record = getattr(self.provider, "record_tool_result", None)
                                if record is not None:
                                    record(result)
                            else:
                                output = action["output"] or {
                                    "ok": False,
                                    "error": "Previous local action outcome is unknown; action was not repeated",
                                }
                                result = ToolResult(call.id, output)
                        else:
                            execution = await registry.execute(call.name, call.arguments)
                            result = ToolResult(call.id, execution.output, execution.attachments)
                            record = getattr(self.provider, "record_tool_result", None)
                            if record is not None:
                                record(result)
                        results.append(result)
                        completion = {"call_id": call.id, "name": call.name, "result": result.output}
                        if result.attachments:
                            completion["attachments"] = [{
                                "type": "image", "mime_type": attachment.mime_type,
                                "byte_count": len(attachment.data), "ephemeral": True,
                            } for attachment in result.attachments]
                        self._emit("tool.completed", completion)
                        tool_outcome = (
                            "ok" if result.output.get("ok") is not False else "error")
                    finally:
                        emit_timeline(
                            "tool.execute", "finished", call_id=call.id,
                            tool_name=call.name, round=round_number,
                            outcome=tool_outcome,
                            duration_seconds=time.monotonic() - tool_started)
                if not continuation_id:
                    raise RuntimeError("Provider did not return a response id for tool continuation")
            if capability_event_state is not None:
                self.store.save_observed_snapshot("runtime.capabilities", capability_event_state[1])
                self._capability_event_states.pop(event.id, None)
            if managed_session:
                synchronized_session_id = getattr(self.provider, "session_id", None)
                if synchronized_session_id is not None:
                    # Tool calls in this completed turn also made their durable
                    # state changes visible through the managed session history.
                    synchronized_state = self.context_builder.authoritative_state(
                        self.resident, self.owner, capabilities)
                    self.store.save_session_authoritative_state(
                        "openai_agents", synchronized_session_id, synchronized_state)
            self._emit("wake.sleeping", {"status": "completed"})
            status = "completed"
            if self.curator is not None and managed_session:
                session_id = getattr(self.provider, "session_id", None)
                completed_turn_id = turn.response_id
                if session_id and completed_turn_id:
                    self.store.request_curator_catch_up(
                        "openai_agents", session_id, completed_turn_id)
                    self._emit("curator.requested", {"status": "pending"})
                    self._curator_coordinator.signal()
            return run_id
        except asyncio.CancelledError as exc:
            self._discard_continuation(continuation_id)
            self._emit("wake.failed", {"error_type": type(exc).__name__, "error": str(exc)})
            raise
        except Exception as exc:
            self._discard_continuation(continuation_id)
            self._emit("wake.failed", {"error_type": type(exc).__name__, "error": str(exc)})
            raise
        finally:
            duration = time.monotonic() - started
            self._emit("wake.finished", {
                "status": status, "duration_seconds": duration, "model_calls": calls,
            })
            schedule_id = event.payload.get("schedule_id") if event.source == "scheduler" else None
            owner_message_id = (
                event.payload.get("message_id")
                if (event.source == "owner" and event.reason == "owner_message"
                    and self._owner_event_authorizations.get(event.id)
                    == event.payload.get("message_id")) else None
            )
            self.store.finish_run(
                run_id, status, duration, calls, schedule_id, owner_message_id)
            # Finalization above is synchronous. Let the probe account for an
            # overdue sample before publishing and clearing this wake's summary.
            if self._event_loop_lag_checkpoint is not None:
                await self._event_loop_lag_checkpoint()
            emit_timeline(
                "wake.process", "finished", outcome=status,
                duration_seconds=time.monotonic() - started,
                max_event_loop_lag_seconds=self._active_max_loop_lag)
            # The final wake timeline write is synchronous too. A second explicit
            # checkpoint ensures an overdue probe cannot lose that blocking.
            if self._event_loop_lag_checkpoint is not None:
                await self._event_loop_lag_checkpoint()
            emit_timeline(
                "event_loop.lag", "summary",
                sample_count=self._active_loop_lag_samples,
                max_event_loop_lag_seconds=self._active_max_loop_lag,
                average_event_loop_lag_seconds=(
                    self._active_total_loop_lag / self._active_loop_lag_samples
                    if self._active_loop_lag_samples else 0.0))
            self._owner_event_authorizations.pop(event.id, None)
            self._active_run_id, self._active_event = None, None
            self._active_max_loop_lag = 0.0
            self._active_total_loop_lag = 0.0
            self._active_loop_lag_samples = 0
            if timeline_token is not None:
                timeline_reporter.reset(timeline_token)

    async def enqueue_due_wakeups(self, queue: asyncio.Queue[WakeEvent]) -> None:
        for scheduled in self.store.claim_due_wakeups(utc_now()):
            event = WakeEvent(
                str(uuid.uuid4()), "scheduler", scheduled["reason"], utc_now(),
                {"schedule_id": scheduled["id"], "scheduled_for": scheduled["due_at"],
                 "context": scheduled["context"]})
            await queue.put(event)

    async def scheduler_loop(self, queue: asyncio.Queue[WakeEvent], stop: asyncio.Event) -> None:
        while not stop.is_set():
            await self.enqueue_due_wakeups(queue)
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.config.scheduler_poll_seconds)
            except TimeoutError:
                pass

    async def _collect_startup_readiness(
            self, queue: asyncio.Queue[WakeEvent], stop: asyncio.Event,
    ) -> tuple[list[asyncio.Task[None]], list[tuple[ReadinessItem, ReadinessResult]]]:
        readiness: asyncio.Queue[ReadinessResult] = asyncio.Queue()
        ordered: list[ReadinessItem] = []
        task_items: dict[asyncio.Task[None], tuple[ReadinessItem, ...]] = {}
        producer_tasks: list[asyncio.Task[None]] = []
        for producer in self.event_producers:
            items = tuple(getattr(producer, "readiness_items", ()))
            ordered.extend(items)
            if items:
                task = asyncio.create_task(producer.run(queue, stop, readiness))
                task_items[task] = items
            else:
                task = asyncio.create_task(producer.run(queue, stop))
            producer_tasks.append(task)

        keys = [item.key for item in ordered]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate startup readiness key")

        results: dict[str, ReadinessResult] = {}
        watched = set(task_items)
        while len(results) < len(ordered):
            receiver = asyncio.create_task(readiness.get())
            done, _ = await asyncio.wait(
                {receiver, *watched}, return_when=asyncio.FIRST_COMPLETED)
            if receiver in done:
                result = receiver.result()
                if result.key in keys and result.key not in results:
                    results[result.key] = result
            else:
                receiver.cancel()
                with suppress(asyncio.CancelledError):
                    await receiver
            for task in done - {receiver}:
                watched.discard(task)
                for item in task_items[task]:
                    results.setdefault(item.key, ReadinessResult(item.key, False))

        return producer_tasks, [(item, results[item.key]) for item in ordered]

    def _render_startup_readiness(
            self, results: list[tuple[ReadinessItem, ReadinessResult]]) -> None:
        for item, result in results:
            status = "OK" if result.ok else "FAILED"
            detail = f" ({result.detail})" if result.detail else ""
            self.diagnostic_output(f"{item.label:.<20} {status}{detail}")
        self.diagnostic_output(
            "All systems GO" if all(result.ok for _, result in results)
            else "Startup completed with connector errors.")

    async def run_interactive(self) -> None:
        queue: asyncio.Queue[WakeEvent | None] = ObservedQueue(
            lambda item, depth: self.observe_enqueue(item, depth)
            if item is not None else None)
        self._event_queue = queue
        stop = asyncio.Event()
        probe = EventLoopLagProbe((self.observe_event_loop_lag,))
        await probe.start()
        self.bind_event_loop_lag_checkpoint(probe.checkpoint)
        scheduler: asyncio.Task[None] | None = None
        dispatcher: asyncio.Task[None] | None = None
        terminal: asyncio.Task[None] | None = None
        producers: list[asyncio.Task[None]] = []

        async def terminal_input() -> None:
            while not stop.is_set():
                try:
                    text = await asyncio.to_thread(input, f"{self.owner.address_name}> ")
                except (EOFError, KeyboardInterrupt):
                    if self._remote_owner_transport:
                        return
                    text = "/quit"
                if text.strip() == "/quit":
                    stop.set()
                    await queue.put(None)
                    return
                if text.strip():
                    event = self.owner_message_event(text)
                    await queue.put(event)

        try:
            await self.enqueue_startup_wakeups(queue)
            scheduler = asyncio.create_task(self.scheduler_loop(queue, stop))
            dispatcher = asyncio.create_task(self.output_dispatcher_loop(stop))
            producers, readiness = await self._collect_startup_readiness(queue, stop)
            self._render_startup_readiness(readiness)
            terminal = asyncio.create_task(terminal_input())
            self.diagnostic_output(
                f"Resident {self.resident.address_name} ({self.resident.id}) is sleeping; /quit stops the process")
            while True:
                event = await queue.get()
                if event is None:
                    break
                self.observe_dequeue(event, queue.qsize())
                try:
                    await self.process(event)
                except Exception:
                    pass
        finally:
            self._event_queue = None
            stop.set()
            if scheduler is not None:
                scheduler.cancel()
            if dispatcher is not None:
                dispatcher.cancel()
            if terminal is not None:
                terminal.cancel()
            for producer in producers:
                producer.cancel()
            await asyncio.gather(
                *(task for task in (scheduler, dispatcher, terminal, *producers)
                  if task is not None),
                return_exceptions=True)
            await self.stop_background_services()
            self.bind_event_loop_lag_checkpoint(None)
            await probe.stop()
            self.diagnostic_output(f"Resident {self.resident.address_name} stopped")
