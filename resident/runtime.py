from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import time
import uuid
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import Any, Awaitable, Callable, Protocol, Sequence

from .capabilities import Capability, diagnostic_capabilities
from .realm import RealmClient
from .config import Config
from .context import ContextBuilder
from .domain import ToolResult, WakeEvent
from .provider import ModelProvider, ResponseInvalid, ResponseRejected
from .observability import EventLoopLagProbe, ObservedQueue, emit_timeline, timeline_reporter
from .outputs import (OutputCapability, capability_for_output, output_schema,
                      schema_fingerprint, validate_disposition)
from .readiness import ReadinessItem, ReadinessResult
from .store import Store, utc_now
from .tools import CORE_TOOL_NAMES, OwnerGuidanceAuthorization, ToolRegistry


class EventProducer(Protocol):
    async def run(self, queue: asyncio.Queue[WakeEvent], stop: asyncio.Event) -> None: ...


class OwnerTransport(Protocol):
    async def send_text(self, content: str) -> None: ...


class CallbackOwnerTransport:
    def __init__(self, callback: Callable[[str], None]):
        self.callback = callback

    async def send_text(self, content: str) -> None:
        self.callback(content)


class ResidentRuntime:
    _NORMAL_DIAGNOSTIC_EVENTS = frozenset({"output.delivery_failed", "wake.failed"})
    def __init__(self, config: Config, provider: ModelProvider, *, store: Store | None = None,
                 capabilities: Sequence[Capability] | None = None,
                 realm_client: RealmClient | None = None,
                 output_capabilities: Sequence[OutputCapability] | None = None,
                 event_producers: list[EventProducer] | None = None,
                 owner_transport: OwnerTransport | None = None,
                 owner_output_enabled: bool | None = None,
                 owner_output: Callable[[str], None] | None = None,
                 diagnostic_output: Callable[[str], None] | None = None):
        if config.keeper_history and realm_client is None:
            raise ValueError("Keeper history requires Realm")
        self.config, self.provider = config, provider
        self.realm_client = realm_client
        initial_capabilities = capabilities if capabilities is not None else diagnostic_capabilities()
        self._capabilities = self._validated_capabilities(initial_capabilities)
        self.store = store or Store(config.data_dir / "resident.sqlite3")
        if self.realm_client is not None:
            self.realm_client.bind_mutation_store(self.store.realm_mutation_request)
        self.resident, self.owner = self.store.provision(
            config.resident_name, config.owner_name, config.personality)
        self._event_queue: asyncio.Queue[WakeEvent | None] | None = None
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
        configured_outputs = list(output_capabilities or ())
        if owner_output_enabled is None:
            owner_output_enabled = config.owner_communication_enabled
        if owner_output_enabled:
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
            ))
        self._output_capabilities = self._validated_output_capabilities(configured_outputs)
        self._output_schema = output_schema(self._output_capabilities)
        self._output_schema_fingerprint = schema_fingerprint(self._output_schema)
        self.context_builder = ContextBuilder(self.store, role=config.role)
        self.conversation_id = self.store.conversation_id()
        self._initialized = False
        self._active_run_id: str | None = None
        self._active_event: WakeEvent | None = None
        self._owner_event_authorizations: dict[str, str] = {}
        self._enqueue_times: dict[str, float] = {}
        self._dequeue_observations: dict[str, tuple[float | None, int]] = {}
        self._active_max_loop_lag = 0.0
        self._active_total_loop_lag = 0.0
        self._active_loop_lag_samples = 0
        self._event_loop_lag_checkpoint: Callable[[], Awaitable[None]] | None = None


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
        await self.initialize()
        for message in self.store.pending_owner_messages():
            event = self._owner_message_wake(
                message["id"], message["content"], message["created_at"])
            await queue.put(event)
        if self._pending_capability_event is not None:
            await queue.put(self._pending_capability_event)
            self._pending_capability_event = None

    def close(self) -> None:
        if self.realm_client is not None:
            self.realm_client.bind_mutation_store(None)
        close_provider = getattr(self.provider, "close", None)
        try:
            if close_provider is not None:
                close_provider()
        finally:
            self.store.close()

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


    def _persist_disposition(
            self, raw: str | None, conversation_id: str, response_id: str, *,
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
            conversation_id, response_id, fingerprint, raw, normalized,
            validation_state, jobs, run_id=run_id, wake_id=wake.id if wake else None)
        if receipt["created"]:
            self._emit("disposition.generated", {
                "disposition_id": receipt["id"], "response_id": response_id,
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
            raise RuntimeError(f"Response final disposition is {validation_state}")
        return receipt


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


    async def initialize(self) -> None:
        if self._initialized:
            return
        if self.conversation_id is None:
            conversation_id = await self.provider.create_conversation()
            self.store.bind_conversation(conversation_id)
            self.conversation_id = conversation_id
        await self.recover_pending_responses()
        self._initialized = True

    async def recover_pending_responses(self) -> None:
        # Recover only a locally recorded final Response. Ambiguous POSTs and
        # interrupted tool loops require an operator fix/reset, not blind replay.
        steps = self.store.unfinished_response_steps()
        runs = {}
        for step in steps:
            runs.setdefault(step['run_id'], []).append(step)
        for run_id, run_steps in runs.items():
            latest = run_steps[-1]
            turn = json.loads(latest['turn_json']) if latest['turn_json'] else None
            if latest['status'] != 'returned' or not turn or turn['tool_calls']:
                raise RuntimeError('Unfinished inference/tool loop; inspect or reset this disposable instance database')
            wake = WakeEvent(**json.loads(latest['wake_json']))
            schema = json.loads(latest['schema_json'])
            self._persist_disposition(turn['message'], latest['conversation_id'], turn['response_id'],
                                      run_id=run_id, wake=wake, schema=schema)
            self.store.finish_run(run_id, 'completed', 0, len(run_steps),
                                  wake.payload.get('schedule_id') if wake.source == 'scheduler' else None,
                                  wake.payload.get('message_id') if wake.source == 'owner' else None)

    async def _execute_tool(self, call, response_id: str, registry: ToolRegistry) -> ToolResult:
        action = self.store.begin_tool_execution(self.conversation_id, response_id,
                                                 call.id, call.name, call.arguments)
        if not action['claimed']:
            if action['attachments_ephemeral']:
                return ToolResult(call.id, {'ok': False, 'error_code': 'attachment_unavailable',
                                           'error': 'Ephemeral result cannot be replayed; request a fresh observation'})
            return ToolResult(call.id, action['output'] or {
                'ok': False, 'error_code': 'unknown_outcome',
                'error': 'Interrupted tool outcome is unknown; action was not repeated'})
        invocation_id = hashlib.sha256(f'{self.conversation_id}\0{call.id}'.encode()).hexdigest()
        execution = await registry.execute(call.name, call.arguments, invocation_id=invocation_id)
        self.store.complete_tool_execution(self.conversation_id, call.id, execution.output,
                                            bool(execution.attachments))
        return ToolResult(call.id, execution.output, execution.attachments)

    async def process(self, event: WakeEvent) -> str:
        await self.initialize()
        await self.recover_pending_responses()
        completed = self.store.completed_event_run(event.id)
        if completed:
            return completed
        if event.source == 'owner' and event.reason == 'owner_message':
            message_id = event.payload.get('message_id')
            if (self._owner_event_authorizations.get(event.id) == message_id
                    and not self.store.is_pending_owner_message(message_id, self.owner.id)):
                return ''
        started = time.monotonic()
        run_id = self.store.start_run(event)
        if self.config.keeper_history:
            self.store.start_keeper_interaction(run_id, event, self.realm_client.game_id, self.realm_client.actor_id)
        self._active_run_id, self._active_event = run_id, event
        self._active_max_loop_lag = self._active_total_loop_lag = 0.0
        self._active_loop_lag_samples = 0
        calls, status = 0, 'failed'
        timeline_token = timeline_reporter.set(self._timeline) if self.config.timeline else None
        if self.config.timeline:
            queued_at, queue_depth = self._dequeue_observations.pop(event.id, (None, 0))
            self._timeline({'operation': 'host.dequeue', 'moment': 'finished', 'event_id': event.id,
                            'queue_depth': queue_depth, 'queue_wait_seconds': None if queued_at is None
                            else started - queued_at})
            emit_timeline('wake.process', 'started')
        try:
            self._emit('wake.started', {'event_id': event.id, 'source': event.source,
                                       'reason': event.reason, 'payload': event.payload})
            context = self.context_builder.build(event)
            realm_state = await self.realm_client.read({}) if self.realm_client else None
            if realm_state is not None:
                document = json.loads(context)
                document['realm_state'] = realm_state
                context = json.dumps(document, ensure_ascii=False)
            self._emit('context.assembled', {'characters': len(context),
                                            'pending_intentions': len(self.store.pending_intentions()),
                                            'recent_messages': 0})
            if self.config.keeper_history:
                self.store.set_keeper_input(run_id, self.conversation_id, context, realm_state)
            authorization = None
            if (event.source == 'owner' and event.reason == 'owner_message'
                    and self._owner_event_authorizations.get(event.id) == event.payload.get('message_id')
                    and self.store.is_pending_owner_message(event.payload.get('message_id'), self.owner.id)):
                authorization = OwnerGuidanceAuthorization(event.payload['message_id'])
            results = []
            tool_call_count = 0
            for round_number in range(self.config.max_tool_rounds + 1):
                # Current local authority is resolved again after each tool batch.
                registry = ToolRegistry(self.store, self._capabilities, self._emit,
                                        current_run_id=run_id, owner_guidance_authorization=authorization)
                instructions = self.context_builder.instructions(self.resident, self.owner, self._capabilities)
                schema = self._output_schema
                step_id = self.store.begin_response_step(run_id, self.conversation_id, round_number, event, schema)
                calls += 1
                operation = 'provider.tool_result_continuation' if results else 'provider.turn'
                provider_started = time.monotonic()
                emit_timeline(operation, 'started', round=round_number, tool_result_count=len(results))
                try:
                    turn = await self.provider.respond(context, registry.specs, results,
                                                       conversation_id=self.conversation_id,
                                                       instructions=instructions, output_schema=schema,
                                                       request_id=step_id)
                except ResponseRejected:
                    self.store.reject_response_step(step_id)
                    raise
                except ResponseInvalid as exc:
                    if exc.response_id:
                        self.store.note_response_id(step_id, exc.response_id)
                    raise
                finally:
                    emit_timeline(operation, 'finished', round=round_number,
                                  duration_seconds=time.monotonic() - provider_started,
                                  outcome='error' if sys.exc_info()[0] else 'ok')
                if not turn.response_id:
                    raise RuntimeError('Response returned no ID')
                self.store.record_response(step_id, turn)
                self._emit('model.responded', {'response_id': turn.response_id,
                    'tool_call_count': len(turn.tool_calls), 'has_message': bool(turn.message),
                    'input_tokens': turn.input_tokens, 'output_tokens': turn.output_tokens,
                    'cached_input_tokens': turn.cached_input_tokens})
                if self.config.keeper_history:
                    self.store.add_keeper_activity(run_id, 'model_turn', {
                        'message': turn.message, 'tool_calls': [
                            {'id': call.id, 'name': call.name, 'arguments': call.arguments} for call in turn.tool_calls]},
                        conversation_id=self.conversation_id, response_id=turn.response_id)
                if not turn.tool_calls:
                    self._persist_disposition(turn.message, self.conversation_id, turn.response_id,
                                              run_id=run_id, wake=event, schema=schema)
                    status = 'completed'
                    break
                if round_number >= self.config.max_tool_rounds:
                    raise RuntimeError('Model exceeded tool-round limit; instance needs inspection/reset')
                tool_call_count += len(turn.tool_calls)
                if tool_call_count > self.config.max_tool_calls:
                    raise RuntimeError('Model exceeded tool-call limit; instance needs inspection/reset')
                results = []
                for call in turn.tool_calls:
                    self._emit('tool.called', {'call_id': call.id, 'name': call.name, 'arguments': call.arguments})
                    tool_started = time.monotonic()
                    emit_timeline('tool.execute', 'started', call_id=call.id, tool_name=call.name, round=round_number)
                    try:
                        # Revocation while inference is in flight must take effect before execution.
                        current_registry = ToolRegistry(self.store, self._capabilities, self._emit,
                            current_run_id=run_id, owner_guidance_authorization=authorization)
                        result = await self._execute_tool(call, turn.response_id, current_registry)
                        results.append(result)
                        self._emit('tool.completed', {'call_id': call.id, 'name': call.name, 'result': result.output,
                            'attachments': [{'mime_type': a.mime_type, 'bytes': len(a.data),
                                             'ephemeral': True} for a in result.attachments]})
                        if self.config.keeper_history:
                            self.store.add_keeper_activity(run_id, 'tool_result', result.output,
                                conversation_id=self.conversation_id, response_id=turn.response_id, call_id=call.id)
                    finally:
                        emit_timeline('tool.execute', 'finished', call_id=call.id, tool_name=call.name,
                                      round=round_number, duration_seconds=time.monotonic() - tool_started,
                                      outcome='error' if sys.exc_info()[0] else 'ok')
            self.store.save_observed_snapshot('runtime.capabilities', self._capability_snapshot(self._capabilities))
            self._emit('wake.sleeping', {'status': 'completed'})
            status = 'completed'
            return run_id
        except (Exception, asyncio.CancelledError) as exc:
            self._emit('wake.failed', {'error_type': type(exc).__name__, 'error': str(exc)})
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
            if self.config.keeper_history:
                self.store.finish_keeper_interaction(run_id, status)
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
            self.bind_event_loop_lag_checkpoint(None)
            await probe.stop()
            self.diagnostic_output(f"Resident {self.resident.address_name} stopped")
