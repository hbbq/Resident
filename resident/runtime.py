from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Callable, Protocol, Sequence

from .capabilities import Capability, diagnostic_capabilities
from .config import Config
from .context import ContextBuilder
from .domain import ToolResult, WakeEvent
from .provider import ModelProvider
from .store import Store, utc_now
from .tools import CORE_TOOL_NAMES, ToolRegistry


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
    _NORMAL_DIAGNOSTIC_EVENTS = frozenset({"communication.failed", "wake.failed"})

    def __init__(self, config: Config, provider: ModelProvider, *, store: Store | None = None,
                 capabilities: Sequence[Capability] | None = None,
                 event_producers: list[EventProducer] | None = None,
                 owner_transport: OwnerTransport | None = None,
                 owner_output: Callable[[str], None] | None = None,
                 diagnostic_output: Callable[[str], None] | None = None):
        self.config, self.provider = config, provider
        initial_capabilities = capabilities if capabilities is not None else diagnostic_capabilities()
        self._capabilities = self._validated_capabilities(initial_capabilities)
        self.store = store or Store(config.data_dir / "resident.sqlite3")
        self.resident, self.owner = self.store.provision(
            config.resident_name, config.owner_name, config.personality)
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
        self.context_builder = ContextBuilder(
            self.store, memory_limit=config.context_memories, message_limit=config.context_messages)
        self._active_run_id: str | None = None
        self._active_event: WakeEvent | None = None

    @property
    def capabilities(self) -> tuple[Capability, ...]:
        return self._capabilities

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
        if self._pending_capability_event is not None:
            await queue.put(self._pending_capability_event)
            self._pending_capability_event = None

    def close(self) -> None:
        self.store.close()

    def owner_message_event(self, content: str) -> WakeEvent:
        message_id = self.store.add_message("inbound", self.owner.id, content)
        return WakeEvent(str(uuid.uuid4()), "owner", "owner_message", utc_now(),
                         {"message_id": message_id, "content": content})

    def _emit(self, event_type: str, data: dict) -> None:
        self.store.journal(event_type, data, self._active_run_id)
        if self.config.verbose or event_type in self._NORMAL_DIAGNOSTIC_EVENTS:
            details = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
            self.diagnostic_output(f"{event_type} {details}")

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

    async def process(self, event: WakeEvent) -> str:
        started = time.monotonic()
        run_id = self.store.start_run(event)
        self._active_run_id, self._active_event = run_id, event
        calls = 0
        status = "failed"
        continuation_id: str | None = None
        capability_event_state = self._capability_event_states.get(event.id)
        try:
            self._emit("wake.started", {
                "event_id": event.id, "source": event.source, "reason": event.reason,
                "occurred_at": event.occurred_at, "payload": event.payload,
            })
            capabilities = capability_event_state[0] if capability_event_state else self._capabilities
            context = self.context_builder.build(self.resident, self.owner, event, capabilities)
            self._emit("context.assembled", {
                "characters": len(context), "memories": len(self.store.recall(
                    event.reason + " " + str(event.payload), self.config.context_memories)),
                "pending_intentions": len(self.store.pending_intentions()),
                "recent_messages": len(self.store.recent_messages(self.config.context_messages)),
            })
            registry = ToolRegistry(self.store, capabilities, self._send_owner_message, self._emit)
            results: list[ToolResult] = []
            for round_number in range(self.config.max_tool_rounds + 1):
                calls += 1
                turn = await self.provider.respond(context, registry.specs, results, continuation_id)
                self._emit("model.responded", {
                    "response_id": turn.response_id, "tool_call_count": len(turn.tool_calls),
                    "has_message": bool(turn.message), "input_tokens": turn.input_tokens,
                    "output_tokens": turn.output_tokens,
                })
                if turn.message:
                    self._emit("model.message", {"content": turn.message})
                if not turn.tool_calls:
                    continuation_id = None
                    break
                continuation_id = turn.response_id
                if round_number >= self.config.max_tool_rounds:
                    self._discard_continuation(continuation_id)
                    raise RuntimeError("Model exceeded the configured tool-round limit")
                results = []
                for call in turn.tool_calls:
                    self._emit("tool.called", {"call_id": call.id, "name": call.name, "arguments": call.arguments})
                    execution = await registry.execute(call.name, call.arguments)
                    results.append(ToolResult(call.id, execution.output, execution.attachments))
                    completion = {"call_id": call.id, "name": call.name, "result": execution.output}
                    if execution.attachments:
                        completion["attachments"] = [{
                            "type": "image", "mime_type": attachment.mime_type,
                            "byte_count": len(attachment.data), "ephemeral": True,
                        } for attachment in execution.attachments]
                    self._emit("tool.completed", completion)
                if not continuation_id:
                    raise RuntimeError("Provider did not return a response id for tool continuation")
            if capability_event_state is not None:
                self.store.save_observed_snapshot("runtime.capabilities", capability_event_state[1])
                self._capability_event_states.pop(event.id, None)
            self._emit("wake.sleeping", {"status": "completed"})
            status = "completed"
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
            self.store.finish_run(run_id, status, duration, calls, schedule_id)
            self._active_run_id, self._active_event = None, None

    async def enqueue_due_wakeups(self, queue: asyncio.Queue[WakeEvent]) -> None:
        for scheduled in self.store.claim_due_wakeups(utc_now()):
            await queue.put(WakeEvent(
                str(uuid.uuid4()), "scheduler", scheduled["reason"], utc_now(),
                {"schedule_id": scheduled["id"], "scheduled_for": scheduled["due_at"],
                 "context": scheduled["context"]}))

    async def scheduler_loop(self, queue: asyncio.Queue[WakeEvent], stop: asyncio.Event) -> None:
        while not stop.is_set():
            await self.enqueue_due_wakeups(queue)
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.config.scheduler_poll_seconds)
            except TimeoutError:
                pass

    async def run_interactive(self) -> None:
        queue: asyncio.Queue[WakeEvent | None] = asyncio.Queue()
        self._event_queue = queue
        await self.enqueue_startup_wakeups(queue)
        stop = asyncio.Event()
        scheduler = asyncio.create_task(self.scheduler_loop(queue, stop))
        producers = [asyncio.create_task(producer.run(queue, stop)) for producer in self.event_producers]

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
                    await queue.put(self.owner_message_event(text))

        terminal = asyncio.create_task(terminal_input())
        self.diagnostic_output(
            f"Resident {self.resident.address_name} ({self.resident.id}) is sleeping; /quit stops the process")
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                try:
                    await self.process(event)
                except Exception:
                    pass
        finally:
            self._event_queue = None
            stop.set()
            scheduler.cancel()
            terminal.cancel()
            for producer in producers:
                producer.cancel()
            await asyncio.gather(scheduler, terminal, *producers, return_exceptions=True)
            self.diagnostic_output(f"Resident {self.resident.address_name} stopped")
