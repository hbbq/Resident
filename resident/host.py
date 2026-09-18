from __future__ import annotations

import asyncio
import uuid
from contextlib import suppress
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

from .capabilities import Capability
from .domain import WakeEvent
from .mailbox import DEFAULT_TTL_SECONDS, Mailbox
from .runtime import EventProducer, ResidentRuntime
from .store import utc_now


@dataclass(frozen=True)
class InstancePolicy:
    subscriptions: frozenset[str]

    def receives(self, event: WakeEvent) -> bool:
        selectors = self.subscriptions
        return ("*" in selectors or event.source in selectors or
                f"{event.source}.{event.reason}" in selectors)


def messaging_capability(mailbox: Mailbox, sender: str,
                         recipients: Callable[[], Iterable[str]]) -> Capability:
    async def send(arguments: dict) -> dict:
        recipient = arguments["recipient"]
        if recipient not in set(recipients()):
            raise ValueError(f"Unknown message recipient: {recipient}")
        message = mailbox.send(
            sender, recipient, arguments["content"],
            ttl_seconds=arguments.get("ttl_seconds", DEFAULT_TTL_SECONDS),
        )
        return {"message_id": message["id"], "status": "pending",
                "expires_at": message["expires_at"]}

    return Capability(
        "messaging", "Shared durable asynchronous logical-address mailbox",
        "messaging_send",
        "Send an asynchronous message to a logical recipient. This does not wait for a reply.",
        {"type": "object", "properties": {
            "recipient": {"type": "string"},
            "content": {"type": "string"},
            "ttl_seconds": {"type": ["integer", "null"], "minimum": 1, "maximum": 86400},
        }, "required": ["recipient", "content"], "additionalProperties": False},
        send,
    )


class RuntimeHost:
    """Owns isolated runtimes and routes shared events according to local policy."""

    def __init__(self, runtimes: Mapping[str, ResidentRuntime],
                 policies: Mapping[str, InstancePolicy], mailbox: Mailbox, *,
                 event_producers: Sequence[EventProducer] = (), default_id: str = "resident",
                 instance_producers: Mapping[str, Sequence[EventProducer]] | None = None,
                 diagnostic_output: Callable[[str], None] | None = None):
        if not runtimes:
            raise ValueError("RuntimeHost requires at least one Resident")
        if set(runtimes) != set(policies):
            raise ValueError("Every Resident runtime must have exactly one routing policy")
        if default_id not in runtimes:
            raise ValueError(f"Default Resident is not configured: {default_id}")
        self.runtimes = dict(runtimes)
        self.policies = dict(policies)
        self.mailbox = mailbox
        self.event_producers = tuple(event_producers)
        self.instance_producers = {
            key: tuple(value) for key, value in (instance_producers or {}).items()
        }
        if set(self.instance_producers) - set(runtimes):
            raise ValueError("Instance producer target is not a configured Resident")
        self.default_id = default_id
        self.diagnostic_output = diagnostic_output or (lambda message: None)
        self.queues: dict[str, asyncio.Queue[WakeEvent | None]] = {
            instance_id: asyncio.Queue() for instance_id in runtimes
        }
        self._stopping = False

    @property
    def recipients(self) -> frozenset[str]:
        return frozenset((*self.runtimes, "owner"))

    async def route(self, event: WakeEvent) -> tuple[str, ...]:
        delivered: list[str] = []
        for instance_id, policy in self.policies.items():
            if policy.receives(event):
                await self.queues[instance_id].put(event)
                delivered.append(instance_id)
        return tuple(delivered)

    async def deliver_mailbox(self) -> int:
        delivered = 0
        for message in self.mailbox.pending():
            recipient = message["recipient"]
            if recipient in self.runtimes:
                event = WakeEvent(
                    str(uuid.uuid4()), "messaging", "message_received", utc_now(),
                    {"message_id": message["id"], "sender": message["sender"],
                     "content": message["content"], "created_at": message["created_at"]},
                )
                await self.queues[recipient].put(event)
                delivered += int(self.mailbox.delivered(message["id"]))
            elif recipient == "owner" and message["sender"] in self.runtimes:
                runtime = self.runtimes[message["sender"]]
                try:
                    await runtime.owner_transport.send_text(message["content"])
                except Exception as exc:
                    self.diagnostic_output(
                        f"Mailbox delivery {message['id']} failed: {type(exc).__name__}")
                else:
                    delivered += int(self.mailbox.delivered(message["id"]))
            # Unknown/offline logical addresses remain pending until TTL expiry.
        return delivered

    async def _mailbox_loop(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            await self.deliver_mailbox()
            try:
                await asyncio.wait_for(stop.wait(), timeout=0.25)
            except TimeoutError:
                pass

    async def _worker(self, instance_id: str, stop: asyncio.Event) -> None:
        runtime, queue = self.runtimes[instance_id], self.queues[instance_id]
        runtime._event_queue = queue
        await runtime.enqueue_startup_wakeups(queue)
        scheduler = asyncio.create_task(runtime.scheduler_loop(queue, stop))
        try:
            while not stop.is_set():
                event = await queue.get()
                if event is None:
                    return
                try:
                    await runtime.process(event)
                except Exception:
                    pass
        finally:
            runtime._event_queue = None
            scheduler.cancel()
            await asyncio.gather(scheduler, return_exceptions=True)

    async def run(self, *, interactive: bool = True) -> None:
        stop = asyncio.Event()
        shared_queue: asyncio.Queue[WakeEvent] = asyncio.Queue()

        async def router() -> None:
            while not stop.is_set():
                await self.route(await shared_queue.get())

        async def terminal() -> None:
            runtime = self.runtimes[self.default_id]
            while not stop.is_set():
                try:
                    text = await asyncio.to_thread(input, f"{runtime.owner.address_name}> ")
                except (EOFError, KeyboardInterrupt):
                    text = "/quit"
                if text.strip() == "/quit":
                    stop.set()
                    return
                if text.strip():
                    await self.queues[self.default_id].put(runtime.owner_message_event(text))

        workers = [asyncio.create_task(self._worker(item, stop)) for item in self.runtimes]
        tasks = [asyncio.create_task(router()), asyncio.create_task(self._mailbox_loop(stop))]
        tasks.extend(asyncio.create_task(producer.run(shared_queue, stop))
                     for producer in self.event_producers)
        for instance_id, producers in self.instance_producers.items():
            tasks.extend(asyncio.create_task(producer.run(self.queues[instance_id], stop))
                         for producer in producers)
        if interactive:
            tasks.append(asyncio.create_task(terminal()))
        self.diagnostic_output(
            f"Runtime host started {len(self.runtimes)} Residents; default={self.default_id}")
        try:
            await stop.wait()
        finally:
            self._stopping = True
            stop.set()
            for queue in self.queues.values():
                await queue.put(None)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, *workers, return_exceptions=True)

    def close(self) -> None:
        for runtime in self.runtimes.values():
            runtime.close()
        self.mailbox.close()
