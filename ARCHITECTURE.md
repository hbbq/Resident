# Resident Architecture

## Instance and host boundary

`RuntimeHost` is the process-level lifecycle and routing boundary. It loads a strict startup-time catalog, constructs one `ResidentRuntime`, provider, queue, and SQLite `Store` per stable declarative ID, and owns shared connector polling and the durable mailbox. Resident-owned state lives under `instances/<id>`; changing declarative prompts or policy refreshes metadata without replacing the durable Resident UUID or provider session binding.

Local policy is resolved before inference. Subscriptions decide which shared events enter an instance queue, while capability grants decide which executable tool schemas enter that instance's context and registry. Neither implies the other, and definitions cannot supply handlers, credentials, arbitrary schemas, or expand their own authority. Instance-private Telegram transports are bound to exactly one queue and store checkpoint. The default terminal route is explicit rather than broadcast.

Generic inter-instance communication is a host-owned asynchronous mailbox, not direct runtime access or RPC. Messages contain logical sender/recipient addresses, content, timestamps, expiry, and `pending`/`delivered`/`expired` state. The tool schema enumerates configured Resident recipients, and handoff creates an ordinary messaging wake. Owner communication uses the separate Owner tool and transport. Delivery records handoff only, and replies are independent messages.

This document records architectural principles that have been decided so far. It intentionally avoids specifying implementation details that have not yet been justified by experience.

## Core principle

Resident's identity is separate from its models, connectors, embodiments, communication transports, and owner.

The runtime should not assume that there can only ever be one Resident instance or that a Resident's owner must be a human. `Resident` describes the agent/runtime concept; a particular instance has its own persistent identity, personality, state, owner, and human-friendly name by which it can be addressed.

Likewise, `owner` is a role and authority relationship rather than a hard-coded person. An owner has an identity and may have a human-friendly name by which Resident addresses it. The initial Resident instance may be owned by a human, while a future Resident instance embodied as a robot could, for example, have another Resident instance as its owner.

Conceptually:

```text
Resident instance
├── Identity / personality
│   └── Address name
├── Owner
│   ├── Identity
│   └── Address name
├── Runtime
├── Managed conversational session
├── Pending intentions
├── Journal
├── Model provider(s)
└── Connectors
    ├── HomeOps
    ├── Robot
    ├── Cameras
    └── ...
```

The list of connectors is illustrative, not a fixed set.

## Runtime

Resident runs as a long-lived service/process. Sleeping does not mean terminating the process: runtime, connector subscriptions, event handling, and scheduling remain alive while no AI inference is taking place.

Resident uses a hybrid event-driven runtime rather than requiring a permanent reasoning loop.

All reasons for Resident to begin thinking are normalized into the same internal concept: a `WakeEvent`. Sources may include:

- connector events;
- capability availability changes, such as a robot coming online;
- messages from the owner;
- scheduled wakeups;
- wakeups previously requested by Resident itself.

A wake event should remain deliberately small and general, conceptually containing a source, timestamp, reason/type, and source-specific payload. The detailed schema should be driven by implementation needs rather than designed exhaustively up front.

Runtime compares a durable, public capability snapshot at startup and whenever capabilities are explicitly replaced, registered, or removed. A newly provisioned Resident records its first baseline silently. Later additions, removals, and public descriptor changes produce a normal `capabilities_changed` wake; executable handlers and connector secrets are outside the snapshot. Each wake uses one capability snapshot for both context and tool registration. There is no capability polling loop, and detection never exercises a capability.

During a wakeup Resident receives an appropriate working context, reasons and possibly acts, and may then sleep again. The durable managed-agent session supplies episodic continuity, so ordinary wakes add the new trigger without replaying prior communication or working context. New sessions receive bounded bootstrap state and handover; changed locally authoritative identity, capability, and standing-guidance state is synchronized durably to existing sessions. Pending intentions, standing Owner guidance, and curated long-term memory remain explicit local state.

Scheduling is a mechanism, not a collection of hard-coded behaviors. Resident should be able to request a future wakeup with a reason/context rather than requiring dedicated classes such as `TemperatureMonitor` or `RobotExplorationBehavior`.

### Runtime versus Resident

Runtime is responsible for deterministic mechanisms such as wake/sleep, event routing, persistence, attention limits, permissions and hard safety boundaries, model invocation, tool execution, scheduling, and logging.

Resident decides what observations mean, what is interesting, what it wants to do, which available capabilities to use, whether to ask its owner, and whether it wants to revisit something later.

> Runtime enables Resident's life; it should not live it on Resident's behalf.

### Managed agent boundary

The initial managed-agent integration uses one long-lived OpenAI Agents session per Resident instance. The binding is local and durable. Losing or deliberately rolling over that remote session must not change Resident identity or erase schedules, pending intentions, connector checkpoints, communication records, or the local journal, but conversational context in that session is not recoverable from those local records.

OpenAI owns conversational session history and the managed turn loop. Resident remains an outbound-only environment bridge: it selects and coalesces wakes, submits factual wake envelopes, executes requested function actions locally, validates every argument, applies deterministic policy, records observable outcomes, and returns results. Function calls are durably claimed before execution; completed results can be replayed without repeating local effects, while an action interrupted before its outcome is recorded is not automatically repeated. HomeOps and other integrated systems remain unaware of OpenAI. Ordinary connector events are processed by the local queue and cannot directly interrupt active work; Owner ingress retains local priority semantics as the runtime evolves.

Agent behavior is declared in repository code and supplied as session configuration, optionally layered over a saved Agent resource. Machine endpoints, device identifiers, credentials, rate and attention limits, and integration policy remain local configuration. Direct function actions are the initial capability bridge; an authenticated MCP/tool-search surface may be considered later if the capability set becomes too large.

## Wake context

The context supplied at a wakeup is Resident's temporary working context, not a dump of its persistent state.

It should always contain enough information for Resident to understand the current wakeup, including:

- Resident's identity/personality and address name;
- its owner's identity and address name;
- current time;
- the complete relevant `WakeEvent`, including its reason/source and associated payload or attachments;
- capabilities currently available to Resident.

The context builder may additionally include relevant pending intentions, recent/relevant communication, and a small amount of recent runtime/journal context when useful.

It must not automatically include the entire communication history or journal.

The journal and communication history are exposed through bounded read-only search capabilities so Resident can investigate its own history without automatically receiving that history in every model context.

Pending intentions can initially be included generously while their number is small. More selective retrieval should only be introduced when there is evidence that it is needed.

A separate persistent `working_state` concept is not required initially; the managed session and pending intentions should be allowed to demonstrate whether another form of continuity is actually necessary.

> ContextBuilder provides enough context to begin thinking, not everything Resident might possibly need.

Long-term memory is an append-only revision graph in the per-Resident store. Active records are retrieved selectively through bounded read-only tools. No durable Curator memory revision is accepted without at least one verified reference to an item in the fetched source page; mixed valid/invalid evidence rejects that mutation, and provenance excerpts and hashes are derived only from the Resident's credential-scrubbed source projection. A separately configured Curator reads that explicit allowlisted projection after a durable cursor and commits accepted decisions plus checkpoint in one transaction. Tool arguments/results and unknown structured fields never enter the projection. Recognizable credential-bearing text structures are deterministically removed both before Curator inference and before its memories or handovers are persisted; arbitrary natural-language secret classification is intentionally not claimed. Each completed managed-session wake durably requests curation through its exact completed-turn ID. A per-Resident coordinator coalesces newer watermarks, serializes execution, bounds pages at that turn, and retries failures with capped exponential backoff; ordinary long-term-memory reads are intentionally eventual-consistent while it lags. Startup reconciliation and reachable rollover/final-handover paths remain awaited barriers. Clean shutdown drains briefly and otherwise leaves the request durable for restart, so correctness does not depend on graceful shutdown. Standing Owner guidance has its own complete revision history and is always present in wake context, subject to deterministic per-entry, active-count, and total serialized-size bounds enforced on each set or replacement.

Session compatibility is field-based. Mutable model settings are patched between turns and their last successfully applied values are persisted independently of the immutable protocol descriptor, so a partial remote representation cannot erase a configured change after restart. Local configuration and capability grants that do not alter the advertised function protocol continue unchanged; revocation can retain an unavailable compatibility handler. Function additions/renames, descriptions or schemas, immutable instructions, saved-Agent ID adoption, explicit new chapters, unrecoverable remote sessions, and security contract revisions use an explicit rollover record. Before a reachable intentional rollover the Curator performs final catch-up; the replacement bootstrap contains bounded memory awareness and a short-lived handover, never the complete store. Rollover persists its exact create request and token before crossing the remote boundary, records the attempt before POST, and atomically binds the returned replacement ID with its protocol and mutable settings. Because the current Agents API has no supported create-idempotency or lookup-by-token contract, an attempted create whose returned ID was not durably bound remains explicitly uncertain and blocks automatic re-creation; a durably bound replacement is completed on restart without another POST.

## Pending intentions

Pending intentions are persistent first-class state separate from conversational context and any future long-term memory store.

An intention describes something Resident may want its future self to continue, revisit, or do when circumstances permit.

For example:

```text
When the robot becomes available, inspect behind the sofa.
```

The initial representation should be deliberately small, conceptually containing an id, free-form content, creation time, and status. It must not grow into a conventional planner/task-management system unless actual Resident behavior demonstrates a need for one.

Communication does not introduce a separate `PendingQuestion` system concept. If Resident wants to record that it is waiting for information from its owner, it may use a pending intention. Runtime does not need to decide which messages are questions or which later messages are answers.

## Observability

An emergent system must be inspectable enough to understand what happened when its behavior is surprising.

### Wake runs

A `WakeRun` is the observable unit of Resident activity from one wakeup until Resident returns to sleep or the run otherwise terminates.

Each run should have a persistent identity and initially record at least:

```text
id
started_at
finished_at
wake_reason
wake_source
status
duration
```

Metrics should be extensible. v0 only needs to require elapsed runtime, but future model-provider information may add metrics such as model-call count, input/output tokens, cost, latency, or other useful usage measurements.

### Terminal output capabilities

Managed Agents sessions distinguish observations, interactive capabilities, and terminal `OutputCapability` requests. Every completed turn emits `{"outputs": [...]}`; an empty array is intentional silence. The exact authorized Owner/display branches are installed through `agent.text.format` and fingerprinted as immutable session protocol. Runtime parses and validates the final assistant `output_text` against that session snapshot and current local authorization. Schema validity never overrides a revoked local grant.

The completed remote turn and binding checkpoint precede local disposition ingestion. `final_dispositions`, `output_requests`, and `output_attempts` then record the receipt and jobs atomically before the wake completes. A missing receipt for the bound completed turn is recovered from that exact assistant item after restart. Stable disposition/output IDs make stream replay, REST reconciliation, and repeated processing idempotent.

Delivery is owned by a background dispatcher, not the model wake. Jobs move through queued, attempting, retry-wait, accepted-by-transport, permanent-failure, uncertain, and policy/unavailable rejection states. V1 retries retryable and interrupted uncertain attempts up to three times with bounded backoff. Telegram and HomeOps do not provide a complete exactly-once contract, so this is deliberately at-least-once and may duplicate an uncertain delivery. HomeOps acknowledgement means accepted into its queue, not physically rendered. A terminal failure durably schedules one safe `output_delivery_failed` event; failures produced while handling that event cannot recursively schedule another. Operational journal events contain identifiers, states, classifications, targets, and counts but never output content or raw transport errors.

The runtime/journal should make it possible to reconstruct the externally relevant lifecycle of a wake run, including approximately:

```text
wake event
→ context supplied
→ model interaction
→ tool calls and results
→ persisted state changes
→ outgoing communication / scheduled work
→ sleep
```

### Structured observable events

Runtime activity should be emitted as structured observable events. Interactive console output, persistent journal storage, and future observability interfaces should consume the same event stream rather than implement separate views of Resident activity.

The runtime concurrency model is one asyncio event loop with one unbounded FIFO queue and worker task per Resident. Each worker awaits the wake itself before dequeuing its next event, while routine completed wakes durably signal an asynchronous per-Resident Curator coordinator and do not wait for incremental curation. The coordinator coalesces requested completed-turn boundaries and serializes Curator work for that Resident; different Residents, shared routing, connector producers, mailbox delivery, schedulers, and terminal input use independent asyncio tasks. Tool calls in one provider action batch execute sequentially. Blocking Agents, Responses, HomeOps, display, Telegram, Curator-model, file-input, and similar operations use `asyncio.to_thread` and therefore share the loop's default executor; Agents polling occupies its worker for the complete remote lifecycle. Camera capture uses an async subprocess and ONVIF uses async HTTP. Ordinary SQLite store, mailbox, journal, JSON/context, and core-tool work is synchronous on the event-loop thread. Consequently, a long wake serializes later wake work for that Resident, default-executor saturation can delay otherwise independent offloads, and a slow synchronous store or CPU operation can stall all loop tasks.

Opt-in timeline journaling (`--timeline` / `RESIDENT_TIMELINE=true`) measures these boundaries without changing them. It records host enqueue/dequeue and queue wait, provider preflight and each turn or tool-result continuation, local tools, HomeOps/display acknowledgement, executor queue versus worker time for model and connector requests, Curator batches, total durable wake time, and event-loop lag aggregated in memory for each active wake with its maximum. Separate `curator.requested`, `curator.started`, `curator.caught_up`, retry, and degraded events distinguish durable wake completion from background lag without attributing delayed work to a later wake. For ordinary wake and tool-result submissions, the Agents adapter opens a live session event stream before submitting and uses exact correlated item, required-action, completed-item, and completed-turn events as the healthy waiting path. A durable per-session wake ledger is committed immediately before each wake POST: absence means fresh, `possibly_accepted` forces exact correlation before any retry, and `settled` retains the exact correlated turn. This lets a fresh wake correlate from its already-open stream without scanning historical items while preserving conservative crash recovery. Turn-wide reducer state is transient but retained across `requires_action` and tool-result streams, so output indexes and verified assistant content remain continuous across tool rounds; completion, terminal failure, fallback, abandonment, and rollover discard it. Initial session input and any stream disconnect, timeout, parser failure, semantic uncertainty, mismatched state, or incomplete output return to exact session/turn/item reconciliation; stream events have no assumed durable replay semantics. Agents stream, fallback, and HTTP spans retain monotonic placement within the enclosing lifecycle so streaming wait, submission, polling/reconciliation, item retrieval, and continuation latency can be reconstructed separately. Fallback spans classify parser, reducer-semantic, and transport uncertainty with safe reason codes, phase, correlation-known state, and unknown-event counts. Records intentionally exclude prompts, arguments, results, credentials, headers, full URLs, event payloads, raw unknown event types, and all session, turn, and event identifiers.

Conceptually:

```text
Runtime event
    ├── Console observer
    ├── Journal observer
    └── future observers (web UI, metrics, etc.)
```

When Resident is run interactively in a terminal, the console observer should make its activity visible as it happens. Useful output includes wake reason, context assembly, observable Resident decisions/rationale, capability/tool calls and results, intention changes, communication, sleep, and run metrics.

Messages that Resident intentionally sends to its owner are communication, not merely diagnostic output. The terminal transport must therefore render them in a clearly distinguishable format so they cannot easily be confused with runtime logs, model diagnostics, or tool output. The exact visual style is an implementation detail, but the distinction should be obvious at a glance.

A model turn's returned text is a wake/model result, not communication. It remains available to the structured journal and verbose/debug observers, but normal terminal presentation and future Owner transports must not interpret it as an Owner-facing message.

The goal is to provide a useful window into Resident's behavior during development and experimentation. This does not require storing or exposing a model's private/internal chain-of-thought. Observable decisions, rationale supplied for actions, model outputs, tool interactions, and state changes are sufficient for debugging and analysis.

## Models

Resident must not depend on a specific AI model or provider.

Intelligence is a runtime capability; identity and continuity belong to Resident.

Capability results may include narrowly typed, ephemeral content such as an image in addition to safe structured
metadata. Providers translate that content into their own model-input representation. Ephemeral content is available
only to the active model continuation: it is not written to the journal or another Resident store, which records only
safe result and attachment metadata.

A future implementation may use local models, cloud models, several capability/cost tiers, or escalation between them. The exact policy is deliberately unspecified for now.

Model calls should eventually be observable enough to measure workload, latency, token usage, and cost. This will allow model choices to be based on actual Resident workloads rather than guesses made before the system exists.

## Connectors

Connectors expose capabilities and observations to Resident and should be independently replaceable and evolvable.

### Minimal connector contract

The v0 connector contract should remain deliberately small. A connector provides:

- an identity and natural-language description sufficient for Resident to understand what the connector represents;
- current availability/status;
- a list of self-describing capabilities;
- a way to emit events into the runtime, which may become `WakeEvent`s.

A capability is the central abstraction in v0. It is conceptually similar to an LLM tool and should provide enough information for Resident to understand and invoke it, such as:

```text
id / name
description
input schema
result
```

Capabilities may represent both observation/read operations and actions. v0 does not require separate core abstractions for resources, sensors, devices, actions, or capability hierarchies. Those distinctions can be introduced later if real connectors demonstrate a need for them.

Connector events do not require an exhaustively declared event taxonomy up front. The runtime needs to be able to receive them and preserve enough source-specific information in the resulting wake event for Resident to understand why it woke.

Connector-visible resources remain domain-specific. For example, a refreshable camera connector may emit `cameras_changed` with safe camera metadata, while another connector can use its own vocabulary and observation lifecycle. Such events do not imply a generic resource hierarchy and do not cause runtime to inspect or test the reported resource.

Connectors describe what Resident can observe or do and the constraints on those actions. They should not encode what Resident should want to do.

Agent-core special cases for specific connectors should be avoided where practical.

### Connector ownership and adaptation

Connectors belong on the Resident side of the boundary. External systems do not need to know about Resident or implement the Resident connector contract themselves.

> External systems do not implement the Resident connector protocol. Resident connectors adapt external systems to it.

For example, HomeOps should expose an API that makes sense for HomeOps. A `HomeOpsConnector` can consume that API and translate HomeOps-specific resources, DTOs, events, and operations into the common concepts Resident understands.

Conceptually:

```text
Resident core
    |
Resident connector contract
    |
HomeOpsConnector
    |
HomeOps API
```

The same principle applies to other systems. A robot connector may adapt a robot-specific protocol, while a camera connector may adapt an RTSP stream or camera API.

The connector contract is more important than its in-process implementation. Early connectors may simply be classes inside the Resident application. The architecture should not unnecessarily prevent a future connector from running as a separate process, on another machine, or in another language.

Resident-specific endpoints should generally not be added to external systems merely to satisfy the connector contract. An external API may of course evolve in ways that make integration easier when those changes also make sense for that external system independently of Resident.

A future generic connector may allow Resident to use sufficiently self-describing APIs without requiring a custom adapter for every service. This is an extension point rather than a v0 requirement.

The first generic external-application connector keeps its operation catalog locally pinned in each Resident definition. It adapts a narrow HTTP invocation endpoint into ordinary `Capability` objects, supplies immutable instance bindings outside model-controlled arguments, and leaves validation and domain authority with the external application. Existing capability grants and managed-session descriptor snapshots remain the authorization and protocol boundaries. Remote metadata cannot expand the catalog in v0.

Physical co-location does not require logical integration. For example, a camera and microphone mounted on a robot may remain separate connector/device identities. Relationships such as `mounted_on`, `powered_by`, or correlated availability may later be declared or inferred by Resident.

## Long-term memory

Long-term memory is deliberately separate from the OpenAI Agents session. The session is working context; curated memory, Owner guidance, provenance, curator checkpoints, handovers, and rollover lineage are local durable subsystems with distinct retention and retrieval semantics.

The former SQLite `memories` table, automatic recall and standing-guidance context lanes, and `remember`/`recall`/`update_memory`/`forget` tools were removed rather than retained as a competing compatibility system. Session rollover therefore currently loses conversational knowledge that has not been represented elsewhere, such as in communication history or a pending intention.

Runtime still maintains an append-only journal of what happened. The journal supports debugging, auditability, and later analysis; it is not long-term memory and must not automatically become the reasoning context for every wakeup.

### World model

A dedicated structured world-model representation is not required initially.

A future Memory Store may represent relationships, observations, hypotheses, regularities, or structured entities, but that representation should be chosen with the curator architecture rather than inferred from the removed legacy schema.

## Communication

Communication should be transport-independent and asynchronous.

The runtime treats communication as messages between a Resident instance and its owner, not as a built-in question/answer protocol. The owner role is not inherently human. A message may be a question, answer, instruction, observation, correction, small talk, or something else; interpreting its meaning and relationship to previous communication belongs to Resident.

Conceptually, a persisted message needs only general communication metadata such as an identity, timestamp, direction/sender, content, and optional attachments. The exact schema should remain small until experience demonstrates additional requirements.

Incoming owner messages wake Resident and are delivered once as part of the corresponding `WakeEvent`. Communication history is persisted independently of conversational context and long-term memory, and is retrieved on demand rather than replayed into a surviving managed session or normal bootstrap.

All intentional outgoing Resident communication, including replies during Owner-initiated wakes, uses the communication capability. That path persists the message and delivers it through the currently configured transport. Model-returned result text is not a fallback transport: if communication is rejected by attention policy or fails in transport, the failure is journaled and the result text is not delivered in its place. Resident may send a question and go back to sleep without waiting for an answer. A later owner message is simply another message and wake event; Resident is responsible for understanding whether it answers something earlier.

If Resident considers an unresolved exchange important enough to revisit, it can create a pending intention. Runtime should not manufacture a pending-question record on Resident's behalf.

Communication transports are replaceable. v0 may use the interactive terminal for both incoming and outgoing messages; later transports may include a web UI, messaging service, another Resident instance, or another mechanism without changing Resident's conceptual communication model.

Conceptually, messages may contain text and/or attachments such as images and audio. v0 may implement text only, but the interface should not unnecessarily make text the permanent assumption.

### Attention budget

Runtime protects the owner's attention with configurable limits. This is a hard mechanism around outgoing communication rather than merely a personality instruction.

Resident may internally want to send more messages than can be delivered. Messages may wait, be prioritized, become obsolete, or potentially be combined. The runtime need not understand whether a message is specifically a question in order to enforce an attention budget.

The exact rate limits and urgency scheme remain open. Important/urgent communication may eventually have a separate policy, but the initial design should avoid an elaborate hard-coded priority system.

## Authority and safety

Owner identity and authority are explicit system concepts. `Owner` is a role, not a synonym for a particular human user.

A Resident instance has an owner identity and an address name for natural communication. The Resident instance likewise has its own persistent identity and address name. These names are presentation/conversation concepts and must not be used as the underlying stable identities.

Normal owner instructions outrank Resident's autonomous goals. A future explicit `sudo`/override marker can communicate that an instruction must not be treated as a suggestion or balanced against Resident's own priorities.

Deterministic safety and system constraints remain above both Resident and owner instructions. Physical connectors should enforce hard boundaries that the reasoning model cannot override.

This deliberately permits future ownership relationships such as one Resident instance owning another without requiring a different agent runtime or authority model.

## Technology and deployment

The initial implementation should use Python 3.12+ as a lightweight asynchronous application, with SQLite for persistent state.

The core should avoid adopting an agent framework. Resident is itself an experiment in agent runtime, memory, context, capabilities, and behavior; framework assumptions about those concepts should not define the architecture prematurely. Small conventional libraries may be used where they solve ordinary infrastructure problems.

Resident should be runnable directly as a normal process, conceptually:

```text
python -m resident
```

Containerization should be supported without being required. The same application should be able to run interactively on an ordinary always-on computer and, where practical, in a Linux container on common architectures such as amd64 or arm64. Raspberry Pi deployment is a possible target but is not a v0 constraint.

Persistent Resident state must live outside process/container lifetime. Replacing or restarting the process/container must not create a new Resident. A persistent data location may contain the SQLite database, attachments, and other durable state introduced later.

A core continuity test for the initial implementation is therefore:

```text
start Resident
→ wake and create persistent state
→ stop/restart process
→ wake again
→ same Resident, with prior persistent state available
```

OS- or CPU-specific dependencies should be avoided where reasonably practical, but portability should not be allowed to complicate the initial experiment unnecessarily.

## Evolution principle

Prefer adding general mechanisms after observing real failure modes over predicting and implementing specific behaviors in advance.

Resident should remain an AI entity using capabilities, not gradually become a conventional rules engine with an LLM attached to it.
