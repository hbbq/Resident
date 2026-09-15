# Resident Architecture

This document records architectural principles that have been decided so far. It intentionally avoids specifying implementation details that have not yet been justified by experience.

## Core principle

Resident's identity is separate from its models, connectors, embodiments, and communication transports.

Conceptually:

```text
Resident
├── Identity / personality
├── Runtime
├── Persistent memory
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

During a wakeup Resident receives an appropriate working context, reasons and possibly acts, persists anything it wants its future self to retain, and may then sleep again.

Scheduling is a mechanism, not a collection of hard-coded behaviors. Resident should be able to request a future wakeup with a reason/context rather than requiring dedicated classes such as `TemperatureMonitor` or `RobotExplorationBehavior`.

### Runtime versus Resident

Runtime is responsible for deterministic mechanisms such as wake/sleep, event routing, persistence, attention limits, permissions and hard safety boundaries, model invocation, tool execution, scheduling, and logging.

Resident decides what observations mean, what is interesting, what it wants to do, which available capabilities to use, what it wants to remember, whether to ask its owner, and whether it wants to revisit something later.

> Runtime enables Resident's life; it should not live it on Resident's behalf.

## Pending intentions

Pending intentions are persistent first-class state separate from ordinary memory.

A memory describes something Resident wants its future self to know. An intention describes something Resident may want its future self to continue, revisit, or do when circumstances permit.

For example:

```text
When the robot becomes available, inspect behind the sofa.
```

The initial representation should be deliberately small, conceptually containing an id, free-form content, creation time, and status. It must not grow into a conventional planner/task-management system unless actual Resident behavior demonstrates a need for one.

Pending questions to the owner have similar asynchronous characteristics, but their exact relationship to intentions and communication state can be decided during implementation.

## Observability

An emergent system must be inspectable enough to understand what happened when its behavior is surprising.

The runtime/journal should make it possible to reconstruct the externally relevant lifecycle of a wakeup, including approximately:

```text
wake event
→ context supplied
→ model interaction
→ tool calls and results
→ persisted state changes
→ outgoing communication / scheduled work
→ sleep
```

Observability should capture inputs, outputs, events, tool interactions, and state transitions needed for debugging and analysis. It does not require storing or exposing a model's private/internal reasoning process.

## Models

Resident must not depend on a specific AI model or provider.

Intelligence is a runtime capability; identity and continuity belong to Resident.

A future implementation may use local models, cloud models, several capability/cost tiers, or escalation between them. The exact policy is deliberately unspecified for now.

Model calls should eventually be observable enough to measure workload, latency, token usage, and cost. This will allow model choices to be based on actual Resident workloads rather than guesses made before the system exists.

## Connectors

Connectors expose capabilities and observations to Resident and should be independently replaceable and evolvable.

A connector may expose concepts such as:

- status and availability;
- resources;
- observations/events;
- actions;
- capabilities;
- constraints.

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

Physical co-location does not require logical integration. For example, a camera and microphone mounted on a robot may remain separate connector/device identities. Relationships such as `mounted_on`, `powered_by`, or correlated availability may later be declared or inferred by Resident.

## Memory

Persistent memory belongs to Resident rather than to the LLM context.

> LLM context is not memory.

The context supplied during a wakeup is temporary working memory. Information that should survive must be stored persistently.

The initial storage implementation can be deliberately simple, with SQLite as a strong v0 candidate.

Resident should have a small general memory interface conceptually similar to:

```text
remember(content)
recall(query)
update_memory(id, content)
forget(id)
```

The exact metadata and retrieval implementation should not be over-specified initially. Timestamps and provenance/source are likely useful; importance scores, confidence, tags, embeddings, and other metadata should be introduced when there is evidence that they are needed.

### Journal versus memory

Runtime maintains an append-only journal of what actually happened. Resident separately chooses what to retain as its own memory.

This distinction is intentional:

> Journal = what happened.
>
> Memory = what Resident chose to remember.

The journal supports debugging, auditability, and later analysis. It should not automatically become the entire reasoning context for every wakeup.

### World model

A dedicated structured world-model representation is not required initially.

Resident may begin by storing ordinary memories such as relationships, observations, hypotheses, and regularities. If experience shows that free-form memory is insufficient, structured entities/relationships or another representation can be introduced later.

## Communication

Communication should be transport-independent and asynchronous.

Conceptually, a message may contain text and/or attachments such as images and audio. v0 does not need to implement every modality, but the interface should not unnecessarily assume text-only communication.

Incoming owner messages wake Resident. Outgoing questions can remain pending indefinitely; a response may arrive immediately, much later, or not at all.

### Attention budget

Runtime protects the owner's attention with configurable limits. This is a hard mechanism around communication rather than merely a personality instruction.

Resident may internally formulate more questions than can be delivered. Questions may wait, be prioritized, become obsolete, be answered by Resident through further investigation, or potentially be combined.

The exact rate limits and urgency scheme remain open. Important/urgent communication may eventually have a separate policy, but the initial design should avoid an elaborate hard-coded priority system.

## Authority and safety

Owner identity and authority are explicit system concepts.

Normal owner instructions outrank Resident's autonomous goals. A future explicit `sudo`/override marker can communicate that an instruction must not be treated as a suggestion or balanced against Resident's own priorities.

Deterministic safety and system constraints remain above both Resident and owner instructions. Physical connectors should enforce hard boundaries that the reasoning model cannot override.

## Evolution principle

Prefer adding general mechanisms after observing real failure modes over predicting and implementing specific behaviors in advance.

Resident should remain an AI entity using capabilities, not gradually become a conventional rules engine with an LLM attached to it.
