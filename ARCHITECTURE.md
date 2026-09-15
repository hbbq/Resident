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

Resident uses a hybrid event-driven runtime rather than requiring a permanent reasoning loop.

Typical wake sources include:

- connector events;
- capability availability changes, such as a robot coming online;
- messages from the owner;
- scheduled wakeups;
- wakeups previously requested by Resident itself.

During a wakeup Resident receives an appropriate working context, reasons and possibly acts, persists anything it wants its future self to retain, and may then sleep again.

Scheduling is a mechanism, not a collection of hard-coded behaviors. Resident should be able to request a future wakeup with a reason/context rather than requiring dedicated classes such as `TemperatureMonitor` or `RobotExplorationBehavior`.

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
