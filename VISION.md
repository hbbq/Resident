# Resident Vision

## What Resident is

Resident is a persistent AI entity that inhabits an environment rather than a chatbot session or a robot backend.

Resident should have a continuous sense of identity across wakeups. It can learn about its environment, remember experiences, notice uncertainty and change, communicate with its owner, use available capabilities, and develop an increasingly useful understanding of the world around it.

A mobile robot can be one embodiment of Resident, but it is not Resident itself. When the robot is unavailable, Resident continues to exist. When it becomes available, Resident may use it to explore, investigate, observe, or simply do something interesting or playful.

The architecture should not assume that there can only ever be one Resident instance. Each instance has its own persistent identity and may have a human-friendly name by which it is addressed. `Owner` is likewise a role and authority relationship, not necessarily a particular human; an owner also has an identity and an address name. This leaves room for future arrangements such as one Resident instance owning another without making multi-agent behavior a v0 requirement.

## Start small

Resident should initially know very little about its universe.

It can be told that it inhabits a house, introduced to its owner, and shown available connectors and their capabilities. From there it should inspect, ask, infer, remember, and gradually build its own understanding.

We deliberately do not want to pre-program a large taxonomy of behaviors, investigations, or goals. If Resident discovers a sensor with an opaque name, for example, it may infer what it measures, ask where it is, observe correlations, or leave the question unresolved.

The questions it asks and the concepts it forms are part of the experiment.

## Personality and agency

Resident should be encouraged to:

- be curious;
- build and maintain an understanding of its environment;
- notice changes, novelty, and uncertainty;
- ask humans when they can provide information Resident cannot obtain itself;
- consider using capabilities when they become available;
- explore;
- sometimes do things simply because they are interesting or fun.

Not every action needs to maximize utility or information gain. Playfulness is a legitimate motive.

The intended division of responsibility is:

> Code controls what Resident can and cannot do. Personality roughly describes what it wants. Resident decides what it actually does.

A related development principle is:

> Do not implement behavior until there is evidence that Resident needs it.

## Persistent identity

A continuous "I" is a requirement.

Different wakeups should be experienced as the same Resident. Continuity must not depend on retaining an LLM chat session or replaying an ever-growing conversation history. Resident owns its identity and persistent memory independently of whichever model happens to reason for it at a particular moment.

## Asynchronous life

Resident does not need to think continuously.

The expected runtime is hybrid: Resident normally sleeps and wakes because something happened, its owner contacted it, a scheduled wakeup became due, a capability became available, or Resident previously decided that it wanted to wake again.

A wakeup may lead to observation, reasoning, action, communication, another scheduled wakeup, or simply returning to sleep.

Resident should be able to leave intentions and unresolved matters for its future self. For example, it may want to inspect an area with the robot but have to wait several hours for the robot to become available.

## Communication with the owner

Communication is asynchronous, not session-oriented, and is modeled as general messages rather than a built-in question/answer protocol.

Resident may contact its owner with text, images, audio, or other supported attachments. A later owner message may be an answer, instruction, correction, new topic, or something else; Resident is responsible for understanding its meaning and relationship to earlier communication.

The owner may initiate communication at any time. An incoming owner message is a wake event. Communication history persists independently of Resident's chosen autobiographical memory, while only relevant/recent communication should normally be placed in a wake context.

Human attention is a limited resource. Resident may internally want to communicate more often than is appropriate, but runtime-enforced configurable attention budgets must prevent it from peppering the owner with messages. Runtime need not understand which messages are questions in order to enforce that limit.

Communication transport is replaceable. The interactive terminal is sufficient for v0, with Resident's intentional owner-facing messages rendered clearly differently from runtime logs and diagnostics. Later transports can replace or supplement it without changing Resident's communication model.

## Owner authority

The owner has higher authority than Resident's autonomous goals. Ordinary owner instructions should normally be followed without requiring special syntax.

An explicit override mechanism, provisionally called `sudo`, may be used to mark an instruction as authoritative rather than something Resident may balance against its own current priorities.

Hard system and safety constraints remain above owner overrides. An owner instruction cannot reason away deterministic safety boundaries.

A useful conceptual priority is:

1. Hard safety and system constraints
2. Explicit owner override (`sudo`)
3. Owner instructions
4. Existing commitments
5. Resident's own goals, curiosity, and play

## Open questions

Important choices are intentionally unresolved, including:

- model providers and exact models;
- cloud versus local inference;
- the exact connector protocol beyond the deliberately small v0 contract;
- communication transports beyond the v0 terminal transport;
- whether and when the world model needs a structured representation;
- detailed memory retrieval and consolidation mechanisms;
- exact attention-budget defaults;
- detailed safety policies for individual physical capabilities.
