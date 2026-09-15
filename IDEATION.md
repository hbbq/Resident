# Resident Ideation

This file guides idea generation for Resident. Read `VISION.md` first; ideas should extend the experiment without quietly turning Resident into a conventional automation system, chatbot, rules engine, or robot controller.

## What to look for

The most valuable ideas usually give Resident **more of a world to inhabit** rather than more prescribed behavior.

Prioritize ideas in areas such as:

- **Connectors** to systems, devices, information sources, services, software, sensors, or embodiments that could become part of Resident's observable world.
- **Capabilities** that let Resident inspect, query, observe, communicate with, or act through something when it chooses to.
- **Events** that let Resident notice factual changes in connected systems without deciding in advance what those changes mean.
- **Communication transports** or media that expand how Resident and Owner can communicate without changing the underlying communication model.
- **Embodiments and interfaces** that give Resident new ways to observe or interact with the physical world.
- **Small enabling mechanisms** that become necessary because real use has exposed a limitation in persistence, memory, wake processing, context, safety, observability, or capability use.
- **Experiments** that help us learn how Resident behaves with its existing architecture or with one new capability.

Connectors are especially fertile territory. Think broadly about things Resident could reasonably be introduced to and then learn to understand through their capabilities and events.

## Core ideation principle

Prefer giving Resident new **possibilities and information** over telling it what to do with them.

A useful rule is:

> Capabilities describe what Resident can do. Events tell Resident what happened. Resident decides what it means.

An idea like "let Resident read AgentController issues and observe workflow changes" fits well.

An idea like "when an issue gets `agent:needs-input`, send Owner a Telegram message" does not. The first expands Resident's world; the second pre-programs Resident's interpretation and behavior.

## Avoid

Do not propose features whose main purpose is to hardcode intelligent behavior that the model can already decide for itself.

In particular, avoid:

- fixed `if event X -> action Y` behavior policies;
- predefined investigation recipes such as automatically checking cameras whenever a particular sensor changes;
- hardcoded notification rules for semantic conditions Resident can reason about;
- large taxonomies of goals, behaviors, moods, situations, or world concepts;
- a structured world model before observed limitations justify one;
- generic frameworks, protocol layers, orchestration machinery, or abstractions merely because they might be useful later;
- features that move interpretation of the world from Resident into deterministic runtime code;
- turning Resident into an AgentController client that autonomously edits code or workflow unless a later experiment explicitly motivates and bounds such authority.

Do not generate mechanisms simply to make Resident appear more agentic. First ask whether the desired behavior could emerge from the personality, existing memory, events, and available capabilities.

## Evidence before mechanism

Resident is deliberately experimental. Existing behavior is evidence.

When considering a runtime or architecture idea, ask:

1. What observed limitation or experiment motivates this?
2. Could Resident already handle it through reasoning, memory, Owner interaction, or an existing capability?
3. Could a smaller connector, capability, event, or prompt/context change expose enough information for Resident to handle it itself?
4. If deterministic machinery is genuinely needed, what is the smallest mechanism that solves the observed problem without prescribing future behavior?

A good reason for a mechanism is something like crash recovery causing duplicate external effects. "Resident might someday need this" is usually not enough.

## Connector ideas

For a connector, consider both sides independently:

- What can Resident **inspect or do** through capabilities?
- What factual changes are worth exposing as **events**?

Keep both surfaces small. Do not expose every endpoint of an external system just because it exists. Prefer capabilities that are understandable to Resident and useful for exploration.

Events should describe observations, not conclusions. For example, "measurement changed", "device became available", "issue labels changed", or "new item appeared" are better than "Owner should be notified" or "this is suspicious".

Read-only is a good default for new connectors. Add write/action capabilities only when there is a concrete reason and their authority and safety boundaries are understood.

External systems do not need to implement a Resident protocol. A Resident-side connector can adapt whatever interface already exists.

## Scope and style of ideas

Prefer ideas that are:

- small enough to experiment with;
- independently useful or interesting;
- reversible;
- observable so we can learn from Resident's response;
- minimally opinionated about what Resident will actually choose to do.

Playful or apparently unnecessary capabilities can still be good ideas. Resident is allowed to explore and do things because they are interesting or fun; ideation does not need to optimize only for productivity.

It is also valid for an ideation pass to produce **no implementation idea**. Sometimes the best next step is to let Resident live with the current capabilities and observe what it does.

## Creating issues

When an idea is worth preserving, create a focused issue describing the new possibility or observed limitation and why it is interesting. Describe constraints that protect the Resident vision, but avoid prematurely specifying implementation details.

For AgentController-managed work, an idea should enter through the normal `agent:idea` workflow and be allowed to go through triage and investigation before implementation decisions are made.
