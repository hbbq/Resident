# Resident

Resident is a persistent AI entity that inhabits an environment, learns about it through available connectors, interacts asynchronously with its owner, and may use physical embodiments such as mobile robots when available.

This repository is intentionally starting with principles rather than a detailed implementation. Resident is an experiment in persistent identity, emergent understanding, curiosity, and tool use. Mechanisms should be added when experience shows that they are needed rather than pre-programming the behavior we hope will emerge.

See [VISION.md](VISION.md) for the current product/behavior vision and [ARCHITECTURE.md](ARCHITECTURE.md) for the initial architectural principles.

## Minimal runtime

The first experimental vertical slice is a Python 3.12+ asynchronous process with durable SQLite state. It preserves Resident and owner identities, curated long-term memory, standing Owner guidance, pending intentions, communication, wake runs, an append-only journal, self-requested scheduled wakeups, and OpenAI Agents session bindings across restarts. Owner messages and due schedules become wake events; inference stops after each bounded model/tool exchange while terminal input and scheduling remain active.

### Declarative Resident instances

The runtime can host multiple independently configured Residents from startup-time YAML definitions. Pass `--residents-dir residents` (or set `RESIDENTS_DIR`); prompt references are resolved below `--prompt-root`, which defaults to the sibling `prompts` directory. Each stable definition ID receives its own database at `DATA_DIR/instances/<id>/resident.sqlite3`, including its durable identity, journal, schedules, tool actions, and Agents session binding. Editing its name, personality, role, or policy does not create a new identity. Shared connector events are polled once and fanned out only to matching `subscriptions`; tools are separately selected by `capabilities`, so observation never grants action authority.

A minimal Dungeon Master can be introduced without Python changes:

```yaml
# residents/dungeon-master.yaml
version: 1
id: dungeon-master
name: Dungeon Master
personality_prompt: dungeon-master.md
role: Run a persistent tabletop campaign.
agent:
  provider: openai-agents
  model: gpt-5.6-luna
  api_key_env: OPENAI_API_KEY
curator:
  model: gpt-5.6-luna
  api_key_env: OPENAI_API_KEY
capabilities: [messaging]
subscriptions: []
```

Definitions accept inline `personality`/`role` or `personality_prompt`/`role_prompt`, Agent settings, per-Resident Curator settings, capability grants, event subscriptions, an optional Telegram Owner transport, and optional body metadata. A Curator is disabled when its block or `model` is omitted; `base_url_env`, `batch_size`, and `max_batches` are optional. YAML aliases, unknown fields, prompt path traversal, and inline secret-shaped fields are rejected. Secrets are named with `*_env` references and resolved only while constructing local resources. A Telegram transport uses `token_env`, `owner_user_id_env`, and `owner_chat_id_env`; one resolved bot token may serve exactly one Resident. Unsuffixed terminal input targets `--default-resident` (`resident` by default), and configuration changes require restart.

Granting `messaging` exposes `messaging_send`. It writes to the process-shared durable mailbox and returns immediately; it is not RPC and does not await a reply. Its tool schema enumerates the configured Resident IDs that can be addressed; Owner communication remains separate through the Managed Agents `notify_owner` final output (or the legacy `send_owner_message` tool on older protocols). Messages default to a five-minute TTL and move from `pending` to `delivered` only when handed to the recipient event queue; expiry and delivery do not imply that a recipient read, understood, acted, or replied. A reply is another independent message.

Legacy environment/CLI startup remains available when no definitions directory is supplied. To explicitly move an existing singleton database into the normal `resident` instance layout before declarative startup, stop the runtime and run:

```powershell
python -m resident --data-dir .resident --migrate-legacy
```

The live adapter uses the beta OpenAI Agents API and defaults to `gpt-5.6-luna`. Resident restores one long-lived managed session rather than creating a session on process restart, stores its binding and immutable protocol descriptor in local SQLite, and submits subsequent wakes with `Idempotency-Key` request headers. Local function actions are claimed before execution and completed results are retained, so a re-delivered action returns its recorded result instead of repeating a display or Owner-message side effect; an interrupted action with an unknown outcome is reported rather than repeated automatically. OpenAI owns episodic/working context; Resident owns identity and the local durable Memory Store. Set an API key and choose a persistent data directory:

```powershell
$env:OPENAI_API_KEY = "..."
python -m resident --data-dir .resident
```

Enter an owner message at the prompt. New Managed Agents sessions return a structured final disposition and request Owner communication with `notify_owner`; Runtime durably queues and delivers it only after the model turn has completed. Replies are ordinary later Owner wakes. The older `send_owner_message` tool remains available only while recovering an already-active old-protocol session, and the Responses fallback retains its legacy tool protocol. Owner communication is rendered as `[Resident -> Owner] ...`. By default, the terminal otherwise shows only a small startup/shutdown status and actionable runtime failures, keeping routine spontaneous wakes nearly invisible. Pass `--verbose` or set `RESIDENT_VERBOSE=true` to show detailed diagnostics. Enter `/quit` to stop. Reusing the data directory reloads the same stable identities, local state, Agents binding, dispositions, and output queue. Display names and personality can be configured with `--resident-name`, `--owner-name`, and `--personality`; later configuration refreshes this metadata while retaining the same durable identity. `RESIDENT_MODEL`, `OPENAI_BASE_URL`, and the equivalent name/data environment variables may also be used. An existing saved Agent resource can be selected with `RESIDENT_OPENAI_AGENT_ID`; repository-defined instructions and local schemas remain authoritative session overrides. Set `RESIDENT_PROVIDER=openai-responses` (or the legacy alias `openai`) to use the temporary Responses fallback.

Before the sleeping prompt, normal output includes one concise readiness line for each enabled integration and an overall result. HomeOps is ready after its first valid latest-measurements poll; AgentController after its first valid snapshot and baseline handling; Telegram after webhook validation and a successful zero-wait `getUpdates` preflight; and ONVIF after every configured camera has either established a usable PullPoint subscription or failed its first attempt. Cameras are reported separately after the configured FFmpeg executable is found locally; startup does not connect to RTSP streams or capture frames. A `FAILED` line records the initial attempt and does not stop the existing background retry loop or provide continuous health monitoring. Detailed causes and later retry diagnostics remain available with `--verbose`.

Agent identity precedence is explicit: `RESIDENT_OPENAI_AGENT_ID` selects a saved reusable Agent. Adopting a different saved Agent, changing immutable instructions, or adding/renaming/changing a function contract causes an intentional, audited rollover; a local tool revocation keeps a compatibility handler for the old contract. Model, `--reasoning-effort`, and `--service-tier` changes are patched on an idle existing session and their last successfully applied values are stored separately from the immutable protocol descriptor. An in-place edit of a saved Agent does not silently replace its existing session snapshot. Missing/expired remote sessions and `--new-chapter` are recorded rollover reasons. Resident state and identity survive every rollover. The exact replacement create request is persisted before POST and a successful replacement is bound transactionally. The current Agents contract exposes neither create idempotency nor lookup by Resident's rollover token; if Resident stops after an attempt may have reached the service but before the returned session ID is durable, restart reports that explicit uncertain state and does not risk creating another replacement automatically.

Long-term memory is curated independently of the Resident model. Set `RESIDENT_CURATOR_MODEL` (and optionally `RESIDENT_CURATOR_API_KEY` / `RESIDENT_CURATOR_BASE_URL`) to enable startup catch-up and bounded incremental consolidation. The Curator checkpoints session-item progress transactionally, and every durable memory revision requires source references verified against the fetched session page. The Resident receives bounded `search_long_term_memory` and `get_long_term_memory` tools instead of the whole store on every wake. A replacement session gets a compact memory index and a transient handover when available. Explicit lasting Owner instructions use `set_owner_guidance` / `remove_owner_guidance`; their revision history is retained while the active set has deterministic entry, count, and serialized-size bounds. New sessions receive that active set in bootstrap; existing sessions receive durable, versioned additions, revisions, and removals only when it changes. Curator input remains an explicit field projection: tool arguments/results, encrypted reasoning, attachment payloads, and unknown structured fields are excluded. Text in that projection and all Curator output are deterministically scrubbed of recognizable credential-bearing structures (including authorization values, password-bearing URLs, private-key blocks, credential assignments, and common service tokens) before crossing or being persisted. This is an enforceable structural boundary, not a claim that arbitrary natural language can be perfectly classified as secret or non-secret.

## Optional Telegram Owner transport

Set all three environment variables below to enable a private, text-only Telegram bot transport:

```powershell
$env:RESIDENT_TELEGRAM_BOT_TOKEN = "..."
$env:RESIDENT_TELEGRAM_OWNER_USER_ID = "123456789"
$env:RESIDENT_TELEGRAM_OWNER_CHAT_ID = "123456789"
python -m resident --data-dir .resident
```

The numeric user and private-chat IDs are an explicit Owner binding provisioned outside Resident. Messages from another user, chat, or a group are ignored; there is no first-message auto-binding. Telegram uses long polling, so Resident needs outbound HTTPS access but no public inbound endpoint, and an existing bot webhook must be removed before use. Polling offsets and update de-duplication are scoped by a one-way token-derived bot identity, so changing bots in an existing data directory does not reuse the previous bot's checkpoint. The token and binding IDs are kept out of model context, journal payloads, normal diagnostics, and persisted state.

When configured, Telegram is authoritative for `notify_owner` delivery and the terminal mirrors attempted messages for local observability; that rendering is not counted as delivery. An authorized inbound update is acknowledged only after its canonical Owner communication and de-duplication record are committed. Inbound Owner messages remain pending until their normal `owner_message` wake completes; pending messages are recreated through that same wake path after restart. A Telegram send failure remains in the durable output retry state machine and never falls back to model output. Messages longer than Telegram's 4,096-character limit are prevented by the output schema. Terminal input remains active in parallel. If standard input is absent or closes, Resident continues running remotely; `/quit` remains available from an attached terminal. Poll and request timeouts can be set with `RESIDENT_TELEGRAM_POLL_SECONDS` and `RESIDENT_TELEGRAM_REQUEST_TIMEOUT_SECONDS`.

Resident can explicitly manage pending intentions, send owner messages, schedule a future wake, and invoke a read-only local time capability. An ongoing Agents session receives only the complete new trigger, correlation/recovery metadata, and any changed locally authoritative identity, capability, or standing-guidance state. Historical communication, intentions, handover, and memory awareness are not replayed on ordinary wakes. A new or replacement session instead receives one bootstrap containing identity and role, current capabilities, active guidance, pending intentions, bounded memory awareness, an available handover, and the trigger exactly once. When older context is useful, `search_communication` provides bounded newest-first message search by text, direction, and time, while `list_wake_history` provides bounded newest-first wake search by reason, source, status, and time. Both support offset pagination. Wake history exposes run metadata and an allowlisted projection of observable journal events; raw wake payloads, model content, tool arguments/results, provider continuation data, and ephemeral attachments are excluded at read time. These tools never alter intentions, delivery, or wake state, although their invocation is recorded like any other tool call. The runtime limits model tool use to eight rounds. Spontaneous messages (those outside an owner-initiated wake) default to three delivered messages per hour; excess messages are persisted as rejected and are never queued. Immediate replies during an owner wake do not consume that budget. Configure this policy with `--spontaneous-message-limit` and `--spontaneous-message-window-seconds` (or their `RESIDENT_...` environment-variable equivalents).

Schema version 12 removes the former local `memories` table and the `remember`, `recall`, `update_memory`, and `forget` tools. Upgrading an existing database discards those legacy records; no compatibility API remains.

The runtime persists a safe public snapshot of available capabilities. The first snapshot is silent; on later starts, or after an explicit programmatic registration/removal while running, additions, removals, and description/schema changes produce a normal `runtime` / `capabilities_changed` wake. Detection never invokes or tests a capability. Connectors may independently emit source-specific events when their visible world changes; Resident decides whether either kind of change warrants investigation or communication.

### Latency timeline diagnostics

Pass `--timeline` or set `RESIDENT_TIMELINE=true` to record opt-in structured `timeline` journal events. The timeline separates host queue wait, wake processing, provider preflight and rounds, local tool execution, HomeOps/display requests, Agents/Responses default-executor queue and worker time, Curator batches and tail, and event-loop lag aggregated per active wake. The Agents adapter uses the live session event stream for healthy wake and tool-result waits; startup, initial session input, interrupted or malformed streams, timeouts, and other uncertain states retain exact HTTP reconciliation. Timeline records distinguish stream waits and explicit reconciliation fallbacks from submission, polling, and item retrieval. Stream observations are transient and are not used as a durable replay cursor. Payloads contain safe identifiers, operation classes, timings, counts, configured timeouts, and outcomes; they exclude prompts, tool arguments or results, credentials, headers, request URLs, and attachment contents. `--verbose` also renders the records. Instrumentation does not change per-Resident serialization or Curator placement.

## Experimental HomeOps connector

Set `RESIDENT_HOMEOPS_URL` (or pass `--homeops-url`) to opt into read-only HomeOps observation. With no URL configured, Resident makes no HomeOps requests. The connector silently establishes a baseline from `GET /api/measurements/latest`, then polls every 30 seconds and emits one wake containing all values changed during that poll. New measurement points count as changes; timestamp-only updates do not. Polling failures are retried without waking Resident or discarding the last successful baseline. These recoverable failures are shown with verbose diagnostics and suppressed in the default terminal mode.

Resident can use `homeops_get_current_measurements` and `homeops_get_measurement_history` to investigate. History accepts a measurement `point_id`, optional ISO-8601 `from_time` and `to_time`, and an optional `limit` from 1 to 5,000. Configure polling and HTTP timeout with `--homeops-poll-seconds` / `RESIDENT_HOMEOPS_POLL_SECONDS` and `--homeops-request-timeout-seconds` / `RESIDENT_HOMEOPS_REQUEST_TIMEOUT_SECONDS`.

### HomeOps-backed displays

Set `RESIDENT_DISPLAYS` to a JSON array of display IDs to expose action-only text display capabilities. Displays reuse `RESIDENT_HOMEOPS_URL` and its request timeout; no separate display backend URL is configured. For example:

```powershell
$env:RESIDENT_HOMEOPS_URL = "http://homeops.local"
$env:RESIDENT_DISPLAYS = '[{"id":"display1","max_length":40}]'
```

For new Managed Agents sessions, each configured action-only display is a target-specific `display` final output rather than a function tool. `max_length` is optional and becomes that target's exact schema constraint. The runtime persists the disposition and output job before completing the wake, then a background dispatcher submits it to HomeOps. HTTP 204 means accepted/queued by HomeOps, not physically displayed. Delivery uses bounded at-least-once retries, so an uncertain interrupted attempt may be duplicated. Permanent or exhausted failures create one safe `output_delivery_failed` wake. Display IDs must be unique, 1-64 characters, and contain only letters, digits, underscores, or hyphens. Resident-specific instructions decide when and what to display; the schema only grants targets and constraints. Old Managed sessions retain `<id>_show_text` only until audited rollover, and the Responses fallback retains the legacy tool.

## Experimental AgentController connector

Set `RESIDENT_AGENTCONTROLLER_SNAPSHOT_PATH` (or pass `--agentcontroller-snapshot-path`) to AgentController's atomically published `output/dashboard.json` to opt into read-only workflow observation. With no path configured, Resident does not read AgentController data. The connector accepts only schema version 1 and treats AgentController's stable `state` values as authoritative; it does not inspect or derive workflow state from GitHub labels.

The first successful observation establishes a silent, durable baseline. Later additions, removals, or changes to an item's title, state, complexity, URL, or `updated_at` produce one factual `agentcontroller` / `workflow_changed` wake per poll. `refreshed_at` alone never produces a wake. Invalid or unavailable snapshots are retried without waking Resident or replacing the last successful baseline, and their diagnostics are visible only in verbose mode. The baseline survives Resident restarts, while AgentController's published `refreshed_at` indicates snapshot freshness and may lag GitHub.

Resident can use the read-only `agentcontroller_list_workflow_items` capability, optionally filtered by exact `owner/name` repository and limited to at most 100 results. v0 exposes no issue body or comment inspection, run-log access, notification rule, or mutation capability. Resident decides whether a factual change warrants Owner communication. The polling interval defaults to 60 seconds and can be configured with `--agentcontroller-poll-seconds` or `RESIDENT_AGENTCONTROLLER_POLL_SECONDS`.

## Experimental camera connector

Set `RESIDENT_CAMERAS` to a JSON array to opt into on-demand, read-only camera access. Each camera requires a stable `id`, display `name`, and secret `url`; `description` is optional safe metadata. For example:

```powershell
$env:RESIDENT_CAMERAS = '[{"id":"entry","name":"Entry camera","description":"Front entry","url":"rtsp://user:password@camera.local/stream"}]'
python -m resident --data-dir .resident
```

Resident can list configured cameras with `camera_list` without connecting to them, and can request one current frame with `camera_capture_frame`. FFmpeg must be installed and available as `ffmpeg` (or configured with `--ffmpeg-executable`). Each request connects only long enough to capture one JPEG frame; unavailable cameras and timeouts are normal tool outcomes. Frames are passed in memory to the model and are never written to Resident state or its journal. Camera URLs, credentials, and FFmpeg error output are not exposed to the model, normal diagnostics, or journal. The temporary Responses fallback uses `store=false`, including image continuations; the Agents path sends a frame only as a managed-session function result.

Capture defaults to RTSP over TCP, an 8-second timeout, 1280x720 maximum output dimensions, and a 2 MB encoded-frame limit. These can be adjusted with `--camera-rtsp-transport`, `--camera-capture-timeout-seconds`, `--camera-max-width`, `--camera-max-height`, and `--camera-max-bytes`, or their corresponding `RESIDENT_...` environment variables. With the Agents adapter, frame bytes are submitted only as the corresponding function result and are not written to Resident's SQLite store.

`RESIDENT_CAMERAS` remains startup configuration. An embedding with a genuinely refreshable camera source can replace the connector's camera set explicitly; added, removed, or changed cameras then produce one `camera` / `cameras_changed` wake containing only safe IDs, names, and descriptions. Endpoint-only changes are detected but the endpoint and credentials are never included in the event.

ONVIF event probing is a separate, opt-in experiment. Add an explicit `onvif` object to a camera; Resident never infers an ONVIF endpoint or credentials from its RTSP URL:

```powershell
$env:RESIDENT_CAMERAS = '[{"id":"entry","name":"Entry camera","url":"rtsp://user:password@camera.local/stream","onvif":{"endpoint":"http://camera.local/onvif/device_service","username":"local-onvif-user","password":"local-onvif-password"}}]'
```

While the connector is active it queries the advertised Event Service topics and property schemas, creates a PullPoint subscription, and pulls notifications with finite timeouts. The first value of each advertised property establishes a baseline; subsequent changes emit a factual `onvif_property_changed` wake containing the camera, topic, source identifiers, declared type, and previous/current values. Repeated values do not wake Resident. In verbose mode, operator diagnostics show advertised topic/schema names and observed payload field names only. Raw XML, payload values, service/subscription URLs, credentials, and transport error text are not logged. The connector does not capture frames, classify events, notify the Owner, or assign meaning to motion/people/etc.; Resident decides what to do with each factual transition. Offline cameras and subscription failures are retried without producing a wake.

For finite manual validation without starting Resident or requiring an OpenAI API key, set `RESIDENT_CAMERAS` locally as above and run:

```powershell
python -m resident.onvif_probe --camera-id entry --pulls 3
```

The probe prints the same sanitized topic/field shape information and attempts to unsubscribe before exiting. Request, pull, and retry timing can be configured for the runtime with `--camera-onvif-request-timeout-seconds`, `--camera-onvif-pull-timeout-seconds`, and `--camera-onvif-retry-seconds`, or the corresponding `RESIDENT_...` variables.

Run the deterministic offline lifecycle suite without credentials:

```powershell
python -m unittest discover -s tests -v
```
