# Resident

Resident is a persistent AI entity that inhabits an environment, learns about it through available connectors, interacts asynchronously with its owner, and may use physical embodiments such as mobile robots when available.

This repository is intentionally starting with principles rather than a detailed implementation. Resident is an experiment in persistent identity, emergent understanding, curiosity, and tool use. Mechanisms should be added when experience shows that they are needed rather than pre-programming the behavior we hope will emerge.

See [VISION.md](VISION.md) for the current product/behavior vision and [ARCHITECTURE.md](ARCHITECTURE.md) for the initial architectural principles.

## Minimal runtime

The first experimental vertical slice is a Python 3.12+ asynchronous process with durable SQLite state. It preserves Resident and owner identities, standing Owner guidance, pending intentions, communication, wake runs, an append-only journal, self-requested scheduled wakeups, and OpenAI Conversation bindings across restarts. Owner messages and due schedules become wake events; inference stops after each bounded model/tool exchange while terminal input and scheduling remain active.

### Declarative Resident instances

See [CAPABILITIES.md](CAPABILITIES.md) for the generated catalog of available tools, outputs, subscription selectors, and their setup requirements. It describes platform choices across configurations rather than the grants in any one Resident definition. Regenerate it with `python -m resident.catalog` and check for drift with `python -m resident.catalog --check`; the offline unittest suite also checks it.

The runtime can host multiple independently configured Residents from startup-time YAML definitions. Pass `--residents-dir residents` (or set `RESIDENTS_DIR`); prompt references are resolved below `--prompt-root`, which defaults to the sibling `prompts` directory. Each stable definition ID receives its own database at `DATA_DIR/instances/<id>/resident.sqlite3`, including its durable identity, journal, schedules, tool actions, and Conversation binding. Editing its name, personality, role, or policy does not create a new identity. Shared connector events are polled once and fanned out only to matching `subscriptions`; tools are separately selected by `capabilities`, so observation never grants action authority.

A minimal Dungeon Master can be introduced without Python changes:

```yaml
# residents/dungeon-master.yaml
version: 1
id: dungeon-master
name: Dungeon Master
personality_prompt: dungeon-master.md
role: Run a persistent tabletop campaign.
agent:
  provider: openai-responses
  model: gpt-5.6-luna
  api_key_env: OPENAI_API_KEY
capabilities: [messaging]
outputs: [notify_owner]
subscriptions: []
```

Definitions accept inline `personality`/`role` or `personality_prompt`/`role_prompt`, model settings, callable `capabilities` grants, terminal `outputs` grants, event subscriptions, an optional Telegram Owner transport, and optional body metadata. `capabilities` controls tools whose results can be used while reasoning during the current wake. `outputs` separately authorizes terminal side effects whose delivery happens after the turn; supported identifiers are `notify_owner` and `display/<display-id>`. A configured Owner route or display makes an output available but does not grant it. Every output grant must name an available route, and `outputs: []` authorizes only a silent disposition. YAML aliases, unknown fields, prompt path traversal, and inline secret-shaped fields are rejected. Secrets are named with `*_env` references and resolved only while constructing local resources. A Telegram transport uses `token_env`, `owner_user_id_env`, and `owner_chat_id_env`; one resolved bot token may serve exactly one Resident. Unsuffixed terminal input targets `--default-resident` (`resident` by default), and configuration changes require restart.

### External application capabilities

Keeper's Realm v1 integration uses the service's native routes. Set
`KEEPER_REALM_GAME_ID` to an existing game ID and `KEEPER_REALM_ACTOR_ID` to an
existing creature ID in that game, then configure `realm.base_url` in
`residents/keeper.yaml` for the trusted Realm service. The game and actor must
be seeded before play. Keeper reads both the actor projection and trusted state
on every wake. Mutations use the latest Realm revision and a stable tool-call-derived idempotency key. Successful calls return the mutation result and fresh views. A conflict requires reassessment; an uncertain result must not be repeated with a new key. Realm v1 has no authentication or operation lookup, so keep it inside a trusted network boundary and inspect Realm state after uncertain mutations.
`keeper_history: true` enables an optional local interaction archive for Realm-backed Residents. It records submitted wake input, Realm snapshots, model calls and tool results for inspection; it is never replayed into the Conversation.
`realm_world_patch` accepts Realm's seven patch sections with its native field
names, including `containment[].child_id` and `parent_id`, and
`observations[].actor_id`. A single patch can create an entity and refer to it
from containment and observations. Rejected requests include bounded Realm
validation details when available, together with the reread guidance.
Entity creation and updates also accept optional `appearance` for canonical,
observable visual characteristics, separate from `description` and player
projection fields.


A Resident definition can pin a small capability catalog for an external application. Providers are instance scoped: their URL, optional bearer-token environment reference, immutable bindings, and operation catalog are available only to that Resident. Grant the provider ID to authorize every configured operation, or grant individual tool names. Tool names must begin with `<provider-id>_`.

```yaml
capabilities: [realm]
external_applications:
  - id: realm
    description: Persistent game state
    base_url: http://realm.local
    bearer_token_env: REALM_TOKEN # optional
    request_timeout_seconds: 10
    bindings:
      game_id: campaign-1
    operations:
      - name: realm_apply_damage
        operation: apply_damage
        description: Apply validated damage to a character.
        mutating: true
        input_schema:
          type: object
          properties:
            character_id: {type: string}
            amount: {type: integer, minimum: 1}
          required: [character_id, amount]
          additionalProperties: false
      - name: realm_get_operation
        operation: get_operation
        description: Reconcile an operation by request identifier.
        input_schema:
          type: object
          properties:
            request_id: {type: string}
          required: [request_id]
          additionalProperties: false
```

Resident sends `POST <base_url>/api/capabilities/invoke` with JSON fields `operation`, `request_id`, `arguments`, and `bindings`. The application returns a JSON object or array. Resident validates model arguments against the pinned schema, keeps bindings outside model control, limits requests to 256 KiB and responses to 1 MiB, and never discovers or grants operations from the remote service. A configured token is sent only as an `Authorization: Bearer` header.

Calls have a bounded timeout and no automatic retry. Resident derives a stable opaque external `request_id` from the durable Conversation/tool-call identity. A timeout or transport failure from a mutating operation returns `unknown_outcome` with that ID; a separately configured reconciliation operation can query it. Authentication, rejection, conflict, unavailability, timeout, and invalid-response failures are classified without returning the URL, credential, headers, or upstream response body. An unavailable application does not prevent startup or remove its tools; calls fail deterministically until it is available. Catalog changes require a configuration restart and are supplied on subsequent Responses; runtime authorization takes effect before execution.

Granting `messaging` exposes `messaging_send`. It writes to the process-shared durable mailbox and returns immediately; it is not RPC and does not await a reply. Its tool schema enumerates the configured Resident IDs that can be addressed; Owner communication remains separate and requires an explicit `outputs: [notify_owner]` grant plus an available Owner route. Owner messages use the `notify_owner` terminal output and background delivery. Messages default to a five-minute TTL and move from `pending` to `delivered` only when handed to the recipient event queue; expiry and delivery do not imply that a recipient read, understood, acted, or replied. A reply is another independent message.

Environment/CLI startup remains available without a definitions directory. Set an API key and choose a persistent data directory:

```powershell
$env:OPENAI_API_KEY = "..."
python -m resident --data-dir .resident
```

Resident uses Responses + Conversations exclusively. A fresh instance establishes local identity and capability state, creates an empty OpenAI Conversation, and persists its ID before inference. Restart reuses that ID. Each request supplies current instructions, personality, role, standing guidance, model/reasoning/service tier, authorized tools and strict disposition schema. Only the new wake and observations enter initial request input; continuations submit function results to the same Conversation. No history replay or `previous_response_id` is used.

Every wake ends in a locally validated `{"outputs": [...]}` disposition. Empty outputs mean intentional silence; refusal, incomplete Responses or invalid/missing dispositions fail the wake. Owner/display jobs are persisted before completion and delivered separately. Enter an Owner message at the prompt, or `/quit` to stop. Use `--verbose` for detailed diagnostics. Names/personality and model configuration refresh on restart without changing the local identity or Conversation binding. Set `RESIDENT_MODEL`, `OPENAI_BASE_URL`, `--reasoning-effort`, and `--service-tier` as appropriate.

Before the sleeping prompt, normal output includes one concise readiness line for each enabled integration and an overall result. HomeOps is ready after its first valid latest-measurements poll; AgentController after its first valid snapshot and baseline handling; Telegram after webhook validation and a successful zero-wait `getUpdates` preflight; and ONVIF after every configured camera has either established a usable PullPoint subscription or failed its first attempt. Cameras are reported separately after the configured FFmpeg executable is found locally; startup does not connect to RTSP streams or capture frames. A `FAILED` line records the initial attempt and does not stop the existing background retry loop or provide continuous health monitoring. Detailed causes and later retry diagnostics remain available with `--verbose`.

### Disposable state and recovery

This schema has no upgrade path. Stop Resident and manually remove existing instance databases before running this implementation. Old Agents sessions, guidance, memories and history are abandoned. Resident never deletes a database automatically.

Responses are checkpointed before submission and after receipt. Completed tool results are replayable by Conversation/call ID without repeating effects; an interrupted tool execution returns an unknown outcome. A recorded final Response can be finalized locally after restart. Ambiguous HTTP submissions or interrupted tool loops block further inference: inspect the state, fix the code, or manually reset the disposable database. There is no remote reconciliation, automatic inference retry, bootstrap, handover or rollover subsystem.

Conversation history initially provides episodic continuity. Curator and semantic memory are removed. Standing guidance remains separate authoritative local state, changed only in an authenticated Owner wake and supplied on every request. Optional `agent.compact_threshold` (or `--compact-threshold` / `RESIDENT_COMPACT_THRESHOLD`) enables Responses server compaction; it is disabled by default and long-running Conversation behavior remains to be observed. Automatic truncation is disabled. Cached input remains part of total input accounting; growing history can reach the model's context limit.

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

Resident can manage intentions, schedule wakes and invoke authorized connectors. New input contains the trigger, timestamp, pending intentions and fresh Realm observations when configured; current instructions hold identity, role and guidance. Bounded `search_communication` and `list_wake_history` tools retrieve local history on demand. History search exposes safe journal summaries rather than raw prompts, tool payloads or images. Tool calls execute sequentially, with eight rounds by default and a bounded total call count. Spontaneous Owner messages have a configurable delivered-message budget; replies during Owner wakes do not consume it. Use `--spontaneous-message-limit` and `--spontaneous-message-window-seconds` to adjust this policy.

The runtime persists a safe public snapshot of available capabilities. The first snapshot is silent; on later starts, or after an explicit programmatic registration/removal while running, additions, removals, and description/schema changes produce a normal `runtime` / `capabilities_changed` wake. Detection never invokes or tests a capability. Connectors may independently emit source-specific events when their visible world changes; Resident decides whether either kind of change warrants investigation or communication.

### Latency timeline diagnostics

Use `--timeline` / `RESIDENT_TIMELINE=true` to record queue wait, model requests, tool execution, connector requests, executor queue/worker time and event-loop lag. Output delivery runs independently. Timing records exclude prompts, tool arguments/results, credentials, URLs and images. `--verbose` also renders these records.

## Experimental HomeOps connector

Set `RESIDENT_HOMEOPS_URL` (or pass `--homeops-url`) to opt into read-only HomeOps observation. With no URL configured, Resident makes no HomeOps requests. The connector silently establishes a baseline from `GET /api/measurements/latest`, then polls every 30 seconds and emits one wake containing all values changed during that poll. New measurement points count as changes; ordinary timestamp-only updates do not. Contact points also wake when HomeOps' `lastOpened` or `lastClosed` changes, even if the current value stays closed. Their existing change payload includes `old_last_opened`, `new_last_opened`, `old_last_closed`, and `new_last_closed`; unknown times remain null. The initial baseline remains silent even when it contains known contact times. Polling failures are retried without waking Resident or discarding the last successful baseline. These recoverable failures are shown with verbose diagnostics and suppressed in the default terminal mode.

Resident can use `homeops_get_current_measurements` and `homeops_get_measurement_history` to investigate. History accepts a measurement `point_id`, optional ISO-8601 `from_time` and `to_time`, and an optional `limit` from 1 to 5,000. The argument-free `homeops_get_weather_forecast` calls `GET /api/forecast` only when invoked for forward-looking weather. It returns the HomeOps forecast object with its metadata, periods in source order, and freshness/stale fields intact. An unavailable or disabled forecast produces a tool error without affecting measurement access or polling; forecasts do not generate measurement wakes. Grant `homeops` to include all three tools (as Watcher does), or grant `homeops_get_weather_forecast` by name. Configure polling and HTTP timeout with `--homeops-poll-seconds` / `RESIDENT_HOMEOPS_POLL_SECONDS` and `--homeops-request-timeout-seconds` / `RESIDENT_HOMEOPS_REQUEST_TIMEOUT_SECONDS`.


### HomeOps-backed displays

Set `RESIDENT_DISPLAYS` to a JSON array of display IDs to expose action-only text display capabilities. Displays reuse `RESIDENT_HOMEOPS_URL` and its request timeout; no separate display backend URL is configured. For example:

```powershell
$env:RESIDENT_HOMEOPS_URL = "http://homeops.local"
$env:RESIDENT_DISPLAYS = '[{"id":"display1","max_length":40}]'
```

With Responses, each configured action-only display is a target-specific `display` final output rather than a function tool. `max_length` is optional and becomes that target's exact schema constraint. The runtime persists the disposition and output job before completing the wake, then a background dispatcher submits it to HomeOps. HTTP 204 means accepted/queued by HomeOps, not physically displayed. Delivery uses bounded at-least-once retries, so an uncertain interrupted attempt may be duplicated. Permanent or exhausted failures create one safe `output_delivery_failed` wake. Display IDs must be unique, 1-64 characters, and contain only letters, digits, underscores, or hyphens. Resident-specific instructions decide when and what to display; the schema only grants targets and constraints.

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

Resident can list configured cameras with `camera_list` without connecting to them, and can request one current frame with `camera_capture_frame`. FFmpeg must be installed and available as `ffmpeg` (or configured with `--ffmpeg-executable`). Each request connects only long enough to capture one JPEG frame; unavailable cameras and timeouts are normal tool outcomes. Frames are passed in memory to the model and are never written to Resident state or its journal. Camera URLs, credentials, and FFmpeg error output are not exposed to the model, normal diagnostics, or journal. Frames are sent as multimodal function results and become remote Conversation content; local SQLite retains only result metadata, not image bytes.

Capture defaults to RTSP over TCP, an 8-second timeout, 1280x720 maximum output dimensions, and a 2 MB encoded-frame limit. These can be adjusted with `--camera-rtsp-transport`, `--camera-capture-timeout-seconds`, `--camera-max-width`, `--camera-max-height`, and `--camera-max-bytes`, or their corresponding `RESIDENT_...` environment variables. Frame bytes are submitted only as the corresponding function result and are not written to Resident's SQLite store.

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
