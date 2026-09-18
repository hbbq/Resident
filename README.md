# Resident

Resident is a persistent AI entity that inhabits an environment, learns about it through available connectors, interacts asynchronously with its owner, and may use physical embodiments such as mobile robots when available.

This repository is intentionally starting with principles rather than a detailed implementation. Resident is an experiment in persistent identity, emergent understanding, curiosity, and tool use. Mechanisms should be added when experience shows that they are needed rather than pre-programming the behavior we hope will emerge.

See [VISION.md](VISION.md) for the current product/behavior vision and [ARCHITECTURE.md](ARCHITECTURE.md) for the initial architectural principles.

## Minimal runtime

The first experimental vertical slice is a Python 3.12+ asynchronous process with durable SQLite state. It preserves Resident and owner identities, memory, pending intentions, communication, wake runs, an append-only journal, and self-requested scheduled wakeups across restarts. Owner messages and due schedules become wake events; inference stops after each bounded model/tool exchange while terminal input and scheduling remain active.

The live adapter uses the beta OpenAI Agents API and defaults to `gpt-5.6-luna`. Resident creates one long-lived managed session with the first accepted wake as its required initial input, stores its replaceable session binding in local SQLite, and submits subsequent wakes with `Idempotency-Key` request headers. Local function actions are claimed before execution and completed results are retained, so a re-delivered action returns its recorded result instead of repeating a display or Owner-message side effect; an interrupted action with an unknown outcome is reported rather than repeated automatically. OpenAI owns conversational turn history; Resident continues to own identity, durable memory, schedules, event filtering, local policy, and all HomeOps/device and Owner transports. Set an API key and choose a persistent data directory:

```powershell
$env:OPENAI_API_KEY = "..."
python -m resident --data-dir .resident
```

Enter an owner message at the prompt. All intentional Resident-to-Owner communication, including direct replies, goes through `send_owner_message` and is rendered as `[Resident -> Owner] ...`. Model-returned text is a wake result for journaling and diagnostics, not a second communication transport; it is hidden by default and shown only with verbose diagnostics. A rejected or failed send never falls back to model-returned text. By default, the terminal otherwise shows only a small startup/shutdown status and actionable runtime failures, keeping routine spontaneous wakes nearly invisible. Pass `--verbose` or set `RESIDENT_VERBOSE=true` to show detailed wake, context, model, tool, memory, connector, and metric diagnostics. Verbosity affects terminal presentation only; the structured journal remains complete. Enter `/quit` to stop. Reusing the data directory reloads the same stable identities, local state, and Agents session binding. Display names and personality can be configured on initial provisioning with `--resident-name`, `--owner-name`, and `--personality`; persisted identity is authoritative on later runs. `RESIDENT_MODEL`, `OPENAI_BASE_URL`, and the equivalent name/data environment variables may also be used. An existing saved Agent resource can be selected with `RESIDENT_OPENAI_AGENT_ID`; repository-defined instructions and local function schemas remain authoritative session overrides. Set `RESIDENT_PROVIDER=openai-responses` (or the legacy alias `openai`) to use the temporary Responses fallback.

Before the sleeping prompt, normal output includes one concise readiness line for each enabled integration and an overall result. HomeOps is ready after its first valid latest-measurements poll; AgentController after its first valid snapshot and baseline handling; Telegram after webhook validation and a successful zero-wait `getUpdates` preflight; and ONVIF after every configured camera has either established a usable PullPoint subscription or failed its first attempt. Cameras are reported separately after the configured FFmpeg executable is found locally; startup does not connect to RTSP streams or capture frames. A `FAILED` line records the initial attempt and does not stop the existing background retry loop or provide continuous health monitoring. Detailed causes and later retry diagnostics remain available with `--verbose`.

Agent identity precedence is explicit: `RESIDENT_OPENAI_AGENT_ID` overrides a different persisted binding and causes a replacement session. When it is unset, the persisted Agent ID provides continuity and is reused if the remote session has disappeared. In both cases, repository-defined instructions and tools are still applied to the session.

## Optional Telegram Owner transport

Set all three environment variables below to enable a private, text-only Telegram bot transport:

```powershell
$env:RESIDENT_TELEGRAM_BOT_TOKEN = "..."
$env:RESIDENT_TELEGRAM_OWNER_USER_ID = "123456789"
$env:RESIDENT_TELEGRAM_OWNER_CHAT_ID = "123456789"
python -m resident --data-dir .resident
```

The numeric user and private-chat IDs are an explicit Owner binding provisioned outside Resident. Messages from another user, chat, or a group are ignored; there is no first-message auto-binding. Telegram uses long polling, so Resident needs outbound HTTPS access but no public inbound endpoint, and an existing bot webhook must be removed before use. Polling offsets and update de-duplication are scoped by a one-way token-derived bot identity, so changing bots in an existing data directory does not reuse the previous bot's checkpoint. The token and binding IDs are kept out of model context, journal payloads, normal diagnostics, and persisted state.

When configured, Telegram is authoritative for `send_owner_message` delivery and the terminal mirrors attempted messages for local observability; that rendering is not counted as delivery. An authorized inbound update is acknowledged only after its canonical Owner communication and de-duplication record are committed. Inbound Owner messages remain pending until their normal `owner_message` wake completes; pending messages are recreated through that same wake path after restart. A Telegram send failure is recorded as `transport_failed`; it is not retried durably and does not fall back to model output. Messages longer than Telegram's 4,096-character limit are sent in content-preserving chunks. Terminal input remains active in parallel. If standard input is absent or closes, Resident continues running remotely; `/quit` remains available from an attached terminal. Poll and request timeouts can be set with `RESIDENT_TELEGRAM_POLL_SECONDS` and `RESIDENT_TELEGRAM_REQUEST_TIMEOUT_SECONDS`.

Resident can explicitly manage semantically classified memory and pending intentions, send owner messages, schedule a future wake, and invoke a read-only local time capability. Memories carry Resident-chosen kind, importance, confidence, provenance, and a standing-guidance marker alongside their content and write origin. Recall remains text-relevance-gated and bounded, using that metadata to rank relevant memories so durable knowledge competes more effectively with transient observations. Separately, wake context includes a small bounded set of Owner-derived preferences and rules that Resident marked as standing, even when wake text is unrelated. Resident interprets this guidance; the runtime does not turn it into event/action rules. Wake context contains the complete trigger but only bounded selections of memories, standing guidance, and communication. When older context is useful, `search_communication` provides bounded newest-first message search by text, direction, and time, while `list_wake_history` provides bounded newest-first wake search by reason, source, status, and time. Both support offset pagination. Wake history exposes run metadata and an allowlisted projection of observable journal events; raw wake payloads, model content, tool arguments/results, provider continuation data, and ephemeral attachments are excluded at read time. These tools never alter memories, intentions, delivery, or wake state, although their invocation is recorded like any other tool call. The runtime limits model tool use to eight rounds. Spontaneous messages (those outside an owner-initiated wake) default to three delivered messages per hour; excess messages are persisted as rejected and are never queued. Immediate replies during an owner wake do not consume that budget. Configure this policy with `--spontaneous-message-limit` and `--spontaneous-message-window-seconds` (or their `RESIDENT_...` environment-variable equivalents).

The runtime persists a safe public snapshot of available capabilities. The first snapshot is silent; on later starts, or after an explicit programmatic registration/removal while running, additions, removals, and description/schema changes produce a normal `runtime` / `capabilities_changed` wake. Detection never invokes or tests a capability. Connectors may independently emit source-specific events when their visible world changes; Resident decides whether either kind of change warrants investigation or communication.

## Experimental HomeOps connector

Set `RESIDENT_HOMEOPS_URL` (or pass `--homeops-url`) to opt into read-only HomeOps observation. With no URL configured, Resident makes no HomeOps requests. The connector silently establishes a baseline from `GET /api/measurements/latest`, then polls every 30 seconds and emits one wake containing all values changed during that poll. New measurement points count as changes; timestamp-only updates do not. Polling failures are retried without waking Resident or discarding the last successful baseline. These recoverable failures are shown with verbose diagnostics and suppressed in the default terminal mode.

Resident can use `homeops_get_current_measurements` and `homeops_get_measurement_history` to investigate. History accepts a measurement `point_id`, optional ISO-8601 `from_time` and `to_time`, and an optional `limit` from 1 to 5,000. Configure polling and HTTP timeout with `--homeops-poll-seconds` / `RESIDENT_HOMEOPS_POLL_SECONDS` and `--homeops-request-timeout-seconds` / `RESIDENT_HOMEOPS_REQUEST_TIMEOUT_SECONDS`.

### HomeOps-backed displays

Set `RESIDENT_DISPLAYS` to a JSON array of display IDs to expose action-only text display capabilities. Displays reuse `RESIDENT_HOMEOPS_URL` and its request timeout; no separate display backend URL is configured. For example:

```powershell
$env:RESIDENT_HOMEOPS_URL = "http://homeops.local"
$env:RESIDENT_DISPLAYS = '[{"id":"display1"}]'
```

Each configured display exposes a flat `<id>_show_text` capability, such as `display1_show_text`, which accepts a required `text` string and enqueues it with HomeOps. Display IDs must be unique, 1-64 characters, and contain only letters, digits, underscores, or hyphens. Display output does not poll, emit wakes, retry delivery, or encode any policy about when or what Resident should display. HomeOps queue delivery is in-memory and at-most-once; HTTP and API failures are returned as tool failures.

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
