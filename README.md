# Resident

Resident is a persistent AI entity that inhabits an environment, learns about it through available connectors, interacts asynchronously with its owner, and may use physical embodiments such as mobile robots when available.

This repository is intentionally starting with principles rather than a detailed implementation. Resident is an experiment in persistent identity, emergent understanding, curiosity, and tool use. Mechanisms should be added when experience shows that they are needed rather than pre-programming the behavior we hope will emerge.

See [VISION.md](VISION.md) for the current product/behavior vision and [ARCHITECTURE.md](ARCHITECTURE.md) for the initial architectural principles.

## Minimal runtime

The first experimental vertical slice is a Python 3.12+ asynchronous process with durable SQLite state. It preserves Resident and owner identities, memory, pending intentions, communication, wake runs, an append-only journal, and self-requested scheduled wakeups across restarts. Owner messages and due schedules become wake events; inference stops after each bounded model/tool exchange while terminal input and scheduling remain active.

The initial live adapter uses OpenAI's Responses API and defaults to `gpt-5.6-luna`. Set an API key and choose a persistent data directory:

```powershell
$env:OPENAI_API_KEY = "..."
python -m resident --data-dir .resident
```

Enter an owner message at the prompt. All intentional Resident-to-Owner communication, including direct replies, goes through `send_owner_message` and is rendered as `[Resident -> Owner] ...`. Model-returned text is a wake result for journaling and diagnostics, not a second communication transport; it is hidden by default and shown only with verbose diagnostics. A rejected or failed send never falls back to model-returned text. By default, the terminal otherwise shows only a small startup/shutdown status and actionable runtime failures, keeping routine spontaneous wakes nearly invisible. Pass `--verbose` or set `RESIDENT_VERBOSE=true` to show detailed wake, context, model, tool, memory, connector, and metric diagnostics. Verbosity affects terminal presentation only; the structured journal remains complete. Enter `/quit` to stop. Reusing the data directory reloads the same stable identities and state. Display names and personality can be configured on initial provisioning with `--resident-name`, `--owner-name`, and `--personality`; persisted identity is authoritative on later runs. `RESIDENT_MODEL`, `OPENAI_BASE_URL`, and the equivalent name/data environment variables may also be used.

Resident can explicitly manage memory and pending intentions, send owner messages, schedule a future wake, and invoke a read-only local time capability. Wake context contains the complete trigger but only a bounded selection of memories and communication. The runtime limits model tool use to eight rounds. Spontaneous messages (those outside an owner-initiated wake) default to three delivered messages per hour; excess messages are persisted as rejected and are never queued. Immediate replies during an owner wake do not consume that budget. Configure this policy with `--spontaneous-message-limit` and `--spontaneous-message-window-seconds` (or their `RESIDENT_...` environment-variable equivalents).

## Experimental HomeOps connector

Set `RESIDENT_HOMEOPS_URL` (or pass `--homeops-url`) to opt into read-only HomeOps observation. With no URL configured, Resident makes no HomeOps requests. The connector silently establishes a baseline from `GET /api/measurements/latest`, then polls every 30 seconds and emits one wake containing all values changed during that poll. New measurement points count as changes; timestamp-only updates do not. Polling failures are retried without waking Resident or discarding the last successful baseline. These recoverable failures are shown with verbose diagnostics and suppressed in the default terminal mode.

Resident can use `homeops_get_current_measurements` and `homeops_get_measurement_history` to investigate. History accepts a measurement `point_id`, optional ISO-8601 `from_time` and `to_time`, and an optional `limit` from 1 to 5,000. Configure polling and HTTP timeout with `--homeops-poll-seconds` / `RESIDENT_HOMEOPS_POLL_SECONDS` and `--homeops-request-timeout-seconds` / `RESIDENT_HOMEOPS_REQUEST_TIMEOUT_SECONDS`.

## Experimental RTSP camera connector

Set `RESIDENT_CAMERAS` to a JSON array to opt into on-demand, read-only camera access. Each camera requires a stable `id`, display `name`, and secret `url`; `description` is optional safe metadata. For example:

```powershell
$env:RESIDENT_CAMERAS = '[{"id":"entry","name":"Entry camera","description":"Front entry","url":"rtsp://user:password@camera.local/stream"}]'
python -m resident --data-dir .resident
```

Resident can list configured cameras with `camera_list` without connecting to them, and can request one current frame with `camera_capture_frame`. FFmpeg must be installed and available as `ffmpeg` (or configured with `--ffmpeg-executable`). Each request connects only long enough to capture one JPEG frame; unavailable cameras and timeouts are normal tool outcomes. Frames are passed in memory to the model and are never written to Resident state or its journal. Camera URLs, credentials, and FFmpeg error output are not exposed to the model, normal diagnostics, or journal. OpenAI Responses requests use `store=false`, including image continuations.

Capture defaults to RTSP over TCP, an 8-second timeout, 1280x720 maximum output dimensions, and a 2 MB encoded-frame limit. These can be adjusted with `--camera-rtsp-transport`, `--camera-capture-timeout-seconds`, `--camera-max-width`, `--camera-max-height`, and `--camera-max-bytes`, or their corresponding `RESIDENT_...` environment variables.

Run the deterministic offline lifecycle suite without credentials:

```powershell
python -m unittest discover -s tests -v
```
