"""Standalone Agents latency probe; no Resident imports or local state access.

Run from the shell containing Resident's OPENAI_API_KEY / OPENAI_BASE_URL:
  python agents_latency_probe.py --source-session <watcher-session-id> --runs 3

The source session is read-only. A private disposable session copies its agent
snapshot (including function declarations and output schema) without changing it.
One create-time test turn is necessary for environment=none and is not measured.
Measured totals start immediately before SSE GET and end after closing the stream;
configuration GET and disposable-session creation are outside those totals.
The disposable session is retained for managed trace inspection. No tools execute.
First-output means first assistant message item, function-call item, or text delta.
SSE times are client observation times after POST acknowledgement, like Resident;
events emitted during POST may already be buffered by the transport.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import statistics
import sys
import time
from urllib.parse import quote
import uuid

import httpx


class ProbeError(Exception):
    pass


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def check(response, operation):
    if 200 <= response.status_code < 300:
        return
    error = ProbeError(f"HTTP {response.status_code}")
    # Only response diagnostics, never request headers or the full URL/query.
    diagnostic = {
        "operation": operation, "method": response.request.method,
        "path": response.request.url.path, "status": response.status_code,
        "response_headers": {
            name: value for name, value in response.headers.items()
            if name in {"x-request-id", "request-id", "openai-request-id",
                        "openai-processing-ms", "openai-version", "retry-after", "content-type"}
            or name.startswith("x-ratelimit-")
        },
    }
    try:
        # Error responses may be streamed too. Read only a bounded diagnostic
        # excerpt; successful SSE responses are never consumed by this check.
        body = bytearray()
        limit = 65536
        for chunk in response.iter_bytes():
            room = limit - len(body)
            body.extend(chunk[:room])
            if len(chunk) > room:
                diagnostic["body_truncated"] = True
                break
    except Exception as exc:
        diagnostic["body_read_error"] = type(exc).__name__
    decoded = body.decode("utf-8", errors="replace")
    try:
        diagnostic["error_body"] = json.loads(decoded)
    except ValueError:
        diagnostic["error_body"] = decoded
    try:
        print(json.dumps(redact_diagnostic(diagnostic), separators=(",", ":")),
              file=sys.stderr, flush=True)
    finally:
        raise error  # Keep the original HTTP failure after reporting diagnostics.


def redact_diagnostic(value):
    # Defend against an error response echoing credentials supplied by the caller.
    sensitive = re.compile(r"authorization|api.?key|password|secret|credential|cookie|(?:^|_)token$", re.I)
    secrets = [v for k, v in os.environ.items()
               if sensitive.search(k) and len(v) >= 4]

    def scrub(item):
        if isinstance(item, dict):
            return {k: "[REDACTED]" if sensitive.search(k) else scrub(v)
                    for k, v in item.items()}
        if isinstance(item, list):
            return [scrub(v) for v in item]
        if isinstance(item, str):
            for secret in sorted(secrets, key=len, reverse=True):
                item = item.replace(secret, "[REDACTED]")
            item = re.sub(r"(?i)\bBearer\s+[^\s\"'<>]+", "Bearer [REDACTED]", item)
            return re.sub(r"\bsk-[A-Za-z0-9_-]+", "[REDACTED]", item)
        return item

    return scrub(value)


def sse(response, deadline):
    # Same data-frame rules as Resident; iter_lines consumes real streaming I/O.
    data = []
    for line in response.iter_lines():
        if time.monotonic() >= deadline:
            raise ProbeError("SSE deadline exceeded")
        if line == "":
            if not data:
                continue
            payload = "\n".join(data)
            data.clear()
            if payload == "[DONE]":
                return
            event = json.loads(payload)
            if not isinstance(event, dict):
                raise ProbeError("SSE event is not an object")
            yield event
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    if data and "\n".join(data) != "[DONE]":
        event = json.loads("\n".join(data))
        if not isinstance(event, dict):
            raise ProbeError("SSE event is not an object")
        yield event


def input_message():
    key = uuid.uuid4().hex
    correlation = hashlib.sha256(key.encode()).hexdigest()
    # Minimal managed Owner wake, retaining Resident's exact echo correlation.
    context = json.dumps({
        "wake_event": {"id": key, "source": "owner", "reason": "owner_message",
                       "occurred_at": utc(), "payload": {"message_id": key, "content": "test"}},
        "resident_wake_correlation": correlation,
    }, separators=(",", ":"))
    return key, correlation, [{"role": "user", "content": [
        {"type": "input_text", "text": context}]}]


def observe(response, result, started, deadline, correlation, session_id=None):
    turn_id = None
    previous = None
    created = {}
    outputs = {}
    result["largest_sse_gap_s"] = 0.0
    for event in sse(response, deadline):
        now, timestamp = time.monotonic(), utc()
        offset = now - started
        kind = event.get("type", "unknown")
        if previous is not None:
            result["largest_sse_gap_s"] = max(result["largest_sse_gap_s"], now - previous)
        previous = now
        result.setdefault("first_sse_s", offset)
        result.setdefault("first_sse_type", kind)
        if kind == "agent.session.created" and session_id is None:
            session_id = event["session"]["id"]
        if event.get("session_id", session_id) != session_id:
            raise ProbeError("SSE session mismatch")
        item = event.get("item") or {}
        turn = event.get("turn") or {}
        tid = event.get("turn_id") or item.get("turn_id") or turn.get("id")
        if kind == "agent.session.turn.created":
            created[tid] = (offset, timestamp)
        if kind in {"agent.session.turn.item.added", "agent.session.turn.item.done"}:
            if item.get("type") == "message" and item.get("role") == "user":
                for part in item.get("content", []):
                    if part.get("type") != "input_text":
                        continue
                    try:
                        document = json.loads(part.get("text", ""))
                    except (ValueError, TypeError):
                        continue
                    if isinstance(document, dict) and document.get("resident_wake_correlation") == correlation:
                        if not tid or (turn_id is not None and turn_id != tid):
                            raise ProbeError("Ambiguous input-to-turn correlation")
                        turn_id = tid
            if ((item.get("type") == "message" and item.get("role") == "assistant")
                    or item.get("type") == "function_call"):
                outputs.setdefault(tid, (offset, timestamp))
        if kind == "agent.session.turn.output_text.delta":
            outputs.setdefault(tid, (offset, timestamp))
        if kind in {"error", "agent.session.failed", "agent.session.requires_action"}:
            raise ProbeError("API error or required action; no tools or recovery in this probe")
        if tid == turn_id and turn_id is not None:
            if kind in {"agent.session.turn.failed", "agent.session.turn.cancelled"}:
                raise ProbeError("Matching turn failed or cancelled")
            if kind == "agent.session.turn.completed":
                if turn.get("id") != turn_id or turn.get("status") != "completed":
                    raise ProbeError("Invalid matching completion event")
                result.update(session_id=session_id, turn_id=turn_id,
                              turn_completed_s=offset, turn_completed_utc=timestamp)
                for label, observations in (("turn_created", created), ("first_output", outputs)):
                    observed = observations.get(turn_id)
                    result[label + "_s"] = observed[0] if observed else None
                    result[label + "_utc"] = observed[1] if observed else None
                return
    raise ProbeError("SSE ended without a correlated turn.completed")


def dump(result):
    print(json.dumps({k: round(v, 6) if isinstance(v, float) else v
                      for k, v in result.items()}, separators=(",", ":")), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--source-session", default=os.getenv("RESIDENT_PROBE_SOURCE_SESSION"),
                        help="Watcher session ID to read configuration from; never receives input")
    parser.add_argument("--timeout", type=float, default=60, help="network read timeout / SSE deadline seconds")
    args = parser.parse_args()
    if args.runs < 1 or args.timeout <= 0:
        parser.error("runs and timeout must be positive")
    if not os.getenv("OPENAI_API_KEY"):
        raise ProbeError("OPENAI_API_KEY is unavailable; run in Resident's configured shell")
    if not args.source_session:
        raise ProbeError("Supply --source-session with Watcher's remote session ID")
    base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    headers = {"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
               "OpenAI-Beta": "agents=v1", "Content-Type": "application/json"}
    # Default HTTPX pooling, HTTP/1.1 only, no transport retries, one client for all runs.
    with httpx.Client(headers=headers, http1=True, http2=False, timeout=args.timeout) as client:
        source = client.get(base + "/agents/sessions/" + quote(args.source_session, safe=""))
        check(source, "source-session fetch")
        snapshot = source.json()
        if snapshot.get("environment", {}).get("type") != "none":
            raise ProbeError("Source must be a conversation-only Resident session")
        agent = {k: v for k, v in snapshot["agent"].items() if k not in {"id", "name"}}
        multi_agent = agent.get("multi_agent")
        if isinstance(multi_agent, dict) and multi_agent.get("max_concurrent_subagents") is None:
            # The read snapshot may contain null; the write schema requires an integer.
            agent["multi_agent"] = {k: v for k, v in multi_agent.items()
                                    if k != "max_concurrent_subagents"}
        # Preserve function declarations; refuse hosted tools / connectors with autonomous effects.
        if any(t.get("type") != "function" for t in agent.get("tools", [])):
            raise ProbeError("Source has hosted tools; cannot make an isolated no-tool probe")
        if agent.get("multi_agent", {}).get("enabled"):
            raise ProbeError("Source enables subagents; cannot make a minimal isolated probe")
        setup = {"phase": "unmeasured_setup"}
        key, correlation, message = input_message()
        started = time.monotonic()
        with client.stream("POST", base + "/agents/sessions", content=json.dumps({
                "agent": agent, "environment": {"type": "none"}, "input": message,
                "stream": True, "metadata": {"managed_by": "agents_latency_probe"}}).encode(),
                headers={"Accept": "text/event-stream"}) as response:
            check(response, "disposable-session creation")
            observe(response, setup, started, started + args.timeout, correlation)
        session_id = setup["session_id"]
        if session_id == args.source_session:
            raise ProbeError("Disposable session identity equals source")
        dump({"phase": "setup", "session_id": session_id, "turn_id": setup["turn_id"],
              "setup_s": time.monotonic() - started, "excluded_from_run_totals": True})
        endpoint = base + "/agents/sessions/" + quote(session_id, safe="") + "/events"
        results = []
        for number in range(1, args.runs + 1):
            key, correlation, message = input_message()
            payload = json.dumps({"events": [{"type": "agent.session.input.message",
                                               "input": message}]}).encode()
            result = {"run": number, "probe_start_utc": utc()}
            started = time.monotonic()
            with client.stream("GET", endpoint, headers={"Accept": "text/event-stream"}) as response:
                result["sse_headers_s"] = time.monotonic() - started
                result["http_version"] = response.http_version
                check(response, "SSE open")
                post_started = time.monotonic()
                result.update(post_start_s=post_started - started, post_start_utc=utc())
                # No checkpoint, sleep, local persistence, or stream consumer before POST.
                with client.stream("POST", endpoint, content=payload,
                                   headers={"Idempotency-Key": "resident-wake:" + key}) as acknowledgement:
                    check(acknowledgement, "input submission")
                    for chunk in acknowledgement.iter_bytes():
                        pass  # Consume/discard acknowledgement incrementally; never log its body.
                    post_ack = time.monotonic()
                    result.update(post_ack_s=post_ack - started, post_duration_s=post_ack - post_started,
                                  post_ack_utc=utc(), post_http_version=acknowledgement.http_version)
                observe(response, result, started, started + args.timeout, correlation, session_id)
            result["total_s"] = time.monotonic() - started
            dump(result)
            results.append(result)
        totals = [r["total_s"] for r in results]
        dump({"summary": True, "total_times_s": [round(t, 6) for t in totals],
              "min_s": min(totals), "median_s": statistics.median(totals), "max_s": max(totals),
              "median_turn_completed_s": statistics.median(r["turn_completed_s"] for r in results),
              "median_post_duration_s": statistics.median(r["post_duration_s"] for r in results)})


if __name__ == "__main__":
    try:
        main()
    except (ProbeError, httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        # HTTPX exceptions can contain URLs; JSON errors can contain payload details.
        print("probe failed: " + (str(exc) if isinstance(exc, ProbeError) else type(exc).__name__),
              file=sys.stderr)
        sys.exit(1)
