"""Minimal independent Responses latency probe, using HTTPX directly.

  python responses_latency_probe.py --runs 3

Uses Resident's OPENAI_API_KEY and optional OPENAI_BASE_URL environment variables.
POST /responses: model=gpt-5.6-luna, reasoning.effort=none, input="test", stream=true.
Minimal mode has no instructions, tools, history, previous_response_id, service-tier override,
background work, setup requests, or retries. Other API settings use their defaults.
One synchronous HTTP/1.1 client uses normal keep-alive pooling for sequential runs.

Add --watcher-context --source-session <Watcher-session-id> to include real exported
agent instructions, bootstrap context and the latest 40 items (change --history-items
up to 100). Function schemas are supplied with tool_choice=none; no tools execute.
Setup uses only read-only remote GETs, outside timing. No Resident state is opened.
Actual returned input/cache/output/reasoning usage is printed, not estimated.

Add --conversation for exactly five new turns on one new persistent Conversation.
Watcher context is seeded once before timing, in API-required batches of up to 20.
Instructions/tools/output schema are unchanged per-response settings. No history
is resent during measured turns. Turns 2 and 5 check recall without resending the
value from turn 1. The Conversation is retained for inspection.

Offsets start immediately before each POST. Total includes stream closure after
response.completed; JSON printing and summary are excluded. First meaningful
output means an assistant message/reasoning item or nonempty text/reasoning delta,
not response.created/in_progress. First text is recorded separately. These are
client observation times; no response text is printed or accumulated.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time
from urllib.parse import quote

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


def dump(result):
    print(json.dumps({k: round(v, 6) if isinstance(v, float) else v
                      for k, v in result.items()}, separators=(",", ":")), flush=True)


def token_usage(usage):
    # Only numeric usage fields; never emit the completed response payload.
    result = {}
    for name in ("input_tokens", "output_tokens", "total_tokens"):
        if isinstance(usage.get(name), int):
            result[name] = usage[name]
    for name, field in (("input_tokens_details", "cached_tokens"),
                        ("output_tokens_details", "reasoning_tokens")):
        value = (usage.get(name) or {}).get(field)
        if isinstance(value, int):
            result[name] = {field: value}
    return result


def observe(response, result, started, deadline, recall_value=None):
    previous = None
    result.update(first_meaningful_output_s=None, first_meaningful_output_type=None,
                  first_text_s=None, first_text_type=None, response_id=None,
                  largest_sse_gap_s=0.0)
    for event in sse(response, deadline):
        now, timestamp = time.monotonic(), utc()
        offset = now - started
        kind = event.get("type", "unknown")
        if previous is not None:
            result["largest_sse_gap_s"] = max(result["largest_sse_gap_s"], now - previous)
        previous = now
        result.setdefault("first_sse_s", offset)
        result.setdefault("first_sse_type", kind)
        metadata = event.get("response") or {}
        response_id = metadata.get("id") or event.get("response_id")
        if response_id:
            if result["response_id"] not in (None, response_id):
                raise ProbeError("Stream response identity changed")
            result["response_id"] = response_id
        if kind in {"error", "response.failed", "response.incomplete"}:
            dump_error = {"stream_event_type": kind, "response_id": result["response_id"],
                          "error": metadata.get("error") or {
                              k: event[k] for k in ("code", "message", "param") if k in event},
                          "incomplete_details": metadata.get("incomplete_details")}
            print(json.dumps(redact_diagnostic(dump_error)), file=sys.stderr, flush=True)
            raise ProbeError("Responses stream failed or ended incomplete")
        item = event.get("item") or {}
        text_event = (kind in {"response.output_text.delta", "response.output_text.done"}
                      and bool(event.get("delta") or event.get("text")))
        meaningful = (
            kind in {"response.output_item.added", "response.output_item.done"}
            and (item.get("type") == "reasoning" or
                 (item.get("type") == "message" and item.get("role") == "assistant"))
        ) or text_event or (
            kind in {"response.reasoning_summary_text.delta", "response.reasoning_text.delta"}
            and bool(event.get("delta"))
        )
        if meaningful and result["first_meaningful_output_s"] is None:
            result.update(first_meaningful_output_s=offset, first_meaningful_output_type=kind,
                          first_meaningful_output_utc=timestamp)
        if text_event and result["first_text_s"] is None:
            result.update(first_text_s=offset, first_text_type=kind, first_text_utc=timestamp)
        if kind == "response.completed":
            if metadata.get("status") != "completed":
                raise ProbeError("Invalid response.completed status")
            result.update(response_completed_s=offset, response_completed_utc=timestamp)
            if recall_value is not None:
                # Inspect only assistant output already present in the final SSE event.
                # No answer accumulation, extra requests, or private response text logging.
                pattern = r"(?<!\d)" + re.escape(recall_value) + r"(?!\d)"
                result["recall_verified"] = any(
                    re.search(pattern, part.get("text", "")) is not None
                    for output in metadata.get("output", [])
                    if output.get("type") == "message" and output.get("role") == "assistant"
                    for part in output.get("content", []) if part.get("type") == "output_text")
            if isinstance(metadata.get("usage"), dict):
                result["usage"] = token_usage(metadata["usage"])
            return
    raise ProbeError("Stream ended without response.completed")


def item_text(item, counts):
    parts = item.get("content", [])
    if isinstance(parts, str):
        return parts
    texts = []
    for part in parts or []:
        if part.get("type") in {"input_text", "output_text"} and isinstance(part.get("text"), str):
            texts.append(part["text"])
        else:
            counts["nontext_parts_omitted"] += 1
    return "\n".join(texts)


def watcher_request(client, base, args, body):
    # Read-only export, once before timing. No Resident imports or SQLite access.
    import yaml  # Already a Resident dependency; minimal mode does not need it.

    definition_path = Path(args.watcher_definition)
    definition = yaml.safe_load(definition_path.read_text(encoding="utf-8"))
    if definition.get("id") != "watcher":
        raise ProbeError("Watcher definition must have id=watcher")
    if not args.source_session:
        raise ProbeError("Watcher mode requires --source-session or RESIDENT_PROBE_SOURCE_SESSION")
    source_path = base + "/agents/sessions/" + quote(args.source_session, safe="")

    def get(path, operation, params=None):
        response = client.get(path, params=params, headers={"OpenAI-Beta": "agents=v1"})
        check(response, operation)
        return response.json()

    snapshot = get(source_path, "Watcher source-session fetch")
    agent = snapshot["agent"]
    if snapshot.get("environment", {}).get("type") != "none":
        raise ProbeError("Watcher source must be a conversation-only session")
    if not isinstance(agent.get("instructions"), str):
        raise ProbeError("Source does not expose agent instructions")
    if any(tool.get("type") != "function" for tool in agent.get("tools", [])):
        raise ProbeError("Source has non-function tools; this mode cannot include hosted tools")
    oldest = get(source_path + "/items", "Watcher bootstrap fetch", {"order": "asc", "limit": 100})
    latest = get(source_path + "/items", "Watcher recent-history fetch",
                 {"order": "desc", "limit": args.history_items})
    recent = list(reversed(latest.get("data") or []))
    counts = {"message": 0, "function_call": 0, "function_call_output": 0,
              "nontext_parts_omitted": 0, "items_omitted": {}}
    bootstrap = None
    bootstrap_id = None
    for item in oldest.get("data") or []:
        if item.get("type") != "message" or item.get("role") != "user":
            continue
        try:
            document = json.loads(item_text(item, {"nontext_parts_omitted": 0}))
        except ValueError:
            continue
        if isinstance(document, dict) and isinstance(document.get("new_session_bootstrap"), dict):
            bootstrap = document["new_session_bootstrap"]
            bootstrap_id = item.get("id")
            break
    inputs = []
    recent_ids = {item.get("id") for item in recent}
    if bootstrap is not None:
        static_source = "remote bootstrap (already in history)"
        if bootstrap_id not in recent_ids:
            # Preserve real static material without replaying the original wake.
            inputs.append({"role": "user", "content": json.dumps({"new_session_bootstrap": bootstrap})})
            static_source = "remote bootstrap (prepended without original wake)"
    else:
        prompt_root = Path(args.prompt_root)
        def prompt(name):
            if name + "_prompt" in definition:
                return (prompt_root / definition[name + "_prompt"]).read_text(encoding="utf-8")
            return definition.get(name, "")
        inputs.append({"role": "developer", "content": json.dumps({"resident": {
            "address_name": definition.get("name", "Watcher"),
            "personality": prompt("personality"), "role": prompt("role")}})})
        static_source = "local definition/personality/role fallback; no remote bootstrap found"

    # Use native Responses history items only for complete call/result pairs.
    calls = {item.get("call_id"): index for index, item in enumerate(recent)
             if item.get("type") == "function_call" and item.get("call_id")
             and item.get("status") in (None, "completed")}
    outputs = {item.get("call_id"): index for index, item in enumerate(recent)
               if item.get("type") == "function_call_output" and item.get("call_id")
               and item.get("status") in (None, "completed")}
    pairs = {call_id for call_id in calls.keys() & outputs.keys() if calls[call_id] < outputs[call_id]}
    seen_calls = set()
    for item in recent:
        kind = item.get("type", "unknown")
        converted = None
        if item.get("status") not in (None, "completed"):
            kind = "unfinished_" + kind
        elif kind == "message" and item.get("role") in {"user", "assistant", "system", "developer"}:
            text = item_text(item, counts)
            if text:
                converted = {"role": item["role"], "content": text}
        elif kind == "function_call" and item.get("call_id") in pairs:
            arguments = item.get("arguments", "{}")
            converted = {"type": "function_call", "call_id": item["call_id"], "name": item["name"],
                         "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments)}
            seen_calls.add(item["call_id"])
        elif kind == "function_call_output" and item.get("call_id") in seen_calls:
            output = item.get("output")
            if isinstance(output, list):
                output = item_text({"content": output}, counts)
            elif not isinstance(output, str):
                output = json.dumps(output if output is not None else {"error": item.get("error")})
            converted = {"type": "function_call_output", "call_id": item["call_id"], "output": output}
        if converted is not None:
            inputs.append(converted)
            counts[kind] += 1
        else:
            counts["items_omitted"][kind] = counts["items_omitted"].get(kind, 0) + 1
    inputs.append({"role": "user", "content": "test"})
    body.update(instructions=agent["instructions"], input=inputs, tool_choice="none",
                tools=[{**{k: tool[k] for k in ("type", "name", "description", "parameters") if k in tool},
                        "strict": False} for tool in agent.get("tools", [])])
    text = agent.get("text") or {}
    if text.get("format"):
        format_config = dict(text["format"])
        if format_config.get("type") == "json_schema":
            if format_config.get("name") is None:
                format_config["name"] = "watcher_disposition"
        body["text"] = {"format": format_config}
        if text.get("verbosity") is not None:
            body["text"]["verbosity"] = text["verbosity"]
    dump({"context_setup": True, "mode": "watcher-context", "source_session_id": args.source_session,
          "source_status_at_fetch": snapshot.get("status"), "source_model": agent.get("model"),
          "static_context_source": static_source, "instructions_characters": len(agent["instructions"]),
          "function_schemas": len(body["tools"]), "output_format_included": "text" in body,
          "recent_items_requested": args.history_items, "recent_items_fetched": len(recent),
          "older_history_omitted": bool(latest.get("has_more")), "history_conversion": counts,
          "included": ["source agent instructions", "real bootstrap identity/personality/role/guidance when exported",
                       "bootstrap capability/memory awareness/handover already in remote text, when present",
                       "recent text messages and authoritative-state updates", "paired historical function calls/results",
                       "function descriptions/parameter schemas", "source output format",
                       "one-time Conversation seed (trailing test removed)" if getattr(args, "conversation", False) else "final user test"],
          "omitted": ["Agents internal system prompt, hidden summaries/compaction and exact context window",
                      "older conversation beyond selected window (except bootstrap)",
                      "reasoning items, images/audio/files, unsupported or unfinished items and orphan tool records",
                      "Agent item IDs/status/phase, tool defer_loading and Agent multi_agent settings",
                      "source service tier/reasoning effort (Responses baseline settings used)",
                      "live connector state, fresh memory/curator/guidance reads from local SQLite", "all side effects"],
          "differences": ["tool_choice=none; function strict=false to preserve optional parameters without schema normalization",
                          "JSON schema format gets name=watcher_disposition only if the source name is absent or null",
                          ("snapshot seeded once; Conversation accumulates history; configurations stay stable per turn"
                           if getattr(args, "conversation", False) else
                           "snapshot frozen once for all runs; no previous_response_id or generated-answer accumulation"),
                          "read-only context fetches excluded from timings; first measured connection is already warm"],
          "request_bytes": len(json.dumps(body).encode()), "token_count": "use returned usage; not estimated"})
    return body


def comparison(results):
    totals = [r["total_s"] for r in results]
    first = [r["first_meaningful_output_s"] for r in results if r["first_meaningful_output_s"] is not None]
    usages = []
    for result in results:
        usage = result.get("usage", {})
        usages.append({"run": result["run"], "input_tokens": usage.get("input_tokens"),
                       "cached_input_tokens": usage.get("input_tokens_details", {}).get("cached_tokens"),
                       "output_tokens": usage.get("output_tokens"),
                       "reasoning_tokens": usage.get("output_tokens_details", {}).get("reasoning_tokens")})
    dump({"comparison": True, "historical_baselines_user_supplied": {
              "minimal_responses_input_tokens": 7, "minimal_responses_median_total_s": 2.107,
              "minimal_responses_median_first_meaningful_output_s_approx": 1.404,
              "agents_watcher_config_median_total_s": 7.993},
          "watcher_responses": {"usage_per_run": usages, "median_total_s": statistics.median(totals),
                                "median_first_meaningful_output_s": statistics.median(first) if first else None,
                                "median_response_completed_s": statistics.median(r["response_completed_s"] for r in results)},
          "median_total_delta_vs_minimal_responses_s": statistics.median(totals) - 2.107,
          "median_total_delta_vs_agents_s": statistics.median(totals) - 7.993,
          "interpretation": "Evaluate actual input/cache tokens alongside latency; this is an exported-context approximation."})


CONVERSATION_MESSAGES = (
    "Please remember the value 4711 for this conversation. Reply briefly to confirm.",
    "What value did I ask you to remember? Include the value in your reply to the Owner.",
    "Say hello to the Owner in one short sentence.",
    "What is two plus two? Reply briefly to the Owner.",
    "What value did I ask you to remember earlier in this conversation? Include it in your Owner reply.",
)


def create_conversation(client, base, body):
    # Upload exported context exactly once; remove the independent probe's final test.
    items = body.pop("input")[:-1]
    if re.search(r"(?<!\d)4711(?!\d)", json.dumps({**body, "items": items})):
        raise ProbeError("Source context already contains recall value; cannot isolate this recall experiment")
    items = [{"type": "message", **item} if "role" in item else item for item in items]
    response = client.post(base + "/conversations", json={"items": items[:20]})
    check(response, "Conversation creation")
    conversation_id = response.json()["id"]
    dump({"conversation_setup": True, "conversation_id": conversation_id,
          "seed_items": len(items), "seed_excludes_recall_value": True,
          "excluded_from_turn_timings": True})
    for index in range(20, len(items), 20):
        response = client.post(base + "/conversations/" + quote(conversation_id, safe="") + "/items",
                               json={"items": items[index:index + 20]})
        check(response, "Conversation context seeding")
    body["conversation"] = conversation_id
    return conversation_id


def usage_columns(result):
    usage = result.get("usage", {})
    inputs = usage.get("input_tokens")
    cached = usage.get("input_tokens_details", {}).get("cached_tokens")
    return {"input_tokens": inputs, "cached_input_tokens": cached,
            "uncached_input_tokens": inputs - cached if inputs is not None and cached is not None else None,
            "output_tokens": usage.get("output_tokens"),
            "reasoning_tokens": usage.get("output_tokens_details", {}).get("reasoning_tokens")}


def conversation_summary(results):
    print("turn | input | cached | uncached | first output | completed | total")
    def cell(value):
        return "n/a" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)
    for result in results:
        print(" | ".join(cell(v) for v in (
            result["run"], result["input_tokens"], result["cached_input_tokens"], result["uncached_input_tokens"],
            result["first_meaningful_output_s"], result["response_completed_s"], result["total_s"])))
    first = [r["first_meaningful_output_s"] for r in results if r["first_meaningful_output_s"] is not None]
    completed = [r["response_completed_s"] for r in results]
    totals = [r["total_s"] for r in results]
    median_later = statistics.median(totals[1:])
    difference = median_later - totals[0]
    growth = {}
    for field in ("input_tokens", "cached_input_tokens", "uncached_input_tokens"):
        values = [r[field] for r in results]
        known = all(value is not None for value in values)
        growth[field] = {"last_minus_first": values[-1] - values[0] if known else None,
                         "grew": values[-1] > values[0] if known else None,
                         "nondecreasing": all(b >= a for a, b in zip(values, values[1:])) if known else None}
    token_totals = {}
    for field in ("input_tokens", "cached_input_tokens", "uncached_input_tokens", "output_tokens", "reasoning_tokens"):
        values = [r[field] for r in results]
        token_totals[field] = sum(values) if all(value is not None for value in values) else None
    dump({"conversation_summary": True, "conversation_id": results[0]["conversation_id"],
          "median_first_meaningful_output_s": statistics.median(first) if first else None,
          "median_response_completed_s": statistics.median(completed), "median_total_s": statistics.median(totals),
          "recall_checks": [{"turn": r["run"], "verified": r["recall_verified"],
                             "request_omits_value": r["recall_request_omits_value"]}
                            for r in results if "recall_verified" in r],
          "token_growth": growth, "measured_token_totals_for_cost_comparison": token_totals,
          "latency_change": {"first_total_s": totals[0], "median_turns_2_to_5_total_s": median_later,
                             "delta_s": difference,
                             "percent": 100 * difference / totals[0] if totals[0] else None,
                             "material_by_descriptive_threshold": abs(difference) >= max(0.5, 0.2 * totals[0]),
                             "threshold": "at least 0.5 seconds AND 20%; descriptive, not statistical significance"},
          "cost_note": "Counts are reported usage, including persistent history; no currency estimate or assumed cache discount.",
          "comparison_note": "Different short inputs and accumulated outputs; this five-turn sequence does not isolate causal latency effects."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=None, help="default 3, or 5 in conversation mode")
    parser.add_argument("--timeout", type=float, default=60,
                        help="network read timeout / SSE deadline seconds")
    parser.add_argument("--watcher-context", action="store_true",
                        help="read Watcher agent/bootstrap/history and include them in each Responses request")
    parser.add_argument("--conversation", action="store_true",
                        help="five turns on one seeded Conversation; implies --watcher-context")
    parser.add_argument("--source-session", default=os.getenv("RESIDENT_PROBE_SOURCE_SESSION"))
    parser.add_argument("--history-items", type=int, default=40, help="recent remote items to include (1-100)")
    parser.add_argument("--watcher-definition", default=str(Path(__file__).parent / "residents" / "watcher.yaml"))
    parser.add_argument("--prompt-root", default=str(Path(__file__).parent / "prompts"))
    args = parser.parse_args()
    if args.runs is None:
        args.runs = 5 if args.conversation else 3
    if args.conversation:
        args.watcher_context = True
        if args.runs != 5:
            parser.error("conversation mode measures exactly five turns; use --runs 5")
    if not 1 <= args.history_items <= 100:
        parser.error("history-items must be between 1 and 100")
    if args.runs < 1 or args.timeout <= 0:
        parser.error("runs and timeout must be positive")
    if not os.getenv("OPENAI_API_KEY"):
        raise ProbeError("OPENAI_API_KEY is unavailable; run in Resident's configured shell")
    endpoint = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/") + "/responses"
    headers = {"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
               "Content-Type": "application/json", "Accept": "text/event-stream"}
    body = {"model": "gpt-5.6-luna", "reasoning": {"effort": "none"}, "input": "test", "stream": True}
    results = []
    with httpx.Client(headers=headers, http1=True, http2=False, timeout=args.timeout) as client:
        if args.watcher_context:
            body = watcher_request(client, endpoint.removesuffix("/responses"), args, body)
        conversation_id = None
        if args.conversation:
            conversation_id = create_conversation(client, endpoint.removesuffix("/responses"), body)
        payload = json.dumps(body).encode()
        for number in range(1, args.runs + 1):
            recall_value = None
            if args.conversation:
                body["input"] = CONVERSATION_MESSAGES[number - 1]
                payload = json.dumps(body).encode()
                recall_value = "4711" if number in (2, 5) else None
            result = {"run": number, "probe_start_utc": utc()}
            if args.conversation:
                result["conversation_id"] = conversation_id
                if recall_value is not None:
                    result["recall_request_omits_value"] = recall_value not in body["input"]
            started = time.monotonic()
            with client.stream("POST", endpoint, content=payload) as response:
                result.update(http_headers_s=time.monotonic() - started,
                              http_version=response.http_version)
                check(response, "Responses submission")
                if response.headers.get("x-request-id"):
                    result["request_id"] = response.headers["x-request-id"]
                observe(response, result, started, started + args.timeout, recall_value=recall_value)
            result["total_s"] = time.monotonic() - started
            if args.conversation:
                result.update(usage_columns(result))
            dump(result)
            results.append(result)
    totals = [r["total_s"] for r in results]
    first_outputs = [r["first_meaningful_output_s"] for r in results
                     if r["first_meaningful_output_s"] is not None]
    dump({"summary": True, "total_times_s": [round(t, 6) for t in totals],
          "min_s": min(totals), "median_s": statistics.median(totals), "max_s": max(totals),
          "median_first_meaningful_output_s": statistics.median(first_outputs) if first_outputs else None,
          "runs_with_meaningful_output": len(first_outputs),
          "median_response_completed_s": statistics.median(r["response_completed_s"] for r in results)})

    if args.conversation:
        conversation_summary(results)
    elif args.watcher_context:
        comparison(results)


if __name__ == "__main__":
    try:
        main()
    except (ProbeError, httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        print("probe failed: " + (str(exc) if isinstance(exc, ProbeError) else type(exc).__name__),
              file=sys.stderr)
        sys.exit(1)
