"""Deterministic tests using the real SSE parser and correlated turn reducer."""
import json
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from resident.observability import timeline_reporter
from resident.provider import (
    OpenAIAgentsProvider, _agents_http_trace, _agents_stream_timing,
)


class StreamResponse:
    def __init__(self, entries, clock, *, trailing=False):
        self.clock = clock
        self.lines = iter([
            line
            for offset, event in entries
            for line in ([(offset, b"data: " + json.dumps(event).encode() + b"\n")]
                         if trailing else
                         [(offset, b"data: " + json.dumps(event).encode() + b"\n"),
                          (offset, b"\n")])
        ])
        self.closed = False
        self.fp = SimpleNamespace(raw=SimpleNamespace(
            _sock=SimpleNamespace(settimeout=lambda _: None)))

    def __iter__(self):
        return self

    def __next__(self):
        offset, line = next(self.lines)
        self.clock[0] = 100.0 + offset
        return line

    def close(self):
        self.closed = True


@contextmanager
def trace_stream(entries, *, enabled=True, trailing=False):
    provider = OpenAIAgentsProvider("key", "model")
    clock = [100.0]
    response = StreamResponse(entries, clock, trailing=trailing)
    trace = {"events": []}
    trace_token = _agents_http_trace.set(trace)
    reporter_token = timeline_reporter.set((lambda _: None) if enabled else None)
    try:
        with patch("resident.provider.time.monotonic", side_effect=lambda: clock[0]), patch(
                "resident.provider.urllib.request.urlopen", return_value=response):
            yield provider, trace, response
    finally:
        timeline_reporter.reset(reporter_token)
        _agents_http_trace.reset(trace_token)
    assert response.closed
    assert _agents_stream_timing.get() is None


def item_event(kind, *, turn="turn", role="assistant", item_type="message"):
    return {"type": "agent.session.turn.item." + kind, "turn_id": turn,
            "output_index": 0, "item": {
                "id": "item-secret", "type": item_type, "role": role,
                "status": "completed", "turn_id": turn,
                "content": [{"type": "output_text", "text": "content-secret"}]}}


def completed(turn="turn", usage=None):
    return {"type": "agent.session.turn.completed", "turn_id": turn,
            "turn": {"id": turn, "status": "completed", "usage": usage}}


def consume(provider, stream, expected="turn", correlation=None, wake_key=None):
    return provider._consume_event_stream(
        "session", stream, expected_turn_id=expected,
        correlation=correlation, wake_key=wake_key)


def test_stream_offsets_gaps_completion_and_nested_usage():
    usage = {"input_tokens": 10, "output_tokens": 4,
             "input_tokens_details": {"cached_tokens": 6}}
    entries = [
        (1, {"type": "agent.session.turn.in_progress"}),
        (2, item_event("added", turn="other")),
        (3, completed("other")),
        (8, item_event("added")),
        (9, item_event("done")),
        (12, completed(usage=usage)),
    ]
    with trace_stream(entries) as (provider, trace, _):
        with provider._open_event_stream("session") as stream:
            turn = consume(provider, stream)
        summary, = trace["events"]
    assert (turn.input_tokens, turn.output_tokens, turn.cached_input_tokens) == (10, 4, 6)
    assert summary["event_count"] == 6
    assert summary["duration_seconds"] == 12
    assert summary["time_to_first_event_seconds"] == 1
    assert summary["first_event_type"] == "agent.session.turn.in_progress"
    assert summary["time_to_first_model_activity_seconds"] == 8
    assert summary["time_to_first_output_seconds"] == 8
    assert summary["first_output_type"] == "agent.session.turn.item.added"
    assert summary["time_to_completion_seconds"] == 12
    assert summary["completion_type"] == "agent.session.turn.completed"
    assert summary["largest_inter_event_gap_seconds"] == 5
    assert "secret" not in json.dumps(summary)


def test_wake_correlation_user_item_is_not_model_activity():
    provider = OpenAIAgentsProvider("key", "model")
    context, correlation = provider._correlated_context("wake", "wake-key")
    user = item_event("added", role="user")
    user.pop("output_index")
    user["item"]["content"] = [{"type": "input_text", "text": context}]
    entries = [(1, user), (5, item_event("added")),
               (6, item_event("done")), (7, completed())]
    with trace_stream(entries) as (provider, trace, _):
        with provider._open_event_stream("session") as stream:
            consume(provider, stream, expected=None, correlation=correlation, wake_key="wake-key")
        summary, = trace["events"]
    assert summary["time_to_first_model_activity_seconds"] == 5
    assert summary["time_to_completion_seconds"] == 7


def test_tool_pause_is_activity_without_output_or_completion():
    action = {"type": "agent.session.requires_action", "session": {
        "id": "session", "required_actions": [{"type": "function_call",
        "turn_id": "turn", "call_id": "call", "name": "tool", "arguments": "{}"}]}}
    with trace_stream([(2, action)]) as (provider, trace, _):
        with provider._open_event_stream("session") as stream:
            assert consume(provider, stream).tool_calls
        summary, = trace["events"]
    assert summary["time_to_first_model_activity_seconds"] == 2
    assert "time_to_first_output_seconds" not in summary
    assert "time_to_completion_seconds" not in summary
    assert summary["largest_inter_event_gap_seconds"] == 0


@pytest.mark.parametrize("entries,trailing", [([], False),
    ([(4, {"type": "unknown", "payload": "secret"})], True)])
def test_eof_and_trailing_event_do_not_invent_model_phases(entries, trailing):
    with trace_stream(entries, trailing=trailing) as (provider, trace, _):
        with pytest.raises(EOFError):
            with provider._open_event_stream("session") as stream:
                consume(provider, stream)
        summary, = trace["events"]
    assert summary["outcome"] == "error"
    assert summary["event_count"] == len(entries)
    assert "time_to_completion_seconds" not in summary
    assert "time_to_first_model_activity_seconds" not in summary
    assert ("time_to_first_event_seconds" in summary) == bool(entries)
    assert "secret" not in json.dumps(summary)


def test_diagnostics_off_does_not_allocate_timing_state():
    with trace_stream([(1, {"type": "unknown"})], enabled=False) as (provider, trace, _):
        with patch("resident.provider._AgentsStreamTiming", side_effect=AssertionError):
            with provider._open_event_stream("session") as stream:
                assert len(list(stream)) == 1
        summary, = trace["events"]
    assert "time_to_first_event_seconds" not in summary


@pytest.mark.parametrize("location", ["turn", "event", "session", "missing"])
def test_usage_from_existing_completion_data_without_requests(location):
    usage = {"input_tokens": 9, "output_tokens": 3,
             "input_tokens_details": {"cached_tokens": 0}}
    provider = OpenAIAgentsProvider("key", "model")
    provider._request = lambda *a, **kw: pytest.fail("usage must not require a request")
    if location in {"turn", "event"}:
        event = completed(usage=usage if location == "turn" else None)
        if location == "event":
            event["usage"] = usage
        turn = consume(provider, iter([item_event("added"), item_event("done"), event]))
    else:
        turn = provider._completed_turn("session", {"usage": usage} if location == "session" else {},
                                        {"id": "turn"}, streamed_message="done")
    assert turn.input_tokens == (None if location == "missing" else 9)
    assert turn.output_tokens == (None if location == "missing" else 3)
    assert turn.cached_input_tokens == (None if location == "missing" else 0)


def test_function_item_is_activity_and_malformed_stream_keeps_partial_timing():
    entries = [(1, item_event("added", item_type="function_call"))]
    with trace_stream(entries) as (provider, trace, response):
        original = list(response.lines)
        response.lines = iter(original + [(3, b"data: {invalid}\n"), (3, b"\n")])
        with pytest.raises(ValueError):
            with provider._open_event_stream("session") as stream:
                consume(provider, stream)
        summary, = trace["events"]
    assert summary["event_count"] == 1
    assert summary["outcome"] == "error"
    assert summary["time_to_first_model_activity_seconds"] == 1
    assert "time_to_first_output_seconds" not in summary
    assert "time_to_completion_seconds" not in summary


def test_runtime_logs_cached_usage_including_zero(tmp_path):
    import asyncio
    from resident.config import Config
    from resident.domain import ModelTurn, WakeEvent
    from resident.runtime import ResidentRuntime
    from resident.store import utc_now

    class Provider:
        async def respond(self, *_args, **_kwargs):
            return ModelTurn("turn", input_tokens=9, output_tokens=3, cached_input_tokens=0)

    runtime = ResidentRuntime(Config(tmp_path), Provider(), owner_output=lambda _: None)
    try:
        asyncio.run(runtime.process(WakeEvent("event", "test", "test", utc_now(), {})))
        row = runtime.store.connection.execute(
            "SELECT data_json FROM journal WHERE event_type='model.responded'").fetchone()
        data = json.loads(row[0])
        assert data["input_tokens"] == 9
        assert data["output_tokens"] == 3
        assert data["cached_input_tokens"] == 0
        assert runtime.store.connection.execute(
            "SELECT count(*) FROM journal WHERE event_type='timeline'").fetchone()[0] == 0
    finally:
        runtime.close()


def test_identityless_items_before_wake_correlation_do_not_mark_model_phases():
    with trace_stream([(1, item_event("added", turn=None))]) as (provider, trace, _):
        with pytest.raises(EOFError):
            with provider._open_event_stream("session") as stream:
                consume(provider, stream, expected=None)
        summary, = trace["events"]
    assert summary["time_to_first_event_seconds"] == 1
    assert "time_to_first_model_activity_seconds" not in summary
    assert "time_to_first_output_seconds" not in summary


@contextmanager
def settings_trace(*, enabled=True):
    provider = OpenAIAgentsProvider("key", "model", reasoning_effort="none")
    trace = {"events": []}
    trace_token = _agents_http_trace.set(trace)
    reporter_token = timeline_reporter.set((lambda _: None) if enabled else None)
    try:
        yield provider, trace
    finally:
        timeline_reporter.reset(reporter_token)
        _agents_http_trace.reset(trace_token)


@pytest.mark.parametrize("reasoning,reason", [
    ({"effort": "low", "summary": None}, "effort_changed"),
    ({"summary": None}, "effort_presence_changed"),
    (None, "reasoning_shape_changed"),
])
def test_settings_difference_reports_real_managed_difference(reasoning, reason):
    with settings_trace() as (provider, trace):
        result = provider._mutable_patch({"model": "model", "reasoning": reasoning})
    assert result == {"reasoning": {"effort": "none"}}
    event, = trace["events"]
    assert event["timeline_operation"] == "openai.agents_settings_difference"
    assert event["changed_settings"] == ["reasoning"]
    assert event["reasoning_source"] == "remote"
    assert event["desired_reasoning_effort"] == "none"
    assert event["desired_reasoning_fields"] == ["effort"]
    assert event["returned_reasoning_fields"] == sorted(reasoning or {})
    assert event["current_reasoning_fields"] == event["returned_reasoning_fields"]
    assert event["reasoning_change_reason"] == reason
    if isinstance(reasoning, dict) and "effort" in reasoning:
        assert event["current_reasoning_effort"] == reasoning["effort"]
    else:
        assert "current_reasoning_effort" not in event


def test_settings_difference_identifies_all_changed_names_without_values():
    with settings_trace() as (provider, trace):
        provider.service_tier = "priority-secret"
        result = provider._mutable_patch({
            "model": "model-secret", "service_tier": "tier-secret",
            "reasoning": {"effort": "low", "summary": "summary-secret"},
            "instructions": "instructions-secret", "messages": ["message-secret"],
        })
    assert set(result) == {"model", "reasoning", "service_tier"}
    event, = trace["events"]
    assert event["changed_settings"] == ["model", "reasoning", "service_tier"]
    assert event["reasoning_change_reason"] == "effort_changed"
    assert "secret" not in json.dumps(event)


@pytest.mark.parametrize("enabled", [True, False])
def test_equal_settings_emit_no_difference(enabled):
    with settings_trace(enabled=enabled) as (provider, trace):
        assert provider._mutable_patch({
            "model": "model", "reasoning": {"effort": "none"}}) == {}
    assert trace["events"] == []


def test_settings_diagnostics_off_preserves_patch_without_building_record():
    with settings_trace(enabled=False) as (provider, trace):
        with patch.object(provider, "_trace_mutable_settings_difference",
                          side_effect=AssertionError("diagnostics must stay off")):
            assert provider._mutable_patch({"reasoning": {"effort": "low"}}) == {
                "reasoning": {"effort": "none"}}
    assert trace["events"] == []


@pytest.mark.parametrize("applied,source,reason", [
    ({"reasoning": {"effort": "low"}}, "persisted", "effort_changed"),
    ({}, "unknown", "current_unknown"),
])
def test_settings_difference_uses_same_fallback_as_comparison(applied, source, reason):
    with settings_trace() as (provider, trace):
        provider._mutable_settings_descriptor = applied
        assert provider._mutable_patch({"model": "model"}) == {
            "reasoning": {"effort": "none"}}
    event, = trace["events"]
    assert event["reasoning_source"] == source
    assert event["returned_reasoning_fields"] == []
    assert event["current_reasoning_fields"] == (["effort"] if applied else [])
    assert event["reasoning_change_reason"] == reason


def test_settings_difference_bounds_names_and_redacts_unknown_effort():
    reasoning = {f"field_{i:02}": "value-secret" for i in range(30)}
    reasoning.update({"effort": "effort-secret", "bad\nname": "secret",
                      "x" * 65: "secret"})
    with settings_trace() as (provider, trace):
        provider._mutable_patch({"model": "model", "reasoning": reasoning})
    event, = trace["events"]
    assert len(event["returned_reasoning_fields"]) == 16
    assert all(len(name) <= 64 and "\n" not in name
               for name in event["returned_reasoning_fields"])
    assert event["current_reasoning_effort"] == "unknown"
    assert "secret" not in json.dumps(event)


def test_settings_difference_keeps_reconciliation_post_and_safe_journal_fields(tmp_path):
    from resident.store import Store

    with settings_trace() as (provider, trace):
        provider._session_id = "session"
        remote = {"model": "model", "reasoning": {"effort": "low", "summary": None}}
        requests = []

        def request(method, path, body=None):
            requests.append((method, path, body))
            return {"id": "session", "status": "idle", "agent": remote}

        provider._request = request
        session, created = provider._ensure_session([])
    assert session["id"] == "session" and not created
    assert requests == [
        ("GET", "/agents/sessions/session", None),
        ("POST", "/agents/sessions/session", {"agent": {"reasoning": {"effort": "none"}}}),
    ]
    event, = trace["events"]
    store = Store(tmp_path / "diagnostics.sqlite3")
    try:
        store.journal("timeline", {"operation": event.pop("timeline_operation"),
                                  "moment": "finished", **event}, run_id="run")
        # Scalar diagnostics also survive the existing safe journal projection.
        safe, truncated = store._safe_journal_events("run")
        assert not truncated
        assert safe[0]["reasoning_source"] == "remote"
        assert safe[0]["desired_reasoning_effort"] == "none"
        assert safe[0]["current_reasoning_effort"] == "low"
        assert safe[0]["reasoning_change_reason"] == "effort_changed"
        row = store.connection.execute(
            "SELECT data_json FROM journal WHERE event_type='timeline'").fetchone()
        recorded = json.loads(row[0])
        assert recorded["changed_settings"] == ["reasoning"]
        assert recorded["returned_reasoning_fields"] == ["effort", "summary"]
    finally:
        store.close()


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("reasoning", [
    {"effort": "none", "summary": None},
    {"effort": "none", "summary": "auto", "server_default": "unmanaged"},
])
def test_matching_managed_reasoning_ignores_returned_defaults(reasoning, enabled):
    with settings_trace(enabled=enabled) as (provider, trace):
        provider._session_id = "session"
        requests = []
        remote = {"model": "model", "reasoning": reasoning}

        def request(method, path, body=None):
            requests.append((method, path, body))
            return {"id": "session", "status": "idle", "agent": remote}

        provider._request = request
        session, created = provider._ensure_session([])
    assert session["agent"] == remote and not created
    assert requests == [("GET", "/agents/sessions/session", None)]
    assert trace["events"] == []


@pytest.mark.parametrize("current,expected", [
    ({"effort": "none", "summary": "auto", "extra": None}, {}),
    ({"effort": "none", "summary": None}, {"effort": "none", "summary": "auto"}),
    ({"effort": "none"}, {"effort": "none", "summary": "auto"}),
])
def test_reasoning_comparison_covers_every_explicit_desired_field(current, expected):
    # Public configuration currently exposes only effort. Exercise the partial
    # object comparator without introducing any new configuration/API settings.
    desired = {"model": "model", "reasoning": {"effort": "none", "summary": "auto"}}
    with settings_trace() as (provider, trace):
        with patch.object(provider, "_desired_mutable_settings", return_value=desired):
            result = provider._mutable_patch({"model": "model", "reasoning": current})
    assert result == ({"reasoning": expected} if expected else {})
    if expected:
        event, = trace["events"]
        assert event["reasoning_change_reason"] == "object_changed_equal_effort"
        assert event["desired_reasoning_fields"] == ["effort", "summary"]
    else:
        assert trace["events"] == []


@pytest.mark.parametrize("current,expected", [
    ({"effort": "none", "summary": None}, {}),
    ({"effort": "low", "summary": None}, {"reasoning": {"effort": "none"}}),
])
def test_partial_reasoning_comparison_also_applies_to_persisted_fallback(current, expected):
    with settings_trace() as (provider, _):
        provider._mutable_settings_descriptor = {"reasoning": current}
        assert provider._mutable_patch({"model": "model"}) == expected


def test_unmanaged_reasoning_and_service_tier_remain_unmanaged():
    with settings_trace() as (provider, trace):
        provider.reasoning_effort = None
        assert provider._mutable_patch({
            "model": "different-model", "reasoning": {"effort": "high", "summary": None},
            "service_tier": "priority"}) == {"model": "model"}
    event, = trace["events"]
    assert event["changed_settings"] == ["model"]
    assert "reasoning_change_reason" not in event
