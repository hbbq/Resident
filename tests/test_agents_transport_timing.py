"""HTTP phase diagnostics only: no live requests or transport substitution."""
import asyncio
import json
import threading
from contextlib import contextmanager
from unittest.mock import patch

import httpx
import pytest

from tests.httpx_support import ResponseMixin

from resident.observability import timeline_reporter
from resident.provider import OpenAIAgentsProvider, _agents_http_trace


@contextmanager
def tracing(enabled=True):
    records = []
    trace = {"events": [], "request_phase": "lifecycle"}
    token = _agents_http_trace.set(trace)
    reporter_token = timeline_reporter.set(records.append if enabled else None)
    try:
        yield trace, records
    finally:
        timeline_reporter.reset(reporter_token)
        _agents_http_trace.reset(token)


class Response(ResponseMixin):
    version = 11

    def __init__(self, clock, payload=b'{"status":"idle"}', status=200):
        self.status_code = status
        self.clock, self.payload = clock, payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def read(self):
        self.clock[0] = 103.0
        return self.payload


@pytest.mark.parametrize("method,path,body,classification", [
    ("GET", "/agents/sessions/session-secret", None, "poll_session"),
    ("POST", "/agents/sessions/session-secret/events", {
        "events": [{"type": "agent.session.input.message", "input": "body-secret"}]},
        "submit_wake"),
    ("POST", "/agents/sessions/session-secret", {"agent": {"model": "model-secret"}},
        "update_session"),
])
def test_rest_header_body_and_parse_phases(method, path, body, classification):
    clock = [100.0]
    provider = OpenAIAgentsProvider("auth-secret", "model")
    response = Response(clock)
    loads = json.loads

    def open_response(*args, **kwargs):
        assert kwargs == {"stream": True}
        assert args[0].extensions["timeout"]["read"] == provider.timeout_seconds
        clock[0] = 102.0
        return response

    def parse(payload):
        clock[0] = 104.0
        return loads(payload)

    with tracing() as (trace, _), patch("resident.provider.time.monotonic", side_effect=lambda: clock[0]), patch(
            "resident.provider.httpx.Client.send", side_effect=open_response), patch(
            "resident.provider.json.loads", side_effect=parse):
        assert provider._request(method, path, body) == {"status": "idle"}
    event, = trace["events"]
    assert event["request"] == classification
    assert event["request_phase"] == "lifecycle"
    assert event["http_version"] == "HTTP/1.1"
    assert event["time_to_headers_seconds"] == 2
    assert event["headers_to_body_seconds"] == 1
    assert event["body_parse_seconds"] == 1
    assert event["headers_to_parsed_body_seconds"] == 2
    assert event["duration_seconds"] == 4
    assert "secret" not in json.dumps(event)


@pytest.mark.parametrize("failure", ["connect", "http", "parse"])
def test_failure_reports_only_observed_phases(failure):
    clock = [100.0]
    provider = OpenAIAgentsProvider("key", "model")

    def open_response(*_args, **_kwargs):
        clock[0] = 102.0
        if failure == "connect":
            raise httpx.ConnectError("secret")
        if failure == "http":
            return Response(clock, b"body-secret", status=400)
        return Response(clock, b"invalid-secret")

    with tracing() as (trace, _), patch("resident.provider.time.monotonic", side_effect=lambda: clock[0]), patch(
            "resident.provider.httpx.Client.send", side_effect=open_response):
        with pytest.raises((RuntimeError, httpx.ConnectError, ValueError)):
            provider._request("GET", "/agents/sessions/secret")
    event, = trace["events"]
    assert event["outcome"] == "error"
    assert ("time_to_headers_seconds" in event) == (failure != "connect")
    assert ("headers_to_body_seconds" in event) == (failure != "connect")
    assert "headers_to_parsed_body_seconds" not in event
    assert "secret" not in json.dumps(event)


def test_diagnostics_off_does_not_inspect_response():
    provider = OpenAIAgentsProvider("key", "model")
    with tracing(enabled=False) as (trace, _), patch(
            "resident.provider.httpx.Client.send", return_value=Response([100])), patch.object(
            provider, "_http_response_timing", side_effect=AssertionError):
        assert provider._request("GET", "/agents/sessions/secret") == {"status": "idle"}
    assert "time_to_headers_seconds" not in trace["events"][0]


@pytest.mark.parametrize("fail", [False, True])
def test_preflight_trace_emitted_on_event_loop_and_context_restored(fail):
    provider = OpenAIAgentsProvider("key", "model")
    provider._session_id = "session-secret"
    event_loop_thread = threading.get_ident()
    records = []
    outer = {"events": []}
    token = _agents_http_trace.set(outer)
    reporter_token = timeline_reporter.set(
        lambda event: records.append((threading.get_ident(), event)))
    response = Response([100])
    response.payload = b'{"status":"idle"}'
    effect = httpx.ConnectError("secret") if fail else None
    try:
        with patch("resident.provider.httpx.Client.send", return_value=response, side_effect=effect):
            if fail:
                with pytest.raises(httpx.ConnectError):
                    asyncio.run(provider.preflight_session())
            else:
                assert asyncio.run(provider.preflight_session()) is None
        assert _agents_http_trace.get() is outer
    finally:
        timeline_reporter.reset(reporter_token)
        _agents_http_trace.reset(token)
    assert outer["events"] == []
    thread, event = records[0]
    assert thread == event_loop_thread
    assert event["operation"] == "openai.agents_http"
    assert event["request_phase"] == "preflight"
    assert event["request"] == "poll_session"
    assert "secret" not in json.dumps(event)


@pytest.mark.parametrize("checkpoint_fails", [False, True])
def test_sse_headers_checkpoint_submit_order(checkpoint_fails):
    provider = OpenAIAgentsProvider("key", "model")
    clock = [100.0]
    order = []

    class Stream(ResponseMixin):
        version = 11

        def close(self):
            order.append("close")

    def open_response(*_args, **_kwargs):
        order.append("subscribe")
        clock[0] = 102.0
        return Stream()

    def checkpoint(*_args):
        order.append("checkpoint")
        clock[0] = 105.0
        if checkpoint_fails:
            raise RuntimeError("checkpoint-secret")

    def submit(*_args):
        assert order == ["subscribe", "checkpoint"]
        order.append("submit")
        clock[0] = 107.0

    provider._mark_wake_attempted = checkpoint
    provider._submit_events = submit
    provider._consume_event_stream = lambda *_args, **_kwargs: "complete"
    provider._fallback_wait = lambda *_args, **_kwargs: "fallback"
    with tracing() as (trace, _), patch("resident.provider.time.monotonic", side_effect=lambda: clock[0]), patch(
            "resident.provider.httpx.Client.send", side_effect=open_response):
        result = provider._submit_wake("session-secret", "message-secret", "wake-secret", "correlation-secret")
    event, = trace["events"]
    assert event["time_to_headers_seconds"] == 2
    assert event["http_version"] == "HTTP/1.1"
    if checkpoint_fails:
        assert result == "fallback"
        assert order == ["subscribe", "checkpoint", "close"]
        assert "headers_to_checkpoint_seconds" not in event
    else:
        assert result == "complete"
        assert order == ["subscribe", "checkpoint", "submit", "close"]
        assert event["headers_to_checkpoint_seconds"] == 3
    assert "secret" not in json.dumps(event)


def test_empty_acknowledgement_has_body_boundary_without_json_parse():
    provider = OpenAIAgentsProvider("key", "model")
    clock = [100.0]

    def opened(*_args, **_kwargs):
        clock[0] = 102.0
        return Response(clock, b"")

    with tracing() as (trace, _), patch("resident.provider.time.monotonic", side_effect=lambda: clock[0]), patch(
            "resident.provider.httpx.Client.send", side_effect=opened), patch(
            "resident.provider.json.loads", side_effect=AssertionError):
        assert provider._request("POST", "/agents/sessions/secret/events", allow_empty=True) == {}
    event, = trace["events"]
    assert event["headers_to_body_seconds"] == 1
    assert event["headers_to_parsed_body_seconds"] == 1
    assert event["body_parse_seconds"] == 0


def test_sse_http_rejection_retains_header_wait_only():
    provider = OpenAIAgentsProvider("key", "model")
    clock = [100.0]

    def rejected(*_args, **_kwargs):
        clock[0] = 102.0
        return Response(clock, b"body-secret", status=400)

    with tracing() as (trace, _), patch("resident.provider.time.monotonic", side_effect=lambda: clock[0]), patch(
            "resident.provider.httpx.Client.send", side_effect=rejected):
        with pytest.raises(RuntimeError):
            with provider._open_event_stream("session-secret"):
                pytest.fail("rejected stream must not yield")
    event, = trace["events"]
    assert event["outcome"] == "error"
    assert event["time_to_headers_seconds"] == 2
    assert "headers_to_checkpoint_seconds" not in event
    assert "secret" not in json.dumps(event)
