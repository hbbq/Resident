"""Transport contracts and real HTTP/1.1 pool integration."""
import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock, patch

import httpx
import pytest

from resident.observability import timeline_reporter
from resident.provider import OpenAIAgentsProvider, _agents_http_trace
from resident.runtime import ResidentRuntime


@contextmanager
def mocked(handler, **kwargs):
    client_type = httpx.Client

    def client(**options):
        return client_type(transport=httpx.MockTransport(handler), **options)

    with patch("resident.provider.httpx.Client", side_effect=client):
        provider = OpenAIAgentsProvider("key", "model", **kwargs)
    try:
        yield provider
    finally:
        provider.close()


class Chunks(httpx.SyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    def __iter__(self):
        yield from self.chunks

    def close(self):
        self.closed = True


def test_rest_wire_contract_empty_ack_and_cookies():
    seen = []

    def handle(request):
        seen.append(request)
        if request.method == "POST":
            return httpx.Response(204)
        return httpx.Response(200, content=b'{"status":"idle"}',
                              headers={"Set-Cookie": "session=unexpected"})

    with mocked(handle, base_url="https://example.test/v1/", timeout_seconds=17) as provider:
        assert provider._request("GET", "/agents/sessions/s?limit=100") == {"status": "idle"}
        provider._submit_events("s", [{"type": "input"}], "wake-key")
        provider._request("GET", "/agents/sessions/s")
    assert str(seen[0].url) == "https://example.test/v1/agents/sessions/s?limit=100"
    post = seen[1]
    assert post.method == "POST"
    assert post.content == json.dumps({"events": [{"type": "input"}]}).encode()
    assert post.headers["Authorization"] == "Bearer key"
    assert post.headers["OpenAI-Beta"] == "agents=v1"
    assert post.headers["Content-Type"] == "application/json"
    assert post.headers["Idempotency-Key"] == "wake-key"
    assert "cookie" not in post.headers and "cookie" not in seen[2].headers
    assert post.headers["Connection"] == "keep-alive"
    assert post.extensions["timeout"] == dict.fromkeys(["connect", "read", "write", "pool"], 17)


@pytest.mark.parametrize("payload,expected", [
    (b': heartbeat\r\nevent: envelope\r\nid: ignored\r\ndata: {"type":\r\ndata: "ok"}\r\n\r\ndata: [DONE]\n\n', [{"type": "ok"}]),
    (b'data: {"type":"trailing"}', [{"type": "trailing"}]),
])
def test_sse_chunk_boundaries_and_close(payload, expected):
    body = Chunks([payload[i:i + 3] for i in range(0, len(payload), 3)])

    def handle(request):
        assert request.method == "GET"
        assert request.headers["Accept"] == "text/event-stream"
        return httpx.Response(200, stream=body)

    with mocked(handle) as provider:
        with provider._open_event_stream("s") as events:
            assert list(events) == expected
        assert body.closed


@pytest.mark.parametrize("payload,reason", [
    (b'data: {"text":"\xff"}\n\n', "sse_invalid_utf8"),
    (b'data: []\n\n', "sse_non_object"),
    (b'data: {broken}\n\n', "sse_invalid_json"),
])
def test_parser_error_closes_stream(payload, reason):
    body = Chunks([payload])
    with mocked(lambda _: httpx.Response(200, stream=body)) as provider:
        with pytest.raises(ValueError, match=reason):
            with provider._open_event_stream("s") as events:
                list(events)
        assert body.closed


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("status,definitive", [(400, True), (409, True), (429, True), (500, False)])
def test_http_errors_preserve_rejection_classification(streaming, status, definitive):
    body = Chunks([b"rejected"])
    with mocked(lambda _: httpx.Response(status, stream=body)) as provider:
        with pytest.raises(RuntimeError, match=f"HTTP {status}: rejected") as caught:
            if streaming:
                with provider._open_event_stream("s"):
                    pytest.fail("HTTP failure cannot yield a stream")
            else:
                provider._request("POST", "/agents/sessions/s/events", {})
        assert provider._submission_was_definitively_rejected(caught.value) is definitive
        assert isinstance(caught.value.__cause__, httpx.HTTPStatusError)
        assert body.closed


@pytest.mark.parametrize("failure", [httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError,
                                     httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout,
                                     httpx.PoolTimeout])
@pytest.mark.parametrize("streaming", [False, True])
def test_transport_errors_and_timeouts_without_retries(failure, streaming):
    calls = []

    def handle(request):
        calls.append(request)
        raise failure("failed", request=request)

    with mocked(handle) as provider:
        expected = TimeoutError if issubclass(failure, httpx.TimeoutException) else failure
        with pytest.raises(expected) as caught:
            if streaming:
                with provider._open_event_stream("s"):
                    pytest.fail("failed connect")
            else:
                provider._submit_events("s", [], "key")
        assert not provider._submission_was_definitively_rejected(caught.value)
        assert provider._stream_fallback_reason(caught.value) == "stream_timeout_or_disconnect"
        assert len(calls) == 1


def test_stream_read_failure_releases_response():
    class Broken(Chunks):
        def __iter__(self):
            yield b": heartbeat\n"
            raise httpx.ReadTimeout("stalled")

    body = Broken([])
    with mocked(lambda _: httpx.Response(200, stream=body)) as provider:
        with pytest.raises(TimeoutError):
            with provider._open_event_stream("s") as events:
                list(events)
        assert body.closed


def test_provider_client_and_runtime_shutdown():
    provider = OpenAIAgentsProvider("key", "model")
    client = provider._client
    runtime = object.__new__(ResidentRuntime)
    runtime.provider = provider
    runtime.realm_client = None
    runtime._curator_coordinator = Mock()
    runtime.store = Mock()
    runtime.close()
    runtime.store.close.assert_called_once()
    assert client.is_closed
    provider.close()
    assert provider._client is client


@contextmanager
def server():
    requests = []
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def do_GET(self):
            requests.append((self.command, self.path, self.client_address, dict(self.headers)))
            if self.path.endswith("/events"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                self.wfile.flush()
                if release.wait(5):
                    payload = b'data: {"type":"ok"}\n\n'
                    try:
                        self.wfile.write(f"{len(payload):x}\r\n".encode() + payload + b"\r\n0\r\n\r\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                        pass
            else:
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

        def do_POST(self):
            requests.append((self.command, self.path, self.client_address, dict(self.headers)))
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
            release.set()

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=httpd.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}", requests, release
    finally:
        release.set()
        httpd.shutdown()
        httpd.server_close()
        worker.join(2)


def test_real_http11_pool_reuses_rest_and_allows_post_with_sse_open():
    with server() as (url, requests, _):
        provider = OpenAIAgentsProvider("key", "model", base_url=url, timeout_seconds=2)
        trace = {"events": []}
        trace_token = _agents_http_trace.set(trace)
        reporter_token = timeline_reporter.set(lambda _: None)
        try:
            provider._request("GET", "/agents/sessions/s")
            provider._request("GET", "/agents/sessions/s")
            assert requests[0][2] == requests[1][2]
            with provider._open_event_stream("s") as events:
                provider._submit_events("s", [{"type": "input"}], "wake")
                assert requests[2][2] != requests[3][2]
                assert list(events) == [{"type": "ok"}]
            provider._request("GET", "/agents/sessions/s")
            assert requests[-1][2] in {entry[2] for entry in requests[:-1]}
            assert all(entry[3].get("Connection") != "close" for entry in requests)
            assert all(event["http_version"] == "HTTP/1.1" for event in trace["events"])
        finally:
            timeline_reporter.reset(reporter_token)
            _agents_http_trace.reset(trace_token)
            provider.close()


def test_real_sse_deadline_includes_time_spent_before_reading():
    with server() as (url, _, _):
        provider = OpenAIAgentsProvider("key", "model", base_url=url, timeout_seconds=0.6)
        try:
            started = time.monotonic()
            with pytest.raises(TimeoutError):
                with provider._open_event_stream("s") as events:
                    time.sleep(0.35)
                    list(events)
            assert 0.5 <= time.monotonic() - started < 0.9
        finally:
            provider.close()


@pytest.mark.parametrize("failure", [400, 500, "disconnect", "timeout"])
def test_wake_submission_checkpoint_rejection_and_uncertainty(failure):
    order = []
    body = Chunks([])

    def handle(request):
        if request.method == "GET":
            order.append("subscribe")
            return httpx.Response(200, stream=body)
        assert order == ["subscribe", "checkpoint"]
        assert not body.closed
        assert request.headers["Idempotency-Key"] == "resident-wake:wake"
        order.append("submit")
        if failure == "disconnect":
            raise httpx.ReadError("unknown outcome", request=request)
        if failure == "timeout":
            raise httpx.ReadTimeout("unknown outcome", request=request)
        return httpx.Response(failure, content=b"rejected")

    with mocked(handle) as provider:
        mark = provider._mark_wake_attempted

        def checkpoint(*args):
            order.append("checkpoint")
            mark(*args)

        provider._mark_wake_attempted = checkpoint
        provider._fallback_wait = Mock(return_value="reconciled")
        if failure == 400:
            with pytest.raises(RuntimeError, match="HTTP 400"):
                provider._submit_wake("s", "context", "wake", "correlation")
            assert provider._load_wake_submission("s", "wake") is None
            provider._fallback_wait.assert_not_called()
        else:
            assert provider._submit_wake("s", "context", "wake", "correlation") == "reconciled"
            assert provider._load_wake_submission("s", "wake") is not None
            provider._fallback_wait.assert_called_once()
        assert order == ["subscribe", "checkpoint", "submit"]
        assert body.closed


@pytest.mark.parametrize("early_exit", [False, True])
def test_deadline_adapter_restored_before_response_close(early_exit):
    from types import SimpleNamespace
    reads = []
    source = iter([b'data: {"type":"ok"}\n\n', b"", b""])

    def read(max_bytes, timeout=None):
        reads.append(timeout)
        return next(source)

    network = SimpleNamespace(read=read)

    class Body(httpx.SyncByteStream):
        def __iter__(self):
            while chunk := network.read(65536, timeout=120):
                yield chunk

        def close(self):
            assert network.read is read

    with mocked(lambda _: httpx.Response(200, stream=Body(),
                                        extensions={"network_stream": network})) as provider:
        with provider._open_event_stream("s") as events:
            assert next(events) == {"type": "ok"}
            if not early_exit:
                assert list(events) == []
        assert network.read is read
        assert reads and all(0 < timeout <= provider.timeout_seconds for timeout in reads)
