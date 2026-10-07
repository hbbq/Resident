# Agents HTTP transport latency investigation

Investigation date: 2026-10-07. Transport and higher-level behavior are unchanged.
The redundant settings POST fix and existing settings diagnostics were already
present in the working tree and were preserved.

## Established connection behavior

Resident uses stock urllib.request.urlopen for both REST and SSE. There is no
custom opener, connection pool, explicit SSLContext, or saved SSLSession in the
adapter. Repository search found no install_opener override.

Inspection of the installed CPython 3.14.7 AbstractHTTPHandler.do_open establishes:

- Each HTTP open creates a new HTTPConnection/HTTPSConnection object. It cannot
  reuse a previous request's TCP connection. Redirects can create further connections.
- do_open overwrites the outgoing Connection header with close, although that
  header is absent from Resident's Request constructor. It also detaches/closes
  the connection's socket handle after obtaining the response. The response can
  still consume its body through its file object, including an ongoing SSE body.
- Requests use HTTP/1.1; the default HTTPS context offers only http/1.1 via ALPN.
  urllib has no HTTP/2 implementation. The new response-version field records
  the actual response version when available.
- HTTPError also means response headers were obtained. URLError/timeout before
  the response boundary does not establish any completed HTTP phase.

An HTTP proxy can retain its own upstream connections; these findings concern
Resident's client-side connection, not hidden proxy-to-origin behavior. External
custom openers could change stock urllib behavior, but none is installed by this
repository. OS DNS caching may reduce DNS latency despite fresh TCP connections.

TLS resumption is available in Python through an explicit SSLSession, but the
inspected HTTPSConnection.connect does not pass one to SSLContext.wrap_socket.
CPython 3.14's default HTTPSHandler retains an SSLContext across calls; the checked
Python 3.11 handler instead leaves context=None, yielding a new default context
per HTTPSConnection. Neither implementation deliberately resumes a client TLS
session. Resumption is therefore not an expected saving in the present adapter;
actual session_reused was not measured. A shared context is not proof of resumption.
See [Python SSL documentation](https://docs.python.org/3/library/ssl.html).

## Observed wake budget

These are the user's supplied production measurements, not new live measurements.

| Request / interval | Observed duration | Interpretation |
| --- | ---: | --- |
| Runtime preflight | 0.547 s | Whole provider preflight; HTTP GET plus local/executor overhead |
| Lifecycle session GET | 0.484 s | Complete GET; connection, endpoint, body and parse combined |
| SSE open start to input POST start | 0.984 s | 1.468 - 0.484; SSE headers plus checkpoint and local preparation |
| Input POST | 1.454 s | Entire request and acknowledgement; header/body split unavailable in old sample |
| Real settings POST | No sample | Redundant POST removed; instrument the next naturally occurring real change |

The combined envelope above is 3.469 s (about 36% of the 9.56 s wake). It is NOT
3.469 s of connection overhead, and includes the entire provider preflight rather
than its GET alone. SSE connection establishment can account for at most the
0.984 s pre-submit interval in this sample; local checkpoint cost is also inside it.

Input acknowledgement finishes at lifecycle +2.922 s. The first turn.created is
observed at lifecycle +7.078 s, leaving 4.156 s after acknowledgement. Completion
is observed at lifecycle +8.797 s, leaving 5.875 s after acknowledgement. Removing
handshakes cannot directly eliminate those later waits. These are client observation
times, not server emission timestamps: SSE consumption starts after POST returns,
so events emitted earlier can already be buffered.

Known connection/handshake overhead: every stock urllib request needs a new
client-side TCP connection and HTTPS handshake. Its numeric duration is unknown.
No measured seconds can be isolated as DNS, TCP or TLS from this sample. Header
wait includes all of them plus upload, network transit, and endpoint/server time.
Do not classify the 4.156 s interval as pure model computation either: it can
include scheduling, endpoint processing, buffering, and stream delivery.

## New opt-in diagnostics

All new observations are behind the existing --timeline / RESIDENT_TIMELINE flag.
Normal runtime operation introduces no extra network requests. No URLs, request
or response bodies, credentials, header values, or message content are recorded.

REST records (operation=openai.agents_http):

- request=poll_session with request_phase=preflight identifies the runtime GET.
- request=poll_session with request_phase=lifecycle identifies lifecycle GETs.
- request=submit_wake identifies input POST; submit_tool_results remains distinct.
- request=update_session now also identifies the actual agent-settings POST.
- time_to_headers_seconds: request timer start to urlopen return.
- headers_to_body_seconds: header boundary to response.read completion.
- body_parse_seconds: complete bytes to JSON parsing completion.
- headers_to_parsed_body_seconds: header boundary to full parsed acknowledgement.
- headers_available_monotonic_seconds and http_version provide a safe response boundary.

Empty allowed acknowledgements report zero JSON parse time. Failures retain only
completed phases; malformed JSON does not report successful parsed-body timing.
HTTP errors report header and error-body boundaries without retaining the body.

SSE records (operation=openai.agents_stream):

- time_to_headers_seconds and headers_available_monotonic_seconds measure when
  urlopen returns with response headers. Stream duration and existing event offsets
  still start at the original stream-open start; their meaning has not changed.
- headers_to_checkpoint_seconds is recorded only after the durable wake-attempt
  checkpoint returns successfully. It includes context-manager yield, event-loop
  scheduling and checkpoint work. It does not imply the interval is solely disk I/O.
- No checkpoint field is invented for tool-result streams, recovered turns,
  creation-time input, or failed checkpoints.

The existing lifecycle_offset_seconds gives input POST start relative to lifecycle
start. Combining SSE start + header wait + checkpoint interval gives checkpoint
completion; the remaining gap to POST start is local submit preparation.
Preflight records are buffered in the worker and emitted on the event-loop thread,
so a SQLite-backed timeline reporter keeps its existing thread ownership.

DNS, TCP connect, TLS handshake, TLS resumption and socket identity are not measured.
Separating them would require transport trace hooks or invasive interception; no
socket monkey-patching, opener replacement or transport replacement was performed.
A successful header return is also not proof that the SSE service has delivered
its first event; that remains the existing first-event observation.

## Endpoint capability and replacement choice

Persistent connections are the expected normal API-client behavior, and modern
Python clients provide pooling. HTTPX explicitly documents Client connection reuse.
There is no evidence here that Agents requires Connection: close. A read-only,
unauthenticated session-path HTTPX persistence probe failed with ConnectError in
this network-restricted environment, before receiving a response. Therefore
endpoint/proxy keep-alive and ALPN negotiation have NOT been verified live; no
endpoint-specific guarantee is inferred from SDK usage.

HTTP/1.1 needs separate connections for the open SSE response and overlapping POST.
A pool can reuse preflight/lifecycle/settings connections, but cannot send the POST
on an occupied SSE HTTP/1.1 connection. Keep at least two connection slots. Closing
an incompletely consumed SSE body may discard that connection, so do not assume
that every wake becomes handshake-free merely by introducing a pool.

HTTP/2 can carry SSE and POST as two streams on one connection to the same origin.
They do not necessarily require separate TCP connections. This requires actual
h2 negotiation, adequate server concurrent-stream limits, and compatible proxy and
client behavior. It can avoid the second connection setup and improve concurrency;
it does not remove server/model wait or change the required checkpoint ordering.
See [HTTPX HTTP/2 documentation](https://www.python-httpx.org/http2/).

The safest replacement candidate is a long-lived synchronous httpx.Client, fitting
the existing blocking worker and SSE iterator. HTTPX is already a direct Resident
dependency (>=0.27,<1), installed at 0.28.1 with httpcore 1.0.9 in the checkout venv,
and used by ONVIF. No OpenAI SDK is installed or declared. aiohttp and aiodns are
also declared, but switching to them would require async restructuring. The h2
optional dependency is not installed in the checkout venv; HTTP/1.1 pooling needs
no new dependency, while an HTTP/2 trial needs httpx[http2].
See [HTTPX clients](https://www.python-httpx.org/advanced/clients/).

Pooling should reduce repeated setup; a meaningful improvement is plausible but
unproven. There is no defensible numeric saving estimate, and this sample does not
support attributing most of the ~9.56 s wake to handshakes. OpenAI's latency guidance
also separates request round-trip cost from other work:
[Latency optimization](https://developers.openai.com/api/docs/guides/latency-optimization).

## Smallest clean later A/B experiment

1. First collect several unchanged urllib --timeline wakes with these diagnostics.
   Capture a real settings POST only when a real configuration change occurs.
2. In an isolated experimental branch, adapt only REST/SSE I/O to one provider-owned
   synchronous HTTPX client. Preserve both GETs, identical request bytes/headers,
   deadlines, subscribe -> checkpoint -> submit, parser, and error/recovery semantics.
   Disable client transport retries and translate HTTPX failures into the existing
   recovery classifications. Do not introduce SDK retries or session behavior.
3. Compare the SAME HTTPX code with keep-alive disabled (max_keepalive_connections=0)
   and enabled (at least two connection slots). Keep HTTP/2 disabled for this first
   comparison. This isolates pooling better than comparing urllib directly to HTTPX.
   Keep the urllib baseline to quantify any additional client-stack difference.
4. Use a disposable session with unchanged model/context settings and interleave
   10-20 simple wakes per arm. Compare header wait, checkpoint interval, POST body/
   parse, first activity, completion, and total wake median/tail. Separate cold-start
   samples from warm reuse and short idle gaps from realistic between-wake gaps.
   Use HTTPX/httpcore trace hooks to count connect/TLS events without logging payloads.
5. Only then compare pooled HTTP/1.1 with pooled HTTP/2; record negotiated protocol
   and connection creation. Confirm SSE and POST actually multiplex rather than
   assuming http2=True guarantees it. Close the provider client on runtime shutdown.

Do not replay a possibly accepted live input POST as a timing probe. The current
patch implements diagnostics only, not any of these future experiment transports.

## Validation

Focused suite: 211 passed, 37 subtests passed (Agents timing, transport timing, runtime),
on both Python 3.11.7 and the checkout Python 3.14.7. For the supported Python 3.14
run, the existing dependency-complete environment's site-packages directory was
appended to sys.path in a one-off test launcher; available pure Python fallbacks
were used. No fake dependency modules were used and no runtime environment was
changed. The broader Resident suite passed 472 tests and 144 subtests on Python 3.11.7;
the final two additional edge-case tests are covered by the focused runs above.
AST syntax checks and git diff --check also passed. Package downloads
and the live endpoint probe were network-blocked.
