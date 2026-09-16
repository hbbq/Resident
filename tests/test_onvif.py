from __future__ import annotations

import asyncio
import json
import time
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch

import httpx

from resident.camera import CameraConnector
from resident.config import CameraConfig, OnvifConfig
from resident.onvif import (
    ADDRESSING, EVENTS, MAX_DIAGNOSTIC_NAMES_LENGTH, MAX_DIAGNOSTIC_NAME_LENGTH,
    NOTIFY_WSDL, TOPICS, OnvifClient, PropertyField, PropertySchema, PullPoint,
)


SECRET_ENDPOINT = "http://camera.test/onvif/device_service"
SECRET_USER = "onvif-user"
SECRET_PASSWORD = "onvif-password"


class BlockingTransport(httpx.AsyncBaseTransport):
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.active = 0
        self.maximum_active = 0
        self.aborted = 0

    async def handle_async_request(self, request):
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        self.started.set()
        try:
            await self.release.wait()
            return httpx.Response(200, content=b"<Envelope/>", request=request)
        except asyncio.CancelledError:
            self.aborted += 1
            raise
        finally:
            self.active -= 1


def client_using(transport):
    def factory(**options):
        return httpx.AsyncClient(transport=transport, **options)
    return factory


class StubClient(OnvifClient):
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls = []
        self.config = OnvifConfig(SECRET_ENDPOINT, SECRET_USER, SECRET_PASSWORD)
        self.request_timeout = 1
        self.pull_timeout = 2
        self._origin = ("http", "camera.test", 80)
        self._trusted_pullpoints = set()
        self._property_schemas = {}

    async def _post(self, destination, action, body, timeout=None, reference_parameters=(),
                    *, trusted_subscription=False):
        self.calls.append((destination, action, ET.tostring(body), timeout, reference_parameters))
        return ET.fromstring(self.responses.pop(0))


class OnvifClientTests(unittest.IsolatedAsyncioTestCase):
    def test_default_pull_messages_timeout_is_five_seconds(self):
        client = OnvifClient(OnvifConfig(SECRET_ENDPOINT, SECRET_USER, SECRET_PASSWORD))

        self.assertEqual(5.0, client.pull_timeout)

    async def test_default_short_timeout_is_sent_in_pull_messages(self):
        client = StubClient(["<Envelope/>"])
        client.pull_timeout = 5.0

        await client.pull(PullPoint("http://camera.test/pull"))

        self.assertIn(b"PT5S", client.calls[0][2])

    async def test_discovers_event_service_and_advertised_topics(self):
        client = StubClient([
            """<Envelope><Events><XAddr>http://camera.test/events</XAddr></Events></Envelope>""",
            """<Envelope xmlns:wstop='http://docs.oasis-open.org/wsn/t-1'>
                 <TopicSet><RuleEngine><CellMotionDetector wstop:topic='true'>
                   <MessageDescription IsProperty='true'><Data>
                     <SimpleItemDescription Name='IsMotion' Type='xsd:boolean'/></Data>
                   </MessageDescription>
                 </CellMotionDetector></RuleEngine></TopicSet>
               </Envelope>""",
        ])

        service, topics, schemas = await client.discover()

        self.assertEqual("http://camera.test/events", service)
        self.assertEqual(("RuleEngine/CellMotionDetector",), topics)
        self.assertEqual((PropertySchema(
            "RuleEngine/CellMotionDetector", data=(PropertyField("IsMotion", "xsd:boolean"),)),),
            schemas)
        self.assertTrue(client.calls[0][1].endswith("/GetCapabilities"))
        self.assertEqual(
            f"{EVENTS}/EventPortType/GetEventPropertiesRequest", client.calls[1][1])

    async def test_subscription_and_pull_use_complete_request_action_uris(self):
        client = StubClient([
            """<Envelope xmlns:wsa='http://www.w3.org/2005/08/addressing'>
              <SubscriptionReference><wsa:Address>http://camera.test/pull</wsa:Address>
              </SubscriptionReference></Envelope>""",
            "<Envelope/>",
        ])

        pullpoint = await client.subscribe("http://camera.test/events")
        await client.pull(pullpoint)

        self.assertEqual(
            f"{EVENTS}/EventPortType/CreatePullPointSubscriptionRequest",
            client.calls[0][1])
        self.assertEqual(
            f"{EVENTS}/PullPointSubscription/PullMessagesRequest", client.calls[1][1])

    def test_topic_traversal_includes_topics_nested_below_a_topic(self):
        topic_set = ET.fromstring(f"""
            <TopicSet xmlns:wstop='{TOPICS}'>
              <RuleEngine wstop:topic='true'>
                <Motion wstop:topic='true'/>
              </RuleEngine>
            </TopicSet>""")

        self.assertEqual(
            ("RuleEngine", "RuleEngine/Motion"), OnvifClient._topic_paths(topic_set))

    def test_advertised_topic_diagnostics_have_per_name_and_total_bounds(self):
        topic_set = ET.fromstring(
            f"<TopicSet xmlns:wstop='{TOPICS}'>"
            + "".join(
                f"<Topic{number}{'x' * 300} wstop:topic='true'/>" for number in range(100)
            )
            + "</TopicSet>")

        topics = OnvifClient._topic_paths(topic_set)

        self.assertTrue(all(len(topic) <= MAX_DIAGNOSTIC_NAME_LENGTH for topic in topics))
        self.assertLessEqual(sum(map(len, topics)), MAX_DIAGNOSTIC_NAMES_LENGTH)

    async def test_pull_reports_only_bounded_shape_not_values_or_raw_xml(self):
        client = StubClient(["""
            <Envelope><NotificationMessage><Topic>tns1:RuleEngine/Motion</Topic>
              <Message><Data><SimpleItem Name='IsMotion' Value='top-secret-value'/>
              </Data></Message></NotificationMessage></Envelope>"""])

        notifications = await client.pull(PullPoint("http://camera.test/pullpoint"))

        self.assertEqual(({
            "topic": "RuleEngine/Motion", "fields": ["IsMotion"],
        },), notifications)
        self.assertNotIn("top-secret-value", json.dumps(notifications))
        self.assertIn(b"PT2S", client.calls[0][2])

    async def test_advertised_property_schema_and_boolean_values_are_parsed(self):
        client = StubClient([
            "<Envelope><Events><XAddr>http://camera.test/events</XAddr></Events></Envelope>",
            f"""<Envelope xmlns:wstop='{TOPICS}'><TopicSet><RuleEngine>
              <CellMotionDetector><Motion wstop:topic='true'>
                <MessageDescription IsProperty='true'><Source>
                  <SimpleItemDescription Name='Rule' Type='xsd:string'/>
                </Source><Data>
                  <SimpleItemDescription Name='IsMotion' Type='xsd:boolean'/>
                </Data></MessageDescription>
              </Motion></CellMotionDetector></RuleEngine></TopicSet></Envelope>""",
            """<Envelope><NotificationMessage>
              <Topic>tns1:RuleEngine/CellMotionDetector/Motion</Topic><Message>
                <Source><SimpleItem Name='Rule' Value='rule-1'/></Source>
                <Data><SimpleItem Name='IsMotion' Value='true'/></Data>
              </Message></NotificationMessage></Envelope>""",
        ])

        _, _, schemas = await client.discover()
        notifications = await client.pull(PullPoint("http://camera.test/pull"))

        self.assertEqual("RuleEngine/CellMotionDetector/Motion", schemas[0].topic)
        self.assertEqual((PropertyField("Rule", "xsd:string"),), schemas[0].source)
        self.assertEqual((PropertyField("IsMotion", "xsd:boolean"),), schemas[0].data)
        self.assertEqual([{
            "name": "IsMotion", "type": "xsd:boolean", "value": True,
            "sources": {"Rule": "rule-1"},
        }], notifications[0]["properties"])

    def test_boolean_property_lexical_values(self):
        self.assertIs(True, OnvifClient._parse_property_value("true", "xsd:boolean"))
        self.assertIs(True, OnvifClient._parse_property_value("1", "xsd:boolean"))
        self.assertIs(False, OnvifClient._parse_property_value("false", "xsd:boolean"))
        self.assertIs(False, OnvifClient._parse_property_value("0", "xsd:boolean"))
        self.assertIsNone(OnvifClient._parse_property_value("secret", "xsd:boolean"))

    async def test_notification_field_diagnostics_have_per_name_and_total_bounds(self):
        fields = "".join(
            f"<SimpleItem Name='Field{number}{'x' * 300}' Value='secret'/>"
            for number in range(32))
        client = StubClient([
            f"<Envelope><NotificationMessage><Message><Data>{fields}</Data></Message>"
            "</NotificationMessage></Envelope>",
        ])

        notifications = await client.pull(PullPoint("http://camera.test/pullpoint"))
        names = notifications[0]["fields"]

        self.assertTrue(all(len(name) <= MAX_DIAGNOSTIC_NAME_LENGTH for name in names))
        self.assertLessEqual(sum(map(len, names)), MAX_DIAGNOSTIC_NAMES_LENGTH)

    async def test_subscription_reference_parameters_are_replayed(self):
        client = StubClient([
            """<Envelope xmlns:wsa='http://www.w3.org/2005/08/addressing'>
              <SubscriptionReference><wsa:Address>http://camera.test/pull</wsa:Address>
                <wsa:ReferenceParameters><Identifier>subscription-1</Identifier>
                </wsa:ReferenceParameters>
              </SubscriptionReference></Envelope>""",
            "<Envelope/>",
        ])

        pullpoint = await client.subscribe("http://camera.test/events")
        await client.pull(pullpoint)

        self.assertEqual("http://camera.test/pull", pullpoint.address)
        self.assertIn(b"subscription-1", pullpoint.reference_parameters[0])
        self.assertEqual(pullpoint.reference_parameters, client.calls[1][4])

    def test_replayed_reference_parameters_are_marked_in_soap_headers(self):
        client = OnvifClient(OnvifConfig(SECRET_ENDPOINT, SECRET_USER, SECRET_PASSWORD))
        parameter = b"<Identifier>subscription-1</Identifier>"

        for action, body in (
                (f"{EVENTS}/PullPointSubscription/PullMessagesRequest",
                 ET.Element(ET.QName(EVENTS, "PullMessages"))),
                (f"{NOTIFY_WSDL}/SubscriptionManager/UnsubscribeRequest",
                 ET.Element("Unsubscribe"))):
            envelope = ET.fromstring(client._envelope(
                action, SECRET_ENDPOINT, body, (parameter,)))
            identifier = next(element for element in envelope.iter()
                              if element.tag == "Identifier")

            self.assertEqual(
                "true", identifier.attrib[f"{{{ADDRESSING}}}IsReferenceParameter"])

    async def test_subscription_tracks_finite_lifetime(self):
        client = StubClient(["""
            <Envelope xmlns:wsa='http://www.w3.org/2005/08/addressing'>
              <SubscriptionReference><wsa:Address>http://camera.test/pull</wsa:Address>
              </SubscriptionReference>
              <CurrentTime>2026-09-16T10:00:00Z</CurrentTime>
              <TerminationTime>2026-09-16T10:02:00Z</TerminationTime>
            </Envelope>"""])
        started = time.monotonic()

        pullpoint = await client.subscribe("http://camera.test/events")

        self.assertIsNotNone(pullpoint.expires_at)
        self.assertAlmostEqual(started + 120, pullpoint.expires_at, delta=1)
        self.assertFalse(pullpoint.expires_within(60))
        self.assertTrue(pullpoint.expires_within(121))

    async def test_pull_timeout_is_limited_to_remaining_subscription_lifetime(self):
        client = StubClient(["<Envelope/>"])
        client.pull_timeout = 30
        client.request_timeout = 10
        pullpoint = PullPoint(
            "http://camera.test/pullpoint", expires_at=time.monotonic() + 2)

        await client.pull(pullpoint)

        timeout = ET.fromstring(client.calls[0][2]).find(f"{{{EVENTS}}}Timeout")
        self.assertIsNotNone(timeout)
        effective_timeout = float(timeout.text.removeprefix("PT").removesuffix("S"))
        self.assertGreater(effective_timeout, 0)
        self.assertLess(effective_timeout, 2)
        self.assertAlmostEqual(effective_timeout + 10, client.calls[0][3], places=5)

    async def test_unsubscribe_uses_ws_base_notification_action(self):
        client = StubClient(["<Envelope/>"])

        await client.unsubscribe(PullPoint("http://camera.test/pull"))

        self.assertEqual(
            f"{NOTIFY_WSDL}/SubscriptionManager/UnsubscribeRequest", client.calls[0][1])

    def test_soap_envelope_uses_password_digest_not_plaintext_password(self):
        client = OnvifClient(OnvifConfig(SECRET_ENDPOINT, SECRET_USER, SECRET_PASSWORD))

        envelope = client._envelope(
            f"{EVENTS}/GetEventProperties", SECRET_ENDPOINT,
            ET.Element(ET.QName(EVENTS, "GetEventProperties")))

        self.assertIn(SECRET_USER.encode(), envelope)
        self.assertNotIn(SECRET_PASSWORD.encode(), envelope)
        self.assertIn(b"PasswordDigest", envelope)

    async def test_http_digest_authentication_is_preserved(self):
        requests = []

        async def handle(request):
            requests.append(request)
            if len(requests) == 1:
                return httpx.Response(401, headers={
                    "WWW-Authenticate": (
                        'Digest realm="camera", nonce="0123456789", '
                        'algorithm=MD5, qop="auth"'),
                }, request=request)
            return httpx.Response(200, content=b"<Envelope/>", request=request)

        client = OnvifClient(
            OnvifConfig(SECRET_ENDPOINT, SECRET_USER, SECRET_PASSWORD),
            client_factory=client_using(httpx.MockTransport(handle)))

        await client._post(
            SECRET_ENDPOINT, f"{EVENTS}/GetEventProperties",
            ET.Element(ET.QName(EVENTS, "GetEventProperties")))

        self.assertEqual(2, len(requests))
        authorization = requests[1].headers["Authorization"]
        self.assertTrue(authorization.startswith("Digest "))
        self.assertIn(f'username="{SECRET_USER}"', authorization)
        self.assertNotIn(SECRET_PASSWORD, authorization)

    def test_discovered_service_cannot_send_credentials_to_another_origin(self):
        client = OnvifClient(OnvifConfig(SECRET_ENDPOINT, SECRET_USER, SECRET_PASSWORD))

        with self.assertRaisesRegex(ValueError, "configured camera origin"):
            client._validate_destination("http://attacker.test/collect")

    def test_discovered_service_cannot_send_credentials_to_another_port(self):
        client = OnvifClient(OnvifConfig(SECRET_ENDPOINT, SECRET_USER, SECRET_PASSWORD))

        with self.assertRaisesRegex(ValueError, "configured camera origin"):
            client._validate_destination("http://camera.test:8080/events")

    async def test_pullpoint_can_use_device_advertised_alternate_port(self):
        requests = []

        async def handle(request):
            requests.append(request)
            content = (
                b"<Envelope xmlns:wsa='http://www.w3.org/2005/08/addressing'>"
                b"<SubscriptionReference><wsa:Address>"
                b"http://camera.test:1024/pullpoint"
                b"</wsa:Address></SubscriptionReference></Envelope>"
                if len(requests) == 1 else b"<Envelope/>"
            )
            return httpx.Response(200, content=content, request=request)

        client = OnvifClient(
            OnvifConfig(
                "http://camera.test:2020/onvif/device_service",
                SECRET_USER, SECRET_PASSWORD),
            client_factory=client_using(httpx.MockTransport(handle)))
        pullpoint = await client.subscribe("http://camera.test:2020/events")

        await client.pull(pullpoint)
        await client.unsubscribe(pullpoint)

        self.assertEqual([2020, 1024, 1024], [request.url.port for request in requests])

    async def test_pullpoint_alternate_port_does_not_allow_another_host(self):
        client = OnvifClient(OnvifConfig(
            "http://camera.test:2020/onvif/device_service",
            SECRET_USER, SECRET_PASSWORD))

        with self.assertRaisesRegex(ValueError, "configured camera origin"):
            await client.pull(PullPoint("http://attacker.test:1024/pullpoint"))

    async def test_unadvertised_same_host_pullpoint_is_not_trusted(self):
        client = OnvifClient(OnvifConfig(
            "http://camera.test:2020/onvif/device_service",
            SECRET_USER, SECRET_PASSWORD))

        with self.assertRaisesRegex(ValueError, "configured camera origin"):
            await client.pull(PullPoint("http://camera.test:1025/not-returned"))

    def test_default_and_explicit_default_ports_are_the_same_origin(self):
        client = OnvifClient(OnvifConfig(
            "https://camera.test:443/onvif/device_service", SECRET_USER, SECRET_PASSWORD))

        client._validate_destination("https://camera.test/events")

    async def test_timed_out_request_is_aborted_before_retry_begins(self):
        transport = BlockingTransport()
        client = OnvifClient(
            OnvifConfig(SECRET_ENDPOINT, SECRET_USER, SECRET_PASSWORD),
            request_timeout=0.1, client_factory=client_using(transport))
        request = ET.Element(ET.QName(EVENTS, "GetEventProperties"))

        with self.assertRaises(TimeoutError):
            await client._post(SECRET_ENDPOINT, "urn:first", request)
        self.assertEqual(0, transport.active)

        transport.started.clear()
        retry = asyncio.create_task(client._post(SECRET_ENDPOINT, "urn:retry", request))
        await transport.started.wait()
        self.assertEqual(1, transport.active)
        retry.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await retry
        self.assertEqual(0, transport.active)
        self.assertEqual(2, transport.aborted)

    async def test_connector_shutdown_aborts_in_flight_request(self):
        transport = BlockingTransport()
        camera = CameraConfig(
            "entry", "Entry", "rtsp://secret@camera.test/live", None,
            OnvifConfig(SECRET_ENDPOINT, SECRET_USER, SECRET_PASSWORD))
        connector = CameraConnector(
            [camera], onvif_client_factory=lambda config, **options: OnvifClient(
                config, client_factory=client_using(transport), **options))
        stop = asyncio.Event()
        task = asyncio.create_task(connector.run(asyncio.Queue(), stop))
        await transport.started.wait()

        stop.set()
        await asyncio.wait_for(task, 0.5)

        self.assertEqual(0, transport.active)
        self.assertEqual(1, transport.aborted)

    async def test_repeated_timeouts_do_not_accumulate_concurrent_requests(self):
        transport = BlockingTransport()
        client = OnvifClient(
            OnvifConfig(SECRET_ENDPOINT, SECRET_USER, SECRET_PASSWORD),
            request_timeout=0.1, client_factory=client_using(transport))
        request = ET.Element(ET.QName(EVENTS, "GetEventProperties"))

        for attempt in range(5):
            with self.assertRaises(TimeoutError):
                await client._post(SECRET_ENDPOINT, f"urn:attempt:{attempt}", request)
            self.assertEqual(0, transport.active)

        self.assertEqual(1, transport.maximum_active)
        self.assertEqual(5, transport.aborted)


class FakeOnvifClient:
    instances = []

    def __init__(self, config, **options):
        self.config = config
        self.options = options
        self.pull_started = asyncio.Event()
        self.unsubscribed = False
        self.__class__.instances.append(self)

    async def discover(self):
        return "internal-event-service", ("RuleEngine/Motion",), ()

    async def subscribe(self, _):
        return PullPoint("internal-pullpoint")

    async def pull(self, _):
        self.pull_started.set()
        await asyncio.Event().wait()

    async def unsubscribe(self, _):
        self.unsubscribed = True


class OnvifConnectorTests(unittest.IsolatedAsyncioTestCase):
    async def test_startup_readiness_aggregates_partial_subscription_failure(self):
        class PartialClient(FakeOnvifClient):
            async def discover(self):
                if "offline" in self.config.endpoint:
                    raise TimeoutError("unreachable")
                return await super().discover()

        cameras = [
            CameraConfig(
                camera_id, camera_id.title(), f"rtsp://secret@{camera_id}.test/live", None,
                OnvifConfig(f"http://{camera_id}.test/onvif", "user", "password"))
            for camera_id in ("online", "offline")
        ]
        connector = CameraConnector(
            cameras, onvif_client_factory=PartialClient, onvif_retry_seconds=10,
            ffmpeg_executable="test-ffmpeg")
        stop, readiness = asyncio.Event(), asyncio.Queue()
        with patch("resident.camera.shutil.which", return_value="/test/ffmpeg"):
            task = asyncio.create_task(connector.run(asyncio.Queue(), stop, readiness))
            camera_result = await readiness.get()
            onvif_result = await readiness.get()
            stop.set()
            await task

        self.assertTrue(camera_result.ok)
        self.assertEqual("2 cameras", camera_result.detail)
        self.assertEqual("onvif", onvif_result.key)
        self.assertFalse(onvif_result.ok)
        self.assertEqual("1/2 subscriptions", onvif_result.detail)

    async def test_property_baseline_duplicates_and_bidirectional_transitions(self):
        stop = asyncio.Event()

        class TransitionClient(FakeOnvifClient):
            values = iter((False, False, True, True, False))

            async def discover(self):
                return (
                    "internal-event-service", ("RuleEngine/Motion",),
                    (PropertySchema(
                        "RuleEngine/Motion",
                        source=(PropertyField("Rule", "xsd:string"),),
                        data=(PropertyField("IsMotion", "xsd:boolean"),)),),
                )

            async def pull(self, _):
                value = next(self.__class__.values)
                if value is False and getattr(self, "pulls", 0) == 4:
                    stop.set()
                self.pulls = getattr(self, "pulls", 0) + 1
                return ({
                    "topic": "RuleEngine/Motion", "fields": ["IsMotion", "Rule"],
                    "properties": [{
                        "name": "IsMotion", "type": "xsd:boolean", "value": value,
                        "sources": {"Rule": "rule-1"},
                    }],
                },)

        TransitionClient.instances.clear()
        TransitionClient.values = iter((False, False, True, True, False))
        camera = CameraConfig(
            "entry", "Entry", "rtsp://secret@camera.test/live", None,
            OnvifConfig(SECRET_ENDPOINT, SECRET_USER, SECRET_PASSWORD))
        diagnostics = []
        connector = CameraConnector(
            [camera], onvif_client_factory=TransitionClient, onvif_retry_seconds=0.01,
            diagnostic_output=diagnostics.append)
        queue = asyncio.Queue()

        await asyncio.wait_for(connector.run(queue, stop), 1)

        events = []
        while not queue.empty():
            events.append(queue.get_nowait())
        self.assertEqual(2, len(events))
        self.assertEqual(
            [(False, True), (True, False)],
            [(event.payload["previous"], event.payload["value"]) for event in events])
        self.assertTrue(all(event.reason == "onvif_property_changed" for event in events))
        self.assertTrue(all(event.payload["topic"] == "RuleEngine/Motion" for event in events))
        self.assertTrue(all(event.payload["sources"] == {"Rule": "rule-1"} for event in events))
        self.assertNotIn("rule-1", "\n".join(diagnostics))

    async def test_camera_refresh_reconciles_only_changed_onvif_workers(self):
        FakeOnvifClient.instances.clear()
        original = {
            camera_id: CameraConfig(
                camera_id, camera_id.title(), f"rtsp://secret@{camera_id}.test/live", None,
                OnvifConfig(f"http://{camera_id}.test/onvif", "user", "password"))
            for camera_id in ("keep", "change", "remove")
        }
        connector = CameraConnector(
            list(original.values()), onvif_client_factory=FakeOnvifClient)
        stop = asyncio.Event()
        task = asyncio.create_task(connector.run(asyncio.Queue(), stop))
        while len(FakeOnvifClient.instances) < 3:
            await asyncio.sleep(0)
        initial = {client.config.endpoint: client for client in FakeOnvifClient.instances}

        connector.replace_cameras([
            CameraConfig(
                "keep", "Renamed", "rtsp://new-secret@keep.test/live", None,
                original["keep"].onvif),
            CameraConfig(
                "change", "Change", original["change"].rtsp_url, None,
                OnvifConfig("http://change.test/onvif", "user", "new-password")),
            CameraConfig(
                "added", "Added", "rtsp://secret@added.test/live", None,
                OnvifConfig("http://added.test/onvif", "user", "password")),
        ])
        while len(FakeOnvifClient.instances) < 5:
            await asyncio.sleep(0)

        self.assertIs(initial["http://keep.test/onvif"], FakeOnvifClient.instances[0])
        self.assertFalse(initial["http://keep.test/onvif"].unsubscribed)
        self.assertTrue(initial["http://change.test/onvif"].unsubscribed)
        self.assertTrue(initial["http://remove.test/onvif"].unsubscribed)
        self.assertEqual(
            ["password", "new-password"],
            [client.config.password for client in FakeOnvifClient.instances
             if client.config.endpoint == "http://change.test/onvif"])
        self.assertEqual(
            1, sum(client.config.endpoint == "http://keep.test/onvif"
                   for client in FakeOnvifClient.instances))

        stop.set()
        await task
        self.assertTrue(all(client.unsubscribed for client in FakeOnvifClient.instances))

    async def test_repeated_unusable_subscriptions_use_retry_backoff(self):
        retry_seconds = 0.1
        stop = asyncio.Event()

        class UnusableSubscriptionClient(FakeOnvifClient):
            subscription_times = []

            async def subscribe(self, _):
                self.__class__.subscription_times.append(time.monotonic())
                if len(self.__class__.subscription_times) == 3:
                    stop.set()
                return PullPoint("internal-pullpoint", expires_at=time.monotonic())

        UnusableSubscriptionClient.instances.clear()
        UnusableSubscriptionClient.subscription_times = []
        camera = CameraConfig(
            "entry", "Entry", "rtsp://secret@camera.test/live", None,
            OnvifConfig(SECRET_ENDPOINT, SECRET_USER, SECRET_PASSWORD))
        connector = CameraConnector(
            [camera], onvif_client_factory=UnusableSubscriptionClient,
            onvif_retry_seconds=retry_seconds)

        await asyncio.wait_for(connector.run(asyncio.Queue(), stop), 1)

        times = UnusableSubscriptionClient.subscription_times
        self.assertEqual(3, len(times))
        self.assertTrue(all(
            later - earlier >= retry_seconds * 0.8
            for earlier, later in zip(times, times[1:])))

    async def test_short_subscription_is_pulled_before_expiry(self):
        stop = asyncio.Event()

        class ExpiringClient(FakeOnvifClient):
            subscriptions = 0
            pulls = 0

            async def subscribe(self, _):
                self.__class__.subscriptions += 1
                return PullPoint("internal-pullpoint", expires_at=time.monotonic() + 1)

            async def pull(self, _):
                self.__class__.pulls += 1
                stop.set()
                return ()

        ExpiringClient.instances.clear()
        ExpiringClient.subscriptions = 0
        ExpiringClient.pulls = 0
        camera = CameraConfig(
            "entry", "Entry", "rtsp://secret@camera.test/live", None,
            OnvifConfig(SECRET_ENDPOINT, SECRET_USER, SECRET_PASSWORD))
        connector = CameraConnector(
            [camera], onvif_client_factory=ExpiringClient,
            onvif_request_timeout_seconds=1, onvif_pull_timeout_seconds=1)

        await connector.run(asyncio.Queue(), stop)

        self.assertEqual(1, ExpiringClient.subscriptions)
        self.assertEqual(1, ExpiringClient.pulls)
        self.assertTrue(all(client.unsubscribed for client in ExpiringClient.instances))

    async def test_probe_is_opt_in_diagnostic_only_and_stops_cleanly(self):
        FakeOnvifClient.instances.clear()
        diagnostics = []
        camera = CameraConfig(
            "entry", "Entry", "rtsp://rtsp-secret@camera.test/live", None,
            OnvifConfig(SECRET_ENDPOINT, SECRET_USER, SECRET_PASSWORD))
        connector = CameraConnector(
            [camera], onvif_client_factory=FakeOnvifClient,
            diagnostic_output=diagnostics.append)
        queue = asyncio.Queue()
        stop = asyncio.Event()

        task = asyncio.create_task(connector.run(queue, stop))
        while not FakeOnvifClient.instances:
            await asyncio.sleep(0)
        client = FakeOnvifClient.instances[0]
        await client.pull_started.wait()
        stop.set()
        await task

        self.assertTrue(queue.empty())
        self.assertTrue(client.unsubscribed)
        output = "\n".join(diagnostics)
        self.assertIn("RuleEngine/Motion", output)
        self.assertNotIn(SECRET_ENDPOINT, output)
        self.assertNotIn(SECRET_USER, output)
        self.assertNotIn(SECRET_PASSWORD, output)

    async def test_failures_retry_without_waking_resident_or_leaking_exception(self):
        class FailingClient(FakeOnvifClient):
            async def discover(self):
                raise RuntimeError(f"failed for {SECRET_ENDPOINT} using {SECRET_PASSWORD}")

        diagnostics = []
        camera = CameraConfig(
            "entry", "Entry", "rtsp://secret@camera.test/live", None,
            OnvifConfig(SECRET_ENDPOINT, SECRET_USER, SECRET_PASSWORD))
        connector = CameraConnector(
            [camera], onvif_client_factory=FailingClient, onvif_retry_seconds=10,
            diagnostic_output=diagnostics.append)
        queue = asyncio.Queue()
        stop = asyncio.Event()

        task = asyncio.create_task(connector.run(queue, stop))
        while not diagnostics:
            await asyncio.sleep(0)
        stop.set()
        await task

        self.assertTrue(queue.empty())
        self.assertNotIn(SECRET_ENDPOINT, diagnostics[0])
        self.assertNotIn(SECRET_PASSWORD, diagnostics[0])

    async def test_client_construction_failure_is_retried(self):
        stop = asyncio.Event()
        attempts = 0

        def factory(config, **options):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ValueError(f"bad endpoint {SECRET_ENDPOINT}")
            stop.set()
            return FakeOnvifClient(config, **options)

        diagnostics = []
        camera = CameraConfig(
            "entry", "Entry", "rtsp://secret@camera.test/live", None,
            OnvifConfig(SECRET_ENDPOINT, SECRET_USER, SECRET_PASSWORD))
        connector = CameraConnector(
            [camera], onvif_client_factory=factory, onvif_retry_seconds=0.01,
            diagnostic_output=diagnostics.append)

        await connector.run(asyncio.Queue(), stop)

        self.assertEqual(2, attempts)
        self.assertNotIn(SECRET_ENDPOINT, "\n".join(diagnostics))


if __name__ == "__main__":
    unittest.main()
