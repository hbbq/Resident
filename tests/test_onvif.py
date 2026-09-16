from __future__ import annotations

import asyncio
import json
import time
import unittest
import xml.etree.ElementTree as ET

from resident.camera import CameraConnector
from resident.config import CameraConfig, OnvifConfig
from resident.onvif import (
    ADDRESSING, EVENTS, MAX_DIAGNOSTIC_NAMES_LENGTH, MAX_DIAGNOSTIC_NAME_LENGTH,
    NOTIFY_WSDL, TOPICS, OnvifClient, PullPoint,
)


SECRET_ENDPOINT = "http://camera.test/onvif/device_service"
SECRET_USER = "onvif-user"
SECRET_PASSWORD = "onvif-password"


class StubClient(OnvifClient):
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls = []
        self.config = OnvifConfig(SECRET_ENDPOINT, SECRET_USER, SECRET_PASSWORD)
        self.request_timeout = 1
        self.pull_timeout = 2

    async def _post(self, destination, action, body, timeout=None, reference_parameters=()):
        self.calls.append((destination, action, ET.tostring(body), timeout, reference_parameters))
        return ET.fromstring(self.responses.pop(0))


class OnvifClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_discovers_event_service_and_advertised_topics(self):
        client = StubClient([
            """<Envelope><Events><XAddr>http://camera.test/events</XAddr></Events></Envelope>""",
            """<Envelope xmlns:wstop='http://docs.oasis-open.org/wsn/t-1'>
                 <TopicSet><RuleEngine><CellMotionDetector wstop:topic='true'>
                   <MessageDescription><Data><SimpleItemDescription Name='IsMotion'/></Data>
                   </MessageDescription>
                 </CellMotionDetector></RuleEngine></TopicSet>
               </Envelope>""",
        ])

        service, topics = await client.discover()

        self.assertEqual("http://camera.test/events", service)
        self.assertEqual(("RuleEngine/CellMotionDetector",), topics)
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
            "topic": "tns1:RuleEngine/Motion", "fields": ["IsMotion"],
        },), notifications)
        self.assertNotIn("top-secret-value", json.dumps(notifications))
        self.assertIn(b"PT2S", client.calls[0][2])

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

    def test_discovered_service_cannot_send_credentials_to_another_origin(self):
        client = OnvifClient(OnvifConfig(SECRET_ENDPOINT, SECRET_USER, SECRET_PASSWORD))

        with self.assertRaisesRegex(ValueError, "configured camera origin"):
            client._validate_destination("http://attacker.test/collect")

    def test_discovered_service_cannot_send_credentials_to_another_port(self):
        client = OnvifClient(OnvifConfig(SECRET_ENDPOINT, SECRET_USER, SECRET_PASSWORD))

        with self.assertRaisesRegex(ValueError, "configured camera origin"):
            client._validate_destination("http://camera.test:8080/events")

    def test_default_and_explicit_default_ports_are_the_same_origin(self):
        client = OnvifClient(OnvifConfig(
            "https://camera.test:443/onvif/device_service", SECRET_USER, SECRET_PASSWORD))

        client._validate_destination("https://camera.test/events")


class FakeOnvifClient:
    instances = []

    def __init__(self, config, **options):
        self.config = config
        self.options = options
        self.pull_started = asyncio.Event()
        self.unsubscribed = False
        self.__class__.instances.append(self)

    async def discover(self):
        return "internal-event-service", ("RuleEngine/Motion",)

    async def subscribe(self, _):
        return PullPoint("internal-pullpoint")

    async def pull(self, _):
        self.pull_started.set()
        await asyncio.Event().wait()

    async def unsubscribe(self, _):
        self.unsubscribed = True


class OnvifConnectorTests(unittest.IsolatedAsyncioTestCase):
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
