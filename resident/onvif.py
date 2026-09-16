from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import os
import time
from typing import Callable, Iterable
import uuid
from urllib.parse import urlsplit
import xml.etree.ElementTree as ET

import httpx

from .config import OnvifConfig


SOAP = "http://www.w3.org/2003/05/soap-envelope"
ADDRESSING = "http://www.w3.org/2005/08/addressing"
WSSE = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd"
WSU = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd"
USERNAME_TOKEN = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0"
SOAP_SECURITY = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-soap-message-security-1.0"
DEVICE = "http://www.onvif.org/ver10/device/wsdl"
EVENTS = "http://www.onvif.org/ver10/events/wsdl"
TOPICS = "http://docs.oasis-open.org/wsn/t-1"
NOTIFY = "http://docs.oasis-open.org/wsn/b-2"
NOTIFY_WSDL = "http://docs.oasis-open.org/wsn/bw-2"
MAX_TOPIC_PATHS = 100
MAX_NOTIFICATION_FIELDS = 32
MAX_DIAGNOSTIC_NAME_LENGTH = 256
MAX_DIAGNOSTIC_NAMES_LENGTH = 4_096
SUBSCRIPTION_SAFETY_MARGIN = 0.1


@dataclass(frozen=True)
class PullPoint:
    address: str
    reference_parameters: tuple[bytes, ...] = field(default=(), repr=False)
    expires_at: float | None = field(default=None, repr=False)

    def expires_within(self, seconds: float) -> bool:
        return self.expires_at is not None and self.expires_at <= time.monotonic() + seconds

    def remaining_lifetime(self) -> float | None:
        if self.expires_at is None:
            return None
        return max(0.0, self.expires_at - time.monotonic())


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _origin(url: str) -> tuple[str, str | None, int | None]:
    parsed = urlsplit(url)
    default_port = {"http": 80, "https": 443}.get(parsed.scheme.lower())
    port = parsed.port if parsed.port is not None else default_port
    return parsed.scheme.lower(), parsed.hostname.lower() if parsed.hostname else None, port


def _bounded_names(names: Iterable[str], count: int) -> tuple[str, ...]:
    bounded: list[str] = []
    remaining = MAX_DIAGNOSTIC_NAMES_LENGTH
    for name in names:
        if len(bounded) >= count or remaining <= 0:
            break
        candidate = name[:min(MAX_DIAGNOSTIC_NAME_LENGTH, remaining)]
        if not candidate or candidate in bounded:
            continue
        bounded.append(candidate)
        remaining -= len(candidate)
    return tuple(bounded)


class OnvifClient:
    """Small ONVIF event probe; raw XML and credentials never leave this boundary."""

    def __init__(self, config: OnvifConfig, *, request_timeout: float = 10.0,
                 pull_timeout: float = 30.0,
                 client_factory: Callable[..., httpx.AsyncClient] = httpx.AsyncClient):
        self.config = config
        self.request_timeout = request_timeout
        self.pull_timeout = pull_timeout
        self._origin = _origin(config.endpoint)
        self._client_factory = client_factory

    def _validate_destination(self, destination: str, *,
                              allow_alternate_port: bool = False) -> None:
        parsed = urlsplit(destination)
        candidate = _origin(destination)
        origin_matches = (
            candidate[:2] == self._origin[:2]
            if allow_alternate_port else candidate == self._origin
        )
        if not origin_matches or parsed.username is not None or parsed.password is not None:
            raise ValueError("ONVIF service address is outside the configured camera origin")

    def _envelope(self, action: str, destination: str, body: ET.Element,
                  reference_parameters: tuple[bytes, ...] = ()) -> bytes:
        envelope = ET.Element(ET.QName(SOAP, "Envelope"))
        header = ET.SubElement(envelope, ET.QName(SOAP, "Header"))
        ET.SubElement(header, ET.QName(ADDRESSING, "Action")).text = action
        ET.SubElement(header, ET.QName(ADDRESSING, "MessageID")).text = f"urn:uuid:{uuid.uuid4()}"
        reply_to = ET.SubElement(header, ET.QName(ADDRESSING, "ReplyTo"))
        ET.SubElement(reply_to, ET.QName(ADDRESSING, "Address")).text = (
            "http://www.w3.org/2005/08/addressing/anonymous")
        ET.SubElement(header, ET.QName(ADDRESSING, "To")).text = destination
        for parameter in reference_parameters:
            element = ET.fromstring(parameter)
            element.set(ET.QName(ADDRESSING, "IsReferenceParameter"), "true")
            header.append(element)
        security = ET.SubElement(header, ET.QName(WSSE, "Security"), {
            ET.QName(SOAP, "mustUnderstand"): "true",
        })
        token = ET.SubElement(security, ET.QName(WSSE, "UsernameToken"))
        ET.SubElement(token, ET.QName(WSSE, "Username")).text = self.config.username
        nonce = os.urandom(16)
        created = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        digest = hashlib.sha1(nonce + created.encode() + self.config.password.encode()).digest()
        password = ET.SubElement(token, ET.QName(WSSE, "Password"), {
            "Type": f"{USERNAME_TOKEN}#PasswordDigest",
        })
        password.text = base64.b64encode(digest).decode()
        ET.SubElement(token, ET.QName(WSSE, "Nonce"), {
            "EncodingType": f"{SOAP_SECURITY}#Base64Binary",
        }).text = base64.b64encode(nonce).decode()
        ET.SubElement(token, ET.QName(WSU, "Created")).text = created
        ET.SubElement(envelope, ET.QName(SOAP, "Body")).append(body)
        return ET.tostring(envelope, encoding="utf-8", xml_declaration=True)

    async def _post(self, destination: str, action: str, body: ET.Element,
                    timeout: float | None = None,
                    reference_parameters: tuple[bytes, ...] = (), *,
                    allow_alternate_port: bool = False) -> ET.Element:
        self._validate_destination(
            destination, allow_alternate_port=allow_alternate_port)
        payload = self._envelope(action, destination, body, reference_parameters)
        request_deadline = timeout or self.request_timeout
        async with asyncio.timeout(max(0.1, request_deadline)):
            async with self._client_factory(
                    auth=httpx.DigestAuth(self.config.username, self.config.password),
                    follow_redirects=False, timeout=None, trust_env=False) as client:
                async with client.stream(
                        "POST", destination, content=payload, headers={
                            "Content-Type": (
                                f'application/soap+xml; charset=utf-8; action="{action}"'),
                        }) as response:
                    response.raise_for_status()
                    document = bytearray()
                    async for chunk in response.aiter_bytes():
                        document.extend(chunk)
                        if len(document) > 2_000_000:
                            raise ValueError("ONVIF response exceeded size limit")
        return ET.fromstring(document)

    async def discover(self) -> tuple[str, tuple[str, ...]]:
        request = ET.Element(ET.QName(DEVICE, "GetCapabilities"))
        ET.SubElement(request, ET.QName(DEVICE, "Category")).text = "Events"
        root = await self._post(
            self.config.endpoint, f"{DEVICE}/GetCapabilities", request)
        event_service = next((element for element in root.iter() if _local_name(element.tag) == "Events"), None)
        xaddr = next((element.text for element in event_service.iter()
                      if _local_name(element.tag) == "XAddr" and element.text), None) if event_service is not None else None
        if not xaddr:
            raise ValueError("ONVIF camera did not advertise an event service")
        properties = await self._post(
            xaddr, f"{EVENTS}/EventPortType/GetEventPropertiesRequest",
            ET.Element(ET.QName(EVENTS, "GetEventProperties")))


        for desc in properties.iter():
            if _local_name(desc.tag) != "MessageDescription":
                continue

            print("DEBUG MessageDescription:",
                "IsProperty=", desc.attrib.get("IsProperty"))

            for section in desc:
                section_name = _local_name(section.tag)
                if section_name not in ("Source", "Key", "Data"):
                    continue

                print(f"  {section_name}:")
                for item in section.iter():
                    if _local_name(item.tag) == "SimpleItemDescription":
                        print(
                            "   ",
                            item.attrib.get("Name"),
                            "->",
                            item.attrib.get("Type"),
                        )


        
        topic_set = next((element for element in properties.iter()
                          if _local_name(element.tag) == "TopicSet"), None)
        topics = self._topic_paths(topic_set) if topic_set is not None else ()
        return xaddr, topics

    @staticmethod
    def _topic_paths(topic_set: ET.Element) -> tuple[str, ...]:
        paths: list[str] = []

        def visit(element: ET.Element, parents: tuple[str, ...]) -> None:
            children = [child for child in element if isinstance(child.tag, str)]
            current = parents + ((_local_name(element.tag),) if element is not topic_set else ())
            if element.attrib.get(f"{{{TOPICS}}}topic", "").lower() == "true":
                paths.append("/".join(current))
            for child in children:
                visit(child, current)

        visit(topic_set, ())
        return _bounded_names(dict.fromkeys(paths), MAX_TOPIC_PATHS)

    async def subscribe(self, event_service: str) -> PullPoint:
        root = await self._post(
            event_service, f"{EVENTS}/EventPortType/CreatePullPointSubscriptionRequest",
            ET.Element(ET.QName(EVENTS, "CreatePullPointSubscription")))
        reference = next((element for element in root.iter()
                          if _local_name(element.tag) == "SubscriptionReference"), None)
        address = next((element.text for element in reference.iter()
                        if _local_name(element.tag) == "Address" and element.text), None) if reference is not None else None
        if not address:
            raise ValueError("ONVIF subscription response did not contain an address")
        parameters = next((element for element in reference.iter()
                           if _local_name(element.tag) == "ReferenceParameters"), None)
        serialized = tuple(ET.tostring(child) for child in parameters)[:32] if parameters is not None else ()
        current_text = next((element.text for element in root.iter()
                             if _local_name(element.tag) == "CurrentTime" and element.text), None)
        termination_text = next((element.text for element in root.iter()
                                 if _local_name(element.tag) == "TerminationTime" and element.text), None)
        expires_at = None
        if termination_text is not None:
            termination = self._parse_datetime(termination_text)
            current = self._parse_datetime(current_text) if current_text is not None else datetime.now(timezone.utc)
            expires_at = time.monotonic() + max(0.0, (termination - current).total_seconds())
            parsed = urlsplit(address)
            print("DEBUG PullPoint scheme:", parsed.scheme)
            print("DEBUG PullPoint host:", parsed.hostname)
            print("DEBUG PullPoint port:", parsed.port)
            print("DEBUG PullPoint path:", parsed.path)
            print("DEBUG PullPoint query:", parsed.query)
            print(
                "DEBUG reference parameters:",
                [_local_name(ET.fromstring(p).tag) for p in serialized]
            )
        return PullPoint(address, serialized, expires_at)

    @staticmethod
    def _parse_datetime(value: str) -> datetime:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    async def pull(self, pullpoint: PullPoint) -> tuple[dict[str, object], ...]:
        pull_timeout = self.pull_timeout
        remaining = pullpoint.remaining_lifetime()
        if remaining is not None:
            pull_timeout = min(pull_timeout, remaining - SUBSCRIPTION_SAFETY_MARGIN)
            if pull_timeout <= 0:
                raise TimeoutError("ONVIF subscription expired before PullMessages")
        request = ET.Element(ET.QName(EVENTS, "PullMessages"))
        ET.SubElement(request, ET.QName(EVENTS, "Timeout")).text = f"PT{pull_timeout:g}S"
        ET.SubElement(request, ET.QName(EVENTS, "MessageLimit")).text = "32"
        root = await self._post(
            pullpoint.address, f"{EVENTS}/PullPointSubscription/PullMessagesRequest", request,
            pull_timeout + self.request_timeout, pullpoint.reference_parameters,
            allow_alternate_port=True)
        summaries = []
        for notification in (element for element in root.iter()
                             if _local_name(element.tag) == "NotificationMessage"):
            topic = next((element.text for element in notification.iter()
                          if _local_name(element.tag) == "Topic" and element.text), "unknown")
            names = sorted({element.attrib["Name"][:MAX_DIAGNOSTIC_NAME_LENGTH]
                            for element in notification.iter()
                            if _local_name(element.tag) in ("SimpleItem", "ElementItem")
                            and "Name" in element.attrib})
            summaries.append({
                "topic": topic[:MAX_DIAGNOSTIC_NAME_LENGTH],
                "fields": list(_bounded_names(names, MAX_NOTIFICATION_FIELDS)),
            })
        return tuple(summaries[:32])

    async def unsubscribe(self, pullpoint: PullPoint) -> None:
        await self._post(
            pullpoint.address, f"{NOTIFY_WSDL}/SubscriptionManager/UnsubscribeRequest",
            ET.Element(ET.QName(NOTIFY, "Unsubscribe")),
            reference_parameters=pullpoint.reference_parameters,
            allow_alternate_port=True)
