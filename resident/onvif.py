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


@dataclass(frozen=True)
class PropertyField:
    name: str
    type: str


@dataclass(frozen=True)
class PropertySchema:
    topic: str
    source: tuple[PropertyField, ...] = ()
    key: tuple[PropertyField, ...] = ()
    data: tuple[PropertyField, ...] = ()


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _origin(url: str) -> tuple[str, str | None, int | None]:
    parsed = urlsplit(url)
    default_port = {"http": 80, "https": 443}.get(parsed.scheme.lower())
    port = parsed.port if parsed.port is not None else default_port
    return parsed.scheme.lower(), parsed.hostname.lower() if parsed.hostname else None, port


def _canonical_topic(topic: str) -> str:
    return "/".join(part.split(":", 1)[-1] for part in topic.strip().split("/") if part)


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
                 pull_timeout: float = 5.0,
                 client_factory: Callable[..., httpx.AsyncClient] = httpx.AsyncClient):
        self.config = config
        self.request_timeout = request_timeout
        self.pull_timeout = pull_timeout
        self._origin = _origin(config.endpoint)
        self._trusted_pullpoints: set[str] = set()
        self._property_schemas: dict[str, PropertySchema] = {}
        self._client_factory = client_factory

    def _validate_destination(self, destination: str, *,
                              trusted_subscription: bool = False) -> None:
        parsed = urlsplit(destination)
        candidate = _origin(destination)
        origin_matches = (destination in self._trusted_pullpoints
                          if trusted_subscription else candidate == self._origin)
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
                    trusted_subscription: bool = False) -> ET.Element:
        self._validate_destination(
            destination, trusted_subscription=trusted_subscription)
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

    async def discover(self) -> tuple[
            str, tuple[str, ...], tuple[PropertySchema, ...]]:
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
        topic_set = next((element for element in properties.iter()
                          if _local_name(element.tag) == "TopicSet"), None)
        topics = self._topic_paths(topic_set) if topic_set is not None else ()
        schemas = self._property_schema(topic_set) if topic_set is not None else ()
        self._property_schemas = {schema.topic: schema for schema in schemas}
        return xaddr, topics, schemas

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

    @staticmethod
    def _property_schema(topic_set: ET.Element) -> tuple[PropertySchema, ...]:
        schemas: list[PropertySchema] = []

        def fields(description: ET.Element, section_name: str) -> tuple[PropertyField, ...]:
            section = next((child for child in description
                            if _local_name(child.tag) == section_name), None)
            if section is None:
                return ()
            result = []
            for item in section.iter():
                if _local_name(item.tag) != "SimpleItemDescription":
                    continue
                name, value_type = item.attrib.get("Name"), item.attrib.get("Type")
                if name and value_type:
                    result.append(PropertyField(
                        name[:MAX_DIAGNOSTIC_NAME_LENGTH],
                        value_type[:MAX_DIAGNOSTIC_NAME_LENGTH]))
            return tuple(result[:MAX_NOTIFICATION_FIELDS])

        def visit(element: ET.Element, parents: tuple[str, ...]) -> None:
            current = parents + ((_local_name(element.tag),) if element is not topic_set else ())
            if element.attrib.get(f"{{{TOPICS}}}topic", "").lower() == "true":
                description = next((child for child in element
                                    if _local_name(child.tag) == "MessageDescription"), None)
                if (description is not None
                        and description.attrib.get("IsProperty", "").lower() == "true"):
                    schemas.append(PropertySchema(
                        "/".join(current), fields(description, "Source"),
                        fields(description, "Key"), fields(description, "Data")))
            for child in element:
                if isinstance(child.tag, str) and _local_name(child.tag) != "MessageDescription":
                    visit(child, current)

        visit(topic_set, ())
        return tuple(schemas[:MAX_TOPIC_PATHS])

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
        parsed = urlsplit(address)
        if (_origin(address)[:2] != self._origin[:2]
                or parsed.username is not None or parsed.password is not None):
            raise ValueError("ONVIF service address is outside the configured camera origin")
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
        self._trusted_pullpoints.add(address)
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
            trusted_subscription=True)
        summaries = []
        for notification in (element for element in root.iter()
                             if _local_name(element.tag) == "NotificationMessage"):
            raw_topic = next((element.text for element in notification.iter()
                              if _local_name(element.tag) == "Topic" and element.text), "unknown")
            topic = _canonical_topic(raw_topic)
            names = sorted({element.attrib["Name"][:MAX_DIAGNOSTIC_NAME_LENGTH]
                            for element in notification.iter()
                            if _local_name(element.tag) in ("SimpleItem", "ElementItem")
                            and "Name" in element.attrib})
            summary: dict[str, object] = {
                "topic": topic[:MAX_DIAGNOSTIC_NAME_LENGTH],
                "fields": list(_bounded_names(names, MAX_NOTIFICATION_FIELDS)),
            }
            schema = self._property_schemas.get(topic)
            if schema is not None:
                source_values = self._section_values(notification, "Source")
                data_values = self._section_values(notification, "Data")
                sources = {
                    field.name: source_values[field.name][:MAX_DIAGNOSTIC_NAMES_LENGTH]
                    for field in schema.source if field.name in source_values
                }
                parsed_properties = []
                for field in schema.data:
                    raw_value = data_values.get(field.name)
                    if raw_value is None:
                        continue
                    value = self._parse_property_value(raw_value, field.type)
                    if value is not None:
                        parsed_properties.append({
                            "name": field.name, "type": field.type, "value": value,
                            "sources": sources,
                        })
                summary["properties"] = parsed_properties
            summaries.append(summary)
        return tuple(summaries[:32])

    @staticmethod
    def _section_values(notification: ET.Element, section_name: str) -> dict[str, str]:
        section = next((element for element in notification.iter()
                        if _local_name(element.tag) == section_name), None)
        if section is None:
            return {}
        return {
            item.attrib["Name"]: item.attrib["Value"]
            for item in section.iter()
            if _local_name(item.tag) == "SimpleItem"
            and "Name" in item.attrib and "Value" in item.attrib
        }

    @staticmethod
    def _parse_property_value(value: str, declared_type: str) -> object | None:
        value_type = declared_type.split(":", 1)[-1].lower()
        if value_type == "boolean":
            normalized = value.strip().lower()
            if normalized in ("true", "1"):
                return True
            if normalized in ("false", "0"):
                return False
            return None
        if value_type in ("byte", "short", "int", "integer", "long",
                          "unsignedbyte", "unsignedshort", "unsignedint", "unsignedlong"):
            try:
                return int(value)
            except ValueError:
                return None
        if value_type in ("decimal", "double", "float"):
            try:
                return float(value)
            except ValueError:
                return None
        if value_type in ("string", "referencetoken"):
            return value[:MAX_DIAGNOSTIC_NAMES_LENGTH]
        return None

    async def unsubscribe(self, pullpoint: PullPoint) -> None:
        try:
            await self._post(
                pullpoint.address, f"{NOTIFY_WSDL}/SubscriptionManager/UnsubscribeRequest",
                ET.Element(ET.QName(NOTIFY, "Unsubscribe")),
                reference_parameters=pullpoint.reference_parameters,
                trusted_subscription=True)
        finally:
            self._trusted_pullpoints.discard(pullpoint.address)
