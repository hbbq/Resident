from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Sequence


OutputHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any] | None]]


@dataclass(frozen=True)
class DeliveryPolicy:
    max_attempts: int = 3
    retry_delays_seconds: tuple[float, ...] = (1.0, 5.0)


@dataclass(frozen=True)
class OutputCapability:
    """A terminal side effect, distinct from an interactive model tool."""

    output_type: str
    description: str
    payload_schema: dict[str, Any]
    route_identity: str
    handler: OutputHandler
    target: str | None = None
    delivery_policy: DeliveryPolicy = DeliveryPolicy()
    legacy_tool_name: str | None = None

    def semantic_descriptor(self) -> dict[str, Any]:
        descriptor = {
            "type": self.output_type,
            "target": self.target,
            "description": self.description,
            "payload_schema": self.payload_schema,
        }
        return json.loads(json.dumps(descriptor, sort_keys=True, separators=(",", ":")))

    @property
    def fingerprint(self) -> str:
        document = {
            "semantic": self.semantic_descriptor(),
            "route_identity": self.route_identity,
            "delivery_policy": {
                "max_attempts": self.delivery_policy.max_attempts,
                "retry_delays_seconds": self.delivery_policy.retry_delays_seconds,
            },
        }
        return hashlib.sha256(json.dumps(
            document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def schema_branch(self) -> dict[str, Any]:
        properties: dict[str, Any] = {
            "type": {"type": "string", "enum": [self.output_type]},
        }
        required = ["type"]
        if self.target is not None:
            properties["target"] = {"type": "string", "enum": [self.target]}
            required.append("target")
        payload_properties = self.payload_schema.get("properties", {})
        properties.update(payload_properties)
        required.extend(self.payload_schema.get("required", []))
        return {
            "type": "object",
            "description": self.description,
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }


def output_schema(capabilities: Sequence[OutputCapability]) -> dict[str, Any]:
    branches = [capability.schema_branch() for capability in capabilities]
    max_items = 8
    if not branches:
        # Keep the Managed Agents supported nested-anyOf shape while making the
        # empty capability set authorize only outputs: [].
        branches = [{
            "type": "object", "properties": {}, "required": [],
            "additionalProperties": False,
        }]
        max_items = 0
    return {
        "type": "object",
        "properties": {
            "outputs": {
                "type": "array",
                "items": {"anyOf": branches},
                "maxItems": max_items,
            },
        },
        "required": ["outputs"],
        "additionalProperties": False,
    }


def schema_fingerprint(schema: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        schema, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_disposition(value: Any, schema: dict[str, Any]) -> str | None:
    if not isinstance(value, dict):
        return "Final disposition must be an object"
    if set(value) != {"outputs"}:
        return "Final disposition must contain only the required outputs property"
    outputs = value["outputs"]
    if not isinstance(outputs, list):
        return "Final disposition outputs must be an array"
    if len(outputs) > schema["properties"]["outputs"]["maxItems"]:
        return "Final disposition has too many outputs"
    branches = schema["properties"]["outputs"]["items"]["anyOf"]
    for ordinal, output in enumerate(outputs):
        errors = [_validate_object(output, branch) for branch in branches]
        if not any(error is None for error in errors):
            return f"Output {ordinal} does not match an authorized output schema"
    return None


def _validate_object(value: Any, schema: dict[str, Any]) -> str | None:
    if not isinstance(value, dict):
        return "not an object"
    properties = schema.get("properties", {})
    if set(value) - set(properties):
        return "additional property"
    if set(schema.get("required", [])) - set(value):
        return "missing property"
    for key, item in value.items():
        constraint = properties[key]
        expected = constraint.get("type")
        if expected == "string" and not isinstance(item, str):
            return "wrong type"
        if expected == "integer" and (not isinstance(item, int) or isinstance(item, bool)):
            return "wrong type"
        if "enum" in constraint and item not in constraint["enum"]:
            return "enum mismatch"
        if isinstance(item, str):
            if len(item) > constraint.get("maxLength", len(item)):
                return "string too long"
            if len(item) < constraint.get("minLength", 0):
                return "string too short"
    return None


def capability_for_output(
        output: dict[str, Any], capabilities: Sequence[OutputCapability],
) -> OutputCapability | None:
    for capability in capabilities:
        if (capability.output_type == output.get("type")
                and capability.target == output.get("target")):
            return capability
    return None
