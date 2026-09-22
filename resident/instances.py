from __future__ import annotations

import json
import math
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .config import SUPPORTED_REASONING_EFFORTS


_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}\Z")
_ENV = re.compile(r"[A-Z_][A-Z0-9_]*\Z")
_EXTERNAL_OPERATION = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_SUBSCRIPTION = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)?\Z")
_SUBSCRIPTION_EVENTS = frozenset({
    "agentcontroller.workflow_changed",
    "camera.cameras_changed",
    "camera.onvif_property_changed",
    "homeops.measurement_changed",
    "messaging.message_received",
})
_SUBSCRIPTION_SELECTORS = (_SUBSCRIPTION_EVENTS |
                           {item.split(".", 1)[0] for item in _SUBSCRIPTION_EVENTS} |
                           {"*"})
_ALLOWED = {
    "version", "id", "name", "enabled", "personality", "personality_prompt",
    "role", "role_prompt", "agent", "curator", "capabilities", "outputs",
    "subscriptions", "owner_transport", "external_applications", "body",
}
_SECRET_WORDS = ("token", "password", "api_key", "secret", "credential")


@dataclass(frozen=True)
class AgentDefinition:
    provider: str = "openai-agents"
    model: str = "gpt-5.6-luna"
    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str = "OPENAI_BASE_URL"
    agent_id_env: str | None = None
    reasoning_effort: str | None = None
    service_tier: str | None = None


@dataclass(frozen=True)
class CuratorDefinition:
    model: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str = "OPENAI_BASE_URL"
    batch_size: int = 50
    max_batches: int = 4


@dataclass(frozen=True)
class OwnerTransportDefinition:
    type: str
    token_env: str
    owner_user_id_env: str
    owner_chat_id_env: str


@dataclass(frozen=True)
class ExternalOperationDefinition:
    name: str
    operation: str
    description: str
    input_schema: dict[str, Any]
    mutating: bool = False


@dataclass(frozen=True)
class ExternalApplicationDefinition:
    id: str
    description: str
    base_url: str
    bearer_token_env: str | None = None
    request_timeout_seconds: float = 10.0
    bindings: dict[str, Any] = field(default_factory=dict)
    operations: tuple[ExternalOperationDefinition, ...] = ()


@dataclass(frozen=True)
class ResidentDefinition:
    id: str
    name: str
    personality: str
    role: str = ""
    enabled: bool = True
    agent: AgentDefinition = field(default_factory=AgentDefinition)
    curator: CuratorDefinition | None = None
    capabilities: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    subscriptions: tuple[str, ...] = ()
    owner_transport: OwnerTransportDefinition | None = None
    external_applications: tuple[ExternalApplicationDefinition, ...] = ()
    body: dict[str, Any] | None = None


@dataclass(frozen=True)
class ResidentCatalog:
    residents: tuple[ResidentDefinition, ...]
    default_id: str

    def by_id(self) -> dict[str, ResidentDefinition]:
        return {resident.id: resident for resident in self.residents}


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a mapping")
    return value


def _string(value: Any, label: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"{label} must be a nonempty string")
    return value.strip()


def _string_list(value: Any, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{label} must be a list of nonempty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} contains duplicates")
    return tuple(value)


def _reasoning_effort(value: Any, label: str) -> str:
    effort = _string(value, label)
    if effort not in SUPPORTED_REASONING_EFFORTS:
        allowed = ", ".join(SUPPORTED_REASONING_EFFORTS)
        raise ValueError(f"{label} must be one of: {allowed}")
    return effort


def _positive_integer(value: Any, label: str, *, maximum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} must be at most {maximum}")
    return value


def _positive_number(value: Any, label: str, *, maximum: float | None = None) -> float:
    if (not isinstance(value, (int, float)) or isinstance(value, bool) or
            (isinstance(value, float) and not math.isfinite(value)) or value <= 0):
        raise ValueError(f"{label} must be a positive number")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} must be at most {maximum:g}")
    return float(value)


def _external_schema(value: Any, label: str, *, root: bool = True) -> dict[str, Any]:
    schema = _mapping(value, label)
    allowed = {
        "type", "description", "enum", "properties", "required", "additionalProperties",
        "items", "minimum", "maximum", "minLength", "maxLength", "minItems", "maxItems",
    }
    unknown = set(schema) - allowed
    if unknown:
        raise ValueError(f"Unsupported JSON Schema fields in {label}: {', '.join(sorted(unknown))}")
    raw_types = schema.get("type")
    types = [raw_types] if isinstance(raw_types, str) else raw_types
    supported = {"object", "array", "string", "integer", "number", "boolean", "null"}
    if (not isinstance(types, list) or not types or
            not all(isinstance(item, str) and item in supported for item in types) or
            len(types) != len(set(types))):
        raise ValueError(f"{label}.type must contain supported unique JSON types")
    if root and types != ["object"]:
        raise ValueError(f"{label} must describe an object")
    if "description" in schema:
        _string(schema["description"], f"{label}.description")
    if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]):
        raise ValueError(f"{label}.enum must be a nonempty list")
    if "object" in types:
        properties = _mapping(schema.get("properties", {}), f"{label}.properties")
        if schema.get("additionalProperties") is not False:
            raise ValueError(f"{label}.additionalProperties must be false")
        for name, child in properties.items():
            if not name:
                raise ValueError(f"{label}.properties keys must be nonempty")
            _external_schema(child, f"{label}.properties.{name}", root=False)
        required = schema.get("required", [])
        if (not isinstance(required, list) or
                not all(isinstance(item, str) for item in required) or
                len(required) != len(set(required)) or not set(required) <= set(properties)):
            raise ValueError(f"{label}.required must contain unique property names")
    if "array" in types:
        if "items" not in schema:
            raise ValueError(f"{label}.items is required for arrays")
        _external_schema(schema["items"], f"{label}.items", root=False)
    if ({"properties", "required", "additionalProperties"} & set(schema) and
            "object" not in types):
        raise ValueError(f"{label} uses object constraints without object type")
    if "items" in schema and "array" not in types:
        raise ValueError(f"{label} uses items without array type")
    if ({"minimum", "maximum"} & set(schema) and
            not ({"integer", "number"} & set(types))):
        raise ValueError(f"{label} uses numeric constraints without a numeric type")
    if ({"minLength", "maxLength"} & set(schema) and "string" not in types):
        raise ValueError(f"{label} uses string constraints without string type")
    if ({"minItems", "maxItems"} & set(schema) and "array" not in types):
        raise ValueError(f"{label} uses array constraints without array type")
    for key in ("minimum", "maximum"):
        if key in schema and (not isinstance(schema[key], (int, float)) or
                              isinstance(schema[key], bool)):
            raise ValueError(f"{label}.{key} must be a number")
    for key in ("minLength", "maxLength", "minItems", "maxItems"):
        if key in schema and (not isinstance(schema[key], (int, float)) or
                              isinstance(schema[key], bool) or schema[key] < 0 or
                              int(schema[key]) != schema[key]):
            raise ValueError(f"{label}.{key} must be a nonnegative integer")
    try:
        encoded = json.dumps(
            schema, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain JSON values") from exc
    if root and len(encoded.encode("utf-8")) > 65_536:
        raise ValueError(f"{label} is too large")
    return schema


def _external_applications(value: Any, label: str) -> tuple[ExternalApplicationDefinition, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    if len(value) > 16:
        raise ValueError(f"{label} must contain at most 16 providers")
    applications: list[ExternalApplicationDefinition] = []
    for index, raw in enumerate(value):
        item_label = f"{label}[{index}]"
        item = _mapping(raw, item_label)
        allowed = {"id", "description", "base_url", "bearer_token_env",
                   "request_timeout_seconds", "bindings", "operations"}
        if set(item) - allowed:
            raise ValueError(f"Unknown fields in {item_label}: {', '.join(sorted(set(item) - allowed))}")
        application_id = _string(item.get("id"), f"{item_label}.id")
        if not _ID.fullmatch(application_id):
            raise ValueError(f"{item_label}.id must be a lowercase safe identifier")
        if application_id == "messaging":
            raise ValueError(f"{item_label}.id is reserved for built-in messaging")
        bindings = _mapping(item.get("bindings", {}), f"{item_label}.bindings")
        try:
            encoded_bindings = json.dumps(
                bindings, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{item_label}.bindings must contain JSON values") from exc
        if len(encoded_bindings.encode("utf-8")) > 16_384:
            raise ValueError(f"{item_label}.bindings is too large")
        operations_raw = item.get("operations")
        if not isinstance(operations_raw, list) or not operations_raw:
            raise ValueError(f"{item_label}.operations must be a nonempty list")
        if len(operations_raw) > 64:
            raise ValueError(f"{item_label}.operations must contain at most 64 operations")
        operations: list[ExternalOperationDefinition] = []
        for operation_index, raw_operation in enumerate(operations_raw):
            operation_label = f"{item_label}.operations[{operation_index}]"
            operation = _mapping(raw_operation, operation_label)
            operation_allowed = {"name", "operation", "description", "input_schema", "mutating"}
            if set(operation) - operation_allowed:
                raise ValueError(
                    f"Unknown fields in {operation_label}: "
                    f"{', '.join(sorted(set(operation) - operation_allowed))}")
            name = _string(operation.get("name"), f"{operation_label}.name")
            if (not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", name) or
                    not name.startswith(f"{application_id}_")):
                raise ValueError(
                    f"{operation_label}.name must be a namespaced tool name beginning "
                    f"with {application_id}_")
            mutating = operation.get("mutating", False)
            if not isinstance(mutating, bool):
                raise ValueError(f"{operation_label}.mutating must be boolean")
            remote_operation = _string(
                operation.get("operation", name), f"{operation_label}.operation")
            if not _EXTERNAL_OPERATION.fullmatch(remote_operation):
                raise ValueError(f"{operation_label}.operation must be a safe operation identifier")
            description = _string(
                operation.get("description"), f"{operation_label}.description")
            if len(description) > 1024:
                raise ValueError(f"{operation_label}.description must be at most 1024 characters")
            operations.append(ExternalOperationDefinition(
                name=name,
                operation=remote_operation,
                description=description,
                input_schema=_external_schema(
                    operation.get("input_schema"), f"{operation_label}.input_schema"),
                mutating=mutating,
            ))
        names = [operation.name for operation in operations]
        if len(names) != len(set(names)):
            raise ValueError(f"{item_label}.operations contains duplicate tool names")
        application_description = _string(
            item.get("description"), f"{item_label}.description")
        if len(application_description) > 1024:
            raise ValueError(f"{item_label}.description must be at most 1024 characters")
        base_url = _string(item.get("base_url"), f"{item_label}.base_url")
        if len(base_url) > 2048:
            raise ValueError(f"{item_label}.base_url must be at most 2048 characters")
        applications.append(ExternalApplicationDefinition(
            id=application_id,
            description=application_description,
            base_url=base_url,
            bearer_token_env=_env_name(
                item.get("bearer_token_env"), f"{item_label}.bearer_token_env", required=False),
            request_timeout_seconds=_positive_number(
                item.get("request_timeout_seconds", 10),
                f"{item_label}.request_timeout_seconds", maximum=120),
            bindings=bindings,
            operations=tuple(operations),
        ))
    ids = [application.id for application in applications]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label} contains duplicate provider ids")
    return tuple(applications)


def _subscriptions(value: Any, label: str) -> tuple[str, ...]:
    selectors = _string_list(value, label)
    malformed = sorted(item for item in selectors
                       if item != "*" and not _SUBSCRIPTION.fullmatch(item))
    if malformed:
        raise ValueError(f"Malformed subscription selectors: {', '.join(malformed)}")
    unknown = sorted(set(selectors) - _SUBSCRIPTION_SELECTORS)
    if unknown:
        raise ValueError(f"Unknown subscription selectors: {', '.join(unknown)}")
    return selectors


def _prompt(root: Path, reference: Any, label: str) -> str:
    reference = _string(reference, label)
    candidate = (root / reference).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must remain below the prompt root") from exc
    if not candidate.is_file():
        raise ValueError(f"{label} does not exist: {reference}")
    return candidate.read_text(encoding="utf-8").strip()


def _env_name(value: Any, label: str, *, required: bool = True) -> str | None:
    if value is None and not required:
        return None
    value = _string(value, label)
    if not _ENV.fullmatch(value):
        raise ValueError(f"{label} must name an environment variable")
    return value


def _reject_secret_values(value: Any, path: str = "definition") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = key.lower()
            if any(word in lowered for word in _SECRET_WORDS) and not lowered.endswith("_env"):
                raise ValueError(f"Inline secret field is forbidden: {path}.{key}")
            _reject_secret_values(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_secret_values(child, f"{path}[{index}]")


def load_resident_definition(path: Path, prompt_root: Path) -> ResidentDefinition:
    try:
        source = path.read_text(encoding="utf-8")
        if any(isinstance(event, yaml.events.AliasEvent) for event in yaml.parse(source)):
            raise ValueError(f"YAML aliases are forbidden in {path.name}")
        document = yaml.safe_load(source)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path.name}: {exc}") from exc
    data = _mapping(document, path.name)
    unknown = set(data) - _ALLOWED
    if unknown:
        raise ValueError(f"Unknown fields in {path.name}: {', '.join(sorted(unknown))}")
    _reject_secret_values(data)
    if data.get("version", 1) != 1:
        raise ValueError(f"Unsupported definition version in {path.name}")
    resident_id = _string(data.get("id"), f"{path.name}.id")
    if not _ID.fullmatch(resident_id):
        raise ValueError(f"{path.name}.id must be a lowercase safe identifier")
    name = _string(data.get("name", resident_id), f"{path.name}.name")
    if "personality" not in data and "personality_prompt" not in data:
        raise ValueError(f"{path.name} must define personality or personality_prompt")
    if "role" not in data and "role_prompt" not in data:
        raise ValueError(f"{path.name} must define role or role_prompt")
    if "personality" in data and "personality_prompt" in data:
        raise ValueError(f"{path.name} must not define both personality and personality_prompt")
    if "role" in data and "role_prompt" in data:
        raise ValueError(f"{path.name} must not define both role and role_prompt")
    personality = (_prompt(prompt_root, data["personality_prompt"], "personality_prompt")
                   if "personality_prompt" in data else
                   _string(data["personality"], "personality"))
    role = (_prompt(prompt_root, data["role_prompt"], "role_prompt")
            if "role_prompt" in data else _string(data["role"], "role"))

    agent_data = _mapping(data.get("agent", {}), f"{path.name}.agent")
    agent_allowed = {"provider", "model", "api_key_env", "base_url_env", "agent_id_env",
                     "reasoning_effort", "service_tier"}
    if set(agent_data) - agent_allowed:
        raise ValueError(f"Unknown agent fields in {path.name}: {', '.join(sorted(set(agent_data) - agent_allowed))}")
    provider = _string(agent_data.get("provider", "openai-agents"), "agent.provider")
    if provider not in {"openai-agents", "openai-responses", "openai"}:
        raise ValueError(f"Unsupported provider for {resident_id}: {provider}")
    agent = AgentDefinition(
        provider=provider,
        model=_string(agent_data.get("model", "gpt-5.6-luna"), "agent.model"),
        api_key_env=_env_name(agent_data.get("api_key_env", "OPENAI_API_KEY"), "agent.api_key_env"),
        base_url_env=_env_name(agent_data.get("base_url_env", "OPENAI_BASE_URL"), "agent.base_url_env"),
        agent_id_env=_env_name(agent_data.get("agent_id_env"), "agent.agent_id_env", required=False),
        reasoning_effort=(
            _reasoning_effort(agent_data["reasoning_effort"], "agent.reasoning_effort")
            if agent_data.get("reasoning_effort") is not None else None),
        service_tier=(
            _string(agent_data["service_tier"], "agent.service_tier")
            if agent_data.get("service_tier") is not None else None),
    )
    curator = None
    if data.get("curator") is not None:
        curator_data = _mapping(data["curator"], f"{path.name}.curator")
        curator_allowed = {"model", "api_key_env", "base_url_env", "batch_size", "max_batches"}
        if set(curator_data) - curator_allowed:
            raise ValueError(
                f"Unknown curator fields in {path.name}: "
                f"{', '.join(sorted(set(curator_data) - curator_allowed))}")
        curator = CuratorDefinition(
            model=(
                _string(curator_data["model"], "curator.model")
                if curator_data.get("model") is not None else None),
            api_key_env=_env_name(
                curator_data.get("api_key_env", "OPENAI_API_KEY"),
                "curator.api_key_env"),
            base_url_env=_env_name(
                curator_data.get("base_url_env", "OPENAI_BASE_URL"),
                "curator.base_url_env"),
            batch_size=_positive_integer(
                curator_data.get("batch_size", 50), "curator.batch_size", maximum=100),
            max_batches=_positive_integer(
                curator_data.get("max_batches", 4), "curator.max_batches"),
        )
    transport = None
    if data.get("owner_transport") is not None:
        item = _mapping(data["owner_transport"], "owner_transport")
        allowed = {"type", "token_env", "owner_user_id_env", "owner_chat_id_env"}
        if set(item) - allowed or item.get("type") != "telegram":
            raise ValueError(f"{resident_id}.owner_transport must be a Telegram environment reference")
        transport = OwnerTransportDefinition(
            "telegram", _env_name(item.get("token_env"), "owner_transport.token_env"),
            _env_name(item.get("owner_user_id_env"), "owner_transport.owner_user_id_env"),
            _env_name(item.get("owner_chat_id_env"), "owner_transport.owner_chat_id_env"),
        )
    enabled = data.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError(f"{path.name}.enabled must be boolean")
    body = data.get("body")
    if body is not None:
        body = _mapping(body, "body")
    return ResidentDefinition(
        id=resident_id, name=name, personality=personality, role=role, enabled=enabled,
        agent=agent, curator=curator,
        capabilities=_string_list(data.get("capabilities"), "capabilities"),
        outputs=_string_list(data.get("outputs"), "outputs"),
        subscriptions=_subscriptions(data.get("subscriptions"), "subscriptions"),
        owner_transport=transport,
        external_applications=_external_applications(
            data.get("external_applications"), "external_applications"),
        body=body,
    )


def load_resident_catalog(directory: Path, *, prompt_root: Path | None = None,
                          default_id: str = "resident") -> ResidentCatalog:
    directory = directory.resolve()
    if not directory.is_dir():
        raise ValueError(f"Resident definitions directory does not exist: {directory}")
    prompt_root = (prompt_root or directory.parent / "prompts").resolve()
    paths = sorted((*directory.glob("*.yaml"), *directory.glob("*.yml")))
    if not paths:
        raise ValueError(f"No resident YAML definitions found in {directory}")
    residents = tuple(load_resident_definition(path, prompt_root) for path in paths)
    ids = [resident.id for resident in residents]
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        raise ValueError(f"Duplicate Resident id: {', '.join(duplicates)}")
    enabled = tuple(resident for resident in residents if resident.enabled)
    if not enabled:
        raise ValueError("At least one Resident must be enabled")
    if default_id not in {resident.id for resident in enabled}:
        raise ValueError(f"Default Resident is not enabled or defined: {default_id}")
    tokens: dict[str, str] = {}
    for resident in enabled:
        if resident.owner_transport:
            token_ref = resident.owner_transport.token_env
            if token_ref in tokens:
                raise ValueError(
                    f"Telegram bot reference {token_ref} is assigned to both {tokens[token_ref]} and {resident.id}")
            tokens[token_ref] = resident.id
    return ResidentCatalog(enabled, default_id)


def resolve_environment(name: str, *, required: bool = True) -> str | None:
    value = os.getenv(name, "").strip()
    if required and not value:
        raise ValueError(f"Required secret environment variable is not set: {name}")
    return value or None


def migrate_legacy_state(data_dir: Path, instance_id: str = "resident") -> Path:
    """Explicitly move the singleton database and SQLite sidecars to an instance directory."""
    if not _ID.fullmatch(instance_id):
        raise ValueError("Migration instance id is invalid")
    source = data_dir / "resident.sqlite3"
    target = data_dir / "instances" / instance_id / "resident.sqlite3"
    if target.exists():
        raise FileExistsError(f"Instance state already exists: {target}")
    if not source.exists():
        raise FileNotFoundError(f"Legacy Resident state does not exist: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm"):
        current = Path(f"{source}{suffix}")
        if current.exists():
            shutil.move(str(current), str(Path(f"{target}{suffix}")))
    return target
