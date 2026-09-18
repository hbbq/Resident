from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}\Z")
_ENV = re.compile(r"[A-Z_][A-Z0-9_]*\Z")
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
    "role", "role_prompt", "agent", "memory", "curator", "capabilities",
    "subscriptions", "owner_transport", "body",
}
_SECRET_WORDS = ("token", "password", "api_key", "secret", "credential")


@dataclass(frozen=True)
class AgentDefinition:
    provider: str = "openai-agents"
    model: str = "gpt-5.6-luna"
    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str = "OPENAI_BASE_URL"
    agent_id_env: str | None = None


@dataclass(frozen=True)
class OwnerTransportDefinition:
    type: str
    token_env: str
    owner_user_id_env: str
    owner_chat_id_env: str


@dataclass(frozen=True)
class ResidentDefinition:
    id: str
    name: str
    personality: str
    role: str = ""
    enabled: bool = True
    agent: AgentDefinition = field(default_factory=AgentDefinition)
    memory: dict[str, Any] = field(default_factory=dict)
    curator: dict[str, Any] = field(default_factory=dict)
    capabilities: tuple[str, ...] = ()
    subscriptions: tuple[str, ...] = ()
    owner_transport: OwnerTransportDefinition | None = None
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
    agent_allowed = {"provider", "model", "api_key_env", "base_url_env", "agent_id_env"}
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
    memory = _mapping(data.get("memory", {}), "memory")
    if set(memory) - {"enabled", "context_limit"}:
        raise ValueError(f"Unknown memory policy fields in {path.name}")
    if "enabled" in memory and not isinstance(memory["enabled"], bool):
        raise ValueError("memory.enabled must be boolean")
    if ("context_limit" in memory and
            (not isinstance(memory["context_limit"], int) or memory["context_limit"] < 0)):
        raise ValueError("memory.context_limit must be a nonnegative integer")
    curator = _mapping(data.get("curator", {}), "curator")
    body = data.get("body")
    if body is not None:
        body = _mapping(body, "body")
    return ResidentDefinition(
        resident_id, name, personality, role, enabled, agent, memory, curator,
        _string_list(data.get("capabilities"), "capabilities"),
        _subscriptions(data.get("subscriptions"), "subscriptions"), transport, body,
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
