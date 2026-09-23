from __future__ import annotations

import asyncio
import sys
from dataclasses import replace

from .agentcontroller import AgentControllerConnector
from .camera import CameraConnector
from .capabilities import Capability, diagnostic_capabilities
from .config import Config
from .display import DisplayConnector
from .external_app import ExternalApplicationConnector
from .realm import RealmClient
from .homeops import HomeOpsConnector
from .host import InstancePolicy, RuntimeHost, messaging_capability
from .instances import (ResidentDefinition, load_resident_catalog, migrate_legacy_state,
                        resolve_environment)
from .mailbox import Mailbox
from .memory import MemoryCurator, OpenAICuratorModel
from .outputs import OutputCapability
from .provider import OpenAIAgentsProvider, OpenAIResponsesProvider
from .runtime import ResidentRuntime
from .telegram import TelegramTransport


class TerminalDiagnostics:
    def __init__(self, verbose: bool):
        self.verbose = verbose

    def runtime(self, message: str) -> None:
        print(f"[runtime] {message}")

    def connector(self, message: str) -> None:
        if self.verbose:
            print(f"[connector] {message}")

    def homeops(self, message: str) -> None:
        if self.verbose:
            print(f"[homeops] {message}")

    def agentcontroller(self, message: str) -> None:
        if self.verbose:
            print(f"[agentcontroller] {message}")

    def camera(self, message: str) -> None:
        if self.verbose:
            print(f"[camera] {message}")

    def telegram(self, message: str) -> None:
        if self.verbose or message.startswith("permanent failure:"):
            print(f"[telegram] {message}")


def _provider(definition: ResidentDefinition):
    api_key = resolve_environment(definition.agent.api_key_env)
    base_url = (resolve_environment(definition.agent.base_url_env, required=False)
                or "https://api.openai.com/v1").rstrip("/")
    agent_id = (resolve_environment(definition.agent.agent_id_env, required=False)
                if definition.agent.agent_id_env else None)
    if definition.agent.provider == "openai-agents":
        return OpenAIAgentsProvider(
            api_key, definition.agent.model, base_url, agent_id=agent_id,
            reasoning_effort=definition.agent.reasoning_effort,
            service_tier=definition.agent.service_tier)
    return OpenAIResponsesProvider(api_key, definition.agent.model, base_url)


def _shared_resources(config: Config, diagnostics: TerminalDiagnostics):
    producers = []
    capabilities: list[Capability] = diagnostic_capabilities()
    output_capabilities: list[OutputCapability] = []
    if config.homeops_url:
        connector = HomeOpsConnector(
            config.homeops_url, poll_seconds=config.homeops_poll_seconds,
            request_timeout_seconds=config.homeops_request_timeout_seconds,
            diagnostic_output=diagnostics.homeops)
        producers.append(connector)
        capabilities.extend(connector.capabilities)
    if config.displays:
        display = DisplayConnector(
            config.homeops_url, config.displays,
            request_timeout_seconds=config.homeops_request_timeout_seconds)
        capabilities.extend(display.capabilities)
        output_capabilities.extend(display.output_capabilities)
    if config.agentcontroller_snapshot_path is not None:
        connector = AgentControllerConnector(
            config.agentcontroller_snapshot_path,
            poll_seconds=config.agentcontroller_poll_seconds,
            diagnostic_output=diagnostics.agentcontroller)
        producers.append(connector)
        capabilities.extend(connector.capabilities)
    if config.cameras:
        connector = CameraConnector(
            config.cameras, timeout_seconds=config.camera_capture_timeout_seconds,
            max_width=config.camera_max_width, max_height=config.camera_max_height,
            max_bytes=config.camera_max_bytes, rtsp_transport=config.camera_rtsp_transport,
            ffmpeg_executable=config.ffmpeg_executable,
            onvif_request_timeout_seconds=config.camera_onvif_request_timeout_seconds,
            onvif_pull_timeout_seconds=config.camera_onvif_pull_timeout_seconds,
            onvif_retry_seconds=config.camera_onvif_retry_seconds,
            diagnostic_output=diagnostics.camera)
        producers.append(connector)
        capabilities.extend(connector.capabilities)
    return producers, capabilities, output_capabilities


def _bind_curator(runtime: ResidentRuntime, config: Config) -> None:
    if not config.curator_model or not isinstance(runtime.provider, OpenAIAgentsProvider):
        return
    runtime.bind_curator(MemoryCurator(
        runtime.store, runtime.provider,
        OpenAICuratorModel(config.curator_api_key, config.curator_model,
                           config.curator_base_url),
        batch_size=config.curator_batch_size, max_batches=config.curator_max_batches))


def _select_capabilities(grants: tuple[str, ...], available: list[Capability]) -> list[Capability]:
    selected = [capability for capability in available
                if capability.name in grants or capability.connector_id in grants]
    known = ({capability.name for capability in available} |
             {capability.connector_id for capability in available} | {"messaging"})
    unknown = sorted(set(grants) - known)
    if unknown:
        raise ValueError(f"Unknown capability grants: {', '.join(unknown)}")
    return selected


def _validate_external_grant_namespace(
        definition: ResidentDefinition, shared: list[Capability]) -> None:
    """Keep provider and tool grants distinct in the simple grant namespace."""
    providers = {application.id for application in definition.external_applications}
    shared_providers = {capability.connector_id for capability in shared} | {"messaging"}
    shared_names = {capability.name for capability in shared} | {"messaging_send"}
    for application in definition.external_applications:
        if application.id in shared_names:
            raise ValueError(
                f"External application id {application.id!r} for {definition.id} "
                "conflicts with a built-in capability name; rename the provider")
        for operation in application.operations:
            if operation.name in providers:
                raise ValueError(
                    f"External tool {operation.name!r} from {application.id} for "
                    f"{definition.id} conflicts with an external provider id; "
                    "rename the provider or tool")
            if operation.name in shared_providers:
                raise ValueError(
                    f"External tool {operation.name!r} from {application.id} for "
                    f"{definition.id} conflicts with a built-in provider id; "
                    "rename the tool")


def _select_outputs(resident_id: str, grants: tuple[str, ...],
                    available: list[OutputCapability], *,
                    owner_available: bool) -> tuple[list[OutputCapability], bool]:
    available_by_id = {output.grant_id: output for output in available}
    supported = set(available_by_id)
    if owner_available:
        supported.add("notify_owner")
    unavailable = sorted(set(grants) - supported)
    if unavailable:
        raise ValueError(
            f"Unknown or unavailable output grants for {resident_id}: "
            f"{', '.join(unavailable)}")
    return ([output for output in available if output.grant_id in grants],
            "notify_owner" in grants)


def _legacy_output_capabilities(
        selected: list[OutputCapability], available: list[Capability],
) -> list[Capability]:
    names = {output.legacy_tool_name for output in selected if output.legacy_tool_name}
    return [capability for capability in available if capability.name in names]


def build_host(config: Config) -> RuntimeHost:
    if config.residents_dir is None:
        raise ValueError("A Resident definitions directory is required")
    catalog = load_resident_catalog(
        config.residents_dir, prompt_root=config.prompt_root, default_id=config.default_resident)
    diagnostics = TerminalDiagnostics(config.verbose)
    producers, available, available_outputs = _shared_resources(config, diagnostics)
    for definition in catalog.residents:
        _validate_external_grant_namespace(definition, available)
    mailbox = Mailbox(config.data_dir / "runtime" / "mailbox.sqlite3")
    for producer in producers:
        bind = getattr(producer, "bind_checkpoint", None)
        if bind is not None:
            scope = producer.checkpoint_scope
            bind(lambda scope=scope: mailbox.observed_snapshot(scope),
                 lambda snapshot, scope=scope: mailbox.save_observed_snapshot(scope, snapshot))
    runtimes: dict[str, ResidentRuntime] = {}
    policies = {}
    private_producers = {}
    resolved_tokens: dict[str, str] = {}
    recipient_addresses = frozenset(definition.id for definition in catalog.residents)
    recipients = lambda: recipient_addresses
    try:
        for definition in catalog.residents:
            curator = definition.curator
            curator_model = curator.model if curator else None
            curator_api_key = (
                resolve_environment(curator.api_key_env)
                if curator and curator_model else None)
            curator_base_url = (
                (resolve_environment(curator.base_url_env, required=False)
                 or "https://api.openai.com/v1").rstrip("/")
                if curator else "https://api.openai.com/v1")
            instance_config = replace(
                config, data_dir=config.data_dir / "instances" / definition.id,
                instance_id=definition.id, resident_name=definition.name,
                personality=definition.personality, role=definition.role,
                owner_communication_enabled=("notify_owner" in definition.outputs),
                provider=definition.agent.provider, model=definition.agent.model,
                reasoning_effort=definition.agent.reasoning_effort,
                service_tier=definition.agent.service_tier,
                curator_model=curator_model,
                curator_api_key=curator_api_key,
                curator_base_url=curator_base_url,
                curator_batch_size=curator.batch_size if curator else 50,
                curator_max_batches=curator.max_batches if curator else 4,
                # The command-line request intentionally targets every enabled
                # Resident constructed for this catalog startup, exactly once.
                new_chapter=config.new_chapter,
                openai_api_key=None, openai_agent_id=None, residents_dir=None)
            transport = None
            if definition.owner_transport:
                item = definition.owner_transport
                token = resolve_environment(item.token_env)
                if token in resolved_tokens:
                    raise ValueError(
                        f"One Telegram bot cannot serve both {resolved_tokens[token]} and {definition.id}")
                resolved_tokens[token] = definition.id
                try:
                    user_id = int(resolve_environment(item.owner_user_id_env))
                    chat_id = int(resolve_environment(item.owner_chat_id_env))
                except ValueError as exc:
                    raise ValueError(f"Telegram IDs for {definition.id} must be integers") from exc
                transport = TelegramTransport(
                    token, user_id, chat_id, poll_seconds=config.telegram_poll_seconds,
                    request_timeout_seconds=config.telegram_request_timeout_seconds,
                    diagnostic_output=diagnostics.telegram)
                private_producers[definition.id] = [transport]
            instance_available = list(available)
            realm_client = None
            if definition.realm is not None:
                item = definition.realm
                realm_client = RealmClient(
                    item.base_url, resolve_environment(item.game_id_env),
                    resolve_environment(item.actor_id_env), item.request_timeout_seconds)
                instance_available.extend(realm_client.capabilities)
            shared_connector_ids = {capability.connector_id for capability in available} | {"messaging"}
            for external_definition in definition.external_applications:
                if realm_client is not None and external_definition.id == "realm":
                    raise ValueError("Native Realm integration conflicts with external provider realm")
                if external_definition.id in shared_connector_ids:
                    raise ValueError(
                        f"External application id conflicts with an existing connector: "
                        f"{external_definition.id}")
                token = (resolve_environment(external_definition.bearer_token_env)
                         if external_definition.bearer_token_env else None)
                external = ExternalApplicationConnector(external_definition, token)
                instance_available.extend(external.capabilities)
            special_messaging = messaging_capability(mailbox, definition.id, recipients)
            inventory_names = [capability.name for capability in instance_available]
            inventory_names.append(special_messaging.name)
            duplicate_names = sorted({name for name in inventory_names
                                      if inventory_names.count(name) > 1})
            if duplicate_names:
                raise ValueError(
                    f"Duplicate available capability names for {definition.id}: "
                    f"{', '.join(duplicate_names)}")
            grants = _select_capabilities(definition.capabilities, instance_available)
            output_grants, owner_output_enabled = _select_outputs(
                definition.id, definition.outputs, available_outputs,
                owner_available=(
                    definition.owner_transport is not None
                    or definition.id == catalog.default_id))
            granted_names = {capability.name for capability in grants}
            grants.extend(capability for capability in _legacy_output_capabilities(
                output_grants, instance_available) if capability.name not in granted_names)
            if "messaging" in definition.capabilities:
                grants.append(special_messaging)
            runtime = ResidentRuntime(
                instance_config, _provider(definition), capabilities=grants,
                realm_client=realm_client,
                output_capabilities=output_grants,
                owner_transport=transport,
                owner_output_enabled=owner_output_enabled,
                diagnostic_output=lambda message, item=definition.id:
                    diagnostics.runtime(f"{item}: {message}"))
            _bind_curator(runtime, instance_config)
            runtimes[definition.id] = runtime
            policies[definition.id] = InstancePolicy(frozenset(definition.subscriptions))
            if transport:
                transport.bind_owner_message(runtime.telegram_owner_message_event)
                scope = transport.offset_checkpoint_scope
                transport.bind_offset_checkpoint(
                    lambda runtime=runtime, scope=scope: runtime.store.observed_snapshot(scope),
                    lambda offset, runtime=runtime, scope=scope:
                        runtime.store.save_observed_snapshot(scope, offset))
        return RuntimeHost(
            runtimes, policies, mailbox, event_producers=producers,
            instance_producers=private_producers, default_id=catalog.default_id,
            diagnostic_output=diagnostics.runtime)
    except Exception:
        for runtime in runtimes.values():
            runtime.close()
        mailbox.close()
        raise


def _legacy_runtime(config: Config) -> ResidentRuntime:
    if not config.openai_api_key:
        raise ValueError("OPENAI_API_KEY must be set for the OpenAI provider")
    provider = (OpenAIAgentsProvider(config.openai_api_key, config.model, config.openai_base_url,
                                     agent_id=config.openai_agent_id,
                                     reasoning_effort=config.reasoning_effort,
                                     service_tier=config.service_tier)
                if config.provider == "openai-agents" else
                OpenAIResponsesProvider(config.openai_api_key, config.model, config.openai_base_url))
    diagnostics = TerminalDiagnostics(config.verbose)
    producers, capabilities, output_capabilities = _shared_resources(config, diagnostics)
    telegram = None
    if config.telegram_bot_token is not None:
        telegram = TelegramTransport(
            config.telegram_bot_token, config.telegram_owner_user_id,
            config.telegram_owner_chat_id, poll_seconds=config.telegram_poll_seconds,
            request_timeout_seconds=config.telegram_request_timeout_seconds,
            diagnostic_output=diagnostics.telegram)
        producers.append(telegram)
    runtime = ResidentRuntime(
        config, provider, capabilities=capabilities, event_producers=producers,
        output_capabilities=output_capabilities,
        owner_transport=telegram, diagnostic_output=diagnostics.runtime)
    _bind_curator(runtime, config)
    for producer in producers:
        bind = getattr(producer, "bind_checkpoint", None)
        if bind is not None:
            scope = producer.checkpoint_scope
            bind(lambda scope=scope: runtime.store.observed_snapshot(scope),
                 lambda snapshot, scope=scope:
                     runtime.store.save_observed_snapshot(scope, snapshot))
    if telegram is not None:
        telegram.bind_owner_message(runtime.telegram_owner_message_event)
        scope = telegram.offset_checkpoint_scope
        telegram.bind_offset_checkpoint(
            lambda: runtime.store.observed_snapshot(scope),
            lambda offset: runtime.store.save_observed_snapshot(scope, offset))
    return runtime


def main() -> int:
    try:
        config = Config.from_env_and_args()
        if config.migrate_legacy:
            target = migrate_legacy_state(config.data_dir)
            print(f"Migrated legacy Resident state to {target}")
            return 0
        if config.residents_dir is not None:
            host = build_host(config)
            try:
                asyncio.run(host.run())
            finally:
                host.close()
        else:
            runtime = _legacy_runtime(config)
            try:
                asyncio.run(runtime.run_interactive())
            finally:
                runtime.close()
        return 0
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
