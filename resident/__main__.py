from __future__ import annotations

import asyncio
import sys

from .config import Config
from .capabilities import diagnostic_capabilities
from .agentcontroller import AgentControllerConnector
from .camera import CameraConnector
from .display import DisplayConnector
from .homeops import HomeOpsConnector
from .provider import OpenAIResponsesProvider
from .runtime import ResidentRuntime
from .telegram import TelegramTransport


class TerminalDiagnostics:
    def __init__(self, verbose: bool):
        self.verbose = verbose

    @staticmethod
    def runtime(message: str) -> None:
        print(f"[runtime] {message}")

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


def main() -> int:
    config = Config.from_env_and_args()
    if not config.openai_api_key:
        print("OPENAI_API_KEY must be set for the OpenAI provider.", file=sys.stderr)
        return 2
    provider = OpenAIResponsesProvider(config.openai_api_key, config.model, config.openai_base_url)
    connectors = []
    capabilities = diagnostic_capabilities()
    terminal_diagnostics = TerminalDiagnostics(config.verbose)
    if config.homeops_url:
        homeops = HomeOpsConnector(
            config.homeops_url, poll_seconds=config.homeops_poll_seconds,
            request_timeout_seconds=config.homeops_request_timeout_seconds,
            diagnostic_output=terminal_diagnostics.homeops,
        )
        connectors.append(homeops)
        capabilities.extend(homeops.capabilities)
    if config.displays:
        assert config.homeops_url is not None
        displays = DisplayConnector(
            config.homeops_url, config.displays,
            request_timeout_seconds=config.homeops_request_timeout_seconds,
        )
        capabilities.extend(displays.capabilities)
    agentcontroller = None
    if config.agentcontroller_snapshot_path is not None:
        agentcontroller = AgentControllerConnector(
            config.agentcontroller_snapshot_path,
            poll_seconds=config.agentcontroller_poll_seconds,
            diagnostic_output=terminal_diagnostics.agentcontroller,
        )
        connectors.append(agentcontroller)
        capabilities.extend(agentcontroller.capabilities)
    if config.cameras:
        cameras = CameraConnector(
            config.cameras, timeout_seconds=config.camera_capture_timeout_seconds,
            max_width=config.camera_max_width, max_height=config.camera_max_height,
            max_bytes=config.camera_max_bytes, rtsp_transport=config.camera_rtsp_transport,
            ffmpeg_executable=config.ffmpeg_executable,
            onvif_request_timeout_seconds=config.camera_onvif_request_timeout_seconds,
            onvif_pull_timeout_seconds=config.camera_onvif_pull_timeout_seconds,
            onvif_retry_seconds=config.camera_onvif_retry_seconds,
            diagnostic_output=terminal_diagnostics.camera,
        )
        connectors.append(cameras)
        capabilities.extend(cameras.capabilities)
    telegram = None
    if config.telegram_bot_token is not None:
        telegram = TelegramTransport(
            config.telegram_bot_token, config.telegram_owner_user_id, config.telegram_owner_chat_id,
            poll_seconds=config.telegram_poll_seconds,
            request_timeout_seconds=config.telegram_request_timeout_seconds,
            diagnostic_output=terminal_diagnostics.telegram,
        )
        connectors.append(telegram)
    runtime = ResidentRuntime(
        config, provider, capabilities=capabilities, event_producers=connectors,
        owner_transport=telegram,
        diagnostic_output=terminal_diagnostics.runtime,
    )
    if agentcontroller is not None:
        checkpoint_scope = agentcontroller.checkpoint_scope
        agentcontroller.bind_checkpoint(
            lambda: runtime.store.observed_snapshot(checkpoint_scope),
            lambda snapshot: runtime.store.save_observed_snapshot(checkpoint_scope, snapshot),
        )
    if telegram is not None:
        telegram.bind_owner_message(runtime.telegram_owner_message_event)
        offset_scope = telegram.offset_checkpoint_scope
        telegram.bind_offset_checkpoint(
            lambda: runtime.store.observed_snapshot(offset_scope),
            lambda offset: runtime.store.save_observed_snapshot(offset_scope, offset),
        )
    try:
        asyncio.run(runtime.run_interactive())
    finally:
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

