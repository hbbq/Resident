from __future__ import annotations

import asyncio
import sys

from .config import Config
from .capabilities import diagnostic_capabilities
from .camera import CameraConnector
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
    if config.cameras:
        cameras = CameraConnector(
            config.cameras, timeout_seconds=config.camera_capture_timeout_seconds,
            max_width=config.camera_max_width, max_height=config.camera_max_height,
            max_bytes=config.camera_max_bytes, rtsp_transport=config.camera_rtsp_transport,
            ffmpeg_executable=config.ffmpeg_executable,
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

