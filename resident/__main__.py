from __future__ import annotations

import asyncio
import sys

from .config import Config
from .capabilities import diagnostic_capabilities
from .homeops import HomeOpsConnector
from .provider import OpenAIResponsesProvider
from .runtime import ResidentRuntime


def main() -> int:
    config = Config.from_env_and_args()
    if not config.openai_api_key:
        print("OPENAI_API_KEY must be set for the OpenAI provider.", file=sys.stderr)
        return 2
    provider = OpenAIResponsesProvider(config.openai_api_key, config.model, config.openai_base_url)
    connectors = []
    capabilities = diagnostic_capabilities()
    if config.homeops_url:
        homeops = HomeOpsConnector(
            config.homeops_url, poll_seconds=config.homeops_poll_seconds,
            request_timeout_seconds=config.homeops_request_timeout_seconds,
            diagnostic_output=lambda message: print(f"[homeops] {message}"),
        )
        connectors.append(homeops)
        capabilities.extend(homeops.capabilities)
    runtime = ResidentRuntime(config, provider, capabilities=capabilities, event_producers=connectors)
    try:
        asyncio.run(runtime.run_interactive())
    finally:
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

