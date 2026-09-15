from __future__ import annotations

import asyncio
import sys

from .config import Config
from .provider import OpenAIResponsesProvider
from .runtime import ResidentRuntime


def main() -> int:
    config = Config.from_env_and_args()
    if not config.openai_api_key:
        print("OPENAI_API_KEY must be set for the OpenAI provider.", file=sys.stderr)
        return 2
    provider = OpenAIResponsesProvider(config.openai_api_key, config.model, config.openai_base_url)
    runtime = ResidentRuntime(config, provider)
    try:
        asyncio.run(runtime.run_interactive())
    finally:
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

