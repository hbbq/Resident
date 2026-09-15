from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PERSONALITY = (
    "You are curious, observant, and playful. Build an understanding of your environment, "
    "preserve useful continuity, respect your owner's instructions, and communicate thoughtfully."
)


@dataclass(frozen=True)
class Config:
    data_dir: Path
    resident_name: str = "Resident"
    owner_name: str = "Owner"
    personality: str = DEFAULT_PERSONALITY
    provider: str = "openai"
    model: str = "gpt-5.6-luna"
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    max_tool_rounds: int = 8
    context_memories: int = 8
    context_messages: int = 8
    spontaneous_message_limit: int = 3
    spontaneous_message_window_seconds: int = 3600
    scheduler_poll_seconds: float = 1.0

    @classmethod
    def from_env_and_args(cls, argv: list[str] | None = None) -> "Config":
        parser = argparse.ArgumentParser(description="Run a persistent Resident instance")
        parser.add_argument("--data-dir", default=os.getenv("RESIDENT_DATA_DIR", ".resident"))
        parser.add_argument("--resident-name", default=os.getenv("RESIDENT_NAME", "Resident"))
        parser.add_argument("--owner-name", default=os.getenv("RESIDENT_OWNER_NAME", "Owner"))
        parser.add_argument("--personality", default=os.getenv("RESIDENT_PERSONALITY", DEFAULT_PERSONALITY))
        parser.add_argument("--provider", choices=("openai",), default=os.getenv("RESIDENT_PROVIDER", "openai"))
        parser.add_argument("--model", default=os.getenv("RESIDENT_MODEL", "gpt-5.6-luna"))
        parser.add_argument("--spontaneous-message-limit", type=int,
                            default=int(os.getenv("RESIDENT_SPONTANEOUS_MESSAGE_LIMIT", "3")))
        parser.add_argument("--spontaneous-message-window-seconds", type=int,
                            default=int(os.getenv("RESIDENT_SPONTANEOUS_MESSAGE_WINDOW_SECONDS", "3600")))
        args = parser.parse_args(argv)
        return cls(
            data_dir=Path(args.data_dir).expanduser(), resident_name=args.resident_name,
            owner_name=args.owner_name, personality=args.personality, provider=args.provider,
            model=args.model, openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            spontaneous_message_limit=max(0, args.spontaneous_message_limit),
            spontaneous_message_window_seconds=max(1, args.spontaneous_message_window_seconds),
        )
