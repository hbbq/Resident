from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Identity:
    id: str
    address_name: str
    personality: str = ""


@dataclass(frozen=True)
class WakeEvent:
    id: str
    source: str
    reason: str
    occurred_at: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    output: dict[str, Any]
    attachments: tuple[ImageAttachment, ...] = ()


@dataclass(frozen=True)
class ImageAttachment:
    data: bytes = field(repr=False)
    mime_type: str = "image/jpeg"
    detail: str = "auto"


@dataclass(frozen=True)
class ToolOutput:
    output: dict[str, Any]
    attachments: tuple[ImageAttachment, ...] = ()


@dataclass(frozen=True)
class ModelTurn:
    response_id: str | None = None
    message: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]

