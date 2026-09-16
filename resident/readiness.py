from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReadinessItem:
    key: str
    label: str


@dataclass(frozen=True)
class ReadinessResult:
    key: str
    ok: bool
    detail: str | None = None

