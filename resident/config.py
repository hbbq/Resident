from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit


DEFAULT_PERSONALITY = (
    "You are curious, observant, and playful. Build an understanding of your environment, "
    "preserve useful continuity, respect your owner's instructions, and communicate thoughtfully."
)

SUPPORTED_REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")


def _environment_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class OnvifConfig:
    endpoint: str = field(repr=False)
    username: str = field(repr=False)
    password: str = field(repr=False)


@dataclass(frozen=True)
class CameraConfig:
    id: str
    name: str
    rtsp_url: str = field(repr=False)
    description: str | None = None
    onvif: OnvifConfig | None = field(default=None, repr=False)


@dataclass(frozen=True)
class DisplayConfig:
    id: str
    max_length: int | None = None


def _displays_from_environment() -> tuple[DisplayConfig, ...]:
    raw = os.getenv("RESIDENT_DISPLAYS", "").strip()
    if not raw:
        return ()
    try:
        items = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("RESIDENT_DISPLAYS must be valid JSON") from exc
    if not isinstance(items, list):
        raise ValueError("RESIDENT_DISPLAYS must be a JSON array")
    displays: list[DisplayConfig] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict) or not set(item) <= {"id", "max_length"} or "id" not in item:
            raise ValueError("Each display must contain id and optional max_length")
        display_id = item.get("id")
        if not isinstance(display_id, str) or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", display_id):
            raise ValueError("Display id must be 1-64 tool-safe identifier characters")
        if display_id in seen:
            raise ValueError(f"Duplicate display id: {display_id}")
        max_length = item.get("max_length")
        if (max_length is not None and
                (not isinstance(max_length, int) or isinstance(max_length, bool)
                 or not 1 <= max_length <= 10000)):
            raise ValueError("Display max_length must be an integer from 1 through 10000")
        seen.add(display_id)
        displays.append(DisplayConfig(display_id, max_length))
    return tuple(displays)


def _cameras_from_environment() -> tuple[CameraConfig, ...]:
    raw = os.getenv("RESIDENT_CAMERAS", "").strip()
    if not raw:
        return ()
    try:
        items = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("RESIDENT_CAMERAS must be valid JSON") from exc
    if not isinstance(items, list):
        raise ValueError("RESIDENT_CAMERAS must be a JSON array")
    cameras: list[CameraConfig] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict) or set(item) - {"id", "name", "url", "description", "onvif"}:
            raise ValueError(
                "Each camera must contain only id, name, url, optional description, and optional onvif")
        camera_id, name, url = item.get("id"), item.get("name"), item.get("url")
        description = item.get("description")
        onvif_item = item.get("onvif")
        if not isinstance(camera_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", camera_id):
            raise ValueError("Camera id must be 1-64 safe identifier characters")
        if camera_id in seen:
            raise ValueError(f"Duplicate camera id: {camera_id}")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Camera {camera_id} must have a nonempty name")
        if not isinstance(url, str) or not url.lower().startswith(("rtsp://", "rtsps://")):
            raise ValueError(f"Camera {camera_id} must have an RTSP URL")
        if description is not None and not isinstance(description, str):
            raise ValueError(f"Camera {camera_id} description must be a string")
        onvif = None
        if onvif_item is not None:
            if not isinstance(onvif_item, dict) or set(onvif_item) != {"endpoint", "username", "password"}:
                raise ValueError(
                    f"Camera {camera_id} onvif must contain endpoint, username, and password")
            endpoint = onvif_item.get("endpoint")
            username = onvif_item.get("username")
            password = onvif_item.get("password")
            try:
                parsed = urlsplit(endpoint) if isinstance(endpoint, str) else None
                port = parsed.port if parsed is not None else None
            except ValueError as exc:
                raise ValueError(
                    f"Camera {camera_id} ONVIF endpoint must have a valid port") from exc
            if parsed is None or parsed.scheme not in ("http", "https") or not parsed.netloc:
                raise ValueError(f"Camera {camera_id} ONVIF endpoint must be an absolute HTTP(S) URL")
            effective_port = port if port is not None else {"http": 80, "https": 443}[parsed.scheme]
            if not 1 <= effective_port <= 65535:
                raise ValueError(f"Camera {camera_id} ONVIF endpoint must have a valid port")
            if not isinstance(username, str) or not username:
                raise ValueError(f"Camera {camera_id} ONVIF username must be nonempty")
            if not isinstance(password, str) or not password:
                raise ValueError(f"Camera {camera_id} ONVIF password must be nonempty")
            onvif = OnvifConfig(endpoint, username, password)
        seen.add(camera_id)
        cameras.append(CameraConfig(
            camera_id, name.strip(), url, description.strip() if description else None, onvif))
    return tuple(cameras)


@dataclass(frozen=True)
class Config:
    data_dir: Path
    residents_dir: Path | None = None
    prompt_root: Path | None = None
    default_resident: str = "resident"
    migrate_legacy: bool = False
    instance_id: str = "resident"
    resident_name: str = "Resident"
    owner_name: str = "Owner"
    personality: str = DEFAULT_PERSONALITY
    role: str = ""
    owner_communication_enabled: bool = True
    provider: str = "openai-agents"
    model: str = "gpt-5.6-luna"
    reasoning_effort: str | None = None
    service_tier: str | None = None
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_agent_id: str | None = None
    new_chapter: bool = False
    curator_model: str | None = None
    curator_api_key: str | None = field(default=None, repr=False)
    curator_base_url: str = "https://api.openai.com/v1"
    curator_batch_size: int = 50
    curator_max_batches: int = 4
    max_tool_rounds: int = 8
    context_messages: int = 8
    keeper_rollover_interactions: int = 8
    keeper_rollover_bytes: int = 16384
    spontaneous_message_limit: int = 3
    spontaneous_message_window_seconds: int = 180
    scheduler_poll_seconds: float = 1.0
    homeops_url: str | None = None
    homeops_poll_seconds: float = 30.0
    homeops_request_timeout_seconds: float = 10.0
    displays: tuple[DisplayConfig, ...] = ()
    agentcontroller_snapshot_path: Path | None = None
    agentcontroller_poll_seconds: float = 60.0
    telegram_bot_token: str | None = field(default=None, repr=False)
    telegram_owner_user_id: int | None = field(default=None, repr=False)
    telegram_owner_chat_id: int | None = field(default=None, repr=False)
    telegram_poll_seconds: float = 30.0
    telegram_request_timeout_seconds: float = 40.0
    cameras: tuple[CameraConfig, ...] = ()
    camera_capture_timeout_seconds: float = 8.0
    camera_max_width: int = 1280
    camera_max_height: int = 720
    camera_max_bytes: int = 2_000_000
    camera_rtsp_transport: str = "tcp"
    camera_onvif_request_timeout_seconds: float = 10.0
    camera_onvif_pull_timeout_seconds: float = 5.0
    camera_onvif_retry_seconds: float = 30.0
    ffmpeg_executable: str = "ffmpeg"
    verbose: bool = False
    timeline: bool = False

    @classmethod
    def from_env_and_args(cls, argv: list[str] | None = None) -> "Config":
        parser = argparse.ArgumentParser(description="Run a persistent Resident instance")
        parser.add_argument("--data-dir", default=os.getenv("RESIDENT_DATA_DIR", ".resident"))
        parser.add_argument("--residents-dir", default=os.getenv("RESIDENTS_DIR"),
                            help="directory of startup-time Resident YAML definitions")
        parser.add_argument("--prompt-root", default=os.getenv("RESIDENT_PROMPT_ROOT"))
        parser.add_argument("--default-resident", default=os.getenv("RESIDENT_DEFAULT_ID", "resident"))
        parser.add_argument("--migrate-legacy", action="store_true",
                            help="move resident.sqlite3 into instances/resident and exit")
        parser.add_argument("--verbose", action="store_true", default=_environment_flag("RESIDENT_VERBOSE"),
                            help="show detailed runtime and connector diagnostics")
        parser.add_argument("--timeline", action="store_true",
                            default=_environment_flag("RESIDENT_TIMELINE"),
                            help="record structured latency timeline events")
        parser.add_argument("--resident-name", default=os.getenv("RESIDENT_NAME", "Resident"))
        parser.add_argument("--owner-name", default=os.getenv("RESIDENT_OWNER_NAME", "Owner"))
        parser.add_argument("--personality", default=os.getenv("RESIDENT_PERSONALITY", DEFAULT_PERSONALITY))
        parser.add_argument("--provider", choices=("openai-agents", "openai-responses", "openai"),
                            default=os.getenv("RESIDENT_PROVIDER", "openai-agents"))
        parser.add_argument("--model", default=os.getenv("RESIDENT_MODEL", "gpt-5.6-luna"))
        parser.add_argument("--reasoning-effort", choices=SUPPORTED_REASONING_EFFORTS,
                            default=os.getenv("RESIDENT_REASONING_EFFORT"))
        parser.add_argument("--service-tier", default=os.getenv("RESIDENT_SERVICE_TIER"))
        parser.add_argument("--curator-model", default=os.getenv("RESIDENT_CURATOR_MODEL"),
                            help="separate model for durable memory consolidation; disabled when omitted")
        parser.add_argument("--new-chapter", action="store_true",
                            default=_environment_flag("RESIDENT_NEW_CHAPTER"),
                            help="intentionally roll over the current Agents session at the next wake")
        parser.add_argument("--curator-batch-size", type=int,
                            default=int(os.getenv("RESIDENT_CURATOR_BATCH_SIZE", "50")))
        parser.add_argument("--curator-max-batches", type=int,
                            default=int(os.getenv("RESIDENT_CURATOR_MAX_BATCHES", "4")))
        parser.add_argument("--keeper-rollover-interactions", type=int,
                            default=int(os.getenv("KEEPER_ROLLOVER_INTERACTIONS", "8")))
        parser.add_argument("--keeper-rollover-bytes", type=int,
                            default=int(os.getenv("KEEPER_ROLLOVER_BYTES", "16384")))
        parser.add_argument("--spontaneous-message-limit", type=int,
                            default=int(os.getenv("RESIDENT_SPONTANEOUS_MESSAGE_LIMIT", "3")))
        parser.add_argument("--spontaneous-message-window-seconds", type=int,
                            default=int(os.getenv("RESIDENT_SPONTANEOUS_MESSAGE_WINDOW_SECONDS", "180")))
        parser.add_argument("--homeops-url", default=os.getenv("RESIDENT_HOMEOPS_URL"))
        parser.add_argument("--homeops-poll-seconds", type=float,
                            default=float(os.getenv("RESIDENT_HOMEOPS_POLL_SECONDS", "30")))
        parser.add_argument("--homeops-request-timeout-seconds", type=float,
                            default=float(os.getenv("RESIDENT_HOMEOPS_REQUEST_TIMEOUT_SECONDS", "10")))
        parser.add_argument("--agentcontroller-snapshot-path",
                            default=os.getenv("RESIDENT_AGENTCONTROLLER_SNAPSHOT_PATH"))
        parser.add_argument("--agentcontroller-poll-seconds", type=float,
                            default=float(os.getenv("RESIDENT_AGENTCONTROLLER_POLL_SECONDS", "60")))
        parser.add_argument("--telegram-poll-seconds", type=float,
                            default=float(os.getenv("RESIDENT_TELEGRAM_POLL_SECONDS", "30")))
        parser.add_argument("--telegram-request-timeout-seconds", type=float,
                            default=float(os.getenv("RESIDENT_TELEGRAM_REQUEST_TIMEOUT_SECONDS", "40")))
        parser.add_argument("--camera-capture-timeout-seconds", type=float,
                            default=float(os.getenv("RESIDENT_CAMERA_CAPTURE_TIMEOUT_SECONDS", "8")))
        parser.add_argument("--camera-max-width", type=int,
                            default=int(os.getenv("RESIDENT_CAMERA_MAX_WIDTH", "1280")))
        parser.add_argument("--camera-max-height", type=int,
                            default=int(os.getenv("RESIDENT_CAMERA_MAX_HEIGHT", "720")))
        parser.add_argument("--camera-max-bytes", type=int,
                            default=int(os.getenv("RESIDENT_CAMERA_MAX_BYTES", "2000000")))
        parser.add_argument("--camera-rtsp-transport", choices=("tcp", "udp"),
                            default=os.getenv("RESIDENT_CAMERA_RTSP_TRANSPORT", "tcp"))
        parser.add_argument("--camera-onvif-request-timeout-seconds", type=float,
                            default=float(os.getenv("RESIDENT_CAMERA_ONVIF_REQUEST_TIMEOUT_SECONDS", "10")))
        parser.add_argument("--camera-onvif-pull-timeout-seconds", type=float,
                            default=float(os.getenv("RESIDENT_CAMERA_ONVIF_PULL_TIMEOUT_SECONDS", "5")))
        parser.add_argument("--camera-onvif-retry-seconds", type=float,
                            default=float(os.getenv("RESIDENT_CAMERA_ONVIF_RETRY_SECONDS", "30")))
        parser.add_argument("--ffmpeg-executable", default=os.getenv("RESIDENT_FFMPEG_EXECUTABLE", "ffmpeg"))
        args = parser.parse_args(argv)
        telegram_token = os.getenv("RESIDENT_TELEGRAM_BOT_TOKEN", "").strip() or None
        telegram_user = os.getenv("RESIDENT_TELEGRAM_OWNER_USER_ID", "").strip() or None
        telegram_chat = os.getenv("RESIDENT_TELEGRAM_OWNER_CHAT_ID", "").strip() or None
        configured_telegram_values = (telegram_token, telegram_user, telegram_chat)
        if any(configured_telegram_values) and not all(configured_telegram_values):
            raise ValueError(
                "Telegram requires RESIDENT_TELEGRAM_BOT_TOKEN, "
                "RESIDENT_TELEGRAM_OWNER_USER_ID, and RESIDENT_TELEGRAM_OWNER_CHAT_ID")
        try:
            telegram_user_id = int(telegram_user) if telegram_user else None
            telegram_chat_id = int(telegram_chat) if telegram_chat else None
        except ValueError as exc:
            raise ValueError("Telegram Owner user and chat IDs must be integers") from exc
        if telegram_user_id is not None and (telegram_user_id <= 0 or telegram_chat_id <= 0):
            raise ValueError("Telegram Owner user and private chat IDs must be positive integers")
        displays = _displays_from_environment()
        if displays and not args.homeops_url:
            raise ValueError("RESIDENT_DISPLAYS requires RESIDENT_HOMEOPS_URL or --homeops-url")
        return cls(
            data_dir=Path(args.data_dir).expanduser(), verbose=args.verbose,
            timeline=args.timeline,
            residents_dir=Path(args.residents_dir).expanduser() if args.residents_dir else None,
            prompt_root=Path(args.prompt_root).expanduser() if args.prompt_root else None,
            default_resident=args.default_resident, migrate_legacy=args.migrate_legacy,
            resident_name=args.resident_name,
            owner_name=args.owner_name, personality=args.personality, provider=args.provider,
            model=args.model, openai_api_key=os.getenv("OPENAI_API_KEY"),
            reasoning_effort=args.reasoning_effort, service_tier=args.service_tier,
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            openai_agent_id=os.getenv("RESIDENT_OPENAI_AGENT_ID", "").strip() or None,
            new_chapter=args.new_chapter,
            curator_model=args.curator_model,
            curator_api_key=os.getenv("RESIDENT_CURATOR_API_KEY") or os.getenv("OPENAI_API_KEY"),
            curator_base_url=os.getenv("RESIDENT_CURATOR_BASE_URL",
                                       os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/"),
            curator_batch_size=max(1, min(args.curator_batch_size, 100)),
            curator_max_batches=max(1, args.curator_max_batches),
            keeper_rollover_interactions=max(0, args.keeper_rollover_interactions),
            keeper_rollover_bytes=max(0, args.keeper_rollover_bytes),
            spontaneous_message_limit=max(0, args.spontaneous_message_limit),
            spontaneous_message_window_seconds=max(1, args.spontaneous_message_window_seconds),
            homeops_url=args.homeops_url.rstrip("/") if args.homeops_url else None,
            homeops_poll_seconds=max(0.1, args.homeops_poll_seconds),
            homeops_request_timeout_seconds=max(0.1, args.homeops_request_timeout_seconds),
            displays=displays,
            agentcontroller_snapshot_path=(
                Path(args.agentcontroller_snapshot_path).expanduser()
                if args.agentcontroller_snapshot_path else None
            ),
            agentcontroller_poll_seconds=max(0.1, args.agentcontroller_poll_seconds),
            telegram_bot_token=telegram_token,
            telegram_owner_user_id=telegram_user_id,
            telegram_owner_chat_id=telegram_chat_id,
            telegram_poll_seconds=max(1.0, args.telegram_poll_seconds),
            telegram_request_timeout_seconds=max(1.0, args.telegram_request_timeout_seconds),
            cameras=_cameras_from_environment(),
            camera_capture_timeout_seconds=max(0.1, args.camera_capture_timeout_seconds),
            camera_max_width=max(1, args.camera_max_width),
            camera_max_height=max(1, args.camera_max_height),
            camera_max_bytes=max(1024, args.camera_max_bytes),
            camera_rtsp_transport=args.camera_rtsp_transport,
            camera_onvif_request_timeout_seconds=max(0.1, args.camera_onvif_request_timeout_seconds),
            camera_onvif_pull_timeout_seconds=max(1.0, args.camera_onvif_pull_timeout_seconds),
            camera_onvif_retry_seconds=max(0.1, args.camera_onvif_retry_seconds),
            ffmpeg_executable=args.ffmpeg_executable,
        )
