from __future__ import annotations

import sqlite3
import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


DEFAULT_TTL_SECONDS = 300


def _now() -> datetime:
    return datetime.now(UTC)


class Mailbox:
    """A process-shared durable, asynchronous logical-address mailbox."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS mailbox_messages(
              id TEXT PRIMARY KEY, sender TEXT NOT NULL, recipient TEXT NOT NULL,
              content TEXT NOT NULL, status TEXT NOT NULL
                CHECK(status IN ('pending','delivered','expired')),
              created_at TEXT NOT NULL, expires_at TEXT);
            CREATE INDEX IF NOT EXISTS idx_mailbox_pending
              ON mailbox_messages(status,recipient,created_at);
            CREATE TABLE IF NOT EXISTS shared_snapshots(
              scope TEXT PRIMARY KEY, data_json TEXT NOT NULL, updated_at TEXT NOT NULL);
        """)
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def send(self, sender: str, recipient: str, content: str,
             *, ttl_seconds: int | None = DEFAULT_TTL_SECONDS) -> dict[str, Any]:
        if not sender or not recipient:
            raise ValueError("Sender and recipient are required")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Message content is required")
        if ttl_seconds is not None and ttl_seconds < 1:
            raise ValueError("Message TTL must be positive or null")
        created = _now()
        expires = created + timedelta(seconds=ttl_seconds) if ttl_seconds is not None else None
        message_id = str(uuid.uuid4())
        with self.connection:
            self.connection.execute(
                "INSERT INTO mailbox_messages VALUES(?,?,?,?,'pending',?,?)",
                (message_id, sender, recipient, content, created.isoformat(),
                 expires.isoformat() if expires else None),
            )
        return self.get(message_id)

    def expire(self) -> int:
        with self.connection:
            cursor = self.connection.execute(
                "UPDATE mailbox_messages SET status='expired' "
                "WHERE status='pending' AND expires_at IS NOT NULL AND expires_at<=?",
                (_now().isoformat(),),
            )
        return cursor.rowcount

    def pending(self) -> list[dict[str, Any]]:
        self.expire()
        return [dict(row) for row in self.connection.execute(
            "SELECT * FROM mailbox_messages WHERE status='pending' ORDER BY created_at,id")]

    def delivered(self, message_id: str) -> bool:
        with self.connection:
            cursor = self.connection.execute(
                "UPDATE mailbox_messages SET status='delivered' WHERE id=? AND status='pending'",
                (message_id,),
            )
        return cursor.rowcount == 1

    def get(self, message_id: str) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT * FROM mailbox_messages WHERE id=?", (message_id,)).fetchone()
        if row is None:
            raise KeyError(message_id)
        return dict(row)

    def observed_snapshot(self, scope: str) -> Any | None:
        row = self.connection.execute(
            "SELECT data_json FROM shared_snapshots WHERE scope=?", (scope,)).fetchone()
        return None if row is None else json.loads(row["data_json"])

    def save_observed_snapshot(self, scope: str, data: Any) -> None:
        with self.connection:
            self.connection.execute("""
                INSERT INTO shared_snapshots(scope,data_json,updated_at) VALUES(?,?,?)
                ON CONFLICT(scope) DO UPDATE SET data_json=excluded.data_json,
                  updated_at=excluded.updated_at
            """, (scope, json.dumps(data, sort_keys=True, separators=(",", ":")),
                  _now().isoformat()))
