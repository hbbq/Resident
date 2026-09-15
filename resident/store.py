from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .domain import Identity, WakeEvent


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self._migrate()

    def close(self) -> None:
        self.connection.close()

    def _migrate(self) -> None:
        self.connection.executescript("""
        CREATE TABLE IF NOT EXISTS schema_version(version INTEGER NOT NULL);
        INSERT INTO schema_version(version) SELECT 5 WHERE NOT EXISTS (SELECT 1 FROM schema_version);
        CREATE TABLE IF NOT EXISTS identities(
          role TEXT PRIMARY KEY CHECK(role IN ('resident','owner')), id TEXT NOT NULL UNIQUE,
          address_name TEXT NOT NULL, personality TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS memories(
          id TEXT PRIMARY KEY, content TEXT NOT NULL, source TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS intentions(
          id TEXT PRIMARY KEY, content TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('pending','completed','cancelled')),
          created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS messages(
          id TEXT PRIMARY KEY, direction TEXT NOT NULL CHECK(direction IN ('inbound','outbound')),
          sender_id TEXT NOT NULL, content TEXT NOT NULL, spontaneous INTEGER NOT NULL DEFAULT 0,
          delivery_status TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS wake_runs(
          id TEXT PRIMARY KEY, event_id TEXT NOT NULL, started_at TEXT NOT NULL, finished_at TEXT,
          wake_reason TEXT NOT NULL, wake_source TEXT NOT NULL, status TEXT NOT NULL,
          duration_seconds REAL, model_calls INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS journal(
          sequence INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE, run_id TEXT,
          event_type TEXT NOT NULL, occurred_at TEXT NOT NULL, data_json TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS scheduled_wakeups(
          id TEXT PRIMARY KEY, due_at TEXT NOT NULL, reason TEXT NOT NULL, context_json TEXT NOT NULL,
          status TEXT NOT NULL CHECK(status IN ('pending','claimed','completed','failed')), created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS observed_snapshots(
          scope TEXT PRIMARY KEY, data_json TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS telegram_owner_updates(
          bot_identity TEXT NOT NULL, update_id INTEGER NOT NULL,
          message_id TEXT NOT NULL UNIQUE REFERENCES messages(id),
          PRIMARY KEY(bot_identity, update_id));
        CREATE TABLE IF NOT EXISTS owner_message_processing(
          message_id TEXT PRIMARY KEY REFERENCES messages(id),
          status TEXT NOT NULL CHECK(status IN ('pending','completed')),
          completed_at TEXT);
        CREATE INDEX IF NOT EXISTS idx_memories_updated ON memories(updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_schedules_due ON scheduled_wakeups(status, due_at);
        """)
        # An already-provisioned Resident predates capability snapshots. Seed an
        # empty baseline so its first run with this feature sees the currently
        # available capabilities as additions. A genuinely new database has no
        # Resident identity yet and will establish its first baseline silently.
        existing_resident = self.connection.execute(
            "SELECT 1 FROM identities WHERE role='resident'"
        ).fetchone()
        existing_capability_snapshot = self.connection.execute(
            "SELECT 1 FROM observed_snapshots WHERE scope='runtime.capabilities'"
        ).fetchone()
        if existing_resident is not None and existing_capability_snapshot is None:
            self.connection.execute(
                "INSERT INTO observed_snapshots(scope,data_json,updated_at) VALUES(?,?,?)",
                ("runtime.capabilities", "{}", utc_now()),
            )
        table_sql = self.connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='scheduled_wakeups'"
        ).fetchone()[0]
        if "'failed'" not in table_sql:
            with self.connection:
                self.connection.execute("DROP INDEX IF EXISTS idx_schedules_due")
                self.connection.execute("ALTER TABLE scheduled_wakeups RENAME TO scheduled_wakeups_v1")
                self.connection.execute("""
                    CREATE TABLE scheduled_wakeups(
                      id TEXT PRIMARY KEY, due_at TEXT NOT NULL, reason TEXT NOT NULL,
                      context_json TEXT NOT NULL, status TEXT NOT NULL
                      CHECK(status IN ('pending','claimed','completed','failed')), created_at TEXT NOT NULL)
                """)
                self.connection.execute("""
                    INSERT INTO scheduled_wakeups(id,due_at,reason,context_json,status,created_at)
                    SELECT id,due_at,reason,context_json,status,created_at FROM scheduled_wakeups_v1
                """)
                self.connection.execute("DROP TABLE scheduled_wakeups_v1")
                self.connection.execute(
                    "CREATE INDEX idx_schedules_due ON scheduled_wakeups(status, due_at)")
        telegram_columns = self.connection.execute(
            "PRAGMA table_info(telegram_owner_updates)"
        ).fetchall()
        if "bot_identity" not in {row["name"] for row in telegram_columns}:
            with self.connection:
                self.connection.execute(
                    "ALTER TABLE telegram_owner_updates RENAME TO telegram_owner_updates_v1")
                self.connection.execute("""
                    CREATE TABLE telegram_owner_updates(
                      bot_identity TEXT NOT NULL, update_id INTEGER NOT NULL,
                      message_id TEXT NOT NULL UNIQUE REFERENCES messages(id),
                      PRIMARY KEY(bot_identity, update_id))
                """)
                self.connection.execute("""
                    INSERT INTO telegram_owner_updates(bot_identity,update_id,message_id)
                    SELECT 'legacy',update_id,message_id FROM telegram_owner_updates_v1
                """)
                self.connection.execute("DROP TABLE telegram_owner_updates_v1")
        self.connection.execute("UPDATE schema_version SET version=5")
        self.connection.execute("UPDATE scheduled_wakeups SET status='pending' WHERE status='claimed'")
        self.connection.commit()

    def observed_snapshot(self, scope: str) -> Any | None:
        row = self.connection.execute(
            "SELECT data_json FROM observed_snapshots WHERE scope=?", (scope,)).fetchone()
        return None if row is None else json.loads(row["data_json"])

    def save_observed_snapshot(self, scope: str, data: Any) -> None:
        encoded = json.dumps(data, sort_keys=True, separators=(",", ":"))
        with self.connection:
            self.connection.execute("""
                INSERT INTO observed_snapshots(scope,data_json,updated_at) VALUES(?,?,?)
                ON CONFLICT(scope) DO UPDATE SET data_json=excluded.data_json,
                    updated_at=excluded.updated_at
            """, (scope, encoded, utc_now()))

    def provision(self, resident_name: str, owner_name: str, personality: str) -> tuple[Identity, Identity]:
        now = utc_now()
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO identities VALUES('resident',?,?,?,?)",
                (str(uuid.uuid4()), resident_name, personality, now),
            )
            self.connection.execute(
                "INSERT OR IGNORE INTO identities VALUES('owner',?,?,?,?)",
                (str(uuid.uuid4()), owner_name, "", now),
            )
        return self.identities()

    def identities(self) -> tuple[Identity, Identity]:
        rows = {r["role"]: r for r in self.connection.execute("SELECT * FROM identities")}
        if "resident" not in rows or "owner" not in rows:
            raise RuntimeError("Resident data has not been provisioned")
        return (
            Identity(rows["resident"]["id"], rows["resident"]["address_name"], rows["resident"]["personality"]),
            Identity(rows["owner"]["id"], rows["owner"]["address_name"]),
        )

    def add_message(self, direction: str, sender_id: str, content: str, *, spontaneous: bool = False,
                    delivery_status: str = "delivered") -> str:
        message_id = str(uuid.uuid4())
        with self.connection:
            self.connection.execute("INSERT INTO messages VALUES(?,?,?,?,?,?,?)", (
                message_id, direction, sender_id, content, int(spontaneous), delivery_status, utc_now()))
        return message_id

    def ingest_owner_message(self, sender_id: str, content: str) -> str:
        """Persist a canonical inbound Owner message with recoverable processing state."""
        message_id = str(uuid.uuid4())
        with self.connection:
            self.connection.execute("INSERT INTO messages VALUES(?,?,?,?,?,?,?)", (
                message_id, "inbound", sender_id, content, 0, "delivered", utc_now()))
            self.connection.execute(
                "INSERT INTO owner_message_processing(message_id,status) VALUES(?,'pending')",
                (message_id,),
            )
        return message_id

    def ingest_telegram_owner_message(self, bot_identity: str, update_id: int, sender_id: str,
                                      content: str) -> str | None:
        """Atomically record a Telegram update or recover its pending canonical message."""
        message_id = str(uuid.uuid4())
        with self.connection:
            self.connection.execute("INSERT INTO messages VALUES(?,?,?,?,?,?,?)", (
                message_id, "inbound", sender_id, content, 0, "delivered", utc_now()))
            claimed = self.connection.execute(
                "INSERT INTO telegram_owner_updates(bot_identity,update_id,message_id) VALUES(?,?,?) "
                "ON CONFLICT(bot_identity,update_id) DO NOTHING",
                (bot_identity, update_id, message_id),
            )
            if claimed.rowcount == 0:
                self.connection.execute("DELETE FROM messages WHERE id=?", (message_id,))
                pending = self.connection.execute("""
                    SELECT telegram_owner_updates.message_id
                    FROM telegram_owner_updates
                    JOIN owner_message_processing
                      ON owner_message_processing.message_id=telegram_owner_updates.message_id
                    WHERE telegram_owner_updates.bot_identity=?
                      AND telegram_owner_updates.update_id=?
                      AND owner_message_processing.status='pending'
                """, (bot_identity, update_id)).fetchone()
                return None if pending is None else pending["message_id"]
            self.connection.execute(
                "INSERT INTO owner_message_processing(message_id,status) VALUES(?,'pending')",
                (message_id,),
            )
        return message_id

    def pending_owner_messages(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.connection.execute("""
            SELECT messages.id, messages.content, messages.created_at
            FROM owner_message_processing
            JOIN messages ON messages.id=owner_message_processing.message_id
            WHERE owner_message_processing.status='pending'
            ORDER BY messages.created_at, messages.id
        """)]

    def update_message_delivery_status(self, message_id: str, delivery_status: str) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE messages SET delivery_status=? WHERE id=?", (delivery_status, message_id))

    def recent_messages(self, limit: int) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT id,direction,content,delivery_status,created_at FROM messages ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def spontaneous_count_since(self, since: str) -> int:
        return int(self.connection.execute(
            "SELECT count(*) FROM messages WHERE direction='outbound' AND spontaneous=1 AND delivery_status='delivered' AND created_at>=?",
            (since,),).fetchone()[0])

    def remember(self, content: str, source: str) -> str:
        item_id, now = str(uuid.uuid4()), utc_now()
        with self.connection:
            self.connection.execute("INSERT INTO memories VALUES(?,?,?,?,?)", (item_id, content, source, now, now))
        return item_id

    def recall(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        terms = [t for t in query.lower().split() if len(t) > 2][:8]
        if terms:
            clause = " OR ".join("lower(content) LIKE ?" for _ in terms)
            params: tuple[Any, ...] = (*(f"%{t}%" for t in terms), limit)
            rows = self.connection.execute(
                f"SELECT id,content,source,updated_at FROM memories WHERE {clause} ORDER BY updated_at DESC LIMIT ?", params
            ).fetchall()
        else:
            rows = self.connection.execute(
                "SELECT id,content,source,updated_at FROM memories ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def update_memory(self, item_id: str, content: str) -> bool:
        with self.connection:
            cursor = self.connection.execute(
                "UPDATE memories SET content=?,updated_at=? WHERE id=?", (content, utc_now(), item_id))
        return cursor.rowcount == 1

    def forget(self, item_id: str) -> bool:
        with self.connection:
            cursor = self.connection.execute("DELETE FROM memories WHERE id=?", (item_id,))
        return cursor.rowcount == 1

    def create_intention(self, content: str) -> str:
        item_id, now = str(uuid.uuid4()), utc_now()
        with self.connection:
            self.connection.execute("INSERT INTO intentions VALUES(?,?, 'pending',?,?)", (item_id, content, now, now))
        return item_id

    def update_intention(self, item_id: str, *, content: str | None, status: str | None) -> bool:
        row = self.connection.execute("SELECT content,status FROM intentions WHERE id=?", (item_id,)).fetchone()
        if row is None:
            return False
        with self.connection:
            self.connection.execute("UPDATE intentions SET content=?,status=?,updated_at=? WHERE id=?", (
                content if content is not None else row["content"], status if status is not None else row["status"],
                utc_now(), item_id))
        return True

    def pending_intentions(self, limit: int = 50) -> list[dict[str, Any]]:
        return [dict(r) for r in self.connection.execute(
            "SELECT id,content,created_at FROM intentions WHERE status='pending' ORDER BY created_at LIMIT ?", (limit,))]

    def start_run(self, event: WakeEvent) -> str:
        run_id = str(uuid.uuid4())
        with self.connection:
            self.connection.execute(
                "INSERT INTO wake_runs(id,event_id,started_at,wake_reason,wake_source,status) VALUES(?,?,?,?,?,'running')",
                (run_id, event.id, utc_now(), event.reason, event.source))
        return run_id

    def finish_run(self, run_id: str, status: str, duration: float, model_calls: int,
                   schedule_id: str | None = None,
                   owner_message_id: str | None = None) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE wake_runs SET finished_at=?,status=?,duration_seconds=?,model_calls=? WHERE id=?",
                (utc_now(), status, duration, model_calls, run_id))
            if schedule_id is not None:
                schedule_status = "completed" if status == "completed" else "failed"
                self.connection.execute(
                    "UPDATE scheduled_wakeups SET status=? WHERE id=? AND status='claimed'",
                    (schedule_status, schedule_id))
            if owner_message_id is not None and status == "completed":
                self.connection.execute("""
                    UPDATE owner_message_processing SET status='completed',completed_at=?
                    WHERE message_id=? AND status='pending'
                """, (utc_now(), owner_message_id))

    def journal(self, event_type: str, data: dict[str, Any], run_id: str | None = None) -> None:
        with self.connection:
            self.connection.execute("INSERT INTO journal(id,run_id,event_type,occurred_at,data_json) VALUES(?,?,?,?,?)", (
                str(uuid.uuid4()), run_id, event_type, utc_now(), json.dumps(data, separators=(",", ":"), default=str)))

    def schedule(self, due_at: str, reason: str, context: dict[str, Any]) -> str:
        schedule_id = str(uuid.uuid4())
        with self.connection:
            self.connection.execute("INSERT INTO scheduled_wakeups VALUES(?,?,?,?,'pending',?)", (
                schedule_id, due_at, reason, json.dumps(context, separators=(",", ":")), utc_now()))
        return schedule_id

    def claim_due_wakeups(self, now: str) -> list[dict[str, Any]]:
        with self.connection:
            rows = self.connection.execute(
                "SELECT id,due_at,reason,context_json FROM scheduled_wakeups WHERE status='pending' AND due_at<=? ORDER BY due_at", (now,)
            ).fetchall()
            if rows:
                self.connection.executemany("UPDATE scheduled_wakeups SET status='claimed' WHERE id=? AND status='pending'", ((r["id"],) for r in rows))
        return [{"id": r["id"], "due_at": r["due_at"], "reason": r["reason"],
                 "context": json.loads(r["context_json"])} for r in rows]
