from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .domain import Identity, WakeEvent


_SAFE_JOURNAL_FIELDS: dict[str, tuple[str, ...]] = {
    "communication.delivered": ("message_id", "delivered", "spontaneous"),
    "communication.failed": ("message_id", "delivered", "spontaneous"),
    "communication.rejected": ("message_id", "delivered", "spontaneous"),
    "context.assembled": (
        "characters", "pending_intentions", "recent_messages"),
    "intention.created": ("intention_id",),
    "intention.updated": ("intention_id", "status"),
    "model.responded": ("tool_call_count", "has_message", "input_tokens", "output_tokens"),
    "tool.called": ("name",),
    "tool.completed": ("name",),
    "wake.failed": ("error_type",),
    "wake.finished": ("status", "duration_seconds", "model_calls"),
    "wake.sleeping": ("status",),
    "wakeup.scheduled": ("schedule_id", "due_at"),
}
_SAFE_JOURNAL_EVENT_LIMIT = 50
MAX_OWNER_GUIDANCE_ENTRY_BYTES = 4096
MAX_ACTIVE_OWNER_GUIDANCE_COUNT = 16
MAX_ACTIVE_OWNER_GUIDANCE_BYTES = 32768


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _validate_owner_guidance_projection(entries: list[dict[str, Any]]) -> None:
    if len(entries) > MAX_ACTIVE_OWNER_GUIDANCE_COUNT:
        raise ValueError(
            f"Active Owner guidance exceeds {MAX_ACTIVE_OWNER_GUIDANCE_COUNT} entries; "
            "replace or remove an existing entry")
    if any(len(entry["content"].encode("utf-8")) > MAX_OWNER_GUIDANCE_ENTRY_BYTES
           for entry in entries):
        raise ValueError(
            f"Owner guidance exceeds {MAX_OWNER_GUIDANCE_ENTRY_BYTES} UTF-8 bytes")
    encoded = json.dumps(
        entries, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if len(encoded) > MAX_ACTIVE_OWNER_GUIDANCE_BYTES:
        raise ValueError(
            f"Active Owner guidance exceeds {MAX_ACTIVE_OWNER_GUIDANCE_BYTES} "
            "serialized UTF-8 bytes; replace or remove an existing entry")


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
        INSERT INTO schema_version(version) SELECT 13 WHERE NOT EXISTS (SELECT 1 FROM schema_version);
        CREATE TABLE IF NOT EXISTS identities(
          role TEXT PRIMARY KEY CHECK(role IN ('resident','owner')), id TEXT NOT NULL UNIQUE,
          address_name TEXT NOT NULL, personality TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL);
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
        CREATE TABLE IF NOT EXISTS agent_session_bindings(
          provider TEXT PRIMARY KEY, session_id TEXT NOT NULL, agent_id TEXT,
          last_turn_id TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS memory_records(
          id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL
            CHECK(status IN ('active','superseded','invalidated')),
          current_revision INTEGER NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS memory_revisions(
          memory_id TEXT NOT NULL REFERENCES memory_records(id), revision INTEGER NOT NULL,
          operation TEXT NOT NULL CHECK(operation IN ('create','update','supersede','invalidate')),
          content TEXT NOT NULL, rationale TEXT NOT NULL DEFAULT '', confidence REAL,
          operation_key TEXT NOT NULL, created_at TEXT NOT NULL,
          PRIMARY KEY(memory_id,revision), UNIQUE(operation_key));
        CREATE TABLE IF NOT EXISTS memory_provenance(
          memory_id TEXT NOT NULL, revision INTEGER NOT NULL, source_session_id TEXT,
          source_item_id TEXT, source_type TEXT NOT NULL, source_timestamp TEXT,
          excerpt TEXT, content_hash TEXT,
          FOREIGN KEY(memory_id,revision) REFERENCES memory_revisions(memory_id,revision));
        CREATE TABLE IF NOT EXISTS owner_guidance(
          id TEXT PRIMARY KEY, content TEXT NOT NULL, status TEXT NOT NULL
            CHECK(status IN ('active','superseded','removed')),
          revision INTEGER NOT NULL, source_session_id TEXT, source_item_id TEXT,
          created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS owner_guidance_revisions(
          guidance_id TEXT NOT NULL REFERENCES owner_guidance(id), revision INTEGER NOT NULL,
          operation TEXT NOT NULL CHECK(operation IN ('set','remove')), content TEXT NOT NULL,
          source_session_id TEXT, source_item_id TEXT, created_at TEXT NOT NULL,
          PRIMARY KEY(guidance_id,revision));
        CREATE TABLE IF NOT EXISTS curator_checkpoints(
          provider TEXT NOT NULL, session_id TEXT NOT NULL, cursor TEXT,
          last_item_id TEXT, handover_draft TEXT, updated_at TEXT NOT NULL,
          PRIMARY KEY(provider,session_id));
        CREATE TABLE IF NOT EXISTS curator_operations(
          operation_key TEXT PRIMARY KEY, session_id TEXT NOT NULL,
          cursor TEXT, applied_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS curator_jobs(
          id TEXT PRIMARY KEY, session_id TEXT NOT NULL, source_cursor TEXT,
          status TEXT NOT NULL CHECK(status IN ('claimed','completed','failed')),
          attempts INTEGER NOT NULL, last_error_type TEXT,
          created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS session_handovers(
          id TEXT PRIMARY KEY, old_session_id TEXT NOT NULL, new_session_id TEXT,
          content TEXT NOT NULL, created_at TEXT NOT NULL, expires_at TEXT NOT NULL,
          consumed_at TEXT);
        CREATE TABLE IF NOT EXISTS session_protocol_descriptors(
          provider TEXT NOT NULL, session_id TEXT NOT NULL, descriptor_json TEXT NOT NULL,
          created_at TEXT NOT NULL, PRIMARY KEY(provider,session_id));
        CREATE TABLE IF NOT EXISTS session_rollovers(
          id TEXT PRIMARY KEY, provider TEXT NOT NULL, old_session_id TEXT,
          new_session_id TEXT, reason TEXT NOT NULL, requested_by TEXT NOT NULL,
          finalization_status TEXT NOT NULL, status TEXT NOT NULL,
          created_at TEXT NOT NULL, completed_at TEXT);
        CREATE TABLE IF NOT EXISTS agent_tool_actions(
          provider TEXT NOT NULL, session_id TEXT NOT NULL, turn_id TEXT NOT NULL,
          call_id TEXT NOT NULL, name TEXT NOT NULL, arguments_json TEXT NOT NULL,
          status TEXT NOT NULL CHECK(status IN ('pending','completed')),
          output_json TEXT, attachments_ephemeral INTEGER NOT NULL DEFAULT 0
            CHECK(attachments_ephemeral IN (0,1)), created_at TEXT NOT NULL, completed_at TEXT,
          PRIMARY KEY(provider,session_id,call_id));
        CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_wake_runs_started ON wake_runs(started_at DESC);
        CREATE INDEX IF NOT EXISTS idx_journal_run_sequence ON journal(run_id, sequence);
        CREATE INDEX IF NOT EXISTS idx_schedules_due ON scheduled_wakeups(status, due_at);
        CREATE INDEX IF NOT EXISTS idx_memory_status_updated ON memory_records(status,updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_memory_provenance_source
          ON memory_provenance(source_session_id,source_item_id);
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
        with self.connection:
            self.connection.execute("DROP TABLE IF EXISTS memories")
        action_columns = {
            row["name"] for row in self.connection.execute("PRAGMA table_info(agent_tool_actions)")
        }
        if "attachments_ephemeral" not in action_columns:
            with self.connection:
                self.connection.execute(
                    "ALTER TABLE agent_tool_actions ADD COLUMN "
                    "attachments_ephemeral INTEGER NOT NULL DEFAULT 0 "
                    "CHECK(attachments_ephemeral IN (0,1))")
        checkpoint_columns = {
            row["name"] for row in self.connection.execute("PRAGMA table_info(curator_checkpoints)")
        }
        if "handover_draft" not in checkpoint_columns:
            with self.connection:
                self.connection.execute(
                    "ALTER TABLE curator_checkpoints ADD COLUMN handover_draft TEXT")
        self.connection.execute("UPDATE schema_version SET version=13")
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

    def agent_session_binding(self, provider: str) -> dict[str, Any] | None:
        row = self.connection.execute("""
            SELECT provider,session_id,agent_id,last_turn_id,created_at,updated_at
            FROM agent_session_bindings WHERE provider=?
        """, (provider,)).fetchone()
        return None if row is None else dict(row)

    def save_agent_session_binding(self, provider: str, session_id: str,
                                   agent_id: str | None, last_turn_id: str | None) -> None:
        now = utc_now()
        with self.connection:
            self.connection.execute("""
                INSERT INTO agent_session_bindings(
                  provider,session_id,agent_id,last_turn_id,created_at,updated_at)
                VALUES(?,?,?,?,?,?)
                ON CONFLICT(provider) DO UPDATE SET
                  session_id=excluded.session_id, agent_id=excluded.agent_id,
                  last_turn_id=excluded.last_turn_id, updated_at=excluded.updated_at
            """, (provider, session_id, agent_id, last_turn_id, now, now))

    def save_session_protocol(self, provider: str, session_id: str,
                              descriptor: dict[str, Any]) -> None:
        encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
        with self.connection:
            self.connection.execute("""
                INSERT INTO session_protocol_descriptors(provider,session_id,descriptor_json,created_at)
                VALUES(?,?,?,?) ON CONFLICT(provider,session_id) DO UPDATE SET
                  descriptor_json=excluded.descriptor_json
            """, (provider, session_id, encoded, utc_now()))

    def session_protocol(self, provider: str, session_id: str) -> dict[str, Any] | None:
        row = self.connection.execute("""
            SELECT descriptor_json FROM session_protocol_descriptors
            WHERE provider=? AND session_id=?
        """, (provider, session_id)).fetchone()
        return None if row is None else json.loads(row["descriptor_json"])

    def begin_session_rollover(self, provider: str, old_session_id: str | None,
                               reason: str, requested_by: str = "runtime") -> str:
        pending = self.connection.execute("""
            SELECT id FROM session_rollovers WHERE provider=? AND old_session_id IS ?
              AND reason=? AND status='pending' ORDER BY created_at DESC LIMIT 1
        """, (provider, old_session_id, reason)).fetchone()
        if pending is not None:
            return pending["id"]
        rollover_id = str(uuid.uuid4())
        with self.connection:
            self.connection.execute("""
                INSERT INTO session_rollovers(
                  id,provider,old_session_id,reason,requested_by,finalization_status,status,created_at)
                VALUES(?,?,?,?,?,'pending','pending',?)
            """, (rollover_id, provider, old_session_id, reason, requested_by, utc_now()))
        return rollover_id

    def recover_session_rollovers(self, provider: str) -> int:
        """Finish the audit edge if a crash occurred after binding the new session."""
        binding = self.agent_session_binding(provider)
        if binding is None:
            return 0
        now = utc_now()
        with self.connection:
            result = self.connection.execute("""
                UPDATE session_rollovers SET new_session_id=?,status='completed',
                  finalization_status=CASE WHEN finalization_status='pending'
                    THEN 'unknown_after_restart' ELSE finalization_status END,completed_at=?
                WHERE provider=? AND status='pending'
                  AND (old_session_id IS NULL OR old_session_id<>?)
            """, (binding["session_id"], now, provider, binding["session_id"]))
        return result.rowcount

    def complete_session_rollover(self, rollover_id: str, new_session_id: str,
                                  finalization_status: str = "completed") -> None:
        with self.connection:
            self.connection.execute("""
                UPDATE session_rollovers SET new_session_id=?,finalization_status=?,
                  status='completed',completed_at=? WHERE id=? AND status='pending'
            """, (new_session_id, finalization_status, utc_now(), rollover_id))

    def fail_session_rollover(self, rollover_id: str,
                              finalization_status: str = "failed") -> None:
        with self.connection:
            self.connection.execute("""
                UPDATE session_rollovers SET finalization_status=?,status='failed',completed_at=?
                WHERE id=? AND status='pending'
            """, (finalization_status, utc_now(), rollover_id))

    def curator_checkpoint(self, provider: str, session_id: str) -> dict[str, Any] | None:
        row = self.connection.execute("""
            SELECT cursor,last_item_id,handover_draft,updated_at FROM curator_checkpoints
            WHERE provider=? AND session_id=?
        """, (provider, session_id)).fetchone()
        return None if row is None else dict(row)

    def apply_curator_batch(self, provider: str, session_id: str, cursor: str | None,
                            last_item_id: str | None, operation_key: str,
                            mutations: list[dict[str, Any]],
                            handover_draft: str | None = None) -> bool:
        """Atomically apply validated curator decisions and advance its source checkpoint."""
        if any(not mutation.get("provenance") for mutation in mutations):
            raise ValueError("Durable Curator memory requires verified provenance")
        now = utc_now()
        with self.connection:
            claimed = self.connection.execute("""
                INSERT INTO curator_operations(operation_key,session_id,cursor,applied_at)
                VALUES(?,?,?,?) ON CONFLICT DO NOTHING
            """, (operation_key, session_id, cursor, now))
            if claimed.rowcount == 0:
                return False
            for index, mutation in enumerate(mutations):
                memory_id = str(mutation.get("memory_id") or uuid.uuid4())
                operation = mutation["operation"]
                existing = self.connection.execute(
                    "SELECT current_revision FROM memory_records WHERE id=?", (memory_id,)
                ).fetchone()
                revision = (existing["current_revision"] + 1) if existing else 1
                if operation != "create" and existing is None:
                    raise ValueError(f"Memory mutation targets an unknown record: {memory_id}")
                status = {"create": "active", "update": "active",
                          "supersede": "superseded", "invalidate": "invalidated"}[operation]
                if existing is None:
                    self.connection.execute("""
                        INSERT INTO memory_records(id,kind,status,current_revision,created_at,updated_at)
                        VALUES(?,?,?,?,?,?)
                    """, (memory_id, mutation.get("kind", "experience"), status, revision, now, now))
                else:
                    self.connection.execute("""
                        UPDATE memory_records SET status=?,current_revision=?,updated_at=? WHERE id=?
                    """, (status, revision, now, memory_id))
                mutation_key = f"{operation_key}:{index}"
                self.connection.execute("""
                    INSERT INTO memory_revisions(
                      memory_id,revision,operation,content,rationale,confidence,operation_key,created_at)
                    VALUES(?,?,?,?,?,?,?,?)
                """, (memory_id, revision, operation, mutation.get("content", ""),
                      mutation.get("rationale", ""), mutation.get("confidence"), mutation_key, now))
                for evidence in mutation.get("provenance", []):
                    self.connection.execute("""
                        INSERT INTO memory_provenance(
                          memory_id,revision,source_session_id,source_item_id,source_type,
                          source_timestamp,excerpt,content_hash) VALUES(?,?,?,?,?,?,?,?)
                    """, (memory_id, revision, session_id,
                          evidence.get("item_id"), evidence.get("source_type", "session_item"),
                          evidence.get("timestamp"), evidence.get("excerpt"),
                          evidence.get("content_hash")))
            self.connection.execute("""
                INSERT INTO curator_checkpoints(
                  provider,session_id,cursor,last_item_id,handover_draft,updated_at)
                VALUES(?,?,?,?,?,?) ON CONFLICT(provider,session_id) DO UPDATE SET
                  cursor=excluded.cursor,last_item_id=excluded.last_item_id,
                  handover_draft=COALESCE(excluded.handover_draft,curator_checkpoints.handover_draft),
                  updated_at=excluded.updated_at
            """, (provider, session_id, cursor, last_item_id, handover_draft, now))
        return True

    def claim_curator_job(self, job_id: str, session_id: str,
                          source_cursor: str | None) -> bool:
        now = utc_now()
        with self.connection:
            row = self.connection.execute(
                "SELECT status FROM curator_jobs WHERE id=?", (job_id,)).fetchone()
            if row is not None and row["status"] == "completed":
                return False
            self.connection.execute("""
                INSERT INTO curator_jobs(
                  id,session_id,source_cursor,status,attempts,created_at,updated_at)
                VALUES(?,?,?,'claimed',1,?,?)
                ON CONFLICT(id) DO UPDATE SET status='claimed',attempts=attempts+1,
                  last_error_type=NULL,updated_at=excluded.updated_at
            """, (job_id, session_id, source_cursor, now, now))
        return True

    def finish_curator_job(self, job_id: str) -> None:
        with self.connection:
            self.connection.execute("""
                UPDATE curator_jobs SET status='completed',updated_at=? WHERE id=?
            """, (utc_now(), job_id))

    def fail_curator_job(self, job_id: str, error_type: str) -> None:
        with self.connection:
            self.connection.execute("""
                UPDATE curator_jobs SET status='failed',last_error_type=?,updated_at=? WHERE id=?
            """, (error_type[:100], utc_now(), job_id))

    def search_memories(self, query: str = "", *, limit: int = 10,
                        offset: int = 0) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 20))
        offset = max(0, min(offset, 1000))
        parameters: list[Any] = []
        where = "WHERE records.status='active'"
        if query:
            escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            where += " AND lower(revisions.content) LIKE lower(?) ESCAPE '\\'"
            parameters.append(f"%{escaped}%")
        parameters.extend((limit, offset))
        rows = self.connection.execute(f"""
            SELECT records.id,records.kind,revisions.content,revisions.confidence,
                   records.updated_at FROM memory_records records
            JOIN memory_revisions revisions ON revisions.memory_id=records.id
              AND revisions.revision=records.current_revision
            {where} ORDER BY records.updated_at DESC,records.id LIMIT ? OFFSET ?
        """, parameters).fetchall()
        return [dict(row) for row in rows]

    def memory_awareness(self, *, limit: int = 8) -> list[dict[str, Any]]:
        """Compact type-level index for replacement-session bootstrap."""
        rows = self.connection.execute("""
            SELECT kind,count(*) AS active_count,max(updated_at) AS most_recent_at
            FROM memory_records WHERE status='active' GROUP BY kind
            ORDER BY most_recent_at DESC,kind LIMIT ?
        """, (max(1, min(limit, 20)),)).fetchall()
        return [dict(row) for row in rows]

    def memory(self, memory_id: str) -> dict[str, Any] | None:
        row = self.connection.execute("""
            SELECT records.id,records.kind,records.status,records.current_revision,
                   revisions.content,revisions.rationale,revisions.confidence,records.updated_at
            FROM memory_records records JOIN memory_revisions revisions
              ON revisions.memory_id=records.id AND revisions.revision=records.current_revision
            WHERE records.id=?
        """, (memory_id,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["provenance"] = [dict(item) for item in self.connection.execute("""
            SELECT source_session_id,source_item_id,source_type,source_timestamp,excerpt,content_hash
            FROM memory_provenance WHERE memory_id=? AND revision=?
        """, (memory_id, row["current_revision"]))]
        return result

    def active_owner_guidance(self) -> list[dict[str, Any]]:
        entries = [dict(row) for row in self.connection.execute("""
            SELECT id,content,revision,updated_at FROM owner_guidance
            WHERE status='active' ORDER BY updated_at,id
        """)]
        _validate_owner_guidance_projection(entries)
        return entries

    def set_owner_guidance(self, content: str, *, guidance_id: str | None = None,
                           source_session_id: str | None = None,
                           source_item_id: str | None = None) -> str:
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Owner guidance must be nonempty text")
        if len(content.encode("utf-8")) > MAX_OWNER_GUIDANCE_ENTRY_BYTES:
            raise ValueError(
                f"Owner guidance exceeds {MAX_OWNER_GUIDANCE_ENTRY_BYTES} UTF-8 bytes")
        guidance_id = guidance_id or str(uuid.uuid4())
        now = utc_now()
        with self.connection:
            current = self.connection.execute(
                "SELECT revision FROM owner_guidance WHERE id=?", (guidance_id,)).fetchone()
            revision = (current["revision"] + 1) if current else 1
            prospective = [dict(row) for row in self.connection.execute("""
                SELECT id,content,revision,updated_at FROM owner_guidance
                WHERE status='active' AND id<>? ORDER BY updated_at,id
            """, (guidance_id,))]
            prospective.append({"id": guidance_id, "content": content,
                                "revision": revision, "updated_at": now})
            prospective.sort(key=lambda entry: (entry["updated_at"], entry["id"]))
            _validate_owner_guidance_projection(prospective)
            self.connection.execute("""
                INSERT INTO owner_guidance(id,content,status,revision,source_session_id,
                  source_item_id,created_at,updated_at) VALUES(?,?,'active',?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET content=excluded.content,status='active',
                  revision=excluded.revision,source_session_id=excluded.source_session_id,
                  source_item_id=excluded.source_item_id,updated_at=excluded.updated_at
            """, (guidance_id, content, revision,
                  source_session_id, source_item_id, now, now))
            self.connection.execute("""
                INSERT INTO owner_guidance_revisions(
                  guidance_id,revision,operation,content,source_session_id,source_item_id,created_at)
                VALUES(?,?,'set',?,?,?,?)
            """, (guidance_id, revision, content, source_session_id, source_item_id, now))
        return guidance_id

    def remove_owner_guidance(self, guidance_id: str) -> bool:
        with self.connection:
            current = self.connection.execute("""
                SELECT content,revision,source_session_id,source_item_id FROM owner_guidance
                WHERE id=? AND status='active'
            """, (guidance_id,)).fetchone()
            if current is None:
                return False
            result = self.connection.execute("""
                UPDATE owner_guidance SET status='removed',revision=revision+1,updated_at=?
                WHERE id=? AND status='active'
            """, (utc_now(), guidance_id))
            self.connection.execute("""
                INSERT INTO owner_guidance_revisions(
                  guidance_id,revision,operation,content,source_session_id,source_item_id,created_at)
                VALUES(?,?,'remove',?,?,?,?)
            """, (guidance_id, current["revision"] + 1, current["content"],
                  current["source_session_id"], current["source_item_id"], utc_now()))
        return result.rowcount == 1

    def create_handover(self, old_session_id: str, content: str, expires_at: str) -> str:
        handover_id = str(uuid.uuid4())
        with self.connection:
            self.connection.execute("""
                INSERT INTO session_handovers(id,old_session_id,content,created_at,expires_at)
                VALUES(?,?,?,?,?)
            """, (handover_id, old_session_id, content, utc_now(), expires_at))
        return handover_id

    def consume_handover(self, handover_id: str, new_session_id: str) -> str | None:
        now = utc_now()
        with self.connection:
            row = self.connection.execute("""
                SELECT content FROM session_handovers WHERE id=? AND consumed_at IS NULL
                  AND expires_at>?
            """, (handover_id, now)).fetchone()
            if row is None:
                return None
            self.connection.execute("""
                UPDATE session_handovers SET new_session_id=?,consumed_at=? WHERE id=?
            """, (new_session_id, now, handover_id))
        return row["content"]

    def begin_agent_tool_action(self, provider: str, session_id: str, turn_id: str,
                                call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        encoded = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
        with self.connection:
            cursor = self.connection.execute("""
                INSERT INTO agent_tool_actions(
                  provider,session_id,turn_id,call_id,name,arguments_json,status,created_at)
                VALUES(?,?,?,?,?,?,'pending',?) ON CONFLICT DO NOTHING
            """, (provider, session_id, turn_id, call_id, name, encoded, utc_now()))
            row = self.connection.execute("""
                SELECT turn_id,name,arguments_json,status,output_json,attachments_ephemeral
                FROM agent_tool_actions WHERE provider=? AND session_id=? AND call_id=?
            """, (provider, session_id, call_id)).fetchone()
        if row["turn_id"] != turn_id or row["name"] != name or row["arguments_json"] != encoded:
            raise RuntimeError("Agents function call id was reused with different action data")
        return {
            "claimed": cursor.rowcount == 1, "status": row["status"],
            "output": json.loads(row["output_json"]) if row["output_json"] else None,
            "attachments_ephemeral": bool(row["attachments_ephemeral"]),
        }

    def complete_agent_tool_action(self, provider: str, session_id: str,
                                   call_id: str, output: dict[str, Any],
                                   attachments_ephemeral: bool = False) -> None:
        encoded = json.dumps(output, sort_keys=True, separators=(",", ":"))
        with self.connection:
            self.connection.execute("""
                UPDATE agent_tool_actions SET status='completed',output_json=?,
                  attachments_ephemeral=?,completed_at=?
                WHERE provider=? AND session_id=? AND call_id=? AND status='pending'
            """, (encoded, int(attachments_ephemeral), utc_now(), provider, session_id, call_id))

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
            # Configuration changes describe the same durable individual. Keep
            # the UUID while refreshing the configured display/prompt metadata.
            self.connection.execute(
                "UPDATE identities SET address_name=?,personality=? WHERE role='resident'",
                (resident_name, personality),
            )
            self.connection.execute(
                "UPDATE identities SET address_name=? WHERE role='owner'", (owner_name,))
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

    def search_messages(self, *, query: str = "", direction: str | None = None,
                        from_time: str | None = None, to_time: str | None = None,
                        limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if query:
            escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            clauses.append("lower(content) LIKE lower(?) ESCAPE '\\'")
            parameters.append(f"%{escaped}%")
        if direction is not None:
            clauses.append("direction=?")
            parameters.append(direction)
        if from_time is not None:
            clauses.append("created_at>=?")
            parameters.append(from_time)
        if to_time is not None:
            clauses.append("created_at<=?")
            parameters.append(to_time)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.extend((limit, offset))
        rows = self.connection.execute(f"""
            SELECT id,direction,content,delivery_status,created_at
            FROM messages{where}
            ORDER BY created_at DESC,id DESC LIMIT ? OFFSET ?
        """, parameters).fetchall()
        return [dict(row) for row in rows]

    def spontaneous_count_since(self, since: str) -> int:
        return int(self.connection.execute(
            "SELECT count(*) FROM messages WHERE direction='outbound' AND spontaneous=1 AND delivery_status='delivered' AND created_at>=?",
            (since,),).fetchone()[0])

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

    def wake_history(self, *, query: str = "", source: str | None = None,
                     status: str | None = None, from_time: str | None = None,
                     to_time: str | None = None, limit: int = 10,
                     offset: int = 0, exclude_run_id: str | None = None) -> list[dict[str, Any]]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if query:
            escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            clauses.append("lower(wake_reason) LIKE lower(?) ESCAPE '\\'")
            parameters.append(f"%{escaped}%")
        for column, value in (("wake_source", source), ("status", status)):
            if value is not None:
                clauses.append(f"{column}=?")
                parameters.append(value)
        if exclude_run_id is not None:
            clauses.append("id<>?")
            parameters.append(exclude_run_id)
        if from_time is not None:
            clauses.append("started_at>=?")
            parameters.append(from_time)
        if to_time is not None:
            clauses.append("started_at<=?")
            parameters.append(to_time)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.extend((limit, offset))
        runs = self.connection.execute(f"""
            SELECT id,started_at,finished_at,wake_reason,wake_source,status,
                   duration_seconds,model_calls
            FROM wake_runs{where}
            ORDER BY started_at DESC,id DESC LIMIT ? OFFSET ?
        """, parameters).fetchall()
        history = []
        for run in runs:
            events, events_truncated = self._safe_journal_events(run["id"])
            history.append({**dict(run), "events": events, "events_truncated": events_truncated})
        return history

    def _safe_journal_events(self, run_id: str) -> tuple[list[dict[str, Any]], bool]:
        event_types = tuple(_SAFE_JOURNAL_FIELDS)
        placeholders = ",".join("?" for _ in event_types)
        rows = self.connection.execute(f"""
            SELECT event_type,occurred_at,data_json FROM journal
            WHERE run_id=? AND event_type IN ({placeholders})
            ORDER BY sequence DESC LIMIT ?
        """, (run_id, *event_types, _SAFE_JOURNAL_EVENT_LIMIT + 1)).fetchall()
        events_truncated = len(rows) > _SAFE_JOURNAL_EVENT_LIMIT
        events: list[dict[str, Any]] = []
        for row in reversed(rows[:_SAFE_JOURNAL_EVENT_LIMIT]):
            allowed_fields = _SAFE_JOURNAL_FIELDS.get(row["event_type"])
            if allowed_fields is None:
                continue
            try:
                data = json.loads(row["data_json"])
            except (json.JSONDecodeError, TypeError):
                data = {}
            if not isinstance(data, dict):
                data = {}
            event = {"type": row["event_type"], "occurred_at": row["occurred_at"]}
            event.update({
                field: data[field] for field in allowed_fields
                if field in data and (data[field] is None or isinstance(data[field], (str, int, float, bool)))
            })
            events.append(event)
        return events, events_truncated

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
