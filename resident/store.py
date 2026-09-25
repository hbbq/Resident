from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import UTC, datetime, timedelta
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
    "disposition.generated": ("disposition_id", "turn_id", "validation_state", "output_count"),
    "output.queued": ("output_id", "output_type", "target"),
    "output.rejected": ("output_id", "output_type", "target", "classification"),
    "output.delivery_attempted": ("output_id", "output_type", "target", "attempt"),
    "output.delivery_succeeded": ("output_id", "output_type", "target", "attempt"),
    "output.delivery_failed": ("output_id", "output_type", "target", "attempt", "classification"),
    "output.failure_event_generated": ("output_id", "output_type", "target", "classification", "attempt_count"),
    "timeline": (
        "operation", "moment", "outcome", "duration_seconds",
        "executor_queue_seconds", "worker_seconds", "queue_wait_seconds",
        "queue_depth", "round", "call_id", "tool_name", "phase",
        "request", "request_timeout_seconds", "display_id", "lag_seconds",
        "max_event_loop_lag_seconds", "average_event_loop_lag_seconds",
        "sample_count", "started_monotonic_seconds", "finished_monotonic_seconds",
        "gap_since_previous_seconds", "lifecycle_offset_seconds",
        "turn_id", "tool_result_count", "event_id"),
    "wake.failed": ("error_type",),
    "wake.finished": ("status", "duration_seconds", "model_calls"),
    "wake.sleeping": ("status",),
    "curator.requested": ("status",),
    "curator.started": ("attempt",),
    "curator.caught_up": ("attempt",),
    "curator.retry_scheduled": ("attempt", "error_type", "retry_seconds"),
    "curator.degraded": ("attempt", "error_type", "retry_seconds"),
    "curator.failed": ("phase", "error_type", "history_status"),
    "wakeup.scheduled": ("schedule_id", "due_at"),
}
_SAFE_JOURNAL_EVENT_LIMIT = 50
MAX_OWNER_GUIDANCE_ENTRY_BYTES = 4096
MAX_ACTIVE_OWNER_GUIDANCE_COUNT = 16
MAX_ACTIVE_OWNER_GUIDANCE_BYTES = 32768


def _create_request_configuration(request: dict[str, Any]) -> tuple[dict, dict]:
    """Reconstruct the applied descriptors solely from a durable create request."""
    agent = request.get("agent") if isinstance(request.get("agent"), dict) else {}
    schema = ((agent.get("text") or {}).get("format") or {}).get("schema")
    output_descriptors = []
    if isinstance(schema, dict):
        branches = (((schema.get("properties") or {}).get("outputs") or {})
                    .get("items", {}).get("anyOf", []))
        for branch in branches:
            properties = branch.get("properties", {})
            type_enum = properties.get("type", {}).get("enum", [])
            target_enum = properties.get("target", {}).get("enum", [])
            if len(type_enum) != 1:
                continue
            payload_properties = {key: value for key, value in properties.items()
                                  if key not in {"type", "target"}}
            payload_required = [key for key in branch.get("required", [])
                                if key not in {"type", "target"}]
            output_descriptors.append({
                "type": type_enum[0] if len(type_enum) == 1 else None,
                "target": target_enum[0] if len(target_enum) == 1 else None,
                "description": branch.get("description"),
                "payload_schema": {
                    "type": "object", "properties": payload_properties,
                    "required": payload_required, "additionalProperties": False,
                },
            })
    protocol = {
        "version": 2 if agent.get("text") else 1,
        "instructions": agent.get("instructions"),
        "tools": [{key: tool.get(key) for key in
                   ("type", "name", "description", "parameters")}
                  for tool in agent.get("tools", []) if isinstance(tool, dict)],
        "saved_agent_id": request.get("agent_id"),
        "environment": request.get("environment"),
        "security_policy_revision": 1,
        "output_schema_fingerprint": (hashlib.sha256(json.dumps(
            schema, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            if schema is not None else None),
        "output_schema": schema,
        "output_capabilities": output_descriptors,
    }
    mutable = {key: agent[key] for key in ("model", "reasoning", "service_tier")
               if key in agent}
    return protocol, mutable


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
        INSERT INTO schema_version(version) SELECT 14 WHERE NOT EXISTS (SELECT 1 FROM schema_version);
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
        CREATE TABLE IF NOT EXISTS keeper_interactions(
          run_id TEXT PRIMARY KEY REFERENCES wake_runs(id), format_version INTEGER NOT NULL DEFAULT 1,
          event_id TEXT NOT NULL, occurred_at TEXT NOT NULL, wake_source TEXT NOT NULL,
          wake_reason TEXT NOT NULL, wake_payload_json TEXT NOT NULL,
          status TEXT NOT NULL, session_id TEXT, input_text TEXT,
          realm_snapshot_json TEXT, realm_game_id TEXT, realm_actor_id TEXT,
          realm_revision INTEGER, input_bytes INTEGER, snapshot_bytes INTEGER,
          input_tokens INTEGER NOT NULL DEFAULT 0, output_tokens INTEGER NOT NULL DEFAULT 0,
          completed_at TEXT);
        CREATE TABLE IF NOT EXISTS keeper_activity(
          sequence INTEGER PRIMARY KEY AUTOINCREMENT,
          run_id TEXT NOT NULL REFERENCES keeper_interactions(run_id),
          occurred_at TEXT NOT NULL, kind TEXT NOT NULL, session_id TEXT,
          turn_id TEXT, call_id TEXT, content_json TEXT NOT NULL,
          UNIQUE(run_id,kind,session_id,turn_id,call_id));
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
          last_item_id TEXT, last_turn_id TEXT, handover_draft TEXT, updated_at TEXT NOT NULL,
          PRIMARY KEY(provider,session_id));
        CREATE TABLE IF NOT EXISTS curator_consumed_turns(
          provider TEXT NOT NULL, session_id TEXT NOT NULL, turn_id TEXT NOT NULL,
          cursor TEXT NOT NULL, PRIMARY KEY(provider,session_id,turn_id));
        CREATE TABLE IF NOT EXISTS curator_operations(
          operation_key TEXT PRIMARY KEY, session_id TEXT NOT NULL,
          cursor TEXT, applied_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS curator_jobs(
          id TEXT PRIMARY KEY, session_id TEXT NOT NULL, source_cursor TEXT,
          status TEXT NOT NULL CHECK(status IN ('claimed','completed','failed')),
          attempts INTEGER NOT NULL, last_error_type TEXT,
          created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS curator_requests(
          provider TEXT NOT NULL, session_id TEXT NOT NULL, target_turn_id TEXT NOT NULL,
          status TEXT NOT NULL CHECK(status IN ('pending','running','retrying')),
          attempts INTEGER NOT NULL DEFAULT 0, next_retry_at TEXT, last_error_type TEXT,
          requested_at TEXT NOT NULL, updated_at TEXT NOT NULL,
          PRIMARY KEY(provider,session_id));
        CREATE TABLE IF NOT EXISTS session_handovers(
          id TEXT PRIMARY KEY, old_session_id TEXT NOT NULL, new_session_id TEXT,
          content TEXT NOT NULL, created_at TEXT NOT NULL, expires_at TEXT NOT NULL,
          consumed_at TEXT);
        CREATE TABLE IF NOT EXISTS session_protocol_descriptors(
          provider TEXT NOT NULL, session_id TEXT NOT NULL, descriptor_json TEXT NOT NULL,
          created_at TEXT NOT NULL, PRIMARY KEY(provider,session_id));
        CREATE TABLE IF NOT EXISTS session_mutable_settings(
          provider TEXT NOT NULL, session_id TEXT NOT NULL, settings_json TEXT NOT NULL,
          updated_at TEXT NOT NULL, PRIMARY KEY(provider,session_id));
        CREATE TABLE IF NOT EXISTS session_authoritative_state(
          provider TEXT NOT NULL, session_id TEXT NOT NULL, state_json TEXT NOT NULL,
          updated_at TEXT NOT NULL, PRIMARY KEY(provider,session_id));
        CREATE TABLE IF NOT EXISTS session_rollovers(
          id TEXT PRIMARY KEY, provider TEXT NOT NULL, old_session_id TEXT,
          new_session_id TEXT, reason TEXT NOT NULL, requested_by TEXT NOT NULL,
          finalization_status TEXT NOT NULL, status TEXT NOT NULL,
          creation_state TEXT NOT NULL DEFAULT 'not_attempted', create_token TEXT,
          create_request_json TEXT, create_request_hash TEXT, handover_id TEXT,
          protocol_descriptor_json TEXT, mutable_settings_json TEXT,
          created_at TEXT NOT NULL, create_started_at TEXT, bound_at TEXT, completed_at TEXT);
        CREATE TABLE IF NOT EXISTS session_rollover_requests(
          provider TEXT PRIMARY KEY, old_session_id TEXT NOT NULL, reason TEXT NOT NULL,
          requested_by TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS agent_tool_actions(
          provider TEXT NOT NULL, session_id TEXT NOT NULL, turn_id TEXT NOT NULL,
          call_id TEXT NOT NULL, name TEXT NOT NULL, arguments_json TEXT NOT NULL,
          status TEXT NOT NULL CHECK(status IN ('pending','completed')),
          output_json TEXT, attachments_ephemeral INTEGER NOT NULL DEFAULT 0
            CHECK(attachments_ephemeral IN (0,1)), created_at TEXT NOT NULL, completed_at TEXT,
          PRIMARY KEY(provider,session_id,call_id));
        CREATE TABLE IF NOT EXISTS realm_mutation_requests(
          idempotency_key TEXT PRIMARY KEY, path TEXT NOT NULL,
          body_json TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS agent_wake_submissions(
          provider TEXT NOT NULL, session_id TEXT NOT NULL, wake_key TEXT NOT NULL,
          correlation TEXT NOT NULL,
          state TEXT NOT NULL CHECK(state IN ('possibly_accepted','settled')),
          turn_id TEXT, wake_id TEXT, wake_source TEXT, wake_reason TEXT,
          attempted_at TEXT NOT NULL, settled_at TEXT,
          PRIMARY KEY(provider,session_id,wake_key));
        CREATE TABLE IF NOT EXISTS final_dispositions(
          id TEXT PRIMARY KEY, provider TEXT NOT NULL, session_id TEXT NOT NULL,
          turn_id TEXT NOT NULL, run_id TEXT, wake_id TEXT,
          schema_fingerprint TEXT NOT NULL, raw_disposition TEXT,
          normalized_disposition_json TEXT, validation_state TEXT NOT NULL,
          created_at TEXT NOT NULL, UNIQUE(provider,session_id,turn_id));
        CREATE TABLE IF NOT EXISTS output_requests(
          id TEXT PRIMARY KEY, disposition_id TEXT NOT NULL REFERENCES final_dispositions(id),
          ordinal INTEGER NOT NULL, output_type TEXT NOT NULL, target TEXT,
          payload_json TEXT NOT NULL, route_identity TEXT, capability_fingerprint TEXT,
          delivery_state TEXT NOT NULL, attempt_count INTEGER NOT NULL DEFAULT 0,
          max_attempts INTEGER NOT NULL DEFAULT 3,
          next_attempt_at TEXT, last_failure_classification TEXT,
          message_id TEXT REFERENCES messages(id), failure_event_generated INTEGER NOT NULL DEFAULT 0,
          created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
          UNIQUE(disposition_id,ordinal));
        CREATE TABLE IF NOT EXISTS output_attempts(
          output_request_id TEXT NOT NULL REFERENCES output_requests(id),
          attempt_number INTEGER NOT NULL, started_at TEXT NOT NULL, finished_at TEXT,
          outcome TEXT NOT NULL, failure_classification TEXT, external_message_id TEXT,
          PRIMARY KEY(output_request_id,attempt_number));
        CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_wake_runs_started ON wake_runs(started_at DESC);
        CREATE INDEX IF NOT EXISTS idx_keeper_interactions_completed
          ON keeper_interactions(status,completed_at DESC);
        CREATE INDEX IF NOT EXISTS idx_keeper_activity_run ON keeper_activity(run_id,sequence);
        CREATE INDEX IF NOT EXISTS idx_journal_run_sequence ON journal(run_id, sequence);
        CREATE INDEX IF NOT EXISTS idx_schedules_due ON scheduled_wakeups(status, due_at);
        CREATE INDEX IF NOT EXISTS idx_memory_status_updated ON memory_records(status,updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_memory_provenance_source
          ON memory_provenance(source_session_id,source_item_id);
        CREATE INDEX IF NOT EXISTS idx_output_dispatch
          ON output_requests(delivery_state,next_attempt_at,created_at);
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
        if "last_turn_id" not in checkpoint_columns:
            with self.connection:
                self.connection.execute(
                    "ALTER TABLE curator_checkpoints ADD COLUMN last_turn_id TEXT")
        wake_submission_columns = {
            row["name"] for row in self.connection.execute(
                "PRAGMA table_info(agent_wake_submissions)")
        }
        with self.connection:
            for name in ("wake_id", "wake_source", "wake_reason"):
                if name not in wake_submission_columns:
                    self.connection.execute(
                        f"ALTER TABLE agent_wake_submissions ADD COLUMN {name} TEXT")
        output_request_columns = {
            row["name"] for row in self.connection.execute("PRAGMA table_info(output_requests)")
        }
        if "max_attempts" not in output_request_columns:
            with self.connection:
                self.connection.execute(
                    "ALTER TABLE output_requests ADD COLUMN "
                    "max_attempts INTEGER NOT NULL DEFAULT 3")
        rollover_columns = {
            row["name"] for row in self.connection.execute("PRAGMA table_info(session_rollovers)")
        }
        rollover_additions = {
            "creation_state": "TEXT NOT NULL DEFAULT 'not_attempted'",
            "create_token": "TEXT",
            "create_request_json": "TEXT",
            "create_request_hash": "TEXT",
            "handover_id": "TEXT",
            "protocol_descriptor_json": "TEXT",
            "mutable_settings_json": "TEXT",
            "create_started_at": "TEXT",
            "bound_at": "TEXT",
        }
        with self.connection:
            for name, declaration in rollover_additions.items():
                if name not in rollover_columns:
                    self.connection.execute(
                        f"ALTER TABLE session_rollovers ADD COLUMN {name} {declaration}")
            # A pending row written by the old implementation may already have
            # crossed the remote POST. Treat it as uncertain, never as an
            # unattempted create merely because the old schema lacked this state.
            self.connection.execute("""
                UPDATE session_rollovers SET creation_state='create_uncertain'
                WHERE status='pending' AND create_request_json IS NULL
                  AND creation_state='not_attempted'
            """)
            legacy_snapshots = self.connection.execute("""
                SELECT id,create_request_json FROM session_rollovers
                WHERE status='pending' AND creation_state='not_attempted'
                  AND create_request_json IS NOT NULL
                  AND (protocol_descriptor_json IS NULL OR mutable_settings_json IS NULL)
            """).fetchall()
            for row in legacy_snapshots:
                request = json.loads(row["create_request_json"])
                protocol, mutable = _create_request_configuration(request)
                self.connection.execute("""
                    UPDATE session_rollovers
                    SET protocol_descriptor_json=?,mutable_settings_json=? WHERE id=?
                """, (
                    json.dumps(protocol, sort_keys=True, separators=(",", ":")),
                    json.dumps(mutable, sort_keys=True, separators=(",", ":")),
                    row["id"],
                ))
        self.connection.execute("UPDATE schema_version SET version=22")
        # A process may stop after transport acceptance but before recording it.
        # Retry uncertain attempts only while the persisted delivery policy allows it.
        now = utc_now()
        interrupted = self.connection.execute("""
            SELECT id,output_type,target,attempt_count,max_attempts,message_id,
                   failure_event_generated
            FROM output_requests
            WHERE delivery_state='attempting'
               OR (delivery_state='retry_wait' AND attempt_count>=max_attempts)
        """).fetchall()
        with self.connection:
            for row in interrupted:
                exhausted = int(row["attempt_count"]) >= int(row["max_attempts"])
                self.connection.execute("""
                    UPDATE output_attempts SET finished_at=?,outcome='delivery_uncertain',
                      failure_classification='interrupted_attempt'
                    WHERE output_request_id=? AND attempt_number=? AND outcome='attempting'
                """, (now, row["id"], row["attempt_count"]))
                if not exhausted:
                    self.connection.execute("""
                        UPDATE output_requests SET delivery_state='retry_wait',next_attempt_at=?,
                          last_failure_classification='interrupted_attempt',updated_at=?
                        WHERE id=? AND delivery_state='attempting'
                    """, (now, now, row["id"]))
                    continue
                self.connection.execute("""
                    UPDATE output_requests SET delivery_state='failed_permanent',
                      next_attempt_at=NULL,last_failure_classification='retries_exhausted',
                      updated_at=? WHERE id=?
                        AND delivery_state IN ('attempting','retry_wait')
                """, (now, row["id"]))
                if row["message_id"]:
                    self.connection.execute(
                        "UPDATE messages SET delivery_status='transport_failed' WHERE id=?",
                        (row["message_id"],))
                if not row["failure_event_generated"]:
                    context = json.dumps({
                        "output_id": row["id"], "output_type": row["output_type"],
                        "target": row["target"],
                        "failure_classification": "retries_exhausted",
                        "attempt_count": row["attempt_count"],
                    }, separators=(",", ":"))
                    self.connection.execute("""
                        UPDATE output_requests SET failure_event_generated=1 WHERE id=?
                    """, (row["id"],))
                    self.connection.execute("""
                        INSERT INTO scheduled_wakeups(
                          id,due_at,reason,context_json,status,created_at)
                        VALUES(?,?, 'output_delivery_failed',?,'pending',?)
                        ON CONFLICT(id) DO NOTHING
                    """, (f"output-failure:{row['id']}", now, context, now))
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

    def session_authoritative_state(self, provider: str,
                                    session_id: str) -> dict[str, Any] | None:
        row = self.connection.execute("""
            SELECT state_json FROM session_authoritative_state
            WHERE provider=? AND session_id=?
        """, (provider, session_id)).fetchone()
        return None if row is None else json.loads(row["state_json"])

    def save_session_authoritative_state(self, provider: str, session_id: str,
                                         state: dict[str, Any]) -> None:
        encoded = json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self.connection:
            self.connection.execute("""
                INSERT INTO session_authoritative_state(
                  provider,session_id,state_json,updated_at) VALUES(?,?,?,?)
                ON CONFLICT(provider,session_id) DO UPDATE SET
                  state_json=excluded.state_json,updated_at=excluded.updated_at
            """, (provider, session_id, encoded, utc_now()))

    def owner_guidance_revision(self, guidance_id: str) -> int | None:
        row = self.connection.execute("""
            SELECT MAX(revision) AS revision FROM owner_guidance_revisions
            WHERE guidance_id=?
        """, (guidance_id,)).fetchone()
        return None if row is None or row["revision"] is None else int(row["revision"])

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

    def agent_wake_submission(self, provider: str, session_id: str,
                              wake_key: str) -> dict[str, Any] | None:
        row = self.connection.execute("""
            SELECT provider,session_id,wake_key,correlation,state,turn_id,
                   wake_id,wake_source,wake_reason,attempted_at,settled_at
            FROM agent_wake_submissions
            WHERE provider=? AND session_id=? AND wake_key=?
        """, (provider, session_id, wake_key)).fetchone()
        return None if row is None else dict(row)

    def mark_agent_wake_submission_attempted(self, provider: str, session_id: str,
                                             wake_key: str,
                                             correlation: str, *, wake_id: str | None = None,
                                             wake_source: str | None = None,
                                             wake_reason: str | None = None) -> None:
        """Durably cross the point of no blind retry before the remote POST."""
        now = utc_now()
        with self.connection:
            self.connection.execute("""
                INSERT INTO agent_wake_submissions(
                  provider,session_id,wake_key,correlation,state,turn_id,wake_id,
                  wake_source,wake_reason,attempted_at,settled_at)
                VALUES(?,?,?,?,'possibly_accepted',NULL,?,?,?,?,NULL)
                ON CONFLICT(provider,session_id,wake_key) DO UPDATE SET
                  correlation=excluded.correlation,state='possibly_accepted',
                  turn_id=NULL,
                  wake_id=COALESCE(excluded.wake_id,agent_wake_submissions.wake_id),
                  wake_source=COALESCE(
                    excluded.wake_source,agent_wake_submissions.wake_source),
                  wake_reason=COALESCE(
                    excluded.wake_reason,agent_wake_submissions.wake_reason),
                  attempted_at=excluded.attempted_at,
                  settled_at=NULL
            """, (provider, session_id, wake_key, correlation, wake_id,
                  wake_source, wake_reason, now))

    def disposition_wake_context(self, provider: str, session_id: str,
                                 turn_id: str) -> dict[str, Any] | None:
        row = self.connection.execute("""
            SELECT wake_id,wake_source,wake_reason FROM agent_wake_submissions
            WHERE provider=? AND session_id=? AND turn_id=?
        """, (provider, session_id, turn_id)).fetchone()
        if row is None or row["wake_source"] is None or row["wake_reason"] is None:
            return None
        return dict(row)

    def correlate_agent_wake_submission(self, provider: str, session_id: str,
                                        wake_key: str, turn_id: str) -> None:
        with self.connection:
            cursor = self.connection.execute("""
                UPDATE agent_wake_submissions SET turn_id=?
                WHERE provider=? AND session_id=? AND wake_key=?
                  AND state='possibly_accepted'
            """, (turn_id, provider, session_id, wake_key))
            if cursor.rowcount != 1:
                raise RuntimeError("Agents wake submission checkpoint is missing")

    def settle_agent_wake_submission(self, provider: str, session_id: str,
                                     turn_id: str) -> None:
        with self.connection:
            self.connection.execute("""
                UPDATE agent_wake_submissions SET state='settled',settled_at=?
                WHERE provider=? AND session_id=? AND turn_id=?
            """, (utc_now(), provider, session_id, turn_id))

    def clear_agent_wake_submission(self, provider: str, session_id: str,
                                    wake_key: str) -> None:
        with self.connection:
            self.connection.execute("""
                DELETE FROM agent_wake_submissions
                WHERE provider=? AND session_id=? AND wake_key=?
            """, (provider, session_id, wake_key))

    def bind_initial_agent_session(self, provider: str, session_id: str,
                                   agent_id: str | None,
                                   create_request: dict[str, Any],
                                   protocol_descriptor: dict[str, Any],
                                   mutable_settings: dict[str, Any]) -> None:
        """Atomically bind a created session and its exact applied configuration."""
        request_protocol, request_mutable = _create_request_configuration(create_request)
        if protocol_descriptor != request_protocol or mutable_settings != request_mutable:
            raise ValueError(
                "Initial create request and configuration descriptors must match")
        descriptor_json = json.dumps(
            protocol_descriptor, sort_keys=True, separators=(",", ":"))
        settings_json = json.dumps(
            mutable_settings, sort_keys=True, separators=(",", ":"))
        now = utc_now()
        with self.connection:
            self.connection.execute("""
                INSERT INTO agent_session_bindings(
                  provider,session_id,agent_id,last_turn_id,created_at,updated_at)
                VALUES(?,?,?,NULL,?,?) ON CONFLICT(provider) DO UPDATE SET
                  session_id=excluded.session_id,agent_id=excluded.agent_id,last_turn_id=NULL,
                  updated_at=excluded.updated_at
            """, (provider, session_id, agent_id, now, now))
            self.connection.execute("""
                INSERT INTO session_protocol_descriptors(provider,session_id,descriptor_json,created_at)
                VALUES(?,?,?,?) ON CONFLICT(provider,session_id) DO UPDATE SET
                  descriptor_json=excluded.descriptor_json
            """, (provider, session_id, descriptor_json, now))
            self.connection.execute("""
                INSERT INTO session_mutable_settings(provider,session_id,settings_json,updated_at)
                VALUES(?,?,?,?) ON CONFLICT(provider,session_id) DO UPDATE SET
                  settings_json=excluded.settings_json,updated_at=excluded.updated_at
            """, (provider, session_id, settings_json, now))

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

    def save_session_mutable_settings(self, provider: str, session_id: str,
                                      settings: dict[str, Any]) -> None:
        encoded = json.dumps(settings, sort_keys=True, separators=(",", ":"))
        with self.connection:
            self.connection.execute("""
                INSERT INTO session_mutable_settings(provider,session_id,settings_json,updated_at)
                VALUES(?,?,?,?) ON CONFLICT(provider,session_id) DO UPDATE SET
                  settings_json=excluded.settings_json,updated_at=excluded.updated_at
            """, (provider, session_id, encoded, utc_now()))

    def session_mutable_settings(self, provider: str,
                                 session_id: str) -> dict[str, Any] | None:
        row = self.connection.execute("""
            SELECT settings_json FROM session_mutable_settings
            WHERE provider=? AND session_id=?
        """, (provider, session_id)).fetchone()
        return None if row is None else json.loads(row["settings_json"])

    def begin_session_rollover(self, provider: str, old_session_id: str | None,
                               reason: str, requested_by: str,
                               create_request: dict[str, Any],
                               protocol_descriptor: dict[str, Any],
                               mutable_settings: dict[str, Any]) -> dict[str, Any]:
        # Once a create may have crossed the remote boundary, its recorded
        # transition excludes every competing create for this provider/session.
        # Its historical reason and snapshots remain authoritative.
        pending = self.connection.execute("""
            SELECT * FROM session_rollovers
            WHERE provider=? AND old_session_id IS ? AND status='pending'
              AND creation_state IN ('create_uncertain','not_attempted')
            ORDER BY CASE creation_state WHEN 'create_uncertain' THEN 0 ELSE 1 END,
              created_at ASC LIMIT 1
        """, (provider, old_session_id)).fetchone()
        if pending is not None:
            result = dict(pending)
            if result.get("create_request_json"):
                result["create_request"] = json.loads(result["create_request_json"])
            if result.get("protocol_descriptor_json"):
                result["protocol_descriptor"] = json.loads(
                    result["protocol_descriptor_json"])
            if result.get("mutable_settings_json"):
                result["mutable_settings"] = json.loads(result["mutable_settings_json"])
            return result
        rollover_id = str(uuid.uuid4())
        create_token = str(uuid.uuid4())
        request = json.loads(json.dumps(create_request))
        request.setdefault("metadata", {})["rollover_token"] = create_token
        encoded = json.dumps(request, sort_keys=True, separators=(",", ":"))
        request_hash = hashlib.sha256(encoded.encode()).hexdigest()
        request_protocol, request_mutable = _create_request_configuration(request)
        if protocol_descriptor != request_protocol or mutable_settings != request_mutable:
            raise ValueError(
                "Rollover create request and configuration descriptors must match")
        descriptor_json = json.dumps(
            protocol_descriptor, sort_keys=True, separators=(",", ":"))
        settings_json = json.dumps(mutable_settings, sort_keys=True, separators=(",", ":"))
        handover = self.pending_handover(old_session_id) if old_session_id else None
        if handover is not None:
            try:
                bootstrap = json.loads(request.get("input", ""))["new_session_bootstrap"]
            except (KeyError, TypeError, json.JSONDecodeError):
                handover = None
            else:
                if bootstrap.get("handover") != handover["content"]:
                    handover = None
        with self.connection:
            self.connection.execute("""
                INSERT INTO session_rollovers(
                  id,provider,old_session_id,reason,requested_by,finalization_status,status,
                  creation_state,create_token,create_request_json,create_request_hash,
                  handover_id,protocol_descriptor_json,mutable_settings_json,created_at)
                VALUES(?,?,?,?,?,'pending','pending','not_attempted',?,?,?,?,?,?,?)
            """, (rollover_id, provider, old_session_id, reason, requested_by,
                  create_token, encoded, request_hash,
                  handover["id"] if handover else None, descriptor_json, settings_json,
                  utc_now()))
        return {
            "id": rollover_id, "provider": provider, "old_session_id": old_session_id,
            "reason": reason, "status": "pending", "creation_state": "not_attempted",
            "create_token": create_token, "create_request": request,
            "create_request_hash": request_hash,
            "handover_id": handover["id"] if handover else None,
            "protocol_descriptor": json.loads(descriptor_json),
            "mutable_settings": json.loads(settings_json),
        }

    def request_session_rollover(self, provider: str, old_session_id: str,
                                 reason: str, requested_by: str = "runtime") -> None:
        """Durably retain a pre-create request while final consolidation runs."""
        now = utc_now()
        with self.connection:
            self.connection.execute("""
                INSERT INTO session_rollover_requests(
                  provider,old_session_id,reason,requested_by,created_at,updated_at)
                VALUES(?,?,?,?,?,?)
                ON CONFLICT(provider) DO UPDATE SET
                  old_session_id=excluded.old_session_id,reason=excluded.reason,
                  requested_by=excluded.requested_by,updated_at=excluded.updated_at
            """, (provider, old_session_id, reason, requested_by, now, now))

    def pending_session_rollover_request(self, provider: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM session_rollover_requests WHERE provider=?", (provider,)).fetchone()
        return None if row is None else dict(row)

    def clear_session_rollover_request(self, provider: str,
                                       old_session_id: str) -> None:
        with self.connection:
            self.connection.execute("""
                DELETE FROM session_rollover_requests
                WHERE provider=? AND old_session_id=?
            """, (provider, old_session_id))

    def pending_session_rollover(self, provider: str) -> dict[str, Any] | None:
        row = self.connection.execute("""
            SELECT * FROM session_rollovers WHERE provider=? AND status='pending'
            ORDER BY CASE WHEN creation_state='create_uncertain' AND old_session_id IS (
                SELECT session_id FROM agent_session_bindings WHERE provider=?
              ) THEN 0 WHEN old_session_id IS (
                SELECT session_id FROM agent_session_bindings WHERE provider=?
              ) THEN 1 ELSE 2 END,
              CASE WHEN creation_state='create_uncertain' THEN created_at END ASC,
              created_at DESC LIMIT 1
        """, (provider, provider, provider)).fetchone()
        if row is None:
            return None
        result = dict(row)
        if result.get("create_request_json"):
            result["create_request"] = json.loads(result["create_request_json"])
        if result.get("protocol_descriptor_json"):
            result["protocol_descriptor"] = json.loads(result["protocol_descriptor_json"])
        if result.get("mutable_settings_json"):
            result["mutable_settings"] = json.loads(result["mutable_settings_json"])
        return result

    def mark_session_rollover_create_started(self, rollover_id: str) -> None:
        with self.connection:
            result = self.connection.execute("""
                UPDATE session_rollovers SET creation_state='create_uncertain',create_started_at=?
                WHERE id=? AND status='pending' AND creation_state='not_attempted'
            """, (utc_now(), rollover_id))
        if result.rowcount != 1:
            raise RuntimeError("Rollover remote creation is not safe to start again")

    def bind_session_rollover(self, rollover_id: str, new_session_id: str,
                              agent_id: str | None,
                              finalization_status: str) -> None:
        now = utc_now()
        row = self.connection.execute(
            "SELECT provider,protocol_descriptor_json,mutable_settings_json "
            "FROM session_rollovers WHERE id=? AND status='pending' "
            "AND creation_state='create_uncertain'", (rollover_id,)).fetchone()
        if row is None:
            raise RuntimeError("Rollover is not awaiting a remote create result")
        provider = row["provider"]
        descriptor_json = row["protocol_descriptor_json"]
        settings_json = row["mutable_settings_json"]
        if descriptor_json is None or settings_json is None:
            raise RuntimeError("Rollover has no durable configuration snapshot")
        with self.connection:
            self.connection.execute("""
                INSERT INTO agent_session_bindings(
                  provider,session_id,agent_id,last_turn_id,created_at,updated_at)
                VALUES(?,?,?,NULL,?,?) ON CONFLICT(provider) DO UPDATE SET
                  session_id=excluded.session_id,agent_id=excluded.agent_id,last_turn_id=NULL,
                  updated_at=excluded.updated_at
            """, (provider, new_session_id, agent_id, now, now))
            self.connection.execute("""
                INSERT INTO session_protocol_descriptors(provider,session_id,descriptor_json,created_at)
                VALUES(?,?,?,?) ON CONFLICT(provider,session_id) DO UPDATE SET
                  descriptor_json=excluded.descriptor_json
            """, (provider, new_session_id, descriptor_json, now))
            self.connection.execute("""
                INSERT INTO session_mutable_settings(provider,session_id,settings_json,updated_at)
                VALUES(?,?,?,?) ON CONFLICT(provider,session_id) DO UPDATE SET
                  settings_json=excluded.settings_json,updated_at=excluded.updated_at
            """, (provider, new_session_id, settings_json, now))
            self.connection.execute("""
                UPDATE session_rollovers SET new_session_id=?,creation_state='bound',
                  finalization_status=?,bound_at=? WHERE id=?
            """, (new_session_id, finalization_status, now, rollover_id))

    def recover_session_rollovers(self, provider: str) -> int:
        """Finish only replacements whose binding was durably committed."""
        now = utc_now()
        binding = self.agent_session_binding(provider)
        with self.connection:
            if binding is not None:
                # Legacy rows did not record the explicit bound state. A binding
                # away from the recorded old session is nevertheless durable
                # proof that the replacement ID reached SQLite.
                self.connection.execute("""
                    UPDATE session_rollovers SET new_session_id=?,creation_state='bound',
                      bound_at=?,finalization_status=CASE
                        WHEN finalization_status='pending' THEN 'unknown_after_restart'
                        ELSE finalization_status END
                    WHERE provider=? AND status='pending'
                      AND create_request_json IS NULL AND creation_state='create_uncertain'
                      AND (old_session_id IS NULL OR old_session_id<>?)
                """, (binding["session_id"], now, provider, binding["session_id"]))
            result = self.connection.execute("""
                UPDATE session_rollovers SET status='completed',completed_at=?
                WHERE provider=? AND status='pending' AND creation_state='bound'
            """, (now, provider))
            self.connection.execute("""
                UPDATE session_handovers SET new_session_id=(
                    SELECT new_session_id FROM session_rollovers
                    WHERE session_rollovers.handover_id=session_handovers.id
                      AND provider=? AND creation_state='bound'
                    ORDER BY bound_at DESC LIMIT 1),consumed_at=?
                WHERE consumed_at IS NULL AND id IN (
                    SELECT handover_id FROM session_rollovers
                    WHERE provider=? AND creation_state='bound' AND handover_id IS NOT NULL)
            """, (provider, now, provider))
        return result.rowcount

    def complete_session_rollover(self, rollover_id: str) -> None:
        with self.connection:
            self.connection.execute("""
                UPDATE session_rollovers SET status='completed',completed_at=?
                WHERE id=? AND status='pending' AND creation_state='bound'
            """, (utc_now(), rollover_id))

    def fail_session_rollover(self, rollover_id: str,
                              finalization_status: str = "failed") -> None:
        with self.connection:
            self.connection.execute("""
                UPDATE session_rollovers SET finalization_status=?,status='failed',
                  creation_state='rejected',completed_at=?
                WHERE id=? AND status='pending'
            """, (finalization_status, utc_now(), rollover_id))

    def curator_checkpoint(self, provider: str, session_id: str) -> dict[str, Any] | None:
        row = self.connection.execute("""
            SELECT cursor,last_item_id,last_turn_id,handover_draft,updated_at FROM curator_checkpoints
            WHERE provider=? AND session_id=?
        """, (provider, session_id)).fetchone()
        return None if row is None else dict(row)

    def curator_turn_consumed(self, provider: str, session_id: str,
                              turn_id: str) -> bool:
        return self.connection.execute("""
            SELECT 1 FROM curator_consumed_turns
            WHERE provider=? AND session_id=? AND turn_id=?
        """, (provider, session_id, turn_id)).fetchone() is not None

    def mark_curator_turn_consumed(self, provider: str, session_id: str,
                                   turn_id: str, cursor: str) -> bool:
        """Verify the checkpoint still sits at a source-proven turn boundary."""
        with self.connection:
            result = self.connection.execute("""
                INSERT INTO curator_consumed_turns(provider,session_id,turn_id,cursor)
                SELECT provider,session_id,?,cursor FROM curator_checkpoints
                WHERE provider=? AND session_id=? AND cursor=? AND last_turn_id=?
                ON CONFLICT DO NOTHING
            """, (turn_id, provider, session_id, cursor, turn_id))
        return result.rowcount == 1

    def record_verified_historical_turn(self, provider: str, session_id: str,
                                        turn_id: str, boundary_cursor: str,
                                        checkpoint_cursor: str) -> bool:
        """Backfill a source-verified boundary under the checkpoint that was scanned."""
        with self.connection:
            result = self.connection.execute("""
                INSERT INTO curator_consumed_turns(provider,session_id,turn_id,cursor)
                SELECT provider,session_id,?,? FROM curator_checkpoints
                WHERE provider=? AND session_id=? AND cursor=?
                ON CONFLICT DO NOTHING
            """, (turn_id, boundary_cursor, provider, session_id, checkpoint_cursor))
        return result.rowcount == 1

    def upgrade_legacy_curator_checkpoint(self, provider: str, session_id: str,
                                          cursor: str, last_item_id: str,
                                          last_turn_id: str) -> bool:
        """Record a verified turn only if the legacy checkpoint is unchanged."""
        with self.connection:
            result = self.connection.execute("""
                UPDATE curator_checkpoints SET last_turn_id=?,updated_at=?
                WHERE provider=? AND session_id=? AND cursor=? AND last_item_id=?
                  AND last_turn_id IS NULL
            """, (last_turn_id, utc_now(), provider, session_id, cursor, last_item_id))
            if result.rowcount:
                self.connection.execute("""
                    INSERT OR IGNORE INTO curator_consumed_turns
                    (provider,session_id,turn_id,cursor) VALUES(?,?,?,?)
                """, (provider, session_id, last_turn_id, cursor))
        return result.rowcount == 1

    def apply_curator_batch(self, provider: str, session_id: str, cursor: str | None,
                            last_item_id: str | None, operation_key: str,
                            mutations: list[dict[str, Any]],
                            handover_operation: str = "keep",
                            handover_draft: str | None = None,
                            last_turn_id: str | None = None,
                            consumed_turns: tuple[tuple[str, str], ...] = ()) -> bool:
        """Atomically apply validated curator decisions and advance its source checkpoint."""
        if any(not mutation.get("provenance") for mutation in mutations):
            raise ValueError("Durable Curator memory requires verified provenance")
        if handover_operation not in {"keep", "replace", "clear"}:
            raise ValueError("Unsupported Curator handover operation")
        if handover_operation == "replace" and not handover_draft:
            raise ValueError("Replacement Curator handover must be nonempty")
        if handover_operation != "replace" and handover_draft is not None:
            raise ValueError("Only handover replacement may provide draft content")
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
                if operation == "create" and existing is not None:
                    raise ValueError(f"Memory create targets an existing record: {memory_id}")
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
                  provider,session_id,cursor,last_item_id,last_turn_id,handover_draft,updated_at)
                VALUES(?,?,?,?,?,?,?) ON CONFLICT(provider,session_id) DO UPDATE SET
                  cursor=excluded.cursor,last_item_id=excluded.last_item_id,
                  last_turn_id=excluded.last_turn_id,
                  handover_draft=CASE ?
                    WHEN 'keep' THEN curator_checkpoints.handover_draft
                    WHEN 'replace' THEN excluded.handover_draft
                    WHEN 'clear' THEN NULL END,
                  updated_at=excluded.updated_at
            """, (provider, session_id, cursor, last_item_id, last_turn_id,
                  handover_draft, now,
                  handover_operation))
            for turn_id, boundary_cursor in consumed_turns:
                self.connection.execute("""
                    INSERT OR IGNORE INTO curator_consumed_turns
                    (provider,session_id,turn_id,cursor) VALUES(?,?,?,?)
                """, (provider, session_id, turn_id, boundary_cursor))
        return True

    def request_curator_catch_up(self, provider: str, session_id: str,
                                 target_turn_id: str) -> bool:
        """Durably coalesce routine work to the newest completed turn."""
        now = utc_now()
        with self.connection:
            if self.curator_turn_consumed(provider, session_id, target_turn_id):
                return False
            current = self.connection.execute("""
                SELECT target_turn_id FROM curator_requests
                WHERE provider=? AND session_id=?
            """, (provider, session_id)).fetchone()
            if current is not None and current["target_turn_id"] == target_turn_id:
                return False
            self.connection.execute("""
                INSERT INTO curator_requests(
                  provider,session_id,target_turn_id,status,attempts,requested_at,updated_at)
                VALUES(?,?,?,'pending',0,?,?)
                ON CONFLICT(provider,session_id) DO UPDATE SET
                  target_turn_id=excluded.target_turn_id,status='pending',attempts=0,
                  next_retry_at=NULL,last_error_type=NULL,requested_at=excluded.requested_at,
                  updated_at=excluded.updated_at
            """, (provider, session_id, target_turn_id, now, now))
        return True

    def curator_request(self, provider: str, session_id: str) -> dict[str, Any] | None:
        row = self.connection.execute("""
            SELECT provider,session_id,target_turn_id,status,attempts,next_retry_at,
                   last_error_type,requested_at,updated_at
            FROM curator_requests WHERE provider=? AND session_id=?
        """, (provider, session_id)).fetchone()
        return None if row is None else dict(row)

    def start_curator_request(self, provider: str, session_id: str,
                              target_turn_id: str) -> int | None:
        now = utc_now()
        with self.connection:
            result = self.connection.execute("""
                UPDATE curator_requests SET status='running',attempts=attempts+1,
                  next_retry_at=NULL,updated_at=?
                WHERE provider=? AND session_id=? AND target_turn_id=?
            """, (now, provider, session_id, target_turn_id))
            if result.rowcount == 0:
                return None
            row = self.connection.execute("""
                SELECT attempts FROM curator_requests WHERE provider=? AND session_id=?
            """, (provider, session_id)).fetchone()
        return int(row["attempts"])

    def retry_curator_request(self, provider: str, session_id: str,
                              target_turn_id: str, error_type: str,
                              retry_seconds: float) -> bool:
        next_retry = (datetime.now(UTC) + timedelta(seconds=retry_seconds)).isoformat()
        with self.connection:
            result = self.connection.execute("""
                UPDATE curator_requests SET status='retrying',next_retry_at=?,
                  last_error_type=?,updated_at=?
                WHERE provider=? AND session_id=? AND target_turn_id=?
            """, (next_retry, error_type[:100], utc_now(), provider, session_id,
                  target_turn_id))
        return result.rowcount > 0

    def complete_curator_request(self, provider: str, session_id: str,
                                 target_turn_id: str | None = None) -> bool:
        parameters: list[Any] = [provider, session_id]
        target_clause = ""
        if target_turn_id is not None:
            target_clause = " AND target_turn_id=?"
            parameters.append(target_turn_id)
        with self.connection:
            result = self.connection.execute(
                "DELETE FROM curator_requests WHERE provider=? AND session_id=?" + target_clause,
                parameters)
        return result.rowcount > 0

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

    def remove_owner_guidance(self, guidance_id: str, *,
                              source_session_id: str | None = None,
                              source_item_id: str | None = None) -> bool:
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
                  source_session_id, source_item_id, utc_now()))
        return result.rowcount == 1

    def is_pending_owner_message(self, message_id: str, owner_id: str) -> bool:
        """Verify the durable message created by the authenticated Owner transport."""
        if not isinstance(message_id, str) or not message_id:
            return False
        return self.connection.execute("""
            SELECT 1
            FROM owner_message_processing
            JOIN messages ON messages.id=owner_message_processing.message_id
            WHERE messages.id=? AND messages.direction='inbound'
              AND messages.sender_id=? AND owner_message_processing.status='pending'
        """, (message_id, owner_id)).fetchone() is not None

    def create_handover(self, old_session_id: str, content: str, expires_at: str) -> str:
        pending = self.pending_handover(old_session_id)
        if pending is not None:
            bound = self.connection.execute(
                "SELECT 1 FROM session_rollovers WHERE handover_id=? AND status='pending'",
                (pending["id"],)).fetchone()
            if bound is None:
                with self.connection:
                    self.connection.execute(
                        "UPDATE session_handovers SET content=?,created_at=?,expires_at=? "
                        "WHERE id=? AND consumed_at IS NULL",
                        (content, utc_now(), expires_at, pending["id"]))
            return pending["id"]
        handover_id = str(uuid.uuid4())
        with self.connection:
            self.connection.execute("""
                INSERT INTO session_handovers(id,old_session_id,content,created_at,expires_at)
                VALUES(?,?,?,?,?)
            """, (handover_id, old_session_id, content, utc_now(), expires_at))
        return handover_id

    def pending_handover(self, old_session_id: str) -> dict[str, Any] | None:
        row = self.connection.execute("""
            SELECT id,old_session_id,new_session_id,content,created_at,expires_at,consumed_at
            FROM session_handovers WHERE old_session_id=? AND consumed_at IS NULL
              AND expires_at>? ORDER BY created_at DESC LIMIT 1
        """, (old_session_id, utc_now())).fetchone()
        return None if row is None else dict(row)

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

    def realm_mutation_request(self, key: str, path: str | None = None,
                               body: dict[str, Any] | None = None
                               ) -> tuple[str, dict[str, Any]] | None:
        """Persist the exact Realm request before its first remote POST."""
        with self.connection:
            if path is not None and body is not None:
                self.connection.execute("""
                    INSERT OR IGNORE INTO realm_mutation_requests
                    (idempotency_key,path,body_json,created_at) VALUES(?,?,?,?)
                """, (key, path, json.dumps(body, allow_nan=False, separators=(",", ":")),
                      utc_now()))
            row = self.connection.execute("""
                SELECT path,body_json FROM realm_mutation_requests WHERE idempotency_key=?
            """, (key,)).fetchone()
        return (row["path"], json.loads(row["body_json"])) if row else None

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

    @staticmethod
    def _disposition_id(provider: str, session_id: str, turn_id: str) -> str:
        return hashlib.sha256(
            f"{provider}\0{session_id}\0{turn_id}".encode()).hexdigest()

    def has_final_disposition(self, provider: str, session_id: str, turn_id: str) -> bool:
        return self.connection.execute("""
            SELECT 1 FROM final_dispositions
            WHERE provider=? AND session_id=? AND turn_id=?
        """, (provider, session_id, turn_id)).fetchone() is not None

    def persist_final_disposition(
            self, provider: str, session_id: str, turn_id: str,
            schema_fingerprint: str, raw_disposition: str | None,
            normalized: dict[str, Any] | None, validation_state: str,
            jobs: list[dict[str, Any]], *, run_id: str | None = None,
            wake_id: str | None = None) -> dict[str, Any]:
        disposition_id = self._disposition_id(provider, session_id, turn_id)
        now = utc_now()
        normalized_json = (json.dumps(normalized, ensure_ascii=False, sort_keys=True,
                                      separators=(",", ":"))
                           if normalized is not None else None)
        created_requests: list[dict[str, Any]] = []
        with self.connection:
            inserted = self.connection.execute("""
                INSERT INTO final_dispositions(
                  id,provider,session_id,turn_id,run_id,wake_id,schema_fingerprint,
                  raw_disposition,normalized_disposition_json,validation_state,created_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(provider,session_id,turn_id) DO NOTHING
            """, (disposition_id, provider, session_id, turn_id, run_id, wake_id,
                  schema_fingerprint, raw_disposition, normalized_json,
                  validation_state, now)).rowcount == 1
            if not inserted:
                return {"id": disposition_id, "created": False, "requests": []}
            for ordinal, job in enumerate(jobs):
                output_id = hashlib.sha256(
                    f"{disposition_id}\0{ordinal}".encode()).hexdigest()
                message_id = None
                if job["output_type"] == "notify_owner":
                    message_id = output_id
                    self.connection.execute("""
                        INSERT INTO messages(
                          id,direction,sender_id,content,spontaneous,delivery_status,created_at)
                        VALUES(?,?,?,?,?,?,?)
                    """, (message_id, "outbound", job["sender_id"],
                          job["payload"]["content"], int(job.get("spontaneous", False)),
                          job.get("message_status", "pending_delivery"), now))
                self.connection.execute("""
                    INSERT INTO output_requests(
                      id,disposition_id,ordinal,output_type,target,payload_json,
                      route_identity,capability_fingerprint,delivery_state,max_attempts,next_attempt_at,
                      last_failure_classification,message_id,created_at,updated_at)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (output_id, disposition_id, ordinal, job["output_type"],
                      job.get("target"), json.dumps(job["payload"], ensure_ascii=False,
                                                   sort_keys=True, separators=(",", ":")),
                      job.get("route_identity"), job.get("capability_fingerprint"),
                      job["delivery_state"], job.get("max_attempts", 3),
                      now if job["delivery_state"] == "queued" else None,
                      job.get("failure_classification"), message_id, now, now))
                if job.get("suppress_failure_event"):
                    self.connection.execute(
                        "UPDATE output_requests SET failure_event_generated=1 WHERE id=?",
                        (output_id,))
                created_requests.append({"id": output_id, **job, "message_id": message_id})
        return {"id": disposition_id, "created": True, "requests": created_requests}

    def claim_output_request(self, now: str | None = None) -> dict[str, Any] | None:
        now = now or utc_now()
        with self.connection:
            row = self.connection.execute("""
                SELECT * FROM output_requests
                WHERE delivery_state IN ('queued','retry_wait')
                  AND attempt_count < max_attempts
                  AND (next_attempt_at IS NULL OR next_attempt_at<=?)
                ORDER BY created_at,ordinal LIMIT 1
            """, (now,)).fetchone()
            if row is None:
                return None
            attempt = int(row["attempt_count"]) + 1
            updated = self.connection.execute("""
                UPDATE output_requests SET delivery_state='attempting',attempt_count=?,
                  updated_at=? WHERE id=? AND delivery_state IN ('queued','retry_wait')
            """, (attempt, now, row["id"])).rowcount
            if updated != 1:
                return None
            self.connection.execute("""
                INSERT INTO output_attempts(
                  output_request_id,attempt_number,started_at,outcome)
                VALUES(?,?,?,'attempting')
            """, (row["id"], attempt, now))
        result = dict(row)
        result["attempt_count"] = attempt
        result["payload"] = json.loads(result.pop("payload_json"))
        return result

    def finish_output_attempt(
            self, output_id: str, attempt: int, state: str, *,
            classification: str | None = None, retry_delay_seconds: float | None = None,
            external_message_id: str | None = None) -> None:
        now = utc_now()
        next_attempt = (datetime.now(UTC) + timedelta(seconds=retry_delay_seconds)).isoformat() \
            if retry_delay_seconds is not None else None
        outcome = ("accepted_by_transport" if state == "accepted_by_transport" else
                   "delivery_uncertain" if state == "delivery_uncertain" else "failed")
        with self.connection:
            self.connection.execute("""
                UPDATE output_attempts SET finished_at=?,outcome=?,failure_classification=?,
                  external_message_id=? WHERE output_request_id=? AND attempt_number=?
            """, (now, outcome, classification, external_message_id, output_id, attempt))
            self.connection.execute("""
                UPDATE output_requests SET delivery_state=?,next_attempt_at=?,
                  last_failure_classification=?,updated_at=? WHERE id=?
            """, (state, next_attempt, classification, now, output_id))
            row = self.connection.execute(
                "SELECT message_id FROM output_requests WHERE id=?", (output_id,)).fetchone()
            if row is not None and row["message_id"]:
                message_status = {
                    "accepted_by_transport": "delivered",
                    "failed_permanent": "transport_failed",
                    "delivery_uncertain": "pending_delivery",
                    "retry_wait": "pending_delivery",
                    "rejected_unavailable": "transport_failed",
                }.get(state)
                if message_status:
                    self.connection.execute(
                        "UPDATE messages SET delivery_status=? WHERE id=?",
                        (message_status, row["message_id"]))

    def generate_output_failure_event(self, output_id: str) -> bool:
        with self.connection:
            changed = self.connection.execute("""
                UPDATE output_requests SET failure_event_generated=1,updated_at=?
                WHERE id=? AND failure_event_generated=0
            """, (utc_now(), output_id)).rowcount == 1
            if not changed:
                return False
            row = self.connection.execute("""
                SELECT output_type,target,last_failure_classification,attempt_count
                FROM output_requests WHERE id=?
            """, (output_id,)).fetchone()
            event_id = f"output-failure:{output_id}"
            context = json.dumps({
                "output_id": output_id, "output_type": row["output_type"],
                "target": row["target"],
                "failure_classification": row["last_failure_classification"],
                "attempt_count": row["attempt_count"],
            }, separators=(",", ":"))
            self.connection.execute("""
                INSERT INTO scheduled_wakeups(id,due_at,reason,context_json,status,created_at)
                VALUES(?,?, 'output_delivery_failed',?,'pending',?) ON CONFLICT(id) DO NOTHING
            """, (event_id, utc_now(), context, utc_now()))
            return True

    def output_request(self, output_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM output_requests WHERE id=?", (output_id,)).fetchone()
        return None if row is None else dict(row)

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

    def spontaneous_attention_count_since(self, since: str) -> int:
        return int(self.connection.execute("""
            SELECT count(*) FROM messages
            WHERE direction='outbound' AND spontaneous=1
              AND delivery_status IN ('pending_delivery','delivered') AND created_at>=?
        """, (since,)).fetchone()[0])

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

    def start_keeper_interaction(self, run_id: str, event: WakeEvent,
                                 game_id: str, actor_id: str) -> None:
        with self.connection:
            self.connection.execute("""
                INSERT INTO keeper_interactions(
                  run_id,event_id,occurred_at,wake_source,wake_reason,wake_payload_json,
                  status,realm_game_id,realm_actor_id)
                VALUES(?,?,?,?,?,?,'running',?,?)
            """, (run_id, event.id, event.occurred_at, event.source, event.reason,
                  json.dumps(event.payload, ensure_ascii=False), game_id, actor_id))

    def set_keeper_input(self, run_id: str, session_id: str | None,
                         input_text: str, realm_snapshot: dict[str, Any] | None) -> None:
        snapshot = (json.dumps(realm_snapshot, ensure_ascii=False)
                    if realm_snapshot is not None else None)
        revision = ((realm_snapshot.get("trusted_state") or {}).get("game") or {}).get(
            "current_revision") if realm_snapshot and "trusted_state" in realm_snapshot else None
        with self.connection:
            self.connection.execute("""
                UPDATE keeper_interactions SET session_id=?,input_text=?,realm_snapshot_json=?,
                  realm_revision=?,input_bytes=?,snapshot_bytes=? WHERE run_id=?
            """, (session_id, input_text, snapshot, revision,
                  len(input_text.encode("utf-8")),
                  len(snapshot.encode("utf-8")) if snapshot is not None else 0, run_id))

    def add_keeper_activity(self, run_id: str, kind: str, content: Any, *,
                            session_id: str | None = None, turn_id: str | None = None,
                            call_id: str | None = None) -> None:
        with self.connection:
            self.connection.execute("""
                INSERT OR IGNORE INTO keeper_activity(
                  run_id,occurred_at,kind,session_id,turn_id,call_id,content_json)
                VALUES(?,?,?,?,?,?,?)
            """, (run_id, utc_now(), kind, session_id, turn_id, call_id,
                  json.dumps(content, ensure_ascii=False)))

    def finish_keeper_interaction(self, run_id: str, status: str) -> None:
        with self.connection:
            self.connection.execute("""
                UPDATE keeper_interactions SET status=?,completed_at=? WHERE run_id=?
            """, (status, utc_now(), run_id))

    def keeper_recent_context(self, count: int, byte_limit: int, *,
                              game_id: str, actor_id: str) -> list[dict[str, Any]]:
        """Newest completed wakes for this Realm game and actor that fit."""
        if count <= 0 or byte_limit <= 0:
            return []
        rows = self.connection.execute("""
            SELECT run_id,occurred_at,wake_source,wake_reason,wake_payload_json
            FROM keeper_interactions
            WHERE status='completed' AND realm_game_id=? AND realm_actor_id=?
            ORDER BY completed_at DESC, rowid DESC LIMIT ?
        """, (game_id, actor_id, count)).fetchall()
        selected: list[dict[str, Any]] = []
        def bootstrap_bytes(entries: list[dict[str, Any]]) -> int:
            # Match the field's nesting and indentation in build_managed_bootstrap.
            wrapper = {"new_session_bootstrap": {
                "preceding_field": None, "keeper_recent_interactions": entries}}
            empty = {"new_session_bootstrap": {"preceding_field": None}}
            return len(json.dumps(wrapper, ensure_ascii=False, indent=2).encode("utf-8")) - len(
                json.dumps(empty, ensure_ascii=False, indent=2).encode("utf-8"))

        for row in rows:
            activities = self.connection.execute("""
                SELECT kind,turn_id,call_id,content_json FROM keeper_activity
                WHERE run_id=? ORDER BY sequence
            """, (row["run_id"],)).fetchall()
            narrative = []
            results = {}
            for item in activities:
                if item["kind"] == "tool_result" and item["turn_id"] and item["call_id"]:
                    data = json.loads(item["content_json"])
                    if isinstance(data, dict):
                        results[(item["turn_id"], item["call_id"])] = data
            for item in activities:
                data = json.loads(item["content_json"])
                if item["kind"] == "model_turn":
                    if data.get("message"):
                        narrative.append({"kind": "keeper_output", "text": data["message"]})
                    for call in data.get("tool_calls") or []:
                        name, arguments = call.get("name"), call.get("arguments")
                        if not isinstance(arguments, dict):
                            continue
                        completion = results.get((item["turn_id"], call.get("id")))
                        if not completion or completion.get("name") != name:
                            continue
                        result = completion.get("result")
                        if not isinstance(result, dict) or result.get("ok") is not True:
                            continue
                        if name == "send_owner_message" and result.get("delivered") is True and isinstance(
                                arguments.get("content"), str):
                            narrative.append({"kind": "player_facing_call", "name": name,
                                              "content": arguments["content"]})
                        elif name == "realm_world_patch":
                            # Only explicit player projections may cross from a
                            # trusted world patch into the rollover narrative.
                            player_views = []
                            for section in ("entities", "entity_updates"):
                                items = arguments.get(section)
                                if not isinstance(items, list):
                                    continue
                                player_views.extend(
                                    item["player"] for item in items
                                    if isinstance(item, dict) and isinstance(
                                        item.get("player"), dict))
                            if player_views:
                                narrative.append({"kind": "player_facing_call", "name": name,
                                                  "player_views": player_views})
                elif item["kind"] == "tool_result":
                    result = data.get("result") or {}
                    # Never replay tool-returned Realm views or mutation bodies as facts.
                    ok = result.get("ok", "error" not in result)
                    if data.get("name") == "send_owner_message":
                        ok = ok and result.get("delivered") is True
                    narrative.append({"kind": "action_outcome", "name": data.get("name"),
                                      "ok": ok,
                                      "error_code": result.get("error_code"),
                                      "outcome": result.get("outcome")})
            entry = {"at": row["occurred_at"], "trigger": {
                "source": row["wake_source"], "reason": row["wake_reason"],
                "payload": json.loads(row["wake_payload_json"])}, "activity": narrative}
            if bootstrap_bytes([entry, *selected]) > byte_limit:
                break
            selected.append(entry)
        return list(reversed(selected))

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
