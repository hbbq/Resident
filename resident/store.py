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
    "model.responded": ("tool_call_count", "has_message", "input_tokens", "output_tokens",
                        "cached_input_tokens"),
    "tool.called": ("name",),
    "tool.completed": ("name",),
    "disposition.generated": ("disposition_id", "response_id", "validation_state", "output_count"),
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
        "response_id", "tool_result_count", "event_id"),
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
        try:
            self._create_schema()
        except Exception:
            self.connection.close()
            raise

    def close(self) -> None:
        self.connection.close()

    def _create_schema(self) -> None:
        # Runtime databases are disposable. There is deliberately no upgrade path.
        version = self.connection.execute("PRAGMA user_version").fetchone()[0]
        tables = self.connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        if tables and version != 100:
            raise ValueError("Unsupported Resident database; stop Resident and delete its disposable instance database")
        self.connection.executescript("""

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
          status TEXT NOT NULL, conversation_id TEXT, input_text TEXT,
          realm_snapshot_json TEXT, realm_game_id TEXT, realm_actor_id TEXT,
          realm_revision INTEGER, input_bytes INTEGER, snapshot_bytes INTEGER,
          input_tokens INTEGER NOT NULL DEFAULT 0, output_tokens INTEGER NOT NULL DEFAULT 0,
          completed_at TEXT);
        CREATE TABLE IF NOT EXISTS keeper_activity(
          sequence INTEGER PRIMARY KEY AUTOINCREMENT,
          run_id TEXT NOT NULL REFERENCES keeper_interactions(run_id),
          occurred_at TEXT NOT NULL, kind TEXT NOT NULL, conversation_id TEXT,
          response_id TEXT, call_id TEXT, content_json TEXT NOT NULL,
          UNIQUE(run_id,kind,conversation_id,response_id,call_id));
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
        CREATE TABLE IF NOT EXISTS owner_guidance(
          id TEXT PRIMARY KEY, content TEXT NOT NULL, status TEXT NOT NULL
            CHECK(status IN ('active','superseded','removed')),
          revision INTEGER NOT NULL, source_message_id TEXT,
          created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS owner_guidance_revisions(
          guidance_id TEXT NOT NULL REFERENCES owner_guidance(id), revision INTEGER NOT NULL,
          operation TEXT NOT NULL CHECK(operation IN ('set','remove')), content TEXT NOT NULL,
          source_message_id TEXT, created_at TEXT NOT NULL,
          PRIMARY KEY(guidance_id,revision));
        CREATE TABLE IF NOT EXISTS realm_mutation_requests(
          idempotency_key TEXT PRIMARY KEY, path TEXT NOT NULL,
          body_json TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS final_dispositions(
          id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL,
          response_id TEXT NOT NULL, run_id TEXT, wake_id TEXT,
          schema_fingerprint TEXT NOT NULL, raw_disposition TEXT,
          normalized_disposition_json TEXT, validation_state TEXT NOT NULL,
          created_at TEXT NOT NULL, UNIQUE(conversation_id,response_id));
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
        CREATE INDEX IF NOT EXISTS idx_output_dispatch
          ON output_requests(delivery_state,next_attempt_at,created_at);
        CREATE TABLE IF NOT EXISTS conversation_binding(
          singleton INTEGER PRIMARY KEY CHECK(singleton=1), conversation_id TEXT NOT NULL,
          created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS response_steps(
          id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES wake_runs(id),
          conversation_id TEXT NOT NULL, round INTEGER NOT NULL, wake_json TEXT NOT NULL,
          schema_json TEXT NOT NULL, response_id TEXT, turn_json TEXT,
          status TEXT NOT NULL CHECK(status IN ('submitting','returned','settled','rejected')),
          created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS tool_executions(
          conversation_id TEXT NOT NULL, call_id TEXT NOT NULL, response_id TEXT NOT NULL,
          name TEXT NOT NULL, arguments_json TEXT NOT NULL,
          status TEXT NOT NULL CHECK(status IN ('pending','completed')),
          output_json TEXT, attachments_ephemeral INTEGER NOT NULL DEFAULT 0,
          created_at TEXT NOT NULL, completed_at TEXT,
          PRIMARY KEY(conversation_id,call_id));
        CREATE INDEX IF NOT EXISTS idx_response_steps_status ON response_steps(status);

        PRAGMA user_version=100;
        """)
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


    def owner_guidance_revision(self, guidance_id: str) -> int | None:
        row = self.connection.execute("""
            SELECT MAX(revision) AS revision FROM owner_guidance_revisions
            WHERE guidance_id=?
        """, (guidance_id,)).fetchone()
        return None if row is None or row["revision"] is None else int(row["revision"])


    def active_owner_guidance(self) -> list[dict[str, Any]]:
        entries = [dict(row) for row in self.connection.execute("""
            SELECT id,content,revision,updated_at FROM owner_guidance
            WHERE status='active' ORDER BY updated_at,id
        """)]
        _validate_owner_guidance_projection(entries)
        return entries

    def set_owner_guidance(self, content: str, *, guidance_id: str | None = None,
                           source_message_id: str | None = None) -> str:
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
                INSERT INTO owner_guidance(id,content,status,revision,source_message_id,created_at,updated_at) VALUES(?,?,'active',?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET content=excluded.content,status='active',
                  revision=excluded.revision,source_message_id=excluded.source_message_id,updated_at=excluded.updated_at
            """, (guidance_id, content, revision,
                  source_message_id, now, now))
            self.connection.execute("""
                INSERT INTO owner_guidance_revisions(
                  guidance_id,revision,operation,content,source_message_id,created_at)
                VALUES(?,?,'set',?,?,?)
            """, (guidance_id, revision, content, source_message_id, now))
        return guidance_id

    def remove_owner_guidance(self, guidance_id: str, *,
                                 source_message_id: str | None = None) -> bool:
        with self.connection:
            current = self.connection.execute("""
                SELECT content,revision,source_message_id FROM owner_guidance
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
                  guidance_id,revision,operation,content,source_message_id,created_at)
                VALUES(?,?,'remove',?,?,?)
            """, (guidance_id, current["revision"] + 1, current["content"],
                  source_message_id, utc_now()))
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


    @staticmethod
    def _disposition_id(conversation_id: str, response_id: str) -> str:
        return hashlib.sha256(
            f"{conversation_id}\0{response_id}".encode()).hexdigest()

    def has_final_disposition(self, conversation_id: str, response_id: str) -> bool:
        return self.connection.execute("""
            SELECT 1 FROM final_dispositions
            WHERE conversation_id=? AND response_id=?
        """, (conversation_id, response_id)).fetchone() is not None

    def persist_final_disposition(
            self, conversation_id: str, response_id: str,
            schema_fingerprint: str, raw_disposition: str | None,
            normalized: dict[str, Any] | None, validation_state: str,
            jobs: list[dict[str, Any]], *, run_id: str | None = None,
            wake_id: str | None = None) -> dict[str, Any]:
        disposition_id = self._disposition_id(conversation_id, response_id)
        now = utc_now()
        normalized_json = (json.dumps(normalized, ensure_ascii=False, sort_keys=True,
                                      separators=(",", ":"))
                           if normalized is not None else None)
        created_requests: list[dict[str, Any]] = []
        with self.connection:
            inserted = self.connection.execute("""
                INSERT INTO final_dispositions(
                  id,conversation_id,response_id,run_id,wake_id,schema_fingerprint,
                  raw_disposition,normalized_disposition_json,validation_state,created_at)
                VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(conversation_id,response_id) DO NOTHING
            """, (disposition_id, conversation_id, response_id, run_id, wake_id,
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
            if run_id is not None:
                self.connection.execute("UPDATE response_steps SET status='settled' WHERE run_id=?", (run_id,))
                if validation_state == 'valid':
                    self.connection.execute("UPDATE wake_runs SET status='completed',finished_at=? WHERE id=?", (now, run_id))
                    step = self.connection.execute("SELECT wake_json FROM response_steps WHERE run_id=? LIMIT 1", (run_id,)).fetchone()
                    if step:
                        wake = json.loads(step['wake_json'])
                        if wake['source'] == 'owner':
                            self.connection.execute("UPDATE owner_message_processing SET status='completed',completed_at=? WHERE message_id=?", (now, wake['payload'].get('message_id')))
                        if wake['source'] == 'scheduler':
                            self.connection.execute("UPDATE scheduled_wakeups SET status='completed' WHERE id=?", (wake['payload'].get('schedule_id'),))
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

    def set_keeper_input(self, run_id: str, conversation_id: str | None,
                         input_text: str, realm_snapshot: dict[str, Any] | None) -> None:
        snapshot = (json.dumps(realm_snapshot, ensure_ascii=False)
                    if realm_snapshot is not None else None)
        revision = ((realm_snapshot.get("trusted_state") or {}).get("game") or {}).get(
            "current_revision") if realm_snapshot and "trusted_state" in realm_snapshot else None
        with self.connection:
            self.connection.execute("""
                UPDATE keeper_interactions SET conversation_id=?,input_text=?,realm_snapshot_json=?,
                  realm_revision=?,input_bytes=?,snapshot_bytes=? WHERE run_id=?
            """, (conversation_id, input_text, snapshot, revision,
                  len(input_text.encode("utf-8")),
                  len(snapshot.encode("utf-8")) if snapshot is not None else 0, run_id))

    def add_keeper_activity(self, run_id: str, kind: str, content: Any, *,
                            conversation_id: str | None = None, response_id: str | None = None,
                            call_id: str | None = None) -> None:
        with self.connection:
            self.connection.execute("""
                INSERT OR IGNORE INTO keeper_activity(
                  run_id,occurred_at,kind,conversation_id,response_id,call_id,content_json)
                VALUES(?,?,?,?,?,?,?)
            """, (run_id, utc_now(), kind, conversation_id, response_id, call_id,
                  json.dumps(content, ensure_ascii=False)))

    def finish_keeper_interaction(self, run_id: str, status: str) -> None:
        with self.connection:
            self.connection.execute("""
                UPDATE keeper_interactions SET status=?,completed_at=? WHERE run_id=?
            """, (status, utc_now(), run_id))


    def finish_run(self, run_id: str, status: str, duration: float, model_calls: int,
                   schedule_id: str | None = None,
                   owner_message_id: str | None = None) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE wake_runs SET finished_at=?,status=CASE WHEN status='completed' THEN status ELSE ? END,duration_seconds=?,model_calls=? WHERE id=?",
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

    def conversation_id(self) -> str | None:
        row = self.connection.execute("SELECT conversation_id FROM conversation_binding WHERE singleton=1").fetchone()
        return row[0] if row else None

    def bind_conversation(self, conversation_id: str) -> None:
        with self.connection:
            self.connection.execute("INSERT INTO conversation_binding VALUES(1,?,?)", (conversation_id, utc_now()))

    def completed_event_run(self, event_id: str) -> str | None:
        row = self.connection.execute("SELECT id FROM wake_runs WHERE event_id=? AND status='completed'", (event_id,)).fetchone()
        return row[0] if row else None

    def begin_response_step(self, run_id: str, conversation_id: str, round_number: int,
                            wake: WakeEvent, schema: dict) -> str:
        from dataclasses import asdict
        step_id = str(uuid.uuid4())
        with self.connection:
            self.connection.execute("""
                INSERT INTO response_steps(id,run_id,conversation_id,round,wake_json,schema_json,status,created_at)
                VALUES(?,?,?,?,?,?,'submitting',?)
            """, (step_id, run_id, conversation_id, round_number, json.dumps(asdict(wake)),
                  json.dumps(schema), utc_now()))
        return step_id

    def record_response(self, step_id: str, turn: Any) -> None:
        from dataclasses import asdict
        with self.connection:
            self.connection.execute("UPDATE response_steps SET response_id=?,turn_json=?,status='returned' WHERE id=?",
                                    (turn.response_id, json.dumps(asdict(turn)), step_id))

    def note_response_id(self, step_id: str, response_id: str) -> None:
        with self.connection:
            self.connection.execute("UPDATE response_steps SET response_id=? WHERE id=?",
                                    (response_id, step_id))

    def reject_response_step(self, step_id: str) -> None:
        with self.connection:
            self.connection.execute("UPDATE response_steps SET status='rejected' WHERE id=?", (step_id,))

    def unfinished_response_steps(self) -> list[dict]:
        return [dict(row) for row in self.connection.execute(
            "SELECT * FROM response_steps WHERE status IN ('submitting','returned') ORDER BY created_at,round")]

    def begin_tool_execution(self, conversation_id: str, response_id: str,
                             call_id: str, name: str, arguments: dict) -> dict:
        encoded = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
        with self.connection:
            claimed = self.connection.execute("""
                INSERT INTO tool_executions(conversation_id,call_id,response_id,name,arguments_json,status,created_at)
                VALUES(?,?,?,?,?,'pending',?) ON CONFLICT DO NOTHING
            """, (conversation_id, call_id, response_id, name, encoded, utc_now())).rowcount == 1
            row = self.connection.execute("SELECT * FROM tool_executions WHERE conversation_id=? AND call_id=?",
                                          (conversation_id, call_id)).fetchone()
        if row['name'] != name or row['arguments_json'] != encoded:
            raise RuntimeError("Function call ID reused with different arguments")
        return {'claimed': claimed, 'status': row['status'],
                'output': json.loads(row['output_json']) if row['output_json'] else None,
                'attachments_ephemeral': bool(row['attachments_ephemeral'])}

    def complete_tool_execution(self, conversation_id: str, call_id: str, output: dict,
                                attachments_ephemeral: bool = False) -> None:
        with self.connection:
            self.connection.execute("""
                UPDATE tool_executions SET status='completed',output_json=?,attachments_ephemeral=?,completed_at=?
                WHERE conversation_id=? AND call_id=? AND status='pending'
            """, (json.dumps(output), int(attachments_ephemeral), utc_now(), conversation_id, call_id))
