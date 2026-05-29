"""STABILITY-048 sidecar operation command log.

Segment 1 keeps this sidecar isolated from pith.db and does not route existing
concept, FTS, retrieval, or raw-capture writes through it.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from app.core.profile import resolve_data_dir
from app.storage.operation_classes import CommandStatus, OperationClass, OperationCommand, utc_now_iso

DEFAULT_COMMAND_LOG_NAME = "pith-command-log.db"
DEFAULT_BUSY_TIMEOUT_MS = 250
DEFAULT_MAX_ATTEMPTS = 3
MAX_ERROR_TEXT_CHARS = 512

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS operation_commands (
  command_id TEXT PRIMARY KEY,
  idempotency_key TEXT NOT NULL UNIQUE,
  origin_id TEXT,
  session_id TEXT,
  agent_id TEXT,
  causal_turn_id TEXT,
  op_class TEXT NOT NULL,
  priority INTEGER NOT NULL,
  freshness_class TEXT NOT NULL,
  deadline_ms INTEGER,
  payload_type TEXT NOT NULL,
  payload_version INTEGER NOT NULL,
  payload_json TEXT NOT NULL,
  status TEXT NOT NULL,
  result_json TEXT,
  not_before TEXT,
  lease_owner TEXT,
  lease_expires_at TEXT,
  attempt_count INTEGER NOT NULL DEFAULT 0,
  producer TEXT,
  last_error_code TEXT,
  last_error TEXT,
  discard_reason TEXT,
  max_attempts INTEGER NOT NULL DEFAULT 3,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_operation_commands_claim
  ON operation_commands(status, op_class, priority, not_before, created_at);
CREATE INDEX IF NOT EXISTS idx_operation_commands_lease
  ON operation_commands(lease_owner, lease_expires_at);
CREATE INDEX IF NOT EXISTS idx_operation_commands_origin_session
  ON operation_commands(origin_id, session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_operation_commands_freshness
  ON operation_commands(freshness_class, status, created_at);
CREATE TABLE IF NOT EXISTS operation_command_meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
"""


def command_log_path(data_dir: Path | None = None) -> Path:
    explicit = os.environ.get("PITH_OPERATION_COMMAND_LOG_PATH")
    if explicit:
        return Path(explicit)
    return (data_dir or resolve_data_dir()) / DEFAULT_COMMAND_LOG_NAME


@dataclass(frozen=True)
class AppendResult:
    command_id: str
    status: CommandStatus
    created: bool
    append_latency_ms: float | None = None


class CommandLog:
    def __init__(
        self,
        path: Path | str | None = None,
        *,
        busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
        initialize: bool = True,
    ) -> None:
        self.path = Path(path) if path is not None else command_log_path()
        self.busy_timeout_ms = max(0, int(busy_timeout_ms))
        if initialize:
            self.initialize()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path), timeout=max(0.001, self.busy_timeout_ms / 1000.0))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)
            self._ensure_columns(conn)
            conn.execute(
                """CREATE INDEX IF NOT EXISTS idx_operation_commands_producer_status_reason
                   ON operation_commands(producer, status, last_error_code, discard_reason, created_at)"""
            )
            now = utc_now_iso()
            conn.execute(
                """INSERT INTO operation_command_meta(key, value, updated_at)
                   VALUES ('schema_version', '1', ?)
                   ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at""",
                (now,),
            )

    def _ensure_columns(self, conn: sqlite3.Connection) -> None:
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(operation_commands)").fetchall()}
        additions = {
            "producer": "TEXT",
            "last_error_code": "TEXT",
            "discard_reason": "TEXT",
            "max_attempts": f"INTEGER NOT NULL DEFAULT {DEFAULT_MAX_ATTEMPTS}",
        }
        for name, ddl in additions.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE operation_commands ADD COLUMN {name} {ddl}")

    def append(
        self,
        command: OperationCommand,
        *,
        producer: str | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> AppendResult:
        now = utc_now_iso()
        started = time.perf_counter()
        try:
            with self._connect() as conn:
                cursor = conn.execute(
                    """INSERT INTO operation_commands (
                        command_id, idempotency_key, origin_id, session_id, agent_id, causal_turn_id,
                        op_class, priority, freshness_class, deadline_ms, payload_type, payload_version,
                        payload_json, status, not_before, producer, max_attempts, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        command.command_id,
                        command.idempotency_key,
                        command.origin_id,
                        command.session_id,
                        command.agent_id,
                        command.causal_turn_id,
                        command.op_class.value,
                        command.priority,
                        command.freshness_class.value,
                        command.deadline_ms,
                        command.payload_type,
                        command.payload_version,
                        command.payload_text(),
                        CommandStatus.QUEUED.value,
                        command.not_before,
                        producer,
                        max(1, int(max_attempts)),
                        command.created_at,
                        now,
                    ),
                )
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                return AppendResult(command.command_id, CommandStatus.QUEUED, cursor.rowcount == 1, elapsed_ms)
        except sqlite3.IntegrityError:
            existing = self.get_by_idempotency_key(command.idempotency_key)
            if existing is None:
                raise
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return AppendResult(
                command_id=str(existing["command_id"]),
                status=CommandStatus(str(existing["status"])),
                created=False,
                append_latency_ms=elapsed_ms,
            )

    def get_by_idempotency_key(self, idempotency_key: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM operation_commands WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            return dict(row) if row else None

    def claim(
        self,
        *,
        lease_owner: str,
        op_classes: Iterable[OperationClass] | None = None,
        limit: int = 10,
        lease_seconds: int = 30,
        now: str | None = None,
    ) -> list[dict[str, Any]]:
        now = now or utc_now_iso()
        expires_at = (datetime.fromisoformat(now) + timedelta(seconds=max(1, lease_seconds))).isoformat()
        class_values = [op.value for op in op_classes] if op_classes else []
        placeholders = ",".join("?" for _ in class_values)
        class_clause = f"AND op_class IN ({placeholders})" if class_values else ""
        params: list[Any] = [now, *class_values, max(1, int(limit))]

        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                f"""SELECT * FROM operation_commands
                    WHERE status IN ('queued', 'deferred')
                      AND (not_before IS NULL OR not_before <= ?)
                      {class_clause}
                    ORDER BY priority DESC, created_at ASC
                    LIMIT ?""",
                params,
            ).fetchall()
            command_ids = [row["command_id"] for row in rows]
            if command_ids:
                marks = ",".join("?" for _ in command_ids)
                conn.execute(
                    f"""UPDATE operation_commands
                        SET status='claimed',
                            lease_owner=?,
                            lease_expires_at=?,
                            attempt_count=attempt_count+1,
                            updated_at=?
                        WHERE command_id IN ({marks})""",
                    [lease_owner, expires_at, now, *command_ids],
                )
            conn.commit()
            return [dict(row) | {"status": CommandStatus.CLAIMED.value, "lease_owner": lease_owner} for row in rows]

    def release_expired_leases(self, *, now: str | None = None) -> int:
        now = now or utc_now_iso()
        with self._connect() as conn:
            cursor = conn.execute(
                """UPDATE operation_commands
                   SET status='queued', lease_owner=NULL, lease_expires_at=NULL, updated_at=?
                   WHERE status='claimed' AND lease_expires_at IS NOT NULL AND lease_expires_at <= ?""",
                (now, now),
            )
            return int(cursor.rowcount or 0)

    def mark_applied(self, command_id: str, *, result: dict[str, Any] | None = None) -> None:
        self._mark(command_id, CommandStatus.APPLIED, result=result)

    def mark_failed(
        self,
        command_id: str,
        error: str,
        *,
        retry_at: str | None = None,
        error_code: str | None = None,
    ) -> None:
        status = CommandStatus.DEFERRED if retry_at else CommandStatus.FAILED
        self._mark(command_id, status, error=error, error_code=error_code, not_before=retry_at)

    def mark_discarded(self, command_id: str, reason: str, *, error: str | None = None) -> None:
        self._mark(
            command_id,
            CommandStatus.DISCARDED,
            error=error or reason,
            error_code=reason,
            discard_reason=reason,
        )

    def _mark(
        self,
        command_id: str,
        status: CommandStatus,
        *,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        error_code: str | None = None,
        discard_reason: str | None = None,
        not_before: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """UPDATE operation_commands
                   SET status=?, result_json=?, last_error_code=?, last_error=?, discard_reason=?, not_before=?,
                       lease_owner=NULL, lease_expires_at=NULL, updated_at=?
                   WHERE command_id=?""",
                (
                    status.value,
                    json.dumps(result or {}, sort_keys=True) if result is not None else None,
                    error_code,
                    _redact_error(error),
                    discard_reason,
                    not_before,
                    utc_now_iso(),
                    command_id,
                ),
            )

    def health(self) -> dict[str, Any]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT producer, op_class, status, COALESCE(last_error_code, discard_reason, '') AS reason,
                          COUNT(*) AS count, MIN(created_at) AS oldest
                   FROM operation_commands
                   GROUP BY producer, op_class, status, reason
                   ORDER BY producer, op_class, status, reason"""
            ).fetchall()
        return {
            "path": str(self.path),
            "available": True,
            "queue": [dict(row) for row in rows],
        }


def _redact_error(error: str | None) -> str | None:
    if error is None:
        return None
    cleaned = "".join(ch if ch.isprintable() else " " for ch in str(error))
    return cleaned[:MAX_ERROR_TEXT_CHARS]
