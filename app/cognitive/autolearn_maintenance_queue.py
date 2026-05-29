"""Queue-only facade for deferred autolearn maintenance.

This module stays below the session layer so cognitive write paths can enqueue
secondary maintenance without violating import boundaries.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

from app.core.datetime_utils import _utc_now_iso

logger = logging.getLogger(__name__)

TASK_GOVERNANCE_RECOMPUTE = "governance_recompute"
TASK_SUBJECT_KEY_SUPERSESSION = "subject_key_supersession"
TASK_SIMILARITY_SUPERSESSION = "similarity_supersession"
VALID_TASK_TYPES = {
    TASK_GOVERNANCE_RECOMPUTE,
    TASK_SUBJECT_KEY_SUPERSESSION,
    TASK_SIMILARITY_SUPERSESSION,
}

AUTOLEARN_MAINTENANCE_DDL = """
CREATE TABLE IF NOT EXISTS autolearn_maintenance_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    concept_id TEXT NOT NULL,
    concept_version TEXT NOT NULL DEFAULT 'v1',
    task_type TEXT NOT NULL CHECK (
        task_type IN ('governance_recompute', 'subject_key_supersession', 'similarity_supersession')
    ),
    status TEXT NOT NULL DEFAULT 'queued' CHECK (
        status IN ('queued', 'running', 'done', 'skipped', 'failed')
    ),
    attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_error TEXT,
    source TEXT
);

CREATE INDEX IF NOT EXISTS idx_autolearn_maintenance_status_next
    ON autolearn_maintenance_queue(status, next_attempt_at, created_at);
CREATE INDEX IF NOT EXISTS idx_autolearn_maintenance_concept
    ON autolearn_maintenance_queue(concept_id, concept_version);
CREATE UNIQUE INDEX IF NOT EXISTS idx_autolearn_maintenance_open_unique
    ON autolearn_maintenance_queue(concept_id, concept_version, task_type)
    WHERE status IN ('queued', 'running');
"""

_drain_kicker: Callable[[], bool] | None = None


def register_autolearn_maintenance_drain_kicker(callback: Callable[[], bool]) -> None:
    """Register the session-layer drain scheduler without importing session here."""
    global _drain_kicker
    _drain_kicker = callback


def kick_autolearn_maintenance_drain() -> bool:
    """Ask the registered session-layer scheduler to drain a tiny batch."""
    if _drain_kicker is None:
        return False
    try:
        return bool(_drain_kicker())
    except Exception as exc:
        logger.debug("autolearn maintenance drain kick callback failed: %s", exc)
        return False


def ensure_autolearn_maintenance_tables(conn) -> None:
    """Create the autolearn maintenance queue table and indexes."""
    conn.executescript(AUTOLEARN_MAINTENANCE_DDL)
    conn.commit()


def _error_text(exc: BaseException | str | None) -> str | None:
    if exc is None:
        return None
    return str(exc)[:500]


def _supersession_disabled() -> bool:
    return os.environ.get("PITH_DISABLE_EVOLVE", "").lower() in ("true", "1")


def _insert_queue_rows(conn, *, concept_id: str, concept_version: str, task_types: list[str], source: str) -> int:
    now = _utc_now_iso()
    queued = 0
    for task_type in task_types:
        if task_type not in VALID_TASK_TYPES:
            raise ValueError(f"Unsupported autolearn maintenance task_type: {task_type}")
        cur = conn.execute(
            """INSERT OR IGNORE INTO autolearn_maintenance_queue
               (concept_id, concept_version, task_type, status, attempts,
                next_attempt_at, created_at, updated_at, source)
               VALUES (?, ?, ?, 'queued', 0, NULL, ?, ?, ?)""",
            (concept_id, concept_version, task_type, now, now, source),
        )
        queued += int(cur.rowcount or 0)
    conn.commit()
    return queued


def enqueue_autolearn_maintenance(
    concept_id: str,
    concept_version: str = "v1",
    *,
    source: str = "autolearn",
    include_similarity: bool = True,
) -> dict[str, Any]:
    """Queue governance and optional similarity work without blocking learning."""
    from app.core.config import get_autolearn_maintenance_enabled, get_autolearn_maintenance_enqueue_timeout_s
    from app.storage import db_immediate

    if not get_autolearn_maintenance_enabled():
        return {"queued": 0, "skipped": True, "reason": "disabled"}

    task_types = [TASK_GOVERNANCE_RECOMPUTE]
    if include_similarity and not _supersession_disabled():
        task_types.append(TASK_SIMILARITY_SUPERSESSION)

    try:
        with db_immediate(
            timeout_s=get_autolearn_maintenance_enqueue_timeout_s(),
            operation="autolearn_maintenance_enqueue",
        ) as conn:
            ensure_autolearn_maintenance_tables(conn)
            queued = _insert_queue_rows(
                conn,
                concept_id=concept_id,
                concept_version=concept_version or "v1",
                task_types=task_types,
                source=source,
            )
        return {"queued": queued, "skipped": False}
    except Exception as exc:
        logger.warning("autolearn_maintenance_enqueue failed for %s (non-fatal): %s", concept_id, exc)
        return {"queued": 0, "skipped": True, "error": _error_text(exc)}


def enqueue_subject_key_supersession(
    concept_id: str,
    concept_version: str = "v1",
    *,
    source: str = "autolearn_subject_key_fallback",
) -> dict[str, Any]:
    """Queue deterministic subject-key supersession fallback."""
    from app.core.config import get_autolearn_maintenance_enabled, get_autolearn_maintenance_enqueue_timeout_s
    from app.storage import db_immediate

    if not get_autolearn_maintenance_enabled():
        return {"queued": 0, "skipped": True, "reason": "disabled"}

    try:
        with db_immediate(
            timeout_s=get_autolearn_maintenance_enqueue_timeout_s(),
            operation="autolearn_maintenance_enqueue_subject_key",
        ) as conn:
            ensure_autolearn_maintenance_tables(conn)
            queued = _insert_queue_rows(
                conn,
                concept_id=concept_id,
                concept_version=concept_version or "v1",
                task_types=[TASK_SUBJECT_KEY_SUPERSESSION],
                source=source,
            )
        return {"queued": queued, "skipped": False}
    except Exception as exc:
        logger.warning("autolearn subject-key fallback enqueue failed for %s (non-fatal): %s", concept_id, exc)
        return {"queued": 0, "skipped": True, "error": _error_text(exc)}


def _row_mapping(row, columns: tuple[str, ...]) -> dict[str, Any]:
    if row is None:
        return {}
    if hasattr(row, "keys"):
        return {key: row[key] for key in row.keys()}
    return {column: row[index] for index, column in enumerate(columns)}


def get_autolearn_maintenance_status(conn=None) -> dict[str, Any]:
    """Return queue health counts for diagnostics."""
    close_conn = False
    if conn is None:
        from app.storage import owned_connection

        ctx = owned_connection()
        conn = ctx.__enter__()
        close_conn = True
    else:
        ctx = None
    try:
        ensure_autolearn_maintenance_tables(conn)
        counts = [
            _row_mapping(row, ("task_type", "status", "count"))
            for row in conn.execute(
                """SELECT task_type, status, COUNT(*) AS count
                   FROM autolearn_maintenance_queue
                   GROUP BY task_type, status
                   ORDER BY task_type, status"""
            ).fetchall()
        ]
        oldest = conn.execute(
            "SELECT MIN(created_at) AS oldest_queued_at FROM autolearn_maintenance_queue WHERE status='queued'"
        ).fetchone()
        newest_failure = conn.execute(
            """SELECT updated_at, task_type, concept_id, last_error
               FROM autolearn_maintenance_queue
               WHERE status='failed'
               ORDER BY updated_at DESC LIMIT 1"""
        ).fetchone()
        max_attempts = conn.execute("SELECT MAX(attempts) AS max_attempts FROM autolearn_maintenance_queue").fetchone()
        oldest_data = _row_mapping(oldest, ("oldest_queued_at",))
        failure_data = _row_mapping(newest_failure, ("updated_at", "task_type", "concept_id", "last_error"))
        attempts_data = _row_mapping(max_attempts, ("max_attempts",))
        return {
            "counts": counts,
            "oldest_queued_at": oldest_data.get("oldest_queued_at"),
            "newest_failure": failure_data or None,
            "max_attempts": attempts_data.get("max_attempts") or 0,
        }
    finally:
        if close_conn and ctx is not None:
            ctx.__exit__(None, None, None)
