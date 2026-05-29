"""INGEST-060-P1 raw turn capture and ingestion ledger helpers."""

from __future__ import annotations

import hashlib
import os
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Any

VALID_SOURCES = {"conversation_turn", "session_end"}
VALID_LEARNING_STATUSES = {"not_started", "attempted", "skipped", "failed"}
DEFAULT_RETENTION_DAYS = 30


def raw_capture_enabled() -> bool:
    """Return whether full raw turn capture is enabled for new writes."""
    return os.environ.get("PITH_FULL_CAPTURE_MODE", "").lower() in {"1", "true", "yes"}


def raw_capture_retention_days() -> int:
    """Read the raw capture retention window from env, defaulting to 30 days."""
    raw = os.environ.get("PITH_RAW_CAPTURE_RETENTION_DAYS", str(DEFAULT_RETENTION_DAYS))
    try:
        return int(raw)
    except (TypeError, ValueError):
        return DEFAULT_RETENTION_DAYS


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _hash_payload(source: str, user_message: str | None, assistant_response: str | None) -> str:
    h = hashlib.sha256()
    h.update(source.encode("utf-8"))
    h.update(b"\0")
    h.update((user_message or "").encode("utf-8"))
    h.update(b"\0")
    h.update((assistant_response or "").encode("utf-8"))
    return h.hexdigest()


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def capture_raw_turn(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    turn_id: str,
    source: str,
    user_message: str | None,
    assistant_response: str | None,
    retention_days: int = DEFAULT_RETENTION_DAYS,
) -> dict[str, Any]:
    """Persist a raw turn payload and ledger row in one transaction."""
    if source not in VALID_SOURCES:
        raise ValueError(f"invalid raw turn source: {source}")
    if not session_id:
        raise ValueError("session_id is required")
    if not turn_id:
        raise ValueError("turn_id is required")

    now = _utc_now_iso()
    user_text = user_message or ""
    assistant_text = assistant_response or ""
    content_hash = _hash_payload(source, user_text, assistant_text)

    cursor = conn.execute(
        """INSERT OR IGNORE INTO raw_turn_payloads
           (session_id, turn_id, source, user_message, assistant_response,
            message_len, response_len, content_hash, captured_at, retention_days)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            session_id,
            turn_id,
            source,
            user_text,
            assistant_text,
            len(user_text),
            len(assistant_text),
            content_hash,
            now,
            retention_days,
        ),
    )
    capture_status = "captured" if cursor.rowcount else "duplicate"
    row = conn.execute(
        """SELECT id FROM raw_turn_payloads
           WHERE session_id=? AND turn_id=? AND source=? AND content_hash=?
           ORDER BY id DESC LIMIT 1""",
        (session_id, turn_id, source, content_hash),
    ).fetchone()
    raw_payload_id = int(row[0]) if row else None

    conn.execute(
        """INSERT INTO turn_ingestion_ledger
           (session_id, turn_id, source, raw_payload_id, capture_status,
            learning_status, concepts_extracted, error, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, 'not_started', 0, NULL, ?, ?)""",
        (session_id, turn_id, source, raw_payload_id, capture_status, now, now),
    )
    return {"status": capture_status, "raw_payload_id": raw_payload_id, "error": None}


def capture_raw_turn_default_db(
    *,
    session_id: str,
    turn_id: str,
    source: str,
    user_message: str | None,
    assistant_response: str | None,
    retention_days: int = DEFAULT_RETENTION_DAYS,
) -> dict[str, Any]:
    """Persist a raw turn payload using the application default storage connection."""
    from app.storage import _db

    with _db() as conn:
        return capture_raw_turn(
            conn,
            session_id=session_id,
            turn_id=turn_id,
            source=source,
            user_message=user_message,
            assistant_response=assistant_response,
            retention_days=retention_days,
        )


def mark_learning_status(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    turn_id: str,
    source: str,
    status: str,
    concepts_extracted: int = 0,
    error: str | None = None,
) -> None:
    """Set ledger learning status for the latest matching raw capture row."""
    if source not in VALID_SOURCES:
        raise ValueError(f"invalid raw turn source: {source}")
    if status not in VALID_LEARNING_STATUSES:
        raise ValueError(f"invalid learning status: {status}")

    conn.execute(
        """UPDATE turn_ingestion_ledger
           SET learning_status=?,
               concepts_extracted=?,
               error=?,
               updated_at=?
           WHERE id = (
               SELECT id FROM turn_ingestion_ledger
               WHERE session_id=? AND turn_id=? AND source=?
               ORDER BY id DESC LIMIT 1
           )""",
        (
            status,
            max(0, int(concepts_extracted or 0)),
            error,
            _utc_now_iso(),
            session_id,
            turn_id,
            source,
        ),
    )


def mark_learning_status_default_db(
    *,
    session_id: str,
    turn_id: str,
    source: str,
    status: str,
    concepts_extracted: int = 0,
    error: str | None = None,
) -> None:
    """Update the turn ingestion ledger using the application default storage connection."""
    from app.storage import _db

    with _db() as conn:
        mark_learning_status(
            conn,
            session_id=session_id,
            turn_id=turn_id,
            source=source,
            status=status,
            concepts_extracted=concepts_extracted,
            error=error,
        )


def get_ingestion_capture_summary(
    conn: sqlite3.Connection,
    *,
    recent_limit: int = 50,
) -> dict[str, Any]:
    """Return raw capture health without exposing raw transcript text."""
    enabled = raw_capture_enabled()
    retention_days = raw_capture_retention_days()
    if not _table_exists(conn, "raw_turn_payloads") or not _table_exists(conn, "turn_ingestion_ledger"):
        return {
            "enabled": enabled,
            "retention_days": retention_days,
            "raw_payload_rows": 0,
            "ledger_rows": 0,
            "captured_24h": 0,
            "capture_failed_24h": 0,
            "learning_failed_24h": 0,
            "oldest_unpurged_raw_captured_at": None,
            "recent_failures": [],
        }

    raw_payload_rows = conn.execute("SELECT COUNT(*) FROM raw_turn_payloads").fetchone()[0]
    ledger_rows = conn.execute("SELECT COUNT(*) FROM turn_ingestion_ledger").fetchone()[0]
    captured_24h = conn.execute(
        """SELECT COUNT(*) FROM turn_ingestion_ledger
           WHERE julianday(created_at) > julianday('now', '-1 day')
             AND capture_status IN ('captured', 'duplicate')"""
    ).fetchone()[0]
    capture_failed_24h = conn.execute(
        """SELECT COUNT(*) FROM turn_ingestion_ledger
           WHERE julianday(created_at) > julianday('now', '-1 day')
             AND capture_status = 'failed'"""
    ).fetchone()[0]
    learning_failed_24h = conn.execute(
        """SELECT COUNT(*) FROM turn_ingestion_ledger
           WHERE julianday(updated_at) > julianday('now', '-1 day')
             AND learning_status = 'failed'"""
    ).fetchone()[0]
    oldest = conn.execute(
        """SELECT MIN(captured_at) FROM raw_turn_payloads
           WHERE purged_at IS NULL
             AND (user_message IS NOT NULL OR assistant_response IS NOT NULL)"""
    ).fetchone()[0]
    failures = conn.execute(
        """SELECT session_id, turn_id, source, capture_status, learning_status, error, updated_at
           FROM turn_ingestion_ledger
           WHERE capture_status='failed' OR learning_status='failed'
           ORDER BY updated_at DESC
           LIMIT ?""",
        (recent_limit,),
    ).fetchall()

    return {
        "enabled": enabled,
        "retention_days": retention_days,
        "raw_payload_rows": raw_payload_rows,
        "ledger_rows": ledger_rows,
        "captured_24h": captured_24h,
        "capture_failed_24h": capture_failed_24h,
        "learning_failed_24h": learning_failed_24h,
        "oldest_unpurged_raw_captured_at": oldest,
        "recent_failures": [
            {
                "session_id": row[0],
                "turn_id": row[1],
                "source": row[2],
                "capture_status": row[3],
                "learning_status": row[4],
                "error": row[5],
                "updated_at": row[6],
            }
            for row in failures
        ],
    }


def purge_expired_raw_payloads(
    conn: sqlite3.Connection,
    *,
    now_iso: str | None = None,
    retention_days: int = DEFAULT_RETENTION_DAYS,
) -> int:
    """Null expired raw payload text while preserving metadata and ledger rows."""
    now = datetime.fromisoformat(now_iso) if now_iso else datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)

    rows = conn.execute(
        """SELECT id, captured_at, retention_days FROM raw_turn_payloads
           WHERE purged_at IS NULL
             AND (user_message IS NOT NULL OR assistant_response IS NOT NULL)"""
    ).fetchall()
    expired_ids: list[int] = []
    for row in rows:
        row_retention = int(row[2] if row[2] is not None else retention_days)
        if row_retention < 0:
            continue
        captured = datetime.fromisoformat(row[1])
        if captured.tzinfo is None:
            captured = captured.replace(tzinfo=UTC)
        if captured + timedelta(days=row_retention) <= now:
            expired_ids.append(int(row[0]))

    if not expired_ids:
        return 0

    placeholders = ",".join("?" for _ in expired_ids)
    conn.execute(
        f"""UPDATE raw_turn_payloads
            SET user_message=NULL,
                assistant_response=NULL,
                purged_at=?,
                purge_reason='retention_expired'
            WHERE id IN ({placeholders})""",
        (now.isoformat(), *expired_ids),
    )
    return len(expired_ids)
