"""INGEST-060-P1 raw turn capture and ingestion ledger helpers."""

from __future__ import annotations

import hashlib
import os
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Any

VALID_SOURCES = {"conversation_turn", "session_end"}
VALID_LEARNING_STATUSES = {"not_started", "attempted", "skipped", "failed"}
VALID_PAYLOAD_COMPLETENESS = {"full_exchange", "user_only", "assistant_only", "empty", "unknown"}
DEFAULT_RETENTION_DAYS = 30
SKIP_REASON_EMPTY_EXTRACTED_CONCEPTS = "empty_extracted_concepts"
ERROR_PREFIX_SKIP_REASON = "skip_reason:"
ERROR_PREFIX_FALLBACK = "fallback:"


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


def build_skip_reason_error(reason: str, *, fallback_status: str | None = None) -> str:
    """Build a compact machine-readable ledger reason string."""
    parts = [f"{ERROR_PREFIX_SKIP_REASON}{reason}"]
    if fallback_status:
        parts.append(f"{ERROR_PREFIX_FALLBACK}{fallback_status}")
    return ";".join(parts)


def _skip_reason_error_like(reason: str) -> str:
    return f"{ERROR_PREFIX_SKIP_REASON}{reason}%"


def classify_payload_completeness(user_message: str | None, assistant_response: str | None) -> str:
    has_user = bool(user_message)
    has_assistant = bool(assistant_response)
    if has_user and has_assistant:
        return "full_exchange"
    if has_user:
        return "user_only"
    if has_assistant:
        return "assistant_only"
    return "empty"


def parse_ledger_error(error: str | None) -> dict[str, str | None]:
    """Parse machine-readable values stored in turn_ingestion_ledger.error."""
    parsed: dict[str, str | None] = {"skip_reason": None, "fallback_status": None}
    if not error:
        return parsed
    for part in str(error).split(";"):
        item = part.strip()
        if item.startswith(ERROR_PREFIX_SKIP_REASON):
            parsed["skip_reason"] = item[len(ERROR_PREFIX_SKIP_REASON):] or None
        elif item.startswith(ERROR_PREFIX_FALLBACK):
            parsed["fallback_status"] = item[len(ERROR_PREFIX_FALLBACK):] or None
    return parsed


def get_turn_ingestion_diagnostic(conn: sqlite3.Connection, raw_capture_ref: dict[str, Any] | None) -> dict[str, Any]:
    """Load canonical turn-learning status for diagnostics without raw text."""
    if not raw_capture_ref:
        return {
            "learning_status": "unknown",
            "concepts_extracted": 0,
            "skip_reason": None,
            "fallback_status": None,
            "sync_handled": False,
        }
    row = conn.execute(
        """SELECT learning_status, concepts_extracted, error
           FROM turn_ingestion_ledger
           WHERE session_id=? AND turn_id=? AND source=?
           ORDER BY id DESC LIMIT 1""",
        (
            raw_capture_ref.get("session_id"),
            raw_capture_ref.get("turn_id"),
            raw_capture_ref.get("source", "conversation_turn"),
        ),
    ).fetchone()
    if not row:
        return {
            "learning_status": "unknown",
            "concepts_extracted": 0,
            "skip_reason": None,
            "fallback_status": None,
            "sync_handled": False,
        }
    error_info = parse_ledger_error(row["error"] if hasattr(row, "keys") else row[2])
    learning_status = row["learning_status"] if hasattr(row, "keys") else row[0]
    concepts_extracted = row["concepts_extracted"] if hasattr(row, "keys") else row[1]
    return {
        "learning_status": learning_status,
        "concepts_extracted": int(concepts_extracted or 0),
        "skip_reason": error_info.get("skip_reason"),
        "fallback_status": error_info.get("fallback_status"),
        "sync_handled": learning_status == "attempted",
    }


def _get_turn_ingestion_diagnostic_default_db(raw_capture_ref: dict[str, Any] | None) -> dict[str, Any]:
    """Load turn-learning diagnostics from the default storage backend."""
    try:
        from app.storage import _db

        with _db(operation="turn_ingestion_diagnostic") as conn:
            return get_turn_ingestion_diagnostic(conn, raw_capture_ref)
    except Exception as exc:
        return {
            "learning_status": "unknown",
            "concepts_extracted": 0,
            "skip_reason": None,
            "fallback_status": None,
            "sync_handled": False,
            "diagnostic_error": f"{type(exc).__name__}: {exc}",
        }


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
    payload_completeness = classify_payload_completeness(user_text, assistant_text)
    content_hash = _hash_payload(source, user_text, assistant_text)

    cursor = conn.execute(
        """INSERT OR IGNORE INTO raw_turn_payloads
           (session_id, turn_id, source, user_message, assistant_response,
            message_len, response_len, payload_completeness, content_hash, captured_at, retention_days)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            session_id,
            turn_id,
            source,
            user_text,
            assistant_text,
            len(user_text),
            len(assistant_text),
            payload_completeness,
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
            "windows": {
                "1h": _empty_ingestion_window_summary(),
                "24h": _empty_ingestion_window_summary(),
            },
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
        "windows": {
            "1h": _ingestion_window_summary(conn, hours=1),
            "24h": _ingestion_window_summary(conn, hours=24),
        },
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


def _empty_ingestion_window_summary() -> dict[str, Any]:
    return {
        "hours": 0,
        "total_rows": 0,
        "attempted_rows": 0,
        "attempted_positive_rows": 0,
        "attempted_zero_rows": 0,
        "skipped_rows": 0,
        "not_started_rows": 0,
        "failed_rows": 0,
        "explicit_empty_skips": 0,
        "concepts_extracted": 0,
        "classification": {
            "response_len_zero": 0,
            "explicit_empty_input": 0,
            "duplicate_capture": 0,
            "failed": 0,
            "unknown_zero_or_skip": 0,
        },
        "lifecycle_jobs": {
            "total_rows": 0,
            "committed_rows": 0,
            "failed_rows": 0,
        },
        "threads": {
            "total_rows": 0,
            "zero_link_rows": 0,
        },
    }


def _ingestion_window_summary(conn: sqlite3.Connection, *, hours: int) -> dict[str, Any]:
    summary = _empty_ingestion_window_summary()
    summary["hours"] = hours
    since_expr = f"-{int(hours)} hours"
    empty_skip_error = build_skip_reason_error(SKIP_REASON_EMPTY_EXTRACTED_CONCEPTS)
    empty_skip_like = _skip_reason_error_like(SKIP_REASON_EMPTY_EXTRACTED_CONCEPTS)
    row = conn.execute(
        """SELECT
               COUNT(*) AS total_rows,
               SUM(CASE WHEN l.learning_status='attempted' THEN 1 ELSE 0 END) AS attempted_rows,
               SUM(CASE WHEN l.learning_status='attempted' AND l.concepts_extracted > 0
                        THEN 1 ELSE 0 END) AS attempted_positive_rows,
               SUM(CASE WHEN l.learning_status='attempted' AND l.concepts_extracted = 0
                        THEN 1 ELSE 0 END) AS attempted_zero_rows,
               SUM(CASE WHEN l.learning_status='skipped' THEN 1 ELSE 0 END) AS skipped_rows,
               SUM(CASE WHEN l.learning_status='not_started' THEN 1 ELSE 0 END) AS not_started_rows,
               SUM(CASE WHEN l.learning_status='failed' THEN 1 ELSE 0 END) AS failed_rows,
               SUM(CASE WHEN l.error = ? OR l.error LIKE ? THEN 1 ELSE 0 END) AS explicit_empty_skips,
               SUM(l.concepts_extracted) AS concepts_extracted,
               SUM(CASE WHEN l.capture_status='failed' OR l.learning_status='failed'
                        THEN 1 ELSE 0 END) AS failed_classification
           FROM turn_ingestion_ledger l
           LEFT JOIN raw_turn_payloads r ON r.id = l.raw_payload_id
           WHERE julianday(l.created_at) > julianday('now', ?)""",
        (empty_skip_error, empty_skip_like, since_expr),
    ).fetchone()
    if row:
        keys = row.keys() if hasattr(row, "keys") else []
        values = {key: row[key] for key in keys} if keys else {}
        if not values:
            names = [
                "total_rows",
                "attempted_rows",
                "attempted_positive_rows",
                "attempted_zero_rows",
                "skipped_rows",
                "not_started_rows",
                "failed_rows",
                "explicit_empty_skips",
                "concepts_extracted",
                "failed_classification",
            ]
            values = dict(zip(names, row))
        for key in (
            "total_rows",
            "attempted_rows",
            "attempted_positive_rows",
            "attempted_zero_rows",
            "skipped_rows",
            "not_started_rows",
            "failed_rows",
            "explicit_empty_skips",
            "concepts_extracted",
        ):
            summary[key] = int(values.get(key) or 0)
        summary["classification"]["failed"] = int(values.get("failed_classification") or 0)

    class_rows = conn.execute(
        """SELECT bucket, COUNT(*) AS rows
           FROM (
               SELECT CASE
                   WHEN l.capture_status='failed' OR l.learning_status='failed' THEN 'failed'
                   WHEN l.error = ? OR l.error LIKE ? THEN 'explicit_empty_input'
                   WHEN l.capture_status='duplicate' THEN 'duplicate_capture'
                   WHEN COALESCE(r.response_len, 0) = 0 THEN 'response_len_zero'
                   ELSE 'unknown_zero_or_skip'
               END AS bucket
               FROM turn_ingestion_ledger l
               LEFT JOIN raw_turn_payloads r ON r.id = l.raw_payload_id
               WHERE julianday(l.created_at) > julianday('now', ?)
                 AND (l.learning_status='skipped'
                      OR (l.learning_status='attempted' AND l.concepts_extracted = 0)
                      OR l.learning_status='failed'
                      OR l.capture_status='failed')
           )
           GROUP BY bucket""",
        (empty_skip_error, empty_skip_like, since_expr),
    ).fetchall()
    for row in class_rows:
        bucket = row["bucket"] if hasattr(row, "keys") else row[0]
        count = row["rows"] if hasattr(row, "keys") else row[1]
        if bucket in summary["classification"]:
            summary["classification"][bucket] = int(count or 0)

    if _table_exists(conn, "lifecycle_jobs"):
        lifecycle = conn.execute(
            """SELECT
                   COUNT(*) AS total_rows,
                   SUM(CASE WHEN status='committed' THEN 1 ELSE 0 END) AS committed_rows,
                   SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS failed_rows
               FROM lifecycle_jobs
               WHERE julianday(created_at) > julianday('now', ?)""",
            (since_expr,),
        ).fetchone()
        summary["lifecycle_jobs"] = {
            "total_rows": int((lifecycle["total_rows"] if hasattr(lifecycle, "keys") else lifecycle[0]) or 0),
            "committed_rows": int((lifecycle["committed_rows"] if hasattr(lifecycle, "keys") else lifecycle[1]) or 0),
            "failed_rows": int((lifecycle["failed_rows"] if hasattr(lifecycle, "keys") else lifecycle[2]) or 0),
        }

    if _table_exists(conn, "threads") and _table_exists(conn, "thread_concept_links"):
        threads = conn.execute(
            """WITH recent_threads AS (
                   SELECT id FROM threads
                   WHERE julianday(created_at) > julianday('now', ?)
               ),
               link_counts AS (
                   SELECT thread_id, COUNT(*) AS links
                   FROM thread_concept_links
                   GROUP BY thread_id
               )
               SELECT
                   COUNT(*) AS total_rows,
                   SUM(CASE WHEN COALESCE(link_counts.links, 0) = 0 THEN 1 ELSE 0 END) AS zero_link_rows
               FROM recent_threads
               LEFT JOIN link_counts ON link_counts.thread_id = recent_threads.id""",
            (since_expr,),
        ).fetchone()
        summary["threads"] = {
            "total_rows": int((threads["total_rows"] if hasattr(threads, "keys") else threads[0]) or 0),
            "zero_link_rows": int((threads["zero_link_rows"] if hasattr(threads, "keys") else threads[1]) or 0),
        }
    return summary


def assert_raw_capture_retention_healthy(conn: sqlite3.Connection) -> dict[str, Any]:
    """Return retention health without exposing raw message or response text."""
    if not _table_exists(conn, "raw_turn_payloads"):
        return {
            "healthy": True,
            "raw_payload_rows": 0,
            "expired_unpurged_raw_rows": 0,
            "purged_rows": 0,
            "oldest_unpurged_raw_captured_at": None,
        }

    raw_payload_rows = conn.execute("SELECT COUNT(*) FROM raw_turn_payloads").fetchone()[0]
    purged_rows = conn.execute(
        "SELECT COUNT(*) FROM raw_turn_payloads WHERE purged_at IS NOT NULL"
    ).fetchone()[0]
    expired_unpurged_raw_rows = conn.execute(
        """SELECT COUNT(*) FROM raw_turn_payloads
           WHERE purged_at IS NULL
             AND (user_message IS NOT NULL OR assistant_response IS NOT NULL)
             AND datetime(captured_at, '+' || retention_days || ' days') < datetime('now')"""
    ).fetchone()[0]
    oldest = conn.execute(
        """SELECT MIN(captured_at) FROM raw_turn_payloads
           WHERE purged_at IS NULL
             AND (user_message IS NOT NULL OR assistant_response IS NOT NULL)"""
    ).fetchone()[0]
    return {
        "healthy": expired_unpurged_raw_rows == 0,
        "raw_payload_rows": raw_payload_rows,
        "expired_unpurged_raw_rows": expired_unpurged_raw_rows,
        "purged_rows": purged_rows,
        "oldest_unpurged_raw_captured_at": oldest,
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
