"""Durable concept-membership obligations for the live retrieval index.

MAINT-097 keeps lifecycle SQL writers independent from the in-process retrieval
singleton. SQLite triggers enqueue exact concept IDs; this module owns bounded
claim, retry, terminal, snapshot, and health operations for the server drain.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

MAX_ATTEMPTS = 5
STALE_RUNNING_SECONDS = 120
ERROR_LIMIT = 500


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _db(*, operation: str):
    from app.storage import _db as storage_db

    return storage_db(operation=operation)


def _db_immediate(*, operation: str):
    from app.storage import _db_immediate as storage_db_immediate

    return storage_db_immediate(operation=operation)


def _read_db(*, operation: str):
    from app.storage.connection import read_snapshot_db

    return read_snapshot_db(operation)


def _row_to_dict(row: Any) -> dict[str, Any]:
    if hasattr(row, "keys"):
        return {key: row[key] for key in row.keys()}
    return dict(row)


def _bounded_error(error: BaseException | str | None) -> str | None:
    if error is None:
        return None
    return str(error)[:ERROR_LIMIT]


def _placeholders(values: Iterable[Any]) -> str:
    return ",".join("?" for _ in values)


def claim_lifecycle_index_events(
    *,
    limit: int = 25,
    now: datetime | None = None,
    stale_running_seconds: int = STALE_RUNNING_SECONDS,
    max_attempts: int = MAX_ATTEMPTS,
) -> list[dict[str, Any]]:
    """Atomically reset stale rows and claim a bounded ready batch."""
    if limit <= 0:
        return []
    current = now or _utc_now()
    now_iso = current.isoformat()
    stale_before = (current - timedelta(seconds=max(1, stale_running_seconds))).isoformat()

    with _db_immediate(operation="lifecycle_index_outbox_claim") as conn:
        conn.execute(
            """UPDATE lifecycle_index_outbox
               SET status='failed', updated_at=?, next_attempt_at=NULL,
                   last_error='stale running row exhausted retry budget'
               WHERE status='running' AND updated_at < ? AND attempts >= ?""",
            (now_iso, stale_before, max_attempts),
        )
        conn.execute(
            """UPDATE lifecycle_index_outbox
               SET status='queued', updated_at=?, next_attempt_at=NULL,
                   last_error='stale running row reset'
               WHERE status='running' AND updated_at < ? AND attempts < ?""",
            (now_iso, stale_before, max_attempts),
        )
        rows = conn.execute(
            """SELECT id, concept_id, event_type, status, attempts,
                      next_attempt_at, created_at, updated_at, last_error, source
               FROM lifecycle_index_outbox
               WHERE status='queued'
                 AND attempts < ?
                 AND (next_attempt_at IS NULL OR next_attempt_at <= ?)
               ORDER BY created_at, id
               LIMIT ?""",
            (max_attempts, now_iso, limit),
        ).fetchall()
        if not rows:
            return []
        ids = [int(row["id"] if hasattr(row, "keys") else row[0]) for row in rows]
        updated = conn.execute(
            f"""UPDATE lifecycle_index_outbox
                SET status='running', attempts=attempts+1, updated_at=?, last_error=NULL
                WHERE id IN ({_placeholders(ids)}) AND status='queued'""",
            (now_iso, *ids),
        )
        if int(updated.rowcount or 0) != len(ids):
            raise RuntimeError("lifecycle index outbox claim lost ownership")
        claimed = []
        for row in rows:
            data = _row_to_dict(row)
            data["status"] = "running"
            data["attempts"] = int(data.get("attempts") or 0) + 1
            data["updated_at"] = now_iso
            claimed.append(data)
        return claimed


def load_membership_snapshots(concept_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Load authoritative active-membership state for exact concept IDs."""
    ids = list(dict.fromkeys(str(value) for value in concept_ids if value))
    snapshots = {
        concept_id: {
            "exists": False,
            "active": False,
            "status": None,
            "is_current": None,
            "updated_at": None,
        }
        for concept_id in ids
    }
    if not ids:
        return snapshots
    with _read_db(operation="lifecycle_index_outbox_membership") as conn:
        rows = conn.execute(
            f"""SELECT id, status, is_current, updated_at
                FROM concepts WHERE id IN ({_placeholders(ids)})""",
            tuple(ids),
        ).fetchall()
    for row in rows:
        data = _row_to_dict(row)
        concept_id = str(data["id"])
        is_current = int(data.get("is_current") or 0)
        status = data.get("status")
        snapshots[concept_id] = {
            "exists": True,
            "active": status == "active" and is_current == 1,
            "status": status,
            "is_current": is_current,
            "updated_at": data.get("updated_at"),
        }
    return snapshots


def terminalize_lifecycle_index_events(
    row_ids: Iterable[int],
    *,
    status: str = "done",
    error: BaseException | str | None = None,
    now: datetime | None = None,
) -> int:
    """Mark claimed rows done or skipped."""
    if status not in {"done", "skipped"}:
        raise ValueError(f"invalid terminal status: {status}")
    ids = list(dict.fromkeys(int(value) for value in row_ids))
    if not ids:
        return 0
    with _db(operation="lifecycle_index_outbox_terminalize") as conn:
        cur = conn.execute(
            f"""UPDATE lifecycle_index_outbox
                SET status=?, updated_at=?, next_attempt_at=NULL, last_error=?
                WHERE id IN ({_placeholders(ids)}) AND status='running'""",
            (status, (now or _utc_now()).isoformat(), _bounded_error(error), *ids),
        )
        return int(cur.rowcount or 0)


def retry_lifecycle_index_events(
    rows: Iterable[dict[str, Any]],
    *,
    error: BaseException | str,
    now: datetime | None = None,
    max_attempts: int = MAX_ATTEMPTS,
) -> dict[str, int]:
    """Requeue failed claims with exponential backoff; fail attempt five."""
    claimed = list(rows)
    if not claimed:
        return {"queued": 0, "failed": 0}
    current = now or _utc_now()
    error_text = _bounded_error(error)
    counts = {"queued": 0, "failed": 0}
    with _db(operation="lifecycle_index_outbox_retry") as conn:
        for row in claimed:
            row_id = int(row["id"])
            attempts = int(row.get("attempts") or 0)
            if attempts >= max_attempts:
                cur = conn.execute(
                    """UPDATE lifecycle_index_outbox
                       SET status='failed', updated_at=?, next_attempt_at=NULL, last_error=?
                       WHERE id=? AND status='running'""",
                    (current.isoformat(), error_text, row_id),
                )
                counts["failed"] += int(cur.rowcount or 0)
                continue
            delay_minutes = 2 ** max(0, attempts - 1)
            next_attempt = (current + timedelta(minutes=delay_minutes)).isoformat()
            cur = conn.execute(
                """UPDATE lifecycle_index_outbox
                   SET status='queued', next_attempt_at=?, updated_at=?, last_error=?
                   WHERE id=? AND status='running'""",
                (next_attempt, current.isoformat(), error_text, row_id),
            )
            counts["queued"] += int(cur.rowcount or 0)
    return counts


def _age_seconds(value: str | None, now: datetime) -> float | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return max(0.0, (now - parsed.astimezone(UTC)).total_seconds())
    except (TypeError, ValueError):
        return None


def summarize_lifecycle_index_outbox(*, now: datetime | None = None) -> dict[str, Any]:
    """Return bounded durable queue evidence; never expose concept content."""
    current = now or _utc_now()
    with _read_db(operation="lifecycle_index_outbox_status") as conn:
        row = conn.execute(
            """SELECT
                   SUM(CASE WHEN status='queued' THEN 1 ELSE 0 END) AS queued_count,
                   SUM(CASE WHEN status='running' THEN 1 ELSE 0 END) AS running_count,
                   SUM(CASE WHEN status='done' THEN 1 ELSE 0 END) AS done_count,
                   SUM(CASE WHEN status='skipped' THEN 1 ELSE 0 END) AS skipped_count,
                   SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS failed_count,
                   MIN(CASE WHEN status='queued' THEN created_at END) AS oldest_queued_at
               FROM lifecycle_index_outbox"""
        ).fetchone()
        latest_failure = conn.execute(
            """SELECT updated_at, last_error
               FROM lifecycle_index_outbox
               WHERE status='failed'
               ORDER BY updated_at DESC, id DESC
               LIMIT 1"""
        ).fetchone()
    data = _row_to_dict(row)
    failure = _row_to_dict(latest_failure) if latest_failure else {}
    return {
        "queued_count": int(data.get("queued_count") or 0),
        "running_count": int(data.get("running_count") or 0),
        "done_count": int(data.get("done_count") or 0),
        "skipped_count": int(data.get("skipped_count") or 0),
        "failed_count": int(data.get("failed_count") or 0),
        "oldest_queued_age_seconds": _age_seconds(data.get("oldest_queued_at"), current),
        "latest_failure_at": failure.get("updated_at"),
        "latest_error": _bounded_error(failure.get("last_error")),
    }
