"""Durable lifecycle job queue helpers for post-response learning work."""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

VALID_SOURCES = {"conversation_turn", "session_learn", "session_end", "maintenance"}
VALID_STAGES = {"learn"}
VALID_STATUSES = {"queued", "running", "committed", "retry", "failed", "skipped"}


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _db(*, timeout_s: float = 30.0, operation: str = "lifecycle_jobs"):
    from app.storage import _db as storage_db

    return storage_db(timeout_s=timeout_s, operation=operation)


def _db_immediate(*, timeout_s: float = 30.0, operation: str = "lifecycle_jobs_immediate"):
    from app.storage import _db_immediate as storage_db_immediate

    return storage_db_immediate(timeout_s=timeout_s, operation=operation)


def _read_db(*, operation: str = "lifecycle_jobs_read"):
    from app.storage.connection import read_snapshot_db

    return read_snapshot_db(operation)


def _row_to_dict(row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    if hasattr(row, "keys"):
        return {key: row[key] for key in row.keys()}
    return dict(row)


def _load_job_by_identity(conn, profile: str, source: str, idempotency_key: str) -> dict[str, Any] | None:
    row = conn.execute(
        """SELECT *
           FROM lifecycle_jobs
           WHERE profile=? AND source=? AND idempotency_key=?""",
        (profile, source, idempotency_key),
    ).fetchone()
    return _row_to_dict(row) if row else None


def load_lifecycle_job_by_identity(
    *,
    profile: str,
    source: str,
    idempotency_key: str,
) -> dict[str, Any] | None:
    """Load a lifecycle job by its stable source/idempotency identity."""
    with _db(operation="lifecycle_load_identity") as conn:
        job = _load_job_by_identity(conn, profile, source, idempotency_key)
    if not job:
        return None
    try:
        job["payload"] = json.loads(job.get("payload_json") or "{}")
    except Exception:
        job["payload"] = {}
    try:
        job["result"] = json.loads(job.get("result_json") or "{}")
    except Exception:
        job["result"] = {}
    return job


def enqueue_lifecycle_job(
    *,
    profile: str,
    source: str,
    idempotency_key: str,
    stage: str,
    payload: dict[str, Any],
    priority: int = 50,
    now: str | None = None,
) -> dict[str, Any]:
    """Insert a queued lifecycle job, returning the existing row on duplicate."""
    if source not in VALID_SOURCES:
        raise ValueError(f"invalid lifecycle job source: {source}")
    if stage not in VALID_STAGES:
        raise ValueError(f"invalid lifecycle job stage: {stage}")
    if not profile:
        raise ValueError("profile is required")
    if not idempotency_key:
        raise ValueError("idempotency_key is required")

    ts = now or _utc_now_iso()
    job_id = f"lj_{uuid.uuid4().hex[:16]}"
    payload_json = json.dumps(payload, sort_keys=True)
    with _db(operation="lifecycle_enqueue") as conn:
        conn.execute(
            """INSERT OR IGNORE INTO lifecycle_jobs
               (job_id, profile, source, idempotency_key, priority, stage, status,
                payload_json, result_json, attempts, last_error, lease_owner,
                lease_expires_at, next_retry_at, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, 'queued', ?, NULL, 0, NULL, NULL, NULL, NULL, ?, ?)""",
            (
                job_id,
                profile,
                source,
                idempotency_key,
                int(priority),
                stage,
                payload_json,
                ts,
                ts,
            ),
        )
        job = _load_job_by_identity(conn, profile, source, idempotency_key)
    return job or {
        "job_id": job_id,
        "profile": profile,
        "source": source,
        "idempotency_key": idempotency_key,
        "stage": stage,
        "status": "queued",
    }


def claim_lifecycle_jobs(
    *,
    profile: str,
    lease_owner: str,
    lease_expires_at: str,
    limit: int,
    now: str | None = None,
    max_attempts: int = 3,
    source: str | None = None,
) -> list[dict[str, Any]]:
    """Claim queued/retry/stale-running jobs for processing."""
    if limit <= 0:
        return []
    if source is not None and source not in VALID_SOURCES:
        raise ValueError(f"invalid lifecycle job source: {source}")
    ts = now or _utc_now_iso()
    claimed: list[dict[str, Any]] = []
    with _db_immediate(operation="lifecycle_claim") as conn:
        rows = conn.execute(
            """SELECT *
               FROM lifecycle_jobs
               WHERE profile=?
                 AND (? IS NULL OR source=?)
                 AND (
                   status='queued'
                   OR (status='retry' AND (next_retry_at IS NULL OR next_retry_at <= ?))
                   OR (status='running' AND lease_expires_at IS NOT NULL AND lease_expires_at <= ?)
                 )
                 AND attempts < ?
               ORDER BY priority ASC, updated_at ASC
               LIMIT ?""",
            (profile, source, source, ts, ts, max_attempts, limit),
        ).fetchall()
        for row in rows:
            data = _row_to_dict(row)
            cur = conn.execute(
                """UPDATE lifecycle_jobs
                   SET status='running',
                       attempts=attempts+1,
                       lease_owner=?,
                       lease_expires_at=?,
                       last_error=NULL,
                       updated_at=?
                   WHERE profile=? AND job_id=? AND status IN ('queued','retry','running')
                     AND (? IS NULL OR source=?)
                     AND attempts < ?""",
                (
                    lease_owner,
                    lease_expires_at,
                    ts,
                    profile,
                    data["job_id"],
                    source,
                    source,
                    max_attempts,
                ),
            )
            if cur.rowcount:
                data["status"] = "running"
                data["attempts"] = int(data.get("attempts") or 0) + 1
                data["lease_owner"] = lease_owner
                data["lease_expires_at"] = lease_expires_at
                try:
                    data["payload"] = json.loads(data.get("payload_json") or "{}")
                except Exception:
                    data["payload"] = {}
                claimed.append(data)
    return claimed


def commit_lifecycle_job(
    *,
    profile: str,
    job_id: str,
    result: dict[str, Any] | None = None,
    now: str | None = None,
) -> None:
    ts = now or _utc_now_iso()
    result_json = json.dumps(result or {}, sort_keys=True)
    retain_payload = os.environ.get("PITH_LIFECYCLE_JOBS_RETAIN_PAYLOADS", "false").lower() in {
        "1",
        "true",
        "yes",
    }
    payload_expr = "payload_json" if retain_payload else "'{}'"
    with _db(operation="lifecycle_commit") as conn:
        conn.execute(
            f"""UPDATE lifecycle_jobs
                SET status='committed',
                    result_json=?,
                    payload_json={payload_expr},
                    last_error=NULL,
                    lease_owner=NULL,
                    lease_expires_at=NULL,
                    next_retry_at=NULL,
                    updated_at=?
                WHERE profile=? AND job_id=?""",
            (result_json, ts, profile, job_id),
        )


def retry_lifecycle_job(
    *,
    profile: str,
    job_id: str,
    error: str,
    next_retry_at: str,
    now: str | None = None,
) -> None:
    ts = now or _utc_now_iso()
    with _db(operation="lifecycle_retry") as conn:
        conn.execute(
            """UPDATE lifecycle_jobs
               SET status='retry',
                   last_error=?,
                   lease_owner=NULL,
                   lease_expires_at=NULL,
                   next_retry_at=?,
                   updated_at=?
               WHERE profile=? AND job_id=?""",
            (error[:1000], next_retry_at, ts, profile, job_id),
        )


def fail_lifecycle_job(
    *,
    profile: str,
    job_id: str,
    error: str,
    now: str | None = None,
) -> None:
    ts = now or _utc_now_iso()
    with _db(operation="lifecycle_fail") as conn:
        conn.execute(
            """UPDATE lifecycle_jobs
               SET status='failed',
                   last_error=?,
                   lease_owner=NULL,
                   lease_expires_at=NULL,
                   updated_at=?
               WHERE profile=? AND job_id=?""",
            (error[:1000], ts, profile, job_id),
        )


def summarize_lifecycle_jobs(*, profile: str, stale_before_iso: str) -> dict[str, Any]:
    with _read_db(operation="lifecycle_summary") as conn:
        rows = conn.execute(
            """SELECT status, updated_at, lease_expires_at
               FROM lifecycle_jobs
               WHERE profile=?""",
            (profile,),
        ).fetchall()
    counts = {status: 0 for status in VALID_STATUSES}
    oldest_queued: str | None = None
    oldest_running: str | None = None
    stale_running = 0
    for row in rows:
        data = _row_to_dict(row)
        status = data.get("status")
        if status in counts:
            counts[status] += 1
        updated_at = data.get("updated_at")
        if status in {"queued", "retry"} and updated_at and (oldest_queued is None or updated_at < oldest_queued):
            oldest_queued = updated_at
        if status == "running":
            if updated_at and (oldest_running is None or updated_at < oldest_running):
                oldest_running = updated_at
            lease_expires_at = data.get("lease_expires_at")
            if (lease_expires_at and lease_expires_at < _utc_now_iso()) or (
                updated_at and updated_at < stale_before_iso
            ):
                stale_running += 1
    return {
        "queued_count": counts["queued"],
        "running_count": counts["running"],
        "retry_count": counts["retry"],
        "failed_count": counts["failed"],
        "committed_count": counts["committed"],
        "skipped_count": counts["skipped"],
        "stale_running_count": stale_running,
        "oldest_queued_updated_at": oldest_queued,
        "oldest_running_updated_at": oldest_running,
    }


def summarize_lifecycle_jobs_by_source(
    *,
    profile: str,
    source: str,
    stale_before_iso: str,
) -> dict[str, Any]:
    if source not in VALID_SOURCES:
        raise ValueError(f"invalid lifecycle job source: {source}")
    with _read_db(operation="lifecycle_summary_source") as conn:
        rows = conn.execute(
            """SELECT status, updated_at, lease_expires_at
               FROM lifecycle_jobs
               WHERE profile=? AND source=?""",
            (profile, source),
        ).fetchall()
    counts = {status: 0 for status in VALID_STATUSES}
    oldest_queued: str | None = None
    oldest_running: str | None = None
    stale_running = 0
    for row in rows:
        data = _row_to_dict(row)
        status = data.get("status")
        if status in counts:
            counts[status] += 1
        updated_at = data.get("updated_at")
        if status in {"queued", "retry"} and updated_at and (oldest_queued is None or updated_at < oldest_queued):
            oldest_queued = updated_at
        if status == "running":
            if updated_at and (oldest_running is None or updated_at < oldest_running):
                oldest_running = updated_at
            lease_expires_at = data.get("lease_expires_at")
            if (lease_expires_at and lease_expires_at < _utc_now_iso()) or (
                updated_at and updated_at < stale_before_iso
            ):
                stale_running += 1
    return {
        "queued_count": counts["queued"],
        "running_count": counts["running"],
        "retry_count": counts["retry"],
        "failed_count": counts["failed"],
        "committed_count": counts["committed"],
        "skipped_count": counts["skipped"],
        "stale_running_count": stale_running,
        "oldest_queued_updated_at": oldest_queued,
        "oldest_running_updated_at": oldest_running,
    }


def cleanup_committed_lifecycle_jobs(*, profile: str, retention_days: int, now: str | None = None) -> int:
    ts = now or _utc_now_iso()
    cutoff = (datetime.fromisoformat(ts) - timedelta(days=retention_days)).isoformat()
    with _db(operation="lifecycle_cleanup") as conn:
        cur = conn.execute(
            """DELETE FROM lifecycle_jobs
               WHERE profile=? AND status='committed' AND updated_at < ?""",
            (profile, cutoff),
        )
        return int(cur.rowcount or 0)
