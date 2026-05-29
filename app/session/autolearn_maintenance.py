"""Deferred maintenance queue for autolearn secondary work."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import timedelta
from typing import Any

from app.cognitive.autolearn_maintenance_queue import (
    TASK_GOVERNANCE_RECOMPUTE,
    TASK_SIMILARITY_SUPERSESSION,
    TASK_SUBJECT_KEY_SUPERSESSION,
    _error_text,
    _supersession_disabled,
    enqueue_autolearn_maintenance,
    enqueue_subject_key_supersession,
    ensure_autolearn_maintenance_tables,
    get_autolearn_maintenance_status,
    register_autolearn_maintenance_drain_kicker,
)
from app.core.datetime_utils import _utc_now, _utc_now_iso

logger = logging.getLogger(__name__)

__all__ = [
    "TASK_GOVERNANCE_RECOMPUTE",
    "TASK_SIMILARITY_SUPERSESSION",
    "TASK_SUBJECT_KEY_SUPERSESSION",
    "enqueue_autolearn_maintenance",
    "enqueue_subject_key_supersession",
    "ensure_autolearn_maintenance_tables",
    "get_autolearn_maintenance_status",
    "kick_autolearn_maintenance_drain",
    "run_autolearn_maintenance_queue",
]

_DRAIN_TASK: asyncio.Task | None = None
_SYNC_DRAIN_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pith-autolearn-drain")
_SYNC_DRAIN_FUTURE: Future | None = None
_SYNC_DRAIN_LOCK = threading.Lock()


def _benchmark_mode_active() -> bool:
    return os.environ.get("PITH_BENCHMARK_MODE", "").lower() in {"1", "true", "yes", "on"}


def _set_busy_timeout(conn) -> None:
    from app.core.config import get_autolearn_maintenance_busy_timeout_ms

    conn.execute(f"PRAGMA busy_timeout={get_autolearn_maintenance_busy_timeout_ms()}")


async def _immediate_drain_bg() -> None:
    from app.core.config import get_autolearn_maintenance_immediate_batch_size
    from app.storage import owned_connection

    try:
        with owned_connection() as conn:
            await run_autolearn_maintenance_queue(conn, get_autolearn_maintenance_immediate_batch_size())
    except Exception as exc:
        logger.warning("autolearn_maintenance_immediate_drain failed (non-fatal): %s", exc)


def _immediate_drain_sync() -> int:
    from app.core.config import get_autolearn_maintenance_immediate_batch_size
    from app.storage import owned_connection

    try:
        with owned_connection() as conn:
            return asyncio.run(
                run_autolearn_maintenance_queue(conn, get_autolearn_maintenance_immediate_batch_size())
            )
    except Exception as exc:
        logger.warning("autolearn_maintenance_sync_drain failed (non-fatal): %s", exc)
        return 0


def _clear_drain_task(task: asyncio.Task) -> None:
    global _DRAIN_TASK
    if _DRAIN_TASK is task:
        _DRAIN_TASK = None


def _clear_sync_drain_future(future: Future) -> None:
    global _SYNC_DRAIN_FUTURE
    try:
        future.result()
    except Exception as exc:
        logger.warning("autolearn_maintenance_sync_drain future failed (non-fatal): %s", exc)
    with _SYNC_DRAIN_LOCK:
        if _SYNC_DRAIN_FUTURE is future:
            _SYNC_DRAIN_FUTURE = None


def kick_autolearn_maintenance_drain() -> bool:
    """Schedule at most one tiny maintenance drain from async or sync call paths."""
    global _DRAIN_TASK, _SYNC_DRAIN_FUTURE
    from app.core.config import (
        get_autolearn_maintenance_enabled,
        get_autolearn_maintenance_immediate_drain_enabled,
        get_autolearn_maintenance_sync_drain_enabled,
    )

    if _benchmark_mode_active():
        # Keep benchmark ingestion single-writer clean; queued maintenance can run later via catchup.
        return False
    if not get_autolearn_maintenance_enabled() or not get_autolearn_maintenance_immediate_drain_enabled():
        return False
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        if not get_autolearn_maintenance_sync_drain_enabled():
            return False
        with _SYNC_DRAIN_LOCK:
            if _SYNC_DRAIN_FUTURE and not _SYNC_DRAIN_FUTURE.done():
                return False
            try:
                _SYNC_DRAIN_FUTURE = _SYNC_DRAIN_EXECUTOR.submit(_immediate_drain_sync)
                _SYNC_DRAIN_FUTURE.add_done_callback(_clear_sync_drain_future)
                return True
            except Exception as exc:
                logger.warning("autolearn_maintenance_sync_drain schedule failed (non-fatal): %s", exc)
                _SYNC_DRAIN_FUTURE = None
                return False
    if _DRAIN_TASK and not _DRAIN_TASK.done():
        return False
    try:
        _DRAIN_TASK = loop.create_task(_immediate_drain_bg(), name="autolearn_maintenance_immediate_drain")
        _DRAIN_TASK.add_done_callback(_clear_drain_task)
        return True
    except Exception as exc:
        logger.warning("autolearn_maintenance_immediate_drain schedule failed (non-fatal): %s", exc)
        _DRAIN_TASK = None
        return False


register_autolearn_maintenance_drain_kicker(kick_autolearn_maintenance_drain)


def _reset_stale_running(conn) -> int:
    from app.core.config import get_autolearn_maintenance_running_stale_seconds

    now = _utc_now()
    stale_cutoff = (now - timedelta(seconds=get_autolearn_maintenance_running_stale_seconds())).isoformat()
    cur = conn.execute(
        """UPDATE autolearn_maintenance_queue
           SET status='queued', updated_at=?, last_error='stale running row reset'
           WHERE status='running' AND updated_at < ?""",
        (now.isoformat(), stale_cutoff),
    )
    conn.commit()
    return int(cur.rowcount or 0)


def _claim_rows(conn, batch_size: int, task_type: str | None = None) -> list[dict[str, Any]]:
    if batch_size <= 0:
        return []
    if task_type is not None and task_type not in {
        TASK_SUBJECT_KEY_SUPERSESSION,
        TASK_GOVERNANCE_RECOMPUTE,
        TASK_SIMILARITY_SUPERSESSION,
    }:
        raise ValueError(f"Unsupported autolearn maintenance task_type: {task_type}")
    now = _utc_now_iso()
    try:
        conn.execute("BEGIN IMMEDIATE")
        rows = (
            _select_ready_rows(conn, now, batch_size, task_type, set())
            if task_type is not None
            else _select_fair_claim_rows(conn, now, batch_size)
        )
        if not rows:
            conn.commit()
            return []
        ids = [int(row["id"] if hasattr(row, "keys") else row[0]) for row in rows]
        placeholders = ",".join("?" for _ in ids)
        updated = conn.execute(
            f"""UPDATE autolearn_maintenance_queue
                SET status='running', attempts=attempts+1, updated_at=?
                WHERE id IN ({placeholders}) AND status='queued'""",
            (now, *ids),
        )
        if int(updated.rowcount or 0) != len(ids):
            conn.rollback()
            return []
        conn.commit()
        return [_row_to_dict(row) for row in rows]
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise


def _select_fair_claim_rows(conn, now: str, batch_size: int) -> list:
    from app.core.config import get_autolearn_maintenance_max_task_share

    ready_counts = dict.fromkeys(
        [TASK_SUBJECT_KEY_SUPERSESSION, TASK_GOVERNANCE_RECOMPUTE, TASK_SIMILARITY_SUPERSESSION],
        0,
    )
    for row in conn.execute(
        """SELECT task_type, COUNT(*) AS count
           FROM autolearn_maintenance_queue
           WHERE status='queued'
             AND (next_attempt_at IS NULL OR next_attempt_at <= ?)
           GROUP BY task_type""",
        (now,),
    ).fetchall():
        task_type = row["task_type"] if hasattr(row, "keys") else row[0]
        count = row["count"] if hasattr(row, "keys") else row[1]
        ready_counts[task_type] = int(count or 0)

    selected: list = []
    selected_ids: set[int] = set()
    selected_counts: dict[str, int] = {}

    def take(task_type: str | None, limit: int) -> None:
        if limit <= 0 or len(selected) >= batch_size:
            return
        rows = _select_ready_rows(conn, now, min(limit, batch_size - len(selected)), task_type, selected_ids)
        selected.extend(rows)
        selected_ids.update(int(row["id"] if hasattr(row, "keys") else row[0]) for row in rows)
        for row in rows:
            selected_task_type = row["task_type"] if hasattr(row, "keys") else row[3]
            selected_counts[selected_task_type] = selected_counts.get(selected_task_type, 0) + 1

    if ready_counts.get(TASK_SUBJECT_KEY_SUPERSESSION, 0) > 0:
        take(TASK_SUBJECT_KEY_SUPERSESSION, 1)

    task_types = [
        task_type
        for task_type in [
            TASK_SUBJECT_KEY_SUPERSESSION,
            TASK_GOVERNANCE_RECOMPUTE,
            TASK_SIMILARITY_SUPERSESSION,
        ]
        if ready_counts.get(task_type, 0) > 0
    ]
    task_types.extend(sorted(task_type for task_type, count in ready_counts.items() if count > 0 and task_type not in task_types))
    if not task_types:
        return selected

    rows_by_task = {
        task_type: _select_ready_rows(conn, now, batch_size, task_type, selected_ids) for task_type in task_types
    }
    mixed_batch = sum(1 for rows in rows_by_task.values() if rows) > 1
    task_cap = batch_size if not mixed_batch else max(1, math.ceil(batch_size * get_autolearn_maintenance_max_task_share()))

    def row_created_at(row) -> str:
        return row["created_at"] if hasattr(row, "keys") else row[5]

    while len(selected) < batch_size:
        candidates = [
            (task_type, rows[0])
            for task_type, rows in rows_by_task.items()
            if rows and selected_counts.get(task_type, 0) < task_cap
        ]
        if not candidates:
            candidates = [(task_type, rows[0]) for task_type, rows in rows_by_task.items() if rows]
        if not candidates:
            break
        task_type, _ = min(candidates, key=lambda item: row_created_at(item[1]))
        row = rows_by_task[task_type].pop(0)
        selected.append(row)
        selected_ids.add(int(row["id"] if hasattr(row, "keys") else row[0]))
        selected_counts[task_type] = selected_counts.get(task_type, 0) + 1
    return selected


def _select_ready_rows(conn, now: str, limit: int, task_type: str | None, excluded_ids: set[int]) -> list:
    params: list[Any] = [now]
    filters = [
        "status='queued'",
        "(next_attempt_at IS NULL OR next_attempt_at <= ?)",
    ]
    if task_type is not None:
        filters.append("task_type=?")
        params.append(task_type)
    if excluded_ids:
        placeholders = ",".join("?" for _ in excluded_ids)
        filters.append(f"id NOT IN ({placeholders})")
        params.extend(sorted(excluded_ids))
    params.append(limit)
    return conn.execute(
        f"""SELECT id, concept_id, concept_version, task_type, attempts, created_at
            FROM autolearn_maintenance_queue
            WHERE {' AND '.join(filters)}
            ORDER BY created_at
            LIMIT ?""",
        tuple(params),
    ).fetchall()


def _row_to_dict(row) -> dict[str, Any]:
    if hasattr(row, "keys"):
        return {key: row[key] for key in row.keys()}
    return {
        "id": row[0],
        "concept_id": row[1],
        "concept_version": row[2],
        "task_type": row[3],
        "attempts": row[4],
        "created_at": row[5] if len(row) > 5 else None,
    }


def _load_processable_concept(conn, concept_id: str):
    return conn.execute(
        "SELECT id, subject_key, status, is_current FROM concepts WHERE id = ?",
        (concept_id,),
    ).fetchone()


def _concept_is_processable(row) -> bool:
    if not row:
        return False
    status = row["status"] if hasattr(row, "keys") else row[2]
    is_current = row["is_current"] if hasattr(row, "keys") else row[3]
    return status != "deleted" and int(is_current or 0) == 1


def _mark_done(conn, row_id: int, status: str, error: BaseException | str | None = None) -> None:
    conn.execute(
        """UPDATE autolearn_maintenance_queue
           SET status=?, updated_at=?, last_error=?
           WHERE id=?""",
        (status, _utc_now_iso(), _error_text(error), row_id),
    )
    conn.commit()


def _mark_retry_or_failed(conn, row: dict[str, Any], error: BaseException) -> None:
    from app.core.config import get_autolearn_maintenance_max_attempts

    current_attempts = int(row.get("attempts") or 0) + 1
    if current_attempts >= get_autolearn_maintenance_max_attempts():
        _mark_done(conn, int(row["id"]), "failed", error)
        return
    delay_minutes = 2 ** max(0, current_attempts - 1)
    next_attempt_at = (_utc_now() + timedelta(minutes=delay_minutes)).isoformat()
    conn.execute(
        """UPDATE autolearn_maintenance_queue
           SET status='queued', next_attempt_at=?, updated_at=?, last_error=?
           WHERE id=?""",
        (next_attempt_at, _utc_now_iso(), _error_text(error), int(row["id"])),
    )
    conn.commit()


def _process_governance_rows(conn, rows: list[dict[str, Any]]) -> int:
    processable = []
    skipped = 0
    for row in rows:
        concept_row = _load_processable_concept(conn, row["concept_id"])
        if not _concept_is_processable(concept_row):
            _mark_done(conn, int(row["id"]), "skipped", "missing or non-current concept")
            skipped += 1
            continue
        processable.append(row)
    if processable:
        from app.authority import batch_compute_authority
        from app.governance.currency import batch_compute_currency

        concept_ids = [row["concept_id"] for row in processable]
        batch_compute_authority(conn, concept_ids=concept_ids)
        batch_compute_currency(conn, concept_ids=concept_ids)
        for row in processable:
            _mark_done(conn, int(row["id"]), "done")
    return len(processable) + skipped


def _process_subject_key_row(conn, row: dict[str, Any]) -> None:
    concept_row = _load_processable_concept(conn, row["concept_id"])
    if not _concept_is_processable(concept_row):
        _mark_done(conn, int(row["id"]), "skipped", "missing or non-current concept")
        return
    subject_key = concept_row["subject_key"] if hasattr(concept_row, "keys") else concept_row[1]
    if not subject_key:
        _mark_done(conn, int(row["id"]), "skipped", "missing subject_key")
        return

    from app.storage import apply_lifecycle_transition_conn

    candidates = conn.execute(
        """SELECT id FROM concepts
           WHERE subject_key = ?
             AND superseded_by IS NULL
             AND id != ?
             AND is_current = 1
             AND status = 'active'""",
        (subject_key, row["concept_id"]),
    ).fetchall()
    for candidate in candidates:
        old_id = candidate["id"] if hasattr(candidate, "keys") else candidate[0]
        apply_lifecycle_transition_conn(
            conn,
            old_id,
            "supersede",
            superseded_by=row["concept_id"],
            reason="RETRIEVAL-072: subject-key dedup",
        )
        break
    _mark_done(conn, int(row["id"]), "done")


def _process_similarity_row(conn, row: dict[str, Any]) -> None:
    concept_row = _load_processable_concept(conn, row["concept_id"])
    if not _concept_is_processable(concept_row):
        _mark_done(conn, int(row["id"]), "skipped", "missing or non-current concept")
        return
    if _supersession_disabled():
        _mark_done(conn, int(row["id"]), "skipped", "supersession disabled")
        return

    from app.cognitive.supersession import check_supersession_on_write
    from app.storage import db_immediate

    with db_immediate(operation="autolearn_similarity_supersession") as tx_conn:
        check_supersession_on_write(row["concept_id"], tx_conn, raise_errors=True)
    _mark_done(conn, int(row["id"]), "done")


async def run_autolearn_maintenance_queue(
    conn,
    batch_size: int | None = None,
    task_type: str | None = None,
) -> int:
    """Drain a bounded batch of queued autolearn maintenance work."""
    from app.core.config import get_autolearn_maintenance_batch_size, get_autolearn_maintenance_enabled

    if not get_autolearn_maintenance_enabled():
        return 0

    _set_busy_timeout(conn)
    ensure_autolearn_maintenance_tables(conn)
    reset_count = _reset_stale_running(conn)
    if reset_count:
        logger.info("autolearn_maintenance_processor reset %d stale running row(s)", reset_count)

    rows = _claim_rows(
        conn,
        get_autolearn_maintenance_batch_size() if batch_size is None else batch_size,
        task_type=task_type,
    )
    if not rows:
        return 0

    processed = 0
    governance_rows = [row for row in rows if row["task_type"] == TASK_GOVERNANCE_RECOMPUTE]
    try:
        processed += _process_governance_rows(conn, governance_rows)
    except Exception as exc:
        for row in governance_rows:
            _mark_retry_or_failed(conn, row, exc)
        processed += len(governance_rows)

    for row in rows:
        if row["task_type"] == TASK_GOVERNANCE_RECOMPUTE:
            continue
        try:
            if row["task_type"] == TASK_SUBJECT_KEY_SUPERSESSION:
                _process_subject_key_row(conn, row)
            elif row["task_type"] == TASK_SIMILARITY_SUPERSESSION:
                _process_similarity_row(conn, row)
            else:
                _mark_done(conn, int(row["id"]), "failed", f"unsupported task_type: {row['task_type']}")
            processed += 1
        except Exception as exc:
            _mark_retry_or_failed(conn, row, exc)
            processed += 1

    logger.info("autolearn_maintenance_processor processed %d row(s)", processed)
    return processed


def get_autolearn_maintenance_drain_plan(
    conn,
    batch_size: int | None = None,
    task_type: str | None = None,
) -> dict[str, Any]:
    """Return a read-only queue drain plan for manual catch-up."""
    from app.core.config import get_autolearn_maintenance_batch_size

    selected_batch_size = get_autolearn_maintenance_batch_size() if batch_size is None else batch_size
    if task_type is not None and task_type not in {
        TASK_SUBJECT_KEY_SUPERSESSION,
        TASK_GOVERNANCE_RECOMPUTE,
        TASK_SIMILARITY_SUPERSESSION,
    }:
        raise ValueError(f"Unsupported autolearn maintenance task_type: {task_type}")
    _set_busy_timeout(conn)
    ensure_autolearn_maintenance_tables(conn)
    now = _utc_now_iso()
    counts = [
        dict(row)
        for row in conn.execute(
            """SELECT status, task_type, COUNT(*) AS count, MIN(created_at) AS oldest_created_at
               FROM autolearn_maintenance_queue
               GROUP BY status, task_type
               ORDER BY status, task_type"""
        ).fetchall()
    ]
    ready_filters = [
        "status='queued'",
        "(next_attempt_at IS NULL OR next_attempt_at <= ?)",
    ]
    ready_params: list[Any] = [now]
    if task_type is not None:
        ready_filters.append("task_type=?")
        ready_params.append(task_type)
    ready_by_task = {
        (row["task_type"] if hasattr(row, "keys") else row[0]): int(row["count"] if hasattr(row, "keys") else row[1])
        for row in conn.execute(
            f"""SELECT task_type, COUNT(*) AS count
               FROM autolearn_maintenance_queue
               WHERE {' AND '.join(ready_filters)}
               GROUP BY task_type
               ORDER BY task_type""",
            tuple(ready_params),
        ).fetchall()
    }
    total_ready = sum(ready_by_task.values())
    estimated_batches = 0
    if selected_batch_size > 0:
        estimated_batches = (total_ready + selected_batch_size - 1) // selected_batch_size
    return {
        "batch_size": selected_batch_size,
        "ready_total": total_ready,
        "ready_by_task": ready_by_task,
        "estimated_batches": estimated_batches,
        "task_type_filter": task_type,
        "counts": counts,
    }


async def run_autolearn_maintenance_catchup(
    conn,
    *,
    max_rows: int,
    max_seconds: int,
    batch_size: int | None = None,
    task_type: str | None = None,
) -> dict[str, Any]:
    """Drain queued autolearn maintenance rows under explicit row and time budgets."""
    from app.core.config import get_autolearn_maintenance_batch_size

    selected_batch_size = get_autolearn_maintenance_batch_size() if batch_size is None else batch_size
    started = time.monotonic()
    before = get_autolearn_maintenance_drain_plan(conn, selected_batch_size, task_type=task_type)
    processed_total = 0
    batches = 0
    while processed_total < max_rows and time.monotonic() - started < max_seconds:
        remaining = max_rows - processed_total
        current_batch_size = min(selected_batch_size, remaining)
        if current_batch_size <= 0:
            break
        processed = await run_autolearn_maintenance_queue(conn, current_batch_size, task_type=task_type)
        if processed <= 0:
            break
        processed_total += processed
        batches += 1

    after = get_autolearn_maintenance_drain_plan(conn, selected_batch_size, task_type=task_type)
    elapsed = time.monotonic() - started
    task_delta = {
        task_type: before["ready_by_task"].get(task_type, 0) - after["ready_by_task"].get(task_type, 0)
        for task_type in sorted(set(before["ready_by_task"]) | set(after["ready_by_task"]))
    }
    return {
        "success": True,
        "processed": processed_total,
        "batches": batches,
        "elapsed_seconds": round(elapsed, 3),
        "max_rows": max_rows,
        "max_seconds": max_seconds,
        "batch_size": selected_batch_size,
        "task_type_filter": task_type,
        "ready_before": before["ready_total"],
        "ready_after": after["ready_total"],
        "processed_by_task_estimate": task_delta,
        "before": before,
        "after": after,
    }
