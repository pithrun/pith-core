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
    "get_autolearn_maintenance_supervisor_status",
    "kick_autolearn_maintenance_drain",
    "run_autolearn_maintenance_queue",
    "run_autolearn_maintenance_supervisor_once",
    "start_autolearn_maintenance_supervisor",
    "stop_autolearn_maintenance_supervisor",
]

_DRAIN_TASK: asyncio.Task | None = None
_SYNC_DRAIN_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pith-autolearn-drain")
_SYNC_DRAIN_FUTURE: Future | None = None
_SYNC_DRAIN_LOCK = threading.Lock()
_SUPERVISOR_THREAD: threading.Thread | None = None
_SUPERVISOR_STOP = threading.Event()
_SUPERVISOR_THREAD_LOCK = threading.Lock()
_SUPERVISOR_PASS_LOCK = threading.Lock()
_SUPERVISOR_STATE_LOCK = threading.Lock()
_SUPERVISOR_STATE: dict[str, Any] = {
    "running": False,
    "last_status": "never_started",
    "last_started_at": None,
    "last_completed_at": None,
    "last_processed": 0,
    "last_error": None,
    "last_reason": None,
}


def _benchmark_mode_active() -> bool:
    return os.environ.get("PITH_BENCHMARK_MODE", "").lower() in {"1", "true", "yes", "on"}


def _pressure_backpressure_active() -> tuple[bool, str, str]:
    try:
        from app.ops.pressure_policy import foreground_pressure_mode, should_defer_background_maintenance
        from app.ops.pressure_state import build_pressure_state

        state = build_pressure_state(use_cache=True)
        mode = foreground_pressure_mode(state)
        if should_defer_background_maintenance(state):
            return True, mode, str(state.pressure_level)
        return False, mode, str(state.pressure_level)
    except Exception:
        return False, "unknown", "unknown"


def _record_metric(metric: str, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
    try:
        from app.ops.metrics import metrics

        metrics.record(metric, value, labels or {})
        metrics.flush()
    except Exception:
        pass


def _set_supervisor_state(**updates: Any) -> dict[str, Any]:
    with _SUPERVISOR_STATE_LOCK:
        _SUPERVISOR_STATE.update(updates)
        return dict(_SUPERVISOR_STATE)


def get_autolearn_maintenance_supervisor_status() -> dict[str, Any]:
    """Return in-memory autonomous maintenance supervisor status."""
    with _SUPERVISOR_THREAD_LOCK:
        thread_alive = bool(_SUPERVISOR_THREAD and _SUPERVISOR_THREAD.is_alive())
    with _SUPERVISOR_STATE_LOCK:
        state = dict(_SUPERVISOR_STATE)
    state["thread_alive"] = thread_alive
    state["stop_requested"] = _SUPERVISOR_STOP.is_set()
    return state


def _required_context_cache_servable() -> tuple[bool, str]:
    try:
        from app.session.required_context_cache import required_context_cache_status

        status = required_context_cache_status()
        if not status.servable:
            return False, f"state={status.state}"
        return True, f"state={status.state}"
    except Exception as exc:
        return False, f"error={_error_text(exc)}"


def _autolearn_pressure_starved(conn) -> bool:
    from app.core.config import (
        get_autolearn_catchup_enabled,
        get_autolearn_maintenance_pressure_starvation_seconds,
    )

    if not get_autolearn_catchup_enabled():
        return False
    threshold = get_autolearn_maintenance_pressure_starvation_seconds()
    if threshold <= 0:
        return True
    try:
        row = conn.execute(
            """SELECT MIN(created_at) AS oldest_created_at
               FROM autolearn_maintenance_queue
               WHERE status='queued'
                 AND (next_attempt_at IS NULL OR next_attempt_at <= ?)""",
            (_utc_now_iso(),),
        ).fetchone()
    except Exception:
        return False
    oldest = row["oldest_created_at"] if hasattr(row, "keys") else row[0] if row else None
    if not oldest:
        return False
    try:
        parsed = _utc_now().__class__.fromisoformat(str(oldest).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=_utc_now().tzinfo)
    except Exception:
        return False
    return (_utc_now() - parsed).total_seconds() >= threshold


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
    pressure_defer, pressure_mode, pressure_level = _pressure_backpressure_active()
    if pressure_defer:
        _record_metric(
            "autolearn_maintenance_pressure_deferred_total",
            1.0,
            {"mode": pressure_mode, "pressure_level": pressure_level, "path": "kick"},
        )
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
            ORDER BY
                CASE source
                    WHEN 'replay_fast_path' THEN 0
                    WHEN 'session_learn' THEN 1
                    ELSE 2
                END,
                created_at
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

    try:
        timeout_ms = float(os.environ.get("PITH_AUTOLEARN_SUPERSESSION_DB_TIMEOUT_MS", "50"))
    except (TypeError, ValueError):
        timeout_ms = 50.0
    timeout_s = max(0.0, timeout_ms / 1000.0)
    started = time.perf_counter()
    with db_immediate(timeout_s=timeout_s, operation="autolearn_similarity_supersession") as tx_conn:
        check_supersession_on_write(row["concept_id"], tx_conn, raise_errors=True)
    _record_metric(
        "autolearn_supersession_batch_ms",
        round((time.perf_counter() - started) * 1000.0, 2),
        {"task_type": TASK_SIMILARITY_SUPERSESSION},
    )
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
    pressure_defer, pressure_mode, pressure_level = _pressure_backpressure_active()
    if pressure_defer:
        if _autolearn_pressure_starved(conn):
            _record_metric(
                "autolearn_maintenance_pressure_starvation_override_total",
                1.0,
                {"mode": pressure_mode, "pressure_level": pressure_level, "path": "drain"},
            )
            batch_size = 1
        else:
            _record_metric(
                "autolearn_maintenance_pressure_deferred_total",
                1.0,
                {"mode": pressure_mode, "pressure_level": pressure_level, "path": "drain"},
            )
            return 0

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
    oldest_ready_at = conn.execute(
        f"""SELECT MIN(created_at) AS oldest_ready_at
           FROM autolearn_maintenance_queue
           WHERE {' AND '.join(ready_filters)}""",
        tuple(ready_params),
    ).fetchone()
    oldest_ready_value = (
        oldest_ready_at["oldest_ready_at"]
        if hasattr(oldest_ready_at, "keys")
        else oldest_ready_at[0] if oldest_ready_at else None
    )
    oldest_ready_age_seconds = None
    if oldest_ready_value:
        try:
            parsed = _utc_now().__class__.fromisoformat(str(oldest_ready_value).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=_utc_now().tzinfo)
            oldest_ready_age_seconds = round(max(0.0, (_utc_now() - parsed).total_seconds()), 1)
        except Exception:
            oldest_ready_age_seconds = None
    estimated_batches = 0
    if selected_batch_size > 0:
        estimated_batches = (total_ready + selected_batch_size - 1) // selected_batch_size
    return {
        "batch_size": selected_batch_size,
        "ready_total": total_ready,
        "ready_by_task": ready_by_task,
        "oldest_ready_at": oldest_ready_value,
        "oldest_ready_age_seconds": oldest_ready_age_seconds,
        "estimated_batches": estimated_batches,
        "task_type_filter": task_type,
        "counts": counts,
    }


def _terminal_task_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int]:
    terminal_statuses = {"done", "skipped", "failed"}

    def _terminal_counts(plan: dict[str, Any]) -> dict[str, int]:
        totals: dict[str, int] = {}
        for row in plan.get("counts", []):
            if row.get("status") not in terminal_statuses:
                continue
            task_type = str(row.get("task_type"))
            totals[task_type] = totals.get(task_type, 0) + int(row.get("count", 0))
        return totals

    before_counts = _terminal_counts(before)
    after_counts = _terminal_counts(after)
    return {
        task_type: max(0, after_counts.get(task_type, 0) - before_counts.get(task_type, 0))
        for task_type in sorted(set(before_counts) | set(after_counts))
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
    ready_task_delta = {
        task_type: before["ready_by_task"].get(task_type, 0) - after["ready_by_task"].get(task_type, 0)
        for task_type in sorted(set(before["ready_by_task"]) | set(after["ready_by_task"]))
    }
    processed_by_task = _terminal_task_delta(before, after)
    oldest_before = before.get("oldest_ready_age_seconds")
    oldest_after = after.get("oldest_ready_age_seconds")
    oldest_delta = None
    if oldest_before is not None and oldest_after is not None:
        oldest_delta = round(float(oldest_before) - float(oldest_after), 1)
    ready_delta = int(before["ready_total"]) - int(after["ready_total"])
    drain_rate = round(processed_total / elapsed, 3) if elapsed > 0 else float(processed_total)
    return {
        "success": True,
        "processed": processed_total,
        "batches": batches,
        "elapsed_seconds": round(elapsed, 3),
        "drain_rate_rows_per_second": drain_rate,
        "max_rows": max_rows,
        "max_seconds": max_seconds,
        "batch_size": selected_batch_size,
        "task_type_filter": task_type,
        "ready_before": before["ready_total"],
        "ready_after": after["ready_total"],
        "ready_delta": ready_delta,
        "oldest_ready_age_before_seconds": oldest_before,
        "oldest_ready_age_after_seconds": oldest_after,
        "oldest_ready_age_delta_seconds": oldest_delta,
        "processed_by_task": processed_by_task,
        "processed_by_task_estimate": ready_task_delta,
        "before": before,
        "after": after,
    }


def run_autolearn_maintenance_supervisor_once(
    *,
    reason: str = "autolearn_supervisor",
    batch_size: int | None = None,
    max_wall_seconds: int | None = None,
) -> dict[str, Any]:
    """Run one autonomous, bounded autolearn maintenance catch-up pass."""
    from app.core.config import (
        get_autolearn_maintenance_enabled,
        get_autolearn_maintenance_supervisor_batch_size,
        get_autolearn_maintenance_supervisor_max_wall_seconds,
    )
    from app.storage import owned_connection

    selected_batch_size = batch_size if batch_size is not None else get_autolearn_maintenance_supervisor_batch_size()
    selected_max_wall = (
        max_wall_seconds
        if max_wall_seconds is not None
        else get_autolearn_maintenance_supervisor_max_wall_seconds()
    )
    started_at = _utc_now_iso()
    if _benchmark_mode_active():
        result = {
            "success": True,
            "status": "skipped_benchmark_mode",
            "processed": 0,
            "reason": reason,
        }
        _set_supervisor_state(
            last_status=result["status"],
            last_started_at=started_at,
            last_completed_at=_utc_now_iso(),
            last_processed=0,
            last_error=None,
            last_reason=reason,
        )
        _record_metric("autolearn_maintenance_supervisor_pass_total", 1.0, {"status": result["status"]})
        return result
    if not get_autolearn_maintenance_enabled():
        result = {
            "success": True,
            "status": "skipped_disabled",
            "processed": 0,
            "reason": reason,
        }
        _set_supervisor_state(
            last_status=result["status"],
            last_started_at=started_at,
            last_completed_at=_utc_now_iso(),
            last_processed=0,
            last_error=None,
            last_reason=reason,
        )
        _record_metric("autolearn_maintenance_supervisor_pass_total", 1.0, {"status": result["status"]})
        return result
    if not _SUPERVISOR_PASS_LOCK.acquire(blocking=False):
        result = {
            "success": True,
            "status": "skipped_overlap",
            "processed": 0,
            "reason": reason,
        }
        _set_supervisor_state(
            last_status=result["status"],
            last_started_at=started_at,
            last_completed_at=_utc_now_iso(),
            last_processed=0,
            last_error=None,
            last_reason=reason,
        )
        _record_metric("autolearn_maintenance_supervisor_pass_total", 1.0, {"status": result["status"]})
        return result
    _set_supervisor_state(
        running=True,
        last_status="running",
        last_started_at=started_at,
        last_error=None,
        last_reason=reason,
    )
    try:
        servable, servable_reason = _required_context_cache_servable()
        if not servable:
            result = {
                "success": True,
                "status": "skipped_required_context_cache",
                "processed": 0,
                "reason": servable_reason,
            }
            _set_supervisor_state(
                running=False,
                last_status=result["status"],
                last_completed_at=_utc_now_iso(),
                last_processed=0,
                last_error=None,
                last_reason=servable_reason,
            )
            _record_metric("autolearn_maintenance_supervisor_pass_total", 1.0, {"status": result["status"]})
            return result
        with owned_connection() as conn:
            result = asyncio.run(
                run_autolearn_maintenance_catchup(
                    conn,
                    max_rows=selected_batch_size,
                    max_seconds=selected_max_wall,
                    batch_size=selected_batch_size,
                )
            )
        processed = int(result.get("processed", 0) or 0)
        status = "processed" if processed > 0 else "idle"
        result = {**result, "status": status, "reason": reason}
        _set_supervisor_state(
            running=False,
            last_status=status,
            last_completed_at=_utc_now_iso(),
            last_processed=processed,
            last_error=None,
            last_reason=reason,
        )
        _record_metric("autolearn_maintenance_supervisor_pass_total", 1.0, {"status": status})
        _record_metric("autolearn_maintenance_supervisor_processed_total", float(processed))
        return result
    except Exception as exc:
        error = _error_text(exc)
        _set_supervisor_state(
            running=False,
            last_status="error",
            last_completed_at=_utc_now_iso(),
            last_processed=0,
            last_error=error,
            last_reason=reason,
        )
        _record_metric("autolearn_maintenance_supervisor_pass_total", 1.0, {"status": "error"})
        _record_metric("autolearn_maintenance_supervisor_error_total", 1.0)
        logger.warning("autolearn_maintenance_supervisor pass failed: %s", exc, exc_info=True)
        return {
            "success": False,
            "status": "error",
            "processed": 0,
            "reason": reason,
            "error": error,
        }
    finally:
        try:
            _SUPERVISOR_PASS_LOCK.release()
        except RuntimeError:
            pass


def _autolearn_maintenance_supervisor_loop(
    *,
    interval_seconds: int,
    batch_size: int,
    max_wall_seconds: int,
) -> None:
    while not _SUPERVISOR_STOP.is_set():
        run_autolearn_maintenance_supervisor_once(
            batch_size=batch_size,
            max_wall_seconds=max_wall_seconds,
        )
        _SUPERVISOR_STOP.wait(interval_seconds)


def start_autolearn_maintenance_supervisor(
    *,
    interval_seconds: int,
    batch_size: int,
    max_wall_seconds: int,
) -> bool:
    """Start the autonomous autolearn maintenance supervisor thread."""
    global _SUPERVISOR_THREAD
    with _SUPERVISOR_THREAD_LOCK:
        if _SUPERVISOR_THREAD and _SUPERVISOR_THREAD.is_alive():
            return False
        _SUPERVISOR_STOP.clear()
        _SUPERVISOR_THREAD = threading.Thread(
            target=_autolearn_maintenance_supervisor_loop,
            kwargs={
                "interval_seconds": interval_seconds,
                "batch_size": batch_size,
                "max_wall_seconds": max_wall_seconds,
            },
            name="pith-autolearn-maintenance-supervisor",
            daemon=True,
        )
        _SUPERVISOR_THREAD.start()
        _set_supervisor_state(last_status="started", last_error=None, last_reason="startup")
        return True


def stop_autolearn_maintenance_supervisor(*, wait: bool = True) -> None:
    """Stop the autonomous autolearn maintenance supervisor thread."""
    global _SUPERVISOR_THREAD
    _SUPERVISOR_STOP.set()
    thread = None
    with _SUPERVISOR_THREAD_LOCK:
        thread = _SUPERVISOR_THREAD
    if wait and thread and thread.is_alive():
        thread.join(timeout=5)
    with _SUPERVISOR_THREAD_LOCK:
        if _SUPERVISOR_THREAD is thread:
            _SUPERVISOR_THREAD = None
    _set_supervisor_state(running=False, last_status="stopped", last_completed_at=_utc_now_iso())
