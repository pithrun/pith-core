"""Server-owned drain for durable lifecycle/index reconciliation events."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from app.storage.lifecycle_index_outbox import (
    claim_lifecycle_index_events,
    load_membership_snapshots,
    retry_lifecycle_index_events,
    summarize_lifecycle_index_outbox,
    terminalize_lifecycle_index_events,
)

logger = logging.getLogger(__name__)

CHECK_INTERVAL_SECONDS = 2.0
BATCH_SIZE = 25
WARNING_QUEUED_AGE_SECONDS = 300.0

_drain_task: asyncio.Task | None = None
_required_worker = False
_disabled_reason: str | None = None
_consecutive_failures = 0
_ticks_total = 0
_drains_total = 0
_rows_completed_total = 0
_rows_retried_total = 0
_rows_failed_total = 0
_last_started_at: float | None = None
_last_completed_at: float | None = None
_last_error: str | None = None
_last_result: dict[str, Any] | None = None


def _bounded_error(error: BaseException | str | None) -> str | None:
    if error is None:
        return None
    return str(error)[:500]


def _rows_by_concept(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["concept_id"]), []).append(row)
    return grouped


def drain_lifecycle_index_outbox(*, limit: int = BATCH_SIZE) -> dict[str, Any]:
    """Claim and reconcile one durable batch without holding a DB transaction."""
    global _drains_total, _rows_completed_total, _rows_retried_total, _rows_failed_total
    global _last_started_at, _last_completed_at, _last_error, _last_result

    _last_started_at = time.time()
    rows = claim_lifecycle_index_events(limit=limit)
    if not rows:
        result = {
            "claimed": 0,
            "unique_concepts": 0,
            "completed": 0,
            "retried": 0,
            "failed": 0,
        }
        _last_completed_at = time.time()
        _last_result = result
        return result

    from app.retrieval import (
        ensure_lifecycle_concepts_indexed,
        evict_lifecycle_concepts,
    )

    grouped = _rows_by_concept(rows)
    concept_ids = list(grouped)
    before = load_membership_snapshots(concept_ids)
    active_ids = [concept_id for concept_id in concept_ids if before[concept_id]["active"]]
    inactive_ids = [concept_id for concept_id in concept_ids if not before[concept_id]["active"]]

    failed_by_id: dict[str, str] = {}
    if inactive_ids:
        eviction = evict_lifecycle_concepts(
            inactive_ids,
            persist=True,
            source="lifecycle_index_outbox",
        )
        for concept_id in eviction.get("failed", []):
            failed_by_id[concept_id] = "batch eviction verification failed"
    if active_ids:
        ensured = ensure_lifecycle_concepts_indexed(
            active_ids,
            persist=True,
            source="lifecycle_index_outbox",
        )
        reason = (
            f"lifecycle ensure deferred: {ensured['deferred']}"
            if ensured.get("deferred")
            else "batch ensure verification failed"
        )
        for concept_id in ensured.get("failed", []):
            failed_by_id[concept_id] = reason

    after = load_membership_snapshots(concept_ids)
    completed = 0
    retried = 0
    terminal_failed = 0
    for concept_id, concept_rows in grouped.items():
        if before[concept_id] != after[concept_id]:
            outcome = retry_lifecycle_index_events(
                concept_rows,
                error="authoritative membership changed during index reconciliation",
            )
            retried += outcome["queued"]
            terminal_failed += outcome["failed"]
            continue
        if concept_id in failed_by_id:
            outcome = retry_lifecycle_index_events(
                concept_rows,
                error=failed_by_id[concept_id],
            )
            retried += outcome["queued"]
            terminal_failed += outcome["failed"]
            continue
        completed += terminalize_lifecycle_index_events(
            [int(row["id"]) for row in concept_rows],
            status="done",
        )

    result = {
        "claimed": len(rows),
        "unique_concepts": len(concept_ids),
        "active_ensures": len(active_ids),
        "inactive_evictions": len(inactive_ids),
        "completed": completed,
        "retried": retried,
        "failed": terminal_failed,
    }
    _drains_total += 1
    _rows_completed_total += completed
    _rows_retried_total += retried
    _rows_failed_total += terminal_failed
    _last_completed_at = time.time()
    _last_error = None
    _last_result = result
    logger.info(
        "lifecycle index outbox drain claimed=%d unique=%d completed=%d retried=%d failed=%d",
        len(rows),
        len(concept_ids),
        completed,
        retried,
        terminal_failed,
    )
    return result


async def _drain_loop() -> None:
    global _consecutive_failures, _ticks_total, _last_error
    logger.info(
        "lifecycle index outbox drain started interval=%.1fs batch=%d",
        CHECK_INTERVAL_SECONDS,
        BATCH_SIZE,
    )
    while True:
        try:
            await asyncio.sleep(CHECK_INTERVAL_SECONDS)
            _ticks_total += 1
            await asyncio.to_thread(drain_lifecycle_index_outbox, limit=BATCH_SIZE)
            _consecutive_failures = 0
        except asyncio.CancelledError:
            logger.info("lifecycle index outbox drain cancelled")
            raise
        except Exception as exc:  # noqa: BLE001 - invariant worker must survive a tick
            _consecutive_failures += 1
            _last_error = _bounded_error(exc)
            logger.error(
                "lifecycle index outbox tick failed consecutive=%d error=%s",
                _consecutive_failures,
                exc,
                exc_info=True,
            )


async def start_lifecycle_index_drain() -> asyncio.Task:
    """Start the required server worker once retrieval initialization is ready."""
    global _drain_task, _required_worker, _disabled_reason
    _required_worker = True
    _disabled_reason = None
    if _drain_task is not None and not _drain_task.done():
        return _drain_task
    _drain_task = asyncio.create_task(_drain_loop(), name="pith-lifecycle-index-outbox")
    return _drain_task


def disable_lifecycle_index_drain(reason: str) -> None:
    """Record an allowed non-running state, currently benchmark-readonly only."""
    global _required_worker, _disabled_reason
    _required_worker = False
    _disabled_reason = _bounded_error(reason)


async def stop_lifecycle_index_drain() -> None:
    global _drain_task
    if _drain_task is not None and not _drain_task.done():
        _drain_task.cancel()
        try:
            await _drain_task
        except asyncio.CancelledError:
            pass
    _drain_task = None


def get_lifecycle_index_outbox_status() -> dict[str, Any]:
    """Combine durable queue state with bounded in-process worker liveness."""
    running = _drain_task is not None and not _drain_task.done()
    try:
        durable = summarize_lifecycle_index_outbox()
    except Exception as exc:
        return {
            "status": "critical",
            "alert": True,
            "required_worker": _required_worker,
            "worker_running": running,
            "disabled_reason": _disabled_reason,
            "queued_count": None,
            "running_count": None,
            "failed_count": None,
            "oldest_queued_age_seconds": None,
            "latest_error": _bounded_error(exc),
            "interval_seconds": CHECK_INTERVAL_SECONDS,
            "batch_size": BATCH_SIZE,
        }

    oldest = durable.get("oldest_queued_age_seconds")
    if durable["failed_count"] > 0 or (_required_worker and not running):
        status = "critical"
    elif _consecutive_failures > 0 or (oldest is not None and oldest > WARNING_QUEUED_AGE_SECONDS):
        status = "warning"
    else:
        status = "healthy"
    latest_error = durable.get("latest_error") or _last_error
    return {
        "status": status,
        "alert": status in {"warning", "critical"},
        "required_worker": _required_worker,
        "worker_running": running,
        "disabled_reason": _disabled_reason,
        **durable,
        "consecutive_failures": _consecutive_failures,
        "ticks_total": _ticks_total,
        "drains_total": _drains_total,
        "rows_completed_total": _rows_completed_total,
        "rows_retried_total": _rows_retried_total,
        "rows_failed_total": _rows_failed_total,
        "last_started_at": _last_started_at,
        "last_completed_at": _last_completed_at,
        "latest_error": latest_error,
        "last_result": _last_result,
        "warning_queued_age_seconds": WARNING_QUEUED_AGE_SECONDS,
        "interval_seconds": CHECK_INTERVAL_SECONDS,
        "batch_size": BATCH_SIZE,
    }
