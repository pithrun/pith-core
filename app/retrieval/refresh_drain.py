"""Steady-state TF-IDF + embedding refresh queue and debounced drain.

RETRIEVAL-125 Phase C. ``add_concept(existing_id)`` is an add-only no-op, so when
an existing concept's searchable text changes (evolve / re-learn), its stored
term-counts and embedding go stale and never self-heal. The bulk rebuild
(rebuild_and_swap_repair) fixes the whole corpus periodically; this queue closes
the steady-state gap by refreshing individual concepts shortly after they drift.

Flow:
  * The single enqueue chokepoint is ``RetrievalEngine._add_concept_inner``: when
    ``index.add_concept`` returns False (already indexed) AND the stored terms
    differ from the freshly-assembled text, the concept id is enqueued here.
  * A dedicated background loop (``start_refresh_drain``) periodically drains the
    queue once it has been quiet for a debounce window (coalescing bursts), or
    once the oldest pending id exceeds a max-wait ceiling (anti-starvation),
    calling ``RetrievalEngine.refresh_concepts`` on the batch.

The whole feature is gated by ``PITH_TFIDF_REFRESH_DRAIN`` (default OFF): when
disabled there is zero hot-path cost (no staleness check, no enqueue) and the
drain loop is never started. Enablement is a deliberate operator flip.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time

logger = logging.getLogger(__name__)


def drain_enabled() -> bool:
    return os.environ.get("PITH_TFIDF_REFRESH_DRAIN", "").strip().lower() in ("1", "true", "yes", "on")


def _clamp_float(name: str, default: float, lo: float, hi: float) -> float:
    try:
        raw = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, raw))


def _clamp_int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        raw = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, raw))


# Quiet period before a batch drains (debounce — coalesces evolve bursts).
DEBOUNCE_S = _clamp_float("PITH_TFIDF_REFRESH_DEBOUNCE_S", 120.0, 10.0, 3600.0)
# How often the loop wakes to check whether a batch is due.
CHECK_INTERVAL_S = _clamp_float("PITH_TFIDF_REFRESH_CHECK_INTERVAL_S", 30.0, 5.0, 3600.0)
# Hard ceiling on how long an id may wait even under sustained activity.
MAX_WAIT_S = _clamp_float("PITH_TFIDF_REFRESH_MAX_WAIT_S", 900.0, 30.0, 86400.0)
# Max ids refreshed in a single drain (one force_idf_recalculation per batch).
MAX_BATCH = _clamp_int("PITH_TFIDF_REFRESH_MAX_BATCH", 200, 1, 5000)
# Bound on queue memory; beyond this, enqueues are dropped (counted).
QUEUE_CAP = _clamp_int("PITH_TFIDF_REFRESH_QUEUE_CAP", 10000, 100, 1_000_000)


class RefreshQueue:
    """Thread-safe de-duplicating set of concept ids pending a refresh."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: set[str] = set()
        self._last_enqueue = 0.0  # monotonic of the most recent enqueue
        self._oldest = 0.0        # monotonic when the set last went empty -> non-empty
        self._dropped = 0

    def enqueue(self, concept_id: str, *, now: float | None = None) -> None:
        if not concept_id:
            return
        now = time.monotonic() if now is None else now
        with self._lock:
            if concept_id in self._pending:
                self._last_enqueue = now
                return
            if len(self._pending) >= QUEUE_CAP:
                self._dropped += 1
                return
            if not self._pending:
                self._oldest = now
            self._pending.add(concept_id)
            self._last_enqueue = now

    def take_due_batch(
        self,
        *,
        debounce_s: float = DEBOUNCE_S,
        max_wait_s: float = MAX_WAIT_S,
        max_batch: int = MAX_BATCH,
        now: float | None = None,
    ) -> list[str]:
        """Return (and remove) a batch iff the queue is due, else []."""
        now = time.monotonic() if now is None else now
        with self._lock:
            if not self._pending:
                return []
            quiet = (now - self._last_enqueue) >= debounce_s
            starved = (now - self._oldest) >= max_wait_s
            if not (quiet or starved):
                return []
            batch = sorted(self._pending)[:max_batch]
            for cid in batch:
                self._pending.discard(cid)
            # Reset the oldest stamp for whatever remains.
            self._oldest = now if self._pending else 0.0
            return batch

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def stats(self) -> dict:
        with self._lock:
            return {"pending": len(self._pending), "dropped": self._dropped}


# Process-wide singleton.
refresh_queue = RefreshQueue()


def drain_due(engine, *, now: float | None = None) -> dict | None:
    """Drain one due batch through ``engine.refresh_concepts``.

    Returns the refresh report, or None if no batch was due. On a deferred result
    (a reflection writer holds the lock) the batch is re-enqueued so nothing is
    lost. Exceptions propagate to the caller (the loop logs + counts them).
    """
    batch = refresh_queue.take_due_batch(now=now)
    if not batch:
        return None
    report = engine.refresh_concepts(batch, persist=True, refresh_embeddings=True)
    _record_drain(report)  # MONITOR-163: retain a compact summary for observability
    if report.get("deferred"):
        # Could not acquire the reflection lock — retry on a later tick.
        for cid in batch:
            refresh_queue.enqueue(cid, now=now)
        logger.info("tfidf refresh drain: deferred (%s) — re-enqueued %d ids", report["deferred"], len(batch))
    else:
        logger.info(
            "tfidf refresh drain: refreshed=%d skipped=%d df_delta=%s embeddings=%d pending=%d",
            len(report.get("refreshed", [])),
            len(report.get("skipped", [])),
            report.get("df_recount_delta"),
            report.get("embeddings_refreshed", 0),
            refresh_queue.pending_count(),
        )
    return report


# --- Background drain loop (mirrors maintenance_scheduler lifecycle) ---

_drain_task: asyncio.Task | None = None
_consecutive_failures = 0
MAX_CONSECUTIVE_FAILURES = 5

# MONITOR-163: compact last-drain summary + durable counters for out-of-process
# observability (a separate monitor process can only see drain state over HTTP via
# /maintenance/status -> get_drain_status). Counts/scalars only — never concept-id
# lists (no id leakage, O(1) memory). _failures_total/_last_error are durable: they
# survive the per-success reset of _consecutive_failures, so an out-of-process monitor
# can detect intermittent failures the volatile counter hides.
_last_drain: dict | None = None
_drains_total = 0
_deferred_total = 0
_failures_total = 0
_last_error: str | None = None


def _record_drain(report: dict) -> None:
    """MONITOR-163: capture a compact summary of one completed drain.

    INVARIANT: reassigns ``_last_drain`` to a FRESH dict — never mutates in place —
    so the lock-free read in ``get_drain_status`` can never observe a torn dict.
    """
    global _last_drain, _drains_total, _deferred_total
    deferred = report.get("deferred")
    if deferred:
        _deferred_total += 1
    else:
        _drains_total += 1
    _last_drain = {
        "finished_at": time.time(),
        "deferred": deferred,
        "refreshed_count": len(report.get("refreshed", [])),
        "skipped_count": len(report.get("skipped", [])),
        "df_recount_delta": report.get("df_recount_delta"),
        "embeddings_refreshed": report.get("embeddings_refreshed", 0),
        "duration_s": report.get("duration_s"),
    }


def get_drain_status() -> dict:
    return {
        "enabled": drain_enabled(),
        "running": _drain_task is not None and not _drain_task.done(),
        "consecutive_failures": _consecutive_failures,
        "failures_total": _failures_total,
        "last_error": _last_error,
        "drains_total": _drains_total,
        "deferred_total": _deferred_total,
        "last_drain": _last_drain,
        "config": {
            "debounce_s": DEBOUNCE_S,
            "check_interval_s": CHECK_INTERVAL_S,
            "max_wait_s": MAX_WAIT_S,
            "max_batch": MAX_BATCH,
            "queue_cap": QUEUE_CAP,
        },
        **refresh_queue.stats(),
    }


async def _drain_loop() -> None:
    global _consecutive_failures, _failures_total, _last_error
    from app.retrieval import retrieval_engine

    logger.info(
        "tfidf refresh drain: started (debounce=%.0fs check=%.0fs max_wait=%.0fs max_batch=%d)",
        DEBOUNCE_S, CHECK_INTERVAL_S, MAX_WAIT_S, MAX_BATCH,
    )
    while True:
        try:
            await asyncio.sleep(CHECK_INTERVAL_S)
            if refresh_queue.pending_count() == 0:
                continue
            # refresh_concepts is synchronous + CPU-bound (force_idf_recalculation).
            await asyncio.to_thread(drain_due, retrieval_engine)
            _consecutive_failures = 0
        except asyncio.CancelledError:
            logger.info("tfidf refresh drain: cancelled (shutdown)")
            raise
        except Exception as e:  # noqa: BLE001 — keep the loop alive
            _failures_total += 1  # MONITOR-163: durable failure count (survives the success-reset below)
            _last_error = repr(e)[:200]
            _consecutive_failures += 1
            logger.error(
                "tfidf refresh drain: tick failed (%d/%d): %s",
                _consecutive_failures, MAX_CONSECUTIVE_FAILURES, e, exc_info=True,
            )
            if _consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                logger.error("tfidf refresh drain: too many consecutive failures — stopping loop")
                return


async def start_refresh_drain() -> asyncio.Task | None:
    """Start the drain loop, or return None if the feature is disabled."""
    global _drain_task
    if not drain_enabled():
        logger.info("tfidf refresh drain: DISABLED (set PITH_TFIDF_REFRESH_DRAIN=1 to enable)")
        return None
    if _drain_task is not None and not _drain_task.done():
        return _drain_task
    _drain_task = asyncio.create_task(_drain_loop(), name="pith-tfidf-refresh-drain")
    return _drain_task


async def stop_refresh_drain() -> None:
    global _drain_task
    if _drain_task is not None and not _drain_task.done():
        _drain_task.cancel()
        try:
            await _drain_task
        except asyncio.CancelledError:
            pass
        logger.info("tfidf refresh drain: stopped")
    _drain_task = None
