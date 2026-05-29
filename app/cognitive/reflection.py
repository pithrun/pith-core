"""Reflection engine for consolidation, decay, cleanup, and forgetting.

Phase 1A: Added forgetting mechanism and access tracker flush.
Forgetting uses salience, access metrics, and staleness —
the archive/recover behavior preserves all data (archive-not-delete).

Phase 1A D4: Deduplication rewrite (merge pipeline) and confidence
recalibration (evidence strength scoring). Both share _compute_evidence_strength().
"""

import logging
import math
import re
import sqlite3
import statistics
import threading
import time
from datetime import UTC, datetime, timedelta

from app.core.datetime_utils import _ensure_aware, _utc_now, _utc_now_iso
# DEBT-239: cognitive must not import governance directly (Contract 3).
def _get_auto_graduate_quarantined():
    """Lazy loader — avoids cognitive→governance static import (Contract 3)."""
    import importlib
    return importlib.import_module("app.governance.quarantine").auto_graduate_quarantined
from app.governance.currency import batch_compute_currency
from app.storage import (
    _db,
    _db_immediate,
    access_tracker,
    apply_lifecycle_transition_conn,
    archive_concept,
    count_orphan_concepts,
    list_concepts,
    list_concepts_full,
    list_concepts_modified_since,
    load_concept,
    save_concept,
)


def _elapsed_ms(start: float) -> float:
    """Milliseconds elapsed since start (monotonic). DEBT-039."""
    return round((time.monotonic() - start) * 1000, 1)


from app.core.models import Concept, Evidence, ReflectionSummary
from app.retrieval import retrieval_engine
# DEBT-237: cognitive must not import session directly (Contract 3).
# Using importlib lazy loader to break static dependency.
def _get_self_model_manager():
    """Lazy loader — avoids cognitive→session static import (Contract 3)."""
    import importlib
    return importlib.import_module("app.session.self_model").self_model_manager

# Configuration
DECAY_RATE = 0.02  # 2% decay per month
MIN_CONFIDENCE_THRESHOLD = 0.1  # Remove concepts below this
CONSOLIDATION_INTERVAL = 3600  # 1 hour in seconds

# Forgetting thresholds — uses salience field and access metrics
FORGETTING_SALIENCE_THRESHOLD = 0.15  # Below this = candidate
FORGETTING_ACCESS_COUNT_THRESHOLD = 2  # Below this = candidate
FORGETTING_STALENESS_DAYS = 90  # Not accessed in N days = candidate
MAX_FORGETTING_PER_CYCLE = 50  # REFLECT-005: Safety cap per reflect cycle

# Deduplication merge threshold
# REFLECT-006: Lowered from 0.92 (unreachable with TF-IDF) to 0.85 (embedding scale)
MERGE_SIMILARITY_THRESHOLD = 0.85  # Embedding cosine >= 0.85 = refinement merge
MAX_MERGES_PER_CYCLE = 10  # Safety cap per reflection cycle

# Confidence recalibration
# Recalibration tuning (post-Tier-1-remediation, 2026-02-18):
# - DAMPING reduced from 0.3 to 0.1: gentler corrections
# - GAP_THRESHOLD 0.10: ignore small confidence-evidence gaps
# - Combined with legacy count bonus, recalibration is intentionally
#   gentle to preserve manually-calibrated confidence values.
# - To strengthen: reduce GAP_THRESHOLD or increase DAMPING.
# REFLECT-013: Increased 0.1→0.15 — halves correction cycles (10→~7) without overshoot risk.
RECALIBRATION_DAMPING = 0.10  # EUNOMIA-038: Reduced from 0.15 — compensates for dead zone removal
RECALIBRATION_GAP_THRESHOLD = 0.00  # EUNOMIA-038: Dead zone removed — creates path-dependent equilibria (was 0.05)
RECALIBRATION_LOG_MIN_CORRECTION = 0.05  # MONITOR-086: Min correction to log per-concept event (was 0.01)

# DEBT-121: Import from shared constants (single source of truth)
from app.core.constants import LEGACY_EVIDENCE_STRENGTH, RECENCY_FLOOR, RECENCY_LAMBDA, RECALIBRATION_K_BASE, RECALIBRATION_K_SLOPE
from app.core.metrics_facade import metrics

# MONITOR-007: Saturation alerting thresholds
SATURATION_THRESHOLD = 0.95  # Score above which a concept is "saturated"
SATURATION_ALERT_PCT = 0.95  # % of concepts that must exceed threshold to trigger alert

# Source reliability baselines by source type
SOURCE_RELIABILITY = {
    "external_data": 0.9,
    "documented_observation": 0.85,
    "document": 0.85,
    "conversation": 0.7,
    "inference": 0.6,
}

logger = logging.getLogger(__name__)


class ReflectionAborted(RuntimeError):
    """Cooperative stop signal for deadline/cancellation-aware reflection."""

    def __init__(self, reason: str, stage: str):
        super().__init__(f"{reason}:{stage}")
        self.reason = reason
        self.stage = stage


def _batch_update_confidence_stability(updates: list[tuple]) -> int:
    """REFLECT-022: Batch-write confidence and stability to DB via direct SQL.

    Replaces per-concept save_concept() (~13ms each) with batch executemany
    + json_set for just the changed fields.

    Args:
        updates: List of (confidence, stability_or_None, concept_id) tuples.

    Returns:
        Count of updated rows, or 0 on failure (transaction rolls back).
    """
    if not updates:
        return 0
    try:
        now = _utc_now_iso()

        with _db() as conn:
            conf_only = [(c, cid) for c, s, cid in updates if s is None]
            conf_and_stab = [(c, s, cid) for c, s, cid in updates if s is not None]

            # Batch 1: confidence-only updates
            if conf_only:
                conn.executemany(
                    "UPDATE concepts SET confidence=?, updated_at=? WHERE id=?",
                    [(c, now, cid) for c, cid in conf_only],
                )
                for c, cid in conf_only:
                    conn.execute(
                        "UPDATE concepts SET data=json_set(data, '$.confidence', ?) WHERE id=?",
                        (c, cid),
                    )

            # Batch 2: confidence + stability updates
            if conf_and_stab:
                conn.executemany(
                    "UPDATE concepts SET confidence=?, stability=?, updated_at=? WHERE id=?",
                    [(c, s, now, cid) for c, s, cid in conf_and_stab],
                )
                for c, s, cid in conf_and_stab:
                    conn.execute(
                        "UPDATE concepts SET data=json_set(data, "
                        "'$.confidence', ?, '$.stability', ?) WHERE id=?",
                        (c, s, cid),
                    )

        logger.info("REFLECT-022: Batch confidence/stability updated %d concepts", len(updates))
        return len(updates)
    except sqlite3.OperationalError as e:
        logger.error("REFLECT-022: Batch update FAILED (rollback): %s", e)
        return 0
    except Exception as e:
        logger.error("REFLECT-022: Unexpected batch update error: %s", e)
        return 0


def _compute_evidence_strength(concept: Concept) -> float:
    """Compute evidence-justified confidence: E(c) = weighted_mean(E(e_i)).

    Handles both:
      - Structured Evidence objects (full strength scoring)
      - Legacy string evidence (pre-migration)

    For structured Evidence:
      E(e) = reliability × directness × corroboration × recency
      (consistency removed in MAINT-009 — had CV=0.000, zero discrimination)
      where corroboration = 1 - exp(-(N-1)) for concepts with N evidence items
            (concept-level, not per-item), else 0.5 for single-evidence concepts
            recency = exp(-λ × age_days)

    For dict evidence (MAINT-006 A.5+B-rev):
      - reliability uses stored reliability_weight with SOURCE_RELIABILITY fallback
      - corroboration is concept-level based on total evidence count

    For legacy string evidence:
      E(e) = LEGACY_EVIDENCE_STRENGTH (0.448)
      Plus count_bonus: min(0.15, legacy_count * 0.015) applied on top of base
      strength when legacy evidence is present (scaled by legacy_fraction when
      mixed with structured evidence). This is undocumented debt from the
      pre-migration era — see DEBT-127.

    Returns 0.0 if concept has no evidence.
    """
    if not concept.evidence:
        return 0.0

    # MAINT-006 B-rev: Concept-level corroboration from evidence count.
    # A concept observed N times is more corroborated than one observed once.
    # Exponential curve saturates quickly: 2→0.63, 3→0.86, 5→0.98.
    all_structured = [ev for ev in concept.evidence if isinstance(ev, (Evidence, dict))]  # noqa: UP038
    evidence_count = len(all_structured)
    concept_corroboration = (
        1.0 - math.exp(-(evidence_count - 1)) if evidence_count > 1 else 0.6
    )  # CONFIDENCE-FIX: floor raised (preserves monotonicity < 2-ev 0.632)

    scores = []

    for ev in concept.evidence:
        if isinstance(ev, Evidence):
            # Structured Evidence object — full strength scoring
            # MAINT-009: 4-factor formula (consistency removed — CV=0.000, never discriminated)
            recency = max(RECENCY_FLOOR, math.exp(-RECENCY_LAMBDA * ev.age_days))  # MAINT-032
            score = ev.reliability_weight * ev.directness * concept_corroboration * recency
            scores.append(score)
        elif isinstance(ev, dict):
            # Dict-form evidence (e.g., from YAML deserialization)
            source_type = ev.get("source_type", "conversation")
            # MAINT-006 A.5: Use stored reliability_weight with lookup fallback
            reliability = ev.get("reliability_weight", SOURCE_RELIABILITY.get(source_type, 0.7))
            directness = ev.get("directness", 0.8)
            # MAINT-009: consistency removed from formula (CV=0.000, never discriminated)

            timestamp = ev.get("timestamp") or ev.get("observed_at")
            if timestamp:
                try:
                    ev_date = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    age_days = max(0, (_utc_now() - _ensure_aware(ev_date)).days)
                    recency = max(RECENCY_FLOOR, math.exp(-RECENCY_LAMBDA * age_days))  # MAINT-032
                except (ValueError, TypeError):
                    recency = 0.8
            else:
                recency = 0.8

            score = reliability * directness * concept_corroboration * recency
            scores.append(score)
        elif isinstance(ev, str):
            # Legacy string evidence — conservative flat baseline
            scores.append(LEGACY_EVIDENCE_STRENGTH)
        # else: unknown type, skip

    if not scores:
        return 0.0

    base = sum(scores) / len(scores)

    # Count bonus for legacy string evidence only.
    # Structured evidence has per-item quality scoring and needs no inflation.
    # +0.015 per legacy item, capped at +0.15 (10 items max effect).
    legacy_count = sum(1 for ev in concept.evidence if isinstance(ev, str))

    if legacy_count == len(concept.evidence):
        # All legacy: full count bonus
        count_bonus = min(0.15, legacy_count * 0.015)
    elif legacy_count > 0:
        # Mixed: proportional bonus for legacy fraction only
        legacy_fraction = legacy_count / len(concept.evidence)
        count_bonus = min(0.15, legacy_count * 0.015) * legacy_fraction
    else:
        # All structured: no bonus needed
        count_bonus = 0.0

    return min(1.0, base + count_bonus)


def _collect_factor_values(
    concept: "Concept",
    factor_lists: dict[str, list],
) -> None:
    """MEASURE-011: Extract per-factor values from a concept's evidence for CV calculation.

    Appends factor values to the provided lists. Handles structured Evidence,
    dict evidence, and legacy string evidence (skipped — no factor breakdown).
    """
    if not concept.evidence:
        return

    all_structured = [ev for ev in concept.evidence if isinstance(ev, (Evidence, dict))]  # noqa: UP038
    evidence_count = len(all_structured)
    concept_corroboration = (
        1.0 - math.exp(-(evidence_count - 1)) if evidence_count > 1 else 0.6
    )  # CONFIDENCE-FIX: floor raised (preserves monotonicity < 2-ev 0.632)

    for ev in concept.evidence:
        if isinstance(ev, Evidence):
            recency = max(RECENCY_FLOOR, math.exp(-RECENCY_LAMBDA * ev.age_days))  # MAINT-032
            # MAINT-009: 4-factor composite (consistency removed from formula)
            composite = ev.reliability_weight * ev.directness * concept_corroboration * recency
            factor_lists["reliability"].append(ev.reliability_weight)
            factor_lists["directness"].append(ev.directness)
            factor_lists["consistency"].append(ev.consistency)  # Still collected for CV tracking
            factor_lists["corroboration"].append(concept_corroboration)
            factor_lists["recency"].append(recency)
            factor_lists["composite"].append(composite)
        elif isinstance(ev, dict):
            source_type = ev.get("source_type", "conversation")
            reliability = ev.get("reliability_weight", SOURCE_RELIABILITY.get(source_type, 0.7))
            directness = ev.get("directness", 0.8)
            consistency = ev.get("consistency", 0.8)  # Still collected for CV tracking

            timestamp = ev.get("timestamp") or ev.get("observed_at")
            if timestamp:
                try:
                    ev_date = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    age_days = max(0, (_utc_now() - _ensure_aware(ev_date)).days)
                    recency = max(RECENCY_FLOOR, math.exp(-RECENCY_LAMBDA * age_days))  # MAINT-032
                except (ValueError, TypeError):
                    recency = 0.8
            else:
                recency = 0.8

            # MAINT-009: 4-factor composite (consistency removed from formula)
            composite = reliability * directness * concept_corroboration * recency
            factor_lists["reliability"].append(reliability)
            factor_lists["directness"].append(directness)
            factor_lists["consistency"].append(consistency)  # Track but don't use in formula
            factor_lists["corroboration"].append(concept_corroboration)
            factor_lists["recency"].append(recency)
            factor_lists["composite"].append(composite)
        # Legacy string evidence: skip — no factor breakdown available


def _compute_factor_cvs(factor_lists: dict[str, list]) -> dict[str, float | None]:
    """MEASURE-011: Compute coefficient of variation for each evidence factor.

    CV = stdev / mean. Returns None for factors with < 2 samples or mean == 0.
    """
    result: dict[str, float | None] = {}
    for factor, values in factor_lists.items():
        if len(values) < 2:
            result[factor] = None
            continue
        mean = statistics.mean(values)
        if mean == 0.0:
            result[factor] = None
            continue
        stdev = statistics.stdev(values)
        result[factor] = round(stdev / mean, 4)
    return result


def _flush_recalibration_events(events: list[tuple]) -> None:
    """DATA-047 + MONITOR-086: Emit summary event + per-concept events for significant corrections only.

    Replaces the old pattern of logging every correction > 0.01. Now:
    - ONE summary event per cycle with aggregate stats (always emitted)
    - Per-concept events ONLY for corrections > RECALIBRATION_LOG_MIN_CORRECTION (0.05)
    """
    if not events:
        return
    import json

    now = _utc_now_iso()

    # MONITOR-086: Compute summary statistics
    direction_counts = {"downward": 0, "upward": 0, "psis_cap": 0}
    corrections = []
    for _cid, _old, _new, _es, direction, correction in events:
        direction_counts[direction] = direction_counts.get(direction, 0) + 1
        corrections.append(abs(correction))

    significant_events = [
        e for e in events if abs(e[5]) > RECALIBRATION_LOG_MIN_CORRECTION
    ]

    summary_details = json.dumps({
        "recalibrated_count": len(events),
        "downward": direction_counts.get("downward", 0),
        "upward": direction_counts.get("upward", 0),
        "psis_cap": direction_counts.get("psis_cap", 0),
        "avg_correction": round(sum(corrections) / len(corrections), 4) if corrections else 0,
        "max_correction": round(max(corrections), 4) if corrections else 0,
        "significant_count": len(significant_events),
    })

    # Build rows: 1 summary + N significant per-concept events
    rows = [(
        "CONFIDENCE_RECALIBRATION_SUMMARY",
        "system",
        summary_details,
        now,
    )]
    for concept_id, old_conf, new_conf, e_strength, direction, correction in significant_events:
        rows.append((
            "CONFIDENCE_RECALIBRATION",
            concept_id,
            json.dumps({
                "old_confidence": round(old_conf, 4),
                "new_confidence": round(new_conf, 4),
                "evidence_strength": round(e_strength, 4),
                "direction": direction,
                "correction": round(correction, 4),
            }),
            now,
        ))
    try:
        with _db() as conn:
            conn.executemany(
                """INSERT INTO governance_events (event_type, concept_id, details, created_at)
                   VALUES (?, ?, ?, ?)""",
                rows,
            )
    except Exception as e:
        logger.debug(f"DATA-047: Batch gov event flush failed (non-fatal): {e}")


class ReflectionEngine:
    """Manages memory consolidation, decay, and cleanup."""

    def __init__(self):
        self.last_reflection = None
        self._cancel_event: threading.Event | None = None
        self._deadline_monotonic: float | None = None
        self._long_step_min_remaining_seconds: float | None = None
        self._current_phase_timings: dict[str, float] | None = None
        self._current_counts: dict[str, int] | None = None
        self._last_completed_step: str | None = None

    def _begin_reflection_run(
        self,
        *,
        cancel_event: threading.Event | None,
        deadline_monotonic: float | None,
        long_step_min_remaining_seconds: float | None = None,
    ) -> None:
        self._cancel_event = cancel_event
        self._deadline_monotonic = deadline_monotonic
        self._long_step_min_remaining_seconds = long_step_min_remaining_seconds
        self._current_phase_timings = {}
        self._current_counts = {
            "concepts_consolidated": 0,
            "concepts_decayed": 0,
            "concepts_recalibrated": 0,
            "concepts_archived": 0,
            "associations_updated": 0,
            "questions_generated": 0,
            "gc_queue_remaining": 0,
            "concepts_graduated": 0,
            "concepts_discarded_quarantine": 0,
            "concepts_time_matured": 0,
            "concepts_assoc_propagated": 0,
            "concepts_auto_associated": 0,
            "concepts_currency_recomputed": 0,
            "concepts_promoted": 0,
        }
        self._last_completed_step = None

    def _end_reflection_run(self) -> None:
        self._cancel_event = None
        self._deadline_monotonic = None
        self._long_step_min_remaining_seconds = None
        self._current_phase_timings = None
        self._current_counts = None
        self._last_completed_step = None

    def _record_phase(self, phase_name: str, elapsed_ms: float) -> None:
        if self._current_phase_timings is not None:
            self._current_phase_timings[phase_name] = elapsed_ms
        self._last_completed_step = phase_name

    def _record_count(self, key: str, value: int) -> None:
        if self._current_counts is not None:
            self._current_counts[key] = value

    def _abort_requested(self) -> bool:
        if self._cancel_event is not None and self._cancel_event.is_set():
            return True
        return self._deadline_monotonic is not None and time.monotonic() >= self._deadline_monotonic

    def _abort_reason(self) -> str:
        if self._cancel_event is not None and self._cancel_event.is_set():
            return "cancelled"
        if self._deadline_monotonic is not None and time.monotonic() >= self._deadline_monotonic:
            return "deadline_exceeded"
        return "aborted"

    def _check_abort(self, stage: str) -> None:
        if self._abort_requested():
            raise ReflectionAborted(self._abort_reason(), stage)

    def _check_long_step_budget(self, stage: str) -> None:
        """Abort before known long steps if the remaining budget cannot absorb them."""
        self._check_abort(stage)
        if self._deadline_monotonic is None or self._long_step_min_remaining_seconds is None:
            return
        remaining_seconds = self._deadline_monotonic - time.monotonic()
        if remaining_seconds < self._long_step_min_remaining_seconds:
            raise ReflectionAborted("deadline_exceeded", stage)

    def _check_abort_every(self, index: int, every: int, stage: str) -> None:
        if index > 0 and index % every == 0:
            self._check_abort(stage)

    def _build_abort_summary(
        self,
        *,
        abort_reason: str | None = None,
        abort_stage: str | None = None,
    ) -> ReflectionSummary:
        counts = self._current_counts or {}
        phase_timings = dict(self._current_phase_timings or {})
        return ReflectionSummary(
            concepts_consolidated=counts.get("concepts_consolidated", 0),
            concepts_decayed=counts.get("concepts_decayed", 0),
            concepts_recalibrated=counts.get("concepts_recalibrated", 0),
            concepts_archived=counts.get("concepts_archived", 0),
            associations_updated=counts.get("associations_updated", 0),
            questions_generated=counts.get("questions_generated", 0),
            timestamp=_utc_now_iso(),
            phase_timings=phase_timings,
            gc_queue_remaining=counts.get("gc_queue_remaining", 0),
            concepts_graduated=counts.get("concepts_graduated", 0),
            concepts_discarded_quarantine=counts.get("concepts_discarded_quarantine", 0),
            concepts_time_matured=counts.get("concepts_time_matured", 0),
            concepts_assoc_propagated=counts.get("concepts_assoc_propagated", 0),
            concepts_auto_associated=counts.get("concepts_auto_associated", 0),
            concepts_currency_recomputed=counts.get("concepts_currency_recomputed", 0),
            concepts_promoted=counts.get("concepts_promoted", 0),
            aborted=True,
            abort_reason=abort_reason or self._abort_reason(),
            last_completed_step=self._last_completed_step,
            abort_stage=abort_stage,
        )

    def should_reflect(self) -> bool:
        """Check if it's time for reflection cycle."""
        if self.last_reflection is None:
            return True
        elapsed = (_utc_now() - _ensure_aware(self.last_reflection)).total_seconds()
        return elapsed >= CONSOLIDATION_INTERVAL

    def reflect(
        self,
        mode: str = "incremental",
        *,
        cancel_event: threading.Event | None = None,
        deadline_monotonic: float | None = None,
        long_step_min_remaining_seconds: float | None = None,
    ) -> ReflectionSummary:
        """Run reflection cycle."""
        _t0 = time.monotonic()
        self._begin_reflection_run(
            cancel_event=cancel_event,
            deadline_monotonic=deadline_monotonic,
            long_step_min_remaining_seconds=long_step_min_remaining_seconds,
        )
        try:
            if mode == "full":
                result = self._full_reflection()
            else:
                result = self._incremental_reflection()
        except ReflectionAborted as exc:
            logger.warning(
                "MAINT-040: Reflection aborted (%s) after %s",
                exc.reason,
                exc.stage,
            )
            result = self._build_abort_summary(abort_reason=exc.reason, abort_stage=exc.stage)
            metrics.record("reflect_budget_abort", 1, {"mode": mode, "reason": exc.reason, "stage": exc.stage})
        finally:
            self._end_reflection_run()
        _reflect_ms = round((time.monotonic() - _t0) * 1000, 1)
        # OBS-03: emit reflect latency to metrics DB
        metrics.record("reflect_latency_ms", _reflect_ms, {"mode": mode})
        # OBS-005: Extended reflection health metrics
        try:
            for _m, _v in [
                ("reflect_concepts_decayed", result.concepts_decayed),
                ("reflect_concepts_archived", result.concepts_archived),
                ("reflect_concepts_consolidated", result.concepts_consolidated),
                ("reflect_concepts_promoted", result.concepts_promoted),
                ("reflect_concepts_time_matured", result.concepts_time_matured),
                ("reflect_assoc_propagated", result.concepts_assoc_propagated),
            ]:
                metrics.record(_m, _v, {"mode": mode})
        except Exception:
            pass  # Best-effort
        return result

    def _incremental_reflection(self) -> ReflectionSummary:
        """Quick reflection on recent changes."""
        _t = {}  # DEBT-008: sub-step timings
        self._current_phase_timings = _t
        access_tracker.flush()
        self._check_abort("access_flush")

        # SCALE-004: Single-load for incremental (2 consumers: decay + recalibrate)
        _s = time.monotonic()
        _all_concepts = list_concepts_full()
        _t["concept_load"] = _elapsed_ms(_s)
        self._record_phase("concept_load", _t["concept_load"])
        logger.info(f"SCALE-004: Loaded {len(_all_concepts)} concepts once for incremental reflection")
        self._check_abort("concept_load")

        _s = time.monotonic()
        decayed = self._apply_decay(_all_concepts)
        _t["decay"] = _elapsed_ms(_s)
        self._record_phase("decay", _t["decay"])
        self._record_count("concepts_decayed", decayed)
        self._check_abort("decay")

        _s = time.monotonic()
        stale_cleaned = self._cleanup_stale_resolved()
        _t["stale_resolved_cleanup"] = _elapsed_ms(_s)
        self._record_phase("stale_resolved_cleanup", _t["stale_resolved_cleanup"])
        self._check_abort("stale_resolved_cleanup")

        _s = time.monotonic()
        recalibrated, factor_cvs = self._recalibrate_confidence(_all_concepts)
        _t["recalibrate"] = _elapsed_ms(_s)
        self._record_phase("recalibrate", _t["recalibrate"])
        self._record_count("concepts_recalibrated", recalibrated)
        self._check_abort("recalibrate")

        _s = time.monotonic()
        cleaned, gc_remaining = self._cleanup_low_confidence()  # DEBT-005: tuple
        _t["cleanup"] = _elapsed_ms(_s)
        self._record_phase("cleanup", _t["cleanup"])
        self._record_count("gc_queue_remaining", gc_remaining)
        self._check_abort("cleanup")

        _s = time.monotonic()
        forgot = self._apply_forgetting()
        _t["forgetting"] = _elapsed_ms(_s)
        self._record_phase("forgetting", _t["forgetting"])
        self._record_count("concepts_archived", forgot)
        self._check_abort("forgetting")

        if forgot > 0:
            try:
                retrieval_engine.build_index()
            except Exception as e:
                logger.warning(f"BUG-042: build_index() failed after forgetting (non-fatal): {e}")

        # Quarantine graduation sweep (RETRIEVAL-003)
        graduated_count, discarded_quarantine_count = self._run_graduation_sweep(_t)
        self._record_count("concepts_graduated", graduated_count)
        self._record_count("concepts_discarded_quarantine", discarded_quarantine_count)
        if graduated_count > 0 or discarded_quarantine_count > 0:
            try:
                retrieval_engine.build_index()
            except Exception as e:
                logger.warning(f"BUG-042: build_index() failed after graduation (non-fatal): {e}")

        # STABILITY-024: PROVISIONAL → ESTABLISHED promotion sweep
        _s = time.monotonic()
        promoted_count = self._promotion_sweep()
        _t["promotion_sweep"] = _elapsed_ms(_s)
        self._record_phase("promotion_sweep", _t["promotion_sweep"])
        self._record_count("concepts_promoted", promoted_count)
        self._check_abort("promotion_sweep")

        # Checkpoint TTL cleanup (non-fatal)
        # HEALTH-009: Lightweight auto-association for recently created orphans.
        # Incremental mode uses a smaller cap (20) vs full mode's 100.
        auto_associated = 0
        try:
            _s = time.monotonic()
            with _db() as aa_conn:  # noqa: F823
                orphan_rows = aa_conn.execute(
                    """SELECT c.id FROM concepts c
                       WHERE c.is_current = 1
                       AND c.created_at > datetime('now', '-1 day')
                       AND NOT EXISTS (
                           SELECT 1 FROM associations a
                           WHERE a.source = c.id OR a.target = c.id
                       )
                       ORDER BY c.created_at DESC LIMIT 20"""
                ).fetchall()
            orphan_ids = [r[0] for r in orphan_rows]
            if orphan_ids:
                auto_associated = self._associate_recently_changed(orphan_ids)
            # MONITOR-002: Persist auto_associated count for trend analysis
            if auto_associated > 0:
                try:
                    aa_conn.execute(
                        """INSERT INTO governance_events (event_type, concept_id, details, created_at)
                           VALUES ('auto_association_batch', 'system',
                                   json_object('count', ?, 'mode', 'incremental', 'orphan_count', ?), ?)""",
                        (auto_associated, len(orphan_ids), _utc_now_iso()),
                    )
                except Exception:
                    logger.debug("MONITOR-010: Non-fatal exception in reflection (suppressed)", exc_info=True)
            _t["auto_association"] = _elapsed_ms(_s)
            self._record_phase("auto_association", _t["auto_association"])
            self._record_count("associations_updated", auto_associated)
        except Exception as e:
            logger.debug(f"HEALTH-009: Incremental auto-association failed (non-fatal): {e}")
        self._check_abort("auto_association")

        checkpoints_cleaned = 0
        try:
            from app.storage import cleanup_expired_checkpoints

            _s = time.monotonic()
            checkpoints_cleaned = cleanup_expired_checkpoints()
            _t["checkpoint_cleanup"] = _elapsed_ms(_s)
            self._record_phase("checkpoint_cleanup", _t["checkpoint_cleanup"])
        except Exception as e:
            logger.warning(f"Checkpoint cleanup failed (non-fatal): {e}")
        self._check_abort("checkpoint_cleanup")

        total_ms = sum(_t.values())
        logger.info("DEBT-008: incremental reflection %.1fms — %s", total_ms, _t)

        # KA-005 / HEALTH-004: Post-reflection KA regression monitoring
        try:
            from app.storage import _db

            with _db() as mon_conn:
                general_row = mon_conn.execute(
                    "SELECT count(*) FROM concepts WHERE knowledge_area='general' AND is_current=1"
                ).fetchone()
                total_row = mon_conn.execute("SELECT count(*) FROM concepts WHERE is_current=1").fetchone()
                general_count = general_row[0] if general_row else 0
                total_count = total_row[0] if total_row else 1
                general_pct = general_count / total_count if total_count > 0 else 0
                if general_pct > 0.10:
                    logger.warning(
                        "HEALTH-004: %.1f%% concepts are 'general' (%d/%d) — KA corruption may be active",
                        general_pct * 100,
                        general_count,
                        total_count,
                    )
        except Exception as e:
            logger.debug(f"HEALTH-004 monitoring failed (non-fatal): {e}")

        # CONTRA-013: Consume accumulated GRAPH_CONTRADICTION_SIGNAL events
        contradiction_signals_consumed = 0
        try:
            _s = time.monotonic()
            from app.cognitive.contradiction import consume_graph_contradiction_signals

            result = consume_graph_contradiction_signals(batch_size=200)
            contradiction_signals_consumed = result.get("processed", 0)
            _t["contradiction_signal_consume"] = _elapsed_ms(_s)
            self._record_phase("contradiction_signal_consume", _t["contradiction_signal_consume"])
            if contradiction_signals_consumed > 0:
                logger.info("CONTRA-013: Consumed %d graph contradiction signals", contradiction_signals_consumed)
        except Exception as e:
            logger.debug(f"CONTRA-013: Signal consumption failed (non-fatal): {e}")
        self._check_abort("contradiction_signal_consume")
        # RETRIEVAL-015: Full-population currency recompute
        _s = time.monotonic()
        currency_updated = self._recompute_currency()
        _t["currency_recompute"] = _elapsed_ms(_s)
        self._record_phase("currency_recompute", _t["currency_recompute"])
        self._record_count("concepts_currency_recomputed", currency_updated)
        self._check_abort("currency_recompute")

        # Refresh topic_activity_cache so TA scores use content_updated_at
        # See: CURRENCY_AR_ANCHOR_DESIGN_v1.md §Change 3
        _s = time.monotonic()
        self._refresh_topic_activity_cache()
        _t["topic_activity_refresh"] = _elapsed_ms(_s)
        self._record_phase("topic_activity_refresh", _t["topic_activity_refresh"])
        self._check_abort("topic_activity_refresh")

        self.last_reflection = _utc_now()
        return ReflectionSummary(
            concepts_consolidated=0,
            concepts_decayed=decayed,
            concepts_recalibrated=recalibrated,
            concepts_archived=forgot,
            associations_updated=auto_associated,
            questions_generated=0,
            timestamp=_utc_now_iso(),
            phase_timings=_t,
            gc_queue_remaining=gc_remaining,
            concepts_graduated=graduated_count,
            concepts_discarded_quarantine=discarded_quarantine_count,
            concepts_promoted=promoted_count,
            concepts_currency_recomputed=currency_updated,
            evidence_cv_composite=factor_cvs.get("composite"),
            evidence_cv_reliability=factor_cvs.get("reliability"),
            evidence_cv_directness=factor_cvs.get("directness"),
            evidence_cv_consistency=factor_cvs.get("consistency"),
            evidence_cv_corroboration=factor_cvs.get("corroboration"),
            evidence_cv_recency=factor_cvs.get("recency"),
        )

    def _recompute_currency(self) -> int:
        """RETRIEVAL-015: Recompute currency scores for all active concepts.

        Called once per reflection cycle to keep currency scores fresh.
        Uses batch_compute_currency with concept_ids=None for full population.
        """
        try:
            with _db_immediate() as conn:
                updated = batch_compute_currency(conn, concept_ids=None)
                if updated > 0:
                    logger.info(
                        "RETRIEVAL-015: Reflection currency recompute — %d concepts updated",
                        updated,
                    )
                return updated
        except Exception as e:
            logger.warning("RETRIEVAL-015: Currency recompute failed: %s", e)
            return 0

    def _refresh_topic_activity_cache(self) -> int:
        """Refresh topic_activity_cache from concepts table.

        Uses content_updated_at (preferred) or updated_at as the activity
        timestamp, consistent with the AR anchor switch.
        See: CURRENCY_AR_ANCHOR_DESIGN_v1.md §Change 3
        """
        try:
            with _db_immediate() as conn:
                cutoff = (_utc_now() - timedelta(days=30)).isoformat()
                conn.execute("DELETE FROM topic_activity_cache")
                conn.execute(
                    """INSERT INTO topic_activity_cache
                           (knowledge_area, activity_count_30d, last_activity_at)
                       SELECT knowledge_area,
                              COUNT(*),
                              MAX(COALESCE(content_updated_at, updated_at))
                       FROM concepts
                       WHERE status = 'active'
                         AND COALESCE(content_updated_at, updated_at) > ?
                       GROUP BY knowledge_area""",
                    (cutoff,),
                )
                refreshed = conn.execute("SELECT changes()").fetchone()[0]
                if refreshed > 0:
                    logger.info(
                        "Topic activity cache refreshed — %d knowledge areas",
                        refreshed,
                    )
                return refreshed
        except Exception as e:
            logger.warning("Topic activity cache refresh failed: %s", e)
            return 0

    def _associate_recently_changed(self, candidate_ids: list[str]) -> int:
        """HEALTH-009: Auto-associate recently matured/strengthened concepts.

        Runs auto_associate_single() on each candidate, capped at
        ASSOC_REFLECTION_MAX_PER_CYCLE to keep reflection under 15s total.
        """
        from app.cognitive.association import auto_associate_single
        from app.core.config import ASSOC_REFLECTION_MAX_PER_CYCLE
        from app.core.models import AutoAssociateSingleRequest

        associated = 0
        cap = ASSOC_REFLECTION_MAX_PER_CYCLE

        # F5-2 amendment: candidate ordering is list-order (maturation then strengthening).
        # Not optimal (orphans-first would be better) but acceptable for v1.
        # If connectivity growth stalls after 10 cycles, add HEALTH-010 for explicit ordering.
        for concept_id in candidate_ids[:cap]:
            try:
                request = AutoAssociateSingleRequest()
                result = auto_associate_single(concept_id, request)
                if result and result.edges_created > 0:
                    associated += result.edges_created
            except Exception:
                continue  # Non-fatal, skip this concept

        logger.info(
            "HEALTH-009: %d new edges from %d candidates (cap=%d)", associated, min(len(candidate_ids), cap), cap
        )
        return associated

    def _full_reflection(self) -> ReflectionSummary:
        """Deep reflection with consolidation."""
        _t = {}  # DEBT-008: sub-step timings
        self._current_phase_timings = _t
        access_tracker.flush()
        self._check_abort("access_flush")

        # SCALE-004: Single-load for full (5 consumers: decay, strengthen, maturation, assoc_prop, recalibrate)
        _s = time.monotonic()
        _all_concepts = list_concepts_full()
        _t["concept_load"] = _elapsed_ms(_s)
        self._record_phase("concept_load", _t["concept_load"])
        logger.info(f"SCALE-004: Loaded {len(_all_concepts)} concepts once for full reflection")
        self._check_abort("concept_load")

        _s = time.monotonic()
        decayed = self._apply_decay(_all_concepts)
        _t["decay"] = _elapsed_ms(_s)
        self._record_phase("decay", _t["decay"])
        self._record_count("concepts_decayed", decayed)
        self._check_abort("decay")

        _s = time.monotonic()
        stale_cleaned = self._cleanup_stale_resolved()
        _t["stale_resolved_cleanup"] = _elapsed_ms(_s)
        self._record_phase("stale_resolved_cleanup", _t["stale_resolved_cleanup"])
        self._check_abort("stale_resolved_cleanup")

        # DATA-042: Factual drift scan (report-only)
        _s = time.monotonic()
        try:
            from app.cognitive.rename_registry import scan_for_drift

            with _db() as drift_conn:
                drift_results = scan_for_drift(drift_conn)
            if drift_results:
                logger.warning(
                    "DATA-042: %d concepts with factual drift detected (top: %s → %s in %s)",
                    len(drift_results),
                    drift_results[0].old_term,
                    drift_results[0].new_term,
                    drift_results[0].concept_id,
                )
        except Exception as e:
            logger.debug(f"DATA-042: Drift scan failed (non-fatal): {e}")
            drift_results = []
        _t["drift_scan"] = _elapsed_ms(_s)
        self._record_phase("drift_scan", _t["drift_scan"])
        self._check_abort("drift_scan")

        _s = time.monotonic()
        strengthened, strengthened_ids, stability_guarded = self._strengthen_accessed(_all_concepts)
        _t["strengthen"] = _elapsed_ms(_s)
        self._record_phase("strengthen", _t["strengthen"])
        self._check_abort("strengthen")

        _s = time.monotonic()
        recalibrated, factor_cvs = self._recalibrate_confidence(_all_concepts)
        _t["recalibrate"] = _elapsed_ms(_s)
        self._record_phase("recalibrate", _t["recalibrate"])
        self._record_count("concepts_recalibrated", recalibrated)
        self._check_abort("recalibrate")

        # STABILITY-001 Components C + D: passive maturation after recalibration, before merge
        _s = time.monotonic()
        time_matured, matured_ids = self._time_based_maturation(_all_concepts)
        _t["time_maturation"] = _elapsed_ms(_s)
        self._record_phase("time_maturation", _t["time_maturation"])
        self._record_count("concepts_time_matured", time_matured)
        self._check_abort("time_maturation")

        _s = time.monotonic()
        assoc_propagated = self._propagate_association_confidence(_all_concepts)
        _t["assoc_propagation"] = _elapsed_ms(_s)
        self._record_phase("assoc_propagation", _t["assoc_propagation"])
        self._record_count("concepts_assoc_propagated", assoc_propagated)
        self._check_abort("assoc_propagation")

        # HEALTH-009: Auto-associate recently changed concepts
        _s = time.monotonic()
        self._check_long_step_budget("auto_association_preflight")
        assoc_candidates = matured_ids + strengthened_ids
        auto_associated = self._associate_recently_changed(assoc_candidates)
        # MONITOR-002: Persist auto_associated count for trend analysis (full mode)
        if auto_associated > 0:
            try:
                from app.storage import _db

                with _db() as _aa_conn:
                    _aa_conn.execute(
                        """INSERT INTO governance_events (event_type, concept_id, details, created_at)
                           VALUES ('auto_association_batch', 'system',
                                   json_object('count', ?, 'mode', 'full', 'candidate_count', ?), ?)""",
                        (auto_associated, len(assoc_candidates), _utc_now_iso()),
                    )
            except Exception:
                logger.debug("MONITOR-010: Non-fatal exception in reflection (suppressed)", exc_info=True)
        _t["auto_association"] = _elapsed_ms(_s)
        self._record_phase("auto_association", _t["auto_association"])
        self._record_count("concepts_auto_associated", auto_associated)
        self._check_abort("auto_association")

        _s = time.monotonic()
        self._check_long_step_budget("merge_preflight")
        merged = self._merge_duplicates()
        _t["merge"] = _elapsed_ms(_s)
        self._record_phase("merge", _t["merge"])
        self._record_count("concepts_consolidated", merged)
        self._check_abort("merge")

        _s = time.monotonic()
        cleaned, gc_remaining = self._cleanup_low_confidence()  # DEBT-005: tuple
        _t["cleanup"] = _elapsed_ms(_s)
        self._record_phase("cleanup", _t["cleanup"])
        self._record_count("gc_queue_remaining", gc_remaining)
        self._check_abort("cleanup")

        # REFLECT-004: Recompute salience BEFORE forgetting so it uses real scores
        _s = time.monotonic()
        try:
            from app.retrieval.salience import recompute_salience

            sal_result = recompute_salience()
            logger.info(f"REFLECT-004: Salience recomputed for {sal_result.get('recomputed', 0)} concepts")
        except Exception as e:
            logger.warning(f"REFLECT-004: Salience recomputation failed (non-fatal): {e}")
        _t["salience"] = _elapsed_ms(_s)
        self._record_phase("salience", _t["salience"])
        self._check_abort("salience")

        _s = time.monotonic()
        forgot = self._apply_forgetting()
        _t["forgetting"] = _elapsed_ms(_s)
        self._record_phase("forgetting", _t["forgetting"])
        self._record_count("concepts_archived", forgot)
        self._check_abort("forgetting")

        # Quarantine graduation sweep (RETRIEVAL-003)
        graduated_count, discarded_quarantine_count = self._run_graduation_sweep(_t)
        self._record_count("concepts_graduated", graduated_count)
        self._record_count("concepts_discarded_quarantine", discarded_quarantine_count)
        self._check_abort("quarantine_graduation")

        # MATURITY-003 Part B: Quarantine release sweep (BEFORE promotion)
        _s = time.monotonic()
        quarantine_released_count = self._quarantine_release_sweep()
        _t["quarantine_release"] = _elapsed_ms(_s)
        self._record_phase("quarantine_release", _t["quarantine_release"])
        self._check_abort("quarantine_release")

        # STABILITY-024: PROVISIONAL → ESTABLISHED promotion sweep
        _s = time.monotonic()
        promoted_count = self._promotion_sweep()
        _t["promotion_sweep"] = _elapsed_ms(_s)
        self._record_phase("promotion_sweep", _t["promotion_sweep"])
        self._record_count("concepts_promoted", promoted_count)
        self._check_abort("promotion_sweep")

        # MATURITY-003 Phase A5: Evidence backfill for stuck concepts
        _s = time.monotonic()
        evidence_backfilled_count = self._evidence_backfill_sweep()
        _t["evidence_backfill"] = _elapsed_ms(_s)
        self._record_phase("evidence_backfill", _t["evidence_backfill"])
        self._check_abort("evidence_backfill")

        _s = time.monotonic()
        self._check_long_step_budget("associations_preflight")
        associations = self._update_associations()
        _t["associations"] = _elapsed_ms(_s)
        self._record_phase("associations", _t["associations"])
        self._record_count("associations_updated", associations)
        self._check_abort("associations")

        _s = time.monotonic()
        self._check_long_step_budget("index_build_preflight")
        try:
            retrieval_engine.build_index()
        except Exception as e:
            logger.warning(f"BUG-042: build_index() failed in full reflection (non-fatal): {e}")
        _t["index_build"] = _elapsed_ms(_s)
        self._record_phase("index_build", _t["index_build"])
        self._check_abort("index_build")

        # SelfModel update: snapshot concepts once, pass to generator
        _s = time.monotonic()
        self._check_long_step_budget("self_model_load_preflight")
        all_concepts = []
        for cid in list_concepts():
            c = load_concept(cid, track_access=False)
            if c:
                all_concepts.append(c)
            self._check_abort_every(len(all_concepts), 100, "self_model_load")
        self._check_long_step_budget("self_model_generate_preflight")
        _get_self_model_manager().generate(all_concepts)
        _t["self_model"] = _elapsed_ms(_s)
        self._record_phase("self_model", _t["self_model"])
        self._check_abort("self_model")

        total_ms = sum(_t.values())
        logger.info("DEBT-008: full reflection %.1fms — %s", total_ms, _t)

        # KA-005 / HEALTH-004: Post-reflection KA regression monitoring
        try:
            from app.storage import _db

            with _db() as mon_conn:
                general_row = mon_conn.execute(
                    "SELECT count(*) FROM concepts WHERE knowledge_area='general' AND is_current=1"
                ).fetchone()
                total_row = mon_conn.execute("SELECT count(*) FROM concepts WHERE is_current=1").fetchone()
                general_count = general_row[0] if general_row else 0
                total_count = total_row[0] if total_row else 1
                general_pct = general_count / total_count if total_count > 0 else 0
                if general_pct > 0.10:
                    logger.warning(
                        "HEALTH-004: %.1f%% concepts are 'general' (%d/%d) — KA corruption may be active",
                        general_pct * 100,
                        general_count,
                        total_count,
                    )
        except Exception as e:
            logger.debug(f"HEALTH-004 monitoring failed (non-fatal): {e}")

        self.last_reflection = _utc_now()
        return ReflectionSummary(
            concepts_consolidated=merged,
            concepts_decayed=decayed,
            concepts_recalibrated=recalibrated,
            concepts_archived=forgot,
            associations_updated=associations,
            questions_generated=0,
            timestamp=_utc_now_iso(),
            phase_timings=_t,
            gc_queue_remaining=gc_remaining,
            concepts_graduated=graduated_count,
            concepts_discarded_quarantine=discarded_quarantine_count,
            concepts_promoted=promoted_count,
            concepts_time_matured=time_matured,
            concepts_assoc_propagated=assoc_propagated,
            concepts_auto_associated=auto_associated,
            evidence_cv_composite=factor_cvs.get("composite"),
            evidence_cv_reliability=factor_cvs.get("reliability"),
            evidence_cv_directness=factor_cvs.get("directness"),
            evidence_cv_consistency=factor_cvs.get("consistency"),
            evidence_cv_corroboration=factor_cvs.get("corroboration"),
            evidence_cv_recency=factor_cvs.get("recency"),
        )

    def _apply_decay(self, _preloaded: list | None = None) -> int:
        """Apply time-based confidence decay (Amendment 5: with validation).

        REFLECT-022: Batch SQL pattern — compute in memory, write once.
        """
        from app.core.config import PSIS_QUARANTINE_CONFIDENCE_CAP, PSIS_QUARANTINE_EVIDENCE_MARKER
        decayed_count = 0
        batch_updates = []

        all_concepts = _preloaded if _preloaded is not None else list_concepts_full()
        for idx, concept in enumerate(all_concepts, start=1):
            self._check_abort_every(idx, 100, "decay")
            if not self._validate_concept(concept):
                continue
            if concept.last_accessed:
                last_access = datetime.fromisoformat(concept.last_accessed)
                days_since = (_utc_now() - _ensure_aware(last_access)).days
            else:
                created = datetime.fromisoformat(concept.created_at)
                days_since = (_utc_now() - _ensure_aware(created)).days
            if days_since > 30:
                months = days_since / 30
                decay_factor = (1 - DECAY_RATE) ** months
                new_confidence = concept.confidence * decay_factor
                if new_confidence != concept.confidence:
                    new_confidence = max(0.0, new_confidence)
                    if PSIS_QUARANTINE_EVIDENCE_MARKER in (concept.evidence or []):
                        new_confidence = min(new_confidence, PSIS_QUARANTINE_CONFIDENCE_CAP)
                    batch_updates.append((new_confidence, None, concept.id))
                    concept.confidence = new_confidence  # SCALE-004: maintain ordering consistency for shared list
                    decayed_count += 1

        written = _batch_update_confidence_stability(batch_updates)
        if written == 0 and batch_updates:
            logger.warning("REFLECT-022: Batch decay failed, falling back to per-concept save")
            for conf, _, cid in batch_updates:
                try:
                    c = load_concept(cid, track_access=False)
                    if c:
                        original_ka = c.metadata.get("knowledge_area") if c.metadata else None
                        c.confidence = conf
                        if original_ka:
                            c.knowledge_area = original_ka
                            if c.metadata:
                                c.metadata["knowledge_area"] = original_ka
                        save_concept(c)
                except Exception as e:
                    logger.warning("REFLECT-022: Fallback save failed for %s: %s", cid, e)
        return decayed_count

    def _strengthen_accessed(self, _preloaded: list | None = None) -> tuple[int, list[str], int]:
        """Boost confidence of frequently accessed concepts.

        REFLECT-022: Batch SQL pattern — compute in memory, write once.
        Returns (strengthened_count, list_of_strengthened_concept_ids, guarded_count)
        """
        from app.core.config import PSIS_QUARANTINE_CONFIDENCE_CAP, PSIS_QUARANTINE_EVIDENCE_MARKER
        strengthened = 0
        guarded = 0
        strengthened_ids: list[str] = []
        batch_updates = []

        all_concepts = _preloaded if _preloaded is not None else list_concepts_full()
        for idx, concept in enumerate(all_concepts, start=1):
            self._check_abort_every(idx, 100, "strengthen")
            if concept.access_count > 2:
                boost = min(0.05, concept.access_count * 0.003)
                new_confidence = min(1.0, concept.confidence + boost)
                if new_confidence != concept.confidence:
                    if PSIS_QUARANTINE_EVIDENCE_MARKER in (concept.evidence or []):
                        new_confidence = min(new_confidence, PSIS_QUARANTINE_CONFIDENCE_CAP)
                    new_stability = concept.stability
                    if concept.stability < self.TIME_MATURATION_MAX_STABILITY:
                        new_stability = min(self.TIME_MATURATION_MAX_STABILITY,
                                            concept.stability + 0.05)
                    else:
                        guarded += 1
                    batch_updates.append((new_confidence, new_stability, concept.id))
                    concept.confidence = new_confidence  # SCALE-004: ordering consistency
                    if new_stability is not None:
                        concept.stability = new_stability  # SCALE-004: ordering consistency
                    strengthened += 1
                    strengthened_ids.append(concept.id)

        written = _batch_update_confidence_stability(batch_updates)
        if written == 0 and batch_updates:
            logger.warning("REFLECT-022: Batch strengthen failed, falling back to per-concept save")
            for conf, stab, cid in batch_updates:
                try:
                    c = load_concept(cid, track_access=False)
                    if c:
                        original_ka = c.metadata.get("knowledge_area") if c.metadata else None
                        c.confidence = conf
                        if stab is not None:
                            c.stability = stab
                        if original_ka:
                            c.knowledge_area = original_ka
                            if c.metadata:
                                c.metadata["knowledge_area"] = original_ka
                        save_concept(c)
                except Exception as e:
                    logger.warning("REFLECT-022: Fallback save failed for %s: %s", cid, e)
        if guarded > 0:
            logger.info(f"STABILITY-SAT guard: {guarded} concepts had stability boost blocked (>= {self.TIME_MATURATION_MAX_STABILITY})")
        return strengthened, strengthened_ids, guarded

    # --- STABILITY-001 Constants ---
    TIME_MATURATION_AGE_DAYS = 14
    TIME_MATURATION_STABILITY_BOOST = 0.05
    TIME_MATURATION_CONFIDENCE_BOOST = 0.02
    TIME_MATURATION_MAX_STABILITY = 0.8
    TIME_MATURATION_MAX_PER_CYCLE = 200
    ASSOC_PROPAGATION_MIN_NEIGHBOR_CONFIDENCE = 0.5
    ASSOC_PROPAGATION_BOOST = 0.02
    ASSOC_PROPAGATION_MAX_BOOST = 0.08
    ASSOC_PROPAGATION_MAX_PER_CYCLE = 150

    def _time_based_maturation(self, _preloaded: list | None = None) -> tuple[int, list[str]]:
        """STABILITY-001 Component C: Passively mature concepts based on age and survival.

        REFLECT-022: Batch SQL pattern.
        Eligibility: active, age >= 14d, stability < 0.8, currency ACTIVE, not QUARANTINED.
        Boost: stability += 0.05, confidence += 0.02 (with association density multiplier).
        Cap: stability maxes at 0.8 (time alone can't make fully stable).
        Returns (count, list_of_matured_concept_ids) for HEALTH-009 association.
        """
        from app.core.config import PSIS_QUARANTINE_CONFIDENCE_CAP, PSIS_QUARANTINE_EVIDENCE_MARKER
        matured = 0
        matured_ids: list[str] = []
        batch_updates = []
        cutoff = (_utc_now() - timedelta(days=self.TIME_MATURATION_AGE_DAYS)).isoformat()

        all_concepts = _preloaded if _preloaded is not None else list_concepts_full()
        for idx, concept in enumerate(all_concepts, start=1):
            self._check_abort_every(idx, 100, "time_maturation")
            if matured >= self.TIME_MATURATION_MAX_PER_CYCLE:
                break
            if concept.stability >= self.TIME_MATURATION_MAX_STABILITY:
                continue
            if concept.created_at > cutoff:
                continue
            currency = getattr(concept, "currency_status", "ACTIVE")
            if currency in ("STALE", "SUPERSEDED"):
                continue
            maturity = getattr(concept, "maturity", "ESTABLISHED")
            if maturity == "QUARANTINED":
                continue

            assoc_count = len(concept.associations) if concept.associations else 0
            if assoc_count >= 6:
                multiplier = 2.0
            elif assoc_count >= 3:
                multiplier = 1.5
            else:
                multiplier = 1.0

            stability_boost = self.TIME_MATURATION_STABILITY_BOOST * multiplier
            confidence_boost = self.TIME_MATURATION_CONFIDENCE_BOOST * multiplier
            new_stability = min(self.TIME_MATURATION_MAX_STABILITY,
                                concept.stability + stability_boost)
            new_confidence = min(1.0, concept.confidence + confidence_boost)

            if new_stability != concept.stability or new_confidence != concept.confidence:
                if PSIS_QUARANTINE_EVIDENCE_MARKER in (concept.evidence or []):
                    new_confidence = min(new_confidence, PSIS_QUARANTINE_CONFIDENCE_CAP)
                batch_updates.append((new_confidence, new_stability, concept.id))
                concept.confidence = new_confidence  # SCALE-004: ordering consistency
                concept.stability = new_stability  # SCALE-004: ordering consistency
                matured += 1
                matured_ids.append(concept.id)

        written = _batch_update_confidence_stability(batch_updates)
        if written == 0 and batch_updates:
            logger.warning("REFLECT-022: Batch maturation failed, falling back")
            for conf, stab, cid in batch_updates:
                try:
                    c = load_concept(cid, track_access=False)
                    if c:
                        original_ka = c.metadata.get("knowledge_area") if c.metadata else None
                        c.confidence = conf
                        c.stability = stab
                        if original_ka:
                            c.knowledge_area = original_ka
                            if c.metadata:
                                c.metadata["knowledge_area"] = original_ka
                        save_concept(c)
                except Exception as e:
                    logger.warning("REFLECT-022: Fallback save failed for %s: %s", cid, e)

        logger.info("STABILITY-001 C: Time maturation boosted %d concepts", matured)
        return matured, matured_ids

    def _propagate_association_confidence(self, _preloaded: list | None = None) -> int:
        """STABILITY-001 Component D: Boost confidence of well-connected concepts.

        REFLECT-022: Batch SQL pattern + single list_concepts_full() call.
        For each concept, count neighbors with confidence >= 0.5.
        Apply per-neighbor boost (0.02), capped at 0.08 total per concept.
        """
        from app.core.config import PSIS_QUARANTINE_CONFIDENCE_CAP, PSIS_QUARANTINE_EVIDENCE_MARKER
        propagated = 0
        batch_updates = []

        all_concepts = _preloaded if _preloaded is not None else list_concepts_full()
        confidence_map = {c.id: c.confidence for c in all_concepts}

        for idx, concept in enumerate(all_concepts, start=1):
            self._check_abort_every(idx, 100, "assoc_propagation")
            if propagated >= self.ASSOC_PROPAGATION_MAX_PER_CYCLE:
                break
            if not concept.associations:
                continue

            eligible_neighbors = sum(
                1 for assoc_id in concept.associations
                if confidence_map.get(assoc_id, 0.0) >= self.ASSOC_PROPAGATION_MIN_NEIGHBOR_CONFIDENCE
            )
            if eligible_neighbors == 0:
                continue

            boost = min(self.ASSOC_PROPAGATION_MAX_BOOST,
                        eligible_neighbors * self.ASSOC_PROPAGATION_BOOST)
            new_confidence = min(1.0, concept.confidence + boost)
            if new_confidence != concept.confidence:
                if PSIS_QUARANTINE_EVIDENCE_MARKER in (concept.evidence or []):
                    new_confidence = min(new_confidence, PSIS_QUARANTINE_CONFIDENCE_CAP)
                batch_updates.append((new_confidence, None, concept.id))
                concept.confidence = new_confidence  # SCALE-004: ordering consistency
                propagated += 1

        written = _batch_update_confidence_stability(batch_updates)
        if written == 0 and batch_updates:
            logger.warning("REFLECT-022: Batch propagation failed, falling back")
            for conf, _, cid in batch_updates:
                try:
                    c = load_concept(cid, track_access=False)
                    if c:
                        original_ka = c.metadata.get("knowledge_area") if c.metadata else None
                        c.confidence = conf
                        if original_ka:
                            c.knowledge_area = original_ka
                            if c.metadata:
                                c.metadata["knowledge_area"] = original_ka
                        save_concept(c)
                except Exception as e:
                    logger.warning("REFLECT-022: Fallback save failed for %s: %s", cid, e)

        logger.info("STABILITY-001 D: Association propagation boosted %d concepts", propagated)
        return propagated

    def _merge_duplicates(self) -> int:
        """Find and merge similar concepts via TF-IDF deduplication pipeline.

        Merge eligibility (ALL must be true):
          - TF-IDF cosine similarity >= 0.92 (dedup merge threshold)
          - Same knowledge_area
          - Neither concept has concept_type == "goal" (Goal Coexistence Rule)
          - Pair not already processed this cycle

        Merge execution:
          1. Schema validation (guaranteed by Pydantic)
          2. Evidence merge — deduplicate by content
          3. Similarity confirmed (>= 0.92)
          4. Confidence = weighted_mean using E(c) weights
             Cannot increase beyond max(a, b) without new evidence
          5. Stability = max(0.0, min(a, b) - 0.05)  (scope expanded)
          6. New version of survivor; loser archived

        Survivor selection: higher-confidence concept survives.
        Age (created_at) is tiebreaker only.

        Idempotency: second run finds loser archived → no action.
        Safety: MAX_MERGES_PER_CYCLE = 10.
        """
        merged = 0
        processed_pairs = set()  # Deduplicate (A,B) and (B,A)

        # REFLECT-021: SQL-narrowed merge scan — only check recently-modified concepts.
        # Merge is symmetric: if concept A (new) is scanned, old concept B appears in
        # top-5 embedding results. Full scan runs weekly as safety net (A10).
        _last_ref = getattr(self, "last_reflection", None)
        _last_full = getattr(self, "_last_full_merge_scan", None)
        _use_full_scan = False
        if _last_ref is None:
            _use_full_scan = True  # Cold start
        elif _last_full is None:
            _use_full_scan = True  # No record of previous full scan
        else:
            days_since_full = (_utc_now() - _ensure_aware(_last_full)).total_seconds() / 86400
            if days_since_full >= 7:
                _use_full_scan = True  # A10: Weekly full scan

        if _use_full_scan:
            all_ids = list_concepts()
            self._last_full_merge_scan = _utc_now()
            logger.info("REFLECT-021: Full merge scan (%d concepts, cold_start=%s)", len(all_ids), _last_ref is None)
        else:
            cutoff = _last_ref.isoformat() if _last_ref else _utc_now_iso()
            all_ids = list_concepts_modified_since(cutoff)
            logger.info("REFLECT-021: Narrowed merge scan (%d concepts since %s)", len(all_ids), cutoff[:19])

        # DEBT-011: Move embedding import outside merge loop (~300 iterations)
        try:
            from app.storage.embedding import embedding_engine

            _use_embeddings = True
        except ImportError:
            _use_embeddings = False

        for idx, concept_id in enumerate(all_ids, start=1):
            self._check_abort_every(idx, 25, "merge")
            if merged >= MAX_MERGES_PER_CYCLE:
                break

            concept = load_concept(concept_id, track_access=False)
            if not concept:
                continue

            # Goal Coexistence Rule: goals never merge
            meta = concept.metadata or {}
            if meta.get("concept_type") == "goal":
                continue

            # REFLECT-006: Use embedding similarity for merge detection (TF-IDF fallback).
            # TF-IDF cosine maxes ~0.52 for paraphrases; embeddings reach 0.85-0.90.
            # DEBT-012: Refactored from exception control flow to clean if/else.
            # A3: Added empty-result fallback — embedding search can return [] without
            # throwing, which would silently skip merge for this concept.
            if _use_embeddings:
                try:
                    matches = embedding_engine.search(concept.summary, top_k=5)
                    if not matches and concept.summary:
                        matches = retrieval_engine.index.search(concept.summary, top_k=5)
                except Exception:
                    matches = retrieval_engine.index.search(concept.summary, top_k=5)
            else:
                matches = retrieval_engine.index.search(concept.summary, top_k=5)

            for match_id, similarity in matches:
                if similarity < MERGE_SIMILARITY_THRESHOLD:
                    continue
                if match_id == concept_id:
                    continue
                if merged >= MAX_MERGES_PER_CYCLE:
                    break

                # Deduplicate pairs — (A,B) and (B,A) are the same merge
                pair_key = tuple(sorted([concept_id, match_id]))
                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)

                match_concept = load_concept(match_id, track_access=False)
                if not match_concept:
                    continue

                # Goal Coexistence Rule
                match_meta = match_concept.metadata or {}
                if match_meta.get("concept_type") == "goal":
                    continue

                # Same knowledge_area required
                concept_area = meta.get("knowledge_area", "")
                match_area = match_meta.get("knowledge_area", "")
                if concept_area != match_area:
                    continue

                # --- Merge execution ---

                # Survivor selection: higher confidence wins; age is tiebreaker
                if concept.confidence > match_concept.confidence:
                    survivor, loser = concept, match_concept
                elif match_concept.confidence > concept.confidence:
                    survivor, loser = match_concept, concept
                else:
                    # Tiebreaker: older concept survives (more established)
                    if concept.created_at <= match_concept.created_at:
                        survivor, loser = concept, match_concept
                    else:
                        survivor, loser = match_concept, concept

                # Evidence merge — deduplicate by content
                existing_evidence_set = set()
                merged_evidence = []
                for ev in survivor.evidence:
                    if isinstance(ev, str):
                        key = ev
                    elif isinstance(ev, dict):
                        key = ev.get("content", str(ev))
                    else:
                        key = getattr(ev, "content", str(ev))
                    if key not in existing_evidence_set:
                        existing_evidence_set.add(key)
                        merged_evidence.append(ev)

                for ev in loser.evidence:
                    if isinstance(ev, str):
                        key = ev
                    elif isinstance(ev, dict):
                        key = ev.get("content", str(ev))
                    else:
                        key = getattr(ev, "content", str(ev))
                    if key not in existing_evidence_set:
                        existing_evidence_set.add(key)
                        merged_evidence.append(ev)

                survivor.evidence = merged_evidence

                # Merge hypotheses — deduplicate by statement
                existing_hyp_statements = {
                    h.statement if hasattr(h, "statement") else str(h) for h in survivor.hypotheses
                }
                for h in loser.hypotheses:
                    stmt = h.statement if hasattr(h, "statement") else str(h)
                    if stmt not in existing_hyp_statements:
                        survivor.hypotheses.append(h)
                        existing_hyp_statements.add(stmt)

                # Merge signals — union
                existing_signals = set(survivor.signals)
                for sig in loser.signals:
                    if sig not in existing_signals:
                        survivor.signals.append(sig)
                        existing_signals.add(sig)

                # Confidence update — weighted by evidence strength E(c)
                e_survivor = _compute_evidence_strength(survivor)
                e_loser = _compute_evidence_strength(loser)
                total_weight = e_survivor + e_loser

                if total_weight > 0:
                    new_conf = (survivor.confidence * e_survivor + loser.confidence * e_loser) / total_weight
                else:
                    # Both have zero evidence strength — simple average
                    new_conf = (survivor.confidence + loser.confidence) / 2

                # Cannot increase beyond max without genuinely new evidence
                max_conf = max(survivor.confidence, loser.confidence)
                survivor.confidence = min(new_conf, max_conf)

                # Stability floor — scope expanded by merge
                survivor.stability = max(0.0, min(survivor.stability, loser.stability) - 0.05)

                # Record merge traceability through the DATA-070 lifecycle gateway.
                try:
                    with _db() as conn:
                        # DATA-048: Decay confidence and set is_current=0 to prevent desync
                        _decayed_conf = max(0.0, (loser.confidence if loser.confidence is not None else 0.5) - 0.3)
                        apply_lifecycle_transition_conn(
                            conn,
                            loser.id,
                            "supersede",
                            superseded_by=survivor.id,
                            reason="reflection duplicate merge",
                            confidence=_decayed_conf,
                        )
                except Exception as e:
                    logger.warning(f"Merge supersession record failed (non-fatal): {e}")

                # KA-005: Guard must protect survivor.knowledge_area (what save_concept reads)
                survivor_ka = survivor.metadata.get("knowledge_area") if survivor.metadata else None
                if survivor_ka:
                    survivor.knowledge_area = survivor_ka
                    if survivor.metadata:
                        survivor.metadata["knowledge_area"] = survivor_ka
                save_concept(survivor)
                merged += 1

                # REFLECT-006 / GA-007: Merge audit log for traceability
                logger.info(
                    f"MERGE_AUDIT: {loser.id} -> {survivor.id} | "
                    f"similarity={similarity:.3f} | "
                    f"conf={survivor.confidence:.3f} | "
                    f"stability={survivor.stability:.3f} | "
                    f"evidence={len(survivor.evidence)} | "
                    f"survivor_summary={survivor.summary[:80]!r} | "
                    f"loser_summary={loser.summary[:80]!r}"
                )

        return merged

    def _recalibrate_confidence(self, _preloaded: list | None = None) -> tuple[int, dict[str, float | None]]:
        """Recalibrate concept confidence toward evidence-justified levels.

        REFLECT-022: Batch SQL pattern — compute in memory, write once.
        RETRIEVAL-080: Blends utility score into recalibration target when
        sufficient feedback samples exist. Also applies type-differentiated
        confidence caps so proven L3 concepts can escape the gravity well.

        E(c) = weighted_mean(E(e_i)) is the evidence-justified confidence.
        Rules:
          - Target = 0.6 × E(c) + 0.4 × utility (when utility_samples ≥ 5)
          - Target = E(c) (fallback when insufficient utility data)
          - If confidence > target: decay toward target with RECALIBRATION_DAMPING
          - If confidence < target: boost toward target with same damping
          - If concept has no evidence: skip
          - Type-differentiated caps: L3+utility>0.6 → 0.85, L3 → 0.7, L1 → 0.6
          - EUNOMIA-038: warmup guard removed; base damping is now 0.10 and
            maintenance runs 1x/day, so convergence is already slower than the
            original 0.15 damping at 4x/day.
        Returns: (recalibrated_count, factor_cvs_dict)
        """
        from app.core.config import (
            L3_CONCEPT_TYPES,
            MIN_UTILITY_SAMPLES,
            PSIS_QUARANTINE_CONFIDENCE_CAP,
            PSIS_QUARANTINE_EVIDENCE_MARKER,
            RECALIBRATION_EVIDENCE_WEIGHT,
            RECALIBRATION_UTILITY_WEIGHT,
            get_feature_flag,
        )
        recalibrated = 0
        recal_events: list[tuple] = []
        factor_lists: dict[str, list] = {
            "reliability": [], "directness": [], "consistency": [],
            "corroboration": [], "recency": [], "composite": [],
        }
        batch_updates = []
        _utility_blended_count = 0

        # RETRIEVAL-080: Check if feedback loop is enabled
        _feedback_loop_on = get_feature_flag("FEEDBACK_LOOP_ENABLED", True)

        all_concepts = _preloaded if _preloaded is not None else list_concepts_full()
        for idx, concept in enumerate(all_concepts, start=1):
            self._check_abort_every(idx, 100, "recalibrate")
            if not concept.evidence:
                continue
            e_c = _compute_evidence_strength(concept)
            _collect_factor_values(concept, factor_lists)
            if e_c <= 0.0:
                continue

            # EUNOMIA-038: Evidence-dependent linear amplifier.
            # K(n) = K_BASE + K_SLOPE * ln(evidence_count)
            # Corrects structural multiplicative bias and breaks corroboration ceiling.
            _ev_count = len(concept.evidence) if concept.evidence else 1
            _k_n = RECALIBRATION_K_BASE + RECALIBRATION_K_SLOPE * math.log(max(_ev_count, 1))
            _amplified_target = e_c * _k_n

            # RETRIEVAL-080: Compute blended target + damping
            target = _amplified_target
            _effective_damping = RECALIBRATION_DAMPING
            if _feedback_loop_on:
                _util_score = concept.utility_score
                _util_samples = concept.utility_samples or 0
                if _util_score is not None and _util_samples >= MIN_UTILITY_SAMPLES:
                    target = (RECALIBRATION_EVIDENCE_WEIGHT * _amplified_target
                              + RECALIBRATION_UTILITY_WEIGHT * _util_score)
                    _utility_blended_count += 1
                # EUNOMIA-038: Warmup guard removed — base damping already
                # reduced from 0.15 to 0.10, and dead zone removed. The gentler
                # convergence at 1x/day makes the guard unnecessary.

            gap = concept.confidence - target
            new_confidence = concept.confidence
            direction = None
            correction = 0.0

            if gap > RECALIBRATION_GAP_THRESHOLD:
                correction = _effective_damping * gap
                new_confidence = max(0.0, concept.confidence - correction)
                direction = "downward"
            elif gap < -RECALIBRATION_GAP_THRESHOLD:
                correction = _effective_damping * abs(gap)
                new_confidence = min(1.0, concept.confidence + correction)
                direction = "upward"
            else:
                # STABILITY-027: Catch-all PSIS cap for within-threshold concepts
                if PSIS_QUARANTINE_EVIDENCE_MARKER in (concept.evidence or []):
                    if concept.confidence > PSIS_QUARANTINE_CONFIDENCE_CAP:
                        old_confidence = concept.confidence
                        new_confidence = PSIS_QUARANTINE_CONFIDENCE_CAP
                        batch_updates.append((new_confidence, None, concept.id))
                        concept.confidence = new_confidence  # SCALE-004: ordering consistency
                        recalibrated += 1
                        recal_events.append(
                            (concept.id, old_confidence, new_confidence, e_c, "psis_cap",
                             new_confidence - old_confidence)
                        )
                continue

            if direction:
                if PSIS_QUARANTINE_EVIDENCE_MARKER in (concept.evidence or []):
                    new_confidence = min(new_confidence, PSIS_QUARANTINE_CONFIDENCE_CAP)

                # RETRIEVAL-080: Apply type-differentiated confidence caps
                if _feedback_loop_on:
                    _util_score = concept.utility_score
                    _ctype = concept.concept_type or "observation"
                    if _ctype in L3_CONCEPT_TYPES and _util_score is not None and _util_score > 0.6:
                        _type_cap = 0.85  # Proven L3: higher ceiling
                    elif _ctype in L3_CONCEPT_TYPES:
                        _type_cap = 0.7   # Standard L3 ceiling
                    else:
                        _type_cap = 0.6   # L1 ceiling (observations less stable)
                    new_confidence = min(new_confidence, _type_cap)

                old_confidence = concept.confidence
                batch_updates.append((new_confidence, None, concept.id))
                concept.confidence = new_confidence  # SCALE-004: ordering consistency
                recalibrated += 1
                if abs(correction) > RECALIBRATION_LOG_MIN_CORRECTION:
                    recal_events.append(
                        (concept.id, old_confidence, new_confidence, e_c, direction, correction)
                    )

        if _utility_blended_count > 0:
            logger.info(f"RETRIEVAL-080: Recalibration blended utility for {_utility_blended_count} concepts")

        written = _batch_update_confidence_stability(batch_updates)
        if written == 0 and batch_updates:
            logger.warning("REFLECT-022: Batch recalibrate failed, falling back to per-concept save")
            for conf, _, cid in batch_updates:
                try:
                    c = load_concept(cid, track_access=False)
                    if c:
                        original_ka = c.metadata.get("knowledge_area") if c.metadata else None
                        c.confidence = conf
                        if original_ka:
                            c.knowledge_area = original_ka
                            if c.metadata:
                                c.metadata["knowledge_area"] = original_ka
                        save_concept(c)
                except Exception as e:
                    logger.warning("REFLECT-022: Fallback save failed for %s: %s", cid, e)

        _flush_recalibration_events(recal_events)
        factor_cvs = _compute_factor_cvs(factor_lists)
        return recalibrated, factor_cvs

    # REFLECT-017: STALE_CONFIDENCE_MULTIPLIER and STALE_CONFIDENCE_FLOOR removed.
    # _cleanup_stale_resolved now archives concepts instead of reducing confidence.

    def _cleanup_stale_resolved(self) -> int:
        """Archive concepts whose summaries indicate resolved/obsolete state.

        MAINT-019 + REFLECT-017: Concepts with explicit status markers like
        [RESOLVED], [SUPERSEDED] should be archived (removed from active retrieval),
        not confidence-reduced. Status transition is the correct semantic — these
        concepts declare themselves replaced. Uses strict 3-tier pattern matching
        (GA-1 amendment) to avoid false positives on descriptive mentions.

        Returns: number of concepts archived.
        """
        # Tier 3 pattern: "is/was/now/been/already resolved/superseded/obsolete"
        _TIER3_PATTERN = re.compile(
            r"(?:is|was|now|been|already)\s+(?:resolved|superseded|obsolete|deprecated)",
            re.IGNORECASE,
        )

        from app.storage import _db

        with _db() as conn:
            # Tier 1+2: Bracketed markers and summary-leading status words
            tier12_candidates = conn.execute(
                """SELECT id, summary, confidence FROM concepts
                   WHERE status = 'active'
                   AND (summary LIKE '%[RESOLVED]%' OR summary LIKE '%[SUPERSEDED]%'
                        OR summary LIKE '%[OBSOLETE]%'
                        OR summary LIKE 'RESOLVED:%' OR summary LIKE 'RESOLVED -%'
                        OR summary LIKE 'SUPERSEDED:%' OR summary LIKE 'SUPERSEDED -%'
                        OR summary LIKE 'OBSOLETE:%' OR summary LIKE 'OBSOLETE -%')
                   AND (superseded_by IS NULL OR superseded_by = '')
                   AND confidence >= 0.1
                   AND always_activate = 0"""
            ).fetchall()

            # Tier 3: Broader LIKE then regex filter
            tier3_candidates = conn.execute(
                """SELECT id, summary, confidence FROM concepts
                   WHERE status = 'active'
                   AND (summary LIKE '%resolved%' OR summary LIKE '%superseded%'
                        OR summary LIKE '%obsolete%')
                   AND (superseded_by IS NULL OR superseded_by = '')
                   AND confidence >= 0.1
                   AND always_activate = 0"""
            ).fetchall()

            # Combine unique candidates
            seen_ids = {}
            for cid, summary, confidence in tier12_candidates:
                seen_ids[cid] = (summary, confidence)
            for cid, summary, confidence in tier3_candidates:
                if cid not in seen_ids and _TIER3_PATTERN.search(summary):
                    seen_ids[cid] = (summary, confidence)

            cleaned = 0
            for idx, (concept_id, (summary, confidence)) in enumerate(seen_ids.items(), start=1):
                self._check_abort_every(idx, 100, "stale_resolved_cleanup")
                # REFLECT-017: Archive instead of confidence reduction.
                # Status transition removes from active retrieval immediately.
                cleaned += apply_lifecycle_transition_conn(conn, concept_id, "archive")

            if cleaned > 0:
                conn.commit()
                logger.info(
                    "REFLECT-017: Stale resolved cleanup — archived %d/%d candidates",
                    cleaned,
                    len(seen_ids),
                )
        return cleaned

    def _cleanup_low_confidence(self) -> tuple[int, int]:
        """Archive concepts with very low confidence (AF-12 + Amendment 5/6/12).

        Safety guardrails:
        - Never GC firmware or always-activate concepts
        - Never GC concepts younger than 7 days
        - Archive (soft delete) with orphaned edge cleanup, never hard delete
        - Batch-limited to MAX_GC_BATCH to prevent timeouts on large brains

        Returns: (cleaned_count, gc_queue_remaining)  # DEBT-005
        """
        MAX_GC_BATCH = 500  # Amendment 12: resource bounds
        cleaned = 0
        skipped_invalid = 0
        candidates_checked = 0
        all_concept_ids = list_concepts()  # DEBT-005: capture total for queue monitoring
        total_concepts = len(all_concept_ids)

        for idx, concept_id in enumerate(all_concept_ids, start=1):
            self._check_abort_every(idx, 100, "cleanup")
            if candidates_checked >= MAX_GC_BATCH:
                remaining = total_concepts - candidates_checked
                logger.info(
                    "DEBT-005: GC batch limit reached (%d), checked %d/%d, %d remaining",
                    MAX_GC_BATCH,
                    candidates_checked,
                    total_concepts,
                    remaining,
                )
                break

            concept = load_concept(concept_id, track_access=False)
            if not concept:
                continue

            # Amendment 5: Validate concept data before processing
            if not self._validate_concept(concept):
                skipped_invalid += 1
                continue

            candidates_checked += 1

            if concept.confidence >= MIN_CONFIDENCE_THRESHOLD:
                continue

            # Safety guardrails: skip protected concepts
            if getattr(concept, "always_activate", False):
                continue
            if getattr(concept, "concept_type", "") == "firmware":
                continue
            # Skip recently created concepts (< 7 days old)
            try:
                created = (
                    datetime.fromisoformat(concept.created_at)
                    if isinstance(concept.created_at, str)
                    else concept.created_at
                )
                if created and (_utc_now() - _ensure_aware(created)).days < 7:
                    continue
            except (ValueError, TypeError, AttributeError):
                continue  # Can't parse date, skip to be safe

            if archive_concept(concept_id, on_archived=lambda cid: retrieval_engine.remove_concept(cid)):
                cleaned += 1
                logger.info(
                    f"GC archived concept {concept_id}: "
                    f"confidence={concept.confidence:.3f}, "
                    f"type={getattr(concept, 'concept_type', 'unknown')}"
                )

        if skipped_invalid > 0:
            logger.warning(f"GC skipped {skipped_invalid} concepts with invalid data")
        # DEBT-005: report remaining queue size
        gc_remaining = max(0, total_concepts - candidates_checked)
        return cleaned, gc_remaining

    @staticmethod
    def _validate_concept(concept) -> bool:
        """Amendment 5: Validate concept has valid fields. Returns True if ok."""
        try:
            if concept.confidence is None:
                return False
            if not isinstance(concept.confidence, (int, float)):  # noqa: UP038
                return False
            if math.isnan(concept.confidence) or math.isinf(concept.confidence):
                return False
            if not (0.0 <= concept.confidence <= 1.0):  # noqa: SIM103
                return False
            return True
        except Exception:
            return False

    def _update_associations(self) -> int:
        """Update association strengths based on co-access patterns."""
        return 0

    def _run_graduation_sweep(self, _t: dict) -> tuple[int, int]:
        """Run quarantine graduation sweep. Returns (graduated_count, discarded_count)."""
        graduated_count = 0
        discarded_quarantine_count = 0
        try:
            _s = time.monotonic()
            grad_result = _get_auto_graduate_quarantined()()
            graduated_count = len(grad_result.get("promoted", []))
            discarded_quarantine_count = len(grad_result.get("discarded", []))
            _t["quarantine_graduation"] = _elapsed_ms(_s)
            self._record_phase("quarantine_graduation", _t["quarantine_graduation"])
        except Exception as e:
            logger.warning(f"Quarantine graduation failed (non-fatal): {e}")
        return graduated_count, discarded_quarantine_count

    # --- STABILITY-024 Constants ---
    PROMOTION_SWEEP_MAX_PER_CYCLE = 500  # MATURITY-004: Raised from 100 to handle initial backlog flood

    def _quarantine_release_sweep(self) -> int:
        """MATURITY-003 Part B: Release QUARANTINED concepts back to PROVISIONAL.

        Concepts stuck in QUARANTINED for >= QUARANTINE_RELEASE_AGE_DAYS with
        evidence >= 1 and access >= 1 are re-checked against content policy.
        If they pass, they're released to PROVISIONAL (where they can then
        promote via Path A/B/C in _promotion_sweep).

        Returns count of concepts released.
        """
        from app.core.config import FEATURE_FLAGS, QUARANTINE_RELEASE_AGE_DAYS, QUARANTINE_RELEASE_CAP

        if not FEATURE_FLAGS.get("QUARANTINE_RELEASE_ENABLED", False):
            return 0

        released = 0

        try:
            with _db() as conn:
                # Find QUARANTINED concepts old enough for release review
                rows = conn.execute(
                    """SELECT id, data, access_count FROM concepts
                       WHERE is_current = 1
                       AND maturity = 'QUARANTINED'
                       AND julianday('now') - julianday(created_at) >= ?
                       LIMIT ?""",
                    (QUARANTINE_RELEASE_AGE_DAYS, QUARANTINE_RELEASE_CAP * 2),
                ).fetchall()
        except Exception as e:
            logger.warning("MATURITY-003: Quarantine release query failed: %s", e)
            return 0

        import json as _json  # DEBT-191: hoisted from loop body

        from app.policy import check_content_policy  # DEBT-191: hoisted from loop body

        for idx, row in enumerate(rows, start=1):
            self._check_abort_every(idx, 50, "quarantine_release")
            if released >= QUARANTINE_RELEASE_CAP:
                break

            concept_id = row[0]
            try:
                data = _json.loads(row[1]) if row[1] else {}
            except Exception:
                continue

            evidence_count = len(data.get("evidence", []))
            # DEBT-190: Read access_count from SQL column (authoritative), not JSON blob (stale)
            access_count = row[2] if row[2] is not None else 0

            # Must have at least 1 evidence and 1 access
            if evidence_count < 1 or access_count < 1:
                continue

            # Re-check content policy with current rules
            summary = data.get("summary", "")
            try:
                if check_content_policy(summary):
                    continue  # Still violates policy — stay quarantined
            except Exception:
                continue  # If policy check fails, don't release

            # Release: QUARANTINED → PROVISIONAL
            concept = load_concept(concept_id, track_access=False)
            if not concept or concept.maturity != "QUARANTINED":
                continue

            original_ka = concept.metadata.get("knowledge_area") if concept.metadata else None
            concept.maturity = "PROVISIONAL"
            if original_ka:
                concept.knowledge_area = original_ka
                if concept.metadata:
                    concept.metadata["knowledge_area"] = original_ka
            save_concept(concept)
            released += 1

            # Log governance event
            try:
                with _db() as _gov_conn:
                    _gov_conn.execute(
                        """INSERT INTO governance_events
                           (event_type, concept_id, details, created_at)
                           VALUES (?, ?, ?, ?)""",
                        (
                            "QUARANTINE_RELEASED",
                            concept_id,
                            _json.dumps({"evidence_count": evidence_count, "access_count": access_count}),
                            _utc_now_iso(),
                        ),
                    )
            except Exception:
                logger.debug("MONITOR-010: Non-fatal exception in reflection (suppressed)", exc_info=True)

            logger.info(
                "MATURITY-003: Released %s QUARANTINED → PROVISIONAL (evidence=%d, access=%d)",
                concept_id,
                evidence_count,
                access_count,
            )

        if released > 0:
            logger.info("MATURITY-003: Quarantine release sweep released %d concepts", released)
        return released

    def _promotion_sweep(self) -> int:
        """STABILITY-024 + MATURITY-003 Part D: Batch promote PROVISIONAL → ESTABLISHED.

        Mirrors the per-concept logic in session._maybe_promote_maturity() but
        runs as a batch sweep during reflection, catching concepts that gained
        enough evidence/access/reinforcement outside of evolution events.

        Promotion rules (OR logic — any path promotes):
          Path A: evidence_count >= PROVISIONAL_PROMOTION_MIN_EVIDENCE
                  AND access_count >= PROVISIONAL_PROMOTION_MIN_ACCESS
          Path B: reinforcement_count >= REINFORCEMENT_PROMOTION_THRESHOLD
          Path C (MATURITY-003 Part D): Temporal promotion for residual concepts
                  age >= TEMPORAL_MATURITY_AGE_DAYS AND evidence >= 1 AND access >= 3
                  AND accessed in last TEMPORAL_MATURITY_RECENCY_DAYS AND NOT contradicted
                  (feature-flagged: TEMPORAL_PROMOTION_ENABLED)
          Path D (MATURITY-005): Access-only graduation for evidence-free concepts
                  evidence_count == 0 AND access_count >= PROVISIONAL_PROMOTION_MIN_ACCESS
                  Handles conv_* extractions that never populate evidence[]
          Path E: Access-starvation escape hatch for evidence-bearing zero-access concepts
                  evidence_count >= 1 AND access_count == 0
                  AND age >= TEMPORAL_MATURITY_AGE_DAYS AND NOT contradicted
                  (feature-flagged: TEMPORAL_PROMOTION_ENABLED, shares cap with Path C)
        """
        from app.core.config import (
            FEATURE_FLAGS,
            PROVISIONAL_PROMOTION_MIN_ACCESS,
            PROVISIONAL_PROMOTION_MIN_EVIDENCE,
            REINFORCEMENT_PROMOTION_THRESHOLD,
            TEMPORAL_MATURITY_AGE_DAYS,
            TEMPORAL_MATURITY_MIN_ACCESS,
            TEMPORAL_MATURITY_MIN_EVIDENCE,
            TEMPORAL_MATURITY_RECENCY_DAYS,
            TEMPORAL_PROMOTION_CAP,
        )

        if not FEATURE_FLAGS.get("INGESTION_VALIDATION_ENABLED", False):
            return 0

        promoted = 0
        self._temporal_promoted_this_cycle = 0  # Reset Path C counter

        # Query PROVISIONAL concepts directly from DB for efficiency
        try:
            with _db() as conn:
                rows = conn.execute(
                    """SELECT id FROM concepts
                       WHERE is_current = 1
                       AND maturity = 'PROVISIONAL'
                       ORDER BY RANDOM()
                       LIMIT ?""",
                    (self.PROMOTION_SWEEP_MAX_PER_CYCLE * 2,),  # Over-fetch for filtering
                ).fetchall()
        except Exception as e:
            logger.warning("STABILITY-024: Failed to query PROVISIONAL concepts: %s", e)
            return 0

        for idx, (concept_id,) in enumerate(rows, start=1):
            self._check_abort_every(idx, 50, "promotion_sweep")
            if promoted >= self.PROMOTION_SWEEP_MAX_PER_CYCLE:
                break

            concept = load_concept(concept_id, track_access=False)
            if not concept:
                continue

            maturity = getattr(concept, "maturity", "ESTABLISHED")
            if maturity != "PROVISIONAL":
                continue  # Race condition guard

            evidence_count = len(concept.evidence) if concept.evidence else 0
            access_count = getattr(concept, "access_count", 0)
            reinforcement = getattr(concept, "reinforcement_count", 0)

            # Path A: evidence + access thresholds
            path_a = (
                evidence_count >= PROVISIONAL_PROMOTION_MIN_EVIDENCE
                and access_count >= PROVISIONAL_PROMOTION_MIN_ACCESS
            )
            # Path B: reinforcement threshold
            path_b = reinforcement >= REINFORCEMENT_PROMOTION_THRESHOLD

            # Path D (MATURITY-005): Access-only graduation for evidence-free concepts.
            # conv_* and similar extraction paths don't populate evidence[]. High
            # access count proves retrieval value in lieu of formal evidence strings.
            path_d = evidence_count == 0 and access_count >= PROVISIONAL_PROMOTION_MIN_ACCESS

            # Path C (MATURITY-003 Part D): Temporal promotion for residual concepts
            path_c = False
            _temporal_enabled = FEATURE_FLAGS.get("TEMPORAL_PROMOTION_ENABLED", False)
            if _temporal_enabled and not path_a and not path_b and not path_d:
                # Only check Path C if A and B didn't already promote
                from datetime import datetime

                created_at_str = getattr(concept, "created_at", None)
                last_accessed_str = getattr(concept, "last_accessed", None)

                if created_at_str and last_accessed_str:
                    try:
                        created_at = datetime.fromisoformat(str(created_at_str).replace("Z", "+00:00"))
                        last_accessed = datetime.fromisoformat(str(last_accessed_str).replace("Z", "+00:00"))
                        now = datetime.now(UTC)
                        age_days = (now - created_at).total_seconds() / 86400
                        recency_days = (now - last_accessed).total_seconds() / 86400

                        if (
                            age_days >= TEMPORAL_MATURITY_AGE_DAYS
                            and evidence_count >= TEMPORAL_MATURITY_MIN_EVIDENCE
                            and access_count >= TEMPORAL_MATURITY_MIN_ACCESS
                            and recency_days <= TEMPORAL_MATURITY_RECENCY_DAYS
                        ):
                            # Check no recent contradictions (last 14 days)
                            _contradicted = False
                            try:
                                with _db() as _contra_conn:
                                    _contra_row = _contra_conn.execute(
                                        """SELECT COUNT(*) FROM governance_events
                                           WHERE concept_id = ?
                                           AND event_type LIKE '%contradiction%'
                                           AND julianday('now') - julianday(created_at) <= ?""",
                                        (concept_id, TEMPORAL_MATURITY_RECENCY_DAYS),
                                    ).fetchone()
                                    _contradicted = (_contra_row[0] or 0) > 0
                            except Exception:
                                _contradicted = True  # Fail safe: don't promote if check fails

                            if not _contradicted:
                                path_c = True
                    except (ValueError, TypeError):
                        pass  # Skip if dates are malformed

            # Path E: Evidence-gated age promotion for zero-access concepts.
            # Breaks the access starvation loop: concepts with evidence but 0 access
            # never get retrieved → never get accessed → never promote. Path E promotes
            # after age gate + evidence + no contradictions, without requiring access.
            # Capped at TEMPORAL_PROMOTION_CAP per cycle (shared with Path C).
            path_e = False
            if (not path_a and not path_b and not path_c and not path_d
                    and access_count == 0 and evidence_count >= 1
                    and _temporal_enabled):
                from datetime import datetime as _dt_e
                _created_str_e = getattr(concept, "created_at", None)
                if _created_str_e:
                    try:
                        _created_e = _dt_e.fromisoformat(str(_created_str_e).replace("Z", "+00:00"))
                        _age_e = (_dt_e.now(UTC) - _created_e).total_seconds() / 86400
                        if _age_e >= TEMPORAL_MATURITY_AGE_DAYS:
                            # Check no contradictions
                            _contra_e = False
                            try:
                                with _db() as _ce_conn:
                                    _ce_row = _ce_conn.execute(
                                        """SELECT COUNT(*) FROM governance_events
                                           WHERE concept_id = ?
                                           AND event_type LIKE '%contradiction%'
                                           AND julianday('now') - julianday(created_at) <= ?""",
                                        (concept_id, TEMPORAL_MATURITY_RECENCY_DAYS),
                                    ).fetchone()
                                    _contra_e = (_ce_row[0] or 0) > 0
                            except Exception:
                                _contra_e = True
                            if not _contra_e:
                                path_e = True
                    except (ValueError, TypeError):
                        pass

            if path_a or path_b or path_c or path_d or path_e:
                # Enforce temporal/starvation promotion cap separately
                if (path_c or path_e) and not path_a and not path_b and not path_d:
                    if not hasattr(self, "_temporal_promoted_this_cycle"):
                        self._temporal_promoted_this_cycle = 0
                    if self._temporal_promoted_this_cycle >= TEMPORAL_PROMOTION_CAP:
                        continue
                    self._temporal_promoted_this_cycle += 1

                promotion_path = []
                if path_a:
                    promotion_path.append("A")
                if path_b:
                    promotion_path.append("B")
                if path_c:
                    promotion_path.append("C")
                if path_d:
                    promotion_path.append("D")
                if path_e:
                    promotion_path.append("E")
                path_label = "+".join(promotion_path)

                original_ka = concept.metadata.get("knowledge_area") if concept.metadata else None
                concept.maturity = "ESTABLISHED"
                concept.maturity_promoted_at = _utc_now_iso()
                concept.maturity_promotion_evidence = (
                    f"Reflection sweep: evidence={evidence_count}, access={access_count}, "
                    f"reinforcement={reinforcement}, path={path_label}"
                )
                # KA-005: Guard KA from being overwritten by save_concept
                if original_ka:
                    concept.knowledge_area = original_ka
                    if concept.metadata:
                        concept.metadata["knowledge_area"] = original_ka
                save_concept(concept)
                promoted += 1

                # MONITOR-030: Log governance event for ALL promotion paths (was: only Path C)
                try:
                    import json as _json

                    _event_details = {
                        "evidence_count": evidence_count,
                        "access_count": access_count,
                        "reinforcement_count": reinforcement,
                        "path": path_label,
                    }
                    # Add temporal-specific fields for Path C
                    if path_c:
                        _event_details["age_days"] = round(age_days, 1)

                    with _db() as _gov_conn:
                        _gov_conn.execute(
                            """INSERT INTO governance_events
                               (event_type, concept_id, details, created_at)
                               VALUES (?, ?, ?, ?)""",
                            (
                                "MATURITY_PROMOTED",
                                concept_id,
                                _json.dumps(_event_details),
                                _utc_now_iso(),
                            ),
                        )
                except Exception:
                    logger.debug("MONITOR-010: Non-fatal exception in reflection (suppressed)", exc_info=True)

                logger.info(
                    "STABILITY-024: Promoted %s PROVISIONAL → ESTABLISHED "
                    "(evidence=%d, access=%d, reinforcement=%d, path=%s)",
                    concept_id,
                    evidence_count,
                    access_count,
                    reinforcement,
                    path_label,
                )

        if promoted > 0:
            logger.info("STABILITY-024: Promotion sweep promoted %d concepts", promoted)
        return promoted

    # --- MATURITY-003 Phase A5: Evidence Backfill ---

    def _evidence_backfill_sweep(self) -> int:
        """MATURITY-003 Phase A5: Backfill evidence for stuck PROVISIONAL concepts.

        One-time-ish job: for v1 PROVISIONAL concepts with access >= 5 and
        evidence == 1, search for semantically similar concepts via embedding
        cosine >= 0.55. If found, add cross-reference evidence without full
        evolution. This unblocks concepts stuck in the TF-IDF dead zone.

        Cap: EVIDENCE_BACKFILL_CAP per reflection cycle.
        Feature-flagged: EVIDENCE_BACKFILL_ENABLED.
        """
        from app.core.config import (
            EVIDENCE_BACKFILL_CAP,
            EVIDENCE_BACKFILL_COSINE_THRESHOLD,
            EVIDENCE_BACKFILL_MIN_ACCESS,
            FEATURE_FLAGS,
        )

        if not FEATURE_FLAGS.get("EVIDENCE_BACKFILL_ENABLED", False):
            return 0

        # Embedding engine must be available for semantic matching
        from app.storage.embedding import embedding_engine

        if not embedding_engine.is_available or embedding_engine.index_size == 0:
            logger.debug("MATURITY-003 A5: Embedding engine unavailable, skipping backfill")
            return 0

        backfilled = 0

        try:
            with _db() as conn:
                # Find v1 PROVISIONAL concepts with high access but only 1 evidence
                rows = conn.execute(
                    """SELECT c.id, c.data FROM concepts c
                       WHERE c.is_current = 1
                       AND c.maturity IN ('PROVISIONAL', 'QUARANTINED')
                       AND c.status = 'active'
                       LIMIT ?""",
                    (EVIDENCE_BACKFILL_CAP * 3,),  # Over-fetch for filtering
                ).fetchall()
        except Exception as e:
            logger.warning("MATURITY-003 A5: Failed to query stuck concepts: %s", e)
            return 0

        import json as _json

        for idx, row in enumerate(rows, start=1):
            self._check_abort_every(idx, 25, "evidence_backfill")
            if backfilled >= EVIDENCE_BACKFILL_CAP:
                break

            concept_id = row[0]
            try:
                data = _json.loads(row[1]) if row[1] else {}
            except Exception:
                continue

            evidence = data.get("evidence", [])
            access_count = data.get("access_count", 0)

            # Only target concepts stuck by evidence: high access, low evidence
            if access_count < EVIDENCE_BACKFILL_MIN_ACCESS or len(evidence) > 1:
                continue

            summary = data.get("summary", "")
            if not summary:
                continue

            # Search for semantically similar concepts via embedding
            try:
                emb_results = embedding_engine.search(summary, top_k=3)
            except Exception as e:
                logger.debug("MATURITY-003 A5: Embedding search failed for %s: %s", concept_id, e)
                continue

            # Find a cross-reference match (not self, above threshold)
            cross_ref_id = None
            cross_ref_score = 0.0
            for match_id, score in emb_results:
                if match_id == concept_id:
                    continue
                if score >= EVIDENCE_BACKFILL_COSINE_THRESHOLD:
                    cross_ref_id = match_id
                    cross_ref_score = score
                    break

            if not cross_ref_id:
                continue

            # Add cross-reference evidence without full evolution
            concept = load_concept(concept_id, track_access=False)
            if not concept or concept.maturity != "PROVISIONAL":
                continue

            new_evidence = {
                "source": "backfill_cross_reference",
                "text": f"Semantically similar to {cross_ref_id} (cosine={cross_ref_score:.3f})",
                "timestamp": _utc_now_iso(),
            }

            if not concept.evidence:
                concept.evidence = []
            concept.evidence.append(new_evidence)

            # Preserve KA (KA-005 guard)
            original_ka = concept.metadata.get("knowledge_area") if concept.metadata else None
            if original_ka:
                concept.knowledge_area = original_ka
                if concept.metadata:
                    concept.metadata["knowledge_area"] = original_ka

            save_concept(concept)
            backfilled += 1

            # Log governance event
            try:
                with _db() as _gov_conn:
                    _gov_conn.execute(
                        """INSERT INTO governance_events
                           (event_type, concept_id, details, created_at)
                           VALUES (?, ?, ?, ?)""",
                        (
                            "EVIDENCE_BACKFILLED",
                            concept_id,
                            _json.dumps(
                                {
                                    "cross_ref_id": cross_ref_id,
                                    "cosine_score": round(cross_ref_score, 4),
                                    "access_count": access_count,
                                }
                            ),
                            _utc_now_iso(),
                        ),
                    )
            except Exception:
                logger.debug("MONITOR-010: Non-fatal exception in reflection (suppressed)", exc_info=True)

            logger.info(
                "MATURITY-003 A5: Backfilled evidence for %s (cross_ref=%s, cosine=%.3f, access=%d)",
                concept_id,
                cross_ref_id,
                cross_ref_score,
                access_count,
            )

        if backfilled > 0:
            logger.info("MATURITY-003 A5: Evidence backfill sweep backfilled %d concepts", backfilled)
        return backfilled

    def _apply_forgetting(self) -> int:
        """Archive concepts that are low-salience, rarely accessed, and stale.

        Forgetting = archiving, NOT deletion. Always recoverable.
        Uses salience field and access metrics as inputs.

        All four criteria must be true (conservative AND logic):
          1. salience_source != "user" (user-explicit salience is never overridden)
          2. salience < FORGETTING_SALIENCE_THRESHOLD (0.15)
          3. access_count < FORGETTING_ACCESS_COUNT_THRESHOLD (2)
          4. last_accessed > FORGETTING_STALENESS_DAYS ago (90 days)

        Bootstrap protection: concepts with last_accessed=None are pre-tracking
        and are SKIPPED until their first tracked access followed by dormancy.
        """
        archived_count = 0
        for concept_id in list_concepts():
            concept = load_concept(concept_id, track_access=False)
            if not concept:
                continue
            # REFLECT-003: Use created_at as fallback for NULL last_accessed.
            # Bootstrap protection retired — pith is mature.
            if concept.last_accessed is None:
                try:
                    last_access = datetime.fromisoformat(concept.created_at)
                except (ValueError, TypeError, AttributeError):
                    continue  # Safety: skip if created_at is missing/malformed
            # --- remaining checks apply to ALL concepts ---
            if concept.salience_source == "user":
                continue
            if concept.salience >= FORGETTING_SALIENCE_THRESHOLD:
                continue
            if concept.access_count >= FORGETTING_ACCESS_COUNT_THRESHOLD:
                continue
            # Staleness calculation: last_access already set for NULL case above
            if concept.last_accessed is not None:
                try:
                    last_access = datetime.fromisoformat(concept.last_accessed)
                except (ValueError, TypeError):
                    continue
            days_since = (_utc_now() - _ensure_aware(last_access)).days
            if days_since < FORGETTING_STALENESS_DAYS:
                continue
            if archive_concept(concept_id, on_archived=lambda cid: retrieval_engine.remove_concept(cid)):
                archived_count += 1
                logger.info(
                    f"Forgot concept {concept_id}: "
                    f"salience={concept.salience}, "
                    f"salience_source={concept.salience_source}, "
                    f"access_count={concept.access_count}, "
                    f"days_since_access={days_since}"
                )
                if archived_count >= MAX_FORGETTING_PER_CYCLE:
                    logger.warning(
                        f"REFLECT-005: Forgetting cap reached ({MAX_FORGETTING_PER_CYCLE}). "
                        f"Deferring remaining candidates to next cycle."
                    )
                    break
        return archived_count

    def analyze_stability(self) -> dict:
        """Analyze overall pith stability."""
        concepts = []
        for concept_id in list_concepts():
            concept = load_concept(concept_id, track_access=False)
            if concept:
                concepts.append(concept)
        if not concepts:
            return {"status": "empty"}
        n = len(concepts)
        avg_confidence = sum(c.confidence for c in concepts) / n
        avg_stability = sum(c.stability for c in concepts) / n
        unstable = [c for c in concepts if c.stability < 0.4]
        uncertain = [c for c in concepts if c.confidence < 0.5]
        conflicted = [c for c in concepts if len(c.hypotheses) >= 2]

        # HEALTH-001: 5-factor weighted health score
        established = sum(1 for c in concepts if getattr(c, "maturity", "") == "ESTABLISHED")
        maturity_health = established / n  # % ESTABLISHED

        # CONNECTIVITY-FIX: Use DB query instead of c.associations JSON blob (was always empty)
        orphan_count = count_orphan_concepts()
        connectivity = (n - orphan_count) / n  # % with associations

        # FRESHNESS_UNIFIED_REDESIGN: Exponential decay health freshness
        from app.core.config import HEALTH_FRESHNESS_HALF_LIFE_DAYS

        _now = _utc_now()
        _ln2 = math.log(2)
        _hl = max(0.1, HEALTH_FRESHNESS_HALF_LIFE_DAYS)
        _freshness_sum = 0.0
        for c in concepts:
            _ts = c.last_organic_access or c.last_accessed or c.created_at
            if _ts:
                try:
                    ts_str = _ts if isinstance(_ts, str) else str(_ts)
                    age_days = (_now - _ensure_aware(
                        datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    )).total_seconds() / 86400.0
                    _freshness_sum += math.exp(-_ln2 / _hl * age_days)
                except Exception:
                    logger.debug("health_freshness_parse_error ts=%s", _ts)
        freshness = _freshness_sum / n if n > 0 else 0.0

        health_score = (
            0.30 * avg_confidence
            + 0.30 * avg_stability
            + 0.15 * maturity_health
            + 0.15 * connectivity
            + 0.10 * freshness
        )

        # MONITOR-007: Saturation detection
        saturation_alerts = []
        metrics_to_check = {
            "confidence": [c.confidence for c in concepts],
            "stability": [c.stability for c in concepts],
        }
        currency_scores = [c.currency_score for c in concepts if c.currency_score is not None]
        if currency_scores:
            metrics_to_check["currency_score"] = currency_scores

        for metric_name, scores in metrics_to_check.items():
            above = sum(1 for s in scores if s > SATURATION_THRESHOLD)
            pct = above / len(scores) if scores else 0
            if pct >= SATURATION_ALERT_PCT:
                saturation_alerts.append(
                    {
                        "metric": metric_name,
                        "above_threshold": above,
                        "total": len(scores),
                        "pct": round(pct, 4),
                    }
                )

        if saturation_alerts:
            health_score *= 0.8  # Penalize saturated health

        return {
            "total_concepts": n,
            "avg_confidence": avg_confidence,
            "avg_stability": avg_stability,
            "unstable_count": len(unstable),
            "uncertain_count": len(uncertain),
            "conflicted_count": len(conflicted),
            "health_score": round(health_score, 4),
            "health_factors": {  # HEALTH-001: transparency
                "confidence": round(avg_confidence, 4),
                "stability": round(avg_stability, 4),
                "maturity": round(maturity_health, 4),
                "connectivity": round(connectivity, 4),
                "freshness": round(freshness, 4),
            },
            "saturation_alerts": saturation_alerts,  # MONITOR-007
        }


# Global instance
reflection_engine = ReflectionEngine()


def run_standalone_promotion() -> int:
    """ARCH-D05: Public entry point for standalone promotion sweep.

    Decoupled from full reflect() so promotion runs even when
    reflection times out (79.6% timeout rate as of 2026-03-18).
    Called by maintenance phase 6 independently of phase 2 (reflection).
    """
    return reflection_engine._promotion_sweep()
