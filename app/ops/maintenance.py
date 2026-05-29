"""Pith Maintenance — Autonomous Cognitive Lifecycle.

Unified scheduler that wires all built-but-unwired autonomous features.
Think of this as Pith's "sleeptime" — it processes, consolidates,
and generates insights while the user is away.

Maintenance phases (executed in order):
  Phase 1: Scheduled async tasks (authority recal, stale-risk detector, etc.)
  Phase 2: Reflection cycle (decay, forgetting, strengthening, merging)
  Phase 3: Experiment generation (synthesis, hypothesis, counterfactual, analogy)
  Phase 4: Curiosity — question generation for weak concepts
  Phase 5: Health report + degradation alerts
  Phase 6: Standalone promotion sweep (ARCH-D05)
  Phase 7: Concept synthesis (SKILL-DEPLOY-001)
  Phase 8: Database backup — WAL-safe, cross-platform (OPS-152)

Can be triggered:
  - CLI: pith maintenance run [--phase N] [--dry-run]
  - API: POST /maintenance {phases: [1,2,3,4,5,6,7,8]}
  - Scheduler: launchd/cron calls the CLI every 6 hours
"""

import asyncio
import functools
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

from app.core.constants import GOV_EVENT_CONTRADICTION_REVIEW
from app.core.datetime_utils import _ensure_aware, _utc_now, _utc_now_iso
from app.storage import owned_connection

logger = logging.getLogger(__name__)

# MAINT-035 / ARGUS-C3-F2: Configurable per-phase timeout, clamped to [30, 600] seconds
# Default 120s preserved. Override with PITH_MAINTENANCE_PHASE_TIMEOUT for large brains.
# Parse+clamp logic extracted into _parse_phase_timeout() for unit-testable config parsing
# without module reload (importlib.reload re-executes all module-level code).


def _parse_phase_timeout(env_value: str | None) -> int:
    """Parse and clamp PITH_MAINTENANCE_PHASE_TIMEOUT env value.

    Args:
        env_value: Raw string from os.environ (or None → use default 120).

    Returns:
        Clamped integer timeout in seconds, within [30, 600].
    """
    try:
        raw = int(env_value) if env_value is not None else 120
    except (ValueError, TypeError):
        logger.warning(
            "MAINT-035: Invalid PITH_MAINTENANCE_PHASE_TIMEOUT=%r, using default 120",
            env_value,
        )
        raw = 120
    clamped = max(30, min(600, raw))
    if raw != clamped:
        logger.warning(
            "PITH_MAINTENANCE_PHASE_TIMEOUT=%d clamped to %d (valid range: 30-600)",
            raw,
            clamped,
        )
    return clamped


PHASE_TIMEOUT_SECONDS = _parse_phase_timeout(os.environ.get("PITH_MAINTENANCE_PHASE_TIMEOUT"))
EXP_TYPE_TIMEOUT_SECONDS = 30  # PERF-025: Per-type timeout for blocking executor calls


def _reflection_timeout_safety_margin(phase_timeout_seconds: int | float) -> float:
    """Return the internal reflection deadline margin before the outer phase timeout."""
    return min(30.0, max(5.0, float(phase_timeout_seconds) * 0.20))


def _reflection_long_step_min_budget(phase_timeout_seconds: int | float) -> float:
    """Return minimum remaining budget required before starting known long steps."""
    return min(45.0, max(10.0, float(phase_timeout_seconds) * 0.20))


REFLECTION_TIMEOUT_SAFETY_MARGIN_SECONDS = _reflection_timeout_safety_margin(PHASE_TIMEOUT_SECONDS)
REFLECTION_LONG_STEP_MIN_BUDGET_SECONDS = _reflection_long_step_min_budget(PHASE_TIMEOUT_SECONDS)
ORPHANED_SUPERSESSION_SENTINEL = "__orphaned_supersession__"


@dataclass
class MaintenanceReport:
    """Results from a maintenance run."""

    started_at: str = ""
    completed_at: str = ""
    phases_run: list[str] = field(default_factory=list)
    results: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    dry_run: bool = False

    def to_dict(self) -> dict:
        return {
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self._duration(),
            "phases_run": self.phases_run,
            "results": self.results,
            "errors": self.errors,
            "warnings": self.warnings,
            "dry_run": self.dry_run,
            "success": len(self.errors) == 0,
        }

    def _duration(self) -> float:
        if not self.started_at or not self.completed_at:
            return 0.0
        try:
            s = datetime.fromisoformat(self.started_at)
            e = datetime.fromisoformat(self.completed_at)
            return round((e - s).total_seconds(), 1)
        except (ValueError, TypeError):
            return 0.0


# =============================================================================
# Phase 1: Scheduled Async Tasks
# =============================================================================


async def phase1_scheduled_tasks(conn=None, dry_run: bool = False) -> dict:
    """Run all overdue scheduled tasks via AsyncTaskRunner.

    Tasks with interval_hours > 0 that haven't run within their interval:
      - currency_scan: REMOVED (CURRENCY-002) — legacy scorer, replaced by RETRIEVAL-015
      - criteria_staleness_detector: COGGOV-014 stale-risk lifecycle scan
      - authority_recalibration (168h/7d): Normalize authority distribution
      - cko_lifecycle (24h): Refresh CKO scores, archive stale CKOs
      - evidence_consolidation (24h): Merge redundant evidence
      - edge_reclassification (24h): Reclassify untyped 'related_to' edges
      - association_discovery (24h): TF-IDF similarity-based auto-association
      - staleness_alerts (24h): Flag high-authority stale concepts
    """
    from app.ops.async_tasks import ensure_async_tables, get_degraded_tasks, task_runner

    if dry_run:
        with owned_connection() as conn:
            ensure_async_tables(conn)
            degraded = get_degraded_tasks(conn)
            return {"dry_run": True, "tasks_due": degraded}

    results = await task_runner.run_scheduled_tasks()

    # STABILITY-036: Per-task reporting — surface success/failure counts
    # so maintenance heartbeat and health endpoint show granular status.
    task_errors = []
    for task_type, result in results.items():
        status = result.get("status", "unknown")
        if status in ("timeout", "failed", "cancelled"):
            task_errors.append(f"{task_type}: {status} — {result.get('error', 'unknown')}")
    tasks_deferred = sum(1 for r in results.values() if r.get("status") == "deferred_budget")

    return {
        "tasks_executed": sum(1 for r in results.values() if r.get("status") != "deferred_budget"),
        "tasks_succeeded": sum(1 for r in results.values() if r.get("status") == "success"),
        "tasks_failed": len(task_errors),
        "tasks_deferred": tasks_deferred,
        "errors": task_errors,
        "details": results,
    }


# =============================================================================
# Phase 2: Reflection Cycle
# =============================================================================


async def phase2_reflection(conn=None, dry_run: bool = False) -> dict:
    """Run full reflection: decay, forgetting, strengthening, merging.

    Uses reflection_engine.reflect(mode='full') which handles:
      - Confidence decay (concepts not accessed in 30+ days)
      - Active forgetting (archive low-salience, low-access concepts)
      - Strengthening (boost frequently-accessed concepts)
      - Duplicate merging (high-similarity same-area concepts)
      - Association cleanup
      - Self-model update
      - Checkpoint TTL cleanup
    """
    from app.cognitive.reflection import reflection_engine

    if dry_run:
        should = reflection_engine.should_reflect()
        return {"dry_run": True, "should_reflect": should}

    loop = asyncio.get_running_loop()
    cancel_event = threading.Event()
    deadline_monotonic = time.monotonic() + max(
        1.0,
        PHASE_TIMEOUT_SECONDS - REFLECTION_TIMEOUT_SAFETY_MARGIN_SECONDS,
    )
    future = loop.run_in_executor(
        None,
        functools.partial(
            reflection_engine.reflect,
            mode="full",
            cancel_event=cancel_event,
            deadline_monotonic=deadline_monotonic,
            long_step_min_remaining_seconds=REFLECTION_LONG_STEP_MIN_BUDGET_SECONDS,
        ),
    )
    try:
        summary = await future
    except asyncio.CancelledError:
        cancel_event.set()
        raise
    # MAINT-026: reflect() runs in thread executor so asyncio.wait_for timeout
    # can cancel the phase if it exceeds PHASE_TIMEOUT_SECONDS (previously
    # reflect() blocked the event loop with no cancellation point).
    # AF-03 fix + Amendment 4: null safety + JSON serialization
    if summary is None:
        return {"reflection_summary": {"status": "skipped", "reason": "no_consolidation_candidates"}}
    summary_dict = summary.model_dump() if hasattr(summary, "model_dump") else summary
    if getattr(summary, "aborted", False):
        return {
            "reflection_summary": summary_dict,
            "budget_aborted": True,
            "abort_reason": getattr(summary, "abort_reason", None),
            "last_completed_step": getattr(summary, "last_completed_step", None),
            "abort_stage": getattr(summary, "abort_stage", None),
        }
    return {"reflection_summary": summary_dict}


# =============================================================================
# Phase 2.4: Auto-Association (RB-03)
# =============================================================================


async def phase2_4_auto_associate(conn, dry_run: bool = False) -> dict:
    """Run batch auto-association to link related concepts.

    Uses TF-IDF cosine similarity to discover and create association
    edges between concepts. Two tiers: direct similarity (tier1) and
    domain-boosted orphan rescue (tier2).

    Budget: 30s max, default thresholds.
    """
    from app.cognitive.association import auto_associate_batch
    from app.core.models import AutoAssociateBatchRequest

    request = AutoAssociateBatchRequest(dry_run=dry_run)
    try:
        result = auto_associate_batch(request)
        return result.model_dump() if hasattr(result, "model_dump") else {"status": "completed"}
    except Exception as e:
        logger.error("Auto-association failed: %s", e, exc_info=True)
        return {"status": "error", "error": str(e)}


# =============================================================================
# Phase 2.5: Contradiction Sweep (§3.4)
# =============================================================================


async def phase2_5_contradiction_sweep(conn, dry_run: bool = False) -> dict:
    """Detect and resolve contradictions using type-ranked tiebreaker.

    Scans ACTIVE decision-tier concepts in same knowledge_area for
    high word-overlap pairs (potential contradictions). Resolves using
    type-ranked resolution (A3) and executes supersession for clear winners.

    Budget: 10s max, 50 concept-pair cap per run.
    """
    from app.cognitive.supersession import (
        SUPERSESSION_SIMILARITY_THRESHOLD,
        execute_supersession,
        resolve_type_ranked,
    )

    t0 = time.time()
    BUDGET_SECONDS = 10.0
    PAIR_CAP = 50

    # Fetch all decision-tier, non-SUPERSEDED concepts grouped by knowledge_area
    rows = conn.execute(
        """SELECT id, summary, concept_type, confidence, authority_score,
                  currency_status, json_extract(data, '$.knowledge_area') as ka
           FROM concepts
           WHERE is_current = 1
             AND status != 'deleted'
             AND concept_type IN ('decision', 'principle', 'constraint')
             AND (currency_status IS NULL OR currency_status != 'SUPERSEDED')
           ORDER BY ka"""
    ).fetchall()

    if dry_run:
        return {"dry_run": True, "eligible_concepts": len(rows)}

    # Group by knowledge_area
    from collections import defaultdict

    ka_groups = defaultdict(list)
    for r in rows:
        ka = r[6] or "general"
        ka_groups[ka].append(r)

    resolved = 0
    flagged_review = 0
    pairs_checked = 0
    skipped_budget = False

    for ka, group in ka_groups.items():
        if len(group) < 2:
            continue
        # Check all pairs within KA (O(n^2) but capped)
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if pairs_checked >= PAIR_CAP:
                    skipped_budget = True
                    break
                if (time.time() - t0) > BUDGET_SECONDS:
                    skipped_budget = True
                    break

                a, b = group[i], group[j]
                a_words = set((a[1] or "").lower().split())
                b_words = set((b[1] or "").lower().split())
                if not a_words or not b_words:
                    continue

                similarity = len(a_words & b_words) / len(a_words | b_words)
                if similarity < SUPERSESSION_SIMILARITY_THRESHOLD:
                    continue

                pairs_checked += 1

                # Type-ranked resolution
                winner = resolve_type_ranked(
                    concept_a_type=a[2] or "observation",
                    concept_a_authority=a[4] or 0.5,
                    concept_b_type=b[2] or "observation",
                    concept_b_authority=b[4] or 0.5,
                )

                if winner == "review":
                    flagged_review += 1
                    # Emit review event
                    now = _utc_now_iso()
                    conn.execute(
                        """INSERT INTO governance_events
                           (event_type, concept_id, details, created_at)
                           VALUES (?, ?, ?, ?)""",
                        (
                            GOV_EVENT_CONTRADICTION_REVIEW,
                            a[0],
                            json.dumps(
                                {
                                    "other_concept_id": b[0],
                                    "similarity": round(similarity, 4),
                                    "knowledge_area": ka,
                                    "source": "maintenance_contradiction_sweep",
                                }
                            ),
                            now,
                        ),
                    )
                    continue

                # Execute supersession: loser gets SUPERSEDED
                if winner == "a":
                    old_id, new_id = b[0], a[0]
                else:
                    old_id, new_id = a[0], b[0]

                try:
                    # DATA-060: execute_supersession requires transactional conn
                    if hasattr(conn, "in_transaction") and not conn.in_transaction:
                        conn.execute("BEGIN IMMEDIATE")
                    result = execute_supersession(
                        old_concept_id=old_id,
                        new_concept_id=new_id,
                        reason=f"contradiction_sweep (similarity={similarity:.3f})",
                        conn=conn,
                    )
                    if hasattr(conn, "in_transaction") and conn.in_transaction:
                        conn.commit()
                    if result.superseded:
                        resolved += 1
                except Exception as e:
                    logger.warning(
                        "Contradiction sweep supersession failed (%s vs %s): %s",
                        a[0],
                        b[0],
                        e,
                    )

            if skipped_budget:
                break
        if skipped_budget:
            break

    conn.commit()

    return {
        "eligible_concepts": len(rows),
        "pairs_checked": pairs_checked,
        "resolved": resolved,
        "flagged_for_review": flagged_review,
        "budget_exceeded": skipped_budget,
        "elapsed_seconds": round(time.time() - t0, 2),
    }


# =============================================================================
# Phase 2.6: Episode Retention (INFRA-002B)
# =============================================================================


async def phase2_6_episode_retention(conn, dry_run: bool = False) -> dict:
    """Run episode retention pipeline (INFRA-002B, Memory Integrity §5.2.5).

    Tiered: 30-day raw purge → 180-day archive → 365-day full delete.
    Always runs purge tier even if EPISODES_ENABLED=False (PII safety).
    """
    if dry_run:
        return {"dry_run": True, "note": "episode retention skipped in dry run"}

    from app.core.config import FEATURE_FLAGS
    from app.features.episodes import purge_expired_raw_text, run_episode_retention_job

    # PII safety: always purge raw text, even if recording is disabled.
    # If someone disables episodes for safety, existing raw PII must still
    # be cleaned up. Only skip archive/delete tiers when disabled.
    if not FEATURE_FLAGS.get("EPISODES_ENABLED", False):
        purged = purge_expired_raw_text()
        return {"status": "recording_disabled_purge_only", "raw_purged": purged}

    return run_episode_retention_job()


# =============================================================================
# Phase 2.7: DATA-048 — Fix supersession desync (is_current=1 on superseded rows)
# =============================================================================


def _migrate_desynced_supersessions(conn) -> dict:
    """DATA-048: One-time (idempotent) migration to fix is_current desync.

    Finds concepts with superseded_by IS NOT NULL AND is_current = 1 and sets
    is_current = 0 with decayed confidence. Safe to run repeatedly — WHERE
    clause excludes already-fixed rows.
    """
    from app.storage import apply_lifecycle_transition_conn

    stats = {"fixed": 0, "skipped": 0, "errors": 0}

    rows = conn.execute(
        """SELECT id, confidence, superseded_by FROM concepts
           WHERE superseded_by IS NOT NULL AND is_current = 1"""
    ).fetchall()

    for concept_id, confidence, superseded_by in rows:
        try:
            superseder_ok = conn.execute(
                "SELECT 1 FROM concepts WHERE id = ?", (superseded_by,)
            ).fetchone()
            if not superseder_ok:
                stats["skipped"] += 1
                continue
            decayed_confidence = max(0.0, (confidence if confidence is not None else 0.5) - 0.3)
            stats["fixed"] += apply_lifecycle_transition_conn(
                conn,
                concept_id,
                "supersede",
                superseded_by=superseded_by,
                reason="DATA-048 desync migration",
                confidence=decayed_confidence,
            )
        except Exception as e:
            logger.warning("DATA-048 migration error for %s: %s", concept_id, e)
            stats["errors"] += 1

    conn.commit()
    logger.info("DATA-048 migration: %s", stats)
    return stats


async def _phase2_7_fix_supersession_desync(conn, dry_run: bool = False) -> dict:
    """Phase 2.7: Run DATA-048 supersession desync migration as a maintenance sub-phase."""
    desynced_count = conn.execute(
        "SELECT COUNT(*) FROM concepts WHERE superseded_by IS NOT NULL AND is_current = 1"
    ).fetchone()[0]
    if desynced_count == 0:
        return {"fixed": 0, "skipped": 0, "errors": 0, "status": "no_desynced_rows"}
    if dry_run:
        return {"would_fix": desynced_count, "status": "dry_run"}
    logger.info("DATA-048: Found %d desynced supersessions, running migration", desynced_count)
    return _migrate_desynced_supersessions(conn)


async def _phase2_8_currency_actuator(conn, dry_run: bool = False) -> dict:
    """Phase 2.8: Currency→Status actuator — converts currency signals into status changes.

    Three tiers (CURRENCY_STATUS_ACTUATOR_SPEC):
      Tier 1: currency_status=SUPERSEDED + is_current=0 + superseded_by set → status='superseded'
      Tier 2: currency_status=SUPERSEDED + superseded_by set → status='superseded'
      Tier 3: currency_status=CONTRADICTED + authority<0.3 + age>14d → archive
    """
    from app.core.config import get_feature_flag
    from app.storage import apply_lifecycle_transition_conn

    if not get_feature_flag("CURRENCY_ACTUATOR_ENABLED", True):
        return {"status": "disabled", "tier1": 0, "tier2": 0, "tier3": 0}

    stats = {
        "tier1": 0,
        "tier2": 0,
        "tier3": 0,
        "tier3_failed": 0,
        "missing_pointer_review": 0,
        "status": "ok",
    }

    # DATA-070: Rows without superseded_by are review cases, not automatic supersessions.
    missing_pointer_row = conn.execute(
        """SELECT COUNT(*) FROM concepts
           WHERE currency_status = 'SUPERSEDED'
             AND status = 'active'
             AND (superseded_by IS NULL OR superseded_by = '')"""
    ).fetchone()
    stats["missing_pointer_review"] = missing_pointer_row[0] if missing_pointer_row else 0

    # --- Tier 1: SUPERSEDED + is_current=0 + pointer but status still 'active' ---
    tier1_rows = conn.execute(
        """SELECT id, superseded_by FROM concepts
           WHERE currency_status = 'SUPERSEDED'
             AND is_current = 0
             AND status = 'active'
             AND superseded_by IS NOT NULL
             AND superseded_by != ''"""
    ).fetchall()

    if tier1_rows:
        for _sid, _superseded_by in tier1_rows:
            if not dry_run:
                apply_lifecycle_transition_conn(
                    conn,
                    _sid,
                    "supersede",
                    superseded_by=_superseded_by,
                    reason="CURRENCY-ACTUATOR Tier 1",
                )
            stats["tier1"] += 1
        logger.info("CURRENCY-ACTUATOR Tier 1: %d concepts → status='superseded'", stats["tier1"])

    # --- Tier 2: SUPERSEDED + superseded_by IS NOT NULL but status still 'active' ---
    # (Catches cases where is_current was already fixed by phase 2.7 but status wasn't)
    tier2_rows = conn.execute(
        """SELECT id, superseded_by FROM concepts
           WHERE currency_status = 'SUPERSEDED'
             AND superseded_by IS NOT NULL
             AND superseded_by != ''
             AND status = 'active'"""
    ).fetchall()

    if tier2_rows:
        for _sid, _superseded_by in tier2_rows:
            if not dry_run:
                apply_lifecycle_transition_conn(
                    conn,
                    _sid,
                    "supersede",
                    superseded_by=_superseded_by,
                    reason="CURRENCY-ACTUATOR Tier 2",
                )
            stats["tier2"] += 1
        logger.info("CURRENCY-ACTUATOR Tier 2: %d concepts → status='superseded'", stats["tier2"])

    # --- Tier 3: CONTRADICTED + low authority + old → archive ---
    # A1 amendment: commit before Tier 3 to avoid dual-connection contention
    # (archive_concept opens its own _db() connection)
    if not dry_run:
        conn.commit()

    tier3_rows = conn.execute(
        """SELECT id FROM concepts
           WHERE currency_status = 'CONTRADICTED'
             AND authority_score < 0.3
             AND status = 'active'
             AND created_at < datetime('now', '-14 days')"""
    ).fetchall()

    if tier3_rows:
        from app.storage import archive_concept

        for row in tier3_rows:
            concept_id = row[0]
            if dry_run:
                stats["tier3"] += 1
                continue
            # A3 amendment: check archive_concept return value; CURRENCY-006: try/except prevents actuator crash
            try:
                success = archive_concept(concept_id)
            except Exception as exc:
                stats["tier3_failed"] += 1
                logger.error("CURRENCY-ACTUATOR Tier 3: archive_concept(%s) raised: %s", concept_id, exc)
                continue
            if success:
                stats["tier3"] += 1
            else:
                stats["tier3_failed"] += 1
                logger.warning("CURRENCY-ACTUATOR Tier 3: archive_concept(%s) returned False", concept_id)

        logger.info(
            "CURRENCY-ACTUATOR Tier 3: %d concepts archived, %d failed",
            stats["tier3"],
            stats["tier3_failed"],
        )

    total = stats["tier1"] + stats["tier2"] + stats["tier3"]
    if total > 0:
        logger.info(
            "CURRENCY-ACTUATOR summary: T1=%d T2=%d T3=%d (total=%d)",
            stats["tier1"], stats["tier2"], stats["tier3"], total,
        )
    return stats


async def _phase2_9_pbc_reconcile(conn, dry_run: bool = False) -> dict:
    """Phase 2.9: PBC reconciliation — mark PRESENT_BOTH_CONTESTED concepts as CONTESTED.

    MAINT-004: Concepts that appear in governance_events with PRESENT_BOTH_CONTESTED
    should be promoted to currency_status='CONTESTED' if not already CONTRADICTED/CONTESTED.
    """
    rows = conn.execute(
        """SELECT DISTINCT concept_id FROM governance_events
           WHERE details LIKE '%PRESENT_BOTH_CONTESTED%'"""
    ).fetchall()
    ids = [r[0] for r in rows]
    updated = 0
    if ids and not dry_run:
        conn.execute(
            f"""UPDATE concepts SET currency_status='CONTESTED'
                WHERE id IN ({','.join('?' * len(ids))})
                AND currency_status NOT IN ('CONTRADICTED','CONTESTED')""",
            ids,
        )
        updated = conn.execute("SELECT changes()").fetchone()[0]
    elif ids and dry_run:
        updated = len(ids)
    logger.info("MAINT-004 PBC reconcile: %d concepts → CONTESTED (dry_run=%s)", updated, dry_run)
    return {"pbc_reconciled": updated, "status": "ok"}


async def _phase2_10_ghost_superseder_cleanup(conn, dry_run: bool = False) -> dict:
    """Phase 2.10: Ghost-superseder cleanup — NULL out dangling superseded_by refs.

    DATA-063: 83 concepts have superseded_by pointing to concept IDs that no longer
    exist in the concepts table. fix_supersession_desync correctly skips these (no live
    superseder to apply), leaving them stuck as is_current=1 with a dangling ref.
    This sub-phase NULLs out those ghost refs so the concepts are cleanly current.
    """
    rows = conn.execute(
        """SELECT COUNT(*) FROM concepts
           WHERE superseded_by IS NOT NULL
             AND superseded_by != :sentinel
             AND superseded_by NOT IN (SELECT id FROM concepts)""",
        {"sentinel": ORPHANED_SUPERSESSION_SENTINEL},
    ).fetchone()
    ghost_count = rows[0] if rows else 0
    fixed = 0
    if ghost_count and not dry_run:
        conn.execute(
            """UPDATE concepts SET superseded_by = NULL
               WHERE superseded_by IS NOT NULL
                 AND superseded_by != :sentinel
                 AND superseded_by NOT IN (SELECT id FROM concepts)""",
            {"sentinel": ORPHANED_SUPERSESSION_SENTINEL},
        )
        fixed = conn.execute("SELECT changes()").fetchone()[0]
    elif ghost_count and dry_run:
        fixed = ghost_count
    logger.info(
        "DATA-063 ghost-superseder cleanup: %d concepts fixed (dry_run=%s)", fixed, dry_run
    )
    return {"ghost_superseders_fixed": fixed, "status": "ok"}


# =============================================================================
# Phase 3: Experiment Generation
# =============================================================================


async def phase3_experiments(conn=None, dry_run: bool = False) -> dict:
    """Generate one experiment of each type if corpus is sufficient.

    Types: cross_domain_synthesis, hypothesis_generation,
           counterfactual, analogy_detection.

    Also archives stale completed experiments.
    """
    from app.features.experiments import (
        _load_experiment_corpus,
        archive_stale_experiments,
        generate_experiment,
    )

    # MAINT-026: Run blocking corpus load in thread executor so asyncio.wait_for
    # can cancel this phase if it exceeds PHASE_TIMEOUT_SECONDS. Previously the
    # preamble loaded 6000+ concept objects synchronously with no await point,
    # making the outer timeout ineffective. concepts_only=dry_run preserves the
    # pre-fix dry_run behavior (fast path: load concepts only, skip assocs/TFIDFCache).
    loop = asyncio.get_running_loop()
    concepts, associations, assoc_counts, salience_ranks, tfidf_cache = (
        await loop.run_in_executor(
            None,
            functools.partial(_load_experiment_corpus, concepts_only=dry_run),
        )
    )

    if dry_run:
        return {"dry_run": True, "corpus_size": len(concepts)}

    experiment_types = [
        "cross_domain_synthesis",
        "hypothesis_generation",
        "counterfactual",
        "analogy_detection",
    ]

    results = {}
    _phase_timings = {}  # DEBT-004: per-type timing
    loop = asyncio.get_running_loop()
    for exp_type in experiment_types:
        _s = time.monotonic()
        try:
            experiment = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    functools.partial(
                        generate_experiment,
                        experiment_type=exp_type,
                        concepts=concepts,
                        associations=associations,
                        assoc_counts=dict(assoc_counts),
                        salience_ranks=salience_ranks,
                        tfidf_cache=tfidf_cache,
                    ),
                ),
                timeout=EXP_TYPE_TIMEOUT_SECONDS,
            )
            if experiment is None:
                # EXP-001: dedup gate blocked this experiment
                results[exp_type] = {"status": "duplicate_skipped"}
                _phase_timings[exp_type] = round((time.monotonic() - _s) * 1000, 1)
                continue
            results[exp_type] = {
                "status": experiment.status,
                "candidates": len(experiment.candidates) if experiment.candidates else 0,
            }
        except TimeoutError:
            results[exp_type] = {"status": "timeout", "timeout_seconds": EXP_TYPE_TIMEOUT_SECONDS}
            logger.error(
                "phase3_experiments: %s TIMED OUT after %ds — corpus cap may need adjustment",
                exp_type,
                EXP_TYPE_TIMEOUT_SECONDS,
            )
        except Exception as e:
            results[exp_type] = {"status": "error", "error": str(e)}
        _phase_timings[exp_type] = round((time.monotonic() - _s) * 1000, 1)

    # Archive stale experiments
    _s = time.monotonic()
    try:
        archived = archive_stale_experiments()
        results["archived_experiments"] = archived
    except Exception as e:
        results["archive_error"] = str(e)
    _phase_timings["archive"] = round((time.monotonic() - _s) * 1000, 1)

    total_ms = sum(_phase_timings.values())
    logger.info("DEBT-004: phase3_experiments %.1fms — %s", total_ms, _phase_timings)
    results["_timings"] = _phase_timings

    return results


# =============================================================================
# Phase 4: Curiosity — Question Generation
# =============================================================================


async def phase4_curiosity(conn=None, dry_run: bool = False) -> dict:
    """Generate questions for weak/uncertain concepts.

    Feeds Pith_questions queue so the next conversation_turn
    can surface gaps to the AI for investigation.
    """
    import app.features.question_queue as question_queue
    from app.features.curiosity import CuriosityEngine

    engine = CuriosityEngine()
    gaps = engine.detect_gaps()

    if dry_run:
        return {"dry_run": True, "gaps_detected": len(gaps)}

    questions_generated = engine.generate_questions()

    # Persist to queue (batch — single load/save cycle)
    question_queue.add_questions(questions_generated)

    return {
        "gaps_detected": len(gaps),
        "questions_generated": len(questions_generated),
    }


# =============================================================================
# Phase 3.5: Experiment Evaluation (RB-01)
# =============================================================================

MIN_CANDIDATE_SCORE = 0.3
LLM_MAX_ATTEMPTS = 3  # [EXP-018-B] Escalate to insufficient_data after N auth/LLM failures


async def phase3_5_evaluate_experiments(conn=None, dry_run: bool = False) -> dict:
    """Resolve experiments via LLM (EXP-003a) or auto-complete fallback (RB-01).

    EXPERIMENT_RESOLUTION_SPEC v1.2:
    - [V-1] 30s time budget with early exit
    - [S-1] Optimistic locking via 'resolving' status
    - [INT-2] Staleness revert for experiments stuck in 'resolving' >5min
    - [I-1] One-time retroactive dedup purge
    - [SF-1] Health check warning if >80% not_meaningful
    """
    from datetime import datetime

    from app.core.models import ExperimentResult
    from app.features.experiments import (
        load_experiments,
        process_experiment_results,
        retroactive_dedup_purge,
        save_experiment,
    )

    BUDGET_SECONDS = 30  # [V-1]
    MAX_PER_RUN = 10
    start = time.monotonic()

    # [INT-2] Revert stale resolving experiments (crash recovery)
    stale_resolving = load_experiments(status=["resolving"], limit=50)
    for exp in stale_resolving:
        if exp.updated_at:
            try:
                age_mins = (_utc_now() - _ensure_aware(datetime.fromisoformat(exp.updated_at))).total_seconds() / 60
                if age_mins > 5:
                    exp.status = "reasoning"
                    exp.updated_at = _utc_now_iso()
                    save_experiment(exp)
                    logger.info("INT-2: Reverted stale resolving experiment %s (%.0fm old)", exp.id[:8], age_mins)
            except (ValueError, TypeError):
                exp.status = "reasoning"
                save_experiment(exp)

    stuck = load_experiments(status=["reasoning"], limit=MAX_PER_RUN)

    if dry_run:
        return {"dry_run": True, "stuck_experiments": len(stuck)}

    # [I-1] One-time retroactive dedup purge (checks DB persistence internally)
    purged = retroactive_dedup_purge()
    if purged > 0:
        logger.info("EXP-001: Retroactive dedup archived %d experiments", purged)
        stuck = load_experiments(status=["reasoning"], limit=MAX_PER_RUN)

    # Check if LLM resolution is available
    llm_available = False
    try:
        from app.features.experiment_llm import check_llm_available, log_health_check, resolve_experiment

        llm_available = check_llm_available()
    except ImportError:
        logger.info("Phase 3.5: experiment_llm not available, using RB-01 fallback")

    # Load concepts for prompt building (only if LLM available)
    concepts = []
    if llm_available:
        from app.storage import list_concepts, load_concept

        concept_ids = list_concepts()
        for cid in concept_ids:
            c = load_concept(cid, track_access=False)
            if c:
                concepts.append(c)

    llm_resolved = 0
    auto_completed = 0
    not_meaningful = 0
    marked_insufficient = 0
    errors = 0

    for exp in stuck:
        # [V-1] Budget check
        if time.monotonic() - start > BUDGET_SECONDS:
            logger.info(
                "Phase 3.5: Budget exhausted after %d experiments",
                llm_resolved + auto_completed + not_meaningful + errors,
            )
            break

        try:
            if not exp.candidates:
                exp.status = "insufficient_data"
                exp.updated_at = _utc_now_iso()
                save_experiment(exp)
                marked_insufficient += 1
                continue

            if llm_available:
                # [EXP-018-B] Gate: skip LLM if max attempts already exhausted
                attempt_count = exp.metadata.get("llm_attempt_count", 0) if exp.metadata else 0
                if attempt_count >= LLM_MAX_ATTEMPTS:
                    exp.status = "insufficient_data"
                    exp.updated_at = _utc_now_iso()
                    save_experiment(exp)
                    marked_insufficient += 1
                    continue

                # [S-1] Optimistic lock
                exp.status = "resolving"
                exp.updated_at = _utc_now_iso()
                save_experiment(exp)

                try:
                    result = await resolve_experiment(exp, concepts=concepts)
                    if result and result.confidence > 0:
                        process_experiment_results(exp.id, result)
                        llm_resolved += 1
                    elif result:
                        process_experiment_results(exp.id, result)
                        not_meaningful += 1
                    else:
                        # resolve_experiment returned None (no prompt template, etc.)
                        exp.status = "reasoning"
                        save_experiment(exp)
                except Exception as e:
                    logger.warning("EXP-003a: LLM failed for %s: %s", exp.id[:8], e)
                    exp.status = "reasoning"  # [S-1] Revert lock
                    exp.updated_at = _utc_now_iso()
                    # [EXP-018-B] Track attempt count; escalate after LLM_MAX_ATTEMPTS failures
                    if exp.metadata is None:
                        exp.metadata = {}
                    exp.metadata["llm_attempt_count"] = exp.metadata.get("llm_attempt_count", 0) + 1
                    if exp.metadata["llm_attempt_count"] >= LLM_MAX_ATTEMPTS:
                        exp.status = "insufficient_data"
                        marked_insufficient += 1
                        logger.warning(
                            "EXP-018: Experiment %s marked insufficient_data after %d LLM failures",
                            exp.id[:8],
                            LLM_MAX_ATTEMPTS,
                        )
                    save_experiment(exp)
                    # [EXP-018-A] Auth circuit breaker: disable LLM for remaining experiments
                    try:
                        from app.features.experiment_llm import is_llm_auth_failed
                        if is_llm_auth_failed():
                            logger.error(
                                "EXP-018: Auth failure confirmed — disabling LLM for remaining "
                                "%d experiments this run",
                                len(stuck) - (llm_resolved + auto_completed + not_meaningful + marked_insufficient + errors + 1),
                            )
                            llm_available = False
                    except ImportError:
                        pass  # B1 not yet deployed — degrade gracefully
                    errors += 1
            else:
                # Fallback: RB-01 auto-complete (no LLM)
                top = max(exp.candidates, key=lambda c: c.score)

                if top.score < MIN_CANDIDATE_SCORE:
                    exp.status = "insufficient_data"
                    exp.updated_at = _utc_now_iso()
                    save_experiment(exp)
                    marked_insufficient += 1
                    continue

                result = ExperimentResult(
                    synthesis=f"Auto-completed from top candidate (score={top.score:.3f}): {top.rationale}",
                    confidence=round(top.score * 0.8, 3),
                    concepts_produced=[],
                    cko_produced=None,
                    reasoning_trace=f"RB-01 auto-complete: candidate={top.candidate_id}, "
                    f"score={top.score:.3f}, type={exp.experiment_type}",
                )
                process_experiment_results(exp.id, result)
                auto_completed += 1
                logger.info("RB-01: Auto-completed experiment %s (score=%.3f)", exp.id[:8], top.score)

        except Exception as e:
            logger.error("Phase 3.5: Failed to evaluate experiment %s: %s", exp.id[:8], e)
            errors += 1

    # [SF-1] Health check
    total_llm = llm_resolved + not_meaningful
    if total_llm > 0 and llm_available:
        log_health_check(not_meaningful, total_llm)

    return {
        "stuck_found": len(stuck),
        "llm_resolved": llm_resolved,
        "not_meaningful": not_meaningful,
        "auto_completed": auto_completed,
        "marked_insufficient": marked_insufficient,
        "errors": errors,
        "llm_available": llm_available,
        "purged": purged,
        "budget_exhausted": (time.monotonic() - start) > BUDGET_SECONDS,
    }


# =============================================================================
# Phase 5: Health Report + Degradation Alerts
# =============================================================================


async def phase5_health_report(conn, dry_run: bool = False) -> dict:
    """Generate comprehensive health report and flag degradation.

    Checks:
      - Async task degradation (overdue scheduled tasks)
      - Pith health metrics (concept count, confidence distribution)
      - Stale concept ratio
      - Association graph density
      - Recent error rates
    """
    from app.cognitive.reflection import reflection_engine
    from app.ops.async_tasks import ensure_async_tables, get_degraded_tasks, task_runner

    ensure_async_tables(conn)

    health = reflection_engine.analyze_stability()
    task_status = task_runner.get_status(conn)
    degraded = get_degraded_tasks(conn)

    # Compute stale concept ratio
    total = conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]
    stale = conn.execute(
        """SELECT COUNT(*) FROM concepts
           WHERE currency_score < 0.3 AND authority_score >= 0.5"""
    ).fetchone()[0]
    stale_ratio = round(stale / max(total, 1), 3)

    # Association density
    assoc_count = conn.execute("SELECT COUNT(*) FROM associations").fetchone()[0]
    density = round(assoc_count / max(total, 1), 2)

    # SYSTEMIC_FIXES_SPEC v1.1 Fix 5: Index integrity check in health phase
    index_integrity = {}
    try:
        from app.retrieval import retrieval_engine

        index_integrity = retrieval_engine.verify_index_integrity()
    except Exception as idx_err:
        index_integrity = {"error": str(idx_err)}

    # Fix 5: Checkpoint staleness report
    checkpoint_health = {}
    try:
        from datetime import datetime as _dt

        from app.storage import list_checkpoints

        checkpoints = list_checkpoints()
        stale_count = 0
        now = _utc_now()
        for cp in checkpoints:
            updated = cp.get("updated_at")
            if updated:
                try:
                    age = (now - _ensure_aware(_dt.fromisoformat(updated))).days
                    if age > 7:
                        stale_count += 1
                except (ValueError, TypeError):
                    pass
        checkpoint_health = {
            "active_checkpoints": len(checkpoints),
            "stale_checkpoints": stale_count,
        }
    except Exception as cp_err:
        checkpoint_health = {"error": str(cp_err)}

    # RB-02: Auto-close stale reflection entries (>24h without completion)
    stale_reflections_closed = 0
    try:
        if not dry_run:
            now_iso = _utc_now_iso()
            cursor = conn.execute(
                """UPDATE reflection_tracking
                   SET completed_at = ?,
                       concepts_returned = 0,
                       reflection_quality = 'timeout'
                   WHERE completed_at IS NULL
                     AND created_at < datetime('now', '-24 hours')""",
                (now_iso,),
            )
            stale_reflections_closed = cursor.rowcount
            if stale_reflections_closed > 0:
                logger.info("RB-02: Auto-closed %d stale reflection entries", stale_reflections_closed)
    except Exception as refl_err:
        logger.warning("RB-02: Reflection cleanup failed (non-fatal): %s", refl_err)

    # STATS-005: Persist health scores to metrics for trending
    try:
        from app.ops.metrics import metrics as _metrics

        h = health if isinstance(health, dict) else {}
        hf = h.get("health_factors", {})  # MONITOR-071: sub-factors live inside health_factors, not top-level
        _metrics.record("pith_health_score", h.get("health_score", 0))
        _metrics.record("pith_maturity_score", hf.get("maturity", 0))
        _metrics.record("pith_connectivity_score", hf.get("connectivity", 0))
        _metrics.record("pith_confidence_avg", hf.get("confidence", 0))
        _metrics.record("pith_freshness_ratio", hf.get("freshness", 0))
        _metrics.flush()
    except Exception as metric_err:
        logger.warning("STATS-005: Health metric recording failed: %s", metric_err)

    # STATS-004: Check bg_task failure rates
    bg_task_alerts = []
    try:
        from datetime import timedelta

        since_24h = (_utc_now() - timedelta(hours=24)).isoformat()
        bg_rows = conn.execute(
            """SELECT json_extract(labels, '$.task') as task_name, metric, SUM(value)
               FROM metrics
               WHERE metric IN ('bg_task_success', 'bg_task_failure')
                 AND timestamp >= ?
               GROUP BY task_name, metric""",
            (since_24h,),
        ).fetchall()
        task_counts = {}
        for task_name, metric, total_val in bg_rows:
            tn = task_name or "unknown"
            if tn not in task_counts:
                task_counts[tn] = {"success": 0, "failure": 0}
            task_counts[tn][metric.replace("bg_task_", "")] = int(total_val)
        for tn, counts in task_counts.items():
            t = counts["success"] + counts["failure"]
            if t > 0 and counts["failure"] / t > 0.10:
                bg_task_alerts.append(
                    {
                        "task": tn,
                        "failure_rate": round(counts["failure"] / t, 3),
                        "failures": counts["failure"],
                        "total": t,
                    }
                )
    except Exception as bg_err:
        logger.warning("STATS-004: bg_task failure check failed: %s", bg_err)

    return {
        "pith_health": health,
        "concept_count": total,
        "stale_concept_ratio": stale_ratio,
        "association_density": density,
        "degraded_tasks": degraded,
        "bg_task_failure_alerts": bg_task_alerts,
        "task_status_summary": {k: v.get("last_successful_run") for k, v in task_status.get("tasks", {}).items()},
        "index_integrity": index_integrity,
        "checkpoint_health": checkpoint_health,
        "stale_reflections_closed": stale_reflections_closed,
    }


# =============================================================================
# Unified Runner
# =============================================================================

# ---------------------------------------------------------------------------
# PERF-089: Metrics retention and rollup
# ---------------------------------------------------------------------------


def _ensure_metrics_rollups_table(conn) -> None:
    """Create metrics_rollups for upgraded DBs and isolated test fixtures."""
    conn.execute(
        """CREATE TABLE IF NOT EXISTS metrics_rollups (
            bucket_start TEXT NOT NULL,
            bucket_days INTEGER NOT NULL DEFAULT 1,
            metric TEXT NOT NULL,
            count INTEGER NOT NULL,
            sum_value REAL NOT NULL,
            min_value REAL NOT NULL,
            max_value REAL NOT NULL,
            avg_value REAL NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (bucket_start, bucket_days, metric)
        )"""
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_metrics_rollups_metric_bucket
           ON metrics_rollups(metric, bucket_start)"""
    )


async def phase5_4_metrics_retention(
    conn,
    dry_run: bool = False,
    raw_retention_days: int = 30,
    rollup_retention_days: int = 365,
    batch_size: int = 5000,
) -> dict:
    """Roll up old raw metrics, then delete represented raw rows.

    Rollup and raw deletion are committed per day/metric bucket. This keeps the
    operation restart-safe: a bucket is either fully represented in rollups and
    raw rows are gone, or no partial bucket mutation commits.
    """
    from datetime import timedelta

    try:
        from app.ops.metrics import metrics as _metrics

        _metrics.flush()
    except Exception as metric_err:
        logger.warning("PERF-089: metrics flush before retention failed: %s", metric_err)

    if raw_retention_days < 1:
        raw_retention_days = 1
    if rollup_retention_days < raw_retention_days:
        rollup_retention_days = raw_retention_days
    if batch_size < 1:
        batch_size = 5000

    now = datetime.now(UTC)
    now_iso = now.strftime("%Y-%m-%dT%H:%M:%S")
    raw_cutoff = (now - timedelta(days=raw_retention_days)).strftime("%Y-%m-%dT%H:%M:%S")
    rollup_cutoff = (now - timedelta(days=rollup_retention_days)).strftime("%Y-%m-%dT%H:%M:%S")

    bucket_rows = conn.execute(
        """SELECT substr(timestamp, 1, 10) AS bucket_day,
                  metric,
                  COUNT(*) AS sample_count,
                  SUM(value) AS sum_value,
                  MIN(value) AS min_value,
                  MAX(value) AS max_value,
                  AVG(value) AS avg_value
           FROM metrics
           WHERE timestamp < ?
           GROUP BY substr(timestamp, 1, 10), metric
           ORDER BY bucket_day, metric""",
        (raw_cutoff,),
    ).fetchall()
    raw_candidate_count = sum(int(row["sample_count"] if hasattr(row, "keys") else row[2]) for row in bucket_rows)
    rollups_table_exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'metrics_rollups'"
    ).fetchone()
    rollup_prune_count = 0
    if rollups_table_exists:
        rollup_prune_count = conn.execute(
            "SELECT COUNT(*) FROM metrics_rollups WHERE bucket_start < ?",
            (rollup_cutoff,),
        ).fetchone()[0]

    result = {
        "raw_retention_days": raw_retention_days,
        "rollup_retention_days": rollup_retention_days,
        "batch_size": batch_size,
        "raw_cutoff": raw_cutoff,
        "rollup_cutoff": rollup_cutoff,
        "dry_run": dry_run,
        "would_rollup_rows": raw_candidate_count,
        "would_rollup_buckets": len(bucket_rows),
        "would_delete_raw": raw_candidate_count,
        "would_delete_rollups": rollup_prune_count,
        "rolled_up_buckets": 0,
        "raw_deleted": 0,
        "rollups_deleted": 0,
    }
    if dry_run:
        return result

    _ensure_metrics_rollups_table(conn)

    for row in bucket_rows:
        bucket_day = row["bucket_day"] if hasattr(row, "keys") else row[0]
        metric_name = row["metric"] if hasattr(row, "keys") else row[1]
        sample_count = int(row["sample_count"] if hasattr(row, "keys") else row[2])
        sum_value = float(row["sum_value"] if hasattr(row, "keys") else row[3])
        min_value = float(row["min_value"] if hasattr(row, "keys") else row[4])
        max_value = float(row["max_value"] if hasattr(row, "keys") else row[5])
        avg_value = float(row["avg_value"] if hasattr(row, "keys") else row[6])
        bucket_start = f"{bucket_day}T00:00:00"

        conn.execute(
            """INSERT INTO metrics_rollups
               (bucket_start, bucket_days, metric, count, sum_value, min_value,
                max_value, avg_value, created_at, updated_at)
               VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(bucket_start, bucket_days, metric) DO UPDATE SET
                   count = metrics_rollups.count + excluded.count,
                   sum_value = metrics_rollups.sum_value + excluded.sum_value,
                   min_value = MIN(metrics_rollups.min_value, excluded.min_value),
                   max_value = MAX(metrics_rollups.max_value, excluded.max_value),
                   avg_value = (
                       metrics_rollups.sum_value + excluded.sum_value
                   ) / (
                       metrics_rollups.count + excluded.count
                   ),
                   updated_at = excluded.updated_at""",
            (
                bucket_start,
                metric_name,
                sample_count,
                sum_value,
                min_value,
                max_value,
                avg_value,
                now_iso,
                now_iso,
            ),
        )

        deleted_for_bucket = 0
        while True:
            cursor = conn.execute(
                """DELETE FROM metrics
                   WHERE rowid IN (
                       SELECT rowid FROM metrics
                       WHERE timestamp < ?
                         AND substr(timestamp, 1, 10) = ?
                         AND metric = ?
                       LIMIT ?
                   )""",
                (raw_cutoff, bucket_day, metric_name, batch_size),
            )
            batch_count = cursor.rowcount
            deleted_for_bucket += batch_count
            if batch_count < batch_size:
                break

        conn.commit()
        result["rolled_up_buckets"] += 1
        result["raw_deleted"] += deleted_for_bucket

    if rollup_prune_count:
        cursor = conn.execute(
            "DELETE FROM metrics_rollups WHERE bucket_start < ?",
            (rollup_cutoff,),
        )
        result["rollups_deleted"] = cursor.rowcount
        conn.commit()

    if result["raw_deleted"] or result["rollups_deleted"]:
        logger.info(
            "PERF-089: metrics retention rolled_up_buckets=%d raw_deleted=%d rollups_deleted=%d",
            result["rolled_up_buckets"],
            result["raw_deleted"],
            result["rollups_deleted"],
        )
    return result


# ---------------------------------------------------------------------------
# CASCADE-001 A1.3: Governance events retention
# ---------------------------------------------------------------------------


async def phase5_5_governance_retention(
    conn, dry_run: bool = False,
    retention_days: int = 90,
    recal_retention_days: int = 7,
    contradiction_retention_days: int = 30,
) -> dict:
    """Three-tier governance events retention (CASCADE-001 A1.3 + MAINT-038 + MAINT-056).

    Tier 1 (7d):  CONFIDENCE_RECALIBRATION, CONFIDENCE_RECALIBRATION_SUMMARY — high-volume, low-value
    Tier 2 (30d): CONTRADICTION_DETECTED, GRAPH_CONTRADICTION_SIGNAL, CONTRADICTION_PHASE_2_COMPLETED
    Tier 3 (90d): authority_reinforcement — original CASCADE-001 retention
    """
    from datetime import datetime, timedelta

    from app.core.constants import (
        GOV_EVENT_AUTHORITY_REINFORCEMENT,
        GOV_EVENT_CONFIDENCE_RECALIBRATION,
        GOV_EVENT_CONFIDENCE_RECALIBRATION_SUMMARY,
        GOV_EVENT_CONTRADICTION_DETECTED,
        GOV_EVENT_CONTRADICTION_PHASE_2_COMPLETED,
        GOV_EVENT_GRAPH_CONTRADICTION_SIGNAL,
    )

    # MAINT-038: Use isoformat() to match stored timestamp format (T separator, not space)
    # Pre-existing bug: datetime.now(UTC) produces aware datetime whose .isoformat()
    # includes +00:00 suffix — strip it for consistent comparison with stored UTC timestamps.
    def _iso_cutoff(days: int) -> str:
        return (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")

    cutoff_7d = _iso_cutoff(recal_retention_days)
    cutoff_30d = _iso_cutoff(contradiction_retention_days)
    cutoff_90d = _iso_cutoff(retention_days)

    tier1_types = (GOV_EVENT_CONFIDENCE_RECALIBRATION, GOV_EVENT_CONFIDENCE_RECALIBRATION_SUMMARY)
    tier2_types = (GOV_EVENT_CONTRADICTION_DETECTED, GOV_EVENT_GRAPH_CONTRADICTION_SIGNAL,
                   GOV_EVENT_CONTRADICTION_PHASE_2_COMPLETED)

    result = {
        "retention_days": retention_days,
        "recal_retention_days": recal_retention_days,
        "contradiction_retention_days": contradiction_retention_days,
        "tier1_deleted": 0, "tier2_deleted": 0, "tier3_deleted": 0,
    }

    if dry_run:
        result["tier1_would_delete"] = conn.execute(
            "SELECT COUNT(*) FROM governance_events WHERE created_at < ? AND event_type IN (?, ?)",
            (cutoff_7d, *tier1_types),
        ).fetchone()[0]
        result["tier2_would_delete"] = conn.execute(
            "SELECT COUNT(*) FROM governance_events WHERE created_at < ? AND event_type IN (?, ?, ?)",
            (cutoff_30d, *tier2_types),
        ).fetchone()[0]
        result["tier3_would_delete"] = conn.execute(
            "SELECT COUNT(*) FROM governance_events WHERE created_at < ? AND event_type = ?",
            (cutoff_90d, GOV_EVENT_AUTHORITY_REINFORCEMENT),
        ).fetchone()[0]
        # Backward compat: tests and callers may check "would_delete" for tier3 (original key)
        result["would_delete"] = result["tier3_would_delete"]
        return result

    # Tier 1: 7-day retention for recalibration events (GOV-005)
    # Batched deletion to avoid long table locks on high-volume events (~1M+/month)
    batch_size = 5000
    tier1_total = 0
    while True:
        c1 = conn.execute(
            "DELETE FROM governance_events WHERE rowid IN ("
            "  SELECT rowid FROM governance_events"
            "  WHERE created_at < ? AND event_type IN (?, ?)"
            "  LIMIT ?"
            ")",
            (cutoff_7d, *tier1_types, batch_size),
        )
        batch_count = c1.rowcount
        if batch_count > 0:
            conn.commit()
        tier1_total += batch_count
        if batch_count < batch_size:
            break
    result["tier1_deleted"] = tier1_total

    # Tier 2: 30-day retention for contradiction events (MAINT-056)
    c2 = conn.execute(
        "DELETE FROM governance_events WHERE created_at < ? AND event_type IN (?, ?, ?)",
        (cutoff_30d, *tier2_types),
    )
    result["tier2_deleted"] = c2.rowcount

    # Tier 3: 90-day retention for authority reinforcement (original CASCADE-001)
    c3 = conn.execute(
        "DELETE FROM governance_events WHERE created_at < ? AND event_type = ?",
        (cutoff_90d, GOV_EVENT_AUTHORITY_REINFORCEMENT),
    )
    result["tier3_deleted"] = c3.rowcount

    total = result["tier1_deleted"] + result["tier2_deleted"] + result["tier3_deleted"]
    if total > 0:
        conn.commit()
        logger.info(
            "MAINT-038/056: governance retention — tier1(7d)=%d, tier2(30d)=%d, tier3(90d)=%d",
            result["tier1_deleted"], result["tier2_deleted"], result["tier3_deleted"],
        )
    result["deleted"] = total  # backward compat key
    return result


async def phase5_6_test_canary_cleanup(conn, dry_run: bool = False) -> dict:
    """Archive accumulated test_zp_integration_* canary concepts (DEBT-199).

    Zero-protocol integration tests write ephemeral probe concepts to the DB
    but have no API endpoint to clean up after themselves. This phase archives
    any concepts whose ID begins with 'test_zp_integration_', preventing
    long-term accumulation in pith stats and retrieval results.
    """
    result = {"archived": 0, "dry_run": dry_run}

    canary_rows = conn.execute(
        "SELECT id FROM concepts WHERE id LIKE 'test_zp_integration_%' AND status != 'archived'"
    ).fetchall()

    if dry_run:
        result["would_archive"] = len(canary_rows)
        return result

    from app.storage import _invalidate_associations_cache, apply_lifecycle_transition_conn

    archived = 0
    for row in canary_rows:
        archived += apply_lifecycle_transition_conn(conn, row["id"], "archive")

    if archived > 0:
        conn.commit()
        _invalidate_associations_cache()
        logger.info("DEBT-199: archived %d test_zp_integration_* canary concepts", archived)

    result["archived"] = archived
    return result


async def phase5_7_incremental_vacuum(conn, dry_run: bool = False) -> dict:
    """MAINT-030: Reclaim freed pages after governance retention pruning.

    Uses incremental_vacuum to reclaim up to 500 pages (~2MB) per maintenance
    cycle. This avoids the long lock of a full VACUUM while still preventing
    monotonic DB growth from deleted governance events.

    Pre-requisite: auto_vacuum must be set to 2 (incremental) on the DB.
    If auto_vacuum=0, incremental_vacuum is a no-op and this logs a warning.
    """
    auto_vacuum_mode = conn.execute("PRAGMA auto_vacuum").fetchone()[0]
    freelist_before = conn.execute("PRAGMA freelist_count").fetchone()[0]

    result = {
        "auto_vacuum_mode": auto_vacuum_mode,
        "freelist_before": freelist_before,
        "pages_freed": 0,
        "dry_run": dry_run,
    }

    if auto_vacuum_mode != 2:
        # incremental_vacuum requires auto_vacuum=2 (incremental mode)
        # If not set, log warning but don't fail — this is a soft dependency
        if freelist_before > 0:
            logger.warning(
                "MAINT-030: auto_vacuum=%d (need 2 for incremental_vacuum), "
                "%d freelist pages cannot be reclaimed. "
                "Run: PRAGMA auto_vacuum=2; VACUUM; to enable.",
                auto_vacuum_mode, freelist_before,
            )
        result["skipped"] = "auto_vacuum not set to incremental (2)"
        return result

    if dry_run:
        result["would_free"] = min(freelist_before, 500)
        return result

    if freelist_before > 0:
        conn.execute("PRAGMA incremental_vacuum(500)")
        freelist_after = conn.execute("PRAGMA freelist_count").fetchone()[0]
        result["pages_freed"] = freelist_before - freelist_after
        if result["pages_freed"] > 0:
            logger.info(
                "MAINT-030: incremental_vacuum freed %d pages (%d remain)",
                result["pages_freed"], freelist_after,
            )

    return result


async def phase5_9_full_vacuum_if_needed(conn, dry_run: bool = False) -> dict:
    """MAINT-039: Full VACUUM when freelist pages exceed threshold.

    Works regardless of auto_vacuum mode (unlike incremental_vacuum which
    requires auto_vacuum=2). Acquires exclusive lock for ~2-5s on a 225MB DB
    — acceptable cost for maintenance cycle cleanup.

    Only runs when freelist_count > VACUUM_FREELIST_THRESHOLD_PAGES (~20MB).
    Safe to call from autocommit connection (isolation_level=None).
    """
    from app.core.config import VACUUM_FREELIST_THRESHOLD_PAGES

    freelist = conn.execute("PRAGMA freelist_count").fetchone()[0]
    result = {
        "freelist_before": freelist,
        "threshold": VACUUM_FREELIST_THRESHOLD_PAGES,
        "vacuumed": False,
        "dry_run": dry_run,
    }

    if freelist < VACUUM_FREELIST_THRESHOLD_PAGES:
        result["skipped"] = f"freelist {freelist} below threshold {VACUUM_FREELIST_THRESHOLD_PAGES}"
        return result

    if dry_run:
        result["would_vacuum"] = True
        return result

    conn.execute("VACUUM")
    freelist_after = conn.execute("PRAGMA freelist_count").fetchone()[0]
    result["vacuumed"] = True
    result["pages_freed"] = freelist - freelist_after
    logger.info(
        "MAINT-039: VACUUM freed %d pages (%d remain in freelist)",
        result["pages_freed"],
        freelist_after,
    )
    return result


# =============================================================================
# Phase 6: Standalone Promotion Sweep (ARCH-D05)
# =============================================================================


async def phase6_promotion_sweep(conn=None, dry_run: bool = False) -> dict:
    """ARCH-D05: Standalone promotion sweep — decoupled from reflection timeout.

    Runs promotion independently so that even when phase2_reflection
    times out (79.6% of cycles as of 2026-03-18), provisional concepts
    still get promoted. Lightweight: ~2-5s for ~700 provisional concepts
    vs 120s+ for full reflection on 6,000+ concepts.

    Also includes M3 compliance self-healing (STABILITY-027): caps any
    quarantined concepts that drifted above the 0.4 confidence ceiling.
    """
    from app.cognitive.reflection import run_standalone_promotion

    if dry_run:
        return {"dry_run": True, "description": "Would run standalone promotion sweep + M3 sweep"}

    loop = asyncio.get_running_loop()
    promoted = await loop.run_in_executor(None, run_standalone_promotion)

    # STABILITY-027: M3 compliance self-healing sweep
    m3_capped = 0
    try:
        from app.core.config import PSIS_QUARANTINE_CONFIDENCE_CAP
        from app.storage import load_concept, save_concept

        with owned_connection() as conn:
            rows = conn.execute(
                """SELECT id, confidence FROM concepts
                   WHERE is_current = 1
                   AND maturity = 'QUARANTINED'
                   AND confidence > ?""",
                (PSIS_QUARANTINE_CONFIDENCE_CAP,),
            ).fetchall()
        for row in rows:
            cid = row["id"] if isinstance(row, dict) else row[0]
            concept = load_concept(cid, track_access=False)
            if concept and concept.confidence > PSIS_QUARANTINE_CONFIDENCE_CAP:
                original_ka = concept.metadata.get("knowledge_area") if concept.metadata else None
                concept.confidence = PSIS_QUARANTINE_CONFIDENCE_CAP
                if original_ka:
                    concept.knowledge_area = original_ka
                    if concept.metadata:
                        concept.metadata["knowledge_area"] = original_ka
                save_concept(concept)
                m3_capped += 1
        if m3_capped > 0:
            logger.info("STABILITY-027: M3 sweep capped %d quarantined concepts to %.1f",
                        m3_capped, PSIS_QUARANTINE_CONFIDENCE_CAP)
    except Exception as e:
        logger.warning("STABILITY-027: M3 sweep failed (non-fatal): %s", e)

    return {"promoted_count": promoted, "m3_capped": m3_capped}


async def _phase2_11_association_quality(conn, dry_run: bool = False) -> dict:
    """RETRIEVAL-080 Phase D: Score association edges by utility correlation.

    Evaluates whether connected concepts share utility patterns. If concept A
    is consistently USED and concept B (connected via association) is consistently
    UNUSED when co-activated, the edge is noise.

    Metrics computed:
    - edge_utility_correlation: Do connected concepts have correlated utilization?
    - weak_edge_count: Edges where source high-utility but target low-utility
    - cross_ka_utility: Are cross-KA bridges higher or lower utility than same-KA?
    """
    from app.core.config import MIN_UTILITY_SAMPLES, get_feature_flag

    if not get_feature_flag("FEEDBACK_LOOP_ENABLED", True):
        return {"status": "skipped", "reason": "feedback_loop_disabled"}

    if dry_run:
        return {"status": "dry_run", "phase": "association_quality"}

    try:
        # Query associations with utility data for both endpoints
        rows = conn.execute("""
            SELECT a.source, a.target, a.mechanism,
                   cs.utility_score AS src_utility, cs.utility_samples AS src_samples,
                   cs.knowledge_area AS src_ka,
                   ct.utility_score AS tgt_utility, ct.utility_samples AS tgt_samples,
                   ct.knowledge_area AS tgt_ka
            FROM associations a
            JOIN concepts cs ON cs.id = a.source AND cs.status = 'active' AND cs.is_current = 1
            JOIN concepts ct ON ct.id = a.target AND ct.status = 'active' AND ct.is_current = 1
            WHERE cs.utility_samples >= ? AND ct.utility_samples >= ?
        """, (MIN_UTILITY_SAMPLES, MIN_UTILITY_SAMPLES)).fetchall()

        if not rows:
            return {"status": "ok", "edges_evaluated": 0, "message": "no edges with sufficient utility data"}

        weak_edges = []
        same_ka_utilities = []
        cross_ka_utilities = []
        utility_diffs = []

        for r in rows:
            src_util = r["src_utility"] or 0.5
            tgt_util = r["tgt_utility"] or 0.5
            src_ka = r["src_ka"] or "general"
            tgt_ka = r["tgt_ka"] or "general"
            diff = abs(src_util - tgt_util)
            utility_diffs.append(diff)

            avg_util = (src_util + tgt_util) / 2.0
            if src_ka == tgt_ka:
                same_ka_utilities.append(avg_util)
            else:
                cross_ka_utilities.append(avg_util)

            # Weak edge: one endpoint high utility (>0.6), other low (<0.3)
            if (src_util > 0.6 and tgt_util < 0.3) or (tgt_util > 0.6 and src_util < 0.3):
                weak_edges.append({
                    "source": r["source"], "target": r["target"],
                    "src_utility": round(src_util, 3), "tgt_utility": round(tgt_util, 3),
                    "mechanism": r["mechanism"],
                })

        result = {
            "status": "ok",
            "edges_evaluated": len(rows),
            "weak_edge_count": len(weak_edges),
            "mean_utility_diff": round(sum(utility_diffs) / len(utility_diffs), 4) if utility_diffs else 0,
            "same_ka_mean_utility": round(sum(same_ka_utilities) / len(same_ka_utilities), 4) if same_ka_utilities else None,
            "cross_ka_mean_utility": round(sum(cross_ka_utilities) / len(cross_ka_utilities), 4) if cross_ka_utilities else None,
            "same_ka_edges": len(same_ka_utilities),
            "cross_ka_edges": len(cross_ka_utilities),
        }

        if weak_edges:
            # Log top 10 weakest edges for observability
            result["weak_edges_sample"] = weak_edges[:10]
            logger.info(f"RETRIEVAL-080: Found {len(weak_edges)} weak association edges out of {len(rows)} evaluated")

        return result

    except Exception as e:
        logger.warning(f"RETRIEVAL-080: Association quality scoring failed: {e}")
        return {"status": "error", "error": str(e)}


# =============================================================================
# OPS-154: Backup alert webhook with escalating cooldown
# =============================================================================

# Escalating cooldown tiers (seconds): 2h → 6h → 24h
_BACKUP_ALERT_COOLDOWNS = [2 * 3600, 6 * 3600, 24 * 3600]


def _ensure_alert_table(mconn) -> None:
    """Create backup_alert_state table if it doesn't exist."""
    mconn.execute("""
        CREATE TABLE IF NOT EXISTS backup_alert_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            last_alert_utc TEXT,
            consecutive_alerts INTEGER DEFAULT 0,
            last_status TEXT,
            updated_utc TEXT
        )
    """)
    # Ensure exactly one row exists
    if mconn.execute("SELECT COUNT(*) FROM backup_alert_state").fetchone()[0] == 0:
        mconn.execute(
            "INSERT INTO backup_alert_state (id, consecutive_alerts) VALUES (1, 0)"
        )
    mconn.commit()


def _maybe_send_backup_alert(data_dir, backup_result: dict) -> None:
    """OPS-154: Send webhook alert if backup is unhealthy and cooldown allows.

    Escalating cooldown: 2h → 6h → 24h. Resets on successful backup.
    Opt-in via PITH_BACKUP_ALERT_WEBHOOK_URL env var.
    """
    import sqlite3 as _sqlite3
    from datetime import datetime as _dt

    webhook_url = os.environ.get("PITH_BACKUP_ALERT_WEBHOOK_URL")
    if not webhook_url:
        return  # Opt-in: no URL = no alerts

    status = backup_result.get("status", "error")
    metrics_db = str(data_dir / "metrics.db")

    try:
        mconn = _sqlite3.connect(metrics_db)
        mconn.row_factory = _sqlite3.Row
        _ensure_alert_table(mconn)

        row = mconn.execute("SELECT * FROM backup_alert_state WHERE id = 1").fetchone()
        consecutive = row["consecutive_alerts"]
        last_alert_str = row["last_alert_utc"]

        if status == "ok":
            # Reset cooldown on success
            mconn.execute("""
                UPDATE backup_alert_state
                SET consecutive_alerts = 0, last_status = 'ok', updated_utc = ?
                WHERE id = 1
            """, (_utc_now_iso(),))
            mconn.commit()
            mconn.close()
            return

        # Determine if cooldown has elapsed
        if last_alert_str:
            last_alert = _ensure_aware(_dt.fromisoformat(last_alert_str))
            cooldown_tier = min(consecutive, len(_BACKUP_ALERT_COOLDOWNS) - 1)
            cooldown_secs = _BACKUP_ALERT_COOLDOWNS[cooldown_tier]
            elapsed = (_utc_now() - last_alert).total_seconds()
            if elapsed < cooldown_secs:
                logger.debug(
                    "OPS-154: Backup alert suppressed — %ds of %ds cooldown remaining",
                    cooldown_secs - elapsed,
                    cooldown_secs,
                )
                mconn.close()
                return

        # Send the alert
        _fire_webhook(webhook_url, backup_result)

        # Update cooldown state
        mconn.execute("""
            UPDATE backup_alert_state
            SET last_alert_utc = ?, consecutive_alerts = ?, last_status = ?, updated_utc = ?
            WHERE id = 1
        """, (_utc_now_iso(), consecutive + 1, status, _utc_now_iso()))
        mconn.commit()
        mconn.close()

    except Exception as e:
        logger.warning("OPS-154: Backup alert failed (non-fatal): %s", e)


def _fire_webhook(url: str, backup_result: dict) -> None:
    """POST backup alert payload to webhook URL. 5s timeout, fire-and-forget."""
    import urllib.request

    payload = json.dumps({
        "event": "pith_backup_alert",
        "status": backup_result.get("status", "unknown"),
        "error": backup_result.get("error"),
        "elapsed_seconds": backup_result.get("elapsed_seconds"),
        "backup_size_bytes": backup_result.get("backup_size_bytes"),
        "concept_count": backup_result.get("concept_count"),
        "integrity": backup_result.get("integrity"),
        "timestamp": _utc_now_iso(),
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            logger.info("OPS-154: Backup alert sent — HTTP %d", resp.status)
    except Exception as e:
        logger.warning("OPS-154: Webhook POST failed: %s", e)


async def _phase7_concept_synthesis(conn=None, dry_run: bool = False) -> dict:
    """SKILL-DEPLOY-001: Run concept synthesis LLM pass."""
    if dry_run:
        return {"status": "dry_run", "phase": "concept_synthesis"}
    try:
        from app.cognitive.synthesis_llm import run_synthesis
        return await run_synthesis()
    except Exception as e:
        logger.warning(f"Phase 7 concept_synthesis failed (non-fatal): {e}")
        return {"status": "error", "error": str(e)}


async def phase8_backup(conn=None, dry_run: bool = False) -> dict:
    """OPS-152: WAL-safe database backup using Python's native sqlite3 backup API.

    Cross-platform replacement for shell-based backup. Uses sqlite3.Connection.backup()
    which is WAL-safe by design — no manual checkpoint needed.

    Backup destination: {data_dir}/pith_backup.db
    Strategy: pages=-1 (single-shot copy), ~2-5s for 500MB DB on SSD.
    """
    import sqlite3 as _sqlite3

    from app.core.profile import resolve_data_dir

    data_dir = resolve_data_dir()
    source_path = data_dir / "pith.db"
    backup_path = data_dir / "pith_backup.db"
    tmp_path = data_dir / "pith_backup.db.tmp"

    result = {
        "phase": "backup",
        "source": str(source_path),
        "destination": str(backup_path),
        "dry_run": dry_run,
    }

    if dry_run:
        result["status"] = "dry_run"
        result["source_size_bytes"] = source_path.stat().st_size if source_path.exists() else 0
        return result

    if not source_path.exists():
        result["status"] = "skipped"
        result["reason"] = "source DB does not exist"
        logger.warning("OPS-152: Backup skipped — %s does not exist", source_path)
        return result

    t0 = time.monotonic()
    try:
        # Open source as read-only to avoid interfering with maintenance conn
        src = _sqlite3.connect(f"file:{source_path}?mode=ro", uri=True)
        try:
            # Write to tmp first, rename on success (atomic on POSIX)
            dst = _sqlite3.connect(str(tmp_path))
            try:
                src.backup(dst, pages=-1)  # Single-shot: WAL-safe, ~2-5s for 500MB
                dst.close()
                # Atomic replace — if this fails, tmp is left for next cycle
                tmp_path.replace(backup_path)  # .replace() is atomic on both POSIX and Windows
            except Exception:
                dst.close()
                # Clean up failed tmp
                if tmp_path.exists():
                    tmp_path.unlink()
                raise
        finally:
            src.close()

        elapsed = round(time.monotonic() - t0, 2)
        backup_size = backup_path.stat().st_size

        # Integrity quick-check on the backup
        verify_conn = _sqlite3.connect(f"file:{backup_path}?mode=ro", uri=True)
        try:
            integrity = verify_conn.execute("PRAGMA quick_check").fetchone()[0]
            concept_count = verify_conn.execute(
                "SELECT COUNT(*) FROM concepts WHERE is_current = 1"
            ).fetchone()[0]
        finally:
            verify_conn.close()

        result.update({
            "status": "ok",
            "elapsed_seconds": elapsed,
            "backup_size_bytes": backup_size,
            "concept_count": concept_count,
            "integrity": integrity,
        })
        logger.info(
            "OPS-152: Backup complete — %d concepts, %.1f MB, %ss, integrity=%s",
            concept_count,
            backup_size / 1024 / 1024,
            elapsed,
            integrity,
        )

        # OPS-154: Check backup health and send webhook alert if needed
        _maybe_send_backup_alert(data_dir, result)

    except Exception as e:
        elapsed = round(time.monotonic() - t0, 2)
        result.update({
            "status": "error",
            "error": str(e),
            "elapsed_seconds": elapsed,
        })
        logger.error("OPS-152: Backup failed after %ss: %s", elapsed, e, exc_info=True)

        # Alert on backup failure too
        _maybe_send_backup_alert(data_dir, result)

    return result


async def _phase9_fts_verbatim_repair(conn, dry_run: bool = False) -> dict:
    """DATA-069: Repair missing FTS verbatim entries.

    Finds canonical conversation fragments not in fts_verbatim and backfills them.
    Runs each maintenance cycle to catch insert-time sync failures.
    """
    from app.storage import repair_fts_verbatim

    if dry_run:
        # Count missing without repairing
        missing_count = conn.execute("""
            SELECT COUNT(*) FROM verbatim_fragments vf
            JOIN concepts c ON c.id = vf.concept_id
            WHERE vf.content IS NOT NULL AND vf.char_count > 0
            AND vf.fragment_type = 'conversation'
            AND c.status = 'active'
            AND NOT EXISTS (
                SELECT 1 FROM fts_verbatim fts WHERE fts.fragment_id = vf.id
            )
        """).fetchone()[0]
        return {"dry_run": True, "missing": missing_count}

    result = repair_fts_verbatim(limit=200)
    return result


# Phase ordering: contradiction_sweep (2.5) runs after reflection (2)
# We use _PHASE_ORDER to control execution sequence with ALL_PHASES for dispatch.
ALL_PHASES = {
    1: ("scheduled_tasks", phase1_scheduled_tasks),
    2: ("reflection", phase2_reflection),
    3: ("experiments", phase3_experiments),
    4: ("curiosity", phase4_curiosity),
    5: ("health_report", phase5_health_report),
    6: ("promotion_sweep", phase6_promotion_sweep),  # ARCH-D05
    7: ("concept_synthesis", _phase7_concept_synthesis),  # SKILL-DEPLOY-001
    8: ("backup", phase8_backup),  # OPS-152
    9: ("fts_verbatim_repair", _phase9_fts_verbatim_repair),  # DATA-069
}

# Sub-phases that auto-run after their parent phase
async def phase5_10_checkpoint_gc(conn=None, dry_run: bool = False) -> dict:
    """CKPT-006: Scheduled checkpoint garbage collection.

    Runs both cleanup mechanisms:
    1. cleanup_expired_checkpoints() — hard delete past TTL
    2. archive_stale_checkpoints() — soft archive >48h no update

    Covers checkpoints from crashed sessions that never hit session_end.
    """
    from app.storage import archive_stale_checkpoints, cleanup_expired_checkpoints

    if dry_run:
        return {"status": "dry_run", "phase": "checkpoint_gc"}

    expired = cleanup_expired_checkpoints()
    archived = archive_stale_checkpoints()

    result = {"expired_deleted": expired, "stale_archived": archived}
    if expired or archived:
        logger.info(f"CKPT-006: Checkpoint GC — deleted {expired} expired, archived {archived} stale")
    return result


_SUB_PHASES = {
    2: [
        ("auto_associate", phase2_4_auto_associate),
        ("contradiction_sweep", phase2_5_contradiction_sweep),
        ("episode_retention", phase2_6_episode_retention),  # INFRA-002B
        ("fix_supersession_desync", _phase2_7_fix_supersession_desync),  # DATA-048
        ("currency_actuator", _phase2_8_currency_actuator),  # CURRENCY-ACTUATOR
        ("pbc_reconcile", _phase2_9_pbc_reconcile),  # MAINT-004
        ("ghost_superseder_cleanup", _phase2_10_ghost_superseder_cleanup),  # DATA-063
        ("association_quality", _phase2_11_association_quality),  # RETRIEVAL-080
    ],
    3: [
        ("evaluate_experiments", phase3_5_evaluate_experiments),
    ],
    5: [
        ("metrics_retention", phase5_4_metrics_retention),  # PERF-089
        ("governance_retention", phase5_5_governance_retention),  # CASCADE-001 A1.3 + GOV-005
        ("test_canary_cleanup", phase5_6_test_canary_cleanup),  # DEBT-199
        ("incremental_vacuum", phase5_7_incremental_vacuum),  # MAINT-030
        ("full_vacuum_if_needed", phase5_9_full_vacuum_if_needed),  # MAINT-039
        ("checkpoint_gc", phase5_10_checkpoint_gc),  # CKPT-006
    ],
}

_NO_CONN_STEPS = {
    phase1_scheduled_tasks,
    phase2_reflection,
    phase3_experiments,
    phase4_curiosity,
    phase3_5_evaluate_experiments,
    phase6_promotion_sweep,
    _phase7_concept_synthesis,
    phase8_backup,
    phase5_10_checkpoint_gc,
}


async def _run_maintenance_step(step_fn, *, dry_run: bool = False):
    """Run one maintenance phase or sub-phase with step-local ownership."""
    if step_fn in _NO_CONN_STEPS:
        return await step_fn(dry_run=dry_run)

    with owned_connection() as conn:
        return await step_fn(conn, dry_run=dry_run)


def _append_partial_failure_warnings(
    report: MaintenanceReport,
    *,
    phase_num: int,
    phase_name: str,
    result: dict,
) -> None:
    """Surface nested phase failures that did not raise phase-level errors."""
    failed_count = int(result.get("tasks_failed") or 0)
    nested_errors = [str(err) for err in (result.get("errors") or []) if str(err)]
    if failed_count <= 0 and not nested_errors:
        return

    detail = "; ".join(nested_errors[:3]) if nested_errors else "nested task failure"
    warning_msg = (
        f"Phase {phase_num} ({phase_name}) completed with "
        f"{max(failed_count, len(nested_errors))} nested task failure(s): {detail}"
    )
    report.warnings.append(warning_msg)
    logger.warning(warning_msg)


async def run_maintenance(
    phases: list[int] | None = None,
    dry_run: bool = False,
) -> MaintenanceReport:
    """Run maintenance phases.

    Args:
        phases: List of phase numbers to run (default: all).
        dry_run: If True, report what would happen without executing.

    Returns:
        MaintenanceReport with results from each phase.
    """
    report = MaintenanceReport(
        started_at=_utc_now_iso(),
        dry_run=dry_run,
    )

    phases_to_run = sorted(set(phases or list(ALL_PHASES.keys())))
    for phase_num in phases_to_run:
        if phase_num not in ALL_PHASES:
            report.errors.append(f"Unknown phase: {phase_num}")
            continue

        phase_name, phase_fn = ALL_PHASES[phase_num]
        logger.info(f"Maintenance phase {phase_num} ({phase_name}): starting...")

        try:
            t0 = time.monotonic()
            # MAINT-001: Timeout guard — prevent phase hangs from blocking entire run
            result = await asyncio.wait_for(
                _run_maintenance_step(phase_fn, dry_run=dry_run),
                timeout=PHASE_TIMEOUT_SECONDS,
            )
            elapsed = round(time.monotonic() - t0, 2)

            report.phases_run.append(phase_name)
            report.results[phase_name] = {
                "elapsed_seconds": elapsed,
                **result,
            }
            _append_partial_failure_warnings(
                report,
                phase_num=phase_num,
                phase_name=phase_name,
                result=result,
            )
            if result.get("budget_aborted"):
                abort_reason = result.get("abort_reason", "unknown")
                last_step = result.get("abort_stage") or result.get("last_completed_step") or "unknown"
                warning_msg = (
                    f"Phase {phase_num} ({phase_name}) aborted before timeout: "
                    f"{abort_reason} after {last_step}"
                )
                report.warnings.append(warning_msg)
                logger.warning(warning_msg)
            logger.info(f"Maintenance phase {phase_num} ({phase_name}): done in {elapsed}s")

            # §3.4: Run sub-phases (e.g., contradiction_sweep after reflection)
            for sub_name, sub_fn in _SUB_PHASES.get(phase_num, []):
                try:
                    t1 = time.monotonic()
                    # MAINT-001 / GA-005: Sub-phases also get timeout guard
                    sub_result = await asyncio.wait_for(
                        _run_maintenance_step(sub_fn, dry_run=dry_run),
                        timeout=PHASE_TIMEOUT_SECONDS,
                    )
                    sub_elapsed = round(time.monotonic() - t1, 2)
                    report.phases_run.append(sub_name)
                    report.results[sub_name] = {
                        "elapsed_seconds": sub_elapsed,
                        **sub_result,
                    }
                    logger.info(f"Maintenance sub-phase ({sub_name}): done in {sub_elapsed}s")
                except TimeoutError:
                    sub_err = f"Sub-phase ({sub_name}) TIMED OUT after {PHASE_TIMEOUT_SECONDS}s"
                    report.errors.append(sub_err)
                    logger.error(sub_err)
                except Exception as sub_e:
                    sub_err = f"Sub-phase ({sub_name}) failed: {str(sub_e)}"
                    report.errors.append(sub_err)
                    logger.error(sub_err, exc_info=True)

        except TimeoutError:
            error_msg = f"Phase {phase_num} ({phase_name}) TIMED OUT after {PHASE_TIMEOUT_SECONDS}s"
            if phase_name == "reflection":
                elapsed = round(time.monotonic() - t0, 2) if "t0" in locals() else PHASE_TIMEOUT_SECONDS
                report.results[phase_name] = {
                    "elapsed_seconds": elapsed,
                    "budget_aborted": True,
                    "abort_reason": "outer_timeout",
                    "last_completed_step": None,
                    "reflection_summary": {
                        "aborted": True,
                        "abort_reason": "outer_timeout",
                        "last_completed_step": None,
                        "phase_timings": {},
                    },
                }
            report.errors.append(error_msg)
            logger.error(error_msg)
        except Exception as e:
            error_msg = f"Phase {phase_num} ({phase_name}) failed: {str(e)}"
            report.errors.append(error_msg)
            logger.error(error_msg, exc_info=True)

    report.completed_at = _utc_now_iso()

    return report


def _write_heartbeat(report: MaintenanceReport, *, source: str = "manual") -> None:
    """Amendment 3: Write heartbeat file after maintenance run.

    Enables monitoring: if heartbeat is stale (>12h), conversation_turn
    can warn that autonomous maintenance is down.
    """
    try:
        from app.core.profile import resolve_data_dir

        data_dir = str(resolve_data_dir())
        heartbeat_path = os.path.join(data_dir, "maintenance_heartbeat.json")
        heartbeat = {
            "source": source,
            "last_run": _utc_now_iso(),
            "status": "ok" if not report.errors else "errors",
            "phases_completed": len(report.phases_run),
            "phases_failed": len(report.errors),
            "phases_warned": len(report.warnings),
            "duration_seconds": report._duration(),
            "errors": report.errors[:5],  # Cap at 5 for file size
            "warnings": report.warnings[:5],
        }
        reflection_result = report.results.get("reflection")
        if reflection_result:
            heartbeat["reflection_status"] = {
                "budget_aborted": bool(reflection_result.get("budget_aborted")),
                "abort_reason": reflection_result.get("abort_reason"),
                "last_completed_step": reflection_result.get("last_completed_step"),
                "abort_stage": reflection_result.get("abort_stage"),
                "elapsed_seconds": reflection_result.get("elapsed_seconds"),
            }
        os.makedirs(os.path.dirname(heartbeat_path), exist_ok=True)
        with open(heartbeat_path, "w") as f:
            json.dump(heartbeat, f, indent=2)
        logger.info(f"Heartbeat written to {heartbeat_path}")
    except Exception as e:
        logger.warning(f"Failed to write heartbeat (non-fatal): {e}")


def run_maintenance_sync(
    phases: list[int] | None = None,
    dry_run: bool = False,
    source: str = "manual",
) -> dict:
    """Synchronous wrapper for CLI and cron usage."""
    report = asyncio.run(run_maintenance(phases=phases, dry_run=dry_run))
    # Amendment 3: Write heartbeat after every run (even dry runs, for monitoring)
    _write_heartbeat(report, source=source)
    return report.to_dict()


def backfill_superseded_concepts(conn) -> dict:
    """SUPER-015: One-time migration to normalize SUPERSEDED concept state.

    For each SUPERSEDED concept, ensure:
    1. superseded_by is set (trace via supersedes association edge if missing)
    2. currency_status = 'SUPERSEDED'

    Returns stats dict with counts of each fix applied.
    """
    stats = {
        "total_superseded": 0,
        "superseded_by_set_via_edge": 0,
        "superseded_by_already_set": 0,
        "untraceable_orphans": 0,
        "errors": 0,
    }

    rows = conn.execute(
        """SELECT id, superseded_by, is_current
           FROM concepts
           WHERE currency_status = 'SUPERSEDED'"""
    ).fetchall()

    stats["total_superseded"] = len(rows)

    for concept_id, superseded_by, is_current in rows:
        try:
            if superseded_by:
                stats["superseded_by_already_set"] += 1
                continue

            # Try to find superseder via 'supersedes' association edge
            edge_row = conn.execute(
                """SELECT source FROM associations
                   WHERE target = ? AND relation = 'supersedes'
                   ORDER BY created_at DESC LIMIT 1""",
                (concept_id,),
            ).fetchone()

            if edge_row:
                superseder_id = edge_row[0]
                superseder_exists = conn.execute(
                    "SELECT 1 FROM concepts WHERE id = ? AND status = 'active'",
                    (superseder_id,),
                ).fetchone()

                if superseder_exists:
                    from app.storage import apply_lifecycle_transition_conn

                    apply_lifecycle_transition_conn(
                        conn,
                        concept_id,
                        "supersede",
                        superseded_by=superseder_id,
                        reason="SUPER-015 edge backfill",
                    )
                    stats["superseded_by_set_via_edge"] += 1
                else:
                    stats["untraceable_orphans"] += 1
            else:
                stats["untraceable_orphans"] += 1

        except Exception as e:
            logger.warning("SUPER-015: backfill error for %s: %s", concept_id, e)
            stats["errors"] += 1

    conn.commit()
    logger.info("SUPER-015: backfill complete — %s", stats)
    return stats


def strip_superseded_prefix(conn) -> dict:
    """SUPER-016: Remove [SUPERSEDED] prefix from concept summaries and re-embed.

    The prefix pollutes embedding space. currency_status='SUPERSEDED' already
    marks the concept; the prefix is redundant.

    Returns stats dict with count of concepts updated.
    """
    stats = {"stripped": 0, "re_embedded": 0, "embed_errors": 0, "skipped": 0}

    rows = conn.execute(
        """SELECT id, summary, data FROM concepts
           WHERE summary LIKE '[SUPERSEDED]%'"""
    ).fetchall()

    for concept_id, summary, data_blob in rows:
        try:
            new_summary = summary
            if summary.startswith("[SUPERSEDED] "):
                new_summary = summary[len("[SUPERSEDED] ") :]
            elif summary.startswith("[SUPERSEDED]"):
                new_summary = summary[len("[SUPERSEDED]") :]

            if not new_summary.strip():
                stats["skipped"] += 1
                continue

            # KA-006: Sync summary to both column AND blob
            try:
                payload = json.loads(data_blob) if data_blob else {}
            except (TypeError, json.JSONDecodeError):
                payload = {}
            from app.storage.concepts import clear_stale_risk_metadata

            payload["summary"] = new_summary.strip()
            clear_stale_risk_metadata(payload)
            _now = _utc_now_iso()
            conn.execute(
                """UPDATE concepts
                   SET summary = ?, updated_at = ?, content_updated_at = ?,
                       staleness_state = NULL, staleness_score = NULL,
                       staleness_reason = NULL, staleness_evaluated_at = NULL,
                       staleness_detector_version = NULL, staleness_consecutive_hits = 0,
                       data = ?
                   WHERE id = ?""",
                (new_summary.strip(), _now, _now, json.dumps(payload), concept_id),
            )
            stats["stripped"] += 1

            # Re-embed with clean summary
            try:
                from app.retrieval import retrieval_engine

                if retrieval_engine and hasattr(retrieval_engine, "embedding_engine"):
                    embedding = retrieval_engine.embedding_engine.embed_text(new_summary.strip())
                    if embedding:
                        import struct

                        blob = struct.pack(f"{len(embedding)}f", *embedding)
                        conn.execute(
                            "UPDATE concepts SET embedding = ? WHERE id = ?",
                            (blob, concept_id),
                        )
                        stats["re_embedded"] += 1
            except Exception as embed_err:
                logger.warning("SUPER-016: re-embed failed for %s: %s", concept_id, embed_err)
                stats["embed_errors"] += 1

        except Exception as e:
            logger.warning("SUPER-016: strip failed for %s: %s", concept_id, e)

    conn.commit()
    logger.info("SUPER-016: prefix strip complete — %s", stats)
    return stats
