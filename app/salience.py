"""Wave 4a — Salience Computation Engine.

5-signal weighted computation producing a score in [0.0, 1.0].
Applied as SAL multiplier at END of retrieval pipeline.

Signals:
  access_frequency (0.25) — normalized access count relative to corpus median
  goal_alignment  (0.30) — concept relevance to active goals
  recency         (0.20) — exponential decay from last access
  dependency      (0.15) — in-degree (how many concepts link to this one)
  thread_membership (0.10) — STUB until Wave 5 Narrative Threads
"""

import logging
import math
import sqlite3
from collections import Counter
from datetime import datetime

from app.datetime_utils import _ensure_aware, _utc_now, _utc_now_iso

logger = logging.getLogger(__name__)


# =============================================================================
# Salience Engine
# =============================================================================


class SalienceEngine:
    """Compute and manage concept salience scores."""

    def __init__(self, weights: dict | None = None):
        from app.config import (
            SALIENCE_W_ACCESS,
            SALIENCE_W_DEPENDENCY,
            SALIENCE_W_GOAL,
            SALIENCE_W_RECENCY,
            SALIENCE_W_THREAD,
        )

        self.weights = weights or {
            "access_frequency": SALIENCE_W_ACCESS,
            "goal_alignment": SALIENCE_W_GOAL,
            "recency": SALIENCE_W_RECENCY,
            "dependency": SALIENCE_W_DEPENDENCY,
            "thread_membership": SALIENCE_W_THREAD,
        }

    def compute_salience(
        self,
        concept,
        all_concepts: list,
        active_goals: list | None = None,
    ) -> tuple[float, str]:
        """Compute salience for a single concept.

        Args:
            concept: Concept object with access_count, last_accessed, associations, etc.
            all_concepts: All concepts in Pith (for normalization).
            active_goals: List of active goal dicts/objects with priority + knowledge_area/linked_concepts.

        Returns:
            (salience_score, salience_reason) tuple.
        """
        access = self._compute_access_frequency(concept, all_concepts)
        goal = self._compute_goal_alignment(concept, active_goals)
        recency = self._compute_recency(concept)
        dep = self._compute_dependency(concept, all_concepts)
        thread = self._compute_thread_membership(concept)

        w = self.weights
        salience = (
            w["access_frequency"] * access
            + w["goal_alignment"] * goal
            + w["recency"] * recency
            + w["dependency"] * dep
            + w["thread_membership"] * thread
        )
        salience = max(0.0, min(1.0, salience))

        # §4a.4 — salience_reason (FIX O2)
        reason = f"access={access:.2f} goal={goal:.2f} recency={recency:.2f} dep={dep:.2f} thread={thread:.2f}"

        return round(salience, 4), reason

    def _compute_access_frequency(self, concept, all_concepts) -> float:
        """Normalized access count relative to corpus median. FIX F1: median=0 guard.

        REFLECT-021: If self._precomputed_median is set, uses that instead of
        re-sorting all_concepts on every call (was O(n log n) per concept = O(n² log n) total).
        """
        median = getattr(self, "_precomputed_median", None)
        if median is None:
            # Fallback: compute from all_concepts (legacy path)
            counts = [getattr(c, "access_count", 0) or 0 for c in all_concepts]
            if not counts:
                return 0.0
            median = sorted(counts)[len(counts) // 2]
            if median == 0:
                median = 1  # [FIX F1] Prevent division by zero in new brains
        access_count = getattr(concept, "access_count", 0) or 0
        return min(access_count / (2 * max(median, 1)), 1.0)

    def _compute_goal_alignment(self, concept, active_goals) -> float:
        """Dot product of concept's knowledge_area with active goals. FIX S2: priority weighted."""
        if not active_goals:
            return 0.0

        concept_ka = getattr(concept, "metadata", {}).get("knowledge_area", "")
        if not concept_ka:
            concept_ka = ""

        total_weighted_alignment = 0.0
        total_priority = 0.0

        for goal in active_goals:
            # Goals may have knowledge_area directly, or linked_concepts,
            # or just a summary. Adapt to what's available.
            goal_ka = ""
            goal_priority = 0.5

            if hasattr(goal, "knowledge_area"):
                goal_ka = goal.knowledge_area or ""
                goal_priority = getattr(goal, "priority", 0.5) or 0.5
            elif isinstance(goal, dict):
                goal_ka = goal.get("knowledge_area", "")
                goal_priority = goal.get("priority", 0.5) or 0.5
            elif hasattr(goal, "priority"):
                goal_priority = goal.priority or 0.5

            # Check alignment: knowledge_area match or concept in linked_concepts
            alignment = 0.0
            if goal_ka and concept_ka == goal_ka:
                alignment = 1.0
            elif (
                hasattr(goal, "linked_concepts")
                and concept.id in (goal.linked_concepts or [])
                or isinstance(goal, dict)
                and concept.id in (goal.get("linked_concepts") or [])
            ):
                alignment = 0.8

            total_weighted_alignment += alignment * goal_priority  # [FIX S2]
            total_priority += goal_priority

        return total_weighted_alignment / total_priority if total_priority > 0 else 0.0

    def _compute_recency(self, concept) -> float:
        """Exponential decay from last access. 7-day half-life."""
        last_accessed = getattr(concept, "last_accessed", None)
        if not last_accessed:
            return 0.0
        try:
            if isinstance(last_accessed, str):
                last_dt = datetime.fromisoformat(last_accessed.replace("Z", "+00:00").replace("+00:00", ""))
            else:
                last_dt = last_accessed
            hours_since = (_utc_now() - _ensure_aware(last_dt)).total_seconds() / 3600
            return math.exp(-hours_since / 168)  # 168h = 7 days
        except (ValueError, TypeError):
            return 0.0

    def _compute_dependency(self, concept, all_concepts) -> float:
        """In-degree: how many other concepts associate to this one. Cap at 10.

        REFLECT-021: If self._precomputed_in_degree is set, uses O(1) lookup instead of
        iterating all_concepts (was O(n) per concept = O(n²) total).
        """
        concept_id = getattr(concept, "id", "")
        in_degree_map = getattr(self, "_precomputed_in_degree", None)
        if in_degree_map is not None:
            in_degree = in_degree_map.get(concept_id, 0)
        else:
            # Fallback: O(n) scan (legacy path)
            in_degree = sum(1 for c in all_concepts if concept_id in (getattr(c, "associations", []) or []))
        return min(in_degree / 10.0, 1.0)

    def _compute_thread_membership(self, concept) -> float:
        """Thread membership signal — active thread count / divisor. (Wave 5)"""
        try:
            from app.threads import compute_thread_membership

            return compute_thread_membership(
                getattr(concept, "id", ""),
                cache=getattr(self, "_thread_cache", None),
            )
        except (ImportError, Exception):
            return 0.0


# =============================================================================
# SAL Multiplier (§4a.2)
# =============================================================================


def apply_sal_multiplier(salience: float, base_score: float) -> float:
    """Salience-Adjusted Lookup multiplier.

    Applied at END of retrieval pipeline, AFTER predictive_activation
    and goal_directed boosts.

    Range: [0.7, 1.3] — salience=0 → 0.7x, salience=1 → 1.3x
    """
    multiplier = 0.7 + (salience * 0.6)
    return base_score * multiplier


# =============================================================================
# Stale Alerts (§4a.3)
# =============================================================================

MAX_STALE_ALERTS = 5  # [FIX F3] Prevent alert flood


def generate_stale_alerts(concepts: list) -> list[dict]:
    """Flag concepts with salience decay for reflection.

    Returns concepts where salience < 0.2 but confidence > 0.5
    (important knowledge that's losing relevance).
    """
    stale = []
    for c in concepts:
        sal = getattr(c, "salience", 0.5) or 0.5
        conf = getattr(c, "confidence", 0.0) or 0.0
        if sal < 0.2 and conf > 0.5:
            stale.append(
                {
                    "concept_id": getattr(c, "id", ""),
                    "salience": sal,
                    "confidence": conf,
                    "summary": getattr(c, "summary", "")[:100],
                }
            )

    stale.sort(key=lambda x: x["salience"])
    return stale[:MAX_STALE_ALERTS]


# =============================================================================
# Bulk Recomputation
# =============================================================================


def _batch_update_salience(updates: list[tuple]) -> int:
    """REFLECT-021 A4/A5: Batch-write salience to DB via direct SQL.

    Replaces per-concept save_concept() (13ms each = 42s at 3K concepts)
    with batch executemany + json_set (64ms total). Updates both the
    real columns (salience, salience_source) and the JSON data blob
    ($.salience, $.salience_source, $.salience_set_at, $.salience_reason).

    Args:
        updates: List of (salience, source, set_at_iso, reason, concept_id) tuples.

    Returns:
        Count of updated rows, or 0 on failure (transaction rolls back).
    """
    if not updates:
        return 0
    try:
        from app.storage import _db

        with _db() as conn:
            # Update real columns
            conn.executemany(
                "UPDATE concepts SET salience=?, salience_source=? WHERE id=?",
                [(s, src, cid) for s, src, _ts, _r, cid in updates],
            )
            # Update JSON blob salience fields
            for sal, src, ts, reason, cid in updates:
                conn.execute(
                    "UPDATE concepts SET data=json_set(data, "
                    "'$.salience', ?, '$.salience_source', ?, "
                    "'$.salience_set_at', ?, '$.salience_reason', ?) "
                    "WHERE id=?",
                    (sal, src, ts, reason, cid),
                )
        logger.info("REFLECT-021: Batch salience updated %d concepts", len(updates))
        return len(updates)
    except sqlite3.OperationalError as e:
        logger.error("REFLECT-021: Batch salience FAILED (rollback): %s", e)
        return 0
    except Exception as e:
        logger.error("REFLECT-021: Unexpected batch salience error: %s", e)
        return 0


def recompute_salience(
    concept_id: str | None = None,
    active_goals: list | None = None,
) -> dict:
    """Recompute salience for one concept or all concepts.

    Args:
        concept_id: Specific concept to recompute, or None for all.
        active_goals: Active goals for alignment computation.
                      If None, auto-loads goal-type concepts from DB.

    Returns:
        Summary dict with counts and any stale alerts.
    """
    from app.storage import list_concepts_full, load_concept, save_concept

    engine = SalienceEngine()
    now_iso = _utc_now_iso()

    # REFLECT-014: Auto-load goals when not provided by caller
    if active_goals is None:
        try:
            from app.storage import _db, get_related_concepts

            with _db() as conn:
                goal_rows = conn.execute(
                    "SELECT id FROM concepts WHERE is_current=1 AND concept_type='goal'"
                ).fetchall()
            active_goals = []
            for row in goal_rows:
                goal_concept = load_concept(row[0], track_access=False)
                if goal_concept:
                    linked = get_related_concepts(row[0], max_depth=1)
                    active_goals.append(
                        {
                            "knowledge_area": (goal_concept.metadata or {}).get("knowledge_area", ""),
                            "priority": goal_concept.salience if goal_concept.salience is not None else 0.5,
                            "linked_concepts": linked[:20],
                        }
                    )
            logger.info(f"REFLECT-014: Auto-loaded {len(active_goals)} goals for salience")
        except Exception as e:
            logger.warning(f"REFLECT-014: Goal auto-load failed (non-fatal): {e}")
            active_goals = []

    # Wave 5: Preload thread membership cache for batch efficiency
    try:
        from app.threads import preload_thread_membership_cache

        engine._thread_cache = preload_thread_membership_cache()
    except (ImportError, Exception):
        engine._thread_cache = {}

    if concept_id:
        concept = load_concept(concept_id, track_access=False)
        if not concept:
            return {"error": f"Concept {concept_id} not found"}
        all_concepts = list_concepts_full()
        sal, reason = engine.compute_salience(concept, all_concepts, active_goals)
        concept.salience = sal
        concept.salience_source = "system"
        concept.salience_set_at = now_iso
        concept.salience_reason = reason
        save_concept(concept)
        return {
            "recomputed": 1,
            "concept_id": concept_id,
            "salience": sal,
            "reason": reason,
        }
    else:
        all_concepts = list_concepts_full()

        # REFLECT-021 Fix 2a: Precompute in-degree map + access median ONCE
        # before the loop. Eliminates O(n²) dependency scan and O(n log n)
        # re-sort per concept.
        in_degree_map = Counter()
        for c in all_concepts:
            for assoc_id in (getattr(c, "associations", []) or []):
                in_degree_map[assoc_id] += 1
        engine._precomputed_in_degree = in_degree_map

        access_counts = sorted(getattr(c, "access_count", 0) or 0 for c in all_concepts)
        engine._precomputed_median = max(access_counts[len(access_counts) // 2], 1) if access_counts else 1

        # REFLECT-021 Fix 2b: Compute salience for all concepts, then batch-write.
        # Replaces per-concept save_concept() (13ms each) with batch SQL (64ms total).
        updates = []
        count = 0
        for c in all_concepts:
            # Skip user-set salience
            if getattr(c, "salience_source", "system") == "user":
                continue
            sal, reason = engine.compute_salience(c, all_concepts, active_goals)
            updates.append((sal, "system", now_iso, reason, c.id))
            count += 1

        # Clean up precomputed state
        engine._precomputed_in_degree = None
        engine._precomputed_median = None

        # Batch write salience to DB
        batch_written = _batch_update_salience(updates)
        if batch_written == 0 and updates:
            logger.warning("REFLECT-021: Batch salience failed, falling back to per-concept save")
            for sal, src, ts, reason, cid in updates:
                try:
                    c_obj = load_concept(cid, track_access=False)
                    if c_obj:
                        c_obj.salience = sal
                        c_obj.salience_source = src
                        c_obj.salience_set_at = ts
                        c_obj.salience_reason = reason
                        save_concept(c_obj)
                except Exception:
                    pass

        stale_alerts = generate_stale_alerts(all_concepts)

        return {
            "recomputed": count,
            "batch_written": batch_written,
            "total_concepts": len(all_concepts),
            "stale_alerts": stale_alerts,
        }
