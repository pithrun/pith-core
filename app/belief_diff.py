"""P4b: Belief Diff — Compare pith state at two points in time.

Implements temporal belief comparison using bi-temporal columns
(valid_from / valid_until) on the concepts table. Returns structured
diffs showing what was added, removed, changed, and unchanged.

Feature flag: BELIEF_DIFF_ENABLED (default: True, read-only/low-risk)
Spec: MEMORY_INTEGRITY_PHASE4_SPEC v1.1, Section 3.2
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.config import FEATURE_FLAGS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ConceptSnapshot:
    """A concept's state at a specific point in time."""

    concept_id: str
    summary: str
    confidence: float
    authority_score: float | None
    currency_score: float | None
    currency_status: str
    epistemic_network: str | None
    knowledge_area: str
    concept_type: str
    maturity: str
    created_at: str
    valid_from: str | None


@dataclass
class ConceptChange:
    """A concept that exists in both states but with different scores."""

    concept_id: str
    summary: str
    knowledge_area: str
    before: dict[str, Any]
    after: dict[str, Any]
    change_type: str  # "authority_shift", "currency_shift", "epistemic_change", "confidence_shift"


@dataclass
class BeliefDiff:
    """Result of comparing two belief states."""

    t1: str
    t2: str
    knowledge_area_filter: str | None
    added: list[dict[str, Any]]
    removed: list[dict[str, Any]]
    changed: list[dict[str, Any]]
    unchanged_count: int
    summary: str
    stats: dict[str, Any]
    diff_time_ms: float


# ---------------------------------------------------------------------------
# Core: get_belief_state_at
# ---------------------------------------------------------------------------


def get_belief_state_at(
    timestamp: str,
    knowledge_area: str | None = None,
) -> dict[str, ConceptSnapshot]:
    """Query Pith's belief state at a specific point in time.

    Uses bi-temporal columns:
    - valid_from <= timestamp (concept existed at that time)
    - valid_until IS NULL OR valid_until > timestamp (not yet superseded)

    Amendment A8: If valid_from is NULL, falls back to created_at.

    Args:
        timestamp: ISO datetime string (e.g., "2026-03-01T00:00:00")
        knowledge_area: Optional filter for specific domain

    Returns:
        Dict mapping concept_id -> ConceptSnapshot
    """
    from app.storage import _db

    sql = """
        SELECT id, summary, confidence, authority_score, currency_score,
               currency_status, epistemic_network, knowledge_area,
               concept_type, maturity, created_at, valid_from
        FROM concepts
        WHERE COALESCE(valid_from, created_at) <= ?
          AND (valid_until IS NULL OR valid_until > ?)
    """
    params: list = [timestamp, timestamp]

    if knowledge_area:
        sql += " AND knowledge_area = ?"
        params.append(knowledge_area)

    state: dict[str, ConceptSnapshot] = {}

    with _db() as conn:
        cursor = conn.execute(sql, params)
        for row in cursor.fetchall():
            snap = ConceptSnapshot(
                concept_id=row[0],
                summary=row[1] or "",
                confidence=row[2] or 0.5,
                authority_score=row[3],
                currency_score=row[4],
                currency_status=row[5] or "ACTIVE",
                epistemic_network=row[6],
                knowledge_area=row[7] or "general",
                concept_type=row[8] or "observation",
                maturity=row[9] or "ESTABLISHED",
                created_at=row[10] or "",
                valid_from=row[11],
            )
            state[snap.concept_id] = snap

    logger.debug("P4b: Belief state at %s: %d concepts (area=%s)", timestamp, len(state), knowledge_area)
    return state


# ---------------------------------------------------------------------------
# Core: diff
# ---------------------------------------------------------------------------


def diff(
    state_t1: dict[str, ConceptSnapshot],
    state_t2: dict[str, ConceptSnapshot],
) -> BeliefDiff:
    """Compare two belief states and produce a structured diff.

    Set operations:
    - added: concepts in t2 but not t1
    - removed: concepts in t1 but not t2 (superseded/deleted)
    - changed: concepts in both but with different scores
    - unchanged: concepts in both with same scores

    Args:
        state_t1: Earlier belief state
        state_t2: Later belief state

    Returns:
        BeliefDiff with all categories
    """
    ids_t1: set[str] = set(state_t1.keys())
    ids_t2: set[str] = set(state_t2.keys())

    added_ids = ids_t2 - ids_t1
    removed_ids = ids_t1 - ids_t2
    common_ids = ids_t1 & ids_t2

    # Build added list
    added = []
    for cid in sorted(added_ids):
        snap = state_t2[cid]
        added.append(_snapshot_to_dict(snap))

    # Build removed list
    removed = []
    for cid in sorted(removed_ids):
        snap = state_t1[cid]
        removed.append(_snapshot_to_dict(snap))

    # Detect changes in common concepts
    changed = []
    unchanged_count = 0

    for cid in sorted(common_ids):
        s1 = state_t1[cid]
        s2 = state_t2[cid]
        changes = _detect_changes(s1, s2)
        if changes:
            changed.append(
                {
                    "concept_id": cid,
                    "summary": s2.summary,
                    "knowledge_area": s2.knowledge_area,
                    "changes": changes,
                }
            )
        else:
            unchanged_count += 1

    # Compute summary statistics
    stats = _compute_stats(state_t1, state_t2, added_ids, removed_ids, changed)

    # Generate human-readable summary
    summary = _generate_summary(added, removed, changed, unchanged_count, stats)

    return BeliefDiff(
        t1="",  # Filled by caller
        t2="",
        knowledge_area_filter=None,
        added=added,
        removed=removed,
        changed=changed,
        unchanged_count=unchanged_count,
        summary=summary,
        stats=stats,
        diff_time_ms=0.0,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _snapshot_to_dict(snap: ConceptSnapshot) -> dict[str, Any]:
    """Convert a snapshot to a serializable dict."""
    return {
        "concept_id": snap.concept_id,
        "summary": snap.summary[:200],
        "confidence": round(snap.confidence, 3),
        "authority_score": round(snap.authority_score, 3) if snap.authority_score else None,
        "currency_score": round(snap.currency_score, 3) if snap.currency_score else None,
        "currency_status": snap.currency_status,
        "epistemic_network": snap.epistemic_network,
        "knowledge_area": snap.knowledge_area,
        "concept_type": snap.concept_type,
        "maturity": snap.maturity,
    }


def _detect_changes(s1: ConceptSnapshot, s2: ConceptSnapshot) -> list[dict[str, Any]]:
    """Detect meaningful changes between two snapshots of the same concept."""
    changes = []

    # Authority shift (threshold: 0.05 to avoid noise)
    a1 = s1.authority_score or 0
    a2 = s2.authority_score or 0
    if abs(a2 - a1) >= 0.05:
        changes.append(
            {
                "type": "authority_shift",
                "before": round(a1, 3),
                "after": round(a2, 3),
                "delta": round(a2 - a1, 3),
            }
        )

    # Currency shift
    c1 = s1.currency_score or 0
    c2 = s2.currency_score or 0
    if abs(c2 - c1) >= 0.05:
        changes.append(
            {
                "type": "currency_shift",
                "before": round(c1, 3),
                "after": round(c2, 3),
                "delta": round(c2 - c1, 3),
            }
        )

    # Confidence shift
    if abs(s2.confidence - s1.confidence) >= 0.05:
        changes.append(
            {
                "type": "confidence_shift",
                "before": round(s1.confidence, 3),
                "after": round(s2.confidence, 3),
                "delta": round(s2.confidence - s1.confidence, 3),
            }
        )

    # Epistemic network change
    if s1.epistemic_network != s2.epistemic_network:
        changes.append(
            {
                "type": "epistemic_change",
                "before": s1.epistemic_network,
                "after": s2.epistemic_network,
            }
        )

    # Currency status change
    if s1.currency_status != s2.currency_status:
        changes.append(
            {
                "type": "currency_status_change",
                "before": s1.currency_status,
                "after": s2.currency_status,
            }
        )

    # Maturity change
    if s1.maturity != s2.maturity:
        changes.append(
            {
                "type": "maturity_change",
                "before": s1.maturity,
                "after": s2.maturity,
            }
        )

    return changes


def _compute_stats(
    state_t1: dict[str, ConceptSnapshot],
    state_t2: dict[str, ConceptSnapshot],
    added_ids: set[str],
    removed_ids: set[str],
    changed: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute summary statistics for the diff."""
    total_t1 = len(state_t1)
    total_t2 = len(state_t2)

    # Churn rate: (added + removed) / max(total_t1, total_t2, 1)
    churn_rate = (len(added_ids) + len(removed_ids)) / max(total_t1, total_t2, 1)

    # Confidence drift: average confidence change for changed concepts
    confidence_deltas = []
    for c in changed:
        for ch in c.get("changes", []):
            if ch["type"] == "confidence_shift":
                confidence_deltas.append(ch["delta"])
    avg_confidence_drift = sum(confidence_deltas) / len(confidence_deltas) if confidence_deltas else 0

    # Area distribution for added concepts
    added_areas: dict[str, int] = {}
    for cid in added_ids:
        area = state_t2[cid].knowledge_area
        added_areas[area] = added_areas.get(area, 0) + 1

    return {
        "total_t1": total_t1,
        "total_t2": total_t2,
        "net_growth": total_t2 - total_t1,
        "churn_rate": round(churn_rate, 4),
        "avg_confidence_drift": round(avg_confidence_drift, 4),
        "added_by_area": added_areas,
    }


def _generate_summary(
    added: list[dict],
    removed: list[dict],
    changed: list[dict],
    unchanged_count: int,
    stats: dict[str, Any],
) -> str:
    """Generate a human-readable summary of the diff."""
    parts = []
    total = len(added) + len(removed) + len(changed) + unchanged_count

    parts.append(f"Belief diff: {stats['total_t1']} → {stats['total_t2']} concepts")

    if added:
        parts.append(f"  +{len(added)} added")
    if removed:
        parts.append(f"  -{len(removed)} removed")
    if changed:
        parts.append(f"  ~{len(changed)} changed")
    parts.append(f"  ={unchanged_count} unchanged")

    if stats["churn_rate"] > 0:
        parts.append(f"  Churn rate: {stats['churn_rate']:.1%}")

    if stats["avg_confidence_drift"] != 0:
        direction = "↑" if stats["avg_confidence_drift"] > 0 else "↓"
        parts.append(f"  Confidence drift: {direction}{abs(stats['avg_confidence_drift']):.3f}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public API: belief_diff
# ---------------------------------------------------------------------------


def belief_diff(
    t1: str,
    t2: str,
    knowledge_area: str | None = None,
) -> dict[str, Any]:
    """Main entry point: compare pith belief state between two timestamps.

    Feature-flag gated by BELIEF_DIFF_ENABLED (default: True).

    Args:
        t1: ISO datetime string for earlier state
        t2: ISO datetime string for later state
        knowledge_area: Optional filter

    Returns:
        Dict with added, removed, changed, unchanged_count, summary, stats
    """
    t0 = time.perf_counter()

    if not FEATURE_FLAGS.get("BELIEF_DIFF_ENABLED", True):
        return {
            "error": "BELIEF_DIFF_ENABLED=False",
            "added": [],
            "removed": [],
            "changed": [],
            "unchanged_count": 0,
            "summary": "Belief diff disabled",
            "stats": {},
            "diff_time_ms": 0,
        }

    # Validate timestamps
    try:
        dt1 = datetime.fromisoformat(t1)
        dt2 = datetime.fromisoformat(t2)
    except (ValueError, TypeError) as e:
        return {"error": f"Invalid timestamp: {e}"}

    if dt1 > dt2:
        # Swap to ensure t1 < t2
        t1, t2 = t2, t1

    # Get belief states
    state_t1 = get_belief_state_at(t1, knowledge_area)
    state_t2 = get_belief_state_at(t2, knowledge_area)

    # Compute diff
    result = diff(state_t1, state_t2)
    result.t1 = t1
    result.t2 = t2
    result.knowledge_area_filter = knowledge_area
    result.diff_time_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        "P4b: Belief diff %s → %s: +%d -%d ~%d =%d (%.1fms)",
        t1[:19],
        t2[:19],
        len(result.added),
        len(result.removed),
        len(result.changed),
        result.unchanged_count,
        result.diff_time_ms,
    )

    # Cap output size for large diffs
    max_items = 50

    return {
        "t1": t1,
        "t2": t2,
        "knowledge_area_filter": knowledge_area,
        "added": result.added[:max_items],
        "added_count": len(result.added),
        "removed": result.removed[:max_items],
        "removed_count": len(result.removed),
        "changed": result.changed[:max_items],
        "changed_count": len(result.changed),
        "unchanged_count": result.unchanged_count,
        "summary": result.summary,
        "stats": result.stats,
        "diff_time_ms": round(result.diff_time_ms, 2),
        "truncated": len(result.added) > max_items
        or len(result.removed) > max_items
        or len(result.changed) > max_items,
    }
