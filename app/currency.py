"""Currency Evaluation — concept-type-aware staleness scoring.

Currency represents how likely a concept is to STILL BE TRUE, independent of
confidence. Uses multi-signal scoring with concept-type-aware half-lives.

Signals (weights from config.py CURRENCY_*_WEIGHT constants):
  - Access recency: concepts the user accesses stay current (primary signal)
  - Topic activity: active knowledge areas keep related concepts current
  - Evidence freshness: fresh evidence refreshes currency
  - Correction history: corrected concepts get temporary boost

See config.py §14.2 for current weight values. Do NOT hardcode weights here.

Score is pre-computed, cached in concepts.currency_score column.
Retrieval reads cached value — zero extra queries at retrieval time.
"""

import json
import logging
import math
import sqlite3
from datetime import datetime, timedelta

from app.config import (
    CURRENCY_ACCESS_RECENCY_WEIGHT,
    CURRENCY_CORRECTION_HISTORY_WEIGHT,
    CURRENCY_EVIDENCE_FRESHNESS_WEIGHT,
    CURRENCY_HALF_LIFE_DEFAULT,
    CURRENCY_HALF_LIVES,
    FACTUAL_TEMPORAL_HALF_LIVES,
    CURRENCY_TOPIC_ACTIVITY_WEIGHT,
    TOPIC_ACTIVITY_NORMALIZATION_MAX,
)
from app.datetime_utils import _ensure_aware, _utc_now, _utc_now_iso

logger = logging.getLogger(__name__)

# Default score when no data is available for a currency component
CURRENCY_NO_DATA_BASELINE = 0.3

# Currency status values
STATUS_ACTIVE = "ACTIVE"
STATUS_SUPERSEDED = "SUPERSEDED"
STATUS_RESOLVED = "RESOLVED"
STATUS_STALE = "STALE"
STATUS_CONTESTED = "CONTESTED"
STATUS_CONTRADICTED = "CONTRADICTED"

# Hard cutoff: below this, concept is excluded from retrieval entirely
CURRENCY_HARD_CUTOFF = 0.20

# CURRENCY-005: Architecture-level types get a lower hard cutoff
# Principles, constraints, methods persist longer — they don't go stale as fast
# as observations or decisions tied to specific implementation state.
CURRENCY_ARCHITECTURE_TYPES = frozenset({"principle", "constraint", "method", "cognitive_strategy", "system_model"})
CURRENCY_ARCHITECTURE_HARD_CUTOFF = 0.05
# CURRENCY-005: Architecture-level types get a lower cutoff — they should only go
# STALE from near-zero activity, not from normal time decay. These types encode
# durable knowledge that outlasts implementation details.
CURRENCY_ARCHITECTURE_TYPES = frozenset({"principle", "constraint", "method", "cognitive_strategy", "system_model"})
CURRENCY_ARCHITECTURE_HARD_CUTOFF = 0.05

# Contested penalty multiplier
CONTESTED_PENALTY = 0.70

# --- Reinforcement Anti-Bias (§5.4.3 CM-H6, H2) ---
# Cap at +10% (was +20%), add contradiction penalty.
# Per-session cap: max 1 reinforcement per session per concept.
REINFORCEMENT_MAX_BONUS = 0.10  # Hard cap on reinforcement bonus
REINFORCEMENT_PER_ACCESS = 0.01  # 0.01 per reinforcement event
REINFORCEMENT_CONTRADICTION_PENALTY = 0.5  # 50% reset on first contradiction


def _days_since(iso_timestamp: str | None) -> float:
    """Calculate days since an ISO timestamp. Returns large number if None."""
    if not iso_timestamp:
        return 365.0  # Treat missing timestamps as very old
    try:
        dt = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
        delta = _utc_now() - _ensure_aware(dt)
        return max(0.0, delta.total_seconds() / 86400.0)
    except (ValueError, TypeError):
        return 365.0


def _half_life_decay(days_old: float, half_life_days: float) -> float:
    """Exponential decay based on half-life.

    Returns value in [0.0, 1.0] where 1.0 = brand new, 0.0 = infinitely old.
    At half_life_days, returns 0.5.
    """
    if half_life_days <= 0:
        return 0.0
    return math.pow(0.5, days_old / half_life_days)


# Floor for access_recency: prevents never-accessed concepts from scoring 0.0
# With ar_weight=0.55 and floor=0.15: min contribution = 0.55*0.15 = 0.0825
# Combined with other components, ensures total stays above HARD_CUTOFF (0.20)
ACCESS_RECENCY_FLOOR = 0.15


def _access_recency_score(
    last_accessed: str | None,
    concept_type: str,
    concept_data: dict | None = None,
) -> float:
    """Score based on when this concept's CONTENT was last updated.

    Uses content_updated_at (true content age) as the primary anchor,
    falling back to created_at. Decoupled from last_accessed to prevent
    system access patterns from saturating the signal.

    Fallback chain (with tier tracking):
      1. content_updated_at — only set when summary actually changes (DATA-020)
      2. created_at — concept birth timestamp
      3. last_accessed — legacy fallback (saturated by load_concept)
      4. floor — ACCESS_RECENCY_FLOOR if all timestamps are None

    See: CURRENCY_AR_ANCHOR_DESIGN_v1.md
    """
    anchor_ts = None
    tier = "floor"
    if concept_data:
        anchor_ts = concept_data.get("content_updated_at")
        if anchor_ts:
            tier = "content_updated_at"
        else:
            anchor_ts = concept_data.get("created_at")
            if anchor_ts:
                tier = "created_at"
    if not anchor_ts:
        anchor_ts = last_accessed
        if anchor_ts:
            tier = "last_accessed"
    logger.debug("AR anchor tier=%s for concept_type=%s", tier, concept_type)

    days = _days_since(anchor_ts)
    half_life = CURRENCY_HALF_LIVES.get(concept_type, CURRENCY_HALF_LIFE_DEFAULT)
    # INGEST-016: Bimodal decay — factual concepts use temporal_category half-life.
    # concept_data IS the parsed JSON blob; is_factual lives in metadata sub-dict (INGEST-019).
    _meta = concept_data.get("metadata", {}) if concept_data and isinstance(concept_data, dict) else {}
    if _meta.get("is_factual"):
        tc = _meta.get("temporal_category")
        half_life = FACTUAL_TEMPORAL_HALF_LIVES.get(tc, half_life)
    raw = _half_life_decay(days, half_life)
    return max(ACCESS_RECENCY_FLOOR, raw)


def _topic_activity_score(
    knowledge_area: str | None,
    conn: sqlite3.Connection | None = None,
) -> float:
    """Score based on whether this concept's knowledge_area has recent activity.

    Uses topic_activity_cache table (O(1) lookup) if available.
    Falls back to counting recent concept updates in the knowledge area.
    """
    if not knowledge_area or not conn:
        return CURRENCY_NO_DATA_BASELINE  # Default moderate activity for unknown areas

    # Try cached activity first
    try:
        row = conn.execute(
            "SELECT activity_count_30d FROM topic_activity_cache WHERE knowledge_area = ?",
            (knowledge_area,),
        ).fetchone()
        if row:
            count = row[0] or 0
            return min(1.0, math.log1p(count) / math.log1p(TOPIC_ACTIVITY_NORMALIZATION_MAX))
    except sqlite3.OperationalError:
        logger.debug("topic_activity_cache table not found, using fallback")

    # Fallback: count recent concepts in this knowledge area
    try:
        cutoff = (_utc_now() - timedelta(days=7)).isoformat()
        row = conn.execute(
            "SELECT COUNT(*) FROM concepts WHERE knowledge_area = ? AND COALESCE(content_updated_at, updated_at) > ?",  # DATA-035
            (knowledge_area, cutoff),
        ).fetchone()
        count = row[0] if row else 0
        return min(1.0, math.log1p(count) / math.log1p(TOPIC_ACTIVITY_NORMALIZATION_MAX))
    except sqlite3.OperationalError:
        return CURRENCY_NO_DATA_BASELINE


def _evidence_freshness_score(concept_data: dict) -> float:
    """Score based on the freshest evidence attached to this concept.

    Newer evidence = higher score. Uses the most recent evidence timestamp.
    """
    evidence_list = concept_data.get("evidence", [])
    if not evidence_list:
        # Fall back to best available age signal.
        # Prefer created_at (true age) over updated_at (inflated by migrations/evolution).
        # last_accessed indicates user engagement — if fresher, concept is more current.
        created_at = concept_data.get("created_at")
        last_accessed_ts = concept_data.get("last_accessed")
        # Pick the most recent of created_at and last_accessed as the "freshness" signal
        candidates = [ts for ts in [created_at, last_accessed_ts] if ts]
        if candidates:
            best_ts = min(candidates, key=_days_since)  # min days = most recent
            days = _days_since(best_ts)
            return _half_life_decay(days, 30)
        # DATA-035: Prefer content_updated_at (true content age) over updated_at (migration-inflated)
        content_updated_at = concept_data.get("content_updated_at")
        updated_at = concept_data.get("updated_at")
        fallback_ts = content_updated_at or updated_at
        if fallback_ts:
            days = _days_since(fallback_ts)
            return _half_life_decay(days, 30)
        return CURRENCY_NO_DATA_BASELINE

    # Find the freshest evidence
    min_days = 365.0
    found_timestamp = False
    for ev in evidence_list:
        if isinstance(ev, dict):
            ts = ev.get("timestamp")
            if ts:
                days = _days_since(ts)
                min_days = min(min_days, days)
                found_timestamp = True
        # String evidence has no timestamp — ignore for freshness

    # Fallback: if NO evidence had a timestamp, use best age signal
    if not found_timestamp:
        created_at = concept_data.get("created_at")
        last_accessed_ts = concept_data.get("last_accessed")
        candidates = [ts for ts in [created_at, last_accessed_ts] if ts]
        if candidates:
            best_ts = min(candidates, key=_days_since)
            days = _days_since(best_ts)
            return _half_life_decay(days, 30)
        # DATA-035: Prefer content_updated_at over updated_at
        fallback_ts = concept_data.get("content_updated_at") or concept_data.get("updated_at")
        if fallback_ts:
            days = _days_since(fallback_ts)
            return _half_life_decay(days, 30)
        return CURRENCY_NO_DATA_BASELINE

    return _half_life_decay(min_days, 30)


def _correction_history_score(concept_id: str, conn: sqlite3.Connection | None = None) -> float:
    """Score based on correction history. Corrected concepts get a temporary boost.

    Rationale: if the user cared enough to correct a concept, it's actively
    maintained and therefore more current than a concept left to drift.
    """
    if not conn:
        return 0.0

    try:
        cutoff = (_utc_now() - timedelta(days=30)).isoformat()
        row = conn.execute(
            "SELECT COUNT(*) FROM corrections WHERE concept_id = ? AND created_at > ?",
            (concept_id, cutoff),
        ).fetchone()
        count = row[0] if row else 0
        return min(1.0, count * 0.3)  # Each correction adds 0.3, capped at 1.0
    except sqlite3.OperationalError:
        return 0.0  # Table doesn't exist yet


def _reinforcement_bonus(concept_data: dict) -> float:
    """Compute reinforcement bonus with anti-bias caps (§5.4.3).

    Capped at REINFORCEMENT_MAX_BONUS (10%).
    Contradiction penalty: 50% reduction if concept has active contradiction.
    """
    reinforcement_count = concept_data.get("reinforcement_count", 0)
    bonus = min(REINFORCEMENT_MAX_BONUS, reinforcement_count * REINFORCEMENT_PER_ACCESS)

    # Contradiction penalty: halve the bonus on first contradiction
    if concept_data.get("has_active_contradiction", False):
        bonus *= REINFORCEMENT_CONTRADICTION_PENALTY

    return bonus


def compute_currency_score(
    concept_id: str,
    concept_type: str,
    concept_data: dict,
    last_accessed: str | None = None,
    knowledge_area: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> float:
    """Compute currency score [0.0 - 1.0].

    Multi-signal scoring using weights from config.py CURRENCY_*_WEIGHT:
      access_recency * ar_weight + topic_activity * ta_weight
      + evidence_freshness * ef_weight + correction_history * ch_weight
      + reinforcement_bonus (capped at +10%, §5.4.3)

    Args:
        concept_id: Concept identifier
        concept_type: From the 6-level hierarchy
        concept_data: Full concept data dict
        last_accessed: ISO timestamp of last access
        knowledge_area: Concept's knowledge area for topic activity lookup
        conn: Optional SQLite connection for topic activity + correction queries

    Returns:
        Currency score clamped to [0.0, 1.0]
    """
    ar = _access_recency_score(last_accessed, concept_type, concept_data=concept_data)
    ta = _topic_activity_score(knowledge_area, conn)
    ef = _evidence_freshness_score(concept_data)
    ch = _correction_history_score(concept_id, conn)
    rb = _reinforcement_bonus(concept_data)

    score = (
        ar * CURRENCY_ACCESS_RECENCY_WEIGHT
        + ta * CURRENCY_TOPIC_ACTIVITY_WEIGHT
        + ef * CURRENCY_EVIDENCE_FRESHNESS_WEIGHT
        + ch * CURRENCY_CORRECTION_HISTORY_WEIGHT
        + rb
    )

    return round(min(1.0, max(0.0, score)), 4)


def determine_currency_status(
    currency_score: float,
    current_status: str = STATUS_ACTIVE,
    concept_type: str | None = None,
) -> str:
    """Determine currency status based on score and current state.

    Does NOT override SUPERSEDED, CONTESTED, CONTRADICTED, or RESOLVED
    (those are set explicitly by contradiction resolution or supersession).
    Only transitions between ACTIVE <-> STALE based on score.

    CURRENCY-005: Architecture-level types (principle, constraint, method,
    cognitive_strategy, system_model) use a lower hard cutoff (0.05 vs 0.20).
    They should only go STALE from near-zero activity, not normal time decay.
    """
    # Don't override explicit status markers
    if current_status in (STATUS_SUPERSEDED, STATUS_CONTESTED, STATUS_RESOLVED, STATUS_CONTRADICTED):
        return current_status

    # CURRENCY-005: Use architecture cutoff for durable knowledge types
    cutoff = CURRENCY_HARD_CUTOFF
    if concept_type and concept_type in CURRENCY_ARCHITECTURE_TYPES:
        cutoff = CURRENCY_ARCHITECTURE_HARD_CUTOFF

    if currency_score < cutoff:
        return STATUS_STALE
    return STATUS_ACTIVE


def batch_compute_currency(conn: sqlite3.Connection, concept_ids: list[str] | None = None) -> int:
    """Recompute and cache currency scores for concepts.

    Args:
        conn: SQLite connection
        concept_ids: Specific concepts to recompute (None = all active)

    Returns:
        Number of concepts updated
    """
    now = _utc_now_iso()

    if concept_ids:
        placeholders = ",".join("?" for _ in concept_ids)
        rows = conn.execute(
            f"""SELECT id, concept_type, last_accessed, knowledge_area,
                       currency_status, data, content_updated_at
                FROM concepts WHERE id IN ({placeholders})""",
            concept_ids,
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT id, concept_type, last_accessed, knowledge_area,
                      currency_status, data, content_updated_at
               FROM concepts WHERE status != 'deleted'"""
        ).fetchall()

    updated = 0
    for row in rows:
        cid = row[0]
        ctype = row[1] or "observation"
        last_acc = row[2]
        karea = row[3]
        curr_status = row[4] or STATUS_ACTIVE

        try:
            cdata = json.loads(row[5]) if row[5] else {}
        except (json.JSONDecodeError, TypeError):
            cdata = {}

        # Inject content_updated_at from SQL column into cdata dict.
        # This column is NOT in the JSON blob (only top-level DB column),
        # so it must be explicitly selected and injected for AR anchor.
        # See: CURRENCY_AR_ANCHOR_DESIGN_v1.md §Change 2.5
        if len(row) > 6 and row[6]:
            cdata["content_updated_at"] = row[6]

        score = compute_currency_score(cid, ctype, cdata, last_acc, karea, conn)
        new_status = determine_currency_status(score, curr_status, concept_type=ctype)

        # KA-006: Sync both column AND blob to prevent desync
        conn.execute(
            """UPDATE concepts
               SET currency_score = ?, currency_status = ?, last_currency_recompute = ?,
                   data = json_set(data,
                       '$.currency_score', ?,
                       '$.currency_status', ?
                   )
               WHERE id = ?""",
            (score, new_status, now, score, new_status, cid),
        )
        updated += 1

    conn.commit()
    logger.info("Currency batch recompute: %d concepts updated", updated)
    return updated
