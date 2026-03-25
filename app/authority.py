"""Epistemic Authority — pre-computed cached scoring for concept trust levels.

Reference: MEMORY_INTEGRITY_SPEC v1.2, §5.1.2 (CM-C1)

Authority determines HOW a concept is presented (CONSTRAINT vs CONTEXT vs BACKGROUND)
and how strongly it influences retrieval ranking.

FORMULA (published per CM-C1 — security-through-obscurity is not defense):

  raw_authority = (type_weight * 0.35) + (evidence_depth * 0.30)
                + (evolution_investment * 0.20) + (stability * 0.15)

  All component weights are configurable in config.py:
    AUTHORITY_TYPE_WEIGHT     = 0.35  (concept_type base weight)
    AUTHORITY_EVIDENCE_WEIGHT = 0.30  (evidence quantity + quality)
    AUTHORITY_EVOLUTION_WEIGHT= 0.20  (version count + correction bonus)
    AUTHORITY_STABILITY_WEIGHT= 0.15  (access stability without contradiction)

  Component scoring:
    type_weight:    AUTHORITY_TYPE_WEIGHTS[concept_type] (0.30-0.90 per hierarchy level)
    evidence_depth: min(1.0, count_factor*0.6 + avg_quality*0.4)
    evolution:      min(1.0, (versions/10)*0.7 + correction_bonus)
    stability:      [0.0, 1.0] from concepts.stability column

POST-PROCESSING PIPELINE (retrieval-time, NOT stored):
  1. raw_authority is STORED in concepts.authority_score column
  2. At retrieval: epistemic cap applied (§5.1.3, epistemic.py)
     effective = min(raw_authority, epistemic_cap)
     Caps vary by network (world_fact/preference/assessment) × status (verified/unverified/stale/contradicted)
  3. Currency score (temporal decay) computed separately (currency.py)
  4. Presentation mode derived from effective authority:
     >= 0.80: CONSTRAINT | 0.60-0.79: DIRECTIVE | 0.40-0.59: CONTEXT | < 0.40: BACKGROUND

NAMING CONVENTION (H16):
  - Column: authority_score (always, never bare "authority")
  - Variable: raw_authority (pre-cap), authority (post-cap/effective)
  - Config: AUTHORITY_* prefix for all related constants

Score is computed on creation, every evolution, and refreshed by async recalibration.
Result is STORED in concepts.authority_score column. Retrieval reads cached value.
"""

import bisect
import json
import logging
import sqlite3

from app.config import (
    AUTHORITY_EVIDENCE_WEIGHT as CFG_EVIDENCE_W,
)
from app.config import (
    AUTHORITY_EVOLUTION_CORRECTION_BONUS,
    AUTHORITY_EVOLUTION_VERSION_DIVISOR,
    AUTHORITY_TYPE_WEIGHT_DEFAULT,
    AUTHORITY_TYPE_WEIGHTS,
    KA_MIN_POPULATION_THRESHOLD,
    PRESENTATION_CONSTRAINT,
    PRESENTATION_CONTEXT,
    PRESENTATION_DIRECTIVE,
)
from app.config import (
    AUTHORITY_EVOLUTION_WEIGHT as CFG_EVOLUTION_W,
)
from app.config import (
    AUTHORITY_STABILITY_WEIGHT as CFG_STABILITY_W,
)
from app.config import (
    AUTHORITY_TYPE_WEIGHT as CFG_TYPE_W,
)
from app.datetime_utils import _utc_now_iso

logger = logging.getLogger(__name__)


def _type_weight(concept_type: str) -> float:
    """Base authority from concept type. Higher-level types carry more weight."""
    return AUTHORITY_TYPE_WEIGHTS.get(concept_type, AUTHORITY_TYPE_WEIGHT_DEFAULT)


def _evidence_depth(concept_data: dict) -> float:
    """Score based on evidence quality and quantity.

    Reads structured evidence from concept's data JSON.
    Score = min(1.0, count_factor * 0.6 + avg_quality * 0.4)
    where count_factor = min(1.0, evidence_count / 5)
    and avg_quality averages reliability * directness per evidence item.
    """
    evidence_list = concept_data.get("evidence", [])
    if not evidence_list:
        return 0.1  # Minimal floor — no evidence at all

    count = len(evidence_list)
    total_quality = 0.0

    for ev in evidence_list:
        if isinstance(ev, dict):
            reliability = ev.get("reliability_weight", 0.7)
            directness = ev.get("directness", 0.8)
            total_quality += reliability * directness
        elif isinstance(ev, str):
            # Legacy string evidence — assume moderate quality
            total_quality += 0.5

    avg_quality = total_quality / count if count > 0 else 0.3
    count_factor = min(1.0, count / 5.0)

    return round(min(1.0, count_factor * 0.6 + avg_quality * 0.4), 4)


def _evolution_investment(concept_data: dict, version_count: int = 1) -> float:
    """Higher investment = more cognitive work = more authority.

    score = min(1.0, (version_count / 10) * 0.7 + (0.3 if has_corrections else 0))
    """
    has_corrections = False
    versions = concept_data.get("versions", [])
    if versions:
        for v in versions:
            if isinstance(v, dict) and v.get("change_type") == "contradiction_flag":
                has_corrections = True
                break

    # Also check top-level change_type
    if concept_data.get("change_type") == "contradiction_flag":
        has_corrections = True

    version_factor = min(1.0, (version_count / AUTHORITY_EVOLUTION_VERSION_DIVISOR) * 0.7)
    correction_bonus = AUTHORITY_EVOLUTION_CORRECTION_BONUS if has_corrections else 0.0

    return round(min(1.0, version_factor + correction_bonus), 4)


def compute_authority_score(
    concept_type: str,
    concept_data: dict,
    stability: float = 0.5,
    version_count: int = 1,
) -> float:
    """Compute epistemic authority score [0.0 - 1.0].

    This is the core scoring function. Called on concept creation,
    evolution, and during recalibration.

    Args:
        concept_type: From the 6-level hierarchy (observation, decision, principle, etc.)
        concept_data: Full concept data dict (contains evidence, versions, etc.)
        stability: Concept stability score [0-1]
        version_count: Number of versions for this concept

    Returns:
        Authority score clamped to [0.0, 1.0]
    """
    tw = _type_weight(concept_type)
    ed = _evidence_depth(concept_data)
    ei = _evolution_investment(concept_data, version_count)
    st = min(1.0, max(0.0, stability))

    score = tw * CFG_TYPE_W + ed * CFG_EVIDENCE_W + ei * CFG_EVOLUTION_W + st * CFG_STABILITY_W

    return round(min(1.0, max(0.0, score)), 4)


def _precompute_ka_percentiles(conn: sqlite3.Connection) -> dict[str, list[float]]:
    """Precompute sorted authority distributions per knowledge_area.

    Federation Phase 0, Component 0.1 (A1: single SQL + O(N log N) sort).
    """
    rows = conn.execute(
        "SELECT knowledge_area, authority_score FROM concepts "
        "WHERE knowledge_area IS NOT NULL AND authority_score IS NOT NULL "
        "AND status = 'active' AND maturity != 'DISCARDED' "
        "ORDER BY knowledge_area, authority_score"
    ).fetchall()

    ka_scores: dict[str, list[float]] = {}
    for row in rows:
        ka = row[0] if isinstance(row, tuple | list) else row["knowledge_area"]
        score = row[1] if isinstance(row, tuple | list) else row["authority_score"]
        ka_scores.setdefault(ka, []).append(score)

    return ka_scores


def compute_ka_relative_authority(
    global_authority: float,
    ka_scores: list[float],
) -> float:
    """Compute KA-relative authority as percentile rank within knowledge_area.

    Federation Phase 0, Component 0.1.
    A2: Threshold=30. A3: NULL-KA excluded by caller. A8: Tie-breaking=0.5.
    """
    if len(ka_scores) < KA_MIN_POPULATION_THRESHOLD:
        return global_authority  # Fallback for thin KAs

    if ka_scores[0] == ka_scores[-1]:
        return 0.5  # A8: Tie-breaking

    pos = bisect.bisect_right(ka_scores, global_authority)
    return round(pos / len(ka_scores), 4)


def get_presentation_mode(authority_score: float) -> str:
    """Map authority score to presentation mode.

    >= 0.80: CONSTRAINT — must be obeyed
    0.60-0.79: DIRECTIVE — should be followed
    0.40-0.59: CONTEXT — informs reasoning
    < 0.40: BACKGROUND — available but not emphasized
    """
    if authority_score >= PRESENTATION_CONSTRAINT:
        return "CONSTRAINT"
    elif authority_score >= PRESENTATION_DIRECTIVE:
        return "DIRECTIVE"
    elif authority_score >= PRESENTATION_CONTEXT:
        return "CONTEXT"
    else:
        return "BACKGROUND"


def format_concept_with_authority(summary: str, authority_score: float, qualifiers: list[str] | None = None) -> str:
    """Format a concept summary with its presentation mode prefix.

    Examples:
        [CONSTRAINT] Never commit secrets to git
        [DIRECTIVE, CONTESTED] Use PostgreSQL for production
        [CONTEXT, EVIDENCE AGING] React is preferred for frontend
        Background concepts have no prefix
    """
    mode = get_presentation_mode(authority_score)
    if mode == "BACKGROUND":
        return summary

    if qualifiers:
        qualifier_str = ", ".join(qualifiers)
        return f"[{mode}, {qualifier_str}] {summary}"
    return f"[{mode}] {summary}"


def batch_compute_authority(conn: sqlite3.Connection, concept_ids: list[str] | None = None) -> int:
    """Recompute and cache authority scores for concepts.

    Args:
        conn: SQLite connection
        concept_ids: Specific concepts to recompute (None = all)

    Returns:
        Number of concepts updated
    """
    now = _utc_now_iso()

    # Federation Phase 0: Precompute KA distributions for percentile ranking
    from app.config import get_feature_flag

    ka_governance_enabled = get_feature_flag("KA_RELATIVE_GOVERNANCE_ENABLED", False)
    ka_distributions: dict[str, list[float]] = {}
    if ka_governance_enabled:
        try:
            ka_distributions = _precompute_ka_percentiles(conn)
        except Exception as e:
            logger.warning("KA percentile precomputation failed, skipping: %s", e)
            ka_governance_enabled = False

    if concept_ids:
        placeholders = ",".join("?" for _ in concept_ids)
        rows = conn.execute(
            f"SELECT id, concept_type, stability, data, knowledge_area FROM concepts WHERE id IN ({placeholders})",
            concept_ids,
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, concept_type, stability, data, knowledge_area FROM concepts WHERE status != 'deleted' AND maturity != 'DISCARDED'"
        ).fetchall()

    updated = 0
    from app.constants import TYPE_AUTHORITY_CAPS

    for row in rows:
        cid = row[0]
        ctype = row[1] or "observation"
        stab = row[2] or 0.5
        ka = row[4] if len(row) > 4 else None

        try:
            cdata = json.loads(row[3]) if row[3] else {}
        except (json.JSONDecodeError, TypeError):
            cdata = {}

        # Get version count
        vc_row = conn.execute("SELECT COUNT(*) FROM concept_versions WHERE id = ?", (cid,)).fetchone()
        version_count = vc_row[0] if vc_row else 1

        score = compute_authority_score(ctype, cdata, stab, version_count)

        # AUTHORITY-001: Compute and persist effective_authority
        type_cap = TYPE_AUTHORITY_CAPS.get(ctype)
        effective = min(score, type_cap) if type_cap is not None else score

        # Federation Phase 0, Component 0.1: KA-relative percentile authority
        ka_rel = None
        if ka_governance_enabled and ka is not None and ka in ka_distributions:
            try:
                ka_rel = compute_ka_relative_authority(score, ka_distributions[ka])
            except Exception:
                ka_rel = None

        conn.execute(
            """UPDATE concepts
               SET authority_score = ?, effective_authority = ?, ka_relative_authority = ?,
                   last_authority_recompute = ?,
                   data = json_set(data,
                       '$.authority_score', ?,
                       '$.effective_authority', ?,
                       '$.ka_relative_authority', ?,
                       '$.last_authority_recompute', ?
                   )
               WHERE id = ?""",
            (score, round(effective, 4), ka_rel, now, score, round(effective, 4), ka_rel, now, cid),
        )
        updated += 1

    conn.commit()
    logger.info("Authority batch recompute: %d concepts updated", updated)
    return updated
