"""Cascade Propagation — Phase 3 v1.1 + CASCADE-001.

Correction cascade (v1.1, WS3):
When a concept is corrected (confidence drops by >=0.3) or superseded
(inter-concept replacement), propagates review to associated concepts.

Positive reinforcement cascade (CASCADE-001):
When a concept is evolved with new evidence from an independent session,
propagates confidence boosts to associated concepts that cite it.

Design decisions (v1.1):
- Depth-1 only (configurable max_depth, default 1)
  Depth-2 touches ~26 concepts avg (~1% of KB per event) — too risky without accuracy data
- max_affected=10 cap prevents runaway cascades
- CASCADE_MAX_DEMOTE=0.15 (A5) caps per-concept confidence reduction
- Amendment A2: Client corrections auto-supersede; auto-detected contradictions
  tag both for human review (no auto-supersession)

CASCADE-001 design decisions:
- Logarithmic dampening: boost = 0.05 / log₂(n+1), prevents runaway amplification
- Hard ceiling at 0.85 (matches extraction.py confidence clamp)
- Session independence: same session cannot boost twice
- 24h cooldown per concept-neighbor pair
- Floor recovery: boost cannot exceed pre_demotion_confidence + 0.15
- BEGIN IMMEDIATE for race condition prevention (A1.1)

Feature-gated by CORRECTION_CASCADE_ENABLED (negative) and REINFORCEMENT_ENABLED (positive).
"""

import json
import logging
import math
import time
from dataclasses import dataclass, field

from app.config import (
    CASCADE_MAX_AFFECTED,
    CASCADE_MAX_DEMOTE,
    CASCADE_MAX_DEPTH,
    REINFORCEMENT_BASE_MAGNITUDE,
    REINFORCEMENT_COOLDOWN_HOURS,
    REINFORCEMENT_ENABLED,
    REINFORCEMENT_EVIDENCE_THRESHOLD,
    REINFORCEMENT_EXCLUDED_RELATIONS,
    REINFORCEMENT_FLOOR_RECOVERY_LIMIT,
    REINFORCEMENT_MAX_CONFIDENCE,
    REINFORCEMENT_MIN_ASSOC_STRENGTH,
)
from app.constants import (
    GOV_EVENT_AUTHORITY_DEMOTION,
    GOV_EVENT_AUTHORITY_REINFORCEMENT,
    GOV_EVENT_AUTHORITY_REVIEW_FLAGGED,
)
from app.datetime_utils import _utc_now_iso
from app.storage import update_concept_data

logger = logging.getLogger(__name__)


@dataclass
class CascadeResult:
    """Result of a correction cascade propagation."""

    source_concept_id: str
    trigger: str = ""  # "correction" | "supersession"
    concepts_reviewed: int = 0
    concepts_demoted: int = 0
    concepts_flagged: int = 0
    cascade_depth: int = 1
    affected_ids: list[str] = field(default_factory=list)
    time_ms: float = 0.0


@dataclass
class ReinforcementResult:
    """Result of a positive reinforcement cascade."""

    source_concept_id: str
    concepts_reinforced: int = 0
    concepts_ceiling_hit: int = 0
    max_boost_magnitude: float = 0.0
    cascade_depth: int = 1
    affected_ids: list[str] = field(default_factory=list)
    time_ms: float = 0.0


def propagate_correction(
    corrected_concept_id: str,
    correction_magnitude: float,
    trigger: str = "correction",
    max_depth: int = None,
    max_affected: int = None,
    conn=None,
) -> CascadeResult:
    """Cascade correction effects to associated concepts.

    v1.1 Rules (depth-1 only):
    1. Load all concepts associated with corrected_concept_id
    2. For each associated concept at depth 1:
       a. If it shares evidence sources with the corrected concept → FLAG for review
       b. If it was created in the same session → DEMOTE confidence
       c. If it has the same knowledge_area AND was created within 1 hour → DEMOTE
    3. Cap at max_affected to prevent runaway cascades
    4. Cap per-concept DEMOTE at CASCADE_MAX_DEMOTE (0.15) — Amendment A5

    Args:
        corrected_concept_id: The concept that was corrected/superseded.
        correction_magnitude: How much confidence dropped (positive value, e.g. 0.5).
        trigger: "correction" or "supersession".
        max_depth: Override CASCADE_MAX_DEPTH (default from config).
        max_affected: Override CASCADE_MAX_AFFECTED (default from config).
        conn: Optional sqlite3 connection. If provided, uses this connection
              instead of calling _get_connection(). Useful for testing with
              in-memory databases or sharing a transaction context.

    Returns:
        CascadeResult with counts of reviewed/demoted/flagged concepts.
    """
    if max_depth is None:
        max_depth = CASCADE_MAX_DEPTH
    if max_affected is None:
        max_affected = CASCADE_MAX_AFFECTED

    result = CascadeResult(
        source_concept_id=corrected_concept_id,
        trigger=trigger,
        cascade_depth=max_depth,
    )
    t0 = time.perf_counter()

    try:
        import json
        from datetime import datetime

        if conn is None:
            from app.storage import _get_connection

            conn = _get_connection()

        # Load the corrected concept's data directly (avoid Pydantic overhead)
        corrected_data = {}
        corrected_row = conn.execute(
            "SELECT data, created_at, knowledge_area FROM concepts WHERE id = ? AND is_current = 1",
            (corrected_concept_id,),
        ).fetchone()
        if not corrected_row:
            logger.warning("Cascade: corrected concept %s not found", corrected_concept_id)
            return result
        try:
            corrected_data = (
                json.loads(corrected_row[0]) if corrected_row[0] and isinstance(corrected_row[0], str) else {}
            )
        except Exception:
            pass

        corrected_evidence = set()
        for e in corrected_data.get("evidence") or []:
            if isinstance(e, str):
                corrected_evidence.add(e.lower().strip())
            elif isinstance(e, dict):
                corrected_evidence.add(str(e.get("content", "")).lower().strip())

        corrected_ka = corrected_data.get("knowledge_area", "") or (corrected_row[2] if corrected_row[2] else "")
        corrected_created = corrected_data.get("created_at", "") or (corrected_row[1] if corrected_row[1] else "")

        # Load associated concepts (depth-1)
        assoc_rows = conn.execute(
            """SELECT DISTINCT
                CASE WHEN source = ? THEN target ELSE source END as assoc_id
               FROM associations
               WHERE source = ? OR target = ?""",
            (corrected_concept_id, corrected_concept_id, corrected_concept_id),
        ).fetchall()

        associated_ids = [row[0] for row in assoc_rows][:max_affected]

        # Compute demote amount (capped by A5)
        raw_demote = correction_magnitude * 0.5
        demote_amount = min(raw_demote, CASCADE_MAX_DEMOTE)

        for assoc_id in associated_ids:
            result.concepts_reviewed += 1
            result.affected_ids.append(assoc_id)

            try:
                assoc_row = conn.execute(
                    "SELECT data, created_at, confidence FROM concepts WHERE id = ? AND is_current = 1",
                    (assoc_id,),
                ).fetchone()

                if not assoc_row:
                    continue

                assoc_data = {}
                try:
                    assoc_data = json.loads(assoc_row[0]) if assoc_row[0] and isinstance(assoc_row[0], str) else {}
                except Exception:
                    pass

                # Check if shares evidence with corrected concept
                assoc_evidence = set()
                for e in assoc_data.get("evidence") or []:
                    if isinstance(e, str):
                        assoc_evidence.add(e.lower().strip())
                    elif isinstance(e, dict):
                        assoc_evidence.add(str(e.get("content", "")).lower().strip())

                shares_evidence = bool(corrected_evidence & assoc_evidence)

                # Check same knowledge area + created within 1 hour
                assoc_ka = assoc_data.get("knowledge_area", "")
                assoc_created = assoc_data.get("created_at", "")
                same_ka_recent = False
                if corrected_ka and assoc_ka == corrected_ka:
                    try:
                        t_corr = datetime.fromisoformat(corrected_created.replace("Z", "+00:00"))
                        t_assoc = datetime.fromisoformat(assoc_created.replace("Z", "+00:00"))
                        if abs((t_corr - t_assoc).total_seconds()) < 3600:
                            same_ka_recent = True
                    except Exception:
                        pass

                # Decision: FLAG or DEMOTE
                should_demote = same_ka_recent or shares_evidence
                if should_demote:
                    old_conf = assoc_row[2] if assoc_row[2] is not None else 0.5
                    new_conf = max(0.0, old_conf - demote_amount)
                    conn.execute(
                        "UPDATE concepts SET confidence = ?, "
                        "data = json_set(data, '$.confidence', ?) "
                        "WHERE id = ? AND is_current = 1",
                        (new_conf, new_conf, assoc_id),
                    )
                    result.concepts_demoted += 1
                    # §3.1 + A1: Log authority_demotion event SAME-TRANSACTION
                    # Only log when actual demotion occurred (new < old)
                    if new_conf < old_conf:
                        _now = _utc_now_iso()
                        conn.execute(
                            """INSERT INTO governance_events
                               (event_type, concept_id, details, created_at)
                               VALUES (?, ?, ?, ?)""",
                            (
                                GOV_EVENT_AUTHORITY_DEMOTION,
                                assoc_id,
                                json.dumps(
                                    {
                                        "trigger": trigger,
                                        "source_concept": corrected_concept_id,
                                        "demotion_amount": round(old_conf - new_conf, 4),
                                        "old_confidence": round(old_conf, 4),
                                        "new_confidence": round(new_conf, 4),
                                        "cascade_depth": 1,
                                        "cascade_rule": "shared_evidence" if shares_evidence else "same_ka_recent",
                                    }
                                ),
                                _now,
                            ),
                        )
                    logger.info(
                        "Cascade DEMOTE: %s confidence %.2f → %.2f (trigger=%s, source=%s)",
                        assoc_id,
                        old_conf,
                        new_conf,
                        trigger,
                        corrected_concept_id,
                    )
                else:
                    # FLAG for review (add cascade_review_pending tag)
                    try:
                        tags = assoc_data.get("tags", [])
                        if "cascade_review_pending" not in tags:
                            tags.append("cascade_review_pending")
                            assoc_data["tags"] = tags
                            # KA-006: Route through write gateway for column sync
                            update_concept_data(conn, assoc_id, assoc_data)
                        # §3.1 + A1: Log review_flagged event
                        _now = _utc_now_iso()
                        conn.execute(
                            """INSERT INTO governance_events
                               (event_type, concept_id, details, created_at)
                               VALUES (?, ?, ?, ?)""",
                            (
                                GOV_EVENT_AUTHORITY_REVIEW_FLAGGED,
                                assoc_id,
                                json.dumps(
                                    {
                                        "trigger": trigger,
                                        "source_concept": corrected_concept_id,
                                        "reason": "cascade_no_demote_criteria",
                                    }
                                ),
                                _now,
                            ),
                        )
                    except Exception:
                        pass
                    result.concepts_flagged += 1
                    logger.info(
                        "Cascade FLAG: %s tagged for review (trigger=%s, source=%s)",
                        assoc_id,
                        trigger,
                        corrected_concept_id,
                    )

            except Exception as e:
                logger.warning("Cascade: failed to process %s: %s", assoc_id, e)

        # §3.1 + A1: Log authority_demotion for the SOURCE concept (direct correction target)
        try:
            src_row = conn.execute(
                "SELECT confidence FROM concepts WHERE id = ? AND is_current = 1",
                (corrected_concept_id,),
            ).fetchone()
            if src_row:
                _now = _utc_now_iso()
                conn.execute(
                    """INSERT INTO governance_events
                       (event_type, concept_id, details, created_at)
                       VALUES (?, ?, ?, ?)""",
                    (
                        GOV_EVENT_AUTHORITY_DEMOTION,
                        corrected_concept_id,
                        json.dumps(
                            {
                                "trigger": "direct_correction",
                                "correction_magnitude": round(correction_magnitude, 4),
                                "current_confidence": round(src_row[0], 4) if src_row[0] else 0,
                            }
                        ),
                        _now,
                    ),
                )
        except Exception as e_src:
            logger.warning("Failed to log source demotion event: %s", e_src)

        conn.commit()

    except Exception as e:
        logger.error("Cascade propagation failed for %s: %s", corrected_concept_id, e)

    result.time_ms = (time.perf_counter() - t0) * 1000

    # WS2: Metric 5 — cascade_propagation_count
    try:
        from app.metrics import metrics as _m5

        _m5.record(
            "cascade_propagation_count",
            result.concepts_reviewed,
            {
                "trigger": trigger,
                "demoted": result.concepts_demoted,
                "flagged": result.concepts_flagged,
            },
        )
    except Exception:
        pass

    logger.info(
        "Cascade complete: source=%s trigger=%s reviewed=%d demoted=%d flagged=%d (%.1fms)",
        corrected_concept_id,
        trigger,
        result.concepts_reviewed,
        result.concepts_demoted,
        result.concepts_flagged,
        result.time_ms,
    )
    return result


# ---------------------------------------------------------------------------
# CASCADE-001: Positive Reinforcement Cascade
# ---------------------------------------------------------------------------


def _parse_boost_state(metadata: dict | None) -> dict:
    """Parse boost state from concept metadata with corruption recovery (A1.4).

    Returns a validated dict with expected fields. If metadata is corrupt or
    missing, returns safe defaults. Logs warnings for corrupt states so
    operators can investigate offline.
    """
    DEFAULT_STATE = {
        "boost_count": 0,
        "last_boost_at": None,
        "last_boost_session_id": None,
        "pre_demotion_confidence": None,
    }

    if not metadata or not isinstance(metadata, dict):
        return DEFAULT_STATE.copy()

    boost_state = metadata.get("_boost_state")

    if boost_state is None:
        return DEFAULT_STATE.copy()

    if not isinstance(boost_state, dict):
        logger.warning(
            "CASCADE-001: corrupt _boost_state (type=%s), resetting to default. Original value: %s",
            type(boost_state).__name__,
            str(boost_state)[:200],
        )
        return DEFAULT_STATE.copy()

    # Validate expected fields, fill missing with defaults
    result = DEFAULT_STATE.copy()
    result.update({k: v for k, v in boost_state.items() if k in DEFAULT_STATE})

    # Type-check boost_count specifically (most critical field)
    if not isinstance(result["boost_count"], (int, float)):
        logger.warning(
            "CASCADE-001: corrupt boost_count=%s, resetting to 0",
            result["boost_count"],
        )
        result["boost_count"] = 0

    return result


def _apply_single_boost(
    conn,
    concept_id: str,
    session_id: str,
    boost_magnitude: float,
    source_concept_id: str,
) -> float | None:
    """Apply a single boost with pessimistic locking to prevent race conditions (A1.1).

    Uses BEGIN IMMEDIATE to acquire a write lock BEFORE reading, serializing
    the check-then-act sequence. Returns the new confidence if boosted, None if
    skipped (cooldown, same session, etc).
    """
    from datetime import datetime

    try:
        conn.execute("BEGIN IMMEDIATE")  # Acquire write lock BEFORE read

        # Load concept (under lock)
        row = conn.execute(
            "SELECT data, confidence FROM concepts WHERE id = ? AND is_current = 1",
            (concept_id,),
        ).fetchone()

        if not row:
            conn.execute("ROLLBACK")
            return None

        data = {}
        try:
            data = json.loads(row[0]) if row[0] and isinstance(row[0], str) else {}
        except Exception:
            data = {}

        old_conf = row[1] if row[1] is not None else 0.5

        # Check cooldown and independence (under lock)
        boost_state = _parse_boost_state(data)

        # Independence: same session cannot boost twice
        if boost_state.get("last_boost_session_id") == session_id:
            logger.debug(
                "Reinforcement: %s skipped (same session %s)",
                concept_id,
                session_id,
            )
            conn.execute("ROLLBACK")
            return None

        # 24-hour cooldown
        last_boost_at = boost_state.get("last_boost_at")
        if last_boost_at:
            try:
                last_dt = datetime.fromisoformat(last_boost_at.replace("Z", "+00:00"))
                now_dt = datetime.fromisoformat(_utc_now_iso().replace("Z", "+00:00"))
                hours_since = (now_dt - last_dt).total_seconds() / 3600
                if hours_since < REINFORCEMENT_COOLDOWN_HOURS:
                    logger.debug(
                        "Reinforcement: %s cooldown (%.1fh since last boost)",
                        concept_id,
                        hours_since,
                    )
                    conn.execute("ROLLBACK")
                    return None
            except Exception:
                pass  # Malformed timestamp — proceed conservatively

        # Compute diminishing boost
        boost_count = boost_state.get("boost_count", 0)
        actual_boost = boost_magnitude / math.log2(boost_count + 2)
        # boost_count+2 because: first boost (count=0) → log2(2)=1 → full magnitude
        # second boost (count=1) → log2(3)=1.58 → 63% magnitude, etc.

        pre_demotion = boost_state.get("pre_demotion_confidence")
        if pre_demotion is None:
            pre_demotion = old_conf

        # Floor recovery ceiling
        floor_ceiling = pre_demotion + REINFORCEMENT_FLOOR_RECOVERY_LIMIT

        new_conf = min(
            REINFORCEMENT_MAX_CONFIDENCE,  # 0.85 hard ceiling
            min(old_conf + actual_boost, floor_ceiling),
        )

        if new_conf <= old_conf:
            conn.execute("ROLLBACK")
            return None

        # Update confidence + reinforcement_count (MAINT-007)
        conn.execute(
            "UPDATE concepts SET confidence = ?, reinforcement_count = reinforcement_count + 1, "
            "data = json_set(data, '$.confidence', ?) "
            "WHERE id = ? AND is_current = 1",
            (round(new_conf, 6), round(new_conf, 6), concept_id),
        )

        # Update boost state metadata
        boost_state["boost_count"] = boost_count + 1
        boost_state["last_boost_at"] = _utc_now_iso()
        boost_state["last_boost_session_id"] = session_id
        if boost_state.get("pre_demotion_confidence") is None:
            boost_state["pre_demotion_confidence"] = pre_demotion
        data["_boost_state"] = boost_state

        # KA-006: Route through write gateway for column sync
        update_concept_data(conn, concept_id, data)

        # Log governance event
        _now = _utc_now_iso()
        conn.execute(
            """INSERT INTO governance_events
               (event_type, concept_id, details, created_at)
               VALUES (?, ?, ?, ?)""",
            (
                GOV_EVENT_AUTHORITY_REINFORCEMENT,
                concept_id,
                json.dumps(
                    {
                        "trigger": "positive_cascade",
                        "source_concept": source_concept_id,
                        "boost_magnitude": round(actual_boost, 6),
                        "old_confidence": round(old_conf, 6),
                        "new_confidence": round(new_conf, 6),
                        "boost_count": boost_count + 1,
                        "triggering_session": session_id,
                    }
                ),
                _now,
            ),
        )

        conn.execute("COMMIT")

        logger.info(
            "Reinforcement BOOST: %s %.4f → %.4f (mag=%.4f, n=%d, session=%s)",
            concept_id,
            old_conf,
            new_conf,
            actual_boost,
            boost_count + 1,
            session_id,
        )
        return new_conf

    except Exception:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        raise


def propagate_reinforcement(
    reinforced_concept_id: str,
    new_evidence_count: int,
    triggering_session_id: str,
    max_depth: int = None,
    max_affected: int = None,
    conn=None,
) -> ReinforcementResult:
    """Cascade positive reinforcement to associated concepts via validated evidence.

    Implements CASCADE-001 positive reinforcement. Fires when a concept is
    evolved with new evidence from an independent session AND the evidence
    sources are validated (cited by external concepts).

    Rules (v1.0):
    1. Load all concepts associated with reinforced_concept_id
    2. For each associated concept at depth 1:
       a. Check if it cites reinforced_concept_id as evidence
       b. If yes AND last boost was >24h ago AND from different session → REINFORCE
       c. Cap boost at max ceiling 0.85
    3. Cap at max_affected to prevent runaway cascades
    4. Diminishing returns: boost_magnitude = base_magnitude / log₂(boost_count + 2)
    5. Floor recovery: boosted confidence cannot exceed (pre_demotion_confidence + 0.15)

    Args:
        reinforced_concept_id: The concept that was evolved with new evidence.
        new_evidence_count: How many NEW evidence sources were added.
        triggering_session_id: Session ID of the session that triggered this.
        max_depth: Override CASCADE_MAX_DEPTH (default from config, depth-1 only).
        max_affected: Override CASCADE_MAX_AFFECTED (default from config).
        conn: Optional sqlite3 connection. If not provided, gets from storage.

    Returns:
        ReinforcementResult with counts of reinforced/ceiling_hit concepts.
    """
    if max_depth is None:
        max_depth = CASCADE_MAX_DEPTH
    if max_affected is None:
        max_affected = CASCADE_MAX_AFFECTED

    # Clamp to depth-1 only
    if max_depth > 1:
        max_depth = 1

    result = ReinforcementResult(
        source_concept_id=reinforced_concept_id,
        cascade_depth=max_depth,
    )
    t0 = time.perf_counter()

    # Feature gate
    if not REINFORCEMENT_ENABLED:
        return result

    # Reinforcement only fires with sufficient new evidence
    if new_evidence_count < REINFORCEMENT_EVIDENCE_THRESHOLD:
        logger.debug(
            "Reinforcement: %s has %d new evidence (threshold %d), skipping",
            reinforced_concept_id,
            new_evidence_count,
            REINFORCEMENT_EVIDENCE_THRESHOLD,
        )
        return result

    # Session ID required for independence check
    if not triggering_session_id:
        logger.debug(
            "Reinforcement: %s skipped — no session_id (MCP tool evolution)",
            reinforced_concept_id,
        )
        return result

    try:
        if conn is None:
            from app.storage import _get_connection

            conn = _get_connection()

        # Load associated concepts (depth-1 only), ordered by strength descending,
        # excluding semantically incompatible relations and weak edges.
        # CASCADE-002: ORDER BY strength DESC ensures max_affected takes top-N by quality.
        # CASCADE-003: Fixes pre-existing ordering bug (was DB insertion order).
        _excluded = tuple(REINFORCEMENT_EXCLUDED_RELATIONS)
        _placeholders = ",".join("?" * len(_excluded))
        # Safe: _placeholders only generates "?,?" strings — values are parameterized
        assoc_rows = conn.execute(
            f"""SELECT assoc_id, MAX(strength) as strength
               FROM (
                   SELECT
                       CASE WHEN source = ? THEN target ELSE source END as assoc_id,
                       strength
                   FROM associations
                   WHERE (source = ? OR target = ?)
                   AND relation NOT IN ({_placeholders})
                   AND strength >= ?
               )
               GROUP BY assoc_id
               ORDER BY strength DESC""",
            (
                reinforced_concept_id,
                reinforced_concept_id,
                reinforced_concept_id,
                *_excluded,
                REINFORCEMENT_MIN_ASSOC_STRENGTH,
            ),
        ).fetchall()

        associated_candidates = assoc_rows[:max_affected]

        for assoc_id, assoc_strength in associated_candidates:
            try:
                # CASCADE-002: Strength-scaled boost replaces broken citation check.
                # Boost magnitude is proportional to association strength:
                #   strength=0.35 → boost=0.0175, strength=0.20 → boost=0.010
                # This is further reduced by diminishing returns in _apply_single_boost.
                scaled_magnitude = REINFORCEMENT_BASE_MAGNITUDE * assoc_strength

                # Delegate to _apply_single_boost (handles locking, cooldown,
                # independence, diminishing returns, and governance logging)
                new_conf = _apply_single_boost(
                    conn=conn,
                    concept_id=assoc_id,
                    session_id=triggering_session_id,
                    boost_magnitude=scaled_magnitude,
                    source_concept_id=reinforced_concept_id,
                )

                if new_conf is not None:
                    if new_conf >= REINFORCEMENT_MAX_CONFIDENCE:
                        result.concepts_ceiling_hit += 1
                    result.concepts_reinforced += 1
                    result.affected_ids.append(assoc_id)
                    result.max_boost_magnitude = max(result.max_boost_magnitude, scaled_magnitude)

            except Exception as e:
                logger.warning(
                    "Reinforcement: failed to process %s: %s",
                    assoc_id,
                    e,
                )

    except Exception as e:
        logger.error(
            "Reinforcement cascade failed for %s: %s",
            reinforced_concept_id,
            e,
        )

    result.time_ms = (time.perf_counter() - t0) * 1000

    # Metric: positive_cascade_propagation_count
    try:
        from app.metrics import metrics

        metrics.record(
            "positive_cascade_propagation_count",
            result.concepts_reinforced,
            {
                "ceiling_hit": result.concepts_ceiling_hit,
                "session": triggering_session_id,
            },
        )
    except Exception:
        pass

    logger.info(
        "Reinforcement cascade: source=%s session=%s reinforced=%d ceiling=%d (%.1fms)",
        reinforced_concept_id,
        triggering_session_id,
        result.concepts_reinforced,
        result.concepts_ceiling_hit,
        result.time_ms,
    )
    return result


def tag_for_review(
    concept_id_a: str,
    concept_id_b: str,
    reason: str,
    conn=None,
) -> None:
    """Amendment A2: Tag two concepts for human review (auto-detected contradiction).

    Neither auto-supersedes. Both get cascade_review_pending tag.
    Reviewed on next conversation_turn that retrieves either concept.
    """
    try:
        import json

        if conn is None:
            from app.storage import _get_connection

            conn = _get_connection()

        for cid in (concept_id_a, concept_id_b):
            row = conn.execute(
                "SELECT data FROM concepts WHERE id = ? AND is_current = 1",
                (cid,),
            ).fetchone()
            if not row:
                continue
            try:
                data = json.loads(row[0]) if row[0] and isinstance(row[0], str) else {}
            except Exception:
                data = {}

            tags = data.get("tags", [])
            if "cascade_review_pending" not in tags:
                tags.append("cascade_review_pending")
            data["tags"] = tags
            data["review_reason"] = reason
            data["review_counterpart"] = concept_id_b if cid == concept_id_a else concept_id_a

            # KA-006: Route through write gateway for column sync
            update_concept_data(conn, cid, data)

        conn.commit()
        logger.info(
            "A2: Tagged %s and %s for human review: %s",
            concept_id_a,
            concept_id_b,
            reason,
        )
    except Exception as e:
        logger.error("A2: Failed to tag for review: %s", e)
