"""Quarantine auto-graduation and re-extraction corroboration.

Memory Integrity Spec v1.2:
- §5.2.6 [H1]: Cascading false rejection mitigation (re-extraction auto-promotes)
- A4-H3 / CM-H1: Hardened auto-graduation criteria

Auto-graduation criteria (RETRIEVAL-005 + MATURITY-002):
1. Age >= 3 days in quarantine (created_at, not updated_at)
2. access_count >= 3
3. Re-validate contradiction at promotion time
4. Auto-discard after 60 days without promotion (created_at)
"""

import logging
from datetime import timedelta

from app.config import FEATURE_FLAGS
from app.contradiction import detect_write_contradiction
from app.datetime_utils import _utc_now, _utc_now_iso
from app.policy_engine import log_policy_event
from app.storage import get_db_connection

logger = logging.getLogger(__name__)

# Hardened thresholds (§5.2.6, A4-H3)
GRADUATION_MIN_AGE_DAYS = 3  # MATURITY-002: reduced from 14
GRADUATION_MIN_ACCESS_COUNT = 3
AUTO_DISCARD_AGE_DAYS = 60
REEXTRACTION_SIMILARITY_THRESHOLD = 0.85


def check_quarantine_reextraction(
    new_summary: str,
    new_knowledge_area: str = "",
) -> str | None:
    """Check if a new concept corroborates a quarantined concept.

    If the same information is extracted again independently, auto-promote
    the quarantined concept to PROVISIONAL (independent corroboration).

    Returns concept_id if a quarantined concept was promoted, None otherwise.
    """
    if not FEATURE_FLAGS.get("QUARANTINE_ENDPOINTS_ENABLED", False):
        return None

    conn = get_db_connection()
    quarantined = conn.execute("""
        SELECT id, summary FROM concepts
        WHERE maturity = 'QUARANTINED'
        ORDER BY updated_at DESC LIMIT 50
    """).fetchall()

    if not quarantined:
        return None

    # Use TF-IDF similarity for matching
    try:
        for q in quarantined:
            from app.retrieval import _compute_tfidf_similarity

            similarity = _compute_tfidf_similarity(new_summary, q["summary"])
            if similarity >= REEXTRACTION_SIMILARITY_THRESHOLD:
                now = _utc_now_iso()
                # KA-006: Sync both column AND blob maturity to prevent desync
                conn.execute(
                    """
                    UPDATE concepts
                    SET maturity = 'PROVISIONAL',
                        data = json_set(data, '$.maturity', 'PROVISIONAL'),
                        updated_at = ?
                    WHERE id = ?
                """,
                    (now, q["id"]),
                )
                conn.commit()

                log_policy_event(
                    rule_id="quarantine_reextraction_promote",
                    severity="LOG",
                    concept_id=q["id"],
                    detail=f"Auto-promoted via re-extraction corroboration (sim={similarity:.2f})",
                    caller_context="check_quarantine_reextraction",
                )
                logger.info("Auto-promoted quarantined concept %s via re-extraction", q["id"])
                return q["id"]
    except Exception as e:
        logger.warning("Re-extraction check failed (non-fatal): %s", e)

    return None


def auto_graduate_quarantined() -> dict[str, list[str]]:
    """Run auto-graduation sweep on quarantined concepts.

    Returns dict with 'promoted' and 'discarded' lists of concept IDs.
    Should be called periodically (e.g., daily maintenance).
    """
    if not FEATURE_FLAGS.get("QUARANTINE_ENDPOINTS_ENABLED", False):
        return {"promoted": [], "discarded": [], "candidates_found": 0, "contradiction_blocked": 0}

    conn = get_db_connection()
    now = _utc_now()
    promoted = []
    discarded = []
    contradiction_blocked = 0

    # Find graduation candidates: age >= N days (by created_at) + access >= 3
    grad_cutoff = (now - timedelta(days=GRADUATION_MIN_AGE_DAYS)).isoformat()
    candidates = conn.execute(
        """
        SELECT id, summary, confidence, access_count, data
        FROM concepts
        WHERE maturity = 'QUARANTINED'
          AND created_at < ?
          AND access_count >= ?
    """,
        (grad_cutoff, GRADUATION_MIN_ACCESS_COUNT),
    ).fetchall()

    candidates_found = len(candidates)

    for row in candidates:
        concept_id = row["id"]
        # Re-validate contradiction at promotion time
        try:
            result = detect_write_contradiction(
                new_summary=row["summary"],
                new_knowledge_area="",
                concept_id=concept_id,
            )
            if result.action != "PASS":
                logger.debug("Quarantined %s still contradicts, skipping graduation", concept_id)
                contradiction_blocked += 1
                continue
        except Exception as e:
            logger.warning("Contradiction re-check failed for %s: %s", concept_id, e)
            contradiction_blocked += 1
            continue

        # Promote to PROVISIONAL
        # KA-006: Sync both column AND blob maturity to prevent desync
        conn.execute(
            """
            UPDATE concepts
            SET maturity = 'PROVISIONAL',
                data = json_set(data, '$.maturity', 'PROVISIONAL'),
                updated_at = ?
            WHERE id = ?
        """,
            (now.isoformat(), concept_id),
        )
        promoted.append(concept_id)

        log_policy_event(
            rule_id="quarantine_auto_graduate",
            severity="LOG",
            concept_id=concept_id,
            detail=f"Auto-graduated: age>{GRADUATION_MIN_AGE_DAYS}d (created_at), access>={GRADUATION_MIN_ACCESS_COUNT}",
            caller_context="auto_graduate_quarantined",
        )

    # Auto-discard concepts quarantined > 60 days (by created_at, not updated_at)
    discard_cutoff = (now - timedelta(days=AUTO_DISCARD_AGE_DAYS)).isoformat()
    stale = conn.execute(
        """
        SELECT id FROM concepts
        WHERE maturity = 'QUARANTINED'
          AND created_at < ?
    """,
        (discard_cutoff,),
    ).fetchall()

    for row in stale:
        conn.execute(
            """
            UPDATE concepts
            SET maturity = 'DISCARDED',
                data = json_set(data, '$.maturity', 'DISCARDED'),
                updated_at = ?
            WHERE id = ?
        """,
            (now.isoformat(), row["id"]),
        )
        discarded.append(row["id"])

        log_policy_event(
            rule_id="quarantine_auto_discard",
            severity="LOG",
            concept_id=row["id"],
            detail=f"Auto-discarded: quarantined >{AUTO_DISCARD_AGE_DAYS} days without promotion",
            caller_context="auto_graduate_quarantined",
        )

    if promoted or discarded:
        conn.commit()

    # Always log graduation funnel summary (even when 0 candidates — confirms the sweep ran)
    logger.info(
        "Quarantine graduation: candidates=%d, contradiction_blocked=%d, graduated=%d, discarded=%d",
        candidates_found,
        contradiction_blocked,
        len(promoted),
        len(discarded),
    )

    return {
        "promoted": promoted,
        "discarded": discarded,
        "candidates_found": candidates_found,
        "contradiction_blocked": contradiction_blocked,
    }
