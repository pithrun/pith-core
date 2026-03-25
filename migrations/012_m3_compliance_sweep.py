"""Migration 012: M3 compliance sweep — cap quarantined concepts above 0.4.

Targets PSIS-quarantined concepts that exceeded the M3 confidence ceiling
due to missing ingest-path guard (STABILITY-027).

Spec: SPRINT_REFLECTION_FIX_SPEC.md (Fix 1, one-time sweep)
Backlog: STABILITY-027, MONITOR-044
"""
import logging

logger = logging.getLogger(__name__)


def migrate(dry_run: bool = False) -> dict:
    """Cap all quarantined concepts with confidence > 0.4 to 0.4.

    Uses the same criteria as MONITOR-044 (storage.py:2343):
    maturity = 'QUARANTINED' AND confidence > 0.4
    """
    from app.config import PSIS_QUARANTINE_CONFIDENCE_CAP
    from app.storage import _db, load_concept, save_concept

    capped = 0
    details = []

    with _db() as conn:
        rows = conn.execute(
            """SELECT id, confidence FROM concepts
               WHERE is_current = 1
               AND maturity = 'QUARANTINED'
               AND confidence > ?""",
            (PSIS_QUARANTINE_CONFIDENCE_CAP,),
        ).fetchall()

    print(f"Found {len(rows)} quarantined concepts above {PSIS_QUARANTINE_CONFIDENCE_CAP} cap")

    for row in rows:
        concept_id = row[0] if isinstance(row, tuple) else row["id"]
        old_confidence = row[1] if isinstance(row, tuple) else row["confidence"]

        details.append({
            "id": concept_id,
            "old_confidence": round(old_confidence, 4),
            "new_confidence": PSIS_QUARANTINE_CONFIDENCE_CAP,
        })

        if not dry_run:
            concept = load_concept(concept_id, track_access=False)
            if concept and concept.confidence > PSIS_QUARANTINE_CONFIDENCE_CAP:
                original_ka = concept.metadata.get("knowledge_area") if concept.metadata else None
                concept.confidence = PSIS_QUARANTINE_CONFIDENCE_CAP
                # KA-005: Preserve knowledge_area through save
                if original_ka:
                    concept.knowledge_area = original_ka
                    if concept.metadata:
                        concept.metadata["knowledge_area"] = original_ka
                save_concept(concept)
                capped += 1
                logger.info(
                    "M3 sweep: capped %s from %.4f to %.4f",
                    concept_id, old_confidence, PSIS_QUARANTINE_CONFIDENCE_CAP,
                )

    result = {
        "found": len(rows),
        "capped": capped,
        "dry_run": dry_run,
        "details": details[:20],  # Cap output for readability
    }

    print(f"{'Would cap' if dry_run else 'Capped'}: {capped if not dry_run else len(rows)} concepts")
    return result


if __name__ == "__main__":
    # Dry run first
    print("=== DRY RUN ===")
    dry_result = migrate(dry_run=True)
    for d in dry_result["details"]:
        print(f"  {d['id']}: {d['old_confidence']} -> {d['new_confidence']}")

    if dry_result["found"] > 0:
        print(f"\n=== EXECUTING ({dry_result['found']} concepts) ===")
        result = migrate(dry_run=False)
        print(f"Done: {result['capped']} capped")
