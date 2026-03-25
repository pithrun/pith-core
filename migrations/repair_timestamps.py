#!/usr/bin/env python3
"""STABILITY-009: Repair NULL last_accessed timestamps using 3-tier fallback.

Tier 1: MAX(created_at) from concept_versions for this concept
Tier 2: MAX(created_at) from governance_events for this concept
Tier 3: Current UTC time (last resort)

Safe to re-run: only updates rows where last_accessed IS NULL or empty.
Uses BEGIN EXCLUSIVE to prevent race conditions (Gauntlet Amendment A3).
Creates backup before migration (Gauntlet Amendment A4).
"""
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

# Default DB path; override via CLI argument
DEFAULT_DB = Path.home() / "pith-data" / "rose" / "pith.db"


def repair_timestamps(db_path: str) -> dict:
    """Run the 3-tier timestamp repair migration.

    Returns dict with counts:
        before_null, tier1_concept_versions, tier2_governance_events,
        tier3_utc_now, after_null, total_active, backup_path.
    """
    db_file = Path(db_path)

    # Gauntlet Amendment A4: Create backup before migration
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = (
        db_file.parent
        / f"{db_file.stem}.pre-stability009-{timestamp}{db_file.suffix}"
    )
    shutil.copy2(db_path, backup_path)
    print(f"Backup created: {backup_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Snapshot before
    before = conn.execute(
        "SELECT COUNT(*) as cnt FROM concepts "
        "WHERE status='active' AND (last_accessed IS NULL OR last_accessed='')"
    ).fetchone()["cnt"]

    now_utc = datetime.now(timezone.utc).isoformat()
    tier1 = tier2 = tier3 = 0

    try:
        conn.execute("BEGIN EXCLUSIVE")

        # Tier 1: concept_versions fallback
        result = conn.execute("""
            UPDATE concepts SET last_accessed = (
                SELECT MAX(cv.created_at) FROM concept_versions cv
                WHERE cv.id = concepts.id
                AND cv.created_at IS NOT NULL AND cv.created_at != ''
            )
            WHERE status = 'active'
            AND (last_accessed IS NULL OR last_accessed = '')
            AND id IN (
                SELECT DISTINCT id FROM concept_versions
                WHERE created_at IS NOT NULL AND created_at != ''
            )
        """)
        tier1 = result.rowcount

        # Tier 2: governance_events fallback (for any remaining NULLs)
        result = conn.execute("""
            UPDATE concepts SET last_accessed = (
                SELECT MAX(ge.created_at) FROM governance_events ge
                WHERE ge.concept_id = concepts.id
                AND ge.created_at IS NOT NULL AND ge.created_at != ''
            )
            WHERE status = 'active'
            AND (last_accessed IS NULL OR last_accessed = '')
            AND id IN (
                SELECT DISTINCT concept_id FROM governance_events
                WHERE created_at IS NOT NULL AND created_at != ''
            )
        """)
        tier2 = result.rowcount

        # Tier 3: UTC now fallback (absolute last resort)
        result = conn.execute("""
            UPDATE concepts SET last_accessed = ?
            WHERE status = 'active'
            AND (last_accessed IS NULL OR last_accessed = '')
        """, (now_utc,))
        tier3 = result.rowcount

        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        conn.close()  # DEBT-144: prevent connection leak on transaction failure
        raise

    # Verify: no NULLs remaining
    after = conn.execute(
        "SELECT COUNT(*) as cnt FROM concepts "
        "WHERE status='active' AND (last_accessed IS NULL OR last_accessed='')"
    ).fetchone()["cnt"]

    total_active = conn.execute(
        "SELECT COUNT(*) FROM concepts WHERE status='active'"
    ).fetchone()[0]

    conn.close()

    return {
        "before_null": before,
        "tier1_concept_versions": tier1,
        "tier2_governance_events": tier2,
        "tier3_utc_now": tier3,
        "after_null": after,
        "total_active": total_active,
        "backup_path": str(backup_path),
    }


if __name__ == "__main__":
    db = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_DB)
    print(f"Running STABILITY-009 timestamp repair on: {db}")
    result = repair_timestamps(db)
    print(f"Before: {result['before_null']} NULL timestamps")
    print(f"Tier 1 (concept_versions): {result['tier1_concept_versions']} repaired")
    print(f"Tier 2 (governance_events): {result['tier2_governance_events']} repaired")
    print(f"Tier 3 (utc_now fallback):  {result['tier3_utc_now']} repaired")
    print(f"After:  {result['after_null']} NULL timestamps remaining")
    print(f"Total active concepts: {result['total_active']}")
    print(f"Backup at: {result['backup_path']}")
    if result["after_null"] > 0:
        print("WARNING: Some NULL timestamps could not be repaired!")
        sys.exit(1)
    print("SUCCESS: All timestamps repaired.")
