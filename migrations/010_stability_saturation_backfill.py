"""Migration 010: STABILITY-SAT Component C — One-time stability backfill.

Resets stability to TIME_MATURATION_MAX_STABILITY (0.8) for active concepts
that were inflated above 0.8 by the uncapped _strengthen_accessed() boost.

Only targets v1 concepts (never evolved) — evolved concepts earned their
stability through the evolution pipeline and should be preserved.

Amendment A1: Creates a rollback snapshot before modifying data.
"""

import logging
import os
import shutil
import sqlite3

logger = logging.getLogger(__name__)

MIGRATION_ID = "010_stability_saturation_backfill"
TARGET_STABILITY = 0.8  # TIME_MATURATION_MAX_STABILITY


def migrate(conn: sqlite3.Connection, *, dry_run: bool = False) -> dict:
    """Reset inflated stability for v1 concepts to 0.8."""
    result = {
        "dry_run": dry_run,
        "snapshot_path": None,
        "eligible_count": 0,
        "updated_count": 0,
        "avg_stability_before": 0.0,
        "avg_stability_after": 0.0,
    }

    # Amendment A1: Rollback snapshot before any writes
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    if db_path and not dry_run:
        snapshot_path = db_path + ".pre_stability_sat_backfill"
        if not os.path.exists(snapshot_path):
            shutil.copy2(db_path, snapshot_path)
            logger.info(f"Created rollback snapshot: {snapshot_path}")
            result["snapshot_path"] = snapshot_path
        else:
            logger.info(f"Snapshot already exists: {snapshot_path}")
            result["snapshot_path"] = snapshot_path

    # Count eligible concepts: active, stability > 0.8, version = 'v1' (never evolved)
    cursor = conn.execute("""
        SELECT COUNT(*), AVG(stability)
        FROM concepts
        WHERE status = 'active'
          AND stability > ?
          AND version = 'v1'
    """, (TARGET_STABILITY,))
    row = cursor.fetchone()
    eligible_count = row[0] or 0
    avg_before = row[1] or 0.0

    result["eligible_count"] = eligible_count
    result["avg_stability_before"] = round(avg_before, 4)

    logger.info(
        f"STABILITY-SAT backfill: {eligible_count} eligible concepts "
        f"(active, stability > {TARGET_STABILITY}, version = 'v1'), "
        f"avg stability = {avg_before:.4f}"
    )

    if eligible_count == 0:
        logger.info("No concepts need backfill — skipping")
        return result

    if dry_run:
        logger.info(f"DRY-RUN: Would reset {eligible_count} concepts to stability = {TARGET_STABILITY}")
        return result

    # Execute the backfill — update BOTH column AND JSON data blob.
    # FIX: The JSON `data` blob is the source of truth for load_concept().
    # Updating only the column causes reflection to restore old values from blob.
    conn.execute("""
        UPDATE concepts
        SET stability = ?,
            data = json_set(data, '$.stability', ?)
        WHERE status = 'active'
          AND stability > ?
          AND version = 'v1'
    """, (TARGET_STABILITY, TARGET_STABILITY, TARGET_STABILITY))
    conn.commit()

    # Verify
    cursor = conn.execute("""
        SELECT COUNT(*), AVG(stability)
        FROM concepts
        WHERE status = 'active'
          AND version = 'v1'
    """)
    post_row = cursor.fetchone()
    result["updated_count"] = eligible_count
    result["avg_stability_after"] = round((post_row[1] or 0.0), 4)

    logger.info(
        f"STABILITY-SAT backfill complete: {eligible_count} concepts "
        f"reset to {TARGET_STABILITY}. Avg stability: {avg_before:.4f} → {result['avg_stability_after']:.4f}"
    )

    return result
