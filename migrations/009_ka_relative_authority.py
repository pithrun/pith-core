"""Migration 009: Add ka_relative_authority column (Federation Phase 0).

Component 0.1 of FEDERATION_ORCHESTRATION_DESIGN v2.1.
Stores precomputed KA-relative percentile rank alongside global authority_score.
Computed during batch_compute_authority(), read at retrieval time.
"""

import logging
import sqlite3

logger = logging.getLogger(__name__)

MIGRATION_ID = "009_ka_relative_authority"


def migrate(conn: sqlite3.Connection, *, dry_run: bool = False) -> dict:
    """Add ka_relative_authority column to concepts table."""
    # Check if column already exists
    cursor = conn.execute("PRAGMA table_info(concepts)")
    columns = {row[1] for row in cursor.fetchall()}

    result = {"column_added": False, "dry_run": dry_run}

    if "ka_relative_authority" in columns:
        logger.info("ka_relative_authority column already exists, skipping")
        result["column_added"] = False
        return result

    if dry_run:
        logger.info("DRY-RUN: Would add ka_relative_authority REAL DEFAULT NULL to concepts")
        result["column_added"] = True
        return result

    conn.execute("ALTER TABLE concepts ADD COLUMN ka_relative_authority REAL DEFAULT NULL")
    conn.commit()

    logger.info("Added ka_relative_authority column to concepts table")
    result["column_added"] = True
    return result
