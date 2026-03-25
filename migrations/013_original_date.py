"""Migration 013: Add original_date column to concepts.

TEMPORAL-002: Stores ISO-8601 partial date extracted from concept text.
E.g. "moved to SF in March 2025" → original_date = "2025-03".
Used by temporal reasoning to anchor concepts in historical time.
"""

import logging

logger = logging.getLogger(__name__)


def migrate(conn):
    """Add original_date column to concepts table."""
    columns = [row[1] for row in conn.execute("PRAGMA table_info(concepts)").fetchall()]
    if "original_date" in columns:
        logger.info("Migration 013: original_date already exists, skipping")
        return

    conn.execute("ALTER TABLE concepts ADD COLUMN original_date TEXT DEFAULT NULL")

    logger.info("Migration 013: Added original_date column (no backfill — extracted at ingest time)")
