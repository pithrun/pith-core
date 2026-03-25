"""Migration 005: Add content_updated_at column to concepts.

DATA-020: Tracks when summary actually changes (vs updated_at which
changes on every access/touch). Used by STABILITY-012 for accurate
freshness detection.
"""

import logging

logger = logging.getLogger(__name__)


def migrate(conn):
    """Add content_updated_at column to concepts table."""
    # Check if column already exists
    columns = [row[1] for row in conn.execute("PRAGMA table_info(concepts)").fetchall()]
    if "content_updated_at" in columns:
        logger.info("Migration 005: content_updated_at already exists, skipping")
        return

    conn.execute("ALTER TABLE concepts ADD COLUMN content_updated_at TEXT DEFAULT NULL")

    # Backfill: set content_updated_at = created_at for all existing concepts
    # (conservative: we don't know when summary last changed)
    conn.execute("UPDATE concepts SET content_updated_at = created_at WHERE content_updated_at IS NULL")

    # Create index for STABILITY-012 freshness queries
    conn.execute("CREATE INDEX IF NOT EXISTS idx_concepts_content_updated_at ON concepts(content_updated_at)")

    logger.info("Migration 005: Added content_updated_at column and backfilled from created_at")
