"""Migration 007: Create evolution backfill staging table.

RETRIEVAL-019: Staging table for progressive evolution backfill pipeline.
Stores candidate pairs, evaluation scores, and commit/rollback tracking.
"""

import logging

logger = logging.getLogger(__name__)


def migrate(conn):
    """Create the evolution_backfill_staging table."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS evolution_backfill_staging (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id TEXT NOT NULL,
            older_concept_id TEXT NOT NULL,
            newer_concept_id TEXT NOT NULL,
            cosine_score REAL NOT NULL,
            composite_score REAL NOT NULL,
            authority_delta REAL,
            type_rank_delta INTEGER,
            content_age_days REAL,
            rationale TEXT,
            execute_result TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            reviewed_at TEXT,
            committed_at TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_backfill_batch ON evolution_backfill_staging(batch_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_backfill_status ON evolution_backfill_staging(status)")
    conn.commit()
    logger.info("Migration 007: Created evolution_backfill_staging table")
