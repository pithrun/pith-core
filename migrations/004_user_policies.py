"""Schema migration 004: User Policies.

Phase 3 completion: Adds user_policies table for user-configurable
pith behavior rules. Integrates with PolicyEngine and PolicyCache.

Per spec: ORIENTATION_V2_AND_PHASE3_COMPLETION_SPEC.md (Section B.1)
CREATE TABLE IF NOT EXISTS — idempotent, safe on every startup.
"""

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


TABLES_DDL = """
-- User Policies (Phase 3: user-configurable pith rules)
CREATE TABLE IF NOT EXISTS user_policies (
    id TEXT PRIMARY KEY,
    policy_type TEXT NOT NULL,       -- 'retention', 'privacy', 'behavior'
    rule TEXT NOT NULL,              -- human-readable rule description
    condition JSON,                  -- when to apply (optional filter)
    action JSON NOT NULL,            -- what to do when triggered
    enabled INTEGER DEFAULT 1,       -- soft delete: 0 = disabled
    priority INTEGER DEFAULT 50,     -- higher = evaluated first
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_user_policies_type
    ON user_policies(policy_type);
CREATE INDEX IF NOT EXISTS idx_user_policies_enabled
    ON user_policies(enabled);
"""


def _create_tables(conn: sqlite3.Connection):
    conn.executescript(TABLES_DDL)
    logger.info("Migration 004: user_policies table created")


def run_migration(conn: sqlite3.Connection = None):
    """Run user policies schema migration. Idempotent."""
    close_conn = False
    if conn is None:
        try:
            from app.profile import resolve_data_dir
            data_dir = resolve_data_dir()
        except ImportError:
            data_dir = Path.home() / "pith-data" / "default"
            logger.warning(
                f"Migration 004: app.profile unavailable, using default: {data_dir}"
            )
        # Check for pith.db first (post-migration), fall back to brain.db (pre-migration)
        pith_path = data_dir / "pith.db"
        brain_path = data_dir / "brain.db"
        db_path = pith_path if pith_path.exists() else pith_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        close_conn = True
        logger.info(f"Migration 004: opened DB at {db_path}")
    try:
        _create_tables(conn)
        conn.commit()
        logger.info("Migration 004 (user_policies): complete")
    except Exception as e:
        logger.error(f"Migration 004 failed: {e}")
        raise
    finally:
        if close_conn:
            conn.close()
