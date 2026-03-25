"""Schema migration 003: Cognitive Domains & Behavioral Directives.

Adds tables for Layer 3 abstraction:
- cognitive_domains: Business-meaningful context groupings
- domain_area_mapping: Maps knowledge_areas to domains with activation weights
- directives: Provider-agnostic behavioral instructions (the "soul")
- directive_versions: Append-only audit trail for directive changes

Per spec: DOMAINS_AND_DIRECTIVES_SPEC.md (Section 4)
All CREATE TABLE IF NOT EXISTS — idempotent, safe on every startup.
"""

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

# NOTE: DB path resolution is handled by app.profile.resolve_data_dir().
# The bash CLI mirrors this logic in resolve_db_path().
# See: app/profile.py (Python source of truth)
# See: scripts/install.sh resolve_db_path() (bash mirror)


TABLES_DDL = """
-- Cognitive Domains (Layer 3 groupings)
CREATE TABLE IF NOT EXISTS cognitive_domains (
    domain_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    activation_triggers TEXT NOT NULL,
    strategic_priority REAL DEFAULT 0.5 CHECK(strategic_priority >= 0.0 AND strategic_priority <= 1.0),
    active INTEGER DEFAULT 1,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

-- Domain-to-knowledge_area mapping
CREATE TABLE IF NOT EXISTS domain_area_mapping (
    domain_id TEXT NOT NULL,
    knowledge_area TEXT NOT NULL,
    activation_weight REAL DEFAULT 0.3 CHECK(activation_weight >= 0.0 AND activation_weight <= 1.0),
    PRIMARY KEY (domain_id, knowledge_area),
    FOREIGN KEY (domain_id) REFERENCES cognitive_domains(domain_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_dam_area ON domain_area_mapping(knowledge_area);

-- Behavioral Directives (provider-agnostic soul)
CREATE TABLE IF NOT EXISTS directives (
    directive_id TEXT PRIMARY KEY,
    category TEXT NOT NULL CHECK(category IN ('persona', 'workflow', 'constraints', 'formatting', 'domain_rules')),
    content TEXT NOT NULL CHECK(length(content) <= 2000),
    priority INTEGER DEFAULT 100,
    active INTEGER DEFAULT 1,
    version INTEGER DEFAULT 1,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_directives_category ON directives(category);
CREATE INDEX IF NOT EXISTS idx_directives_active ON directives(active);

-- Directive version history (append-only audit trail)
CREATE TABLE IF NOT EXISTS directive_versions (
    directive_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    content TEXT NOT NULL,
    category TEXT NOT NULL,
    priority INTEGER NOT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (directive_id, version),
    FOREIGN KEY (directive_id) REFERENCES directives(directive_id) ON DELETE CASCADE
);
"""


def _create_tables(conn: sqlite3.Connection):
    """Execute all DDL statements."""
    for statement in TABLES_DDL.strip().split(";"):
        statement = statement.strip()
        if statement:
            try:
                conn.execute(statement)
            except sqlite3.OperationalError as e:
                err = str(e).lower()
                if "already exists" in err or "duplicate" in err:
                    pass  # Idempotent
                else:
                    raise


def run_migration(conn: sqlite3.Connection = None):
    """Run domains & directives schema migration. Idempotent.

    Args:
        conn: Optional existing DB connection. If None, opens a new connection
              using the profile-aware path from app.profile.resolve_data_dir().
    """
    close_conn = False
    if conn is None:
        try:
            from app.profile import resolve_data_dir
            data_dir = resolve_data_dir()
        except ImportError:
            # Fallback: if profile module unavailable (standalone migration),
            # use default path
            data_dir = Path.home() / "pith-data" / "default"
            logger.warning(
                f"Migration 003: app.profile unavailable, using default: {data_dir}"
            )
        # Check for pith.db first (post-migration), fall back to brain.db (pre-migration)
        pith_path = data_dir / "pith.db"
        brain_path = data_dir / "brain.db"
        db_path = pith_path if pith_path.exists() else pith_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        close_conn = True
        logger.info(f"Migration 003: opened DB at {db_path}")
    try:
        _create_tables(conn)
        conn.commit()
        logger.info("Migration 003 (domains_directives): complete")
    except Exception as e:
        logger.error(f"Migration 003 failed: {e}", exc_info=True)
        raise
    finally:
        if close_conn:
            conn.close()
