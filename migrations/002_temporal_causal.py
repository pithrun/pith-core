#!/usr/bin/env python3
"""
Migration 002: Temporal Retrieval + Cognitive Graph Enrichment

Adds:
- valid_from, valid_until, superseded_by, supersession_reason on concepts
- mechanism, direction, chain_id on associations
- classification_log table for router observability
- cko_edges table for CKO internal graph
- Temporal + causal indexes
- Backfill: valid_from = created_at for existing concepts [H3]

Safe: All operations use IF NOT EXISTS / column detection.
Idempotent: Re-running is a no-op.

Spec refs: TEMPORAL_RETRIEVAL_SPEC.md §10.1, §10.2, §10.3, §17.4
"""

import sqlite3
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from app.profile import resolve_data_dir
    DATA_DIR = str(resolve_data_dir())
except ImportError:
    DATA_DIR = os.environ.get("PITH_DATA_DIR", os.environ.get("DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")))
# Check for pith.db first (post-migration), fall back to brain.db (pre-migration)
_pith_path = os.path.join(DATA_DIR, "pith.db")
_brain_path = os.path.join(DATA_DIR, "brain.db")
DB_PATH = _pith_path if os.path.exists(_pith_path) else _pith_path


def get_existing_columns(conn: sqlite3.Connection, table: str) -> set:
    """Return set of column names for a table."""
    cursor = conn.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cursor.fetchall()}


def run_migration(db_path: str = None):
    """Execute migration 002."""
    path = db_path or DB_PATH
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    
    migrated = []
    
    try:
        # --- 1. Concepts: temporal-relational columns [§10.2] ---
        concept_cols = get_existing_columns(conn, "concepts")
        
        if "valid_from" not in concept_cols:
            conn.execute("ALTER TABLE concepts ADD COLUMN valid_from TEXT DEFAULT NULL")
            migrated.append("concepts.valid_from")
        
        if "valid_until" not in concept_cols:
            conn.execute("ALTER TABLE concepts ADD COLUMN valid_until TEXT DEFAULT NULL")
            migrated.append("concepts.valid_until")
        
        if "superseded_by" not in concept_cols:
            conn.execute("ALTER TABLE concepts ADD COLUMN superseded_by TEXT DEFAULT NULL")
            migrated.append("concepts.superseded_by")
        
        if "supersession_reason" not in concept_cols:
            conn.execute("ALTER TABLE concepts ADD COLUMN supersession_reason TEXT DEFAULT NULL")
            migrated.append("concepts.supersession_reason")
        
        # --- 2. Associations: causal chain columns [§10.1] ---
        assoc_cols = get_existing_columns(conn, "associations")
        
        if "mechanism" not in assoc_cols:
            conn.execute("ALTER TABLE associations ADD COLUMN mechanism TEXT DEFAULT NULL")
            migrated.append("associations.mechanism")
        
        if "direction" not in assoc_cols:
            conn.execute("""ALTER TABLE associations ADD COLUMN direction TEXT DEFAULT 'bidirectional'""")
            migrated.append("associations.direction")
        
        if "chain_id" not in assoc_cols:
            conn.execute("ALTER TABLE associations ADD COLUMN chain_id TEXT DEFAULT NULL")
            migrated.append("associations.chain_id")
        
        # --- 3. Classification log table [§17.4, M52] ---
        conn.execute("""
            CREATE TABLE IF NOT EXISTS classification_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TEXT DEFAULT (datetime('now')),
                input_source TEXT,
                input_length INTEGER,
                classification TEXT NOT NULL,
                confidence REAL,
                was_overridden INTEGER DEFAULT 0,
                override_value TEXT DEFAULT NULL,
                supplementary_executed INTEGER DEFAULT 0,
                supplementary_latency_ms REAL DEFAULT NULL,
                supplementary_concepts_added INTEGER DEFAULT 0,
                post_hoc_warning INTEGER DEFAULT 0
            )
        """)
        migrated.append("classification_log table")
        
        # --- 4. CKO internal edges table [§10.3, M3] ---
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cko_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cko_id TEXT NOT NULL,
                from_concept TEXT NOT NULL,
                to_concept TEXT NOT NULL,
                relation TEXT NOT NULL,
                mechanism TEXT DEFAULT NULL,
                UNIQUE(cko_id, from_concept, to_concept, relation)
            )
        """)
        migrated.append("cko_edges table")
        
        # --- 5. Indexes [L2] ---
        indexes = [
            ("idx_concepts_created_at", "concepts", "created_at"),
            ("idx_concepts_updated_at", "concepts", "updated_at"),
            ("idx_concepts_valid_from", "concepts", "valid_from"),
            ("idx_concepts_valid_until", "concepts", "valid_until"),
            ("idx_concepts_superseded_by", "concepts", "superseded_by"),
            ("idx_assoc_chain_id", "associations", "chain_id"),
            ("idx_assoc_direction", "associations", "direction"),
            ("idx_assoc_relation", "associations", "relation"),
            ("idx_classification_log_ts", "classification_log", "timestamp"),
            ("idx_classification_log_class", "classification_log", "classification"),
            ("idx_cko_edges_cko_id", "cko_edges", "cko_id"),
        ]
        
        for idx_name, table, column in indexes:
            conn.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table}({column})")
        migrated.append(f"{len(indexes)} indexes")
        
        # --- 6. Backfill: valid_from = created_at [H3, P0 migration] ---
        cursor = conn.execute(
            "UPDATE concepts SET valid_from = created_at WHERE valid_from IS NULL"
        )
        backfilled = cursor.rowcount
        if backfilled > 0:
            migrated.append(f"backfilled valid_from on {backfilled} concepts")
        
        conn.commit()
        
        summary = f"Migration 002 complete: {', '.join(migrated)}"
        logger.info(summary)
        print(summary)
        return {"status": "success", "changes": migrated, "backfilled": backfilled}
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Migration 002 failed: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    import sys
    db = sys.argv[1] if len(sys.argv) > 1 else None
    run_migration(db)
