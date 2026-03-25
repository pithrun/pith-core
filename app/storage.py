"""SQLite storage backend for Pith.

Replaces YAML filesystem storage with single SQLite database.
All function signatures preserved for consumer compatibility except:
  - save_concept() returns None (no callers used returned Path)
  - knowledge_area CRUD removed (derived via DISTINCT query)
  - AccessTracker replaced with compat shim (direct DB writes in load_concept)

Phase 1B P0.2: Migration from YAML to SQLite for 10-20x performance improvement.
"""

import json
import logging
import math
import os
import re
import sqlite3
import threading
from datetime import UTC, timedelta
from pathlib import Path

from app.datetime_utils import _utc_now, _utc_now_iso

# AGENT-001: agent_id validation
_AGENT_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")

# DATA-028: advisory lock for restore_concept to prevent concurrent double-restore races
_restore_concept_lock = threading.Lock()


def validate_agent_id(agent_id: str) -> str:
    """Validate and normalize agent_id. Returns validated value or 'default'."""
    if not agent_id or not isinstance(agent_id, str):
        return "default"
    agent_id = agent_id.strip()
    if not agent_id or not _AGENT_ID_PATTERN.match(agent_id):
        logger.warning(f"Invalid agent_id rejected: {agent_id[:20]!r}")
        return "default"
    return agent_id


def _clamp_score(value, low: float = 0.0, high: float = 1.0):
    """DEBT-182: Clamp a score to [low, high]. Returns None if input is None."""
    if value is None:
        return None
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return None


from contextlib import contextmanager, suppress

from app.models import Concept

logger = logging.getLogger(__name__)


def _safe_json_loads(data, context: str = "unknown") -> dict | None:
    """Bug 7 fix: Safely decode JSON data column with UTF-8 error handling.

    SQLite may return bytes or str depending on the data. If the data column
    has corrupted UTF-8 bytes, this replaces bad characters instead of crashing.
    Returns None if the data cannot be decoded at all.
    """
    if data is None:
        return None
    try:
        if isinstance(data, bytes):
            data = data.decode("utf-8", errors="replace")
        return json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.error(f"Bug 7: Corrupted data in {context}: {type(e).__name__}: {e}")
        return None


# --- Schema DDL ---
# Extracted from migrate_to_sqlite.py so fresh installs create tables automatically.
# Uses CREATE TABLE IF NOT EXISTS — safe to run on every startup.
SCHEMA_DDL = """
-- Core concepts table (latest version only, hot reads)
CREATE TABLE IF NOT EXISTS concepts (
    id TEXT PRIMARY KEY,
    version TEXT NOT NULL,
    summary TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.5,
    stability REAL NOT NULL DEFAULT 0.5,
    knowledge_area TEXT,
    concept_type TEXT DEFAULT 'insight',
    status TEXT NOT NULL DEFAULT 'active',
    salience REAL NOT NULL DEFAULT 0.5,
    salience_source TEXT NOT NULL DEFAULT 'system',
    maturity TEXT NOT NULL DEFAULT 'ESTABLISHED',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_accessed TEXT,
    last_organic_access TEXT,
    access_count INTEGER DEFAULT 0,
    data JSON NOT NULL,
    embedding BLOB,
    embedding_version INTEGER DEFAULT 0,
    always_activate INTEGER DEFAULT 0,
    authority_score REAL DEFAULT NULL CHECK(authority_score IS NULL OR (authority_score >= 0.0 AND authority_score <= 1.0)),
    currency_score REAL DEFAULT NULL CHECK(currency_score IS NULL OR (currency_score >= 0.0 AND currency_score <= 1.0)),
    currency_status TEXT DEFAULT 'ACTIVE',
    last_authority_recompute TEXT DEFAULT NULL,
    last_currency_recompute TEXT DEFAULT NULL,
    valid_from DATETIME DEFAULT NULL,
    valid_until DATETIME DEFAULT NULL,
    superseded_by TEXT DEFAULT NULL,
    supersession_reason TEXT DEFAULT NULL,
    epistemic_network TEXT DEFAULT 'assessment',
    verification_status TEXT DEFAULT 'unverified',
    verification_fraction REAL DEFAULT 0.0,
    effective_authority REAL DEFAULT NULL,
    ka_relative_authority REAL DEFAULT NULL,
    is_current INTEGER DEFAULT 1,
    superseded_at TEXT DEFAULT NULL,
    version_chain_head TEXT DEFAULT NULL,
    reinforcement_count INTEGER DEFAULT 0,
    agent_id TEXT NOT NULL DEFAULT 'default',
    session_id TEXT DEFAULT NULL,
    content_updated_at TEXT DEFAULT NULL,
    original_date TEXT DEFAULT NULL,
    protected INTEGER DEFAULT 0,
    fragment_keywords TEXT DEFAULT NULL
);

CREATE INDEX IF NOT EXISTS idx_concepts_session ON concepts(session_id);
-- FTS5 full-text index for BM25 keyword search (RETRIEVAL-042 upgrade)
CREATE VIRTUAL TABLE IF NOT EXISTS fts_concepts
    USING fts5(concept_id UNINDEXED, summary, tokenize='porter ascii');


-- Version history (all versions, append-only)
CREATE TABLE IF NOT EXISTS concept_versions (
    id TEXT NOT NULL,
    version TEXT NOT NULL,
    data JSON NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (id, version)
);

-- Association edges
CREATE TABLE IF NOT EXISTS associations (
    source TEXT NOT NULL,
    target TEXT NOT NULL,
    relation TEXT NOT NULL,
    strength REAL DEFAULT 0.5,
    created_at TEXT NOT NULL,
    mechanism TEXT DEFAULT NULL,
    direction TEXT DEFAULT 'bidirectional',
    chain_id TEXT DEFAULT NULL,
    PRIMARY KEY (source, target, relation)
);

-- Self-model (singleton + version history)
CREATE TABLE IF NOT EXISTS self_model (
    id TEXT PRIMARY KEY DEFAULT 'current',
    version INTEGER NOT NULL DEFAULT 1,
    data JSON NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS self_model_versions (
    version INTEGER PRIMARY KEY,
    data JSON NOT NULL,
    created_at TEXT NOT NULL
);

-- Sessions
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    learning_event_count INTEGER DEFAULT 0,
    context_hint TEXT,
    agent_id TEXT NOT NULL DEFAULT 'default',
    data JSON,
    concepts_created INTEGER DEFAULT 0,
    concepts_evolved INTEGER DEFAULT 0,
    model_id TEXT DEFAULT NULL,
    pressure_score REAL DEFAULT NULL,
    last_learning_at TEXT DEFAULT NULL,
    last_heartbeat TEXT DEFAULT NULL,
    working_context_json TEXT DEFAULT NULL
);

-- Execution checkpoints (ephemeral resumption state, NOT concepts)
CREATE TABLE IF NOT EXISTS checkpoints (
    task_id TEXT PRIMARY KEY,
    session_id TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    description TEXT NOT NULL,
    done TEXT DEFAULT '[]',
    active TEXT DEFAULT '',
    next TEXT DEFAULT '[]',
    blockers TEXT DEFAULT '[]',
    context TEXT DEFAULT '{}',
    concept_refs TEXT DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    save_count INTEGER DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_status ON checkpoints(status);
CREATE INDEX IF NOT EXISTS idx_checkpoints_updated ON checkpoints(updated_at);

-- Key-value metadata
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Indexes for hot queries
CREATE INDEX IF NOT EXISTS idx_concepts_knowledge_area ON concepts(knowledge_area);
CREATE INDEX IF NOT EXISTS idx_concepts_confidence ON concepts(confidence);
CREATE INDEX IF NOT EXISTS idx_concepts_last_accessed ON concepts(last_accessed);
CREATE INDEX IF NOT EXISTS idx_concepts_stability ON concepts(stability);
CREATE INDEX IF NOT EXISTS idx_concepts_status ON concepts(status);
CREATE INDEX IF NOT EXISTS idx_concepts_salience ON concepts(salience);
CREATE INDEX IF NOT EXISTS idx_concepts_salience_source ON concepts(salience_source);
CREATE INDEX IF NOT EXISTS idx_concepts_maturity ON concepts(maturity);
CREATE INDEX IF NOT EXISTS idx_concepts_concept_type ON concepts(concept_type);
CREATE INDEX IF NOT EXISTS idx_concepts_created_at ON concepts(created_at);
CREATE INDEX IF NOT EXISTS idx_concepts_updated_at ON concepts(updated_at);
CREATE INDEX IF NOT EXISTS idx_concepts_currency_status ON concepts(currency_status);
CREATE INDEX IF NOT EXISTS idx_concepts_epistemic ON concepts(epistemic_network);
CREATE INDEX IF NOT EXISTS idx_concepts_is_current ON concepts(is_current);
CREATE INDEX IF NOT EXISTS idx_concepts_superseded_by ON concepts(superseded_by);
CREATE INDEX IF NOT EXISTS idx_concepts_valid_from ON concepts(valid_from);
CREATE INDEX IF NOT EXISTS idx_concepts_valid_until ON concepts(valid_until);
CREATE INDEX IF NOT EXISTS idx_concepts_verification ON concepts(verification_status);
CREATE INDEX IF NOT EXISTS idx_concepts_version_chain ON concepts(version_chain_head);
CREATE INDEX IF NOT EXISTS idx_concepts_strategic ON concepts(authority_score, currency_score);
CREATE INDEX IF NOT EXISTS idx_concepts_temporal_snapshot ON concepts(knowledge_area, created_at, confidence);
CREATE INDEX IF NOT EXISTS idx_assoc_source ON associations(source);
CREATE INDEX IF NOT EXISTS idx_assoc_target ON associations(target);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
CREATE INDEX IF NOT EXISTS idx_sessions_heartbeat ON sessions(last_heartbeat) WHERE status = 'active';

-- Firmware table: static developer-controlled operational knowledge (P0-5)
-- Separate from concepts table — firmware shares zero code paths with concepts.
-- Only written by seed_firmware.py on server startup. Never mutated at runtime.
CREATE TABLE IF NOT EXISTS firmware (
    id TEXT PRIMARY KEY,
    summary TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'general',
    firmware_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Reflection tracking: automatic metacognitive extraction compliance (Auto-Reflection spec §8)
CREATE TABLE IF NOT EXISTS reflection_tracking (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    trigger_type TEXT NOT NULL,
    prompts_sent INTEGER NOT NULL DEFAULT 0,
    concepts_returned INTEGER NOT NULL DEFAULT 0,
    reflection_quality TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    prompt_data TEXT
);

CREATE INDEX IF NOT EXISTS idx_reflection_session ON reflection_tracking(session_id);
CREATE INDEX IF NOT EXISTS idx_reflection_trigger ON reflection_tracking(trigger_type);

-- Wave 4b: Cognitive traces (structured learning event log)
CREATE TABLE IF NOT EXISTS traces (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    trigger_type TEXT NOT NULL,
    concept_refs TEXT DEFAULT '[]',
    agent_id TEXT DEFAULT 'default',
    data JSON NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_traces_session ON traces(session_id);
CREATE INDEX IF NOT EXISTS idx_traces_trigger ON traces(trigger_type);

-- Wave 4b: Prediction tracking for confidence calibration
CREATE TABLE IF NOT EXISTS predictions (
    id TEXT PRIMARY KEY,
    concept_id TEXT NOT NULL,
    confidence_at_retrieval REAL NOT NULL,
    retrieved_at TEXT NOT NULL,
    session_id TEXT NOT NULL,
    outcome TEXT DEFAULT 'pending',
    outcome_at TEXT,
    outcome_source TEXT DEFAULT 'evolution'
);

CREATE INDEX IF NOT EXISTS idx_predictions_concept ON predictions(concept_id);
CREATE INDEX IF NOT EXISTS idx_predictions_session ON predictions(session_id);
CREATE INDEX IF NOT EXISTS idx_predictions_outcome ON predictions(outcome);

-- Wave 5: Narrative threads
CREATE TABLE IF NOT EXISTS threads (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_activity_at TEXT NOT NULL,
    completed_at TEXT,
    urgency TEXT DEFAULT 'normal',
    agent_id TEXT DEFAULT 'default',
    data JSON NOT NULL
);

CREATE TABLE IF NOT EXISTS thread_concept_links (
    thread_id TEXT NOT NULL,
    concept_id TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'member',
    added_at TEXT NOT NULL,
    added_by TEXT DEFAULT 'system',
    PRIMARY KEY (thread_id, concept_id)
);

CREATE INDEX IF NOT EXISTS idx_threads_status ON threads(status);
CREATE INDEX IF NOT EXISTS idx_threads_agent ON threads(agent_id);
CREATE INDEX IF NOT EXISTS idx_threads_last_activity ON threads(last_activity_at);
CREATE INDEX IF NOT EXISTS idx_thread_links_concept ON thread_concept_links(concept_id);
CREATE INDEX IF NOT EXISTS idx_thread_links_thread ON thread_concept_links(thread_id);
CREATE INDEX IF NOT EXISTS idx_traces_created ON traces(created_at);

-- AGENT-002: Agent tokens for MCP HTTP access
CREATE TABLE IF NOT EXISTS agent_tokens (
    token TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    label TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    revoked_at TEXT,
    last_used_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_agent_tokens_agent ON agent_tokens(agent_id);

-- Wave 6: Experiments (cognitive experiment engine)
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    experiment_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    candidates TEXT DEFAULT '[]',
    result TEXT,
    concept_ids_produced TEXT DEFAULT '[]',
    cko_ids_produced TEXT DEFAULT '[]',
    thread_id TEXT,
    config_snapshot TEXT DEFAULT '{}',
    generation_time_ms INTEGER,
    processing_time_ms INTEGER,
    metadata TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_experiments_type ON experiments(experiment_type);
CREATE INDEX IF NOT EXISTS idx_experiments_updated ON experiments(updated_at);
CREATE INDEX IF NOT EXISTS idx_experiments_thread ON experiments(thread_id);

-- Policy violations audit log (Memory Integrity Spec v1.2, §5.3 / Amendment 4 Gap 1)
CREATE TABLE IF NOT EXISTS policy_violations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_id TEXT NOT NULL,
    severity TEXT NOT NULL,
    concept_id TEXT DEFAULT '',
    detail TEXT NOT NULL,
    caller_context TEXT DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_policy_violations_severity ON policy_violations(severity);
CREATE INDEX IF NOT EXISTS idx_policy_violations_rule ON policy_violations(rule_id);
CREATE INDEX IF NOT EXISTS idx_policy_violations_created ON policy_violations(created_at);

-- Episodes table — metadata-focused, PII-safe (§5.2.5, resolves C12)
-- Raw text has 30-day retention; metadata is permanent for audit trail.
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    extracted_concept_ids TEXT DEFAULT '[]',
    concept_changes TEXT DEFAULT '[]',
    intent_summary TEXT DEFAULT '',
    classification TEXT DEFAULT '',
    world_timestamp TEXT,
    created_at TEXT NOT NULL,
    raw_user_message TEXT,
    raw_assistant_response TEXT,
    raw_purged_at TEXT,
    temporal_filter_outcome TEXT DEFAULT '',
    UNIQUE(session_id, turn_number)
);

CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);
CREATE INDEX IF NOT EXISTS idx_episodes_created ON episodes(created_at);
CREATE INDEX IF NOT EXISTS idx_episodes_classification ON episodes(classification);

-- Classification log table [M52: router observability]
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
);
CREATE INDEX IF NOT EXISTS idx_classification_log_ts ON classification_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_classification_log_class ON classification_log(classification);

-- Metrics table (WS2: structured observability)
CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    metric TEXT NOT NULL,
    value REAL NOT NULL,
    labels TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_metrics_ts ON metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(metric, timestamp);

-- Resume Context: rolling session snapshots for cross-session continuity
-- Spec: RESUME_CONTEXT_SPEC.md v1.1
CREATE TABLE IF NOT EXISTS resume_snapshots (
    session_id TEXT PRIMARY KEY,
    captured_at TEXT NOT NULL,
    active_task TEXT,
    task_domain TEXT,
    pinned_concepts TEXT DEFAULT '[]',
    last_exchange_gist TEXT,
    turn_count INTEGER DEFAULT 0,
    learning_events INTEGER DEFAULT 0,
    tools_used TEXT DEFAULT '[]',
    checkpoint_summary TEXT DEFAULT '{}',
    expires_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_resume_snapshots_captured ON resume_snapshots(captured_at);

-- Tier 2 LLM contradiction cache (WS1: persist across server restarts)
CREATE TABLE IF NOT EXISTS tier2_cache (
    cache_key TEXT PRIMARY KEY,
    score REAL NOT NULL,
    method TEXT NOT NULL,
    reason TEXT DEFAULT '',
    provider TEXT DEFAULT '',
    contradiction_type TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tier2_cache_expires ON tier2_cache(expires_at);

-- Cognitive Domains (Layer 3 — canonical DDL, also in migration 003)
-- Defensive: CREATE IF NOT EXISTS makes this idempotent with migration 003.
-- CLI-FIX-SPEC §Fix-3: Single source of truth for domain schema.
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

CREATE TABLE IF NOT EXISTS domain_area_mapping (
    domain_id TEXT NOT NULL,
    knowledge_area TEXT NOT NULL,
    activation_weight REAL DEFAULT 0.3 CHECK(activation_weight >= 0.0 AND activation_weight <= 1.0),
    PRIMARY KEY (domain_id, knowledge_area),
    FOREIGN KEY (domain_id) REFERENCES cognitive_domains(domain_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_dam_area ON domain_area_mapping(knowledge_area);

-- Dynamic Knowledge Areas (KA-ARCH-001: emergent ontology)
-- KAs are living entities with lifecycle states, not static classification buckets.
-- Seeds from taxonomy.json, grows via novel KA creation, evolves via promote/merge/decay.
CREATE TABLE IF NOT EXISTS knowledge_areas (
    name TEXT PRIMARY KEY,                    -- lowercase canonical name
    status TEXT NOT NULL DEFAULT 'provisional'
        CHECK(status IN ('seed', 'provisional', 'established', 'mature', 'archived')),
    description TEXT,                          -- auto-generated from concept cluster
    concept_count INTEGER DEFAULT 0,           -- derived count (periodic recount, not per-write)
    first_seen TEXT DEFAULT (datetime('now')),  -- when first proposed
    last_seen TEXT DEFAULT (datetime('now')),   -- when last concept assigned
    parent_ka TEXT,                             -- for merge lineage tracking
    aliases TEXT DEFAULT '[]',                  -- JSON array of known aliases
    embedding BLOB,                            -- cached embedding vector (384-dim)
    confidence REAL DEFAULT 0.0,               -- promotion confidence score
    source TEXT DEFAULT 'novel',               -- how it was created: seed/novel/split/merge
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_ka_status ON knowledge_areas(status);
CREATE INDEX IF NOT EXISTS idx_ka_concept_count ON knowledge_areas(concept_count DESC);

-- Behavioral Directives (Layer 3 — canonical DDL, also in migration 003)
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

-- CONTRA-001: Durable contradiction resolution outcomes
CREATE TABLE IF NOT EXISTS contradiction_resolutions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    concept_a_id TEXT NOT NULL,
    concept_b_id TEXT NOT NULL,
    contradiction_type TEXT NOT NULL,
    detection_phase INTEGER NOT NULL,
    similarity_score REAL,
    action TEXT NOT NULL,
    winner_id TEXT,
    loser_id TEXT,
    reason TEXT,
    source TEXT DEFAULT 'retrieval',
    session_id TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_contra_res_concepts ON contradiction_resolutions(concept_a_id, concept_b_id);
CREATE INDEX IF NOT EXISTS idx_contra_res_action ON contradiction_resolutions(action);
CREATE INDEX IF NOT EXISTS idx_contra_res_created ON contradiction_resolutions(created_at);

-- INGEST-037: Verbatim evidence fragment storage
-- Preserves raw text alongside semantic concepts for citation, code, and document provenance.
-- FK on concept_id is NOT enforced (no PRAGMA foreign_keys) — lifecycle managed manually
-- to avoid CASCADE conflicts with nonlossy version cleanup (AF1 resolution).
CREATE TABLE IF NOT EXISTS verbatim_fragments (
    id              TEXT PRIMARY KEY,
    concept_id      TEXT NOT NULL,
    concept_version TEXT,
    evidence_id     TEXT,
    fragment_type   TEXT NOT NULL DEFAULT 'text',
    content         TEXT,
    pointer_uri     TEXT,
    pointer_meta    TEXT,
    char_count      INTEGER DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    source_hash     TEXT,
    inherited_from  TEXT
);
CREATE INDEX IF NOT EXISTS idx_vf_concept ON verbatim_fragments(concept_id);
CREATE INDEX IF NOT EXISTS idx_vf_type ON verbatim_fragments(fragment_type);
CREATE INDEX IF NOT EXISTS idx_vf_hash ON verbatim_fragments(source_hash);
CREATE INDEX IF NOT EXISTS idx_vf_version ON verbatim_fragments(concept_id, concept_version);
"""

# Configure logging
logger = logging.getLogger(__name__)

# Configuration — profile-aware data directory and DB path
from app.profile import resolve_data_dir, resolve_db_path

# TOOLING-039: Warn if PITH_PROFILE is unset — standalone scripts may connect to wrong DB
if not os.environ.get("PITH_PROFILE") and not os.environ.get("PITH_DATA_DIR"):
    logger.warning("PITH_PROFILE not set — using default profile. Set PITH_PROFILE=<name> for standalone scripts.")

DATA_DIR = resolve_data_dir()
DB_PATH = resolve_db_path(DATA_DIR)
INDEX_DIR = DATA_DIR / "index"

# Kept for backward compat — consumers import these
CONCEPTS_DIR = DATA_DIR / "concepts"

# Ensure directories exist (INDEX_DIR used by TF-IDF, still filesystem)
INDEX_DIR.mkdir(parents=True, exist_ok=True)


def _migrate_db_filename() -> None:
    """Auto-migrate brain.db → pith.db on first startup after update.

    WAL safety protocol (per gauntlet amendment A1):
    1. Check if brain.db exists and pith.db does NOT
    2. Checkpoint WAL to flush pending writes
    3. Rename all 3 files: brain.db, brain.db-wal, brain.db-shm
    4. Verify new pith.db opens successfully
    5. If any step fails, revert all renames and log error
    """
    old_db = DATA_DIR / "brain.db"
    new_db = DATA_DIR / "pith.db"

    if new_db.exists():
        # Already migrated. Clean up ghost brain.db if it's empty (0 bytes).
        if old_db.exists() and old_db.stat().st_size == 0:
            old_db.unlink()
            logger.info("Removed ghost 0-byte brain.db (migration already complete)")
        return

    if not old_db.exists():
        return  # No database to migrate

    # Pre-check: data directory must be writable
    if not os.access(DATA_DIR, os.W_OK):
        logger.warning("Data directory not writable — skipping DB migration")
        return

    logger.info("Migrating database: brain.db → pith.db")

    # Log WAL size for diagnostics
    wal_path = DATA_DIR / "brain.db-wal"
    if wal_path.exists():
        logger.info("WAL size: %d bytes", wal_path.stat().st_size)

    # Step 1: Checkpoint WAL to flush all pending writes
    try:
        conn = sqlite3.connect(str(old_db))
        result = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        conn.close()
        # result = (busy, log, checkpointed)
        if result and result[0] != 0:
            logger.error(
                "Database locked by another process (busy=%d) — migration deferred. "
                "Stop all Pith instances before updating.",
                result[0],
            )
            return
        if result and result[1] != result[2]:
            logger.warning(
                "WAL not fully checkpointed (log=%d, checkpointed=%d) — another process may have the DB open",
                result[1],
                result[2],
            )
        logger.info("WAL checkpoint complete")
    except Exception as e:
        logger.error("WAL checkpoint failed, aborting migration: %s", e)
        return

    # Step 2: Rename all files (db, wal, shm)
    sidecar_exts = ["", "-wal", "-shm"]
    renamed = []
    try:
        for ext in sidecar_exts:
            old_path = DATA_DIR / f"brain.db{ext}"
            new_path = DATA_DIR / f"pith.db{ext}"
            if old_path.exists():
                old_path.rename(new_path)
                renamed.append((new_path, old_path))  # (current, revert-to)
                logger.info("Renamed: %s → %s", old_path.name, new_path.name)
    except Exception as e:
        logger.error("Rename failed at %s, reverting: %s", ext, e)
        # Revert all successful renames
        for current, original in reversed(renamed):
            try:
                current.rename(original)
                logger.info("Reverted: %s → %s", current.name, original.name)
            except Exception as re_err:
                logger.critical("REVERT FAILED for %s: %s — manual intervention required", current, re_err)
        return

    # Step 3: Verify new database opens
    try:
        conn = sqlite3.connect(str(new_db))
        conn.execute("SELECT count(*) FROM concepts")
        conn.close()
        logger.info("Migration verified: pith.db opens successfully")
    except Exception as e:
        logger.error("Verification failed, reverting: %s", e)
        for current, original in reversed(renamed):
            with suppress(Exception):
                current.rename(original)
        return

    logger.info("Database migration complete: brain.db → pith.db")


# Run migration at module load (before any DB access)
_migrate_db_filename()


# --- Database Connection ---

# --- Phase 4.5: Storage Backend Shims ---
# All connection/transaction management delegates to StorageBackend.
# These shims maintain backward compatibility for 41+ callers across
# the codebase — zero call-site changes required.
#
# The real logic lives in app/storage_backend.py (SQLiteBackend).
# These functions are thin pass-throughs.


def _get_connection() -> sqlite3.Connection:
    """Get persistent connection via storage backend.

    Phase 4.5 shim — delegates to get_backend().get_connection().
    All initialization (pragmas, DDL, migrations) handled by backend.
    """
    from app.storage_backend import get_backend

    return get_backend().get_connection()


def get_db_connection() -> sqlite3.Connection:
    """Public wrapper for embedding engine and other direct callers."""
    return _get_connection()


@contextmanager
def _db():
    """Transaction context manager — delegates to backend.db()."""
    from app.storage_backend import get_backend

    with get_backend().db() as conn:
        yield conn


@contextmanager
def _db_immediate():
    """Serialized write transaction — delegates to backend.db_immediate()."""
    from app.storage_backend import get_backend

    with get_backend().db_immediate() as conn:
        yield conn


def db_immediate():
    """Public access to BEGIN IMMEDIATE transaction context manager."""
    return _db_immediate()


# Module-level compat: keep access_tracker as a no-op shim.
# Direct DB writes happen inside load_concept() now.


class _AccessTrackerShim:
    """Compatibility shim — access tracking is now direct DB writes."""

    def record_access(self, concept_id: str) -> None:
        pass  # No-op: access tracked directly in load_concept

    def flush(self) -> int:
        return 0  # No pending writes — all immediate

    @property
    def pending_count(self) -> int:
        return 0


access_tracker = _AccessTrackerShim()


# --- Concept CRUD ---

# KA-005: Sentinel values that indicate "not meaningfully classified"
_KA_SENTINELS = {None, "", "general", "unclassified", "unknown"}


def _resolve_knowledge_area(concept, meta: dict) -> str:
    """Resolve knowledge_area with sentinel-awareness + taxonomy normalization.

    KA-005 fix: The old or-chain `concept.knowledge_area or meta.get(...) or 'general'`
    treated "general" as truthy, preventing metadata fallback from ever being reached.
    This helper treats sentinel values (None, '', 'general', 'unclassified', 'unknown')
    as "not classified" and falls through to metadata, which often contains the correct KA
    from reclassification or ingestion.

    KA-004: All resolved KAs are normalized through the taxonomy before returning.
    Uses permissive mode (strict=False) — novel KAs pass through with WARNING log.
    Wrapped in try-except so taxonomy failures never break concept writes.
    """
    # Priority 1: concept-level KA if it's a real classification
    concept_ka = getattr(concept, "knowledge_area", None)
    if concept_ka and concept_ka not in _KA_SENTINELS:
        resolved = concept_ka
    # Priority 2: metadata KA (often set by reclassification)
    elif (meta_ka := (meta.get("knowledge_area") if meta else None)) and meta_ka not in _KA_SENTINELS:
        resolved = meta_ka
    # Priority 3: nested metadata inside concept data
    elif hasattr(concept, "metadata") and isinstance(concept.metadata, dict):
        nested_ka = concept.metadata.get("knowledge_area")
        if nested_ka and nested_ka not in _KA_SENTINELS:
            resolved = nested_ka
        else:
            resolved = concept_ka if concept_ka else "general"
    else:
        # Fallback: preserve existing non-sentinel or default
        resolved = concept_ka if concept_ka else "general"

    # KA-004: Enforce taxonomy normalization at storage layer (single chokepoint)
    try:
        from app.taxonomy import normalize_knowledge_area

        normalized, source = normalize_knowledge_area(resolved, strict=False)
        if source == "novel":
            logger.warning("KA-004: Novel KA '%s' accepted at storage layer (no canonical match)", resolved)
        return normalized
    except Exception as e:
        logger.error("KA-004: Taxonomy normalization failed — writing raw KA '%s': %s", resolved, e)
        return resolved


def update_concept_data(
    conn, concept_id: str, data: dict, *, extra_sets: str = "", extra_params: tuple = (), require_current: bool = True
) -> int:
    """Gateway for writing concept JSON blobs with automatic column sync.

    KA-006: Any code that writes `SET data = ?` on the concepts table should
    use this function instead of raw SQL.  It ensures knowledge_area, confidence,
    maturity and other dual-tracked fields stay in sync between the SQL columns
    and the JSON blob, preventing the class of desync bugs that KA-005 fixed
    in save_concept.

    Args:
        conn: SQLite connection (caller manages transaction / commit).
        concept_id: The concept UUID to update.
        data: The full JSON-serialisable dict to write as the blob.
        extra_sets: Optional additional SET clause fragments (e.g. "currency_status = ?").
        extra_params: Parameters corresponding to extra_sets placeholders.

    Returns:
        Number of rows updated (0 or 1).
    """
    # --- KA sentinel-aware resolution (mirrors _resolve_knowledge_area logic) ---
    meta = data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {}
    blob_ka = data.get("knowledge_area")
    meta_ka = meta.get("knowledge_area")

    if blob_ka in _KA_SENTINELS and meta_ka and meta_ka not in _KA_SENTINELS:
        data["knowledge_area"] = meta_ka
        resolved_ka = meta_ka
    elif blob_ka and blob_ka not in _KA_SENTINELS:
        resolved_ka = blob_ka
    else:
        resolved_ka = blob_ka or "general"

    # --- Sync dual-tracked fields from blob → columns ---
    now = _utc_now_iso()
    set_parts = ["data = ?", "knowledge_area = ?", "updated_at = ?"]
    params: list = [json.dumps(data), resolved_ka, now]

    # DATA-020: Check if summary changed → set content_updated_at
    new_summary = data.get("summary")
    if new_summary is not None:
        old_row = conn.execute("SELECT summary FROM concepts WHERE id = ?", (concept_id,)).fetchone()
        if old_row and old_row[0] != new_summary:
            set_parts.append("content_updated_at = ?")
            params.append(now)

    confidence = data.get("confidence")
    if confidence is not None:
        set_parts.append("confidence = ?")
        params.append(round(float(confidence), 6))

    maturity = data.get("maturity")
    if maturity is not None:
        set_parts.append("maturity = ?")
        params.append(maturity)

    if extra_sets:
        set_parts.append(extra_sets)
        params.extend(extra_params)

    params.append(concept_id)
    set_clause = ", ".join(set_parts)
    where = "WHERE id = ? AND is_current = 1" if require_current else "WHERE id = ?"
    cursor = conn.execute(
        f"UPDATE concepts SET {set_clause} {where}",
        tuple(params),
    )
    return cursor.rowcount



def _sync_fts5(conn, concept_id: str, summary: str | None = None, delete: bool = False):
    """Sync a single concept to the FTS5 full-text index.

    Called by save_concept (upsert) and delete paths. Non-fatal on failure.
    RETRIEVAL-042 upgrade: keeps FTS5 index in sync with concepts table.
    INGEST-037-L4 Amendment A1: Self-reads fragment_keywords from concepts table
    so ALL callers automatically produce enriched FTS5 entries.
    """
    try:
        if delete:
            conn.execute("DELETE FROM fts_concepts WHERE concept_id = ?", (concept_id,))
        elif summary:
            # INGEST-037 L4: Self-read fragment keywords from DB
            enriched = summary
            try:
                fk_row = conn.execute(
                    "SELECT fragment_keywords FROM concepts WHERE id = ?",
                    (concept_id,),
                ).fetchone()
                fk = fk_row[0] if fk_row and fk_row[0] else None
                if fk:
                    enriched = f"{summary} [frag: {fk}]"
            except Exception:
                pass  # Column may not exist yet (pre-migration) — degrade gracefully

            # Upsert: delete old entry then insert new
            conn.execute("DELETE FROM fts_concepts WHERE concept_id = ?", (concept_id,))
            conn.execute(
                "INSERT INTO fts_concepts(concept_id, summary) VALUES (?, ?)",
                (concept_id, enriched),
            )
    except Exception as e:
        logger.warning(f"FTS5 sync failed for {concept_id}: {e}")


def save_concept(concept: Concept) -> None:
    """Save concept to SQLite. Writes to both concepts (latest) and concept_versions.

    Uses INSERT for new concepts (inherits column defaults like always_activate=0)
    and UPDATE for existing concepts (preserves column-level flags like always_activate).

    BUG FIX: Previously used INSERT OR REPLACE which deletes-then-inserts, wiping
    any columns not in the INSERT list (e.g., always_activate) back to DEFAULT.
    """
    data = concept.model_dump()
    meta = data.get("metadata", {})
    now = _utc_now_iso()

    # DEBT-185: Resolve KA once, sync to blob before serialization.
    # Previously the column got the resolved KA but json.dumps(data) kept the stale value.
    resolved_ka = _resolve_knowledge_area(concept, meta)
    data["knowledge_area"] = resolved_ka

    # GATE-BENCHMARKS: Skip benchmark concepts outside benchmark mode.
    # RETRIEVAL-061 excluded them at retrieval; this gates at ingestion.
    if resolved_ka == "pith_benchmarks":
        from app.config import BENCHMARK as _bm_gate
        if not _bm_gate.enabled:
            logger.info(
                "GATE-BENCHMARKS: Skipping benchmark concept %s (not in benchmark mode)",
                concept.id,
            )
            return  # Don't write — benchmark data excluded at ingestion

    # FIX-M3: Enforce M3 confidence ceiling at write time.
    # STABILITY-026/027 only guard ingest. evolve_concept legacy fallback bypasses them.
    # nonlossy.py:242 has its own M3 check; this is defense-in-depth for ALL write paths.
    from app.config import PSIS_QUARANTINE_CONFIDENCE_CAP, PSIS_QUARANTINE_EVIDENCE_MARKER
    _concept_evidence = data.get("evidence", [])
    if isinstance(_concept_evidence, list) and PSIS_QUARANTINE_EVIDENCE_MARKER in _concept_evidence:
        if concept.confidence > PSIS_QUARANTINE_CONFIDENCE_CAP:
            logger.info(
                "FIX-M3: Clamping PSIS concept %s confidence %.3f → %.1f at write time",
                concept.id, concept.confidence, PSIS_QUARANTINE_CONFIDENCE_CAP,
            )
            concept.confidence = PSIS_QUARANTINE_CONFIDENCE_CAP
            data["confidence"] = PSIS_QUARANTINE_CONFIDENCE_CAP

    with _db() as conn:
        # Check if concept already exists
        exists = conn.execute("SELECT 1 FROM concepts WHERE id = ?", (concept.id,)).fetchone()

        if exists:
            # UPDATE existing — preserves always_activate and other flag columns
            # DATA-020: Detect summary change → set content_updated_at
            old_summary_row = conn.execute("SELECT summary FROM concepts WHERE id = ?", (concept.id,)).fetchone()
            _content_updated_at_clause = ""
            _content_updated_at_params = []
            if old_summary_row and old_summary_row[0] != concept.summary:
                _content_updated_at_clause = ", content_updated_at = ?"
                _content_updated_at_params = [now]
            # FIX-1 (EVOLUTION_CHAIN_BREAK): Prevent in-memory model from overwriting
            # DB superseded_by back to NULL. If DB has a non-NULL superseded_by but
            # the in-memory model has None (loaded before supersession), preserve DB value.
            _superseded_by_val = getattr(concept, "superseded_by", None)
            if _superseded_by_val is None:
                _db_superseded = conn.execute(
                    "SELECT superseded_by FROM concepts WHERE id = ?", (concept.id,)
                ).fetchone()
                if _db_superseded and _db_superseded[0] is not None:
                    _superseded_by_val = _db_superseded[0]
                    logger.info(
                        "FIX-1: Preserving DB superseded_by=%s for %s (in-memory was None)",
                        _superseded_by_val,
                        concept.id,
                    )

            # AGENT-004: Include session_id only if concept has one
            # (don't overwrite existing session_id with NULL on evolution)
            session_id_val = getattr(concept, "session_id", None)
            if session_id_val:
                conn.execute(
                    """
                    UPDATE concepts SET
                        version = ?, summary = ?, confidence = ?, stability = ?,
                        knowledge_area = ?, concept_type = ?, status = ?,
                        salience = ?, salience_source = ?, maturity = ?,
                        updated_at = ?, last_accessed = ?, access_count = ?,
                        session_id = ?,
                        authority_score = ?, effective_authority = ?,
                        currency_score = ?, currency_status = ?,
                        superseded_by = ?, epistemic_network = ?,
                        reinforcement_count = ?,
                        original_date = ?,
                        data = ?
                    WHERE id = ?
                """,
                    (
                        concept.version,
                        concept.summary,
                        concept.confidence,
                        concept.stability,
                        resolved_ka,  # DEBT-185: use pre-resolved KA (synced to blob)
                        getattr(concept, "concept_type", "insight"),
                        concept.status,
                        getattr(concept, "salience", 0.5),
                        getattr(concept, "salience_source", "system"),
                        getattr(concept, "maturity", "ESTABLISHED"),
                        now,
                        getattr(concept, "last_accessed", None),
                        getattr(concept, "access_count", 0),
                        session_id_val,
                        _clamp_score(getattr(concept, "authority_score", None)),  # DEBT-182
                        _clamp_score(getattr(concept, "effective_authority", None)),  # DEBT-182
                        _clamp_score(getattr(concept, "currency_score", None)),  # DEBT-182
                        getattr(concept, "currency_status", None),
                        _superseded_by_val,  # FIX-1: Use guarded value
                        getattr(concept, "epistemic_network", None),
                        getattr(concept, "reinforcement_count", None),
                        getattr(concept, "original_date", None),  # TEMPORAL-002
                        json.dumps(data),
                        concept.id,
                    ),
                )
            else:
                conn.execute(
                    """
                    UPDATE concepts SET
                        version = ?, summary = ?, confidence = ?, stability = ?,
                        knowledge_area = ?, concept_type = ?, status = ?,
                        salience = ?, salience_source = ?, maturity = ?,
                        updated_at = ?, last_accessed = ?, access_count = ?,
                        authority_score = ?, effective_authority = ?,
                        currency_score = ?, currency_status = ?,
                        superseded_by = ?, epistemic_network = ?,
                        reinforcement_count = ?,
                        original_date = ?,
                        data = ?
                    WHERE id = ?
                """,
                    (
                        concept.version,
                        concept.summary,
                        concept.confidence,
                        concept.stability,
                        resolved_ka,  # DEBT-185: use pre-resolved KA (synced to blob)
                        getattr(concept, "concept_type", "insight"),
                        concept.status,
                        getattr(concept, "salience", 0.5),
                        getattr(concept, "salience_source", "system"),
                        getattr(concept, "maturity", "ESTABLISHED"),
                        now,
                        getattr(concept, "last_accessed", None),
                        getattr(concept, "access_count", 0),
                        _clamp_score(getattr(concept, "authority_score", None)),  # DEBT-182
                        _clamp_score(getattr(concept, "effective_authority", None)),  # DEBT-182
                        _clamp_score(getattr(concept, "currency_score", None)),  # DEBT-182
                        getattr(concept, "currency_status", None),
                        _superseded_by_val,  # FIX-1: Use guarded value
                        getattr(concept, "epistemic_network", None),
                        getattr(concept, "reinforcement_count", None),
                        getattr(concept, "original_date", None),  # TEMPORAL-002
                        json.dumps(data),
                        concept.id,
                    ),
                )
        else:
            # INSERT new — column defaults (always_activate=0) apply correctly
            _content_updated_at_params = [now]  # DATA-020: new concept = content is new
            validated_aid = validate_agent_id(meta.get("agent_id", "default"))
            conn.execute(
                """
                INSERT INTO concepts
                (id, version, summary, confidence, stability, knowledge_area,
                 concept_type, status, salience, salience_source, maturity,
                 created_at, updated_at, last_accessed, access_count, agent_id,
                 session_id, authority_score, effective_authority,
                 currency_score, currency_status, superseded_by,
                 epistemic_network, reinforcement_count, content_updated_at,
                 valid_from, original_date, data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    concept.id,
                    concept.version,
                    concept.summary,
                    concept.confidence,
                    concept.stability,
                    resolved_ka,  # DEBT-185: use pre-resolved KA (synced to blob)
                    getattr(concept, "concept_type", "insight"),
                    concept.status,
                    getattr(concept, "salience", 0.5),
                    getattr(concept, "salience_source", "system"),
                    getattr(concept, "maturity", "ESTABLISHED"),
                    concept.created_at,
                    now,
                    getattr(concept, "last_accessed", None),
                    getattr(concept, "access_count", 0),
                    validated_aid,  # 16: agent_id
                    getattr(concept, "session_id", None),  # 17: session_id
                    _clamp_score(getattr(concept, "authority_score", None)),  # 18: DEBT-137b + DEBT-182
                    _clamp_score(getattr(concept, "effective_authority", None)),  # 19: DEBT-137b + DEBT-182
                    _clamp_score(getattr(concept, "currency_score", None)),  # 20: DEBT-137b + DEBT-182
                    getattr(concept, "currency_status", None),  # 21: DEBT-137b
                    getattr(concept, "superseded_by", None),  # 22: DEBT-137b
                    getattr(concept, "epistemic_network", None),  # 23: DEBT-137b
                    getattr(concept, "reinforcement_count", None),  # 24: DEBT-137b
                    now,  # 25: DATA-020 content_updated_at
                    concept.created_at,  # 26: INGEST-016 valid_from = created_at
                    getattr(concept, "original_date", None),  # 27: TEMPORAL-002
                    json.dumps(data),  # 28: data (always last)
                ),
            )

        # DATA-020: Set content_updated_at when summary actually changed
        if _content_updated_at_params:
            conn.execute(
                "UPDATE concepts SET content_updated_at = ? WHERE id = ?",
                (now, concept.id),
            )

        # RETRIEVAL-042 upgrade: Sync FTS5 full-text index
        _sync_fts5(conn, concept.id, concept.summary)

        # Insert version record (append-only history)
        conn.execute(
            """
            INSERT OR IGNORE INTO concept_versions (id, version, data, created_at)
            VALUES (?, ?, ?, ?)
        """,
            (concept.id, concept.version, json.dumps(data), concept.created_at),
        )


# --- Conn-aware helpers for atomic evolve (§5.2.2) ---


def load_concept_conn(conn, concept_id: str) -> Concept | None:
    """Load latest concept using an existing connection (no separate transaction).

    Used inside _db_immediate() blocks for atomic read-modify-write cycles.
    Does NOT track access (internal operation).
    """
    row = conn.execute(
        """SELECT data, authority_score, currency_score, currency_status, knowledge_area, access_count, effective_authority, reinforcement_count, last_accessed, last_organic_access, ka_relative_authority, status, superseded_by, maturity, original_date, protected
           FROM concepts WHERE id = ?""",
        (concept_id,),
    ).fetchone()
    if not row:
        return None
    data = _safe_json_loads(row["data"], context=f"load_concept_conn({concept_id})")
    if data is None:
        return None
    try:
        data["authority_score"] = row["authority_score"]
        data["currency_score"] = row["currency_score"]
        data["currency_status"] = row["currency_status"] or "ACTIVE"
        data["access_count"] = row["access_count"] or 0
        data["effective_authority"] = row["effective_authority"]
        data["reinforcement_count"] = row["reinforcement_count"] or 0
        data["ka_relative_authority"] = row["ka_relative_authority"]
        # MATURITY-006: Inject maturity from DB column (canonical source).
        # Old concepts lack maturity in JSON blob, causing Pydantic to default to
        # ESTABLISHED — masking true PROVISIONAL state from the promotion sweep.
        data["maturity"] = row["maturity"] or "PROVISIONAL"
        # CURRENCY-001: Inject last_accessed from SQL to prevent desync.
        # Pre-RETRIEVAL-012, load_concept wrote last_accessed to SQL only (not JSON).
        if row["last_accessed"]:
            data["last_accessed"] = row["last_accessed"]
        # DATA-065: Inject last_organic_access from SQL column.
        if row["last_organic_access"]:
            data["last_organic_access"] = row["last_organic_access"]
        # DATA-018: Inject status from SQL column (not stored in JSON blob).
        data["status"] = row["status"] or "active"
        # TEMPORAL-002: Inject original_date from DB column
        if row["original_date"] is not None:
            data["original_date"] = row["original_date"]
        # MAINT-030: Hydrate superseded_by from DB column
        # Eliminates FIX-1 per-save DB reads during maintenance
        _sup_by = row["superseded_by"]
        if _sup_by is not None:
            data["superseded_by"] = _sup_by
        # COGGOV-005: Hydrate protected flag from DB column
        try:
            data["protected"] = bool(row["protected"]) if row["protected"] is not None else False
        except (IndexError, KeyError):
            pass
    except (IndexError, KeyError):
        pass
    if "knowledge_area" not in data or data["knowledge_area"] is None:
        meta = data.get("metadata", {})
        if meta.get("knowledge_area"):
            data["knowledge_area"] = meta["knowledge_area"]
        else:
            try:
                if row["knowledge_area"]:
                    data["knowledge_area"] = row["knowledge_area"]
            except (IndexError, KeyError):
                pass
    # FIX-2(A3): Inject defaults for missing required Pydantic fields
    _required_defaults = {
        "id": concept_id,
        "version": "v1",
        "created_at": _utc_now_iso(),
        "summary": "",
        "confidence": 0.5,
    }
    for _field, _default in _required_defaults.items():
        if _field not in data or data[_field] is None:
            data[_field] = _default
    try:
        return Concept(**data)
    except Exception as e:
        logger.error("load_concept_conn(%s) Pydantic error after defaults: %s", concept_id, e)
        return None


def get_next_version_conn(conn, concept_id: str) -> str:
    """Get next version number using an existing connection."""
    row = conn.execute(
        "SELECT version FROM concept_versions WHERE id = ? ORDER BY version DESC LIMIT 1", (concept_id,)
    ).fetchone()
    if not row:
        return "v1"
    try:
        num = int(row["version"][1:])
        return f"v{num + 1}"
    except (ValueError, IndexError):
        return "v1"


def save_concept_conn(conn, concept: "Concept") -> None:
    """Save concept using an existing connection (no separate transaction).

    Same logic as save_concept() but operates on a provided connection
    for use within _db_immediate() atomic blocks.
    """
    data = concept.model_dump()
    meta = data.get("metadata", {})
    now = _utc_now_iso()

    # DEBT-185: Resolve KA once, sync to blob before serialization.
    resolved_ka = _resolve_knowledge_area(concept, meta)
    data["knowledge_area"] = resolved_ka

    exists = conn.execute("SELECT 1 FROM concepts WHERE id = ?", (concept.id,)).fetchone()

    if exists:
        # DATA-020: Detect summary change → set content_updated_at
        _old_summary_row = conn.execute("SELECT summary FROM concepts WHERE id = ?", (concept.id,)).fetchone()
        _summary_changed = _old_summary_row and _old_summary_row[0] != concept.summary
        # AGENT-004: Include session_id only if concept has one
        # (don't overwrite existing session_id with NULL on evolution)
        session_id_val = getattr(concept, "session_id", None)
        if session_id_val:
            conn.execute(
                """
                UPDATE concepts SET
                    version = ?, summary = ?, confidence = ?, stability = ?,
                    knowledge_area = ?, concept_type = ?, status = ?,
                    salience = ?, salience_source = ?, maturity = ?,
                    updated_at = ?, last_accessed = ?, access_count = ?,
                    session_id = ?,
                    authority_score = ?, effective_authority = ?,
                    currency_score = ?, currency_status = ?,
                    superseded_by = ?, epistemic_network = ?,
                    reinforcement_count = ?,
                    original_date = ?,
                    data = ?
                WHERE id = ?
            """,
                (
                    concept.version,
                    concept.summary,
                    concept.confidence,
                    concept.stability,
                    resolved_ka,  # DEBT-185: use pre-resolved KA (synced to blob)
                    getattr(concept, "concept_type", "insight"),
                    concept.status,
                    getattr(concept, "salience", 0.5),
                    getattr(concept, "salience_source", "system"),
                    getattr(concept, "maturity", "ESTABLISHED"),
                    now,
                    getattr(concept, "last_accessed", None),
                    getattr(concept, "access_count", 0),
                    session_id_val,
                    _clamp_score(getattr(concept, "authority_score", None)),  # DEBT-187
                    _clamp_score(getattr(concept, "effective_authority", None)),  # DEBT-187
                    _clamp_score(getattr(concept, "currency_score", None)),  # DEBT-187
                    getattr(concept, "currency_status", None),
                    getattr(concept, "superseded_by", None),
                    getattr(concept, "epistemic_network", None),
                    getattr(concept, "reinforcement_count", None),
                    getattr(concept, "original_date", None),  # TEMPORAL-002
                    json.dumps(data),
                    concept.id,
                ),
            )
        else:
            conn.execute(
                """
                UPDATE concepts SET
                    version = ?, summary = ?, confidence = ?, stability = ?,
                    knowledge_area = ?, concept_type = ?, status = ?,
                    salience = ?, salience_source = ?, maturity = ?,
                    updated_at = ?, last_accessed = ?, access_count = ?,
                    authority_score = ?, effective_authority = ?,
                    currency_score = ?, currency_status = ?,
                    superseded_by = ?, epistemic_network = ?,
                    reinforcement_count = ?,
                    original_date = ?,
                    data = ?
                WHERE id = ?
            """,
                (
                    concept.version,
                    concept.summary,
                    concept.confidence,
                    concept.stability,
                    resolved_ka,  # DEBT-185: use pre-resolved KA (synced to blob)
                    getattr(concept, "concept_type", "insight"),
                    concept.status,
                    getattr(concept, "salience", 0.5),
                    getattr(concept, "salience_source", "system"),
                    getattr(concept, "maturity", "ESTABLISHED"),
                    now,
                    getattr(concept, "last_accessed", None),
                    getattr(concept, "access_count", 0),
                    _clamp_score(getattr(concept, "authority_score", None)),  # DEBT-187
                    _clamp_score(getattr(concept, "effective_authority", None)),  # DEBT-187
                    _clamp_score(getattr(concept, "currency_score", None)),  # DEBT-187
                    getattr(concept, "currency_status", None),
                    getattr(concept, "superseded_by", None),
                    getattr(concept, "epistemic_network", None),
                    getattr(concept, "reinforcement_count", None),
                    getattr(concept, "original_date", None),  # TEMPORAL-002
                    json.dumps(data),
                    concept.id,
                ),
            )
    else:
        validated_aid = validate_agent_id(meta.get("agent_id", "default"))
        conn.execute(
            """
            INSERT INTO concepts
            (id, version, summary, confidence, stability, knowledge_area,
             concept_type, status, salience, salience_source, maturity,
             created_at, updated_at, last_accessed, access_count, agent_id,
             session_id, authority_score, effective_authority,
             currency_score, currency_status, superseded_by,
             epistemic_network, reinforcement_count, original_date, data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                concept.id,
                concept.version,
                concept.summary,
                concept.confidence,
                concept.stability,
                resolved_ka,  # DEBT-185: use pre-resolved KA (synced to blob)
                getattr(concept, "concept_type", "insight"),
                concept.status,
                getattr(concept, "salience", 0.5),
                getattr(concept, "salience_source", "system"),
                getattr(concept, "maturity", "ESTABLISHED"),
                concept.created_at,
                now,
                getattr(concept, "last_accessed", None),
                getattr(concept, "access_count", 0),
                validated_aid,  # 16: agent_id
                getattr(concept, "session_id", None),  # 17: session_id
                _clamp_score(getattr(concept, "authority_score", None)),  # 18: DEBT-137b + DEBT-187
                _clamp_score(getattr(concept, "effective_authority", None)),  # 19: DEBT-137b + DEBT-187
                _clamp_score(getattr(concept, "currency_score", None)),  # 20: DEBT-137b + DEBT-187
                getattr(concept, "currency_status", None),  # 21: DEBT-137b
                getattr(concept, "superseded_by", None),  # 22: DEBT-137b
                getattr(concept, "epistemic_network", None),  # 23: DEBT-137b
                getattr(concept, "reinforcement_count", None),  # 24: DEBT-137b
                getattr(concept, "original_date", None),  # 25: TEMPORAL-002
                json.dumps(data),  # 26: data (always last)
            ),
        )

    # DATA-020: Set content_updated_at when summary actually changed
    if exists and _summary_changed:
        conn.execute(
            "UPDATE concepts SET content_updated_at = ? WHERE id = ?",
            (now, concept.id),
        )
    elif not exists:
        # New concept: content_updated_at = now
        conn.execute(
            "UPDATE concepts SET content_updated_at = ? WHERE id = ?",
            (now, concept.id),
        )

    # RETRIEVAL-042 upgrade: Sync FTS5 full-text index
    _sync_fts5(conn, concept.id, concept.summary)

    conn.execute(
        """
        INSERT OR IGNORE INTO concept_versions (id, version, data, created_at)
        VALUES (?, ?, ?, ?)
    """,
        (concept.id, concept.version, json.dumps(data), concept.created_at),
    )


def load_concept(concept_id: str, version: str = "latest", track_access: bool = True) -> Concept | None:
    """Load concept from SQLite.

    Args:
        concept_id: The concept identifier.
        version: 'latest' for current, 'all' for all versions, or specific 'v1', 'v2'.
        track_access: If True, increments access_count and updates last_accessed.
            Internal scans (reflection, decay) pass False to avoid inflating metrics.
    """
    if version == "all":
        return load_all_versions(concept_id)

    with _db() as conn:
        if version == "latest":
            row = conn.execute(
                """SELECT data, authority_score, currency_score, currency_status, knowledge_area, access_count, effective_authority, reinforcement_count, last_accessed, last_organic_access, ka_relative_authority, status, superseded_by, maturity, protected
                   FROM concepts WHERE id = ?""",
                (concept_id,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT data FROM concept_versions WHERE id = ? AND version = ?", (concept_id, version)
            ).fetchone()

        if not row:
            return None

        # Direct DB access tracking (replaces AccessTracker batching)
        # IMPORTANT: This is the SOLE organic access write path.
        # last_organic_access is updated ONLY here — NOT in save_concept().
        # If adding a new caller with track_access=True, ensure it represents
        # genuine user-initiated retrieval, not batch/maintenance operations.
        if track_access:
            # RETRIEVAL-012 + CURRENCY-003: Update last_accessed ONLY on retrieval
            # activation (pith_conversation_turn), NOT on evolution (learning.py).
            # This is the sole path that refreshes last_accessed, keeping
            # access_recency_score meaningful for currency computation.
            # DATA-065: Also update last_organic_access for freshness discrimination.
            _now = _utc_now_iso()
            conn.execute(
                """
                UPDATE concepts SET access_count = access_count + 1,
                    reinforcement_count = reinforcement_count + 1,
                    last_accessed = ?,
                    last_organic_access = ?,
                    data = json_set(data,
                        '$.last_accessed', ?,
                        '$.last_organic_access', ?,
                        '$.access_count', access_count + 1,
                        '$.reinforcement_count', reinforcement_count + 1)
                WHERE id = ?
            """,
                (_now, _now, _now, _now, concept_id),
            )

        data = _safe_json_loads(row["data"], context=f"load_concept({concept_id}, {version})")
        if data is None:
            return None
        # Inject governance scores from DB columns (not stored in JSON blob)
        if version == "latest":
            try:
                data["authority_score"] = row["authority_score"]
                data["currency_score"] = row["currency_score"]
                data["currency_status"] = row["currency_status"] or "ACTIVE"
                data["access_count"] = row["access_count"] or 0
                data["effective_authority"] = row["effective_authority"]
                data["reinforcement_count"] = row["reinforcement_count"] or 0
                data["ka_relative_authority"] = row["ka_relative_authority"]
                # CURRENCY-001: Inject last_accessed from SQL to prevent desync.
                if row["last_accessed"]:
                    data["last_accessed"] = row["last_accessed"]
                # DATA-065: Inject last_organic_access from SQL column.
                if row["last_organic_access"]:
                    data["last_organic_access"] = row["last_organic_access"]
                # DATA-018: Inject status from SQL column (not stored in JSON blob).
                data["status"] = row["status"] or "active"
                # MATURITY-006: Inject maturity from DB column (canonical source).
                # Old concepts lack maturity in JSON blob, causing Pydantic to default to
                # ESTABLISHED — masking true PROVISIONAL state from the promotion sweep.
                data["maturity"] = row["maturity"] or "PROVISIONAL"
                # MAINT-030: Hydrate superseded_by from DB column
                # Eliminates FIX-1 per-save DB reads during maintenance
                _sup_by = row["superseded_by"]
                if _sup_by is not None:
                    data["superseded_by"] = _sup_by
                # COGGOV-005: Hydrate protected flag from DB column
                try:
                    data["protected"] = bool(row["protected"]) if row["protected"] is not None else False
                except (IndexError, KeyError):
                    pass
            except (IndexError, KeyError):
                pass  # Governance columns not yet migrated
        # Inject knowledge_area from DB column if not already in JSON blob
        if "knowledge_area" not in data or data["knowledge_area"] is None:
            # Try metadata dict first (where save_concept writes it from)
            meta = data.get("metadata", {})
            if meta.get("knowledge_area"):
                data["knowledge_area"] = meta["knowledge_area"]
            elif version == "latest":
                try:
                    if row["knowledge_area"]:
                        data["knowledge_area"] = row["knowledge_area"]
                except (IndexError, KeyError):
                    pass
        # FIX-2(A3): Inject defaults for missing required Pydantic fields
        # to prevent ValidationError crashes across all 40+ callers.
        _required_defaults = {
            "id": concept_id,
            "version": "v1",
            "created_at": _utc_now_iso(),
            "summary": "",
            "confidence": 0.5,
        }
        for _field, _default in _required_defaults.items():
            if _field not in data or data[_field] is None:
                data[_field] = _default
        try:
            return Concept(**data)
        except Exception as e:
            logger.error("load_concept(%s) Pydantic error after defaults: %s", concept_id, e)
            return None


def load_all_versions(concept_id: str) -> list[Concept]:
    """Load all versions of a concept, ordered by version."""
    with _db() as conn:
        rows = conn.execute("SELECT data FROM concept_versions WHERE id = ? ORDER BY version", (concept_id,)).fetchall()
    results = []
    for r in rows:
        d = _safe_json_loads(r["data"], context=f"load_concept_versions({concept_id})")
        if d is not None:
            results.append(Concept(**d))
    return results


def list_concepts() -> list[str]:
    """List all active, current concept IDs.

    CURRENCY-008: Added is_current=1 filter. Previously returned 5894 concepts
    including 3530 superseded versions (is_current=0) that polluted health score
    aggregation, recalibration, and all reflection operations.
    """
    with _db() as conn:
        rows = conn.execute("SELECT id FROM concepts WHERE status = 'active' AND is_current = 1 ORDER BY id").fetchall()
    return [r["id"] for r in rows]


def list_concepts_modified_since(cutoff_iso: str) -> list[str]:
    """List concept IDs modified or created since a cutoff timestamp.

    REFLECT-021: Used by _merge_duplicates() to narrow scan to recently-changed
    concepts instead of iterating the full population. Falls back to full scan
    if cutoff is None.

    Args:
        cutoff_iso: ISO timestamp string. Returns concepts where
            content_updated_at > cutoff OR created_at > cutoff.
    """
    with _db() as conn:
        rows = conn.execute(
            """SELECT id FROM concepts
               WHERE status = 'active' AND is_current = 1
               AND (content_updated_at > ? OR created_at > ?)
               ORDER BY id""",
            (cutoff_iso, cutoff_iso),
        ).fetchall()
    return [r["id"] for r in rows]


def list_concepts_full() -> list[Concept]:
    """List all active concepts as full Concept objects in a single query.

    Used by session_start and other operations needing full models.
    Does NOT increment access_count (bulk read, not individual access).
    """
    with _db() as conn:
        rows = conn.execute(
            """SELECT data, authority_score, currency_score, currency_status
               FROM concepts WHERE status = 'active' AND is_current = 1 ORDER BY id"""
        ).fetchall()
    concepts = []
    for r in rows:
        try:
            data = _safe_json_loads(r["data"], context="list_concepts_full")
            if data is None:
                continue
            # Inject governance scores from DB columns
            try:
                data["authority_score"] = r["authority_score"]
                data["currency_score"] = r["currency_score"]
                data["currency_status"] = r["currency_status"] or "ACTIVE"
            except (IndexError, KeyError):
                pass
            concepts.append(Concept(**data))
        except Exception as e:
            logger.warning(f"list_concepts_full: failed to parse concept: {e}")
    return concepts


def list_concepts_for_indexing() -> list[dict]:
    """Lightweight bulk query for index building — returns raw dicts, not Concept models.

    Skips Pydantic construction (4ms/concept overhead) by returning parsed JSON
    dicts. Used exclusively by build_index where full model validation is unnecessary.
    """
    with _db() as conn:
        rows = conn.execute("SELECT id, data FROM concepts WHERE status = 'active' ORDER BY id").fetchall()
    results = []
    for r in rows:
        try:
            data = _safe_json_loads(r["data"], context=f"list_concepts_for_indexing({r['id']})")
            if data is None:
                continue
            data["_id"] = r["id"]
            results.append(data)
        except Exception as e:
            logger.warning(f"list_concepts_for_indexing: failed to parse: {e}")
    return results


def get_next_version(concept_id: str) -> str:
    """Get next version number for a concept."""
    with _db() as conn:
        row = conn.execute(
            "SELECT version FROM concept_versions WHERE id = ? ORDER BY version DESC LIMIT 1", (concept_id,)
        ).fetchone()
    if not row:
        return "v1"
    # Extract number from "v3" -> 3, return "v4"
    try:
        num = int(row["version"][1:])
        return f"v{num + 1}"
    except (ValueError, IndexError):
        return "v1"


# --- Associations ---

import time as _time_mod

# PERF-016: Module-level association cache
_associations_cache: dict | None = None
_associations_cache_ts: float = 0.0
_ASSOCIATIONS_CACHE_TTL_S: float = 60.0  # 60-second TTL

# MONITOR-031/042: Cache hit/miss counters (module-level, reset on process restart)
_assoc_cache_hits: int = 0
_assoc_cache_misses: int = 0
_adjacency_cache_hits: int = 0
_adjacency_cache_misses: int = 0

# PERF-023: Adjacency graph cache (derived from association list, same TTL)
_adjacency_graph_cache: dict[str, dict[str, float]] | None = None
_adjacency_graph_cache_ts: float = 0.0


def _invalidate_associations_cache() -> None:
    """Invalidate the associations cache. Call after any write to associations table."""
    global _associations_cache, _associations_cache_ts, _adjacency_graph_cache, _adjacency_graph_cache_ts
    _associations_cache = None
    _associations_cache_ts = 0.0
    _adjacency_graph_cache = None  # PERF-023
    _adjacency_graph_cache_ts = 0.0  # PERF-023


def load_associations() -> dict:
    """Load association graph with module-level TTL cache (PERF-016).

    Returns dict with 'associations' list and 'metadata'.
    Cache is invalidated on writes via _invalidate_associations_cache()
    and expires after _ASSOCIATIONS_CACHE_TTL_S seconds.
    """
    global _associations_cache, _associations_cache_ts, _assoc_cache_hits, _assoc_cache_misses

    now = _time_mod.monotonic()
    if _associations_cache is not None and (now - _associations_cache_ts) < _ASSOCIATIONS_CACHE_TTL_S:
        _assoc_cache_hits += 1
        return _associations_cache
    _assoc_cache_misses += 1

    with _db() as conn:
        rows = conn.execute("SELECT source, target, relation, strength, created_at FROM associations").fetchall()

    edges = [
        {
            "source": r["source"],
            "target": r["target"],
            "relation": r["relation"],
            "strength": r["strength"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]

    result = {"associations": edges, "metadata": {"storage": "sqlite"}}
    _associations_cache = result
    _associations_cache_ts = now
    return result


def get_adjacency_graph() -> dict[str, dict[str, float]]:
    """Return adjacency dict mapping concept_id → {neighbor_id: strength, ...} (PERF-023).

    Builds from load_associations() and caches independently with the same
    60-second TTL. Invalidated alongside the association cache via
    _invalidate_associations_cache() on any write.

    Callers (get_related_concepts, _spread_activation) pay <0.01ms on cache hit
    instead of 37ms for a full DB scan + adjacency rebuild.

    ARCH-O02: Changed from list[str] to dict[str, float] to preserve per-edge
    strength. Callers iterating neighbors get dict keys (same IDs as before).
    Callers needing strength use graph[src][tgt].
    """
    global _adjacency_graph_cache, _adjacency_graph_cache_ts, _adjacency_cache_hits, _adjacency_cache_misses

    now = _time_mod.monotonic()
    if _adjacency_graph_cache is not None and (now - _adjacency_graph_cache_ts) < _ASSOCIATIONS_CACHE_TTL_S:
        _adjacency_cache_hits += 1
        return _adjacency_graph_cache
    _adjacency_cache_misses += 1

    assoc_data = load_associations()
    graph: dict[str, dict[str, float]] = {}
    for edge in assoc_data["associations"]:
        src, tgt = edge["source"], edge["target"]
        strength = edge.get("strength", 0.5)
        graph.setdefault(src, {})[tgt] = strength
        graph.setdefault(tgt, {})[src] = strength

    _adjacency_graph_cache = graph
    _adjacency_graph_cache_ts = now
    return graph


def count_associations() -> int:
    """Count total association edges. Internal utility — not exposed via API."""
    with _db() as conn:
        row = conn.execute("SELECT COUNT(*) as cnt FROM associations").fetchone()
    return row["cnt"] if row else 0


def count_orphan_concepts() -> int:
    """Count active concepts with no association edges. Internal — used in pith_stats()."""
    with _db() as conn:
        row = conn.execute("""
            SELECT COUNT(*) as cnt FROM concepts
            WHERE status = 'active'
            AND id NOT IN (
                SELECT source FROM associations
                UNION
                SELECT target FROM associations
            )
        """).fetchone()
    return row["cnt"] if row else 0


def add_association(concept_a: str, concept_b: str, relation: str, strength: float = 0.5) -> None:
    """Add a single association edge (idempotent).

    Direction is normalized: source < target alphabetically. This ensures
    that A→B and B→A produce the same row, preventing duplicate edges
    regardless of argument order.
    """
    source, target = sorted([concept_a, concept_b])
    with _db() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO associations (source, target, relation, strength, created_at) VALUES (?, ?, ?, ?, ?)",
            (source, target, relation, strength, _utc_now_iso()),
        )
    _invalidate_associations_cache()  # PERF-016


def get_all_association_triples() -> set:
    """Return all existing association edges as a set of (source, target, relation) tuples.

    Used by the auto-association pipeline for efficient O(1) duplicate checking
    before bulk edge insertion.
    """
    with _db() as conn:
        rows = conn.execute("SELECT source, target, relation FROM associations").fetchall()
    return {(r["source"], r["target"], r["relation"]) for r in rows}


def get_knowledge_area_map() -> dict:
    """Return a dict of concept_id → knowledge_area for all active concepts.

    Lightweight query for the auto-association pipeline's Tier 2 domain matching.
    """
    with _db() as conn:
        rows = conn.execute("SELECT id, knowledge_area FROM concepts WHERE status = 'active'").fetchall()
    return {r["id"]: r["knowledge_area"] for r in rows}


def get_related_concepts(concept_id: str, max_depth: int = 2) -> list[str]:
    """Get concepts related to a given concept via BFS edge traversal (PERF-023).

    Uses get_adjacency_graph() — cached, <0.01ms on hit vs 37ms DB scan.
    BFS replaces recursive DFS: cleaner, avoids stack pressure on large graphs.
    """
    graph = get_adjacency_graph()
    if concept_id not in graph:
        return []

    related: set[str] = set()
    frontier: set[str] = {concept_id}

    for _ in range(max_depth):
        next_frontier: set[str] = set()
        for cid in frontier:
            for neighbor in graph.get(cid, {}):  # ARCH-O02: dict default
                if neighbor not in related and neighbor != concept_id:
                    related.add(neighbor)
                    next_frontier.add(neighbor)
        frontier = next_frontier
        if not frontier:
            break

    return list(related)


# --- Archive / Restore (status-based soft delete) ---


def archive_concept(concept_id: str) -> bool:
    """Archive a concept by setting status to 'archived'.

    Amendment 6: Also cleans up orphaned edges pointing to/from the archived concept.
    """
    with _db() as conn:
        row = conn.execute("SELECT id FROM concepts WHERE id = ? AND status = 'active'", (concept_id,)).fetchone()
        if not row:
            logger.warning(f"Cannot archive {concept_id}: not found or already archived")
            return False

        conn.execute(
            # DATA-048: Set is_current=0 on archive to prevent supersession desync
            "UPDATE concepts SET status = 'archived', is_current = 0, updated_at = ? WHERE id = ?",
            (_utc_now_iso(), concept_id),
        )

        # RETRIEVAL-042 upgrade: Remove from FTS5 index on archive
        _sync_fts5(conn, concept_id, delete=True)

        # Amendment 6: Clean up orphaned edges
        orphaned = conn.execute(
            "DELETE FROM associations WHERE source = ? OR target = ?", (concept_id, concept_id)
        ).rowcount
        if orphaned > 0:
            logger.info(f"Removed {orphaned} orphaned edges for archived concept {concept_id}")
            _invalidate_associations_cache()  # PERF-016

        logger.info(f"Archived concept: {concept_id}")

        # SYSTEMIC_FIXES_SPEC v1.1 Fix 3: Notify retrieval index on archive.
        # Without this, archived concepts linger as ghost entries until next
        # full reflection rebuild. Uses existing remove_concept() method.
        try:
            from app.retrieval import retrieval_engine

            retrieval_engine.remove_concept(concept_id)
            logger.debug(f"Removed archived concept {concept_id} from retrieval index")
        except Exception as idx_err:
            logger.warning(f"Failed to remove {concept_id} from index (non-fatal): {idx_err}")

        return True


def restore_concept(concept_id: str) -> bool:
    """Restore a concept from archived back to active.

    DATA-028: _restore_concept_lock prevents concurrent restore races where two
    callers could both pass the archived-check before either commits the UPDATE.
    """
    with _restore_concept_lock:
        with _db() as conn:
            row = conn.execute(
                "SELECT id FROM concepts WHERE id = ? AND status = 'archived'", (concept_id,)
            ).fetchone()
            if not row:
                logger.warning(f"Cannot restore {concept_id}: not in archive")
                return False

            conn.execute(
                "UPDATE concepts SET status = 'active', updated_at = ? WHERE id = ?",
                (_utc_now_iso(), concept_id),
            )
            logger.info(f"Restored concept: {concept_id}")
            return True


def list_archived_concepts() -> list[str]:
    """List all archived concept IDs."""
    with _db() as conn:
        rows = conn.execute("SELECT id FROM concepts WHERE status = 'archived' ORDER BY id").fetchall()
    return [r["id"] for r in rows]


# --- SelfModel Persistence ---


def save_self_model(model_data: dict) -> Path:
    """Save SelfModel to SQLite. Returns a Path for compat (unused by callers)."""
    version = model_data.get("version", 1)
    now = _utc_now_iso()
    data_json = json.dumps(model_data)

    with _db() as conn:
        # Upsert current
        conn.execute(
            "INSERT OR REPLACE INTO self_model (id, version, data, updated_at) VALUES (?, ?, ?, ?)",
            ("current", version, data_json, now),
        )
        # Append to version history
        conn.execute(
            "INSERT OR IGNORE INTO self_model_versions (version, data, created_at) VALUES (?, ?, ?)",
            (version, data_json, model_data.get("generated_at", now)),
        )

    logger.info(f"SelfModel saved: v{version}")
    # Return a Path for backward compat (no callers use this)
    return DB_PATH


def load_self_model() -> dict | None:
    """Load current SelfModel from SQLite."""
    with _db() as conn:
        row = conn.execute("SELECT data FROM self_model WHERE id = 'current'").fetchone()
    if not row:
        return None
    return _safe_json_loads(row["data"], context="load_self_model")


# ──────────────────────────────────────────────────────
# Session persistence
# ──────────────────────────────────────────────────────


def save_session(
    session_id: str,
    started_at: str,
    status: str = "active",
    context_hint: str = "",
    learning_event_count: int = 0,
    agent_id: str = "default",
    model_id: str = "unknown",
) -> None:
    """Insert a new session row."""
    validated_aid = validate_agent_id(agent_id)
    with _db() as conn:
        conn.execute(
            """INSERT INTO sessions (id, started_at, status, context_hint,
               learning_event_count, agent_id, data, model_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                started_at,
                status,
                context_hint or "",
                learning_event_count,
                validated_aid,
                json.dumps({"session_id": session_id}),
                model_id or "unknown",
            ),
        )
    logger.info(f"Session saved: {session_id} status={status} agent_id={validated_aid}")


def update_session(session_id: str, **kwargs) -> bool:
    """Update specific session fields. Returns True if row was updated."""
    allowed = {
        "ended_at",
        "status",
        "learning_event_count",
        "context_hint",
        "last_learning_at",
        "concepts_created",
        "concepts_evolved",
        "data",
        "model_id",
        "last_heartbeat",
        "working_context_json",
        "last_previous_response",  # SESSION-009: dropout recovery
    }
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return False

    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [session_id]

    with _db() as conn:
        cursor = conn.execute(f"UPDATE sessions SET {set_clause} WHERE id = ?", values)
    return cursor.rowcount > 0


def load_session(session_id: str) -> dict | None:
    """Load a single session by ID."""
    with _db() as conn:
        row = conn.execute(
            "SELECT id, started_at, ended_at, status, learning_event_count, "
            "context_hint, last_learning_at, last_heartbeat, working_context_json "
            "FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
    if not row:
        return None
    return dict(row)


def list_sessions(status: str = None, limit: int = 50, since: str = None) -> list[dict]:
    """Query sessions with optional filters. Returns newest-first."""
    query = (
        "SELECT id, started_at, ended_at, status, learning_event_count, context_hint, last_learning_at FROM sessions"
    )
    conditions = []
    params = []

    if status:
        conditions.append("status = ?")
        params.append(status)
    if since:
        conditions.append("started_at >= ?")
        params.append(since)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY started_at DESC LIMIT ?"
    params.append(limit)

    with _db() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def count_sessions(since: str = None) -> int:
    """Count sessions, optionally since a cutoff datetime."""
    if since:
        query = "SELECT COUNT(*) FROM sessions WHERE started_at >= ?"
        params = (since,)
    else:
        query = "SELECT COUNT(*) FROM sessions"
        params = ()

    with _db() as conn:
        row = conn.execute(query, params).fetchone()
    return row[0] if row else 0


def get_pith_stats_aggregates() -> dict:
    """Aggregate pith stats (concept counts, avg confidence, KA breakdown, orphans).

    Called by pith_stats() MCP endpoint. Use this, not count_* individually,
    to avoid N+1 queries on the stats path.
    """
    with _db() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*) as total_concepts,
                COALESCE(AVG(confidence), 0.0) as avg_confidence,
                COALESCE(AVG(stability), 0.0) as avg_stability,
                COUNT(DISTINCT knowledge_area) as knowledge_areas
            FROM concepts
            WHERE status = 'active'
        """).fetchone()

        versions_row = conn.execute("SELECT COUNT(*) as total FROM concept_versions").fetchone()

        # DEBT-006: Per-knowledge-area breakdown
        ka_rows = conn.execute("""
            SELECT
                COALESCE(knowledge_area, 'unknown') as ka,
                COUNT(*) as count,
                ROUND(AVG(confidence), 4) as avg_conf,
                ROUND(AVG(stability), 4) as avg_stab
            FROM concepts
            WHERE status = 'active'
            GROUP BY knowledge_area
            ORDER BY count DESC
        """).fetchall()
        ka_breakdown = [
            {
                "knowledge_area": r["ka"],
                "count": r["count"],
                "avg_confidence": r["avg_conf"],
                "avg_stability": r["avg_stab"],
            }
            for r in ka_rows
        ]

        # DEBT-003: Orphan concept count
        orphan_row = conn.execute("""
            SELECT COUNT(*) as cnt FROM concepts c
            WHERE c.status = 'active'
            AND c.id NOT IN (SELECT source FROM associations)
            AND c.id NOT IN (SELECT target FROM associations)
        """).fetchone()

        # DEBT-007: Evidence provenance summary
        # HEALTH-005: json_valid guard prevents 500 on malformed data blobs
        evidence_row = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN json_valid(data)
                    AND json_array_length(json_extract(data, '$.evidence')) > 0
                    THEN 1 ELSE 0 END) as with_evidence,
                SUM(CASE WHEN NOT json_valid(data)
                    OR json_array_length(json_extract(data, '$.evidence')) = 0
                    OR json_extract(data, '$.evidence') IS NULL
                    THEN 1 ELSE 0 END) as without_evidence,
                ROUND(AVG(CASE WHEN json_valid(data)
                    THEN json_array_length(json_extract(data, '$.evidence'))
                    ELSE 0 END), 2) as avg_evidence_count
            FROM concepts
            WHERE status = 'active'
            AND json_valid(data) = 1
        """).fetchone()

        # HEALTH-005: data quality metrics for observability
        data_quality_row = conn.execute("""
            SELECT
                SUM(CASE WHEN last_accessed IS NULL OR last_accessed = ''
                    THEN 1 ELSE 0 END) as null_timestamps,
                SUM(CASE WHEN json_valid(data) = 0
                    THEN 1 ELSE 0 END) as bad_json
            FROM concepts WHERE status = 'active'
        """).fetchone()

        # MONITOR-003: reinforcement_count distribution
        rc_rows = conn.execute("""
            SELECT CASE
                WHEN reinforcement_count IS NULL OR reinforcement_count = 0 THEN '0'
                WHEN reinforcement_count BETWEEN 1 AND 3 THEN '1-3'
                WHEN reinforcement_count BETWEEN 4 AND 10 THEN '4-10'
                ELSE '10+'
            END as bucket, COUNT(*) as cnt
            FROM concepts WHERE status = 'active' GROUP BY bucket
        """).fetchall()
        rc_dist = {r["bucket"]: r["cnt"] for r in rc_rows}

        # MONITOR-003: access_count distribution
        ac_rows = conn.execute("""
            SELECT CASE
                WHEN access_count IS NULL OR access_count = 0 THEN '0'
                WHEN access_count BETWEEN 1 AND 5 THEN '1-5'
                WHEN access_count BETWEEN 6 AND 20 THEN '6-20'
                ELSE '20+'
            END as bucket, COUNT(*) as cnt
            FROM concepts WHERE status = 'active' GROUP BY bucket
        """).fetchall()
        ac_dist = {r["bucket"]: r["cnt"] for r in ac_rows}

        # MONITOR-004: last_accessed temporal distribution
        la_rows = conn.execute("""
            SELECT CASE
                WHEN last_accessed IS NULL THEN 'never'
                WHEN last_accessed > datetime('now', '-1 day') THEN 'last_24h'
                WHEN last_accessed > datetime('now', '-7 days') THEN 'last_7d'
                WHEN last_accessed > datetime('now', '-30 days') THEN 'last_30d'
                ELSE 'older'
            END as recency, COUNT(*) as cnt
            FROM concepts WHERE status = 'active' GROUP BY recency
        """).fetchall()
        la_dist = {r["recency"]: r["cnt"] for r in la_rows}

        # MONITOR-029: Governance sweep zero-count alert
        gov_24h_row = conn.execute("""
            SELECT
                SUM(CASE WHEN event_type = 'MATURITY_PROMOTED' THEN 1 ELSE 0 END) as promotions_24h,
                SUM(CASE WHEN event_type LIKE '%QUARANTINE%' THEN 1 ELSE 0 END) as quarantine_24h,
                SUM(CASE WHEN event_type LIKE '%BACKFILL%' THEN 1 ELSE 0 END) as backfill_24h,
                COUNT(*) as total_gov_events_24h
            FROM governance_events
            WHERE created_at > datetime('now', '-1 day')
        """).fetchone()
        _active_concepts = row["total_concepts"] or 0
        _total_gov_24h = gov_24h_row["total_gov_events_24h"] or 0
        _gov_sweep_alert = _active_concepts > 1000 and _total_gov_24h == 0

        # MEASURE-008: Experiment concept efficacy tracking
        exp_efficacy_row = conn.execute("""
            SELECT
                COUNT(*) as total,
                COALESCE(AVG(access_count), 0) as avg_access,
                COALESCE(AVG(confidence), 0) as avg_conf,
                SUM(CASE WHEN access_count > 0 THEN 1 ELSE 0 END) as retrieved
            FROM concepts
            WHERE status = 'active' AND json_valid(data) = 1
            AND json_extract(data, '$.evidence') LIKE '%experiment:%'
        """).fetchone()
        regular_access_row = conn.execute("""
            SELECT COALESCE(AVG(access_count), 0) as avg_access
            FROM concepts WHERE status = 'active'
            AND (NOT json_valid(data)
                 OR json_extract(data, '$.evidence') NOT LIKE '%experiment:%')
        """).fetchone()

        # MONITOR-036: score-range validation — flag out-of-[0,1] scores
        oor_row = conn.execute("""
            SELECT
                SUM(CASE WHEN authority_score IS NOT NULL
                    AND (authority_score < 0.0 OR authority_score > 1.0) THEN 1 ELSE 0 END) as auth_oor,
                SUM(CASE WHEN effective_authority IS NOT NULL
                    AND (effective_authority < 0.0 OR effective_authority > 1.0) THEN 1 ELSE 0 END) as eff_auth_oor,
                SUM(CASE WHEN confidence IS NOT NULL
                    AND (confidence < 0.0 OR confidence > 1.0) THEN 1 ELSE 0 END) as conf_oor
            FROM concepts WHERE status = 'active'
        """).fetchone()

        # MONITOR-038: always-activate concept count
        always_activate_row = conn.execute(
            "SELECT COUNT(*) as cnt FROM concepts WHERE always_activate = 1 AND status = 'active'"
        ).fetchone()

        # MONITOR-033: ka_relative_authority coverage
        ka_ra_row = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN ka_relative_authority IS NOT NULL THEN 1 ELSE 0 END) as with_ka_ra
            FROM concepts WHERE status = 'active'
        """).fetchone()

        # MONITOR-024: reflection_tracking table analytics
        rt_rows = conn.execute("""
            SELECT trigger_type,
                   COUNT(*) as cnt,
                   SUM(CASE WHEN reflection_quality = 'timeout' THEN 1 ELSE 0 END) as timeouts,
                   AVG(concepts_returned) as avg_returned,
                   AVG(prompts_sent) as avg_prompts
            FROM reflection_tracking
            GROUP BY trigger_type
        """).fetchall()
        rt_total_row = conn.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN reflection_quality = 'timeout' THEN 1 ELSE 0 END) as total_timeouts,
                   SUM(CASE WHEN reflection_quality = 'auto_closed' THEN 1 ELSE 0 END) as total_auto_closed,
                   AVG(concepts_returned) as avg_concepts_returned
            FROM reflection_tracking
        """).fetchone()
        _rt_by_trigger = {
            r["trigger_type"]: {
                "count": r["cnt"],
                "timeouts": r["timeouts"],
                "avg_concepts_returned": round(r["avg_returned"] or 0, 2),
                "avg_prompts_sent": round(r["avg_prompts"] or 0, 2),
            }
            for r in rt_rows
        }

        # MONITOR-032: zombie concepts (is_current=1, not archived, no associations)
        zombie_row = conn.execute("""
            SELECT COUNT(*) as cnt FROM concepts
            WHERE is_current = 1
              AND status != 'archived'
              AND id NOT IN (
                  SELECT source FROM associations
                  UNION
                  SELECT target FROM associations
              )
        """).fetchone()

        # MONITOR-011: Index consistency — active vs superseded concept counts
        index_consistency_row = conn.execute("""
            SELECT
                SUM(CASE WHEN is_current = 1 AND status = 'active' THEN 1 ELSE 0 END) as active_current,
                SUM(CASE WHEN is_current = 0 AND status = 'active' THEN 1 ELSE 0 END) as active_superseded,
                COUNT(*) as total_all
            FROM concepts
        """).fetchone()

        # MONITOR-013: FIX-1 effectiveness — concepts with is_current=1 AND superseded_by set
        # (resurrection guard bypass: if count > 0, FIX-1 missed a resurrection)
        fix1_zombie_row = conn.execute("""
            SELECT COUNT(*) as cnt FROM concepts
            WHERE is_current = 1
              AND superseded_by IS NOT NULL
              AND superseded_by NOT IN ('', '__orphaned_supersession__')
              AND status = 'active'
        """).fetchone()

        # MONITOR-049: CTX-007 compaction survival format monitoring
        # Liveness probe: sample one eligible concept, confirm formatter produces [CRITICAL-CONTEXT].
        from app.config import FEATURE_FLAGS as _ff
        _csf_enabled = _ff.get("COMPACTION_SURVIVAL_FORMAT", False)
        _csf_row = conn.execute("""
            SELECT COUNT(*) as eligible
            FROM concepts
            WHERE status = 'active'
              AND concept_type IN ('constraint', 'decision', 'principle')
        """).fetchone()
        _csf_probe_ok = False
        if _csf_enabled and (_csf_row["eligible"] or 0) > 0:
            _probe = conn.execute("""
                SELECT id, summary, concept_type FROM concepts
                WHERE status = 'active' AND concept_type IN ('constraint', 'decision', 'principle')
                LIMIT 1
            """).fetchone()
            if _probe:
                from app.session import SessionManager
                _formatted = SessionManager._format_for_compaction_survival(
                    _probe["id"], _probe["summary"], _probe["concept_type"]
                )
                _csf_probe_ok = "[CRITICAL-CONTEXT" in _formatted

        # ARGUS-S23-F1: Token budget monitoring for COMPACTION_SURVIVAL_FORMAT
        # Estimate token overhead: avg summary chars of eligible concepts × ~4 chars/token + tag overhead
        _csf_avg_chars = 0
        _csf_est_tokens = 0
        if _csf_enabled and (_csf_row["eligible"] or 0) > 0:
            _csf_len_row = conn.execute("""
                SELECT ROUND(AVG(LENGTH(COALESCE(summary, ''))), 0) as avg_chars
                FROM concepts
                WHERE status = 'active'
                  AND concept_type IN ('constraint', 'decision', 'principle')
            """).fetchone()
            _csf_avg_chars = int(_csf_len_row["avg_chars"] or 0)
            # ~4 chars/token + 20 tokens tag overhead per concept; cap at 10 (always-activate pool)
            _concepts_in_budget = min(10, _csf_row["eligible"] or 0)
            _csf_est_tokens = round((_csf_avg_chars / 4 + 20) * _concepts_in_budget)

        # MONITOR-035: currency health alert
        _curr_row = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN currency_status = 'CONTRADICTED' THEN 1 ELSE 0 END) as contradicted,
                AVG(currency_score) as mean_score
            FROM concepts
            WHERE is_current = 1 AND status != 'archived'
        """).fetchone()

        # MONITOR-069: Factual coverage metrics
        factual_row = conn.execute("""
            SELECT
                COUNT(*) as total_active,
                SUM(CASE WHEN json_valid(data) AND json_extract(data, '$.metadata.is_factual') = 1
                    THEN 1 ELSE 0 END) as factual_count,
                SUM(CASE WHEN valid_from IS NOT NULL AND valid_from != ''
                    THEN 1 ELSE 0 END) as has_valid_from
            FROM concepts
            WHERE status = 'active' AND is_current = 1
        """).fetchone()
        _factual_total = factual_row["total_active"] or 1
        _factual_count = factual_row["factual_count"] or 0
        _has_valid_from = factual_row["has_valid_from"] or 0
        _factual_rate = round(_factual_count / _factual_total * 100, 1)

        # ARGUS-S25-F2: experiment_generation task health monitoring
        # Apply MONITOR-049 liveness pattern: last run recency + status.
        # Defensive: async_task_runs may not exist if ensure_async_tables hasn't run yet.
        try:
            _exp_gen_row = conn.execute("""
                SELECT status, started_at, completed_at, duration_ms, items_processed
                FROM async_task_runs
                WHERE task_type = 'experiment_generation'
                ORDER BY started_at DESC
                LIMIT 1
            """).fetchone()
        except Exception:
            _exp_gen_row = None
        if _exp_gen_row:
            _exp_gen_age_row = conn.execute(
                "SELECT ROUND((julianday('now') - julianday(?)) * 24, 1) as hours_ago",
                (_exp_gen_row["started_at"],),
            ).fetchone()
            _exp_gen_hours = _exp_gen_age_row["hours_ago"] if _exp_gen_age_row else None
            _exp_gen_status = (
                "healthy"
                if (_exp_gen_row["status"] == "success" and _exp_gen_hours is not None and _exp_gen_hours <= 48.0)
                else "stale"
                if (_exp_gen_hours is not None and _exp_gen_hours > 48.0)
                else "degraded"
            )
            _exp_gen_info: dict = {
                "last_status": _exp_gen_row["status"],
                "last_run_hours_ago": _exp_gen_hours,
                "last_duration_ms": _exp_gen_row["duration_ms"],
                "items_processed": _exp_gen_row["items_processed"],
                "health": _exp_gen_status,
            }
        else:
            _exp_gen_info = {
                "health": "never_run",
                "last_status": None,
                "last_run_hours_ago": None,
                "last_duration_ms": None,
                "items_processed": None,
            }


        # MONITOR-034: Stuck PROVISIONAL concepts (evidence<1 OR access<5 AND reinforcement<8)
        _stuck_prov_row = conn.execute("""
            SELECT
                COUNT(*) as stuck,
                (SELECT COUNT(*) FROM concepts
                 WHERE status='active' AND maturity='PROVISIONAL') as total_prov
            FROM concepts
            WHERE status='active' AND maturity='PROVISIONAL'
            AND (
                COALESCE(json_array_length(json_extract(data,'$.evidence')), 0) < 1
                OR (access_count < 5 AND reinforcement_count < 8)
            )
        """).fetchone()
        _stuck_prov = _stuck_prov_row["stuck"] if _stuck_prov_row else 0
        _total_prov = _stuck_prov_row["total_prov"] if _stuck_prov_row else 0

        # MONITOR-018: Stale session buildup (no ended_at, started >24h ago)
        _stale_sess_row = conn.execute("""
            SELECT COUNT(*) as cnt FROM sessions
            WHERE ended_at IS NULL
            AND started_at < datetime('now', '-24 hours')
        """).fetchone()
        _stale_sessions = _stale_sess_row["cnt"] if _stale_sess_row else 0

        # MONITOR-009: Context pressure trend (sessions.pressure_score)
        _pressure_row = conn.execute(
            """
            SELECT
                ROUND(AVG(CASE WHEN started_at > datetime('now','-1 day')
                               THEN pressure_score END), 3) AS avg_24h,
                COUNT(CASE WHEN started_at > datetime('now','-1 day')
                            AND pressure_score > 0.7 THEN 1 END) AS high_pressure_24h,
                ROUND(AVG(CASE WHEN started_at BETWEEN datetime('now','-7 days')
                               AND datetime('now','-1 day')
                               THEN pressure_score END), 3) AS avg_prior_7d
            FROM sessions
            WHERE pressure_score IS NOT NULL
            """
        ).fetchone()
        _p_avg_24h = (
            float(_pressure_row["avg_24h"])
            if _pressure_row and _pressure_row["avg_24h"] is not None
            else None
        )
        _p_high_24h = int(_pressure_row["high_pressure_24h"] or 0) if _pressure_row else 0
        _p_avg_7d = (
            float(_pressure_row["avg_prior_7d"])
            if _pressure_row and _pressure_row["avg_prior_7d"] is not None
            else None
        )
        if _p_avg_24h is not None and _p_avg_7d is not None:
            _p_trend = "rising" if _p_avg_24h > _p_avg_7d + 0.05 else (
                "falling" if _p_avg_24h < _p_avg_7d - 0.05 else "stable"
            )
        else:
            _p_trend = "unknown"

        # MONITOR-056: Cross-KA guard activation rate (24h from metrics)
        _cross_ka_row = conn.execute("""
            SELECT COALESCE(SUM(value), 0) as total_24h
            FROM metrics
            WHERE metric = 'cross_ka_guard_activations'
            AND timestamp > datetime('now', '-1 day')
        """).fetchone()
        _cross_ka_24h = int(_cross_ka_row["total_24h"] or 0)

        # MONITOR-058: Episode count for health monitoring
        try:
            _ep_row = conn.execute("SELECT COUNT(*) as cnt FROM episodes").fetchone()
            _episode_count = _ep_row["cnt"] if _ep_row else 0
        except Exception:
            _episode_count = -1  # Table may not exist in all environments

        # MONITOR-044: PSIS M3 compliance — quarantined concepts over confidence cap
        _psis_overcap_row = conn.execute("""
            SELECT COUNT(*) as over_cap FROM concepts
            WHERE maturity = 'QUARANTINED' AND confidence > 0.4
        """).fetchone()
        _psis_overcap = _psis_overcap_row["over_cap"] if _psis_overcap_row else 0

        # MONITOR-051: Analogy suggestion rate — 24h count from metrics table
        _analogy_metric_row = conn.execute("""
            SELECT COALESCE(SUM(value), 0) as total_24h,
                   COUNT(*) as turns_24h
            FROM metrics
            WHERE metric = 'analogy_suggestions_count'
            AND timestamp > datetime('now', '-1 day')
        """).fetchone()
        _analogy_total_24h = int(_analogy_metric_row["total_24h"] or 0)
        _analogy_turns_24h = int(_analogy_metric_row["turns_24h"] or 0)

        # MONITOR-070: Decay distribution by is_factual split
        _decay_rows = conn.execute("""
            SELECT
                CASE WHEN json_valid(data) AND json_extract(data, '$.metadata.is_factual') = 1
                     THEN 'factual' ELSE 'non_factual' END as kind,
                COUNT(*) as cnt,
                ROUND(AVG(COALESCE(currency_score, 0.0)), 4) as avg_score,
                ROUND(MIN(COALESCE(currency_score, 0.0)), 4) as min_score,
                ROUND(MAX(COALESCE(currency_score, 0.0)), 4) as max_score
            FROM concepts
            WHERE status = 'active' AND is_current = 1 AND currency_score IS NOT NULL
            GROUP BY kind
        """).fetchall()
        _decay_dist = {
            r["kind"]: {
                "count": r["cnt"],
                "avg_currency_score": r["avg_score"],
                "min_currency_score": r["min_score"],
                "max_currency_score": r["max_score"],
            }
            for r in _decay_rows
        }

    # MONITOR-041: canary window elapsed check
    import datetime as _dt
    from app import config as _cfg
    _canary_start = _dt.date.fromisoformat(getattr(_cfg, "EVOLUTION_CANARY_START_DATE", "2026-03-13"))
    _canary_elapsed = (_dt.date.today() - _canary_start).days
    _canary_window_passed = _canary_elapsed >= _cfg.EVOLUTION_CANARY_DURATION_DAYS

    # MONITOR-035: derive currency health alert status
    _curr_total = _curr_row["total"] or 0
    _contradicted = _curr_row["contradicted"] or 0
    _mean_score = _curr_row["mean_score"] or 0.0
    if _curr_total == 0:
        _curr_alert = "UNKNOWN"
        _contradicted_pct = 0.0
    else:
        _contradicted_pct = round(_contradicted / _curr_total * 100, 2)
        if _contradicted_pct > 50.0 or _mean_score < 0.5:
            _curr_alert = "CRITICAL"
        elif _contradicted_pct > 35.0 or _mean_score < 0.7:
            _curr_alert = "DEGRADED"
        else:
            _curr_alert = "HEALTHY"

    # MONITOR-073: KA canonical drift — count active concepts with non-canonical knowledge_area
    # KA-006: Use get_canonical_areas() (seed+established+mature from DB) instead of the
    # hardcoded _CANONICAL_KA_DESCRIPTIONS dict, which only covers embedding-classified KAs
    # and was missing architecture_gaps, ip_protection, product_operations, pith_benchmarks.
    from app.taxonomy import get_canonical_areas as _get_canonical_areas
    _canonical_kas = _get_canonical_areas()
    _placeholders = ",".join("?" * len(_canonical_kas))
    with _db() as _ka_conn:
        _ka_drift_row = _ka_conn.execute(
            f"""
            SELECT COUNT(*) as cnt FROM concepts
            WHERE is_current = 1
              AND status = 'active'
              AND knowledge_area NOT IN ({_placeholders})
            """,
            list(_canonical_kas),
        ).fetchone()
    _ka_non_canonical_count = _ka_drift_row["cnt"] if _ka_drift_row else 0

    return {
        "total_concepts": row["total_concepts"],
        "avg_confidence": round(row["avg_confidence"], 4),
        "avg_stability": round(row["avg_stability"], 4),
        "knowledge_areas": row["knowledge_areas"],
        "total_versions": versions_row["total"],
        "ka_breakdown": ka_breakdown,
        "orphan_concepts": orphan_row["cnt"],
        "evidence_stats": {
            "with_evidence": evidence_row["with_evidence"],
            "without_evidence": evidence_row["without_evidence"],
            "avg_evidence_per_concept": evidence_row["avg_evidence_count"],
        },
        # HEALTH-005: surface data quality in stats
        "data_quality": {
            "null_timestamps": data_quality_row["null_timestamps"],
            "bad_json": data_quality_row["bad_json"],
        },
        # MONITOR-003: engagement distribution histograms
        "reinforcement_count_distribution": rc_dist,
        "access_count_distribution": ac_dist,
        # MONITOR-004: temporal access recency distribution
        "last_accessed_distribution": la_dist,
        # MONITOR-029: governance sweep health
        "governance_sweep_24h": {
            "promotions": gov_24h_row["promotions_24h"] or 0,
            "quarantine_events": gov_24h_row["quarantine_24h"] or 0,
            "backfill_events": gov_24h_row["backfill_24h"] or 0,
            "total_events": _total_gov_24h,
            "zero_count_alert": _gov_sweep_alert,
        },
        # MEASURE-008: experiment concept efficacy
        "experiment_efficacy": {
            "experiment_concepts": exp_efficacy_row["total"] or 0,
            "avg_access_experiment": round(exp_efficacy_row["avg_access"] or 0, 2),
            "avg_access_regular": round(regular_access_row["avg_access"] or 0, 2),
            "avg_confidence": round(exp_efficacy_row["avg_conf"] or 0, 4),
            "retrieved_at_least_once": exp_efficacy_row["retrieved"] or 0,
            "retrieval_rate": round((exp_efficacy_row["retrieved"] or 0) / max(exp_efficacy_row["total"] or 0, 1), 4),
        },
        # MONITOR-036: score-range validation
        "score_range_validation": {
            "authority_out_of_range": oor_row["auth_oor"] or 0,
            "effective_authority_out_of_range": oor_row["eff_auth_oor"] or 0,
            "confidence_out_of_range": oor_row["conf_oor"] or 0,
        },
        # MONITOR-038: always-activate concept count
        "always_activate_count": always_activate_row["cnt"] or 0,
        # MONITOR-033: ka_relative_authority coverage
        "ka_relative_authority_coverage": {
            "total_active": ka_ra_row["total"] or 0,
            "with_ka_ra": ka_ra_row["with_ka_ra"] or 0,
            "coverage_pct": round((ka_ra_row["with_ka_ra"] or 0) / max(ka_ra_row["total"] or 0, 1) * 100, 1),
        },
        # MONITOR-041: canary window elapsed
        "evolution_canary": {
            "mode": getattr(_cfg, "EVOLUTION_CANARY_MODE", True),
            "elapsed_days": _canary_elapsed,
            "window_days": _cfg.EVOLUTION_CANARY_DURATION_DAYS,
            "window_passed": _canary_window_passed,
        },
        # MONITOR-024: reflection cycle analytics
        "reflection_tracking": {
            "total_cycles": rt_total_row["total"] or 0,
            "total_timeouts": rt_total_row["total_timeouts"] or 0,
            "total_auto_closed": rt_total_row["total_auto_closed"] or 0,
            "avg_concepts_returned": round(rt_total_row["avg_concepts_returned"] or 0, 2),
            "timeout_rate": round(
                (rt_total_row["total_timeouts"] or 0) / max(rt_total_row["total"] or 0, 1), 4
            ),
            "by_trigger_type": _rt_by_trigger,
        },
        # MONITOR-032: zombie concept count
        "zombie_count": zombie_row["cnt"] if zombie_row else 0,
        # MONITOR-011: index consistency — active vs superseded breakdown
        # MONITOR-060: superseded_pct alert thresholds (warn>70%, critical>85%)
        "index_consistency": {
            "active_current": index_consistency_row["active_current"] or 0,
            "active_superseded": index_consistency_row["active_superseded"] or 0,
            "total_all": index_consistency_row["total_all"] or 0,
            "superseded_pct": round(
                (index_consistency_row["active_superseded"] or 0)
                / max(index_consistency_row["total_all"] or 0, 1)
                * 100,
                1,
            ),
            "alert_level": (
                "critical"
                if round(
                    (index_consistency_row["active_superseded"] or 0)
                    / max(index_consistency_row["total_all"] or 0, 1) * 100, 1
                ) > 85.0
                else "warn"
                if round(
                    (index_consistency_row["active_superseded"] or 0)
                    / max(index_consistency_row["total_all"] or 0, 1) * 100, 1
                ) > 70.0
                else "ok"
            ),
        },
        # MONITOR-013: FIX-1 effectiveness — resurrected zombie count (should be 0)
        "fix1_zombie_alert": {
            "resurrected_count": fix1_zombie_row["cnt"] if fix1_zombie_row else 0,
            "alert": (fix1_zombie_row["cnt"] or 0) > 0,
        },
        # MONITOR-035: currency health alert
        "currency_health": {
            "status": _curr_alert,
            "total_is_current": _curr_total,
            "contradicted_pct": _contradicted_pct,
            "mean_currency_score": round(_mean_score, 4),
            "thresholds": {
                "degraded_if_contradicted_pct_above": 35.0,
                "critical_if_contradicted_pct_above": 50.0,
                "degraded_if_mean_score_below": 0.7,
                "critical_if_mean_score_below": 0.5,
            },
        },        # MONITOR-069: Factual coverage metrics
        "factual_coverage": {
            "total_active": _factual_total,
            "factual_count": _factual_count,
            "factual_rate_pct": _factual_rate,
            "has_valid_from": _has_valid_from,
            "valid_from_coverage_pct": round(_has_valid_from / max(_factual_count, 1) * 100, 1),
            "status": "healthy" if 20 <= _factual_rate <= 40 else ("low" if _factual_rate < 20 else "high"),
        },

        # MONITOR-049: CTX-007 compaction survival format health
        # MONITOR-049 + ARGUS-S23-F1: CSF health + token budget metrics
        "compaction_survival": {
            "flag_enabled": _csf_enabled,
            "eligible_concepts": _csf_row["eligible"] or 0,
            "formatter_live": _csf_probe_ok,
            "avg_summary_chars": _csf_avg_chars,
            "estimated_tokens_overhead": _csf_est_tokens,
            "status": "disabled" if not _csf_enabled else (
                "healthy" if _csf_probe_ok else (
                    "no_eligible_concepts" if (_csf_row["eligible"] or 0) == 0
                    else "formatter_broken"
                )
            ),
        },
        # ARGUS-S25-F2: experiment_generation task health
        "experiment_generation": _exp_gen_info,
        # MONITOR-056: Cross-KA guard activation rate (24h)
        "cross_ka_guard_rate": {
            "activations_24h": _cross_ka_24h,
        },
        # MONITOR-034: Stuck PROVISIONAL monitoring
        "stuck_provisional": {
            "stuck_count": _stuck_prov,
            "total_provisional": _total_prov,
            "stuck_pct": round(_stuck_prov / max(_total_prov, 1) * 100, 1),
            "alert": _stuck_prov > 200,
        },
        # MONITOR-018: Stale session buildup
        "stale_sessions": {
            "count": _stale_sessions,
            "alert": _stale_sessions > 10,
        },
        # MONITOR-009: Context pressure trend
        "pressure_trend": {
            "avg_24h": _p_avg_24h,
            "high_pressure_sessions_24h": _p_high_24h,
            "avg_prior_7d": _p_avg_7d,
            "trend": _p_trend,
            "alert": (_p_avg_24h is not None and _p_avg_24h > 0.6) or _p_high_24h > 3,
        },
        # MONITOR-058: Episode count
        "episode_count": _episode_count,
        # MONITOR-031/042: Association and adjacency cache hit/miss counters
        "association_cache_stats": {
            "hits": _assoc_cache_hits,
            "misses": _assoc_cache_misses,
            "hit_rate_pct": round(
                _assoc_cache_hits / max(_assoc_cache_hits + _assoc_cache_misses, 1) * 100, 1
            ),
        },
        "adjacency_cache_stats": {
            "hits": _adjacency_cache_hits,
            "misses": _adjacency_cache_misses,
            "hit_rate_pct": round(
                _adjacency_cache_hits / max(_adjacency_cache_hits + _adjacency_cache_misses, 1) * 100, 1
            ),
        },
        # MONITOR-044: PSIS M3 compliance alert
        "psis_m3_compliance": {
            "quarantined_over_cap": _psis_overcap,
            "cap_threshold": 0.4,
            "alert": _psis_overcap > 0,
        },
        # MONITOR-051: Analogy suggestion rate (24h window from metrics table)
        "analogy_suggestion_rate": {
            "total_suggestions_24h": _analogy_total_24h,
            "turns_with_suggestions_24h": _analogy_turns_24h,
        },
        # MONITOR-070: Currency score decay distribution by is_factual
        "decay_distribution": _decay_dist,
        # MONITOR-073: KA canonical drift — active concepts with non-canonical knowledge_area
        "ka_canonical_drift": {
            "non_canonical_count": _ka_non_canonical_count,
            "alert": _ka_non_canonical_count > 0,
        },
    }


def get_memory_projection_data() -> dict:
    """Compute growth velocity, per-KA health, and capacity projection.

    HEALTH-002: Answers "what will be" vs pith_stats "what is".

    Returns dict with:
      - growth_velocity: concepts/day over recent windows (7d, 14d, 30d)
      - ka_velocity: per-KA growth/decay rates (last 7d vs previous 7d)
      - maturity_flow: maturity transitions
      - capacity_projection: at current rate, when do we hit capacity thresholds
      - retrieval_activity: conversation turns per day (last 14d)
    """
    with _db() as conn:
        c = conn.cursor()
        result = {}

        # Growth velocity: concepts created per day over windows
        for window_label, days in [("7d", 7), ("14d", 14), ("30d", 30)]:
            c.execute(
                "SELECT COUNT(*) FROM concepts WHERE created_at > datetime('now', ?)",
                (f"-{days} days",),
            )
            count = c.fetchone()[0]
            result.setdefault("growth_velocity", {})[window_label] = {
                "total_created": count,
                "per_day": round(count / days, 1),
            }

        # Per-KA velocity: compare last 7d vs previous 7d
        c.execute("""
            SELECT knowledge_area,
                   SUM(CASE WHEN created_at > datetime('now', '-7 days') THEN 1 ELSE 0 END) as recent,
                   SUM(CASE WHEN created_at BETWEEN datetime('now', '-14 days')
                       AND datetime('now', '-7 days') THEN 1 ELSE 0 END) as previous
            FROM concepts
            GROUP BY knowledge_area
            HAVING recent > 0 OR previous > 0
            ORDER BY recent DESC
        """)
        ka_velocity = []
        for row in c.fetchall():
            ka, recent, previous = row[0], row[1], row[2]
            delta = recent - previous
            direction = "growing" if delta > 0 else "shrinking" if delta < 0 else "stable"
            ka_velocity.append(
                {
                    "knowledge_area": ka,
                    "last_7d": recent,
                    "prev_7d": previous,
                    "delta": delta,
                    "direction": direction,
                }
            )
        result["ka_velocity"] = ka_velocity

        # Maturity distribution + recent changes
        c.execute("SELECT maturity, COUNT(*) FROM concepts GROUP BY maturity")
        result["maturity_distribution"] = {row[0]: row[1] for row in c.fetchall()}

        c.execute("""
            SELECT maturity, COUNT(*) FROM concepts
            WHERE updated_at > datetime('now', '-7 days')
              AND maturity IN ('ESTABLISHED', 'PROVISIONAL')
            GROUP BY maturity
        """)
        result["recent_maturity_changes"] = {row[0]: row[1] for row in c.fetchall()}

        # Capacity projection: linear extrapolation
        c.execute("SELECT COUNT(*) FROM concepts")
        total = c.fetchone()[0]
        daily_rate = result["growth_velocity"]["7d"]["per_day"]

        CAPACITY_THRESHOLDS = [5000, 10000, 25000, 50000]
        projections = {}
        for threshold in CAPACITY_THRESHOLDS:
            if total >= threshold:
                projections[str(threshold)] = "already_reached"
            elif daily_rate > 0:
                days_to_reach = round((threshold - total) / daily_rate, 0)
                from datetime import datetime, timedelta

                est_date = (datetime.now(UTC) + timedelta(days=days_to_reach)).strftime("%Y-%m-%d")
                projections[str(threshold)] = {
                    "days_from_now": int(days_to_reach),
                    "estimated_date": est_date,  # DEBT-097: computed here instead of deferred
                }
            else:
                projections[str(threshold)] = "no_growth"
        result["capacity_projection"] = projections

        # Retrieval pressure: conversation turns per day
        c.execute("""
            SELECT DATE(created_at) as d, COUNT(*) FROM governance_events
            WHERE event_type = 'conversation_turn_complete'
              AND created_at > datetime('now', '-14 days')
            GROUP BY d ORDER BY d
        """)
        result["retrieval_activity"] = [{"date": row[0], "turns": row[1]} for row in c.fetchall()]

        return result


def get_distribution_report() -> dict:
    """Compute distribution statistics for all 7 retrieval blend factors.

    MEASURE-005 diagnostic: tracks whether inputs are discriminative.
    Returns per-factor: histogram (10 buckets), mean, stddev, % in dominant bucket.
    """
    with _db() as conn:
        c = conn.cursor()
        report = {}

        def _column_dist(col_name: str, where: str = "") -> dict:
            where_clause = f"WHERE {where}" if where else ""
            # Use CAST to bucket into 0.1 ranges
            c.execute(f"""
                SELECT CAST({col_name} * 10 AS INTEGER) / 10.0 as bucket, COUNT(*)
                FROM concepts {where_clause}
                GROUP BY bucket ORDER BY bucket
            """)
            rows = c.fetchall()
            if not rows:
                return {"histogram": {}, "mean": 0, "stddev": 0, "dominant_bucket_pct": 0, "discriminative": False}

            histogram = {}
            values = []
            for val, cnt in rows:
                bucket_key = f"{val:.1f}" if val is not None else "NULL"
                histogram[bucket_key] = cnt
                if val is not None:
                    values.extend([float(val)] * cnt)

            total = sum(histogram.values())
            dominant_pct = max(histogram.values()) / total * 100 if total > 0 else 0
            mean = sum(values) / len(values) if values else 0
            variance = sum((v - mean) ** 2 for v in values) / len(values) if values else 0

            return {
                "histogram": histogram,
                "count": total,
                "mean": round(mean, 4),
                "stddev": round(math.sqrt(variance), 4),
                "dominant_bucket_pct": round(dominant_pct, 1),
                "discriminative": dominant_pct < 50,
            }

        report["confidence"] = _column_dist("confidence")
        report["stability"] = _column_dist("stability")
        report["authority_score"] = _column_dist("authority_score", where="authority_score IS NOT NULL")
        report["currency_score"] = _column_dist("currency_score", where="currency_score IS NOT NULL")

        # Summary
        discriminative_count = sum(1 for f in report.values() if isinstance(f, dict) and f.get("discriminative", False))
        report["summary"] = {
            "factors_measured": 4,
            "factors_discriminative": discriminative_count,
            "factors_collapsed": 4 - discriminative_count,
            "note": "context_boost and goal_boost are query-time; emb_score varies per query",
        }

        return report


def recover_interrupted_sessions() -> int:
    """Mark any status='active' sessions as 'interrupted'. Returns count fixed."""
    now = _utc_now_iso()
    with _db() as conn:
        cursor = conn.execute(
            "UPDATE sessions SET status = 'interrupted', ended_at = ? WHERE status = 'active'", (now,)
        )
    count = cursor.rowcount
    if count > 0:
        logger.warning(f"Recovered {count} interrupted session(s) from previous run")
    return count


def load_concepts_by_type(concept_types: list, limit: int = 20, min_confidence: float = 0.0) -> list:
    """Load active concepts filtered by concept_type.

    Used for ambient principle retrieval — surfaces principles, methods,
    and strategies regardless of keyword match, ordered by confidence desc.
    """
    if not concept_types:
        return []
    placeholders = ",".join("?" * len(concept_types))
    with _db() as conn:
        rows = conn.execute(
            f"""
            SELECT id, summary, confidence, concept_type, knowledge_area, data
            FROM concepts
            WHERE status = 'active'
              AND concept_type IN ({placeholders})
              AND confidence >= ?
            ORDER BY confidence DESC
            LIMIT ?
        """,
            (*concept_types, min_confidence, limit),
        ).fetchall()

    results = []
    for row in rows:
        results.append(
            {
                "concept_id": row["id"],
                "summary": row["summary"],
                "confidence": row["confidence"],
                "concept_type": row["concept_type"],
                "knowledge_area": row["knowledge_area"],
            }
        )
    return results


def load_recent_concepts_by_types(
    concept_types: list,
    since_iso: str = None,
    limit: int = 5,
    min_confidence: float = 0.40,
    order_by: str = "created_at DESC",
    require_active_currency: bool = False,
    exclude_quarantined: bool = False,
) -> list:
    """Load concepts filtered by concept_type, ordered by recency or confidence.

    Used for orientation enrichment — surfaces decisions, principles,
    and findings to include in resumption briefings.

    S7.1: since_iso is now optional. When None, queries across ALL time
    (for strategic context that transcends recency). order_by allows
    sorting by confidence DESC for importance-based retrieval.
    """
    if not concept_types:
        return []
    # Validate order_by to prevent SQL injection
    allowed_orders = {"created_at DESC", "confidence DESC", "created_at ASC"}
    if order_by not in allowed_orders:
        order_by = "created_at DESC"

    placeholders = ",".join("?" * len(concept_types))
    conditions = [
        "status = 'active'",
        f"concept_type IN ({placeholders})",
        "confidence >= ?",
    ]
    params = list(concept_types) + [min_confidence]

    if since_iso is not None:
        conditions.append("created_at >= ?")
        params.append(since_iso)
    if require_active_currency:
        conditions.append("currency_status = 'ACTIVE'")
    if exclude_quarantined:
        conditions.append("maturity != 'QUARANTINED'")

    params.append(limit)
    where_clause = " AND ".join(conditions)

    with _db() as conn:
        rows = conn.execute(
            f"""
            SELECT id, summary, confidence, concept_type, knowledge_area
            FROM concepts
            WHERE {where_clause}
            ORDER BY {order_by}
            LIMIT ?
        """,
            tuple(params),
        ).fetchall()

    results = []
    for row in rows:
        results.append(
            {
                "concept_id": row["id"],
                "summary": row["summary"],
                "confidence": row["confidence"],
                "concept_type": row["concept_type"],
                "knowledge_area": row["knowledge_area"],
            }
        )
    return results


def load_recent_concepts(
    since_iso: str, limit: int = 10, min_confidence: float = 0.35, exclude_stale: bool = False
) -> list:
    """Load recent concepts of ALL types, ordered by recency.

    Used for orientation: sources WHERE BEEN and WHERE NOW from knowledge
    layer instead of stale checkpoint data. Returns the most recently
    created concepts regardless of concept_type.

    CONCEPT_LIFECYCLE_SPEC L1: exclude_stale filters out concepts with
    non-ACTIVE currency_status (STALE, SUPERSEDED, etc.).
    """
    conditions = [
        "status = 'active'",
        "confidence >= ?",
        "created_at >= ?",
    ]
    params = [min_confidence, since_iso]

    if exclude_stale:
        conditions.append("currency_status = 'ACTIVE'")

    where_clause = " AND ".join(conditions)

    with _db() as conn:
        rows = conn.execute(
            f"""
            SELECT id, summary, confidence, concept_type, knowledge_area,
                   created_at, maturity, currency_status
            FROM concepts
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT ?
        """,
            tuple(params) + (limit,),
        ).fetchall()

    results = []
    for row in rows:
        results.append(
            {
                "concept_id": row["id"],
                "summary": row["summary"],
                "confidence": row["confidence"],
                "concept_type": row["concept_type"],
                "knowledge_area": row["knowledge_area"],
                "created_at": row["created_at"],
                "maturity": row["maturity"] if "maturity" in row.keys() else "ESTABLISHED",
                "currency_status": row["currency_status"] if "currency_status" in row.keys() else "ACTIVE",
            }
        )
    return results


def load_always_activate_concepts() -> list:
    """Load all concepts flagged as always_activate.

    These concepts are injected into EVERY conversation_turn response
    regardless of topic or search relevance. Used for operational constraints
    that must fire at tool-selection time (e.g., 'use Desktop Commander for host paths').

    P1-1: Always-Activate concept tags.
    GOVERNANCE: Capped at MAX_ALWAYS_ACTIVATE (config.py) to prevent budget creep.
    """
    from app.config import MAX_ALWAYS_ACTIVATE

    with _db() as conn:
        rows = conn.execute("""
            SELECT id, summary, confidence, concept_type, knowledge_area
            FROM concepts
            WHERE status = 'active'
              AND always_activate = 1
            ORDER BY confidence DESC
            LIMIT ?
        """, (MAX_ALWAYS_ACTIVATE,)).fetchall()

    return [
        {
            "concept_id": row["id"],
            "summary": row["summary"],
            "confidence": row["confidence"],
            "concept_type": row["concept_type"],
            "knowledge_area": row["knowledge_area"],
        }
        for row in rows
    ]


def set_always_activate(concept_id: str, value: bool) -> bool:
    """Set or unset always_activate flag on a concept.

    Returns True if concept was found and updated, False otherwise.
    Raises ValueError if enabling would exceed MAX_ALWAYS_ACTIVATE cap.

    GOVERNANCE: Write-side guard prevents AA budget creep.
    Without this, always_activate flags accumulate unbounded and consume
    contextual retrieval slots (each AA concept costs one slot from
    CONTEXT_BUDGET_MAIN on every conversation turn).
    """
    from app.config import MAX_ALWAYS_ACTIVATE

    with _db() as conn:
        if value:
            # Write-side cap enforcement: refuse to flag beyond MAX_ALWAYS_ACTIVATE
            current_count = conn.execute(
                "SELECT COUNT(*) FROM concepts WHERE always_activate = 1 AND status = 'active' AND id != ?",
                (concept_id,)
            ).fetchone()[0]
            if current_count >= MAX_ALWAYS_ACTIVATE:
                raise ValueError(
                    f"Cannot set always_activate: already at cap ({current_count}/{MAX_ALWAYS_ACTIVATE}). "
                    f"Unset an existing always-activate concept first."
                )
        cursor = conn.execute("UPDATE concepts SET always_activate = ? WHERE id = ?", (1 if value else 0, concept_id))
        return cursor.rowcount > 0


# --- Firmware (P0-5) ---


def load_firmware() -> list:
    """Load all firmware entries.

    Returns list of dicts with id, summary, category, firmware_version.
    Called by conversation_turn to inject static operational knowledge.
    """
    with _db() as conn:
        rows = conn.execute("""
            SELECT id, summary, category, firmware_version
            FROM firmware
            ORDER BY category, id
        """).fetchall()

    return [
        {
            "id": row["id"],
            "summary": row["summary"],
            "category": row["category"],
            "firmware_version": row["firmware_version"],
        }
        for row in rows
    ]


def save_firmware(firmware_id: str, summary: str, category: str, firmware_version: str) -> None:
    """Upsert a firmware entry. Called only by seed_firmware.py on server startup."""
    now = _utc_now_iso()
    with _db() as conn:
        conn.execute(
            """
            INSERT INTO firmware (id, summary, category, firmware_version, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                summary = excluded.summary,
                category = excluded.category,
                firmware_version = excluded.firmware_version,
                updated_at = excluded.updated_at
        """,
            (firmware_id, summary, category, firmware_version, now, now),
        )


def count_concepts_by_type_tier(since_iso: str = None) -> dict:
    """Count active concepts grouped by abstraction tier. Internal analytics.

    Returns: {
        'L1_observations': int,  (observation, pattern, goal, constraint)
        'L3_abstractions': int,  (principle, method, heuristic, cognitive_strategy)
        'L2_decisions': int,     (decision)
        'total': int,
        'ratio': float,          (L3 / total, 0.0 if total=0)
    }

    If since_iso is provided, only counts concepts created after that date.
    RETRO-001: Used to detect when retrospective is needed.
    """
    L1_TYPES = ("observation", "pattern", "goal", "constraint")
    L2_TYPES = ("decision",)
    L3_TYPES = ("principle", "method", "heuristic", "cognitive_strategy")

    where_clause = "WHERE status = 'active'"
    params = []
    if since_iso:
        where_clause += " AND created_at >= ?"
        params.append(since_iso)

    with _db() as conn:
        rows = conn.execute(
            f"""
            SELECT concept_type, COUNT(*) as cnt
            FROM concepts
            {where_clause}
            GROUP BY concept_type
        """,
            params,
        ).fetchall()

    counts = {row["concept_type"]: row["cnt"] for row in rows}

    l1 = sum(counts.get(t, 0) for t in L1_TYPES)
    l2 = sum(counts.get(t, 0) for t in L2_TYPES)
    l3 = sum(counts.get(t, 0) for t in L3_TYPES)
    total = l1 + l2 + l3

    return {
        "L1_observations": l1,
        "L2_decisions": l2,
        "L3_abstractions": l3,
        "total": total,
        "ratio": round(l3 / max(total, 1), 3),
    }


def get_metadata(key: str) -> str | None:
    """Get a metadata value by key."""
    with _db() as conn:
        row = conn.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_metadata(key: str, value: str) -> None:
    """Set a metadata value (upsert)."""
    with _db() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO metadata (key, value, updated_at)
            VALUES (?, ?, ?)
        """,
            (key, value, _utc_now_iso()),
        )


def load_session_velocity(cutoff_iso: str, prior_cutoff_iso: str = None) -> dict:
    """Load session performance data for cognitive velocity computation.

    Returns aggregate stats for sessions in the window (cutoff → now),
    and optionally for the prior window (prior_cutoff → cutoff) for trend comparison.
    """

    def _aggregate(since: str, until: str = None) -> dict:
        with _db() as conn:
            where = "WHERE status IN ('ended', 'active') AND started_at >= ?"
            params = [since]
            if until:
                where += " AND started_at < ?"
                params.append(until)
            row = conn.execute(
                f"""
                SELECT
                    COUNT(*) as session_count,
                    COALESCE(SUM(learning_event_count), 0) as total_learning_events,
                    COALESCE(SUM(concepts_created), 0) as total_concepts_created,
                    COALESCE(SUM(concepts_evolved), 0) as total_concepts_evolved
                FROM sessions {where}
            """,
                params,
            ).fetchone()
        return {
            "session_count": row["session_count"],
            "total_learning_events": row["total_learning_events"],
            "total_concepts_created": row["total_concepts_created"],
            "total_concepts_evolved": row["total_concepts_evolved"],
        }

    current = _aggregate(cutoff_iso)
    prior = _aggregate(prior_cutoff_iso, cutoff_iso) if prior_cutoff_iso else None
    return {"current": current, "prior": prior}


# --- Checkpoint CRUD ---
# Execution checkpoints: ephemeral resumption state, NOT knowledge concepts.
# Different lifecycle (TTL-based), no garbage detection, no embedding.

MAX_CHECKPOINTS = 50
DEFAULT_TTL_DAYS = 7
STALE_CHECKPOINT_HOURS = 48  # CKPT-001: Archive checkpoints with no update in this many hours
COMPLETED_TTL_DAYS = 1


def save_checkpoint(
    task_id: str,
    description: str,
    status: str = "active",
    done: list = None,
    active: str = "",
    next_items: list = None,
    blockers: list = None,
    context: dict = None,
    concept_refs: list = None,
    session_id: str = None,
    ttl_days: int = None,
) -> dict:
    """Upsert checkpoint by task_id. done[] is union-merged (append-only)."""
    now = _utc_now_iso()
    ttl = ttl_days or DEFAULT_TTL_DAYS
    expires_at = (_utc_now() + timedelta(days=ttl)).isoformat()

    with _db() as conn:
        existing = conn.execute(
            "SELECT done, save_count, created_at, status FROM checkpoints WHERE task_id = ?", (task_id,)
        ).fetchone()

        # CKPT-001: Lifecycle validation — reject saves to terminal states
        # Allow reopen if caller explicitly passes an active status
        if existing and existing["status"] in ("complete", "archived") and status not in (
            "complete",
            "archived",
            "active",
            "planning",
        ):
            logger.warning(
                f"CKPT-001: Rejected save to {task_id} — status '{existing['status']}' is terminal. "
                f"Pass explicit status='active' to reopen."
            )
            return load_checkpoint(task_id=task_id) or {
                "task_id": task_id,
                "status": existing["status"],
                "error": "terminal_state",
            }

        if existing:
            # Union-merge done[] (never remove items)
            old_done = json.loads(existing["done"]) if existing["done"] else []
            new_done = list(set(old_done + (done or [])))
            save_count = (existing["save_count"] or 0) + 1
            created_at = existing["created_at"]
        else:
            new_done = done or []
            save_count = 1
            created_at = now

        conn.execute(
            """
            INSERT OR REPLACE INTO checkpoints
            (task_id, session_id, status, description, done, active, next,
             blockers, context, concept_refs, created_at, updated_at, expires_at, save_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                task_id,
                session_id,
                status,
                description,
                json.dumps(new_done),
                active or "",
                json.dumps(next_items or []),
                json.dumps(blockers or []),
                json.dumps(context or {}),
                json.dumps(concept_refs or []),
                created_at,
                now,
                expires_at,
                save_count,
            ),
        )

        # FIFO eviction if over max
        count = conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
        if count > MAX_CHECKPOINTS:
            excess = count - MAX_CHECKPOINTS
            conn.execute(
                """
                DELETE FROM checkpoints WHERE task_id IN (
                    SELECT task_id FROM checkpoints
                    WHERE status NOT IN ('active', 'blocked')
                    ORDER BY updated_at ASC LIMIT ?
                )
            """,
                (excess,),
            )

    logger.info(f"Checkpoint saved: {task_id} status={status} save_count={save_count}")
    return {
        "task_id": task_id,
        "status": status,
        "description": description,
        "done": new_done,
        "active": active or "",
        "next": next_items or [],
        "blockers": blockers or [],
        "save_count": save_count,
        "created_at": created_at,
        "updated_at": now,
        "expires_at": expires_at,
    }


def load_checkpoint(task_id: str = None, max_age_hours: int = 24, session_id: str = None) -> dict | None:
    """Load checkpoint by task_id, or most recently updated non-completed."""
    cutoff = (_utc_now() - timedelta(hours=max_age_hours)).isoformat()

    with _db() as conn:
        if task_id:
            row = conn.execute(
                "SELECT * FROM checkpoints WHERE task_id = ? AND expires_at > ?", (task_id, _utc_now_iso())
            ).fetchone()
        elif session_id:
            # CONTEXT-001 Fix 9: Session-scoped checkpoint — find checkpoint created by this session
            row = conn.execute(
                """
                SELECT * FROM checkpoints
                WHERE status NOT IN ('complete', 'archived') AND expires_at > ? AND updated_at > ?
                ORDER BY updated_at DESC LIMIT 1
            """,
                (_utc_now_iso(), cutoff),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT * FROM checkpoints
                WHERE status NOT IN ('complete', 'archived') AND expires_at > ? AND updated_at > ?
                ORDER BY updated_at DESC LIMIT 1
            """,
                (_utc_now_iso(), cutoff),
            ).fetchone()

    if not row:
        return None

    return {
        "task_id": row["task_id"],
        "session_id": row["session_id"],
        "status": row["status"],
        "description": row["description"],
        "done": json.loads(row["done"]) if row["done"] else [],
        "active": row["active"] or "",
        "next": json.loads(row["next"]) if row["next"] else [],
        "blockers": json.loads(row["blockers"]) if row["blockers"] else [],
        "context": json.loads(row["context"]) if row["context"] else {},
        "concept_refs": json.loads(row["concept_refs"]) if row["concept_refs"] else [],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "expires_at": row["expires_at"],
        "save_count": row["save_count"],
    }


def list_checkpoints() -> list:
    """List all non-expired checkpoints, newest first.

    Returns full checkpoint data including next[], blockers[], active
    for orientation enrichment and anticipation (S6.3).
    """
    now = _utc_now_iso()
    with _db() as conn:
        rows = conn.execute(
            """
            SELECT task_id, status, description, active, next, blockers,
                   done, updated_at, save_count
            FROM checkpoints WHERE expires_at > ?
            ORDER BY updated_at DESC
        """,
            (now,),
        ).fetchall()
    results = []
    for r in rows:
        results.append(
            {
                "task_id": r["task_id"],
                "status": r["status"],
                "description": r["description"],
                "active": r["active"] or "",
                "next": json.loads(r["next"]) if r["next"] else [],
                "blockers": json.loads(r["blockers"]) if r["blockers"] else [],
                "done": json.loads(r["done"]) if r["done"] else [],
                "updated_at": r["updated_at"],
                "save_count": r["save_count"],
            }
        )
    return results


def complete_checkpoint(task_id: str) -> dict | None:
    """Mark checkpoint complete, set short TTL."""
    now = _utc_now_iso()
    short_ttl = (_utc_now() + timedelta(days=COMPLETED_TTL_DAYS)).isoformat()

    with _db() as conn:
        row = conn.execute("SELECT active, done FROM checkpoints WHERE task_id = ?", (task_id,)).fetchone()
        if not row:
            return None

        # Move active to done
        done = json.loads(row["done"]) if row["done"] else []
        if row["active"] and row["active"] not in done:
            done.append(row["active"])

        conn.execute(
            """
            UPDATE checkpoints SET status = 'complete', active = '',
                next = '[]', done = ?, updated_at = ?, expires_at = ?
            WHERE task_id = ?
        """,
            (json.dumps(done), now, short_ttl, task_id),
        )

    logger.info(f"Checkpoint completed: {task_id}")
    return load_checkpoint(task_id)


def touch_checkpoint(task_id: str, ttl_days: int = DEFAULT_TTL_DAYS) -> dict | None:
    """Extend TTL without changing content."""
    now = _utc_now_iso()
    new_expires = (_utc_now() + timedelta(days=ttl_days)).isoformat()

    with _db() as conn:
        cursor = conn.execute(
            """
            UPDATE checkpoints SET expires_at = ?, updated_at = ?
            WHERE task_id = ?
        """,
            (new_expires, now, task_id),
        )
        if cursor.rowcount == 0:
            return None

    return load_checkpoint(task_id)


def cleanup_expired_checkpoints() -> int:
    """Delete expired checkpoints regardless of status."""
    now = _utc_now_iso()
    with _db() as conn:
        # Log what we're about to delete for auditability
        zombies = conn.execute("SELECT task_id, status FROM checkpoints WHERE expires_at < ?", (now,)).fetchall()
        for z in zombies:
            logger.info(f"Cleaning expired checkpoint: {z[0]} (status={z[1]})")

        cursor = conn.execute("DELETE FROM checkpoints WHERE expires_at < ?", (now,))
    deleted = cursor.rowcount
    if deleted:
        logger.info(f"Cleaned up {deleted} expired checkpoint(s)")
    return deleted


def archive_stale_checkpoints(max_age_hours: int = STALE_CHECKPOINT_HOURS, exclude_session_id: str = None) -> int:
    """Archive checkpoints that haven't been updated in max_age_hours.

    CKPT-001: Stale checkpoints (no update in 48h) get archived rather than
    completed, because a stale checkpoint may represent paused work — not
    finished work. Archived checkpoints are excluded from load_checkpoint()
    and working_context but remain in DB for audit.

    NOTE: Auto-COMPLETE is a separate mechanism in staleness.py:587-611 with
    stricter guards (save_count>=2, non-empty done, empty next+active, >1h old).
    This function only ARCHIVES (soft-delete for stale items).
    """
    if max_age_hours < 1:
        max_age_hours = STALE_CHECKPOINT_HOURS  # Clamp invalid input

    cutoff = (_utc_now() - timedelta(hours=max_age_hours)).isoformat()

    with _db() as conn:
        query = """
            UPDATE checkpoints SET status = 'archived', updated_at = ?
            WHERE status IN ('active', 'paused', 'planning')
            AND updated_at < ?
        """
        params = [_utc_now_iso(), cutoff]

        if exclude_session_id:
            query += " AND (session_id IS NULL OR session_id != ?)"
            params.append(exclude_session_id)

        cursor = conn.execute(query, params)

    archived = cursor.rowcount
    if archived:
        logger.info(f"CKPT-001: Archived {archived} stale checkpoint(s) (cutoff={max_age_hours}h)")
    return archived


# CKPT-002: Compression constants
MEDIUM_TRUNCATE_CHARS = 100
MEDIUM_MAX_LIST_ITEMS = 10


def get_checkpoint_effectiveness() -> dict:
    """CKPT-007: Legacy wrapper — calls get_checkpoint_dashboard() for backward compat."""
    dashboard = get_checkpoint_dashboard()
    return dashboard.get("checkpoint_lifecycle", {})


def get_checkpoint_dashboard() -> dict:
    """MEASURE-020: Comprehensive checkpoint & coverage measurement dashboard.

    Returns 4 metric categories:
    1. checkpoint_lifecycle — status distribution, stale/completion rates, nudge compliance
    2. compaction_recovery — event count, avg recovery_quality, quality distribution
    3. coverage_distribution — score histogram, threshold analysis (BENCH-015)
    4. session_health — drop rate, learning event distribution

    All queries are cold-path (maintenance/API). Not called during conversation_turn.
    """
    import json as _dj

    with _db() as conn:
        # --- Category 1: Checkpoint lifecycle ---
        total = conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
        status_dist = {}
        for row in conn.execute("SELECT status, COUNT(*) FROM checkpoints GROUP BY status").fetchall():
            status_dist[row[0]] = row[1]
        archived = status_dist.get("archived", 0)
        complete = status_dist.get("complete", 0)

        nudge_count = conn.execute(
            "SELECT COUNT(*) FROM governance_events WHERE event_type='checkpoint_nudge_fired'"
        ).fetchone()[0]
        save_count = conn.execute(
            "SELECT COUNT(*) FROM governance_events WHERE event_type='checkpoint_save'"
        ).fetchone()[0]

        avg_saves = conn.execute(
            "SELECT ROUND(AVG(save_count), 1) FROM checkpoints WHERE save_count > 0"
        ).fetchone()[0] or 0

        checkpoint_lifecycle = {
            "total_checkpoints": total,
            "status_distribution": status_dist,
            "stale_rate": round(archived / total * 100, 1) if total > 0 else 0,
            "completion_rate": round(complete / total * 100, 1) if total > 0 else 0,
            "nudge_events": nudge_count,
            "save_events": save_count,
            "nudge_compliance": round(save_count / nudge_count * 100, 1) if nudge_count > 0 else None,
            "avg_saves_per_checkpoint": avg_saves,
        }

        # --- Category 2: Compaction recovery ---
        comp_events = conn.execute(
            "SELECT details FROM governance_events WHERE event_type='compaction_reinjection'"
        ).fetchall()
        comp_count = len(comp_events)
        comp_qualities = []
        for row in comp_events:
            try:
                d = _dj.loads(row[0])
                q = d.get("recovery_quality")
                if q is not None:
                    comp_qualities.append(q)
            except Exception:
                pass

        compaction_recovery = {
            "total_events": comp_count,
            "avg_recovery_quality": round(sum(comp_qualities) / len(comp_qualities), 3) if comp_qualities else None,
            "quality_distribution": {
                "high_0.8_plus": len([q for q in comp_qualities if q >= 0.8]),
                "medium_0.5_0.8": len([q for q in comp_qualities if 0.5 <= q < 0.8]),
                "low_below_0.5": len([q for q in comp_qualities if q < 0.5]),
            },
            "has_resume_rate": round(
                len([r for r in comp_events if '"has_resume": true' in (r[0] or "")]) / comp_count * 100, 1
            ) if comp_count > 0 else None,
        }

        # --- Category 3: Coverage distribution (BENCH-015) ---
        cov_rows = conn.execute(
            "SELECT details FROM governance_events WHERE event_type='coverage_score_recorded' "
            "ORDER BY created_at DESC LIMIT 1000"
        ).fetchall()
        cov_scores = []
        above_threshold_counts = []
        for row in cov_rows:
            try:
                d = _dj.loads(row[0])
                cs = d.get("coverage_score")
                if cs is not None:
                    cov_scores.append(cs)
                at = d.get("above_threshold")
                if at is not None:
                    above_threshold_counts.append(at)
            except Exception:
                pass

        cov_total = len(cov_scores)
        coverage_distribution = {
            "total_recorded": cov_total,
            "histogram": {
                "0.00-0.15": len([s for s in cov_scores if s < 0.15]),
                "0.15-0.30": len([s for s in cov_scores if 0.15 <= s < 0.30]),
                "0.30-0.35": len([s for s in cov_scores if 0.30 <= s < 0.35]),
                "0.35-0.45": len([s for s in cov_scores if 0.35 <= s < 0.45]),
                "0.45-0.60": len([s for s in cov_scores if 0.45 <= s < 0.60]),
                "0.60+": len([s for s in cov_scores if s >= 0.60]),
            } if cov_total > 0 else {},
            "mean_score": round(sum(cov_scores) / cov_total, 4) if cov_total > 0 else None,
            "median_score": round(sorted(cov_scores)[cov_total // 2], 4) if cov_total > 0 else None,
            "threshold_analysis": {
                "current_threshold": 0.35,
                "pct_above_threshold": round(
                    len([s for s in cov_scores if s >= 0.35]) / cov_total * 100, 1
                ) if cov_total > 0 else None,
                "mean_above_threshold_count": round(
                    sum(above_threshold_counts) / len(above_threshold_counts), 1
                ) if above_threshold_counts else None,
            },
        }

        # --- Category 4: Session health ---
        total_sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        short_sessions = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE learning_event_count <= 2"
        ).fetchone()[0]

        session_health = {
            "total_sessions": total_sessions,
            "drop_rate": round(short_sessions / total_sessions * 100, 1) if total_sessions > 0 else 0,
            "learning_distribution": {},
        }
        for row in conn.execute("""
            SELECT
                CASE
                    WHEN learning_event_count <= 1 THEN '0-1'
                    WHEN learning_event_count <= 5 THEN '2-5'
                    WHEN learning_event_count <= 20 THEN '6-20'
                    ELSE '20+'
                END as bucket,
                COUNT(*)
            FROM sessions GROUP BY bucket ORDER BY bucket
        """).fetchall():
            session_health["learning_distribution"][row[0]] = row[1]

    return {
        "checkpoint_lifecycle": checkpoint_lifecycle,
        "compaction_recovery": compaction_recovery,
        "coverage_distribution": coverage_distribution,
        "session_health": session_health,
        "generated_at": _utc_now_iso(),
    }


def analyze_session_drops() -> dict:
    """Analyze session drop rate with full taxonomy and pressure correlation.

    Returns mutually exclusive session categories that sum to total sessions,
    plus pressure × drop-rate cross-tabulation for trend tracking.

    Cold-path only — called via checkpoint endpoint, not conversation_turn.
    """
    with _db() as conn:
        total = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        if total == 0:
            return {"total_sessions": 0, "taxonomy": {}, "pressure_correlation": {}}

        # --- Taxonomy (mutually exclusive, exhaustive) ---
        taxonomy = {}

        taxonomy["quick_lookup"] = conn.execute(
            "SELECT COUNT(*) FROM sessions "
            "WHERE status = 'ended' AND learning_event_count = 0 "
            "AND (julianday(ended_at) - julianday(started_at)) * 1440 < 1"
        ).fetchone()[0]

        taxonomy["warmup_abandoned"] = conn.execute(
            "SELECT COUNT(*) FROM sessions "
            "WHERE status = 'ended' AND learning_event_count = 0 "
            "AND (julianday(ended_at) - julianday(started_at)) * 1440 BETWEEN 1 AND 10"
        ).fetchone()[0]

        taxonomy["brief_exchange"] = conn.execute(
            "SELECT COUNT(*) FROM sessions "
            "WHERE status = 'ended' AND learning_event_count BETWEEN 1 AND 2 "
            "AND (julianday(ended_at) - julianday(started_at)) * 1440 < 5"
        ).fetchone()[0]

        taxonomy["engaged_low_learn"] = conn.execute(
            "SELECT COUNT(*) FROM sessions "
            "WHERE status = 'ended' AND learning_event_count BETWEEN 1 AND 2 "
            "AND (julianday(ended_at) - julianday(started_at)) * 1440 BETWEEN 5 AND 10"
        ).fetchone()[0]

        taxonomy["lost_work"] = conn.execute(
            "SELECT COUNT(*) FROM sessions "
            "WHERE status = 'ended' AND learning_event_count <= 2 "
            "AND (julianday(ended_at) - julianday(started_at)) * 1440 > 10"
        ).fetchone()[0]

        taxonomy["interrupted"] = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE status = 'interrupted'"
        ).fetchone()[0]

        taxonomy["healthy"] = conn.execute(
            "SELECT COUNT(*) FROM sessions "
            "WHERE status = 'ended' AND learning_event_count > 2"
        ).fetchone()[0]

        taxonomy["active"] = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE status = 'active'"
        ).fetchone()[0]

        # Derived metrics
        dropped = sum(v for k, v in taxonomy.items() if k not in ("healthy", "active"))
        checkpoint_beneficiaries = taxonomy["lost_work"] + taxonomy["interrupted"]

        # --- Pressure × drop correlation ---
        pressure_corr = {}
        for label, lo, hi in [
            ("high", 0.25, 999.0),
            ("medium", 0.10, 0.25),
            ("low", 0.0, 0.10),
        ]:
            row = conn.execute(
                "SELECT "
                "  SUM(CASE WHEN learning_event_count <= 2 THEN 1 ELSE 0 END), "
                "  SUM(CASE WHEN learning_event_count > 2 THEN 1 ELSE 0 END) "
                "FROM sessions "
                "WHERE pressure_score IS NOT NULL "
                "AND pressure_score >= ? AND pressure_score < ?",
                (lo, hi),
            ).fetchone()
            d, a = row[0] or 0, row[1] or 0
            pressure_corr[label] = {
                "dropped": d,
                "active": a,
                "drop_rate": round(d / (d + a) * 100, 1) if (d + a) > 0 else None,
            }

    return {
        "total_sessions": total,
        "taxonomy": taxonomy,
        "drop_rate_pct": round(dropped / total * 100, 1),
        "checkpoint_beneficiaries": checkpoint_beneficiaries,
        "checkpoint_beneficiary_pct": round(checkpoint_beneficiaries / total * 100, 1),
        "benign_drop_pct": round((dropped - checkpoint_beneficiaries) / total * 100, 1),
        "pressure_correlation": pressure_corr,
        "generated_at": _utc_now_iso(),
    }


def analyze_coverage_threshold(candidate_thresholds: list[float] | None = None) -> dict:
    """BENCH-015: Analyze coverage_score distribution against candidate thresholds.

    Runs an eligibility sweep: for each threshold, computes the percentage of
    queries that would pass/fail. Helps calibrate COVERAGE_RELEVANCE_THRESHOLD.

    Args:
        candidate_thresholds: List of thresholds to test. Defaults to
            [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    """
    import json as _at_json

    if candidate_thresholds is None:
        candidate_thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    with _db() as conn:
        rows = conn.execute(
            "SELECT details FROM governance_events WHERE event_type='coverage_score_recorded'"
        ).fetchall()

    scores = []
    for row in rows:
        try:
            d = _at_json.loads(row[0])
            cs = d.get("coverage_score")
            if cs is not None:
                scores.append(cs)
        except Exception:
            pass

    if not scores:
        return {"error": "No coverage_score data recorded yet. Run some conversation_turns first."}

    total = len(scores)
    mean = sum(scores) / total
    results = {}
    for thresh in candidate_thresholds:
        above = len([s for s in scores if s >= thresh])
        results[str(thresh)] = {
            "above_count": above,
            "above_pct": round(above / total * 100, 1),
            "below_count": total - above,
            "below_pct": round((total - above) / total * 100, 1),
        }

    return {
        "total_samples": total,
        "mean": round(mean, 4),
        "median": round(sorted(scores)[total // 2], 4),
        "std_dev": round((sum((s - mean) ** 2 for s in scores) / total) ** 0.5, 4),
        "threshold_sweep": results,
        "current_threshold": 0.35,
        "recommendation": None,  # Populated after sufficient data (>100 samples)
    }


def compress_checkpoint(checkpoint: dict) -> dict:
    """CKPT-002: Compress checkpoint content using TTL tier classification.

    DURABLE fields: kept verbatim (task_id, description, done, status)
    MEDIUM fields: truncated to first 100 chars each item (active, next, blockers, concept_refs)
    PERISHABLE fields: stripped entirely (context)

    Returns a new dict with compressed content. Does not mutate input.
    """
    from app.models import CHECKPOINT_FIELD_TTL, CheckpointTTLTier

    compressed = {}
    for field, value in checkpoint.items():
        tier = CHECKPOINT_FIELD_TTL.get(field)

        if tier == CheckpointTTLTier.DURABLE:
            compressed[field] = value  # Keep verbatim
        elif tier == CheckpointTTLTier.MEDIUM:
            # Truncate: strings to 100 chars, lists to first 10 items with truncated strings
            if isinstance(value, str):
                compressed[field] = value[:MEDIUM_TRUNCATE_CHARS]
            elif isinstance(value, list):
                compressed[field] = [
                    (item[:MEDIUM_TRUNCATE_CHARS] if isinstance(item, str) else item)
                    for item in value[:MEDIUM_MAX_LIST_ITEMS]
                ]
            else:
                compressed[field] = value
        elif tier == CheckpointTTLTier.PERISHABLE:
            # Strip entirely — replace with empty equivalent
            if isinstance(value, dict):
                compressed[field] = {}
            elif isinstance(value, list):
                compressed[field] = []
            elif isinstance(value, str):
                compressed[field] = ""
            else:
                compressed[field] = None
        else:
            # Unknown field — keep as-is (forward compatibility)
            compressed[field] = value

    return compressed


# ============================================================
# Resume Context — Rolling Session Snapshots
# Spec: RESUME_CONTEXT_SPEC.md v1.1
# ============================================================

RESUME_SNAPSHOT_TTL_DAYS = 7


def save_resume_snapshot(
    session_id: str,
    active_task: str | None = None,
    task_domain: str | None = None,
    pinned_concepts: list | None = None,
    last_exchange_gist: str | None = None,
    turn_count: int = 0,
    learning_events: int = 0,
    tools_used: list | None = None,
    checkpoint_summary: dict | None = None,  # CONTEXT-001: Checkpoint state for working_context
) -> dict:
    """Upsert rolling snapshot for a session. Replaces prior snapshot entirely."""
    now = _utc_now_iso()
    expires_at = (_utc_now() + timedelta(days=RESUME_SNAPSHOT_TTL_DAYS)).isoformat()

    with _db() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO resume_snapshots
            (session_id, captured_at, active_task, task_domain, pinned_concepts,
             last_exchange_gist, turn_count, learning_events, tools_used,
             checkpoint_summary, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                session_id,
                now,
                (active_task or "")[:80],  # v1.1: enforce 80 char cap
                task_domain,
                json.dumps(pinned_concepts or []),
                (last_exchange_gist or "")[:120],  # v1.1: enforce 120 char cap
                turn_count,
                learning_events,
                json.dumps((tools_used or [])[:5]),  # v1.1: cap at 5 tools
                json.dumps(checkpoint_summary or {}),  # CONTEXT-001
                expires_at,
            ),
        )

    logger.debug(f"Resume snapshot saved: session={session_id} task={active_task}")
    return {
        "session_id": session_id,
        "captured_at": now,
        "active_task": active_task,
        "task_domain": task_domain,
        "pinned_concepts": pinned_concepts or [],
        "last_exchange_gist": last_exchange_gist,
        "turn_count": turn_count,
        "learning_events": learning_events,
        "tools_used": tools_used or [],
        "checkpoint_summary": checkpoint_summary or {},  # CONTEXT-001
        "expires_at": expires_at,
    }


def load_resume_snapshot(prior_session_id: str | None = None) -> dict | None:
    """Load resume snapshot for injection.

    If prior_session_id given, load that session's snapshot.
    Otherwise, load the most recent non-expired snapshot.
    """
    now = _utc_now_iso()

    with _db() as conn:
        if prior_session_id:
            row = conn.execute(
                "SELECT * FROM resume_snapshots WHERE session_id = ? AND expires_at > ?", (prior_session_id, now)
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT * FROM resume_snapshots
                WHERE expires_at > ?
                ORDER BY captured_at DESC LIMIT 1
            """,
                (now,),
            ).fetchone()

    if not row:
        return None

    # CONTEXT-001: Defensive — checkpoint_summary column may not exist pre-migration
    _cs_raw = None
    try:
        _cs_raw = row["checkpoint_summary"]
    except (KeyError, IndexError):
        pass

    return {
        "session_id": row["session_id"],
        "captured_at": row["captured_at"],
        "active_task": row["active_task"],
        "task_domain": row["task_domain"],
        "pinned_concepts": json.loads(row["pinned_concepts"]) if row["pinned_concepts"] else [],
        "last_exchange_gist": row["last_exchange_gist"],
        "turn_count": row["turn_count"],
        "learning_events": row["learning_events"],
        "tools_used": json.loads(row["tools_used"]) if row["tools_used"] else [],
        "checkpoint_summary": json.loads(_cs_raw) if _cs_raw else {},  # CONTEXT-001
        "expires_at": row["expires_at"],
    }


def cleanup_expired_snapshots() -> int:
    """Delete expired resume snapshots. Called alongside checkpoint cleanup."""
    now = _utc_now_iso()
    with _db() as conn:
        cursor = conn.execute("DELETE FROM resume_snapshots WHERE expires_at < ?", (now,))
    deleted = cursor.rowcount
    if deleted:
        logger.info(f"Cleaned up {deleted} expired resume snapshot(s)")
    return deleted


# =============================================================================
# AGENT-002: Agent Token CRUD
# =============================================================================


def create_agent_token(agent_id: str, label: str = "") -> dict:
    """Create a new bearer token for an agent."""
    import secrets

    agent_id = validate_agent_id(agent_id)
    if agent_id == "default":
        raise ValueError("Cannot create token for 'default' agent_id — provide a real agent_id")
    token = f"pith_{secrets.token_urlsafe(32)}"
    now = _utc_now_iso()
    with _db() as conn:
        conn.execute(
            "INSERT INTO agent_tokens (token, agent_id, label, created_at) VALUES (?, ?, ?, ?)",
            (token, agent_id, label, now),
        )
    logger.info(f"Agent token created for agent_id={agent_id} label={label!r}")
    return {"token": token, "agent_id": agent_id, "label": label, "created_at": now}


def resolve_agent_token(token: str) -> str | None:
    """Resolve a bearer token to an agent_id. Returns None if invalid/revoked."""
    if not token or not isinstance(token, str) or not token.startswith("pith_"):
        return None
    with _db() as conn:
        row = conn.execute(
            "SELECT agent_id FROM agent_tokens WHERE token = ? AND revoked_at IS NULL",
            (token,),
        ).fetchone()
        if row:
            conn.execute(
                "UPDATE agent_tokens SET last_used_at = ? WHERE token = ?",
                (_utc_now_iso(), token),
            )
            return row[0]
    return None


def revoke_agent_token(token: str) -> bool:
    """Revoke a token. Returns True if token existed and was revoked."""
    with _db() as conn:
        cursor = conn.execute(
            "UPDATE agent_tokens SET revoked_at = ? WHERE token = ? AND revoked_at IS NULL",
            (_utc_now_iso(), token),
        )
        return cursor.rowcount > 0


def list_agent_tokens(agent_id: str = None) -> list:
    """List tokens, optionally filtered by agent_id. Tokens are masked in output."""
    with _db() as conn:
        if agent_id:
            rows = conn.execute(
                "SELECT token, agent_id, label, created_at, revoked_at, last_used_at "
                "FROM agent_tokens WHERE agent_id = ? ORDER BY created_at DESC",
                (agent_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT token, agent_id, label, created_at, revoked_at, last_used_at "
                "FROM agent_tokens ORDER BY created_at DESC"
            ).fetchall()
    return [
        {
            "token_prefix": r[0][:9] + "...",
            "agent_id": r[1],
            "label": r[2],
            "created_at": r[3],
            "revoked_at": r[4],
            "last_used_at": r[5],
        }
        for r in rows
    ]



# --- RETRIEVAL-024: Cross-domain query expansion support ---

def get_high_authority_concepts_by_ka(knowledge_area: str, limit: int = 3) -> list[dict]:
    """Get highest-authority active concepts for a knowledge area.

    Used by S1.7 cross-domain query expansion and S4.2 cross-domain injection.
    Returns concepts ordered by authority_score descending, excluding
    SUPERSEDED/STALE/CONTRADICTED concepts.

    Args:
        knowledge_area: The knowledge area to query.
        limit: Maximum number of concepts to return.

    Returns:
        List of dicts with 'id' and 'summary' keys.
    """
    with _db() as conn:
        rows = conn.execute(
            """SELECT id, summary FROM concepts
               WHERE status = 'active' AND knowledge_area = ?
               AND currency_status NOT IN ('SUPERSEDED', 'STALE', 'CONTRADICTED')
               AND confidence >= 0.5
               ORDER BY authority_score DESC NULLS LAST
               LIMIT ?""",
            (knowledge_area, limit),
        ).fetchall()
    return [{"id": r[0], "summary": r[1]} for r in rows]


# --- INGEST-037: Verbatim fragment CRUD ---

VERBATIM_BUDGET_PER_CONCEPT = 10_000  # chars (~2.5K tokens)
VERBATIM_BUDGET_TOTAL = 50_000_000  # 50MB total across all concepts

# INGEST-037 Layer 4: Fragment keyword enrichment
FRAGMENT_KEYWORD_CAP = 200  # max chars of keywords per concept

# SQL reserved words to exclude from keyword extraction
_SQL_STOPWORDS = frozenset({
    "SELECT", "FROM", "WHERE", "AND", "OR", "NOT", "IN", "IS", "NULL",
    "INSERT", "INTO", "VALUES", "UPDATE", "SET", "DELETE", "CREATE",
    "TABLE", "INDEX", "DROP", "ALTER", "ADD", "COLUMN", "PRIMARY",
    "KEY", "DEFAULT", "INTEGER", "TEXT", "REAL", "BLOB", "IF", "EXISTS",
    "JOIN", "LEFT", "RIGHT", "INNER", "OUTER", "ON", "AS", "ORDER",
    "BY", "GROUP", "HAVING", "LIMIT", "OFFSET", "UNION", "ALL",
    "CASE", "WHEN", "THEN", "ELSE", "END", "LIKE", "BETWEEN",
    "TRUE", "FALSE", "WITH", "DISTINCT", "COUNT", "SUM", "AVG",
    "MIN", "MAX", "ASC", "DESC", "CAST", "VARCHAR", "BOOLEAN",
    "THE", "FOR", "THIS", "THAT", "WAS", "ARE", "BUT", "HAS",
})


def extract_fragment_keywords(content: str, fragment_type: str = "text") -> str:
    """Extract distinguishing technical keywords from fragment content.

    INGEST-037 Layer 4: Returns space-separated keyword string suitable for
    appending to concept searchable_text and FTS5 summary.

    Prioritizes: SQL function names, table/column identifiers, CamelCase,
    UPPER_CASE, and snake_case tokens. Filters binary content, stopwords,
    and common English words.

    Max output: FRAGMENT_KEYWORD_CAP chars.
    """
    import re

    if not content or not content.strip():
        return ""

    # Skip binary-looking content
    if "bytearray" in content or "\\x" in content[:100]:
        return ""

    tokens: list[str] = []

    # SQL function names: UPPER_CASE identifiers with optional parens
    tokens.extend(re.findall(r'\b([A-Z][A-Z_]{2,})\b', content))

    # Table/column names after SQL keywords
    for match in re.finditer(r'(?:FROM|JOIN|INTO|TABLE|UPDATE)\s+(\w+)', content, re.IGNORECASE):
        tok = match.group(1)
        if len(tok) >= 3:
            tokens.append(tok)

    # CamelCase identifiers
    tokens.extend(re.findall(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b', content))

    # snake_case identifiers (3+ chars, at least one underscore)
    tokens.extend(re.findall(r'\b([a-z][a-z0-9]*_[a-z0-9_]+)\b', content))

    # UPPER_SNAKE_CASE identifiers (like NOAA_GFS0P25)
    tokens.extend(re.findall(r'\b([A-Z][A-Z0-9]*_[A-Z0-9_]+)\b', content))

    # Filter: remove stopwords, short tokens, and numeric-only tokens
    filtered = []
    seen = set()
    for tok in tokens:
        upper = tok.upper()
        if upper in _SQL_STOPWORDS:
            continue
        if len(tok) < 3:
            continue
        # Skip purely numeric tokens or scientific notation
        if re.match(r'^[\d.eE+\-]+$', tok):
            continue
        if upper not in seen:
            seen.add(upper)
            filtered.append(tok)

    # Build keyword string, respecting cap
    kw_str = ""
    for kw in filtered:
        candidate = f"{kw_str} {kw}".strip() if kw_str else kw
        if len(candidate) > FRAGMENT_KEYWORD_CAP:
            break
        kw_str = candidate

    return kw_str


def _recompute_fragment_keywords(conn, concept_id: str) -> str | None:
    """Recompute fragment_keywords for a concept from all its fragments.

    INGEST-037 Layer 4: Called after fragment save/delete to keep keywords current.
    Returns the new keyword string (or None if no fragments).
    """
    # INGEST-038: Exclude conversation fragments — keyword enrichment is for code/SQL/config only
    rows = conn.execute(
        "SELECT content, fragment_type FROM verbatim_fragments WHERE concept_id = ? AND fragment_type != 'conversation' ORDER BY created_at ASC",
        (concept_id,),
    ).fetchall()

    if not rows:
        return None

    # Collect keywords from all fragments, deduplicate
    all_keywords: list[str] = []
    seen = set()
    for content, ftype in rows:
        kw = extract_fragment_keywords(content or "", ftype or "text")
        for tok in kw.split():
            upper = tok.upper()
            if upper not in seen:
                seen.add(upper)
                all_keywords.append(tok)

    # Build keyword string respecting cap
    kw_str = ""
    for kw in all_keywords:
        candidate = f"{kw_str} {kw}".strip() if kw_str else kw
        if len(candidate) > FRAGMENT_KEYWORD_CAP:
            break
        kw_str = candidate

    return kw_str if kw_str else None


def save_verbatim_fragment(
    concept_id: str,
    fragment_type: str = "text",
    content: str | None = None,
    pointer_uri: str | None = None,
    pointer_meta: dict | None = None,
    evidence_id: str | None = None,
    concept_version: str | None = None,
    inherited_from: str | None = None,
    skip_enrichment: bool = False,
) -> str | None:
    """Store a verbatim fragment for a concept. Returns fragment ID or None if budget exceeded."""
    import hashlib
    import json
    import uuid

    char_count = len(content) if content else 0

    # Budget check: per-concept
    with _db() as conn:
        existing = conn.execute(
            "SELECT COALESCE(SUM(char_count), 0) FROM verbatim_fragments WHERE concept_id = ?",
            (concept_id,),
        ).fetchone()[0]
        if existing + char_count > VERBATIM_BUDGET_PER_CONCEPT and char_count > 0:
            logger.warning(
                "INGEST-037: Per-concept verbatim budget exceeded for %s (%d + %d > %d)",
                concept_id, existing, char_count, VERBATIM_BUDGET_PER_CONCEPT,
            )
            return None

        # Dedup via source_hash
        source_hash = None
        if content:
            source_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            dup = conn.execute(
                "SELECT id FROM verbatim_fragments WHERE concept_id = ? AND source_hash = ?",
                (concept_id, source_hash),
            ).fetchone()
            if dup:
                logger.debug("INGEST-037: Dedup — fragment already exists for %s (hash=%s)", concept_id, source_hash[:12])
                return dup[0]

        fragment_id = f"vf_{uuid.uuid4().hex[:16]}"
        pointer_meta_json = json.dumps(pointer_meta) if pointer_meta else None

        conn.execute(
            """INSERT INTO verbatim_fragments
               (id, concept_id, concept_version, evidence_id, fragment_type,
                content, pointer_uri, pointer_meta, char_count, source_hash, inherited_from)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (fragment_id, concept_id, concept_version, evidence_id, fragment_type,
             content, pointer_uri, pointer_meta_json, char_count, source_hash, inherited_from),
        )

    logger.debug("INGEST-037: Saved verbatim fragment %s for concept %s (%s, %d chars)",
                 fragment_id, concept_id, fragment_type, char_count)

    # INGEST-037 Layer 4: Recompute fragment keywords after successful save
    # INGEST-038: skip_enrichment=True for conversation fragments (avoids search index noise)
    if not skip_enrichment and os.environ.get("PITH_FRAGMENT_ENRICHMENT", "true").lower() != "false":
        try:
            with _db() as kw_conn:
                new_kw = _recompute_fragment_keywords(kw_conn, concept_id)
                kw_conn.execute(
                    "UPDATE concepts SET fragment_keywords = ? WHERE id = ?",
                    (new_kw, concept_id),
                )
            n_kw = len(new_kw.split()) if new_kw else 0
            logger.info(
                "INGEST-037-L4: enriched concept %s with %d keywords (%d chars)",
                concept_id, n_kw, len(new_kw) if new_kw else 0,
            )
        except Exception as e:
            logger.warning("INGEST-037-L4: keyword enrichment failed for %s: %s", concept_id, e)

    return fragment_id


def _get_fragments_by_ids(fragment_ids: list[str]) -> dict[str, dict]:
    """INGEST-038: Batch-fetch verbatim fragments by ID. Returns {id: fragment_dict}."""
    import json as _json_batch

    if not fragment_ids:
        return {}
    with _db() as conn:
        placeholders = ",".join("?" for _ in fragment_ids)
        rows = conn.execute(
            f"""SELECT id, concept_id, concept_version, evidence_id, fragment_type,
                       content, pointer_uri, pointer_meta, char_count,
                       created_at, source_hash, inherited_from
                FROM verbatim_fragments WHERE id IN ({placeholders})""",
            fragment_ids,
        ).fetchall()
    result = {}
    for r in rows:
        meta = None
        if r[7]:
            try:
                meta = _json_batch.loads(r[7])
            except Exception:
                meta = r[7]
        result[r[0]] = {
            "id": r[0], "concept_id": r[1], "concept_version": r[2],
            "evidence_id": r[3], "fragment_type": r[4], "content": r[5],
            "pointer_uri": r[6], "pointer_meta": meta, "char_count": r[8],
            "created_at": r[9], "source_hash": r[10], "inherited_from": r[11],
        }
    return result


def get_verbatim_fragments(concept_id: str, limit: int = 10) -> list[dict]:
    """Get verbatim fragments for a concept, ordered by creation time."""
    import json

    with _db() as conn:
        rows = conn.execute(
            """SELECT id, concept_version, evidence_id, fragment_type,
                      content, pointer_uri, pointer_meta, char_count,
                      created_at, source_hash, inherited_from
               FROM verbatim_fragments
               WHERE concept_id = ?
               ORDER BY created_at ASC
               LIMIT ?""",
            (concept_id, limit),
        ).fetchall()

    results = []
    for r in rows:
        meta = None
        if r[6]:
            try:
                meta = json.loads(r[6])
            except Exception:
                meta = r[6]
        results.append({
            "id": r[0],
            "concept_id": concept_id,
            "concept_version": r[1],
            "evidence_id": r[2],
            "fragment_type": r[3],
            "content": r[4],
            "pointer_uri": r[5],
            "pointer_meta": meta,
            "char_count": r[7],
            "created_at": r[8],
            "source_hash": r[9],
            "inherited_from": r[10],
        })

    # INGEST-038: Batch-resolve verbatim:// pointers
    _pointer_map = {}
    for _f in results:
        _uri = _f.get("pointer_uri") or ""
        if _uri.startswith("verbatim://"):
            _pointer_map[_f["id"]] = _uri[len("verbatim://"):]
    if _pointer_map:
        _canonical_ids = list(set(_pointer_map.values()))
        _canonicals = _get_fragments_by_ids(_canonical_ids)
        for _f in results:
            _cid = _pointer_map.get(_f["id"])
            if _cid and _cid in _canonicals:
                _f["content"] = _canonicals[_cid].get("content")
                _f["resolved_from"] = _cid
            elif _cid:
                # Dangling pointer — canonical was deleted or never saved
                _f["content"] = None
                _f["resolved_from"] = _cid
                _f["resolution_error"] = "canonical_not_found"

    return results


def delete_verbatim_fragment(fragment_id: str) -> bool:
    """Delete a specific verbatim fragment. Returns True if deleted."""
    with _db() as conn:
        cursor = conn.execute(
            "DELETE FROM verbatim_fragments WHERE id = ?", (fragment_id,)
        )
    return cursor.rowcount > 0


def delete_verbatim_fragments_for_concept(concept_id: str) -> int:
    """Delete all verbatim fragments for a concept. Returns count deleted."""
    with _db() as conn:
        cursor = conn.execute(
            "DELETE FROM verbatim_fragments WHERE concept_id = ?", (concept_id,)
        )
        deleted = cursor.rowcount
        # INGEST-037 Layer 4: Clear fragment keywords when all fragments removed
        if deleted > 0:
            conn.execute(
                "UPDATE concepts SET fragment_keywords = NULL WHERE id = ?",
                (concept_id,),
            )
    return deleted


def get_verbatim_stats() -> dict:
    """Get aggregate stats for verbatim fragments."""
    with _db() as conn:
        total = conn.execute("SELECT COUNT(*) FROM verbatim_fragments").fetchone()[0]
        total_chars = conn.execute("SELECT COALESCE(SUM(char_count), 0) FROM verbatim_fragments").fetchone()[0]
        concepts_with = conn.execute("SELECT COUNT(DISTINCT concept_id) FROM verbatim_fragments").fetchone()[0]
        by_type = conn.execute(
            "SELECT fragment_type, COUNT(*) FROM verbatim_fragments GROUP BY fragment_type"
        ).fetchall()
    return {
        "total_fragments": total,
        "total_chars": total_chars,
        "concepts_with_verbatim": concepts_with,
        "by_type": {r[0]: r[1] for r in by_type},
    }
