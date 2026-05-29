"""Governance Schema Migration — idempotent schema changes + checkpoint system.

Follows the established pattern from storage.py (B5, P0.3, P1-1 migrations):
  try: ALTER TABLE ... ; except OperationalError: pass

Adds governance columns to concepts table and creates new tables for
governance events, topic activity cache, corrections, and skills.

All migrations are idempotent — safe to run on every startup.
"""

import json
import logging
import re
import sqlite3

from app.core.config import BENCHMARK_READONLY
from app.core.datetime_utils import _utc_now_iso

logger = logging.getLogger(__name__)

_CREATE_INDEX_RE = re.compile(
    r"create\s+index\s+(?:if\s+not\s+exists\s+)?\S+\s+on\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)",
    re.IGNORECASE,
)


def _create_index_columns_available(conn: sqlite3.Connection, sql: str) -> bool:
    """Return False when a CREATE INDEX references columns absent from the target table."""
    match = _CREATE_INDEX_RE.match(sql.strip())
    if not match:
        return True
    table_name, column_expr = match.groups()
    available = {row[1] for row in conn.execute(f"PRAGMA table_info({table_name})")}
    requested = {
        part.strip().split()[0].strip('"`[]')
        for part in column_expr.split(",")
        if part.strip()
    }
    return requested.issubset(available)

# Migration registry — each entry is (migration_id, description, sql_statements)
# sql_statements is a list of (sql, description) tuples
GOVERNANCE_MIGRATIONS = [
    (
        "GOV-001",
        "Add authority_score and currency_score columns to concepts",
        [
            (
                "ALTER TABLE concepts ADD COLUMN authority_score REAL DEFAULT NULL",
                "authority_score column (cached, pre-computed)",
            ),
            (
                "ALTER TABLE concepts ADD COLUMN currency_score REAL DEFAULT NULL",
                "currency_score column (cached, pre-computed)",
            ),
            (
                "ALTER TABLE concepts ADD COLUMN currency_status TEXT DEFAULT 'ACTIVE'",
                "currency_status column (ACTIVE/SUPERSEDED/RESOLVED/STALE/CONTESTED)",
            ),
            (
                "ALTER TABLE concepts ADD COLUMN last_authority_recompute TEXT DEFAULT NULL",
                "timestamp of last authority recomputation",
            ),
            (
                "ALTER TABLE concepts ADD COLUMN last_currency_recompute TEXT DEFAULT NULL",
                "timestamp of last currency recomputation",
            ),
        ],
    ),
    (
        "GOV-002",
        "Create governance_events table",
        [
            (
                """CREATE TABLE IF NOT EXISTS governance_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    turn_id TEXT,
                    event_type TEXT NOT NULL,
                    concept_id TEXT,
                    details TEXT,
                    latency_remaining_ms REAL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )""",
                "governance_events table",
            ),
            (
                "CREATE INDEX IF NOT EXISTS idx_gov_events_session ON governance_events(session_id)",
                "governance_events session index",
            ),
            (
                "CREATE INDEX IF NOT EXISTS idx_gov_events_type ON governance_events(event_type)",
                "governance_events type index",
            ),
            (
                "CREATE INDEX IF NOT EXISTS idx_gov_events_created ON governance_events(created_at)",
                "governance_events created_at index (for 30-day pruning)",
            ),
        ],
    ),
    (
        "GOV-003",
        "Create topic_activity_cache table",
        [
            (
                """CREATE TABLE IF NOT EXISTS topic_activity_cache (
                    knowledge_area TEXT PRIMARY KEY,
                    last_activity_at TEXT NOT NULL,
                    activity_count_30d INTEGER DEFAULT 0,
                    last_recomputed TEXT NOT NULL DEFAULT (datetime('now'))
                )""",
                "topic_activity_cache table (for currency scoring)",
            ),
        ],
    ),
    (
        "GOV-004",
        "Create corrections table",
        [
            (
                """CREATE TABLE IF NOT EXISTS corrections (
                    id TEXT PRIMARY KEY,
                    concept_id TEXT NOT NULL,
                    session_id TEXT,
                    correction_type TEXT NOT NULL,
                    old_value TEXT,
                    new_value TEXT,
                    reason TEXT,
                    cascade_complete INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )""",
                "corrections table (correction cascade tracking)",
            ),
            (
                "CREATE INDEX IF NOT EXISTS idx_corrections_concept ON corrections(concept_id)",
                "corrections concept_id index",
            ),
        ],
    ),
    (
        "GOV-005",
        "Create concept_skills table",
        [
            (
                """CREATE TABLE IF NOT EXISTS concept_skills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    concept_id TEXT NOT NULL,
                    skill_type TEXT NOT NULL,
                    skill_summary TEXT NOT NULL,
                    extracted_from_session TEXT,
                    confidence REAL DEFAULT 0.5,
                    is_constraint INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                )""",
                "concept_skills table (extracted skills + constraints)",
            ),
            (
                "CREATE INDEX IF NOT EXISTS idx_skills_concept ON concept_skills(concept_id)",
                "concept_skills concept_id index",
            ),
        ],
    ),
    (
        "GOV-006",
        "Create migration_checkpoints table",
        [
            (
                """CREATE TABLE IF NOT EXISTS migration_checkpoints (
                    migration_id TEXT PRIMARY KEY,
                    description TEXT,
                    status TEXT NOT NULL DEFAULT 'completed',
                    applied_at TEXT NOT NULL DEFAULT (datetime('now')),
                    rollback_sql TEXT
                )""",
                "migration_checkpoints table (idempotent migration tracking)",
            ),
        ],
    ),
    (
        "GOV-007",
        "Extend corrections table for Wave 2 correction capture protocol",
        [
            (
                "ALTER TABLE corrections ADD COLUMN error_cause TEXT DEFAULT NULL",
                "error_cause column (ErrorClassification taxonomy)",
            ),
            (
                "ALTER TABLE corrections ADD COLUMN corrected_claim TEXT DEFAULT ''",
                "corrected_claim column (what was wrong)",
            ),
            (
                "ALTER TABLE corrections ADD COLUMN correct_claim TEXT DEFAULT ''",
                "correct_claim column (what is correct)",
            ),
            (
                "ALTER TABLE corrections ADD COLUMN affected_concept_ids TEXT DEFAULT '[]'",
                "affected_concept_ids column (JSON array)",
            ),
            (
                "ALTER TABLE corrections ADD COLUMN detection_confidence REAL DEFAULT 0.5",
                "detection_confidence column",
            ),
            (
                "ALTER TABLE corrections ADD COLUMN skill_extracted INTEGER DEFAULT 0",
                "skill_extracted column (for skill extraction pipeline)",
            ),
            (
                "CREATE INDEX IF NOT EXISTS idx_corrections_session ON corrections(session_id)",
                "corrections session_id index",
            ),
            (
                "CREATE INDEX IF NOT EXISTS idx_corrections_error_cause ON corrections(error_cause)",
                "corrections error_cause index (for recurring pattern detection)",
            ),
        ],
    ),
    (
        "GOV-008",
        "Phase 2: Add epistemic classification columns to concepts (§5.5.1-§5.5.3)",
        [
            (
                "ALTER TABLE concepts ADD COLUMN epistemic_network TEXT DEFAULT 'assessment'",
                "epistemic_network column (world_fact | preference | assessment)",
            ),
            (
                "ALTER TABLE concepts ADD COLUMN verification_status TEXT DEFAULT 'unverified'",
                "verification_status column (verified | unverified | stale | contradicted)",
            ),
            (
                "ALTER TABLE concepts ADD COLUMN verification_fraction REAL DEFAULT 0.0",
                "verification_fraction column (0.0-1.0 continuous scoring, §5.5.2)",
            ),
            (
                "ALTER TABLE concepts ADD COLUMN effective_authority REAL DEFAULT NULL",
                "effective_authority column (authority after epistemic cap, computed at retrieval)",
            ),
            (
                "CREATE INDEX IF NOT EXISTS idx_concepts_epistemic ON concepts(epistemic_network)",
                "epistemic_network index for network distribution queries",
            ),
            (
                "CREATE INDEX IF NOT EXISTS idx_concepts_verification ON concepts(verification_status)",
                "verification_status index for verification audit queries",
            ),
        ],
    ),
    (
        "GOV-009",
        "Phase 2: Non-lossy evolution columns + temporal snapshot indexes (§5.2.4, §5.4.4)",
        [
            (
                "ALTER TABLE concepts ADD COLUMN is_current INTEGER DEFAULT 1",
                "is_current column (1=current version, 0=superseded)",
            ),
            (
                "ALTER TABLE concepts ADD COLUMN superseded_at TEXT DEFAULT NULL",
                "superseded_at column (ISO timestamp when this version was superseded)",
            ),
            (
                "ALTER TABLE concepts ADD COLUMN superseded_by TEXT DEFAULT NULL",
                "superseded_by column (concept_id of newer version)",
            ),
            (
                "ALTER TABLE concepts ADD COLUMN version_chain_head TEXT DEFAULT NULL",
                "version_chain_head column (concept_id of the original in the chain)",
            ),
            (
                "ALTER TABLE concepts ADD COLUMN reinforcement_count INTEGER DEFAULT 0",
                "reinforcement_count column (for anti-bias capped reinforcement §5.4.3)",
            ),
            (
                "CREATE INDEX IF NOT EXISTS idx_concepts_is_current ON concepts(is_current)",
                "is_current index for filtering current versions",
            ),
            (
                "CREATE INDEX IF NOT EXISTS idx_concepts_temporal_snapshot ON concepts(is_current, created_at, superseded_at)",
                "temporal snapshot composite index (§5.4.4)",
            ),
            (
                "CREATE INDEX IF NOT EXISTS idx_concepts_ka_current_created ON concepts(knowledge_area, is_current, created_at)",
                "KA lifecycle recount index for grouped promotion maintenance",
            ),
            (
                "CREATE INDEX IF NOT EXISTS idx_concepts_current_ka_created ON concepts(is_current, knowledge_area, created_at)",
                "KA lifecycle recount covering index for current grouped scans",
            ),
            (
                "CREATE INDEX IF NOT EXISTS idx_concepts_version_chain ON concepts(version_chain_head, created_at)",
                "version chain walking index (§5.4.4)",
            ),
        ],
    ),
    (
        "GOV-010",
        "Phase 2: Version history archive table + retention policy (§5.4.2)",
        [
            (
                """CREATE TABLE IF NOT EXISTS concept_versions_archive (
                    id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    summary TEXT,
                    data JSON NOT NULL,
                    created_at TEXT NOT NULL,
                    superseded_at TEXT,
                    archived_at TEXT NOT NULL DEFAULT (datetime('now')),
                    PRIMARY KEY (id, version)
                )""",
                "concept_versions_archive table for old superseded versions",
            ),
            (
                "CREATE INDEX IF NOT EXISTS idx_archive_id ON concept_versions_archive(id)",
                "archive id index for version chain lookups",
            ),
        ],
    ),
    (
        "GOV-011",
        "Phase 2: Migrate existing concepts to assessment/unverified epistemic defaults (H8, H9)",
        [
            (
                """UPDATE concepts SET epistemic_network = 'assessment'
                   WHERE epistemic_network IS NULL OR epistemic_network = 'assessment'""",
                "default all existing concepts to assessment network (safest classification)",
            ),
            (
                """UPDATE concepts SET verification_status = 'unverified'
                   WHERE verification_status IS NULL OR verification_status = 'unverified'""",
                "default all existing concepts to unverified status",
            ),
            (
                """UPDATE concepts SET verification_fraction = 0.0
                   WHERE verification_fraction IS NULL OR verification_fraction = 0.0""",
                "default all existing concepts to 0.0 verification fraction",
            ),
            (
                """UPDATE concepts SET is_current = 1
                   WHERE is_current IS NULL""",
                "mark all existing concepts as current version",
            ),
            (
                """UPDATE concepts SET reinforcement_count = 0
                   WHERE reinforcement_count IS NULL""",
                "initialize reinforcement count for existing concepts",
            ),
        ],
    ),
    (
        "GOV-012",
        "Phase 3: Create benchmark_scores table for PoisonBench tracking (§5.7.6)",
        [
            (
                """CREATE TABLE IF NOT EXISTS benchmark_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    benchmark_name TEXT NOT NULL,
                    dimension TEXT NOT NULL,
                    score REAL NOT NULL,
                    phase TEXT,
                    run_at TEXT NOT NULL,
                    git_commit TEXT
                )""",
                "benchmark score tracking table",
            ),
            (
                """CREATE INDEX IF NOT EXISTS idx_benchmark_scores_lookup
                   ON benchmark_scores(benchmark_name, dimension, run_at DESC)""",
                "benchmark scores lookup index",
            ),
            (
                """CREATE TABLE IF NOT EXISTS poisonbench_results (
                    run_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    overall_score REAL NOT NULL,
                    result_json JSON NOT NULL,
                    version TEXT DEFAULT '1.0',
                    created_at TEXT DEFAULT (datetime('now'))
                )""",
                "poisonbench full results table",
            ),
            (
                """CREATE INDEX IF NOT EXISTS idx_poisonbench_timestamp
                   ON poisonbench_results(timestamp DESC)""",
                "poisonbench results timestamp index",
            ),
        ],
    ),
    (
        "GOV-013",
        "S7.1: Indexes for strategic orientation query (concept_type, currency_status, composite)",
        [
            (
                "CREATE INDEX IF NOT EXISTS idx_concepts_concept_type ON concepts(concept_type)",
                "concept_type index for strategic orientation query filtering",
            ),
            (
                "CREATE INDEX IF NOT EXISTS idx_concepts_currency_status ON concepts(currency_status)",
                "currency_status index for ACTIVE currency filtering",
            ),
            (
                """CREATE INDEX IF NOT EXISTS idx_concepts_strategic
                   ON concepts(status, concept_type, confidence DESC)
                   WHERE status = 'active'""",
                "composite partial index for strategic orientation query (S7.1)",
            ),
        ],
    ),
    # Fix 4: concepts_created/concepts_evolved columns for cognitive velocity
    (
        "GOV-014",
        "Fix 4: Add concepts_created and concepts_evolved to sessions table for cognitive velocity",
        [
            (
                "ALTER TABLE sessions ADD COLUMN concepts_created INTEGER DEFAULT 0",
                "add concepts_created column to sessions",
            ),
            (
                "ALTER TABLE sessions ADD COLUMN concepts_evolved INTEGER DEFAULT 0",
                "add concepts_evolved column to sessions",
            ),
        ],
    ),
    (
        "GOV-015",
        "CONTRA-012: Backfill pre-TB-2 SUPPRESS_LOSER targets as CONTRADICTED",
        [
            (
                """UPDATE concepts
                   SET currency_status = 'CONTRADICTED'
                   WHERE id IN (
                       SELECT DISTINCT ge.concept_id
                       FROM governance_events ge
                       JOIN concepts c ON c.id = ge.concept_id
                       WHERE json_valid(ge.details)
                         AND json_extract(ge.details, '$.action') = 'SUPPRESS_LOSER'
                         AND c.status = 'active'
                         AND (c.currency_status IS NULL
                              OR c.currency_status IN ('ACTIVE', 'CONTESTED'))
                   )""",
                "Backfill pre-TB-2 SUPPRESS_LOSER targets as CONTRADICTED",
            ),
        ],
    ),
    (
        "AGENT-004",
        "Add session_id column to concepts table with index and backfill from metadata",
        [
            ("ALTER TABLE concepts ADD COLUMN session_id TEXT DEFAULT NULL", "add session_id column to concepts table"),
            (
                "CREATE INDEX IF NOT EXISTS idx_concepts_session ON concepts(session_id)",
                "session_id index for federation queries",
            ),
            (
                """UPDATE concepts SET session_id = json_extract(data, '$.metadata.source_session')
               WHERE session_id IS NULL
               AND data IS NOT NULL
               AND json_valid(data)
               AND json_extract(data, '$.metadata.source_session') IS NOT NULL""",
                "backfill session_id from existing JSON metadata",
            ),
        ],
    ),
    (
        "GOV-016",
        "FEDERATION L1.5: Add model_id to sessions for model provenance tracking",
        [
            (
                "ALTER TABLE sessions ADD COLUMN model_id TEXT NOT NULL DEFAULT 'unknown'",
                "add model_id column to sessions table",
            ),
        ],
    ),
    (
        "GOV-017",
        "Federation L2: federation_events + bridge tables",
        [
            (
                """CREATE TABLE IF NOT EXISTS federation_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                concept_id TEXT,
                source_session_id TEXT,
                source_model_id TEXT DEFAULT 'unknown',
                source_agent_id TEXT DEFAULT 'default',
                payload JSON NOT NULL,
                origin_brain TEXT,
                bridge_depth INTEGER DEFAULT 0,
                consumed INTEGER DEFAULT 0,
                consumed_at TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )""",
                "federation_events table",
            ),
            (
                "CREATE INDEX IF NOT EXISTS idx_fed_events_unconsumed ON federation_events(consumed) WHERE consumed = 0",
                "federation_events unconsumed index",
            ),
            (
                """CREATE TABLE IF NOT EXISTS bridge_event_consumption (
                bridge_id TEXT NOT NULL,
                event_id INTEGER NOT NULL,
                consumed_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (bridge_id, event_id)
            )""",
                "bridge_event_consumption junction table",
            ),
            (
                """CREATE TABLE IF NOT EXISTS schema_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )""",
                "schema_meta for version tracking",
            ),
            (
                "INSERT OR IGNORE INTO schema_meta (key, value) VALUES ('schema_version', '1.0')",
                "initial schema version",
            ),
        ],
    ),
    (
        "STABILITY-006A",
        "Repair currency_status dual-write desyncs from TB-2 contradiction persistence",
        [
            (
                """UPDATE concepts
                   SET data = json_set(data, '$.currency_status', currency_status)
                   WHERE is_current = 1
                   AND json_valid(data)
                   AND currency_status IS NOT NULL
                   AND currency_status != COALESCE(json_extract(data, '$.currency_status'), '')""",
                "Sync JSON $.currency_status from SQL column for all desynced concepts",
            ),
        ],
    ),
    (
        "CURRENCY-001",
        # MAINT-024: NEVER use datetime('now') or strftime(...,'now') for last_accessed
        # in any migration SQL. Doing so poisons freshness scores (forces freshness→1.0)
        # for ALL concepts for ~24h post-migration. Always preserve existing timestamps:
        # use json_extract(data, '$.last_accessed') for JSON sync, or created_at for
        # backfill, or leave NULL. See scripts/backfill_last_accessed.py for safe pattern.
        "Repair last_accessed SQL-JSON desync from pre-RETRIEVAL-012 retrieval tracking",
        [
            (
                """UPDATE concepts
                   SET last_accessed = json_extract(data, '$.last_accessed')
                   WHERE json_valid(data)
                   AND json_extract(data, '$.last_accessed') IS NOT NULL
                   AND json_extract(data, '$.last_accessed') != ''
                   AND (last_accessed IS NULL
                        OR last_accessed != json_extract(data, '$.last_accessed'))""",
                "Sync SQL last_accessed FROM JSON blob where JSON has a value",
            ),
            (
                """UPDATE concepts
                   SET last_accessed = NULL
                   WHERE json_valid(data)
                   AND (json_extract(data, '$.last_accessed') IS NULL
                        OR json_extract(data, '$.last_accessed') = '')
                   AND last_accessed IS NOT NULL""",
                "NULL out SQL last_accessed where JSON blob has no value",
            ),
        ],
    ),
    (
        "MONITOR-001",
        "Add pressure_score column to sessions table for CTX-003 trend analysis",
        [
            (
                "ALTER TABLE sessions ADD COLUMN pressure_score REAL DEFAULT NULL",
                "Add pressure_score column to sessions table",
            ),
        ],
    ),
    (
        "GOV-018",
        "FED-013: Session registry columns for federation heartbeat",
        [
            (
                "ALTER TABLE sessions ADD COLUMN last_heartbeat TEXT",
                "sessions.last_heartbeat column for federation registry",
            ),
            (
                "ALTER TABLE sessions ADD COLUMN working_context_json TEXT",
                "sessions.working_context_json column for federation registry",
            ),
            (
                "CREATE INDEX IF NOT EXISTS idx_sessions_heartbeat ON sessions(last_heartbeat) WHERE status = 'active'",
                "sessions heartbeat index for stale cleanup queries",
            ),
        ],
    ),
    (
        "GOV-019",
        "Add last_learning_at column to sessions table (DEBT-177 DDL sync)",
        [
            (
                "ALTER TABLE sessions ADD COLUMN last_learning_at TEXT DEFAULT NULL",
                "last_learning_at column for session learning timestamp tracking",
            ),
        ],
    ),
    (
        "GOV-020",
        "DATA-040: Clean up zombie concepts (superseded_at set but superseded_by NULL)",
        [
            (
                "UPDATE concepts SET superseded_by = '__orphaned_supersession__' "
                "WHERE superseded_at IS NOT NULL AND superseded_by IS NULL",
                "Tag orphaned superseded concepts with sentinel value",
            ),
        ],
    ),
    (
        "GOV-021",
        "CONTRA-001: Create contradiction_resolutions table",
        [
            (
                "CREATE TABLE IF NOT EXISTS contradiction_resolutions ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "concept_a_id TEXT NOT NULL, "
                "concept_b_id TEXT NOT NULL, "
                "contradiction_type TEXT NOT NULL, "
                "detection_phase INTEGER NOT NULL, "
                "similarity_score REAL, "
                "action TEXT NOT NULL, "
                "winner_id TEXT, "
                "loser_id TEXT, "
                "reason TEXT, "
                "source TEXT DEFAULT 'retrieval', "
                "session_id TEXT, "
                "created_at TEXT NOT NULL DEFAULT (datetime('now')))",
                "contradiction_resolutions table",
            ),
            (
                "CREATE INDEX IF NOT EXISTS idx_contra_res_concepts "
                "ON contradiction_resolutions(concept_a_id, concept_b_id)",
                "contradiction_resolutions concept pair index",
            ),
            (
                "CREATE INDEX IF NOT EXISTS idx_contra_res_action ON contradiction_resolutions(action)",
                "contradiction_resolutions action index",
            ),
        ],
    ),
    (
        "GOV-025",
        "MATURITY-003: Data integrity cleanup + composite index",
        [
            # Backfill version_chain_head for concepts missing it (~400 concepts)
            (
                "UPDATE concepts SET version_chain_head = id WHERE version_chain_head IS NULL AND is_current = 1",
                "Backfill version_chain_head for current concepts missing it",
            ),
            # Tag orphaned non-current concepts (is_current=0 with no current version)
            (
                "UPDATE concepts SET maturity = 'QUARANTINED' "
                "WHERE is_current = 0 AND id NOT IN ("
                "SELECT id FROM concepts WHERE is_current = 1"
                ") AND maturity != 'QUARANTINED'",
                "Quarantine orphaned non-current concepts",
            ),
            # Composite index for contradiction recency queries
            (
                "CREATE INDEX IF NOT EXISTS idx_concepts_maturity_updated ON concepts(maturity, updated_at)",
                "Composite index for maturity + recency queries",
            ),
        ],
    ),
    (
        "COGGOV-005",
        "Add protected column to concepts for governance safety guards",
        [
            (
                "ALTER TABLE concepts ADD COLUMN protected INTEGER DEFAULT 0",
                "protected flag — immune from automated governance suppression",
            ),
        ],
    ),
    (
        "DATA-065",
        "Add last_organic_access column to distinguish organic vs bulk access timestamps",
        [
            (
                "ALTER TABLE concepts ADD COLUMN last_organic_access TEXT DEFAULT NULL",
                "last_organic_access column — set ONLY by load_concept(track_access=True)",
            ),
            (
                "CREATE INDEX IF NOT EXISTS idx_concepts_last_organic_access ON concepts(last_organic_access)",
                "Index for freshness queries on organic access timestamps",
            ),
            # Conservative backfill: copy last_accessed to last_organic_access for
            # concepts whose last_accessed is NOT part of a bulk timestamp cluster.
            # A timestamp shared by >50 concepts in the same second is classified as bulk.
            (
                """UPDATE concepts SET last_organic_access = last_accessed
                   WHERE last_accessed IS NOT NULL
                   AND (SELECT COUNT(*) FROM concepts c2
                        WHERE substr(c2.last_accessed, 1, 19) = substr(concepts.last_accessed, 1, 19)) <= 50""",
                "Backfill last_organic_access for non-bulk concepts (threshold: >50 siblings = bulk)",
            ),
        ],
    ),
    (
        "INGEST-037-L4",
        "Add fragment_keywords column — cached keywords from verbatim fragments for retrieval enrichment",
        [
            (
                "ALTER TABLE concepts ADD COLUMN fragment_keywords TEXT DEFAULT NULL",
                "fragment_keywords column — extracted technical terms from verbatim fragments",
            ),
        ],
    ),
    (
        "RETRIEVAL-070",
        "Create FTS5 verbatim index for keyword search over conversation fragments",
        [
            (
                """CREATE VIRTUAL TABLE IF NOT EXISTS fts_verbatim
                   USING fts5(fragment_id UNINDEXED, concept_id UNINDEXED,
                              user_content, full_content,
                              tokenize='porter ascii')""",
                "fts_verbatim FTS5 table for keyword search over INGEST-038 verbatim fragments",
            ),
        ],
    ),
    (
        "RETRIEVAL-104",
        "Add edit_provenance column for entity chain filtering",
        [
            (
                "ALTER TABLE concepts ADD COLUMN edit_provenance TEXT DEFAULT NULL",
                "add edit_provenance column to concepts table",
            ),
        ],
    ),
    (
        "COGGOV-012",
        "Add user_message column to corrections for Path D+a zero-extraction",
        [
            (
                "ALTER TABLE corrections ADD COLUMN user_message TEXT DEFAULT ''",
                "store raw user correction message for evidence-append and backtesting",
            ),
        ],
    ),
    (
        "EUNOMIA-040-F3",
        "Add subject_key column + index for RETRIEVAL-072 subject-key dedup optimization",
        [
            (
                "ALTER TABLE concepts ADD COLUMN subject_key TEXT DEFAULT NULL",
                "subject_key column — pre-computed _extract_subject_key for indexed lookup",
            ),
            (
                "CREATE INDEX IF NOT EXISTS idx_concepts_subject_key ON concepts(subject_key) WHERE subject_key IS NOT NULL",
                "Partial index on subject_key for RETRIEVAL-072 dedup queries",
            ),
        ],
    ),
    (
        "COGGOV-014",
        "Add stale-risk lifecycle columns for criteria-based staleness detector",
        [
            (
                "ALTER TABLE concepts ADD COLUMN staleness_state TEXT DEFAULT NULL",
                "staleness_state column (AGING / REVIEW / NULL)",
            ),
            (
                "ALTER TABLE concepts ADD COLUMN staleness_score REAL DEFAULT NULL",
                "staleness_score column (criteria detector score)",
            ),
            (
                "ALTER TABLE concepts ADD COLUMN staleness_reason TEXT DEFAULT NULL",
                "staleness_reason column (JSON rationale)",
            ),
            (
                "ALTER TABLE concepts ADD COLUMN staleness_evaluated_at TEXT DEFAULT NULL",
                "staleness_evaluated_at column (last detector evaluation)",
            ),
            (
                "ALTER TABLE concepts ADD COLUMN staleness_detector_version TEXT DEFAULT NULL",
                "staleness_detector_version column",
            ),
            (
                "ALTER TABLE concepts ADD COLUMN staleness_consecutive_hits INTEGER DEFAULT 0",
                "staleness_consecutive_hits column",
            ),
            (
                "CREATE INDEX IF NOT EXISTS idx_concepts_staleness_state ON concepts(staleness_state)",
                "index stale-risk lifecycle state for review queues",
            ),
        ],
    ),
    (
        "RETRIEVAL-119",
        "Add active KA authority index for bounded cross-domain expansion lookups",
        [
            (
                """CREATE INDEX IF NOT EXISTS idx_concepts_active_ka_authority
                   ON concepts(knowledge_area, authority_score DESC, confidence)
                   WHERE status = 'active'
                     AND currency_status NOT IN ('SUPERSEDED', 'STALE', 'CONTRADICTED')""",
                "partial index for active high-authority concepts by knowledge area",
            ),
        ],
    ),
]


def _backfill_fts_verbatim(conn: sqlite3.Connection) -> int:
    """RETRIEVAL-070: Backfill fts_verbatim from existing verbatim_fragments.

    Only processes conversation-type fragments (INGEST-038 format).
    Extracts USER portion for user_content column.
    Idempotent — skips fragments already in fts_verbatim.
    """
    try:
        # Check if both tables exist
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table') AND name IN ('verbatim_fragments', 'fts_verbatim')"
        ).fetchall()}
        if 'verbatim_fragments' not in tables or 'fts_verbatim' not in tables:
            return 0

        # Get existing fragment_ids already in fts_verbatim
        existing_ids = {r[0] for r in conn.execute(
            "SELECT fragment_id FROM fts_verbatim"
        ).fetchall()}

        # Fetch conversation fragments not yet indexed
        rows = conn.execute(
            "SELECT id, concept_id, content FROM verbatim_fragments "
            "WHERE fragment_type = 'conversation' AND content IS NOT NULL"
        ).fetchall()

        backfilled = 0
        for frag_id, concept_id, content in rows:
            if frag_id in existing_ids:
                continue
            # Extract USER portion from INGEST-038 format
            user_content = content
            full_content = content
            if '[USER]' in content and '\n\n[ASSISTANT]' in content:
                user_content = content.split('\n\n[ASSISTANT]')[0].replace('[USER] ', '')
            conn.execute(
                "INSERT INTO fts_verbatim(fragment_id, concept_id, user_content, full_content) "
                "VALUES (?, ?, ?, ?)",
                (frag_id, concept_id, user_content, full_content),
            )
            backfilled += 1

        if backfilled:
            conn.commit()
        logger.info("RETRIEVAL-070: Backfilled %d/%d conversation fragments into fts_verbatim", backfilled, len(rows))
        return backfilled
    except Exception as e:
        logger.warning("RETRIEVAL-070: fts_verbatim backfill failed (non-fatal): %s", e)
        return 0


def _is_migration_applied(conn: sqlite3.Connection, migration_id: str) -> bool:
    """Check if a migration has already been recorded in checkpoints."""
    try:
        row = conn.execute(
            "SELECT 1 FROM migration_checkpoints WHERE migration_id = ?",
            (migration_id,),
        ).fetchone()
        return row is not None
    except sqlite3.OperationalError:
        # migration_checkpoints table doesn't exist yet — not applied
        return False


def _record_migration(conn: sqlite3.Connection, migration_id: str, description: str) -> None:
    """Record a successful migration in the checkpoint table."""
    conn.execute(
        "INSERT OR IGNORE INTO migration_checkpoints (migration_id, description, applied_at) VALUES (?, ?, ?)",
        (migration_id, description, _utc_now_iso()),
    )


def run_governance_migrations(conn: sqlite3.Connection) -> dict:
    """Run all governance migrations idempotently.

    Returns a summary dict: {applied: [...], skipped: [...], errors: [...]}.
    Safe to call on every startup.
    """
    # Guard: GOV migrations require concepts table (Fix 2 + A4)
    table_check = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='concepts'").fetchone()
    if not table_check:
        db_path = conn.execute("PRAGMA database_list").fetchone()
        logger.debug("GOV migrations skipped: concepts table not yet created (db=%s)", db_path)
        return {"applied": [], "skipped": [], "errors": []}

    applied = []
    skipped = []
    errors = []

    # Bootstrap: ensure migration_checkpoints table exists before anything else
    for mid, mdesc, mstmts in GOVERNANCE_MIGRATIONS:
        if mid == "GOV-006":
            for sql, stmt_desc in mstmts:
                try:
                    conn.execute(sql)
                    conn.commit()
                except sqlite3.OperationalError:
                    pass  # Already exists
            break

    for migration_id, description, statements in GOVERNANCE_MIGRATIONS:
        if _is_migration_applied(conn, migration_id):
            skipped.append(migration_id)
            continue

        migration_errors = []
        for sql, stmt_desc in statements:
            try:
                if not _create_index_columns_available(conn, sql):
                    logger.info("GOV migration %s skipped unavailable optional index: %s", migration_id, stmt_desc)
                    continue
                conn.execute(sql)
                conn.commit()
                logger.info("GOV migration %s: %s", migration_id, stmt_desc)
            except sqlite3.OperationalError as e:
                err_str = str(e).lower()
                if "duplicate column" in err_str or "already exists" in err_str:
                    pass  # Idempotent — already applied
                else:
                    migration_errors.append(f"{stmt_desc}: {e}")
                    logger.error("GOV migration %s failed on %s: %s", migration_id, stmt_desc, e)

        if migration_errors:
            errors.extend(migration_errors)
        else:
            _record_migration(conn, migration_id, description)
            applied.append(migration_id)
            logger.info("GOV migration %s complete: %s", migration_id, description)

    # PERF-007: Invalidate federation table cache if any migration was applied
    # (a migration may have created federation_events table mid-session)
    if applied:
        try:
            from app.session import SessionManager

            SessionManager._reset_federation_cache()
        except ImportError:
            pass

    # Migration: GOV-002 — seed explicit anti_terms for always-activate constraint concepts
    _CONSTRAINT_ANTI_TERMS = {
        "constraint_consumer_internal_separation": [
            "add to pith-beta", "save in pith-beta", "put in pith-beta",
            "commit to pith-beta", "store in pith-beta"
        ],
        "constraint_docs_to_pith_internal": [
            "save to pith-beta", "commit to pith-beta", "write to pith-beta",
            "put in pith-beta", "push to pith-beta"
        ],
        "constraint_mac_mini_home_username_pith": [
            "stale local user path", "deprecated local username",
            "legacy home path", "stale home path"
        ],
        "constraint_no_assumptions_verify_first": [
            "i assume", "must be", "probably", "likely", "without verifying",
            "without checking", "i believe"
        ],
    }

    seeded_count = 0
    for concept_id, anti_terms in _CONSTRAINT_ANTI_TERMS.items():
        try:
            existing = conn.execute(
                "SELECT json_extract(data, '$.anti_terms') FROM concepts WHERE id = ?",
                (concept_id,),
            ).fetchone()
            # Only seed if not already populated (idempotent)
            if existing and existing[0] is None:
                conn.execute(
                    "UPDATE concepts SET data = json_patch(data, json_object('anti_terms', json(?))), "
                    "updated_at = ? WHERE id = ?",
                    (json.dumps(anti_terms), _utc_now_iso(), concept_id),
                )
                seeded_count += 1
        except Exception as e:
            logger.warning("GOV-002 migration: failed to seed anti_terms for %s: %s", concept_id, e)
    if seeded_count:
        conn.commit()
    logger.info("GOV-002: Seeded anti_terms for %d constraint concepts", seeded_count)

    # RETRIEVAL-070: Backfill fts_verbatim from existing verbatim_fragments
    if "RETRIEVAL-070" in applied:
        _backfill_fts_verbatim(conn)

    # DATA-064: Fix FTS index parity issues
    try:
        backfilled, concepts_cleaned, verbatim_cleaned = migrate_fts_parity_data064(conn)
        logger.info(
            "DATA-064: FTS parity migration: %d backfilled, %d concepts cleaned, %d verbatim cleaned",
            backfilled, concepts_cleaned, verbatim_cleaned
        )
    except Exception as e:
        logger.warning("DATA-064: FTS parity migration failed: %s", e)

    summary = {"applied": applied, "skipped": skipped, "errors": errors}
    logger.info(
        "Governance migration summary: %s applied, %s skipped, %s errors", len(applied), len(skipped), len(errors)
    )
    return summary


def migrate_fts_parity_data064(conn):
    """DATA-064: Fix FTS index parity issues.
    
    1. Backfill fts_concepts for active concepts missing entries
    2. Remove orphaned fts_concepts entries for non-active concepts
    3. Remove orphaned fts_verbatim entries for non-active concepts
    """
    if BENCHMARK_READONLY:
        logger.info("DATA-064: FTS parity migration skipped (PITH_BENCHMARK_READONLY)")
        return 0, 0, 0

    from app.storage import _sync_fts5
    
    # Phase 1: Backfill missing fts_concepts entries
    missing = conn.execute("""
        SELECT c.id, c.summary FROM concepts c
        WHERE c.status = 'active' AND c.id NOT IN (
            SELECT concept_id FROM fts_concepts
        )
    """).fetchall()
    for cid, summary in missing:
        _sync_fts5(conn, cid, summary)
    backfilled = len(missing)
    
    # Phase 2: Remove orphaned fts_concepts (superseded/retired/deleted concepts)
    orphan_result = conn.execute("""
        DELETE FROM fts_concepts WHERE concept_id NOT IN (
            SELECT id FROM concepts WHERE status = 'active'
        )
    """)
    concepts_cleaned = orphan_result.rowcount
    
    # Phase 3: Remove orphaned fts_verbatim (superseded/retired/deleted concepts)
    verbatim_result = conn.execute("""
        DELETE FROM fts_verbatim WHERE concept_id NOT IN (
            SELECT id FROM concepts WHERE status = 'active'
        )
    """)
    verbatim_cleaned = verbatim_result.rowcount
    
    conn.commit()
    return backfilled, concepts_cleaned, verbatim_cleaned


def migrate_eunomia_040_backfill_subject_key(conn: sqlite3.Connection) -> int:
    """EUNOMIA-040 Fix 3: Backfill subject_key for existing concepts.

    One-time migration: compute _extract_subject_key(summary) for all active
    concepts missing subject_key. Uses the same regex as the write-time path.
    """
    # CI-029: importlib lazy loader — avoids storage→session static import (Contract 2).
    # Same pattern as DEBT-237..240 precedents in app/cognitive/ and app/governance/.
    import importlib
    _extract_subject_key = importlib.import_module("app.session.helpers")._extract_subject_key

    rows = conn.execute(
        "SELECT id, summary FROM concepts WHERE subject_key IS NULL AND superseded_by IS NULL"
    ).fetchall()
    updated = 0
    for concept_id, summary in rows:
        key = _extract_subject_key(summary or "")
        if key:
            conn.execute(
                "UPDATE concepts SET subject_key = ? WHERE id = ?", (key, concept_id)
            )
            updated += 1
    conn.commit()
    logger.info(
        "EUNOMIA-040: Backfilled subject_key for %d/%d active concepts", updated, len(rows)
    )
    return updated
