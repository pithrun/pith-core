"""Storage sub-module: utils.

Constants, validation helpers, schema DDL, and DB filename migration.
Extracted from storage/__init__.py during Item 2b decomposition.
"""
import json
import logging
import os
import re
import sqlite3
from contextlib import suppress

logger = logging.getLogger(__name__)

# AGENT-001: agent_id validation
_AGENT_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")

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
    staleness_state TEXT DEFAULT NULL,
    staleness_score REAL DEFAULT NULL CHECK(staleness_score IS NULL OR (staleness_score >= 0.0 AND staleness_score <= 1.0)),
    staleness_reason TEXT DEFAULT NULL,
    staleness_evaluated_at TEXT DEFAULT NULL,
    staleness_detector_version TEXT DEFAULT NULL,
    staleness_consecutive_hits INTEGER DEFAULT 0,
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
    fragment_keywords TEXT DEFAULT NULL,
    utility_score REAL DEFAULT 0.5,
    utility_samples INTEGER DEFAULT 0,
    utility_updated TEXT DEFAULT NULL,
    last_synthesis_evaluated_at TEXT DEFAULT NULL,
    edit_provenance TEXT DEFAULT NULL,
    subject_key TEXT DEFAULT NULL
);

CREATE INDEX IF NOT EXISTS idx_concepts_session ON concepts(session_id);
-- FTS5 full-text index for BM25 keyword search (RETRIEVAL-042 upgrade)
CREATE VIRTUAL TABLE IF NOT EXISTS fts_concepts
    USING fts5(concept_id UNINDEXED, summary, tokenize='porter ascii');

-- RETRIEVAL-070: FTS5 full-text index over verbatim fragments
-- Enables keyword search through raw conversation text (INGEST-038 captures).
-- user_content: USER portion only (higher precision for episodic queries)
-- full_content: Full [USER]+[ASSISTANT] text (broader recall)
CREATE VIRTUAL TABLE IF NOT EXISTS fts_verbatim
    USING fts5(fragment_id UNINDEXED, concept_id UNINDEXED,
               user_content, full_content,
               tokenize='porter ascii');

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
    platform_hint TEXT NOT NULL DEFAULT 'unknown',
    origin_id TEXT DEFAULT NULL,
    pressure_score REAL DEFAULT NULL,
    last_learning_at TEXT DEFAULT NULL,
    last_heartbeat TEXT DEFAULT NULL,
    working_context_json TEXT DEFAULT NULL
);

-- Execution checkpoints (ephemeral resumption state, NOT concepts)
CREATE TABLE IF NOT EXISTS checkpoints (
    task_id TEXT PRIMARY KEY,
    session_id TEXT,
    origin_id TEXT,
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
CREATE INDEX IF NOT EXISTS idx_checkpoints_origin_updated
    ON checkpoints(origin_id, updated_at) WHERE origin_id IS NOT NULL;

-- Origin-aware sync state for idempotent checkpoint replay
CREATE TABLE IF NOT EXISTS checkpoint_sync_state (
    task_id TEXT NOT NULL,
    origin_id TEXT NOT NULL,
    last_op_id INTEGER NOT NULL,
    payload_hash TEXT,
    checkpoint_updated_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (task_id, origin_id)
);

CREATE INDEX IF NOT EXISTS idx_checkpoint_sync_updated ON checkpoint_sync_state(updated_at);


-- Durable replay state for idempotent consumer write requests
CREATE TABLE IF NOT EXISTS write_request_replays (
    endpoint TEXT NOT NULL,
    profile TEXT NOT NULL,
    request_id TEXT NOT NULL,
    status TEXT NOT NULL,
    response_json TEXT,
    request_json TEXT DEFAULT NULL,
    attempt_count INTEGER DEFAULT 0,
    last_error TEXT DEFAULT NULL,
    lease_owner TEXT DEFAULT NULL,
    lease_expires_at TEXT DEFAULT NULL,
    next_retry_at TEXT DEFAULT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (endpoint, profile, request_id)
);

CREATE INDEX IF NOT EXISTS idx_write_request_replays_updated ON write_request_replays(updated_at);
CREATE INDEX IF NOT EXISTS idx_write_request_replays_endpoint_status_updated
ON write_request_replays(endpoint, profile, status, updated_at);

-- Durable lifecycle job queue for bounded post-response learning/enrichment.
CREATE TABLE IF NOT EXISTS lifecycle_jobs (
    job_id TEXT PRIMARY KEY,
    profile TEXT NOT NULL,
    source TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 50,
    stage TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    result_json TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    lease_owner TEXT,
    lease_expires_at TEXT,
    next_retry_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(profile, source, idempotency_key)
);

CREATE INDEX IF NOT EXISTS idx_lifecycle_jobs_ready
ON lifecycle_jobs(profile, status, priority, updated_at);

CREATE INDEX IF NOT EXISTS idx_lifecycle_jobs_stage_status
ON lifecycle_jobs(profile, stage, status, updated_at);

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
CREATE INDEX IF NOT EXISTS idx_concepts_ka_current_created ON concepts(knowledge_area, is_current, created_at);
CREATE INDEX IF NOT EXISTS idx_concepts_current_ka_created ON concepts(is_current, knowledge_area, created_at);
CREATE INDEX IF NOT EXISTS idx_concepts_utility ON concepts(utility_score) WHERE is_current = 1;

-- RETRIEVAL-080: L1 retrieval utility feedback ledger
CREATE TABLE IF NOT EXISTS retrieval_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    concept_id TEXT NOT NULL,
    activation_rank INTEGER NOT NULL,
    relevance_score REAL,
    utilization_score REAL NOT NULL,
    utilization_class TEXT NOT NULL,
    keyword_overlap REAL DEFAULT 0.0,
    ka_match INTEGER DEFAULT 0,
    id_reference INTEGER DEFAULT 0,
    position_signal REAL DEFAULT 0.0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(session_id, turn_number, concept_id)
);
CREATE INDEX IF NOT EXISTS idx_rf_concept ON retrieval_feedback(concept_id);
CREATE INDEX IF NOT EXISTS idx_rf_session ON retrieval_feedback(session_id);
CREATE INDEX IF NOT EXISTS idx_rf_util_class ON retrieval_feedback(utilization_class);

CREATE INDEX IF NOT EXISTS idx_assoc_source ON associations(source);
CREATE INDEX IF NOT EXISTS idx_assoc_target ON associations(target);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
CREATE INDEX IF NOT EXISTS idx_sessions_heartbeat ON sessions(last_heartbeat) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_sessions_origin_status ON sessions(origin_id, status, started_at) WHERE origin_id IS NOT NULL;

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

CREATE TABLE IF NOT EXISTS thread_reorg_batches (
    batch_id TEXT PRIMARY KEY,
    source_thread_id TEXT NOT NULL,
    target_mode TEXT NOT NULL,
    status TEXT NOT NULL,
    planned_count INTEGER NOT NULL DEFAULT 0,
    committed_count INTEGER NOT NULL DEFAULT 0,
    detached_count INTEGER NOT NULL DEFAULT 0,
    recall14_before REAL,
    recall14_after REAL,
    precision14_before REAL,
    precision14_after REAL,
    control_recall14_before REAL,
    control_recall14_after REAL,
    control_precision14_before REAL,
    control_precision14_after REAL,
    evaluation_set_id TEXT,
    notes_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    committed_at TEXT,
    rolled_back_at TEXT
);

CREATE TABLE IF NOT EXISTS thread_reorg_batch_members (
    batch_id TEXT NOT NULL,
    concept_id TEXT NOT NULL,
    target_thread_id TEXT NOT NULL,
    action TEXT NOT NULL,
    status TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'member',
    legacy_membership_before INTEGER NOT NULL DEFAULT 1,
    rationale_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    applied_at TEXT,
    rolled_back_at TEXT,
    PRIMARY KEY (batch_id, concept_id, target_thread_id, action)
);

CREATE INDEX IF NOT EXISTS idx_thread_reorg_batches_status
    ON thread_reorg_batches(status);
CREATE INDEX IF NOT EXISTS idx_thread_reorg_members_batch
    ON thread_reorg_batch_members(batch_id);
CREATE INDEX IF NOT EXISTS idx_thread_reorg_members_concept
    ON thread_reorg_batch_members(concept_id);

CREATE TABLE IF NOT EXISTS thread_reorg_seed_candidates (
    concept_id TEXT PRIMARY KEY,
    source_session_id TEXT,
    source_trace_id TEXT,
    knowledge_area TEXT,
    status TEXT NOT NULL DEFAULT 'queued',
    reason TEXT NOT NULL,
    notes_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    promoted_thread_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_thread_reorg_seed_status
    ON thread_reorg_seed_candidates(status);

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

-- INGEST-060-P1: Full raw turn capture ledger.
-- Disabled by default at runtime; schema exists so diagnostics and rollback stay simple.
CREATE TABLE IF NOT EXISTS raw_turn_payloads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    source TEXT NOT NULL CHECK(source IN ('conversation_turn', 'session_end')),
    user_message TEXT,
    assistant_response TEXT,
    message_len INTEGER NOT NULL DEFAULT 0,
    response_len INTEGER NOT NULL DEFAULT 0,
    content_hash TEXT NOT NULL,
    captured_at TEXT NOT NULL,
    retention_days INTEGER NOT NULL DEFAULT 30,
    purged_at TEXT,
    purge_reason TEXT,
    UNIQUE(session_id, turn_id, source, content_hash)
);

CREATE TABLE IF NOT EXISTS turn_ingestion_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    source TEXT NOT NULL CHECK(source IN ('conversation_turn', 'session_end')),
    raw_payload_id INTEGER,
    capture_status TEXT NOT NULL CHECK(capture_status IN ('captured', 'duplicate', 'failed')),
    learning_status TEXT NOT NULL DEFAULT 'not_started'
        CHECK(learning_status IN ('not_started', 'attempted', 'skipped', 'failed')),
    concepts_extracted INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(raw_payload_id) REFERENCES raw_turn_payloads(id)
);

CREATE INDEX IF NOT EXISTS idx_raw_turn_payloads_session
    ON raw_turn_payloads(session_id, captured_at);
CREATE INDEX IF NOT EXISTS idx_raw_turn_payloads_purge
    ON raw_turn_payloads(captured_at, purged_at);
CREATE INDEX IF NOT EXISTS idx_turn_ingestion_ledger_session
    ON turn_ingestion_ledger(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_turn_ingestion_ledger_status
    ON turn_ingestion_ledger(capture_status, learning_status, updated_at);

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

CREATE TABLE IF NOT EXISTS metrics_rollups (
    bucket_start TEXT NOT NULL,
    bucket_days INTEGER NOT NULL DEFAULT 1,
    metric TEXT NOT NULL,
    count INTEGER NOT NULL,
    sum_value REAL NOT NULL,
    min_value REAL NOT NULL,
    max_value REAL NOT NULL,
    avg_value REAL NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (bucket_start, bucket_days, metric)
);
CREATE INDEX IF NOT EXISTS idx_metrics_rollups_metric_bucket
    ON metrics_rollups(metric, bucket_start);

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
    expires_at TEXT NOT NULL,
    topic_keywords TEXT DEFAULT ''
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

-- SKILL-DEPLOY-001: Synthesis cycle tracking
CREATE TABLE IF NOT EXISTS synthesis_tracking (
    cycle_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    clusters_evaluated INTEGER DEFAULT 0,
    clusters_meaningful INTEGER DEFAULT 0,
    concepts_created INTEGER DEFAULT 0,
    concepts_deduplicated INTEGER DEFAULT 0,
    llm_calls INTEGER DEFAULT 0,
    llm_failures INTEGER DEFAULT 0,
    total_ms REAL DEFAULT 0.0,
    details JSON DEFAULT NULL
);
CREATE INDEX IF NOT EXISTS idx_synthesis_tracking_created ON synthesis_tracking(created_at);
"""

# Configuration — profile-aware data directory and DB path
logger = logging.getLogger(__name__)

# Configuration — profile-aware data directory and DB path
from app.core.profile import resolve_data_dir, resolve_db_path

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
