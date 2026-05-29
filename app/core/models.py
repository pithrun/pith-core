"""Data models for Pith.

Schema v2.1 — Extended with cognitive layer fields
Phase 0: Schema & Graph Foundation
Phase 1A: Salience, Maturity, Cognitive Self-Awareness

Changes from v2:
- Salience: salience, salience_source, salience_set_at, salience_reason
- Maturity: maturity, quarantine_entered, maturity_promoted_at, maturity_promotion_evidence
- Constants: SALIENCE_SOURCES, MATURITY_LEVELS

Changes from v1:
- Evidence: structured objects replacing List[str] with provenance and strength scoring
- Concept: concept_type, scope_conditions, failure_modes (classification + boundaries)
- Concept: change_type, change_reason, model_signature, updated_at (evolution metadata)
- Concept: content_hash, parent_hash (cryptographic lineage)
- Hypothesis: hypothesis_id, competing_with (competing hypothesis tracking)
- Association: evidence_refs, created_at, last_validated (extended link metadata)

All new fields have defaults — backward compatible with v1/v2 YAML.
"""

import json
import re
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from app.core.datetime_utils import _utc_now_iso

ORIGIN_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


# --- CKPT-003: Checkpoint Field TTL Classification ---


class CheckpointTTLTier(str, Enum):
    """Durability classification for checkpoint fields.

    DURABLE: Decisions, outcomes — never expire (kept across compressions)
    MEDIUM: Task state, context — 48h useful life, summarizable
    PERISHABLE: Line numbers, error messages — session-scoped, strip on session end
    """

    DURABLE = "durable"
    MEDIUM = "medium"
    PERISHABLE = "perishable"


# Field-to-tier mapping for checkpoint content
# Used by Sprint 2 CKPT-002 compression to decide what to keep/summarize/strip
CHECKPOINT_FIELD_TTL = {
    "task_id": CheckpointTTLTier.DURABLE,
    "description": CheckpointTTLTier.DURABLE,
    "done": CheckpointTTLTier.DURABLE,  # Decision log — permanent value
    "status": CheckpointTTLTier.DURABLE,
    "active": CheckpointTTLTier.MEDIUM,  # Current work item — stale after session
    "next": CheckpointTTLTier.MEDIUM,  # Planned items — useful for ~48h
    "blockers": CheckpointTTLTier.MEDIUM,  # May resolve externally
    "context": CheckpointTTLTier.PERISHABLE,  # Line numbers, error snippets, temp state
    "concept_refs": CheckpointTTLTier.MEDIUM,  # Concept IDs — useful while concepts exist
}


# --- Structured Evidence with Provenance and Strength Scoring ---


class Evidence(BaseModel):
    """Structured evidence object with provenance tracking and strength scoring.

    Evidence is the epistemic foundation. Without structured evidence,
    confidence scores are meaningless.
    """

    model_config = {"protected_namespaces": ()}

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_type: str = "conversation"  # conversation, document, observation, inference, external_data
    content: str = ""  # Content excerpt from the source
    source_reference: str | None = None  # URL, doc ID, session ref
    timestamp: str = Field(default_factory=lambda: _utc_now_iso())
    model_origin: str | None = None  # Which model produced this evidence

    # Evidence strength scoring factors
    # E(e) = SourceReliability × Directness × Consistency × Corroboration × Recency
    reliability_weight: float = (
        0.7  # Source weights: external_data=0.9, documented=0.85, conversation=0.7, inference=0.6
    )
    directness: float = 0.8  # How directly this evidence supports the concept
    consistency: float = 0.8  # DEPRECATED (MAINT-009): Not used in formula. Kept for backward compat.
    corroboration_count: int = 0  # Number of independent supporting sources
    age_days: float = 0.0  # For recency decay calculation

    # P0.2: Extraction source tracking
    extraction_source: str = "heuristic"  # "heuristic" | "client" | "adapter:{name}"
    corroboration_type: str | None = None  # "same_source" | "cross_source" | None

    # Source-anchoring for drift detection (DATA-041)
    file_path: str | None = None  # Relative path from repo root (e.g., "app/storage.py")
    commit_hash: str | None = None  # Short git hash at evidence creation time
    line_range: str | None = None  # Line range referenced (e.g., "33-80")
    verified_at: str | None = None  # ISO timestamp of last source verification

    @classmethod
    def from_source(
        cls,
        content: str,
        source_type: str = "conversation",
        file_path: str | None = None,
        commit_hash: str | None = None,
        line_range: str | None = None,
        **kwargs,
    ) -> "Evidence":
        """Create evidence with optional source-anchoring metadata (DATA-041).

        Use this factory when source location is known (e.g., adapter-extracted
        evidence from code files, git commits, or documentation).
        """
        return cls(
            content=content,
            source_type=source_type,
            file_path=file_path,
            commit_hash=commit_hash,
            line_range=line_range,
            **kwargs,
        )


class ContextTelemetry(BaseModel):
    """Structured client telemetry for context pressure decisions.

    Fields are intentionally permissive. The conversation_turn merge helper,
    not request-model validation, decides whether malformed telemetry is safe
    to reject and fall back from.
    """

    schema_version: str = "1.0"
    pressure_ratio: Any = None
    measurement_source: Any = "unknown"
    measurement_confidence: Any = "low"
    measurement_scope: Any = "unknown"
    used_tokens: Any = None
    window_size_tokens: Any = None
    source_metadata: dict[str, Any] | None = None


class WorkspaceFinding(BaseModel):
    """Client-reported workspace audit finding."""

    code: str = ""
    severity: str = ""
    message: str = ""


class WorkspaceContext(BaseModel):
    """Client-reported local workspace classification for operational policy."""

    repo_root: str = ""
    current_path: str = ""
    classification: str = "unknown"
    current_branch: str = ""
    branch_owner: str = ""
    active_worktree_count: int = 0
    findings: list[WorkspaceFinding] = []


# --- Competing Hypothesis Structure ---


class Hypothesis(BaseModel):
    """Competing hypothesis within a concept.

    Hypotheses coexist — never overwritten.
    """

    hypothesis_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = []  # Legacy: string evidence refs
    evidence_refs: list[str] = []  # References to Evidence object IDs
    competing_with: list[str] = []  # hypothesis_ids this competes against


# --- Core Concept Structure ---

# Valid concept_type values — Knowledge Hierarchy (6 levels)
# L1-L2: Grounded knowledge (word-grounded garbage detection)
# L3-L6: Abstract knowledge (coherence + provenance garbage detection)
CONCEPT_TYPES = [
    # --- L1: Observations (what happened) ---
    "observation",  # Direct observation or stated fact
    "pattern",  # Recurring pattern identified across observations
    # --- L2: Decisions (what we chose) ---
    "decision",  # Explicit choice with rationale
    "goal",  # Desired outcome or objective
    "constraint",  # Limitation or boundary condition
    "hypothesis",  # Unverified proposition
    # --- L3: Principles (reusable rules) ---
    "principle",  # General rule or guideline, broadly applicable
    # --- L4: Methods (how to work) ---
    "method",  # Reusable process or workflow for achieving outcomes
    "process",  # Sequence of steps (legacy compat, equivalent to method)
    # --- L5: Heuristics & Cognitive Strategies (meta-reasoning) ---
    "heuristic",  # Rule of thumb for when to apply what
    "cognitive_strategy",  # Meta-reasoning pattern — HOW to think about problems
    # --- L6: Cognitive Pattern Recognition (universal learning) ---
    "system_model",  # Model of how a system/entity reasons or behaves
    # --- Legacy compat (not in hierarchy but in DB) ---
    "client_extraction",  # Legacy: auto-assigned by old pipeline, treat as observation
    # --- Wave 4b: User Preferences ---
    "preference",  # L1.5: User-stated behavioral preference (grounded type)
]

# TIER2-DAY1: Set for O(1) membership checks and leakage detection in /learning_metrics
CONCEPT_TYPES_SET = set(CONCEPT_TYPES)

# Types that use ABSTRACT garbage detection (coherence + provenance, no word grounding)
ABSTRACT_CONCEPT_TYPES = {"principle", "method", "heuristic", "cognitive_strategy", "system_model"}

# Types that use GROUNDED garbage detection (word overlap with source text)
GROUNDED_CONCEPT_TYPES = {
    "observation",
    "pattern",
    "decision",
    "goal",
    "constraint",
    "hypothesis",
    "process",
    "client_extraction",
    "preference",
}

# Valid change_type values for evolution metadata
CHANGE_TYPES = [
    "creation",  # Initial creation
    "refinement",  # Improved understanding
    "generalization",  # Broader applicability
    "specialization",  # Narrower, more specific
    "contradiction_flag",  # Conflicting evidence found
    "merge",  # Combined with another concept
    "split",  # Divided into sub-concepts
]

# Valid salience_source values
SALIENCE_SOURCES = [
    "user",  # User explicitly assigned
    "system",  # System-inferred from metrics
    "goal",  # Inherited from active goal
    "thread",  # Inherited from active thread
    "decay",  # Set by decay process
]

# Valid maturity levels for concept lifecycle
# Memory Integrity Spec v1.2, §5.1.1: Added DISCARDED for quarantine auto-expiry/rejection
MATURITY_LEVELS = [
    "QUARANTINED",  # Newly proposed, not yet validated
    "PROVISIONAL",  # Passed initial gates, in graph but flagged
    "ESTABLISHED",  # Fully integrated, passed corroboration requirements
    "DISCARDED",  # Failed quarantine or explicitly rejected (kept for audit)
]

# Maturity levels that should be EXCLUDED from all retrieval paths
# DISCARDED concepts are kept for forensic audit but never served
MATURITY_EXCLUDE_FROM_RETRIEVAL = {"DISCARDED"}


class Concept(BaseModel):
    """Core concept structure — canonical schema.

    Concepts are append-evolved, never overwritten.
    All new fields have defaults for backward compatibility with v1 YAML.
    """

    model_config = {"protected_namespaces": ()}
    # --- Identity ---
    id: str
    version: str
    created_at: str
    supersedes: str | None = None
    superseded_by: str | None = None  # MAINT-030: ID of concept that replaced this one

    @field_validator("supersedes", mode="before")
    @classmethod
    def coerce_supersedes(cls, v: Any) -> str | None:
        """[DATA-051] Coerce list → str|None to fix legacy array storage bug."""
        if isinstance(v, list):
            return v[0] if v else None
        return v

    # --- Cognitive Classification ---
    concept_type: str = "observation"  # Must be in CONCEPT_TYPES

    # --- Core Knowledge Payload ---
    summary: str  # ≤500 chars (enforced in validation, not here for migration safety)
    evidence: list[Any] = []  # List[Evidence] or List[str] — migration handles conversion
    signals: list[str] = []
    associations: list[str] = []
    hypotheses: list[Hypothesis] = []

    # --- Scope & Failure Boundaries ---
    scope_conditions: str | None = None  # Where this concept applies
    failure_modes: str | None = None  # When this concept breaks

    # --- Evolution Metadata ---
    change_type: str = "creation"  # Must be in CHANGE_TYPES
    change_reason: str | None = None  # Why this version exists
    model_signature: str | None = None  # Which model/adapter produced this version
    updated_at: str | None = None  # Last modification time

    # TEMPORAL-002: Date the knowledge actually refers to (vs created_at = ingestion time).
    # E.g. "moved to SF in March 2025" → original_date = "2025-03". ISO-8601 partial.
    original_date: str | None = None
    valid_from: str | None = None
    content_updated_at: str | None = None

    # RETRIEVAL-104: Edit-chain provenance for entity chain filtering.
    # JSON array of question_ids (benchmark) or session_ids (production).
    # NULL = universal provenance (original fact, serves all chains).
    edit_provenance: str | None = None

    # EUNOMIA-040 Fix 3: Pre-computed subject key for indexed RETRIEVAL-072 dedup.
    # Populated by _extract_subject_key(summary) at learn-time.
    subject_key: str | None = None

    # --- Confidence & Stability ---
    confidence: float = Field(ge=0.0, le=1.0)
    stability: float = Field(ge=0.0, le=1.0, default=0.5)

    # --- Salience: Personal importance, independent of confidence ---
    salience: float = Field(ge=0.0, le=1.0, default=0.5)
    salience_source: str = "system"  # user | system | goal | thread | decay
    salience_set_at: str | None = None  # ISO8601 — when salience was last set
    salience_reason: str | None = None  # Why this salience level was assigned

    # --- Concept Maturity Lifecycle ---
    maturity: str = "ESTABLISHED"  # QUARANTINED | PROVISIONAL | ESTABLISHED | DISCARDED | PROVISIONAL | ESTABLISHED
    quarantine_entered: str | None = None  # ISO8601 — when concept entered current maturity
    maturity_promoted_at: str | None = None  # ISO8601 — when last promoted
    maturity_promotion_evidence: str | None = None  # What triggered promotion

    # --- Concept Status (DATA-018) ---
    status: str = "active"  # active | archived | superseded | corrupted

    # --- Cryptographic Lineage ---
    content_hash: str | None = None  # SHA-256 of canonical content
    parent_hash: str | None = None  # Hash of parent version

    # --- Activation & Access ---
    last_accessed: str | None = None
    last_organic_access: str | None = None
    access_count: int = 0
    reinforcement_count: int = 0  # DATA-017: Cumulative retrieval reinforcements
    embedding_version: int = 1

    # --- Governance (pre-computed cached scores, populated by migration GOV-001) ---
    authority_score: float | None = None  # Epistemic authority [0.0-1.0], None = not yet computed
    currency_score: float | None = None  # Currency (still-true likelihood) [0.0-1.0]
    currency_status: str = "ACTIVE"  # ACTIVE / STALE / SUPERSEDED / CONTESTED / RESOLVED
    staleness_state: str | None = None  # COGGOV-014: AGING / REVIEW / None
    staleness_score: float | None = None  # COGGOV-014: stale-risk score [0.0-1.0]
    staleness_reason: str | None = None  # COGGOV-014: JSON-encoded scoring rationale
    staleness_evaluated_at: str | None = None  # COGGOV-014: last detector evaluation time
    staleness_detector_version: str | None = None  # COGGOV-014: detector build/version tag
    staleness_consecutive_hits: int = 0  # COGGOV-014: consecutive aging-band detections
    effective_authority: float | None = None  # DATA-015: Combined authority (evidence + governance)
    ka_relative_authority: float | None = None  # Federation Phase 0: KA-relative percentile rank

    # --- Wave 4b: Epistemic Intelligence + Traces + PWT ---
    has_correction: bool = False  # [FIX F3] Set by record_correction()
    source_trace_id: str | None = None  # [X2] Reverse linkage to creating trace
    session_id: str | None = None  # AGENT-004: Direct session linkage for federation
    conflicting_preferences: list[str] = []  # [N2] Preference conflict tracking
    provenance_migrated: bool = True  # [FIX F1] False for pre-4b concepts

    # --- Knowledge Domain ---
    knowledge_area: str = "general"  # Domain/topic area (e.g., 'architecture', 'debugging')

    # --- Epistemic Classification (Retrieval Defense §3.1) ---
    epistemic_network: str | None = None  # "world_fact" | "preference" | "assessment" | None (unclassified)
    verification_status: str | None = None  # "verified" | "unverified" | "stale" | "contradicted" | None
    verification_fraction: float = 0.0  # 0.0 (fully unverified) to 1.0 (fully verified)

    # --- Protection (COGGOV-005: Kumiho-inspired safety guards) ---
    protected: bool = False  # Immune from automated governance (contradiction suppression, etc.)

    # --- RETRIEVAL-080: Feedback Loop Utility ---
    utility_score: float | None = None  # Rolling utility estimate [0.1-0.9], None = not yet populated
    utility_samples: int = 0  # Number of feedback events accumulated
    utility_updated: str | None = None  # Last update timestamp

    # --- Extensible metadata ---
    metadata: dict[str, Any] = {}


# --- Verbatim Fragment Model (INGEST-037) ---


class VerbatimFragment(BaseModel):
    """A raw text fragment preserved alongside the semantic concept."""

    fragment_type: str = "text"  # text, code, formula, table, pointer
    content: str | None = None  # Raw text (None for pointers)
    pointer_uri: str | None = None  # External reference URI
    pointer_meta: dict | None = None  # Retrieval parameters


# --- Request/Response Models (unchanged interface) ---


class ConceptProposal(BaseModel):
    """Proposal for new concept."""

    concept_id: str
    summary: str
    knowledge_area: str
    evidence: list[str] = []
    signals: list[str] = []
    associations: list[str] = []
    confidence: float = Field(ge=0.0, le=1.0)
    hypotheses: list[Hypothesis] = []
    concept_type: str = "observation"
    always_activate: bool = False  # P1-1: Inject into every conversation_turn
    agent_id: str = "default"  # AGENT-001: Multi-agent scoping
    original_date: str | None = None  # TEMPORAL-003: ISO-8601 partial date when fact originated
    verbatim_fragments: list[VerbatimFragment] = []  # INGEST-037: Raw text preservation

    @field_validator("concept_type")
    @classmethod
    def validate_concept_type(cls, v):
        if v not in CONCEPT_TYPES:
            return "observation"
        return v


class ConceptEvolution(BaseModel):
    """Proposal to evolve existing concept."""

    concept_id: str
    new_summary: str | None = None
    new_evidence: list[str | dict] = []
    new_signals: list[str] = []
    new_associations: list[str] = []
    new_hypotheses: list[Hypothesis] = []
    confidence_change: float = 0.0
    new_concept_type: str | None = None
    new_metadata: dict[str, Any] = {}
    always_activate: bool | None = None  # P1-1: Set/unset always-activate flag
    session_id: str | None = None  # CASCADE-001 A1.2: For reinforcement independence check
    raw_evidence_count: int = 0  # CASCADE-001 A1.5: Insight-level evidence source count for cascade trigger

    @field_validator("new_concept_type")
    @classmethod
    def validate_new_concept_type(cls, v):
        if v is not None and v not in CONCEPT_TYPES:
            return None  # Ignore invalid reclassification
        return v


class KnowledgeArea(BaseModel):
    """Collection of related concepts."""

    id: str
    created_at: str
    description: str
    concepts: list[str] = []
    metadata: dict[str, Any] = {}


# --- Extended Association Schema ---


class Association(BaseModel):
    """Directional link between concepts with evidence support."""

    concept_a: str
    concept_b: str
    relation: str  # See RELATION_TYPES below
    strength: float = Field(ge=0.0, le=1.0, default=0.5)
    evidence_refs: list[str] = []  # Evidence IDs supporting this link
    created_at: str = Field(default_factory=lambda: _utc_now_iso())
    last_validated: str | None = None  # Last time this link was confirmed


# Valid relation types for concept associations
RELATION_TYPES = [
    "related_to",  # General association
    "supports",  # A provides evidence for B
    "contradicts",  # A conflicts with B
    "part_of",  # A is a component of B
    "derived_from",  # A was derived from B
    "causes",  # A causes B
    "enables",  # A enables B
    "constrains",  # A limits B
    "specializes",  # A is a specific case of B
    "generalizes",  # A is a broader form of B
]


# --- Search & Query Models (unchanged interface) ---


class SearchQuery(BaseModel):
    """Search request."""

    query: str
    context: str | None = None
    goal: str | None = None
    max_results: int = 5
    min_confidence: float = 0.0
    # Federation Phase 0, Component 0.3: KA-aware query routing
    ka_boost: list[str] | None = None
    ka_boost_weight: float = Field(default=0.2, ge=0.0, le=0.5)  # DEBT-197: constrain [0,0.5]


class SearchResult(BaseModel):
    """Search result item."""

    concept_id: str
    version: str
    summary: str
    confidence: float
    relevance_score: float
    knowledge_area: str | None = None
    ka_relative_authority: float | None = None  # Federation Phase 0 (A7)
    maturity: str | None = None  # MATURITY-001: for API-level filtering
    created_at: str | None = None  # RETRIEVAL-053: for recency boost
    edit_provenance: str | None = None  # RETRIEVAL-104: for chain filter
    metadata: dict[str, Any] | None = Field(default=None, exclude=True)


class Question(BaseModel):
    """Curiosity-generated question."""

    concept_id: str
    question: str
    priority: float
    created_at: str
    reasons: list[str] = []


class ReflectionSummary(BaseModel):
    """Summary of reflection cycle."""

    concepts_consolidated: int
    concepts_decayed: int
    concepts_recalibrated: int = 0  # Overconfidence correction count
    concepts_archived: int = 0  # Forgetting mechanism output
    associations_updated: int
    questions_generated: int
    timestamp: str
    phase_timings: dict[str, float] = {}  # DEBT-008: sub-step durations in ms
    gc_queue_remaining: int = 0  # DEBT-005: concepts deferred after batch cap
    concepts_graduated: int = 0  # RETRIEVAL-003: quarantine graduation count
    concepts_discarded_quarantine: int = 0  # RETRIEVAL-003: quarantine auto-discard count
    concepts_time_matured: int = 0  # STABILITY-001 C: passive time-based maturation count
    concepts_assoc_propagated: int = 0  # STABILITY-001 D: association confidence propagation count
    concepts_auto_associated: int = 0  # HEALTH-009: new edges created by auto-association in reflection
    concepts_currency_recomputed: int = 0  # RETRIEVAL-015: full-pop currency recompute count
    concepts_promoted: int = 0  # STABILITY-024: PROVISIONAL→ESTABLISHED promotion sweep count

    # MEASURE-011: Per-factor evidence strength CVs (coefficient of variation)
    evidence_cv_composite: float | None = None
    evidence_cv_reliability: float | None = None
    evidence_cv_directness: float | None = None
    evidence_cv_consistency: float | None = None
    evidence_cv_corroboration: float | None = None
    evidence_cv_recency: float | None = None
    aborted: bool = False  # MAINT-040: cooperative budget/cancel abort surfaced to callers
    abort_reason: str | None = None
    last_completed_step: str | None = None
    abort_stage: str | None = None


class PithStats(BaseModel):
    """Overall pith statistics."""

    total_concepts: int
    total_versions: int
    avg_confidence: float
    avg_stability: float
    knowledge_areas: int
    associations: int
    pending_questions: int


# ============================================================
# SelfModel Schema — Cognitive Self-Assessment
# Phase 1A D5: Cognitive self-assessment singleton
# ============================================================


class ToolCapability(BaseModel):
    """Individual MCP tool capability record."""

    tool_name: str
    operational: bool = True
    performance_grade: float = 0.5  # [0.0-1.0] from benchmarks
    known_limitations: list[str] = []
    last_benchmarked: str | None = None  # ISO8601


class CognitiveOperation(BaseModel):
    """Cognitive operation maturity record.
    Maturity enum: experimental | operational | proven (3-level scale).
    """

    operation: str
    maturity: str = "operational"  # experimental | operational | proven
    success_rate: float = 0.0
    failure_patterns: list[str] = []


class CapacityLimits(BaseModel):
    """System capacity measurements."""

    concept_count: int = 0
    concept_ceiling_estimate: int = 10000  # Conservative estimate for CPU-only
    retrieval_latency_ms: float = 0.0
    reflection_duration_ms: float = 0.0


class CognitiveCapabilityInventory(BaseModel):
    """Complete capability inventory: tools, operations, capacity."""

    tool_capabilities: list[ToolCapability] = []
    cognitive_operations: list[CognitiveOperation] = []
    capacity_limits: CapacityLimits = CapacityLimits()


class KnowledgeAreaProfile(BaseModel):
    """Per-area knowledge distribution metrics."""

    knowledge_area: str
    concept_count: int = 0
    avg_confidence: float = 0.0
    avg_stability: float = 0.0
    evidence_coverage: float = 0.0  # % with structured evidence
    last_activity: str | None = None  # ISO8601


class PredictionRecord(BaseModel):
    """Prediction tracking for calibration (Wave 4b)."""

    concept_id: str
    confidence_at_retrieval: float
    retrieved_at: str
    session_id: str
    outcome: str = "pending"  # pending|confirmed|revised|corrected|stale
    outcome_at: str | None = None
    outcome_source: str = "evolution"  # evolution|correction|reflection|stale_timeout


class CalibrationBin(BaseModel):
    """Binned calibration data for ECE computation (Wave 4b)."""

    bin_lower: float
    bin_upper: float
    prediction_count: int = 0
    revision_count: int = 0
    avg_predicted: float = 0.0
    avg_actual: float = 0.0
    gap: float = 0.0


class ConfidenceBreakdown(BaseModel):
    """3-way confidence cap breakdown (Wave 4b PWT) [FIX O1]."""

    raw: float
    test_ceiling: float
    provenance_score: float
    effective: float
    binding_cap: str  # "raw"|"test_status"|"provenance"


class ConfidenceCalibration(BaseModel):
    """Confidence accuracy assessment — Wave 4b real implementation."""

    total_predictions_logged: int = 0
    predictions_with_outcomes: int = 0
    calibration_bins: list[CalibrationBin] = []
    expected_calibration_error: float = 0.0
    ece_computable: bool = False  # [FIX CS1] False until >50 predictions
    test_status_distribution: dict[str, int] = {}
    calibration_maturity: str = "insufficient_testing"
    overconfidence_areas: list[str] = []
    underconfidence_areas: list[str] = []
    calibration_method: str = "prediction_tracking_v1"
    last_calibrated: str | None = None


class BlindSpot(BaseModel):
    """Identified knowledge gap."""

    description: str
    severity: str = "minor"  # minor | moderate | critical
    detected_by: str = "reflection"  # curiosity_engine | reflection | user_flag
    detected_at: str | None = None


class KnowledgeHealth(BaseModel):
    """Global knowledge health metrics."""

    total_concepts: int = 0
    avg_confidence: float = 0.0
    avg_stability: float = 0.0
    contradiction_density: float = 0.0
    orphan_concept_count: int = 0  # 0 associations
    stale_concept_count: int = 0  # Not accessed in > 30 days
    health_score: float = 0.0  # HEALTH-001: 5-factor weighted score


class EpistemicProfile(BaseModel):
    """What this pith knows and where it's weak."""

    knowledge_distribution: list[KnowledgeAreaProfile] = []
    confidence_calibration: ConfidenceCalibration = ConfidenceCalibration()
    blind_spots: list[BlindSpot] = []
    knowledge_health: KnowledgeHealth = KnowledgeHealth()


class GranularityProfile(BaseModel):
    """Concept granularity tendencies. STUB in Phase 1A."""

    avg_concepts_per_learning_event: float = 0.0
    avg_summary_length: float = 0.0
    tendency: str = "balanced"  # over_granular | balanced | under_granular


class LinkingProfile(BaseModel):
    """Association linking tendencies. STUB in Phase 1A."""

    avg_associations_per_concept: float = 0.0
    cross_domain_link_ratio: float = 0.0
    tendency: str = "balanced"  # over_linked | balanced | isolated


class ConfidenceProfile(BaseModel):
    """Confidence assignment tendencies. STUB in Phase 1A."""

    initial_confidence_avg: float = 0.0
    evolution_confidence_delta_avg: float = 0.0
    tendency: str = "calibrated"  # cautious | calibrated | overconfident


class LearningVelocity(BaseModel):
    """Learning rate tendencies. STUB in Phase 1A."""

    concepts_per_session_avg: float = 0.0
    evolution_events_per_session_avg: float = 0.0
    new_vs_evolve_ratio: float = 0.0


class CognitiveTendencies(BaseModel):
    """How this pith tends to reason. STUB in Phase 1A."""

    granularity_profile: GranularityProfile = GranularityProfile()
    linking_profile: LinkingProfile = LinkingProfile()
    confidence_profile: ConfidenceProfile = ConfidenceProfile()
    learning_velocity: LearningVelocity = LearningVelocity()


class CorrectionRecord(BaseModel):
    """Individual error correction record. STUB in Phase 1A."""

    error_id: str
    concept_id: str
    error_type: str = "factual"  # factual | structural | confidence_miscalibration | scope_error
    description: str = ""
    corrected_at: str | None = None
    corrected_by: str = ""


class RecurringPattern(BaseModel):
    """Recurring error pattern. STUB in Phase 1A."""

    pattern_id: str
    description: str
    frequency: int = 0
    severity: str = "low"  # low | medium | high
    mitigation: str = ""


class ErrorHistory(BaseModel):
    """Correction and error tracking. STUB in Phase 1A."""

    corrections: list[CorrectionRecord] = []
    recurring_patterns: list[RecurringPattern] = []
    total_corrections: int = 0
    correction_rate: float = 0.0  # Corrections per 100 concepts
    most_common_error_type: str = "none"


class ModelHistoryEntry(BaseModel):
    """Record of a model that has connected to this pith."""

    model_config = {"protected_namespaces": ()}

    model_signature: str
    first_session: str | None = None
    last_session: str | None = None
    session_count: int = 0
    concepts_contributed: int = 0
    evolutions_contributed: int = 0


class CognitiveMilestone(BaseModel):
    """Notable pith achievement."""

    milestone_id: str
    description: str
    achieved_at: str | None = None
    significance: str = "minor"  # minor | moderate | major


class IdentityContinuity(BaseModel):
    """What persists across model swaps — Pith's continuous identity."""

    model_config = {"protected_namespaces": ()}

    pith_created_at: str | None = None  # ISO8601 — earliest concept
    total_sessions: int = 0
    total_learning_events: int = 0  # Lifetime concept creations + evolutions
    model_history: list[ModelHistoryEntry] = []
    cognitive_milestones: list[CognitiveMilestone] = []
    current_strategic_focus: str = ""
    current_maturity_stage: str = "developing"  # nascent | developing | operational | mature


class SelfModel(BaseModel):
    """Singleton cognitive self-assessment meta-object.

    Not stored alongside regular concepts — lives in data/self_model/.
    Append-only versioned with cryptographic lineage (content_hash chain).
    """

    model_config = {"protected_namespaces": ()}

    model_id: str = "self"  # Singleton identifier
    version: int = 1  # Incremented on each update
    updated_at: str | None = None  # ISO8601
    updated_by: str = "pith_deterministic"  # Phase 1A: no model adapter

    capabilities: CognitiveCapabilityInventory = CognitiveCapabilityInventory()
    epistemic_profile: EpistemicProfile = EpistemicProfile()
    tendencies: CognitiveTendencies = CognitiveTendencies()
    error_history: ErrorHistory = ErrorHistory()
    identity: IdentityContinuity = IdentityContinuity()

    content_hash: str = ""  # SHA-256 of canonical content
    parent_hash: str = ""  # Hash of prior version


class IntrospectIdentity(BaseModel):
    """Introspect summary mode: identity slice."""

    pith_age_days: int = 0
    total_sessions: int = 0
    current_maturity: str = "developing"
    current_focus: str = ""


class IntrospectHealth(BaseModel):
    """Introspect summary mode: health slice."""

    concept_count: int = 0
    avg_confidence: float = 0.0
    contradiction_density: float = 0.0


class IntrospectSummary(BaseModel):
    """Introspect summary mode response.

    Note: 'weakest_areas' is a Phase 1A fallback for 'top_gaps'.
    Uses lowest-confidence areas since blind_spots is stubbed.
    Will switch to 'top_gaps' with real blind spot data in Phase 1B.
    """

    identity: IntrospectIdentity = IntrospectIdentity()
    health: IntrospectHealth = IntrospectHealth()
    top_strengths: list[str] = []  # Top 3 knowledge areas by confidence
    weakest_areas: list[str] = []  # Phase 1A fallback for top_gaps
    recent_errors: list[str] = []  # Last 3 corrections (STUB: empty)


# ============================================================
# Present Moment Orientation — Session Context
# Phase 1A D7: Session Middleware
# ============================================================


class RecentConceptSummary(BaseModel):
    """Recently created concept summary for orientation context."""

    concept_id: str
    summary: str  # First 100 chars
    knowledge_area: str = "unknown"
    created_at: str | None = None


class ConceptEvolutionRecord(BaseModel):
    """Recently evolved concept record for orientation context."""

    concept_id: str
    summary: str
    change_type: str = ""
    change_reason: str = ""
    evolved_at: str | None = None


class ConceptDecayRecord(BaseModel):
    """Recently decayed concept record for orientation context."""

    concept_id: str
    summary: str
    previous_confidence: float = 0.0
    current_confidence: float = 0.0


class RecentEvolutionSummary(BaseModel):
    """Where-been summary: what changed recently."""

    time_window: str = "7_days"
    concepts_created: list[RecentConceptSummary] = []
    concepts_evolved: list[ConceptEvolutionRecord] = []
    concepts_decayed: list[ConceptDecayRecord] = []  # STUB — decay doesn't annotate change_type yet
    contradictions_detected: list[dict] = []  # STUB Phase 1A
    corrections_made: list[dict] = []  # STUB Phase 1A
    session_count_in_window: int = 0  # STUB Phase 1A
    total_learning_events_in_window: int = 0


class AreaStrength(BaseModel):
    """Knowledge area strength/weakness entry for orientation."""

    knowledge_area: str
    concept_count: int = 0
    avg_confidence: float = 0.0
    reason: str = ""


class ActiveUncertainty(BaseModel):
    """Uncertain concept entry for orientation."""

    concept_id: str
    summary: str
    confidence: float = 0.0
    uncertainty_type: str = "low_confidence"


class PendingQuestionSummary(BaseModel):
    """Pending question from curiosity engine for orientation."""

    question: str
    concept_id: str = ""
    priority: float = 0.0


class CognitiveVelocity(BaseModel):
    """Self-awareness: how fast Pith is growing and changing."""

    sessions_in_window: int = 0
    concepts_created_in_window: int = 0
    concepts_evolved_in_window: int = 0
    learning_events_in_window: int = 0
    avg_concepts_per_session: float = 0.0
    avg_learning_events_per_session: float = 0.0
    knowledge_growth_rate: float = 0.0  # concepts per day
    trend: str = "insufficient_data"  # accelerating | steady | decelerating | insufficient_data
    trend_detail: str = ""  # Human-readable explanation


class CurrentStateAssessment(BaseModel):
    """Where-am assessment: current knowledge state."""

    knowledge_health: dict = {}  # Reused from SelfModel epistemic
    strongest_areas: list[AreaStrength] = []
    weakest_areas: list[AreaStrength] = []
    active_uncertainties: list[ActiveUncertainty] = []
    pending_questions: list[PendingQuestionSummary] = []
    cognitive_velocity: CognitiveVelocity = CognitiveVelocity()  # Self-awareness


class GoalSummary(BaseModel):
    """Active goal entry for orientation."""

    goal_id: str
    summary: str
    priority: float = 0.5
    progress_indicator: str = "in_progress"
    linked_concepts: list[str] = []


class CuriosityFrontierItem(BaseModel):
    """Curiosity frontier entry — unexplored knowledge edge."""

    gap_description: str
    priority_score: float = 0.0


class StrategicPriority(BaseModel):
    """A strategic priority synthesized from high-confidence decisions."""

    concept_id: str
    summary: str
    confidence: float = 0.5
    source_type: str = "decision"  # decision, pattern, principle


class RecommendedAction(BaseModel):
    """A recommended next action synthesized from knowledge gaps or priorities."""

    description: str
    rationale: str
    priority: float = 0.5


class ActiveDirectionality(BaseModel):
    """Where-going directionality: active goals, priorities, and frontiers."""

    active_goals: list[GoalSummary] = []
    strategic_priorities: list[StrategicPriority] = []
    curiosity_frontier: list[CuriosityFrontierItem] = []
    next_recommended_actions: list[RecommendedAction] = []


class PresentMomentOrientation(BaseModel):
    """Complete orientation payload: where-been, where-am, where-going."""

    generated_at: str | None = None
    generated_by: str = "pith_deterministic"
    where_been: RecentEvolutionSummary = RecentEvolutionSummary()
    where_am: CurrentStateAssessment = CurrentStateAssessment()
    where_going: ActiveDirectionality = ActiveDirectionality()
    open_threads: list[dict] = []  # Wave 5: narrative thread summaries
    experiment_summary: dict[str, Any] | None = None  # Wave 6: active experiments summary
    workstreams: dict[str, Any] | None = None  # OPS-517: explicit orientation status only
    orientation_hash: str = ""


class SessionInfo(BaseModel):
    """Session lifecycle tracking."""

    session_id: str
    started_at: str | None = None
    ended_at: str | None = None
    status: str = "active"  # active | ended
    context_hint: str = ""
    learning_event_count: int = 0  # Tracks propose/evolve calls
    last_learning_at: str | None = None  # ISO timestamp of most recent learning event
    concepts_created: int = 0  # Self-awareness: concepts created this session
    concepts_evolved: int = 0  # Self-awareness: concepts evolved this session
    agent_id: str = "default"  # AGENT-001: Multi-agent scoping
    model_id: str = "unknown"  # FEDERATION L1.5: model provenance
    platform_hint: str = "unknown"  # SESSION-012 v0.3: caller provenance
    origin_id: str | None = None  # SESSION-015: stable client/thread closeout binding
    # Auto-reflection state (T2 bookmarks)
    reflection_bookmarks: list[dict] = []  # Accumulated T2 bookmarks
    reflection_turn_counter: int = 0  # Turns since last T2 check
    concepts_since_last_bookmark: list[str] = []  # Concept IDs since last bookmark


class SessionStartResponse(BaseModel):
    """Response from session_start — bootstrap payload."""

    session: SessionInfo
    introspect_summary: dict = {}
    orientation: dict = {}


# --- P1.2: conversation_turn + session_learn Models ---


class ConversationTurnRequest(BaseModel):
    """Pre-response context activation + auto-learning request.

    Given the current conversation context, finds and activates the most
    relevant existing knowledge so the AI can use it in its response.

    If previous_response is provided, the server auto-learns from the
    previous exchange (previous message + previous response) BEFORE doing
    retrieval. This eliminates the need for a separate session_learn call,
    closing the learning feedback loop structurally.

    If extracted_concepts_json is provided alongside previous_response,
    Tier 2 (client-extracted) concepts are merged with Tier 1 (heuristic)
    extraction for higher-quality learning.
    """

    message: str
    request_id: str | None = None
    conversation_context: str = ""
    session_id: str | None = None
    max_concepts: int = Field(default=8, ge=1, le=200)  # TEST-088: floor=1 (0 returns empty silently), ceil=200 (RETRIEVAL-096 combo needs 60)
    include_predictions: bool = False
    # Auto-learning: if provided, learn from previous exchange before retrieval
    previous_response: str | None = None
    previous_message: str | None = None
    # Tier 2: client-extracted concepts from previous exchange (JSON string)
    extracted_concepts_json: str | None = None
    # Tier 2: client classification hint (bypasses regex classifier)
    classification_hint: str | None = None
    # CTX Phase 4: Client-side compaction signal (future-proofing)
    compaction_detected: bool | None = None
    # CTX-004 future-proof: client-reported context utilization (0.0-1.0)
    context_pressure: float | None = None
    # CTX-TELEMETRY-001: structured, model-agnostic context telemetry
    context_telemetry: ContextTelemetry | None = None
    # AGENT-001: Multi-agent scoping
    agent_id: str = "default"
    # AGENT-002: Scoped retrieval ('agent' = own knowledge only, 'global' = all)
    scope: str = "global"
    # RETRIEVAL-056: Include deprecated (STALE/SUPERSEDED/CONTRADICTED) concepts in results
    include_deprecated: bool = False
    # VERBATIM-SURFACE Fix 2: Surface verbatim fragments by default.
    # Concept relevance scoring acts as the quality gate (validated by LongMemEval).
    include_verbatim: bool = True
    # FEDERATION L1.5: Model provenance tracking
    model_id: str = "unknown"
    platform_hint: str = "unknown"  # SESSION-012 v0.3: client platform (cowork, claude-code, etc.)
    transport_mode: str | None = None  # SESSION-012 binding safety: route header plumbing
    workspace_context: WorkspaceContext | None = None
    # SESSION-014: stable client/thread binding for checkpoint authority.
    origin_id: str | None = None
    current_task_id: str | None = None
    context_authority_mode: str = "balanced"

    @field_validator("model_id", mode="before")
    @classmethod
    def validate_model_id(cls, v):
        if v is None or (isinstance(v, str) and v.strip() == ""):
            return "unknown"
        if isinstance(v, str):
            return v.strip()[:200]
        return "unknown"

    @field_validator("origin_id", mode="before")
    @classmethod
    def validate_origin_id(cls, v):
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("origin_id must be a string")
        v = v.strip()
        if not v:
            return None
        if not ORIGIN_ID_RE.fullmatch(v):
            raise ValueError("origin_id must match ^[A-Za-z0-9._:-]{1,128}$")
        return v

    @field_validator("current_task_id", mode="before")
    @classmethod
    def normalize_current_task_id(cls, v):
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("current_task_id must be a string")
        v = v.strip()
        return v[:200] if v else None

    @field_validator("context_authority_mode", mode="before")
    @classmethod
    def validate_context_authority_mode(cls, v):
        if v is None or (isinstance(v, str) and not v.strip()):
            return "balanced"
        if not isinstance(v, str):
            raise ValueError("context_authority_mode must be a string")
        mode = v.strip().lower()
        if mode not in {"strict", "balanced", "permissive"}:
            raise ValueError("context_authority_mode must be strict, balanced, or permissive")
        return mode


class ActivatedConcept(BaseModel):
    """Concept activated for conversation context.

    Contains the concept summary, confidence, and relevance score plus
    trimmed evidence (top 2 items) and 1-hop association IDs.
    """

    concept_id: str
    summary: str
    confidence: float
    relevance_score: float
    knowledge_area: str
    key_evidence: list[str] = []  # Top 2 evidence items
    associations: list[str] = []  # Associated concept IDs (1-hop)
    shadow_expanded: bool = False  # True if included via association shadow expansion (S4.1)
    # TEMPORAL_AWARENESS v2.4: Temporal fields computed at serve time
    age_minutes: int | None = None  # Minutes since created_at. Integer avoids float noise.
    freshness_label: str | None = None  # Human-readable: "this session", "18 minutes ago", "yesterday"
    # RETRIEVAL-014 Layer 1e: Supersession fields for downstream inspection
    superseded_by: str | None = None  # ID of concept that replaced this one
    currency_status: str | None = None  # ACTIVE / STALE / SUPERSEDED / CONTESTED / RESOLVED
    staleness_state: str | None = None  # COGGOV-014: AGING / REVIEW
    ka_relative_authority: float | None = None  # Federation Phase 0: KA-relative percentile rank
    serial_order: int | None = None  # RETRIEVAL-037c: DB rowid for conflict resolution ordering
    # LOCOMO q10: DB-backed temporal provenance copied from Concept when available.
    created_at: str | None = None
    valid_from: str | None = None
    content_updated_at: str | None = None
    session_id: str | None = None
    # BENCH-045: Session-local evidence surface annotations
    is_session_local_evidence: bool = False
    evidence_role: str | None = None
    slot_subject: str | None = None
    slot_attribute: str | None = None
    slot_group_id: str | None = None
    grounding_priority: float | None = None
    # TEMPORAL-002: When the knowledge refers to, not when it was learned.
    original_date: str | None = None
    # INGEST-037 Layer 3: Verbatim fragments attached to this concept (opt-in via include_verbatim)
    verbatim_fragments: list[dict] = []
    # RETRIEVAL-104: Edit provenance for chain filter (JSON array of question_ids, NULL=universal)
    edit_provenance: str | None = None
    # Benchmark-only source identity. Optional and absent for normal user concepts;
    # surfaced so private benchmark adapters can evaluate source recall without
    # embedding provenance markers into natural-language summaries.
    beam_source_key: str | None = None
    beam_source_turn_id: str | None = None
    # Some benchmark source corpora use composite turn indexes such as "2,17".
    # Treat this as provenance identity, not a numeric rank.
    beam_source_turn_index: str | int | None = None
    beam_source_batch_idx: int | None = None
    beam_source_role: str | None = None
    beam_role: str | None = None
    # Observe-only branch provenance envelope for diagnostics. Runtime ranking
    # and answer construction must not treat this field as an authority selector.
    branch_provenance: dict[str, Any] | None = None


class TrustSignal(BaseModel):
    """Calibrated uncertainty signal for a concept.

    Communicates WHY a concept might be less trustworthy, not just its score.
    Generated per-concept during context assembly.
    """

    concept_id: str
    authority: float = 0.0
    currency: float = 0.0
    presentation_mode: str = "BACKGROUND"  # CONSTRAINT | DIRECTIVE | CONTEXT | BACKGROUND
    qualifiers: list[str] = []  # EVIDENCE_AGING, LOW_CORROBORATION, CONTESTED, etc.
    trust_explanation: str = ""  # Human-readable explanation
    needs_revalidation: bool = False
    has_contradiction: bool = False
    evidence_count: int = 0
    days_since_last_evidence: int = 0


# Valid trust qualifiers
TRUST_QUALIFIERS = [
    "EVIDENCE_AGING",  # days_since_last_evidence > 30
    "LOW_CORROBORATION",  # evidence_count < 2
    "CONTESTED",  # currency_status == 'CONTESTED'
    "STALE_RISK",  # authority >= 0.60 AND currency < 0.50
    "SINGLE_SOURCE",  # all evidence from same session
    "HIGH_CONFIDENCE",  # authority >= 0.80 AND currency >= 0.80 AND evidence >= 3
]


class ConversationTurnResponse(BaseModel):
    """Pre-response context activation response.

    Returns activated concepts with graph density metric showing
    how connected the knowledge graph is (associations / concepts ratio).
    """

    activated_concepts: list[ActivatedConcept]
    activation_count: int
    bind_status: str | None = None  # SESSION-012: bound / unbound
    binding_source: str | None = None  # SESSION-012: explicit_request / auto_create / in_memory_active / exec_fallback_omitted
    resolved_session_id: str | None = None  # SESSION-012: authoritative session chosen for this turn
    predictions: list[dict] = []
    graph_density: float = 0.0  # associations / concepts ratio
    processing_time_ms: float
    checkpoint_suggested: bool = False  # True when session needs a checkpoint
    checkpoint_reason: str | None = None  # Why checkpoint is suggested
    checkpoint_payload: dict | None = None  # CTX-003: Pre-composed checkpoint for client to save
    staleness_filtered_count: int = 0  # Count of stale concepts silently excluded (S5.5)
    shadow_expanded_count: int = 0  # Count of concepts added via shadow expansion (S4.1)
    is_first_call: bool = False  # S0: True on first conversation_turn in this session
    is_resumption: bool = False  # B5.1: True when prior session existed within 24h
    orientation_summary: str | None = None  # S6: Pre-written orientation for behavioral bootstrap
    greeting_hint: str | None = None  # S6: Behavioral directive for greeting style
    # S7: active_checkpoints REMOVED from conversation_turn response.
    # Principle: "remove the lazy path" — checkpoint data in the response
    # was the #1 parroting target. Agent must explicitly call
    # pith_checkpoint(action='load') when it needs task state.
    # This is a server-side architectural fix, not a nudge.
    auto_learned: dict | None = None  # S-1: Auto-learn result (one-turn delay when BACKGROUND_AUTOLEARN_ENABLED)
    # PERF-FORT-3: Load pressure notification — surfaces degradation to client
    load_pressure: dict | None = None  # {level: normal|elevated|critical, phases_deferred: [...], message: str}
    budget_warnings: list[str] = []  # Proactive flags when any learning budget category is hit/near-limit
    extraction_request: dict | None = None  # B1: Active extraction request for gap filling
    retrospective_nudge: dict | None = None  # RETRO-001: Nudge when L1/L3 ratio is poor
    retroactive_reflection: dict | None = None  # T1: Orphaned session reflection prompts
    reflection_bookmarks: list[dict] | None = None  # T2: In-flight observation hints
    governance_summary: dict | None = None  # GOV: Governance pipeline telemetry (phases, latency, events)
    source_set_trace: dict | None = None  # RETRIEVAL-112: trace-only source-set completeness telemetry
    canary_retrieval_trace: dict | None = None  # RETRIEVAL-113: MH262 canary diagnostic-only retrieval trace
    terminal_conflict_trace: dict | None = None  # MAB: trace-only same-key terminal conflict telemetry
    correction_signals: dict | None = None  # CCL §3c: Compounding correction loop results
    constraint_set: dict | None = None  # GOV-W2.5: Constraint set for this turn
    coverage_confidence: dict | None = None  # FIX1: Coverage quality signal (sparse/absent knowledge)
    coverage_score: float | None = None  # QUALITY-002: Ratio of semantic matches to max requested (0.0-1.0)
    retrieval_budget_trace: dict | None = None  # BENCH: requested/effective max_concepts observability
    grounded_slot_subject: str | None = None  # BENCH-045: Grounded slot subject when local grounding fired
    grounded_slot_attribute: str | None = None  # BENCH-045: Grounded slot attribute when local grounding fired
    grounding_mode: str | None = None  # BENCH-045: direct / synthesized / missing
    grounding_confidence: float | None = None  # BENCH-045: Confidence score for local grounding
    blind_spot_match: dict | None = None  # FIX1b: Blind spot cross-reference match
    # PRODUCT-003: Confidence-gated abstention signal
    abstention_signal: dict | None = None  # {should_abstain, confidence, reason, level}
    checkpoint_resume_available: bool = False  # CKPT-005: True when recent checkpoint can be resumed
    pressure_source_used: str | None = None  # CTX-TELEMETRY-001: heuristic / legacy_context_pressure / structured_context_telemetry
    directives: list[dict] | None = None  # S4.8: Behavioral directives (Tier 2)
    directive_budget_warning: str | None = None  # S4.8: Warning when directives truncated
    activated_domains: list[str] | None = None  # S1.5: Which cognitive domains activated
    # ARCH-001: Model-agnostic skill routing — pith recommends skills to read before responding
    recommended_skills: list[str] = []  # Skill file paths the caller should read before responding
    analogy_suggestions: list[dict] | None = None  # EXP-025: Demand-side cross-KA analogy suggestions
    # STABILITY-012: Factual freshness warnings for stale concrete references
    freshness_warnings: list[dict] | None = None  # Concepts with potentially stale file paths, versions, URLs
    # C1: Engine-side per-hop chain answer (FC benchmark port)
    chain_answer: str | None = None
    # C1: Benchmark-gated chain answer decision diagnostics.
    chain_answer_diagnostics: dict | None = None
    # Resume Context v1.1: Cross-session continuity injection
    resume_context: str | None = None  # RC: Machine-readable resume block for session continuity
    resume_context_tier: str | None = None  # RC: FRESH / RECENT / STALE / NONE
    resume_context_suppressed: bool = False  # RC: True if resume was available but drift-suppressed
    # Context Management Integration: Mid-session context resilience
    context_priority_hints: dict | None = None  # CTX-1: Priority metadata for compaction survival
    compaction_detected: bool = False  # CTX-2: True when server detected mid-session compaction
    # TEMPORAL_AWARENESS v2.4: Absolute temporal reference frame
    # PRICING-003: Upgrade nudge when conversation turn budget is exhausted
    upgrade_nudge: dict | None = None
    # PRICING-007: Recall gap attribution when learning was capped
    recall_gap_attribution: dict | None = None
    server_time_utc: str | None = None  # ISO 8601 timestamp at response generation
    # CONTEXT-001: Structured working context returned every turn
    working_context: dict | None = None  # Unified state block for context continuity
    # WORKSTREAMS-002: Explicit active Workstream context when the turn-context flag is enabled
    active_workstream: dict | None = None
    # Workstreams API parity: compact read-only activation state hint, no context block.
    workstream_activation: dict | None = None
    # SAL V0: Structured activation summary (None when SAL disabled or fallback)
    structured_summary: dict | None = None
    # SAL V1: Formatted context string for LLM consumption (None when below threshold)
    sal_context: str | None = None
    chain_hint: str | None = None  # RETRIEVAL-037d: Reasoning chain hint from multihop decomposition


class SessionLearnRequest(BaseModel):
    """Post-response learning request.

    Given a completed exchange (user message + AI response), extracts
    new knowledge, evolves existing concepts, and builds associations.
    Optionally accepts client-extracted concepts for Tier 2 processing.
    """

    user_message: str
    assistant_response: str
    request_id: str | None = None
    session_id: str | None = None
    knowledge_area: str = "conversation"
    auto_associate: bool = True
    extracted_concepts: list[dict] | None = None  # P0.2: client-extracted concepts
    extracted_concepts_json: str | None = None  # Fallback clients mirror conversation_turn payload shape
    observation_date: str | None = None
    timestamp: int | float | str | None = None
    # AGENT-001: Multi-agent scoping
    agent_id: str = "default"
    # FEDERATION L1.5: Model provenance (forwarded from ConversationTurnRequest)
    model_id: str = "unknown"
    # RETRIEVAL-021: Activation-learning bridge — concept IDs activated during retrieval
    activated_concept_ids: list[str] | None = None
    # SESSION-LEARN-MISMATCH-001: Diagnostic — identifies dispatch origin
    trigger_path: str = "unknown"  # "auto_learn" | "direct_mcp" | "unknown"

    @model_validator(mode="before")
    @classmethod
    def normalize_extracted_concepts_json(cls, data: Any) -> Any:
        """Accept fallback client's canonical extracted_concepts_json shape."""
        if not isinstance(data, dict):
            return data
        if data.get("extracted_concepts") is not None:
            return data
        raw = data.get("extracted_concepts_json")
        if raw in (None, ""):
            return data
        if not isinstance(raw, str):
            raise ValueError("extracted_concepts_json must be a JSON string")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("extracted_concepts_json must be valid JSON") from exc
        if not isinstance(parsed, list):
            raise ValueError("extracted_concepts_json must decode to a JSON array")
        normalized = dict(data)
        normalized["extracted_concepts"] = parsed
        return normalized


class SessionEndRequest(BaseModel):
    """Optional request body for session_end with last-exchange flush.

    When provided, the server auto-learns from the final exchange before
    closing the session. This prevents the last exchange's knowledge from
    being lost (Mechanism C).
    """

    request_id: str | None = None
    session_id: str | None = None
    origin_id: str | None = None
    previous_response: str | None = None
    previous_message: str | None = None
    extracted_concepts_json: str | None = None
    # AGENT-001: Multi-agent scoping
    agent_id: str = "default"

    @field_validator("session_id", mode="before")
    @classmethod
    def normalize_session_id(cls, v):
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("session_id must be a string")
        v = v.strip()
        return v[:200] if v else None

    @field_validator("origin_id", mode="before")
    @classmethod
    def validate_origin_id(cls, v):
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("origin_id must be a string")
        v = v.strip()
        if not v:
            return None
        if not ORIGIN_ID_RE.fullmatch(v):
            raise ValueError("origin_id must match ^[A-Za-z0-9._:-]{1,128}$")
        return v


class LearnedConcept(BaseModel):
    """Newly created concept from session learning.

    All auto-created concepts have PROVISIONAL maturity and confidence
    floored at 0.35 minimum.
    """

    concept_id: str
    summary: str
    confidence: float
    knowledge_area: str
    concept_type: str = "observation"  # Attack 8 fix: expose type for gap analysis


class EvolvedConcept(BaseModel):
    """Existing concept evolved with new evidence from conversation."""

    concept_id: str
    version: str
    change: str


class SessionLearnResponse(BaseModel):
    """Post-response learning response.

    Includes full accounting of what happened: concepts created, evolved,
    duplicates skipped, concepts below confidence floor, and errors.
    """

    concepts_created: list[LearnedConcept]
    concepts_evolved: list[EvolvedConcept]
    associations_created: int
    duplicates_skipped: int
    concepts_skipped: int = 0  # Below confidence floor (0.35)
    errors: int = 0  # Concepts that failed during processing
    processing_time_ms: float
    learning_events: int
    extraction_source_breakdown: dict = {}  # {"heuristic": N, "client": N}
    learning_budget_remaining: int = 999999  # compatibility field; learning is uncapped
    garbage_rejected: int = 0  # concepts that failed garbage detection
    rejection_details: list[dict] = []  # per-concept rejection reasons [{index, reason, summary_preview, stage}]
    budget_warnings: list[str] = []  # proactive flags when any budget category is hit/near-limit
    session_warning: str | None = None  # EC12: warn if no active session or ID mismatch
    concepts_superseded: int = 0  # S3.5: concepts marked [SUPERSEDED] by contradiction detection
    supersession_details: list[dict] = []  # S3.5: details of superseded concepts [{old_id, new_id, reason}]
    persistence_state: str = "committed"
    processing_state: str = "committed"
    request_id: str | None = None
    retry_after_seconds: float | None = None


# --- Auto-Association Models (P1.3) ---


class AutoAssociateBatchRequest(BaseModel):
    """Request body for batch auto-association pipeline."""

    tier1_threshold: float = Field(default=0.18, ge=0.05, le=0.50)  # ARCH-O07: raised from 0.12
    tier2_threshold: float = Field(default=0.06, ge=0.03, le=0.30)
    max_edges_per_concept: int = Field(default=8, ge=1, le=20)
    tier2_enabled: bool = Field(default=True)
    dry_run: bool = Field(default=False)

    @model_validator(mode="after")
    def validate_thresholds(self):
        if self.tier2_enabled and self.tier2_threshold >= self.tier1_threshold:
            raise ValueError("tier2_threshold must be less than tier1_threshold")
        return self


class AutoAssociateBatchResponse(BaseModel):
    """Response from batch auto-association pipeline."""

    index_synced: int
    pairs_evaluated: int
    tier1_edges_created: int
    tier2_edges_created: int
    edges_skipped_existing: int
    edges_skipped_cap: int
    orphans_before: int
    orphans_after: int
    processing_time_ms: float
    dry_run: bool


class AutoAssociateMatch(BaseModel):
    """Single match result from single-concept auto-association."""

    target_id: str
    cosine_score: float
    edge_created: bool


class AutoAssociateSingleRequest(BaseModel):
    """Request body for single-concept auto-association."""

    threshold: float = Field(default=0.12, ge=0.05, le=0.50)
    max_edges: int = Field(default=5, ge=1, le=20)


class AutoAssociateSingleResponse(BaseModel):
    """Response from single-concept auto-association."""

    concept_id: str
    edges_created: int
    edges_skipped_existing: int
    matches: list[AutoAssociateMatch]
    processing_time_ms: float


# ============================================================
# Wave 5 — Narrative Threads
# ============================================================


class ThreadConceptLink(BaseModel):
    """Link between a thread and a concept with role metadata."""

    thread_id: str
    concept_id: str
    role: str = "member"  # member|seed|outcome|blocker|evidence
    added_at: str = Field(default_factory=lambda: _utc_now_iso())
    added_by: str = "system"  # system|user|auto


class WorkstreamDiscoveryState(BaseModel):
    """Discovery eligibility state for Workstream activation."""

    tier: str = "needs_hygiene_review"
    reason_codes: list[str] = []
    source: str = "unset"
    run_id: str | None = None
    last_evaluated_at: str | None = None
    eligible_until: str | None = None
    previous_tier: str | None = None
    promoted_by: str | None = None
    promoted_at: str | None = None
    promotion_reason: str | None = None


class WorkstreamMetadata(BaseModel):
    """User-curated product metadata for an explicit Workstream."""

    kind: str = "workstream"
    current_objective: str = ""
    current_summary: str = ""
    next_action: str = ""
    blockers: list[str] = []
    quality_state: str = "ok"  # ok|needs_review|blocked
    created_by: str = "user"
    updated_by: str = "user"
    parent_workstream_id: str | None = None
    parent_title: str | None = None
    relationship: str | None = None  # child|related
    discovery_state: WorkstreamDiscoveryState | None = None


class NarrativeThread(BaseModel):
    """Ongoing work stream / topic thread with lifecycle management."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str = ""
    status: str = "active"  # active|paused|completed|abandoned
    created_at: str = Field(default_factory=lambda: _utc_now_iso())
    updated_at: str = Field(default_factory=lambda: _utc_now_iso())
    last_activity_at: str = Field(default_factory=lambda: _utc_now_iso())
    last_auto_activity_at: str | None = None  # [FIX T1] Separate auto-activity
    completed_at: str | None = None
    urgency: str = "normal"  # low|normal|high
    agent_id: str = "default"  # FC-MA-5.1

    # Content links
    concept_ids: list[str] = []
    trace_ids: list[str] = []
    goal_ids: list[str] = []
    knowledge_areas: list[str] = []

    # Audit trail [FIX I2]
    status_history: list[dict[str, Any]] = []

    # Metadata
    predecessor_id: str | None = None  # Link to predecessor thread
    workstream: WorkstreamMetadata | None = None


class ThreadSummary(BaseModel):
    """Lightweight thread summary for orientation display."""

    thread_id: str
    title: str
    status: str
    urgency: str = "normal"
    days_since_activity: int = 0
    concept_count: int = 0
    goal_ids: list[str] = []
    staleness_warning: bool = False


# ============================================================
# Wave 4b — Cognitive Traces
# ============================================================


class TraceRecord(BaseModel):
    """Cognitive trace record — structured learning event log (Wave 4b).

    5 structured fields capture the full reasoning arc:
    situation → intent → assessment → justification → reflection.
    Reflection field filled during reflection cycles, not at creation.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    created_at: str = Field(default_factory=lambda: _utc_now_iso())
    trigger_type: str = "learning_event"  # learning_event|correction|reflection|user_assertion

    # 5 structured fields
    situation: str = ""  # What was happening
    intent: str = ""  # What the agent was trying to do
    assessment: str = ""  # What was concluded
    justification: str = ""  # Why that conclusion
    reflection: str = ""  # Post-hoc evaluation (filled during reflection)

    concept_refs: list[str] = []
    agent_id: str = "default"  # FC-MA-4b2: Multi-agent forward compat


# ============================================================
# Wave 6 — Experiment Engine
# ============================================================


class ExperimentCandidate(BaseModel):
    """A candidate for cognitive experimentation."""

    candidate_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    experiment_type: str
    concept_ids: list[str] = []  # concepts involved
    score: float = 0.0  # composite relevance score [0.0, 1.0]
    score_components: dict[str, Any] = {}  # breakdown: {"term_sim": 0.35, ...}
    rationale: str = ""  # human-readable explanation
    metadata: dict[str, Any] = {}  # type-specific data


class ExperimentResult(BaseModel):
    """Result from model-driven experiment processing."""

    synthesis: str = ""  # model-generated synthesis/reasoning text
    confidence: float = 0.0  # model-reported confidence [0.0, 1.0]
    concepts_produced: list[dict[str, Any]] = []  # concept specs to create
    cko_produced: dict[str, Any] | None = None  # optional CKO spec
    reasoning_trace: str = ""  # model's step-by-step reasoning


class Experiment(BaseModel):
    """Full experiment record with lifecycle tracking."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    experiment_type: str
    status: str = "pending"  # pending|candidate_generation|reasoning|completed|archived|insufficient_data
    created_at: str = Field(default_factory=lambda: _utc_now_iso())
    updated_at: str = Field(default_factory=lambda: _utc_now_iso())
    candidates: list[ExperimentCandidate] = []
    result: ExperimentResult | None = None
    concept_ids_produced: list[str] = []
    cko_ids_produced: list[str] = []
    thread_id: str | None = None
    config_snapshot: dict[str, Any] = {}
    generation_time_ms: int | None = None  # [O2 fix]
    processing_time_ms: int | None = None  # [O2 fix]
    metadata: dict[str, Any] = {}
