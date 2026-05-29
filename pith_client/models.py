"""Pydantic response models for the Pith API.

All models use ``extra="allow"`` so new server-side fields
are preserved without requiring a client upgrade.
"""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class _PithModel(BaseModel):
    """Base with forward-compatible config."""
    model_config = {"extra": "allow", "populate_by_name": True}


# ── Tier 1: Core ──────────────────────────────────────────────


class SessionResponse(_PithModel):
    """Response from session_start / session_end."""
    session_id: str = ""
    status: str = ""
    message: str = ""
    reflection: Optional[dict[str, Any]] = None


class VerbatimFragment(_PithModel):
    """A verbatim fragment attached to a concept."""
    id: str = ""
    concept_id: str = ""
    fragment_type: str = ""
    content: str = ""
    char_count: int = 0
    created_at: Optional[str] = None


class Concept(_PithModel):
    """A single knowledge concept."""
    concept_id: str = ""
    summary: str = ""
    confidence: float = 0.0
    relevance_score: float = 0.0
    knowledge_area: str = ""
    key_evidence: list[str] = Field(default_factory=list)
    associations: list[str] = Field(default_factory=list)
    age_minutes: Optional[float] = None
    freshness_label: Optional[str] = None
    currency_status: Optional[str] = None
    verbatim_fragments: list[VerbatimFragment] = Field(default_factory=list)


class Constraint(_PithModel):
    """A constraint from the constraint assembly phase."""
    concept_id: str = ""
    constraint: str = ""
    authority: float = 0.0
    anti_terms: list[str] = Field(default_factory=list)
    presentation_mode: str = ""


class ConstraintSet(_PithModel):
    """Constraint set returned in conversation turns."""
    constraints: list[Constraint] = Field(default_factory=list)
    constraint_count: int = 0


class ConversationTurnResponse(_PithModel):
    """Full response from conversation_turn."""
    activated_concepts: list[Concept] = Field(default_factory=list)
    activation_count: int = 0
    bind_status: Optional[str] = None
    binding_source: Optional[str] = None
    resolved_session_id: Optional[str] = None
    orientation_summary: Optional[str] = None
    constraint_set: Optional[ConstraintSet] = None
    checkpoint_suggested: bool = False
    checkpoint_reason: Optional[str] = None
    is_resumption: bool = False
    is_first_call: bool = False
    extraction_request: Optional[list[str]] = None
    processing_time_ms: float = 0.0


class LearnResponse(_PithModel):
    """Response from session_learn."""
    status: str = ""
    concepts_extracted: int = 0
    concepts_stored: int = 0
    session_id: str = ""


class SearchResult(_PithModel):
    """Response from pith_search.

    Server returns ``{"results": [...], "ambient_context": {...}}``.
    """
    results: list[Concept] = Field(default_factory=list)
    ambient_context: dict[str, Any] = Field(default_factory=dict)


class StatsResponse(_PithModel):
    """Response from pith_stats."""
    total_concepts: int = 0
    knowledge_areas: dict[str, int] = Field(default_factory=dict)
    avg_confidence: float = 0.0
    graph_density: float = 0.0


class HealthResponse(_PithModel):
    """Response from /pith_health — cognitive health analysis.

    The server returns a rich dict from ``reflection_engine.analyze_stability()``
    plus model_stats and federation_status. Common fields are declared;
    additional fields are captured via ``extra="allow"``.
    """
    status: str = ""
    stability_score: Optional[float] = None
    total_concepts: Optional[int] = None
    knowledge_areas: Optional[dict[str, Any]] = None


class OrientResponse(_PithModel):
    """Response from pith_orient."""
    summary: str = ""
    active_goals: list[str] = Field(default_factory=list)
    session_id: Optional[str] = None


class CheckpointResponse(_PithModel):
    """Response from checkpoint save/load/touch/complete/list."""
    status: str = ""
    save_count: int = 0
    task_id: str = ""
    description: str = ""


class SessionInfo(_PithModel):
    """A single session entry from sessions_list."""
    session_id: str = Field(default="", alias="id")
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    status: str = ""
    turn_count: int = 0
    learning_events: int = 0
    learning_event_count: int = 0
    context_hint: Optional[str] = None
    last_learning_at: Optional[str] = None


class SessionsListResponse(_PithModel):
    """Historical wrapper response from sessions_list."""
    sessions: list[SessionInfo] = Field(default_factory=list)
    total: int = 0


class ConceptWriteResponse(_PithModel):
    """Write-status envelope for concept create/evolve operations."""
    status: str = ""
    concept_id: str = ""
    version: str = ""
    previous_version: Optional[str] = None
    message: str = ""
    associations_created: Optional[int] = None
    ambient_context: dict[str, Any] = Field(default_factory=dict)


# ── Tier 2: Extended ──────────────────────────────────────────


class LinkResponse(_PithModel):
    """Response from link_concepts.

    Server returns: ``{"status": "linked", "concept_a": ...,
    "concept_b": ..., "relation": ..., "message": ...}``
    """
    status: str = ""
    concept_a: str = ""
    concept_b: str = ""
    relation: str = ""
    message: str = ""


class Question(_PithModel):
    """An uncertain-knowledge question."""
    concept_id: str = ""
    summary: str = ""
    confidence: float = 0.0
    knowledge_area: str = ""


class ValidationResult(_PithModel):
    """Response from validate_response."""
    valid: bool = True
    violations: list[dict[str, Any]] = Field(default_factory=list)
    constraints_checked: int = 0


class BeliefDiffResponse(_PithModel):
    """Response from belief_diff."""
    diffs: list[dict[str, Any]] = Field(default_factory=list)
    total_changes: int = 0


class ImportResponse(_PithModel):
    """Response from import_conversation."""
    status: str = ""
    concepts_imported: int = Field(default=0, alias="concepts_created")
    chunks_processed: int = 0


class CKO(_PithModel):
    """A Compound Knowledge Object."""
    id: str = ""
    title: str = ""
    synthesis: str = ""
    knowledge_area: str = ""
    cko_type: str = ""
    concept_ids: list[str] = Field(default_factory=list)
    status: str = ""
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class CKOListResponse(_PithModel):
    """Response from cko_list / cko_search."""
    items: list[CKO] = Field(default_factory=list)
    total: int = 0


class ThreadsResponse(_PithModel):
    """Response from pith_threads."""
    threads: list[dict[str, Any]] = Field(default_factory=list)


class TracesResponse(_PithModel):
    """Response from pith_traces."""
    traces: list[dict[str, Any]] = Field(default_factory=list)


class LearningMetricsResponse(_PithModel):
    """Response from learning_metrics."""
    extraction_rate: float = 0.0
    concepts_per_turn: float = 0.0
    learning_debt: int = 0


# ── Tier 3: Platform ─────────────────────────────────────────


class MetricsDashboard(_PithModel):
    """Response from metrics/dashboard — the Critical 8."""
    metrics: dict[str, Any] = Field(default_factory=dict)
    timestamp: Optional[str] = None
    period_hours: float = 24.0


class BackgroundTasksResponse(_PithModel):
    """Response from metrics/bg_tasks."""
    tasks: list[dict[str, Any]] = Field(default_factory=list)
    running: int = 0
    queued: int = 0


class MetricsSummaryResponse(_PithModel):
    """Response from metrics/summary."""
    summary: dict[str, Any] = Field(default_factory=dict)
    period_days: int = 7


class HealthTrendResponse(_PithModel):
    """Response from metrics/health_trend."""
    trend: list[dict[str, Any]] = Field(default_factory=list)
    period_days: int = 7


class BenchmarkResponse(_PithModel):
    """Response from pith/benchmark."""
    status: str = ""
    results: dict[str, Any] = Field(default_factory=dict)


class MigrationResponse(_PithModel):
    """Response from migrate_epistemic_networks."""
    status: str = ""
    migrated: int = 0
    skipped: int = 0
