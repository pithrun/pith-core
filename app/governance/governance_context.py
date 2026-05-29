"""Governance Context — unified state flowing through conversation_turn pipeline.

Implements GovernanceContext dataclass that flows S2 -> S2.5 -> S4 -> S4.8 -> S5 -> S6,
accumulating governance decisions across phases. Includes latency watchdog and
governance event logging.

Created for Wave 1 of Cognitive Governance Architecture.
"""

import logging
import os
import time
import time as _time_mod
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.core.config import BENCHMARK_READONLY
from app.core.constants import (
    GOV_EVENT_EPISTEMIC_CLASSIFICATION,
    GOV_EVENT_GOVERNANCE_CONTEXT_CREATED,
    GOV_EVENT_LATENCY_DEGRADATION,
    GOV_EVENT_LATENCY_WARNING,
    GOV_EVENT_WRITE_CONTEXT_CREATED,
)
from app.core.datetime_utils import _utc_now_iso

logger = logging.getLogger(__name__)

# A3 (v1.2): Soft→Hard enforcement config. Set True after observability period.
GOVERNANCE_HARD_ENFORCEMENT = os.getenv("GOVERNANCE_HARD_ENFORCEMENT", "true").lower() == "true"


class PhasePriority(Enum):
    """Pipeline phase priority for latency watchdog."""

    REQUIRED = "required"  # S2 (retrieval), S4 (budget) — always execute
    OPTIONAL = "optional"  # S2.5, S4.8, S5, S6 — skip under pressure


@dataclass
class IngestionValidationEvent:
    """Write-path validation event — logged when a concept is validated at ingestion.

    Memory Integrity Spec v1.2 §5.8.4 (H18).
    """

    concept_id: str
    validation_result: str  # "PASS" | "SOFT_REJECT" | "HARD_REJECT"
    reason: str
    contradiction_score: float = 0.0
    tier_used: int = 0  # 1, 2, or 3 (0 = no tier used)
    latency_ms: float = 0.0
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = _utc_now_iso()


@dataclass
class PolicyDecisionEvent:
    """Policy engine decision event — logged when PolicyEngine evaluates a concept.

    Memory Integrity Spec v1.2 §5.8.4 (H18).
    """

    policy_name: str
    concept_id: str
    decision: str  # "BLOCK" | "WARN" | "PASS"
    severity: str
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = _utc_now_iso()


@dataclass
class EpistemicClassificationEvent:
    """Epistemic classification event — logged when a concept is classified by network.

    Memory Integrity Spec v1.2 §5.8.4 (H18).
    """

    concept_id: str
    network: str  # e.g. "world_facts", "personal_preferences", "decisions"
    verification_status: str  # e.g. "verified", "unverified", "contested"
    raw_authority: float = 0.0
    effective_authority: float = 0.0  # After epistemic cap
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = _utc_now_iso()


@dataclass
class GovernanceEvent:
    """Single governance decision logged during a turn."""

    event_type: str  # e.g. "retrieval_complete", "authority_boost", "latency_degradation"
    concept_id: str | None  # Which concept this event concerns (None for system events)
    details: dict[str, Any]  # Arbitrary event payload
    timestamp: str = ""  # ISO format
    latency_remaining_ms: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = _utc_now_iso()


@dataclass
class ScoredConcept:
    """A concept with its governance-enhanced score."""

    concept_id: str
    raw_embedding_score: float = 0.0
    authority_score: float = 0.0
    currency_score: float = 0.0
    confidence: float = 0.0
    stability: float = 0.0
    context_boost: float = 0.0
    goal_boost: float = 0.0
    final_score: float = 0.0
    presentation_mode: str = "BACKGROUND"  # CONSTRAINT / DIRECTIVE / CONTEXT / BACKGROUND
    knowledge_area: str = "unknown"  # For budget query-relevance filtering
    concept_type: str = "unknown"  # For budget GUARANTEED tier concept_type gate (Config fix)


@dataclass
class SuppressionEvent:
    """Record of a concept being suppressed from results."""

    concept_id: str
    reason: str  # e.g. "currency_below_threshold", "retracted", "circuit_breaker"
    score_at_suppression: float = 0.0


@dataclass
class ContradictionPair:
    """Two concepts that contradict each other."""

    concept_a_id: str
    concept_b_id: str
    similarity_score: float = 0.0
    resolution: str | None = None  # "a_preferred", "b_preferred", "both_contested", None


def _get_governance_budget() -> float:
    """MAINT-021: Load total governance latency budget from config.

    Returns GOVERNANCE_TOTAL_LATENCY_BUDGET_MS from app.core.config, falling back
    to 2000.0 if config is unavailable (import error or missing attribute).
    Distinct from config.LATENCY_BUDGET_TOTAL_MS (90.0, per-phase watchdog).
    """
    try:
        from app.core.config import GOVERNANCE_TOTAL_LATENCY_BUDGET_MS

        return float(GOVERNANCE_TOTAL_LATENCY_BUDGET_MS)
    except (ImportError, AttributeError):
        return 2000.0


@dataclass
class GovernanceContext:
    """Unified governance state flowing through the entire conversation_turn pipeline.

    Constructed once at pipeline start, passed to every governance phase,
    accumulated across S2 -> S2.5 -> S4 -> S4.8 -> S5 -> S6.
    Persisted to governance_events table at turn end.
    """

    # --- Populated at S2 (retrieval + authority/currency scoring) ---
    scored_concepts: list[ScoredConcept] = field(default_factory=list)
    suppressed_concepts: list[SuppressionEvent] = field(default_factory=list)

    # --- Populated at S2.5 (contradiction detection) ---
    contradictions_detected: list[ContradictionPair] = field(default_factory=list)
    contested_concepts: list[str] = field(default_factory=list)

    # --- Populated at S4-S4.8 (injection phases) ---
    graph_walk_additions: list[str] = field(default_factory=list)
    skill_injections: list[str] = field(default_factory=list)

    # --- Populated at S5 (budget assembly) ---
    budget_allocation: dict[str, list[str]] = field(default_factory=dict)  # tier -> concept_ids
    overflow_summaries: list[str] = field(default_factory=list)
    constraint_set: list[dict[str, Any]] = field(default_factory=list)

    # --- Populated at S6 (bootstrap) ---
    bootstrap_constraints_loaded: int = 0
    governance_actions_pending: list[str] = field(default_factory=list)

    # --- Write-path defense events (Phase 1, §5.8.4 H18) ---
    ingestion_events: list[IngestionValidationEvent] = field(default_factory=list)
    policy_events: list[PolicyDecisionEvent] = field(default_factory=list)
    epistemic_events: list[EpistemicClassificationEvent] = field(default_factory=list)

    # --- Metadata ---
    LATENCY_BUDGET_TOTAL_MS: float = (
        _get_governance_budget()
    )  # MAINT-021: config-driven (was hardcoded 2000.0 PERF-016)
    circuit_breaker_tripped: bool = False
    governance_events: list[GovernanceEvent] = field(default_factory=list)
    _start_time_ns: int = field(default_factory=time.perf_counter_ns, repr=False)

    # --- Phase tracking ---
    phases_executed: list[str] = field(default_factory=list)
    phases_skipped: list[str] = field(default_factory=list)

    # --- Phase-internal wall-clock timeouts (EUNOMIA-039 Fix 2) ---
    _phase_start_times: dict[str, float] = field(default_factory=dict, repr=False)
    _phase_timeouts: dict[str, float] = field(default_factory=dict, repr=False)
    phase_timeout_events: list[dict[str, Any]] = field(default_factory=list)

    def log_event(self, event_type: str, concept_id: str | None, details: dict[str, Any]) -> None:
        """Log a governance decision. Every governance action is traced."""
        self.governance_events.append(
            GovernanceEvent(
                event_type=event_type,
                concept_id=concept_id,
                details=details,
                latency_remaining_ms=max(0.0, self.LATENCY_BUDGET_TOTAL_MS - self.elapsed_ms()),
            )
        )

    # --- Phase-internal wall-clock timeouts (EUNOMIA-039 Fix 2) ---

    def start_phase_timer(self, phase_name: str, timeout_ms: float) -> None:
        """Begin a wall-clock timer for a pipeline phase.

        Called at phase entry. check_phase_timeout() uses these values
        to determine if the phase has exceeded its budget.
        """
        self._phase_start_times[phase_name] = _time_mod.perf_counter()
        self._phase_timeouts[phase_name] = timeout_ms

    def check_phase_timeout(self, phase_name: str) -> bool:
        """Check if a phase has exceeded its wall-clock timeout.

        Returns True if timed out (caller should abort the phase).
        Returns False if within budget or if no timer was started.
        Logs a PHASE_TIMEOUT governance event on first timeout detection.
        """
        start = self._phase_start_times.get(phase_name)
        timeout = self._phase_timeouts.get(phase_name)
        if start is None or timeout is None:
            return False
        elapsed_ms = (_time_mod.perf_counter() - start) * 1000
        if elapsed_ms > timeout:
            event = {
                "phase": phase_name,
                "elapsed_ms": round(elapsed_ms, 1),
                "timeout_ms": timeout,
                "action": "timeout_abort",
            }
            self.phase_timeout_events.append(event)
            self.log_event("PHASE_TIMEOUT", None, event)
            return True
        return False

    # --- Typed event loggers (§5.8.4 H18) ---

    def log_ingestion_validation(
        self,
        concept_id: str,
        validation_result: str,
        reason: str,
        contradiction_score: float = 0.0,
        tier_used: int = 0,
        latency_ms: float = 0.0,
    ) -> None:
        """Log an ingestion validation event and mirror to generic event stream."""
        evt = IngestionValidationEvent(
            concept_id=concept_id,
            validation_result=validation_result,
            reason=reason,
            contradiction_score=contradiction_score,
            tier_used=tier_used,
            latency_ms=latency_ms,
        )
        self.ingestion_events.append(evt)
        self.log_event(
            f"INGESTION_{validation_result}",
            concept_id,
            {
                "reason": reason,
                "contradiction_score": contradiction_score,
                "tier_used": tier_used,
                "latency_ms": latency_ms,
            },
        )

    def log_policy_decision(
        self,
        policy_name: str,
        concept_id: str,
        decision: str,
        severity: str,
    ) -> None:
        """Log a policy engine decision event and mirror to generic event stream."""
        evt = PolicyDecisionEvent(
            policy_name=policy_name,
            concept_id=concept_id,
            decision=decision,
            severity=severity,
        )
        self.policy_events.append(evt)
        self.log_event(
            f"POLICY_{decision}",
            concept_id,
            {"policy_name": policy_name, "severity": severity},
        )

    def log_epistemic_classification(
        self,
        concept_id: str,
        network: str,
        verification_status: str,
        raw_authority: float = 0.0,
        effective_authority: float = 0.0,
    ) -> None:
        """Log an epistemic classification event and mirror to generic event stream."""
        evt = EpistemicClassificationEvent(
            concept_id=concept_id,
            network=network,
            verification_status=verification_status,
            raw_authority=raw_authority,
            effective_authority=effective_authority,
        )
        self.epistemic_events.append(evt)
        self.log_event(
            GOV_EVENT_EPISTEMIC_CLASSIFICATION,
            concept_id,
            {
                "network": network,
                "verification_status": verification_status,
                "raw_authority": raw_authority,
                "effective_authority": effective_authority,
            },
        )

    def flush_events_to_db(self, conn, session_id: str | None = None) -> int:
        """Persist governance events to the governance_events DB table.

        Called at end of conversation_turn. Without this, events only exist
        in memory and benchmarks that query the DB always see 0 events.

        Args:
            conn: Database connection.
            session_id: CTX-008 — session attribution for analytics queries.

        Returns count of events flushed.
        """
        if BENCHMARK_READONLY:
            return 0
        if not self.governance_events:
            return 0
        import json

        flushed = 0
        for evt in self.governance_events:
            try:
                conn.execute(
                    """INSERT INTO governance_events
                       (session_id, event_type, concept_id, details,
                        latency_remaining_ms, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        session_id,
                        evt.event_type,
                        evt.concept_id,
                        json.dumps(evt.details) if evt.details else None,
                        evt.latency_remaining_ms,
                        evt.timestamp,
                    ),
                )
                flushed += 1
            except Exception:
                pass  # Don't let event persistence break the pipeline
        if flushed:
            conn.commit()
        return flushed

    def check_latency_budget(
        self, phase_name: str, phase_cost_ms: float, priority: PhasePriority = PhasePriority.OPTIONAL
    ) -> bool:
        """Wall-clock latency watchdog (PERF-016).

        Compares actual elapsed time against LATENCY_BUDGET_TOTAL_MS.
        Required phases always execute. Optional phases skip when elapsed > budget.
        phase_cost_ms is retained for logging/estimation but not used for decisions.
        """
        try:
            elapsed = self.elapsed_ms()
        except Exception:
            self.phases_executed.append(phase_name)
            return True  # Fail-open: allow phase if timing is broken

        if priority == PhasePriority.REQUIRED:
            self.phases_executed.append(phase_name)
            if elapsed > self.LATENCY_BUDGET_TOTAL_MS:
                self.log_event(
                    GOV_EVENT_LATENCY_WARNING,
                    None,
                    {
                        "phase": phase_name,
                        "elapsed_ms": round(elapsed, 1),
                        "budget_ms": self.LATENCY_BUDGET_TOTAL_MS,
                        "message": "Required phase executed over budget",
                    },
                )
                logger.warning(
                    "Latency watchdog: required phase %s at %.1fms (budget: %.1fms)",
                    phase_name,
                    elapsed,
                    self.LATENCY_BUDGET_TOTAL_MS,
                )
            return True

        # Optional phase — skip if we've exceeded budget
        if elapsed > self.LATENCY_BUDGET_TOTAL_MS:
            self.phases_skipped.append(phase_name)
            self.log_event(
                GOV_EVENT_LATENCY_DEGRADATION,
                None,
                {
                    "phase": phase_name,
                    "elapsed_ms": round(elapsed, 1),
                    "budget_ms": self.LATENCY_BUDGET_TOTAL_MS,
                    "action": "skipped",
                },
            )
            logger.info(
                "Latency watchdog: skipping optional phase %s (%.1fms elapsed, budget: %.1fms)",
                phase_name,
                elapsed,
                self.LATENCY_BUDGET_TOTAL_MS,
            )
            return False

        self.phases_executed.append(phase_name)
        return True

    def elapsed_ms(self) -> float:
        """Total elapsed time since context creation."""
        return (time.perf_counter_ns() - self._start_time_ns) / 1_000_000

    def _set_start_time_offset(self, elapsed_ms: float) -> None:
        """Shift start time so that elapsed_ms() returns ~elapsed_ms immediately.

        Used in tests to simulate elapsed time without real sleeps. Avoids
        direct mutation of the private _start_time_ns field.
        """
        self._start_time_ns = time.perf_counter_ns() - int(elapsed_ms * 1_000_000)

    def finalize(self) -> dict[str, Any]:
        """Produce summary metadata for turn response and event persistence."""
        elapsed = self.elapsed_ms()
        return {
            "total_elapsed_ms": round(elapsed, 2),
            "latency_remaining_ms": round(max(0.0, self.LATENCY_BUDGET_TOTAL_MS - elapsed), 2),
            "circuit_breaker_tripped": self.circuit_breaker_tripped,
            "phases_executed": self.phases_executed,
            "phases_skipped": self.phases_skipped,
            "events_logged": len(self.governance_events),
            "ingestion_validations": len(self.ingestion_events),
            "policy_decisions": len(self.policy_events),
            "epistemic_classifications": len(self.epistemic_events),
            "concepts_scored": len(self.scored_concepts),
            "concepts_suppressed": len(self.suppressed_concepts),
            "contradictions_found": len(self.contradictions_detected),
            "governance_actions_pending": len(self.governance_actions_pending),
        }


def create_governance_context(
    circuit_breaker_tripped: bool = False,
) -> GovernanceContext:
    """Factory for GovernanceContext. Called once at start of conversation_turn.

    Invariant I-6: GovernanceContext is constructed exactly once per turn.
    """
    ctx = GovernanceContext(
        circuit_breaker_tripped=circuit_breaker_tripped,
    )
    ctx.log_event(
        GOV_EVENT_GOVERNANCE_CONTEXT_CREATED,
        None,
        {
            "latency_budget_ms": GovernanceContext.LATENCY_BUDGET_TOTAL_MS,
            "circuit_breaker_tripped": circuit_breaker_tripped,
        },
    )
    return ctx


@contextmanager
def write_governance_context(operation: str = "write"):
    """Context manager for write-path governance (§5.8.4 H18).

    Creates a GovernanceContext scoped to a single write operation
    (propose_concept, evolve_concept, session_learn). Automatically
    flushes events to DB on exit.

    Usage:
        with write_governance_context("propose_concept") as gov_ctx:
            # ... do write-path work, passing gov_ctx to internal functions ...
            gov_ctx.log_ingestion_validation(...)

    When GOVERNANCE_EVENT_WIRING_ENABLED is False, yields a no-op context
    that silently discards events (zero overhead).
    """
    from app.core.config import FEATURE_FLAGS

    if not FEATURE_FLAGS.get("GOVERNANCE_EVENT_WIRING_ENABLED", False):
        yield None
        return

    ctx = GovernanceContext()
    ctx.log_event(
        GOV_EVENT_WRITE_CONTEXT_CREATED,
        None,
        {"operation": operation},
    )
    try:
        yield ctx
    finally:
        # Flush events to DB — non-fatal if it fails
        try:
            from app.storage import _db

            with _db() as conn:
                ctx.flush_events_to_db(conn)
        except Exception:
            logger.warning(
                "write_governance_context: failed to flush %d events for %s",
                len(ctx.governance_events),
                operation,
            )
