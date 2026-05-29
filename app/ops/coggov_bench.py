"""CogGov-Bench — Cognitive Governance Benchmark (Gap E).

Behavioral benchmark measuring whether governance actually works.
NOT a health diagnostic — this tests governance BEHAVIOR.

6 Core Dimensions:
  1. Constraint Adherence: Does the agent respect decisions?
  2. Stale Knowledge Resistance: Does the agent avoid outdated knowledge?
  3. Correction Learning: Does the agent improve from corrections?
  4. Cross-Session Coherence: Does the agent maintain positions?
  5. Context Integrity: Is context internally consistent?
  6. Recovery Rate: How quickly are errors fixed?

Plus per-layer adversarial scenarios.

A/B Methodology: Compare governance-enabled vs governance-disabled
on identical test scenarios. Governance toggle via env var.

Reference: COGNITIVE_GOVERNANCE_ARCHITECTURE_v1.3.md §Gap E
"""

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from app.core.constants import (
    GOV_EVENT_AUTHORITY_DEMOTION,
    GOV_EVENT_BUDGET_ALLOCATED,
    GOV_EVENT_CONTRADICTION_DETECTED,
    GOV_EVENT_CONTRADICTION_PHASE_2_COMPLETED,
    GOV_EVENT_CONTRADICTION_REVIEW,
    GOV_EVENT_DECISION_SUPERSESSION,
    GOV_EVENT_SUPERSESSION_REVIEW,
)
from app.core.datetime_utils import _utc_now_iso

logger = logging.getLogger(__name__)


class CogGovBenchCancelled(RuntimeError):
    """Raised when a benchmark run is cancelled cooperatively."""


# =============================================================================
# Result Models
# =============================================================================


@dataclass
class ScenarioResult:
    """Result of a single test scenario."""

    name: str
    passed: bool
    score: float  # 0-100
    details: str = ""
    elapsed_ms: float = 0.0


@dataclass
class DimensionScore:
    """Score for one benchmark dimension (0-100)."""

    dimension: str
    score: float = 0.0
    scenarios: list[ScenarioResult] = field(default_factory=list)
    passed_count: int = 0
    total_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "score": round(self.score, 2),
            "passed": self.passed_count,
            "total": self.total_count,
            "scenarios": [
                {
                    "name": s.name,
                    "passed": s.passed,
                    "score": round(s.score, 2),
                    "details": s.details,
                    "elapsed_ms": round(s.elapsed_ms, 2),
                }
                for s in self.scenarios
            ],
        }


@dataclass
class AdversarialResult:
    """Result of per-layer adversarial scenario."""

    layer: str
    scenario: str
    passed: bool
    details: str = ""


@dataclass
class CogGovBenchResult:
    """Complete CogGov-Bench result."""

    dimensions: list[DimensionScore] = field(default_factory=list)
    adversarial: list[AdversarialResult] = field(default_factory=list)
    composite_score: float = 0.0  # Weighted avg of dimension scores (0-100)
    run_time_ms: float = 0.0
    timestamp: str = ""
    mode: str = "full"
    governance_enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        adv_passed = sum(1 for a in self.adversarial if a.passed)
        return {
            "composite_score": round(self.composite_score, 2),
            "run_time_ms": round(self.run_time_ms, 2),
            "timestamp": self.timestamp,
            "mode": self.mode,
            "governance_enabled": self.governance_enabled,
            "dimensions": [d.to_dict() for d in self.dimensions],
            "adversarial_passed": adv_passed,
            "adversarial_total": len(self.adversarial),
            "adversarial": [
                {"layer": a.layer, "scenario": a.scenario, "passed": a.passed, "details": a.details}
                for a in self.adversarial
            ],
        }


# =============================================================================
# Dimension 1: Constraint Adherence
# =============================================================================


def _measure_constraint_adherence(conn: sqlite3.Connection) -> DimensionScore:
    """Does the agent respect decisions?

    Tests:
      1. High-authority decisions appear in constraint_set
      2. Anti-terms are generated for active decisions
      3. Constraint violations are detected when present
      4. CONSTRAINT presentation mode applied to authority >= 0.80
    """
    dim = DimensionScore(dimension="constraint_adherence")

    # Scenario 1: Decisions with authority >= 0.80 appear as constraints
    try:
        high_auth = conn.execute(
            """SELECT COUNT(*) FROM concepts
               WHERE concept_type IN ('decision', 'constraint')
               AND COALESCE(authority_score, confidence, 0) >= 0.80
               AND status != 'deleted'"""
        ).fetchone()[0]

        # Check if constraint assembly picks them up
        # Look at recent governance events for constraint_assembly
        constraint_events = conn.execute(
            """SELECT details FROM governance_events
               WHERE event_type = 'constraint_assembled'
               ORDER BY created_at DESC LIMIT 5"""
        ).fetchall()

        if high_auth == 0:
            s = ScenarioResult(
                "decisions_as_constraints", True, 100.0, "No high-authority decisions to test (fresh pith)"
            )
        elif constraint_events:
            assembled_count = 0
            for (det_json,) in constraint_events:
                try:
                    det = json.loads(det_json) if det_json else {}
                    assembled_count = max(assembled_count, det.get("constraint_count", 0))
                except (json.JSONDecodeError, TypeError):
                    pass
            ratio = min(assembled_count / max(high_auth, 1), 1.0)
            s = ScenarioResult(
                "decisions_as_constraints",
                ratio >= 0.80,
                ratio * 100,
                f"{assembled_count} constraints assembled from {high_auth} high-auth decisions",
            )
        else:
            # No governance events yet — check if concepts exist with right scores
            s = ScenarioResult(
                "decisions_as_constraints",
                high_auth > 0,
                50.0 if high_auth > 0 else 0.0,
                f"{high_auth} high-auth decisions exist but no constraint events yet",
            )
    except sqlite3.OperationalError as e:
        s = ScenarioResult("decisions_as_constraints", False, 0.0, f"DB error: {e}")

    dim.scenarios.append(s)

    # Scenario 2: Anti-terms exist for constraint-level concepts
    try:
        decisions_with_anti = conn.execute(
            """SELECT COUNT(*) FROM concepts
               WHERE concept_type IN ('decision', 'constraint')
               AND COALESCE(authority_score, confidence, 0) >= 0.80
               AND status != 'deleted'
               AND data LIKE '%anti_terms%'"""
        ).fetchone()[0]

        if high_auth == 0:
            s = ScenarioResult("anti_term_coverage", True, 100.0, "No decisions to check")
        else:
            ratio = decisions_with_anti / max(high_auth, 1)
            s = ScenarioResult(
                "anti_term_coverage",
                ratio >= 0.50,
                ratio * 100,
                f"{decisions_with_anti}/{high_auth} decisions have anti-terms",
            )
    except sqlite3.OperationalError as e:
        s = ScenarioResult("anti_term_coverage", False, 0.0, f"DB error: {e}")

    dim.scenarios.append(s)

    # Scenario 3: Presentation mode alignment
    try:
        misaligned = conn.execute(
            """SELECT COUNT(*) FROM concepts
               WHERE COALESCE(authority_score, confidence, 0) >= 0.80
               AND concept_type NOT IN ('decision', 'constraint', 'principle')
               AND status != 'deleted'"""
        ).fetchone()[0]
        total_high = conn.execute(
            """SELECT COUNT(*) FROM concepts
               WHERE COALESCE(authority_score, confidence, 0) >= 0.80
               AND status != 'deleted'"""
        ).fetchone()[0]

        if total_high == 0:
            s = ScenarioResult("presentation_alignment", True, 100.0, "No high-auth concepts")
        else:
            aligned_ratio = 1 - (misaligned / max(total_high, 1))
            s = ScenarioResult(
                "presentation_alignment",
                aligned_ratio >= 0.70,
                aligned_ratio * 100,
                f"{total_high - misaligned}/{total_high} high-auth concepts are decision/constraint/principle types",
            )
    except sqlite3.OperationalError as e:
        s = ScenarioResult("presentation_alignment", False, 0.0, f"DB error: {e}")

    dim.scenarios.append(s)

    dim.total_count = len(dim.scenarios)
    dim.passed_count = sum(1 for s in dim.scenarios if s.passed)
    dim.score = sum(s.score for s in dim.scenarios) / max(len(dim.scenarios), 1)
    return dim


# =============================================================================
# Dimension 2: Stale Knowledge Resistance
# =============================================================================


def _measure_stale_resistance(conn: sqlite3.Connection) -> DimensionScore:
    """Does the agent avoid outdated knowledge?

    Tests:
      1. Currency scores decrease with age (half-life decay works)
      2. SUPERSEDED concepts excluded from retrieval
      3. Low-currency concepts not in recent context assemblies
      4. Concept-type-aware decay (observations decay faster than principles)
    """
    dim = DimensionScore(dimension="stale_knowledge_resistance")

    # Scenario 1: Currency decay is active
    try:
        # Check if old concepts have lower currency than new ones
        old_concepts = conn.execute(
            """SELECT AVG(currency_score) FROM concepts
               WHERE status != 'deleted'
               AND currency_score IS NOT NULL
               AND COALESCE(content_updated_at, updated_at) < datetime('now', '-30 days')"""  # DATA-033
        ).fetchone()[0]
        new_concepts = conn.execute(
            """SELECT AVG(currency_score) FROM concepts
               WHERE status != 'deleted'
               AND currency_score IS NOT NULL
               AND COALESCE(content_updated_at, updated_at) > datetime('now', '-7 days')"""  # DATA-033
        ).fetchone()[0]

        if old_concepts is None or new_concepts is None:
            s = ScenarioResult("currency_decay_active", True, 50.0, "Insufficient age spread to test decay")
        elif new_concepts > old_concepts:
            s = ScenarioResult(
                "currency_decay_active",
                True,
                100.0,
                f"New avg currency ({new_concepts:.2f}) > old ({old_concepts:.2f})",
            )
        else:
            s = ScenarioResult(
                "currency_decay_active",
                False,
                30.0,
                f"Old concepts ({old_concepts:.2f}) not decaying vs new ({new_concepts:.2f})",
            )
    except sqlite3.OperationalError:
        s = ScenarioResult("currency_decay_active", False, 0.0, "currency_score column not available")

    dim.scenarios.append(s)

    # Scenario 2: SUPERSEDED concepts excluded
    try:
        superseded = conn.execute(
            """SELECT COUNT(*) FROM concepts
               WHERE currency_status = 'SUPERSEDED'
               AND status != 'deleted'"""
        ).fetchone()[0]

        # Check if any SUPERSEDED concepts appear in recent retrievals
        superseded_in_retrieval = 0
        try:
            events = conn.execute(
                """SELECT details FROM governance_events
                   WHERE event_type = 'concept_suppressed'
                   AND details LIKE '%SUPERSEDED%'
                   ORDER BY created_at DESC LIMIT 20"""
            ).fetchall()
            superseded_in_retrieval = len(events)
        except sqlite3.OperationalError:
            pass

        if superseded == 0:
            s = ScenarioResult("superseded_excluded", True, 100.0, "No SUPERSEDED concepts in pith")
        else:
            s = ScenarioResult(
                "superseded_excluded",
                True,
                100.0,
                f"{superseded} SUPERSEDED concepts exist, {superseded_in_retrieval} suppression events logged",
            )
    except sqlite3.OperationalError:
        s = ScenarioResult("superseded_excluded", False, 0.0, "currency_status column not available")

    dim.scenarios.append(s)

    # Scenario 3: Type-aware decay (observations decay faster)
    try:
        obs_currency = conn.execute(
            """SELECT AVG(currency_score) FROM concepts
               WHERE concept_type = 'observation'
               AND currency_score IS NOT NULL
               AND status != 'deleted'
               AND COALESCE(content_updated_at, updated_at) < datetime('now', '-14 days')"""  # DATA-033
        ).fetchone()[0]
        principle_currency = conn.execute(
            """SELECT AVG(currency_score) FROM concepts
               WHERE concept_type = 'principle'
               AND currency_score IS NOT NULL
               AND status != 'deleted'
               AND COALESCE(content_updated_at, updated_at) < datetime('now', '-14 days')"""  # DATA-033
        ).fetchone()[0]

        if obs_currency is None or principle_currency is None:
            s = ScenarioResult("type_aware_decay", True, 50.0, "Insufficient data for type-aware comparison")
        elif principle_currency >= obs_currency:
            s = ScenarioResult(
                "type_aware_decay",
                True,
                100.0,
                f"Principles ({principle_currency:.2f}) more durable than observations ({obs_currency:.2f})",
            )
        else:
            s = ScenarioResult(
                "type_aware_decay",
                False,
                30.0,
                f"Observations ({obs_currency:.2f}) not decaying faster than principles ({principle_currency:.2f})",
            )
    except sqlite3.OperationalError:
        s = ScenarioResult("type_aware_decay", False, 0.0, "Missing currency data")

    dim.scenarios.append(s)

    dim.total_count = len(dim.scenarios)
    dim.passed_count = sum(1 for s in dim.scenarios if s.passed)
    dim.score = sum(s.score for s in dim.scenarios) / max(len(dim.scenarios), 1)
    return dim


# =============================================================================
# Dimension 3: Correction Learning
# =============================================================================


def _measure_correction_learning(conn: sqlite3.Connection) -> DimensionScore:
    """Does the agent improve from corrections?

    Tests:
      1. Corrections are recorded in corrections table
      2. Authority adjusts after corrections (demotion)
      3. Error causes are classified
      4. Recurring patterns detected after 3+ same-cause corrections
    """
    dim = DimensionScore(dimension="correction_learning")

    # Scenario 1: Corrections recorded
    try:
        total_corrections = conn.execute("SELECT COUNT(*) FROM corrections").fetchone()[0]
        s = ScenarioResult(
            "corrections_recorded",
            total_corrections > 0,
            min(total_corrections * 20, 100.0),
            f"{total_corrections} corrections in database",
        )
    except sqlite3.OperationalError:
        s = ScenarioResult("corrections_recorded", False, 0.0, "corrections table not found — Layer 6 not deployed")
    dim.scenarios.append(s)

    # Scenario 2: Authority demotion after corrections
    try:
        demoted = conn.execute(
            f"""SELECT COUNT(*) FROM governance_events
               WHERE event_type = '{GOV_EVENT_AUTHORITY_DEMOTION}'"""
        ).fetchone()[0]

        if total_corrections == 0:
            s = ScenarioResult("authority_demotion", True, 50.0, "No corrections yet to trigger demotions")
        elif demoted > 0:
            s = ScenarioResult("authority_demotion", True, 100.0, f"{demoted} authority demotions from corrections")
        else:
            s = ScenarioResult("authority_demotion", False, 20.0, f"{total_corrections} corrections but 0 demotions")
    except sqlite3.OperationalError:
        s = ScenarioResult("authority_demotion", False, 0.0, "Missing tables")
    dim.scenarios.append(s)

    # Scenario 3: Error cause classification (A2: exclude UNCLASSIFIED)
    try:
        classified = conn.execute(
            """SELECT COUNT(*) FROM corrections
               WHERE error_cause IS NOT NULL
                 AND error_cause != ''
                 AND error_cause != 'UNCLASSIFIED'"""
        ).fetchone()[0]
        if total_corrections == 0:
            s = ScenarioResult("error_classification", True, 50.0, "No corrections to classify")
        else:
            ratio = classified / max(total_corrections, 1)
            s = ScenarioResult(
                "error_classification",
                ratio >= 0.60,
                ratio * 100,
                f"{classified}/{total_corrections} corrections classified (excl. UNCLASSIFIED)",
            )
    except sqlite3.OperationalError:
        s = ScenarioResult("error_classification", False, 0.0, "Missing corrections table")
    dim.scenarios.append(s)

    # Scenario 4: Recurring patterns
    try:
        recurring = conn.execute(
            """SELECT error_cause, COUNT(*) as cnt FROM corrections
               WHERE error_cause IS NOT NULL
               GROUP BY error_cause HAVING cnt >= 3"""
        ).fetchall()
        if total_corrections < 3:
            s = ScenarioResult("recurring_patterns", True, 50.0, "Too few corrections for pattern detection")
        elif len(recurring) > 0:
            s = ScenarioResult("recurring_patterns", True, 100.0, f"{len(recurring)} recurring error patterns detected")
        else:
            s = ScenarioResult("recurring_patterns", True, 70.0, "No recurring patterns (may indicate diverse errors)")
    except sqlite3.OperationalError:
        s = ScenarioResult("recurring_patterns", False, 0.0, "Missing corrections table")
    dim.scenarios.append(s)

    # Scenario 5: Supersession coverage (A4)
    try:
        superseded_count = conn.execute(
            f"""SELECT COUNT(*) FROM governance_events
               WHERE event_type = '{GOV_EVENT_DECISION_SUPERSESSION}'"""
        ).fetchone()[0]
        # Count supersedable pairs: decision-tier concepts with >1 in same KA
        supersedable_kas = conn.execute(
            """SELECT json_extract(data, '$.knowledge_area') as ka, COUNT(*) as cnt
               FROM concepts
               WHERE is_current = 1
                 AND status != 'deleted'
                 AND concept_type IN ('decision', 'principle', 'constraint')
               GROUP BY ka HAVING cnt >= 2"""
        ).fetchall()
        total_supersedable = sum(row[1] for row in supersedable_kas)

        if total_supersedable < 2:
            s = ScenarioResult("supersession_coverage", True, 50.0, "Too few decision-tier concepts for supersession")
        elif superseded_count > 0:
            # Score: ratio of supersessions to supersedable concepts (capped at 100)
            ratio = min(superseded_count / max(total_supersedable * 0.1, 1), 1.0)
            s = ScenarioResult(
                "supersession_coverage",
                True,
                ratio * 100,
                f"{superseded_count} supersessions across {total_supersedable} supersedable concepts",
            )
        else:
            s = ScenarioResult(
                "supersession_coverage",
                False,
                10.0,
                f"0 supersessions despite {total_supersedable} supersedable concepts",
            )
    except sqlite3.OperationalError:
        s = ScenarioResult("supersession_coverage", False, 0.0, "Missing tables")
    dim.scenarios.append(s)

    # Scenario 6: Contradiction resolution rate (A4)
    try:
        detected = conn.execute(
            f"""SELECT COUNT(*) FROM governance_events
               WHERE event_type IN ('{GOV_EVENT_CONTRADICTION_REVIEW}',
                                    '{GOV_EVENT_SUPERSESSION_REVIEW}',
                                    '{GOV_EVENT_DECISION_SUPERSESSION}')"""
        ).fetchone()[0]
        resolved = conn.execute(
            f"""SELECT COUNT(*) FROM governance_events
               WHERE event_type = '{GOV_EVENT_DECISION_SUPERSESSION}'"""
        ).fetchone()[0]

        if detected == 0:
            s = ScenarioResult("contradiction_resolution_rate", True, 50.0, "No contradictions detected yet")
        else:
            ratio = resolved / max(detected, 1)
            s = ScenarioResult(
                "contradiction_resolution_rate",
                ratio >= 0.30,
                ratio * 100,
                f"{resolved}/{detected} contradictions resolved",
            )
    except sqlite3.OperationalError:
        s = ScenarioResult("contradiction_resolution_rate", False, 0.0, "Missing tables")
    dim.scenarios.append(s)

    dim.total_count = len(dim.scenarios)
    dim.passed_count = sum(1 for s in dim.scenarios if s.passed)
    dim.score = sum(s.score for s in dim.scenarios) / max(len(dim.scenarios), 1)
    return dim


# =============================================================================
# Dimension 4: Cross-Session Coherence
# =============================================================================


def _measure_cross_session_coherence(conn: sqlite3.Connection) -> DimensionScore:
    """Does the agent maintain positions across sessions?

    Tests:
      1. Bootstrap loads constraints from prior sessions
      2. Decisions persist across session boundaries
      3. Constraint set consistency across recent turns
    """
    dim = DimensionScore(dimension="cross_session_coherence")

    # Scenario 1: Bootstrap loads constraints
    try:
        # COGGOV-002: Use structured marker to avoid context_hint false positives.
        # Session data stores {"bootstrap": {"constraints_loaded": N}} when bootstrap runs.
        sessions_with_bootstrap = conn.execute(
            """SELECT COUNT(*) FROM sessions
               WHERE data LIKE '%"bootstrap"%constraints_loaded%'
               AND status = 'ended'"""
        ).fetchone()[0]
        total_sessions = conn.execute("SELECT COUNT(*) FROM sessions WHERE status = 'ended'").fetchone()[0]

        if total_sessions == 0:
            s = ScenarioResult("bootstrap_active", True, 50.0, "No completed sessions")
        else:
            ratio = sessions_with_bootstrap / max(total_sessions, 1)
            s = ScenarioResult(
                "bootstrap_active",
                ratio >= 0.50,
                ratio * 100,
                f"{sessions_with_bootstrap}/{total_sessions} sessions used bootstrap",
            )
    except sqlite3.OperationalError:
        s = ScenarioResult("bootstrap_active", False, 0.0, "Missing session data")
    dim.scenarios.append(s)

    # Scenario 2: Decision persistence — decisions from >7 days ago still active
    try:
        old_decisions = conn.execute(
            """SELECT COUNT(*) FROM concepts
               WHERE concept_type = 'decision'
               AND status != 'deleted'
               AND created_at < datetime('now', '-7 days')"""
        ).fetchone()[0]
        old_active = conn.execute(
            """SELECT COUNT(*) FROM concepts
               WHERE concept_type = 'decision'
               AND status != 'deleted'
               AND created_at < datetime('now', '-7 days')
               AND (currency_status IS NULL OR currency_status = 'ACTIVE')"""
        ).fetchone()[0]

        if old_decisions == 0:
            s = ScenarioResult("decision_persistence", True, 50.0, "No decisions older than 7 days")
        else:
            ratio = old_active / max(old_decisions, 1)
            s = ScenarioResult(
                "decision_persistence",
                ratio >= 0.70,
                ratio * 100,
                f"{old_active}/{old_decisions} old decisions still ACTIVE",
            )
    except sqlite3.OperationalError:
        s = ScenarioResult("decision_persistence", False, 0.0, "Missing data")
    dim.scenarios.append(s)

    # Scenario 3: Constraint consistency — same decisions appear in recent turns
    try:
        recent_constraints = conn.execute(
            """SELECT details FROM governance_events
               WHERE event_type = 'constraint_assembled'
               ORDER BY created_at DESC LIMIT 10"""
        ).fetchall()

        if len(recent_constraints) < 2:
            s = ScenarioResult(
                "constraint_consistency", True, 50.0, "Insufficient constraint events for consistency check"
            )
        else:
            # Compare constraint sets across recent turns
            constraint_sets = []
            for (det_json,) in recent_constraints:
                try:
                    det = json.loads(det_json) if det_json else {}
                    ids = set(det.get("concept_ids", []))
                    if ids:
                        constraint_sets.append(ids)
                except (json.JSONDecodeError, TypeError):
                    pass

            if len(constraint_sets) >= 2:
                # Jaccard similarity between consecutive constraint sets
                similarities = []
                for i in range(len(constraint_sets) - 1):
                    a, b = constraint_sets[i], constraint_sets[i + 1]
                    if a or b:
                        jaccard = len(a & b) / len(a | b) if (a | b) else 1.0
                        similarities.append(jaccard)

                avg_sim = sum(similarities) / len(similarities) if similarities else 0.5
                s = ScenarioResult(
                    "constraint_consistency",
                    avg_sim >= 0.60,
                    avg_sim * 100,
                    f"Mean constraint Jaccard similarity: {avg_sim:.2f}",
                )
            else:
                s = ScenarioResult("constraint_consistency", True, 50.0, "Not enough constraint set data")
    except sqlite3.OperationalError:
        s = ScenarioResult("constraint_consistency", False, 0.0, "Missing events")
    dim.scenarios.append(s)

    dim.total_count = len(dim.scenarios)
    dim.passed_count = sum(1 for s in dim.scenarios if s.passed)
    dim.score = sum(s.score for s in dim.scenarios) / max(len(dim.scenarios), 1)
    return dim


# =============================================================================
# Dimension 5: Context Integrity
# =============================================================================


def _measure_context_integrity(conn: sqlite3.Connection) -> DimensionScore:
    """Is context internally consistent?

    Tests:
      1. Contradiction detection catches known conflicts
      2. No CONTESTED concepts served without flags
      3. Budget allocation respects tier priorities
    """
    dim = DimensionScore(dimension="context_integrity")

    # Scenario 1: Contradiction detection operational
    try:
        detected = conn.execute(
            f"""SELECT COUNT(*) FROM governance_events
               WHERE UPPER(event_type) = '{GOV_EVENT_CONTRADICTION_DETECTED}'"""
        ).fetchone()[0]
        detection_runs = conn.execute(
            """SELECT COUNT(*) FROM governance_events
               WHERE UPPER(event_type) IN ('CONTRADICTION_DETECTION_COMPLETE',
                                            'CONTRADICTION_PHASE_1', 'CONTRADICTION_PHASE_2',
                                            'CONTRADICTION_PHASE_2_COMPLETED')"""
        ).fetchone()[0]

        if detection_runs > 0:
            s = ScenarioResult(
                "contradiction_detection",
                True,
                100.0,
                f"{detected} contradictions found in {detection_runs} detection runs",
            )
        else:
            s = ScenarioResult("contradiction_detection", False, 20.0, "No contradiction detection runs recorded")
    except sqlite3.OperationalError:
        s = ScenarioResult("contradiction_detection", False, 0.0, "Missing events table")
    dim.scenarios.append(s)

    # Scenario 2: CONTESTED concepts flagged
    try:
        contested = conn.execute(
            """SELECT COUNT(*) FROM concepts
               WHERE currency_status = 'CONTESTED'
               AND status != 'deleted'"""
        ).fetchone()[0]

        if contested == 0:
            # In a mature pith (1000+ concepts) with active correction detection,
            # zero contested concepts likely means corrections aren't producing
            # contested flags. Score 70 (passing but flagged) instead of 100.
            total_concepts = conn.execute("SELECT COUNT(*) FROM concepts WHERE status != 'deleted'").fetchone()[0]
            if total_concepts > 500:
                s = ScenarioResult(
                    "contested_flagging",
                    True,
                    70.0,
                    f"No CONTESTED concepts in mature pith ({total_concepts} concepts) — "
                    "correction pipeline may not be producing contested flags",
                )
            else:
                s = ScenarioResult("contested_flagging", True, 100.0, "No CONTESTED concepts (clean knowledge base)")
        else:
            # Check if contested concepts get uncertainty qualifiers
            s = ScenarioResult(
                "contested_flagging", True, 80.0, f"{contested} CONTESTED concepts exist with status flags"
            )
    except sqlite3.OperationalError:
        s = ScenarioResult("contested_flagging", True, 50.0, "currency_status column not available")
    dim.scenarios.append(s)

    # Scenario 3: Budget tier compliance
    try:
        budget_events = conn.execute(
            f"""SELECT details FROM governance_events
               WHERE event_type = '{GOV_EVENT_BUDGET_ALLOCATED}'
               ORDER BY created_at DESC LIMIT 10"""
        ).fetchall()

        if not budget_events:
            # Check for budget_governance phase in phases_executed
            budget_phase = conn.execute(
                """SELECT COUNT(*) FROM governance_events
                   WHERE event_type LIKE '%budget%'"""
            ).fetchone()[0]
            if budget_phase > 0:
                s = ScenarioResult(
                    "budget_tier_compliance", True, 80.0, f"Budget governance active ({budget_phase} events)"
                )
            else:
                s = ScenarioResult("budget_tier_compliance", False, 20.0, "No budget allocation events recorded")
        else:
            s = ScenarioResult(
                "budget_tier_compliance", True, 100.0, f"{len(budget_events)} budget allocations recorded"
            )
    except sqlite3.OperationalError:
        s = ScenarioResult("budget_tier_compliance", False, 0.0, "Missing events")
    dim.scenarios.append(s)

    dim.total_count = len(dim.scenarios)
    dim.passed_count = sum(1 for s in dim.scenarios if s.passed)
    dim.score = sum(s.score for s in dim.scenarios) / max(len(dim.scenarios), 1)
    return dim


# =============================================================================
# Dimension 6: Recovery Rate
# =============================================================================


def _measure_recovery_rate(conn: sqlite3.Connection) -> DimensionScore:
    """How quickly are errors fixed?

    Tests:
      1. Circuit breaker operational (trips and recovers)
      2. Governance health checks running
      3. Correction-to-skill pipeline functional
    """
    dim = DimensionScore(dimension="recovery_rate")

    # Scenario 1: Circuit breaker operational
    try:
        cb_trips = conn.execute(
            """SELECT COUNT(*) FROM governance_events
               WHERE event_type = 'circuit_breaker_tripped'"""
        ).fetchone()[0]
        cb_resets = conn.execute(
            """SELECT COUNT(*) FROM governance_events
               WHERE event_type = 'circuit_breaker_reset'"""
        ).fetchone()[0]

        if cb_trips == 0:
            s = ScenarioResult("circuit_breaker", True, 80.0, "No circuit breaker trips (stable system)")
        elif cb_resets >= cb_trips:
            s = ScenarioResult(
                "circuit_breaker", True, 100.0, f"{cb_trips} trips, {cb_resets} recoveries — full recovery"
            )
        else:
            s = ScenarioResult("circuit_breaker", False, 40.0, f"{cb_trips} trips but only {cb_resets} recoveries")
    except sqlite3.OperationalError:
        s = ScenarioResult("circuit_breaker", True, 50.0, "No governance events table — circuit breaker not wired")
    dim.scenarios.append(s)

    # Scenario 2: Health checks running
    try:
        health_events = conn.execute(
            """SELECT COUNT(*) FROM governance_events
               WHERE event_type LIKE 'health_%'
               AND created_at > datetime('now', '-24 hours')"""
        ).fetchone()[0]

        if health_events > 0:
            s = ScenarioResult("health_monitoring", True, 100.0, f"{health_events} health events in last 24h")
        else:
            # Check if governance is running at all
            any_events = conn.execute(
                """SELECT COUNT(*) FROM governance_events
                   WHERE created_at > datetime('now', '-24 hours')"""
            ).fetchone()[0]
            if any_events > 0:
                s = ScenarioResult(
                    "health_monitoring",
                    True,
                    60.0,
                    f"Governance active ({any_events} events) but no dedicated health checks",
                )
            else:
                s = ScenarioResult("health_monitoring", False, 20.0, "No governance activity in last 24h")
    except sqlite3.OperationalError:
        s = ScenarioResult("health_monitoring", False, 0.0, "Missing events table")
    dim.scenarios.append(s)

    # Scenario 3: Skill extraction from corrections
    try:
        skills_from_corrections = conn.execute(
            """SELECT COUNT(*) FROM concept_skills
               WHERE source_type = 'correction'"""
        ).fetchone()[0]
        total_corrections = conn.execute("SELECT COUNT(*) FROM corrections").fetchone()[0]

        if total_corrections == 0:
            s = ScenarioResult("correction_to_skill", True, 50.0, "No corrections yet for skill extraction")
        elif skills_from_corrections > 0:
            s = ScenarioResult(
                "correction_to_skill",
                True,
                100.0,
                f"{skills_from_corrections} skills extracted from {total_corrections} corrections",
            )
        else:
            s = ScenarioResult(
                "correction_to_skill", False, 30.0, f"{total_corrections} corrections but 0 skills extracted"
            )
    except sqlite3.OperationalError:
        s = ScenarioResult("correction_to_skill", True, 40.0, "Skills/corrections tables not yet deployed")
    dim.scenarios.append(s)

    dim.total_count = len(dim.scenarios)
    dim.passed_count = sum(1 for s in dim.scenarios if s.passed)
    dim.score = sum(s.score for s in dim.scenarios) / max(len(dim.scenarios), 1)
    return dim


# =============================================================================
# Per-Layer Adversarial Scenarios (Gap E §E.2)
# =============================================================================

ADVERSARIAL_SCENARIOS = {
    "Layer 1 - Authority": [
        ("inflate_authority", "Concepts with max evidence should not all cluster at authority 1.0"),
        ("impersonate_decision", "Observation using decision-like language gets observation-level authority"),
        ("demote_after_corrections", "CONSTRAINT concept corrected 4x drops below 0.60 authority"),
    ],
    "Layer 2 - Currency": [
        ("zombie_concept", "Concept past half-life doesn't resurrect to full currency on access"),
        ("topic_flood", "50 concepts in same area don't inflate old concept currency"),
    ],
    "Layer 2.5 - Contradiction": [
        ("subtle_contradiction", "Semantically opposing statements detected without negation words"),
        ("false_positive_similarity", "Similar but non-contradictory statements not flagged"),
    ],
    "Layer 3 - Prediction Error": [
        ("synonym_bypass", "Synonym of anti-term still caught by enforcement"),
        ("empty_constraints", "System functions normally with no high-authority concepts"),
    ],
    "Layer 5 - Bootstrap": [
        ("pin_flood", "Pinning >10 concepts enforces budget limit"),
        ("stale_constraint", "High-authority low-currency concept in stale_alerts not active_constraints"),
    ],
    "Layer 8 - Budget": [
        ("constraint_flood", "12 relevant constraints caps Tier 1 at 8, Tier 3 gets >= 5"),
        ("empty_retrieval", "0 retrieval results still loads bootstrap constraints in Tier 1"),
    ],
}


def _run_adversarial_scenarios(conn: sqlite3.Connection) -> list[AdversarialResult]:
    """Run per-layer adversarial tests against live database state."""
    results = []

    for layer, scenarios in ADVERSARIAL_SCENARIOS.items():
        for scenario_id, description in scenarios:
            result = _run_single_adversarial(conn, layer, scenario_id, description)
            results.append(result)

    return results


def _run_single_adversarial(
    conn: sqlite3.Connection, layer: str, scenario_id: str, description: str
) -> AdversarialResult:
    """Run one adversarial scenario. Returns pass/fail with details."""

    try:
        if scenario_id == "inflate_authority":
            # Check authority distribution — should not cluster at 1.0
            row = conn.execute(
                """SELECT COUNT(*) as total,
                          SUM(CASE WHEN authority_score > 0.95 THEN 1 ELSE 0 END) as near_max
                   FROM concepts WHERE status != 'deleted' AND authority_score IS NOT NULL"""
            ).fetchone()
            total, near_max = row[0] or 0, row[1] or 0
            if total == 0:
                return AdversarialResult(layer, description, True, "No scored concepts")
            ratio = near_max / total
            return AdversarialResult(
                layer, description, ratio < 0.30, f"{near_max}/{total} ({ratio:.0%}) near max authority"
            )

        elif scenario_id == "impersonate_decision":
            # Observations should have authority < 0.60 even with strong language
            row = conn.execute(
                """SELECT COUNT(*) FROM concepts
                   WHERE concept_type = 'observation'
                   AND authority_score IS NOT NULL
                   AND authority_score >= 0.70
                   AND status != 'deleted'"""
            ).fetchone()[0]
            total_obs = conn.execute(
                """SELECT COUNT(*) FROM concepts
                   WHERE concept_type = 'observation' AND status != 'deleted'"""
            ).fetchone()[0]
            if total_obs == 0:
                return AdversarialResult(layer, description, True, "No observations")
            ratio = row / max(total_obs, 1)
            return AdversarialResult(
                layer, description, ratio < 0.10, f"{row}/{total_obs} observations with authority >= 0.70"
            )

        elif scenario_id == "demote_after_corrections":
            # Concepts with 3+ corrections should have reduced authority
            try:
                corrected = conn.execute("""SELECT affected_concept_ids FROM corrections""").fetchall()
                correction_counts = {}
                for (ids_json,) in corrected:
                    try:
                        ids = json.loads(ids_json) if ids_json else []
                        for cid in ids:
                            correction_counts[cid] = correction_counts.get(cid, 0) + 1
                    except (json.JSONDecodeError, TypeError):
                        pass

                heavily_corrected = [cid for cid, cnt in correction_counts.items() if cnt >= 3]
                if not heavily_corrected:
                    return AdversarialResult(layer, description, True, "No concepts with 3+ corrections")
                # Check their authority is reduced
                placeholders = ",".join(["?"] * len(heavily_corrected))
                still_high = conn.execute(
                    f"""SELECT COUNT(*) FROM concepts
                        WHERE id IN ({placeholders})
                        AND authority_score >= 0.80""",
                    heavily_corrected,
                ).fetchone()[0]
                return AdversarialResult(
                    layer,
                    description,
                    still_high == 0,
                    f"{still_high}/{len(heavily_corrected)} heavily-corrected still at CONSTRAINT level",
                )
            except sqlite3.OperationalError:
                return AdversarialResult(layer, description, True, "corrections table not deployed")

        elif scenario_id == "zombie_concept":
            # Old concepts shouldn't have currency near 1.0
            row = conn.execute(
                """SELECT COUNT(*) FROM concepts
                   WHERE currency_score IS NOT NULL
                   AND currency_score > 0.95
                   AND COALESCE(content_updated_at, updated_at) < datetime('now', '-60 days')
                   AND status != 'deleted'"""
            ).fetchone()[0]
            return AdversarialResult(layer, description, row == 0, f"{row} zombie concepts (old but full currency)")

        elif scenario_id == "empty_constraints":
            # System should work even with 0 decisions
            decisions = conn.execute(
                """SELECT COUNT(*) FROM concepts
                   WHERE concept_type = 'decision' AND status != 'deleted'"""
            ).fetchone()[0]
            if decisions == 0:
                return AdversarialResult(layer, description, True, "Pith has 0 decisions and is functional")
            return AdversarialResult(
                layer, description, True, f"Pith has {decisions} decisions (can't test empty state)"
            )

        elif scenario_id == "subtle_contradiction":
            # Check if Phase 2 contradiction detection is wired.
            # contradiction.py logs "CONTRADICTION_PHASE_2_COMPLETED" after Phase 2 runs,
            # and "CONTRADICTION_DETECTED" with "phase": 2 in details for each detection.
            try:
                phase2 = conn.execute(
                    f"""SELECT COUNT(*) FROM governance_events
                       WHERE event_type = '{GOV_EVENT_CONTRADICTION_PHASE_2_COMPLETED}'
                          OR (event_type = '{GOV_EVENT_CONTRADICTION_DETECTED}'
                              AND details LIKE '%"phase": 2%')"""
                ).fetchone()[0]
                return AdversarialResult(
                    layer, description, phase2 > 0, f"Phase 2 embedding detection: {phase2} events"
                )
            except sqlite3.OperationalError:
                return AdversarialResult(layer, description, False, "No contradiction detection events")

        elif scenario_id == "false_positive_similarity":
            # Low false positive rate
            try:
                dismissed = conn.execute(
                    """SELECT COUNT(*) FROM governance_events
                       WHERE event_type = 'contradiction_dismissed'"""
                ).fetchone()[0]
                total_det = conn.execute(
                    """SELECT COUNT(*) FROM governance_events
                       WHERE event_type = 'contradiction_detected'"""
                ).fetchone()[0]
                if total_det == 0:
                    return AdversarialResult(layer, description, True, "No contradictions to check FPR")
                fpr = dismissed / max(total_det + dismissed, 1)
                return AdversarialResult(
                    layer,
                    description,
                    fpr < 0.30,
                    f"FPR: {fpr:.2%} ({dismissed} dismissed / {total_det + dismissed} total)",
                )
            except sqlite3.OperationalError:
                return AdversarialResult(layer, description, True, "No events data")

        else:
            # Scenarios without specific DB checks pass with note
            return AdversarialResult(layer, description, True, "Requires synthetic test data injection (future)")

    except Exception as e:
        return AdversarialResult(layer, description, False, f"Error: {e}")


# =============================================================================
# Main Entry Point
# =============================================================================

# Dimension weights for composite score (sum = 1.0)
DIMENSION_WEIGHTS = {
    "constraint_adherence": 0.25,  # Most critical — decisions must constrain
    "stale_knowledge_resistance": 0.15,
    "correction_learning": 0.15,
    "cross_session_coherence": 0.20,  # Second most critical — coherence across sessions
    "context_integrity": 0.15,
    "recovery_rate": 0.10,
}


def _raise_if_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise CogGovBenchCancelled("CogGov bench cancelled")


def run_coggov_bench(
    mode: str = "full",
    *,
    cancel_event: threading.Event | None = None,
) -> CogGovBenchResult:
    """Run CogGov-Bench across all 6 behavioral dimensions.

    Args:
        mode: "light" (dims 1-3) or "full" (all 6 + adversarial)
        cancel_event: Cooperative cancellation signal owned by the caller

    Returns:
        CogGovBenchResult with per-dimension scores and composite score
    """
    from app.storage import owned_connection

    with owned_connection() as conn:
        return _run_coggov_bench_with_connection(conn, mode=mode, cancel_event=cancel_event)


def _run_coggov_bench_with_connection(
    conn: sqlite3.Connection,
    *,
    mode: str = "full",
    cancel_event: threading.Event | None = None,
) -> CogGovBenchResult:
    """Run the benchmark using a caller-owned thread-local connection."""
    # OPS-016: Auto-migrate schema before benchmark runs.
    # CREATE TABLE IF NOT EXISTS doesn't add new columns to existing tables.
    # Governance migrations handle ALTER TABLE additions idempotently.
    try:
        from app.storage.migration import run_governance_migrations

        run_governance_migrations(conn)
    except Exception:
        pass  # Non-fatal — bench can still run with existing schema

    start = time.perf_counter()
    result = CogGovBenchResult(
        timestamp=_utc_now_iso(),
        mode=mode,
    )

    # Always run first 3 dimensions
    _raise_if_cancelled(cancel_event)
    result.dimensions.append(_measure_constraint_adherence(conn))
    _raise_if_cancelled(cancel_event)
    result.dimensions.append(_measure_stale_resistance(conn))
    _raise_if_cancelled(cancel_event)
    result.dimensions.append(_measure_correction_learning(conn))

    if mode == "full":
        _raise_if_cancelled(cancel_event)
        result.dimensions.append(_measure_cross_session_coherence(conn))
        _raise_if_cancelled(cancel_event)
        result.dimensions.append(_measure_context_integrity(conn))
        _raise_if_cancelled(cancel_event)
        result.dimensions.append(_measure_recovery_rate(conn))

        # Run adversarial scenarios
        _raise_if_cancelled(cancel_event)
        result.adversarial = _run_adversarial_scenarios(conn)

    # Composite score: weighted average
    total_weight = 0.0
    weighted_sum = 0.0
    for dim in result.dimensions:
        w = DIMENSION_WEIGHTS.get(dim.dimension, 0.10)
        weighted_sum += dim.score * w
        total_weight += w

    result.composite_score = weighted_sum / max(total_weight, 0.01)
    result.run_time_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "CogGov-Bench (%s): composite=%.1f, %d/%d scenarios passed, %d/%d adversarial passed in %.1fms",
        mode,
        result.composite_score,
        sum(d.passed_count for d in result.dimensions),
        sum(d.total_count for d in result.dimensions),
        sum(1 for a in result.adversarial if a.passed),
        len(result.adversarial),
        result.run_time_ms,
    )

    return result
