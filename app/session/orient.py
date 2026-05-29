"""OrientMixin — present-moment orientation pipeline.

Extracted from session/__init__.py lines 1041-1462 per ARCH-009.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Optional

from app.core.constants import (
    FRESHNESS_EARLIER_TODAY_UPPER,
    FRESHNESS_HOURS_AGO_UPPER,
    FRESHNESS_JUST_NOW_MINS,
    FRESHNESS_MINUTES_AGO_UPPER,
    FRESHNESS_ONE_HOUR_UPPER,
    FRESHNESS_YESTERDAY_UPPER,
    GOV_EVENT_CCL_VIOLATIONS_DETECTED,
    GOV_EVENT_CIRCUIT_BREAKER_TRIPPED,
    GOV_EVENT_COMPACTION_REINJECTION,
    GOV_EVENT_CONTRADICTION_REVIEW,
    GOV_EVENT_CONVERSATION_TURN_COMPLETE,
    GOV_EVENT_GRAPH_CONTRADICTION_SIGNAL,
    GOV_EVENT_RESUME_CONTEXT_INJECTION,
    MINUTES_PER_HOUR,
)
from app.core.datetime_utils import _ensure_aware, _utc_now, _utc_now_iso
from app.core.models import (
    ActivatedConcept,
    ActiveDirectionality,
    ActiveUncertainty,
    AreaStrength,
    Concept,
    ConceptEvolution,
    ConceptEvolutionRecord,
    ConversationTurnRequest,
    ConversationTurnResponse,
    CuriosityFrontierItem,
    CurrentStateAssessment,
    EvolvedConcept,
    GoalSummary,
    LearnedConcept,
    PendingQuestionSummary,
    PresentMomentOrientation,
    RecentConceptSummary,
    RecentEvolutionSummary,
    SearchResult,
    SessionEndRequest,
    SessionInfo,
    SessionLearnRequest,
    SessionLearnResponse,
    SessionStartResponse,
)
from app.session.self_model import self_model_manager
from app.storage import (
    _get_connection,
    cleanup_expired_snapshots,
    count_associations,
    count_sessions,
    get_related_concepts,
    list_concepts,
    load_associations,
    load_concept,
    load_recent_concepts,
    load_resume_snapshot,
    load_session_velocity,
    recover_interrupted_sessions,
    save_concept,
    # Resume Context v1.1
    save_resume_snapshot,
    save_session,
    update_session,
)
from app.cognitive.taxonomy import (  # DEBT-030/DEBT-108
    classify_knowledge_area,
    infer_knowledge_area,
    normalize_knowledge_area,
)

from app.session.helpers import DEFAULT_WINDOW, TIME_WINDOWS

logger = logging.getLogger(__name__)

from app.session.helpers import DEFAULT_WINDOW, TIME_WINDOWS

logger = logging.getLogger(__name__)


class OrientMixin:
    """Mixin providing orient methods for SessionManager."""

    def orient(
        self,
        concepts: list[Concept],
        time_window: str = DEFAULT_WINDOW,
        include_workstreams: bool = False,
        workstream_limit: int = 8,
        origin_id: str | None = None,
        session_id: str | None = None,
    ) -> PresentMomentOrientation:
        """Generate present moment orientation.

        Args:
            concepts: Pre-loaded concepts (avoids redundant storage scan).
            time_window: "1_day" | "7_days" | "30_days" | "all"
            include_workstreams: Explicit opt-in for compact Workstream status.
            workstream_limit: Maximum Workstream rows to return when opted in.
            origin_id: Optional checkpoint origin for active binding status.
            session_id: Optional checkpoint session for active binding status.
        """
        now = _utc_now()
        delta = TIME_WINDOWS.get(time_window, TIME_WINDOWS[DEFAULT_WINDOW])
        cutoff = (now - delta).isoformat()

        where_been = self._compute_where_been(concepts, cutoff, time_window)
        where_am = self._compute_where_am(concepts)
        where_going = self._compute_where_going(concepts)
        workstreams = None
        if include_workstreams:
            workstreams = self._compute_workstreams_status(
                origin_id=origin_id,
                session_id=session_id,
                limit=workstream_limit,
            )

        orientation = PresentMomentOrientation(
            generated_at=now.isoformat(),
            generated_by="pith_deterministic",
            where_been=where_been,
            where_am=where_am,
            where_going=where_going,
            open_threads=self._compute_open_threads(),
            experiment_summary=self._compute_experiment_summary(),
            workstreams=workstreams,
        )

        # Compute orientation hash for cache invalidation (exclude timestamps)
        data = orientation.model_dump(exclude={"orientation_hash", "generated_at"})
        if orientation.workstreams is None:
            data.pop("workstreams", None)
        orientation.orientation_hash = hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode()
        ).hexdigest()

        return orientation

    def _compute_where_been(self, concepts: list[Concept], cutoff: str, window_label: str) -> RecentEvolutionSummary:
        """Recent evolution summary: what changed in the time window."""
        created = []
        evolved = []
        total_events = 0

        for c in concepts:
            # Recently created
            if c.created_at and c.created_at >= cutoff:
                created.append(
                    RecentConceptSummary(
                        concept_id=c.id,
                        summary=c.summary[:100],
                        knowledge_area=(c.metadata or {}).get("knowledge_area", "unknown"),
                        created_at=c.created_at,
                    )
                )
                total_events += 1

            # Recently evolved (version > v1 AND updated in window)
            if c.updated_at and c.updated_at >= cutoff and c.version and c.version != "v1":
                evolved.append(
                    ConceptEvolutionRecord(
                        concept_id=c.id,
                        summary=c.summary[:100],
                        change_type=c.change_type or "",
                        change_reason=c.change_reason or "",
                        evolved_at=c.updated_at,
                    )
                )
                total_events += 1

        # Sort by recency
        created.sort(key=lambda x: x.created_at or "", reverse=True)
        evolved.sort(key=lambda x: x.evolved_at or "", reverse=True)

        # ARCH-O01: Query recent contradictions from DB (capped at 20)
        contradictions = []
        try:
            from app.storage import _db
            with _db() as conn:
                contra_rows = conn.execute(
                    """SELECT concept_a_id, concept_b_id, contradiction_type,
                              action, reason, created_at
                       FROM contradiction_resolutions
                       WHERE created_at >= ?
                       ORDER BY created_at DESC LIMIT 20""",
                    (cutoff,),
                ).fetchall()
                contradictions = [
                    {
                        "concept_a_id": r["concept_a_id"],
                        "concept_b_id": r["concept_b_id"],
                        "type": r["contradiction_type"],
                        "action": r["action"],
                        "reason": r["reason"][:200] if r["reason"] else "",
                        "detected_at": r["created_at"],
                    }
                    for r in contra_rows
                ]
        except Exception as e:
            logger.warning(f"ARCH-O01: Failed to load contradictions (non-fatal): {e}")

        # ARCH-O01: Query recent corrections from DB (capped at 20)
        corrections = []
        try:
            from app.storage import _db
            with _db() as conn:
                corr_rows = conn.execute(
                    """SELECT id, correction_type, corrected_claim, correct_claim,
                              detection_confidence, created_at
                       FROM corrections
                       WHERE created_at >= ?
                       ORDER BY created_at DESC LIMIT 20""",
                    (cutoff,),
                ).fetchall()
                corrections = [
                    {
                        "correction_id": r["id"],
                        "type": r["correction_type"],
                        "corrected_claim": r["corrected_claim"][:200] if r["corrected_claim"] else "",
                        "correct_claim": r["correct_claim"][:200] if r["correct_claim"] else "",
                        "confidence": r["detection_confidence"],
                        "corrected_at": r["created_at"],
                    }
                    for r in corr_rows
                ]
        except Exception as e:
            logger.warning(f"ARCH-O01: Failed to load corrections (non-fatal): {e}")

        return RecentEvolutionSummary(
            time_window=window_label,
            concepts_created=created[:20],
            concepts_evolved=evolved[:20],
            concepts_decayed=[],  # STUB — decay doesn't annotate change_type
            contradictions_detected=contradictions,  # ARCH-O01: wired
            corrections_made=corrections,  # ARCH-O01: wired
            session_count_in_window=count_sessions(since=cutoff),
            total_learning_events_in_window=total_events,
        )

    def _compute_where_am(self, concepts: list[Concept]) -> CurrentStateAssessment:
        """Current state assessment: knowledge health, strengths, weaknesses, uncertainties.

        Reuses SelfModel epistemic profile for health/areas to avoid recomputation.
        """
        # Get epistemic data from SelfModel (cached or generate)
        sm = self_model_manager.load()
        if sm is None:
            sm = self_model_manager.generate(concepts)

        ep = sm.epistemic_profile
        kh = ep.knowledge_health

        health = {
            "total_concepts": kh.total_concepts,
            "total_associations": 0,
            "avg_confidence": kh.avg_confidence,
            "avg_stability": kh.avg_stability,
            "contradiction_density": kh.contradiction_density,
            "evidence_coverage": 0.0,
        }

        # Count associations
        health["total_associations"] = count_associations()

        # Evidence coverage: % of concepts with any evidence
        if concepts:
            with_evidence = sum(1 for c in concepts if c.evidence)
            health["evidence_coverage"] = round(with_evidence / len(concepts), 3)

        # Strongest areas (top 3 by avg_confidence, min 2 concepts)
        dist = sorted(
            ep.knowledge_distribution,
            key=lambda d: d.avg_confidence,
            reverse=True,
        )
        strongest = [
            AreaStrength(
                knowledge_area=d.knowledge_area,
                concept_count=d.concept_count,
                avg_confidence=round(d.avg_confidence, 3),
                reason="Highest avg confidence",
            )
            for d in dist[:3]
            if d.concept_count >= 2
        ]

        # Weakest areas (bottom 3 by avg_confidence, min 2 concepts)
        multi_concept = [d for d in dist if d.concept_count >= 2]
        weakest = [
            AreaStrength(
                knowledge_area=d.knowledge_area,
                concept_count=d.concept_count,
                avg_confidence=round(d.avg_confidence, 3),
                reason="Lowest avg confidence",
            )
            for d in reversed(multi_concept[-3:])
        ]

        # Active uncertainties (top 5 lowest-confidence concepts)
        sorted_by_conf = sorted(concepts, key=lambda c: c.confidence)
        uncertainties = [
            ActiveUncertainty(
                concept_id=c.id,
                summary=c.summary[:100],
                confidence=c.confidence,
                uncertainty_type=("low_stability" if c.stability < 0.3 else "low_confidence"),
            )
            for c in sorted_by_conf[:5]
        ]

        # Pending questions from curiosity engine
        import app.features.question_queue as question_queue

        raw_questions = question_queue.get_questions(limit=5)
        questions = [
            PendingQuestionSummary(
                question=q.get("question", ""),
                concept_id=q.get("concept_id", ""),
                priority=q.get("priority", 0.0),
            )
            for q in raw_questions
        ]

        # --- Cognitive velocity (self-awareness) ---
        cognitive_velocity = self._compute_cognitive_velocity()

        return CurrentStateAssessment(
            knowledge_health=health,
            strongest_areas=strongest,
            weakest_areas=weakest,
            active_uncertainties=uncertainties,
            pending_questions=questions,
            cognitive_velocity=cognitive_velocity,
        )

    def _compute_cognitive_velocity(self) -> "CognitiveVelocity":
        """Compute self-awareness metrics: how fast is Pith growing?

        Uses 7-day current window vs 7-day prior window for trend detection.
        Gracefully returns defaults on any error.
        """
        from app.core.models import CognitiveVelocity

        try:
            now = _utc_now()
            window_days = 7
            current_cutoff = (now - timedelta(days=window_days)).isoformat()
            prior_cutoff = (now - timedelta(days=window_days * 2)).isoformat()

            velocity_data = load_session_velocity(current_cutoff, prior_cutoff)
            current = velocity_data["current"]
            prior = velocity_data.get("prior")

            sessions = current["session_count"]
            created = current["total_concepts_created"]
            evolved = current["total_concepts_evolved"]
            learning_events = current["total_learning_events"]

            avg_concepts = round(created / max(sessions, 1), 2)
            avg_learning = round(learning_events / max(sessions, 1), 2)
            growth_rate = round(created / max(window_days, 1), 2)

            # Trend detection: compare current vs prior window
            trend = "insufficient_data"
            trend_detail = ""
            if prior and prior["session_count"] >= 2 and sessions >= 2:
                prior_rate = prior["total_concepts_created"] / max(window_days, 1)
                if growth_rate > prior_rate * 1.25:
                    trend = "accelerating"
                    trend_detail = f"Growth rate {growth_rate}/day vs {round(prior_rate, 2)}/day in prior window"
                elif growth_rate < prior_rate * 0.75:
                    trend = "decelerating"
                    trend_detail = f"Growth rate {growth_rate}/day vs {round(prior_rate, 2)}/day in prior window"
                else:
                    trend = "steady"
                    trend_detail = f"Growth rate ~{growth_rate}/day (stable)"
            elif sessions >= 1:
                trend_detail = f"Only {sessions} session(s) in current window — need more data"

            return CognitiveVelocity(
                sessions_in_window=sessions,
                concepts_created_in_window=created,
                concepts_evolved_in_window=evolved,
                learning_events_in_window=learning_events,
                avg_concepts_per_session=avg_concepts,
                avg_learning_events_per_session=avg_learning,
                knowledge_growth_rate=growth_rate,
                trend=trend,
                trend_detail=trend_detail,
            )
        except Exception as e:
            logger.error(f"Cognitive velocity computation failed: {e}")
            return CognitiveVelocity()

    def _compute_where_going(self, concepts: list[Concept]) -> ActiveDirectionality:
        """Active directionality: goals, priorities, curiosity frontiers, actions.

        Synthesizes direction from multiple signals:
        - goal-type concepts (explicit goals)
        - high-confidence decisions (strategic direction)
        - weakest knowledge areas (growth frontiers)
        - question queue (curiosity gaps)
        """
        from app.core.models import RecommendedAction, StrategicPriority

        # 1. Extract goal-type concepts
        goals = []
        for c in concepts:
            if c.concept_type == "goal":
                linked = get_related_concepts(c.id, max_depth=1)
                goals.append(
                    GoalSummary(
                        goal_id=c.id,
                        summary=c.summary[:100],
                        priority=c.salience if c.salience else 0.5,
                        progress_indicator="in_progress",
                        linked_concepts=linked[:10],
                    )
                )
        goals.sort(key=lambda g: g.priority, reverse=True)

        # 2. Strategic priorities from high-confidence decisions + principles
        priority_types = {"decision", "principle", "constraint"}
        priority_candidates = [c for c in concepts if c.concept_type in priority_types and c.confidence >= 0.55]
        priority_candidates.sort(key=lambda c: c.confidence, reverse=True)
        strategic_priorities = [
            StrategicPriority(
                concept_id=c.id,
                summary=c.summary[:120],
                confidence=round(c.confidence, 3),
                source_type=c.concept_type or "decision",
            )
            for c in priority_candidates[:5]
        ]

        # 3. Curiosity frontier from question queue + weakest areas
        import app.features.question_queue as question_queue

        raw_q = question_queue.get_questions(limit=3)
        frontier = [
            CuriosityFrontierItem(
                gap_description=q.get("question", ""),
                priority_score=q.get("priority", 0.0),
            )
            for q in raw_q
        ]

        # If no questions queued, synthesize frontier from weakest areas
        if not frontier:
            area_stats: dict[str, list] = {}
            for c in concepts:
                ka = (c.metadata or {}).get("knowledge_area", "unknown")
                if ka not in area_stats:
                    area_stats[ka] = []
                area_stats[ka].append(c.confidence)

            weak_areas = [
                (ka, sum(confs) / len(confs), len(confs))
                for ka, confs in area_stats.items()
                if len(confs) >= 3  # only areas with enough concepts
            ]
            weak_areas.sort(key=lambda x: x[1])  # lowest avg confidence first
            frontier = [
                CuriosityFrontierItem(
                    gap_description=f"Knowledge area '{wa[0]}' has low avg confidence ({wa[1]:.2f} across {wa[2]} concepts)",
                    priority_score=round(1.0 - wa[1], 2),
                )
                for wa in weak_areas[:3]
            ]

        # 4. Recommended actions from recent high-salience unresolved patterns
        actions = []
        recent_patterns = [
            c
            for c in concepts
            if c.concept_type in {"pattern", "observation"} and c.confidence >= 0.6 and c.salience and c.salience >= 0.5
        ]
        recent_patterns.sort(key=lambda c: (c.salience or 0) * c.confidence, reverse=True)
        for c in recent_patterns[:3]:
            actions.append(
                RecommendedAction(
                    description=f"Address: {c.summary[:80]}",
                    rationale=f"High salience ({c.salience:.2f}) + confidence ({c.confidence:.2f})",
                    priority=round((c.salience or 0.5) * c.confidence, 2),
                )
            )

        return ActiveDirectionality(
            active_goals=goals[:5],
            strategic_priorities=strategic_priorities,
            curiosity_frontier=frontier,
            next_recommended_actions=actions,
        )

    def _compute_open_threads(self) -> list:
        """Wave 5: Compute open thread summaries for orientation."""
        try:
            from app.features.threads import compute_open_threads

            summaries = compute_open_threads()
            return [s.model_dump() for s in summaries]
        except (ImportError, Exception) as e:
            logger.debug(f"Wave 5: open_threads skipped: {e}")
            return []

    def _compute_workstreams_status(
        self,
        origin_id: str | None = None,
        session_id: str | None = None,
        limit: int = 8,
    ) -> dict:
        """OPS-517: compact Workstreams status for explicit pith_orient calls only."""
        try:
            from app.features.threads import (
                _load_workstream_skip,
                _split_workstream_activation_candidates,
                classify_workstream_threads,
                load_active_workstream_binding_checkpoint,
            )

            effective_limit = max(1, min(int(limit), 25))
            classified = classify_workstream_threads(
                include_maintenance=False,
                limit=effective_limit,
            )
            split = _split_workstream_activation_candidates(classified.get("threads") or [])
            candidate_counts = split.get("counts", {})
            active_checkpoint = (
                load_active_workstream_binding_checkpoint(origin_id=origin_id, session_id=session_id)
                if origin_id or session_id
                else None
            )
            if active_checkpoint:
                context = active_checkpoint.get("context", {})
                active_binding = {
                    "status": active_checkpoint.get("status"),
                    "binding_source": active_checkpoint.get("selection_source"),
                    "binding_status": active_checkpoint.get("selection_source"),
                    "thread_id": context.get("workstream_thread_id"),
                    "checkpoint_task_id": active_checkpoint.get("task_id"),
                }
            else:
                active_binding = None
            explicit_skip_checkpoint = (
                _load_workstream_skip(origin_id=origin_id, session_id=session_id, current_task_id=None)
                if origin_id or session_id
                else None
            )
            explicit_skip = (
                {
                    "status": explicit_skip_checkpoint.get("status"),
                    "checkpoint_task_id": explicit_skip_checkpoint.get("task_id"),
                    "skip_reason": (explicit_skip_checkpoint.get("context") or {}).get("skip_reason"),
                    "current_task_id": (explicit_skip_checkpoint.get("context") or {}).get("current_task_id"),
                }
                if explicit_skip_checkpoint
                else None
            )
            next_operator_action = "bind_or_skip"
            if active_binding:
                next_operator_action = "continue_bound_workstream"
            elif explicit_skip:
                next_operator_action = "skip_recorded"
            return {
                "status": "ok",
                "generated_at": _utc_now().isoformat(),
                "scope": "status_only",
                "read_only": True,
                "classes": classified.get("classes", {}),
                "total": classified.get("total", 0),
                "returned": len(classified.get("threads") or []),
                "truncated": classified.get("truncated", False),
                "active_binding": active_binding,
                "explicit_skip": explicit_skip,
                "candidate_counts": candidate_counts,
                "proof_candidate_count": candidate_counts.get("proof_or_maintenance", 0),
                "next_operator_action": next_operator_action,
                "threads": classified.get("threads", []),
                "caveats": [
                    "codex_direct_mcp_transport_unproven",
                    "status_only_not_instruction_authority",
                ],
            }
        except Exception as e:
            logger.debug(f"OPS-517: workstreams status skipped: {e}")
            return {"status": "error", "scope": "status_only", "read_only": True, "error": type(e).__name__}

    def _compute_experiment_summary(self) -> dict | None:
        """Wave 6: Compute active experiments summary for orientation."""
        try:
            from app.features.experiments import load_experiments

            active_experiments = load_experiments(status=["reasoning", "completed"], limit=5)
            if not active_experiments:
                return None
            return {
                "active_count": len([e for e in active_experiments if e.status == "reasoning"]),
                "recent_completed": len([e for e in active_experiments if e.status == "completed"]),
                "types_active": list(set(e.experiment_type for e in active_experiments)),
                "concepts_produced_total": sum(len(e.concept_ids_produced) for e in active_experiments),
            }
        except (ImportError, Exception) as e:
            logger.debug(f"Wave 6: experiment_summary skipped: {e}")
            return None
