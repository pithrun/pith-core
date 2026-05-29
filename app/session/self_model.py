"""SelfModel manager — cognitive self-assessment singleton.

Phase 1A D5: Computes Pith's model of itself from live concept data.
Real data for tool_capabilities, cognitive_operations, capacity_limits,
knowledge_distribution, knowledge_health, and identity_continuity.
Stub data for confidence_calibration, blind_spots, tendencies, error_history.

Key design: generate() accepts pre-loaded concepts from the reflection cycle
to avoid double-scanning storage and prevent race conditions with forgetting.
"""

import hashlib
import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta

from app.core.config import BENCHMARK_READONLY
from app.core.datetime_utils import _ensure_aware, _utc_now, _utc_now_iso
from app.core.models import (
    BlindSpot,
    CalibrationBin,
    CapacityLimits,
    CognitiveCapabilityInventory,
    CognitiveMilestone,
    CognitiveOperation,
    CognitiveTendencies,
    Concept,
    ConfidenceCalibration,
    ConfidenceProfile,
    CorrectionRecord,
    EpistemicProfile,
    ErrorHistory,
    GranularityProfile,
    IdentityContinuity,
    IntrospectHealth,
    IntrospectIdentity,
    IntrospectSummary,
    KnowledgeAreaProfile,
    KnowledgeHealth,
    LearningVelocity,
    LinkingProfile,
    RecurringPattern,
    SelfModel,
    ToolCapability,
)
from app.storage import count_orphan_concepts, count_sessions, load_associations, load_self_model, save_self_model

logger = logging.getLogger(__name__)

# MCP tool registry — what this pith can do
MCP_TOOLS = [
    ToolCapability(
        tool_name="pith_search",
        operational=True,
        performance_grade=0.7,
        known_limitations=[
            "TF-IDF misses semantic similarity",
            "0.1 score threshold filters valid results in small corpus",
        ],
    ),
    ToolCapability(tool_name="pith_get_concept", operational=True, performance_grade=0.9),
    ToolCapability(
        tool_name="pith_related_concepts",
        operational=True,
        performance_grade=0.7,
        known_limitations=["Association graph sparsely populated"],
    ),
    ToolCapability(tool_name="pith_propose_concept", operational=True, performance_grade=0.9),
    ToolCapability(tool_name="pith_evolve_concept", operational=True, performance_grade=0.9),
    ToolCapability(tool_name="pith_link_concepts", operational=True, performance_grade=0.8),
    ToolCapability(tool_name="pith_reflect", operational=True, performance_grade=0.8),
    ToolCapability(tool_name="pith_health", operational=True, performance_grade=0.9),
    ToolCapability(tool_name="pith_stats", operational=True, performance_grade=0.9),
    ToolCapability(
        tool_name="pith_search (curiosity)",
        operational=True,
        performance_grade=0.6,
        known_limitations=["Priority scoring is basic"],
    ),
    ToolCapability(tool_name="pith_questions", operational=True, performance_grade=0.7),
    ToolCapability(
        tool_name="pith_activate_context",
        operational=True,
        performance_grade=0.6,
        known_limitations=["Activation boost is session-scoped, not persisted"],
    ),
    ToolCapability(
        tool_name="pith_set_goal",
        operational=True,
        performance_grade=0.5,
        known_limitations=["Goal-directed retrieval not yet fully wired"],
    ),
    ToolCapability(tool_name="pith_recover", operational=True, performance_grade=0.9),
    ToolCapability(
        tool_name="pith_import_conversation",
        operational=True,
        performance_grade=0.6,
        known_limitations=["Extraction is regex-based, not model-driven"],
    ),
]

# Cognitive operations — maturity: experimental | operational | proven
COGNITIVE_OPERATIONS = [
    CognitiveOperation(operation="concept_creation", maturity="proven", success_rate=0.95),
    CognitiveOperation(operation="concept_evolution", maturity="proven", success_rate=0.90),
    CognitiveOperation(operation="reflection_cycle", maturity="proven", success_rate=0.95),
    CognitiveOperation(operation="time_decay", maturity="proven", success_rate=0.95),
    CognitiveOperation(operation="access_strengthening", maturity="proven", success_rate=0.90),
    CognitiveOperation(
        operation="deduplication_merge",
        maturity="operational",
        success_rate=0.85,
        failure_patterns=["Legacy string evidence gets flat 0.448 strength estimate"],
    ),
    CognitiveOperation(
        operation="confidence_recalibration",
        maturity="operational",
        success_rate=0.85,
        failure_patterns=["Aggressive recency decay on old evidence"],
    ),
    CognitiveOperation(
        operation="forgetting",
        maturity="operational",
        success_rate=0.90,
        failure_patterns=["Pre-tracking concepts skipped (bootstrap protection)"],
    ),
    CognitiveOperation(operation="curiosity_generation", maturity="operational", success_rate=0.70),
    CognitiveOperation(
        operation="predictive_activation",
        maturity="experimental",
        success_rate=0.50,
        failure_patterns=["Activation boost not persisted across sessions"],
    ),
    CognitiveOperation(
        operation="goal_directed_retrieval",
        maturity="experimental",
        success_rate=0.40,
        failure_patterns=["Goal context not deeply integrated into search ranking"],
    ),
]


class SelfModelManager:
    """Manages Pith's self-model singleton.

    generate() accepts pre-loaded concepts to avoid double-scanning storage
    and prevent race conditions with forgetting in the same reflection cycle.
    """

    def generate(self, concepts: list[Concept]) -> SelfModel:
        """Compute complete self-model from live concept data.

        Args:
            concepts: Pre-loaded list of active concepts (from reflection cycle
                      or explicit load). Avoids re-reading from storage.

        Returns:
            SelfModel with all real metrics computed and stubs defaulted.
        """
        now = _utc_now_iso()

        # Load previous version for hash chain
        previous = self.load()
        parent_hash = previous.content_hash if previous else ""
        new_version = (previous.version + 1) if previous else 1

        capabilities = self._compute_capabilities(concepts)
        epistemic = self._compute_epistemic_profile(concepts)
        identity = self._compute_identity(concepts)

        self_model = SelfModel(
            model_id="self",
            version=new_version,
            updated_at=now,
            updated_by="pith_deterministic",
            capabilities=capabilities,
            epistemic_profile=epistemic,
            tendencies=self._compute_tendencies(concepts),
            error_history=self._compute_error_history(concepts),
            identity=identity,
            parent_hash=parent_hash,
        )

        # Compute content hash for integrity lineage
        self_model.content_hash = self._compute_hash(self_model)

        if BENCHMARK_READONLY:
            logger.info("BENCH-INFRA-009: SelfModel persistence skipped (PITH_BENCHMARK_READONLY)")
        else:
            self.save(self_model)
        logger.info(
            f"SelfModel v{new_version} generated: {len(concepts)} concepts, hash={self_model.content_hash[:12]}..."
        )
        return self_model

    def get_blind_spots(self) -> list[BlindSpot]:
        """Return most recently computed blind spots (cached from last reflection).

        Does NOT trigger recomputation. Returns empty list if self_model
        hasn't been generated yet this session (cold-start case).
        Coverage confidence (Fix 1a) works independently — this is a BONUS signal.

        Budget: <0.5ms (reads from storage, no computation)
        """
        try:
            model = self.load()
            if model and model.epistemic_profile:
                return model.epistemic_profile.blind_spots or []
        except Exception as e:
            logger.warning(f"get_blind_spots failed (non-fatal): {e}")
        return []

    def _compute_capabilities(self, concepts: list[Concept]) -> CognitiveCapabilityInventory:
        """Compute tool capabilities, cognitive operations, and capacity limits."""
        return CognitiveCapabilityInventory(
            tool_capabilities=MCP_TOOLS,
            cognitive_operations=COGNITIVE_OPERATIONS,
            capacity_limits=CapacityLimits(
                concept_count=len(concepts),
                concept_ceiling_estimate=10000,
                retrieval_latency_ms=50.0,  # Estimated from TF-IDF index
                reflection_duration_ms=2000.0,  # Estimated from full cycle
            ),
        )

    def _compute_epistemic_profile(self, concepts: list[Concept]) -> EpistemicProfile:
        """Compute knowledge distribution and health metrics."""
        if not concepts:
            return EpistemicProfile()

        # Group by knowledge_area
        areas: dict[str, list[Concept]] = defaultdict(list)
        for c in concepts:
            area = (c.metadata or {}).get("knowledge_area", "unknown")
            areas[area].append(c)

        # Build per-area profiles
        distribution = []
        for area_name, area_concepts in sorted(areas.items()):
            confidences = [c.confidence for c in area_concepts]
            stabilities = [c.stability for c in area_concepts]

            # Evidence coverage: % of concepts with at least one non-string evidence
            structured_count = sum(1 for c in area_concepts if any(not isinstance(e, str) for e in c.evidence))
            evidence_coverage = structured_count / len(area_concepts) if area_concepts else 0.0

            # Last activity: most recent created_at or last_accessed
            last_times = []
            for c in area_concepts:
                if c.last_accessed:
                    last_times.append(c.last_accessed)
                elif c.created_at:
                    last_times.append(c.created_at)
            last_activity = max(last_times) if last_times else None

            distribution.append(
                KnowledgeAreaProfile(
                    knowledge_area=area_name,
                    concept_count=len(area_concepts),
                    avg_confidence=sum(confidences) / len(confidences),
                    avg_stability=sum(stabilities) / len(stabilities),
                    evidence_coverage=evidence_coverage,
                    last_activity=last_activity,
                )
            )

        # Knowledge health metrics
        health = self._compute_knowledge_health(concepts)

        # Wave 4b: Real calibration + blind spots
        calibration = self._compute_calibration(concepts)
        blind_spots = self._compute_blind_spots(concepts, areas)

        return EpistemicProfile(
            knowledge_distribution=distribution,
            confidence_calibration=calibration,
            blind_spots=blind_spots,
            knowledge_health=health,
        )

    def _compute_calibration(self, concepts: list[Concept]) -> ConfidenceCalibration:
        """Compute confidence calibration from prediction tracking data (Wave 4b).

        Uses SQL aggregation [FIX S1]. ECE guard: not computed until >50 predictions [FIX CS1].
        """
        try:
            from app.core.config import CALIBRATION_MIN_PREDICTIONS_FOR_ECE, CALIBRATION_UNTESTED_RATIO_THRESHOLD
            from app.ops.traces import get_calibration_data
        except ImportError:
            return ConfidenceCalibration()

        cal_data = get_calibration_data()
        total = cal_data.get("total_predictions", 0)
        with_outcomes = cal_data.get("predictions_with_outcomes", 0)
        bins_data = cal_data.get("bins", [])

        # Build CalibrationBin objects
        bins = [CalibrationBin(**b) for b in bins_data]

        # ECE computation [FIX CS1]
        ece_computable = with_outcomes >= CALIBRATION_MIN_PREDICTIONS_FOR_ECE
        ece = 0.0
        if ece_computable:
            total_in_bins = sum(b.prediction_count for b in bins)
            if total_in_bins > 0:
                ece = sum(b.prediction_count * b.gap for b in bins) / total_in_bins

        # Test-status distribution
        test_dist = {"corrected": 0, "untested": 0, "lightly_tested": 0, "validated": 0}
        try:
            from app.retrieval.provenance import get_test_status_label

            for c in concepts:
                label = get_test_status_label(c)
                test_dist[label] = test_dist.get(label, 0) + 1
        except ImportError:
            pass

        # Calibration maturity
        total_concepts = len(concepts)
        if total_concepts > 0:
            untested_ratio = test_dist.get("untested", 0) / total_concepts
        else:
            untested_ratio = 1.0

        if total_concepts < 20:
            maturity = "insufficient_testing"
        elif untested_ratio > CALIBRATION_UNTESTED_RATIO_THRESHOLD:
            maturity = "developing"
        else:
            maturity = "mature"

        # Over/under confidence areas
        overconf = []
        underconf = []
        for b in bins:
            if b.prediction_count >= 5 and b.gap > 0.15:
                if b.avg_predicted > b.avg_actual:
                    overconf.append(f"{b.bin_lower:.1f}-{b.bin_upper:.1f}")
                else:
                    underconf.append(f"{b.bin_lower:.1f}-{b.bin_upper:.1f}")

        return ConfidenceCalibration(
            total_predictions_logged=total,
            predictions_with_outcomes=with_outcomes,
            calibration_bins=bins,
            expected_calibration_error=round(ece, 4),
            ece_computable=ece_computable,
            test_status_distribution=test_dist,
            calibration_maturity=maturity,
            overconfidence_areas=overconf,
            underconfidence_areas=underconf,
            calibration_method="prediction_tracking_v1",
            last_calibrated=_utc_now_iso(),
        )

    def _compute_blind_spots(self, concepts: list[Concept], areas: dict[str, list[Concept]]) -> list[BlindSpot]:
        """Compute blind spots with 3 detection rules (Wave 4b).

        Rules:
        1. Knowledge areas referenced in associations but <3 concepts
        2. Areas with 5+ concepts but avg confidence <0.4
        3. High-confidence concepts with zero associations (isolated knowledge)

        Guards: Returns empty if <20 concepts [FIX CS2]. Capped at 10 alerts.
        Pre-builds area_lookup dict from loaded concepts [FIX S2].
        """
        try:
            from app.core.config import BLIND_SPOT_MIN_CONCEPTS
        except ImportError:
            BLIND_SPOT_MIN_CONCEPTS = 20

        if len(concepts) < BLIND_SPOT_MIN_CONCEPTS:
            return []

        blind_spots = []

        # Rule 1: Areas referenced in associations but with <3 concepts
        graph = load_associations()
        referenced_areas = set()
        concept_area_map = {}
        for c in concepts:
            area = (c.metadata or {}).get("knowledge_area", "unknown")
            concept_area_map[c.id] = area

        for source_id, relations in graph.items():
            if isinstance(relations, dict):
                for targets in relations.values():
                    if isinstance(targets, list):
                        for tid in targets:
                            if tid in concept_area_map:
                                referenced_areas.add(concept_area_map[tid])

        for area in referenced_areas:
            area_concepts = areas.get(area, [])
            if 0 < len(area_concepts) < 3:
                blind_spots.append(
                    BlindSpot(
                        description=f"Sparse area '{area}': referenced in associations but only {len(area_concepts)} concept(s)",
                        severity="moderate",
                        detected_by="reflection",
                        detected_at=_utc_now_iso(),
                    )
                )

        # Rule 2: Areas with 5+ concepts but avg confidence <0.4
        for area_name, area_concepts in areas.items():
            if len(area_concepts) >= 5:
                avg_conf = sum(c.confidence for c in area_concepts) / len(area_concepts)
                if avg_conf < 0.4:
                    blind_spots.append(
                        BlindSpot(
                            description=f"Low-confidence area '{area_name}': {len(area_concepts)} concepts, avg confidence {avg_conf:.2f}",
                            severity="moderate",
                            detected_by="reflection",
                            detected_at=_utc_now_iso(),
                        )
                    )

        # Rule 3: High-confidence concepts with zero associations
        linked_ids = set(graph.keys())
        for relations in graph.values():
            if isinstance(relations, dict):
                for targets in relations.values():
                    if isinstance(targets, list):
                        linked_ids.update(targets)

        isolated_high = [c for c in concepts if c.confidence >= 0.7 and c.id not in linked_ids]
        if len(isolated_high) >= 3:
            blind_spots.append(
                BlindSpot(
                    description=f"{len(isolated_high)} high-confidence concepts with zero associations (isolated knowledge)",
                    severity="minor",
                    detected_by="reflection",
                    detected_at=_utc_now_iso(),
                )
            )

        return blind_spots[:10]  # Cap at 10

    def _compute_tendencies(self, concepts: list[Concept]) -> CognitiveTendencies:
        """Compute cognitive tendencies from concept data (Wave 4b).

        Replaces all CognitiveTendencies stubs with real computation:
        - Granularity: avg summary length → over_granular|balanced|under_granular
        - Linking: associations per concept → over_linked|balanced|isolated
        - Confidence: initial confidence avg → cautious|calibrated|overconfident
        - Learning velocity: concepts per session, new-vs-evolve ratio
        """
        if not concepts:
            return CognitiveTendencies()

        # Granularity
        summary_lengths = [len(c.summary) for c in concepts]
        avg_summary_len = sum(summary_lengths) / len(summary_lengths)
        if avg_summary_len > 300:
            gran_tendency = "over_granular"
        elif avg_summary_len < 80:
            gran_tendency = "under_granular"
        else:
            gran_tendency = "balanced"

        # Linking
        graph = load_associations()
        assoc_counts = []
        linked_ids = set(graph.keys())
        for c in concepts:
            count = 0
            if c.id in graph:
                relations = graph[c.id]
                if isinstance(relations, dict):
                    for targets in relations.values():
                        if isinstance(targets, list):
                            count += len(targets)
            assoc_counts.append(count)

        avg_assoc = sum(assoc_counts) / len(assoc_counts) if assoc_counts else 0
        # Cross-domain: associations between different knowledge areas
        cross_domain = 0
        total_links = 0
        concept_area = {c.id: (c.metadata or {}).get("knowledge_area", "unknown") for c in concepts}
        for source_id, relations in graph.items():
            if isinstance(relations, dict):
                for targets in relations.values():
                    if isinstance(targets, list):
                        for tid in targets:
                            total_links += 1
                            if source_id in concept_area and tid in concept_area:
                                if concept_area[source_id] != concept_area[tid]:
                                    cross_domain += 1
        cross_ratio = cross_domain / total_links if total_links > 0 else 0.0

        if avg_assoc > 5:
            link_tendency = "over_linked"
        elif avg_assoc < 1:
            link_tendency = "isolated"
        else:
            link_tendency = "balanced"

        # Confidence
        confidences = [c.confidence for c in concepts]
        avg_conf = sum(confidences) / len(confidences)
        # Check evolution deltas (version > 1 concepts)
        evolved = [c for c in concepts if c.version and c.version != "v1"]
        if avg_conf > 0.75:
            conf_tendency = "overconfident"
        elif avg_conf < 0.35:
            conf_tendency = "cautious"
        else:
            conf_tendency = "calibrated"

        # Learning velocity
        total_sessions = max(count_sessions(), 1)
        new_count = sum(1 for c in concepts if c.version in ("v1", "1"))
        evolve_count = len(concepts) - new_count
        concepts_per_session = len(concepts) / total_sessions
        new_vs_evolve = new_count / max(evolve_count, 1)

        return CognitiveTendencies(
            granularity_profile=GranularityProfile(
                avg_concepts_per_learning_event=concepts_per_session,
                avg_summary_length=round(avg_summary_len, 1),
                tendency=gran_tendency,
            ),
            linking_profile=LinkingProfile(
                avg_associations_per_concept=round(avg_assoc, 2),
                cross_domain_link_ratio=round(cross_ratio, 3),
                tendency=link_tendency,
            ),
            confidence_profile=ConfidenceProfile(
                initial_confidence_avg=round(avg_conf, 3),
                evolution_confidence_delta_avg=0.0,  # Would need per-version tracking
                tendency=conf_tendency,
            ),
            learning_velocity=LearningVelocity(
                concepts_per_session_avg=round(concepts_per_session, 2),
                evolution_events_per_session_avg=round(evolve_count / total_sessions, 2),
                new_vs_evolve_ratio=round(new_vs_evolve, 2),
            ),
        )

    def _compute_knowledge_health(self, concepts: list[Concept]) -> KnowledgeHealth:
        """Compute global knowledge health metrics."""
        if not concepts:
            return KnowledgeHealth()

        avg_conf = sum(c.confidence for c in concepts) / len(concepts)
        avg_stab = sum(c.stability for c in concepts) / len(concepts)

        # Orphan concepts: use DB query (not JSON blob — CONNECTIVITY-FIX)
        orphan_count = count_orphan_concepts()

        # FRESHNESS_UNIFIED_REDESIGN: Tighten stale window to 7d, remove freshness from health formula
        _stale_cutoff = (_utc_now() - timedelta(days=7)).isoformat()
        stale_count = 0
        for c in concepts:
            _ts = c.last_organic_access or c.last_accessed or c.created_at
            if _ts and _ts < _stale_cutoff:
                stale_count += 1
            elif not _ts:
                stale_count += 1  # No timestamp = stale

        # 4-factor weighted score (freshness removed — observational duplicate of reflection.py)
        n = len(concepts)
        established = sum(1 for c in concepts if getattr(c, "maturity", "") == "ESTABLISHED")
        maturity_health = established / n if n > 0 else 0
        connectivity = (n - orphan_count) / n if n > 0 else 0
        health = 0.35 * avg_conf + 0.35 * avg_stab + 0.15 * maturity_health + 0.15 * connectivity

        return KnowledgeHealth(
            total_concepts=n,
            avg_confidence=round(avg_conf, 4),
            avg_stability=round(avg_stab, 4),
            contradiction_density=0.0,
            orphan_concept_count=orphan_count,
            stale_concept_count=stale_count,
            health_score=round(health, 4),
        )

    def _compute_identity(self, concepts: list[Concept]) -> IdentityContinuity:
        """Compute identity continuity — what persists across model swaps."""
        if not concepts:
            return IdentityContinuity()

        # Pith birth = earliest concept creation
        created_dates = [c.created_at for c in concepts if c.created_at]
        pith_created = min(created_dates) if created_dates else None

        # Total learning events = sum of version numbers (each version = one event)
        total_events = sum(int(c.version.lstrip("v")) if c.version.startswith("v") else 1 for c in concepts)

        # Auto-detect milestones
        milestones = []
        count = len(concepts)
        for threshold, desc, sig in [
            (50, "Reached 50 concepts", "minor"),
            (100, "Reached 100 concepts", "moderate"),
            (200, "Reached 200 concepts", "major"),
            (500, "Reached 500 concepts", "major"),
        ]:
            if count >= threshold:
                milestones.append(
                    CognitiveMilestone(
                        milestone_id=f"concept_count_{threshold}",
                        description=desc,
                        significance=sig,
                    )
                )

        # Knowledge area diversity milestone
        areas = set((c.metadata or {}).get("knowledge_area", "unknown") for c in concepts)
        if len(areas) >= 5:
            milestones.append(
                CognitiveMilestone(
                    milestone_id="knowledge_diversity_5",
                    description=f"Knowledge spans {len(areas)} areas",
                    significance="moderate",
                )
            )

        return IdentityContinuity(
            pith_created_at=pith_created,
            total_sessions=count_sessions(),
            total_learning_events=total_events,
            model_history=[],  # Not yet tracked
            cognitive_milestones=milestones,
            current_strategic_focus="Phase 1A: Cognitive layer implementation",
            current_maturity_stage="developing",
        )

    def _compute_error_history(self, concepts: list[Concept]) -> ErrorHistory:
        """Compute error history from corrections DB table.

        Populates CorrectionRecords and detects RecurringPatterns
        (3+ corrections in same domain within 30 days → pattern).
        """
        from app.storage import read_snapshot_db

        corrections: list[CorrectionRecord] = []
        patterns: list[RecurringPattern] = []

        try:
            with read_snapshot_db("compute_error_history") as conn:
                rows = conn.execute(
                    """SELECT c.id, c.concept_id, c.correction_type,
                              COALESCE(c.corrected_claim, '') || ' → ' || COALESCE(c.correct_claim, '') AS correction_text,
                              json_extract(co.data, '$.metadata.knowledge_area') AS knowledge_area,
                              c.created_at
                       FROM corrections c
                       LEFT JOIN concepts co ON c.concept_id = co.id
                       ORDER BY c.created_at DESC
                       LIMIT 100"""
                ).fetchall()

                for row in rows:
                    corrections.append(
                        CorrectionRecord(
                            error_id=str(row[0]),
                            concept_id=row[1] or "",
                            error_type=row[2] or "factual",
                            description=(row[3] or "")[:200],
                            corrected_at=row[5],
                            corrected_by="user",
                        )
                    )

                # Detect recurring patterns: 3+ corrections in same knowledge_area
                # within last 30 days
                area_rows = conn.execute(
                    """SELECT COALESCE(json_extract(co.data, '$.metadata.knowledge_area'), 'general') AS ka,
                              COUNT(*) as cnt
                       FROM corrections c
                       LEFT JOIN concepts co ON c.concept_id = co.id
                       WHERE c.created_at >= datetime('now', '-30 days')
                       GROUP BY ka
                       HAVING cnt >= 3
                       ORDER BY cnt DESC"""
                ).fetchall()

                for area_name, count in area_rows:
                    patterns.append(
                        RecurringPattern(
                            pattern_id=f"pattern_{area_name}",
                            description=f"Recurring corrections in {area_name} ({count} in 30 days)",
                            frequency=count,
                            severity="high" if count >= 5 else "medium",
                            mitigation=f"Review {area_name} concepts for accuracy",
                        )
                    )

        except Exception as e:
            logger.warning(f"Error computing error history: {e}")

        total = len(corrections)
        total_concepts = len(concepts) if concepts else 1

        # Find most common error type
        type_counts: dict[str, int] = {}
        for c in corrections:
            type_counts[c.error_type] = type_counts.get(c.error_type, 0) + 1
        most_common = max(type_counts, key=type_counts.get) if type_counts else "none"

        return ErrorHistory(
            corrections=corrections[:20],  # Cap at 20 for serialization
            recurring_patterns=patterns,
            total_corrections=total,
            correction_rate=round(total / total_concepts * 100, 2),
            most_common_error_type=most_common,
        )

    def _compute_hash(self, self_model: SelfModel) -> str:
        """SHA-256 content hash for integrity lineage."""
        # Hash everything except content_hash and parent_hash
        data = self_model.model_dump(exclude={"content_hash", "parent_hash"})
        canonical = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def load(self) -> SelfModel | None:
        """Load current self-model from storage."""
        data = load_self_model()
        if data is None:
            return None
        try:
            return SelfModel(**data)
        except Exception as e:
            logger.warning(f"Failed to load SelfModel: {e}")
            return None

    def save(self, self_model: SelfModel) -> None:
        """Save self-model to storage with version history."""
        data = self_model.model_dump()
        save_self_model(data)

    def introspect(
        self,
        mode: str = "summary",
        update: bool = False,
        concepts: list[Concept] | None = None,
        generate_if_missing: bool = True,
    ):
        """Return cognitive self-assessment.

        Args:
            mode: summary | full | capability_check | epistemic_check
            update: If True, recompute SelfModel before returning.
                    Latency follows 'full' target (500ms) regardless of mode.
            concepts: Pre-loaded concepts for update=True. If None and update=True,
                      concepts will be loaded from storage.
            generate_if_missing: If False, return an explicit not-ready payload
                                 instead of scanning storage to build a missing
                                 self-model. Used by latency-sensitive bootstrap
                                 paths.

        Returns:
            Mode-appropriate slice of the SelfModel.
        """
        # Get or generate SelfModel
        sm = self.load()

        if update or (sm is None and generate_if_missing):
            # Need concepts to generate
            if concepts is None:
                from app.storage import list_concepts, load_concept

                concepts = []
                for cid in list_concepts():
                    c = load_concept(cid, track_access=False)
                    if c:
                        concepts.append(c)
            sm = self.generate(concepts)
        elif sm is None:
            if mode == "summary":
                return {
                    "status": "not_ready",
                    "reason": "self_model_cache_missing",
                }
            return {}

        # Return mode-appropriate slice
        if mode == "full":
            return sm.model_dump()

        elif mode == "capability_check":
            return sm.capabilities.model_dump()

        elif mode == "epistemic_check":
            return sm.epistemic_profile.model_dump()

        elif mode == "summary":
            return self._build_summary(sm)

        else:
            raise ValueError(f"Unknown introspect mode: {mode}")

    def _build_summary(self, sm: SelfModel) -> dict:
        """Build introspect summary response from SelfModel."""
        # Identity slice
        pith_age_days = 0
        if sm.identity.pith_created_at:
            try:
                created = datetime.fromisoformat(sm.identity.pith_created_at)
                pith_age_days = (_utc_now() - _ensure_aware(created)).days
            except (ValueError, TypeError):
                pass

        identity = IntrospectIdentity(
            pith_age_days=pith_age_days,
            total_sessions=sm.identity.total_sessions,
            current_maturity=sm.identity.current_maturity_stage,
            current_focus=sm.identity.current_strategic_focus,
        )

        # Health slice
        kh = sm.epistemic_profile.knowledge_health
        health = IntrospectHealth(
            concept_count=kh.total_concepts,
            avg_confidence=round(kh.avg_confidence, 4),
            contradiction_density=kh.contradiction_density,
        )

        # Top strengths: top 3 knowledge areas by avg_confidence
        dist = sorted(
            sm.epistemic_profile.knowledge_distribution,
            key=lambda d: d.avg_confidence,
            reverse=True,
        )
        top_strengths = [
            f"{d.knowledge_area} ({d.concept_count} concepts, conf={d.avg_confidence:.2f})" for d in dist[:3]
        ]

        # Weakest areas: bottom 3 by avg_confidence (Phase 1A fallback for top_gaps)
        weakest = (
            [f"{d.knowledge_area} ({d.concept_count} concepts, conf={d.avg_confidence:.2f})" for d in dist[-3:]]
            if len(dist) >= 3
            else [
                f"{d.knowledge_area} ({d.concept_count} concepts, conf={d.avg_confidence:.2f})" for d in reversed(dist)
            ]
        )

        # Recent errors: last 3 corrections (STUB — empty in Phase 1A)
        recent_errors = [c.description for c in sm.error_history.corrections[-3:]]

        summary = IntrospectSummary(
            identity=identity,
            health=health,
            top_strengths=top_strengths,
            weakest_areas=weakest,
            recent_errors=recent_errors,
        )
        return summary.model_dump()


# Global instance
self_model_manager = SelfModelManager()


def update_self_model(conn=None) -> SelfModel | None:
    """Convenience function for async_tasks.py — regenerate self-model from current concepts."""
    from app.storage import list_concepts, load_concept

    try:
        concept_ids = list_concepts()
        concepts = []
        for cid in concept_ids:
            c = load_concept(cid, track_access=False)
            if c:
                concepts.append(c)
        return self_model_manager.generate(concepts)
    except Exception as e:
        logger.warning(f"Self-model update failed: {e}")
        return None
