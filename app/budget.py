"""Context Budget Governance — dynamic allocation based on epistemic quality.

Implements 4-tier budget allocation with two-pass assembly:
  Tier 1 (GUARANTEED): Constraints + active decisions (authority >= 0.80)
  Tier 2 (PRIORITY): Directives + goals + skills + pinned context
  Tier 3 (FILL): Context + background from retrieval
  Tier 4 (OVERFLOW): Compressed one-liner summaries

Two-pass design:
  Pass 1 (S3): Preliminary tier caps from retrieval results
  Pass 2 (S5): Final allocation after all injection phases complete
"""

import logging
from typing import Any

from app.config import (
    CONTEXT_BUDGET_MAIN,
    OVERFLOW_SUMMARY_MAX,
    PIN_BUDGET,
    PRESENTATION_CONSTRAINT,
    PRESENTATION_DIRECTIVE,
    TIER_FILL,
    TIER_GUARANTEED,
    TIER_OVERFLOW,
    TIER_PRIORITY,
)
from app.constants import GOV_EVENT_BUDGET_ALLOCATED
from app.governance_context import GovernanceContext, ScoredConcept

logger = logging.getLogger(__name__)


class BudgetAllocation:
    """Result of budget allocation — which concepts go in which tier."""

    def __init__(self, total_slots: int = CONTEXT_BUDGET_MAIN):
        self.total_slots = total_slots
        self.tiers: dict[str, list[str]] = {
            TIER_GUARANTEED: [],
            TIER_PRIORITY: [],
            TIER_FILL: [],
            TIER_OVERFLOW: [],
        }
        self.overflow_summaries: list[str] = []
        self.rejected: list[tuple[str, str]] = []  # (concept_id, reason)

    @property
    def total_allocated(self) -> int:
        return sum(len(v) for k, v in self.tiers.items() if k != TIER_OVERFLOW)

    @property
    def remaining_slots(self) -> int:
        return max(0, self.total_slots - self.total_allocated)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_slots": self.total_slots,
            "allocated": self.total_allocated,
            "tiers": {k: list(v) for k, v in self.tiers.items()},
            "overflow_count": len(self.overflow_summaries),
            "rejected_count": len(self.rejected),
        }


def allocate_budget(
    scored_concepts: list[ScoredConcept],
    gov_ctx: GovernanceContext,
    always_activate_ids: list[str] = None,
    query_knowledge_areas: list[str] = None,
    total_slots: int = CONTEXT_BUDGET_MAIN,
) -> BudgetAllocation:
    """Two-pass budget allocation (this is Pass 2 — final allocation).

    Called at S5 position after all injection phases are complete.
    Receives the full candidate pool: retrieval + graph walk + shadow +
    ambient principles + always-activate + firmware + skills.

    Args:
        scored_concepts: All candidate concepts with governance scores
        gov_ctx: GovernanceContext accumulating pipeline state
        always_activate_ids: Concept IDs that must always be included
        query_knowledge_areas: Knowledge areas relevant to current query
        total_slots: Total concept budget

    Returns:
        BudgetAllocation with tier assignments
    """
    always_activate_ids = always_activate_ids or []
    query_knowledge_areas = query_knowledge_areas or []
    alloc = BudgetAllocation(total_slots=total_slots)

    # Sort by final_score descending for priority ordering
    sorted_concepts = sorted(scored_concepts, key=lambda c: c.final_score, reverse=True)

    # Track allocated concept IDs to prevent duplicates
    allocated_ids = set()

    # --- TIER 1: GUARANTEED (constraints + always-activate) ---
    tier1_cap = 8  # Max guaranteed slots

    # Always-activate concepts first (they're guaranteed regardless of score)
    for sc in sorted_concepts:
        if sc.concept_id in always_activate_ids and sc.concept_id not in allocated_ids:
            if len(alloc.tiers[TIER_GUARANTEED]) < PIN_BUDGET:
                alloc.tiers[TIER_GUARANTEED].append(sc.concept_id)
                allocated_ids.add(sc.concept_id)

    # Query-relevant constraints (§8.3: filter by knowledge_area relevance)
    relevant_constraints = 0
    for sc in sorted_concepts:
        if sc.concept_id in allocated_ids:
            continue
        if sc.authority_score >= PRESENTATION_CONSTRAINT and getattr(sc, "concept_type", None) == "constraint":
            # Config fix (RETRIEVAL_ARCHITECTURE_SPEC v1.1): Only constraint-type
            # concepts qualify for GUARANTEED tier via authority. High-authority
            # methods/principles/heuristics go to PRIORITY instead. This prevents
            # 3 non-AA concepts (method_multi_context_docker_build, etc.) from
            # consuming fixed guaranteed slots in every context window.
            #
            # Query-relevance gate: if we know the query's knowledge areas,
            # only allocate guaranteed slots to constraints in those areas.
            # Constraints in unrelated areas go to PRIORITY tier instead.
            if query_knowledge_areas and sc.knowledge_area not in ("unknown", "general", "unclassified"):
                if sc.knowledge_area not in query_knowledge_areas:
                    # Demote to priority tier — still included, just not guaranteed
                    if len(alloc.tiers[TIER_PRIORITY]) < 6:
                        alloc.tiers[TIER_PRIORITY].append(sc.concept_id)
                        allocated_ids.add(sc.concept_id)
                    continue
            if len(alloc.tiers[TIER_GUARANTEED]) < tier1_cap:
                alloc.tiers[TIER_GUARANTEED].append(sc.concept_id)
                allocated_ids.add(sc.concept_id)
                relevant_constraints += 1

    # If constraint overload: reduce to ensure Tier 3 gets >= 5 slots
    if len(alloc.tiers[TIER_GUARANTEED]) > 5 and alloc.remaining_slots < 5:
        excess = len(alloc.tiers[TIER_GUARANTEED]) - 5
        demoted = alloc.tiers[TIER_GUARANTEED][-excess:]
        alloc.tiers[TIER_GUARANTEED] = alloc.tiers[TIER_GUARANTEED][:-excess]
        # Move excess to PRIORITY tier
        for cid in demoted:
            alloc.tiers[TIER_PRIORITY].append(cid)

    # --- TIER 2: PRIORITY (directives + goals + skills) ---
    tier2_cap = min(6, alloc.remaining_slots)

    for sc in sorted_concepts:
        if sc.concept_id in allocated_ids:
            continue
        if len(alloc.tiers[TIER_PRIORITY]) >= tier2_cap:
            break
        if sc.authority_score >= PRESENTATION_DIRECTIVE:
            alloc.tiers[TIER_PRIORITY].append(sc.concept_id)
            allocated_ids.add(sc.concept_id)

    # Skill injections get PRIORITY tier if they fit
    for skill_id in gov_ctx.skill_injections:
        if skill_id not in allocated_ids and len(alloc.tiers[TIER_PRIORITY]) < tier2_cap:
            alloc.tiers[TIER_PRIORITY].append(skill_id)
            allocated_ids.add(skill_id)

    # --- TIER 3: FILL (remaining by score) ---
    for sc in sorted_concepts:
        if sc.concept_id in allocated_ids:
            continue
        if alloc.remaining_slots <= 0:
            break
        alloc.tiers[TIER_FILL].append(sc.concept_id)
        allocated_ids.add(sc.concept_id)

    # --- TIER 4: OVERFLOW (compressed summaries for remaining) ---
    overflow_count = 0
    for sc in sorted_concepts:
        if sc.concept_id in allocated_ids:
            continue
        if overflow_count >= OVERFLOW_SUMMARY_MAX:
            alloc.rejected.append((sc.concept_id, "beyond_overflow_limit"))
            continue
        alloc.overflow_summaries.append(sc.concept_id)
        alloc.tiers[TIER_OVERFLOW].append(sc.concept_id)
        overflow_count += 1

    # Log allocation to governance context
    gov_ctx.budget_allocation = {k: list(v) for k, v in alloc.tiers.items()}
    gov_ctx.overflow_summaries = list(alloc.overflow_summaries)
    gov_ctx.log_event(
        GOV_EVENT_BUDGET_ALLOCATED,
        None,
        {
            "total_candidates": len(scored_concepts),
            "allocated": alloc.total_allocated,
            "tier_counts": {k: len(v) for k, v in alloc.tiers.items()},
            "overflow": len(alloc.overflow_summaries),
            "rejected": len(alloc.rejected),
        },
    )

    logger.info(
        "Budget allocation: %d/%d slots filled (T1:%d T2:%d T3:%d overflow:%d)",
        alloc.total_allocated,
        total_slots,
        len(alloc.tiers[TIER_GUARANTEED]),
        len(alloc.tiers[TIER_PRIORITY]),
        len(alloc.tiers[TIER_FILL]),
        len(alloc.overflow_summaries),
    )

    return alloc
