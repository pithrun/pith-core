"""Goal-directed retrieval v2 — structured scoring for task-based concept activation.

Replaces v1 keyword matching with structured scoring using knowledge_area,
concept_type, recency, and authority signals (all 100% populated in production).

Prototype results (3,138 live concepts):
  - Ship Launch:        30% → 50% target KA hit rate (+20pp)
  - Debug Production:   40% → 90% (+50pp)
  - Competitive Analysis: 20% → 80% (+60pp)
  - Architecture Review: 50% → 90% (+40pp)

See the internal GOAL_DIRECTED_V2 design notes.
"""

import logging
from datetime import datetime, timezone

from app.core.models import Concept
from app.storage import load_concept

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Goal Taxonomy v2 — structured scoring profiles
# ---------------------------------------------------------------------------
GOAL_TAXONOMY = {
    "ship_launch": {
        "description": "Preparing to ship publicly",
        "target_kas": ["product_strategy", "business_strategy", "operations"],
        "target_types": ["decision", "constraint", "observation"],
        "recency_weight": 0.7,
        "authority_weight": 0.6,
    },
    "debug_production": {
        "description": "Diagnosing a production issue",
        "target_kas": ["debugging", "architecture", "implementation", "performance"],
        "target_types": ["observation", "pattern", "method"],
        "recency_weight": 0.9,
        "authority_weight": 0.3,
    },
    "competitive_analysis": {
        "description": "Analyzing competitive landscape",
        "target_kas": ["competitive_analysis", "product_strategy", "business_strategy"],
        "target_types": ["observation", "decision", "pattern"],
        "recency_weight": 0.6,
        "authority_weight": 0.5,
    },
    "architecture_review": {
        "description": "Reviewing system architecture",
        "target_kas": ["architecture", "design_principles", "performance"],
        "target_types": ["decision", "method", "principle"],
        "recency_weight": 0.3,
        "authority_weight": 0.7,
    },
    "improve_process": {
        "description": "Improving workflows and processes",
        "target_kas": ["process", "operations"],
        "target_types": ["method", "heuristic", "pattern"],
        "recency_weight": 0.5,
        "authority_weight": 0.5,
    },
    "solve_problem": {
        "description": "Solving a specific problem",
        "target_kas": ["debugging", "implementation"],
        "target_types": ["observation", "method"],
        "recency_weight": 0.8,
        "authority_weight": 0.4,
    },
    "make_decision": {
        "description": "Making a strategic or technical decision",
        "target_kas": ["product_strategy", "architecture", "business_strategy"],
        "target_types": ["decision", "principle", "observation"],
        "recency_weight": 0.4,
        "authority_weight": 0.7,
    },
    "plan_project": {
        "description": "Planning a project or roadmap",
        "target_kas": ["process", "product_strategy", "operations"],
        "target_types": ["decision", "method", "constraint"],
        "recency_weight": 0.5,
        "authority_weight": 0.5,
    },
    # --- Legacy goal names mapped to new taxonomy ---
    "diagnose_issue": {
        "description": "Diagnosing a production issue (legacy alias)",
        "target_kas": ["debugging", "architecture", "implementation", "performance"],
        "target_types": ["observation", "pattern", "method"],
        "recency_weight": 0.9,
        "authority_weight": 0.3,
    },
    "optimize_performance": {
        "description": "Optimizing system performance (legacy alias)",
        "target_kas": ["performance", "architecture", "implementation"],
        "target_types": ["observation", "method", "pattern"],
        "recency_weight": 0.7,
        "authority_weight": 0.5,
    },
    "understand_system": {
        "description": "Understanding system architecture (legacy alias)",
        "target_kas": ["architecture", "design_principles", "implementation"],
        "target_types": ["principle", "method", "observation"],
        "recency_weight": 0.3,
        "authority_weight": 0.6,
    },
    "learn_topic": {
        "description": "Learning about a topic (legacy alias)",
        "target_kas": ["general", "architecture", "product_strategy"],
        "target_types": ["principle", "observation", "method"],
        "recency_weight": 0.3,
        "authority_weight": 0.6,
    },
}

# Goal inference patterns — map query keywords to goal names
_GOAL_PATTERNS = {
    "ship_launch": ["ship", "launch", "release", "deploy", "go live", "publish"],
    "debug_production": ["debug", "fix", "broken", "error", "crash", "failing", "bug"],
    "competitive_analysis": ["competitor", "competitive", "market", "landscape", "vs", "alternative"],
    "architecture_review": ["architecture", "design", "structure", "refactor", "pattern"],
    "improve_process": ["improve", "streamline", "workflow", "process", "efficiency"],
    "solve_problem": ["solve", "resolve", "issue", "problem", "root cause"],
    "make_decision": ["should i", "which", "decide", "choose", "compare", "tradeoff"],
    "plan_project": ["plan", "roadmap", "timeline", "schedule", "milestone", "sprint"],
    "diagnose_issue": ["diagnose", "why", "what's wrong", "investigate"],
    "optimize_performance": ["faster", "slower", "performance", "optimize", "latency"],
    "understand_system": ["explain", "how does", "understand", "what is"],
    "learn_topic": ["learn", "teach me", "tutorial", "guide"],
}


def _parse_dt(s: str | None) -> datetime | None:
    """Parse ISO datetime string, handling timezone variations."""
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        # Normalize to UTC naive for comparison
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except (ValueError, TypeError):
        return None


def _recency_score(updated_at: str | None) -> float:
    """Compute recency factor from updated_at timestamp.

    Returns 0.0-1.0 where 1.0 = updated today, decaying over time.
    """
    dt = _parse_dt(updated_at)
    if not dt:
        return 0.1  # Unknown age gets minimal score
    now = datetime.utcnow()
    days_old = (now - dt).total_seconds() / 86400
    if days_old < 1:
        return 1.0
    elif days_old < 7:
        return 0.7
    elif days_old < 30:
        return 0.3
    return 0.1


class GoalDirectedRetrieval:
    """Boost concept relevance based on current task/goal.

    v2: Uses structured scoring (knowledge_area, concept_type, recency,
    authority) instead of keyword matching against summaries/signals.
    """

    def __init__(self):
        self.active_goal: str | None = None
        self.goal_context: dict = {}

    def set_goal(self, goal: str, context: dict = None):
        """Set the current goal for retrieval boosting.

        Context can include 'source': 'explicit' | 'inferred' to control
        blend weight. Inferred goals use softer weighting (85/15) to avoid
        overriding embedding scores when goal inference is uncertain.
        """
        self.active_goal = goal
        self.goal_context = context or {}

    def clear_goal(self):
        """Clear the active goal."""
        self.active_goal = None
        self.goal_context = {}

    def infer_goal(self, query: str) -> str | None:
        """Infer goal from query text using keyword patterns."""
        query_lower = query.lower()
        for goal, keywords in _GOAL_PATTERNS.items():
            if any(kw in query_lower for kw in keywords):
                return goal
        return None

    def compute_goal_relevance(self, concept: Concept) -> float:
        """Compute how relevant a concept is to the active goal.

        v2 scoring components (all use 100%-populated fields):
          1. Knowledge area alignment  (0 – 0.40)
          2. Concept type alignment    (0 – 0.25)
          3. Recency signal            (0 – 0.20)
          4. Authority/confidence      (0 – 0.15)
        Max possible score: 1.0
        """
        if not self.active_goal or self.active_goal not in GOAL_TAXONOMY:
            return 0.0

        goal = GOAL_TAXONOMY[self.active_goal]
        score = 0.0

        # --- 1. Knowledge area alignment (0–0.4) ---
        ka = getattr(concept, "knowledge_area", None) or concept.metadata.get("knowledge_area", "")
        if ka in goal["target_kas"]:
            idx = goal["target_kas"].index(ka)
            # First match = 0.4, second = 0.34, third = 0.28, floor at 0.10
            ka_score = 0.4 * (1.0 - idx * 0.15)
            score += max(0.10, ka_score)

        # --- 2. Concept type alignment (0–0.25) ---
        ct = getattr(concept, "concept_type", "") or ""
        if ct in goal["target_types"]:
            idx = goal["target_types"].index(ct)
            # First match = 0.25, second = 0.2125, third = 0.175, floor at 0.05
            ct_score = 0.25 * (1.0 - idx * 0.15)
            score += max(0.05, ct_score)

        # --- 3. Recency signal (0–0.2) ---
        updated_at = getattr(concept, "updated_at", None)
        recency = _recency_score(updated_at)
        # Normalize by goal's recency_weight (0.5 is neutral)
        score += 0.2 * recency * (goal["recency_weight"] / 0.5)
        # Cap recency contribution at 0.2
        score = min(score, (score - 0.2 * recency * (goal["recency_weight"] / 0.5)) + 0.2)

        # --- 4. Authority/confidence signal (0–0.15) ---
        confidence = getattr(concept, "confidence", 0.5)
        score += 0.15 * confidence * (goal["authority_weight"] / 0.5)
        # Cap authority contribution at 0.15
        total_without_auth = score - 0.15 * confidence * (goal["authority_weight"] / 0.5)
        score = min(score, total_without_auth + 0.15)

        return min(1.0, score)

    def boost_scores_by_goal(
        self, concept_scores: list[tuple], concept_cache: dict | None = None
    ) -> list[tuple]:
        """Boost retrieval scores based on goal relevance.

        Args:
            concept_scores: List of (concept_id, score) tuples
            concept_cache: Optional {concept_id: concept} from Phase 1 (PERF-018).
                          When provided, avoids N DB reads per result.

        Returns:
            List of (concept_id, boosted_score) tuples, re-sorted by final score.
        """
        if not self.active_goal:
            return concept_scores

        # Explicit goals get full 60/40 blend; inferred goals get softer 85/15
        # to avoid overriding good embedding scores with uncertain goal inference.
        source = self.goal_context.get("source", "explicit")
        if source == "inferred":
            base_weight, goal_weight = 0.85, 0.15
        else:
            base_weight, goal_weight = 0.60, 0.40

        boosted = []
        boost_count = 0

        for concept_id, base_score in concept_scores:
            # PERF-018: cache-first lookup — avoids N DB reads when cache provided
            concept = concept_cache.get(concept_id) if concept_cache else None
            if concept is None:
                concept = load_concept(concept_id, track_access=False)
            if not concept:
                boosted.append((concept_id, base_score))
                continue

            goal_relevance = self.compute_goal_relevance(concept)

            # Combine base score with goal relevance
            final_score = (base_score * base_weight) + (goal_relevance * goal_weight)

            if goal_relevance > 0.1:
                boost_count += 1

            boosted.append((concept_id, final_score))

        # Re-sort by boosted score
        boosted.sort(key=lambda x: x[1], reverse=True)

        if boost_count > 0:
            logger.info(
                "goal_directed v2: goal=%s source=%s weights=%.0f/%.0f boosted=%d/%d concepts",
                self.active_goal,
                source,
                base_weight * 100,
                goal_weight * 100,
                boost_count,
                len(concept_scores),
            )

        return boosted


# Global instance
goal_directed = GoalDirectedRetrieval()
