"""Cold-path retrieval policy shadow recommendation generation.

This module does not change live retrieval behavior. It turns curated policy
gold rows into evaluator-compatible candidate recommendations.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

SCHEMA_VERSION = "retrieval_policy_candidate_recommendations.v1"
DEFAULT_POLICY_VERSION = "retrieval_policy_shadow_recommendation.v1"
MODE_SHADOW_ONLY = "shadow_only"

KEEP_CURRENT = "keep_current"
RECOMMENDATION_ONLY = "recommendation_only"
EXISTING_MODE_COMPARISON = "existing_mode_comparison"

DEFAULT_CLASS_ACTIONS: dict[str, str] = {
    "aggregate_source_set": "force_source_set",
    "multihop_relation": "force_source_set",
    "sparse_no_coverage": "repair_sparse",
    "contradiction_supersession_sensitive": "force_abstention",
}


class RetrievalPolicyRecommendationError(ValueError):
    """Raised when shadow recommendation input is invalid."""


@dataclass(frozen=True)
class PolicyGoldRow:
    pair_id: str
    query_class: str


def load_policy_gold_rows(gold_payload: dict[str, Any]) -> list[PolicyGoldRow]:
    """Load minimal, content-free policy-gold row data from a JSON payload."""

    raw_rows = gold_payload.get("pairs")
    if not isinstance(raw_rows, list):
        raise RetrievalPolicyRecommendationError("gold payload must contain a pairs list")

    rows: list[PolicyGoldRow] = []
    for idx, raw in enumerate(raw_rows, start=1):
        if not isinstance(raw, dict):
            raise RetrievalPolicyRecommendationError(f"row {idx}: must be an object")
        pair_id = str(raw.get("id") or "").strip()
        query_class = str(raw.get("class") or "").strip()
        if not pair_id:
            raise RetrievalPolicyRecommendationError(f"row {idx}: id is required")
        if not query_class:
            raise RetrievalPolicyRecommendationError(f"{pair_id}: class is required")
        rows.append(PolicyGoldRow(pair_id=pair_id, query_class=query_class))
    return rows


def validate_curated_gold_payload(gold_payload: dict[str, Any]) -> None:
    """Require curated live gold by default for shadow recommendation evidence."""

    metadata = gold_payload.get("metadata")
    if not isinstance(metadata, dict):
        raise RetrievalPolicyRecommendationError("metadata is required")
    if metadata.get("dataset_type") != "curated_live_gold":
        raise RetrievalPolicyRecommendationError(
            "metadata.dataset_type must be curated_live_gold"
        )
    if metadata.get("review_status") not in {"operator_curated", "curated"}:
        raise RetrievalPolicyRecommendationError(
            "metadata.review_status must be operator_curated or curated"
        )
    if metadata.get("generated_candidate_allowed") is not False:
        raise RetrievalPolicyRecommendationError(
            "metadata.generated_candidate_allowed must be false"
        )


def build_candidate_recommendations(
    rows: list[PolicyGoldRow],
    *,
    enabled_classes: set[str] | None = None,
    max_affected_per_class: int = 1,
    policy_version: str = DEFAULT_POLICY_VERSION,
    class_actions: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build evaluator-compatible shadow recommendations for policy-gold rows."""

    actions = class_actions or DEFAULT_CLASS_ACTIONS
    enabled = set(actions) if enabled_classes is None else set(enabled_classes)
    unknown = sorted(enabled.difference(actions))
    if unknown:
        raise RetrievalPolicyRecommendationError(f"unknown enabled classes: {unknown}")
    if max_affected_per_class < 0:
        raise RetrievalPolicyRecommendationError("max_affected_per_class must be >= 0")

    affected_counts: Counter[str] = Counter()
    recommendations: list[dict[str, str]] = []
    for row in rows:
        action = KEEP_CURRENT
        comparison_kind = RECOMMENDATION_ONLY
        behavioral_change = "none"
        reason = "shadow generator control row"
        if (
            row.query_class in enabled
            and affected_counts[row.query_class] < max_affected_per_class
        ):
            action = actions[row.query_class]
            affected_counts[row.query_class] += 1
            comparison_kind = EXISTING_MODE_COMPARISON
            behavioral_change = f"shadow candidate for {row.query_class}"
            reason = f"shadow generator recommends {action} for {row.query_class}"

        recommendations.append(
            {
                "pair_id": row.pair_id,
                "recommended_action": action,
                "reason": reason,
                "comparison_kind": comparison_kind,
                "behavioral_change": behavioral_change,
            }
        )

    affected_pair_ids = [
        row["pair_id"]
        for row in recommendations
        if row["recommended_action"] != KEEP_CURRENT
    ]
    return {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "policy_version": policy_version,
            "mode": MODE_SHADOW_ONLY,
            "generator": "app.retrieval.policy_recommendations",
            "full_replay_required": True,
            "promotion_note": (
                "Recommendation output is not promotion evidence without full "
                "curated-gold production-turn replay."
            ),
            "enabled_classes": sorted(enabled),
            "max_affected_per_class": max_affected_per_class,
            "affected_pair_count": len(affected_pair_ids),
            "affected_classes": sorted(affected_counts),
        },
        "recommendations": recommendations,
    }
