"""Default-off context preference for role-bearing metadata packets.

This module intentionally knows nothing about benchmark gold labels, source
identity, or source order. It only inspects product-shaped metadata already
present on activated concepts.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

ALLOWED_EVIDENCE_ROLES = frozenset(
    {
        "instruction_obligation",
        "summary_milestone",
        "exact_detail",
        "correction_update",
        "contradiction_side",
    }
)

FORBIDDEN_EVIDENCE_ROLES = frozenset({"structured_count"})

FORBIDDEN_PRIVATE_KEYS = frozenset(
    {
        "beam_expected_answer",
        "expected_answer",
        "expected_answers",
        "expected_source",
        "expected_sources",
        "gold",
        "gold_answer",
        "gold_source",
        "gold_sources",
        "q_id",
        "row_id",
        "target_answer",
        "target_source",
    }
)

_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "did",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "or",
        "should",
        "the",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "you",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_INSTRUCTION_TERMS = frozenset(
    {
        "constraint",
        "default",
        "format",
        "future",
        "guideline",
        "instruction",
        "must",
        "preference",
        "prefer",
        "recommend",
        "requirement",
        "rule",
        "style",
        "should",
    }
)
_SUMMARY_TERMS = frozenset({"history", "milestone", "progress", "recap", "status", "summary", "summarize"})
_EXACT_DETAIL_TERMS = frozenset(
    {
        "amount",
        "date",
        "detail",
        "exact",
        "format",
        "item",
        "name",
        "place",
        "price",
        "specific",
    }
)
_CORRECTION_TERMS = frozenset({"correct", "corrected", "current", "latest", "now", "updated"})


@dataclass(frozen=True)
class MetadataPacketPreferenceResult:
    activated_concepts: tuple[object, ...]
    trace: dict[str, Any]


def apply_metadata_packet_context_preference(
    question: str,
    activated_concepts: Sequence[object],
    *,
    enabled: bool,
    max_promotions: int = 3,
) -> MetadataPacketPreferenceResult:
    """Promote compatible role-bearing packets without changing context length."""

    concepts = tuple(activated_concepts)
    trace = _base_trace(enabled=enabled, candidate_count=len(concepts))
    if not enabled:
        trace["stop_reason"] = "disabled"
        return MetadataPacketPreferenceResult(activated_concepts=concepts, trace=trace)

    if max_promotions <= 0:
        trace["stop_reason"] = "max_promotions_not_positive"
        return MetadataPacketPreferenceResult(activated_concepts=concepts, trace=trace)

    question_tokens = _tokens(question)
    eligible: list[tuple[float, int, object, str]] = []
    rejected: dict[str, str] = {}
    structured_count_excluded = False
    forbidden_material_detected = False

    for index, concept in enumerate(concepts):
        concept_id = _concept_id(concept, index)
        rejected_reason = _rejection_reason(concept, question_tokens)
        if rejected_reason:
            rejected[concept_id] = rejected_reason
            if rejected_reason == "structured_count_excluded":
                structured_count_excluded = True
            if rejected_reason == "forbidden_material_detected":
                forbidden_material_detected = True
            continue

        role = str(_metadata_value(concept, "evidence_role"))
        priority = float(_metadata_value(concept, "grounding_priority"))
        slot_overlap = len(question_tokens & _slot_tokens(concept))
        relevance = _numeric_value(_read_attr(concept, "relevance_score"), default=0.0)
        eligible.append((priority + (slot_overlap * 0.05) + (relevance * 0.01), index, concept, role))

    trace["eligible_role_packet_count"] = len(eligible)
    trace["structured_count_excluded"] = structured_count_excluded
    trace["forbidden_material_detected"] = forbidden_material_detected
    trace["rejected_ids"] = list(rejected)
    trace["rejection_reasons"] = rejected

    if not eligible:
        trace["stop_reason"] = "no_eligible_role_packets"
        return MetadataPacketPreferenceResult(activated_concepts=concepts, trace=trace)

    ranked = sorted(eligible, key=lambda item: (item[0], -item[1]), reverse=True)
    promoted = ranked[:max_promotions]
    promoted_indices = {item[1] for item in promoted}
    promoted_ids = [_concept_id(item[2], item[1]) for item in promoted]

    reordered = tuple(item[2] for item in promoted) + tuple(
        concept for index, concept in enumerate(concepts) if index not in promoted_indices
    )

    trace["candidate_count_after"] = len(reordered)
    trace["promoted_ids"] = promoted_ids
    trace["displaced_ids"] = [
        _concept_id(concept, index)
        for index, concept in enumerate(concepts[: len(promoted)])
        if index not in promoted_indices
    ]
    trace["selected_packet_count_delta"] = len(reordered) - len(concepts)
    trace["estimated_token_count_delta"] = _window_token_count(reordered, len(promoted)) - _window_token_count(
        concepts,
        len(promoted),
    )
    trace["role_bearing_share_delta"] = round(
        _role_bearing_share(reordered, len(promoted)) - _role_bearing_share(concepts, len(promoted)),
        4,
    )
    trace["stop_reason"] = "promoted"
    return MetadataPacketPreferenceResult(activated_concepts=reordered, trace=trace)


def _base_trace(*, enabled: bool, candidate_count: int) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "candidate_count_before": candidate_count,
        "candidate_count_after": candidate_count,
        "eligible_role_packet_count": 0,
        "promoted_ids": [],
        "displaced_ids": [],
        "rejected_ids": [],
        "rejection_reasons": {},
        "selected_packet_count_delta": 0,
        "estimated_token_count_delta": 0,
        "role_bearing_share_delta": 0.0,
        "structured_count_excluded": False,
        "forbidden_material_detected": False,
        "extra_llm_calls": 0,
        "stop_reason": "not_started",
    }


def _rejection_reason(concept: object, question_tokens: set[str]) -> str | None:
    if _contains_forbidden_material(concept):
        return "forbidden_material_detected"

    role = _metadata_value(concept, "evidence_role")
    if not isinstance(role, str) or role not in ALLOWED_EVIDENCE_ROLES:
        if role in FORBIDDEN_EVIDENCE_ROLES:
            return "structured_count_excluded"
        return "evidence_role_not_allowed"

    if role == "contradiction_side" and not _has_authoritative_branch_evidence(concept):
        return "contradiction_side_missing_authority"

    for field_name in ("slot_subject", "slot_attribute", "slot_group_id"):
        value = _metadata_value(concept, field_name)
        if not isinstance(value, str) or not value.strip():
            return f"{field_name}_missing"

    priority = _metadata_value(concept, "grounding_priority")
    if isinstance(priority, bool) or not isinstance(priority, Real):
        return "grounding_priority_invalid"
    if priority < 0.0 or priority > 1.0:
        return "grounding_priority_out_of_range"

    if not _is_query_compatible(role, question_tokens, concept):
        return "query_not_compatible"
    return None


def _is_query_compatible(role: str, question_tokens: set[str], concept: object) -> bool:
    slot_overlap = bool(question_tokens & _slot_tokens(concept))
    if role == "instruction_obligation":
        return slot_overlap or bool(question_tokens & _INSTRUCTION_TERMS)
    if role == "summary_milestone":
        return slot_overlap or bool(question_tokens & _SUMMARY_TERMS)
    if role == "exact_detail":
        return slot_overlap or bool(question_tokens & _EXACT_DETAIL_TERMS)
    if role == "correction_update":
        return slot_overlap or bool(question_tokens & _CORRECTION_TERMS)
    if role == "contradiction_side":
        return slot_overlap
    return False


def _has_authoritative_branch_evidence(concept: object) -> bool:
    branch = _metadata_value(concept, "branch_provenance")
    if not isinstance(branch, Mapping):
        return False
    return bool(
        branch.get("validated_authoritative_branch")
        or branch.get("authoritative_current_state")
        or branch.get("current_state_authority"),
    )


def _slot_tokens(concept: object) -> set[str]:
    values = (
        _metadata_value(concept, "slot_subject"),
        _metadata_value(concept, "slot_attribute"),
        _metadata_value(concept, "slot_group_id"),
    )
    tokens: set[str] = set()
    for value in values:
        if isinstance(value, str):
            tokens.update(_tokens(value))
    return tokens


def _tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    return {token for token in _TOKEN_RE.findall(text.lower()) if len(token) > 2 and token not in _STOPWORDS}


def _concept_id(concept: object, index: int) -> str:
    value = _read_attr(concept, "concept_id")
    if value is None:
        value = _read_attr(concept, "id")
    return str(value) if value is not None else f"index:{index}"


def _metadata_value(concept: object, name: str) -> Any:
    value = _read_attr(concept, name)
    if value is not None:
        return value
    metadata = _read_attr(concept, "metadata")
    if isinstance(metadata, Mapping):
        return metadata.get(name)
    return None


def _read_attr(value: object, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _numeric_value(value: Any, *, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        return default
    return float(value)


def _contains_forbidden_material(value: object) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str) and key.lower() in FORBIDDEN_PRIVATE_KEYS:
                return True
            if _contains_forbidden_material(child):
                return True
        return False
    metadata = getattr(value, "metadata", None)
    if isinstance(metadata, Mapping) and _contains_forbidden_material(metadata):
        return True
    if isinstance(value, (str, bytes, int, float, bool, type(None))):
        return False
    if isinstance(value, Sequence):
        return any(_contains_forbidden_material(child) for child in value)
    return False


def _role_bearing_share(concepts: Sequence[object], window_size: int) -> float:
    if window_size <= 0:
        return 0.0
    window = tuple(concepts[:window_size])
    if not window:
        return 0.0
    role_bearing = sum(1 for concept in window if _metadata_value(concept, "evidence_role") in ALLOWED_EVIDENCE_ROLES)
    return role_bearing / len(window)


def _window_token_count(concepts: Sequence[object], window_size: int) -> int:
    return sum(_estimated_token_count(concept) for concept in concepts[: max(0, window_size)])


def _estimated_token_count(concept: object) -> int:
    summary = _read_attr(concept, "summary")
    if not isinstance(summary, str):
        return 0
    return max(1, len(summary.split()))
