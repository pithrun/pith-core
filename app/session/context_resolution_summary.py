"""Product-facing context resolution summary for conversation turns.

SUPER-020 aggregates existing trust, freshness, abstention, and coverage signals.
It does not retrieve, mutate, arbitrate, or verify external truth.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

MAX_CONTEXT_IDS = 8
SCHEMA_VERSION = "context_resolution_summary.v1"
RUNTIME_SUPPORT_SCOPE_AUTHORITIES = frozenset(
    {
        "retrieval_137_lexical_admission",
        "retrieval_141_existing_support",
        "retrieval_153_semantic_recovery",
    }
)
SEMANTIC_RECOVERY_SUPPORT_AUTHORITY = "retrieval_153_semantic_recovery"


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else []


def _concept_id(concept: Any) -> str | None:
    concept_id = _field(concept, "concept_id") or _field(concept, "id")
    return str(concept_id) if concept_id else None


def _bounded_unique(values: Sequence[Any] | None, *, limit: int = MAX_CONTEXT_IDS) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        if value is None:
            continue
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _coverage_status(
    coverage_score: float | None,
    coverage_confidence: Mapping[str, Any] | None,
) -> str:
    confidence = _as_dict(coverage_confidence)
    level = str(confidence.get("level") or "").lower()
    if level == "absent_knowledge" or coverage_score == 0:
        return "absent"
    if level == "sparse_coverage":
        return "sparse"
    if coverage_score is None:
        return "unknown"
    if coverage_score < 0.15:
        return "absent"
    if coverage_score < 0.40:
        return "sparse"
    return "sufficient"


def _abstention_payload(abstention_signal: Mapping[str, Any] | None) -> dict[str, Any] | None:
    signal = _as_dict(abstention_signal)
    if not signal.get("should_abstain"):
        return None
    return {
        key: signal.get(key)
        for key in ("level", "reason", "confidence")
        if signal.get(key) is not None
    }


def _lexical_support_payload(lexical_support: Mapping[str, Any] | None) -> dict[str, Any] | None:
    support = _as_dict(lexical_support)
    if not support.get("applied"):
        return None
    return {
        "applied": True,
        "support_level": support.get("support_level"),
        "trust_modifier": support.get("trust_modifier"),
        "support_ids": _bounded_unique(_as_list(support.get("support_ids"))),
        "admitted_ids": _bounded_unique(_as_list(support.get("admitted_ids"))),
        "contested_ids": _bounded_unique(_as_list(support.get("contested_ids"))),
        "reason": support.get("reason"),
        "path": support.get("path"),
        "trace_authority": support.get("trace_authority"),
        "runtime_eligible": support.get("runtime_eligible") is True,
    }


def _replacement_pairs(conflicts: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for conflict in conflicts:
        superseded_id = conflict.get("concept_id")
        current_id = conflict.get("replacement_concept_id")
        if not superseded_id or not current_id:
            continue
        key = (str(superseded_id), str(current_id))
        if key in seen:
            continue
        seen.add(key)
        pairs.append(
            {
                "superseded_id": key[0],
                "current_id": key[1],
                "reason": str(conflict.get("reason") or "superseded_by_newer_pith_evidence"),
            }
        )
        if len(pairs) >= MAX_CONTEXT_IDS:
            break
    return pairs


def _has_unresolved_supersession(conflicts: Sequence[Mapping[str, Any]]) -> bool:
    unresolved_reasons = {
        "superseded_replacement_load_budget_exhausted",
        "superseded_replacement_not_current",
        "superseded_replacement_not_available",
    }
    return any(str(conflict.get("reason") or "") in unresolved_reasons for conflict in conflicts)


def _degraded_conflicts_exactly_supported_contested(
    *,
    degraded_ids: Sequence[str],
    conflicts: Sequence[Mapping[str, Any]],
    lexical_support: Mapping[str, Any] | None,
    unresolved_supersession: bool,
) -> bool:
    """Return true only for exact lexical support of contested degraded IDs."""

    if unresolved_supersession or not degraded_ids or not lexical_support:
        return False

    support_ids = set(_bounded_unique(_as_list(lexical_support.get("support_ids"))))
    contested_ids = set(_bounded_unique(_as_list(lexical_support.get("contested_ids"))))
    degraded_set = set(degraded_ids)
    if not support_ids or not degraded_set.issubset(support_ids):
        return False

    conflicts_by_id: dict[str, list[Mapping[str, Any]]] = {}
    for conflict in conflicts:
        conflict_id = conflict.get("concept_id") or conflict.get("subject")
        if conflict_id:
            conflicts_by_id.setdefault(str(conflict_id), []).append(conflict)

    for concept_id in degraded_set:
        matching_conflicts = conflicts_by_id.get(concept_id)
        if not matching_conflicts:
            return False
        for conflict in matching_conflicts:
            status = str(conflict.get("currency_status") or "").upper()
            reason = str(conflict.get("reason") or "")
            if status == "CONTRADICTED":
                return False
            if status == "CONTESTED":
                continue
            if (
                not status
                and concept_id in contested_ids
                and reason == "contested_or_contradicted_memory"
            ):
                continue
            return False

    return True


def _semantic_support_scope_applies(lexical_support: Mapping[str, Any] | None) -> bool:
    support = _as_dict(lexical_support)
    return (
        support.get("applied") is True
        and support.get("runtime_eligible") is True
        and support.get("trace_authority") == SEMANTIC_RECOVERY_SUPPORT_AUTHORITY
        and bool(_as_list(support.get("support_ids")))
    )


def _runtime_support_scope_applies(lexical_support: Mapping[str, Any] | None) -> bool:
    support = _as_dict(lexical_support)
    return (
        support.get("applied") is True
        and support.get("runtime_eligible") is True
        and support.get("trace_authority") in RUNTIME_SUPPORT_SCOPE_AUTHORITIES
        and bool(_as_list(support.get("support_ids")))
    )


def _is_semantic_recovery_support(lexical_support: Mapping[str, Any] | None) -> bool:
    support = _as_dict(lexical_support)
    return support.get("trace_authority") == SEMANTIC_RECOVERY_SUPPORT_AUTHORITY


def _explanation(
    *,
    decision: str,
    resolution_action: str,
    coverage_status: str,
    abstention: dict[str, Any] | None,
    lexical_support: dict[str, Any] | None = None,
) -> str:
    if decision == "abstained":
        return "Pith did not find enough reliable context to treat the activated memory as current."
    if resolution_action == "used_exact_lexical_support":
        if lexical_support and lexical_support.get("trust_modifier") == "caution":
            return "Pith found exact lexical support, but lifecycle evidence requires caution."
        return "Pith found exact lexical support in the activated context."
    if resolution_action == "used_runtime_support_scope":
        return "Pith found runtime support for the relevant context and isolated unrelated degraded context."
    if resolution_action == "used_current_replacement":
        return "Pith used newer evidence for the current answer and kept superseded memory out of primary context."
    if resolution_action == "included_history":
        return "Pith included older superseded memory because the request asked for history or provenance."
    if resolution_action == "marked_contested":
        return "Pith found contested or contradicted context and marked the turn as degraded."
    if resolution_action == "marked_stale":
        return "Pith found stale or aging context and marked it for caution."
    if abstention or coverage_status in {"sparse", "absent"}:
        return "Pith found limited context, so the answer should be treated with caution."
    return "Pith found no stale, superseded, or contested lifecycle evidence in the activated context."


def build_context_resolution_summary(
    *,
    activated_concepts: Sequence[Any],
    context_freshness_decision: Mapping[str, Any] | None,
    context_freshness_conflicts: Sequence[Mapping[str, Any]] | None,
    context_trust_decision: Mapping[str, Any] | None,
    context_trust_conflicts: Sequence[Mapping[str, Any]] | None,
    abstention_signal: Mapping[str, Any] | None,
    coverage_score: float | None,
    coverage_confidence: Mapping[str, Any] | None,
    lexical_support: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a bounded product summary from existing response diagnostics."""

    freshness_decision = _as_dict(context_freshness_decision)
    trust_decision = _as_dict(context_trust_decision)
    freshness_conflicts = [
        _as_dict(conflict) for conflict in _as_list(context_freshness_conflicts)
        if isinstance(conflict, Mapping)
    ]
    trust_conflicts = [
        _as_dict(conflict) for conflict in _as_list(context_trust_conflicts)
        if isinstance(conflict, Mapping)
    ]
    counts = _as_dict(freshness_decision.get("counts"))

    primary_ids = _bounded_unique(_concept_id(concept) for concept in activated_concepts)
    suppressed_ids = _bounded_unique(
        list(_as_list(freshness_decision.get("suppressed_concept_ids")))
        + [
            conflict.get("concept_id")
            for conflict in freshness_conflicts
            if conflict.get("trust") == "suppressed"
        ]
    )
    historical_ids = _bounded_unique(
        conflict.get("concept_id")
        for conflict in freshness_conflicts
        if conflict.get("reason") == "superseded_memory_included_for_history"
    )
    caution_ids = _bounded_unique(
        conflict.get("concept_id")
        for conflict in freshness_conflicts
        if conflict.get("trust") == "used_with_caution"
    )
    degraded_ids = _bounded_unique(
        [
            conflict.get("concept_id") or conflict.get("subject")
            for conflict in freshness_conflicts + trust_conflicts
            if conflict.get("trust") == "degraded"
        ]
    )
    pairs = _replacement_pairs(freshness_conflicts)
    coverage_status = _coverage_status(coverage_score, coverage_confidence)
    abstention = _abstention_payload(abstention_signal)
    lexical_support_summary = _lexical_support_payload(lexical_support)
    hard_abstention = bool(abstention and abstention.get("level") == "hard")
    unresolved_supersession = _has_unresolved_supersession(freshness_conflicts)
    semantic_support_scope = _semantic_support_scope_applies(lexical_support_summary)
    runtime_support_scope = _runtime_support_scope_applies(lexical_support_summary)
    support_ids = set(_bounded_unique(_as_list((lexical_support_summary or {}).get("support_ids"))))
    support_scoped_degraded_ids = (
        _bounded_unique([concept_id for concept_id in degraded_ids if concept_id in support_ids])
        if runtime_support_scope
        else []
    )
    unrelated_degraded_ids = (
        _bounded_unique([concept_id for concept_id in degraded_ids if concept_id not in support_ids])
        if runtime_support_scope
        else []
    )
    blocking_degraded_ids = support_scoped_degraded_ids if runtime_support_scope else degraded_ids
    support_scope_applied = bool(
        runtime_support_scope
        and degraded_ids
        and not blocking_degraded_ids
        and not unresolved_supersession
    )
    supported_contested_degraded = _degraded_conflicts_exactly_supported_contested(
        degraded_ids=blocking_degraded_ids,
        conflicts=freshness_conflicts + trust_conflicts,
        lexical_support=lexical_support_summary,
        unresolved_supersession=unresolved_supersession,
    )

    if counts.get("suppressed") or pairs and suppressed_ids:
        resolution_action = "used_current_replacement"
    elif counts.get("historical_included") or historical_ids:
        resolution_action = "included_history"
    elif support_scope_applied:
        resolution_action = "used_runtime_support_scope"
    elif blocking_degraded_ids or unresolved_supersession or str(trust_decision.get("decision") or "") == "degraded":
        resolution_action = "marked_contested"
    elif caution_ids:
        resolution_action = "marked_stale"
    elif lexical_support_summary:
        resolution_action = "used_exact_lexical_support"
    else:
        resolution_action = "none"

    if hard_abstention:
        decision = "abstained"
    elif (
        (blocking_degraded_ids and not supported_contested_degraded)
        or unresolved_supersession
        or str(trust_decision.get("decision") or "") == "degraded"
    ):
        decision = "degraded"
    elif (
        suppressed_ids
        or historical_ids
        or caution_ids
        or abstention
            or (coverage_status in {"sparse", "absent"} and not lexical_support_summary)
            or (
                support_scope_applied
                and not _is_semantic_recovery_support(lexical_support_summary)
            )
            or (
                lexical_support_summary is not None
                and lexical_support_summary.get("trust_modifier") == "caution"
        )
    ):
        decision = "used_with_caution"
    else:
        decision = "trusted"

    if hard_abstention or coverage_status == "absent":
        currentness = "unknown"
    elif historical_ids and primary_ids:
        currentness = "mixed"
    elif historical_ids:
        currentness = "historical"
    elif caution_ids or unresolved_supersession or (
        coverage_status == "sparse" and not lexical_support_summary
    ):
        currentness = "stale_or_uncertain"
    elif primary_ids:
        currentness = "current"
    else:
        currentness = "unknown"

    return {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "resolution_action": resolution_action,
        "currentness": currentness,
        "primary_context_ids": primary_ids,
        "suppressed_context_ids": suppressed_ids,
        "historical_context_ids": historical_ids,
        "caution_context_ids": caution_ids,
        "degraded_context_ids": degraded_ids,
        "support_scoped_degraded_context_ids": support_scoped_degraded_ids,
        "unrelated_degraded_context_ids": unrelated_degraded_ids,
        "support_scope_applied": support_scope_applied,
        "replacement_pairs": pairs,
        "abstention": abstention,
        "lexical_evidence_support": lexical_support_summary,
        "coverage": {
            "score": coverage_score,
            "status": coverage_status,
        },
        "explanation": _explanation(
            decision=decision,
            resolution_action=resolution_action,
            coverage_status=coverage_status,
            abstention=abstention,
            lexical_support=lexical_support_summary,
        ),
    }
