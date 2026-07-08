"""Automatic lifecycle freshness and supersession trust for context turns.

SUPER-018 productizes existing concept lifecycle evidence at read time. It does
not verify external truth and does not mutate concept state.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

STALE_STATES = {"AGING", "REVIEW"}
SUPERSEDED_CURRENCIES = {"SUPERSEDED"}
CAUTION_CURRENCIES = {"STALE"}
DEGRADED_CURRENCIES = {"CONTESTED", "CONTRADICTED"}
SUPERSESSION_SENTINELS = {"", "__orphaned_supersession__"}
HISTORICAL_INTENT_TERMS = (
    "history",
    "historical",
    "previous",
    "prior",
    "formerly",
    "before",
    "why did this change",
    "what changed",
    "changed",
    "over time",
    "evolved",
    "audit",
    "provenance",
    "timeline",
    "evolution",
    "old version",
)
CURRENT_STATE_INTENT_TERMS = (
    "current",
    "current state",
    "latest",
    "now",
    "today",
)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _normalized(value: Any) -> str:
    return str(value or "").strip().upper()


def _record_metric(name: str, value: float, labels: dict[str, str] | None = None) -> None:
    try:
        from app.ops.metrics import metrics

        metrics.record(name, value, labels or {})
    except Exception:
        return


def _valid_current_head(concept: Any) -> bool:
    if concept is None:
        return False
    if str(_field(concept, "status", "active") or "").lower() in {
        "archived",
        "deleted",
        "superseded",
    }:
        return False
    if _field(concept, "is_current", 1) not in (1, True, None):
        return False
    return _normalized(_field(concept, "currency_status", "ACTIVE")) != "SUPERSEDED"


def _matches_intent_term(lowered: str, term: str) -> bool:
    if " " in term:
        return re.search(rf"\b{re.escape(term)}\b", lowered) is not None
    return re.search(rf"\b{re.escape(term)}\b", lowered) is not None


def _has_historical_intent(query_text: str | None) -> bool:
    lowered = (query_text or "").lower()
    return any(_matches_intent_term(lowered, term) for term in HISTORICAL_INTENT_TERMS)


def _has_current_state_intent(query_text: str | None) -> bool:
    lowered = (query_text or "").lower()
    return any(_matches_intent_term(lowered, term) for term in CURRENT_STATE_INTENT_TERMS)


def classify_freshness_intent(query_text: str | None) -> str:
    if _has_historical_intent(query_text):
        return "historical"
    if _has_current_state_intent(query_text):
        return "current_state"
    return "neutral"


def _resolve_replacement(
    concept: Any,
    activated_by_id: dict[str, Any],
    load_replacement_concept_fn: Callable[[str], Any] | None,
    *,
    admit_missing_replacement_heads: bool,
    replacement_load_budget_remaining: int,
) -> tuple[str | None, Any | None, str]:
    replacement_id = _field(concept, "superseded_by")
    if not replacement_id:
        return None, None, "missing_pointer"
    if replacement_id in SUPERSESSION_SENTINELS:
        return None, None, "missing_pointer"
    if replacement_id in activated_by_id:
        return replacement_id, activated_by_id[replacement_id], "already_activated"
    if not admit_missing_replacement_heads or load_replacement_concept_fn is None:
        return None, None, "not_admitted"
    if replacement_load_budget_remaining <= 0:
        return None, None, "budget_exhausted"
    replacement = load_replacement_concept_fn(str(replacement_id))
    if replacement is None:
        return None, None, "not_found"
    if not _valid_current_head(replacement):
        return None, None, "invalid_replacement"
    return str(replacement_id), replacement, "loaded"


def _concept_annotation(
    concept: Any,
    *,
    trust: str,
    reason: str,
    explanation: str,
    replacement_id: str | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": "context_freshness_conflict.v1",
        "concept_id": _field(concept, "concept_id") or _field(concept, "id"),
        "trust": trust,
        "reason": reason,
        "explanation": explanation,
        "currency_status": _field(concept, "currency_status"),
        "staleness_state": _field(concept, "staleness_state"),
        "freshness_label": _field(concept, "freshness_label"),
    }
    if replacement_id:
        payload["replacement_concept_id"] = replacement_id
    return {key: value for key, value in payload.items() if value is not None}


def apply_context_freshness_trust(
    activated_concepts: list[Any],
    *,
    load_replacement_concept_fn: Callable[..., Any] | None = None,
    load_concept_fn: Callable[..., Any] | None = None,
    include_deprecated: bool = False,
    suppress_confirmed_superseded: bool = True,
    query_text: str | None = None,
    admit_missing_replacement_heads: bool = True,
    max_replacement_loads: int = 5,
    required_context_ids: set[str] | list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Annotate lifecycle trust and suppress only confirmed superseded context.

    Returns a dict so the caller can atomically replace ``activated_concepts`` and
    surface bounded product-facing explanations in the response.
    """

    activated = list(activated_concepts or [])
    required_ids = {str(concept_id) for concept_id in (required_context_ids or []) if concept_id}
    activated_by_id = {
        str(_field(concept, "concept_id") or _field(concept, "id")): concept
        for concept in activated
        if _field(concept, "concept_id") or _field(concept, "id")
    }
    suppressed_ids: set[str] = set()
    conflicts: list[dict[str, Any]] = []
    admitted_replacements: list[Any] = []
    historical_intent = _has_historical_intent(query_text)
    replacement_loader = load_replacement_concept_fn or load_concept_fn
    replacement_loads_attempted = 0
    counts = {
        "suppressed": 0,
        "used_with_caution": 0,
        "degraded": 0,
        "superseded_not_suppressed": 0,
        "admitted_replacements": 0,
        "historical_included": 0,
        "replacement_load_rejected": 0,
        "replacement_load_budget_exhausted": 0,
    }

    for concept in activated:
        concept_id = str(_field(concept, "concept_id") or _field(concept, "id") or "")
        currency_status = _normalized(_field(concept, "currency_status", "ACTIVE"))
        staleness_state = _normalized(_field(concept, "staleness_state"))
        status = str(_field(concept, "status", "") or "").lower()
        superseded_by = _field(concept, "superseded_by")

        if currency_status in SUPERSEDED_CURRENCIES or status == "superseded" or superseded_by:
            remaining_budget = max(0, max_replacement_loads - replacement_loads_attempted)
            replacement_id, replacement, replacement_source = _resolve_replacement(
                concept,
                activated_by_id,
                replacement_loader,
                admit_missing_replacement_heads=admit_missing_replacement_heads,
                replacement_load_budget_remaining=remaining_budget,
            )
            if replacement_source in {"loaded", "invalid_replacement"}:
                replacement_loads_attempted += 1
            if replacement_source == "budget_exhausted":
                counts["replacement_load_budget_exhausted"] += 1
            elif replacement_source == "invalid_replacement":
                counts["replacement_load_rejected"] += 1
            if replacement_id and replacement_source == "loaded" and replacement is not None:
                activated_by_id[replacement_id] = replacement
                admitted_replacements.append(replacement)
                counts["admitted_replacements"] += 1

            if replacement_id and historical_intent and not include_deprecated:
                counts["historical_included"] += 1
                conflicts.append(
                    _concept_annotation(
                        concept,
                        trust="used_with_caution",
                        reason="superseded_memory_included_for_history",
                        explanation="This memory is superseded, but the query asks for historical context; Pith also included the current replacement.",
                        replacement_id=replacement_id,
                    )
                )
                continue

            if replacement_id and suppress_confirmed_superseded and not include_deprecated:
                suppressed_ids.add(concept_id)
                counts["suppressed"] += 1
                conflicts.append(
                    _concept_annotation(
                        concept,
                        trust="suppressed",
                        reason="superseded_by_newer_pith_evidence",
                        explanation="Newer Pith evidence superseded this memory; using the replacement instead.",
                        replacement_id=replacement_id,
                    )
                )
            else:
                counts["superseded_not_suppressed"] += 1
                if replacement_id and include_deprecated:
                    reason = "superseded_memory_included_by_request"
                    explanation = "This memory is superseded, but deprecated context was explicitly requested."
                elif replacement_id:
                    reason = "superseded_memory_shadow_only"
                    explanation = "This memory is superseded, but suppression is running in annotation-only mode."
                elif replacement_source == "budget_exhausted":
                    reason = "superseded_replacement_load_budget_exhausted"
                    explanation = "This memory is marked superseded, but Pith hit the replacement admission budget before confirming the current replacement."
                elif replacement_source == "invalid_replacement":
                    reason = "superseded_replacement_not_current"
                    explanation = "This memory is marked superseded, but the replacement was not valid current Pith evidence."
                else:
                    reason = "superseded_replacement_not_available"
                    explanation = "This memory is marked superseded, but Pith could not confirm a usable replacement in this turn."
                conflicts.append(
                    _concept_annotation(
                        concept,
                        trust="degraded",
                        reason=reason,
                        explanation=explanation,
                        replacement_id=replacement_id,
                    )
                )
            continue

        if currency_status in CAUTION_CURRENCIES or staleness_state in STALE_STATES:
            counts["used_with_caution"] += 1
            conflicts.append(
                _concept_annotation(
                    concept,
                    trust="used_with_caution",
                    reason="stale_or_aging_memory",
                    explanation="This memory may be stale, so Pith should treat it as context rather than current truth.",
                )
            )
            continue

        if currency_status == "CONTESTED" and concept_id in required_ids:
            counts["used_with_caution"] += 1
            conflicts.append(
                _concept_annotation(
                    concept,
                    trust="used_with_caution",
                    reason="contested_required_context",
                    explanation=(
                        "This required context has contested lifecycle evidence, so Pith "
                        "should use it with caution rather than treat it as authoritative."
                    ),
                )
            )
            continue

        if currency_status in DEGRADED_CURRENCIES:
            counts["degraded"] += 1
            conflicts.append(
                _concept_annotation(
                    concept,
                    trust="degraded",
                    reason="contested_or_contradicted_memory",
                    explanation="This memory has contested or contradicted lifecycle evidence and should not be treated as authoritative.",
                )
            )

    filtered = [
        concept
        for concept in activated
        if str(_field(concept, "concept_id") or _field(concept, "id") or "") not in suppressed_ids
    ]
    filtered.extend(admitted_replacements)
    if counts["suppressed"]:
        action = "suppressed_superseded"
        decision = "used_with_caution"
    elif conflicts:
        action = "annotated_lifecycle_risk"
        decision = "used_with_caution"
    else:
        action = "trusted"
        decision = "trusted"

    for key, value in counts.items():
        if value:
            _record_metric(
                f"context_freshness.{key}_count",
                float(value),
                {"decision": decision, "action": action},
            )

    return {
        "activated_concepts": filtered,
        "conflicts": conflicts or None,
        "decision": {
            "schema_version": "context_freshness_decision.v1",
            "decision": decision,
            "trust": "automatic_lifecycle_evidence",
            "action": action,
            "counts": counts,
            "suppressed_concept_ids": sorted(suppressed_ids),
            "explanation": (
                "Pith used lifecycle evidence to suppress superseded memory and mark stale or contested memory."
                if conflicts
                else "Pith found no stale or superseded lifecycle evidence in activated context."
            ),
        }
        if conflicts
        else None,
    }
