"""Bounded per-turn latency trace payloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

TRACE_SCHEMA_VERSION = "turn_latency_trace.v1"
MAX_PHASE_KEYS = 16
MAX_SUBPHASE_KEYS = 24
MAX_COUNT_KEYS = 24
MAX_DEADLINE_RECORDS = 20
MAX_STRING_CHARS = 96
PHASE_PRIORITY_KEYS = (
    "ct_phase_autolearn_ms",
    "ct_phase_health_ms",
    "ct_phase_correction_ms",
    "ct_phase_search_lightweight_ms",
    "ct_phase_retrieval_ms",
    "ct_phase_retrieval_post_search_ms",
    "ct_phase_graph_ms",
    "ct_phase_graph_index_load_ms",
    "ct_phase_graph_expand_ms",
    "ct_phase_injection_ms",
    "ct_phase_evolution_ms",
    "ct_phase_contradiction_ms",
    "ct_phase_coactivation_ms",
    "ct_phase_contradiction_detect_ms",
    "ct_phase_constraint_assembly_ms",
    "ct_phase_assembly_ms",
)
FORBIDDEN_DETAIL_KEYS = {
    "activated_concept_ids",
    "activated_summaries",
    "concept_ids",
    "concept_summary",
    "embedding",
    "embeddings",
    "message",
    "prompt",
    "prompts",
    "response",
    "response_text",
    "summary",
    "user_message",
}


def _round_ms(value: Any) -> float | None:
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return None


def _clean_string(value: Any) -> str:
    text = str(value)
    if len(text) <= MAX_STRING_CHARS:
        return text
    return text[: MAX_STRING_CHARS - 3] + "..."


def _bounded_numeric_map(
    values: Mapping[str, Any],
    *,
    limit: int,
    priority_keys: Sequence[str] = (),
) -> dict[str, float]:
    bounded: dict[str, float] = {}
    ordered_keys = [key for key in priority_keys if key in values]
    priority_seen = set(ordered_keys)
    ordered_keys.extend(key for key in sorted(values) if key not in priority_seen)
    for key in ordered_keys[:limit]:
        rounded = _round_ms(values.get(key))
        if rounded is not None:
            bounded[_clean_string(key)] = rounded
    return bounded


def _bounded_deadline_records(records: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    if not records:
        return []

    bounded: list[dict[str, Any]] = []
    for record in records[:MAX_DEADLINE_RECORDS]:
        clean: dict[str, Any] = {}
        for key, value in record.items():
            if str(key) in FORBIDDEN_DETAIL_KEYS:
                continue
            clean_key = _clean_string(key)
            if isinstance(value, (int, float)):
                clean[clean_key] = _round_ms(value)
            elif value is None:
                clean[clean_key] = None
            else:
                clean[clean_key] = _clean_string(value)
        bounded.append(clean)
    return bounded


def build_turn_latency_trace(
    *,
    request_id: str | None,
    elapsed_ms: float,
    deadline: Any | None,
    phases_ms: Mapping[str, Any],
    subphase_ms: Mapping[str, Any],
    counts: Mapping[str, Any],
    answer_path_labels: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a low-cardinality per-turn trace payload for governance events."""

    budget_ms = None
    remaining_ms = None
    deadline_skips: Sequence[Mapping[str, Any]] | None = None
    deadline_overruns: Sequence[Mapping[str, Any]] | None = None
    deadline_phase_modes: Sequence[Mapping[str, Any]] | None = None
    if deadline is not None:
        budget_fn = getattr(deadline, "budget_ms", None)
        remaining_fn = getattr(deadline, "remaining_ms", None)
        if callable(budget_fn):
            budget_ms = _round_ms(budget_fn())
        if callable(remaining_fn):
            remaining_ms = _round_ms(remaining_fn())
        deadline_skips = getattr(deadline, "skips", None)
        deadline_overruns = getattr(deadline, "overruns", None)
        deadline_phase_modes = getattr(deadline, "phase_modes", None)

    clean_answer_path = {
        _clean_string(key): _clean_string(value)
        for key, value in (answer_path_labels or {}).items()
    }
    if "request_id" in clean_answer_path:
        clean_answer_path.pop("request_id", None)

    rounded_elapsed = _round_ms(elapsed_ms) or 0.0
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "request_id": _clean_string(request_id) if request_id else None,
        "elapsed_ms": rounded_elapsed,
        "budget_ms": budget_ms,
        "remaining_ms": remaining_ms,
        "over_budget": bool(budget_ms is not None and rounded_elapsed > budget_ms),
        "answer_path": clean_answer_path,
        "phase_ms": _bounded_numeric_map(
            phases_ms,
            limit=MAX_PHASE_KEYS,
            priority_keys=PHASE_PRIORITY_KEYS,
        ),
        "subphase_ms": _bounded_numeric_map(subphase_ms, limit=MAX_SUBPHASE_KEYS),
        "counts": _bounded_numeric_map(counts, limit=MAX_COUNT_KEYS),
        "deadline_skips": _bounded_deadline_records(deadline_skips),
        "deadline_overruns": _bounded_deadline_records(deadline_overruns),
        "deadline_phase_modes": _bounded_deadline_records(deadline_phase_modes),
    }
