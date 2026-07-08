"""Trace-only retrieval policy observability for conversation turns."""

from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "retrieval_policy_trace.v1"
MODE_OBSERVE_ONLY = "observe_only"
EXPOSURE_DEBUG_RESPONSE_FIELD = "debug_response_field"

MAX_ROUTER_SIGNALS = 12
MAX_ACTIVATED_IDS = 20
MAX_CANDIDATE_STAGE_IDS = 30
MAX_DEADLINE_SKIPS = 12

RETRIEVAL_SCALED_PHASES = (
    "ct_phase_retrieval_ms",
    "ct_phase_graph_ms",
    "ct_phase_injection_ms",
    "ct_phase_contradiction_ms",
)


def _safe_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any, *, digits: int = 4) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _safe_ms(mapping: Any, key: str) -> float:
    if not isinstance(mapping, dict):
        return 0.0
    value = _safe_float(mapping.get(key), digits=4)
    if value is None or value < 0:
        return 0.0
    return value


def _safe_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text[:120]


def _safe_labels(values: Any, *, limit: int) -> list[str]:
    if not isinstance(values, (list, tuple, set)):
        return []
    labels: list[str] = []
    for value in values:
        label = _safe_label(value)
        if label is not None:
            labels.append(label)
        if len(labels) >= limit:
            break
    return labels


def _concept_id(value: Any) -> str | None:
    cid = getattr(value, "concept_id", None)
    if cid is None and isinstance(value, dict):
        cid = value.get("concept_id")
    label = _safe_label(cid)
    return label


def build_latency_components_ms(
    *,
    processing_time_ms: Any,
    phase_ms: dict[str, Any] | None = None,
    stage3_metric_ms: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Build replay-safe latency components from measured turn timers.

    `stage3_metric_ms` is accepted for call-site symmetry and future extensions;
    this helper intentionally uses parent phase timers to avoid double-counting
    subphase attribution already contained in those parent spans.
    """
    total = _safe_float(processing_time_ms, digits=2)
    if total is None or total < 0:
        total = 0.0
    retrieval_scaled_ms = sum(_safe_ms(phase_ms, key) for key in RETRIEVAL_SCALED_PHASES)
    retrieval_scaled_ms = min(round(retrieval_scaled_ms, 2), total)
    candidate_extra_ms = 0.0
    fixed_ms = max(0.0, round(total - retrieval_scaled_ms - candidate_extra_ms, 2))
    return {
        "fixed_ms": fixed_ms,
        "retrieval_scaled_ms": retrieval_scaled_ms,
        "candidate_extra_ms": candidate_extra_ms,
    }


def _activated_ids(activated_concepts: Any) -> tuple[list[str], bool, int]:
    if not isinstance(activated_concepts, list):
        return [], False, 0
    ids: list[str] = []
    for concept in activated_concepts:
        cid = _concept_id(concept)
        if cid is not None:
            ids.append(cid)
        if len(ids) >= MAX_ACTIVATED_IDS:
            break
    return ids, len(activated_concepts) > MAX_ACTIVATED_IDS, len(activated_concepts)


def _all_activated_ids(activated_concepts: Any) -> list[str]:
    if not isinstance(activated_concepts, list):
        return []
    ids: list[str] = []
    for concept in activated_concepts:
        cid = _concept_id(concept)
        if cid is not None:
            ids.append(cid)
    return ids


def _expected_id_diagnostics(
    expected_concept_ids: Any,
    activated_concepts: Any,
) -> dict[str, Any] | None:
    if not isinstance(expected_concept_ids, (list, tuple, set)):
        return None
    expected_ids = tuple(
        dict.fromkeys(
            label
            for value in expected_concept_ids
            if (label := _safe_label(value)) is not None
        )
    )
    if not expected_ids:
        return None
    all_activated_ids = _all_activated_ids(activated_concepts)
    activated_rank = {
        concept_id: rank
        for rank, concept_id in enumerate(all_activated_ids, start=1)
        if concept_id in expected_ids
    }
    return {
        "expected_ids_checked": len(expected_ids),
        "expected_ids_activated_count": len(activated_rank),
        "expected_ids_missing_count": len(expected_ids) - len(activated_rank),
        "expected_rank_min": min(activated_rank.values()) if activated_rank else None,
        "trace_authority": "diagnostic_expected_ids_only",
        "runtime_eligible": False,
    }


def _candidate_stage_ids(value: Any) -> tuple[list[str], bool]:
    if not isinstance(value, (list, tuple)):
        return [], False
    ids = _safe_labels(value, limit=MAX_CANDIDATE_STAGE_IDS)
    return ids, len(value) > len(ids)


def _candidate_stage_trace(candidate_stages: Any) -> dict[str, Any] | None:
    if not isinstance(candidate_stages, dict):
        return None
    sanitized: dict[str, Any] = {}
    for stage_name, stage_value in candidate_stages.items():
        safe_stage_name = _safe_label(stage_name)
        if safe_stage_name is None or not isinstance(stage_value, dict):
            continue
        stage_payload: dict[str, Any] = {}
        for prefix in ("before", "after"):
            count_key = f"{prefix}_count"
            ids_key = f"{prefix}_ids"
            truncated_key = f"{prefix}_ids_truncated"
            count = _safe_int(stage_value.get(count_key))
            if count is not None:
                stage_payload[count_key] = count
            ids, ids_truncated = _candidate_stage_ids(stage_value.get(ids_key))
            if ids:
                stage_payload[ids_key] = ids
            explicit_truncated = _safe_bool(stage_value.get(truncated_key))
            count_truncated = count is not None and bool(ids) and count > len(ids)
            if explicit_truncated or ids_truncated or count_truncated:
                stage_payload[truncated_key] = True
            elif ids:
                stage_payload[truncated_key] = False
        if stage_payload:
            sanitized[safe_stage_name] = stage_payload
    return sanitized or None


def _deadline_skips(deadline: Any) -> tuple[list[dict[str, Any]], int]:
    skips = getattr(deadline, "skips", None)
    if not isinstance(skips, list):
        return [], 0
    bounded: list[dict[str, Any]] = []
    for skip in skips[:MAX_DEADLINE_SKIPS]:
        if not isinstance(skip, dict):
            continue
        bounded.append(
            {
                "phase": _safe_label(skip.get("phase")),
                "reason": _safe_label(skip.get("reason")),
                "priority": _safe_label(skip.get("priority")),
            }
        )
    return bounded, len(skips)


def _deadline_remaining_ms(deadline: Any) -> float | None:
    remaining = getattr(deadline, "remaining_ms", None)
    if not callable(remaining):
        return None
    return _safe_float(remaining(), digits=2)


def build_retrieval_policy_trace(
    *,
    adaptive_config: Any = None,
    answer_path_admission: Any = None,
    question_classification: dict | None = None,
    turn_deadline: Any = None,
    source_set_trace: dict | None = None,
    coverage_confidence: dict | None = None,
    coverage_score: float | None = None,
    governance_summary: dict | None = None,
    requested_max_concepts: int | None = None,
    effective_max_concepts: int | None = None,
    activated_concepts: list[Any] | None = None,
    expected_concept_ids: list[str] | tuple[str, ...] | set[str] | None = None,
    candidate_stages: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a bounded, content-free retrieval-policy trace.

    The trace is diagnostic data only. It must not read storage, call retrieval,
    inspect raw user text, or mutate any input.
    """
    activated_ids, activated_ids_truncated, activated_count = _activated_ids(
        activated_concepts or []
    )
    expected_diagnostics = _expected_id_diagnostics(
        expected_concept_ids,
        activated_concepts or [],
    )
    deadline_skips, deadline_skip_count = _deadline_skips(turn_deadline)
    deadline_overruns = getattr(turn_deadline, "overruns", None)
    if not isinstance(deadline_overruns, list):
        deadline_overrun_count = 0
    else:
        deadline_overrun_count = len(deadline_overruns)

    source_debts = source_set_trace.get("debts") if isinstance(source_set_trace, dict) else None
    if not isinstance(source_debts, list):
        source_debt_count = None
    else:
        source_debt_count = len(source_debts)

    candidate_flow = {
        "requested_max_concepts": _safe_int(requested_max_concepts),
        "effective_max_concepts": _safe_int(effective_max_concepts),
        "activated_count": activated_count,
        "activated_ids": activated_ids,
        "activated_ids_truncated": activated_ids_truncated,
        "source_set_required": (
            _safe_bool(source_set_trace.get("source_set_required"))
            if isinstance(source_set_trace, dict)
            else None
        ),
        "source_set_debt_count": source_debt_count,
    }
    if expected_diagnostics is not None:
        candidate_flow["expected_id_diagnostics"] = expected_diagnostics
    safe_candidate_stages = _candidate_stage_trace(candidate_stages)
    if safe_candidate_stages is not None:
        candidate_flow["candidate_stages"] = safe_candidate_stages

    return {
        "schema_version": SCHEMA_VERSION,
        "mode": MODE_OBSERVE_ONLY,
        "exposure": EXPOSURE_DEBUG_RESPONSE_FIELD,
        "query_class": {
            "router_signals": _safe_labels(
                getattr(adaptive_config, "signals", None),
                limit=MAX_ROUTER_SIGNALS,
            ),
            "router_is_adaptive": _safe_bool(
                getattr(adaptive_config, "is_adaptive", None)
            ),
            "router_use_multihop": _safe_bool(
                getattr(adaptive_config, "use_multihop", None)
            ),
            "router_use_entity_chain": _safe_bool(
                getattr(adaptive_config, "use_entity_chain", None)
            ),
            "answer_path_mode": _safe_label(
                getattr(answer_path_admission, "mode", None)
            ),
            "answer_path_reason": _safe_label(
                getattr(answer_path_admission, "reason", None)
            ),
            "question_classification": _safe_label(
                question_classification.get("classification")
                if isinstance(question_classification, dict)
                else None
            ),
            "question_confidence": _safe_float(
                question_classification.get("confidence")
                if isinstance(question_classification, dict)
                else None
            ),
        },
        "admission": {
            "observe_only": True,
            "allow_multihop": _safe_bool(
                getattr(answer_path_admission, "allow_multihop", None)
            ),
            "allow_entity_chain": _safe_bool(
                getattr(answer_path_admission, "allow_entity_chain", None)
            ),
            "allow_graph": _safe_bool(
                getattr(answer_path_admission, "allow_graph", None)
            ),
            "allow_optional_injection": _safe_bool(
                getattr(answer_path_admission, "allow_optional_injection", None)
            ),
            "max_concepts_cap": _safe_int(
                getattr(answer_path_admission, "max_concepts_cap", None)
            ),
            "deadline_enabled": bool(getattr(turn_deadline, "enabled", False)),
            "deadline_skip_count": deadline_skip_count,
            "deadline_skips": deadline_skips,
            "deadline_overrun_count": deadline_overrun_count,
        },
        "candidate_flow": candidate_flow,
        "health": {
            "coverage_level": (
                _safe_label(coverage_confidence.get("level"))
                if isinstance(coverage_confidence, dict)
                else None
            ),
            "coverage_score": _safe_float(coverage_score),
            "governance_circuit_breaker_tripped": (
                _safe_bool(governance_summary.get("circuit_breaker_tripped"))
                if isinstance(governance_summary, dict)
                else None
            ),
            "latency_remaining_ms": (
                _safe_float(governance_summary.get("latency_remaining_ms"), digits=2)
                if isinstance(governance_summary, dict)
                else None
            ),
            "turn_deadline_remaining_ms": _deadline_remaining_ms(turn_deadline),
        },
        "policy_recommendation": {
            "action": MODE_OBSERVE_ONLY,
            "reason": "slice0_no_behavior_change",
        },
    }
