"""Context evaluation telemetry.

Telemetry intentionally captures no payload text, summaries, constraints, or
verbatim evidence. It only builds low-cardinality counters/metadata and defers
metric recording to the existing post-response task hook.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

CAPTURE_MODE_ENV = "PITH_CONTEXT_EVAL_CAPTURE_MODE"
METADATA_SAMPLE_RATE_ENV = "PITH_CONTEXT_EVAL_METADATA_SAMPLE_RATE"
METADATA_HIGH_LATENCY_MS_ENV = "PITH_CONTEXT_EVAL_METADATA_HIGH_LATENCY_MS"
METADATA_TOP_IDS_MAX_ENV = "PITH_CONTEXT_EVAL_METADATA_TOP_IDS_MAX"

MODE_OFF = "off"
MODE_COUNTERS_ONLY = "counters_only"
MODE_METADATA_SAMPLED = "metadata_sampled"
VALID_CAPTURE_MODES = {MODE_OFF, MODE_COUNTERS_ONLY, MODE_METADATA_SAMPLED}

METADATA_EVENT_VERSION = "context_eval_metadata.v1"
DEFAULT_METADATA_SAMPLE_RATE = 0.05
DEFAULT_METADATA_HIGH_LATENCY_MS = 3500.0
DEFAULT_METADATA_TOP_IDS_MAX = 8

COUNTER_METRICS = (
    "context_eval.turn_seen",
    "context_eval.capture_attempted",
    "context_eval.capture_succeeded",
    "context_eval.capture_dropped",
    "context_eval.capture_exception",
)


def context_eval_capture_mode() -> str:
    mode = os.environ.get(CAPTURE_MODE_ENV, MODE_OFF).strip().lower()
    return mode if mode in VALID_CAPTURE_MODES else MODE_OFF


def _float_env(name: str, default: float, min_value: float, max_value: float | None = None) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    if value < min_value:
        return min_value
    if max_value is not None and value > max_value:
        return max_value
    return value


def _int_env(name: str, default: int, min_value: int, max_value: int | None = None) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    if value < min_value:
        return min_value
    if max_value is not None and value > max_value:
        return max_value
    return value


def metadata_sample_rate() -> float:
    return _float_env(METADATA_SAMPLE_RATE_ENV, DEFAULT_METADATA_SAMPLE_RATE, 0.0, 1.0)


def metadata_high_latency_ms() -> float:
    return _float_env(METADATA_HIGH_LATENCY_MS_ENV, DEFAULT_METADATA_HIGH_LATENCY_MS, 0.0)


def metadata_top_ids_max() -> int:
    return _int_env(METADATA_TOP_IDS_MAX_ENV, DEFAULT_METADATA_TOP_IDS_MAX, 0, 25)


def _safe_label(value: Any, default: str = "unknown", max_len: int = 40) -> str:
    if value is None:
        return default
    return str(value)[:max_len]


def _hash_prefix(value: Any, length: int = 12) -> str | None:
    if value is None:
        return None
    text = str(value)
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _concept_id(concept: Any) -> str | None:
    if isinstance(concept, dict):
        value = concept.get("concept_id") or concept.get("id")
    else:
        value = getattr(concept, "concept_id", None) or getattr(concept, "id", None)
    if value is None:
        return None
    return str(value)[:80]


def _concept_score(concept: Any) -> float | None:
    if isinstance(concept, dict):
        value = concept.get("relevance_score", concept.get("score"))
    else:
        value = getattr(concept, "relevance_score", None)
        if value is None:
            value = getattr(concept, "score", None)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _score_bucket(value: float | None) -> str:
    if value is None:
        return "none"
    if value < 0.25:
        return "lt_0_25"
    if value < 0.5:
        return "0_25_0_5"
    if value < 0.75:
        return "0_5_0_75"
    return "gte_0_75"


def _top_concept_ids(response: Any) -> list[str]:
    cap = metadata_top_ids_max()
    if cap <= 0:
        return []
    ids: list[str] = []
    for concept in _as_list(getattr(response, "activated_concepts", None)):
        concept_id = _concept_id(concept)
        if concept_id:
            ids.append(concept_id)
        if len(ids) >= cap:
            break
    return ids


def _base_labels(request: Any, response: Any, mode: str) -> dict[str, str]:
    return {
        "mode": mode,
        "has_working_context": str(bool(getattr(response, "working_context", None))).lower(),
        "has_resume_context": str(bool(getattr(response, "resume_context", None))).lower(),
        "has_constraint_set": str(bool(getattr(response, "constraint_set", None))).lower(),
        "pressure_source": _safe_label(getattr(response, "pressure_source_used", None)),
        "platform": _safe_label(getattr(request, "platform_hint", None)),
    }


def build_context_eval_counter_event(request: Any, response: Any) -> dict[str, Any] | None:
    """Build a payload-free context-eval telemetry event for post-response recording."""
    mode = context_eval_capture_mode()
    if mode == MODE_OFF:
        return None

    start = time.perf_counter()
    if mode == MODE_METADATA_SAMPLED:
        return _build_metadata_sampled_event(request, response, start)

    labels = _base_labels(request, response, mode)
    capture_latency_ms = round((time.perf_counter() - start) * 1000.0, 4)
    return {
        "labels": labels,
        "metrics": {
            "context_eval.turn_seen": 1.0,
            "context_eval.capture_attempted": 1.0,
            "context_eval.capture_succeeded": 1.0,
            "context_eval.capture_dropped": 0.0,
            "context_eval.capture_exception": 0.0,
            "context_eval.queue_depth": 0.0,
            "context_eval.capture_latency_ms": capture_latency_ms,
            "context_eval.mode": 1.0,
        },
    }


def _metadata_sample_decision(request: Any, response: Any, top_ids: list[str]) -> tuple[bool, str, str]:
    governance_summary = _as_dict(getattr(response, "governance_summary", None))
    working_context = _as_dict(getattr(response, "working_context", None))
    constraint_set = _as_dict(getattr(response, "constraint_set", None))
    processing_time_ms = getattr(response, "processing_time_ms", None)

    basis_parts = [
        _hash_prefix(getattr(response, "resolved_session_id", None)) or "",
        _hash_prefix(getattr(request, "origin_id", None)) or "",
        _hash_prefix(getattr(request, "workstream_id", None)) or "",
        str(getattr(response, "activation_count", 0) or 0),
        ",".join(top_ids),
    ]
    basis_hash = hashlib.sha256("|".join(basis_parts).encode("utf-8")).hexdigest()

    if getattr(response, "resume_context", None):
        return True, "forced_resume_context", basis_hash[:12]
    if working_context.get("checkpoint"):
        return True, "forced_checkpoint_context", basis_hash[:12]
    if _as_list(constraint_set.get("constraints")):
        return True, "forced_constraints", basis_hash[:12]
    if _as_list(governance_summary.get("phases_skipped")):
        return True, "forced_governance_skipped", basis_hash[:12]
    try:
        if float(processing_time_ms) >= metadata_high_latency_ms():
            return True, "forced_high_latency", basis_hash[:12]
    except (TypeError, ValueError):
        pass

    threshold = int(metadata_sample_rate() * 10_000)
    if int(basis_hash[:8], 16) % 10_000 < threshold:
        return True, "random_sample", basis_hash[:12]
    return False, "not_selected", basis_hash[:12]


def _build_metadata_sampled_event(request: Any, response: Any, start: float) -> dict[str, Any]:
    top_ids = _top_concept_ids(response)
    selected, sample_reason, basis_prefix = _metadata_sample_decision(request, response, top_ids)
    labels = _base_labels(request, response, MODE_METADATA_SAMPLED)
    labels.update(
        {
            "event_version": METADATA_EVENT_VERSION,
            "sample_reason": sample_reason,
            "sample_basis_hash_prefix": basis_prefix,
        }
    )

    if selected:
        governance_summary = _as_dict(getattr(response, "governance_summary", None))
        coverage_confidence = _as_dict(getattr(response, "coverage_confidence", None))
        abstention_signal = _as_dict(getattr(response, "abstention_signal", None))
        for key, value in (
            ("session_hash", _hash_prefix(getattr(response, "resolved_session_id", None))),
            ("origin_hash", _hash_prefix(getattr(request, "origin_id", None))),
            ("workstream_hash", _hash_prefix(getattr(request, "workstream_id", None))),
        ):
            if value:
                labels[key] = value
        labels.update(
            {
                "has_coverage_confidence": str(bool(coverage_confidence)).lower(),
                "has_abstention_signal": str(bool(abstention_signal)).lower(),
                "has_active_workstream": str(bool(getattr(response, "active_workstream", None))).lower(),
                "coverage_level": _safe_label(coverage_confidence.get("level")),
                "abstention_level": _safe_label(abstention_signal.get("level")),
                "top_concept_ids": ",".join(top_ids),
                "top_score_bucket": _score_bucket(
                    _concept_score(_as_list(getattr(response, "activated_concepts", None))[0])
                    if _as_list(getattr(response, "activated_concepts", None))
                    else None
                ),
            }
        )
    metadata_payload_bytes = len(json.dumps(labels, sort_keys=True))
    capture_latency_ms = round((time.perf_counter() - start) * 1000.0, 4)

    activated_concepts = _as_list(getattr(response, "activated_concepts", None))
    constraint_set = _as_dict(getattr(response, "constraint_set", None))
    working_context = _as_dict(getattr(response, "working_context", None))
    governance_summary = _as_dict(getattr(response, "governance_summary", None))
    metrics = {
        "context_eval.turn_seen": 1.0,
        "context_eval.capture_attempted": 1.0,
        "context_eval.capture_succeeded": 1.0 if selected else 0.0,
        "context_eval.capture_dropped": 0.0 if selected else 1.0,
        "context_eval.capture_exception": 0.0,
        "context_eval.queue_depth": 0.0,
        "context_eval.capture_latency_ms": capture_latency_ms,
        "context_eval.mode": 1.0,
        "context_eval.sample_eligible": 1.0,
        "context_eval.sample_selected": 1.0 if selected else 0.0,
        "context_eval.metadata_payload_bytes": float(metadata_payload_bytes if selected else 0),
        "context_eval.activated_concepts_count": float(len(activated_concepts) if selected else 0),
        "context_eval.constraint_count": float(len(_as_list(constraint_set.get("constraints"))) if selected else 0),
        "context_eval.working_context_pinned_count": float(len(_as_list(working_context.get("pinned_concepts"))) if selected else 0),
        "context_eval.analogy_count": float(len(_as_list(getattr(response, "analogy_suggestions", None))) if selected else 0),
        "context_eval.freshness_warning_count": float(len(_as_list(getattr(response, "freshness_warnings", None))) if selected else 0),
        "context_eval.governance_phases_executed_count": float(
            len(_as_list(governance_summary.get("phases_executed"))) if selected else 0
        ),
        "context_eval.governance_phases_skipped_count": float(
            len(_as_list(governance_summary.get("phases_skipped"))) if selected else 0
        ),
        "context_eval.response_processing_time_ms": float(getattr(response, "processing_time_ms", 0.0) or 0.0)
        if selected
        else 0.0,
    }
    return {"labels": labels, "metrics": metrics}


def record_context_eval_counter_event(event: dict[str, Any] | None) -> None:
    """Record a context-eval telemetry event; failures never escape post-response work."""
    if not event:
        return
    try:
        from app.ops.metrics import metrics

        labels = dict(event.get("labels") or {})
        for metric_name, value in (event.get("metrics") or {}).items():
            metric_labels = labels if metric_name != "context_eval.mode" else {"mode": labels.get("mode", MODE_OFF)}
            metrics.record(metric_name, float(value), metric_labels)
    except Exception:
        return
