"""Observe-only context-trust outcome classification for conversation turns."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

SCHEMA_VERSION = "context_trust_outcome.v1"
MAX_REASONS = 8
MAX_REASON_LEN = 120
RUNTIME_SUPPORT_SCOPE_AUTHORITIES = frozenset(
    {
        "retrieval_137_lexical_admission",
        "retrieval_141_existing_support",
        "retrieval_153_semantic_recovery",
    }
)

PRIMARY_TRUSTED_CONTEXT = "trusted_context"
PRIMARY_APPROPRIATE_ABSTENTION = "appropriate_abstention"
PRIMARY_SPARSE_ACCEPTABLE = "sparse_acceptable"
PRIMARY_RETRIEVAL_MISS_FAILURE = "retrieval_miss_failure"
PRIMARY_STALE_CONTEXT_FAILURE = "stale_context_failure"
PRIMARY_TRACE_MISSING_FAILURE = "trace_missing_failure"
PRIMARY_QUERY_PROVENANCE_FAILURE = "query_provenance_failure"
PRIMARY_UNSUPPORTED_UNKNOWN = "unsupported_unknown"

_ACTION_BY_PRIMARY = {
    PRIMARY_TRUSTED_CONTEXT: "proceed",
    PRIMARY_APPROPRIATE_ABSTENTION: "accept_or_monitor_abstention",
    PRIMARY_SPARSE_ACCEPTABLE: "monitor_sparse_coverage",
    PRIMARY_RETRIEVAL_MISS_FAILURE: "inspect_retrieval_candidate_flow",
    PRIMARY_STALE_CONTEXT_FAILURE: "inspect_context_freshness",
    PRIMARY_TRACE_MISSING_FAILURE: "inspect_trace_coverage",
    PRIMARY_QUERY_PROVENANCE_FAILURE: "inspect_query_provenance",
    PRIMARY_UNSUPPORTED_UNKNOWN: "inspect_raw_diagnostics",
}


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else []


def _bounded_labels(values: Sequence[Any] | None, *, limit: int = MAX_REASONS) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = str(value or "").strip()
        if not text:
            continue
        text = text[:MAX_REASON_LEN]
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _payload(primary: str, reasons: Sequence[Any], limitations: Sequence[Any] | None = None) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "primary": primary,
        "reasons": _bounded_labels(reasons) or ["classified_from_existing_diagnostics"],
        "operator_action": _ACTION_BY_PRIMARY.get(primary, "inspect_raw_diagnostics"),
        "observe_only": True,
        "runtime_behavior_changed": False,
        "evidence_limitations": _bounded_labels(limitations),
    }


def _coverage_status(summary: Mapping[str, Any], trace: Mapping[str, Any]) -> tuple[str | None, str | None]:
    coverage = _as_dict(summary.get("coverage"))
    health = _as_dict(trace.get("health"))
    return (
        str(coverage.get("status") or "").lower() or None,
        str(health.get("coverage_level") or "").lower() or None,
    )


def _is_sparse_or_absent(summary: Mapping[str, Any], trace: Mapping[str, Any]) -> bool:
    coverage_status, coverage_level = _coverage_status(summary, trace)
    return coverage_status in {"sparse", "absent"} or coverage_level in {
        "sparse_coverage",
        "no_results",
        "no_strong_match",
        "absent_knowledge",
    }


def _abstention_level(summary: Mapping[str, Any]) -> str | None:
    abstention = _as_dict(summary.get("abstention"))
    level = str(abstention.get("level") or "").lower()
    return level or None


def _expected_missing_count(trace: Mapping[str, Any]) -> int:
    candidate_flow = _as_dict(trace.get("candidate_flow"))
    diagnostics = _as_dict(candidate_flow.get("expected_id_diagnostics"))
    try:
        return int(diagnostics.get("expected_ids_missing_count") or 0)
    except (TypeError, ValueError):
        return 0


def _expected_checked(trace: Mapping[str, Any]) -> bool:
    candidate_flow = _as_dict(trace.get("candidate_flow"))
    diagnostics = _as_dict(candidate_flow.get("expected_id_diagnostics"))
    try:
        return int(diagnostics.get("expected_ids_checked") or 0) > 0
    except (TypeError, ValueError):
        return False


def _query_provenance_failed(query_intent_trace: Mapping[str, Any]) -> bool:
    if not query_intent_trace:
        return False
    raw_hash = query_intent_trace.get("raw_query_hash")
    assembled_hash = query_intent_trace.get("assembled_query_hash")
    hashes_differ = bool(raw_hash and assembled_hash and raw_hash != assembled_hash)
    effective_source = str(query_intent_trace.get("effective_query_source") or "")
    if query_intent_trace.get("assembled_context_used") is True:
        return True
    if hashes_differ and query_intent_trace.get("contamination_guard_blocked") is False:
        return True
    return hashes_differ and effective_source in {"assembled_query", "query_text"}


def _has_exact_support(summary: Mapping[str, Any], trace: Mapping[str, Any]) -> bool:
    support = _as_dict(summary.get("lexical_evidence_support")) or _as_dict(
        trace.get("lexical_evidence_support")
    )
    return support.get("applied") is True and bool(_as_list(support.get("support_ids")))


def _has_current_or_historical_resolution(summary: Mapping[str, Any]) -> bool:
    return str(summary.get("resolution_action") or "") in {
        "used_current_replacement",
        "included_history",
        "used_exact_lexical_support",
        "used_runtime_support_scope",
    }


def _has_runtime_support_scope(summary: Mapping[str, Any]) -> bool:
    support = _as_dict(summary.get("lexical_evidence_support"))
    return (
        summary.get("support_scope_applied") is True
        and support.get("applied") is True
        and support.get("runtime_eligible") is True
        and support.get("trace_authority") in RUNTIME_SUPPORT_SCOPE_AUTHORITIES
        and bool(_as_list(support.get("support_ids")))
        and bool(_as_list(summary.get("unrelated_degraded_context_ids")))
        and not bool(_as_list(summary.get("support_scoped_degraded_context_ids")))
    )


def classify_context_trust_outcome(
    *,
    context_resolution_summary: Mapping[str, Any] | None,
    retrieval_policy_trace: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Classify sparse/abstention outcomes from existing bounded diagnostics only."""

    summary = _as_dict(context_resolution_summary)
    trace = _as_dict(retrieval_policy_trace)
    if not summary and not trace:
        return _payload(PRIMARY_UNSUPPORTED_UNKNOWN, ["missing_summary_and_policy_trace"])

    query_intent_trace = _as_dict(trace.get("query_intent_trace"))
    fallback_trace = _as_dict(trace.get("fallback_before_abstention"))
    abstention_level = _abstention_level(summary)
    hard_abstention = abstention_level == "hard"
    any_abstention = abstention_level in {"hard", "soft"}
    sparse_or_absent = _is_sparse_or_absent(summary, trace)

    if _query_provenance_failed(query_intent_trace):
        return _payload(
            PRIMARY_QUERY_PROVENANCE_FAILURE,
            ["assembled_query_used_as_effective_query"],
        )

    if (hard_abstention or sparse_or_absent) and not query_intent_trace:
        reasons = ["hard_or_sparse_without_query_intent_trace"]
        if hard_abstention and not fallback_trace:
            reasons.append("hard_abstention_without_fallback_trace")
        return _payload(PRIMARY_TRACE_MISSING_FAILURE, reasons)

    missing_count = _expected_missing_count(trace)
    if missing_count > 0:
        return _payload(
            PRIMARY_RETRIEVAL_MISS_FAILURE,
            [f"expected_ids_missing_count={missing_count}"],
        )

    degraded_ids = _as_list(summary.get("degraded_context_ids"))
    stale_action = str(summary.get("resolution_action") or "") in {
        "marked_contested",
        "marked_stale",
    }
    if (
        (degraded_ids or stale_action)
        and not _has_runtime_support_scope(summary)
        and not _has_exact_support(summary, trace)
        and not _has_current_or_historical_resolution(summary)
    ):
        return _payload(PRIMARY_STALE_CONTEXT_FAILURE, ["degraded_or_stale_context_without_resolution"])

    limitations: list[str] = []
    if not _expected_checked(trace):
        limitations.append("expected_ids_not_supplied")
    if hard_abstention and not fallback_trace:
        limitations.append("fallback_trace_not_supplied")

    if any_abstention:
        return _payload(
            PRIMARY_APPROPRIATE_ABSTENTION,
            [f"{abstention_level}_abstention_with_required_trace"],
            limitations,
        )

    if sparse_or_absent:
        return _payload(PRIMARY_SPARSE_ACCEPTABLE, ["sparse_coverage_with_required_trace"], limitations)

    if _has_current_or_historical_resolution(summary):
        return _payload(PRIMARY_TRUSTED_CONTEXT, ["context_state_resolved"])

    if str(summary.get("decision") or "") == "trusted":
        return _payload(PRIMARY_TRUSTED_CONTEXT, ["summary_decision_trusted"])

    return _payload(PRIMARY_UNSUPPORTED_UNKNOWN, ["no_matching_context_trust_outcome"], limitations)
