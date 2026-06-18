"""Safe replay preparation for retrieval-policy production-turn evidence.

This module prepares cold-path replay artifacts. It does not change retrieval
behavior and it does not call model providers. Promotion authority remains with
``policy_replay_capture`` and ``retrieval_policy_shadow_eval``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

KEEP_CURRENT = "keep_current"
SOURCE_DIAGNOSTIC_REQUIRED_CLASSES = {
    "aggregate_source_set",
    "multihop_relation",
    "temporal_current_state",
    "contradiction_supersession_sensitive",
}
REQUIRED_LATENCY_COMPONENTS = ("fixed_ms", "retrieval_scaled_ms", "candidate_extra_ms")
HIDDEN_GOLD_KEYS = {
    "expected_concept_ids",
    "expected_answer_fragments",
    "forbidden_answer_fragments",
    "promotion_verdict",
    "candidate_recommendation",
    "recommended_action",
}


class SafeReplayCaptureError(ValueError):
    """Raised when a safe replay artifact would violate evidence hygiene."""


class SourceFinalContextUnavailable(SafeReplayCaptureError):
    """Raised when source-sensitive trace evidence cannot be derived."""


@dataclass(frozen=True)
class SafeReplayGoldRow:
    pair_id: str
    query_class: str
    query: str
    expected_concept_ids: tuple[str, ...]
    expected_answer_fragments: tuple[str, ...]
    forbidden_answer_fragments: tuple[str, ...]


@dataclass(frozen=True)
class SafeReplayRecommendation:
    pair_id: str
    recommended_action: str


def load_json_object(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise SafeReplayCaptureError(f"{path}: expected JSON object")
    return payload


def load_gold_rows(gold_payload: dict[str, Any]) -> list[SafeReplayGoldRow]:
    raw_pairs = gold_payload.get("pairs")
    if not isinstance(raw_pairs, list):
        raise SafeReplayCaptureError("gold payload must contain pairs list")
    rows: list[SafeReplayGoldRow] = []
    for idx, raw in enumerate(raw_pairs, start=1):
        if not isinstance(raw, dict):
            raise SafeReplayCaptureError(f"gold row {idx}: must be an object")
        pair_id = str(raw.get("id") or "").strip()
        query = str(raw.get("query") or "").strip()
        if not pair_id:
            raise SafeReplayCaptureError(f"gold row {idx}: id is required")
        if not query:
            raise SafeReplayCaptureError(f"{pair_id}: query is required")
        rows.append(
            SafeReplayGoldRow(
                pair_id=pair_id,
                query_class=str(raw.get("class") or "").strip(),
                query=query,
                expected_concept_ids=_string_tuple(raw.get("expected_concept_ids")),
                expected_answer_fragments=_string_tuple(raw.get("expected_answer_fragments")),
                forbidden_answer_fragments=_string_tuple(raw.get("forbidden_answer_fragments")),
            )
        )
    return rows


def load_recommendations(
    recommendations_payload: dict[str, Any] | None,
    gold_rows: list[SafeReplayGoldRow],
) -> dict[str, SafeReplayRecommendation]:
    recommendations = {
        row.pair_id: SafeReplayRecommendation(row.pair_id, KEEP_CURRENT)
        for row in gold_rows
    }
    if recommendations_payload is None:
        return recommendations
    raw_recommendations = recommendations_payload.get("recommendations")
    if not isinstance(raw_recommendations, list):
        raise SafeReplayCaptureError("recommendations payload must contain recommendations list")
    pair_ids = set(recommendations)
    for idx, raw in enumerate(raw_recommendations, start=1):
        if not isinstance(raw, dict):
            raise SafeReplayCaptureError(f"recommendation {idx}: must be an object")
        pair_id = str(raw.get("pair_id") or raw.get("id") or "").strip()
        if not pair_id:
            raise SafeReplayCaptureError(f"recommendation {idx}: pair_id is required")
        if pair_id not in pair_ids:
            raise SafeReplayCaptureError(f"{pair_id}: recommendation does not match gold")
        action = str(raw.get("recommended_action") or "").strip()
        if not action:
            raise SafeReplayCaptureError(f"{pair_id}: recommended_action is required")
        recommendations[pair_id] = SafeReplayRecommendation(pair_id, action)
    return recommendations


def build_replay_manifest(
    *,
    gold_payload: dict[str, Any],
    recommendations_payload: dict[str, Any] | None = None,
    source_db_path: Path | None = None,
    artifact_root: Path | None = None,
    mode: str = "simulate",
) -> dict[str, Any]:
    """Build a replay manifest without exposing hidden gold fields to answer packets."""

    rows = load_gold_rows(gold_payload)
    recommendations = load_recommendations(recommendations_payload, rows)
    manifest_rows = []
    for row in rows:
        recommendation = recommendations[row.pair_id]
        affected = recommendation.recommended_action != KEEP_CURRENT
        query_overlap = _query_overlap(row)
        manifest_rows.append(
            {
                "pair_id": row.pair_id,
                "class": row.query_class,
                "query": row.query,
                "affected": affected,
                "source_sensitive": row.query_class in SOURCE_DIAGNOSTIC_REQUIRED_CLASSES,
                "answer_evidence_risk": (
                    "query_contains_expected_fragment" if query_overlap else "none"
                ),
                "query_overlap_fragments": query_overlap,
            }
        )
    return {
        "metadata": {
            "schema_version": "retrieval_policy_safe_replay_manifest.v1",
            "mode": mode,
            "source_db_path": str(source_db_path) if source_db_path else None,
            "artifact_root": str(artifact_root) if artifact_root else None,
            "pair_count": len(rows),
            "affected_pair_ids": [
                item["pair_id"] for item in manifest_rows if item["affected"]
            ],
            "promotion_authority": "none_manifest_only",
        },
        "rows": manifest_rows,
    }


def build_answer_packets(
    *,
    gold_payload: dict[str, Any],
    manifest_payload: dict[str, Any],
    context_by_pair: dict[str, Any] | None = None,
    fail_on_leak: bool = True,
) -> dict[str, Any]:
    """Build answer packets that exclude hidden evaluator fields."""

    rows_by_id = {row.pair_id: row for row in load_gold_rows(gold_payload)}
    context_by_pair = context_by_pair or {}
    packets = []
    leak_reports = []
    for manifest_row in _manifest_rows(manifest_payload):
        pair_id = str(manifest_row.get("pair_id") or "").strip()
        if pair_id not in rows_by_id:
            raise SafeReplayCaptureError(f"{pair_id}: manifest row not present in gold")
        packet = {
            "pair_id": pair_id,
            "query": str(manifest_row.get("query") or ""),
            "allowed_context": context_by_pair.get(pair_id, {}),
            "instructions": "Answer the query using only the allowed context when it is relevant.",
        }
        leak_report = validate_answer_packet_no_hidden_gold(packet, rows_by_id[pair_id])
        if leak_report["hidden_leaks"]:
            leak_reports.append(leak_report)
        packets.append(packet)
    if fail_on_leak and leak_reports:
        leaked = [report["pair_id"] for report in leak_reports]
        raise SafeReplayCaptureError(f"hidden gold leakage detected for pairs: {leaked}")
    return {
        "metadata": {
            "schema_version": "retrieval_policy_answer_packets.v1",
            "packet_count": len(packets),
            "hidden_leak_count": len(leak_reports),
        },
        "packets": packets,
        "leak_reports": leak_reports,
    }


def validate_answer_packet_no_hidden_gold(
    packet: dict[str, Any],
    gold_row: SafeReplayGoldRow,
) -> dict[str, Any]:
    """Report hidden evaluator leakage while allowing query/context overlap."""

    hidden_leaks: list[dict[str, str]] = []
    overlap: list[dict[str, str]] = []
    _scan_packet(
        packet,
        gold_row,
        path=(),
        hidden_leaks=hidden_leaks,
        overlap=overlap,
    )
    return {
        "pair_id": gold_row.pair_id,
        "hidden_leaks": hidden_leaks,
        "allowed_overlap": overlap,
    }


def validate_replay_profile_paths(
    *,
    source_db_path: Path,
    replay_profile_dir: Path,
    active_profile_dir: Path | None = None,
) -> None:
    """Reject replay targets that could write into the active profile."""

    source_db = source_db_path.expanduser().resolve()
    replay_dir = replay_profile_dir.expanduser().resolve()
    source_dir = source_db.parent
    active_dir = (
        active_profile_dir.expanduser().resolve()
        if active_profile_dir is not None
        else source_dir
    )
    if replay_dir == source_dir or replay_dir == active_dir:
        raise SafeReplayCaptureError("replay profile directory must not equal active source profile")
    if _is_relative_to(replay_dir, source_dir) or _is_relative_to(replay_dir, active_dir):
        raise SafeReplayCaptureError("replay profile directory must not be inside active profile")
    if _is_relative_to(source_dir, replay_dir) or _is_relative_to(active_dir, replay_dir):
        raise SafeReplayCaptureError("active profile must not be inside replay profile")


def build_trace_payload(
    *,
    gold_payload: dict[str, Any],
    recommendations_payload: dict[str, Any] | None,
    responses_payload: dict[str, Any],
) -> dict[str, Any]:
    """Build trace JSON rows from externally captured isolated API responses."""

    rows = load_gold_rows(gold_payload)
    recommendations = load_recommendations(recommendations_payload, rows)
    responses = _normalize_response_rows(responses_payload)
    trace_rows = []
    for row in rows:
        response_row = responses.get(row.pair_id)
        if response_row is None:
            continue
        affected = recommendations[row.pair_id].recommended_action != KEEP_CURRENT
        trace_rows.append(
            build_trace_row_from_response(
                gold_row=row,
                response_row=response_row,
                affected=affected,
            )
        )
    return {
        "metadata": {
            "schema_version": "retrieval_policy_safe_trace_rows.v1",
            "trace_count": len(trace_rows),
        },
        "traces": trace_rows,
    }


def build_trace_row_from_response(
    *,
    gold_row: SafeReplayGoldRow,
    response_row: dict[str, Any],
    affected: bool,
) -> dict[str, Any]:
    """Convert one captured isolated API response into replay trace shape."""

    response = response_row.get("response") if isinstance(response_row.get("response"), dict) else response_row
    if not isinstance(response, dict):
        raise SafeReplayCaptureError(f"{gold_row.pair_id}: response must be an object")
    trace_id = str(
        response_row.get("trace_id")
        or response.get("trace_id")
        or response.get("resolved_session_id")
        or gold_row.pair_id
    )
    source_final_context = _derive_source_final_context(
        gold_row=gold_row,
        response=response,
        affected=affected,
    )
    latency_components = _latency_components_from_response(
        gold_row=gold_row,
        response_row=response_row,
        affected=affected,
    )
    return {
        "pair_id": gold_row.pair_id,
        "trace_id": trace_id,
        "source_final_context": source_final_context,
        "latency_components_ms": latency_components,
        "latency_ms": _optional_float(response_row.get("latency_ms")),
    }


def _derive_source_final_context(
    *,
    gold_row: SafeReplayGoldRow,
    response: dict[str, Any],
    affected: bool,
) -> dict[str, Any]:
    raw_context = response.get("source_final_context")
    if raw_context is None:
        raw_context = response.get("source_set_trace")
    if raw_context is None:
        raw_context = {}
    if raw_context and not isinstance(raw_context, dict):
        raise SafeReplayCaptureError(f"{gold_row.pair_id}: source context must be an object")
    context = dict(raw_context)
    activated_ids = _activated_ids(response)
    if activated_ids:
        context.setdefault("activated_ids", activated_ids)
    retrieval_policy_trace = response.get("retrieval_policy_trace")
    if isinstance(retrieval_policy_trace, dict):
        candidate_flow = retrieval_policy_trace.get("candidate_flow")
        if isinstance(candidate_flow, dict):
            context.setdefault("source_set_required", candidate_flow.get("source_set_required"))
        context.setdefault("retrieval_policy_trace", retrieval_policy_trace)

    required = affected and gold_row.query_class in SOURCE_DIAGNOSTIC_REQUIRED_CLASSES
    if not required:
        return context
    missing = [
        key
        for key in ("row_break_class", "admission_miss_count")
        if key not in context
    ]
    if missing:
        raise SourceFinalContextUnavailable(
            f"{gold_row.pair_id}: source_final_context_unavailable missing {missing}"
        )
    return context


def _latency_components_from_response(
    *,
    gold_row: SafeReplayGoldRow,
    response_row: dict[str, Any],
    affected: bool,
) -> dict[str, float]:
    raw_components = response_row.get("latency_components_ms")
    if not raw_components:
        nested_response = response_row.get("response")
        if isinstance(nested_response, dict):
            raw_components = nested_response.get("latency_components_ms")
    raw_components = raw_components or {}
    if raw_components and not isinstance(raw_components, dict):
        raise SafeReplayCaptureError(f"{gold_row.pair_id}: latency_components_ms must be an object")
    if affected:
        missing = [key for key in REQUIRED_LATENCY_COMPONENTS if key not in raw_components]
        if missing:
            raise SafeReplayCaptureError(
                f"{gold_row.pair_id}: missing latency_components_ms keys: {missing}"
            )
    components: dict[str, float] = {}
    for key, value in raw_components.items():
        try:
            components[str(key)] = float(value)
        except (TypeError, ValueError) as exc:
            raise SafeReplayCaptureError(
                f"{gold_row.pair_id}: latency component {key!r} must be numeric"
            ) from exc
    return components


def _scan_packet(
    value: Any,
    gold_row: SafeReplayGoldRow,
    *,
    path: tuple[str, ...],
    hidden_leaks: list[dict[str, str]],
    overlap: list[dict[str, str]],
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = (*path, str(key))
            if str(key) in HIDDEN_GOLD_KEYS:
                hidden_leaks.append({"path": ".".join(child_path), "match": str(key)})
            _scan_packet(
                child,
                gold_row,
                path=child_path,
                hidden_leaks=hidden_leaks,
                overlap=overlap,
            )
        return
    if isinstance(value, list):
        for idx, child in enumerate(value):
            _scan_packet(
                child,
                gold_row,
                path=(*path, str(idx)),
                hidden_leaks=hidden_leaks,
                overlap=overlap,
            )
        return
    if value is None:
        return

    text = str(value).casefold()
    if not text:
        return
    allowed_path = path[:1] in {("query",), ("allowed_context",)}
    for expected in (*gold_row.expected_concept_ids, *gold_row.expected_answer_fragments):
        if expected and expected.casefold() in text:
            target = overlap if allowed_path else hidden_leaks
            target.append({"path": ".".join(path), "match": expected})


def _normalize_response_rows(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_rows = payload.get("responses") or payload.get("turns") or payload
    if isinstance(raw_rows, list):
        rows = {}
        for idx, raw in enumerate(raw_rows, start=1):
            if not isinstance(raw, dict):
                raise SafeReplayCaptureError(f"response row {idx}: must be an object")
            pair_id = str(raw.get("pair_id") or raw.get("id") or "").strip()
            if not pair_id:
                raise SafeReplayCaptureError(f"response row {idx}: pair_id is required")
            rows[pair_id] = raw
        return rows
    if isinstance(raw_rows, dict):
        rows = {}
        for pair_id, raw in raw_rows.items():
            if not isinstance(raw, dict):
                raise SafeReplayCaptureError(f"{pair_id}: response row must be an object")
            rows[str(pair_id)] = {**raw, "pair_id": str(pair_id)}
        return rows
    raise SafeReplayCaptureError("responses payload must contain an object or list")


def _manifest_rows(manifest_payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_rows = manifest_payload.get("rows")
    if not isinstance(raw_rows, list):
        raise SafeReplayCaptureError("manifest payload must contain rows list")
    rows: list[dict[str, Any]] = []
    for idx, raw in enumerate(raw_rows, start=1):
        if not isinstance(raw, dict):
            raise SafeReplayCaptureError(f"manifest row {idx}: must be an object")
        rows.append(raw)
    return rows


def _query_overlap(row: SafeReplayGoldRow) -> list[str]:
    normalized_query = row.query.casefold()
    return [
        fragment
        for fragment in row.expected_answer_fragments
        if fragment.casefold() in normalized_query
    ]


def _activated_ids(response: dict[str, Any]) -> list[str]:
    activated = response.get("activated_concepts") or []
    if not isinstance(activated, list):
        return []
    concept_ids = []
    for item in activated:
        if isinstance(item, dict):
            concept_id = str(item.get("concept_id") or item.get("id") or "").strip()
            if concept_id:
                concept_ids.append(concept_id)
    return concept_ids


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise SafeReplayCaptureError("latency_ms must be numeric") from exc


def _string_tuple(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise SafeReplayCaptureError("expected a list of strings")
    return tuple(str(item).strip() for item in raw if str(item).strip())


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False
