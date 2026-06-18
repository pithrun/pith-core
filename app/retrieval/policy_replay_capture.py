"""Offline retrieval-policy production replay evidence assembly.

This module validates externally captured assistant answers and Pith trace
diagnostics, then emits the production-turn replay schema consumed by the
cold-path retrieval policy shadow evaluator. It does not call the live Pith API.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "retrieval_policy_production_replay.v1"
EVIDENCE_TYPE = "production_turn_replay"
KEEP_CURRENT = "keep_current"
SOURCE_DIAGNOSTIC_REQUIRED_CLASSES = (
    "aggregate_source_set",
    "multihop_relation",
    "temporal_current_state",
    "contradiction_supersession_sensitive",
)
REQUIRED_LATENCY_COMPONENTS = (
    "fixed_ms",
    "retrieval_scaled_ms",
    "candidate_extra_ms",
)


class RetrievalPolicyReplayCaptureError(ValueError):
    """Raised when replay evidence cannot safely be assembled."""

    def __init__(
        self,
        message: str,
        *,
        stale_expected_concepts: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.stale_expected_concepts = stale_expected_concepts or []


@dataclass(frozen=True)
class ReplayGoldPair:
    pair_id: str
    query_class: str
    expected_ids: tuple[str, ...]
    answer_support_required: bool
    expected_answer_fragments: tuple[str, ...]
    forbidden_answer_fragments: tuple[str, ...]


@dataclass(frozen=True)
class ReplayRecommendation:
    pair_id: str
    recommended_action: str


def load_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""

    with path.open() as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise RetrievalPolicyReplayCaptureError(f"{path}: expected a JSON object")
    return payload


def build_production_replay_payload(
    *,
    gold_payload: dict[str, Any],
    answers_payload: dict[str, Any],
    trace_payload: dict[str, Any],
    candidate_recommendations_payload: dict[str, Any] | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Build evaluator-compatible production-turn replay evidence."""

    pairs = _load_gold_pairs(gold_payload)
    pair_ids = {pair.pair_id for pair in pairs}
    answers = _normalize_rows(answers_payload, row_keys=("answers", "turns"))
    traces = _normalize_rows(trace_payload, row_keys=("traces", "turns"))
    recommendations = _load_recommendations(candidate_recommendations_payload, pairs)
    affected_pair_ids = {
        pair_id
        for pair_id, recommendation in recommendations.items()
        if recommendation.recommended_action != KEEP_CURRENT
    }

    stale_expected_concepts = []
    if db_path is not None:
        stale_expected_concepts = find_stale_expected_concepts(pairs, db_path)
        if stale_expected_concepts:
            raise RetrievalPolicyReplayCaptureError(
                "gold contains expected concepts that are not active/current",
                stale_expected_concepts=stale_expected_concepts,
            )

    unknown_answer_ids = sorted(set(answers).difference(pair_ids))
    if unknown_answer_ids:
        raise RetrievalPolicyReplayCaptureError(
            f"answers contain pair ids not present in gold: {unknown_answer_ids}"
        )
    unknown_trace_ids = sorted(set(traces).difference(pair_ids))
    if unknown_trace_ids:
        raise RetrievalPolicyReplayCaptureError(
            f"traces contain pair ids not present in gold: {unknown_trace_ids}"
        )

    turns = []
    for pair in pairs:
        answer_row = answers.get(pair.pair_id)
        if answer_row is None:
            raise RetrievalPolicyReplayCaptureError(f"{pair.pair_id}: missing answer row")
        trace_row = traces.get(pair.pair_id, {})
        answer = _answer_from_row(answer_row, pair.pair_id)
        _validate_answer_support(pair, answer)
        affected = pair.pair_id in affected_pair_ids
        source_final_context = _source_final_context_from_row(trace_row, pair, affected=affected)
        latency_components_ms = _latency_components_from_row(trace_row, pair, affected=affected)
        latency_ms = _optional_float(
            answer_row.get("latency_ms", trace_row.get("latency_ms"))
        )
        turns.append(
            {
                "pair_id": pair.pair_id,
                "evidence_type": EVIDENCE_TYPE,
                "answer": answer,
                "latency_ms": latency_ms,
                "trace_id": str(
                    trace_row.get("trace_id")
                    or trace_row.get("turn_id")
                    or answer_row.get("trace_id")
                    or answer_row.get("turn_id")
                    or pair.pair_id
                ),
                "source_final_context": source_final_context,
                "latency_components_ms": latency_components_ms,
            }
        )

    return {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "assembler": "app.retrieval.policy_replay_capture",
            "pair_count": len(pairs),
            "affected_pair_count": len(affected_pair_ids),
            "affected_pair_ids": sorted(affected_pair_ids),
            "promotion_authority": "diagnostic_only_until_shadow_evaluator_passes",
        },
        "turns": turns,
    }


def find_stale_expected_concepts(
    pairs: list[ReplayGoldPair],
    db_path: Path,
) -> list[dict[str, Any]]:
    """Return expected concept labels that are missing or not active/current."""

    concept_to_pairs: dict[str, list[str]] = {}
    for pair in pairs:
        for concept_id in pair.expected_ids:
            concept_to_pairs.setdefault(concept_id, []).append(pair.pair_id)
    if not concept_to_pairs:
        return []

    conn = sqlite3.connect(db_path)
    try:
        columns = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(concepts)").fetchall()
        }
        required_columns = {"id", "status", "is_current"}
        missing_columns = sorted(required_columns.difference(columns))
        if missing_columns:
            raise RetrievalPolicyReplayCaptureError(
                f"{db_path}: concepts table missing columns: {missing_columns}"
            )
        superseded_expr = "superseded_by" if "superseded_by" in columns else "NULL"
        summary_expr = "summary" if "summary" in columns else "NULL"
        placeholders = ",".join("?" for _ in concept_to_pairs)
        rows = conn.execute(
            f"SELECT id, status, is_current, {superseded_expr}, {summary_expr} "
            f"FROM concepts WHERE id IN ({placeholders})",
            sorted(concept_to_pairs),
        ).fetchall()
    except sqlite3.Error as exc:
        raise RetrievalPolicyReplayCaptureError(f"{db_path}: unable to inspect concepts table: {exc}") from exc
    finally:
        conn.close()

    found = {str(row[0]): row for row in rows}
    stale: list[dict[str, Any]] = []
    for concept_id in sorted(concept_to_pairs):
        row = found.get(concept_id)
        if row is None:
            stale.append(
                {
                    "concept_id": concept_id,
                    "pair_ids": concept_to_pairs[concept_id],
                    "status": "missing",
                    "is_current": None,
                    "superseded_by": None,
                    "summary": None,
                }
            )
            continue
        _concept_id, status, is_current, superseded_by, summary = row
        if status != "active" or int(is_current or 0) != 1:
            stale.append(
                {
                    "concept_id": concept_id,
                    "pair_ids": concept_to_pairs[concept_id],
                    "status": status,
                    "is_current": is_current,
                    "superseded_by": superseded_by,
                    "summary": summary,
                }
            )
    return stale


def write_stale_report(path: Path, stale_expected_concepts: list[dict[str, Any]]) -> None:
    """Write a machine-readable stale-gold report."""

    path.write_text(
        json.dumps(
            {
                "schema_version": "retrieval_policy_stale_expected_concepts.v1",
                "stale_expected_concepts": stale_expected_concepts,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _load_gold_pairs(gold_payload: dict[str, Any]) -> list[ReplayGoldPair]:
    raw_pairs = gold_payload.get("pairs")
    if not isinstance(raw_pairs, list):
        raise RetrievalPolicyReplayCaptureError("gold payload must contain a pairs list")

    pairs: list[ReplayGoldPair] = []
    for idx, raw in enumerate(raw_pairs, start=1):
        if not isinstance(raw, dict):
            raise RetrievalPolicyReplayCaptureError(f"gold row {idx}: must be an object")
        pair_id = str(raw.get("id") or "").strip()
        if not pair_id:
            raise RetrievalPolicyReplayCaptureError(f"gold row {idx}: id is required")
        expected_ids = _string_tuple(raw.get("expected_concept_ids"))
        if not expected_ids:
            raise RetrievalPolicyReplayCaptureError(
                f"{pair_id}: expected_concept_ids must be a non-empty list"
            )
        pairs.append(
            ReplayGoldPair(
                pair_id=pair_id,
                query_class=str(raw.get("class") or "").strip(),
                expected_ids=expected_ids,
                answer_support_required=bool(raw.get("answer_support_required", True)),
                expected_answer_fragments=_string_tuple(raw.get("expected_answer_fragments")),
                forbidden_answer_fragments=_string_tuple(raw.get("forbidden_answer_fragments")),
            )
        )
    return pairs


def _load_recommendations(
    payload: dict[str, Any] | None,
    pairs: list[ReplayGoldPair],
) -> dict[str, ReplayRecommendation]:
    pair_ids = {pair.pair_id for pair in pairs}
    recommendations = {
        pair.pair_id: ReplayRecommendation(pair.pair_id, KEEP_CURRENT) for pair in pairs
    }
    if payload is None:
        return recommendations

    raw_recommendations = payload.get("recommendations")
    if not isinstance(raw_recommendations, list):
        raise RetrievalPolicyReplayCaptureError(
            "candidate recommendations payload must contain a recommendations list"
        )
    for idx, raw in enumerate(raw_recommendations, start=1):
        if not isinstance(raw, dict):
            raise RetrievalPolicyReplayCaptureError(
                f"recommendation {idx}: must be an object"
            )
        pair_id = str(raw.get("pair_id") or raw.get("id") or "").strip()
        if not pair_id:
            raise RetrievalPolicyReplayCaptureError(
                f"recommendation {idx}: pair_id is required"
            )
        if pair_id not in pair_ids:
            raise RetrievalPolicyReplayCaptureError(
                f"{pair_id}: recommendation does not match a gold pair"
            )
        action = str(raw.get("recommended_action") or "").strip()
        if not action:
            raise RetrievalPolicyReplayCaptureError(
                f"{pair_id}: recommended_action is required"
            )
        recommendations[pair_id] = ReplayRecommendation(pair_id, action)
    return recommendations


def _normalize_rows(
    payload: dict[str, Any],
    *,
    row_keys: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    raw_rows: Any = None
    for key in row_keys:
        if key in payload:
            raw_rows = payload[key]
            break
    if raw_rows is None:
        raw_rows = payload

    if isinstance(raw_rows, list):
        rows: dict[str, dict[str, Any]] = {}
        for idx, raw in enumerate(raw_rows, start=1):
            if not isinstance(raw, dict):
                raise RetrievalPolicyReplayCaptureError(f"row {idx}: must be an object")
            pair_id = str(raw.get("pair_id") or raw.get("id") or "").strip()
            if not pair_id:
                raise RetrievalPolicyReplayCaptureError(f"row {idx}: pair_id is required")
            rows[pair_id] = raw
        return rows

    if isinstance(raw_rows, dict):
        rows = {}
        for pair_id, raw in raw_rows.items():
            if isinstance(raw, dict):
                rows[str(pair_id)] = {**raw, "pair_id": str(pair_id)}
            else:
                rows[str(pair_id)] = {"pair_id": str(pair_id), "answer": str(raw)}
        return rows

    raise RetrievalPolicyReplayCaptureError("expected rows to be an object or list")


def _answer_from_row(row: dict[str, Any], pair_id: str) -> str:
    answer = str(row.get("answer") or row.get("response") or "").strip()
    if not answer:
        raise RetrievalPolicyReplayCaptureError(f"{pair_id}: answer is required")
    return answer


def _validate_answer_support(pair: ReplayGoldPair, answer: str) -> None:
    expected_hits = [
        fragment
        for fragment in pair.expected_answer_fragments
        if fragment.casefold() in answer.casefold()
    ]
    if (
        pair.answer_support_required
        and len(expected_hits) != len(pair.expected_answer_fragments)
    ):
        missing = sorted(set(pair.expected_answer_fragments).difference(expected_hits))
        raise RetrievalPolicyReplayCaptureError(
            f"{pair.pair_id}: answer missing expected fragments: {missing}"
        )
    forbidden_hits = [
        fragment
        for fragment in pair.forbidden_answer_fragments
        if fragment.casefold() in answer.casefold()
    ]
    if forbidden_hits:
        raise RetrievalPolicyReplayCaptureError(
            f"{pair.pair_id}: answer contains forbidden fragments: {forbidden_hits}"
        )


def _source_final_context_from_row(
    row: dict[str, Any],
    pair: ReplayGoldPair,
    *,
    affected: bool,
) -> dict[str, Any]:
    source_final_context = row.get("source_final_context") or {}
    if source_final_context and not isinstance(source_final_context, dict):
        raise RetrievalPolicyReplayCaptureError(
            f"{pair.pair_id}: source_final_context must be an object"
        )
    if (
        affected
        and pair.query_class in SOURCE_DIAGNOSTIC_REQUIRED_CLASSES
        and not source_final_context
    ):
        raise RetrievalPolicyReplayCaptureError(
            f"{pair.pair_id}: missing source_final_context for source-sensitive pair"
        )
    return source_final_context


def _latency_components_from_row(
    row: dict[str, Any],
    pair: ReplayGoldPair,
    *,
    affected: bool,
) -> dict[str, float]:
    raw_components = row.get("latency_components_ms") or {}
    if raw_components and not isinstance(raw_components, dict):
        raise RetrievalPolicyReplayCaptureError(
            f"{pair.pair_id}: latency_components_ms must be an object"
        )
    if affected:
        missing = [
            key for key in REQUIRED_LATENCY_COMPONENTS if key not in raw_components
        ]
        if missing:
            raise RetrievalPolicyReplayCaptureError(
                f"{pair.pair_id}: missing latency_components_ms keys: {missing}"
            )
    components: dict[str, float] = {}
    for key, value in raw_components.items():
        try:
            components[str(key)] = float(value)
        except (TypeError, ValueError) as exc:
            raise RetrievalPolicyReplayCaptureError(
                f"{pair.pair_id}: latency component {key!r} must be numeric"
            ) from exc
    return components


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise RetrievalPolicyReplayCaptureError("latency_ms must be numeric") from exc


def _string_tuple(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise RetrievalPolicyReplayCaptureError("expected a list of strings")
    return tuple(str(item).strip() for item in raw if str(item).strip())
