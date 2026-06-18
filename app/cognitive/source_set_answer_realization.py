"""Observe-only source-set answer realization helpers.

This module is intentionally pure. Runtime callers may use it to build a
diagnostic candidate, but it does not mutate answer text or response payloads.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

SCHEMA_VERSION = "source_set_answer_dry_run.v1"
SHADOW_COMPARATOR_SCHEMA_VERSION = "source_set_answer_shadow_comparator.v1"
SHADOW_ATTRIBUTION_VALUE_MAX_LENGTH = 160
INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"

_CORRECTION_TAG_PATTERN = re.compile(r"\[CORRECTED[^\]]*(?:\]|$)", flags=re.IGNORECASE)
_RELATIVE_DATE_TAG_PATTERN = re.compile(r"\[as of [^\]]*(?:\]|$)", flags=re.IGNORECASE)
_QUERY_SUPPORT_SURFACE_PATTERNS = (
    re.compile(r"claim set:\s*(.+)$", flags=re.IGNORECASE | re.DOTALL),
    re.compile(r"around this claim:\s*(.+)$", flags=re.IGNORECASE | re.DOTALL),
    re.compile(r"current claim should win[^:]*:\s*(.+)$", flags=re.IGNORECASE | re.DOTALL),
    re.compile(r"identified by\s+.+/\s*(.+)$", flags=re.IGNORECASE | re.DOTALL),
)


@dataclass(frozen=True)
class SourceSetSupportLine:
    support_id: str
    concept_id: str | None
    text: str
    source: str
    source_index: int


@dataclass(frozen=True)
class SourceSetSupportRef:
    support_id: str
    concept_id: str | None
    source: str
    source_index: int
    support_text_hash: str


@dataclass(frozen=True)
class SourceSetAnswerDryRunResult:
    enabled: bool
    would_emit: bool
    would_refuse: bool
    answer: str | None
    support_ids: tuple[str, ...]
    matched_concept_id: str | None
    query_surface: str
    normalized_query_surface: str
    support_line_count: int
    refusal_reason: str | None
    elapsed_ms: float


@dataclass(frozen=True)
class SourceSetAnswerShadowComparatorResult:
    enabled: bool
    candidate_kind: str
    classification: str
    answer: str | None
    support_refs: tuple[SourceSetSupportRef, ...]
    matched_concept_ids: tuple[str, ...]
    query_surface: str
    normalized_query_surface: str
    support_line_count: int
    refusal_reason: str | None
    elapsed_ms: float


def _strip_source_set_annotations(text: str) -> str:
    cleaned = _CORRECTION_TAG_PATTERN.sub("", text)
    cleaned = _RELATIVE_DATE_TAG_PATTERN.sub("", cleaned)
    return cleaned.strip()


def _normalize_source_set_text(text: str) -> str:
    cleaned = _strip_source_set_annotations(text).lower()
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
    return " ".join(cleaned.split())


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _excerpt(text: str | None, limit: int) -> str | None:
    if text is None:
        return None
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip()


def _field_value(item: object, name: str) -> Any:
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


def _string_values(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        stripped = value.strip()
        return (stripped,) if stripped else ()
    if isinstance(value, Sequence) and not isinstance(value, bytes):
        output: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                output.append(item.strip())
        return tuple(output)
    return ()


def extract_source_set_query_surface(question: str) -> str:
    for pattern in _QUERY_SUPPORT_SURFACE_PATTERNS:
        match = pattern.search(question)
        if match:
            return _strip_source_set_annotations(match.group(1))
    return _strip_source_set_annotations(question)


def collect_source_set_support_lines(
    activated_concepts: Sequence[object],
) -> tuple[SourceSetSupportLine, ...]:
    support_lines: list[SourceSetSupportLine] = []
    for concept_index, concept in enumerate(activated_concepts):
        concept_id = _field_value(concept, "concept_id") or _field_value(concept, "id")
        clean_concept_id = str(concept_id) if concept_id is not None else None
        for source, values in (
            ("summary", _string_values(_field_value(concept, "summary"))),
            ("key_evidence", _string_values(_field_value(concept, "key_evidence"))),
        ):
            for value_index, text in enumerate(values):
                support_lines.append(
                    SourceSetSupportLine(
                        support_id=f"c{concept_index}:{source}:{value_index}",
                        concept_id=clean_concept_id,
                        text=text,
                        source=source,
                        source_index=value_index,
                    )
                )
    return tuple(support_lines)


def _exact_matches(
    normalized_query_surface: str,
    support_lines: Sequence[SourceSetSupportLine],
) -> tuple[SourceSetSupportLine, ...]:
    if not normalized_query_surface:
        return ()
    return tuple(
        support_line
        for support_line in support_lines
        if normalized_query_surface in _normalize_source_set_text(support_line.text)
    )


def _exact_match(
    normalized_query_surface: str,
    support_lines: Sequence[SourceSetSupportLine],
) -> SourceSetSupportLine | None:
    matches = _exact_matches(normalized_query_surface, support_lines)
    return matches[0] if matches else None


def _support_ref(support_line: SourceSetSupportLine) -> SourceSetSupportRef:
    return SourceSetSupportRef(
        support_id=support_line.support_id,
        concept_id=support_line.concept_id,
        source=support_line.source,
        source_index=support_line.source_index,
        support_text_hash=_sha256_text(support_line.text),
    )


def build_source_set_answer_dry_run(
    question: str,
    activated_concepts: Sequence[object],
) -> SourceSetAnswerDryRunResult:
    started = time.perf_counter()
    query_surface = extract_source_set_query_surface(question)
    normalized_query_surface = _normalize_source_set_text(query_surface)
    support_lines = collect_source_set_support_lines(activated_concepts)
    matched = _exact_match(normalized_query_surface, support_lines)
    elapsed_ms = (time.perf_counter() - started) * 1000
    if matched is None:
        return SourceSetAnswerDryRunResult(
            enabled=True,
            would_emit=False,
            would_refuse=True,
            answer=None,
            support_ids=(),
            matched_concept_id=None,
            query_surface=query_surface,
            normalized_query_surface=normalized_query_surface,
            support_line_count=len(support_lines),
            refusal_reason="query_surface_not_present_in_context",
            elapsed_ms=elapsed_ms,
        )
    return SourceSetAnswerDryRunResult(
        enabled=True,
        would_emit=True,
        would_refuse=False,
        answer=matched.text,
        support_ids=(matched.support_id,),
        matched_concept_id=matched.concept_id,
        query_surface=query_surface,
        normalized_query_surface=normalized_query_surface,
        support_line_count=len(support_lines),
        refusal_reason=None,
        elapsed_ms=elapsed_ms,
    )


def source_set_answer_dry_run_event_payload(
    result: SourceSetAnswerDryRunResult,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "enabled": result.enabled,
        "would_emit": result.would_emit,
        "would_refuse": result.would_refuse,
        "query_surface_hash": _sha256_text(result.query_surface),
        "query_surface_excerpt": _excerpt(result.query_surface, 160),
        "normalized_query_surface_hash": _sha256_text(result.normalized_query_surface),
        "matched_support_id": result.support_ids[0] if result.support_ids else None,
        "matched_concept_id": result.matched_concept_id,
        "support_line_count": result.support_line_count,
        "candidate_answer_excerpt": _excerpt(result.answer, 240),
        "candidate_answer_hash": _sha256_text(result.answer) if result.answer else None,
        "refusal_reason": result.refusal_reason,
        "elapsed_ms": round(result.elapsed_ms, 3),
    }


def build_source_set_answer_shadow_comparator(
    question: str,
    activated_concepts: Sequence[object],
) -> SourceSetAnswerShadowComparatorResult:
    started = time.perf_counter()
    query_surface = extract_source_set_query_surface(question)
    normalized_query_surface = _normalize_source_set_text(query_surface)
    support_lines = collect_source_set_support_lines(activated_concepts)
    matches = _exact_matches(normalized_query_surface, support_lines)
    elapsed_ms = (time.perf_counter() - started) * 1000
    if not matches:
        return SourceSetAnswerShadowComparatorResult(
            enabled=True,
            candidate_kind="structured_provenance",
            classification="would_refuse",
            answer=None,
            support_refs=(),
            matched_concept_ids=(),
            query_surface=query_surface,
            normalized_query_surface=normalized_query_surface,
            support_line_count=len(support_lines),
            refusal_reason="query_surface_not_present_in_context",
            elapsed_ms=elapsed_ms,
        )
    normalized_answers = {_normalize_source_set_text(match.text) for match in matches}
    if len(normalized_answers) > 1:
        return SourceSetAnswerShadowComparatorResult(
            enabled=True,
            candidate_kind="structured_provenance",
            classification="would_conflict",
            answer=None,
            support_refs=tuple(_support_ref(match) for match in matches),
            matched_concept_ids=tuple(
                sorted({match.concept_id for match in matches if match.concept_id is not None})
            ),
            query_surface=query_surface,
            normalized_query_surface=normalized_query_surface,
            support_line_count=len(support_lines),
            refusal_reason="conflicting_support_lines",
            elapsed_ms=elapsed_ms,
        )
    matched = matches[0]
    return SourceSetAnswerShadowComparatorResult(
        enabled=True,
        candidate_kind="structured_provenance",
        classification="would_emit",
        answer=matched.text,
        support_refs=(_support_ref(matched),),
        matched_concept_ids=tuple(
            concept_id for concept_id in (matched.concept_id,) if concept_id is not None
        ),
        query_surface=query_surface,
        normalized_query_surface=normalized_query_surface,
        support_line_count=len(support_lines),
        refusal_reason=None,
        elapsed_ms=elapsed_ms,
    )


def source_set_answer_shadow_comparator_event_payload(
    result: SourceSetAnswerShadowComparatorResult,
    *,
    include_excerpts: bool = False,
    request_id: str | None = None,
    origin_id: str | None = None,
    shadow_run_id: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": SHADOW_COMPARATOR_SCHEMA_VERSION,
        "enabled": result.enabled,
        "candidate_kind": result.candidate_kind,
        "classification": result.classification,
        "query_surface_hash": _sha256_text(result.query_surface),
        "normalized_query_surface_hash": _sha256_text(result.normalized_query_surface),
        "matched_support_ids": [ref.support_id for ref in result.support_refs],
        "matched_concept_ids": list(result.matched_concept_ids),
        "matched_support_refs": [asdict(ref) for ref in result.support_refs],
        "support_line_count": result.support_line_count,
        "candidate_answer_hash": _sha256_text(result.answer) if result.answer else None,
        "refusal_reason": result.refusal_reason,
        "elapsed_ms": round(result.elapsed_ms, 3),
    }
    for key, value in (
        ("request_id", request_id),
        ("origin_id", origin_id),
        ("shadow_run_id", shadow_run_id),
    ):
        cleaned_value = _bounded_shadow_attribution_value(value)
        if cleaned_value is not None:
            payload[key] = cleaned_value
    if include_excerpts:
        payload["query_surface_excerpt"] = _excerpt(result.query_surface, 160)
        payload["candidate_answer_excerpt"] = _excerpt(result.answer, 240)
    return payload


def _bounded_shadow_attribution_value(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned:
        return None
    return cleaned[:SHADOW_ATTRIBUTION_VALUE_MAX_LENGTH]
