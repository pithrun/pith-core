"""Pure answerability inspection helpers for source-set diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal

from app.cognitive.source_set_answer_realization import (
    SourceSetAnswerShadowComparatorResult,
    build_source_set_answer_shadow_comparator,
)

ANSWERABILITY_INSPECTION_SCHEMA_VERSION = "answerability_inspection.v1"
AnswerabilityStatus = Literal["supported", "unsupported", "conflict", "observability_gap"]


@dataclass(frozen=True)
class AnswerabilityInspection:
    status: AnswerabilityStatus
    source_classification: str
    candidate_kind: str
    support_line_count: int
    matched_support_refs: tuple[dict[str, object], ...]
    matched_concept_ids: tuple[str, ...]
    refusal_reason: str | None
    answer_present: bool
    elapsed_ms: float


def answerability_inspection_from_shadow_result(
    result: SourceSetAnswerShadowComparatorResult,
) -> AnswerabilityInspection:
    status: AnswerabilityStatus
    refusal_reason = result.refusal_reason
    if result.classification == "would_emit":
        status = "supported"
    elif result.classification == "would_conflict":
        status = "conflict"
    elif result.classification == "would_refuse" and result.support_line_count > 0:
        status = "unsupported"
    elif result.classification == "would_refuse":
        status = "observability_gap"
    else:
        status = "observability_gap"
        refusal_reason = refusal_reason or "unknown_comparator_classification"

    return AnswerabilityInspection(
        status=status,
        source_classification=result.classification,
        candidate_kind=result.candidate_kind,
        support_line_count=result.support_line_count,
        matched_support_refs=tuple(asdict(ref) for ref in result.support_refs),
        matched_concept_ids=tuple(result.matched_concept_ids),
        refusal_reason=refusal_reason,
        answer_present=result.answer is not None,
        elapsed_ms=result.elapsed_ms,
    )


def inspect_source_set_answerability(
    *,
    question: str,
    activated_concepts: Sequence[object],
) -> AnswerabilityInspection:
    return answerability_inspection_from_shadow_result(
        build_source_set_answer_shadow_comparator(
            question=question,
            activated_concepts=activated_concepts,
        )
    )


def answerability_inspection_payload(
    inspection: AnswerabilityInspection,
) -> dict[str, object]:
    return {
        "schema_version": ANSWERABILITY_INSPECTION_SCHEMA_VERSION,
        "status": inspection.status,
        "source_classification": inspection.source_classification,
        "candidate_kind": inspection.candidate_kind,
        "support_line_count": inspection.support_line_count,
        "matched_support_refs": list(inspection.matched_support_refs),
        "matched_concept_ids": list(inspection.matched_concept_ids),
        "refusal_reason": inspection.refusal_reason,
        "answer_present": inspection.answer_present,
        "elapsed_ms": round(inspection.elapsed_ms, 3),
    }
