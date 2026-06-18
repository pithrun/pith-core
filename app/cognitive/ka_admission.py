"""Knowledge-area admission state resolution helpers."""

from __future__ import annotations

from app.core.datetime_utils import _utc_now_iso
from app.core.ka_admission import (
    KA_ADMISSION_STATES,
    KA_QUEUED_UNCLASSIFIED_STALE_HOURS,
    KA_STATE_CLASSIFIED,
    KA_STATE_DETERMINISTIC_AMBIGUOUS,
    KA_STATE_FALLBACK_FAILURE,
    KA_STATE_INTENTIONAL_GENERAL,
    KA_STATE_QUEUED_UNCLASSIFIED,
    is_ka_duplicate_merge_eligible,
    is_valid_ka_admission_state,
    metadata_patch,
    normalise_ka_value,
    storage_guard_unreviewed_general,
)

_KA_SENTINELS = {None, "", "general", "unclassified", "unknown"}

__all__ = [
    "KA_ADMISSION_STATES",
    "KA_QUEUED_UNCLASSIFIED_STALE_HOURS",
    "KA_STATE_CLASSIFIED",
    "KA_STATE_DETERMINISTIC_AMBIGUOUS",
    "KA_STATE_FALLBACK_FAILURE",
    "KA_STATE_INTENTIONAL_GENERAL",
    "KA_STATE_QUEUED_UNCLASSIFIED",
    "is_ka_duplicate_merge_eligible",
    "is_valid_ka_admission_state",
    "resolve_ka_admission",
    "storage_guard_unreviewed_general",
]


def _normalise(value: str | None) -> str:
    return normalise_ka_value(value)


def _safe_non_general_classification(summary: str) -> tuple[str | None, str | None, float | None]:
    from app.cognitive.taxonomy import classify_knowledge_area

    classified, source, confidence = classify_knowledge_area(summary or "", "general", strict=True)
    if classified and classified not in _KA_SENTINELS:
        if source == "inferred" or (source == "embedding" and confidence is not None and confidence >= 0.80):
            return classified, source, confidence
    return None, source, confidence


def _metadata_patch(**kwargs) -> dict:
    return metadata_patch(**kwargs)


def resolve_ka_admission(
    *,
    summary: str,
    knowledge_area: str | None,
    ka_source: str | None,
    ka_confidence: float | None = None,
    raw_area: str | None = None,
    extraction_source: str | None = None,
    trusted_intentional_general: bool = False,
    now: str | None = None,
) -> tuple[str, dict]:
    """Resolve final KA plus admission metadata for a new concept."""
    now = now or _utc_now_iso()
    area = _normalise(knowledge_area)
    raw = _normalise(raw_area)
    source = _normalise(ka_source)
    extractor = _normalise(extraction_source)

    if area and area not in _KA_SENTINELS:
        return area, _metadata_patch(
            state=KA_STATE_CLASSIFIED,
            source=ka_source,
            now=now,
            confidence=ka_confidence,
        )

    if trusted_intentional_general and area == "general":
        return "general", _metadata_patch(
            state=KA_STATE_INTENTIONAL_GENERAL,
            source=ka_source or extraction_source,
            now=now,
            confidence=ka_confidence,
        )

    try:
        safe_area, safe_source, safe_confidence = _safe_non_general_classification(summary)
    except Exception:
        return "unclassified", _metadata_patch(
            state=KA_STATE_FALLBACK_FAILURE,
            source=ka_source or extraction_source or "classifier_error",
            now=now,
            confidence=ka_confidence,
        )

    if safe_area:
        return safe_area, _metadata_patch(
            state=KA_STATE_CLASSIFIED,
            source=safe_source,
            now=now,
            confidence=safe_confidence,
        )

    if raw == "general" and (extractor in {"client", "propose"} or source in {"canonical", "client"}):
        return "general", _metadata_patch(
            state=KA_STATE_DETERMINISTIC_AMBIGUOUS,
            source=ka_source or extraction_source,
            now=now,
            confidence=ka_confidence,
        )

    return "unclassified", _metadata_patch(
        state=KA_STATE_QUEUED_UNCLASSIFIED,
        source=ka_source or extraction_source or "default",
        now=now,
        confidence=ka_confidence,
    )
