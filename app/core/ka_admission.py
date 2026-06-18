"""Storage-safe knowledge-area admission helpers."""

from __future__ import annotations

from app.core.datetime_utils import _utc_now_iso

KA_STATE_CLASSIFIED = "classified"
KA_STATE_INTENTIONAL_GENERAL = "intentional_general"
KA_STATE_DETERMINISTIC_AMBIGUOUS = "deterministic_ambiguous"
KA_STATE_QUEUED_UNCLASSIFIED = "queued_unclassified"
KA_STATE_FALLBACK_FAILURE = "fallback_failure"

KA_ADMISSION_STATES = frozenset(
    {
        KA_STATE_CLASSIFIED,
        KA_STATE_INTENTIONAL_GENERAL,
        KA_STATE_DETERMINISTIC_AMBIGUOUS,
        KA_STATE_QUEUED_UNCLASSIFIED,
        KA_STATE_FALLBACK_FAILURE,
    }
)
KA_QUEUED_UNCLASSIFIED_STALE_HOURS = 24


def is_valid_ka_admission_state(state: object) -> bool:
    return isinstance(state, str) and state in KA_ADMISSION_STATES


def is_ka_duplicate_merge_eligible(metadata: dict | None) -> bool:
    state = (metadata or {}).get("ka_admission_state")
    return state not in {
        KA_STATE_DETERMINISTIC_AMBIGUOUS,
        KA_STATE_QUEUED_UNCLASSIFIED,
        KA_STATE_FALLBACK_FAILURE,
    }


def normalise_ka_value(value: str | None) -> str:
    return (value or "").strip().lower()


def metadata_patch(
    *,
    state: str,
    source: str | None,
    now: str,
    confidence: float | None = None,
) -> dict:
    patch = {
        "ka_admission_state": state,
        "ka_admission_source": source or "unknown",
        "ka_admission_at": now,
    }
    if confidence is not None:
        patch["ka_confidence"] = confidence
    if state in {KA_STATE_INTENTIONAL_GENERAL, KA_STATE_DETERMINISTIC_AMBIGUOUS}:
        patch["ka_reviewed_at"] = now
    if state == KA_STATE_DETERMINISTIC_AMBIGUOUS:
        patch["knowledge_area_review"] = KA_STATE_DETERMINISTIC_AMBIGUOUS
        patch["knowledge_area_reviewed_at"] = now
    return patch


def storage_guard_unreviewed_general(data: dict, *, now: str | None = None) -> tuple[str | None, dict]:
    """Prevent new rows from persisting unreviewed general through storage bypasses."""
    meta = data.setdefault("metadata", {})
    state = meta.get("ka_admission_state")
    if data.get("knowledge_area") != "general" or is_valid_ka_admission_state(state):
        return None, data

    now = now or _utc_now_iso()
    meta.update(
        metadata_patch(
            state=KA_STATE_FALLBACK_FAILURE,
            source="storage_guard",
            now=now,
        )
    )
    meta["knowledge_area"] = "unclassified"
    data["knowledge_area"] = "unclassified"
    return "unclassified", data
