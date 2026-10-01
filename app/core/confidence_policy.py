"""Shared confidence-cap policy for reflection and monitoring."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.core.config import (
    L3_CONCEPT_TYPES,
    MIN_UTILITY_SAMPLES,
    PROVEN_L3_UTILITY_THRESHOLD,
    PSIS_QUARANTINE_CONFIDENCE_CAP,
    PSIS_QUARANTINE_EVIDENCE_MARKER,
)


def is_proven_l3(
    concept_type: str | None,
    utility_score: float | None,
    utility_samples: int | None,
) -> bool:
    """Return whether utility evidence qualifies an L3 concept for the higher cap."""
    return (
        concept_type in L3_CONCEPT_TYPES
        and utility_score is not None
        and utility_score > PROVEN_L3_UTILITY_THRESHOLD
        and (utility_samples or 0) >= MIN_UTILITY_SAMPLES
    )


def effective_confidence_cap(
    *,
    concept_type: str | None,
    utility_score: float | None,
    utility_samples: int | None,
    evidence: Sequence[Any] | None,
    feedback_loop_on: bool,
) -> float:
    """Resolve the strongest applicable confidence cap."""
    if PSIS_QUARANTINE_EVIDENCE_MARKER in (evidence or ()):
        return PSIS_QUARANTINE_CONFIDENCE_CAP
    if not feedback_loop_on:
        return 1.0
    if concept_type in L3_CONCEPT_TYPES:
        return (
            0.85
            if is_proven_l3(concept_type, utility_score, utility_samples)
            else 0.70
        )
    return 0.60


def bound_positive_confidence(current: float, candidate: float, cap: float) -> float:
    """Bound a positive write without turning a boost phase into a demotion."""
    return current if current >= cap else min(candidate, cap)
