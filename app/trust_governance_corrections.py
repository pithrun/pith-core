"""Read-only trust-governance correction candidate detection."""

from __future__ import annotations

import re
from typing import Any

SCHEMA_VERSION = "trust_governance_correction_candidate.v0"

_PATTERNS: tuple[tuple[str, str, float, str], ...] = (
    ("mark_source_of_truth", r"\b(source of truth|authoritative|authority)\b", 0.82, "source-of-truth marker"),
    ("mark_current", r"\b(this is current|this should govern|use this now|current version)\b", 0.8, "current marker"),
    ("mark_outdated", r"\b(outdated|stale|old|no longer current)\b", 0.78, "outdated marker"),
    ("mark_superseded", r"\b(supersedes|superseded by|replaces|replaced by)\b", 0.82, "supersession marker"),
    ("mark_not_authoritative", r"\b(not authoritative|do not treat .* authoritative|should not govern)\b", 0.84, "negative authority marker"),
)
_QUESTION_LIKE_PATTERN = re.compile(
    r"^\s*(can|could|would|why|what|which|show|tell|explain|is|are|does|do|should)\b",
    re.IGNORECASE,
)


def classify_trust_governance_correction(message: str | None) -> dict[str, Any]:
    text = (message or "").strip()
    lowered = text.casefold()
    if _looks_like_question(text):
        return {
            "schema_version": SCHEMA_VERSION,
            "candidate": False,
            "intent": None,
            "mutates_authority": False,
            "confidence": 0.0,
            "reason": "question-like text is not a correction declaration",
        }

    best: tuple[str, float, str] | None = None
    for intent, pattern, confidence, reason in _PATTERNS:
        if re.search(pattern, lowered):
            if best is None or confidence > best[1]:
                best = (intent, confidence, reason)

    if best is None:
        return {
            "schema_version": SCHEMA_VERSION,
            "candidate": False,
            "intent": None,
            "mutates_authority": False,
            "confidence": 0.0,
            "reason": "no trust-governance correction marker",
        }

    intent, confidence, reason = best
    return {
        "schema_version": SCHEMA_VERSION,
        "candidate": True,
        "intent": intent,
        "mutates_authority": False,
        "confidence": confidence,
        "reason": reason,
    }


def _looks_like_question(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped.endswith("?") or _QUESTION_LIKE_PATTERN.search(stripped))
