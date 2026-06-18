"""Source-backed exact-detail packet metadata helpers."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping
from typing import Any

_FLAG = "PITH_EXACT_DETAIL_PACKET_METADATA"

_MAX_FIELD = 240
_MAX_SOURCE = 1200

_GROUNDING_KEYS = frozenset(
    {
        "evidence_role",
        "slot_subject",
        "slot_attribute",
        "slot_group_id",
        "grounding_priority",
    }
)
_FORBIDDEN_KEYS = frozenset(
    {
        "answer",
        "answer_key",
        "answer_string",
        "benchmark_private",
        "expected_answer",
        "expected_source_ref",
        "expected_source_refs",
        "gold_answer",
        "gold_id",
        "judge_score",
        "pass_fail_label",
        "qid",
        "question_id",
        "row_id",
        "rubric",
        "score_label",
        "source_chat_ids",
    }
)
_FORBIDDEN_VALUE_RE = re.compile(
    r"\b(?:expected answer|expected source|gold answer|gold id|judge score|"
    r"pass/fail|pass fail|question id|row id|rubric|source chat ids)\b",
    re.IGNORECASE,
)
_SUBJECT_USER_RE = re.compile(r"\b(?:i|me|my|mine|user|client)\b", re.IGNORECASE)
_SUBJECT_ASSISTANT_RE = re.compile(r"\bassistant\b", re.IGNORECASE)

_ATTRIBUTE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("price", re.compile(r"\b(?:ticket price|price|cost|amount|total|budget)\b", re.IGNORECASE)),
    ("color", re.compile(r"\b(?:color|colour)\b", re.IGNORECASE)),
    ("size", re.compile(r"\b(?:size|dimension|dimensions)\b", re.IGNORECASE)),
    ("address", re.compile(r"\b(?:address|street address)\b", re.IGNORECASE)),
    ("location", re.compile(r"\b(?:location|place|venue|city|room)\b", re.IGNORECASE)),
    ("date", re.compile(r"\b(?:date|day|deadline)\b", re.IGNORECASE)),
    ("time", re.compile(r"\b(?:time|hour|appointment time)\b", re.IGNORECASE)),
    ("name", re.compile(r"\b(?:name|person|contact)\b", re.IGNORECASE)),
    ("title", re.compile(r"\b(?:title|book title|movie title|song title)\b", re.IGNORECASE)),
    ("model", re.compile(r"\b(?:model|version|sku)\b", re.IGNORECASE)),
    ("confirmation_code", re.compile(r"\b(?:confirmation code|code|pin)\b", re.IGNORECASE)),
    ("order_id", re.compile(r"\b(?:order id|order number|reservation id|booking id)\b", re.IGNORECASE)),
    ("phone", re.compile(r"\b(?:phone|phone number|telephone)\b", re.IGNORECASE)),
    ("email", re.compile(r"\b(?:email|email address)\b", re.IGNORECASE)),
)
_EXACT_VALUE_RE = re.compile(
    r"(?:"
    r"\$[0-9][0-9,]*(?:\.[0-9]{1,2})?"
    r"|\b[0-9][0-9,]*(?:\.[0-9]+)?\b"
    r"|[A-Z0-9]{2,}(?:-[A-Z0-9]{2,})+"
    r"|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    r"|\+?[0-9][0-9() .-]{6,}[0-9]"
    r"|\"[^\"]{2,80}\""
    r"|(?<![A-Za-z])'[^']{2,80}'(?![A-Za-z])"
    r")"
)
_COLOR_VALUE_RE = re.compile(
    r"\b(?:red|orange|yellow|green|blue|purple|pink|black|white|gray|grey|"
    r"brown|silver|gold|navy|teal|maroon|beige|cream)\b",
    re.IGNORECASE,
)
_PROPER_VALUE_RE = re.compile(r"\b(?:is|was|=|:)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4})\b")


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _clean(value: Any, limit: int = _MAX_FIELD) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip(" \t\r\n-:;,.")[:limit]


def contains_forbidden_exact_detail_material(value: object) -> bool:
    """Return true when input contains benchmark-private or answer-key material."""
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).casefold() in _FORBIDDEN_KEYS:
                return True
            if contains_forbidden_exact_detail_material(nested):
                return True
    elif isinstance(value, (list, tuple)):
        return any(contains_forbidden_exact_detail_material(item) for item in value)
    elif isinstance(value, str):
        return bool(_FORBIDDEN_VALUE_RE.search(value))
    return False


def _source_text(insight: Mapping[str, object], request: object) -> str:
    evidence = " ".join(str(item) for item in insight.get("evidence", []) if item)
    parts = [
        str(insight.get("summary") or ""),
        evidence,
    ]
    return _clean(" ".join(part for part in parts if part), _MAX_SOURCE)


def _slot_subject(text: str) -> str:
    if _SUBJECT_USER_RE.search(text):
        return "user"
    if _SUBJECT_ASSISTANT_RE.search(text):
        return "assistant"
    return "source"


def _attribute_match(text: str) -> tuple[str, re.Match[str]] | None:
    for attribute, pattern in _ATTRIBUTE_PATTERNS:
        match = pattern.search(text)
        if match:
            return attribute, match
    return None


def _has_source_value(attribute: str, text: str, match: re.Match[str]) -> tuple[bool, float]:
    window = text[match.start() : match.end() + 160]
    if _EXACT_VALUE_RE.search(window):
        return True, 0.9
    if attribute == "color" and _COLOR_VALUE_RE.search(window):
        return True, 0.75
    if attribute in {"name", "title", "location", "model"} and _PROPER_VALUE_RE.search(window):
        return True, 0.75
    return False, 0.0


def derive_exact_detail_packet_metadata(source_text: str) -> dict[str, object]:
    """Derive one conservative exact-detail packet from source-visible text."""
    text = _clean(source_text, _MAX_SOURCE)
    if not text:
        return {}
    found = _attribute_match(text)
    if not found:
        return {}
    attribute, match = found
    has_value, priority = _has_source_value(attribute, text, match)
    if not has_value:
        return {}

    subject = _slot_subject(text)
    source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    slot_group_hash = hashlib.sha1(f"{subject}|{attribute}|{source_hash}".encode("utf-8")).hexdigest()[:16]
    return {
        "evidence_role": "exact_detail",
        "slot_subject": subject,
        "slot_attribute": attribute,
        "slot_group_id": f"exact_detail:{slot_group_hash}",
        "grounding_priority": priority,
    }


def normalise_exact_detail_packet_metadata(
    insight: Mapping[str, object],
    request: object,
) -> dict[str, object]:
    """Return exact-detail grounding metadata when enabled and source-backed."""
    if not _env_flag(_FLAG):
        return {}
    if not isinstance(insight, Mapping):
        return {}

    client_metadata = insight.get("metadata") if isinstance(insight.get("metadata"), Mapping) else {}
    if any(key in client_metadata for key in _GROUNDING_KEYS):
        return {}
    if contains_forbidden_exact_detail_material(insight):
        return {}

    return derive_exact_detail_packet_metadata(_source_text(insight, request))
