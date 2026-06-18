"""Source-backed grouped count packet metadata helpers."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Any

_FLAG = "PITH_GROUPED_COUNT_PACKET_METADATA"

_MAX_TEXT = 160
_MAX_DOMAIN = 80
_MAX_PURPOSE = 240
_MAX_EVIDENCE = 500
_MAX_MEMBERS = 50
_MAX_SOURCE_EVIDENCE = 5

_ALLOWED_DERIVATION_METHODS = frozenset(
    {
        "client_provided_grouped_count_v1",
        "source_text_grouped_count_v1",
        "user_selected_grouped_set_v1",
    }
)
_FORBIDDEN_KEYS = frozenset(
    {
        "answer",
        "answer_string",
        "benchmark_private",
        "expected_answer",
        "expected_source_ref",
        "expected_source_refs",
        "gold_id",
        "gold_ids",
        "qid",
        "question_id",
        "rubric",
        "source_chat_ids",
        "source_ref",
    }
)
_COUNT_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
_COUNT_TOKEN = r"\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten"
_GROUP_BOUNDARY_RE = re.compile(
    rf"\b(?P<count>{_COUNT_TOKEN})\s+"
    r"(?P<label>[a-z][a-z0-9 '\-/]{1,120}?)"
    r"(?:\s*:\s*|\s+(?:are|include|includes|including)\s+)"
    r"(?P<members>[^.\n;]{3,500})",
    re.IGNORECASE,
)
_SELECTION_SIGNAL_RE = re.compile(
    r"\b(?:i|we|user|client)\b.{0,80}"
    r"\b(?:selected|selecting|chose|choose|picked|pick|decided|mentioned|"
    r"want(?:ed)? to explore|want(?:ed)? to cover|listed|named)\b",
    re.IGNORECASE,
)
_ASSISTANT_ONLY_SUGGESTION_RE = re.compile(
    r"\b(?:recommend|recommends|recommended|recommendation|suggest|suggests|suggested|suggestion|"
    r"try|tries|tried|consider|considers|considered|could)\b",
    re.IGNORECASE,
)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _clean_text(value: Any, limit: int = _MAX_TEXT) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip(" \t\r\n-:;,.")
    return text[:limit]


def _coerce_confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


def _parse_count(value: Any) -> int | None:
    text = str(value or "").strip().lower()
    if text.isdigit():
        count = int(text)
    else:
        count = _COUNT_WORDS.get(text)
    if count is None or count < 1 or count > _MAX_MEMBERS:
        return None
    return count


def contains_forbidden_grouped_count_packet_material(value: object) -> bool:
    """Return true when a packet contains benchmark-private or answer-key material."""
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).casefold() in _FORBIDDEN_KEYS:
                return True
            if contains_forbidden_grouped_count_packet_material(nested):
                return True
    elif isinstance(value, (list, tuple)):
        return any(contains_forbidden_grouped_count_packet_material(item) for item in value)
    return False


def _sanitize_source_evidence(items: Any) -> list[Any]:
    if not isinstance(items, list):
        return []
    sanitized: list[Any] = []
    for item in items[:_MAX_SOURCE_EVIDENCE]:
        if isinstance(item, str):
            text = _clean_text(item, _MAX_EVIDENCE)
            if text:
                sanitized.append(text)
        elif isinstance(item, Mapping):
            entry: dict[str, str] = {}
            for key in ("verbatim", "role", "evidence_id", "fragment_id", "session_id", "turn_id"):
                value = item.get(key)
                if value not in (None, ""):
                    entry[key] = _clean_text(value, _MAX_EVIDENCE)
            if entry:
                sanitized.append(entry)
    return sanitized


def _split_members(text: str) -> list[str]:
    normalized = re.sub(r"\b(?:and|or)\b", ",", text, flags=re.IGNORECASE)
    parts = [
        _clean_text(re.sub(r"^(?:the|a|an)\s+", "", part.strip(), flags=re.IGNORECASE))
        for part in normalized.split(",")
    ]
    members: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if not part:
            continue
        key = part.casefold()
        if key in seen:
            continue
        members.append(part)
        seen.add(key)
    return members[:_MAX_MEMBERS]


def sanitize_grouped_count_packet(raw: Mapping[str, object]) -> dict[str, object]:
    """Return a bounded grouped count packet, or `{}` when unsafe or malformed."""
    if contains_forbidden_grouped_count_packet_material(raw):
        return {}

    method = str(raw.get("derivation_method") or "")
    if method not in _ALLOWED_DERIVATION_METHODS:
        return {}

    count = _parse_count(raw.get("count"))
    members_raw = raw.get("members")
    if count is None or not isinstance(members_raw, list):
        return {}

    members = []
    seen: set[str] = set()
    for item in members_raw[:_MAX_MEMBERS]:
        member = _clean_text(item)
        if not member:
            continue
        key = member.casefold()
        if key not in seen:
            members.append(member)
            seen.add(key)
    if not members or count != len(members):
        return {}

    source_evidence = _sanitize_source_evidence(raw.get("source_evidence"))
    if not source_evidence:
        return {}

    group_label = _clean_text(raw.get("group_label"))
    if not group_label:
        return {}

    packet: dict[str, object] = {
        "schema_version": 1,
        "packet_type": "grouped_count",
        "group_label": group_label,
        "count": count,
        "members": members,
        "source_evidence": source_evidence,
        "derivation_method": method,
        "confidence": _coerce_confidence(raw.get("confidence")),
    }
    for key, limit in (
        ("subject", _MAX_TEXT),
        ("domain", _MAX_DOMAIN),
        ("purpose", _MAX_PURPOSE),
        ("observed_at", _MAX_DOMAIN),
    ):
        value = _clean_text(raw.get(key), limit)
        if value:
            packet[key] = value
    return packet


def derive_grouped_count_packet(
    summary: str,
    evidence_text: str,
    user_message: str,
    assistant_response: str,
    confidence: object,
) -> dict[str, object]:
    """Derive a conservative grouped count packet from source-visible text."""
    source_candidates = [
        text
        for text in (
            summary,
            evidence_text,
            " ".join(text for text in (summary, evidence_text) if text),
        )
        if text and text.strip()
    ]
    for source_text in source_candidates:
        if _ASSISTANT_ONLY_SUGGESTION_RE.search(source_text) and not _SELECTION_SIGNAL_RE.search(source_text):
            continue

        match = _GROUP_BOUNDARY_RE.search(source_text)
        if not match:
            continue

        count = _parse_count(match.group("count"))
        members = _split_members(match.group("members"))
        if count is None or count != len(members):
            continue

        method = (
            "user_selected_grouped_set_v1"
            if _SELECTION_SIGNAL_RE.search(source_text)
            else "source_text_grouped_count_v1"
        )
        packet = {
            "schema_version": 1,
            "packet_type": "grouped_count",
            "subject": "user" if re.search(r"\b(?:i|we|user|client)\b", source_text, re.I) else "source",
            "domain": "general",
            "group_label": match.group("label"),
            "count": count,
            "members": members,
            "source_evidence": [{"verbatim": source_text[:_MAX_EVIDENCE], "role": "user"}],
            "derivation_method": method,
            "confidence": _coerce_confidence(confidence),
        }
        sanitized = sanitize_grouped_count_packet(packet)
        if sanitized:
            return sanitized
    return {}


def normalise_grouped_count_packet_metadata(
    insight: Mapping[str, object],
    request: object,
) -> dict[str, object]:
    """Return grouped count packet metadata when enabled and source-backed."""
    if not _env_flag(_FLAG):
        return {}

    client_metadata = insight.get("metadata") if isinstance(insight.get("metadata"), Mapping) else {}
    raw_packet = client_metadata.get("grouped_count_packet") if isinstance(client_metadata, Mapping) else None
    packet = sanitize_grouped_count_packet(raw_packet) if isinstance(raw_packet, Mapping) else {}
    if not packet:
        evidence_text = " ".join(str(item) for item in insight.get("evidence", []) if item)
        packet = derive_grouped_count_packet(
            summary=str(insight.get("summary") or ""),
            evidence_text=evidence_text,
            user_message=getattr(request, "user_message", "") or "",
            assistant_response=getattr(request, "assistant_response", "") or "",
            confidence=insight.get("confidence"),
        )
    return {"grouped_count_packet": packet} if packet else {}
