"""Canonical searchable-text assembly for the retrieval index (RETRIEVAL-125, A3).

ONE shared helper used by every row-based text-assembly site so the stale-index
audit, the incremental add path, and the in-place refresh path produce
byte-identical searchable text. Divergence between these sites silently corrupts
staleness measurement (a concept indexed with text X but audited against text Y
reads as false-stale or false-fresh), which is exactly the failure A3 closes.

Scope note: this is the *row-based* assembly (operates on a DB row / mapping with
a ``data`` JSON blob + ``summary`` + ``fragment_keywords``). The Pydantic-object
rebuild path (``RetrievalEngine._concept_to_document`` → ``build_index``) is a
separate assembly that additionally folds in ``concept.hypotheses`` (a top-level
column, NOT present in the ``data`` blob). Unifying that path is out of A3 scope
(whole-corpus blast radius) and tracked separately.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

EMBEDDING_TEXT_CONTRACT_VERSION = 1

__all__ = [
    "EMBEDDING_TEXT_CONTRACT_VERSION",
    "build_searchable_text",
    "build_searchable_text_from_concept",
    "embedding_freshness_metadata",
    "embedding_text_hash",
    "parse_json_blob",
    "stringify_list",
]


def parse_json_blob(value: Any) -> dict[str, Any]:
    """Parse a concept ``data`` blob into a dict; tolerate dicts, JSON, junk."""
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def stringify_list(values: Any) -> str:
    if not isinstance(values, list):
        return ""
    return " ".join(str(value) for value in values if value is not None)


def embedding_text_hash(text: str) -> str:
    """Stable provenance hash for the exact text used to build an embedding."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def embedding_freshness_metadata(text: str, *, refreshed_at: str | None = None) -> dict[str, Any]:
    return {
        "embedding_text_hash": embedding_text_hash(text),
        "embedding_text_contract_version": EMBEDDING_TEXT_CONTRACT_VERSION,
        "embedding_refreshed_at": refreshed_at or datetime.now(UTC).isoformat(),
    }


def _row_get(row: Mapping[str, Any] | Any, key: str) -> Any:
    """Read ``key`` from a sqlite3.Row or a plain mapping, tolerating absence."""
    # sqlite3.Row exposes .keys() but not .get(); mappings have .get().
    keys = getattr(row, "keys", None)
    if callable(keys):
        try:
            return row[key] if key in row.keys() else None
        except (IndexError, KeyError):
            return None
    if isinstance(row, Mapping):
        return row.get(key)
    return None


def build_searchable_text(row: Mapping[str, Any] | Any) -> str:
    """Assemble the searchable text for a concept row.

    Mirrors the field set and ordering historically inlined in
    ``RetrievalEngine._add_concept_inner``: summary, knowledge_area,
    concept_type, evidence, signals, implications, events, fragment_keywords.
    Empty parts are dropped and the result is stripped.
    """
    data = parse_json_blob(_row_get(row, "data"))

    summary = data.get("summary", "") or (_row_get(row, "summary") or "") or ""

    evidence_texts: list[str] = []
    for evidence in data.get("evidence") or []:
        if isinstance(evidence, str):
            evidence_texts.append(evidence)
        elif isinstance(evidence, dict):
            evidence_texts.append(str(evidence.get("content", "")))

    metadata = data.get("metadata")
    knowledge_area = metadata.get("knowledge_area", "") if isinstance(metadata, dict) else ""
    concept_type = data.get("concept_type", "")

    implications_text = stringify_list(data.get("implications"))

    event_texts: list[str] = []
    for event in data.get("events", []):
        if not isinstance(event, dict):
            continue
        event_parts = [str(event.get("action", ""))]
        if event.get("cause"):
            event_parts.append(f"because {event['cause']}")
        if event.get("consequence"):
            event_parts.append(f"resulting in {event['consequence']}")
        if event.get("actors"):
            actors = event["actors"]
            if isinstance(actors, list):
                event_parts.append(f"involving {', '.join(str(actor) for actor in actors)}")
            else:
                event_parts.append(f"involving {actors}")
        event_texts.append(" ".join(event_parts))

    fragment_keywords = data.get("fragment_keywords", "") or ""
    if not fragment_keywords:
        fragment_keywords = _row_get(row, "fragment_keywords") or ""

    parts = [
        summary,
        knowledge_area,
        concept_type,
        " ".join(evidence_texts),
        stringify_list(data.get("signals")),
        implications_text,
        " ".join(event_texts),
        fragment_keywords,
    ]
    return " ".join(part for part in parts if part).strip()


def build_searchable_text_from_concept(concept: Any) -> str:
    """Assemble searchable text from a full Concept-like object.

    This preserves the legacy ``RetrievalEngine._concept_to_document`` field
    contract while giving embedding freshness one shared contract owner.
    """
    evidence_texts = []
    for evidence in getattr(concept, "evidence", []) or []:
        if isinstance(evidence, str):
            evidence_texts.append(evidence)
        elif isinstance(evidence, dict):
            evidence_texts.append(str(evidence.get("content", "")))
        elif hasattr(evidence, "content"):
            evidence_texts.append(str(evidence.content))

    metadata = getattr(concept, "metadata", {}) or {}
    if not isinstance(metadata, dict):
        metadata = {}

    parts = [
        getattr(concept, "summary", ""),
        " ".join(str(signal) for signal in (getattr(concept, "signals", []) or [])),
        " ".join(evidence_texts),
        metadata.get("knowledge_area", ""),
    ]

    for hypothesis in getattr(concept, "hypotheses", []) or []:
        description = getattr(hypothesis, "description", "")
        if description:
            parts.append(str(description))

    for implication in metadata.get("implications", []) or []:
        if isinstance(implication, str):
            parts.append(implication)

    for event in metadata.get("events", []) or []:
        if not isinstance(event, dict):
            continue
        event_parts = [str(event.get("action", ""))]
        if event.get("cause"):
            event_parts.append(f"because {event['cause']}")
        if event.get("consequence"):
            event_parts.append(f"resulting in {event['consequence']}")
        if event.get("actors"):
            actors = event["actors"]
            if isinstance(actors, list):
                event_parts.append(f"involving {', '.join(str(actor) for actor in actors)}")
            else:
                event_parts.append(f"involving {actors}")
        parts.append(" ".join(event_parts))

    return " ".join(part for part in parts if part).strip()
