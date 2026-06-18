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

import json
from typing import Any, Mapping

__all__ = ["build_searchable_text", "parse_json_blob", "stringify_list"]


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
        fragment_keywords = (_row_get(row, "fragment_keywords") or "")

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
