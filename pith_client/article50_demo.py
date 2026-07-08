"""Article 50 evidence demo packet helpers.

This module builds a narrow, technical evidence packet for demo readiness. It
does not determine legal scope or compliance.
"""
from __future__ import annotations

import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PACKET_SCHEMA_VERSION = "article50_demo_packet.v1"
PACKET_GENERATOR = "pith_article50_demo"
PACKET_GENERATOR_VERSION = "0.1.0"
LEGAL_SCOPE_CAVEAT = (
    "Pith provides technical evidence for review. This packet does not determine "
    "legal scope or certify EU AI Act compliance."
)
OUTPUT_CLASSIFICATIONS = {
    "ai_generated",
    "ai_assisted",
    "human_reviewed",
    "not_classified",
}
REVIEW_ACTIONS = {"approved", "edited", "rejected", "escalated"}
MAX_DISCLOSURE_TEXT_CHARS = 4000


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _get_mapping_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _canonical_json_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _packet_without_hash(packet: dict[str, Any]) -> dict[str, Any]:
    copy_packet = copy.deepcopy(packet)
    integrity = copy_packet.get("packet_integrity")
    if isinstance(integrity, dict):
        integrity.pop("packet_hash", None)
    return copy_packet


def compute_packet_hash(packet: dict[str, Any]) -> str:
    """Compute the packet hash while excluding the hash field itself."""
    return hashlib.sha256(_canonical_json_bytes(_packet_without_hash(packet))).hexdigest()


def _require(mapping: Any, prefix: str, keys: list[str], missing: set[str]) -> None:
    if not isinstance(mapping, dict):
        missing.add(prefix)
        return
    for key in keys:
        value = mapping.get(key)
        if value is None or value == "":
            missing.add(f"{prefix}.{key}")


def validate_packet(packet: dict[str, Any], *, public_readiness: bool = False) -> dict[str, Any]:
    """Validate a demo packet and return pass/fail metadata."""
    missing: set[str] = set()
    errors: list[str] = []

    if packet.get("schema_version") != PACKET_SCHEMA_VERSION:
        errors.append("schema_version must be article50_demo_packet.v1")

    caveat = str(packet.get("legal_scope_caveat") or "")
    caveat_lower = caveat.lower()
    if "technical evidence" not in caveat_lower or "compliance" not in caveat_lower:
        missing.add("legal_scope_caveat")

    interaction = packet.get("interaction")
    disclosure_event = packet.get("disclosure_event")
    memory_influence = packet.get("memory_influence")
    output_provenance = packet.get("output_provenance")
    human_review = packet.get("human_review")

    _require(interaction, "interaction", ["interaction_id", "surface_id", "session_id", "started_at"], missing)
    _require(
        disclosure_event,
        "disclosure_event",
        ["shown_at", "disclosure_text", "disclosure_text_hash", "surface"],
        missing,
    )
    _require(
        output_provenance,
        "output_provenance",
        ["output_id", "classification", "classified_by", "classified_at", "classification_caveat"],
        missing,
    )
    _require(human_review, "human_review", ["reviewer", "reviewed_at", "action", "scope"], missing)

    if isinstance(disclosure_event, dict):
        if disclosure_event.get("shown") is not True:
            errors.append("disclosure_event.shown must be true")
        disclosure_text = disclosure_event.get("disclosure_text")
        if isinstance(disclosure_text, str) and len(disclosure_text) > MAX_DISCLOSURE_TEXT_CHARS:
            errors.append("disclosure_event.disclosure_text exceeds max length")
        expected_hash = _sha256_text(disclosure_text) if isinstance(disclosure_text, str) else None
        if expected_hash and disclosure_event.get("disclosure_text_hash") != expected_hash:
            errors.append("disclosure_event.disclosure_text_hash does not match disclosure_text")

    if isinstance(output_provenance, dict):
        classification = output_provenance.get("classification")
        if classification not in OUTPUT_CLASSIFICATIONS:
            errors.append("output_provenance.classification is invalid")

    if isinstance(human_review, dict):
        if human_review.get("reviewed") is not True:
            errors.append("human_review.reviewed must be true")
        if human_review.get("action") not in REVIEW_ACTIONS:
            errors.append("human_review.action is invalid")

    if not isinstance(memory_influence, dict):
        missing.add("memory_influence")
    elif public_readiness:
        if not memory_influence.get("trace_id"):
            missing.add("memory_influence.trace_id")
        retrieved_context = memory_influence.get("retrieved_context")
        if not isinstance(retrieved_context, list) or not retrieved_context:
            missing.add("memory_influence.retrieved_context")
        elif not any(isinstance(item, dict) and item.get("concept_id") for item in retrieved_context):
            missing.add("memory_influence.retrieved_context.concept_id")

    source_mode = "unknown"
    validation = packet.get("validation")
    if isinstance(validation, dict):
        source_mode = str(validation.get("source_mode") or source_mode)

    integrity = packet.get("packet_integrity")
    if not isinstance(integrity, dict):
        missing.add("packet_integrity")
    else:
        _require(integrity, "packet_integrity", ["generator", "generator_version"], missing)
        existing_hash = integrity.get("packet_hash")
        if existing_hash:
            expected_packet_hash = compute_packet_hash(packet)
            if existing_hash != expected_packet_hash:
                errors.append("packet_integrity.packet_hash does not match packet")

    return {
        "status": "fail" if missing or errors else "pass",
        "missing_fields": sorted(missing),
        "errors": errors,
        "source_mode": source_mode,
    }


def build_packet(
    *,
    interaction: dict[str, Any],
    disclosure_event: dict[str, Any],
    memory_influence: dict[str, Any],
    output_provenance: dict[str, Any],
    human_review: dict[str, Any],
    generator: str = PACKET_GENERATOR,
    generator_version: str = PACKET_GENERATOR_VERSION,
    source_repo_commit: str | None = None,
    source_mode: str = "fixture",
    generated_at: str | None = None,
    public_readiness: bool = False,
) -> dict[str, Any]:
    """Assemble, validate, and hash a demo evidence packet."""
    packet: dict[str, Any] = {
        "schema_version": PACKET_SCHEMA_VERSION,
        "generated_at": generated_at or _utc_now_iso(),
        "legal_scope_caveat": LEGAL_SCOPE_CAVEAT,
        "interaction": interaction,
        "disclosure_event": disclosure_event,
        "memory_influence": memory_influence,
        "output_provenance": output_provenance,
        "human_review": human_review,
        "validation": {
            "status": "fail",
            "missing_fields": [],
            "errors": [],
            "source_mode": source_mode,
        },
        "packet_integrity": {
            "packet_hash": "",
            "generator": generator,
            "generator_version": generator_version,
            "source_repo_commit": source_repo_commit,
        },
    }
    packet["validation"] = validate_packet(packet, public_readiness=public_readiness)
    packet["packet_integrity"]["packet_hash"] = compute_packet_hash(packet)
    return packet


def build_fixture_packet(*, generated_at: str | None = None, public_readiness: bool = True) -> dict[str, Any]:
    """Return a deterministic valid packet for tests and smoke runs."""
    disclosure_text = "You are interacting with an AI assistant."
    timestamp = generated_at or "2026-07-04T00:00:00Z"
    return build_packet(
        generated_at=timestamp,
        source_mode="fixture",
        public_readiness=public_readiness,
        interaction={
            "interaction_id": "demo-interaction-001",
            "surface_id": "fixture_chat",
            "session_id": "fixture-session-001",
            "started_at": timestamp,
            "user_region": "EU-demo",
        },
        disclosure_event={
            "shown": True,
            "shown_at": timestamp,
            "disclosure_text": disclosure_text,
            "disclosure_text_hash": _sha256_text(disclosure_text),
            "disclosure_template_id": "fixture-disclosure",
            "disclosure_template_version": "v1",
            "surface": "fixture_chat",
        },
        memory_influence={
            "trace_id": "fixture-trace-001",
            "output_id": "fixture-output-001",
            "concept_refs": ["fixture-concept-001"],
            "retrieved_context": [
                {
                    "concept_id": "fixture-concept-001",
                    "summary": "Fixture memory used for Article 50 evidence packet validation.",
                    "created_at": timestamp,
                    "session_id": "fixture-session-001",
                    "source_trace_id": "fixture-trace-001",
                    "edit_provenance": None,
                }
            ],
        },
        output_provenance={
            "output_id": "fixture-output-001",
            "classification": "ai_assisted",
            "classified_by": "demo_operator",
            "classified_at": timestamp,
            "classification_caveat": "Technical workflow state only; not a legal classification.",
        },
        human_review={
            "reviewed": True,
            "reviewer": "fixture-reviewer",
            "reviewed_at": timestamp,
            "action": "approved",
            "scope": "Fixture output reviewed for packet mechanics.",
        },
    )


def trace_response_to_memory_influence(
    response: object,
    *,
    trace_id: str,
    output_id: str,
) -> dict[str, Any]:
    """Convert a /pith_traces action=get response into memory influence fields."""
    trace = _get_mapping_value(response, "trace", {}) or {}
    linked_concepts = _get_mapping_value(response, "linked_concepts", []) or []
    concept_refs = _get_mapping_value(trace, "concept_refs", []) or []

    retrieved_context = []
    for concept in linked_concepts:
        concept_id = _get_mapping_value(concept, "id") or _get_mapping_value(concept, "concept_id")
        if not concept_id:
            continue
        retrieved_context.append(
            {
                "concept_id": concept_id,
                "summary": _get_mapping_value(concept, "summary"),
                "created_at": _get_mapping_value(concept, "created_at"),
                "session_id": _get_mapping_value(concept, "session_id"),
                "source_trace_id": _get_mapping_value(concept, "source_trace_id") or trace_id,
                "edit_provenance": _get_mapping_value(concept, "edit_provenance"),
            }
        )

    return {
        "trace_id": _get_mapping_value(trace, "id") or trace_id,
        "output_id": output_id,
        "concept_refs": concept_refs,
        "retrieved_context": retrieved_context,
    }


def packet_to_json(packet: dict[str, Any]) -> str:
    """Return canonical pretty JSON."""
    return json.dumps(packet, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def _md_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def packet_to_markdown(packet: dict[str, Any]) -> str:
    """Return a human-readable summary derived from packet fields."""
    interaction = packet.get("interaction", {})
    validation = packet.get("validation", {})
    disclosure = packet.get("disclosure_event", {})
    memory = packet.get("memory_influence", {})
    output = packet.get("output_provenance", {})
    review = packet.get("human_review", {})
    integrity = packet.get("packet_integrity", {})
    context_items = memory.get("retrieved_context") if isinstance(memory, dict) else []
    if not isinstance(context_items, list):
        context_items = []

    rows = [
        ("Interaction ID", _get_mapping_value(interaction, "interaction_id")),
        ("Session ID", _get_mapping_value(interaction, "session_id")),
        ("Source Mode", _get_mapping_value(validation, "source_mode")),
        ("Validation Status", _get_mapping_value(validation, "status")),
        ("Disclosure Surface", _get_mapping_value(disclosure, "surface")),
        ("Trace ID", _get_mapping_value(memory, "trace_id")),
        ("Output ID", _get_mapping_value(output, "output_id")),
        ("Output Classification", _get_mapping_value(output, "classification")),
        ("Reviewer", _get_mapping_value(review, "reviewer")),
        ("Review Action", _get_mapping_value(review, "action")),
        ("Review Scope", _get_mapping_value(review, "scope")),
        ("Packet Hash", _get_mapping_value(integrity, "packet_hash")),
    ]
    lines = [
        "# Article 50 Demo Evidence Packet",
        "",
        packet.get("legal_scope_caveat", LEGAL_SCOPE_CAVEAT),
        "",
        "| Field | Value |",
        "|---|---|",
    ]
    lines.extend(f"| {_md_escape(key)} | {_md_escape(value)} |" for key, value in rows)
    lines.extend(["", "## Retrieved Context", ""])
    if context_items:
        lines.extend(["| Concept ID | Summary |", "|---|---|"])
        for item in context_items:
            lines.append(f"| {_md_escape(_get_mapping_value(item, 'concept_id'))} | {_md_escape(_get_mapping_value(item, 'summary'))} |")
    else:
        lines.append("No retrieved context recorded.")
    return "\n".join(lines) + "\n"


def write_packet_exports(packet: dict[str, Any], output_dir: Path) -> dict[str, str]:
    """Write packet.json and packet.md to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "packet.json"
    markdown_path = output_dir / "packet.md"
    json_path.write_text(packet_to_json(packet), encoding="utf-8")
    markdown_path.write_text(packet_to_markdown(packet), encoding="utf-8")
    return {
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "packet_hash": str(packet.get("packet_integrity", {}).get("packet_hash", "")),
    }
