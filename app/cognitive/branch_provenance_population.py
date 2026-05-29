"""Validated branch provenance population for ingest write paths.

This module is observe-only. It prepares metadata for storage, but does not
participate in retrieval ranking, answer construction, or branch selection.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from app.cognitive.branch_provenance_metadata import (
    BRANCH_AUTHORITY_EVENT_METADATA_KEY,
    BRANCH_PROVENANCE_METADATA_KEY,
    build_branch_provenance_metadata,
    validate_branch_provenance_metadata,
)

BranchProvenancePopulationStatus = Literal["ready", "invalid_input", "abstained"]
_BENCHMARK_PRIVATE_RE = re.compile(r"\b(?:no\d+|gold_answer|expected_answer|expected_source_refs|score_label)\b", re.I)
_SOURCE_TYPE_ALLOWLIST = {"session_learn", "benchmark_source", "conversation", "synthetic_fixture"}


@dataclass(frozen=True)
class BranchProvenancePopulationResult:
    metadata_patch: dict[str, Any]
    status: BranchProvenancePopulationStatus
    reason: str


def preserve_or_build_branch_provenance(
    insight: Mapping[str, Any],
    *,
    request_metadata: Mapping[str, Any] | None = None,
    source_index: Mapping[str, Any] | None = None,
) -> BranchProvenancePopulationResult:
    """Return a safe metadata patch for branch provenance.

    Phase 1 preserves only caller-supplied branch provenance envelopes that pass
    readiness validation. Phase 2 source-backed construction is intentionally
    scaffolded but disabled until parser false-positive coverage exists.
    """

    metadata = insight.get("metadata") if isinstance(insight.get("metadata"), Mapping) else {}
    envelope = metadata.get(BRANCH_PROVENANCE_METADATA_KEY)
    if envelope is not None:
        validation = validate_branch_provenance_metadata(envelope)
        if validation.ready:
            return BranchProvenancePopulationResult(
                metadata_patch={BRANCH_PROVENANCE_METADATA_KEY: dict(envelope)},
                status="ready",
                reason="validated_client_envelope",
            )
        return BranchProvenancePopulationResult(
            metadata_patch={},
            status="invalid_input",
            reason="invalid_client_envelope",
        )

    source_backed = _build_source_backed_branch_provenance(insight, source_index=source_index)
    if source_backed is not None:
        return source_backed

    _ = request_metadata
    return BranchProvenancePopulationResult(
        metadata_patch={},
        status="abstained",
        reason="no_client_envelope",
    )


def _build_source_backed_branch_provenance(
    insight: Mapping[str, Any],
    *,
    source_index: Mapping[str, Any] | None,
) -> BranchProvenancePopulationResult | None:
    if not source_index:
        return None
    source_event_id = _nonempty_str(source_index.get("source_event_id"))
    source_text = _nonempty_str(source_index.get("source_text"))
    source_sequence = source_index.get("source_sequence")
    if source_event_id is None and source_text is None and source_sequence is None:
        return None
    if source_event_id is None or source_text is None or not isinstance(source_sequence, int) or source_sequence < 0:
        return BranchProvenancePopulationResult(
            metadata_patch={},
            status="abstained",
            reason="missing_source_identity",
        )
    source_span = _nonempty_str(source_index.get("source_span"))
    source_type = _nonempty_str(source_index.get("source_type")) or "session_learn"
    if source_type not in _SOURCE_TYPE_ALLOWLIST:
        return BranchProvenancePopulationResult(
            metadata_patch={},
            status="abstained",
            reason="unsupported_source_type",
        )
    if any(_has_benchmark_private_marker(value) for value in (source_event_id, source_span, source_type)):
        return BranchProvenancePopulationResult(
            metadata_patch={},
            status="abstained",
            reason="benchmark_private_source_identity",
        )
    summary = _nonempty_str(insight.get("summary"))
    if summary is None:
        return BranchProvenancePopulationResult(
            metadata_patch={},
            status="abstained",
            reason="missing_assertion_summary",
        )
    try:
        envelope = build_branch_provenance_metadata(
            summary,
            source_event_id=source_event_id,
            source_sequence=source_sequence,
            source_text=source_text,
            source_span=source_span,
            source_type=source_type,
            authority_reason=_nonempty_str(source_index.get("authority_reason")),
            branch_authority_event=_authority_event(source_index.get(BRANCH_AUTHORITY_EVENT_METADATA_KEY)),
            conflict_resolution_state=_nonempty_str(source_index.get("conflict_resolution_state")) or "active",
            branch_resolution_state=_nonempty_str(source_index.get("branch_resolution_state")),
        ).to_metadata()
    except ValueError:
        return BranchProvenancePopulationResult(
            metadata_patch={},
            status="abstained",
            reason="unsupported_assertion",
        )

    validation = validate_branch_provenance_metadata(envelope)
    if not validation.ready:
        return BranchProvenancePopulationResult(
            metadata_patch={},
            status="invalid_input",
            reason="invalid_source_backed_envelope",
        )
    return BranchProvenancePopulationResult(
        metadata_patch={BRANCH_PROVENANCE_METADATA_KEY: envelope},
        status="ready",
        reason="source_backed_assertion",
    )


def _nonempty_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _authority_event(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    return None


def _has_benchmark_private_marker(value: str | None) -> bool:
    return bool(value and _BENCHMARK_PRIVATE_RE.search(value))
