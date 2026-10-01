"""Dry-run trust maintenance repair work-unit builder."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "trust_maintenance_repair_queue.v1"
DEFAULT_REPORT_ROOT = Path.home() / ".pith" / "reports" / "trust-governance" / "maintenance-repair-queue"

AUTHORITY_DECISIONS = {"approve_gold", "candidate_only", "exclude", "uncertain"}
REPAIR_DISPOSITIONS = {"none", "repair_needed", "tombstone_cleanup", "needs_context", "uncertain"}
REPAIR_REQUEST_DISPOSITIONS = {"repair_needed", "tombstone_cleanup"}
LEDGER_REVIEW_DECISIONS = {"approve_repair", "reject_repair", "needs_context", "not_supersession", "uncertain"}
LEGACY_HUMAN_QC_MAP = {
    "approve_gold": ("approve_gold", "none"),
    "candidate_only": ("candidate_only", "none"),
    "exclude": ("exclude", "none"),
    "repair_needed": ("exclude", "repair_needed"),
    "uncertain": ("uncertain", "uncertain"),
}

STRUCTURAL_BLOCKERS = {
    "broken_pointer",
    "cycle_detected",
    "depth_cap_reached",
    "self_loop",
    "terminal_missing",
}
REVIEW_REQUIRED_FLAGS = {
    "low_subject_overlap",
    "replacement_older",
    "replacement_thinner",
    "terminal_not_current",
}
OBSERVABILITY_FLAGS = {
    "missing_governance_event",
    "missing_supersedes_edge",
    "replacement_not_retrieved",
    "terminal_head_not_retrieved",
}
MECHANICAL_FLAGS = {
    "missing_reason",
    "missing_subject_key",
    "missing_superseded_at",
    "old_currency_not_superseded",
    "old_status_not_superseded",
}

LEDGER_MECHANICAL_CLASSIFICATION = "mechanical_metadata_candidate"
LEDGER_BAD_EDGE_SUPERSESSION_CLASSIFICATION = "bad_edge_supersession_candidate"
LEDGER_REVIEWED_EDGE_SUPERSESSION_CLASSIFICATION = "reviewed_edge_supersession_candidate"
LEDGER_BLOCKED_STRUCTURAL = "blocked_structural"
LEDGER_BLOCKED_VERIFIER = "blocked_verifier"
LEDGER_REVIEW_SEMANTIC = "review_required_semantic"
LEDGER_REVIEW_PROVENANCE = "review_required_provenance"
LEDGER_NO_ACTION = "no_action"
LEDGER_NO_ACTION_HISTORICAL_ORPHAN = "no_action_historical_orphan"
BAD_EDGE_STRUCTURAL_CLASSES = {"cycle_pair", "unsafe_orphan_pointer", "unsafe_orphan_path"}
NORMALIZE_SUPERSESSION_CHAIN_FIELDS = (
    "status",
    "currency_status",
    "is_current",
    "subject_key",
    "superseded_by",
    "supersession_reason",
    "superseded_at",
)


def load_decision_rows(path: Path) -> list[dict[str, str]]:
    """Load raw TSV decision rows with stripped string values."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [{key: (value or "").strip() for key, value in row.items()} for row in reader]


def build_repair_work_units(
    audit_report: dict[str, Any],
    decision_summary: dict[str, Any],
    decision_rows: list[dict[str, str]],
    *,
    audit_report_path: Path,
    decisions_tsv_path: Path,
) -> dict[str, Any]:
    """Build a deterministic dry-run manifest from audit rows and reviewed decisions."""
    audit_results = audit_report.get("results")
    if not isinstance(audit_results, list):
        raise ValueError("audit_report must contain a results list")

    audit_by_old_id = {
        str(row.get("old_id")): row for row in audit_results if isinstance(row, dict) and row.get("old_id")
    }
    warnings = _input_warnings(audit_report, decision_summary, decision_rows, audit_by_old_id)
    units = [_build_unit(row, audit_by_old_id.get(row.get("old_id", ""))) for row in decision_rows]
    units.sort(key=lambda unit: (unit["chain_key"].get("old_id") or "", unit["row_number"]))
    counts = _summarize(units)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "apply_enabled": False,
        "boundary": "Dry-run maintenance readiness only; repairability is not authority approval.",
        "input_fingerprint": {
            "audit_report_path": str(audit_report_path),
            "decisions_tsv_path": str(decisions_tsv_path),
            "audit_report_sha256": _sha256(audit_report_path),
            "decisions_tsv_sha256": _sha256(decisions_tsv_path),
            "audit_generated_at": (audit_report.get("summary") or {}).get("generated_at"),
            "decision_generated_at_values": decision_summary.get("generated_at_values", []),
            "audit_result_count": len(audit_results),
            "decision_row_count": len(decision_rows),
            "rows_reviewed": decision_summary.get("rows_reviewed", len(decision_rows)),
            "warnings": warnings,
        },
        "summary": counts,
        "work_units": units,
    }


def build_repair_work_units_from_ledger(
    ledger_bundle: Any,
    decision_rows: list[dict[str, str]],
    *,
    ledger_path: Path,
    manifest_path: Path,
    decisions_path: Path,
) -> dict[str, Any]:
    """Build dry-run maintenance work units from verified ledger decisions."""
    if isinstance(ledger_bundle, dict):
        lines = list(ledger_bundle.get("lines", []))
        manifest = dict(ledger_bundle.get("manifest", {}))
    else:
        lines = list(getattr(ledger_bundle, "lines", []))
        manifest = dict(getattr(ledger_bundle, "manifest", {}))
    verified_sha = str(manifest.get("verified_ledger_sha256") or "")
    source_snapshot = manifest.get("source_snapshot") if isinstance(manifest.get("source_snapshot"), dict) else {}
    by_line_number = {str(line.get("line_number")): line for line in lines}
    units = [
        _build_ledger_unit(row, by_line_number.get(str(row.get("line_number") or "")), verified_sha, source_snapshot)
        for row in decision_rows
    ]
    units.sort(key=lambda unit: (unit["chain_key"].get("old_id") or "", unit["row_number"]))
    summary = _summarize_ledger_units(units)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "apply_enabled": False,
        "boundary": "Dry-run maintenance readiness only; repairability is not authority approval.",
        "input_fingerprint": {
            "ledger_path": str(ledger_path),
            "manifest_path": str(manifest_path),
            "decisions_path": str(decisions_path),
            "ledger_sha256": manifest.get("ledger_sha256"),
            "verified_ledger_sha256": verified_sha,
            "ledger_file_sha256": _sha256(ledger_path),
            "manifest_file_sha256": _sha256(manifest_path),
            "decisions_tsv_sha256": _sha256(decisions_path),
            "source_snapshot": source_snapshot,
            "ledger_row_count": len(lines),
            "decision_row_count": len(decision_rows),
        },
        "summary": summary,
        "work_units": units,
    }


def write_repair_queue_artifacts(
    manifest: dict[str, Any],
    output_dir: Path | None = None,
) -> dict[str, str]:
    """Write JSON and Markdown dry-run artifacts to a collision-safe directory."""
    target = _output_dir(output_dir)
    manifest_path = target / "trust_maintenance_repair_queue_manifest.json"
    summary_path = target / "trust_maintenance_repair_queue_summary.md"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    summary_path.write_text(render_repair_queue_markdown(manifest), encoding="utf-8")
    return {"output_dir": str(target), "manifest_json": str(manifest_path), "summary_md": str(summary_path)}


def render_repair_queue_markdown(manifest: dict[str, Any]) -> str:
    """Render a compact human-readable summary for operator review."""
    summary = manifest.get("summary") if isinstance(manifest.get("summary"), dict) else {}
    fingerprint = manifest.get("input_fingerprint") if isinstance(manifest.get("input_fingerprint"), dict) else {}
    lines = [
        "# Trust Maintenance Repair Queue Dry Run",
        "",
        f"- Status: {summary.get('status', 'UNKNOWN')}",
        f"- Total work units: {summary.get('total_work_units', 0)}",
        f"- Safe to repair later: {summary.get('mechanical_safe_count', 0)}",
        f"- Needs review: {summary.get('review_required_count', 0)}",
        f"- Blocked: {summary.get('blocked_count', 0)}",
        f"- Apply enabled: {manifest.get('apply_enabled', False)}",
        f"- Boundary: {manifest.get('boundary', '')}",
        f"- Audit report: {fingerprint.get('audit_report_path', 'unknown')}",
        f"- Decisions TSV: {fingerprint.get('decisions_tsv_path', 'unknown')}",
        "",
        "## Operation Counts",
    ]
    for key, value in sorted((summary.get("operation_counts") or {}).items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Proposed Operation Counts"])
    for key, value in sorted((summary.get("proposed_operation_counts") or {}).items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Blocker Counts"])
    for key, value in sorted((summary.get("blocker_counts") or {}).items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Deferred Operation Blocker Counts"])
    for key, value in sorted((summary.get("operation_blocker_counts") or {}).items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Review Note Counts"])
    for key, value in sorted((summary.get("review_note_counts") or {}).items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Observability Note Counts"])
    for key, value in sorted((summary.get("observability_note_counts") or {}).items()):
        lines.append(f"- {key}: {value}")
    warnings = fingerprint.get("warnings") or []
    if warnings:
        lines.extend(["", "## Input Warnings"])
        for warning in warnings:
            lines.append(f"- {warning}")
    return "\n".join(lines) + "\n"


def _build_unit(decision_row: dict[str, str], audit_row: dict[str, Any] | None) -> dict[str, Any]:
    normalized, label_errors = _normalize_labels(decision_row)
    issue_flags = sorted(set((audit_row or {}).get("issue_flags") or _split_flags(decision_row.get("issue_flags", ""))))
    safety_blockers = list(label_errors)
    review_notes: list[str] = []
    observability_notes = [f"observability:{flag}" for flag in issue_flags if flag in OBSERVABILITY_FLAGS]
    operations: list[dict[str, Any]] = []
    operation_blockers: list[str] = []

    if audit_row is None:
        safety_blockers.append("audit_row_not_found")
    if not _has_review(decision_row):
        safety_blockers.append("unreviewed")
    if normalized["repair_disposition"] not in REPAIR_REQUEST_DISPOSITIONS:
        safety_blockers.append("repair_not_requested")
    safety_blockers.extend(flag for flag in issue_flags if flag in STRUCTURAL_BLOCKERS)

    review_required = sorted(flag for flag in issue_flags if flag in REVIEW_REQUIRED_FLAGS)
    if review_required:
        review_notes.extend(f"review_required:{flag}" for flag in review_required)
    if not safety_blockers:
        operations, proposed_operation_blockers = _proposed_operations(audit_row or {}, issue_flags)
        for blocker in proposed_operation_blockers:
            if blocker.startswith("unsupported_issue_flag:"):
                safety_blockers.append(blocker)
            else:
                operation_blockers.append(blocker)

    if safety_blockers:
        classification = "blocked"
    elif operations:
        classification = "mechanical_safe"
    elif operation_blockers:
        classification = "blocked"
    elif review_notes:
        classification = "review_required"
    else:
        classification = "blocked"
        safety_blockers.append("no_supported_operation")

    chain_key = {
        "old_id": decision_row.get("old_id") or (audit_row or {}).get("old_id"),
        "replacement_id": decision_row.get("replacement_id") or (audit_row or {}).get("replacement_id"),
        "terminal_head_id": decision_row.get("terminal_head_id") or (audit_row or {}).get("terminal_head_id"),
    }
    unit = {
        "work_unit_id": _work_unit_id(chain_key, issue_flags, normalized["repair_disposition"]),
        "row_number": decision_row.get("row_number") or "",
        "chain_key": chain_key,
        "authority_decision": normalized["authority_decision"],
        "repair_disposition": normalized["repair_disposition"],
        "classification": classification,
        "issue_flags": issue_flags,
        "proposed_operations": operations,
        "safety_blockers": sorted(set(safety_blockers)),
        "operation_blockers": sorted(set(operation_blockers)),
        "review_notes": sorted(set(review_notes)),
        "observability_notes": sorted(set(observability_notes)),
        "source_evidence": {
            "generated_at": decision_row.get("generated_at") or "",
            "reviewer_notes": decision_row.get("reviewer_notes") or "",
        },
        "apply_enabled": False,
        "apply_contract": {
            "mode": "dry_run_only",
            "future_apply_requires": [
                "pre_mutation_snapshot",
                "input_hash_match",
                "reviewed_mechanical_safe_classification",
                "transaction",
                "rollback_plan",
                "post_apply_audit",
            ],
        },
    }
    return unit


def _build_ledger_unit(
    decision_row: dict[str, str],
    ledger_line: dict[str, Any] | None,
    verified_sha: str,
    source_snapshot: dict[str, Any],
) -> dict[str, Any]:
    line = ledger_line or {}
    issue_flags = sorted(set(line.get("issue_flags") or _split_flags(decision_row.get("issue_flags", ""))))
    structural_class = str(line.get("structural_class") or "none")
    structural_severity = str(line.get("structural_severity") or "none")
    decision = decision_row.get("review_decision", "")
    blockers: list[str] = []
    review_notes: list[str] = []
    operations: list[dict[str, Any]] = []
    if ledger_line is None:
        blockers.append("ledger_line_not_found")
    if decision not in LEDGER_REVIEW_DECISIONS:
        blockers.append(f"invalid_review_decision:{decision or 'blank'}")
    if line.get("verifier_status") != "passed":
        classification = LEDGER_BLOCKED_VERIFIER
        blockers.extend(str(item) for item in line.get("blocking_reasons") or [])
    elif structural_class == "orphan_tombstone":
        classification = LEDGER_NO_ACTION_HISTORICAL_ORPHAN
        review_notes.append("historical_orphan:safe_tombstone")
    elif (
        decision == "approve_repair"
        and structural_class in BAD_EDGE_STRUCTURAL_CLASSES
        and set(issue_flags) & STRUCTURAL_BLOCKERS
    ):
        classification = LEDGER_BAD_EDGE_SUPERSESSION_CLASSIFICATION
        operations = [_clear_supersession_edge_operation(_ledger_audit_like(line))]
    elif set(issue_flags) & STRUCTURAL_BLOCKERS:
        classification = LEDGER_BLOCKED_STRUCTURAL
        blockers.extend(flag for flag in issue_flags if flag in STRUCTURAL_BLOCKERS)
    elif decision in {"reject_repair", "not_supersession"}:
        classification = LEDGER_NO_ACTION
    elif decision in {"needs_context", "uncertain"}:
        classification = LEDGER_REVIEW_SEMANTIC
        review_notes.append(f"review_decision:{decision}")
    elif decision == "approve_repair" and "low_subject_overlap" in issue_flags:
        operations, operation_blockers = _reviewed_edge_supersession_operations(
            line,
            decision_row,
            issue_flags,
            verified_sha,
        )
        if operation_blockers:
            classification = LEDGER_REVIEW_SEMANTIC
            review_notes.extend(f"review_required:{blocker}" for blocker in operation_blockers)
            blockers.extend(blocker for blocker in operation_blockers if blocker.startswith("missing_"))
        else:
            classification = LEDGER_REVIEWED_EDGE_SUPERSESSION_CLASSIFICATION
    elif set(issue_flags) & REVIEW_REQUIRED_FLAGS:
        classification = LEDGER_REVIEW_SEMANTIC
        review_notes.extend(f"review_required:{flag}" for flag in issue_flags if flag in REVIEW_REQUIRED_FLAGS)
    elif set(issue_flags) & OBSERVABILITY_FLAGS:
        classification = LEDGER_REVIEW_PROVENANCE
        review_notes.extend(f"provenance_required:{flag}" for flag in issue_flags if flag in OBSERVABILITY_FLAGS)
    elif decision == "approve_repair" and set(issue_flags) <= MECHANICAL_FLAGS:
        classification = LEDGER_MECHANICAL_CLASSIFICATION
        operations, operation_blockers = _proposed_operations(_ledger_audit_like(line), issue_flags)
        blockers.extend(operation_blockers)
    else:
        classification = LEDGER_REVIEW_PROVENANCE
        unsupported = sorted(
            set(issue_flags) - MECHANICAL_FLAGS - STRUCTURAL_BLOCKERS - REVIEW_REQUIRED_FLAGS - OBSERVABILITY_FLAGS
        )
        review_notes.extend(f"unsupported_or_unclassified:{flag}" for flag in unsupported)

    if blockers and classification == LEDGER_MECHANICAL_CLASSIFICATION:
        classification = LEDGER_REVIEW_PROVENANCE

    chain_key = {
        "old_id": decision_row.get("old_id") or line.get("old_id"),
        "replacement_id": line.get("replacement_id"),
        "terminal_head_id": line.get("terminal_head_id"),
    }
    work_unit_id = _ledger_work_unit_id(chain_key, issue_flags, decision, verified_sha)
    for operation in operations:
        if isinstance(operation, dict) and isinstance(operation.get("provenance"), dict):
            operation["provenance"]["work_unit_id"] = work_unit_id

    return {
        "work_unit_id": work_unit_id,
        "row_number": decision_row.get("line_number") or line.get("line_number") or "",
        "chain_key": chain_key,
        "review_decision": decision or "blank",
        "classification": classification,
        "issue_flags": issue_flags,
        "structural_class": structural_class,
        "structural_severity": structural_severity,
        "proposed_operations": (
            operations
            if classification
            in {
                LEDGER_MECHANICAL_CLASSIFICATION,
                LEDGER_BAD_EDGE_SUPERSESSION_CLASSIFICATION,
                LEDGER_REVIEWED_EDGE_SUPERSESSION_CLASSIFICATION,
            }
            else []
        ),
        "safety_blockers": sorted(set(blockers)),
        "operation_blockers": [],
        "review_notes": sorted(set(review_notes)),
        "observability_notes": [],
        "source_evidence": {
            "reviewer": decision_row.get("reviewer", ""),
            "reviewed_at": decision_row.get("reviewed_at", ""),
            "reviewer_notes": decision_row.get("reviewer_notes", ""),
            "verified_line_hash": line.get("verified_line_hash"),
            "verified_ledger_sha256": verified_sha,
            "structural_class": structural_class,
            "structural_severity": structural_severity,
            "source_snapshot": source_snapshot,
        },
        "apply_enabled": False,
        "apply_contract": {
            "mode": "dry_run_only",
            "future_apply_requires": [
                "verified_ledger_sha256_match",
                "verified_line_hash_match",
                "reviewed_applyable_repair_candidate",
                "pre_mutation_snapshot",
                "transaction",
                "rollback_plan",
                "post_apply_audit",
            ],
        },
    }


def _reviewed_edge_supersession_operations(
    line: dict[str, Any],
    decision_row: dict[str, str],
    issue_flags: list[str],
    verified_sha: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    blockers: list[str] = []
    review_flags = set(issue_flags) & REVIEW_REQUIRED_FLAGS
    extra_review_flags = sorted(review_flags - {"low_subject_overlap"})
    blockers.extend(extra_review_flags)
    if set(issue_flags) & STRUCTURAL_BLOCKERS:
        blockers.extend(sorted(set(issue_flags) & STRUCTURAL_BLOCKERS))
    reviewer = str(decision_row.get("reviewer") or "").strip()
    reviewed_at = str(decision_row.get("reviewed_at") or "").strip()
    reviewer_notes = str(decision_row.get("reviewer_notes") or "").strip()
    if not reviewer:
        blockers.append("missing_reviewer")
    if not reviewed_at:
        blockers.append("missing_reviewed_at")
    if not reviewer_notes:
        blockers.append("missing_reviewer_notes")

    before = line.get("before") if isinstance(line.get("before"), dict) else {}
    old_id = str(line.get("old_id") or decision_row.get("old_id") or "").strip()
    replacement_id = str(line.get("replacement_id") or "").strip()
    terminal_head_id = str(line.get("terminal_head_id") or "").strip()
    replacement_subject_key = str(before.get("replacement_subject_key") or "").strip()
    if not old_id:
        blockers.append("missing_old_id")
    if not replacement_id:
        blockers.append("missing_replacement_id")
    if terminal_head_id and replacement_id and terminal_head_id != replacement_id:
        blockers.append("terminal_not_current")
    if not replacement_subject_key:
        blockers.append("missing_subject_key_source")

    if blockers:
        return [], sorted(set(blockers))

    operation_before = {field: before.get(f"old_{field}") for field in NORMALIZE_SUPERSESSION_CHAIN_FIELDS}
    operation_before["is_current"] = 1 if "old_still_current" in set(issue_flags) else 0
    operation_before["subject_key"] = before.get("old_subject_key")
    operation_before["superseded_by"] = line.get("replacement_id")
    operation_before["supersession_reason"] = before.get("supersession_reason")
    operation_before["superseded_at"] = before.get("superseded_at")
    operation_after = {
        "status": "superseded",
        "currency_status": "SUPERSEDED",
        "is_current": 0,
        "subject_key": replacement_subject_key,
        "superseded_by": replacement_id,
        "supersession_reason": str(before.get("supersession_reason") or reviewer_notes),
        "superseded_at": str(before.get("superseded_at") or reviewed_at),
    }
    operation = {
        "operation": "normalize_supersession_chain",
        "table": "concepts",
        "concept_id": old_id,
        "field": "superseded_by",
        "before": operation_before,
        "after": operation_after,
        "provenance": {
            "repair_id": "SUPER-022",
            "verified_ledger_sha256": verified_sha,
            "reviewer": reviewer,
            "reviewed_at": reviewed_at,
        },
        "dry_run": True,
    }
    return [operation], []


def _normalize_labels(row: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    authority = row.get("authority_decision", "")
    repair = row.get("repair_disposition", "")
    human_qc = row.get("human_qc", "")
    if (not authority and not repair) and human_qc in LEGACY_HUMAN_QC_MAP:
        authority, repair = LEGACY_HUMAN_QC_MAP[human_qc]
    errors: list[str] = []
    if authority and authority not in AUTHORITY_DECISIONS:
        errors.append(f"invalid_authority_decision:{authority}")
    if repair and repair not in REPAIR_DISPOSITIONS:
        errors.append(f"invalid_repair_disposition:{repair}")
    return {
        "authority_decision": authority or "blank",
        "repair_disposition": repair or "blank",
    }, errors


def _has_review(row: dict[str, str]) -> bool:
    return bool(row.get("authority_decision") or row.get("repair_disposition") or row.get("human_qc"))


def _proposed_operations(audit_row: dict[str, Any], issue_flags: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    operations: list[dict[str, Any]] = []
    blockers: list[str] = []
    old_id = audit_row.get("old_id")
    if "missing_reason" in issue_flags:
        operations.append(
            _operation(
                "concepts",
                old_id,
                "supersession_reason",
                audit_row.get("supersession_reason"),
                f"Reviewed explicit supersession to {audit_row.get('replacement_id')}",
            )
        )
    if "missing_superseded_at" in issue_flags:
        timestamp = audit_row.get("replacement_created_at") or audit_row.get("terminal_created_at")
        if timestamp:
            operations.append(
                _operation("concepts", old_id, "superseded_at", audit_row.get("superseded_at"), timestamp)
            )
        else:
            blockers.append("missing_superseded_at_source")
    if "old_status_not_superseded" in issue_flags:
        operations.append(_operation("concepts", old_id, "status", audit_row.get("old_status"), "superseded"))
    if "old_currency_not_superseded" in issue_flags:
        operations.append(
            _operation("concepts", old_id, "currency_status", audit_row.get("old_currency_status"), "SUPERSEDED")
        )
    if "missing_subject_key" in issue_flags:
        subject_key = audit_row.get("replacement_subject_key")
        if subject_key:
            operations.append(
                _operation("concepts", old_id, "subject_key", audit_row.get("old_subject_key"), subject_key)
            )
        else:
            blockers.append("missing_subject_key_source")
    unsupported = sorted(
        set(issue_flags) - MECHANICAL_FLAGS - STRUCTURAL_BLOCKERS - REVIEW_REQUIRED_FLAGS - OBSERVABILITY_FLAGS
    )
    if unsupported:
        blockers.extend(f"unsupported_issue_flag:{flag}" for flag in unsupported)
    return operations, blockers


def _operation(table: str, concept_id: str | None, field: str, before: Any, after: Any) -> dict[str, Any]:
    return {
        "operation": "set_field",
        "table": table,
        "concept_id": concept_id,
        "field": field,
        "before": before,
        "after": after,
        "dry_run": True,
    }


def _clear_supersession_edge_operation(audit_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "operation": "clear_supersession_edge",
        "table": "concepts",
        "concept_id": audit_row.get("old_id"),
        "field": "superseded_by",
        "before": {
            "superseded_by": audit_row.get("replacement_id"),
            "supersession_reason": audit_row.get("supersession_reason"),
            "superseded_at": audit_row.get("superseded_at"),
        },
        "after": {
            "superseded_by": None,
            "supersession_reason": None,
            "superseded_at": None,
        },
        "preserve_fields": [
            "status",
            "currency_status",
            "is_current",
            "authority_score",
            "effective_authority",
            "version_chain_head",
            "subject_key",
        ],
        "dry_run": True,
    }


def _summarize(units: list[dict[str, Any]]) -> dict[str, Any]:
    classification_counts: dict[str, int] = {}
    operation_counts: dict[str, int] = {}
    proposed_operation_counts: dict[str, int] = {}
    blocker_counts: dict[str, int] = {}
    operation_blocker_counts: dict[str, int] = {}
    review_note_counts: dict[str, int] = {}
    observability_note_counts: dict[str, int] = {}
    issue_counts: dict[str, int] = {}
    for unit in units:
        classification_counts[unit["classification"]] = classification_counts.get(unit["classification"], 0) + 1
        for op in unit["proposed_operations"]:
            key = f"{op.get('table')}.{op.get('field')}"
            proposed_operation_counts[key] = proposed_operation_counts.get(key, 0) + 1
            if unit["classification"] == "mechanical_safe":
                operation_counts[key] = operation_counts.get(key, 0) + 1
        for blocker in unit["safety_blockers"]:
            blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1
        for blocker in unit.get("operation_blockers", []):
            operation_blocker_counts[blocker] = operation_blocker_counts.get(blocker, 0) + 1
        for note in unit.get("review_notes", []):
            review_note_counts[note] = review_note_counts.get(note, 0) + 1
        for note in unit.get("observability_notes", []):
            observability_note_counts[note] = observability_note_counts.get(note, 0) + 1
        for flag in unit["issue_flags"]:
            issue_counts[flag] = issue_counts.get(flag, 0) + 1
    return {
        "status": "SUCCESS",
        "total_work_units": len(units),
        "mechanical_safe_count": classification_counts.get("mechanical_safe", 0),
        "review_required_count": classification_counts.get("review_required", 0),
        "blocked_count": classification_counts.get("blocked", 0),
        "classification_counts": dict(sorted(classification_counts.items())),
        "operation_counts": dict(sorted(operation_counts.items())),
        "proposed_operation_counts": dict(sorted(proposed_operation_counts.items())),
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "operation_blocker_counts": dict(sorted(operation_blocker_counts.items())),
        "review_note_counts": dict(sorted(review_note_counts.items())),
        "observability_note_counts": dict(sorted(observability_note_counts.items())),
        "issue_counts": dict(sorted(issue_counts.items())),
    }


def _summarize_ledger_units(units: list[dict[str, Any]]) -> dict[str, Any]:
    classification_counts: dict[str, int] = {}
    operation_counts: dict[str, int] = {}
    blocker_counts: dict[str, int] = {}
    review_note_counts: dict[str, int] = {}
    issue_counts: dict[str, int] = {}
    for unit in units:
        classification = unit["classification"]
        classification_counts[classification] = classification_counts.get(classification, 0) + 1
        for op in unit["proposed_operations"]:
            key = f"{op.get('table')}.{op.get('field')}"
            operation_counts[key] = operation_counts.get(key, 0) + 1
        for blocker in unit["safety_blockers"]:
            blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1
        for note in unit.get("review_notes", []):
            review_note_counts[note] = review_note_counts.get(note, 0) + 1
        for flag in unit["issue_flags"]:
            issue_counts[flag] = issue_counts.get(flag, 0) + 1

    blocked_count = sum(
        count for classification, count in classification_counts.items() if classification.startswith("blocked_")
    )
    review_required_count = sum(
        count
        for classification, count in classification_counts.items()
        if classification.startswith("review_required_")
    )
    return {
        "status": "SUCCESS",
        "total_work_units": len(units),
        "mechanical_safe_count": classification_counts.get(LEDGER_MECHANICAL_CLASSIFICATION, 0),
        "bad_edge_repair_count": classification_counts.get(LEDGER_BAD_EDGE_SUPERSESSION_CLASSIFICATION, 0),
        "reviewed_edge_repair_count": classification_counts.get(LEDGER_REVIEWED_EDGE_SUPERSESSION_CLASSIFICATION, 0),
        "review_required_count": review_required_count,
        "blocked_count": blocked_count,
        "classification_counts": dict(sorted(classification_counts.items())),
        "operation_counts": dict(sorted(operation_counts.items())),
        "proposed_operation_counts": dict(sorted(operation_counts.items())),
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "operation_blocker_counts": {},
        "review_note_counts": dict(sorted(review_note_counts.items())),
        "observability_note_counts": {},
        "issue_counts": dict(sorted(issue_counts.items())),
    }


def _input_warnings(
    audit_report: dict[str, Any],
    decision_summary: dict[str, Any],
    decision_rows: list[dict[str, str]],
    audit_by_old_id: dict[str, dict[str, Any]],
) -> list[str]:
    warnings: list[str] = []
    audit_generated_at = (audit_report.get("summary") or {}).get("generated_at")
    decision_generated = set(decision_summary.get("generated_at_values") or [])
    if audit_generated_at and decision_generated and audit_generated_at not in decision_generated:
        warnings.append("generated_at_mismatch")
    if int(decision_summary.get("rows_reviewed") or len(decision_rows)) != len(decision_rows):
        warnings.append("decision_row_count_mismatch")
    missing = [
        row.get("old_id") for row in decision_rows if row.get("old_id") and row.get("old_id") not in audit_by_old_id
    ]
    if missing:
        warnings.append(f"audit_row_missing_for_decisions:{len(missing)}")
    return warnings


def _work_unit_id(chain_key: dict[str, Any], issue_flags: list[str], repair_disposition: str) -> str:
    payload = {
        "old_id": chain_key.get("old_id") or "",
        "replacement_id": chain_key.get("replacement_id") or "",
        "terminal_head_id": chain_key.get("terminal_head_id") or "",
        "issue_flags": sorted(issue_flags),
        "repair_disposition": repair_disposition,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _ledger_work_unit_id(
    chain_key: dict[str, Any],
    issue_flags: list[str],
    review_decision: str,
    verified_sha: str,
) -> str:
    payload = {
        "old_id": chain_key.get("old_id") or "",
        "replacement_id": chain_key.get("replacement_id") or "",
        "terminal_head_id": chain_key.get("terminal_head_id") or "",
        "issue_flags": sorted(issue_flags),
        "review_decision": review_decision,
        "verified_ledger_sha256": verified_sha,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _ledger_audit_like(line: dict[str, Any]) -> dict[str, Any]:
    before = line.get("before") if isinstance(line.get("before"), dict) else {}
    return {
        "old_id": line.get("old_id"),
        "replacement_id": line.get("replacement_id"),
        "terminal_head_id": line.get("terminal_head_id"),
        "issue_flags": line.get("issue_flags") or [],
        **before,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _split_flags(value: str) -> list[str]:
    return sorted(item.strip() for item in value.split(",") if item.strip())


def _output_dir(base: Path | None) -> Path:
    root = base or DEFAULT_REPORT_ROOT / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    candidate = root.expanduser()
    suffix = 1
    while candidate.exists() and any(candidate.iterdir()):
        candidate = root.with_name(f"{root.name}-{suffix}")
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate
