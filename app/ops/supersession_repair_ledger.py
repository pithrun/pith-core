"""Deterministic supersession repair ledger and copy-DB proof utilities."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.ops.supersession_chain_quality import (
    LIFECYCLE_FLAGS,
    PARITY_FLAGS,
    PROVENANCE_FLAGS,
    SEMANTIC_FLAGS,
    STRUCTURAL_FLAGS,
    audit_currency_parity,
    audit_supersession_chains,
)
from app.ops.trust_maintenance_repair_queue import (
    LEDGER_BAD_EDGE_SUPERSESSION_CLASSIFICATION,
    LEDGER_BLOCKED_STRUCTURAL,
    LEDGER_BLOCKED_VERIFIER,
    LEDGER_MECHANICAL_CLASSIFICATION,
    LEDGER_REVIEW_PROVENANCE,
    LEDGER_REVIEW_SEMANTIC,
    LEDGER_REVIEWED_EDGE_SUPERSESSION_CLASSIFICATION,
    build_repair_work_units_from_ledger,
    load_decision_rows,
)
from app.storage import DB_PATH, apply_lifecycle_transition_conn

LINE_SCHEMA_VERSION = "supersession_repair_ledger.line.v1"
MANIFEST_SCHEMA_VERSION = "supersession_repair_ledger.manifest.v1"
SUMMARY_SCHEMA_VERSION = "supersession_repair_ledger.summary.v1"
APPLY_REPORT_SCHEMA_VERSION = "supersession_repair_ledger.apply_report.v1"
CURATED_REPAIR_MANIFEST_SCHEMA_VERSION = "supersession_repair_ledger.curated_repair_manifest.v1"
APPROVED_MECHANICAL_APPLY_REPORT_SCHEMA_VERSION = "supersession_repair_ledger.approved_mechanical_apply_report.v1"
LIVE_APPROVED_MECHANICAL_APPLY_REPORT_SCHEMA_VERSION = (
    "supersession_repair_ledger.live_approved_mechanical_apply_report.v1"
)
SCOPED_APPLY_VERIFICATION_SCHEMA_VERSION = "supersession_repair_ledger.scoped_apply_verification.v1"
SCOPED_APPLY_CLAIM_BOUNDARY = (
    "Scoped apply readiness validates reviewed selected rows only; ambient full-ledger drift remains diagnostic."
)

LEDGER_FILENAME = "ledger.jsonl"
MANIFEST_FILENAME = "manifest.json"
SUMMARY_FILENAME = "summary.json"
REVIEW_PACKET_FILENAME = "review_packet.md"
DECISION_TEMPLATE_FILENAME = "ledger_review_decisions.tsv"
APPLY_REPORT_FILENAME = "apply_report.json"
APPROVED_MECHANICAL_APPLY_REPORT_FILENAME = "approved_mechanical_apply_report.json"
LIVE_APPROVED_MECHANICAL_PREFLIGHT_REPORT_FILENAME = "live_apply_preflight_report.json"
LIVE_APPROVED_MECHANICAL_APPLY_REPORT_FILENAME = "live_apply_report.json"
LIVE_APPROVED_MECHANICAL_ROW_EVIDENCE_FILENAME = "live_apply_row_evidence.jsonl"
LIVE_APPROVED_MECHANICAL_ROLLBACK_RUNBOOK_FILENAME = "rollback_runbook.md"
REPAIRABILITY_CONTRACT_SCHEMA_VERSION = "supersession_repairability_contract.v1"
REPAIRABILITY_CONTRACT_ROW_SCHEMA_VERSION = "supersession_repairability_contract.row.v1"
REPAIRABILITY_CONTRACT_SUMMARY_SCHEMA_VERSION = "supersession_repairability_contract.summary.v1"
REPAIRABILITY_CONTRACT_FILENAME = "repairability_contract.json"
REPAIRABILITY_CONTRACT_SUMMARY_FILENAME = "repairability_contract_summary.json"
REPAIRABILITY_OPERATOR_PACKET_FILENAME = "repairability_operator_packet.md"
REPAIRABILITY_EVIDENCE_REPORT_FILENAME = "repairability_evidence_report.md"
MECHANICAL_CANDIDATE_DECISION_TEMPLATE_FILENAME = "mechanical_candidate_review_decisions.tsv"
CURRENCY_PARITY_REPORT_FILENAME = "currency_parity_report.json"

HASH_RE = re.compile(r"^[a-f0-9]{64}$")
GENERATED_STATUS = "pending_verification"
PASSED_STATUS = "passed"
BLOCKED_STATUS = "blocked"
WARN_STATUS = "WARN"
VERIFICATION_FIELDS = {"cross_checks", "verifier_status", "blocking_reasons", "verified_line_hash"}
REVIEW_PACKET_MODES = {"stratified", "severity-first", "first-n"}
DECISION_TEMPLATE_SCOPES = {"packet", "full-ledger"}
REVIEW_DECISIONS = {"approve_repair", "reject_repair", "needs_context", "not_supersession", "uncertain"}
APPROVED_MECHANICAL_FIELDS = {
    "currency_status",
    "status",
    "subject_key",
    "superseded_at",
    "supersession_reason",
}
CLEAR_SUPERSESSION_EDGE_OPERATION = "clear_supersession_edge"
NORMALIZE_SUPERSESSION_CHAIN_OPERATION = "normalize_supersession_chain"
SUPERSESSION_EDGE_FIELDS = ("superseded_by", "supersession_reason", "superseded_at")
NORMALIZE_SUPERSESSION_BEFORE_FIELDS = (
    "status",
    "currency_status",
    "is_current",
    "subject_key",
    "superseded_by",
    "supersession_reason",
    "superseded_at",
)
PRESERVED_EDGE_REPAIR_FIELDS = (
    "status",
    "currency_status",
    "is_current",
    "authority_score",
    "effective_authority",
    "version_chain_head",
    "subject_key",
)
APPLYABLE_REPAIR_CLASSIFICATIONS = {
    LEDGER_MECHANICAL_CLASSIFICATION,
    LEDGER_BAD_EDGE_SUPERSESSION_CLASSIFICATION,
    LEDGER_REVIEWED_EDGE_SUPERSESSION_CLASSIFICATION,
}
DISALLOWED_REVIEWER_MARKERS = {"synthetic", "automation", "automated", "non_approval"}
OPERATION_BLOCKER_FIELDS = {"missing_subject_key_source", "missing_superseded_at_source"}
DECISION_TEMPLATE_FIELDS = [
    "line_number",
    "old_id",
    "verified_line_hash",
    "verified_ledger_sha256",
    "source_old_ids_sha256",
    "review_decision",
    "reviewer",
    "reviewed_at",
    "reviewer_notes",
]
DECISION_LINE_IDENTITY_FIELDS = [
    "line_number",
    "old_id",
    "verified_line_hash",
    "verified_ledger_sha256",
    "source_old_ids_sha256",
]


@dataclass(frozen=True)
class LedgerBundle:
    """In-memory representation of ledger lines plus manifest."""

    lines: list[dict[str, Any]]
    manifest: dict[str, Any]
    source_audit: dict[str, Any]


@dataclass(frozen=True)
class LedgerVerification:
    """Verification result for a repair ledger."""

    status: str
    hard_fail_count: int
    line_count: int
    verifier_status_counts: dict[str, int]
    duplicate_old_ids: list[str]
    omitted_old_ids: list[str]
    unexpected_old_ids: list[str]
    line_failures: list[dict[str, Any]]
    hard_integrity_failures: list[dict[str, Any]] | None = None
    row_trust_failures: list[dict[str, Any]] | None = None
    blocked_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "hard_fail_count": self.hard_fail_count,
            "line_count": self.line_count,
            "verifier_status_counts": self.verifier_status_counts,
            "duplicate_old_ids": self.duplicate_old_ids,
            "omitted_old_ids": self.omitted_old_ids,
            "unexpected_old_ids": self.unexpected_old_ids,
            "line_failures": self.line_failures,
            "hard_integrity_failures": self.hard_integrity_failures or [],
            "row_trust_failures": self.row_trust_failures or [],
            "blocked_count": self.blocked_count,
        }


def generate_ledger(
    conn: sqlite3.Connection,
    *,
    source_db_path: str | None = None,
    generated_at: str | None = None,
    strong_source_hash: bool = False,
) -> LedgerBundle:
    """Generate one deterministic ledger line per explicit supersession audit row."""

    generated_at = generated_at or datetime.now(UTC).isoformat()
    audit = audit_supersession_chains(conn, max_review_examples=0)
    results = sorted(audit["results"], key=lambda row: row["old_id"])
    source_snapshot = _source_snapshot(
        results,
        audit["summary"],
        source_db_path,
        generated_at,
        strong_source_hash=strong_source_hash,
    )

    lines: list[dict[str, Any]] = []
    for line_number, audit_row in enumerate(results, start=1):
        lines.append(_build_line(line_number, audit_row, source_snapshot))

    manifest = _build_manifest(lines, audit, source_snapshot, generated_at)
    return LedgerBundle(lines=lines, manifest=manifest, source_audit=audit)


def verify_ledger(conn: sqlite3.Connection, bundle: LedgerBundle) -> LedgerVerification:
    """Verify every ledger line against audit replay and direct SQL probes."""

    _verified_bundle, verification = verify_bundle(conn, bundle, strict_row_trust=True)
    return verification


def verify_bundle(
    conn: sqlite3.Connection,
    bundle: LedgerBundle,
    *,
    strict_row_trust: bool = False,
) -> tuple[LedgerBundle, LedgerVerification]:
    """Return a bundle annotated with verifier truth plus a structured verification result."""

    audit = audit_supersession_chains(conn, max_review_examples=0)
    audit_rows = {row["old_id"]: row for row in audit["results"]}
    expected_old_ids = set(audit_rows)
    line_old_ids = [str(line.get("old_id") or "") for line in bundle.lines]
    duplicates = sorted(item for item, count in Counter(line_old_ids).items() if item and count > 1)
    observed_old_ids = {item for item in line_old_ids if item}
    omitted = sorted(expected_old_ids - observed_old_ids)
    unexpected = sorted(observed_old_ids - expected_old_ids)

    hard_failures: list[dict[str, Any]] = []
    row_failures: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    verified_lines: list[dict[str, Any]] = []
    for index, line in enumerate(bundle.lines, start=1):
        hard_line_failures: list[str] = []
        row_line_failures: list[str] = []
        if line.get("line_number") != index:
            hard_line_failures.append("line_number_mismatch")
        stored_hash = line.get("ledger_line_hash")
        if _line_hash(line) != stored_hash and _legacy_line_hash(line) != stored_hash:
            hard_line_failures.append("ledger_line_hash_mismatch")
        old_id = str(line.get("old_id") or "")
        audit_row = audit_rows.get(old_id)
        if not audit_row:
            hard_line_failures.append("source_row_missing")
        elif _audit_row_hash(audit_row) != line.get("source_audit_row_hash"):
            hard_line_failures.append("source_audit_row_hash_mismatch")
        row_line_failures.extend(_audit_replay_disagreements(line, audit_row))
        row_line_failures.extend(_direct_sql_disagreements(conn, line))

        expected_status = PASSED_STATUS if not hard_line_failures and not row_line_failures else BLOCKED_STATUS
        status_counts[expected_status] += 1
        stored_status = line.get("verifier_status")
        if stored_status not in {GENERATED_STATUS, expected_status}:
            row_line_failures.append("stored_verifier_status_mismatch")

        line_failures = sorted(set(hard_line_failures + row_line_failures))
        verified_line = dict(line)
        verified_line["cross_checks"] = _line_cross_checks(hard_line_failures, row_line_failures)
        verified_line["verifier_status"] = expected_status
        verified_line["blocking_reasons"] = line_failures
        verified_line["verified_line_hash"] = _verified_line_hash(verified_line)
        verified_lines.append(verified_line)

        if hard_line_failures:
            hard_failures.append(
                {
                    "line_number": index,
                    "old_id": old_id or None,
                    "failures": sorted(set(hard_line_failures)),
                }
            )
        if row_line_failures:
            row_failures.append(
                {
                    "line_number": index,
                    "old_id": old_id or None,
                    "failures": sorted(set(row_line_failures)),
                }
            )

    if duplicates:
        hard_failures.append(
            {"line_number": None, "old_id": None, "failures": ["duplicate_old_ids"], "ids": duplicates}
        )
    if omitted:
        hard_failures.append({"line_number": None, "old_id": None, "failures": ["omitted_old_ids"], "ids": omitted})
    if unexpected:
        hard_failures.append(
            {"line_number": None, "old_id": None, "failures": ["unexpected_old_ids"], "ids": unexpected}
        )

    strict_fail_count = len(hard_failures) + (len(row_failures) if strict_row_trust else 0)
    if hard_failures or (strict_row_trust and row_failures):
        status = "FAIL"
    elif row_failures:
        status = WARN_STATUS
    else:
        status = "PASS"
    verification = LedgerVerification(
        status=status,
        hard_fail_count=strict_fail_count,
        line_count=len(bundle.lines),
        verifier_status_counts=dict(sorted(status_counts.items())),
        duplicate_old_ids=duplicates,
        omitted_old_ids=omitted,
        unexpected_old_ids=unexpected,
        line_failures=[*hard_failures, *row_failures],
        hard_integrity_failures=hard_failures,
        row_trust_failures=row_failures,
        blocked_count=status_counts.get(BLOCKED_STATUS, 0),
    )
    verified_manifest = _verified_manifest(bundle.manifest, verified_lines, verification)
    return LedgerBundle(lines=verified_lines, manifest=verified_manifest, source_audit=audit), verification


def _failure_reason_counts(failures: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for failure in failures:
        for reason in failure.get("failures") or []:
            counts[str(reason)] += 1
    return dict(sorted(counts.items()))


def _scoped_apply_verification(
    verified_bundle: LedgerBundle,
    verification: LedgerVerification,
    apply_units: list[dict[str, Any]],
) -> dict[str, Any]:
    verified_by_line = {str(line.get("line_number") or ""): line for line in verified_bundle.lines}
    selected_line_numbers = [str(unit.get("row_number") or "") for unit in apply_units]
    selected_line_number_set = set(selected_line_numbers)
    selected_failures = [
        failure
        for failure in verification.line_failures
        if str(failure.get("line_number") or "") in selected_line_number_set
    ]
    ambient_failures = [
        failure
        for failure in verification.line_failures
        if str(failure.get("line_number") or "") not in selected_line_number_set
    ]
    missing_selected_rows: list[str] = []
    hash_mismatches: list[dict[str, Any]] = []
    for unit in apply_units:
        row_number = str(unit.get("row_number") or "")
        line = verified_by_line.get(row_number)
        if not line:
            missing_selected_rows.append(row_number)
            continue
        expected_hash = (unit.get("source_evidence") or {}).get("verified_line_hash")
        if line.get("verified_line_hash") != expected_hash:
            hash_mismatches.append({"row_number": row_number, "old_id": line.get("old_id")})

    errors: list[str] = []
    if missing_selected_rows:
        errors.append(f"missing_selected_rows:{len(missing_selected_rows)}")
    if selected_failures:
        errors.append(f"selected_row_failures:{len(selected_failures)}")
    if hash_mismatches:
        errors.append(f"selected_verified_line_hash_mismatches:{len(hash_mismatches)}")

    return {
        "schema_version": SCOPED_APPLY_VERIFICATION_SCHEMA_VERSION,
        "status": "FAIL" if errors else "PASS",
        "claim_boundary": SCOPED_APPLY_CLAIM_BOUNDARY,
        "selected_line_count": len(apply_units),
        "selected_line_numbers": selected_line_numbers,
        "selected_failures": selected_failures,
        "selected_verified_line_hash_mismatches": hash_mismatches,
        "missing_selected_rows": missing_selected_rows,
        "ambient_full_ledger_status": verification.status,
        "ambient_failure_count": len(ambient_failures),
        "ambient_failure_reason_counts": _failure_reason_counts(ambient_failures),
        "full_ledger_line_count": verification.line_count,
        "full_ledger_blocked_count": verification.blocked_count,
        "errors": errors,
    }


def write_ledger_files(bundle: LedgerBundle, out_dir: Path) -> dict[str, Path]:
    """Write deterministic ledger artifacts into out_dir."""

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "ledger": out_dir / LEDGER_FILENAME,
        "manifest": out_dir / MANIFEST_FILENAME,
        "summary": out_dir / SUMMARY_FILENAME,
        "review_packet": out_dir / REVIEW_PACKET_FILENAME,
    }
    with paths["ledger"].open("w", encoding="utf-8") as handle:
        for line in bundle.lines:
            handle.write(_canonical_json(line))
            handle.write("\n")
    paths["manifest"].write_text(json.dumps(bundle.manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["summary"].write_text(json.dumps(_summary(bundle), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["review_packet"].write_text(render_review_packet(bundle), encoding="utf-8")
    return paths


def load_ledger(ledger_path: Path, manifest_path: Path) -> LedgerBundle:
    """Load ledger files written by write_ledger_files."""

    lines: list[dict[str, Any]] = []
    with ledger_path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            text = raw_line.strip()
            if text:
                lines.append(json.loads(text))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return LedgerBundle(lines=lines, manifest=manifest, source_audit={})


def _approved_mechanical_apply_plan(
    *,
    out_dir: Path,
    decisions_path: Path,
    queue_manifest_path: Path,
    approved_verified_ledger_sha256: str,
    approved_decisions_sha256: str,
    approved_queue_sha256: str,
    required_decision_line_set_path: Path | None = None,
    approved_required_decision_line_set_sha256: str = "",
    allow_partial_review: bool = False,
) -> tuple[LedgerBundle, dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    for label, value in {
        "approved_verified_ledger_sha256": approved_verified_ledger_sha256,
        "approved_decisions_sha256": approved_decisions_sha256,
        "approved_queue_sha256": approved_queue_sha256,
    }.items():
        if not HASH_RE.fullmatch(value):
            raise ValueError(f"{label} must be a 64-character lowercase hex digest")
    if _file_sha256(decisions_path) != approved_decisions_sha256:
        raise ValueError("approved decisions hash mismatch")
    if _file_sha256(queue_manifest_path) != approved_queue_sha256:
        raise ValueError("approved queue hash mismatch")

    stored_bundle = load_ledger(out_dir / LEDGER_FILENAME, out_dir / MANIFEST_FILENAME)
    stored_sha = str(stored_bundle.manifest.get("verified_ledger_sha256") or "")
    if stored_sha != approved_verified_ledger_sha256:
        raise ValueError(f"approved verified ledger hash mismatch: manifest has {stored_sha}")

    required_line_set_report = _not_required_line_set_report()
    if allow_partial_review and required_decision_line_set_path is None:
        raise ValueError("--allow-partial-review requires --required-decision-line-set-tsv for approved mechanical apply")
    if required_decision_line_set_path is not None:
        if not HASH_RE.fullmatch(approved_required_decision_line_set_sha256):
            raise ValueError("approved_required_decision_line_set_sha256 must be a 64-character lowercase hex digest")
        if _file_sha256(required_decision_line_set_path) != approved_required_decision_line_set_sha256:
            raise ValueError("approved required decision line-set hash mismatch")
        required_line_set_report = validate_required_decision_line_set(
            decisions_path,
            required_decision_line_set_path,
        )
        if required_line_set_report["status"] == "FAIL":
            raise ValueError(f"required line-set validation failed:{required_line_set_report['errors']}")

    decision_report = validate_review_decisions(
        stored_bundle,
        decisions_path,
        allow_partial=required_decision_line_set_path is not None,
    )
    if decision_report["status"] == "FAIL":
        raise ValueError(f"decision validation failed:{decision_report['errors']}")

    decision_rows = load_decision_rows(decisions_path)
    queue_manifest = json.loads(queue_manifest_path.read_text(encoding="utf-8"))
    generated_queue = build_repair_work_units_from_ledger(
        stored_bundle,
        decision_rows,
        ledger_path=out_dir / LEDGER_FILENAME,
        manifest_path=out_dir / MANIFEST_FILENAME,
        decisions_path=decisions_path,
    )
    if _stable_queue_projection(queue_manifest) != _stable_queue_projection(generated_queue):
        raise ValueError("queue manifest stable projection mismatch")

    apply_units = [
        unit for unit in generated_queue["work_units"] if unit.get("classification") in APPLYABLE_REPAIR_CLASSIFICATIONS
    ]
    _validate_apply_units(apply_units)
    return stored_bundle, decision_report, required_line_set_report, generated_queue, apply_units


def apply_approved_mechanical_repairs(
    *,
    source_db_path: Path,
    out_dir: Path,
    decisions_path: Path,
    queue_manifest_path: Path,
    approved_verified_ledger_sha256: str,
    approved_decisions_sha256: str,
    approved_queue_sha256: str,
    required_decision_line_set_path: Path | None = None,
    approved_required_decision_line_set_sha256: str = "",
    copy_db_path: Path | None = None,
    dry_run: bool = True,
    allow_partial_review: bool = False,
) -> dict[str, Any]:
    """Apply hash-pinned reviewed mechanical repair units to a copied DB."""

    stored_bundle, decision_report, required_line_set_report, generated_queue, apply_units = _approved_mechanical_apply_plan(
        out_dir=out_dir,
        decisions_path=decisions_path,
        queue_manifest_path=queue_manifest_path,
        approved_verified_ledger_sha256=approved_verified_ledger_sha256,
        approved_decisions_sha256=approved_decisions_sha256,
        approved_queue_sha256=approved_queue_sha256,
        required_decision_line_set_path=required_decision_line_set_path,
        approved_required_decision_line_set_sha256=approved_required_decision_line_set_sha256,
        allow_partial_review=allow_partial_review,
    )

    if dry_run:
        report = _approved_mechanical_apply_report(
            "DRY_RUN",
            stored_bundle,
            decision_report,
            required_line_set_report,
            generated_queue,
            apply_units,
            changed_rows=0,
            target_db_path=copy_db_path,
        )
        _write_approved_mechanical_apply_report(out_dir, report)
        return report
    if copy_db_path is None:
        raise ValueError("--copy-db is required for non-dry-run approved mechanical apply")
    if source_db_path.resolve() == copy_db_path.resolve():
        raise ValueError("--copy-db must not equal --db-path")
    if copy_db_path.exists():
        raise ValueError(f"--copy-db already exists: {copy_db_path}")

    _sqlite_backup(source_db_path, copy_db_path)
    changed_rows = 0
    with _connect(copy_db_path) as conn:
        verified_bundle, verification = verify_bundle(conn, stored_bundle, strict_row_trust=False)
        scoped_verification = _scoped_apply_verification(verified_bundle, verification, apply_units)
        if scoped_verification["status"] != "PASS":
            raise ValueError(f"scoped ledger verification blocked apply:{scoped_verification['errors']}")
        with conn:
            for unit in apply_units:
                changed_rows += _apply_concept_metadata_operations_conn(
                    conn,
                    str(unit["chain_key"]["old_id"]),
                    list(unit.get("proposed_operations") or []),
                )

    report = _approved_mechanical_apply_report(
        "PASS",
        stored_bundle,
        decision_report,
        required_line_set_report,
        generated_queue,
        apply_units,
        changed_rows=changed_rows,
        target_db_path=copy_db_path,
        scoped_verification=scoped_verification,
    )
    _write_approved_mechanical_apply_report(out_dir, report)
    return report


def apply_approved_mechanical_repairs_live(
    *,
    source_db_path: Path,
    out_dir: Path,
    decisions_path: Path,
    queue_manifest_path: Path,
    approved_verified_ledger_sha256: str,
    approved_decisions_sha256: str,
    approved_queue_sha256: str,
    required_decision_line_set_path: Path | None = None,
    approved_required_decision_line_set_sha256: str = "",
    dry_run: bool = True,
    confirm_live_apply: bool = False,
    operator_approval_token: str = "",
    maintenance_window: bool = False,
    force_active_db_holders: int | None = None,
    evidence_root: Path | None = None,
    backup_db_path: Path | None = None,
    allow_partial_review: bool = False,
    scorecard_runner: Callable[[Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Dry-run or apply approved mechanical repairs directly to the source DB."""

    if dry_run and confirm_live_apply:
        raise ValueError("--dry-run cannot be combined with --confirm-live-apply")
    source_db_path = source_db_path.expanduser().resolve()
    evidence_dir = _live_apply_evidence_dir(evidence_root)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    stored_bundle, decision_report, required_line_set_report, generated_queue, apply_units = _approved_mechanical_apply_plan(
        out_dir=out_dir,
        decisions_path=decisions_path,
        queue_manifest_path=queue_manifest_path,
        approved_verified_ledger_sha256=approved_verified_ledger_sha256,
        approved_decisions_sha256=approved_decisions_sha256,
        approved_queue_sha256=approved_queue_sha256,
        required_decision_line_set_path=required_decision_line_set_path,
        approved_required_decision_line_set_sha256=approved_required_decision_line_set_sha256,
        allow_partial_review=allow_partial_review,
    )
    expected_token = _live_apply_approval_token(
        approved_verified_ledger_sha256,
        approved_decisions_sha256,
        approved_queue_sha256,
        len(apply_units),
    )
    active_holders = _active_db_holders(source_db_path)
    before_metrics = _live_db_metrics(source_db_path)
    integrity = _sqlite_integrity_check(source_db_path)
    lock_available, lock_message = _probe_sqlite_write_lock(source_db_path)
    preflight = {
        "schema_version": LIVE_APPROVED_MECHANICAL_APPLY_REPORT_SCHEMA_VERSION,
        "mode": "live_approved_mechanical_preflight",
        "status": "PASS" if integrity == "ok" and lock_available else "FAIL",
        "source_db_path": str(source_db_path),
        "source_integrity_check": integrity,
        "write_lock_available": lock_available,
        "write_lock_message": lock_message,
        "active_db_holder_count": active_holders["count"],
        "active_db_holders": active_holders["holders"],
        "active_db_holders_forced": False,
        "expected_operator_approval_token": expected_token,
        "eligible_line_count": len(apply_units),
        "decision_validation_status": decision_report.get("status"),
        "required_line_set_validation": required_line_set_report,
        "queue_summary": generated_queue.get("summary", {}),
        "before_metrics": before_metrics,
    }
    _write_live_apply_preflight_report(evidence_dir, preflight)

    if integrity != "ok":
        raise ValueError("source DB integrity check failed")
    if not lock_available:
        raise ValueError(f"source DB write lock unavailable:{lock_message}")

    with _connect_readonly(source_db_path) as conn:
        verified_bundle, verification = verify_bundle(conn, stored_bundle, strict_row_trust=False)
        scoped_verification = _scoped_apply_verification(verified_bundle, verification, apply_units)

    if dry_run or not confirm_live_apply:
        report = _live_apply_report(
            "DRY_RUN" if scoped_verification["status"] == "PASS" else "ALARM",
            stored_bundle,
            decision_report,
            required_line_set_report,
            generated_queue,
            apply_units,
            changed_rows=0,
            committed=False,
            source_db_path=source_db_path,
            evidence_dir=evidence_dir,
            expected_token=expected_token,
            preflight=preflight,
            backup_report=None,
            before_metrics=before_metrics,
            after_metrics=before_metrics,
            post_checks={},
            rollback_runbook_path=None,
            scoped_verification=scoped_verification,
        )
        _write_live_apply_report(evidence_dir, report)
        return report

    if operator_approval_token != expected_token:
        raise ValueError("operator approval token mismatch")
    if not maintenance_window:
        raise ValueError("--maintenance-window is required for confirmed live apply")
    if active_holders["count"] < 0:
        raise ValueError(f"active DB holder probe failed:{active_holders.get('error') or 'unknown'}")
    if active_holders["count"] and force_active_db_holders != active_holders["count"]:
        raise ValueError("--force-active-db-holders must match observed active DB holder count")

    preflight["active_db_holders_forced"] = bool(active_holders["count"])
    _write_live_apply_preflight_report(evidence_dir, preflight)
    backup_path = (backup_db_path.expanduser() if backup_db_path else _default_live_apply_backup_path(source_db_path)).resolve()
    if backup_path == source_db_path:
        raise ValueError("backup DB path must not equal source DB path")
    if backup_path.exists():
        raise ValueError(f"backup DB already exists:{backup_path}")
    _sqlite_backup(source_db_path, backup_path)
    backup_report = _db_file_report(backup_path, strong_hash=True)
    backup_report["integrity_check"] = _sqlite_integrity_check(backup_path)
    if backup_report["integrity_check"] != "ok":
        raise ValueError("pre-apply backup integrity check failed")

    changed_rows = 0
    row_evidence: list[dict[str, Any]] = []
    with _connect(source_db_path) as conn:
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("BEGIN IMMEDIATE")
        try:
            verified_bundle, verification = verify_bundle(conn, stored_bundle, strict_row_trust=False)
            scoped_verification = _scoped_apply_verification(verified_bundle, verification, apply_units)
            if scoped_verification["status"] != "PASS":
                raise ValueError(f"scoped ledger verification blocked live apply:{scoped_verification['errors']}")
            for unit in apply_units:
                concept_id = str(unit["chain_key"]["old_id"])
                before_row = _fetch_concept_repair_row(conn, concept_id)
                changed_rows += _apply_concept_metadata_operations_conn(
                    conn,
                    concept_id,
                    list(unit.get("proposed_operations") or []),
                )
                after_row = _fetch_concept_repair_row(conn, concept_id)
                row_evidence.append(_live_apply_row_evidence(unit, before_row, after_row))
            if changed_rows != len(apply_units):
                raise ValueError("changed row count did not equal eligible line count")
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    _write_jsonl(evidence_dir / LIVE_APPROVED_MECHANICAL_ROW_EVIDENCE_FILENAME, row_evidence)
    rollback_runbook_path = evidence_dir / LIVE_APPROVED_MECHANICAL_ROLLBACK_RUNBOOK_FILENAME
    _write_live_apply_rollback_runbook(rollback_runbook_path, source_db_path, backup_path)
    after_metrics = _live_db_metrics(source_db_path)
    post_checks = _run_live_apply_post_checks(
        source_db_path,
        before_metrics=before_metrics,
        after_metrics=after_metrics,
        apply_units=apply_units,
        changed_rows=changed_rows,
        row_evidence_count=len(row_evidence),
        scorecard_runner=scorecard_runner,
    )
    status = "PASS" if _post_checks_pass(post_checks) else "ALARM"
    report = _live_apply_report(
        status,
        stored_bundle,
        decision_report,
        required_line_set_report,
        generated_queue,
        apply_units,
        changed_rows=changed_rows,
        committed=True,
        source_db_path=source_db_path,
        evidence_dir=evidence_dir,
        expected_token=expected_token,
        preflight=preflight,
        backup_report=backup_report,
        before_metrics=before_metrics,
        after_metrics=after_metrics,
        post_checks=post_checks,
        rollback_runbook_path=rollback_runbook_path,
        scoped_verification=scoped_verification,
    )
    _write_live_apply_report(evidence_dir, report)
    return report


def apply_curated_repairs(
    *,
    source_db_path: Path,
    out_dir: Path,
    curated_manifest_path: Path,
    approved_curated_manifest_sha256: str,
    copy_db_path: Path | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Apply a hash-pinned curated normalization manifest to a copied DB."""

    out_dir.mkdir(parents=True, exist_ok=True)
    bundle, decision_report, required_line_set_report, curated_manifest, apply_units = _curated_repair_apply_plan(
        curated_manifest_path=curated_manifest_path,
        approved_curated_manifest_sha256=approved_curated_manifest_sha256,
    )
    if dry_run:
        report = _approved_mechanical_apply_report(
            "DRY_RUN",
            bundle,
            decision_report,
            required_line_set_report,
            curated_manifest,
            apply_units,
            changed_rows=0,
            target_db_path=copy_db_path,
        )
        report.update(
            {
                "mode": "curated_repair_apply",
                "operator_surface": "curated_repair",
                "legacy_operator_surface": None,
                "curated_manifest_sha256": approved_curated_manifest_sha256,
            }
        )
        _write_approved_mechanical_apply_report(out_dir, report)
        return report
    if copy_db_path is None:
        raise ValueError("--copy-db is required for non-dry-run curated repair apply")
    if source_db_path.resolve() == copy_db_path.resolve():
        raise ValueError("--copy-db must not equal --db-path")
    if copy_db_path.exists():
        raise ValueError(f"--copy-db already exists: {copy_db_path}")

    _sqlite_backup(source_db_path, copy_db_path)
    changed_rows = 0
    with _connect(copy_db_path) as conn, conn:
        for unit in apply_units:
            changed_rows += _apply_concept_metadata_operations_conn(
                conn,
                str(unit["chain_key"]["old_id"]),
                list(unit.get("proposed_operations") or []),
            )

    report = _approved_mechanical_apply_report(
        "PASS",
        bundle,
        decision_report,
        required_line_set_report,
        curated_manifest,
        apply_units,
        changed_rows=changed_rows,
        target_db_path=copy_db_path,
    )
    report.update(
        {
            "mode": "curated_repair_apply",
            "operator_surface": "curated_repair",
            "legacy_operator_surface": None,
            "curated_manifest_sha256": approved_curated_manifest_sha256,
        }
    )
    _write_approved_mechanical_apply_report(out_dir, report)
    return report


def apply_curated_repairs_live(
    *,
    source_db_path: Path,
    curated_manifest_path: Path,
    approved_curated_manifest_sha256: str,
    dry_run: bool = True,
    confirm_live_apply: bool = False,
    operator_approval_token: str = "",
    maintenance_window: bool = False,
    force_active_db_holders: int | None = None,
    evidence_root: Path | None = None,
    backup_db_path: Path | None = None,
    scorecard_runner: Callable[[Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Dry-run or apply a hash-pinned curated normalization manifest to the source DB."""

    if dry_run and confirm_live_apply:
        raise ValueError("--dry-run cannot be combined with --confirm-live-apply")
    source_db_path = source_db_path.expanduser().resolve()
    evidence_dir = _live_apply_evidence_dir(evidence_root)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    bundle, decision_report, required_line_set_report, curated_manifest, apply_units = _curated_repair_apply_plan(
        curated_manifest_path=curated_manifest_path,
        approved_curated_manifest_sha256=approved_curated_manifest_sha256,
    )
    expected_token = _curated_live_apply_approval_token(approved_curated_manifest_sha256, len(apply_units))
    active_holders = _active_db_holders(source_db_path)
    before_metrics = _live_db_metrics(source_db_path)
    integrity = _sqlite_integrity_check(source_db_path)
    lock_available, lock_message = _probe_sqlite_write_lock(source_db_path)
    preflight = {
        "schema_version": LIVE_APPROVED_MECHANICAL_APPLY_REPORT_SCHEMA_VERSION,
        "mode": "live_curated_repair_preflight",
        "status": "PASS" if integrity == "ok" and lock_available else "FAIL",
        "source_db_path": str(source_db_path),
        "source_integrity_check": integrity,
        "write_lock_available": lock_available,
        "write_lock_message": lock_message,
        "active_db_holder_count": active_holders["count"],
        "active_db_holders": active_holders["holders"],
        "active_db_holders_forced": False,
        "expected_operator_approval_token": expected_token,
        "eligible_line_count": len(apply_units),
        "decision_validation_status": decision_report.get("status"),
        "required_line_set_validation": required_line_set_report,
        "queue_summary": curated_manifest.get("summary", {}),
        "curated_manifest_sha256": approved_curated_manifest_sha256,
        "before_metrics": before_metrics,
    }
    _write_live_apply_preflight_report(evidence_dir, preflight)

    if integrity != "ok":
        raise ValueError("source DB integrity check failed")
    if not lock_available:
        raise ValueError(f"source DB write lock unavailable:{lock_message}")

    if dry_run or not confirm_live_apply:
        report = _live_apply_report(
            "DRY_RUN",
            bundle,
            decision_report,
            required_line_set_report,
            curated_manifest,
            apply_units,
            changed_rows=0,
            committed=False,
            source_db_path=source_db_path,
            evidence_dir=evidence_dir,
            expected_token=expected_token,
            preflight=preflight,
            backup_report=None,
            before_metrics=before_metrics,
            after_metrics=before_metrics,
            post_checks={},
            rollback_runbook_path=None,
        )
        _mark_curated_live_report(report, approved_curated_manifest_sha256)
        _write_live_apply_report(evidence_dir, report)
        return report

    if operator_approval_token != expected_token:
        raise ValueError("operator approval token mismatch")
    if not maintenance_window:
        raise ValueError("--maintenance-window is required for confirmed live apply")
    if active_holders["count"] < 0:
        raise ValueError(f"active DB holder probe failed:{active_holders.get('error') or 'unknown'}")
    if active_holders["count"] and force_active_db_holders != active_holders["count"]:
        raise ValueError("--force-active-db-holders must match observed active DB holder count")

    preflight["active_db_holders_forced"] = bool(active_holders["count"])
    _write_live_apply_preflight_report(evidence_dir, preflight)
    backup_path = (backup_db_path.expanduser() if backup_db_path else _default_live_apply_backup_path(source_db_path)).resolve()
    if backup_path == source_db_path:
        raise ValueError("backup DB path must not equal source DB path")
    if backup_path.exists():
        raise ValueError(f"backup DB already exists:{backup_path}")
    _sqlite_backup(source_db_path, backup_path)
    backup_report = _db_file_report(backup_path, strong_hash=True)
    backup_report["integrity_check"] = _sqlite_integrity_check(backup_path)
    if backup_report["integrity_check"] != "ok":
        raise ValueError("pre-apply backup integrity check failed")

    changed_rows = 0
    row_evidence: list[dict[str, Any]] = []
    with _connect(source_db_path) as conn:
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("BEGIN IMMEDIATE")
        try:
            for unit in apply_units:
                concept_id = str(unit["chain_key"]["old_id"])
                before_row = _fetch_concept_repair_row(conn, concept_id)
                changed_rows += _apply_concept_metadata_operations_conn(
                    conn,
                    concept_id,
                    list(unit.get("proposed_operations") or []),
                )
                after_row = _fetch_concept_repair_row(conn, concept_id)
                row_evidence.append(_live_apply_row_evidence(unit, before_row, after_row))
            if changed_rows != len(apply_units):
                raise ValueError("changed row count did not equal eligible line count")
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    _write_jsonl(evidence_dir / LIVE_APPROVED_MECHANICAL_ROW_EVIDENCE_FILENAME, row_evidence)
    rollback_runbook_path = evidence_dir / LIVE_APPROVED_MECHANICAL_ROLLBACK_RUNBOOK_FILENAME
    _write_live_apply_rollback_runbook(rollback_runbook_path, source_db_path, backup_path)
    after_metrics = _live_db_metrics(source_db_path)
    post_checks = _run_live_apply_post_checks(
        source_db_path,
        before_metrics=before_metrics,
        after_metrics=after_metrics,
        apply_units=apply_units,
        changed_rows=changed_rows,
        row_evidence_count=len(row_evidence),
        scorecard_runner=scorecard_runner,
    )
    status = "PASS" if _post_checks_pass(post_checks) else "ALARM"
    report = _live_apply_report(
        status,
        bundle,
        decision_report,
        required_line_set_report,
        curated_manifest,
        apply_units,
        changed_rows=changed_rows,
        committed=True,
        source_db_path=source_db_path,
        evidence_dir=evidence_dir,
        expected_token=expected_token,
        preflight=preflight,
        backup_report=backup_report,
        before_metrics=before_metrics,
        after_metrics=after_metrics,
        post_checks=post_checks,
        rollback_runbook_path=rollback_runbook_path,
    )
    _mark_curated_live_report(report, approved_curated_manifest_sha256)
    _write_live_apply_report(evidence_dir, report)
    return report


def apply_copy(
    *,
    source_db_path: Path,
    copy_db_path: Path,
    out_dir: Path,
    approved_ledger_sha256: str,
    scorecard_runner: Callable[[Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create a SQLite backup copy and apply only verified mutation-capable ledger lines."""

    if not HASH_RE.fullmatch(approved_ledger_sha256):
        raise ValueError("--approved-ledger-sha256 must be a 64-character lowercase hex digest")
    if source_db_path.resolve() == copy_db_path.resolve():
        raise ValueError("--copy-db must not equal --db-path")
    if copy_db_path.exists():
        raise ValueError(f"--copy-db already exists: {copy_db_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    _sqlite_backup(source_db_path, copy_db_path)

    stored_bundle = load_ledger(out_dir / LEDGER_FILENAME, out_dir / MANIFEST_FILENAME)
    stored_sha = str(
        stored_bundle.manifest.get("verified_ledger_sha256") or stored_bundle.manifest.get("ledger_sha256") or ""
    )
    if stored_sha != approved_ledger_sha256:
        raise ValueError(f"approved hash mismatch: manifest has {stored_sha}")

    with _connect(copy_db_path) as conn:
        before_bundle = generate_ledger(conn, source_db_path=str(copy_db_path))
        copy_snapshot_parity = _snapshot_matches(stored_bundle.manifest, before_bundle.manifest)
        verification = verify_ledger(conn, stored_bundle)
        if not copy_snapshot_parity or verification.status != "PASS":
            report = _apply_report(
                "FAIL", "blocked_apply_copy", stored_bundle, verification, copy_snapshot_parity, 0, None
            )
            _write_apply_report(out_dir, report)
            return report

        eligible = [line for line in stored_bundle.lines if line.get("proposed_action") == "repair_lifecycle_metadata"]
        changed_rows = 0
        if eligible:
            with conn:
                for line in eligible:
                    changed_rows += apply_lifecycle_transition_conn(
                        conn,
                        str(line["old_id"]),
                        "supersede",
                        superseded_by=str(line["replacement_id"]),
                        reason=str(line.get("action_rationale") or "SUPERSESSION-REPAIR-LEDGER-V1 repair"),
                    )
        after_bundle = generate_ledger(conn, source_db_path=str(copy_db_path))
        trust_report = (
            scorecard_runner(copy_db_path)
            if scorecard_runner
            else {"status": "SKIPPED", "reason": "scorecard_runner_not_supplied"}
        )

    mode = "apply_copy" if eligible else "noop_apply_copy"
    status = "PASS" if trust_report.get("status") in {"PASS", "SKIPPED"} else "FAIL"
    report = _apply_report(status, mode, stored_bundle, verification, copy_snapshot_parity, changed_rows, after_bundle)
    report["eligible_line_count"] = len(eligible)
    report["trust_scorecard_status"] = trust_report["status"]
    report["trust_scorecard_chain_audit_status"] = (trust_report.get("chain_audit") or {}).get("status")
    _write_apply_report(out_dir, report)
    return report


def render_review_packet(bundle: LedgerBundle, *, max_lines: int = 50) -> str:
    """Render a compatibility review packet from ledger lines."""
    if bundle.manifest.get("verified_ledger_sha256"):
        return render_verified_review_packet(bundle, max_lines=max_lines)
    lines = [
        "# Supersession Repair Ledger Diagnostic Packet",
        "",
        f"Generated at: {bundle.manifest.get('generated_at')}",
        f"Ledger SHA-256: `{bundle.manifest.get('ledger_sha256')}`",
        "",
        "Status: diagnostic-only; this packet is not verified and must not be used for approval.",
        "",
        "Generated classifications are maintenance candidates, not semantic truth labels.",
        "",
    ]
    for item in bundle.lines[:max_lines]:
        lines.extend(
            [
                f"## {item['line_number']}. `{item['old_id']}`",
                "",
                f"- Proposed action: `{item['proposed_action']}`",
                f"- Verifier status: `{item['verifier_status']}`",
                f"- Replacement: `{item.get('replacement_id') or 'missing'}`",
                f"- Terminal head: `{item.get('terminal_head_id') or 'missing'}`",
                f"- Issues: {', '.join(item.get('issue_flags') or []) or 'none'}",
                f"- Old timestamp: `{(item.get('before') or {}).get('old_created_at') or 'missing'}`",
                f"- Replacement timestamp: `{(item.get('before') or {}).get('replacement_created_at') or 'missing'}`",
                "",
                "**Old excerpt**",
                "",
                f"> {(item.get('before') or {}).get('old_summary') or '[missing]'}",
                "",
                "**Replacement excerpt**",
                "",
                f"> {(item.get('before') or {}).get('replacement_summary') or '[missing]'}",
                "",
            ]
        )
    return "\n".join(lines)


def render_verified_review_packet(
    bundle: LedgerBundle,
    *,
    max_lines: int = 50,
    mode: str = "stratified",
) -> str:
    """Render an approval-capable review packet from verified ledger lines."""
    if mode not in REVIEW_PACKET_MODES:
        raise ValueError(f"unknown review packet mode: {mode}")
    verified_sha = str(bundle.manifest.get("verified_ledger_sha256") or "")
    if not HASH_RE.fullmatch(verified_sha):
        raise ValueError("verified_ledger_sha256 is required for approval-capable review packets")
    hard_failures = bundle.manifest.get("hard_integrity_fail_count")
    if hard_failures:
        raise ValueError("hard ledger-integrity failures block approval-capable review packets")

    source_snapshot = bundle.manifest.get("source_snapshot") or {}
    selected = _select_packet_lines(bundle.lines, max_lines=max_lines, mode=mode)
    lines = [
        "# Supersession Repair Ledger Review Packet v3",
        "",
        f"Generated at: {bundle.manifest.get('generated_at')}",
        f"Verification status: `{bundle.manifest.get('status')}`",
        f"Ledger SHA-256: `{bundle.manifest.get('ledger_sha256')}`",
        f"Verified ledger SHA-256: `{verified_sha}`",
        f"Rows in full ledger: {len(bundle.lines)}",
        f"Rows in packet: {len(selected)}",
        f"Blocked rows: {bundle.manifest.get('blocked_count', 0)}",
        "",
        "Source snapshot:",
        f"- DB path: `{source_snapshot.get('source_db_path') or 'unknown'}`",
        f"- DB size: `{source_snapshot.get('source_db_size')}`",
        f"- DB mtime ns: `{source_snapshot.get('source_db_mtime_ns')}`",
        f"- Old-id hash: `{source_snapshot.get('source_old_ids_sha256')}`",
        "",
        "Decision options: approve_repair, reject_repair, needs_context, not_supersession, uncertain.",
        "",
    ]
    for item in selected:
        lines.extend(_packet_line_markdown(item, verified_sha))
    return "\n".join(lines)


def write_decision_template(
    bundle: LedgerBundle,
    path: Path,
    *,
    max_lines: int = 50,
    mode: str = "stratified",
    scope: str = "packet",
) -> None:
    """Write a TSV template bound to verified ledger and line hashes."""
    if scope not in DECISION_TEMPLATE_SCOPES:
        raise ValueError(f"unknown decision template scope: {scope}")
    verified_sha = str(bundle.manifest.get("verified_ledger_sha256") or "")
    if not HASH_RE.fullmatch(verified_sha):
        raise ValueError("verified_ledger_sha256 is required for decision templates")
    selected = (
        bundle.lines
        if scope == "full-ledger"
        else _select_packet_lines(bundle.lines, max_lines=max_lines, mode=mode)
    )
    source_snapshot = bundle.manifest.get("source_snapshot") or {}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DECISION_TEMPLATE_FIELDS, delimiter="\t")
        writer.writeheader()
        for line in selected:
            writer.writerow(
                {
                    "line_number": line.get("line_number"),
                    "old_id": line.get("old_id"),
                    "verified_line_hash": line.get("verified_line_hash"),
                    "verified_ledger_sha256": verified_sha,
                    "source_old_ids_sha256": source_snapshot.get("source_old_ids_sha256"),
                    "review_decision": "",
                    "reviewer": "",
                    "reviewed_at": "",
                    "reviewer_notes": "",
                }
            )


def validate_review_decisions(
    bundle: LedgerBundle,
    decisions_path: Path,
    *,
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Validate TSV review decisions against verified ledger hashes."""
    verified_sha = str(bundle.manifest.get("verified_ledger_sha256") or "")
    if not HASH_RE.fullmatch(verified_sha):
        return {"status": "FAIL", "errors": ["verified_ledger_sha256_missing"], "rows_reviewed": 0}
    source_snapshot = bundle.manifest.get("source_snapshot") or {}
    expected_source_hash = str(source_snapshot.get("source_old_ids_sha256") or "")
    expected_by_line = {str(line.get("line_number")): line for line in bundle.lines}
    errors: list[str] = []
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    with decisions_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing_columns = [field for field in DECISION_TEMPLATE_FIELDS if field not in (reader.fieldnames or [])]
        if missing_columns:
            return {"status": "FAIL", "errors": [f"missing_columns:{','.join(missing_columns)}"], "rows_reviewed": 0}
        for row_number, row in enumerate(reader, start=2):
            normalized = {key: (value or "").strip() for key, value in row.items()}
            rows.append(normalized)
            line_number = normalized.get("line_number", "")
            if line_number in seen:
                errors.append(f"duplicate_line_number:{line_number}")
            seen.add(line_number)
            line = expected_by_line.get(line_number)
            if not line:
                errors.append(f"unknown_line_number:{line_number or row_number}")
                continue
            if normalized.get("old_id") != line.get("old_id"):
                errors.append(f"old_id_mismatch:{line_number}")
            if normalized.get("verified_line_hash") != line.get("verified_line_hash"):
                errors.append(f"verified_line_hash_mismatch:{line_number}")
            if normalized.get("verified_ledger_sha256") != verified_sha:
                errors.append(f"verified_ledger_sha256_mismatch:{line_number}")
            if normalized.get("source_old_ids_sha256") != expected_source_hash:
                errors.append(f"source_old_ids_sha256_mismatch:{line_number}")
            decision = normalized.get("review_decision", "")
            if decision not in REVIEW_DECISIONS:
                errors.append(f"invalid_review_decision:{line_number}:{decision or 'blank'}")
            if not normalized.get("reviewer"):
                errors.append(f"missing_reviewer:{line_number}")
            if not normalized.get("reviewed_at"):
                errors.append(f"missing_reviewed_at:{line_number}")
    missing_review_count = len(bundle.lines) - len(seen)
    partial = missing_review_count > 0
    if partial and not allow_partial:
        errors.append(f"missing_review_count:{missing_review_count}")
    status = "FAIL" if errors else ("PARTIAL" if partial else "PASS")
    return {
        "status": status,
        "errors": sorted(errors),
        "rows_reviewed": len(rows),
        "ledger_row_count": len(bundle.lines),
        "missing_review_count": missing_review_count,
        "partial": partial,
        "decision_counts": dict(sorted(Counter(row.get("review_decision", "") for row in rows).items())),
        "rows": rows,
    }


def _decision_line_identity(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(str(row.get(field) or "").strip() for field in DECISION_LINE_IDENTITY_FIELDS)


def _not_required_line_set_report() -> dict[str, Any]:
    return {
        "status": "NOT_REQUIRED",
        "errors": [],
        "required_line_count": 0,
        "decision_line_count": 0,
        "identity_fields": list(DECISION_LINE_IDENTITY_FIELDS),
    }


def validate_required_decision_line_set(
    decisions_path: Path,
    required_line_set_path: Path,
) -> dict[str, Any]:
    """Validate that reviewed decisions exactly cover a required line-set template."""

    decision_rows = load_decision_rows(decisions_path)
    required_rows = load_decision_rows(required_line_set_path)
    decision_ids = [_decision_line_identity(row) for row in decision_rows]
    required_ids = [_decision_line_identity(row) for row in required_rows]
    duplicate_decision_ids = sorted(item for item, count in Counter(decision_ids).items() if count > 1)
    duplicate_required_ids = sorted(item for item, count in Counter(required_ids).items() if count > 1)
    missing = sorted(set(required_ids) - set(decision_ids))
    extra = sorted(set(decision_ids) - set(required_ids))
    errors: list[str] = []
    if duplicate_required_ids:
        errors.append(f"duplicate_required_line_ids:{len(duplicate_required_ids)}")
    if duplicate_decision_ids:
        errors.append(f"duplicate_decision_line_ids:{len(duplicate_decision_ids)}")
    if missing:
        errors.append(f"missing_required_line_ids:{len(missing)}")
    if extra:
        errors.append(f"unexpected_decision_line_ids:{len(extra)}")
    return {
        "status": "FAIL" if errors else "PASS",
        "errors": errors,
        "required_line_count": len(required_rows),
        "decision_line_count": len(decision_rows),
        "identity_fields": list(DECISION_LINE_IDENTITY_FIELDS),
    }


def main(argv: list[str] | None = None, *, scorecard_runner: Callable[[Path], dict[str, Any]] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate and verify supersession repair ledger artifacts.")
    parser.add_argument("--db-path", default=str(DB_PATH), help="Source SQLite DB path")
    parser.add_argument("--out-dir", required=True, help="Directory for ledger artifacts")
    parser.add_argument("--verify", action="store_true", help="Verify existing ledger artifacts in --out-dir")
    parser.add_argument("--copy-db", help="SQLite backup copy path for apply operations")
    parser.add_argument("--apply-copy", action="store_true", help="Apply verified mutation-capable rows to a copied DB")
    parser.add_argument(
        "--apply-approved-repair",
        dest="apply_approved_mechanical",
        action="store_true",
        help="Apply hash-pinned approved repair queue rows",
    )
    parser.add_argument(
        "--apply-approved-repair-live",
        dest="apply_approved_mechanical_live",
        action="store_true",
        help="Dry-run or apply hash-pinned approved repairs to --db-path with live safeguards",
    )
    parser.add_argument(
        "--apply-approved-mechanical",
        dest="apply_approved_mechanical",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--apply-approved-mechanical-live",
        dest="apply_approved_mechanical_live",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--apply-curated-repair",
        action="store_true",
        help="Apply a hash-pinned curated normalize_supersession_chain manifest to a copied DB",
    )
    parser.add_argument(
        "--apply-curated-repair-live",
        action="store_true",
        help="Dry-run or apply a hash-pinned curated normalize_supersession_chain manifest to --db-path",
    )
    parser.add_argument("--decisions-tsv", help="Ledger-bound reviewed decisions TSV")
    parser.add_argument("--queue-manifest-json", help="Reviewed repair queue manifest JSON")
    parser.add_argument("--curated-repair-manifest-json", help="Operator-reviewed curated repair manifest JSON")
    parser.add_argument("--approved-ledger-sha256", default="", help="Approved manifest ledger SHA-256")
    parser.add_argument("--approved-verified-ledger-sha256", default="", help="Approved verified ledger SHA-256")
    parser.add_argument("--approved-decisions-sha256", default="", help="Approved decisions TSV SHA-256")
    parser.add_argument("--approved-queue-sha256", default="", help="Approved queue manifest SHA-256")
    parser.add_argument(
        "--approved-curated-repair-manifest-sha256",
        default="",
        help="Approved curated repair manifest SHA-256",
    )
    parser.add_argument(
        "--required-decision-line-set-tsv",
        help="Hash-pinned TSV defining the required reviewed line set for partial mechanical apply",
    )
    parser.add_argument(
        "--approved-required-decision-line-set-sha256",
        default="",
        help="Approved required decision line-set TSV SHA-256",
    )
    parser.add_argument(
        "--confirm-live-apply",
        action="store_true",
        help="Permit live DB mutation when all live safeguards pass",
    )
    parser.add_argument(
        "--maintenance-window",
        action="store_true",
        help="Acknowledge this is an operator-controlled maintenance window",
    )
    parser.add_argument("--operator-approval-token", default="", help="Exact live apply approval token from dry-run report")
    parser.add_argument(
        "--force-active-db-holders",
        type=int,
        help="Force live apply only when value matches observed active DB holder count",
    )
    parser.add_argument(
        "--evidence-root",
        help="Directory for live apply evidence; defaults under ~/.pith/reports/maintenance",
    )
    parser.add_argument(
        "--backup-db",
        help="Pre-apply backup DB path; defaults under ~/pith-data/<profile>/snapshots",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not mutate; write an approved mechanical dry-run report",
    )
    parser.add_argument("--write-verified", action="store_true", help="Write verified ledger artifacts")
    parser.add_argument(
        "--review-packet-mode",
        choices=sorted(REVIEW_PACKET_MODES),
        default="stratified",
        help="Review packet row selection mode",
    )
    parser.add_argument("--review-packet-size", type=int, default=50, help="Maximum review packet rows")
    parser.add_argument("--decision-template", action="store_true", help="Write a ledger-bound decision TSV template")
    parser.add_argument(
        "--decision-template-scope",
        choices=sorted(DECISION_TEMPLATE_SCOPES),
        default="packet",
        help="Decision TSV scope: bounded packet rows or every verified ledger row",
    )
    parser.add_argument(
        "--repairability-contract",
        action="store_true",
        help="Write read-only repairability contract artifacts",
    )
    parser.add_argument(
        "--currency-parity-report",
        action="store_true",
        help="Write a read-only SQL/JSON currency parity classification report",
    )
    parser.add_argument("--repairability-queue-json", help="Queue manifest JSON for repairability contract")
    parser.add_argument("--repairability-output-dir", help="Output directory for repairability contract artifacts")
    parser.add_argument("--validate-decisions", help="Validate a ledger-bound decision TSV")
    parser.add_argument("--allow-partial-review", action="store_true", help="Allow validating a partial decision TSV")
    parser.add_argument("--strong-source-hash", action="store_true", help="Include source DB SHA-256 in snapshot")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args(argv)

    db_path = Path(args.db_path).expanduser()
    out_dir = Path(args.out_dir).expanduser()

    try:
        if args.validate_decisions and not args.verify:
            raise ValueError("--validate-decisions requires --verify; refusing to generate or overwrite ledger artifacts")

        if args.currency_parity_report:
            with _connect_readonly(db_path) as conn:
                report = audit_currency_parity(conn)
            out_dir.mkdir(parents=True, exist_ok=True)
            report_path = out_dir / CURRENCY_PARITY_REPORT_FILENAME
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            payload = {
                "status": "PASS",
                "mode": "currency_parity_report",
                "claim_boundary": report["claim_boundary"],
                "paths": {"currency_parity_report": str(report_path)},
                "summary": report["summary"],
            }
            return _emit(payload, json_mode=args.json)

        if args.apply_curated_repair_live:
            if not args.curated_repair_manifest_json:
                raise ValueError("--apply-curated-repair-live requires --curated-repair-manifest-json")
            report = apply_curated_repairs_live(
                source_db_path=db_path,
                curated_manifest_path=Path(args.curated_repair_manifest_json).expanduser(),
                approved_curated_manifest_sha256=args.approved_curated_repair_manifest_sha256,
                dry_run=args.dry_run or not args.confirm_live_apply,
                confirm_live_apply=args.confirm_live_apply,
                operator_approval_token=args.operator_approval_token,
                maintenance_window=args.maintenance_window,
                force_active_db_holders=args.force_active_db_holders,
                evidence_root=Path(args.evidence_root).expanduser() if args.evidence_root else None,
                backup_db_path=Path(args.backup_db).expanduser() if args.backup_db else None,
                scorecard_runner=scorecard_runner,
            )
            return _emit(report, json_mode=args.json)

        if args.apply_curated_repair:
            if not args.curated_repair_manifest_json:
                raise ValueError("--apply-curated-repair requires --curated-repair-manifest-json")
            report = apply_curated_repairs(
                source_db_path=db_path,
                out_dir=out_dir,
                curated_manifest_path=Path(args.curated_repair_manifest_json).expanduser(),
                approved_curated_manifest_sha256=args.approved_curated_repair_manifest_sha256,
                copy_db_path=Path(args.copy_db).expanduser() if args.copy_db else None,
                dry_run=args.dry_run or not args.copy_db,
            )
            return _emit(report, json_mode=args.json)

        if args.apply_approved_mechanical_live:
            if not args.decisions_tsv:
                raise ValueError("--apply-approved-repair-live requires --decisions-tsv")
            if not args.queue_manifest_json:
                raise ValueError("--apply-approved-repair-live requires --queue-manifest-json")
            report = apply_approved_mechanical_repairs_live(
                source_db_path=db_path,
                out_dir=out_dir,
                decisions_path=Path(args.decisions_tsv).expanduser(),
                queue_manifest_path=Path(args.queue_manifest_json).expanduser(),
                approved_verified_ledger_sha256=args.approved_verified_ledger_sha256,
                approved_decisions_sha256=args.approved_decisions_sha256,
                approved_queue_sha256=args.approved_queue_sha256,
                required_decision_line_set_path=(
                    Path(args.required_decision_line_set_tsv).expanduser()
                    if args.required_decision_line_set_tsv
                    else None
                ),
                approved_required_decision_line_set_sha256=args.approved_required_decision_line_set_sha256,
                dry_run=args.dry_run or not args.confirm_live_apply,
                confirm_live_apply=args.confirm_live_apply,
                operator_approval_token=args.operator_approval_token,
                maintenance_window=args.maintenance_window,
                force_active_db_holders=args.force_active_db_holders,
                evidence_root=Path(args.evidence_root).expanduser() if args.evidence_root else None,
                backup_db_path=Path(args.backup_db).expanduser() if args.backup_db else None,
                allow_partial_review=args.allow_partial_review,
            )
            return _emit(report, json_mode=args.json)

        if args.apply_approved_mechanical:
            if not args.decisions_tsv:
                raise ValueError("--apply-approved-repair requires --decisions-tsv")
            if not args.queue_manifest_json:
                raise ValueError("--apply-approved-repair requires --queue-manifest-json")
            report = apply_approved_mechanical_repairs(
                source_db_path=db_path,
                out_dir=out_dir,
                decisions_path=Path(args.decisions_tsv).expanduser(),
                queue_manifest_path=Path(args.queue_manifest_json).expanduser(),
                approved_verified_ledger_sha256=args.approved_verified_ledger_sha256,
                approved_decisions_sha256=args.approved_decisions_sha256,
                approved_queue_sha256=args.approved_queue_sha256,
                required_decision_line_set_path=(
                    Path(args.required_decision_line_set_tsv).expanduser()
                    if args.required_decision_line_set_tsv
                    else None
                ),
                approved_required_decision_line_set_sha256=args.approved_required_decision_line_set_sha256,
                copy_db_path=Path(args.copy_db).expanduser() if args.copy_db else None,
                dry_run=args.dry_run or not args.copy_db,
                allow_partial_review=args.allow_partial_review,
            )
            return _emit(report, json_mode=args.json)

        if args.apply_copy:
            if not args.copy_db:
                raise ValueError("--apply-copy requires --copy-db")
            report = apply_copy(
                source_db_path=db_path,
                copy_db_path=Path(args.copy_db).expanduser(),
                out_dir=out_dir,
                approved_ledger_sha256=args.approved_ledger_sha256,
                scorecard_runner=scorecard_runner,
            )
            return _emit(report, json_mode=args.json)

        if args.verify:
            if (args.decision_template or args.repairability_contract) and not (
                args.write_verified or args.validate_decisions
            ):
                bundle = load_ledger(out_dir / LEDGER_FILENAME, out_dir / MANIFEST_FILENAME)
                paths: dict[str, Path] = {}
                repairability_report: dict[str, Any] | None = None
                if args.decision_template:
                    write_decision_template(
                        bundle,
                        out_dir / DECISION_TEMPLATE_FILENAME,
                        max_lines=args.review_packet_size,
                        mode=args.review_packet_mode,
                        scope=args.decision_template_scope,
                    )
                    paths["decision_template"] = out_dir / DECISION_TEMPLATE_FILENAME
                if args.repairability_contract:
                    queue_manifest = (
                        _load_json_manifest(Path(args.repairability_queue_json).expanduser())
                        if args.repairability_queue_json
                        else None
                    )
                    repairability_output_dir = (
                        Path(args.repairability_output_dir).expanduser()
                        if args.repairability_output_dir
                        else out_dir / "repairability"
                    )
                    contract = build_repairability_contract_from_ledger(bundle, queue_manifest)
                    repairability_paths = write_repairability_contract_files(
                        contract,
                        repairability_output_dir,
                        max_packet_rows=args.review_packet_size,
                    )
                    paths.update(repairability_paths)
                    repairability_report = {
                        "summary": contract.get("summary"),
                        "paths": {key: str(path) for key, path in repairability_paths.items()},
                    }
                payload = {
                    "status": bundle.manifest.get("status", "PASS"),
                    "mode": "verify_artifact_outputs",
                    "ledger_sha256": bundle.manifest.get("ledger_sha256"),
                    "verified_ledger_sha256": bundle.manifest.get("verified_ledger_sha256"),
                    "line_count": len(bundle.lines),
                    "paths": {key: str(path) for key, path in paths.items()},
                }
                if repairability_report is not None:
                    payload["repairability_contract"] = repairability_report
                return _emit(payload, json_mode=args.json, exit_on_fail=True)
            with _connect(db_path) as conn:
                bundle = load_ledger(out_dir / LEDGER_FILENAME, out_dir / MANIFEST_FILENAME)
                if (
                    args.write_verified
                    or args.validate_decisions
                    or args.decision_template
                    or args.repairability_contract
                ):
                    bundle, verification = verify_bundle(conn, bundle, strict_row_trust=False)
                    paths: dict[str, Path] = {}
                    if args.write_verified:
                        paths = write_ledger_files(bundle, out_dir)
                    decision_report: dict[str, Any] | None = None
                    if args.validate_decisions:
                        decision_report = validate_review_decisions(
                            bundle,
                            Path(args.validate_decisions).expanduser(),
                            allow_partial=args.allow_partial_review,
                        )
                    if args.decision_template:
                        write_decision_template(
                            bundle,
                            out_dir / DECISION_TEMPLATE_FILENAME,
                            max_lines=args.review_packet_size,
                            mode=args.review_packet_mode,
                            scope=args.decision_template_scope,
                        )
                        paths["decision_template"] = out_dir / DECISION_TEMPLATE_FILENAME
                    repairability_report: dict[str, Any] | None = None
                    if args.repairability_contract:
                        queue_manifest = (
                            _load_json_manifest(Path(args.repairability_queue_json).expanduser())
                            if args.repairability_queue_json
                            else None
                        )
                        repairability_output_dir = (
                            Path(args.repairability_output_dir).expanduser()
                            if args.repairability_output_dir
                            else out_dir / "repairability"
                        )
                        contract = build_repairability_contract_from_ledger(bundle, queue_manifest)
                        repairability_paths = write_repairability_contract_files(
                            contract,
                            repairability_output_dir,
                            max_packet_rows=args.review_packet_size,
                        )
                        paths.update(repairability_paths)
                        repairability_report = {
                            "summary": contract.get("summary"),
                            "paths": {key: str(path) for key, path in repairability_paths.items()},
                        }
                    payload = {
                        **verification.to_dict(),
                        "mode": "verify_bundle",
                        "paths": {key: str(path) for key, path in paths.items()},
                    }
                    if decision_report is not None:
                        payload["decision_validation"] = decision_report
                        if decision_report["status"] == "FAIL":
                            payload["status"] = "FAIL"
                    if repairability_report is not None:
                        payload["repairability_contract"] = repairability_report
                    return _emit(payload, json_mode=args.json, exit_on_fail=True)
                verification = verify_ledger(conn, bundle)
            return _emit(verification.to_dict(), json_mode=args.json, exit_on_fail=True)

        with _connect_readonly(db_path) as conn:
            bundle = generate_ledger(
                conn,
                source_db_path=str(db_path.resolve()),
                strong_source_hash=args.strong_source_hash,
            )
            verification: LedgerVerification | None = None
            if args.write_verified or args.decision_template or args.repairability_contract:
                bundle, verification = verify_bundle(conn, bundle, strict_row_trust=False)
        paths = write_ledger_files(bundle, out_dir)
        if args.decision_template:
            write_decision_template(
                bundle,
                out_dir / DECISION_TEMPLATE_FILENAME,
                max_lines=args.review_packet_size,
                mode=args.review_packet_mode,
                scope=args.decision_template_scope,
            )
            paths["decision_template"] = out_dir / DECISION_TEMPLATE_FILENAME
        repairability_report: dict[str, Any] | None = None
        if args.repairability_contract:
            queue_manifest = (
                _load_json_manifest(Path(args.repairability_queue_json).expanduser())
                if args.repairability_queue_json
                else None
            )
            repairability_output_dir = (
                Path(args.repairability_output_dir).expanduser()
                if args.repairability_output_dir
                else out_dir / "repairability"
            )
            contract = build_repairability_contract_from_ledger(bundle, queue_manifest)
            repairability_paths = write_repairability_contract_files(
                contract,
                repairability_output_dir,
                max_packet_rows=args.review_packet_size,
            )
            paths.update(repairability_paths)
            repairability_report = {
                "summary": contract.get("summary"),
                "paths": {key: str(path) for key, path in repairability_paths.items()},
            }
        report = {
            "status": verification.status if verification else "PASS",
            "mode": "generate_verified" if verification else "generate",
            "ledger_sha256": bundle.manifest["ledger_sha256"],
            "verified_ledger_sha256": bundle.manifest.get("verified_ledger_sha256"),
            "line_count": len(bundle.lines),
            "action_counts": bundle.manifest["action_counts"],
            "paths": {key: str(path) for key, path in paths.items()},
        }
        if verification:
            report["verification"] = verification.to_dict()
        if repairability_report is not None:
            report["repairability_contract"] = repairability_report
        return _emit(report, json_mode=args.json)
    except Exception as exc:
        return _emit({"status": "FAIL", "error": str(exc)}, json_mode=args.json, exit_on_fail=True)


def _build_line(line_number: int, audit_row: dict[str, Any], source_snapshot: dict[str, Any]) -> dict[str, Any]:
    action, rationale = _proposed_action(audit_row)
    line: dict[str, Any] = {
        "schema_version": LINE_SCHEMA_VERSION,
        "line_number": line_number,
        "row_id": audit_row["old_id"],
        "old_id": audit_row["old_id"],
        "replacement_id": audit_row.get("replacement_id"),
        "terminal_head_id": audit_row.get("terminal_head_id"),
        "fitness_class": audit_row.get("fitness_class"),
        "structural_class": audit_row.get("structural_class") or "none",
        "structural_severity": audit_row.get("structural_severity") or "none",
        "issue_flags": sorted(audit_row.get("issue_flags") or []),
        "proposed_action": action,
        "action_rationale": rationale,
        "before": _before(audit_row),
        "after": _after(audit_row, action, rationale),
        "source_snapshot": source_snapshot,
        "source_audit_row_hash": _audit_row_hash(audit_row),
        "cross_checks": {
            "audit_replay_validator": "pending",
            "direct_sql_validator": "pending",
        },
        "verifier_status": GENERATED_STATUS,
        "blocking_reasons": [],
    }
    line["ledger_line_hash"] = _line_hash(line)
    return line


def _source_snapshot(
    results: list[dict[str, Any]],
    summary: dict[str, Any],
    source_db_path: str | None,
    generated_at: str,
    *,
    strong_source_hash: bool = False,
) -> dict[str, Any]:
    snapshot = {
        "generated_at": generated_at,
        "source_total_chains": summary.get("total_chains", len(results)),
        "source_old_ids_sha256": _old_ids_sha256(results),
        "audit_schema_version": "supersession_chain_quality.v1",
    }
    snapshot.update(_db_fingerprint(source_db_path, strong_source_hash=strong_source_hash))
    return snapshot


def _build_manifest(
    lines: list[dict[str, Any]],
    audit: dict[str, Any],
    source_snapshot: dict[str, Any],
    generated_at: str,
) -> dict[str, Any]:
    line_ids = [line["old_id"] for line in lines]
    duplicates = sorted(item for item, count in Counter(line_ids).items() if count > 1)
    manifest_without_hash = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "claim_boundary": "Internal supersession repair ledger; not a semantic truth label or public benchmark claim.",
        "generated_at": generated_at,
        "source_snapshot": source_snapshot,
        "source_audit_summary": audit["summary"],
        "source_row_count": len(audit["results"]),
        "ledger_row_count": len(lines),
        "action_counts": dict(sorted(Counter(line["proposed_action"] for line in lines).items())),
        "verifier_status_counts": dict(sorted(Counter(line["verifier_status"] for line in lines).items())),
        "duplicate_old_ids": duplicates,
        "omitted_old_ids": [],
        "line_hashes_sha256": hashlib.sha256(
            "\n".join(line["ledger_line_hash"] for line in lines).encode()
        ).hexdigest(),
    }
    ledger_sha = hashlib.sha256(
        _canonical_json({"manifest": manifest_without_hash, "lines": lines}).encode("utf-8")
    ).hexdigest()
    manifest_without_hash["ledger_sha256"] = ledger_sha
    manifest_without_hash["status"] = "PASS" if not duplicates else "FAIL"
    return manifest_without_hash


def _summary(bundle: LedgerBundle) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": bundle.manifest["status"],
        "ledger_sha256": bundle.manifest["ledger_sha256"],
        "line_count": len(bundle.lines),
        "action_counts": bundle.manifest["action_counts"],
        "source_audit_summary": bundle.manifest["source_audit_summary"],
    }


def build_repairability_contract_from_ledger(
    bundle: LedgerBundle,
    queue_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a read-only operator contract over ledger and optional queue rows."""
    queue_units = _repairability_queue_units(queue_manifest)
    rows = [
        _repairability_contract_row(line, bundle=bundle, queue_units=queue_units)
        for line in bundle.lines
    ]
    contract = {
        "schema_version": REPAIRABILITY_CONTRACT_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "claim_boundary": (
            "Internal supersession repairability contract; "
            "not semantic approval or public benchmark evidence."
        ),
        "ledger_sha256": bundle.manifest.get("ledger_sha256"),
        "verified_ledger_sha256": bundle.manifest.get("verified_ledger_sha256"),
        "source_snapshot": bundle.manifest.get("source_snapshot") or {},
        "queue_summary": (
            queue_manifest.get("summary")
            if isinstance(queue_manifest, dict) and isinstance(queue_manifest.get("summary"), dict)
            else {}
        ),
        "rows": rows,
    }
    contract["summary"] = summarize_repairability_contract(contract)
    return contract


def summarize_repairability_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Return compact metrics for a repairability contract."""
    rows = contract.get("rows") if isinstance(contract.get("rows"), list) else []
    operation_counter: Counter[str] = Counter()
    operation_blockers: Counter[str] = Counter()
    safety_blockers: Counter[str] = Counter()
    ledger_actions: Counter[str] = Counter()
    queue_classes: Counter[str] = Counter()
    approval_states: Counter[str] = Counter()
    matched_queue_rows = 0
    operation_rows = 0
    operation_rows_with_evidence = 0

    for row in rows:
        if not isinstance(row, dict):
            continue
        ledger_actions.update([str(row.get("ledger_action") or "missing")])
        queue_classification = str(row.get("queue_classification") or "missing")
        queue_classes.update([queue_classification])
        approval_states.update([str(row.get("approval_state") or "none")])
        if queue_classification != "queue_not_supplied":
            matched_queue_rows += 1
        for blocker in row.get("operation_blockers") or []:
            operation_blockers.update([str(blocker)])
        for blocker in row.get("safety_blockers") or []:
            safety_blockers.update([str(blocker)])
        operations = row.get("operation_plan") or []
        if operations:
            operation_rows += 1
            if _repairability_operation_evidence_complete(row):
                operation_rows_with_evidence += 1
        for operation in operations:
            if isinstance(operation, dict):
                operation_counter.update([f"{operation.get('table')}.{operation.get('field')}"])

    total_rows = len(rows)
    blocked_count = queue_classes.get(LEDGER_BLOCKED_STRUCTURAL, 0) + queue_classes.get(LEDGER_BLOCKED_VERIFIER, 0)
    review_required_count = queue_classes.get(LEDGER_REVIEW_PROVENANCE, 0) + queue_classes.get(
        LEDGER_REVIEW_SEMANTIC,
        0,
    )
    operation_candidate_count = sum(
        1
        for row in rows
        if isinstance(row, dict)
        and row.get("operation_plan")
        and not row.get("operation_blockers")
        and not row.get("safety_blockers")
    )

    return {
        "schema_version": REPAIRABILITY_CONTRACT_SUMMARY_SCHEMA_VERSION,
        "status": "PASS",
        "claim_boundary": contract.get("claim_boundary"),
        "total_rows": total_rows,
        "ledger_action_counts": dict(sorted(ledger_actions.items())),
        "queue_classification_counts": dict(sorted(queue_classes.items())),
        "approval_state_counts": dict(sorted(approval_states.items())),
        "operation_counts": dict(sorted(operation_counter.items())),
        "operation_blocker_counts": dict(sorted(operation_blockers.items())),
        "safety_blocker_counts": dict(sorted(safety_blockers.items())),
        "contract_row_coverage_rate": 1.0,
        "queue_row_match_rate": (matched_queue_rows / total_rows) if total_rows else 1.0,
        "operation_evidence_completeness_rate": (
            operation_rows_with_evidence / operation_rows if operation_rows else 1.0
        ),
        "action_summary": {
            "mechanical_candidate_count": queue_classes.get(LEDGER_MECHANICAL_CLASSIFICATION, 0),
            "blocked_count": blocked_count,
            "review_required_count": review_required_count,
            "operation_candidate_count": operation_candidate_count,
            "has_live_mutation": False,
        },
    }


def render_repairability_evidence_report(
    contract: dict[str, Any],
    *,
    mechanical_candidate_template_path: Path | None = None,
) -> str:
    """Render compact repairability evidence without row payloads."""
    summary = contract.get("summary") if isinstance(contract.get("summary"), dict) else {}
    source_snapshot = contract.get("source_snapshot") if isinstance(contract.get("source_snapshot"), dict) else {}
    action_summary = summary.get("action_summary") if isinstance(summary.get("action_summary"), dict) else {}
    template_path = str(mechanical_candidate_template_path) if mechanical_candidate_template_path else "not written"
    lines = [
        "# Supersession Repairability Evidence Report",
        "",
        f"Generated: `{contract.get('generated_at') or 'unknown'}`",
        f"Claim boundary: {contract.get('claim_boundary')}",
        f"Ledger SHA-256: `{contract.get('ledger_sha256') or 'missing'}`",
        f"Verified ledger SHA-256: `{contract.get('verified_ledger_sha256') or 'missing'}`",
        f"Source DB path: `{source_snapshot.get('source_db_path') or 'missing'}`",
        f"Source DB SHA-256: `{source_snapshot.get('source_db_sha256') or 'missing'}`",
        f"Source old IDs SHA-256: `{source_snapshot.get('source_old_ids_sha256') or 'missing'}`",
        "",
        "## Action Summary",
        "",
        f"- No live changes applied: `{not bool(action_summary.get('has_live_mutation'))}`",
        f"- Total rows: {summary.get('total_rows', 0)}",
        f"- Mechanical candidates requiring approval: {action_summary.get('mechanical_candidate_count', 0)}",
        f"- Blocked rows: {action_summary.get('blocked_count', 0)}",
        f"- Rows requiring human review: {action_summary.get('review_required_count', 0)}",
        f"- Operation candidates: {action_summary.get('operation_candidate_count', 0)}",
        f"- Mechanical candidate decision template: `{template_path}`",
        "",
        "## Metrics",
        "",
        f"- Queue classifications: `{json.dumps(summary.get('queue_classification_counts') or {}, sort_keys=True)}`",
        f"- Approval states: `{json.dumps(summary.get('approval_state_counts') or {}, sort_keys=True)}`",
        f"- Operation counts: `{json.dumps(summary.get('operation_counts') or {}, sort_keys=True)}`",
        f"- Operation blockers: `{json.dumps(summary.get('operation_blocker_counts') or {}, sort_keys=True)}`",
        f"- Safety blockers: `{json.dumps(summary.get('safety_blocker_counts') or {}, sort_keys=True)}`",
        f"- Operation evidence completeness: {summary.get('operation_evidence_completeness_rate')}",
        "",
    ]
    return "\n".join(lines)


def render_repairability_operator_packet(
    contract: dict[str, Any],
    *,
    max_rows: int = 200,
) -> str:
    """Render a compact human-readable packet for repairability review."""
    if max_rows < 1:
        raise ValueError("max_rows must be >= 1")
    summary = contract.get("summary") if isinstance(contract.get("summary"), dict) else {}
    action_summary = summary.get("action_summary") if isinstance(summary.get("action_summary"), dict) else {}
    rows = contract.get("rows") if isinstance(contract.get("rows"), list) else []
    lines = [
        "# Supersession Repairability Operator Packet",
        "",
        f"Generated: `{contract.get('generated_at') or 'unknown'}`",
        "",
        "## Action Summary",
        "",
        "- No live changes have been applied.",
        f"- Total rows: {summary.get('total_rows', 0)}",
        f"- Mechanical candidates requiring explicit approval: {action_summary.get('mechanical_candidate_count', 0)}",
        f"- Rows requiring human review: {action_summary.get('review_required_count', 0)}",
        f"- Blocked rows: {action_summary.get('blocked_count', 0)}",
        f"- Operation candidates: {action_summary.get('operation_candidate_count', 0)}",
        f"- Review template: `{MECHANICAL_CANDIDATE_DECISION_TEMPLATE_FILENAME}`",
        "",
        "## Evidence Boundary",
        "",
        f"Claim boundary: {contract.get('claim_boundary')}",
        f"Ledger SHA-256: `{contract.get('ledger_sha256') or 'missing'}`",
        f"Verified ledger SHA-256: `{contract.get('verified_ledger_sha256') or 'missing'}`",
        "",
        "## Summary",
        "",
        f"- Total rows: {summary.get('total_rows', 0)}",
        f"- Ledger actions: `{json.dumps(summary.get('ledger_action_counts') or {}, sort_keys=True)}`",
        f"- Queue classifications: `{json.dumps(summary.get('queue_classification_counts') or {}, sort_keys=True)}`",
        f"- Approval states: `{json.dumps(summary.get('approval_state_counts') or {}, sort_keys=True)}`",
        f"- Operation blockers: `{json.dumps(summary.get('operation_blocker_counts') or {}, sort_keys=True)}`",
        f"- Operation evidence completeness: {summary.get('operation_evidence_completeness_rate')}",
        "",
        f"Showing first {min(len(rows), max_rows)} rows by operator priority.",
        "",
    ]
    for row in sorted((row for row in rows if isinstance(row, dict)), key=_repairability_packet_sort_key)[:max_rows]:
        lines.extend(_repairability_packet_row(row))
    return "\n".join(lines) + "\n"


def write_mechanical_candidate_decision_template(contract: dict[str, Any], path: Path) -> None:
    """Write a blank, hash-bound decision template for mechanical repair candidates."""
    rows = contract.get("rows") if isinstance(contract.get("rows"), list) else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DECISION_TEMPLATE_FIELDS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            if not _is_mechanical_candidate_template_row(row):
                continue
            evidence = row.get("source_evidence") if isinstance(row.get("source_evidence"), dict) else {}
            writer.writerow(
                {
                    "line_number": row.get("line_number"),
                    "old_id": row.get("old_id"),
                    "verified_line_hash": evidence.get("verified_line_hash"),
                    "verified_ledger_sha256": evidence.get("verified_ledger_sha256"),
                    "source_old_ids_sha256": evidence.get("source_old_ids_sha256"),
                    "review_decision": "",
                    "reviewer": "",
                    "reviewed_at": "",
                    "reviewer_notes": "",
                }
            )


def _is_mechanical_candidate_template_row(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    return (
        row.get("queue_classification") == LEDGER_MECHANICAL_CLASSIFICATION
        and bool(row.get("operation_plan"))
        and not row.get("operation_blockers")
        and not row.get("safety_blockers")
    )


def write_repairability_contract_files(
    contract: dict[str, Any],
    output_dir: Path,
    *,
    max_packet_rows: int = 200,
) -> dict[str, Path]:
    """Write full contract, compact summary, and operator packet artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "repairability_contract": output_dir / REPAIRABILITY_CONTRACT_FILENAME,
        "repairability_summary": output_dir / REPAIRABILITY_CONTRACT_SUMMARY_FILENAME,
        "repairability_operator_packet": output_dir / REPAIRABILITY_OPERATOR_PACKET_FILENAME,
        "repairability_evidence_report": output_dir / REPAIRABILITY_EVIDENCE_REPORT_FILENAME,
        "mechanical_candidate_decision_template": output_dir / MECHANICAL_CANDIDATE_DECISION_TEMPLATE_FILENAME,
    }
    paths["repairability_contract"].write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["repairability_summary"].write_text(
        json.dumps(contract.get("summary") or {}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["repairability_operator_packet"].write_text(
        render_repairability_operator_packet(contract, max_rows=max_packet_rows),
        encoding="utf-8",
    )
    write_mechanical_candidate_decision_template(contract, paths["mechanical_candidate_decision_template"])
    paths["repairability_evidence_report"].write_text(
        render_repairability_evidence_report(
            contract,
            mechanical_candidate_template_path=paths["mechanical_candidate_decision_template"],
        ),
        encoding="utf-8",
    )
    return paths


def _repairability_queue_units(queue_manifest: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(queue_manifest, dict):
        return {}
    units = queue_manifest.get("work_units")
    if not isinstance(units, list):
        return {}
    by_key: dict[str, dict[str, Any]] = {}
    for unit in units:
        if not isinstance(unit, dict):
            continue
        row_number = unit.get("row_number")
        if row_number is not None:
            by_key[f"row:{row_number}"] = unit
        chain_key = unit.get("chain_key") if isinstance(unit.get("chain_key"), dict) else {}
        old_id = chain_key.get("old_id")
        if old_id:
            by_key[f"old:{old_id}"] = unit
    return by_key


def _repairability_contract_row(
    line: dict[str, Any],
    *,
    bundle: LedgerBundle,
    queue_units: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    unit = queue_units.get(f"row:{line.get('line_number')}") or queue_units.get(f"old:{line.get('old_id')}")
    operation_blockers, safety_blockers = _repairability_blockers(unit)
    before = line.get("before") if isinstance(line.get("before"), dict) else {}
    source_snapshot = (
        bundle.manifest.get("source_snapshot")
        if isinstance(bundle.manifest.get("source_snapshot"), dict)
        else {}
    )
    row = {
        "schema_version": REPAIRABILITY_CONTRACT_ROW_SCHEMA_VERSION,
        "line_number": line.get("line_number"),
        "old_id": line.get("old_id"),
        "replacement_id": line.get("replacement_id"),
        "terminal_head_id": line.get("terminal_head_id"),
        "structural_class": line.get("structural_class"),
        "structural_severity": line.get("structural_severity"),
        "ledger_action": line.get("proposed_action"),
        "ledger_reason": line.get("action_rationale"),
        "queue_classification": unit.get("classification") if unit else "queue_not_supplied",
        "issue_flags": sorted(line.get("issue_flags") or []),
        "operation_plan": unit.get("proposed_operations") if unit else [],
        "operation_blockers": operation_blockers,
        "safety_blockers": safety_blockers,
        "approval_state": _repairability_approval_state(unit, line, bundle),
        "source_evidence": {
            "ledger_line_hash": line.get("ledger_line_hash"),
            "verified_line_hash": line.get("verified_line_hash"),
            "verified_ledger_sha256": bundle.manifest.get("verified_ledger_sha256"),
            "source_old_ids_sha256": source_snapshot.get("source_old_ids_sha256"),
            "source_db_path": source_snapshot.get("source_db_path"),
            "old_created_at": before.get("old_created_at"),
            "replacement_created_at": before.get("replacement_created_at"),
            "terminal_created_at": before.get("terminal_created_at"),
            "old_subject_key": before.get("old_subject_key"),
            "replacement_subject_key": before.get("replacement_subject_key"),
        },
    }
    return row


def _repairability_blockers(unit: dict[str, Any] | None) -> tuple[list[str], list[str]]:
    if not unit:
        return [], []
    operation_blockers = {str(item) for item in unit.get("operation_blockers") or []}
    safety_blockers: set[str] = set()
    for blocker in unit.get("safety_blockers") or []:
        blocker_text = str(blocker)
        if blocker_text in OPERATION_BLOCKER_FIELDS:
            operation_blockers.add(blocker_text)
        else:
            safety_blockers.add(blocker_text)
    return sorted(operation_blockers), sorted(safety_blockers)


def _repairability_approval_state(
    unit: dict[str, Any] | None,
    line: dict[str, Any],
    bundle: LedgerBundle,
) -> str:
    if not unit:
        return "none"
    decision = str(unit.get("review_decision") or "").strip()
    evidence = unit.get("source_evidence") if isinstance(unit.get("source_evidence"), dict) else {}
    reviewer = str(evidence.get("reviewer") or "").strip().lower()
    reviewed_at = str(evidence.get("reviewed_at") or "").strip()
    if not decision:
        return "none"
    if reviewer and any(marker in reviewer for marker in DISALLOWED_REVIEWER_MARKERS):
        return "diagnostic_synthetic"
    if not reviewer or not reviewed_at:
        return "none"
    hashes_match = (
        evidence.get("verified_ledger_sha256") == bundle.manifest.get("verified_ledger_sha256")
        and evidence.get("verified_line_hash") == line.get("verified_line_hash")
    )
    if decision == "approve_repair" and hashes_match:
        return "hash_pinned_approved"
    return "operator_reviewed"


def _repairability_operation_evidence_complete(row: dict[str, Any]) -> bool:
    evidence = row.get("source_evidence") if isinstance(row.get("source_evidence"), dict) else {}
    if not evidence.get("verified_ledger_sha256") or not evidence.get("verified_line_hash"):
        return False
    for operation in row.get("operation_plan") or []:
        if not isinstance(operation, dict):
            return False
        if not operation.get("table") or not operation.get("concept_id") or not operation.get("field"):
            return False
    return True


def _repairability_packet_sort_key(row: dict[str, Any]) -> tuple[int, int]:
    classification = row.get("queue_classification")
    if classification == LEDGER_MECHANICAL_CLASSIFICATION:
        priority = 0
    elif row.get("operation_blockers"):
        priority = 1
    elif row.get("safety_blockers"):
        priority = 2
    elif classification == "queue_not_supplied":
        priority = 3
    else:
        priority = 4
    return (priority, int(row.get("line_number") or 0))


def _repairability_packet_row(row: dict[str, Any]) -> list[str]:
    return [
        f"## {row.get('line_number')}. `{row.get('old_id')}`",
        "",
        f"- Ledger action: `{row.get('ledger_action')}`",
        f"- Queue classification: `{row.get('queue_classification')}`",
        f"- Approval state: `{row.get('approval_state')}`",
        f"- Replacement: `{row.get('replacement_id') or 'missing'}`",
        f"- Issues: {', '.join(row.get('issue_flags') or []) or 'none'}",
        f"- Operations: `{json.dumps(row.get('operation_plan') or [], sort_keys=True)}`",
        f"- Operation blockers: {', '.join(row.get('operation_blockers') or []) or 'none'}",
        f"- Safety blockers: {', '.join(row.get('safety_blockers') or []) or 'none'}",
        f"- Evidence: `{json.dumps(row.get('source_evidence') or {}, sort_keys=True)}`",
        "",
    ]


def _proposed_action(audit_row: dict[str, Any]) -> tuple[str, str]:
    flags = set(audit_row.get("issue_flags") or [])
    if audit_row.get("structural_class") == "orphan_tombstone":
        return "no_action_historical_orphan", "safe legacy orphan tombstone; no repair required"
    if flags & STRUCTURAL_FLAGS:
        return "exclude_structural", "structural issue blocks automatic repair"
    if flags and flags <= (LIFECYCLE_FLAGS | PARITY_FLAGS):
        return "repair_lifecycle_metadata", "lifecycle/parity-only issue is eligible for copy-DB repair"
    if flags & (SEMANTIC_FLAGS | PROVENANCE_FLAGS):
        return "review_only", "semantic or provenance uncertainty requires human review"
    return "no_action", "no repair issue detected"


def _select_packet_lines(lines: list[dict[str, Any]], *, max_lines: int, mode: str) -> list[dict[str, Any]]:
    if max_lines < 1:
        raise ValueError("--review-packet-size must be >= 1")
    if mode == "first-n":
        return lines[:max_lines]
    if mode == "severity-first":
        return sorted(lines, key=_packet_severity_key)[:max_lines]

    groups = [
        [line for line in lines if line.get("verifier_status") == BLOCKED_STATUS],
        [line for line in lines if set(line.get("issue_flags") or []) & STRUCTURAL_FLAGS],
        [line for line in lines if set(line.get("issue_flags") or []) & SEMANTIC_FLAGS],
        [line for line in lines if set(line.get("issue_flags") or []) & PROVENANCE_FLAGS],
        [line for line in lines if line.get("proposed_action") == "repair_lifecycle_metadata"],
        [line for line in lines if line.get("proposed_action") == "no_action"],
        [line for line in lines if line.get("proposed_action") == "no_action_historical_orphan"],
    ]
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    while len(selected) < max_lines:
        added = False
        for group in groups:
            while group and str(group[0].get("old_id")) in seen:
                group.pop(0)
            if group:
                item = group.pop(0)
                selected.append(item)
                seen.add(str(item.get("old_id")))
                added = True
                if len(selected) >= max_lines:
                    break
        if not added:
            break
    if len(selected) < max_lines:
        for item in lines:
            old_id = str(item.get("old_id"))
            if old_id not in seen:
                selected.append(item)
                seen.add(old_id)
                if len(selected) >= max_lines:
                    break
    return selected


def _packet_severity_key(line: dict[str, Any]) -> tuple[int, int]:
    flags = set(line.get("issue_flags") or [])
    if line.get("verifier_status") == BLOCKED_STATUS:
        severity = 0
    elif line.get("proposed_action") == "no_action_historical_orphan":
        severity = 5
    elif flags & STRUCTURAL_FLAGS:
        severity = 1
    elif flags & (SEMANTIC_FLAGS | PROVENANCE_FLAGS):
        severity = 2
    elif line.get("proposed_action") == "repair_lifecycle_metadata":
        severity = 3
    else:
        severity = 4
    return (severity, int(line.get("line_number") or 0))


def _packet_line_markdown(item: dict[str, Any], verified_sha: str) -> list[str]:
    return [
        f"## {item['line_number']}. `{item['old_id']}`",
        "",
        f"- Proposed action: `{item['proposed_action']}`",
        f"- Verifier status: `{item.get('verifier_status')}`",
        f"- Blocking reasons: {', '.join(item.get('blocking_reasons') or []) or 'none'}",
        f"- Verified line hash: `{item.get('verified_line_hash')}`",
        f"- Verified ledger hash: `{verified_sha}`",
        f"- Replacement: `{item.get('replacement_id') or 'missing'}`",
        f"- Terminal head: `{item.get('terminal_head_id') or 'missing'}`",
        f"- Structural class: `{item.get('structural_class') or 'none'}` ({item.get('structural_severity') or 'none'})",
        f"- Issues: {', '.join(item.get('issue_flags') or []) or 'none'}",
        f"- Old timestamp: `{(item.get('before') or {}).get('old_created_at') or 'missing'}`",
        f"- Replacement timestamp: `{(item.get('before') or {}).get('replacement_created_at') or 'missing'}`",
        "",
        "**Old excerpt**",
        "",
        f"> {(item.get('before') or {}).get('old_summary') or '[missing]'}",
        "",
        "**Replacement excerpt**",
        "",
        f"> {(item.get('before') or {}).get('replacement_summary') or '[missing]'}",
        "",
        "**Your QC:** approve_repair / reject_repair / needs_context / not_supersession / uncertain",
        "",
    ]


def _before(audit_row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "old_status",
        "old_currency_status",
        "old_created_at",
        "replacement_created_at",
        "terminal_created_at",
        "superseded_at",
        "supersession_reason",
        "old_subject_key",
        "replacement_subject_key",
        "old_summary",
        "replacement_summary",
        "terminal_summary",
    ]
    return {key: audit_row.get(key) for key in keys}


def _after(audit_row: dict[str, Any], action: str, rationale: str) -> dict[str, Any]:
    if action != "repair_lifecycle_metadata":
        return dict(_before(audit_row))
    return {
        **_before(audit_row),
        "old_status": "superseded",
        "old_currency_status": "SUPERSEDED",
        "superseded_by": audit_row.get("replacement_id"),
        "supersession_reason": audit_row.get("supersession_reason") or rationale,
    }


def _audit_replay_disagreements(line: dict[str, Any], audit_row: dict[str, Any] | None) -> list[str]:
    if not audit_row:
        return []
    checks = {
        "replacement_id": audit_row.get("replacement_id"),
        "terminal_head_id": audit_row.get("terminal_head_id"),
        "fitness_class": audit_row.get("fitness_class"),
        "structural_class": audit_row.get("structural_class") or "none",
        "structural_severity": audit_row.get("structural_severity") or "none",
        "issue_flags": sorted(audit_row.get("issue_flags") or []),
    }
    return [f"audit_replay_{key}_mismatch" for key, expected in checks.items() if line.get(key) != expected]


def _direct_sql_disagreements(conn: sqlite3.Connection, line: dict[str, Any]) -> list[str]:
    old_id = str(line.get("old_id") or "")
    row = _fetch_one(conn, "SELECT * FROM concepts WHERE id = ?", (old_id,))
    if not row:
        return ["direct_sql_old_missing"]
    failures: list[str] = []
    if row.get("superseded_by") != line.get("replacement_id"):
        failures.append("direct_sql_replacement_mismatch")
    data = _safe_json(row.get("data"))
    if isinstance(data, dict):
        if data.get("status") is not None and data.get("status") != row.get("status"):
            failures.append("direct_sql_json_status_mismatch")
        if data.get("currency_status") is not None and data.get("currency_status") != row.get("currency_status"):
            failures.append("direct_sql_json_currency_mismatch")
        if data.get("superseded_by") is not None and data.get("superseded_by") != row.get("superseded_by"):
            failures.append("direct_sql_json_superseded_by_mismatch")
    replacement_id = line.get("replacement_id")
    issue_flags = set(line.get("issue_flags") or [])
    if (
        replacement_id
        and "broken_pointer" not in issue_flags
        and not _fetch_one(conn, "SELECT id FROM concepts WHERE id = ?", (replacement_id,))
    ):
        failures.append("direct_sql_replacement_missing")
    return failures


def _line_cross_checks(hard_failures: list[str], row_failures: list[str]) -> dict[str, str]:
    audit_failures = [item for item in hard_failures + row_failures if item.startswith("audit_replay_")]
    direct_failures = [item for item in row_failures if item.startswith("direct_sql_")]
    integrity_failures = [item for item in hard_failures if not item.startswith("audit_replay_")]
    return {
        "audit_replay_validator": BLOCKED_STATUS if audit_failures else PASSED_STATUS,
        "direct_sql_validator": BLOCKED_STATUS if direct_failures else PASSED_STATUS,
        "ledger_integrity_validator": BLOCKED_STATUS if integrity_failures else PASSED_STATUS,
    }


def _verified_manifest(
    manifest: dict[str, Any],
    verified_lines: list[dict[str, Any]],
    verification: LedgerVerification,
) -> dict[str, Any]:
    verified = dict(manifest)
    verified["status"] = verification.status
    verified["verification"] = verification.to_dict()
    verified["verifier_status_counts"] = verification.verifier_status_counts
    verified["blocked_count"] = verification.blocked_count
    verified["hard_integrity_fail_count"] = len(verification.hard_integrity_failures or [])
    verified["row_trust_blocker_count"] = len(verification.row_trust_failures or [])
    verified["verified_line_hashes_sha256"] = hashlib.sha256(
        "\n".join(str(line.get("verified_line_hash") or "") for line in verified_lines).encode("utf-8")
    ).hexdigest()
    without_verified_hash = dict(verified)
    without_verified_hash.pop("verified_ledger_sha256", None)
    verified["verified_ledger_sha256"] = hashlib.sha256(
        _canonical_json({"manifest": without_verified_hash, "lines": verified_lines}).encode("utf-8")
    ).hexdigest()
    return verified


def _db_fingerprint(source_db_path: str | None, *, strong_source_hash: bool = False) -> dict[str, Any]:
    fingerprint: dict[str, Any] = {
        "source_db_path": None,
        "source_db_size": None,
        "source_db_mtime_ns": None,
    }
    if not source_db_path:
        return fingerprint
    path = Path(source_db_path).expanduser()
    try:
        resolved = path.resolve()
        stat = resolved.stat()
    except OSError:
        fingerprint["source_db_path"] = str(path)
        return fingerprint
    fingerprint.update(
        {
            "source_db_path": str(resolved),
            "source_db_size": stat.st_size,
            "source_db_mtime_ns": stat.st_mtime_ns,
        }
    )
    if strong_source_hash:
        fingerprint["source_db_sha256"] = hashlib.sha256(resolved.read_bytes()).hexdigest()
    return fingerprint


def _snapshot_matches(left_manifest: dict[str, Any], right_manifest: dict[str, Any]) -> bool:
    left = left_manifest.get("source_snapshot") or {}
    right = right_manifest.get("source_snapshot") or {}
    return left.get("source_total_chains") == right.get("source_total_chains") and left.get(
        "source_old_ids_sha256"
    ) == right.get("source_old_ids_sha256")


def _apply_report(
    status: str,
    mode: str,
    bundle: LedgerBundle,
    verification: LedgerVerification,
    copy_snapshot_parity: bool,
    changed_rows: int,
    after_bundle: LedgerBundle | None,
) -> dict[str, Any]:
    return {
        "schema_version": APPLY_REPORT_SCHEMA_VERSION,
        "status": status,
        "mode": mode,
        "ledger_sha256": bundle.manifest.get("ledger_sha256"),
        "line_count": len(bundle.lines),
        "action_counts": bundle.manifest.get("action_counts", {}),
        "verification": verification.to_dict(),
        "copy_snapshot_parity": copy_snapshot_parity,
        "changed_rows": changed_rows,
        "before": bundle.manifest.get("source_audit_summary"),
        "after": (after_bundle.manifest.get("source_audit_summary") if after_bundle else None),
    }


def _approved_mechanical_apply_report(
    status: str,
    bundle: LedgerBundle,
    decision_report: dict[str, Any],
    required_line_set_report: dict[str, Any],
    queue_manifest: dict[str, Any],
    apply_units: list[dict[str, Any]],
    *,
    changed_rows: int,
    target_db_path: Path | None,
    scoped_verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = queue_manifest.get("summary") if isinstance(queue_manifest.get("summary"), dict) else {}
    return {
        "schema_version": APPROVED_MECHANICAL_APPLY_REPORT_SCHEMA_VERSION,
        "status": status,
        "mode": "approved_mechanical_apply",
        "operator_surface": "approved_repair",
        "legacy_operator_surface": "approved_mechanical",
        "claim_boundary": "Internal copied-DB reviewed repair evidence; not semantic authority approval.",
        "ledger_sha256": bundle.manifest.get("ledger_sha256"),
        "verified_ledger_sha256": bundle.manifest.get("verified_ledger_sha256"),
        "decision_validation_status": decision_report.get("status"),
        "decision_counts": decision_report.get("decision_counts", {}),
        "required_line_set_validation": required_line_set_report,
        "queue_summary": summary,
        "eligible_line_count": len(apply_units),
        "changed_rows": changed_rows,
        "target_db_path": str(target_db_path) if target_db_path else None,
        "applied_work_unit_ids": [unit.get("work_unit_id") for unit in apply_units],
        "scoped_verification": scoped_verification,
    }


def _live_apply_report(
    status: str,
    bundle: LedgerBundle,
    decision_report: dict[str, Any],
    required_line_set_report: dict[str, Any],
    queue_manifest: dict[str, Any],
    apply_units: list[dict[str, Any]],
    *,
    changed_rows: int,
    committed: bool,
    source_db_path: Path,
    evidence_dir: Path,
    expected_token: str,
    preflight: dict[str, Any],
    backup_report: dict[str, Any] | None,
    before_metrics: dict[str, int],
    after_metrics: dict[str, int],
    post_checks: dict[str, Any],
    rollback_runbook_path: Path | None,
    scoped_verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = queue_manifest.get("summary") if isinstance(queue_manifest.get("summary"), dict) else {}
    return {
        "schema_version": LIVE_APPROVED_MECHANICAL_APPLY_REPORT_SCHEMA_VERSION,
        "status": status,
        "mode": "live_approved_mechanical_apply",
        "operator_surface": "approved_repair_live",
        "legacy_operator_surface": "approved_mechanical_live",
        "claim_boundary": "Internal live reviewed repair evidence; not semantic or implicit supersession approval.",
        "ledger_sha256": bundle.manifest.get("ledger_sha256"),
        "verified_ledger_sha256": bundle.manifest.get("verified_ledger_sha256"),
        "decision_validation_status": decision_report.get("status"),
        "decision_counts": decision_report.get("decision_counts", {}),
        "required_line_set_validation": required_line_set_report,
        "queue_summary": summary,
        "eligible_line_count": len(apply_units),
        "changed_rows": changed_rows,
        "committed": committed,
        "source_db_path": str(source_db_path),
        "evidence_dir": str(evidence_dir),
        "expected_operator_approval_token": expected_token,
        "preflight": preflight,
        "backup": backup_report,
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
        "post_checks": post_checks,
        "rollback_runbook_path": str(rollback_runbook_path) if rollback_runbook_path else None,
        "applied_work_unit_ids": [unit.get("work_unit_id") for unit in apply_units],
        "scoped_verification": scoped_verification,
    }


def _write_apply_report(out_dir: Path, report: dict[str, Any]) -> None:
    (out_dir / APPLY_REPORT_FILENAME).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_approved_mechanical_apply_report(out_dir: Path, report: dict[str, Any]) -> None:
    (out_dir / APPROVED_MECHANICAL_APPLY_REPORT_FILENAME).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_live_apply_preflight_report(evidence_dir: Path, report: dict[str, Any]) -> None:
    (evidence_dir / LIVE_APPROVED_MECHANICAL_PREFLIGHT_REPORT_FILENAME).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_live_apply_report(evidence_dir: Path, report: dict[str, Any]) -> None:
    (evidence_dir / LIVE_APPROVED_MECHANICAL_APPLY_REPORT_FILENAME).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sqlite_backup(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    source_conn = sqlite3.connect(f"file:{source.resolve()}?mode=ro", uri=True)
    try:
        dest_conn = sqlite3.connect(str(dest))
        try:
            source_conn.backup(dest_conn)
        finally:
            dest_conn.close()
    finally:
        source_conn.close()


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def _connect_readonly(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _live_apply_approval_token(
    verified_ledger_sha256: str,
    decisions_sha256: str,
    queue_sha256: str,
    eligible_count: int,
) -> str:
    return f"MAINT-071:APPLY:{verified_ledger_sha256}:{decisions_sha256}:{queue_sha256}:{eligible_count}"


def _curated_live_apply_approval_token(curated_manifest_sha256: str, eligible_count: int) -> str:
    return f"MAINT-082:APPLY:{curated_manifest_sha256}:{eligible_count}"


def _live_apply_evidence_dir(evidence_root: Path | None) -> Path:
    if evidence_root is not None:
        return evidence_root.expanduser()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path.home() / ".pith" / "reports" / "maintenance" / "supersession_live_repair_apply" / timestamp


def _default_live_apply_backup_path(source_db_path: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return source_db_path.parent / "snapshots" / f"pith_{timestamp}_supersession-live-repair-preapply.db"


def _sqlite_integrity_check(path: Path) -> str:
    with sqlite3.connect(str(path)) as conn:
        row = conn.execute("PRAGMA integrity_check").fetchone()
    return str(row[0] if row else "")


def _probe_sqlite_write_lock(path: Path) -> tuple[bool, str]:
    conn = sqlite3.connect(str(path), timeout=2)
    conn.execute("PRAGMA busy_timeout=2000")
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("ROLLBACK")
        return True, "Write lock acquired successfully"
    except sqlite3.OperationalError as exc:
        return False, f"Lock contention: {exc}"
    finally:
        conn.close()


def _active_db_holders(path: Path) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["lsof", str(path)],
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"count": -1, "holders": [], "error": str(exc)}
    lines = [line for line in completed.stdout.splitlines()[1:] if line.strip()]
    current_pid = str(os.getpid())
    holders = [line for line in lines if current_pid not in line.split()]
    return {"count": len(holders), "holders": holders, "error": None}


def _db_file_report(path: Path, *, strong_hash: bool) -> dict[str, Any]:
    report: dict[str, Any] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
    }
    if strong_hash:
        report["sha256"] = _file_sha256(path)
    return report


def _live_db_metrics(path: Path) -> dict[str, int]:
    with _connect_readonly(path) as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS concepts,
                COUNT(subject_key) AS subject_keys,
                COUNT(supersession_reason) AS supersession_reasons,
                COUNT(superseded_at) AS superseded_timestamps,
                SUM(CASE WHEN status = 'superseded' AND currency_status != 'SUPERSEDED' THEN 1 ELSE 0 END)
                    AS split_superseded_currency,
                SUM(CASE WHEN superseded_by IS NOT NULL AND superseded_by != '' THEN 1 ELSE 0 END) AS superseded_edges
            FROM concepts
            """
        ).fetchone()
    return {key: int(row[key] or 0) for key in row.keys()}


def _fetch_concept_repair_row(conn: sqlite3.Connection, concept_id: str) -> dict[str, Any]:
    fields = [
        "id",
        "status",
        "currency_status",
        "superseded_by",
        "supersession_reason",
        "superseded_at",
        "subject_key",
        "is_current",
        "data",
    ]
    fields.extend(field for field in PRESERVED_EDGE_REPAIR_FIELDS if field not in fields and _table_has_column(conn, "concepts", field))
    row = _fetch_one(
        conn,
        f"SELECT {', '.join(fields)} FROM concepts WHERE id = ?",
        (concept_id,),
    )
    if not row:
        raise ValueError(f"concept not found:{concept_id}")
    return row


def _live_apply_row_evidence(
    unit: dict[str, Any],
    before_row: dict[str, Any],
    after_row: dict[str, Any],
) -> dict[str, Any]:
    return {
        "work_unit_id": unit.get("work_unit_id"),
        "row_number": unit.get("row_number"),
        "concept_id": (unit.get("chain_key") or {}).get("old_id"),
        "source_evidence": unit.get("source_evidence"),
        "proposed_operations": unit.get("proposed_operations") or [],
        "before": before_row,
        "after": after_row,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _load_json_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON manifest must be an object: {path}")
    return payload


def _curated_repair_apply_plan(
    *,
    curated_manifest_path: Path,
    approved_curated_manifest_sha256: str,
) -> tuple[LedgerBundle, dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if not HASH_RE.fullmatch(approved_curated_manifest_sha256):
        raise ValueError("approved_curated_manifest_sha256 must be a 64-character lowercase hex digest")
    if _file_sha256(curated_manifest_path) != approved_curated_manifest_sha256:
        raise ValueError("approved curated repair manifest hash mismatch")
    manifest = _load_json_manifest(curated_manifest_path)
    if manifest.get("schema_version") != CURATED_REPAIR_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported curated repair manifest schema_version")
    reviewer = str(manifest.get("reviewer") or "").strip()
    reviewed_at = str(manifest.get("reviewed_at") or "").strip()
    if not reviewer:
        raise ValueError("curated repair manifest reviewer is required")
    if any(marker in reviewer.lower() for marker in DISALLOWED_REVIEWER_MARKERS):
        raise ValueError(f"synthetic reviewer rejected:{reviewer.lower()}")
    if not reviewed_at:
        raise ValueError("curated repair manifest reviewed_at is required")

    raw_units = manifest.get("work_units")
    if not isinstance(raw_units, list) or not raw_units:
        raise ValueError("curated repair manifest requires non-empty work_units")
    apply_units = [
        _curated_apply_unit(raw_unit, manifest, approved_curated_manifest_sha256)
        for raw_unit in raw_units
    ]
    work_unit_ids = [str(unit.get("work_unit_id") or "") for unit in apply_units]
    old_ids = [str((unit.get("chain_key") or {}).get("old_id") or "") for unit in apply_units]
    duplicate_work_unit_ids = sorted(item for item, count in Counter(work_unit_ids).items() if item and count > 1)
    duplicate_old_ids = sorted(item for item, count in Counter(old_ids).items() if item and count > 1)
    if duplicate_work_unit_ids:
        raise ValueError(f"duplicate curated repair work_unit_id:{','.join(duplicate_work_unit_ids)}")
    if duplicate_old_ids:
        raise ValueError(f"duplicate curated repair old_id:{','.join(duplicate_old_ids)}")
    _validate_apply_units(apply_units)
    summary = _curated_manifest_summary(apply_units)
    manifest = dict(manifest)
    manifest["summary"] = summary
    manifest["apply_enabled"] = False
    manifest["curated_manifest_sha256"] = approved_curated_manifest_sha256
    bundle = LedgerBundle(
        lines=[],
        manifest={
            "schema_version": CURATED_REPAIR_MANIFEST_SCHEMA_VERSION,
            "status": "PASS",
            "ledger_sha256": approved_curated_manifest_sha256,
            "verified_ledger_sha256": approved_curated_manifest_sha256,
            "source_snapshot": manifest.get("source_snapshot") or {},
            "curated_manifest_sha256": approved_curated_manifest_sha256,
        },
        source_audit={},
    )
    decision_report = {
        "status": "PASS",
        "decision_counts": {"approve_repair": len(apply_units)},
        "rows_reviewed": len(apply_units),
        "ledger_row_count": len(apply_units),
        "missing_review_count": 0,
        "partial": False,
    }
    return bundle, decision_report, _not_required_line_set_report(), manifest, apply_units


def _curated_apply_unit(
    raw_unit: Any,
    manifest: dict[str, Any],
    manifest_sha256: str,
) -> dict[str, Any]:
    if not isinstance(raw_unit, dict):
        raise ValueError("curated repair work unit must be an object")
    unit = json.loads(json.dumps(raw_unit))
    work_unit_id = str(unit.get("work_unit_id") or "").strip()
    if not work_unit_id:
        raise ValueError("curated repair work_unit_id is required")
    if unit.get("classification") != "curated_normalize_supersession_chain":
        raise ValueError(f"unsupported curated repair classification:{unit.get('classification')}")
    unit["review_decision"] = unit.get("review_decision") or "approve_repair"
    if unit["review_decision"] != "approve_repair":
        raise ValueError(f"non-approved curated unit selected:{work_unit_id}")
    if unit.get("safety_blockers") or unit.get("operation_blockers"):
        raise ValueError(f"curated repair unit has blockers:{work_unit_id}")
    chain_key = unit.get("chain_key") if isinstance(unit.get("chain_key"), dict) else {}
    old_id = str(chain_key.get("old_id") or "").strip()
    if not old_id:
        raise ValueError(f"curated repair unit old_id is required:{work_unit_id}")
    operations = unit.get("proposed_operations")
    if not isinstance(operations, list) or len(operations) != 1:
        raise ValueError(f"curated repair unit must have exactly one operation:{work_unit_id}")
    operation = operations[0]
    if not isinstance(operation, dict):
        raise ValueError(f"curated repair operation must be an object:{work_unit_id}")
    operation["provenance"] = {
        **(operation.get("provenance") if isinstance(operation.get("provenance"), dict) else {}),
        "repair_id": manifest.get("repair_id") or "MAINT-082",
        "manifest_sha256": manifest_sha256,
        "work_unit_id": work_unit_id,
    }
    unit["chain_key"] = {
        **chain_key,
        "old_id": old_id,
    }
    unit["proposed_operations"] = [operation]
    unit["source_evidence"] = {
        **(unit.get("source_evidence") if isinstance(unit.get("source_evidence"), dict) else {}),
        "reviewer": manifest.get("reviewer"),
        "reviewed_at": manifest.get("reviewed_at"),
        "curated_manifest_sha256": manifest_sha256,
        "repair_id": manifest.get("repair_id") or "MAINT-082",
    }
    unit["safety_blockers"] = list(unit.get("safety_blockers") or [])
    unit["operation_blockers"] = list(unit.get("operation_blockers") or [])
    return unit


def _curated_manifest_summary(apply_units: list[dict[str, Any]]) -> dict[str, Any]:
    operation_counts: Counter[str] = Counter()
    classification_counts: Counter[str] = Counter()
    for unit in apply_units:
        classification_counts.update([str(unit.get("classification") or "missing")])
        for operation in unit.get("proposed_operations") or []:
            operation_counts.update([f"{operation.get('operation')}"])
    return {
        "status": "SUCCESS",
        "total_work_units": len(apply_units),
        "mechanical_safe_count": len(apply_units),
        "bad_edge_repair_count": 0,
        "review_required_count": 0,
        "blocked_count": 0,
        "classification_counts": dict(sorted(classification_counts.items())),
        "operation_counts": dict(sorted(operation_counts.items())),
        "proposed_operation_counts": dict(sorted(operation_counts.items())),
        "blocker_counts": {},
        "operation_blocker_counts": {},
        "review_note_counts": {},
        "observability_note_counts": {},
        "issue_counts": {},
    }


def _mark_curated_live_report(report: dict[str, Any], curated_manifest_sha256: str) -> None:
    report.update(
        {
            "mode": "live_curated_repair_apply",
            "operator_surface": "curated_repair_live",
            "legacy_operator_surface": None,
            "curated_manifest_sha256": curated_manifest_sha256,
        }
    )


def _write_live_apply_rollback_runbook(path: Path, live_db_path: Path, backup_db_path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Supersession Live Repair Rollback Runbook",
                "",
                f"Live DB path: `{live_db_path}`",
                f"Pre-apply backup DB path: `{backup_db_path}`",
                "",
                "1. Stop Pith server and scheduled writers.",
                f"2. Copy `{live_db_path}` to a failed-apply quarantine path.",
                f"3. Restore `{backup_db_path}` to `{live_db_path}`.",
                "4. Run `sqlite3 <live-db-path> 'PRAGMA integrity_check;'` and require `ok`.",
                "5. Restart Pith.",
                "6. Re-run the supersession and trust-governance scorecards.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _run_live_apply_post_checks(
    db_path: Path,
    *,
    before_metrics: dict[str, int],
    after_metrics: dict[str, int],
    apply_units: list[dict[str, Any]],
    changed_rows: int,
    row_evidence_count: int,
    scorecard_runner: Callable[[Path], dict[str, Any]] | None,
) -> dict[str, Any]:
    expected = _expected_metric_deltas(apply_units)
    metrics_ok = (
        after_metrics["subject_keys"] - before_metrics["subject_keys"] == expected["subject_keys"]
        and after_metrics["supersession_reasons"] - before_metrics["supersession_reasons"]
        == expected["supersession_reasons"]
        and after_metrics["superseded_timestamps"] - before_metrics["superseded_timestamps"]
        == expected["superseded_timestamps"]
        and before_metrics["split_superseded_currency"] - after_metrics["split_superseded_currency"]
        == expected["split_superseded_currency"]
        and after_metrics["superseded_edges"] - before_metrics["superseded_edges"] == expected["superseded_edges"]
    )
    checks: dict[str, Any] = {
        "source_integrity_check": _sqlite_integrity_check(db_path),
        "expected_metric_deltas": expected,
        "metric_deltas": {
            key: after_metrics[key] - before_metrics[key] for key in before_metrics.keys()
        },
        "metrics_match_expected": metrics_ok,
        "row_evidence_count": row_evidence_count,
        "row_evidence_complete": row_evidence_count == changed_rows,
    }
    if scorecard_runner is not None:
        checks["scorecards"] = [scorecard_runner(db_path)]
    else:
        checks["scorecards"] = _run_default_scorecards()
    return checks


def _expected_metric_deltas(apply_units: list[dict[str, Any]]) -> dict[str, int]:
    deltas = {
        "subject_keys": 0,
        "supersession_reasons": 0,
        "superseded_timestamps": 0,
        "split_superseded_currency": 0,
        "superseded_edges": 0,
    }
    for unit in apply_units:
        operations = unit.get("proposed_operations") or []
        op_by_field = {operation.get("field"): operation for operation in operations}
        clear_edge_op = next(
            (
                operation
                for operation in operations
                if operation.get("operation") == CLEAR_SUPERSESSION_EDGE_OPERATION
            ),
            None,
        )
        if clear_edge_op:
            before = clear_edge_op.get("before") if isinstance(clear_edge_op.get("before"), dict) else {}
            if before.get("superseded_by"):
                deltas["superseded_edges"] -= 1
            if before.get("supersession_reason") is not None:
                deltas["supersession_reasons"] -= 1
            if before.get("superseded_at") is not None:
                deltas["superseded_timestamps"] -= 1
            continue
        normalize_op = next(
            (
                operation
                for operation in operations
                if operation.get("operation") == NORMALIZE_SUPERSESSION_CHAIN_OPERATION
            ),
            None,
        )
        if normalize_op:
            before = normalize_op.get("before") if isinstance(normalize_op.get("before"), dict) else {}
            after = normalize_op.get("after") if isinstance(normalize_op.get("after"), dict) else {}
            if not before.get("subject_key") and after.get("subject_key"):
                deltas["subject_keys"] += 1
            if bool(after.get("superseded_by")) != bool(before.get("superseded_by")):
                deltas["superseded_edges"] += 1 if after.get("superseded_by") else -1
            if bool(after.get("supersession_reason")) != bool(before.get("supersession_reason")):
                deltas["supersession_reasons"] += 1 if after.get("supersession_reason") else -1
            if bool(after.get("superseded_at")) != bool(before.get("superseded_at")):
                deltas["superseded_timestamps"] += 1 if after.get("superseded_at") else -1
            before_bad_currency = before.get("status") == "superseded" and before.get("currency_status") != "SUPERSEDED"
            after_bad_currency = after.get("status") == "superseded" and after.get("currency_status") != "SUPERSEDED"
            if before_bad_currency != after_bad_currency:
                deltas["split_superseded_currency"] += 1 if before_bad_currency else -1
            continue
        subject_op = op_by_field.get("subject_key")
        if subject_op and not subject_op.get("before") and subject_op.get("after"):
            deltas["subject_keys"] += 1
        reason_op = op_by_field.get("supersession_reason")
        if reason_op and not reason_op.get("before") and reason_op.get("after"):
            deltas["supersession_reasons"] += 1
        currency_op = op_by_field.get("currency_status")
        if currency_op and currency_op.get("before") != "SUPERSEDED" and currency_op.get("after") == "SUPERSEDED":
            deltas["split_superseded_currency"] += 1
    return deltas


def _run_default_scorecards() -> list[dict[str, Any]]:
    repo_root = Path(__file__).resolve().parents[2]
    scripts = [
        repo_root / "scripts" / "supersession_trust_scorecard.py",
        repo_root / "scripts" / "trust_governance_scorecard.py",
        repo_root / "scripts" / "trust_governance_effectiveness_scorecard.py",
    ]
    results: list[dict[str, Any]] = []
    for script in scripts:
        if not script.exists():
            results.append({"script": str(script), "status": "MISSING"})
            continue
        completed = subprocess.run(
            [sys.executable, str(script), "--json"],
            cwd=str(repo_root),
            text=True,
            capture_output=True,
            check=False,
            timeout=120,
        )
        payload: dict[str, Any]
        try:
            payload = json.loads(completed.stdout or "{}")
        except json.JSONDecodeError:
            payload = {"raw_stdout": completed.stdout}
        results.append(_compact_scorecard_payload(payload, script=script, returncode=completed.returncode))
    return results


def _compact_scorecard_payload(payload: dict[str, Any], *, script: Path, returncode: int) -> dict[str, Any]:
    return {
        "script": str(script),
        "returncode": returncode,
        "schema_version": payload.get("schema_version"),
        "status": payload.get("status"),
        "result": payload.get("result"),
        "target_status": payload.get("target_status"),
        "metrics": payload.get("metrics"),
        "alarms": payload.get("alarms"),
        "case_count": payload.get("case_count"),
        "claim_boundary": payload.get("claim_boundary"),
    }


def _post_checks_pass(post_checks: dict[str, Any]) -> bool:
    if post_checks.get("source_integrity_check") != "ok":
        return False
    if not post_checks.get("metrics_match_expected"):
        return False
    if not post_checks.get("row_evidence_complete"):
        return False
    for scorecard in post_checks.get("scorecards") or []:
        if scorecard.get("returncode", 0) != 0:
            return False
        status = str(scorecard.get("status") or scorecard.get("result") or "").upper()
        if status and status not in {"PASS", "SUCCESS"}:
            return False
    return True



def _fetch_one(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...]) -> dict[str, Any] | None:
    cursor = conn.execute(sql, params)
    row = cursor.fetchone()
    if not row:
        return None
    columns = [item[0] for item in cursor.description]
    return dict(zip(columns, tuple(row), strict=True))


def _table_has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return any(row[1] == column for row in conn.execute(f"PRAGMA table_info({table})").fetchall())


def _apply_concept_metadata_operations_conn(
    conn: sqlite3.Connection,
    concept_id: str,
    operations: list[dict[str, Any]],
) -> int:
    if not concept_id:
        raise ValueError("operation concept_id is required")
    if not operations:
        return 0

    row = _fetch_concept_repair_row(conn, concept_id)
    if not row:
        raise ValueError(f"concept not found:{concept_id}")

    if len(operations) == 1 and operations[0].get("operation") == NORMALIZE_SUPERSESSION_CHAIN_OPERATION:
        return _apply_normalize_supersession_chain_conn(conn, concept_id, operations[0], row)
    if any(operation.get("operation") == NORMALIZE_SUPERSESSION_CHAIN_OPERATION for operation in operations):
        raise ValueError("normalize_supersession_chain cannot be combined with other operations")
    if len(operations) == 1 and operations[0].get("operation") == CLEAR_SUPERSESSION_EDGE_OPERATION:
        return _apply_clear_supersession_edge_conn(conn, concept_id, operations[0], row)
    if any(operation.get("operation") == CLEAR_SUPERSESSION_EDGE_OPERATION for operation in operations):
        raise ValueError("clear_supersession_edge cannot be combined with other operations")

    assignments: list[str] = ["updated_at = :updated_at"]
    params: dict[str, Any] = {"concept_id": concept_id, "updated_at": datetime.now(UTC).isoformat()}
    json_fields: list[tuple[str, str]] = []
    seen_fields: set[str] = set()
    status_repair = False
    currency_repair = False

    for index, operation in enumerate(operations):
        if operation.get("operation") != "set_field" or operation.get("table") != "concepts":
            raise ValueError(f"unsupported operation shape:{operation!r}")
        if operation.get("concept_id") != concept_id:
            raise ValueError("operation concept_id mismatch")
        field = str(operation.get("field") or "")
        if field not in APPROVED_MECHANICAL_FIELDS:
            raise ValueError(f"unsupported operation field:{field}")
        if field in seen_fields:
            raise ValueError(f"duplicate operation field:{field}")
        seen_fields.add(field)
        if row.get(field) != operation.get("before"):
            raise ValueError(f"operation before-value mismatch:{field}")
        value = operation.get("after")
        if field in {"subject_key", "superseded_at", "supersession_reason"} and not str(value or "").strip():
            raise ValueError(f"{field} requires a non-empty value")
        if field == "currency_status" and value != "SUPERSEDED":
            raise ValueError("currency_status repairs may only set SUPERSEDED")
        if field == "status" and value != "superseded":
            raise ValueError("status repairs may only set superseded")
        status_repair = status_repair or field == "status"
        currency_repair = currency_repair or field == "currency_status"

        param_name = f"value_{index}"
        assignments.append(f"{field} = :{param_name}")
        params[param_name] = value
        if field in {"status", "currency_status", "subject_key"}:
            json_fields.append((field, param_name))
        if field == "status":
            assignments.append("is_current = 0")

    if status_repair and row.get("currency_status") != "SUPERSEDED" and not currency_repair:
        raise ValueError("status repair requires paired currency_status repair")

    json_sql = ""
    if json_fields:
        json_expr = "COALESCE(data, '{}')"
        for field, param_name in json_fields:
            json_expr = f"json_set({json_expr}, '$.{field}', :{param_name})"
        json_sql = f", data = {json_expr}"

    sql = f"""
        UPDATE concepts
        SET {", ".join(assignments)}
            {json_sql}
        WHERE id = :concept_id
    """
    return conn.execute(sql, params).rowcount


def _apply_clear_supersession_edge_conn(
    conn: sqlite3.Connection,
    concept_id: str,
    operation: dict[str, Any],
    row: dict[str, Any],
) -> int:
    if operation.get("table") != "concepts":
        raise ValueError(f"unsupported operation shape:{operation!r}")
    if operation.get("concept_id") != concept_id:
        raise ValueError("operation concept_id mismatch")
    before = operation.get("before")
    after = operation.get("after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ValueError("clear_supersession_edge requires before and after objects")
    if set(before.keys()) != set(SUPERSESSION_EDGE_FIELDS):
        raise ValueError("clear_supersession_edge before fields mismatch")
    if set(after.keys()) != set(SUPERSESSION_EDGE_FIELDS):
        raise ValueError("clear_supersession_edge after fields mismatch")
    if any(after.get(field) is not None for field in SUPERSESSION_EDGE_FIELDS):
        raise ValueError("clear_supersession_edge after values must be null")
    if not before.get("superseded_by"):
        raise ValueError("clear_supersession_edge requires a non-empty superseded_by before value")
    for field in SUPERSESSION_EDGE_FIELDS:
        if row.get(field) != before.get(field):
            raise ValueError(f"operation before-value mismatch:{field}")
    if not before.get("superseded_by"):
        raise ValueError("clear_supersession_edge requires a non-empty superseded_by before value")

    params = {"concept_id": concept_id, "updated_at": datetime.now(UTC).isoformat()}
    return conn.execute(
        """
        UPDATE concepts
        SET superseded_by = NULL,
            supersession_reason = NULL,
            superseded_at = NULL,
            updated_at = :updated_at,
            data = json_remove(
                COALESCE(data, '{}'),
                '$.superseded_by',
                '$.supersession_reason',
                '$.superseded_at'
            )
        WHERE id = :concept_id
        """,
        params,
    ).rowcount


def _apply_normalize_supersession_chain_conn(
    conn: sqlite3.Connection,
    concept_id: str,
    operation: dict[str, Any],
    row: dict[str, Any],
) -> int:
    _validate_normalize_supersession_chain_operation(operation)
    if operation.get("concept_id") != concept_id:
        raise ValueError("operation concept_id mismatch")
    before = operation["before"]
    after = operation["after"]
    for field in NORMALIZE_SUPERSESSION_BEFORE_FIELDS:
        if row.get(field) != before.get(field):
            raise ValueError(f"operation before-value mismatch:{field}")

    superseded_by = after.get("superseded_by")
    subject_key = str(after.get("subject_key") or "").strip()
    if superseded_by:
        target = _fetch_concept_repair_row(conn, str(superseded_by))
        target_subject = target.get("subject_key")
        if target_subject and target_subject != subject_key:
            raise ValueError("normalize_supersession_chain target subject_key mismatch")

    params = {
        "concept_id": concept_id,
        "updated_at": datetime.now(UTC).isoformat(),
        "status": after.get("status"),
        "currency_status": after.get("currency_status"),
        "is_current": int(after.get("is_current") or 0),
        "subject_key": subject_key,
        "superseded_by": superseded_by,
        "supersession_reason": after.get("supersession_reason"),
        "superseded_at": after.get("superseded_at"),
    }
    if superseded_by:
        data_sql = """
            json_set(
                COALESCE(data, '{}'),
                '$.status', :status,
                '$.currency_status', :currency_status,
                '$.subject_key', :subject_key,
                '$.superseded_by', :superseded_by,
                '$.supersession_reason', :supersession_reason,
                '$.superseded_at', :superseded_at
            )
        """
    else:
        data_sql = """
            json_remove(
                json_set(
                    COALESCE(data, '{}'),
                    '$.status', :status,
                    '$.currency_status', :currency_status,
                    '$.subject_key', :subject_key
                ),
                '$.superseded_by',
                '$.supersession_reason',
                '$.superseded_at'
            )
        """
    changed = conn.execute(
        f"""
        UPDATE concepts
        SET status = :status,
            currency_status = :currency_status,
            is_current = :is_current,
            subject_key = :subject_key,
            superseded_by = :superseded_by,
            supersession_reason = :supersession_reason,
            superseded_at = :superseded_at,
            updated_at = :updated_at,
            data = {data_sql}
        WHERE id = :concept_id
        """,
        params,
    ).rowcount
    if not changed:
        return 0
    if superseded_by:
        _ensure_supersession_provenance_conn(conn, concept_id, str(superseded_by), operation)
    return changed


def _ensure_supersession_provenance_conn(
    conn: sqlite3.Connection,
    old_id: str,
    new_id: str,
    operation: dict[str, Any],
) -> None:
    now = datetime.now(UTC).isoformat()
    edge_exists = conn.execute(
        "SELECT 1 FROM associations WHERE source = ? AND target = ? AND relation = 'supersedes' LIMIT 1",
        (new_id, old_id),
    ).fetchone()
    if not edge_exists:
        conn.execute(
            """INSERT INTO associations (source, target, relation, strength, created_at, mechanism, direction, chain_id)
               VALUES (?, ?, 'supersedes', 0.9, ?, 'maint077_operator_repair', 'forward', ?)""",
            (new_id, old_id, now, new_id),
        )

    provenance = operation.get("provenance") if isinstance(operation.get("provenance"), dict) else {}
    details = {
        "repair_id": provenance.get("repair_id") or "MAINT-077",
        "manifest_sha256": provenance.get("manifest_sha256") or "",
        "work_unit_id": provenance.get("work_unit_id") or "",
        "superseded_by": new_id,
        "reason": operation["after"].get("supersession_reason"),
    }
    event_exists = conn.execute(
        """
        SELECT 1
        FROM governance_events
        WHERE event_type = 'decision_supersession'
          AND concept_id = ?
          AND json_valid(COALESCE(details, '{}'))
          AND COALESCE(json_extract(details, '$.repair_id'), '') = ?
          AND COALESCE(json_extract(details, '$.manifest_sha256'), '') = ?
          AND COALESCE(json_extract(details, '$.work_unit_id'), '') = ?
          AND COALESCE(json_extract(details, '$.superseded_by'), '') = ?
        LIMIT 1
        """,
        (
            old_id,
            details["repair_id"],
            details["manifest_sha256"],
            details["work_unit_id"],
            details["superseded_by"],
        ),
    ).fetchone()
    if event_exists:
        return
    conn.execute(
        """INSERT INTO governance_events (event_type, concept_id, details, created_at)
           VALUES ('decision_supersession', ?, ?, ?)""",
        (old_id, json.dumps(details, sort_keys=True), now),
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_queue_projection(manifest: dict[str, Any]) -> dict[str, Any]:
    fingerprint = manifest.get("input_fingerprint") if isinstance(manifest.get("input_fingerprint"), dict) else {}
    return {
        "schema_version": manifest.get("schema_version"),
        "input_fingerprint": {
            key: value
            for key, value in fingerprint.items()
            if key not in {"ledger_path", "manifest_path", "decisions_path"}
        },
        "summary": manifest.get("summary"),
        "work_units": manifest.get("work_units"),
    }


def _validate_apply_units(units: list[dict[str, Any]]) -> None:
    for unit in units:
        evidence = unit.get("source_evidence") if isinstance(unit.get("source_evidence"), dict) else {}
        reviewer = str(evidence.get("reviewer") or "").strip().lower()
        if not reviewer:
            raise ValueError(f"missing reviewer:{unit.get('work_unit_id')}")
        if any(marker in reviewer for marker in DISALLOWED_REVIEWER_MARKERS):
            raise ValueError(f"synthetic reviewer rejected:{reviewer}")
        if not str(evidence.get("reviewed_at") or "").strip():
            raise ValueError(f"missing reviewed_at:{unit.get('work_unit_id')}")
        if unit.get("review_decision") != "approve_repair":
            raise ValueError(f"non-approved unit selected:{unit.get('work_unit_id')}")
        operations = unit.get("proposed_operations") or []
        if not operations:
            raise ValueError(f"approved mechanical unit has no operations:{unit.get('work_unit_id')}")
        operation_types = {operation.get("operation") for operation in operations}
        if NORMALIZE_SUPERSESSION_CHAIN_OPERATION in operation_types:
            if operation_types != {NORMALIZE_SUPERSESSION_CHAIN_OPERATION} or len(operations) != 1:
                raise ValueError(f"mixed normalize_supersession_chain unit rejected:{unit.get('work_unit_id')}")
            _validate_normalize_supersession_chain_operation(operations[0])
            continue
        if CLEAR_SUPERSESSION_EDGE_OPERATION in operation_types:
            if operation_types != {CLEAR_SUPERSESSION_EDGE_OPERATION} or len(operations) != 1:
                raise ValueError(f"mixed clear_supersession_edge unit rejected:{unit.get('work_unit_id')}")
            _validate_clear_supersession_edge_operation(operations[0])
            continue
        seen_fields: set[str] = set()
        for operation in operations:
            if operation.get("operation") != "set_field" or operation.get("table") != "concepts":
                raise ValueError(f"unsupported operation shape:{operation!r}")
            field = operation.get("field")
            if field not in APPROVED_MECHANICAL_FIELDS:
                raise ValueError(f"unsupported operation field:{field}")
            if field in seen_fields:
                raise ValueError(f"duplicate operation field:{field}")
            seen_fields.add(field)


def _validate_clear_supersession_edge_operation(operation: dict[str, Any]) -> None:
    if operation.get("operation") != CLEAR_SUPERSESSION_EDGE_OPERATION or operation.get("table") != "concepts":
        raise ValueError(f"unsupported operation shape:{operation!r}")
    if operation.get("field") not in {None, "superseded_by"}:
        raise ValueError(f"unsupported operation field:{operation.get('field')}")
    if not operation.get("concept_id"):
        raise ValueError("clear_supersession_edge concept_id is required")
    before = operation.get("before")
    after = operation.get("after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ValueError("clear_supersession_edge requires before and after objects")
    if set(before.keys()) != set(SUPERSESSION_EDGE_FIELDS):
        raise ValueError("clear_supersession_edge before fields mismatch")
    if set(after.keys()) != set(SUPERSESSION_EDGE_FIELDS):
        raise ValueError("clear_supersession_edge after fields mismatch")
    if any(after.get(field) is not None for field in SUPERSESSION_EDGE_FIELDS):
        raise ValueError("clear_supersession_edge after values must be null")


def _validate_normalize_supersession_chain_operation(operation: dict[str, Any]) -> None:
    if operation.get("operation") != NORMALIZE_SUPERSESSION_CHAIN_OPERATION or operation.get("table") != "concepts":
        raise ValueError(f"unsupported operation shape:{operation!r}")
    if operation.get("field") not in {None, "superseded_by"}:
        raise ValueError(f"unsupported operation field:{operation.get('field')}")
    if not operation.get("concept_id"):
        raise ValueError("normalize_supersession_chain concept_id is required")
    before = operation.get("before")
    after = operation.get("after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ValueError("normalize_supersession_chain requires before and after objects")
    if set(before.keys()) != set(NORMALIZE_SUPERSESSION_BEFORE_FIELDS):
        raise ValueError("normalize_supersession_chain before fields mismatch")
    missing_after = [field for field in NORMALIZE_SUPERSESSION_BEFORE_FIELDS if field not in after]
    if missing_after:
        raise ValueError(f"normalize_supersession_chain after fields missing:{','.join(sorted(missing_after))}")
    subject_key = str(after.get("subject_key") or "").strip()
    if not subject_key:
        raise ValueError("normalize_supersession_chain subject_key is required")
    status = after.get("status")
    currency_status = after.get("currency_status")
    is_current = int(after.get("is_current") or 0)
    superseded_by = after.get("superseded_by")
    if status == "active":
        if currency_status != "ACTIVE" or is_current != 1:
            raise ValueError("active normalize_supersession_chain rows must be ACTIVE and current")
        if superseded_by or after.get("supersession_reason") or after.get("superseded_at"):
            raise ValueError("active normalize_supersession_chain rows cannot retain supersession fields")
    elif status == "superseded":
        if currency_status != "SUPERSEDED" or is_current != 0:
            raise ValueError("superseded normalize_supersession_chain rows must be SUPERSEDED and non-current")
        if not str(superseded_by or "").strip():
            raise ValueError("superseded normalize_supersession_chain rows require superseded_by")
        if not str(after.get("supersession_reason") or "").strip():
            raise ValueError("superseded normalize_supersession_chain rows require supersession_reason")
        if not str(after.get("superseded_at") or "").strip():
            raise ValueError("superseded normalize_supersession_chain rows require superseded_at")
    else:
        raise ValueError(f"unsupported normalize_supersession_chain status:{status}")


def _audit_row_hash(row: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(row).encode("utf-8")).hexdigest()


def _line_hash(line: dict[str, Any]) -> str:
    clone = dict(line)
    clone.pop("ledger_line_hash", None)
    for field in VERIFICATION_FIELDS:
        clone.pop(field, None)
    return hashlib.sha256(_canonical_json(clone).encode("utf-8")).hexdigest()


def _legacy_line_hash(line: dict[str, Any]) -> str:
    clone = dict(line)
    clone.pop("ledger_line_hash", None)
    clone.pop("verified_line_hash", None)
    return hashlib.sha256(_canonical_json(clone).encode("utf-8")).hexdigest()


def _verified_line_hash(line: dict[str, Any]) -> str:
    clone = dict(line)
    clone.pop("verified_line_hash", None)
    return hashlib.sha256(_canonical_json(clone).encode("utf-8")).hexdigest()


def _old_ids_sha256(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256("\n".join(sorted(str(row["old_id"]) for row in rows)).encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _safe_json(value: Any) -> Any:
    try:
        return json.loads(value) if isinstance(value, str) and value else None
    except json.JSONDecodeError:
        return None


def _emit(payload: dict[str, Any], *, json_mode: bool, exit_on_fail: bool = False) -> int:
    if json_mode:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(_human_status(payload))
    return 1 if exit_on_fail and payload.get("status") == "FAIL" else 0


def _human_status(payload: dict[str, Any]) -> str:
    lines = [f"Status: {payload.get('status')}", f"Mode: {payload.get('mode', 'verify')}"]
    if payload.get("ledger_sha256"):
        lines.append(f"Ledger: {payload['ledger_sha256']}")
    if payload.get("line_count") is not None:
        lines.append(f"Lines: {payload['line_count']}")
    if payload.get("error"):
        lines.append(f"Error: {payload['error']}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
