#!/usr/bin/env python3
"""Convert reviewed implicit supersession TSV packets into gold extension JSON."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ops.implicit_supersession_proposals import RETENTION_MODES, REVIEW_DECISIONS

SCHEMA_VERSION = "implicit_supersession_review_gold_extension.v1"
DECISION_TO_EXPECTED = {
    "approve_supersession": ("review_recommended", True),
    "reject_not_supersession": ("not_supersession", False),
    "needs_more_context": ("needs_context", True),
    "partial_retain": ("needs_context", True),
}
REVIEW_METADATA_FIELDS = (
    "review_source",
    "source_packet_paths",
    "first_seen_at",
    "latest_seen_at",
    "risk_class",
    "source_packet_count",
)


def build_extension(rows: list[dict[str, str]], *, review_source: str = "") -> dict[str, Any]:
    accepted_cases: list[dict[str, Any]] = []
    errors: list[str] = []
    skipped = 0
    decision_counts: Counter[str] = Counter()
    seen_case_ids: set[str] = set()

    for row_number, row in enumerate(rows, start=2):
        decision = (row.get("review_decision") or "").strip()
        if not decision:
            skipped += 1
            continue
        decision_counts[decision] += 1
        row_errors = _validate_row(row_number, row, decision)
        case_id = _case_id(row)
        if case_id in seen_case_ids:
            row_errors.append(f"row {row_number}: duplicate case id {case_id}")
        if row_errors:
            errors.extend(row_errors)
            continue
        seen_case_ids.add(case_id)
        expected_disposition, expected_proposal = DECISION_TO_EXPECTED[decision]
        source = (row.get("review_source") or review_source or "").strip()
        accepted_cases.append(
            {
                "id": case_id,
                "split": "live_reviewed",
                "old": _concept_payload(row, "old", review_source=source),
                "new": _concept_payload(row, "new", review_source=source),
                "expected": {
                    "disposition": expected_disposition,
                    "proposal": expected_proposal,
                    "review_decision": decision,
                    "retention_mode": (row.get("retention_mode") or "").strip(),
                    "review_rationale": (row.get("review_rationale") or "").strip(),
                    "review_source": source,
                    **_review_metadata(row),
                },
            }
        )

    status = "PASS" if not errors else "FAIL"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "generated_at": datetime.now(UTC).isoformat(),
        "review_source": review_source,
        "counts": {
            "rows_read": len(rows),
            "rows_accepted": len(accepted_cases),
            "rows_skipped": skipped,
            "validation_errors": len(errors),
            "decision_counts": dict(sorted(decision_counts.items())),
        },
        "errors": errors,
        "cases": accepted_cases,
    }


def _validate_row(row_number: int, row: dict[str, str], decision: str) -> list[str]:
    errors: list[str] = []
    if decision not in REVIEW_DECISIONS:
        errors.append(f"row {row_number}: invalid review_decision {decision!r}")
        return errors
    retention_mode = (row.get("retention_mode") or "").strip()
    if retention_mode and retention_mode not in RETENTION_MODES:
        errors.append(f"row {row_number}: invalid retention_mode {retention_mode!r}")
    if decision == "approve_supersession" and retention_mode != "replace":
        errors.append(f"row {row_number}: approve_supersession requires retention_mode=replace")
    if decision == "partial_retain" and retention_mode != "partial_retain":
        errors.append(f"row {row_number}: partial_retain requires retention_mode=partial_retain")
    if decision in {"approve_supersession", "partial_retain"} and not (
        row.get("review_rationale") or ""
    ).strip():
        errors.append(f"row {row_number}: {decision} requires review_rationale")
    for field in ("old_id", "new_id", "old_summary", "new_summary", "subject_key"):
        if not (row.get(field) or "").strip():
            errors.append(f"row {row_number}: missing required field {field}")
    if decision == "approve_supersession":
        for field in ("old_created_at", "new_created_at"):
            if not (row.get(field) or "").strip():
                errors.append(f"row {row_number}: approve_supersession missing {field}")
    return errors


def _case_id(row: dict[str, str]) -> str:
    old_id = (row.get("old_id") or "").strip()
    new_id = (row.get("new_id") or "").strip()
    return f"review-{old_id}-{new_id}"


def _review_metadata(row: dict[str, str]) -> dict[str, str]:
    metadata = {
        field: (row.get(field) or "").strip()
        for field in REVIEW_METADATA_FIELDS
        if (row.get(field) or "").strip()
    }
    source_packets = metadata.get("source_packet_paths")
    if source_packets and not metadata.get("source_path"):
        metadata["source_path"] = source_packets.split("|", 1)[0]
    return metadata


def _concept_payload(row: dict[str, str], prefix: str, *, review_source: str = "") -> dict[str, Any]:
    return {
        "id": (row.get(f"{prefix}_id") or "").strip(),
        "summary": (row.get(f"{prefix}_summary") or "").strip(),
        "subject_key": (row.get("subject_key") or "").strip(),
        "created_at": (row.get(f"{prefix}_created_at") or "").strip(),
        "data": {
            "review_schema_version": (row.get("review_schema_version") or "").strip(),
            "review_group_key": (row.get("review_group_key") or "").strip(),
            "review_source": review_source,
            "reviewer": (row.get("reviewer") or "").strip(),
            "reviewed_at": (row.get("reviewed_at") or "").strip(),
            **_review_metadata(row),
        },
    }


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_path.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-tsv", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--review-source", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = build_extension(_load_rows(args.review_tsv), review_source=args.review_source)
    if report["status"] == "PASS":
        _write_json_atomic(args.json_output, report)
    if args.json or report["status"] != "PASS":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        counts = report["counts"]
        print(
            "implicit supersession review ingest: "
            f"status={report['status']}, "
            f"rows_read={counts['rows_read']}, "
            f"rows_accepted={counts['rows_accepted']}, "
            f"rows_skipped={counts['rows_skipped']}, "
            f"output={args.json_output}"
        )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
