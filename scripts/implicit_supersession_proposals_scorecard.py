#!/usr/bin/env python3
"""Scorecard for implicit supersession proposal discovery."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ops.implicit_supersession_proposals import (
    CLAIM_BOUNDARY,
    discover_proposals,
    open_readonly_connection,
    write_report_artifacts,
)
from scripts.eval.implicit_supersession_proposals_eval import evaluate_cases

SCHEMA_VERSION = "implicit_supersession_proposals_scorecard.v1"
REVIEW_EXTENSION_SCHEMA_VERSION = "implicit_supersession_review_gold_extension.v1"
READINESS_USER_REVIEW_SOURCE = "user_reviewed"
READINESS_AGENT_SUGGESTED_SOURCE = "agent_suggested"
DEFAULT_REVIEW_EXTENSION_DIR = (
    Path.home() / ".pith" / "reports" / "monitoring" / "implicit-supersession-review-gold"
)


def _load_gold(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["cases"] if isinstance(payload, dict) else payload


def _iter_review_extension_files(
    paths: list[Path],
    directories: list[Path],
) -> tuple[list[Path], list[str]]:
    files: list[Path] = []
    errors: list[str] = []
    seen: set[Path] = set()

    for path in paths:
        expanded = path.expanduser()
        if not expanded.exists():
            errors.append(f"review_extension_missing:{expanded}")
            continue
        resolved = expanded.resolve()
        if resolved not in seen:
            seen.add(resolved)
            files.append(resolved)

    for directory in directories:
        expanded_dir = directory.expanduser()
        if not expanded_dir.exists():
            continue
        if not expanded_dir.is_dir():
            errors.append(f"review_extension_dir_not_directory:{expanded_dir}")
            continue
        for candidate in sorted(expanded_dir.glob("*.json")):
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                files.append(resolved)

    return files, errors


def _load_review_extensions(
    *,
    paths: list[Path] | None = None,
    directories: list[Path] | None = None,
    existing_case_ids: set[str] | None = None,
) -> dict[str, Any]:
    extension_paths = list(paths or [])
    extension_dirs = list(directories or [DEFAULT_REVIEW_EXTENSION_DIR])
    configured = bool(extension_paths or directories)
    files, errors = _iter_review_extension_files(extension_paths, extension_dirs)
    case_ids = set(existing_case_ids or set())
    cases: list[dict[str, Any]] = []
    decision_counts: Counter[str] = Counter()
    files_loaded: list[str] = []
    files_rejected: list[str] = []

    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"review_extension_invalid_json:{path}:{exc}")
            files_rejected.append(str(path))
            continue
        if payload.get("schema_version") != REVIEW_EXTENSION_SCHEMA_VERSION:
            errors.append(f"review_extension_wrong_schema:{path}:{payload.get('schema_version')}")
            files_rejected.append(str(path))
            continue
        if payload.get("status") != "PASS":
            errors.append(f"review_extension_status_not_pass:{path}:{payload.get('status')}")
            files_rejected.append(str(path))
            continue
        extension_cases = payload.get("cases")
        if not isinstance(extension_cases, list):
            errors.append(f"review_extension_cases_not_list:{path}")
            files_rejected.append(str(path))
            continue

        file_case_errors: list[str] = []
        file_case_ids: set[str] = set()
        file_cases: list[dict[str, Any]] = []
        file_decision_counts: Counter[str] = Counter()
        for case in extension_cases:
            case_id = str(case.get("id") or "").strip() if isinstance(case, dict) else ""
            if not case_id:
                file_case_errors.append(f"review_extension_missing_case_id:{path}")
                continue
            if case_id in case_ids or case_id in file_case_ids:
                file_case_errors.append(f"review_extension_duplicate_case_id:{path}:{case_id}")
                continue
            file_case_ids.add(case_id)
            file_cases.append(case)
            expected = case.get("expected") if isinstance(case, dict) else {}
            if isinstance(expected, dict):
                decision = str(expected.get("review_decision") or "").strip()
                if decision:
                    file_decision_counts[decision] += 1
        if file_case_errors:
            errors.extend(file_case_errors)
            files_rejected.append(str(path))
        else:
            case_ids.update(file_case_ids)
            cases.extend(file_cases)
            decision_counts.update(file_decision_counts)
            files_loaded.append(str(path))

    review_readiness = _review_readiness_summary(cases)
    return {
        "schema_version": REVIEW_EXTENSION_SCHEMA_VERSION,
        "configured": configured,
        "default_dir": str(DEFAULT_REVIEW_EXTENSION_DIR),
        "files_discovered": len(files),
        "files_loaded": files_loaded,
        "files_rejected": files_rejected,
        "case_count": len(cases),
        "decision_counts": dict(sorted(decision_counts.items())),
        **review_readiness,
        "errors": errors,
        "cases": cases,
    }


def _case_expected(case: dict[str, Any]) -> dict[str, Any]:
    expected = case.get("expected") if isinstance(case, dict) else {}
    return expected if isinstance(expected, dict) else {}


def _concept_data(case: dict[str, Any], side: str) -> dict[str, Any]:
    concept = case.get(side) if isinstance(case, dict) else {}
    data = concept.get("data") if isinstance(concept, dict) else {}
    return data if isinstance(data, dict) else {}


def _review_source(case: dict[str, Any]) -> str:
    expected_source = str(_case_expected(case).get("review_source") or "").strip()
    if expected_source:
        return expected_source
    old_source = str(_concept_data(case, "old").get("review_source") or "").strip()
    new_source = str(_concept_data(case, "new").get("review_source") or "").strip()
    if old_source and old_source == new_source:
        return old_source
    return old_source or new_source or "unknown"


def _has_source_packet(case: dict[str, Any]) -> bool:
    expected = _case_expected(case)
    if str(expected.get("source_packet_paths") or "").strip():
        return True
    if str(expected.get("source_packet_path") or "").strip():
        return True
    for side in ("old", "new"):
        data = _concept_data(case, side)
        if str(data.get("source_packet_paths") or "").strip() or str(data.get("source_packet_path") or "").strip():
            return True
    return False


def _has_temporal_pair(case: dict[str, Any]) -> bool:
    old = case.get("old") if isinstance(case, dict) else {}
    new = case.get("new") if isinstance(case, dict) else {}
    return bool(
        isinstance(old, dict)
        and isinstance(new, dict)
        and str(old.get("created_at") or "").strip()
        and str(new.get("created_at") or "").strip()
    )


def _ratio(numerator: int, denominator: int) -> float:
    return 1.0 if denominator == 0 else numerator / denominator


def _coverage_status(value: float, denominator: int, empty_status: str) -> str:
    if denominator == 0:
        return empty_status
    return "PASS" if value >= 1.0 else "FAIL"


def _review_readiness_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    source_counts: Counter[str] = Counter()
    user_decision_counts: Counter[str] = Counter()
    agent_decision_counts: Counter[str] = Counter()
    legacy_decision_counts: Counter[str] = Counter()
    user_reviewed_count = 0
    user_source_packet_complete_count = 0
    user_approved_count = 0
    user_approved_temporal_complete_count = 0
    user_approved_source_packet_complete_count = 0
    for case in cases:
        source = _review_source(case)
        source_counts[source] += 1
        decision = str(_case_expected(case).get("review_decision") or "").strip()
        if source == READINESS_USER_REVIEW_SOURCE:
            user_reviewed_count += 1
            if decision:
                user_decision_counts[decision] += 1
            if _has_source_packet(case):
                user_source_packet_complete_count += 1
            if decision == "approve_supersession":
                user_approved_count += 1
                if _has_temporal_pair(case):
                    user_approved_temporal_complete_count += 1
                if _has_source_packet(case):
                    user_approved_source_packet_complete_count += 1
        elif source == READINESS_AGENT_SUGGESTED_SOURCE:
            if decision:
                agent_decision_counts[decision] += 1
        else:
            if decision:
                legacy_decision_counts[decision] += 1
    user_reject_or_needs_context_count = int(user_decision_counts.get("reject_not_supersession") or 0) + int(
        user_decision_counts.get("needs_more_context") or 0
    )
    return {
        "review_source_counts": dict(sorted(source_counts.items())),
        "readiness_counts": {
            "user_reviewed_case_count": user_reviewed_count,
            "user_approved_case_count": user_approved_count,
            "user_reject_or_needs_context_count": user_reject_or_needs_context_count,
            "user_decision_counts": dict(sorted(user_decision_counts.items())),
            "agent_suggested_decision_counts": dict(sorted(agent_decision_counts.items())),
            "legacy_or_unknown_decision_counts": dict(sorted(legacy_decision_counts.items())),
            "user_source_packet_complete_count": user_source_packet_complete_count,
            "user_approved_temporal_complete_count": user_approved_temporal_complete_count,
            "user_approved_source_packet_complete_count": user_approved_source_packet_complete_count,
            "source_packet_completeness": _ratio(user_source_packet_complete_count, user_reviewed_count),
            "approved_temporal_completeness": _ratio(user_approved_temporal_complete_count, user_approved_count),
            "approved_source_packet_completeness": _ratio(
                user_approved_source_packet_complete_count, user_approved_count
            ),
        },
    }


def _safe_apply_readiness(
    *,
    eval_report: dict[str, Any],
    review_extensions: dict[str, Any],
    min_reviewed_cases_for_readiness: int,
    min_approved_cases_for_readiness: int,
    min_rejected_or_needs_context_cases_for_readiness: int,
) -> dict[str, Any]:
    readiness_counts = review_extensions.get("readiness_counts") or {}
    reviewed_case_count = int(readiness_counts.get("user_reviewed_case_count") or 0)
    approved_case_count = int(readiness_counts.get("user_approved_case_count") or 0)
    reject_or_needs_context_count = int(readiness_counts.get("user_reject_or_needs_context_count") or 0)
    reasons: list[str] = []

    if review_extensions.get("errors"):
        reasons.append("review_extension_errors")
    if eval_report.get("status") != "PASS":
        reasons.append("scorecard_status_not_pass")
    if any(status == "FAIL" for status in (eval_report.get("target_status") or {}).values()):
        reasons.append("target_status_failure")
    if int((eval_report.get("metrics") or {}).get("mutation_count") or 0) != 0:
        reasons.append("mutation_count_nonzero")
    if reasons:
        status = "BLOCKED"
    else:
        if reviewed_case_count < int(min_reviewed_cases_for_readiness):
            reasons.append(
                f"reviewed_case_count_below_threshold:{reviewed_case_count}<"
                f"{int(min_reviewed_cases_for_readiness)}"
            )
        if approved_case_count < int(min_approved_cases_for_readiness):
            reasons.append(
                f"approved_case_count_below_threshold:{approved_case_count}<"
                f"{int(min_approved_cases_for_readiness)}"
            )
        if reject_or_needs_context_count < int(min_rejected_or_needs_context_cases_for_readiness):
            reasons.append(
                f"reject_or_needs_context_count_below_threshold:{reject_or_needs_context_count}<"
                f"{int(min_rejected_or_needs_context_cases_for_readiness)}"
            )
        if float(readiness_counts.get("source_packet_completeness") or 0.0) < 1.0:
            reasons.append("source_packet_completeness_below_1.0")
        if float(readiness_counts.get("approved_temporal_completeness") or 0.0) < 1.0:
            reasons.append("approved_temporal_completeness_below_1.0")
        if float(readiness_counts.get("approved_source_packet_completeness") or 0.0) < 1.0:
            reasons.append("approved_source_packet_completeness_below_1.0")
        status = "NOT_READY" if reasons else "CANDIDATE"

    return {
        "status": status,
        "claim_boundary": "Readiness is scorecard evidence only; it does not authorize memory mutation.",
        "reviewed_case_count": reviewed_case_count,
        "approved_case_count": approved_case_count,
        "reject_or_needs_context_count": reject_or_needs_context_count,
        "min_reviewed_cases_for_readiness": int(min_reviewed_cases_for_readiness),
        "min_approved_cases_for_readiness": int(min_approved_cases_for_readiness),
        "min_rejected_or_needs_context_cases_for_readiness": int(
            min_rejected_or_needs_context_cases_for_readiness
        ),
        "readiness_counts": readiness_counts,
        "reasons": reasons,
        "mutation_authorized": False,
    }


def build_scorecard(
    *,
    gold_path: Path,
    db_path: Path | None = None,
    proposal_out_dir: Path | None = None,
    max_groups: int = 200,
    max_group_size: int = 8,
    max_candidates: int = 200,
    review_extension_paths: list[Path] | None = None,
    review_extension_dirs: list[Path] | None = None,
    min_reviewed_cases_for_readiness: int = 20,
    min_approved_cases_for_readiness: int = 5,
    min_rejected_or_needs_context_cases_for_readiness: int = 5,
) -> dict[str, Any]:
    base_cases = _load_gold(gold_path)
    review_extensions = _load_review_extensions(
        paths=review_extension_paths,
        directories=review_extension_dirs,
        existing_case_ids={str(case.get("id") or "") for case in base_cases},
    )
    eval_report = evaluate_cases(base_cases + list(review_extensions["cases"]))
    safe_apply_readiness = _safe_apply_readiness(
        eval_report=eval_report,
        review_extensions=review_extensions,
        min_reviewed_cases_for_readiness=min_reviewed_cases_for_readiness,
        min_approved_cases_for_readiness=min_approved_cases_for_readiness,
        min_rejected_or_needs_context_cases_for_readiness=min_rejected_or_needs_context_cases_for_readiness,
    )
    status = eval_report["status"]
    target_status = dict(eval_report["target_status"])
    readiness_counts = review_extensions.get("readiness_counts") or {}
    user_reviewed_count = int(readiness_counts.get("user_reviewed_case_count") or 0)
    user_approved_count = int(readiness_counts.get("user_approved_case_count") or 0)
    target_status["source_packet_completeness"] = _coverage_status(
        float(readiness_counts.get("source_packet_completeness") or 0.0),
        user_reviewed_count,
        "NOT_APPLICABLE_NO_USER_REVIEWS",
    )
    target_status["approved_temporal_completeness"] = _coverage_status(
        float(readiness_counts.get("approved_temporal_completeness") or 0.0),
        user_approved_count,
        "NOT_APPLICABLE_NO_USER_APPROVALS",
    )
    target_status["approved_source_packet_completeness"] = _coverage_status(
        float(readiness_counts.get("approved_source_packet_completeness") or 0.0),
        user_approved_count,
        "NOT_APPLICABLE_NO_USER_APPROVALS",
    )
    if review_extensions["errors"]:
        status = "FAIL"
    live_summary: dict[str, Any] | None = None
    artifact_paths: dict[str, str] = {}
    if db_path and proposal_out_dir:
        with open_readonly_connection(db_path) as conn:
            proposal_report = discover_proposals(
                conn,
                max_groups=max_groups,
                max_group_size=max_group_size,
                max_candidates=max_candidates,
            )
        artifact_paths = write_report_artifacts(proposal_report, proposal_out_dir)
        proposal_report["artifact_paths"] = artifact_paths
        write_report_artifacts(proposal_report, proposal_out_dir)
        live_summary = proposal_report.get("summary") or {}

    review_extensions_summary = dict(review_extensions)
    review_extensions_summary.pop("cases", None)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "claim_boundary": CLAIM_BOUNDARY,
        "mode": "fixture_scorecard_with_review_extensions_and_optional_live_readonly_packet",
        "metrics": eval_report["metrics"],
        "target_status": target_status,
        "case_count": eval_report["case_count"],
        "split_counts": eval_report["split_counts"],
        "confusion": eval_report["confusion"],
        "review_extensions": review_extensions_summary,
        "safe_apply_readiness": safe_apply_readiness,
        "live_summary": live_summary,
        "artifact_paths": artifact_paths,
        "rows": eval_report["rows"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gold",
        type=Path,
        default=ROOT / "scripts" / "eval" / "implicit_supersession_proposals_gold.json",
    )
    parser.add_argument("--db-path", type=Path, default=None)
    parser.add_argument("--proposal-out-dir", type=Path, default=None)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--max-groups", type=int, default=200)
    parser.add_argument("--max-group-size", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=200)
    parser.add_argument("--review-extension", type=Path, action="append", default=[])
    parser.add_argument("--review-extension-dir", type=Path, action="append", default=None)
    parser.add_argument("--min-reviewed-cases-for-readiness", type=int, default=20)
    parser.add_argument("--min-approved-cases-for-readiness", type=int, default=5)
    parser.add_argument("--min-rejected-or-needs-context-cases-for-readiness", type=int, default=5)
    args = parser.parse_args()

    report = build_scorecard(
        gold_path=args.gold,
        db_path=args.db_path,
        proposal_out_dir=args.proposal_out_dir,
        max_groups=args.max_groups,
        max_group_size=args.max_group_size,
        max_candidates=args.max_candidates,
        review_extension_paths=args.review_extension,
        review_extension_dirs=args.review_extension_dir,
        min_reviewed_cases_for_readiness=args.min_reviewed_cases_for_readiness,
        min_approved_cases_for_readiness=args.min_approved_cases_for_readiness,
        min_rejected_or_needs_context_cases_for_readiness=args.min_rejected_or_needs_context_cases_for_readiness,
    )
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json or not args.json_output:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
