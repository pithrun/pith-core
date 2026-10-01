#!/usr/bin/env python3
"""Evaluate implicit supersession proposal classification against curated cases."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ops.implicit_supersession_proposals import CLAIM_BOUNDARY, classify_pair

SCHEMA_VERSION = "implicit_supersession_proposals_eval.v1"
TARGETS = {
    "review_precision": 0.85,
    "review_recall": 0.60,
    "false_link_rate": 0.05,
    "evidence_completeness": 0.95,
    "temporal_completeness": 0.95,
    "no_known_correctness": 0.90,
}


def evaluate_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    true_positive = false_positive = false_negative = true_negative = 0
    no_known_total = no_known_correct = 0
    evidence_total = evidence_complete = 0
    temporal_total = temporal_complete = 0

    for case in cases:
        result = classify_pair(case["old"], case["new"])
        expected = case["expected"]
        expected_proposal = bool(expected["proposal"])
        observed_proposal = bool(result["proposal"])
        if observed_proposal and expected_proposal:
            true_positive += 1
        elif observed_proposal and not expected_proposal:
            false_positive += 1
        elif not observed_proposal and expected_proposal:
            false_negative += 1
        else:
            true_negative += 1

        if case.get("split") == "no_known_authority":
            no_known_total += 1
            if result["disposition"] in {"needs_context", "not_supersession"}:
                no_known_correct += 1

        if expected.get("requires_evidence", True):
            evidence_total += 1
            if result["evidence_complete"]:
                evidence_complete += 1

        if expected.get("requires_temporal", True):
            temporal_total += 1
            if result.get("old_created_at") and result.get("new_created_at"):
                temporal_complete += 1

        split_counts[case.get("split", "unknown")] += 1
        rows.append(
            {
                "id": case["id"],
                "split": case.get("split", "unknown"),
                "expected_disposition": expected["disposition"],
                "observed_disposition": result["disposition"],
                "expected_proposal": expected_proposal,
                "observed_proposal": observed_proposal,
                "correct": (
                    result["disposition"] == expected["disposition"]
                    and observed_proposal == expected_proposal
                ),
                "signals": result["signals"],
                "score": result["score"],
                "evidence_complete": result["evidence_complete"],
            }
        )

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    metrics = {
        "review_precision": true_positive / precision_denominator if precision_denominator else 1.0,
        "review_recall": true_positive / recall_denominator if recall_denominator else 1.0,
        "false_link_rate": false_positive / len(cases) if cases else 0.0,
        "evidence_completeness": evidence_complete / evidence_total if evidence_total else 1.0,
        "temporal_completeness": temporal_complete / temporal_total if temporal_total else 1.0,
        "no_known_correctness": no_known_correct / no_known_total if no_known_total else 1.0,
        "case_accuracy": sum(1 for row in rows if row["correct"]) / len(cases) if cases else 0.0,
        "latency_ms": {"total": (time.perf_counter() - started) * 1000.0},
        "mutation_count": 0,
    }
    target_status = {
        "review_precision": "PASS" if metrics["review_precision"] >= TARGETS["review_precision"] else "FAIL",
        "review_recall": "PASS" if metrics["review_recall"] >= TARGETS["review_recall"] else "FAIL",
        "false_link_rate": "PASS" if metrics["false_link_rate"] <= TARGETS["false_link_rate"] else "FAIL",
        "evidence_completeness": (
            "PASS" if metrics["evidence_completeness"] >= TARGETS["evidence_completeness"] else "FAIL"
        ),
        "temporal_completeness": (
            "PASS" if metrics["temporal_completeness"] >= TARGETS["temporal_completeness"] else "FAIL"
        ),
        "no_known_correctness": (
            "PASS" if metrics["no_known_correctness"] >= TARGETS["no_known_correctness"] else "FAIL"
        ),
        "mutation_count": "PASS" if metrics["mutation_count"] == 0 else "FAIL",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if all(value == "PASS" for value in target_status.values()) else "FAIL",
        "claim_boundary": CLAIM_BOUNDARY,
        "targets": TARGETS,
        "metrics": metrics,
        "target_status": target_status,
        "case_count": len(cases),
        "split_counts": dict(sorted(split_counts.items())),
        "confusion": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
        },
        "rows": rows,
    }


def _load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload["cases"] if isinstance(payload, dict) else payload
    if len(cases) < 60:
        raise ValueError(f"expected at least 60 gold cases, found {len(cases)}")
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path(__file__).with_name("implicit_supersession_proposals_gold.json"),
    )
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = evaluate_cases(_load_cases(args.gold))
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json or not args.json_output:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
