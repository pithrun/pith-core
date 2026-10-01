"""Read-only supersession chain quality audit utilities."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from datetime import UTC, datetime
from typing import Any

REQUIRED_COLUMNS = {
    "concepts": {
        "id",
        "summary",
        "status",
        "currency_status",
        "is_current",
        "superseded_by",
        "supersession_reason",
        "superseded_at",
        "subject_key",
        "created_at",
        "data",
    },
    "associations": {"source", "target", "relation"},
    "governance_events": {"event_type", "concept_id"},
}

SUPERSESSION_EVENTS = {
    "decision_supersession",
    "supersession_review_needed",
    "supersession_quality_degradation",
}

STRUCTURAL_FLAGS = {
    "broken_pointer",
    "self_loop",
    "cycle_detected",
    "terminal_missing",
    "depth_cap_reached",
}
LIFECYCLE_FLAGS = {
    "old_still_current",
    "old_status_not_superseded",
    "old_currency_not_superseded",
    "terminal_not_current",
    "missing_superseded_at",
    "missing_reason",
}
PARITY_FLAGS = {
    "json_status_mismatch",
    "json_currency_mismatch",
    "json_superseded_by_mismatch",
}
PROVENANCE_FLAGS = {
    "missing_supersedes_edge",
    "missing_governance_event",
}
SEMANTIC_FLAGS = {
    "replacement_thinner",
    "replacement_older",
    "low_subject_overlap",
    "missing_subject_key",
}
RETRIEVAL_FLAGS = {
    "retrieval_probe_error",
    "replacement_not_retrieved",
    "terminal_head_not_retrieved",
}

ORPHANED_SUPERSESSION_SENTINEL = "__orphaned_supersession__"
CURRENCY_PARITY_BOUNDARY = "Read-only classifier; candidate rows are not approved repairs."

STRUCTURAL_CLASS_NONE = "none"
STRUCTURAL_CLASS_ORPHAN_TOMBSTONE = "orphan_tombstone"
STRUCTURAL_CLASS_UNSAFE_ORPHAN_POINTER = "unsafe_orphan_pointer"
STRUCTURAL_CLASS_UNSAFE_ORPHAN_PATH = "unsafe_orphan_path"
STRUCTURAL_CLASS_TRUE_BROKEN_POINTER = "true_broken_pointer"
STRUCTURAL_CLASS_CYCLE_PAIR = "cycle_pair"
STRUCTURAL_CLASS_CYCLE_CHAIN = "cycle_chain"
STRUCTURAL_CLASS_LIFECYCLE_CONTRADICTION = "lifecycle_contradiction"

STRUCTURAL_SEVERITY_NONE = "none"
STRUCTURAL_SEVERITY_HISTORICAL_ORPHAN = "historical_orphan"
STRUCTURAL_SEVERITY_NEEDS_REVIEW = "needs_review"
STRUCTURAL_SEVERITY_CRITICAL = "critical"


def validate_schema(conn: sqlite3.Connection) -> None:
    """Raise ValueError if the DB lacks required audit tables or columns."""
    missing: dict[str, list[str]] = {}
    table_names = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    for table, required in REQUIRED_COLUMNS.items():
        if table not in table_names:
            missing[table] = sorted(required)
            continue
        columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        absent = sorted(required - columns)
        if absent:
            missing[table] = absent
    if missing:
        raise ValueError(f"missing required schema columns: {missing}")


def audit_supersession_chains(
    conn: sqlite3.Connection,
    *,
    max_depth: int = 64,
    max_review_examples: int = 50,
    retrieval_probe_rows: dict[str, dict[str, Any]] | None = None,
    review_mode: str = "severity-first",
) -> dict[str, Any]:
    """Audit all explicit supersession chains in an open SQLite connection."""
    validate_schema(conn)
    nodes = _load_nodes(conn)
    provenance = _load_provenance(conn)
    rows = [node for node in nodes.values() if node.get("superseded_by")]

    results: list[dict[str, Any]] = []
    retrieval_probe_rows = retrieval_probe_rows or {}
    for row in sorted(rows, key=lambda item: item["id"]):
        chain = walk_chain(row["id"], nodes, max_depth=max_depth)
        direct_replacement_id = row.get("superseded_by")
        row_provenance = {
            "has_supersedes_edge": (direct_replacement_id, row["id"]) in provenance["edges"],
            "has_governance_event": row["id"] in provenance["event_concepts"],
        }
        classified = classify_chain(row, chain, row_provenance)
        retrieval = retrieval_probe_rows.get(row["id"])
        if retrieval:
            classified["retrieval"] = retrieval
            classified["issue_flags"].extend(_retrieval_issue_flags(retrieval))
            classified["issue_flags"] = sorted(set(classified["issue_flags"]))
            classified["fitness_class"] = _fitness_for_flags(
                classified["issue_flags"],
                structural_class=classified.get("structural_class"),
            )
        results.append(classified)

    packet = build_review_packet(results, max_examples=max_review_examples, review_mode=review_mode)
    summary = summarize_audit(results, review_packet_size=len(packet))
    summary["review_mode"] = review_mode
    return {
        "schema_version": "supersession_chain_quality.v1",
        "summary": summary,
        "results": results,
        "review_packet": packet,
    }


def audit_currency_parity(conn: sqlite3.Connection) -> dict[str, Any]:
    """Classify lifecycle currency SQL/JSON drift without mutating the DB."""
    validate_schema(conn)
    nodes = _load_nodes(conn)
    rows = []
    for row in sorted(nodes.values(), key=lambda item: item["id"]):
        parsed_data, data_valid = _parse_json_object(row.get("data"))
        sql_currency = row.get("currency_status")
        json_currency = parsed_data.get("currency_status") if data_valid else None
        sql_json_mismatch = bool(
            data_valid
            and json_currency is not None
            and sql_currency is not None
            and json_currency != sql_currency
        )
        split_superseded_currency = bool(
            row.get("status") == "superseded"
            and sql_currency is not None
            and sql_currency != "SUPERSEDED"
        )
        missing_json_currency = bool(data_valid and sql_currency is not None and json_currency is None)
        invalid_json = not data_valid
        if not (
            sql_json_mismatch
            or split_superseded_currency
            or missing_json_currency
            or invalid_json
        ):
            continue
        replacement_id = row.get("superseded_by")
        replacement_exists = bool(
            replacement_id
            and (
                replacement_id in nodes
                or (
                    replacement_id == ORPHANED_SUPERSESSION_SENTINEL
                    and _is_safe_orphan_tombstone(row)
                )
            )
        )
        rows.append(
            classify_currency_parity_row(
                row,
                json_currency_status=json_currency,
                data_valid=data_valid,
                replacement_exists=replacement_exists,
            )
        )

    by_classification = Counter(row["classification"] for row in rows)
    by_sql_currency = Counter(str(row.get("currency_status") or "missing") for row in rows)
    by_json_currency = Counter(str(row.get("json_currency_status") or "missing") for row in rows)
    summary = {
        "total_rows": len(rows),
        "sql_json_currency_mismatch": sum(1 for row in rows if row["sql_json_currency_mismatch"]),
        "split_superseded_currency": sum(1 for row in rows if row["split_superseded_currency"]),
        "candidate_parity_repair": sum(1 for row in rows if row["candidate_for_reviewed_repair"]),
        "policy_review_required": by_classification.get("policy_review_required", 0),
        "missing_json_currency": by_classification.get("missing_json_currency", 0),
        "invalid_json": by_classification.get("invalid_json", 0),
        "by_sql_currency_status": dict(sorted(by_sql_currency.items())),
        "by_json_currency_status": dict(sorted(by_json_currency.items())),
        "by_classification": dict(sorted(by_classification.items())),
    }
    return {
        "schema_version": "currency_parity.v1",
        "claim_boundary": CURRENCY_PARITY_BOUNDARY,
        "generated_at": datetime.now(UTC).isoformat(),
        "summary": summary,
        "rows": rows,
    }


def classify_currency_parity_row(
    row: dict[str, Any],
    *,
    json_currency_status: str | None,
    data_valid: bool,
    replacement_exists: bool,
) -> dict[str, Any]:
    """Classify one concept row for read-only currency parity review."""
    sql_currency = row.get("currency_status")
    status = row.get("status")
    is_current = _as_int(row.get("is_current"))
    sql_json_mismatch = bool(
        data_valid
        and json_currency_status is not None
        and sql_currency is not None
        and json_currency_status != sql_currency
    )
    split_superseded_currency = bool(
        status == "superseded"
        and sql_currency is not None
        and sql_currency != "SUPERSEDED"
    )
    candidate_for_reviewed_repair = bool(
        sql_json_mismatch
        and status == "superseded"
        and is_current == 0
        and json_currency_status == "SUPERSEDED"
        and sql_currency == "CONTESTED"
        and replacement_exists
    )

    if not data_valid:
        classification = "invalid_json"
        reason = "concept data is malformed JSON; do not infer lifecycle intent"
    elif candidate_for_reviewed_repair:
        classification = "candidate_parity_repair"
        reason = "row is already historical in SQL and JSON mirror says SUPERSEDED; candidate-only for a later reviewed repair"
    elif split_superseded_currency and sql_currency in {"CONTESTED", "CONTRADICTED"} and json_currency_status == sql_currency:
        classification = "policy_review_required"
        reason = "superseded row preserves an aligned conflict currency state; do not collapse automatically"
    elif sql_json_mismatch:
        classification = "sql_json_currency_mismatch"
        reason = "SQL and JSON currency disagree, but repair policy is not proven"
    elif data_valid and sql_currency is not None and json_currency_status is None:
        classification = "missing_json_currency"
        reason = "SQL currency exists but JSON mirror lacks currency_status"
    else:
        classification = "lifecycle_currency_split_sql_json_aligned"
        reason = "superseded row has aligned non-SUPERSEDED currency outside automatic repair policy"

    return {
        "concept_id": row["id"],
        "status": status,
        "currency_status": sql_currency,
        "json_currency_status": json_currency_status,
        "is_current": is_current,
        "superseded_by": row.get("superseded_by"),
        "replacement_exists": replacement_exists,
        "classification": classification,
        "repair_eligible": False,
        "candidate_for_reviewed_repair": candidate_for_reviewed_repair,
        "reason": reason,
        "summary": row.get("summary") or "",
        "sql_json_currency_mismatch": sql_json_mismatch,
        "split_superseded_currency": split_superseded_currency,
        "claim_boundary": CURRENCY_PARITY_BOUNDARY,
    }


def walk_chain(root_id: str, nodes: dict[str, dict[str, Any]], *, max_depth: int = 64) -> dict[str, Any]:
    """Follow superseded_by pointers from root and detect structural problems."""
    visited: set[str] = set()
    path: list[str] = []
    current_id = root_id
    root_node = nodes.get(root_id)
    direct_replacement_id = root_node.get("superseded_by") if root_node else None
    flags: list[str] = []
    terminal_id: str | None = None

    for depth in range(max_depth + 1):
        if current_id in visited:
            flags.append("cycle_detected")
            terminal_id = current_id
            break
        node = nodes.get(current_id)
        if node is None:
            flags.append("terminal_missing" if depth > 0 else "broken_pointer")
            terminal_id = current_id
            break
        visited.add(current_id)
        path.append(current_id)
        next_id = node.get("superseded_by")
        if not next_id:
            terminal_id = current_id
            break
        if next_id == current_id:
            flags.append("self_loop")
            terminal_id = current_id
            break
        if next_id not in nodes:
            flags.append("broken_pointer")
            terminal_id = next_id
            break
        current_id = next_id
    else:
        flags.append("depth_cap_reached")
        terminal_id = current_id

    terminal_node = nodes.get(terminal_id) if terminal_id else None
    return {
        "path": path,
        "depth": max(0, len(path) - 1),
        "direct_replacement": nodes.get(direct_replacement_id) if direct_replacement_id else None,
        "terminal_head_id": terminal_id,
        "terminal": terminal_node,
        "structural_flags": sorted(set(flags)),
    }


def classify_chain(
    row: dict[str, Any],
    chain: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Classify one supersession chain into a fitness class and issue flags."""
    flags: list[str] = list(chain.get("structural_flags") or [])
    replacement = chain.get("direct_replacement")
    direct_replacement_id = row.get("superseded_by")

    if _as_int(row.get("is_current")) == 1:
        flags.append("old_still_current")
    if row.get("status") != "superseded":
        flags.append("old_status_not_superseded")
    if row.get("currency_status") != "SUPERSEDED":
        flags.append("old_currency_not_superseded")
    if not row.get("superseded_at"):
        flags.append("missing_superseded_at")
    if not row.get("supersession_reason"):
        flags.append("missing_reason")

    terminal = chain.get("terminal")
    if terminal and (_as_int(terminal.get("is_current")) != 1 or terminal.get("status") != "active"):
        flags.append("terminal_not_current")

    flags.extend(_json_parity_flags(row))

    if not provenance.get("has_supersedes_edge"):
        flags.append("missing_supersedes_edge")
    if not provenance.get("has_governance_event"):
        flags.append("missing_governance_event")

    if replacement:
        flags.extend(_semantic_flags(row, replacement))
    else:
        flags.append("broken_pointer")

    issue_flags = sorted(set(flags))
    structural_class = _structural_class(row, chain, set(issue_flags))
    structural_severity = _structural_severity(structural_class)
    return {
        "old_id": row["id"],
        "replacement_id": direct_replacement_id,
        "terminal_head_id": chain.get("terminal_head_id"),
        "depth": chain.get("depth", 0),
        "old_created_at": row.get("created_at"),
        "replacement_created_at": (replacement or {}).get("created_at"),
        "terminal_created_at": (terminal or {}).get("created_at") if terminal else None,
        "superseded_at": row.get("superseded_at"),
        "supersession_reason": row.get("supersession_reason"),
        "old_status": row.get("status"),
        "old_currency_status": row.get("currency_status"),
        "replacement_status": (replacement or {}).get("status"),
        "replacement_currency_status": (replacement or {}).get("currency_status"),
        "terminal_status": (terminal or {}).get("status") if terminal else None,
        "terminal_currency_status": (terminal or {}).get("currency_status") if terminal else None,
        "old_summary": row.get("summary") or "",
        "replacement_summary": (replacement or {}).get("summary") or "",
        "terminal_summary": (terminal or {}).get("summary") or "",
        "old_subject_key": row.get("subject_key"),
        "replacement_subject_key": (replacement or {}).get("subject_key"),
        "structural_class": structural_class,
        "structural_severity": structural_severity,
        "issue_flags": issue_flags,
        "fitness_class": _fitness_for_flags(issue_flags, structural_class=structural_class),
        "retrieval": None,
    }


def summarize_audit(results: list[dict[str, Any]], *, review_packet_size: int = 0) -> dict[str, Any]:
    """Aggregate chain audit rows into stable metrics."""
    by_class = Counter(row["fitness_class"] for row in results)
    by_issue = Counter(flag for row in results for flag in row["issue_flags"])
    by_structural_class = Counter(str(row.get("structural_class") or STRUCTURAL_CLASS_NONE) for row in results)
    by_structural_severity = Counter(
        str(row.get("structural_severity") or STRUCTURAL_SEVERITY_NONE) for row in results
    )
    by_dimension = {
        "structural": sum(1 for row in results if STRUCTURAL_FLAGS & set(row["issue_flags"])),
        "lifecycle": sum(1 for row in results if LIFECYCLE_FLAGS & set(row["issue_flags"])),
        "sql_json_parity": sum(1 for row in results if PARITY_FLAGS & set(row["issue_flags"])),
        "provenance": sum(1 for row in results if PROVENANCE_FLAGS & set(row["issue_flags"])),
        "semantic_heuristic": sum(1 for row in results if SEMANTIC_FLAGS & set(row["issue_flags"])),
        "retrieval_observability": sum(1 for row in results if row.get("retrieval")),
    }
    sentinel_orphan_counts = {
        "total": by_structural_class.get(STRUCTURAL_CLASS_ORPHAN_TOMBSTONE, 0)
        + by_structural_class.get(STRUCTURAL_CLASS_UNSAFE_ORPHAN_POINTER, 0),
        "safe": by_structural_class.get(STRUCTURAL_CLASS_ORPHAN_TOMBSTONE, 0),
        "unsafe": by_structural_class.get(STRUCTURAL_CLASS_UNSAFE_ORPHAN_POINTER, 0),
        "unsafe_path": by_structural_class.get(STRUCTURAL_CLASS_UNSAFE_ORPHAN_PATH, 0),
    }
    true_structural_counts = {
        "broken_pointer": by_structural_class.get(STRUCTURAL_CLASS_TRUE_BROKEN_POINTER, 0),
        "unsafe_orphan_pointer": by_structural_class.get(STRUCTURAL_CLASS_UNSAFE_ORPHAN_POINTER, 0),
        "unsafe_orphan_path": by_structural_class.get(STRUCTURAL_CLASS_UNSAFE_ORPHAN_PATH, 0),
        "cycle_detected": by_structural_class.get(STRUCTURAL_CLASS_CYCLE_PAIR, 0)
        + by_structural_class.get(STRUCTURAL_CLASS_CYCLE_CHAIN, 0),
        "old_still_current": sum(
            1
            for row in results
            if "old_still_current" in set(row.get("issue_flags") or [])
            and row.get("structural_class") != STRUCTURAL_CLASS_ORPHAN_TOMBSTONE
        ),
    }
    total = len(results)
    return {
        "total_chains": total,
        "fitness_counts": dict(sorted(by_class.items())),
        "fitness_rates": {
            key: (value / total if total else 0.0)
            for key, value in sorted(by_class.items())
        },
        "issue_counts": dict(sorted(by_issue.items())),
        "dimension_counts": by_dimension,
        "structural_class_counts": dict(sorted(by_structural_class.items())),
        "structural_severity_counts": dict(sorted(by_structural_severity.items())),
        "sentinel_orphan_counts": sentinel_orphan_counts,
        "true_structural_counts": true_structural_counts,
        "review_packet_size": review_packet_size,
        "max_depth": max((int(row.get("depth") or 0) for row in results), default=0),
    }


REVIEW_MODES = {"severity-first", "stratified", "representative"}


def build_review_packet(
    results: list[dict[str, Any]],
    *,
    max_examples: int = 50,
    review_mode: str = "severity-first",
) -> list[dict[str, Any]]:
    """Return deterministic examples for human review."""
    if review_mode not in REVIEW_MODES:
        raise ValueError(f"unsupported review_mode: {review_mode}")
    if max_examples < 1:
        return []
    if review_mode == "severity-first":
        risk_order = {
            "exclude": 0,
            "repair_needed": 1,
            "candidate_only": 2,
            "trust_label_candidate": 3,
            "historical_orphan": 4,
        }
        ordered = sorted(
            results,
            key=lambda item: (
                risk_order.get(item["fitness_class"], 99),
                -len(item.get("issue_flags") or []),
                item["old_id"],
            ),
        )
        return [_packet_row(row) for row in ordered[:max_examples]]
    if review_mode == "representative":
        return _representative_review_packet(results, max_examples=max_examples)

    buckets = {
        "trust_label_candidate": [],
        "candidate_only": [],
        "repair_needed": [],
        "exclude": [],
        "historical_orphan": [],
    }
    for row in sorted(results, key=lambda item: (item["fitness_class"], item["old_id"])):
        buckets.setdefault(row["fitness_class"], []).append(row)

    selected: list[dict[str, Any]] = []
    order = ["trust_label_candidate", "candidate_only", "repair_needed", "exclude", "historical_orphan"]
    while len(selected) < max_examples and any(buckets.values()):
        progressed = False
        for fitness_class in order:
            bucket = buckets.get(fitness_class) or []
            if bucket and len(selected) < max_examples:
                selected.append(_packet_row(bucket.pop(0)))
                progressed = True
        if not progressed:
            break
    return selected


def _representative_review_packet(results: list[dict[str, Any]], *, max_examples: int) -> list[dict[str, Any]]:
    issue_families = [
        STRUCTURAL_FLAGS,
        LIFECYCLE_FLAGS,
        PARITY_FLAGS,
        PROVENANCE_FLAGS,
        SEMANTIC_FLAGS,
        RETRIEVAL_FLAGS,
    ]
    class_order = {
        "trust_label_candidate": 0,
        "candidate_only": 1,
        "repair_needed": 2,
        "exclude": 3,
        "historical_orphan": 4,
    }
    ordered = sorted(
        results,
        key=lambda item: (
            class_order.get(item["fitness_class"], 99),
            -len(item.get("issue_flags") or []),
            item["old_id"],
        ),
    )
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def add(row: dict[str, Any]) -> None:
        selected.append(_packet_row(row))
        selected_ids.add(row["old_id"])

    for family in issue_families:
        if len(selected) >= max_examples:
            break
        for row in ordered:
            if row["old_id"] not in selected_ids and family & set(row.get("issue_flags") or []):
                add(row)
                break

    buckets: dict[str, list[dict[str, Any]]] = {
        "trust_label_candidate": [],
        "candidate_only": [],
        "repair_needed": [],
        "exclude": [],
        "historical_orphan": [],
    }
    for row in ordered:
        if row["old_id"] not in selected_ids:
            buckets.setdefault(row["fitness_class"], []).append(row)

    while len(selected) < max_examples and any(buckets.values()):
        progressed = False
        for fitness_class in ("trust_label_candidate", "candidate_only", "repair_needed", "exclude", "historical_orphan"):
            bucket = buckets.get(fitness_class) or []
            if bucket and len(selected) < max_examples:
                add(bucket.pop(0))
                progressed = True
        if not progressed:
            break
    return selected


def render_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Supersession Chain Quality Summary",
        "",
        f"- Total chains: {summary['total_chains']}",
        f"- Review packet size: {summary['review_packet_size']}",
        f"- Max depth: {summary['max_depth']}",
        "",
        "## Fitness Counts",
    ]
    for key, value in summary["fitness_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Dimension Counts"])
    for key, value in summary["dimension_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Issue Counts"])
    for key, value in summary["issue_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Structural Class Counts"])
    for key, value in (summary.get("structural_class_counts") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## True Structural Counts"])
    for key, value in (summary.get("true_structural_counts") or {}).items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def render_review_packet_markdown(
    packet: list[dict[str, Any]],
    *,
    generated_at: str | None = None,
    summary: dict[str, Any] | None = None,
    review_mode: str = "severity-first",
) -> str:
    lines = [
        "# Supersession Chain Review Packet",
        "",
        f"Generated at: {generated_at or 'not recorded'}",
        "",
        "Generated classifications are candidate labels, not truth.",
        "",
        "Quick decision options: approve_gold, candidate_only, repair_needed, exclude, uncertain.",
        "",
        "Decision guide:",
        "- approve_gold: chain is correct, current item genuinely governs the old item.",
        "- repair_needed: chain is directionally right, but metadata or links need deterministic repair.",
        "- candidate_only: plausible but not strong enough to govern without more evidence.",
        "- exclude: wrong, structurally broken, or unsafe to use as trust evidence.",
        "- uncertain: needs more context before promotion or repair.",
        "",
    ]
    lines.extend(_review_packet_header(summary or {}, review_mode=review_mode))
    for index, row in enumerate(packet, start=1):
        row_lines = [
            f"## {index}. {_decision_label(row)}",
            "",
            f"**Suggested QC:** `{_suggested_qc(row)}`",
            "",
            f"- Old: `{row['old_id']}` ({_status_label(row, 'old')})",
            f"- Replacement: `{row.get('replacement_id') or 'missing'}` ({_status_label(row, 'replacement')})",
            f"- Terminal head: `{row.get('terminal_head_id') or 'missing'}` ({_status_label(row, 'terminal')})",
            f"- Depth: {row.get('depth')}",
            f"- Structural class: `{row.get('structural_class') or STRUCTURAL_CLASS_NONE}` ({row.get('structural_severity') or STRUCTURAL_SEVERITY_NONE})",
            f"- Issues: {_issue_summary(row.get('issue_flags') or [])}",
            f"- Timeline: {_timeline(row)}",
            f"- Supersession reason: {row.get('supersession_reason') or 'missing'}",
            f"- Subject: old `{row.get('old_subject_key') or 'missing'}` -> replacement `{row.get('replacement_subject_key') or 'missing'}`",
            f"- Retrieval probe (packet-sample only): {_retrieval_summary(row.get('retrieval'))}",
            "",
            "**Old source excerpt**",
            "",
            f"> {row.get('old_summary') or '[missing]'}",
            "",
            "**Replacement source excerpt**",
            "",
            f"> {row.get('replacement_summary') or '[missing]'}",
            "",
        ]
        terminal_id = row.get("terminal_head_id")
        if terminal_id and terminal_id != row.get("replacement_id"):
            row_lines.extend(
                [
                    "**Terminal head source excerpt**",
                    "",
                    f"> {row.get('terminal_summary') or '[missing]'}",
                    "",
                ]
            )
        row_lines.extend(
            [
                "**Your QC:** [ ] approve_gold  [ ] candidate_only  [ ] repair_needed  [ ] exclude  [ ] uncertain",
                "",
                "**Notes:**",
                "",
            ]
        )
        lines.extend(row_lines)
    return "\n".join(lines)


def _load_nodes(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    rows = _fetch_dicts(
        conn,
        """SELECT id, summary, status, currency_status, is_current,
                  superseded_by, supersession_reason, superseded_at,
                  subject_key, created_at, data
           FROM concepts""",
    )
    return {row["id"]: row for row in rows}


def _load_provenance(conn: sqlite3.Connection) -> dict[str, Any]:
    edge_rows = _fetch_dicts(conn, "SELECT source, target FROM associations WHERE relation = 'supersedes'")
    event_rows = _fetch_dicts(
        conn,
        f"""SELECT concept_id FROM governance_events
            WHERE event_type IN ({",".join("?" for _ in SUPERSESSION_EVENTS)})
              AND concept_id IS NOT NULL""",
        tuple(sorted(SUPERSESSION_EVENTS)),
    )
    return {
        "edges": {(row["source"], row["target"]) for row in edge_rows},
        "event_concepts": {row["concept_id"] for row in event_rows},
    }


def _fetch_dicts(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    cursor = conn.execute(sql, params)
    columns = [item[0] for item in cursor.description]
    return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def _json_parity_flags(row: dict[str, Any]) -> list[str]:
    data = _safe_json(row.get("data"))
    flags: list[str] = []
    if not isinstance(data, dict):
        return flags
    if data.get("status") is not None and data.get("status") != row.get("status"):
        flags.append("json_status_mismatch")
    if data.get("currency_status") is not None and data.get("currency_status") != row.get("currency_status"):
        flags.append("json_currency_mismatch")
    if data.get("superseded_by") is not None and data.get("superseded_by") != row.get("superseded_by"):
        flags.append("json_superseded_by_mismatch")
    return flags


def _semantic_flags(row: dict[str, Any], replacement: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    old_summary = row.get("summary") or ""
    new_summary = replacement.get("summary") or ""
    if old_summary and len(new_summary) < (len(old_summary) / 2):
        flags.append("replacement_thinner")
    if _compare_temporal_stamps(replacement.get("created_at"), row.get("created_at")) == -1:
        flags.append("replacement_older")
    old_subject = row.get("subject_key")
    new_subject = replacement.get("subject_key")
    if not old_subject or not new_subject:
        flags.append("missing_subject_key")
    elif old_subject != new_subject:
        flags.append("low_subject_overlap")
    return flags


def _retrieval_issue_flags(retrieval: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    if retrieval.get("retrieval_probe_error"):
        flags.append("retrieval_probe_error")
    if retrieval.get("replacement_rank") is None:
        flags.append("replacement_not_retrieved")
    if retrieval.get("terminal_rank") is None:
        flags.append("terminal_head_not_retrieved")
    return flags


def _is_safe_orphan_tombstone(row: dict[str, Any]) -> bool:
    return (
        row.get("superseded_by") == ORPHANED_SUPERSESSION_SENTINEL
        and _as_int(row.get("is_current")) == 0
        and row.get("status") == "superseded"
        and row.get("currency_status") == "SUPERSEDED"
        and bool(row.get("superseded_at"))
        and bool(row.get("supersession_reason"))
    )


def _structural_class(row: dict[str, Any], chain: dict[str, Any], flags: set[str]) -> str:
    if row.get("superseded_by") == ORPHANED_SUPERSESSION_SENTINEL:
        if _is_safe_orphan_tombstone(row):
            return STRUCTURAL_CLASS_ORPHAN_TOMBSTONE
        return STRUCTURAL_CLASS_UNSAFE_ORPHAN_POINTER
    if chain.get("terminal_head_id") == ORPHANED_SUPERSESSION_SENTINEL:
        return STRUCTURAL_CLASS_UNSAFE_ORPHAN_PATH
    if "cycle_detected" in flags:
        return STRUCTURAL_CLASS_CYCLE_PAIR if len(chain.get("path") or []) == 2 else STRUCTURAL_CLASS_CYCLE_CHAIN
    if flags & {"broken_pointer", "terminal_missing", "depth_cap_reached", "self_loop"}:
        return STRUCTURAL_CLASS_TRUE_BROKEN_POINTER
    if flags & LIFECYCLE_FLAGS:
        return STRUCTURAL_CLASS_LIFECYCLE_CONTRADICTION
    return STRUCTURAL_CLASS_NONE


def _structural_severity(structural_class: str) -> str:
    if structural_class == STRUCTURAL_CLASS_NONE:
        return STRUCTURAL_SEVERITY_NONE
    if structural_class == STRUCTURAL_CLASS_ORPHAN_TOMBSTONE:
        return STRUCTURAL_SEVERITY_HISTORICAL_ORPHAN
    if structural_class == STRUCTURAL_CLASS_LIFECYCLE_CONTRADICTION:
        return STRUCTURAL_SEVERITY_NEEDS_REVIEW
    return STRUCTURAL_SEVERITY_CRITICAL


def _fitness_for_flags(flags: list[str], *, structural_class: str | None = None) -> str:
    if structural_class == STRUCTURAL_CLASS_ORPHAN_TOMBSTONE:
        return "historical_orphan"
    flag_set = set(flags)
    if flag_set & STRUCTURAL_FLAGS:
        return "exclude"
    if flag_set & (LIFECYCLE_FLAGS | PARITY_FLAGS):
        return "repair_needed"
    if flag_set & (PROVENANCE_FLAGS | SEMANTIC_FLAGS | RETRIEVAL_FLAGS):
        return "candidate_only"
    return "trust_label_candidate"


def _packet_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "old_id": row["old_id"],
        "replacement_id": row.get("replacement_id"),
        "terminal_head_id": row.get("terminal_head_id"),
        "depth": row.get("depth"),
        "old_created_at": row.get("old_created_at"),
        "replacement_created_at": row.get("replacement_created_at"),
        "terminal_created_at": row.get("terminal_created_at"),
        "superseded_at": row.get("superseded_at"),
        "supersession_reason": row.get("supersession_reason"),
        "old_status": row.get("old_status"),
        "old_currency_status": row.get("old_currency_status"),
        "replacement_status": row.get("replacement_status"),
        "replacement_currency_status": row.get("replacement_currency_status"),
        "terminal_status": row.get("terminal_status"),
        "terminal_currency_status": row.get("terminal_currency_status"),
        "structural_class": row.get("structural_class"),
        "structural_severity": row.get("structural_severity"),
        "fitness_class": row["fitness_class"],
        "issue_flags": row.get("issue_flags", []),
        "old_subject_key": row.get("old_subject_key"),
        "replacement_subject_key": row.get("replacement_subject_key"),
        "old_summary": _truncate(row.get("old_summary") or ""),
        "replacement_summary": _truncate(row.get("replacement_summary") or ""),
        "terminal_summary": _truncate(row.get("terminal_summary") or ""),
        "retrieval": row.get("retrieval"),
    }


def _truncate(value: str, limit: int = 240) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _decision_label(row: dict[str, Any]) -> str:
    old_id = row.get("old_id") or "missing"
    replacement_id = row.get("replacement_id") or "missing"
    return f"{row.get('fitness_class')} - {old_id} -> {replacement_id}"


def _suggested_qc(row: dict[str, Any]) -> str:
    fitness_class = row.get("fitness_class")
    if fitness_class == "trust_label_candidate":
        return "approve_gold if the summaries describe the same subject and the replacement truly governs"
    if fitness_class == "repair_needed":
        return "repair_needed unless the relationship itself is wrong"
    if fitness_class == "exclude":
        return "exclude unless the structural issue is a known recoverable artifact"
    if fitness_class == "historical_orphan":
        return "no action if this is a safe historical orphan tombstone"
    return "candidate_only unless you can verify a governing replacement"


def _status_label(row: dict[str, Any], prefix: str) -> str:
    status = row.get(f"{prefix}_status") or "missing"
    currency = row.get(f"{prefix}_currency_status") or "missing"
    return f"status={status}, currency={currency}"


def _issue_summary(flags: list[str]) -> str:
    if not flags:
        return "none"
    grouped = {
        "structural": sorted(set(flags) & STRUCTURAL_FLAGS),
        "lifecycle": sorted(set(flags) & LIFECYCLE_FLAGS),
        "parity": sorted(set(flags) & PARITY_FLAGS),
        "provenance": sorted(set(flags) & PROVENANCE_FLAGS),
        "semantic": sorted(set(flags) & SEMANTIC_FLAGS),
        "retrieval": sorted(set(flags) & RETRIEVAL_FLAGS),
    }
    parts = []
    for group, group_flags in grouped.items():
        if group_flags:
            parts.append(f"{group}: {', '.join(group_flags)}")
    return "; ".join(parts)


def _timeline(row: dict[str, Any]) -> str:
    old_created = row.get("old_created_at") or "old_created_at missing"
    superseded_at = row.get("superseded_at") or "superseded_at missing"
    replacement_created = row.get("replacement_created_at") or "replacement_created_at missing"
    terminal_created = row.get("terminal_created_at") or "terminal_created_at missing"
    order_note = _temporal_order_note(row)
    return (
        f"old created {old_created} | superseded {superseded_at} | "
        f"replacement created {replacement_created} | terminal created {terminal_created} | {order_note}"
    )


def _temporal_order_note(row: dict[str, Any]) -> str:
    old_created = row.get("old_created_at")
    replacement_created = row.get("replacement_created_at")
    superseded_at = row.get("superseded_at")
    notes: list[str] = []
    if old_created and replacement_created:
        created_order = _compare_temporal_stamps(replacement_created, old_created)
        if created_order == -1:
            notes.append("replacement predates old item")
        elif created_order == 0:
            notes.append("replacement has same creation timestamp")
        elif created_order == 1:
            notes.append("replacement is newer than old item")
        else:
            notes.append("replacement/old temporal order unverified")
    if superseded_at and replacement_created and _compare_temporal_stamps(superseded_at, replacement_created) == -1:
        notes.append("superseded_at predates replacement creation")
    return "; ".join(notes) or "temporal order unknown"


def _review_packet_header(summary: dict[str, Any], *, review_mode: str) -> list[str]:
    if not summary:
        return [
            "## Review Triage",
            "",
            f"- Review mode: {review_mode}",
            "- Snapshot freshness: unavailable; regenerate before review if this packet is not current.",
            "- Retrieval probe scope: packet-sample only.",
            "",
        ]
    generated_at = summary.get("generated_at") or "not recorded"
    valid_until = summary.get("review_valid_until") or "not recorded"
    lines = [
        "## Review Triage",
        "",
        f"- Source snapshot as of: {summary.get('source_snapshot_as_of') or generated_at}",
        f"- Review valid until: {valid_until}",
        f"- Freshness warning: Regenerate this packet before review if now is after {valid_until}.",
        f"- Source core unchanged during run: {summary.get('source_chain_core_unchanged', 'unknown')}",
        f"- Observed fingerprint unchanged during run: {summary.get('source_observed_fingerprint_unchanged', 'unknown')}",
        f"- Total chains: {summary.get('total_chains', 'unknown')}",
        f"- Packet rows: {summary.get('review_packet_size', 'unknown')}",
        f"- Review mode: {summary.get('review_mode') or review_mode}",
        f"- Retrieval probe status: {summary.get('review_packet_retrieval_probe_status') or summary.get('retrieval_observability_status') or 'unknown'}",
        f"- Retrieval probe scope: {summary.get('review_packet_retrieval_probe_scope') or 'packet_sample'}; packet-sample only, not all chains.",
        "- Full row payload: available only when the CLI is run with --include-row-details.",
        "",
        "### Fitness Counts",
    ]
    for key, value in (summary.get("fitness_counts") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "### Top Issue Counts"])
    issue_counts = sorted((summary.get("issue_counts") or {}).items(), key=lambda item: (-item[1], item[0]))
    for key, value in issue_counts[:8]:
        lines.append(f"- {key}: {value}")
    if not issue_counts:
        lines.append("- none")
    lines.append("")
    return lines


def _parse_temporal_stamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _compare_temporal_stamps(left: Any, right: Any) -> int | None:
    left_dt = _parse_temporal_stamp(left)
    right_dt = _parse_temporal_stamp(right)
    if left_dt is None or right_dt is None:
        return None
    if left_dt < right_dt:
        return -1
    if left_dt > right_dt:
        return 1
    return 0


def _retrieval_summary(retrieval: dict[str, Any] | None) -> str:
    if not retrieval:
        return "not run"
    if retrieval.get("retrieval_probe_error"):
        return f"error: {retrieval['retrieval_probe_error']}"
    return (
        f"old rank={_rank_label(retrieval.get('old_rank'))}, "
        f"replacement rank={_rank_label(retrieval.get('replacement_rank'))}, "
        f"terminal rank={_rank_label(retrieval.get('terminal_rank'))}, "
        f"query={_truncate(str(retrieval.get('query') or ''), 120)!r}"
    )


def _rank_label(value: Any) -> str:
    return str(value) if value is not None else "not in top 10"


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}


def _parse_json_object(value: Any) -> tuple[dict[str, Any], bool]:
    if isinstance(value, dict):
        return value, True
    if not value:
        return {}, True
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}, False
    return (parsed, True) if isinstance(parsed, dict) else ({}, False)


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
