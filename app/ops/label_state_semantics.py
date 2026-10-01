"""Read-only label-state semantics resolver for concept currentness."""

from __future__ import annotations

import hashlib
import json
import shlex
import shutil
import sqlite3
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CLAIM_BOUNDARY = "Read-only label-state semantics audit; not repair approval."
RESOLVER_PRECEDENCE_VERSION = "label_state_semantics.v1"
ORPHAN_SENTINEL = "__orphaned_supersession__"

CURRENT_CURRENCY = {"ACTIVE", "CURRENT"}
RISK_CURRENCY_BY_STATE = {
    "CONTRADICTED": "current_risk_contradicted",
    "CONTESTED": "current_risk_contested",
    "STALE": "current_risk_stale",
}
SUPERSEDED_CURRENCY = {"SUPERSEDED", "DEPRECATED"}
HISTORICAL_STATUSES = {"archived", "deleted", "inactive"}
SUPERSEDED_STATUSES = {"superseded", "deprecated"}
LIFECYCLE_CONFLICT_STATUSES = {"archived", "deleted", "inactive"}
REQUIRED_CONCEPT_COLUMNS = {
    "id",
    "status",
    "currency_status",
    "data",
    "is_current",
    "superseded_by",
    "superseded_at",
    "subject_key",
    "staleness_state",
    "summary",
    "created_at",
    "updated_at",
    "content_updated_at",
    "last_accessed",
    "valid_from",
    "valid_until",
}
SIX_REPAIRED_ROW_IDS = (
    "conv_4d025ca1b23a",
    "conv_5aab0874f499",
    "conv_7c73fed3cccb",
    "conv_7dbe485671aa",
    "conv_c6c20d9b6689",
    "conv_ffc2731927f6",
)
ALL_RESOLVED_STATES = (
    "current_clean",
    "current_risk_stale",
    "current_risk_contested",
    "current_risk_contradicted",
    "inconsistent_current_label",
    "current_lifecycle_conflict",
    "superseded_tail",
    "edge_current_conflict",
    "edge_label_conflict",
    "orphaned_edge",
    "historical_no_edge",
    "storage_desync",
    "ambiguous",
)
OPERATOR_PACKET_SCHEMA_VERSION = "label_state_semantics_operator_packet.v1"
REVIEW_PACKET_SCHEMA_VERSION = "label_state_semantics_review_packet.v1"
REVIEW_DECISIONS_SCHEMA_VERSION = "label_state_semantics_review_decisions.v1"
DRY_RUN_REPAIR_PLAN_SCHEMA_VERSION = "label_state_semantics_dry_run_repair_plan.v1"
LIVE_REPAIR_APPLY_SCHEMA_VERSION = "label_state_semantics_live_repair_apply.v1"
LIVE_REPAIR_APPLY_PLAN_SCHEMA_VERSION = "label_state_semantics_live_repair_apply_plan.v1"
LIVE_APPLY_ALLOWED_ACTIONS = {"review_edge_state"}
DEFAULT_LIVE_APPLY_BACKUP_RETENTION = 2
MIN_BACKUP_FREE_BYTES = 1_073_741_824
MIN_BACKUP_FREE_RATIO = 2.5
APPROVE_DECISION = "approve_dry_run"
REJECT_DECISION = "reject"
DEFER_DECISION = "defer"
ALLOWED_REVIEW_DECISIONS = {APPROVE_DECISION, REJECT_DECISION, DEFER_DECISION}
ACTION_LABELS = {
    "review_label_state": "Review label-state evidence before repair.",
    "review_edge_state": "Review supersession edge evidence before repair.",
    "recover_identity_evidence": "Find replacement identity/source evidence before repair.",
    "review_storage_desync": "Resolve SQL/JSON storage disagreement before repair.",
    "manual_review": "Manually inspect rows the resolver cannot classify.",
}

READ_SQL = """
SELECT
  c.id, c.status, c.currency_status, c.data, c.is_current,
  c.superseded_by, c.superseded_at, c.subject_key, c.staleness_state,
  c.created_at, c.updated_at, c.content_updated_at, c.last_accessed,
  c.valid_from, c.valid_until,
  substr(c.summary, 1, 240) AS summary_preview,
  CASE
    WHEN c.superseded_by IS NULL OR trim(c.superseded_by) = '' THEN NULL
    WHEN c.superseded_by = '__orphaned_supersession__' THEN 0
    WHEN r.id IS NULL THEN 0
    ELSE 1
  END AS replacement_exists
FROM concepts c
LEFT JOIN concepts r ON r.id = c.superseded_by
ORDER BY c.id
"""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _coerce_str(value: object) -> str:
    return str(value or "").strip()


def _coerce_bool_int(value: object) -> bool | None:
    if value is None:
        return None
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return None


def _parse_json_object(value: object) -> tuple[dict[str, Any], bool]:
    if not value:
        return {}, False
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return {}, True
    if isinstance(parsed, dict):
        return parsed, False
    return {}, True


def _row_get(row: Mapping[str, Any], key: str, default: Any = None) -> Any:
    try:
        return row[key]
    except (KeyError, IndexError):
        return default


def _repairability_for_state(state: str) -> tuple[str, str]:
    if state in {"current_clean", "superseded_tail", "historical_no_edge"}:
        return "no_action", "none"
    if state in {
        "current_risk_stale",
        "current_risk_contested",
        "current_risk_contradicted",
        "inconsistent_current_label",
        "current_lifecycle_conflict",
    }:
        return "label_review_candidate", "review_label_state"
    if state in {"edge_current_conflict", "edge_label_conflict"}:
        return "edge_review_candidate", "review_edge_state"
    if state == "orphaned_edge":
        return "identity_backfill_candidate", "recover_identity_evidence"
    if state == "storage_desync":
        return "storage_desync_blocker", "review_storage_desync"
    return "blocked_ambiguous", "manual_review"


def resolve_label_state(row: Mapping[str, Any]) -> dict[str, Any]:
    """Classify a concept row into one canonical read-only state."""

    data, malformed_json = _parse_json_object(_row_get(row, "data"))
    concept_id = _coerce_str(_row_get(row, "id"))
    status = _coerce_str(_row_get(row, "status")).casefold()
    currency_status = _coerce_str(_row_get(row, "currency_status")).upper()
    json_currency_status = _coerce_str(data.get("currency_status")).upper()
    is_current = _coerce_bool_int(_row_get(row, "is_current"))
    superseded_by = _coerce_str(_row_get(row, "superseded_by"))
    staleness_state = _coerce_str(_row_get(row, "staleness_state")).upper()
    replacement_exists_raw = _row_get(row, "replacement_exists")
    replacement_exists = _coerce_bool_int(replacement_exists_raw)

    evidence: list[str] = []
    has_edge = bool(superseded_by)
    has_replacement = replacement_exists is True
    is_orphan_edge = superseded_by == ORPHAN_SENTINEL or (has_edge and replacement_exists is False)
    row_says_superseded = status in SUPERSEDED_STATUSES or currency_status in SUPERSEDED_CURRENCY
    labels_say_historical = row_says_superseded or status in HISTORICAL_STATUSES
    row_says_historical = labels_say_historical or is_current is False

    if malformed_json:
        evidence.append("malformed_json_data")
    if json_currency_status and currency_status and json_currency_status != currency_status:
        state = "storage_desync"
        evidence.append("sql_json_currency_mismatch")
    elif is_orphan_edge:
        state = "orphaned_edge"
        evidence.append("missing_or_sentinel_replacement")
    elif has_edge and is_current is True:
        state = "edge_current_conflict"
        evidence.append("current_row_has_superseded_by")
    elif has_edge and not labels_say_historical:
        state = "edge_label_conflict"
        evidence.append("edge_without_historical_label")
    elif has_edge and (row_says_historical or has_replacement):
        state = "superseded_tail"
        evidence.append("historical_row_has_replacement")
    elif not has_edge and is_current is True and row_says_superseded:
        state = "inconsistent_current_label"
        evidence.append("current_row_has_superseded_label")
    elif not has_edge and is_current is True and status in LIFECYCLE_CONFLICT_STATUSES:
        state = "current_lifecycle_conflict"
        evidence.append(f"current_row_status_{status}")
    elif (
        not has_edge and is_current is True and (currency_status == "CONTRADICTED" or staleness_state == "CONTRADICTED")
    ):
        state = "current_risk_contradicted"
        evidence.append("current_row_contradicted")
    elif not has_edge and is_current is True and (currency_status == "CONTESTED" or staleness_state == "CONTESTED"):
        state = "current_risk_contested"
        evidence.append("current_row_contested")
    elif not has_edge and is_current is True and (currency_status == "STALE" or staleness_state == "STALE"):
        state = "current_risk_stale"
        evidence.append("current_row_stale")
    elif (
        not has_edge
        and is_current is True
        and status in {"active", "current", ""}
        and currency_status in CURRENT_CURRENCY
    ):
        state = "current_clean"
        evidence.append("active_current_no_edge")
    elif not has_edge and row_says_historical:
        state = "historical_no_edge"
        evidence.append("historical_without_replacement")
    else:
        state = "ambiguous"
        evidence.append("insufficient_or_unclassified_state_evidence")

    repairability_class, recommended_action = _repairability_for_state(state)
    return {
        "concept_id": concept_id,
        "resolved_state": state,
        "repairability_class": repairability_class,
        "recommended_action": recommended_action,
        "evidence": evidence,
        "status": status,
        "currency_status": currency_status,
        "json_currency_status": json_currency_status,
        "is_current": is_current,
        "superseded_by": superseded_by,
        "superseded_at": _coerce_str(_row_get(row, "superseded_at")),
        "created_at": _coerce_str(_row_get(row, "created_at")),
        "updated_at": _coerce_str(_row_get(row, "updated_at")),
        "content_updated_at": _coerce_str(_row_get(row, "content_updated_at")),
        "last_accessed": _coerce_str(_row_get(row, "last_accessed")),
        "valid_from": _coerce_str(_row_get(row, "valid_from")),
        "valid_until": _coerce_str(_row_get(row, "valid_until")),
        "subject_key": _coerce_str(_row_get(row, "subject_key")),
        "staleness_state": staleness_state,
        "summary_preview": _coerce_str(_row_get(row, "summary_preview")),
        "replacement_exists": replacement_exists,
        "malformed_json": malformed_json,
        "confidence": 0.6 if state == "ambiguous" or malformed_json else 1.0,
    }


def require_schema(conn: sqlite3.Connection) -> None:
    table = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'concepts'").fetchone()
    if table is None:
        raise ValueError("missing required table: concepts")
    columns = {row[1] for row in conn.execute("PRAGMA table_info(concepts)").fetchall()}
    missing = sorted(REQUIRED_CONCEPT_COLUMNS - columns)
    if missing:
        raise ValueError(f"missing concepts columns: {', '.join(missing)}")


def connect_read_only(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def load_rows_from_connection(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    require_schema(conn)
    return [dict(row) for row in conn.execute(READ_SQL).fetchall()]


def load_rows(db_path: Path) -> list[dict[str, Any]]:
    with connect_read_only(db_path) as conn:
        return load_rows_from_connection(conn)


def classify_rows(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [resolve_label_state(row) for row in rows]


def _status(value: bool) -> str:
    return "PASS" if value else "FAIL"


def _six_row_gate(classified_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {row["concept_id"]: row for row in classified_rows}
    present = [concept_id for concept_id in SIX_REPAIRED_ROW_IDS if concept_id in by_id]
    if not present:
        return {"status": "SKIP", "present": [], "failures": [], "expected_state": "inconsistent_current_label"}
    failures = [
        {"concept_id": concept_id, "resolved_state": by_id[concept_id]["resolved_state"]}
        for concept_id in present
        if by_id[concept_id]["resolved_state"] != "inconsistent_current_label"
    ]
    return {
        "status": _status(not failures),
        "present": present,
        "failures": failures,
        "expected_state": "inconsistent_current_label",
    }


def _load_gold(gold_path: Path) -> dict[str, str]:
    payload = json.loads(gold_path.read_text(encoding="utf-8"))
    rows = payload.get("rows", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("gold must be a list or an object with rows")
    gold: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("gold rows must be objects")
        concept_id = _coerce_str(row.get("concept_id") or row.get("id"))
        expected_state = _coerce_str(row.get("expected_state") or row.get("resolved_state"))
        if not concept_id or expected_state not in ALL_RESOLVED_STATES:
            raise ValueError("gold rows require concept_id/id and a valid expected_state")
        gold[concept_id] = expected_state
    return gold


def _score_gold(classified_rows: list[dict[str, Any]], gold_path: Path | None) -> dict[str, Any]:
    if gold_path is None:
        return {"status": "unavailable", "reason": "no reviewed gold supplied"}
    gold = _load_gold(gold_path)
    by_id = {row["concept_id"]: row for row in classified_rows}
    scored = []
    for concept_id, expected_state in gold.items():
        observed = by_id.get(concept_id, {}).get("resolved_state")
        scored.append(
            {
                "concept_id": concept_id,
                "expected_state": expected_state,
                "observed_state": observed,
                "correct": observed == expected_state,
            }
        )
    correct = sum(1 for row in scored if row["correct"])
    accuracy = correct / len(scored) if scored else 0.0
    return {"status": "available", "accuracy": accuracy, "correct": correct, "total": len(scored), "rows": scored}


def summarize(classified_rows: list[dict[str, Any]]) -> dict[str, Any]:
    state_counts = Counter(row["resolved_state"] for row in classified_rows)
    repairability_counts = Counter(row["repairability_class"] for row in classified_rows)
    action_counts = Counter(row["recommended_action"] for row in classified_rows)
    return {
        "total_concepts": len(classified_rows),
        "resolved_state_counts": {state: state_counts.get(state, 0) for state in ALL_RESOLVED_STATES},
        "repairability_class_counts": dict(sorted(repairability_counts.items())),
        "recommended_action_counts": dict(sorted(action_counts.items())),
        "malformed_json_count": sum(1 for row in classified_rows if row["malformed_json"]),
    }


def _examples_by_state(rows: list[dict[str, Any]], limit: int) -> dict[str, list[dict[str, Any]]]:
    examples: dict[str, list[dict[str, Any]]] = {}
    bounded_limit = max(0, int(limit))
    if bounded_limit == 0:
        return {}
    for row in rows:
        bucket = examples.setdefault(row["resolved_state"], [])
        if len(bucket) < bounded_limit:
            bucket.append(row)
    return dict(sorted(examples.items()))


def build_report_from_connection(
    conn: sqlite3.Connection,
    db_path: Path,
    *,
    example_limit: int = 5,
    gold_path: Path | None = None,
) -> dict[str, Any]:
    raw_rows = load_rows_from_connection(conn)
    classified_rows = classify_rows(raw_rows)
    gold_score = _score_gold(classified_rows, gold_path)
    six_gate = _six_row_gate(classified_rows)
    gates = {
        "schema_valid": {"status": "PASS"},
        "six_repaired_rows_classified": six_gate,
        "no_generated_accuracy_claim": {"status": "PASS" if gold_path is None else "SKIP"},
        "read_only_sql_contract": {"status": "PASS"},
        "state_coverage": {"status": "RUNTIME_NOT_APPLICABLE"},
    }
    if gold_path is not None:
        gates["gold_accuracy"] = {"status": _status(gold_score.get("accuracy", 0.0) >= 0.95)}
    blocking_failures = [
        name
        for name, gate in gates.items()
        if gate.get("status") == "FAIL" or (name == "gold_accuracy" and gate.get("status") != "PASS")
    ]
    return {
        "schema_version": "label_state_semantics_audit.v1",
        "resolver_precedence_version": RESOLVER_PRECEDENCE_VERSION,
        "status": "PASS" if not blocking_failures else "FAIL",
        "claim_boundary": CLAIM_BOUNDARY,
        "generated_at": _now_iso(),
        "db_path": str(db_path),
        "summary": summarize(classified_rows),
        "gates": gates,
        "gold_score": gold_score,
        "rows": classified_rows,
        "examples_by_state": _examples_by_state(classified_rows, example_limit),
    }


def build_report(db_path: Path, *, example_limit: int = 5, gold_path: Path | None = None) -> dict[str, Any]:
    with connect_read_only(db_path) as conn:
        return build_report_from_connection(conn, db_path, example_limit=example_limit, gold_path=gold_path)


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Label-State Semantics Audit",
        "",
        f"Generated at: `{report['generated_at']}`",
        f"Database: `{report['db_path']}`",
        f"Boundary: {report['claim_boundary']}",
        f"Status: `{report['status']}`",
        "",
        "## Summary",
        "",
        f"- Total concepts: {summary['total_concepts']}",
        f"- Malformed JSON rows: {summary['malformed_json_count']}",
        "",
        "## Resolved States",
        "",
    ]
    for state, count in summary["resolved_state_counts"].items():
        lines.append(f"- `{state}`: {count}")
    lines.extend(["", "## Repairability Classes", ""])
    for key, count in summary["repairability_class_counts"].items():
        lines.append(f"- `{key}`: {count}")
    lines.extend(["", "## Gates", ""])
    for key, gate in report["gates"].items():
        lines.append(f"- `{key}`: {gate.get('status')}")
    lines.extend(["", "## Examples", ""])
    for state, examples in report["examples_by_state"].items():
        lines.append(f"### {state}")
        lines.append("")
        for row in examples:
            evidence = ", ".join(row["evidence"])
            lines.append(f"- `{row['concept_id']}` | action `{row['recommended_action']}` | evidence: {evidence}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _compact_gold_score(gold_score: Mapping[str, Any]) -> dict[str, Any]:
    compact = {
        key: value for key, value in gold_score.items() if key in {"status", "reason", "accuracy", "correct", "total"}
    }
    if compact.get("status") == "available":
        correct = int(compact.get("correct", 0))
        total = int(compact.get("total", 0))
        compact["reviewed_case_correctness"] = f"{correct}/{total}"
    return compact


def _sample_ids(rows: list[Mapping[str, Any]], key: str, value: str, limit: int) -> list[str]:
    sample_limit = max(0, int(limit))
    if sample_limit == 0:
        return []
    ids: list[str] = []
    for row in rows:
        if row.get(key) == value:
            ids.append(str(row.get("concept_id", "")))
        if len(ids) >= sample_limit:
            break
    return ids


def build_operator_packet(report: dict[str, Any], *, max_ids_per_group: int = 10) -> dict[str, Any]:
    """Build a compact read-only packet for operator maintenance review."""

    summary = report["summary"]
    rows = report.get("rows", [])
    if not isinstance(rows, list):
        rows = []
    repairability_counts = summary.get("repairability_class_counts", {})
    action_counts = summary.get("recommended_action_counts", {})
    next_actions: list[dict[str, Any]] = []
    for repairability_class, count in sorted(repairability_counts.items()):
        if repairability_class == "no_action" or int(count) <= 0:
            continue
        matching_rows = [row for row in rows if row.get("repairability_class") == repairability_class]
        action_counter = Counter(str(row.get("recommended_action", "manual_review")) for row in matching_rows)
        recommended_action = action_counter.most_common(1)[0][0] if action_counter else "manual_review"
        next_actions.append(
            {
                "repairability_class": repairability_class,
                "count": int(count),
                "recommended_action": recommended_action,
                "action_label": ACTION_LABELS.get(recommended_action, ACTION_LABELS["manual_review"]),
                "sample_concept_ids": _sample_ids(
                    matching_rows, "repairability_class", repairability_class, max_ids_per_group
                ),
                "sample_limit": max(0, int(max_ids_per_group)),
                "requires_operator_review": True,
            }
        )

    return {
        "schema_version": OPERATOR_PACKET_SCHEMA_VERSION,
        "status": report["status"],
        "generated_at": report["generated_at"],
        "db_path": report["db_path"],
        "claim_boundary": report["claim_boundary"],
        "resolver_precedence_version": report.get("resolver_precedence_version"),
        "summary": {
            "total_concepts": summary["total_concepts"],
            "malformed_json_count": summary["malformed_json_count"],
            "resolved_state_counts": summary["resolved_state_counts"],
            "repairability_class_counts": repairability_counts,
            "recommended_action_counts": action_counts,
        },
        "gates": report["gates"],
        "gold_score": _compact_gold_score(report.get("gold_score", {})),
        "next_actions": next_actions,
        "decision_required": report["status"] != "PASS" or bool(next_actions),
        "claim_boundary_notes": [
            CLAIM_BOUNDARY,
            "Operator packet is read-only evidence; it does not approve or apply repairs.",
            "Reviewed-gold accuracy only covers reviewed cases supplied to this run.",
        ],
    }


def render_operator_summary(packet: dict[str, Any]) -> str:
    summary = packet["summary"]
    lines = [
        "Label-State Trust Maintenance",
        f"Status: {packet['status']}",
        f"Generated: {packet['generated_at']}",
        f"Database: {packet['db_path']}",
        f"Boundary: {packet['claim_boundary']}",
        f"Resolver: {packet.get('resolver_precedence_version')}",
        "",
        "Scope",
        f"- Total concepts reviewed: {summary['total_concepts']}",
        f"- Malformed JSON rows: {summary['malformed_json_count']}",
        f"- Decision required: {'yes' if packet['decision_required'] else 'no'}",
        "",
        "Maintenance Actions",
    ]
    if packet["next_actions"]:
        for action in packet["next_actions"]:
            sample = ", ".join(action["sample_concept_ids"]) or "none"
            lines.append(
                f"- {action['repairability_class']}: {action['count']} rows; {action['action_label']} samples: {sample}"
            )
    else:
        lines.append("- none")
    lines.extend(["", "Gates"])
    for key, gate in packet["gates"].items():
        lines.append(f"- {key}: {gate.get('status')}")
    gold_score = packet.get("gold_score", {})
    lines.extend(["", "Reviewed Gold"])
    if gold_score.get("status") == "available":
        lines.append(f"- {gold_score.get('reviewed_case_correctness')} correct; accuracy {gold_score.get('accuracy')}")
    else:
        lines.append(f"- {gold_score.get('reason', 'not supplied')}")
    lines.extend(["", "Notes"])
    for note in packet["claim_boundary_notes"]:
        lines.append(f"- {note}")
    return "\n".join(lines).rstrip() + "\n"


def _matching_review_rows(
    rows: list[Mapping[str, Any]],
    *,
    repairability_class: str | None = None,
    recommended_action: str | None = None,
) -> list[Mapping[str, Any]]:
    matching = []
    for row in rows:
        if repairability_class is not None and row.get("repairability_class") != repairability_class:
            continue
        if recommended_action is not None and row.get("recommended_action") != recommended_action:
            continue
        if repairability_class is None and recommended_action is None and row.get("repairability_class") == "no_action":
            continue
        matching.append(row)
    return matching


def _review_row(row: Mapping[str, Any]) -> dict[str, Any]:
    recommended_action = str(row.get("recommended_action", "manual_review"))
    return {
        "concept_id": str(row.get("concept_id", "")),
        "resolved_state": str(row.get("resolved_state", "")),
        "repairability_class": str(row.get("repairability_class", "")),
        "recommended_action": recommended_action,
        "evidence": list(row.get("evidence", [])) if isinstance(row.get("evidence"), list) else [],
        "status": str(row.get("status", "")),
        "currency_status": str(row.get("currency_status", "")),
        "json_currency_status": str(row.get("json_currency_status", "")),
        "is_current": row.get("is_current"),
        "superseded_by": str(row.get("superseded_by", "")),
        "superseded_at": str(row.get("superseded_at", "")),
        "created_at": str(row.get("created_at", "")),
        "updated_at": str(row.get("updated_at", "")),
        "content_updated_at": str(row.get("content_updated_at", "")),
        "last_accessed": str(row.get("last_accessed", "")),
        "valid_from": str(row.get("valid_from", "")),
        "valid_until": str(row.get("valid_until", "")),
        "subject_key": str(row.get("subject_key", "")),
        "staleness_state": str(row.get("staleness_state", "")),
        "summary_preview": str(row.get("summary_preview", "")),
        "replacement_exists": row.get("replacement_exists"),
        "review_prompt": ACTION_LABELS.get(recommended_action, ACTION_LABELS["manual_review"]),
    }


def compute_review_packet_hash(packet: Mapping[str, Any]) -> str:
    rows = []
    for row in packet.get("rows", []):
        if not isinstance(row, Mapping):
            continue
        rows.append(
            {
                "concept_id": row.get("concept_id", ""),
                "resolved_state": row.get("resolved_state", ""),
                "repairability_class": row.get("repairability_class", ""),
                "recommended_action": row.get("recommended_action", ""),
                "superseded_by": row.get("superseded_by", ""),
                "superseded_at": row.get("superseded_at", ""),
                "updated_at": row.get("updated_at", ""),
                "content_updated_at": row.get("content_updated_at", ""),
            }
        )
    payload = {
        "schema_version": packet.get("schema_version"),
        "db_path": packet.get("db_path"),
        "filters": packet.get("filters", {}),
        "total_matching_rows": packet.get("total_matching_rows"),
        "included_row_count": packet.get("included_row_count"),
        "rows": rows,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_review_packet(
    report: dict[str, Any],
    *,
    repairability_class: str | None = None,
    recommended_action: str | None = None,
    max_rows: int = 50,
) -> dict[str, Any]:
    """Build a bounded row-level packet for explicit operator review."""

    rows = report.get("rows", [])
    if not isinstance(rows, list):
        rows = []
    matching_rows = _matching_review_rows(
        rows,
        repairability_class=repairability_class,
        recommended_action=recommended_action,
    )
    bounded_max = max(0, int(max_rows))
    included_rows = [_review_row(row) for row in matching_rows[:bounded_max]]
    repairability_counts = Counter(str(row.get("repairability_class", "")) for row in matching_rows)
    action_counts = Counter(str(row.get("recommended_action", "")) for row in matching_rows)
    packet = {
        "schema_version": REVIEW_PACKET_SCHEMA_VERSION,
        "status": report["status"],
        "generated_at": report["generated_at"],
        "db_path": report["db_path"],
        "claim_boundary": report["claim_boundary"],
        "resolver_precedence_version": report.get("resolver_precedence_version"),
        "filters": {
            "repairability_class": repairability_class,
            "recommended_action": recommended_action,
            "max_rows": bounded_max,
        },
        "total_matching_rows": len(matching_rows),
        "included_row_count": len(included_rows),
        "truncated": len(included_rows) < len(matching_rows),
        "review_required": bool(included_rows),
        "metrics": {
            "matching_repairability_class_counts": dict(sorted(repairability_counts.items())),
            "matching_recommended_action_counts": dict(sorted(action_counts.items())),
        },
        "rows": included_rows,
        "claim_boundary_notes": [
            CLAIM_BOUNDARY,
            "Review packet rows are candidates, not approvals.",
            "A dry-run repair plan requires an explicit reviewed decision ledger.",
        ],
    }
    packet["packet_filter_hash"] = compute_review_packet_hash(packet)
    return packet


def render_review_packet_markdown(packet: dict[str, Any]) -> str:
    filters = packet.get("filters", {})
    lines = [
        "Label-State Review Packet",
        f"Status: {packet['status']}",
        f"Generated: {packet['generated_at']}",
        f"Database: {packet['db_path']}",
        f"Packet hash: {packet['packet_filter_hash']}",
        f"Boundary: {packet['claim_boundary']}",
        "",
        "Filters",
        f"- repairability_class: {filters.get('repairability_class')}",
        f"- recommended_action: {filters.get('recommended_action')}",
        f"- max_rows: {filters.get('max_rows')}",
        "",
        "Scope",
        f"- Matching rows: {packet['total_matching_rows']}",
        f"- Included rows: {packet['included_row_count']}",
        f"- Truncated: {'yes' if packet['truncated'] else 'no'}",
        f"- Review required: {'yes' if packet['review_required'] else 'no'}",
        "",
        "Rows",
    ]
    if packet["rows"]:
        for row in packet["rows"]:
            evidence = ", ".join(row.get("evidence", [])) or "none"
            lines.extend(
                [
                    f"- {row['concept_id']}",
                    f"  state: {row['resolved_state']}",
                    f"  class: {row['repairability_class']}",
                    f"  action: {row['recommended_action']}",
                    f"  evidence: {evidence}",
                    f"  created_at: {row['created_at']}",
                    f"  updated_at: {row['updated_at']}",
                    f"  content_updated_at: {row['content_updated_at']}",
                    f"  last_accessed: {row['last_accessed']}",
                    f"  valid_from: {row['valid_from']}",
                    f"  valid_until: {row['valid_until']}",
                    f"  superseded_by: {row['superseded_by']}",
                    f"  superseded_at: {row['superseded_at']}",
                    f"  subject_key: {row['subject_key']}",
                    f"  summary: {row['summary_preview']}",
                ]
            )
    else:
        lines.append("- none")
    lines.extend(["", "Notes"])
    for note in packet.get("claim_boundary_notes", []):
        lines.append(f"- {note}")
    return "\n".join(lines).rstrip() + "\n"


def load_review_decisions(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("review decisions root must be an object")
    return payload


def _validation_error(code: str, message: str, *, concept_id: str | None = None) -> dict[str, str]:
    error = {"code": code, "message": message}
    if concept_id is not None:
        error["concept_id"] = concept_id
    return error


def validate_review_decisions(
    packet: Mapping[str, Any],
    decisions_payload: Mapping[str, Any],
) -> dict[str, Any]:
    packet_rows = packet.get("rows", [])
    if not isinstance(packet_rows, list):
        packet_rows = []
    rows_by_id = {str(row.get("concept_id", "")): row for row in packet_rows if isinstance(row, Mapping)}
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    valid_decisions: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicate_count = 0
    unknown_count = 0

    if decisions_payload.get("schema_version") != REVIEW_DECISIONS_SCHEMA_VERSION:
        errors.append(_validation_error("unsupported_schema_version", "unsupported review decisions schema_version"))
    if decisions_payload.get("packet_filter_hash") != packet.get("packet_filter_hash"):
        errors.append(_validation_error("packet_hash_mismatch", "packet_filter_hash does not match active packet"))

    raw_decisions = decisions_payload.get("decisions")
    if not isinstance(raw_decisions, list):
        errors.append(_validation_error("decisions_not_list", "decisions must be a list"))
        raw_decisions = []

    for index, raw_decision in enumerate(raw_decisions):
        line_error_count = len(errors)
        if not isinstance(raw_decision, Mapping):
            errors.append(_validation_error("decision_not_object", f"decision at index {index} must be an object"))
            continue
        concept_id = _coerce_str(raw_decision.get("concept_id"))
        decision = _coerce_str(raw_decision.get("decision"))
        proposed_action = _coerce_str(raw_decision.get("proposed_action"))
        reviewed_state = _coerce_str(raw_decision.get("reviewed_state"))
        evidence_note = _coerce_str(raw_decision.get("evidence_note"))

        if not concept_id:
            errors.append(_validation_error("missing_concept_id", "decision concept_id is required"))
            continue
        if concept_id in seen:
            duplicate_count += 1
            errors.append(
                _validation_error("duplicate_concept_id", "duplicate concept_id in decisions", concept_id=concept_id)
            )
            continue
        seen.add(concept_id)
        packet_row = rows_by_id.get(concept_id)
        if packet_row is None:
            unknown_count += 1
            errors.append(
                _validation_error(
                    "unknown_concept_id", "concept_id is not included in active packet", concept_id=concept_id
                )
            )
            continue
        if decision not in ALLOWED_REVIEW_DECISIONS:
            errors.append(
                _validation_error("unsupported_decision", "unsupported review decision", concept_id=concept_id)
            )
        if decision in {APPROVE_DECISION, REJECT_DECISION} and not evidence_note:
            errors.append(
                _validation_error(
                    "missing_evidence_note", "approval/rejection requires evidence_note", concept_id=concept_id
                )
            )
        if decision == APPROVE_DECISION and proposed_action != packet_row.get("recommended_action"):
            errors.append(
                _validation_error(
                    "proposed_action_mismatch", "approval proposed_action must match packet row", concept_id=concept_id
                )
            )
        if reviewed_state and reviewed_state != packet_row.get("resolved_state"):
            errors.append(
                _validation_error(
                    "reviewed_state_mismatch", "reviewed_state must match packet row", concept_id=concept_id
                )
            )
        if decision == DEFER_DECISION and not evidence_note:
            warnings.append(
                _validation_error(
                    "defer_without_evidence_note", "defer decision has no evidence_note", concept_id=concept_id
                )
            )
        if len(errors) == line_error_count:
            valid_decisions.append(
                {
                    "concept_id": concept_id,
                    "decision": decision,
                    "proposed_action": proposed_action,
                    "reviewed_state": reviewed_state,
                    "evidence_note": evidence_note,
                    "replacement_concept_id": _coerce_str(raw_decision.get("replacement_concept_id")),
                    "packet_row": dict(packet_row),
                }
            )

    missing_review_count = max(0, len(rows_by_id) - len(seen & set(rows_by_id)))
    if missing_review_count:
        warnings.append(
            _validation_error(
                "packet_rows_without_decisions",
                f"{missing_review_count} packet row(s) have no decision line",
            )
        )
    approved_count = sum(1 for row in valid_decisions if row["decision"] == APPROVE_DECISION)
    rejected_count = sum(1 for row in valid_decisions if row["decision"] == REJECT_DECISION)
    deferred_count = sum(1 for row in valid_decisions if row["decision"] == DEFER_DECISION)
    approval_evidence_complete_rate = (
        sum(1 for row in valid_decisions if row["decision"] == APPROVE_DECISION and row["evidence_note"])
        / approved_count
        if approved_count
        else 1.0
    )
    included_row_count = int(packet.get("included_row_count", len(rows_by_id)) or 0)
    reviewed_coverage_rate = len(seen & set(rows_by_id)) / included_row_count if included_row_count else 1.0
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "warnings": warnings,
        "valid_decisions": valid_decisions,
        "metrics": {
            "reviewed_decision_count": len(raw_decisions),
            "valid_decision_count": len(valid_decisions),
            "invalid_decision_count": len(raw_decisions) - len(valid_decisions),
            "approved_count": approved_count,
            "rejected_count": rejected_count,
            "deferred_count": deferred_count,
            "unknown_concept_id_count": unknown_count,
            "duplicate_decision_count": duplicate_count,
            "approval_evidence_complete_rate": approval_evidence_complete_rate,
            "reviewed_coverage_rate": reviewed_coverage_rate,
        },
    }


def _simulated_effect(action: str, decision: Mapping[str, Any]) -> str:
    if action in {"review_label_state", "review_edge_state"}:
        return "reduce_repairability"
    if action == "recover_identity_evidence":
        return "no_reduction_requires_identity"
    return "no_reduction_manual_review"


def build_dry_run_repair_plan(
    report: dict[str, Any],
    packet: Mapping[str, Any],
    decisions_payload: Mapping[str, Any],
) -> dict[str, Any]:
    validation = validate_review_decisions(packet, decisions_payload)
    baseline_counts = dict(report["summary"].get("repairability_class_counts", {}))
    simulated_counts = dict(baseline_counts)
    operations: list[dict[str, Any]] = []
    approved_repairability_reduction = 0

    if validation["status"] == "PASS":
        for decision in validation["valid_decisions"]:
            if decision["decision"] != APPROVE_DECISION:
                continue
            packet_row = decision["packet_row"]
            action = str(packet_row.get("recommended_action", "manual_review"))
            repairability_class = str(packet_row.get("repairability_class", ""))
            effect = _simulated_effect(action, decision)
            if effect == "reduce_repairability" and simulated_counts.get(repairability_class, 0) > 0:
                simulated_counts[repairability_class] = int(simulated_counts.get(repairability_class, 0)) - 1
                simulated_counts["no_action"] = int(simulated_counts.get("no_action", 0)) + 1
                approved_repairability_reduction += 1
            operations.append(
                {
                    "concept_id": decision["concept_id"],
                    "operation": action,
                    "resolved_state": packet_row.get("resolved_state"),
                    "repairability_class": repairability_class,
                    "recommended_action": action,
                    "evidence_note": decision["evidence_note"],
                    "simulated_effect": effect,
                }
            )

    simulated_remaining = sum(int(count) for key, count in simulated_counts.items() if key != "no_action")
    return {
        "schema_version": DRY_RUN_REPAIR_PLAN_SCHEMA_VERSION,
        "status": validation["status"],
        "dry_run": True,
        "mutation_allowed": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "generated_at": _now_iso(),
        "db_path": report["db_path"],
        "packet_filter_hash": packet.get("packet_filter_hash"),
        "decision_summary": validation["metrics"],
        "baseline_counts": baseline_counts,
        "simulated_after_counts": dict(sorted(simulated_counts.items())),
        "estimated_improvement": {
            "approved_repairability_reduction": approved_repairability_reduction,
            "simulated_remaining_repairable_rows": simulated_remaining,
        },
        "operations": operations,
        "validation_errors": validation["errors"],
        "validation_warnings": validation["warnings"],
        "claim_boundary_notes": [
            CLAIM_BOUNDARY,
            "This is a dry-run plan only; it does not approve or apply repairs.",
            "Simulated after-counts are not database truth.",
        ],
    }


def _backup_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _default_backup_dir() -> Path:
    return Path.home() / ".pith" / "backups" / "maint091"


def create_live_apply_backup(
    db_path: Path,
    backup_dir: Path | None = None,
    *,
    retention: int = DEFAULT_LIVE_APPLY_BACKUP_RETENTION,
) -> dict[str, Any]:
    """Create a WAL-safe SQLite backup before reviewed live repair apply."""

    source_path = Path(db_path)
    if not source_path.exists():
        raise ValueError(f"source DB does not exist: {source_path}")
    destination_dir = backup_dir if backup_dir is not None else _default_backup_dir()
    destination_dir.mkdir(parents=True, exist_ok=True)
    source_size = source_path.stat().st_size
    available_before = shutil.disk_usage(destination_dir).free
    required_free = max(MIN_BACKUP_FREE_BYTES, int(source_size * MIN_BACKUP_FREE_RATIO))
    if available_before < required_free:
        raise ValueError(
            f"insufficient free space for backup: available={available_before} required={required_free}"
        )

    backup_path = destination_dir / f"label_state_semantics_live_apply_{_backup_timestamp()}.db"
    suffix = 1
    while backup_path.exists():
        backup_path = destination_dir / f"label_state_semantics_live_apply_{_backup_timestamp()}_{suffix}.db"
        suffix += 1

    with sqlite3.connect(str(source_path)) as src, sqlite3.connect(str(backup_path)) as dst:
        src.backup(dst, pages=-1)

    with sqlite3.connect(str(backup_path)) as verify_conn:
        quick_check = str(verify_conn.execute("PRAGMA quick_check").fetchone()[0])
        concept_count = int(verify_conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0])
    if quick_check.lower() != "ok":
        raise ValueError(f"backup quick_check failed: {quick_check}")

    pruned_paths: list[str] = []
    keep = max(0, int(retention))
    backups = sorted(
        destination_dir.glob("label_state_semantics_live_apply_*.db"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for old_backup in backups[keep:]:
        old_backup.unlink()
        pruned_paths.append(str(old_backup))

    return {
        "path": str(backup_path),
        "quick_check": quick_check,
        "concept_count": concept_count,
        "source_size_bytes": source_size,
        "available_bytes_before": available_before,
        "required_free_bytes": required_free,
        "retention": keep,
        "pruned_paths": pruned_paths,
    }


def build_live_repair_apply_plan(
    report: dict[str, Any],
    packet: Mapping[str, Any],
    decisions_payload: Mapping[str, Any],
    *,
    expected_packet_hash: str | None,
    max_apply_rows: int | None = None,
) -> dict[str, Any]:
    """Build a mutation-ready plan without performing writes."""

    validation = validate_review_decisions(packet, decisions_payload)
    errors = list(validation["errors"])
    warnings = list(validation["warnings"])
    packet_hash = _coerce_str(packet.get("packet_filter_hash"))
    expected_hash = _coerce_str(expected_packet_hash)
    if not expected_hash:
        errors.append(_validation_error("missing_expected_packet_hash", "expected packet hash is required"))
    elif expected_hash != packet_hash:
        errors.append(_validation_error("expected_packet_hash_mismatch", "expected packet hash does not match packet"))

    operations: list[dict[str, Any]] = []
    approved_unsupported: list[dict[str, str]] = []
    if not errors:
        for decision in validation["valid_decisions"]:
            if decision["decision"] != APPROVE_DECISION:
                continue
            packet_row = decision["packet_row"]
            action = str(packet_row.get("recommended_action", "manual_review"))
            if action not in LIVE_APPLY_ALLOWED_ACTIONS:
                approved_unsupported.append(
                    {
                        "concept_id": decision["concept_id"],
                        "action": action,
                        "reason": "unsupported_live_apply_action",
                    }
                )
                continue
            operations.append(
                {
                    "concept_id": decision["concept_id"],
                    "operation": action,
                    "resolved_state": packet_row.get("resolved_state"),
                    "repairability_class": packet_row.get("repairability_class"),
                    "recommended_action": action,
                    "superseded_by": packet_row.get("superseded_by"),
                    "updated_at": packet_row.get("updated_at"),
                    "evidence_note": decision["evidence_note"],
                    "packet_row": dict(packet_row),
                }
            )
    for unsupported in approved_unsupported:
        errors.append(
            _validation_error(
                "unsupported_live_apply_action",
                f"live apply only supports {sorted(LIVE_APPLY_ALLOWED_ACTIONS)}",
                concept_id=unsupported["concept_id"],
            )
        )

    capped = False
    if max_apply_rows is not None:
        bounded = max(0, int(max_apply_rows))
        if len(operations) > bounded:
            capped = True
            warnings.append(
                _validation_error(
                    "max_apply_rows_capped",
                    f"operations capped from {len(operations)} to {bounded}",
                )
            )
            operations = operations[:bounded]

    baseline_counts = dict(report["summary"].get("repairability_class_counts", {}))
    expected_reduction = Counter(str(operation["repairability_class"]) for operation in operations)
    return {
        "schema_version": LIVE_REPAIR_APPLY_PLAN_SCHEMA_VERSION,
        "status": "PASS" if not errors else "FAIL",
        "dry_run": False,
        "mutation_allowed": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "generated_at": _now_iso(),
        "db_path": report["db_path"],
        "packet_filter_hash": packet_hash,
        "expected_packet_hash": expected_hash,
        "decision_summary": validation["metrics"],
        "baseline_counts": baseline_counts,
        "expected_reduction": dict(sorted(expected_reduction.items())),
        "operation_count": len(operations),
        "operations": operations,
        "validation_errors": errors,
        "validation_warnings": warnings,
        "approved_unsupported": approved_unsupported,
        "capped": capped,
        "claim_boundary_notes": [
            CLAIM_BOUNDARY,
            "This plan is eligible for live apply only with explicit operator confirmation.",
            "v1 live apply supports review_edge_state only.",
        ],
    }


def _rollback_command(db_path: Path, backup_path: str | None) -> str | None:
    if not backup_path:
        return None
    return f"stop pith server, then cp {shlex.quote(backup_path)} {shlex.quote(str(db_path))}, then restart pith"


def _apply_failure(
    db_path: Path,
    *,
    errors: list[dict[str, str]],
    operator_confirmed: bool = False,
    packet_hash: str | None = None,
    expected_packet_hash: str | None = None,
    backup: dict[str, Any] | None = None,
    before_counts: Mapping[str, Any] | None = None,
    after_counts: Mapping[str, Any] | None = None,
    changed_rows: list[str] | None = None,
    warnings: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    rolled_back_rows = list(changed_rows or [])
    return {
        "schema_version": LIVE_REPAIR_APPLY_SCHEMA_VERSION,
        "status": "FAIL",
        "dry_run": False,
        "mutation_allowed": False,
        "operator_confirmed": operator_confirmed,
        "claim_boundary": CLAIM_BOUNDARY,
        "generated_at": _now_iso(),
        "db_path": str(db_path),
        "packet_filter_hash": packet_hash,
        "expected_packet_hash": expected_packet_hash,
        "backup": backup,
        "before_counts": dict(before_counts or {}),
        "after_counts": dict(after_counts or {}),
        "applied_count": 0,
        "changed_rows": [],
        "rolled_back_rows": rolled_back_rows,
        "validation_errors": errors,
        "validation_warnings": list(warnings or []),
        "rollback_command": _rollback_command(db_path, backup.get("path") if backup else None),
    }


def apply_reviewed_repair_plan(
    db_path: Path,
    packet: Mapping[str, Any],
    decisions_payload: Mapping[str, Any],
    *,
    operator_confirmed: bool,
    expected_packet_hash: str | None,
    backup_dir: Path | None = None,
    max_apply_rows: int | None = None,
) -> dict[str, Any]:
    """Apply approved deterministic edge repairs with backup and exact row guards."""

    db_path = Path(db_path)
    packet_hash = _coerce_str(packet.get("packet_filter_hash"))
    expected_hash = _coerce_str(expected_packet_hash)
    if not operator_confirmed:
        return _apply_failure(
            db_path,
            errors=[_validation_error("operator_confirmation_required", "live apply requires operator confirmation")],
            operator_confirmed=False,
            packet_hash=packet_hash,
            expected_packet_hash=expected_hash,
        )

    filters = packet.get("filters", {})
    if not isinstance(filters, Mapping):
        filters = {}
    report = build_report(db_path)
    live_packet = build_review_packet(
        report,
        repairability_class=filters.get("repairability_class"),
        recommended_action=filters.get("recommended_action"),
        max_rows=int(filters.get("max_rows", 50) or 50),
    )
    plan = build_live_repair_apply_plan(
        report,
        live_packet,
        decisions_payload,
        expected_packet_hash=expected_hash,
        max_apply_rows=max_apply_rows,
    )
    if plan["status"] != "PASS":
        return _apply_failure(
            db_path,
            errors=plan["validation_errors"],
            operator_confirmed=True,
            warnings=plan["validation_warnings"],
            packet_hash=plan.get("packet_filter_hash"),
            expected_packet_hash=expected_hash,
            before_counts=plan.get("baseline_counts"),
        )
    if not plan["operations"]:
        return {
            "schema_version": LIVE_REPAIR_APPLY_SCHEMA_VERSION,
            "status": "PASS",
            "dry_run": False,
            "mutation_allowed": False,
            "operator_confirmed": True,
            "claim_boundary": CLAIM_BOUNDARY,
            "generated_at": _now_iso(),
            "db_path": str(db_path),
            "packet_filter_hash": plan.get("packet_filter_hash"),
            "expected_packet_hash": expected_hash,
            "backup": None,
            "before_counts": plan.get("baseline_counts", {}),
            "after_counts": plan.get("baseline_counts", {}),
            "expected_reduction": plan.get("expected_reduction", {}),
            "applied_count": 0,
            "changed_rows": [],
            "validation_errors": [],
            "validation_warnings": plan.get("validation_warnings", []),
            "rollback_command": None,
            "claim_boundary_notes": plan.get("claim_boundary_notes", []),
        }

    backup = create_live_apply_backup(db_path, backup_dir)
    changed_rows: list[str] = []
    applied_at = _now_iso()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("BEGIN IMMEDIATE")
        before_report = build_report_from_connection(conn, db_path)
        before_counts = before_report["summary"].get("repairability_class_counts", {})
        rows_by_id = {row["concept_id"]: row for row in before_report.get("rows", [])}

        for operation in plan["operations"]:
            concept_id = str(operation["concept_id"])
            current_row = rows_by_id.get(concept_id)
            packet_row = operation["packet_row"]
            if current_row is None:
                raise ValueError(f"row disappeared before apply: {concept_id}")
            guard_checks = {
                "resolved_state": current_row.get("resolved_state") == packet_row.get("resolved_state"),
                "recommended_action": current_row.get("recommended_action") == packet_row.get("recommended_action"),
                "superseded_by": current_row.get("superseded_by") == packet_row.get("superseded_by"),
                "updated_at": current_row.get("updated_at") == packet_row.get("updated_at"),
                "replacement_exists": current_row.get("replacement_exists") is True,
            }
            failed_guards = [key for key, passed in guard_checks.items() if not passed]
            if failed_guards:
                raise ValueError(f"row guard failed for {concept_id}: {', '.join(failed_guards)}")
            metadata = {
                "schema_version": LIVE_REPAIR_APPLY_SCHEMA_VERSION,
                "applied_at": applied_at,
                "packet_filter_hash": plan.get("packet_filter_hash"),
                "reviewed_state": operation.get("resolved_state"),
                "repairability_class": operation.get("repairability_class"),
                "recommended_action": operation.get("recommended_action"),
                "evidence_note": operation.get("evidence_note"),
            }
            conn.execute(
                """
                UPDATE concepts
                SET
                  status = 'superseded',
                  currency_status = 'SUPERSEDED',
                  data = json_set(
                    CASE WHEN json_valid(data) THEN data ELSE '{}' END,
                    '$.currency_status',
                    'SUPERSEDED',
                    '$.metadata.label_state_semantics_repair',
                    json(?)
                  ),
                  is_current = 0,
                  superseded_at = COALESCE(NULLIF(superseded_at, ''), ?),
                  updated_at = ?
                WHERE
                  id = ?
                  AND superseded_by = ?
                  AND updated_at = ?
                  AND COALESCE(TRIM(superseded_by), '') <> ''
                  AND EXISTS (
                    SELECT 1 FROM concepts replacement
                    WHERE replacement.id = concepts.superseded_by
                  )
                """,
                (
                    json.dumps(metadata, sort_keys=True),
                    applied_at,
                    applied_at,
                    concept_id,
                    operation["superseded_by"],
                    operation["updated_at"],
                ),
            )
            changed = int(conn.execute("SELECT changes()").fetchone()[0])
            if changed != 1:
                raise ValueError(f"expected one changed row for {concept_id}, got {changed}")
            changed_rows.append(concept_id)

        after_report = build_report_from_connection(conn, db_path)
        after_counts = after_report["summary"].get("repairability_class_counts", {})
        before_edge = int(before_counts.get("edge_review_candidate", 0) or 0)
        after_edge = int(after_counts.get("edge_review_candidate", 0) or 0)
        expected_edge_reduction = int(plan.get("expected_reduction", {}).get("edge_review_candidate", 0) or 0)
        if after_edge != before_edge - expected_edge_reduction:
            raise ValueError(
                f"post-audit edge reduction mismatch: before={before_edge} after={after_edge} expected={expected_edge_reduction}"
            )
        conn.commit()
    except Exception as exc:
        conn.rollback()
        return _apply_failure(
            db_path,
            errors=[_validation_error("live_apply_failed", str(exc))],
            operator_confirmed=True,
            warnings=plan.get("validation_warnings", []),
            packet_hash=plan.get("packet_filter_hash"),
            expected_packet_hash=expected_hash,
            backup=backup,
            before_counts=locals().get("before_counts", plan.get("baseline_counts", {})),
            after_counts=locals().get("after_counts", {}),
            changed_rows=changed_rows,
        )
    finally:
        conn.close()

    return {
        "schema_version": LIVE_REPAIR_APPLY_SCHEMA_VERSION,
        "status": "PASS",
        "dry_run": False,
        "mutation_allowed": True,
        "operator_confirmed": True,
        "claim_boundary": CLAIM_BOUNDARY,
        "generated_at": _now_iso(),
        "db_path": str(db_path),
        "packet_filter_hash": plan.get("packet_filter_hash"),
        "expected_packet_hash": expected_hash,
        "backup": backup,
        "before_counts": dict(before_counts),
        "after_counts": dict(after_counts),
        "expected_reduction": plan.get("expected_reduction", {}),
        "applied_count": len(changed_rows),
        "changed_rows": changed_rows,
        "validation_errors": [],
        "validation_warnings": plan.get("validation_warnings", []),
        "rollback_command": _rollback_command(db_path, backup.get("path")),
        "claim_boundary_notes": [
            CLAIM_BOUNDARY,
            "Reviewed live apply was operator-confirmed and hash-bound.",
            "Rollback requires stopping Pith before restoring the emitted backup.",
        ],
    }


def render_dry_run_repair_plan_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "Label-State Dry-Run Repair Plan",
        f"Status: {plan['status']}",
        f"Generated: {plan['generated_at']}",
        f"Database: {plan['db_path']}",
        f"Dry run: {'yes' if plan['dry_run'] else 'no'}",
        f"Mutation allowed: {'yes' if plan['mutation_allowed'] else 'no'}",
        f"Packet hash: {plan.get('packet_filter_hash')}",
        f"Boundary: {plan['claim_boundary']}",
        "",
        "Decision Metrics",
    ]
    for key, value in plan["decision_summary"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "Estimated Improvement"])
    for key, value in plan["estimated_improvement"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "Validation Errors"])
    if plan["validation_errors"]:
        for error in plan["validation_errors"]:
            lines.append(f"- {error.get('code')}: {error.get('message')} {error.get('concept_id', '')}".rstrip())
    else:
        lines.append("- none")
    lines.extend(["", "Operations"])
    if plan["operations"]:
        for operation in plan["operations"]:
            lines.append(f"- {operation['concept_id']}: {operation['operation']} ({operation['simulated_effect']})")
    else:
        lines.append("- none")
    lines.extend(["", "Notes"])
    for note in plan.get("claim_boundary_notes", []):
        lines.append(f"- {note}")
    return "\n".join(lines).rstrip() + "\n"


def render_live_repair_apply_markdown(result: dict[str, Any]) -> str:
    lines = [
        "Label-State Reviewed Live Repair Apply",
        f"Status: {result['status']}",
        f"Generated: {result['generated_at']}",
        f"Database: {result['db_path']}",
        f"Mutation allowed: {'yes' if result['mutation_allowed'] else 'no'}",
        f"Operator confirmed: {'yes' if result['operator_confirmed'] else 'no'}",
        f"Packet hash: {result.get('packet_filter_hash')}",
        f"Expected packet hash: {result.get('expected_packet_hash')}",
        f"Boundary: {result['claim_boundary']}",
        "",
        "Apply Result",
        f"- applied_count: {result.get('applied_count', 0)}",
        f"- changed_rows: {', '.join(result.get('changed_rows', [])) or 'none'}",
        f"- rolled_back_rows: {', '.join(result.get('rolled_back_rows', [])) or 'none'}",
    ]
    expected_reduction = result.get("expected_reduction", {})
    if isinstance(expected_reduction, Mapping):
        lines.append(f"- expected_reduction: {dict(expected_reduction)}")
    backup = result.get("backup")
    lines.extend(["", "Backup"])
    if isinstance(backup, Mapping):
        lines.append(f"- path: {backup.get('path')}")
        lines.append(f"- quick_check: {backup.get('quick_check')}")
        lines.append(f"- concept_count: {backup.get('concept_count')}")
        lines.append(f"- pruned_paths: {', '.join(backup.get('pruned_paths', [])) or 'none'}")
    else:
        lines.append("- none")
    lines.extend(["", "Counts"])
    lines.append(f"- before: {result.get('before_counts', {})}")
    lines.append(f"- after: {result.get('after_counts', {})}")
    lines.extend(["", "Validation Errors"])
    errors = result.get("validation_errors", [])
    if errors:
        for error in errors:
            lines.append(f"- {error.get('code')}: {error.get('message')} {error.get('concept_id', '')}".rstrip())
    else:
        lines.append("- none")
    lines.extend(["", "Validation Warnings"])
    warnings = result.get("validation_warnings", [])
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning.get('code')}: {warning.get('message')} {warning.get('concept_id', '')}".rstrip())
    else:
        lines.append("- none")
    lines.extend(["", "Rollback"])
    lines.append(f"- {result.get('rollback_command') or 'none'}")
    lines.extend(["", "Notes"])
    for note in result.get("claim_boundary_notes", [CLAIM_BOUNDARY]):
        lines.append(f"- {note}")
    return "\n".join(lines).rstrip() + "\n"


def write_outputs(
    report: dict[str, Any],
    out_dir: Path,
    *,
    max_ids_per_group: int = 10,
    review_packet: dict[str, Any] | None = None,
    dry_run_repair_plan: dict[str, Any] | None = None,
    live_repair_apply_result: dict[str, Any] | None = None,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "label_state_semantics_audit.json"
    md_path = out_dir / "label_state_semantics_audit.md"
    operator_packet = build_operator_packet(report, max_ids_per_group=max_ids_per_group)
    operator_json_path = out_dir / "label_state_semantics_operator_packet.json"
    operator_md_path = out_dir / "label_state_semantics_operator_packet.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    operator_json_path.write_text(json.dumps(operator_packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    operator_md_path.write_text(render_operator_summary(operator_packet), encoding="utf-8")
    paths = {
        "json": str(json_path),
        "markdown": str(md_path),
        "operator_packet_json": str(operator_json_path),
        "operator_packet_markdown": str(operator_md_path),
    }
    if review_packet is not None:
        review_json_path = out_dir / "label_state_semantics_review_packet.json"
        review_md_path = out_dir / "label_state_semantics_review_packet.md"
        review_json_path.write_text(json.dumps(review_packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        review_md_path.write_text(render_review_packet_markdown(review_packet), encoding="utf-8")
        paths["review_packet_json"] = str(review_json_path)
        paths["review_packet_markdown"] = str(review_md_path)
    if dry_run_repair_plan is not None:
        plan_json_path = out_dir / "label_state_semantics_dry_run_repair_plan.json"
        plan_md_path = out_dir / "label_state_semantics_dry_run_repair_plan.md"
        plan_json_path.write_text(json.dumps(dry_run_repair_plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        plan_md_path.write_text(render_dry_run_repair_plan_markdown(dry_run_repair_plan), encoding="utf-8")
        paths["dry_run_repair_plan_json"] = str(plan_json_path)
        paths["dry_run_repair_plan_markdown"] = str(plan_md_path)
    if live_repair_apply_result is not None:
        apply_json_path = out_dir / "label_state_semantics_live_repair_apply.json"
        apply_md_path = out_dir / "label_state_semantics_live_repair_apply.md"
        apply_json_path.write_text(
            json.dumps(live_repair_apply_result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        apply_md_path.write_text(render_live_repair_apply_markdown(live_repair_apply_result), encoding="utf-8")
        paths["live_repair_apply_json"] = str(apply_json_path)
        paths["live_repair_apply_markdown"] = str(apply_md_path)
    return paths
