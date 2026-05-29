"""DATA-071 pointerless lifecycle repair utility.

Dry-run emits a deterministic row-level ledger. Apply requires the approved
ledger hash, backup confirmation, and explicit allowlist for active restores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.core.datetime_utils import _utc_now_iso
from app.ops.lifecycle_invariants import check_lifecycle_invariants
from app.ops.maintenance import ORPHANED_SUPERSESSION_SENTINEL
from app.storage import DB_PATH

HASH_RE = re.compile(r"^[a-f0-9]{64}$")
APPLY_TIME_PLACEHOLDER = "<apply_time>"


@dataclass(frozen=True)
class Decision:
    concept_id: str
    action: str
    reason: str
    before: dict[str, Any]
    after: dict[str, Any]
    edge_count: int
    active_current_superseder_count: int
    active_current_superseder_id: str | None = None


def _row_dict(row: sqlite3.Row) -> dict[str, Any]:
    keys = (
        "id",
        "status",
        "is_current",
        "currency_status",
        "superseded_by",
        "superseded_at",
        "supersession_reason",
        "always_activate",
        "protected",
        "concept_type",
        "summary",
    )
    return {key: row[key] for key in keys if key in row.keys()}


def _canonical_payload(decisions: list[Decision]) -> str:
    return json.dumps(
        [asdict(decision) for decision in sorted(decisions, key=lambda item: item.concept_id)],
        sort_keys=True,
        separators=(",", ":"),
    )


def decisions_sha256(decisions: list[Decision]) -> str:
    return hashlib.sha256(_canonical_payload(decisions).encode("utf-8")).hexdigest()


def classify_rows(conn: sqlite3.Connection) -> list[Decision]:
    conn.row_factory = sqlite3.Row
    decisions: list[Decision] = []

    active_rows = conn.execute(
        """SELECT id, status, is_current, currency_status, superseded_by, superseded_at,
                  supersession_reason, always_activate, protected, concept_type, summary
           FROM concepts
           WHERE status = 'active'
             AND is_current = 0
             AND (superseded_by IS NULL OR superseded_by = '')
           ORDER BY id"""
    ).fetchall()
    for row in active_rows:
        before = _row_dict(row)
        protected = bool(row["always_activate"]) or bool(row["protected"]) or row["concept_type"] == "constraint"
        if protected:
            after = dict(before)
            after.update(
                {
                    "status": "active",
                    "is_current": 1,
                    "currency_status": "ACTIVE",
                    "superseded_by": None,
                    "superseded_at": None,
                    "updated_at": APPLY_TIME_PLACEHOLDER,
                }
            )
            decisions.append(
                Decision(
                    row["id"],
                    "restore_active_current_manual_allowlist",
                    "active non-current protected/constraint row requires explicit restore allowlist",
                    before,
                    after,
                    0,
                    0,
                )
            )
        else:
            after = dict(before)
            after.update({"status": "archived", "is_current": 0, "updated_at": APPLY_TIME_PLACEHOLDER})
            decisions.append(
                Decision(
                    row["id"],
                    "archive_active_noncurrent_no_successor",
                    "active non-current row has no pointer and no protected/constraint signal",
                    before,
                    after,
                    0,
                    0,
                )
            )

    superseded_rows = conn.execute(
        """SELECT c.id, c.status, c.is_current, c.currency_status, c.superseded_by,
                  c.superseded_at, c.supersession_reason, c.always_activate,
                  c.protected, c.concept_type, c.summary,
                  COUNT(a.source) AS edge_count,
                  SUM(CASE WHEN s.status = 'active' AND s.is_current = 1 THEN 1 ELSE 0 END)
                      AS active_current_superseder_count,
                  GROUP_CONCAT(CASE WHEN s.status = 'active' AND s.is_current = 1 THEN s.id END)
                      AS active_current_superseder_ids
           FROM concepts c
           LEFT JOIN associations a ON a.target = c.id AND a.relation = 'supersedes'
           LEFT JOIN concepts s ON s.id = a.source
           WHERE c.status = 'superseded'
             AND (c.superseded_by IS NULL OR c.superseded_by = '')
           GROUP BY c.id
           ORDER BY c.id"""
    ).fetchall()
    for row in superseded_rows:
        before = _row_dict(row)
        edge_count = int(row["edge_count"] or 0)
        active_count = int(row["active_current_superseder_count"] or 0)
        if active_count == 1:
            superseder_id = str(row["active_current_superseder_ids"])
            after = dict(before)
            after.update(
                {
                    "status": "superseded",
                    "is_current": 0,
                    "currency_status": "SUPERSEDED",
                    "superseded_by": superseder_id,
                    "superseded_at": row["superseded_at"] or APPLY_TIME_PLACEHOLDER,
                    "updated_at": APPLY_TIME_PLACEHOLDER,
                }
            )
            decisions.append(
                Decision(
                    row["id"],
                    "recover_unique_active_current_superseder",
                    "exactly one active-current supersedes edge points at this row",
                    before,
                    after,
                    edge_count,
                    active_count,
                    superseder_id,
                )
            )
        else:
            after = dict(before)
            after.update(
                {
                    "status": "superseded",
                    "is_current": 0,
                    "currency_status": "SUPERSEDED",
                    "superseded_by": ORPHANED_SUPERSESSION_SENTINEL,
                    "updated_at": APPLY_TIME_PLACEHOLDER,
                }
            )
            decisions.append(
                Decision(
                    row["id"],
                    "mark_orphan_sentinel",
                    "no unique active-current superseder can be recovered",
                    before,
                    after,
                    edge_count,
                    active_count,
                )
            )
    return decisions


def write_ledger(decisions: list[Decision], path: str | Path) -> str:
    sha = decisions_sha256(decisions)
    payload = {
        "sha256": sha,
        "decision_count": len(decisions),
        "action_counts": _action_counts(decisions),
        "decisions": [asdict(decision) for decision in decisions],
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sha


def _action_counts(decisions: list[Decision]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for decision in decisions:
        counts[decision.action] = counts.get(decision.action, 0) + 1
    return dict(sorted(counts.items()))


def _json_status_sql(*, remove_superseded_by: bool = False) -> str:
    if remove_superseded_by:
        return (
            "data = json_remove(json_set(COALESCE(data, '{}'), "
            "'$.status', :status, '$.currency_status', :currency_status), '$.superseded_by')"
        )
    return (
        "data = json_set(COALESCE(data, '{}'), '$.status', :status, "
        "'$.currency_status', :currency_status, '$.superseded_by', :superseded_by)"
    )


def apply_decisions(
    conn: sqlite3.Connection,
    *,
    approved_ledger_sha256: str,
    backup_confirmed: bool,
    restore_allowlist: set[str] | None = None,
) -> dict[str, Any]:
    if not backup_confirmed:
        raise ValueError("--backup-confirmed is required for apply")
    if not HASH_RE.fullmatch(approved_ledger_sha256):
        raise ValueError("--approved-ledger-sha256 must be a 64-character lowercase hex digest")

    before = check_lifecycle_invariants(conn)
    decisions = classify_rows(conn)
    current_sha = decisions_sha256(decisions)
    if current_sha != approved_ledger_sha256:
        raise ValueError(f"classifier drift or unapproved ledger: current sha {current_sha}")

    restore_allowlist = restore_allowlist or set()
    changed = 0
    now = _utc_now_iso()
    with conn:
        for decision in decisions:
            params = {
                "id": decision.concept_id,
                "now": now,
                "status": decision.after["status"],
                "currency_status": decision.after.get("currency_status"),
                "superseded_by": decision.after.get("superseded_by"),
                "reason": f"DATA-071 {decision.action}",
            }
            if decision.action == "restore_active_current_manual_allowlist":
                if decision.concept_id not in restore_allowlist:
                    raise ValueError(f"restore decision {decision.concept_id} missing --allow-restore-id")
                cursor = conn.execute(
                    f"""UPDATE concepts
                        SET status = 'active',
                            is_current = 1,
                            currency_status = 'ACTIVE',
                            superseded_by = NULL,
                            superseded_at = NULL,
                            supersession_reason = NULL,
                            updated_at = :now,
                            {_json_status_sql(remove_superseded_by=True)}
                        WHERE id = :id
                          AND status = 'active'
                          AND is_current = 0
                          AND (superseded_by IS NULL OR superseded_by = '')""",
                    params,
                )
            elif decision.action == "archive_active_noncurrent_no_successor":
                cursor = conn.execute(
                    """UPDATE concepts
                       SET status = 'archived',
                           is_current = 0,
                           updated_at = :now,
                           data = json_set(COALESCE(data, '{}'), '$.status', 'archived')
                       WHERE id = :id
                         AND status = 'active'
                         AND is_current = 0
                         AND (superseded_by IS NULL OR superseded_by = '')""",
                    params,
                )
            elif decision.action in {"recover_unique_active_current_superseder", "mark_orphan_sentinel"}:
                cursor = conn.execute(
                    f"""UPDATE concepts
                        SET status = 'superseded',
                            is_current = 0,
                            currency_status = 'SUPERSEDED',
                            superseded_by = :superseded_by,
                            superseded_at = COALESCE(superseded_at, :now),
                            supersession_reason = COALESCE(supersession_reason, :reason),
                            updated_at = :now,
                            {_json_status_sql()}
                        WHERE id = :id
                          AND status = 'superseded'
                          AND (superseded_by IS NULL OR superseded_by = '')""",
                    params,
                )
            else:  # pragma: no cover - defensive guard for future actions
                raise ValueError(f"unknown DATA-071 action: {decision.action}")
            changed += cursor.rowcount

    after = check_lifecycle_invariants(conn)
    return {
        "before": before,
        "after": after,
        "action_counts": _action_counts(decisions),
        "changed_rows": changed,
        "ledger_sha256": current_sha,
    }


def _connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def main() -> int:
    parser = argparse.ArgumentParser(description="DATA-071 pointerless lifecycle repair")
    parser.add_argument("--db", default=str(DB_PATH), help="SQLite DB path")
    parser.add_argument("--ledger-path", required=True, help="Path for dry-run/apply ledger JSON")
    parser.add_argument("--apply", action="store_true", help="Apply approved decisions")
    parser.add_argument("--approved-ledger-sha256", default="", help="Approved canonical ledger SHA-256")
    parser.add_argument("--backup-confirmed", action="store_true", help="Confirm fresh DATA-071 backup exists")
    parser.add_argument("--allow-restore-id", action="append", default=[], help="Allow active-current restore for id")
    args = parser.parse_args()

    with _connect(args.db) as conn:
        if args.apply:
            report = apply_decisions(
                conn,
                approved_ledger_sha256=args.approved_ledger_sha256,
                backup_confirmed=args.backup_confirmed,
                restore_allowlist=set(args.allow_restore_id),
            )
            report["mode"] = "apply"
            report["ledger_path"] = args.ledger_path
        else:
            decisions = classify_rows(conn)
            sha = write_ledger(decisions, args.ledger_path)
            report = {
                "mode": "dry_run",
                "ledger_path": args.ledger_path,
                "ledger_sha256": sha,
                "action_counts": _action_counts(decisions),
                "decision_count": len(decisions),
                "before": check_lifecycle_invariants(conn),
            }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
