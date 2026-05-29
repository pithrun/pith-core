"""DATA-070 lifecycle invariant checks and repair utilities."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from app.storage import DB_PATH, apply_lifecycle_transition_conn


def _count(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> int:
    row = conn.execute(sql, params).fetchone()
    return int(row[0]) if row else 0


def _ids(conn: sqlite3.Connection, sql: str, limit: int | None = None) -> list[tuple]:
    query = sql
    params: tuple[Any, ...] = ()
    if limit is not None:
        query = f"{sql} LIMIT ?"
        params = (limit,)
    return list(conn.execute(query, params).fetchall())


def check_lifecycle_invariants(conn: sqlite3.Connection) -> dict[str, int]:
    """Return counts of known lifecycle drift classes."""
    checks = {
        "sql_json_status_mismatch": _count(
            conn,
            """SELECT COUNT(*) FROM concepts
               WHERE json_type(data, '$.status') IS NOT NULL
                 AND json_extract(data, '$.status') != status""",
        ),
        "active_is_current_0": _count(
            conn,
            "SELECT COUNT(*) FROM concepts WHERE status = 'active' AND is_current = 0",
        ),
        "active_superseded_by": _count(
            conn,
            """SELECT COUNT(*) FROM concepts
               WHERE status = 'active'
                 AND superseded_by IS NOT NULL
                 AND superseded_by != ''""",
        ),
        "non_active_is_current_1": _count(
            conn,
            "SELECT COUNT(*) FROM concepts WHERE status != 'active' AND is_current = 1",
        ),
        "superseded_currency_not_superseded": _count(
            conn,
            """SELECT COUNT(*) FROM concepts
               WHERE status = 'superseded'
                 AND COALESCE(currency_status, '') != 'SUPERSEDED'""",
        ),
        "superseded_missing_pointer": _count(
            conn,
            """SELECT COUNT(*) FROM concepts
               WHERE status = 'superseded'
                 AND (superseded_by IS NULL OR superseded_by = '')""",
        ),
        "active_noncurrent_missing_pointer": _count(
            conn,
            """SELECT COUNT(*) FROM concepts
               WHERE status = 'active'
                 AND is_current = 0
                 AND (superseded_by IS NULL OR superseded_by = '')""",
        ),
    }
    checks["total"] = sum(checks.values())
    return checks


def repair_lifecycle_invariants(
    conn: sqlite3.Connection,
    *,
    dry_run: bool = True,
    limit: int | None = None,
) -> dict[str, Any]:
    """Repair deterministic lifecycle drift and report unresolved review cases."""
    before = check_lifecycle_invariants(conn)
    report: dict[str, Any] = {
        "dry_run": dry_run,
        "limit": limit,
        "before": before,
        "repaired": {
            "sql_json_status_mismatch": 0,
            "active_with_pointer_superseded": 0,
            "non_active_current_cleared": 0,
            "superseded_currency_mirrored": 0,
        },
        "unresolved": {
            "active_noncurrent_missing_pointer": before["active_noncurrent_missing_pointer"],
            "superseded_missing_pointer": before["superseded_missing_pointer"],
        },
    }
    if dry_run:
        return report

    mismatch_ids = _ids(
        conn,
        """SELECT id FROM concepts
           WHERE json_type(data, '$.status') IS NOT NULL
             AND json_extract(data, '$.status') != status
           ORDER BY id""",
        limit,
    )
    for (concept_id,) in mismatch_ids:
        conn.execute(
            """UPDATE concepts
               SET data = json_set(COALESCE(data, '{}'), '$.status', status)
               WHERE id = ?""",
            (concept_id,),
        )
        report["repaired"]["sql_json_status_mismatch"] += 1

    supersede_rows = _ids(
        conn,
        """SELECT id, superseded_by FROM concepts
           WHERE status = 'active'
             AND superseded_by IS NOT NULL
             AND superseded_by != ''
           ORDER BY id""",
        limit,
    )
    for concept_id, superseded_by in supersede_rows:
        report["repaired"]["active_with_pointer_superseded"] += apply_lifecycle_transition_conn(
            conn,
            concept_id,
            "supersede",
            superseded_by=superseded_by,
            reason="DATA-070 lifecycle invariant repair",
        )

    non_active_current_ids = _ids(
        conn,
        "SELECT id FROM concepts WHERE status != 'active' AND is_current = 1 ORDER BY id",
        limit,
    )
    for (concept_id,) in non_active_current_ids:
        conn.execute(
            """UPDATE concepts
               SET is_current = 0,
                   data = json_set(COALESCE(data, '{}'), '$.status', status)
               WHERE id = ?""",
            (concept_id,),
        )
        report["repaired"]["non_active_current_cleared"] += 1

    bad_currency_ids = _ids(
        conn,
        """SELECT id FROM concepts
           WHERE status = 'superseded'
             AND COALESCE(currency_status, '') != 'SUPERSEDED'
           ORDER BY id""",
        limit,
    )
    for (concept_id,) in bad_currency_ids:
        conn.execute(
            """UPDATE concepts
               SET currency_status = 'SUPERSEDED',
                   data = json_set(
                       COALESCE(data, '{}'),
                       '$.status', 'superseded',
                       '$.currency_status', 'SUPERSEDED'
                   )
               WHERE id = ?""",
            (concept_id,),
        )
        report["repaired"]["superseded_currency_mirrored"] += 1

    conn.commit()
    report["after"] = check_lifecycle_invariants(conn)
    report["unresolved"] = {
        "active_noncurrent_missing_pointer": report["after"]["active_noncurrent_missing_pointer"],
        "superseded_missing_pointer": report["after"]["superseded_missing_pointer"],
    }
    return report


def _connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def main() -> int:
    parser = argparse.ArgumentParser(description="Check or repair DATA-070 lifecycle invariants.")
    parser.add_argument("--db", default=str(DB_PATH), help="SQLite DB path")
    parser.add_argument("--apply", action="store_true", help="Apply deterministic repairs")
    parser.add_argument("--limit", type=int, default=None, help="Limit rows per repair class")
    args = parser.parse_args()

    with _connect(args.db) as conn:
        result = repair_lifecycle_invariants(conn, dry_run=not args.apply, limit=args.limit)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
