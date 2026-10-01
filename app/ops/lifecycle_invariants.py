"""DATA-070 lifecycle invariant checks and repair utilities."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from app.storage import DB_PATH

LIFECYCLE_CLASSIFIER_SCHEMA_VERSION = 1
LIFECYCLE_PROJECTION_KEYS = (
    "id",
    "status",
    "is_current",
    "currency_status",
    "superseded_by",
    "superseded_at",
    "supersession_reason",
    "json_status",
    "json_currency_status",
    "json_superseded_by",
)
INVARIANT_BUCKETS = (
    "sql_json_status_mismatch",
    "active_is_current_0",
    "active_superseded_by",
    "non_active_is_current_1",
    "superseded_currency_not_superseded",
    "superseded_missing_pointer",
    "active_noncurrent_missing_pointer",
)


def _count(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> int:
    row = conn.execute(sql, params).fetchone()
    return int(row[0]) if row else 0


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


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _optional_column_sql(conn: sqlite3.Connection, column: str) -> str:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(concepts)").fetchall()}
    return column if column in columns else f"NULL AS {column}"


def lifecycle_projection(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    """Return the stable lifecycle-only projection used by DATA-078 hashes."""
    return {key: row[key] for key in LIFECYCLE_PROJECTION_KEYS}


def lifecycle_projection_sha256(row: sqlite3.Row | dict[str, Any]) -> str:
    return _canonical_sha256(lifecycle_projection(row))


def collect_lifecycle_violation_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Collect the exact seven monitor predicates as per-row bucket sets."""
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        f"""SELECT id, status, is_current, currency_status, superseded_by,
                   {_optional_column_sql(conn, "superseded_at")},
                   {_optional_column_sql(conn, "supersession_reason")},
                   data
            FROM concepts
            WHERE (json_type(data, '$.status') IS NOT NULL
                   AND json_extract(data, '$.status') != status)
               OR (status = 'active' AND is_current = 0)
               OR (status = 'active' AND superseded_by IS NOT NULL AND superseded_by != '')
               OR (status != 'active' AND is_current = 1)
               OR (status = 'superseded' AND COALESCE(currency_status, '') != 'SUPERSEDED')
               OR (status = 'superseded' AND (superseded_by IS NULL OR superseded_by = ''))
               OR (status = 'active' AND is_current = 0
                   AND (superseded_by IS NULL OR superseded_by = ''))
            ORDER BY id"""
    ).fetchall()

    result: list[dict[str, Any]] = []
    for row in rows:
        data = json.loads(row["data"]) if row["data"] else {}
        if not isinstance(data, dict):
            raise ValueError(f"concept {row['id']} data must be a JSON object")
        status = row["status"]
        is_current = int(row["is_current"] or 0)
        pointer = row["superseded_by"]
        buckets: list[str] = []
        if "status" in data and data.get("status") != status:
            buckets.append("sql_json_status_mismatch")
        if status == "active" and is_current == 0:
            buckets.append("active_is_current_0")
        if status == "active" and pointer is not None and pointer != "":
            buckets.append("active_superseded_by")
        if status != "active" and is_current == 1:
            buckets.append("non_active_is_current_1")
        if status == "superseded" and (row["currency_status"] or "") != "SUPERSEDED":
            buckets.append("superseded_currency_not_superseded")
        if status == "superseded" and (pointer is None or pointer == ""):
            buckets.append("superseded_missing_pointer")
        if status == "active" and is_current == 0 and (pointer is None or pointer == ""):
            buckets.append("active_noncurrent_missing_pointer")
        projection = {
            "id": row["id"],
            "status": status,
            "is_current": is_current,
            "currency_status": row["currency_status"],
            "superseded_by": pointer,
            "superseded_at": row["superseded_at"],
            "supersession_reason": row["supersession_reason"],
            "json_status": data.get("status"),
            "json_currency_status": data.get("currency_status"),
            "json_superseded_by": data.get("superseded_by"),
        }
        result.append(
            {
                "concept_id": row["id"],
                "buckets": buckets,
                "structural": "superseded_missing_pointer" in buckets,
                "projection": projection,
                "projection_sha256": _canonical_sha256(projection),
            }
        )
    return result


def _bucket_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {bucket: 0 for bucket in INVARIANT_BUCKETS}
    for row in rows:
        for bucket in row["buckets"]:
            counts[bucket] += 1
    return counts


def _validate_reviewed_observations(
    path: str | Path | None,
    structural_rows: list[dict[str, Any]],
) -> tuple[set[str], str | None, list[str]]:
    if not structural_rows:
        return set(), None, []
    if path is None:
        return set(), None, ["observation_file_not_configured"]
    resolved = Path(path)
    if not resolved.is_file():
        return set(), None, ["observation_file_missing"]
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return set(), None, ["observation_file_invalid_json"]
    if not isinstance(payload, dict):
        return set(), None, ["observation_manifest_not_object"]

    errors: list[str] = []
    if payload.get("schema_version") != 1:
        errors.append("observation_schema_unsupported")
    if payload.get("mode") != "reviewed_observations":
        errors.append("observation_mode_not_reviewed")
    for key in ("reviewer", "reviewed_at", "review_statement"):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            errors.append(f"observation_{key}_missing")
    reviewed_at = payload.get("reviewed_at")
    if isinstance(reviewed_at, str) and reviewed_at.strip():
        try:
            parsed_reviewed_at = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
            if parsed_reviewed_at.tzinfo is None or parsed_reviewed_at.utcoffset() is None:
                raise ValueError("timezone missing")
        except ValueError:
            errors.append("observation_reviewed_at_invalid")
    supplied_hash = payload.get("manifest_sha256")
    canonical_payload = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    actual_hash = _canonical_sha256(canonical_payload)
    if not isinstance(supplied_hash, str) or supplied_hash != actual_hash:
        errors.append("observation_manifest_hash_mismatch")

    entries = payload.get("entries")
    if not isinstance(entries, list):
        errors.append("observation_entries_invalid")
        entries = []
    live = {row["concept_id"]: row for row in structural_rows}
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            errors.append("observation_entry_not_object")
            continue
        concept_id = entry.get("concept_id")
        if not isinstance(concept_id, str) or not concept_id:
            errors.append("observation_entry_id_invalid")
            continue
        if concept_id in seen:
            errors.append("observation_entry_duplicate_id")
            continue
        seen.add(concept_id)
        current = live.get(concept_id)
        if current is None:
            errors.append("observation_entry_not_currently_structural")
            continue
        if entry.get("classification") != "provenance_review_required":
            errors.append("observation_entry_classification_invalid")
        if entry.get("lifecycle_sha256") != current["projection_sha256"]:
            errors.append("observation_entry_fingerprint_mismatch")
        if entry.get("lifecycle_projection") != current["projection"]:
            errors.append("observation_entry_projection_mismatch")
    if seen != set(live):
        errors.append("observation_structural_id_set_mismatch")
    if errors:
        return set(), supplied_hash if isinstance(supplied_hash, str) else None, sorted(set(errors))
    return seen, supplied_hash, []


def classify_lifecycle_invariants(
    conn: sqlite3.Connection,
    *,
    reviewed_observations_path: str | Path | None = None,
) -> dict[str, Any]:
    """Separate raw drift from exact fingerprinted structural observations."""
    raw = check_lifecycle_invariants(conn)
    rows = collect_lifecycle_violation_rows(conn)
    per_row_counts = _bucket_counts(rows)
    if any(per_row_counts[bucket] != raw[bucket] for bucket in INVARIANT_BUCKETS):
        raise RuntimeError("lifecycle invariant row classification disagrees with raw checker")

    structural_rows = [row for row in rows if row["structural"]]
    reviewed_ids, manifest_hash, validation_errors = _validate_reviewed_observations(
        reviewed_observations_path,
        structural_rows,
    )
    reviewed_rows = [row for row in structural_rows if row["concept_id"] in reviewed_ids]
    unreviewed_rows = [row for row in structural_rows if row["concept_id"] not in reviewed_ids]
    actionable_rows = [row for row in rows if not row["structural"] or row["concept_id"] not in reviewed_ids]
    actionable_buckets = _bucket_counts(actionable_rows)

    return {
        "schema_version": LIFECYCLE_CLASSIFIER_SCHEMA_VERSION,
        "raw": raw,
        "raw_distinct": len(rows),
        "actionable": {
            "buckets": actionable_buckets,
            "total": sum(actionable_buckets.values()),
            "distinct": len(actionable_rows),
        },
        "reviewed_structural": {
            "total": sum(_bucket_counts(reviewed_rows).values()),
            "distinct": len(reviewed_rows),
        },
        "unreviewed_structural": {
            "total": sum(_bucket_counts(unreviewed_rows).values()),
            "distinct": len(unreviewed_rows),
        },
        "observation_manifest_sha256": manifest_hash,
        "observation_validation_errors": validation_errors,
    }


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
    if not dry_run:
        raise RuntimeError("unguarded lifecycle apply disabled; use DATA-078 ledger")
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
