"""Guarded DATA-078 lifecycle invariant repair and observation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import sqlite3
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from app.core.datetime_utils import _utc_now_iso
from app.ops.lifecycle_invariants import (
    check_lifecycle_invariants,
    classify_lifecycle_invariants,
    collect_lifecycle_violation_rows,
)
from app.storage import DB_PATH

CLASSIFIER_VERSION = "data078-lifecycle-v1"
HASH_RE = re.compile(r"^[a-f0-9]{64}$")


@dataclass(frozen=True)
class RepairDecision:
    concept_id: str
    action: str
    reason: str
    matched_buckets: list[str]
    before: dict[str, Any]
    before_sha256: str
    after: dict[str, Any]


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _sha256_payload(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    supplied = Path(path).expanduser()
    if supplied.is_symlink():
        raise ValueError("artifact output path must not be a symlink")
    target = supplied.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def decisions_sha256(decisions: list[RepairDecision]) -> str:
    return _sha256_payload([asdict(item) for item in sorted(decisions, key=lambda item: item.concept_id)])


def _action_counts(decisions: list[RepairDecision]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for decision in decisions:
        counts[decision.action] = counts.get(decision.action, 0) + 1
    return dict(sorted(counts.items()))


def _projection_sha256(projection: dict[str, Any]) -> str:
    return _sha256_payload(projection)


def _decision_for_row(row: dict[str, Any]) -> RepairDecision | None:
    if row["structural"]:
        return None
    before = dict(row["projection"])
    after = dict(before)
    status = before["status"]
    pointer = before["superseded_by"]
    buckets = list(row["buckets"])

    if status == "active" and pointer is not None and pointer != "":
        action = "terminalize_active_with_pointer"
        reason = "existing_supersession_pointer_is_authoritative"
        after.update(
            {
                "status": "superseded",
                "is_current": 0,
                "currency_status": "SUPERSEDED",
                "json_status": "superseded",
                "json_currency_status": "SUPERSEDED",
                "json_superseded_by": pointer,
            }
        )
    elif status != "active" and int(before["is_current"] or 0) == 1:
        action = "normalize_nonactive_current"
        reason = "nonactive_sql_status_is_authoritative"
        after["is_current"] = 0
        after["json_status"] = status
        if status == "superseded" and pointer not in (None, ""):
            after["currency_status"] = "SUPERSEDED"
            after["json_currency_status"] = "SUPERSEDED"
            after["json_superseded_by"] = pointer
    elif status == "superseded" and pointer not in (None, "") and before["currency_status"] != "SUPERSEDED":
        action = "normalize_superseded_currency"
        reason = "superseded_sql_status_and_pointer_are_authoritative"
        after.update(
            {
                "currency_status": "SUPERSEDED",
                "json_status": "superseded",
                "json_currency_status": "SUPERSEDED",
                "json_superseded_by": pointer,
            }
        )
    else:
        raise ValueError(f"unclassified actionable lifecycle row: {row['concept_id']} buckets={buckets}")

    return RepairDecision(
        concept_id=row["concept_id"],
        action=action,
        reason=reason,
        matched_buckets=buckets,
        before=before,
        before_sha256=row["projection_sha256"],
        after=after,
    )


def build_repair_decisions(conn: sqlite3.Connection) -> list[RepairDecision]:
    decisions = [decision for row in collect_lifecycle_violation_rows(conn) if (decision := _decision_for_row(row))]
    return sorted(decisions, key=lambda item: item.concept_id)


def _source_fingerprint(db_path: Path) -> dict[str, Any]:
    stat = db_path.stat()
    return {
        "db_path": str(db_path.resolve()),
        "db_size_bytes": stat.st_size,
        "db_mtime_ns": stat.st_mtime_ns,
    }


def build_repair_ledger(conn: sqlite3.Connection, *, db_path: str | Path) -> dict[str, Any]:
    decisions = build_repair_decisions(conn)
    raw = check_lifecycle_invariants(conn)
    structural = [row for row in collect_lifecycle_violation_rows(conn) if row["structural"]]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "mode": "repair_ledger",
        "classifier_version": CLASSIFIER_VERSION,
        "generated_at": _utc_now_iso(),
        "source": _source_fingerprint(Path(db_path)),
        "raw": raw,
        "actionable_total": sum(len(item.matched_buckets) for item in decisions),
        "actionable_distinct": len(decisions),
        "structural_total": sum(len(row["buckets"]) for row in structural),
        "structural_distinct": len(structural),
        "action_counts": _action_counts(decisions),
        "decision_count": len(decisions),
        "decisions": [asdict(item) for item in decisions],
    }
    payload["ledger_sha256"] = decisions_sha256(decisions)
    return payload


def build_observation_template(conn: sqlite3.Connection, *, db_path: str | Path) -> dict[str, Any]:
    structural = [row for row in collect_lifecycle_violation_rows(conn) if row["structural"]]
    entries = []
    for row in structural:
        prior_review = int(row["projection"]["is_current"] or 0) == 1
        entries.append(
            {
                "concept_id": row["concept_id"],
                "classification": "provenance_review_required",
                "matched_buckets": row["buckets"],
                "lifecycle_projection": row["projection"],
                "lifecycle_sha256": row["projection_sha256"],
                "evidence_reason": (
                    "prior_lifecycle_review_currentness_decision" if prior_review else "unknown_successor_identity"
                ),
                "evidence_reference": ("MAINT-077/MAINT-088" if prior_review else "DATA-078 RCA structural boundary"),
            }
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "mode": "observation_template",
        "classifier_version": CLASSIFIER_VERSION,
        "generated_at": _utc_now_iso(),
        "source": _source_fingerprint(Path(db_path)),
        "structural_total": sum(len(entry["matched_buckets"]) for entry in entries),
        "structural_distinct": len(entries),
        "entries": entries,
    }
    payload["template_sha256"] = _sha256_payload(payload)
    return payload


def finalize_observations(
    template_path: str | Path,
    *,
    approved_template_sha256: str,
    reviewer: str,
    reviewed_at: str,
    review_statement: str,
) -> dict[str, Any]:
    if not HASH_RE.fullmatch(approved_template_sha256):
        raise ValueError("approved template SHA-256 must be 64 lowercase hex")
    template = json.loads(Path(template_path).read_text(encoding="utf-8"))
    if template.get("mode") != "observation_template" or template.get("schema_version") != 1:
        raise ValueError("observation template schema or mode invalid")
    supplied = template.get("template_sha256")
    canonical = {key: value for key, value in template.items() if key != "template_sha256"}
    actual = _sha256_payload(canonical)
    if not isinstance(supplied, str) or not hmac.compare_digest(supplied, actual):
        raise ValueError("observation template hash mismatch")
    if not hmac.compare_digest(actual, approved_template_sha256):
        raise ValueError("unapproved observation template hash")
    try:
        parsed_reviewed_at = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("reviewed_at must be a timezone-aware ISO timestamp") from exc
    if (
        not reviewer.strip()
        or not review_statement.strip()
        or parsed_reviewed_at.tzinfo is None
        or parsed_reviewed_at.utcoffset() is None
    ):
        raise ValueError("reviewer, timezone-aware reviewed_at, and review statement are required")
    payload = {
        "schema_version": 1,
        "mode": "reviewed_observations",
        "classifier_version": template.get("classifier_version"),
        "approved_template_sha256": actual,
        "reviewer": reviewer.strip(),
        "reviewed_at": reviewed_at,
        "review_statement": review_statement.strip(),
        "source": template.get("source"),
        "entries": template.get("entries"),
    }
    payload["manifest_sha256"] = _sha256_payload(payload)
    return payload


def _load_approved_ledger(path: str | Path, approved_sha256: str) -> tuple[dict[str, Any], list[RepairDecision]]:
    if not HASH_RE.fullmatch(approved_sha256):
        raise ValueError("approved ledger SHA-256 must be 64 lowercase hex")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("mode") != "repair_ledger":
        raise ValueError("repair ledger schema or mode invalid")
    decisions = [RepairDecision(**item) for item in payload.get("decisions", [])]
    actual = decisions_sha256(decisions)
    supplied = payload.get("ledger_sha256")
    if not isinstance(supplied, str) or not hmac.compare_digest(supplied, actual):
        raise ValueError("repair ledger internal hash mismatch")
    if not hmac.compare_digest(actual, approved_sha256):
        raise ValueError("unapproved repair ledger hash")
    return payload, decisions


def _open_readonly(path: str | Path) -> sqlite3.Connection:
    resolved = Path(path).expanduser().resolve()
    conn = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _integrity(conn: sqlite3.Connection) -> str:
    row = conn.execute("PRAGMA integrity_check").fetchone()
    return str(row[0]) if row else "missing"


def _structural_fingerprints(conn: sqlite3.Connection) -> dict[str, str]:
    return {
        row["concept_id"]: row["projection_sha256"]
        for row in collect_lifecycle_violation_rows(conn)
        if row["structural"]
    }


def _verify_backup(
    backup_path: str | Path,
    *,
    backup_sha256: str,
    approved_decisions: list[RepairDecision],
    structural_fingerprints: dict[str, str],
) -> dict[str, Any]:
    path = Path(backup_path).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError("backup path must be a regular non-symlink file")
    if not HASH_RE.fullmatch(backup_sha256):
        raise ValueError("backup SHA-256 must be 64 lowercase hex")
    actual_sha = _file_sha256(path)
    if not hmac.compare_digest(actual_sha, backup_sha256):
        raise ValueError("backup SHA-256 mismatch")
    with _open_readonly(path) as conn:
        if _integrity(conn) != "ok":
            raise ValueError("backup integrity check failed")
        backup_decisions = build_repair_decisions(conn)
        if not hmac.compare_digest(decisions_sha256(backup_decisions), decisions_sha256(approved_decisions)):
            raise ValueError("backup repair decision state mismatch")
        if _structural_fingerprints(conn) != structural_fingerprints:
            raise ValueError("backup structural state mismatch")
        concept_count = int(conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0])
    return {"path": str(path), "sha256": actual_sha, "concept_count": concept_count, "integrity": "ok"}


def _active_membership(projection: dict[str, Any]) -> bool:
    return projection["status"] == "active" and int(projection["is_current"] or 0) == 1


def apply_approved_repairs(
    conn: sqlite3.Connection,
    *,
    ledger_path: str | Path,
    approved_ledger_sha256: str,
    reviewed_observations_path: str | Path,
    approved_observations_sha256: str,
    backup_path: str | Path,
    backup_sha256: str,
) -> dict[str, Any]:
    ledger, approved_decisions = _load_approved_ledger(ledger_path, approved_ledger_sha256)
    observations = json.loads(Path(reviewed_observations_path).read_text(encoding="utf-8"))
    supplied_observation_hash = observations.get("manifest_sha256")
    if not HASH_RE.fullmatch(approved_observations_sha256):
        raise ValueError("approved observations SHA-256 must be 64 lowercase hex")
    if not isinstance(supplied_observation_hash, str) or not hmac.compare_digest(
        supplied_observation_hash, approved_observations_sha256
    ):
        raise ValueError("unapproved observations manifest hash")
    canonical_observations = {key: value for key, value in observations.items() if key != "manifest_sha256"}
    if not hmac.compare_digest(_sha256_payload(canonical_observations), approved_observations_sha256):
        raise ValueError("reviewed observations internal hash mismatch")
    observation_entries = observations.get("entries")
    if not isinstance(observation_entries, list):
        raise ValueError("reviewed observations entries invalid")
    approved_structural = {
        entry["concept_id"]: entry["lifecycle_sha256"]
        for entry in observation_entries
        if isinstance(entry, dict)
        and isinstance(entry.get("concept_id"), str)
        and isinstance(entry.get("lifecycle_sha256"), str)
    }

    conn.row_factory = sqlite3.Row
    if _integrity(conn) != "ok":
        raise ValueError("target integrity check failed")
    backup = _verify_backup(
        backup_path,
        backup_sha256=backup_sha256,
        approved_decisions=approved_decisions,
        structural_fingerprints=approved_structural,
    )

    trigger = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='trigger' AND name='trg_concepts_lifecycle_index_membership'"
    ).fetchone()
    if trigger is None:
        raise ValueError("lifecycle index outbox trigger is not installed")
    outbox_before = int(conn.execute("SELECT COUNT(*) FROM lifecycle_index_outbox").fetchone()[0])
    expected_outbox = sum(
        _active_membership(item.before) != _active_membership(item.after) for item in approved_decisions
    )

    changed = 0
    now = _utc_now_iso()
    conn.execute("BEGIN IMMEDIATE")
    try:
        current_rows = collect_lifecycle_violation_rows(conn)
        current_by_id = {row["concept_id"]: row for row in current_rows}
        current_decisions = [decision for row in current_rows if (decision := _decision_for_row(row))]
        current_sha = decisions_sha256(current_decisions)
        if not hmac.compare_digest(current_sha, approved_ledger_sha256):
            raise ValueError(f"classifier drift or unapproved target state: current sha {current_sha}")
        structural_before = _structural_fingerprints(conn)
        if structural_before != approved_structural:
            raise ValueError("target structural state does not match approved observations")
        classified = classify_lifecycle_invariants(
            conn,
            reviewed_observations_path=reviewed_observations_path,
        )
        if classified["observation_validation_errors"]:
            raise ValueError(f"reviewed observations invalid: {classified['observation_validation_errors']}")
        for decision in approved_decisions:
            current_row = current_by_id.get(decision.concept_id)
            if current_row is None:
                raise ValueError(f"repair target is no longer actionable: {decision.concept_id}")
            before = current_row["projection"]
            if not hmac.compare_digest(_projection_sha256(before), decision.before_sha256):
                raise ValueError(f"repair target before-state drift: {decision.concept_id}")
            data_row = conn.execute("SELECT data FROM concepts WHERE id = ?", (decision.concept_id,)).fetchone()
            data = json.loads(data_row[0]) if data_row and data_row[0] else {}
            if not isinstance(data, dict):
                raise ValueError(f"repair target JSON is not an object: {decision.concept_id}")
            data["status"] = decision.after["json_status"]
            data["currency_status"] = decision.after["json_currency_status"]
            if decision.after["json_superseded_by"] is None:
                data.pop("superseded_by", None)
            else:
                data["superseded_by"] = decision.after["json_superseded_by"]
            cursor = conn.execute(
                """UPDATE concepts
                   SET status = :after_status,
                       is_current = :after_is_current,
                       currency_status = :after_currency_status,
                       superseded_by = :after_superseded_by,
                       data = :after_data,
                       updated_at = :now
                   WHERE id = :concept_id
                     AND status IS :before_status
                     AND is_current IS :before_is_current
                     AND currency_status IS :before_currency_status
                     AND superseded_by IS :before_superseded_by""",
                {
                    "after_status": decision.after["status"],
                    "after_is_current": decision.after["is_current"],
                    "after_currency_status": decision.after["currency_status"],
                    "after_superseded_by": decision.after["superseded_by"],
                    "after_data": json.dumps(data, sort_keys=True),
                    "now": now,
                    "concept_id": decision.concept_id,
                    "before_status": decision.before["status"],
                    "before_is_current": decision.before["is_current"],
                    "before_currency_status": decision.before["currency_status"],
                    "before_superseded_by": decision.before["superseded_by"],
                },
            )
            if cursor.rowcount != 1:
                raise ValueError(f"repair target update lost exact before state: {decision.concept_id}")
            conn.execute("DELETE FROM fts_concepts WHERE concept_id = ?", (decision.concept_id,))
            conn.execute("DELETE FROM fts_verbatim WHERE concept_id = ?", (decision.concept_id,))
            changed += 1

        after = classify_lifecycle_invariants(
            conn,
            reviewed_observations_path=reviewed_observations_path,
        )
        if after["actionable"]["total"] != 0:
            raise ValueError(f"post-repair actionable lifecycle drift remains: {after['actionable']}")
        if _structural_fingerprints(conn) != structural_before:
            raise ValueError("post-repair structural fingerprints changed")
        if changed != len(approved_decisions):
            raise ValueError("changed row count does not match approved decision count")
        outbox_after = int(conn.execute("SELECT COUNT(*) FROM lifecycle_index_outbox").fetchone()[0])
        if outbox_after - outbox_before != expected_outbox:
            raise ValueError("lifecycle index outbox event count mismatch")
        if str(conn.execute("PRAGMA quick_check").fetchone()[0]) != "ok":
            raise ValueError("post-repair quick check failed")
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    return {
        "schema_version": 1,
        "mode": "apply",
        "applied_at": now,
        "ledger_sha256": ledger["ledger_sha256"],
        "observation_manifest_sha256": supplied_observation_hash,
        "backup": backup,
        "changed_rows": changed,
        "expected_outbox_events": expected_outbox,
        "before": classified,
        "after": after,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Guarded DATA-078 lifecycle invariant repair")
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--ledger-path")
    parser.add_argument("--observation-template-path")
    parser.add_argument("--finalize-observations", action="store_true")
    parser.add_argument("--approved-template-sha256", default="")
    parser.add_argument("--reviewed-observations-path")
    parser.add_argument("--reviewer", default="")
    parser.add_argument("--reviewed-at", default="")
    parser.add_argument("--review-statement", default="")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--approved-ledger-sha256", default="")
    parser.add_argument("--approved-observations-sha256", default="")
    parser.add_argument("--backup-path")
    parser.add_argument("--backup-sha256", default="")
    parser.add_argument("--apply-report-path")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.finalize_observations:
        if not args.observation_template_path or not args.reviewed_observations_path:
            raise SystemExit("finalize requires observation template and reviewed output paths")
        payload = finalize_observations(
            args.observation_template_path,
            approved_template_sha256=args.approved_template_sha256,
            reviewer=args.reviewer,
            reviewed_at=args.reviewed_at,
            review_statement=args.review_statement,
        )
        _atomic_write_json(args.reviewed_observations_path, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.apply:
        required = (
            args.ledger_path,
            args.reviewed_observations_path,
            args.backup_path,
            args.apply_report_path,
        )
        if not all(required):
            raise SystemExit("apply requires ledger, reviewed observations, backup, and report paths")
        conn = sqlite3.connect(str(Path(args.db).expanduser().resolve()))
        conn.row_factory = sqlite3.Row
        try:
            payload = apply_approved_repairs(
                conn,
                ledger_path=args.ledger_path,
                approved_ledger_sha256=args.approved_ledger_sha256,
                reviewed_observations_path=args.reviewed_observations_path,
                approved_observations_sha256=args.approved_observations_sha256,
                backup_path=args.backup_path,
                backup_sha256=args.backup_sha256,
            )
        finally:
            conn.close()
        _atomic_write_json(args.apply_report_path, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if not args.ledger_path or not args.observation_template_path:
        raise SystemExit("dry-run requires --ledger-path and --observation-template-path")
    with _open_readonly(args.db) as conn:
        conn.execute("BEGIN")
        ledger = build_repair_ledger(conn, db_path=args.db)
        observations = build_observation_template(conn, db_path=args.db)
        conn.rollback()
    _atomic_write_json(args.ledger_path, ledger)
    _atomic_write_json(args.observation_template_path, observations)
    report = {
        "mode": "dry_run",
        "ledger_path": str(Path(args.ledger_path).resolve()),
        "ledger_sha256": ledger["ledger_sha256"],
        "decision_count": ledger["decision_count"],
        "action_counts": ledger["action_counts"],
        "observation_template_path": str(Path(args.observation_template_path).resolve()),
        "observation_template_sha256": observations["template_sha256"],
        "structural_distinct": observations["structural_distinct"],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
