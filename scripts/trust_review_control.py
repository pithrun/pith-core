#!/usr/bin/env python3
"""User-facing implicit supersession review workflow."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sqlite3
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ops.implicit_supersession_proposals import CLAIM_BOUNDARY, RETENTION_MODES, REVIEW_DECISIONS
from scripts.implicit_supersession_proposals_scorecard import build_scorecard
from scripts.ingest_implicit_supersession_review_packet import build_extension

SCHEMA_VERSION = "trust_review_control.v1"
ERROR_SCHEMA_VERSION = "trust_review_control_error.v1"
CALIBRATION_SCHEMA_VERSION = "trust_review_calibration.v1"
FOCUS_SCHEMA_VERSION = "trust_review_focus.v1"
EDGE_QUEUE_SCHEMA_VERSION = "trust_review_edge_queue.v1"
EDGE_FOCUS_SCHEMA_VERSION = "trust_review_edge_focus.v1"
EDGE_DECISIONS_SCHEMA_VERSION = "trust_review_edge_decisions.v1"
EDGE_BATCH_SCHEMA_VERSION = "trust_review_edge_batch.v1"
EDGE_CONSUME_SCHEMA_VERSION = "trust_review_decision_consume.v1"
EDGE_CONSUME_POINTER_SCHEMA_VERSION = "trust_review_consume_pointer.v1"
PACKET_SCHEMA_VERSION = "implicit_supersession_proposals.v1"
EDGE_REPORT_MODE = "supersession_edge_semantics_audit"
EDGE_REVIEW_CLAIM_BOUNDARY = "Edge-risk review decisions are read-only evidence; they do not mutate Pith authority."
EDGE_CONSUME_CLAIM_BOUNDARY = (
    "Edge-risk decision consume is a measured dry-run bridge; it does not mutate Pith authority."
)
DEFAULT_REPORTS_DIR = Path.home() / ".pith" / "reports" / "monitoring"
DEFAULT_REVIEW_EXTENSION_DIR = DEFAULT_REPORTS_DIR / "implicit-supersession-review-gold"
DEFAULT_EDGE_DECISION_DIR = DEFAULT_REPORTS_DIR / "trust-review-edge-decisions"
DEFAULT_EDGE_CONSUME_DIR = DEFAULT_REPORTS_DIR / "trust-review-decision-consumer"
DEFAULT_EDGE_BATCH_DIR = DEFAULT_REPORTS_DIR / "trust-review-edge-batches"
DEFAULT_CALIBRATION_DIR = DEFAULT_REPORTS_DIR / "implicit-supersession-review-calibration"
DEFAULT_GOLD_PATH = ROOT / "scripts" / "eval" / "implicit_supersession_proposals_gold.json"
MAX_CANDIDATES = 500
DEFAULT_EDGE_REVIEW_LIMIT = 20
DEFAULT_EDGE_REVIEW_BATCH_LIMIT = 10
MAX_EDGE_DECISION_ARTIFACT_BYTES = 1_048_576
MAX_EDGE_SOURCE_REPORT_BYTES = 67_108_864
PROPOSAL_QUEUE_ALIASES = {"", "proposal", "proposals", "implicit", "implicit-proposals"}
EDGE_RISK_QUEUE_ALIASES = {"edge", "edge-risk", "supersession-edge-risk"}
EDGE_REVIEW_DECISIONS = {"approve_repair", "reject_repair", "needs_context", "not_supersession", "uncertain"}
EDGE_REPAIR_DECISIONS_REQUIRING_RATIONALE = {"approve_repair"}
EDGE_BATCH_STOP_RULE = (
    "Stop if source_report_path, source_generated_at, or source_report_sha256 differs from the current "
    "edge-risk report; regenerate the batch instead of editing or consuming stale decisions."
)
REVIEW_OPERATIONS = {
    "review_next",
    "review_status",
    "review_list",
    "review_show",
    "review_focus",
    "review_batch",
    "review_decide",
    "review_run",
    "review_export",
    "review_consume",
    "review_measure",
    "review_calibration",
}
GENERIC_SEQUENCE_SUBJECT_MARKERS = {
    "next recommended move",
    "the next recommended move",
    "the correct next step",
    "next step",
    "next gate",
    "next lane",
    "active lane",
    "proceed with",
    "lets proceed",
    "let's proceed",
    "resolve the blocked",
}
CALIBRATION_TSV_FIELDS = [
    "old_id",
    "new_id",
    "subject_key",
    "risk_class",
    "disposition",
    "score",
    "old_created_at",
    "new_created_at",
    "first_seen_at",
    "latest_seen_at",
    "source_packet_count",
    "source_packet_paths",
    "signals",
    "evidence_complete",
    "old_summary",
    "new_summary",
    "review_schema_version",
    "review_group_key",
    "review_decision",
    "retention_mode",
    "reviewer",
    "reviewed_at",
    "review_rationale",
    "review_source",
    "expected_disposition",
]


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _error(message: str, *, code: str, field: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "error": True,
        "schema_version": ERROR_SCHEMA_VERSION,
        "code": code,
        "message": message,
        "mutates_authority": False,
        "mutation_count": 0,
    }
    if field:
        result["field"] = field
    return result


def _expand_path(value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    return Path(text).expanduser() if text else None


def _payload_int(
    payload: dict[str, Any], field: str, default: int, *, minimum: int = 0
) -> tuple[int | None, dict[str, Any] | None]:
    if field not in payload or payload[field] is None:
        return default, None
    value = payload[field]
    if isinstance(value, bool):
        return None, _error(f"{field} must be an integer >= {minimum}", code="INVALID_REVIEW_PAYLOAD", field=field)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None, _error(f"{field} must be an integer >= {minimum}", code="INVALID_REVIEW_PAYLOAD", field=field)
    if parsed < minimum:
        return None, _error(f"{field} must be an integer >= {minimum}", code="INVALID_REVIEW_PAYLOAD", field=field)
    return parsed, None


def _reports_dir(payload: dict[str, Any]) -> Path:
    return _expand_path(payload.get("reports_dir")) or DEFAULT_REPORTS_DIR


def latest_packet_path(reports_dir: Path = DEFAULT_REPORTS_DIR) -> Path | None:
    root = reports_dir.expanduser()
    if not root.exists() or not root.is_dir():
        return None
    candidates = sorted(root.glob("implicit-supersession-proposals-*/implicit_supersession_proposals.json"))
    for path in reversed(candidates):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("schema_version") == PACKET_SCHEMA_VERSION and isinstance(payload.get("candidates"), list):
            return path
    return None


def latest_edge_report_path(reports_dir: Path = DEFAULT_REPORTS_DIR) -> Path | None:
    root = reports_dir.expanduser()
    if not root.exists() or not root.is_dir():
        return None
    candidates = sorted(root.glob("supersession-edge-semantics-*/supersession_edge_semantics_audit.json"))
    for path in reversed(candidates):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("mode") == EDGE_REPORT_MODE and isinstance(payload.get("rows"), list):
            return path
    return None


def _json_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def latest_edge_decision_artifact_path(reports_dir: Path = DEFAULT_REPORTS_DIR) -> Path | None:
    for item in reversed(_edge_decision_artifact_inventory(reports_dir)):
        if item["provenance"]["eligible_for_continuation"]:
            return item["path"]
    return None


def edge_decision_artifact_paths(reports_dir: Path = DEFAULT_REPORTS_DIR) -> list[Path]:
    root = reports_dir.expanduser() / "trust-review-edge-decisions"
    if not root.exists() or not root.is_dir():
        return []
    valid: list[tuple[tuple[str, str], Path]] = []
    for path in root.glob("trust_review_edge_decisions-*.json"):
        payload, _error_code = _read_edge_decision_json(path)
        if payload and payload.get("schema_version") == EDGE_DECISIONS_SCHEMA_VERSION:
            valid.append((_artifact_sort_key(path, payload), path))
    return [path for _sort_key, path in sorted(valid)]


def _artifact_sort_key(path: Path, payload: dict[str, Any]) -> tuple[str, str]:
    return (str(payload.get("generated_at") or ""), str(path))


def _bounded_file_bytes(
    path: Path,
    max_bytes: int,
    *,
    missing_code: str,
    unreadable_code: str,
    too_large_code: str,
) -> tuple[bytes | None, str | None]:
    expanded = path.expanduser()
    if not expanded.exists():
        return None, missing_code
    if not expanded.is_file():
        return None, unreadable_code
    try:
        with expanded.open("rb") as handle:
            raw = handle.read(max_bytes + 1)
    except FileNotFoundError:
        return None, missing_code
    except OSError:
        return None, unreadable_code
    if len(raw) > max_bytes:
        return None, too_large_code
    return raw, None


def _validate_edge_decision_artifact_payload(artifact: Any) -> dict[str, Any] | None:
    if not isinstance(artifact, dict):
        return _error("edge decision artifact must be a JSON object", code="INVALID_DECISION_ARTIFACT")
    if artifact.get("schema_version") != EDGE_DECISIONS_SCHEMA_VERSION:
        return _error(
            f"decision artifact must use schema {EDGE_DECISIONS_SCHEMA_VERSION}",
            code="INVALID_DECISION_ARTIFACT",
            field="schema_version",
        )
    if artifact.get("mutates_authority") is not False:
        return _error(
            "decision artifact must be read-only: mutates_authority must be false",
            code="INVALID_DECISION_ARTIFACT",
            field="mutates_authority",
        )
    rows = artifact.get("decisions") if isinstance(artifact.get("decisions"), list) else artifact.get("rows")
    if not isinstance(rows, list) or not rows:
        return _error("decision artifact must include at least one row", code="INVALID_DECISION_ARTIFACT")
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            return _error("decision rows must be objects", code="INVALID_DECISION_ARTIFACT")
        if row.get("mutates_authority") is not False:
            return _error("decision rows must be read-only", code="INVALID_DECISION_ARTIFACT")
        decision = str(row.get("review_decision") or "").strip()
        if decision not in EDGE_REVIEW_DECISIONS:
            return _error(
                f"Invalid edge review_decision {decision!r}",
                code="INVALID_DECISION_ARTIFACT",
                field="review_decision",
            )
        if not str(row.get("reviewer") or "").strip():
            return _error("decision rows require reviewer", code="INVALID_DECISION_ARTIFACT", field="reviewer")
        old_id = str(row.get("old_id") or "").strip()
        replacement_id = str(row.get("replacement_id") or row.get("new_id") or "").strip()
        if not old_id or not replacement_id:
            return _error("decision rows require old_id and replacement_id", code="INVALID_DECISION_ARTIFACT")
        key = (old_id, replacement_id)
        if key in seen:
            return _error(
                f"duplicate decision row: {old_id} -> {replacement_id}",
                code="INVALID_DECISION_ARTIFACT",
            )
        seen.add(key)
    return None


def _read_edge_decision_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    raw, error = _bounded_file_bytes(
        path,
        MAX_EDGE_DECISION_ARTIFACT_BYTES,
        missing_code="decision_artifact_missing",
        unreadable_code="decision_artifact_unreadable",
        too_large_code="decision_artifact_too_large",
    )
    if error:
        return None, error
    assert raw is not None
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, "invalid_decision_artifact"
    return (payload, None) if isinstance(payload, dict) else (None, "invalid_decision_artifact")


def _read_edge_decision_candidate(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    payload, error = _read_edge_decision_json(path)
    if error:
        return payload, error
    validation_error = _validate_edge_decision_artifact_payload(payload)
    return (payload, None) if validation_error is None else (payload, "invalid_decision_artifact")


def _edge_decision_candidate_paths(reports_dir: Path) -> list[Path]:
    root = reports_dir.expanduser() / "trust-review-edge-decisions"
    if not root.exists() or not root.is_dir():
        return []
    return sorted(root.glob("trust_review_edge_decisions-*.json"))


def _edge_decision_source_evidence(
    source_path: Path,
    cache: dict[Path, dict[str, Any]],
) -> dict[str, Any]:
    resolved = source_path.expanduser().resolve(strict=False)
    cached = cache.get(resolved)
    if cached is not None:
        return cached
    raw, error = _bounded_file_bytes(
        resolved,
        MAX_EDGE_SOURCE_REPORT_BYTES,
        missing_code="source_missing",
        unreadable_code="source_unreadable",
        too_large_code="source_too_large",
    )
    if error:
        result = {"exists": error != "source_missing", "error": error}
    else:
        assert raw is not None
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            result = {
                "exists": True,
                "source_shape_valid": False,
                "source_generated_at": None,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "error": "invalid_source_report",
            }
        else:
            source_shape_valid = (
                isinstance(payload, dict)
                and payload.get("mode") == EDGE_REPORT_MODE
                and isinstance(payload.get("rows"), list)
            )
            result = {
                "exists": True,
                "source_shape_valid": source_shape_valid,
                "source_generated_at": payload.get("generated_at") if isinstance(payload, dict) else None,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "error": None if source_shape_valid else "invalid_source_report",
            }
    cache[resolved] = result
    return result


def _edge_decision_artifact_provenance(
    path: Path,
    *,
    payload: dict[str, Any] | None = None,
    artifact_error: str | None = None,
    source_cache: dict[Path, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    artifact = payload
    if artifact is None and artifact_error is None:
        artifact, artifact_error = _read_edge_decision_candidate(path)
    reasons: list[str] = []
    checks = {
        "decision_payload_valid": False,
        "source_report_exists": False,
        "source_shape_valid": False,
        "source_timestamp_matches": False,
        "source_hash_matches": False,
    }
    if artifact_error:
        reasons.append(artifact_error)
    elif not artifact or _validate_edge_decision_artifact_payload(artifact) is not None:
        reasons.append("invalid_decision_artifact")
    else:
        checks["decision_payload_valid"] = True
        source_text = str(artifact.get("source_report_path") or "").strip()
        if not source_text:
            reasons.append("missing_source_path")
        else:
            evidence = _edge_decision_source_evidence(
                Path(source_text),
                source_cache if source_cache is not None else {},
            )
            checks["source_report_exists"] = evidence.get("exists") is True
            if evidence.get("error"):
                reasons.append(str(evidence["error"]))
            checks["source_shape_valid"] = evidence.get("source_shape_valid") is True
            if checks["source_shape_valid"]:
                checks["source_timestamp_matches"] = bool(
                    str(artifact.get("source_generated_at") or "").strip()
                ) and str(artifact.get("source_generated_at")) == str(evidence.get("source_generated_at") or "")
                if not checks["source_timestamp_matches"]:
                    reasons.append("source_timestamp_mismatch")
            expected_hash = str(artifact.get("source_report_sha256") or "").strip()
            if not expected_hash:
                reasons.append("missing_source_hash")
            elif evidence.get("sha256") is not None:
                checks["source_hash_matches"] = expected_hash == evidence.get("sha256")
                if not checks["source_hash_matches"]:
                    reasons.append("source_hash_mismatch")
    reasons = list(dict.fromkeys(reasons))
    eligible = not reasons
    return {
        "provenance_status": "verified" if eligible else "rejected",
        "eligible_for_continuation": eligible,
        "rejection_reasons": reasons,
        "checks": checks,
    }


def _edge_decision_artifact_inventory(reports_dir: Path) -> list[dict[str, Any]]:
    source_cache: dict[Path, dict[str, Any]] = {}
    inventory: list[dict[str, Any]] = []
    for path in _edge_decision_candidate_paths(reports_dir):
        payload, artifact_error = _read_edge_decision_candidate(path)
        provenance = _edge_decision_artifact_provenance(
            path,
            payload=payload,
            artifact_error=artifact_error,
            source_cache=source_cache,
        )
        inventory.append({"path": path, "payload": payload, "provenance": provenance})
    return sorted(
        inventory,
        key=lambda item: _artifact_sort_key(item["path"], item["payload"] or {}),
    )


def latest_edge_consume_artifact_path(reports_dir: Path = DEFAULT_REPORTS_DIR) -> Path | None:
    root = reports_dir.expanduser() / "trust-review-decision-consumer"
    if not root.exists() or not root.is_dir():
        return None
    candidates = [
        *root.glob("trust-review-decision-consume-*/trust_review_decision_consume_report.json"),
        *root.glob("trust_review_consume_pointer-*.json"),
    ]
    valid: list[tuple[tuple[str, str], Path]] = []
    for path in candidates:
        payload = _json_payload(path)
        if not payload:
            continue
        if payload.get("schema_version") not in {EDGE_CONSUME_SCHEMA_VERSION, EDGE_CONSUME_POINTER_SCHEMA_VERSION}:
            continue
        valid.append((_artifact_sort_key(path, payload), path))
    return max(valid, default=(None, None))[1] if valid else None


def _packet_paths_from_payload(payload: dict[str, Any]) -> list[Path]:
    raw_paths = payload.get("packet_paths")
    if raw_paths is None:
        raw_paths = payload.get("packets")
    if raw_paths is None:
        return []
    if isinstance(raw_paths, str):
        raw_paths = [part for part in raw_paths.split(",") if part.strip()]
    if not isinstance(raw_paths, list):
        return []
    return [Path(str(path)).expanduser() for path in raw_paths if str(path).strip()]


def _iter_calibration_packet_paths(payload: dict[str, Any]) -> list[Path]:
    paths = _packet_paths_from_payload(payload)
    if not paths:
        reports_dir = _reports_dir(payload).expanduser()
        paths = sorted(reports_dir.glob("implicit-supersession-proposals-*/implicit_supersession_proposals.json"))
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        try:
            resolved = path.expanduser().resolve()
        except OSError:
            resolved = path.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _resolve_packet_path(payload: dict[str, Any]) -> tuple[Path | None, dict[str, Any] | None]:
    packet_path = _expand_path(payload.get("packet_path") or payload.get("packet"))
    if packet_path is None:
        packet_path = latest_packet_path(_reports_dir(payload))
    if packet_path is None:
        return None, _error(
            "No implicit supersession proposal packet found. Run the implicit-supersession-proposals monitor first.",
            code="NO_IMPLICIT_SUPERSESSION_PACKET",
            field="packet_path",
        )
    return packet_path, None


def _resolve_edge_report_path(payload: dict[str, Any]) -> tuple[Path | None, dict[str, Any] | None]:
    report_path = _expand_path(payload.get("edge_report_path") or payload.get("edge_report"))
    if report_path is None:
        report_path = latest_edge_report_path(_reports_dir(payload))
    if report_path is None:
        return None, _error(
            "No supersession edge audit report found. Run Trust Health or the supersession edge monitor first.",
            code="NO_SUPERSESSION_EDGE_REPORT",
            field="edge_report_path",
        )
    return report_path, None


def load_packet(path: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    try:
        payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, _error(f"Packet not found: {path}", code="NO_IMPLICIT_SUPERSESSION_PACKET", field="packet_path")
    except (OSError, json.JSONDecodeError) as exc:
        return None, _error(f"Invalid packet JSON: {exc}", code="INVALID_IMPLICIT_SUPERSESSION_PACKET")
    if payload.get("schema_version") != PACKET_SCHEMA_VERSION:
        return None, _error(
            f"Packet has wrong schema: {payload.get('schema_version')}",
            code="INVALID_IMPLICIT_SUPERSESSION_PACKET",
        )
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return None, _error("Packet candidates must be a list", code="INVALID_IMPLICIT_SUPERSESSION_PACKET")
    if len(candidates) > MAX_CANDIDATES:
        return None, _error(
            f"Packet has too many candidates: {len(candidates)}>{MAX_CANDIDATES}",
            code="INVALID_IMPLICIT_SUPERSESSION_PACKET",
        )
    return payload, None


def load_edge_report(path: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    try:
        payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, _error(f"Edge report not found: {path}", code="NO_SUPERSESSION_EDGE_REPORT", field="edge_report_path")
    except (OSError, json.JSONDecodeError) as exc:
        return None, _error(f"Invalid edge report JSON: {exc}", code="INVALID_SUPERSESSION_EDGE_REPORT")
    if payload.get("mode") != EDGE_REPORT_MODE:
        return None, _error(
            f"Edge report has wrong mode: {payload.get('mode')}",
            code="INVALID_SUPERSESSION_EDGE_REPORT",
        )
    if not isinstance(payload.get("rows"), list):
        return None, _error("Edge report rows must be a list", code="INVALID_SUPERSESSION_EDGE_REPORT")
    return payload, None


def _packet_from_payload(payload: dict[str, Any]) -> tuple[Path | None, dict[str, Any] | None, dict[str, Any] | None]:
    packet_path, error = _resolve_packet_path(payload)
    if error:
        return None, None, error
    assert packet_path is not None
    packet, error = load_packet(packet_path)
    if error:
        return packet_path, None, error
    return packet_path, packet, None


def _edge_report_from_payload(
    payload: dict[str, Any],
) -> tuple[Path | None, dict[str, Any] | None, dict[str, Any] | None]:
    report_path, error = _resolve_edge_report_path(payload)
    if error:
        return None, None, error
    assert report_path is not None
    report, error = load_edge_report(report_path)
    if error:
        return report_path, None, error
    return report_path, report, None


def _review_queue(payload: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    queue = str(payload.get("queue") or "").strip().lower()
    if queue in PROPOSAL_QUEUE_ALIASES:
        return "proposal", None
    if queue in EDGE_RISK_QUEUE_ALIASES:
        return "edge-risk", None
    return None, _error(f"Unsupported review queue: {queue}", code="INVALID_REVIEW_PAYLOAD", field="queue")


def _candidate_key(candidate: dict[str, Any]) -> tuple[str, str]:
    return (str(candidate.get("old_id") or ""), str(candidate.get("new_id") or ""))


def _find_candidate(
    packet: dict[str, Any], old_id: Any, new_id: Any
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    old_text = str(old_id or "").strip()
    new_text = str(new_id or "").strip()
    if not old_text:
        return None, _error("old_id is required", code="INVALID_REVIEW_DECISION", field="old_id")
    if not new_text:
        return None, _error("new_id is required", code="INVALID_REVIEW_DECISION", field="new_id")
    for candidate in packet.get("candidates") or []:
        if isinstance(candidate, dict) and _candidate_key(candidate) == (old_text, new_text):
            return candidate, None
    return None, _error(
        f"Unknown review candidate: {old_text} -> {new_text}",
        code="UNKNOWN_REVIEW_CANDIDATE",
    )


def _candidate_brief(candidate: dict[str, Any]) -> dict[str, Any]:
    brief = _add_identity_labels(
        {
            "old_id": candidate.get("old_id"),
            "new_id": candidate.get("new_id"),
            "subject_key": candidate.get("subject_key"),
            "disposition": candidate.get("disposition"),
            "score": candidate.get("score"),
            "old_created_at": candidate.get("old_created_at"),
            "new_created_at": candidate.get("new_created_at"),
            "signals": candidate.get("signals") or [],
            "evidence_complete": bool(candidate.get("evidence_complete")),
            "old_summary": candidate.get("old_summary"),
            "new_summary": candidate.get("new_summary"),
            "old_source": candidate.get("old_source"),
            "new_source": candidate.get("new_source"),
        }
    )
    brief.pop("old_source", None)
    brief.pop("new_source", None)
    return brief


def _is_generic_sequence_subject(subject_key: Any, signals: list[Any]) -> bool:
    subject = str(subject_key or "").lower()
    if any(marker in subject for marker in GENERIC_SEQUENCE_SUBJECT_MARKERS):
        return True
    return "sequence_guard" in {str(signal) for signal in signals}


def _risk_class(candidate: dict[str, Any]) -> str:
    signals = candidate.get("signals") if isinstance(candidate.get("signals"), list) else []
    if _is_generic_sequence_subject(candidate.get("subject_key"), signals):
        return "generic_sequence_subject"
    if candidate.get("disposition") == "review_recommended":
        return "high_signal_replacement"
    if candidate.get("disposition") == "needs_context":
        return "insufficient_or_ambiguous"
    return "unknown"


SIGNAL_EXPLANATIONS = {
    "incomplete_or_ambiguous_evidence": "evidence may be incomplete or ambiguous",
    "new_resolved": "newer belief appears resolved",
    "newer_candidate": "newer candidate exists",
    "replacement_language": "newer text uses replacement language",
    "sequence_guard": "generic sequence wording needs extra review",
}

DISPOSITION_GUIDANCE = {
    "needs_context": "Needs context: inspect evidence before approving.",
    "review_recommended": "Review recommended: likely supersession, but still requires explicit approval.",
    "accepted": "Accepted by review evidence.",
    "rejected": "Rejected by review evidence.",
}

REVIEW_DECISION_EXPLANATIONS = {
    "approve_supersession": "Newer belief should govern; keep the older belief only as historical context.",
    "reject_not_supersession": "Do not link these as a supersession chain; both beliefs may remain separate.",
    "needs_more_context": "Do not decide yet; inspect more source evidence first.",
    "partial_retain": "Newer belief governs part of the subject, but older belief remains useful.",
}

EDGE_REVIEW_DECISION_EXPLANATIONS = {
    "approve_repair": "Approve this edge as a repair candidate; a later repair step must still apply it.",
    "reject_repair": "Reject repair for this edge; keep it out of automated maintenance.",
    "needs_context": "Do not decide yet; inspect more source evidence first.",
    "not_supersession": "This pair should not be treated as a supersession chain.",
    "uncertain": "The evidence is not clear enough to classify confidently.",
}


def _truncate_text(value: Any, *, limit: int = 180) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        return "unknown"
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _parse_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _relative_age(value: Any) -> str:
    parsed = _parse_datetime(value)
    if parsed is None:
        return "age unknown"
    delta = datetime.now(UTC) - parsed
    if delta.days < 0:
        return "future timestamp"
    if delta.days == 0:
        hours = delta.seconds // 3600
        return "today" if hours == 0 else f"{hours}h ago"
    if delta.days < 31:
        return f"{delta.days}d ago"
    return f"{delta.days // 30}mo ago"


def _format_time(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown (age unknown)"
    return f"{text} ({_relative_age(text)})"


def _format_time_utc(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown UTC (age unknown)"
    suffix = "" if text.endswith(("Z", "+00:00")) else " UTC"
    return f"{text}{suffix} ({_relative_age(text)})"


def _format_score(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "unknown"


def _format_signals(signals: Any) -> str:
    if not isinstance(signals, list) or not signals:
        return "none"
    return "; ".join(SIGNAL_EXPLANATIONS.get(str(signal), str(signal).replace("_", " ")) for signal in signals)


def _short_concept_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    if len(text) <= 12:
        return text
    return f"{text[:6]}...{text[-6:]}"


def _compact_source_label(source: Any) -> str:
    if not isinstance(source, dict) or not source:
        return "source unavailable"
    provenance = str(source.get("provenance") or "").strip()
    evidence = source.get("evidence") if isinstance(source.get("evidence"), list) else []
    first = evidence[0] if evidence and isinstance(evidence[0], dict) else {}
    bits = []
    if provenance:
        bits.append(f"provenance={provenance}")
    source_type = str(first.get("source_type") or "").strip()
    extraction = str(first.get("extraction_source") or "").strip()
    reference = str(first.get("source_reference") or "").strip()
    if source_type:
        bits.append(f"type={source_type}")
    if extraction:
        bits.append(f"extraction={extraction}")
    if reference:
        bits.append(f"ref={reference}")
    return _truncate_text(", ".join(bits), limit=140) if bits else "source unavailable"


def _belief_identity(candidate: dict[str, Any], prefix: str) -> dict[str, Any]:
    concept_id = str(candidate.get(f"{prefix}_id") or "").strip()
    learned_at = str(candidate.get(f"{prefix}_created_at") or "").strip()
    identity = {
        "concept_id": concept_id or "unknown",
        "short_id": _short_concept_id(concept_id),
        "title": _truncate_text(candidate.get(f"{prefix}_summary"), limit=96),
        "learned_at": learned_at or "",
        "learned_label": _format_time(learned_at) if learned_at else "learned unknown",
        "source_label": _compact_source_label(candidate.get(f"{prefix}_source")),
    }
    identity["display"] = (
        f"{identity['title']} [{identity['short_id']}; {identity['learned_label']}; {identity['source_label']}]"
    )
    return identity


def _add_identity_labels(candidate: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(candidate)
    old_identity = _belief_identity(enriched, "old")
    new_identity = _belief_identity(enriched, "new")
    subject = str(enriched.get("subject_key") or "unknown").strip() or "unknown"
    enriched["old_identity"] = old_identity
    enriched["new_identity"] = new_identity
    enriched["review_label"] = f"{subject}: {old_identity['title']} -> {new_identity['title']}"
    return enriched


def _source_lines(label: str, source: Any) -> list[str]:
    if not isinstance(source, dict) or not source:
        return ["  Source:    evidence unavailable"]
    lines = [f"  Source:    provenance={source.get('provenance') or 'unknown'}"]
    evidence = source.get("evidence") if isinstance(source.get("evidence"), list) else []
    if not evidence:
        lines.append("  Evidence:  unavailable")
        return lines
    first = evidence[0] if isinstance(evidence[0], dict) else {}
    source_bits = [
        f"type={first.get('source_type') or 'unknown'}",
        f"extraction={first.get('extraction_source') or 'unknown'}",
        f"method={first.get('evidence_method') or 'unknown'}",
    ]
    if first.get("timestamp"):
        source_bits.append(f"time={first.get('timestamp')}")
    if first.get("source_reference"):
        source_bits.append(f"ref={first.get('source_reference')}")
    lines.append(f"  Evidence:  {label}; " + ", ".join(source_bits))
    if first.get("content"):
        lines.append(f"             {_truncate_text(first.get('content'), limit=220)}")
    return lines


def _inspect_command(candidate: dict[str, Any]) -> str:
    return f"pith trust review show {candidate.get('old_id')} {candidate.get('new_id')}"


def _decision_command(candidate: dict[str, Any]) -> str:
    return (
        f"pith trust review decide {candidate.get('old_id')} {candidate.get('new_id')} "
        "--decision needs_more_context --retention-mode keep_both --reviewer <name>"
    )


def _decision_command_templates(candidate: dict[str, Any]) -> list[str]:
    prefix = f"pith trust review decide {candidate.get('old_id')} {candidate.get('new_id')}"
    return [
        (
            f"{prefix} --decision approve_supersession --retention-mode replace "
            '--reviewer <name> --rationale "<why the newer belief governs>"'
        ),
        f"{prefix} --decision reject_not_supersession --retention-mode keep_both --reviewer <name>",
        f"{prefix} --decision needs_more_context --retention-mode keep_both --reviewer <name>",
        (
            f"{prefix} --decision partial_retain --retention-mode partial_retain "
            '--reviewer <name> --rationale "<what changed and what remains useful>"'
        ),
    ]


def _decision_needed(candidate: dict[str, Any]) -> str:
    disposition = str(candidate.get("disposition") or "unknown")
    if disposition == "review_recommended":
        return "Decide whether the newer belief should govern, or reject the supersession link."
    if disposition == "needs_context":
        return "Inspect evidence and choose needs_more_context unless you can justify a stronger decision."
    return "Inspect evidence before choosing a review decision."


def _format_candidate_list_packet(candidate: dict[str, Any], index: int) -> list[str]:
    disposition = str(candidate.get("disposition") or "unknown")
    old_identity = (
        candidate.get("old_identity")
        if isinstance(candidate.get("old_identity"), dict)
        else _belief_identity(candidate, "old")
    )
    new_identity = (
        candidate.get("new_identity")
        if isinstance(candidate.get("new_identity"), dict)
        else _belief_identity(candidate, "new")
    )
    return [
        f"  Candidate {index} - {disposition.replace('_', ' ')} (score {_format_score(candidate.get('score'))})",
        f"    Review: {candidate.get('review_label') or 'unknown'}",
        f"    Subject: {str(candidate.get('subject_key') or 'unknown').strip() or 'unknown'}",
        f"    Old: {old_identity.get('display') or 'unknown'}",
        f"    New: {new_identity.get('display') or 'unknown'}",
        f"    Why flagged: {_format_signals(candidate.get('signals'))}",
        f"    Decision needed: {_decision_needed(candidate)}",
        f"    Inspect: {_inspect_command(candidate)}",
    ]


def _edge_candidate_key(candidate: dict[str, Any]) -> tuple[str, str]:
    return (str(candidate.get("old_id") or ""), str(candidate.get("replacement_id") or candidate.get("new_id") or ""))


def _edge_source_report_sha256(path: Path) -> str:
    return hashlib.sha256(path.expanduser().read_bytes()).hexdigest()


def _edge_source_row_hash(candidate: dict[str, Any]) -> str:
    payload = {
        key: candidate.get(key)
        for key in (
            "old_id",
            "replacement_id",
            "edge_kind",
            "repairability_class",
            "risk_reason",
            "supersession_reason",
            "superseded_at",
            "old_subject_key",
            "replacement_subject_key",
        )
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _edge_candidate_brief(row: dict[str, Any], *, report_path: Path, generated_at: str) -> dict[str, Any]:
    replacement_id = str(row.get("replacement_id") or "").strip()
    old_id = str(row.get("old_id") or "").strip()
    old_summary = row.get("old_summary_preview") or row.get("summary_preview")
    replacement_summary = row.get("replacement_summary_preview") or row.get("new_summary_preview")
    return {
        "old_id": old_id,
        "new_id": replacement_id,
        "replacement_id": replacement_id,
        "old_subject_key": row.get("old_subject_key"),
        "replacement_subject_key": row.get("replacement_subject_key"),
        "old_summary_preview": old_summary,
        "replacement_summary_preview": replacement_summary,
        "edge_kind": row.get("edge_kind"),
        "repairability_class": row.get("repairability_class"),
        "recommended_disposition": row.get("recommended_disposition"),
        "recommended_operation": row.get("recommended_operation"),
        "risk_reason": row.get("risk_reason"),
        "state_flags": row.get("state_flags") if isinstance(row.get("state_flags"), list) else [],
        "summary_preview": row.get("summary_preview"),
        "superseded_at": row.get("superseded_at"),
        "supersession_reason": row.get("supersession_reason"),
        "maintenance_boundary": row.get("maintenance_boundary"),
        "source_report_path": str(report_path),
        "source_generated_at": generated_at,
    }


def _manual_review_edge_candidates(report: dict[str, Any], report_path: Path) -> list[dict[str, Any]]:
    generated_at = str(report.get("generated_at") or "")
    candidates: list[dict[str, Any]] = []
    for row in report.get("rows") or []:
        if not isinstance(row, dict):
            continue
        if row.get("recommended_disposition") != "manual_review_required":
            continue
        candidate = _edge_candidate_brief(row, report_path=report_path, generated_at=generated_at)
        if not candidate["old_id"] or not candidate["replacement_id"]:
            continue
        candidates.append(candidate)
    return candidates


def _edge_focus_sort_key(candidate: dict[str, Any]) -> tuple[int, int, str, str, str]:
    edge_priority = {"unsafe_ambiguous": 0, "semantic_similarity_cross_subject": 1}.get(
        str(candidate.get("edge_kind") or ""),
        2,
    )
    flags = {str(flag) for flag in candidate.get("state_flags") or []}
    risk_priority = 0 if "answer_governance_risk" in flags else 1
    old_id, replacement_id = _edge_candidate_key(candidate)
    return (edge_priority, risk_priority, str(candidate.get("superseded_at") or ""), old_id, replacement_id)


def _focused_edge_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    return sorted(candidates, key=_edge_focus_sort_key)[0]


def _edge_decision_rows_from_artifact_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("decisions") if isinstance(payload.get("decisions"), list) else payload.get("rows")
    return [row for row in rows or [] if isinstance(row, dict)]


def _source_current_edge_decision_artifacts(
    *,
    reports_dir: Path,
    report_path: Path,
    report_generated_at: Any,
    source_report_sha256: str,
) -> list[dict[str, Any]]:
    current: list[dict[str, Any]] = []
    expected_report_path = report_path.expanduser().resolve(strict=False)
    for path in edge_decision_artifact_paths(reports_dir):
        payload = _json_payload(path)
        if not payload:
            continue
        artifact_report_path = Path(str(payload.get("source_report_path") or "")).expanduser().resolve(strict=False)
        if artifact_report_path != expected_report_path:
            continue
        if str(payload.get("source_generated_at") or "") != str(report_generated_at or ""):
            continue
        if str(payload.get("source_report_sha256") or "") != source_report_sha256:
            continue
        current.append({**payload, "_artifact_path": str(path)})
    return current


def _source_current_reviewed_edge_keys(
    *,
    reports_dir: Path,
    report_path: Path,
    report_generated_at: Any,
    source_report_sha256: str,
) -> set[tuple[str, str]]:
    reviewed: set[tuple[str, str]] = set()
    for artifact in _source_current_edge_decision_artifacts(
        reports_dir=reports_dir,
        report_path=report_path,
        report_generated_at=report_generated_at,
        source_report_sha256=source_report_sha256,
    ):
        for row in _edge_decision_rows_from_artifact_payload(artifact):
            old_id = str(row.get("old_id") or "").strip()
            replacement_id = str(row.get("replacement_id") or row.get("new_id") or "").strip()
            if old_id and replacement_id:
                reviewed.add((old_id, replacement_id))
    return reviewed


def _summarize_source_current_edge_decisions(
    *,
    reports_dir: Path,
    report_path: Path,
    report_generated_at: Any,
    source_report_sha256: str,
    fallback_decision: dict[str, Any],
) -> dict[str, Any]:
    artifacts = _source_current_edge_decision_artifacts(
        reports_dir=reports_dir,
        report_path=report_path,
        report_generated_at=report_generated_at,
        source_report_sha256=source_report_sha256,
    )
    if not artifacts:
        return dict(fallback_decision)

    decisions_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for artifact in artifacts:
        for row in _edge_decision_rows_from_artifact_payload(artifact):
            old_id = str(row.get("old_id") or "").strip()
            replacement_id = str(row.get("replacement_id") or row.get("new_id") or "").strip()
            if old_id and replacement_id:
                decisions_by_key[(old_id, replacement_id)] = row

    decision_counts: dict[str, int] = {}
    for row in decisions_by_key.values():
        decision = str(row.get("review_decision") or row.get("decision") or "unknown")
        decision_counts[decision] = decision_counts.get(decision, 0) + 1

    latest_artifact = artifacts[-1]
    latest_path = latest_artifact.get("_artifact_path")
    return {
        "available": True,
        "path": latest_path,
        "schema_version": EDGE_DECISIONS_SCHEMA_VERSION,
        "status": latest_artifact.get("status"),
        "generated_at": latest_artifact.get("generated_at"),
        "source_report_path": str(report_path),
        "source_generated_at": report_generated_at,
        "decision_count": len(decisions_by_key),
        "approved_repair_count": decision_counts.get("approve_repair", 0),
        "no_action_count": decision_counts.get("reject_repair", 0) + decision_counts.get("not_supersession", 0),
        "blocked_by_review_count": decision_counts.get("needs_context", 0) + decision_counts.get("uncertain", 0),
        "source_current_decision_artifact_count": len(artifacts),
        "source_current_decision_artifact_paths": [str(row.get("_artifact_path") or "") for row in artifacts],
        "mutates_authority": any(artifact.get("mutates_authority") is True for artifact in artifacts),
        "mutation_count": sum(int(artifact.get("mutation_count") or 0) for artifact in artifacts),
    }


def _find_edge_candidate(
    candidates: list[dict[str, Any]], old_id: Any, replacement_id: Any
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    old_text = str(old_id or "").strip()
    replacement_text = str(replacement_id or "").strip()
    if not old_text:
        return None, _error("old_id is required", code="INVALID_REVIEW_DECISION", field="old_id")
    if not replacement_text:
        return None, _error("new_id is required", code="INVALID_REVIEW_DECISION", field="new_id")
    for candidate in candidates:
        if _edge_candidate_key(candidate) == (old_text, replacement_text):
            return candidate, None
    return None, _error(
        f"Unknown edge-risk review row: {old_text} -> {replacement_text}",
        code="UNKNOWN_REVIEW_CANDIDATE",
    )


def _edge_inspect_command(candidate: dict[str, Any]) -> str:
    return f"pith trust review show --queue edge-risk {candidate.get('old_id')} {candidate.get('replacement_id')}"


def _edge_decision_command_templates(candidate: dict[str, Any]) -> list[str]:
    prefix = (
        f"pith trust review decide --queue edge-risk {candidate.get('old_id')} "
        f"{candidate.get('replacement_id')}"
    )
    return [
        f'{prefix} --decision approve_repair --reviewer <name> --rationale "<why this repair is valid>"',
        f"{prefix} --decision reject_repair --reviewer <name>",
        f"{prefix} --decision needs_context --reviewer <name>",
        f"{prefix} --decision not_supersession --reviewer <name>",
        f"{prefix} --decision uncertain --reviewer <name>",
    ]


def _format_flags(flags: Any) -> str:
    if not isinstance(flags, list) or not flags:
        return "none"
    return ", ".join(str(flag) for flag in flags)


def _format_edge_candidate_list_packet(candidate: dict[str, Any], index: int) -> list[str]:
    return [
        f"  Edge {index} - {str(candidate.get('edge_kind') or 'unknown').replace('_', ' ')}",
        f"    Possible supersession: {candidate.get('old_id')} -> {candidate.get('replacement_id')}",
        f"    Why this needs review: {_truncate_text(candidate.get('risk_reason'), limit=180)}",
        f"    Old subject: {_truncate_text(candidate.get('old_subject_key'), limit=320)}",
        f"    Replacement subject: {_truncate_text(candidate.get('replacement_subject_key'), limit=320)}",
        f"    Old summary: {_truncate_text(candidate.get('old_summary_preview') or candidate.get('summary_preview'), limit=260)}",
        f"    Replacement summary: {_truncate_text(candidate.get('replacement_summary_preview'), limit=260)}",
        f"    Supersession reason: {_truncate_text(candidate.get('supersession_reason'), limit=180)}",
        f"    Superseded at: {_format_time_utc(candidate.get('superseded_at'))}",
        f"    Inspect: {_edge_inspect_command(candidate)}",
    ]


def _format_edge_candidate_detail(candidate: dict[str, Any]) -> str:
    lines = [
        "Edge-risk row:",
        f"  Possible supersession: {candidate.get('old_id')} -> {candidate.get('replacement_id')}",
        f"  Edge kind: {candidate.get('edge_kind') or 'unknown'}",
        f"  Repair class: {candidate.get('repairability_class') or 'unknown'}",
        f"  Disposition: {candidate.get('recommended_disposition') or 'unknown'}",
        f"  Recommended operation: {candidate.get('recommended_operation') or 'none'}",
        f"  Why this needs review: {_truncate_text(candidate.get('risk_reason'), limit=260)}",
        f"  Old subject: {_truncate_text(candidate.get('old_subject_key'), limit=500)}",
        f"  Replacement subject: {_truncate_text(candidate.get('replacement_subject_key'), limit=500)}",
        f"  Old summary: {_truncate_text(candidate.get('old_summary_preview') or candidate.get('summary_preview'), limit=700)}",
        f"  Replacement summary: {_truncate_text(candidate.get('replacement_summary_preview'), limit=700)}",
        f"  Supersession reason: {_truncate_text(candidate.get('supersession_reason'), limit=260)}",
        f"  Superseded at: {_format_time_utc(candidate.get('superseded_at'))}",
        f"  State flags: {_format_flags(candidate.get('state_flags'))}",
        f"  Source report: {candidate.get('source_report_path') or 'unknown'}",
        f"  Report generated: {candidate.get('source_generated_at') or 'unknown'}",
        "  Decision options:",
        *[f"    - {decision}: {explanation}" for decision, explanation in EDGE_REVIEW_DECISION_EXPLANATIONS.items()],
    ]
    if candidate.get("maintenance_boundary"):
        lines.append(f"  Boundary: {_truncate_text(candidate.get('maintenance_boundary'), limit=260)}")
    return "\n".join(lines)


def _candidate_calibration_row(
    candidate: dict[str, Any],
    *,
    first_seen_at: str,
    latest_seen_at: str,
    source_packet_paths: list[str],
) -> dict[str, Any]:
    row = _candidate_brief(candidate)
    row.update(
        {
            "risk_class": _risk_class(candidate),
            "first_seen_at": first_seen_at,
            "latest_seen_at": latest_seen_at,
            "source_packet_count": len(source_packet_paths),
            "source_packet_paths": list(source_packet_paths),
            "review_schema_version": str(candidate.get("review_schema_version") or ""),
            "review_group_key": str(candidate.get("review_group_key") or ""),
            "review_decision": "",
            "retention_mode": "",
            "reviewer": "",
            "reviewed_at": "",
            "review_rationale": "",
            "review_source": "user_reviewed",
            "expected_disposition": str(candidate.get("expected_disposition") or candidate.get("disposition") or ""),
        }
    )
    return row


def _load_calibration_candidates(paths: list[Path]) -> tuple[dict[tuple[str, str], dict[str, Any]], list[str]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    errors: list[str] = []
    for path in paths:
        packet, error = load_packet(path)
        if error:
            errors.append(f"{path}:{error.get('code')}")
            continue
        assert packet is not None
        generated_at = str(packet.get("generated_at") or "")
        for candidate in packet.get("candidates") or []:
            if not isinstance(candidate, dict):
                errors.append(f"{path}:candidate_not_object")
                continue
            key = _candidate_key(candidate)
            if not key[0] or not key[1]:
                errors.append(f"{path}:candidate_missing_old_or_new_id")
                continue
            packet_text = str(path)
            if key not in records:
                records[key] = _candidate_calibration_row(
                    candidate,
                    first_seen_at=generated_at,
                    latest_seen_at=generated_at,
                    source_packet_paths=[packet_text],
                )
                continue
            record = records[key]
            if generated_at and (not record.get("first_seen_at") or generated_at < str(record.get("first_seen_at"))):
                record["first_seen_at"] = generated_at
            if generated_at and generated_at > str(record.get("latest_seen_at") or ""):
                record["latest_seen_at"] = generated_at
            source_paths = list(record.get("source_packet_paths") or [])
            if packet_text not in source_paths:
                source_paths.append(packet_text)
            record["source_packet_paths"] = source_paths
            record["source_packet_count"] = len(source_paths)
    return records, errors


def _write_text_no_overwrite(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    path.write_text(text, encoding="utf-8")


def _render_calibration_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    lines = [
        "# Implicit Supersession Review Calibration Queue",
        "",
        f"- Generated: {report.get('generated_at')}",
        f"- Status: {report.get('status')}",
        f"- Boundary: {report.get('claim_boundary')}",
        f"- Mutation count: {report.get('mutation_count', 0)}",
        f"- Packet files scanned: {summary.get('packet_count', 0)}",
        f"- Unique candidates: {summary.get('unique_candidate_count', 0)}",
        f"- Readiness floor: {summary.get('minimum_reviewable_cases', 0)}",
        "",
        "## Risk Classes",
        "",
    ]
    for key, value in (summary.get("risk_class_counts") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Queue", ""])
    for index, candidate in enumerate(report.get("candidates") or [], start=1):
        lines.extend(
            [
                f"### {index}. {candidate.get('old_id')} -> {candidate.get('new_id')}",
                "",
                f"- Risk class: `{candidate.get('risk_class')}`",
                f"- Disposition: `{candidate.get('disposition')}` score={candidate.get('score')}",
                f"- Subject: `{candidate.get('subject_key') or 'unknown'}`",
                f"- Old created: {candidate.get('old_created_at') or 'unknown'}",
                f"- New created: {candidate.get('new_created_at') or 'unknown'}",
                f"- First seen: {candidate.get('first_seen_at') or 'unknown'}",
                f"- Latest seen: {candidate.get('latest_seen_at') or 'unknown'}",
                f"- Source packets: {candidate.get('source_packet_count', 0)}",
                f"- Evidence complete: {candidate.get('evidence_complete')}",
                f"- Signals: {', '.join(str(v) for v in candidate.get('signals') or []) or 'none'}",
                f"- Review decision: `{candidate.get('review_decision') or '<approve_supersession | reject_not_supersession | needs_more_context | partial_retain>'}`",
                f"- Retention mode: `{candidate.get('retention_mode') or '<replace | keep_both | partial_retain>'}`",
                "",
                f"Old: {candidate.get('old_summary')}",
                "",
                f"New: {candidate.get('new_summary')}",
                "",
                "Rationale: <required for approve_supersession or partial_retain>",
                "",
            ]
        )
    if report.get("errors"):
        lines.extend(["## Packet Errors", ""])
        for error in report["errors"]:
            lines.append(f"- {error}")
        lines.append("")
    return "\n".join(lines)


def _render_calibration_tsv(report: dict[str, Any]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=CALIBRATION_TSV_FIELDS, delimiter="\t", extrasaction="ignore")
    writer.writeheader()
    for candidate in report.get("candidates") or []:
        row = dict(candidate)
        row["signals"] = ",".join(str(v) for v in candidate.get("signals") or [])
        row["source_packet_paths"] = "|".join(str(v) for v in candidate.get("source_packet_paths") or [])
        writer.writerow(row)
    return output.getvalue()


def _write_calibration_artifacts(report: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json_path": output_dir / "implicit_supersession_review_calibration.json",
        "markdown_path": output_dir / "implicit_supersession_review_calibration.md",
        "tsv_path": output_dir / "implicit_supersession_review_calibration.tsv",
    }
    for path in paths.values():
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    artifact_paths = {key: str(value) for key, value in paths.items()}
    report_for_write = dict(report)
    report_for_write["artifact_paths"] = artifact_paths
    _write_text_no_overwrite(paths["json_path"], json.dumps(report_for_write, indent=2, sort_keys=True) + "\n")
    _write_text_no_overwrite(paths["markdown_path"], _render_calibration_markdown(report_for_write))
    _write_text_no_overwrite(paths["tsv_path"], _render_calibration_tsv(report_for_write))
    return artifact_paths


def _base_result(operation: str, packet_path: Path | None = None) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "operation": operation,
        "claim_boundary": CLAIM_BOUNDARY,
        "mutates_authority": False,
        "mutation_count": 0,
        "packet_path": str(packet_path) if packet_path else None,
    }


def _summarize_edge_decision_artifact(
    path: Path | None,
    *,
    provenance: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if path is None:
        return {"available": False, "path": None, "provenance": None}
    artifact_error: str | None = None
    if payload is None:
        payload, artifact_error = _read_edge_decision_candidate(path)
    provenance = provenance or _edge_decision_artifact_provenance(
        path,
        payload=payload,
        artifact_error=artifact_error,
    )
    if not payload or payload.get("schema_version") != EDGE_DECISIONS_SCHEMA_VERSION:
        return {
            "available": False,
            "path": str(path),
            "error": "invalid_decision_artifact",
            "provenance": provenance,
        }
    rows = _edge_decision_rows_from_artifact_payload(payload)
    decision_counts: dict[str, int] = {}
    for row in rows:
        decision = str(row.get("review_decision") or "")
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
    available = provenance.get("eligible_for_continuation") is True
    return {
        "available": available,
        "path": str(path),
        "error": None if available else "source_unverifiable",
        "provenance": provenance,
        "schema_version": payload.get("schema_version"),
        "status": payload.get("status"),
        "generated_at": payload.get("generated_at"),
        "source_report_path": payload.get("source_report_path"),
        "source_generated_at": payload.get("source_generated_at"),
        "decision_count": len(rows),
        "approved_repair_count": decision_counts.get("approve_repair", 0),
        "no_action_count": decision_counts.get("reject_repair", 0) + decision_counts.get("not_supersession", 0),
        "blocked_by_review_count": decision_counts.get("needs_context", 0) + decision_counts.get("uncertain", 0),
        "mutates_authority": payload.get("mutates_authority") is True,
        "mutation_count": 0,
    }


def _summarize_latest_edge_decision_artifact(reports_dir: Path) -> dict[str, Any]:
    inventory = _edge_decision_artifact_inventory(reports_dir)
    eligible = [row for row in inventory if row["provenance"]["eligible_for_continuation"]]
    rejected = [row for row in inventory if not row["provenance"]["eligible_for_continuation"]]
    selected = eligible[-1] if eligible else None
    summary = _summarize_edge_decision_artifact(
        selected["path"] if selected else None,
        provenance=selected["provenance"] if selected else None,
        payload=selected["payload"] if selected else None,
    )
    summary["candidate_artifact_count"] = len(inventory)
    summary["rejected_artifact_count"] = len(rejected)
    summary["latest_rejected_artifact"] = (
        {
            "path": str(rejected[-1]["path"]),
            "generated_at": (rejected[-1]["payload"] or {}).get("generated_at"),
            "rejection_reasons": rejected[-1]["provenance"]["rejection_reasons"],
        }
        if rejected
        else None
    )
    return summary


def _consume_pointer_summary(result: dict[str, Any], *, source_report_available: bool = True) -> dict[str, Any]:
    dry_run = result.get("dry_run_summary") if isinstance(result.get("dry_run_summary"), dict) else {}
    paths = result.get("paths") if isinstance(result.get("paths"), dict) else {}
    return {
        "schema_version": EDGE_CONSUME_POINTER_SCHEMA_VERSION,
        "generated_at": result.get("generated_at") or _utc_now_iso(),
        "queue": result.get("queue") or "edge-risk",
        "status": result.get("status"),
        "consume_report_path": paths.get("consume_report"),
        "source_report_available": source_report_available,
        "decision_file": result.get("decision_file"),
        "decision_count": int(result.get("decision_count") or 0),
        "approved_repair_count": int(result.get("approved_repair_count") or 0),
        "no_action_count": int(result.get("no_action_count") or 0),
        "blocked_by_review_count": int(result.get("blocked_by_review_count") or 0),
        "eligible_line_count": int(dry_run.get("eligible_line_count") or 0),
        "changed_rows": int(result.get("changed_rows") or 0),
        "mutates_authority": False,
        "mutation_count": 0,
        "claim_boundary": EDGE_CONSUME_CLAIM_BOUNDARY,
    }


def _summarize_edge_consume_artifact(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"available": False, "path": None}
    payload = _json_payload(path)
    if not payload:
        return {"available": False, "path": str(path), "error": "invalid_consume_artifact"}
    schema = payload.get("schema_version")
    if schema == EDGE_CONSUME_POINTER_SCHEMA_VERSION:
        report_path = _expand_path(payload.get("consume_report_path"))
        source_available = bool(report_path and report_path.exists())
        return {
            "available": True,
            "path": str(path),
            "schema_version": schema,
            "generated_at": payload.get("generated_at"),
            "status": payload.get("status"),
            "consume_report_path": payload.get("consume_report_path"),
            "source_report_available": source_available,
            "decision_file": payload.get("decision_file"),
            "decision_count": int(payload.get("decision_count") or 0),
            "approved_repair_count": int(payload.get("approved_repair_count") or 0),
            "no_action_count": int(payload.get("no_action_count") or 0),
            "blocked_by_review_count": int(payload.get("blocked_by_review_count") or 0),
            "eligible_line_count": int(payload.get("eligible_line_count") or 0),
            "changed_rows": int(payload.get("changed_rows") or 0),
            "mutates_authority": payload.get("mutates_authority") is True,
            "mutation_count": int(payload.get("mutation_count") or 0),
        }
    if schema != EDGE_CONSUME_SCHEMA_VERSION:
        return {"available": False, "path": str(path), "error": "invalid_consume_schema"}
    dry_run = payload.get("dry_run_summary") if isinstance(payload.get("dry_run_summary"), dict) else {}
    return {
        "available": True,
        "path": str(path),
        "schema_version": schema,
        "generated_at": payload.get("generated_at"),
        "status": payload.get("status"),
        "consume_report_path": str(path),
        "source_report_available": True,
        "decision_file": payload.get("decision_file"),
        "decision_count": int(payload.get("decision_count") or 0),
        "approved_repair_count": int(payload.get("approved_repair_count") or 0),
        "no_action_count": int(payload.get("no_action_count") or 0),
        "blocked_by_review_count": int(payload.get("blocked_by_review_count") or 0),
        "eligible_line_count": int(dry_run.get("eligible_line_count") or 0),
        "changed_rows": int(payload.get("changed_rows") or 0),
        "mutates_authority": payload.get("mutates_authority") is True,
        "mutation_count": int(payload.get("mutation_count") or 0),
    }


def _review_artifact_summaries(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    reports_dir = _reports_dir(payload)
    decision_path = _expand_path(payload.get("decision_file") or payload.get("decision_file_path"))
    if decision_path is not None:
        decision = _summarize_edge_decision_artifact(decision_path)
        rejected = decision.get("available") is not True
        decision["candidate_artifact_count"] = 1
        decision["rejected_artifact_count"] = 1 if rejected else 0
        decision["latest_rejected_artifact"] = (
            {
                "path": str(decision_path),
                "generated_at": decision.get("generated_at"),
                "rejection_reasons": (decision.get("provenance") or {}).get("rejection_reasons") or [],
            }
            if rejected
            else None
        )
    else:
        decision = _summarize_latest_edge_decision_artifact(reports_dir)
    consume_path = _expand_path(payload.get("consume_report") or payload.get("consume_report_path"))
    if consume_path is None:
        consume_path = latest_edge_consume_artifact_path(reports_dir)
    return decision, _summarize_edge_consume_artifact(consume_path)


def _parse_artifact_time(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _decision_newer_than_consume(decision: dict[str, Any], consume: dict[str, Any]) -> bool:
    decision_time = _parse_artifact_time(decision.get("generated_at"))
    consume_time = _parse_artifact_time(consume.get("generated_at"))
    return bool(decision_time and (consume_time is None or decision_time > consume_time))


def _artifact_continuation(decision: dict[str, Any], consume: dict[str, Any]) -> dict[str, Any]:
    decision_path = str(decision.get("path") or "")
    if decision.get("available") and (not consume.get("available") or _decision_newer_than_consume(decision, consume)):
        return {
            "kind": "consume_latest_decision",
            "action": "Consume the latest edge decision artifact into measured dry-run evidence.",
            "command": f"pith trust review consume --queue edge-risk --decision-file {decision_path}",
        }
    if consume.get("available") and int(consume.get("eligible_line_count") or 0) <= 0:
        return {
            "kind": "review_more_edges",
            "action": "Latest consume report has no approved repair lines; review more edge-risk rows before repair.",
            "command": "pith trust review focus --queue edge-risk",
        }
    if consume.get("available"):
        return {
            "kind": "run_hash_pinned_dry_run",
            "action": "Latest consume report has eligible approved repairs; run the hash-pinned dry-run before any live repair.",
            "command": "Use the dry-run command printed in the latest consume report.",
        }
    return {
        "kind": "inspect_edge_risk",
        "action": "Inspect edge-risk rows and export a reviewed decision artifact.",
        "command": "pith trust review focus --queue edge-risk",
    }


def _edge_review_progress(
    *,
    candidate_count: int,
    edge_report_path: str,
    report_generated_at: Any,
    edge_decision: dict[str, Any],
    consume_report: dict[str, Any],
) -> dict[str, Any]:
    decision_source_current = (
        edge_decision.get("available") is True
        and str(edge_decision.get("source_report_path") or "") == str(edge_report_path or "")
        and str(edge_decision.get("source_generated_at") or "") == str(report_generated_at or "")
    )
    reviewed_count = int(edge_decision.get("decision_count") or 0) if decision_source_current else 0
    pending_count = max(candidate_count - reviewed_count, 0)
    coverage_ratio = 1.0 if candidate_count <= 0 else round(reviewed_count / candidate_count, 4)
    next_safe_command = f"pith trust review focus --queue edge-risk --edge-report {edge_report_path}"
    if decision_source_current and _decision_newer_than_consume(edge_decision, consume_report):
        next_safe_command = (
            "pith trust review consume --queue edge-risk "
            f"--decision-file {edge_decision.get('path')}"
        )
    elif decision_source_current and int(consume_report.get("eligible_line_count") or 0) > 0:
        next_safe_command = "Use the dry-run command printed in the latest consume report."
    return {
        "candidate_count": candidate_count,
        "reviewed_current_report_count": reviewed_count,
        "pending_review_count": pending_count,
        "coverage_ratio": coverage_ratio,
        "decision_source_current": decision_source_current,
        "decision_source_report_path": edge_decision.get("source_report_path"),
        "decision_source_generated_at": edge_decision.get("source_generated_at"),
        "approved_repair_count": int(edge_decision.get("approved_repair_count") or 0) if decision_source_current else 0,
        "no_action_count": int(edge_decision.get("no_action_count") or 0) if decision_source_current else 0,
        "blocked_by_review_count": int(edge_decision.get("blocked_by_review_count") or 0)
        if decision_source_current
        else 0,
        "source_current_decision_artifact_count": (
            int(edge_decision.get("source_current_decision_artifact_count") or 0) if decision_source_current else 0
        ),
        "source_current_decision_artifact_paths": (
            edge_decision.get("source_current_decision_artifact_paths") or [] if decision_source_current else []
        ),
        "eligible_repair_count": int(consume_report.get("eligible_line_count") or 0) if decision_source_current else 0,
        "consume_status": consume_report.get("status") if consume_report.get("available") else None,
        "next_safe_command": next_safe_command,
        "claim_boundary": (
            "Progress counts only decisions whose source report path and generated timestamp "
            "match the current edge-risk report."
        ),
    }


def build_review_status_from_payload(payload: Any) -> dict[str, Any]:
    payload = payload if isinstance(payload, dict) else {}
    queue, error = _review_queue(payload)
    if error:
        return error
    edge_decision, consume_report = _review_artifact_summaries(payload)
    if queue == "edge-risk":
        report_path, report, error = _edge_report_from_payload(payload)
        if error:
            return error
        assert report is not None
        assert report_path is not None
        candidates = _manual_review_edge_candidates(report, report_path)
        edge_report_path = str(report_path)
        report_generated_at = report.get("generated_at")
        source_report_sha256 = _edge_source_report_sha256(report_path)
        progress_decision = _summarize_source_current_edge_decisions(
            reports_dir=_reports_dir(payload),
            report_path=report_path,
            report_generated_at=report_generated_at,
            source_report_sha256=source_report_sha256,
            fallback_decision=edge_decision,
        )
        result = _base_result("review_status", None)
        result.update(
            {
                "schema_version": EDGE_QUEUE_SCHEMA_VERSION,
                "queue": "edge-risk",
                "status": "ready",
                "candidate_count": len(candidates),
                "edge_report_path": edge_report_path,
                "report_generated_at": report_generated_at,
                "edge_decision_artifact": edge_decision,
                "consume_report": consume_report,
                "review_progress": _edge_review_progress(
                    candidate_count=len(candidates),
                    edge_report_path=edge_report_path,
                    report_generated_at=report_generated_at,
                    edge_decision=progress_decision,
                    consume_report=consume_report,
                ),
            }
        )
        return result
    packet_path, packet, error = _packet_from_payload(payload)
    if error:
        return error
    assert packet is not None
    summary = packet.get("summary") if isinstance(packet.get("summary"), dict) else {}
    result = _base_result("review_status", packet_path)
    result.update(
        {
            "status": "ready",
            "generated_at": packet.get("generated_at"),
            "candidate_count": len(packet.get("candidates") or []),
            "review_recommended_count": summary.get("review_recommended_count", 0),
            "needs_context_count": summary.get("needs_context_count", 0),
            "edge_decision_artifact": edge_decision,
            "consume_report": consume_report,
        }
    )
    return result


def _trust_health_args(payload: dict[str, Any]) -> argparse.Namespace:
    from scripts import trust_health_status

    return argparse.Namespace(
        history_limit=int(payload.get("history_limit") or trust_health_status.DEFAULT_HISTORY_LIMIT),
        compact_log_path=str(payload.get("compact_log_path") or trust_health_status.DEFAULT_COMPACT_LOG_PATH),
        supersession_compact_log_path=str(
            payload.get("supersession_compact_log_path") or trust_health_status.DEFAULT_SUPERSESSION_COMPACT_LOG_PATH
        ),
        reports_dir=str(payload.get("reports_dir") or trust_health_status.DEFAULT_REPORTS_DIR),
        health_url=str(payload.get("health_url") or trust_health_status.DEFAULT_HEALTH_URL),
        warn_after_seconds=int(payload.get("warn_after_seconds") or trust_health_status.DEFAULT_WARN_AFTER_SECONDS),
        critical_after_seconds=int(
            payload.get("critical_after_seconds") or trust_health_status.DEFAULT_CRITICAL_AFTER_SECONDS
        ),
        max_log_bytes=int(payload.get("max_log_bytes") or trust_health_status.DEFAULT_MAX_LOG_BYTES),
        no_runtime_health=not bool(payload.get("runtime_health", True)),
        no_scheduler=not bool(payload.get("scheduler", True)),
        no_supersession_scheduler=not bool(payload.get("supersession_scheduler", True)),
    )


def _supersession_risk_counts(trust_health: dict[str, Any]) -> dict[str, int]:
    supersession = (
        trust_health.get("supersession_edges") if isinstance(trust_health.get("supersession_edges"), dict) else {}
    )
    latest = supersession.get("latest_run") if isinstance(supersession.get("latest_run"), dict) else {}
    risks = latest.get("risk_counts") if isinstance(latest.get("risk_counts"), dict) else {}
    keys = (
        "manual_review_required",
        "missing_identity_edges",
        "reflection_duplicate_cross_subject",
        "answer_governance_risk",
        "total_edges",
    )
    result: dict[str, int] = {}
    for key in keys:
        try:
            result[key] = int(risks.get(key) or 0)
        except (TypeError, ValueError):
            result[key] = 0
    return result


def build_review_next_from_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        payload = {}
    from scripts import trust_health_status

    selected_queue, queue_error = _review_queue(payload)
    trust_health = trust_health_status.build_status(_trust_health_args(payload))
    packet_status = build_review_status_from_payload(payload)
    packet_available = not packet_status.get("error")
    edge_decision, consume_report = _review_artifact_summaries(payload)
    artifact_continuation = _artifact_continuation(edge_decision, consume_report)
    broad_counts = _supersession_risk_counts(trust_health)
    health_steps = [
        str(step)
        for step in (
            trust_health.get("user_next_steps") if isinstance(trust_health.get("user_next_steps"), list) else []
        )
    ]

    safe_commands = ["pith trust-health", "pith trust review next"]
    if broad_counts.get("manual_review_required", 0) > 0:
        safe_commands.extend(["pith trust review list --queue edge-risk", "pith trust review focus --queue edge-risk"])
    if packet_available:
        safe_commands.extend(["pith trust review list", "pith trust review show <old_id> <new_id>"])
    else:
        safe_commands.append("pith trust review status")
    decision_commands = ["pith trust review decide", "pith trust review run"]

    recommended_actions: list[dict[str, Any]] = []
    if trust_health.get("status") in {"NO_EVIDENCE", "ALARM"} or not health_steps:
        recommended_actions.append(
            {
                "rank": 1,
                "kind": "refresh_or_inspect_trust_health",
                "action": trust_health.get("primary_user_action") or "Run Trust Health before review.",
                "command": "pith trust-health",
            }
        )
    else:
        recommended_actions.append(
            {
                "rank": 1,
                "kind": "broad_supersession_edge_review",
                "action": health_steps[0],
                "command": "pith trust-health",
            }
        )
    if selected_queue == "edge-risk" and packet_available:
        recommended_actions.append(
            {
                "rank": 2,
                "kind": "selected_edge_risk_report",
                "action": "Inspect the selected edge-risk report with source-current progress.",
                "command": "pith trust review list --queue edge-risk",
            }
        )
    elif packet_available and int(packet_status.get("candidate_count") or 0) > 0:
        recommended_actions.append(
            {
                "rank": 2,
                "kind": "latest_implicit_proposal_packet",
                "action": "Inspect the latest implicit supersession proposal packet.",
                "command": "pith trust review list",
            }
        )
    else:
        recommended_actions.append(
            {
                "rank": 2,
                "kind": "latest_implicit_proposal_packet",
                "action": packet_status.get("message") or "No latest implicit proposal packet is available.",
                "command": "pith trust review status",
            }
        )

    if artifact_continuation:
        recommended_actions.append(
            {
                "rank": len(recommended_actions) + 1,
                **artifact_continuation,
            }
        )
        command = artifact_continuation.get("command")
        if command and command not in safe_commands and not str(command).startswith("Use the dry-run command"):
            safe_commands.append(str(command))

    edge_queue_selected = selected_queue == "edge-risk"
    latest_source = "selected_edge_risk_report" if edge_queue_selected else "implicit_supersession_proposals"
    review_scope = "Trust Health and selected edge-risk reports" if edge_queue_selected else "Trust Health and latest proposal packets"
    return {
        "schema_version": "trust_review_next.v1",
        "operation": "review_next",
        "status": "ready" if not trust_health.get("error") else "needs_trust_health",
        "claim_boundary": (
            f"Read-only trust maintenance guidance. This summarizes {review_scope}; it does not approve "
            "or apply supersession repairs."
        ),
        "mutates_authority": False,
        "mutation_count": 0,
        "selected_review_queue": selected_queue if not queue_error else None,
        "primary_user_action": trust_health.get("primary_user_action"),
        "trust_health_status": trust_health.get("overall_user_status") or trust_health.get("status"),
        "trust_health_alarm_summary": (
            trust_health.get("alarm_summary") if isinstance(trust_health.get("alarm_summary"), dict) else {}
        ),
        "trust_health_runtime": trust_health.get("runtime") if isinstance(trust_health.get("runtime"), dict) else {},
        "broad_supersession_edge_queue": {
            "source": "trust_health_supersession_edge_semantics",
            "risk_counts": broad_counts,
            "evidence_path": (
                ((trust_health.get("supersession_edges") or {}).get("evidence") or {}).get("latest_report_path")
                if isinstance(trust_health.get("supersession_edges"), dict)
                else None
            ),
            "next_steps": health_steps,
        },
        "latest_proposal_packet_queue": {
            "source": latest_source,
            "available": packet_available,
            "status": packet_status.get("status") if packet_available else "unavailable",
            "packet_path": packet_status.get("packet_path"),
            "edge_report_path": packet_status.get("edge_report_path"),
            "candidate_count": int(packet_status.get("candidate_count") or 0) if packet_available else 0,
            "review_recommended_count": int(packet_status.get("review_recommended_count") or 0)
            if packet_available
            else 0,
            "needs_context_count": int(packet_status.get("needs_context_count") or 0) if packet_available else 0,
            "error": None if packet_available else packet_status,
        },
        "latest_edge_decision_artifact": edge_decision,
        "latest_consume_report": consume_report,
        "review_progress": packet_status.get("review_progress") if packet_available else None,
        "artifact_continuation": artifact_continuation,
        "recommended_actions": recommended_actions,
        "safe_inspection_commands": safe_commands,
        "decision_commands": decision_commands,
        "queue_distinction": (
            (
                "Broad supersession edge-risk counts come from Trust Health edge-semantics evidence. "
                "Selected edge-risk report counts come from the explicit --edge-report argument and may differ "
                "from the latest scheduled Trust Health report."
            )
            if selected_queue == "edge-risk"
            else (
                "Broad supersession edge-risk counts come from Trust Health edge-semantics evidence. "
                "Latest proposal packet counts are a smaller read-only candidate packet and are not the same queue."
            )
        ),
    }


def build_review_list_from_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        payload = {}
    queue, error = _review_queue(payload)
    if error:
        return error
    if queue == "edge-risk":
        return build_edge_review_list_from_payload(payload)
    packet_path, packet, error = _packet_from_payload(payload)
    if error:
        return error
    assert packet is not None
    candidates = [_candidate_brief(row) for row in packet.get("candidates") or [] if isinstance(row, dict)]
    result = _base_result("review_list", packet_path)
    result.update({"status": "ready", "candidate_count": len(candidates), "candidates": candidates})
    return result


def build_edge_review_list_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    report_path, report, error = _edge_report_from_payload(payload)
    if error:
        return error
    assert report is not None
    assert report_path is not None
    limit, error = _payload_int(payload, "limit", DEFAULT_EDGE_REVIEW_LIMIT, minimum=0)
    if error:
        return error
    assert limit is not None
    candidates = sorted(_manual_review_edge_candidates(report, report_path), key=_edge_focus_sort_key)
    shown = candidates[:limit]
    edge_decision, consume_report = _review_artifact_summaries(payload)
    edge_report_path = str(report_path)
    report_generated_at = report.get("generated_at")
    source_report_sha256 = _edge_source_report_sha256(report_path)
    progress_decision = _summarize_source_current_edge_decisions(
        reports_dir=_reports_dir(payload),
        report_path=report_path,
        report_generated_at=report_generated_at,
        source_report_sha256=source_report_sha256,
        fallback_decision=edge_decision,
    )
    result = _base_result("review_list", None)
    result.update(
        {
            "schema_version": EDGE_QUEUE_SCHEMA_VERSION,
            "queue": "edge-risk",
            "status": "ready" if candidates else "no_candidates",
            "candidate_count": len(candidates),
            "shown_count": len(shown),
            "limit": limit,
            "edge_report_path": edge_report_path,
            "report_generated_at": report_generated_at,
            "candidates": shown,
            "review_progress": _edge_review_progress(
                candidate_count=len(candidates),
                edge_report_path=edge_report_path,
                report_generated_at=report_generated_at,
                edge_decision=progress_decision,
                consume_report=consume_report,
            ),
        }
    )
    return result


def build_review_show_from_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return _error("review_show payload must be an object", code="INVALID_REVIEW_PAYLOAD")
    queue, error = _review_queue(payload)
    if error:
        return error
    if queue == "edge-risk":
        return build_edge_review_show_from_payload(payload)
    packet_path, packet, error = _packet_from_payload(payload)
    if error:
        return error
    assert packet is not None
    candidate, error = _find_candidate(packet, payload.get("old_id"), payload.get("new_id"))
    if error:
        return error
    assert candidate is not None
    result = _base_result("review_show", packet_path)
    result.update({"status": "ready", "candidate": _add_identity_labels(candidate)})
    return result


def build_edge_review_show_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    report_path, report, error = _edge_report_from_payload(payload)
    if error:
        return error
    assert report is not None
    assert report_path is not None
    candidates = _manual_review_edge_candidates(report, report_path)
    candidate, error = _find_edge_candidate(candidates, payload.get("old_id"), payload.get("new_id"))
    if error:
        return error
    assert candidate is not None
    result = _base_result("review_show", None)
    result.update(
        {
            "schema_version": EDGE_QUEUE_SCHEMA_VERSION,
            "queue": "edge-risk",
            "status": "ready",
            "candidate": candidate,
            "candidate_count": len(candidates),
            "edge_report_path": str(report_path),
            "report_generated_at": report.get("generated_at"),
        }
    )
    return result


def _candidate_focus_sort_key(candidate: dict[str, Any]) -> tuple[int, float, int, str, str]:
    disposition_priority = {"review_recommended": 0, "needs_context": 1}.get(str(candidate.get("disposition") or ""), 2)
    try:
        score_sort = -float(candidate.get("score"))
    except (TypeError, ValueError):
        score_sort = float("inf")
    evidence_priority = 0 if bool(candidate.get("evidence_complete")) else 1
    old_id, new_id = _candidate_key(candidate)
    return (disposition_priority, score_sort, evidence_priority, old_id, new_id)


def _focused_candidate(packet: dict[str, Any]) -> dict[str, Any] | None:
    candidates = [row for row in packet.get("candidates") or [] if isinstance(row, dict)]
    if not candidates:
        return None
    return sorted(candidates, key=_candidate_focus_sort_key)[0]


def build_review_focus_from_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        payload = {}
    queue, error = _review_queue(payload)
    if error:
        return error
    if queue == "edge-risk":
        return build_edge_review_focus_from_payload(payload)
    packet_path, packet, error = _packet_from_payload(payload)
    if error:
        return error
    assert packet is not None
    result = _base_result("review_focus", packet_path)
    candidate_count = len([row for row in packet.get("candidates") or [] if isinstance(row, dict)])
    candidate = _focused_candidate(packet)
    if candidate is None:
        result.update(
            {
                "schema_version": FOCUS_SCHEMA_VERSION,
                "status": "no_candidates",
                "candidate": None,
                "candidate_count": 0,
                "focus_reason": "No review candidates are present in this packet.",
                "safe_next_commands": ["pith trust review status", "pith trust review list", "pith trust review next"],
                "decision_command_templates": [],
                "post_export_measure_command_template": "pith trust review measure --extension <output_path>",
            }
        )
        return result
    enriched = _add_identity_labels(candidate)
    inspect_command = _inspect_command(enriched)
    result.update(
        {
            "schema_version": FOCUS_SCHEMA_VERSION,
            "status": "ready",
            "candidate": enriched,
            "candidate_count": candidate_count,
            "focus_reason": (
                f"Selected {enriched.get('disposition') or 'unknown'} candidate with score "
                f"{_format_score(enriched.get('score'))}; complete_evidence={bool(enriched.get('evidence_complete'))}."
            ),
            "inspect_command": inspect_command,
            "safe_next_commands": ["pith trust review focus", inspect_command, "pith trust review list"],
            "decision_command_templates": _decision_command_templates(enriched),
            "post_export_measure_command_template": "pith trust review measure --extension <output_path>",
        }
    )
    return result


def build_edge_review_focus_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    report_path, report, error = _edge_report_from_payload(payload)
    if error:
        return error
    assert report is not None
    assert report_path is not None
    candidates = _manual_review_edge_candidates(report, report_path)
    candidate = _focused_edge_candidate(candidates)
    result = _base_result("review_focus", None)
    result.update(
        {
            "schema_version": EDGE_FOCUS_SCHEMA_VERSION,
            "queue": "edge-risk",
            "candidate_count": len(candidates),
            "edge_report_path": str(report_path),
            "report_generated_at": report.get("generated_at"),
        }
    )
    if candidate is None:
        result.update(
            {
                "status": "no_candidates",
                "candidate": None,
                "focus_reason": "No manual-review edge-risk rows are present in this report.",
                "safe_next_commands": [
                    "pith trust review status --queue edge-risk",
                    "pith trust review list --queue edge-risk",
                    "pith trust review next",
                ],
                "decision_command_templates": [],
            }
        )
        return result
    result.update(
        {
            "status": "ready",
            "candidate": candidate,
            "focus_reason": (
                f"Selected {candidate.get('edge_kind') or 'unknown'} edge-risk row with "
                f"repair class {candidate.get('repairability_class') or 'unknown'}."
            ),
            "inspect_command": _edge_inspect_command(candidate),
            "safe_next_commands": [
                "pith trust review focus --queue edge-risk",
                _edge_inspect_command(candidate),
                "pith trust review list --queue edge-risk",
            ],
            "decision_command_templates": _edge_decision_command_templates(candidate),
        }
    )
    return result


def build_edge_review_batch_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    report_path, report, error = _edge_report_from_payload(payload)
    if error:
        return error
    assert report is not None
    assert report_path is not None
    limit, error = _payload_int(payload, "limit", DEFAULT_EDGE_REVIEW_BATCH_LIMIT, minimum=0)
    if error:
        return error
    assert limit is not None
    try:
        source_report_sha256 = _edge_source_report_sha256(report_path)
    except OSError as exc:
        return _error(
            f"Failed to hash edge report: {exc}", code="INVALID_SUPERSESSION_EDGE_REPORT", field="edge_report_path"
        )
    candidates = sorted(_manual_review_edge_candidates(report, report_path), key=_edge_focus_sort_key)
    reviewed_keys = _source_current_reviewed_edge_keys(
        reports_dir=_reports_dir(payload),
        report_path=report_path,
        report_generated_at=report.get("generated_at"),
        source_report_sha256=source_report_sha256,
    )
    include_reviewed = bool(payload.get("include_reviewed"))
    pending_candidates = (
        candidates if include_reviewed else [candidate for candidate in candidates if _edge_candidate_key(candidate) not in reviewed_keys]
    )
    draft_candidates = pending_candidates[:limit]
    generated_at = _utc_now_iso()
    review_source = str(payload.get("review_source") or "pith trust review batch")
    output_dir = _edge_batch_output_dir(_expand_path(payload.get("output_dir")))
    draft_path = output_dir / "trust_review_edge_decision_draft.json"
    markdown_path = output_dir / "trust_review_edge_decision_packet.md"
    reviewed_decisions_path = output_dir / "trust_review_edge_decisions_reviewed.json"
    run_command = f"pith trust review run --queue edge-risk --decision-file {draft_path} --output {reviewed_decisions_path}"
    consume_command = f"pith trust review consume --queue edge-risk --decision-file {reviewed_decisions_path}"
    metrics = {
        "candidate_count": len(candidates),
        "source_current_decision_artifact_count": len(
            _source_current_edge_decision_artifacts(
                reports_dir=_reports_dir(payload),
                report_path=report_path,
                report_generated_at=report.get("generated_at"),
                source_report_sha256=source_report_sha256,
            )
        ),
        "reviewed_current_report_count": len(reviewed_keys),
        "pending_before_batch_count": len(pending_candidates),
        "draft_count": len(draft_candidates),
        "pending_after_batch_count": max(len(pending_candidates) - len(draft_candidates), 0),
        "include_reviewed": include_reviewed,
    }
    batch = {
        "schema_version": EDGE_BATCH_SCHEMA_VERSION,
        "operation": "review_batch",
        "queue": "edge-risk",
        "status": "drafted" if draft_candidates else "no_pending_candidates",
        "generated_at": generated_at,
        "claim_boundary": EDGE_REVIEW_CLAIM_BOUNDARY,
        "mutates_authority": False,
        "mutation_count": 0,
        "mutation_authorized": False,
        "source_report_path": str(report_path),
        "source_generated_at": str(report.get("generated_at") or ""),
        "source_report_sha256": source_report_sha256,
        "candidate_count": len(candidates),
        "limit": limit,
        "include_reviewed": include_reviewed,
        "metrics": metrics,
        "decisions": [_edge_batch_decision_draft(candidate, review_source=review_source) for candidate in draft_candidates],
        "allowed_decisions": sorted(EDGE_REVIEW_DECISIONS),
        "decision_guidance": EDGE_REVIEW_DECISION_EXPLANATIONS,
        "stop_rule": EDGE_BATCH_STOP_RULE,
        "safe_next_commands": [
            f"open {markdown_path}",
            f"open {draft_path}",
            run_command,
            consume_command,
        ],
    }
    paths = _write_edge_batch_artifacts(batch, output_dir)
    batch["paths"] = paths
    batch["output_dir"] = paths["output_dir"]
    batch["decision_draft_path"] = paths["decision_draft"]
    batch["markdown_packet_path"] = paths["markdown_packet"]
    return batch


def _read_decision_file(path: Path) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None]:
    payload, error = _decision_file_payload(path)
    if error:
        return None, error
    rows = payload.get("decisions") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return None, _error(
            "decision_file must contain a JSON array or an object with a decisions array",
            code="INVALID_REVIEW_DECISION",
            field="decision_file",
        )
    if not all(isinstance(row, dict) for row in rows):
        return None, _error(
            "decision_file rows must be JSON objects", code="INVALID_REVIEW_DECISION", field="decision_file"
        )
    return list(rows), None


def _decision_objects(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if isinstance(payload.get("decisions"), list):
        decisions = payload["decisions"]
        if not all(isinstance(row, dict) for row in decisions):
            return [], _error("decisions rows must be JSON objects", code="INVALID_REVIEW_DECISION", field="decisions")
        return list(decisions), None
    decision_file = _expand_path(payload.get("decision_file"))
    if decision_file is not None:
        loaded, error = _read_decision_file(decision_file)
        if error:
            return [], error
        assert loaded is not None
        return loaded, None
    if payload.get("old_id") or payload.get("new_id") or payload.get("decision"):
        return [
            {
                "old_id": payload.get("old_id"),
                "new_id": payload.get("new_id"),
                "decision": payload.get("decision"),
                "retention_mode": payload.get("retention_mode"),
                "reviewer": payload.get("reviewer"),
                "rationale": payload.get("rationale") or payload.get("review_rationale"),
            }
        ], None
    return [], _error("review decisions are required", code="INVALID_REVIEW_DECISION", field="decisions")


def _decision_to_row(
    packet: dict[str, Any],
    decision_payload: dict[str, Any],
    *,
    default_reviewer: str = "",
    review_source: str = "",
) -> tuple[dict[str, str] | None, dict[str, Any] | None]:
    candidate, error = _find_candidate(packet, decision_payload.get("old_id"), decision_payload.get("new_id"))
    if error:
        return None, error
    assert candidate is not None
    decision = str(decision_payload.get("decision") or "").strip()
    if decision not in REVIEW_DECISIONS:
        return None, _error(
            f"Invalid review_decision {decision!r}; allowed: {', '.join(sorted(REVIEW_DECISIONS))}",
            code="INVALID_REVIEW_DECISION",
            field="decision",
        )
    retention_mode = str(decision_payload.get("retention_mode") or "").strip()
    if not retention_mode:
        retention_mode = (
            "replace"
            if decision == "approve_supersession"
            else "partial_retain"
            if decision == "partial_retain"
            else "keep_both"
        )
    if retention_mode not in RETENTION_MODES:
        return None, _error(
            f"Invalid retention_mode {retention_mode!r}; allowed: {', '.join(sorted(RETENTION_MODES))}",
            code="INVALID_REVIEW_DECISION",
            field="retention_mode",
        )
    rationale = str(decision_payload.get("rationale") or decision_payload.get("review_rationale") or "").strip()
    if decision in {"approve_supersession", "partial_retain"} and not rationale:
        return None, _error(f"{decision} requires rationale", code="INVALID_REVIEW_DECISION", field="rationale")
    reviewer = str(decision_payload.get("reviewer") or default_reviewer or "").strip()
    if not reviewer:
        return None, _error("reviewer is required", code="INVALID_REVIEW_DECISION", field="reviewer")
    return {
        "old_id": str(candidate.get("old_id") or ""),
        "new_id": str(candidate.get("new_id") or ""),
        "subject_key": str(candidate.get("subject_key") or ""),
        "old_created_at": str(candidate.get("old_created_at") or ""),
        "new_created_at": str(candidate.get("new_created_at") or ""),
        "old_summary": str(candidate.get("old_summary") or ""),
        "new_summary": str(candidate.get("new_summary") or ""),
        "review_schema_version": str(candidate.get("review_schema_version") or ""),
        "review_group_key": str(candidate.get("review_group_key") or ""),
        "review_decision": decision,
        "retention_mode": retention_mode,
        "reviewer": reviewer,
        "reviewed_at": str(decision_payload.get("reviewed_at") or _utc_now_iso()),
        "review_rationale": rationale,
        "review_source": str(decision_payload.get("review_source") or review_source or ""),
    }, None


def _decision_rows(
    packet: dict[str, Any],
    payload: dict[str, Any],
) -> tuple[list[dict[str, str]], dict[str, Any] | None]:
    decisions, error = _decision_objects(payload)
    if error:
        return [], error
    seen: set[tuple[str, str]] = set()
    rows: list[dict[str, str]] = []
    for decision in decisions:
        key = (str(decision.get("old_id") or ""), str(decision.get("new_id") or ""))
        if key in seen:
            return [], _error(f"Duplicate review decision: {key[0]} -> {key[1]}", code="INVALID_REVIEW_DECISION")
        seen.add(key)
        row, error = _decision_to_row(
            packet,
            decision,
            default_reviewer=str(payload.get("reviewer") or ""),
            review_source=str(payload.get("review_source") or ""),
        )
        if error:
            return [], error
        assert row is not None
        rows.append(row)
    return rows, None


def _edge_decision_to_row(
    candidates: list[dict[str, Any]],
    decision_payload: dict[str, Any],
    *,
    report_path: Path,
    source_report_sha256: str,
    default_reviewer: str = "",
    review_source: str = "",
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    candidate, error = _find_edge_candidate(
        candidates,
        decision_payload.get("old_id"),
        decision_payload.get("replacement_id") or decision_payload.get("new_id"),
    )
    if error:
        return None, error
    assert candidate is not None
    decision = str(decision_payload.get("decision") or decision_payload.get("review_decision") or "").strip()
    if decision not in EDGE_REVIEW_DECISIONS:
        return None, _error(
            f"Invalid edge review_decision {decision!r}; allowed: {', '.join(sorted(EDGE_REVIEW_DECISIONS))}",
            code="INVALID_REVIEW_DECISION",
            field="decision",
        )
    rationale = str(decision_payload.get("rationale") or decision_payload.get("review_rationale") or "").strip()
    if decision in EDGE_REPAIR_DECISIONS_REQUIRING_RATIONALE and not rationale:
        return None, _error(f"{decision} requires rationale", code="INVALID_REVIEW_DECISION", field="rationale")
    reviewer = str(decision_payload.get("reviewer") or default_reviewer or "").strip()
    if not reviewer:
        return None, _error("reviewer is required", code="INVALID_REVIEW_DECISION", field="reviewer")

    row = dict(candidate)
    row.update(
        {
            "schema_version": EDGE_DECISIONS_SCHEMA_VERSION,
            "review_decision": decision,
            "reviewer": reviewer,
            "reviewed_at": str(decision_payload.get("reviewed_at") or _utc_now_iso()),
            "review_rationale": rationale,
            "review_source": str(decision_payload.get("review_source") or review_source or ""),
            "source_report_path": str(report_path),
            "source_report_sha256": source_report_sha256,
            "source_row_hash": _edge_source_row_hash(candidate),
            "mutates_authority": False,
            "mutation_count": 0,
        }
    )
    return row, None


def _edge_decision_rows(
    candidates: list[dict[str, Any]],
    payload: dict[str, Any],
    *,
    report_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    decisions, error = _decision_objects(payload)
    if error:
        return [], error
    if not decisions:
        return [], _error("edge review decisions are required", code="INVALID_REVIEW_DECISION", field="decisions")
    try:
        source_report_sha256 = _edge_source_report_sha256(report_path)
    except OSError as exc:
        return [], _error(
            f"Failed to hash edge report: {exc}", code="INVALID_SUPERSESSION_EDGE_REPORT", field="edge_report_path"
        )
    seen: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []
    for decision in decisions:
        key = (
            str(decision.get("old_id") or "").strip(),
            str(decision.get("replacement_id") or decision.get("new_id") or "").strip(),
        )
        if key in seen:
            return [], _error(f"Duplicate edge review decision: {key[0]} -> {key[1]}", code="INVALID_REVIEW_DECISION")
        seen.add(key)
        row, error = _edge_decision_to_row(
            candidates,
            decision,
            report_path=report_path,
            source_report_sha256=source_report_sha256,
            default_reviewer=str(payload.get("reviewer") or ""),
            review_source=str(payload.get("review_source") or ""),
        )
        if error:
            return [], error
        assert row is not None
        rows.append(row)
    return rows, None


def _default_output_path() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_REVIEW_EXTENSION_DIR / f"implicit_supersession_review_gold_extension-{stamp}.json"


def _default_edge_decision_output_path() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_EDGE_DECISION_DIR / f"trust_review_edge_decisions-{stamp}.json"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temp_path = Path(temp_name)
    try:
        with open(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.expanduser().read_bytes()).hexdigest()


def _default_edge_batch_output_dir() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_EDGE_BATCH_DIR / f"trust-review-edge-batch-{stamp}"


def _edge_batch_output_dir(requested: Path | None) -> Path:
    root = (requested or _default_edge_batch_output_dir()).expanduser()
    candidate = root
    suffix = 1
    while candidate.exists() and any(candidate.iterdir()):
        candidate = root.with_name(f"{root.name}-{suffix}")
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def _edge_batch_decision_draft(candidate: dict[str, Any], *, review_source: str) -> dict[str, Any]:
    return {
        "old_id": candidate.get("old_id"),
        "new_id": candidate.get("replacement_id") or candidate.get("new_id"),
        "replacement_id": candidate.get("replacement_id") or candidate.get("new_id"),
        "decision": "",
        "reviewer": "",
        "rationale": "",
        "review_source": review_source,
        "allowed_decisions": sorted(EDGE_REVIEW_DECISIONS),
        "decision_guidance": EDGE_REVIEW_DECISION_EXPLANATIONS,
        "context": {
            "edge_kind": candidate.get("edge_kind"),
            "repairability_class": candidate.get("repairability_class"),
            "old_subject_key": candidate.get("old_subject_key"),
            "replacement_subject_key": candidate.get("replacement_subject_key"),
            "old_summary_preview": candidate.get("old_summary_preview") or candidate.get("summary_preview"),
            "replacement_summary_preview": candidate.get("replacement_summary_preview"),
            "summary_preview": candidate.get("summary_preview"),
            "risk_reason": candidate.get("risk_reason"),
            "supersession_reason": candidate.get("supersession_reason"),
            "superseded_at": candidate.get("superseded_at"),
            "state_flags": candidate.get("state_flags") if isinstance(candidate.get("state_flags"), list) else [],
        },
    }


def _render_edge_batch_markdown(batch: dict[str, Any]) -> str:
    metrics = batch.get("metrics") if isinstance(batch.get("metrics"), dict) else {}
    lines = [
        "# Trust Review Edge-Risk Batch",
        "",
        f"- Generated: {batch.get('generated_at')}",
        f"- Status: {batch.get('status')}",
        f"- Source report: {batch.get('source_report_path')}",
        f"- Source generated: {batch.get('source_generated_at')}",
        f"- Source sha256: {batch.get('source_report_sha256')}",
        "- Timestamp convention: UTC; superseded_at is when the old concept was marked superseded",
        f"- Total candidates: {metrics.get('candidate_count', 0)}",
        f"- Reviewed current report: {metrics.get('reviewed_current_report_count', 0)}",
        f"- Pending before batch: {metrics.get('pending_before_batch_count', 0)}",
        f"- Draft count: {metrics.get('draft_count', 0)}",
        f"- Pending after batch: {metrics.get('pending_after_batch_count', 0)}",
        f"- Include reviewed: {batch.get('include_reviewed', False)}",
        f"- Boundary: {batch.get('claim_boundary')}",
        "",
        f"Stop rule: {batch.get('stop_rule') or EDGE_BATCH_STOP_RULE}",
        "",
        "Edit the JSON draft next to this packet. Use one decision per row: approve_repair, reject_repair, needs_context, not_supersession, or uncertain. approve_repair requires rationale.",
        "",
        "Decision guidance:",
        *[f"- `{decision}`: {explanation}" for decision, explanation in EDGE_REVIEW_DECISION_EXPLANATIONS.items()],
        "",
        "## Safe Next Commands",
        "",
    ]
    for command in batch.get("safe_next_commands") or []:
        lines.append(f"- `{command}`")
    lines.extend(["", "## Candidates", ""])
    for index, decision in enumerate(batch.get("decisions") or [], start=1):
        context = decision.get("context") if isinstance(decision.get("context"), dict) else {}
        lines.extend(
            [
                f"### {index}. {decision.get('old_id')} -> {decision.get('replacement_id') or decision.get('new_id')}",
                "",
                f"- Edge kind: `{context.get('edge_kind') or 'unknown'}`",
                f"- Repair class: `{context.get('repairability_class') or 'unknown'}`",
                f"- Old subject: {context.get('old_subject_key') or 'unknown'}",
                f"- Replacement subject: {context.get('replacement_subject_key') or 'unknown'}",
                f"- Superseded at: {_format_time_utc(context.get('superseded_at'))}",
                f"- State flags: {_format_flags(context.get('state_flags'))}",
                f"- Risk reason: {_truncate_text(context.get('risk_reason'), limit=500)}",
                f"- Supersession reason: {_truncate_text(context.get('supersession_reason'), limit=500)}",
                "",
                f"Old summary: {_truncate_text(context.get('old_summary_preview') or context.get('summary_preview'), limit=1000)}",
                "",
                f"Replacement summary: {_truncate_text(context.get('replacement_summary_preview'), limit=1000)}",
                "",
                "Decision: `<approve_repair | reject_repair | needs_context | not_supersession | uncertain>`",
                "Rationale: `<required for approve_repair>`",
                "",
            ]
        )
    return "\n".join(lines)


def _write_edge_batch_artifacts(batch: dict[str, Any], output_dir: Path) -> dict[str, str]:
    draft_path = output_dir / "trust_review_edge_decision_draft.json"
    packet_path = output_dir / "trust_review_edge_decision_packet.md"
    _write_json_atomic(draft_path, batch)
    _write_text_no_overwrite(packet_path, _render_edge_batch_markdown(batch))
    return {"decision_draft": str(draft_path), "markdown_packet": str(packet_path), "output_dir": str(output_dir)}


def _decision_file_payload(path: Path) -> tuple[Any, dict[str, Any] | None]:
    try:
        return json.loads(path.expanduser().read_text(encoding="utf-8")), None
    except (OSError, json.JSONDecodeError) as exc:
        return None, _error(f"Invalid decision file: {exc}", code="INVALID_REVIEW_DECISION", field="decision_file")


def _edge_decision_source_lock_from_payload(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    decision_file = _expand_path(payload.get("decision_file"))
    if decision_file is None:
        return None, None
    raw, error = _decision_file_payload(decision_file)
    if error:
        return None, error
    if not isinstance(raw, dict) or raw.get("schema_version") != EDGE_BATCH_SCHEMA_VERSION:
        return None, None
    source_path = str(raw.get("source_report_path") or "").strip()
    if not source_path:
        return None, _error("batch draft missing source_report_path", code="INVALID_REVIEW_DECISION", field="source_report_path")
    raw_candidate_count = raw.get("candidate_count") or 0
    try:
        candidate_count = int(raw_candidate_count)
    except (TypeError, ValueError):
        return None, _error(
            "batch draft candidate_count must be an integer",
            code="INVALID_REVIEW_DECISION",
            field="candidate_count",
        )
    return {
        "source_report_path": source_path,
        "source_generated_at": str(raw.get("source_generated_at") or ""),
        "source_report_sha256": str(raw.get("source_report_sha256") or ""),
        "candidate_count": candidate_count,
        "decision_file": str(decision_file),
    }, None


def _apply_edge_decision_source_lock(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    lock, error = _edge_decision_source_lock_from_payload(payload)
    if error:
        return payload, error
    if not lock:
        return payload, None
    working = dict(payload)
    explicit_report = _expand_path(working.get("edge_report_path") or working.get("edge_report"))
    locked_report = Path(lock["source_report_path"]).expanduser()
    if explicit_report is not None and explicit_report.resolve(strict=False) != locked_report.resolve(strict=False):
        return payload, _error(
            "decision draft source report conflicts with explicit edge_report_path",
            code="DRIFT_BLOCKED",
            field="edge_report_path",
        )
    working["edge_report_path"] = str(locked_report)
    try:
        observed_sha = _file_sha256(locked_report)
    except OSError as exc:
        return payload, _error(f"Failed to hash draft source report: {exc}", code="DRIFT_BLOCKED", field="source_report_sha256")
    if lock.get("source_report_sha256") and observed_sha != lock["source_report_sha256"]:
        return payload, _error("decision draft source report hash drifted", code="DRIFT_BLOCKED", field="source_report_sha256")
    report, load_error = load_edge_report(locked_report)
    if load_error:
        return payload, load_error
    assert report is not None
    if str(lock.get("source_generated_at") or "") != str(report.get("generated_at") or ""):
        return payload, _error("decision draft source report timestamp drifted", code="DRIFT_BLOCKED", field="source_generated_at")
    rows = _manual_review_edge_candidates(report, locked_report)
    if int(lock.get("candidate_count") or 0) and int(lock.get("candidate_count") or 0) != len(rows):
        return payload, _error("decision draft source candidate count drifted", code="DRIFT_BLOCKED", field="candidate_count")
    return working, None


def _default_edge_consume_output_dir() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_EDGE_CONSUME_DIR / f"trust-review-decision-consume-{stamp}"


def _default_edge_consume_pointer_path(reports_dir: Path) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    root = reports_dir.expanduser() / "trust-review-decision-consumer"
    path = root / f"trust_review_consume_pointer-{stamp}.json"
    suffix = 1
    while path.exists():
        path = root / f"trust_review_consume_pointer-{stamp}-{suffix}.json"
        suffix += 1
    return path


def _edge_consume_output_dir(requested: Path | None) -> Path:
    root = requested or _default_edge_consume_output_dir()
    candidate = root.expanduser()
    suffix = 1
    while candidate.exists() and any(candidate.iterdir()):
        candidate = root.with_name(f"{root.name}-{suffix}")
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def _write_edge_consume_pointer(result: dict[str, Any], reports_dir: Path) -> Path:
    pointer_path = _default_edge_consume_pointer_path(reports_dir)
    pointer = _consume_pointer_summary(result, source_report_available=True)
    _write_json_atomic(pointer_path, pointer)
    return pointer_path


def _load_edge_decision_artifact(path: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    raw, read_error = _bounded_file_bytes(
        path,
        MAX_EDGE_DECISION_ARTIFACT_BYTES,
        missing_code="decision_artifact_missing",
        unreadable_code="decision_artifact_unreadable",
        too_large_code="decision_artifact_too_large",
    )
    if read_error:
        return None, _error(
            f"Failed to load edge decision artifact: {read_error}",
            code="INVALID_DECISION_ARTIFACT",
            field="decision_file",
        )
    assert raw is not None
    try:
        artifact = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, _error(
            f"Failed to load edge decision artifact: {exc}",
            code="INVALID_DECISION_ARTIFACT",
            field="decision_file",
        )
    validation_error = _validate_edge_decision_artifact_payload(artifact)
    return (None, validation_error) if validation_error else (artifact, None)


def _load_edge_source_report_for_consume(
    row: dict[str, Any],
    override_path: Path | None,
) -> tuple[Path | None, dict[str, Any] | None, dict[str, Any] | None]:
    path = override_path or _expand_path(row.get("source_report_path"))
    if path is None:
        return None, None, _error(
            "source report path is required for edge decision consume",
            code="INVALID_DECISION_ARTIFACT",
            field="source_report_path",
        )
    report, error = load_edge_report(path)
    if error:
        return path, None, error
    assert report is not None
    expected = str(row.get("source_report_sha256") or "")
    observed = _file_sha256(path)
    if expected != observed:
        return path, None, _error(
            f"source report hash drifted for {path}",
            code="DRIFT_BLOCKED",
            field="source_report_sha256",
        )
    return path, report, None


def _source_edge_candidate_for_row(
    row: dict[str, Any],
    report: dict[str, Any],
    report_path: Path,
) -> dict[str, Any] | None:
    old_id = str(row.get("old_id") or "")
    replacement_id = str(row.get("replacement_id") or row.get("new_id") or "")
    for candidate in _manual_review_edge_candidates(report, report_path):
        if _edge_candidate_key(candidate) == (old_id, replacement_id):
            return candidate
    return None


def _validate_edge_decision_source_rows(
    rows: list[dict[str, Any]],
    *,
    override_report_path: Path | None,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, Any] | None]:
    source_reports: dict[str, tuple[Path, dict[str, Any]]] = {}
    validated: list[dict[str, Any]] = []
    metrics = {
        "source_report_count": 0,
        "source_report_hash_match_count": 0,
        "source_row_hash_match_count": 0,
        "drifted_count": 0,
    }
    for row in rows:
        path, report, error = _load_edge_source_report_for_consume(row, override_report_path)
        if error:
            metrics["drifted_count"] += 1
            return [], metrics, error
        assert path is not None and report is not None
        source_reports.setdefault(str(path), (path, report))
        metrics["source_report_hash_match_count"] += 1
        candidate = _source_edge_candidate_for_row(row, report, path)
        if candidate is None:
            metrics["drifted_count"] += 1
            return [], metrics, _error(
                f"source row missing for {row.get('old_id')} -> {row.get('replacement_id') or row.get('new_id')}",
                code="DRIFT_BLOCKED",
                field="source_row_hash",
            )
        if _edge_source_row_hash(candidate) != str(row.get("source_row_hash") or ""):
            metrics["drifted_count"] += 1
            return [], metrics, _error(
                f"source row hash drifted for {row.get('old_id')} -> {row.get('replacement_id') or row.get('new_id')}",
                code="DRIFT_BLOCKED",
                field="source_row_hash",
            )
        metrics["source_row_hash_match_count"] += 1
        normalized = dict(row)
        normalized["replacement_id"] = str(row.get("replacement_id") or row.get("new_id") or "")
        validated.append(normalized)
    metrics["source_report_count"] = len(source_reports)
    return validated, metrics, None


def _write_edge_consume_decision_tsv(
    rows: list[dict[str, Any]],
    ledger_lines: list[dict[str, Any]],
    path: Path,
) -> tuple[dict[str, int], dict[str, Any] | None]:
    from app.ops.supersession_repair_ledger import DECISION_TEMPLATE_FIELDS

    ledger_by_old_id = {str(line.get("old_id") or ""): line for line in ledger_lines}
    metrics = {
        "matched_ledger_line_count": 0,
        "replacement_match_count": 0,
        "ledger_replacement_drift_count": 0,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=DECISION_TEMPLATE_FIELDS, delimiter="\t")
    writer.writeheader()
    for row in rows:
        old_id = str(row.get("old_id") or "")
        replacement_id = str(row.get("replacement_id") or "")
        ledger_line = ledger_by_old_id.get(old_id)
        if not ledger_line:
            return metrics, _error(f"ledger row missing for {old_id}", code="LEDGER_BLOCKED")
        metrics["matched_ledger_line_count"] += 1
        if str(ledger_line.get("replacement_id") or "") != replacement_id:
            metrics["ledger_replacement_drift_count"] += 1
            return metrics, _error(
                f"ledger replacement drifted for {old_id}: expected {replacement_id}, found {ledger_line.get('replacement_id')}",
                code="LEDGER_BLOCKED",
                field="replacement_id",
            )
        metrics["replacement_match_count"] += 1
        writer.writerow(
            {
                "line_number": ledger_line.get("line_number"),
                "old_id": old_id,
                "verified_line_hash": ledger_line.get("verified_line_hash"),
                "verified_ledger_sha256": (ledger_line.get("source_snapshot") or {}).get("verified_ledger_sha256")
                or "",
                "source_old_ids_sha256": ((ledger_line.get("source_snapshot") or {}).get("source_old_ids_sha256") or ""),
                "review_decision": row.get("review_decision"),
                "reviewer": row.get("reviewer"),
                "reviewed_at": row.get("reviewed_at"),
                "reviewer_notes": row.get("review_rationale") or row.get("reviewer_notes") or "",
            }
        )
    _write_text_no_overwrite(path, output.getvalue())
    return metrics, None


def _rewrite_verified_ledger_hash_in_decision_tsv(path: Path, verified_ledger_sha256: str) -> None:
    rows = load_csv_rows(path)
    output = io.StringIO()
    from app.ops.supersession_repair_ledger import DECISION_TEMPLATE_FIELDS

    writer = csv.DictWriter(output, fieldnames=DECISION_TEMPLATE_FIELDS, delimiter="\t")
    writer.writeheader()
    for row in rows:
        row = dict(row)
        row["verified_ledger_sha256"] = verified_ledger_sha256
        writer.writerow({field: row.get(field, "") for field in DECISION_TEMPLATE_FIELDS})
    path.write_text(output.getvalue(), encoding="utf-8")


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [{key: (value or "").strip() for key, value in row.items()} for row in reader]


def _write_required_line_set_tsv(decisions_path: Path, required_path: Path) -> None:
    from app.ops.supersession_repair_ledger import DECISION_TEMPLATE_FIELDS

    rows = load_csv_rows(decisions_path)
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=DECISION_TEMPLATE_FIELDS, delimiter="\t")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in DECISION_TEMPLATE_FIELDS})
    _write_text_no_overwrite(required_path, output.getvalue())


def _export_extension(
    rows: list[dict[str, str]], payload: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    output_path = _expand_path(payload.get("output_path") or payload.get("output")) or _default_output_path()
    report = build_extension(rows, review_source=str(payload.get("review_source") or "pith trust review"))
    if report.get("status") != "PASS":
        return report, _error("Review extension validation failed", code="REVIEW_EXTENSION_VALIDATION_FAILED")
    try:
        _write_json_atomic(output_path, report)
    except (OSError, FileExistsError) as exc:
        return report, _error(
            f"Failed to write reviewed extension: {exc}", code="REVIEW_EXTENSION_WRITE_FAILED", field="output_path"
        )
    report["output_path"] = str(output_path)
    return report, None


def _edge_decision_summary(rows: list[dict[str, Any]], *, source_report_sha256: str) -> dict[str, Any]:
    decision_counts: dict[str, int] = {}
    for row in rows:
        decision = str(row.get("review_decision") or "unknown")
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
    return {
        "decision_count": len(rows),
        "decision_counts": dict(sorted(decision_counts.items())),
        "approved_repair_count": decision_counts.get("approve_repair", 0),
        "rejected_or_no_action_count": (
            decision_counts.get("reject_repair", 0) + decision_counts.get("not_supersession", 0)
        ),
        "needs_context_count": decision_counts.get("needs_context", 0),
        "uncertain_count": decision_counts.get("uncertain", 0),
        "source_report_sha256": source_report_sha256,
        "mutation_authorized": False,
    }


def _export_edge_decisions(
    rows: list[dict[str, Any]],
    payload: dict[str, Any],
    *,
    report_path: Path,
    report_generated_at: str = "",
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    output_path = _expand_path(payload.get("output_path") or payload.get("output")) or _default_edge_decision_output_path()
    source_report_sha256 = rows[0].get("source_report_sha256") if rows else _edge_source_report_sha256(report_path)
    report = {
        "schema_version": EDGE_DECISIONS_SCHEMA_VERSION,
        "status": "exported",
        "generated_at": _utc_now_iso(),
        "decision_count": len(rows),
        "summary": _edge_decision_summary(rows, source_report_sha256=str(source_report_sha256 or "")),
        "source_report_path": str(report_path),
        "source_report_sha256": str(source_report_sha256 or ""),
        "source_generated_at": report_generated_at,
        "decisions": rows,
        "claim_boundary": EDGE_REVIEW_CLAIM_BOUNDARY,
        "mutates_authority": False,
        "mutation_count": 0,
        "mutation_authorized": False,
    }
    try:
        _write_json_atomic(output_path, report)
    except (OSError, FileExistsError) as exc:
        return report, _error(
            f"Failed to write edge review decisions: {exc}",
            code="REVIEW_EDGE_DECISIONS_WRITE_FAILED",
            field="output_path",
        )
    report["output_path"] = str(output_path)
    return report, None


def _measure_extensions(payload: dict[str, Any], extension_path: Path | None = None) -> dict[str, Any]:
    review_extension = extension_path or _expand_path(payload.get("review_extension") or payload.get("extension"))
    review_extension_dir = _expand_path(payload.get("review_extension_dir") or payload.get("extension_dir"))
    min_reviewed, error = _payload_int(payload, "min_reviewed_cases_for_readiness", 20)
    if error:
        return error
    min_approved, error = _payload_int(payload, "min_approved_cases_for_readiness", 5)
    if error:
        return error
    min_rejected, error = _payload_int(payload, "min_rejected_or_needs_context_cases_for_readiness", 5)
    if error:
        return error
    report = build_scorecard(
        gold_path=DEFAULT_GOLD_PATH,
        review_extension_paths=[review_extension] if review_extension else [],
        review_extension_dirs=[review_extension_dir] if review_extension_dir else ([] if review_extension else None),
        min_reviewed_cases_for_readiness=min_reviewed if min_reviewed is not None else 20,
        min_approved_cases_for_readiness=min_approved if min_approved is not None else 5,
        min_rejected_or_needs_context_cases_for_readiness=min_rejected if min_rejected is not None else 5,
    )
    return {
        "status": report.get("status"),
        "case_count": report.get("case_count"),
        "metrics": report.get("metrics") or {},
        "target_status": report.get("target_status") or {},
        "review_extensions": report.get("review_extensions") or {},
        "safe_apply_readiness": report.get("safe_apply_readiness") or {},
        "mutation_authorized": bool((report.get("safe_apply_readiness") or {}).get("mutation_authorized")),
    }


def build_review_decide_from_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return _error("review_decide payload must be an object", code="INVALID_REVIEW_PAYLOAD")
    queue, error = _review_queue(payload)
    if error:
        return error
    if queue == "edge-risk":
        return build_edge_review_decide_from_payload(payload)
    packet_path, packet, error = _packet_from_payload(payload)
    if error:
        return error
    assert packet is not None
    rows, error = _decision_rows(packet, payload)
    if error:
        return error
    result = _base_result("review_decide", packet_path)
    result.update({"status": "ready", "rows": rows, "decision_count": len(rows)})
    return result


def build_edge_review_decide_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    report_path, report, error = _edge_report_from_payload(payload)
    if error:
        return error
    assert report is not None
    assert report_path is not None
    candidates = _manual_review_edge_candidates(report, report_path)
    rows, error = _edge_decision_rows(candidates, payload, report_path=report_path)
    if error:
        return error
    result = _base_result("review_decide", None)
    result.update(
        {
            "schema_version": EDGE_DECISIONS_SCHEMA_VERSION,
            "status": "ready",
            "claim_boundary": EDGE_REVIEW_CLAIM_BOUNDARY,
            "queue": "edge-risk",
            "edge_report_path": str(report_path),
            "report_generated_at": report.get("generated_at"),
            "rows": rows,
            "decision_count": len(rows),
            "summary": _edge_decision_summary(
                rows,
                source_report_sha256=str(rows[0].get("source_report_sha256") if rows else ""),
            ),
        }
    )
    return result


def build_review_export_from_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return _error("review_export payload must be an object", code="INVALID_REVIEW_PAYLOAD")
    queue, error = _review_queue(payload)
    if error:
        return error
    if queue == "edge-risk":
        return build_edge_review_export_from_payload(payload)
    packet_path, packet, error = _packet_from_payload(payload)
    if error:
        return error
    assert packet is not None
    rows, error = _decision_rows(packet, payload)
    if error:
        return error
    extension, error = _export_extension(rows, payload)
    if error:
        return {**error, "extension": extension}
    result = _base_result("review_export", packet_path)
    result.update({"status": "exported", "extension": extension, "output_path": extension.get("output_path")})
    return result


def build_edge_review_export_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    payload, source_lock_error = _apply_edge_decision_source_lock(payload)
    if source_lock_error:
        return source_lock_error
    report_path, report, error = _edge_report_from_payload(payload)
    if error:
        return error
    assert report is not None
    assert report_path is not None
    candidates = _manual_review_edge_candidates(report, report_path)
    rows, error = _edge_decision_rows(candidates, payload, report_path=report_path)
    if error:
        return error
    artifact, error = _export_edge_decisions(
        rows,
        payload,
        report_path=report_path,
        report_generated_at=str(report.get("generated_at") or ""),
    )
    if error:
        return {**error, "artifact": artifact}
    result = _base_result("review_export", None)
    result.update(
        {
            "schema_version": EDGE_DECISIONS_SCHEMA_VERSION,
            "status": "exported",
            "claim_boundary": EDGE_REVIEW_CLAIM_BOUNDARY,
            "queue": "edge-risk",
            "edge_report_path": str(report_path),
            "report_generated_at": report.get("generated_at"),
            "artifact": artifact,
            "output_path": artifact.get("output_path"),
            "decision_count": len(rows),
            "summary": artifact.get("summary") or {},
        }
    )
    return result


def build_review_measure_from_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        payload = {}
    result = _base_result("review_measure")
    result.update({"status": "measured", "measurement": _measure_extensions(payload)})
    return result


def build_review_calibration_from_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        payload = {}
    paths = _iter_calibration_packet_paths(payload)
    records, errors = _load_calibration_candidates(paths)
    candidates = sorted(
        records.values(),
        key=lambda row: (
            str(row.get("risk_class") or ""),
            str(row.get("latest_seen_at") or ""),
            str(row.get("old_id") or ""),
            str(row.get("new_id") or ""),
        ),
    )
    if payload.get("limit") is not None:
        limit, error = _payload_int(payload, "limit", 0, minimum=1)
        if error:
            return error
        assert limit is not None
        candidates = candidates[:limit]
    risk_counts: dict[str, int] = {}
    for candidate in candidates:
        risk = str(candidate.get("risk_class") or "unknown")
        risk_counts[risk] = risk_counts.get(risk, 0) + 1
    minimum, error = _payload_int(payload, "minimum_reviewable_cases", 20, minimum=0)
    if error:
        return error
    assert minimum is not None
    status = "ready" if len(candidates) >= minimum else "insufficient_candidate_volume"
    result = _base_result("review_calibration")
    result.update(
        {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "status": status,
            "generated_at": _utc_now_iso(),
            "packet_paths": [str(path) for path in paths],
            "errors": errors,
            "summary": {
                "packet_count": len(paths),
                "unique_candidate_count": len(candidates),
                "minimum_reviewable_cases": minimum,
                "risk_class_counts": dict(sorted(risk_counts.items())),
                "source_packet_complete_count": sum(
                    1 for row in candidates if row.get("source_packet_count") and row.get("source_packet_paths")
                ),
                "temporal_complete_count": sum(
                    1 for row in candidates if row.get("old_created_at") and row.get("new_created_at")
                ),
            },
            "candidates": candidates,
        }
    )
    if not payload.get("no_write"):
        output_dir = _expand_path(payload.get("output_dir"))
        if output_dir is None:
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            output_dir = DEFAULT_CALIBRATION_DIR / f"implicit-supersession-review-calibration-{stamp}"
        try:
            result["artifact_paths"] = _write_calibration_artifacts(result, output_dir)
        except (OSError, FileExistsError) as exc:
            return _error(
                f"Failed to write calibration artifacts: {exc}", code="CALIBRATION_WRITE_FAILED", field="output_dir"
            )
    return result


def _prompt_decisions(packet: dict[str, Any], reviewer: str) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for candidate in packet.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        print(_format_candidate_detail(candidate))
        decision = input(
            "Decision [approve_supersession/reject_not_supersession/needs_more_context/partial_retain]: "
        ).strip()
        retention_mode = input("Retention mode [replace/keep_both/partial_retain]: ").strip()
        rationale = input("Rationale: ").strip()
        decisions.append(
            {
                "old_id": candidate.get("old_id"),
                "new_id": candidate.get("new_id"),
                "decision": decision,
                "retention_mode": retention_mode,
                "reviewer": reviewer,
                "rationale": rationale,
            }
        )
    return decisions


def build_review_run_from_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        payload = {}
    queue, error = _review_queue(payload)
    if error:
        return error
    if queue == "edge-risk":
        return build_edge_review_run_from_payload(payload)
    packet_path, packet, error = _packet_from_payload(payload)
    if error:
        return error
    assert packet is not None
    working_payload = dict(payload)
    if "decisions" not in working_payload and "decision_file" not in working_payload:
        if not sys.stdin.isatty():
            return _error(
                "review_run requires decisions or decision_file when stdin is not a TTY", code="INVALID_REVIEW_DECISION"
            )
        reviewer = str(payload.get("reviewer") or "").strip() or input("Reviewer: ").strip()
        working_payload["decisions"] = _prompt_decisions(packet, reviewer)
    rows, error = _decision_rows(packet, working_payload)
    if error:
        return error
    result = _base_result("review_run", packet_path)
    result.update({"status": "reviewed", "decision_count": len(rows), "rows": rows})
    if not payload.get("no_export"):
        extension, error = _export_extension(rows, working_payload)
        if error:
            return {**error, "rows": rows, "extension": extension}
        result.update({"extension": extension, "output_path": extension.get("output_path")})
        if not payload.get("no_measure"):
            result["measurement"] = _measure_extensions(working_payload, _expand_path(extension.get("output_path")))
    return result


def build_edge_review_run_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if "decisions" not in payload and "decision_file" not in payload:
        return _error(
            (
                "review_run --queue edge-risk requires decision_file or decisions. "
                "Use pith trust review focus --queue edge-risk to inspect candidates first."
            ),
            code="INVALID_REVIEW_PAYLOAD",
            field="decision_file",
        )
    result = build_edge_review_export_from_payload(payload)
    if result.get("error"):
        return result
    result["operation"] = "review_run"
    result["status"] = "reviewed"
    return result


def build_edge_review_consume_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    decision_path = _expand_path(payload.get("decision_file") or payload.get("decision_file_path"))
    if decision_path is None:
        return _error("review_consume requires decision_file", code="INVALID_REVIEW_PAYLOAD", field="decision_file")
    artifact, error = _load_edge_decision_artifact(decision_path)
    if error:
        return error
    assert artifact is not None
    rows = artifact.get("decisions") if isinstance(artifact.get("decisions"), list) else artifact.get("rows")
    assert isinstance(rows, list)
    override_report_path = _expand_path(payload.get("edge_report_path"))
    validated_rows, source_metrics, error = _validate_edge_decision_source_rows(
        rows,
        override_report_path=override_report_path,
    )
    if error:
        return {**error, "source_metrics": source_metrics}

    output_dir = _edge_consume_output_dir(_expand_path(payload.get("output_dir") or payload.get("ledger_dir")))
    db_path = _expand_path(payload.get("db_path"))
    if db_path is None:
        from app.storage import DB_PATH

        db_path = Path(DB_PATH)
    if not db_path.is_file():
        return _error(f"db_path does not exist or is not a file: {db_path}", code="INVALID_REVIEW_PAYLOAD", field="db_path")
    ledger_dir = output_dir / "ledger"
    decisions_tsv_path = output_dir / "ledger_review_decisions.tsv"
    required_tsv_path = output_dir / "required_decision_line_set.tsv"
    queue_dir = output_dir / "repair_queue"
    consume_report_path = output_dir / "trust_review_decision_consume_report.json"

    try:
        from app.ops.supersession_repair_ledger import (
            apply_approved_mechanical_repairs,
            generate_ledger,
            verify_bundle,
            write_ledger_files,
        )
        from app.ops.trust_maintenance_repair_queue import (
            build_repair_work_units_from_ledger,
            load_decision_rows,
            write_repair_queue_artifacts,
        )

        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            generated = generate_ledger(conn, source_db_path=str(db_path.resolve() if db_path.exists() else db_path))
            verified, verification = verify_bundle(conn, generated, strict_row_trust=False)
        write_ledger_files(verified, ledger_dir)
        ledger_metrics, error = _write_edge_consume_decision_tsv(validated_rows, verified.lines, decisions_tsv_path)
        if error:
            return {**error, "source_metrics": source_metrics, "ledger_metrics": ledger_metrics}
        verified_sha = str(verified.manifest.get("verified_ledger_sha256") or "")
        _rewrite_verified_ledger_hash_in_decision_tsv(decisions_tsv_path, verified_sha)
        _write_required_line_set_tsv(decisions_tsv_path, required_tsv_path)
        decision_rows = load_decision_rows(decisions_tsv_path)
        queue_manifest = build_repair_work_units_from_ledger(
            verified,
            decision_rows,
            ledger_path=ledger_dir / "ledger.jsonl",
            manifest_path=ledger_dir / "manifest.json",
            decisions_path=decisions_tsv_path,
        )
        queue_paths = write_repair_queue_artifacts(queue_manifest, output_dir=queue_dir)
        decisions_sha = _file_sha256(decisions_tsv_path)
        queue_sha = _file_sha256(Path(queue_paths["manifest_json"]))
        required_sha = _file_sha256(required_tsv_path)
        dry_run = apply_approved_mechanical_repairs(
            source_db_path=db_path,
            out_dir=ledger_dir,
            decisions_path=decisions_tsv_path,
            queue_manifest_path=Path(queue_paths["manifest_json"]),
            approved_verified_ledger_sha256=verified_sha,
            approved_decisions_sha256=decisions_sha,
            approved_queue_sha256=queue_sha,
            required_decision_line_set_path=required_tsv_path,
            approved_required_decision_line_set_sha256=required_sha,
            dry_run=True,
            allow_partial_review=True,
        )
    except (OSError, ValueError, sqlite3.Error) as exc:
        return _error(f"review_consume failed: {exc}", code="LEDGER_BLOCKED")

    summary = artifact.get("summary") if isinstance(artifact.get("summary"), dict) else {}
    queue_summary = queue_manifest.get("summary") if isinstance(queue_manifest.get("summary"), dict) else {}
    status = "SUCCESS"
    if int(dry_run.get("eligible_line_count") or 0) == 0:
        status = "NO_APPROVED_REPAIRS"
    result = {
        "schema_version": EDGE_CONSUME_SCHEMA_VERSION,
        "operation": "review_consume",
        "queue": "edge-risk",
        "status": status,
        "generated_at": _utc_now_iso(),
        "claim_boundary": EDGE_CONSUME_CLAIM_BOUNDARY,
        "decision_file": str(decision_path),
        "db_path": str(db_path),
        "mutates_authority": False,
        "mutation_count": 0,
        "changed_rows": int(dry_run.get("changed_rows") or 0),
        "decision_count": len(validated_rows),
        "approved_repair_count": int(summary.get("approved_repair_count") or 0),
        "no_action_count": int(summary.get("rejected_or_no_action_count") or 0),
        "blocked_by_review_count": int(summary.get("needs_context_count") or 0) + int(summary.get("uncertain_count") or 0),
        "source_metrics": source_metrics,
        "ledger_metrics": {
            **ledger_metrics,
            "ledger_line_count": len(verified.lines),
            "verified_ledger_sha256": verified_sha,
            "verification_status": verification.status,
        },
        "queue_summary": queue_summary,
        "dry_run_summary": {
            "status": dry_run.get("status"),
            "eligible_line_count": int(dry_run.get("eligible_line_count") or 0),
            "changed_rows": int(dry_run.get("changed_rows") or 0),
            "decision_validation_status": dry_run.get("decision_validation_status"),
            "required_line_set_validation": dry_run.get("required_line_set_validation"),
        },
        "paths": {
            "output_dir": str(output_dir),
            "consume_report": str(consume_report_path),
            "ledger_dir": str(ledger_dir),
            "ledger_jsonl": str(ledger_dir / "ledger.jsonl"),
            "ledger_manifest": str(ledger_dir / "manifest.json"),
            "decisions_tsv": str(decisions_tsv_path),
            "required_decision_line_set_tsv": str(required_tsv_path),
            "repair_queue_manifest": queue_paths["manifest_json"],
            "repair_queue_summary": queue_paths["summary_md"],
            "approved_repair_dry_run": str(ledger_dir / "approved_mechanical_apply_report.json"),
        },
        "next_commands": [
            (
                "python3 scripts/supersession_repair_ledger.py "
                f"--db-path {db_path} --out-dir {ledger_dir} --apply-approved-repair "
                f"--decisions-tsv {decisions_tsv_path} --queue-manifest-json {queue_paths['manifest_json']} "
                f"--approved-verified-ledger-sha256 {verified_sha} --approved-decisions-sha256 {decisions_sha} "
                f"--approved-queue-sha256 {queue_sha} --required-decision-line-set-tsv {required_tsv_path} "
                f"--approved-required-decision-line-set-sha256 {required_sha} --dry-run --json"
            ),
            (
                "python3 scripts/supersession_repair_ledger.py "
                f"--db-path {db_path} --out-dir {ledger_dir} --apply-approved-repair-live "
                f"--decisions-tsv {decisions_tsv_path} --queue-manifest-json {queue_paths['manifest_json']} "
                f"--approved-verified-ledger-sha256 {verified_sha} --approved-decisions-sha256 {decisions_sha} "
                f"--approved-queue-sha256 {queue_sha} --required-decision-line-set-tsv {required_tsv_path} "
                f"--approved-required-decision-line-set-sha256 {required_sha} --dry-run --json"
            ),
        ],
    }
    _write_json_atomic(consume_report_path, result)
    try:
        pointer_path = _write_edge_consume_pointer(result, _reports_dir(payload))
    except (OSError, FileExistsError) as exc:
        result["consume_pointer_error"] = {
            "code": "CONSUME_POINTER_WRITE_FAILED",
            "message": str(exc),
        }
    else:
        result["paths"]["consume_pointer"] = str(pointer_path)
    return result


def build_from_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return _error("trust review payload must be an object", code="INVALID_REVIEW_PAYLOAD")
    operation = str(payload.get("operation") or "").strip()
    if operation == "review_next":
        return build_review_next_from_payload(payload)
    if operation == "review_status":
        return build_review_status_from_payload(payload)
    if operation == "review_list":
        return build_review_list_from_payload(payload)
    if operation == "review_show":
        return build_review_show_from_payload(payload)
    if operation == "review_focus":
        return build_review_focus_from_payload(payload)
    if operation == "review_batch":
        queue, error = _review_queue(payload)
        if error:
            return error
        if queue != "edge-risk":
            return _error("review_batch only supports queue=edge-risk", code="INVALID_REVIEW_PAYLOAD", field="queue")
        return build_edge_review_batch_from_payload(payload)
    if operation == "review_decide":
        return build_review_decide_from_payload(payload)
    if operation == "review_export":
        return build_review_export_from_payload(payload)
    if operation == "review_consume":
        queue, error = _review_queue(payload)
        if error:
            return error
        if queue != "edge-risk":
            return _error("review_consume only supports queue=edge-risk", code="INVALID_REVIEW_PAYLOAD", field="queue")
        return build_edge_review_consume_from_payload(payload)
    if operation == "review_measure":
        return build_review_measure_from_payload(payload)
    if operation == "review_calibration":
        return build_review_calibration_from_payload(payload)
    if operation == "review_run":
        return build_review_run_from_payload(payload)
    return _error(
        f"Unsupported trust review operation: {operation}", code="INVALID_REVIEW_OPERATION", field="operation"
    )


def _format_candidate_detail(candidate: dict[str, Any]) -> str:
    disposition = str(candidate.get("disposition") or "unknown")
    old_identity = (
        candidate.get("old_identity")
        if isinstance(candidate.get("old_identity"), dict)
        else _belief_identity(candidate, "old")
    )
    new_identity = (
        candidate.get("new_identity")
        if isinstance(candidate.get("new_identity"), dict)
        else _belief_identity(candidate, "new")
    )
    lines = [
        "Decision packet",
        f"  Subject:   {candidate.get('subject_key') or 'unknown'}",
        f"  Status:    {disposition.replace('_', ' ')} (score {_format_score(candidate.get('score'))})",
        f"  Guidance:  {DISPOSITION_GUIDANCE.get(disposition, 'Review manually before deciding.')}",
        f"  Why:       {_format_signals(candidate.get('signals'))}",
        f"  Read-only: complete_evidence={candidate.get('evidence_complete')}; no authority mutation performed",
        f"  Old identity: {old_identity.get('display') or 'unknown'}",
        f"  New identity: {new_identity.get('display') or 'unknown'}",
        f"  Inspect:   {_inspect_command(candidate)}",
        f"  Decide:    {_decision_command(candidate)}",
        "  Decision options:",
        *[f"    - {decision}: {explanation}" for decision, explanation in REVIEW_DECISION_EXPLANATIONS.items()],
        "",
        "Old belief",
        f"  Concept:   {candidate.get('old_id')}",
        f"  Learned:   {_format_time(candidate.get('old_created_at'))}",
        f"  Summary:   {_truncate_text(candidate.get('old_summary'), limit=500)}",
    ]
    lines.extend(_source_lines("old belief source", candidate.get("old_source")))
    lines.extend(
        [
            "",
            "Proposed newer belief",
            f"  Concept:   {candidate.get('new_id')}",
            f"  Learned:   {_format_time(candidate.get('new_created_at'))}",
            f"  Summary:   {_truncate_text(candidate.get('new_summary'), limit=500)}",
        ]
    )
    lines.extend(_source_lines("new belief source", candidate.get("new_source")))
    return "\n".join(lines)


def _format_review_artifact_lines(result: dict[str, Any], *, indent: str = "  ") -> list[str]:
    decision = result.get("latest_edge_decision_artifact") or result.get("edge_decision_artifact")
    consume = result.get("latest_consume_report") or result.get("consume_report")
    if not isinstance(decision, dict) and not isinstance(consume, dict):
        return []
    decision = decision if isinstance(decision, dict) else {}
    consume = consume if isinstance(consume, dict) else {}
    provenance = decision.get("provenance") if isinstance(decision.get("provenance"), dict) else {}
    lines = ["", f"{indent}Latest review artifacts:"]
    lines.extend(
        [
            f"{indent}  Decision available: {decision.get('available', False)}",
            f"{indent}  Decision generated: {decision.get('generated_at') or 'unknown'}",
            f"{indent}  Decision count: {decision.get('decision_count', 0)}",
            f"{indent}  Decision artifact: {decision.get('path') or 'none'}",
            f"{indent}  Decision provenance: {provenance.get('provenance_status') or 'unavailable'}",
            f"{indent}  Rejected decision artifacts: {decision.get('rejected_artifact_count', 0)}",
            f"{indent}  Consume available: {consume.get('available', False)}",
            f"{indent}  Consume status: {consume.get('status') or 'unknown'}",
            f"{indent}  Consume generated: {consume.get('generated_at') or 'unknown'}",
            f"{indent}  Eligible repairs: {consume.get('eligible_line_count', 0)}",
            f"{indent}  Changed rows: {consume.get('changed_rows', 0)}",
            f"{indent}  Source report available: {consume.get('source_report_available', False)}",
            f"{indent}  Consume artifact: {consume.get('consume_report_path') or consume.get('path') or 'none'}",
        ]
    )
    rejected = decision.get("latest_rejected_artifact")
    if isinstance(rejected, dict):
        reasons = ", ".join(str(value) for value in rejected.get("rejection_reasons") or [])
        lines.extend(
            [
                f"{indent}  Newest rejected artifact: {rejected.get('path') or 'unknown'}",
                f"{indent}  Rejection reason: {reasons or 'unknown'}",
            ]
        )
    continuation = result.get("artifact_continuation")
    if isinstance(continuation, dict) and continuation.get("action"):
        lines.append(f"{indent}  Next: {continuation.get('action')}")
        if continuation.get("command"):
            lines.append(f"{indent}  Command: {continuation.get('command')}")
    return lines


def _format_review_progress_lines(result: dict[str, Any], *, indent: str = "  ") -> list[str]:
    progress = result.get("review_progress")
    if not isinstance(progress, dict):
        return []
    candidate_count = int(progress.get("candidate_count") or 0)
    reviewed_count = int(progress.get("reviewed_current_report_count") or 0)
    coverage_pct = float(progress.get("coverage_ratio") or 0.0) * 100.0
    source_current = "true" if progress.get("decision_source_current") is True else "false"
    lines = [
        "",
        f"{indent}Review progress:",
        f"{indent}  Reviewed current report: {reviewed_count}/{candidate_count} ({coverage_pct:.1f}%)",
        f"{indent}  Pending review: {progress.get('pending_review_count', 0)}",
        f"{indent}  Decision source current: {source_current}",
        f"{indent}  Approved repairs: {progress.get('approved_repair_count', 0)}",
        f"{indent}  No action: {progress.get('no_action_count', 0)}",
        f"{indent}  Needs context/uncertain: {progress.get('blocked_by_review_count', 0)}",
        f"{indent}  Eligible dry-run repairs: {progress.get('eligible_repair_count', 0)}",
    ]
    if progress.get("next_safe_command"):
        lines.append(f"{indent}  Next safe command: {progress.get('next_safe_command')}")
    if progress.get("decision_source_current") is False and progress.get("decision_source_report_path"):
        lines.append(
            f"{indent}  Source warning: latest decision artifact is stale for this report; regenerate or review against the current edge report."
        )
        lines.append(f"{indent}  Stale decision source: {progress.get('decision_source_report_path')}")
    if progress.get("claim_boundary"):
        lines.append(f"{indent}  Boundary: {progress.get('claim_boundary')}")
    return lines


def format_review_text(result: dict[str, Any]) -> str:
    if result.get("error"):
        return f"[Trust Review]\n  Status: error\n  Code:   {result.get('code')}\n  Error:  {result.get('message')}"
    operation = result.get("operation")
    lines = ["[Trust Review]", f"  Operation: {operation}", f"  Status:    {result.get('status', 'unknown')}"]
    queue = str(result.get("queue") or "proposal")
    if queue == "proposal":
        lines.append("  Queue:     proposal (default implicit proposal packet)")
        if operation in {"review_status", "review_next"}:
            lines.append("  Edge-risk: use `pith trust review status --queue edge-risk` for Trust Health edge rows")
    else:
        lines.append(f"  Queue:     {queue}")
    if result.get("packet_path"):
        lines.append(f"  Packet:    {result.get('packet_path')}")
    if result.get("edge_report_path"):
        lines.append(f"  Edge report: {result.get('edge_report_path')}")
    if result.get("report_generated_at"):
        lines.append(f"  Generated: {result.get('report_generated_at')}")
    if operation in {"review_status", "review_list"}:
        lines.append(f"  Candidates:{result.get('candidate_count', 0)}")
        if queue == "edge-risk":
            lines.extend(_format_review_progress_lines(result))
    if operation == "review_status":
        lines.extend(_format_review_artifact_lines(result))
    if operation == "review_list" and queue == "edge-risk":
        lines.append(f"  Shown:     {result.get('shown_count', 0)} of {result.get('candidate_count', 0)}")
        lines.append(f"  Limit:     {result.get('limit', DEFAULT_EDGE_REVIEW_LIMIT)}")
    if operation == "review_next":
        from scripts import trust_health_status

        broad = (
            result.get("broad_supersession_edge_queue")
            if isinstance(result.get("broad_supersession_edge_queue"), dict)
            else {}
        )
        latest = (
            result.get("latest_proposal_packet_queue")
            if isinstance(result.get("latest_proposal_packet_queue"), dict)
            else {}
        )
        selected_queue = str(result.get("selected_review_queue") or "proposal")
        risks = broad.get("risk_counts") if isinstance(broad.get("risk_counts"), dict) else {}
        alarm_status = {"alarm_summary": result.get("trust_health_alarm_summary") or {}}
        edge_queue_selected = selected_queue == "edge-risk"
        latest_label = "Selected edge-risk report" if edge_queue_selected else "Latest implicit proposal packet"
        packet_label = "Report" if edge_queue_selected else "Packet"
        latest_path = latest.get("edge_report_path") if edge_queue_selected else latest.get("packet_path")
        lines.extend(
            [
                f"  Trust:     {result.get('trust_health_status', 'unknown')}",
                f"  Primary:   {result.get('primary_user_action') or 'none'}",
                "  Start here: run the first safe command below; do not use decision commands until you have inspected a specific candidate.",
                "",
                "  Broad supersession edge-risk queue:",
                f"    Manual review: {risks.get('manual_review_required', 0)}",
                f"    Missing identity: {risks.get('missing_identity_edges', 0)}",
                f"    Cross-subject duplicates: {risks.get('reflection_duplicate_cross_subject', 0)}",
                f"    Evidence: {broad.get('evidence_path') or 'unknown'}",
                "",
                f"  {latest_label}:",
                f"    Available: {latest.get('available', False)}",
            ]
        )
        if edge_queue_selected:
            lines.append(f"    Manual review candidates: {latest.get('candidate_count', 0)}")
        else:
            lines.append(f"    Candidates: {latest.get('candidate_count', 0)}")
            lines.append(f"    Needs context: {latest.get('needs_context_count', 0)}")
        lines.extend(
            [
                f"    {packet_label}: {latest_path or 'none'}",
                "",
                f"  Distinction: {result.get('queue_distinction')}",
            ]
        )
        lines.extend(_format_review_progress_lines(result))
        lines.extend(_format_review_artifact_lines(result))
        lines.extend(trust_health_status.format_alarm_summary_lines(alarm_status, indent="  "))
        lines.extend(["", "  Safe next commands:"])
        lines.extend(f"    - {command}" for command in result.get("safe_inspection_commands") or [])
        lines.append("  Decision commands, only after explicit review:")
        lines.extend(f"    - {command}" for command in result.get("decision_commands") or [])
    if operation == "review_calibration":
        summary = result.get("summary") or {}
        lines.extend(
            [
                f"  Packets:   {summary.get('packet_count', 0)}",
                f"  Candidates:{summary.get('unique_candidate_count', 0)}",
                f"  Minimum:   {summary.get('minimum_reviewable_cases', 0)}",
            ]
        )
        for key, value in (summary.get("risk_class_counts") or {}).items():
            lines.append(f"    - {key}: {value}")
        artifacts = result.get("artifact_paths") or {}
        if artifacts.get("markdown_path"):
            lines.append(f"  Packet:    {artifacts.get('markdown_path')}")
    if operation == "review_list":
        if queue == "edge-risk":
            lines.append("  Edge-risk review rows:")
            for index, candidate in enumerate(result.get("candidates") or [], start=1):
                lines.extend(_format_edge_candidate_list_packet(candidate, index))
        else:
            lines.append("  Review packets:")
            for index, candidate in enumerate(result.get("candidates") or [], start=1):
                lines.extend(_format_candidate_list_packet(candidate, index))
    if operation == "review_show" and isinstance(result.get("candidate"), dict):
        detail = (
            _format_edge_candidate_detail(result["candidate"])
            if queue == "edge-risk"
            else _format_candidate_detail(result["candidate"])
        )
        lines.extend(["", detail])
    if operation == "review_focus":
        lines.extend(["", "  Focused candidate:", f"    Why selected: {result.get('focus_reason') or 'unknown'}"])
        if result.get("inspect_command"):
            lines.append(f"    Inspect: {result.get('inspect_command')}")
        safe_commands = result.get("safe_next_commands") if isinstance(result.get("safe_next_commands"), list) else []
        if safe_commands:
            lines.append("    Safe next commands:")
            lines.extend(f"      - {command}" for command in safe_commands)
        templates = (
            result.get("decision_command_templates")
            if isinstance(result.get("decision_command_templates"), list)
            else []
        )
        if templates:
            lines.append("    Decision templates:")
            lines.extend(f"      - {command}" for command in templates)
        if result.get("post_export_measure_command_template"):
            lines.append(f"    After export: {result.get('post_export_measure_command_template')}")
        if isinstance(result.get("candidate"), dict):
            detail = (
                _format_edge_candidate_detail(result["candidate"])
                if queue == "edge-risk"
                else _format_candidate_detail(result["candidate"])
            )
            lines.extend(["", detail])
    if operation == "review_batch":
        metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        lines.extend(
            [
                f"  Candidates:{metrics.get('candidate_count', result.get('candidate_count', 0))}",
                f"  Reviewed:  {metrics.get('reviewed_current_report_count', 0)} current-source rows",
                f"  Pending:   {metrics.get('pending_before_batch_count', 0)} before batch",
                f"  Drafted:   {metrics.get('draft_count', 0)}",
                f"  Remaining: {metrics.get('pending_after_batch_count', 0)} after this draft",
                f"  Source SHA:{result.get('source_report_sha256') or 'unknown'}",
                f"  Markdown:  {result.get('markdown_packet_path') or 'none'}",
                f"  Draft:     {result.get('decision_draft_path') or 'none'}",
            ]
        )
        commands = result.get("safe_next_commands") if isinstance(result.get("safe_next_commands"), list) else []
        if commands:
            lines.append("  Safe next commands:")
            lines.extend(f"    - {command}" for command in commands)
    if operation in {"review_decide", "review_run", "review_export"}:
        lines.append(f"  Decisions: {result.get('decision_count', 0)}")
        summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
        if summary:
            lines.append(f"  Approve repair: {summary.get('approved_repair_count', 0)}")
            lines.append(f"  No action: {summary.get('rejected_or_no_action_count', 0)}")
            lines.append(f"  Needs context: {summary.get('needs_context_count', 0)}")
            lines.append(f"  Uncertain: {summary.get('uncertain_count', 0)}")
        if result.get("output_path"):
            lines.append(f"  Output:    {result.get('output_path')}")
    if operation == "review_consume":
        source_metrics = result.get("source_metrics") if isinstance(result.get("source_metrics"), dict) else {}
        ledger_metrics = result.get("ledger_metrics") if isinstance(result.get("ledger_metrics"), dict) else {}
        dry_run = result.get("dry_run_summary") if isinstance(result.get("dry_run_summary"), dict) else {}
        lines.extend(
            [
                f"  Decisions: {result.get('decision_count', 0)}",
                f"  Approved:  {result.get('approved_repair_count', 0)}",
                f"  No action: {result.get('no_action_count', 0)}",
                f"  Blocked:   {result.get('blocked_by_review_count', 0)}",
                f"  Drifted:   {source_metrics.get('drifted_count', 0)}",
                f"  Ledger:    {ledger_metrics.get('verification_status', 'unknown')} ({ledger_metrics.get('matched_ledger_line_count', 0)} matched)",
                f"  Eligible:  {dry_run.get('eligible_line_count', 0)}",
                f"  Changed:   {result.get('changed_rows', 0)}",
            ]
        )
        paths = result.get("paths") if isinstance(result.get("paths"), dict) else {}
        if paths.get("consume_report"):
            lines.append(f"  Report:    {paths.get('consume_report')}")
        if paths.get("repair_queue_manifest"):
            lines.append(f"  Queue:     {paths.get('repair_queue_manifest')}")
        commands = result.get("next_commands") if isinstance(result.get("next_commands"), list) else []
        if commands:
            lines.append("  Next commands:")
            lines.extend(f"    - {command}" for command in commands)
    measurement = result.get("measurement")
    if isinstance(measurement, dict):
        readiness = measurement.get("safe_apply_readiness") or {}
        review_extensions = measurement.get("review_extensions") or {}
        lines.extend(
            [
                f"  Scorecard: {measurement.get('status')}",
                f"  Reviewed:  {review_extensions.get('case_count', 0)}",
                f"  Readiness: {readiness.get('status')}",
                f"  Mutation:  authorized={readiness.get('mutation_authorized', False)}",
            ]
        )
    lines.append("")
    lines.append("Review evidence is read-only; it does not mutate Pith authority.")
    return "\n".join(lines)


def _common_packet_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--packet", dest="packet_path", help="Implicit proposal packet JSON for proposal queue review.")
    parser.add_argument("--queue", help="Review queue. Use edge-risk for broad Trust Health edge rows.")
    parser.add_argument("--edge-report", dest="edge_report_path", help="Supersession edge semantics report JSON.")
    parser.add_argument("--limit", help="Maximum rows to show for list/focus-style commands.")
    parser.add_argument("--reports-dir", help="Monitoring reports root. Defaults to ~/.pith/reports/monitoring.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text.")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Review implicit supersession proposals and edge-risk trust maintenance rows.",
        epilog=(
            "Review commands are read-only by default. Edge-risk consume builds measured repair dry-run evidence; "
            "live repair still requires the separate hash-pinned supersession repair ledger command it prints."
        ),
    )
    sub = parser.add_subparsers(dest="action", required=True)
    _common_packet_args(sub.add_parser("next", help="Show the safest next trust-maintenance action."))
    _common_packet_args(sub.add_parser("status", help="Summarize review queue availability."))
    _common_packet_args(sub.add_parser("list", help="List review candidates without mutating authority."))
    _common_packet_args(sub.add_parser("focus", help="Select one high-priority candidate to inspect."))
    batch = sub.add_parser("batch", help="Create an editable edge-risk review batch draft.")
    batch.add_argument("--output-dir", help="Directory for the batch Markdown and decision draft artifacts.")
    batch.add_argument("--review-source", default="pith trust review batch", help="Source label stored in draft rows.")
    batch.add_argument("--include-reviewed", action="store_true", help="Include source-current rows already reviewed.")
    _common_packet_args(batch)
    show = sub.add_parser("show", help="Show one candidate with evidence and decision options.")
    show.add_argument("old_id", help="Older concept id.")
    show.add_argument("new_id", help="Newer or replacement concept id.")
    _common_packet_args(show)
    decide = sub.add_parser("decide", help="Record one explicit review decision as read-only evidence.")
    decide.add_argument("old_id", help="Older concept id.")
    decide.add_argument("new_id", help="Newer or replacement concept id.")
    decide.add_argument("--decision", required=True, help="Decision label. Edge-risk: approve_repair, reject_repair, needs_context, not_supersession, uncertain.")
    decide.add_argument("--retention-mode", default="", help="Proposal queue retention mode; ignored for edge-risk decisions.")
    decide.add_argument("--reviewer", required=True, help="Human reviewer name or handle.")
    decide.add_argument("--rationale", default="", help="Required for approve_repair and approval-style proposal decisions.")
    _common_packet_args(decide)
    run = sub.add_parser("run", help="Batch review decisions into a read-only review artifact.")
    run.add_argument("--decision-file", help="JSON file containing decision objects.")
    run.add_argument("--reviewer", help="Default reviewer for interactive or batch decisions.")
    run.add_argument("--output", dest="output_path", help="Output artifact path.")
    run.add_argument("--review-source", default="pith trust review run", help="Source label stored in the artifact.")
    run.add_argument("--no-export", action="store_true", help="Do not write an extension/artifact.")
    run.add_argument("--no-measure", action="store_true", help="Do not run the review scorecard after export.")
    _common_packet_args(run)
    export = sub.add_parser("export", help="Export reviewed decisions as a read-only artifact.")
    export.add_argument("--decision-file", required=True, help="JSON file containing decision objects.")
    export.add_argument("--output", dest="output_path", help="Output artifact path.")
    export.add_argument("--review-source", default="pith trust review export", help="Source label stored in the artifact.")
    _common_packet_args(export)
    consume = sub.add_parser(
        "consume",
        help="Validate edge-risk decision artifacts and build a measured repair dry-run.",
        description=(
            "Validate a trust_review_edge_decisions.v1 artifact, reject source drift, "
            "and build measured repair dry-run evidence without mutating authority."
        ),
    )
    consume.add_argument("--decision-file", required=True, help="Path to a trust_review_edge_decisions.v1 artifact.")
    consume.add_argument("--output-dir", help="Directory for consume, ledger, queue, and dry-run reports.")
    consume.add_argument("--ledger-dir", dest="output_dir", help=argparse.SUPPRESS)
    consume.add_argument("--db-path", help="SQLite DB path. Defaults to the active Pith profile DB.")
    _common_packet_args(consume)
    measure = sub.add_parser("measure", help="Measure proposal-review coverage and readiness.")
    measure.add_argument("--extension", dest="review_extension", help="Review extension JSON path.")
    measure.add_argument("--extension-dir", dest="review_extension_dir", help="Directory containing review extension JSON files.")
    measure.add_argument("--min-reviewed-cases-for-readiness", type=int, help="Minimum reviewed cases for readiness.")
    measure.add_argument("--min-approved-cases-for-readiness", type=int, help="Minimum approved cases for readiness.")
    measure.add_argument("--min-rejected-or-needs-context-cases-for-readiness", type=int, help="Minimum rejected/needs-context cases for readiness.")
    measure.add_argument("--json", action="store_true", help="Print JSON instead of text.")
    calibration = sub.add_parser("calibration", help="Build a deduplicated historical proposal-review calibration packet.")
    calibration.add_argument("--reports-dir", help="Monitoring reports root.")
    calibration.add_argument("--packet", dest="packet_paths", action="append", help="Proposal packet JSON path; repeatable.")
    calibration.add_argument("--output-dir", help="Output directory.")
    calibration.add_argument("--limit", type=int, help="Maximum calibration rows.")
    calibration.add_argument("--minimum-reviewable-cases", type=int, help="Minimum rows required for ready status.")
    calibration.add_argument("--no-write", action="store_true", help="Return calibration JSON without writing artifacts.")
    calibration.add_argument("--json", action="store_true", help="Print JSON instead of text.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = vars(args)
    action = payload.pop("action")
    payload["operation"] = f"review_{action}"
    json_output = bool(payload.pop("json", False))
    result = build_from_payload(payload)
    if json_output:
        print(json.dumps(result, sort_keys=True))
    else:
        print(format_review_text(result))
    return 1 if result.get("error") is True else 0


if __name__ == "__main__":
    raise SystemExit(main())
