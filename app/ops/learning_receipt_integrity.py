"""Read-only, bounded learning receipt checks for the existing lifecycle monitor."""

import json
import sqlite3
import time
from collections import Counter
from datetime import datetime, timedelta, timezone

from pydantic import ValidationError

from app.core.models import ClientLearningReceipt

UTC = timezone.utc  # noqa: UP017 - runtime supports Python 3.10
MAX_ROWS = 1000
MAX_RESPONSE_BYTES = 262144
SQL_BUDGET_SECONDS = 2.0


def receipt_issue(body):
    """Return one bounded reason; never return stored learning content."""
    if not isinstance(body, dict):
        return "invalid_response"
    raw = body.get("client_learning_receipt")
    if raw is None:
        return "legacy_missing_receipt"
    if not isinstance(raw, dict):
        return "invalid_receipt"
    if type(raw.get("version")) is not int:
        return "invalid_version"
    if raw["version"] != 1:
        return "unsupported_version"
    try:
        receipt = ClientLearningReceipt.model_validate(raw, strict=True)
    except (ValidationError, ValueError):
        return "invalid_partition"
    accepted, errors = body.get("accepted_learning_events"), body.get("errors")
    if any(type(value) is not int or value < 0 for value in (accepted, errors)):
        return "invalid_counts"
    if body.get("processing_state") != "committed" or body.get("persistence_state") != "committed":
        return "committed_state_mismatch"
    saved = sum(item.status in {"created", "evolved"} for item in receipt.items)
    failed = sum(item.status == "error" for item in receipt.items)
    if accepted < saved or errors < failed:
        return "count_underreports_receipt"
    for item in receipt.items:
        is_saved = item.status in {"created", "evolved"}
        if is_saved and (not item.concept_id or not item.concept_id.strip()):
            return "saved_identity_missing"
        if is_saved != (item.persistence_evidence == "reported_saved"):
            return "saved_evidence_mismatch"
    state = body.get("learning_capture_state")
    if state == "degraded_terminal_session":
        return None if body.get("session_linkage_state") == "terminal_mismatch" else "terminal_linkage_mismatch"
    if not isinstance(state, str) or state not in {
        "accepted",
        "partial",
        "zero_learning",
        "error",
        "rejected",
        "deferred",
    }:
        return "unknown_capture_state"
    if (state in {"accepted", "partial"}) != (accepted > 0):
        return "capture_count_mismatch"
    adverse = bool(receipt.deferred_ranges) or any(
        item.status in {"error", "rejected", "deferred"} for item in receipt.items
    )
    if state == "accepted" and (errors or adverse):
        return "accepted_hides_partial"
    if state == "zero_learning" and (errors or adverse):
        return "zero_hides_outcome"
    if accepted == 0 and errors and state != "error":
        return "capture_error_mismatch"
    if state == "error" and not errors:
        return "error_without_error_count"
    return None


def scan_learning_receipts(conn, profile, *, now=None, limit=MAX_ROWS):
    """Use an owned read-only connection; caller closes it. No brain writes."""
    if type(limit) is not int or not 1 <= limit <= MAX_ROWS:
        raise ValueError("limit must be an integer from 1 to MAX_ROWS")
    current = now or datetime.now(UTC)
    cutoff = (current.astimezone(UTC) - timedelta(hours=24)).replace(tzinfo=None).isoformat()
    report = {
        "status": "OBSERVE",
        "window_start_utc": cutoff,
        "limit": limit,
        "scanned": 0,
        "valid": 0,
        "unverified": 0,
        "violations": 0,
        "truncated": False,
        "reasons": {},
        "capture_states": {},
    }
    reasons, states = Counter(), Counter()
    deadline = time.monotonic() + SQL_BUDGET_SECONDS
    conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
    try:
        cursor = conn.execute(
            """
            SELECT CASE WHEN length(CAST(response_json AS BLOB)) <= ?
                        THEN response_json ELSE NULL END, length(CAST(response_json AS BLOB))
            FROM write_request_replays
            WHERE endpoint = 'session_learn' AND profile = ? AND status = 'committed'
              AND updated_at >= ? AND updated_at <= ?
            ORDER BY updated_at DESC LIMIT ?
        """,
            (MAX_RESPONSE_BYTES, profile, cutoff, current.astimezone(UTC).replace(tzinfo=None).isoformat(), limit + 1),
        )
        for index, (raw, size) in enumerate(cursor):
            if index == limit:
                report["truncated"] = True
                break
            if time.monotonic() >= deadline:
                reasons["query_unavailable"] += 1
                break
            report["scanned"] += 1
            if size is not None and size > MAX_RESPONSE_BYTES:
                issue = "oversized_response"
            else:
                try:
                    body = json.loads(raw) if raw is not None else None
                    issue = receipt_issue(body)
                    state = body.get("learning_capture_state") if isinstance(body, dict) else None
                    if state in (
                        "accepted",
                        "partial",
                        "zero_learning",
                        "error",
                        "rejected",
                        "deferred",
                        "degraded_terminal_session",
                    ):
                        states[state] += 1
                except (TypeError, ValueError, RecursionError):
                    issue = "invalid_response"
            if issue is None:
                report["valid"] += 1
            elif issue in {"legacy_missing_receipt", "unsupported_version", "oversized_response"}:
                report["unverified"] += 1
                reasons[issue] += 1
            else:
                report["violations"] += 1
                reasons[issue] += 1
        cursor.close()
    except sqlite3.Error:
        reasons["query_unavailable"] += 1
    finally:
        conn.set_progress_handler(None, 0)
    if report["violations"] or reasons["query_unavailable"]:
        report["status"] = "FAILURE"
    elif report["scanned"] and not report["unverified"] and not report["truncated"]:
        report["status"] = "SUCCESS"
    report["reasons"], report["capture_states"] = dict(reasons), dict(states)
    return report
