"""Lifecycle remediation helpers for validated current-head concepts."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from app.core.datetime_utils import _utc_now_iso
from app.storage.concepts import resolve_contested_current_head_conn

CANONICAL_CURRENT_TYPES = frozenset({"constraint", "principle", "method", "system_model"})


@dataclass(frozen=True)
class LifecycleRemediationDecision:
    concept_id: str
    eligible: bool
    reason: str
    mode: str = "dry_run"
    prior_currency_status: str | None = None
    concept_type: str | None = None


def _loads(data: str | bytes | None) -> dict:
    if not data:
        return {}
    try:
        loaded = json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def evaluate_canonical_current_contested(
    conn: sqlite3.Connection,
    concept_id: str,
    *,
    required_summary_substrings: tuple[str, ...],
) -> LifecycleRemediationDecision:
    """Return whether a contested current concept is eligible for remediation."""
    row = conn.execute(
        """SELECT id, status, is_current, superseded_by, currency_status,
                  concept_type, always_activate, summary, data
           FROM concepts
           WHERE id = ?""",
        (concept_id,),
    ).fetchone()
    if not row:
        return LifecycleRemediationDecision(concept_id, False, "missing_concept")

    data = _loads(row["data"])
    if row["status"] != "active" or int(row["is_current"] or 0) != 1:
        return LifecycleRemediationDecision(concept_id, False, "not_active_current")
    if row["superseded_by"]:
        return LifecycleRemediationDecision(concept_id, False, "already_has_replacement")
    if row["currency_status"] != "CONTESTED":
        return LifecycleRemediationDecision(concept_id, False, "not_contested")
    if row["concept_type"] not in CANONICAL_CURRENT_TYPES:
        return LifecycleRemediationDecision(concept_id, False, "not_canonical_type")

    summary = (row["summary"] or "").lower()
    if not all(fragment.lower() in summary for fragment in required_summary_substrings):
        return LifecycleRemediationDecision(concept_id, False, "summary_assertion_not_verified")

    has_correction = bool(data.get("correction_evidence") or data.get("correction_supersession"))
    if not has_correction:
        return LifecycleRemediationDecision(concept_id, False, "missing_correction_provenance")

    return LifecycleRemediationDecision(
        concept_id,
        True,
        "validated_current_head",
        prior_currency_status=row["currency_status"],
        concept_type=row["concept_type"],
    )


def protects_lifecycle_remediated_current_head(row: sqlite3.Row) -> bool:
    """Return true when maintenance must not re-contest a remediated current head."""
    if row["status"] != "active" or int(row["is_current"] or 0) != 1:
        return False
    if row["superseded_by"] or row["currency_status"] != "ACTIVE":
        return False
    if row["concept_type"] not in CANONICAL_CURRENT_TYPES:
        return False

    data = _loads(row["data"])
    history = data.get("lifecycle_remediation_history")
    has_remediation_history = isinstance(history, list) and bool(history)
    has_remediation_marker = data.get("change_type") == "lifecycle_remediation"
    has_correction = bool(data.get("correction_evidence") or data.get("correction_supersession"))
    has_remediation_provenance = has_remediation_marker or has_remediation_history
    return bool(data.get("currency_status") == "ACTIVE" and has_correction and has_remediation_provenance)


def apply_canonical_current_remediation(
    conn: sqlite3.Connection,
    concept_id: str,
    *,
    required_summary_substrings: tuple[str, ...],
    reason: str,
    dry_run: bool = True,
) -> LifecycleRemediationDecision:
    """Dry-run or apply canonical-current contested remediation."""
    decision = evaluate_canonical_current_contested(
        conn,
        concept_id,
        required_summary_substrings=required_summary_substrings,
    )
    if not decision.eligible or dry_run:
        return decision

    changed = resolve_contested_current_head_conn(conn, concept_id, reason=reason)
    if changed != 1:
        return LifecycleRemediationDecision(concept_id, False, "mutation_not_applied")

    conn.execute(
        """INSERT INTO governance_events (event_type, concept_id, details, created_at)
           VALUES (?, ?, ?, ?)""",
        (
            "lifecycle_remediation_current_head_resolved",
            concept_id,
            json.dumps(
                {
                    "reason": reason,
                    "prior_currency_status": decision.prior_currency_status,
                    "remediation": "resolve_contested_current_head",
                },
                sort_keys=True,
            ),
            _utc_now_iso(),
        ),
    )
    return LifecycleRemediationDecision(concept_id, True, "remediated", mode="apply")
