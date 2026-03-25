"""Firmware Deprecation Policy — CM-M5, Memory Integrity Spec v1.2.

TODO(cleanup-sprint-2026-03-02): This module has 298 lines, 11+ test references
in test_phase2_temporal_epistemic.py, but ZERO production callers. KEEP decision
made during cleanup audit — has test coverage and doesn't cause harm. Revisit
for removal if firmware concept is fully deprecated in a future phase.

Manages the lifecycle of always_activate (firmware) concepts:
  - Detects stale firmware (not accessed in FIRMWARE_STALE_DAYS)
  - Deprecates firmware with audit trail
  - Blocks auto-learn from setting always_activate (D6.1 defense)
  - Blocks evolution of firmware concepts without elevated verification (D6.2 defense)
  - Enforces FIRMWARE_MAX_ACTIVE cap

Feature-gated on FIRMWARE_DEPRECATION_ENABLED (default: False).
"""

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from app.config import (
    FEATURE_FLAGS,
    FIRMWARE_MAX_ACTIVE,
    FIRMWARE_PROTECTED_PREFIXES,
    FIRMWARE_STALE_DAYS,
)
from app.constants import GOV_EVENT_FIRMWARE_DEPRECATED
from app.datetime_utils import _utc_now, _utc_now_iso

logger = logging.getLogger(__name__)


@dataclass
class FirmwareHealthReport:
    """Result of scanning all firmware concepts for health issues."""

    total_firmware: int = 0
    active_firmware: int = 0
    stale_firmware: list[dict[str, Any]] = field(default_factory=list)
    over_cap: bool = False
    deprecation_candidates: list[str] = field(default_factory=list)


@dataclass
class DeprecationResult:
    """Result of a firmware deprecation action."""

    deprecated: bool = False
    concept_id: str = ""
    reason: str = ""
    previous_state: dict[str, Any] | None = None


def is_firmware_protected(concept_id: str) -> bool:
    """Check if a concept ID is protected from deprecation.

    Protected concepts include system firmware entries (firmware: prefix)
    which are managed by seed_firmware.py, not user-created.
    """
    return any(concept_id.startswith(prefix) for prefix in FIRMWARE_PROTECTED_PREFIXES)


def check_firmware_health(conn: sqlite3.Connection) -> FirmwareHealthReport:
    """Scan all always_activate concepts and report health status.

    Returns:
        FirmwareHealthReport with stale concepts and deprecation candidates.
    """
    if not FEATURE_FLAGS.get("FIRMWARE_DEPRECATION_ENABLED", False):
        return FirmwareHealthReport()

    report = FirmwareHealthReport()
    cutoff = (_utc_now() - timedelta(days=FIRMWARE_STALE_DAYS)).isoformat()

    # Get all always_activate concepts
    rows = conn.execute(
        "SELECT id, summary, last_accessed, created_at, knowledge_area FROM concepts WHERE always_activate = 1"
    ).fetchall()

    report.total_firmware = len(rows)
    report.active_firmware = len(rows)

    for row in rows:
        concept_id = row[0]
        last_accessed = row[2]  # may be None

        # Check staleness: no access or last access before cutoff
        is_stale = False
        if last_accessed is None:
            # Never accessed — check if created before cutoff
            created_at = row[3]
            if created_at and created_at < cutoff:
                is_stale = True
        elif last_accessed < cutoff:
            is_stale = True

        if is_stale:
            stale_entry = {
                "concept_id": concept_id,
                "summary": row[1],
                "last_accessed": last_accessed,
                "created_at": row[3],
                "knowledge_area": row[4],
                "protected": is_firmware_protected(concept_id),
            }
            report.stale_firmware.append(stale_entry)

            # Only non-protected stale concepts are deprecation candidates
            if not is_firmware_protected(concept_id):
                report.deprecation_candidates.append(concept_id)

    report.over_cap = report.total_firmware > FIRMWARE_MAX_ACTIVE
    return report


def deprecate_firmware(
    concept_id: str,
    reason: str,
    conn: sqlite3.Connection,
    force: bool = False,
) -> DeprecationResult:
    """Remove always_activate flag from a concept with audit trail.

    Args:
        concept_id: The concept to deprecate.
        reason: Human-readable reason for deprecation.
        conn: SQLite connection.
        force: If True, bypass protection checks (for admin use).

    Returns:
        DeprecationResult with success status and details.
    """
    if not FEATURE_FLAGS.get("FIRMWARE_DEPRECATION_ENABLED", False):
        return DeprecationResult(
            deprecated=False,
            concept_id=concept_id,
            reason="FIRMWARE_DEPRECATION_ENABLED is False",
        )

    # Protection check
    if not force and is_firmware_protected(concept_id):
        return DeprecationResult(
            deprecated=False,
            concept_id=concept_id,
            reason=f"Concept {concept_id} is protected (firmware: prefix)",
        )

    # Verify concept exists and is firmware
    row = conn.execute(
        "SELECT id, summary, always_activate, last_accessed, knowledge_area FROM concepts WHERE id = ?",
        (concept_id,),
    ).fetchone()

    if not row:
        return DeprecationResult(
            deprecated=False,
            concept_id=concept_id,
            reason=f"Concept {concept_id} not found",
        )

    if not row[2]:  # always_activate is 0
        return DeprecationResult(
            deprecated=False,
            concept_id=concept_id,
            reason=f"Concept {concept_id} is not firmware (always_activate=0)",
        )

    previous_state = {
        "concept_id": row[0],
        "summary": row[1],
        "always_activate": row[2],
        "last_accessed": row[3],
        "knowledge_area": row[4],
    }

    # Remove always_activate flag
    now = _utc_now_iso()
    conn.execute(
        "UPDATE concepts SET always_activate = 0, updated_at = ? WHERE id = ?",
        (now, concept_id),
    )

    # Log deprecation event to governance_events if table exists
    try:
        conn.execute(
            "INSERT INTO governance_events (event_type, concept_id, details, created_at) VALUES (?, ?, ?, ?)",
            (
                GOV_EVENT_FIRMWARE_DEPRECATED,
                concept_id,
                reason,
                now,
            ),
        )
    except sqlite3.OperationalError:
        # governance_events table may not exist yet
        logger.debug("governance_events table not available for deprecation audit")

    conn.commit()

    logger.info("Firmware deprecated: %s (reason: %s)", concept_id, reason)

    return DeprecationResult(
        deprecated=True,
        concept_id=concept_id,
        reason=reason,
        previous_state=previous_state,
    )


def validate_firmware_write(
    concept_id: str,
    always_activate: bool,
    source: str,
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    """Validate whether a write operation can set always_activate.

    D6.1 defense: Only explicit pith_set_always_activate or pith_propose_concept
    with the flag can set always_activate. Auto-learn (session_learn) is blocked.

    D6.2 defense: Evolution of existing firmware concepts requires the source
    to be 'explicit' (pith_propose_concept, pith_set_always_activate, admin).

    Args:
        concept_id: Target concept ID.
        always_activate: Whether the write wants to set always_activate=True.
        source: Origin of the write ('session_learn', 'propose_concept',
                'set_always_activate', 'admin').
        conn: Optional connection for checking existing firmware state.

    Returns:
        Dict with 'allowed' (bool) and 'reason' (str).
    """
    if not FEATURE_FLAGS.get("FIRMWARE_DEPRECATION_ENABLED", False):
        return {"allowed": True, "reason": "firmware policy disabled"}

    ELEVATED_SOURCES = {"propose_concept", "set_always_activate", "admin"}

    # D6.1: Block auto-learn from setting always_activate
    if always_activate and source not in ELEVATED_SOURCES:
        return {
            "allowed": False,
            "reason": f"Source '{source}' cannot set always_activate. Only {ELEVATED_SOURCES} are permitted.",
        }

    # D6.2: Check if this is an evolution of existing firmware
    if conn is not None:
        row = conn.execute(
            "SELECT always_activate FROM concepts WHERE id = ?",
            (concept_id,),
        ).fetchone()

        if row and row[0] == 1:
            # Existing firmware — require elevated source for any modification
            if source not in ELEVATED_SOURCES:
                return {
                    "allowed": False,
                    "reason": f"Firmware concept {concept_id} requires elevated "
                    f"verification for modification. Source '{source}' "
                    f"is not permitted.",
                }

    return {"allowed": True, "reason": "firmware write validated"}


def run_firmware_deprecation_scan(conn: sqlite3.Connection) -> dict[str, Any]:
    """Run a full firmware deprecation scan — intended for maintenance jobs.

    Checks health, auto-deprecates stale non-protected firmware, and returns
    a summary of actions taken.
    """
    if not FEATURE_FLAGS.get("FIRMWARE_DEPRECATION_ENABLED", False):
        return {"skipped": True, "reason": "FIRMWARE_DEPRECATION_ENABLED is False"}

    report = check_firmware_health(conn)
    results = {
        "total_firmware": report.total_firmware,
        "stale_count": len(report.stale_firmware),
        "deprecation_candidates": len(report.deprecation_candidates),
        "over_cap": report.over_cap,
        "deprecated": [],
        "errors": [],
    }

    for candidate_id in report.deprecation_candidates:
        try:
            result = deprecate_firmware(
                concept_id=candidate_id,
                reason=f"Auto-deprecated: not accessed in {FIRMWARE_STALE_DAYS}+ days",
                conn=conn,
            )
            if result.deprecated:
                results["deprecated"].append(candidate_id)
        except Exception as e:
            results["errors"].append({"concept_id": candidate_id, "error": str(e)})

    return results
