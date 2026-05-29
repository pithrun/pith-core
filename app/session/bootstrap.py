"""Functional Cognitive Bootstrap — governance-aware session initialization.

Loads decisions as behavioral constraints, surfaces stale alerts, reports
governance actions taken since last session, and provides progressive
disclosure of cognitive state.

Replaces the current orientation-only bootstrap with full governance context.
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Any

from app.authority import format_concept_with_authority
from app.core.config import (
    BOOTSTRAP_DECISION_AUTHORITY_MIN,
    PIN_BUDGET,
    PRESENTATION_CONSTRAINT,
)
from app.core.constants import GOV_EVENT_BOOTSTRAP_COMPLETE
from app.governance.currency import _days_since
from app.core.datetime_utils import _utc_now
from app.governance.governance_context import GovernanceContext

logger = logging.getLogger(__name__)

_BOOTSTRAP_SELECT = """SELECT id, summary, concept_type, authority_score, currency_score,
                              currency_status, stability, updated_at, data, confidence,
                              content_updated_at
                       FROM concepts
                       WHERE status != 'deleted'
                         AND (currency_status IS NULL OR currency_status != 'SUPERSEDED')"""


class ConstraintDirective:
    """A high-authority concept loaded as a behavioral constraint."""

    def __init__(
        self,
        concept_id: str,
        summary: str,
        authority: float,
        currency: float,
        concept_type: str,
        qualifiers: list[str] = None,
    ):
        self.concept_id = concept_id
        self.summary = summary
        self.authority = authority
        self.currency = currency
        self.concept_type = concept_type
        self.qualifiers = qualifiers or []

    def format(self) -> str:
        return format_concept_with_authority(self.summary, self.authority, self.qualifiers)

    def to_dict(self) -> dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "summary": self.summary,
            "authority": self.authority,
            "currency": self.currency,
            "concept_type": self.concept_type,
            "qualifiers": self.qualifiers,
            "formatted": self.format(),
        }


class StaleAlert:
    """Alert for a high-authority concept that is losing currency."""

    def __init__(self, concept_id: str, summary: str, authority: float, currency: float, days_since_update: float):
        self.concept_id = concept_id
        self.summary = summary
        self.authority = authority
        self.currency = currency
        self.days_since_update = days_since_update

    def to_dict(self) -> dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "summary": self.summary,
            "authority": self.authority,
            "currency": self.currency,
            "days_since_update": round(self.days_since_update, 1),
            "message": f"High-authority concept aging: '{self.summary[:50]}...' "
            f"(authority={self.authority:.2f}, currency={self.currency:.2f})",
        }


class StaleCheckpointAlert:
    """Alert for a checkpoint that hasn't been updated recently."""

    def __init__(self, task_id: str, status: str, days_stale: int, description: str):
        self.task_id = task_id
        self.status = status
        self.days_stale = days_stale
        self.description = description

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "stale_checkpoint",
            "task_id": self.task_id,
            "status": self.status,
            "days_stale": self.days_stale,
            "description": self.description,
            "message": f"Checkpoint '{self.task_id}' not updated in {self.days_stale}d (status={self.status})",
        }


class BootstrapResult:
    """Complete bootstrap result for session initialization."""

    def __init__(self):
        self.session_id: str = ""
        self.is_resumption: bool = False
        self.active_constraints: list[ConstraintDirective] = []
        self.active_decisions: list[ConstraintDirective] = []
        self.active_goals: list[ConstraintDirective] = []
        self.stale_alerts: list[StaleAlert] = []
        self.stale_checkpoint_alerts: list[StaleCheckpointAlert] = []
        self.active_skills: list[str] = []
        self.governance_actions_pending: list[str] = []
        self.recent_work_summary: str = ""
        self.open_threads: list[str] = []
        self.constraints_loaded: int = 0
        self.decisions_loaded: int = 0
        self.total_candidates: int = 0

    @property
    def coverage_ratio(self) -> float:
        loaded = self.constraints_loaded + self.decisions_loaded
        if self.total_candidates == 0:
            return 1.0
        return round(loaded / self.total_candidates, 3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "is_resumption": self.is_resumption,
            "active_constraints": [c.to_dict() for c in self.active_constraints],
            "active_decisions": [c.to_dict() for c in self.active_decisions],
            "active_goals": [c.to_dict() for c in self.active_goals],
            "stale_alerts": [a.to_dict() for a in self.stale_alerts],
            "stale_checkpoint_alerts": [a.to_dict() for a in self.stale_checkpoint_alerts],
            "active_skills": self.active_skills,
            "governance_actions_pending": self.governance_actions_pending,
            "recent_work_summary": self.recent_work_summary,
            "open_threads": self.open_threads,
            "constraints_loaded": self.constraints_loaded,
            "decisions_loaded": self.decisions_loaded,
            "total_candidates": self.total_candidates,
            "coverage_ratio": self.coverage_ratio,
        }


def _get_qualifiers(authority: float, currency: float, evidence_count: int, days_since_evidence: float) -> list[str]:
    """Generate uncertainty qualifiers for a concept."""
    qualifiers = []
    if days_since_evidence > 30:
        qualifiers.append("EVIDENCE_AGING")
    if evidence_count < 2:
        qualifiers.append("LOW_CORROBORATION")
    if authority >= 0.60 and currency < 0.50:
        qualifiers.append("STALE_RISK")
    if authority >= 0.80 and currency >= 0.80 and evidence_count >= 3:
        qualifiers.append("HIGH_CONFIDENCE")
    return qualifiers


def _count_evidence(data_json: str) -> int:
    try:
        data = json.loads(data_json) if data_json else {}
        return len(data.get("evidence", []))
    except (json.JSONDecodeError, TypeError):
        return 0


def _newest_evidence_days(data_json: str) -> float:
    try:
        data = json.loads(data_json) if data_json else {}
        evidence = data.get("evidence", [])
        if not evidence:
            # DATA-034: Prefer content_updated_at (true content age) over updated_at (migration-inflated)
            return _days_since(data.get("content_updated_at") or data.get("updated_at"))
        min_days = 365.0
        for ev in evidence:
            if isinstance(ev, dict) and ev.get("timestamp"):
                min_days = min(min_days, _days_since(ev["timestamp"]))
        return min_days
    except (json.JSONDecodeError, TypeError):
        return 365.0


def build_bootstrap(
    conn: sqlite3.Connection,
    session_id: str,
    is_resumption: bool = False,
    last_session_ended_at: str | None = None,
    gov_ctx: GovernanceContext | None = None,
) -> BootstrapResult:
    """Build governance-aware bootstrap for session start.

    Loads high-authority constraints and decisions, detects stale alerts,
    and retrieves pending governance actions since last session.

    Args:
        conn: SQLite connection
        session_id: Current session ID
        is_resumption: Whether this is resuming a previous session
        last_session_ended_at: When the user's last session ended (for governance actions)
        gov_ctx: Optional GovernanceContext to populate

    Returns:
        BootstrapResult with full governance context for bootstrap
    """
    result = BootstrapResult()
    result.session_id = session_id
    result.is_resumption = is_resumption

    rows_by_id: dict[str, sqlite3.Row | tuple[Any, ...]] = {}

    def add_candidate_rows(sql_suffix: str, params: tuple[Any, ...], limit: int) -> None:
        rows = conn.execute(f"{_BOOTSTRAP_SELECT} {sql_suffix} LIMIT ?", (*params, limit)).fetchall()
        for candidate in rows:
            rows_by_id[candidate[0]] = candidate

    # Avoid a full-brain scan during session_start. Each slice is bounded to the
    # fields the bootstrap actually surfaces.
    add_candidate_rows(
        """AND COALESCE(authority_score, confidence, 0) >= ?
           ORDER BY COALESCE(authority_score, confidence, 0) DESC""",
        (PRESENTATION_CONSTRAINT,),
        PIN_BUDGET,
    )
    add_candidate_rows(
        """AND COALESCE(authority_score, confidence, 0) >= ?
           AND COALESCE(currency_score, 1.0) >= 0.50
           AND concept_type IN ('decision', 'constraint', 'principle', 'method')
           ORDER BY COALESCE(authority_score, confidence, 0) DESC""",
        (BOOTSTRAP_DECISION_AUTHORITY_MIN,),
        PIN_BUDGET * 4,
    )
    add_candidate_rows(
        """AND COALESCE(currency_score, 1.0) >= 0.50
           AND concept_type = 'goal'
           ORDER BY COALESCE(authority_score, confidence, 0) DESC""",
        (),
        5,
    )
    add_candidate_rows(
        """AND COALESCE(authority_score, confidence, 0) >= 0.60
           AND COALESCE(currency_score, 1.0) < 0.50
           ORDER BY COALESCE(authority_score, confidence, 0) DESC""",
        (),
        3,
    )

    rows = sorted(
        rows_by_id.values(),
        key=lambda row: row[3] if row[3] is not None else (row[9] if row[9] is not None else 0.0),
        reverse=True,
    )
    result.total_candidates = len(rows)
    _loaded_ids = set()  # Track IDs already loaded to prevent double-counting

    for row in rows:
        cid, summary, ctype, auth, curr, curr_status, stab, updated_at, data_json, conf, content_updated_at = row
        # Authority fallback: use confidence when authority_score not yet batch-computed
        auth = auth if auth is not None else (conf if conf is not None else 0.0)
        curr = curr or 1.0
        ctype = ctype or "observation"

        ev_count = _count_evidence(data_json)
        ev_days = _newest_evidence_days(data_json)
        qualifiers = _get_qualifiers(auth, curr, ev_count, ev_days)

        # CONTESTED qualifier
        if curr_status == "CONTESTED":
            qualifiers.append("CONTESTED")

        directive = ConstraintDirective(
            concept_id=cid,
            summary=summary or "",
            authority=auth,
            currency=curr,
            concept_type=ctype,
            qualifiers=qualifiers,
        )

        # Constraints: authority >= 0.80
        if auth >= PRESENTATION_CONSTRAINT and result.constraints_loaded < PIN_BUDGET:
            result.active_constraints.append(directive)
            result.constraints_loaded += 1
            _loaded_ids.add(cid)

        # Decisions/directives: authority >= 0.50 with decent currency (§5.2)
        elif auth >= BOOTSTRAP_DECISION_AUTHORITY_MIN and curr >= 0.50 and result.decisions_loaded < PIN_BUDGET:
            if ctype in ("decision", "constraint", "principle", "method"):
                result.active_decisions.append(directive)
                result.decisions_loaded += 1
                _loaded_ids.add(cid)

        # Active goals (avoid double-counting concepts already in constraints/decisions)
        if ctype == "goal" and curr >= 0.50 and cid not in _loaded_ids:
            if len(result.active_goals) < 5:
                result.active_goals.append(directive)

        # Stale alerts: high authority but currency dropping
        if auth >= 0.60 and curr < 0.50 and curr_status != "SUPERSEDED":
            days = _days_since(content_updated_at or updated_at)  # DATA-034
            result.stale_alerts.append(
                StaleAlert(
                    concept_id=cid,
                    summary=summary or "",
                    authority=auth,
                    currency=curr,
                    days_since_update=days,
                )
            )

    # MONITOR-021: Cap stale alerts at 3, prioritize by authority (highest first)
    result.stale_alerts.sort(key=lambda a: a.authority, reverse=True)
    result.stale_alerts = result.stale_alerts[:3]

    # --- Checkpoint staleness alerts ---
    try:
        from app.storage import list_checkpoints

        now_dt = _utc_now()
        stale_threshold = timedelta(hours=48)
        checkpoints = list_checkpoints()

        for cp in checkpoints:
            if cp.get("status") in ("active", "planning", "paused", "blocked"):
                try:
                    updated = datetime.fromisoformat(cp["updated_at"])
                    days_stale = (now_dt - updated).days
                    if (now_dt - updated) > stale_threshold:
                        result.stale_checkpoint_alerts.append(
                            StaleCheckpointAlert(
                                task_id=cp["task_id"],
                                status=cp["status"],
                                days_stale=days_stale,
                                description=cp.get("description", "")[:80],
                            )
                        )
                except (ValueError, TypeError):
                    pass

        # MONITOR-021: Cap at 3 to prevent alert flood
        result.stale_checkpoint_alerts = result.stale_checkpoint_alerts[:3]
    except Exception as e:
        logger.debug(f"Bootstrap checkpoint alerts skipped: {e}")

    # Governance actions since last session
    if last_session_ended_at:
        result.governance_actions_pending = _get_governance_actions_since(conn, last_session_ended_at)

    # Populate governance context if provided
    if gov_ctx:
        gov_ctx.bootstrap_constraints_loaded = result.constraints_loaded
        gov_ctx.governance_actions_pending = result.governance_actions_pending
        gov_ctx.log_event(
            GOV_EVENT_BOOTSTRAP_COMPLETE,
            None,
            {
                "constraints_loaded": result.constraints_loaded,
                "decisions_loaded": result.decisions_loaded,
                "stale_alerts": len(result.stale_alerts),
                "governance_actions": len(result.governance_actions_pending),
                "coverage_ratio": result.coverage_ratio,
            },
        )

    logger.info(
        "Bootstrap: %d constraints, %d decisions, %d stale alerts, %d governance actions",
        result.constraints_loaded,
        result.decisions_loaded,
        len(result.stale_alerts),
        len(result.governance_actions_pending),
    )

    return result


def _get_governance_actions_since(conn: sqlite3.Connection, since_timestamp: str) -> list[str]:
    """Query governance_events for automated actions since last session."""
    try:
        rows = conn.execute(
            """SELECT event_type, concept_id, details
               FROM governance_events
               WHERE created_at > ?
                 AND event_type IN ('concept_promoted', 'concept_demoted',
                                    'cko_archived', 'skill_deprecated',
                                    'recalibration_applied')
               ORDER BY created_at DESC
               LIMIT 20""",
            (since_timestamp,),
        ).fetchall()

        actions = []
        for row in rows:
            etype = row[0]
            cid = row[1] or "system"
            try:
                details = json.loads(row[2]) if row[2] else {}
            except (json.JSONDecodeError, TypeError):
                details = {}
            reason = details.get("reason", "")
            actions.append(f"{etype}: {cid}" + (f" ({reason})" if reason else ""))

        return actions
    except sqlite3.OperationalError:
        return []  # governance_events table doesn't exist yet
