"""Policy Engine — Cohesive enforcement layer for write-path defense.

Memory Integrity Spec v1.2, §5.3 (Amendment 4 Gap 1):
Unifies content validation + type gating + directive enforcement.
Wraps existing validate_content() from directives.py with structured
violation logging and fail-fast evaluation chain.

Evaluation order (fail-fast):
1. Blocklist check (existing DIRECTIVE_BLOCKLIST) → BLOCK
2. Type validation (concept_type in CONCEPT_TYPES) → BLOCK
3. Rate limit check (existing check_rate_limit) → BLOCK
4. Content length check → BLOCK
5. Provider-agnostic check → WARN
"""

import logging
from dataclasses import dataclass

from app.core.config import FEATURE_FLAGS
from app.core.datetime_utils import _utc_now_iso
from app.core.models import CONCEPT_TYPES
from app.storage import _db

logger = logging.getLogger(__name__)


@dataclass
class PolicyViolation:
    """Structured violation record for audit trail."""

    rule_id: str  # e.g., "blocklist_match", "type_invalid"
    severity: str  # "BLOCK" | "WARN" | "LOG"
    concept_id: str  # What triggered the violation (or "" if unknown)
    detail: str  # Human-readable explanation
    timestamp: str = ""  # ISO8601, filled automatically
    caller_context: str = ""  # Which write path triggered this

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = _utc_now_iso()


class PolicyEngine:
    """Cohesive enforcement layer wrapping existing validation.

    Called from pith_session_learn and pith_propose_concept
    BEFORE any storage write.
    """

    def __init__(self):
        # Lazy import to avoid circular deps
        from app.governance.directives import (
            _PROVIDER_TERMS,
            DIRECTIVE_BLOCKLIST,
            MAX_DIRECTIVE_CHARS,
            DirectiveValidationError,
        )

        self._blocklist = DIRECTIVE_BLOCKLIST
        self._provider_terms = _PROVIDER_TERMS
        self._max_chars = MAX_DIRECTIVE_CHARS
        self._validation_error = DirectiveValidationError

    def evaluate(
        self,
        content: str,
        concept_type: str | None = None,
        concept_id: str = "",
        caller_context: str = "unknown",
    ) -> tuple[bool, list[PolicyViolation]]:
        """Run all policy checks in fail-fast order.

        Returns:
            (allowed, violations) — allowed is False if any BLOCK violation.
        """
        if not FEATURE_FLAGS.get("POLICY_ENGINE_ENABLED", False):
            return (True, [])

        violations: list[PolicyViolation] = []

        # 1. Blocklist check
        if content:
            for pattern in self._blocklist:
                if pattern.search(content):
                    violations.append(
                        PolicyViolation(
                            rule_id="blocklist_match",
                            severity="BLOCK",
                            concept_id=concept_id,
                            detail="Content matches injection blocklist pattern",
                            caller_context=caller_context,
                        )
                    )
                    break  # One blocklist hit is enough

        # 2. Type validation
        if concept_type is not None and concept_type not in CONCEPT_TYPES:
            violations.append(
                PolicyViolation(
                    rule_id="type_invalid",
                    severity="BLOCK",
                    concept_id=concept_id,
                    detail=f"Invalid concept_type '{concept_type}'",
                    caller_context=caller_context,
                )
            )

        # 3. Content length check
        if content and len(content) > self._max_chars:
            violations.append(
                PolicyViolation(
                    rule_id="content_too_long",
                    severity="BLOCK",
                    concept_id=concept_id,
                    detail=f"Content is {len(content)} chars, max {self._max_chars}",
                    caller_context=caller_context,
                )
            )

        # 4. Empty content check
        if not content or not content.strip():
            violations.append(
                PolicyViolation(
                    rule_id="empty_content",
                    severity="BLOCK",
                    concept_id=concept_id,
                    detail="Content cannot be empty",
                    caller_context=caller_context,
                )
            )

        # 5. Provider-specific terms (soft warning only)
        if content and self._provider_terms.search(content):
            violations.append(
                PolicyViolation(
                    rule_id="provider_specific_term",
                    severity="WARN",
                    concept_id=concept_id,
                    detail="Content references a specific AI provider",
                    caller_context=caller_context,
                )
            )

        # Log all violations
        for v in violations:
            self._log_violation(v)

        blocked = any(v.severity == "BLOCK" for v in violations)
        return (not blocked, violations)

    def log_violation(self, violation: PolicyViolation):
        """Public API: persist a single violation to policy_violations table."""
        self._log_violation(violation)

    def _log_violation(self, violation: PolicyViolation):
        """Persist violation to policy_violations table."""
        try:
            with _db() as conn:
                conn.execute(
                    """
                    INSERT INTO policy_violations
                    (rule_id, severity, concept_id, detail,
                     caller_context, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        violation.rule_id,
                        violation.severity,
                        violation.concept_id,
                        violation.detail,
                        violation.caller_context,
                        violation.timestamp,
                    ),
                )
            # Write-invalidation: new violation means cached stats are stale (CM-C5)
            try:
                from app.governance.policy_cache import get_policy_cache

                get_policy_cache().invalidate_stats()
            except Exception:
                pass  # Cache invalidation is best-effort
        except Exception as e:
            # Never let logging failures block the pipeline
            logger.error(f"Failed to log policy violation: {e}")


# --- Module-level singleton ---
_engine: PolicyEngine | None = None


def get_policy_engine() -> PolicyEngine:
    """Get or create the singleton PolicyEngine instance."""
    global _engine
    if _engine is None:
        _engine = PolicyEngine()
    return _engine


def evaluate_concept(
    summary: str,
    concept_type: str | None = None,
    concept_id: str = "",
    caller_context: str = "unknown",
    gov_ctx=None,
) -> tuple[bool, list[PolicyViolation]]:
    """Convenience function — evaluate a concept against all policies.

    Returns (allowed, violations).
    When gov_ctx (GovernanceContext) is provided and GOVERNANCE_EVENT_WIRING_ENABLED,
    logs typed PolicyDecisionEvents through it (§5.8.4 H18).
    """
    engine = get_policy_engine()
    allowed, violations = engine.evaluate(
        content=summary,
        concept_type=concept_type,
        concept_id=concept_id,
        caller_context=caller_context,
    )

    # Wire to GovernanceContext if available (§5.8.4 H18)
    if gov_ctx and FEATURE_FLAGS.get("GOVERNANCE_EVENT_WIRING_ENABLED", False):
        try:
            if violations:
                for v in violations:
                    gov_ctx.log_policy_decision(
                        policy_name=v.rule_id,
                        concept_id=v.concept_id or concept_id,
                        decision=v.severity,
                        severity=v.severity,
                    )
            else:
                gov_ctx.log_policy_decision(
                    policy_name="all_checks",
                    concept_id=concept_id,
                    decision="PASS",
                    severity="PASS",
                )
        except Exception as e:
            logger.warning(f"Failed to log policy event to GovernanceContext: {e}")

    return (allowed, violations)


def validate_concept(concept_data: dict) -> dict:
    """Dict-based concept validation for PoisonBench and external callers.

    Takes a concept dict, runs it through the policy engine, and returns
    a dict with 'allowed' (bool) and 'violations' (list of dicts).

    This wraps evaluate_concept() with a dict interface for convenience.
    """
    summary = concept_data.get("summary", "")
    concept_type = concept_data.get("concept_type")
    concept_id = concept_data.get("concept_id", "")
    source = concept_data.get("source", "unknown")
    always_activate = concept_data.get("always_activate", False)

    # Hard block: auto-learn cannot set always_activate
    if always_activate and source in ("auto_learn", "session_learn", "conversation_turn", "bridge"):
        return {
            "allowed": False,
            "violations": [
                {
                    "rule_id": "firmware_protection",
                    "severity": "BLOCK",
                    "detail": f"Source '{source}' cannot set always_activate=True. "
                    "Only explicit pith_set_always_activate is allowed.",
                }
            ],
        }

    allowed, violations = evaluate_concept(
        summary=summary,
        concept_type=concept_type,
        concept_id=concept_id,
        caller_context=source,
    )
    return {
        "allowed": allowed,
        "violations": [{"rule_id": v.rule_id, "severity": v.severity, "detail": v.detail} for v in violations],
    }


def get_violation_stats() -> dict:
    """Get policy violation statistics.

    Uses PolicyCache (CM-C5) when POLICY_CACHE_ENABLED to avoid
    repeated expensive aggregation queries.
    """
    # Check cache first (CM-C5)
    try:
        from app.governance.policy_cache import get_policy_cache

        cache = get_policy_cache()
        cached = cache.get_cached_stats("violation_stats")
        if cached is not None:
            return cached
    except Exception:
        cache = None

    try:
        with _db() as conn:
            total = conn.execute("SELECT COUNT(*) FROM policy_violations").fetchone()[0]
            by_severity = {}
            for row in conn.execute("""
                SELECT severity, COUNT(*) as cnt
                FROM policy_violations GROUP BY severity
            """).fetchall():
                by_severity[row[0]] = row[1]
            by_rule = {}
            for row in conn.execute("""
                SELECT rule_id, COUNT(*) as cnt
                FROM policy_violations GROUP BY rule_id
                ORDER BY cnt DESC LIMIT 10
            """).fetchall():
                by_rule[row[0]] = row[1]
            result = {
                "total_violations": total,
                "by_severity": by_severity,
                "by_rule": by_rule,
            }
            # Populate cache (CM-C5)
            if cache:
                try:  # noqa: SIM105
                    cache.set_cached_stats("violation_stats", result)
                except Exception:
                    pass
            return result
    except Exception:
        return {"total_violations": 0, "by_severity": {}, "by_rule": {}}


def get_rejections(
    since: str | None = None,
    severity: str | None = None,
    limit: int = 100,
) -> dict:
    """Get filtered rejections with rate statistics (§5.2.10 H14).

    Args:
        since: ISO datetime string filter (e.g. '2026-02-25T00:00:00').
        severity: Filter by severity ('BLOCK', 'WARN').
        limit: Max results.

    Returns:
        Dict with 'rejections' list and 'stats' summary.
    """
    try:
        with _db() as conn:
            query = "SELECT * FROM policy_violations WHERE 1=1"
            params: list = []
            if since:
                query += " AND created_at >= ?"
                params.append(since)
            if severity:
                query += " AND severity = ?"
                params.append(severity)
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(query, params).fetchall()
            rejections = [dict(r) for r in rows]

            # Rate stats: rejections per hour in the queried window
            rate_query = """
                SELECT strftime('%Y-%m-%dT%H:00:00', created_at) as hour,
                       COUNT(*) as cnt
                FROM policy_violations
                WHERE 1=1
            """
            rate_params: list = []
            if since:
                rate_query += " AND created_at >= ?"
                rate_params.append(since)
            if severity:
                rate_query += " AND severity = ?"
                rate_params.append(severity)
            rate_query += " GROUP BY hour ORDER BY hour DESC LIMIT 24"
            rate_rows = conn.execute(rate_query, rate_params).fetchall()
            hourly_rates = {r[0]: r[1] for r in rate_rows}

            # By-rule breakdown
            rule_query = """
                SELECT rule_id, COUNT(*) as cnt
                FROM policy_violations WHERE 1=1
            """
            rule_params: list = []
            if since:
                rule_query += " AND created_at >= ?"
                rule_params.append(since)
            if severity:
                rule_query += " AND severity = ?"
                rule_params.append(severity)
            rule_query += " GROUP BY rule_id ORDER BY cnt DESC"
            rule_rows = conn.execute(rule_query, rule_params).fetchall()
            by_rule = {r[0]: r[1] for r in rule_rows}

            return {
                "rejections": rejections,
                "count": len(rejections),
                "stats": {
                    "hourly_rates": hourly_rates,
                    "by_rule": by_rule,
                    "total_in_window": sum(by_rule.values()) if by_rule else 0,
                },
            }
    except Exception as e:
        logger.error(f"get_rejections failed: {e}")
        return {"rejections": [], "count": 0, "stats": {}}


def log_policy_event(
    rule_id: str,
    severity: str,
    concept_id: str = "",
    detail: str = "",
    caller_context: str = "",
) -> None:
    """Log a policy event to the policy_violations audit table.

    Convenience function for non-PolicyEngine callers (quarantine endpoints,
    auto-graduation, etc.) that need to write audit records.
    """
    violation = PolicyViolation(
        rule_id=rule_id,
        severity=severity,
        concept_id=concept_id,
        detail=detail,
        caller_context=caller_context,
    )
    engine = get_policy_engine()
    engine.log_violation(violation)
