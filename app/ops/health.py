"""Governance Health Check & Circuit Breaker — 5 indicators, auto-trip, auto-reset.

Monitors governance system health with 5 indicators. If 2+ fail simultaneously,
the circuit breaker trips and all governance is bypassed in favor of raw
embedding retrieval. Auto-resets when failures drop below 2.

Health checks run every 5 minutes (configurable).
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from app.core.config import (
    CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    CIRCUIT_BREAKER_RECOVERY_INTERVAL_MINUTES,
    HEALTH_AUTHORITY_ZERO_THRESHOLD,
    HEALTH_CHECK_INTERVAL_MINUTES,
    HEALTH_CONTRADICTION_FP_RATE,
    HEALTH_GOVERNANCE_EVENT_OVERFLOW,
    HEALTH_RECALIBRATION_STALE_HOURS,
)
from app.core.datetime_utils import _ensure_aware, _utc_now, _utc_now_iso

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class HealthCheckResult:
    """Result from a single health indicator."""

    indicator: str
    healthy: bool
    detail: str
    checked_at: str = ""


@dataclass
class GovernanceHealthReport:
    """Full health report from all 5 indicators."""

    healthy: bool = True
    circuit_breaker_tripped: bool = False
    checks: list[HealthCheckResult] = field(default_factory=list)
    failure_count: int = 0
    checked_at: str = ""
    check_time_ms: float = 0.0


# =============================================================================
# Health Indicators
# =============================================================================


class GovernanceHealthCheck:
    """5 health indicators. If 2+ fire, circuit breaker trips.

    Each check method returns (healthy: bool, detail: str).
    """

    def __init__(self, conn=None):
        self._conn = conn

    def check_authority_distribution(self) -> tuple[bool, str]:
        """UNHEALTHY if >50% of concepts have authority_score == 0 or NULL."""
        if not self._conn:
            return (True, "no_connection")

        try:
            total_row = self._conn.execute("SELECT COUNT(*) FROM concepts WHERE status != 'deleted'").fetchone()
            total = total_row[0] if total_row else 0

            if total == 0:
                return (True, "no_concepts")

            zero_row = self._conn.execute(
                """SELECT COUNT(*) FROM concepts
                   WHERE status != 'deleted'
                   AND (authority_score = 0.0 OR authority_score IS NULL)"""
            ).fetchone()
            zero_count = zero_row[0] if zero_row else 0

            ratio = zero_count / total
            healthy = ratio < HEALTH_AUTHORITY_ZERO_THRESHOLD
            return (healthy, f"zero_authority_ratio={ratio:.2f} ({zero_count}/{total})")

        except Exception as e:
            return (True, f"check_error: {e}")

    def check_currency_freshness(self) -> tuple[bool, str]:
        """UNHEALTHY if last currency recompute was >72h ago."""
        if not self._conn:
            return (True, "no_connection")

        try:
            row = self._conn.execute(
                """SELECT MAX(last_currency_recompute) FROM concepts
                   WHERE last_currency_recompute IS NOT NULL"""
            ).fetchone()

            if not row or not row[0]:
                return (False, "no_currency_recompute_ever")

            last_recompute = datetime.fromisoformat(row[0])
            age_hours = (_utc_now() - _ensure_aware(last_recompute)).total_seconds() / 3600

            healthy = age_hours < HEALTH_RECALIBRATION_STALE_HOURS
            return (healthy, f"last_recompute={age_hours:.1f}h_ago")

        except Exception as e:
            return (True, f"check_error: {e}")

    def check_contradiction_fpr(self) -> tuple[bool, str]:
        """UNHEALTHY if contradiction false positive rate >30%.

        Reads from governance_events table. Needs at least 10 events
        to make a determination.
        """
        if not self._conn:
            return (True, "no_connection")

        try:
            rows = self._conn.execute(
                """SELECT event_type FROM governance_events
                   WHERE event_type IN ('contradiction_confirmed', 'contradiction_dismissed')
                   ORDER BY created_at DESC LIMIT 100"""
            ).fetchall()

            if len(rows) < 10:
                return (True, "insufficient_data")

            dismissed = sum(1 for r in rows if r[0] == "contradiction_dismissed")
            fpr = dismissed / len(rows)
            healthy = fpr < HEALTH_CONTRADICTION_FP_RATE
            return (healthy, f"contradiction_fpr={fpr:.2f} ({dismissed}/{len(rows)})")

        except Exception as e:
            return (True, f"check_error: {e}")

    def check_event_overflow(self) -> tuple[bool, str]:
        """UNHEALTHY if governance_events table exceeds overflow threshold."""
        if not self._conn:
            return (True, "no_connection")

        try:
            cutoff = (_utc_now() - timedelta(days=30)).isoformat()
            row = self._conn.execute(
                """SELECT COUNT(*) FROM governance_events
                   WHERE created_at > ?""",
                (cutoff,),
            ).fetchone()

            count = row[0] if row else 0
            healthy = count < HEALTH_GOVERNANCE_EVENT_OVERFLOW
            return (healthy, f"event_count_30d={count}")

        except Exception as e:
            return (True, f"check_error: {e}")

    def check_latency_compliance(self) -> tuple[bool, str]:
        """UNHEALTHY if >30% of recent governance phases exceeded budget."""
        if not self._conn:
            return (True, "no_connection")

        try:
            rows = self._conn.execute(
                """SELECT details FROM governance_events
                   WHERE event_type = 'PHASE_COMPLETED'
                   ORDER BY created_at DESC LIMIT 50"""
            ).fetchall()

            if len(rows) < 10:
                return (True, "insufficient_data")

            over_budget = 0
            for row in rows:
                try:
                    details = json.loads(row[0]) if row[0] else {}
                    remaining = details.get("remaining_ms", 0)
                    if remaining < 0:
                        over_budget += 1
                except (json.JSONDecodeError, TypeError):
                    continue

            ratio = over_budget / len(rows)
            healthy = ratio < 0.30
            return (healthy, f"over_budget_ratio={ratio:.2f} ({over_budget}/{len(rows)})")

        except Exception as e:
            return (True, f"check_error: {e}")

    def run_all(self) -> GovernanceHealthReport:
        """Run all 5 health indicators and determine circuit breaker state."""
        t0 = time.perf_counter()
        now = _utc_now_iso()

        indicators = [
            ("authority_distribution", self.check_authority_distribution),
            ("currency_freshness", self.check_currency_freshness),
            ("contradiction_fpr", self.check_contradiction_fpr),
            ("event_overflow", self.check_event_overflow),
            ("latency_compliance", self.check_latency_compliance),
        ]

        report = GovernanceHealthReport(checked_at=now)

        for name, check_fn in indicators:
            try:
                healthy, detail = check_fn()
            except Exception as e:
                healthy, detail = True, f"check_exception: {e}"

            result = HealthCheckResult(
                indicator=name,
                healthy=healthy,
                detail=detail,
                checked_at=now,
            )
            report.checks.append(result)

            if not healthy:
                report.failure_count += 1

        # Circuit breaker logic: 2+ failures = trip
        report.circuit_breaker_tripped = report.failure_count >= CIRCUIT_BREAKER_FAILURE_THRESHOLD
        report.healthy = not report.circuit_breaker_tripped
        report.check_time_ms = (time.perf_counter() - t0) * 1000

        if report.circuit_breaker_tripped:
            logger.warning(
                "CIRCUIT BREAKER TRIPPED: %d/%d health checks failed — bypassing governance. Failures: %s",
                report.failure_count,
                len(indicators),
                [c.detail for c in report.checks if not c.healthy],
            )
        else:
            logger.debug(
                "Health check: %d/%d healthy in %.1fms",
                len(indicators) - report.failure_count,
                len(indicators),
                report.check_time_ms,
            )

        return report


# =============================================================================
# Circuit Breaker State
# =============================================================================


class CircuitBreaker:
    """Manages circuit breaker state for governance bypass.

    When tripped, all optional governance phases are skipped and
    retrieval uses raw embedding scores only.
    """

    def __init__(self):
        self._tripped = False
        self._last_check: datetime | None = None
        self._last_report: GovernanceHealthReport | None = None
        self._trip_count = 0

    @property
    def is_tripped(self) -> bool:
        return self._tripped

    @property
    def last_report(self) -> GovernanceHealthReport | None:
        return self._last_report

    def check_and_update(self, conn=None) -> GovernanceHealthReport:
        """Run health check and update circuit breaker state.

        Auto-resets if failures drop below threshold.
        """
        checker = GovernanceHealthCheck(conn)
        report = checker.run_all()

        self._last_check = _utc_now()
        self._last_report = report

        if report.circuit_breaker_tripped:
            if not self._tripped:
                self._tripped = True
                self._trip_count += 1
                logger.warning("Circuit breaker TRIPPED (trip #%d)", self._trip_count)
        else:
            if self._tripped:
                self._tripped = False
                logger.info("Circuit breaker AUTO-RESET — governance restored")

        return report

    def should_check(self) -> bool:
        """Whether enough time has passed for the next health check.

        HEALTH-011: When tripped, probe at CIRCUIT_BREAKER_RECOVERY_INTERVAL_MINUTES
        (1 min) instead of the normal HEALTH_CHECK_INTERVAL_MINUTES (5 min).
        This implements half-open behaviour — faster recovery probing.
        """
        if self._last_check is None:
            return True
        elapsed = (_utc_now() - _ensure_aware(self._last_check)).total_seconds() / 60
        interval = (
            CIRCUIT_BREAKER_RECOVERY_INTERVAL_MINUTES
            if self._tripped
            else HEALTH_CHECK_INTERVAL_MINUTES
        )
        return elapsed >= interval

    def to_dict(self) -> dict[str, Any]:
        """Serialize for API response."""
        return {
            "tripped": self._tripped,
            "recovery_mode": self._tripped,  # HEALTH-011: half-open probing active when tripped
            "trip_count": self._trip_count,
            "last_check": self._last_check.isoformat() if self._last_check else None,
            "last_report": {
                "healthy": self._last_report.healthy,
                "failure_count": self._last_report.failure_count,
                "failures": [
                    {"indicator": c.indicator, "detail": c.detail}
                    for c in (self._last_report.checks if self._last_report else [])
                    if not c.healthy
                ],
            }
            if self._last_report
            else None,
        }


# Module-level singleton
circuit_breaker = CircuitBreaker()
