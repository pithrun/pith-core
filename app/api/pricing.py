"""Conversation-turn metering for Pith pricing tiers.

PRICING-002: Count concept-producing turns against daily budget.
PRICING-003: Generate upgrade_nudge when budget exhausted.
PRICING-005: Dev mode bypass via PITH_DEV_MODE=true.
PREVIEW-001: Free developer preview is unlimited unless limits are explicitly enabled.
"""

import logging
import os
from datetime import UTC, datetime
from enum import StrEnum

from app.core.config import (
    DAILY_TURN_BUDGET_DEFAULT,
    DAILY_TURN_BUDGET_DEV,
    DAILY_TURN_BUDGET_ENTERPRISE,
    DAILY_TURN_BUDGET_FREE,
    DAILY_TURN_BUDGET_PRO,
)

logger = logging.getLogger(__name__)


class BudgetZone(StrEnum):
    """PRICING-006: Budget depletion zones for quality escalation."""

    NORMAL = "normal"  # >40% remaining
    CONSERVATION = "conservation"  # 20-40% remaining
    CRITICAL = "critical"  # >0% but <20% remaining
    EXHAUSTED = "exhausted"  # 0% remaining


# PRICING-005: Dev mode bypass
PITH_DEV_MODE = os.environ.get("PITH_DEV_MODE", "false").lower() == "true"


def _truthy_env(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def usage_limits_enabled() -> bool:
    """Return True only when usage limits are explicitly enabled.

    Public/free developer preview installs must not impose Pith turn, query,
    or learning caps by default. This keeps metering code available for future
    opt-in environments without degrading preview users.
    """
    return _truthy_env(os.environ.get("PITH_USAGE_LIMITS_ENABLED"))


def dev_mode_active() -> bool:
    """Check whether Pith should run with unlimited local usage."""
    return PITH_DEV_MODE or not usage_limits_enabled()


# Tier-to-budget mapping
_TIER_BUDGETS = {
    "free": DAILY_TURN_BUDGET_FREE,
    "pro": DAILY_TURN_BUDGET_PRO,
    "enterprise": DAILY_TURN_BUDGET_ENTERPRISE,
    "dev": DAILY_TURN_BUDGET_DEV,
}


class ConversationMeter:
    """In-memory daily turn counter. Resets on date change or server restart.

    Counts conversation_turn calls where auto-learning produced concepts.
    In free developer preview this is diagnostic-only unless usage limits are
    explicitly enabled.
    """

    def __init__(self):
        self._date_key: str = ""
        self._turn_count: int = 0
        self._tier: str = os.environ.get("PITH_TIER", "dev" if dev_mode_active() else "default")
        self._daily_limit: int = _TIER_BUDGETS.get(self._tier, DAILY_TURN_BUDGET_DEFAULT)
        self._capped_at: str | None = None  # PRICING-007: ISO timestamp when budget hit 0

    def _reset_if_new_day(self) -> None:
        """Reset counter on date rollover."""
        today = datetime.now(UTC).strftime("%Y%m%d")
        if self._date_key != today:
            self._date_key = today
            self._turn_count = 0
            self._capped_at = None  # PRICING-007: Reset capped timestamp on new day

    def check_turn_budget(self) -> int:
        """Return remaining turns for today. -1 if dev mode (unlimited)."""
        if dev_mode_active():
            return -1
        self._reset_if_new_day()
        return max(0, self._daily_limit - self._turn_count)

    def consume_turn(self) -> int:
        """Consume one turn. Returns remaining budget.

        Returns -1 in dev mode (unlimited). No-op in dev mode.
        """
        if dev_mode_active():
            return -1  # Dev mode: unlimited, no counting
        self._reset_if_new_day()
        _old_zone = self.get_budget_zone()
        self._turn_count += 1
        _new_zone = self.get_budget_zone()
        remaining = self._daily_limit - self._turn_count
        # MONITOR-001: Emit zone transition event
        if _old_zone != _new_zone:
            try:
                from app.ops.metrics import metrics as _bz_metrics
                _bz_metrics.record("budget_zone_transition", 1.0, {
                    "from_zone": _old_zone.value,
                    "to_zone": _new_zone.value,
                    "turns_used": self._turn_count,
                    "daily_limit": self._daily_limit,
                    "tier": self._tier,
                })
            except Exception:
                pass  # Metrics are best-effort
        if remaining <= 0:
            # PRICING-007: Record capped_at on first exhaustion
            if self._capped_at is None:
                self._capped_at = datetime.now(UTC).isoformat()
            logger.info(
                f"PRICING-002: Turn budget exhausted ({self._turn_count}/{self._daily_limit}, tier={self._tier})"
            )
        elif remaining <= self._daily_limit * 0.1:
            logger.info(f"PRICING-002: Turn budget low ({remaining}/{self._daily_limit} remaining, tier={self._tier})")
        return max(0, remaining)

    def get_upgrade_nudge(self) -> dict | None:
        """PRICING-003: Return upgrade nudge payload if budget exhausted."""
        if dev_mode_active():
            return None
        self._reset_if_new_day()
        remaining = self._daily_limit - self._turn_count
        if remaining <= 0:
            return {
                "type": "turn_budget_exhausted",
                "current_tier": self._tier,
                "daily_limit": self._daily_limit,
                "turns_used": self._turn_count,
                "message": (
                    f"Daily conversation turn limit reached "
                    f"({self._daily_limit} turns on {self._tier} tier). "
                    "Upgrade for higher limits."
                ),
                "upgrade_url": None,  # Placeholder — populated by billing integration
            }
        return None

    def get_status(self) -> dict:
        """Return current metering status for diagnostics."""
        self._reset_if_new_day()
        unlimited = dev_mode_active()
        return {
            "tier": self._tier,
            "daily_limit": -1 if unlimited else self._daily_limit,
            "turns_used": self._turn_count,
            "turns_remaining": self.check_turn_budget(),
            "dev_mode": unlimited,
            "usage_limits_enabled": usage_limits_enabled(),
            "unlimited_usage": unlimited,
            "date_key": self._date_key,
            "budget_zone": self.get_budget_zone().value,  # PRICING-006
            "capped_at": self._capped_at,  # PRICING-007
        }

    def get_budget_zone(self) -> BudgetZone:
        """PRICING-006: Return current budget depletion zone.

        Zones determine concept confidence thresholds:
          NORMAL (>=40%): standard thresholds
          CONSERVATION (20%-39%): elevated thresholds
          CRITICAL (<20%): high thresholds only
          EXHAUSTED (0%): no learning
        """
        if dev_mode_active():
            return BudgetZone.NORMAL  # Dev mode: always normal
        self._reset_if_new_day()
        remaining = self._daily_limit - self._turn_count
        if remaining <= 0 or self._daily_limit <= 0:
            return BudgetZone.EXHAUSTED  # Also handles misconfigured zero-limit
        pct = remaining / self._daily_limit
        if pct >= 0.40:
            return BudgetZone.NORMAL
        elif pct >= 0.20:
            return BudgetZone.CONSERVATION
        else:
            return BudgetZone.CRITICAL

    def get_recall_gap_attribution(self) -> dict | None:
        """PRICING-007: Return recall gap attribution if budget was capped today.

        Clients surface this to users so they know conversations after
        capped_at were not captured by Pith.
        """
        if dev_mode_active() or self._capped_at is None:
            return None
        return {
            "capped_at": self._capped_at,
            "date": self._date_key,
            "message": (
                f"Some conversations from today weren't captured because "
                f"you'd used all your Pith-powered messages. "
                f"Learning paused at {self._capped_at}."
            ),
        }

    def log_exhaustion_telemetry(
        self, session_id: str | None = None, session_in_progress: bool = False, concepts_created_today: int = 0
    ) -> dict:
        """PRICING-004: Return telemetry payload for PITH_MESSAGES_EXHAUSTED event.

        Caller is responsible for writing this to governance_events.
        Returns the event details dict for logging.
        """
        return {
            "event_type": "PITH_MESSAGES_EXHAUSTED",
            "time_of_day_utc": datetime.now(UTC).strftime("%H:%M:%S"),
            "pith_messages_used_today": self._turn_count,
            "daily_limit": self._daily_limit,
            "tier": self._tier,
            "session_in_progress": session_in_progress,
            "concepts_created_today": concepts_created_today,
            "capped_at": self._capped_at,
        }


# Module-level singleton
conversation_meter = ConversationMeter()
