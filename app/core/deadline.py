"""Request-local deadline helpers for latency-sensitive paths."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TurnDeadline:
    """Cooperative request deadline for conversation-turn enrichment work."""

    start_monotonic: float
    deadline_monotonic: float | None
    enabled: bool
    request_id: str | None = None
    skips: list[dict[str, Any]] = field(default_factory=list)
    overruns: list[dict[str, Any]] = field(default_factory=list)
    phase_modes: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def disabled(cls) -> "TurnDeadline":
        now = time.monotonic()
        return cls(start_monotonic=now, deadline_monotonic=None, enabled=False)

    @classmethod
    def from_budget_ms(
        cls,
        budget_ms: float | None,
        *,
        enabled: bool,
        request_id: str | None = None,
    ) -> "TurnDeadline":
        now = time.monotonic()
        if not enabled or budget_ms is None or budget_ms <= 0:
            return cls(
                start_monotonic=now,
                deadline_monotonic=None,
                enabled=False,
                request_id=request_id,
            )
        return cls(
            start_monotonic=now,
            deadline_monotonic=now + (budget_ms / 1000.0),
            enabled=True,
            request_id=request_id,
        )

    def elapsed_ms(self) -> float:
        return max(0.0, (time.monotonic() - self.start_monotonic) * 1000.0)

    def budget_ms(self) -> float | None:
        if self.deadline_monotonic is None:
            return None
        return max(0.0, (self.deadline_monotonic - self.start_monotonic) * 1000.0)

    def remaining_ms(self) -> float | None:
        if not self.enabled or self.deadline_monotonic is None:
            return None
        return max(0.0, (self.deadline_monotonic - time.monotonic()) * 1000.0)

    def expired(self) -> bool:
        remaining = self.remaining_ms()
        return remaining is not None and remaining <= 0.0

    def can_start(self, phase: str, min_remaining_ms: float = 0.0) -> bool:
        remaining = self.remaining_ms()
        if remaining is None:
            return True
        minimum = max(0.0, min_remaining_ms)
        if minimum == 0.0:
            return remaining > 0.0
        return remaining >= minimum

    def child_budget_ms(
        self,
        phase: str,
        requested_ms: float,
        min_remaining_ms: float = 0.0,
    ) -> float:
        requested = max(0.0, requested_ms)
        remaining = self.remaining_ms()
        if remaining is None:
            return requested
        if remaining < max(0.0, min_remaining_ms):
            return 0.0
        return min(requested, remaining)

    def optional_minimum_ms(self, min_remaining_ms: float, protected_tail_ms: float = 0.0) -> float:
        """Reserve protected tail budget before optional work starts."""
        return max(0.0, min_remaining_ms) + max(0.0, protected_tail_ms)

    def protected_phase_mode(
        self,
        phase: str,
        *,
        full_min_remaining_ms: float,
        lite_min_remaining_ms: float,
        criticality: str = "protected_governance",
    ) -> str:
        """Return full/lite/emergency mode without skipping a protected phase."""
        remaining = self.remaining_ms()
        if remaining is None:
            mode = "full"
        elif remaining >= max(0.0, full_min_remaining_ms):
            mode = "full"
        elif remaining >= max(0.0, lite_min_remaining_ms):
            mode = "lite"
        else:
            mode = "emergency_minimal"
        self.record_phase_mode(phase, mode, criticality=criticality)
        return mode

    def record_phase_mode(self, phase: str, mode: str, *, criticality: str) -> None:
        if not self.enabled:
            return
        self.phase_modes.append({
            "phase": phase,
            "mode": mode,
            "criticality": criticality,
            "elapsed_ms": round(self.elapsed_ms(), 2),
            "remaining_ms": self.remaining_ms(),
        })

    def skip(self, phase: str, reason: str, **details: Any) -> None:
        if not self.enabled:
            return
        self.skips.append({
            "phase": phase,
            "reason": reason,
            "elapsed_ms": round(self.elapsed_ms(), 2),
            "remaining_ms": self.remaining_ms(),
            **details,
        })

    def overrun(self, phase: str, **details: Any) -> None:
        if not self.enabled:
            return
        budget = self.budget_ms()
        if budget is None:
            return
        elapsed = self.elapsed_ms()
        if elapsed <= budget:
            return
        self.overruns.append({
            "phase": phase,
            "elapsed_ms": round(elapsed, 2),
            "overrun_ms": round(elapsed - budget, 2),
            **details,
        })
