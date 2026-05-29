"""Request-local Stage 3 optional work budget accounting."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

MetricRecorder = Callable[[str, float, dict[str, str]], None]


@dataclass
class Stage3OptionalBudget:
    """Track request-local foreground budget for optional injection expansion."""

    budget_ms: float
    recorder: MetricRecorder | None = None
    spent_ms: float = 0.0

    def remaining_ms(self) -> float:
        return max(0.0, float(self.budget_ms) - self.spent_ms)

    def can_start(self, unit: str, *, min_remaining_ms: float) -> bool:
        if self.remaining_ms() >= max(0.0, float(min_remaining_ms)):
            return True
        self._record(
            "ct_stage3_optional_skip_total",
            1.0,
            {"unit": unit, "reason": "stage3_budget_before_start"},
        )
        return False

    def record(self, unit: str, elapsed_ms: float) -> None:
        observed_ms = max(0.0, float(elapsed_ms or 0.0))
        self.spent_ms += observed_ms
        self._record("ct_stage3_optional_spent_ms", round(observed_ms, 2), {"unit": unit})

    def _record(self, metric: str, value: float, labels: dict[str, str]) -> None:
        if self.recorder is None:
            return
        self.recorder(metric, value, labels)
