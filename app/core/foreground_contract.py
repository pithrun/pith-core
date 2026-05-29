"""Foreground latency contract helpers for optional request-path work."""

from __future__ import annotations

import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Deque

from app.core.deadline import TurnDeadline

MetricRecorder = Callable[[str, float, dict[str, str]], None]
_RECOVERY_PROBE_REASONS = frozenset(
    {
        "latency_over_limit",
        "recent_p95_over_limit",
        "recovery_probe_over_limit",
    }
)


class ForegroundContractMode(str, Enum):
    OFF = "off"
    SHADOW = "shadow"
    ENFORCE = "enforce"


class ForegroundDecision(str, Enum):
    RUN = "run"
    SHADOW_RUN = "shadow_run"
    WOULD_SKIP = "would_skip"
    SKIP = "skip"


@dataclass(frozen=True)
class ForegroundContractConfig:
    unit: str
    criticality: str
    min_remaining_ms: float
    recent_p95_limit_ms: float
    mode: ForegroundContractMode = ForegroundContractMode.SHADOW
    enabled: bool = True
    circuit_ttl_s: float = 60.0
    max_samples: int = 64
    skip_when_cold: bool = False
    recovery_probe_enabled: bool = False
    reset_samples_on_successful_probe: bool = True


@dataclass(frozen=True)
class ForegroundContractDecision:
    unit: str
    decision: ForegroundDecision
    reason: str
    remaining_ms: float | None
    mode: ForegroundContractMode

    def metric_labels(self, *, answer_path: str = "unknown") -> dict[str, str]:
        return {
            "unit": self.unit,
            "decision": self.decision.value,
            "reason": self.reason,
            "answer_path": _bounded_label(answer_path),
            "mode": self.mode.value,
        }


@dataclass
class ForegroundUnitHealth:
    max_samples: int = 64
    samples_ms: Deque[float] = field(init=False)
    circuit_open_until: float = 0.0
    circuit_reason: str = ""
    recovery_probe_in_flight: bool = False

    def __post_init__(self) -> None:
        self.samples_ms = deque(maxlen=max(1, int(self.max_samples)))

    def record_latency_ms(self, elapsed_ms: float) -> None:
        self.samples_ms.append(max(0.0, float(elapsed_ms)))

    def p95_ms(self) -> float | None:
        if not self.samples_ms:
            return None
        ordered = sorted(self.samples_ms)
        index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1))
        return ordered[index]

    def open_circuit(self, reason: str, ttl_s: float, *, now: float | None = None) -> None:
        current = time.monotonic() if now is None else now
        self.circuit_open_until = current + max(0.0, ttl_s)
        self.circuit_reason = _bounded_label(reason)
        self.recovery_probe_in_flight = False

    def circuit_open(self, *, now: float | None = None) -> bool:
        current = time.monotonic() if now is None else now
        return current < self.circuit_open_until

    def mark_recovery_probe(self) -> None:
        self.recovery_probe_in_flight = True

    def consume_recovery_probe(self) -> bool:
        was_probe = self.recovery_probe_in_flight
        self.recovery_probe_in_flight = False
        return was_probe

    def reset_samples(self) -> None:
        self.samples_ms.clear()


class ForegroundContract:
    """Process-local shadow/enforcement decision state for foreground work."""

    def __init__(self, recorder: MetricRecorder | None = None) -> None:
        self._health: dict[str, ForegroundUnitHealth] = {}
        self._lock = threading.Lock()
        self._recorder = recorder

    def decide(
        self,
        config: ForegroundContractConfig,
        *,
        deadline: TurnDeadline | None = None,
        answer_path: str = "unknown",
    ) -> ForegroundContractDecision:
        mode = _normalize_mode(config.mode)
        remaining = deadline.remaining_ms() if deadline is not None else None
        if not config.enabled or mode is ForegroundContractMode.OFF:
            return ForegroundContractDecision(
                unit=config.unit,
                decision=ForegroundDecision.RUN,
                reason="disabled",
                remaining_ms=remaining,
                mode=mode,
            )

        reason = "healthy"
        should_skip = False
        with self._lock:
            health = self._health_for_locked(config)
            if deadline is not None and not deadline.can_start(
                config.unit,
                min_remaining_ms=config.min_remaining_ms,
            ):
                reason = "deadline_before_start"
                should_skip = True
            elif health.circuit_open():
                reason = health.circuit_reason or "circuit_open"
                should_skip = True
            elif health.recovery_probe_in_flight:
                reason = "recovery_probe_in_flight"
                should_skip = True
            else:
                recent_p95 = health.p95_ms()
                if recent_p95 is None and config.skip_when_cold:
                    reason = "cold_start_no_samples"
                    should_skip = True
                elif recent_p95 is not None and recent_p95 > config.recent_p95_limit_ms:
                    if _can_recovery_probe(config, health):
                        health.mark_recovery_probe()
                        reason = "recovery_probe"
                    else:
                        health.open_circuit("recent_p95_over_limit", config.circuit_ttl_s)
                        reason = "recent_p95_over_limit"
                        should_skip = True

        if should_skip:
            action = (
                ForegroundDecision.SKIP
                if mode is ForegroundContractMode.ENFORCE
                else ForegroundDecision.WOULD_SKIP
            )
        else:
            action = (
                ForegroundDecision.RUN
                if mode is ForegroundContractMode.ENFORCE
                else ForegroundDecision.SHADOW_RUN
            )
        decision = ForegroundContractDecision(
            unit=config.unit,
            decision=action,
            reason=reason,
            remaining_ms=remaining,
            mode=mode,
        )
        self._record_decision(decision, answer_path=answer_path)
        return decision

    def record_latency_ms(
        self,
        config: ForegroundContractConfig,
        elapsed_ms: float,
        *,
        answer_path: str = "unknown",
    ) -> None:
        if not config.enabled or _normalize_mode(config.mode) is ForegroundContractMode.OFF:
            return
        circuit_open_reason = ""
        with self._lock:
            health = self._health_for_locked(config)
            was_recovery_probe = health.consume_recovery_probe()
            if (
                was_recovery_probe
                and elapsed_ms <= config.recent_p95_limit_ms
                and config.reset_samples_on_successful_probe
            ):
                health.reset_samples()
                health.circuit_reason = ""
            health.record_latency_ms(elapsed_ms)
            if elapsed_ms > config.recent_p95_limit_ms:
                circuit_open_reason = (
                    "recovery_probe_over_limit"
                    if was_recovery_probe and config.recovery_probe_enabled
                    else "latency_over_limit"
                )
                health.open_circuit(circuit_open_reason, config.circuit_ttl_s)
        if circuit_open_reason:
            self._record(
                "ct_foreground_contract_circuit_open_total",
                1.0,
                {
                    "unit": config.unit,
                    "reason": circuit_open_reason,
                    "mode": _normalize_mode(config.mode).value,
                },
            )
        self._record(
            "ct_foreground_contract_wait_ms",
            max(0.0, float(elapsed_ms)),
            {
                "unit": config.unit,
                "decision": "observed",
                "answer_path": _bounded_label(answer_path),
                "mode": _normalize_mode(config.mode).value,
            },
        )

    def cancel_recovery_probe(self, config: ForegroundContractConfig) -> None:
        if not config.enabled or _normalize_mode(config.mode) is ForegroundContractMode.OFF:
            return
        with self._lock:
            health = self._health_for_locked(config)
            health.consume_recovery_probe()

    def health_snapshot(self, unit: str) -> dict[str, float | str | bool | None]:
        with self._lock:
            health = self._health.get(unit)
            if health is None:
                return {"unit": unit, "sample_count": 0, "p95_ms": None, "circuit_open": False}
            return {
                "unit": unit,
                "sample_count": len(health.samples_ms),
                "p95_ms": health.p95_ms(),
                "circuit_open": health.circuit_open(),
                "circuit_reason": health.circuit_reason,
                "recovery_probe_in_flight": health.recovery_probe_in_flight,
            }

    def _health_for_locked(self, config: ForegroundContractConfig) -> ForegroundUnitHealth:
        health = self._health.get(config.unit)
        if health is None or health.samples_ms.maxlen != max(1, int(config.max_samples)):
            health = ForegroundUnitHealth(max_samples=config.max_samples)
            self._health[config.unit] = health
        return health

    def _record_decision(self, decision: ForegroundContractDecision, *, answer_path: str) -> None:
        self._record("ct_foreground_contract_decision_total", 1.0, decision.metric_labels(answer_path=answer_path))

    def _record(self, name: str, value: float, labels: dict[str, str]) -> None:
        if self._recorder is None:
            return
        self._recorder(name, value, {str(key): _bounded_label(value) for key, value in labels.items()})


_CONTRACT: ForegroundContract | None = None
_CONTRACT_LOCK = threading.Lock()


def foreground_contract_mode_from_env() -> ForegroundContractMode:
    return _normalize_mode(os.environ.get("PITH_FOREGROUND_CONTRACT_MODE", "shadow"))


def _can_recovery_probe(config: ForegroundContractConfig, health: ForegroundUnitHealth) -> bool:
    return (
        bool(config.recovery_probe_enabled)
        and not health.recovery_probe_in_flight
        and health.circuit_reason in _RECOVERY_PROBE_REASONS
    )


def foreground_contract_mode_for_unit(unit: str) -> ForegroundContractMode:
    """Return global mode, overridden by a unit-specific env var when valid."""
    global_mode = foreground_contract_mode_from_env()
    raw = os.environ.get(f"PITH_FOREGROUND_CONTRACT_MODE_{_unit_env_suffix(unit)}")
    if raw is None or not raw.strip():
        return global_mode
    normalized = str(raw).strip().lower()
    try:
        return ForegroundContractMode(normalized)
    except ValueError:
        return global_mode


def get_foreground_contract(recorder: MetricRecorder | None = None) -> ForegroundContract:
    global _CONTRACT
    with _CONTRACT_LOCK:
        if _CONTRACT is None:
            _CONTRACT = ForegroundContract(recorder=recorder)
        elif recorder is not None and _CONTRACT._recorder is None:
            _CONTRACT._recorder = recorder
        return _CONTRACT


def _normalize_mode(mode: ForegroundContractMode | str) -> ForegroundContractMode:
    if isinstance(mode, ForegroundContractMode):
        return mode
    normalized = str(mode or "").strip().lower()
    try:
        return ForegroundContractMode(normalized)
    except ValueError:
        return ForegroundContractMode.SHADOW


def _unit_env_suffix(unit: str) -> str:
    suffix = []
    for char in str(unit or ""):
        suffix.append(char.upper() if char.isalnum() else "_")
    return "".join(suffix).strip("_") or "UNKNOWN"


def _bounded_label(value: object) -> str:
    text = str(value or "unknown").strip().lower()
    safe = []
    for char in text[:64]:
        if char.isalnum() or char in {"_", "-", "."}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe) or "unknown"
