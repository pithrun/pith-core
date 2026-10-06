"""Foreground latency contract helpers for optional request-path work."""

from __future__ import annotations

import math
import os
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from app.core.deadline import TurnDeadline

MetricRecorder = Callable[[str, float, dict[str, str]], None]
FOREGROUND_CONTRACT_DECISION_METRIC = "ct_foreground_contract_decision_total"
FOREGROUND_CONTRACT_CIRCUIT_OPEN_METRIC = "ct_foreground_contract_circuit_open_total"
FOREGROUND_CONTRACT_CIRCUIT_RECOVERY_METRIC = "ct_foreground_contract_circuit_recovery_total"
FOREGROUND_CONTRACT_CONFIG_INVALID_METRIC = "ct_foreground_contract_config_invalid_total"
FOREGROUND_CONTRACT_RECOVERY_PROBE_METRIC = "ct_foreground_contract_recovery_probe_total"
FOREGROUND_CONTRACT_WAIT_METRIC = "ct_foreground_contract_wait_ms"
DEFAULT_MIN_SAMPLES_FOR_PERCENTILE = 20
_RECOVERY_PROBE_REASONS = frozenset(
    {
        "latency_over_limit",
        "recent_p95_over_limit",
        "recovery_probe_over_limit",
    }
)
_RECOVERY_PROBE_GATE_ENV = "PITH_FOREGROUND_RECOVERY_PROBES_ENABLED"
_ENV_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_ENV_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


class ForegroundContractMode(str, Enum):
    OFF = "off"
    SHADOW = "shadow"
    ENFORCE = "enforce"


class ForegroundDecision(str, Enum):
    RUN = "run"
    SHADOW_RUN = "shadow_run"
    WOULD_SKIP = "would_skip"
    SKIP = "skip"


class ForegroundContractConfigError(ValueError):
    """Raised when a foreground unit's latency policy is internally unsafe."""

    def __init__(self, *, unit: str, field: str, value: object, message: str) -> None:
        super().__init__(f"{unit or 'unknown'}: {field}={value!r}: {message}")
        self.unit = _bounded_label(unit)
        self.field = _bounded_label(field)


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
    min_samples_for_percentile: int = DEFAULT_MIN_SAMPLES_FOR_PERCENTILE
    skip_when_cold: bool = False
    recovery_probe_enabled: bool | None = None
    reset_samples_on_successful_probe: bool = True

    def __post_init__(self) -> None:
        if not str(self.unit or "").strip():
            raise ForegroundContractConfigError(
                unit="unknown",
                field="unit",
                value=self.unit,
                message="must be non-empty",
            )
        if not str(self.criticality or "").strip():
            raise ForegroundContractConfigError(
                unit=self.unit,
                field="criticality",
                value=self.criticality,
                message="must be non-empty",
            )
        for field_name in ("min_remaining_ms", "recent_p95_limit_ms", "circuit_ttl_s"):
            raw_value = getattr(self, field_name)
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ForegroundContractConfigError(
                    unit=self.unit,
                    field=field_name,
                    value=raw_value,
                    message="must be numeric",
                ) from exc
            if not math.isfinite(value) or value < 0.0:
                raise ForegroundContractConfigError(
                    unit=self.unit,
                    field=field_name,
                    value=raw_value,
                    message="must be finite and nonnegative",
                )
            object.__setattr__(self, field_name, value)
        for field_name in ("max_samples", "min_samples_for_percentile"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ForegroundContractConfigError(
                    unit=self.unit,
                    field=field_name,
                    value=value,
                    message="must be an integer >= 1",
                )
        if float(self.min_remaining_ms) < float(self.recent_p95_limit_ms):
            raise ForegroundContractConfigError(
                unit=self.unit,
                field="min_remaining_ms",
                value=self.min_remaining_ms,
                message="must be >= recent_p95_limit_ms",
            )


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


@dataclass(frozen=True)
class ForegroundContractConfigResult:
    unit: str
    mode: ForegroundContractMode
    config: ForegroundContractConfig | None = None
    error: ForegroundContractConfigError | None = None

    @property
    def valid(self) -> bool:
        return self.config is not None and self.error is None


@dataclass(frozen=True)
class ForegroundLatencySample:
    elapsed_ms: float
    recorded_at: float


@dataclass
class ForegroundUnitHealth:
    max_samples: int = 64
    min_samples_for_percentile: int = DEFAULT_MIN_SAMPLES_FOR_PERCENTILE
    samples: deque[ForegroundLatencySample] = field(init=False)
    circuit_open_until: float = 0.0
    circuit_opened_at: float | None = None
    circuit_reason: str = ""
    circuit_mode: str = ""
    recovery_probe_in_flight: bool = False

    def __post_init__(self) -> None:
        self.samples = deque(maxlen=max(1, int(self.max_samples)))

    def record_latency_ms(self, elapsed_ms: float, *, now: float) -> None:
        self.samples.append(
            ForegroundLatencySample(
                elapsed_ms=max(0.0, float(elapsed_ms)),
                recorded_at=float(now),
            )
        )

    def p95_ms(self) -> float | None:
        if len(self.samples) < int(self.min_samples_for_percentile):
            return None
        ordered = sorted(sample.elapsed_ms for sample in self.samples)
        index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1))
        return ordered[index]

    def max_ms(self) -> float | None:
        return max((sample.elapsed_ms for sample in self.samples), default=None)

    def oldest_sample_age_ms(self, *, now: float) -> float | None:
        if not self.samples:
            return None
        return max(0.0, (float(now) - self.samples[0].recorded_at) * 1000.0)

    def open_circuit(
        self,
        reason: str,
        ttl_s: float,
        *,
        mode: ForegroundContractMode,
        now: float | None = None,
    ) -> None:
        current = time.monotonic() if now is None else now
        self.circuit_open_until = current + max(0.0, ttl_s)
        self.circuit_opened_at = current
        self.circuit_reason = _bounded_label(reason)
        self.circuit_mode = mode.value
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
        self.samples.clear()

    def retire_open_epoch_samples(self) -> int:
        if self.circuit_opened_at is None:
            return 0
        before = len(self.samples)
        self.samples = deque(
            (
                sample
                for sample in self.samples
                if sample.recorded_at > self.circuit_opened_at
            ),
            maxlen=max(1, int(self.max_samples)),
        )
        return before - len(self.samples)

    def clear_circuit(self) -> tuple[str, str]:
        prior = (self.circuit_reason, self.circuit_mode)
        self.circuit_open_until = 0.0
        self.circuit_opened_at = None
        self.circuit_reason = ""
        self.circuit_mode = ""
        self.recovery_probe_in_flight = False
        return prior


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
        open_reason = ""
        recovery_event: tuple[str, str, str] | None = None
        now = time.monotonic()
        with self._lock:
            health = self._health_for_locked(config)
            if deadline is not None and not deadline.can_start(
                config.unit,
                min_remaining_ms=config.min_remaining_ms,
            ):
                reason = "deadline_before_start"
                should_skip = True
            elif health.circuit_open(now=now):
                reason = health.circuit_reason or "circuit_open"
                should_skip = True
            elif health.recovery_probe_in_flight:
                reason = "recovery_probe_in_flight"
                should_skip = True
            elif health.circuit_reason and _can_recovery_probe(config, health, mode):
                health.mark_recovery_probe()
                reason = "recovery_probe"
            else:
                retired = health.retire_open_epoch_samples() if health.circuit_reason else 0
                recent_p95 = health.p95_ms()
                if recent_p95 is None and not health.samples and config.skip_when_cold:
                    prior_reason, prior_mode = health.clear_circuit()
                    if prior_reason:
                        recovery_event = (
                            prior_reason,
                            prior_mode,
                            "open_epoch_expired" if retired else "cold_start",
                        )
                    reason = "cold_start_no_samples"
                    should_skip = True
                elif recent_p95 is not None and recent_p95 > config.recent_p95_limit_ms:
                    reason = "recent_p95_over_limit"
                    health.open_circuit(
                        reason,
                        config.circuit_ttl_s,
                        mode=mode,
                        now=now,
                    )
                    open_reason = reason
                    should_skip = True
                else:
                    prior_reason, prior_mode = health.clear_circuit()
                    if prior_reason:
                        recovery_event = (
                            prior_reason,
                            prior_mode,
                            "open_epoch_expired" if retired else "p95_within_limit",
                        )

        if open_reason:
            self._record_circuit_open(config, open_reason, mode=mode)
        if recovery_event:
            prior_reason, prior_mode, recovery_reason = recovery_event
            self._record_circuit_recovery(
                config,
                prior_reason=prior_reason,
                prior_mode=prior_mode,
                recovery_reason=recovery_reason,
            )

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
        mode = _normalize_mode(config.mode)
        now = time.monotonic()
        circuit_open_reason = ""
        recovery_probe_outcome = ""
        recovery_event: tuple[str, str, str] | None = None
        with self._lock:
            health = self._health_for_locked(config)
            was_recovery_probe = health.consume_recovery_probe()
            if was_recovery_probe:
                recovery_probe_outcome = (
                    "success" if elapsed_ms <= config.recent_p95_limit_ms else "over_limit"
                )
            if (
                was_recovery_probe
                and elapsed_ms <= config.recent_p95_limit_ms
            ):
                if config.reset_samples_on_successful_probe:
                    health.reset_samples()
                prior_reason, prior_mode = health.clear_circuit()
                if prior_reason:
                    recovery_event = (
                        prior_reason,
                        prior_mode,
                        "recovery_probe_success",
                    )
            health.record_latency_ms(elapsed_ms, now=now)
            if elapsed_ms > config.recent_p95_limit_ms:
                circuit_open_reason = (
                    "recovery_probe_over_limit"
                    if was_recovery_probe
                    else "latency_over_limit"
                )
                health.open_circuit(
                    circuit_open_reason,
                    config.circuit_ttl_s,
                    mode=mode,
                    now=now,
                )
        if circuit_open_reason:
            self._record_circuit_open(config, circuit_open_reason, mode=mode)
        if recovery_event:
            prior_reason, prior_mode, recovery_reason = recovery_event
            self._record_circuit_recovery(
                config,
                prior_reason=prior_reason,
                prior_mode=prior_mode,
                recovery_reason=recovery_reason,
            )
        if recovery_probe_outcome:
            self._record(
                FOREGROUND_CONTRACT_RECOVERY_PROBE_METRIC,
                1.0,
                {
                    "unit": config.unit,
                    "mode": _normalize_mode(config.mode).value,
                    "outcome": recovery_probe_outcome,
                },
            )
        self._record(
            FOREGROUND_CONTRACT_WAIT_METRIC,
            max(0.0, float(elapsed_ms)),
            {
                "unit": config.unit,
                "decision": "observed",
                "answer_path": _bounded_label(answer_path),
                "mode": _normalize_mode(config.mode).value,
            },
        )

    def cancel_recovery_probe(self, config: ForegroundContractConfig) -> None:
        mode = _normalize_mode(config.mode)
        if not config.enabled or mode is ForegroundContractMode.OFF:
            return
        with self._lock:
            health = self._health_for_locked(config)
            was_recovery_probe = health.consume_recovery_probe()
        if was_recovery_probe:
            self._record(
                FOREGROUND_CONTRACT_RECOVERY_PROBE_METRIC,
                1.0,
                {
                    "unit": config.unit,
                    "mode": mode.value,
                    "outcome": "cancelled",
                },
            )

    def health_snapshot(self, unit: str) -> dict[str, float | str | bool | None]:
        with self._lock:
            health = self._health.get(unit)
            if health is None:
                return {
                    "unit": unit,
                    "sample_count": 0,
                    "percentile_sample_floor": DEFAULT_MIN_SAMPLES_FOR_PERCENTILE,
                    "percentile_ready": False,
                    "p95_ms": None,
                    "max_ms": None,
                    "oldest_sample_age_ms": None,
                    "circuit_open": False,
                    "circuit_reason": "",
                    "recovery_pending": False,
                    "recovery_probe_in_flight": False,
                }
            now = time.monotonic()
            circuit_open = health.circuit_open(now=now)
            return {
                "unit": unit,
                "sample_count": len(health.samples),
                "percentile_sample_floor": health.min_samples_for_percentile,
                "percentile_ready": len(health.samples) >= health.min_samples_for_percentile,
                "p95_ms": health.p95_ms(),
                "max_ms": health.max_ms(),
                "oldest_sample_age_ms": health.oldest_sample_age_ms(now=now),
                "circuit_open": circuit_open,
                "circuit_reason": health.circuit_reason,
                "recovery_pending": (
                    bool(health.circuit_reason)
                    and not circuit_open
                    and not health.recovery_probe_in_flight
                ),
                "recovery_probe_in_flight": health.recovery_probe_in_flight,
            }

    def _health_for_locked(self, config: ForegroundContractConfig) -> ForegroundUnitHealth:
        health = self._health.get(config.unit)
        if (
            health is None
            or health.samples.maxlen != config.max_samples
            or health.min_samples_for_percentile != config.min_samples_for_percentile
        ):
            health = ForegroundUnitHealth(
                max_samples=config.max_samples,
                min_samples_for_percentile=config.min_samples_for_percentile,
            )
            self._health[config.unit] = health
        return health

    def _record_circuit_open(
        self,
        config: ForegroundContractConfig,
        reason: str,
        *,
        mode: ForegroundContractMode,
    ) -> None:
        self._record(
            FOREGROUND_CONTRACT_CIRCUIT_OPEN_METRIC,
            1.0,
            {"unit": config.unit, "reason": reason, "mode": mode.value},
        )

    def _record_circuit_recovery(
        self,
        config: ForegroundContractConfig,
        *,
        prior_reason: str,
        prior_mode: str,
        recovery_reason: str,
    ) -> None:
        self._record(
            FOREGROUND_CONTRACT_CIRCUIT_RECOVERY_METRIC,
            1.0,
            {
                "unit": config.unit,
                "mode": prior_mode,
                "prior_reason": prior_reason,
                "recovery_reason": recovery_reason,
            },
        )

    def _record_decision(self, decision: ForegroundContractDecision, *, answer_path: str) -> None:
        self._record(FOREGROUND_CONTRACT_DECISION_METRIC, 1.0, decision.metric_labels(answer_path=answer_path))

    def _record(self, name: str, value: float, labels: dict[str, str]) -> None:
        if self._recorder is None:
            return
        try:
            self._recorder(name, value, {str(key): _bounded_label(value) for key, value in labels.items()})
        except Exception:
            pass


def build_foreground_contract_config(
    *,
    recorder: MetricRecorder | None = None,
    caller: str,
    **kwargs: object,
) -> ForegroundContractConfigResult:
    """Build a validated config and convert operator mistakes into bounded state."""
    unit = str(kwargs.get("unit") or "unknown")
    mode = _normalize_mode(kwargs.get("mode", ForegroundContractMode.SHADOW))
    try:
        config = ForegroundContractConfig(**kwargs)  # type: ignore[arg-type]
    except ForegroundContractConfigError as exc:
        if recorder is not None:
            try:
                recorder(
                    FOREGROUND_CONTRACT_CONFIG_INVALID_METRIC,
                    1.0,
                    {
                        "unit": _bounded_label(exc.unit),
                        "field": _bounded_label(exc.field),
                        "caller": _bounded_label(caller),
                        "mode": mode.value,
                    },
                )
            except Exception:
                pass
        return ForegroundContractConfigResult(
            unit=unit,
            mode=mode,
            error=exc,
        )
    return ForegroundContractConfigResult(
        unit=config.unit,
        mode=_normalize_mode(config.mode),
        config=config,
    )


_CONTRACT: ForegroundContract | None = None
_CONTRACT_LOCK = threading.Lock()


def foreground_contract_mode_from_env() -> ForegroundContractMode:
    return _normalize_mode(os.environ.get("PITH_FOREGROUND_CONTRACT_MODE", "shadow"))


def _recovery_probe_gate_from_env() -> bool | None:
    raw = os.environ.get(_RECOVERY_PROBE_GATE_ENV)
    if raw is None:
        return None
    normalized = str(raw).strip().lower()
    if normalized in _ENV_TRUE_VALUES:
        return True
    if normalized in _ENV_FALSE_VALUES:
        return False
    return False


def _recovery_probe_allowed(
    config: ForegroundContractConfig,
    mode: ForegroundContractMode,
) -> bool:
    gate = _recovery_probe_gate_from_env()
    if gate is False or config.recovery_probe_enabled is False:
        return False
    if config.recovery_probe_enabled is True:
        return True
    return gate is True and mode is ForegroundContractMode.ENFORCE


def _can_recovery_probe(
    config: ForegroundContractConfig,
    health: ForegroundUnitHealth,
    mode: ForegroundContractMode,
) -> bool:
    return (
        _recovery_probe_allowed(config, mode)
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
