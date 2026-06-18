"""In-memory local service contention signals for pressure admission."""

from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import asdict, dataclass
from threading import RLock
from typing import Any

from app.core.datetime_utils import _utc_now_iso

LOCAL_CONTENTION_METRICS = frozenset(
    {
        "conversation_turn_latency_ms",
        "db_boundary_request_scope_duration_ms",
        "db_boundary_managed_hold_ms",
        "session_learn_latency_ms",
        "learn_pipeline_latency_ms",
    }
)
_DB_METRICS = frozenset({"db_boundary_request_scope_duration_ms", "db_boundary_managed_hold_ms"})
_LEARN_METRICS = frozenset({"session_learn_latency_ms", "learn_pipeline_latency_ms"})
_MAX_LABELS = 8
_MAX_LABEL_CHARS = 96


@dataclass(frozen=True)
class _Sample:
    monotonic_at: float
    timestamp: str
    value_ms: float
    labels: dict[str, str]


@dataclass(frozen=True)
class LocalContentionSignal:
    metric: str
    state: str
    max_ms: float
    p95_ms: float
    count: int
    newest_age_seconds: float | None
    reason_codes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LocalContentionSnapshot:
    enabled: bool
    active: bool
    state: str
    signals: list[dict[str, Any]]
    reason_codes: list[str]
    observed_at: str
    sample_age_ms: float
    window_seconds: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_lock = RLock()
_samples: dict[str, deque[_Sample]] = {metric: deque() for metric in LOCAL_CONTENTION_METRICS}
_snapshot_cache: dict[str, Any] = {"loaded_at": 0.0, "signature": None, "snapshot": None}


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _bounded_labels(labels: dict | None) -> dict[str, str]:
    clean: dict[str, str] = {}
    for key, value in list((labels or {}).items())[:_MAX_LABELS]:
        clean_key = str(key)[:_MAX_LABEL_CHARS]
        clean_value = str(value)[:_MAX_LABEL_CHARS]
        clean[clean_key] = clean_value
    return clean


def _signature() -> tuple[str | None, ...]:
    return tuple(
        os.environ.get(name)
        for name in (
            "PITH_LOCAL_CONTENTION_PRESSURE_ENABLED",
            "PITH_LOCAL_CONTENTION_FORCE_STATE",
            "PITH_LOCAL_CONTENTION_CACHE_TTL_SECONDS",
            "PITH_LOCAL_CONTENTION_SIGNAL_TTL_SECONDS",
            "PITH_LOCAL_CONTENTION_MIN_SAMPLES",
            "PITH_LOCAL_CONTENTION_CT_DEGRADED_P95_MS",
            "PITH_LOCAL_CONTENTION_CT_CRITICAL_MAX_MS",
            "PITH_LOCAL_CONTENTION_CT_CRITICAL_FRESH_SECONDS",
            "PITH_LOCAL_CONTENTION_DB_HIGH_MS",
            "PITH_LOCAL_CONTENTION_LEARN_HIGH_MS",
            "PITH_LOCAL_CONTENTION_LEARN_CRITICAL_MS",
            "PITH_LOCAL_CONTENTION_LEARN_CRITICAL_FRESH_SECONDS",
        )
    )


def _invalidate_cache() -> None:
    _snapshot_cache.update({"loaded_at": 0.0, "signature": None, "snapshot": None})


def reset_local_contention_state_for_tests() -> None:
    with _lock:
        for bucket in _samples.values():
            bucket.clear()
        _invalidate_cache()


def record_local_contention_metric(metric_name: str, value_ms: float, labels: dict | None = None, timestamp: str | None = None) -> None:
    if metric_name not in LOCAL_CONTENTION_METRICS:
        return
    if not _env_bool("PITH_LOCAL_CONTENTION_PRESSURE_ENABLED", True):
        return
    try:
        sample = _Sample(
            monotonic_at=time.monotonic(),
            timestamp=timestamp or _utc_now_iso(),
            value_ms=float(value_ms),
            labels=_bounded_labels(labels),
        )
    except (TypeError, ValueError):
        return
    max_samples = max(1, _env_int("PITH_LOCAL_CONTENTION_MAX_SAMPLES_PER_METRIC", 256))
    with _lock:
        bucket = _samples.setdefault(metric_name, deque())
        bucket.append(sample)
        while len(bucket) > max_samples:
            bucket.popleft()
        _invalidate_cache()


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(len(ordered) * percentile), len(ordered) - 1)
    return round(ordered[index], 2)


def _newest_threshold_age_seconds(fresh: list[_Sample], threshold_ms: float, *, now_mono: float) -> float | None:
    threshold_samples = [sample for sample in fresh if sample.value_ms > threshold_ms]
    if not threshold_samples:
        return None
    return round(max(0.0, now_mono - max(sample.monotonic_at for sample in threshold_samples)), 2)


def _classify_metric(metric: str, fresh: list[_Sample], *, now_mono: float) -> LocalContentionSignal | None:
    if not fresh:
        return None
    values = [sample.value_ms for sample in fresh]
    max_ms = round(max(values), 2)
    p95_ms = _percentile(values, 0.95)
    count = len(values)
    min_samples = max(1, _env_int("PITH_LOCAL_CONTENTION_MIN_SAMPLES", 3))
    state = "none"
    reason_codes: list[str] = []
    if metric == "conversation_turn_latency_ms":
        critical = _env_float("PITH_LOCAL_CONTENTION_CT_CRITICAL_MAX_MS", 15000.0)
        critical_fresh_seconds = max(1.0, _env_float("PITH_LOCAL_CONTENTION_CT_CRITICAL_FRESH_SECONDS", 120.0))
        degraded = _env_float("PITH_LOCAL_CONTENTION_CT_DEGRADED_P95_MS", 3500.0)
        critical_age = _newest_threshold_age_seconds(fresh, critical, now_mono=now_mono)
        if critical_age is not None and critical_age <= critical_fresh_seconds:
            state = "critical"
            reason_codes.append("local_ct_fresh_critical")
        elif count >= min_samples and p95_ms > degraded:
            state = "high"
            reason_codes.append(
                "local_ct_aged_critical_diagnostic" if critical_age is not None else "local_ct_p95_high"
            )
        elif count < min_samples and max_ms > degraded:
            state = "moderate"
            reason_codes.append(
                "local_ct_aged_critical_diagnostic" if critical_age is not None else "local_ct_low_sample"
            )
    elif metric in _DB_METRICS:
        high = _env_float("PITH_LOCAL_CONTENTION_DB_HIGH_MS", 10000.0)
        if max_ms > high:
            state = "high"
            reason_codes.append("local_db_boundary_high")
    elif metric in _LEARN_METRICS:
        critical = _env_float("PITH_LOCAL_CONTENTION_LEARN_CRITICAL_MS", 30000.0)
        critical_fresh_seconds = max(1.0, _env_float("PITH_LOCAL_CONTENTION_LEARN_CRITICAL_FRESH_SECONDS", 120.0))
        high = _env_float("PITH_LOCAL_CONTENTION_LEARN_HIGH_MS", 15000.0)
        critical_age = _newest_threshold_age_seconds(fresh, critical, now_mono=now_mono)
        if critical_age is not None and critical_age <= critical_fresh_seconds:
            state = "critical"
            reason_codes.append("local_learning_fresh_critical")
        elif max_ms > high:
            state = "high"
            reason_codes.append(
                "local_learning_aged_critical_diagnostic" if critical_age is not None else "local_learning_high"
            )
    if state == "none":
        return None
    newest_age = round(max(0.0, now_mono - max(sample.monotonic_at for sample in fresh)), 2)
    return LocalContentionSignal(metric, state, max_ms, p95_ms, count, newest_age, reason_codes)


def build_local_contention_snapshot(*, use_cache: bool = True) -> LocalContentionSnapshot:
    started = time.perf_counter()
    now_mono = time.monotonic()
    observed_at = _utc_now_iso()
    signature = _signature()
    cache_ttl = max(0.0, _env_float("PITH_LOCAL_CONTENTION_CACHE_TTL_SECONDS", 2.0))
    if use_cache and _snapshot_cache.get("snapshot") is not None:
        loaded_at = float(_snapshot_cache.get("loaded_at") or 0.0)
        if _snapshot_cache.get("signature") == signature and now_mono - loaded_at <= cache_ttl:
            return _snapshot_cache["snapshot"]

    if not _env_bool("PITH_LOCAL_CONTENTION_PRESSURE_ENABLED", True):
        snapshot = LocalContentionSnapshot(False, False, "none", [], ["local_contention_disabled"], observed_at, 0.0, 0)
        _snapshot_cache.update({"loaded_at": now_mono, "signature": signature, "snapshot": snapshot})
        return snapshot

    forced = os.environ.get("PITH_LOCAL_CONTENTION_FORCE_STATE")
    if forced:
        state = forced.strip().lower()
        if state not in {"none", "moderate", "high", "critical", "unknown"}:
            state = "unknown"
        active = state in {"high", "critical"}
        snapshot = LocalContentionSnapshot(
            True,
            active,
            state,
            [],
            [f"local_contention_forced_{state}"],
            observed_at,
            round((time.perf_counter() - started) * 1000.0, 2),
            int(_env_float("PITH_LOCAL_CONTENTION_SIGNAL_TTL_SECONDS", 600.0)),
        )
        _snapshot_cache.update({"loaded_at": now_mono, "signature": signature, "snapshot": snapshot})
        return snapshot

    ttl = max(1.0, _env_float("PITH_LOCAL_CONTENTION_SIGNAL_TTL_SECONDS", 600.0))
    cutoff = now_mono - ttl
    with _lock:
        fresh_by_metric = {
            metric: [sample for sample in bucket if sample.monotonic_at >= cutoff]
            for metric, bucket in _samples.items()
        }
    signals = [
        signal
        for metric, fresh in fresh_by_metric.items()
        if (signal := _classify_metric(metric, fresh, now_mono=now_mono)) is not None
    ]
    severity_rank = {"none": 0, "moderate": 1, "high": 2, "critical": 3, "unknown": -1}
    state = "none"
    for signal in signals:
        if severity_rank[signal.state] > severity_rank[state]:
            state = signal.state
    reason_codes = list(dict.fromkeys(code for signal in signals for code in signal.reason_codes))
    snapshot = LocalContentionSnapshot(
        True,
        state in {"high", "critical"},
        state,
        [signal.to_dict() for signal in signals],
        reason_codes,
        observed_at,
        round((time.perf_counter() - started) * 1000.0, 2),
        int(ttl),
    )
    _snapshot_cache.update({"loaded_at": now_mono, "signature": signature, "snapshot": snapshot})
    return snapshot
