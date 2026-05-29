"""Session-local cache for required conversation-turn context.

PERF-086 keeps small answer-path turns from blocking on required instruction
loads after the optional gates have already fired.
"""

from __future__ import annotations

import concurrent.futures
import copy
import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

DEFAULT_TTL_MS = 5000.0
DEFAULT_MAX_STALE_MS = 60000.0
DEFAULT_SERVING_MAX_STALE_MS = 300000.0
DEFAULT_HEALTH_MAX_AGE_MS = 60000.0
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RequiredContextPayload:
    always_on: list[dict[str, Any]]
    firmware_entries: list[dict[str, Any]]
    directives_response: dict[str, Any]


@dataclass(frozen=True)
class RequiredContextStats:
    state: str
    age_ms: float | None = None
    refresh_ms: float = 0.0
    always_activate_ms: float = 0.0
    firmware_ms: float = 0.0
    directives_ms: float = 0.0
    error: str | None = None
    refresh_scheduled: bool = False
    refresh_in_flight: bool = False
    component_errors: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RequiredContextStatus:
    state: str
    age_ms: float | None
    ttl_ms: float
    health_max_age_ms: float
    serving_max_stale_ms: float
    refresh_after_ms: float
    servable: bool
    refresh_in_flight: bool = False


@dataclass(frozen=True)
class _CacheEntry:
    payload: RequiredContextPayload
    loaded_at: float


@dataclass(frozen=True)
class _RefreshResult:
    payload: RequiredContextPayload
    loaded_at: float
    stats: RequiredContextStats
    update_cache: bool = True


@dataclass(frozen=True)
class RequiredContextLoaders:
    load_always_activate: Callable[[], list[dict[str, Any]]]
    load_firmware: Callable[[], list[dict[str, Any]]]
    load_directives_budgeted: Callable[[], dict[str, Any]]


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _serving_max_stale_ms(default: float | None = None) -> float:
    if default is not None:
        return default
    if "PITH_REQUIRED_CONTEXT_CACHE_SERVING_MAX_STALE_MS" in os.environ:
        return _env_float(
            "PITH_REQUIRED_CONTEXT_CACHE_SERVING_MAX_STALE_MS",
            DEFAULT_SERVING_MAX_STALE_MS,
        )
    if "PITH_REQUIRED_CONTEXT_CACHE_MAX_STALE_MS" in os.environ:
        return _env_float("PITH_REQUIRED_CONTEXT_CACHE_MAX_STALE_MS", DEFAULT_MAX_STALE_MS)
    return DEFAULT_SERVING_MAX_STALE_MS


def _refresh_after_ms(serving_max_stale_ms: float, health_max_age_ms: float) -> float:
    configured = os.environ.get("PITH_REQUIRED_CONTEXT_CACHE_REFRESH_AFTER_MS")
    if configured is not None:
        try:
            return max(0.0, float(configured))
        except (TypeError, ValueError):
            pass
    return max(0.0, min(serving_max_stale_ms / 2.0, health_max_age_ms))


def _copy_payload(payload: RequiredContextPayload) -> RequiredContextPayload:
    return RequiredContextPayload(
        always_on=copy.deepcopy(payload.always_on),
        firmware_entries=copy.deepcopy(payload.firmware_entries),
        directives_response=copy.deepcopy(payload.directives_response),
    )


def _default_directives_response() -> dict[str, Any]:
    return {"directives": [], "budget_warning": None}


def _default_payload() -> RequiredContextPayload:
    return RequiredContextPayload(
        always_on=[],
        firmware_entries=[],
        directives_response=_default_directives_response(),
    )


def _default_loaders() -> RequiredContextLoaders:
    from app.governance.directives import load_directives_budgeted
    from app.storage import load_always_activate_concepts, load_firmware

    return RequiredContextLoaders(
        load_always_activate=load_always_activate_concepts,
        load_firmware=load_firmware,
        load_directives_budgeted=load_directives_budgeted,
    )


class RequiredContextCache:
    """Short-lived cache for required instruction payloads."""

    def __init__(
        self,
        *,
        time_fn: Callable[[], float] | None = None,
        ttl_ms: float | None = None,
        max_stale_ms: float | None = None,
    ) -> None:
        self._time_fn = time_fn or time.perf_counter
        self._ttl_ms = ttl_ms
        self._max_stale_ms = max_stale_ms
        self._entry: _CacheEntry | None = None
        self._lock = threading.RLock()
        self._refresh_executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._refresh_future: concurrent.futures.Future | None = None
        self._shutdown = False

    def clear(self) -> None:
        with self._lock:
            self._entry = None

    def shutdown(self, *, wait: bool = True) -> None:
        with self._lock:
            self._shutdown = True
            executor = self._refresh_executor
            self._refresh_executor = None
            self._refresh_future = None
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=True)

    def is_warm(self, *, max_age_ms: float | None = None) -> bool:
        """Return whether a required-context payload is already cached."""
        with self._lock:
            entry = self._entry
            if entry is None:
                return False
            if max_age_ms is None:
                return True
            age_ms = (self._time_fn() - entry.loaded_at) * 1000.0
            return age_ms <= max_age_ms

    def status(
        self,
        *,
        health_max_age_ms: float | None = None,
        serving_max_stale_ms: float | None = None,
    ) -> RequiredContextStatus:
        """Return detailed cache status without mutating cache state."""
        ttl_ms = self._ttl_ms if self._ttl_ms is not None else _env_float(
            "PITH_REQUIRED_CONTEXT_CACHE_TTL_MS",
            DEFAULT_TTL_MS,
        )
        health_window_ms = (
            health_max_age_ms
            if health_max_age_ms is not None
            else _env_float(
                "PITH_REQUIRED_CONTEXT_CACHE_HEALTH_MAX_AGE_MS",
                DEFAULT_HEALTH_MAX_AGE_MS,
            )
        )
        serving_window_ms = (
            serving_max_stale_ms
            if serving_max_stale_ms is not None
            else _serving_max_stale_ms(self._max_stale_ms)
        )
        refresh_after = _refresh_after_ms(serving_window_ms, health_window_ms)
        with self._lock:
            entry = self._entry
            refresh_in_flight = self._has_refresh_in_flight_locked()
            if entry is None:
                return RequiredContextStatus(
                    state="empty",
                    age_ms=None,
                    ttl_ms=ttl_ms,
                    health_max_age_ms=health_window_ms,
                    serving_max_stale_ms=serving_window_ms,
                    refresh_after_ms=refresh_after,
                    servable=False,
                    refresh_in_flight=refresh_in_flight,
                )
            age_ms = (self._time_fn() - entry.loaded_at) * 1000.0
            rounded_age = round(age_ms, 2)
            if age_ms <= health_window_ms:
                state = "fresh"
                servable = True
            elif age_ms <= serving_window_ms:
                state = "stale_but_servable"
                servable = True
            else:
                state = "cold_degraded"
                servable = False
            return RequiredContextStatus(
                state=state,
                age_ms=rounded_age,
                ttl_ms=ttl_ms,
                health_max_age_ms=health_window_ms,
                serving_max_stale_ms=serving_window_ms,
                refresh_after_ms=refresh_after,
                servable=servable,
                refresh_in_flight=refresh_in_flight,
            )

    def prewarm(
        self,
        *,
        loaders: RequiredContextLoaders | None = None,
    ) -> RequiredContextStats:
        """Populate the cache before the first user-facing turn."""
        loaders = loaders or _default_loaders()
        max_stale_ms = _serving_max_stale_ms(self._max_stale_ms)
        with self._lock:
            if self._shutdown:
                return RequiredContextStats(state="shutdown")
            previous = self._entry
            previous_age_ms = (
                (self._time_fn() - previous.loaded_at) * 1000.0
                if previous is not None
                else None
            )
        result = self._refresh_without_lock(loaders, previous, previous_age_ms, max_stale_ms)
        if result.update_cache:
            with self._lock:
                if not self._shutdown:
                    self._entry = _CacheEntry(payload=result.payload, loaded_at=result.loaded_at)
                else:
                    return RequiredContextStats(state="shutdown")
        return result.stats

    def get(
        self,
        *,
        prefer_stale_fallback: bool,
        stale_first: bool = False,
        background_refresh: bool = True,
        allow_sync_refresh: bool = True,
        loaders: RequiredContextLoaders | None = None,
    ) -> tuple[RequiredContextPayload, RequiredContextStats]:
        loaders = loaders or _default_loaders()
        ttl_ms = self._ttl_ms if self._ttl_ms is not None else _env_float(
            "PITH_REQUIRED_CONTEXT_CACHE_TTL_MS",
            DEFAULT_TTL_MS,
        )
        max_stale_ms = _serving_max_stale_ms(self._max_stale_ms)
        health_max_age_ms = _env_float(
            "PITH_REQUIRED_CONTEXT_CACHE_HEALTH_MAX_AGE_MS",
            DEFAULT_HEALTH_MAX_AGE_MS,
        )
        refresh_after_ms = _refresh_after_ms(max_stale_ms, health_max_age_ms)
        now = self._time_fn()
        with self._lock:
            entry = self._entry
            age_ms = ((now - entry.loaded_at) * 1000.0) if entry is not None else None
            if entry is not None and age_ms is not None and age_ms <= ttl_ms:
                return _copy_payload(entry.payload), RequiredContextStats(
                    state="fresh_hit",
                    age_ms=round(age_ms, 2),
                    refresh_in_flight=self._has_refresh_in_flight_locked(),
                )
            stale_eligible = (
                entry is not None
                and age_ms is not None
                and age_ms <= max_stale_ms
            )
            if stale_first and stale_eligible:
                scheduled = (
                    self._schedule_refresh_locked(loaders, max_stale_ms)
                    if background_refresh
                    else False
                )
                return _copy_payload(entry.payload), RequiredContextStats(
                    state="stale_first",
                    age_ms=round(age_ms, 2),
                    refresh_scheduled=scheduled,
                    refresh_in_flight=not scheduled and self._has_refresh_in_flight_locked(),
                )
            if (
                prefer_stale_fallback
                and stale_eligible
            ):
                scheduled = (
                    self._schedule_refresh_locked(loaders, max_stale_ms)
                    if background_refresh
                    and age_ms is not None
                    and age_ms >= refresh_after_ms
                    else False
                )
                return _copy_payload(entry.payload), RequiredContextStats(
                    state="stale_fallback",
                    age_ms=round(age_ms, 2),
                    refresh_scheduled=scheduled,
                    refresh_in_flight=not scheduled and self._has_refresh_in_flight_locked(),
                )
            if not allow_sync_refresh:
                scheduled = (
                    self._schedule_refresh_locked(loaders, max_stale_ms)
                    if background_refresh
                    else False
                )
                return _default_payload(), RequiredContextStats(
                    state="cold_degraded",
                    age_ms=round(age_ms, 2) if age_ms is not None else None,
                    error="sync_refresh_disabled",
                    refresh_scheduled=scheduled,
                    refresh_in_flight=not scheduled and self._has_refresh_in_flight_locked(),
                )
        result = self._refresh_without_lock(loaders, entry, age_ms, max_stale_ms)
        if result.update_cache:
            with self._lock:
                if not self._shutdown:
                    self._entry = _CacheEntry(payload=result.payload, loaded_at=result.loaded_at)
        return _copy_payload(result.payload), result.stats

    def _has_refresh_in_flight_locked(self) -> bool:
        return self._refresh_future is not None and not self._refresh_future.done()

    def _ensure_refresh_executor_locked(self) -> concurrent.futures.ThreadPoolExecutor:
        if self._refresh_executor is None:
            self._refresh_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="pith-required-context-refresh",
            )
        return self._refresh_executor

    def _schedule_refresh_locked(
        self,
        loaders: RequiredContextLoaders,
        max_stale_ms: float,
    ) -> bool:
        if self._shutdown or self._has_refresh_in_flight_locked():
            return False
        executor = self._ensure_refresh_executor_locked()
        future = executor.submit(self._refresh_background, loaders, max_stale_ms)
        self._refresh_future = future
        future.add_done_callback(self._on_refresh_done)
        return True

    def _refresh_background(
        self,
        loaders: RequiredContextLoaders,
        max_stale_ms: float,
    ) -> None:
        with self._lock:
            if self._shutdown:
                return
            previous = self._entry
            previous_age_ms = (
                (self._time_fn() - previous.loaded_at) * 1000.0
                if previous is not None
                else None
            )
        result = self._refresh_without_lock(loaders, previous, previous_age_ms, max_stale_ms)
        if result.update_cache:
            with self._lock:
                if not self._shutdown:
                    self._entry = _CacheEntry(payload=result.payload, loaded_at=result.loaded_at)

    def _on_refresh_done(self, future: concurrent.futures.Future) -> None:
        try:
            exc = None if future.cancelled() else future.exception()
            if exc is not None:
                logger.warning("Required context background refresh failed: %s", exc)
        except Exception as exc:
            logger.warning("Required context background refresh status failed: %s", exc)
        finally:
            with self._lock:
                if self._refresh_future is future:
                    self._refresh_future = None

    def _refresh_without_lock(
        self,
        loaders: RequiredContextLoaders,
        previous: _CacheEntry | None,
        previous_age_ms: float | None,
        max_stale_ms: float,
    ) -> _RefreshResult:
        refresh_start = self._time_fn()
        component_errors: dict[str, str] = {}
        always_ms = 0.0
        firmware_ms = 0.0
        directives_ms = 0.0
        try:
            started = self._time_fn()
            always_on = loaders.load_always_activate()
            always_ms = (self._time_fn() - started) * 1000.0

            started = self._time_fn()
            firmware_entries = loaders.load_firmware()
            firmware_ms = (self._time_fn() - started) * 1000.0
        except Exception as exc:
            if (
                previous is not None
                and previous_age_ms is not None
                and previous_age_ms <= max_stale_ms
            ):
                refresh_ms = (self._time_fn() - refresh_start) * 1000.0
                return _RefreshResult(
                    payload=previous.payload,
                    loaded_at=previous.loaded_at,
                    update_cache=False,
                    stats=RequiredContextStats(
                        state="refresh_error",
                        age_ms=round(previous_age_ms, 2),
                        refresh_ms=round(refresh_ms, 2),
                        always_activate_ms=round(always_ms, 2),
                        firmware_ms=round(firmware_ms, 2),
                        directives_ms=round(directives_ms, 2),
                        error=type(exc).__name__,
                        component_errors={"required_context": type(exc).__name__},
                    ),
                )
            raise

        started = self._time_fn()
        try:
            directives_response = loaders.load_directives_budgeted()
        except Exception as exc:
            component_errors["directives"] = type(exc).__name__
            if (
                previous is not None
                and previous_age_ms is not None
                and previous_age_ms <= max_stale_ms
            ):
                directives_response = copy.deepcopy(previous.payload.directives_response)
            else:
                directives_response = _default_directives_response()
        directives_ms = (self._time_fn() - started) * 1000.0

        payload = RequiredContextPayload(
            always_on=copy.deepcopy(always_on),
            firmware_entries=copy.deepcopy(firmware_entries),
            directives_response=copy.deepcopy(directives_response),
        )
        loaded_at = self._time_fn()
        refresh_ms = (loaded_at - refresh_start) * 1000.0
        state = "refresh_error" if component_errors else ("cold_miss" if previous is None else "refresh")
        return _RefreshResult(
            payload=payload,
            loaded_at=loaded_at,
            stats=RequiredContextStats(
                state=state,
                age_ms=0.0,
                refresh_ms=round(refresh_ms, 2),
                always_activate_ms=round(always_ms, 2),
                firmware_ms=round(firmware_ms, 2),
                directives_ms=round(directives_ms, 2),
                error=",".join(sorted(component_errors.values())) if component_errors else None,
                component_errors=component_errors,
            ),
        )


_GLOBAL_CACHE = RequiredContextCache()


def get_required_context(
    *,
    prefer_stale_fallback: bool,
    stale_first: bool = False,
    background_refresh: bool = True,
    allow_sync_refresh: bool = True,
    loaders: RequiredContextLoaders | None = None,
) -> tuple[RequiredContextPayload, RequiredContextStats]:
    """Return required context using the process-local cache."""
    return _GLOBAL_CACHE.get(
        prefer_stale_fallback=prefer_stale_fallback,
        stale_first=stale_first,
        background_refresh=background_refresh,
        allow_sync_refresh=allow_sync_refresh,
        loaders=loaders,
    )


def prewarm_required_context(
    *,
    loaders: RequiredContextLoaders | None = None,
) -> RequiredContextStats:
    """Populate process-local required context before serving turns."""
    return _GLOBAL_CACHE.prewarm(loaders=loaders)


def required_context_cache_status(
    *,
    health_max_age_ms: float | None = None,
    serving_max_stale_ms: float | None = None,
) -> RequiredContextStatus:
    """Return detailed process-local required-context cache status."""
    return _GLOBAL_CACHE.status(
        health_max_age_ms=health_max_age_ms,
        serving_max_stale_ms=serving_max_stale_ms,
    )


def is_required_context_cache_warm(*, max_age_ms: float | None = None) -> bool:
    """Return whether process-local required context has a cached payload."""
    return _GLOBAL_CACHE.is_warm(max_age_ms=max_age_ms)


def clear_required_context_cache() -> None:
    """Clear process-local required context cache for tests and maintenance."""
    _GLOBAL_CACHE.clear()


def shutdown_required_context_cache(*, wait: bool = True) -> None:
    """Stop required-context background refresh work during server shutdown."""
    _GLOBAL_CACHE.shutdown(wait=wait)
