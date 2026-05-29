"""Runtime policy for answer-path enforcement.

The policy is process-local by design. It provides a reversible canary control
surface without persisting enforcement state across restarts.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta

from app.core.datetime_utils import _ensure_aware, _utc_now

_TRUE_VALUES = {"1", "true", "yes", "on"}
_SOURCE_ALLOWLIST = {"env", "runtime_api", "canary", "test"}
_MODE_ALLOWLIST = {"small", "standard", "deep", "first_call_resumption"}
_DEFAULT_ENFORCE_MODES = ("small", "standard")


class AnswerPathPolicyError(ValueError):
    """Raised when a runtime answer-path policy request is invalid."""


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUE_VALUES


def _max_ttl_seconds() -> int:
    try:
        return max(1, int(os.environ.get("PITH_ANSWER_PATH_POLICY_MAX_TTL_SECONDS", "900")))
    except (TypeError, ValueError):
        return 900


def _sanitize_source(source: str | None) -> str:
    candidate = (source or "runtime_api").strip()
    return candidate if candidate in _SOURCE_ALLOWLIST else "runtime_api"


def _sanitize_modes(modes: object | None) -> tuple[str, ...]:
    if modes is None:
        return _DEFAULT_ENFORCE_MODES
    if isinstance(modes, str):
        raw_modes = modes.split(",")
    else:
        try:
            raw_modes = list(modes)  # type: ignore[arg-type]
        except TypeError:
            return _DEFAULT_ENFORCE_MODES
    cleaned: list[str] = []
    for raw_mode in raw_modes:
        mode = str(raw_mode).strip().lower()
        if mode in _MODE_ALLOWLIST and mode not in cleaned:
            cleaned.append(mode)
    return tuple(cleaned)


def _env_enforce_modes() -> tuple[str, ...]:
    return _sanitize_modes(os.environ.get("PITH_ANSWER_PATH_ENFORCE_MODES"))


@dataclass(frozen=True)
class AnswerPathPolicySnapshot:
    """Immutable view of answer-path runtime policy."""

    observe_only: bool
    enforcement_enabled: bool
    source: str
    state: str
    generation: int
    enforce_modes: tuple[str, ...] = _DEFAULT_ENFORCE_MODES
    expires_at: str | None = None
    runtime_active: bool = False

    def labels(self) -> dict[str, str]:
        """Return low-cardinality labels safe for operational metrics."""
        return {
            "policy_state": self.state,
            "policy_source": self.source,
            "policy_runtime_active": str(self.runtime_active).lower(),
            "policy_enforce_modes": "+".join(self.enforce_modes) if self.enforce_modes else "none",
        }

    def mode_enforced(self, mode: str | None) -> bool:
        """Return whether answer-path enforcement applies to a classified mode."""
        return (mode or "").strip().lower() in set(self.enforce_modes)


class AnswerPathRuntimePolicy:
    """Thread-safe process-local answer-path policy."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._generation = 0
        self._runtime_observe_only: bool | None = None
        self._runtime_enforcement_enabled: bool | None = None
        self._runtime_enforce_modes: tuple[str, ...] | None = None
        self._runtime_source: str | None = None
        self._runtime_expires_at: datetime | None = None

    def _default_snapshot_locked(self) -> AnswerPathPolicySnapshot:
        observe_only = _env_bool("PITH_ANSWER_PATH_OBSERVE_ONLY", True)
        enforcement_enabled = _env_bool("PITH_ANSWER_PATH_ENFORCEMENT", False)
        return AnswerPathPolicySnapshot(
            observe_only=observe_only,
            enforcement_enabled=enforcement_enabled,
            source="env",
            state="env_enforced" if enforcement_enabled and not observe_only else "env_observe_only",
            generation=self._generation,
            enforce_modes=_env_enforce_modes(),
            runtime_active=False,
        )

    def _runtime_snapshot_locked(self) -> AnswerPathPolicySnapshot:
        expires_at = self._runtime_expires_at.isoformat() if self._runtime_expires_at else None
        observe_only = bool(self._runtime_observe_only)
        enforcement_enabled = bool(self._runtime_enforcement_enabled)
        return AnswerPathPolicySnapshot(
            observe_only=observe_only,
            enforcement_enabled=enforcement_enabled,
            source=self._runtime_source or "runtime_api",
            state="runtime_enforced" if enforcement_enabled and not observe_only else "runtime_observe_only",
            generation=self._generation,
            enforce_modes=self._runtime_enforce_modes or _env_enforce_modes(),
            expires_at=expires_at,
            runtime_active=True,
        )

    def _clear_runtime_locked(self) -> None:
        self._runtime_observe_only = None
        self._runtime_enforcement_enabled = None
        self._runtime_enforce_modes = None
        self._runtime_source = None
        self._runtime_expires_at = None

    def snapshot(self) -> AnswerPathPolicySnapshot:
        """Return the active policy, expiring runtime overrides if needed."""
        with self._lock:
            if self._runtime_observe_only is None:
                return self._default_snapshot_locked()
            if self._runtime_expires_at and _utc_now() >= self._runtime_expires_at:
                self._clear_runtime_locked()
                self._generation += 1
                return self._default_snapshot_locked()
            return self._runtime_snapshot_locked()

    def set_runtime(
        self,
        *,
        observe_only: bool,
        enforcement_enabled: bool,
        ttl_seconds: int,
        enforce_modes: object | None = None,
        source: str | None = None,
    ) -> AnswerPathPolicySnapshot:
        """Install a bounded runtime policy override."""
        if not isinstance(observe_only, bool) or not isinstance(enforcement_enabled, bool):
            raise AnswerPathPolicyError("observe_only and enforcement_enabled must be strict booleans")
        if not isinstance(ttl_seconds, int) or isinstance(ttl_seconds, bool):
            raise AnswerPathPolicyError("ttl_seconds must be a strict integer")
        max_ttl = _max_ttl_seconds()
        if ttl_seconds < 1 or ttl_seconds > max_ttl:
            raise AnswerPathPolicyError(f"ttl_seconds must be between 1 and {max_ttl}")
        if not observe_only and not enforcement_enabled:
            raise AnswerPathPolicyError("enforcement_enabled must be true when observe_only is false")

        with self._lock:
            self._generation += 1
            self._runtime_observe_only = observe_only
            self._runtime_enforcement_enabled = enforcement_enabled
            self._runtime_enforce_modes = _sanitize_modes(enforce_modes)
            self._runtime_source = _sanitize_source(source)
            self._runtime_expires_at = _ensure_aware(_utc_now() + timedelta(seconds=ttl_seconds))
            return self._runtime_snapshot_locked()

    def reset(self, *, source: str | None = None) -> AnswerPathPolicySnapshot:
        """Clear runtime override and return the fallback policy."""
        with self._lock:
            self._generation += 1
            self._clear_runtime_locked()
            snapshot = self._default_snapshot_locked()
            if source:
                return AnswerPathPolicySnapshot(
                    observe_only=snapshot.observe_only,
                    enforcement_enabled=snapshot.enforcement_enabled,
                    source=_sanitize_source(source),
                    state=snapshot.state,
                    generation=snapshot.generation,
                    enforce_modes=snapshot.enforce_modes,
                    expires_at=snapshot.expires_at,
                    runtime_active=snapshot.runtime_active,
                )
            return snapshot

    def max_ttl_seconds(self) -> int:
        """Return the configured maximum runtime override TTL."""
        return _max_ttl_seconds()


_POLICY = AnswerPathRuntimePolicy()


def get_answer_path_policy() -> AnswerPathRuntimePolicy:
    """Return the process-local answer-path policy singleton."""
    return _POLICY
