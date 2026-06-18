"""Shared pressure-mode policy helpers."""

from __future__ import annotations

import os
from enum import StrEnum
from typing import Any


class PressureWorkClass(StrEnum):
    REQUIRED_CORE = "required_core"
    REQUIRED_DEGRADED = "required_degraded"
    OPTIONAL_ENRICHMENT = "optional_enrichment"
    BACKGROUND_MAINTENANCE = "background_maintenance"


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def pressure_payload(state: Any | None) -> dict[str, Any]:
    if hasattr(state, "to_dict"):
        return state.to_dict()
    return state if isinstance(state, dict) else {}


def pressure_level(payload_or_state: Any | None) -> str:
    payload = pressure_payload(payload_or_state)
    return str(payload.get("pressure_level") or "none")


def foreground_pressure_mode(payload_or_state: Any | None) -> str:
    if not _env_bool("PITH_FOREGROUND_PRESSURE_MODE_ENABLED", True):
        return "off"
    payload = pressure_payload(payload_or_state)
    level = pressure_level(payload)
    host_state = str((payload.get("host_pressure") or {}).get("state") or "none")
    if _env_bool("PITH_HOST_PRESSURE_OBSERVE_ONLY", False) and host_state in {"moderate", "high", "critical"}:
        return "observe"
    if level == "critical" or host_state == "critical":
        return "critical"
    if level == "high" or host_state == "high" or payload.get("active_contention"):
        return "protected"
    return "off"


def should_defer_background_maintenance(payload_or_state: Any | None) -> bool:
    if not _env_bool("PITH_LIFECYCLE_PRESSURE_BACKPRESSURE_ENABLED", True):
        return False
    return foreground_pressure_mode(payload_or_state) in {"protected", "critical"}


def should_defer_health_diagnostics(payload_or_state: Any | None) -> bool:
    mode = foreground_pressure_mode(payload_or_state)
    return mode in {"protected", "critical"}
