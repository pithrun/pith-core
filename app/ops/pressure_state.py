"""Operational pressure-state helpers for health and turn traces."""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.datetime_utils import _utc_now_iso
from app.core.profile import resolve_data_dir

LEASE_SCHEMA_VERSION = "maintenance_active_lease.v1"
LEASE_FILENAME = "maintenance_active_lease.json"
DEFAULT_LEASE_TTL_SECONDS = 900.0
_CACHE_TTL_SECONDS = 1.0
_MAX_MARKER_STRING_CHARS = 128

_lease_cache: dict[str, Any] = {
    "loaded_at": 0.0,
    "path": None,
    "mtime_ns": None,
    "size": None,
    "active": None,
    "stale": None,
}


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


def _parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _bounded_string(value: Any, default: str = "unknown") -> str:
    text = str(value or default)
    if len(text) <= _MAX_MARKER_STRING_CHARS:
        return text
    return text[: _MAX_MARKER_STRING_CHARS - 3] + "..."


def _safe_source(value: Any) -> str:
    text = _bounded_string(value, "unknown")
    allowed = {"external_launchd", "manual_cli", "manual", "api", "unknown"}
    return text if text in allowed else "unknown"


def _safe_phase(value: Any) -> str:
    text = _bounded_string(value, "maintenance")
    allowed = {"maintenance", "reflection", "scheduled_tasks", "experiments", "curiosity", "health_report", "unknown"}
    return text if text in allowed else "maintenance"


@dataclass(frozen=True)
class MaintenanceLease:
    schema_version: str
    run_id: str
    source: str
    pid: int
    phase: str
    started_at: str
    updated_at: str
    expected_timeout_seconds: float
    ttl_seconds: float
    command: str
    dry_run: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PressureState:
    active_contention: bool
    active_sources: list[str]
    in_process_reflection: dict[str, Any] | None
    external_maintenance: dict[str, Any] | None
    stale_external_maintenance: dict[str, Any] | None
    host_pressure: dict[str, Any] | None
    local_service_contention: dict[str, Any] | None
    pressure_level: str
    reason_codes: list[str]
    observed_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def maintenance_lease_path(data_dir: Path | None = None) -> Path:
    base = Path(data_dir) if data_dir is not None else Path(resolve_data_dir())
    return base / LEASE_FILENAME


def write_maintenance_lease(
    *,
    source: str,
    phase: str = "maintenance",
    command: str = "app.ops.maintenance_cli run",
    expected_timeout_seconds: float = 420.0,
    ttl_seconds: float | None = None,
    dry_run: bool = False,
    run_id: str | None = None,
) -> MaintenanceLease:
    now = _utc_now_iso()
    lease = MaintenanceLease(
        schema_version=LEASE_SCHEMA_VERSION,
        run_id=run_id or f"maint-{now}-{os.getpid()}-{uuid.uuid4().hex[:8]}",
        source=_safe_source(source),
        pid=os.getpid(),
        phase=_safe_phase(phase),
        started_at=now,
        updated_at=now,
        expected_timeout_seconds=float(expected_timeout_seconds),
        ttl_seconds=float(ttl_seconds if ttl_seconds is not None else _env_float(
            "PITH_MAINTENANCE_ACTIVE_LEASE_TTL_SECONDS",
            DEFAULT_LEASE_TTL_SECONDS,
        )),
        command=_bounded_string(command, "app.ops.maintenance_cli run"),
        dry_run=bool(dry_run),
    )
    path = maintenance_lease_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent), delete=False) as handle:
        json.dump(lease.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_name = handle.name
    os.replace(temp_name, path)
    _invalidate_cache()
    return lease


def clear_maintenance_lease(*, run_id: str, pid: int | None = None) -> bool:
    path = maintenance_lease_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        _invalidate_cache()
        return False
    except Exception:
        _invalidate_cache()
        return False
    if payload.get("run_id") != run_id:
        return False
    if pid is not None:
        try:
            marker_pid = int(payload.get("pid") or -1)
        except (TypeError, ValueError):
            return False
        if marker_pid != int(pid):
            return False
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    _invalidate_cache()
    return True


def _invalidate_cache() -> None:
    _lease_cache.update({"loaded_at": 0.0, "path": None, "mtime_ns": None, "size": None, "active": None, "stale": None})


def _validate_lease_payload(payload: Any, *, now: datetime) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not isinstance(payload, dict):
        return None, {"state": "invalid", "reason": "invalid_json_shape"}
    if payload.get("schema_version") != LEASE_SCHEMA_VERSION:
        return None, {"state": "invalid", "reason": "unsupported_schema_version"}
    updated = _parse_iso(payload.get("updated_at"))
    if updated is None:
        return None, {"state": "invalid", "reason": "invalid_updated_at"}
    try:
        ttl_seconds = float(payload.get("ttl_seconds", DEFAULT_LEASE_TTL_SECONDS))
        pid = int(payload.get("pid"))
    except (TypeError, ValueError):
        return None, {"state": "invalid", "reason": "invalid_numeric_field"}
    age_seconds = max(0.0, (now - updated).total_seconds())
    clean = {
        "schema_version": LEASE_SCHEMA_VERSION,
        "run_id": _bounded_string(payload.get("run_id")),
        "source": _safe_source(payload.get("source")),
        "pid": pid,
        "phase": _safe_phase(payload.get("phase")),
        "started_at": _bounded_string(payload.get("started_at")),
        "updated_at": _bounded_string(payload.get("updated_at")),
        "expected_timeout_seconds": float(payload.get("expected_timeout_seconds") or 0.0),
        "ttl_seconds": ttl_seconds,
        "command": _bounded_string(payload.get("command"), "app.ops.maintenance_cli run"),
        "dry_run": bool(payload.get("dry_run", False)),
        "age_seconds": round(age_seconds, 2),
    }
    if age_seconds > ttl_seconds:
        stale = dict(clean)
        stale["state"] = "stale"
        return None, stale
    clean["state"] = "active"
    return clean, None


def read_maintenance_lease(*, now: datetime | None = None, use_cache: bool = True) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    path = maintenance_lease_path()
    current_mono = time.monotonic()
    try:
        stat = path.stat()
    except FileNotFoundError:
        if use_cache and _lease_cache.get("path") == str(path) and current_mono - float(_lease_cache.get("loaded_at") or 0.0) <= _CACHE_TTL_SECONDS:
            return _lease_cache.get("active"), _lease_cache.get("stale")
        _lease_cache.update({"loaded_at": current_mono, "path": str(path), "mtime_ns": None, "size": None, "active": None, "stale": None})
        return None, None

    if (
        use_cache
        and _lease_cache.get("path") == str(path)
        and _lease_cache.get("mtime_ns") == stat.st_mtime_ns
        and _lease_cache.get("size") == stat.st_size
        and current_mono - float(_lease_cache.get("loaded_at") or 0.0) <= _CACHE_TTL_SECONDS
    ):
        return _lease_cache.get("active"), _lease_cache.get("stale")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        active, stale = _validate_lease_payload(payload, now=now or datetime.now(UTC))
    except Exception:
        active, stale = None, {"state": "invalid", "reason": "invalid_marker_json"}
    _lease_cache.update(
        {
            "loaded_at": current_mono,
            "path": str(path),
            "mtime_ns": stat.st_mtime_ns,
            "size": stat.st_size,
            "active": active,
            "stale": stale,
        }
    )
    return active, stale


def build_pressure_state(
    *,
    active_reflection: dict[str, Any] | None = None,
    use_cache: bool = True,
) -> PressureState:
    observed_at = _utc_now_iso()
    if not _env_bool("PITH_PRESSURE_STATE_ENABLED", True):
        return PressureState(
            active_contention=False,
            active_sources=[],
            in_process_reflection=active_reflection,
            external_maintenance=None,
            stale_external_maintenance=None,
            host_pressure=None,
            local_service_contention=None,
            pressure_level="none",
            reason_codes=["pressure_state_disabled"],
            observed_at=observed_at,
        )

    external, stale = read_maintenance_lease(use_cache=use_cache)
    try:
        from app.ops.host_pressure import build_host_pressure_snapshot

        host_pressure = build_host_pressure_snapshot(use_cache=use_cache).to_dict()
    except Exception:
        host_pressure = {
            "enabled": True,
            "state": "unknown",
            "reason_codes": ["host_pressure_unavailable"],
        }
    try:
        from app.ops.local_contention import build_local_contention_snapshot

        local_service_contention = build_local_contention_snapshot(use_cache=use_cache).to_dict()
    except Exception:
        local_service_contention = {
            "enabled": True,
            "active": False,
            "state": "unknown",
            "reason_codes": ["local_contention_unavailable"],
        }
    sources: list[str] = []
    reason_codes: list[str] = []
    if active_reflection:
        sources.append("in_process_reflection")
        reason_codes.append("in_process_reflection_active")
    if external:
        sources.append("external_maintenance")
        reason_codes.append("external_maintenance_active")
    if stale:
        reason_codes.append("external_maintenance_stale")
    host_state = str(host_pressure.get("state") or "unknown") if isinstance(host_pressure, dict) else "unknown"
    if isinstance(host_pressure, dict):
        reason_codes.extend(str(code) for code in host_pressure.get("reason_codes") or [])
    if host_state in {"high", "critical"}:
        sources.append("host_resource_pressure")
    local_state = (
        str(local_service_contention.get("state") or "unknown")
        if isinstance(local_service_contention, dict)
        else "unknown"
    )
    if isinstance(local_service_contention, dict):
        reason_codes.extend(str(code) for code in local_service_contention.get("reason_codes") or [])
    if local_state in {"high", "critical"}:
        sources.append("local_service_contention")

    active_contention = bool(sources)
    if host_state == "critical" or local_state == "critical":
        pressure_level = "critical"
    elif active_reflection or external or host_state == "high" or local_state == "high":
        pressure_level = "high"
    elif host_state == "moderate" or local_state == "moderate":
        pressure_level = "moderate"
    else:
        pressure_level = "none"
    return PressureState(
        active_contention=active_contention,
        active_sources=sources,
        in_process_reflection=dict(active_reflection) if isinstance(active_reflection, dict) else None,
        external_maintenance=external,
        stale_external_maintenance=stale,
        host_pressure=host_pressure,
        local_service_contention=local_service_contention,
        pressure_level=pressure_level,
        reason_codes=list(dict.fromkeys(reason_codes)),
        observed_at=observed_at,
    )


def pressure_metric_labels(state: PressureState | dict[str, Any] | None) -> dict[str, str]:
    if hasattr(state, "to_dict"):
        payload = state.to_dict()  # type: ignore[union-attr]
    elif isinstance(state, dict):
        payload = state
    else:
        payload = {}
    sources = list(payload.get("active_sources") or [])
    if len(sources) > 1:
        source = "multiple"
    elif sources:
        source = str(sources[0])
    elif payload.get("stale_external_maintenance"):
        source = "stale_external_maintenance"
    else:
        source = "none"
    allowed_sources = {
        "none",
        "in_process_reflection",
        "external_maintenance",
        "multiple",
        "stale_external_maintenance",
        "host_resource_pressure",
        "local_service_contention",
    }
    host_pressure = payload.get("host_pressure") if isinstance(payload.get("host_pressure"), dict) else {}
    return {
        "pressure_level": str(payload.get("pressure_level") or "unknown"),
        "active_contention": str(bool(payload.get("active_contention"))).lower(),
        "active_contention_source": source if source in allowed_sources else "unknown",
        "host_pressure_state": str(host_pressure.get("state") or "unknown"),
        "host_memory_state": str(host_pressure.get("memory_state") or "unknown"),
    }
