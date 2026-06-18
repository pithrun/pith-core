"""Bounded host pressure sampling for runtime admission."""

from __future__ import annotations

import os
import platform
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Any

from app.core.datetime_utils import _utc_now_iso
from app.core.fork_safety import should_suppress_optional_subprocess

_CACHE_TTL_SECONDS = 1.0
_snapshot_cache: dict[str, Any] = {"loaded_at": 0.0, "signature": None, "snapshot": None}
_memory_counter_cache: dict[str, Any] = {"loaded_at": None, "pageouts": None, "swapouts": None}


@dataclass(frozen=True)
class HostPressureSnapshot:
    enabled: bool
    state: str
    load_1m: float | None
    load_per_core: float | None
    cpu_count: int | None
    memory_state: str | None
    reason_codes: list[str]
    observed_at: str
    sample_age_ms: float
    memory_signal_state: str | None = None
    memory_signal_confidence: str | None = None
    memory_artifact_reason_codes: list[str] | None = None
    memory_corrob_reason_codes: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _MemoryPressureResult:
    state: str | None
    reason_codes: list[str]
    signal_state: str | None = None
    confidence: str | None = None
    artifact_reason_codes: list[str] | None = None
    corrob_reason_codes: list[str] | None = None


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


_SEVERITY_RANK = {"unknown": -1, "none": 0, "moderate": 1, "high": 2, "critical": 3}


def _max_known_state(*states: str | None) -> str:
    best = "none"
    saw_unknown = False
    for state in states:
        clean = str(state or "none")
        if clean == "unknown":
            saw_unknown = True
            continue
        if _SEVERITY_RANK.get(clean, -1) > _SEVERITY_RANK[best]:
            best = clean
    if best == "none" and saw_unknown:
        return "unknown"
    return best


def _classify_load(load_per_core: float, reason_codes: list[str]) -> str:
    moderate = _env_float("PITH_HOST_PRESSURE_LOAD_PER_CORE_MODERATE", 1.25)
    high = _env_float("PITH_HOST_PRESSURE_LOAD_PER_CORE_HIGH", 1.75)
    critical = _env_float("PITH_HOST_PRESSURE_LOAD_PER_CORE_CRITICAL", 2.50)
    if load_per_core >= critical:
        reason_codes.append("host_load_critical")
        return "critical"
    if load_per_core >= high:
        reason_codes.append("host_load_high")
        return "high"
    if load_per_core >= moderate:
        reason_codes.append("host_load_moderate")
        return "moderate"
    return "none"


def _snapshot_cache_signature() -> tuple[str | None, ...]:
    return tuple(
        os.environ.get(name)
        for name in (
            "PITH_HOST_PRESSURE_ENABLED",
            "PITH_HOST_PRESSURE_FORCE_STATE",
            "PITH_HOST_PRESSURE_LOAD_PER_CORE_MODERATE",
            "PITH_HOST_PRESSURE_LOAD_PER_CORE_HIGH",
            "PITH_HOST_PRESSURE_LOAD_PER_CORE_CRITICAL",
            "PITH_HOST_PRESSURE_MEMORY_ENABLED",
            "PITH_HOST_PRESSURE_VM_STAT_TIMEOUT_MS",
            "PITH_HOST_PRESSURE_MEMORY_CRITICAL_SWAP_USED_RATIO",
            "PITH_HOST_PRESSURE_MEMORY_HIGH_SWAP_USED_RATIO",
            "PITH_HOST_PRESSURE_MEMORY_MODERATE_SWAP_USED_RATIO",
            "PITH_HOST_PRESSURE_MEMORY_CRITICAL_SWAP_FREE_MB",
            "PITH_HOST_PRESSURE_MEMORY_HIGH_COMPRESSOR_RATIO",
            "PITH_HOST_PRESSURE_MEMORY_MODERATE_FREE_RATIO",
            "PITH_HOST_PRESSURE_DARWIN_CORROBORATION_ENABLED",
            "PITH_HOST_PRESSURE_MEMORY_PRESSURE_FREE_PCT_HIGH",
            "PITH_HOST_PRESSURE_PAGEOUTS_PER_SEC_HIGH",
            "PITH_HOST_PRESSURE_SWAPOUTS_PER_SEC_HIGH",
        )
    )


def _run_memory_command(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=_env_float("PITH_HOST_PRESSURE_VM_STAT_TIMEOUT_MS", 100.0) / 1000.0,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def _parse_swapusage_mb(output: str) -> dict[str, float] | None:
    match = re.search(
        r"total\s*=\s*([0-9.]+)([KMG])\s+used\s*=\s*([0-9.]+)([KMG])\s+free\s*=\s*([0-9.]+)([KMG])",
        output,
        re.IGNORECASE,
    )
    if not match:
        return None

    def _to_mb(value: str, unit: str) -> float:
        amount = float(value)
        unit = unit.upper()
        if unit == "K":
            return amount / 1024.0
        if unit == "G":
            return amount * 1024.0
        return amount

    return {
        "total_mb": _to_mb(match.group(1), match.group(2)),
        "used_mb": _to_mb(match.group(3), match.group(4)),
        "free_mb": _to_mb(match.group(5), match.group(6)),
    }


def _parse_vm_stat(output: str) -> dict[str, int] | None:
    page_size_match = re.search(r"page size of\s+([0-9]+)\s+bytes", output, re.IGNORECASE)
    if not page_size_match:
        return None
    stats: dict[str, int] = {"page_size": int(page_size_match.group(1))}
    wanted = {
        "Pages free": "free_pages",
        "Pages speculative": "speculative_pages",
        "Pages occupied by compressor": "compressor_pages",
        "Pageouts": "pageouts",
        "Swapouts": "swapouts",
        "Pageins": "pageins",
        "Swapins": "swapins",
    }
    for line in output.splitlines():
        for prefix, key in wanted.items():
            if line.strip().startswith(prefix):
                value_match = re.search(r":\s*([0-9]+)", line)
                if value_match:
                    stats[key] = int(value_match.group(1))
    return stats


def _physical_memory_mb() -> float | None:
    try:
        return (float(os.sysconf("SC_PAGE_SIZE")) * float(os.sysconf("SC_PHYS_PAGES"))) / (1024.0 * 1024.0)
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _parse_memory_pressure_free_pct(output: str) -> int | None:
    match = re.search(r"System-wide memory free percentage:\s*([0-9]+)%", output, re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _vm_counter_rates(vm_stats: dict[str, int] | None) -> dict[str, float | None]:
    if not vm_stats:
        return {"pageouts_per_sec": None, "swapouts_per_sec": None}
    now = time.monotonic()
    previous_at = _memory_counter_cache.get("loaded_at")
    previous_pageouts = _memory_counter_cache.get("pageouts")
    previous_swapouts = _memory_counter_cache.get("swapouts")
    current_pageouts = vm_stats.get("pageouts")
    current_swapouts = vm_stats.get("swapouts")
    _memory_counter_cache.update(
        {"loaded_at": now, "pageouts": current_pageouts, "swapouts": current_swapouts}
    )
    if (
        previous_at is None
        or previous_pageouts is None
        or previous_swapouts is None
        or current_pageouts is None
        or current_swapouts is None
    ):
        return {"pageouts_per_sec": None, "swapouts_per_sec": None}
    elapsed = max(0.001, now - float(previous_at))
    return {
        "pageouts_per_sec": max(0.0, float(current_pageouts - int(previous_pageouts)) / elapsed),
        "swapouts_per_sec": max(0.0, float(current_swapouts - int(previous_swapouts)) / elapsed),
    }


def _sample_memory_state() -> _MemoryPressureResult:
    if not _env_bool("PITH_HOST_PRESSURE_MEMORY_ENABLED", platform.system() == "Darwin"):
        return _MemoryPressureResult(None, [])
    if platform.system() != "Darwin":
        return _MemoryPressureResult(None, [])
    if should_suppress_optional_subprocess("host_pressure_memory"):
        return _MemoryPressureResult(
            None,
            ["host_memory_subprocess_suppressed_fork_safety"],
            signal_state="unknown",
            confidence="suppressed",
        )

    reason_codes: list[str] = []
    states: list[str] = []
    artifact_reason_codes: list[str] = []
    corrob_reason_codes: list[str] = []
    swap_output = _run_memory_command(["sysctl", "vm.swapusage"])
    vm_output = _run_memory_command(["vm_stat"])
    memory_pressure_output = _run_memory_command(["memory_pressure"])
    corroboration_enabled = _env_bool("PITH_HOST_PRESSURE_DARWIN_CORROBORATION_ENABLED", True)

    swap = _parse_swapusage_mb(swap_output or "") if swap_output else None
    if swap and swap["total_mb"] > 0:
        used_ratio = swap["used_mb"] / swap["total_mb"]
        if (
            used_ratio >= _env_float("PITH_HOST_PRESSURE_MEMORY_CRITICAL_SWAP_USED_RATIO", 0.90)
            or (
                swap["used_mb"] > 0
                and swap["free_mb"] <= _env_float("PITH_HOST_PRESSURE_MEMORY_CRITICAL_SWAP_FREE_MB", 1024.0)
            )
        ):
            states.append("critical")
            artifact_reason_codes.append("host_memory_swap_critical")
        elif used_ratio >= _env_float("PITH_HOST_PRESSURE_MEMORY_HIGH_SWAP_USED_RATIO", 0.75):
            states.append("high")
            artifact_reason_codes.append("host_memory_swap_high")
        elif used_ratio >= _env_float("PITH_HOST_PRESSURE_MEMORY_MODERATE_SWAP_USED_RATIO", 0.50):
            states.append("moderate")
            artifact_reason_codes.append("host_memory_swap_moderate")

    vm_stats = _parse_vm_stat(vm_output or "") if vm_output else None
    physical_mb = _physical_memory_mb()
    if vm_stats and physical_mb and physical_mb > 0:
        page_mb = float(vm_stats["page_size"]) / (1024.0 * 1024.0)
        compressor_ratio = (float(vm_stats.get("compressor_pages", 0)) * page_mb) / physical_mb
        free_ratio = (
            float(vm_stats.get("free_pages", 0) + vm_stats.get("speculative_pages", 0)) * page_mb
        ) / physical_mb
        if compressor_ratio >= _env_float("PITH_HOST_PRESSURE_MEMORY_HIGH_COMPRESSOR_RATIO", 0.25):
            states.append("high")
            artifact_reason_codes.append("host_memory_compressor_high")
        if free_ratio <= _env_float("PITH_HOST_PRESSURE_MEMORY_MODERATE_FREE_RATIO", 0.10):
            states.append("moderate")
            artifact_reason_codes.append("host_memory_free_low")

    if memory_pressure_output:
        free_pct = _parse_memory_pressure_free_pct(memory_pressure_output)
        if free_pct is None:
            reason_codes.append("host_memory_pressure_parse_failed")
        elif free_pct <= _env_float("PITH_HOST_PRESSURE_MEMORY_PRESSURE_FREE_PCT_HIGH", 25.0):
            corrob_reason_codes.append("host_memory_pressure_free_pct_low")
    rates = _vm_counter_rates(vm_stats)
    pageouts_per_sec = rates.get("pageouts_per_sec")
    swapouts_per_sec = rates.get("swapouts_per_sec")
    if pageouts_per_sec is not None and pageouts_per_sec >= _env_float("PITH_HOST_PRESSURE_PAGEOUTS_PER_SEC_HIGH", 100.0):
        corrob_reason_codes.append("host_memory_pageouts_high")
    if swapouts_per_sec is not None and swapouts_per_sec >= _env_float("PITH_HOST_PRESSURE_SWAPOUTS_PER_SEC_HIGH", 1.0):
        corrob_reason_codes.append("host_memory_swapouts_high")

    if not swap and not vm_stats:
        return _MemoryPressureResult(
            "unknown",
            ["host_memory_pressure_unavailable"],
            signal_state="unknown",
            confidence="unknown",
        )
    if (swap_output and not swap) or (vm_output and not vm_stats):
        reason_codes.append("host_memory_pressure_parse_failed")
    signal_state = _max_known_state(*states)
    if signal_state == "critical":
        if not corroboration_enabled or corrob_reason_codes:
            return _MemoryPressureResult(
                "critical",
                list(dict.fromkeys(reason_codes + artifact_reason_codes + corrob_reason_codes)),
                signal_state=signal_state,
                confidence="current" if corrob_reason_codes else "critical_floor",
                artifact_reason_codes=list(dict.fromkeys(artifact_reason_codes)),
                corrob_reason_codes=list(dict.fromkeys(corrob_reason_codes)),
            )
        return _MemoryPressureResult(
            "moderate",
            list(dict.fromkeys(reason_codes + artifact_reason_codes)),
            signal_state=signal_state,
            confidence="advisory",
            artifact_reason_codes=list(dict.fromkeys(artifact_reason_codes)),
            corrob_reason_codes=list(dict.fromkeys(corrob_reason_codes)),
        )
    if not corroboration_enabled:
        return _MemoryPressureResult(
            signal_state,
            list(dict.fromkeys(reason_codes + artifact_reason_codes)),
            signal_state=signal_state,
            confidence="current" if signal_state in {"high", "moderate"} else signal_state,
            artifact_reason_codes=list(dict.fromkeys(artifact_reason_codes)),
            corrob_reason_codes=list(dict.fromkeys(corrob_reason_codes)),
        )
    if signal_state == "high" and corrob_reason_codes:
        return _MemoryPressureResult(
            "high",
            list(dict.fromkeys(reason_codes + artifact_reason_codes + corrob_reason_codes)),
            signal_state=signal_state,
            confidence="current",
            artifact_reason_codes=list(dict.fromkeys(artifact_reason_codes)),
            corrob_reason_codes=list(dict.fromkeys(corrob_reason_codes)),
        )
    if signal_state in {"high", "moderate"}:
        return _MemoryPressureResult(
            "moderate",
            list(dict.fromkeys(reason_codes + artifact_reason_codes)),
            signal_state=signal_state,
            confidence="advisory",
            artifact_reason_codes=list(dict.fromkeys(artifact_reason_codes)),
            corrob_reason_codes=list(dict.fromkeys(corrob_reason_codes)),
        )
    if signal_state == "none" and reason_codes:
        return _MemoryPressureResult(
            "unknown",
            list(dict.fromkeys(reason_codes)),
            signal_state=signal_state,
            confidence="unknown",
            artifact_reason_codes=list(dict.fromkeys(artifact_reason_codes)),
            corrob_reason_codes=list(dict.fromkeys(corrob_reason_codes)),
        )
    return _MemoryPressureResult(
        signal_state,
        list(dict.fromkeys(reason_codes + artifact_reason_codes)),
        signal_state=signal_state,
        confidence=signal_state,
        artifact_reason_codes=list(dict.fromkeys(artifact_reason_codes)),
        corrob_reason_codes=list(dict.fromkeys(corrob_reason_codes)),
    )


def build_host_pressure_snapshot(*, use_cache: bool = True) -> HostPressureSnapshot:
    now_mono = time.monotonic()
    signature = _snapshot_cache_signature()
    if use_cache and _snapshot_cache.get("snapshot") is not None:
        loaded_at = float(_snapshot_cache.get("loaded_at") or 0.0)
        if _snapshot_cache.get("signature") == signature and now_mono - loaded_at <= _CACHE_TTL_SECONDS:
            return _snapshot_cache["snapshot"]

    started = time.perf_counter()
    observed_at = _utc_now_iso()
    if not _env_bool("PITH_HOST_PRESSURE_ENABLED", True):
        snapshot = HostPressureSnapshot(False, "none", None, None, None, None, ["host_pressure_disabled"], observed_at, 0.0)
        _snapshot_cache.update({"loaded_at": now_mono, "signature": signature, "snapshot": snapshot})
        return snapshot

    forced = os.environ.get("PITH_HOST_PRESSURE_FORCE_STATE")
    if forced:
        state = forced.strip().lower()
        if state not in {"none", "moderate", "high", "critical", "unknown"}:
            state = "unknown"
        snapshot = HostPressureSnapshot(
            True,
            state,
            None,
            None,
            os.cpu_count(),
            None,
            [f"host_pressure_forced_{state}"],
            observed_at,
            round((time.perf_counter() - started) * 1000.0, 2),
        )
        _snapshot_cache.update({"loaded_at": now_mono, "signature": signature, "snapshot": snapshot})
        return snapshot

    reason_codes: list[str] = []
    try:
        load_1m = float(os.getloadavg()[0])
        cpu_count = max(1, int(os.cpu_count() or 1))
        load_per_core = load_1m / cpu_count
        load_state = _classify_load(load_per_core, reason_codes)
        memory_result = _sample_memory_state()
        reason_codes.extend(memory_result.reason_codes)
        state = _max_known_state(load_state, memory_result.state)
    except Exception:
        snapshot = HostPressureSnapshot(
            True,
            "unknown",
            None,
            None,
            os.cpu_count(),
            None,
            ["host_pressure_unavailable"],
            observed_at,
            round((time.perf_counter() - started) * 1000.0, 2),
            memory_signal_state="unknown",
            memory_signal_confidence="unknown",
        )
        _snapshot_cache.update({"loaded_at": now_mono, "signature": signature, "snapshot": snapshot})
        return snapshot

    snapshot = HostPressureSnapshot(
        True,
        state,
        round(load_1m, 4),
        round(load_per_core, 4),
        cpu_count,
        memory_result.state,
        reason_codes,
        observed_at,
        round((time.perf_counter() - started) * 1000.0, 2),
        memory_signal_state=memory_result.signal_state,
        memory_signal_confidence=memory_result.confidence,
        memory_artifact_reason_codes=memory_result.artifact_reason_codes,
        memory_corrob_reason_codes=memory_result.corrob_reason_codes,
    )
    _snapshot_cache.update({"loaded_at": now_mono, "signature": signature, "snapshot": snapshot})
    return snapshot
