"""Cleanup utilities for abandoned incremental TF-IDF index temp dirs.

MAINT-059: stale ``.incremental.tmp.<pid>`` directories can remain when a process
dies outside the Python exception path during atomic index save. This module is
operator-invoked and dry-run-first; apply mode moves candidates to quarantine
instead of deleting them.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_MIN_AGE_HOURS = 24.0
_LsofRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]


def resolve_index_dir(
    profile: str | None = None,
    data_dir: str | None = None,
    index_dir: str | None = None,
) -> Path:
    """Resolve the directory that contains the live index and temp siblings."""
    if index_dir:
        return Path(index_dir).expanduser().resolve()

    if data_dir:
        base = Path(data_dir).expanduser().resolve()
    else:
        resolved_profile = profile or os.environ.get("PITH_PROFILE") or "default"
        base = Path.home() / "pith-data" / resolved_profile

    return base / "index"


def scan_incremental_tmp_dirs(
    index_dir: Path,
    index_name: str = "incremental",
    min_age_hours: float = DEFAULT_MIN_AGE_HOURS,
    now: float | None = None,
    pid_exists: Callable[[int], bool] | None = None,
) -> dict[str, Any]:
    """Scan for stale incremental temp dirs without modifying disk."""
    index_dir = Path(index_dir)
    current_time = now if now is not None else datetime.now(UTC).timestamp()
    pid_checker = pid_exists or _pid_exists
    pattern = re.compile(rf"^\.{re.escape(index_name)}\.tmp\.(\d+)$")

    candidates: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    if not index_dir.exists():
        return {
            "status": "missing_index_dir",
            "index_dir": str(index_dir),
            "index_name": index_name,
            "min_age_hours": min_age_hours,
            "candidates": candidates,
            "skipped": skipped,
            "candidate_count": 0,
            "skipped_count": 0,
        }

    for child in sorted(index_dir.iterdir(), key=lambda p: p.name):
        match = pattern.match(child.name)
        if not match:
            continue

        pid = int(match.group(1))
        if child.is_symlink():
            skipped.append(_entry(child, pid, "skipped", "symlink"))
            continue

        try:
            stat = child.stat(follow_symlinks=False)
        except OSError as exc:
            skipped.append(_entry(child, pid, "skipped", f"stat_error: {exc}"))
            continue

        if not child.is_dir():
            skipped.append(_entry(child, pid, "skipped", "not_directory", stat))
            continue

        age_hours = max(0.0, (current_time - stat.st_mtime) / 3600)
        if age_hours < min_age_hours:
            skipped.append(_entry(child, pid, "skipped", "too_young", stat, age_hours))
            continue

        if pid_checker(pid):
            skipped.append(_entry(child, pid, "skipped", "live_pid", stat, age_hours))
            continue

        candidates.append(_entry(child, pid, "candidate", "eligible", stat, age_hours))

    return {
        "status": "scanned",
        "index_dir": str(index_dir),
        "index_name": index_name,
        "min_age_hours": min_age_hours,
        "candidates": candidates,
        "skipped": skipped,
        "candidate_count": len(candidates),
        "skipped_count": len(skipped),
    }


def cleanup_incremental_tmp_dirs(
    index_dir: Path,
    *,
    apply: bool = False,
    quarantine_root: Path | None = None,
    index_name: str = "incremental",
    min_age_hours: float = DEFAULT_MIN_AGE_HOURS,
    require_open_file_check: bool = True,
    now: float | None = None,
    pid_exists: Callable[[int], bool] | None = None,
    lsof_runner: _LsofRunner | None = None,
) -> dict[str, Any]:
    """Dry-run or quarantine stale incremental temp dirs."""
    scan = scan_incremental_tmp_dirs(
        index_dir,
        index_name=index_name,
        min_age_hours=min_age_hours,
        now=now,
        pid_exists=pid_exists,
    )

    if not apply:
        return {
            **scan,
            "status": "dry_run",
            "dry_run": True,
            "moved_count": 0,
            "failed_count": 0,
            "manifest_path": None,
        }

    index_dir = Path(index_dir)
    if require_open_file_check:
        _assert_no_open_temp_files(index_dir, index_name, lsof_runner=lsof_runner)

    run_id = _run_id(now)
    root = Path(quarantine_root) if quarantine_root else index_dir.parent / "quarantine"
    quarantine_dir = root / "index_tmp_cleanup" / run_id
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    for candidate in scan["candidates"]:
        src = Path(candidate["source"])
        dest = quarantine_dir / src.name
        record = {
            **candidate,
            "destination": str(dest),
        }

        try:
            if src.is_symlink():
                record.update(status="skipped", reason="symlink")
            elif not src.exists():
                record.update(status="failed", reason="source_missing")
            elif dest.exists():
                record.update(status="failed", reason="destination_exists")
            else:
                shutil.move(str(src), str(dest))
                record.update(status="moved", reason="quarantined")
        except Exception as exc:  # pragma: no cover - defensive manifest detail
            record.update(status="failed", reason=f"move_error: {exc}")
        entries.append(record)

    entries.extend({**item, "destination": None} for item in scan["skipped"])

    manifest = {
        "run_id": run_id,
        "started_at": _iso_from_timestamp(now),
        "index_dir": str(index_dir),
        "index_name": index_name,
        "min_age_hours": min_age_hours,
        "dry_run": False,
        "entries": entries,
    }
    manifest_path = quarantine_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    moved_count = sum(1 for entry in entries if entry["status"] == "moved")
    failed_count = sum(1 for entry in entries if entry["status"] == "failed")

    return {
        **scan,
        "status": "applied" if failed_count == 0 else "partial_failure",
        "dry_run": False,
        "entries": entries,
        "moved_count": moved_count,
        "failed_count": failed_count,
        "manifest_path": str(manifest_path),
        "quarantine_dir": str(quarantine_dir),
    }


def rollback_incremental_tmp_cleanup(manifest_path: Path) -> dict[str, Any]:
    """Restore quarantined entries marked as moved in a cleanup manifest."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    restored: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    for entry in manifest.get("entries", []):
        if entry.get("status") != "moved":
            continue

        source = Path(entry["source"])
        destination = Path(entry["destination"])
        result = {
            "source": str(source),
            "destination": str(destination),
        }

        try:
            if source.exists():
                result.update(status="failed", reason="source_exists")
                failed.append(result)
                continue
            if not destination.exists():
                result.update(status="failed", reason="destination_missing")
                failed.append(result)
                continue
            source.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(destination), str(source))
            result.update(status="restored", reason="rollback")
            restored.append(result)
        except Exception as exc:  # pragma: no cover - defensive rollback detail
            result.update(status="failed", reason=f"rollback_error: {exc}")
            failed.append(result)

    return {
        "status": "rolled_back" if not failed else "partial_failure",
        "manifest_path": str(manifest_path),
        "restored_count": len(restored),
        "failed_count": len(failed),
        "restored": restored,
        "failed": failed,
    }


def _assert_no_open_temp_files(
    index_dir: Path,
    index_name: str,
    *,
    lsof_runner: _LsofRunner | None = None,
) -> None:
    runner = lsof_runner or _run_lsof
    try:
        result = runner(["lsof", "+D", str(index_dir)])
    except FileNotFoundError as exc:
        raise RuntimeError("lsof unavailable; refusing apply mode") from exc

    stdout = result.stdout or ""
    stderr = result.stderr or ""
    no_open_files = result.returncode == 1 and not stdout.strip() and not stderr.strip()
    if result.returncode not in (0, 1) or (result.returncode == 1 and not no_open_files):
        raise RuntimeError(
            f"lsof failed with code {result.returncode}: {stderr.strip() or stdout.strip()}"
        )

    needle = f".{index_name}.tmp."
    open_temp_lines = [line for line in stdout.splitlines() if needle in line]
    if open_temp_lines:
        raise RuntimeError(
            "open incremental temp files detected; refusing apply mode: "
            + "; ".join(open_temp_lines[:5])
        )


def _run_lsof(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, timeout=30, check=False)


def _entry(
    path: Path,
    pid: int,
    status: str,
    reason: str,
    stat: os.stat_result | None = None,
    age_hours: float | None = None,
) -> dict[str, Any]:
    return {
        "source": str(path),
        "name": path.name,
        "pid": pid,
        "status": status,
        "reason": reason,
        "mtime": stat.st_mtime if stat else None,
        "age_hours": round(age_hours, 3) if age_hours is not None else None,
        "size_bytes": _path_size_bytes(path) if stat and path.is_dir() else None,
    }


def _path_size_bytes(path: Path) -> int:
    total = 0
    for root, dirs, files in os.walk(path, followlinks=False):
        dirs[:] = [
            dirname
            for dirname in dirs
            if not (Path(root) / dirname).is_symlink()
        ]
        for filename in files:
            candidate = Path(root) / filename
            try:
                if not candidate.is_symlink():
                    total += candidate.stat(follow_symlinks=False).st_size
            except OSError:
                continue
    return total


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _run_id(now: float | None = None) -> str:
    return datetime.fromtimestamp(now, UTC).strftime("%Y%m%dT%H%M%SZ") if now else (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )


def _iso_from_timestamp(now: float | None = None) -> str:
    return (
        datetime.fromtimestamp(now, UTC).isoformat()
        if now
        else datetime.now(UTC).isoformat()
    )
