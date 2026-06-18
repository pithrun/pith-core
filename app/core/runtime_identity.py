"""Runtime identity helpers for process and DB contention diagnostics."""

from __future__ import annotations

import multiprocessing
import os
import subprocess
from functools import lru_cache
from pathlib import Path

from app.core.fork_safety import should_suppress_optional_subprocess
from app.core.profile import get_active_profile, resolve_data_dir


def _runtime_source_dir() -> Path:
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def _resolve_git_commit() -> str:
    env_commit = os.environ.get("PITH_GIT_COMMIT", "").strip()
    if env_commit:
        return env_commit
    if should_suppress_optional_subprocess("runtime_identity_git_commit"):
        return "unknown"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_runtime_source_dir()),
            check=True,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


@lru_cache(maxsize=1)
def get_runtime_identity() -> dict[str, str | int]:
    """Return cached identity fields safe to attach to every log record."""
    data_dir = resolve_data_dir()
    return {
        "pid": os.getpid(),
        "process_name": multiprocessing.current_process().name,
        "runtime_role": os.environ.get("PITH_RUNTIME_ROLE", "server"),
        "profile": get_active_profile(),
        "data_dir": str(data_dir),
        "git_commit": _resolve_git_commit(),
    }


class RuntimeIdentityLogFilter:
    """Populate runtime identity fields on log records before formatting."""

    def filter(self, record) -> bool:
        identity = get_runtime_identity()
        record.pid = identity["pid"]
        record.process_name = identity["process_name"]
        record.runtime_role = identity["runtime_role"]
        record.profile = identity["profile"]
        record.data_dir = identity["data_dir"]
        record.git_commit = identity["git_commit"]
        return True
