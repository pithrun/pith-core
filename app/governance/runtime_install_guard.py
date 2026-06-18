"""Guards against binding installed runtime aliases to unsafe git worktrees."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from app.core.config import REPO_HYGIENE_RUNTIME_ROOT_MARKERS
from app.core.fork_safety import should_suppress_optional_subprocess

INSTALLED_RUNTIME_ROOT = Path("~/.pith/pith-server").expanduser()
UNSAFE_RUNTIME_CLASSIFICATIONS = {"canonical_checkout", "unregistered_worktree"}
SESSION_WORKTREE_DIRNAME = "_" + "_".join(("session", "worktrees"))


class RuntimeInstallGuardError(RuntimeError):
    """Raised when the installed runtime path resolves to an unsafe location."""

    def __init__(self, report: dict[str, Any]):
        self.report = report
        detail = report.get("detail") or "Unsafe installed runtime path"
        super().__init__(detail)


def _git_output(path: Path, *args: str) -> str | None:
    if should_suppress_optional_subprocess("runtime_install_guard_git"):
        return None
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except Exception:
        return None
    return completed.stdout.strip()


def _repo_root(path: Path) -> Path | None:
    target = path if path.is_dir() else path.parent
    output = _git_output(target, "rev-parse", "--show-toplevel")
    if not output:
        return None
    return Path(output).resolve()


def _primary_worktree_path(repo_root: Path) -> Path | None:
    output = _git_output(repo_root, "worktree", "list", "--porcelain")
    if not output:
        return None
    for line in output.splitlines():
        if line.startswith("worktree "):
            return Path(line.split(" ", 1)[1]).resolve()
    return None


def _looks_like_installed_runtime(invocation_path: str | os.PathLike[str] | None) -> bool:
    if not invocation_path:
        return False
    candidate = Path(invocation_path).expanduser()
    try:
        return candidate == INSTALLED_RUNTIME_ROOT or INSTALLED_RUNTIME_ROOT in candidate.parents
    except Exception:
        return False


def _path_contains_dir(path: Path, dirname: str) -> bool:
    return any(part == dirname for part in path.parts)


def classify_runtime_path(runtime_path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Classify the installed runtime alias target.

    Standalone installs outside git repos are valid. Git-backed installs are unsafe
    only when they resolve to the primary checkout or to a session worktree.
    """

    runtime_root = Path(runtime_path or INSTALLED_RUNTIME_ROOT).expanduser().resolve()
    report: dict[str, Any] = {
        "invocation_root": str(Path(runtime_path).expanduser()) if runtime_path else str(INSTALLED_RUNTIME_ROOT),
        "resolved_root": str(runtime_root),
        "repo_root": "",
        "classification": "standalone_install",
        "violation": False,
        "detail": "",
    }

    repo_root = _repo_root(runtime_root)
    if repo_root is None:
        if should_suppress_optional_subprocess("runtime_install_guard_git"):
            report["classification"] = "runtime_install_guard_unverified_fork_safety"
            report["detail"] = (
                "Installed runtime git classification skipped because the API process is fork-sensitive."
            )
            return report
        report["detail"] = (
            f"Installed runtime resolves to standalone path {runtime_root} (allowed)."
        )
        return report

    primary = _primary_worktree_path(repo_root)
    report["repo_root"] = str(repo_root)

    resolved_repo_root = str(repo_root)
    if primary and repo_root == primary:
        classification = "canonical_checkout"
    elif any(marker and marker in resolved_repo_root for marker in REPO_HYGIENE_RUNTIME_ROOT_MARKERS):
        classification = "runtime_release_worktree"
    elif _path_contains_dir(repo_root, SESSION_WORKTREE_DIRNAME):
        classification = "unregistered_worktree"
    else:
        classification = "non_session_worktree"

    violation = classification in UNSAFE_RUNTIME_CLASSIFICATIONS
    report["classification"] = classification
    report["violation"] = violation
    if violation:
        report["detail"] = (
            "Installed runtime path is unsafe: "
            f"resolved_root={runtime_root} classification={classification}. "
            "Expected ~/.pith/pith-server to resolve to a standalone install or a "
            "runtime release worktree under /_release_worktrees/."
        )
    else:
        report["detail"] = (
            f"Installed runtime resolves to {runtime_root} classification={classification} (allowed)."
        )
    return report


def ensure_safe_installed_runtime(
    *,
    invocation_path: str | os.PathLike[str] | None = None,
    runtime_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Fail closed when an installed runtime invocation resolves to an unsafe path."""

    if not _looks_like_installed_runtime(invocation_path):
        return {
            "invocation_root": str(invocation_path) if invocation_path else "",
            "resolved_root": "",
            "repo_root": "",
            "classification": "not_installed_runtime",
            "violation": False,
            "detail": "Invocation is not running from ~/.pith/pith-server; installed-runtime guard skipped.",
        }

    report = classify_runtime_path(runtime_path=runtime_path)
    if report["violation"]:
        raise RuntimeInstallGuardError(report)
    return report
