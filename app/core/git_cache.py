"""Git state cache for source-drift detection (PERF-013).

Captures git state once at session_start, provides O(1) lookups
for changed/renamed files during the session. Phase 2 consumers
(DATA-042 rename registry, DATA-044 scope-aware decay) use this
instead of per-turn git subprocess calls.
"""

import logging
import os
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

# Default to pith-beta repo path
DEFAULT_REPO_PATH = os.environ.get("PITH_REPO_PATH", os.path.expanduser("~/Desktop/pith-beta"))


@dataclass
class GitCache:
    """Session-scoped git state cache."""

    repo_path: str = DEFAULT_REPO_PATH
    current_hash: str | None = None
    changed_files: dict[str, str] = field(default_factory=dict)  # path -> status (M/A/D/R)
    renamed_files: dict[str, str] = field(default_factory=dict)  # old_path -> new_path
    populated_at: str | None = None
    _populated: bool = False

    def populate(self, lookback_commits: int = 20) -> None:
        """Cache git state. Safe to call multiple times (idempotent)."""
        if self._populated:
            return
        try:
            # Current commit hash
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=self.repo_path,
            )
            if result.returncode == 0:
                self.current_hash = result.stdout.strip()

            # Changed files (working tree + staged vs HEAD)
            result = subprocess.run(
                ["git", "diff", "--name-status", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=self.repo_path,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if not line:
                        continue
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        status = parts[0][0]  # M, A, D, R, C
                        path = parts[1]
                        self.changed_files[path] = status
                        # Handle renames (R100\told\tnew)
                        if status == "R" and len(parts) >= 3:
                            self.renamed_files[parts[1]] = parts[2]

            # Also check recent commits for renames
            result = subprocess.run(
                ["git", "diff", "--name-status", "--diff-filter=R", f"HEAD~{lookback_commits}..HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=self.repo_path,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if not line:
                        continue
                    parts = line.split("\t")
                    if len(parts) >= 3 and parts[0].startswith("R"):
                        self.renamed_files[parts[1]] = parts[2]

            self.populated_at = datetime.now(UTC).isoformat()
            self._populated = True

            logger.info(
                f"GitCache populated: hash={self.current_hash}, "
                f"changed={len(self.changed_files)}, "
                f"renamed={len(self.renamed_files)}"
            )

        except FileNotFoundError:
            logger.warning("git not found — GitCache disabled")
            self._populated = True  # Don't retry
        except subprocess.TimeoutExpired:
            logger.warning("git commands timed out — GitCache disabled")
            self._populated = True
        except Exception as e:
            logger.warning(f"GitCache population failed: {e}")
            self._populated = True

    def is_file_changed(self, path: str) -> bool:
        """Check if a file has been modified since cache population."""
        return path in self.changed_files

    def get_new_path(self, old_path: str) -> str | None:
        """Get the new path if a file was renamed, else None."""
        return self.renamed_files.get(old_path)

    def get_status(self, path: str) -> str | None:
        """Get the change status of a file (M/A/D/R), or None."""
        return self.changed_files.get(path)
