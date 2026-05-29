"""Scope-aware decay — only decay claims about changed code regions (DATA-044).

When a file changes, naive decay would penalize ALL concepts referencing that file.
Scope-aware decay checks whether the specific lines/functions referenced in evidence
actually overlap with the git diff hunks. If they don't overlap, the concept is
unaffected and should not be decayed.

This prevents over-decay during active development where many files change but
each concept only cares about a specific region.
"""

import logging
import os
import re
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)

REPO_PATH = os.environ.get("PITH_REPO_PATH", os.path.expanduser("~/Desktop/pith-beta"))


@dataclass
class ChangedRegion:
    """A region of changed lines in a file."""

    file_path: str
    start_line: int
    end_line: int


def get_changed_regions(file_path: str, repo_path: str = REPO_PATH) -> list[ChangedRegion]:
    """Get the specific line ranges that changed in a file using git diff.

    Returns list of ChangedRegion with start/end lines that were modified.
    Empty list if file is unmodified or git fails.
    """
    try:
        result = subprocess.run(
            ["git", "diff", "-U0", "HEAD", "--", file_path],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=repo_path,
        )
        if result.returncode != 0 or not result.stdout:
            return []

        regions = []
        # Parse unified diff hunk headers: @@ -old_start,old_count +new_start,new_count @@
        hunk_re = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
        for match in hunk_re.finditer(result.stdout):
            new_start = int(match.group(3))
            new_count = int(match.group(4)) if match.group(4) else 1
            if new_count > 0:
                regions.append(
                    ChangedRegion(
                        file_path=file_path,
                        start_line=new_start,
                        end_line=new_start + new_count - 1,
                    )
                )
        return regions

    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        logger.debug(f"get_changed_regions failed for {file_path}: {e}")
        return []


def ranges_overlap(ev_range: str | None, regions: list[ChangedRegion]) -> bool:
    """Check if an evidence line_range overlaps with any changed region.

    Args:
        ev_range: Evidence line_range string like "33-80" or "150" or None
        regions: List of changed regions from git diff

    Returns:
        True if there's overlap (evidence may be stale).
        True if ev_range is None (can't determine scope, conservative).
    """
    if not ev_range or not regions:
        # No line range on evidence = can't scope, assume affected (conservative)
        return bool(regions)

    # Parse evidence range
    try:
        if "-" in ev_range:
            parts = ev_range.split("-")
            ev_start = int(parts[0])
            ev_end = int(parts[1])
        else:
            ev_start = ev_end = int(ev_range)
    except (ValueError, IndexError):
        return True  # Can't parse = conservative

    # Check overlap with any changed region
    return any(ev_start <= region.end_line and ev_end >= region.start_line for region in regions)


def should_decay_evidence(
    file_path: str | None,
    line_range: str | None,
    changed_files: dict[str, str] | None = None,
    repo_path: str = REPO_PATH,
) -> bool:
    """Determine if an evidence item should be subject to decay.

    Args:
        file_path: Evidence file_path (e.g., "app/storage.py")
        line_range: Evidence line_range (e.g., "33-80")
        changed_files: From GitCache.changed_files (path -> status)
        repo_path: Repository root path

    Returns:
        True if the evidence should be decayed (file changed in overlapping region).
        False if evidence is safe (file unchanged or change is out of scope).
    """
    if not file_path:
        return False  # No file reference = not a code claim, skip

    # Check if file is in the changed set
    if changed_files and file_path not in changed_files:
        return False  # File not changed at all

    # File is changed — check if the change overlaps with evidence scope
    if line_range:
        regions = get_changed_regions(file_path, repo_path)
        if regions:
            return ranges_overlap(line_range, regions)
        # No regions parseable but file is marked changed = conservative decay
        return True

    # No line_range but file is changed = conservative (decay)
    return True
