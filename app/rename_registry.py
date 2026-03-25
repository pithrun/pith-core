"""Factual drift detection via rename tracking (DATA-042).

Scans concepts with source-anchored evidence (file_path from DATA-041/043)
and detects when referenced files have been renamed, moved, or deleted.
Also scans evidence content for known renamed strings (e.g., brain.db → pith.db).

Used by reflection and staleness systems to flag/downgrade drifted concepts.
"""

import json
import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

REPO_PATH = os.environ.get("PITH_REPO_PATH", os.path.expanduser("~/Desktop/pith-beta"))

# Known renames: old_string -> new_string
# These are historical renames that won't show up in recent git history
KNOWN_RENAMES = {
    "brain.db": "pith.db",
    "pith-system": "pith-beta",
    "pith_system": "pith_beta",
    "pith-mcp": "pith",
    "pith_mcp": "pith",
    "/pith-system/": "/pith-beta/",
    "brain.db path": "pith.db path",
}


@dataclass
class DriftFinding:
    """A single source-drift detection result."""

    concept_id: str
    evidence_id: str
    drift_type: str  # "file_missing", "file_renamed", "content_stale"
    old_reference: str  # What the evidence says
    new_reference: str | None = None  # What it should say (if known)
    confidence: float = 0.8  # How confident we are this is real drift
    severity: str = "medium"  # low, medium, high


@dataclass
class RenameRegistry:
    """Session-scoped rename detection engine."""

    repo_path: str = REPO_PATH
    git_renames: dict[str, str] = field(default_factory=dict)  # from GitCache
    findings: list[DriftFinding] = field(default_factory=list)

    def load_git_renames(self, git_cache=None) -> None:
        """Import renames from GitCache if available."""
        if git_cache and hasattr(git_cache, "renamed_files"):
            self.git_renames.update(git_cache.renamed_files)

    def scan_concepts(self, db_path: str) -> list[DriftFinding]:
        """Scan all active concepts for source drift. Returns findings."""
        import sqlite3

        self.findings = []
        conn = sqlite3.connect(db_path)

        cursor = conn.execute("SELECT id, data FROM concepts WHERE status = 'active'")

        for row in cursor:
            concept_id = row[0]
            try:
                data = json.loads(row[1])
            except (json.JSONDecodeError, TypeError):
                continue

            evidence = data.get("evidence", [])
            for e in evidence:
                if not isinstance(e, dict):
                    continue

                ev_id = e.get("id", "unknown")
                file_path = e.get("file_path")
                content = e.get("content", "")

                # Check 1: file_path references a missing file
                if file_path and file_path not in ("server.js",):
                    full_path = os.path.join(self.repo_path, file_path)
                    if not os.path.exists(full_path):
                        # Check if it was renamed via git
                        new_path = self.git_renames.get(file_path)
                        if new_path:
                            self.findings.append(
                                DriftFinding(
                                    concept_id=concept_id,
                                    evidence_id=ev_id,
                                    drift_type="file_renamed",
                                    old_reference=file_path,
                                    new_reference=new_path,
                                    confidence=0.95,
                                    severity="high",
                                )
                            )
                        else:
                            # File is gone, no known rename
                            self.findings.append(
                                DriftFinding(
                                    concept_id=concept_id,
                                    evidence_id=ev_id,
                                    drift_type="file_missing",
                                    old_reference=file_path,
                                    confidence=0.7,
                                    severity="medium",
                                )
                            )

                # Check 2: evidence content contains known stale strings
                for old_str, new_str in KNOWN_RENAMES.items():
                    if old_str in content:
                        self.findings.append(
                            DriftFinding(
                                concept_id=concept_id,
                                evidence_id=ev_id,
                                drift_type="content_stale",
                                old_reference=old_str,
                                new_reference=new_str,
                                confidence=0.85,
                                severity="medium",
                            )
                        )
                        break  # One finding per evidence item for content

        conn.close()

        logger.info(
            f"RenameRegistry scan complete: {len(self.findings)} findings "
            f"({sum(1 for f in self.findings if f.drift_type == 'file_missing')} missing, "
            f"{sum(1 for f in self.findings if f.drift_type == 'file_renamed')} renamed, "
            f"{sum(1 for f in self.findings if f.drift_type == 'content_stale')} content_stale)"
        )

        return self.findings

    def get_affected_concept_ids(self) -> set[str]:
        """Return unique concept IDs with at least one drift finding."""
        return {f.concept_id for f in self.findings}

    def get_high_severity(self) -> list[DriftFinding]:
        """Return only high-severity findings."""
        return [f for f in self.findings if f.severity == "high"]
