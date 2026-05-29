"""Version chain integrity diagnostics.

Memory Integrity Spec v1.2, §5.2.4 [resolves C11]
Adapted for our schema: concepts (latest only) + concept_versions (append-only history).

Health checks:
1. Version mismatch: concepts.version != latest in concept_versions
2. Orphaned concepts: id in concepts but missing from concept_versions
3. Gap detection: version numbers not contiguous (v1, v2, v4 — missing v3)
"""

import logging

from app.storage import get_db_connection

logger = logging.getLogger(__name__)


def check_version_chain_integrity() -> list[str]:
    """Verify all version chains are consistent.

    Returns list of violation descriptions. Empty list = healthy.
    Run during startup and as periodic health check.
    """
    violations = []
    conn = get_db_connection()

    # Check 1: concepts.version matches latest version in concept_versions
    mismatches = conn.execute("""
        SELECT c.id, c.version as concepts_version, cv.max_version
        FROM concepts c
        LEFT JOIN (
            SELECT id, MAX(version) as max_version
            FROM concept_versions
            GROUP BY id
        ) cv ON c.id = cv.id
        WHERE c.version != cv.max_version OR cv.max_version IS NULL
    """).fetchall()

    for row in mismatches:
        if row["max_version"] is None:
            violations.append(f"Orphaned concept {row['id']}: in concepts table but no version history")
        else:
            violations.append(
                f"Version mismatch {row['id']}: concepts={row['concepts_version']}, latest_history={row['max_version']}"
            )

    # Check 2: Version gap detection (v1, v2, v4 = gap at v3)
    all_ids = conn.execute("SELECT DISTINCT id FROM concept_versions").fetchall()

    for id_row in all_ids:
        concept_id = id_row["id"]
        versions = conn.execute(
            "SELECT version FROM concept_versions WHERE id = ? ORDER BY version", (concept_id,)
        ).fetchall()

        version_nums = []
        for v in versions:
            try:
                version_nums.append(int(v["version"][1:]))  # "v3" -> 3
            except (ValueError, IndexError):
                violations.append(f"Malformed version {concept_id}: '{v['version']}' is not v<N> format")

        if version_nums:
            expected = list(range(1, max(version_nums) + 1))
            missing = set(expected) - set(version_nums)
            if missing:
                violations.append(
                    f"Version gap {concept_id}: missing {sorted(missing)} in chain {sorted(version_nums)}"
                )

    if violations:
        logger.warning("Version chain integrity: %d violations found", len(violations))
        for v in violations:
            logger.warning("  %s", v)
    else:
        logger.debug("Version chain integrity: all chains healthy")

    return violations
