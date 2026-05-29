"""Migration 006: Clean up non-canonical concept_types and corrupted timestamps.

INGEST-004: Remap 63 concepts with junk types to canonical CONCEPT_TYPES.
Also fix 85 concepts with corrupted content_updated_at/created_at values.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

# Mapping from junk type → canonical type
TYPE_REMAP = {
    "architecture": "observation",
    "discovery": "observation",
    "performance": "observation",
    "requirement": "constraint",
    "recommendation": "decision",
    "comparison": "observation",
    "solution": "method",
    "causal": "pattern",
    "heurist": "heuristic",
    "insight": "observation",
    "pr": "observation",
    "problem": "observation",
    # Sprint 17a additions (STABILITY-018):
    "process": "method",
    "client_extraction": "observation",
}

MIGRATION_ID = "DATA-006-JUNK-TYPE-CLEANUP"
DESCRIPTION = "Migration 006 junk type and corrupted timestamp cleanup"
FORCE_ENV_VAR = "PITH_FORCE_MIGRATION_006"
READONLY_SKIP_ENV = "PITH_BENCHMARK_READONLY"


def needs_migration(conn):
    """Return true when migration 006 still has cleanup work to perform."""
    placeholders = ",".join("?" for _ in TYPE_REMAP)
    if conn.execute(
        f"SELECT 1 FROM concepts WHERE concept_type IN ({placeholders}) LIMIT 1",
        tuple(TYPE_REMAP.keys()),
    ).fetchone():
        return True
    if conn.execute(
        """SELECT 1 FROM concepts
           WHERE content_updated_at IS NOT NULL
             AND length(content_updated_at) < 10
           LIMIT 1"""
    ).fetchone():
        return True
    return bool(
        conn.execute(
            """SELECT 1 FROM concepts
           WHERE created_at IS NOT NULL
             AND length(created_at) < 10
           LIMIT 1"""
        ).fetchone()
    )


def migrate(conn):
    """Remap junk concept_types and fix corrupted timestamps."""
    if os.environ.get("PITH_BENCHMARK_READONLY", "").lower() in ("true", "1"):
        logger.info("Migration 006 skipped (PITH_BENCHMARK_READONLY)")
        return {"status": "skipped", "reason": "benchmark_readonly"}

    # Phase 1: Remap junk types
    total_remapped = 0
    for old_type, new_type in TYPE_REMAP.items():
        # Store old type in data JSON for audit trail before remapping
        rows = conn.execute("SELECT id, data FROM concepts WHERE concept_type = ?", (old_type,)).fetchall()
        for row_id, data_str in rows:
            try:
                data = json.loads(data_str) if data_str else {}
                data["_original_concept_type"] = old_type
                conn.execute(
                    "UPDATE concepts SET concept_type = ?, data = ? WHERE id = ?",
                    (new_type, json.dumps(data), row_id),
                )
                total_remapped += 1
            except (json.JSONDecodeError, Exception) as e:
                logger.warning("Migration 006: Failed to remap %s: %s", row_id, e)

    # Phase 2: Fix corrupted timestamps
    # Corrupted values are very short (< 10 chars) like "2", "20"
    ts_fixed = 0

    # Fix content_updated_at from updated_at (which is more likely correct)
    fixed = conn.execute(
        """UPDATE concepts SET content_updated_at = updated_at
           WHERE content_updated_at IS NOT NULL
             AND length(content_updated_at) < 10
             AND updated_at IS NOT NULL
             AND length(updated_at) >= 10"""
    ).rowcount
    ts_fixed += fixed

    # Fix created_at from updated_at where created_at is corrupted
    fixed2 = conn.execute(
        """UPDATE concepts SET created_at = updated_at
           WHERE created_at IS NOT NULL
             AND length(created_at) < 10
             AND updated_at IS NOT NULL
             AND length(updated_at) >= 10"""
    ).rowcount
    ts_fixed += fixed2

    # Fix content_updated_at that's still bad (no good updated_at either)
    # Set to a reasonable default
    fixed3 = conn.execute(
        """UPDATE concepts SET content_updated_at = '2026-02-15T00:00:00+00:00'
           WHERE content_updated_at IS NOT NULL
             AND length(content_updated_at) < 10"""
    ).rowcount
    ts_fixed += fixed3

    # Fix created_at that's still bad
    fixed4 = conn.execute(
        """UPDATE concepts SET created_at = '2026-02-15T00:00:00+00:00'
           WHERE created_at IS NOT NULL
             AND length(created_at) < 10"""
    ).rowcount
    ts_fixed += fixed4

    logger.info(
        "Migration 006: Remapped %d concepts (%d junk types), fixed %d corrupted timestamps",
        total_remapped,
        len(
            [
                t
                for t in TYPE_REMAP
                if conn.execute(
                    "SELECT 1 FROM concepts WHERE data LIKE ?", (f'%"_original_concept_type": "{t}"%',)
                ).fetchone()
            ]
        ),
        ts_fixed,
    )
    return {"status": "success", "remapped": total_remapped, "timestamps_fixed": ts_fixed}
