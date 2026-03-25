"""Migration 008: Fix KA blob/column desync (DEBT-185).

save_concept() was writing the resolved KA to the SQL column but serializing the
original (stale) KA into the JSON blob.  This migration repairs existing desyncs
by copying the authoritative column value into the blob.
"""

import json
import logging
import sqlite3

logger = logging.getLogger(__name__)

MIGRATION_ID = "008_ka_blob_sync"


def migrate(conn: sqlite3.Connection, *, dry_run: bool = False) -> dict:
    """Sync knowledge_area from SQL column → JSON blob for all desynced rows."""
    cursor = conn.execute(
        """SELECT id, knowledge_area, data FROM concepts
           WHERE json_valid(data)
             AND json_extract(data, '$.knowledge_area') IS NOT NULL
             AND json_extract(data, '$.knowledge_area') != knowledge_area"""
    )
    rows = cursor.fetchall()

    fixed = 0
    for row in rows:
        cid, col_ka, data_json = row
        if dry_run:
            blob_ka = json.loads(data_json).get("knowledge_area")
            logger.info("DRY-RUN: %s  column=%s  blob=%s", cid, col_ka, blob_ka)
            fixed += 1
            continue
        conn.execute(
            """UPDATE concepts
               SET data = json_set(data, '$.knowledge_area', ?)
               WHERE id = ?""",
            (col_ka, cid),
        )
        fixed += 1

    if not dry_run and fixed > 0:
        conn.commit()

    logger.info("DEBT-185: %s %d KA blob/column desyncs", "Would fix" if dry_run else "Fixed", fixed)
    return {"migration": MIGRATION_ID, "fixed": fixed, "dry_run": dry_run}
