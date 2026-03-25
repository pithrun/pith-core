#!/usr/bin/env python3
"""
STABILITY-006 Data Repair Migration

Repairs SQL/JSON desyncs from two sources:
  M1: Recovery artifacts (.recover on 2026-03-08 zeroed SQL columns)
  M2: KA normalization (async_tasks.py wrote SQL without syncing JSON)

Run AFTER code fixes (FIX-S1 through FIX-S10) are deployed.
Requires SQLite 3.38+ for json_set/json_valid/json_extract.

Usage:
  PITH_PROFILE=rose python3 migrations/stability_006_repair.py [--dry-run]
"""
import sqlite3
import sys
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


def resolve_db_path() -> Path:
    profile = os.environ.get("PITH_PROFILE", "rose")
    return Path.home() / "pith-data" / profile / "pith.db"


def run_migration(db_path: Path, dry_run: bool = False):
    now = datetime.now(timezone.utc).isoformat()
    print(f"=== STABILITY-006 Migration — {now} ===")
    print(f"DB: {db_path}")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}\n")

    # Verify DB exists
    if not db_path.exists():
        print(f"ERROR: DB not found at {db_path}")
        sys.exit(1)

    # Backup
    if not dry_run:
        backup = db_path.parent / f"pith.db.pre-stability-006-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"Creating backup: {backup}")
        shutil.copy2(db_path, backup)
        print("Backup created.\n")

    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row

    # Verify SQLite version
    ver = db.execute("SELECT sqlite_version()").fetchone()[0]
    print(f"SQLite version: {ver}")
    parts = ver.split(".")[:3]
    major, minor = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    if major < 3 or (major == 3 and minor < 38):
        print("ERROR: SQLite 3.38+ required for json_set")
        sys.exit(1)

    # ========== PRE-MIGRATION COUNTS ==========
    print("\n--- Pre-migration counts ---")

    m1a = db.execute("""
        SELECT COUNT(*) as c FROM concepts
        WHERE currency_status = '' AND json_valid(data)
          AND json_extract(data, '$.currency_status') IS NOT NULL
          AND json_extract(data, '$.currency_status') != ''
    """).fetchone()["c"]
    print(f"M1a (empty currency_status, JSON has value): {m1a}")

    m1b = db.execute("""
        SELECT COUNT(*) as c FROM concepts
        WHERE maturity = '' AND json_valid(data)
          AND json_extract(data, '$.maturity') IS NOT NULL
          AND json_extract(data, '$.maturity') != ''
    """).fetchone()["c"]
    print(f"M1b (empty maturity, JSON has value): {m1b}")

    m1c = db.execute("""
        SELECT COUNT(*) as c FROM concepts
        WHERE confidence = 0.0 AND json_valid(data)
          AND json_extract(data, '$.confidence') IS NOT NULL
          AND CAST(json_extract(data, '$.confidence') AS REAL) > 0.0
    """).fetchone()["c"]
    print(f"M1c (zeroed confidence, JSON has value): {m1c}")

    m1d = db.execute("""
        SELECT COUNT(*) as c FROM concepts
        WHERE maturity = 'DISCARDED' AND json_valid(data)
          AND json_extract(data, '$.maturity') = 'QUARANTINED'
    """).fetchone()["c"]
    print(f"M1d (DISCARDED/QUARANTINED mismatch): {m1d}")

    m2 = db.execute("""
        SELECT COUNT(*) as c FROM concepts
        WHERE json_valid(data)
          AND knowledge_area IS NOT NULL AND knowledge_area != ''
          AND json_extract(data, '$.knowledge_area') IS NOT NULL
          AND knowledge_area != json_extract(data, '$.knowledge_area')
    """).fetchone()["c"]
    print(f"M2 (KA desync): {m2}")

    if dry_run:
        print("\n[DRY RUN] No changes made.")
        db.close()
        return

    # ========== EXECUTE MIGRATIONS ==========
    print("\n--- Executing migrations ---")

    # M1a: Restore currency_status from JSON
    r = db.execute("""
        UPDATE concepts
        SET currency_status = json_extract(data, '$.currency_status')
        WHERE currency_status = ''
          AND json_valid(data)
          AND json_extract(data, '$.currency_status') IS NOT NULL
          AND json_extract(data, '$.currency_status') != ''
    """)
    print(f"M1a: {r.rowcount} rows updated (expected ~{m1a})")

    # M1b: Restore maturity from JSON
    r = db.execute("""
        UPDATE concepts
        SET maturity = json_extract(data, '$.maturity')
        WHERE maturity = ''
          AND json_valid(data)
          AND json_extract(data, '$.maturity') IS NOT NULL
          AND json_extract(data, '$.maturity') != ''
    """)
    print(f"M1b: {r.rowcount} rows updated (expected ~{m1b})")

    # M1c: Restore confidence from JSON
    r = db.execute("""
        UPDATE concepts
        SET confidence = CAST(json_extract(data, '$.confidence') AS REAL)
        WHERE confidence = 0.0
          AND json_valid(data)
          AND json_extract(data, '$.confidence') IS NOT NULL
          AND CAST(json_extract(data, '$.confidence') AS REAL) > 0.0
    """)
    print(f"M1c: {r.rowcount} rows updated (expected ~{m1c})")

    # M1d: Sync JSON maturity FROM SQL for DISCARDED concepts
    r = db.execute("""
        UPDATE concepts
        SET data = json_set(data, '$.maturity', maturity)
        WHERE maturity = 'DISCARDED'
          AND json_valid(data)
          AND json_extract(data, '$.maturity') = 'QUARANTINED'
    """)
    print(f"M1d: {r.rowcount} rows updated (expected ~{m1d})")

    # M2: Sync JSON $.knowledge_area FROM SQL column
    r = db.execute("""
        UPDATE concepts
        SET data = json_set(data, '$.knowledge_area', knowledge_area)
        WHERE json_valid(data)
          AND knowledge_area IS NOT NULL
          AND knowledge_area != ''
          AND json_extract(data, '$.knowledge_area') IS NOT NULL
          AND knowledge_area != json_extract(data, '$.knowledge_area')
    """)
    print(f"M2: {r.rowcount} rows updated (expected ~{m2})")

    db.commit()
    print("\nAll migrations committed.")

    # ========== POST-MIGRATION VERIFICATION ==========
    print("\n--- Post-migration verification ---")

    checks = {
        "M1a_remaining": """SELECT COUNT(*) FROM concepts
            WHERE currency_status = '' AND json_valid(data)
              AND json_extract(data, '$.currency_status') IS NOT NULL
              AND json_extract(data, '$.currency_status') != ''""",
        "M1b_remaining": """SELECT COUNT(*) FROM concepts
            WHERE maturity = '' AND json_valid(data)
              AND json_extract(data, '$.maturity') IS NOT NULL
              AND json_extract(data, '$.maturity') != ''""",
        "M1c_remaining": """SELECT COUNT(*) FROM concepts
            WHERE confidence = 0.0 AND json_valid(data)
              AND json_extract(data, '$.confidence') IS NOT NULL
              AND CAST(json_extract(data, '$.confidence') AS REAL) > 0.0""",
        "M1d_remaining": """SELECT COUNT(*) FROM concepts
            WHERE maturity = 'DISCARDED' AND json_valid(data)
              AND json_extract(data, '$.maturity') = 'QUARANTINED'""",
        "M2_remaining": """SELECT COUNT(*) FROM concepts
            WHERE json_valid(data)
              AND knowledge_area IS NOT NULL AND knowledge_area != ''
              AND json_extract(data, '$.knowledge_area') IS NOT NULL
              AND knowledge_area != json_extract(data, '$.knowledge_area')""",
    }

    all_clear = True
    for name, query in checks.items():
        remaining = db.execute(query).fetchone()[0]
        status = "OK" if remaining == 0 else f"WARN: {remaining} remaining"
        if remaining > 0:
            all_clear = False
        print(f"  {name}: {status}")

    db.close()

    if all_clear:
        print("\n=== MIGRATION COMPLETE — all checks passed ===")
    else:
        print("\n=== MIGRATION COMPLETE — some checks have remaining items ===")
        print("Review warnings above. Remaining items may be from bad-JSON concepts (expected: 26).")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    db_path = resolve_db_path()
    run_migration(db_path, dry_run)
