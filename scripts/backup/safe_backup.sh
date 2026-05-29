#!/bin/bash
# ============================================================
# Pith — Safe Database Backup
# ============================================================
# Creates a consistent backup of the Pith database using SQLite's
# backup API (via .backup command). This is the ONLY safe way
# to back up a WAL-mode database while the server is running.
#
# DO NOT use: cp pith.db backup.db (corrupts if WAL active)
# DO NOT use: tar/zip on live DB (same corruption risk)
#
# Usage: bash scripts/backup/safe_backup.sh [--output /path/to/backup.db]
#
# Cron example (every 3 hours, 6am-11pm):
#   0 6,9,12,15,18,21 * * * cd /path/to/pith && bash scripts/backup/safe_backup.sh >> data/backup.log 2>&1
# ============================================================
set -e

# OPS-002: Accept SCRIPT_DIR/PROJECT_DIR as env vars for launchd compatibility
# (BASH_SOURCE is unavailable when invoked via eval/source workaround)
SCRIPT_DIR="${SCRIPT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
PROJECT_DIR="${PROJECT_DIR:-$(dirname "$(dirname "$SCRIPT_DIR")")}"

# Find database using shared resolution (mirrors app/profile.py)
PITH_HOME="${PITH_HOME:-$HOME/.pith}"
source "$SCRIPT_DIR/../resolve_db.sh" 2>/dev/null || {
    echo "ERROR: resolve_db.sh not found"; exit 1;
}
if ! resolve_pith_db; then
    # Fall back for error reporting
    DB_PATH="$HOME/pith-data/${PITH_PROFILE:-default}/pith.db"
    DATA_DIR="$HOME/pith-data/${PITH_PROFILE:-default}"
fi
ARCHIVE_DIR="$DATA_DIR/backups"
mkdir -p "$ARCHIVE_DIR" 2>/dev/null || true
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Parse args
OUTPUT_PATH=""
QUIET=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output) OUTPUT_PATH="$2"; shift 2 ;;
        --quiet|-q) QUIET=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Default output
if [ -z "$OUTPUT_PATH" ]; then
    OUTPUT_PATH="$ARCHIVE_DIR/pith_backup_${TIMESTAMP}.db"
fi

# Verify source exists
if [ ! -f "$DB_PATH" ]; then
    echo "ERROR: Database not found at $DB_PATH"
    echo "  Has Pith been started at least once? Run: pith start"
    exit 1
fi

# Check sqlite3 is available
if ! command -v sqlite3 &>/dev/null; then
    echo "ERROR: sqlite3 not found. Install it:"
    echo "  Mac:   brew install sqlite"
    echo "  Linux: sudo apt install sqlite3"
    exit 1
fi

log() { [ "$QUIET" = false ] && echo "$@" || true; }

log "Creating safe backup of $(basename "$DB_PATH")..."
log "  Source: $DB_PATH"
log "  Output: $OUTPUT_PATH"

# Use SQLite .backup command (safe even with active WAL)
sqlite3 "$DB_PATH" ".backup '$OUTPUT_PATH'"

# Verify the backup
INTEGRITY=$(sqlite3 "$OUTPUT_PATH" "PRAGMA integrity_check;" 2>&1)
CONCEPTS=$(sqlite3 "$OUTPUT_PATH" "SELECT COUNT(*) FROM concepts;" 2>&1 || echo "0")
LATEST=$(sqlite3 "$OUTPUT_PATH" "SELECT MAX(created_at) FROM concepts;" 2>&1 || echo "N/A")

if [ "$INTEGRITY" = "ok" ]; then
    SIZE=$(du -h "$OUTPUT_PATH" | cut -f1)
    log ""
    log "✓ Backup successful!"
    log "  Size: $SIZE"
    log "  Concepts: $CONCEPTS"
    log "  Latest: $LATEST"
    log "  Integrity: OK"
else
    echo ""
    echo "✗ BACKUP INTEGRITY CHECK FAILED!"
    echo "  $INTEGRITY"
    rm -f "$OUTPUT_PATH"
    exit 1
fi

# Data presence assertion — catch empty-db backups early
if [ "$CONCEPTS" = "0" ] || [ -z "$CONCEPTS" ]; then
    echo "⚠ WARNING: Backup contains 0 concepts."
    echo "  This might indicate the wrong database was backed up."
    echo "  Check that PITH_HOME is set correctly."
    # Don't exit — a 0-concept backup is still valid for fresh installs
fi

# --- Retention: keep only the last N automated backups ---
KEEP=${KEEP_BACKUPS:-3}
if [ -d "$ARCHIVE_DIR" ]; then
    BACKUPS=($(ls -1t "$ARCHIVE_DIR"/pith_backup_*.db 2>/dev/null | grep -v '\-shm$\|\-wal$'))
    COUNT=${#BACKUPS[@]}
    if [ "$COUNT" -gt "$KEEP" ]; then
        PRUNED=0
        for OLD_BACKUP in "${BACKUPS[@]:$KEEP}"; do
            rm -f "$OLD_BACKUP" "${OLD_BACKUP}-shm" "${OLD_BACKUP}-wal"
            ((PRUNED+=1))
        done
        log "  Retention: kept $KEEP, pruned $PRUNED old backup(s)"
    else
        log "  Retention: $COUNT backup(s), no pruning needed (keep=$KEEP)"
    fi
fi
