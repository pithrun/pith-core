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
# Usage: bash scripts/backup/safe_backup.sh [--output /path/to/backup.db] [--quiet] [--dry-run]
#
# Cron example (every 3 hours, 6am-11pm):
#   0 6,9,12,15,18,21 * * * cd /path/to/pith && bash scripts/backup/safe_backup.sh >> data/backup.log 2>&1
# ============================================================
set -euo pipefail
trap 'echo "ERROR: Backup command failed" >&2; exit 1' ERR

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
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Parse args
OUTPUT_PATH=""
QUIET=false
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output)
            if [[ $# -lt 2 || -z "$2" ]]; then
                echo "ERROR: --output requires a non-empty path"; exit 1
            fi
            OUTPUT_PATH="$2"; shift 2 ;;
        --quiet|-q) QUIET=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Validate before any backup or retention write; avoid arithmetic evaluation of input.
KEEP=${KEEP_BACKUPS:-3}
if [[ ! "$KEEP" =~ ^[0-9]{1,9}$ ]] || ((10#$KEEP < 1)); then
    echo "ERROR: KEEP_BACKUPS must be an integer from 1 to 999999999"
    exit 1
fi
KEEP=$((10#$KEEP))

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

# Preview the command plan without creating directories, a backup, or pruning files.
# Retention candidates are evaluated only after a real backup passes integrity checks.
if [ "$DRY_RUN" = true ]; then
    printf 'Dry run: would back up %s to %s and retain the newest %s regular automated backup(s).\n' "$DB_PATH" "$OUTPUT_PATH" "$KEEP"
    exit 2
fi
mkdir -p "$ARCHIVE_DIR"

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
if [ -d "$ARCHIVE_DIR" ]; then
    # Keep paths as array elements; textual ls output splits legal filenames.
    BACKUPS=()
    COUNT=0
    for BACKUP in "$ARCHIVE_DIR"/pith_backup_*.db; do
        if [[ -f "$BACKUP" && ! -L "$BACKUP" ]]; then
            BACKUPS[COUNT]="$BACKUP"
            COUNT=$((COUNT + 1))
        fi
    done
    if [ "$COUNT" -gt "$KEEP" ]; then
        # Select only the newest KEEP entries: O(COUNT * KEEP), Bash 3.2 compatible.
        for ((I=0; I<KEEP; I++)); do
            NEWEST=$I
            for ((J=I+1; J<COUNT; J++)); do
                if [[ "${BACKUPS[J]}" -nt "${BACKUPS[NEWEST]}" ]]; then
                    NEWEST=$J
                fi
            done
            SWAP="${BACKUPS[I]}"
            BACKUPS[I]="${BACKUPS[NEWEST]}"
            BACKUPS[NEWEST]="$SWAP"
        done
        PRUNED=0
        for OLD_BACKUP in "${BACKUPS[@]:$KEEP}"; do
            rm -f -- "$OLD_BACKUP" "${OLD_BACKUP}-shm" "${OLD_BACKUP}-wal"
            ((PRUNED+=1))
        done
        log "  Retention: kept $KEEP, pruned $PRUNED old backup(s)"
    else
        log "  Retention: $COUNT backup(s), no pruning needed (keep=$KEEP)"
    fi
fi
