#!/bin/bash
# ============================================================
# Pith — Backup Sync
# ============================================================
# Syncs the latest backup to additional locations for
# redundancy. Auto-detects Google Drive and iCloud on macOS.
# Also supports a custom BACKUP_DIR for any platform.
#
# Usage: bash scripts/backup/sync_backups.sh
#
# Environment:
#   BACKUP_DIR    — Custom sync target (e.g., ~/my-backups, /mnt/nas/pith)
#   KEEP_SYNCED   — Max backups to retain per tier (default: 5)
#
# Cron example (daily at 2:30am):
#   30 2 * * * cd /path/to/pith && bash scripts/backup/sync_backups.sh >> data/sync.log 2>&1
# ============================================================
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
# Resolve backup directory from profile (same as safe_backup.sh)
source "$SCRIPT_DIR/../resolve_db.sh" 2>/dev/null || true
if type resolve_pith_db &>/dev/null && resolve_pith_db; then
    ARCHIVE_DIR="$DATA_DIR/backups"
else
    ARCHIVE_DIR="$PROJECT_DIR/data/archive"
fi
LOG_FILE="$PROJECT_DIR/data/sync.log"
KEEP=${KEEP_SYNCED:-5}

log() { echo "$(date '+%Y-%m-%d %H:%M:%S'): $1" | tee -a "$LOG_FILE"; }

# --- Find latest backup ---
# Find latest backup (supports both pith_backup_* and legacy pith_backup_* naming)
LATEST=$(ls -t "$ARCHIVE_DIR"/pith_backup_*.db "$ARCHIVE_DIR"/pith_backup_*.db 2>/dev/null | head -1)
if [ -z "$LATEST" ]; then
    log "ERROR — No backups found in $ARCHIVE_DIR"
    log "  Run safe_backup.sh first: bash scripts/backup/safe_backup.sh"
    exit 1
fi

LATEST_NAME=$(basename "$LATEST")
SYNCED=0
ERRORS=0

# Helper: sync to a directory with retention
sync_to_dir() {
    local DIR="$1"
    local LABEL="$2"

    if [ ! -d "$DIR" ]; then
        log "SKIP $LABEL: directory not found ($DIR)"
        return
    fi

    if cp "$LATEST" "$DIR/"; then
        SYNCED=$((SYNCED + 1))
        log "✓ $LABEL: synced $LATEST_NAME"
        # Retain last N backups
        cd "$DIR"
        # Retain last N backups (both naming conventions)
        ls -t pith_backup_*.db pith_backup_*.db 2>/dev/null | tail -n +$((KEEP + 1)) | xargs rm -f 2>/dev/null || true
    else
        ERRORS=$((ERRORS + 1))
        log "✗ $LABEL: FAILED to copy $LATEST_NAME"
    fi
}

# --- Tier 1: Custom backup directory (any platform) ---
if [ -n "$BACKUP_DIR" ]; then
    mkdir -p "$BACKUP_DIR" 2>/dev/null || true
    sync_to_dir "$BACKUP_DIR" "Custom"
fi

# --- Tier 2: Google Drive (macOS auto-detect) ---
GDRIVE_BASE=""
if [ -d "$HOME/Google Drive/My Drive" ]; then
    GDRIVE_BASE="$HOME/Google Drive/My Drive"
elif [ -d "$HOME/Library/CloudStorage" ]; then
    CS=$(find "$HOME/Library/CloudStorage" -maxdepth 1 -name "GoogleDrive-*" -type d 2>/dev/null | head -1)
    [ -n "$CS" ] && GDRIVE_BASE="$CS/My Drive"
fi
if [ -n "$GDRIVE_BASE" ]; then
    GDRIVE_DIR="$GDRIVE_BASE/pith-backups"
    mkdir -p "$GDRIVE_DIR" 2>/dev/null || true
    sync_to_dir "$GDRIVE_DIR" "GDrive"
fi

# --- Tier 3: iCloud (macOS auto-detect) ---
ICLOUD_DIR="$HOME/Library/Mobile Documents/com~apple~CloudDocs/pith-backups"
if [ -d "$HOME/Library/Mobile Documents/com~apple~CloudDocs" ]; then
    mkdir -p "$ICLOUD_DIR" 2>/dev/null || true
    sync_to_dir "$ICLOUD_DIR" "iCloud"
fi

# --- Tier 4: Local redundancy (canonical backup dir) ---
# OPS-012: Skip when LOCAL_DIR matches ARCHIVE_DIR (source==dest causes cp error)
LOCAL_DIR="$HOME/pith-data/${PITH_PROFILE:-default}/backups"
if [ "$(cd "$LOCAL_DIR" 2>/dev/null && pwd)" != "$(cd "$ARCHIVE_DIR" 2>/dev/null && pwd)" ]; then
    mkdir -p "$LOCAL_DIR" 2>/dev/null || true
    sync_to_dir "$LOCAL_DIR" "Local"
else
    log "SKIP Local: same as archive dir ($ARCHIVE_DIR)"
fi

# --- Summary ---
log "Sync complete: $SYNCED succeeded, $ERRORS failed (from $LATEST_NAME)"

if [ "$ERRORS" -gt 0 ]; then
    exit 1
fi
