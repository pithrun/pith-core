#!/bin/bash
# ============================================================
# Pith — Setup Automated Backup Schedule (macOS launchd)
# ============================================================
# Installs launchd agents for automated backups on macOS.
# Previous cron-based approach fails on macOS Sequoia+ due to
# TCC directory-access restrictions on cron. launchd agents
# inherit the user's TCC grants, solving the permission issue.
#
# Schedule:
#   - safe_backup.sh:   Every 3 hours (via StartInterval)
#   - sync_backups.sh:  Daily at 2:30am (via StartCalendarInterval)
#
# Usage: bash scripts/backup/setup_schedule.sh
#        bash scripts/backup/setup_schedule.sh --remove
#        bash scripts/backup/setup_schedule.sh --status
# ============================================================
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"
BACKUP_LABEL="com.pith.backup"
SYNC_LABEL="com.pith.sync"
BACKUP_PLIST="$LAUNCH_AGENTS_DIR/$BACKUP_LABEL.plist"
SYNC_PLIST="$LAUNCH_AGENTS_DIR/$SYNC_LABEL.plist"

# --- Helper: unload and remove a plist ---
_remove_agent() {
    local label="$1"
    local plist="$2"
    if [ -f "$plist" ]; then
        launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
        rm -f "$plist"
        echo "  Removed $label"
    fi
}

# --- Helper: migrate legacy cron entries ---
_remove_legacy_cron() {
    local current
    current=$(crontab -l 2>/dev/null || true)
    if echo "$current" | grep -qE "Pith|Pith PCI"; then
        echo "$current" | grep -vE "Pith|Pith PCI" | \
            grep -v "safe_backup.sh" | grep -v "sync_backups.sh" | crontab -
        echo "  Migrated: removed legacy cron entries"
    fi
}

# --- Status mode ---
if [ "$1" = "--status" ]; then
    echo "Pith Backup Schedule Status"
    echo "==========================="
    for label in "$BACKUP_LABEL" "$SYNC_LABEL"; do
        if launchctl print "gui/$(id -u)/$label" &>/dev/null; then
            echo "  $label: LOADED (running)"
        elif [ -f "$LAUNCH_AGENTS_DIR/$label.plist" ]; then
            echo "  $label: INSTALLED (not loaded)"
        else
            echo "  $label: NOT INSTALLED"
        fi
    done
    if crontab -l 2>/dev/null | grep -qE "Pith|Pith PCI"; then
        echo ""
        echo "  ⚠ Legacy cron entries detected. Run without flags to migrate."
    fi
    exit 0
fi

# --- Remove mode ---
if [ "$1" = "--remove" ]; then
    echo "Removing Pith backup schedule..."
    _remove_agent "$BACKUP_LABEL" "$BACKUP_PLIST"
    _remove_agent "$SYNC_LABEL" "$SYNC_PLIST"
    _remove_legacy_cron
    echo "✓ Backup schedule removed"
    exit 0
fi

# --- Check platform ---
if [ "$(uname)" != "Darwin" ]; then
    echo "⚠ launchd is macOS-only. For Linux, use systemd timers or cron."
    exit 1
fi

# --- Check if already installed ---
if [ -f "$BACKUP_PLIST" ] && [ -f "$SYNC_PLIST" ]; then
    echo "⚠ Backup schedule already installed as launchd agents."
    echo "  To update: bash scripts/backup/setup_schedule.sh --remove && re-run"
    echo "  Status:    bash scripts/backup/setup_schedule.sh --status"
    exit 0
fi

# --- Migrate legacy cron if present ---
_remove_legacy_cron

# --- Create dirs ---
mkdir -p "$LAUNCH_AGENTS_DIR" "$PROJECT_DIR/data"

# --- Generate backup plist (every 3 hours = 10800 seconds) ---
cat > "$BACKUP_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${BACKUP_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>${SCRIPT_DIR}/safe_backup.sh</string>
        <string>--quiet</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${PROJECT_DIR}</string>
    <key>StartInterval</key>
    <integer>10800</integer>
    <key>StandardOutPath</key>
    <string>${PROJECT_DIR}/data/backup.log</string>
    <key>StandardErrorPath</key>
    <string>${PROJECT_DIR}/data/backup.log</string>
    <key>RunAtLoad</key>
    <true/>
</dict>
</plist>
PLIST

# --- Generate sync plist (daily at 2:30am) ---
cat > "$SYNC_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${SYNC_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>${SCRIPT_DIR}/sync_backups.sh</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${PROJECT_DIR}</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>2</integer>
        <key>Minute</key>
        <integer>30</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>${PROJECT_DIR}/data/sync.log</string>
    <key>StandardErrorPath</key>
    <string>${PROJECT_DIR}/data/sync.log</string>
</dict>
</plist>
PLIST

# --- Load agents ---
launchctl bootstrap "gui/$(id -u)" "$BACKUP_PLIST"
launchctl bootstrap "gui/$(id -u)" "$SYNC_PLIST"

echo "✓ Backup schedule installed via launchd!"
echo ""
echo "  Backup: Every 3 hours (com.pith.backup)"
echo "  Sync:   Daily at 2:30am (com.pith.sync)"
echo ""
echo "  Logs:   $PROJECT_DIR/data/backup.log"
echo "          $PROJECT_DIR/data/sync.log"
echo ""
echo "  Status: bash scripts/backup/setup_schedule.sh --status"
echo "  Remove: bash scripts/backup/setup_schedule.sh --remove"
echo ""
echo "  Note: launchd agents inherit your TCC permissions."
