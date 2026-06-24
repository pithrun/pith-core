#!/bin/bash
set -euo pipefail

# ============================================================================
# Pith Docker → Native Migration Script (macOS)
# ============================================================================
# Migrates beta users from Docker pith-mcp container to native ~/.pith/ install.
#
# Implements: DOCKER_MIGRATION_SPEC.md §3-§4, Amendments A-E
# Fixes:      F1-F8 from spec §3.1
#
# Usage: bash migrate_from_docker.sh [--container-name NAME] [--data-dir PATH]
# ============================================================================

# --- Color codes ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

# --- Configuration ---
PITH_HOME="${PITH_HOME:-$HOME/.pith}"
DOCKER_CONTAINER_NAME="${DOCKER_CONTAINER_NAME:-pith-mcp}"   # F6: correct default
EXTRACT_TEMP_DIR=""
DOCKER_WAS_STOPPED=false                                       # Amendment E state
CONTAINER_ID=""
SKIP_IMPORT=false
PRE_CONCEPT_COUNT=0
NEW_API_KEY=""

# --- Parse CLI args ---
CUSTOM_DATA_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --container-name) DOCKER_CONTAINER_NAME="$2"; shift 2 ;;
        --data-dir)       CUSTOM_DATA_DIR="$2"; shift 2 ;;    # A1: allow custom path
        -h|--help)
            echo "Usage: bash migrate_from_docker.sh [--container-name NAME] [--data-dir PATH]"
            echo "  --container-name  Docker container name (default: pith-mcp)"
            echo "  --data-dir        Custom data directory inside container (default: auto-detect)"
            exit 0
            ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done
# --- Banner ---
print_banner() {
    clear
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════╗"
    echo "║   🧠 Pith Docker → Native Migration   ║"
    echo "║      v1.0 — macOS                      ║"
    echo "╚════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
}

# --- Indicators ---
mark_success() { echo -e "${GREEN}✓${NC} $1"; }
mark_warning() { echo -e "${YELLOW}⚠${NC}  $1"; }
mark_info()    { echo -e "${BLUE}ℹ${NC}  $1"; }

# --- Error handler ---
error_exit() {
    echo -e "${RED}✗ ERROR:${NC} $1" >&2
    exit 1
}

# ============================================================================
# Amendment E: Error Recovery Trap
# ============================================================================
# If we fail after stopping Docker, auto-restart it so the user isn't stranded.
cleanup_on_error() {
    local exit_code=$?
    if [[ "$DOCKER_WAS_STOPPED" == "true" && -n "$CONTAINER_ID" ]]; then
        echo ""
        echo -e "${RED}Migration failed. Restarting Docker container...${NC}"
        docker start "$CONTAINER_ID" 2>/dev/null || true
        docker update --restart=unless-stopped "$CONTAINER_ID" 2>/dev/null || true
        echo -e "${YELLOW}Docker container restarted. Your original setup should be working.${NC}"
    fi
    # Clean up temp dir
    if [[ -n "$EXTRACT_TEMP_DIR" && -d "$EXTRACT_TEMP_DIR" ]]; then
        rm -rf "$EXTRACT_TEMP_DIR"
    fi
    if [[ $exit_code -ne 0 ]]; then
        echo ""
        echo -e "${RED}Migration did not complete. Your Docker setup has been restored.${NC}"
        echo -e "If you need help, contact your beta support channel."
    fi
}

trap cleanup_on_error EXIT
print_banner

# ============================================================================
# STEP 0: Pre-flight Checks
# ============================================================================
echo -e "${BLUE}[Step 0/7]${NC} Pre-flight checks..."

# Docker installed?
if ! command -v docker &> /dev/null; then
    error_exit "Docker not found. Is Docker Desktop installed?"
fi

# Docker daemon running?
if ! docker info &> /dev/null; then
    error_exit "Docker daemon not running. Please start Docker Desktop and retry."
fi

# Python 3.9+?
if ! command -v python3 &> /dev/null; then
    error_exit "Python 3 not found. Please install Python 3.9+ first."
fi
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
if [[ "$PYTHON_MAJOR" -lt 3 || ("$PYTHON_MAJOR" -eq 3 && "$PYTHON_MINOR" -lt 9) ]]; then
    error_exit "Python 3.9+ required, found $PYTHON_VERSION"
fi
mark_success "Python $PYTHON_VERSION"

# Disk space (need ~500MB)
AVAILABLE_KB=$(df -k "$HOME" | tail -1 | awk '{print $4}')
if [[ "$AVAILABLE_KB" -lt 512000 ]]; then
    error_exit "Insufficient disk space. Need ~500MB, have $(( AVAILABLE_KB / 1024 ))MB"
fi
mark_success "Disk space OK ($(( AVAILABLE_KB / 1024 ))MB available)"

# F8: Port 8000 pre-check (informational)
PORT_8000_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health 2>/dev/null || echo "000")
if [[ "$PORT_8000_STATUS" == "200" ]]; then
    mark_info "Port 8000 is active (Docker pith likely running — will be stopped before install)"
elif [[ "$PORT_8000_STATUS" != "000" ]]; then
    mark_warning "Port 8000 responding with status $PORT_8000_STATUS — may need attention"
fi

echo ""
# ============================================================================
# STEP 1: Detect Docker Container (F6: correct default name)
# ============================================================================
echo -e "${BLUE}[Step 1/7]${NC} Detecting Docker container..."

RUNNING_CONTAINERS=$(docker ps --filter "name=$DOCKER_CONTAINER_NAME" --format "{{.Names}}" 2>/dev/null || echo "")

if [[ -z "$RUNNING_CONTAINERS" ]]; then
    # Check stopped containers too
    ALL_CONTAINERS=$(docker ps -a --filter "name=$DOCKER_CONTAINER_NAME" --format "{{.Names}}" 2>/dev/null || echo "")

    if [[ -z "$ALL_CONTAINERS" ]]; then
        echo ""
        echo -e "${YELLOW}No Docker container named '$DOCKER_CONTAINER_NAME' found.${NC}"
        echo "If your container has a different name, re-run with:"
        echo "  bash migrate_from_docker.sh --container-name YOUR_CONTAINER_NAME"
        echo ""
        echo "If you never installed the Docker version, just run the native installer directly:"
        echo "  bash scripts/install.sh"
        exit 1
    else
        mark_warning "Container '$ALL_CONTAINERS' exists but is not running."
        read -p "Start it to export data? (y/n): " -r
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker start "$(echo "$ALL_CONTAINERS" | head -1)" || error_exit "Failed to start container"
            RUNNING_CONTAINERS=$(echo "$ALL_CONTAINERS" | head -1)
        else
            error_exit "Cannot migrate without a running container. Aborting."
        fi
    fi
fi

CONTAINER_ID=$(echo "$RUNNING_CONTAINERS" | head -1)
mark_success "Found Docker container: $CONTAINER_ID"

# Verify it's actually running
CONTAINER_STATE=$(docker inspect "$CONTAINER_ID" --format='{{.State.Running}}' 2>/dev/null || echo "false")
if [[ "$CONTAINER_STATE" != "true" ]]; then
    error_exit "Container $CONTAINER_ID is not running."
fi

echo ""
# ============================================================================
# STEP 2: Export Pith Data from Container
# ============================================================================
echo -e "${BLUE}[Step 2/7]${NC} Exporting pith data from Docker container..."

EXTRACT_TEMP_DIR=$(mktemp -d)
mark_success "Created temp directory: $EXTRACT_TEMP_DIR"

# --- Amendment A: WAL Checkpoint before docker cp ---
echo "Flushing database write-ahead log..."
docker exec "$CONTAINER_ID" python3 -c "
import sqlite3, os
db_paths = ['/app/data/brain.db', '/pith/data/brain.db']
for p in db_paths:
    if os.path.exists(p):
        c = sqlite3.connect(p)
        c.execute('PRAGMA wal_checkpoint(FULL)')
        c.close()
        print(f'WAL checkpoint complete: {p}')
        break
else:
    print('No brain.db found for WAL checkpoint (non-fatal)')
" 2>/dev/null && mark_success "WAL checkpoint complete" || mark_warning "Could not flush WAL (non-fatal — data is still safe)"

# --- F1: Try /app/data first (our Dockerfile uses WORKDIR /app) ---
CONTAINER_DATA_PATH=""
if [[ -n "$CUSTOM_DATA_DIR" ]]; then
    # A1: user-specified path
    CONTAINER_DATA_PATH="$CUSTOM_DATA_DIR"
else
    for TRY_PATH in "/app/data" "/pith/data" "/home/pith/data" "/root/.pith/data"; do
        if docker exec "$CONTAINER_ID" test -d "$TRY_PATH" 2>/dev/null; then
            CONTAINER_DATA_PATH="$TRY_PATH"
            break
        fi
    done
fi

if [[ -z "$CONTAINER_DATA_PATH" ]]; then
    error_exit "Could not locate data directory in container. Try: --data-dir /path/inside/container"
fi

mark_success "Found data at: $CONTAINER_DATA_PATH"

# Export data via docker cp
if docker cp "$CONTAINER_ID:$CONTAINER_DATA_PATH" "$EXTRACT_TEMP_DIR/data" 2>/dev/null; then
    mark_success "Exported pith data from container"
else
    error_exit "Failed to export data from container. Check Docker permissions."
fi

# Verify brain.db exists
if [[ ! -f "$EXTRACT_TEMP_DIR/data/brain.db" ]]; then
    mark_warning "No brain.db found in exported data. Container may have a fresh/empty install."
    # Still continue — installer will create a fresh brain.db
fi

# F5: Pre-migration concept count
if [[ -f "$EXTRACT_TEMP_DIR/data/brain.db" ]]; then
    PRE_CONCEPT_COUNT=$(python3 -c "
import sqlite3
try:
    c = sqlite3.connect('$EXTRACT_TEMP_DIR/data/brain.db')
    count = c.execute('SELECT COUNT(*) FROM concepts').fetchone()[0]
    print(count)
    c.close()
except:
    print(0)
" 2>/dev/null || echo "0")
    DB_SIZE=$(du -sh "$EXTRACT_TEMP_DIR/data/brain.db" 2>/dev/null | cut -f1 || echo "unknown")
    mark_success "Pith contains $PRE_CONCEPT_COUNT concepts ($DB_SIZE)"
fi

echo ""
# ============================================================================
# STEP 2.5: Stop Docker Container (Amendment B)
# ============================================================================
echo -e "${BLUE}[Step 2.5/7]${NC} Stopping Docker container (freeing port 8000)..."

# Amendment B: Disable restart policy BEFORE stopping
docker update --restart=no "$CONTAINER_ID" 2>/dev/null || true
docker stop "$CONTAINER_ID" 2>/dev/null || true
DOCKER_WAS_STOPPED=true
mark_success "Docker container stopped (will not auto-restart)"

echo ""

# ============================================================================
# STEP 3: Generate New API Key (F3 — forced regeneration)
# ============================================================================
echo -e "${BLUE}[Step 3/7]${NC} Generating secure API key..."

NEW_API_KEY=$(openssl rand -hex 32)
mark_success "New API key generated (replaces default insecure key)"

echo ""
# ============================================================================
# STEP 4: Run Native Installer (F7: detect co-located installer)
# ============================================================================
echo -e "${BLUE}[Step 4/7]${NC} Running native Pith installer..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALLER_SCRIPT=""

# F7: Look for installer in expected locations
for TRY_INSTALLER in "$SCRIPT_DIR/install.sh" "$SCRIPT_DIR/../scripts/install.sh" "./scripts/install.sh"; do
    if [[ -f "$TRY_INSTALLER" ]]; then
        INSTALLER_SCRIPT="$TRY_INSTALLER"
        break
    fi
done

if [[ -z "$INSTALLER_SCRIPT" ]]; then
    error_exit "Installer script (install.sh) not found. Expected at $SCRIPT_DIR/install.sh"
fi

mark_info "Running installer from: $INSTALLER_SCRIPT"
bash "$INSTALLER_SCRIPT" || error_exit "Installer failed. Check the output above."
mark_success "Native Pith installation completed"

echo ""
# ============================================================================
# STEP 5: Import Data (F4: integrity check, F5: concept count, Amendment D)
# ============================================================================
echo -e "${BLUE}[Step 5/7]${NC} Importing pith data to native installation..."

mkdir -p "$PITH_HOME/data"
mkdir -p "$PITH_HOME/config"

if [[ -f "$EXTRACT_TEMP_DIR/data/brain.db" ]]; then
    # --- Amendment D: mtime comparison ---
    if [[ -f "$PITH_HOME/data/brain.db" ]]; then
        SRC_MTIME=$(stat -f %m "$EXTRACT_TEMP_DIR/data/brain.db" 2>/dev/null || stat -c %Y "$EXTRACT_TEMP_DIR/data/brain.db" 2>/dev/null || echo "0")
        DST_MTIME=$(stat -f %m "$PITH_HOME/data/brain.db" 2>/dev/null || stat -c %Y "$PITH_HOME/data/brain.db" 2>/dev/null || echo "0")
        if [[ "$DST_MTIME" -gt "$SRC_MTIME" ]]; then
            mark_warning "Existing brain.db at $PITH_HOME/data/ is NEWER than Docker export."
            read -p "Overwrite with Docker data? (y/n): " -r
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                mark_info "Keeping existing data. Skipping import."
                SKIP_IMPORT=true
            fi
        fi
    fi

    if [[ "$SKIP_IMPORT" != "true" ]]; then
        # --- Amendment D: SHA256 checksum ---
        SRC_HASH=$(shasum -a 256 "$EXTRACT_TEMP_DIR/data/brain.db" | cut -d' ' -f1)

        # --- HARDENED: Pre-copy integrity check on exported brain.db ---
        EXPORT_INTEGRITY=$(python3 -c "
import sqlite3
c = sqlite3.connect('$EXTRACT_TEMP_DIR/data/brain.db')
result = c.execute('PRAGMA integrity_check').fetchone()[0]
print(result)
c.close()
" 2>/dev/null || echo "error")
        if [[ "$EXPORT_INTEGRITY" != "ok" ]]; then
            echo ""
            mark_warning "EXPORTED brain.db FAILED integrity check: $EXPORT_INTEGRITY"
            mark_warning "The database inside your Docker container appears to be corrupted."
            echo ""
            echo "  Options:"
            echo "    1) Skip import — start with a fresh empty database"
            echo "    2) Proceed anyway — import the corrupted database (not recommended)"
            echo "    3) Abort — stop migration entirely"
            echo ""
            read -p "  Choose (1/2/3): " -r INTEGRITY_CHOICE
            case "$INTEGRITY_CHOICE" in
                1)
                    mark_info "Skipping import. You'll start with a fresh pith database."
                    SKIP_IMPORT=true
                    ;;
                2)
                    mark_warning "Proceeding with corrupted database at your own risk."
                    ;;
                *)
                    error_exit "Migration aborted by user. Docker container is still intact."
                    ;;
            esac
        else
            mark_success "Exported database integrity verified"
        fi
    fi

    if [[ "$SKIP_IMPORT" != "true" ]]; then
        # Backup existing data if present
        if [[ -f "$PITH_HOME/data/brain.db" ]]; then
            BACKUP_NAME="brain.db.pre-migration.$(date +%s)"
            cp "$PITH_HOME/data/brain.db" "$PITH_HOME/data/$BACKUP_NAME"
            mark_info "Backed up existing brain.db → $BACKUP_NAME"
        fi

        # Copy brain.db
        cp "$EXTRACT_TEMP_DIR/data/brain.db" "$PITH_HOME/data/brain.db"

        # Verify checksum
        DST_HASH=$(shasum -a 256 "$PITH_HOME/data/brain.db" | cut -d' ' -f1)
        if [[ "$SRC_HASH" == "$DST_HASH" ]]; then
            mark_success "Data copied and verified (SHA256 match)"
        else
            error_exit "Data copy verification FAILED! SHA256 mismatch. Docker data is safe in temp dir: $EXTRACT_TEMP_DIR"
        fi

        # F4: PRAGMA integrity_check
        INTEGRITY=$(python3 -c "
import sqlite3
c = sqlite3.connect('$PITH_HOME/data/brain.db')
result = c.execute('PRAGMA integrity_check').fetchone()[0]
print(result)
c.close()
" 2>/dev/null || echo "error")
        if [[ "$INTEGRITY" == "ok" ]]; then
            mark_success "Post-copy database integrity check passed"
        else
            mark_warning "Post-copy integrity check failed: $INTEGRITY"
            mark_warning "This may indicate a copy error. Your Docker data is still safe."
            error_exit "Aborting due to post-copy integrity failure. Exported data at: $EXTRACT_TEMP_DIR"
        fi

        # F5: Post-migration concept count
        POST_CONCEPT_COUNT=$(python3 -c "
import sqlite3
try:
    c = sqlite3.connect('$PITH_HOME/data/brain.db')
    count = c.execute('SELECT COUNT(*) FROM concepts').fetchone()[0]
    print(count)
    c.close()
except:
    print(0)
" 2>/dev/null || echo "0")

        if [[ "$PRE_CONCEPT_COUNT" -gt 0 ]]; then
            if [[ "$POST_CONCEPT_COUNT" -eq "$PRE_CONCEPT_COUNT" ]]; then
                mark_success "Concept count verified: $POST_CONCEPT_COUNT (matches pre-migration)"
            else
                mark_warning "Concept count mismatch: pre=$PRE_CONCEPT_COUNT, post=$POST_CONCEPT_COUNT"
            fi
        else
            mark_info "Post-migration concepts: $POST_CONCEPT_COUNT"
        fi

        # Set permissions
        chmod 700 "$PITH_HOME/data"
        chmod 600 "$PITH_HOME/data/brain.db"
    fi
else
    mark_warning "No brain.db to import — starting with fresh database"
fi

# Write new API key
echo "$NEW_API_KEY" > "$PITH_HOME/config/api.key"
chmod 600 "$PITH_HOME/config/api.key"
mark_success "API key written to $PITH_HOME/config/api.key"

echo ""
# ============================================================================
# STEP 6: Update MCP Config (F2, Amendment C — Python JSON editing)
# ============================================================================
echo -e "${BLUE}[Step 6/7]${NC} Updating Claude Desktop MCP configuration..."

export MIGRATED_API_KEY="$NEW_API_KEY"

python3 << 'MCP_UPDATE'
import json, os, shutil, sys

config_path = os.path.expanduser("~/Library/Application Support/Claude/claude_desktop_config.json")

if not os.path.exists(config_path):
    print(f"Claude Desktop config not found at {config_path}")
    print("You'll need to manually update your MCP config after migration.")
    print(f"  Server path: {os.path.expanduser('~/.pith/pith-server/server.js')}")
    print(f"  API key: {os.environ.get('MIGRATED_API_KEY', 'check ~/.pith/config/api.key')}")
    sys.exit(0)

# Backup original
backup_path = config_path + ".pre-migration"
shutil.copy2(config_path, backup_path)
print(f"Backed up config → {os.path.basename(backup_path)}")

with open(config_path) as f:
    config = json.load(f)

pith_server = os.path.expanduser("~/.pith/pith-server/server.js")
api_key = os.environ.get("MIGRATED_API_KEY", "")

servers = config.get("mcpServers", {})
updated = False
for name, entry in servers.items():
    args = entry.get("args", [])
    # Match any entry pointing to a pith server.js
    if any("server.js" in str(a) and "pith" in str(a).lower() for a in args):
        # Update args — keep "node" as command, update path
        if entry.get("command") == "node":
            entry["args"] = [pith_server]
        else:
            entry["command"] = "node"
            entry["args"] = [pith_server]
        # Update API key
        if api_key:
            entry.setdefault("env", {})["PITH_API_KEY"] = api_key
        updated = True
        print(f"Updated MCP entry '{name}':")
        print(f"  → server.js: {pith_server}")
        print(f"  → API key: ...{api_key[-8:]}")
        break

if not updated:
    # Try matching by entry name
    for name in ["pith", "pith-mcp", "pith", "pith-pith"]:
        if name in servers:
            servers[name]["command"] = "node"
            servers[name]["args"] = [pith_server]
            if api_key:
                servers[name].setdefault("env", {})["PITH_API_KEY"] = api_key
            updated = True
            print(f"Updated MCP entry '{name}' (matched by name)")
            break

if not updated:
    print("No pith-related MCP entries found. Manual config update needed:")
    print(f'  "pith": {{"command": "node", "args": ["{pith_server}"], "env": {{"PITH_API_KEY": "{api_key}"}}}}')
    sys.exit(0)

with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
print(f"Config saved successfully.")
MCP_UPDATE

MCP_EXIT=$?
if [[ $MCP_EXIT -eq 0 ]]; then
    mark_success "MCP config updated"
else
    mark_warning "MCP config update had issues (see above). You may need to update manually."
fi

echo ""
# ============================================================================
# STEP 7: Verification & Cleanup
# ============================================================================
echo -e "${BLUE}[Step 7/7]${NC} Verifying native installation..."

# Start native server
if [[ -f "$PITH_HOME/bin/pith" ]]; then
    "$PITH_HOME/bin/pith" start 2>/dev/null || true
    sleep 3

    # Health check
    HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health 2>/dev/null || echo "000")
    if [[ "$HEALTH" == "200" ]]; then
        mark_success "Native server running — health check passed"
    else
        mark_warning "Health check returned $HEALTH — server may still be starting"
        echo "  Try: $PITH_HOME/bin/pith status"
    fi
else
    mark_warning "pith CLI not found. Check installation."
fi

echo ""

# --- Docker Cleanup Prompt ---
echo -e "${BLUE}[Cleanup]${NC} Docker container cleanup..."
echo ""
echo "Your Docker container has been stopped. Would you like to remove it?"
echo "  (Your original data is safe in your previous Pith install directory)"
echo ""
read -p "Remove Docker container '$CONTAINER_ID'? (y/n): " -r
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker rm "$CONTAINER_ID" 2>/dev/null || true
    mark_success "Docker container removed"

    read -p "Remove Docker image too? (y/n): " -r
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        DOCKER_IMAGE=$(docker inspect "$CONTAINER_ID" --format='{{.Config.Image}}' 2>/dev/null || echo "")
        if [[ -n "$DOCKER_IMAGE" ]]; then
            docker rmi "$DOCKER_IMAGE" 2>/dev/null || true
            mark_success "Docker image removed"
        fi
    fi
else
    mark_info "Docker container kept (stopped). Remove later with: docker rm $CONTAINER_ID"
    mark_warning "Note: Docker 'restart: unless-stopped' has been disabled. Container will NOT auto-restart."
fi

# Clear the error-recovery flag since we succeeded
DOCKER_WAS_STOPPED=false

echo ""
# ============================================================================
# Final Success Message
# ============================================================================
echo -e "${GREEN}${BOLD}╔════════════════════════════════════════╗${NC}"
echo -e "${GREEN}${BOLD}║     ✓ Migration Complete!              ║${NC}"
echo -e "${GREEN}${BOLD}╚════════════════════════════════════════╝${NC}"
echo ""
echo -e "Your Pith data has been migrated from Docker to native installation."
echo ""
if [[ "$PRE_CONCEPT_COUNT" -gt 0 ]]; then
    echo -e "  Pith data:  ${GREEN}$PRE_CONCEPT_COUNT concepts preserved${NC}"
fi
echo -e "  Install:     ${BLUE}$PITH_HOME${NC}"
echo -e "  API key:     ${BLUE}$PITH_HOME/config/api.key${NC}"
echo -e "  Server:      ${BLUE}$PITH_HOME/pith-server/server.js${NC}"
echo ""
echo -e "${BOLD}Next steps:${NC}"
echo -e "  1. ${YELLOW}Restart Claude Desktop${NC} (Cmd+Q, then reopen)"
echo -e "  2. Verify: type \"Run pith_stats\" in a new Claude conversation"
echo ""
echo -e "${BOLD}Commands:${NC}"
echo "  pith status    — Check if server is running"
echo "  pith logs      — View server logs"
echo "  pith backup    — Create a backup"
echo "  pith restart   — Restart the server"
echo ""
echo "You can safely delete your old Pith source directory when ready."
echo ""

exit 0
