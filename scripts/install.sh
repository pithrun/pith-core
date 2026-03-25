#!/bin/bash
set -euo pipefail

# Pith Installer v1.0.0
# macOS/Linux bash installer

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DOWNLOAD_URL="${DOWNLOAD_URL:-https://github.com/esteyangandrew/pith-core/releases/latest/download}"
CHECKSUM_URL="${CHECKSUM_URL:-https://github.com/esteyangandrew/pith-core/releases/latest/download}"
PITH_HOME="${PITH_HOME:-$HOME/.pith}"
PITH_VERSION="1.0.0"
STEP_COUNT=9
CURRENT_STEP=0

# FIX S1: SHA-256 checksum variables
PITH_SERVER_FILENAME="pith-server-latest.tar.gz"
PITH_CHECKSUM_FILENAME="pith-server-latest.sha256"

# Print banner
print_banner() {
    clear
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════╗"
    echo "║   🧠 Pith Installer v${PITH_VERSION}       ║"
    echo "║      macOS/Linux Edition               ║"
    echo "╚════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
}

# Step indicator
print_step() {
    local step_num=$1
    local step_name=$2
    CURRENT_STEP=$step_num
    echo -e "${BLUE}[Step ${step_num}/${STEP_COUNT}]${NC} ${step_name}"
}

# Success indicator
mark_success() {
    echo -e "${GREEN}✓${NC} $1"
}

# Warning indicator
mark_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

# Error handler
error_exit() {
    echo -e "${RED}✗ ERROR:${NC} $1" >&2
    exit 1
}

# Cleanup on error
cleanup_on_failure() {
    echo ""
    echo -e "${RED}⚠ Installation failed at: Step ${CURRENT_STEP}${NC}"
    echo "  To retry: bash scripts/install.sh"
    echo "  To clean up: rm -rf $PITH_HOME/.venv"
    exit 1
}
trap cleanup_on_failure ERR
trap 'error_exit "Installation interrupted at step ${CURRENT_STEP}"' INT TERM

print_banner

# ============================================================================
# STEP 1: System Check
# ============================================================================
print_step 1 "System check (OS detection, Python, disk space, venv)"

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS_TYPE="macos"
    OS_NAME="macOS"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS_TYPE="linux"
    OS_NAME="Linux"
else
    error_exit "Unsupported OS: $OSTYPE. Only macOS and Linux are supported."
fi
mark_success "OS: $OS_NAME"

# FIX P1: macOS Xcode CLT detection
if [[ "$OS_TYPE" == "macos" ]]; then
    PYTHON_PATH=$(which python3 2>/dev/null || true)
    
    # Check if using Xcode CLT shim
    if [[ -n "$PYTHON_PATH" ]]; then
        if [[ "$PYTHON_PATH" == "/usr/bin/python3" ]]; then
            XCLT_PYTHON=$(/usr/bin/python3 --version 2>&1 | grep -o "3\.[0-9]*")
            if [[ "$XCLT_PYTHON" =~ ^3\.[0-9]+$ ]]; then
                # Xcode CLT provides older shim, recommend real installation
                mark_warning "Using Xcode CLT Python shim. Consider installing Python 3.9+ via Homebrew or python.org"
            fi
        fi
    fi
fi

# Check Python version
PYTHON_EXECUTABLE=$(command -v python3 || command -v python || true)
if [[ -z "$PYTHON_EXECUTABLE" ]]; then
    error_exit "Python 3 not found. Please install Python 3.9 or later."
fi

PYTHON_VERSION=$("$PYTHON_EXECUTABLE" --version 2>&1 | grep -oE "[0-9]+\.[0-9]+")
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [[ $PYTHON_MAJOR -lt 3 ]] || { [[ $PYTHON_MAJOR -eq 3 ]] && [[ $PYTHON_MINOR -lt 9 ]]; }; then
    error_exit "Python 3.9+ required. Found: $PYTHON_VERSION"
fi
mark_success "Python: $PYTHON_VERSION"

# Check for Microsoft Store Python (Windows Subsystem detection - warn on Linux)
if [[ "$OS_TYPE" == "linux" ]]; then
    if grep -qi microsoft /proc/version 2>/dev/null; then
        mark_warning "Running on WSL/Windows Subsystem. Performance may vary."
    fi
fi

# Check disk space (3GB required)
DISK_AVAILABLE=$(df "$HOME" | tail -1 | awk '{print $4}' | xargs)
DISK_REQUIRED_KB=$((3 * 1024 * 1024))  # 3GB in KB

if command -v numfmt &> /dev/null; then
    DISK_DISPLAY=$(numfmt --to=iec $((DISK_AVAILABLE * 1024)) 2>/dev/null || echo "${DISK_AVAILABLE}KB")
else
    DISK_DISPLAY="${DISK_AVAILABLE}KB"
fi

if [[ $DISK_AVAILABLE -lt $DISK_REQUIRED_KB ]]; then
    error_exit "Insufficient disk space. Required: 3GB, Available: $DISK_DISPLAY"
fi
mark_success "Disk space: $DISK_DISPLAY available"

# Check venv availability
if ! "$PYTHON_EXECUTABLE" -m venv --help &>/dev/null; then
    error_exit "Python venv module not available. Install python3-venv package."
fi
mark_success "Python venv module available"

echo ""

# ============================================================================
# STEP 2: Create Directory Structure
# ============================================================================
print_step 2 "Create ~/.pith/ directory structure"

mkdir -p "$PITH_HOME"/{bin,data,config,logs,cache,backups}
chmod 700 "$PITH_HOME"
mark_success "Created $PITH_HOME with subdirectories"

echo ""

# ============================================================================
# STEP 3: Download Pith Server with Checksum Verification
# ============================================================================
print_step 3 "Download Pith server with SHA-256 checksum verification [FIX S1]"

PITH_SERVER_PATH="$PITH_HOME/pith-server"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIST_DIR="$(dirname "$SCRIPT_DIR")"
DOWNLOAD_SUCCESS=false

# SYMLINK_SWAP_SPEC A2: Skip file copy if pith-server is a symlink (dev mode)
if [[ -L "$PITH_SERVER_PATH" ]]; then
    echo "⚠️  $PITH_SERVER_PATH is a symlink (dev mode). Skipping file copy."
    echo "    Target: $(readlink "$PITH_SERVER_PATH")"
    echo "    To update, pull changes in the dev repo directly."
    DOWNLOAD_SUCCESS=true

# Strategy 1: Detect running from distribution directory (most common for beta)
elif [[ -f "$DIST_DIR/app/server.py" ]] && [[ -f "$DIST_DIR/pith_mcp.py" ]]; then
    echo "Detected distribution directory at: $DIST_DIR"
    mkdir -p "$PITH_SERVER_PATH"
    # Copy app files from distribution to install location
    cp -r "$DIST_DIR/app" "$PITH_SERVER_PATH/"
    cp "$DIST_DIR/pith_mcp.py" "$PITH_SERVER_PATH/"
    cp "$DIST_DIR/skill_deployer.py" "$PITH_SERVER_PATH/" 2>/dev/null || true
    cp "$DIST_DIR/requirements.txt" "$PITH_SERVER_PATH/"
    cp "$DIST_DIR/.env.example" "$PITH_SERVER_PATH/" 2>/dev/null || true
    # Copy scripts (configure_clients.py etc.)
    if [[ -d "$DIST_DIR/scripts" ]]; then
        cp -r "$DIST_DIR/scripts" "$PITH_SERVER_PATH/"
    fi
    # Copy migrations if present
    if [[ -d "$DIST_DIR/migrations" ]]; then
        cp -r "$DIST_DIR/migrations" "$PITH_SERVER_PATH/"
    fi
    DOWNLOAD_SUCCESS=true
    mark_success "Installed from distribution directory"

# Strategy 2: Local tarball (created by build-release.sh)
elif [[ -f "$DIST_DIR/pith-server-latest.tar.gz" ]]; then
    mark_success "Using local Pith server tarball"
    mkdir -p "$PITH_SERVER_PATH"
    tar -xzf "$DIST_DIR/pith-server-latest.tar.gz" \
        -C "$PITH_SERVER_PATH"
    DOWNLOAD_SUCCESS=true

# Strategy 3: Download from hosted URL (GitHub Releases / CDN)
elif [[ -n "$DOWNLOAD_URL" ]]; then
    echo "Attempting download from: $DOWNLOAD_URL"
    
    TEMP_DOWNLOAD_DIR=$(mktemp -d)
    trap "rm -rf $TEMP_DOWNLOAD_DIR" EXIT
    
    if curl -fsSL --max-time 30 "$DOWNLOAD_URL/$PITH_SERVER_FILENAME" \
        -o "$TEMP_DOWNLOAD_DIR/$PITH_SERVER_FILENAME" 2>/dev/null && \
       curl -fsSL --max-time 30 "$CHECKSUM_URL/$PITH_CHECKSUM_FILENAME" \
        -o "$TEMP_DOWNLOAD_DIR/$PITH_CHECKSUM_FILENAME" 2>/dev/null; then
        
        # Verify checksum
        cd "$TEMP_DOWNLOAD_DIR"
        if sha256sum -c "$PITH_CHECKSUM_FILENAME" >/dev/null 2>&1 || \
           shasum -a 256 -c "$PITH_CHECKSUM_FILENAME" >/dev/null 2>&1; then
            mark_success "Download successful and checksum verified"
            mkdir -p "$PITH_SERVER_PATH"
            tar -xzf "$PITH_SERVER_FILENAME" -C "$PITH_SERVER_PATH"
            DOWNLOAD_SUCCESS=true
        else
            mark_warning "Checksum verification failed"
        fi
        cd - > /dev/null
    else
        mark_warning "Download failed"
    fi
fi

if [[ "$DOWNLOAD_SUCCESS" == false ]]; then
    error_exit "Could not locate Pith server files. Run this script from inside the pith-beta directory: cd /path/to/pith-beta && bash scripts/install.sh"
fi

echo ""

# ============================================================================
# STEP 4: Python venv Setup with Health Check
# ============================================================================
print_step 4 "Python venv setup with health check [FIX R1, R2, R3]"

VENV_PATH="$PITH_HOME/venv"

# FIX R1: Detect broken existing venv, recreate if needed
if [[ -d "$VENV_PATH" ]]; then
    if ! "$VENV_PATH/bin/python" -c "import sys; sys.exit(0)" 2>/dev/null; then
        mark_warning "Existing venv is broken, recreating"
        rm -rf "$VENV_PATH"
    fi
fi

# Create venv
if [[ ! -d "$VENV_PATH" ]]; then
    "$PYTHON_EXECUTABLE" -m venv "$VENV_PATH"
    mark_success "Created Python virtual environment"
else
    mark_success "Using existing virtual environment"
fi

# Activate venv
source "$VENV_PATH/bin/activate"

# Upgrade pip
pip install --quiet --upgrade pip setuptools wheel 2>/dev/null || true
mark_success "Updated pip, setuptools, wheel"

# Install dependencies in two phases: core (required) and ML (optional)
echo "Installing dependencies (this may take a moment)..."
REQ_FILE=""
if [[ -f "$PITH_SERVER_PATH/requirements.txt" ]]; then
    REQ_FILE="$PITH_SERVER_PATH/requirements.txt"
else
    REQ_FILE="$DIST_DIR/requirements.txt"
fi

# Phase 1: Core deps (must succeed) — everything except numpy/scikit-learn
CORE_TMP=$(mktemp)
grep -v -E '^(numpy|scikit-learn)' "$REQ_FILE" > "$CORE_TMP"
# NOTE: pip commands inside if-conditions to prevent set -e from aborting on failure
if ! pip install --quiet -r "$CORE_TMP" 2>/dev/null; then
    # Pinned versions may lack wheels for this Python — retry unpinned
    echo "  Retrying with flexible versions..."
    # Strip only exact pins (==), keep minimum pins (>=) and extras ([standard])
    sed 's/==[^,]*//' "$CORE_TMP" > "${CORE_TMP}.flex"
    if ! pip install --quiet -r "${CORE_TMP}.flex" 2>/dev/null; then
        # Final retry: show errors
        if ! pip install -r "${CORE_TMP}.flex"; then
            rm -f "$CORE_TMP" "${CORE_TMP}.flex"
            mark_error "Failed to install core dependencies"
            exit 1
        fi
    fi
    rm -f "${CORE_TMP}.flex"
fi
rm -f "$CORE_TMP"
mark_success "Installed core dependencies"

# Phase 2: ML deps (optional) — numpy + scikit-learn for TF-IDF search
echo "Installing ML dependencies (numpy, scikit-learn)..."
if pip install --quiet numpy scikit-learn 2>/dev/null; then
    mark_success "Installed ML dependencies (numpy, scikit-learn)"
else
    mark_warning "Could not install numpy/scikit-learn (Python $(python3 --version 2>&1 | cut -d' ' -f2) may lack binary wheels)."
    echo "  Pith will run without TF-IDF search. This is non-critical."
fi

# Platform-aware embedding installation (F7+F8)
EMBED_LOG="$PITH_HOME/logs/embedding_install.log"
mkdir -p "$PITH_HOME/logs"

install_pytorch() {
    local OS_NAME
    OS_NAME=$(uname -s)
    local ARCH
    ARCH=$(uname -m)

    if [ "$OS_NAME" = "Darwin" ]; then
        if [ "$ARCH" = "arm64" ]; then
            # Apple Silicon — latest PyTorch with MPS support
            pip install --quiet torch 2>"$EMBED_LOG"
        else
            # Intel Mac — check Python version ceiling
            local PY_MINOR
            PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
            if [ "$PY_MINOR" -gt 12 ]; then
                echo "⚠ Python 3.$PY_MINOR on Intel Mac. PyTorch 2.2.2 requires Python ≤3.12."
                echo "  Embeddings unavailable. Pith will use TF-IDF search."
                return 1
            fi
            echo "⚠ Intel Mac detected. Installing PyTorch 2.2.2 (last supported version)."
            pip install --quiet "torch==2.2.2" 2>"$EMBED_LOG"
        fi
    elif [ "$OS_NAME" = "Linux" ]; then
        pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu 2>"$EMBED_LOG"
    else
        echo "⚠ Unknown OS. Attempting default PyTorch install..."
        pip install --quiet torch 2>"$EMBED_LOG"
    fi
}

install_embeddings() {
    install_pytorch
    local TORCH_EXIT=$?

    if [ $TORCH_EXIT -eq 0 ]; then
        # Pin sentence-transformers to <4.0 for Intel Mac compat safety
        pip install --quiet "sentence-transformers>=3.0.0,<4.0.0" 2>>"$EMBED_LOG"
        if [ $? -eq 0 ]; then
            mark_success "Semantic embeddings enabled (all-MiniLM-L6-v2)"
            # Pre-download model to avoid first-use delay
            python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" 2>/dev/null
            echo "embeddings=true" > "$PITH_HOME/.install_capabilities"
            echo "pytorch=$(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null)" >> "$PITH_HOME/.install_capabilities"
            echo "arch=$(uname -m)" >> "$PITH_HOME/.install_capabilities"
            return 0
        fi
    fi

    mark_warning "Could not install PyTorch/sentence-transformers."
    echo "  Pith will run with TF-IDF search (reduced semantic quality)."
    echo "  This is fully functional — embeddings are an optional upgrade."
    echo "  Details: $EMBED_LOG"
    echo "embeddings=false" > "$PITH_HOME/.install_capabilities"
    echo "reason=install_failed" >> "$PITH_HOME/.install_capabilities"
    return 1
}

echo "Installing embeddings (platform-aware)..."
install_embeddings || true

deactivate

echo ""

# ============================================================================
# STEP 5: Generate API Key with Secure Permissions
# ============================================================================
print_step 5 "Generate API key with secure file permissions [FIX S2]"

API_KEY_FILE="$PITH_HOME/config/api.key"
if [[ ! -f "$API_KEY_FILE" ]]; then
    # Generate 32-byte random key as hex (64 chars)
    API_KEY=$(openssl rand -hex 32)
    echo "$API_KEY" > "$API_KEY_FILE"
    # FIX S2: Secure file permissions (owner read-only)
    chmod 600 "$API_KEY_FILE"
    mark_success "Generated API key: ${API_KEY:0:16}... (saved to $API_KEY_FILE)"
    # Create .env with API key
    if [[ ! -f "$PITH_HOME/.env" ]]; then
        echo "PITH_API_KEY=$API_KEY" > "$PITH_HOME/.env"
        chmod 600 "$PITH_HOME/.env"
        mark_success "Created .env with secure permissions"
    fi
else
    mark_success "API key already exists"
fi

echo ""

# ============================================================================
# STEP 6: Configure MCP Clients
# ============================================================================
print_step 6 "Configure MCP clients using configure_clients.py"

# Try real configure_clients.py first (supports 6 MCP clients)
source "$VENV_PATH/bin/activate"
API_KEY=$(cat "$API_KEY_FILE" 2>/dev/null || echo "")
CONFIGURE_SCRIPT="$PITH_SERVER_PATH/scripts/configure_clients.py"

if [[ -f "$CONFIGURE_SCRIPT" ]] && python3 "$CONFIGURE_SCRIPT" \
    --server-path "$PITH_SERVER_PATH/pith_mcp.py" \
    --python-cmd "$VENV_PATH/bin/python3" \
    --api-key "$API_KEY" \
    --project-dir "$PITH_SERVER_PATH" \
    --platform "$(uname -s | tr '[:upper:]' '[:lower:]')" \
    --json 2>/dev/null; then
    mark_success "MCP clients configured (6 clients)"
else
    # Fallback: configure Claude Desktop only
    mark_warning "Full client config unavailable. Configuring Claude Desktop only..."
    CLAUDE_CONFIG=""
    if [[ "$OS_TYPE" == "macos" ]]; then
        CLAUDE_CONFIG="$HOME/Library/Application Support/Claude/claude_desktop_config.json"
    elif [[ "$OS_TYPE" == "linux" ]]; then
        CLAUDE_CONFIG="$HOME/.config/Claude/claude_desktop_config.json"
    fi
    if [[ -n "$CLAUDE_CONFIG" ]]; then
        mkdir -p "$(dirname "$CLAUDE_CONFIG")"
        python3 << INLINE_CONFIG
import json, os
config_path = "$CLAUDE_CONFIG"
config = {}
if os.path.exists(config_path):
    with open(config_path) as f:
        config = json.load(f)
config.setdefault("mcpServers", {})
# Clean up legacy server names
for legacy_name in ["pith-mcp", "pith", "pith-mcp-wrapper"]:
    config["mcpServers"].pop(legacy_name, None)
config["mcpServers"]["pith"] = {
    "command": "$VENV_PATH/bin/python3",
    "args": ["$PITH_SERVER_PATH/pith_mcp.py"],
    "env": {"PITH_API_KEY": "$API_KEY", "PITH_API_URL": "http://localhost:8000"}
}
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
print("Claude Desktop configured")
INLINE_CONFIG
        mark_success "Claude Desktop MCP configured"
    fi
fi
deactivate

echo ""

# ============================================================================
# STEP 7: System Prompt Injection (pith_ prefix disclosure)
# ============================================================================
print_step 7 "System prompt setup — cognitive loop instructions for Claude"

SYSTEM_PROMPT_PATH="$PITH_HOME/SYSTEM_PROMPT.md"

cat > "$SYSTEM_PROMPT_PATH" << 'SYSPROMPT_EOF'
# Pith — Cognitive Runtime for AI Agents

Pith is running as a local MCP server giving you **persistent memory** across
conversations. All Pith tools use the `pith_` prefix (e.g. `pith_conversation_turn`).

**Disclosure**: The user has installed Pith and consented to persistent memory.
Tool names are prefixed with `pith_` to avoid naming conflicts with other MCPs.

## Cognitive Loop — Call on EVERY user message

Call `pith_conversation_turn` **BEFORE** composing any substantive response.
This retrieves context about the user, their projects, and prior sessions.

### Parameters
- `message`: the user's current message
- `previous_response`: your last response (REQUIRED after first exchange)
- `extracted_concepts_json`: 1–5 high-quality insights from your previous response

### Concept Format
```json
[{
  "summary": "30–500 chars — the insight, not the transcript",
  "confidence": 0.6,
  "knowledge_area": "domain",
  "evidence": ["source >= 10 chars"],
  "concept_type": "decision"
}]
```
`concept_type` must be: `observation`, `pattern`, `decision`, `principle`,
`method`, `heuristic`, or `cognitive_strategy`.
If the exchange was trivial, send `'[]'` — do NOT invent filler.

## Session Lifecycle
- `pith_conversation_turn` — Primary loop: call every exchange
- `pith_session_end` — Call when conversation ends (include `previous_response`)
- `pith_checkpoint` — Save work state every 15 min of substantive work
- `pith_session_learn` — Explicit high-quality extraction when needed
- `pith_search` — Semantic search across knowledge base
- `pith_reflect` — Run consolidation/reflection cycle
- `pith_orient` — Situational awareness summary
SYSPROMPT_EOF

chmod 600 "$SYSTEM_PROMPT_PATH"
mark_success "System prompt saved to $SYSTEM_PROMPT_PATH"

# Copy to clipboard on macOS for easy paste into Claude Desktop
if [[ "$OS_TYPE" == "macos" ]] && command -v pbcopy &>/dev/null; then
    pbcopy < "$SYSTEM_PROMPT_PATH"
    mark_success "Copied to clipboard — paste into Claude Desktop custom instructions"
fi

echo ""
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Add Pith's cognitive instructions to Claude Desktop:${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  1. Open Claude Desktop → Settings → Custom Instructions"
if [[ "$OS_TYPE" == "macos" ]] && command -v pbcopy &>/dev/null; then
    echo "  2. Paste (already in your clipboard)"
else
    echo "  2. Paste the contents of:"
    echo "     ${YELLOW}$SYSTEM_PROMPT_PATH${NC}"
fi
echo "  3. Save — Claude will now use Pith's cognitive loop automatically"
echo ""
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# ============================================================================
# STEP 8: Auto-start Setup
# ============================================================================
print_step 8 "Auto-start setup (launchd/systemd) and backup scheduler"

if [[ "$OS_TYPE" == "macos" ]]; then
    # FIX P2: macOS launchd plist with post-load verification
    LAUNCHD_PLIST="$HOME/Library/LaunchAgents/dev.pith.server.plist"
    mkdir -p "$(dirname "$LAUNCHD_PLIST")"
    
    cat > "$LAUNCHD_PLIST" << LAUNCHD_CONFIG
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>dev.pith.server</string>
    <key>ProgramArguments</key>
    <array>
        <string>$PITH_HOME/bin/pith</string>
        <string>start</string>
$(if [ -n "$PITH_PROFILE" ]; then cat << PROFILE_ARGS
        <string>--profile</string>
        <string>$PITH_PROFILE</string>
PROFILE_ARGS
fi)
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$PITH_HOME/logs/pith.log</string>
    <key>StandardErrorPath</key>
    <string>$PITH_HOME/logs/pith.err</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PITH_HOME</key>
        <string>$PITH_HOME</string>
$(if [ -n "$PITH_PROFILE" ]; then cat << PROFILE_ENV
        <key>PITH_PROFILE</key>
        <string>$PITH_PROFILE</string>
PROFILE_ENV
fi)
    </dict>
</dict>
</plist>
LAUNCHD_CONFIG
    
    chmod 644 "$LAUNCHD_PLIST"
    mark_success "Created launchd plist at $LAUNCHD_PLIST"
    # NOTE: Do NOT launchctl load here — the pith CLI script is created in Step 8.
    # RunAtLoad will start the service on next login. Loading now would cause:
    # (1) execution of non-existent $PITH_HOME/bin/pith, (2) KeepAlive retry loop,
    # (3) port 8000 collision with the health check that follows in Step 8.
    mark_success "Pith will auto-start on next login (launchd RunAtLoad)"
    
elif [[ "$OS_TYPE" == "linux" ]]; then
    # Linux: systemd user service
    SYSTEMD_USER_DIR="$HOME/.config/systemd/user"
    mkdir -p "$SYSTEMD_USER_DIR"
    
    cat > "$SYSTEMD_USER_DIR/pith-server.service" << SYSTEMD_CONFIG
[Unit]
Description=Pith Server
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$PITH_HOME
ExecStart=$PITH_HOME/bin/pith start
Restart=always
RestartSec=10
Environment="PITH_HOME=$PITH_HOME"
StandardOutput=append:$PITH_HOME/logs/pith.log
StandardError=append:$PITH_HOME/logs/pith.err

[Install]
WantedBy=default.target
SYSTEMD_CONFIG
    
    chmod 644 "$SYSTEMD_USER_DIR/pith-server.service"
    mark_success "Created systemd user service"
    
    if systemctl --user daemon-reload 2>/dev/null && \
       systemctl --user enable pith-server.service 2>/dev/null; then
        mark_success "Enabled systemd service (Pith will start at login)"
    else
        mark_warning "Could not enable systemd service (may require additional setup)"
    fi
fi

# Setup backup wrapper (delegates to WAL-safe script)
BACKUP_WRAPPER="$PITH_HOME/bin/backup"
cat > "$BACKUP_WRAPPER" << 'BACKUP_WRAPPER_CONTENT'
#!/bin/bash
# Pith backup wrapper — delegates to WAL-safe backup script
PITH_HOME="${PITH_HOME:-$HOME/.pith}"
SAFE_BACKUP="$PITH_HOME/pith-server/scripts/backup/safe_backup.sh"
if [[ -f "$SAFE_BACKUP" ]]; then
    bash "$SAFE_BACKUP" "$@"
else
    echo "Error: safe_backup.sh not found at $SAFE_BACKUP"
    echo "Run the installer again or create a backup manually."
    exit 1
fi
BACKUP_WRAPPER_CONTENT

chmod +x "$BACKUP_WRAPPER"

# Schedule backup
if [[ "$OS_TYPE" == "macos" ]]; then
    # Use launchd timer instead of cron (crontab write hangs on some macOS configs)
    BACKUP_PLIST="$HOME/Library/LaunchAgents/dev.pith.backup.plist"
    cat > "$BACKUP_PLIST" << BACKUP_PLIST_CONTENT
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>dev.pith.backup</string>
    <key>ProgramArguments</key>
    <array>
        <string>$PITH_HOME/bin/backup</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>2</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>$PITH_HOME/logs/backup.log</string>
    <key>StandardErrorPath</key>
    <string>$PITH_HOME/logs/backup.log</string>
</dict>
</plist>
BACKUP_PLIST_CONTENT
    mark_success "Scheduled daily backups at 2:00 AM (launchd)"
elif [[ "$OS_TYPE" == "linux" ]]; then
    # Create systemd timer
    TIMER_PATH="$HOME/.config/systemd/user/pith-backup.timer"
    SERVICE_PATH="$HOME/.config/systemd/user/pith-backup.service"
    
    cat > "$SERVICE_PATH" << 'BACKUP_SERVICE'
[Unit]
Description=Pith Backup Service
After=pith-server.service

[Service]
Type=oneshot
ExecStart=/home/%(user)s/.pith/bin/backup
BACKUP_SERVICE
    
    cat > "$TIMER_PATH" << 'BACKUP_TIMER'
[Unit]
Description=Pith Daily Backup Timer
Requires=pith-backup.service

[Timer]
OnCalendar=*-*-* 02:00:00
Persistent=true

[Install]
WantedBy=timers.target
BACKUP_TIMER
    
    systemctl --user daemon-reload 2>/dev/null || true
    systemctl --user enable pith-backup.timer 2>/dev/null || true
    mark_success "Scheduled daily backups at 2:00 AM via systemd timer"
fi

echo ""

# ============================================================================
# STEP 9: Health Check
# ============================================================================
print_step 9 "Health check (30s timeout)"

# Create CLI script first (needed for health check)
# === CLI_TEMPLATE_START ===
cat > "$PITH_HOME/bin/pith" << 'PITH_CLI_SCRIPT'
#!/bin/bash
# Pith CLI wrapper
PITH_VERSION="__PITH_VERSION__"

PITH_HOME="${PITH_HOME:-$HOME/.pith}"
VENV_PATH="$PITH_HOME/venv"
PITH_SERVER_PATH="$PITH_HOME/pith-server"

# Parse --profile flag from any position
PROFILE=""
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile)
            PROFILE="$2"
            shift 2
            ;;
        --profile=*)
            PROFILE="${1#*=}"
            shift
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done
set -- "${ARGS[@]}"

# Source persistent .env if it exists (sets PITH_PROFILE, etc.)
if [[ -f "$PITH_HOME/.env" ]]; then
    set -a  # auto-export sourced variables
    source "$PITH_HOME/.env"
    set +a
fi

# Export profile as env var so app/profile.py picks it up
# CLI --profile flag overrides .env
if [[ -n "$PROFILE" ]]; then
    export PITH_PROFILE="$PROFILE"
    echo "Using profile: $PROFILE (data: $HOME/pith-data/$PROFILE/)"
elif [[ -n "${PITH_PROFILE:-}" ]]; then
    echo "Using profile from .env: $PITH_PROFILE (data: $HOME/pith-data/$PITH_PROFILE/)"
fi

# Ensure venv is activated
if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
    echo "Error: Pith venv not found at $VENV_PATH"
    exit 1
fi

source "$VENV_PATH/bin/activate"

# --- Profile-aware path resolution (mirrors app/profile.py) ---
# NOTE: Python source of truth is app/profile.py. Keep in sync.
resolve_data_dir() {
    if [[ -n "${PITH_DATA_DIR:-}" ]]; then echo "$PITH_DATA_DIR"; return; fi
    if [[ -n "${PITH_PROFILE:-}" ]]; then echo "$HOME/pith-data/$PITH_PROFILE"; return; fi
    if [[ -n "${PITH_DATA_DIR:-}" ]]; then echo "$PITH_DATA_DIR"; return; fi
    echo "$HOME/pith-data/default"
}

resolve_db_path() {
    echo "$(resolve_data_dir)/pith.db"
}

resolve_log_path() {
    local DATA_DIR
    DATA_DIR=$(resolve_data_dir)
    for candidate in \
        "$DATA_DIR/logs/pith.log" \
        "$PITH_SERVER_PATH/logs/pith.log" \
        "$PITH_HOME/logs/pith.log"; do
        if [[ -f "$candidate" ]]; then
            echo "$candidate"
            return
        fi
    done
    echo ""
}

validate_pid() {
    local PID_FILE="$PITH_HOME/pith.pid"
    if [[ ! -f "$PID_FILE" ]]; then return 1; fi
    local PID_VAL
    PID_VAL=$(cat "$PID_FILE")
    if ! ps -p "$PID_VAL" > /dev/null 2>&1; then
        rm -f "$PID_FILE"
        return 1
    fi
    local CMD
    CMD=$(ps -p "$PID_VAL" -o comm= 2>/dev/null)
    if [[ "$CMD" != *"python"* && "$CMD" != *"uvicorn"* ]]; then
        echo "Warning: PID $PID_VAL is $CMD, not Pith. Cleaning stale PID."
        rm -f "$PID_FILE"
        return 1
    fi
    return 0
}

check_pith_health() {
    local RESPONSE
    RESPONSE=$(curl -s --max-time 3 http://127.0.0.1:8000/health 2>/dev/null)
    if [[ -z "$RESPONSE" ]]; then echo "Unreachable"; return 1; fi
    if echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('service')=='pith'" 2>/dev/null; then
        echo "OK (Pith)"
        return 0
    else
        echo "Port 8000 responding but NOT Pith"
        return 1
    fi
}

check_port() {
    # Try lsof (macOS/most Linux), then ss (modern Linux), then netstat
    if command -v lsof >/dev/null 2>&1; then
        lsof -i :8000 -sTCP:LISTEN -t 2>/dev/null | head -1
    elif command -v ss >/dev/null 2>&1; then
        ss -tlnp 'sport = :8000' 2>/dev/null | awk 'NR>1{gsub(/.*pid=/,""); gsub(/,.*/,""); print; exit}'
    elif command -v netstat >/dev/null 2>&1; then
        netstat -tlnp 2>/dev/null | awk '/:8000 /{gsub(/.*\//,""); print; exit}'
    fi
}

case "${1:-status}" in
    start)
        echo "Starting Pith Server..."

        # Pre-check: port availability
        BLOCKING_PID=$(check_port)
        if [[ -n "$BLOCKING_PID" ]]; then
            BLOCKING_CMD=$(ps -p "$BLOCKING_PID" -o comm= 2>/dev/null || echo "unknown")
            echo "Error: Port 8000 is already in use by $BLOCKING_CMD (PID $BLOCKING_PID)"
            echo "Stop the other process or use 'pith stop' first."
            exit 1
        fi

        # Pre-check: stale PID cleanup
        if [[ -f "$PITH_HOME/pith.pid" ]]; then
            OLD_PID=$(cat "$PITH_HOME/pith.pid")
            if ! ps -p "$OLD_PID" > /dev/null 2>&1; then
                echo "Cleaning up stale PID file..."
                rm -f "$PITH_HOME/pith.pid"
            else
                echo "Error: Pith already running (PID $OLD_PID). Use 'pith stop' first."
                exit 1
            fi
        fi

        cd "$PITH_SERVER_PATH"
        python -m uvicorn app.server:app --host 127.0.0.1 --port 8000 --log-level info &
        SERVER_PID=$!

        # Poll for startup (up to 10s, check every 500ms)
        echo -n "Waiting for server..."
        for i in $(seq 1 20); do
            sleep 0.5
            if ! ps -p "$SERVER_PID" > /dev/null 2>&1; then
                echo " FAILED"
                echo "Server process exited during startup. Check logs:"
                LOG_FILE=$(resolve_log_path)
                if [[ -n "$LOG_FILE" ]]; then tail -20 "$LOG_FILE"; fi
                exit 1
            fi
            if curl -s --max-time 1 http://127.0.0.1:8000/health >/dev/null 2>&1; then
                echo " OK"
                echo "$SERVER_PID" > "$PITH_HOME/pith.pid"
                echo "Pith started successfully (PID: $SERVER_PID)"
                break
            fi
            echo -n "."
        done

        # Final check — if we exhausted the loop
        if [[ ! -f "$PITH_HOME/pith.pid" ]] || [[ "$(cat "$PITH_HOME/pith.pid" 2>/dev/null)" != "$SERVER_PID" ]]; then
            if ps -p "$SERVER_PID" > /dev/null 2>&1; then
                echo ""
                echo "Warning: Server running but health check not responding yet (PID: $SERVER_PID)"
                echo "$SERVER_PID" > "$PITH_HOME/pith.pid"
            else
                echo ""
                echo "Failed to start Pith"
                exit 1
            fi
        fi
        ;;
    stop)
        echo "Stopping Pith Server..."
        if [[ -f "$PITH_HOME/pith.pid" ]]; then
            kill $(cat "$PITH_HOME/pith.pid") 2>/dev/null || true
            rm -f "$PITH_HOME/pith.pid"
            echo "Pith stopped"
        else
            # No PID file — check if server is running anyway (manual start, etc.)
            ORPHAN_PID=$(check_port)
            if [[ -n "$ORPHAN_PID" ]]; then
                echo "Found orphan Pith process (PID: $ORPHAN_PID) — stopping..."
                kill "$ORPHAN_PID" 2>/dev/null || true
                rm -f "$PITH_HOME/pith.pid"
                echo "Pith stopped"
            else
                echo "Pith is not running"
            fi
        fi
        ;;
    restart)
        # OPS-099: Under launchd KeepAlive, stop+start races — launchd respawns within ~200ms
        # after stop, so start finds port 8000 occupied and exits 1. Use kickstart -k instead.
        _LAUNCHD_SVC="gui/$(id -u)/dev.pith.server"
        if launchctl print "$_LAUNCHD_SVC" >/dev/null 2>&1; then
            echo "Restarting via launchctl kickstart -k (launchd-managed)..."
            launchctl kickstart -k "$_LAUNCHD_SVC"
        else
            $0 stop
            sleep 1
            $0 start
        fi
        ;;
    status)
        if validate_pid; then
            echo "Pith is running (PID: $(cat "$PITH_HOME/pith.pid"))"
            HEALTH_RESULT=$(check_pith_health)
            echo "Health: $HEALTH_RESULT"
        else
            # No PID file — but server might be running (manual start, crashed PID file, etc.)
            HEALTH_RESULT=$(check_pith_health)
            if [[ "$HEALTH_RESULT" == "OK (Pith)" ]]; then
                ORPHAN_PID=$(check_port)
                echo "Pith is running (PID: ${ORPHAN_PID:-unknown}) [recovered — PID file was missing]"
                echo "Health: $HEALTH_RESULT"
                # Auto-recover: write PID file so future commands work normally
                if [[ -n "$ORPHAN_PID" ]]; then
                    echo "$ORPHAN_PID" > "$PITH_HOME/pith.pid"
                    echo "PID file restored."
                fi
            else
                echo "Pith is not running"
            fi
        fi
        ;;
    logs)
        LOG_FILE=$(resolve_log_path)
        if [[ -n "$LOG_FILE" ]]; then
            echo "Tailing: $LOG_FILE"
            tail -f "$LOG_FILE"
        else
            echo "No logs found. Checked:"
            echo "  $(resolve_data_dir)/logs/pith.log"
            echo "  $PITH_SERVER_PATH/logs/pith.log"
            echo "  $PITH_HOME/logs/pith.log"
        fi
        ;;
    backup)
        "$PITH_HOME/bin/backup"
        ;;
    restore)
        # Safe restore: stop → copy → verify → start
        BACKUP_FILE="${2:-}"
        if [[ -z "$BACKUP_FILE" ]]; then
            # Find most recent backup
            BACKUP_FILE=$(ls -t "$PITH_HOME/backups/"*.db 2>/dev/null | head -1)
            if [[ -z "$BACKUP_FILE" ]]; then
                echo "No backups found in $PITH_HOME/backups/"
                echo "Usage: pith restore [backup_file]"
                exit 1
            fi
            echo "Using most recent backup: $BACKUP_FILE"
        fi
        if [[ ! -f "$BACKUP_FILE" ]]; then
            echo "Backup file not found: $BACKUP_FILE"
            exit 1
        fi
        # Stop server before restore
        $0 stop
        sleep 1
        # Copy backup to profile-aware data directory
        RESTORE_DB=$(resolve_db_path)
        mkdir -p "$(dirname "$RESTORE_DB")"
        cp "$BACKUP_FILE" "$RESTORE_DB"
        # Verify integrity
        if python3 -c "import sqlite3; c=sqlite3.connect('$RESTORE_DB'); print(c.execute('PRAGMA integrity_check').fetchone()[0]); c.close()" 2>/dev/null | grep -q "ok"; then
            echo "✓ Database integrity verified"
        else
            echo "⚠ Database integrity check failed — backup may be corrupted"
        fi
        # Restart
        $0 start
        echo "Restored from: $BACKUP_FILE"
        ;;
    update)
        echo "Checking for updates..."
        source "$VENV_PATH/bin/activate"
        # Run any migration scripts if they exist
        if [[ -f "$PITH_SERVER_PATH/migrations/run.sh" ]]; then
            echo "Running migrations..."
            bash "$PITH_SERVER_PATH/migrations/run.sh"
        fi
        # Upgrade core deps
        pip install --quiet --upgrade -r "$PITH_SERVER_PATH/requirements.txt" 2>/dev/null || true
        # Re-run platform-aware embedding installation
        echo "Checking embedding dependencies..."
        # Source the install_embeddings function if available
        INSTALLER="$PITH_SERVER_PATH/scripts/install.sh"
        if [[ -f "$INSTALLER" ]]; then
            # Extract and run the install functions
            source <(sed -n '/^install_pytorch()/,/^}/p; /^install_embeddings()/,/^}/p' "$INSTALLER" 2>/dev/null) 2>/dev/null
            install_embeddings 2>/dev/null || true
        fi
        echo "Update complete"
        ;;
    version)
        echo "Pith v${PITH_VERSION:-unknown}"
        echo "Python: $(python3 --version 2>/dev/null || echo 'not found')"
        echo "OS: $(uname -s) $(uname -m)"
        # Read capabilities
        if [[ -f "$PITH_HOME/.install_capabilities" ]]; then
            while IFS='=' read -r key value; do
                case "$key" in
                    embeddings) echo "Embeddings: $value" ;;
                    pytorch) echo "PyTorch: $value" ;;
                    reason) echo "Note: $value" ;;
                esac
            done < "$PITH_HOME/.install_capabilities"
        else
            echo "Embeddings: unknown (no capability marker)"
        fi
        # Check if running
        if [[ -f "$PITH_HOME/pith.pid" ]] && ps -p $(cat "$PITH_HOME/pith.pid") > /dev/null 2>&1; then
            echo "Status: running (PID: $(cat "$PITH_HOME/pith.pid"))"
        else
            echo "Status: stopped"
        fi
        ;;
    uninstall)
        echo "WARNING: This will uninstall Pith and remove all data."
        read -p "Are you sure? (type 'yes' to confirm): " -r
        if [[ $REPLY == "yes" ]]; then
            $0 stop
            if [[ "$OSTYPE" == "darwin"* ]]; then
                launchctl unload "$HOME/Library/LaunchAgents/dev.pith.server.plist" 2>/dev/null || true
                launchctl unload "$HOME/Library/LaunchAgents/dev.pith.backup.plist" 2>/dev/null || true
                rm -f "$HOME/Library/LaunchAgents/dev.pith.server.plist"
                rm -f "$HOME/Library/LaunchAgents/dev.pith.backup.plist"
            elif [[ "$OSTYPE" == "linux"* ]]; then
                systemctl --user stop pith-server.service 2>/dev/null || true
                systemctl --user disable pith-server.service 2>/dev/null || true
                systemctl --user stop pith-backup.timer 2>/dev/null || true
                systemctl --user disable pith-backup.timer 2>/dev/null || true
                rm -f "$HOME/.config/systemd/user/pith-server.service"
                rm -f "$HOME/.config/systemd/user/pith-backup.service"
                rm -f "$HOME/.config/systemd/user/pith-backup.timer"
                systemctl --user daemon-reload 2>/dev/null || true
            fi
            rm -rf "$PITH_HOME"
            echo "Pith uninstalled"
        else
            echo "Uninstall cancelled"
        fi
        ;;
    profiles)
        echo "Available profiles in $HOME/pith-data/:"
        if [[ -d "$HOME/pith-data" ]]; then
            for d in "$HOME/pith-data"/*/; do
                PROF_NAME=$(basename "$d")
                [[ "$PROF_NAME" == "." || "$PROF_NAME" == ".." ]] && continue
                DB_SIZE=""
                if [[ -f "$d/pith.db" ]]; then
                    DB_SIZE=" ($(du -sh "$d/pith.db" 2>/dev/null | cut -f1))"
                elif [[ -f "$d/brain.db" ]]; then
                    DB_SIZE=" ($(du -sh "$d/brain.db" 2>/dev/null | cut -f1))"
                elif [[ -f "$d/data/pith.db" ]]; then
                    DB_SIZE=" ($(du -sh "$d/data/pith.db" 2>/dev/null | cut -f1))"
                elif [[ -f "$d/data/brain.db" ]]; then
                    DB_SIZE=" ($(du -sh "$d/data/brain.db" 2>/dev/null | cut -f1))"
                fi
                ACTIVE=""
                [[ "$PROF_NAME" == "${PITH_PROFILE:-default}" ]] && ACTIVE=" [active]"
                echo "  • $PROF_NAME$DB_SIZE$ACTIVE"
            done
        else
            echo "  No profiles found. Create one with: mkdir -p ~/pith-data/myprofile/"
        fi
        ;;
    maintenance)
        MAINT_CMD="${ARGS[1]:-run}"
        case "$MAINT_CMD" in
            run)
                echo "Running Pith maintenance cycle..."
                cd "$PITH_SERVER_PATH"
                python3 -m app.maintenance_cli run "${ARGS[@]:2}"
                ;;
            status)
                cd "$PITH_SERVER_PATH"
                python3 -m app.maintenance_cli status
                ;;
            install)
                echo "Installing maintenance scheduler..."
                cd "$PITH_SERVER_PATH"
                python3 -m app.maintenance_cli install
                ;;
            uninstall)
                cd "$PITH_SERVER_PATH"
                python3 -m app.maintenance_cli uninstall
                ;;
            *)
                echo "Usage: pith maintenance {run|status|install|uninstall}"
                echo "  run [--phases 1,2,3] [--dry-run]  Run maintenance cycle"
                echo "  status                             Show task status"
                echo "  install                            Install launchd scheduler (every 6h)"
                echo "  uninstall                          Remove scheduler"
                ;;
        esac
        ;;
    report)
        echo "Pith Diagnostics Report"
        echo "=============================="
        echo "Generated: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        echo ""

        # System
        echo "[System]"
        echo "  OS:           $(uname -s) $(uname -r) $(uname -m)"
        echo "  Shell:        $SHELL"
        echo "  Python:       $(python3 --version 2>/dev/null || echo 'not found')"
        NODE_VER=$(node --version 2>/dev/null)
        [[ -n "$NODE_VER" ]] && echo "  Node:         $NODE_VER"
        if [[ "$OSTYPE" == "darwin"* ]]; then
            DISK_FREE=$(df -h / | awk 'NR==2{print $4}')
        else
            DISK_FREE=$(df -h / | awk 'NR==2{print $4}')
        fi
        echo "  Disk Free:    $DISK_FREE"
        echo ""

        # Installation
        echo "[Installation]"
        echo "  Pith Home:    $PITH_HOME"
        echo "  Version:      ${PITH_VERSION:-unknown}"
        if [[ -f "$PITH_SERVER_PATH/pith_mcp.py" ]]; then
            echo "  MCP Bridge:   pith_mcp.py (Python)"
        fi
        if [[ -f "$PITH_HOME/.install_capabilities" ]]; then
            while IFS= read -r line; do
                echo "  $line"
            done < "$PITH_HOME/.install_capabilities"
        else
            echo "  Embeddings:   unknown (no .install_capabilities)"
        fi
        echo ""

        # Server
        echo "[Server]"
        if validate_pid; then
            PID_VAL=$(cat "$PITH_HOME/pith.pid")
            echo "  Status:       Running (PID $PID_VAL)"
            if [[ "$OSTYPE" == "darwin"* ]]; then
                START_TIME=$(ps -p "$PID_VAL" -o lstart= 2>/dev/null)
                [[ -n "$START_TIME" ]] && echo "  Started:      $START_TIME"
            else
                ELAPSED=$(ps -p "$PID_VAL" -o etime= 2>/dev/null | xargs)
                [[ -n "$ELAPSED" ]] && echo "  Uptime:       $ELAPSED"
            fi
        else
            echo "  Status:       Not running"
        fi
        echo "  Port:         8000"
        HEALTH_RESULT=$(check_pith_health)
        echo "  Health:       $HEALTH_RESULT"
        echo ""

        # Database (profile-aware path)
        echo "[Database]"
        DB_PATH=$(resolve_db_path)
        if [[ -f "$DB_PATH" ]]; then
            DB_SIZE=$(du -sh "$DB_PATH" 2>/dev/null | cut -f1)
            echo "  Path:         $DB_PATH"
            echo "  Size:         $DB_SIZE"
            CONCEPTS=$(python3 -c "import sqlite3; c=sqlite3.connect('$DB_PATH'); print(c.execute('SELECT COUNT(*) FROM concepts').fetchone()[0]); c.close()" 2>/dev/null)
            [[ -n "$CONCEPTS" ]] && echo "  Concepts:     $CONCEPTS"
            WAL_MODE=$(python3 -c "import sqlite3; c=sqlite3.connect('$DB_PATH'); print(c.execute('PRAGMA journal_mode').fetchone()[0]); c.close()" 2>/dev/null)
            [[ -n "$WAL_MODE" ]] && echo "  Journal:      $WAL_MODE"
        else
            echo "  Path:         $DB_PATH (not created yet)"
        fi
        echo ""

        # MCP Clients
        echo "[MCP Clients]"
        # Use parallel arrays (bash 3.2 compat — no declare -A on macOS)
        CLIENT_NAMES=("Claude Desktop" "Cursor" "Windsurf" "Cline" "Continue")
        if [[ "$OSTYPE" == "darwin"* ]]; then
            CLIENT_PATHS=(
                "$HOME/Library/Application Support/Claude/claude_desktop_config.json"
                "$HOME/Library/Application Support/Cursor/User/globalStorage/cursor.mcp/config.json"
                "$HOME/Library/Application Support/Windsurf/User/globalStorage/windsurf.mcp/config.json"
                "$HOME/Library/Application Support/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json"
                "$HOME/.continue/config.json"
            )
        else
            CLIENT_PATHS=(
                "$HOME/.config/Claude/claude_desktop_config.json"
                "$HOME/.config/Cursor/User/globalStorage/cursor.mcp/config.json"
                "$HOME/.config/Windsurf/User/globalStorage/windsurf.mcp/config.json"
                "$HOME/.config/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json"
                "$HOME/.continue/config.json"
            )
        fi
        for i in "${!CLIENT_NAMES[@]}"; do
            client="${CLIENT_NAMES[$i]}"
            CFG="${CLIENT_PATHS[$i]}"
            PADDING=$(printf '%*s' $((16 - ${#client})) '')
            if [[ -f "$CFG" ]]; then
                if grep -q '"pith"' "$CFG" 2>/dev/null; then
                    echo "  ${client}:${PADDING}configured"
                else
                    echo "  ${client}:${PADDING}present (pith server not found)"
                fi
            else
                echo "  ${client}:${PADDING}not found"
            fi
        done
        echo ""

        # Auto-start
        echo "[Auto-start]"
        if [[ "$OSTYPE" == "darwin"* ]]; then
            PLIST="$HOME/Library/LaunchAgents/dev.pith.server.plist"
            if [[ -f "$PLIST" ]]; then
                echo "  Server:       LaunchAgent installed"
                LOADED=$(launchctl list 2>/dev/null | grep "dev.pith.server" || true)
                [[ -n "$LOADED" ]] && echo "  Server load:  yes" || echo "  Server load:  no (starts at next login)"
            else
                echo "  Server:       LaunchAgent not installed"
            fi
            BPLIST="$HOME/Library/LaunchAgents/dev.pith.backup.plist"
            if [[ -f "$BPLIST" ]]; then
                echo "  Backup:       LaunchAgent installed (daily 2:00 AM)"
                BLOADED=$(launchctl list 2>/dev/null | grep "dev.pith.backup" || true)
                [[ -n "$BLOADED" ]] && echo "  Backup load:  yes" || echo "  Backup load:  no (starts at next login)"
            else
                echo "  Backup:       LaunchAgent not installed"
            fi
        else
            SVC=$(systemctl --user is-active pith-server.service 2>/dev/null)
            echo "  systemd:      ${SVC:-not installed}"
            TIMER=$(systemctl --user is-active pith-backup.timer 2>/dev/null)
            echo "  backup timer: ${TIMER:-not installed}"
        fi
        echo ""

        # Backups
        echo "[Backups]"
        BACKUP_DIR="$PITH_HOME/backups"
        if [[ -d "$BACKUP_DIR" ]]; then
            BACKUP_COUNT=$(ls "$BACKUP_DIR"/*.db 2>/dev/null | wc -l | xargs)
            echo "  Count:        $BACKUP_COUNT"
            if [[ "$BACKUP_COUNT" -gt 0 ]]; then
                LATEST=$(ls -t "$BACKUP_DIR"/*.db 2>/dev/null | head -1)
                echo "  Latest:       $(stat -f '%Sm' -t '%Y-%m-%dT%H:%M:%S' "$LATEST" 2>/dev/null || stat -c '%y' "$LATEST" 2>/dev/null | cut -d. -f1)"
            fi
        else
            echo "  No backup directory"
        fi
        echo ""

        # API Key (redacted)
        KEY_FILE="$PITH_HOME/config/api.key"
        if [[ -f "$KEY_FILE" ]]; then
            KEY_VAL=$(cat "$KEY_FILE" | tr -d '[:space:]')
            if [[ ${#KEY_VAL} -gt 8 ]]; then
                echo "[API Key]"
                echo "  Status:       Present (${KEY_VAL:0:8}...)"
            fi
        fi
        ;;
    stats)
        # Quick knowledge stats — lightweight alternative to 'pith report'
        DB_PATH=$(resolve_db_path)
        if [[ ! -f "$DB_PATH" ]]; then
            echo "No database found at $DB_PATH"
            echo "Run 'pith start' first to initialize."
            exit 1
        fi

        # Query stats from DB directly (works even if server is stopped)
        python3 -c "
import sqlite3, os
db_path = '$DB_PATH'
conn = sqlite3.connect(db_path)
try:
    concepts = conn.execute('SELECT COUNT(*) FROM concepts WHERE status = \"active\"').fetchone()[0]
    total = conn.execute('SELECT COUNT(*) FROM concepts').fetchone()[0]
    kas = conn.execute('SELECT COUNT(DISTINCT knowledge_area) FROM concepts WHERE status = \"active\"').fetchone()[0]
    associations = conn.execute('SELECT COUNT(*) FROM associations').fetchone()[0]
    sessions = conn.execute('SELECT COUNT(*) FROM sessions').fetchone()[0]
    db_size = os.path.getsize(db_path)
    db_mb = db_size / (1024 * 1024)
    print(f'Pith Stats')
    print(f'══════════════════════════')
    print(f'  Concepts:        {concepts:,} active ({total:,} total)')
    print(f'  Knowledge Areas: {kas}')
    print(f'  Associations:    {associations:,}')
    print(f'  Sessions:        {sessions:,}')
    print(f'  Database:        {db_mb:.1f} MB')
    print(f'  Path:            {db_path}')
except Exception as e:
    print(f'Error reading stats: {e}')
    exit(1)
finally:
    conn.close()
" || { echo "Error: failed to read stats from $DB_PATH"; exit 1; }
        ;;
    protocol)
        # Show Pith cognitive loop instructions for Claude Desktop custom instructions
        SYSPROMPT="$PITH_HOME/SYSTEM_PROMPT.md"
        if [[ ! -f "$SYSPROMPT" ]]; then
            echo "System prompt not found at $SYSPROMPT"
            echo "Re-run the installer to regenerate it."
            exit 1
        fi
        cat "$SYSPROMPT"
        echo ""
        if [[ "$(uname)" == "Darwin" ]] && command -v pbcopy &>/dev/null; then
            pbcopy < "$SYSPROMPT"
            echo "--- Copied to clipboard. Paste into Claude Desktop → Settings → Custom Instructions ---"
        else
            echo "--- Paste the above into Claude Desktop → Settings → Custom Instructions ---"
        fi
        ;;
    *)
        echo "Usage: pith [--profile NAME] {start|stop|restart|status|stats|logs|backup|restore|update|version|report|profiles|maintenance|protocol|uninstall}"
        exit 1
        ;;
esac

deactivate
PITH_CLI_SCRIPT
# === CLI_TEMPLATE_END ===

chmod +x "$PITH_HOME/bin/pith"
# Replace version placeholder (heredoc is single-quoted so vars don't expand)
sed -i.bak "s/__PITH_VERSION__/$PITH_VERSION/g" "$PITH_HOME/bin/pith" && rm -f "$PITH_HOME/bin/pith.bak"
mark_success "Created pith CLI at $PITH_HOME/bin/pith"

# Run health check
echo "Running health check..."
source "$VENV_PATH/bin/activate"

# Pre-check: if something is already running on port 8000 and healthy, skip
# This handles: (1) dev env where Docker pith is on 8000, (2) re-running installer
EXISTING_HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health 2>/dev/null || echo "000")
if [[ "$EXISTING_HEALTH" == "200" ]]; then
    mark_success "Pith server already running on port 8000 — health check passed"
    deactivate
    # Skip to final success message
else
# Start server in background for health check
# NOTE: no 'timeout' on macOS; the Python script self-limits to 30 iterations (1s each)
python3 << 'HEALTH_CHECK_SCRIPT' &
import sys
import time
import subprocess
import os

PITH_HOME = os.environ.get('PITH_HOME', os.path.expanduser('~/.pith'))
PITH_SERVER = os.path.join(PITH_HOME, 'pith-server')

os.chdir(PITH_SERVER)
proc = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'app.server:app', '--host', '127.0.0.1', '--port', '8000'],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

passed = False
try:
    for i in range(30):
        time.sleep(1)
        import urllib.request
        try:
            response = urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)
            if response.status == 200:
                print("Health check passed")
                passed = True
                break
        except Exception:
            pass
    if not passed:
        print("Health check timeout")
finally:
    proc.terminate()
sys.exit(0 if passed else 1)
HEALTH_CHECK_SCRIPT

wait $!
HEALTH_CHECK_EXIT=$?

deactivate

if [[ $HEALTH_CHECK_EXIT -eq 0 ]]; then
    mark_success "Health check passed"
else
    mark_warning "Health check did not complete (may complete on first run)"
fi

echo ""
fi  # end of port pre-check else block

# ============================================================================
# STEP 8b: Auto-configure shell PATH [FIX A1]
# ============================================================================
PATH_ENTRY="export PATH=\"\$HOME/.pith/bin:\$PATH\""
PATH_ADDED=false

# Detect user's shell and choose profile file
SHELL_NAME=$(basename "${SHELL:-/bin/bash}")
case "$SHELL_NAME" in
    zsh)
        SHELL_RC_FILES=("$HOME/.zshrc" "$HOME/.zprofile")
        ;;
    bash)
        SHELL_RC_FILES=("$HOME/.bashrc" "$HOME/.bash_profile")
        ;;
    *)
        SHELL_RC_FILES=("$HOME/.profile")
        ;;
esac

for SHELL_RC in "${SHELL_RC_FILES[@]}"; do
    if [[ -f "$SHELL_RC" ]] || [[ "$SHELL_RC" == "${SHELL_RC_FILES[0]}" ]]; then
        # Idempotency guard: only add if not already present
        if ! grep -q '.pith/bin' "$SHELL_RC" 2>/dev/null; then
            echo "" >> "$SHELL_RC"
            echo "# Pith CLI" >> "$SHELL_RC"
            echo "$PATH_ENTRY" >> "$SHELL_RC"
            mark_success "Added Pith to PATH in $SHELL_RC"
            PATH_ADDED=true
        else
            mark_success "PATH already configured in $SHELL_RC"
            PATH_ADDED=true
        fi
    fi
done

if [[ "$PATH_ADDED" == false ]]; then
    mark_warning "Could not auto-configure PATH. Add manually: $PATH_ENTRY"
fi

echo ""

# ============================================================================
# Final Success Message
# ============================================================================
print_banner

echo -e "${GREEN}✓ Installation Complete!${NC}"
echo ""
echo "Pith is installed at: ${BLUE}$PITH_HOME${NC}"
echo ""
echo -e "${BLUE}Quick Start:${NC}"
if [[ "$PATH_ADDED" == true ]]; then
    echo "  1. Reload your shell: ${YELLOW}source ${SHELL_RC_FILES[0]}${NC} (or open a new terminal)"
else
    echo "  1. Add to PATH: ${YELLOW}export PATH=\"$PITH_HOME/bin:\$PATH\"${NC}"
    echo "     Add this line to your ${YELLOW}~/.bashrc${NC}, ${YELLOW}~/.zshrc${NC}, or equivalent"
fi
echo ""
echo -e "${BLUE}Available Commands:${NC}"
echo "  • ${YELLOW}pith start${NC}   - Start the Pith server"
echo "  • ${YELLOW}pith stop${NC}    - Stop the server"
echo "  • ${YELLOW}pith status${NC}  - Check server status"
echo "  • ${YELLOW}pith logs${NC}    - View server logs"
echo "  • ${YELLOW}pith backup${NC}  - Create manual backup (WAL-safe)"
echo "  • ${YELLOW}pith restore${NC} - Restore from backup"
echo "  • ${YELLOW}pith update${NC}  - Update Pith"
echo "  • ${YELLOW}pith version${NC} - Show version and system info"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "  1. Open a new terminal (or reload your shell profile)"
echo "  2. Start the server: ${YELLOW}pith start${NC}"
echo "  3. Add cognitive instructions to Claude Desktop:"
echo "     ${YELLOW}pith protocol${NC}  (copies to clipboard — paste into Custom Instructions)"
echo "  4. View logs: ${YELLOW}pith logs${NC}"
echo "  5. Check status: ${YELLOW}pith status${NC}"
echo ""
echo -e "${BLUE}Documentation:${NC}"
echo "  https://docs.pith.dev"
echo "  https://github.com/esteyangandrew/pith-core"
echo ""

exit 0
