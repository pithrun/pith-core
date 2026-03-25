#!/bin/bash
#
# Pith — MCP Client Configuration
#
# Lightweight script to (re)configure MCP clients for an existing
# Pith installation. Delegates to configure_clients.py.
#
# Usage: bash scripts/setup-mcp.sh
#
set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Pith MCP Client Setup${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Resolve paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PITH_HOME="${PITH_HOME:-$HOME/.pith}"

# Detect installed vs distribution context
PITH_HOME_DIR="${PITH_HOME:-$HOME/.pith}"
if [[ -f "$PITH_HOME_DIR/pith-server/pith_mcp.py" ]]; then
    SERVER_PATH="$PITH_HOME_DIR/pith-server/pith_mcp.py"
    CONFIGURE_SCRIPT="$PITH_HOME_DIR/pith-server/scripts/configure_clients.py"
    API_KEY_FILE="$PITH_HOME_DIR/config/api.key"
    PYTHON_CMD="$PITH_HOME_DIR/.venv/bin/python3"
elif [[ -f "$PROJECT_DIR/pith_mcp.py" ]]; then
    SERVER_PATH="$PROJECT_DIR/pith_mcp.py"
    CONFIGURE_SCRIPT="$PROJECT_DIR/scripts/configure_clients.py"
    API_KEY_FILE="$PROJECT_DIR/.env"
    PYTHON_CMD="python3"
else
    echo -e "${RED}✗ Pith MCP bridge not found${NC}"
    echo "  Run install.sh first, or run this from the distribution directory."
    exit 1
fi

# Check Python 3 (replaces Node.js check — Python is the only runtime now)
echo "Checking prerequisites..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found${NC}"
    echo "Please install Python 3.9+: https://python.org/"
    exit 1
fi
echo -e "${GREEN}✓ Python $(python3 --version)${NC}"

# Read API key
API_KEY=""
if [[ -f "$API_KEY_FILE" ]] && [[ "$API_KEY_FILE" == *.key ]]; then
    API_KEY=$(cat "$API_KEY_FILE" 2>/dev/null)
elif [[ -f "$API_KEY_FILE" ]]; then
    API_KEY=$(grep "PITH_API_KEY=" "$API_KEY_FILE" 2>/dev/null | cut -d'=' -f2)
fi
if [[ -z "$API_KEY" ]]; then
    echo -e "${YELLOW}⚠ No API key found — generate one during install${NC}"
    API_KEY="dev-api-key-change-in-production"
fi

# Detect platform
PLATFORM="macos"
if [[ "$(uname -s)" == "Linux" ]]; then
    PLATFORM="linux"
fi

# Use configure_clients.py if available
if [[ -f "$CONFIGURE_SCRIPT" ]]; then
    echo ""
    echo "Configuring MCP clients..."
    python3 "$CONFIGURE_SCRIPT" \
        --server-path "$SERVER_PATH" \
        --python-cmd "$PYTHON_CMD" \
        --api-key "$API_KEY" \
        --project-dir "$(dirname "$SERVER_PATH")" \
        --platform "$PLATFORM"
    echo ""
    echo -e "${GREEN}✓ MCP clients configured (Python MCP bridge)${NC}"
    echo "  Note: Using Python MCP bridge (pith_mcp.py). Node.js is no longer required."
else
    echo -e "${YELLOW}⚠ configure_clients.py not found, configuring Claude Desktop only${NC}"
    # Inline fallback for Claude Desktop
    if [[ "$(uname)" == "Darwin" ]]; then
        CONFIG_DIR="$HOME/Library/Application Support/Claude"
    else
        CONFIG_DIR="$HOME/.config/Claude"
    fi
    CONFIG_FILE="$CONFIG_DIR/claude_desktop_config.json"
    mkdir -p "$CONFIG_DIR"

    python3 -c "
import json, os
config_path = '$CONFIG_FILE'
config = {}
if os.path.exists(config_path):
    with open(config_path) as f:
        config = json.load(f)
config.setdefault('mcpServers', {})
for legacy in ['pith-mcp', 'pith', 'pith-mcp-wrapper']:
    config['mcpServers'].pop(legacy, None)
config['mcpServers']['pith'] = {
    'command': '$PYTHON_CMD',
    'args': ['$SERVER_PATH'],
    'env': {'PITH_API_KEY': '$API_KEY', 'PITH_API_URL': 'http://localhost:8000'}
}
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
print('Configured pith (Python MCP bridge) in Claude Desktop')
"
fi

echo ""
echo -e "${BLUE}================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${BLUE}================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Restart Claude Desktop completely (Cmd+Q, then reopen)"
echo "  2. Look for the hammer icon (MCP tools available)"
echo "  3. Test with: 'Can you check Pith stats?'"
echo ""
echo "Windows users:"
echo "  Run: powershell scripts/install.ps1"
echo "  Config: %APPDATA%\Claude\claude_desktop_config.json"
echo ""
