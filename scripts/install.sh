#!/bin/bash
set -euo pipefail

# Pith Installer v1.0.0
# macOS developer preview installer; Linux remains an unverified source/developer path.

# Configuration
DOWNLOAD_URL="${DOWNLOAD_URL:-https://github.com/pithrun/pith-core/releases/latest/download}"
CHECKSUM_URL="${CHECKSUM_URL:-https://github.com/pithrun/pith-core/releases/latest/download}"
PITH_LOCAL_ONLY_INSTALL="${PITH_LOCAL_ONLY_INSTALL:-0}"
PITH_PYTHON="${PITH_PYTHON:-}"
PITH_AUTO_PYTHON="${PITH_AUTO_PYTHON:-0}"
PITH_NO_AUTO_PYTHON="${PITH_NO_AUTO_PYTHON:-0}"
PITH_REPAIR_RUNTIME="${PITH_REPAIR_RUNTIME:-0}"
PITH_FORCE_MANAGED_PYTHON="${PITH_FORCE_MANAGED_PYTHON:-0}"
# Keep PITH_VERSION on line 18.
# scripts/version-bump.sh and TEST-090 depend on this exact location.
PITH_VERSION="1.0.2"
detect_account_home() {
    if command -v dscl >/dev/null 2>&1; then
        dscl . -read "/Users/$(id -un)" NFSHomeDirectory 2>/dev/null | awk '{print $2; exit}' && return
    fi
    if command -v getent >/dev/null 2>&1; then
        getent passwd "$(id -un)" 2>/dev/null | cut -d: -f6 && return
    fi
    echo "$HOME"
}
PITH_ACCOUNT_HOME="${PITH_ACCOUNT_HOME:-$(detect_account_home)}"
if [[ -z "$PITH_ACCOUNT_HOME" ]]; then
    PITH_ACCOUNT_HOME="$HOME"
fi
PITH_CANONICAL_HOME="$PITH_ACCOUNT_HOME/.pith"
PITH_DEFAULT_HOME="$HOME/.pith"
PITH_HOME="${PITH_HOME:-$PITH_DEFAULT_HOME}"
PITH_SKIP_GLOBAL_CLI_LINK="${PITH_SKIP_GLOBAL_CLI_LINK:-0}"
PITH_FORCE_GLOBAL_CLI_LINK="${PITH_FORCE_GLOBAL_CLI_LINK:-0}"
STEP_COUNT=9
PITH_DEFAULT_PORT="${PITH_DEFAULT_PORT:-8000}"
PITH_PORT_SCAN_MAX="${PITH_PORT_SCAN_MAX:-8020}"
PITH_PORT="${PITH_PORT:-}"
CURRENT_STEP=0

# Color codes. Disable colors when stdout is not a terminal so install.command
# logs and tee output never show raw escape sequences.
if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    NC=''
fi

PITH_RUNTIME_ID="${PITH_RUNTIME_ID:-cpython-3.12.13+20260504-aarch64-apple-darwin-install_only_stripped}"
PITH_RUNTIME_VERSION="${PITH_RUNTIME_VERSION:-3.12.13}"
PITH_RUNTIME_PLATFORM="${PITH_RUNTIME_PLATFORM:-macos}"
PITH_RUNTIME_ARCH="${PITH_RUNTIME_ARCH:-arm64}"
PITH_RUNTIME_SOURCE="${PITH_RUNTIME_SOURCE:-astral-sh/python-build-standalone}"
PITH_RUNTIME_LICENSE="${PITH_RUNTIME_LICENSE:-CPython distribution from astral-sh/python-build-standalone; preserve upstream runtime notices}"
PITH_RUNTIME_URL="${PITH_RUNTIME_URL:-https://github.com/astral-sh/python-build-standalone/releases/download/20260504/cpython-3.12.13%2B20260504-aarch64-apple-darwin-install_only_stripped.tar.gz}"
PITH_RUNTIME_SHA256="${PITH_RUNTIME_SHA256:-dbba2cb07d0c5c1e641aefefe78c5706ff7a01e2c4d1de18e8447522af37431e}"
PITH_RUNTIME_SIZE_BYTES="${PITH_RUNTIME_SIZE_BYTES:-24943612}"
PITH_RUNTIME_ROOT="$PITH_HOME/runtime/python"
PITH_RUNTIME_META="$PITH_HOME/config/python-runtime.json"

# FIX S1: SHA-256 checksum variables
PITH_SERVER_FILENAME="pith-server-latest.tar.gz"
PITH_CHECKSUM_FILENAME="pith-server-latest.sha256"

download_release_file() {
    local url="$1"
    local output_path="$2"

    if [[ -n "${PITH_DOWNLOAD_BEARER_TOKEN:-}" ]]; then
        printf 'header = "Authorization: Bearer %s"\n' "$PITH_DOWNLOAD_BEARER_TOKEN" | \
            curl -fsSL --max-time 30 --config - "$url" -o "$output_path" 2>/dev/null
    else
        curl -fsSL --max-time 30 "$url" -o "$output_path" 2>/dev/null
    fi
}

# Print banner
print_banner() {
    clear 2>/dev/null || true
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════╗"
    echo "║   🧠 Pith Installer v${PITH_VERSION}       ║"
    echo "║      macOS Developer Preview           ║"
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

mark_error() {
    echo -e "${RED}✗${NC} $1" >&2
}

# Error handler
error_exit() {
    echo -e "${RED}✗ ERROR:${NC} $1" >&2
    exit 1
}

validate_port_value() {
    local value="$1"
    local label="${2:-PITH_PORT}"
    if [[ ! "$value" =~ ^[0-9]+$ ]] || (( value < 1 || value > 65535 )); then
        error_exit "$label must be an integer from 1 to 65535, got '$value'."
    fi
}

port_in_use_value() {
    local port="$1"
    PITH_CHECK_PORT="$port" python3 - <<'PY'
import os
import socket
import sys

port = int(os.environ["PITH_CHECK_PORT"])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(("127.0.0.1", port))
except OSError:
    sys.exit(0)
finally:
    s.close()
sys.exit(1)
PY
}

pith_service_on_port() {
    local port="$1"
    local response
    response=$(curl -s --max-time 3 "http://127.0.0.1:${port}/health" 2>/dev/null || true)
    [[ -n "$response" ]] || return 1
    echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('service')=='pith'" 2>/dev/null
}

load_existing_pith_port() {
    if [[ -z "${PITH_PORT:-}" && -f "$PITH_HOME/.env" ]]; then
        local existing_port
        existing_port=$(grep '^PITH_PORT=' "$PITH_HOME/.env" 2>/dev/null | tail -1 | cut -d= -f2- | sed 's/[[:space:]"]//g' || true)
        if [[ -n "$existing_port" ]]; then
            PITH_PORT="$existing_port"
        fi
    fi
}

select_pith_port() {
    validate_port_value "$PITH_DEFAULT_PORT" "PITH_DEFAULT_PORT"
    validate_port_value "$PITH_PORT_SCAN_MAX" "PITH_PORT_SCAN_MAX"
    if (( PITH_PORT_SCAN_MAX < PITH_DEFAULT_PORT )); then
        error_exit "PITH_PORT_SCAN_MAX must be greater than or equal to PITH_DEFAULT_PORT."
    fi

    if [[ -n "${PITH_PORT:-}" ]]; then
        validate_port_value "$PITH_PORT" "PITH_PORT"
        if port_in_use_value "$PITH_PORT"; then
            error_exit "PITH_PORT=$PITH_PORT is already in use. Stop the existing process or choose another PITH_PORT."
        fi
        export PITH_PORT
        mark_success "Using requested Pith port $PITH_PORT"
        return
    fi

    if ! port_in_use_value "$PITH_DEFAULT_PORT"; then
        PITH_PORT="$PITH_DEFAULT_PORT"
        export PITH_PORT
        mark_success "Using default Pith port $PITH_PORT"
        return
    fi

    if pith_service_on_port "$PITH_DEFAULT_PORT"; then
        error_exit "Pith is already running on port $PITH_DEFAULT_PORT. Stop the existing Pith service before rerunning install."
    fi

    local candidate
    for (( candidate=PITH_DEFAULT_PORT + 1; candidate<=PITH_PORT_SCAN_MAX; candidate++ )); do
        if ! port_in_use_value "$candidate"; then
            PITH_PORT="$candidate"
            export PITH_PORT
            mark_warning "Port $PITH_DEFAULT_PORT is in use; using alternate Pith port $PITH_PORT."
            return
        fi
    done

    error_exit "No free Pith port found in ${PITH_DEFAULT_PORT}-${PITH_PORT_SCAN_MAX}. Set PITH_PORT to an available port and rerun install."
}

upsert_pith_env() {
    local key="$1"
    local value="$2"
    local env_file="$PITH_HOME/.env"
    mkdir -p "$PITH_HOME"
    touch "$env_file"
    if grep -q "^${key}=" "$env_file"; then
        sed -i.bak "s|^${key}=.*|${key}=${value}|" "$env_file" && rm -f "$env_file.bak"
    else
        echo "${key}=${value}" >> "$env_file"
    fi
    chmod 600 "$env_file"
}

ensure_pith_env_value() {
    local key="$1"
    local value="$2"
    local env_file="$PITH_HOME/.env"
    mkdir -p "$PITH_HOME"
    touch "$env_file"
    if ! grep -q "^${key}=" "$env_file"; then
        echo "${key}=${value}" >> "$env_file"
    fi
    chmod 600 "$env_file"
}

persist_pith_port_config() {
    upsert_pith_env "PITH_PORT" "$PITH_PORT"
    mark_success "Configured Pith API port: $PITH_PORT"
}

persist_preview_usage_config() {
    upsert_pith_env "PITH_USAGE_LIMITS_ENABLED" "false"
    upsert_pith_env "PITH_DEV_MODE" "true"
    upsert_pith_env "PITH_TIER" "dev"
    mark_success "Configured free developer preview with no active local Pith usage caps"
}

private_beta_pause() {
    local prompt="${1:-Private beta setup paused. Press Return to continue...}"
    if [[ "${PITH_PRIVATE_BETA:-0}" == "1" && "${PITH_SKIP_PAUSES:-0}" != "1" && -t 0 ]]; then
        echo ""
        read -r -p "$prompt" _ || true
        echo ""
    fi
}

migrate_legacy_env_aliases() {
    local env_file="$PITH_HOME/.env"
    [[ -f "$env_file" ]] || return 0
    "$PYTHON_EXECUTABLE" - "$env_file" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

env_file = Path(sys.argv[1])
alias_map = {
    "BRAIN_API_KEY": "PITH_API_KEY",
    "BRAIN_API_URL": "PITH_API_URL",
    "BRAIN_DATA_DIR": "PITH_DATA_DIR",
}
lines = env_file.read_text(encoding="utf-8").splitlines()
seen = set()
legacy_values = {}
for line in lines:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        continue
    key, value = stripped.split("=", 1)
    key = key.strip()
    if key in alias_map and alias_map[key] not in legacy_values:
        legacy_values[alias_map[key]] = value
    else:
        seen.add(key)

output = []
changed = False
for line in lines:
    stripped = line.strip()
    if stripped and not stripped.startswith("#") and "=" in stripped:
        key = stripped.split("=", 1)[0].strip()
        if key in alias_map:
            changed = True
            continue
    output.append(line)

for canonical, value in legacy_values.items():
    if canonical not in seen:
        output.append(f"{canonical}={value}")
        changed = True

if changed:
    env_file.write_text("\n".join(output).rstrip() + "\n", encoding="utf-8")
PY
}

surface_label() {
    case "$1" in
        claude_desktop) echo "Claude Desktop" ;;
        codex) echo "Codex" ;;
        vscode) echo "VS Code" ;;
        claude_code) echo "Claude Code / Claude CLI" ;;
        cursor) echo "Cursor" ;;
        windsurf) echo "Windsurf" ;;
        cline) echo "Cline" ;;
        project) echo "Project MCP templates" ;;
        *) echo "$1" ;;
    esac
}

surface_detail() {
    case "$1" in
        claude_desktop) echo "MCP config plus manual Instructions for Claude step" ;;
        codex) echo "HTTP/API lifecycle instructions in ~/.codex/AGENTS.md plus optional MCP config for tool access" ;;
        vscode) echo "MCP config plus Copilot Agent Chat instruction file" ;;
        claude_code) echo "MCP config plus Pith lifecycle instructions/hooks where supported by the installed Claude Code version" ;;
        cursor) echo "MCP config template; add Cursor User Rule or AGENTS.md for default Pith invocation" ;;
        windsurf) echo "Experimental MCP config template; not launch-verified" ;;
        cline) echo "Experimental MCP settings template; not launch-verified" ;;
        project) echo ".mcp.json and .vscode/mcp.json templates inside the installed server folder" ;;
        *) echo "MCP config" ;;
    esac
}

surface_detected() {
    case "$1" in
        claude_desktop)
            [[ -d "$HOME/Library/Application Support/Claude" || -d "$HOME/.config/Claude" ]]
            ;;
        codex)
            [[ -d "$HOME/.codex" ]]
            ;;
        vscode)
            [[ -d "$HOME/.vscode" || -d "$HOME/Library/Application Support/Code" || -d "/Applications/Visual Studio Code.app" || -d "$HOME/Applications/Visual Studio Code.app" ]]
            ;;
        claude_code)
            [[ -d "$HOME/.claude" || -f "$HOME/.claude.json" ]]
            ;;
        cursor)
            [[ -d "$HOME/.cursor" ]]
            ;;
        windsurf)
            [[ -d "$HOME/.codeium/windsurf" ]]
            ;;
        cline)
            [[ -d "$HOME/Library/Application Support/Code/User/globalStorage/saoudrizwan.claude-dev" || -d "$HOME/.config/Code/User/globalStorage/saoudrizwan.claude-dev" ]]
            ;;
        project)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

normalize_surface_list() {
    printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]'
}

surface_selected() {
    local surface="$1"
    local selected="${PITH_SELECTED_SURFACES:-all}"
    [[ "$selected" == "all" ]] && return 0
    [[ "$selected" == "none" ]] && return 1
    [[ ",$selected," == *",$surface,"* ]]
}

surface_selected_and_detected() {
    local surface="$1"
    surface_selected "$surface" && surface_detected "$surface"
}

select_install_surfaces() {
    if [[ -n "${PITH_SELECTED_SURFACES+x}" ]]; then
        PITH_SELECTED_SURFACES="$(normalize_surface_list "${PITH_SELECTED_SURFACES:-none}")"
        if [[ -z "$PITH_SELECTED_SURFACES" ]]; then
            PITH_SELECTED_SURFACES="none"
        fi
        if [[ "$PITH_SELECTED_SURFACES" == "none" ]]; then
            mark_warning "Skipping AI app surface configuration. You can rerun the installer later."
        fi
        return
    fi

    PITH_SELECTED_SURFACES="$(normalize_surface_list "${PITH_CLIENTS:-all}")"
    local surfaces=(claude_desktop codex vscode claude_code cursor windsurf cline project)

    if [[ -n "${PITH_CLIENTS:-}" ]]; then
        return
    fi

    if [[ "${PITH_PRIVATE_BETA:-0}" != "1" || "${PITH_SKIP_PAUSES:-0}" == "1" || ! -t 0 ]]; then
        PITH_SELECTED_SURFACES="claude_desktop,claude_code,codex,vscode,cursor,project"
        return
    fi

    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  Choose where Pith should be installed${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "  Pith always installs the local service and CLI."
    echo "  These choices only decide which AI app configuration files Pith writes now."
    echo "  Choose which AI app surfaces should receive Pith configuration:"
    echo ""

    local default_selection=()
    local i=1
    local surface status
    for surface in "${surfaces[@]}"; do
        if surface_detected "$surface"; then
            status="detected"
            default_selection+=("$surface")
        else
            status="not detected"
        fi
        printf "  %d. %-27s [%s] — %s\n" "$i" "$(surface_label "$surface")" "$status" "$(surface_detail "$surface")"
        i=$((i + 1))
    done

    local default_csv="none"
    if [[ ${#default_selection[@]} -gt 0 ]]; then
        default_csv="$(IFS=,; echo "${default_selection[*]}")"
    fi

    echo ""
    echo "  Press Return for all detected surfaces."
    echo "  Type numbers like 1,2,3 to choose exactly, 'all' for every detected surface, or 'none' to skip app configuration."
    echo ""

    local answer=""
    read -r -p "Install Pith into which surfaces? [all detected] " answer || answer=""
    answer="$(normalize_surface_list "$answer")"

    if [[ -z "$answer" || "$answer" == "all" ]]; then
        PITH_SELECTED_SURFACES="$default_csv"
    elif [[ "$answer" == "none" ]]; then
        PITH_SELECTED_SURFACES="none"
    else
        local selected=()
        local token
        IFS=',' read -ra tokens <<< "$answer"
        for token in "${tokens[@]}"; do
            if [[ "$token" =~ ^[0-9]+$ ]] && (( token >= 1 && token <= ${#surfaces[@]} )); then
                selected+=("${surfaces[$((token - 1))]}")
            fi
        done
        if [[ ${#selected[@]} -eq 0 ]]; then
            mark_warning "No valid surface choices recognized; using all detected surfaces."
            PITH_SELECTED_SURFACES="$default_csv"
        else
            PITH_SELECTED_SURFACES="$(IFS=,; echo "${selected[*]}")"
        fi
    fi

    echo ""
    if [[ "$PITH_SELECTED_SURFACES" == "none" ]]; then
        mark_warning "Skipping AI app surface configuration. You can rerun the installer later."
    else
        echo "Selected surfaces:"
        IFS=',' read -ra selected_surfaces <<< "$PITH_SELECTED_SURFACES"
        for surface in "${selected_surfaces[@]}"; do
            echo "  • $(surface_label "$surface")"
        done
    fi
    private_beta_pause "Private beta pause: review selected surfaces, then press Return to continue..."
}

json_escape() {
    printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

runtime_timestamp() {
    date -u '+%Y-%m-%dT%H:%M:%SZ'
}

python_version_string() {
    local exe="$1"
    "$exe" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))' 2>/dev/null
}

is_compatible_python() {
    local exe="$1"
    [[ -x "$exe" ]] || return 1
    "$exe" -c 'import sys; raise SystemExit(0 if (sys.version_info >= (3, 10) and sys.version_info < (3, 13)) else 1)' 2>/dev/null
}

write_python_runtime_metadata() {
    local managed_by="$1"
    local exe="$2"
    local version="$3"
    mkdir -p "$PITH_HOME/config"
    cat > "$PITH_RUNTIME_META" <<EOF
{
  "managed_by": "$(json_escape "$managed_by")",
  "runtime_id": "$(json_escape "$PITH_RUNTIME_ID")",
  "python_version": "$(json_escape "$version")",
  "platform": "$(json_escape "$PITH_RUNTIME_PLATFORM")",
  "arch": "$(json_escape "$(uname -m)")",
  "source": "$(json_escape "$PITH_RUNTIME_SOURCE")",
  "source_url": "$(json_escape "$PITH_RUNTIME_URL")",
  "sha256": "$(json_escape "$PITH_RUNTIME_SHA256")",
  "license": "$(json_escape "$PITH_RUNTIME_LICENSE")",
  "installed_at": "$(runtime_timestamp)",
  "python_executable": "$(json_escape "$exe")"
}
EOF
}

runtime_metadata_managed_by_pith() {
    [[ -f "$PITH_RUNTIME_META" ]] && grep -q '"managed_by"[[:space:]]*:[[:space:]]*"pith"' "$PITH_RUNTIME_META"
}

remove_managed_python_runtime() {
    if runtime_metadata_managed_by_pith; then
        rm -rf "$PITH_RUNTIME_ROOT" "$PITH_RUNTIME_META"
        mark_warning "Removed corrupt Pith-managed Python runtime"
    fi
}

configure_codex_agents_instructions() {
    python3 - <<'PY'
from pathlib import Path
import shutil
import time

home = Path.home()
codex_dir = home / ".codex"
if not codex_dir.is_dir():
    raise SystemExit(0)

path = codex_dir / "AGENTS.md"
start = "<!-- PITH COGNITIVE LOOP: START -->"
end = "<!-- PITH COGNITIVE LOOP: END -->"
body = """# Pith Cognitive Loop

Pith is installed locally. For Codex, use the local HTTP/API command as the primary cognitive lifecycle path because Codex MCP stdio transport can restart or close between turns.

On every substantive user message, run `conversation_turn` before composing the response. Include `"surface_id": "codex_local_api"` and a stable `origin_id` for this Codex thread/workspace. After the first successful call returns `resolved_session_id`, include that value as `session_id` on later lifecycle calls when available. Also include `previous_message`, `previous_response`, and `extracted_concepts_json` after the first exchange. Send JSON on stdin and parse the last non-empty output line as JSON because the wrapper may print a profile banner first:

```bash
~/.pith/bin/pith api conversation_turn --stdin-json
```

For checkpoints and closeout, use the matching lifecycle operations:

```bash
~/.pith/bin/pith api checkpoint --stdin-json
~/.pith/bin/pith api session_end --stdin-json
```

For lifecycle evidence reports, use `~/.pith/bin/pith api lifecycle_status --stdin-json` with the relevant `surface_id`, `session_id`, `origin_id`, or `workspace_id`. For cross-surface source coverage evidence, use `~/.pith/bin/pith api surface_activity --stdin-json` with `requested_surfaces` such as `"claude_code,codex_local_api,local_api_cli"` and `include_codex_local=true`. Unsupported or sparse surfaces must report that state rather than inferring success from instructions or memory.

`pith api-fallback ...` remains as a legacy/recovery alias. Pith MCP tools with the `pith_` prefix may also be available in Codex and are useful for richer tool access when the MCP transport is healthy. Do not depend on MCP-only access for the core cognitive lifecycle.

For trivial exchanges, use `[]` for `extracted_concepts_json`. For substantive implementation or deployment work, extracted concepts must include concrete `verified: <check>` evidence.
"""
block = f"{start}\n{body}\n{end}\n"

existing = path.read_text(encoding="utf-8") if path.exists() else ""
if start in existing and end in existing:
    before, rest = existing.split(start, 1)
    _, after = rest.split(end, 1)
    new_text = before.rstrip() + "\n\n" + block + after.lstrip()
else:
    new_text = (existing.rstrip() + "\n\n" if existing.strip() else "") + block

if new_text != existing:
    if path.exists():
        backup_path = path.with_name(path.name + f".backup.{int(time.time())}")
        shutil.copy2(path, backup_path)
    path.write_text(new_text, encoding="utf-8")
    print(f"Codex AGENTS instructions configured: {path}")
PY
}

find_existing_python() {
    local candidates=()

    if [[ -n "$PITH_PYTHON" ]]; then
        if is_compatible_python "$PITH_PYTHON"; then
            PYTHON_EXECUTABLE="$PITH_PYTHON"
            PITH_SELECTED_PYTHON_SOURCE="external"
            return 0
        fi
        error_exit "PITH_PYTHON is set but is not Python >=3.10,<3.13: $PITH_PYTHON"
    fi

    if [[ "$PITH_FORCE_MANAGED_PYTHON" != "1" ]]; then
        candidates+=(
            "$PITH_RUNTIME_ROOT/bin/python3"
            "$(command -v python3.12 2>/dev/null || true)"
            "$(command -v python3.11 2>/dev/null || true)"
            "$(command -v python3.10 2>/dev/null || true)"
            "$(command -v python3 2>/dev/null || true)"
            "$(command -v python 2>/dev/null || true)"
            "/opt/homebrew/bin/python3.12"
            "/opt/homebrew/bin/python3.11"
            "/opt/homebrew/bin/python3.10"
            "/usr/local/bin/python3.12"
            "/usr/local/bin/python3.11"
            "/usr/local/bin/python3.10"
            "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
            "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3"
            "/Library/Frameworks/Python.framework/Versions/3.10/bin/python3"
        )

        for candidate in "${candidates[@]}"; do
            [[ -n "$candidate" ]] || continue
            if is_compatible_python "$candidate"; then
                PYTHON_EXECUTABLE="$candidate"
                if [[ "$candidate" == "$PITH_RUNTIME_ROOT/bin/python3" ]]; then
                    PITH_SELECTED_PYTHON_SOURCE="pith-managed"
                else
                    PITH_SELECTED_PYTHON_SOURCE="external"
                fi
                return 0
            fi
        done
    fi

    if [[ -e "$PITH_RUNTIME_ROOT/bin/python3" ]]; then
        remove_managed_python_runtime
    fi
    return 1
}

verify_runtime_checksum() {
    local archive="$1"
    local actual=""
    if command -v shasum >/dev/null 2>&1; then
        actual="$(shasum -a 256 "$archive" | awk '{print $1}')"
    elif command -v sha256sum >/dev/null 2>&1; then
        actual="$(sha256sum "$archive" | awk '{print $1}')"
    else
        error_exit "No SHA-256 tool found. Install shasum or sha256sum."
    fi
    [[ "$actual" == "$PITH_RUNTIME_SHA256" ]] || error_exit "Python runtime checksum mismatch. Expected $PITH_RUNTIME_SHA256, got $actual"
}

validate_runtime_archive_layout() {
    local archive="$1"
    local bad_member=""
    bad_member="$(tar -tzf "$archive" | awk '$0 !~ /^python\// { print; exit }')"
    [[ -z "$bad_member" ]] || error_exit "Python runtime archive contains unexpected path: $bad_member"
    tar -tzf "$archive" | grep -qx 'python/bin/python3' || error_exit "Python runtime archive missing python/bin/python3"
}

provision_pith_python_runtime() {
    [[ "$OS_TYPE" == "macos" && "$(uname -m)" == "$PITH_RUNTIME_ARCH" ]] || {
        error_exit "Automatic Python provisioning is currently supported only on macOS arm64. Install Python 3.10-3.12 manually or set PITH_NO_AUTO_PYTHON=1."
    }

    local tmpdir archive extract_dir runtime_tmp runtime_exe runtime_version
    tmpdir="$(mktemp -d)"
    archive="$tmpdir/python-runtime.tar.gz"
    extract_dir="$tmpdir/extract"
    runtime_tmp="$PITH_RUNTIME_ROOT.tmp"

    echo "Downloading Pith-managed Python $PITH_RUNTIME_VERSION ($((PITH_RUNTIME_SIZE_BYTES / 1024 / 1024)) MB)..."
    if ! curl -L --fail --connect-timeout 20 --retry 2 --output "$archive" "$PITH_RUNTIME_URL"; then
        rm -rf "$tmpdir"
        error_exit "Could not download Pith-managed Python runtime from pinned URL."
    fi

    verify_runtime_checksum "$archive"
    validate_runtime_archive_layout "$archive"

    mkdir -p "$extract_dir" "$(dirname "$PITH_RUNTIME_ROOT")"
    tar -xzf "$archive" -C "$extract_dir"
    runtime_exe="$extract_dir/python/bin/python3"
    [[ -x "$runtime_exe" ]] || { rm -rf "$tmpdir"; error_exit "Extracted Python runtime is missing executable python/bin/python3"; }
    is_compatible_python "$runtime_exe" || { rm -rf "$tmpdir"; error_exit "Extracted Python runtime is not compatible"; }

    rm -rf "$runtime_tmp"
    mv "$extract_dir/python" "$runtime_tmp"
    rm -rf "$PITH_RUNTIME_ROOT"
    mv "$runtime_tmp" "$PITH_RUNTIME_ROOT"
    rm -rf "$tmpdir"

    PYTHON_EXECUTABLE="$PITH_RUNTIME_ROOT/bin/python3"
    PITH_SELECTED_PYTHON_SOURCE="pith-managed"
    runtime_version="$(python_version_string "$PYTHON_EXECUTABLE")"
    write_python_runtime_metadata "pith" "$PYTHON_EXECUTABLE" "$runtime_version"
    mark_success "Installed Pith-managed Python $runtime_version at $PITH_RUNTIME_ROOT"
}

ensure_python_runtime() {
    if [[ "$PITH_AUTO_PYTHON" == "1" && "$PITH_NO_AUTO_PYTHON" == "1" ]]; then
        error_exit "PITH_AUTO_PYTHON=1 and PITH_NO_AUTO_PYTHON=1 cannot both be set."
    fi

    if find_existing_python; then
        local version
        version="$(python_version_string "$PYTHON_EXECUTABLE")"
        write_python_runtime_metadata "${PITH_SELECTED_PYTHON_SOURCE:-external}" "$PYTHON_EXECUTABLE" "$version"
        mark_success "Python: $version ($PITH_SELECTED_PYTHON_SOURCE: $PYTHON_EXECUTABLE)"
        return 0
    fi

    if [[ "$PITH_NO_AUTO_PYTHON" == "1" ]]; then
        error_exit "Python 3.10-3.12 required. Install Python from python.org/Homebrew, or unset PITH_NO_AUTO_PYTHON to allow a Pith-managed runtime."
    fi

    if [[ "$PITH_AUTO_PYTHON" != "1" ]]; then
        if [[ -t 0 ]]; then
            echo "No compatible Python 3.10-3.12 was found."
            echo "Pith can install a managed Python $PITH_RUNTIME_VERSION under $PITH_RUNTIME_ROOT."
            echo "Source: $PITH_RUNTIME_SOURCE"
            echo "Download: $((PITH_RUNTIME_SIZE_BYTES / 1024 / 1024)) MB"
            echo "Uninstall removes this runtime with $PITH_HOME."
            read -r -p "Install Pith-managed Python now? [y/N] " REPLY
            case "$REPLY" in
                y|Y|yes|YES) ;;
                *) error_exit "Python 3.10-3.12 required. Install Python manually or rerun with PITH_AUTO_PYTHON=1." ;;
            esac
        else
            error_exit "Python 3.10-3.12 required. Noninteractive install needs PITH_AUTO_PYTHON=1 to provision a Pith-managed runtime."
        fi
    fi

    provision_pith_python_runtime
}

safe_rm_pith_server_path() {
    local target="$1"
    local expected="$PITH_HOME/pith-server"

    if [[ -z "$target" || "$target" == "/" || "$target" == "$HOME" || "$target" == "$PITH_HOME" ]]; then
        error_exit "Refusing to remove unsafe server path: ${target:-<empty>}"
    fi
    if [[ "$target" != "$expected" ]]; then
        error_exit "Refusing to remove server path outside PITH_HOME: $target"
    fi
    if [[ -L "$target" ]]; then
        error_exit "Refusing to replace symlinked server path: $target"
    fi
    if [[ -e "$target" ]]; then
        chmod -R u+w "$target" 2>/dev/null || true
        rm -rf "$target"
    fi
}

normalize_server_tree_modes() {
    local server_tree="$1"
    find "$server_tree" -type d -exec chmod u+rwx {} + 2>/dev/null || true
    find "$server_tree" -type f -exec chmod u+rw {} + 2>/dev/null || true
    if [[ -d "$server_tree/scripts" ]]; then
        find "$server_tree/scripts" -type f -name "*.sh" -exec chmod u+x {} + 2>/dev/null || true
    fi
}

validate_staged_server_tree() {
    local stage="$1"
    local missing=()

    [[ -d "$stage/app" ]] || missing+=("app/")
    [[ -d "$stage/pith_client" ]] || missing+=("pith_client/")
    [[ -f "$stage/pith_mcp.py" ]] || missing+=("pith_mcp.py")
    [[ -f "$stage/requirements.txt" ]] || missing+=("requirements.txt")
    [[ -f "$stage/scripts/install.sh" ]] || missing+=("scripts/install.sh")

    if [[ ${#missing[@]} -gt 0 ]]; then
        mark_error "Staged Pith server tree is incomplete: ${missing[*]}"
        return 1
    fi
}

activate_staged_server_tree() {
    local stage="$1"
    local target="$2"

    validate_staged_server_tree "$stage" || return 1
    normalize_server_tree_modes "$stage"
    safe_rm_pith_server_path "$target"
    mv "$stage" "$target"
}

install_server_from_dir() {
    local source_dir="$1"
    local target="$2"
    local stage

    stage="$(mktemp -d "$PITH_HOME/pith-server.stage.XXXXXX")"
    if ! {
        cp -R "$source_dir/app" "$stage/" &&
        cp -R "$source_dir/pith_client" "$stage/" &&
        cp "$source_dir/pith_mcp.py" "$stage/" &&
        cp "$source_dir/requirements.txt" "$stage/"
    }; then
        chmod -R u+w "$stage" 2>/dev/null || true
        rm -rf "$stage"
        return 1
    fi
    cp "$source_dir/skill_deployer.py" "$stage/" 2>/dev/null || true
    cp "$source_dir/.env.example" "$stage/" 2>/dev/null || true
    if [[ -d "$source_dir/scripts" ]] && ! cp -R "$source_dir/scripts" "$stage/"; then
        chmod -R u+w "$stage" 2>/dev/null || true
        rm -rf "$stage"
        return 1
    fi
    if [[ -d "$source_dir/migrations" ]] && ! cp -R "$source_dir/migrations" "$stage/"; then
        chmod -R u+w "$stage" 2>/dev/null || true
        rm -rf "$stage"
        return 1
    fi
    if ! activate_staged_server_tree "$stage" "$target"; then
        chmod -R u+w "$stage" 2>/dev/null || true
        rm -rf "$stage"
        return 1
    fi
}

install_server_from_tarball() {
    local archive="$1"
    local target="$2"
    local stage

    stage="$(mktemp -d "$PITH_HOME/pith-server.stage.XXXXXX")"
    if ! tar -xzf "$archive" -C "$stage"; then
        chmod -R u+w "$stage" 2>/dev/null || true
        rm -rf "$stage"
        return 1
    fi
    if ! activate_staged_server_tree "$stage" "$target"; then
        chmod -R u+w "$stage" 2>/dev/null || true
        rm -rf "$stage"
        return 1
    fi
}

# Cleanup on error
cleanup_on_failure() {
    echo ""
    echo -e "${RED}⚠ Installation failed at: Step ${CURRENT_STEP}${NC}"
    echo "  To retry: bash scripts/install.sh"
    echo "  To remove a partial install:"
    echo "    chmod -R u+w $PITH_HOME 2>/dev/null || true"
    echo "    rm -rf ${PITH_SERVER_PATH:-$PITH_HOME/pith-server} $PITH_HOME/venv"
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
    error_exit "Unsupported OS: $OSTYPE. The public developer preview is verified on macOS; Linux is an unverified source/developer path."
fi
mark_success "OS: $OS_NAME"
if [[ "$OS_TYPE" == "linux" ]]; then
    mark_warning "Linux install path is executable but not launch-verified. Use for source/developer testing and verify manually."
fi

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

# FIX P1: macOS Xcode CLT detection
if [[ "$OS_TYPE" == "macos" ]]; then
    PYTHON_PATH=$(which python3 2>/dev/null || true)

    # Check if using Xcode CLT shim
    if [[ -n "$PYTHON_PATH" ]]; then
        if [[ "$PYTHON_PATH" == "/usr/bin/python3" ]]; then
            XCLT_PYTHON=$(/usr/bin/python3 --version 2>&1 | grep -o "3\.[0-9]*" || true)
            if [[ "$XCLT_PYTHON" =~ ^3\.[0-9]+$ ]]; then
                mark_warning "Using Xcode CLT Python shim. Pith can provision Python 3.12 if needed."
            fi
        fi
    fi
fi

if [[ "$PITH_REPAIR_RUNTIME" == "1" ]]; then
    if runtime_metadata_managed_by_pith; then
        rm -rf "$PITH_RUNTIME_ROOT" "$PITH_RUNTIME_META"
        mark_warning "Repair mode removed existing Pith-managed Python runtime"
    fi
    PITH_FORCE_MANAGED_PYTHON=1
    PITH_AUTO_PYTHON=1
fi

ensure_python_runtime

if [[ "$PITH_REPAIR_RUNTIME" == "1" ]]; then
    mark_success "Python runtime repair complete"
    exit 0
fi

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

load_existing_pith_port
select_pith_port

echo ""

# ============================================================================
# STEP 3: Download Pith Server with Checksum Verification
# ============================================================================
print_step 3 "Download Pith server with SHA-256 checksum verification [FIX S1]"

PITH_SERVER_PATH="$PITH_HOME/pith-server"
SCRIPT_SOURCE="${BASH_SOURCE[0]:-}"
if [[ -n "$SCRIPT_SOURCE" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_SOURCE")" && pwd)"
    DIST_DIR="$(dirname "$SCRIPT_DIR")"
else
    SCRIPT_DIR=""
    DIST_DIR=""
fi
DOWNLOAD_SUCCESS=false

# Runtime alias guard: installed symlinks are valid only when they point to a
# dedicated runtime release worktree, not to the canonical development checkout.
if [[ -L "$PITH_SERVER_PATH" ]]; then
    SYMLINK_TARGET="$(readlink "$PITH_SERVER_PATH")"
    RESOLVED_TARGET="$(realpath "$PITH_SERVER_PATH" 2>/dev/null || true)"
    if [[ "$RESOLVED_TARGET" == *"/_release_worktrees/"* ]]; then
        echo "✓ $PITH_SERVER_PATH points at runtime release worktree. Skipping file copy."
        echo "  Target: $SYMLINK_TARGET"
        DOWNLOAD_SUCCESS=true
    else
        echo "❌ $PITH_SERVER_PATH points at an unsafe symlink target."
        echo "   Target: ${SYMLINK_TARGET:-unknown}"
        echo "   Resolved: ${RESOLVED_TARGET:-unknown}"
        echo "   Expected: ~/.pith/pith-server to point at a standalone install or a"
        echo "             runtime release worktree under /_release_worktrees/."
        echo "   Repair the runtime alias before running install again."
        exit 1
    fi

# Strategy 1: Detect running from distribution directory (most common for beta)
elif [[ -n "$DIST_DIR" && -f "$DIST_DIR/app/api/server.py" ]] && [[ -f "$DIST_DIR/pith_mcp.py" ]]; then
    echo "Detected distribution directory at: $DIST_DIR"
    install_server_from_dir "$DIST_DIR" "$PITH_SERVER_PATH"
    DOWNLOAD_SUCCESS=true
    mark_success "Installed from distribution directory"

# Strategy 2: Local tarball (created by build-release.sh)
elif [[ -n "$DIST_DIR" && -f "$DIST_DIR/pith-server-latest.tar.gz" ]]; then
    mark_success "Using local Pith server tarball"
    install_server_from_tarball "$DIST_DIR/pith-server-latest.tar.gz" "$PITH_SERVER_PATH"
    DOWNLOAD_SUCCESS=true

# Strategy 3: Download from hosted URL (GitHub Releases / CDN)
elif [[ "$PITH_LOCAL_ONLY_INSTALL" != "1" && -n "$DOWNLOAD_URL" ]]; then
    echo "Attempting download from: $DOWNLOAD_URL"
    
    TEMP_DOWNLOAD_DIR=$(mktemp -d)
    trap "rm -rf $TEMP_DOWNLOAD_DIR" EXIT
    
    DOWNLOAD_FETCHED=false
    if download_release_file "$DOWNLOAD_URL/$PITH_SERVER_FILENAME" \
        "$TEMP_DOWNLOAD_DIR/$PITH_SERVER_FILENAME" && \
       download_release_file "$CHECKSUM_URL/$PITH_CHECKSUM_FILENAME" \
        "$TEMP_DOWNLOAD_DIR/$PITH_CHECKSUM_FILENAME"; then
        DOWNLOAD_FETCHED=true
    fi
    unset PITH_DOWNLOAD_BEARER_TOKEN

    if [[ "$DOWNLOAD_FETCHED" == true ]]; then
        
        # Verify checksum
        cd "$TEMP_DOWNLOAD_DIR"
        if sha256sum -c "$PITH_CHECKSUM_FILENAME" >/dev/null 2>&1 || \
           shasum -a 256 -c "$PITH_CHECKSUM_FILENAME" >/dev/null 2>&1; then
            mark_success "Download successful and checksum verified"
            install_server_from_tarball "$TEMP_DOWNLOAD_DIR/$PITH_SERVER_FILENAME" "$PITH_SERVER_PATH"
            DOWNLOAD_SUCCESS=true
        else
            mark_warning "Checksum verification failed"
        fi
        cd - > /dev/null
    else
        mark_warning "Download failed"
    fi
elif [[ "$PITH_LOCAL_ONLY_INSTALL" == "1" ]]; then
    echo "Local-only install requested, but no local distribution or tarball was found."
fi

if [[ "$DOWNLOAD_SUCCESS" == false ]]; then
    error_exit "Could not locate Pith server files. Run this script from inside the extracted Pith directory: cd /path/to/pith && bash scripts/install.sh"
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

API_KEY=$(tr -d '\r\n' < "$API_KEY_FILE" 2>/dev/null || echo "")
if [[ -z "$API_KEY" ]]; then
    mark_error "API key missing or empty at $API_KEY_FILE"
    exit 1
fi
ensure_pith_env_value "PITH_API_KEY" "$API_KEY"
migrate_legacy_env_aliases
persist_pith_port_config
persist_preview_usage_config

echo ""

# ============================================================================
# STEP 6: Choose and Configure MCP Clients
# ============================================================================
print_step 6 "Choose and configure AI app surfaces"

select_install_surfaces

# Try real configure_clients.py first (supports standard MCP clients including Codex)
source "$VENV_PATH/bin/activate"
CONFIGURE_SCRIPT="$PITH_SERVER_PATH/scripts/configure_clients.py"

if [[ "${PITH_SELECTED_SURFACES:-all}" == "none" ]]; then
    mark_warning "No AI app surfaces selected. Local service and CLI installation will continue."
elif [[ -f "$CONFIGURE_SCRIPT" ]] && python3 "$CONFIGURE_SCRIPT" \
    --server-path "$PITH_SERVER_PATH/pith_mcp.py" \
    --python-cmd "$VENV_PATH/bin/python3" \
    --source-key-from-file "$PITH_HOME/.env" \
    --project-dir "$PITH_SERVER_PATH" \
    --platform "$OS_TYPE" \
    --api-url "http://localhost:$PITH_PORT" \
    --clients "${PITH_SELECTED_SURFACES:-all}" \
    --json 2>/dev/null; then
    mark_success "Selected MCP clients configured"
else
    # Fallback: configure Claude Desktop only
    if ! surface_selected claude_desktop; then
        mark_warning "Full client config unavailable and Claude Desktop was not selected; skipping AI app config."
    else
    mark_warning "Full client config unavailable. Configuring Claude Desktop only..."
    API_KEY=$(tr -d '\r\n' < "$API_KEY_FILE" 2>/dev/null || echo "")
    if [[ -z "$API_KEY" ]]; then
        mark_error "API key missing or empty at $API_KEY_FILE"
        exit 1
    fi
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
    "env": {
        "PITH_API_KEY": "$API_KEY",
        "PITH_API_URL": "http://localhost:$PITH_PORT",
        "PITH_SURFACE_ID": "claude_desktop_mcp",
    }
}
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
print("Claude Desktop configured")
INLINE_CONFIG
        mark_success "Claude Desktop MCP configured"
    fi
    fi
fi
deactivate

echo ""

# ============================================================================
# STEP 7: System Prompt Injection (pith_ prefix disclosure)
# ============================================================================
print_step 7 "Agent cognitive loop instructions"

SYSTEM_PROMPT_PATH="$PITH_HOME/SYSTEM_PROMPT.md"
CURSOR_GLOBAL_RULE_PATH="$PITH_HOME/CURSOR_GLOBAL_RULE.txt"

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
- `previous_message`: the user's previous message after the first exchange
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
- `pith_checkpoint` — Save/resume state for long-running work when appropriate
- `pith_session_learn` — Explicit high-quality extraction when needed
- `pith_search` — Semantic search across knowledge base
- `pith_reflect` — Run consolidation/reflection cycle
- `pith_orient` — Situational awareness summary

## Fallback if MCP Transport Fails
If direct `pith_` MCP tools return transport errors or time out, use the local
CLI fallback for the same lifecycle operation:

```bash
~/.pith/bin/pith api conversation_turn --stdin-json
~/.pith/bin/pith api checkpoint --stdin-json
~/.pith/bin/pith api session_end --stdin-json
```

Send the JSON payload on stdin. The command may print a profile banner before
the payload; parse the last non-empty output line as JSON. For lifecycle
evidence reports, use `~/.pith/bin/pith api lifecycle_status --stdin-json`.
For cross-surface source coverage evidence, use
`~/.pith/bin/pith api surface_activity --stdin-json`; treat this as coverage
evidence, not a semantic summary.
SYSPROMPT_EOF

chmod 600 "$SYSTEM_PROMPT_PATH"
mark_success "System prompt saved to $SYSTEM_PROMPT_PATH"

cat > "$CURSOR_GLOBAL_RULE_PATH" << 'CURSOR_RULE_EOF'
Pith is installed locally as a cognitive runtime for this AI agent.

Before every substantive response, retrieve Pith context first. Prefer the Pith
MCP tool `pith_conversation_turn`. Send the user's current message. After the
first exchange, also send the previous user message, your previous response, and
`extracted_concepts_json`.

Use Pith's canonical concept format for extracted concepts:
summary, confidence, knowledge_area, evidence, and concept_type. If the exchange
is trivial, send an empty concept list `[]` instead of inventing filler.

If direct Pith MCP tools are unavailable or transport-broken, use the local API
command with JSON on stdin:

`~/.pith/bin/pith api conversation_turn --stdin-json`

For long-running work, use `~/.pith/bin/pith api checkpoint --stdin-json` when
state should be saved. When a conversation ends, use
`~/.pith/bin/pith api session_end --stdin-json` when appropriate.

For lifecycle evidence reports, use
`~/.pith/bin/pith api lifecycle_status --stdin-json`; unsupported surfaces must
report `unsupported` rather than infer success.
For cross-surface source coverage evidence, use
`~/.pith/bin/pith api surface_activity --stdin-json`; sparse or missing surfaces
must trigger fallback artifacts rather than inferred coverage.

The API command may print a profile banner before the JSON payload. Parse the
last non-empty output line as JSON. If both MCP and API access fail, say Pith is
unavailable in this Cursor configuration and continue using only current
conversation context. Do not silently skip the Pith cognitive loop.

For terminal checks, prefer the absolute command `~/.pith/bin/pith` so Cursor
does not depend on its default shell PATH.
CURSOR_RULE_EOF

chmod 600 "$CURSOR_GLOBAL_RULE_PATH"
mark_success "Cursor Global Rule snippet saved to $CURSOR_GLOBAL_RULE_PATH"

PROMPT_COPIED=false
if surface_selected_and_detected claude_desktop; then
    # Copy to clipboard on macOS for easy paste into Claude Desktop
    if [[ "$OS_TYPE" == "macos" ]] && command -v pbcopy &>/dev/null; then
        if pbcopy < "$SYSTEM_PROMPT_PATH" 2>/dev/null; then
            PROMPT_COPIED=true
            mark_success "Copied to clipboard — paste into Claude Desktop Instructions for Claude"
        else
            mark_warning "Could not copy to clipboard. Paste from $SYSTEM_PROMPT_PATH instead."
        fi
    fi

    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  Add Pith's cognitive instructions to Claude Desktop:${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "  1. Open Claude Desktop → Settings → General → Instructions for Claude"
    if [[ "$PROMPT_COPIED" == true ]]; then
        echo "  2. Paste (already in your clipboard)"
    else
        echo "  2. Paste the contents of:"
        echo "     ${YELLOW}$SYSTEM_PROMPT_PATH${NC}"
    fi
    echo "  3. Save — Claude will now use Pith's cognitive loop automatically"
    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    private_beta_pause "Private beta pause: note the Claude Desktop instructions above, then press Return to continue..."
else
    if surface_selected claude_desktop; then
        mark_warning "Claude Desktop was selected but not detected. Claude instructions are saved at $SYSTEM_PROMPT_PATH."
    fi
fi

CURSOR_RULE_COPIED=false
if surface_selected_and_detected cursor; then
    if [[ "$OS_TYPE" == "macos" ]] && command -v pbcopy &>/dev/null; then
        if pbcopy < "$CURSOR_GLOBAL_RULE_PATH" 2>/dev/null; then
            CURSOR_RULE_COPIED=true
            mark_success "Copied to clipboard — paste into Cursor Global/User Rule"
        else
            mark_warning "Could not copy Cursor rule to clipboard. Paste from $CURSOR_GLOBAL_RULE_PATH instead."
        fi
    fi

    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  Add Pith's cognitive instructions to Cursor:${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "  1. Open Cursor Settings → Rules"
    echo "  2. Add a Global/User Rule"
    if [[ "$CURSOR_RULE_COPIED" == true ]]; then
        echo "  3. Paste (already in your clipboard)"
    else
        echo "  3. Paste the contents of:"
        echo "     ${YELLOW}$CURSOR_GLOBAL_RULE_PATH${NC}"
    fi
    echo "  4. Save — Cursor will have default instructions to call Pith before substantive responses"
    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    private_beta_pause "Private beta pause: note the Cursor Global Rule instructions above, then press Return to continue..."
elif surface_selected cursor; then
    mark_warning "Cursor was selected but not detected. Cursor Global Rule snippet is saved at $CURSOR_GLOBAL_RULE_PATH."
fi

if surface_selected_and_detected codex; then
    configure_codex_agents_instructions
    echo -e "${GREEN}✓${NC} Codex cognitive-loop instructions configured"
elif surface_selected codex; then
    mark_warning "Codex not detected. If you install Codex later, rerun this installer or client configuration so ~/.codex/AGENTS.md receives Pith instructions."
fi
if surface_selected vscode && [[ -f "$HOME/.copilot/instructions/pith-cognitive-loop.instructions.md" ]]; then
    echo -e "${GREEN}✓${NC} VS Code Copilot cognitive-loop instructions configured"
fi

# ============================================================================
# STEP 8: Auto-start Setup
# ============================================================================
print_step 8 "Auto-start setup (launchd/systemd) and backup scheduler"

if [[ "$OS_TYPE" == "macos" ]]; then
    # FIX P2: macOS launchd plist with post-load verification
    LAUNCHD_PLIST="$HOME/Library/LaunchAgents/dev.pith.server.plist"
    mkdir -p "$(dirname "$LAUNCHD_PLIST")"

    # PROCESS-181: snapshot the existing PITH_PORT before overwriting, so a re-run that
    # changes the port can warn that the on-disk plist now diverges from the running
    # service (this step intentionally does not launchctl reload).
    _existing_plist_port=""
    if [[ -f "$LAUNCHD_PLIST" ]]; then
        _existing_plist_port=$(grep -A1 '<key>PITH_PORT</key>' "$LAUNCHD_PLIST" \
            | grep -oE '[0-9]+' | head -1 || true)
    fi

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
        <string>serve</string>
$(if [ -n "${PITH_PROFILE:-}" ]; then cat << PROFILE_ARGS
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
        <key>PITH_PORT</key>
        <string>$PITH_PORT</string>
$(if [ -n "${PITH_PROFILE:-}" ]; then cat << PROFILE_ENV
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
    # PROCESS-181: if the port changed on a re-run, the running service still uses the old
    # port until the plist is reloaded — warn with the exact command (warning only; no reload).
    _new_plist_port=$(grep -A1 '<key>PITH_PORT</key>' "$LAUNCHD_PLIST" \
        | grep -oE '[0-9]+' | head -1 || true)
    if [[ -n "$_existing_plist_port" && "$_existing_plist_port" != "$_new_plist_port" ]]; then
        mark_warning "launchd config changed (port: $_existing_plist_port → $_new_plist_port)."
        echo "  If Pith is running, reload the service to apply the new port:"
        echo "    launchctl unload $LAUNCHD_PLIST && launchctl load -w $LAUNCHD_PLIST"
    fi
    unset _existing_plist_port _new_plist_port
    # NOTE: Do NOT launchctl load here — the pith CLI script is created in Step 8.
    # RunAtLoad will start the service on next login. Loading now would cause:
    # (1) execution of non-existent $PITH_HOME/bin/pith, (2) KeepAlive retry loop,
    # (3) configured-port collision with the health check that follows in Step 8.
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
ExecStart=$PITH_HOME/bin/pith serve
Restart=always
RestartSec=10
Environment="PITH_HOME=$PITH_HOME"
Environment="PITH_PORT=$PITH_PORT"
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

PITH_CLI_SOURCE="${BASH_SOURCE[0]:-$0}"
PITH_CLI_BIN_DIR="$(cd "$(dirname "$PITH_CLI_SOURCE")" && pwd -P)"
PITH_CLI_DEFAULT_HOME="$(cd "$PITH_CLI_BIN_DIR/.." && pwd -P)"
PITH_HOME="${PITH_HOME:-$PITH_CLI_DEFAULT_HOME}"
VENV_PATH="$PITH_HOME/venv"
PITH_SERVER_PATH="$PITH_HOME/pith-server"
PITH_RUNTIME_META="$PITH_HOME/config/python-runtime.json"

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
PITH_EXPLICIT_DATA_DIR="${PITH_DATA_DIR:-}"
PITH_EXPLICIT_PROFILE="${PITH_PROFILE:-}"
PITH_EXPLICIT_API_KEY="${PITH_API_KEY:-}"
if [[ -f "$PITH_HOME/.env" ]]; then
    set -a  # auto-export sourced variables
    source "$PITH_HOME/.env"
    set +a
fi
if [[ -n "$PITH_EXPLICIT_PROFILE" ]]; then
    export PITH_PROFILE="$PITH_EXPLICIT_PROFILE"
fi
if [[ -n "$PITH_EXPLICIT_DATA_DIR" ]]; then
    export PITH_DATA_DIR="$PITH_EXPLICIT_DATA_DIR"
fi
if [[ -n "$PITH_EXPLICIT_API_KEY" ]]; then
    export PITH_API_KEY="$PITH_EXPLICIT_API_KEY"
fi

# CLI --profile flag overrides .env before profile-derived paths are resolved.
if [[ -n "$PROFILE" ]]; then
    export PITH_PROFILE="$PROFILE"
fi

PITH_PORT="${PITH_PORT:-8000}"
PITH_DEFAULT_HOME="$HOME/.pith"
if [[ -z "${PITH_DATA_DIR:-}" ]]; then
    if [[ -n "${PITH_PROFILE:-}" ]]; then
        PITH_DATA_DIR="$HOME/pith-data/$PITH_PROFILE"
    elif [[ "$PITH_HOME" != "$PITH_DEFAULT_HOME" ]]; then
        PITH_DATA_DIR="$(cd "$PITH_HOME/.." && pwd -P)/pith-data/default"
    else
        PITH_DATA_DIR="$HOME/pith-data/default"
    fi
fi
if [[ -z "${PITH_LAUNCH_AGENTS_DIR:-}" && "$PITH_HOME" != "$PITH_DEFAULT_HOME" ]]; then
    PITH_LAUNCH_AGENTS_DIR="$(cd "$PITH_HOME/.." && pwd -P)/Library/LaunchAgents"
fi
PITH_API_URL="${PITH_API_URL:-http://localhost:$PITH_PORT}"
export PITH_HOME PITH_PORT PITH_DATA_DIR PITH_LAUNCH_AGENTS_DIR PITH_API_URL PITH_VERSION PITH_RUNTIME_META

# Report active profile after path resolution.
if [[ -n "$PROFILE" ]]; then
    echo "Using profile: $PROFILE (data: $HOME/pith-data/$PROFILE/)" >&2
elif [[ -n "$PITH_EXPLICIT_DATA_DIR" ]]; then
    echo "Using data dir override: $PITH_DATA_DIR" >&2
elif [[ -n "${PITH_PROFILE:-}" ]]; then
    echo "Using profile from .env: $PITH_PROFILE (data: $HOME/pith-data/$PITH_PROFILE/)" >&2
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
    if [[ "$PITH_HOME" != "$PITH_DEFAULT_HOME" ]]; then echo "$(cd "$PITH_HOME/.." && pwd -P)/pith-data/default"; return; fi
    echo "$HOME/pith-data/default"
}

resolve_db_path() {
    echo "$(resolve_data_dir)/pith.db"
}

resolve_backup_dir() {
    echo "$(resolve_data_dir)/backups"
}

list_backup_files() {
    local backup_dir
    backup_dir="$(resolve_backup_dir)"
    if [[ ! -d "$backup_dir" ]]; then
        return 0
    fi
    find "$backup_dir" -maxdepth 1 -type f -name 'pith_backup_*.db' ! -name '*-wal' ! -name '*-shm' -print 2>/dev/null | sort -r
}

find_latest_backup() {
    list_backup_files | head -1
}

remove_client_config_entries() {
    python3 - <<'PY'
import json
import shutil
import time
from pathlib import Path

home = Path.home()
server_names = {"pith", "pith-mcp", "pith-mcp-wrapper"}

json_configs = [
    home / "Library/Application Support/Claude/claude_desktop_config.json",
    home / "Library/Application Support/Cursor/User/globalStorage/cursor.mcp/config.json",
    home / "Library/Application Support/Windsurf/User/globalStorage/windsurf.mcp/config.json",
    home / "Library/Application Support/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json",
    home / ".continue/config.json",
    home / ".config/Claude/claude_desktop_config.json",
    home / ".config/Cursor/User/globalStorage/cursor.mcp/config.json",
    home / ".config/Windsurf/User/globalStorage/windsurf.mcp/config.json",
    home / ".config/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json",
]

def backup(path: Path) -> Path:
    backup_path = path.with_name(path.name + f".backup.{int(time.time())}")
    shutil.copy2(path, backup_path)
    return backup_path

for path in json_configs:
    if not path.is_file():
        continue
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Skipped unparsable client config: {path} ({exc})")
        continue
    changed = False
    for root in ("mcpServers", "mcp_servers"):
        value = payload.get(root)
        if isinstance(value, dict):
            for name in list(value):
                if name in server_names:
                    del value[name]
                    changed = True
    if changed:
        backup_path = backup(path)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Removed Pith MCP entry from {path} (backup: {backup_path})")

codex_config = home / ".codex/config.toml"
if codex_config.is_file():
    lines = codex_config.read_text(encoding="utf-8").splitlines(keepends=True)
    output = []
    changed = False
    skip = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped.strip("[]")
            parts = section.split(".")
            skip = (
                len(parts) >= 2
                and parts[0] == "mcp_servers"
                and parts[1] in server_names
            )
            if skip:
                changed = True
                continue
        if not skip:
            output.append(line)
    if changed:
        backup_path = backup(codex_config)
        codex_config.write_text("".join(output), encoding="utf-8")
        print(f"Removed Pith MCP entry from {codex_config} (backup: {backup_path})")

codex_agents = home / ".codex/AGENTS.md"
if codex_agents.is_file():
    text = codex_agents.read_text(encoding="utf-8")
    start = "<!-- PITH COGNITIVE LOOP: START -->"
    end = "<!-- PITH COGNITIVE LOOP: END -->"
    changed = False
    if start in text and end in text:
        before, rest = text.split(start, 1)
        _, after = rest.split(end, 1)
        text = (before + after).strip() + "\n"
        changed = True
    elif text.lstrip().startswith("# Pith Cognitive Loop"):
        text = ""
        changed = True
    if changed:
        backup_path = backup(codex_agents)
        if text.strip():
            codex_agents.write_text(text, encoding="utf-8")
            print(f"Removed Pith instructions from {codex_agents} (backup: {backup_path})")
        else:
            codex_agents.unlink()
            print(f"Removed Pith instructions file {codex_agents} (backup: {backup_path})")
PY
}

remove_shell_path_entries() {
    python3 - <<'PY'
from pathlib import Path
import shutil
import time

home = Path.home()
rc_files = [
    home / ".zshrc",
    home / ".zprofile",
    home / ".zshenv",
    home / ".bashrc",
    home / ".bash_profile",
    home / ".profile",
]

for path in rc_files:
    if not path.is_file():
        continue
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    output = []
    changed = False
    i = 0
    while i < len(lines):
        line = lines[i]
        next_line = lines[i + 1] if i + 1 < len(lines) else ""
        if line.strip() == "# Pith CLI" and ".pith/bin" in next_line:
            changed = True
            i += 2
            if output and output[-1].strip() == "":
                output.pop()
            continue
        if ".pith/bin" in line:
            changed = True
            i += 1
            continue
        output.append(line)
        i += 1
    if changed:
        backup_path = path.with_name(path.name + f".backup.{int(time.time())}")
        shutil.copy2(path, backup_path)
        path.write_text("".join(output), encoding="utf-8")
        print(f"Removed Pith PATH entry from {path} (backup: {backup_path})")
PY
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

runtime_log_path() {
    local DATA_DIR
    DATA_DIR=$(resolve_data_dir)
    mkdir -p "$DATA_DIR/logs"
    echo "$DATA_DIR/logs/pith.log"
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
    RESPONSE=$(curl -s --max-time 3 "http://127.0.0.1:${PITH_PORT}/health" 2>/dev/null)
    if [[ -z "$RESPONSE" ]]; then echo "Unreachable"; return 1; fi
    if echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('service')=='pith'" 2>/dev/null; then
        echo "OK (Pith)"
        return 0
    else
        echo "Port $PITH_PORT responding but NOT Pith"
        return 1
    fi
}

check_port() {
    # Try lsof (macOS/most Linux), then ss (modern Linux), then netstat
    if command -v lsof >/dev/null 2>&1; then
        lsof -i ":$PITH_PORT" -sTCP:LISTEN -t 2>/dev/null | head -1
    elif command -v ss >/dev/null 2>&1; then
        ss -tlnp 2>/dev/null | awk -v port=":$PITH_PORT" '$4 ~ port {gsub(/.*pid=/,""); gsub(/,.*/,""); print; exit}'
    elif command -v netstat >/dev/null 2>&1; then
        netstat -tlnp 2>/dev/null | awk -v port=":$PITH_PORT" '$4 ~ port {gsub(/.*\//,""); print; exit}'
    fi
}

runtime_meta_value() {
    local key="$1"
    [[ -f "$PITH_RUNTIME_META" ]] || return 1
    grep "\"$key\"" "$PITH_RUNTIME_META" 2>/dev/null | head -1 | sed -E 's/^[^:]*:[[:space:]]*"([^"]*)".*/\1/'
}

print_runtime_provenance() {
    if [[ ! -f "$PITH_RUNTIME_META" ]]; then
        echo "  Runtime:      unknown (no python-runtime.json)"
        return
    fi
    local managed runtime_id py_exe sha source
    managed="$(runtime_meta_value managed_by || echo unknown)"
    runtime_id="$(runtime_meta_value runtime_id || echo unknown)"
    py_exe="$(runtime_meta_value python_executable || echo unknown)"
    sha="$(runtime_meta_value sha256 || echo unknown)"
    source="$(runtime_meta_value source || echo unknown)"
    echo "  Runtime:      $managed"
    echo "  Runtime ID:   $runtime_id"
    echo "  Runtime exe:  $py_exe"
    echo "  Runtime src:  $source"
    echo "  Runtime sha:  $sha"
}

port_in_use() {
    PITH_CHECK_PORT="$PITH_PORT" python3 - <<'PY'
import os
import socket
import sys

port = int(os.environ["PITH_CHECK_PORT"])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(("127.0.0.1", port))
except OSError:
    sys.exit(0)
finally:
    s.close()
sys.exit(1)
PY
}

is_help_request() {
    [[ "${ARGS[1]:-}" == "-h" || "${ARGS[1]:-}" == "--help" ]]
}

print_wrapper_help() {
    case "$1" in
        serve)
            echo "Usage: pith serve"
            echo "  Run the Pith API server in the foreground for a process manager."
            ;;
        start)
            echo "Usage: pith start"
            echo "  Start the Pith API server in the background."
            ;;
        stop)
            echo "Usage: pith stop"
            echo "  Stop the Pith API server for this PITH_HOME/PITH_PORT."
            ;;
        restart)
            echo "Usage: pith restart"
            echo "  Restart the Pith API server."
            ;;
        logs)
            echo "Usage: pith logs [snapshot [--json] [--file {pith,err,both}] [--lines N]]"
            echo "  Tail the Pith log, or print a bounded snapshot."
            ;;
        backup)
            echo "Usage: pith backup"
            echo "  Run the configured WAL-safe backup helper."
            ;;
        restore)
            echo "Usage: pith restore [backup_file]"
            echo "  Restore a profile database from a backup file."
            ;;
        update)
            echo "Usage: pith update"
            echo "  Run local dependency and migration update checks."
            ;;
        version)
            echo "Usage: pith version"
            echo "  Show installed Pith and runtime version information."
            ;;
        runtime)
            echo "Usage: pith runtime {status|repair}"
            echo "  status  Show Python runtime provenance"
            echo "  repair  Reinstall the managed Python runtime when eligible"
            ;;
        uninstall)
            echo "Usage: pith uninstall"
            echo "  Remove Pith after an interactive confirmation prompt."
            ;;
        profiles)
            echo "Usage: pith profiles"
            echo "  List local Pith profiles."
            ;;
        maintenance)
            echo "Usage: pith maintenance {run|status|install|uninstall}"
            echo "  run [--phases 1,2,3] [--dry-run]  Run maintenance cycle"
            echo "  status                             Show task status"
            echo "  install                            Install optional launchd scheduler"
            echo "  uninstall                          Remove optional launchd scheduler"
            ;;
        stats)
            echo "Usage: pith stats"
            echo "  Show quick knowledge-base statistics from the active profile database."
            ;;
        protocol)
            echo "Usage: pith protocol"
            echo "  Print Pith cognitive-loop instructions and copy them to clipboard when available."
            ;;
        *)
            echo "Usage: pith [--profile NAME] {serve|start|stop|restart|status|health|stats|logs|search|concept|orient|sessions|metrics|doctor|clients|support|import|api|api-fallback|backup|restore|update|version|report|profiles|maintenance|protocol|runtime|uninstall}"
            ;;
    esac
}

case "${1:-status}" in
    serve)
        if is_help_request; then print_wrapper_help serve; exit 0; fi
        # Foreground server mode for process managers.
        # launchd/systemd must supervise the long-running uvicorn process, not
        # the short-lived `pith start` wrapper that daemonizes and exits.
        echo "Serving Pith Server..."
        cd "$PITH_SERVER_PATH"
        exec python -m uvicorn app.api.server:app \
            --host "${PITH_HOST:-127.0.0.1}" \
            --port "${PITH_PORT:-8000}" \
            --workers "${PITH_UVICORN_WORKERS:-1}" \
            --log-level "${PITH_LOG_LEVEL:-info}"
        ;;
    start)
        if is_help_request; then print_wrapper_help start; exit 0; fi
        echo "Starting Pith Server..."

        # Pre-check: port availability
        BLOCKING_PID=$(check_port)
        if [[ -n "$BLOCKING_PID" ]] || port_in_use; then
            BLOCKING_CMD=$(ps -p "$BLOCKING_PID" -o comm= 2>/dev/null || echo "unknown")
            if [[ -n "$BLOCKING_PID" ]]; then
                echo "Error: Port $PITH_PORT is already in use by $BLOCKING_CMD (PID $BLOCKING_PID)"
            else
                echo "Error: Port $PITH_PORT is already in use by another process."
            fi
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
        RUNTIME_LOG=$(runtime_log_path)
        SERVER_PID=$(PITH_SERVER_PATH="$PITH_SERVER_PATH" PITH_PORT="$PITH_PORT" RUNTIME_LOG="$RUNTIME_LOG" python3 - <<'PY'
import os
import subprocess
import sys

env = os.environ.copy()
log_path = os.environ["RUNTIME_LOG"]
server_path = os.environ["PITH_SERVER_PATH"]
port = os.environ["PITH_PORT"]
log_file = open(log_path, "ab", buffering=0)
proc = subprocess.Popen(
    [
        sys.executable,
        "-m",
        "uvicorn",
        "app.api.server:app",
        "--host",
        "127.0.0.1",
        "--port",
        port,
        "--log-level",
        "info",
    ],
    cwd=server_path,
    stdin=subprocess.DEVNULL,
    stdout=log_file,
    stderr=subprocess.STDOUT,
    env=env,
    start_new_session=True,
    close_fds=True,
)
print(proc.pid)
PY
)
        if [[ -z "$SERVER_PID" ]]; then
            echo "Failed to launch Pith server process."
            exit 1
        fi

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
            if check_pith_health >/dev/null 2>&1; then
                sleep 0.2
                if ! ps -p "$SERVER_PID" > /dev/null 2>&1; then
                    echo " FAILED"
                    echo "Server process exited after initial health response. Check logs:"
                    LOG_FILE=$(resolve_log_path)
                    if [[ -n "$LOG_FILE" ]]; then tail -20 "$LOG_FILE"; fi
                    exit 1
                fi
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
        if is_help_request; then print_wrapper_help stop; exit 0; fi
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
        if is_help_request; then print_wrapper_help restart; exit 0; fi
        # OPS-099: Under launchd KeepAlive, stop+start races — launchd respawns within ~200ms
        # after stop, so start finds the configured port occupied and exits 1. Use kickstart -k instead.
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
        if [[ " ${ARGS[*]:1} " != *" --json "* && ! is_help_request ]]; then
            echo "Port:         $PITH_PORT"
        fi
        cd "$PITH_SERVER_PATH"
        python3 -m app.ops.support_cli status "${ARGS[@]:1}"
        ;;
    health)
        cd "$PITH_SERVER_PATH"
        python3 -m app.ops.health_cli "${ARGS[@]:1}"
        ;;
    logs)
        if is_help_request; then print_wrapper_help logs; exit 0; fi
        if [[ "${ARGS[1]:-}" == "snapshot" ]]; then
            cd "$PITH_SERVER_PATH"
            python3 -m app.ops.read_cli logs "${ARGS[@]:1}"
        else
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
        fi
        ;;
    import)
        cd "$PITH_SERVER_PATH"
        python3 -m app.ops.import_cli "${ARGS[@]:1}"
        ;;
    search|concept|orient|sessions|metrics)
        cd "$PITH_SERVER_PATH"
        python3 -m app.ops.read_cli "$1" "${ARGS[@]:1}"
        ;;
    doctor|clients|support)
        cd "$PITH_SERVER_PATH"
        python3 -m app.ops.support_cli "$1" "${ARGS[@]:1}"
        ;;
    api|api-fallback)
        COMMAND_NAME="${1:-}"
        shift
        TRANSPORT_MODE="first_class_api"
        if [[ "$COMMAND_NAME" == "api-fallback" ]]; then
            TRANSPORT_MODE="exec_http_fallback"
        fi
        if [[ "${PITH_EXEC_FALLBACK_ENABLED:-1}" != "1" ]]; then
            printf '{"error":true,"code":"DISABLED","message":"exec HTTP API command disabled","transport_mode":"%s"}\n' "$TRANSPORT_MODE"
            exit 64
        fi
        cd "$PITH_SERVER_PATH"
        python -m pith_client.cli "$@" --transport-mode "$TRANSPORT_MODE"
        ;;
    backup)
        if is_help_request; then print_wrapper_help backup; exit 0; fi
        "$PITH_HOME/bin/backup"
        ;;
    restore)
        if is_help_request; then print_wrapper_help restore; exit 0; fi
        # Safe restore: stop → copy → verify → start
        BACKUP_FILE="${2:-}"
        if [[ -z "$BACKUP_FILE" ]]; then
            BACKUP_FILE=$(find_latest_backup)
            if [[ -z "$BACKUP_FILE" ]]; then
                echo "No backups found in $(resolve_backup_dir)"
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
        if is_help_request; then print_wrapper_help update; exit 0; fi
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
        if is_help_request; then print_wrapper_help version; exit 0; fi
        echo "Pith v${PITH_VERSION:-unknown}"
        echo "Python: $("$VENV_PATH/bin/python3" --version 2>/dev/null || echo 'not found')"
        echo "Python exe: $VENV_PATH/bin/python3"
        echo "OS: $(uname -s) $(uname -m)"
        print_runtime_provenance
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
        # Use the same status collector as `pith status` so launchd/readyz-backed
        # services do not look stopped when the legacy PID file is missing.
        STATUS_JSON=$({ cd "$PITH_SERVER_PATH" && python3 -m app.ops.support_cli status --json; } 2>/dev/null || true)
        STATUS_LINE=$(STATUS_JSON="$STATUS_JSON" python3 - <<'PY' 2>/dev/null || true
import json
import os
import sys

try:
    data = json.loads(os.environ.get("STATUS_JSON", ""))
except Exception:
    sys.exit(1)

if data.get("running"):
    pid = data.get("pid")
    if pid:
        print(f"Status: running (PID: {pid})")
    else:
        print("Status: running")
else:
    print("Status: stopped")
PY
)
        if [[ -n "$STATUS_LINE" ]]; then
            echo "$STATUS_LINE"
        elif [[ -f "$PITH_HOME/pith.pid" ]] && ps -p $(cat "$PITH_HOME/pith.pid") > /dev/null 2>&1; then
            echo "Status: running (PID: $(cat "$PITH_HOME/pith.pid"))"
        else
            echo "Status: stopped"
        fi
        ;;
    runtime)
        if is_help_request; then print_wrapper_help runtime; exit 0; fi
        case "${2:-status}" in
            status)
                print_runtime_provenance
                ;;
            repair)
                INSTALLER="$PITH_SERVER_PATH/scripts/install.sh"
                if [[ ! -f "$INSTALLER" ]]; then
                    echo "Runtime repair needs the installer at $INSTALLER"
                    echo "Run from the extracted beta artifact: PITH_AUTO_PYTHON=1 PITH_REPAIR_RUNTIME=1 bash scripts/install.sh"
                    exit 1
                fi
                if [[ -f "$PITH_RUNTIME_META" ]] && [[ "$(runtime_meta_value managed_by || true)" != "pith" ]]; then
                    echo "Refusing to repair non-Pith Python runtime: $(runtime_meta_value python_executable || echo unknown)"
                    exit 1
                fi
                PITH_AUTO_PYTHON=1 PITH_REPAIR_RUNTIME=1 bash "$INSTALLER"
                ;;
            *)
                echo "Usage: pith runtime {status|repair}"
                exit 1
                ;;
        esac
        ;;
    uninstall)
        if is_help_request; then print_wrapper_help uninstall; exit 0; fi
        DATA_DIR="$(resolve_data_dir)"
        echo "WARNING: This will uninstall Pith and remove all data for this profile."
        echo "  Install: $PITH_HOME"
        echo "  Data:    $DATA_DIR"
        if [[ -f "$PITH_RUNTIME_META" ]]; then
            echo "  Python:  $(runtime_meta_value managed_by || echo unknown) runtime at $(runtime_meta_value python_executable || echo unknown)"
        fi
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
            remove_client_config_entries
            remove_shell_path_entries
            rm -rf "$PITH_HOME"
            if [[ "$DATA_DIR" == "$HOME/pith-data" || "$DATA_DIR" == "$HOME/pith-data/"* ]]; then
                rm -rf "$DATA_DIR"
                rmdir "$HOME/pith-data" 2>/dev/null || true
            else
                echo "Skipped data removal outside ~/pith-data: $DATA_DIR"
            fi
            echo "Pith uninstalled"
        else
            echo "Uninstall cancelled"
        fi
        ;;
    profiles)
        if is_help_request; then print_wrapper_help profiles; exit 0; fi
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
        if is_help_request; then print_wrapper_help maintenance; exit 0; fi
        MAINT_CMD="${ARGS[1]:-run}"
        case "$MAINT_CMD" in
            run)
                echo "Running Pith maintenance cycle..."
                cd "$PITH_SERVER_PATH"
                python3 -m app.ops.maintenance_cli run "${ARGS[@]:2}"
                ;;
            status)
                cd "$PITH_SERVER_PATH"
                python3 -m app.ops.maintenance_cli status
                ;;
            install)
                echo "Installing maintenance scheduler..."
                cd "$PITH_SERVER_PATH"
                python3 -m app.ops.maintenance_cli install
                ;;
            uninstall)
                cd "$PITH_SERVER_PATH"
                python3 -m app.ops.maintenance_cli uninstall
                ;;
            *)
                echo "Usage: pith maintenance {run|status|install|uninstall}"
                echo "  run [--phases 1,2,3] [--dry-run]  Run maintenance cycle"
                echo "  status                             Show task status"
                echo "  install                            Install optional launchd scheduler (daily 3:00 AM)"
                echo "  uninstall                          Remove optional launchd scheduler"
                ;;
        esac
        ;;
    report)
        cd "$PITH_SERVER_PATH"
        python3 -m app.ops.support_cli report "${ARGS[@]:1}"
        ;;
    stats)
        if is_help_request; then print_wrapper_help stats; exit 0; fi
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
        if is_help_request; then print_wrapper_help protocol; exit 0; fi
        # Show Pith cognitive loop instructions for Claude Desktop Instructions for Claude
        SYSPROMPT="$PITH_HOME/SYSTEM_PROMPT.md"
        if [[ ! -f "$SYSPROMPT" ]]; then
            echo "System prompt not found at $SYSPROMPT"
            echo "Re-run the installer to regenerate it."
            exit 1
        fi
        cat "$SYSPROMPT"
        echo ""
        if [[ "$(uname)" == "Darwin" ]] && command -v pbcopy &>/dev/null; then
            if pbcopy < "$SYSPROMPT" 2>/dev/null; then
                echo "--- Copied to clipboard. Paste into Claude Desktop → Settings → General → Instructions for Claude ---"
            else
                echo "--- Clipboard unavailable. Paste the text above into Claude Desktop → Settings → General → Instructions for Claude ---"
            fi
        else
            echo "--- Paste the above into Claude Desktop → Settings → General → Instructions for Claude ---"
        fi
        ;;
    *)
        echo "Usage: pith [--profile NAME] {serve|start|stop|restart|status|health|stats|logs|search|concept|orient|sessions|metrics|doctor|clients|support|import|api|api-fallback|backup|restore|update|version|report|profiles|maintenance|protocol|runtime|uninstall}"
        exit 1
        ;;
esac

COMMAND_STATUS=$?
deactivate
exit $COMMAND_STATUS
PITH_CLI_SCRIPT
# === CLI_TEMPLATE_END ===

chmod +x "$PITH_HOME/bin/pith"
# Replace version placeholder (heredoc is single-quoted so vars don't expand)
sed -i.bak "s/__PITH_VERSION__/$PITH_VERSION/g" "$PITH_HOME/bin/pith" && rm -f "$PITH_HOME/bin/pith.bak"
mark_success "Created pith CLI at $PITH_HOME/bin/pith"

# Run health check
echo "Running health check..."
source "$VENV_PATH/bin/activate"

verify_installed_pith_health() {
    local status_output=""
    local attempt
    for attempt in {1..10}; do
        status_output="$("$PITH_HOME/bin/pith" status 2>&1 || true)"
        if grep -q "Health: OK (Pith)" <<< "$status_output"; then
            return 0
        fi
        sleep 1
    done
    echo "$status_output"
    return 1
}

verify_installed_pith_durable_health() {
    local delay_seconds="${PITH_DURABILITY_CHECK_SECONDS:-10}"
    local status_output=""
    if ! verify_installed_pith_health; then
        return 1
    fi
    sleep "$delay_seconds"
    status_output="$("$PITH_HOME/bin/pith" status 2>&1 || true)"
    if grep -q "Health: OK (Pith)" <<< "$status_output"; then
        return 0
    fi
    echo "$status_output"
    return 1
}

# The install smoke must validate this install, not an unrelated server already
# bound to the default port.
if port_in_use_value "$PITH_PORT"; then
    deactivate
    error_exit "Port $PITH_PORT is already in use. Stop the existing Pith service or any process on port $PITH_PORT and rerun install."
else
    if "$PITH_HOME/bin/pith" start; then
        if verify_installed_pith_health; then
            deactivate
            mark_success "Health check passed"
        else
            deactivate
            error_exit "Pith started but health check did not report OK."
        fi
    else
        deactivate
        error_exit "Pith failed to start during install health check."
    fi

    echo ""
fi  # end of port availability else block

# ============================================================================
# STEP 8b: Auto-configure shell PATH [FIX A1]
# ============================================================================
PATH_ENTRY="export PATH=\"\$HOME/.pith/bin:\$PATH\""
PATH_ADDED=false

# Detect user's shell and choose profile files. macOS terminal sessions are
# not consistent about which zsh startup file they read first, so configure
# both interactive and login surfaces.
SHELL_NAME=$(basename "${SHELL:-/bin/bash}")
case "$SHELL_NAME" in
    zsh)
        SHELL_RC_FILES=("$HOME/.zshenv" "$HOME/.zshrc" "$HOME/.zprofile" "$HOME/.profile")
        SHELL_RC_CREATE_ALL=true
        ;;
    bash)
        SHELL_RC_FILES=("$HOME/.bashrc" "$HOME/.bash_profile" "$HOME/.profile")
        SHELL_RC_CREATE_ALL=true
        ;;
    *)
        SHELL_RC_FILES=("$HOME/.profile")
        SHELL_RC_CREATE_ALL=false
        ;;
esac

for SHELL_RC in "${SHELL_RC_FILES[@]}"; do
    if [[ -f "$SHELL_RC" ]] || [[ "$SHELL_RC" == "${SHELL_RC_FILES[0]}" ]] || [[ "$SHELL_RC_CREATE_ALL" == true ]]; then
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

GLOBAL_CLI_PATH=""
GLOBAL_CLI_LINK_SKIPPED=false
if [[ "$PITH_SKIP_GLOBAL_CLI_LINK" == "1" ]]; then
    mark_warning "Skipping global pith CLI link because PITH_SKIP_GLOBAL_CLI_LINK=1."
    GLOBAL_CLI_LINK_SKIPPED=true
elif [[ "$PITH_HOME" != "$PITH_CANONICAL_HOME" && "$PITH_FORCE_GLOBAL_CLI_LINK" != "1" ]]; then
    mark_warning "Skipping global pith CLI link for non-canonical PITH_HOME: $PITH_HOME"
    GLOBAL_CLI_LINK_SKIPPED=true
else
    for GLOBAL_CLI_DIR in /usr/local/bin /opt/homebrew/bin; do
        if [[ -d "$GLOBAL_CLI_DIR" && -w "$GLOBAL_CLI_DIR" ]]; then
            CANDIDATE="$GLOBAL_CLI_DIR/pith"
            if [[ ! -e "$CANDIDATE" || -L "$CANDIDATE" ]]; then
                ln -sf "$PITH_HOME/bin/pith" "$CANDIDATE"
                GLOBAL_CLI_PATH="$CANDIDATE"
                mark_success "Linked pith CLI at $GLOBAL_CLI_PATH for app-launched shells"
                break
            elif "$CANDIDATE" version >/dev/null 2>&1; then
                GLOBAL_CLI_PATH="$CANDIDATE"
                mark_success "pith CLI already available at $GLOBAL_CLI_PATH"
                break
            else
                mark_warning "Found existing non-Pith command at $CANDIDATE; leaving it unchanged."
            fi
        fi
    done
fi

if [[ -z "$GLOBAL_CLI_PATH" && "$GLOBAL_CLI_LINK_SKIPPED" != true ]]; then
    mark_warning "Could not link pith into /usr/local/bin or /opt/homebrew/bin. Shell profiles were still updated, but app-launched shells may need the absolute command $PITH_HOME/bin/pith."
fi

echo ""

if verify_installed_pith_durable_health >/dev/null 2>&1; then
    mark_success "Final service durability check passed"
else
    mark_warning "Final service durability check failed; retrying once..."
    "$PITH_HOME/bin/pith" start >/dev/null 2>&1 || true
    if verify_installed_pith_durable_health >/dev/null 2>&1; then
        mark_success "Final service durability check passed after retry"
    else
        error_exit "Pith service is not healthy after install. Run $PITH_HOME/bin/pith logs for details."
    fi
fi

# ============================================================================
# Final Success Message
# ============================================================================
print_banner

echo -e "${GREEN}✓ Installation Complete!${NC}"
echo ""
echo -e "Pith is installed at: ${BLUE}$PITH_HOME${NC}"
echo -e "Pith API URL:       ${BLUE}http://localhost:$PITH_PORT${NC}"
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
echo "  • ${YELLOW}pith health${NC}  - Check operational health/readiness"
echo "  • ${YELLOW}pith logs${NC}    - View server logs"
echo "  • ${YELLOW}pith import${NC}  - Import conversation exports safely"
echo "  • ${YELLOW}pith api${NC}     - First-class local HTTP/API lifecycle calls"
echo "  • ${YELLOW}pith backup${NC}  - Create manual backup (WAL-safe)"
echo "  • ${YELLOW}pith restore${NC} - Restore from backup"
echo "  • ${YELLOW}pith update${NC}  - Update Pith"
echo "  • ${YELLOW}pith version${NC} - Show version and system info"
echo ""
echo -e "${BLUE}Required setup:${NC}"
SETUP_STEP=1
echo "  ${SETUP_STEP}. Open a new terminal (or reload your shell profile)"
SETUP_STEP=$((SETUP_STEP + 1))
if surface_selected_and_detected claude_desktop; then
    echo "  ${SETUP_STEP}. Claude Desktop instructions: if you pasted them during Step 7, this is already done."
    echo "     To redo later: ${YELLOW}pith protocol${NC}  (copies prompt for Settings → General → Instructions for Claude)"
    SETUP_STEP=$((SETUP_STEP + 1))
fi
echo "  ${SETUP_STEP}. Restart each configured AI client completely before testing it"
if surface_selected_and_detected cursor; then
    echo "  Cursor: MCP config is installed, but Cursor also needs a Global/User Rule for default Pith invocation."
    echo "     Paste ${YELLOW}$CURSOR_GLOBAL_RULE_PATH${NC} into Cursor Settings → Rules."
fi
if surface_selected_and_detected windsurf; then
    echo "  Windsurf: experimental MCP template is installed; this path is not launch-verified."
fi
echo ""
echo -e "${BLUE}Verification checks:${NC}"
echo "  1. Core install: ${YELLOW}pith status${NC}  (expect Health: OK (Pith))"
VERIFY_STEP=2
if surface_selected_and_detected claude_desktop; then
    echo "  ${VERIFY_STEP}. Claude Desktop: start a fresh conversation and confirm a Pith tool call"
    echo "     Log: ${YELLOW}~/Library/Logs/Claude/mcp-server-pith.log${NC}"
    VERIFY_STEP=$((VERIFY_STEP + 1))
fi
if surface_selected_and_detected claude_code; then
    echo "  ${VERIFY_STEP}. Claude Code: run ${YELLOW}claude mcp get pith${NC}, then start a fresh Claude Code session and confirm a Pith tool call"
    VERIFY_STEP=$((VERIFY_STEP + 1))
fi
if surface_selected_and_detected codex; then
    echo "  ${VERIFY_STEP}. Codex: confirm ${YELLOW}~/.codex/AGENTS.md${NC} exists and references ${YELLOW}pith api conversation_turn${NC}"
    echo "     If Codex was installed after Pith, rerun this installer or client configuration"
    VERIFY_STEP=$((VERIFY_STEP + 1))
fi
if surface_selected_and_detected vscode; then
    echo "  ${VERIFY_STEP}. VS Code: restart VS Code, then run ${YELLOW}MCP: List Servers${NC}"
    echo "     User config: ${YELLOW}~/Library/Application Support/Code/User/mcp.json${NC}"
    echo "     Copilot instructions: ${YELLOW}~/.copilot/instructions/pith-cognitive-loop.instructions.md${NC}"
    echo "     In Chat Diagnostics, confirm that instruction file is loaded for Agent Chat"
    VERIFY_STEP=$((VERIFY_STEP + 1))
fi
if surface_selected_and_detected cursor; then
    echo "  ${VERIFY_STEP}. Cursor: ask a normal project-context question and confirm a Pith tool call in Cursor's tool activity"
    echo "     Config: ${YELLOW}~/.cursor/mcp.json${NC}; rule setup is separate from MCP config"
    VERIFY_STEP=$((VERIFY_STEP + 1))
fi
if surface_selected_and_detected windsurf; then
    echo "  ${VERIFY_STEP}. Windsurf (experimental): manually confirm whether Cascade calls Pith"
    echo "     Config: ${YELLOW}~/.codeium/windsurf/mcp_config.json${NC}"
    VERIFY_STEP=$((VERIFY_STEP + 1))
fi
echo "  ${VERIFY_STEP}. View logs if a check fails: ${YELLOW}pith logs${NC}"
echo ""
if [[ "${PITH_PRIVATE_BETA:-0}" == "1" || "${PITH_LOCAL_ONLY_INSTALL:-0}" == "1" || -f "$DIST_DIR/.private-beta" ]]; then
    echo -e "${BLUE}Private Beta:${NC}"
    echo "  Use the instructions included with this artifact."
    echo "  Do not use public repository release flows for this build."
    if [[ -n "${PITH_INSTALL_LOG:-}" ]]; then
        echo "  Install log: ${YELLOW}$PITH_INSTALL_LOG${NC}"
    fi
else
    echo -e "${BLUE}Documentation:${NC}"
    echo "  https://docs.pith.dev"
    echo "  https://github.com/pithrun/pith-core"
fi
echo ""
private_beta_pause "Private beta setup complete. Review the next steps above, then press Return to close this window..."

exit 0
