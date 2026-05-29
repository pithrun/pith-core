#!/usr/bin/env bash
set -euo pipefail

# Read-only macOS private beta surface smoke.
# It summarizes installed apps, CLIs, config files, Pith health, HTTP/API,
# Claude MCP logs, and known internal leak markers without printing secrets.

HOME_DIR="${HOME}"
PITH_BIN="${PITH_BIN:-$HOME_DIR/.pith/bin/pith}"
CLAUDE_LOG="$HOME_DIR/Library/Logs/Claude/mcp-server-pith.log"

section() {
    printf '\n== %s ==\n' "$1"
}

present_path() {
    local path="$1"
    if [[ -e "$path" ]]; then
        printf 'present: %s\n' "$path"
    else
        printf 'missing: %s\n' "$path"
    fi
}

section "Pith Health"
printf 'host: %s\n' "$(hostname 2>/dev/null || echo unknown)"
printf 'user: %s\n' "$(id -un 2>/dev/null || echo unknown)"
if [[ -x "$PITH_BIN" ]]; then
    printf 'pith_bin: %s\n' "$PITH_BIN"
    "$PITH_BIN" status 2>&1 | sed -n '1,12p'
else
    printf 'pith_bin: missing (%s)\n' "$PITH_BIN"
fi

section "Apps"
for app in \
    "/Applications/Claude.app" \
    "/Applications/Codex.app" \
    "/Applications/Visual Studio Code.app" \
    "/Applications/Cursor.app" \
    "/Applications/Windsurf.app" \
    "/Applications/Claude Code.app"; do
    present_path "$app"
done

section "CLIs"
for cmd in pith codex claude cursor windsurf code; do
    if command -v "$cmd" >/dev/null 2>&1; then
        printf 'present: %s -> %s\n' "$cmd" "$(command -v "$cmd")"
    else
        printf 'missing: %s\n' "$cmd"
    fi
done

section "Configs"
CONFIGS=(
    "$HOME_DIR/Library/Application Support/Claude/claude_desktop_config.json"
    "$HOME_DIR/.codex/config.toml"
    "$HOME_DIR/.codex/AGENTS.md"
    "$HOME_DIR/Library/Application Support/Code/User/mcp.json"
    "$HOME_DIR/.copilot/instructions/pith-cognitive-loop.instructions.md"
    "$HOME_DIR/.pith/pith-server/.mcp.json"
    "$HOME_DIR/.pith/pith-server/.vscode/mcp.json"
    "$HOME_DIR/.cursor/mcp.json"
    "$HOME_DIR/.codeium/windsurf/mcp_config.json"
    "$HOME_DIR/.claude.json"
)
for cfg in "${CONFIGS[@]}"; do
    present_path "$cfg"
done

section "HTTP/API"
if [[ -x "$PITH_BIN" ]]; then
    tmp="$(mktemp)"
    trap 'rm -f "$tmp"' EXIT
    if printf '{"message":"macOS surface smoke","previous_response":"[]","extracted_concepts_json":"[]"}\n' \
        | "$PITH_BIN" api conversation_turn --stdin-json >"$tmp" 2>&1; then
        last_line="$(awk 'NF { line=$0 } END { print line }' "$tmp")"
        if printf '%s' "$last_line" | grep -q '^{' && ! printf '%s' "$last_line" | grep -q '"error"[[:space:]]*:[[:space:]]*true'; then
            echo "api: ok"
        else
            echo "api: returned non-success JSON"
        fi
    else
        echo "api: failed"
        tail -n 3 "$tmp" | sed 's/^\(.\{0,240\}\).*/\1 .../'
    fi
else
    echo "api: skipped (pith binary missing)"
fi

section "Claude MCP Log"
if [[ -f "$CLAUDE_LOG" ]]; then
    echo "log: $CLAUDE_LOG"
    tail -n 20 "$CLAUDE_LOG" | sed 's/^\(.\{0,240\}\).*/\1 .../'
else
    echo "log: missing ($CLAUDE_LOG)"
fi

section "Leak Scan"
scan_targets=()
for path in \
    "$HOME_DIR/.pith/SYSTEM_PROMPT.md" \
    "$HOME_DIR/.codex/AGENTS.md" \
    "$HOME_DIR/.codex/config.toml" \
    "$HOME_DIR/Library/Application Support/Claude/claude_desktop_config.json" \
    "$HOME_DIR/Library/Application Support/Code/User/mcp.json" \
    "$HOME_DIR/.copilot/instructions/pith-cognitive-loop.instructions.md" \
    "$HOME_DIR/.pith/pith-server/.mcp.json" \
    "$HOME_DIR/.pith/pith-server/.vscode/mcp.json"; do
    [[ -e "$path" ]] && scan_targets+=("$path")
done

home_pattern=$(printf '%s\n' "$HOME_DIR" | sed 's/[][(){}.^$?*+|\/\\]/\\&/g')
if [[ ${#scan_targets[@]} -eq 0 ]]; then
    echo "leak_scan: skipped (no configured surfaces found)"
elif grep -InE "Rose|clawd|${home_pattern}" "${scan_targets[@]}" 2>/dev/null; then
    echo "leak_scan: findings"
else
    echo "leak_scan: clean"
fi

section "Manual Checks"
echo "claude_custom_instructions: user must confirm pith protocol was pasted"
echo "claude_fresh_conversation: user must confirm a Pith tool call appears after a fresh prompt"
echo "vscode_runtime: user must confirm Pith appears in MCP: List Servers and can list/invoke tools"
echo "vscode_copilot_instructions: user must confirm Chat Diagnostics loads ~/.copilot/instructions/pith-cognitive-loop.instructions.md"
