#!/usr/bin/env python3
"""
Pith — Multi-Client Configuration
Detects installed clients and writes Pith MCP/API configuration templates.

Usage:
    python3 scripts/configure_clients.py --server-path /path/to/pith_mcp.py --api-key <key>

Configuration templates: Claude Desktop, Claude Code, VS Code, Cursor, Windsurf, Cline, Codex.
Runtime support still requires verification in each client.
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:
    import tomllib
except ImportError:  # pragma: no cover - Python <3.11 fallback
    tomllib = None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.governance.runtime_install_guard import classify_runtime_path


# ============================================================
# Constants
# ============================================================

LEGACY_SERVER_NAMES = ["pith-mcp", "pith", "pith-mcp-wrapper"]


def _session_audit_script_path():
    override = os.environ.get("PITH_SESSION_AUDIT_SCRIPT")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".pith" / "scripts" / "session_isolation_audit.py"


def _normalize_api_key(api_key):
    """Reject empty API keys before touching client configs."""
    value = (api_key or "").strip()
    if not value:
        raise argparse.ArgumentTypeError(
            "PITH API key must be non-empty. Refusing to write client config with blank auth."
        )
    return value


def _load_api_key_from_file(path):
    """Load a key from either a .env file or a raw key file."""
    expanded = os.path.expanduser(path)
    if not os.path.isfile(expanded):
        raise argparse.ArgumentTypeError(f"API key source file not found: {expanded}")

    with open(expanded, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if expanded.endswith(".key"):
        return _normalize_api_key(content)

    for line in content.splitlines():
        if line.startswith("PITH_API_KEY="):
            return _normalize_api_key(line.split("=", 1)[1])

    raise argparse.ArgumentTypeError(
        f"No PITH_API_KEY entry found in API key source file: {expanded}"
    )


def _reject_archive_only_workspace(server_path, allow_noncanonical_server=False):
    """Refuse to point clients at archive-only preserve lanes unless override enabled."""
    if allow_noncanonical_server:
        return  # TIER4-004: Skip guard when user explicitly overrides
    audit_script = _session_audit_script_path()
    if not audit_script.is_file():
        return

    repo_path = Path(server_path).resolve().parent
    try:
        completed = subprocess.run(
            ["python3", str(audit_script), "--repo", str(repo_path), "--mode", "warn", "--json"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except Exception:
        return

    if completed.returncode not in (0, 1):
        return

    try:
        payload = json.loads(completed.stdout or "{}")
    except Exception:
        return

    classification = payload.get("classification")
    usage_policy = payload.get("usage_policy")
    if classification in {"archive_only_lane", "unregistered_worktree", "canonical_checkout"} or usage_policy == "archive_only":
        raise argparse.ArgumentTypeError(
            "Refusing to configure MCP clients with a non-runnable workspace target. "
            "Use ~/.pith/pith-server or a registered active session worktree instead."
        )


def _validate_server_path(server_path, allow_noncanonical_server=False):
    """Refuse to repoint global clients to a random checkout unless explicitly overridden."""
    resolved = os.path.realpath(os.path.abspath(server_path))
    canonical = os.path.realpath(os.path.expanduser("~/.pith/pith-server/pith_mcp.py"))

    if not os.path.isfile(resolved):
        raise argparse.ArgumentTypeError(f"Server path not found: {server_path}")

    _reject_archive_only_workspace(resolved, allow_noncanonical_server=allow_noncanonical_server)

    installed_bridge = os.path.realpath(os.path.expanduser("~/.pith/pith-server/pith_mcp.py"))
    if resolved == installed_bridge:
        report = classify_runtime_path(os.path.expanduser("~/.pith/pith-server"))
        if report["violation"]:
            raise argparse.ArgumentTypeError(
                "Refusing to configure global MCP clients with an unsafe installed runtime "
                f"({report['classification']}: {report['resolved_root']}). "
                "Repair ~/.pith/pith-server so it points to a standalone install or a "
                "runtime release worktree before configuring clients."
            )

    if allow_noncanonical_server or resolved == canonical:
        return resolved

    raise argparse.ArgumentTypeError(
        "Refusing to configure global MCP clients with non-canonical server path "
        f"{resolved}. Install Pith to ~/.pith/pith-server first, or pass "
        "--allow-noncanonical-server for an intentional override."
    )

# ============================================================
# Client Definitions
# ============================================================

def _detect_platform():
    s = platform.system().lower()
    if s == "darwin":
        return "macos"
    elif s == "linux":
        return "linux"
    elif s == "windows":
        return "windows"
    return "unknown"


def _expand(path, plat):
    """Expand ~ and %APPDATA%/%LOCALAPPDATA% in paths."""
    path = os.path.expanduser(path)
    if plat == "windows":
        if "%APPDATA%" in path:
            appdata = os.environ.get("APPDATA", "")
            path = path.replace("%APPDATA%", appdata)
        if "%LOCALAPPDATA%" in path:
            localappdata = os.environ.get("LOCALAPPDATA", "")
            path = path.replace("%LOCALAPPDATA%", localappdata)
    return path


# Each client: detection dir(s), config file path, JSON root key, extra fields
CLIENT_REGISTRY = {
    "claude_desktop": {
        "label": "Claude Desktop",
        "detect_dirs": {
            "macos":   "~/Library/Application Support/Claude",
            "linux":   "~/.config/Claude",
            "windows": "%APPDATA%/Claude",
        },
        "config_file": {
            "macos":   "~/Library/Application Support/Claude/claude_desktop_config.json",
            "linux":   "~/.config/Claude/claude_desktop_config.json",
            "windows": "%APPDATA%/Claude/claude_desktop_config.json",
        },
        "json_root": "mcpServers",
        "extra_fields": {},
    },
    "claude_code": {
        "label": "Claude Code",
        "detect_dirs": {
            "macos":   "~/.claude",
            "linux":   "~/.claude",
            "windows": "~/.claude",
        },
        "config_file": {
            "macos":   "~/.claude.json",
            "linux":   "~/.claude.json",
            "windows": "~/.claude.json",
        },
        "json_root": "mcpServers",
        "extra_fields": {},
    },
    "cursor": {
        "label": "Cursor",
        "detect_dirs": {
            "macos":   "~/.cursor",
            "linux":   "~/.cursor",
            "windows": "~/.cursor",
        },
        "config_file": {
            "macos":   "~/.cursor/mcp.json",
            "linux":   "~/.cursor/mcp.json",
            "windows": "~/.cursor/mcp.json",
        },
        "json_root": "mcpServers",
        "extra_fields": {},
    },
    "windsurf": {
        "label": "Windsurf",
        "detect_dirs": {
            "macos":   "~/.codeium/windsurf",
            "linux":   "~/.codeium/windsurf",
            "windows": "~/.codeium/windsurf",
        },
        "config_file": {
            "macos":   "~/.codeium/windsurf/mcp_config.json",
            "linux":   "~/.codeium/windsurf/mcp_config.json",
            "windows": "~/.codeium/windsurf/mcp_config.json",
        },
        "json_root": "mcpServers",
        "extra_fields": {},
    },
    "cline": {
        "label": "Cline",
        "detect_dirs": {
            "macos":   "~/Library/Application Support/Code/User/globalStorage/saoudrizwan.claude-dev",
            "linux":   "~/.config/Code/User/globalStorage/saoudrizwan.claude-dev",
            "windows": "%APPDATA%/Code/User/globalStorage/saoudrizwan.claude-dev",
        },
        "config_file": {
            "macos":   "~/Library/Application Support/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json",
            "linux":   "~/.config/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json",
            "windows": "%APPDATA%/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json",
        },
        "json_root": "mcpServers",
        "extra_fields": {
            "alwaysAllow": [
                "pith_conversation_turn",
                "pith_session_start",
                "pith_session_end",
                "pith_session_learn",
                "pith_checkpoint",
            ],
            "disabled": False,
        },
    },
}

# Codex is special — global TOML config, not JSON
CODEX_CONFIG = {
    "label": "Codex",
    "detect_dirs": {
        "macos": "~/.codex",
        "linux": "~/.codex",
        "windows": "~/.codex",
    },
    "config_file": {
        "macos": "~/.codex/config.toml",
        "linux": "~/.codex/config.toml",
        "windows": "~/.codex/config.toml",
    },
}

# VS Code is special — user/profile and project-level config, different JSON schema
VSCODE_CONFIG = {
    "label": "VS Code",
    "detect_dirs": {
        "macos":   "~/.vscode",
        "linux":   "~/.vscode",
        "windows": "~/.vscode",
    },
    "config_file": {
        "macos":   "~/Library/Application Support/Code/User/mcp.json",
        "linux":   "~/.config/Code/User/mcp.json",
        "windows": "%APPDATA%/Code/User/mcp.json",
    },
    "app_dirs": {
        "macos": [
            "/Applications/Visual Studio Code.app",
            "~/Applications/Visual Studio Code.app",
        ],
        "linux": [],
        "windows": [],
    },
}

VSCODE_USER_INSTRUCTIONS_FILE = "~/.copilot/instructions/pith-cognitive-loop.instructions.md"

CODEX_AGENTS_START = "<!-- PITH COGNITIVE LOOP: START -->"
CODEX_AGENTS_END = "<!-- PITH COGNITIVE LOOP: END -->"
CODEX_AGENTS_BODY = """# Pith Cognitive Loop

Pith is installed locally. For Codex, use the local HTTP/API command as the primary cognitive lifecycle path because Codex MCP stdio transport can restart or close between turns.

On every substantive user message, run `conversation_turn` before composing the response. Include a stable `origin_id` for this Codex thread/workspace. After the first successful call returns `resolved_session_id`, include that value as `session_id` on later lifecycle calls when available. Also include `previous_message`, `previous_response`, and `extracted_concepts_json` after the first exchange. Send JSON on stdin and parse the last non-empty output line as JSON because the wrapper may print a profile banner first:

```bash
~/.pith/bin/pith api conversation_turn --stdin-json
```

For checkpoints and closeout, use the matching lifecycle operations:

```bash
~/.pith/bin/pith api checkpoint --stdin-json
~/.pith/bin/pith api session_end --stdin-json
```

`pith api-fallback ...` remains as a legacy/recovery alias. Pith MCP tools with the `pith_` prefix may also be available in Codex and are useful for richer tool access when the MCP transport is healthy. Do not depend on MCP-only access for the core cognitive lifecycle.

For trivial exchanges, use `[]` for `extracted_concepts_json`. For substantive implementation or deployment work, extracted concepts must include concrete `verified: <check>` evidence.
"""


def _build_vscode_copilot_instructions():
    return """---
applyTo: "**"
description: "Use the local Pith cognitive loop in Agent mode when tools are enabled."
---
# Pith Cognitive Loop

Pith is installed locally. In VS Code Agent mode, retrieve Pith context before answering substantive user messages when Pith MCP tools are enabled for the request.

Preferred path: use direct MCP tools when they are available in the tools picker. Call `#tool:pith_conversation_turn` / `pith_conversation_turn` before composing a response. Include `previous_message`, `previous_response`, and `extracted_concepts_json` after the first exchange.

Fallback path: if direct Pith tools are unavailable or transport-broken and terminal commands are allowed, run `~/.pith/bin/pith api conversation_turn --stdin-json`. Send the JSON payload on stdin and parse the last non-empty output line as JSON because the wrapper may print a profile banner before the payload. Use `pith api-fallback` only as a recovery alias if `pith api` is unavailable.

For checkpoints and closeout, use `#tool:pith_checkpoint` / `pith_checkpoint` and `#tool:pith_session_end` / `pith_session_end` when direct tools are healthy. If not, use `~/.pith/bin/pith api checkpoint --stdin-json` and `~/.pith/bin/pith api session_end --stdin-json`.

Use `[]` for `extracted_concepts_json` when the exchange is trivial. For implementation, deployment, or operational decisions, extracted concepts must include concrete `verified: <check>` evidence.
"""



# ============================================================
# Core Logic
# ============================================================

def detect_clients(plat):
    """Detect which MCP clients are installed by checking config directories."""
    detected = {}
    for client_id, info in CLIENT_REGISTRY.items():
        detect_dir = info["detect_dirs"].get(plat)
        if detect_dir:
            expanded = _expand(detect_dir, plat)
            if os.path.isdir(expanded):
                detected[client_id] = info
    codex_dir = CODEX_CONFIG["detect_dirs"].get(plat)
    if codex_dir and os.path.isdir(_expand(codex_dir, plat)):
        detected["codex"] = CODEX_CONFIG
    # VS Code: check separately
    vscode_dir = VSCODE_CONFIG["detect_dirs"].get(plat)
    if vscode_dir and os.path.isdir(_expand(vscode_dir, plat)):
        detected["vscode"] = VSCODE_CONFIG
    else:
        for app_dir in VSCODE_CONFIG.get("app_dirs", {}).get(plat, []):
            if os.path.isdir(_expand(app_dir, plat)):
                detected["vscode"] = VSCODE_CONFIG
                break
    return detected


class ResolutionError(RuntimeError):
    """Raised when no usable Python interpreter can be located. MCP-PYTHON-RES-001."""
    pass


def _interpreter_has_mcp(python_path):
    """Return True iff the given python interpreter can import `mcp`."""
    try:
        result = subprocess.run(
            [python_path, "-c", "import mcp"],
            capture_output=True, timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def _detect_python_cmd(server_path):
    """Detect the best python3 command for running pith_mcp.py.

    Order of preference (MCP-PYTHON-RES-001):
    1. Venv python3 next to the server file (../venv/bin/python3 or ../.venv/...)
    2. Canonical Pith install venv ($PITH_HOME/venv/bin/python3, default ~/.pith/venv)
    3. System python3 on PATH — only if it imports `mcp` successfully
    4. Refuse to return: raise ResolutionError so caller surfaces a clear failure

    Never returns a bare 'python3' string — that defers the resolution to the
    MCP host's PATH at launch time, which on macOS+Homebrew is a moving target.
    """
    server_dir = os.path.dirname(os.path.abspath(server_path))
    parent_dir = os.path.dirname(server_dir)
    # FED-033: Check Unix (bin/python3) and Windows (Scripts/python.exe) venv layouts.
    import sys as _sys
    if _sys.platform == "win32":
        venv_candidates = [("Scripts", "python.exe")]
    else:
        venv_candidates = [("bin", "python3")]

    # Layer 1: adjacent venv (existing behavior + mcp validation)
    for base in [server_dir, parent_dir]:
        for venv_name in ["venv", ".venv"]:
            for subdir, exe in venv_candidates:
                candidate = os.path.join(base, venv_name, subdir, exe)
                if os.path.isfile(candidate) and _interpreter_has_mcp(candidate):
                    return candidate

    # Layer 2: canonical Pith install venv
    pith_home = os.environ.get("PITH_HOME", os.path.expanduser("~/.pith"))
    for venv_name in ["venv", ".venv"]:
        for subdir, exe in venv_candidates:
            canonical = os.path.join(pith_home, venv_name, subdir, exe)
            if os.path.isfile(canonical) and _interpreter_has_mcp(canonical):
                return canonical

    # Layer 3: system python — ONLY if it has mcp installed
    sys_py = shutil.which("python3") or shutil.which("python")
    if sys_py and _interpreter_has_mcp(sys_py):
        return sys_py

    # Layer 4: refuse — caller must surface
    raise ResolutionError(
        "No Python interpreter with the `mcp` package is reachable. "
        "Tried: adjacent venv, $PITH_HOME/venv, and system python3. "
        "Run scripts/install.sh or `pip install -r requirements.txt` into a venv."
    )


def _resolve_python_or_exit(server_path, python_cmd):
    """Resolve the python interpreter for an MCP entry, or exit cleanly on failure.

    MCP-PYTHON-RES-001 v1.3. Wraps `_detect_python_cmd` so callers never have to
    handle `ResolutionError` inline. On failure, writes a persistent diag file
    (for install.sh-invoked runs that redirect stderr to /dev/null) AND prints
    to stderr, then exits 2.

    Returns: resolved interpreter path (str) — guaranteed non-empty, usable.
    Exits: 2 on ResolutionError (POSIX "command-line usage error").
    """
    if python_cmd:
        return python_cmd
    try:
        return _detect_python_cmd(server_path)
    except ResolutionError as e:
        diag_path = os.path.join(
            os.environ.get("PITH_HOME", os.path.expanduser("~/.pith")),
            "diagnostics",
            "mcp_resolution_error.json",
        )
        try:
            os.makedirs(os.path.dirname(diag_path), exist_ok=True)
            with open(diag_path, "w") as f:
                json.dump({
                    "error": "mcp_python_resolution_failed",
                    "reason": str(e)[:500],
                    "server_path": server_path,
                    "remediation": "Run scripts/install.sh to create ~/.pith/venv with mcp installed.",
                    "doctor_command": "bash ~/.pith/scripts/pith_mcp_doctor.sh",
                }, f, indent=2)
        except OSError:
            pass
        print(f"ERROR: {e}", file=sys.stderr)
        print("Hint: run scripts/install.sh to create ~/.pith/venv with mcp installed.", file=sys.stderr)
        print(f"Diag: {diag_path}", file=sys.stderr)
        sys.exit(2)


def _build_standard_payload(server_path, api_key, python_cmd=None, extra_fields=None, api_url="http://localhost:8000"):
    """Build the standard mcpServers.pith entry."""
    cmd = _resolve_python_or_exit(server_path, python_cmd)
    entry = {
        "command": cmd,
        "args": [server_path],
        "env": {
            "PITH_API_KEY": api_key,
            "PITH_API_URL": api_url,
        },
    }
    if extra_fields:
        entry.update(extra_fields)
    return entry


def _backup_file(filepath):
    """Create timestamped backup of a config file."""
    if os.path.isfile(filepath):
        backup = f"{filepath}.backup.{int(time.time())}"
        shutil.copy2(filepath, backup)
        return backup
    return None


def _read_json(filepath):
    """Read JSON file, return empty dict if missing or invalid."""
    if not os.path.isfile(filepath):
        return {}
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def _write_json(filepath, data):
    """Write JSON file, creating parent dirs if needed."""
    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _validate_json(filepath):
    """Validate that a file contains valid JSON."""
    try:
        with open(filepath, "r") as f:
            json.load(f)
        return True
    except (json.JSONDecodeError, IOError):
        return False


def _escape_toml_string(value):
    """Escape string content for a TOML basic string."""
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _build_codex_toml_block(server_path, api_key, python_cmd=None, api_url="http://localhost:8000"):
    """Build the Codex TOML block for mcp_servers.pith."""
    entry = _build_standard_payload(server_path, api_key, python_cmd=python_cmd, api_url=api_url)
    return "\n".join(
        [
            "[mcp_servers.pith]",
            f'command = "{_escape_toml_string(entry["command"])}"',
            f'args = ["{_escape_toml_string(entry["args"][0])}"]',
            "[mcp_servers.pith.env]",
            f'PITH_API_KEY = "{_escape_toml_string(entry["env"]["PITH_API_KEY"])}"',
            f'PITH_API_URL = "{_escape_toml_string(entry["env"]["PITH_API_URL"])}"',
            "",
        ]
    )


def _find_owned_codex_blocks(text):
    """Return owned Codex MCP blocks for all legacy/current Pith server names."""
    lines = text.splitlines()
    blocks = []
    if not lines:
        return blocks

    headers = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            headers.append((idx, stripped))

    header_index = {header: idx for idx, header in headers}
    header_positions = [idx for idx, _ in headers]

    for name in LEGACY_SERVER_NAMES:
        server_header = f"[mcp_servers.{name}]"
        env_header = f"[mcp_servers.{name}.env]"
        if server_header not in header_index:
            continue
        server_idx = header_index[server_header]
        env_idx = header_index.get(env_header)
        if env_idx is not None and env_idx < server_idx:
            raise argparse.ArgumentTypeError(
                f"Invalid Codex config: env block appears before server block for {name}."
            )

        block_end = len(lines)
        anchor = env_idx if env_idx is not None else server_idx
        for next_idx in header_positions:
            if next_idx > anchor:
                block_end = next_idx
                break
        blocks.append(
            {
                "name": name,
                "start": server_idx,
                "end": block_end,
                "has_env": env_idx is not None,
            }
        )

    return sorted(blocks, key=lambda item: item["start"])


def _replace_or_append_codex_pith_block(existing_text, block_text):
    """Replace the owned Pith block or append a new one when safe."""
    stripped = existing_text.strip()
    blocks = _find_owned_codex_blocks(existing_text)
    if len(blocks) > 1:
        raise argparse.ArgumentTypeError(
            "Refusing to rewrite Codex config with multiple Pith MCP blocks. Clean up duplicates first."
        )

    if len(blocks) == 1:
        lines = existing_text.splitlines()
        block = blocks[0]
        before = lines[:block["start"]]
        after = lines[block["end"] :]
        rendered = "\n".join(before)
        if rendered and not rendered.endswith("\n"):
            rendered += "\n"
        rendered += block_text.rstrip() + "\n"
        if after:
            if not rendered.endswith("\n\n"):
                rendered += "\n"
            rendered += "\n".join(after).rstrip() + "\n"
        return rendered

    if not stripped:
        return block_text

    if tomllib is None:
        raise argparse.ArgumentTypeError(
            "Refusing to merge non-empty Codex TOML without parser support on this Python version. "
            "Upgrade to Python 3.11+ or add the Pith block manually once."
        )

    tomllib.loads(existing_text)
    suffix = existing_text if existing_text.endswith("\n") else existing_text + "\n"
    if not suffix.endswith("\n\n"):
        suffix += "\n"
    return suffix + block_text


def _validate_codex_config_text(text, api_url="http://localhost:8000"):
    """Validate rendered Codex config content for the owned Pith block."""
    blocks = _find_owned_codex_blocks(text)
    if len(blocks) != 1 or blocks[0]["name"] != "pith":
        return False
    if "pith_codex_bridge.py" in text:
        return False

    if tomllib is not None:
        try:
            payload = tomllib.loads(text)
        except Exception:
            return False
        try:
            entry = payload["mcp_servers"]["pith"]
            command = entry["command"]
            args = entry["args"]
            env = entry["env"]
        except Exception:
            return False
        return (
            isinstance(command, str)
            and isinstance(args, list)
            and len(args) == 1
            and isinstance(args[0], str)
            and args[0].endswith("pith_mcp.py")
            and isinstance(env, dict)
            and bool(env.get("PITH_API_KEY"))
            and env.get("PITH_API_URL") == api_url
        )

    return (
        "[mcp_servers.pith]" in text
        and "[mcp_servers.pith.env]" in text
        and f'PITH_API_URL = "{api_url}"' in text
        and "pith_mcp.py" in text
    )



def configure_standard_client(client_id, info, server_path, api_key, plat, dry_run=False, python_cmd=None, api_url="http://localhost:8000"):
    """Configure a standard mcpServers-based client (Claude Desktop, Code, Cursor, Windsurf, Cline)."""
    config_path = _expand(info["config_file"][plat], plat)
    label = info["label"]
    entry = _build_standard_payload(server_path, api_key, python_cmd=python_cmd, extra_fields=info.get("extra_fields"), api_url=api_url)

    if dry_run:
        return {"client": label, "action": "would_configure", "path": config_path}

    # Backup existing
    backup = _backup_file(config_path)

    # Read, merge, write
    config = _read_json(config_path)
    root_key = info["json_root"]
    if root_key not in config:
        config[root_key] = {}
    # Clean up legacy server names
    for legacy_name in LEGACY_SERVER_NAMES:
        config.get(root_key, {}).pop(legacy_name, None)
    config[root_key]["pith"] = entry
    _write_json(config_path, config)

    # Validate
    if not _validate_json(config_path):
        # Restore backup
        if backup and os.path.isfile(backup):
            shutil.copy2(backup, config_path)
        return {"client": label, "action": "error", "error": "JSON validation failed after write", "path": config_path}

    result = {"client": label, "action": "configured", "path": config_path}
    if backup:
        result["backup"] = backup
    return result


def configure_codex(server_path, api_key, plat, dry_run=False, python_cmd=None, api_url="http://localhost:8000"):
    """Configure Codex via ~/.codex/config.toml."""
    config_path = _expand(CODEX_CONFIG["config_file"][plat], plat)
    label = CODEX_CONFIG["label"]
    if dry_run:
        return {"client": label, "action": "would_configure", "path": config_path}

    backup = _backup_file(config_path)
    existing_text = ""
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as handle:
            existing_text = handle.read()

    block_text = _build_codex_toml_block(server_path, api_key, python_cmd=python_cmd, api_url=api_url)
    rendered = _replace_or_append_codex_pith_block(existing_text, block_text)
    parent = os.path.dirname(config_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as handle:
        handle.write(rendered)

    if not _validate_codex_config_text(rendered, api_url=api_url):
        if backup and os.path.isfile(backup):
            shutil.copy2(backup, config_path)
        return {
            "client": label,
            "action": "error",
            "error": "Codex TOML validation failed after write",
            "path": config_path,
        }

    result = {"client": label, "action": "configured", "path": config_path}
    if backup:
        result["backup"] = backup
    return result


def configure_codex_agents_instructions(plat, dry_run=False):
    """Install/update the Codex AGENTS.md managed Pith cognitive-loop block."""
    config_dir = _expand(CODEX_CONFIG["detect_dirs"][plat], plat)
    config_path = os.path.join(config_dir, "AGENTS.md")
    block = f"{CODEX_AGENTS_START}\n{CODEX_AGENTS_BODY}\n{CODEX_AGENTS_END}\n"

    if dry_run:
        return {
            "client": "Codex",
            "scope": "agents-instructions",
            "action": "would_configure",
            "path": config_path,
        }

    existing = ""
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as handle:
            existing = handle.read()

    if CODEX_AGENTS_START in existing and CODEX_AGENTS_END in existing:
        before, rest = existing.split(CODEX_AGENTS_START, 1)
        _, after = rest.split(CODEX_AGENTS_END, 1)
        new_text = before.rstrip() + "\n\n" + block + after.lstrip()
    else:
        new_text = (existing.rstrip() + "\n\n" if existing.strip() else "") + block

    if new_text == existing:
        return {
            "client": "Codex",
            "scope": "agents-instructions",
            "action": "unchanged",
            "path": config_path,
        }

    backup = _backup_file(config_path)
    parent = os.path.dirname(config_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as handle:
        handle.write(new_text)

    result = {
        "client": "Codex",
        "scope": "agents-instructions",
        "action": "configured",
        "path": config_path,
    }
    if backup:
        result["backup"] = backup
    return result


def configure_vscode(server_path, api_key, project_dir, dry_run=False, python_cmd=None, api_url="http://localhost:8000"):
    """Generate .vscode/mcp.json with VS Code's servers schema."""
    cmd = _resolve_python_or_exit(server_path, python_cmd)
    vscode_dir = os.path.join(project_dir, ".vscode")
    config_path = os.path.join(vscode_dir, "mcp.json")

    payload = {
        "servers": {
            "pith": {
                "type": "stdio",
                "command": cmd,
                "args": [server_path],
                "env": {
                    "PITH_API_KEY": api_key,
                    "PITH_API_URL": api_url,
                },
            }
        }
    }

    if dry_run:
        return {"file": ".vscode/mcp.json", "action": "would_generate", "path": config_path}

    backup = _backup_file(config_path)
    # Merge with existing if present
    existing = _read_json(config_path)
    if "servers" not in existing:
        existing["servers"] = {}
    # Clean up legacy server names
    for legacy_name in LEGACY_SERVER_NAMES:
        existing.get("servers", {}).pop(legacy_name, None)
    existing["servers"]["pith"] = payload["servers"]["pith"]
    _write_json(config_path, existing)

    result = {"file": ".vscode/mcp.json", "action": "generated", "path": config_path}
    if backup:
        result["backup"] = backup
    return result


def configure_vscode_user(server_path, api_key, plat, dry_run=False, python_cmd=None, api_url="http://localhost:8000"):
    """Generate VS Code user-profile mcp.json so Pith is available across workspaces."""
    cmd = _resolve_python_or_exit(server_path, python_cmd)
    config_template = VSCODE_CONFIG["config_file"].get(plat)
    if not config_template:
        raise ValueError(f"VS Code user config path is not defined for platform: {plat}")
    config_path = _expand(config_template, plat)

    payload = {
        "type": "stdio",
        "command": cmd,
        "args": [server_path],
        "env": {
            "PITH_API_KEY": api_key,
            "PITH_API_URL": api_url,
        },
    }

    if dry_run:
        return {"client": "VS Code", "scope": "user", "action": "would_generate", "path": config_path}

    backup = _backup_file(config_path)
    existing = _read_json(config_path)
    if "servers" not in existing:
        existing["servers"] = {}
    for legacy_name in LEGACY_SERVER_NAMES:
        existing.get("servers", {}).pop(legacy_name, None)
    existing["servers"]["pith"] = payload
    _write_json(config_path, existing)

    result = {"client": "VS Code", "scope": "user", "action": "generated", "path": config_path}
    if backup:
        result["backup"] = backup
    return result


def configure_vscode_user_instructions(plat, dry_run=False):
    """Generate a VS Code Copilot user instruction file for the Pith cognitive loop."""
    config_path = _expand(VSCODE_USER_INSTRUCTIONS_FILE, plat)
    content = _build_vscode_copilot_instructions()

    if dry_run:
        return {
            "client": "VS Code",
            "scope": "user-instructions",
            "action": "would_generate",
            "path": config_path,
        }

    existing = ""
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as handle:
            existing = handle.read()

    if existing == content:
        return {
            "client": "VS Code",
            "scope": "user-instructions",
            "action": "unchanged",
            "path": config_path,
        }

    backup = _backup_file(config_path)
    parent = os.path.dirname(config_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as handle:
        handle.write(content)

    result = {
        "client": "VS Code",
        "scope": "user-instructions",
        "action": "generated",
        "path": config_path,
    }
    if backup:
        result["backup"] = backup
    return result


def generate_project_mcp_json(server_path, api_key, project_dir, dry_run=False, python_cmd=None, api_url="http://localhost:8000"):
    """Generate .mcp.json (Claude Code project-level config) in project root."""
    cmd = _resolve_python_or_exit(server_path, python_cmd)
    config_path = os.path.join(project_dir, ".mcp.json")

    payload = {
        "mcpServers": {
            "pith": {
                "command": cmd,
                "args": [server_path],
                "env": {
                    "PITH_API_KEY": api_key,
                    "PITH_API_URL": api_url,
                },
            }
        }
    }

    if dry_run:
        return {"file": ".mcp.json", "action": "would_generate", "path": config_path}

    backup = _backup_file(config_path)
    existing = _read_json(config_path)
    if "mcpServers" not in existing:
        existing["mcpServers"] = {}
    # Clean up legacy server names
    for legacy_name in LEGACY_SERVER_NAMES:
        existing.get("mcpServers", {}).pop(legacy_name, None)
    existing["mcpServers"]["pith"] = payload["mcpServers"]["pith"]
    _write_json(config_path, existing)

    result = {"file": ".mcp.json", "action": "generated", "path": config_path}
    if backup:
        result["backup"] = backup
    return result


# ============================================================
# .gitignore Helper
# ============================================================

def update_gitignore(project_dir, dry_run=False):
    """Safely add MCP config entries to .gitignore."""
    gitignore_path = os.path.join(project_dir, ".gitignore")
    entries_to_add = [".mcp.json", ".vscode/mcp.json"]
    results = []

    existing_content = ""
    if os.path.isfile(gitignore_path):
        with open(gitignore_path, "r") as f:
            existing_content = f.read()

    lines = existing_content.strip().split("\n") if existing_content.strip() else []
    existing_entries = {line.strip() for line in lines}

    to_add = [e for e in entries_to_add if e not in existing_entries]

    if not to_add:
        return {"action": "unchanged", "path": gitignore_path, "reason": "entries already present"}

    if dry_run:
        return {"action": "would_add", "path": gitignore_path, "entries": to_add}

    # Check if files are already git-tracked
    warnings = []
    for entry in to_add:
        full_path = os.path.join(project_dir, entry)
        if os.path.isfile(full_path):
            # Check git tracking (non-fatal if git not available)
            try:
                import subprocess
                result = subprocess.run(
                    ["git", "ls-files", "--error-unmatch", entry],
                    cwd=project_dir, capture_output=True, text=True
                )
                if result.returncode == 0:
                    warnings.append(f"{entry} is git-tracked; run 'git rm --cached {entry}' to untrack")
            except (FileNotFoundError, OSError):
                pass  # git not available, skip check

    # Append entries
    with open(gitignore_path, "a") as f:
        if existing_content and not existing_content.endswith("\n"):
            f.write("\n")
        f.write("\n# Pith MCP config files (auto-generated, contain API keys)\n")
        for entry in to_add:
            f.write(f"{entry}\n")

    result = {"action": "updated", "path": gitignore_path, "added": to_add}
    if warnings:
        result["warnings"] = warnings
    return result


# ============================================================
# Main Orchestration
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Write Pith MCP configuration templates for detected clients",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Configuration templates: Claude Desktop, Claude Code, Cursor, Windsurf, "
            "Cline, Codex, VS Code. Runtime support must be verified in each client."
        )
    )
    parser.add_argument("--server-path", required=True,
                        help="Absolute path to pith_mcp.py (MCP bridge)")
    parser.add_argument("--api-key", default=None, type=_normalize_api_key,
                        help="Pith API key for authentication")
    parser.add_argument("--source-key-from-file", nargs="?", const="~/.pith/.env",
                        default="~/.pith/.env",
                        help="Load PITH_API_KEY from file when --api-key is not passed (default: ~/.pith/.env)")
    parser.add_argument("--python-cmd", default=None,
                        help="Python interpreter path (default: auto-detect venv or system python3)")
    parser.add_argument("--api-url", default=os.environ.get("PITH_API_URL", "http://localhost:8000"),
                        help="Pith HTTP API URL to write into MCP client env (default: $PITH_API_URL or http://localhost:8000)")
    parser.add_argument("--project-dir", default=None,
                        help="Project directory for .mcp.json and .vscode/mcp.json (default: script parent)")
    parser.add_argument("--platform", default=None, choices=["macos", "linux", "windows"],
                        help="Override platform detection")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be configured without making changes")
    parser.add_argument("--allow-noncanonical-server", action="store_true",
                        help="Allow configuring clients against a non-~/.pith/pith-server bridge")
    parser.add_argument("--json", action="store_true", dest="json_output",
                        help="Output results as JSON (for install.sh consumption)")
    parser.add_argument("--skip-gitignore", action="store_true",
                        help="Skip .gitignore update")
    parser.add_argument("--skip-project", action="store_true",
                        help="Skip project-level configs (.mcp.json, .vscode/mcp.json)")
    parser.add_argument(
        "--clients",
        default="all",
        help=(
            "Comma-separated client IDs to configure. Use all, none, or any of: "
            "claude_desktop, claude_code, codex, vscode, cursor, windsurf, cline, project."
        ),
    )

    args = parser.parse_args()

    plat = args.platform or _detect_platform()
    if plat == "unknown":
        print("ERROR: Could not detect platform. Use --platform flag.", file=sys.stderr)
        sys.exit(1)

    project_dir = args.project_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    server_path = os.path.abspath(args.server_path)

    try:
        server_path = _validate_server_path(
            server_path,
            allow_noncanonical_server=args.allow_noncanonical_server,
        )
    except argparse.ArgumentTypeError as e:
        parser.error(str(e))

    try:
        api_key = args.api_key or _load_api_key_from_file(args.source_key_from_file)
    except argparse.ArgumentTypeError as e:
        parser.error(str(e))
    api_url = args.api_url

    allowed_client_ids = set(CLIENT_REGISTRY) | {"codex", "vscode", "project", "all", "none"}
    requested = {item.strip() for item in args.clients.split(",") if item.strip()}
    if not requested:
        requested = {"all"}
    unknown = sorted(requested - allowed_client_ids)
    if unknown:
        parser.error(f"Unknown --clients value(s): {', '.join(unknown)}")
    if "all" in requested and len(requested) > 1:
        parser.error("--clients=all cannot be combined with specific client IDs")
    if "none" in requested and len(requested) > 1:
        parser.error("--clients=none cannot be combined with specific client IDs")

    configure_all = "all" in requested
    configure_none = "none" in requested

    def wants(client_id):
        return configure_all or (not configure_none and client_id in requested)

    # --- Phase 1: Detect installed clients ---
    detected = detect_clients(plat)
    # Remove vscode from global detection (it's project-level only)
    codex_detected = "codex" in detected and wants("codex")
    vscode_detected = "vscode" in detected and wants("vscode")
    detected.pop("codex", None)
    detected.pop("vscode", None)
    detected = {
        client_id: info
        for client_id, info in detected.items()
        if wants(client_id)
    }

    # --- Phase 2: Configure global clients ---
    results = {"detected": list(detected.keys()), "configured": [], "skipped": [], "errors": []}

    if codex_detected:
        results["detected"].append("codex")
    if vscode_detected:
        results["detected"].append("vscode")

    for client_id, info in detected.items():
        try:
            r = configure_standard_client(client_id, info, server_path, api_key, plat, args.dry_run, python_cmd=args.python_cmd, api_url=api_url)
            if r.get("action") == "error":
                results["errors"].append(r)
            else:
                results["configured"].append(r)
        except Exception as e:
            results["errors"].append({"client": info["label"], "action": "error", "error": str(e)})

    if codex_detected:
        try:
            r = configure_codex(server_path, api_key, plat, args.dry_run, python_cmd=args.python_cmd, api_url=api_url)
            if r.get("action") == "error":
                results["errors"].append(r)
            else:
                results["configured"].append(r)
            agents_result = configure_codex_agents_instructions(plat, args.dry_run)
            results["configured"].append(agents_result)
        except Exception as e:
            results["errors"].append({"client": CODEX_CONFIG["label"], "action": "error", "error": str(e)})

    # --- Phase 3: Project-level configs ---
    project_selected = wants("project")
    if not args.skip_project and project_selected:
        try:
            r = generate_project_mcp_json(server_path, api_key, project_dir, args.dry_run, python_cmd=args.python_cmd, api_url=api_url)
            results["configured"].append(r)
        except Exception as e:
            results["errors"].append({"file": ".mcp.json", "action": "error", "error": str(e)})

    if not args.skip_project and project_selected and vscode_detected:
        try:
            r = configure_vscode(server_path, api_key, project_dir, args.dry_run, python_cmd=args.python_cmd, api_url=api_url)
            results["configured"].append(r)
        except Exception as e:
            results["errors"].append({"file": ".vscode/mcp.json", "action": "error", "error": str(e)})

    if vscode_detected:
        try:
            r = configure_vscode_user(server_path, api_key, plat, args.dry_run, python_cmd=args.python_cmd, api_url=api_url)
            results["configured"].append(r)
        except Exception as e:
            results["errors"].append({"client": "VS Code", "scope": "user", "action": "error", "error": str(e)})
        try:
            r = configure_vscode_user_instructions(plat, args.dry_run)
            results["configured"].append(r)
        except Exception as e:
            results["errors"].append({"client": "VS Code", "scope": "user-instructions", "action": "error", "error": str(e)})

    # --- Phase 4: .gitignore ---
    if not args.skip_gitignore and not args.skip_project and project_selected:
        try:
            r = update_gitignore(project_dir, args.dry_run)
            results["gitignore"] = r
        except Exception as e:
            results["gitignore"] = {"action": "error", "error": str(e)}

    # --- Output ---
    if args.json_output:
        print(json.dumps(results, indent=2))
    else:
        # Human-readable output
        print(f"\n{'='*50}")
        print(f"Pith Client Configuration")
        print(f"{'='*50}")
        print(f"Platform: {plat}")
        print(f"Server:   {server_path}")
        print(f"Detected: {', '.join(results['detected']) or 'none'}")
        print(f"{'='*50}\n")

        if args.dry_run:
            print("[DRY RUN] No changes made.\n")

        for r in results["configured"]:
            label = r.get("client") or r.get("file")
            action = r.get("action", "unknown")
            path = r.get("path", "")
            icon = "✅" if "configure" in action or "generate" in action else "📋"
            print(f"  {icon} {label}: {action}")
            if path:
                print(f"     → {path}")
            if r.get("backup"):
                print(f"     📦 Backup: {r['backup']}")

        for r in results["errors"]:
            label = r.get("client") or r.get("file")
            print(f"  ❌ {label}: {r.get('error', 'unknown error')}")

        if "gitignore" in results:
            gi = results["gitignore"]
            if gi.get("action") == "updated":
                print(f"\n  📝 .gitignore updated: added {', '.join(gi['added'])}")
            elif gi.get("action") == "unchanged":
                print(f"\n  📝 .gitignore: already up to date")
            if gi.get("warnings"):
                for w in gi["warnings"]:
                    print(f"     ⚠️  {w}")

        print(f"\n{'='*50}")
        total = len(results["configured"])
        errs = len(results["errors"])
        print(f"Done: {total} configured, {errs} errors")

        if not args.dry_run:
            print("\n⚠️  Pith MCP config is managed by install.sh.")
            print("   Don't use 'claude mcp add pith' separately.")
        print()

    sys.exit(1 if results["errors"] else 0)


if __name__ == "__main__":
    main()
