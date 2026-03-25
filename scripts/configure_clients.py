#!/usr/bin/env python3
"""
Pith — Multi-Client MCP Configuration
Detects installed MCP clients and configures each to use the Pith server.

Usage:
    python3 scripts/configure_clients.py --server-path /path/to/pith_mcp.py --api-key <key>

Supports: Claude Desktop, Claude Code, VS Code, Cursor, Windsurf, Cline
"""

import argparse
import json
import os
import platform
import shutil
import sys
import time


# ============================================================
# Constants
# ============================================================

LEGACY_SERVER_NAMES = ["pith-mcp", "pith", "pith-mcp-wrapper"]

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

# VS Code is special — project-level config only, different JSON schema
VSCODE_CONFIG = {
    "label": "VS Code",
    "detect_dirs": {
        "macos":   "~/.vscode",
        "linux":   "~/.vscode",
        "windows": "~/.vscode",
    },
}



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
    # VS Code: check separately
    vscode_dir = VSCODE_CONFIG["detect_dirs"].get(plat)
    if vscode_dir and os.path.isdir(_expand(vscode_dir, plat)):
        detected["vscode"] = VSCODE_CONFIG
    return detected


def _detect_python_cmd(server_path):
    """Detect the best python3 command for running pith_mcp.py.

    Checks (in order):
    1. Venv python3 next to the server file (../venv/bin/python3)
    2. System python3 on PATH
    Falls back to 'python3' if nothing else found.
    """
    server_dir = os.path.dirname(os.path.abspath(server_path))
    parent_dir = os.path.dirname(server_dir)
    # FED-033: Check Unix (bin/python3) and Windows (Scripts/python.exe) venv layouts.
    # Windows venvs use Scripts/python.exe; Unix venvs use bin/python3.
    import sys as _sys
    if _sys.platform == "win32":
        venv_candidates = [("Scripts", "python.exe")]
    else:
        venv_candidates = [("bin", "python3")]
    for base in [server_dir, parent_dir]:
        for venv_name in ["venv", ".venv"]:
            for subdir, exe in venv_candidates:
                candidate = os.path.join(base, venv_name, subdir, exe)
                if os.path.isfile(candidate):
                    return candidate
    # Fallback to system python3
    return shutil.which("python3") or shutil.which("python") or "python3"


def _build_standard_payload(server_path, api_key, python_cmd=None, extra_fields=None):
    """Build the standard mcpServers.pith entry."""
    cmd = python_cmd or _detect_python_cmd(server_path)
    entry = {
        "command": cmd,
        "args": [server_path],
        "env": {
            "PITH_API_KEY": api_key,
            "PITH_API_URL": "http://localhost:8000",
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



def configure_standard_client(client_id, info, server_path, api_key, plat, dry_run=False, python_cmd=None):
    """Configure a standard mcpServers-based client (Claude Desktop, Code, Cursor, Windsurf, Cline)."""
    config_path = _expand(info["config_file"][plat], plat)
    label = info["label"]
    entry = _build_standard_payload(server_path, api_key, python_cmd=python_cmd, extra_fields=info.get("extra_fields"))

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


def configure_vscode(server_path, api_key, project_dir, dry_run=False, python_cmd=None):
    """Generate .vscode/mcp.json with VS Code's servers schema."""
    cmd = python_cmd or _detect_python_cmd(server_path)
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
                    "PITH_API_URL": "http://localhost:8000",
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


def generate_project_mcp_json(server_path, api_key, project_dir, dry_run=False, python_cmd=None):
    """Generate .mcp.json (Claude Code project-level config) in project root."""
    cmd = python_cmd or _detect_python_cmd(server_path)
    config_path = os.path.join(project_dir, ".mcp.json")

    payload = {
        "mcpServers": {
            "pith": {
                "command": cmd,
                "args": [server_path],
                "env": {
                    "PITH_API_KEY": api_key,
                    "PITH_API_URL": "http://localhost:8000",
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
        description="Configure MCP clients to use the Pith server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Supported clients: Claude Desktop, Claude Code, Cursor, Windsurf, Cline, VS Code"
    )
    parser.add_argument("--server-path", required=True,
                        help="Absolute path to pith_mcp.py (MCP bridge)")
    parser.add_argument("--api-key", required=True,
                        help="Pith API key for authentication")
    parser.add_argument("--python-cmd", default=None,
                        help="Python interpreter path (default: auto-detect venv or system python3)")
    parser.add_argument("--project-dir", default=None,
                        help="Project directory for .mcp.json and .vscode/mcp.json (default: script parent)")
    parser.add_argument("--platform", default=None, choices=["macos", "linux", "windows"],
                        help="Override platform detection")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be configured without making changes")
    parser.add_argument("--json", action="store_true", dest="json_output",
                        help="Output results as JSON (for install.sh consumption)")
    parser.add_argument("--skip-gitignore", action="store_true",
                        help="Skip .gitignore update")
    parser.add_argument("--skip-project", action="store_true",
                        help="Skip project-level configs (.mcp.json, .vscode/mcp.json)")

    args = parser.parse_args()

    plat = args.platform or _detect_platform()
    if plat == "unknown":
        print("ERROR: Could not detect platform. Use --platform flag.", file=sys.stderr)
        sys.exit(1)

    project_dir = args.project_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    server_path = os.path.abspath(args.server_path)

    # --- Phase 1: Detect installed clients ---
    detected = detect_clients(plat)
    # Remove vscode from global detection (it's project-level only)
    vscode_detected = "vscode" in detected
    detected.pop("vscode", None)

    # --- Phase 2: Configure global clients ---
    results = {"detected": list(detected.keys()), "configured": [], "skipped": [], "errors": []}

    if vscode_detected:
        results["detected"].append("vscode")

    for client_id, info in detected.items():
        try:
            r = configure_standard_client(client_id, info, server_path, args.api_key, plat, args.dry_run, python_cmd=args.python_cmd)
            if r.get("action") == "error":
                results["errors"].append(r)
            else:
                results["configured"].append(r)
        except Exception as e:
            results["errors"].append({"client": info["label"], "action": "error", "error": str(e)})

    # --- Phase 3: Project-level configs ---
    if not args.skip_project:
        try:
            r = generate_project_mcp_json(server_path, args.api_key, project_dir, args.dry_run, python_cmd=args.python_cmd)
            results["configured"].append(r)
        except Exception as e:
            results["errors"].append({"file": ".mcp.json", "action": "error", "error": str(e)})

        if vscode_detected:
            try:
                r = configure_vscode(server_path, args.api_key, project_dir, args.dry_run, python_cmd=args.python_cmd)
                results["configured"].append(r)
            except Exception as e:
                results["errors"].append({"file": ".vscode/mcp.json", "action": "error", "error": str(e)})

    # --- Phase 4: .gitignore ---
    if not args.skip_gitignore and not args.skip_project:
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
        print(f"Pith MCP Client Configuration")
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
