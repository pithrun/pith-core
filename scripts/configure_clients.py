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
import textwrap
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
PITH_CLAUDE_CODE_HOOK_SCRIPT_NAME = "claude-code-pith-lifecycle.py"
PITH_CLAUDE_CODE_HOOK_VERSION = "claude-code-pith-lifecycle.v5"


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

CLIENT_SURFACE_IDS = {
    "claude_desktop": "claude_desktop_mcp",
    "claude_code": "claude_code",
    "cursor": "cursor_mcp",
    "windsurf": "windsurf_mcp",
    "cline": "cline_mcp",
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
                    "doctor_command": "bash ~/.pith/pith-server/scripts/pith_mcp_doctor.sh",
                }, f, indent=2)
        except OSError:
            pass
        print(f"ERROR: {e}", file=sys.stderr)
        print("Hint: run scripts/install.sh to create ~/.pith/venv with mcp installed.", file=sys.stderr)
        print(f"Diag: {diag_path}", file=sys.stderr)
        sys.exit(2)


def _build_standard_payload(
    server_path,
    api_key,
    python_cmd=None,
    extra_fields=None,
    api_url="http://localhost:8000",
    surface_id=None,
):
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
    if surface_id:
        entry["env"]["PITH_SURFACE_ID"] = surface_id
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


def _pith_home_path():
    return Path(os.environ.get("PITH_HOME", "~/.pith")).expanduser()


def _claude_code_settings_path(plat):
    # Claude Code uses ~/.claude/settings.json for user-scope hooks on macOS,
    # Linux, and Windows. On Windows, Path.home() handles the user profile root.
    return Path.home() / ".claude" / "settings.json"


def _claude_code_hook_script_content():
    return textwrap.dedent(
        r'''
        #!/usr/bin/env python3
        """Pith lifecycle hook for Claude Code.

        This script is generated by Pith's installer. It is intentionally
        fail-soft: lifecycle sync failures are logged locally and never block
        Claude Code's normal prompt/response flow.
        """

        import hashlib
        import json
        import os
        import re
        import subprocess
        import sys
        import time
        from pathlib import Path

        PITH_HOME = Path(__file__).resolve().parents[1]
        PITH_CLI = PITH_HOME / "bin" / "pith"
        HOOK_VERSION = "claude-code-pith-lifecycle.v5"
        STATE_DIR = PITH_HOME / "cache" / "claude-code-lifecycle"
        LOG_PATH = PITH_HOME / "logs" / "claude-code-lifecycle.log"
        MIN_LEARNABLE_RESPONSE_CHARS = 30
        MAX_STOP_LEARN_SUMMARY_CHARS = 480
        RETRY_MAX_ITEMS = 20
        RETRY_MAX_ATTEMPTS = 3
        RETRY_TTL_SECONDS = 24 * 60 * 60
        RETRY_QUEUE_KEY = "retry_queue"


        def _log(message):
            try:
                LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
                with LOG_PATH.open("a", encoding="utf-8") as handle:
                    handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}\n")
            except OSError:
                pass


        def _load_input():
            try:
                return json.load(sys.stdin)
            except Exception as exc:
                _log(f"invalid hook input: {exc}")
                return {}


        def _safe_key(value):
            return hashlib.sha256((value or "unknown").encode("utf-8")).hexdigest()[:24]


        def _sha256_text(value):
            return hashlib.sha256((value or "").encode("utf-8")).hexdigest()


        def _next_turn_seq(state):
            try:
                current = int(state.get("hook_turn_seq") or 0)
            except (TypeError, ValueError):
                current = 0
            current += 1
            state["hook_turn_seq"] = current
            return current


        def _retry_queue(state):
            queue = state.get(RETRY_QUEUE_KEY)
            return queue if isinstance(queue, list) else []


        def _set_retry_queue(state, queue):
            state[RETRY_QUEUE_KEY] = queue[-RETRY_MAX_ITEMS:]


        def _prune_retry_queue(state, now=None):
            now = now or time.time()
            kept = []
            for item in _retry_queue(state):
                try:
                    attempts = int(item.get("attempts") or 0)
                    first_seen = float(item.get("first_seen_at") or now)
                except (AttributeError, TypeError, ValueError):
                    continue
                if attempts >= RETRY_MAX_ATTEMPTS:
                    _log(f"backstop_retry_dropped attempts request_id={item.get('request_id')}")
                    continue
                if now - first_seen > RETRY_TTL_SECONDS:
                    _log(f"backstop_retry_dropped ttl request_id={item.get('request_id')}")
                    continue
                kept.append(item)
            _set_retry_queue(state, kept)


        def _origin_for(event):
            cwd = event.get("cwd") or ""
            digest = _safe_key(cwd)
            return f"claude_code:{digest}"


        def _workspace_id_for(event):
            cwd = event.get("cwd") or ""
            return "cwd:" + _safe_key(cwd)


        def _state_path(event):
            session_id = event.get("session_id") or event.get("cwd") or "unknown"
            return STATE_DIR / f"{_safe_key(session_id)}.json"


        def _read_state(path):
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}


        def _write_state(path, state):
            try:
                STATE_DIR.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
                path.chmod(0o600)
            except OSError as exc:
                _log(f"state write failed: {exc}")


        def _last_json_line(text):
            for line in reversed((text or "").splitlines()):
                stripped = line.strip()
                if stripped.startswith("{") and stripped.endswith("}"):
                    return stripped
            return ""


        def _call_pith_result(operation, payload, timeout=3.0):
            if not PITH_CLI.exists():
                error = f"pith cli missing: {PITH_CLI}"
                _log(error)
                return None, error
            env = os.environ.copy()
            env["PITH_HOME"] = str(PITH_HOME)
            try:
                result = subprocess.run(
                    [str(PITH_CLI), "api", operation, "--stdin-json"],
                    input=json.dumps(payload),
                    text=True,
                    capture_output=True,
                    timeout=timeout,
                    env=env,
                    check=False,
                )
            except Exception as exc:
                error = f"{operation} call failed: {exc}"
                _log(error)
                return None, error
            line = _last_json_line(result.stdout)
            if result.returncode != 0 and not line:
                error = f"{operation} exited {result.returncode}: {(result.stderr or result.stdout)[:500]}"
                _log(error)
                return None, error
            if not line:
                error = f"{operation} returned no json payload"
                _log(error)
                return None, error
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                error = f"{operation} json parse failed: {exc}"
                _log(error)
                return None, error
            if result.returncode != 0:
                error = f"{operation} exited {result.returncode} after json payload"
                _log(error)
                return parsed, error
            return parsed, None


        def _call_pith(operation, payload, timeout=3.0):
            response, _error = _call_pith_result(operation, payload, timeout=timeout)
            return response


        def _format_model_visible_binding(state):
            session_id = state.get("pith_session_id") or state.get("pre_response_ct_session_id")
            origin_id = state.get("pre_response_ct_origin_id")
            workspace_id = state.get("pre_response_ct_workspace_id")
            if not session_id or not origin_id:
                return []
            payload = {
                "session_id": session_id,
                "origin_id": origin_id,
                "surface_id": "claude_code",
                "platform_hint": "claude-code",
                "workspace_id": workspace_id,
                "context_delivery_mode": "mcp_tool_call",
                "surface_lifecycle_version": "1.0",
                "response_mode": "compact",
            }
            return [
                "Model-visible Pith lifecycle binding:",
                json.dumps(payload, sort_keys=True, separators=(",", ":")),
                "When calling mcp__pith__pith_conversation_turn for this turn, reuse these exact fields.",
                "For lifecycle evidence reports, run ~/.pith/bin/pith api lifecycle_status --stdin-json with the same session_id, origin_id, and surface_id.",
                "For cross-surface source coverage evidence, run ~/.pith/bin/pith api surface_activity --stdin-json with requested_surfaces such as claude_code,codex_local_api,local_api_cli; treat this as coverage evidence, not a semantic summary.",
            ]


        def _format_context(response, state):
            if not isinstance(response, dict):
                return ""
            concepts = response.get("activated_concepts") or []
            orientation = response.get("orientation_summary")
            has_context_payload = bool(concepts or orientation)
            if has_context_payload:
                lines = ["Pith lifecycle: context delivered before this response."]
                lines.append("Context payload: delivered")
            else:
                lines = [
                    "Pith lifecycle: turn registered before this response; no retrieved context was delivered.",
                    "Context payload: registration_only",
                    "Do not claim Pith context was retrieved for this turn.",
                ]
            resolved = response.get("resolved_session_id")
            if resolved:
                lines.append(f"Session: {resolved}")
            lines.extend(_format_model_visible_binding(state))
            if "is_first_call" in response:
                lines.append(f"First call: {bool(response.get('is_first_call'))}")
            if orientation:
                lines.append(f"Orientation: {orientation}")
            working_context = response.get("working_context") or {}
            checkpoint = working_context.get("checkpoint") or {}
            resume_hint = checkpoint.get("resume_hint")
            if resume_hint:
                lines.append(f"Checkpoint: {resume_hint}")
            if concepts:
                lines.append("Relevant memory:")
                for concept in concepts[:6]:
                    summary = str(concept.get("summary") or "").strip()
                    if summary:
                        lines.append(f"- {summary[:400]}")
            text = "\n".join(lines).strip()
            return text[:4000]


        def _format_degraded_context(reason):
            safe_reason = str(reason or "unknown")[:200]
            return (
                "Pith lifecycle: degraded before this response.\n"
                f"Reason: {safe_reason}\n"
                "Do not claim Pith context was retrieved for this turn."
            )[:500]


        def _emit_user_prompt_context(text):
            if not text:
                return
            payload = {
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": text,
                }
            }
            print(json.dumps(payload, separators=(",", ":")))


        PITH_CT_TOOL_NAME = "mcp__pith__pith_conversation_turn"


        def _t0_request_id(event, state, prompt):
            raw = "|".join([
                str(event.get("session_id") or ""),
                str(event.get("cwd") or ""),
                str(state.get("hook_turn_seq") or 0),
                str(prompt or ""),
                str(state.get("previous_message") or ""),
                _sha256_text(state.get("previous_response") or ""),
            ])
            return "claude-code-t0:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:48]


        def _build_t0_payload(event, state, prompt):
            payload = {
                "origin_id": _origin_for(event),
                "message": prompt,
                "conversation_context": "Claude Code T0 lifecycle hook (UserPromptSubmit pre-response)",
                "request_id": _t0_request_id(event, state, prompt),
                "platform_hint": "claude-code",
                "surface_id": "claude_code",
                "workspace_id": _workspace_id_for(event),
                "context_delivery_mode": "hook_additional_context",
                "surface_lifecycle_version": "1.0",
                "max_concepts": 1,
                "include_verbatim": False,
            }
            if state.get("pith_session_id"):
                payload["session_id"] = state["pith_session_id"]
            return payload


        def _response_detail(response):
            if not isinstance(response, dict):
                return {}
            detail = response.get("detail")
            return detail if isinstance(detail, dict) else {}


        def _is_stale_session_error(response, error, cached_session_id):
            detail = _response_detail(response)
            response_error = str(detail.get("error") or response.get("error") if isinstance(response, dict) else "")
            response_reason = str(detail.get("reason") or "")
            response_session_id = str(detail.get("session_id") or "")
            text = " ".join([
                response_error,
                response_reason,
                response_session_id,
                str(error or ""),
                json.dumps(response)[:500] if isinstance(response, dict) else "",
            ]).lower()
            if "stale_session_id" in text:
                return True
            if "invalid_session_id" not in text:
                return False
            return bool(cached_session_id) and (
                response_session_id == cached_session_id
                or cached_session_id.lower() in text
            )


        def _handle_user_prompt(event, state_path, state):
            prompt = event.get("prompt") or ""
            _next_turn_seq(state)
            state["model_fired_ct"] = False
            state["manual_ct_fired"] = False
            state["pending_prompt"] = prompt
            state["last_prompt_at"] = time.time()
            _prune_retry_queue(state)
            if os.environ.get("PITH_CLAUDE_CODE_T0_LIFECYCLE", "1").lower() in ("0", "false", "off", "no"):
                state["pre_response_ct_status"] = "skipped_disabled"
                _write_state(state_path, state)
                return
            payload = _build_t0_payload(event, state, prompt)
            response, error = _call_pith_result("conversation_turn", payload, timeout=4.0)
            cached_session_id = payload.get("session_id")
            if cached_session_id and _is_stale_session_error(response, error, cached_session_id):
                state["pre_response_ct_stale_retry"] = True
                state["pre_response_ct_stale_session_id"] = cached_session_id
                state.pop("pith_session_id", None)
                payload = _build_t0_payload(event, state, prompt)
                response, error = _call_pith_result("conversation_turn", payload, timeout=4.0)
            if isinstance(response, dict) and response.get("resolved_session_id"):
                state["pith_session_id"] = response["resolved_session_id"]
                state["pre_response_ct_request_id"] = payload.get("request_id")
                state["pre_response_ct_session_id"] = response["resolved_session_id"]
                state["pre_response_ct_origin_id"] = payload.get("origin_id")
                state["pre_response_ct_workspace_id"] = payload.get("workspace_id")
                state["pre_response_ct_surface_id"] = payload.get("surface_id")
                state["pre_response_ct_status"] = "ok"
                state["model_fired_ct"] = True
                if error:
                    state["pre_response_ct_warning"] = error
                _emit_user_prompt_context(_format_context(response, state))
            else:
                reason = error or "missing_resolved_session_id"
                state["pre_response_ct_request_id"] = payload.get("request_id")
                state["pre_response_ct_status"] = "degraded_" + str(reason).split(":", 1)[0].replace(" ", "_")[:80]
                _emit_user_prompt_context(_format_degraded_context(reason))
            _write_state(state_path, state)


        def _json_loads_maybe(value):
            if not isinstance(value, str):
                return value
            text = value.strip()
            if not text:
                return value
            try:
                return json.loads(text)
            except (TypeError, ValueError):
                return value


        def _walk_first_key(value, key):
            if isinstance(value, dict):
                if key in value:
                    return value.get(key)
                for child in value.values():
                    found = _walk_first_key(child, key)
                    if found is not None:
                        return found
            elif isinstance(value, list):
                for child in value:
                    found = _walk_first_key(child, key)
                    if found is not None:
                        return found
            return None


        def _extract_response_text(value):
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                if isinstance(value.get("text"), str):
                    return value["text"]
                content = value.get("content")
                if isinstance(content, list):
                    chunks = []
                    for item in content:
                        text = _extract_response_text(item)
                        if text:
                            chunks.append(text)
                    return "\n".join(chunks)
            if isinstance(value, list):
                return "\n".join(filter(None, (_extract_response_text(item) for item in value)))
            return ""


        def _parse_model_ct_response(tool_response):
            try:
                raw_text = tool_response if isinstance(tool_response, str) else json.dumps(tool_response)
            except (TypeError, ValueError):
                raw_text = str(tool_response)
            parsed_value = _json_loads_maybe(tool_response)
            response_text = _extract_response_text(parsed_value)
            text_value = _json_loads_maybe(response_text)
            json_value = text_value if isinstance(text_value, (dict, list)) else parsed_value
            resolved_session_id = _walk_first_key(json_value, "resolved_session_id")
            json_field_match = resolved_session_id is not None
            if resolved_session_id is None and raw_text:
                match = re.search(r'"resolved_session_id"\s*:\s*"([^"]+)"', raw_text)
                if match:
                    resolved_session_id = match.group(1)
            is_error = False
            if isinstance(json_value, dict):
                is_error = bool(json_value.get("is_error") or json_value.get("isError") or json_value.get("error") is True)
            if '"is_error": true' in raw_text.lower() or '"iserror": true' in raw_text.lower():
                is_error = True
            return {
                "resolved_session_id": resolved_session_id,
                "surface_id": _walk_first_key(json_value, "surface_id"),
                "origin_id": _walk_first_key(json_value, "origin_id"),
                "bind_status": _walk_first_key(json_value, "bind_status"),
                "response_mode": _walk_first_key(json_value, "response_mode"),
                "response_chars": len(raw_text or ""),
                "is_error": is_error,
                "json_field_match": json_field_match,
            }


        def _model_ct_coherence(state, event, parsed):
            if parsed.get("is_error"):
                return "failed", "model_conversation_turn_error"
            if not parsed.get("resolved_session_id"):
                return "unknown", "missing_resolved_session_id"
            if not parsed.get("json_field_match"):
                return "unknown", "resolved_session_id_not_json_field"
            expected_session = state.get("pith_session_id")
            if expected_session and parsed.get("resolved_session_id") != expected_session:
                return "failed", "session_mismatch"
            surface_id = parsed.get("surface_id")
            if surface_id != "claude_code":
                return "failed" if surface_id else "unknown", "surface_mismatch_or_missing"
            origin_id = parsed.get("origin_id")
            expected_origin = _origin_for(event)
            if origin_id and origin_id != expected_origin:
                return "failed", "origin_mismatch"
            if not expected_session:
                return "unknown", "missing_hook_session"
            return "passed", "model_conversation_turn_matches_hook_session"


        def _handle_post_tool_use(event, state_path, state):
            # A1: exact tool-name match only. pith_search / pith_checkpoint and any
            # other tool must NOT set the marker, or the backstop would be wrongly
            # skipped and the turn lost.
            if event.get("tool_name") != PITH_CT_TOOL_NAME:
                return
            parsed = _parse_model_ct_response(event.get("tool_response"))
            status, reason = _model_ct_coherence(state, event, parsed)
            state["model_ct_session_id"] = parsed.get("resolved_session_id")
            state["model_ct_surface_id"] = parsed.get("surface_id")
            state["model_ct_origin_id"] = parsed.get("origin_id")
            state["model_ct_response_mode"] = parsed.get("response_mode")
            state["model_ct_response_chars"] = parsed.get("response_chars")
            state["model_ct_coherence_status"] = status
            state["model_ct_coherence_reason"] = reason
            if status == "passed" or (status == "unknown" and not state.get("pith_session_id")):
                state["model_fired_ct"] = True
                state["manual_ct_fired"] = True
            else:
                state["manual_ct_fired"] = False
            _write_state(state_path, state)


        def _backstop_request_id(event, state, payload):
            raw = "|".join([
                str(event.get("session_id") or ""),
                str(event.get("cwd") or ""),
                str(state.get("hook_turn_seq") or 0),
                str(payload.get("message") or ""),
                str(payload.get("previous_message") or ""),
                _sha256_text(payload.get("previous_response") or ""),
            ])
            return "claude-code-backstop:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:48]


        def _build_backstop_payload(event, state):
            previous_response = state.get("previous_response") or ""
            if len(previous_response) < MIN_LEARNABLE_RESPONSE_CHARS:
                _log("backstop_skipped_short_previous_response")
                return None
            payload = {
                "origin_id": _origin_for(event),
                "message": state.get("pending_prompt", ""),
                "previous_message": state.get("previous_message", ""),
                "previous_response": previous_response,
                "conversation_context": "Claude Code backstop lifecycle hook (model did not call conversation_turn)",
            }
            if state.get("pith_session_id"):
                payload["session_id"] = state["pith_session_id"]
            payload["request_id"] = _backstop_request_id(event, state, payload)
            return payload


        def _stop_learn_request_id(event, state, pending, response):
            raw = "|".join([
                str(event.get("session_id") or ""),
                str(event.get("cwd") or ""),
                str(state.get("hook_turn_seq") or 0),
                str(pending or ""),
                _sha256_text(response or ""),
            ])
            return "claude-code-stop-learn:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:48]


        def _build_bounded_stop_learn_summary(pending, response):
            prefix = f"Claude Code response captured for prompt '{(pending or '')[:120]}': "
            summary_source = " ".join((response or "").split())
            remaining = max(0, MAX_STOP_LEARN_SUMMARY_CHARS - len(prefix))
            if len(summary_source) > remaining:
                if remaining > 3:
                    summary_source = summary_source[: remaining - 3].rstrip() + "..."
                else:
                    summary_source = summary_source[:remaining]
            return (prefix + summary_source)[:MAX_STOP_LEARN_SUMMARY_CHARS]


        def _build_stop_learn_payload(event, state, pending, response):
            if not pending or len(response or "") < MIN_LEARNABLE_RESPONSE_CHARS:
                return None
            summary = _build_bounded_stop_learn_summary(pending, response)
            payload = {
                "user_message": pending,
                "assistant_response": response[-15000:],
                "knowledge_area": "conversation",
                "trigger_path": "claude_code_stop_hook",
                "request_id": _stop_learn_request_id(event, state, pending, response),
                "extracted_concepts": [
                    {
                        "summary": summary,
                        "confidence": 0.55,
                        "knowledge_area": "conversation",
                        "concept_type": "observation",
                        "evidence": [
                            "verified: Claude Code Stop hook captured this assistant response after UserPromptSubmit lifecycle registration"
                        ],
                    }
                ],
            }
            if state.get("pith_session_id"):
                payload["session_id"] = state["pith_session_id"]
            return payload


        def _classify_stop_learn_response(resp):
            if not isinstance(resp, dict):
                return None
            try:
                accepted_events = int(resp.get("accepted_learning_events") or resp.get("learning_events") or 0)
            except (TypeError, ValueError):
                accepted_events = 0
            capture_state = str(resp.get("learning_capture_state") or "")
            if accepted_events > 0 or capture_state == "accepted":
                return "committed"
            if resp.get("persistence_state") == "failed" or resp.get("processing_state") == "failed":
                return "failed"
            if resp.get("persistence_state") == "committed" or resp.get("processing_state") == "committed":
                return "degraded_zero_learning"
            status = str(resp.get("status") or resp.get("processing_state") or "ok")
            return status


        def _reconcile_stop_learn_status(state, request_id):
            if not request_id:
                return None
            resp, _error = _call_pith_result(
                "write_request_status",
                {"endpoint": "session_learn", "request_id": request_id},
                timeout=0.75,
            )
            if not isinstance(resp, dict):
                return None
            replay_status = str(resp.get("status") or resp.get("processing_state") or "")
            if replay_status:
                state["last_stop_learn_replay_status"] = replay_status
            status = _classify_stop_learn_response(resp.get("summary") if isinstance(resp.get("summary"), dict) else resp)
            if status:
                return status
            if replay_status == "failed":
                return "failed"
            return None


        def _fire_stop_learn(event, state, pending, response):
            payload = _build_stop_learn_payload(event, state, pending, response)
            if not payload:
                state["last_stop_learn_status"] = "skipped"
                return
            resp, error = _call_pith_result("session_learn", payload, timeout=4.0)
            request_id = payload.get("request_id")
            state["last_stop_learn_request_id"] = request_id
            if isinstance(resp, dict):
                status = _classify_stop_learn_response(resp)
                if status in {"processing", "unknown_pending"}:
                    status = _reconcile_stop_learn_status(state, request_id) or status
                state["last_stop_learn_status"] = status or "ok"
                state["last_stop_learn_learning_events"] = resp.get("learning_events")
                state["last_stop_learn_accepted_learning_events"] = resp.get("accepted_learning_events")
                state["last_stop_learn_learning_capture_state"] = resp.get("learning_capture_state")
                state["last_stop_learn_session_linkage_state"] = resp.get("session_linkage_state")
                if state["last_stop_learn_status"] == "committed":
                    state["last_stop_learn_response_hash"] = _sha256_text(response)
                else:
                    state.pop("last_stop_learn_response_hash", None)
                _log(f"stop_learn_sent request_id={request_id} status={state['last_stop_learn_status']}")
                return
            reconciled = _reconcile_stop_learn_status(state, request_id)
            state["last_stop_learn_status"] = reconciled or ("unknown_pending" if request_id else "failed")
            state["last_stop_learn_error"] = error or "unknown_error"
            _log(f"stop_learn_pending request_id={request_id} status={state['last_stop_learn_status']} error={error}")


        def _queue_backstop_retry(state, payload, error):
            queue = [
                item for item in _retry_queue(state)
                if item.get("request_id") != payload.get("request_id")
            ]
            now = time.time()
            item = {
                "request_id": payload.get("request_id"),
                "operation": "conversation_turn",
                "payload": payload,
                "attempts": 1,
                "first_seen_at": now,
                "last_attempt_at": now,
                "last_error": error or "unknown_error",
            }
            queue.append(item)
            _set_retry_queue(state, queue)
            _log(f"backstop_retry_queued request_id={payload.get('request_id')} error={error}")


        def _fire_backstop_ct(event, state, assistant_response):
            payload = _build_backstop_payload(event, state)
            if not payload:
                return
            resp, error = _call_pith_result("conversation_turn", payload, timeout=3.0)
            if isinstance(resp, dict):
                resolved = resp.get("resolved_session_id")
                if resolved:
                    state["pith_session_id"] = resolved
                _log(f"backstop_sent request_id={payload.get('request_id')}")
                return
            _queue_backstop_retry(state, payload, error)


        def _replay_one_retry(state):
            _prune_retry_queue(state)
            queue = _retry_queue(state)
            if not queue:
                return
            item = queue.pop(0)
            payload = item.get("payload") if isinstance(item, dict) else None
            if not isinstance(payload, dict):
                _set_retry_queue(state, queue)
                return
            resp, error = _call_pith_result(item.get("operation") or "conversation_turn", payload, timeout=3.0)
            if isinstance(resp, dict):
                resolved = resp.get("resolved_session_id")
                if resolved:
                    state["pith_session_id"] = resolved
                _log(f"backstop_retry_succeeded request_id={item.get('request_id')}")
                _set_retry_queue(state, queue)
                return
            try:
                item["attempts"] = int(item.get("attempts") or 0) + 1
            except (TypeError, ValueError):
                item["attempts"] = RETRY_MAX_ATTEMPTS
            item["last_attempt_at"] = time.time()
            item["last_error"] = error or "unknown_error"
            queue.append(item)
            _set_retry_queue(state, queue)
            _prune_retry_queue(state)


        def _latest_assistant_message(transcript_path):
            if not transcript_path:
                return ""
            path = Path(transcript_path)
            if not path.is_file():
                return ""
            try:
                lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                return ""
            for raw in reversed(lines[-200:]):
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if event.get("type") != "assistant":
                    continue
                message = event.get("message") or {}
                parts = message.get("content") or []
                chunks = []
                for part in parts:
                    if isinstance(part, dict) and part.get("type") == "text":
                        chunks.append(str(part.get("text") or ""))
                text = "\n".join(chunks).strip()
                if text:
                    return text[-8000:]
            return ""


        def _handle_stop(event, state_path, state):
            response = _latest_assistant_message(event.get("transcript_path"))
            pending = state.get("pending_prompt") or ""
            # Backstop: if the model did not (successfully) call conversation_turn
            # this turn, fire a capture-only conversation_turn so no turn is lost.
            # _fire_backstop_ct reads state['previous_*'] as the PRIOR turn, so it
            # must run BEFORE we advance the pointers below.
            if not state.get("model_fired_ct") and pending:
                _fire_backstop_ct(event, state, response)
            if pending and response:
                _fire_stop_learn(event, state, pending, response)
            if pending:
                state["previous_message"] = pending
            if response:
                state["previous_response"] = response
            state["last_stop_at"] = time.time()
            state["model_fired_ct"] = False
            state["pending_prompt"] = ""
            _write_state(state_path, state)


        def _handle_session_end(event, state_path, state):
            payload = {
                "origin_id": _origin_for(event),
            }
            if state.get("pith_session_id"):
                payload["session_id"] = state["pith_session_id"]
            previous_response = state.get("previous_response", "")
            previous_hash = _sha256_text(previous_response)
            if (
                previous_response
                and previous_hash != state.get("last_stop_learn_response_hash")
            ):
                payload["previous_message"] = state.get("previous_message", "")
                payload["previous_response"] = previous_response
            _call_pith("session_end", payload, timeout=3.0)


        def _handle_pre_compact(event, state_path, state):
            # SESSION-006: compaction fires PreCompact (not SessionEnd). Capture any
            # un-fired in-flight turn so its learning is not lost, then checkpoint so
            # pre-compaction state is durable. This is a compaction_checkpoint, NOT a
            # session_end (the session continues after compaction).
            if not state.get("model_fired_ct") and state.get("pending_prompt"):
                _fire_backstop_ct(event, state, state.get("previous_response", ""))
                state["model_fired_ct"] = True
            payload = {
                "origin_id": _origin_for(event),
                "action": "save",
                "description": "claude-code pre-compaction checkpoint",
            }
            if state.get("pith_session_id"):
                payload["session_id"] = state["pith_session_id"]
            _call_pith("checkpoint", payload, timeout=3.0)
            _write_state(state_path, state)


        def main():
            event = _load_input()
            name = event.get("hook_event_name")
            state_path = _state_path(event)
            state = _read_state(state_path)
            if name in ("UserPromptSubmit", "PostToolUse"):
                _replay_one_retry(state)
            if name == "UserPromptSubmit":
                _handle_user_prompt(event, state_path, state)
            elif name == "PostToolUse":
                _handle_post_tool_use(event, state_path, state)
            elif name == "Stop":
                _handle_stop(event, state_path, state)
            elif name == "PreCompact":
                _handle_pre_compact(event, state_path, state)
            elif name == "SessionEnd":
                _handle_session_end(event, state_path, state)
            _write_state(state_path, state)
            return 0


        if __name__ == "__main__":
            raise SystemExit(main())
        '''
    ).lstrip()


def _write_claude_code_hook_script(pith_home, dry_run=False):
    script_path = pith_home / "hooks" / PITH_CLAUDE_CODE_HOOK_SCRIPT_NAME
    if dry_run:
        return {"path": str(script_path), "action": "would_generate"}
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(_claude_code_hook_script_content(), encoding="utf-8")
    script_path.chmod(0o700)
    return {"path": str(script_path), "action": "generated"}


def _pith_hook_handler(python_cmd, script_path, event_name, timeout=5):
    return {
        "type": "command",
        "command": python_cmd,
        "args": [str(script_path)],
        "timeout": timeout,
        "statusMessage": f"Syncing Pith {event_name} lifecycle",
    }


def _remove_existing_pith_hook_handlers(groups, script_path):
    cleaned = []
    script = str(script_path)
    for group in groups if isinstance(groups, list) else []:
        if not isinstance(group, dict):
            cleaned.append(group)
            continue
        hooks = group.get("hooks")
        if not isinstance(hooks, list):
            cleaned.append(group)
            continue
        retained = []
        for hook in hooks:
            if not isinstance(hook, dict):
                retained.append(hook)
                continue
            args = hook.get("args") or []
            if script in [str(arg) for arg in args]:
                continue
            retained.append(hook)
        if retained:
            updated = dict(group)
            updated["hooks"] = retained
            cleaned.append(updated)
    return cleaned


def _merge_claude_code_hook(settings, event_name, handler, script_path, matcher=None):
    hooks = settings.setdefault("hooks", {})
    groups = _remove_existing_pith_hook_handlers(hooks.get(event_name, []), script_path)
    group = {"hooks": [handler]}
    if matcher is not None:
        group["matcher"] = matcher
    groups.append(group)
    hooks[event_name] = groups


def configure_claude_code_lifecycle_hooks(python_cmd, plat, dry_run=False):
    """Install Pith-owned Claude Code lifecycle hooks in ~/.claude/settings.json."""
    pith_home = _pith_home_path()
    script_result = _write_claude_code_hook_script(pith_home, dry_run=dry_run)
    script_path = Path(script_result["path"])
    settings_path = _claude_code_settings_path(plat)

    if dry_run:
        return {
            "client": "Claude Code",
            "scope": "lifecycle-hooks",
            "action": "would_configure",
            "path": str(settings_path),
            "hook_script": str(script_path),
            "hook_version": PITH_CLAUDE_CODE_HOOK_VERSION,
        }

    backup = _backup_file(str(settings_path))
    settings = _read_json(str(settings_path))
    _merge_claude_code_hook(
        settings,
        "UserPromptSubmit",
        _pith_hook_handler(python_cmd, script_path, "user_prompt", timeout=5),
        script_path,
    )
    # Backstop marker: PostToolUse fires only for the exact pith conversation_turn
    # MCP tool, recording that the model captured this turn itself (no double-fire).
    _merge_claude_code_hook(
        settings,
        "PostToolUse",
        _pith_hook_handler(python_cmd, script_path, "post_tool_use", timeout=5),
        script_path,
        matcher="mcp__pith__pith_conversation_turn",
    )
    _merge_claude_code_hook(
        settings,
        "Stop",
        _pith_hook_handler(python_cmd, script_path, "stop", timeout=5),
        script_path,
    )
    # SESSION-006: capture in-flight turn + checkpoint before context compaction.
    _merge_claude_code_hook(
        settings,
        "PreCompact",
        _pith_hook_handler(python_cmd, script_path, "pre_compact", timeout=5),
        script_path,
    )
    _merge_claude_code_hook(
        settings,
        "SessionEnd",
        _pith_hook_handler(python_cmd, script_path, "session_end", timeout=5),
        script_path,
    )
    _write_json(str(settings_path), settings)
    if not _validate_json(str(settings_path)):
        if backup and os.path.isfile(backup):
            shutil.copy2(backup, settings_path)
        return {
            "client": "Claude Code",
            "scope": "lifecycle-hooks",
            "action": "error",
            "error": "Claude Code settings validation failed after write",
            "path": str(settings_path),
        }
    result = {
        "client": "Claude Code",
        "scope": "lifecycle-hooks",
        "action": "configured",
        "path": str(settings_path),
        "hook_script": str(script_path),
        "hook_version": PITH_CLAUDE_CODE_HOOK_VERSION,
    }
    if backup:
        result["backup"] = backup
    return result


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
    entry = _build_standard_payload(
        server_path,
        api_key,
        python_cmd=python_cmd,
        api_url=api_url,
        surface_id="codex_local_api",
    )
    return "\n".join(
        [
            "[mcp_servers.pith]",
            f'command = "{_escape_toml_string(entry["command"])}"',
            f'args = ["{_escape_toml_string(entry["args"][0])}"]',
            "[mcp_servers.pith.env]",
            f'PITH_API_KEY = "{_escape_toml_string(entry["env"]["PITH_API_KEY"])}"',
            f'PITH_API_URL = "{_escape_toml_string(entry["env"]["PITH_API_URL"])}"',
            f'PITH_SURFACE_ID = "{_escape_toml_string(entry["env"]["PITH_SURFACE_ID"])}"',
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
            and env.get("PITH_SURFACE_ID") == "codex_local_api"
        )

    return (
        "[mcp_servers.pith]" in text
        and "[mcp_servers.pith.env]" in text
        and f'PITH_API_URL = "{api_url}"' in text
        and 'PITH_SURFACE_ID = "codex_local_api"' in text
        and "pith_mcp.py" in text
    )



def configure_standard_client(client_id, info, server_path, api_key, plat, dry_run=False, python_cmd=None, api_url="http://localhost:8000"):
    """Configure a standard mcpServers-based client (Claude Desktop, Code, Cursor, Windsurf, Cline)."""
    config_path = _expand(info["config_file"][plat], plat)
    label = info["label"]
    entry = _build_standard_payload(
        server_path,
        api_key,
        python_cmd=python_cmd,
        extra_fields=info.get("extra_fields"),
        api_url=api_url,
        surface_id=CLIENT_SURFACE_IDS.get(client_id),
    )

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
                    "PITH_SURFACE_ID": "vscode_copilot_mcp",
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
            "PITH_SURFACE_ID": "vscode_copilot_mcp",
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
                    "PITH_SURFACE_ID": "claude_code",
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
            if client_id == "claude_code" and r.get("action") != "error":
                hook_result = configure_claude_code_lifecycle_hooks(
                    _resolve_python_or_exit(server_path, args.python_cmd),
                    plat,
                    args.dry_run,
                )
                if hook_result.get("action") == "error":
                    results["errors"].append(hook_result)
                else:
                    results["configured"].append(hook_result)
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
