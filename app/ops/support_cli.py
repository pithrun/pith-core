"""Operational support CLI commands for Pith.

These commands are intentionally read-only except for `support bundle`, which
writes a bounded, redacted diagnostic archive.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
import os
import platform
import re
import shutil
import socket
import sqlite3
import stat
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    import tomllib
except ImportError:  # pragma: no cover - Python <3.11 support
    tomllib = None

from app.ops.app_surface_proof import (
    AppSurfaceProofError,
    ProofExpectations,
    load_proof_ledger,
)
from app.ops.claude_host_contract import collect_claude_host_diagnostics
from app.ops.codex_plugin_contract import (
    CODEX_PLUGIN_DEFAULT_TOOLS_APPROVAL_MODE,
    CODEX_PLUGIN_SUPPORT_READBACK_TIMEOUT_SECONDS,
    CodexPluginContractError,
    codex_plugin_secret_values_match,
    inspect_codex_plugin_operation_lock,
    load_codex_plugin_package,
    load_installed_source_state,
    normalize_codex_plugin_marketplace_display_name,
    parse_codex_plugin_list,
    validate_codex_plugin_payloads,
)

DEFAULT_TIMEOUT = 5.0
VENV_PATH_MAX_BYTES = 4096
SUPPORT_BUNDLE_VERSION = 1
LEGACY_SERVER_NAMES = {"pith", "pith-mcp", "pith-mcp-wrapper"}
LAUNCHD_LABEL = "dev.pith.server"
PRODUCTION_CLIENT_SURFACE_ORDER = (
    "claude_desktop",
    "claude_code",
    "vscode",
    "cursor",
    "codex",
)
APP_SURFACE_PARITY_SCHEMA_VERSION = "pith_app_surface_parity.v1"
APP_SURFACE_PARITY_ORDER = (
    "local_api_cli",
    "codex",
    "claude_code",
    "claude_desktop",
    "claude_chat",
    "claude_cowork",
    "chatgpt",
)
CHATGPT_UNSUPPORTED_STATUS = "not_supported_or_connector_missing"
CHATGPT_PROCESS_DETECTED_STATUS = "process_detected_connector_missing"
CHATGPT_REMOTE_CONNECTOR_REQUIRED_STATUS = "remote_mcp_connector_required"
CODEX_CHATGPT_PLUGIN_UNKNOWN_STATUS = "codex_app_process_detected_plugin_unknown"
CODEX_CHATGPT_PLUGIN_MISSING_STATUS = "codex_chatgpt_plugin_package_missing"
CODEX_CHATGPT_PLUGIN_WRITTEN_STATUS = "codex_chatgpt_plugin_package_written_not_installed"
CODEX_CHATGPT_PLUGIN_CONFIGURED_STATUS = "codex_chatgpt_plugin_configured_requires_live_proof"
CODEX_CHATGPT_PLUGIN_SCHEMA_INVALID_STATUS = "codex_chatgpt_plugin_package_schema_invalid"
CODEX_CHATGPT_PLUGIN_UPDATE_IN_PROGRESS_STATUS = "codex_chatgpt_plugin_update_in_progress"
CODEX_CHATGPT_PLUGIN_DISABLED_STATUS = "codex_chatgpt_plugin_installed_disabled"
CODEX_CHATGPT_PLUGIN_VERSION_MISMATCH_STATUS = "codex_chatgpt_plugin_installed_version_mismatch"
CODEX_CHATGPT_PLUGIN_SOURCE_UNRESOLVABLE_STATUS = "codex_chatgpt_plugin_installed_source_unresolvable"
CODEX_CHATGPT_PLUGIN_PACKAGE_MISMATCH_STATUS = "codex_chatgpt_plugin_installed_package_mismatch"
CODEX_CHATGPT_PLUGIN_READBACK_ERROR_STATUS = "codex_chatgpt_plugin_install_readback_error"
CODEX_DESKTOP_MCP_READBACK_TIMEOUT_SECONDS = 15
HOST_VARIANT_LOADER_UNVERIFIED_STATUS = "host_variant_loader_unverified_requires_same_turn_proof"
HOST_VARIANT_LOADER_UNVERIFIED_NOT_CONFIGURED_STATUS = "host_variant_loader_unverified_not_configured"
SECRET_KEY_RE = re.compile(r"(api[_-]?key|token|secret|password|private[_-]?key|access[_-]?key|auth)", re.I)
ENV_ASSIGNMENT_RE = re.compile(
    r"\b([A-Z0-9_]*(?:API_KEY|TOKEN|SECRET|PASSWORD|PRIVATE_KEY|ACCESS_KEY|AUTH)[A-Z0-9_]*)=(\"[^\"]*\"|'[^']*'|\S+)"
)
JSON_SECRET_RE = re.compile(
    r'("?[A-Za-z0-9_.-]*(?:api[_-]?key|token|secret|password|private[_-]?key|access[_-]?key|auth)[A-Za-z0-9_.-]*"?\s*[:=]\s*)("[^"]*"|\'[^\']*\'|[^\s,}]+)',
    re.I,
)
HEADER_SECRET_RE = re.compile(r"((?:X-API-Key|Authorization):\s*)(Bearer\s+)?\S+", re.I)
LONG_TOKEN_RE = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{12,}|[A-Za-z0-9_-]{40,})\b")
SAFE_PUBLIC_DIAGNOSTIC_FIELDS = {
    "generated_version",
    "installed_public_package_digest",
    "installed_source_kind",
    "installed_version",
    "plugin_install_readback_state",
    "plugin_mcp_schema_status",
    "public_package_digest",
    "session_active_source",
    "status",
}

APP_PROOF_ENV = {
    "ledger": "PITH_APP_PROOF_LEDGER",
    "run_id": "PITH_APP_PROOF_RUN_ID",
    "artifact": "PITH_APP_PROOF_ARTIFACT_SHA256",
    "package": "PITH_APP_PROOF_PACKAGE_VERSION",
    "target_os_build": "PITH_APP_PROOF_TARGET_OS_BUILD",
    "host_versions": "PITH_APP_PROOF_HOST_VERSIONS_JSON",
    "key": "PITH_RELEASE_PROOF_KEY",
    "key_id": "PITH_RELEASE_PROOF_KEY_ID",
}


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _pith_home() -> Path:
    return Path(os.environ.get("PITH_HOME", str(Path.home() / ".pith"))).expanduser()


def _pith_server_path() -> Path:
    return Path(os.environ.get("PITH_SERVER_PATH", str(_pith_home() / "pith-server"))).expanduser()


def _diagnostic_venv_python() -> tuple[Path | None, str | None]:
    """Resolve the installed venv without hiding an invalid or stale selection."""
    home = _pith_home()
    windows = platform.system() == "Windows"
    suffix = Path("Scripts/python.exe" if windows else "bin/python3")
    default = home / "venv" / suffix
    record = home / "config" / "venv.path"
    try:
        record_stat = record.lstat()
    except FileNotFoundError:
        return default, None
    except OSError:
        return None, "configured venv path is unreadable"

    try:
        if record.is_symlink() or not stat.S_ISREG(record_stat.st_mode):
            return None, "configured venv path is not a regular file"
        if record_stat.st_size > VENV_PATH_MAX_BYTES:
            return None, "configured venv path is oversized"
        with record.open("rb") as stream:
            raw = stream.read(VENV_PATH_MAX_BYTES + 1)
        if len(raw) > VENV_PATH_MAX_BYTES:
            return None, "configured venv path is oversized"
        value = raw.decode("utf-8-sig").strip()
        configured = Path(value)
        if not value or "\0" in value or not configured.is_absolute():
            return None, "configured venv path is invalid"

        expected = home / "venv"
        if windows:
            local_app_data = Path(os.environ.get("LOCALAPPDATA") or home.parent / "AppData" / "Local")
            name = hashlib.sha256(str(home).encode("utf-8")).hexdigest()[:12]
            expected = local_app_data / "Pith" / "venvs" / name
        normalized = os.path.normcase(os.path.abspath(configured))
        allowed = os.path.normcase(os.path.abspath(expected))
        if (normalized.casefold() if windows else normalized) != (allowed.casefold() if windows else allowed):
            return None, "configured venv path is outside the managed runtime root"
        python = expected / suffix
        if windows:
            # Do not execute a configured path redirected outside the managed venv.
            for path in (expected.parent, expected, python.parent, python):
                if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
                    return None, "configured venv path contains a link"
        return python, None
    except (OSError, UnicodeError, ValueError):
        return None, "configured venv path is unreadable or invalid"


def _data_dir() -> Path:
    if value := os.environ.get("PITH_DATA_DIR"):
        return Path(value).expanduser()
    if profile := os.environ.get("PITH_PROFILE"):
        return Path.home() / "pith-data" / profile
    return Path.home() / "pith-data" / "default"


def _base_url(args_base_url: str | None = None) -> str:
    if args_base_url:
        return args_base_url.rstrip("/")
    if env_url := os.environ.get("PITH_API_URL"):
        return env_url.rstrip("/")
    return f"http://127.0.0.1:{os.environ.get('PITH_PORT', '8000')}"


def _fetch_json(base_url: str, path: str, timeout: float) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(f"{base_url}{path}", timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        return {"ok": False, "error": redact_text(str(exc))}
    except TimeoutError:
        return {"ok": False, "error": "timeout"}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"ok": False, "error": "non-json response"}
    return data if isinstance(data, dict) else {"payload": data}


def _run(args: list[str], timeout: float = 3.0) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(args, text=True, capture_output=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None


def _process_exists(pid: int | None) -> bool:
    if not pid:
        return False
    if platform.system() == "Windows":
        return _windows_process_exists(pid)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _windows_process_exists(pid: int) -> bool:
    try:
        import ctypes

        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(process_query_limited_information, False, int(pid))
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    except Exception:
        return False


def _process_command(pid: int | None) -> str | None:
    if not pid:
        return None
    if platform.system() == "Windows":
        return None
    result = _run(["ps", "-p", str(pid), "-o", "comm="], timeout=1.5)
    if not result or result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _pid_file_status(pith_home: Path) -> dict[str, Any]:
    path = pith_home / "pith.pid"
    if not path.exists():
        return {"path": str(path), "state": "missing", "pid": None, "running": False}
    raw = path.read_text(encoding="utf-8", errors="replace").strip()
    try:
        pid = int(raw)
    except ValueError:
        return {"path": str(path), "state": "invalid", "pid": raw, "running": False}
    running = _process_exists(pid)
    return {
        "path": str(path),
        "state": "valid" if running else "stale",
        "pid": pid,
        "running": running,
        "command": _process_command(pid),
    }


def _launchd_status(timeout: float = 2.0) -> dict[str, Any]:
    if platform.system() != "Darwin" or not shutil.which("launchctl"):
        return {"available": False, "loaded": False}
    service = f"gui/{os.getuid()}/{LAUNCHD_LABEL}"
    result = _run(["launchctl", "print", service], timeout=timeout)
    if not result or result.returncode != 0:
        return {"available": True, "loaded": False, "service": service}
    state = None
    pid = None
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if state is None and stripped.startswith("state ="):
            state = stripped.split("=", 1)[1].strip()
        match = re.match(r"pid\s*=\s*(\d+)", stripped, re.I)
        if match:
            pid = int(match.group(1))
    return {
        "available": True,
        "loaded": True,
        "service": service,
        "state": state,
        "pid": pid,
        "running": state == "running" and _process_exists(pid),
    }


def _port_from_base_url(base_url: str) -> int:
    parsed = urlparse(base_url)
    return parsed.port or (443 if parsed.scheme == "https" else 80)


def _port_status(port: int) -> dict[str, Any]:
    if platform.system() == "Windows":
        try:
            with socket.create_connection(("127.0.0.1", int(port)), timeout=1.0):
                return {"port": port, "listening": True, "pid": None, "method": "socket"}
        except OSError as exc:
            return {
                "port": port,
                "listening": False,
                "pid": None,
                "method": "socket",
                "error": redact_text(exc),
            }
    if shutil.which("lsof"):
        result = _run(["lsof", "-i", f":{port}", "-sTCP:LISTEN", "-t"], timeout=1.5)
        if result and result.returncode == 0:
            first = next((line.strip() for line in result.stdout.splitlines() if line.strip()), None)
            if first and first.isdigit():
                pid = int(first)
                return {"port": port, "listening": True, "pid": pid, "command": _process_command(pid)}
    return {"port": port, "listening": False, "pid": None}


def redact_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = ENV_ASSIGNMENT_RE.sub(r"\1=<redacted>", text)
    text = HEADER_SECRET_RE.sub(r"\1<redacted>", text)
    text = JSON_SECRET_RE.sub(r"\1<redacted>", text)
    text = LONG_TOKEN_RE.sub("<redacted>", text)
    return text


def _redact_obj(value: Any) -> Any:
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if key in SAFE_PUBLIC_DIAGNOSTIC_FIELDS:
                output[key] = item
            elif SECRET_KEY_RE.search(str(key)):
                if isinstance(item, bool) or item is None or isinstance(item, (int, float)):
                    output[key] = item
                else:
                    output[key] = "<redacted:present>" if item else "<redacted:empty>"
            else:
                output[key] = _redact_obj(item)
        return output
    if isinstance(value, list):
        return [_redact_obj(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def _print_json(data: Any) -> None:
    print(json.dumps(_redact_obj(data), indent=2, sort_keys=True, default=str))


def _load_configure_clients():
    try:
        return importlib.import_module("scripts.configure_clients")
    except Exception:
        return None


def _as_path_specs(path_value: Any) -> list[str]:
    if not path_value:
        return []
    if isinstance(path_value, (list, tuple)):
        return [str(item) for item in path_value if item]
    return [str(path_value)]


def _expand_config_paths(
    path_value: Any,
    plat: str,
    configure_clients: Any | None,
    *,
    include_missing_glob_children: bool = False,
) -> list[Path]:
    if configure_clients and hasattr(configure_clients, "_expand_path_candidates"):
        return [
            Path(item).expanduser()
            for item in configure_clients._expand_path_candidates(
                path_value,
                plat,
                include_missing_glob_children=include_missing_glob_children,
            )
        ]
    if configure_clients and hasattr(configure_clients, "_expand"):
        return [Path(configure_clients._expand(item, plat)).expanduser() for item in _as_path_specs(path_value)]
    return [Path(os.path.expanduser(item)) for item in _as_path_specs(path_value)]


def _select_report_config_path(path_value: Any, plat: str, configure_clients: Any | None) -> Path:
    candidates = _expand_config_paths(
        path_value,
        plat,
        configure_clients,
        include_missing_glob_children=True,
    )
    if not candidates:
        return Path("")
    for path in candidates:
        if path.is_file():
            return path
    for path in candidates:
        if path.parent.is_dir():
            return path
    return candidates[0]


def _any_path_is_dir(path_value: Any, plat: str, configure_clients: Any | None) -> bool:
    return any(path.is_dir() for path in _expand_config_paths(path_value, plat, configure_clients))


def _json_has_pith_config(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    roots = [payload.get("mcpServers"), payload.get("mcp_servers"), payload.get("servers")]
    return any(isinstance(root, dict) and any(name in root for name in LEGACY_SERVER_NAMES) for root in roots)


def _plugin_mcp_schema_state(manifest_path: Path, mcp_path: Path) -> dict[str, Any]:
    state: dict[str, Any] = {
        "plugin_mcp_schema_status": "missing",
        "plugin_mcp_schema_reason": "Plugin MCP file is missing.",
        "plugin_mcp_server_names": [],
    }
    if not manifest_path.is_file() or not mcp_path.is_file():
        return state
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload = json.loads(mcp_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        state["plugin_mcp_schema_status"] = "parse_error"
        state["plugin_mcp_schema_reason"] = f"Plugin package could not be parsed: {redact_text(exc)}"
        return state
    state.update(validate_codex_plugin_payloads(manifest, payload))
    if state["plugin_mcp_schema_status"] == "valid":
        state["generated_version"] = manifest["version"]
    return state


def _client_release_validation(client_id: str) -> dict[str, Any]:
    return {
        "production_validated": False,
        "evidence": None,
        "claim": "configuration_surface_only",
    }


def _client_config_state(clients_by_id: dict[str, dict[str, Any]], client_id: str) -> str:
    item = clients_by_id.get(client_id)
    if not item:
        return "not_detected"
    if item.get("pith_configured"):
        return "configured"
    if item.get("detected") or item.get("config_exists"):
        return "present_without_pith"
    return "not_detected"


def _windows_named_process_state(process_name: str) -> dict[str, Any]:
    if platform.system() != "Windows":
        return {"detected": False, "process_count": 0, "process_paths": []}
    if re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", process_name) is None:
        return {"detected": False, "process_count": 0, "process_paths": [], "error": "invalid_process_name"}
    result = _run(
        [
            "powershell.exe",
            "-NoLogo",
            "-NoProfile",
            "-Command",
            (
                f"Get-Process -Name {process_name} -ErrorAction SilentlyContinue | "
                "Select-Object -ExpandProperty Path | ConvertTo-Json -Compress"
            ),
        ],
        timeout=2.0,
    )
    if not result or result.returncode != 0:
        return {"detected": False, "process_count": 0, "process_paths": [], "error": "process_probe_failed"}
    try:
        payload = json.loads(result.stdout or "[]")
    except json.JSONDecodeError:
        payload = []
    if isinstance(payload, str):
        paths = [payload]
    elif isinstance(payload, list):
        paths = [str(item) for item in payload if item]
    else:
        paths = []
    return {
        "detected": bool(paths),
        "process_count": len(paths),
        "process_paths": sorted(set(paths)),
    }


def _windows_chatgpt_process_state() -> dict[str, Any]:
    return _windows_named_process_state("ChatGPT")


def _resolve_codex_cli_for_support() -> str | None:
    if resolved := shutil.which("codex"):
        return resolved
    if platform.system() != "Windows":
        return None
    candidates: list[Path] = []
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        candidates.extend(Path(local_appdata).glob("OpenAI/Codex/bin/*/codex.exe"))
        candidates.append(Path(local_appdata) / "Microsoft" / "WindowsApps" / "codex.exe")
        for root in Path(local_appdata).glob("Packages/OpenAI.Codex_*/LocalCache/Local"):
            candidates.extend(root.glob("OpenAI/Codex/bin/*/codex.exe"))
    for program_root in filter(
        None,
        [os.environ.get("PROGRAMFILES"), os.environ.get("PROGRAMW6432"), r"C:\Program Files"],
    ):
        for root in Path(program_root).glob("WindowsApps/OpenAI.Codex_*/app"):
            candidates.append(root / "resources" / "codex.exe")
            candidates.append(root / "Codex.exe")
    return next((str(candidate) for candidate in candidates if candidate.is_file()), None)


def _codex_desktop_mcp_state() -> dict[str, Any]:
    """Read back Codex's MCP entry without returning environment values."""
    state: dict[str, Any] = {
        "desktop_mcp_readback_state": "cli_unavailable",
        "desktop_mcp_listed": False,
        "desktop_mcp_enabled": None,
        "desktop_mcp_transport_type": None,
        "desktop_mcp_command_matches_installed_runtime": False,
    }
    codex_cli = _resolve_codex_cli_for_support()
    if not codex_cli:
        return state
    result = _run(
        [codex_cli, "mcp", "list", "--json"],
        timeout=CODEX_DESKTOP_MCP_READBACK_TIMEOUT_SECONDS,
    )
    if not result or result.returncode != 0:
        state["desktop_mcp_readback_state"] = "readback_error"
        if result:
            state["desktop_mcp_readback_error"] = redact_text(result.stderr or result.stdout)[:500]
        return state
    try:
        payload = json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        state["desktop_mcp_readback_state"] = "invalid_json"
        return state
    entries = payload.get("servers") if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        state["desktop_mcp_readback_state"] = "invalid_schema"
        return state
    matches = [entry for entry in entries if isinstance(entry, dict) and entry.get("name") == "pith"]
    if len(matches) != 1:
        state["desktop_mcp_readback_state"] = "missing" if not matches else "duplicate"
        return state
    entry = matches[0]
    transport = entry.get("transport")
    state["desktop_mcp_listed"] = True
    state["desktop_mcp_enabled"] = entry.get("enabled") is True and entry.get("disabled_reason") is None
    if not isinstance(transport, dict):
        state["desktop_mcp_readback_state"] = "invalid_transport"
        return state
    state["desktop_mcp_transport_type"] = transport.get("type")
    home = _pith_home()

    def normalize_path(value: Any) -> str:
        rendered = str(value).replace("\\", "/")
        return rendered.lower() if platform.system() == "Windows" else os.path.normcase(rendered)

    expected_commands = {
        normalize_path(home / "venv" / "Scripts" / "python.exe"),
        normalize_path(home / "runtime" / "python" / "python.exe"),
        normalize_path(home / "venv" / "bin" / "python3"),
        normalize_path(home / "runtime" / "python" / "bin" / "python3"),
    }
    venv_path_file = home / "config" / "venv.path"
    try:
        if not venv_path_file.is_symlink() and venv_path_file.stat().st_size <= 4096:
            configured_venv = Path(venv_path_file.read_text(encoding="utf-8-sig").strip())
            expected_commands.add(normalize_path(configured_venv / "Scripts" / "python.exe"))
            expected_commands.add(normalize_path(configured_venv / "bin" / "python3"))
    except (FileNotFoundError, OSError, UnicodeError):
        pass
    command = transport.get("command")
    args = transport.get("args")
    normalized_command = normalize_path(command) if isinstance(command, str) else ""
    expected_bridge = normalize_path(home / "pith-server" / "pith_mcp.py")
    normalized_args = [normalize_path(item) for item in args] if isinstance(args, list) else []
    command_matches = normalized_command in expected_commands and normalized_args == [expected_bridge]
    state["desktop_mcp_command_matches_installed_runtime"] = command_matches
    if transport.get("type") != "stdio":
        state["desktop_mcp_readback_state"] = "unsupported_transport"
    elif not state["desktop_mcp_enabled"]:
        state["desktop_mcp_readback_state"] = "disabled"
    elif not command_matches:
        state["desktop_mcp_readback_state"] = "command_mismatch"
    else:
        state["desktop_mcp_readback_state"] = "listed"
    return state


def _codex_chatgpt_plugin_state() -> dict[str, Any]:
    plugin_root = Path.home() / "plugins" / "pith"
    manifest_path = plugin_root / ".codex-plugin" / "plugin.json"
    mcp_path = plugin_root / ".mcp.json"
    marketplace_path = Path.home() / ".agents" / "plugins" / "marketplace.json"
    state: dict[str, Any] = {
        "plugin_root": str(plugin_root),
        "manifest_path": str(manifest_path),
        "mcp_path": str(mcp_path),
        "marketplace_path": str(marketplace_path),
        "plugin_root_exists": plugin_root.is_dir(),
        "manifest_exists": manifest_path.is_file(),
        "mcp_exists": mcp_path.is_file(),
        "marketplace_exists": marketplace_path.is_file(),
        "marketplace_entry_exists": False,
        "package_written": False,
        "package_valid": False,
        "marketplace_registered": False,
        "plugin_install_readback_state": "not_checked",
    }
    lock_state = inspect_codex_plugin_operation_lock(plugin_root)
    state["operation_lock"] = lock_state
    if lock_state["state"] != "absent":
        state["plugin_install_readback_state"] = "update_in_progress"
        return state
    state.update(_plugin_mcp_schema_state(manifest_path, mcp_path))
    if marketplace_path.is_file():
        try:
            payload = json.loads(marketplace_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            state["marketplace_error"] = str(exc)
        else:
            interface = payload.get("interface")
            if isinstance(interface, dict):
                try:
                    state["marketplace_display_name"] = normalize_codex_plugin_marketplace_display_name(
                        interface.get("displayName")
                    )
                except CodexPluginContractError as exc:
                    state["marketplace_error"] = str(exc)
            plugins = payload.get("plugins")
            if isinstance(plugins, list):
                state["marketplace_entry_exists"] = any(
                    isinstance(item, dict) and item.get("name") == "pith" for item in plugins
                )
    state["package_written"] = bool(state["manifest_exists"] and state["mcp_exists"])
    state["package_valid"] = bool(state["package_written"] and state["plugin_mcp_schema_status"] == "valid")
    state["marketplace_registered"] = bool(state["marketplace_entry_exists"])
    source_package = None
    if state["package_valid"]:
        try:
            source_package = load_codex_plugin_package(plugin_root)
        except CodexPluginContractError as exc:
            state["plugin_install_readback_state"] = exc.code
            state["plugin_install_readback_error"] = redact_text(exc)
            state["package_valid"] = False
        else:
            state["public_package_digest"] = source_package.public_digest
    if codex_cli := _resolve_codex_cli_for_support():
        state["codex_cli"] = codex_cli
    codex_cli = state.get("codex_cli")
    if not codex_cli or not source_package:
        if state["plugin_install_readback_state"] == "not_checked":
            state["plugin_install_readback_state"] = "codex_cli_unavailable" if not codex_cli else "source_invalid"
        return state
    result = _run(
        [str(codex_cli), "plugin", "list", "--json"],
        timeout=CODEX_PLUGIN_SUPPORT_READBACK_TIMEOUT_SECONDS,
    )
    if not result or result.returncode != 0:
        state["plugin_install_readback_state"] = "readback_error"
        if result:
            state["plugin_install_readback_error"] = redact_text(result.stderr or result.stdout)[:500]
        return state
    try:
        installed_state = parse_codex_plugin_list(
            json.loads(result.stdout.strip()),
            "pith@personal",
            str(state["generated_version"]),
        )
        installed_package = load_installed_source_state(installed_state["entry"])
    except (json.JSONDecodeError, CodexPluginContractError) as exc:
        state["plugin_install_readback_state"] = getattr(exc, "code", "invalid_json")
        state["plugin_install_readback_error"] = redact_text(exc)
        return state
    public_match = source_package.public_digest == installed_package.public_digest
    secret_match = codex_plugin_secret_values_match(source_package.mcp_payload, installed_package.mcp_payload)
    state.update({key: value for key, value in installed_state.items() if key != "entry"})
    state.update(
        {
            "plugin_install_readback_state": "matched" if public_match and secret_match else "package_mismatch",
            "installed_public_package_digest": installed_package.public_digest,
            "public_package_matches_source": public_match,
            "secret_values_match": secret_match,
            "installed_package_matches_source": public_match and secret_match,
            "requires_new_chat_proof": public_match and secret_match,
        }
    )
    return state


def _chatgpt_app_surface_fields(platform_name: str) -> dict[str, Any]:
    """Report ChatGPT Chat's remote transport boundary separately from Codex Work."""
    process_state = (
        _windows_chatgpt_process_state()
        if platform_name == "windows"
        else {"detected": False, "process_count": 0, "process_paths": []}
    )
    return {
        "status": CHATGPT_REMOTE_CONNECTOR_REQUIRED_STATUS,
        "status_reason": (
            "ChatGPT Chat cannot call the installer-managed localhost MCP server directly. "
            "Configure a remote MCP app or OpenAI Secure MCP Tunnel; the local Pith plugin is for Codex Work."
        ),
        "process_detected": bool(process_state.get("detected")),
        "process_count": process_state.get("process_count"),
        "process_paths": process_state.get("process_paths"),
        "local_mcp_supported": False,
        "local_plugin_surface": "codex_work",
        "transport_requirement": "remote_mcp_or_secure_mcp_tunnel",
        "tunnel_management_url": "https://platform.openai.com/settings/organization/tunnels",
        "connector_settings_url": "https://chatgpt.com/#settings/Connectors",
        "developer_mode_url": "https://developers.openai.com/api/docs/guides/developer-mode",
        "session_active_source": "not_proven_remote_connector_missing",
    }


def _app_surface_row(
    surface_id: str,
    label: str,
    status: str,
    status_reason: str,
    proof_required: str,
    weak_evidence_not_connected: list[str],
    release_blocker_for_full_app_parity: bool,
    **extra: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "surface_id": surface_id,
        "label": label,
        "status": status,
        "status_reason": status_reason,
        "proof_required": proof_required,
        "weak_evidence_not_connected": weak_evidence_not_connected,
        "release_blocker_for_full_app_parity": release_blocker_for_full_app_parity,
    }
    row.update({key: value for key, value in extra.items() if value is not None})
    return row


def _collect_app_surface_proof_validation() -> dict[str, Any]:
    ledger = os.environ.get(APP_PROOF_ENV["ledger"])
    if not ledger:
        return {
            "status": "not_provided",
            "evidence_tier": None,
            "evidence_freshness": "unavailable",
            "artifact_match": False,
            "host_version_match": False,
            "attestation_status": "not_checked",
        }
    values = {name: os.environ.get(env_name) for name, env_name in APP_PROOF_ENV.items()}
    required = ("run_id", "artifact", "package", "target_os_build", "host_versions", "key", "key_id")
    if any(not values.get(name) for name in required):
        return {
            "status": "invalid",
            "error_code": "proof_context_incomplete",
            "evidence_tier": None,
            "evidence_freshness": "unavailable",
            "artifact_match": False,
            "host_version_match": False,
            "attestation_status": "missing",
        }
    try:
        host_versions = json.loads(str(values["host_versions"]))
        if not isinstance(host_versions, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in host_versions.items()
        ):
            raise ValueError
        expectations = ProofExpectations(
            run_id=str(values["run_id"]),
            artifact_sha256=str(values["artifact"]),
            package_version=str(values["package"]),
            target_os_build=str(values["target_os_build"]),
            host_versions=host_versions,
            key_id=str(values["key_id"]),
        )
        return load_proof_ledger(str(ledger), expectations=expectations, key=str(values["key"]))
    except (ValueError, json.JSONDecodeError):
        return {
            "status": "invalid",
            "error_code": "invalid_host_version_context",
            "evidence_tier": None,
            "evidence_freshness": "unavailable",
            "artifact_match": False,
            "host_version_match": False,
            "attestation_status": "invalid",
        }
    except AppSurfaceProofError as exc:
        return {
            "status": "invalid",
            "error_code": exc.code,
            "evidence_tier": None,
            "evidence_freshness": "unavailable",
            "artifact_match": False,
            "host_version_match": False,
            "attestation_status": "invalid",
        }


def _apply_app_surface_proof(matrix: dict[str, Any], proof: dict[str, Any]) -> dict[str, Any]:
    matrix["proof_validation"] = proof
    if proof.get("status") == "valid":
        connected = set(proof.get("connected_pairs") or [])
        for row in matrix.get("surfaces") or []:
            surface_id = row.get("surface_id")
            required_pairs: list[str] = []
            if surface_id == "chatgpt":
                required_pairs = ["chatgpt/chat"]
            elif surface_id == "codex":
                required_pairs = ["chatgpt/work"]
            elif surface_id == "claude_chat":
                required_pairs = ["claude_chat/desktop"]
            elif surface_id == "claude_code":
                required_pairs = ["claude_code/code"]
            if required_pairs and all(pair in connected for pair in required_pairs):
                row.update(
                    {
                        "status": "same_turn_connected",
                        "status_reason": "Fresh release-attested same-turn proof passed for the required host variant(s).",
                        "release_blocker_for_full_app_parity": False,
                        "proof_pairs": required_pairs,
                        "evidence_tier": "release_attested",
                        "evidence_freshness": "fresh",
                        "attestation_status": "valid",
                    }
                )
    matrix["blockers"] = [
        str(row["surface_id"])
        for row in matrix.get("surfaces") or []
        if row.get("release_blocker_for_full_app_parity") is True
    ]
    matrix["claimable"] = not matrix["blockers"]
    return matrix


def _build_app_surface_parity(clients: list[dict[str, Any]], platform_name: str | None = None) -> dict[str, Any]:
    clients_by_id = {str(item.get("id")): item for item in clients if item.get("id")}
    platform_name = platform_name or platform.system().lower()
    codex_state = _client_config_state(clients_by_id, "codex")
    claude_code_state = _client_config_state(clients_by_id, "claude_code")
    claude_desktop_state = _client_config_state(clients_by_id, "claude_desktop")
    claude_desktop_configured = claude_desktop_state == "configured"
    claude_host_diagnostics = clients_by_id.get("claude_desktop", {}).get("host_diagnostics") or {}
    claude_extension_state = claude_host_diagnostics.get("extension_installation_state")
    claude_content_state = claude_host_diagnostics.get("extension_content_state")
    claude_prepared_extension = claude_host_diagnostics.get("prepared_extension") or {}
    claude_prepared_state = claude_prepared_extension.get("state")
    claude_prepared_paths = claude_prepared_extension.get("package_paths")
    claude_prepared_path = (
        claude_prepared_paths[0]
        if isinstance(claude_prepared_paths, list)
        and len(claude_prepared_paths) == 1
        and isinstance(claude_prepared_paths[0], str)
        else None
    )
    codex_plugin_state = _codex_chatgpt_plugin_state()
    codex_desktop_mcp_state = _codex_desktop_mcp_state()
    if platform_name == "windows" and claude_content_state == "stale":
        claude_desktop_status = "extension_update_required"
        claude_desktop_reason = (
            "Installed Pith content differs from the validated staged MCPB; update it through Claude Settings. "
            "Opening the package or matching its version does not prove replacement; recheck installed content."
        )
    elif platform_name == "windows" and (
        claude_content_state == "unknown"
        or (claude_extension_state == "installed" and claude_content_state != "current")
    ):
        claude_desktop_status = "extension_content_unverified"
        claude_desktop_reason = (
            "Pith extension currency is unverified; check the validated staged package and installed files. "
            "Registry presence alone is not readiness."
        )
    elif claude_extension_state == "installed":
        claude_desktop_status = "extension_installed_requires_live_proof"
        claude_desktop_reason = (
            "Claude's extension registry contains Pith; a full app restart and same-turn host proof are still required."
        )
    elif claude_prepared_state == "prepared":
        claude_desktop_status = "extension_prepared_install_required"
        claude_desktop_reason = (
            "The Pith MCPB is prepared but Claude has not recorded it as installed; install it from Claude Settings."
        )
    else:
        claude_desktop_status = (
            "legacy_config_present_extension_missing" if claude_desktop_state == "configured" else "not_configured"
        )
        claude_desktop_reason = "Legacy Claude MCP JSON presence does not prove the current host loaded Pith; install the Pith MCPB extension."
    chatgpt_fields = _chatgpt_app_surface_fields(platform_name)
    codex_client = clients_by_id.get("codex", {})
    codex_mode = codex_client.get("codex_pith_configuration_mode")
    if codex_mode == "duplicate":
        codex_status = "duplicate_registration"
        codex_reason = "A legacy global Pith server shadows the Codex Work plugin registration."
    elif codex_state == "configured":
        codex_status = "configured_requires_lifecycle_proof"
        codex_reason = "Codex Work plugin configuration is present but still requires same-turn proof."
    else:
        codex_status = "not_configured"
        codex_reason = "Codex Work does not have one approval-enabled plugin-only Pith registration."
    rows = [
        _app_surface_row(
            "local_api_cli",
            "CLI / local API",
            "requires_current_connection_proof",
            "Local API health and status are support signals; connection requires current conversation_turn proof.",
            "pith api connection_proof --stdin-json",
            ["pith status", "pith health", "local API health"],
            False,
        ),
        _app_surface_row(
            "codex",
            "Codex",
            codex_status,
            codex_reason,
            "same-turn pith_conversation_turn with bind_status=bound and no auth_error",
            ["MCP config presence", "pith status", "pith health"],
            True,
            client_id="codex",
            config_state=codex_state,
            codex_pith_configuration_mode=codex_mode,
            codex_plugin_registration_configured=codex_client.get("codex_plugin_registration_configured"),
            legacy_global_registration_present=codex_client.get("legacy_global_registration_present"),
            duplicate_pith_registration=codex_client.get("duplicate_pith_registration"),
            **codex_desktop_mcp_state,
            **codex_plugin_state,
        ),
        _app_surface_row(
            "claude_code",
            "Claude Code",
            "configured_requires_model_visible_proof" if claude_code_state == "configured" else "not_configured",
            "Claude Code must show model-visible same-turn Pith fields before parity can be claimed.",
            "same-turn pith_conversation_turn with bind_status=bound and model-visible fields",
            ["MCP config presence", "pith_bridge_status", "pith status"],
            True,
            client_id="claude_code",
            config_state=claude_code_state,
        ),
        _app_surface_row(
            "claude_desktop",
            "Claude Desktop",
            claude_desktop_status,
            claude_desktop_reason,
            "fresh host session pith_connection_proof or same-turn pith_conversation_turn",
            ["MCP config presence", "pith_bridge_status", "pith status"],
            platform_name == "windows"
            and claude_desktop_status in {"extension_update_required", "extension_content_unverified"},
            client_id="claude_desktop",
            config_state=claude_desktop_state,
            config_conflict_state=claude_host_diagnostics.get("conflict_state"),
            host_loaded_status=claude_host_diagnostics.get("host_loaded_status"),
            extension_installation_state=claude_extension_state,
            extension_content_state=claude_content_state,
            extension_content=claude_host_diagnostics.get("extension_content"),
            pith_extension_installation_count=claude_host_diagnostics.get("pith_extension_installation_count"),
            prepared_extension_state=claude_prepared_state,
            prepared_extension_path=claude_prepared_path,
        ),
        _app_surface_row(
            "claude_chat",
            "Claude Chat",
            HOST_VARIANT_LOADER_UNVERIFIED_STATUS
            if claude_desktop_configured
            else HOST_VARIANT_LOADER_UNVERIFIED_NOT_CONFIGURED_STATUS,
            "Claude Chat loader exposure is not proven by Claude Desktop config and needs its own same-turn proof.",
            "fresh Claude Chat pith_connection_proof or same-turn pith_conversation_turn",
            ["MCP config presence", "pith_bridge_status", "pith status"],
            True,
            host_variant_of="claude_desktop",
            config_state=claude_desktop_state,
        ),
        _app_surface_row(
            "claude_cowork",
            "Claude Cowork",
            HOST_VARIANT_LOADER_UNVERIFIED_STATUS
            if claude_desktop_configured
            else HOST_VARIANT_LOADER_UNVERIFIED_NOT_CONFIGURED_STATUS,
            "Claude Cowork loader exposure is not proven by Claude Desktop config and needs its own same-turn proof.",
            "fresh Claude Cowork pith_connection_proof or same-turn pith_conversation_turn",
            ["MCP config presence", "pith_bridge_status", "pith status"],
            False,
            host_variant_of="claude_desktop",
            config_state=claude_desktop_state,
        ),
        _app_surface_row(
            "chatgpt",
            "ChatGPT",
            chatgpt_fields["status"],
            chatgpt_fields["status_reason"],
            "Install or expose a ChatGPT connector, then produce same-turn pith_conversation_turn proof.",
            ["local API health", "pith status", "MCP config presence"],
            True,
            process_detected=chatgpt_fields.get("process_detected"),
            process_count=chatgpt_fields.get("process_count"),
            process_paths=chatgpt_fields.get("process_paths"),
            local_mcp_supported=chatgpt_fields.get("local_mcp_supported"),
            local_plugin_surface=chatgpt_fields.get("local_plugin_surface"),
            transport_requirement=chatgpt_fields.get("transport_requirement"),
            tunnel_management_url=chatgpt_fields.get("tunnel_management_url"),
            connector_settings_url=chatgpt_fields.get("connector_settings_url"),
            developer_mode_url=chatgpt_fields.get("developer_mode_url"),
            session_active_source=chatgpt_fields.get("session_active_source"),
        ),
    ]
    blockers = [row["surface_id"] for row in rows if row["release_blocker_for_full_app_parity"]]
    return {
        "schema_version": APP_SURFACE_PARITY_SCHEMA_VERSION,
        "claimable": not blockers,
        "blockers": blockers,
        "surfaces": rows,
    }


def _text_has_pith_config(path: Path, marker: str) -> bool:
    if not path.is_file():
        return False
    try:
        return marker in path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False


def _codex_pith_config_state(path: Path) -> dict[str, Any]:
    """Classify Codex's Pith registration without accepting shadowed duplicates."""
    state: dict[str, Any] = {
        "pith_configured": False,
        "codex_config_parse_status": "missing",
        "codex_pith_configuration_mode": "absent",
        "codex_plugin_registration_configured": False,
        "legacy_global_registration_present": False,
        "duplicate_pith_registration": False,
    }
    if not path.is_file():
        return state
    if tomllib is None:
        state["codex_config_parse_status"] = "parser_unavailable"
        return state
    try:
        text = path.read_text(encoding="utf-8-sig")
        payload = tomllib.loads(text)
    except (OSError, UnicodeError, tomllib.TOMLDecodeError):
        state["codex_config_parse_status"] = "invalid"
        state["codex_pith_configuration_mode"] = "invalid"
        return state

    state["codex_config_parse_status"] = "valid"
    mcp_servers = payload.get("mcp_servers")
    legacy = isinstance(mcp_servers, dict) and isinstance(mcp_servers.get("pith"), dict)
    plugins = payload.get("plugins")
    plugin_package = plugins.get("pith@personal") if isinstance(plugins, dict) else None
    plugin_servers = plugin_package.get("mcp_servers") if isinstance(plugin_package, dict) else None
    plugin = plugin_servers.get("pith") if isinstance(plugin_servers, dict) else None
    plugin_configured = (
        isinstance(plugin, dict)
        and plugin.get("enabled") is True
        and plugin.get("default_tools_approval_mode") == CODEX_PLUGIN_DEFAULT_TOOLS_APPROVAL_MODE
    )
    duplicate = legacy and isinstance(plugin, dict)
    if duplicate:
        mode = "duplicate"
    elif plugin_configured:
        mode = "plugin_only"
    elif isinstance(plugin, dict):
        mode = "plugin_preferences_incomplete"
    elif legacy:
        mode = "legacy_global_only"
    else:
        mode = "absent"
    state.update(
        {
            "pith_configured": plugin_configured and not legacy,
            "codex_pith_configuration_mode": mode,
            "codex_plugin_registration_configured": plugin_configured,
            "legacy_global_registration_present": legacy,
            "duplicate_pith_registration": duplicate,
        }
    )
    return state


def collect_clients_status() -> dict[str, Any]:
    configure_clients = _load_configure_clients()
    plat = configure_clients._detect_platform() if configure_clients else platform.system().lower()
    clients: list[dict[str, Any]] = []
    if configure_clients:
        registry = dict(getattr(configure_clients, "CLIENT_REGISTRY", {}))
        for client_id in PRODUCTION_CLIENT_SURFACE_ORDER:
            if client_id == "vscode":
                vscode = getattr(configure_clients, "VSCODE_CONFIG", {})
                if not vscode:
                    continue
                config_path = _select_report_config_path(vscode["config_file"].get(plat, ""), plat, configure_clients)
                clients.append(
                    {
                        "id": "vscode",
                        "label": vscode.get("label", "VS Code"),
                        "detected": _any_path_is_dir(vscode["detect_dirs"].get(plat, ""), plat, configure_clients)
                        or config_path.is_file(),
                        "config_exists": config_path.is_file(),
                        "pith_configured": _json_has_pith_config(config_path),
                        "config_path": str(config_path),
                        "release_validation": _client_release_validation("vscode"),
                    }
                )
                continue
            if client_id == "codex":
                codex = getattr(configure_clients, "CODEX_CONFIG", {})
                if not codex:
                    continue
                config_path = _select_report_config_path(codex["config_file"].get(plat, ""), plat, configure_clients)
                codex_config_state = _codex_pith_config_state(config_path)
                clients.append(
                    {
                        "id": "codex",
                        "label": codex.get("label", "Codex"),
                        "detected": _any_path_is_dir(codex["detect_dirs"].get(plat, ""), plat, configure_clients),
                        "config_exists": config_path.is_file(),
                        "config_path": str(config_path),
                        "release_validation": _client_release_validation("codex"),
                        **codex_config_state,
                    }
                )
                continue
            info = registry.get(client_id)
            if not info:
                continue
            config_path = _select_report_config_path(info["config_file"].get(plat, ""), plat, configure_clients)
            clients.append(
                {
                    "id": client_id,
                    "label": info.get("label", client_id),
                    "detected": _any_path_is_dir(info["detect_dirs"].get(plat, ""), plat, configure_clients),
                    "config_exists": config_path.is_file(),
                    "pith_configured": _json_has_pith_config(config_path),
                    "config_path": str(config_path),
                    "release_validation": _client_release_validation(client_id),
                }
            )
    claude_diagnostics: dict[str, Any] = {
        "diagnostic_status": "unavailable",
        "host_loaded_status": "not_observed",
        "connected_status": "not_proven",
    }
    claude_client = next((item for item in clients if item.get("id") == "claude_desktop"), None)
    if claude_client:
        process_paths = None
        process_probe_error = None
        if plat == "windows":
            process_state = _windows_named_process_state("Claude")
            if "error" in process_state:
                process_probe_error = str(process_state["error"])
            else:
                process_paths = process_state.get("process_paths")
        try:
            claude_diagnostics = collect_claude_host_diagnostics(
                platform_name=plat,
                selected_config_path=claude_client.get("config_path"),
                process_paths=process_paths,
                process_probe_error=process_probe_error,
            )
        except Exception:
            claude_diagnostics = {
                "schema_version": "pith_claude_host_diagnostics.v1",
                "diagnostic_status": "unavailable",
                "host_loaded_status": "not_observed",
                "connected_status": "not_proven",
            }
        claude_client["host_diagnostics"] = claude_diagnostics
        extension_installed = claude_diagnostics.get("extension_installation_state") == "installed"
        claude_client["pith_extension_installed"] = extension_installed
        claude_client["pith_extension_installation_count"] = claude_diagnostics.get(
            "pith_extension_installation_count", 0
        )
        claude_client["prepared_extension_state"] = (claude_diagnostics.get("prepared_extension") or {}).get("state")
        if plat == "windows":
            claude_client["pith_configured"] = (
                extension_installed and claude_diagnostics.get("extension_content_state") == "current"
            )
        elif extension_installed:
            claude_client["pith_configured"] = True
    configured = sum(1 for item in clients if item["pith_configured"])
    detected = sum(1 for item in clients if item["detected"])
    app_surface_parity = _apply_app_surface_proof(
        _build_app_surface_parity(clients, plat),
        _collect_app_surface_proof_validation(),
    )
    return {
        "platform": plat,
        "detected_count": detected,
        "configured_count": configured,
        "clients": clients,
        "claude_host_diagnostics": claude_diagnostics,
        "app_surface_parity": app_surface_parity,
    }


def _health_label(health: dict[str, Any], readyz: dict[str, Any]) -> str:
    if health.get("ok") is False and health.get("service") != "pith":
        return "Unreachable"
    if health.get("service") == "pith" and health.get("status") != "unhealthy":
        return "OK (Pith)"
    if readyz.get("service") == "pith" and readyz.get("mode") == "ready":
        return "OK (Pith)"
    if health.get("ok") is False or readyz.get("ok") is False:
        return "Unreachable"
    return "Port responding but NOT Pith"


def collect_service_status(base_url: str, timeout: float) -> dict[str, Any]:
    pith_home = _pith_home()
    pid_file = _pid_file_status(pith_home)
    launchd = _launchd_status()
    port = _port_from_base_url(base_url)
    port_state = _port_status(port)
    health = _fetch_json(base_url, "/health", timeout)
    readyz = _fetch_json(base_url, "/readyz", timeout)

    health_is_pith = health.get("service") == "pith" and health.get("status") != "unhealthy"
    readyz_is_ready = readyz.get("service") == "pith" and (
        readyz.get("mode") == "ready"
        or readyz.get("process_state") == "running"
        or readyz.get("status") in {"healthy", "ok"}
    )
    launchd_running = bool(launchd.get("running"))
    pid_running = bool(pid_file.get("running"))
    service_reachable = bool(readyz_is_ready or health_is_pith)
    process_alive_unreachable = bool(pid_running and not service_reachable)
    running = service_reachable or launchd_running

    pid = pid_file.get("pid") if pid_running else None
    pid_source = "pid_file" if pid_running else None
    if not pid and launchd_running:
        pid = launchd.get("pid")
        pid_source = "launchd"
    if not pid and port_state.get("listening") and (health_is_pith or readyz_is_ready):
        pid = port_state.get("pid")
        pid_source = "port"

    if process_alive_unreachable:
        state = "process_alive_health_unreachable"
    elif not running:
        state = "not_running"
    elif pid_source == "launchd":
        state = "running"
    elif pid_file.get("state") in {"missing", "stale", "invalid"} and pid_source != "pid_file":
        state = f"running_pid_file_{pid_file.get('state')}"
    else:
        state = "running"

    return {
        "generated_at": _now(),
        "base_url": base_url,
        "port": port,
        "running": running,
        "state": state,
        "service_reachable": service_reachable,
        "process_alive_unreachable": process_alive_unreachable,
        "pid": pid,
        "pid_source": pid_source,
        "health_label": _health_label(health, readyz),
        "pid_file": pid_file,
        "launchd": launchd,
        "port_check": port_state,
        "health": health,
        "readyz": readyz,
    }


def _db_path() -> Path:
    return _data_dir() / "pith.db"


def _db_summary() -> dict[str, Any]:
    path = _db_path()
    summary: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return summary
    summary["size_bytes"] = path.stat().st_size
    try:
        with sqlite3.connect(path) as conn:
            summary["concepts"] = conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]
            summary["journal_mode"] = conn.execute("PRAGMA journal_mode").fetchone()[0]
    except Exception as exc:
        summary["error"] = redact_text(exc)
    return summary


def _runtime_summary() -> dict[str, Any]:
    path = Path(os.environ.get("PITH_RUNTIME_META", str(_pith_home() / "config" / "python-runtime.json"))).expanduser()
    if not path.is_file():
        return {"path": str(path), "exists": False}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        return {"path": str(path), "exists": True, "error": redact_text(exc)}
    keys = ("managed_by", "runtime_id", "python_executable", "source", "sha256")
    return {"path": str(path), "exists": True, **{key: payload.get(key) for key in keys}}


def collect_report_status(base_url: str, timeout: float) -> dict[str, Any]:
    pith_home = _pith_home()
    server_path = _pith_server_path()
    venv_python, venv_error = _diagnostic_venv_python()
    py_version = _run([str(venv_python), "--version"], timeout=2.0) if venv_python and venv_python.is_file() else None
    clients = collect_clients_status()
    return {
        "generated_at": _now(),
        "system": {
            "os": f"{platform.system()} {platform.release()} {platform.machine()}",
            "shell": os.environ.get("SHELL", ""),
            "python": (py_version.stdout or py_version.stderr).strip() if py_version else "not found",
            "python_exe": str(venv_python) if venv_python else None,
            "python_resolution_error": venv_error,
            "disk_free_bytes": shutil.disk_usage("/").free,
        },
        "installation": {
            "pith_home": str(pith_home),
            "server_path": str(server_path),
            "version": os.environ.get("PITH_VERSION", "unknown"),
            "runtime": _runtime_summary(),
        },
        "server": collect_service_status(base_url, timeout),
        "database": _db_summary(),
        "clients": clients,
        "backups": _backup_summary(),
        "trust_health": _trust_health_summary(timeout),
    }


def _trust_health_summary(timeout: float) -> dict[str, Any]:
    try:
        from scripts import trust_health_status

        args = argparse.Namespace(
            compact_log_path=str(trust_health_status.DEFAULT_COMPACT_LOG_PATH),
            supersession_compact_log_path=str(trust_health_status.DEFAULT_SUPERSESSION_COMPACT_LOG_PATH),
            reports_dir=str(trust_health_status.DEFAULT_REPORTS_DIR),
            health_url=trust_health_status.DEFAULT_HEALTH_URL,
            history_limit=trust_health_status.DEFAULT_HISTORY_LIMIT,
            warn_after_seconds=trust_health_status.DEFAULT_WARN_AFTER_SECONDS,
            critical_after_seconds=trust_health_status.DEFAULT_CRITICAL_AFTER_SECONDS,
            max_log_bytes=trust_health_status.DEFAULT_MAX_LOG_BYTES,
            no_runtime_health=timeout <= 0,
            no_scheduler=False,
            no_supersession_scheduler=False,
        )
        return trust_health_status.build_status(args)
    except Exception as exc:
        return {
            "schema_version": "trust_health_status.unavailable.v0",
            "status": "UNKNOWN",
            "error": redact_text(str(exc)),
        }


def _backup_summary() -> dict[str, Any]:
    backup_dir = _data_dir() / "backups"
    if not backup_dir.is_dir():
        return {"directory": str(backup_dir), "exists": False, "count": 0}
    backups = sorted(backup_dir.glob("pith_backup_*.db"), reverse=True)
    return {
        "directory": str(backup_dir),
        "exists": True,
        "count": len(backups),
        "latest": str(backups[0]) if backups else None,
    }


def _client_state(item: dict[str, Any]) -> str:
    if item.get("pith_configured"):
        return "configured"
    if item.get("detected") or item.get("config_exists"):
        return "present (pith server not found)"
    return "not found"


def _format_client_line(item: dict[str, Any]) -> str:
    label = str(item.get("label") or item.get("id") or "unknown")
    padding = " " * max(1, 16 - len(label))
    validation = item.get("release_validation") if isinstance(item.get("release_validation"), dict) else {}
    suffix = ""
    if validation.get("production_validated"):
        suffix = f" [production validated: {validation.get('evidence')}]"
    return f"  {label}:{padding}{_client_state(item)}{suffix}"


def format_status(data: dict[str, Any]) -> str:
    lines: list[str] = []
    if data.get("state") == "process_alive_health_unreachable":
        pid = data.get("pid")
        if pid:
            lines.append(f"Pith process exists but health is unreachable (PID: {pid})")
        else:
            lines.append("Pith process exists but health is unreachable")
        lines.append(f"Health: {data['health_label']}")
    elif data["running"]:
        pid = data.get("pid")
        suffix = ""
        if data.get("state") == "running_pid_file_missing":
            suffix = " [PID file missing; verified by service checks]"
        elif data.get("state") == "running_pid_file_stale":
            suffix = " [stale PID file; verified by service checks]"
        elif data.get("state") == "running_pid_file_invalid":
            suffix = " [invalid PID file; verified by service checks]"
        if pid:
            lines.append(f"Pith is running (PID: {pid}){suffix}")
        else:
            lines.append(f"Pith is running{suffix}")
        lines.append(f"Health: {data['health_label']}")
    else:
        lines.append("Pith is not running")
        lines.append(f"Health: {data['health_label']}")
    return "\n".join(lines)


def _format_app_surface_parity_lines(app_surface_parity: Any) -> list[str]:
    if not isinstance(app_surface_parity, dict):
        return [
            "",
            "[App Surface Parity]",
            "  Claimable:    no",
            "  Blockers:     unavailable",
            "  Status:       unavailable",
        ]
    app_surface_parity = _redact_obj(app_surface_parity)

    blockers = app_surface_parity.get("blockers")
    blocker_text = ", ".join(str(item) for item in blockers) if isinstance(blockers, list) and blockers else "none"
    surfaces = app_surface_parity.get("surfaces")
    claimable = app_surface_parity.get("claimable") is True and isinstance(surfaces, list)
    lines = [
        "",
        "[App Surface Parity]",
        f"  Claimable:    {'yes' if claimable else 'no'}",
        f"  Blockers:     {blocker_text}",
    ]
    if not isinstance(surfaces, list):
        lines.append("  Status:       unavailable")
        return lines

    for item in surfaces:
        if not isinstance(item, dict):
            continue
        label = item.get("label") or item.get("surface_id") or "unknown"
        status = item.get("status") or "unknown"
        reason = item.get("status_reason") or ""
        suffix = f" - {reason}" if reason else ""
        lines.append(f"  {label}: {status}{suffix}")

        if item.get("surface_id") == "chatgpt" and item.get("transport_requirement"):
            connector_url = item.get("connector_settings_url")
            if connector_url:
                lines.append(f"    Action: configure remote MCP or Secure MCP Tunnel at {connector_url}")
        if item.get("surface_id") == "claude_desktop" and status in {
            "extension_prepared_install_required",
            "extension_update_required",
            "extension_content_unverified",
        }:
            path = item.get("prepared_extension_path")
            if path:
                lines.append(f"    Extension: {path}")
    return lines


def format_report(data: dict[str, Any]) -> str:
    system = data["system"]
    install = data["installation"]
    server = data["server"]
    db = data["database"]
    backups = data["backups"]
    trust_health = data.get("trust_health") if isinstance(data.get("trust_health"), dict) else {}
    lines = [
        "Pith Diagnostics Report",
        "==============================",
        f"Generated: {data['generated_at']}",
        "",
        "[System]",
        f"  OS:           {system['os']}",
        f"  Shell:        {system['shell']}",
        f"  Python:       {system['python']}",
        f"  Python exe:   {system['python_exe']}",
        f"  Disk Free:    {system['disk_free_bytes'] // (1024 * 1024)} MB",
        "",
        "[Installation]",
        f"  Pith Home:    {install['pith_home']}",
        f"  Version:      {install['version']}",
        f"  Server Path:  {install['server_path']}",
    ]
    runtime = install["runtime"]
    if runtime.get("exists"):
        lines.extend(
            [
                f"  Runtime:      {runtime.get('managed_by') or 'unknown'}",
                f"  Runtime ID:   {runtime.get('runtime_id') or 'unknown'}",
                f"  Runtime exe:  {runtime.get('python_executable') or 'unknown'}",
                f"  Runtime src:  {runtime.get('source') or 'unknown'}",
                f"  Runtime sha:  {runtime.get('sha256') or 'unknown'}",
            ]
        )
    else:
        lines.append("  Runtime:      unknown (no python-runtime.json)")
    lines.extend(["", "[Server]"])
    if server["running"]:
        pid = f" (PID {server['pid']})" if server.get("pid") else ""
        lines.append(f"  Status:       Running{pid}")
    else:
        lines.append("  Status:       Not running")
    lines.extend(
        [
            f"  Port:         {server['port']}",
            f"  Health:       {server['health_label']}",
            f"  Ready:        {server['readyz'].get('mode', 'unknown')}",
            "",
            "[Database]",
        ]
    )
    if db["exists"]:
        lines.append(f"  Path:         {db['path']}")
        lines.append(f"  Size:         {int(db.get('size_bytes', 0)) // 1024} KB")
        if "concepts" in db:
            lines.append(f"  Concepts:     {db['concepts']}")
        if "journal_mode" in db:
            lines.append(f"  Journal:      {db['journal_mode']}")
        if "error" in db:
            lines.append(f"  Error:        {db['error']}")
    else:
        lines.append(f"  Path:         {db['path']} (not created yet)")
    validated = [
        item.get("label") or item.get("id")
        for item in data["clients"]["clients"]
        if isinstance(item.get("release_validation"), dict) and item["release_validation"].get("production_validated")
    ]
    lines.extend(["", "[Client Surfaces]"])
    lines.append(f"  Production validated: {', '.join(validated) if validated else 'none'}")
    lines.extend(_format_client_line(item) for item in data["clients"]["clients"])
    app_surface_start = len(lines)
    lines.extend(_format_app_surface_parity_lines(data.get("clients", {}).get("app_surface_parity")))
    app_surface_end = len(lines)
    lines.extend(
        [
            "",
            "[Backups]",
            f"  Directory:    {backups['directory']}",
            f"  Count:        {backups['count']}",
        ]
    )
    if backups.get("latest"):
        lines.append(f"  Latest:       {backups['latest']}")
    latest_run = trust_health.get("latest_run") if isinstance(trust_health.get("latest_run"), dict) else {}
    latest_metrics = latest_run.get("metrics") if isinstance(latest_run.get("metrics"), dict) else {}
    freshness = trust_health.get("freshness") if isinstance(trust_health.get("freshness"), dict) else {}
    evidence = trust_health.get("evidence") if isinstance(trust_health.get("evidence"), dict) else {}
    supersession = (
        trust_health.get("supersession_edges") if isinstance(trust_health.get("supersession_edges"), dict) else {}
    )
    supersession_latest = supersession.get("latest_run") if isinstance(supersession.get("latest_run"), dict) else {}
    supersession_risks = (
        supersession_latest.get("risk_counts") if isinstance(supersession_latest.get("risk_counts"), dict) else {}
    )
    lines.extend(
        [
            "",
            "[Trust Health]",
            f"  Status:       {trust_health.get('overall_user_status', trust_health.get('status', 'UNKNOWN'))}",
            f"  Freshness:    {freshness.get('status', 'unknown')}",
            f"  Latest run:   {latest_run.get('timestamp') or 'unknown'}",
            f"  Alarms:       {len(latest_run.get('alarms') or [])}",
            f"  False gov:    {latest_metrics.get('false_governance_rate', 'unknown')}",
            f"  False abstain:{latest_metrics.get('false_abstention_rate', 'unknown')}",
            f"  Report:       {evidence.get('latest_report_path') or latest_run.get('report_path') or 'unknown'}",
            f"  Supersession: {supersession.get('status', 'NO_EVIDENCE')}",
            f"  Answer risk:  {supersession_risks.get('answer_governance_risk', 'unknown')}",
            f"  Missing id:   {supersession_risks.get('missing_identity_edges', 'unknown')}",
            f"  Manual review:{supersession_risks.get('manual_review_required', 'unknown')}",
        ]
    )
    meaning = trust_health.get("user_meaning")
    if meaning:
        lines.append(f"  Meaning:      {meaning}")
    try:
        from scripts import trust_health_status

        lines.extend(trust_health_status.format_alarm_summary_lines(trust_health, indent="  "))
    except Exception:
        pass
    next_steps = trust_health.get("user_next_steps") if isinstance(trust_health.get("user_next_steps"), list) else []
    if next_steps:
        lines.append("  Next:")
        for index, step in enumerate(next_steps, start=1):
            lines.append(f"    {index}. {step}")
    if trust_health.get("error"):
        lines.append(f"  Error:        {trust_health['error']}")
    return "\n".join(
        line if app_surface_start <= index < app_surface_end else redact_text(line) for index, line in enumerate(lines)
    )


def collect_doctor_status(base_url: str, timeout: float) -> dict[str, Any]:
    pith_home = _pith_home()
    server_path = _pith_server_path()
    venv_python, venv_error = _diagnostic_venv_python()
    data_dir = _data_dir()
    db_path = data_dir / "pith.db"
    health = _fetch_json(base_url, "/health", timeout)
    readyz = _fetch_json(base_url, "/readyz", timeout)
    clients = collect_clients_status()
    checks = {
        "pith_home_exists": pith_home.is_dir(),
        "server_path_exists": server_path.is_dir(),
        "venv_python_exists": venv_python is not None and venv_python.is_file(),
        "data_dir_exists": data_dir.is_dir(),
        "db_exists": db_path.is_file(),
        "api_key_configured": bool(os.environ.get("PITH_API_KEY") or (pith_home / ".env").is_file()),
        "http_health_ok": health.get("status") != "unhealthy" and health.get("ok", True) is not False,
        "http_ready": readyz.get("mode") == "ready",
    }
    return {
        "generated_at": _now(),
        "base_url": base_url,
        "paths": {
            "pith_home": str(pith_home),
            "server_path": str(server_path),
            "data_dir": str(data_dir),
            "db_path": str(db_path),
            "venv_python": str(venv_python) if venv_python else None,
        },
        "venv_resolution_error": venv_error,
        "checks": checks,
        "health": health,
        "readyz": readyz,
        "clients_summary": {
            "detected_count": clients["detected_count"],
            "configured_count": clients["configured_count"],
        },
        "status": "ok" if all(checks.values()) else "warn",
    }


def _log_paths() -> dict[str, Path]:
    return {"pith": _pith_home() / "logs" / "pith.log", "err": _pith_home() / "logs" / "pith.err"}


def _tail_redacted(path: Path, lines: int) -> list[str]:
    if not path.exists():
        return [f"<missing: {path}>"]
    return [redact_text(line) for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]]


def _safe_bundle_path(output: str | None) -> Path:
    if output:
        path = Path(output).expanduser()
    else:
        stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        path = _pith_home() / "diagnostics" / "support-bundles" / f"pith-support-{stamp}.zip"
    if path.suffix != ".zip":
        path = path.with_suffix(".zip")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _support_env_snapshot() -> dict[str, Any]:
    allowlist = [
        "PITH_HOME",
        "PITH_PROFILE",
        "PITH_PORT",
        "PITH_API_URL",
        "PITH_DATA_DIR",
        "PITH_LAUNCH_AGENTS_DIR",
    ]
    secret_presence = sorted(key for key in os.environ if SECRET_KEY_RE.search(key))
    return {
        "allowlisted": {key: redact_text(os.environ.get(key, "")) for key in allowlist if key in os.environ},
        "secret_keys_present": secret_presence,
    }


def build_support_bundle(output: str | None, base_url: str, timeout: float, lines: int) -> dict[str, Any]:
    bundle_path = _safe_bundle_path(output)
    doctor = collect_doctor_status(base_url, timeout)
    clients = collect_clients_status()
    logs = {name: _tail_redacted(path, lines) for name, path in _log_paths().items()}
    manifest = {
        "bundle_version": SUPPORT_BUNDLE_VERSION,
        "generated_at": _now(),
        "redaction": "secret-like values redacted; raw concepts/conversations/API payloads intentionally excluded",
        "files": ["manifest.json", "doctor.json", "clients.json", "env.json", "logs/pith.log", "logs/pith.err"],
    }
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
        archive.writestr("doctor.json", json.dumps(_redact_obj(doctor), indent=2, sort_keys=True, default=str))
        archive.writestr("clients.json", json.dumps(_redact_obj(clients), indent=2, sort_keys=True, default=str))
        archive.writestr("env.json", json.dumps(_redact_obj(_support_env_snapshot()), indent=2, sort_keys=True))
        archive.writestr("logs/pith.log", "\n".join(logs["pith"]) + "\n")
        archive.writestr("logs/pith.err", "\n".join(logs["err"]) + "\n")
    return {"path": str(bundle_path), "manifest": manifest}


def cmd_doctor(args: argparse.Namespace) -> int:
    data = collect_doctor_status(_base_url(args.base_url), args.timeout)
    if args.json:
        _print_json(data)
    else:
        print(f"Pith doctor: {data['status'].upper()}")
        for name, ok in data["checks"].items():
            print(f"  {'ok' if ok else 'warn'} {name}")
        print(
            f"  clients: {data['clients_summary']['configured_count']}/"
            f"{data['clients_summary']['detected_count']} detected configured"
        )
    return 0 if data["status"] == "ok" else 1


def cmd_clients(args: argparse.Namespace) -> int:
    data = collect_clients_status()
    if args.json:
        _print_json(data)
    else:
        print(f"Clients: {data['configured_count']}/{data['detected_count']} detected configured")
        for item in data["clients"]:
            print(
                f"- {item['label']}: detected={'yes' if item['detected'] else 'no'} "
                f"configured={'yes' if item['pith_configured'] else 'no'}"
            )
        print("\n".join(_format_app_surface_parity_lines(data.get("app_surface_parity"))))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    data = collect_service_status(_base_url(args.base_url), args.timeout)
    if args.json:
        _print_json(data)
    else:
        print(format_status(data))
    return 0 if data["running"] else 1


def cmd_report(args: argparse.Namespace) -> int:
    data = collect_report_status(_base_url(args.base_url), args.timeout)
    if args.json:
        _print_json(data)
    else:
        print(format_report(data))
    return 0


def cmd_support_bundle(args: argparse.Namespace) -> int:
    data = build_support_bundle(args.output, _base_url(args.base_url), args.timeout, args.lines)
    if args.json:
        _print_json(data)
    else:
        print(f"Support bundle: {data['path']}")
        print("  redaction: enabled")
        print("  excluded: raw concepts, conversations, provider payloads")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pith", description="Pith support commands")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="Run read-only install and service diagnostics")
    doctor.add_argument("--json", action="store_true")
    doctor.add_argument("--base-url")
    doctor.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    doctor.set_defaults(func=cmd_doctor)

    clients = sub.add_parser("clients", help="Show detected and configured client surfaces")
    clients.add_argument("clients_command", nargs="?", default="status", choices=["status", "list"])
    clients.add_argument("--json", action="store_true")
    clients.set_defaults(func=cmd_clients)

    status = sub.add_parser("status", help="Show service status from PID, launchd, port, health, and readiness checks")
    status.add_argument("--json", action="store_true")
    status.add_argument("--base-url")
    status.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    status.set_defaults(func=cmd_status)

    report = sub.add_parser("report", help="Generate a redacted diagnostics report")
    report.add_argument("--json", action="store_true")
    report.add_argument("--base-url")
    report.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    report.set_defaults(func=cmd_report)

    support = sub.add_parser("support", help="Create redacted support artifacts")
    support_sub = support.add_subparsers(dest="support_command", required=True)
    bundle = support_sub.add_parser("bundle", help="Create a redacted support bundle zip")
    bundle.add_argument("--output")
    bundle.add_argument("--json", action="store_true")
    bundle.add_argument("--base-url")
    bundle.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    bundle.add_argument("--lines", type=int, default=80)
    bundle.set_defaults(func=cmd_support_bundle)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except RuntimeError as exc:
        print(f"Error: {redact_text(exc)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
