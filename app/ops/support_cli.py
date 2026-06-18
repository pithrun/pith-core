"""Operational support CLI commands for Pith.

These commands are intentionally read-only except for `support bundle`, which
writes a bounded, redacted diagnostic archive.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import os
import platform
import re
import shutil
import sqlite3
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


DEFAULT_TIMEOUT = 5.0
SUPPORT_BUNDLE_VERSION = 1
LEGACY_SERVER_NAMES = {"pith", "pith-mcp", "pith-mcp-wrapper"}
LAUNCHD_LABEL = "dev.pith.server"
PRODUCTION_VALIDATED_CLIENT_SURFACES = {
    "claude_desktop": "mcp_runtime_verified",
    "codex": "api_lifecycle_verified",
}
PRODUCTION_CLIENT_SURFACE_ORDER = (
    "claude_desktop",
    "claude_code",
    "vscode",
    "cursor",
    "codex",
)
SECRET_KEY_RE = re.compile(r"(api[_-]?key|token|secret|password|private[_-]?key|access[_-]?key|auth)", re.I)
ENV_ASSIGNMENT_RE = re.compile(r"\b([A-Z0-9_]*(?:API_KEY|TOKEN|SECRET|PASSWORD|PRIVATE_KEY|ACCESS_KEY|AUTH)[A-Z0-9_]*)=(\"[^\"]*\"|'[^']*'|\S+)")
JSON_SECRET_RE = re.compile(
    r'("?[A-Za-z0-9_.-]*(?:api[_-]?key|token|secret|password|private[_-]?key|access[_-]?key|auth)[A-Za-z0-9_.-]*"?\s*[:=]\s*)("[^"]*"|\'[^\']*\'|[^\s,}]+)',
    re.I,
)
HEADER_SECRET_RE = re.compile(r"((?:X-API-Key|Authorization):\s*)(Bearer\s+)?\S+", re.I)
LONG_TOKEN_RE = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{12,}|[A-Za-z0-9_-]{40,})\b")


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _pith_home() -> Path:
    return Path(os.environ.get("PITH_HOME", str(Path.home() / ".pith"))).expanduser()


def _pith_server_path() -> Path:
    return Path(os.environ.get("PITH_SERVER_PATH", str(_pith_home() / "pith-server"))).expanduser()


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
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _process_command(pid: int | None) -> str | None:
    if not pid:
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
            if SECRET_KEY_RE.search(str(key)):
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


def _expand_config_path(path_value: str, plat: str, configure_clients: Any | None) -> Path:
    if configure_clients and hasattr(configure_clients, "_expand"):
        return Path(configure_clients._expand(path_value, plat)).expanduser()
    return Path(os.path.expanduser(path_value))


def _json_has_pith_config(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    roots = [payload.get("mcpServers"), payload.get("mcp_servers"), payload.get("servers")]
    for root in roots:
        if isinstance(root, dict) and any(name in root for name in LEGACY_SERVER_NAMES):
            return True
    return False


def _client_release_validation(client_id: str) -> dict[str, Any]:
    evidence = PRODUCTION_VALIDATED_CLIENT_SURFACES.get(client_id)
    return {
        "production_validated": evidence is not None,
        "evidence": evidence,
        "claim": "validated_production_surface" if evidence else "configuration_surface_only",
    }


def _text_has_pith_config(path: Path, marker: str) -> bool:
    if not path.is_file():
        return False
    try:
        return marker in path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False


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
                config_path = _expand_config_path(vscode["config_file"].get(plat, ""), plat, configure_clients)
                detect_path = _expand_config_path(vscode["detect_dirs"].get(plat, ""), plat, configure_clients)
                clients.append({
                    "id": "vscode",
                    "label": vscode.get("label", "VS Code"),
                    "detected": detect_path.is_dir() or config_path.is_file(),
                    "config_exists": config_path.is_file(),
                    "pith_configured": _json_has_pith_config(config_path),
                    "config_path": str(config_path),
                    "release_validation": _client_release_validation("vscode"),
                })
                continue
            if client_id == "codex":
                codex = getattr(configure_clients, "CODEX_CONFIG", {})
                if not codex:
                    continue
                config_path = _expand_config_path(codex["config_file"].get(plat, ""), plat, configure_clients)
                detect_path = _expand_config_path(codex["detect_dirs"].get(plat, ""), plat, configure_clients)
                clients.append({
                    "id": "codex",
                    "label": codex.get("label", "Codex"),
                    "detected": detect_path.is_dir(),
                    "config_exists": config_path.is_file(),
                    "pith_configured": _text_has_pith_config(config_path, "[mcp_servers.pith]"),
                    "config_path": str(config_path),
                    "release_validation": _client_release_validation("codex"),
                })
                continue
            info = registry.get(client_id)
            if not info:
                continue
            config_path = _expand_config_path(info["config_file"].get(plat, ""), plat, configure_clients)
            detect_path = _expand_config_path(info["detect_dirs"].get(plat, ""), plat, configure_clients)
            clients.append({
                "id": client_id,
                "label": info.get("label", client_id),
                "detected": detect_path.is_dir(),
                "config_exists": config_path.is_file(),
                "pith_configured": _json_has_pith_config(config_path),
                "config_path": str(config_path),
                "release_validation": _client_release_validation(client_id),
            })
    configured = sum(1 for item in clients if item["pith_configured"])
    detected = sum(1 for item in clients if item["detected"])
    return {"platform": plat, "detected_count": detected, "configured_count": configured, "clients": clients}


def _health_label(health: dict[str, Any], readyz: dict[str, Any]) -> str:
    if health.get("ok") is False and not (health.get("service") == "pith"):
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
    running = pid_running or readyz_is_ready or health_is_pith or launchd_running

    pid = pid_file.get("pid") if pid_running else None
    pid_source = "pid_file" if pid_running else None
    if not pid and launchd_running:
        pid = launchd.get("pid")
        pid_source = "launchd"
    if not pid and port_state.get("listening") and (health_is_pith or readyz_is_ready):
        pid = port_state.get("pid")
        pid_source = "port"

    if not running:
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
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"path": str(path), "exists": True, "error": redact_text(exc)}
    keys = ("managed_by", "runtime_id", "python_executable", "source", "sha256")
    return {"path": str(path), "exists": True, **{key: payload.get(key) for key in keys}}


def collect_report_status(base_url: str, timeout: float) -> dict[str, Any]:
    pith_home = _pith_home()
    server_path = _pith_server_path()
    venv_python = pith_home / "venv" / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python3")
    py_version = _run([str(venv_python), "--version"], timeout=2.0)
    clients = collect_clients_status()
    return {
        "generated_at": _now(),
        "system": {
            "os": f"{platform.system()} {platform.release()} {platform.machine()}",
            "shell": os.environ.get("SHELL", ""),
            "python": (py_version.stdout or py_version.stderr).strip() if py_version else "not found",
            "python_exe": str(venv_python),
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
    if data["running"]:
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


def format_report(data: dict[str, Any]) -> str:
    system = data["system"]
    install = data["installation"]
    server = data["server"]
    db = data["database"]
    backups = data["backups"]
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
        lines.extend([
            f"  Runtime:      {runtime.get('managed_by') or 'unknown'}",
            f"  Runtime ID:   {runtime.get('runtime_id') or 'unknown'}",
            f"  Runtime exe:  {runtime.get('python_executable') or 'unknown'}",
            f"  Runtime src:  {runtime.get('source') or 'unknown'}",
            f"  Runtime sha:  {runtime.get('sha256') or 'unknown'}",
        ])
    else:
        lines.append("  Runtime:      unknown (no python-runtime.json)")
    lines.extend(["", "[Server]"])
    if server["running"]:
        pid = f" (PID {server['pid']})" if server.get("pid") else ""
        lines.append(f"  Status:       Running{pid}")
    else:
        lines.append("  Status:       Not running")
    lines.extend([
        f"  Port:         {server['port']}",
        f"  Health:       {server['health_label']}",
        f"  Ready:        {server['readyz'].get('mode', 'unknown')}",
        "",
        "[Database]",
    ])
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
        if isinstance(item.get("release_validation"), dict)
        and item["release_validation"].get("production_validated")
    ]
    lines.extend(["", "[Client Surfaces]"])
    lines.append(f"  Production validated: {', '.join(validated) if validated else 'none'}")
    lines.extend(_format_client_line(item) for item in data["clients"]["clients"])
    lines.extend([
        "",
        "[Backups]",
        f"  Directory:    {backups['directory']}",
        f"  Count:        {backups['count']}",
    ])
    if backups.get("latest"):
        lines.append(f"  Latest:       {backups['latest']}")
    return "\n".join(redact_text(line) for line in lines)


def collect_doctor_status(base_url: str, timeout: float) -> dict[str, Any]:
    pith_home = _pith_home()
    server_path = _pith_server_path()
    venv_python = pith_home / "venv" / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python3")
    data_dir = _data_dir()
    db_path = data_dir / "pith.db"
    health = _fetch_json(base_url, "/health", timeout)
    readyz = _fetch_json(base_url, "/readyz", timeout)
    clients = collect_clients_status()
    checks = {
        "pith_home_exists": pith_home.is_dir(),
        "server_path_exists": server_path.is_dir(),
        "venv_python_exists": venv_python.is_file(),
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
        },
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
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
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
