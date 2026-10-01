"""Shared contract helpers for the local Codex/ChatGPT plugin package."""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import json
import os
import re
import secrets
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CODEX_PLUGIN_MCP_ROOT = "mcpServers"
CODEX_PLUGIN_LIST_TIMEOUT_SECONDS = 45
CODEX_PLUGIN_SUPPORT_READBACK_TIMEOUT_SECONDS = 5
CODEX_PLUGIN_MAX_PACKAGE_BYTES = 10 * 1024 * 1024
CODEX_PLUGIN_LOCK_STALE_SECONDS = 10 * 60
CODEX_PLUGIN_LOCK_SCHEMA_VERSION = "pith_codex_plugin_lock.v1"
CODEX_PLUGIN_LOCK_NAME = ".pith-configure.lock"
CODEX_PLUGIN_ERROR_SNIPPET_CHARS = 500
CODEX_PLUGIN_MARKETPLACE_NAME = "personal"
CODEX_PLUGIN_MARKETPLACE_DISPLAY_NAME = "Personal"
CODEX_PLUGIN_DEFAULT_TOOLS_APPROVAL_MODE = "approve"

SEMVER_RE = re.compile(
    r"^(0|[1-9]\d*)\."
    r"(0|[1-9]\d*)\."
    r"(0|[1-9]\d*)"
    r"(?:-(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*)(?:\."
    r"(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*))*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)
SECRET_KEY_RE = re.compile(
    r"(?:api[_-]?key|token|secret|password|credential|private[_-]?key|access[_-]?key|auth)",
    re.IGNORECASE,
)
ENV_ASSIGNMENT_RE = re.compile(
    r"\b([A-Z0-9_]*(?:API_KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|PRIVATE_KEY|ACCESS_KEY|AUTH)"
    r"[A-Z0-9_]*)=(\"[^\"]*\"|'[^']*'|\S+)",
    re.IGNORECASE,
)
JSON_SECRET_RE = re.compile(
    r'("?[A-Za-z0-9_.-]*(?:api[_-]?key|token|secret|password|credential|private[_-]?key|access[_-]?key|auth)'
    r'[A-Za-z0-9_.-]*"?\s*[:=]\s*)("[^"]*"|\'[^\']*\'|[^\s,}]+)',
    re.IGNORECASE,
)
HEADER_SECRET_RE = re.compile(r"((?:X-API-Key|Authorization):\s*)(Bearer\s+)?\S+", re.IGNORECASE)
LONG_TOKEN_RE = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{12,}|[A-Za-z0-9_-]{40,})\b")


class CodexPluginContractError(ValueError):
    """A classified, safe-to-report plugin contract failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def normalize_codex_plugin_marketplace_display_name(value: object) -> str:
    if not isinstance(value, str):
        raise CodexPluginContractError(
            "invalid_marketplace_display_name",
            "Plugin marketplace interface.displayName must be a string.",
        )
    display_name = value.strip()
    if not display_name or len(display_name) > 128 or any(ord(char) < 32 for char in display_name):
        raise CodexPluginContractError(
            "invalid_marketplace_display_name",
            "Plugin marketplace interface.displayName must be a single-line value of 1-128 characters.",
        )
    return display_name


@dataclass(frozen=True)
class LoadedCodexPluginPackage:
    root: Path
    manifest: dict[str, Any]
    mcp_payload: dict[str, Any]
    public_digest: str


def with_codex_cachebuster(base_version: str, now: dt.datetime | None = None) -> str:
    if not isinstance(base_version, str):
        raise CodexPluginContractError("invalid_version", "Plugin base version must be a string.")
    value = base_version.strip()
    if not value or len(value) > 128:
        raise CodexPluginContractError("invalid_version", "Plugin base version must contain 1-128 characters.")
    version_prefix = value.split("+", 1)[0]
    if SEMVER_RE.fullmatch(version_prefix) is None:
        raise CodexPluginContractError("invalid_version", "Plugin base version must be strict semver.")
    instant = now or dt.datetime.now(dt.UTC)
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise CodexPluginContractError("invalid_timestamp", "Plugin version timestamp must be timezone-aware.")
    timestamp = instant.astimezone(dt.UTC).strftime("%Y%m%d%H%M%S")
    return f"{version_prefix}+codex.local-{timestamp}"


def validate_codex_plugin_payloads(manifest: object, mcp_payload: object) -> dict[str, Any]:
    state: dict[str, Any] = {
        "plugin_mcp_schema_status": "invalid",
        "plugin_mcp_schema_reason": "Plugin package is invalid.",
        "plugin_mcp_server_names": [],
    }
    if not isinstance(manifest, dict) or not isinstance(mcp_payload, dict):
        state["plugin_mcp_schema_reason"] = "Plugin manifest and MCP payload must be JSON objects."
        return state
    if manifest.get("name") != "pith":
        state["plugin_mcp_schema_reason"] = "Plugin manifest name must be pith."
        return state
    version = manifest.get("version")
    if not isinstance(version, str) or SEMVER_RE.fullmatch(version) is None:
        state["plugin_mcp_schema_reason"] = "Plugin manifest version must be strict semver."
        return state
    if manifest.get("mcpServers") != "./.mcp.json":
        state["plugin_mcp_schema_reason"] = "Plugin manifest mcpServers must point to ./.mcp.json."
        return state
    if set(mcp_payload) != {CODEX_PLUGIN_MCP_ROOT}:
        state["plugin_mcp_schema_reason"] = "Plugin MCP file must contain only top-level mcpServers."
        return state
    root = mcp_payload.get(CODEX_PLUGIN_MCP_ROOT)
    if not isinstance(root, dict):
        state["plugin_mcp_schema_reason"] = "Plugin MCP field mcpServers must be an object."
        return state
    state["plugin_mcp_server_names"] = sorted(str(name) for name in root)
    if set(root) != {"pith"} or not isinstance(root.get("pith"), dict):
        state["plugin_mcp_schema_reason"] = "Plugin MCP field mcpServers must contain exactly one pith object."
        return state
    entry = root["pith"]
    command = entry.get("command")
    args = entry.get("args")
    env = entry.get("env")
    if not isinstance(command, str) or not command.strip():
        state["plugin_mcp_schema_reason"] = "Plugin MCP pith command must be a non-empty string."
        return state
    if not isinstance(args, list) or not args or not all(isinstance(arg, str) and arg for arg in args):
        state["plugin_mcp_schema_reason"] = "Plugin MCP pith args must be a non-empty string array."
        return state
    if not isinstance(env, dict):
        state["plugin_mcp_schema_reason"] = "Plugin MCP pith env must be an object."
        return state
    approval_mode = entry.get("default_tools_approval_mode")
    state["plugin_default_tools_approval_mode"] = approval_mode
    if approval_mode != CODEX_PLUGIN_DEFAULT_TOOLS_APPROVAL_MODE:
        state["plugin_mcp_schema_reason"] = (
            "Plugin MCP pith default_tools_approval_mode must be approve."
        )
        return state
    state["plugin_mcp_schema_status"] = "valid"
    state["plugin_mcp_schema_reason"] = "Plugin package uses documented mcpServers.pith."
    return state


def _redact_secret_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): "<redacted>" if SECRET_KEY_RE.search(str(key)) else _redact_secret_values(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_secret_values(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def public_codex_plugin_digest(manifest: dict[str, Any], mcp_payload: dict[str, Any]) -> str:
    payload = {"manifest": manifest, "mcp": _redact_secret_values(mcp_payload)}
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _secret_pairs(value: Any, path: tuple[str, ...] = ()) -> list[tuple[str, str]]:
    output: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key in sorted(value, key=lambda item: str(item)):
            key_text = str(key)
            item = value[key]
            current_path = (*path, key_text)
            if SECRET_KEY_RE.search(key_text):
                output.append((".".join(current_path), _canonical_json(item)))
            else:
                output.extend(_secret_pairs(item, current_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            output.extend(_secret_pairs(item, (*path, str(index))))
    return output


def codex_plugin_secret_values_match(left_mcp: dict[str, Any], right_mcp: dict[str, Any]) -> bool:
    left = _canonical_json(_secret_pairs(left_mcp)).encode("utf-8")
    right = _canonical_json(_secret_pairs(right_mcp)).encode("utf-8")
    return hmac.compare_digest(left, right)


def parse_codex_plugin_list(payload: object, selector: str, expected_version: str) -> dict[str, Any]:
    if isinstance(payload, dict):
        installed = payload.get("installed")
        shape = "object_installed"
    elif isinstance(payload, list):
        installed = payload
        shape = "legacy_bare_array"
    else:
        raise CodexPluginContractError("invalid_list_root", "Codex plugin list must be an object or array.")
    if not isinstance(installed, list):
        raise CodexPluginContractError("invalid_installed_list", "Codex plugin list installed field must be an array.")
    matches = [item for item in installed if isinstance(item, dict) and item.get("pluginId") == selector]
    if not matches:
        raise CodexPluginContractError("not_installed", f"Codex plugin {selector} is not installed.")
    if len(matches) != 1:
        raise CodexPluginContractError("duplicate_selector", f"Codex plugin {selector} appears more than once.")
    entry = matches[0]
    if entry.get("installed") is not True:
        raise CodexPluginContractError("not_installed", f"Codex plugin {selector} is not marked installed.")
    if entry.get("enabled") is not True:
        raise CodexPluginContractError("installed_disabled", f"Codex plugin {selector} is disabled.")
    if entry.get("version") != expected_version:
        raise CodexPluginContractError(
            "installed_version_mismatch",
            f"Codex plugin {selector} version does not match the generated package.",
        )
    source = entry.get("source")
    if not isinstance(source, dict):
        raise CodexPluginContractError("installed_source_unresolvable", "Installed plugin source must be an object.")
    source_kind = source.get("source")
    source_path = source.get("path")
    if source_kind != "local":
        raise CodexPluginContractError("installed_source_unresolvable", "Installed plugin source must be local.")
    if not isinstance(source_path, str) or not source_path.strip() or len(source_path) > 4096:
        raise CodexPluginContractError("installed_source_unresolvable", "Installed plugin source path is missing.")
    if "\x00" in source_path or source_path.startswith(("\\\\", "//")) or not Path(source_path).is_absolute():
        raise CodexPluginContractError(
            "installed_source_unresolvable",
            "Installed plugin source path must be an absolute local filesystem path.",
        )
    return {
        "shape": shape,
        "plugin_id": selector,
        "installed": True,
        "enabled": True,
        "installed_version": entry["version"],
        "installed_source_kind": source_kind,
        "installed_source_path": source_path,
        "entry": entry,
    }


def _read_json_child(root: Path, relative_path: Path) -> dict[str, Any]:
    resolved_root = root.expanduser().resolve()
    child = (resolved_root / relative_path).resolve()
    if not child.is_relative_to(resolved_root):
        raise CodexPluginContractError("source_path_escape", f"Plugin file {relative_path} escapes its source root.")
    try:
        size = child.stat().st_size
    except OSError as exc:
        raise CodexPluginContractError(
            "source_unreadable", f"Plugin file {relative_path} is unreadable: {exc}"
        ) from exc
    if size > CODEX_PLUGIN_MAX_PACKAGE_BYTES:
        raise CodexPluginContractError("source_oversized", f"Plugin file {relative_path} exceeds the size limit.")
    try:
        payload = json.loads(child.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CodexPluginContractError("source_invalid_json", f"Plugin file {relative_path} is invalid: {exc}") from exc
    if not isinstance(payload, dict):
        raise CodexPluginContractError("source_invalid_json", f"Plugin file {relative_path} must contain an object.")
    return payload


def load_codex_plugin_package(root: str | Path) -> LoadedCodexPluginPackage:
    resolved_root = Path(root).expanduser().resolve()
    manifest_path = Path(".codex-plugin") / "plugin.json"
    mcp_path = Path(".mcp.json")
    try:
        total_size = (resolved_root / manifest_path).resolve().stat().st_size + (
            resolved_root / mcp_path
        ).resolve().stat().st_size
    except OSError as exc:
        raise CodexPluginContractError("source_unreadable", f"Plugin package is unreadable: {exc}") from exc
    if total_size > CODEX_PLUGIN_MAX_PACKAGE_BYTES:
        raise CodexPluginContractError("source_oversized", "Plugin package exceeds the size limit.")
    manifest = _read_json_child(resolved_root, manifest_path)
    mcp_payload = _read_json_child(resolved_root, mcp_path)
    validation = validate_codex_plugin_payloads(manifest, mcp_payload)
    if validation["plugin_mcp_schema_status"] != "valid":
        raise CodexPluginContractError("source_schema_invalid", str(validation["plugin_mcp_schema_reason"]))
    return LoadedCodexPluginPackage(
        root=resolved_root,
        manifest=manifest,
        mcp_payload=mcp_payload,
        public_digest=public_codex_plugin_digest(manifest, mcp_payload),
    )


def load_installed_source_state(entry: dict[str, Any]) -> LoadedCodexPluginPackage:
    source = entry.get("source")
    if not isinstance(source, dict) or not isinstance(source.get("path"), str):
        raise CodexPluginContractError("installed_source_unresolvable", "Installed plugin source path is missing.")
    return load_codex_plugin_package(source["path"])


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass


def _default_pid_is_running(pid: int) -> bool | None:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None
    return True


def _parse_lock(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CodexPluginContractError("operation_lock_invalid", f"Plugin operation lock is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise CodexPluginContractError("operation_lock_invalid", "Plugin operation lock must contain an object.")
    return payload


class CodexPluginOperationLock:
    def __init__(
        self,
        plugin_root: Path,
        *,
        now: dt.datetime | None = None,
        pid: int | None = None,
        pid_is_running: Callable[[int], bool | None] | None = None,
    ):
        self.plugin_root = plugin_root
        self.path = plugin_root / CODEX_PLUGIN_LOCK_NAME
        self.now = now or dt.datetime.now(dt.UTC)
        self.pid = pid if pid is not None else os.getpid()
        self.pid_is_running = pid_is_running or _default_pid_is_running
        self.owner_token = secrets.token_hex(16)
        self.acquired = False

    def __enter__(self) -> CodexPluginOperationLock:
        if self.now.tzinfo is None or self.now.utcoffset() is None:
            raise CodexPluginContractError("operation_lock_invalid_time", "Plugin operation lock time must be aware.")
        self.plugin_root.mkdir(parents=True, exist_ok=True)
        self._acquire_or_reclaim_once()
        return self

    def _acquire_or_reclaim_once(self) -> None:
        payload = {
            "schema_version": CODEX_PLUGIN_LOCK_SCHEMA_VERSION,
            "pid": self.pid,
            "created_at": self.now.astimezone(dt.UTC).isoformat(),
            "owner_token": self.owner_token,
        }
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            existing = _parse_lock(self.path)
            if existing.get("schema_version") != CODEX_PLUGIN_LOCK_SCHEMA_VERSION:
                raise CodexPluginContractError("operation_lock_invalid", "Plugin operation lock schema is invalid.")
            try:
                created_at = dt.datetime.fromisoformat(str(existing["created_at"]))
                existing_pid = int(existing["pid"])
                existing_token = str(existing["owner_token"])
            except (KeyError, TypeError, ValueError) as exc:
                raise CodexPluginContractError(
                    "operation_lock_invalid", "Plugin operation lock fields are invalid."
                ) from exc
            if created_at.tzinfo is None or created_at.utcoffset() is None:
                raise CodexPluginContractError(
                    "operation_lock_invalid", "Plugin operation lock timestamp must be aware."
                )
            age = (self.now.astimezone(dt.UTC) - created_at.astimezone(dt.UTC)).total_seconds()
            running = self.pid_is_running(existing_pid)
            if age <= CODEX_PLUGIN_LOCK_STALE_SECONDS or running is not False:
                raise CodexPluginContractError(
                    "operation_in_progress", "Another Pith plugin configuration is active or unresolved."
                )
            current = _parse_lock(self.path)
            if current.get("owner_token") != existing_token:
                raise CodexPluginContractError("operation_in_progress", "Plugin operation lock ownership changed.")
            try:
                self.path.unlink()
            except OSError as exc:
                raise CodexPluginContractError(
                    "operation_lock_reclaim_failed", f"Stale plugin lock could not be reclaimed: {exc}"
                ) from exc
            return self._acquire_or_reclaim_once()
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.acquired = True

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if not self.acquired:
            return
        try:
            current = _parse_lock(self.path)
            if current.get("owner_token") == self.owner_token:
                self.path.unlink(missing_ok=True)
        except (CodexPluginContractError, OSError):
            pass
        finally:
            self.acquired = False


def inspect_codex_plugin_operation_lock(plugin_root: Path) -> dict[str, Any]:
    path = plugin_root / CODEX_PLUGIN_LOCK_NAME
    if not path.exists():
        return {"state": "absent", "path": str(path)}
    try:
        payload = _parse_lock(path)
    except CodexPluginContractError as exc:
        return {"state": "invalid", "path": str(path), "error": str(exc)}
    return {
        "state": "present",
        "path": str(path),
        "schema_version": payload.get("schema_version"),
        "pid": payload.get("pid"),
        "created_at": payload.get("created_at"),
    }


def redact_diagnostic_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = ENV_ASSIGNMENT_RE.sub(r"\1=<redacted>", text)
    text = HEADER_SECRET_RE.sub(r"\1<redacted>", text)
    text = JSON_SECRET_RE.sub(r"\1<redacted>", text)
    text = LONG_TOKEN_RE.sub("<redacted>", text)
    return text[:CODEX_PLUGIN_ERROR_SNIPPET_CHARS]
