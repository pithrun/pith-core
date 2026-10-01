"""Bounded, secret-safe diagnostics for Claude host configuration."""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import re
import stat as stat_module
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CLAUDE_HOST_DIAGNOSTICS_SCHEMA_VERSION = "pith_claude_host_diagnostics.v1"
CLAUDE_CONFIG_MAX_BYTES = 1024 * 1024
CLAUDE_EXTENSION_REGISTRY_MAX_BYTES = 2 * 1024 * 1024
CLAUDE_MCPB_MAX_BYTES = 4 * 1024 * 1024
CLAUDE_MCPB_MEMBERS = ["manifest.json", "server/index.cjs", "server/windows_bootstrap.py"]
CLAUDE_MCPB_EXECUTABLE_SHA256 = {
    "server/index.cjs": "7f1ce07d07659a3025b70a6dc4565d6d7552e6de0f766c315b8dc81b6afe3eb2",
    "server/windows_bootstrap.py": "2dad3259e292ab7273befd37e6b849fdca2aaff57743f9f2633de6463c66685d",
}
CLAUDE_PACKAGE_CHILD_LIMIT = 32
CLAUDE_LOG_FILE_LIMIT = 200
CLAUDE_LOCAL_EXTENSION_ID = "local.mcpb.pith.pith"
_SAFE_VERSION_RE = re.compile(r"^[0-9A-Za-z.+-]{1,128}$")

_SECRET_KEY_RE = re.compile(
    r"(?:api[_-]?key|token|secret|password|credential|private[_-]?key|access[_-]?key|auth)",
    re.IGNORECASE,
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _unique_extension_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_extension_json_key")
        result[key] = value
    return result


def validate_claude_mcpb_executable_members(archive: zipfile.ZipFile) -> None:
    """Require executable members to match the reviewed release sources exactly."""
    for member, expected_digest in CLAUDE_MCPB_EXECUTABLE_SHA256.items():
        raw = archive.read(member)
        if not raw or len(raw) > CLAUDE_CONFIG_MAX_BYTES or hashlib.sha256(raw).hexdigest() != expected_digest:
            raise ValueError(f"Claude MCPB executable member contract is invalid: {member}")


def _sanitize(value: Any, *, redact_all_values: bool = False) -> Any:
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if redact_all_values or _SECRET_KEY_RE.search(key_text):
                output[key_text] = "<redacted>"
            else:
                output[key_text] = _sanitize(item)
        return output
    if isinstance(value, list):
        return ["<redacted>" if redact_all_values else _sanitize(item) for item in value]
    return "<redacted>" if redact_all_values else value


def _public_config_payload(payload: dict[str, Any]) -> dict[str, Any]:
    sanitized = _sanitize(payload)
    roots = sanitized.get("mcpServers")
    if isinstance(roots, dict):
        for entry in roots.values():
            if isinstance(entry, dict) and "env" in entry:
                entry["env"] = _sanitize(entry["env"], redact_all_values=True)
    return sanitized


def _public_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(_public_config_payload(payload)).encode("utf-8")).hexdigest()


def _within_root(candidate: Path, root: Path) -> bool:
    try:
        candidate.absolute().relative_to(root.absolute())
    except ValueError:
        return False
    return True


def _windows_path_is_reparse_point(path: Path, *, os_name: str | None = None) -> bool:
    """Detect Windows reparse points on every supported Python version."""
    if (os_name or os.name) != "nt":
        return False
    try:
        attributes = path.lstat().st_file_attributes
    except FileNotFoundError:
        return False
    except (AttributeError, OSError):
        return True
    mask = getattr(stat_module, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400)
    return bool(attributes & mask)


def claude_path_has_link_component(path: Path, anchor: Path) -> bool:
    """Reject symlinks and Windows junctions anywhere in an existing path chain."""
    current = path.absolute()
    floor = anchor.absolute()
    if not _within_root(current, floor):
        return True
    while True:
        try:
            if current.is_symlink():
                return True
            is_junction = getattr(current, "is_junction", None)
            if is_junction is not None and is_junction():
                return True
            if _windows_path_is_reparse_point(current):
                return True
        except OSError:
            return True
        if current == floor:
            return False
        if current.parent == current:
            return True
        current = current.parent


def _read_bounded_extension_file(path: Path, anchor: Path, limit: int) -> bytes:
    """Read a regular, unlinked snapshot without trusting host manifest paths."""
    if claude_path_has_link_component(path, anchor):
        raise ValueError("link_component_rejected")
    before = path.lstat()
    if not stat_module.S_ISREG(before.st_mode) or before.st_size > limit:
        raise ValueError("not_regular_or_oversized")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if not stat_module.S_ISREG(opened.st_mode) or not os.path.samestat(before, opened):
            raise ValueError("file_changed")
        raw = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    current = path.lstat()
    if (
        len(raw) > limit
        or len(raw) != after.st_size
        or not os.path.samestat(after, current)
        or (opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)
        != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        or claude_path_has_link_component(path, anchor)
    ):
        raise ValueError("file_changed_or_oversized")
    return raw


def _extension_launch_digest(manifest: dict[str, Any]) -> str:
    # Include inherited launch fields, not just the win32 override.
    launch = {key: manifest.get(key) for key in ("manifest_version", "name", "version", "server")}
    return hashlib.sha256(_canonical_json(launch).encode("utf-8")).hexdigest()


def _inspect_config(candidate: Path, root: Path, anchor: Path | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(candidate),
        "root": str(root),
        "exists": False,
        "state": "missing",
        "pith_configured": False,
    }
    if not _within_root(candidate, root):
        result["state"] = "path_escape"
        return result
    if claude_path_has_link_component(candidate, anchor or root):
        result["state"] = "link_component_rejected"
        return result
    try:
        if candidate.is_symlink():
            result["state"] = "symlink_rejected"
            return result
        stat = candidate.stat()
    except FileNotFoundError:
        return result
    except OSError:
        result["state"] = "unreadable"
        return result
    result["exists"] = True
    result["size_bytes"] = stat.st_size
    if stat.st_size > CLAUDE_CONFIG_MAX_BYTES:
        result["state"] = "oversized"
        return result
    try:
        raw = candidate.read_bytes()
    except OSError:
        result["state"] = "unreadable"
        return result
    if len(raw) > CLAUDE_CONFIG_MAX_BYTES:
        result["state"] = "oversized"
        return result
    try:
        payload = json.loads(raw.decode("utf-8"), parse_constant=_reject_json_constant)
    except (UnicodeError, ValueError, json.JSONDecodeError):
        result["state"] = "malformed_json"
        return result
    if not isinstance(payload, dict):
        result["state"] = "schema_invalid"
        return result
    servers = payload.get("mcpServers")
    if not isinstance(servers, dict):
        result["state"] = "schema_invalid"
        result["public_digest"] = _public_digest(payload)
        return result
    entry = servers.get("pith")
    configured = isinstance(entry, dict)
    result.update(
        {
            "state": "configured" if configured else "not_configured",
            "pith_configured": configured,
            "public_digest": _public_digest(payload),
        }
    )
    return result


def collect_windows_claude_candidate_inventory(home: Path, environ: Mapping[str, str]) -> dict[str, Any]:
    appdata = Path(environ.get("APPDATA") or home / "AppData" / "Roaming")
    localappdata = Path(environ.get("LOCALAPPDATA") or home / "AppData" / "Local")
    output = [(appdata / "Claude" / "claude_desktop_config.json", appdata / "Claude", "classic")]
    packages_root = localappdata / "Packages"
    inventory_error_code = None
    try:
        package_roots = sorted(packages_root.glob("Claude_*"), key=lambda item: str(item).lower())
    except OSError:
        package_roots = []
        inventory_error_code = "packages_enumeration_failed"
    for package_root in package_roots[:CLAUDE_PACKAGE_CHILD_LIMIT]:
        root = package_root / "LocalCache" / "Roaming" / "Claude"
        output.append((root / "claude_desktop_config.json", root, "store"))
    return {
        "candidates": output,
        "appdata": appdata,
        "packages_root": packages_root,
        "package_child_count": len(package_roots),
        "overflow_count": max(0, len(package_roots) - CLAUDE_PACKAGE_CHILD_LIMIT),
        "inventory_error_code": inventory_error_code,
        "complete": inventory_error_code is None and len(package_roots) <= CLAUDE_PACKAGE_CHILD_LIMIT,
    }


def _windows_candidates(home: Path, environ: Mapping[str, str]) -> list[tuple[Path, Path, str]]:
    return collect_windows_claude_candidate_inventory(home, environ)["candidates"]


def _platform_candidates(home: Path, environ: Mapping[str, str], platform_name: str) -> list[tuple[Path, Path, str]]:
    if platform_name == "windows":
        return _windows_candidates(home, environ)
    if platform_name in {"darwin", "macos"}:
        root = home / "Library" / "Application Support" / "Claude"
        return [(root / "claude_desktop_config.json", root, "macos")]
    root = home / ".config" / "Claude"
    return [(root / "claude_desktop_config.json", root, "linux")]


def _extension_registry_candidates(
    home: Path,
    environ: Mapping[str, str],
    platform_name: str,
    platform_candidates: Sequence[tuple[Path, Path, str]] | None = None,
) -> list[tuple[Path, Path, str]]:
    output: list[tuple[Path, Path, str]] = []
    seen: set[str] = set()
    candidates = platform_candidates or _platform_candidates(home, environ, platform_name)
    for _config, root, kind in candidates:
        path = root / "extensions-installations.json"
        key = os.path.normcase(str(path))
        if key not in seen:
            output.append((path, root, kind))
            seen.add(key)
    return output


def _inspect_extension_registry(candidate: Path, root: Path, anchor: Path | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(candidate),
        "root": str(root),
        "exists": False,
        "state": "missing",
        "pith_extension_count": 0,
        "pith_versions": [],
        "signature_states": [],
        "local_pith_identity_registered": False,
    }
    if not _within_root(candidate, root):
        result["state"] = "path_escape"
        return result
    if claude_path_has_link_component(candidate, anchor or root):
        result["state"] = "link_component_rejected"
        return result
    try:
        if candidate.is_symlink():
            result["state"] = "symlink_rejected"
            return result
        file_stat = candidate.stat()
    except FileNotFoundError:
        return result
    except OSError:
        result["state"] = "unreadable"
        return result
    result["exists"] = True
    result["size_bytes"] = file_stat.st_size
    if file_stat.st_size > CLAUDE_EXTENSION_REGISTRY_MAX_BYTES:
        result["state"] = "oversized"
        return result
    try:
        raw = _read_bounded_extension_file(candidate, anchor or root, CLAUDE_EXTENSION_REGISTRY_MAX_BYTES)
        payload = json.loads(
            raw.decode("utf-8-sig"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_extension_json_object,
        )
    except (OSError, UnicodeError, ValueError, RecursionError):
        result["state"] = "malformed_json"
        return result
    extensions = payload.get("extensions") if isinstance(payload, dict) else None
    if not isinstance(extensions, dict):
        result["state"] = "schema_invalid"
        return result
    versions: set[str] = set()
    signatures: set[str] = set()
    count = 0
    for extension_id, record in extensions.items():
        manifest = record.get("manifest") if isinstance(record, dict) else None
        if not isinstance(manifest, dict) or manifest.get("name") != "pith":
            continue
        count += 1
        if extension_id == CLAUDE_LOCAL_EXTENSION_ID and record.get("id", extension_id) == extension_id:
            result["local_pith_identity_registered"] = True
        version = manifest.get("version")
        if isinstance(version, str) and _SAFE_VERSION_RE.fullmatch(version):
            versions.add(version)
        signatures.add("present" if record.get("signatureInfo") else "absent")
    result.update(
        {
            "state": "pith_installed" if count else "pith_not_installed",
            "pith_extension_count": count,
            "pith_versions": sorted(versions),
            "signature_states": sorted(signatures),
        }
    )
    return result


def _inspect_prepared_extensions(home: Path) -> dict[str, Any]:
    root = home / ".pith" / "integrations"
    result: dict[str, Any] = {
        "path": str(root),
        "state": "missing",
        "package_count": 0,
        "package_paths": [],
        "versions": [],
        "digests": [],
        "content_contracts": [],
    }
    if not root.is_dir() or claude_path_has_link_component(root, home):
        return result
    try:
        candidates = sorted(root.glob("pith-claude-*.mcpb"), key=lambda item: item.name)
    except OSError:
        result["state"] = "unreadable"
        return result
    versions: list[str] = []
    package_paths: list[str] = []
    digests: list[str] = []
    content_contracts: list[dict[str, Any]] = []
    overflow_count = max(0, len(candidates) - CLAUDE_PACKAGE_CHILD_LIMIT)
    invalid_count = overflow_count
    for path in candidates[:CLAUDE_PACKAGE_CHILD_LIMIT]:
        try:
            if path.is_symlink() or not path.is_file() or path.stat().st_size > CLAUDE_MCPB_MAX_BYTES:
                invalid_count += 1
                continue
            version = path.name[len("pith-claude-") : -len(".mcpb")]
            if not _SAFE_VERSION_RE.fullmatch(version):
                invalid_count += 1
                continue
            package_raw = _read_bounded_extension_file(path, home, CLAUDE_MCPB_MAX_BYTES)
            with zipfile.ZipFile(io.BytesIO(package_raw)) as archive:
                if archive.namelist() != CLAUDE_MCPB_MEMBERS or any(
                    info.file_size > CLAUDE_CONFIG_MAX_BYTES or stat_module.S_ISLNK(info.external_attr >> 16)
                    for info in archive.infolist()
                ):
                    invalid_count += 1
                    continue
                validate_claude_mcpb_executable_members(archive)
                manifest = json.loads(
                    archive.read("manifest.json").decode("utf-8"),
                    parse_constant=_reject_json_constant,
                    object_pairs_hook=_unique_extension_json_object,
                )
            server = manifest.get("server") if isinstance(manifest, dict) else None
            config = server.get("mcp_config") if isinstance(server, dict) else None
            if not (
                isinstance(manifest, dict)
                and manifest.get("manifest_version") == "0.3"
                and manifest.get("name") == "pith"
                and manifest.get("version") == version
                and isinstance(server, dict)
                and set(server) == {"type", "entry_point", "mcp_config"}
                and server.get("type") == "node"
                and server.get("entry_point") == "server/index.cjs"
                and isinstance(config, dict)
                and set(config) == {"command", "args"}
                and config.get("command") == "node"
                and "env" not in config
                and config.get("args")
                == [
                    "${__dirname}/server/index.cjs",
                    "--pith-home",
                    "${HOME}/.pith",
                    "--surface-id",
                    "claude_desktop_mcp",
                ]
            ):
                invalid_count += 1
                continue
            versions.append(version)
            package_paths.append(str(path))
            digest = hashlib.sha256(package_raw).hexdigest()
            digests.append(digest)
            content_contracts.append(
                {
                    "package_path": str(path),
                    "package_sha256": digest,
                    "manifest_launch_sha256": _extension_launch_digest(manifest),
                    "executable_sha256": dict(CLAUDE_MCPB_EXECUTABLE_SHA256),
                }
            )
        except (OSError, UnicodeError, ValueError, KeyError, RuntimeError, zipfile.BadZipFile, RecursionError):
            invalid_count += 1
    state = "invalid" if invalid_count else ("prepared" if versions else "empty")
    result.update(
        {
            "state": state,
            "package_count": len(versions),
            "package_paths": [] if invalid_count else sorted(package_paths),
            "invalid_count": invalid_count,
            "overflow_count": overflow_count,
            "versions": sorted(versions),
            "digests": sorted(digests),
            "content_contracts": [] if invalid_count else content_contracts,
        }
    )
    return result


def _inspect_windows_extension_content(
    registries: list[dict[str, Any]], prepared: dict[str, Any], inventory: dict[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "state": "unknown",
        "reason": "reference_unavailable",
        "installed_path": None,
        "reference_package_path": None,
        "reference_package_sha256": None,
        "manifest_launch_matches": None,
        "executable_members_match": None,
    }
    references = prepared.get("content_contracts") or []
    reference = references[0] if prepared["state"] == "prepared" and len(references) == 1 else None
    if reference:
        result["reference_package_path"] = reference["package_path"]
        result["reference_package_sha256"] = reference["package_sha256"]
    if not inventory["complete"] or any(
        item["state"] not in {"missing", "pith_installed", "pith_not_installed"} for item in registries
    ):
        result["reason"] = "inventory_unverified"
        return result
    registered = [item for item in registries if item["pith_extension_count"]]
    if not registered:
        result.update(state="not_installed", reason="not_registered")
        return result
    if sum(item["pith_extension_count"] for item in registered) != 1:
        result["reason"] = "multiple_registrations"
        return result
    registry = registered[0]
    if not registry["local_pith_identity_registered"]:
        result["reason"] = "registration_identity_unverified"
        return result
    root = Path(registry["root"])
    installed = root / "Claude Extensions" / CLAUDE_LOCAL_EXTENSION_ID
    result["installed_path"] = str(installed)
    if reference is None:
        return result
    anchor = inventory["appdata"] if registry["kind"] == "classic" else inventory["packages_root"]
    try:
        raw = _read_bounded_extension_file(installed / "manifest.json", anchor, CLAUDE_CONFIG_MAX_BYTES)
        manifest = json.loads(
            raw.decode("utf-8-sig"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_extension_json_object,
        )
        if not isinstance(manifest, dict) or not isinstance(manifest.get("server"), dict):
            raise ValueError("invalid_manifest")
        result["manifest_launch_matches"] = _extension_launch_digest(manifest) == reference["manifest_launch_sha256"]
        if not result["manifest_launch_matches"]:
            result.update(state="stale", reason="manifest_launch_mismatch")
            return result
        matches = {
            member: hashlib.sha256(
                _read_bounded_extension_file(installed / member, anchor, CLAUDE_CONFIG_MAX_BYTES)
            ).hexdigest()
            == digest
            for member, digest in reference["executable_sha256"].items()
        }
        result["executable_members_match"] = matches
        if not all(matches.values()):
            result.update(state="stale", reason="executable_mismatch")
        else:
            result.update(state="current", reason="matches_staged_package")
    except (OSError, UnicodeError, ValueError, RecursionError):
        result["reason"] = "installed_content_unverified"
    return result


def _log_locations(home: Path, environ: Mapping[str, str], platform_name: str) -> list[dict[str, Any]]:
    roots: list[tuple[Path, str]] = []
    if platform_name == "windows":
        appdata = Path(environ.get("APPDATA") or home / "AppData" / "Roaming")
        localappdata = Path(environ.get("LOCALAPPDATA") or home / "AppData" / "Local")
        roots.append((appdata / "Claude" / "logs", "classic"))
        packages_root = localappdata / "Packages"
        try:
            package_roots = sorted(packages_root.glob("Claude_*"), key=lambda item: str(item).lower())
        except OSError:
            package_roots = []
        roots.extend(
            (root / "LocalCache" / "Roaming" / "Claude" / "logs", "store")
            for root in package_roots[:CLAUDE_PACKAGE_CHILD_LIMIT]
        )
    elif platform_name in {"darwin", "macos"}:
        roots.append((home / "Library" / "Logs" / "Claude", "macos"))
    else:
        roots.append((home / ".config" / "Claude" / "logs", "linux"))
    output: list[dict[str, Any]] = []
    for root, kind in roots:
        item: dict[str, Any] = {
            "path": str(root),
            "kind": kind,
            "exists": root.is_dir(),
            "file_count": 0,
            "newest_modified_at": None,
            "content_inspected": False,
        }
        if root.is_dir() and not root.is_symlink():
            try:
                files = [path for path in list(root.iterdir())[:CLAUDE_LOG_FILE_LIMIT] if path.is_file()]
                item["file_count"] = len(files)
                modified = [path.stat().st_mtime for path in files]
                item["newest_modified_at"] = max(modified) if modified else None
            except OSError:
                item["state"] = "unreadable"
        output.append(item)
    return output


def collect_claude_host_diagnostics(
    *,
    home: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
    platform_name: str | None = None,
    selected_config_path: str | Path | None = None,
    process_paths: Sequence[str] | None = None,
    process_probe_error: str | None = None,
) -> dict[str, Any]:
    """Return bounded diagnostics without claiming that a Claude host loaded Pith."""

    home_path = Path(home or Path.home()).expanduser()
    env = dict(os.environ if environ is None else environ)
    plat = (platform_name or platform.system()).lower()
    inventory = collect_windows_claude_candidate_inventory(home_path, env) if plat == "windows" else None
    platform_candidates = inventory["candidates"] if inventory else _platform_candidates(home_path, env, plat)
    candidates: list[dict[str, Any]] = []
    for candidate, root, kind in platform_candidates:
        anchor = (
            inventory["appdata"]
            if inventory and kind == "classic"
            else inventory["packages_root"]
            if inventory
            else root
        )
        inspected = _inspect_config(candidate, root, anchor)
        inspected["kind"] = kind
        candidates.append(inspected)
    existing = [item for item in candidates if item["exists"]]
    configured = [item for item in candidates if item["pith_configured"]]
    selected = str(Path(selected_config_path).expanduser()) if selected_config_path else None
    if len(configured) > 1:
        conflict = "multiple_configured"
    elif selected and configured and selected != configured[0]["path"]:
        conflict = "selected_path_conflict"
    elif len(existing) > 1 and not configured:
        conflict = "multiple_existing_unconfigured"
    else:
        conflict = "none"
    safe_process_paths = sorted(
        {
            str(path)
            for path in (process_paths or [])
            if isinstance(path, str) and path and "claude" in Path(path).name.lower()
        }
    )
    extension_registries: list[dict[str, Any]] = []
    for candidate, root, kind in _extension_registry_candidates(home_path, env, plat, platform_candidates):
        anchor = (
            inventory["appdata"]
            if inventory and kind == "classic"
            else inventory["packages_root"]
            if inventory
            else root
        )
        inspected = _inspect_extension_registry(candidate, root, anchor)
        inspected["kind"] = kind
        extension_registries.append(inspected)
    installed_extensions = [item for item in extension_registries if item["pith_extension_count"] > 0]
    prepared_extension = _inspect_prepared_extensions(home_path)
    extension_content = (
        _inspect_windows_extension_content(extension_registries, prepared_extension, inventory)
        if inventory is not None
        else {"state": "not_checked", "reason": "windows_only"}
    )
    return {
        "schema_version": CLAUDE_HOST_DIAGNOSTICS_SCHEMA_VERSION,
        "diagnostic_status": "degraded" if inventory and not inventory["complete"] else "ok",
        "platform": plat,
        "selected_config_path": selected,
        "candidate_count": len(candidates),
        "candidate_inventory_complete": inventory["complete"] if inventory else True,
        "candidate_overflow_count": inventory["overflow_count"] if inventory else 0,
        "package_child_count": inventory["package_child_count"] if inventory else 0,
        "candidate_inventory_error": inventory["inventory_error_code"] if inventory else None,
        "existing_count": len(existing),
        "configured_count": len(configured),
        "conflict_state": conflict,
        "candidates": candidates,
        "prepared_extension": prepared_extension,
        "extension_registry_count": len(extension_registries),
        "extension_registry_existing_count": sum(1 for item in extension_registries if item["exists"]),
        "pith_extension_installation_count": sum(item["pith_extension_count"] for item in extension_registries),
        "extension_installation_state": "installed" if installed_extensions else "not_installed",
        "extension_content_state": extension_content["state"],
        "extension_content": extension_content,
        "extension_registries": extension_registries,
        "log_locations": _log_locations(home_path, env, plat),
        "policy_channels": [
            {"name": "Microsoft-Windows-AppModel-Runtime/Admin", "query_status": "not_queried"},
            {"name": "Microsoft-Windows-AppLocker/EXE and DLL", "query_status": "not_queried"},
        ]
        if plat == "windows"
        else [],
        "process_observation": "observed" if process_paths is not None else "unavailable",
        "process_probe_error": process_probe_error,
        "process_count": len(safe_process_paths),
        "process_paths": safe_process_paths,
        "host_loaded_status": "not_observed",
        "connected_status": "not_proven",
    }
