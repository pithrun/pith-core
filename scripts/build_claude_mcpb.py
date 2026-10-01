#!/usr/bin/env python3
"""Build the deterministic Pith Claude Desktop MCPB artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import tempfile
import zipfile
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "pith_claude_mcpb_build.v1"
MANIFEST_TEMPLATE = "manifest.template.json"
LAUNCHER = "server/index.cjs"
WINDOWS_BOOTSTRAP = "server/windows_bootstrap.py"
EXPECTED_SOURCE_FILES = (MANIFEST_TEMPLATE, LAUNCHER, WINDOWS_BOOTSTRAP)
MAX_SOURCE_BYTES = 1024 * 1024
MAX_BUNDLE_BYTES = 4 * 1024 * 1024
SEMVER_RE = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-((?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*)(?:\.(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*))*))?"
    r"(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
)
SECRET_RE = re.compile(rb"PITH_API_KEY\s*[=:]", re.IGNORECASE)


class McpbBuildError(ValueError):
    """Raised when bundle source or output violates the Pith contract."""


def validate_pith_version(value: str) -> str:
    version = str(value or "")
    if len(version) > 128 or not SEMVER_RE.fullmatch(version):
        raise McpbBuildError("Pith version must be strict semver")
    return version


def render_manifest(template: dict[str, Any], version: str) -> dict[str, Any]:
    version = validate_pith_version(version)
    payload = json.loads(json.dumps(template))
    if payload.get("version") != "__PITH_VERSION__":
        raise McpbBuildError("manifest template version placeholder is missing")
    payload["version"] = version
    server = payload.get("server")
    if (
        not isinstance(server, dict)
        or set(server) != {"type", "entry_point", "mcp_config"}
        or server.get("type") != "node"
        or server.get("entry_point") != LAUNCHER
    ):
        raise McpbBuildError("manifest server contract is invalid")
    config = server.get("mcp_config")
    if not isinstance(config, dict) or set(config) != {"command", "args"} or config.get("command") != "node":
        raise McpbBuildError("manifest launch contract is invalid")
    args = config.get("args")
    if args != [
        "${__dirname}/server/index.cjs",
        "--pith-home",
        "${HOME}/.pith",
        "--surface-id",
        "claude_desktop_mcp",
    ]:
        raise McpbBuildError("manifest launcher arguments are invalid")
    return payload


def _read_source_file(source_dir: Path, relative_path: str) -> bytes:
    source_root = source_dir.resolve()
    candidate = source_dir / relative_path
    if candidate.is_symlink():
        raise McpbBuildError(f"symlink source rejected: {relative_path}")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(source_root)
        file_stat = resolved.stat()
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise McpbBuildError(f"invalid source file: {relative_path}") from exc
    if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size > MAX_SOURCE_BYTES:
        raise McpbBuildError(f"source file is not a bounded regular file: {relative_path}")
    content = resolved.read_bytes()
    if SECRET_RE.search(content):
        raise McpbBuildError(f"secret assignment marker rejected: {relative_path}")
    return content


def _zip_info(name: str, executable: bool = False) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    mode = 0o755 if executable else 0o644
    info.external_attr = (stat.S_IFREG | mode) << 16
    return info


def _write_checksum(output_path: Path, digest: str) -> Path:
    checksum_path = Path(f"{output_path}.sha256")
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="ascii",
            newline="\n",
            prefix=f".{checksum_path.name}.",
            suffix=".tmp",
            dir=checksum_path.parent,
            delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(f"{digest}  {output_path.name}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, checksum_path)
        temporary = None
    finally:
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    return checksum_path


def build_bundle(source_dir: Path, output_path: Path, version: str) -> dict[str, Any]:
    source_dir = Path(source_dir)
    output_path = Path(output_path)
    if source_dir.is_symlink() or not source_dir.is_dir():
        raise McpbBuildError("MCPB source must be a regular directory")
    discovered = {
        path.relative_to(source_dir).as_posix() for path in source_dir.rglob("*") if path.is_file() or path.is_symlink()
    }
    if discovered != set(EXPECTED_SOURCE_FILES):
        raise McpbBuildError("MCPB source file set is invalid")
    template_bytes = _read_source_file(source_dir, MANIFEST_TEMPLATE)
    launcher_bytes = _read_source_file(source_dir, LAUNCHER)
    windows_bootstrap_bytes = _read_source_file(source_dir, WINDOWS_BOOTSTRAP)
    try:
        template = json.loads(template_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise McpbBuildError("manifest template is not valid UTF-8 JSON") from exc
    if not isinstance(template, dict):
        raise McpbBuildError("manifest template root must be an object")
    manifest = render_manifest(template, version)
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent, delete=False
        ) as handle:
            temp_name = handle.name
        with zipfile.ZipFile(temp_name, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            archive.writestr(_zip_info("manifest.json"), manifest_bytes)
            archive.writestr(_zip_info(LAUNCHER, executable=True), launcher_bytes)
            archive.writestr(_zip_info(WINDOWS_BOOTSTRAP), windows_bootstrap_bytes)
        size_bytes = os.path.getsize(temp_name)
        if size_bytes > MAX_BUNDLE_BYTES:
            raise McpbBuildError("MCPB exceeds the maximum bundle size")
        os.replace(temp_name, output_path)
        temp_name = None
    finally:
        if temp_name:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass

    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    try:
        checksum_path = _write_checksum(output_path, digest)
    except OSError:
        output_path.unlink(missing_ok=True)
        raise
    return {
        "schema_version": SCHEMA_VERSION,
        "path": str(output_path.resolve()),
        "version": validate_pith_version(version),
        "sha256": digest,
        "checksum_path": str(checksum_path.resolve()),
        "size_bytes": output_path.stat().st_size,
        "members": ["manifest.json", LAUNCHER, WINDOWS_BOOTSTRAP],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    try:
        result = build_bundle(args.source, args.output, args.version)
    except (McpbBuildError, OSError) as exc:
        print(json.dumps({"schema_version": SCHEMA_VERSION, "status": "error", "error": str(exc)}))
        return 1
    print(json.dumps({"status": "built", **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
