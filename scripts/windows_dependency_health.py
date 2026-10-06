#!/usr/bin/env python3
"""Read-only installer acceptance for recorded core runtime files and imports.

Not a package-content integrity check, hostile-code sandbox, or repair command.
Run using the selected interpreter with -I -B; never import the Pith application.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import csv
import importlib
import io
import json
import os
import re
import subprocess
import sys
import unicodedata
from importlib import metadata
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from packaging.requirements import Requirement

MAX_REQUIREMENTS_BYTES = 1024 * 1024
MAX_ROOTS = 256
MAX_DISTRIBUTIONS = 400
MAX_EDGES = 10_000
MAX_RECORD_BYTES = 8 * 1024 * 1024
MAX_RECORD_ROWS = 100_000
MAX_TOTAL_ROWS = 250_000
MAX_FIELD_CHARS = 4096
MAX_DIAGNOSTIC_PATH = 512
MAX_EXAMPLES = 20
MAX_OUTPUT_BYTES = 1024 * 1024
IMPORT_TIMEOUT_SECONDS = 60
CRITICAL_IMPORTS = (
    "pydantic",
    "pydantic_core.core_schema",
    "fastapi",
    "uvicorn",
    "dotenv",
    "multipart",
    "numpy",
    "sklearn",
    "httpx",
    "requests",
    "anthropic",
    "openai",
    "mcp",
)
_NAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?\Z")
_NATIVE = re.compile(r"(?:\.py|\.pyd|\.dll|\.dylib|\.so(?:\.\d+)*)\Z", re.I)
_HASH = re.compile(r"[A-Za-z0-9_+-]+=([A-Za-z0-9_-]+={0,2})\Z")


class HealthError(ValueError):
    """Bounded stable diagnostics, never arbitrary exception text."""

    def __init__(self, code: str, distribution: str = "", path: str = ""):
        super().__init__(code)
        self.code = code
        self.distribution = distribution if _NAME.fullmatch(distribution) else ""
        # Unsafe input must not expose absolute paths or control characters.
        self.path = path[:MAX_DIAGNOSTIC_PATH] if _safe_path(path) else ""

    def diagnostic(self) -> dict:
        result = {"code": self.code}
        if self.distribution:
            result["distribution"] = self.distribution
        if self.path:
            result["path"] = self.path
        return result


def _packaging():
    try:
        from packaging.markers import default_environment
        from packaging.requirements import Requirement
        from packaging.utils import canonicalize_name
        from packaging.version import Version
    except ImportError:
        raise HealthError("packaging_unavailable") from None
    return Requirement, canonicalize_name, Version, default_environment


def _requirement(text: str, code: str):
    Requirement, _, _, _ = _packaging()
    if not isinstance(text, str) or not text or len(text) > MAX_FIELD_CHARS:
        raise HealthError(code)
    try:
        requirement = Requirement(text)
        if requirement.url:
            raise ValueError
        return requirement
    except (ValueError, TypeError):
        raise HealthError(code) from None


def _applicable(requirement, environment, extras=("",)):
    return requirement.marker is None or any(
        requirement.marker.evaluate({**environment, "extra": extra}) for extra in extras
    )


def _strip_comment(line: str) -> str:
    quote = None
    for index, character in enumerate(line):
        if character in "\"'":
            if quote == character:
                quote = None
            elif quote is None:
                quote = character
        if character == "#" and quote is None and (index == 0 or line[index - 1].isspace()):
            return line[:index].strip()
    return line.strip()


def parse_requirements(path: Path) -> list[Requirement]:
    """Parse deterministic pip root input without options, includes or URLs."""
    _, _, _, default_environment = _packaging()
    try:
        with Path(path).open("rb") as stream:
            raw = stream.read(MAX_REQUIREMENTS_BYTES + 1)
    except (OSError, TypeError, ValueError):
        raise HealthError("requirements_unreadable") from None
    if len(raw) > MAX_REQUIREMENTS_BYTES:
        raise HealthError("requirements_invalid")
    try:
        text = raw.decode("utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig")
    except UnicodeError:
        raise HealthError("requirements_invalid") from None
    roots = []
    logical_lines = 0
    for line in text.splitlines():
        line = _strip_comment(line)
        if not line:
            continue
        logical_lines += 1
        if logical_lines > MAX_ROOTS or line.startswith("-") or line.endswith("\\") or "\x00" in line:
            raise HealthError("requirements_invalid")
        requirement = _requirement(line, "requirements_invalid")
        if _applicable(requirement, default_environment()):
            roots.append(requirement)
    if not roots:
        raise HealthError("requirements_empty")
    return roots


def resolve_closure(roots, lookup=metadata.distribution, environment=None) -> dict:
    """Resolve every version edge; revisit metadata when requested extras grow."""
    Requirement, canonicalize_name, Version, default_environment = _packaging()
    environment = default_environment() if environment is None else environment
    try:
        applicable = [root for root in roots if isinstance(root, Requirement) and _applicable(root, environment)]
        if len(applicable) != len(roots):
            if any(not isinstance(root, Requirement) for root in roots):
                raise HealthError("requirements_invalid")
        if not applicable:
            raise HealthError("requirements_empty")
        if len(roots) > MAX_ROOTS:
            raise HealthError("closure_limit")
        pending = applicable + [Requirement("packaging")]
        selected, extras, processed = {}, {}, {}
        edges = 0
        while pending:
            requirement = pending.pop()
            edges += 1
            if edges > MAX_EDGES:
                raise HealthError("closure_limit")
            if requirement.url:
                raise HealthError("metadata_invalid")
            name = canonicalize_name(requirement.name)
            if name not in selected:
                if len(selected) >= MAX_DISTRIBUTIONS:
                    raise HealthError("closure_limit")
                try:
                    dist = lookup(name)
                except metadata.PackageNotFoundError:
                    raise HealthError("distribution_missing", name) from None
                installed_name = dist.metadata.get("Name", "")
                if (
                    not isinstance(installed_name, str)
                    or not _NAME.fullmatch(installed_name)
                    or canonicalize_name(installed_name) != name
                ):
                    raise HealthError("metadata_invalid", name)
                selected[name] = dist
                extras[name] = {""}
            dist = selected[name]
            try:
                version = Version(dist.version)
            except (TypeError, ValueError):
                raise HealthError("metadata_invalid", name) from None
            if not requirement.specifier.contains(version, prereleases=True):
                raise HealthError("version_mismatch", name)
            extras[name].update(canonicalize_name(extra) for extra in requirement.extras)
            active_extras = frozenset(extras[name])
            if processed.get(name) == active_extras:
                continue
            processed[name] = active_extras
            for text in dist.requires or ():
                edges += 1
                if edges > MAX_EDGES:
                    raise HealthError("closure_limit")
                dependency = _requirement(text, "metadata_invalid")
                if _applicable(dependency, environment, active_extras):
                    pending.append(dependency)
                    if len(pending) + edges > MAX_EDGES:
                        raise HealthError("closure_limit")
        return selected
    except HealthError:
        raise
    except Exception:
        raise HealthError("metadata_invalid") from None


def _safe_path(path) -> bool:
    if not isinstance(path, str) or not path or len(path) > MAX_FIELD_CHARS:
        return False
    if any(unicodedata.category(character) in ("Cc", "Cs") for character in path):
        return False
    return not (path.startswith("/") or "\\" in path or ":" in path or PureWindowsPath(path).is_absolute())


def _valid_record(row):
    if len(row) != 3 or any(len(field) > MAX_FIELD_CHARS for field in row):
        return False
    path, digest, size = row
    if not path or any(unicodedata.category(character) in ("Cc", "Cs") for field in row for character in field):
        return False
    if size and (not size.isascii() or not size.isdecimal()):
        return False
    if digest:
        match = _HASH.fullmatch(digest)
        if not match:
            return False
        try:
            encoded = match[1].rstrip("=")
            decoded = base64.b64decode(encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True)
            if not decoded:
                return False
        except ValueError:
            return False
    return True


def audit_records(selected, prefix: Path) -> dict:
    """Audit raw RECORD inventory, checking containment before target existence."""
    result = {"runtime_file_count": 0, "failure_count": 0, "failures": [], "advisory_count": 0, "advisories": []}
    prefix = Path(prefix).resolve()
    total_rows = 0

    def add(code, name="", path="", advisory=False):
        count, examples = ("advisory_count", "advisories") if advisory else ("failure_count", "failures")
        result[count] += 1
        if len(result[examples]) < MAX_EXAMPLES:
            result[examples].append(HealthError(code, name, path).diagnostic())

    for name, dist in selected.items():
        try:
            raw = dist.read_text("RECORD")
            if not raw or not raw.strip():
                add("record_missing", name)
                continue
            if len(raw.encode("utf-8")) > MAX_RECORD_BYTES:
                add("record_limit", name)
                continue
            row_count = 0
            for row in csv.reader(io.StringIO(raw, newline=""), strict=True):
                row_count += 1
                total_rows += 1
                if row_count > MAX_RECORD_ROWS or total_rows > MAX_TOTAL_ROWS:
                    add("record_limit", name)
                    return result
                if not _valid_record(row):
                    add("record_invalid", name)
                    continue
                path = row[0]
                if not _safe_path(path):
                    add("unsafe_record_path", name)
                    continue
                target = Path(dist.locate_file(path)).resolve()
                if not target.is_relative_to(prefix):
                    add("unsafe_record_path", name)
                    continue
                runtime = bool(_NATIVE.search(path))
                if runtime:
                    result["runtime_file_count"] += 1
                if path.lower().endswith((".pyc", ".pyo")):
                    continue
                if not target.is_file():
                    add(
                        "runtime_file_missing" if runtime else "non_runtime_file_missing",
                        name,
                        path,
                        advisory=not runtime,
                    )
            if row_count == 0:
                add("record_missing", name)
        except (csv.Error, UnicodeError, OSError, ValueError, TypeError):
            add("record_invalid", name)
    if result["runtime_file_count"] == 0:
        add("runtime_files_empty")
    return result


def _imports_only() -> dict:
    result = {
        "schema_version": 1,
        "scope": "critical_library_imports",
        "status": "pass",
        "imports_checked": 0,
        "failure_count": 0,
        "failures": [],
    }
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        for module in CRITICAL_IMPORTS:
            result["imports_checked"] += 1
            try:
                importlib.import_module(module)
            except Exception as error:
                result["failure_count"] += 1
                result["failures"].append({"module": module, "exception": type(error).__name__[:MAX_DIAGNOSTIC_PATH]})
    if result["failure_count"]:
        result["status"] = "fail"
    return result


def check_imports() -> dict:
    """Require real subprocess success AND one complete typed pass receipt."""
    try:
        child = subprocess.run(
            [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--imports-only"],
            capture_output=True,
            timeout=IMPORT_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise HealthError("import_timeout") from None
    except OSError:
        raise HealthError("import_failed") from None
    if child.returncode != 0:
        raise HealthError("import_failed")
    try:
        if len(child.stdout) > MAX_OUTPUT_BYTES:
            raise ValueError
        receipt = json.loads(child.stdout)
        if not isinstance(receipt, dict):
            raise ValueError
        for field, expected in (
            ("schema_version", 1),
            ("imports_checked", len(CRITICAL_IMPORTS)),
            ("failure_count", 0),
        ):
            if type(receipt.get(field)) is not int or receipt[field] != expected:
                raise ValueError
        for field, expected in (("scope", "critical_library_imports"), ("status", "pass")):
            if type(receipt.get(field)) is not str or receipt[field] != expected:
                raise ValueError
        if type(receipt.get("failures")) is not list or receipt["failures"]:
            raise ValueError
        return receipt
    except (ValueError, TypeError, UnicodeError):
        raise HealthError("import_result_invalid") from None


def run_check(requirements: Path) -> dict:
    result = {
        "schema_version": 1,
        "scope": "core_runtime_dependencies",
        "status": "fail",
        "distribution_count": 0,
        "runtime_file_count": 0,
        "failure_count": 0,
        "failures": [],
        "advisory_count": 0,
        "advisories": [],
        "import_status": "not_checked",
        "imports_checked": 0,
    }
    try:
        selected = resolve_closure(parse_requirements(requirements))
        result["distribution_count"] = len(selected)
        result.update(audit_records(selected, Path(sys.prefix)))
        if result["failure_count"]:
            return result
        receipt = check_imports()
        result["imports_checked"] = receipt["imports_checked"]
        result["import_status"] = "pass"
        result["status"] = "pass"
    except Exception as error:
        failure = error if isinstance(error, HealthError) else HealthError("internal_error")
        result["failure_count"] += 1
        if len(result["failures"]) < MAX_EXAMPLES:
            result["failures"].append(failure.diagnostic())
        if failure.code.startswith("import_"):
            result["import_status"] = "fail"
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--requirements", type=Path)
    mode.add_argument("--imports-only", action="store_true")
    arguments = parser.parse_args(argv)
    result = _imports_only() if arguments.imports_only else run_check(arguments.requirements)
    print(json.dumps(result, ensure_ascii=True, separators=(",", ":")))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
