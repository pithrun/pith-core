#!/usr/bin/env python3
"""Public-surface safety scanner for pith-core pull requests."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ALLOWLIST_PATH = REPO_ROOT / ".github" / "public-safety-allowlist.txt"

TEXT_EXTENSIONS = {
    "",
    ".cfg",
    ".css",
    ".env",
    ".gitignore",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".lock",
    ".md",
    ".ps1",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}

BINARY_EXTENSIONS = {
    ".7z",
    ".avi",
    ".bin",
    ".bmp",
    ".class",
    ".dll",
    ".dmg",
    ".exe",
    ".gif",
    ".gz",
    ".ico",
    ".jar",
    ".jpg",
    ".jpeg",
    ".mov",
    ".mp4",
    ".pdf",
    ".png",
    ".so",
    ".tar",
    ".tgz",
    ".webp",
    ".zip",
}

ALLOWED_CHANGED_BINARY_PATHS = {
    "demo/demo.gif",
    "pith-server-latest.tar.gz",
}


@dataclass(frozen=True)
class Finding:
    source: str
    label: str
    value: str


def run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return result.stdout


def tracked_files() -> list[Path]:
    return [
        REPO_ROOT / line
        for line in run_git("ls-files").splitlines()
        if line and not line.startswith(".git/")
    ]


def changed_files() -> list[Path]:
    base_ref = os.environ.get("GITHUB_BASE_REF")
    if base_ref:
        names = run_git("diff", "--name-only", f"origin/{base_ref}...HEAD").splitlines()
        if names:
            return [REPO_ROOT / name for name in names]
    names = run_git("diff", "--name-only", "HEAD~1..HEAD").splitlines()
    return [REPO_ROOT / name for name in names if name]


def load_allowlist() -> set[str]:
    if not ALLOWLIST_PATH.exists():
        return set()
    entries: set[str] = set()
    for line in ALLOWLIST_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            entries.add(stripped)
    return entries


def is_allowed(value: str, allowlist: set[str]) -> bool:
    return any(entry in value or entry in str(value) for entry in allowlist)


def is_text_file(path: Path) -> bool:
    if path.suffix.lower() in BINARY_EXTENSIONS:
        return False
    if path.suffix.lower() not in TEXT_EXTENSIONS:
        return False
    try:
        sample = path.read_bytes()[:4096]
    except OSError:
        return False
    return b"\x00" not in sample


def risk_patterns() -> list[tuple[str, re.Pattern[str]]]:
    internal_repo = "-".join(["pith", "internal"])
    worktree_marker = "_session" + "_worktrees"
    rc_token = "_".join(["PITH", "INSTALL", "RC", "TOKEN"])
    release_rc = "-".join(["pith", "release", "rc"])
    archive_volume = "/" + "/".join(["Volumes", "Pith-Archive"])
    return [
        (
            "absolute local user path",
            re.compile(
                r"(?<![A-Za-z0-9_.-])(?:/Users/[A-Za-z0-9._-]+|/home/(?!pith(?:/|$)|root(?:/|$))[A-Za-z0-9._-]+)"
            ),
        ),
        ("windows user profile path", re.compile(r"[A-Za-z]:\\\\Users\\\\[A-Za-z0-9._-]+")),
        ("session worktree marker", re.compile(re.escape(worktree_marker), re.IGNORECASE)),
        ("internal repo name", re.compile(re.escape(internal_repo), re.IGNORECASE)),
        ("private archive path", re.compile(re.escape(archive_volume), re.IGNORECASE)),
        ("release candidate token name", re.compile(re.escape(rc_token), re.IGNORECASE)),
        ("install release-candidate path", re.compile(re.escape("install-" + "rc"), re.IGNORECASE)),
        ("release candidate bucket", re.compile(re.escape(release_rc), re.IGNORECASE)),
        ("private release phrase", re.compile(re.escape("internal " + "launch"), re.IGNORECASE)),
        ("review packet phrase", re.compile(re.escape("approval " + "packet"), re.IGNORECASE)),
        ("nonpublic staging phrase", re.compile(re.escape("private " + "staging"), re.IGNORECASE)),
        (
            "secret-like assignment",
            re.compile(
                r"(?i)(?:api[_-]?key|password|secret|token)"
                r"[A-Z0-9_ -]{0,32}\s*[:=]\s*['\"][A-Za-z0-9_./+=-]{12,}['\"]"
            ),
        ),
        ("openai-style secret", re.compile(r"sk-[A-Za-z0-9]{20,}")),
        ("bearer credential", re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]{20,}")),
    ]


def scan_text(source: str, text: str, allowlist: set[str]) -> list[Finding]:
    findings: list[Finding] = []
    for label, pattern in risk_patterns():
        for match in pattern.finditer(text):
            value = match.group(0)
            if label == "secret-like assignment":
                lowered = value.lower()
                if any(token in lowered for token in ("your-", "your_", "example", "placeholder")):
                    continue
            if not is_allowed(value, allowlist):
                findings.append(Finding(source, label, value))
    return findings


def metadata_text() -> str:
    parts: list[str] = []
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if event_path and Path(event_path).exists():
        try:
            event = json.loads(Path(event_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            event = {}
        pull_request = event.get("pull_request") or {}
        for key in ("title", "body"):
            value = pull_request.get(key)
            if isinstance(value, str):
                parts.append(f"pull_request.{key}: {value}")
    log = run_git("log", "--format=%s%n%b", "-20")
    if log:
        parts.append("commit_messages:\n" + log)
    return "\n\n".join(parts)


def binary_findings(paths: list[Path]) -> list[Finding]:
    findings: list[Finding] = []
    for path in paths:
        rel = path.relative_to(REPO_ROOT).as_posix()
        suffixes = "".join(path.suffixes).lower()
        if path.suffix.lower() not in BINARY_EXTENSIONS and not suffixes.endswith(".tar.gz"):
            continue
        if rel not in ALLOWED_CHANGED_BINARY_PATHS:
            findings.append(Finding(rel, "unexpected changed binary artifact", rel))
    return findings


def main() -> int:
    allowlist = load_allowlist()
    findings: list[Finding] = []

    for path in tracked_files():
        if path == ALLOWLIST_PATH or not path.exists() or not is_text_file(path):
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        findings.extend(scan_text(rel, text, allowlist))

    metadata = metadata_text()
    if metadata:
        findings.extend(scan_text("pull request / commit metadata", metadata, allowlist))

    findings.extend(binary_findings(changed_files()))

    if findings:
        print("Public-safety scan failed:", file=sys.stderr)
        for finding in findings:
            print(
                f"{finding.source}: {finding.label}: {finding.value}",
                file=sys.stderr,
            )
        return 1

    print("Public-safety scan passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
