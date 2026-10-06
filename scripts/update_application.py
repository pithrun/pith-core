#!/usr/bin/env python3
"""OPS-602: update application and wrapper through their existing channel owner."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import platform as platform  # exported to the source-only private adapter
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import urllib.parse
import urllib.request
import uuid
import zipfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

PUBLIC_REPO = "pithrun/pith-core"
PUBLIC_ASSETS = {"install.sh", "install.ps1", "pith-server-latest.tar.gz", "pith-server-latest.zip",
                 "pith-server-latest.sha256", "pith-server-latest.zip.sha256"}
MAX_DOWNLOAD = 64 * 1024 * 1024
MAX_EXPANDED = 256 * 1024 * 1024
MAX_ENTRIES = 100000
VERSION = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")
OWNED_ROOTS = {"app", "pith_client", "scripts", "migrations", "integrations"}
LOCAL_CONFIG = {".env", ".pith-deploy.json"}
TEXT_EXTENSIONS = {".py", ".sh", ".ps1", ".cmd", ".json", ".toml", ".txt", ".md", ".example"}


class UpdateRefused(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def git_environment(env=None):
    # Hooks export repository routing/config variables. Never let them override
    # the explicitly selected source, worktree, or deploy target in a child.
    source = os.environ if env is None else env
    transport = {"GIT_SSH", "GIT_SSH_COMMAND", "GIT_SSH_VARIANT", "GIT_TERMINAL_PROMPT"}
    return {name: value for name, value in source.items()
            if not name.startswith("GIT_") or name in transport}


def run(args, *, env=None, timeout=60, log=None):
    clean_env = git_environment(env)
    try:
        result = subprocess.run([str(a) for a in args], env=clean_env, timeout=timeout,
                                text=True, stdout=log or subprocess.PIPE,
                                stderr=log or subprocess.PIPE, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise UpdateRefused("operation_unavailable") from exc
    if result.returncode:
        raise UpdateRefused("operation_failed")
    return (result.stdout or "").strip()


def git(root: Path, *args):
    return run(["git", "-C", root, *args])


def version(value: str):
    if not isinstance(value, str):
        raise UpdateRefused("invalid_version")
    match = VERSION.fullmatch(value)
    if not match:
        raise UpdateRefused("invalid_version")
    return tuple(int(n) for n in match.groups())


def source_version(root: Path) -> str:
    no_links(root / "scripts/install.sh")
    text = (root / "scripts/install.sh").read_text(encoding="utf-8")
    return installer_version(text)


def installer_version(text: str) -> str:
    match = re.search(r'^PITH_VERSION="([^"]+)"$', text, re.M)
    if not match:
        raise UpdateRefused("installed_version_unknown")
    version(match[1])
    return match[1]


def is_origin(value: str, repo: str):
    return value in {f"git@github.com:{repo}.git", f"https://github.com/{repo}.git",
                     f"https://github.com/{repo}"}


def no_links(path: Path):
    # Inspect ancestors too: resolving first would hide an external link/junction.
    for item in (path, *path.parents):
        if item.exists() or item.is_symlink():
            info = item.lstat()
            if item.is_symlink() or getattr(info, "st_file_attributes", 0) & 0x400:
                raise UpdateRefused("linked_installation")


def inventory(root: Path):
    no_links(root)
    files = {}
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in dirs + names:
            path = Path(directory) / name
            no_links(path)
            if path.is_file():
                files[path.relative_to(root).as_posix()] = path
            elif not path.is_dir():
                raise UpdateRefused("special_installation_file")
            if len(files) > MAX_ENTRIES:
                raise UpdateRefused("installation_too_large")
    return files


def owned_bytecode(name: str):
    parts = PurePosixPath(name).parts
    if not name.endswith(".pyc"):
        return False
    return ((len(parts) > 1 and parts[0] in OWNED_ROOTS)
            or (len(parts) == 1 and name in {"pith_mcp.pyc", "skill_deployer.pyc"})
            or (len(parts) == 2 and parts[0] == "__pycache__"
                and re.fullmatch(r"(pith_mcp|skill_deployer)\.[^.]+(?:\.opt-[0-9]+)?\.pyc", parts[1])))


def normalized(name: str, data: bytes):
    return (data.removeprefix(b"\xef\xbb\xbf").replace(b"\r\n", b"\n")
            if PurePosixPath(name).suffix in TEXT_EXTENSIONS else data)


def member_name(name: str):
    if not isinstance(name, str) or any(ord(char) < 32 for char in name) or len(name.encode("utf-8")) > 4096:
        raise UpdateRefused("unsafe_archive")
    if "\\" in name or ":" in name or name.startswith("/"):
        raise UpdateRefused("unsafe_archive")
    parts = PurePosixPath(name).parts
    if ".." in parts or not parts:
        raise UpdateRefused("unsafe_archive")
    if any(len(part.encode("utf-8")) > 255 for part in parts):
        raise UpdateRefused("unsafe_archive")
    return PurePosixPath(*parts).as_posix()


def archive_files(data: bytes, windows: bool):
    files, seen, expanded, count = {}, set(), 0, 0

    def accept(name, size, link, directory):
        nonlocal expanded, count
        count += 1
        name = member_name(name)
        key = name.casefold() if windows else name
        if windows and any(part.rstrip(" .") != part or
                           any(char in part for char in '<>"|?*') or
                           re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", part)
                           for part in PurePosixPath(name).parts):
            raise UpdateRefused("unsafe_archive")
        if link or key in seen or count > MAX_ENTRIES or size < 0:
            raise UpdateRefused("unsafe_archive")
        seen.add(key)
        expanded += size
        if expanded > MAX_EXPANDED:
            raise UpdateRefused("archive_too_large")
        return None if directory else name

    try:
        if windows:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                for entry in archive.infolist():
                    mode = entry.external_attr >> 16
                    name = accept(entry.filename, entry.file_size, stat.S_ISLNK(mode), entry.is_dir())
                    if name:
                        files[name] = archive.read(entry)
        else:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
                for entry in archive:
                    name = accept(entry.name, entry.size, not (entry.isfile() or entry.isdir()), entry.isdir())
                    if name:
                        files[name] = archive.extractfile(entry).read()
    except (ValueError, OSError, tarfile.TarError, zipfile.BadZipFile) as exc:
        raise UpdateRefused("invalid_archive") from exc
    for required in ("app/api/server.py", "pith_client/cli.py", "pith_mcp.py", "requirements.txt", "scripts/install.sh"):
        if required not in files:
            raise UpdateRefused("incomplete_archive")
    return files


class TrustedRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlparse(newurl)
        if parsed.scheme != "https" or parsed.hostname not in {"github.com", "release-assets.githubusercontent.com", "objects.githubusercontent.com"}:
            raise UpdateRefused("untrusted_redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(url: str):
    if not isinstance(url, str) or len(url) > 2048:
        raise UpdateRefused("untrusted_download")
    parsed = urllib.parse.urlparse(url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise UpdateRefused("untrusted_download") from exc
    if parsed.scheme != "https" or parsed.hostname not in {"api.github.com", "github.com"} or parsed.username or parsed.password or port not in (None, 443) or parsed.fragment or parsed.query:
        raise UpdateRefused("untrusted_download")
    request = urllib.request.Request(url, headers={"User-Agent": "Pith-Application-Updater", "Accept": "application/vnd.github+json"})
    with urllib.request.build_opener(TrustedRedirect()).open(request, timeout=30) as response:
        data = response.read(MAX_DOWNLOAD + 1)
    if len(data) > MAX_DOWNLOAD:
        raise UpdateRefused("download_too_large")
    return data


def release(tag="latest"):
    if tag != "latest":
        version(tag)
    suffix = "latest" if tag == "latest" else "tags/" + tag
    info = json.loads(download(f"https://api.github.com/repos/{PUBLIC_REPO}/releases/{suffix}"))
    if not isinstance(info, dict) or not isinstance(info.get("assets"), list):
        raise UpdateRefused("release_unverified")
    version(info["tag_name"])
    if info.get("draft") or info.get("prerelease") or (tag != "latest" and info["tag_name"] != tag):
        raise UpdateRefused("unstable_release")
    return info


def asset(info, name: str):
    if not isinstance(name, str) or name not in PUBLIC_ASSETS:
        raise UpdateRefused("release_asset_name_invalid")
    if not isinstance(info, dict) or not isinstance(info.get("assets"), list):
        raise UpdateRefused("release_asset_unverified")
    version(info.get("tag_name"))
    entries = [item for item in info["assets"] if isinstance(item, dict) and item.get("name") == name]
    if len(entries) != 1:
        raise UpdateRefused("release_asset_missing")
    entry = entries[0]
    url = f"https://github.com/{PUBLIC_REPO}/releases/download/{info['tag_name']}/{name}"
    digest = entry.get("digest", "")
    if entry.get("browser_download_url") != url or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise UpdateRefused("release_asset_unverified")
    size = entry.get("size")
    if not isinstance(size, int) or isinstance(size, bool) or not 0 < size <= MAX_DOWNLOAD:
        raise UpdateRefused("release_asset_size_invalid")
    data = download(url)
    if len(data) != size or hashlib.sha256(data).hexdigest() != digest[7:]:
        raise UpdateRefused("release_asset_mismatch")
    return data


def compare_installed(root: Path, expected):
    actual = inventory(root)
    for name, data in expected.items():
        path = actual.get(name)
        if path is None or normalized(name, path.read_bytes()) != normalized(name, data):
            raise UpdateRefused("application_modified")
    for name in actual.keys() - expected.keys():
        if name not in LOCAL_CONFIG and not owned_bytecode(name):
            raise UpdateRefused("application_extra_files")
    return actual


def compare_wrapper(home: Path, installed_version: str):
    root = home / "pith-server"
    if os.name == "nt":
        path = home / "bin/pith-cli.ps1"
        text = (root / "scripts/templates/pith_cli.ps1").read_text(encoding="utf-8-sig")
        # Respect the existing managed venv selection rather than guessing a path.
        existing = path.read_text(encoding="utf-8-sig")
        match = re.search(r'^\$VenvPath = "([^"]+)"', existing, re.M)
        if not match:
            raise UpdateRefused("wrapper_modified")
        text = text.replace("__PITH_HOME__", str(home)).replace("__VENV_PATH__", match[1]).replace("__PITH_VERSION__", installed_version)
    else:
        path = wrapper(home)
        source = (root / "scripts/install.sh").read_text(encoding="utf-8")
        marker = 'cat > "$PITH_HOME/bin/pith" << \'PITH_CLI_SCRIPT\''
        start = source.index(marker)
        begin = source.index("\n", start) + 1
        text = source[begin:source.index("\nPITH_CLI_SCRIPT", begin)].replace("__PITH_VERSION__", installed_version)
    no_links(path)
    if normalized(path.name, path.read_bytes()).rstrip(b"\n") != normalized(path.name, text.encode()).rstrip(b"\n"):
        raise UpdateRefused("wrapper_modified")


def wrapper(home: Path):
    return home / "bin" / ("pith.cmd" if os.name == "nt" else "pith")


def wrapper_command(home: Path, *args):
    path = wrapper(home)
    no_links(path)
    if os.name == "nt":
        # cmd parsing is avoided; call the PowerShell generator's installed file.
        return ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", home / "bin/pith-cli.ps1", *args]
    return [path, *args]


def settings(home: Path):
    result = {}
    for relative in ("config/api.key", ".env", "pith-server/.env"):
        path = home / relative
        no_links(path)
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8-sig")
        if relative == "config/api.key":
            result[relative] = hashlib.sha256(text.strip().encode()).hexdigest()
        else:
            for line in text.splitlines():
                key, sep, value = line.partition("=")
                if sep and key in {"PITH_API_KEY", "PITH_DATA_DIR", "PITH_PROFILE", "PITH_PORT", "PORT"}:
                    result[relative + ":" + key] = hashlib.sha256(value.strip().encode()).hexdigest()
    return result


@contextmanager
def update_lock(home: Path):
    path = home / "state/application-update.lock"
    no_links(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise UpdateRefused("update_lock_present") from exc
    try:
        yield
    finally:
        path.rmdir()


def recovery(home: Path, channel: str, target: str):
    parent = home / "state/recovery"
    no_links(parent)
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / ("application-update-" + uuid.uuid4().hex)
    path.mkdir(mode=0o700)
    for relative in ("bin/pith", "bin/pith.cmd", "bin/pith-cli.ps1", "config/api.key",
                     "config/venv.path", "config/python-runtime.json", ".env"):
        source = home / relative
        if not source.exists():
            continue
        no_links(source)
        target_path = path / relative
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            inventory(source)
            shutil.copytree(source, target_path)
        else:
            shutil.copy2(source, target_path)
    if channel == "public":
        inventory(home / "pith-server")
        shutil.copytree(home / "pith-server", path / "pith-server")
    receipt = {"schema_version": 1, "channel": channel, "target": target, "status": "prepared"}
    write_receipt(path, receipt)
    return path, receipt


def write_receipt(path: Path, receipt):
    temp = path / ".receipt.tmp"
    temp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.chmod(0o600)
    temp.replace(path / "receipt.json")


def health():
    port = os.environ.get("PITH_PORT", "8000")
    if not port.isdigit() or not 1 <= int(port) <= 65535:
        raise UpdateRefused("invalid_port")
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
        return json.load(response)


def confirm_installed(home: Path, target_version: str, commit=None):
    output = run(wrapper_command(home, "version"))
    if not re.search(r"^Pith v" + re.escape(target_version) + r"\s*$", output, re.M):
        raise UpdateRefused("wrapper_version_mismatch")
    observed = health()
    if observed.get("version") != target_version or observed.get("mode") != "ready" or observed.get("retrieval_state") != "ready":
        raise UpdateRefused("runtime_not_ready")
    if commit and (observed.get("runtime_head") != commit or observed.get("git_commit_matches_head") is not True):
        raise UpdateRefused("runtime_commit_mismatch")
    return {"version": target_version, "commit": commit, "ready": True}


def private_plan(home: Path):
    # The private source adapter is intentionally absent from public artifacts.
    # A public installation cannot acquire private authority through a flag.
    adapter_path = Path(__file__).resolve().with_name("private_application_update.py")
    no_links(adapter_path)
    if not adapter_path.is_file():
        raise UpdateRefused("private_adapter_unavailable")
    spec = importlib.util.spec_from_file_location("_pith_private_application_update", adapter_path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    return adapter.private_plan(home, sys.modules[__name__])


def apply_private(home: Path, plan):
    # Recheck after taking the update lock; never substitute a newer target silently.
    current = private_plan(home)
    if current != plan:
        raise UpdateRefused("target_changed")
    path, receipt = recovery(home, "private", plan["target"])
    receipt["previous_commit"] = plan["current"]
    env = os.environ.copy()
    env["PITH_HOME"] = str(home)
    env["PITH_CANONICAL_REPO_ROOT"] = plan["canonical"]
    env["PITH_DEPLOY_EXPECTED_COMMIT"] = plan["target"]
    env["POST_DEPLOY_CLI_CANARY_INSTALL"] = "0"
    try:
        with (path / "operation.log").open("w", encoding="utf-8") as log:
            run(["bash", Path(plan["canonical"]) / "scripts/deploy.sh"], env=env, timeout=1800, log=log)
        if git(Path(plan["runtime"]), "rev-parse", "HEAD") != plan["target"]:
            raise UpdateRefused("runtime_commit_mismatch")
        receipt["proof"] = confirm_installed(home, plan["version"], plan["target"])
        receipt["status"] = "complete"
    except Exception:
        receipt["status"] = "failed"
        raise
    finally:
        write_receipt(path, receipt)
        print("Update recovery: " + str(path), file=sys.stderr)
    return receipt


def public_plan(home: Path):
    no_links(home / "pith-server")
    if (home / "pith-server/.git").exists():
        raise UpdateRefused("git_installation_requires_private_workflow")
    installed = source_version(home / "pith-server")
    latest = release()
    target = latest["tag_name"].removeprefix("v")
    if version(target) < version(installed):
        raise UpdateRefused("release_downgrade")
    return {"channel": "public", "current": installed, "target": target, "release": latest}


def apply_public(home: Path, plan):
    windows = os.name == "nt"
    package = "pith-server-latest.zip" if windows else "pith-server-latest.tar.gz"
    installer = "install.ps1" if windows else "install.sh"
    baseline = asset(release("v" + plan["current"]), package)
    old_files = archive_files(baseline, windows)
    compare_installed(home / "pith-server", old_files)
    compare_wrapper(home, plan["current"])
    payload = asset(plan["release"], package)
    target_files = archive_files(payload, windows)
    if installer_version(target_files["scripts/install.sh"].decode("utf-8")) != plan["target"]:
        raise UpdateRefused("payload_version_mismatch")
    script = asset(plan["release"], installer)
    if windows:
        matches = re.findall(r'(?m)^\s*\[string\]\$PithVersion\s*=\s*"([^"]+)"', script.decode("utf-8-sig"))
        script_version = matches[0] if len(matches) == 1 else None
    else:
        script_version = installer_version(script.decode("utf-8-sig"))
    if script_version != plan["target"]:
        raise UpdateRefused("installer_version_mismatch")
    if source_version(home / "pith-server") != plan["current"]:
        raise UpdateRefused("target_changed")
    before = settings(home)
    if "config/api.key" not in before:
        raise UpdateRefused("installed_key_missing")
    path, receipt = recovery(home, "public", plan["target"])
    (path / package).write_bytes(payload)
    (path / installer).write_bytes(script)
    checksum = package + (".sha256" if windows else "")
    if not windows:
        checksum = "pith-server-latest.sha256"
    (path / checksum).write_text(hashlib.sha256(payload).hexdigest() + "  " + package + "\n", encoding="ascii")
    env = os.environ.copy()
    env.update(PITH_HOME=str(home), PITH_RELEASE_CHANNEL="public", PITH_LOCAL_ONLY_INSTALL="1",
               PITH_REQUIRE_LOCAL_PACKAGE="1", PITH_SELECTED_SURFACES="none", PITH_SKIP_PAUSES="1",
               PITH_SKIP_CLIENT_SETUP="1", PITH_SKIP_CLIENT_UI="1", PITH_TELEMETRY_DISABLED="1",
               PITH_SKIP_GLOBAL_CLI_LINK="1")
    command = (["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", path / installer]
               if windows else ["bash", path / installer])
    try:
        compare_installed(home / "pith-server", old_files)
        compare_wrapper(home, plan["current"])
        run(wrapper_command(home, "stop"), timeout=60)
        # The baseline proved every application file is owned. Retain that entire
        # tree so obsolete modules cannot survive Windows' overlay installation.
        # Known server configuration is carried into the new managed child.
        old_tree = path / "retired-server"
        (home / "pith-server").replace(old_tree)
        try:
            compare_installed(old_tree, old_files)
        except Exception:
            old_tree.replace(home / "pith-server")
            run(wrapper_command(home, "start"), timeout=60)
            raise UpdateRefused("application_changed_during_update")
        (home / "pith-server").mkdir()
        if (old_tree / ".env").exists():
            shutil.copy2(old_tree / ".env", home / "pith-server/.env")
        with (path / "operation.log").open("w", encoding="utf-8") as log:
            run(command, env=env, timeout=1800, log=log)
        compare_installed(home / "pith-server", target_files)
        receipt["proof"] = confirm_installed(home, plan["target"])
        after = settings(home)
        if any(after.get(key) != value for key, value in before.items()):
            raise UpdateRefused("protected_settings_changed")
        receipt["asset_sha256"] = hashlib.sha256(payload).hexdigest()
        receipt["status"] = "complete"
    except Exception:
        receipt["status"] = "failed"
        raise
    finally:
        write_receipt(path, receipt)
        print("Update recovery: " + str(path), file=sys.stderr)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path.home() / ".pith")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    home = args.home.absolute()
    try:
        if not home.is_dir():
            raise UpdateRefused("installation_missing")
        private = (home / "pith-server").is_symlink()
        plan = private_plan(home) if private else public_plan(home)
        if args.check:
            result = {k: v for k, v in plan.items() if k != "release"}
            result["status"] = "available" if plan["current"] != plan["target"] else "current"
        else:
            with update_lock(home):
                result = apply_private(home, plan) if private else apply_public(home, plan)
        print(json.dumps(result, sort_keys=True) if args.json else
              f"Pith {result['channel']} update: {result['status']} ({result['target']})")
        return 0
    except Exception as exc:
        code = exc.code if isinstance(exc, UpdateRefused) else "update_unavailable"
        print(json.dumps({"status": "refused", "reason": code}) if args.json else "Pith update refused: " + code,
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
