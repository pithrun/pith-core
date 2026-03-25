"""
skill_deployer.py — Cross-platform Agent Skill deployment (Python port of skill-deployer.js)

Deploys skills from ~/.claude/skills/ (canonical store) to:
- Claude Code: zero-copy (canonical IS discovery path)
- Cursor: zero-copy (reads ~/.claude/skills/ natively)
- Codex: symlink ~/.codex/skills/ → ~/.claude/skills/
- Cowork: file copy + manifest update to active UUID session slot

Zero external dependencies (stdlib only).
"""

import json
import os
import platform
import re
import shutil
import sys
from pathlib import Path


# --- Constants ---
HOME = Path.home()
CANONICAL_SKILLS_DIR = HOME / ".claude" / "skills"
COWORK_SKILLS_DIR = HOME / "Documents" / "Claude" / "skills"
MARKER_FILE = ".pith-managed"
MAX_SLOT_SEARCH_DEPTH = 3

SURFACES = {
    "claude-code": {"type": "zero-copy", "path": CANONICAL_SKILLS_DIR},
    "codex": {
        "type": "symlink",
        "path": HOME / ".codex" / "skills",
        "parentDir": HOME / ".codex",
    },
    "cowork": {
        "type": "copy",
        "basePath": HOME / "Library" / "Application Support" / "Claude"
                    / "local-agent-mode-sessions" / "skills-plugin",
    },
}

# Track last deployed Cowork slot for needsDeploy detection
_last_deployed_slot_path = None


def discover_skills():
    """Discover pith skills in canonical and Cowork directories.
    A valid skill = directory containing SKILL.md.
    Canonical wins on ID collision.
    """
    seen = {}
    for skills_dir in [COWORK_SKILLS_DIR, CANONICAL_SKILLS_DIR]:
        if not skills_dir.is_dir():
            continue
        for entry in sorted(skills_dir.iterdir()):
            if entry.is_dir() and (entry / "SKILL.md").is_file():
                seen[entry.name] = {"id": entry.name, "path": str(entry)}
    return list(seen.values())


def _sync_to_canonical(skills, log_fn):
    """One-way sync Cowork-sourced skills to canonical store."""
    synced = 0
    skipped = 0
    canonical = str(CANONICAL_SKILLS_DIR)
    CANONICAL_SKILLS_DIR.mkdir(parents=True, exist_ok=True)

    for skill in skills:
        if skill["path"].startswith(canonical):
            continue
        target = CANONICAL_SKILLS_DIR / skill["id"]
        if target.exists() and not (target / MARKER_FILE).exists():
            log_fn("sync", "skip", f"{skill['id']}: exists in canonical, not pith-managed")
            skipped += 1
            continue
        try:
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(skill["path"], target)
            (target / MARKER_FILE).write_text(json.dumps({
                "synced_at": _now_iso(),
                "source": skill["path"],
                "source_type": "cowork",
            }))
            synced += 1
        except OSError as e:
            log_fn("sync", "error", f"{skill['id']}: {e}")

    if synced:
        log_fn("sync", "ok", f"Synced {synced} skills from Cowork → canonical")
    return {"synced": synced, "skipped": skipped}


def _deploy_to_codex(skills, log_fn):
    """Deploy skills to Codex via symlinks."""
    parent_dir = SURFACES["codex"]["parentDir"]
    target_path = SURFACES["codex"]["path"]

    if not parent_dir.is_dir():
        log_fn("codex", "skip", "Codex not installed (~/.codex/ not found)")
        return {"status": "skipped", "reason": "codex_not_installed"}

    results = {"status": "ok", "skills": [], "warnings": []}

    if not target_path.exists():
        try:
            target_path.symlink_to(CANONICAL_SKILLS_DIR)
            log_fn("codex", "ok", f"Symlinked {target_path} → {CANONICAL_SKILLS_DIR}")
            results["skills"] = [s["id"] for s in skills]
            return results
        except OSError as e:
            return {"status": "error", "error": str(e)}

    if target_path.is_symlink():
        if target_path.resolve() == CANONICAL_SKILLS_DIR.resolve():
            log_fn("codex", "ok", "Symlink already correct")
            results["skills"] = [s["id"] for s in skills]
            return results
        log_fn("codex", "warn", f"Points to {os.readlink(target_path)}, managed by another tool")
        return {"status": "skipped", "reason": "managed_by_other_tool"}

    if not target_path.is_dir():
        return {"status": "skipped", "reason": "not_a_directory"}

    # Real directory — per-skill symlinks
    for skill in skills:
        skill_target = target_path / skill["id"]
        try:
            if skill_target.exists() or skill_target.is_symlink():
                if skill_target.is_symlink():
                    if skill_target.resolve() == Path(skill["path"]).resolve():
                        results["skills"].append(skill["id"])
                        continue
                    skill_target.unlink()
                elif not (skill_target / MARKER_FILE).exists():
                    results["warnings"].append(f"{skill['id']}: exists, not pith-managed")
                    continue
            skill_target.symlink_to(skill["path"])
            results["skills"].append(skill["id"])
        except OSError as e:
            results["warnings"].append(f"{skill['id']}: {e}")

    log_fn("codex", "ok", f"Per-skill symlinks: {len(results['skills'])}/{len(skills)}")
    return results


def _find_active_cowork_slot(base_path):
    """Find the active Cowork slot (most recently modified manifest.json)."""
    latest_ts = 0
    latest_path = None

    def walk(d, depth=0):
        nonlocal latest_ts, latest_path
        if depth > MAX_SLOT_SEARCH_DEPTH:
            return
        try:
            for entry in Path(d).iterdir():
                if entry.name == "manifest.json" and entry.is_file():
                    mtime = entry.stat().st_mtime
                    if mtime > latest_ts:
                        latest_ts = mtime
                        latest_path = entry
                elif entry.is_dir() and not entry.name.startswith("."):
                    walk(entry, depth + 1)
        except (PermissionError, OSError):
            pass

    walk(base_path)
    if not latest_path:
        return None
    return {"manifestPath": str(latest_path), "dir": str(latest_path.parent), "timestamp": latest_ts}


def _parse_skill_frontmatter(skill_md_path):
    """Parse YAML-ish frontmatter from a SKILL.md file."""
    try:
        content = Path(skill_md_path).read_text(encoding="utf-8")
        match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
        if not match:
            return {}
        fm = match.group(1)
        name_m = re.search(r"^name:\s*(.+)$", fm, re.MULTILINE)
        name = name_m.group(1).strip() if name_m else None
        desc_m = re.search(r"^description:\s*>?\s*\n?([\s\S]*?)(?=\n\w|\n---)", fm, re.MULTILINE)
        if desc_m:
            description = " ".join(line.strip() for line in desc_m.group(1).split("\n") if line.strip())
        else:
            inline_m = re.search(r"^description:\s*(?!>)(.+)$", fm, re.MULTILINE)
            description = inline_m.group(1).strip() if inline_m else None
        return {"name": name, "description": description}
    except (OSError, UnicodeDecodeError):
        return {}


def _update_cowork_manifest(manifest_path, skills, log_fn):
    """Update Cowork manifest.json with user skill entries."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(manifest.get("skills"), list):
        log_fn("cowork", "warn", "manifest.json has no skills array")
        return
    existing_ids = {s.get("skillId") for s in manifest["skills"]}
    now = _now_iso()
    added = 0
    for skill in skills:
        if skill["id"] not in existing_ids:
            meta = _parse_skill_frontmatter(os.path.join(skill["path"], "SKILL.md"))
            manifest["skills"].insert(0, {
                "skillId": skill["id"],
                "name": meta.get("name") or skill["id"],
                "description": meta.get("description") or "",
                "creatorType": "user",
                "updatedAt": now,
                "enabled": True,
            })
            added += 1
    if added:
        Path(manifest_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        log_fn("cowork", "ok", f"Manifest: {added} entries added")


def _deploy_to_cowork(skills, log_fn):
    """Deploy skills to Cowork via file copy + manifest update. macOS only."""
    global _last_deployed_slot_path

    if platform.system() != "Darwin":
        log_fn("cowork", "skip", "Cowork is macOS only")
        return {"status": "skipped", "reason": "not_macos"}

    base_path = SURFACES["cowork"]["basePath"]
    if not base_path.is_dir():
        log_fn("cowork", "skip", "Cowork not installed")
        return {"status": "skipped", "reason": "cowork_not_installed"}

    slot = _find_active_cowork_slot(base_path)
    if not slot:
        log_fn("cowork", "warn", "No active Cowork manifest.json found")
        return {"status": "error", "error": "no_active_slot"}

    skills_dir = Path(slot["dir"]) / "skills"
    results = {"status": "ok", "skills": [], "warnings": []}

    for skill in skills:
        target_dir = skills_dir / skill["id"]
        try:
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.copytree(skill["path"], target_dir)
            (target_dir / MARKER_FILE).write_text(json.dumps({
                "deployed_at": _now_iso(), "source": skill["path"],
            }))
            if (target_dir / "SKILL.md").is_file():
                results["skills"].append(skill["id"])
            else:
                results["warnings"].append(f"{skill['id']}: copy ok but SKILL.md missing")
        except OSError as e:
            results["warnings"].append(f"{skill['id']}: {e}")

    try:
        _update_cowork_manifest(slot["manifestPath"], skills, log_fn)
    except (OSError, json.JSONDecodeError) as e:
        results["warnings"].append(f"manifest update: {e}")

    _last_deployed_slot_path = slot["dir"]
    log_fn("cowork", "ok", f"Copied {len(results['skills'])}/{len(skills)} skills")
    return results


def _smoke_test(skills, surface_results):
    """Post-deploy smoke test — verify SKILL.md readable at each target."""
    passes = []
    failures = []

    # Claude Code / Cursor — canonical store
    for skill in skills:
        p = CANONICAL_SKILLS_DIR / skill["id"] / "SKILL.md"
        if p.is_file():
            passes.append({"surface": "claude-code", "skill": skill["id"]})
        else:
            failures.append({"surface": "claude-code", "skill": skill["id"], "reason": "SKILL.md missing"})

    # Codex
    codex = surface_results.get("codex", {})
    if codex.get("status") == "ok":
        codex_path = SURFACES["codex"]["path"]
        for sid in codex.get("skills", []):
            p = codex_path / sid / "SKILL.md"
            if p.is_file():
                passes.append({"surface": "codex", "skill": sid})
            else:
                failures.append({"surface": "codex", "skill": sid, "reason": "SKILL.md missing"})

    return {"passes": passes, "failures": failures}


def _now_iso():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def deploy_skills(status_only=False):
    """Main entry point. Deploy skills to all surfaces, or return status only."""
    if status_only:
        return get_deploy_status()

    if os.environ.get("SKILL_DEPLOY_DISABLED") == "true":
        return {"status": "disabled", "surfaces": {}, "logs": []}

    logs = []

    def log_fn(surface, level, msg):
        logs.append({"surface": surface, "level": level, "msg": msg, "ts": _now_iso()})

    skills = discover_skills()
    if not skills:
        log_fn("all", "info", "No skills found in canonical or Cowork stores")
        return {"status": "empty", "surfaces": {}, "logs": logs}

    # Sync Cowork → canonical
    sync_result = _sync_to_canonical(skills, log_fn)
    canonical_skills = discover_skills() if sync_result["synced"] > 0 else skills

    log_fn("all", "info", f"Found {len(canonical_skills)} skills in {CANONICAL_SKILLS_DIR}")
    results = {
        "status": "ok",
        "surfaces": {
            "claude-code": {"status": "ok", "skills": [s["id"] for s in canonical_skills], "note": "zero-copy"},
        },
        "logs": logs,
        "skillCount": len(canonical_skills),
        "syncResult": sync_result,
    }

    # Codex
    try:
        results["surfaces"]["codex"] = _deploy_to_codex(canonical_skills, log_fn)
    except Exception as e:
        log_fn("codex", "error", f"Unhandled: {e}")
        results["surfaces"]["codex"] = {"status": "error", "error": str(e)}

    # Cowork
    try:
        results["surfaces"]["cowork"] = _deploy_to_cowork(canonical_skills, log_fn)
    except Exception as e:
        log_fn("cowork", "error", f"Unhandled: {e}")
        results["surfaces"]["cowork"] = {"status": "error", "error": str(e)}

    # Smoke test
    smoke = _smoke_test(canonical_skills, results["surfaces"])
    results["smokeTest"] = smoke
    if smoke["failures"]:
        log_fn("all", "warn", f"Smoke test: {len(smoke['failures'])} failures")
    else:
        log_fn("all", "ok", f"Smoke test passed: {len(smoke['passes'])} surfaces verified")

    return results


def get_deploy_status():
    """Get current deployment status across all surfaces."""
    skills = discover_skills()
    status = {
        "canonicalDir": str(CANONICAL_SKILLS_DIR),
        "canonicalExists": CANONICAL_SKILLS_DIR.is_dir(),
        "skillCount": len(skills),
        "skills": [s["id"] for s in skills],
        "surfaces": {},
    }

    status["surfaces"]["claude-code"] = {
        "deployed": len(skills) > 0,
        "path": str(CANONICAL_SKILLS_DIR),
    }

    codex_path = SURFACES["codex"]["path"]
    if codex_path.exists() or codex_path.is_symlink():
        is_sym = codex_path.is_symlink()
        status["surfaces"]["codex"] = {
            "deployed": True,
            "isSymlink": is_sym,
            "target": str(os.readlink(codex_path)) if is_sym else None,
            "path": str(codex_path),
        }
    else:
        status["surfaces"]["codex"] = {"deployed": False, "path": str(codex_path)}

    if platform.system() == "Darwin":
        base_path = SURFACES["cowork"]["basePath"]
        if base_path.is_dir():
            slot = _find_active_cowork_slot(base_path)
            user_skills_in_manifest = 0
            needs_deploy = False
            if slot:
                try:
                    manifest = json.loads(Path(slot["manifestPath"]).read_text())
                    user_skills_in_manifest = sum(
                        1 for s in manifest.get("skills", []) if s.get("creatorType") == "user"
                    )
                except (OSError, json.JSONDecodeError):
                    pass
                needs_deploy = _last_deployed_slot_path != slot["dir"] or user_skills_in_manifest == 0
            status["surfaces"]["cowork"] = {
                "deployed": bool(slot) and user_skills_in_manifest > 0,
                "activeSlot": slot["dir"] if slot else None,
                "userSkillsInManifest": user_skills_in_manifest,
                "needsDeploy": needs_deploy,
                "path": str(base_path),
            }
        else:
            status["surfaces"]["cowork"] = {"deployed": False, "path": str(base_path)}
    else:
        status["surfaces"]["cowork"] = {"deployed": False, "reason": "not_macos"}

    return status
