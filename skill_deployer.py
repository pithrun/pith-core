"""
skill_deployer.py — Cross-platform Agent Skill deployment (Python port of skill-deployer.js)

Deploys skills from the active canonical store to:
- Claude Code: compatibility path under ~/.claude/skills/
- Cursor: compatibility path under ~/.claude/skills/
- Codex: managed materialized copies under ~/.codex/skills/
- Cowork: file copy + manifest update to active UUID session slot

Legacy mode keeps ~/.claude/skills/ as canonical. Canonical hub mode
(`PITH_SKILL_CANONICAL_HUB_ENABLED=true`) uses ~/.pith/skills/ as the only
authoritative source and treats client paths as generated compatibility surfaces.

Zero external dependencies (stdlib only).
"""

import json
import os
import platform
import re
import shutil
from pathlib import Path


# --- Constants ---
HOME = Path.home()
PITH_CANONICAL_SKILLS_DIR = HOME / ".pith" / "skills"
CLAUDE_COMPAT_SKILLS_DIR = HOME / ".claude" / "skills"
CANONICAL_SKILLS_DIR = CLAUDE_COMPAT_SKILLS_DIR  # legacy alias for callers/tests
COWORK_SKILLS_DIR = HOME / "Documents" / "Claude" / "skills"
LEGACY_DISCOVERY_DIRS = [COWORK_SKILLS_DIR, CLAUDE_COMPAT_SKILLS_DIR]
AUXILIARY_SKILL_DIRS = [HOME / ".agents" / "skills"]
REGISTRY_BACKUP_QUARANTINE_DIR = HOME / ".agents" / "skills-archive"
REGISTRY_BACKUP_DIR_RE = re.compile(r"^skills\.[^/]*-bak\.\d+$")
PREFERRED_CODEX_SKILL_ROOT = HOME / ".codex" / "skills"
REGISTRY_AUDIT_SAMPLE_LIMIT = 12
SKILL_DESCRIPTION_MAX_LENGTH = 1024
SKILL_DESCRIPTION_WARN_LENGTH = 850
SKILL_METADATA_HARD_ISSUES = {
    "missing_frontmatter",
    "missing_name",
    "missing_description",
    "description_over_1024",
    "read_error",
}
CANONICAL_HUB_FLAG = "PITH_SKILL_CANONICAL_HUB_ENABLED"
MARKER_FILE = ".pith-managed"
MAX_SLOT_SEARCH_DEPTH = 3

SURFACES = {
    "claude-code": {"type": "compat", "path": CLAUDE_COMPAT_SKILLS_DIR},
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

# SKILLS-001: Track ALL deployed Cowork slot paths (set) — supports multi-slot deployment
_last_deployed_slot_paths: set = set()


def _canonical_hub_enabled():
    return os.environ.get(CANONICAL_HUB_FLAG) == "true"


def _active_canonical_dir():
    if _canonical_hub_enabled():
        return PITH_CANONICAL_SKILLS_DIR
    return CANONICAL_SKILLS_DIR


def _legacy_discovery_dirs():
    return [COWORK_SKILLS_DIR, CANONICAL_SKILLS_DIR]


def _discover_from_dir(skills_dir):
    if not skills_dir.is_dir():
        return []
    return [
        {"id": entry.name, "path": str(entry)}
        for entry in sorted(skills_dir.iterdir())
        if entry.is_dir() and (entry / "SKILL.md").is_file()
    ]


def discover_skills():
    """Discover pith skills from the active source.

    Canonical hub mode reads only ~/.pith/skills. Legacy mode keeps the previous
    merged discovery behavior, with ~/.claude/skills winning collisions.
    A valid skill = directory containing SKILL.md.
    """
    if _canonical_hub_enabled():
        return _discover_from_dir(PITH_CANONICAL_SKILLS_DIR)

    seen = {}
    for skills_dir in _legacy_discovery_dirs():
        for skill in _discover_from_dir(skills_dir):
            seen[skill["id"]] = skill
    return list(seen.values())


def _hash_skill_tree(skill_dir):
    """Hash a full skill directory, excluding deploy marker files."""
    import hashlib

    root = Path(skill_dir)
    if root.is_symlink():
        root = root.resolve()
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.name == MARKER_FILE or not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


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


def migrate_legacy_skills(dry_run=False, log_fn=None):
    """Promote legacy skill stores into ~/.pith/skills with conflict detection."""
    log_fn = log_fn or (lambda *_: None)
    candidates = {}

    for skills_dir in _legacy_discovery_dirs():
        for skill in _discover_from_dir(skills_dir):
            candidates.setdefault(skill["id"], []).append(skill)

    promoted = []
    conflicts = []
    skipped = []

    for skill_id, entries in sorted(candidates.items()):
        hashes = {}
        for entry in entries:
            try:
                hashes.setdefault(_hash_skill_tree(entry["path"]), []).append(entry)
            except OSError as e:
                conflicts.append({"skill": skill_id, "reason": f"hash_error: {e}"})
        if len(hashes) > 1:
            conflicts.append({
                "skill": skill_id,
                "reason": "legacy_sources_disagree",
                "paths": [entry["path"] for entry in entries],
            })
            continue
        if not hashes:
            continue

        matching_entries = next(iter(hashes.values()))
        source = next(
            (entry for entry in matching_entries if Path(entry["path"]).parent == CANONICAL_SKILLS_DIR),
            matching_entries[0],
        )
        target = PITH_CANONICAL_SKILLS_DIR / skill_id
        source_hash = next(iter(hashes))

        if target.exists():
            try:
                target_hash = _hash_skill_tree(target)
            except OSError as e:
                conflicts.append({"skill": skill_id, "reason": f"target_hash_error: {e}"})
                continue
            if target_hash == source_hash:
                skipped.append({"skill": skill_id, "reason": "already_current"})
                continue
            conflicts.append({
                "skill": skill_id,
                "reason": "canonical_target_differs",
                "target": str(target),
                "source": source["path"],
            })
            continue

        promoted.append({"skill": skill_id, "source": source["path"], "target": str(target)})
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source["path"], target)

    if conflicts:
        log_fn("migration", "error", f"{len(conflicts)} skill conflict(s) detected")
        return {
            "status": "error",
            "dryRun": dry_run,
            "promoted": promoted if dry_run else [],
            "skipped": skipped,
            "conflicts": conflicts,
        }

    log_fn("migration", "ok", f"{len(promoted)} skill(s) promoted to {PITH_CANONICAL_SKILLS_DIR}")
    return {
        "status": "ok",
        "dryRun": dry_run,
        "promoted": promoted,
        "skipped": skipped,
        "conflicts": [],
    }


def _target_matches_source(target_dir, source_dir):
    try:
        return _hash_skill_tree(target_dir) == _hash_skill_tree(source_dir)
    except OSError:
        return False


def _deploy_to_codex(skills, log_fn):
    """Deploy skills to Codex as materialized copies.

    Codex registry discovery may not follow symlinked skill directories, so the
    generated surface keeps real directories even when the canonical store uses
    symlinks elsewhere.
    """
    parent_dir = SURFACES["codex"]["parentDir"]
    target_path = SURFACES["codex"]["path"]
    canonical_dir = _active_canonical_dir()

    if not parent_dir.is_dir():
        log_fn("codex", "skip", "Codex not installed (~/.codex/ not found)")
        return {"status": "skipped", "reason": "codex_not_installed"}

    results = {"status": "ok", "skills": [], "warnings": []}

    if not target_path.exists():
        try:
            target_path.mkdir(parents=True)
        except OSError as e:
            return {"status": "error", "error": str(e)}

    if target_path.is_symlink():
        if target_path.resolve() == canonical_dir.resolve():
            target_path.unlink()
            target_path.mkdir(parents=True)
            log_fn("codex", "ok", f"Replaced top-level symlink with materialized directory: {target_path}")
        else:
            log_fn("codex", "warn", f"Points to {os.readlink(target_path)}, managed by another tool")
            return {"status": "skipped", "reason": "managed_by_other_tool"}

    if not target_path.is_dir():
        return {"status": "skipped", "reason": "not_a_directory"}

    # Real directory — per-skill materialized copies
    for skill in skills:
        skill_target = target_path / skill["id"]
        source_dir = Path(skill["path"])
        try:
            if skill_target.exists() or skill_target.is_symlink():
                if skill_target.is_symlink():
                    if skill_target.resolve() == source_dir.resolve():
                        skill_target.unlink()
                    else:
                        results["warnings"].append(f"{skill['id']}: symlink managed by another tool")
                        continue
                elif not (skill_target / MARKER_FILE).exists() and not _target_matches_source(skill_target, skill["path"]):
                    results["warnings"].append(f"{skill['id']}: exists, not pith-managed")
                    continue
                else:
                    shutil.rmtree(skill_target)
            shutil.copytree(source_dir, skill_target)
            (skill_target / MARKER_FILE).write_text(json.dumps({
                "deployed_at": _now_iso(),
                "source": str(source_dir),
                "strategy": "codex_materialize",
            }))
            results["skills"].append(skill["id"])
        except OSError as e:
            results["warnings"].append(f"{skill['id']}: {e}")

    log_fn("codex", "ok", f"Materialized copies: {len(results['skills'])}/{len(skills)}")
    return results


def _deploy_to_claude_compat(skills, log_fn, strategy="symlink"):
    """Deploy canonical skills to ~/.claude/skills via symlinks or managed copies."""
    compat_path = SURFACES["claude-code"]["path"]
    compat_path.mkdir(parents=True, exist_ok=True)

    results = {"status": "ok", "skills": [], "warnings": [], "strategy": strategy}
    for skill in skills:
        source_dir = Path(skill["path"])
        target_dir = compat_path / skill["id"]
        try:
            if target_dir.is_symlink():
                if target_dir.resolve() == source_dir.resolve() and strategy == "symlink":
                    results["skills"].append(skill["id"])
                    continue
                results["warnings"].append(f"{skill['id']}: symlink managed by another tool")
                continue
            elif target_dir.exists():
                if (target_dir / MARKER_FILE).exists() or _target_matches_source(target_dir, source_dir):
                    shutil.rmtree(target_dir)
                else:
                    results["warnings"].append(f"{skill['id']}: exists in ~/.claude/skills and is unmanaged")
                    continue

            if strategy == "symlink":
                target_dir.symlink_to(source_dir, target_is_directory=True)
            else:
                shutil.copytree(source_dir, target_dir)
                (target_dir / MARKER_FILE).write_text(json.dumps({
                    "deployed_at": _now_iso(),
                    "source": str(source_dir),
                    "strategy": strategy,
                }))
            results["skills"].append(skill["id"])
        except OSError as e:
            results["warnings"].append(f"{skill['id']}: {e}")

    if results["warnings"] and not results["skills"]:
        results["status"] = "warning"
    log_fn("claude-code", "ok", f"{strategy} deploy: {len(results['skills'])}/{len(skills)} skills")
    return results


def _find_all_cowork_slots(base_path):
    """SKILLS-001: Return ALL Cowork session slots — eliminates session-targeting race condition.

    Prior heuristic (most recently created/modified slot) was unreliable: Cowork updates
    manifests independently, so the deploy could target a stale slot. Returning all slots
    guarantees every session gets skills, regardless of timing.
    """
    slots = []
    def walk(d, depth=0):
        if depth > MAX_SLOT_SEARCH_DEPTH:
            return
        try:
            for entry in Path(d).iterdir():
                if entry.name == "manifest.json" and entry.is_file():
                    slots.append({
                        "manifestPath": str(entry),
                        "dir": str(entry.parent),
                    })
                elif entry.is_dir() and not entry.name.startswith("."):
                    walk(entry, depth + 1)
        except (PermissionError, OSError):
            pass
    walk(base_path)
    return slots


def _parse_skill_frontmatter(skill_md_path):
    """Parse YAML-ish frontmatter from a SKILL.md file."""
    try:
        content = Path(skill_md_path).read_text(encoding="utf-8")
        match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
        if not match:
            return {"error": "missing_frontmatter"}
        fm = match.group(1)
        name_m = re.search(r"^name:\s*(.+)$", fm, re.MULTILINE)
        name = name_m.group(1).strip().strip("\"'") if name_m else None
        desc_m = re.search(r"^description:\s*>?\s*\n?([\s\S]*?)(?=\n\w|\n---)", fm, re.MULTILINE)
        if desc_m:
            description = " ".join(line.strip() for line in desc_m.group(1).split("\n") if line.strip())
        else:
            inline_m = re.search(r"^description:\s*(?!>)(.+)$", fm, re.MULTILINE)
            description = inline_m.group(1).strip().strip("\"'") if inline_m else None
        return {"name": name, "description": description}
    except (OSError, UnicodeDecodeError):
        return {"error": "read_error"}


def _skill_metadata_record(label, skill_id, skill_path):
    skill_md = Path(skill_path) / "SKILL.md"
    meta = _parse_skill_frontmatter(skill_md)
    issues = []
    if meta.get("error"):
        issues.append(meta["error"])
    if not meta.get("name"):
        issues.append("missing_name")
    if not meta.get("description"):
        issues.append("missing_description")
    description = meta.get("description") or ""
    description_length = len(description)
    if description_length > SKILL_DESCRIPTION_MAX_LENGTH:
        issues.append("description_over_1024")
    elif description_length >= SKILL_DESCRIPTION_WARN_LENGTH:
        issues.append("description_near_limit")
    return {
        "root": label,
        "skill": skill_id,
        "name": meta.get("name") or "",
        "descriptionLength": description_length,
        "issues": issues,
        "path": str(skill_md),
        "hardFailure": bool(SKILL_METADATA_HARD_ISSUES.intersection(issues)),
    }


def _summarize_skill_metadata(records):
    hard_records = [record for record in records if record["hardFailure"]]
    warning_records = [
        record for record in records
        if not record["hardFailure"] and "description_near_limit" in record["issues"]
    ]
    status = "error" if hard_records else "warning" if warning_records else "ok"
    return {
        "status": status,
        "summary": {
            "skillFilesAudited": len(records),
            "hardFailures": len(hard_records),
            "warnings": len(warning_records),
            "overLimit": sum("description_over_1024" in record["issues"] for record in records),
            "nearLimit": sum("description_near_limit" in record["issues"] for record in records),
        },
        "hardFailureSamples": hard_records[:REGISTRY_AUDIT_SAMPLE_LIMIT],
        "warningSamples": warning_records[:REGISTRY_AUDIT_SAMPLE_LIMIT],
    }


def _audit_skill_metadata_for_skills(label, skills):
    return _summarize_skill_metadata([
        _skill_metadata_record(label, skill["id"], skill["path"])
        for skill in skills
    ])


def _audit_skill_metadata_root(label, root):
    records = []
    for skill in _discover_from_dir(root):
        records.append(_skill_metadata_record(label, skill["id"], skill["path"]))
    return _summarize_skill_metadata(records)


def _audit_generated_skill_metadata():
    roots = {
        "claude_compat_direct": SURFACES["claude-code"]["path"],
        "codex_user": SURFACES["codex"]["path"],
    }
    for root in AUXILIARY_SKILL_DIRS:
        roots[f"auxiliary:{root.name}"] = root
    return {
        label: _audit_skill_metadata_root(label, root)
        for label, root in roots.items()
        if root.exists()
    }


def _is_registry_backup_dir(path):
    return bool(REGISTRY_BACKUP_DIR_RE.match(Path(path).name))


def _iter_recursive_skill_files(root):
    if not root.is_dir():
        return []
    return [path for path in sorted(root.rglob("SKILL.md")) if path.is_file()]


def _path_has_registry_backup_component(path):
    return any(_is_registry_backup_dir(part) for part in Path(path).parts)


def _skill_name_from_file(skill_md_path):
    meta = _parse_skill_frontmatter(skill_md_path)
    return meta.get("name") or Path(skill_md_path).parent.name


def _audit_cross_root_duplicates(canonical_skills):
    roots = [PREFERRED_CODEX_SKILL_ROOT] + AUXILIARY_SKILL_DIRS
    by_name = {}
    for root in roots:
        if not root.is_dir():
            continue
        for skill in _discover_from_dir(root):
            skill_md = Path(skill["path"]) / "SKILL.md"
            name = _skill_name_from_file(skill_md)
            by_name.setdefault(name, []).append(skill["path"])

    duplicates = {
        name: paths
        for name, paths in sorted(by_name.items())
        if len(paths) > 1
    }
    canonical_ids = {skill["id"] for skill in canonical_skills}
    duplicate_canonical_ids = sorted(name for name in duplicates if name in canonical_ids)
    return {
        "status": "duplicate_roots" if duplicates else "ok",
        "preferredRoot": str(PREFERRED_CODEX_SKILL_ROOT),
        "duplicateSkillNames": duplicate_canonical_ids,
        "duplicateCount": len(duplicates),
        "samples": [
            {"name": name, "paths": paths}
            for name, paths in list(duplicates.items())[:REGISTRY_AUDIT_SAMPLE_LIMIT]
        ],
    }


def _dedupe_paths(paths):
    seen = set()
    result = []
    for path in paths:
        candidate = Path(path)
        key = str(candidate.resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result


def _registry_pollution_roots():
    """Roots whose recursive backup trees can pollute client skill registries."""
    return _dedupe_paths([
        SURFACES["claude-code"]["path"],
        SURFACES["codex"]["path"],
        *AUXILIARY_SKILL_DIRS,
    ])


def _safe_timestamp():
    return re.sub(r"[^0-9A-Za-z._-]", "-", _now_iso())


def _quarantine_registry_backup_dirs(root, log_fn):
    """Move top-level skills.*-bak.* directories out of active registry roots."""
    root = Path(root)
    result = {"root": str(root), "quarantined": [], "errors": []}
    if not root.is_dir():
        return result

    for entry in sorted(root.iterdir()):
        if entry.is_symlink() or not entry.is_dir() or not _is_registry_backup_dir(entry.name):
            continue
        quarantine_root = REGISTRY_BACKUP_QUARANTINE_DIR / f"skill-registry-pollution-{_safe_timestamp()}"
        destination = quarantine_root / entry.name
        suffix = 1
        while destination.exists():
            suffix += 1
            destination = quarantine_root / f"{entry.name}.{suffix}"
        try:
            quarantine_root.mkdir(parents=True, exist_ok=True)
            shutil.move(str(entry), str(destination))
            result["quarantined"].append({
                "source": str(entry),
                "destination": str(destination),
            })
            log_fn("registry", "ok", f"Quarantined backup directory: {entry}")
        except OSError as e:
            result["errors"].append({"path": str(entry), "error": str(e)})
            log_fn("registry", "warn", f"Could not quarantine backup directory {entry}: {e}")
    return result


def _audit_registry_pollution(canonical_skills):
    roots = []
    total_skill_files = 0
    backup_skill_files = []
    names = {}
    errors = []

    for root in _registry_pollution_roots():
        root_result = {"path": str(root)}
        if not root.exists():
            root_result["status"] = "missing"
            roots.append(root_result)
            continue
        if not root.is_dir():
            root_result["status"] = "error"
            root_result["error"] = "not_a_directory"
            roots.append(root_result)
            errors.append({"path": str(root), "error": "not_a_directory"})
            continue
        try:
            skill_files = _iter_recursive_skill_files(root)
        except OSError as e:
            error = str(e)
            roots.append({"path": str(root), "status": "error", "error": error})
            errors.append({"path": str(root), "error": error})
            continue

        total_skill_files += len(skill_files)
        for skill_md in skill_files:
            name = _skill_name_from_file(skill_md)
            names.setdefault(name, []).append(str(skill_md))
            if _path_has_registry_backup_component(skill_md):
                backup_skill_files.append(str(skill_md))
        roots.append({"path": str(root), "status": "ok", "skillFiles": len(skill_files)})

    duplicated = {name: paths for name, paths in names.items() if len(paths) > 1}
    cross_root = _audit_cross_root_duplicates(canonical_skills)
    status = "ok"
    if errors:
        status = "error"
    elif backup_skill_files:
        status = "polluted"
    elif cross_root["status"] != "ok":
        status = "duplicate_roots"

    return {
        "status": status,
        "roots": roots,
        "totalSkillFiles": total_skill_files,
        "backupSkillFiles": len(backup_skill_files),
        "duplicatedSkillNames": len(duplicated),
        "maxDuplicateCount": max((len(paths) for paths in duplicated.values()), default=0),
        "sampleBackupPaths": backup_skill_files[:REGISTRY_AUDIT_SAMPLE_LIMIT],
        "crossRootDuplicates": cross_root,
        "errors": errors,
        "recommendation": (
            "run quarantine dry-run; restart Codex; re-check active skills"
            if backup_skill_files else
            "verify Codex loader root precedence/deduplication if active skills remain wrong"
        ),
    }


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

    # SKILLS-001: Deploy to ALL slots — not just the "most recently created" one
    slots = _find_all_cowork_slots(base_path)
    if not slots:
        log_fn("cowork", "warn", "No Cowork manifest.json found")
        return {"status": "error", "error": "no_slots_found"}

    results = {"status": "ok", "skills": [], "warnings": [], "slotsDeployed": 0}

    for slot in slots:
        skills_dir = Path(slot["dir"]) / "skills"
        slot_skill_count = 0

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
                    slot_skill_count += 1
                    if skill["id"] not in results["skills"]:
                        results["skills"].append(skill["id"])
                else:
                    results["warnings"].append(f"{skill['id']}@{slot['dir']}: SKILL.md missing")
            except OSError as e:
                results["warnings"].append(f"{skill['id']}@{slot['dir']}: {e}")

        try:
            _update_cowork_manifest(slot["manifestPath"], skills, log_fn)
        except (OSError, json.JSONDecodeError) as e:
            results["warnings"].append(f"manifest@{slot['dir']}: {e}")

        _last_deployed_slot_paths.add(slot["dir"])
        results["slotsDeployed"] += 1

    # MONITOR-125: Track slotsDeployed count for adoption monitoring
    if results["slotsDeployed"] > 0:
        try:
            import subprocess as _subp
            _subp.run(
                ["curl", "-s", "-X", "POST", "http://localhost:8000/metrics/record",
                 "-H", "Content-Type: application/json",
                 "-d", f'{{"metric":"cowork_slots_deployed","value":{results["slotsDeployed"]}}}'],
                timeout=1, capture_output=True
            )
        except Exception:
            pass
        import logging as _sl_log
        _sl_log.getLogger("skill_deployer").info(
            "MONITOR-125: Cowork slotsDeployed=%d skills=%d",
            results["slotsDeployed"], len(results["skills"])
        )
        log_fn("cowork", "ok", f"Slot {slot['dir']}: {slot_skill_count}/{len(skills)} skills")

    # A1: Error if all slots failed
    if results["slotsDeployed"] == 0:
        results["status"] = "error"
        log_fn("cowork", "error", "All slot deployments failed")

    log_fn("cowork", "ok", f"Deployed to {results['slotsDeployed']} slot(s), {len(results['skills'])} unique skills")
    return results


def _smoke_test(skills, surface_results):
    """Post-deploy smoke test — verify SKILL.md readable at each target."""
    passes = []
    failures = []

    # Claude Code / Cursor — compatibility discovery path
    claude_path = SURFACES["claude-code"]["path"]
    for skill in skills:
        p = claude_path / skill["id"] / "SKILL.md"
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


def _verify_surface_parity(skills, surface_results, log_fn):
    """Compare generated surfaces against canonical skill directories."""
    parity = {"status": "ok", "drift": [], "skipped": []}
    surfaces = ("claude-code", "codex")

    for surface in surfaces:
        result = surface_results.get(surface, {})
        if result.get("status") not in {"ok", "warning"}:
            parity["skipped"].append({"surface": surface, "reason": result.get("reason", "not_deployed")})
            continue
        surface_path = SURFACES[surface]["path"]
        for skill in skills:
            canonical_dir = Path(skill["path"])
            target_dir = surface_path / skill["id"]
            if not target_dir.exists() and not target_dir.is_symlink():
                parity["drift"].append({"surface": surface, "skill": skill["id"], "reason": "missing"})
                continue
            if target_dir.is_symlink():
                resolved = target_dir.resolve()
                if resolved != canonical_dir.resolve():
                    parity["drift"].append({
                        "surface": surface,
                        "skill": skill["id"],
                        "reason": "symlink_target_mismatch",
                        "target": str(resolved),
                    })
                    continue
            if _hash_skill_tree(target_dir) != _hash_skill_tree(canonical_dir):
                parity["drift"].append({"surface": surface, "skill": skill["id"], "reason": "hash_mismatch"})

    if _canonical_hub_enabled():
        parity["auxiliaryAudit"] = _audit_auxiliary_skill_dirs(skills)
        parity["registryPollution"] = _audit_registry_pollution(skills)

    if parity["drift"]:
        parity["status"] = "drift"
        log_fn("parity", "warn", f"{len(parity['drift'])} drift item(s) detected")
    else:
        log_fn("parity", "ok", "No drift detected")
    return parity


def _audit_auxiliary_skill_dirs(skills):
    canonical_by_id = {skill["id"]: skill for skill in skills}
    audit = []
    for skills_dir in AUXILIARY_SKILL_DIRS:
        if not skills_dir.is_dir():
            audit.append({"path": str(skills_dir), "status": "missing"})
            continue
        drift = []
        for skill in _discover_from_dir(skills_dir):
            canonical = canonical_by_id.get(skill["id"])
            if not canonical:
                drift.append({"skill": skill["id"], "reason": "not_in_canonical"})
                continue
            if _hash_skill_tree(skill["path"]) != _hash_skill_tree(canonical["path"]):
                drift.append({"skill": skill["id"], "reason": "hash_mismatch"})
        missing = sorted(set(canonical_by_id) - {skill["id"] for skill in _discover_from_dir(skills_dir)})
        drift.extend({"skill": skill_id, "reason": "missing_from_auxiliary"} for skill_id in missing)
        audit.append({"path": str(skills_dir), "status": "drift" if drift else "ok", "drift": drift})
    return audit


def repair_generated_surfaces(skills, log_fn):
    """Restore generated surfaces from canonical content.

    Directory targets for canonical skill IDs are treated as generated drift and
    replaced. Symlinks pointing outside the canonical skill remain skipped because
    they may be owned by another tool.
    """
    skipped = []
    backup_quarantine = []
    for root in _registry_pollution_roots():
        quarantine_result = _quarantine_registry_backup_dirs(root, log_fn)
        if quarantine_result["quarantined"] or quarantine_result["errors"]:
            backup_quarantine.append(quarantine_result)

    for surface in ("claude-code", "codex"):
        surface_path = Path(SURFACES[surface]["path"])
        for skill in skills:
            target = surface_path / skill["id"]
            if target.is_symlink():
                resolved = target.resolve()
                canonical = Path(skill["path"]).resolve()
                if resolved != canonical:
                    skipped.append({"surface": surface, "skill": skill["id"], "reason": "unmanaged_symlink"})
                    continue
            elif target.exists() and not _target_matches_source(target, skill["path"]):
                shutil.rmtree(target)
    results = {
        "status": "ok" if not skipped and not any(r["errors"] for r in backup_quarantine) else "warning",
        "skipped": skipped,
        "backupQuarantine": backup_quarantine,
        "surfaces": {
            "claude-code": _deploy_to_claude_compat(skills, log_fn, strategy="symlink"),
            "codex": _deploy_to_codex(skills, log_fn),
        },
    }
    return results


def _now_iso():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _deploy_skills_legacy(logs, log_fn):
    """Legacy deploy path: ~/.claude/skills remains canonical."""
    skills = discover_skills()
    if not skills:
        log_fn("all", "info", "No skills found in canonical or Cowork stores")
        return {"status": "empty", "surfaces": {}, "logs": logs}

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

    try:
        results["surfaces"]["codex"] = _deploy_to_codex(canonical_skills, log_fn)
    except Exception as e:
        log_fn("codex", "error", f"Unhandled: {e}")
        results["surfaces"]["codex"] = {"status": "error", "error": str(e)}

    try:
        results["surfaces"]["cowork"] = _deploy_to_cowork(canonical_skills, log_fn)
    except Exception as e:
        log_fn("cowork", "error", f"Unhandled: {e}")
        results["surfaces"]["cowork"] = {"status": "error", "error": str(e)}

    smoke = _smoke_test(canonical_skills, results["surfaces"])
    results["smokeTest"] = smoke
    if smoke["failures"]:
        log_fn("all", "warn", f"Smoke test: {len(smoke['failures'])} failures")
    else:
        log_fn("all", "ok", f"Smoke test passed: {len(smoke['passes'])} surfaces verified")
    return results


def deploy_skills(*, status_only=False, migrate=False, dry_run=False, repair=False, verify_parity=False):
    """Main entry point. Deploy skills to all surfaces, or return status only."""
    if status_only:
        return get_deploy_status(verify_parity=verify_parity)

    if os.environ.get("SKILL_DEPLOY_DISABLED") == "true":
        return {"status": "disabled", "surfaces": {}, "logs": []}

    logs = []

    def log_fn(surface, level, msg):
        logs.append({"surface": surface, "level": level, "msg": msg, "ts": _now_iso()})

    skills = discover_skills()
    metadata_preflight = _audit_skill_metadata_for_skills("active_canonical", skills)
    if metadata_preflight["status"] == "error":
        log_fn("metadata", "error", "Skill metadata preflight failed")
        return {
            "status": "error",
            "error": "skill_metadata_preflight_failed",
            "metadataAudit": {"canonical": metadata_preflight},
            "logs": logs,
        }
    if metadata_preflight["status"] == "warning":
        log_fn("metadata", "warn", "Skill metadata preflight has warnings")

    if not _canonical_hub_enabled():
        result = _deploy_skills_legacy(logs, log_fn)
        result["metadataAudit"] = {"canonical": metadata_preflight}
        return result

    migration_result = None
    if migrate:
        migration_result = migrate_legacy_skills(dry_run=dry_run, log_fn=log_fn)
        if migration_result["status"] == "error":
            return {"status": "error", "migration": migration_result, "logs": logs}
        if dry_run:
            return {"status": "dry_run", "migration": migration_result, "logs": logs}

    if not skills:
        log_fn("all", "info", "No skills found in Pith canonical store")
        return {"status": "empty", "surfaces": {}, "logs": logs, "migration": migration_result}

    log_fn("all", "info", f"Found {len(skills)} skills in {PITH_CANONICAL_SKILLS_DIR}")
    results = {
        "status": "ok",
        "surfaces": {},
        "logs": logs,
        "skillCount": len(skills),
        "migration": migration_result,
        "featureFlagEnabled": True,
        "metadataAudit": {"canonical": metadata_preflight},
    }

    try:
        claude_result = _deploy_to_claude_compat(skills, log_fn, strategy="symlink")
        results["surfaces"]["claude-code"] = claude_result
        claude_smoke = _smoke_test(skills, results["surfaces"])
        claude_failures = [f for f in claude_smoke["failures"] if f["surface"] == "claude-code"]
        if claude_failures and claude_result.get("strategy") == "symlink":
            log_fn("claude-code", "warn", "symlink compatibility failed; falling back to materialize")
            results["surfaces"]["claude-code"] = _deploy_to_claude_compat(
                skills, log_fn, strategy="materialize"
            )
    except Exception as e:
        log_fn("claude-code", "error", f"Unhandled: {e}")
        results["surfaces"]["claude-code"] = {"status": "error", "error": str(e)}

    try:
        results["surfaces"]["codex"] = _deploy_to_codex(skills, log_fn)
    except Exception as e:
        log_fn("codex", "error", f"Unhandled: {e}")
        results["surfaces"]["codex"] = {"status": "error", "error": str(e)}

    try:
        results["surfaces"]["cowork"] = _deploy_to_cowork(skills, log_fn)
    except Exception as e:
        log_fn("cowork", "error", f"Unhandled: {e}")
        results["surfaces"]["cowork"] = {"status": "error", "error": str(e)}

    if repair:
        results["repair"] = repair_generated_surfaces(skills, log_fn)
        results["surfaces"].update(results["repair"].get("surfaces", {}))

    smoke = _smoke_test(skills, results["surfaces"])
    results["smokeTest"] = smoke
    if smoke["failures"]:
        log_fn("all", "warn", f"Smoke test: {len(smoke['failures'])} failures")
    else:
        log_fn("all", "ok", f"Smoke test passed: {len(smoke['passes'])} surfaces verified")

    if verify_parity or not status_only:
        results["parity"] = _verify_surface_parity(skills, results["surfaces"], log_fn)

    results["metadataAudit"]["generated"] = _audit_generated_skill_metadata()
    generated_failures = sum(
        audit["summary"]["hardFailures"]
        for audit in results["metadataAudit"]["generated"].values()
    )
    if generated_failures:
        results["status"] = "error"
        results["error"] = "generated_skill_metadata_post_deploy_failed"
        log_fn("metadata", "error", f"{generated_failures} generated metadata hard failure(s)")

    return results


def get_deploy_status(*, verify_parity=False):
    """Get current deployment status across all surfaces."""
    skills = discover_skills()
    canonical_dir = _active_canonical_dir()
    status = {
        "canonicalDir": str(canonical_dir),
        "canonicalExists": canonical_dir.is_dir(),
        "featureFlagEnabled": _canonical_hub_enabled(),
        "skillCount": len(skills),
        "skills": [s["id"] for s in skills],
        "surfaces": {},
        "metadataAudit": {
            "canonical": _audit_skill_metadata_for_skills("active_canonical", skills)
        },
    }

    status["surfaces"]["claude-code"] = {
        "deployed": len(skills) > 0,
        "path": str(SURFACES["claude-code"]["path"]),
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
            # SKILLS-001: Check ALL slots — needsDeploy if any slot has 0 user skills
            slots = _find_all_cowork_slots(base_path)
            slots_with_user_skills = 0
            slot_summary = []
            for slot in slots:
                user_skills_in_slot = 0
                try:
                    manifest = json.loads(Path(slot["manifestPath"]).read_text())
                    user_skills_in_slot = sum(
                        1 for s in manifest.get("skills", []) if s.get("creatorType") == "user"
                    )
                except (OSError, json.JSONDecodeError):
                    pass
                if user_skills_in_slot > 0:
                    slots_with_user_skills += 1
                slot_summary.append({
                    "dir": slot["dir"],
                    "userSkills": user_skills_in_slot,
                    "deployed": slot["dir"] in _last_deployed_slot_paths,
                })
            needs_deploy = len(slots) > 0 and slots_with_user_skills < len(slots)
            status["surfaces"]["cowork"] = {
                "deployed": slots_with_user_skills > 0,
                "slotsTotal": len(slots),
                "slotsWithUserSkills": slots_with_user_skills,
                "needsDeploy": needs_deploy,
                "slotSummary": slot_summary,
                "path": str(base_path),
            }
        else:
            status["surfaces"]["cowork"] = {"deployed": False, "path": str(base_path)}
    else:
        status["surfaces"]["cowork"] = {"deployed": False, "reason": "not_macos"}

    if _canonical_hub_enabled():
        status["registryPollution"] = _audit_registry_pollution(skills)
        status["metadataAudit"]["generated"] = _audit_generated_skill_metadata()

    if verify_parity:
        surface_results = {
            name: {"status": "ok" if info.get("deployed") else "skipped"}
            for name, info in status["surfaces"].items()
        }
        status["parity"] = _verify_surface_parity(skills, surface_results, lambda *_: None)

    return status


_AUTO_DEPLOY_COOLDOWN_SECONDS = 300
_last_auto_deploy_ts = 0.0


def _deployer_owned_generated_hard_failures(status):
    """Return hard-failure counts for generated surfaces this deployer owns."""
    generated = status.get("metadataAudit", {}).get("generated", {})
    watched_roots = ("codex_user", "claude_compat_direct")
    return {
        root: generated[root]["summary"]["hardFailures"]
        for root in watched_roots
        if generated.get(root, {}).get("summary", {}).get("hardFailures", 0) > 0
    }

def auto_deploy_if_needed():
    """Auto-deploy skills to Cowork if needed, with 5-minute debounce.

    Called from conversation_turn pipeline. Returns deploy result or None if skipped.
    """
    import time
    global _last_auto_deploy_ts

    now = time.time()
    if now - _last_auto_deploy_ts < _AUTO_DEPLOY_COOLDOWN_SECONDS:
        return None

    # Check if deploy is needed
    try:
        status = deploy_skills(status_only=True)
        cowork_status = status.get("surfaces", {}).get("cowork", {})
        needs_repair = bool(_deployer_owned_generated_hard_failures(status))
        if not cowork_status.get("needsDeploy", False) and not needs_repair:
            return None
    except Exception:
        return None

    # Deploy — set cooldown AFTER deploy so failed deploys don't eat retry window (DEBT-225)
    result = deploy_skills(status_only=False, repair=needs_repair)
    _last_auto_deploy_ts = time.time()

    # Post-deploy verification
    verify = deploy_skills(status_only=True)
    cowork_verify = verify.get("surfaces", {}).get("cowork", {})
    if cowork_verify.get("needsDeploy", False):
        result["warning"] = "Post-deploy verification failed: skills still need deploy"
    generated_verify_failures = _deployer_owned_generated_hard_failures(verify)
    if needs_repair and generated_verify_failures:
        result["warning"] = (
            "Post-deploy verification failed: generated skill metadata still has hard failures"
        )

    return result
