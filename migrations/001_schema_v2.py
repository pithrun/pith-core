#!/usr/bin/env python3
"""
Migration 001: Schema v1 → v2
Phase 0: Schema & Graph Foundation

Transforms existing concept YAML files to include:
- Evidence strings → structured Evidence objects
- New fields: concept_type, scope_conditions, failure_modes
- New fields: change_type, change_reason, model_signature, updated_at
- New fields: content_hash, parent_hash
- Hypothesis: hypothesis_id, competing_with
- Association: evidence_refs, created_at, last_validated

Safe: Creates .pre_migration backup of each concept before modifying.
Idempotent: Skips concepts that already have structured evidence.
Reversible: .pre_migration files can restore original state.

Usage:
  python3 -m migrations.001_schema_v2 --dry-run
  python3 -m migrations.001_schema_v2
"""

import yaml
import json
import hashlib
import uuid
import os
import sys
import shutil
from pathlib import Path
from datetime import datetime

try:
    from app.profile import resolve_data_dir
    DATA_DIR = resolve_data_dir()
except ImportError:
    DATA_DIR = Path(os.environ.get("PITH_DATA_DIR", os.environ.get("DATA_DIR", "/app/data")))
CONCEPTS_DIR = DATA_DIR / "concepts"
ASSOCIATIONS_PATH = DATA_DIR / "associations" / "graph.json"


# Source type weights per Spec K
SOURCE_WEIGHTS = {
    "external_data": 0.9,
    "documented_observation": 0.85,
    "document": 0.85,
    "conversation": 0.7,
    "observation": 0.75,
    "inference": 0.6,
}


def compute_content_hash(data: dict) -> str:
    """Compute deterministic SHA-256 hash of concept content (Spec H.4)."""
    hashable = {
        "id": data.get("id", ""),
        "version": data.get("version", ""),
        "summary": data.get("summary", ""),
        "evidence": str(data.get("evidence", [])),
        "confidence": data.get("confidence", 0),
    }
    canonical = json.dumps(hashable, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def guess_source_type(evidence_str: str) -> str:
    """Infer source_type from evidence string content."""
    lower = evidence_str.lower()
    if any(kw in lower for kw in ["spec ", "mapcs", "section", "specification"]):
        return "document"
    if any(kw in lower for kw in ["user decided", "user stated", "user preference", "session"]):
        return "conversation"
    if any(kw in lower for kw in ["observed", "measured", "tested", "verified"]):
        return "observation"
    if any(kw in lower for kw in ["inferred", "derived", "suggests", "implies", "likely"]):
        return "inference"
    if any(kw in lower for kw in ["http", "url", "api", "external"]):
        return "external_data"
    return "conversation"  # Default


def migrate_evidence(evidence_list: list, created_at: str) -> list:
    """Convert string evidence to structured Evidence objects."""
    migrated = []
    for item in evidence_list:
        if isinstance(item, str):
            source_type = guess_source_type(item)
            migrated.append({
                "id": str(uuid.uuid4()),
                "source_type": source_type,
                "content": item,
                "source_reference": None,
                "timestamp": created_at,
                "model_origin": None,
                "reliability_weight": SOURCE_WEIGHTS.get(source_type, 0.7),
                "directness": 0.8,
                "consistency": 0.8,
                "corroboration_count": 0,
                "age_days": 0.0,
            })
        elif isinstance(item, dict) and "id" in item:
            # Already migrated — skip
            migrated.append(item)
        else:
            # Unknown format — wrap as string
            migrated.append({
                "id": str(uuid.uuid4()),
                "source_type": "conversation",
                "content": str(item),
                "source_reference": None,
                "timestamp": created_at,
                "model_origin": None,
                "reliability_weight": 0.7,
                "directness": 0.8,
                "consistency": 0.8,
                "corroboration_count": 0,
                "age_days": 0.0,
            })
    return migrated


def migrate_hypotheses(hypotheses: list) -> list:
    """Add hypothesis_id and competing_with to existing hypotheses."""
    migrated = []
    for h in hypotheses:
        if isinstance(h, dict):
            if "hypothesis_id" not in h:
                h["hypothesis_id"] = str(uuid.uuid4())
            if "competing_with" not in h:
                h["competing_with"] = []
            if "evidence_refs" not in h:
                h["evidence_refs"] = []
            migrated.append(h)
    return migrated


def migrate_concept_data(data: dict) -> dict:
    """Apply all v2 schema migrations to a concept dict."""
    created_at = data.get("created_at", datetime.utcnow().isoformat())

    # --- Evidence migration ---
    evidence = data.get("evidence", [])
    needs_evidence_migration = any(isinstance(e, str) for e in evidence)
    if needs_evidence_migration:
        data["evidence"] = migrate_evidence(evidence, created_at)

    # --- Spec A.1: concept_type ---
    if "concept_type" not in data:
        data["concept_type"] = "observation"  # Safe default

    # --- Spec A.2: scope & failure ---
    if "scope_conditions" not in data:
        data["scope_conditions"] = None
    if "failure_modes" not in data:
        data["failure_modes"] = None

    # --- Spec A.3: evolution metadata ---
    if "change_type" not in data:
        version = data.get("version", "v1")
        data["change_type"] = "creation" if version == "v1" else "refinement"
    if "change_reason" not in data:
        data["change_reason"] = "Pre-v2 migration — original creation"
    if "model_signature" not in data:
        data["model_signature"] = None
    if "updated_at" not in data:
        data["updated_at"] = data.get("last_accessed", created_at)

    # --- Spec H.4: cryptographic lineage ---
    if "content_hash" not in data:
        data["content_hash"] = compute_content_hash(data)
    if "parent_hash" not in data:
        data["parent_hash"] = None

    # --- Spec A.5: hypothesis migration ---
    if data.get("hypotheses"):
        data["hypotheses"] = migrate_hypotheses(data["hypotheses"])

    return data


def migrate_concept_dir(concept_dir: Path, dry_run: bool = False) -> dict:
    """Migrate all version files in a concept directory.
    
    Returns: {"concept_id": str, "versions_migrated": int, "skipped": bool, "error": str|None}
    """
    concept_id = concept_dir.name
    result = {"concept_id": concept_id, "versions_migrated": 0, "skipped": False, "error": None}
    
    try:
        # Find all version files (not latest.yaml which is a symlink/copy)
        version_files = sorted(concept_dir.glob("v*.yaml"))
        if not version_files:
            result["skipped"] = True
            return result
        
        # Check if already migrated (first evidence item is a dict with 'id')
        with open(version_files[-1], 'r') as f:
            check_data = yaml.safe_load(f)
        evidence = check_data.get("evidence", [])
        if evidence and isinstance(evidence[0], dict) and "id" in evidence[0]:
            result["skipped"] = True
            return result
        
        if not dry_run:
            # Backup entire concept dir
            backup_dir = concept_dir.parent / f"{concept_id}.pre_migration"
            if not backup_dir.exists():
                shutil.copytree(concept_dir, backup_dir)

        # Migrate each version file
        for vf in version_files:
            with open(vf, 'r') as f:
                data = yaml.safe_load(f)
            
            migrated = migrate_concept_data(data)
            
            if not dry_run:
                with open(vf, 'w') as f:
                    yaml.dump(migrated, f, default_flow_style=False, allow_unicode=True)
            
            result["versions_migrated"] += 1
        
        # Update latest.yaml (recreate symlink to latest version)
        if not dry_run:
            latest_path = concept_dir / "latest.yaml"
            latest_version = version_files[-1].name
            if latest_path.exists() or latest_path.is_symlink():
                latest_path.unlink()
            latest_path.symlink_to(latest_version)
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


def run_migration(dry_run: bool = False):
    """Run the full migration across all concepts."""
    print(f"\n{'='*60}")
    print(f"Migration 001: Schema v1 → v2")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"{'='*60}\n")
    
    if not CONCEPTS_DIR.exists():
        print(f"ERROR: Concepts directory not found: {CONCEPTS_DIR}")
        return
    
    concept_dirs = sorted([d for d in CONCEPTS_DIR.iterdir() if d.is_dir() and not d.name.endswith('.pre_migration')])
    print(f"Found {len(concept_dirs)} concept directories\n")
    
    stats = {"migrated": 0, "skipped": 0, "errors": 0, "versions_total": 0}
    errors = []
    
    for concept_dir in concept_dirs:
        result = migrate_concept_dir(concept_dir, dry_run=dry_run)
        
        if result["error"]:
            stats["errors"] += 1
            errors.append(result)
            print(f"  ✗ {result['concept_id']}: {result['error']}")
        elif result["skipped"]:
            stats["skipped"] += 1
        else:
            stats["migrated"] += 1
            stats["versions_total"] += result["versions_migrated"]
            if not dry_run:
                print(f"  ✓ {result['concept_id']} ({result['versions_migrated']} versions)")
    
    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"  Migrated: {stats['migrated']} concepts ({stats['versions_total']} version files)")
    print(f"  Skipped (already migrated): {stats['skipped']}")
    print(f"  Errors: {stats['errors']}")
    if dry_run:
        print(f"\n  This was a DRY RUN. No files were modified.")
        print(f"  Run without --dry-run to apply changes.")
    else:
        print(f"\n  Backups saved as <concept_id>.pre_migration directories.")
    print(f"{'='*60}\n")
    
    if errors:
        print("ERRORS:")
        for e in errors:
            print(f"  {e['concept_id']}: {e['error']}")
    
    return stats


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    run_migration(dry_run=dry_run)
