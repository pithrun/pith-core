"""User-facing trust-governance correction preview and profile-local apply."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.core.profile import resolve_data_dir
from app.trust_governance_corrections import classify_trust_governance_correction

OVERRIDE_SCHEMA_VERSION = "trust_governance_authority_overrides.v1"
ENVELOPE_SCHEMA_VERSION = "trust_correction_envelope.v1"
APPLY_SCHEMA_VERSION = "trust_correction_apply_result.v1"
IMPLICIT_CANDIDATE_SCHEMA_VERSION = "trust_implicit_supersession_candidates.v1"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def authority_override_path(data_dir: Path | None = None) -> Path:
    root = data_dir or resolve_data_dir()
    return root / "trust_governance" / "authority_overrides.json"


def _empty_overrides(now: str | None = None) -> dict[str, Any]:
    stamp = now or _utc_now_iso()
    return {
        "schema_version": OVERRIDE_SCHEMA_VERSION,
        "created_at": stamp,
        "updated_at": stamp,
        "corrections": [],
        "domain_overrides": {},
    }


def load_authority_overrides(path: Path | None = None) -> dict[str, Any]:
    override_path = path or authority_override_path()
    if not override_path.exists():
        return _empty_overrides()
    try:
        payload = json.loads(override_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty_overrides()
    if not isinstance(payload, dict) or payload.get("schema_version") != OVERRIDE_SCHEMA_VERSION:
        return _empty_overrides()
    if not isinstance(payload.get("domain_overrides"), dict):
        return _empty_overrides()
    if not isinstance(payload.get("corrections"), list):
        return _empty_overrides()
    return payload


def save_authority_overrides(payload: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    override_path = path or authority_override_path()
    override_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path: Path | None = None
    if override_path.exists():
        backup_root = resolve_data_dir() / "backups" if path is None else override_path.parent / "backups"
        backup_path = backup_root / f"trust_governance_authority_overrides_{_stamp()}.json"
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(override_path, backup_path)

    fd, tmp_name = tempfile.mkstemp(prefix=f"{override_path.name}.", suffix=".tmp", dir=str(override_path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        tmp_path.replace(override_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return {
        "override_path": str(override_path),
        "backup_path": str(backup_path) if backup_path else None,
    }


def build_correction_preview(question: str | None, message: str | None) -> dict[str, Any]:
    question_text = (question or "").strip()
    message_text = (message or "").strip()
    if not question_text:
        return _error("question must be a non-empty string", "question")
    if not message_text:
        return _error("message must be a non-empty string", "message")

    from app.trust_governance import resolve_trust_governance

    classified = classify_trust_governance_correction(message_text)
    resolution = resolve_trust_governance(question_text)
    now = _utc_now_iso()
    current_id = resolution.current_authority.item_id if resolution.current_authority else None
    target_ids = [current_id] if current_id else []
    if classified.get("candidate") is not True:
        ambiguity_state = "not_a_correction"
        proposed_action = "none"
    elif resolution.intent == "ambiguous_domain":
        ambiguity_state = "ambiguous_domain"
        proposed_action = "none"
    elif resolution.intent == "no_known_authority" or not resolution.domain_id:
        ambiguity_state = "no_known_authority"
        proposed_action = "none"
    elif classified.get("intent") in {"mark_source_of_truth", "mark_current"} and current_id:
        ambiguity_state = "unambiguous"
        proposed_action = "apply_authority_override"
    else:
        ambiguity_state = "review_required"
        proposed_action = "persist_candidate"

    envelope = {
        "schema_version": ENVELOPE_SCHEMA_VERSION,
        "operation": "preview_correction",
        "source_message": message_text,
        "question": question_text,
        "intent": classified.get("intent"),
        "domain_id": resolution.domain_id,
        "target_item_ids": target_ids,
        "replacement_text": message_text,
        "ambiguity_state": ambiguity_state,
        "confidence": float(classified.get("confidence") or 0.0),
        "proposed_action": proposed_action,
        "mutates_authority": False,
        "evidence": {
            "source_type": "user_correction",
            "created_at": now,
            "current_authority_before": current_id,
            "matched_domain_ids": list(resolution.matched_domain_ids),
        },
        "next_answer_expectation": {
            "expected_domain_id": resolution.domain_id,
            "expected_current_title": _title_from_message(message_text),
        },
    }
    envelope["envelope_hash"] = _stable_hash(envelope)
    return envelope


def apply_correction(question: str | None, message: str | None, *, confirm: bool = False) -> dict[str, Any]:
    envelope = build_correction_preview(question, message)
    if envelope.get("error"):
        return envelope
    if not confirm:
        return {
            "schema_version": APPLY_SCHEMA_VERSION,
            "status": "rejected",
            "mutates_authority": False,
            "reason": "confirm_required",
            "envelope": envelope,
        }
    if envelope.get("proposed_action") != "apply_authority_override":
        return {
            "schema_version": APPLY_SCHEMA_VERSION,
            "status": "rejected",
            "mutates_authority": False,
            "reason": str(envelope.get("ambiguity_state") or "not_applicable"),
            "envelope": envelope,
        }

    from app.trust_governance import clear_trust_governance_registry_cache, resolve_trust_governance

    before = resolve_trust_governance(str(envelope["question"]))
    overrides = load_authority_overrides()
    now = _utc_now_iso()
    overrides.setdefault("schema_version", OVERRIDE_SCHEMA_VERSION)
    overrides.setdefault("created_at", now)
    overrides["updated_at"] = now
    corrections = overrides.setdefault("corrections", [])
    domain_overrides = overrides.setdefault("domain_overrides", {})
    domain_id = str(envelope["domain_id"])
    override_id = _override_item_id(domain_id, str(envelope["envelope_hash"]))
    current_before = before.current_authority.item_id if before.current_authority else None
    item = {
        "id": override_id,
        "title": _title_from_message(str(envelope["replacement_text"])),
        "state": "current",
        "scope": f"user correction for {domain_id}",
        "governs": True,
        "superseded_by": None,
        "replaces": [current_before] if current_before else [],
        "aliases": [str(envelope["replacement_text"]), "user correction", "current source of truth"],
        "source_path": f"profile://trust-governance/correction/{envelope['envelope_hash']}",
        "source_commit": None,
        "source_url": None,
        "reason": f"User-confirmed correction applied at {now}.",
        "concept_ids": [],
    }
    corrections.append({**envelope, "applied_at": now, "override_item_id": override_id})
    domain_overrides[domain_id] = {
        "current_item": item,
        "previous_current_item_id": current_before,
        "envelope_hash": envelope["envelope_hash"],
        "updated_at": now,
    }
    save_result = save_authority_overrides(overrides)
    clear_trust_governance_registry_cache()
    after = resolve_trust_governance(str(envelope["question"]))
    after_current = after.current_authority.item_id if after.current_authority else None
    return {
        "schema_version": APPLY_SCHEMA_VERSION,
        "status": "applied" if after_current == override_id else "applied_unverified",
        "mutates_authority": True,
        "override_path": save_result["override_path"],
        "backup_path": save_result["backup_path"],
        "envelope_hash": envelope["envelope_hash"],
        "before": {"current_authority": current_before},
        "after": {"current_authority": after_current},
        "next_answer_proof": {
            "status": "current" if after_current == override_id else "unknown",
            "domain_id": after.domain_id,
            "current_authority": after.evidence.get("current_authority"),
        },
        "envelope": envelope,
    }


def build_implicit_supersession_candidates() -> dict[str, Any]:
    from app.trust_governance import load_trust_governance_registry

    candidates: list[dict[str, Any]] = []
    for domain in load_trust_governance_registry():
        current = domain.by_id().get(domain.current_item_id)
        if current is None:
            continue
        for item in domain.items:
            if item.item_id == current.item_id or not item.is_historical:
                continue
            candidates.append(
                {
                    "domain_id": domain.domain_id,
                    "candidate_class": "review_candidate",
                    "old_item_id": item.item_id,
                    "replacement_item_id": current.item_id,
                    "mutates_authority": False,
                    "reason": "historical or scoped item has a newer current authority in the same domain",
                    "evidence": {
                        "old_source_path": item.source_path,
                        "replacement_source_path": current.source_path,
                    },
                }
            )
    return {
        "schema_version": IMPLICIT_CANDIDATE_SCHEMA_VERSION,
        "status": "review_only",
        "implicit_auto_apply_rate": 0.0,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def merge_authority_overrides(registry: Sequence[Any], overrides: Mapping[str, Any]) -> tuple[Any, ...]:
    domain_overrides = overrides.get("domain_overrides") if isinstance(overrides, Mapping) else None
    if not isinstance(domain_overrides, Mapping):
        return tuple(registry)
    merged: list[Any] = []
    for domain in registry:
        raw_override = domain_overrides.get(domain.domain_id)
        if not isinstance(raw_override, Mapping):
            merged.append(domain)
            continue
        raw_item = raw_override.get("current_item")
        if not isinstance(raw_item, Mapping):
            merged.append(domain)
            continue
        override_item = _item_from_override(raw_item)
        previous_id = str(raw_override.get("previous_current_item_id") or domain.current_item_id)
        items = []
        replaced = False
        for item in domain.items:
            if item.item_id == override_item.item_id:
                items.append(override_item)
                replaced = True
            elif item.item_id == previous_id and item.item_id != override_item.item_id:
                items.append(replace(item, state="superseded", governs=False, superseded_by=override_item.item_id))
            else:
                items.append(item)
        if not replaced:
            items.insert(0, override_item)
        merged.append(replace(domain, current_item_id=override_item.item_id, items=tuple(items)))
    return tuple(merged)


def clear_trust_governance_correction_caches() -> None:
    from app.trust_governance import clear_trust_governance_registry_cache

    clear_trust_governance_registry_cache()


def _item_from_override(raw: Mapping[str, Any]) -> Any:
    from app.trust_governance import TrustGovernanceItem

    return TrustGovernanceItem(
        item_id=str(raw["id"]).strip(),
        title=str(raw["title"]).strip(),
        state=str(raw.get("state") or "current").strip(),
        scope=str(raw.get("scope") or "user correction").strip(),
        governs=bool(raw.get("governs", True)),
        superseded_by=_optional_string(raw.get("superseded_by")),
        replaces=tuple(str(item) for item in raw.get("replaces") or [] if str(item).strip()),
        aliases=tuple(str(item) for item in raw.get("aliases") or [] if str(item).strip()),
        source_path=_optional_string(raw.get("source_path")),
        source_commit=_optional_string(raw.get("source_commit")),
        source_url=_optional_string(raw.get("source_url")),
        reason=str(raw.get("reason") or "User-confirmed authority override.").strip(),
        concept_ids=tuple(str(item) for item in raw.get("concept_ids") or [] if str(item).strip()),
    )


def _error(message: str, field: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "error": True,
        "code": "INVALID_TRUST_CORRECTION_PAYLOAD",
        "schema_version": APPLY_SCHEMA_VERSION,
        "message": message,
        "mutates_authority": False,
    }
    if field:
        result["field"] = field
    return result


def _stable_hash(payload: Mapping[str, Any]) -> str:
    canonical = {k: v for k, v in payload.items() if k != "envelope_hash"}
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _override_item_id(domain_id: str, envelope_hash: str) -> str:
    safe_domain = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in domain_id)
    return f"user_authority_{safe_domain}_{envelope_hash}"


def _title_from_message(message: str) -> str:
    text = " ".join(message.split())
    if len(text) <= 96:
        return text
    return text[:93].rstrip() + "..."


def _optional_string(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
