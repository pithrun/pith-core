"""Manifest-backed authority-chain resolution for Pith continuity.

Authority Chain v1 is intentionally narrow: launch/public-copy artifacts only,
with no DB schema migration. The resolver is deterministic so scorecards can
measure current-vs-historical behavior without an LLM judge.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

DEFAULT_AUTHORITY_CHAIN_MANIFEST = (
    Path(__file__).resolve().parent / "data" / "authority_chains" / "launch_public_copy.json"
)
AUTHORITY_CHAIN_ANSWER_SCHEMA_VERSION = "authority_chain_answer.v1"

CURRENT_INTENT_TERMS = frozenset(
    {
        "approved",
        "authority",
        "authoritative",
        "current",
        "final",
        "govern",
        "governing",
        "governs",
        "latest",
        "landed",
        "now",
        "official",
        "replacement",
        "source-of-truth",
        "source",
        "still",
        "truth",
    }
)
HISTORICAL_INTENT_TERMS = frozenset(
    {
        "earlier",
        "historical",
        "history",
        "old",
        "older",
        "previous",
        "previously",
        "prior",
        "superseded",
    }
)
AUTHORITY_DOMAIN_TERMS = frozenset(
    {
        "announcement",
        "copy",
        "hackernews",
        "hn",
        "landing",
        "launch",
        "messaging",
        "narrative",
        "pith",
        "post",
        "public",
        "release",
        "site",
        "website",
    }
)
CHANNEL_TERMS = frozenset({"hn", "hackernews", "hacker", "news", "x", "twitter", "site", "website", "landing"})
ALIAS_STOPWORD_TERMS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "versus",
        "vs",
        "what",
        "which",
        "with",
    }
)
GENERIC_ALIAS_TERMS = (
    CURRENT_INTENT_TERMS
    | HISTORICAL_INTENT_TERMS
    | ALIAS_STOPWORD_TERMS
    | {
        "announcement",
        "channel",
        "copy",
        "draft",
        "launch",
        "pith",
        "post",
        "public",
        "source",
        "truth",
        "version",
    }
)
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]*")


@dataclass(frozen=True)
class AuthorityArtifact:
    artifact_id: str
    title: str
    state: str
    channel: str
    governs: bool
    superseded_by: str | None
    replaces: tuple[str, ...]
    aliases: tuple[str, ...]
    source_path: str | None
    source_commit: str | None
    source_url: str | None
    reason: str
    concept_ids: tuple[str, ...]

    @property
    def is_current(self) -> bool:
        return self.state == "current" and self.governs

    @property
    def is_historical(self) -> bool:
        return self.state in {"historical", "superseded", "channel_draft"}


@dataclass(frozen=True)
class AuthorityResolution:
    chain_id: str
    intent: str
    query_text: str
    current_authority: AuthorityArtifact | None
    historical_items: tuple[AuthorityArtifact, ...]
    ranked_artifact_ids: tuple[str, ...]
    expanded_terms: tuple[str, ...]
    evidence: dict[str, Any]

    @property
    def has_authority(self) -> bool:
        return self.current_authority is not None or bool(self.historical_items)


@dataclass(frozen=True)
class AuthorityManifest:
    chain_id: str
    current_artifact_id: str
    artifacts: tuple[AuthorityArtifact, ...]

    def by_id(self) -> dict[str, AuthorityArtifact]:
        return {artifact.artifact_id: artifact for artifact in self.artifacts}


def _tokens(text: str | None) -> set[str]:
    lowered = (text or "").casefold()
    boundary_split = re.sub(r"[-/]+", " ", lowered)
    return set(TOKEN_RE.findall(lowered)) | set(TOKEN_RE.findall(boundary_split))


def _string_tuple(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError("expected list")
    return tuple(str(item).strip() for item in raw if str(item).strip())


def _optional_string(raw: Any) -> str | None:
    text = str(raw or "").strip()
    return text or None


def validate_authority_manifest(data: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if data.get("schema_version") != "authority_chain_manifest.v1":
        errors.append("schema_version must be authority_chain_manifest.v1")
    chain_id = str(data.get("chain_id") or "").strip()
    if not chain_id:
        errors.append("chain_id is required")
    current_artifact_id = str(data.get("current_artifact_id") or "").strip()
    if not current_artifact_id:
        errors.append("current_artifact_id is required")
    artifacts = data.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        errors.append("artifacts must be a non-empty list")
        return errors

    seen: set[str] = set()
    for idx, raw in enumerate(artifacts, start=1):
        if not isinstance(raw, dict):
            errors.append(f"artifact {idx}: must be object")
            continue
        artifact_id = str(raw.get("id") or "").strip()
        if not artifact_id:
            errors.append(f"artifact {idx}: id is required")
            continue
        if artifact_id in seen:
            errors.append(f"{artifact_id}: duplicate id")
        seen.add(artifact_id)
        for field in ("title", "state", "channel", "reason"):
            if not str(raw.get(field) or "").strip():
                errors.append(f"{artifact_id}: {field} is required")
        if raw.get("state") not in {"current", "historical", "superseded", "channel_draft", "fallback"}:
            errors.append(f"{artifact_id}: invalid state")
        if not (raw.get("source_path") or raw.get("source_commit") or raw.get("source_url")):
            errors.append(f"{artifact_id}: at least one source_path/source_commit/source_url is required")
    if current_artifact_id and current_artifact_id not in seen:
        errors.append("current_artifact_id must reference an artifact id")
    return errors


def _artifact_from_raw(raw: Mapping[str, Any]) -> AuthorityArtifact:
    return AuthorityArtifact(
        artifact_id=str(raw["id"]).strip(),
        title=str(raw["title"]).strip(),
        state=str(raw["state"]).strip(),
        channel=str(raw["channel"]).strip(),
        governs=bool(raw.get("governs", False)),
        superseded_by=_optional_string(raw.get("superseded_by")),
        replaces=_string_tuple(raw.get("replaces") or []),
        aliases=_string_tuple(raw.get("aliases") or []),
        source_path=_optional_string(raw.get("source_path")),
        source_commit=_optional_string(raw.get("source_commit")),
        source_url=_optional_string(raw.get("source_url")),
        reason=str(raw["reason"]).strip(),
        concept_ids=_string_tuple(raw.get("concept_ids") or []),
    )


@lru_cache(maxsize=16)
def load_authority_manifest(path: str | None = None) -> AuthorityManifest:
    manifest_path = Path(path) if path else DEFAULT_AUTHORITY_CHAIN_MANIFEST
    with manifest_path.open() as fh:
        data = json.load(fh)
    errors = validate_authority_manifest(data)
    if errors:
        raise ValueError("; ".join(errors))
    return AuthorityManifest(
        chain_id=str(data["chain_id"]).strip(),
        current_artifact_id=str(data["current_artifact_id"]).strip(),
        artifacts=tuple(_artifact_from_raw(raw) for raw in data["artifacts"]),
    )


def classify_authority_query(query_text: str, *, manifest: AuthorityManifest | None = None) -> str:
    query_tokens = _tokens(query_text)
    lowered = (query_text or "").casefold()
    scoped_domain_terms = AUTHORITY_DOMAIN_TERMS - {"pith", "post"}
    has_domain = bool(query_tokens & scoped_domain_terms) or "hacker news" in lowered
    has_current = bool(query_tokens & CURRENT_INTENT_TERMS) or "source of truth" in lowered
    has_historical = bool(query_tokens & HISTORICAL_INTENT_TERMS)
    has_channel = bool(query_tokens & CHANNEL_TERMS) or "hacker news" in lowered
    matches = _matching_artifacts(query_text, manifest or load_authority_manifest())
    has_manifest_alias = bool(matches)
    has_replacement_intent = bool(
        query_tokens & {"replacement", "replaced", "supersede", "superseded", "supersedes", "override"}
    )
    has_channel_derivative_match = any(artifact.state == "channel_draft" for artifact in matches)

    if not (has_domain or has_manifest_alias):
        return "no_known_authority"
    if has_channel_derivative_match and _suppresses_channel_current_context(query_text):
        return "historical_lookup"
    if has_historical and (has_current or has_replacement_intent):
        return "mixed_current_historical"
    if has_current and (has_historical or has_channel):
        return "mixed_current_historical"
    if has_current:
        return "current_authority"
    if has_historical or has_channel:
        return "historical_lookup"
    if has_domain or has_manifest_alias:
        return "artifact_lookup"
    return "no_known_authority"


def resolve_authority_chain(
    query_text: str,
    *,
    manifest_path: str | None = None,
    manifest: AuthorityManifest | None = None,
) -> AuthorityResolution:
    loaded = manifest or load_authority_manifest(manifest_path)
    intent = classify_authority_query(query_text, manifest=loaded)
    by_id = loaded.by_id()
    current = by_id.get(loaded.current_artifact_id)
    matches = _matching_artifacts(query_text, loaded)
    historical_matches = tuple(artifact for artifact in matches if artifact.is_historical)

    if intent == "no_known_authority":
        current_for_answer = None
        historical_for_answer: tuple[AuthorityArtifact, ...] = ()
    elif intent == "historical_lookup":
        channel_derivative_matches = _current_channel_derivative_matches(
            historical_matches,
            current=current,
        )
        current_for_answer = (
            current if channel_derivative_matches and not _suppresses_channel_current_context(query_text) else None
        )
        historical_for_answer = historical_matches or tuple(a for a in loaded.artifacts if a.is_historical)[:3]
    elif intent == "artifact_lookup":
        current_for_answer = current
        historical_for_answer = historical_matches
    elif intent == "mixed_current_historical":
        historical_for_answer = historical_matches or tuple(a for a in loaded.artifacts if a.is_historical)[:3]
        suppress_channel_current = _current_channel_derivative_matches(
            historical_for_answer, current=current
        ) and _suppresses_channel_current_context(query_text)
        current_for_answer = None if suppress_channel_current else current
    else:
        current_for_answer = current
        historical_for_answer = historical_matches

    ranked = _rank_artifacts(
        loaded.artifacts,
        intent=intent,
        matches=matches,
        current=current,
        query_text=query_text,
    )
    evidence = build_authority_evidence(
        chain_id=loaded.chain_id,
        intent=intent,
        current=current_for_answer,
        historical=historical_for_answer,
    )
    return AuthorityResolution(
        chain_id=loaded.chain_id,
        intent=intent,
        query_text=query_text,
        current_authority=current_for_answer,
        historical_items=historical_for_answer,
        ranked_artifact_ids=tuple(artifact.artifact_id for artifact in ranked),
        expanded_terms=_expanded_terms_for_resolution(intent, current_for_answer, historical_for_answer),
        evidence=evidence,
    )


def build_authority_evidence(
    *,
    chain_id: str,
    intent: str,
    current: AuthorityArtifact | None,
    historical: Sequence[AuthorityArtifact],
) -> dict[str, Any]:
    return {
        "schema_version": "authority_chain_evidence.v1",
        "chain_id": chain_id,
        "intent": intent,
        "current_authority": _artifact_evidence(current) if current else None,
        "historical_items": [_artifact_evidence(item) for item in historical],
        "reason_current_governs": current.reason if current else None,
        "no_known_authority": current is None and not historical,
    }


def _build_legacy_authority_chain_answer(
    query_text: str,
    *,
    manifest_path: str | None = None,
    manifest: AuthorityManifest | None = None,
) -> tuple[str | None, dict[str, Any]]:
    """Build a deterministic answer for manifest-covered authority queries."""

    resolution = resolve_authority_chain(
        query_text,
        manifest_path=manifest_path,
        manifest=manifest,
    )
    diagnostics = {
        "schema_version": AUTHORITY_CHAIN_ANSWER_SCHEMA_VERSION,
        "mode": "authority_chain_manifest",
        "answer_present": resolution.has_authority,
        "intent": resolution.intent,
        "chain_id": resolution.chain_id,
        "current_artifact_id": (resolution.current_authority.artifact_id if resolution.current_authority else None),
        "historical_artifact_ids": [artifact.artifact_id for artifact in resolution.historical_items],
        "ranked_artifact_ids": list(resolution.ranked_artifact_ids),
        "evidence": resolution.evidence,
    }

    if not resolution.has_authority:
        return None, diagnostics

    answer_parts: list[str] = []
    if resolution.current_authority is not None:
        current = resolution.current_authority
        answer_parts.append(
            "Current authority: "
            f"{_format_authority_artifact(current)}. "
            f"It governs current launch/public-copy decisions. Reason: {current.reason}"
        )

    if resolution.historical_items:
        historical_text = "; ".join(_format_authority_artifact(artifact) for artifact in resolution.historical_items)
        if resolution.current_authority is not None:
            answer_parts.append(
                "Historical/context items: "
                f"{historical_text}. These are historical or channel-specific and do not "
                "govern current launch/public-copy decisions."
            )
        else:
            answer_parts.append(
                "Historical artifact: "
                f"{historical_text}. It is historical or channel-specific and does not "
                "govern current launch/public-copy decisions."
            )

    return " ".join(answer_parts), diagnostics


def build_authority_chain_answer(
    query_text: str,
    *,
    manifest_path: str | None = None,
    manifest: AuthorityManifest | None = None,
) -> tuple[str | None, dict[str, Any]]:
    """Build a deterministic governed answer.

    Explicit manifest calls remain the legacy launch/public-copy authority-chain
    path. Default calls use the generalized trust-governance registry and expose
    legacy diagnostics fields for existing consumers.
    """

    if manifest_path is not None or manifest is not None:
        return _build_legacy_authority_chain_answer(
            query_text,
            manifest_path=manifest_path,
            manifest=manifest,
        )

    try:
        from app.trust_governance import build_trust_governance_answer

        answer, trust_diagnostics = build_trust_governance_answer(query_text)
    except Exception:
        return _build_legacy_authority_chain_answer(query_text)

    domain_id = trust_diagnostics.get("domain_id")
    diagnostics = {
        "schema_version": AUTHORITY_CHAIN_ANSWER_SCHEMA_VERSION,
        "mode": "trust_governance_registry",
        "answer_present": bool(trust_diagnostics.get("answer_present")),
        "intent": trust_diagnostics.get("intent"),
        "chain_id": domain_id,
        "current_artifact_id": trust_diagnostics.get("current_item_id"),
        "historical_artifact_ids": list(trust_diagnostics.get("historical_item_ids") or []),
        "ranked_artifact_ids": list(trust_diagnostics.get("ranked_item_ids") or []),
        "evidence": trust_diagnostics.get("evidence") or {},
        "trust_governance": trust_diagnostics,
    }
    return answer, diagnostics


def authority_metadata_patch(
    *,
    artifact_id: str,
    state: str,
    chain_id: str = "launch_public_copy",
    superseded_by: str | None = None,
    source_path: str | None = None,
    source_commit: str | None = None,
    source_url: str | None = None,
) -> dict[str, Any]:
    """Return the concept data patch closeout should apply for an artifact."""

    return {
        "authority_chain": {
            "schema_version": "authority_chain_metadata.v1",
            "chain_id": chain_id,
            "artifact_id": artifact_id,
            "state": state,
            "is_current_authority": state == "current",
            "superseded_by": superseded_by,
            "source_path": source_path,
            "source_commit": source_commit,
            "source_url": source_url,
        }
    }


def closeout_authority_metadata_patches(
    *,
    new_current_artifact_id: str,
    previous_artifact_ids: Iterable[str],
    chain_id: str = "launch_public_copy",
) -> dict[str, dict[str, Any]]:
    patches = {
        new_current_artifact_id: authority_metadata_patch(
            artifact_id=new_current_artifact_id,
            state="current",
            chain_id=chain_id,
        )
    }
    for artifact_id in previous_artifact_ids:
        if artifact_id == new_current_artifact_id:
            continue
        patches[artifact_id] = authority_metadata_patch(
            artifact_id=artifact_id,
            state="historical",
            chain_id=chain_id,
            superseded_by=new_current_artifact_id,
        )
    return patches


def expand_authority_query_terms(query_text: str) -> tuple[str, ...]:
    resolution = resolve_authority_chain(query_text)
    return resolution.expanded_terms


def order_authority_candidates(
    concept_ids: Sequence[str],
    query_text: str,
    *,
    manifest_path: str | None = None,
) -> list[str]:
    resolution = resolve_authority_chain(query_text, manifest_path=manifest_path)
    if not resolution.has_authority:
        return list(concept_ids)
    manifest = load_authority_manifest(manifest_path)
    artifact_by_concept: dict[str, AuthorityArtifact] = {}
    for artifact in manifest.artifacts:
        for concept_id in artifact.concept_ids:
            artifact_by_concept[concept_id] = artifact
    if not any(concept_id in artifact_by_concept for concept_id in concept_ids):
        return list(concept_ids)
    rank = {artifact_id: idx for idx, artifact_id in enumerate(resolution.ranked_artifact_ids)}

    def sort_key(concept_id: str) -> tuple[int, int, str]:
        artifact = artifact_by_concept.get(concept_id)
        if artifact is None:
            return (1, 9999, concept_id)
        stale_penalty = 0
        if resolution.intent in {"current_authority", "mixed_current_historical"} and artifact.is_historical:
            stale_penalty = 1
        return (stale_penalty, rank.get(artifact.artifact_id, 999), concept_id)

    return sorted(concept_ids, key=sort_key)


def _format_authority_artifact(artifact: AuthorityArtifact) -> str:
    source = _format_authority_source(artifact)
    return (
        f"{artifact.title} ({artifact.artifact_id}; state={artifact.state}; "
        f"channel={artifact.channel}; Source: {source})"
    )


def _format_authority_source(artifact: AuthorityArtifact) -> str:
    source_bits: list[str] = []
    if artifact.source_path:
        source_bits.append(artifact.source_path)
    if artifact.source_commit:
        source_bits.append(f"@ {artifact.source_commit}")
    if artifact.source_url:
        source_bits.append(f"URL: {artifact.source_url}")
    if not source_bits:
        return "source unavailable"
    return " ".join(source_bits)


def _artifact_evidence(artifact: AuthorityArtifact | None) -> dict[str, Any] | None:
    if artifact is None:
        return None
    return {
        "artifact_id": artifact.artifact_id,
        "title": artifact.title,
        "state": artifact.state,
        "channel": artifact.channel,
        "source_path": artifact.source_path,
        "source_commit": artifact.source_commit,
        "source_url": artifact.source_url,
        "reason": artifact.reason,
    }


def _matching_artifacts(query_text: str, manifest: AuthorityManifest) -> tuple[AuthorityArtifact, ...]:
    query_tokens = _tokens(query_text)
    lowered = (query_text or "").casefold()
    matches: list[AuthorityArtifact] = []
    for artifact in manifest.artifacts:
        alias_tokens = set()
        alias_texts = list(artifact.aliases) + [artifact.artifact_id, artifact.title, artifact.channel]
        for alias in alias_texts:
            alias_tokens.update(_tokens(alias))
        specific_overlap = query_tokens & (alias_tokens - GENERIC_ALIAS_TERMS)
        phrase_match = any(
            alias.casefold() in lowered and (_tokens(alias) - GENERIC_ALIAS_TERMS)
            for alias in artifact.aliases
            if len(alias) > 2
        )
        if specific_overlap or phrase_match:
            matches.append(artifact)
    return tuple(matches)


def _current_channel_derivative_matches(
    artifacts: Sequence[AuthorityArtifact],
    *,
    current: AuthorityArtifact | None,
) -> tuple[AuthorityArtifact, ...]:
    if current is None:
        return ()
    return tuple(
        artifact
        for artifact in artifacts
        if artifact.state == "channel_draft" and artifact.superseded_by == current.artifact_id
    )


def _suppresses_channel_current_context(query_text: str) -> bool:
    lowered = (query_text or "").casefold()
    if any(
        phrase in lowered
        for phrase in (
            "not the governing copy",
            "not governing copy",
            "not the source of truth",
            "draft only",
            "only the draft",
        )
    ):
        return True
    query_tokens = _tokens(query_text)
    explicit_current_terms = query_tokens & (CURRENT_INTENT_TERMS - {"source"})
    return "draft" in query_tokens and not explicit_current_terms and "source of truth" not in lowered


def _rank_artifacts(
    artifacts: Sequence[AuthorityArtifact],
    *,
    intent: str,
    matches: Sequence[AuthorityArtifact],
    current: AuthorityArtifact | None,
    query_text: str,
) -> list[AuthorityArtifact]:
    matched_ids = {artifact.artifact_id for artifact in matches}

    def rank_key(artifact: AuthorityArtifact) -> tuple[int, int, str]:
        matched = artifact.artifact_id in matched_ids
        if intent == "historical_lookup":
            channel_derivative_context = not _suppresses_channel_current_context(query_text) and any(
                item.state == "channel_draft" and current and item.superseded_by == current.artifact_id
                for item in matches
            )
            if artifact.is_historical and matched:
                state_rank = 0
            elif channel_derivative_context and artifact.is_current:
                state_rank = 1
            elif artifact.is_historical:
                state_rank = 2
            else:
                state_rank = 3
        elif intent == "mixed_current_historical":
            suppress_channel_current = _suppresses_channel_current_context(query_text) and any(
                item.state == "channel_draft" and current and item.superseded_by == current.artifact_id
                for item in matches
            )
            if suppress_channel_current:
                state_rank = 0 if artifact.is_historical and matched else 1 if artifact.is_historical else 2
            else:
                state_rank = 0 if artifact.is_current else 1 if matched else 2
        elif intent == "current_authority":
            state_rank = 0 if artifact.is_current else 2 if artifact.is_historical else 1
        else:
            state_rank = 0 if matched else 1
        current_rank = 0 if current and artifact.artifact_id == current.artifact_id else 1
        return (state_rank, current_rank, artifact.artifact_id)

    return sorted(artifacts, key=rank_key)


def _expanded_terms_for_resolution(
    intent: str,
    current: AuthorityArtifact | None,
    historical: Sequence[AuthorityArtifact],
) -> tuple[str, ...]:
    terms = ["authority", "current", "historical", "source", "commit"]
    if current:
        terms.extend(_tokens(current.title))
        terms.extend(_tokens(current.artifact_id))
    for artifact in historical[:2]:
        terms.extend(_tokens(artifact.title))
        terms.extend(_tokens(artifact.artifact_id))
    if intent == "no_known_authority":
        return ()
    return tuple(dict.fromkeys(term for term in terms if len(term) > 2))
