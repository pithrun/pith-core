"""Domain-neutral trust and belief governance resolution.

The registry resolver is deterministic by design. It answers only when a query
matches a configured governance domain with domain-specific evidence; generic
"latest/current/source of truth" phrasing alone is not enough.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

TRUST_GOVERNANCE_DOMAIN_SCHEMA_VERSION = "trust_governance_domain.v1"
TRUST_GOVERNANCE_ANSWER_SCHEMA_VERSION = "trust_governance_answer.v1"
DEFAULT_TRUST_GOVERNANCE_REGISTRY = Path(__file__).resolve().parent / "data" / "trust_governance_domains"

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]*")
VALID_STATES = frozenset({"current", "historical", "superseded", "scoped", "channel_draft", "fallback"})
HISTORICAL_STATES = frozenset({"historical", "superseded", "scoped", "channel_draft"})
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
        "before",
        "earlier",
        "historical",
        "history",
        "old",
        "older",
        "previous",
        "previously",
        "prior",
        "supersede",
        "superseded",
        "was",
    }
)
GENERIC_TERMS = (
    CURRENT_INTENT_TERMS
    | HISTORICAL_INTENT_TERMS
    | {
        "a",
        "an",
        "and",
        "are",
        "artifact",
        "as",
        "at",
        "be",
        "billing",
        "by",
        "claim",
        "copy",
        "decision",
        "do",
        "does",
        "domain",
        "fact",
        "for",
        "from",
        "give",
        "in",
        "is",
        "it",
        "launch",
        "known",
        "me",
        "of",
        "on",
        "or",
        "our",
        "policy",
        "pith",
        "post",
        "pricing",
        "public",
        "research",
        "should",
        "show",
        "the",
        "to",
        "tracking",
        "version",
        "what",
        "which",
        "with",
    }
)


@dataclass(frozen=True)
class TrustGovernanceItem:
    item_id: str
    title: str
    state: str
    scope: str
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
        return self.state in HISTORICAL_STATES


@dataclass(frozen=True)
class TrustGovernanceDomain:
    domain_id: str
    title: str
    current_item_id: str
    scope_terms: tuple[str, ...]
    items: tuple[TrustGovernanceItem, ...]

    def by_id(self) -> dict[str, TrustGovernanceItem]:
        return {item.item_id: item for item in self.items}


@dataclass(frozen=True)
class TrustGovernanceResolution:
    domain_id: str | None
    intent: str
    query_text: str
    current_authority: TrustGovernanceItem | None
    historical_items: tuple[TrustGovernanceItem, ...]
    ranked_item_ids: tuple[str, ...]
    expanded_terms: tuple[str, ...]
    evidence: dict[str, Any]
    matched_domain_ids: tuple[str, ...]
    ambiguity_state: str | None = None

    @property
    def has_authority(self) -> bool:
        return self.current_authority is not None or bool(self.historical_items)


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


def validate_trust_governance_domain(data: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if data.get("schema_version") != TRUST_GOVERNANCE_DOMAIN_SCHEMA_VERSION:
        errors.append("schema_version must be trust_governance_domain.v1")
    domain_id = str(data.get("domain_id") or "").strip()
    if not domain_id:
        errors.append("domain_id is required")
    if not str(data.get("title") or "").strip():
        errors.append("title is required")
    current_item_id = str(data.get("current_item_id") or "").strip()
    if not current_item_id:
        errors.append("current_item_id is required")
    scope_terms = data.get("scope_terms")
    if not isinstance(scope_terms, list) or not any(str(term).strip() for term in scope_terms):
        errors.append("scope_terms must be a non-empty list")
    raw_items = data.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        errors.append("items must be a non-empty list")
        return errors

    seen: set[str] = set()
    for idx, raw in enumerate(raw_items, start=1):
        if not isinstance(raw, dict):
            errors.append(f"item {idx}: must be object")
            continue
        item_id = str(raw.get("id") or "").strip()
        if not item_id:
            errors.append(f"item {idx}: id is required")
            continue
        if item_id in seen:
            errors.append(f"{item_id}: duplicate id")
        seen.add(item_id)
        for field in ("title", "state", "scope", "reason"):
            if not str(raw.get(field) or "").strip():
                errors.append(f"{item_id}: {field} is required")
        if raw.get("state") not in VALID_STATES:
            errors.append(f"{item_id}: invalid state")
        if not (raw.get("source_path") or raw.get("source_commit") or raw.get("source_url")):
            errors.append(f"{item_id}: at least one source_path/source_commit/source_url is required")
    if current_item_id and current_item_id not in seen:
        errors.append("current_item_id must reference an item id")
    return errors


def _item_from_raw(raw: Mapping[str, Any]) -> TrustGovernanceItem:
    return TrustGovernanceItem(
        item_id=str(raw["id"]).strip(),
        title=str(raw["title"]).strip(),
        state=str(raw["state"]).strip(),
        scope=str(raw["scope"]).strip(),
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


def _domain_from_raw(raw: Mapping[str, Any]) -> TrustGovernanceDomain:
    return TrustGovernanceDomain(
        domain_id=str(raw["domain_id"]).strip(),
        title=str(raw["title"]).strip(),
        current_item_id=str(raw["current_item_id"]).strip(),
        scope_terms=_string_tuple(raw.get("scope_terms") or []),
        items=tuple(_item_from_raw(item) for item in raw["items"]),
    )


@lru_cache(maxsize=8)
def _load_base_trust_governance_registry(path: str | None = None) -> tuple[TrustGovernanceDomain, ...]:
    registry_path = Path(path) if path else DEFAULT_TRUST_GOVERNANCE_REGISTRY
    if not registry_path.exists():
        return ()
    domains: list[TrustGovernanceDomain] = []
    for manifest_path in sorted(registry_path.glob("*.json")):
        with manifest_path.open() as fh:
            data = json.load(fh)
        errors = validate_trust_governance_domain(data)
        if errors:
            raise ValueError(f"{manifest_path}: {'; '.join(errors)}")
        domains.append(_domain_from_raw(data))
    return tuple(domains)


def load_trust_governance_registry(
    path: str | None = None,
    *,
    include_profile_overrides: bool = True,
) -> tuple[TrustGovernanceDomain, ...]:
    registry = _load_base_trust_governance_registry(path)
    if path is not None or not include_profile_overrides:
        return registry
    try:
        from app.trust_governance_correction_control import load_authority_overrides, merge_authority_overrides

        return merge_authority_overrides(registry, load_authority_overrides())
    except Exception:
        return registry


def clear_trust_governance_registry_cache() -> None:
    _load_base_trust_governance_registry.cache_clear()


def classify_trust_governance_query(
    query_text: str,
    *,
    registry_path: str | None = None,
    include_profile_overrides: bool = True,
) -> str:
    selection = _select_domain(
        query_text,
        load_trust_governance_registry(registry_path, include_profile_overrides=include_profile_overrides),
    )
    if selection["status"] == "no_known":
        return "no_known_authority"
    if selection["status"] == "ambiguous":
        return "ambiguous_domain"
    return _classify_intent(query_text, selection["matches"])


def is_trust_governance_intent_query(query_text: str) -> bool:
    query_tokens = _tokens(query_text)
    lowered = (query_text or "").casefold()
    return bool(query_tokens & (CURRENT_INTENT_TERMS | HISTORICAL_INTENT_TERMS)) or "source of truth" in lowered


def resolve_trust_governance(
    query_text: str,
    *,
    registry_path: str | None = None,
    include_profile_overrides: bool = True,
) -> TrustGovernanceResolution:
    registry = load_trust_governance_registry(registry_path, include_profile_overrides=include_profile_overrides)
    selection = _select_domain(query_text, registry)
    if selection["status"] != "selected":
        intent = "ambiguous_domain" if selection["status"] == "ambiguous" else "no_known_authority"
        matched_ids = tuple(domain.domain_id for domain in selection.get("matched_domains", ()))
        evidence = _build_evidence(
            domain_id=None,
            intent=intent,
            current=None,
            historical=(),
            matched_domain_ids=matched_ids,
            ambiguity_state=selection["status"],
        )
        return TrustGovernanceResolution(
            domain_id=None,
            intent=intent,
            query_text=query_text,
            current_authority=None,
            historical_items=(),
            ranked_item_ids=(),
            expanded_terms=(),
            evidence=evidence,
            matched_domain_ids=matched_ids,
            ambiguity_state=selection["status"],
        )

    domain = selection["domain"]
    matches = tuple(selection["matches"])
    intent = _classify_intent(query_text, matches)
    by_id = domain.by_id()
    current = by_id.get(domain.current_item_id)
    historical_matches = tuple(item for item in matches if item.is_historical)

    if intent == "no_known_authority":
        current_for_answer = None
        historical_for_answer: tuple[TrustGovernanceItem, ...] = ()
    elif intent == "historical_lookup":
        current_for_answer = _current_for_historical_query(query_text, current, historical_matches)
        historical_for_answer = historical_matches or tuple(item for item in domain.items if item.is_historical)[:3]
    elif intent == "mixed_current_historical":
        current_for_answer = current
        historical_for_answer = historical_matches or tuple(item for item in domain.items if item.is_historical)[:3]
    elif intent == "artifact_lookup":
        current_for_answer = current if not historical_matches else None
        historical_for_answer = historical_matches
    else:
        current_for_answer = current
        historical_for_answer = historical_matches

    ranked = _rank_items(domain.items, intent=intent, matches=matches, current=current)
    matched_domain_ids = (domain.domain_id,)
    evidence = _build_evidence(
        domain_id=domain.domain_id,
        intent=intent,
        current=current_for_answer,
        historical=historical_for_answer,
        matched_domain_ids=matched_domain_ids,
        ambiguity_state=None,
    )
    return TrustGovernanceResolution(
        domain_id=domain.domain_id,
        intent=intent,
        query_text=query_text,
        current_authority=current_for_answer,
        historical_items=historical_for_answer,
        ranked_item_ids=tuple(item.item_id for item in ranked),
        expanded_terms=_expanded_terms(intent, current_for_answer, historical_for_answer, domain),
        evidence=evidence,
        matched_domain_ids=matched_domain_ids,
        ambiguity_state=None,
    )


def build_trust_governance_answer(
    query_text: str,
    *,
    registry_path: str | None = None,
    include_profile_overrides: bool = True,
) -> tuple[str | None, dict[str, Any]]:
    resolution = resolve_trust_governance(
        query_text,
        registry_path=registry_path,
        include_profile_overrides=include_profile_overrides,
    )
    diagnostics = {
        "schema_version": TRUST_GOVERNANCE_ANSWER_SCHEMA_VERSION,
        "mode": "trust_governance_registry",
        "answer_present": resolution.has_authority,
        "intent": resolution.intent,
        "domain_id": resolution.domain_id,
        "current_item_id": resolution.current_authority.item_id if resolution.current_authority else None,
        "historical_item_ids": [item.item_id for item in resolution.historical_items],
        "ranked_item_ids": list(resolution.ranked_item_ids),
        "matched_domain_ids": list(resolution.matched_domain_ids),
        "ambiguity_state": resolution.ambiguity_state,
        "evidence": resolution.evidence,
    }
    if not resolution.has_authority:
        return None, diagnostics

    answer_parts: list[str] = []
    domain_title = _domain_title(resolution.domain_id)
    if resolution.current_authority is not None:
        current = resolution.current_authority
        answer_parts.append(
            "Current authority: "
            f"{_format_item(current)}. "
            f"It governs current {domain_title} decisions. Reason: {current.reason}"
        )
    if resolution.historical_items:
        historical_text = "; ".join(_format_item(item) for item in resolution.historical_items)
        if resolution.current_authority is not None:
            answer_parts.append(
                "Historical/context items: "
                f"{historical_text}. These are historical, superseded, or scoped and do not "
                f"govern current {domain_title} decisions."
            )
        else:
            answer_parts.append(
                "Historical artifact: "
                f"{historical_text}. It is historical, superseded, or scoped and does not "
                f"govern current {domain_title} decisions."
            )
    return " ".join(answer_parts), diagnostics


def _select_domain(query_text: str, registry: Sequence[TrustGovernanceDomain]) -> dict[str, Any]:
    query_tokens = _tokens(query_text)
    lowered = (query_text or "").casefold()
    boundary_normalized = re.sub(r"[-/]+", " ", lowered)
    scored: list[tuple[int, int, TrustGovernanceDomain, tuple[TrustGovernanceItem, ...]]] = []
    for domain in registry:
        domain_terms: set[str] = set()
        for term in domain.scope_terms:
            domain_terms.update(_tokens(term))
        domain_specific_overlap = query_tokens & (domain_terms - GENERIC_TERMS)
        matches = _matching_items(query_text, domain)
        item_specific_overlap = _specific_item_overlap(query_tokens, matches)
        phrase_score = 0
        for term in domain.scope_terms:
            term_text = term.casefold()
            if len(term_text) > 3 and (term_text in lowered or term_text in boundary_normalized):
                phrase_score += 1
        score = len(domain_specific_overlap) + (2 * len(item_specific_overlap)) + phrase_score
        if score > 0:
            scored.append((score, len(item_specific_overlap), domain, matches))

    if not scored:
        return {"status": "no_known", "matched_domains": ()}
    scored.sort(key=lambda row: (-row[0], -row[1], row[2].domain_id))
    top_score = scored[0][0]
    top = [row for row in scored if row[0] == top_score]
    if len(top) > 1:
        return {"status": "ambiguous", "matched_domains": tuple(row[2] for row in top)}
    _, _, domain, matches = scored[0]
    return {"status": "selected", "domain": domain, "matches": matches, "matched_domains": (domain,)}


def _matching_items(query_text: str, domain: TrustGovernanceDomain) -> tuple[TrustGovernanceItem, ...]:
    query_tokens = _tokens(query_text)
    lowered = (query_text or "").casefold()
    matches: list[TrustGovernanceItem] = []
    for item in domain.items:
        alias_tokens: set[str] = set()
        alias_texts = list(item.aliases) + [item.item_id, item.title, item.scope]
        for alias in alias_texts:
            alias_tokens.update(_tokens(alias))
        specific_overlap = query_tokens & (alias_tokens - GENERIC_TERMS)
        phrase_match = any(
            alias.casefold() in lowered and (_tokens(alias) - GENERIC_TERMS)
            for alias in item.aliases
            if len(alias) > 2
        )
        if specific_overlap or phrase_match:
            matches.append(item)
    return tuple(matches)


def _specific_item_overlap(
    query_tokens: set[str],
    matches: Sequence[TrustGovernanceItem],
) -> set[str]:
    overlap: set[str] = set()
    for item in matches:
        alias_tokens: set[str] = set()
        for alias in list(item.aliases) + [item.item_id, item.title, item.scope]:
            alias_tokens.update(_tokens(alias))
        overlap.update(query_tokens & (alias_tokens - GENERIC_TERMS))
    return overlap


def _classify_intent(query_text: str, matches: Sequence[TrustGovernanceItem]) -> str:
    query_tokens = _tokens(query_text)
    lowered = (query_text or "").casefold()
    has_current = bool(query_tokens & CURRENT_INTENT_TERMS) or "source of truth" in lowered
    has_historical = bool(query_tokens & HISTORICAL_INTENT_TERMS)
    has_historical_match = any(item.is_historical for item in matches)
    has_current_match = any(item.is_current for item in matches)
    if has_current and (has_historical or has_historical_match):
        return "mixed_current_historical" if not has_current_match or has_historical else "current_authority"
    if has_current:
        return "current_authority"
    if has_historical and has_current_match:
        return "mixed_current_historical"
    if has_current_match and has_historical_match:
        return "mixed_current_historical"
    if has_current_match:
        return "artifact_lookup"
    if has_historical or has_historical_match:
        return "historical_lookup"
    if matches:
        return "artifact_lookup"
    return "no_known_authority"


def _current_for_historical_query(
    query_text: str,
    current: TrustGovernanceItem | None,
    historical_matches: Sequence[TrustGovernanceItem],
) -> TrustGovernanceItem | None:
    if current is None:
        return None
    query_tokens = _tokens(query_text)
    lowered = (query_text or "").casefold()
    asks_currentness = bool(query_tokens & CURRENT_INTENT_TERMS) or "still current" in lowered or "governs now" in lowered
    if asks_currentness:
        return current
    if any(item.state in {"scoped", "channel_draft"} for item in historical_matches):
        return current
    return None


def _rank_items(
    items: Sequence[TrustGovernanceItem],
    *,
    intent: str,
    matches: Sequence[TrustGovernanceItem],
    current: TrustGovernanceItem | None,
) -> list[TrustGovernanceItem]:
    matched_ids = {item.item_id for item in matches}

    def rank_key(item: TrustGovernanceItem) -> tuple[int, int, str]:
        matched = item.item_id in matched_ids
        if intent in {"current_authority", "mixed_current_historical"}:
            state_rank = 0 if item.is_current else 1 if matched else 2
        elif intent == "historical_lookup":
            scoped_context = any(match.state in {"channel_draft", "scoped"} for match in matches)
            if item.is_historical and matched:
                state_rank = 0
            elif scoped_context and item.is_current:
                state_rank = 1
            elif item.is_historical:
                state_rank = 2
            else:
                state_rank = 3
        elif intent == "artifact_lookup":
            state_rank = 0 if matched else 1
        else:
            state_rank = 0 if item.is_current else 1
        current_rank = 0 if current and item.item_id == current.item_id else 1
        return (state_rank, current_rank, item.item_id)

    return sorted(items, key=rank_key)


def _build_evidence(
    *,
    domain_id: str | None,
    intent: str,
    current: TrustGovernanceItem | None,
    historical: Sequence[TrustGovernanceItem],
    matched_domain_ids: Sequence[str],
    ambiguity_state: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": "trust_governance_evidence.v1",
        "domain_id": domain_id,
        "intent": intent,
        "current_authority": _item_evidence(current) if current else None,
        "historical_items": [_item_evidence(item) for item in historical],
        "reason_current_governs": current.reason if current else None,
        "no_known_authority": current is None and not historical,
        "matched_domain_ids": list(matched_domain_ids),
        "ambiguity_state": ambiguity_state,
    }


def _item_evidence(item: TrustGovernanceItem | None) -> dict[str, Any] | None:
    if item is None:
        return None
    return {
        "item_id": item.item_id,
        "title": item.title,
        "state": item.state,
        "scope": item.scope,
        "source_path": item.source_path,
        "source_commit": item.source_commit,
        "source_url": item.source_url,
        "reason": item.reason,
    }


def _expanded_terms(
    intent: str,
    current: TrustGovernanceItem | None,
    historical: Sequence[TrustGovernanceItem],
    domain: TrustGovernanceDomain,
) -> tuple[str, ...]:
    if intent in {"no_known_authority", "ambiguous_domain"}:
        return ()
    terms = ["authority", "current", "historical", "source", "commit"]
    terms.extend(_tokens(domain.title))
    for term in domain.scope_terms[:8]:
        terms.extend(_tokens(term))
    if current:
        terms.extend(_tokens(current.title))
        terms.extend(_tokens(current.item_id))
    for item in historical[:2]:
        terms.extend(_tokens(item.title))
        terms.extend(_tokens(item.item_id))
    return tuple(dict.fromkeys(term for term in terms if len(term) > 2))


def _format_item(item: TrustGovernanceItem) -> str:
    source = _format_source(item)
    return f"{item.title} ({item.item_id}; state={item.state}; scope={item.scope}; Source: {source})"


def _format_source(item: TrustGovernanceItem) -> str:
    source_bits: list[str] = []
    if item.source_path:
        source_bits.append(item.source_path)
    if item.source_commit:
        source_bits.append(f"@ {item.source_commit}")
    if item.source_url:
        source_bits.append(f"URL: {item.source_url}")
    return " ".join(source_bits) if source_bits else "source unavailable"


def _domain_title(domain_id: str | None) -> str:
    if not domain_id:
        return "trust-governance"
    if domain_id == "launch_public_copy":
        return "launch/public-copy"
    return domain_id.replace("_", " ")
