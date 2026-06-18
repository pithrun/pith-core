"""Retrieval-only query intent expansion.

This module is intentionally separate from write-time KA taxonomy. It turns
common user-facing aliases into retrieval hints without changing concept
classification semantics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import re
import time
from typing import Iterable


_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class RetrievalAlias:
    aliases: tuple[str, ...]
    canonical_terms: tuple[str, ...]
    target_kas: tuple[str, ...]
    source: str = "registry"


@dataclass(frozen=True)
class QueryIntentExpansion:
    original_query: str
    normalized_query: str
    expanded_query: str
    query_variants: tuple[str, ...]
    expanded_terms: tuple[str, ...]
    inferred_kas: tuple[str, ...]
    matched_aliases: tuple[dict, ...]
    confidence: float
    source: str
    elapsed_ms: float
    input_scope: str
    expansion_input_source: str
    raw_query_hash: str
    assembled_query_hash: str
    deduped_alias_count: int
    contamination_guard_blocked: bool

    def to_trace(self) -> dict:
        payload = asdict(self)
        payload["schema_version"] = "query_intent_expansion.v1"
        return payload


_REGISTRY: tuple[RetrievalAlias, ...] = (
    RetrievalAlias(
        aliases=("gtm", "go to market", "go-to-market", "go_to_market", "launch plan"),
        canonical_terms=("go to market", "launch", "distribution", "positioning"),
        target_kas=("product_strategy", "business_strategy", "competitive_analysis"),
    ),
    RetrievalAlias(
        aliases=("roadmap", "product roadmap", "feature roadmap", "launch roadmap"),
        canonical_terms=("roadmap", "product strategy", "feature prioritization"),
        target_kas=("product_strategy", "project_status"),
    ),
    RetrievalAlias(
        aliases=("pricing", "price point", "monetization", "business model", "revenue model"),
        canonical_terms=("pricing", "monetization", "business model", "revenue"),
        target_kas=("business_strategy", "product_strategy"),
    ),
    RetrievalAlias(
        aliases=("moat", "competitive moat", "defensibility", "competitive landscape"),
        canonical_terms=("competitive advantage", "defensibility", "competitive landscape"),
        target_kas=("competitive_analysis", "business_strategy"),
    ),
    RetrievalAlias(
        aliases=("category", "category design", "narrative", "positioning"),
        canonical_terms=("category design", "positioning", "narrative"),
        target_kas=("business_strategy", "product_strategy"),
    ),
    RetrievalAlias(
        aliases=("rca", "root cause", "root cause analysis", "diagnose", "diagnosis"),
        canonical_terms=("root cause analysis", "debugging", "failure analysis"),
        target_kas=("debugging", "process"),
    ),
    RetrievalAlias(
        aliases=("bench", "benchmark", "benchmarks", "mab", "lme", "longmemeval"),
        canonical_terms=("benchmark", "evaluation", "memory benchmark"),
        target_kas=("pith_benchmarks", "testing"),
    ),
    RetrievalAlias(
        aliases=("architecture", "system design", "technical design"),
        canonical_terms=("architecture", "system design", "technical design"),
        target_kas=("architecture", "design_principles"),
    ),
    RetrievalAlias(
        aliases=("deploy", "deployment", "production", "ops", "operations"),
        canonical_terms=("deployment", "operations", "production readiness"),
        target_kas=("operations", "implementation"),
    ),
    RetrievalAlias(
        aliases=("spec", "specification", "design doc", "implementation plan"),
        canonical_terms=("specification", "implementation planning", "process"),
        target_kas=("specification", "process", "implementation"),
    ),
    RetrievalAlias(
        aliases=("security", "attack surface", "threat model", "vulnerability"),
        canonical_terms=("security", "attack surface", "threat model"),
        target_kas=("security",),
    ),
)


_SKIP_TAXONOMY_SINGLE_ALIASES = {
    "analysis",
    "brand",
    "communication",
    "conversation",
    "ecology",
    "market",
    "personal",
    "protocol",
    "skills",
    "technical",
    "test",
    "monitoring",
    "operations",
    "pricing",
    "testing",
}

_ALLOW_TAXONOMY_SINGLE_ALIASES = {
    "api",
    "automation",
    "benchmarking",
    "classification",
    "coding",
    "coggov",
    "deployment",
    "development",
    "devops",
    "engineering",
    "governance",
    "infra",
    "infrastructure",
    "ingestion",
    "maintenance",
    "milestones",
    "observability",
    "ops",
    "poisonbench",
    "positioning",
    "psis",
    "python",
    "reliability",
    "retrieval",
    "storage",
    "tooling",
    "workflow",
}


def _normalize(text: str) -> str:
    return " ".join(_TOKEN_RE.findall((text or "").casefold()))


def _safe_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def _dedupe(items: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        clean = " ".join(str(item).split()).strip()
        key = clean.casefold()
        if clean and key not in seen:
            seen.add(key)
            out.append(clean)
    return tuple(out)


def _alias_matches(normalized_query: str, alias: str) -> bool:
    normalized_alias = _normalize(alias)
    if not normalized_alias:
        return False
    alias_tokens = normalized_alias.split()
    if len(alias_tokens) == 1:
        return re.search(rf"\b{re.escape(alias_tokens[0])}\b", normalized_query) is not None
    return re.search(rf"\b{re.escape(normalized_alias)}\b", normalized_query) is not None


@lru_cache(maxsize=1)
def _taxonomy_alias_entries() -> tuple[RetrievalAlias, ...]:
    """Read high-signal taxonomy aliases as retrieval hints.

    Single generic aliases are deliberately excluded. The goal is broad recall
    without making words like "market" or "analysis" route every query.
    """
    try:
        from app.core.taxonomy_utils import _load_seed

        _, alias_map = _load_seed()
    except Exception:
        return ()

    entries: list[RetrievalAlias] = []
    for alias, canonical in sorted((alias_map or {}).items()):
        normalized_alias = _normalize(str(alias))
        if not normalized_alias:
            continue
        if " " not in normalized_alias:
            if normalized_alias in _SKIP_TAXONOMY_SINGLE_ALIASES:
                continue
            if normalized_alias not in _ALLOW_TAXONOMY_SINGLE_ALIASES:
                continue
            if len(normalized_alias) < 4:
                continue
        elif len(normalized_alias) < 4:
            continue
        entries.append(
            RetrievalAlias(
                aliases=(str(alias),),
                canonical_terms=(str(alias).replace("_", " "), str(canonical).replace("_", " ")),
                target_kas=(str(canonical),),
                source="taxonomy_alias_map",
            )
        )
    return tuple(entries)


def _registry_entries() -> tuple[RetrievalAlias, ...]:
    return _REGISTRY + _taxonomy_alias_entries()


@lru_cache(maxsize=512)
def expand_query_intent(
    raw_query: str,
    *,
    assembled_query: str | None = None,
    input_scope: str = "query_argument",
    expansion_input_source: str = "query_argument",
) -> QueryIntentExpansion:
    start = time.perf_counter()
    original = raw_query or ""
    assembled = assembled_query if assembled_query is not None else original
    normalized = _normalize(original)

    matched: list[dict] = []
    terms: list[str] = []
    kas: list[str] = []
    sources: set[str] = set()
    matched_alias_keys: set[str] = set()
    matched_semantic_keys: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()

    if normalized:
        for entry in _registry_entries():
            for alias in entry.aliases:
                if _alias_matches(normalized, alias):
                    alias_key = _normalize(alias)
                    semantic_key = (
                        tuple(_normalize(term) for term in entry.canonical_terms),
                        tuple(sorted(_normalize(ka) for ka in entry.target_kas)),
                    )
                    if alias_key in matched_alias_keys or semantic_key in matched_semantic_keys:
                        break
                    matched_alias_keys.add(alias_key)
                    matched_semantic_keys.add(semantic_key)
                    matched.append(
                        {
                            "alias": alias,
                            "canonical_terms": list(entry.canonical_terms),
                            "target_kas": list(entry.target_kas),
                            "source": entry.source,
                        }
                    )
                    terms.extend(entry.canonical_terms)
                    kas.extend(entry.target_kas)
                    sources.add(entry.source)
                    break

    terms_t = _dedupe(terms)
    kas_t = _dedupe(kas)
    expanded_query = assembled
    if terms_t:
        expanded_query = f"{assembled} {' '.join(terms_t)}".strip()

    variants = [original]
    for match in matched[:3]:
        match_terms = _dedupe(match.get("canonical_terms", ()))
        if match_terms:
            variants.append(f"{original} {' '.join(match_terms)}".strip())
    if expanded_query != original:
        variants.append(expanded_query)

    confidence = 0.0
    if matched:
        confidence = min(0.95, 0.45 + 0.15 * len(matched) + 0.05 * len(kas_t))

    elapsed_ms = round((time.perf_counter() - start) * 1000.0, 3)
    contamination_guard_blocked = assembled_query is not None and _normalize(assembled) != normalized
    return QueryIntentExpansion(
        original_query=original,
        normalized_query=normalized,
        expanded_query=expanded_query,
        query_variants=_dedupe(variants),
        expanded_terms=terms_t,
        inferred_kas=kas_t,
        matched_aliases=tuple(matched),
        confidence=round(confidence, 3),
        source="+".join(sorted(sources)) if sources else "none",
        elapsed_ms=elapsed_ms,
        input_scope=input_scope,
        expansion_input_source=expansion_input_source,
        raw_query_hash=_safe_hash(original),
        assembled_query_hash=_safe_hash(assembled),
        deduped_alias_count=len(matched),
        contamination_guard_blocked=contamination_guard_blocked,
    )
