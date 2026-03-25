"""INGEST-017: Structural fact classification.

Two-layer classifier for concept factuality and query intent.
Layer 1: Concept classification at ingestion time (structural signals).
Layer 2: Query classification at retrieval time (regex patterns).

Both layers are pure Python — zero LLM cost, zero API calls.
"""

import re
import logging

logger = logging.getLogger(__name__)

# ── Layer 1: Concept Classification Signals ──────────────────────────

# Regex patterns for structural signal detection
_HAS_NUMBER = re.compile(r'\b\d+(?:\.\d+)?(?:%|x|px|ms|s|mb|gb|k|m)?\b', re.IGNORECASE)
# PREC-001: Unified entity detection replaces inline regex.
from app.entity_detector import has_specific_entities as _has_named_entity_check
_HAS_METRIC = re.compile(r'\b\d+(?:\.\d+)?%|\b\d+/\d+\b|\bscore[d]?\s*[:=]?\s*\d+', re.IGNORECASE)
_HAS_DATE = re.compile(
    r'\b(?:202[0-9]|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b'
    r'|'
    r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
    re.IGNORECASE,
)
_HAS_VERSION = re.compile(r'\bv\d+(?:\.\d+)*\b|\bversion\s*\d+', re.IGNORECASE)

# Knowledge areas with high factual density
FACT_HEAVY_KAS = frozenset({
    "operations", "infrastructure", "deployment", "personal",
    "team", "company", "product", "pricing",
})

# Concept types that are inherently factual vs abstract
FACTUAL_TYPES = frozenset({"observation", "decision"})
ABSTRACT_TYPES = frozenset({"principle", "heuristic", "method", "cognitive_strategy"})

# Signal weights (from 150-concept prototype)
SIGNAL_WEIGHTS = {
    "has_number": 1.0,  # Lowered from 2.0 — bare numbers too common (A5 backfill review)
    "has_named_entity": 1.5,
    "has_metric": 1.0,
    "has_date": 1.5,
    "fact_heavy_ka": 1.0,
    "has_version": 0.5,
    "factual_type": 0.5,
    "abstract_type": -1.5,
}

# Classification threshold (concepts scoring >= this are FACTUAL)
FACTUAL_THRESHOLD = 2.0


def classify_concept(summary: str, concept_type: str, knowledge_area: str) -> dict:
    """Classify a concept as factual or abstract using structural signals.

    Args:
        summary: Concept summary text.
        concept_type: From the 6-level hierarchy (observation, decision, etc.)
        knowledge_area: Domain/topic area.

    Returns:
        dict with keys: is_factual (bool), factual_score (float),
        temporal_category (str|None), signals_fired (list[str])
    """
    signals_fired = []
    score = 0.0

    if _HAS_NUMBER.search(summary):
        signals_fired.append("has_number")
        score += SIGNAL_WEIGHTS["has_number"]

    if _has_named_entity_check(summary):
        signals_fired.append("has_named_entity")
        score += SIGNAL_WEIGHTS["has_named_entity"]

    if _HAS_METRIC.search(summary):
        signals_fired.append("has_metric")
        score += SIGNAL_WEIGHTS["has_metric"]

    if _HAS_DATE.search(summary):
        signals_fired.append("has_date")
        score += SIGNAL_WEIGHTS["has_date"]

    if knowledge_area and knowledge_area.lower() in FACT_HEAVY_KAS:
        signals_fired.append("fact_heavy_ka")
        score += SIGNAL_WEIGHTS["fact_heavy_ka"]

    if _HAS_VERSION.search(summary):
        signals_fired.append("has_version")
        score += SIGNAL_WEIGHTS["has_version"]

    if concept_type in FACTUAL_TYPES:
        signals_fired.append("factual_type")
        score += SIGNAL_WEIGHTS["factual_type"]

    if concept_type in ABSTRACT_TYPES:
        signals_fired.append("abstract_type")
        score += SIGNAL_WEIGHTS["abstract_type"]

    is_factual = score >= FACTUAL_THRESHOLD
    temporal_category = _infer_temporal_category(summary, signals_fired) if is_factual else None

    return {
        "is_factual": is_factual,
        "factual_score": round(score, 2),
        "temporal_category": temporal_category,
        "signals_fired": signals_fired,
    }


def _infer_temporal_category(summary: str, signals: list[str]) -> str:
    """Infer temporal category from structural signals.

    Returns one of: identity, role, activity, relational, architectural.
    """
    s_lower = summary.lower()

    # Identity: personal pronouns + named entities (name, email, location)
    if re.search(r'\b(?:my name|i am |i\'m |my email|my address|i live)\b', s_lower):
        return "identity"

    # Role: employment/company references
    if re.search(r'\b(?:work at|work for|work as|employer|my job|my role|my title|my company)\b', s_lower):
        return "role"

    # Relational: people references
    if re.search(r'\b(?:my partner|my wife|my husband|my manager|my boss|cofounder|colleague|team member)\b', s_lower):
        return "relational"

    # Activity: metrics, dates, version numbers → fast-changing
    if "has_metric" in signals or "has_date" in signals or "has_version" in signals:
        return "activity"

    # Architectural: system design claims
    if re.search(r'\b(?:architecture|system|infrastructure|deployment|pipeline|schema)\b', s_lower):
        return "architectural"

    # Default: activity (most conservative — fastest decay)
    return "activity"


# ── Layer 2: Query Classification ────────────────────────────────────

# Structural query patterns (replace 18 hardcoded markers)
_QUERY_PATTERNS = {
    "personal_fact": re.compile(
        r'\b(?:what|where|who|when)\b.*\b(?:my|i|me|our)\b'
        r'|'
        r'\b(?:my|i|me|our)\b.*\b(?:what|where|who|when)\b',
        re.IGNORECASE,
    ),
    "recall_request": re.compile(
        r'\b(?:remind me|remember|recall|do you know)\b',
        re.IGNORECASE,
    ),
    "entity_lookup": re.compile(
        r'\b(?:tell me about|what is|who is|where is)\b\s+\w',
        re.IGNORECASE,
    ),
    "temporal_query": re.compile(
        r'\b(?:when did|latest|last time|how long|most recent)\b',
        re.IGNORECASE,
    ),
    "quantitative": re.compile(
        r'\b(?:how many|how much|count|total|score|percentage|rate)\b',
        re.IGNORECASE,
    ),
}

# Anti-factual patterns (queries seeking principles/exploration, not facts)
_ANTI_PATTERNS = {
    "principle_seeking": re.compile(
        r'\b(?:how should|the best|best way|best practice|recommend|approach to)\b',
        re.IGNORECASE,
    ),
    "exploration": re.compile(
        r'\b(?:brainstorm|ideas for|what if we|options for|explore)\b',
        re.IGNORECASE,
    ),
}


def classify_batch(concepts: list[dict]) -> dict:
    """Classify a batch of concepts and emit an INFO-level session summary.

    OPS-067: Provides session-level observability for fact classification.
    Emits once per batch call — use at ingestion batch boundaries.

    Args:
        concepts: List of dicts with keys: summary, concept_type, knowledge_area.

    Returns:
        dict with keys: total, factual, abstract, factual_rate, signal_counts.
    """
    total = len(concepts)
    factual = 0
    signal_counts: dict[str, int] = {}

    for c in concepts:
        result = classify_concept(
            c.get("summary", ""),
            c.get("concept_type", "observation"),
            c.get("knowledge_area", "general"),
        )
        if result["is_factual"]:
            factual += 1
        for sig in result["signals_fired"]:
            signal_counts[sig] = signal_counts.get(sig, 0) + 1

    abstract = total - factual
    factual_rate = round(factual / total, 3) if total else 0.0

    logger.info(
        "OPS-067 classify_batch: total=%d factual=%d abstract=%d rate=%.1f%% top_signals=%s",
        total,
        factual,
        abstract,
        factual_rate * 100,
        sorted(signal_counts.items(), key=lambda x: -x[1])[:3],
    )
    return {
        "total": total,
        "factual": factual,
        "abstract": abstract,
        "factual_rate": factual_rate,
        "signal_counts": signal_counts,
    }


def is_fact_seeking_query(query_text: str) -> bool:
    """Classify whether a query is seeking factual information.

    Replaces the hardcoded _FACT_QUERY_MARKERS from INGEST-015.

    Args:
        query_text: The user's query string.

    Returns:
        True if the query matches a fact-seeking pattern and no anti-pattern.
    """
    # Anti-patterns take priority — if seeking principles/exploration, not fact-seeking
    for name, pattern in _ANTI_PATTERNS.items():
        if pattern.search(query_text):
            logger.debug("INGEST-017: query anti-pattern '%s' matched", name)
            return False

    # Check fact-seeking patterns
    for name, pattern in _QUERY_PATTERNS.items():
        if pattern.search(query_text):
            logger.debug("INGEST-017: query pattern '%s' matched", name)
            return True

    return False
