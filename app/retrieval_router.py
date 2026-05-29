"""RETRIEVAL-060: Adaptive Retrieval Router for conversation_turn S2.

Lightweight query classifier that dynamically selects retrieval strategies
based on query signals. No LLM calls — regex/heuristic classification only.

Strategies are COMPOSABLE — a counting+temporal query gets both broad sweep
AND recency weighting. Each strategy adds to the result set, not replaces it.

Feature-flagged via PITH_ADAPTIVE_RETRIEVAL.

Query Archetypes:
  - counting/aggregation: "How many X?" → broad sweep + entity chain
  - relational chain: "Where does the person who X live?" → multihop decomposition
  - temporal/update: "Where do I currently keep X?" → recency boost
  - entity recall: "You mentioned HAMT" → entity chain keyword precision
  - preference: "What kind of X do I like?" → wide retrieval, diverse
  - standard: no special signal → normal embedding + entity chain supplement
"""

import re
import os
import logging
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("pith.retrieval_router")

# Feature flag
ADAPTIVE_RETRIEVAL_ENABLED = os.environ.get(
    "PITH_ADAPTIVE_RETRIEVAL", ""
).lower() in ("true", "1")

# ---------------------------------------------------------------------------
# Signal detectors — lightweight regex, zero LLM calls
# ---------------------------------------------------------------------------

_SIG_COUNTING = re.compile(
    r'\b(?:how many|how much total|total number|count of|number of|'
    r'how often|how many times|all the .{1,30} I\b|'
    r'list all|what are all|every .{1,20} I\b)\b',
    re.IGNORECASE,
)

_SIG_RELATIONAL = re.compile(
    r'\b(?:'
    # Original patterns — explicit relational indicators
    r'where the|in which|to which|from which|attended by|'
    r'the person who|the one who|the (?:broadcaster|organization) who|'
    r'who .{1,20} the .{1,20} that|'
    # RETRIEVAL-088: Multi-hop chain indicators for MAB-FC style questions.
    # "of the <role/entity-type>" chains — the core MH signal. Questions
    # like "What is the country of citizenship of the performer of..." have
    # nested "of the X" clauses requiring entity chain traversal.
    r'of the (?:country|person|author|creator|founder|spouse|'
    r'employer|performer|institution|organization|sport|city|'
    r'language|continent|capital|birthplace|religion|genre|'
    r'director|manager|developer|officeholder|individual|'
    r'CEO|chief|head|home country|notable work|place of death|'
    r'company|manufacturer|nation|broadcaster|HQ)|'
    # "associated with" chains (sport/position associations)
    r'associated with|'
    # Possessive multi-hop: "X's birthplace/country/etc."
    r"'s (?:birthplace|citizenship|country|spouse|employer|position)|"
    # Passive chains: "played by", "created by", etc.
    r'(?:held|played|created|written|performed|developed|'
    r'established|founded|managed|coached) by|'
    # Relational head phrases: "the birthplace of", "capital city of", etc.
    r'(?:home country|birthplace|capital city|place of (?:birth|death|formation)|'
    r'location of (?:formation|headquarters)|continent of origin|'
    r'language used|country of origin|chief of state|chief executive|'
    r'notable work|broadcaster) of|'
    r'place of origin for the (?:broadcaster|organization)|'
    # "where X holds/was" — location-possessive chains
    r'where \w+ (?:holds?|held|was|is|are|originated)|'
    # Misc relational signals
    r'recognized as the|commonly known as|'
    # "did/is the author/creator/spouse of"
    r'(?:did|does|is|was) the (?:author|creator|founder|spouse|manager|director) of|'
    # Chain reasoning indicators
    r'credited with|responsible for the|'
    r'(?:belong|belongs) to|that originated|country that|'
    r'would you categorize|holds the position|'
    # Language chain patterns
    r'speak,? write|languages? (?:spoken|written|used|did)'
    r')\b',
    re.IGNORECASE,
)

_SIG_TEMPORAL = re.compile(
    r'\b(?:currently|right now|at this point|these days|nowadays|'
    r'recently|lately|as of now|at the moment|'
    r'today|this week|this month|this past|'
    r'most recent(?:ly)?|latest|last time|still|anymore|'
    r"what'?s new|update me|catch me up|"
    r'do I (?:still|now)|have I (?:changed|updated|switched))\b',
    re.IGNORECASE,
)

_SIG_RECALL = re.compile(
    r'\b(?:remind me|you mentioned|we discussed|we talked|you told me|'
    r'you said|you were referring|you recommended|you suggested|'
    r'can you remind|what (?:was|were) .{1,20} you (?:said|mentioned|told))\b',
    re.IGNORECASE,
)

_SIG_PREFERENCE = re.compile(
    r'\b(?:prefer|favorite|favourite|like most|enjoy most|'
    r'what (?:type|kind) of [^?.!]{1,80}\b'
    r'(?:enjoy|enjoys|like|likes|love|loves|prefer|prefers|favorite|favourite)\b|'
    r'suggestions? (?:for|on|about)|recommend(?:ation)?s? for|'
    r'what (?:do|would) I (?:like|prefer|enjoy))\b',
    re.IGNORECASE,
)

_SIG_MOVIE_TYPE_PREFERENCE = re.compile(
    r'\bwhat (?:type|kind) of (?:movies?|films?) [^?.!]{1,80}\b'
    r'(?:enjoy|enjoys|like|likes|love|loves|prefer|prefers|favorite|favourite|watch|watching)\b',
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# RetrievalConfig — the output of classification
# ---------------------------------------------------------------------------

@dataclass
class RetrievalConfig:
    """Dynamic retrieval parameters for a single query."""

    # Strategy toggles
    use_multihop: bool = False
    use_entity_chain: bool = False
    force_entity_chain: bool = False  # override PITH_ENTITY_CHAIN env var

    # Retrieval tuning
    top_k_multiplier: float = 1.0     # applied to effective_max_concepts
    recency_boost: float = 0.0        # [0, 1] weight boost for recent concepts
    entity_chain_budget_ms: int = 150  # time budget for entity chain

    # Diagnostics
    signals: list = field(default_factory=list)
    raw_query: str = ""

    @property
    def is_adaptive(self) -> bool:
        """True if any non-default strategy is active."""
        return (
            self.use_multihop
            or self.use_entity_chain
            or self.top_k_multiplier != 1.0
            or self.recency_boost > 0.0
        )


# ---------------------------------------------------------------------------
# Core classifier
# ---------------------------------------------------------------------------

def classify_query(query: str) -> RetrievalConfig:
    """Classify a query and return optimal retrieval configuration.

    Zero LLM calls. Pure regex/heuristic. <1ms.
    """
    config = RetrievalConfig(raw_query=query)
    signals = []

    # Detect signals
    if _SIG_COUNTING.search(query):
        signals.append("counting")
    if _SIG_RELATIONAL.search(query):
        signals.append("relational")
    if _SIG_TEMPORAL.search(query):
        signals.append("temporal")
    if _SIG_RECALL.search(query):
        signals.append("recall")
    if _SIG_PREFERENCE.search(query):
        signals.append("preference")

    if not signals:
        signals.append("standard")

    config.signals = signals

    # --- Strategy composition ---

    # Counting/aggregation: broad sweep, entity chain for completeness
    if "counting" in signals:
        config.top_k_multiplier = max(config.top_k_multiplier, 1.5)
        config.use_entity_chain = True
        config.force_entity_chain = True
        config.entity_chain_budget_ms = 250  # more time for thoroughness

    # Relational: multihop decomposition + entity chain for depth
    # Entity chain (5ms, 0 LLM calls) provides iterative keyword-hop
    # traversal that catches multi-hop chains multihop's LLM decomposer
    # may miss — especially under rate limits. Always run both.
    if "relational" in signals:
        config.use_multihop = True
        config.use_entity_chain = True
        config.force_entity_chain = True
        # RETRIEVAL-077: Increase budget from 200→500ms for relational queries.
        # Entity chain needs 3-4 hops for multi-hop questions. At 200ms, 35%
        # of chains exhaust budget at hop 2 before reaching bridging facts.
        # Successful full-chain traversals take ~300-500ms on 2k-4k brains.
        config.entity_chain_budget_ms = max(config.entity_chain_budget_ms, 500)

    # Temporal: recency boost
    if "temporal" in signals:
        config.recency_boost = max(config.recency_boost, 0.3)
        config.use_entity_chain = True  # keyword helps find latest state

    # Recall: entity chain precision (keyword match on specific terms)
    if "recall" in signals:
        config.use_entity_chain = True
        config.force_entity_chain = True
        config.entity_chain_budget_ms = max(config.entity_chain_budget_ms, 200)

    # Preference: wider retrieval for pattern diversity
    if "preference" in signals:
        config.top_k_multiplier = max(config.top_k_multiplier, 1.3)
        config.use_entity_chain = True

    # Standard: entity chain as lightweight supplement
    if "standard" in signals and not config.use_entity_chain:
        config.use_entity_chain = True  # always-on supplement

    log.info(
        f"RETRIEVAL-060: classify signals={signals} "
        f"multihop={config.use_multihop} entity_chain={config.use_entity_chain} "
        f"top_k_mult={config.top_k_multiplier} recency={config.recency_boost}"
    )

    return config


def query_bridge_terms(query: str, signals: list[str] | tuple[str, ...]) -> list[str]:
    """Return conservative lexical bridge terms for classified query shapes."""
    if "preference" not in signals:
        return []
    if _SIG_MOVIE_TYPE_PREFERENCE.search(query):
        return ["genre", "genres", "love", "loves"]
    return []


# ---------------------------------------------------------------------------
# Recency boost scorer — used in S2 reranking
# ---------------------------------------------------------------------------

def apply_recency_boost(
    results: list,
    boost: float,
    max_age_days: float = 30.0,
) -> list:
    """Apply recency boost to search results.

    Newer concepts get a score bump proportional to `boost`.
    Results must have .created_at or .timestamp attribute (ISO string or epoch).
    Returns the same list with modified relevance_score.
    """
    if boost <= 0.0 or not results:
        return results

    import datetime
    now = datetime.datetime.now(datetime.timezone.utc)

    for r in results:
        # Try to get a timestamp
        ts = getattr(r, 'created_at', None) or getattr(r, 'timestamp', None)
        if ts is None:
            continue

        if isinstance(ts, str):
            try:
                ts = datetime.datetime.fromisoformat(ts.replace('Z', '+00:00'))
            except (ValueError, TypeError):
                continue
        elif isinstance(ts, (int, float)):
            ts = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)

        if not isinstance(ts, datetime.datetime):
            continue

        # RETRIEVAL-053: Ensure timezone-aware for subtraction with `now` (UTC)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=datetime.timezone.utc)

        age_days = (now - ts).total_seconds() / 86400.0
        # Linear decay: full boost at age=0, zero boost at max_age_days
        recency_factor = max(0.0, 1.0 - (age_days / max_age_days))
        base_score = getattr(r, 'relevance_score', 0.5)
        r.relevance_score = base_score + (boost * recency_factor)

    return results


# ---------------------------------------------------------------------------
# Convenience: get config or default (for non-adaptive mode)
# ---------------------------------------------------------------------------

def get_retrieval_config(query: str) -> Optional[RetrievalConfig]:
    """Return RetrievalConfig if adaptive retrieval is enabled, else None."""
    if not ADAPTIVE_RETRIEVAL_ENABLED:
        return None
    return classify_query(query)
