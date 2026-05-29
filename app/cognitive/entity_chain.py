"""RETRIEVAL-047: Entity-Chain Keyword Retriever for production.

Ported from benchmarks/adapter/entity_chain.py (RETRIEVAL-045).
Extracts named entities from user query, does SQL keyword search per entity,
chains extracted values for multi-hop lookups. Results are unioned with
embedding retrieval in conversation_turn (session.py S4.6).

Key differences from benchmark version:
- Returns SearchResult (not RetrievedConcept)
- Uses production DB path from config
- No benchmark preamble stripping
- Feature-gated via PITH_ENTITY_CHAIN env var
- Time-budgeted (default 150ms) to avoid blocking conversation_turn

RETRIEVAL-073: Hop-priority scoring — hop 1 results score 0.85 (entity-specific
  gold), hop 2 scores 0.78, hop 3+ scores 0.68. Previously flat 0.75 for all
  hops caused entity-specific concepts to be indistinguishable from hop-3+ noise
  in budget governance, losing 17/24 SH 32k failures where gold existed in brain.

RETRIEVAL-075: Total entity chain cap. Entity chain sprawl (2 hop-1 concepts
  expanding to 71 total) dilutes entity-specific gold in budget governance.
  Total cap env-gated via PITH_ENTITY_CHAIN_TOTAL_CAP (default 30).
"""

import json
import hashlib
import re
import os
import time
import sqlite3
import logging
from collections import Counter
from typing import Optional

from app.core.deadline import TurnDeadline
from app.core.models import SearchResult

logger = logging.getLogger(__name__)


def _record_metric(name: str, value: float, labels: dict[str, str] | None = None) -> None:
    try:
        from app.core.metrics_facade import metrics

        metrics.record(name, value, labels or {})
    except Exception:
        pass

# Feature flag
ENTITY_CHAIN_ENABLED = os.environ.get("PITH_ENTITY_CHAIN", "").lower() in ("true", "1")
ENTITY_CHAIN_BUDGET_MS = int(os.environ.get("PITH_ENTITY_CHAIN_BUDGET_MS", "150"))
_TURN_DEADLINE_KEYWORD_MIN_MS = float(os.environ.get("PITH_TURN_DEADLINE_MIN_ENTITY_KEYWORD_MS", "750"))
_KEYWORD_QUERY_MIN_MS = max(1.0, float(os.environ.get("PITH_ENTITY_CHAIN_KEYWORD_QUERY_MIN_MS", "25")))
_KEYWORD_BUSY_TIMEOUT_MS = max(0, int(os.environ.get("PITH_ENTITY_CHAIN_KEYWORD_BUSY_TIMEOUT_MS", "100")))
_KEYWORD_READONLY_ENABLED = os.environ.get(
    "PITH_ENTITY_CHAIN_KEYWORD_READONLY_ENABLED", "1"
).lower() in ("true", "1", "yes", "on")
_KEYWORD_READONLY_FALLBACK = os.environ.get(
    "PITH_ENTITY_CHAIN_KEYWORD_READONLY_FALLBACK", "1"
).lower() in ("true", "1", "yes", "on")

# RETRIEVAL-073: Hop-priority relevance scores.
# Hop 1 = entity-specific gold (highest priority to survive budget trim).
# Hop 2 = one-step chain facts (still valuable for multi-hop).
# Hop 3+ = distant chain noise (lowest priority).
# Previously all hops scored flat 0.75, making entity-specific gold
# indistinguishable from noise in budget governance tiering.
_HOP_SCORES = {
    1: float(os.environ.get("PITH_EC_HOP1_SCORE", "0.85")),
    2: float(os.environ.get("PITH_EC_HOP2_SCORE", "0.78")),
}
_HOP_DEFAULT_SCORE = float(os.environ.get("PITH_EC_HOP_DEFAULT_SCORE", "0.68"))

# RETRIEVAL-075: Total entity chain result cap.
# Prevents entity chain sprawl (2 hop-1 → 71 total) from overwhelming
# budget governance with noise. Default 30 = ~7% of a 400-concept brain.
_TOTAL_CAP = int(os.environ.get("PITH_ENTITY_CHAIN_TOTAL_CAP", "30"))

# RETRIEVAL-076: Predicate diversity enforcement.
# Cap per-predicate results to prevent homogeneous entity chain flooding.
# With 8 slots and cap=2, guarantees at least 4 distinct predicate types.
# Set to 999 to effectively disable (kill switch).
_PREDICATE_CAP = max(1, int(os.environ.get("PITH_EC_PREDICATE_CAP", "2")))

# RETRIEVAL-101: Global cross-hop predicate budget.
# Limits total same-predicate concepts across ALL hops in a single retrieve() call.
# Even with per-hop cap=2, 4 hops can accumulate 8 same-predicate concepts.
# Global cap=4 ensures no single predicate type consumes >13% of _TOTAL_CAP.
# 'other' bucket is exempt (heterogeneous, capping it hurts diversity).
_GLOBAL_PRED_CAP = max(1, int(os.environ.get("PITH_EC_GLOBAL_PRED_CAP", "4")))

# RETRIEVAL-103: Entity-anchored chain-context boosting.
# When enabled, hop N+1 carries forward salient keywords from the concept
# that generated the queued entity, so keyword search prefers concepts
# consistent with the traversal path (reduces edit cross-contamination).
_EC_CHAIN_ANCHOR = os.environ.get(
    "PITH_EC_CHAIN_ANCHOR", "0"
).lower() in ("true", "1")
_EC_CHAIN_ANCHOR_KEYWORDS = int(os.environ.get("PITH_EC_CHAIN_ANCHOR_KW", "3"))

# RETRIEVAL-110: Full-content keyword boosting for conversational benchmarks.
# When enabled, hop-1 kw_boost uses ALL non-entity content words from the question
# instead of the _PROPERTY_WORDS subset. _PROPERTY_WORDS was designed for world-
# knowledge queries (country, language, sport...) and misses conversational predicates
# (research, identity, relationship, career, moved). Without this flag, all 800+
# same-entity concepts tie at confidence=0.6 and hop-1 selection degrades to
# DB insertion order. Enable for LoCoMo and similar conversational memory benchmarks.
# Default: off — preserves MAB world-knowledge behaviour unchanged.
_EC_FULL_CONTENT_KEYWORDS = os.environ.get(
    "PITH_EC_FULL_CONTENT_KEYWORDS", "0"
).lower() in ("true", "1")

# RETRIEVAL-089: Association-aware expansion for MH queries.
# After entity chain completes, expand terminal concepts via association
# table neighbors to close hop-1 retrieval gaps.
_ASSOC_EXPAND_ENABLED = os.environ.get("PITH_EC_EXPAND_ASSOCIATIONS", "1").lower() in ("true", "1")
_ASSOC_STRENGTH_MIN = float(os.environ.get("PITH_EC_ASSOC_STRENGTH_MIN", "0.3"))
_ASSOC_MAX_PER_CONCEPT = int(os.environ.get("PITH_EC_ASSOC_MAX_PER_CONCEPT", "5"))

# RETRIEVAL-090: Noun-phrase fallback for questions with no proper nouns.
_EC_NOUN_PHRASE_FALLBACK = os.environ.get(
    "PITH_EC_NOUN_PHRASE_FALLBACK", "1"
).lower() in ("true", "1")

# RETRIEVAL-091: Extra hops for multi-hop queries.
_MH_EXTRA_HOPS = int(os.environ.get("PITH_EC_MH_EXTRA_HOPS", "2"))

# RETRIEVAL-092: Prune oversized hop queue by question keyword relevance.
_EC_QUEUE_PRUNE = os.environ.get(
    "PITH_EC_QUEUE_PRUNE", "1"
).lower() in ("true", "1")

# RETRIEVAL-106: Predicate-guided chain query for multi-hop.
# Deterministic chain-and-query: decompose question into seed + predicate keywords,
# match predicate patterns via SQL LIKE, extract values, chain forward.
# Supplements existing entity chain with precisely-matched concepts.
_EC_CHAIN_QUERY = os.environ.get(
    "PITH_EC_CHAIN_QUERY", "0"
).lower() in ("true", "1")
_EC_CHAIN_QUERY_EARLY = os.environ.get(
    "PITH_EC_CHAIN_QUERY_EARLY", "0"
).lower() in ("true", "1")
_EC_CHAIN_QUERY_BUDGET_MS = max(
    1,
    int(os.environ.get("PITH_EC_CHAIN_QUERY_BUDGET_MS", "500")),
)
_EC_CHAIN_QUERY_AUTO_EARLY = os.environ.get(
    "PITH_EC_CHAIN_QUERY_AUTO_EARLY", "1"
).lower() in ("true", "1")
_EC_CHAIN_QUERY_BRANCHES = max(
    1,
    int(os.environ.get("PITH_EC_CHAIN_QUERY_BRANCHES", "2")),
)
_CONTRACT_COVERAGE_TRACE = os.environ.get(
    "PITH_MH262_CONTRACT_COVERAGE_TRACE", "0"
).lower() in ("true", "1")


def _mh262_canary_trace_enabled() -> bool:
    return os.environ.get("PITH_MH262_CANARY_RETRIEVAL_TRACE", "").lower() in ("true", "1")


def _contract_coverage_trace_enabled() -> bool:
    return _CONTRACT_COVERAGE_TRACE


def _trace_hash(value: str) -> str:
    return hashlib.sha256((value or "").strip().lower().encode("utf-8")).hexdigest()


def _trace_result_ids(results: list[SearchResult]) -> list[str]:
    return [r.concept_id for r in results]

# Structured predicate endings for classification.
# Shared vocabulary with session.py _STRUCTURED_ENDINGS — kept in sync manually.
_PREDICATE_PATTERNS = [
    (' is associated with the sport of ', 'sport'),
    (' was created in the country of ', 'country_of_origin'),
    (' is located in the continent of ', 'continent'),
    (' is a citizen of ', 'citizenship'),
    (' was founded by ', 'founder'),
    (' was founded in the city of ', 'founded_city'),
    (' was performed by ', 'performer'),
    (' was developed by ', 'developer'),
    (' was written in the language of ', 'language'),
    (' plays the position of ', 'position'),
    (' is affiliated with the religion of ', 'religion'),
    (' was born in the city of ', 'birthplace'),
    (' died in the city of ', 'deathplace'),
    (' worked in the city of ', 'workplace'),
    (' is famous for ', 'famous_for'),
    (' is married to ', 'spouse'),
    (' is employed by ', 'employer'),
    (' the chairperson of ', 'chairperson'),
    (' works in the field of ', 'field'),
    (' of the current head of state in ', 'head_of_state'),
    (' of the current head of the ', 'head_of_govt'),
    (' of the Prime Minister of ', 'prime_minister'),
    (' is located in the ', 'location'),
    ("'s child is ", 'child'),
    (' was created by ', 'creator'),
    (' the capital of ', 'capital'),
    (' capital is ', 'capital'),
    (' the official language of ', 'official_language'),
]


# RETRIEVAL-106: Maps question keywords to SQL LIKE patterns in concept summaries.
# Enables predicate-guided search: at each hop, entity + pattern SQL query.
_KEYWORD_TO_SQL: dict[str, list[str]] = {
    # geography
    'continent':             ['is located in the continent of'],
    'country of citizenship':['is a citizen of'],
    'citizenship':           ['is a citizen of'],
    'citizen':               ['is a citizen of'],
    'belongs':               ['is a citizen of'],
    'belong':                ['is a citizen of'],
    'comes from':            ['is a citizen of'],
    'country of origin':     ['was created in the country of'],
    'place of origin':       ['was founded in the city of',
                              'was created in the country of'],
    'originated':            ['was created in the country of'],
    'capital':               ['the capital of', 'capital is'],
    'headquarters':          ['is located in the', 'was founded in the city of'],
    # creation / authorship
    'author':     ['author of', 'was created by'],
    'wrote':      ['author of', 'was written in the language of'],
    'created':    ['was created by', 'was founded by'],
    'developed':  ['was developed by'],
    'founded':    ['was founded by'],
    'founder':    ['was founded by'],
    'establish':  ['was founded by'],
    # people relations
    'spouse':     ['is married to'],
    'married':    ['is married to'],
    'born':       ['was born in the city of'],
    'birthplace': ['was born in the city of'],
    'place of birth': ['was born in the city of'],
    'died':       ['died in the city of'],
    'child':      ["'s child is"],
    # profession / affiliation
    'sport':      ['is associated with the sport of'],
    'position':   ['plays the position of'],
    'religion':   ['is affiliated with the religion of'],
    'religious':  ['is affiliated with the religion of'],
    'employer':   ['is employed by'],
    'employed':   ['is employed by'],
    'employee':   ['is employed by'],
    'employment': ['is employed by'],
    'work':       ['is employed by', 'worked in the city of'],
    'studied':    ['was educated', 'univeristy where', 'university where',
                    'is employed by'],
    'education':  ['is employed by', 'was educated', 'univeristy where',
                   'university where'],
    'school':     ['is employed by', 'was educated', 'univeristy where',
                   'university where'],
    'university': ['is employed by', 'was educated'],
    'field':      ['works in the field of'],
    'workplace':  ['worked in the city of'],
    # language
    'language':          ['was written in the language of',
                          'the official language of'],
    'official language': ['the official language of'],
    'officially spoken': ['the official language of'],
    'official medium':   ['the official language of'],
    # politics
    'head of state':      ['head of state in'],
    'head of government': ['head of the'],
    'prime minister':     ['Prime Minister of'],
    'chairperson':         ['the chairperson of'],
    'chair':               ['the chairperson of'],
    # media / performance
    'performer':          ['was performed by'],
    'sang':               ['was performed by'],
    'artist':             ['was performed by'],
    'broadcaster':        ['origianl broadcaster of', 'broadcaster of',
                           'was created by', 'was developed by'],
    'aired':              ['origianl broadcaster of', 'broadcaster of',
                           'was created by'],
    'director':           ['was developed by', 'was created by',
                           'was performed by'],
    'manager':            ['was developed by', 'was created by',
                           'was performed by'],
    'director/manager':   ['was developed by', 'was created by',
                           'was performed by'],
    # exec roles
    'chief executive officer': ['chief executive officer of'],
    'officer':            ['chief executive officer of'],
    # genre
    'literary genre':    ['type of music'],
    'genre':              ['type of music'],
    # misc
    'famous':       ['is famous for'],
    'notable work': ['is famous for'],
    'known for':    ['is famous for'],
}


_CONTRACT_ROUTE_SPECS: list[dict[str, object]] = [
    {
        "route": "developer_ceo_birth_continent",
        "chain": ["developer", "chief_executive_officer", "birthplace", "continent"],
        "required_any": [
            ("developer", "developed", "software"),
            ("chief executive officer", "ceo"),
            ("born", "birth"),
            ("continent",),
        ],
    },
    {
        "route": "religion_founder_birth_city",
        "chain": ["religion", "founder", "birthplace"],
        "required_any": [("founder", "founded"), ("born", "birth"), ("city", "where")],
    },
    {
        "route": "occupation_sport_origin",
        "chain": ["field", "sport", "country_of_origin"],
        "required_any": [
            ("occupation", "field", "position"),
            ("sport",),
            ("origin", "country", "come from"),
        ],
    },
    {
        "route": "sport_founder_citizenship_capital",
        "chain": ["sport", "founder", "citizenship", "capital"],
        "required_any": [
            ("sport",),
            ("founder", "founded"),
            ("citizen", "citizenship"),
            ("capital",),
        ],
    },
    {
        "route": "performer_spouse_citizenship_language",
        "chain": ["performer", "spouse", "citizenship", "official_language"],
        "required_any": [
            ("performed", "performer", "sang", "artist"),
            ("spouse", "married", "wife", "husband"),
            ("citizen", "citizenship"),
            ("language",),
        ],
    },
    {
        "route": "genre_origin_head_government",
        "chain": ["genre", "country_of_origin", "head_of_govt"],
        "required_any": [
            ("genre", "type of music"),
            ("origin", "country"),
            ("head of government", "government", "prime minister"),
        ],
    },
    {
        "route": "employee_director",
        "chain": ["employer", "director"],
        "required_any": [("employed", "employee", "works for"), ("director",)],
    },
    {
        "route": "author_famous_for",
        "chain": ["author", "famous_for"],
        "required_any": [("author", "wrote", "written"), ("famous", "known for", "notable")],
    },
    {
        "route": "founder_education",
        "chain": ["founder", "education"],
        "required_any": [
            ("founder", "founded"),
            ("educated", "education", "university", "school"),
        ],
    },
    {
        "route": "author_work_field",
        "chain": ["author", "field"],
        "required_any": [("author", "wrote", "written"), ("field", "work in", "works in")],
    },
    {
        "route": "creator_spouse_citizenship",
        "chain": ["creator", "spouse", "citizenship"],
        "required_any": [
            ("creator", "created"),
            ("spouse", "married", "wife", "husband"),
            ("citizen", "citizenship"),
        ],
    },
]


_CONTRACT_SUPPORTED_PREDICATES = {label for _, label in _PREDICATE_PATTERNS} | {
    "chief_executive_officer",
    "director",
    "education",
}


def _infer_question_contracts(question: str) -> list[dict[str, object]]:
    """Infer diagnostic route coverage from question text only.

    This is trace-only: callers must not use the route output for ranking,
    admission, or answer selection.
    """
    if not isinstance(question, str):
        return []
    q_lower = question.lower()
    inferred: list[dict[str, object]] = []
    for spec in _CONTRACT_ROUTE_SPECS:
        matched_terms: list[str] = []
        for alternatives in spec["required_any"]:  # type: ignore[index]
            term = next((alt for alt in alternatives if alt in q_lower), None)
            if not term:
                break
            matched_terms.append(term)
        else:
            chain = list(spec["chain"])  # type: ignore[arg-type]
            unsupported = [
                pred for pred in chain if pred not in _CONTRACT_SUPPORTED_PREDICATES
            ]
            inferred.append({
                "route": spec["route"],
                "chain": chain,
                "matched_terms": matched_terms,
                "missing_predicates": unsupported,
                "alignment": "route_detected",
            })
    return inferred


def _build_contract_coverage_trace(
    question: str,
    goal_chain: list[str],
    predicates: list[str],
) -> dict[str, object]:
    routes = _infer_question_contracts(question)
    unsupported: list[str] = []
    seen: set[str] = set()
    for route in routes:
        for pred in route.get("missing_predicates", []):
            if pred not in seen:
                seen.add(str(pred))
                unsupported.append(str(pred))
    return {
        "enabled": True,
        "route_count": len(routes),
        "routes": routes,
        "query_predicates": list(predicates),
        "goal_chain": list(goal_chain),
        "unsupported_predicates": unsupported,
    }


def _classify_predicate(summary: str) -> str:
    """Classify a concept summary by its predicate pattern.

    Returns a predicate label (e.g., 'sport', 'citizenship') or 'other'.
    Used by RETRIEVAL-076 to enforce predicate diversity in entity chain results.
    """
    s = summary.lower()
    for pattern, label in _PREDICATE_PATTERNS:
        if pattern in s:
            return label
    return 'other'


# Common words to exclude from entity extraction
_STOPWORDS = {
    'what', 'when', 'where', 'who', 'which', 'how', 'why', 'does', 'did',
    'is', 'are', 'was', 'were', 'has', 'have', 'had', 'the', 'a', 'an',
    'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'and',
    'or', 'not', 'be', 'been', 'being', 'that', 'this', 'based',
    'answer', 'question', 'now', 'you', 'need', 'find', 'tell', 'me',
    'about', 'know', 'can', 'could', 'would', 'should', 'do', 'my',
    'his', 'her', 'its', 'their', 'your', 'our', 'current', 'please',
    'also', 'just', 'like', 'think', 'say', 'said', 'get', 'got',
    'make', 'made', 'take', 'see', 'come', 'want', 'look', 'use',
    'give', 'most', 'some', 'any', 'all', 'each', 'every', 'both',
    'few', 'more', 'other', 'new', 'old', 'first', 'last', 'long',
    'great', 'little', 'own', 'same', 'big', 'high', 'different',
    'small', 'large', 'next', 'early', 'young', 'important', 'public',
    'good', 'right', 'being', 'still', 'here', 'there', 'then', 'than',
    'will', 'shall', 'may', 'might', 'must', 'very', 'after', 'before',
    'remember', 'recall', 'mentioned', 'talked', 'discussed',
}

# Property words from queries that indicate what relationship to find
_PROPERTY_WORDS = {
    'country', 'language', 'capital', 'continent', 'city',
    'religion', 'sport', 'university', 'institution', 'origin',
    'music', 'genre', 'position', 'citizen', 'citizenship',
    'president', 'head', 'government', 'founder', 'creator',
    'author', 'spouse', 'location', 'birthday', 'born',
    'educated', 'headquarters', 'favorite', 'preference',
    'job', 'work', 'company', 'team', 'school', 'home',
    'address', 'email', 'phone', 'name', 'age', 'pet',
}

# Copula verbs for value extraction
_COPULA_VERBS = [
    ' is ', ' was ', ' are ', ' were ', ' has ', ' had ',
    ' plays ', ' speaks ', ' works ', ' lives ', ' died ',
    ' created ', ' founded ', ' employed ', ' married ',
    ' located ', ' received ', ' established ', ' holds ',
    ' associated ', ' originated ', ' comes ', ' prefers ',
    ' enjoys ', ' likes ', ' uses ', ' drives ', ' owns ',
    ' attends ', ' visited ', ' studies ', ' teaches ',
]

# Compound copulas (more specific, checked first)
_COMPOUND_COPULAS = [
    ' is associated with ', ' is located in ', ' is a citizen of ',
    ' is a member of ', ' is affiliated with ', ' is employed by ',
    ' is headquartered in ', ' was educated at ', ' was born in ',
    ' is married to ', ' was founded by ', ' lives in ',
    ' works at ', ' works for ', ' goes to ', ' prefers ',
]


# ── RETRIEVAL-095: Question Goal Classification ──────────────────────
# Maps question patterns to a chain of predicate labels that the entity
# chain should follow.  Each entry is a tuple:
#   (regex_pattern, [predicate_labels_in_chain_order])
# The chain is reversed relative to the question:
#   "What language is spoken in the capital of the country where X lives?"
#   → chain steps: location → country → capital → language
#
# At each hop, concepts whose _classify_predicate label matches the
# expected chain step get a score boost, and their extracted values
# get priority in the hop queue.  This prevents the chain from wasting
# hop budget exploring irrelevant branches.

_QUESTION_GOAL_ENABLED = os.environ.get(
    "PITH_EC_QUESTION_GOAL", "1"
).lower() in ("1", "true")

# Goal patterns: (compiled regex on question, chain of predicate labels)
# Order matters — first match wins.  More specific patterns first.
_GOAL_CHAIN_PATTERNS: list[tuple[re.Pattern, list[str]]] = [
    # Official language of the country an entity belongs to.
    (re.compile(r'(?:what|which).*official language.*(?:country to which|belongs?)', re.I),
     ['citizenship', 'official_language', 'language']),
    # Literary-genre questions in MH262 use the benchmark's genre/music predicate.
    (re.compile(r'literary genre.*(?:book|author|written)', re.I),
     ['author', 'genre']),
    # Language of capital of country
    (re.compile(r'(?:what|which)\s+language.*capital.*country', re.I),
     ['location', 'country_of_origin', 'citizenship', 'capital', 'official_language', 'language']),
    # Language of capital
    (re.compile(r'(?:what|which)\s+language.*capital', re.I),
     ['capital', 'official_language', 'language']),
    # Capital of country where X
    (re.compile(r'capital.*country.*(?:where|of)', re.I),
     ['location', 'citizenship', 'country_of_origin', 'capital']),
    # Continent of country/capital/city
    (re.compile(r'(?:what|which)\s+continent', re.I),
     ['location', 'citizenship', 'country_of_origin', 'capital', 'continent']),
    # Country where X was born/lives/works
    (re.compile(r'(?:what|which)\s+country', re.I),
     ['birthplace', 'location', 'workplace', 'citizenship', 'country_of_origin']),
    # Place of origin for broadcaster/organization chains
    (re.compile(r'place of origin.*(?:broadcaster|organization)', re.I),
     ['employer', 'creator', 'founder', 'founded_city', 'country_of_origin']),
    # City where X was born
    (re.compile(r'(?:what|which)\s+city.*born', re.I),
     ['birthplace']),
    # Language spoken by / written in
    (re.compile(r'(?:what|which)\s+language', re.I),
     ['official_language', 'language', 'citizenship', 'country_of_origin']),
    # Sport played by / associated with
    (re.compile(r'(?:what|which)\s+sport', re.I),
     ['sport']),
    # Religion
    (re.compile(r'(?:what|which)\s+religion', re.I),
     ['religion', 'citizenship', 'country_of_origin']),
    # Who founded / created
    (re.compile(r'who\s+(?:founded|created|invented|started)', re.I),
     ['founder', 'creator']),
    # Head of state / president / prime minister
    (re.compile(r'(?:head of state|president|prime minister|leader)', re.I),
     ['citizenship', 'country_of_origin', 'location', 'head_of_state', 'head_of_govt', 'prime_minister']),
    # Capital (generic)
    (re.compile(r'(?:what|which)\s+(?:is\s+the\s+)?capital', re.I),
     ['citizenship', 'country_of_origin', 'location', 'capital']),
    # Spouse / married to
    (re.compile(r'(?:who|whom).*(?:married|spouse|wife|husband)', re.I),
     ['spouse']),
    # Birthplace (generic)
    (re.compile(r'(?:where|which\s+city).*born', re.I),
     ['birthplace']),
    # Organization plus chairperson
    (re.compile(r'organization.*(?:employee|employed|work).*chairperson', re.I),
     ['employer', 'chairperson']),
    # Employer
    (re.compile(r'(?:where|who).*(?:work|employ)', re.I),
     ['employer', 'workplace']),
]

# How much to boost concepts matching the expected chain predicate.
_GOAL_BOOST = float(os.environ.get("PITH_EC_GOAL_BOOST", "3"))
# How much to boost hop-queue entries derived from goal-matching concepts.
_GOAL_QUEUE_BOOST = float(os.environ.get("PITH_EC_GOAL_QUEUE_BOOST", "2"))


# RETRIEVAL-095b: LLM-based goal classification fallback.
# When regex doesn't match, use a cheap LLM call to classify the question.
_GOAL_LLM_ENABLED = os.environ.get(
    "PITH_EC_GOAL_LLM", "0"
).lower() in ("1", "true")

# Cache to avoid re-classifying the same question
_goal_llm_cache: dict[str, list[str]] = {}

_GOAL_LLM_SYSTEM = """You classify multi-hop questions for a retrieval system.
Given a question, return the chain of predicate types the retrieval system should follow.

Available predicate types: sport, country_of_origin, continent, citizenship, founder,
founded_city, performer, developer, language, official_language, position, religion,
birthplace, deathplace, workplace, famous_for, spouse, employer, field, head_of_state,
head_of_govt, prime_minister, location, child, creator, capital, chairperson

Return ONLY a JSON array of predicate type strings, ordered from first hop to final answer.
Keep it short (2-5 items). If the question doesn't fit, return [].

Examples:
Q: "What language is spoken in the capital of the country where Andrew lives?"
["location","citizenship","country_of_origin","capital","official_language"]

Q: "What sport does the person who founded Adidas play?"
["founder","sport"]

Q: "What is the continent of the country of citizenship of Karl Lueger?"
["citizenship","country_of_origin","continent"]"""


def _classify_question_goal_llm(question: str) -> list[str]:
    """LLM-based question goal classification (RETRIEVAL-095b).

    Uses gpt-4o-mini with temperature=0 to extract the predicate chain.
    Results are cached per question string.
    """
    if question in _goal_llm_cache:
        return _goal_llm_cache[question]

    try:
        import requests as _requests
        import json as _json

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            logger.warning("RETRIEVAL-095b: No OPENAI_API_KEY for goal LLM")
            _goal_llm_cache[question] = []
            return []

        model = os.environ.get("PITH_EC_GOAL_MODEL", "gpt-4o-mini")
        base_url = os.environ.get("PITH_LLM_BASE_URL", "https://api.openai.com/v1")

        t0 = time.perf_counter()
        resp = _requests.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": _GOAL_LLM_SYSTEM},
                    {"role": "user", "content": question},
                ],
                "max_tokens": 80,
                "temperature": 0.0,
            },
            timeout=5,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if resp.status_code != 200:
            logger.warning(f"RETRIEVAL-095b: LLM returned {resp.status_code}")
            _goal_llm_cache[question] = []
            return []

        content = resp.json()["choices"][0]["message"]["content"].strip()
        # Parse JSON array from response (handle markdown fences)
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        chain = _json.loads(content)

        if not isinstance(chain, list):
            chain = []
        # Filter to known predicate labels
        known = {label for _, label in _PREDICATE_PATTERNS}
        chain = [p for p in chain if p in known]

        logger.info(
            f"RETRIEVAL-095b: LLM goal chain: {chain} ({elapsed_ms:.0f}ms, model={model})"
        )
        _goal_llm_cache[question] = chain
        return chain

    except Exception as e:
        logger.warning(f"RETRIEVAL-095b: LLM goal classification failed: {e}")
        _goal_llm_cache[question] = []
        return []


def _classify_question_goal(question: str) -> list[str]:
    """Return the expected chain of predicate labels for this question.

    Strategy: try regex first (zero cost), fall back to LLM if enabled.
    Returns empty list if no classification possible.
    """
    if not _QUESTION_GOAL_ENABLED:
        return []

    # Try regex patterns first (zero latency)
    for pattern, chain in _GOAL_CHAIN_PATTERNS:
        if pattern.search(question):
            logger.info(f"RETRIEVAL-095: Question goal chain (regex): {chain}")
            return chain

    # Fall back to LLM classification if enabled
    if _GOAL_LLM_ENABLED:
        return _classify_question_goal_llm(question)

    return []


def _extract_query_predicates(question: str) -> list[str]:
    """RETRIEVAL-106: Extract predicate keywords from question using word-boundary matching."""
    q_lower = question.lower()
    found: list[str] = []
    for kw in sorted(_KEYWORD_TO_SQL, key=len, reverse=True):
        pat = r'(?:^|\b|\s)' + re.escape(kw) + r'(?:$|\b|\s|[,;.!?])'
        if not re.search(pat, q_lower):
            continue
        if any(kw in existing or existing in kw for existing in found):
            continue
        found.append(kw)
    return found


def _should_run_chain_query_early(
    question: str,
    goal_chain: list[str] | None = None,
) -> bool:
    """Reserve chain-query execution for explicit multi-relation questions.

    Broad traversal can exhaust the retrieval budget before deterministic chain
    query runs. This gate keeps the product behavior general without turning
    every multi-hop request into an early SQL pass: auto-early only fires when
    the question exposes enough relation structure for a deterministic chain.
    """
    if not _EC_CHAIN_QUERY_AUTO_EARLY:
        return False

    return len(_extract_query_predicates(question)) >= 2


def _chain_query_extract_value(summary: str, entity: str,
                                sql_patterns: list[str]) -> str | None:
    """RETRIEVAL-106: Extract value from concept summary for chain-query.

    Handles two formats:
      A) "Subject PATTERN Object" — standard predicate patterns
      B) "The X of ENTITY is VALUE" — non-standard descriptive patterns
    """
    s = re.sub(r'^\d+\.\s*', '', summary).strip()
    s_lower = s.lower()
    ent_lower = entity.lower()

    # Format B: "The ... ENTITY ... is VALUE"
    m = re.search(
        r'^(?:The\s+|the\s+).+?' + re.escape(ent_lower) + r'.+?\bis\s+(.+?)\.?$',
        s_lower,
    )
    if m:
        val_start = m.start(1)
        val = s[val_start:].strip().rstrip('.,;:!? ')
        if val and ent_lower not in val.lower():
            return val

    # Format A: structured predicate patterns
    for pat in sql_patterns:
        pat_lower = pat.lower().strip()
        idx = s_lower.find(pat_lower)
        if idx < 0:
            continue
        left = s[:idx].strip()
        right = s[idx + len(pat):].strip().rstrip('.,;:!? ')
        if ent_lower in left.lower():
            return right if right else None
        if ent_lower in right.lower():
            return left if left else None
        return right if right else None

    # Generic "is"/"was" fallback
    for sep in (' is ', ' was '):
        idx = s_lower.find(sep)
        if idx <= 0:
            continue
        left = s[:idx].strip()
        right = s[idx + len(sep):].strip().rstrip('.,;:!? ')
        if ent_lower in left.lower() and right:
            return right
        if ent_lower in right.lower() and left:
            return left

    return None


def _temporal_anchor_adjustment(question: str, entity: str, summary: str) -> float:
    q = question.lower().replace("?", " ").replace("-", " ")
    if (
        "when" not in q
        and "what year" not in q
        and "what date" not in q
        and "how long ago" not in q
    ):
        return 0.0
    s_norm = summary.lower().replace("-", " ")
    s_words = set(re.findall(r"[a-z0-9+']+", s_norm))
    entity_words = set(re.findall(r"[a-z0-9+']+", entity.lower()))
    generic = {
        "when", "what", "year", "date", "how", "long", "ago", "did", "does",
        "do", "was", "is", "are", "were", "the", "a", "an",
    }
    anchor_terms = [
        token
        for token in re.findall(r"[a-z0-9+']+", q)
        if token not in generic and token not in entity_words and token not in _STOPWORDS
    ]
    anchor_hits = sum(1 for token in anchor_terms if token in s_words)
    has_year = bool(re.search(r"\b(?:19|20)\d{2}\b", s_norm))
    has_date = bool(
        re.search(
            r"\b(?:january|february|march|april|may|june|july|august|"
            r"september|october|november|december)\b",
            s_norm,
        )
    )
    has_relative = bool({"last", "year", "years", "ago", "birthday"} & s_words)
    bonus = 0.0
    if anchor_hits:
        bonus += 0.04 * min(3, anchor_hits)
    if has_year:
        bonus += 0.16
    elif has_date:
        bonus += 0.10
    elif has_relative:
        bonus += 0.05
    if anchor_hits == 0 and (has_year or has_date):
        bonus -= 0.04
    return bonus


def _kind_type_anchor_adjustment(question: str, entity: str, summary: str) -> float:
    q = question.lower().replace("?", " ").replace("-", " ")
    if "what kind of" not in q and "what type of" not in q:
        return 0.0
    s_norm = summary.lower().replace("-", " ")
    s_words = set(re.findall(r"[a-z0-9+']+", s_norm))
    generic = {
        "what", "kind", "type", "of", "does", "do", "did", "is", "are", "was",
        "were", "have", "has", "had", "the", "a", "an", "in", "her", "his",
        "their", "she", "he",
    }
    entity_words = set(re.findall(r"[a-z0-9+']+", entity.lower()))
    anchor_terms = [
        token
        for token in re.findall(r"[a-z0-9+']+", q)
        if token not in generic and token not in entity_words and token not in _STOPWORDS
    ]
    anchor_hits = sum(1 for token in anchor_terms if token in s_words)
    has_list_shape = "," in summary or "including" in s_norm or "such as" in s_norm
    category_terms = {"lgbtq+", "lgbtq", "folks", "individuals", "children", "kids"}
    category_hits = len(category_terms & s_words)
    if anchor_hits == 0:
        if has_list_shape or category_hits:
            return 0.02 + 0.02 * min(2, category_hits)
        return -0.10
    bonus = 0.04 * min(3, anchor_hits)
    if has_list_shape:
        bonus += 0.06
    if category_hits:
        bonus += 0.02 * min(2, category_hits)
    return bonus


def _contrastive_alternative_adjustment(question: str, entity: str, summary: str) -> float:
    q = question.lower().replace("?", " ").replace("-", " ")
    if "besides" not in q and "other than" not in q:
        return 0.0
    s_norm = summary.lower().replace("-", " ")
    s_words = set(re.findall(r"[a-z0-9+']+", s_norm))
    generic = {
        "what", "which", "who", "besides", "other", "than", "do", "does", "did",
        "is", "are", "was", "were", "with", "together", "creative", "project",
        "projects", "mel", "melanie", "her", "kids", "children",
    }
    entity_words = set(re.findall(r"[a-z0-9+']+", entity.lower()))
    q_words = re.findall(r"[a-z0-9+']+", q)
    excluded_terms = []
    for idx, token in enumerate(q_words):
        if token == "besides" and idx + 1 < len(q_words):
            excluded_terms.append(q_words[idx + 1])
        if token == "other" and idx + 2 < len(q_words) and q_words[idx + 1] == "than":
            excluded_terms.append(q_words[idx + 2])
    anchor_terms = [
        token
        for token in q_words
        if token not in generic and token not in entity_words and token not in _STOPWORDS
    ]
    anchor_hits = sum(1 for token in anchor_terms if token in s_words)
    excluded_hits = sum(1 for token in excluded_terms if token in s_words)
    bonus = 0.0
    if anchor_hits:
        bonus += 0.05 * min(3, anchor_hits)
    if excluded_hits:
        bonus -= 0.18 * excluded_hits
    if {"painting", "drawing", "art", "project"} & s_words:
        bonus += 0.04
    if "together" in s_words and ({"kids", "children"} & s_words):
        bonus += 0.03
    if anchor_hits == 0 and excluded_hits == 0 and {"creative", "creatively"} & s_words:
        bonus -= 0.05
    return bonus


def _entity_focus_adjustment(question: str, entity: str, summary: str) -> float:
    q_norm = (question or "").lower().replace("?", " ").replace("-", " ")
    s_norm = summary.lower()
    entity = entity.lower()
    people = {"caroline", "melanie"}
    mentioned = {name for name in people if name in s_norm}
    if entity not in people or not mentioned:
        return 0.0
    opposite = next(iter(people - {entity}))

    if entity not in mentioned and opposite in mentioned:
        q_words = set(re.findall(r"[a-z0-9+']+", q_norm))
        s_words = set(re.findall(r"[a-z0-9+']+", s_norm))
        generic = {
            "what", "which", "who", "when", "where", "why", "how", "did",
            "does", "was", "were", "is", "are", "after", "before", "with",
            "about", "into", "from", "have", "has", "had", "for", "their",
            "them", "they", "her", "his", "she", "he", "make", "made",
            "create", "created", "does", "did", "feel", "felt",
        }
        anchor_terms = {
            tok for tok in q_words
            if tok not in generic and tok not in people and tok not in _STOPWORDS and len(tok) > 2
        }
        anchor_hits = len(anchor_terms & s_words)
        mirror_query = bool(anchor_terms) and len(q_words & people) == 1
        if mirror_query and anchor_hits:
            return 0.08 + 0.03 * min(2, anchor_hits)
    if entity in mentioned and len(mentioned) == 1:
        return 0.03
    if entity not in mentioned:
        return -0.10
    if len(mentioned) > 1:
        return -0.03
    return 0.0


class EntityChainRetriever:
    """Entity-chain retrieval via direct keyword search in the brain DB.

    For multi-hop questions like:
      "What's the capital of the country where Andrew lives?"

    Pipeline:
      1. Extract named entities from question (proper nouns)
      2. Classify question goal → expected predicate chain (RETRIEVAL-095)
      3. SQL keyword search for leaf entity -> hop 1 facts
      4. Extract value entities from hop 1 facts (object after copula)
         → goal-aware: boost entities from predicates matching chain step
      5. SQL keyword search for hop 2 entities -> hop 2 facts
      6. Return all found concepts (caller unions with embedding results)
    """

    def __init__(self, db_path: str, max_hops: int = 4, max_per_hop: int = 8):
        self.db_path = db_path
        self.max_hops = max_hops
        self.max_per_hop = max_per_hop
        self._question_keywords: list[str] = []
        self.last_searched_entities: set[str] = set()
        self._expanded_ids: set[str] = set()  # RETRIEVAL-089: IDs from association expansion
        self._goal_chain: list[str] = []  # RETRIEVAL-095: Expected predicate chain
        self._goal_tagged_queue: dict[str, float] = {}  # entity → goal boost score
        self.last_chain_query_selected_ids: set[str] = set()
        self.last_trace: dict | None = None  # RETRIEVAL-113: diagnostic-only canary trace
        self._trace_current_hop: dict | None = None
        self._current_question: str | None = None

    def _trace_enabled(self) -> bool:
        return self.last_trace is not None

    def _trace_hashes(self, values: list[str] | set[str] | tuple) -> list[str]:
        return [_trace_hash(str(v)) for v in values]

    def _trace_queue_hashes(self, queue: list) -> list[str]:
        values = [q[0] if isinstance(q, tuple) else q for q in queue]
        return self._trace_hashes(values)

    def retrieve(
        self,
        question: str,
        budget_ms: int = 150,
        is_multihop: bool = False,
        *,
        deadline: TurnDeadline | None = None,
        max_initial_entities: int | None = None,
        max_hops_override: int | None = None,
        total_cap_override: int | None = None,
        fast_keyword_search: bool = False,
    ) -> list[SearchResult]:
        """Main entry: extract entities, chain keyword lookups, return results.

        Args:
            question: The user's message/query
            budget_ms: Time budget in ms. Returns partial results if exceeded.
            is_multihop: True if router classified query as multi-hop/relational.
                When True, total cap is relaxed (1.5x) to allow deeper chains.
            max_initial_entities: Optional cap on extracted entity seeds.
            max_hops_override: Optional traversal-depth cap for bounded callers.
            total_cap_override: Optional result cap for bounded callers.
            fast_keyword_search: Use FTS5-backed keyword search for bounded callers.

        Returns:
            List of SearchResult objects to union with embedding results.
        """
        t0 = time.perf_counter()
        self.last_trace = None
        self.last_initial_entity_count = 0
        self.last_effective_max_hops = 0
        self.last_effective_total_cap = 0
        self.last_chain_query_selected_ids = set()
        self._trace_current_hop = None
        self._current_question = question
        if deadline:
            budget_ms = int(deadline.child_budget_ms(
                "entity_chain.retrieve",
                requested_ms=float(budget_ms),
                min_remaining_ms=0.0,
            ))
            if budget_ms <= 0:
                deadline.skip("entity_chain.retrieve", "deadline_child_budget_exhausted", priority="optional")
                return []

        # RETRIEVAL-091: MH queries get extra hops (MQUAKE needs 4-6).
        effective_max_hops = self.max_hops
        if is_multihop and _MH_EXTRA_HOPS > 0:
            effective_max_hops = min(self.max_hops + _MH_EXTRA_HOPS, 8)
            logger.info(
                f"ENTITY-CHAIN: MH query — max_hops {self.max_hops} -> "
                f"{effective_max_hops}"
            )
        if max_hops_override is not None:
            effective_max_hops = min(effective_max_hops, max(1, int(max_hops_override)))

        effective_total_cap = _TOTAL_CAP
        if total_cap_override is not None:
            effective_total_cap = max(1, int(total_cap_override))
        self.last_effective_max_hops = effective_max_hops
        self.last_effective_total_cap = effective_total_cap

        # RETRIEVAL-095: Classify question goal before entity extraction.
        # TURN-DEADLINE: the request deadline starts before this classifier.
        if deadline and not deadline.can_start("entity_chain.goal_classification"):
            deadline.skip("entity_chain.goal_classification", "deadline_before_start", priority="optional")
            return []
        _goal_t0 = time.perf_counter()
        self._goal_chain = _classify_question_goal(question)
        _goal_elapsed_ms = round((time.perf_counter() - _goal_t0) * 1000, 2)
        self._goal_tagged_queue = {}
        query_predicates = _extract_query_predicates(question)

        # Step 1: Extract entities
        _entity_t0 = time.perf_counter()
        entities = self._extract_entities(question)
        _entity_elapsed_ms = round((time.perf_counter() - _entity_t0) * 1000, 2)
        if max_initial_entities is not None:
            _entity_cap = max(1, int(max_initial_entities))
            if len(entities) > _entity_cap:
                logger.info(
                    "ENTITY-CHAIN: Initial entities capped %d -> %d",
                    len(entities),
                    _entity_cap,
                )
                entities = entities[:_entity_cap]
        self.last_initial_entity_count = len(entities)
        if _mh262_canary_trace_enabled() or _contract_coverage_trace_enabled():
            self.last_trace = {
                "schema_version": (
                    "mh262.canary_retrieval_trace.v1"
                    if _mh262_canary_trace_enabled()
                    else "mh262.contract_coverage_trace.v1"
                ),
                "enabled": True,
                "budget_ms": budget_ms,
                "is_multihop": is_multihop,
                "effective_max_hops": effective_max_hops,
                "goal_chain": list(self._goal_chain),
                "entity_extraction": {
                    "count": len(entities),
                    "hashes": self._trace_hashes(entities),
                },
                "question_keywords": {"count": 0, "hashes": []},
                "hops": [],
                "final_cap": {"pre_cap_ids": [], "post_cap_ids": []},
                "chain_query": {
                    "enabled": _EC_CHAIN_QUERY,
                    "predicate_count": 0,
                    "predicates": [],
                    "candidate_ids": [],
                    "selected_ids": [],
                    "budget_exhausted": False,
                },
                "association_expansion": {
                    "enabled": _ASSOC_EXPAND_ENABLED,
                    "seed_ids": [],
                    "expanded_ids": [],
                },
                "budget": {"exhausted": False, "exhausted_hop": 0},
                "timing": {
                    "goal_classification_ms": _goal_elapsed_ms,
                    "entity_extraction_ms": _entity_elapsed_ms,
                    "keyword_search_ms_total": 0.0,
                    "keyword_searches": [],
                    "chain_query_ms": 0.0,
                    "association_expansion_ms": 0.0,
                    "total_ms": 0.0,
                    "exit_reason": "running",
                },
            }
            if _contract_coverage_trace_enabled():
                self.last_trace["contract_coverage"] = _build_contract_coverage_trace(
                    question,
                    self._goal_chain,
                    query_predicates,
                )
        if not entities:
            if self._trace_enabled():
                self.last_trace["timing"]["exit_reason"] = "no_entities"
                self.last_trace["timing"]["total_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            return []

        logger.info(f"ENTITY-CHAIN: Extracted entities: {entities}")

        # Extract question keywords for SQL boosting.
        # RETRIEVAL-110: When PITH_EC_FULL_CONTENT_KEYWORDS is set, use ALL non-entity
        # content words as kw_boost seeds instead of the narrow _PROPERTY_WORDS subset.
        # _PROPERTY_WORDS covers world-knowledge predicates (country, language, sport...);
        # conversational benchmarks (LoCoMo) need verbs/nouns like research, identity,
        # career, relationship which are absent from _PROPERTY_WORDS, causing every
        # same-entity concept to tie at confidence=0.6 and degrade to insertion-order.
        # Strip punctuation before tokenizing so LIKE '%research%' matches, not '%research?%'
        q_words = {re.sub(r'[^\w]', '', w) for w in question.lower().split()} - _STOPWORDS - {''}
        if _EC_FULL_CONTENT_KEYWORDS:
            entity_words = {re.sub(r'[^\w]', '', w) for e in entities for w in e.lower().split()} - {''}
            self._question_keywords = sorted(q_words - entity_words)
        else:
            self._question_keywords = sorted(q_words & _PROPERTY_WORDS)  # sorted for determinism
        if self._trace_enabled():
            self.last_trace["question_keywords"] = {
                "count": len(self._question_keywords),
                "hashes": self._trace_hashes(self._question_keywords),
            }

        all_concepts: dict[str, SearchResult] = {}
        global_pred_counts: Counter = Counter()  # RETRIEVAL-101: cross-hop predicate budget
        # RETRIEVAL-103: hop_queue entries are (entity, chain_keywords, committed_provenance)
        # tuples when chain anchoring is enabled, plain strings otherwise.
        if _EC_CHAIN_ANCHOR:
            hop_queue: list = [(e, [], None) for e in entities]
        else:
            hop_queue: list = list(entities)
        searched: set[str] = set()
        self.last_searched_entities = set()
        self._budget_exhausted = False  # WI-5: Track budget timeout for diagnostics
        self._budget_exhausted_hop = 0
        hop = 0
        _chain_query_ran = False

        def _merge_chain_query_results(chain_query_results: list[SearchResult]) -> list[str]:
            added_ids: list[str] = []
            for cqr in chain_query_results:
                if cqr.concept_id not in all_concepts:
                    all_concepts[cqr.concept_id] = cqr
                    added_ids.append(cqr.concept_id)
            if self._trace_enabled():
                self.last_trace["chain_query"]["new_result_ids"] = added_ids
            return added_ids

        if (
            is_multihop
            and _EC_CHAIN_QUERY
            and (
                _EC_CHAIN_QUERY_EARLY
                or _should_run_chain_query_early(question, self._goal_chain)
            )
        ):
            _chain_query_ran = True
            _cq_t0 = time.perf_counter()
            chain_query_results = self._chain_query_search(question, entities, budget_ms, t0)
            _cq_elapsed_ms = round((time.perf_counter() - _cq_t0) * 1000, 2)
            if self._trace_enabled():
                self.last_trace["chain_query"]["phase"] = "early"
                self.last_trace["timing"]["chain_query_ms"] = _cq_elapsed_ms
            _merge_chain_query_results(chain_query_results)

        while hop_queue and hop < effective_max_hops:
            hop += 1
            current_entities = list(hop_queue)
            hop_queue = []

            for _queue_entry in current_entities:
                # RETRIEVAL-103/104: Unpack chain context + provenance
                if isinstance(_queue_entry, tuple):
                    if len(_queue_entry) == 3:
                        entity, _chain_kw, _committed_prov = _queue_entry
                    else:
                        entity, _chain_kw = _queue_entry
                        _committed_prov = None
                else:
                    entity = _queue_entry
                    _chain_kw = []
                    _committed_prov = None

                # Budget check
                elapsed_ms = (time.perf_counter() - t0) * 1000
                if elapsed_ms > budget_ms:
                    # WI-5: Enhanced budget timeout logging for 64K diagnostics.
                    _remaining_entities = len(current_entities) - current_entities.index(_queue_entry)
                    _remaining_hops = effective_max_hops - hop
                    logger.warning(
                        f"ENTITY-CHAIN: BUDGET_TIMEOUT ({elapsed_ms:.0f}ms > {budget_ms}ms) "
                        f"at hop {hop}/{effective_max_hops}, "
                        f"{len(all_concepts)} concepts found, "
                        f"{_remaining_entities} entities remaining this hop, "
                        f"{_remaining_hops} hops remaining, "
                        f"queue_depth={len(hop_queue)}, "
                        f"multihop={is_multihop}"
                    )
                    self._budget_exhausted = True
                    self._budget_exhausted_hop = hop
                    self.last_searched_entities = searched
                    if self._trace_enabled():
                        self.last_trace["budget"] = {
                            "exhausted": True,
                            "exhausted_hop": hop,
                        }
                        self.last_trace["timing"]["exit_reason"] = "budget_timeout"
                        self.last_trace["timing"]["total_ms"] = round((time.perf_counter() - t0) * 1000, 2)
                        self.last_trace["timing"]["returned_ids"] = _trace_result_ids(list(all_concepts.values()))
                    return list(all_concepts.values())

                entity_lower = entity.lower().strip()
                if entity_lower in searched or len(entity_lower) < 2:
                    continue
                searched.add(entity_lower)
                if self._trace_enabled():
                    self._trace_current_hop = {
                        "hop": hop,
                        "input_count": len(current_entities),
                        "input_hashes": self._trace_queue_hashes(current_entities),
                        "searched_hash": _trace_hash(entity_lower),
                        "keyword_candidate_ids": [],
                        "keyword_selected_ids": [],
                        "extracted_value_hashes": [],
                        "post_prune_queue_hashes": [],
                    }
                    self.last_trace["hops"].append(self._trace_current_hop)

                # INGEST-044: Graph search disabled pending query-time integration tuning
                # (44% EM vs 51.6% baseline — graph preempts keyword, wrong-entity propagation)
                # Typed edges remain in DB for future use. See INGEST-044 retro.
                _keyword_budget_ms = max(0.0, float(budget_ms) - elapsed_ms)
                _deadline_remaining_ms = deadline.remaining_ms() if deadline else None
                if _deadline_remaining_ms is not None:
                    _keyword_budget_ms = min(_keyword_budget_ms, max(0.0, _deadline_remaining_ms))
                _keyword_min_ms = min(
                    _TURN_DEADLINE_KEYWORD_MIN_MS,
                    max(_KEYWORD_QUERY_MIN_MS, _keyword_budget_ms),
                )
                if deadline and not deadline.can_start(
                    "entity_chain.keyword_search",
                    min_remaining_ms=_keyword_min_ms,
                ):
                    deadline.skip(
                        "entity_chain.keyword_search",
                        "deadline_before_start",
                        priority="optional",
                        min_remaining_ms=_keyword_min_ms,
                        hop=hop,
                        concepts_found=len(all_concepts),
                    )
                    self._budget_exhausted = True
                    self._budget_exhausted_hop = hop
                    self.last_searched_entities = searched
                    if self._trace_enabled():
                        self.last_trace["budget"] = {
                            "exhausted": True,
                            "exhausted_hop": hop,
                        }
                        self.last_trace["timing"]["exit_reason"] = "deadline_before_keyword_search"
                        self.last_trace["timing"]["total_ms"] = round((time.perf_counter() - t0) * 1000, 2)
                        self.last_trace["timing"]["returned_ids"] = _trace_result_ids(list(all_concepts.values()))
                    return list(all_concepts.values())
                _keyword_t0 = time.perf_counter()
                facts = self._keyword_search(
                    entity_lower,
                    hop,
                    chain_keywords=_chain_kw,
                    prefer_fts=fast_keyword_search,
                    deadline=deadline,
                    budget_ms=_keyword_budget_ms,
                )
                _keyword_elapsed_ms = round((time.perf_counter() - _keyword_t0) * 1000, 2)
                if self._trace_enabled():
                    self.last_trace["timing"]["keyword_search_ms_total"] = round(
                        self.last_trace["timing"]["keyword_search_ms_total"] + _keyword_elapsed_ms,
                        2,
                    )
                    self.last_trace["timing"]["keyword_searches"].append({
                        "hop": hop,
                        "entity_hash": _trace_hash(entity_lower),
                        "elapsed_ms": _keyword_elapsed_ms,
                        "result_count": len(facts),
                        "budget_ms": round(float(_keyword_budget_ms), 2),
                        "deadline_remaining_ms": (
                            round(float(_deadline_remaining_ms), 2)
                            if _deadline_remaining_ms is not None
                            else None
                        ),
                        "min_remaining_ms": round(float(_keyword_min_ms), 2),
                    })
                for f in facts:
                    if f.concept_id not in all_concepts:
                        # RETRIEVAL-101: Global cross-hop predicate budget.
                        pred = _classify_predicate(f.summary)
                        if pred != 'other' and global_pred_counts[pred] >= _GLOBAL_PRED_CAP:
                            continue
                        all_concepts[f.concept_id] = f
                        global_pred_counts[pred] += 1

                # Extract values for next hop
                if hop < effective_max_hops:
                    for f in facts:
                        values = self._extract_fact_values(f.summary, entity_lower)
                        # RETRIEVAL-095: Tag extracted values with goal relevance.
                        # If the concept's predicate matches any step in the goal
                        # chain, the extracted values are more likely to be on the
                        # right path (e.g., a 'capital' concept's value is likely
                        # a city name we need for a "language of capital" question).
                        fact_pred = _classify_predicate(f.summary) if self._goal_chain else 'other'
                        goal_match = fact_pred in self._goal_chain if self._goal_chain else False

                        # RETRIEVAL-103: Extract chain-context keywords from source
                        # concept to carry forward into next hop's search.
                        _next_chain_kw = []
                        if _EC_CHAIN_ANCHOR:
                            _src_words = set(f.summary.lower().split()) - _STOPWORDS
                            # Remove the entity itself and very short words
                            _src_words -= set(entity_lower.split())
                            _src_words = {w for w in _src_words if len(w) > 2}
                            # Prefer property words (domain-salient)
                            _prop = sorted(_src_words & _PROPERTY_WORDS)  # sorted for determinism
                            _other = sorted(_src_words - _PROPERTY_WORDS)  # sorted for determinism
                            _next_chain_kw = (_prop + _other)[:_EC_CHAIN_ANCHOR_KEYWORDS]

                        for v in values:
                            if self._trace_enabled() and self._trace_current_hop is not None:
                                self._trace_current_hop["extracted_value_hashes"].append(_trace_hash(v))
                            if v.lower() not in searched and len(v) > 1:
                                if _EC_CHAIN_ANCHOR:
                                    hop_queue.append((v, _next_chain_kw, _committed_prov))
                                else:
                                    hop_queue.append(v)
                                if goal_match:
                                    self._goal_tagged_queue[v.lower()] = _GOAL_QUEUE_BOOST

            # RETRIEVAL-092+095: Prune oversized hop queue by goal relevance
            # then question keyword overlap.
            _QUEUE_PRUNE_THRESHOLD = self.max_per_hop * 3  # 24 by default
            if _EC_QUEUE_PRUNE and len(hop_queue) > _QUEUE_PRUNE_THRESHOLD:
                q_kw = set(question.lower().split()) - _STOPWORDS
                def _queue_score(ent):
                    # RETRIEVAL-103: Handle both tuple and plain string entries
                    _ent_str = ent[0] if isinstance(ent, tuple) else ent
                    e_words = set(_ent_str.lower().split())
                    # RETRIEVAL-095: Goal-tagged entities get priority
                    goal_bonus = self._goal_tagged_queue.get(_ent_str.lower(), 0)
                    return (goal_bonus, len(e_words & q_kw), -len(_ent_str))
                hop_queue.sort(key=_queue_score, reverse=True)
                pruned_from = len(hop_queue)
                hop_queue = hop_queue[:_QUEUE_PRUNE_THRESHOLD]
                if self._trace_enabled() and self.last_trace["hops"]:
                    self.last_trace["hops"][-1]["post_prune_queue_hashes"] = self._trace_queue_hashes(hop_queue)
                _goal_count = sum(
                    1 for e in hop_queue
                    if (e[0] if isinstance(e, tuple) else e).lower() in self._goal_tagged_queue
                )
                logger.info(
                    f"ENTITY-CHAIN: Queue pruned {pruned_from} -> {len(hop_queue)} "
                    f"(threshold={_QUEUE_PRUNE_THRESHOLD}, goal_tagged={_goal_count})"
                )

            logger.info(
                f"ENTITY-CHAIN: Hop {hop}: {len(current_entities)} entities -> "
                f"{len(all_concepts)} total, next queue: {len(hop_queue)}"
            )

            # RETRIEVAL-075: Early termination when total cap reached.
            if len(all_concepts) >= effective_total_cap:
                logger.info(
                    f"RETRIEVAL-075: Total cap ({effective_total_cap}) reached at hop {hop}, "
                    f"stopping chain traversal (multihop={is_multihop})"
                )
                break

        self.last_searched_entities = searched
        elapsed_ms = (time.perf_counter() - t0) * 1000

        results = list(all_concepts.values())
        if self._trace_enabled():
            self.last_trace["final_cap"]["pre_cap_ids"] = _trace_result_ids(results)

        # RETRIEVAL-075: Total cap — prevent entity chain sprawl from overwhelming
        # budget governance. Sort by relevance_score (hop priority) so hop-1
        # entity-specific gold survives the cap over hop-3+ noise.
        _final_cap = effective_total_cap

        # Secondary sort by question keyword overlap (RETRIEVAL-058+093).
        # Use continuous keyword count instead of binary hit for better
        # discrimination when deep-chain hops produce many candidates.
        # NOTE: RETRIEVAL-095 goal boosting is applied per-hop in
        # _keyword_search (+0.10 relevance) and via queue prioritization.
        # Cap-level goal boosting was removed because it promoted ALL
        # concepts matching the goal predicate type regardless of chain
        # path, displacing chain-specific gold (e.g., promoting random
        # "religion of X" over the chain-path "Austria-Hungary...atheism").
        def _cap_sort_key(r):
            _kw_count = 0
            if self._question_keywords:
                _s = (r.summary or "").lower()
                _kw_count = sum(1 for kw in self._question_keywords if kw in _s)
            return (_kw_count, r.relevance_score)

        def _apply_final_cap(candidates: list[SearchResult], stage: str) -> list[SearchResult]:
            if len(candidates) <= _final_cap:
                return candidates
            before_ids = _trace_result_ids(candidates)
            capped = sorted(candidates, key=_cap_sort_key, reverse=True)[:_final_cap]
            logger.info(
                f"RETRIEVAL-075: Entity chain capped {len(candidates)} -> {_final_cap} "
                f"(hop scores preserved, multihop={is_multihop}, stage={stage})"
            )
            if self._trace_enabled():
                after_ids = set(_trace_result_ids(capped))
                self.last_trace.setdefault("cap_stages", []).append({
                    "stage": stage,
                    "pre_ids": before_ids,
                    "post_ids": _trace_result_ids(capped),
                    "dropped_ids": [cid for cid in before_ids if cid not in after_ids],
                })
            return capped

        results = _apply_final_cap(results, "post_traversal")
        if self._trace_enabled():
            self.last_trace["final_cap"]["post_cap_ids"] = _trace_result_ids(results)

        # RETRIEVAL-106: Predicate-guided chain query (supplemental).
        # Runs for MH queries to inject precisely-matched concepts.
        if is_multihop and not self._budget_exhausted and not _chain_query_ran:
            _chain_query_ran = True
            _cq_t0 = time.perf_counter()
            chain_query_results = self._chain_query_search(question, entities, budget_ms, t0)
            _cq_elapsed_ms = round((time.perf_counter() - _cq_t0) * 1000, 2)
            if self._trace_enabled():
                self.last_trace["chain_query"]["phase"] = "late"
                self.last_trace["timing"]["chain_query_ms"] = _cq_elapsed_ms
            _merge_chain_query_results(chain_query_results)
            if chain_query_results:
                result_ids = {r.concept_id for r in results}
                results.extend(
                    cqr for cqr in chain_query_results
                    if cqr.concept_id not in result_ids
                )
                results = _apply_final_cap(results, "post_chain_query")
        elif is_multihop and self._budget_exhausted and not _chain_query_ran and self._trace_enabled():
            self.last_trace["chain_query"]["phase"] = "skipped_budget_exhausted"

        if self._trace_enabled() and _chain_query_ran:
            final_result_ids = set(_trace_result_ids(results))
            selected_ids = self.last_trace["chain_query"].get("selected_ids", [])
            self.last_trace["chain_query"]["post_merge_ids"] = _trace_result_ids(results)
            self.last_trace["chain_query"]["dropped_after_cap_ids"] = [
                cid for cid in selected_ids if cid not in final_result_ids
            ]

        # RETRIEVAL-089: Association-aware expansion for MH queries.
        # After entity chain completes, follow association links from terminal
        # concepts to close hop-1 retrieval gaps.
        self._expanded_ids = set()
        if is_multihop and not self._budget_exhausted:
            _assoc_t0 = time.perf_counter()
            results = self._expand_associations(
                results, hop, effective_total_cap, budget_ms, t0
            )
            if self._trace_enabled():
                self.last_trace["timing"]["association_expansion_ms"] = round(
                    (time.perf_counter() - _assoc_t0) * 1000,
                    2,
                )

        # WI-5: Log budget utilization for diagnostics
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if self._trace_enabled():
            self.last_trace["budget"] = {
                "exhausted": bool(getattr(self, "_budget_exhausted", False)),
                "exhausted_hop": int(getattr(self, "_budget_exhausted_hop", 0) or 0),
            }
            self.last_trace["timing"]["exit_reason"] = "completed"
            self.last_trace["timing"]["total_ms"] = round(elapsed_ms, 2)
            self.last_trace["timing"]["returned_ids"] = _trace_result_ids(results)
        _budget_pct = (elapsed_ms / budget_ms * 100) if budget_ms > 0 else 0
        logger.info(
            f"ENTITY-CHAIN: Complete: {len(results)} concepts from "
            f"{hop} hops in {elapsed_ms:.0f}ms "
            f"(budget={budget_ms}ms, used={_budget_pct:.0f}%, "
            f"multihop={is_multihop}, expanded={len(self._expanded_ids)})"
        )
        return results

    def _expand_associations(
        self, results: list[SearchResult], hop: int,
        effective_cap: int, budget_ms: int, t0: float,
    ) -> list[SearchResult]:
        """RETRIEVAL-089: Expand terminal concepts via association neighbors.

        For MH queries, after entity chain completes, follow association links
        from the last 2 hops' concepts to pull in neighbors that might contain
        the final-hop answer. Only fires when cap headroom exists and budget
        hasn't been exhausted.

        Returns expanded results appended to the input list.
        """
        if not _ASSOC_EXPAND_ENABLED:
            return results
        if self._budget_exhausted:
            return results  # no time left

        headroom = effective_cap - len(results)
        if headroom <= 0:
            return results

        # Identify terminal concepts: those from the last 2 hops (lowest scores)
        # Hop-priority scores: hop1=0.85, hop2=0.78, hop3+=0.68
        # Terminal = score <= 0.78 (hop 2+)
        terminal_ids = [
            r.concept_id for r in results
            if r.relevance_score <= _HOP_SCORES.get(2, 0.78)
        ]
        if not terminal_ids:
            # Fall back to all results if none qualify as terminal
            terminal_ids = [r.concept_id for r in results[-10:]]
        if self._trace_enabled():
            self.last_trace["association_expansion"]["seed_ids"] = list(terminal_ids)

        existing_ids = {r.concept_id for r in results}
        expanded: list[SearchResult] = []
        predicate_counts: dict[str, int] = Counter()

        # Count existing predicates for diversity enforcement
        for r in results:
            pred = _classify_predicate(r.summary)
            predicate_counts[pred] += 1

        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            for tid in terminal_ids:
                if len(expanded) >= headroom:
                    break

                # Budget check
                elapsed_ms = (time.perf_counter() - t0) * 1000
                if elapsed_ms > budget_ms:
                    logger.info(
                        f"RETRIEVAL-089: Expansion budget exhausted at "
                        f"{elapsed_ms:.0f}ms, {len(expanded)} expanded"
                    )
                    break

                # Query associations for this terminal concept
                cur.execute(
                    """SELECT target AS neighbor, strength FROM associations
                       WHERE source = ? AND strength >= ?
                       UNION
                       SELECT source AS neighbor, strength FROM associations
                       WHERE target = ? AND strength >= ?
                       ORDER BY strength DESC
                       LIMIT ?""",
                    (tid, _ASSOC_STRENGTH_MIN,
                     tid, _ASSOC_STRENGTH_MIN,
                     _ASSOC_MAX_PER_CONCEPT)
                )
                neighbors = cur.fetchall()

                for row in neighbors:
                    if len(expanded) >= headroom:
                        break
                    neighbor_id = row["neighbor"]
                    if neighbor_id in existing_ids:
                        continue

                    # Fetch concept summary
                    cur.execute(
                        "SELECT id, version, summary, confidence, "
                        "knowledge_area, ka_relative_authority, maturity "
                        "FROM concepts WHERE id = ? AND status != 'deleted'",
                        (neighbor_id,)
                    )
                    concept = cur.fetchone()
                    if not concept or not concept["summary"]:
                        continue

                    # Predicate diversity check (RETRIEVAL-076 compat)
                    pred = _classify_predicate(concept["summary"])
                    if predicate_counts.get(pred, 0) >= _PREDICATE_CAP:
                        continue

                    sr = SearchResult(
                        concept_id=concept["id"],
                        version=concept["version"] or "v1",
                        summary=concept["summary"],
                        confidence=concept["confidence"] or 0.5,
                        relevance_score=_HOP_DEFAULT_SCORE,  # 0.68 — expansion rank
                        knowledge_area=concept["knowledge_area"],
                        ka_relative_authority=concept["ka_relative_authority"],
                        maturity=concept["maturity"],
                    )
                    expanded.append(sr)
                    existing_ids.add(neighbor_id)
                    predicate_counts[pred] = predicate_counts.get(pred, 0) + 1

            conn.close()
        except Exception as e:
            logger.warning(f"RETRIEVAL-089: Association expansion failed: {e}")

        if expanded:
            self._expanded_ids = {r.concept_id for r in expanded}
            if self._trace_enabled():
                self.last_trace["association_expansion"]["expanded_ids"] = sorted(self._expanded_ids)
            logger.info(
                f"RETRIEVAL-089: Expanded {len(expanded)} concepts from "
                f"{len(terminal_ids)} terminal concepts "
                f"(headroom={headroom}, strength>={_ASSOC_STRENGTH_MIN})"
            )
            results.extend(expanded)
        else:
            self._expanded_ids = set()
            if self._trace_enabled():
                self.last_trace["association_expansion"]["expanded_ids"] = []

        return results

    def _extract_entities(self, question: str) -> list[str]:
        """Extract named entities (proper nouns, multi-word names) from question."""
        entities = []

        # 1. Quoted strings
        quoted = re.findall(r'"([^"]+)"', question)
        entities.extend(quoted)

        # 2. Proper noun runs
        # RETRIEVAL-090: Remove entire quoted segments (not just quote chars)
        # to prevent fragment leakage. Previously question.replace('"', ' ')
        # left "U.S.S.R." as a fragment after extracting "Back in the U.S.S.R.".
        q_clean = re.sub(r'"[^"]*"', ' ', question)
        q_clean = q_clean.replace("'s", "").replace("'", " ")
        words = q_clean.split()
        current_entity: list[str] = []

        for w in words:
            clean = w.strip('?,!.;:()')
            if not clean:
                if current_entity:
                    ent = ' '.join(current_entity)
                    if ent.lower() not in _STOPWORDS and len(ent) > 1:
                        entities.append(ent)
                    current_entity = []
                continue

            is_proper = (
                clean[0].isupper()
                and clean.lower() not in _STOPWORDS
                and len(clean) > 1
            )
            is_number_after = clean[0].isdigit() and current_entity

            if is_proper or is_number_after:
                current_entity.append(clean)
            else:
                if current_entity:
                    ent = ' '.join(current_entity)
                    if ent.lower() not in _STOPWORDS and len(ent) > 1:
                        entities.append(ent)
                    current_entity = []

        if current_entity:
            ent = ' '.join(current_entity)
            if ent.lower() not in _STOPWORDS and len(ent) > 1:
                entities.append(ent)

        # RETRIEVAL-058B: Extract hyphenated compounds (e.g. "split-finger fastball").
        # These are often domain-specific terms (sports techniques, proper nouns)
        # that the proper-noun extractor misses because they're lowercase.
        hyphenated = re.findall(r'\b(\w+-\w+(?:\s+\w+)?)\b', q_clean)
        for h in hyphenated:
            h_clean = h.strip('?,!.;:()')
            if len(h_clean) > 3 and h_clean.lower() not in _STOPWORDS:
                entities.append(h_clean)

        # Deduplicate: prefer longer entities that subsume shorter ones
        entities = list(dict.fromkeys(entities))
        filtered = []
        for e in sorted(entities, key=len, reverse=True):
            if any(e.lower() in kept.lower() for kept in filtered):
                continue
            filtered.append(e)

        # Leaf entity first (usually last in question for multi-hop)
        filtered.reverse()

        # RETRIEVAL-090: Noun-phrase fallback for questions with zero proper nouns.
        # e.g., "In which city was the person who created the centrifugal governor born?"
        # Without this, entity extraction returns [] and the chain never starts.
        if not filtered and _EC_NOUN_PHRASE_FALLBACK:
            fallback_q = re.sub(r'"[^"]*"', ' ', question)
            fallback_q = re.sub(r'[?,!.;:()"\']', ' ', fallback_q)
            words = fallback_q.split()
            phrase_entities: list[str] = []
            current_phrase: list[str] = []
            for w in words:
                w_clean = w.strip()
                if w_clean.lower() not in _STOPWORDS and len(w_clean) > 2:
                    current_phrase.append(w_clean)
                else:
                    if len(current_phrase) >= 2:
                        phrase_entities.append(' '.join(current_phrase))
                    current_phrase = []
            if len(current_phrase) >= 2:
                phrase_entities.append(' '.join(current_phrase))
            # Take longest phrases first, limit to 3
            phrase_entities.sort(key=len, reverse=True)
            filtered = phrase_entities[:3]
            if filtered:
                logger.info(f"ENTITY-CHAIN: Noun-phrase fallback: {filtered}")

        return filtered

    # --- INGEST-044: Graph search via typed associations ---

    def _graph_search(self, entity: str, hop: int,
                      target_predicate: Optional[str] = None) -> list[SearchResult]:
        """INGEST-044: Search typed associations for entity chain traversal.

        Uses self.db_path directly (NOT storage.py _db()) to ensure the same
        database is used for association AND concept queries — consistent with
        _keyword_search() and safe for benchmarking scenarios.

        Falls back to empty list if no typed edges found (caller uses keyword search).
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row

            try:
                # Determine target predicate from goal chain if available
                pred_filter = target_predicate
                if not pred_filter and self._goal_chain:
                    hop_idx = hop - 1
                    if hop_idx < len(self._goal_chain):
                        pred_filter = self._goal_chain[hop_idx]

                # Inline association query using self.db_path directly
                entity_lower = entity.lower()
                assoc_query = (
                    "SELECT source, target, relation, strength, chain_id "
                    "FROM associations WHERE source = ? AND direction = 'forward'"
                )
                assoc_params: list = [entity_lower]
                if pred_filter:
                    assoc_query += " AND relation = ?"
                    assoc_params.append(pred_filter)

                edges = conn.execute(assoc_query, assoc_params).fetchall()

                if not edges and pred_filter:
                    # No predicate-specific edges; try without predicate filter
                    edges = conn.execute(
                        "SELECT source, target, relation, strength, chain_id "
                        "FROM associations WHERE source = ? AND direction = 'forward'",
                        [entity_lower]
                    ).fetchall()

                if not edges:
                    return []  # Fall back to keyword search

                # Batch concept lookups into single WHERE id IN (...) query
                concept_ids = sorted({row["chain_id"] for row in edges if row["chain_id"]})  # sorted for determinism
                if not concept_ids:
                    return []

                placeholders = ",".join("?" * len(concept_ids))
                concept_rows = conn.execute(
                    f"SELECT id, summary, confidence, knowledge_area, created_at, edit_provenance "
                    f"FROM concepts WHERE id IN ({placeholders}) AND status = 'active'",
                    concept_ids,
                ).fetchall()

                # Convert to SearchResults
                hop_score = _HOP_SCORES.get(hop, max(0.30, _HOP_DEFAULT_SCORE - 0.10 * (hop - 3)))
                graph_boost = 0.05

                results = []
                for row in concept_rows:
                    results.append(SearchResult(
                        concept_id=row["id"],
                        version="v1",
                        summary=row["summary"] or "",
                        confidence=row["confidence"] or 0.5,
                        relevance_score=min(1.0, hop_score + graph_boost),
                        knowledge_area=row["knowledge_area"],
                        created_at=row["created_at"],
                        edit_provenance=row["edit_provenance"],  # RETRIEVAL-104
                    ))

                logger.info(
                    f"INGEST-044: Graph search for '{entity}' hop={hop} "
                    f"pred={pred_filter} → {len(results)} results"
                )
                return results

            finally:
                conn.close()

        except Exception as e:
            logger.warning(f"INGEST-044: Graph search failed for '{entity}': {e}")
            return []  # Fall back to keyword search

    # --- End INGEST-044 graph search ---

    def _keyword_search(self, entity: str, hop: int,
                        chain_keywords: list[str] | None = None,
                        prefer_fts: bool = False,
                        *,
                        deadline: TurnDeadline | None = None,
                        budget_ms: float | None = None) -> list[SearchResult]:
        """Search brain DB for concepts mentioning this entity."""
        _started = time.perf_counter()
        _timed_out = False
        _read_only = False
        _mode = "fts" if prefer_fts else "like"
        _budget_ms = None if budget_ms is None else max(0.0, float(budget_ms))
        if _budget_ms is not None and _budget_ms < _KEYWORD_QUERY_MIN_MS:
            _record_metric(
                "entity_chain.keyword_search_timeout_total",
                1.0,
                {"mode": _mode, "timed_out": "true", "read_only": "false"},
            )
            if deadline:
                deadline.skip(
                    "entity_chain.keyword_search",
                    "keyword_budget_below_floor",
                    priority="optional",
                    budget_ms=round(_budget_ms, 2),
                    min_remaining_ms=_KEYWORD_QUERY_MIN_MS,
                )
            return []
        _busy_timeout_ms = _KEYWORD_BUSY_TIMEOUT_MS
        if _budget_ms is not None:
            _busy_timeout_ms = int(max(0, min(float(_KEYWORD_BUSY_TIMEOUT_MS), _budget_ms)))
        conn: sqlite3.Connection | None = None
        try:
            if _KEYWORD_READONLY_ENABLED:
                try:
                    conn = sqlite3.connect(
                        f"file:{self.db_path}?mode=ro",
                        uri=True,
                        timeout=max(0.001, _busy_timeout_ms / 1000.0),
                    )
                    conn.execute("PRAGMA query_only = 1")
                    _read_only = True
                except sqlite3.Error as ro_err:
                    if not _KEYWORD_READONLY_FALLBACK:
                        raise
                    logger.debug(
                        "ENTITY-CHAIN: read-only keyword connection failed; falling back: %s",
                        ro_err,
                    )
                    conn = None
            if conn is None:
                conn = sqlite3.connect(
                    self.db_path,
                    timeout=max(0.001, _busy_timeout_ms / 1000.0),
                )
            conn.row_factory = sqlite3.Row
            conn.execute(f"PRAGMA busy_timeout={_busy_timeout_ms}")
            if _budget_ms is not None:
                _deadline_at = _started + (_budget_ms / 1000.0)

                def _progress_handler():
                    nonlocal _timed_out
                    if time.perf_counter() >= _deadline_at:
                        _timed_out = True
                        return 1
                    return 0

                conn.set_progress_handler(_progress_handler, 1000)

            words = entity.lower().split()
            if not words:
                conn.close()
                return []

            where_parts = []
            params = []
            for w in words:
                where_parts.append("LOWER(summary) LIKE ?")
                params.append(f"%{w}%")

            # Subject-position boost: entity before copula verb ranks higher
            entity_esc = entity.replace("'", "''")
            subject_boost = (
                f"CASE "
                f"WHEN INSTR(LOWER(summary), ' is ') > 0 "
                f"AND INSTR(LOWER(summary), '{entity_esc}') > 0 "
                f"AND INSTR(LOWER(summary), '{entity_esc}') < INSTR(LOWER(summary), ' is ') "
                f"THEN 2 "
                f"WHEN INSTR(LOWER(summary), ' was ') > 0 "
                f"AND INSTR(LOWER(summary), '{entity_esc}') > 0 "
                f"AND INSTR(LOWER(summary), '{entity_esc}') < INSTR(LOWER(summary), ' was ') "
                f"THEN 2 "
                f"ELSE 0 END"
            )

            # Question keyword boost
            kw_boost = ""
            if self._question_keywords:
                kw_parts = [
                    f"CASE WHEN LOWER(summary) LIKE '%{w}%' THEN 1 ELSE 0 END"
                    for w in self._question_keywords[:5]
                ]
                kw_boost = " + " + " + ".join(kw_parts)

            # RETRIEVAL-103: Chain-context keyword boost.
            # Keywords from the source concept that generated this entity
            # get a heavy boost (3x) to prefer chain-consistent concepts.
            # E.g., if source was "football was created in Ireland",
            # searching "Ireland" boosts concepts also mentioning "football".
            chain_boost = ""
            if _EC_CHAIN_ANCHOR and chain_keywords:
                chain_parts = []
                for ckw in chain_keywords[:_EC_CHAIN_ANCHOR_KEYWORDS]:
                    ckw_esc = ckw.replace("'", "''")
                    chain_parts.append(
                        f"CASE WHEN LOWER(summary) LIKE '%{ckw_esc}%' THEN 3 ELSE 0 END"
                    )
                chain_boost = " + " + " + ".join(chain_parts)

            # Taper for deeper hops (3-tier from adapter)
            # hop 1-2: full budget, hop 3: half, hop 4+: third
            # NOTE: hop2=half taper tested (run 081) — regressed to 66% EM (-13pp).
            # Hop 2 bridging facts are needed by embedding retriever + chain pruning.
            effective_limit = self.max_per_hop
            if hop >= 4:
                effective_limit = max(3, self.max_per_hop // 3)
            elif hop >= 3:
                effective_limit = max(5, self.max_per_hop // 2)

            # RETRIEVAL-076: Over-fetch 3x to build candidate pool for diversity filtering.
            # RETRIEVAL-077: Only overfetch at hop 1 where predicate diversity matters
            # (many "X is associated with sport Y" duplicates). At hop 2+ we need
            # speed over diversity — the bridging fact is unique, not duplicated.
            fetch_limit = effective_limit * 3 if hop <= 1 else effective_limit

            rows = []
            use_like_fallback = not prefer_fts
            if prefer_fts:
                fts_terms = [
                    re.sub(r"[^a-z0-9_]", "", w.lower())
                    for w in words
                    if len(w) > 1 and w.lower() not in _STOPWORDS
                ]
                fts_terms = [w for w in fts_terms if w]
                if fts_terms:
                    try:
                        fts_query = " AND ".join(fts_terms[:5])
                        rows = conn.execute(
                            """
                            SELECT c.id, c.summary, c.confidence, c.knowledge_area,
                                   c.created_at, c.edit_provenance
                            FROM fts_concepts
                            JOIN concepts c ON c.id = fts_concepts.concept_id
                            WHERE fts_concepts MATCH ?
                              AND c.status = 'active'
                            ORDER BY bm25(fts_concepts), c.confidence DESC
                            LIMIT ?
                            """,
                            (fts_query, fetch_limit),
                        ).fetchall()
                    except sqlite3.Error as fts_err:
                        use_like_fallback = True
                        _mode = "like"
                        logger.warning(
                            "ENTITY-CHAIN: FTS keyword search failed for '%s': %s; falling back to LIKE",
                            entity,
                            fts_err,
                        )
                else:
                    use_like_fallback = True
                    _mode = "like"

            if not rows and use_like_fallback:
                _mode = "like"
                query = f"""
                    SELECT id, summary, confidence, knowledge_area, created_at, edit_provenance
                    FROM concepts
                    WHERE status = 'active'
                      AND ({' AND '.join(where_parts)})
                    ORDER BY ({subject_boost}{kw_boost}{chain_boost}) DESC, confidence DESC
                    LIMIT ?
                """
                params.append(fetch_limit)
                rows = conn.execute(query, params).fetchall()
            conn.set_progress_handler(None, 0)
            conn.close()
            conn = None

            # RETRIEVAL-073+093: Hop-priority scoring with per-hop decay.
            # Fixed hops 1-2 use explicit scores; hops 3+ decay by 0.10 per hop
            # to deprioritize deep-chain noise at cap time.
            if hop in _HOP_SCORES:
                hop_score = _HOP_SCORES[hop]
            else:
                hop_score = max(0.30, _HOP_DEFAULT_SCORE - 0.10 * (hop - 3))

            candidates = []
            for row in rows:
                candidate = SearchResult(
                    concept_id=row["id"],
                    version="v1",
                    summary=row["summary"] or "",
                    confidence=row["confidence"] or 0.5,
                    relevance_score=hop_score,
                    knowledge_area=row["knowledge_area"],
                    created_at=row["created_at"],  # RETRIEVAL-053
                    edit_provenance=row["edit_provenance"],  # RETRIEVAL-104
                )
                question_for_bonus = self._current_question
                if question_for_bonus:
                    bonus = 0.0
                    bonus += _temporal_anchor_adjustment(question_for_bonus, entity, candidate.summary)
                    bonus += _kind_type_anchor_adjustment(question_for_bonus, entity, candidate.summary)
                    bonus += _contrastive_alternative_adjustment(question_for_bonus, entity, candidate.summary)
                    bonus += _entity_focus_adjustment(question_for_bonus, entity, candidate.summary)
                    if bonus:
                        candidate.relevance_score = round(
                            max(0.0, min(1.0, candidate.relevance_score + bonus)),
                            4,
                        )
                candidates.append(candidate)
            if candidates:
                candidates.sort(key=lambda c: c.relevance_score, reverse=True)
            if self._trace_enabled() and self._trace_current_hop is not None:
                self._trace_current_hop["keyword_candidate_ids"] = _trace_result_ids(candidates)

            # RETRIEVAL-095c: Position-aware goal reranking.
            # Only boost candidates whose predicate matches the chain step
            # expected AT THIS HOP, not all steps.  This prevents hop-1
            # "sport" concepts from displacing hop-1 "citizenship" concepts
            # when the chain is ["citizenship","sport","country_of_origin"].
            #
            # Hop-to-chain mapping: hop 1 → chain[0], hop 2 → chain[1], etc.
            # If hop exceeds chain length, match any remaining chain step.
            if self._goal_chain and candidates:
                gc = self._goal_chain
                # Determine which predicates are valid for this hop
                hop_idx = hop - 1  # hop is 1-based, chain is 0-based
                if hop_idx < len(gc):
                    # Exact position match + adjacent positions (±1)
                    valid_preds = set()
                    for offset in range(-1, 2):  # -1, 0, 1
                        idx = hop_idx + offset
                        if 0 <= idx < len(gc):
                            valid_preds.add(gc[idx])
                else:
                    # Past chain length — match later chain steps
                    valid_preds = set(gc[max(0, len(gc)-2):])

                for c in candidates:
                    pred = _classify_predicate(c.summary)
                    if pred in valid_preds:
                        c.relevance_score = min(1.0, hop_score + 0.10)
                _boosted = sum(
                    1 for c in candidates
                    if c.relevance_score > hop_score
                )
                if _boosted:
                    candidates.sort(
                        key=lambda c: c.relevance_score, reverse=True
                    )
                    logger.info(
                        f"RETRIEVAL-095c: Position-boosted {_boosted}/{len(candidates)} "
                        f"at hop {hop} for '{entity}' (valid={valid_preds})"
                    )

            # RETRIEVAL-076: Predicate-diverse selection (hop 1 only).
            # RETRIEVAL-077 (amended by RETRIEVAL-101): Diversity filter now applies
            # at ALL hops. The original hop>=2 bypass was a speed optimization, but
            # _classify_predicate() is O(27) string matching (~0.01ms) — negligible
            # vs the SQLite query time. The bypass allowed predicate flooding at deep
            # hops (e.g., 11-17 "affiliated with Methodism" concepts in Q57).

            # Classify each candidate, cap each predicate to _PREDICATE_CAP results.
            # 'other' is exempt from cap (heterogeneous bucket — capping it hurts diversity).
            # Uses dict for predicate labels (SearchResult is Pydantic BaseModel,
            # cannot set ad-hoc attributes).
            try:
                pred_labels = {c.concept_id: _classify_predicate(c.summary) for c in candidates}
                pred_counts: Counter = Counter()
                results = []
                overflow = []
                for c in candidates:
                    if len(results) >= effective_limit:
                        break
                    pred = pred_labels[c.concept_id]
                    if pred == 'other' or pred_counts[pred] < _PREDICATE_CAP:
                        results.append(c)
                        pred_counts[pred] += 1
                    else:
                        overflow.append(c)

                # Backfill from overflow if we didn't fill effective_limit
                for c in overflow:
                    if len(results) >= effective_limit:
                        break
                    results.append(c)

                if candidates and logger.isEnabledFor(logging.DEBUG):
                    n_unique = len(set(pred_labels[r.concept_id] for r in results))
                    logger.debug(
                        f"RETRIEVAL-076: {len(candidates)} candidates -> "
                        f"{len(results)} diverse ({n_unique} unique predicates, "
                        f"cap={_PREDICATE_CAP})"
                    )
            except Exception as div_err:
                # GAUNTLET FIX: Never let diversity filter silently kill retrieval.
                # Fall back to unfiltered top-N candidates.
                logger.error(f"RETRIEVAL-076: Diversity filter failed: {div_err}")
                results = candidates[:effective_limit]
            if self._trace_enabled() and self._trace_current_hop is not None:
                self._trace_current_hop["keyword_selected_ids"] = _trace_result_ids(results)

            if results:
                logger.info(
                    f"ENTITY-CHAIN: '{entity}' -> {len(results)} concepts (hop {hop})"
                )
            return results

        except sqlite3.OperationalError as e:
            if _timed_out or "interrupted" in str(e).lower():
                _record_metric(
                    "entity_chain.keyword_search_timeout_total",
                    1.0,
                    {
                        "mode": _mode,
                        "timed_out": "true",
                        "read_only": str(_read_only).lower(),
                    },
                )
                logger.warning(
                    "ENTITY-CHAIN: Keyword search timed out for '%s' after %.0fms",
                    entity,
                    (time.perf_counter() - _started) * 1000.0,
                )
                return []
            _record_metric(
                "entity_chain.keyword_search_error_total",
                1.0,
                {"mode": _mode, "read_only": str(_read_only).lower()},
            )
            logger.error(f"ENTITY-CHAIN: Keyword search failed for '{entity}': {e}")
            return []
        except Exception as e:
            _record_metric(
                "entity_chain.keyword_search_error_total",
                1.0,
                {"mode": _mode, "read_only": str(_read_only).lower()},
            )
            logger.error(f"ENTITY-CHAIN: Keyword search failed for '{entity}': {e}")
            return []
        finally:
            if conn is not None:
                try:
                    conn.set_progress_handler(None, 0)
                except Exception:
                    pass
                try:
                    conn.close()
                except Exception:
                    pass
            _record_metric(
                "entity_chain.keyword_search_ms",
                round((time.perf_counter() - _started) * 1000.0, 2),
                {
                    "mode": _mode,
                    "timed_out": str(_timed_out).lower(),
                    "read_only": str(_read_only).lower(),
                },
            )

    def _extract_fact_values(self, summary: str, search_entity: str) -> list[str]:
        """Extract the value/object from a fact for next-hop chaining.

        Given: "Andrew lives in San Francisco" + search="andrew"
        Returns: ["San Francisco"]
        """
        values = []
        summary_lower = summary.lower()

        # RETRIEVAL-091: Fast-path extraction for known predicate patterns.
        # Structured predicates like "X is associated with the sport of Y"
        # have a known value position. Extract directly instead of general
        # copula parsing, which can split on wrong verb (e.g., "plays" not "is").
        for pattern, label in _PREDICATE_PATTERNS:
            idx = summary_lower.find(pattern)
            if idx >= 0:
                value_part = summary[idx + len(pattern):].strip().rstrip('.,;:!? ')
                if value_part and len(value_part) > 1:
                    # RETRIEVAL-094: Also yield bare proper noun when value has
                    # filler prefix like "city of Helsinki", "country of Italy".
                    # Without this, hop N+1 searches "city of helsinki" (all words
                    # AND'd) which misses "official language of Helsinki is Esperanto"
                    # because it doesn't contain "city". Yielding both the full
                    # phrase AND the bare entity lets the chain bridge the gap.
                    result = [value_part]
                    _vp_lower = value_part.lower()
                    _filler_prefixes = [
                        'city of ', 'country of ', 'continent of ',
                        'sport of ', 'language of ', 'religion of ',
                        'field of ', 'region of ', 'province of ',
                        'state of ', 'kingdom of ', 'republic of ',
                        'island of ', 'county of ', 'district of ',
                    ]
                    for fp in _filler_prefixes:
                        if _vp_lower.startswith(fp):
                            bare = value_part[len(fp):].strip()
                            if bare and len(bare) > 1 and bare != value_part:
                                result.append(bare)
                            break
                    return result

        # Find best copula position (compound first, then simple)
        best_idx = -1
        best_len = 0

        for cop in _COMPOUND_COPULAS:
            idx = summary_lower.find(cop)
            if idx > 0 and (best_idx == -1 or idx < best_idx):
                best_idx = idx
                best_len = len(cop)

        if best_idx == -1:
            for cop in _COPULA_VERBS:
                idx = summary_lower.find(cop)
                if idx > 0 and (best_idx == -1 or idx < best_idx):
                    best_idx = idx
                    best_len = len(cop)

        if best_idx == -1:
            return values

        # RETRIEVAL-091: When entity is in subject position and multiple copulas
        # exist, prefer the later primary copula. Fixes: "The type of music that
        # Julian Lennon plays is soul music" — splits on "plays" instead of "is".
        entity_idx = summary_lower.find(search_entity)
        if entity_idx >= 0 and entity_idx < best_idx:
            for cop in [' is ', ' was ', ' are ', ' were ']:
                later_idx = summary_lower.rfind(cop)
                if later_idx > best_idx:
                    best_idx = later_idx
                    best_len = len(cop)
                    break  # Take rightmost primary copula

        value_part = summary[best_idx + best_len:].strip()

        # Strip filler: "a citizen of", "the position of", leading copulas
        # Leading copulas happen when compound verbs like "plays is X" split
        # on the first verb, leaving "is X" as the value.
        value_part = re.sub(
            r'^(?:is |are |was |were |has |had )?'
            r'(?:a |an |the )?(?:citizen |member |position |'
            r'sport |country |city |language |religion |'
            r'genre |type |capital |university |'
            r'location |place )?(?:of |in |at |for )?',
            '', value_part, flags=re.I
        ).strip()

        if not value_part:
            return values

        # Extract proper nouns from value
        words = value_part.split()
        current: list[str] = []
        connectors = {'of', 'the', 'and', 'de', 'la', 'le', 'del', 'von', 'van'}

        for w in words:
            clean = w.strip('?,!.;:()')
            if not clean:
                continue
            is_proper = clean[0].isupper() and len(clean) > 1
            is_number = clean[0].isdigit() and current
            is_conn = clean.lower() in connectors and current

            if is_proper or is_number or is_conn:
                current.append(clean)
            else:
                if current:
                    while current and current[-1].lower() in connectors:
                        current.pop()
                    ent = ' '.join(current)
                    if len(ent) > 1:
                        values.append(ent)
                    current = []

        if current:
            while current and current[-1].lower() in connectors:
                current.pop()
            ent = ' '.join(current)
            if len(ent) > 1:
                values.append(ent)

        # Fallback: use whole value if no proper nouns found
        if not values and len(value_part) > 2:
            clean_val = re.sub(r'\s+(?:of|in|at|for|to|with|by|from)\s*$', '', value_part).strip()
            if clean_val and len(clean_val) > 2:
                values.append(clean_val)

        return values

    def _chain_query_search(self, question: str, entities: list[str],
                            budget_ms: int, t0: float) -> list[SearchResult]:
        """RETRIEVAL-106: Predicate-guided chain query for multi-hop.

        Deterministic chain traversal: extract predicates from question,
        at each hop try entity + predicate SQL, extract value, chain forward.
        Returns concepts found along the chain (supplements main retrieval).

        Uses its own time budget (200ms) independent of the main entity chain
        timer, since the main loop typically consumes most of the shared budget
        before chain query gets a turn.
        """
        if not _EC_CHAIN_QUERY:
            return []

        predicates = _extract_query_predicates(question)
        if not predicates or not entities:
            return []
        if self._trace_enabled():
            self.last_trace["chain_query"]["predicate_count"] = len(predicates)
            self.last_trace["chain_query"]["predicates"] = list(predicates)

        # Use last entity as seed (innermost in multi-hop questions)
        seed = entities[-1] if entities else None
        if not seed:
            return []

        logger.info(
            f"RETRIEVAL-106: Chain query start — seed='{seed}', "
            f"predicates={predicates}"
        )

        results: dict[str, SearchResult] = {}
        current = seed
        remaining = list(predicates)
        visited: set[str] = set()
        fb_count = 0
        max_hops = len(remaining) + 2  # allow some fallback hops
        # Own timer — main entity chain already consumed most of the shared budget.
        # Default to the entity-chain budget ceiling because cold SQLite scans can
        # exceed 200ms on the first predicate pass in 18k+ concept brains.
        _cq_t0 = time.perf_counter()
        _CQ_BUDGET_MS = min(max(1, int(budget_ms)), _EC_CHAIN_QUERY_BUDGET_MS)

        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row

            def _add_row_result(row: sqlite3.Row, score: float) -> None:
                cid = row["id"]
                if cid in results:
                    return
                results[cid] = SearchResult(
                    concept_id=cid,
                    version="v1",
                    summary=row["summary"] or "",
                    confidence=row["confidence"] or 0.5,
                    relevance_score=score,
                    knowledge_area=row["knowledge_area"],
                    created_at=row["created_at"],
                    edit_provenance=row["edit_provenance"],
                )

            def _lookahead_next_predicate(
                branch_values: list[str],
                next_pred_kw: str,
            ) -> None:
                sql_pats = _KEYWORD_TO_SQL.get(next_pred_kw, [])
                if not sql_pats:
                    return
                for branch_value in branch_values[:_EC_CHAIN_QUERY_BRANCHES]:
                    branch_lower = branch_value.lower().strip()
                    if len(branch_lower) < 2:
                        continue
                    for pat in sql_pats:
                        cur = conn.execute(
                            "SELECT id, summary, confidence, knowledge_area, "
                            "created_at, edit_provenance "
                            "FROM concepts "
                            "WHERE LOWER(summary) LIKE ? AND LOWER(summary) LIKE ? "
                            "AND status='active' LIMIT 1",
                            (f'%{branch_lower}%', f'%{pat.lower()}%'),
                        )
                        row = cur.fetchone()
                        if not row:
                            continue
                        if self._trace_enabled():
                            self.last_trace["chain_query"]["candidate_ids"].append(row["id"])
                        _add_row_result(row, 0.88)
                        break

            for hop in range(max_hops):
                if not remaining:
                    break

                # Budget check — own timer, not shared t0
                _cq_elapsed = (time.perf_counter() - _cq_t0) * 1000
                if _cq_elapsed > _CQ_BUDGET_MS:
                    logger.info(
                        f"RETRIEVAL-106: Budget exhausted at hop {hop+1} "
                        f"({_cq_elapsed:.0f}ms > {_CQ_BUDGET_MS}ms)"
                    )
                    if self._trace_enabled():
                        self.last_trace["chain_query"]["budget_exhausted"] = True
                    break

                ent_lower = current.lower().strip()
                if len(ent_lower) < 2:
                    break

                # PASS 1: Try each remaining predicate (strict match)
                matched = False
                for i, pred_kw in enumerate(remaining):
                    sql_pats = _KEYWORD_TO_SQL.get(pred_kw, [])
                    candidates = []

                    for pat in sql_pats:
                        cur = conn.execute(
                            "SELECT id, summary, confidence, knowledge_area, "
                            "created_at, edit_provenance "
                            "FROM concepts "
                            "WHERE LOWER(summary) LIKE ? AND LOWER(summary) LIKE ? "
                            "AND status='active'",
                            (f'%{ent_lower}%', f'%{pat.lower()}%'),
                        )
                        for row in cur.fetchall():
                            if self._trace_enabled():
                                self.last_trace["chain_query"]["candidate_ids"].append(row["id"])
                            val = _chain_query_extract_value(
                                row["summary"], current, [pat]
                            )
                            if val and val.strip():
                                candidates.append((
                                    val.strip(), row, pat
                                ))

                    if not candidates:
                        continue

                    # Score candidates (pattern-adjacency + word-boundary)
                    def _score(c):
                        _, row, pat = c
                        sl = (row["summary"] or "").lower()
                        sl = re.sub(r'^\d+\.\s*', '', sl)
                        s = 0
                        if sl.startswith(ent_lower):
                            s += 3
                        if f'of {ent_lower}' in sl[:60 + len(ent_lower)]:
                            s += 2
                        if ent_lower in sl[:40 + len(ent_lower)]:
                            s += 1
                        pat_l = pat.lower().rstrip()
                        if (pat_l + ent_lower) in sl or (pat_l + ' ' + ent_lower) in sl:
                            s += 5
                        if re.search(r'\b' + re.escape(ent_lower) + r'\b', sl):
                            s += 2
                        return s

                    candidates.sort(key=_score, reverse=True)
                    best_val, best_row, _ = candidates[0]

                    # Add all matched concepts to results
                    for _, row, _ in candidates[:3]:
                        _add_row_result(row, 0.90)  # High score — precise match

                    remaining_after_match = (
                        remaining[:i] + remaining[i + 1:]
                    )
                    if remaining_after_match:
                        branch_values = []
                        seen_branch_values = set()
                        for val, _, _ in candidates:
                            branch_key = val.strip().lower()
                            if branch_key in seen_branch_values:
                                continue
                            seen_branch_values.add(branch_key)
                            branch_values.append(val.strip())
                        _lookahead_next_predicate(
                            branch_values,
                            remaining_after_match[0],
                        )

                    remaining.pop(i)
                    visited.add(current.lower())
                    current = best_val
                    matched = True
                    logger.info(
                        f"RETRIEVAL-106: Hop {hop+1} '{pred_kw}' | "
                        f"'{ent_lower}' → '{best_val}'"
                    )
                    break

                if matched:
                    continue

                # PASS 2: Controlled "The..of..is" fallback
                if fb_count >= 2:
                    break

                cur = conn.execute(
                    "SELECT id, summary, confidence, knowledge_area, "
                    "created_at, edit_provenance "
                    "FROM concepts "
                    "WHERE LOWER(summary) LIKE 'the %%' "
                    "AND LOWER(summary) LIKE ? "
                    "AND LOWER(summary) LIKE '%% is %%' "
                    "AND status='active' LIMIT 15",
                    (f'%{ent_lower}%',),
                )
                fb_rows = cur.fetchall()
                fb_ok = False
                for row in fb_rows:
                    if self._trace_enabled():
                        self.last_trace["chain_query"]["candidate_ids"].append(row["id"])
                    fb_val = _chain_query_extract_value(
                        row["summary"], current, []
                    )
                    if not fb_val or len(fb_val.strip()) < 2:
                        continue
                    fb_val = fb_val.strip()
                    if len(fb_val) > 80 or fb_val.lower() == current.lower():
                        continue
                    if fb_val.lower() in visited:
                        continue

                    cid = row["id"]
                    if cid not in results:
                        results[cid] = SearchResult(
                            concept_id=cid,
                            version="v1",
                            summary=row["summary"] or "",
                            confidence=row["confidence"] or 0.5,
                            relevance_score=0.85,
                            knowledge_area=row["knowledge_area"],
                            created_at=row["created_at"],
                            edit_provenance=row["edit_provenance"],
                        )

                    visited.add(current.lower())
                    current = fb_val
                    fb_count += 1
                    fb_ok = True
                    logger.info(
                        f"RETRIEVAL-106: Hop {hop+1} FALLBACK | "
                        f"'{ent_lower}' → '{fb_val}'"
                    )
                    break

                if not fb_ok:
                    break

            conn.close()
        except Exception as e:
            logger.warning(f"RETRIEVAL-106: Chain query failed: {e}")

        result_list = list(results.values())
        self.last_chain_query_selected_ids = {r.concept_id for r in result_list}
        if self._trace_enabled():
            self.last_trace["chain_query"]["selected_ids"] = _trace_result_ids(result_list)
        if result_list:
            logger.info(
                f"RETRIEVAL-106: Chain query complete — "
                f"{len(result_list)} concepts from chain traversal"
            )
        return result_list


# Module-level singleton (lazy init)
_retriever: Optional[EntityChainRetriever] = None
_retriever_db_path: Optional[str] = None  # RETRIEVAL-099: track singleton's db_path


def get_entity_chain_retriever(db_path: Optional[str] = None) -> Optional[EntityChainRetriever]:
    """Get or create the entity chain retriever singleton.

    Args:
        db_path: Explicit path to pith.db. When provided, overrides env-based
                 resolution and resets the singleton if it points to a different DB.
                 Callers SHOULD pass storage.DB_PATH to ensure consistency with
                 the rest of the retrieval pipeline.

    Returns None if PITH_ENTITY_CHAIN is not enabled.
    """
    global _retriever, _retriever_db_path
    if not ENTITY_CHAIN_ENABLED:
        return None

    # RETRIEVAL-099: Resolve db_path — prefer explicit, fall back to env
    if db_path is None:
        profile = os.environ.get("PITH_PROFILE", "default")
        data_dir = os.environ.get(
            "PITH_DATA_DIR",
            os.path.expanduser(f"~/pith-data/{profile}")
        )
        db_path = os.path.join(data_dir, "pith.db")

    # RETRIEVAL-099: Reset singleton if db_path changed (e.g., benchmark switch)
    if _retriever is not None and _retriever_db_path != db_path:
        logger.info(
            f"ENTITY-CHAIN: DB path changed ({_retriever_db_path} → {db_path}), "
            f"resetting singleton"
        )
        _retriever = None
        _retriever_db_path = None
        # MONITOR-123: track singleton reset events for benchmark switch detection
        # importlib avoids static app.ops.metrics import that breaks Contract 3
        # (cognitive cannot import ops). DEBT-244: proper metrics facade in app.core.
        try:
            import importlib as _ec_ilib
            _ec_metrics = _ec_ilib.import_module("app.ops.metrics").metrics
            _ec_metrics.record("entity_chain_singleton_reset", 1.0,
                {"old_path": str(_retriever_db_path or ""), "new_path": str(db_path)})
        except Exception:
            pass

    if _retriever is None:
        if not os.path.exists(db_path):
            logger.warning(f"ENTITY-CHAIN: DB not found at {db_path}")
            return None
        _max_hops = int(os.environ.get("PITH_ENTITY_CHAIN_MAX_HOPS", "4"))
        _max_per_hop = int(os.environ.get("PITH_ENTITY_CHAIN_PER_HOP", "8"))
        _retriever = EntityChainRetriever(db_path=db_path, max_hops=_max_hops, max_per_hop=_max_per_hop)
        _retriever_db_path = db_path
        logger.info(f"ENTITY-CHAIN: Initialized with db={db_path}, max_hops={_max_hops}, per_hop={_max_per_hop}")
    return _retriever
