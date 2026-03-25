"""Session middleware — lifecycle management + present moment orientation.

Phase 1A D7: Implements orientation (where-been/where-am/where-going) and session bookends.
Phase 1B P1.1: Session persistence to SQLite, startup recovery, stub retirement.

Key design: session_start loads concepts ONCE and passes to both introspect
and orient — single disk scan, no redundant reads.

Stub surface area (2 stubs remaining for Phase 1B+ retirement):
  - contradictions_detected (needs contradiction graph wiring — contradiction.py runs but output not connected)
  - corrections_made (needs error tracking wiring — correction.py runs but output not connected)

Retired stubs:
  - open_threads → self._compute_open_threads() calls app.threads [Phase 1B+]
  - next_recommended_actions → populated via actions logic [Phase 1B+]
  - strategic_priorities → populated via priority extraction [Phase 1B+]
  - session_count_in_window → count_sessions(since=cutoff) [P1.1]
"""

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Optional

from app.constants import (
    FRESHNESS_EARLIER_TODAY_UPPER,
    FRESHNESS_HOURS_AGO_UPPER,
    FRESHNESS_JUST_NOW_MINS,
    FRESHNESS_MINUTES_AGO_UPPER,
    FRESHNESS_ONE_HOUR_UPPER,
    FRESHNESS_YESTERDAY_UPPER,
    GOV_EVENT_CCL_VIOLATIONS_DETECTED,
    GOV_EVENT_CIRCUIT_BREAKER_TRIPPED,
    GOV_EVENT_COMPACTION_REINJECTION,
    GOV_EVENT_CONTRADICTION_REVIEW,
    GOV_EVENT_CONVERSATION_TURN_COMPLETE,
    GOV_EVENT_GRAPH_CONTRADICTION_SIGNAL,
    GOV_EVENT_RESUME_CONTEXT_INJECTION,
    MINUTES_PER_HOUR,
)
from app.datetime_utils import _ensure_aware, _utc_now, _utc_now_iso
from app.models import (
    ActivatedConcept,
    ActiveDirectionality,
    ActiveUncertainty,
    AreaStrength,
    Concept,
    ConceptEvolution,
    ConceptEvolutionRecord,
    ConversationTurnRequest,
    ConversationTurnResponse,
    CuriosityFrontierItem,
    CurrentStateAssessment,
    EvolvedConcept,
    GoalSummary,
    LearnedConcept,
    PendingQuestionSummary,
    PresentMomentOrientation,
    RecentConceptSummary,
    RecentEvolutionSummary,
    SearchResult,
    SessionEndRequest,
    SessionInfo,
    SessionLearnRequest,
    SessionLearnResponse,
    SessionStartResponse,
)
from app.self_model import self_model_manager
from app.storage import (
    _get_connection,
    cleanup_expired_snapshots,
    count_associations,
    count_sessions,
    get_related_concepts,
    list_concepts,
    load_associations,
    load_concept,
    load_recent_concepts,
    load_resume_snapshot,
    load_session_velocity,
    recover_interrupted_sessions,
    save_concept,
    # Resume Context v1.1
    save_resume_snapshot,
    save_session,
    update_session,
)
from app.taxonomy import (  # DEBT-030/DEBT-108
    classify_knowledge_area,
    infer_knowledge_area,
    normalize_knowledge_area,
)

logger = logging.getLogger(__name__)


class _BudgetSkip(Exception):
    """Raised to skip a pipeline phase when budget is exhausted (Fix 5b)."""

    pass


# Time window → timedelta mapping
TIME_WINDOWS = {
    "1_day": timedelta(days=1),
    "7_days": timedelta(days=7),
    "30_days": timedelta(days=30),
    "all": timedelta(days=36500),  # ~100 years
}
DEFAULT_WINDOW = "7_days"

# --- Recency baseline constants (shared by S4.9 injection and S7 orientation) ---
# A-C8: Unified thresholds prevent mismatch between what orientation says
# and what retrieval surfaces. Change these together.
RECENCY_WINDOW_HOURS = 6
RECENCY_MIN_CONFIDENCE = 0.50
RECENCY_MAX_INJECT = 2
RECENCY_RELEVANCE_SCORE = 0.35  # A-C16: configurable, tuned post-deploy
QUARANTINE_RECENCY_EXEMPT_HOURS = 72  # RETRIEVAL-004: young quarantined concepts bypass maturity gate

# S7.1 Fix 2: Content quality filter patterns for orientation
ORIENTATION_EXCLUDE_PATTERNS = ("delete", "placeholder", "safe to remove")

# DEBT-027: Hoisted from conversation_turn method body (was re-parsed every call)
_SUPERSEDED_S4_MULTIPLIER = float(os.environ.get("SUPERSEDED_S4_MULTIPLIER", "0.15"))
_CONTRADICTED_S4_MULTIPLIER = float(os.environ.get("CONTRADICTED_S4_MULTIPLIER", "0.30"))

# --- ORIENTATION_V2: Module-level compiled regexes for orientation filtering ---
# Moved from method body per gauntlet finding 1.4 (avoid re-compilation per call)
import re as _re

_RESOLVED_PATTERNS = _re.compile(
    r"\b(RESOLVED|FIXED|SUPERSEDED|DEPRECATED|UNBLOCKED|COMPLETED|CLOSED)\b", _re.IGNORECASE
)

_COMMIT_RECORD_PATTERNS = _re.compile(
    r"(implemented and committed|committed as [a-f0-9]{7}|ALL \d+ FIXES IMPLEMENTED|"
    r"v\d+\.\d+ —.*IMPLEMENTED|SPEC v\d)",
    _re.IGNORECASE,
)

_BACKLOG_PATTERNS = _re.compile(r"BACKLOG[:\s]|EXECUTION CHECKPOINT|cleanup tasks|todo items", _re.IGNORECASE)

_IMPLEMENTATION_DETAIL_PATTERNS = _re.compile(
    r"\b(tokenizer|backslash|byte\.order|serializ|deserializ)\b", _re.IGNORECASE
)

_TEMPORAL_MEMORY_QUERY = _re.compile(
    r"\b(what did we|what have we|what was our|what were|"
    r"do you recall|do you remember|tell me about|"
    r"what happened in|anything from|what came up in|"
    r"what did i|what have i|recap of)\b",
    _re.IGNORECASE,
)


# --- COVERAGE-001: Module-level LLM client singleton for coverage validation ---
_coverage_llm_client = None


def _get_coverage_client():
    """Lazy-init OpenRouter client for coverage classification.

    Module-level function — called from SessionManager._classify_coverage_signals().
    Uses OpenRouter (not Anthropic) to avoid burning API credits for coverage.
    """
    global _coverage_llm_client
    if _coverage_llm_client is None:
        _or_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not _or_key:
            logger.warning("COVERAGE-001: OPENROUTER_API_KEY not set — coverage disabled")
            return None
        from openai import OpenAI as _OAI
        _coverage_llm_client = _OAI(
            base_url="https://openrouter.ai/api/v1", api_key=_or_key,
            timeout=2.0,  # Hard timeout: 2 seconds max
        )
    return _coverage_llm_client


def _is_not_resolved(concept: dict) -> bool:
    """Return True if concept summary does NOT contain resolved-state text."""
    return not _RESOLVED_PATTERNS.search(concept.get("summary", ""))


# ================================================================
# RETRIEVAL-066v2: LLM query decomposition for compositional questions
# ================================================================
# Ported from benchmarks/adapter/retrieval.py. Complements
# retrieval_multihop.py's regex decomposer with semantic compositionality
# detection via LLM. Fires AFTER RETRIEVAL-048 if coverage is still
# sparse, gated by PITH_QUERY_DECOMPOSITION=true.
# ================================================================

_DECOMP_CONFIDENCE_THRESHOLD = 0.7
_DECOMP_TRIGGER_LEVELS = frozenset(
    {"no_results", "no_strong_match", "incomplete", "sparse"}
)
_DECOMP_MIN_SLOTS_PER_SUBQUERY = 3


def _decompose_query_llm(question: str) -> list[str] | None:
    """Decompose a compositional question into independent sub-queries.

    RETRIEVAL-066v2: Uses cheap LLM call to detect whether the question
    requires combining facts from unrelated contexts, and if so, splits it.

    Returns None if not compositional. Returns list of 2-3 sub-queries if it is.
    """
    client = _get_coverage_client()
    if client is None:
        return None

    prompt = (
        f'Analyze this question about a user\'s personal history:\n\n'
        f'"{question}"\n\n'
        'Does answering this question require combining TWO OR MORE independent facts '
        'that would likely come from DIFFERENT conversations/contexts?\n\n'
        'Examples of compositional questions:\n'
        '- "How old will I be when Rachel gets married?" -> needs (1) wedding date, (2) user age\n'
        '- "How many years older is my grandma than me?" -> needs (1) grandma age, (2) user age\n'
        '- "Did I visit Paris before or after starting my new job?" -> needs (1) trip date, (2) job date\n\n'
        'Examples of NON-compositional questions:\n'
        '- "How many albums have I bought?" -> single topic\n'
        '- "What beer did you recommend?" -> single conversation recall\n\n'
        'If COMPOSITIONAL: Reply with JSON: {"confidence": 0.0-1.0, "sub_queries": ["q1", "q2"]}\n'
        'If NOT COMPOSITIONAL: Reply with just "NONE"\n\n'
        'Reply with ONLY JSON or "NONE". No explanation.'
    )

    try:
        resp = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )
        reply = resp.choices[0].message.content.strip()

        if reply.upper().startswith("NONE"):
            return None

        parsed = json.loads(reply) if reply.startswith("{") else None
        if parsed is None:
            return None

        if isinstance(parsed, dict):
            conf = float(parsed.get("confidence", 0))
            sub_queries = parsed.get("sub_queries", [])
            if conf < _DECOMP_CONFIDENCE_THRESHOLD:
                logger.info(
                    f"RETRIEVAL-066v2: Decomposition confidence {conf:.2f} below "
                    f"threshold {_DECOMP_CONFIDENCE_THRESHOLD}, skipping"
                )
                return None
            if isinstance(sub_queries, list) and len(sub_queries) >= 2:
                return [str(sq) for sq in sub_queries[:3]]
            return None

        # Backward compat: plain JSON array
        if isinstance(parsed, list) and len(parsed) >= 2:
            return [str(sq) for sq in parsed[:3]]
        return None

    except Exception as e:
        logger.warning(f"RETRIEVAL-066v2: Decomposition failed ({e}), skipping")
        return None


def _is_strategic(concept: dict) -> bool:
    """Return True if concept content is genuinely strategic, not an implementation detail.

    ORIENTATION_V2 Fix A1: Content-based filter for strategic orientation layer.
    Catches commit records, backlog entries, and implementation details that got
    misclassified as 'decision' or 'principle'.

    Gauntlet 3.2 fix: Trust explicit 'PRINCIPLE:' prefix — if the client
    deliberately labeled it, don't second-guess based on keywords alone.
    """
    summary = concept.get("summary", "")
    ctype = concept.get("concept_type", "observation")

    # Commit records are NOT strategic decisions
    if _COMMIT_RECORD_PATTERNS.search(summary):
        return False

    # Backlog/checkpoint entries are NOT strategic decisions
    if _BACKLOG_PATTERNS.search(summary):
        return False

    # Implementation details labeled as principles — but TRUST explicit PRINCIPLE: prefix
    # (gauntlet finding 3.2: don't over-filter legitimate principles about implementation)
    # Review fix: strip() before startswith() to handle leading whitespace
    if ctype == "principle" and _IMPLEMENTATION_DETAIL_PATTERNS.search(summary):
        stripped = summary.strip()
        if not stripped.startswith("PRINCIPLE:") and not stripped.startswith("[PRINCIPLE]"):
            return False

    return True


def _validate_concept_type(summary: str, claimed_type: str) -> str:
    """Validate and correct concept_type based on content signals.

    ORIENTATION_V2 Fix A4: Ingestion gate for content-type consistency.
    Applied at concept creation time to prevent future misclassification.

    TUNE-EXTRACTION: Also UPGRADES observations with structural markers.
    Client defaults to "observation" when uncertain — detect upgradable patterns.

    Gauntlet 3.2 fix: Trust explicit 'PRINCIPLE:' prefix.
    """
    # TUNE-EXTRACTION: Upgrade observations with structural signals.
    if claimed_type == "observation":
        stripped = summary.strip()
        s_lower = summary.lower()
        # Explicit prefix markers (highest confidence)
        if stripped.startswith("PRINCIPLE:") or stripped.startswith("[PRINCIPLE]"):
            return "principle"
        if stripped.startswith("HEURISTIC:") or stripped.startswith("[HEURISTIC]"):
            return "heuristic"
        if stripped.startswith("METHOD:") or stripped.startswith("[METHOD]"):
            return "method"
        if stripped.startswith("DECISION") and ":" in stripped[:12]:
            return "decision"
        if stripped.startswith("PATTERN:") or stripped.startswith("[PATTERN]"):
            return "pattern"
        if stripped.startswith("CONSTRAINT:") or stripped.startswith("[CONSTRAINT]"):
            return "constraint"
        # Structural signals (moderate confidence — gauntlet B2: require action verb)
        _has_imperative = any(kw in s_lower for kw in ("should ", "require", "enforce", "ensure", "must "))
        if _has_imperative and any(kw in s_lower for kw in ("always ", "never ", "rule:")):
            if len(summary) > 60:
                return "constraint"
        # Causal / conditional patterns → heuristic
        if ("→" in summary or "->" in summary) and any(kw in s_lower for kw in ("when ", "if ", "trigger")):
            return "heuristic"

    if claimed_type == "principle":
        stripped = summary.strip()
        if not stripped.startswith("PRINCIPLE:") and not stripped.startswith("[PRINCIPLE]"):
            if _IMPLEMENTATION_DETAIL_PATTERNS.search(summary):
                return "observation"
            if _COMMIT_RECORD_PATTERNS.search(summary):
                return "observation"
    if claimed_type == "decision":
        if _BACKLOG_PATTERNS.search(summary):
            return "observation"
        if _COMMIT_RECORD_PATTERNS.search(summary):
            return "observation"
    return claimed_type


# Reflection trigger: if learning events exceed this in a session, trigger reflection
REFLECTION_TRIGGER_THRESHOLD = 5


# --- TEMPORAL_AWARENESS v2.4: Freshness computation ---
def _compute_freshness(
    created_at_iso: str,
    now: "datetime",
    current_session_start: "datetime | None" = None,
) -> "tuple[int | None, str | None]":
    """Compute age_minutes and human-readable freshness_label for a concept.

    Pure datetime arithmetic — zero DB queries. Safe under concurrent sessions:
    each conversation_turn has exactly one current_session.
    """
    if not created_at_iso:
        return None, None
    try:
        created = _ensure_aware(datetime.fromisoformat(created_at_iso))
    except (ValueError, TypeError):
        return None, None
    delta = now - created
    age_minutes = max(0, int(delta.total_seconds() / 60))

    # Session-relative label takes priority (v2.3)
    if current_session_start and created >= current_session_start:
        label = "this session"
    elif age_minutes < FRESHNESS_JUST_NOW_MINS:
        label = "just now"
    elif age_minutes < FRESHNESS_MINUTES_AGO_UPPER:
        label = f"{age_minutes} minutes ago"
    elif age_minutes < FRESHNESS_ONE_HOUR_UPPER:
        label = "~1 hour ago"
    elif age_minutes < FRESHNESS_HOURS_AGO_UPPER:
        label = f"~{age_minutes // MINUTES_PER_HOUR} hours ago"
    elif age_minutes < FRESHNESS_EARLIER_TODAY_UPPER:
        label = "earlier today"
    elif age_minutes < FRESHNESS_YESTERDAY_UPPER:
        label = "yesterday"
    else:
        label = f"{age_minutes // FRESHNESS_EARLIER_TODAY_UPPER} days ago"  # 1440 min = 24h = 1 day

    return age_minutes, label


# P0-PRECISION: Named entity detection for evolution specificity guard.
# Lightweight heuristic — checks for patterns that indicate specific details
# (proper nouns, dollar amounts, times, numbers with units).
# Used by _evolve_existing_from_dedup to prevent summary re-lossification.
# PREC-001: Now delegates to app.entity_detector for unified patterns.
_PRECISION_GUARD_BLOCKS: int = 0  # Counter for /brain_health observability
_NAMED_ENTITY_PATTERNS = [
    _re.compile(r'\$[\d,]+'),                          # Dollar amounts ($4,500)
    _re.compile(r'\d{1,2}:\d{2}\s*(?:[AaPp][Mm])?'),   # Times (6:30 PM, 14:30)
    _re.compile(r'\d+\s*(?:GB|MB|TB|kg|lbs|oz|miles|km|mph|%)', _re.IGNORECASE),  # Numbers with units
    _re.compile(r'(?<![.!?]\s)[A-Z\u00C0-\u024F][a-z\u00E0-\u024F]{2,}(?:\s+(?:[a-z]{1,3}\s+)?[A-Z\u00C0-\u024F][a-z\u00E0-\u024F]{2,})+'),  # Multi-word proper nouns with optional lowercase connectors (Café Luna, Seco de Cordero, São Paulo)
    _re.compile(r'(?<=\s)[A-Z\u00C0-\u024F][a-z\u00E0-\u024F]{2,}(?:\s+(?:or|and|&)\s+[A-Z\u00C0-\u024F][a-z\u00E0-\u024F]{2,})'),  # Paired proper nouns (Pilsner or Lager, Mac and Cheese)
    _re.compile(r'\b\d{4}-\d{2}-\d{2}\b'),             # ISO dates (2026-03-17)
    _re.compile(r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}', _re.IGNORECASE),  # Named dates
]


def _has_named_entities(text: str) -> bool:
    """Check if text contains likely named entities or specific details.

    DEPRECATED: Delegates to app.entity_detector.has_specific_entities.
    Kept as alias for backward compatibility. The unified patterns in
    entity_detector.py supersede _NAMED_ENTITY_PATTERNS above.
    Ref: PREC-001 spec.
    """
    from app.entity_detector import has_specific_entities
    return has_specific_entities(text)



# ── Module-level subject-key extraction (RETRIEVAL-072) ─────────────────
# Lifted from _conflict_prefilter inner scope so write-time supersession
# and other callers can reuse the same keying logic.

_SK_STRUCTURED_ENDINGS = [
    ' is famous for ', ' was created in the country of ',
    ' was born in the city of ', ' died in the city of ',
    ' is located in the continent of ', ' is located in the city of ',
    ' is associated with the sport of ',
    ' is affiliated with the religion of ',
    ' speaks the language of ', ' is married to ', ' is a citizen of ',
    ' is employed by ', ' was founded in the city of ',
    ' was founded by ', ' was performed by ', ' was developed by ',
    ' was written in the language of ', ' works in the field of ',
    ' worked in the city of ', ' plays the position of ',
    ' of the current head of state in ', ' of the current head of the ',
    ' of the Prime Minister of ',
    ' is located in the ',
    ' was created by ',
    "\'s child is ",
    ' educated is ',
    ' was educated is ',
]

_SK_ENTITY_EXTENDING = {
    ' of the current head of state in ',
    ' of the current head of the ',
    ' of the prime minister of ',
}

_SK_COPULAS = [' is ', ' was ', ' are ', ' were ', ' has ', ' had ',
               ' plays ', ' speaks ', ' works ', ' died ', ' performed ',
               ' created ', ' founded ', ' affiliated ', ' employed ', ' married ',
               ' took ', ' got ', ' bought ', ' went ', ' started ', ' finished ',
               ' made ', ' gave ', ' spent ', ' earned ', ' lost ', ' moved ',
               ' plans ', ' wants ', ' likes ', ' enjoys ', ' prefers ']

_SK_STOP_WORDS = frozenset({'a', 'an', 'the', 'their', 'his', 'her', 'its', 'my',
                             'in', 'on', 'at', 'to', 'for', 'of', 'from', 'with',
                             'and', 'or', 'but', 'that', 'which', 'who', 'where',
                             'been', 'being', 'have', 'having', 'also', 'very',
                             'just', 'about', 'really', 'currently', 'recently',
                             'specifically', 'especially', 'new', 'old'})

_SK_PLACEHOLDERS = {'n', '$x', 'time', 'date'}


def _sk_content_word_count(text: str) -> int:
    count = 0
    for w in text.split():
        w_clean = w.strip('.,;:!?\'()"\'\'\'-').lower()
        if w_clean and w_clean not in _SK_STOP_WORDS and w_clean not in _SK_PLACEHOLDERS:
            count += 1
    return count


def _extract_subject_key(text: str) -> str:
    """Extract subject+predicate key from a concept summary.

    RETRIEVAL-072: Module-level version of _conflict_prefilter._extract_key.
    Used by write-time supersession and conflict prefilter.

    Returns a normalized lowercase key string, or empty string on failure.
    """
    if not text:
        return ""
    text_lower = text.lower()

    # Strip common prefixes that break key matching
    for _prefix in ('established fact: ', 'known fact: ', 'confirmed: '):
        if text_lower.startswith(_prefix):
            text_lower = text_lower[len(_prefix):]
            break

    # Phase 1: Structured pattern matching
    for ending in _SK_STRUCTURED_ENDINGS:
        idx = text_lower.find(ending.lower())
        if idx >= 0:
            key_end = idx + len(ending)
            if ending.lower().strip() in [e.strip() for e in _SK_ENTITY_EXTENDING]:
                remainder = text_lower[key_end:]
                gov_is = remainder.find(' government is ')
                is_idx = remainder.find(' is ')
                if gov_is >= 0:
                    key_end += gov_is + len(' government is ')
                elif is_idx >= 0:
                    key_end += is_idx + len(' is ')
            return text_lower[:key_end]

    # Phase 1.5: rfind fallback for structured facts with long subjects
    for _sep in (' is ', ' was '):
        _ridx = text_lower.rfind(_sep)
        if _ridx > 5:
            _rfind_subject = text_lower[:_ridx + len(_sep)]
            if _sk_content_word_count(text_lower[:_ridx]) >= 3:
                return _rfind_subject

    # Phase 2: Adaptive copula keying
    for cop in _SK_COPULAS:
        idx = text_lower.find(cop)
        if idx > 0:
            subject = text_lower[:idx].strip()
            predicate = text_lower[idx + len(cop):].strip()
            pred_norm = _re.sub(r'\$[\d,]+\.?\d*', '$X', predicate)
            pred_norm = _re.sub(r'\b\d{1,2}:\d{2}\s*(?:am|pm)?\b', 'TIME', pred_norm, flags=_re.I)
            pred_norm = _re.sub(r'\b\d{4}[/-]\d{2}[/-]\d{2}\b', 'DATE', pred_norm)
            pred_norm = _re.sub(r'\b\d+\b', 'N', pred_norm)
            if subject.startswith('the ') and ' of ' in subject:
                return subject
            if _sk_content_word_count(pred_norm) <= 2:
                return subject
            else:
                return f"{subject} | {' '.join(pred_norm.split()[:6])}"

    # Phase 3: Last-resort fallback
    return ' '.join(text_lower.split()[:8])



def _conflict_prefilter(concepts: list) -> list:
    """RETRIEVAL-037b v4: Unified conflict resolution with structured pattern matching.

    Two-phase subject-key extraction:
    Phase 1: 24 structured predicate patterns (from rebuild_brains.py extract_fact_key).
             Matches knowledge-base facts like "X is associated with the sport of Y".
             Never fires on personal memory ("User has..." / "User is...").
    Phase 2: v3 adaptive copula keying (fallback for unmatched concepts).
             Handles personal memory and free-form summaries.

    Then keeps highest serial_order per subject key.
    Concepts without serial_order pass through unfiltered.
    """
    # ── Phase 1: Structured predicate patterns ──────────────────────────
    # Specific patterns from rebuild_brains.py extract_fact_key.
    # These extract "subject+predicate" keys that stop BEFORE the object value.
    # E.g., "Steve Sax is associated with the sport of baseball"
    #     → key: "steve sax is associated with the sport of "
    _STRUCTURED_ENDINGS = [
        ' is famous for ', ' was created in the country of ',
        ' was born in the city of ', ' died in the city of ',
        ' is located in the continent of ', ' is located in the city of ',
        ' is associated with the sport of ',
        ' is affiliated with the religion of ',
        ' speaks the language of ', ' is married to ', ' is a citizen of ',
        ' is employed by ', ' was founded in the city of ',
        ' was founded by ', ' was performed by ', ' was developed by ',
        ' was written in the language of ', ' works in the field of ',
        ' worked in the city of ', ' plays the position of ',
        ' of the current head of state in ', ' of the current head of the ',
        ' of the Prime Minister of ',
        ' is located in the ',
        # RETRIEVAL-037b v4.1: Additional patterns from rebuild_brains.py + RCA
        ' was created by ',       # "Vito Corleone was created by X"
        "'s child is ",           # "Aristotle's child is X"
        ' educated is ',          # "...where X was educated is Y"
        ' was educated is ',      # variant
    ]
    # Patterns where the entity after the predicate extends into the key
    # E.g., "The name of the current head of the Italy government is X"
    #     → key includes "italy government is " (entity + " government is ")
    _ENTITY_EXTENDING = {
        ' of the current head of state in ',
        ' of the current head of the ',
        ' of the prime minister of ',
    }

    # ── Phase 2: v3 adaptive copula keying (unchanged) ──────────────────
    _COPULAS = [' is ', ' was ', ' are ', ' were ', ' has ', ' had ',
                ' plays ', ' speaks ', ' works ', ' died ', ' performed ',
                ' created ', ' founded ', ' affiliated ', ' employed ', ' married ',
                ' took ', ' got ', ' bought ', ' went ', ' started ', ' finished ',
                ' made ', ' gave ', ' spent ', ' earned ', ' lost ', ' moved ',
                ' plans ', ' wants ', ' likes ', ' enjoys ', ' prefers ']
    _STOP_WORDS = frozenset({'a', 'an', 'the', 'their', 'his', 'her', 'its', 'my',
                              'in', 'on', 'at', 'to', 'for', 'of', 'from', 'with',
                              'and', 'or', 'but', 'that', 'which', 'who', 'where',
                              'been', 'being', 'have', 'having', 'also', 'very',
                              'just', 'about', 'really', 'currently', 'recently',
                              'specifically', 'especially', 'new', 'old'})
    _PLACEHOLDERS = {'n', '$x', 'time', 'date'}

    def _content_word_count(text: str) -> int:
        count = 0
        for w in text.split():
            w_clean = w.strip('.,;:!?\'()"\'\'-').lower()
            if w_clean and w_clean not in _STOP_WORDS and w_clean not in _PLACEHOLDERS:
                count += 1
        return count

    def _extract_key(text: str) -> str:
        """Extract subject+predicate key. Tries structured patterns first, then v3 fallback."""
        text_lower = text.lower()

        # RETRIEVAL-037b v4.1: Strip common prefixes that break key matching
        for _prefix in ('established fact: ', 'known fact: ', 'confirmed: '):
            if text_lower.startswith(_prefix):
                text_lower = text_lower[len(_prefix):]
                break

        # Phase 1: Structured pattern matching
        for ending in _STRUCTURED_ENDINGS:
            idx = text_lower.find(ending.lower())
            if idx >= 0:
                key_end = idx + len(ending)
                # Entity-extending patterns: key includes the governing entity
                if ending.lower().strip() in [e.strip() for e in _ENTITY_EXTENDING]:
                    remainder = text_lower[key_end:]
                    gov_is = remainder.find(' government is ')
                    is_idx = remainder.find(' is ')
                    if gov_is >= 0:
                        key_end += gov_is + len(' government is ')
                    elif is_idx >= 0:
                        key_end += is_idx + len(' is ')
                return text_lower[:key_end]

        # Phase 1.5: rfind fallback for structured facts with long subjects.
        # Uses rebuild_brains.py approach: rightmost separator splits subject from object.
        # Only fires when the subject is long (>= 3 content words) — avoids over-dedup
        # on short personal-memory subjects like "user" or "andrew".
        # RETRIEVAL-037d: Lowered from 4 to 3 to catch "chairperson of Harvard University",
        # "author of Starship Troopers" (3 content words) that previously fell through to
        # Phase 2 copula keying where asymmetric object word counts caused different keys.
        for _sep in (' is ', ' was '):
            _ridx = text_lower.rfind(_sep)
            if _ridx > 5:
                _rfind_subject = text_lower[:_ridx + len(_sep)]
                if _content_word_count(text_lower[:_ridx]) >= 3:
                    return _rfind_subject

        # Phase 2: v3 adaptive copula keying (personal memory, free-form)
        for cop in _COPULAS:
            idx = text_lower.find(cop)
            if idx > 0:
                subject = text_lower[:idx].strip()
                predicate = text_lower[idx + len(cop):].strip()
                pred_norm = _re.sub(r'\$[\d,]+\.?\d*', '$X', predicate)
                pred_norm = _re.sub(r'\b\d{1,2}:\d{2}\s*(?:am|pm)?\b', 'TIME', pred_norm, flags=_re.I)
                pred_norm = _re.sub(r'\b\d{4}[/-]\d{2}[/-]\d{2}\b', 'DATE', pred_norm)
                pred_norm = _re.sub(r'\b\d+\b', 'N', pred_norm)
                # RETRIEVAL-037d: Definitional patterns ("the X of Y is Z") always use
                # subject-only key. This prevents asymmetric keys when two objects
                # straddle the cwc<=2 boundary (e.g., "Peter Diamandis" cwc=2 vs
                # "Lawrence S. Bacow" cwc=3). Personal memory subjects (no "the...of")
                # still use the predicate-inclusive key to avoid over-dedup.
                if subject.startswith('the ') and ' of ' in subject:
                    return subject
                if _content_word_count(pred_norm) <= 2:
                    return subject
                else:
                    return f"{subject} | {' '.join(pred_norm.split()[:6])}"

        # Phase 3: Last-resort fallback — first 8 words
        return ' '.join(text_lower.split()[:8])

    def _get_serial(c) -> int:
        """Extract serial order, falling back to evidence Serial #N: tags, then 0."""
        so = getattr(c, 'serial_order', None)
        if so is not None:
            return so
        # Try evidence strings (MAB-style Serial #N: tags)
        for ev in (getattr(c, 'key_evidence', None) or []):
            ev_str = ev if isinstance(ev, str) else str(ev)
            m = _re.search(r'Serial #(\d+):', ev_str)
            if m:
                return int(m.group(1))
        return 0

    best_per_subject: dict[str, object] = {}

    for c in concepts:
        text = (getattr(c, 'summary', '') or '')
        key = _extract_key(text)
        c_serial = _get_serial(c)

        existing = best_per_subject.get(key)
        if existing is None or c_serial > _get_serial(existing):
            best_per_subject[key] = c

    return list(best_per_subject.values())


def _extract_object_value(summary: str) -> str | None:
    """Extract the object value (rightmost entity) from a fact summary.

    E.g., 'Steve Sax is associated with the sport of baseball' -> 'baseball'
          'The capital of France is Harare' -> 'Harare'
    """
    _OBJ_PATTERNS = [
        _re.compile(r'is associated with the sport of (.+)$', _re.I),
        _re.compile(r'was created in the country of (.+)$', _re.I),
        _re.compile(r'is a citizen of (.+)$', _re.I),
        _re.compile(r'was created by (.+)$', _re.I),
        _re.compile(r'died in the city of (.+)$', _re.I),
        _re.compile(r'was born in the city of (.+)$', _re.I),
        _re.compile(r'plays the position of (.+)$', _re.I),
        _re.compile(r'(?:The )?capital of .+? is (.+)$', _re.I),
        _re.compile(r'chief executive officer of .+? is (.+)$', _re.I),
        _re.compile(r'(?:The )?author of .+? is (.+)$', _re.I),
        _re.compile(r'is married to (.+)$', _re.I),
        _re.compile(r'is famous for (.+)$', _re.I),
        _re.compile(r'head of state in .+? is (.+)$', _re.I),
        _re.compile(r'head of the .+? government is (.+)$', _re.I),
        _re.compile(r'(?:The )?director of .+? is (.+)$', _re.I),
        _re.compile(r'was developed by (.+)$', _re.I),
        _re.compile(r'was written in the language of (.+)$', _re.I),
        _re.compile(r'(?:The )?official language of .+? is (.+)$', _re.I),
        _re.compile(r'is located in the continent of (.+)$', _re.I),
        _re.compile(r"'s child is (.+)$", _re.I),
        _re.compile(r'was performed by (.+)$', _re.I),
        _re.compile(r'type of music that .+? plays is (.+)$', _re.I),
        _re.compile(r'is employed by (.+)$', _re.I),
        _re.compile(r'was founded in the city of (.+)$', _re.I),
        _re.compile(r'is located in the city of (.+)$', _re.I),
        _re.compile(r'was founded by (.+)$', _re.I),
    ]
    for pat in _OBJ_PATTERNS:
        m = pat.search(summary)
        if m:
            return m.group(1).strip().rstrip('.')
    # Fallback: take everything after last ' is ' or ' was '
    for verb in [' is ', ' was ']:
        idx = summary.rfind(verb)
        if idx > 0:
            return summary[idx + len(verb):].strip().rstrip('.')
    return None


def _chain_aware_prune(concepts: list, destroyed_concepts: list) -> list:
    """Remove orphaned chain fragments after subject-level dedup.

    When a conflict loser is destroyed (e.g., "Steve Sax → baseball"),
    downstream facts keyed on the destroyed OBJECT become orphaned.

    Key insight: an object is only TRULY orphaned if no surviving concept
    still references it. Uses reference counting, not global elimination.

    Args:
        concepts: surviving concepts after subject-level dedup
        destroyed_concepts: concepts removed by subject-level dedup
    Returns:
        filtered concept list with orphans removed
    """
    # Step 1: Collect objects from destroyed concepts (orphan candidates)
    destroyed_objects: set[str] = set()
    for c in destroyed_concepts:
        summary = (getattr(c, 'summary', '') or '')
        obj_val = _extract_object_value(summary)
        if obj_val:
            destroyed_objects.add(obj_val.lower())

    if not destroyed_objects:
        return concepts

    # Step 2: Collect objects still referenced by surviving concepts
    surviving_objects: set[str] = set()
    for c in concepts:
        summary = (getattr(c, 'summary', '') or '')
        obj_val = _extract_object_value(summary)
        if obj_val:
            surviving_objects.add(obj_val.lower())

    # Step 3: Truly orphaned = destroyed but not still referenced
    orphaned_objects = destroyed_objects - surviving_objects

    if not orphaned_objects:
        return concepts

    # Step 4: Iteratively remove concepts whose subject references orphaned objects
    total_pruned = 0
    filtered = list(concepts)
    for _round in range(5):  # Max 5 chain hops (MAB FC chains ≤4 deep)
        pruned_this_round = []
        kept_this_round = []
        for c in filtered:
            summary = (getattr(c, 'summary', '') or '').lower()
            is_orphan = False
            for orph in orphaned_objects:
                if summary.startswith(orph) or summary.startswith(f"the {orph}"):
                    is_orphan = True
                    break
                # Catch "the capital of X" / "the official language of X"
                if f"of {orph} " in summary or summary.endswith(f"of {orph}"):
                    for verb in [' is ', ' was ']:
                        vidx = summary.find(verb)
                        if vidx > 0 and f"of {orph}" in summary[:vidx]:
                            is_orphan = True
                            break
                    if is_orphan:
                        break
            if is_orphan:
                pruned_this_round.append(c)
                obj_val = _extract_object_value(
                    (getattr(c, 'summary', '') or ''))
                if obj_val:
                    orphaned_objects.add(obj_val.lower())
            else:
                kept_this_round.append(c)

        total_pruned += len(pruned_this_round)
        filtered = kept_this_round
        if not pruned_this_round:
            break

    if total_pruned > 0:
        logger.info(
            f"RETRIEVAL-037b-chain: Pruned {total_pruned} orphaned chain fragments "
            f"(orphaned objects: {', '.join(sorted(orphaned_objects)[:5])})"
        )
    return filtered


class SessionManager:
    """Manages session lifecycle and orientation generation."""

    # CTX-005: Compaction recovery quality factor weights (sum to 1.0)
    COMPACTION_QUALITY_HAS_SNAPSHOT: float = 0.4
    COMPACTION_QUALITY_HAS_TASK: float = 0.2
    COMPACTION_QUALITY_HAS_PINNED: float = 0.2
    COMPACTION_QUALITY_HAS_GIST: float = 0.2

    def __init__(self):
        self.current_session: SessionInfo | None = None
        self._recovery_done: bool = False
        self._conversation_turn_called: bool = False  # S0: first-call detection
        self._last_conversation_turn_at: float | None = None  # S0: timestamp for conversation boundary detection
        self._last_orientation_served_at: float | None = None  # S6: fallback orientation re-serve
        # B1: Active extraction request tracking (Attack 2 anti-nagging + Attack 5 suppression)
        self._last_extraction_request_types: set = set()
        self._suppressed_gap_types: set = set()
        # RETRO-001: One retrospective check per session
        self._retro_checked_this_session: bool = False
        # GOV-W2: Track activated concepts for correction detection on next turn
        self._last_activated_concept_ids: list[str] = []
        # GOV-W2: Cache activated concept dicts with embeddings for Layer 4 drift detection
        self._last_activated_concept_dicts: list[dict[str, Any]] = []
        # CTX Phase 2: Compaction detection state
        self._consecutive_empty_extractions: int = 0
        self._last_compaction_detected_at: float | None = None  # Cooldown tracking (CTX-2)
        self._compaction_false_positive_count: int = 0  # Session circuit breaker (CTX-2)
        self._episode_turn_counter: int = 0  # INFRA-002: monotonic per-session turn counter for episodes
        self._promoted_this_session: set[str] = set()  # ARCH-D05: rate-limit promotion checks
        self._cumulative_response_bytes: int = 0  # CTX-003: cumulative previous_response bytes for pressure scoring
        # CONCEPT_LIFECYCLE_SPEC L4: Track concept IDs created during session
        self._session_concept_ids: set[str] = set()
        # STABILITY-013: Strong references prevent GC of fire-and-forget tasks
        self._background_tasks: set = set()
        # PERF-FORT-2: Background auto-learn state (cross-turn)
        self._learn_executor = None  # Lazy-init ThreadPoolExecutor(max_workers=1)
        self._last_autolearn_result: dict | None = None  # Previous turn's auto_learned dict
        self._last_autolearn_result_obj = None  # Previous turn's SessionLearnResponse object
        self._last_autolearn_budget_warnings: list = []  # Previous turn's budget warnings
        # PERF-013: Git state cache — populated once at session_start
        self.git_cache: GitCache | None = None
        self._cached_pinned_concepts: list[dict] | None = None  # CONTEXT-001: Turn-scoped cache
        self._cached_pinned_concepts_turn: int = -1  # Turn number when cache was set

    def _on_bg_task_done(self, task) -> None:
        """Done callback for background tasks. Logs errors, removes reference, emits metrics."""
        self._background_tasks.discard(task)
        try:
            exc = task.exception()
            if exc:
                logger.error(f"Background task {task.get_name()} failed: {exc}")
                from app.metrics import metrics as _bg_metrics

                _bg_metrics.record("bg_task_failure", 1.0, {"task": task.get_name()})
            else:
                from app.metrics import metrics as _bg_metrics

                _bg_metrics.record("bg_task_success", 1.0, {"task": task.get_name()})
        except (asyncio.CancelledError, asyncio.InvalidStateError):
            from app.metrics import metrics as _bg_metrics

            _bg_metrics.record("bg_task_cancelled", 1.0, {"task": task.get_name()})
            pass  # Cancellation during shutdown is expected, not an error

    def _background_autolearn(
        self,
        learn_request,
        extracted: list | None,
        request_message: str,
        prev_msg: str,
        prev_response: str,
    ):
        """PERF-FORT-2: Background auto-learn — runs in executor thread.

        Performs session_learn + episode recording + reflection tracking + pricing
        without blocking the conversation_turn response path.

        NOTE: session_learn uses the shared DB connection via storage backend.
        The storage backend's RLock serializes access, so background writes
        wait for main-path reads to complete and vice versa. This is safe
        but may add latency to whichever thread is waiting.
        """
        try:
            auto_learn_result = self.session_learn(learn_request)
            logger.info(
                f"S-1-BG: Auto-learned: {auto_learn_result.learning_events} events, "
                f"sources={auto_learn_result.extraction_source_breakdown}"
            )

            # Track rejected-after-request gaps
            if auto_learn_result and auto_learn_result.garbage_rejected > 0 and self._last_extraction_request_types:
                self._suppressed_gap_types.update(self._last_extraction_request_types)

            # --- INFRA-002: Episode recording (moved from main path) ---
            try:
                from app.config import FEATURE_FLAGS as _ep_ff
                if self.current_session and _ep_ff.get("EPISODES_ENABLED", False):
                    from app.episodes import record_episode
                    self._episode_turn_counter += 1
                    _ep_concept_ids = [c.concept_id for c in auto_learn_result.concepts_created]
                    _ep_changes = [
                        {"action": "created", "id": c.concept_id} for c in auto_learn_result.concepts_created
                    ] + [{"action": "evolved", "id": c.concept_id} for c in auto_learn_result.concepts_evolved]
                    record_episode(
                        session_id=self.current_session.session_id,
                        turn_number=self._episode_turn_counter,
                        intent_summary=(learn_request.knowledge_area or "")[:500],
                        classification=(learn_request.knowledge_area or "")[:200],
                        extracted_concept_ids=_ep_concept_ids,
                        concept_changes=_ep_changes,
                        raw_user_message=request_message[:5000] if request_message else None,
                        raw_assistant_response=(prev_response or "")[:5000] or None,
                    )
            except Exception as e:
                logger.warning(f"INFRA-002-BG: Episode recording failed (non-fatal): {e}")

            # --- RB-02: Reflection completion tracking (moved from main path) ---
            if auto_learn_result and auto_learn_result.learning_events > 0:
                try:
                    from app.storage import _db
                    from app.datetime_utils import _utc_now_iso as _bg_utc_now_iso
                    with _db() as conn:
                        conn.execute(
                            """UPDATE reflection_tracking
                               SET completed_at = ?,
                                   concepts_returned = ?,
                                   reflection_quality = 'auto_closed'
                               WHERE id = (
                                   SELECT id FROM reflection_tracking
                                   WHERE completed_at IS NULL
                                   ORDER BY created_at DESC LIMIT 1
                               )""",
                            (_bg_utc_now_iso(), auto_learn_result.learning_events),
                        )
                except Exception as e:
                    logger.debug(f"RB-02-BG: Reflection tracking failed (non-fatal): {e}")

            # --- PRICING-002: Meter concept-producing turns (moved from main path) ---
            if auto_learn_result and auto_learn_result.learning_events > 0:
                try:
                    from app.pricing import conversation_meter
                    conversation_meter.consume_turn()
                except Exception as e:
                    logger.debug(f"PRICING-002-BG: Metering failed (non-fatal): {e}")

            # Store results for next turn's consumption (A1 amendment)
            # Build the auto_learned summary dict
            _auto_learned_dict = None
            _budget_warnings = auto_learn_result.budget_warnings or []
            if auto_learn_result.learning_events > 0:
                _auto_learned_dict = {
                    "events": auto_learn_result.learning_events,
                    "concepts_created": [c.concept_id for c in auto_learn_result.concepts_created],
                    "concepts_evolved": [c.concept_id for c in auto_learn_result.concepts_evolved],
                    "budget_warnings": _budget_warnings,
                }
            # Atomic assignment — GIL protects reference swap
            self._last_autolearn_result = _auto_learned_dict
            self._last_autolearn_result_obj = auto_learn_result
            self._last_autolearn_budget_warnings = _budget_warnings

        except Exception as e:
            logger.error(f"S-1-BG: Background auto-learn failed: {e}", exc_info=True)
            self._last_autolearn_result = None
            self._last_autolearn_result_obj = None
            self._last_autolearn_budget_warnings = []

    def _load_all_concepts(self) -> list[Concept]:
        """Load all active concepts from storage. Single scan point."""
        concepts = []
        for cid in list_concepts():
            c = load_concept(cid, track_access=False)
            if c:
                concepts.append(c)
        return concepts

    def orient(
        self,
        concepts: list[Concept],
        time_window: str = DEFAULT_WINDOW,
    ) -> PresentMomentOrientation:
        """Generate present moment orientation.

        Args:
            concepts: Pre-loaded concepts (avoids redundant storage scan).
            time_window: "1_day" | "7_days" | "30_days" | "all"
        """
        now = _utc_now()
        delta = TIME_WINDOWS.get(time_window, TIME_WINDOWS[DEFAULT_WINDOW])
        cutoff = (now - delta).isoformat()

        where_been = self._compute_where_been(concepts, cutoff, time_window)
        where_am = self._compute_where_am(concepts)
        where_going = self._compute_where_going(concepts)

        orientation = PresentMomentOrientation(
            generated_at=now.isoformat(),
            generated_by="pith_deterministic",
            where_been=where_been,
            where_am=where_am,
            where_going=where_going,
            open_threads=self._compute_open_threads(),
            experiment_summary=self._compute_experiment_summary(),
        )

        # Compute orientation hash for cache invalidation (exclude timestamps)
        data = orientation.model_dump(exclude={"orientation_hash", "generated_at"})
        orientation.orientation_hash = hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode()
        ).hexdigest()

        return orientation

    def _compute_where_been(self, concepts: list[Concept], cutoff: str, window_label: str) -> RecentEvolutionSummary:
        """Recent evolution summary: what changed in the time window."""
        created = []
        evolved = []
        total_events = 0

        for c in concepts:
            # Recently created
            if c.created_at and c.created_at >= cutoff:
                created.append(
                    RecentConceptSummary(
                        concept_id=c.id,
                        summary=c.summary[:100],
                        knowledge_area=(c.metadata or {}).get("knowledge_area", "unknown"),
                        created_at=c.created_at,
                    )
                )
                total_events += 1

            # Recently evolved (version > v1 AND updated in window)
            if c.updated_at and c.updated_at >= cutoff and c.version and c.version != "v1":
                evolved.append(
                    ConceptEvolutionRecord(
                        concept_id=c.id,
                        summary=c.summary[:100],
                        change_type=c.change_type or "",
                        change_reason=c.change_reason or "",
                        evolved_at=c.updated_at,
                    )
                )
                total_events += 1

        # Sort by recency
        created.sort(key=lambda x: x.created_at or "", reverse=True)
        evolved.sort(key=lambda x: x.evolved_at or "", reverse=True)

        # ARCH-O01: Query recent contradictions from DB (capped at 20)
        contradictions = []
        try:
            from app.storage import _db
            with _db() as conn:
                contra_rows = conn.execute(
                    """SELECT concept_a_id, concept_b_id, contradiction_type,
                              action, reason, created_at
                       FROM contradiction_resolutions
                       WHERE created_at >= ?
                       ORDER BY created_at DESC LIMIT 20""",
                    (cutoff,),
                ).fetchall()
                contradictions = [
                    {
                        "concept_a_id": r["concept_a_id"],
                        "concept_b_id": r["concept_b_id"],
                        "type": r["contradiction_type"],
                        "action": r["action"],
                        "reason": r["reason"][:200] if r["reason"] else "",
                        "detected_at": r["created_at"],
                    }
                    for r in contra_rows
                ]
        except Exception as e:
            logger.warning(f"ARCH-O01: Failed to load contradictions (non-fatal): {e}")

        # ARCH-O01: Query recent corrections from DB (capped at 20)
        corrections = []
        try:
            from app.storage import _db
            with _db() as conn:
                corr_rows = conn.execute(
                    """SELECT id, correction_type, corrected_claim, correct_claim,
                              detection_confidence, created_at
                       FROM corrections
                       WHERE created_at >= ?
                       ORDER BY created_at DESC LIMIT 20""",
                    (cutoff,),
                ).fetchall()
                corrections = [
                    {
                        "correction_id": r["id"],
                        "type": r["correction_type"],
                        "corrected_claim": r["corrected_claim"][:200] if r["corrected_claim"] else "",
                        "correct_claim": r["correct_claim"][:200] if r["correct_claim"] else "",
                        "confidence": r["detection_confidence"],
                        "corrected_at": r["created_at"],
                    }
                    for r in corr_rows
                ]
        except Exception as e:
            logger.warning(f"ARCH-O01: Failed to load corrections (non-fatal): {e}")

        return RecentEvolutionSummary(
            time_window=window_label,
            concepts_created=created[:20],
            concepts_evolved=evolved[:20],
            concepts_decayed=[],  # STUB — decay doesn't annotate change_type
            contradictions_detected=contradictions,  # ARCH-O01: wired
            corrections_made=corrections,  # ARCH-O01: wired
            session_count_in_window=count_sessions(since=cutoff),
            total_learning_events_in_window=total_events,
        )

    def _compute_where_am(self, concepts: list[Concept]) -> CurrentStateAssessment:
        """Current state assessment: knowledge health, strengths, weaknesses, uncertainties.

        Reuses SelfModel epistemic profile for health/areas to avoid recomputation.
        """
        # Get epistemic data from SelfModel (cached or generate)
        sm = self_model_manager.load()
        if sm is None:
            sm = self_model_manager.generate(concepts)

        ep = sm.epistemic_profile
        kh = ep.knowledge_health

        health = {
            "total_concepts": kh.total_concepts,
            "total_associations": 0,
            "avg_confidence": kh.avg_confidence,
            "avg_stability": kh.avg_stability,
            "contradiction_density": kh.contradiction_density,
            "evidence_coverage": 0.0,
        }

        # Count associations
        health["total_associations"] = count_associations()

        # Evidence coverage: % of concepts with any evidence
        if concepts:
            with_evidence = sum(1 for c in concepts if c.evidence)
            health["evidence_coverage"] = round(with_evidence / len(concepts), 3)

        # Strongest areas (top 3 by avg_confidence, min 2 concepts)
        dist = sorted(
            ep.knowledge_distribution,
            key=lambda d: d.avg_confidence,
            reverse=True,
        )
        strongest = [
            AreaStrength(
                knowledge_area=d.knowledge_area,
                concept_count=d.concept_count,
                avg_confidence=round(d.avg_confidence, 3),
                reason="Highest avg confidence",
            )
            for d in dist[:3]
            if d.concept_count >= 2
        ]

        # Weakest areas (bottom 3 by avg_confidence, min 2 concepts)
        multi_concept = [d for d in dist if d.concept_count >= 2]
        weakest = [
            AreaStrength(
                knowledge_area=d.knowledge_area,
                concept_count=d.concept_count,
                avg_confidence=round(d.avg_confidence, 3),
                reason="Lowest avg confidence",
            )
            for d in reversed(multi_concept[-3:])
        ]

        # Active uncertainties (top 5 lowest-confidence concepts)
        sorted_by_conf = sorted(concepts, key=lambda c: c.confidence)
        uncertainties = [
            ActiveUncertainty(
                concept_id=c.id,
                summary=c.summary[:100],
                confidence=c.confidence,
                uncertainty_type=("low_stability" if c.stability < 0.3 else "low_confidence"),
            )
            for c in sorted_by_conf[:5]
        ]

        # Pending questions from curiosity engine
        from app import question_queue

        raw_questions = question_queue.get_questions(limit=5)
        questions = [
            PendingQuestionSummary(
                question=q.get("question", ""),
                concept_id=q.get("concept_id", ""),
                priority=q.get("priority", 0.0),
            )
            for q in raw_questions
        ]

        # --- Cognitive velocity (self-awareness) ---
        cognitive_velocity = self._compute_cognitive_velocity()

        return CurrentStateAssessment(
            knowledge_health=health,
            strongest_areas=strongest,
            weakest_areas=weakest,
            active_uncertainties=uncertainties,
            pending_questions=questions,
            cognitive_velocity=cognitive_velocity,
        )

    def _compute_cognitive_velocity(self) -> "CognitiveVelocity":
        """Compute self-awareness metrics: how fast is Pith growing?

        Uses 7-day current window vs 7-day prior window for trend detection.
        Gracefully returns defaults on any error.
        """
        from app.models import CognitiveVelocity

        try:
            now = _utc_now()
            window_days = 7
            current_cutoff = (now - timedelta(days=window_days)).isoformat()
            prior_cutoff = (now - timedelta(days=window_days * 2)).isoformat()

            velocity_data = load_session_velocity(current_cutoff, prior_cutoff)
            current = velocity_data["current"]
            prior = velocity_data.get("prior")

            sessions = current["session_count"]
            created = current["total_concepts_created"]
            evolved = current["total_concepts_evolved"]
            learning_events = current["total_learning_events"]

            avg_concepts = round(created / max(sessions, 1), 2)
            avg_learning = round(learning_events / max(sessions, 1), 2)
            growth_rate = round(created / max(window_days, 1), 2)

            # Trend detection: compare current vs prior window
            trend = "insufficient_data"
            trend_detail = ""
            if prior and prior["session_count"] >= 2 and sessions >= 2:
                prior_rate = prior["total_concepts_created"] / max(window_days, 1)
                if growth_rate > prior_rate * 1.25:
                    trend = "accelerating"
                    trend_detail = f"Growth rate {growth_rate}/day vs {round(prior_rate, 2)}/day in prior window"
                elif growth_rate < prior_rate * 0.75:
                    trend = "decelerating"
                    trend_detail = f"Growth rate {growth_rate}/day vs {round(prior_rate, 2)}/day in prior window"
                else:
                    trend = "steady"
                    trend_detail = f"Growth rate ~{growth_rate}/day (stable)"
            elif sessions >= 1:
                trend_detail = f"Only {sessions} session(s) in current window — need more data"

            return CognitiveVelocity(
                sessions_in_window=sessions,
                concepts_created_in_window=created,
                concepts_evolved_in_window=evolved,
                learning_events_in_window=learning_events,
                avg_concepts_per_session=avg_concepts,
                avg_learning_events_per_session=avg_learning,
                knowledge_growth_rate=growth_rate,
                trend=trend,
                trend_detail=trend_detail,
            )
        except Exception as e:
            logger.error(f"Cognitive velocity computation failed: {e}")
            return CognitiveVelocity()

    def _compute_where_going(self, concepts: list[Concept]) -> ActiveDirectionality:
        """Active directionality: goals, priorities, curiosity frontiers, actions.

        Synthesizes direction from multiple signals:
        - goal-type concepts (explicit goals)
        - high-confidence decisions (strategic direction)
        - weakest knowledge areas (growth frontiers)
        - question queue (curiosity gaps)
        """
        from app.models import RecommendedAction, StrategicPriority

        # 1. Extract goal-type concepts
        goals = []
        for c in concepts:
            if c.concept_type == "goal":
                linked = get_related_concepts(c.id, max_depth=1)
                goals.append(
                    GoalSummary(
                        goal_id=c.id,
                        summary=c.summary[:100],
                        priority=c.salience if c.salience else 0.5,
                        progress_indicator="in_progress",
                        linked_concepts=linked[:10],
                    )
                )
        goals.sort(key=lambda g: g.priority, reverse=True)

        # 2. Strategic priorities from high-confidence decisions + principles
        priority_types = {"decision", "principle", "constraint"}
        priority_candidates = [c for c in concepts if c.concept_type in priority_types and c.confidence >= 0.55]
        priority_candidates.sort(key=lambda c: c.confidence, reverse=True)
        strategic_priorities = [
            StrategicPriority(
                concept_id=c.id,
                summary=c.summary[:120],
                confidence=round(c.confidence, 3),
                source_type=c.concept_type or "decision",
            )
            for c in priority_candidates[:5]
        ]

        # 3. Curiosity frontier from question queue + weakest areas
        from app import question_queue

        raw_q = question_queue.get_questions(limit=3)
        frontier = [
            CuriosityFrontierItem(
                gap_description=q.get("question", ""),
                priority_score=q.get("priority", 0.0),
            )
            for q in raw_q
        ]

        # If no questions queued, synthesize frontier from weakest areas
        if not frontier:
            area_stats: dict[str, list] = {}
            for c in concepts:
                ka = (c.metadata or {}).get("knowledge_area", "unknown")
                if ka not in area_stats:
                    area_stats[ka] = []
                area_stats[ka].append(c.confidence)

            weak_areas = [
                (ka, sum(confs) / len(confs), len(confs))
                for ka, confs in area_stats.items()
                if len(confs) >= 3  # only areas with enough concepts
            ]
            weak_areas.sort(key=lambda x: x[1])  # lowest avg confidence first
            frontier = [
                CuriosityFrontierItem(
                    gap_description=f"Knowledge area '{wa[0]}' has low avg confidence ({wa[1]:.2f} across {wa[2]} concepts)",
                    priority_score=round(1.0 - wa[1], 2),
                )
                for wa in weak_areas[:3]
            ]

        # 4. Recommended actions from recent high-salience unresolved patterns
        actions = []
        recent_patterns = [
            c
            for c in concepts
            if c.concept_type in {"pattern", "observation"} and c.confidence >= 0.6 and c.salience and c.salience >= 0.5
        ]
        recent_patterns.sort(key=lambda c: (c.salience or 0) * c.confidence, reverse=True)
        for c in recent_patterns[:3]:
            actions.append(
                RecommendedAction(
                    description=f"Address: {c.summary[:80]}",
                    rationale=f"High salience ({c.salience:.2f}) + confidence ({c.confidence:.2f})",
                    priority=round((c.salience or 0.5) * c.confidence, 2),
                )
            )

        return ActiveDirectionality(
            active_goals=goals[:5],
            strategic_priorities=strategic_priorities,
            curiosity_frontier=frontier,
            next_recommended_actions=actions,
        )

    def _compute_open_threads(self) -> list:
        """Wave 5: Compute open thread summaries for orientation."""
        try:
            from app.threads import compute_open_threads

            summaries = compute_open_threads()
            return [s.model_dump() for s in summaries]
        except (ImportError, Exception) as e:
            logger.debug(f"Wave 5: open_threads skipped: {e}")
            return []

    def _compute_experiment_summary(self) -> dict | None:
        """Wave 6: Compute active experiments summary for orientation."""
        try:
            from app.experiments import load_experiments

            active_experiments = load_experiments(status=["reasoning", "completed"], limit=5)
            if not active_experiments:
                return None
            return {
                "active_count": len([e for e in active_experiments if e.status == "reasoning"]),
                "recent_completed": len([e for e in active_experiments if e.status == "completed"]),
                "types_active": list(set(e.experiment_type for e in active_experiments)),
                "concepts_produced_total": sum(len(e.concept_ids_produced) for e in active_experiments),
            }
        except (ImportError, Exception) as e:
            logger.debug(f"Wave 6: experiment_summary skipped: {e}")
            return None

    def start_session(self, context_hint: str = "", agent_id: str = "default", session_date: str | None = None) -> SessionStartResponse:
        """Session bootstrap protocol: load orientation + introspect in single call.

        SINGLE CONCEPT LOAD PATH: loads concepts once, passes to both
        introspect(summary) and orient(). No double disk scan.
        Persists session to SQLite for restart survival.
        """
        # One-time recovery: mark orphan active sessions from prior runs
        recovered_info = None
        if not self._recovery_done:
            recovered = recover_interrupted_sessions()
            self._recovery_done = True
            if recovered:
                logger.warning(
                    f"Startup recovery: {recovered} interrupted session(s) — "
                    f"knowledge from those sessions may be incomplete. "
                    f"Context compaction or session drop likely occurred."
                )
                recovered_info = {
                    "orphaned_sessions": recovered,
                    "warning": "Previous session(s) ended without pith_session_end. "
                    "Learning from those sessions may be incomplete.",
                }

        session_id = str(uuid.uuid4())[:8]
        now = _utc_now_iso()

        self._conversation_turn_called = False  # S0: reset on new session
        self._last_orientation_served_at = None  # S6.1: reset so orientation re-serves
        # RAGAS-DIAG-001 Fix 3c: Store session_date for temporal anchoring in extraction
        self._session_date = session_date
        self._retro_checked_this_session = False  # RETRO-001: allow one check per session
        self._session_concept_ids = set()  # CONCEPT_LIFECYCLE_SPEC L4: reset per session

        self.current_session = SessionInfo(
            session_id=session_id,
            started_at=now,
            status="active",
            context_hint=context_hint,
            learning_event_count=0,
            agent_id=agent_id,
        )

        # Persist to SQLite
        save_session(
            session_id=session_id,
            started_at=now,
            status="active",
            context_hint=context_hint,
            learning_event_count=0,
            agent_id=agent_id,
            model_id=getattr(self, "_current_model_id", "unknown"),
        )

        # PERF-013: Populate git cache at session start
        try:
            from app.git_cache import GitCache

            self.git_cache = GitCache()
            self.git_cache.populate()
        except Exception as e:
            logger.warning(f"GitCache init failed (non-fatal): {e}")
            self.git_cache = None

        # === SINGLE LOAD: one disk scan for everything ===
        concepts = self._load_all_concepts()

        # Pass pre-loaded concepts to both subsystems
        introspect_data = self_model_manager.introspect(mode="summary", update=True, concepts=concepts)
        orientation = self.orient(concepts=concepts)

        # --- Checkpoint auto-load: surface recent execution state ---
        active_checkpoint = None
        try:
            from app.storage import load_checkpoint

            # If context_hint looks like a task_id, try loading that specific checkpoint
            cp = load_checkpoint(task_id=context_hint, max_age_hours=48) if context_hint else None
            if not cp:
                cp = load_checkpoint(max_age_hours=24)  # fallback to most recent
            if cp:
                active_checkpoint = {
                    "task_id": cp["task_id"],
                    "status": cp["status"],
                    "description": cp["description"],
                    "active": cp["active"],
                    "next": cp["next"],
                    "blockers": cp["blockers"],
                    "updated_at": cp["updated_at"],
                    "save_count": cp["save_count"],
                }
                logger.info(f"Checkpoint auto-loaded: {cp['task_id']} (status={cp['status']})")
        except Exception as e:
            logger.warning(f"Checkpoint auto-load failed (non-fatal): {e}")

        # --- GOV: Functional cognitive bootstrap ---
        # Loads high-authority constraints/decisions, surfaces stale alerts,
        # and retrieves governance actions since last session.
        bootstrap_data = None
        try:
            from app.bootstrap import build_bootstrap
            from app.storage import _db

            with _db() as conn:
                # Find last session end time for governance action tracking
                last_ended = None
                try:
                    row = conn.execute(
                        "SELECT ended_at FROM sessions WHERE status='ended' ORDER BY ended_at DESC LIMIT 1"
                    ).fetchone()
                    if row:
                        last_ended = row[0]
                except Exception:
                    pass

                bootstrap_result = build_bootstrap(
                    conn=conn,
                    session_id=session_id,
                    is_resumption=False,
                    last_session_ended_at=last_ended,
                )
                bootstrap_data = bootstrap_result.to_dict()
                logger.info(
                    f"Bootstrap: {bootstrap_result.constraints_loaded} constraints, "
                    f"{bootstrap_result.decisions_loaded} decisions, "
                    f"{len(bootstrap_result.stale_alerts)} stale alerts"
                )
        except Exception as e:
            logger.warning(f"Bootstrap failed (non-fatal): {e}")

        logger.info(
            f"Session {session_id} started (persisted): {len(concepts)} concepts loaded, context='{context_hint[:50]}'"
        )

        response = SessionStartResponse(
            session=self.current_session,
            introspect_summary=introspect_data,
            orientation=orientation.model_dump(),
        )

        # Attach checkpoint and recovery info to response (outside Pydantic model)
        result = response.model_dump()
        if bootstrap_data:
            result["bootstrap"] = bootstrap_data
            # P3-4: Persist bootstrap marker to session record
            try:
                import json as _json

                from app.storage import update_session

                session_data = _json.dumps(
                    {
                        "session_id": session_id,
                        "bootstrap": {
                            "constraints_loaded": bootstrap_result.constraints_loaded,
                            "decisions_loaded": bootstrap_result.decisions_loaded,
                            "stale_alerts": len(bootstrap_result.stale_alerts),
                        },
                    }
                )
                update_session(session_id, data=session_data)

                # FED-013: Initialize heartbeat on session creation
                try:
                    from app.federation import get_registry

                    get_registry().update_heartbeat(session_id)
                except Exception:
                    pass  # Non-fatal — registry may not be enabled

            except Exception as e:
                logger.warning(f"P3-4: Bootstrap persistence failed (non-fatal): {e}")
        if active_checkpoint:
            result["active_checkpoint"] = active_checkpoint
        if recovered_info:
            result["recovered_sessions"] = recovered_info
            # T1: Check orphaned sessions for retroactive reflection
            # REFLECT-021: T1 gated in auto_reflection.check_orphaned_sessions_for_reflection
            try:
                from app.auto_reflection import (
                    check_orphaned_sessions_for_reflection,
                    mark_session_reflected,
                    record_reflection_event,
                )

                # Query orphaned sessions from DB
                from app.storage import _db

                with _db() as conn:
                    orphan_rows = conn.execute(
                        """SELECT id, started_at, ended_at, status,
                                  learning_event_count, data
                           FROM sessions
                           WHERE status IN ('interrupted', 'recovered')
                           ORDER BY ended_at DESC LIMIT 5"""
                    ).fetchall()
                orphan_sessions = [
                    {
                        "id": r[0],
                        "started_at": r[1],
                        "ended_at": r[2],
                        "status": r[3],
                        "learning_event_count": r[4],
                        "data": r[5],
                    }
                    for r in orphan_rows
                ]
                retro = check_orphaned_sessions_for_reflection(orphan_sessions)
                if retro:
                    result["retroactive_reflection"] = retro
                    record_reflection_event(
                        session_id=session_id,
                        trigger_type="T1_retroactive",
                        prompts_sent=len(retro.get("prompts", [])),
                        prompt_data=retro.get("prompts"),
                    )
                    # Mark the orphaned session so T1 doesn't fire again
                    mark_session_reflected(retro["orphaned_session_id"])
                    logger.info("T1 retroactive reflection attached to session_start response")
            except Exception as e:
                logger.warning(f"T1 retroactive reflection failed (non-fatal): {e}")

        # --- [dropout-recovery] C2b: Startup scan for dropout-missed sessions ---
        # Catches sessions that ended between server restarts (missed by C2 at end_session).
        # TTL-gated: only sessions ended within last 24h (A3). LIMIT 10 caps startup cost (A3).
        try:
            from app.config import get_feature_flag as _c2b_flag
            if _c2b_flag("PITH_SESSION_END_AUTOLEARN_ENABLED", True):
                from app.storage import _db as _c2b_db
                with _c2b_db() as _c2b_conn:
                    _dropout_rows = _c2b_conn.execute(
                        """SELECT id, last_previous_response
                           FROM sessions
                           WHERE status = 'ended'
                             AND learning_event_count = 0
                             AND last_previous_response IS NOT NULL
                             AND ended_at >= datetime('now', '-24 hours')
                           ORDER BY ended_at DESC
                           LIMIT 10"""
                    ).fetchall()
                for _d_id, _d_resp in _dropout_rows:
                    if not _d_resp or len(_d_resp) < 30:
                        continue
                    logger.info(
                        f"[dropout-recovery] C2b: replaying missed session {_d_id}, "
                        f"stored_len={len(_d_resp)}"
                    )
                    try:
                        _c2b_req = SessionLearnRequest(
                            user_message="",
                            assistant_response=_d_resp,
                            knowledge_area="conversation",
                            extracted_concepts=None,
                            session_id=_d_id,
                        )
                        _c2b_result = self.session_learn(_c2b_req)
                        logger.info(
                            f"[dropout-recovery] C2b: session {_d_id} captured "
                            f"{_c2b_result.learning_events} events"
                        )
                        # Clear stored response after confirmed dispatch
                        update_session(_d_id, last_previous_response=None)
                    except Exception as _c2b_item_err:
                        logger.warning(
                            f"[dropout-recovery] C2b: replay failed for {_d_id} "
                            f"(non-fatal): {_c2b_item_err}"
                        )
        except Exception as _c2b_outer_err:
            logger.warning(f"[dropout-recovery] C2b startup scan failed (non-fatal): {_c2b_outer_err}")

        result["previous_session_ended"] = recovered_info is None

        # --- RC-A: Cleanup expired resume snapshots (piggybacking on session start) ---
        try:
            cleaned = cleanup_expired_snapshots()
            if cleaned:
                logger.info(f"RC-A: Cleaned {cleaned} expired resume snapshot(s)")
        except Exception as e:
            logger.warning(f"RC-A: Snapshot cleanup failed (non-fatal): {e}")

        return result

    def end_session(self, end_request: SessionEndRequest | None = None) -> dict:
        """End current session. Optionally flush last exchange before closing.
        Flush access tracker, trigger reflection if learning_event_count >= threshold.
        Persists final state to SQLite."""
        if not self.current_session:
            return {"status": "no_active_session"}

        # --- C1: Last-exchange flush (Mechanism C) ---
        last_learn_result = None
        if end_request and end_request.previous_response and len(end_request.previous_response) >= 30:
            try:
                # Parse Tier 2 concepts
                extracted = None
                if end_request.extracted_concepts_json:
                    try:
                        parsed = json.loads(end_request.extracted_concepts_json)
                        if isinstance(parsed, list) and len(parsed) > 0:
                            extracted = parsed
                    except json.JSONDecodeError:
                        pass

                learn_req = SessionLearnRequest(
                    user_message=end_request.previous_message or "",
                    assistant_response=end_request.previous_response[:15000],
                    knowledge_area="conversation",
                    extracted_concepts=extracted,
                    session_id=self.current_session.session_id if self.current_session else None,
                )
                last_learn_result = self.session_learn(learn_req)
                logger.info(
                    f"Session end flush: {last_learn_result.learning_events} events, "
                    f"sources={last_learn_result.extraction_source_breakdown}"
                )
            except Exception as e:
                logger.warning(f"Session end flush failed (non-fatal): {e}")

        # --- [dropout-recovery] C2: Flush stored last_previous_response on auto-end ---
        # Fires when caller passed no previous_response (auto-end path delivers end_request=None),
        # session has a stored response (C1 wrote it on the last turn), and no learning has
        # occurred this session (natural guard — prevents double-dispatch per A2 amendment).
        _caller_has_response = (
            end_request
            and end_request.previous_response
            and len(end_request.previous_response) >= 30
        )
        from app.config import get_feature_flag as _c2_get_flag
        if (
            not _caller_has_response
            and self.current_session.learning_event_count == 0
            and _c2_get_flag("PITH_SESSION_END_AUTOLEARN_ENABLED", True)
        ):
            try:
                from app.storage import _db as _storage_db
                with _storage_db() as _conn:
                    _row = _conn.execute(
                        "SELECT last_previous_response FROM sessions WHERE id = ?",
                        (self.current_session.session_id,),
                    ).fetchone()
                _stored_response = _row[0] if _row else None
                if _stored_response and len(_stored_response) >= 30:
                    logger.info(
                        f"[dropout-recovery] C2: dispatching auto-learn for session "
                        f"{self.current_session.session_id}, stored_len={len(_stored_response)}"
                    )
                    _c2_learn_req = SessionLearnRequest(
                        user_message="",
                        assistant_response=_stored_response,
                        knowledge_area="conversation",
                        extracted_concepts=None,
                        session_id=self.current_session.session_id,
                    )
                    last_learn_result = self.session_learn(_c2_learn_req)
                    logger.info(
                        f"[dropout-recovery] C2: captured {last_learn_result.learning_events} events"
                    )
                    # A2 amendment: clear stored response after confirmed dispatch (prevent double-dispatch)
                    update_session(
                        self.current_session.session_id,
                        last_previous_response=None,
                    )
            except Exception as _c2_err:
                logger.warning(f"[dropout-recovery] C2 flush failed (non-fatal): {_c2_err}")

        self.current_session.ended_at = _utc_now_iso()
        self.current_session.status = "ended"

        # Persist to SQLite
        update_session(
            self.current_session.session_id,
            ended_at=self.current_session.ended_at,
            status="ended",
            learning_event_count=self.current_session.learning_event_count,
        )

        # Flush access tracker
        from app.storage import access_tracker

        flushed = access_tracker.flush()

        result = {
            "status": "ended",
            "session_id": self.current_session.session_id,
            "duration_seconds": self._session_duration(),
            "learning_events": self.current_session.learning_event_count,
            "access_records_flushed": flushed,
            "reflection_triggered": False,
            "last_exchange_flushed": last_learn_result is not None and last_learn_result.learning_events > 0,
        }

        # ARCH-002: Capture session state before clearing.
        _session_copy = self.current_session
        _concept_ids_copy = set(self._session_concept_ids) if self._session_concept_ids else set()
        _duration = self._session_duration()
        self.current_session = None

        # SUPPRESS-EMPTY-SESSIONS: Skip expensive end-of-session processing for 0-event sessions.
        # C2 dropout-recovery already ran above. Check current state (gauntlet B6: use _session_copy
        # which was captured AFTER C2 could have fired session_learn → record_learning_event).
        if _session_copy.learning_event_count == 0:
            logger.info(
                "SUPPRESS-EMPTY-SESSIONS: Skipping heavy session-end processing for %s "
                "(0 learning events after C2 recovery attempt)",
                _session_copy.session_id,
            )
            result["suppressed_empty_session"] = True
            logger.info(
                f"Session {_session_copy.session_id} ended: "
                f"0 learning events (empty session, heavy processing skipped)"
            )
            return result

        # ARCH-002: Dispatch heavy tasks as background work when event loop available.
        # Heavy phase: reflection, T3, staleness, threads, currency, checkpoint, scheduled tasks.
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():

                async def _bg():
                    try:
                        self._run_end_session_heavy(_session_copy, _concept_ids_copy, _duration)
                    except Exception as e:
                        logger.warning(f"ARCH-002: Background end_session failed: {e}")

                task = loop.create_task(_bg(), name="end_session_heavy")
                self._background_tasks.add(task)
                task.add_done_callback(self._on_bg_task_done)
                result["heavy_tasks"] = "scheduled_background"
            else:
                raise RuntimeError("no loop")
        except RuntimeError:
            # Sync fallback (CLI, tests)
            self._run_end_session_heavy(_session_copy, _concept_ids_copy, _duration)

        logger.info(
            f"Session {_session_copy.session_id} ended: "
            f"{_session_copy.learning_event_count} learning events, "
            f"heavy={result.get('heavy_tasks', 'sync')}"
        )

        return result

    def _run_end_session_heavy(self, session_copy, session_concept_ids: set, session_duration: float = 0):
        """ARCH-002: Heavy end-session tasks — runs in background when event loop available.

        Includes: reflection, T3 prompts, staleness reconciliation, thread staleness,
        currency refresh, checkpoint auto-save, and scheduled async tasks.
        All operations are best-effort (non-fatal on failure).
        """
        result = {}  # Local result dict for logging only

        # C3: Trigger reflection if enough learning AND minimum duration met
        if session_copy.learning_event_count >= REFLECTION_TRIGGER_THRESHOLD and session_duration >= 300:
            try:
                from app.reflection import reflection_engine

                reflection_engine.reflect(mode="incremental")
                result["reflection_triggered"] = True
                logger.info(f"Session end triggered reflection: {session_copy.learning_event_count} learning events")
            except Exception as e:
                logger.error(f"Session-end reflection failed: {e}")
                result["reflection_error"] = str(e)

        # --- T3: Full session-end reflection ---
        # Generate targeted reflection prompts for L1→L3 synthesis
        if session_copy:
            try:
                from app.auto_reflection import (
                    _find_session_concepts,
                    generate_session_end_reflection,
                    record_reflection_event,
                )

                session_concept_ids = _find_session_concepts(session_copy.session_id)
                t3_reflection = generate_session_end_reflection(
                    session_concept_ids=session_concept_ids,
                    learning_event_count=session_copy.learning_event_count,
                    session_duration_seconds=session_duration,
                    unprocessed_bookmarks=session_copy.reflection_bookmarks,
                )
                if t3_reflection:
                    result["reflection_prompts"] = t3_reflection
                    result["reflection_required"] = True
                    record_reflection_event(
                        session_id=session_copy.session_id,
                        trigger_type="T3_session_end",
                        prompts_sent=len(t3_reflection.get("prompts", [])),
                        prompt_data=t3_reflection.get("prompts"),
                    )
                    logger.info(f"T3 session-end reflection: {len(t3_reflection.get('prompts', []))} prompts generated")
            except Exception as e:
                logger.warning(f"T3 session-end reflection failed (non-fatal): {e}")

        # --- Trigger 2: Staleness checkpoint reconciliation ---
        # Cross-reference checkpoint done[] items against existing concepts
        # to evolve any that are stale relative to checkpoint progress.
        if self.current_session and session_copy.learning_event_count > 0:
            try:
                from app.retrieval import retrieval_engine
                from app.staleness import reconcile_checkpoint_concepts

                staleness_result = reconcile_checkpoint_concepts(
                    retrieval_engine=retrieval_engine,
                    supersede_fn=self._supersede_concept,
                )
                if staleness_result.concepts_staled > 0:
                    result["staleness_t2"] = {
                        "evolved": staleness_result.concepts_staled,
                        "details": staleness_result.details,
                        "time_ms": staleness_result.time_ms,
                    }
                    logger.info(
                        f"Staleness T2: Evolved {staleness_result.concepts_staled} stale concepts at session end"
                    )
            except Exception as e:
                logger.warning(f"Staleness T2 reconciliation failed (non-fatal): {e}")

        # --- Trigger 2b: Session-scoped concept reconciliation ---
        # CONCEPT_LIFECYCLE_SPEC L2: Detect in-session status transitions
        # (planned→committed, proposed→implemented) without checkpoint dependency.
        if self.current_session and session_copy.started_at:
            try:
                from app.retrieval import retrieval_engine
                from app.staleness import reconcile_session_concepts

                t2b_result = reconcile_session_concepts(
                    session_start_iso=session_copy.started_at,
                    retrieval_engine=retrieval_engine,
                    supersede_fn=self._supersede_concept,
                )
                if t2b_result.concepts_staled > 0:
                    result["staleness_t2b"] = {
                        "superseded": t2b_result.concepts_staled,
                        "details": t2b_result.details,
                        "time_ms": t2b_result.time_ms,
                    }
                    logger.info(
                        f"Staleness T2b: Superseded {t2b_result.concepts_staled} "
                        f"stale concepts via session reconciliation"
                    )
            except Exception as e:
                logger.warning(f"Staleness T2b reconciliation failed (non-fatal): {e}")

        # --- Thread staleness detection (Wave 5) ---
        try:
            from app.threads import detect_stale_threads

            thread_actions = detect_stale_threads()
            if thread_actions:
                result["thread_staleness"] = thread_actions
                logger.info(f"Thread staleness: {len(thread_actions)} actions taken")
        except Exception as e:
            logger.debug(f"Thread staleness detection skipped: {e}")

        # --- RESOLVE-CONTRADICTIONS: Drain contradiction signal backlog at session end ---
        # consume_graph_contradiction_signals() exists in contradiction.py but was
        # only wired to reflection. Running here ensures steady-state drainage
        # regardless of whether reflection triggered. batch_size=50 keeps latency bounded.
        try:
            from app.contradiction import consume_graph_contradiction_signals
            _contra_drain = consume_graph_contradiction_signals(batch_size=50)
            if _contra_drain.get("newly_resolved", 0) > 0:
                result["contradiction_drainage"] = {
                    "newly_resolved": _contra_drain["newly_resolved"],
                    "remaining": _contra_drain["total_events"] - _contra_drain["newly_resolved"],
                }
                logger.info(
                    "RESOLVE-CONTRADICTIONS: Drained %d contradiction signals at session end "
                    "(remaining: %d)",
                    _contra_drain["newly_resolved"],
                    _contra_drain["total_events"] - _contra_drain["newly_resolved"],
                )
        except Exception as e:
            logger.warning("RESOLVE-CONTRADICTIONS: Contradiction drainage failed (non-fatal): %s", e)

        # --- CONCEPT_LIFECYCLE_SPEC L4: Session-end currency refresh ---
        # Refresh currency_status for session-created concepts so the NEXT
        # session's orientation has fresh data.
        if session_concept_ids:
            try:
                from app.currency import batch_compute_currency
                from app.storage import _db

                with _db() as conn:
                    updated = batch_compute_currency(conn, list(session_concept_ids))
                    if updated > 0:
                        logger.info(
                            f"LIFECYCLE L4: Session-end currency refresh — "
                            f"{updated}/{len(session_concept_ids)} concepts"
                        )
            except Exception as e:
                logger.warning(f"LIFECYCLE L4: Session-end currency refresh failed: {e}")

        # --- CKPT-001: Checkpoint lifecycle management on session end ---
        try:
            from app.storage import archive_stale_checkpoints, load_checkpoint, save_checkpoint

            # Phase 1: Archive stale checkpoints (>48h no update, not from this session)
            current_sid = session_copy.session_id if session_copy else None
            archived_count = archive_stale_checkpoints(exclude_session_id=current_sid)
            if archived_count > 0:
                result["checkpoints_archived"] = archived_count
                logger.info(f"CKPT-001: Archived {archived_count} stale checkpoint(s)")

            # Phase 2: Auto-save current session's checkpoint as paused
            # NOTE: Auto-COMPLETE is handled by staleness.py:587-611 (reconcile_checkpoint_concepts)
            # which runs separately in the heavy phase with stricter guards (save_count>=2, done non-empty).
            # We only handle active→paused here.
            if session_copy and session_copy.learning_event_count > 0:
                cp = load_checkpoint(max_age_hours=24)
                if cp and cp["status"] in ("active", "planning"):
                    # CKPT-002: Compress before saving as paused
                    from app.storage import compress_checkpoint
                    compressed = compress_checkpoint(cp)
                    save_checkpoint(
                        task_id=cp["task_id"],
                        description=cp["description"],
                        status="paused",
                        done=compressed.get("done", cp["done"]),
                        active=compressed.get("active", cp["active"]),
                        next_items=compressed.get("next", cp["next"]),
                        blockers=compressed.get("blockers", cp.get("blockers")),
                        context=compressed.get("context", cp.get("context")),
                        session_id=current_sid,
                    )
                    result["checkpoint_auto_saved"] = cp["task_id"]
                    logger.info(f"CKPT-001: Checkpoint {cp['task_id']} → paused")
        except Exception as e:
            logger.warning(f"CKPT-001: Checkpoint lifecycle failed (non-fatal): {e}")

        # KA-005: Run scheduled async tasks (including ka_reclassification) on session end.
        # Previously only called via /maintenance endpoint, meaning ka_reclassification
        # never ran automatically despite having a TASK_CONFIGS entry.
        # STABILITY-013: Fire-and-forget — do NOT block event loop.
        if session_copy and result.get("reflection_triggered"):
            try:
                import asyncio

                from app.async_tasks import task_runner

                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop and loop.is_running():
                    # STABILITY-013: Fire-and-forget background task.
                    # Session-end tasks are best-effort; results not needed for response.
                    # CancelledError during shutdown is expected — finally ensures cleanup.
                    async def _run_session_end_tasks_bg():
                        bg_conn = None
                        try:
                            bg_conn = _get_connection()  # DEBT-211: use module-level import
                            bg_results = await task_runner.run_scheduled_tasks(bg_conn)
                            if bg_results:
                                logger.info(f"Background session-end tasks completed: {list(bg_results.keys())}")
                        except Exception as e:
                            logger.warning(f"Background session-end tasks failed: {e}")
                        finally:
                            if bg_conn:
                                try:  # noqa: SIM105
                                    bg_conn.close()
                                except Exception:
                                    pass

                    task = loop.create_task(_run_session_end_tasks_bg(), name="session_end_scheduled_tasks")
                    # CRITICAL: Store strong reference to prevent GC (Gauntlet A1)
                    self._background_tasks.add(task)
                    task.add_done_callback(self._on_bg_task_done)
                    result["scheduled_tasks"] = {"status": "scheduled_background"}
                else:
                    # Sync path (CLI, no event loop) — create own connection
                    sync_conn = _get_connection()  # DEBT-211: use module-level import
                    try:
                        sched_results = asyncio.run(task_runner.run_scheduled_tasks(sync_conn))
                        if sched_results:
                            result["scheduled_tasks"] = {
                                k: v.get("status", "unknown") for k, v in sched_results.items()
                            }
                            logger.info(f"Session-end scheduled tasks: {list(sched_results.keys())}")
                    finally:
                        sync_conn.close()
            except Exception as e:
                logger.warning(f"Session-end scheduled tasks failed (non-fatal): {e}")

    def record_learning_event(self):
        """Increment learning event counter. Called by propose/evolve endpoints.
        Persists updated count to SQLite."""
        if self.current_session:
            self.current_session.learning_event_count += 1
            update_session(
                self.current_session.session_id,
                learning_event_count=self.current_session.learning_event_count,
            )

    def register_implicit_learning_event(
        self, event_type: str, concept_id: str, summary: str, source: str = "implicit"
    ):
        """C3: Register a learning event from propose/evolve/link operations.

        Appends session metadata (not a storage write). Only fires if a session
        is active. Records the event and increments the learning event counter.

        Args:
            event_type: concept_proposed | concept_evolved | concepts_linked
            concept_id: The concept ID involved
            summary: Brief description (truncated to 200 chars)
            source: Event source identifier
        """
        if not self.current_session:
            return

        event = {
            "type": event_type,
            "concept_id": concept_id,
            "summary": summary[:200],
            "source": source,
            "timestamp": _utc_now_iso(),
        }

        # Store on session metadata (in-memory list)
        if not hasattr(self.current_session, "_implicit_events"):
            self.current_session._implicit_events = []
        self.current_session._implicit_events.append(event)

        # Count toward reflection threshold
        self.record_learning_event()
        logger.debug(f"C3: Implicit learning event: {event_type} {concept_id}")

    def conversation_turn(self, request: ConversationTurnRequest) -> ConversationTurnResponse:
        """Pre-response context activation. Read-only, target <50ms.

        5-step pipeline:
          S1: Query expansion — keyword extraction from message + context
          S2: TF-IDF retrieval — fetch top N×2 candidates
          S3: Activation boost — recency, co-activation, goal relevance
          S4: Graph walk — 1-hop associations for top candidates (graceful degradation)
          S5: Context assembly — trim evidence, compute graph_density
        """
        t0 = time.perf_counter()
        auto_learn_result = None
        correction_signals_response = None  # CCL §3c: populated by step 0

        # FEDERATION L1.5: Capture model provenance from request
        self._current_model_id = getattr(request, "model_id", "unknown")

        # INGEST-037 Layer 3: Extract verbatim flag from request
        _include_verbatim = getattr(request, "include_verbatim", False)

        # --- S-0: Session recovery after server restart ---
        # Server restart clears in-memory self.current_session but SQLite persists.
        # The MCP client (server.js) still has cachedSessionId and won't call
        # session_start again, so we recover the active session from DB here.
        if self.current_session is None:
            try:
                conn = _get_connection()
                row = conn.execute(
                    "SELECT id, started_at, status, context_hint, learning_event_count, agent_id "
                    "FROM sessions WHERE status = 'active' ORDER BY started_at DESC LIMIT 1"
                ).fetchone()
                if row:
                    self.current_session = SessionInfo(
                        session_id=row[0],
                        started_at=row[1],
                        status=row[2],
                        context_hint=row[3] or "",
                        learning_event_count=row[4] or 0,
                        agent_id=row[5] or "default",
                    )
                    logger.info(
                        f"S-0: Session recovered from DB after restart: {row[0]} (started={row[1]}, events={row[4]})"
                    )
                    # INFRA-006: Recover episode turn counter from DB
                    # Without this, counter resets to 0 and INSERT OR IGNORE
                    # silently drops episodes until counter catches up.
                    try:
                        max_turn = conn.execute(
                            "SELECT MAX(turn_number) FROM episodes WHERE session_id = ?", (row[0],)
                        ).fetchone()
                        if max_turn and max_turn[0] is not None:
                            self._episode_turn_counter = max_turn[0]
                            logger.info(
                                f"S-0: Episode turn counter recovered: {max_turn[0]} (session {row[0][:12]}...)"
                            )
                    except Exception as e:
                        logger.warning(f"S-0: Episode counter recovery failed (non-fatal): {e}")
            except Exception as e:
                logger.warning(f"S-0: Session recovery failed (non-fatal): {e}")

        # SESSION-001: Auto-create session if S-0 found nothing
        # Prevents tracking loss for users who skip brain_session_start
        if self.current_session is None:
            try:
                import uuid as _uuid
                _auto_sid = f"auto_{_uuid.uuid4().hex[:8]}"
                _now = _utc_now_iso()
                _auto_agent = getattr(request, "agent_id", None) or "default"
                save_session(
                    session_id=_auto_sid,
                    started_at=_now,
                    status="active",
                    context_hint="auto",
                    learning_event_count=0,
                    agent_id=_auto_agent,
                    model_id=getattr(self, "_current_model_id", "unknown"),
                )
                self.current_session = SessionInfo(
                    session_id=_auto_sid,
                    started_at=_now,
                    status="active",
                    context_hint="auto",
                    learning_event_count=0,
                    agent_id=_auto_agent,
                )
                logger.info(f"SESSION-001: Auto-created session {_auto_sid} (user skipped session_start)")
            except Exception as e:
                logger.warning(f"SESSION-001: Auto-session creation failed (non-fatal): {e}")

        # FEDERATION L1.5: Persist model_id to session record
        # PERF-005: Dirty-check — only update if model_id actually changed
        if self.current_session:
            _existing_mid = getattr(self.current_session, "model_id", None)
            if _existing_mid != self._current_model_id:
                try:
                    update_session(
                        self.current_session.session_id,
                        model_id=self._current_model_id,
                    )
                except Exception as e:
                    logger.warning(f"L1.5: model_id update failed (non-fatal): {e}")

        # --- GOV: GovernanceContext created AFTER auto-learn (PERF-020) ---
        # gov_ctx initialized to None here; created post-auto-learn so the 2000ms
        # governance budget measures governance phases only (not the ~1610ms auto-learn).
        # All pre-autolearn gov_ctx uses (health check, CCL) are guarded with `if gov_ctx:`.
        gov_ctx = None

        # --- GOV-W2: Health check & circuit breaker (budget: 2ms) ---
        # Runs periodic health checks (every 5 min). If 2+ indicators fail,
        # trips circuit breaker → all optional governance phases skipped.
        circuit_breaker_active = False
        try:
            from app.health import circuit_breaker

            _skip_health = False
            if gov_ctx:
                if not gov_ctx.check_latency_budget("health_check", 2.0, PhasePriority.OPTIONAL):
                    from app.governance_context import GOVERNANCE_HARD_ENFORCEMENT

                    if GOVERNANCE_HARD_ENFORCEMENT:
                        _skip_health = True
                    else:
                        logger.info("SOFT_SKIP: health_check would be skipped (observability mode)")
            if not _skip_health and circuit_breaker.should_check():
                health_report = circuit_breaker.check_and_update(conn=_get_connection())
                if gov_ctx and health_report.circuit_breaker_tripped:
                    gov_ctx.log_event(
                        GOV_EVENT_CIRCUIT_BREAKER_TRIPPED,
                        None,
                        {
                            "failure_count": health_report.failure_count,
                            "failures": [c.detail for c in health_report.checks if not c.healthy],
                        },
                    )
            circuit_breaker_active = circuit_breaker.is_tripped
            # WS2: Metric 8 — circuit_breaker_trip_count
            if circuit_breaker_active:
                try:
                    from app.metrics import metrics as _m8

                    _m8.record("circuit_breaker_trip_count", 1)
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"GOV-W2: Health check failed (non-fatal): {e}")

        # --- STEP 0: CCL — Validate previous response (§3c.2) ---
        # Runs BEFORE auto-learn so violations are detected before new learning occurs.
        # Compounding correction loop: validate LLM's previous response against the
        # constraint_set that was active when it was generated.
        if request.previous_response and getattr(self, "_previous_constraint_set", None):
            try:
                from app.prediction_error import _extract_terms, apply_correction_compound, validate_previous_response

                current_topic_terms = _extract_terms(request.message or "")[:20]
                validation_result = validate_previous_response(
                    previous_response=request.previous_response,
                    stored_constraint_set=self._previous_constraint_set,
                    current_topic_terms=current_topic_terms,
                )

                compounds_applied = []
                if validation_result.get("status") == "violations_found":
                    compounds_applied = apply_correction_compound(validation_result.get("violations", []))
                    logger.info(
                        f"CCL §3c: {len(validation_result['violations'])} violations detected, "
                        f"{len(compounds_applied)} compounds applied"
                    )
                    if gov_ctx:
                        gov_ctx.log_event(
                            GOV_EVENT_CCL_VIOLATIONS_DETECTED,
                            None,
                            {
                                "violations": len(validation_result["violations"]),
                                "compounds": len(compounds_applied),
                            },
                        )

                correction_signals_response = {
                    "status": validation_result.get("status", "skipped"),
                    "violations": validation_result.get("violations", []),
                    "compounds_applied": compounds_applied,
                }
            except Exception as e:
                logger.warning(f"CCL §3c: Validation failed (non-fatal): {e}")

        # --- S-1: Auto-learn from previous exchange (Tier 1 + Tier 2) ---
        # Closes the learning feedback loop structurally: instead of requiring
        # a separate session_learn call (which the AI forgets ~70% of the time),
        # piggyback learning on conversation_turn which already fires reliably.
        # Tier 1 (heuristic) gives ~60-70% capture automatically.
        # Tier 2 (extracted_concepts_json) adds client-extracted concepts for ~85%+.
        # CTX-003: Accumulate previous_response bytes for pressure scoring
        self._cumulative_response_bytes += len(request.previous_response or "")

        MAX_PREVIOUS_RESPONSE = 15000  # Attack 1: ~3,750 tokens, prevent payload bloat
        if request.previous_response and len(request.previous_response) >= 30:
            try:
                # [dropout-recovery] C1: Store last_previous_response for orphan flush safety net.
                # If session ends without a subsequent turn (auto-end path), C2 in end_session
                # reads this and dispatches auto-learn retroactively. Zero behavioral change.
                if self.current_session:
                    update_session(
                        self.current_session.session_id,
                        last_previous_response=request.previous_response[:MAX_PREVIOUS_RESPONSE],
                    )
            except Exception as _c1_err:
                logger.debug(f"[dropout-recovery] C1 store failed (non-fatal): {_c1_err}")

            # --- FEEDBACK-001: L1 Retrieval Utility Scoring ---
            # Measures whether previously activated concepts were actually used
            # in the LLM's response. Heuristic-only, target <10ms.
            try:
                from app.config import get_feature_flag as _gff_fb
                if _gff_fb("FEEDBACK_L1_ENABLED", True) and self._last_activated_concept_ids:
                    from app.feedback import score_retrieval_utility
                    _l1_scores = score_retrieval_utility(
                        activated_concept_ids=self._last_activated_concept_ids,
                        previous_response=request.previous_response[:MAX_PREVIOUS_RESPONSE],
                        session_id=self.current_session.session_id if self.current_session else None,
                        turn_number=self._episode_turn_counter,
                    )
                    if _l1_scores:
                        _used = sum(1 for s in _l1_scores if s['class'] == 'USED')
                        _unused = sum(1 for s in _l1_scores if s['class'] == 'UNUSED')
                        logger.info(
                            f"FEEDBACK-001: L1 scored {len(_l1_scores)} concepts — "
                            f"USED={_used}, UNUSED={_unused}"
                        )
                        # Record to metrics
                        try:
                            from app.metrics import metrics as _fb_metrics
                            _fb_metrics.record("l1_used_ratio", _used / len(_l1_scores) if _l1_scores else 0)
                            _fb_metrics.record("l1_unused_ratio", _unused / len(_l1_scores) if _l1_scores else 0)
                        except Exception:
                            pass
            except Exception as _fb_err:
                logger.warning(f"FEEDBACK-001: L1 scoring failed (non-fatal): {_fb_err}")

            try:
                prev_msg = request.previous_message or ""
                prev_response = request.previous_response[:MAX_PREVIOUS_RESPONSE]

                # Parse Tier 2 concepts if provided
                extracted = None
                if request.extracted_concepts_json:
                    try:
                        parsed = json.loads(request.extracted_concepts_json)
                        if isinstance(parsed, list):
                            extracted = parsed if len(parsed) > 0 else None
                            if extracted:
                                logger.info(f"S-1: Received {len(parsed)} Tier 2 concepts")
                    except json.JSONDecodeError:
                        logger.warning("S-1: extracted_concepts_json invalid JSON, Tier 1 only")

                # AGENT-001: Forward agent_id from request (request is authoritative source)
                _req_aid = getattr(request, "agent_id", "default")
                # FEDERATION L1.5: Forward model_id from request
                _req_mid = getattr(request, "model_id", "unknown")
                learn_request = SessionLearnRequest(
                    user_message=prev_msg,
                    assistant_response=prev_response,
                    knowledge_area="conversation",
                    extracted_concepts=extracted,  # None = Tier 1 only; list = Tier 1 + Tier 2
                    session_id=self.current_session.session_id if self.current_session else None,
                    agent_id=_req_aid,
                    model_id=_req_mid,
                    # RETRIEVAL-021: Forward activated concept IDs for dedup bias
                    activated_concept_ids=self._last_activated_concept_ids or None,
                )
                # PERF-FORT-2: Dispatch auto-learn to background thread.
                # Returns immediately — learning completes ~1s later.
                # Results available as _last_autolearn_result on NEXT turn.
                from app.config import get_feature_flag
                if get_feature_flag("BACKGROUND_AUTOLEARN_ENABLED", True):
                    # PERF-FORT-2: Snapshot previous result BEFORE dispatch.
                    # Background thread will overwrite _last_autolearn_result,
                    # so capture the stable value for this turn's response.
                    _snapshot_autolearn = getattr(self, '_last_autolearn_result', None)
                    _snapshot_autolearn_obj = getattr(self, '_last_autolearn_result_obj', None)
                    _snapshot_autolearn_bw = getattr(self, '_last_autolearn_budget_warnings', []) or []
                    import concurrent.futures as _cf_fort2
                    if not hasattr(self, '_learn_executor') or self._learn_executor is None:
                        self._learn_executor = _cf_fort2.ThreadPoolExecutor(
                            max_workers=1, thread_name_prefix="autolearn"
                        )
                    # A2 amendment: Check queue depth before submitting
                    _pending = self._learn_executor._work_queue.qsize() if hasattr(self._learn_executor, '_work_queue') else 0
                    if _pending > 50:
                        logger.error(f"S-1: Auto-learn queue depth={_pending}, dropping — system under extreme load")
                    else:
                        if _pending > 10:
                            logger.warning(f"S-1: Auto-learn queue depth={_pending}, learning may be falling behind")
                        # PERF-FORT-2: Defer dispatch to end of conversation_turn
                        # to avoid DB lock contention with main-path writes.
                        # Store args for deferred dispatch.
                        _deferred_autolearn_args = (learn_request, extracted, request.message, prev_msg, prev_response)
                    logger.info("S-1: Auto-learn prepared for deferred background dispatch")
                    # auto_learn_result stays None — main path uses snapshots
                    # Store snapshots in local vars for downstream consumers
                    _bg_snapshot_auto_learned = _snapshot_autolearn
                    _bg_snapshot_learn_obj = _snapshot_autolearn_obj
                    _bg_snapshot_budget_warnings = _snapshot_autolearn_bw
                else:
                    # Synchronous fallback (feature flag OFF — rollback path)
                    auto_learn_result = self.session_learn(learn_request)
                    logger.info(
                        f"S-1: Auto-learned (sync): {auto_learn_result.learning_events} events, "
                        f"sources={auto_learn_result.extraction_source_breakdown}"
                    )
                    if auto_learn_result and auto_learn_result.garbage_rejected > 0 and self._last_extraction_request_types:
                        self._suppressed_gap_types.update(self._last_extraction_request_types)

            except Exception as e:
                logger.warning(f"S-1: Auto-learn failed (non-fatal): {e}")

        # --- ARCH-D05: Periodic KA promotion (every 30 min, background) ---
        try:
            from app.taxonomy import _should_run_promotion, promote_knowledge_areas, _record_promotion_run
            from app.config import KA_PROMOTION_INTERVAL_MINUTES
            if _should_run_promotion(KA_PROMOTION_INTERVAL_MINUTES):
                def _bg_promote():
                    try:
                        transitions = promote_knowledge_areas()
                        _record_promotion_run()
                        if transitions:
                            logger.info(f"ARCH-D05: Periodic KA promotion: {len(transitions)} transitions: {transitions}")
                        else:
                            logger.debug("ARCH-D05: Periodic KA promotion ran, 0 transitions")
                    except Exception as e:
                        logger.error(f"ARCH-D05: Periodic promotion failed: {e}")
                # PERF-FORT-4: Always use background executor — sync handlers
                # have no event loop, so asyncio.get_event_loop() always raises
                # RuntimeError, causing _bg_promote() to run synchronously (+200ms).
                import concurrent.futures as _cf_fort4
                if not hasattr(self, '_learn_executor') or self._learn_executor is None:
                    self._learn_executor = _cf_fort4.ThreadPoolExecutor(
                        max_workers=1, thread_name_prefix="autolearn"
                    )
                self._learn_executor.submit(_bg_promote)
        except Exception as e:
            logger.debug(f"ARCH-D05: Promotion check skipped: {e}")

        # --- GOV: Create GovernanceContext NOW (PERF-020: after auto-learn) ---
        # Budget clock starts here — governance phases get the full 2000ms budget.
        try:
            from app.governance_context import PhasePriority, create_governance_context

            gov_ctx = create_governance_context()
        except Exception as e:
            logger.warning(f"GOV: GovernanceContext creation failed (non-fatal): {e}")

        t_autolearn = time.perf_counter()

        # --- INFRA-002: Episode recording (Memory Integrity §5.2.5) ---
        # Records per-turn metadata for audit trail. Non-critical path.
        # Uses monotonic _episode_turn_counter (not learning_event_count)
        # to guarantee UNIQUE(session_id, turn_number) — see INFRA_FIXES_AMENDMENT v3.
        # PERF-FORT-2: When background auto-learn is active, episode recording
        # is handled in _background_autolearn. Only run here in sync mode.
        _episode_id = None  # INFRA-005: pre-initialized for deferred metadata update at S2.5
        try:
            from app.config import FEATURE_FLAGS as _ep_ff, get_feature_flag as _gff_ep

            # PERF-FORT-2: Skip main-path episode recording when background handles it
            _bg_autolearn_active = _gff_ep("BACKGROUND_AUTOLEARN_ENABLED", True) and auto_learn_result is None
            if self.current_session and _ep_ff.get("EPISODES_ENABLED", False) and not _bg_autolearn_active:
                from app.episodes import record_episode

                self._episode_turn_counter += 1
                _episode_turn = self._episode_turn_counter

                _ep_concept_ids = []
                _ep_changes = []
                if auto_learn_result:
                    _ep_concept_ids = [c.concept_id for c in auto_learn_result.concepts_created]
                    _ep_changes = [
                        {"action": "created", "id": c.concept_id} for c in auto_learn_result.concepts_created
                    ] + [{"action": "evolved", "id": c.concept_id} for c in auto_learn_result.concepts_evolved]

                _episode_id = record_episode(
                    session_id=self.current_session.session_id,
                    turn_number=_episode_turn,
                    intent_summary=(request.classification_hint or "")[:500],
                    classification=(request.classification_hint or "")[:200],
                    extracted_concept_ids=_ep_concept_ids,
                    concept_changes=_ep_changes,
                    raw_user_message=request.message[:5000] if request.message else None,
                    raw_assistant_response=(request.previous_response or "")[:5000] or None,
                )
        except Exception as e:
            logger.warning(f"INFRA-002: Episode recording failed (non-fatal): {e}", exc_info=True)

        # --- RB-02: Reflection completion tracking ---
        # If auto-learn produced concepts, close the most recent open reflection entry.
        # REFLECT-020: Match by most-recent open entry (not session_id) because T1/T2
        # triggers fire at session boundaries — concepts arrive in the NEXT session,
        # so session_id match always fails, causing 88% false "timeout" rate.
        if auto_learn_result and auto_learn_result.learning_events > 0:
            try:
                from app.storage import _db

                with _db() as conn:
                    conn.execute(
                        """UPDATE reflection_tracking
                           SET completed_at = ?,
                               concepts_returned = ?,
                               reflection_quality = 'auto_closed'
                           WHERE id = (
                               SELECT id FROM reflection_tracking
                               WHERE completed_at IS NULL
                               ORDER BY created_at DESC LIMIT 1
                           )""",
                        (_utc_now_iso(), auto_learn_result.learning_events),
                    )
            except Exception as e:
                logger.debug(f"RB-02: Reflection completion tracking failed (non-fatal): {e}")

        # --- CTX Phase 0: Baseline measurement (CTX-9 gauntlet amendment) ---
        # Track how often previous_response is absent/short after turn 5+ to establish
        # the baseline mid-session amnesia rate BEFORE any intervention.
        turn_count = self.current_session.learning_event_count if self.current_session else 0
        try:
            if turn_count >= 5:
                prev_resp = request.previous_response or ""
                has_amnesia_signal = len(prev_resp) < 100
                has_empty_extraction = request.extracted_concepts_json in (None, "", "[]")
                logger.info(
                    f"CTX-P0: Baseline measurement — turn={turn_count}, "
                    f"prev_response_len={len(prev_resp)}, "
                    f"amnesia_signal={has_amnesia_signal}, "
                    f"empty_extraction={has_empty_extraction}"
                )
                try:
                    from app.metrics import metrics as _ctx_metrics

                    _ctx_metrics.record(
                        "ctx_baseline_turn_count",
                        1,
                        {
                            "amnesia_signal": str(has_amnesia_signal),
                            "empty_extraction": str(has_empty_extraction),
                            "turn_count_bucket": str(min(turn_count // 5 * 5, 50)),
                        },
                    )
                except Exception:
                    pass
        except Exception as ctx_p0_err:
            logger.warning(f"CTX-P0: Baseline measurement failed (non-fatal): {ctx_p0_err}")

        # --- CTX S-0.5: Compaction detection (CTX-2, CTX-3, CTX-5 gauntlet amendments) ---
        # Runs AFTER auto-learn (S-1) but BEFORE correction detection + retrieval.
        # Position matters: auto-learn processes previous_response first,
        # then compaction detection decides if context was likely lost.
        # If compaction detected: re-inject critical context from snapshot, skip stale auto-learn.
        compaction_was_detected = False
        try:
            from app.config import FEATURE_FLAGS as _ctx_ff

            _ctx_compaction_enabled = _ctx_ff.get("COMPACTION_DETECTION_ENABLED", False)
        except ImportError:
            _ctx_compaction_enabled = False
        if _ctx_compaction_enabled and not circuit_breaker_active:
            try:
                compaction_was_detected = self._detect_compaction(request)
                if compaction_was_detected:
                    logger.info("CTX S-0.5: Compaction detected — will re-inject context")
                    try:
                        from app.metrics import metrics as _comp_metrics

                        _comp_metrics.record(
                            "compaction_detected",
                            1,
                            {
                                "turn_count": str(turn_count),
                            },
                        )
                    except Exception:
                        pass
            except Exception as comp_err:
                logger.warning(f"CTX S-0.5: Compaction detection failed (non-fatal): {comp_err}")

        t_health = time.perf_counter()  # PERF-016: Phase A checkpoint

        # --- GOV-W2: Correction detection (budget: 2ms) ---
        # Detect if the user's current message is correcting the agent.
        # Uses 4-layer heuristics with two-signal rule.
        # If detected, record correction and trigger governance recomputation.
        correction_detected = None
        try:
            if gov_ctx:
                _budget_ok = gov_ctx.check_latency_budget("correction_detection", 2.0, PhasePriority.OPTIONAL)
                if not _budget_ok:
                    from app.governance_context import GOVERNANCE_HARD_ENFORCEMENT

                    if GOVERNANCE_HARD_ENFORCEMENT:
                        logger.info("HARD_SKIP: correction_detection skipped (budget exhausted)")
                        raise _BudgetSkip()
                    else:
                        logger.info("SOFT_SKIP: correction_detection would be skipped (observability mode)")
            from app.correction import detect_correction, identify_affected_concepts, record_correction

            # Layer 4 drift detection: pass previous turn's cached concept dicts + embedding engine
            prev_concepts = self._last_activated_concept_dicts if self._last_activated_concept_dicts else None
            try:
                from app.embedding import embedding_engine as _emb_engine
            except Exception:
                _emb_engine = None
            correction_event = detect_correction(
                message=request.message,
                activated_concepts=prev_concepts,
                embedding_engine=_emb_engine,
            )
            if correction_event:
                logger.info(
                    f"GOV-W2: Correction detected (confidence={correction_event.detection_confidence:.2f}, "
                    f"signals={len(correction_event.signals)})"
                )
                # Identify affected concepts from previous turn context
                recent_ids = self._last_activated_concept_ids[:5]

                conn = _get_connection()
                affected = identify_affected_concepts(
                    correction_event,
                    recent_ids,
                    conn=conn,
                )

                session_id = self.current_session.session_id if self.current_session else "unknown"
                record = record_correction(
                    correction_event,
                    affected,
                    session_id,
                    conn=conn,
                    gov_ctx=gov_ctx,
                )
                if record:
                    correction_detected = {
                        "correction_id": record.id,
                        "confidence": record.detection_confidence,
                        "affected_concepts": record.affected_concept_ids,
                    }
        except _BudgetSkip:
            pass  # Phase skipped due to budget exhaustion
        except Exception as e:
            logger.warning(f"GOV-W2: Correction detection failed (non-fatal): {e}")

        # --- B1: Active extraction request ---
        # PERF-FORT-2/A1: Use previous turn's snapshot when background mode active
        extraction_request = None
        try:
            _learn_src = auto_learn_result or _bg_snapshot_learn_obj
        except NameError:
            _learn_src = auto_learn_result  # No background mode — sync path only
        if _learn_src is not None:
            extraction_request = self._generate_extraction_request(
                _learn_src,
                (request.previous_message or "") + " " + (request.previous_response or ""),
                request.message,
            )

        # Lazy import to avoid circular dependency at module load
        from app.retrieval import retrieval_engine

        # --- FIX 2: Topic shift detection (budget: <1ms) ---
        # Detect if current query diverges from session context.
        # If shift detected, clear context to prevent anchoring bias.
        topic_shift_detected = self._detect_topic_shift(request.message, request.conversation_context)
        effective_context = request.conversation_context
        if topic_shift_detected:
            effective_context = None  # Fresh retrieval without session anchoring
            # Reset spreading activation to prevent prior-topic boost
            try:
                from app.config import ENHANCED_RETRIEVAL

                if ENHANCED_RETRIEVAL:
                    from app.predictive import predictive_activation

                    predictive_activation.reset_activations()
                    logger.info("TOPIC-SHIFT: Spreading activation reset for fresh retrieval")
            except Exception as e:
                logger.warning(f"TOPIC-SHIFT: Activation reset failed (non-fatal): {e}")

        t_correction = time.perf_counter()  # PERF-016: Phase A checkpoint
        _ct_phase_correction_ms = (t_correction - t_health) * 1000
        if _ct_phase_correction_ms > 50:  # MONITOR-C016: latency threshold alert
            logger.warning(
                f"MONITOR-C016: correction_detection latency {_ct_phase_correction_ms:.1f}ms exceeds 50ms threshold"
            )

        # --- S1: Build search query (budget: 0ms) ---
        # Pass full natural language to embedding search (no keyword mangling)
        search_query = request.message
        if effective_context:
            search_query = f"{request.message} {effective_context[:500]}"

        # --- S1.5: Domain activation (budget: 1ms) ---
        # Scan message for domain triggers, compute area boosts.
        # Boosts are applied to TF-IDF scores in S2 before ranking.
        try:
            from app.domains import apply_domain_boost

            domain_boost_areas, activated_domain_ids = apply_domain_boost(request.message)
        except Exception as e:
            logger.warning(f"S1.5: Domain activation failed (non-fatal): {e}")
            domain_boost_areas, activated_domain_ids = {}, []

        # --- S1.7: Cross-domain query expansion (RETRIEVAL-024) (budget: 2ms) ---
        # When domain activation or keyword scan identifies cross-domain topics,
        # inject high-authority concept summaries from related domains as query
        # expansion terms. Uses DOMAIN_BRIDGES map for domain relationships.
        DOMAIN_BRIDGES = {
            "product_strategy": ["architecture", "operations", "implementation"],
            "business_strategy": ["architecture", "operations"],
            "go_to_market": ["architecture", "operations", "implementation"],
            "architecture": ["product_strategy", "operations"],
            "operations": ["architecture", "implementation"],
            "process": ["architecture", "operations"],
            "debugging": ["architecture", "implementation"],
            "review_methodology": ["architecture", "process"],
        }
        QUERY_EXPANSION_MAX_TERMS = 3
        QUERY_EXPANSION_TERM_LEN = 40
        CROSS_DOMAIN_EXPANSION_ENABLED = os.environ.get(
            "CROSS_DOMAIN_EXPANSION_ENABLED", "true"
        ).lower() == "true"
        CROSS_DOMAIN_KEYWORDS = {
            "architecture": ["distribution", "packaging", "install", "deploy",
                             "ship", "release", "bundle", "binary", "pip install",
                             "onboarding", "quickstart"],
            "operations": ["scale", "monitor", "alert", "uptime", "sla",
                           "production", "infrastructure"],
        }

        try:
            bridge_kas = set()
            bridge_terms = []

            if CROSS_DOMAIN_EXPANSION_ENABLED and not circuit_breaker_active:
                # Primary path: domain activation detected cross-domain topics
                if domain_boost_areas:
                    activated_kas = set(domain_boost_areas.keys())
                    related_kas = set()
                    for ka in activated_kas:
                        related_kas.update(DOMAIN_BRIDGES.get(ka, []))
                    bridge_kas = related_kas - activated_kas

                # Fallback path: keyword scan when domain activation misses
                if not bridge_kas:
                    msg_lower = request.message.lower()
                    for target_ka, keywords in CROSS_DOMAIN_KEYWORDS.items():
                        if any(kw in msg_lower for kw in keywords):
                            bridge_kas.add(target_ka)

                if bridge_kas:
                    from app.storage import get_high_authority_concepts_by_ka

                    for ka in list(bridge_kas)[:3]:  # Cap domain fan-out
                        ha_concepts = get_high_authority_concepts_by_ka(
                            ka, limit=QUERY_EXPANSION_MAX_TERMS
                        )
                        for c in ha_concepts:
                            terms = c["summary"][:QUERY_EXPANSION_TERM_LEN].strip()
                            bridge_terms.append(terms)

                    if bridge_terms:
                        expansion = " ".join(bridge_terms)
                        search_query = f"{search_query} {expansion}"
                        logger.info(
                            f"S1.7: Cross-domain expansion: +{len(bridge_terms)} terms "
                            f"from {bridge_kas}"
                        )
        except Exception as e:
            logger.warning(f"S1.7: Cross-domain expansion failed (non-fatal): {e}")

        # --- S7: Proportional concept count (budget: 0ms) ---
        # Short messages (greetings, brief queries) don't need full retrieval.
        # Server-side reduction of data surface area reduces listing temptation.
        # Note: ambient principles, always-activate, and firmware are injected
        # AFTER retrieval and are NOT affected by this cap.
        # RETRIEVAL-S7-BYPASS: Questions bypass the cap — short questions need
        # full retrieval to find relevant concepts (validated +1.0 EM, +1.9 F1).
        SHORT_MESSAGE_THRESHOLD = 30  # chars
        SHORT_MESSAGE_MAX_CONCEPTS = 3
        _INTERROGATIVE_PREFIXES = frozenset({
            'what', 'who', 'where', 'when', 'how', 'why', 'which',
            'does', 'did', 'is', 'are', 'was', 'were', 'can', 'could',
            'will', 'would', 'has', 'have', 'do', 'should', 'shall',
        })
        effective_max_concepts = request.max_concepts
        _msg_stripped = request.message.strip()
        _is_question = (
            '?' in _msg_stripped
            or _msg_stripped.split()[0].lower().rstrip('?.,!') in _INTERROGATIVE_PREFIXES
            if _msg_stripped else False
        )
        if len(_msg_stripped) <= SHORT_MESSAGE_THRESHOLD and not _is_question:
            effective_max_concepts = min(request.max_concepts, SHORT_MESSAGE_MAX_CONCEPTS)
            logger.info(
                f"S7: Short message ({len(_msg_stripped)} chars) — "
                f"capping retrieval to {effective_max_concepts} concepts"
            )
        elif len(_msg_stripped) <= SHORT_MESSAGE_THRESHOLD and _is_question:
            logger.info(
                f"S7: Short message ({len(_msg_stripped)} chars) but detected as question — "
                f"keeping full retrieval ({effective_max_concepts} concepts)"
            )

        # RETRIEVAL-BUDGET-FLOOR-001: Minimum concept budget for question queries.
        # Benchmark showed max_concepts < 12 kills SH recall (server returns 28-37
        # concepts for factual questions; capping below 12 discards critical context).
        _QUESTION_BUDGET_FLOOR = 12
        if _is_question and effective_max_concepts < _QUESTION_BUDGET_FLOOR:
            logger.info(
                f"RETRIEVAL-BUDGET-FLOOR-001: Question budget floor "
                f"{effective_max_concepts} → {_QUESTION_BUDGET_FLOOR}"
            )
            effective_max_concepts = _QUESTION_BUDGET_FLOOR

        # --- CONFIG-001: Complexity-based retrieval scaling ---
        # Multi-hop questions and entity-rich queries need more retrieval slots.
        # Default max_concepts=8 is tuned for simple queries. Complex queries
        # (multi-hop, proper nouns, relationship questions) benefit from 2x budget.
        _complexity_boosted = False
        try:
            from app.entity_chain import EntityChainRetriever
            _ec_test = EntityChainRetriever(db_path='/dev/null')
            _entities = _ec_test._extract_entities(search_query)
            _is_complex = (
                len(_entities) >= 2  # Multi-entity query
                or any(kw in search_query.lower() for kw in (
                    'capital', 'country', 'language', 'citizen', 'founder',
                    'headquarter', 'born', 'married', 'works at', 'lives in',
                ))
            )
            if _is_complex and effective_max_concepts < 20:
                _old_max = effective_max_concepts
                # RETRIEVAL-051/F2: Graduated boost proportional to brain size.
                # Avoids cold-start gap where small brains get zero boost.
                try:
                    from app.embedding import embedding_engine as _cfg_emb
                    _cfg_brain_size = _cfg_emb.index_size
                except Exception:
                    _cfg_brain_size = 200  # safe fallback: full boost
                _min_brain = int(os.environ.get('PITH_COMPLEXITY_MIN_BRAIN', '100'))
                _boost_factor = min(2.0, max(1.0, _cfg_brain_size / _min_brain)) if _min_brain > 0 else 2.0
                effective_max_concepts = min(20, int(effective_max_concepts * _boost_factor))
                _complexity_boosted = True
                logger.info(
                    f"CONFIG-001: Complex query ({len(_entities)} entities, "
                    f"brain={_cfg_brain_size}) — boost {_boost_factor:.1f}x, "
                    f"max_concepts {_old_max} -> {effective_max_concepts}"
                )
        except Exception as _cfg_e:
            logger.debug(f"CONFIG-001: Complexity detection failed (non-fatal): {_cfg_e}")

        # --- S2: Embedding retrieval (budget: 25ms) ---
        # Fetch 2× max_concepts to leave room for filtering/reranking
        # Uses lightweight search path — skips full concept preload scan
        if gov_ctx:
            gov_ctx.check_latency_budget("S2_retrieval", 25.0, PhasePriority.REQUIRED)
        _req_agent_id = getattr(request, "agent_id", "default")
        _req_scope = getattr(request, "scope", "global")
        _t_search_lw_start = time.perf_counter()  # PERF-017: search_lightweight sub-metric

        # --- RETRIEVAL-060: Adaptive retrieval router ---
        # Classifies query to dynamically select retrieval strategies.
        # When enabled, overrides static env var checks for multihop/entity-chain.
        # When disabled, falls through to existing static behavior (zero change).
        _adaptive_config = None
        try:
            from app.retrieval_router import get_retrieval_config
            _adaptive_config = get_retrieval_config(request.message or search_query)
            if _adaptive_config and _adaptive_config.is_adaptive:
                # Apply top_k multiplier from router (capped at 30 per GAUNTLET A1)
                if _adaptive_config.top_k_multiplier > 1.0:
                    _old_eff = effective_max_concepts
                    effective_max_concepts = min(30, int(effective_max_concepts * _adaptive_config.top_k_multiplier))
                    logger.info(
                        f"RETRIEVAL-060: top_k boost {_adaptive_config.top_k_multiplier}x "
                        f"({_old_eff} -> {effective_max_concepts}) "
                        f"signals={_adaptive_config.signals}"
                    )
        except Exception as _ar_e:
            logger.debug(f"RETRIEVAL-060: Router init failed (non-fatal): {_ar_e}")

        # --- RETRIEVAL-037a: Multi-hop retrieval gate ---
        _multihop_enabled = os.environ.get("PITH_MULTIHOP_ENABLED", "").lower() in ("true", "1")
        # RETRIEVAL-060: Router can also enable multihop dynamically
        if _adaptive_config and _adaptive_config.use_multihop:
            _multihop_enabled = True
        _multihop_used = False
        _multihop_clauses: list[str] = []
        _mh_retriever = None

        if _multihop_enabled:
            try:
                from app.retrieval_multihop import ProductionMultiHopRetriever
                if ProductionMultiHopRetriever.is_multihop_question(search_query):
                    _mh_retriever = ProductionMultiHopRetriever(
                        retrieval_engine,
                        max_hops=int(os.environ.get("PITH_MULTIHOP_MAX_HOPS", "3")),
                        per_hop_k=int(os.environ.get("PITH_MULTIHOP_PER_HOP_K", "10")),
                        min_relevance=float(os.environ.get("PITH_MULTIHOP_MIN_RELEVANCE", "0.15")),
                        budget_ms=float(os.environ.get("PITH_MULTIHOP_BUDGET_MS", "150")),
                    )
                    search_results = _mh_retriever.retrieve(
                        search_query, effective_max_concepts * 2,
                        agent_id=_req_agent_id if _req_agent_id != "default" else None,
                        scope=_req_scope,
                    )
                    _multihop_used = True
                    _multihop_clauses = getattr(_mh_retriever, 'decomposed_clauses', [])
                    logger.info(
                        f"RETRIEVAL-037a: Multihop retrieval used — "
                        f"{len(_multihop_clauses)} clauses, {len(search_results)} results"
                    )
            except Exception as e:
                logger.warning(f"RETRIEVAL-037a: Multihop gate failed (falling back): {e}")

        if not _multihop_used:
            search_results = retrieval_engine.search_lightweight(
                search_query,
                top_k=effective_max_concepts * 2,
                min_confidence=0.0,
                agent_id=_req_agent_id if _req_agent_id != "default" else None,
                scope=_req_scope,
                include_deprecated=getattr(request, "include_deprecated", False),  # RETRIEVAL-056
            )
        _t_search_lw_end = time.perf_counter()  # PERF-017: search_lightweight sub-metric

        # --- RETRIEVAL-RERANK: Two-stage LLM reranker ---
        # When enabled, reranks embedding search results using LLM scoring.
        # Runs BEFORE top_results slicing to give reranker the full candidate pool.
        # Feature-gated. Adds ~300-500ms latency (Haiku).
        # [GAUNTLET A2: Insert before RETRIEVAL-046 so reranker sees raw embedding scores]
        _RERANKER_ENABLED = os.environ.get('PITH_RERANKER', '').lower() in ('true', '1')
        if _RERANKER_ENABLED and search_results:
            try:
                from app.reranker import rerank_results
                _rr_stage1_k = int(os.environ.get('PITH_RERANKER_STAGE1_K', '40'))
                _rr_candidates = search_results[:_rr_stage1_k]
                # [GAUNTLET A3: Use raw user message, not firmware-decorated search_query]
                search_results = rerank_results(request.message or search_query, _rr_candidates)
                logger.info(f'RETRIEVAL-RERANK: Reranked {len(_rr_candidates)} candidates')
            except Exception as _rr_e:
                logger.warning(f'RETRIEVAL-RERANK: Reranker failed (non-fatal): {_rr_e}')

        # --- RETRIEVAL-046: Chain-guided relevance demotion ---
        # When multihop decomposes a query into clauses, demote search results
        # whose summaries share no key terms with any clause. This reduces noise
        # that drowns out on-chain answers (63% of consumer test failures).
        # Soft demotion (score *= factor) instead of hard pruning to avoid
        # attention-shift regressions seen in benchmark v5.
        _chain_demotion_enabled = os.environ.get(
            "PITH_CHAIN_DEMOTION", ""
        ).lower() in ("true", "1")
        try:
            _chain_demotion_factor = float(
                os.environ.get("PITH_CHAIN_DEMOTION_FACTOR", "0.6")
            )
            _chain_demotion_factor = max(0.0, min(1.0, _chain_demotion_factor))
        except (ValueError, TypeError):
            _chain_demotion_factor = 0.6
        try:
            _chain_demotion_min_clauses = int(
                os.environ.get("PITH_CHAIN_DEMOTION_MIN_CLAUSES", "2")
            )
        except (ValueError, TypeError):
            _chain_demotion_min_clauses = 2
        if (
            _chain_demotion_enabled
            and search_results
        ):
            # RETRIEVAL-047: Entity + identifier extraction (replaces generic word splitting)
            # Extract named entities (proper nouns) and technical identifiers from:
            #   (a) the original search query
            #   (b) hop context summaries (discovered entities from chain traversal)
            # Entities are 30-60x more selective than generic clause terms (E2 evidence).
            _ENTITY_STOPWORDS = {
                "what", "when", "where", "who", "which", "how", "why", "does",
                "did", "is", "are", "was", "were", "has", "have", "had", "the",
                "a", "an", "in", "on", "at", "to", "for", "of", "with", "by",
                "from", "and", "or", "not", "be", "been", "being", "that", "this",
            }
            # Technical identifier pattern: PERF-016, FC_mh_64k, RETRIEVAL-045, v5
            _IDENT_RE = _re.compile(r'[A-Z][A-Za-z0-9_-]{2,}(?:\d+)?')
            # Hyphenated compound terms: zero-protocol, entity-chain
            _COMPOUND_RE = _re.compile(r'[a-z]+-[a-z]+', _re.IGNORECASE)

            def _extract_chain_entities(text: str) -> set[str]:
                """Extract named entities + technical identifiers from text."""
                ents: set[str] = set()
                # 1. Technical identifiers (FC_mh_64k, PERF-016, RETRIEVAL-045)
                for m in _IDENT_RE.finditer(text):
                    token = m.group()
                    if token.lower() not in _ENTITY_STOPWORDS and len(token) > 2:
                        ents.add(token.lower())
                # 2. Proper noun runs (capitalized word sequences)
                words = text.replace('"', ' ').replace("'", " ").split()
                current_entity: list[str] = []
                for w in words:
                    clean = w.strip("?,!.;:()")
                    if not clean:
                        if current_entity:
                            ent = " ".join(current_entity)
                            if ent.lower() not in _ENTITY_STOPWORDS and len(ent) > 2:
                                ents.add(ent.lower())
                            current_entity = []
                        continue
                    is_proper = (
                        clean[0].isupper()
                        and clean.lower() not in _ENTITY_STOPWORDS
                        and len(clean) > 1
                    )
                    is_number = clean[0].isdigit() and current_entity
                    if is_proper or is_number:
                        current_entity.append(clean)
                    else:
                        if current_entity:
                            ent = " ".join(current_entity)
                            if ent.lower() not in _ENTITY_STOPWORDS and len(ent) > 2:
                                ents.add(ent.lower())
                            current_entity = []
                if current_entity:
                    ent = " ".join(current_entity)
                    if ent.lower() not in _ENTITY_STOPWORDS and len(ent) > 2:
                        ents.add(ent.lower())
                # 3. Hyphenated compounds (zero-protocol, entity-chain)
                for m in _COMPOUND_RE.finditer(text):
                    token = m.group()
                    if len(token) > 5 and token.lower() not in _ENTITY_STOPWORDS:
                        ents.add(token.lower())
                return ents

            # Extract from query + hop context summaries
            _chain_entities: set[str] = _extract_chain_entities(search_query)
            _hop_summaries = getattr(_mh_retriever, 'hop_context_summaries', [])
            for _hop_summary in _hop_summaries:
                if isinstance(_hop_summary, str) and _hop_summary:  # A2: type guard
                    _chain_entities |= _extract_chain_entities(_hop_summary)

            # A1: Safety cap — limit to 20 most selective (longest) entities
            if len(_chain_entities) > 20:
                _chain_entities = set(sorted(_chain_entities, key=len, reverse=True)[:20])

            if _chain_entities:
                # Pre-compile word-boundary patterns for each entity
                _entity_patterns = []
                for ent in _chain_entities:
                    try:
                        _entity_patterns.append(
                            _re.compile(r'\b' + _re.escape(ent) + r'\b', _re.IGNORECASE)
                        )
                    except _re.error:
                        continue  # Skip malformed patterns

                _demoted_count = 0
                _pruned_ids: set = set()
                for sr in search_results:
                    summary = (getattr(sr, "summary", "") or "")
                    has_overlap = any(
                        pat.search(summary) for pat in _entity_patterns
                    )
                    if not has_overlap:
                        if _chain_demotion_factor == 0.0:
                            # Hard pruning: remove off-chain concepts entirely
                            _pruned_ids.add(getattr(sr, 'concept_id', id(sr)))
                        else:
                            sr.relevance_score *= _chain_demotion_factor
                        _demoted_count += 1

                # Safety floor (A1): if hard pruning removes >90% of results,
                # fall back to soft demotion to avoid empty context
                _total_before_prune = len(search_results)
                if _pruned_ids and len(_pruned_ids) > 0.9 * _total_before_prune:
                    logger.warning(
                        f"RETRIEVAL-048: Hard prune safety floor triggered — "
                        f"would remove {len(_pruned_ids)}/{_total_before_prune} "
                        f"results. Falling back to soft demotion (factor=0.6)."
                    )
                    _pruned_ids.clear()
                    for sr in search_results:
                        summary = (getattr(sr, "summary", "") or "")
                        has_overlap = any(
                            pat.search(summary) for pat in _entity_patterns
                        )
                        if not has_overlap:
                            sr.relevance_score *= 0.6

                # Remove hard-pruned concepts, then re-sort
                if _pruned_ids:
                    search_results = [
                        sr for sr in search_results
                        if getattr(sr, 'concept_id', id(sr)) not in _pruned_ids
                    ]
                search_results.sort(
                    key=lambda r: r.relevance_score, reverse=True
                )
                logger.info(
                    f"RETRIEVAL-048: Chain demotion: {_demoted_count}/{_total_before_prune} "
                    f"{'pruned' if _chain_demotion_factor == 0.0 else 'demoted'} "
                    f"(factor={_chain_demotion_factor}, "
                    f"chain_entities={len(_chain_entities)}, "
                    f"multihop={'yes' if _multihop_used else 'no'}, "
                    f"entities={sorted(_chain_entities)[:5]})"
                )

        # --- S2 addendum: Apply domain boosts to search results ---
        if domain_boost_areas and search_results:
            for result in search_results:
                area = getattr(result, "knowledge_area", None)
                if area and area in domain_boost_areas:
                    result.relevance_score += domain_boost_areas[area]
            # Re-sort by boosted scores
            search_results.sort(key=lambda r: r.relevance_score, reverse=True)

        # --- RETRIEVAL-060b: Recency boost (adaptive router) ---
        # [GAUNTLET A2]: Placed AFTER chain demotion so demotion acts on raw
        # embedding scores. Recency boost applied to post-demotion results.
        # BENCH-INFRA-007: Skip recency boost in benchmark mode (time-dependent scoring).
        _benchmark_recency_skip = os.environ.get("PITH_BENCHMARK_MODE", "").lower() in ("true", "1")
        if _adaptive_config and _adaptive_config.recency_boost > 0.0 and search_results and not _benchmark_recency_skip:
            try:
                from app.retrieval_router import apply_recency_boost
                search_results = apply_recency_boost(
                    search_results,
                    boost=_adaptive_config.recency_boost,
                    max_age_days=30.0,
                )
                logger.info(
                    f"RETRIEVAL-060b: Recency boost {_adaptive_config.recency_boost} "
                    f"applied to {len(search_results)} results"
                )
            except Exception as _rb_e:
                logger.debug(f"RETRIEVAL-060b: Recency boost failed (non-fatal): {_rb_e}")

        # --- THREAD-001: Active thread concept boost (budget: <5ms) ---
        # Concepts linked to active narrative threads get a small relevance boost.
        # Applied BEFORE final sort so thread-relevant concepts surface higher.
        # GA-009: Query is a single indexed JOIN — cache if profiling shows >5ms.
        if search_results:
            try:
                from app.threads import get_active_thread_concept_ids

                thread_concept_ids = get_active_thread_concept_ids()
                if thread_concept_ids:
                    for result in search_results:
                        if result.concept_id in thread_concept_ids:
                            result.relevance_score += 0.05
                    search_results.sort(key=lambda r: r.relevance_score, reverse=True)
            except Exception:
                pass  # Fail open — thread boost is non-critical

        # --- S3: Activation boost (budget: 5ms) ---
        # Already applied inside retrieval_engine.search() via predictive_activation
        # Results are already boosted and re-sorted. No extra step needed here.

        # --- S2.5: Question classification (budget: 1ms) ---
        # Classify the user's question to determine if supplementary retrieval is needed.
        # Uses fast keyword heuristics, no LLM calls. Non-fatal on failure.
        question_classification = None
        inferred_dates = None
        try:
            from app.router import ENABLE_COGNITIVE_ROUTER, classify_question, infer_dates

            if ENABLE_COGNITIVE_ROUTER:
                # Tier 2: Client classification hint bypasses regex classifier
                VALID_HINTS = {
                    "temporal_activity",
                    "temporal_state",
                    "causal_backward",
                    "causal_forward",
                    "evolution",
                    "compositional",
                    "contradiction",
                }
                hint = getattr(request, "classification_hint", None)
                if hint and isinstance(hint, str):  # A-C5: type check
                    hint = hint.strip().lower()  # A-C12: normalize
                else:
                    hint = None

                if hint and hint in VALID_HINTS:
                    question_classification = {
                        "classification": hint,
                        "confidence": 0.95,
                        "input_source": "client_hint",
                    }
                    logger.info(f"S2.5: Using client hint: {hint}")
                elif hint:
                    # A-C5: Log warning for invalid hints (debugging aid)
                    logger.warning(f"S2.5: Invalid classification_hint '{hint}', falling back to regex")
                    question_classification = classify_question(
                        message=search_query,
                        user_raw_message=request.message,
                    )
                else:
                    question_classification = classify_question(
                        message=search_query,
                        user_raw_message=request.message,
                    )
                inferred_dates = infer_dates(request.message)
                if question_classification.get("classification") != "general":
                    logger.info(
                        f"S2.5: Classified as {question_classification['classification']} "
                        f"(confidence={question_classification.get('confidence', 0):.2f})"
                    )
        except Exception as e:
            logger.warning(f"S2.5: Question classification failed (non-fatal): {e}")

        # --- S2.5b: Date auto-upgrade (A-M6, RETRIEVAL-029) ---
        # If a date was found but classification is general, auto-upgrade to
        # temporal_state — BUT only if the message contains temporal memory query
        # phrasing. "What is the weather like today?" has a date but no temporal
        # memory intent. "What did we discuss today?" has both.
        # _TEMPORAL_MEMORY_QUERY defined at module level (RETRIEVAL-029 fix)
        if (
            question_classification
            and question_classification.get("classification") == "general"
            and inferred_dates
            and inferred_dates.get("since")
            and _TEMPORAL_MEMORY_QUERY.search(request.message or "")
        ):
            question_classification = {
                "classification": "temporal_state",
                "confidence": 0.60,
                "input_source": "date_auto_upgrade",
            }
            logger.info("S2.5b: Auto-upgraded to temporal_state (date + memory query pattern)")

        # --- INFRA-005: Backfill episode metadata from S2.5 classification ---
        # Episode was recorded early (before S2.5). Now that classification
        # is available, update the episode with server-derived metadata.
        if _episode_id and question_classification:
            try:
                from app.episodes import update_episode_metadata

                _ep_classification = question_classification.get("classification", "")
                _ep_confidence = question_classification.get("confidence", 0)
                _ep_source = question_classification.get("input_source", "unknown")

                # Build intent summary from classification + message snippet
                # Format: "{classification} (conf={confidence}, src={source}): {message_prefix}"
                _ep_msg_prefix = (request.message or "")[:200].strip()
                _ep_intent = (
                    (f"{_ep_classification} (conf={_ep_confidence:.2f}, src={_ep_source}): {_ep_msg_prefix}")
                    if _ep_classification != "general"
                    else _ep_msg_prefix
                )

                update_episode_metadata(
                    episode_id=_episode_id,
                    intent_summary=_ep_intent[:500],
                    classification=_ep_classification,
                )
            except Exception as e:
                logger.warning(f"INFRA-005: Episode metadata backfill failed (non-fatal): {e}")

        # --- S2.6: Conditional temporal boost + date filter (budget: 3ms) ---
        # For temporal questions, boost recently-modified concepts in search results,
        # then filter to date range if inferred_dates are available.
        # RETRIEVAL-023: Added Phase 2 date range filter with graceful degradation.
        try:
            if question_classification and question_classification.get("classification", "").startswith("temporal"):
                from app.temporal import temporal_boost

                # Phase 1: Apply recency boost (existing behavior)
                concept_cache_s26 = {}  # Cache concepts for reuse in filter phase
                for sr in search_results:
                    concept_data = load_concept(sr.concept_id)
                    if concept_data:
                        concept_cache_s26[sr.concept_id] = concept_data
                        if concept_data.updated_at:
                            boost_result = temporal_boost(concept_data.updated_at)
                            if boost_result.get("status") == "success":
                                multiplier = boost_result.get("boost_multiplier", 1.0)
                                if multiplier > 1.0:
                                    sr.relevance_score = round(sr.relevance_score * multiplier, 4)
                search_results.sort(key=lambda x: x.relevance_score, reverse=True)
                logger.info("S2.6: Applied temporal boost to search results")

                # Phase 2: Date range filter (RETRIEVAL-023)
                filtered = list(search_results)  # DEBT-215: defensive default for outcome persistence
                original_count = len(search_results)
                if inferred_dates and (inferred_dates.get("since") or inferred_dates.get("until")):
                    since = inferred_dates.get("since", "")
                    until = inferred_dates.get("until", "")
                    filtered = []
                    for sr in search_results:
                        cd = concept_cache_s26.get(sr.concept_id)
                        if cd and cd.created_at:
                            ts = cd.created_at[:10]  # Temporal anchor: when knowledge originated (RETRIEVAL-029)
                            if since and ts < since[:10]:
                                continue
                            if until and ts >= until[:10]:
                                continue
                        filtered.append(sr)
                    # Graceful degradation: keep all results if filter is too aggressive
                    original_count = len(search_results)
                    if len(filtered) >= 3:
                        search_results = filtered
                        logger.info(f"S2.6: Temporal filter applied: {len(filtered)} of {original_count} concepts in range")
                    else:
                        logger.info(f"S2.6: Temporal filter too aggressive ({len(filtered)} survivors), keeping all {len(search_results)} results with boost only")

                # RETRIEVAL-029: Persist temporal filter outcome for observability
                if _episode_id:
                    import json as _json
                    _filter_outcome = {"action": "skipped", "before": len(search_results), "after": len(search_results)}
                    if inferred_dates and (inferred_dates.get("since") or inferred_dates.get("until")):
                        if len(filtered) >= 3:
                            _filter_outcome = {"action": "filtered", "before": original_count, "after": len(search_results)}
                        else:
                            _filter_outcome = {"action": "fallback", "before": original_count, "after": original_count}
                    try:
                        from app.episodes import update_episode_metadata as _update_ep
                        _update_ep(episode_id=_episode_id, temporal_filter_outcome=_json.dumps(_filter_outcome))
                    except Exception:
                        logger.warning("RETRIEVAL-029: temporal_filter_outcome persistence failed (non-fatal)", exc_info=True)

                # Phase 3: Enrich temporal query context with observed dates (INGEST-016)
                # Annotates search results with the date each concept was first observed.
                # Uses created_at as temporal anchor (valid_from = created_at per Fix 1).
                # This gives the answer-generation LLM date info for temporal arithmetic
                # (e.g., "how many weeks ago did I attend X?").
                _temporal_annotations = {}
                for sr in search_results:
                    cd = concept_cache_s26.get(sr.concept_id)
                    if cd and cd.created_at:
                        _temporal_annotations[sr.concept_id] = cd.created_at[:10]
                if _temporal_annotations:
                    logger.info(f"S2.6: Temporal annotations added for {len(_temporal_annotations)} concepts")
                
                # Store annotations for use in answer formatting (downstream in conversation_turn)
                self._s26_temporal_annotations = _temporal_annotations
        except Exception as e:
            logger.warning(f"S2.6: Temporal boost/filter failed (non-fatal): {e}")

        t_retrieval = time.perf_counter()  # PERF-016: Phase A checkpoint

        # --- S4: 1-hop graph walk (budget: 8ms) ---
        # For top candidates, fetch direct associations. Graceful degradation:
        # if concept has no edges (225 of 249 are orphans), skip silently.
        _skip_graph_walk = False
        if circuit_breaker_active:
            _skip_graph_walk = True
            logger.info("CIRCUIT_BREAKER_SKIP: S4_graph_walk skipped")
            if gov_ctx:
                gov_ctx.phases_skipped.append("S4_graph_walk")
        if gov_ctx:
            if not gov_ctx.check_latency_budget("S4_graph_walk", 8.0, PhasePriority.OPTIONAL):
                from app.governance_context import GOVERNANCE_HARD_ENFORCEMENT

                if GOVERNANCE_HARD_ENFORCEMENT:
                    _skip_graph_walk = True
                    logger.info("HARD_SKIP: S4_graph_walk skipped (budget exhausted)")
                else:
                    logger.info("SOFT_SKIP: S4_graph_walk would be skipped (observability mode)")
        top_results = search_results[:effective_max_concepts]

        # --- RETRIEVAL-048: Coverage-triggered re-query ---
        # If initial retrieval returned sparse/weak results, expand budget and re-search.
        # Benchmark version used hard entity-overlap pruning; production uses softer
        # signal: if top_score < threshold AND we have budget, re-query with 3x slots.
        _requery_fired = False
        try:
            _REQUERY_THRESHOLD = float(os.environ.get('PITH_REQUERY_THRESHOLD', '0.25'))
            _top_score = max((r.relevance_score for r in top_results), default=0.0)
            _sparse = len(top_results) < 3 or _top_score < _REQUERY_THRESHOLD
            if _sparse and not _skip_graph_walk:
                # RETRIEVAL-051/F1: Cap re-query budget proportional to brain size.
                # Prevents candidate pool dilution on small brains.
                _rq_uncapped = effective_max_concepts * 3
                try:
                    from app.embedding import embedding_engine as _rq_emb
                    _rq_brain_size = _rq_emb.index_size
                except Exception:
                    _rq_brain_size = 10000  # safe fallback: no cap
                _rq_cap_ratio = float(os.environ.get('PITH_REQUERY_CAP_RATIO', '0.3'))
                _rq_budget = min(_rq_uncapped, max(effective_max_concepts, int(_rq_brain_size * _rq_cap_ratio)))
                if _rq_budget < _rq_uncapped:
                    logger.info(
                        f"RETRIEVAL-051: Re-query capped {_rq_uncapped} -> {_rq_budget} "
                        f"(brain={_rq_brain_size}, ratio={_rq_cap_ratio})"
                    )
                logger.info(
                    f"RETRIEVAL-048: Sparse results (top_score={_top_score:.3f}, "
                    f"count={len(top_results)}) — re-querying with budget={_rq_budget}"
                )
                _rq_results = retrieval_engine.search_lightweight(
                    search_query,
                    top_k=_rq_budget,
                    min_confidence=0.0,
                    agent_id=_req_agent_id if _req_agent_id != 'default' else None,
                    scope=_req_scope,
                )
                # Merge: union by concept_id, prefer higher relevance
                _existing = {r.concept_id: r for r in top_results}
                _rq_added = 0
                for rr in _rq_results:
                    if rr.concept_id not in _existing:
                        _existing[rr.concept_id] = rr
                        _rq_added += 1
                if _rq_added > 0:
                    top_results = sorted(
                        _existing.values(),
                        key=lambda x: x.relevance_score,
                        reverse=True,
                    )[:effective_max_concepts * 2]  # Allow expanded pool for downstream filtering
                    _requery_fired = True
                    logger.info(f"RETRIEVAL-048: Re-query added {_rq_added} concepts (total={len(top_results)})")
        except Exception as _rq_e:
            logger.warning(f"RETRIEVAL-048: Re-query failed (non-fatal): {_rq_e}")

        # --- RETRIEVAL-066v2: LLM decomposition for compositional questions ---
        # Fires AFTER RETRIEVAL-048 if coverage is still weak. Uses LLM to
        # detect compositional questions and run independent sub-queries.
        _decomp_enabled = os.environ.get(
            'PITH_QUERY_DECOMPOSITION', ''
        ).lower() in ('true', '1')
        if _decomp_enabled and not _skip_graph_walk:
            try:
                _066_top_score = max(
                    (r.relevance_score for r in top_results), default=0.0
                )
                _066_sparse = len(top_results) < 3 or _066_top_score < 0.25
                # Also trigger if router classified as compositional
                _066_is_compositional = (
                    question_classification
                    and question_classification.get("classification") == "compositional"
                )
                if _066_sparse or _066_is_compositional:
                    _sub_queries = _decompose_query_llm(search_query)
                    if _sub_queries and len(_sub_queries) > 1:
                        logger.info(
                            f"RETRIEVAL-066v2: Decomposing into {len(_sub_queries)} "
                            f"sub-queries: {_sub_queries}"
                        )
                        _existing_066 = {r.concept_id: r for r in top_results}
                        _per_sq_unique: list[list] = []
                        for _sq_i, _sq in enumerate(_sub_queries):
                            _sq_results = retrieval_engine.search_lightweight(
                                _sq,
                                top_k=max(12, effective_max_concepts),
                                min_confidence=0.0,
                                agent_id=(
                                    _req_agent_id
                                    if _req_agent_id != "default"
                                    else None
                                ),
                                scope=_req_scope,
                            )
                            _sq_new = []
                            for _sr in _sq_results:
                                if _sr.concept_id not in _existing_066:
                                    _existing_066[_sr.concept_id] = _sr
                                    _sq_new.append(_sr)
                                elif (
                                    _sr.relevance_score
                                    > _existing_066[_sr.concept_id].relevance_score
                                ):
                                    _existing_066[_sr.concept_id] = _sr
                            _per_sq_unique.append(_sq_new)

                        # Provenance-aware merge: reserve min slots per sub-query
                        _primary_ids = {r.concept_id for r in top_results}
                        _reserved_ids: set[str] = set()
                        for _sq_new_list in _per_sq_unique:
                            _unique = [
                                c
                                for c in _sq_new_list
                                if c.concept_id not in _primary_ids
                            ]
                            for _c in sorted(
                                _unique,
                                key=lambda x: x.relevance_score,
                                reverse=True,
                            )[:_DECOMP_MIN_SLOTS_PER_SUBQUERY]:
                                _reserved_ids.add(_c.concept_id)

                        _reserved = [
                            _existing_066[cid]
                            for cid in _reserved_ids
                            if cid in _existing_066
                        ]
                        _remaining = [
                            c
                            for c in _existing_066.values()
                            if c.concept_id not in _reserved_ids
                        ]
                        _remaining.sort(
                            key=lambda c: c.relevance_score, reverse=True
                        )
                        _cap_066 = int(effective_max_concepts * 1.5)
                        _merged = _reserved + _remaining[
                            : max(0, _cap_066 - len(_reserved))
                        ]
                        _merged.sort(
                            key=lambda c: c.relevance_score, reverse=True
                        )

                        if len(_merged) > len(top_results):
                            logger.info(
                                f"RETRIEVAL-066v2: Merged {len(_existing_066)} unique "
                                f"from {len(_sub_queries)} sub-queries -> {len(_merged)} "
                                f"(reserved {len(_reserved_ids)} sub-query slots, "
                                f"was {len(top_results)})"
                            )
                            top_results = _merged
                        else:
                            logger.info(
                                "RETRIEVAL-066v2: Decomposed retrieval not better, "
                                "keeping original"
                            )
            except Exception as _066_e:
                logger.warning(
                    f"RETRIEVAL-066v2: Failed (non-fatal): {_066_e}"
                )

        # --- RETRIEVAL-037c: serial_order_map built later (moved to pre-activation) ---
        _serial_order_map: dict[str, int] = {}

        association_map: dict[str, list[str]] = {}
        edges = []  # Initialize for S4.1 scope; populated in S4 if top_results exist
        maturity_filtered_count = 0  # W3: Track total maturity-filtered concepts (S2.9 + S4)

        if _skip_graph_walk:
            top_results = top_results  # Keep results, skip graph enrichment
        elif top_results:
            # Single load of all associations for graph walk
            graph_data = load_associations()
            edges = graph_data.get("associations", [])

            # --- GAP D: Edge-type-aware graph walk (spec D.1) ---
            # Edge type behavior:
            #   supports:     traverse eagerly (strength * 1.2)
            #   contradicts:  DON'T traverse, flag as contradiction signal
            #   part_of:      traverse eagerly (strength * 1.1)
            #   derived_from: traverse (strength * 1.0)
            #   enables:      traverse (strength * 1.0)
            #   constrains:   traverse, reduce score (strength * 0.8)
            #   related_to:   traverse (strength * 0.7) — PENALTY for untyped
            EDGE_TYPE_MULTIPLIER = {
                "structural_analogy": 1.5,  # BMB-SPEC: Cross-domain structural parallels (LLM-identified)
                "supports": 1.2,
                "part_of": 1.1,
                "derived_from": 1.0,
                "enables": 1.0,
                "constrains": 0.8,
                "related_to": 0.7,
            }
            # 'contradicts' is intentionally NOT in the multiplier map — never traversed

            # Build adjacency index (bidirectional, edge-type-aware)
            adjacency: dict[str, list[str]] = {}
            edge_relations: dict[tuple[str, str], str] = {}
            contradiction_signals: list[tuple[str, str]] = []

            for edge in edges:
                src, tgt = edge["source"], edge["target"]
                relation = edge.get("relation", "related_to")

                if relation == "contradicts":
                    # Don't traverse, but flag as contradiction signal
                    contradiction_signals.append((src, tgt))
                    if gov_ctx:
                        gov_ctx.log_event(
                            GOV_EVENT_GRAPH_CONTRADICTION_SIGNAL,
                            None,
                            {
                                "source": src,
                                "target": tgt,
                            },
                        )
                    continue

                adjacency.setdefault(src, []).append(tgt)
                adjacency.setdefault(tgt, []).append(src)
                edge_relations[(src, tgt)] = relation
                edge_relations[(tgt, src)] = relation

            for result in top_results:
                neighbors = adjacency.get(result.concept_id, [])
                association_map[result.concept_id] = neighbors

        # --- S4.1: Association Shadow Expansion (budget: 5ms) ---
        # Promote 1-hop neighbors of top results into the result set
        # when association strength exceeds threshold. Key insight: S4 already
        # loads the graph — this is pure in-memory work, no extra DB calls.
        # Addresses retrieval Failure 4: decisions stored separately from
        # the facts they modify are invisible to embedding search alone.
        # RETRIEVAL-055: Shadow expansion default ON. Evidence is mixed:
        # - FC_mh (no gate): -24pp (91% off vs 67% on) → shadow adds noise
        # - LME v3 (with RETRIEVAL-GATE): shadow OFF loses Q1 → shadow fills
        #   useful context into slots freed by gate demotion
        # Net: shadow is complementary with RETRIEVAL-GATE. Keep on by default,
        # re-evaluate if gate is disabled.
        SHADOW_EXPANSION_ENABLED = os.environ.get("SHADOW_EXPANSION_ENABLED", "true").lower() == "true"
        SHADOW_MIN_STRENGTH = float(os.environ.get("SHADOW_MIN_STRENGTH", "0.3"))
        SHADOW_LIMIT = int(os.environ.get("SHADOW_LIMIT", "3"))

        shadow_expanded = []
        edge_strength: dict[tuple[str, str], float] = {}  # populated by S4.1 or S4.1b

        if SHADOW_EXPANSION_ENABLED and top_results and edges:
            # Build strength index with edge-type multipliers (O(E))
            edge_strength = {}
            for edge in edges:
                src, tgt = edge["source"], edge["target"]
                relation = edge.get("relation", "related_to")
                if relation == "contradicts":
                    continue  # Already excluded from adjacency
                s = edge.get("strength", 0.5)
                multiplier = EDGE_TYPE_MULTIPLIER.get(relation, 0.7)
                effective_strength = s * multiplier
                edge_strength[(src, tgt)] = effective_strength
                edge_strength[(tgt, src)] = effective_strength

            existing_ids = {r.concept_id for r in top_results}
            candidates: list[tuple[str, float, str]] = []  # (id, strength, parent)

            for result in top_results:
                neighbors = adjacency.get(result.concept_id, [])
                for neighbor_id in neighbors:
                    if neighbor_id in existing_ids:
                        continue
                    strength = edge_strength.get((result.concept_id, neighbor_id), 0.0)
                    if strength >= SHADOW_MIN_STRENGTH:
                        candidates.append((neighbor_id, strength, result.concept_id))
                        existing_ids.add(neighbor_id)  # prevent dupes

            # Sort by strength descending, take top SHADOW_LIMIT
            candidates.sort(key=lambda x: x[1], reverse=True)
            candidates = candidates[:SHADOW_LIMIT]

            # Load concepts and create SearchResult objects
            # W3-S4: Maturity gate for graph walk — block QUARANTINED/DISCARDED from
            # entering the activation set through shadow expansion (Bug 1 fix).
            _s4_blocked_maturities = {"QUARANTINED", "DISCARDED"}
            _s4_maturity_gate_active = False
            try:
                from app.config import FEATURE_FLAGS as _s4_ff

                _s4_maturity_gate_active = _s4_ff.get("INGESTION_VALIDATION_ENABLED", False)
            except Exception:
                pass

            # RETRIEVAL-004: 72h recency exemption cutoff (computed once, used in loop)
            _s4_recency_cutoff = (_utc_now() - timedelta(hours=QUARANTINE_RECENCY_EXEMPT_HOURS)).isoformat()

            # RETRIEVAL-002 / SUPER-012: Soft scoring multiplier for SUPERSEDED concepts
            # in S4 graph walk. 0.15x means a superseded concept needs ~7x higher base
            # relevance to beat an active concept. Not a hard block — preserves access
            # to stranded knowledge while strongly deprioritizing it.
            # DEBT-027: SUPERSEDED_S4_MULTIPLIER hoisted to module level

            s4_maturity_filtered = 0
            s4_superseded_penalized = 0
            s4_contradicted_penalized = 0  # MONITOR-045
            for cid, strength, parent_id in candidates:
                concept = load_concept(cid)
                if concept and concept.confidence >= 0.0:
                    # W3-S4: Check maturity before allowing graph walk entry
                    if _s4_maturity_gate_active:
                        concept_maturity = getattr(concept, "maturity", "ESTABLISHED")
                        if concept_maturity in _s4_blocked_maturities:
                            # RETRIEVAL-004: Exempt QUARANTINED concepts < 72h old
                            _s4_exempt = (
                                concept_maturity == "QUARANTINED"
                                and getattr(concept, "created_at", "") > _s4_recency_cutoff
                            )
                            if not _s4_exempt:
                                s4_maturity_filtered += 1
                                logger.info(
                                    f"W3-S4: Blocked {cid} from graph walk entry "
                                    f"(maturity={concept_maturity}, parent={parent_id})"
                                )
                                continue

                    # RETRIEVAL-002: Soft penalty for SUPERSEDED concepts
                    _effective_strength = strength
                    # NOTE: currency_status is a TOP-LEVEL Concept attribute, NOT in metadata dict.
                    # See models.py:212, storage.py:756. Existing pattern: getattr (session.py:2070).
                    _concept_currency = getattr(concept, "currency_status", "ACTIVE")
                    if _concept_currency == "SUPERSEDED":
                        # RETRIEVAL-056: Skip entirely — successor should carry the knowledge.
                        # Was: soft penalty at 0.15x (RETRIEVAL-002). Now: hard skip.
                        s4_superseded_penalized += 1
                        continue
                    elif _concept_currency == "CONTRADICTED":
                        _effective_strength = strength * _CONTRADICTED_S4_MULTIPLIER
                        s4_contradicted_penalized += 1  # MONITOR-045

                    shadow_result = SearchResult(
                        concept_id=concept.id,
                        version=concept.version,
                        summary=concept.summary,
                        confidence=concept.confidence,
                        relevance_score=round(_effective_strength * 0.5, 4),
                        knowledge_area=concept.metadata.get("knowledge_area"),
                    )
                    shadow_expanded.append(shadow_result)
                    # Also populate association_map for this concept
                    association_map[cid] = adjacency.get(cid, [])

            if s4_superseded_penalized > 0:
                logger.info(
                    "RETRIEVAL-002: Penalized %d SUPERSEDED concepts (x%s)",
                    s4_superseded_penalized,
                    _SUPERSEDED_S4_MULTIPLIER,
                )
            if s4_contradicted_penalized > 0:  # MONITOR-045
                logger.info(
                    "CURRENCY-003: Penalized %d CONTRADICTED concepts (x%s)",
                    s4_contradicted_penalized,
                    _CONTRADICTED_S4_MULTIPLIER,
                )
            if s4_maturity_filtered > 0:
                maturity_filtered_count += s4_maturity_filtered
                logger.info(f"W3-S4: Filtered {s4_maturity_filtered} quarantined concepts from graph walk")
            if shadow_expanded:
                top_results.extend(shadow_expanded)
                logger.info(f"S4.1: Shadow-expanded {len(shadow_expanded)} concepts (threshold={SHADOW_MIN_STRENGTH})")
        elif not SHADOW_EXPANSION_ENABLED:
            logger.debug("S4.1: Shadow expansion disabled by env var")

        # --- S4.1b: DECISION concept 2-hop expansion (RETRIEVAL-041, budget: 4ms) ---
        # Dynamically extends walk depth to 2 hops — but ONLY when:
        #   (a) the current activation set has no concepts from strategic KAs, AND
        #   (b) the edge graph has been loaded (not circuit-broken)
        # This is adaptive: if governs edges (RETRIEVAL-041 Part A) are present,
        # DECISION concepts surface via normal 1-hop S4.1 and this block is a no-op.
        # It fires as a fallback when the graph hasn't been backfilled yet, or when
        # the DECISION concept was created before RETRIEVAL-041 shipped.
        DECISION_SHADOW_ENABLED = os.environ.get("DECISION_SHADOW_ENABLED", "true").lower() == "true"
        _DECISION_SHADOW_KAS = frozenset({
            "product_strategy", "business_strategy",
            "strategic_recommendation", "strategy",
        })
        DECISION_SHADOW_LIMIT = int(os.environ.get("DECISION_SHADOW_LIMIT", "2"))

        if (
            DECISION_SHADOW_ENABLED
            and top_results
            and edges
            and not _skip_graph_walk
            and adjacency
        ):
            _active_kas = {r.knowledge_area for r in top_results if r.knowledge_area}
            _has_strategic = bool(_active_kas & _DECISION_SHADOW_KAS)

            if not _has_strategic:
                # Lazily build edge_strength if S4.1 shadow expansion was disabled
                if not edge_strength:
                    for _e in edges:
                        _src, _tgt = _e["source"], _e["target"]
                        if _e.get("relation") == "contradicts":
                            continue
                        _s = _e.get("strength", 0.5)
                        _m = EDGE_TYPE_MULTIPLIER.get(_e.get("relation", "related_to"), 0.7)
                        _es = _s * _m
                        edge_strength[(_src, _tgt)] = _es
                        edge_strength[(_tgt, _src)] = _es

                # Multi-hop DECISION expansion: hop1 → hop2 → hop3 (adaptive)
                # Evaluates ALL hop depths for DECISION concepts in strategic KAs.
                # Hop1 was previously skipped (used only as bridge) — RETRIEVAL-041b fix.
                _existing_ids = {r.concept_id for r in top_results}
                _decision_added = 0

                # Helper: evaluate a candidate concept for DECISION inclusion
                def _try_add_decision(cid: str, score: float, hop_depth: int) -> bool:
                    nonlocal _decision_added
                    if _decision_added >= DECISION_SHADOW_LIMIT:
                        return False
                    if cid in _existing_ids:
                        return False
                    _c = load_concept(cid)
                    if not _c:
                        return False
                    _c_ka = _c.metadata.get("knowledge_area") if _c.metadata else None
                    if _c_ka not in _DECISION_SHADOW_KAS:
                        return False
                    if getattr(_c, "concept_type", "") != "decision":
                        return False
                    _c_currency = getattr(_c, "currency_status", "ACTIVE")
                    if _c_currency in ("SUPERSEDED", "STALE"):
                        return False
                    # Discount factor scales with hop depth
                    _discount = {1: 0.50, 2: 0.35, 3: 0.20}.get(hop_depth, 0.15)
                    top_results.append(SearchResult(
                        concept_id=_c.id,
                        version=_c.version,
                        summary=_c.summary,
                        confidence=_c.confidence,
                        relevance_score=round(score * _discount, 4),
                        knowledge_area=_c_ka,
                    ))
                    association_map[cid] = adjacency.get(cid, [])
                    _existing_ids.add(cid)
                    _decision_added += 1
                    return True

                # --- Hop 1: Direct neighbors (RETRIEVAL-041b fix) ---
                # S4.1 skips these when edge_strength < SHADOW_MIN_STRENGTH.
                # For DECISION concepts, we use a relaxed threshold.
                _DECISION_HOP1_MIN = SHADOW_MIN_STRENGTH * 0.3  # 0.09 at default
                _hop1_candidates: list[tuple[str, float]] = []
                for _r in top_results[:5]:
                    for _hop1_id in adjacency.get(_r.concept_id, []):
                        if _hop1_id in _existing_ids:
                            continue
                        _s1 = edge_strength.get((_r.concept_id, _hop1_id), 0.0)
                        if _s1 >= _DECISION_HOP1_MIN:
                            _hop1_candidates.append((_hop1_id, _s1))

                _hop1_candidates.sort(key=lambda x: x[1], reverse=True)
                for _cid, _score in _hop1_candidates:
                    _try_add_decision(_cid, _score, hop_depth=1)

                # --- Hop 2: neighbors-of-neighbors ---
                if _decision_added < DECISION_SHADOW_LIMIT:
                    _hop2_candidates: dict[str, tuple[float, str]] = {}
                    for _r in top_results[:5]:
                        for _hop1_id in adjacency.get(_r.concept_id, []):
                            if _hop1_id in _existing_ids:
                                continue
                            _s1 = edge_strength.get((_r.concept_id, _hop1_id), 0.0)
                            for _hop2_id in adjacency.get(_hop1_id, []):
                                if _hop2_id in _existing_ids or _hop2_id == _r.concept_id:
                                    continue
                                _s2 = edge_strength.get((_hop1_id, _hop2_id), 0.0)
                                _combined = _s1 * _s2
                                if _combined > _hop2_candidates.get(_hop2_id, (0.0, ""))[0]:
                                    _hop2_candidates[_hop2_id] = (_combined, _r.concept_id)

                    _hop2_sorted = sorted(_hop2_candidates.items(), key=lambda x: x[1][0], reverse=True)
                    for _cid, (_score, _parent) in _hop2_sorted:
                        if _decision_added >= DECISION_SHADOW_LIMIT:
                            break
                        if _score < SHADOW_MIN_STRENGTH * 0.4:
                            break
                        _try_add_decision(_cid, _score, hop_depth=2)

                # --- Hop 3: 3-hop fallback (adaptive, fires only when hop1+hop2 yield 0) ---
                DECISION_HOP3_ENABLED = os.environ.get(
                    "DECISION_HOP3_ENABLED", "true"
                ).lower() == "true"
                if (
                    DECISION_HOP3_ENABLED
                    and _decision_added == 0
                    and DECISION_SHADOW_LIMIT > 0
                ):
                    _hop3_candidates: dict[str, tuple[float, str]] = {}
                    _HOP3_MIN = SHADOW_MIN_STRENGTH * 0.15  # very relaxed for 3-hop
                    for _r in top_results[:5]:
                        for _h1 in adjacency.get(_r.concept_id, []):
                            _s1 = edge_strength.get((_r.concept_id, _h1), 0.0)
                            if _s1 < _HOP3_MIN:
                                continue
                            for _h2 in adjacency.get(_h1, []):
                                if _h2 == _r.concept_id:
                                    continue
                                _s2 = edge_strength.get((_h1, _h2), 0.0)
                                if _s1 * _s2 < _HOP3_MIN:
                                    continue
                                for _h3 in adjacency.get(_h2, []):
                                    if _h3 in _existing_ids or _h3 == _h1 or _h3 == _r.concept_id:
                                        continue
                                    _s3 = edge_strength.get((_h2, _h3), 0.0)
                                    _combined = _s1 * _s2 * _s3
                                    if _combined > _hop3_candidates.get(_h3, (0.0, ""))[0]:
                                        _hop3_candidates[_h3] = (_combined, _r.concept_id)

                    _hop3_sorted = sorted(
                        _hop3_candidates.items(), key=lambda x: x[1][0], reverse=True
                    )
                    for _cid, (_score, _parent) in _hop3_sorted[:10]:  # limit scan
                        if _decision_added >= DECISION_SHADOW_LIMIT:
                            break
                        if _score < _HOP3_MIN:
                            break
                        _try_add_decision(_cid, _score, hop_depth=3)

                if _decision_added > 0:
                    logger.info(
                        "RETRIEVAL-041 S4.1b: Added %d DECISION concepts via multi-hop walk "
                        "(no strategic KA in activation set)",
                        _decision_added,
                    )

        # --- S4.2: Cross-domain relevance injection (RETRIEVAL-025) (budget: 3ms) ---
        # When activation set is domain-homogeneous (>80% same KA), inject
        # high-authority concepts from causally related domains to prevent
        # cross-domain blind spots.
        CROSS_DOMAIN_INJECTION_ENABLED = os.environ.get(
            "CROSS_DOMAIN_INJECTION_ENABLED", "true"
        ).lower() == "true"
        CROSS_DOMAIN_HOMOGENEITY_THRESHOLD = 0.80
        CROSS_DOMAIN_INJECT_LIMIT = 2

        if CROSS_DOMAIN_INJECTION_ENABLED and top_results and not _skip_graph_walk:
            try:
                # Compute KA distribution of current activation set
                ka_counts: dict[str, int] = {}
                for r in top_results:
                    ka = getattr(r, "knowledge_area", None) or "unknown"
                    ka_counts[ka] = ka_counts.get(ka, 0) + 1

                total_activated = len(top_results)
                dominant_ka = max(ka_counts, key=ka_counts.get) if ka_counts else None
                dominant_ratio = (
                    ka_counts.get(dominant_ka, 0) / max(total_activated, 1)
                    if dominant_ka
                    else 0.0
                )

                if dominant_ratio >= CROSS_DOMAIN_HOMOGENEITY_THRESHOLD and dominant_ka:
                    related_kas = DOMAIN_BRIDGES.get(dominant_ka, [])
                    existing_ids = {r.concept_id for r in top_results}
                    injected = []

                    for related_ka in related_kas:
                        if len(injected) >= CROSS_DOMAIN_INJECT_LIMIT:
                            break
                        from app.storage import get_high_authority_concepts_by_ka

                        candidates = get_high_authority_concepts_by_ka(
                            related_ka, limit=5
                        )
                        for cand in candidates:
                            if cand["id"] in existing_ids:
                                continue
                            if len(injected) >= CROSS_DOMAIN_INJECT_LIMIT:
                                break
                            concept = load_concept(cand["id"], track_access=False)
                            if not concept:
                                continue
                            inject_result = SearchResult(
                                concept_id=concept.id,
                                version=concept.version,
                                summary=concept.summary,
                                confidence=concept.confidence,
                                relevance_score=0.35,
                                knowledge_area=concept.metadata.get("knowledge_area"),
                            )
                            injected.append(inject_result)
                            existing_ids.add(cand["id"])

                    if injected:
                        top_results.extend(injected)
                        logger.info(
                            f"S4.2: Cross-domain injection: +{len(injected)} concepts "
                            f"from {related_kas} (dominant={dominant_ka} "
                            f"at {dominant_ratio:.0%})"
                        )
            except Exception as e:
                logger.warning(f"S4.2: Cross-domain injection failed (non-fatal): {e}")
        elif not CROSS_DOMAIN_INJECTION_ENABLED:
            logger.debug("S4.2: Cross-domain injection disabled by env var")

        # --- S4.5: Supplementary retrieval dispatch (budget: 100ms) ---
        # For non-general classifications, dispatch to temporal/causal modules
        # for additional context. Hard timeout at 100ms. Non-fatal.
        supplementary_results = []
        try:
            if question_classification and question_classification.get("classification") != "general":
                from app.router import ENABLE_COGNITIVE_ROUTER, dispatch_supplementary, log_classification

                if ENABLE_COGNITIVE_ROUTER:
                    best_id = top_results[0].concept_id if top_results else None
                    sid = request.session_id or (self.current_session.session_id if self.current_session else "unknown")
                    supplementary_results = dispatch_supplementary(
                        classification=question_classification["classification"],
                        dates=inferred_dates or {"since": None, "until": None},
                        best_concept_id=best_id,
                        session_id=sid,
                    )
                    if supplementary_results:
                        # Merge supplementary into top_results, deduplicating
                        existing_ids = {r.concept_id for r in top_results}
                        for sup in supplementary_results:
                            sup_id = sup.get("concept_id") or sup.get("id")
                            if sup_id and sup_id not in existing_ids:
                                sup_result = SearchResult(
                                    concept_id=sup_id,
                                    version=sup.get("version", "v1"),
                                    summary=sup.get("summary", ""),
                                    confidence=sup.get("confidence", 0.5),
                                    relevance_score=sup.get("relevance_score", 0.3),
                                    knowledge_area=sup.get("knowledge_area"),
                                )
                                top_results.append(sup_result)
                                existing_ids.add(sup_id)
                        logger.info(f"S4.5: Added {len(supplementary_results)} supplementary results")

                    # Log classification for analytics
                    try:
                        log_classification(
                            session_id=sid,
                            input_source=question_classification.get("input_source", "processed"),
                            input_length=len(request.message),
                            classification=question_classification["classification"],
                            confidence=question_classification.get("confidence", 0.0),
                            was_overridden=question_classification.get("input_source") == "forced",
                        )
                    except Exception as log_err:
                        logger.debug(f"S4.5: Classification logging failed: {log_err}")
        except Exception as e:
            logger.warning(f"S4.5: Supplementary retrieval failed (non-fatal): {e}")

        # --- S4.6: Double-counting correction (F8) ---
        # If S4.5 fired for temporal queries, concepts that got BOTH S2.6
        # temporal boost AND S4.5 supplementary inclusion are double-counted.
        # Fix: reverse the S2.6 boost for any concept also in S4.5 results.
        # Math: boosted_score / multiplier = original_score
        try:
            if supplementary_results and question_classification:
                cls = question_classification.get("classification", "")
                if cls.startswith("temporal"):
                    from app.temporal import temporal_boost

                    supplementary_ids = set()
                    for sup in supplementary_results:
                        sup_id = sup.get("concept_id") or sup.get("id")
                        if sup_id:
                            supplementary_ids.add(sup_id)

                    corrections = 0
                    for sr in search_results:
                        if sr.concept_id in supplementary_ids:
                            concept_data = load_concept(sr.concept_id)
                            if concept_data and concept_data.updated_at:
                                boost_result = temporal_boost(concept_data.updated_at)
                                multiplier = boost_result.get("boost_multiplier", 1.0)
                                if multiplier > 1.0:
                                    sr.relevance_score = round(sr.relevance_score / multiplier, 4)
                                    corrections += 1
                    if corrections:
                        search_results.sort(key=lambda x: x.relevance_score, reverse=True)
                        logger.info(f"S4.6: Corrected {corrections} double-counted temporal concepts")
        except Exception as e:
            logger.warning(f"S4.6: Double-counting correction failed (non-fatal): {e}")

        # --- S4.7: Entity-chain keyword retrieval (RETRIEVAL-047) ---
        # For queries containing named entities, do SQL keyword search per entity,
        # chain values through copula extraction for multi-hop lookups.
        # Unions results with embedding retrieval. Feature-gated, time-budgeted.
        try:
            from app.entity_chain import ENTITY_CHAIN_ENABLED, ENTITY_CHAIN_BUDGET_MS, get_entity_chain_retriever

            _ec_enabled = ENTITY_CHAIN_ENABLED
            _ec_budget = ENTITY_CHAIN_BUDGET_MS
            # RETRIEVAL-060: Adaptive router override
            if _adaptive_config and _adaptive_config.force_entity_chain:
                _ec_enabled = True
                _ec_budget = _adaptive_config.entity_chain_budget_ms
            elif _adaptive_config and _adaptive_config.use_entity_chain and not _ec_enabled:
                _ec_enabled = True  # soft enable from router

            if _ec_enabled:
                _ec_retriever = get_entity_chain_retriever()
                if _ec_retriever:
                    # RETRIEVAL-051/F4: Pass raw user message, not decorated search_query.
                    # search_query includes firmware/constraint text that causes noisy
                    # entity extraction (paths, full sentences as "entities").
                    _ec_input = request.message or search_query  # fallback if message is None
                    _ec_results = _ec_retriever.retrieve(
                        _ec_input, budget_ms=_ec_budget
                    )
                    if _ec_results:
                        existing_ids = {r.concept_id for r in top_results}
                        # RETRIEVAL-051/F5: Cap entity chain additions proportional to brain size.
                        # Without cap, entity chain adds 68-109 concepts on production,
                        # bypassing all upstream retrieval caps (F1/F2).
                        try:
                            from app.embedding import embedding_engine as _ec_emb
                            _ec_brain = _ec_emb.index_size
                        except Exception:
                            _ec_brain = 10000
                        _ec_max = max(15, int(_ec_brain * 0.07))  # 7% of brain, min 15
                        _ec_new = []
                        for ecr in _ec_results:
                            if ecr.concept_id not in existing_ids:
                                _ec_new.append(ecr)
                                existing_ids.add(ecr.concept_id)
                        # Keep top by relevance if over cap.
                        # RETRIEVAL-058: When capping, prioritize results whose summary
                        # contains question property keywords (capital, language, etc.)
                        # so chain-completing facts survive over noise. Only affects
                        # internal entity chain ordering — does NOT inflate relevance_score.
                        _ec_pre_cap = len(_ec_new)
                        if _ec_pre_cap > _ec_max:
                            _ec_qkw = getattr(_ec_retriever, '_question_keywords', [])
                            def _ec_sort_key(r):
                                _kw_hit = 0
                                if _ec_qkw:
                                    _s = (r.summary or "").lower()
                                    _kw_hit = 1 if any(kw in _s for kw in _ec_qkw) else 0
                                return (_kw_hit, r.relevance_score)
                            _ec_new.sort(key=_ec_sort_key, reverse=True)
                            _ec_new = _ec_new[:_ec_max]
                            logger.info(
                                f"RETRIEVAL-051: Entity-chain capped {_ec_pre_cap} -> {_ec_max} "
                                f"(brain={_ec_brain})"
                            )
                        top_results.extend(_ec_new)
                        _ec_added = len(_ec_new)
                        if _ec_added:
                            logger.info(
                                f"S4.7: Entity-chain added {_ec_added} concepts "
                                f"(searched entities: {_ec_retriever.last_searched_entities})"
                            )
        except Exception as e:
            logger.warning(f"S4.7: Entity-chain retrieval failed (non-fatal): {e}")

        t_graph = time.perf_counter()  # PERF-016: Phase A checkpoint

        # --- S4.8: Ambient principle injection (budget: 5ms) ---
        # Surface high-confidence principles/methods/strategies regardless of
        # keyword match. These are "ambient" knowledge that should always be
        # available when relevant to the knowledge area.
        from app.models import ABSTRACT_CONCEPT_TYPES
        from app.storage import load_concepts_by_type

        # Collect knowledge areas from search results for scoping
        result_areas = set()
        for r in top_results:
            if r.knowledge_area:
                result_areas.add(r.knowledge_area)

        # Fetch top abstract concepts, deduplicate against search results
        existing_ids = {r.concept_id for r in top_results}
        ambient_principles = load_concepts_by_type(
            concept_types=list(ABSTRACT_CONCEPT_TYPES),
            limit=10,  # Fetch more than 3 to allow filtering below
            min_confidence=0.40,
        )
        # Filter: only inject if (a) not already in results, (b) knowledge_area
        # overlaps with retrieval results, and (c) knowledge_area is specific
        # (not "general"/"unknown"). This closes the phantom concept bug where
        # 3 high-confidence general-area concepts (method_multi_context_docker_build,
        # etc.) were injected into every query regardless of topic relevance.
        # The result_areas gate implements the original design intent: "ambient
        # knowledge that should always be available when relevant to the
        # knowledge area."
        ambient_injected = []
        for ap in ambient_principles:
            if len(ambient_injected) >= 3:
                break
            if ap["concept_id"] in existing_ids:
                continue
            ap_area = ap.get("knowledge_area", "general")
            # Skip generic concepts — they have no topic signal to match
            if ap_area in ("general", "unknown", "unclassified", ""):
                continue
            # Knowledge-area gate: only inject if area matches retrieval results
            if result_areas and ap_area not in result_areas:
                continue
            ambient_injected.append(ap)
            existing_ids.add(ap["concept_id"])

        # --- S4.6: Always-activate concept injection (budget: 2ms) ---
        # P1-1: Concepts flagged always_activate=true get injected into EVERY
        # conversation_turn regardless of topic. Solves Desktop Commander-class
        # retrieval misses where knowledge exists but doesn't activate at
        # tool-selection time because retrieval is topic-keyed, not action-keyed.
        from app.storage import load_always_activate_concepts

        always_on = load_always_activate_concepts()
        always_on_injected = []
        for ao in always_on:
            if ao["concept_id"] not in existing_ids:
                always_on_injected.append(ao)
                existing_ids.add(ao["concept_id"])
        if always_on_injected:
            logger.info(f"S4.6: Injected {len(always_on_injected)} always-activate concept(s)")

        # --- S4.7: Firmware injection (P0-5) (budget: 1ms) ---
        # Static developer-controlled operational knowledge from firmware table.
        # Physically isolated from concepts — separate table, no TF-IDF index,
        # no association edges, no reflection/decay. ROM model: only updated
        # by seed_firmware.py on server startup.
        from app.storage import load_firmware

        firmware_entries = load_firmware()
        if firmware_entries:
            logger.info(f"S4.7: Injected {len(firmware_entries)} firmware entries")

        # --- S4.8: Directive injection (budget: 1ms) ---
        # User-controlled behavioral instructions (Tier 2 in injection hierarchy).
        # Delivered as structured data, separate from concepts.
        # Budget-aware: priority-ordered truncation at 8,000 char aggregate.
        directives_response = {"directives": [], "budget_warning": None}
        try:
            from app.directives import load_directives_budgeted

            directives_response = load_directives_budgeted()
            if directives_response["directives"]:
                logger.info(
                    f"S4.8: Injected {len(directives_response['directives'])} directives "
                    f"({directives_response['total_chars']} chars)"
                )
                if directives_response["budget_warning"]:
                    logger.warning(f"S4.8: {directives_response['budget_warning']}")
        except Exception as e:
            logger.warning(f"S4.8: Directive injection failed (non-fatal): {e}")

        # Track fact-supplemented IDs for downstream CE-gate (SUPPLEMENT-GATE)
        _fs_supplemented_ids = set()

        # --- RETRIEVAL-050: Fact supplement layer (association edge traversal) ---
        # When abstractions/patterns are retrieved but lack specific entities,
        # traverse association edges to pull in linked observations with dates,
        # names, numbers. Addresses 'abstraction drowning'. Budget: 6 supplements.
        _FACT_SUPPLEMENT_ENABLED = os.environ.get('PITH_FACT_SUPPLEMENT', '').lower() in ('true', '1')
        if _FACT_SUPPLEMENT_ENABLED and top_results and edges:
            try:
                _fs_existing_ids = {r.concept_id for r in top_results}
                _fs_assoc_targets = set()
                for r in top_results:
                    for aid in association_map.get(r.concept_id, []):
                        if aid not in _fs_existing_ids:
                            _fs_assoc_targets.add(aid)

                if _fs_assoc_targets:
                    _fs_conn = _get_connection()
                    _fs_ph = ','.join('?' * len(_fs_assoc_targets))
                    _fs_rows = _fs_conn.execute(
                        f"""SELECT id, summary, confidence, knowledge_area, concept_type, status
                           FROM concepts WHERE id IN ({_fs_ph})""",
                        list(_fs_assoc_targets),
                    ).fetchall()

                    from app.entity_detector import has_specific_entities
                    _fs_candidates = []
                    _fs_q_words = set(search_query.lower().split())
                    for row in _fs_rows:
                        _fs_summary = row[1] or ''
                        _fs_ctype = row[4] or 'observation'
                        _fs_status = row[5] or 'active'
                        if _fs_status in ('archived', 'deleted', 'superseded'):
                            continue
                        _fs_score = 0.0
                        if _fs_ctype == 'observation':
                            _fs_score += 0.5
                        if has_specific_entities(_fs_summary):
                            _fs_score += 0.5
                        _fs_s_words = set(_fs_summary.lower().split())
                        _fs_overlap = len(_fs_q_words & _fs_s_words)
                        _fs_score += min(_fs_overlap * 0.1, 0.3)
                        # RETRIEVAL-GATE-F2: Cap fact supplement below semantic range
                        _FS_SCORE_CAP = float(os.environ.get('PITH_FS_SCORE_CAP', '0.7'))
                        _fs_score = min(_fs_score, _FS_SCORE_CAP)
                        if _fs_score >= 0.5:
                            _fs_candidates.append((_fs_score, row))

                    _fs_candidates.sort(key=lambda x: x[0], reverse=True)
                    _FS_BUDGET = int(os.environ.get('PITH_FACT_SUPPLEMENT_BUDGET', '6'))
                    _fs_added = 0
                    for _fs_sc, _fs_r in _fs_candidates[:_FS_BUDGET]:
                        top_results.append(SearchResult(
                            concept_id=_fs_r[0],
                            version='v1',
                            summary=_fs_r[1],
                            confidence=_fs_r[2] or 0.5,
                            relevance_score=round(_fs_sc, 4),
                            knowledge_area=_fs_r[3] or 'unknown',
                        ))
                        _fs_supplemented_ids.add(_fs_r[0])
                        _fs_added += 1
                    if _fs_added:
                        logger.info(
                            f'RETRIEVAL-050: Supplemented {_fs_added} facts from '
                            f'{len(_fs_assoc_targets)} association targets '
                            f'({len(_fs_candidates)} scored candidates)'
                        )
            except Exception as _fs_e:
                logger.warning(f'RETRIEVAL-050: Fact supplement failed (non-fatal): {_fs_e}')


        # Track keyword-supplemented IDs for downstream exemption (Fix 3 chain pruning)
        # [GAUNTLET A1: initialized unconditionally so Fix 3 can reference it even if Fix 2 disabled]
        _kw_supplemented_ids = set()

        # --- RETRIEVAL-042: Keyword search supplement (BM25-style fallback) ---
        # When embedding search misses concepts containing question keywords,
        # SQL LIKE matching surfaces them. Runs AFTER embedding + fact supplement.
        # Feature-gated, budget-limited. Ported from adapter/retrieval.py:279-387.
        _KW_SUPPLEMENT_ENABLED = os.environ.get(
            'PITH_KEYWORD_SUPPLEMENT', ''
        ).lower() in ('true', '1')
        if _KW_SUPPLEMENT_ENABLED and top_results:
            try:
                _kw_conn = _get_connection()
                _kw_existing_ids = {r.concept_id for r in top_results}

                # Extract meaningful keywords (skip stopwords)
                _KW_STOPWORDS = frozenset({
                    'what', 'when', 'where', 'who', 'which', 'how', 'why',
                    'is', 'are', 'was', 'were', 'did', 'does', 'do', 'has',
                    'have', 'had', 'the', 'a', 'an', 'in', 'on', 'at', 'to',
                    'for', 'of', 'with', 'by', 'from', 'and', 'or', 'not',
                    'be', 'been', 'being', 'would', 'could', 'should', 'will',
                    'can', 'may', 'might', 'shall', 'it', 'its', 'this',
                    'that', 'these', 'those', 'if', 'still', 'likely',
                })
                # Use raw user message (not decorated search_query which includes
                # firmware/constraint text). Same rationale as RETRIEVAL-051/F4.
                _kw_query = request.message or search_query
                _kw_words = [
                    w.strip('?.,!\'\'"').lower()
                    for w in _kw_query.split()
                    if w.strip('?.,!\'\'"').lower() not in _KW_STOPWORDS
                    and len(w.strip('?.,!\'\'"')) > 2
                ]

                if _kw_words:
                    # RETRIEVAL-042 upgrade: FTS5 BM25 replaces SQL LIKE
                    # Sanitize keywords for FTS5 MATCH syntax
                    # [GAUNTLET A4: Strip special chars that FTS5 interprets as operators]
                    # RETRIEVAL-074: Split hyphenated tokens instead of stripping hyphens.
                    # "salem-keizer" was becoming "salemkeizer" (no FTS match).
                    # Now becomes ["salem", "keizer"] → FTS matches both components.
                    import re as _kw_re
                    _fts_safe_words = []
                    for w in _kw_words:
                        # Split on hyphens first, then sanitize each part
                        _parts = w.split('-') if '-' in w else [w]
                        for _part in _parts:
                            cleaned = _kw_re.sub(r'[^\w]', '', _part)
                            if cleaned and len(cleaned) > 2:
                                _fts_safe_words.append(cleaned)
                    if not _fts_safe_words:
                        _fts_safe_words = _kw_words  # Fallback if sanitization empties list

                    # Build FTS5 match expression: combine keywords with OR
                    # FTS5 handles tokenization, stemming, IDF, TF saturation, doc length norm
                    _fts_query = " OR ".join(_fts_safe_words)

                    _kw_sql = """
                        SELECT f.concept_id, c.summary, c.confidence, c.knowledge_area,
                               c.concept_type, bm25(fts_concepts) as bm25_score
                        FROM fts_concepts f
                        JOIN concepts c ON c.id = f.concept_id
                        WHERE fts_concepts MATCH ?
                          AND c.status = 'active' AND c.is_current = 1
                        ORDER BY bm25(fts_concepts)
                        LIMIT 20
                    """
                    _kw_rows = _kw_conn.execute(_kw_sql, (_fts_query,)).fetchall()

                    _KW_BUDGET = int(os.environ.get('PITH_KEYWORD_SUPPLEMENT_BUDGET', '8'))
                    _kw_added = 0
                    for _kw_r in _kw_rows:
                        _kw_cid, _kw_summary, _kw_conf, _kw_ka, _kw_ctype, _kw_bm25 = _kw_r
                        if _kw_cid in _kw_existing_ids:
                            continue
                        if _kw_added >= _KW_BUDGET:
                            break
                        # BM25 score from FTS5 (negative = more relevant, SQLite convention)
                        # Normalize to 0-1 range for compatibility with relevance_score field
                        # Empirical range: rare term ~ -2.6, multi-word ~ -5.2
                        # [GAUNTLET A3: Extracted constant, calibrated from 10.0->5.0]
                        _BM25_SCORE_NORMALIZER = float(os.environ.get('PITH_BM25_NORMALIZER', '5.0'))
                        # RETRIEVAL-GATE-F2: Cap supplement scores below semantic match range.
                        # Previously min(1.0, ...) created a ceiling where generic-keyword
                        # matches piled up at 1.0, outranking semantically relevant concepts.
                        # Q10 RCA: "recommendations" (59 matches) pushed YouTube/protein bar
                        # concepts to 1.0, burying gold "coffee creamer" concept at position 12.
                        _KW_SCORE_CAP = float(os.environ.get('PITH_KW_SCORE_CAP', '0.7'))
                        from app.entity_detector import has_specific_entities
                        _kw_score = min(_KW_SCORE_CAP, abs(_kw_bm25) / _BM25_SCORE_NORMALIZER)
                        if has_specific_entities(_kw_summary or ''):
                            _kw_score = min(_KW_SCORE_CAP, _kw_score + 0.2)
                        top_results.append(SearchResult(
                            concept_id=_kw_cid,
                            version='v1',
                            summary=_kw_summary,
                            confidence=_kw_conf if _kw_conf is not None else 0.5,
                            relevance_score=round(_kw_score, 4),
                            knowledge_area=_kw_ka or 'unknown',
                        ))
                        _kw_existing_ids.add(_kw_cid)
                        _kw_supplemented_ids.add(_kw_cid)
                        _kw_added += 1
                    if _kw_added:
                        logger.info(
                            f'RETRIEVAL-042: BM25 supplement added {_kw_added} concepts '
                            f'(query="{_fts_query}", {len(_kw_rows)} candidates)'
                        )
            except Exception as _kw_e:
                logger.warning(f'RETRIEVAL-042: Keyword supplement failed (non-fatal): {_kw_e}')

        # --- SUPPLEMENT-GATE: CE rerank supplement-added concepts ---
        # Q10 RCA: Keyword supplement adds concepts matching generic words
        # ("recommendations", "trying") that outscore semantically relevant
        # embedding results. CE-gate supplement concepts specifically to
        # demote irrelevant supplements and let gold rise.
        # Source-based gating (not score-based) — targets KW + fact supplements.
        _SUPP_GATE_ENABLED = os.environ.get('PITH_SUPPLEMENT_GATE', '').lower() in ('true', '1')
        _SUPP_GATE_CE_THRESHOLD = float(os.environ.get('PITH_SUPPLEMENT_GATE_CE', '0.15'))
        _all_supplement_ids = _kw_supplemented_ids | _fs_supplemented_ids
        if _SUPP_GATE_ENABLED and _all_supplement_ids and top_results:
            try:
                _supg_supplements = [
                    (i, r) for i, r in enumerate(top_results)
                    if r.concept_id in _all_supplement_ids
                ]
                if _supg_supplements:
                    from app.reranker import _get_cross_encoder
                    _supg_model = _get_cross_encoder()
                    _supg_query = request.message or search_query
                    _supg_pairs = [(_supg_query, r.summary or '') for _, r in _supg_supplements]

                    import numpy as np
                    _supg_scores = _supg_model.predict(_supg_pairs, show_progress_bar=False)
                    if not hasattr(_supg_scores, '__iter__'):
                        _supg_scores = [_supg_scores]
                    _supg_scores = np.array(_supg_scores, dtype=np.float32)

                    _supg_demoted = 0
                    _supg_kept = 0
                    _supg_details = []
                    for idx, ((orig_idx, r), ce_score) in enumerate(zip(_supg_supplements, _supg_scores)):
                        ce_score_f = float(ce_score)
                        if ce_score_f < _SUPP_GATE_CE_THRESHOLD:
                            _old_score = r.relevance_score
                            # Demote to CE score (floor 0.05) — lets gold rise above
                            r.relevance_score = max(0.05, ce_score_f)
                            _supg_details.append(
                                f'  DEMOTE {r.concept_id}: {_old_score:.3f}->{r.relevance_score:.3f} '
                                f'CE={ce_score_f:.4f} "{(r.summary or "")[:60]}"'
                            )
                            _supg_demoted += 1
                        else:
                            _supg_details.append(
                                f'  KEEP   {r.concept_id}: score={r.relevance_score:.3f} '
                                f'CE={ce_score_f:.4f} "{(r.summary or "")[:60]}"'
                            )
                            _supg_kept += 1

                    if _supg_demoted > 0:
                        top_results.sort(key=lambda x: x.relevance_score, reverse=True)

                    logger.info(
                        f'SUPPLEMENT-GATE: CE-scored {len(_supg_supplements)} supplement concepts '
                        f'(demoted={_supg_demoted}, kept={_supg_kept}, '
                        f'ce_gate={_SUPP_GATE_CE_THRESHOLD})\n'
                        + '\n'.join(_supg_details)
                    )
            except Exception as _supg_e:
                logger.warning(f'SUPPLEMENT-GATE: Failed (non-fatal): {_supg_e}')

        # --- RETRIEVAL-045v5: Chain-guided context pruning ---
        # When entity chain traces a deep path (>= threshold searched entities),
        # prune standard retrieval concepts that share NO entity with the chain.
        # Feature-gated, threshold-configurable. Ported from pith_agent.py:462-490.
        _CHAIN_PRUNE_ENV = os.environ.get(
            'PITH_CHAIN_CONTEXT_PRUNE', ''
        ).lower() in ('true', '1')
        # RETRIEVAL-CHAIN-GATE-001: Only chain-context-prune on multihop queries.
        # Uses _adaptive_config from RETRIEVAL-060 router (set at ~line 2665).
        _CHAIN_PRUNE_ENABLED = (
            _CHAIN_PRUNE_ENV
            and _adaptive_config is not None
            and _adaptive_config.use_multihop
        )
        _CHAIN_PRUNE_THRESHOLD = int(os.environ.get('PITH_CHAIN_PRUNE_THRESHOLD', '4'))
        if _CHAIN_PRUNE_ENABLED and top_results:
            try:
                # Get searched entities from entity chain retriever (set by S4.7)
                _cp_searched = set()
                try:
                    from app.entity_chain import get_entity_chain_retriever
                    _cp_ecr = get_entity_chain_retriever()
                    if _cp_ecr:
                        _cp_searched = getattr(_cp_ecr, 'last_searched_entities', set()) or set()
                except Exception:
                    pass

                if len(_cp_searched) >= _CHAIN_PRUNE_THRESHOLD:
                    _cp_before = len(top_results)
                    _cp_pruned = []
                    for r in top_results:
                        # Always keep entity-chain and shadow-expanded concepts
                        if r.concept_id in {sr.concept_id for sr in shadow_expanded}:
                            _cp_pruned.append(r)
                            continue
                        # Keyword-supplemented concepts: only exempt if they also
                        # overlap with chain entities (RETRIEVAL-045v5b fix)
                        if r.concept_id in _kw_supplemented_ids:
                            _cp_summary_lower_kw = (r.summary or '').lower()
                            _cp_kw_match = any(
                                ent.lower() in _cp_summary_lower_kw
                                for ent in _cp_searched
                                if len(ent) > 2
                            )
                            if _cp_kw_match:
                                _cp_pruned.append(r)
                                continue
                            # else: fall through — kw-supplemented but off-chain, will be pruned
                        # Check entity overlap with chain
                        _cp_summary_lower = (r.summary or '').lower()
                        _cp_match = any(
                            ent.lower() in _cp_summary_lower
                            for ent in _cp_searched
                            if len(ent) > 2
                        )
                        if _cp_match:
                            _cp_pruned.append(r)
                        # else: pruned (no entity overlap with chain)
                    top_results = _cp_pruned
                    _cp_removed = _cp_before - len(top_results)
                    if _cp_removed:
                        logger.info(
                            f'RETRIEVAL-045v5: Chain-guided prune: {_cp_before}->{len(top_results)} '
                            f'(removed {_cp_removed}, chain_ents={len(_cp_searched)}, '
                            f'threshold={_CHAIN_PRUNE_THRESHOLD})'
                        )
            except Exception as _cp_e:
                logger.warning(f'RETRIEVAL-045v5: Chain-guided prune failed (non-fatal): {_cp_e}')

        # --- RETRIEVAL-045v4: Gold-first reordering ---
        # Promote entity-chain-relevant concepts to front of context.
        # Uses entity overlap to identify chain concepts.
        # Feature-gated. Adapted from pith_agent.py:498-511.
        # Skip gold-first if chain-order is also enabled (chain-order subsumes it) [GAUNTLET B1]
        _GOLD_FIRST_ENABLED = os.environ.get(
            'PITH_GOLD_FIRST_REORDER', ''
        ).lower() in ('true', '1')
        _CHAIN_ORDER_ALSO_ON = os.environ.get('PITH_CHAIN_ORDER', '').lower() in ('true', '1')
        if _GOLD_FIRST_ENABLED and _CHAIN_ORDER_ALSO_ON:
            logger.info('RETRIEVAL-045v4: Skipped (PITH_CHAIN_ORDER takes precedence)')
            _GOLD_FIRST_ENABLED = False
        if _GOLD_FIRST_ENABLED and top_results:
            try:
                _gf_searched = set()
                try:
                    from app.entity_chain import get_entity_chain_retriever
                    _gf_ecr = get_entity_chain_retriever()
                    if _gf_ecr:
                        _gf_searched = getattr(_gf_ecr, 'last_searched_entities', set()) or set()
                except Exception:
                    pass

                if _gf_searched:
                    def _is_on_chain(r):
                        s = (r.summary or '').lower()
                        return any(ent.lower() in s for ent in _gf_searched if len(ent) > 2)

                    _gf_chain = [r for r in top_results if _is_on_chain(r)]
                    _gf_rest = [r for r in top_results if not _is_on_chain(r)]
                    if _gf_chain:
                        top_results = _gf_chain + _gf_rest
                        logger.info(
                            f'RETRIEVAL-045v4: Gold-first reorder: promoted '
                            f'{len(_gf_chain)} on-chain concepts to front'
                        )
            except Exception as _gf_e:
                logger.warning(f'RETRIEVAL-045v4: Gold-first reorder failed (non-fatal): {_gf_e}')

        # --- RETRIEVAL-036: Chain-ordered context ---
        # Sort: on-chain concepts first (by relevance), then standard (by relevance).
        # Feature-gated. Adapted from pith_agent.py:512-535.
        _CHAIN_ORDER_ENABLED = os.environ.get(
            'PITH_CHAIN_ORDER', ''
        ).lower() in ('true', '1')
        if _CHAIN_ORDER_ENABLED and top_results:
            try:
                _co_searched = set()
                try:
                    from app.entity_chain import get_entity_chain_retriever
                    _co_ecr = get_entity_chain_retriever()
                    if _co_ecr:
                        _co_searched = getattr(_co_ecr, 'last_searched_entities', set()) or set()
                except Exception:
                    pass

                if _co_searched:
                    def _co_on_chain(r):
                        s = (r.summary or '').lower()
                        return any(ent.lower() in s for ent in _co_searched if len(ent) > 2)

                    _co_chain = sorted(
                        [r for r in top_results if _co_on_chain(r)],
                        key=lambda x: x.relevance_score, reverse=True,
                    )
                    _co_standard = sorted(
                        [r for r in top_results if not _co_on_chain(r)],
                        key=lambda x: x.relevance_score, reverse=True,
                    )
                    top_results = _co_chain + _co_standard
                    logger.info(
                        f'RETRIEVAL-036: Chain-ordered {len(_co_chain)} chain + '
                        f'{len(_co_standard)} standard concepts'
                    )
            except Exception as _co_e:
                logger.warning(f'RETRIEVAL-036: Chain-ordered context failed (non-fatal): {_co_e}')

        # --- RETRIEVAL-GATE: Cross-encoder reranker gate on inflated-score concepts ---
        # Score inflation from stacking domain boost (+0.15), temporal boost (×1.15),
        # keyword supplement (up to 1.0), and fact supplement (up to 1.0) can push
        # augmented concepts above 1.0, outranking semantically relevant results.
        # This gate re-scores concepts with relevance_score > _GATE_SCORE_THRESHOLD
        # through the cross-encoder against the raw user query. If the cross-encoder
        # score is below _GATE_CE_THRESHOLD, the concept is demoted to the CE score
        # (capped at _GATE_DEMOTE_CAP) so it doesn't drown out semantic matches.
        # Feature-gated via PITH_SCORE_GATE env var. Budget: ~20-40ms for 5-15 concepts.
        _SCORE_GATE_ENABLED = os.environ.get('PITH_SCORE_GATE', '').lower() in ('true', '1')
        _GATE_SCORE_THRESHOLD = float(os.environ.get('PITH_SCORE_GATE_THRESHOLD', '1.0'))
        _GATE_CE_THRESHOLD = float(os.environ.get('PITH_SCORE_GATE_CE_THRESHOLD', '0.3'))
        _GATE_DEMOTE_CAP = float(os.environ.get('PITH_SCORE_GATE_DEMOTE_CAP', '0.4'))
        if _SCORE_GATE_ENABLED and top_results:
            try:
                _sg_inflated = [
                    (i, r) for i, r in enumerate(top_results)
                    if r.relevance_score > _GATE_SCORE_THRESHOLD
                ]
                if _sg_inflated:
                    from app.reranker import _get_cross_encoder
                    _sg_model = _get_cross_encoder()
                    _sg_query = request.message or search_query
                    _sg_pairs = [(_sg_query, r.summary or '') for _, r in _sg_inflated]

                    import numpy as np
                    _sg_scores = _sg_model.predict(_sg_pairs, show_progress_bar=False)
                    if not hasattr(_sg_scores, '__iter__'):
                        _sg_scores = [_sg_scores]
                    _sg_scores = np.array(_sg_scores, dtype=np.float32)

                    _sg_demoted = 0
                    _sg_kept = 0
                    _sg_details = []
                    for idx, ((orig_idx, r), ce_score) in enumerate(zip(_sg_inflated, _sg_scores)):
                        ce_score_f = float(ce_score)
                        if ce_score_f < _GATE_CE_THRESHOLD:
                            # Demote: cap relevance to the lesser of CE score and demote cap
                            _old_score = r.relevance_score
                            r.relevance_score = min(_GATE_DEMOTE_CAP, max(0.05, ce_score_f))
                            _sg_details.append(
                                f'  DEMOTE {r.concept_id}: {_old_score:.3f}->{r.relevance_score:.3f} CE={ce_score_f:.4f} "{(r.summary or "")[:60]}"'
                            )
                            _sg_demoted += 1
                        else:
                            _sg_details.append(
                                f'  KEEP   {r.concept_id}: score={r.relevance_score:.3f} CE={ce_score_f:.4f} "{(r.summary or "")[:60]}"'
                            )
                            _sg_kept += 1

                    # Re-sort by relevance after demotion
                    if _sg_demoted > 0:
                        top_results.sort(key=lambda x: x.relevance_score, reverse=True)

                    _sg_elapsed = time.perf_counter()
                    logger.info(
                        f'RETRIEVAL-GATE: Scored {len(_sg_inflated)} inflated concepts '
                        f'(demoted={_sg_demoted}, kept={_sg_kept}, '
                        f'threshold={_GATE_SCORE_THRESHOLD}, ce_gate={_GATE_CE_THRESHOLD})\n'
                        + '\n'.join(_sg_details)
                    )
            except Exception as _sg_e:
                logger.warning(f'RETRIEVAL-GATE: Score gate failed (non-fatal): {_sg_e}')

        # --- S5: Context assembly (budget: 3ms) ---
        # Build response: trim evidence to top 2 per concept, compute graph_density
        shadow_ids = {r.concept_id for r in shadow_expanded}  # S4.1 shadow tracking

        # PERF: Load all concepts once into a cache. Previously load_concept was
        # called 4-5 times per concept per turn (S5 assembly, contradiction detection,
        # budget governance, constraint assembly, staleness filtering) = 80-100 DB reads.
        # With cache: ~20 DB reads total.
        _concept_cache: dict = {}
        all_candidate_ids = {r.concept_id for r in top_results}
        for cid in all_candidate_ids:
            c = load_concept(cid, track_access=True)
            if c:
                _concept_cache[cid] = c

        # --- ARCH-D05: Maturity promotion on retrieval access ---
        # Rate-limited to once per concept per session to avoid hot-path overhead
        for cid, c in _concept_cache.items():
            if (getattr(c, "maturity", "ESTABLISHED") == "PROVISIONAL"
                    and cid not in self._promoted_this_session):
                try:
                    self._maybe_promote_maturity(cid)
                    self._promoted_this_session.add(cid)
                except Exception as e:
                    logger.debug(f"ARCH-D05: Promotion check failed for {cid}: {e}")

        # --- S2.9: Maturity gate with circuit breaker (Retrieval Defense W3) ---
        # maturity_filtered_count initialized before S4 to accumulate S4 + S2.9 counts
        maturity_gate_bypassed = False
        try:
            from app.config import FEATURE_FLAGS as _ff

            if _ff.get("INGESTION_VALIDATION_ENABLED", False):
                BLOCKED_MATURITIES = {"QUARANTINED", "DISCARDED"}
                _recency_cutoff = (_utc_now() - timedelta(hours=QUARANTINE_RECENCY_EXEMPT_HOURS)).isoformat()
                MIN_ACTIVATION_FLOOR = 3
                pre_filter_count = len(top_results)

                def _maturity_of(cid):
                    return getattr(_concept_cache.get(cid), "maturity", "ESTABLISHED")

                filtered_results = [
                    r
                    for r in top_results
                    if _maturity_of(r.concept_id) not in BLOCKED_MATURITIES
                    or (
                        _maturity_of(r.concept_id) == "QUARANTINED"
                        and getattr(_concept_cache.get(r.concept_id), "created_at", "") > _recency_cutoff
                    )
                ]
                if len(filtered_results) >= MIN_ACTIVATION_FLOOR:
                    s29_filtered = pre_filter_count - len(filtered_results)
                    maturity_filtered_count += s29_filtered
                    top_results = filtered_results
                    if s29_filtered > 0:
                        logger.info(
                            f"W3: Maturity gate filtered {s29_filtered} concepts at S2.9 "
                            f"({pre_filter_count} → {len(top_results)}, total filtered={maturity_filtered_count})"
                        )
                else:
                    # Circuit breaker: don't empty the result set
                    maturity_gate_bypassed = True
                    logger.warning(
                        f"W3: Maturity gate BYPASSED — only {len(filtered_results)} of "
                        f"{pre_filter_count} would survive (floor={MIN_ACTIVATION_FLOOR})"
                    )
        except Exception as e:
            logger.warning(f"W3: Maturity gate failed (non-fatal): {e}")

        # --- Wave 4b: Batch prediction INSERT [FIX C1] ---
        # Log predictions for all retrieved concepts for calibration tracking
        try:
            from app.traces import batch_log_predictions

            pred_rows = [
                {"concept_id": cid, "confidence_at_retrieval": _concept_cache[cid].confidence} for cid in _concept_cache
            ]
            sid = request.session_id or (self.current_session.session_id if self.current_session else "unknown")
            if pred_rows:
                batch_log_predictions(pred_rows, sid)
        except Exception as e:
            logger.debug(f"Wave 4b: prediction logging skipped: {e}")

        activated = []
        # TEMPORAL_AWARENESS v2.4: Hoist temporal computation inputs (compute once)
        _ta_now = _utc_now()
        _ta_session_start = None
        if self.current_session and self.current_session.started_at:
            try:  # noqa: SIM105
                _ta_session_start = _ensure_aware(datetime.fromisoformat(self.current_session.started_at))
            except (ValueError, TypeError):
                pass

        # --- RETRIEVAL-037c: Fetch serial ordering for conflict resolution ---
        # Moved here from pre-graph-walk to cover ALL final top_results (including graph-walked).
        # Uses ROW_NUMBER(ORDER BY created_at) for temporal ordering.
        if top_results:
            try:
                _sr_conn = _get_connection()
                _sr_ids = [r.concept_id for r in top_results]
                _sr_placeholders = ",".join("?" * len(_sr_ids))
                _sr_rows = _sr_conn.execute(
                    f"SELECT id, temporal_rank FROM ("
                    f"  SELECT id, ROW_NUMBER() OVER (ORDER BY created_at ASC) as temporal_rank"
                    f"  FROM concepts WHERE status = 'active'"
                    f") WHERE id IN ({_sr_placeholders})",
                    _sr_ids,
                ).fetchall()
                _serial_order_map = {row[0]: row[1] for row in _sr_rows}
                if _serial_order_map:
                    logger.info(f'RETRIEVAL-037c: serial_order_map built for {len(_serial_order_map)}/{len(_sr_ids)} concepts')
            except Exception as e:
                logger.debug(f"RETRIEVAL-037c: temporal rank fetch failed (non-fatal): {e}")
        for result in top_results:
            concept = _concept_cache.get(result.concept_id)
            if not concept:
                continue

            # Extract top 2 evidence items (prefer structured Evidence content)
            key_evidence = self._extract_top_evidence(concept.evidence, limit=2)

            # Get 1-hop associations for this concept
            assoc_ids = association_map.get(result.concept_id, [])

            # GOV: Generate trust signal (uncertainty qualifiers)
            trust_sig = None
            try:
                from app.uncertainty import build_trust_signal

                concept_data = concept.metadata if hasattr(concept, "metadata") and concept.metadata else {}
                auth = concept.authority_score if concept.authority_score is not None else concept.confidence
                curr = concept.currency_score if concept.currency_score is not None else 0.5
                curr_status = getattr(concept, "currency_status", None) or "ACTIVE"
                trust_sig = build_trust_signal(
                    concept_id=result.concept_id,
                    concept_data=concept_data,
                    authority_score=auth or 0.0,
                    currency_score=curr or 0.5,
                    currency_status=curr_status,
                )
            except Exception:
                pass  # Trust signals are enrichment, not critical path

            # TEMPORAL_AWARENESS v2.4: Compute freshness from concept object
            _ta_age, _ta_label = _compute_freshness(concept.created_at, _ta_now, _ta_session_start)

            # DEBT-207: Annotate experiment-origin concepts in retrieval responses
            _concept_signals = getattr(concept, "signals", None) or []
            _is_exp_origin = any(s.startswith("experiment:") for s in _concept_signals)
            _display_summary = f"[EXP] {result.summary}" if _is_exp_origin else result.summary

            # INGEST-016: Append observed date for temporal queries (Phase 3)
            if hasattr(self, '_s26_temporal_annotations') and result.concept_id in self._s26_temporal_annotations:
                _display_summary += f" [observed: {self._s26_temporal_annotations[result.concept_id]}]"

            # RETRIEVAL-034 Layer 1: Temporal annotation for stale concepts
            # Proven prefix pattern ([SUPERSEDED], [EXP], [ALWAYS]) — AI naturally
            # produces temporal framing from in-text "[as of X days ago]" annotation.
            from app.config import STALE_TRANSPARENCY_ENABLED
            if STALE_TRANSPARENCY_ENABLED:
                if (curr_status in ("CONTRADICTED", "CONTESTED")
                        and _ta_label and _ta_label != "unknown date"
                        and "[SUPERSEDED]" not in _display_summary):
                    _display_summary = f"[as of {_ta_label}] {_display_summary}"

            # INGEST-037 Layer 3: Fetch verbatim fragments if requested
            _vf_list = []
            if _include_verbatim:
                try:
                    from app.storage import get_verbatim_fragments
                    _vf_list = get_verbatim_fragments(result.concept_id, limit=5)
                except Exception as _vf_err:
                    logger.debug(f"INGEST-037: Fragment fetch failed for {result.concept_id}: {_vf_err}")

            activated.append(
                ActivatedConcept(
                    concept_id=result.concept_id,
                    summary=_display_summary,
                    confidence=result.confidence,
                    relevance_score=round(result.relevance_score, 4),
                    knowledge_area=result.knowledge_area or "unknown",
                    key_evidence=key_evidence,
                    associations=assoc_ids[:10],  # Cap at 10 associations
                    shadow_expanded=(result.concept_id in shadow_ids),  # S4.1 tag
                    trust_signal=trust_sig,
                    age_minutes=_ta_age,
                    freshness_label=_ta_label,
                    currency_status=curr_status,  # RETRIEVAL-034 Layer 3
                    ka_relative_authority=getattr(result, "ka_relative_authority", None),
                    serial_order=_serial_order_map.get(result.concept_id),  # RETRIEVAL-037c
                    original_date=getattr(concept, 'original_date', None) if concept else None,  # TEMPORAL-002
                    verbatim_fragments=_vf_list,  # INGEST-037 Layer 3
                )
            )

        # Append ambient principles (from S4.8) to activated list
        for ap in ambient_injected:
            _ta_age, _ta_label = _compute_freshness(ap.get("created_at"), _ta_now, _ta_session_start)
            # RETRIEVAL-034 Layer 3: Surface currency_status for ambient concepts
            _ap_c = _concept_cache.get(ap["concept_id"])
            _ap_curr = getattr(_ap_c, "currency_status", None) or "ACTIVE" if _ap_c else "ACTIVE"
            activated.append(
                ActivatedConcept(
                    concept_id=ap["concept_id"],
                    summary=f"[PRINCIPLE] {ap['summary']}",
                    confidence=ap["confidence"],
                    relevance_score=0.0,  # Not keyword-matched; ambient injection
                    knowledge_area=ap.get("knowledge_area", "general"),
                    key_evidence=[],
                    associations=[],
                    age_minutes=_ta_age,
                    freshness_label=_ta_label,
                    currency_status=_ap_curr,  # RETRIEVAL-034 Layer 3
                )
            )

        # Append synthetic counting results (RETRIEVAL-026)
        # Counting dispatch returns synthetic concepts with no DB backing.
        # These bypass the main activation loop (which requires load_concept).
        for result in top_results:
            if result.concept_id.startswith("counting_result_"):
                activated.append(
                    ActivatedConcept(
                        concept_id=result.concept_id,
                        summary=result.summary,
                        confidence=result.confidence,
                        relevance_score=round(result.relevance_score, 4),
                        knowledge_area=result.knowledge_area or "aggregate",
                        key_evidence=[],
                        associations=[],
                    )
                )

        # Append always-activate concepts (from S4.6)
        for ao in always_on_injected:
            # P4-PREREQ: Ensure concept is in cache BEFORE freshness computation
            if ao["concept_id"] not in _concept_cache:
                _ao_concept = load_concept(ao["concept_id"], track_access=False)
                if _ao_concept:
                    _concept_cache[ao["concept_id"]] = _ao_concept
            _ao_c = _concept_cache.get(ao["concept_id"])
            _ta_age, _ta_label = _compute_freshness(
                _ao_c.created_at if _ao_c else ao.get("created_at"), _ta_now, _ta_session_start
            )
            # RETRIEVAL-034 Layer 3: Surface currency_status for always-activate
            _ao_curr = getattr(_ao_c, "currency_status", None) or "ACTIVE" if _ao_c else "ACTIVE"
            activated.append(
                ActivatedConcept(
                    concept_id=ao["concept_id"],
                    summary=f"[ALWAYS] {ao['summary']}",
                    confidence=ao["confidence"],
                    relevance_score=0.0,  # Always-injected regardless of topic
                    knowledge_area=ao.get("knowledge_area", "general"),
                    key_evidence=[],
                    associations=[],
                    age_minutes=_ta_age,
                    freshness_label=_ta_label,
                    currency_status=_ao_curr,  # RETRIEVAL-034 Layer 3
                )
            )

        # Append firmware entries (from S4.7) — uses synthetic concept IDs
        firmware_ids = set()
        for fw in firmware_entries:
            fw_id = f"firmware:{fw['id']}"
            firmware_ids.add(fw_id)
            activated.append(
                ActivatedConcept(
                    concept_id=fw_id,
                    summary=f"[FIRMWARE] {fw['summary']}",
                    confidence=1.0,  # Firmware is developer-verified truth
                    relevance_score=0.0,  # Always-injected regardless of topic
                    knowledge_area=fw.get("category", "system"),
                    key_evidence=[],
                    associations=[],
                )
            )

        # --- CKO-003: Compound Knowledge Object retrieval (budget: 5ms) ---
        # Surface relevant CKOs alongside individual concepts.
        # CKOs bundle related concepts into coherent wholes (analyses, plans).
        try:
            from app.cko import search_ckos

            _cko_results = search_ckos(_get_connection(), max_results=2)  # DEBT-211: use module-level import
            for cko in _cko_results:
                activated.append(
                    ActivatedConcept(
                        concept_id=f"cko:{cko.id}",
                        summary=f"[CKO] {cko.title}: {cko.synthesis[:200]}",
                        confidence=cko.confidence or 0.5,
                        relevance_score=0.0,
                        knowledge_area=cko.knowledge_area or "general",
                        key_evidence=[],
                        associations=cko.concept_ids[:5] if cko.concept_ids else [],
                    )
                )
        except Exception:
            pass  # CKO retrieval is enrichment, not critical path

        # --- S4.9: Recency baseline injection (budget: 2ms) ---
        # WHY: Temporal retrieval has 40% classifier recall (L1). When classifier
        # misses, NO recent concepts surface. This injects 1-2 recent concepts as
        # a floor, ensuring the agent always has access to the latest work context
        # regardless of classification accuracy. This is a band-aid; true fix is
        # improving classify_question() or client-side hints (Tier 2). [A-H14]
        try:
            cutoff = (_utc_now() - timedelta(hours=RECENCY_WINDOW_HOURS)).isoformat()
            recent = load_recent_concepts(since_iso=cutoff, limit=5, min_confidence=RECENCY_MIN_CONFIDENCE)

            # F1 + A-C4: Filter wrong correction concepts
            recent = [c for c in recent if not c["concept_id"].startswith("correction_")]

            # F3: Filter auto-learned (low-quality auto-extracted concepts)
            recent = [c for c in recent if c.get("confidence", 0) >= RECENCY_MIN_CONFIDENCE]

            # Bug 5 fix: Filter QUARANTINED/DISCARDED from recency injection
            # Third maturity gate (after S2.9 and S4.1) — prevents quarantined
            # concepts from entering activation via the recency path.
            _recency_blocked_maturities = {"QUARANTINED", "DISCARDED"}
            _s49_recency_cutoff = (_utc_now() - timedelta(hours=QUARANTINE_RECENCY_EXEMPT_HOURS)).isoformat()
            recency_pre_count = len(recent)
            recent = [
                c
                for c in recent
                if c.get("maturity", "ESTABLISHED") not in _recency_blocked_maturities
                or (c.get("maturity", "ESTABLISHED") == "QUARANTINED" and c.get("created_at", "") > _s49_recency_cutoff)
            ]
            recency_maturity_filtered = recency_pre_count - len(recent)
            if recency_maturity_filtered > 0:
                logger.info(
                    f"S4.9: Maturity gate filtered {recency_maturity_filtered} "
                    f"quarantined/discarded concepts from recency injection"
                )

            # F9: Dedup against ALL already-activated concept IDs
            recency_existing_ids = {ac.concept_id for ac in activated}
            candidates = [c for c in recent if c["concept_id"] not in recency_existing_ids]

            recency_injected = 0
            for c in candidates[:RECENCY_MAX_INJECT]:
                _ta_age, _ta_label = _compute_freshness(c.get("created_at"), _ta_now, _ta_session_start)
                activated.append(
                    ActivatedConcept(
                        concept_id=c["concept_id"],
                        summary=c["summary"],
                        confidence=c["confidence"],
                        relevance_score=RECENCY_RELEVANCE_SCORE,
                        knowledge_area=c.get("knowledge_area", "general"),
                        key_evidence=[],
                        associations=[],
                        shadow_expanded=False,
                        age_minutes=_ta_age,
                        freshness_label=_ta_label,
                    )
                )
                recency_existing_ids.add(c["concept_id"])
                recency_injected += 1

            # A-H16: Always log, include filter stats
            logger.info(
                f"S4.9: Recency injection — found={len(recent)}, "
                f"after_filters={len(candidates)}, injected={recency_injected}"
            )
        except Exception as e:
            # A-C10: Specific exception types — don't mask ImportError
            logger.warning(f"S4.9: Recency injection failed ({type(e).__name__}): {e}")

        t_injection = time.perf_counter()  # PERF-016: Phase A checkpoint

        # --- S5.5: Staleness filtering (budget: 2ms) ---
        # SILENTLY EXCLUDE stale concepts rather than flagging them.
        # Principle: "absence recoverable, stale not" — sending stale data causes
        # the AI to act on garbage (catastrophic), while omitting data merely
        # reduces context (recoverable).
        staleness_filtered_count = 0
        now = _utc_now()
        STALE_THRESHOLD_HOURS = 48
        # BENCH-FIX: Disable staleness filter in benchmark mode.
        # Benchmark brains have fixed created_at timestamps that cross the
        # 48h threshold during multi-day sprint cycles, causing catastrophic
        # 91-question regressions (all facts filtered as "stale").
        if os.environ.get("PITH_BENCHMARK_MODE", "false").lower() == "true":
            STALE_THRESHOLD_HOURS = 999_999
        PLAN_INDICATORS = {"goal", "decision", "observation", "constraint"}
        # P1-1 fix: always-activate concepts must never be staleness-filtered
        # P0-5: firmware entries must never be staleness-filtered
        always_on_ids = {ao["concept_id"] for ao in always_on_injected} | firmware_ids
        filtered_activated = []
        for ac in activated:
            # P1-1: Skip staleness check for always-activate concepts
            if ac.concept_id in always_on_ids:
                filtered_activated.append(ac)
                continue
            concept = _concept_cache.get(ac.concept_id)
            if not concept:
                filtered_activated.append(ac)
                continue
            # Only filter v1 concepts (never evolved) that are old enough
            if concept.version and concept.version != "v1":
                filtered_activated.append(ac)
                continue
            created = concept.created_at
            if not created:
                filtered_activated.append(ac)
                continue
            try:
                if isinstance(created, str):
                    created_dt = _ensure_aware(
                        datetime.fromisoformat(created.replace("Z", "+00:00").replace("+00:00", ""))
                    )
                else:
                    created_dt = _ensure_aware(created) if isinstance(created, datetime) else created
                age_hours = (now - created_dt).total_seconds() / 3600
                if age_hours > STALE_THRESHOLD_HOURS and concept.concept_type in PLAN_INDICATORS:
                    staleness_filtered_count += 1
                    logger.debug(
                        f"S5.5: silently excluded stale concept '{ac.concept_id}' "
                        f"({concept.concept_type}, {age_hours:.0f}h old, never evolved)"
                    )
                    continue  # Exclude from results
            except Exception:
                pass  # Staleness detection is best-effort
            filtered_activated.append(ac)

        activated = filtered_activated

        # --- S5.6: Structured Activation Layer (SAL V0) ---
        # Post-filter, pre-response structural analysis.
        # Toggle boundary: FEATURE_FLAGS["SAL_ENABLED"] = False -> zero overhead.
        _sal_result = None
        from app.config import get_feature_flag as _get_ff
        if _get_ff("SAL_ENABLED"):
            try:
                from app.structured_activation import process_sal
                from app.storage import get_adjacency_graph  # Cached <0.01ms
                _sal_result = process_sal(
                    activated_concepts=activated,
                    concept_cache=_concept_cache,
                    query=search_query or request.message or "",
                    adjacency_graph=get_adjacency_graph(),  # Cached dict, not raw edges list
                )
                if _sal_result and not _sal_result.get("fallback_used"):
                    logger.info(
                        f"SAL: mode={_sal_result.get('mode_used')}, "
                        f"clusters={len(_sal_result.get('clusters', []))}, "
                        f"surprise={len(_sal_result.get('surprise_buffer', []))}, "
                        f"latency={_sal_result.get('processing_time_ms', 0):.1f}ms"
                    )
            except Exception as _sal_e:
                logger.warning(f"SAL: Processing failed (non-fatal): {_sal_e}")
                _sal_result = None

        # --- SAL V1 Consumer: Transform raw SAL dict into LLM context string ---
        _sal_context = None
        if _sal_result is not None:
            try:
                from app.sal_consumer import format_sal_context
                _sal_context = format_sal_context(_sal_result)
            except Exception as _sal_consumer_e:
                logger.warning(f"SAL consumer: Failed (non-fatal): {_sal_consumer_e}")
                _sal_context = None

        # --- S5.5b: STALE-002 CONTESTED concept de-ranking (budget: <1ms) ---
        # Concepts flagged CONTESTED/CONTRADICTED by contradiction detection should
        # not compete equally with ACTIVE concepts for limited retrieval slots.
        _contested_demotion = float(os.environ.get("PITH_CONTESTED_DEMOTION", "0.5"))
        if _contested_demotion < 1.0:
            _contested_demoted_count = 0
            for ac in activated:
                _ac_concept = _concept_cache.get(ac.concept_id)
                if _ac_concept:
                    _ac_currency = getattr(_ac_concept, "currency_status", "ACTIVE")
                    if _ac_currency in ("CONTESTED", "CONTRADICTED"):
                        # A5: Guard against relevance_score=None
                        _prev_score = ac.relevance_score if ac.relevance_score is not None else 1.0
                        ac.relevance_score = _prev_score * _contested_demotion
                        _contested_demoted_count += 1
                        logger.debug(
                            f"STALE-002: de-ranked {_ac_currency} concept '{ac.concept_id}' "
                            f"relevance {_prev_score:.3f} → {ac.relevance_score:.3f}"
                        )
            if _contested_demoted_count:
                logger.debug(f"STALE-002: de-ranked {_contested_demoted_count} CONTESTED concepts")

        # --- S5.6: RETRIEVAL-013 Temporal evolution check (budget: <2ms) ---
        try:
            import time as _evo_time

            from app.config import EVOLUTION_COSINE_MAX, EVOLUTION_COSINE_MIN, EVOLUTION_SUPPRESSION_WEIGHT
            from app.embedding import embedding_engine as _evo_emb
            from app.supersession import TYPE_RANK

            _s56_t0 = _evo_time.perf_counter()
            _evo_pairs_evaluated = 0
            _evo_pairs_suppressed = 0
            _evo_total_suppression = 0.0
            # RETRIEVAL-016: Per-precondition skip counters
            _skip = {
                "no_cache": 0,
                "diff_ka": 0,
                "no_time": 0,
                "same_time": 0,
                "no_auth": 0,
                "auth_lte": 0,
                "type_rank": 0,
                "no_embed": 0,
                "cosine_oor": 0,
            }

            # RETRIEVAL-018: Backfill _concept_cache for concepts added after initial
            # cache build (shadow expansion S4.1, recency S4.9, etc.)
            for _ac in activated:
                if _ac.concept_id not in _concept_cache:
                    _backfill_c = load_concept(_ac.concept_id, track_access=False)
                    if _backfill_c:
                        _concept_cache[_ac.concept_id] = _backfill_c

            # G2-A2: Only retrieval results participate (not ambient/AA/firmware)
            retrieval_candidates = [ac for ac in activated if ac.relevance_score > 0]

            if (
                len(retrieval_candidates) >= 2
                and _evo_emb is not None
                and getattr(_evo_emb, "_id_to_pos", None) is not None
                and getattr(_evo_emb, "_index_matrix", None) is not None
            ):
                n_rc = len(retrieval_candidates)
                for i in range(n_rc):
                    for j in range(i + 1, n_rc):
                        ac_a = retrieval_candidates[i]
                        ac_b = retrieval_candidates[j]

                        # G2-A1: Look up full Concept from cache for metadata
                        concept_a = _concept_cache.get(ac_a.concept_id)
                        concept_b = _concept_cache.get(ac_b.concept_id)
                        if not concept_a or not concept_b:
                            _skip["no_cache"] += 1
                            continue

                        # Precondition 1: Same knowledge area
                        ka_a = getattr(concept_a, "knowledge_area", None) or concept_a.metadata.get(
                            "knowledge_area", ""
                        )
                        ka_b = getattr(concept_b, "knowledge_area", None) or concept_b.metadata.get(
                            "knowledge_area", ""
                        )
                        if ka_a != ka_b:
                            _skip["diff_ka"] += 1
                            continue

                        # Precondition 2: Determine temporal order (B is newer)
                        ca = getattr(concept_a, "created_at", None)
                        cb = getattr(concept_b, "created_at", None)
                        if not ca or not cb:
                            _skip["no_time"] += 1
                            continue
                        ca_str = ca if isinstance(ca, str) else str(ca)
                        cb_str = cb if isinstance(cb, str) else str(cb)
                        if ca_str == cb_str:
                            _skip["same_time"] += 1
                            continue

                        # Orient: older=A, newer=B
                        if cb_str > ca_str:
                            older_ac, newer_ac = ac_a, ac_b
                            older_c, newer_c = concept_a, concept_b
                        else:
                            older_ac, newer_ac = ac_b, ac_a
                            older_c, newer_c = concept_b, concept_a

                        # Precondition 3+5: Both have authority, B > A
                        auth_a = getattr(older_c, "authority_score", None)
                        auth_b = getattr(newer_c, "authority_score", None)
                        if auth_a is None or auth_b is None:
                            _skip["no_auth"] += 1
                            continue
                        if auth_b <= auth_a:
                            _skip["auth_lte"] += 1
                            continue

                        # Precondition 6: Type maturity (B >= A)
                        type_a = getattr(older_c, "concept_type", None) or older_c.metadata.get(
                            "concept_type", "observation"
                        )
                        type_b = getattr(newer_c, "concept_type", None) or newer_c.metadata.get(
                            "concept_type", "observation"
                        )
                        rank_a = TYPE_RANK.get(type_a, 1)
                        rank_b = TYPE_RANK.get(type_b, 1)
                        if rank_b < rank_a:
                            _skip["type_rank"] += 1
                            continue

                        # Precondition 4+7: Both have embeddings, cosine in [0.50, 0.82)
                        pos_a = _evo_emb._id_to_pos.get(older_ac.concept_id)
                        pos_b = _evo_emb._id_to_pos.get(newer_ac.concept_id)
                        if pos_a is None or pos_b is None:
                            _skip["no_embed"] += 1
                            continue

                        cosine = float(_evo_emb._index_matrix[pos_a] @ _evo_emb._index_matrix[pos_b])
                        if not (EVOLUTION_COSINE_MIN <= cosine < EVOLUTION_COSINE_MAX):
                            _skip["cosine_oor"] += 1
                            continue

                        _evo_pairs_evaluated += 1

                        # V2 Additive penalty computation
                        cosine_factor = (cosine - EVOLUTION_COSINE_MIN) / (EVOLUTION_COSINE_MAX - EVOLUTION_COSINE_MIN)
                        authority_delta = min(1.0, (auth_b - auth_a) / 0.20)
                        type_gap = rank_b - rank_a
                        type_factor = min(1.0, type_gap / 3) if type_gap > 0 else 0.0

                        signal_strength = 0.40 * cosine_factor + 0.35 * authority_delta + 0.25 * type_factor
                        suppression = signal_strength * EVOLUTION_SUPPRESSION_WEIGHT
                        older_ac.relevance_score *= 1.0 - suppression

                        _evo_pairs_suppressed += 1
                        _evo_total_suppression += suppression

            _s56_ms = (_evo_time.perf_counter() - _s56_t0) * 1000
            # RETRIEVAL-016/018: Always log skip breakdown so we can see what kills pairs
            _total_skips = sum(_skip.values())
            _skip_str = " ".join(f"{k}={v}" for k, v in _skip.items() if v > 0)
            if _evo_pairs_evaluated > 0:
                _avg_supp = (_evo_total_suppression / _evo_pairs_suppressed * 100) if _evo_pairs_suppressed else 0
                logger.info(
                    f"S5.6: evaluated={_evo_pairs_evaluated} "
                    f"suppressed={_evo_pairs_suppressed} avg_suppression={_avg_supp:.1f}% "
                    f"skips({_total_skips}): {_skip_str} "
                    f"duration={_s56_ms:.2f}ms"
                )
            else:
                logger.info(
                    f"S5.6: NO pairs passed preconditions. "
                    f"candidates={len(retrieval_candidates)} "
                    f"skips({_total_skips}): {_skip_str or 'none'} "
                    f"duration={_s56_ms:.2f}ms"
                )

            # OBS-004: Emit S5.6 evolution metrics for observability
            try:
                from app.metrics import metrics as _s56_metrics

                _s56_metrics.record("evolution_pairs_evaluated", _evo_pairs_evaluated)
                _s56_metrics.record("evolution_suppressed_count", _evo_pairs_suppressed)
                if _evo_pairs_suppressed > 0:
                    _s56_metrics.record(
                        "evolution_avg_suppression",
                        round(_evo_total_suppression / _evo_pairs_suppressed, 4),
                    )
                _s56_metrics.record("evolution_duration_ms", round(_s56_ms, 2))
            except Exception:
                pass  # Metrics are non-critical

        except ImportError as _s56_imp_err:
            # RETRIEVAL-018: ImportError was silently swallowed — now visible
            logger.info(f"S5.6: SKIPPED — ImportError: {_s56_imp_err}")
        except Exception as s56_err:
            logger.info(f"S5.6: FAILED (non-fatal): {s56_err}")

        t_evolution = time.perf_counter()  # PERF-016: Phase A checkpoint

        # --- BENCH-014: Co-activation association (budget: <10ms) ---
        # When concepts co-activate during retrieval, that's evidence of semantic
        # relatedness that TF-IDF can't capture (e.g., "allergy" + "doctor" for
        # a health query). Creates "co_activated" edges between top co-activated
        # pairs that don't already share an association.
        try:
            _coact_ids = [
                ac.concept_id for ac in activated
                if ac.relevance_score and ac.relevance_score > 0.30
                and ac.concept_id not in always_on_ids
            ]
            if len(_coact_ids) >= 2:
                from app.storage import add_association, get_all_association_triples
                _existing_triples = get_all_association_triples()
                _coact_created = 0
                # Link top pairs (cap at 3 new edges per turn to control budget)
                for i in range(min(len(_coact_ids), 4)):
                    for j in range(i + 1, min(len(_coact_ids), 4)):
                        src, tgt = sorted([_coact_ids[i], _coact_ids[j]])
                        triple = (src, tgt, "co_activated")
                        if triple not in _existing_triples:
                            add_association(src, tgt, "co_activated", 0.30)
                            _existing_triples.add(triple)
                            _coact_created += 1
                            if _coact_created >= 3:
                                break
                    if _coact_created >= 3:
                        break
                if _coact_created > 0:
                    logger.info(f"BENCH-014: Co-activation — {_coact_created} edges created from {len(_coact_ids)} concepts")
                else:
                    logger.info(f"BENCH-014: Co-activation — 0 new edges ({len(_coact_ids)} concepts, all pairs exist)")
        except Exception as e:
            logger.debug(f"BENCH-014: Co-activation failed (non-fatal): {e}")

        # --- FIX 1: Coverage confidence + blind spot cross-reference (budget: <5ms) ---
        coverage_confidence = None
        blind_spot_match = None
        try:
            # Convert activated concepts to dicts for coverage computation
            activated_dicts = [
                {
                    "concept_id": ac.concept_id,
                    "summary": ac.summary,  # COVERAGE-001: needed for abstraction detection
                    "relevance_score": ac.relevance_score,
                    "knowledge_area": ac.knowledge_area,
                }
                for ac in activated
            ]
            coverage_confidence = self._compute_coverage_confidence(activated_dicts, request.message)
            blind_spot_match = self._check_blind_spot_relevance(request.message, coverage_confidence)
            if coverage_confidence:
                logger.info(
                    f"FIX1: Coverage signal: {coverage_confidence.get('level')} "
                    f"(top_score={coverage_confidence.get('top_score', 'N/A')})"
                )
            if blind_spot_match:
                logger.info(f"FIX1b: Blind spot match: {blind_spot_match.get('blind_spot_match', '')[:60]}")
        except Exception as e:
            logger.warning(f"FIX1: Coverage confidence failed (non-fatal): {e}")

        # --- QUALITY-002: Numeric coverage_score (budget: <1ms) ---
        # Mean relevance score of semantic matches. Measures retrieval QUALITY
        # not just quantity — fixes adjacent-unknown blind spot where count-based
        # ratio falsely reported high confidence for queries semantically near
        # known domains but factually unknown.
        # Thresholds (from live validation): ≥0.45 → high confidence,
        # 0.30-0.45 → uncertain, <0.30 → no relevant knowledge.
        # Excludes always-activate and firmware injections.
        coverage_score = None
        try:
            if activated:
                semantic_scores = [
                    ac.relevance_score
                    for ac in activated
                    if ac.concept_id not in always_on_ids
                    and ac.relevance_score is not None
                    and ac.relevance_score > 0
                ]
                if semantic_scores:
                    coverage_score = round(
                        sum(semantic_scores) / len(semantic_scores), 4
                    )
                else:
                    coverage_score = 0.0
                logger.debug(
                    f"QUALITY-002: coverage_score={coverage_score} "
                    f"(semantic_matches={len(semantic_scores)}, "
                    f"mean_relevance={coverage_score})"
                )
                # MEASURE-020: Persist coverage metrics for BENCH-015 calibration
                try:
                    try:
                        from app.config import COVERAGE_RELEVANCE_THRESHOLD as _cov_thresh
                    except (ImportError, AttributeError):
                        _cov_thresh = 0.35
                    from app.storage import _db as _cov_db
                    from app.datetime_utils import _utc_now_iso as _cov_utc_now
                    with _cov_db() as _cov_conn:
                        _cov_conn.execute(
                            "INSERT INTO governance_events (session_id, event_type, details, created_at) "
                            "VALUES (?, ?, ?, ?)",
                            (
                                self.current_session.session_id if self.current_session else None,
                                "coverage_score_recorded",
                                json.dumps({
                                    "coverage_score": coverage_score,
                                    "semantic_match_count": len(semantic_scores),
                                    "threshold": _cov_thresh,
                                    "above_threshold": len([s for s in semantic_scores if s > _cov_thresh]),
                                }),
                                _cov_utc_now(),
                            ),
                        )
                except Exception:
                    pass  # Non-fatal — measurement should never break the pipeline
        except Exception as e:
            logger.debug(f"QUALITY-002: coverage_score failed (non-fatal): {e}")

        # --- PRODUCT-003: Abstention signal (budget: <1ms) ---
        abstention_signal = None
        try:
            abstention_signal = self._compute_abstention_signal(coverage_confidence, coverage_score)
            if abstention_signal:
                logger.info(
                    f"PRODUCT-003: Abstention recommended: level={abstention_signal['level']}, "
                    f"confidence={abstention_signal['confidence']}"
                )
                # A1: Observability metric for dashboarding
                try:
                    from app.metrics import metrics as _abs_metrics
                    _abs_metrics.record("abstention_fired", 1.0)
                    _abs_metrics.record("abstention_confidence", abstention_signal["confidence"])
                except Exception:
                    pass  # Metrics are non-critical
        except Exception as e:
            logger.debug(f"PRODUCT-003: abstention_signal failed (non-fatal): {e}")

        # --- FIX 3: Post-retrieval extraction request gaps (budget: <5ms) ---
        # Gap 7: Coverage-triggered extraction (depends on Fix 1 coverage_confidence)
        # Gap 8: Topic freshness extraction
        # Appends to existing extraction_request from B1 (pre-retrieval gaps)
        try:
            post_retrieval_items = []

            # Gap 7: Sparse coverage → prompt for knowledge building
            if coverage_confidence and coverage_confidence.get("level") in (
                "no_strong_match",
                "sparse_coverage",
                "no_results",
                "incomplete",   # COVERAGE-001: LLM-detected coverage gap
                "uncertain",    # COVERAGE-001: LLM-detected abstraction mismatch
            ):
                post_retrieval_items.append(
                    {
                        "type": "any",
                        "prompt": (
                            "The pith has sparse knowledge in the area you're discussing. "
                            "If you share insights, decisions, or context about this topic, "
                            "include them in extracted_concepts_json to build up this knowledge area."
                        ),
                        "priority": "medium",
                    }
                )

            # Gap 8: Topic freshness — top results all older than 30 days
            if activated and not coverage_confidence:  # Only check when coverage is adequate
                try:
                    stale_threshold = now - timedelta(days=30)
                    top_areas = set()
                    all_top_stale = True
                    for ac in activated[:5]:
                        concept = _concept_cache.get(ac.concept_id)
                        if concept and concept.created_at:
                            created_str = (
                                concept.created_at
                                if isinstance(concept.created_at, str)
                                else concept.created_at.isoformat()
                            )
                            created_dt = _ensure_aware(
                                datetime.fromisoformat(created_str.replace("Z", "+00:00").replace("+00:00", ""))
                            )
                            if created_dt > stale_threshold:
                                all_top_stale = False
                                break
                            area = (concept.metadata or {}).get("knowledge_area", "unknown")
                            top_areas.add(area)
                    if all_top_stale and top_areas:
                        areas_str = ", ".join(sorted(top_areas))
                        post_retrieval_items.append(
                            {
                                "type": "observation",
                                "prompt": (
                                    f"The pith's knowledge in {areas_str} appears outdated "
                                    f"(all top results >30 days old). If the current state has "
                                    f"changed, extract updated observations."
                                ),
                                "priority": "low",
                            }
                        )
                except Exception:
                    pass  # Freshness check is best-effort

            # Merge into extraction_request
            if post_retrieval_items:
                if extraction_request is None:
                    extraction_request = {"items": post_retrieval_items}
                elif isinstance(extraction_request, dict):
                    existing = extraction_request.get("items", [])
                    extraction_request["items"] = existing + post_retrieval_items
                logger.info(f"FIX3: Added {len(post_retrieval_items)} post-retrieval extraction items")
        except Exception as e:
            logger.warning(f"FIX3: Post-retrieval extraction failed (non-fatal): {e}")

        # --- GOV-W2: Contradiction detection (budget: 10ms) ---
        # Runs after staleness filtering, before context assembly finalizes.
        # Detects pairwise contradictions among retrieval survivors using
        # 3-phase algorithm: keyword negation, embedding similarity, soft detection.
        contradiction_result = None
        try:
            # TB-5: Circuit breaker skips optional contradiction detection
            if circuit_breaker_active:
                logger.info("CIRCUIT_BREAKER_SKIP: contradiction_detection skipped")
                if gov_ctx:
                    gov_ctx.phases_skipped.append("contradiction_detection")
                raise _BudgetSkip()
            if gov_ctx:
                # Fix 5a (v1.2): Corrected from 600ms to 10ms. The 600ms conflated
                # Phases 1-3 (~10ms) with Tier 2 LLM (~500ms). Phase 2 has its own
                # 5ms internal budget gate. Tier 2 is gated by feature flag.
                _budget_ok = gov_ctx.check_latency_budget("contradiction_detection", 10.0, PhasePriority.OPTIONAL)
                if not _budget_ok:
                    from app.governance_context import GOVERNANCE_HARD_ENFORCEMENT

                    if GOVERNANCE_HARD_ENFORCEMENT:
                        logger.info("HARD_SKIP: contradiction_detection skipped (budget exhausted)")
                        raise _BudgetSkip()
                    else:
                        logger.info("SOFT_SKIP: contradiction_detection would be skipped (observability mode)")
            from app.contradiction import ScoredConcept, detect_retrieval_contradictions

            # BENCHMARK-003: Skip contradiction detection in benchmark mode.
            # Contradictions require LLM calls per pair — expensive and wasted
            # when each Pith instance lives for one question and is destroyed.
            # UNLESS explicitly allowed (needed when auto-association is enabled
            # to properly mark superseded concepts — see Q1 RCA 2026-03-19).
            from app.config import BENCHMARK as _bm_contra
            if _bm_contra.skip_retrieval_contradictions:
                logger.debug("BENCHMARK-003: Skipping contradiction detection")
                raise _BudgetSkip()

            # Build ScoredConcept list from activated concepts (uses concept cache)
            scored_survivors = []
            for ac in activated:
                concept = _concept_cache.get(ac.concept_id)
                emb = None
                if concept and hasattr(concept, "metadata"):
                    # Try to get cached embedding from the search index
                    try:
                        from app.embedding import embedding_engine

                        pos = embedding_engine._id_to_pos.get(ac.concept_id)
                        if pos is not None and embedding_engine._index_matrix is not None:
                            emb = embedding_engine._index_matrix[pos]
                    except Exception:
                        pass

                scored_survivors.append(
                    ScoredConcept(
                        concept_id=ac.concept_id,
                        summary=ac.summary,
                        knowledge_area=ac.knowledge_area or "unknown",
                        authority_score=concept.authority_score if concept and concept.authority_score else 0.0,
                        currency_score=concept.currency_score if concept and concept.currency_score else 0.5,
                        embedding=emb,
                        created_at=concept.created_at if concept else None,  # LIFECYCLE-001
                        concept_type=concept.concept_type if concept else None,  # LIFECYCLE-001
                    )
                )

            if len(scored_survivors) >= 2:
                contradiction_result = detect_retrieval_contradictions(scored_survivors, gov_ctx)

                # TB-2: Persist contradiction resolutions to DB BEFORE cascade floor
                # (cascade floor clears suppressed_ids for retrieval, but DB must record the resolution)
                if contradiction_result.suppressed_ids or contradiction_result.contested_ids:
                    try:
                        from app.datetime_utils import _utc_now_iso
                        from app.storage import db_immediate

                        with db_immediate() as _contra_conn:
                            now = _utc_now_iso()
                            for loser_id in contradiction_result.suppressed_ids:
                                _contra_conn.execute(
                                    """UPDATE concepts
                                       SET currency_status = 'CONTRADICTED',
                                           data = json_set(data, '$.currency_status', 'CONTRADICTED'),
                                           updated_at = ?
                                       WHERE id = ? AND currency_status != 'CONTRADICTED'""",
                                    (now, loser_id),
                                )
                            for contested_id in contradiction_result.contested_ids:
                                _contra_conn.execute(
                                    """UPDATE concepts
                                       SET currency_status = 'CONTESTED',
                                           data = json_set(data, '$.currency_status', 'CONTESTED'),
                                           updated_at = ?
                                       WHERE id = ? AND currency_status NOT IN ('CONTRADICTED', 'CONTESTED')""",
                                    (now, contested_id),
                                )
                            # commit handled by _db() context manager
                            logger.info(
                                "TB-2: Persisted %d suppressed, %d contested",
                                len(contradiction_result.suppressed_ids),
                                len(contradiction_result.contested_ids),
                            )
                    except Exception as e:
                        logger.warning("TB-2: Contradiction persistence failed: %s", e)
                        # GA-N01: Track persistence failures in metrics
                        try:
                            from app.metrics import metrics as _m_contra

                            _m_contra.record("contradiction_persistence_failures", 1)
                        except Exception:
                            pass

                if contradiction_result.suppressed_ids:
                    # W5: Cascade floor — don't suppress if it would drop below 3 activated concepts
                    CONTRADICTION_CASCADE_FLOOR = 3
                    post_suppression_count = len(
                        [ac for ac in activated if ac.concept_id not in set(contradiction_result.suppressed_ids)]
                    )
                    if post_suppression_count >= CONTRADICTION_CASCADE_FLOOR:
                        activated = [
                            ac for ac in activated if ac.concept_id not in set(contradiction_result.suppressed_ids)
                        ]
                        logger.info(
                            f"GOV-W2: Suppressed {len(contradiction_result.suppressed_ids)} contradicted concepts"
                        )
                    else:
                        logger.warning(
                            "GOV-W2: Cascade floor hit — suppression would reduce activated from %d to %d (floor=%d), skipping",
                            len(activated),
                            post_suppression_count,
                            CONTRADICTION_CASCADE_FLOOR,
                        )
                        contradiction_result.suppressed_ids = []  # Clear so downstream doesn't count them
        except Exception as e:
            logger.warning(f"GOV-W2: Contradiction detection failed (non-fatal): {e}")

        t_contradiction = time.perf_counter()  # PERF-016: Phase A checkpoint

        # --- GOV-W2.25: Budget governance (budget: 2ms) ---
        # 4-tier budget allocation: GUARANTEED > PRIORITY > FILL > OVERFLOW.
        # Caps activated concepts at CONTEXT_BUDGET_MAIN (default 20) slots.
        # Tier 4 overflow concepts get compressed one-liner summaries.
        budget_allocation_response = None
        try:
            # TB-5: Circuit breaker skips optional budget governance
            if circuit_breaker_active:
                logger.info("CIRCUIT_BREAKER_SKIP: budget_governance skipped")
                if gov_ctx:
                    gov_ctx.phases_skipped.append("budget_governance")
                raise _BudgetSkip()
            if gov_ctx:
                _budget_ok = gov_ctx.check_latency_budget("budget_governance", 2.0, PhasePriority.OPTIONAL)
                if not _budget_ok:
                    from app.governance_context import GOVERNANCE_HARD_ENFORCEMENT

                    if GOVERNANCE_HARD_ENFORCEMENT:
                        logger.info("HARD_SKIP: budget_governance skipped (budget exhausted)")
                        raise _BudgetSkip()
                    else:
                        logger.info("SOFT_SKIP: budget_governance would be skipped (observability mode)")
            from app.budget import allocate_budget
            from app.governance_context import ScoredConcept as BudgetScoredConcept

            # Build ScoredConcept list from activated concepts (uses concept cache)
            budget_candidates = []
            query_areas = list(result_areas) if result_areas else []
            for ac in activated:
                concept = _concept_cache.get(ac.concept_id)
                auth = 0.0
                ka = ac.knowledge_area or "unknown"
                if concept:
                    auth = concept.authority_score if concept.authority_score is not None else concept.confidence
                # Config fix: pass concept_type for GUARANTEED tier gate
                ct = "unknown"
                if concept and hasattr(concept, "concept_type"):
                    ct = concept.concept_type or "unknown"
                budget_candidates.append(
                    BudgetScoredConcept(
                        concept_id=ac.concept_id,
                        authority_score=auth or 0.0,
                        final_score=ac.relevance_score or 0.0,
                        confidence=ac.confidence or 0.0,
                        knowledge_area=ka,
                        concept_type=ct,
                    )
                )

            # Always-activate + firmware IDs bypass budget trimming
            aa_ids = [ao["concept_id"] for ao in always_on_injected] + [
                f"firmware:{fw['id']}" for fw in firmware_entries
            ]

            # MEASURE-027: Adaptive context budget based on brain size.
            # CONTEXT_BUDGET_MAIN=20 collapses accuracy from 100% to 26% when
            # concept count exceeds budget. Scale budget with brain size.
            # Formula: min(max(20, concept_count * 0.5), 50)
            # Evidence: H4/H5 gauntlet — budget=50 achieves 100% at 100 concepts.
            from app.config import CONTEXT_BUDGET_MAIN as _static_budget
            _adaptive_budget = _static_budget  # default from config/env
            try:
                from app.embedding import embedding_engine as _budget_emb
                _brain_size = _budget_emb.index_size
                if _brain_size > 0:
                    if _static_budget <= 50:
                        # Default path: scale up conservatively, cap at 50
                        _adaptive_budget = min(max(20, int(_brain_size * 0.5)), 50)
                    else:
                        # User override: respect explicit high budget (diagnostic / large-brain mode)
                        _adaptive_budget = _static_budget
                    logger.warning(f"MEASURE-027: Adaptive budget {_static_budget}→{_adaptive_budget} "
                                   f"(brain_size={_brain_size})")
            except Exception:
                pass  # Fall back to static config

            alloc = allocate_budget(
                budget_candidates,
                gov_ctx,
                always_activate_ids=aa_ids,
                query_knowledge_areas=query_areas,
                total_slots=_adaptive_budget,
            )

            # Filter activated to tiers 1-3 only
            allowed_ids = set()
            for tier_name in ["guaranteed", "priority", "fill"]:
                allowed_ids.update(alloc.tiers.get(tier_name, []))

            # Always keep firmware and always-activate regardless of budget
            allowed_ids.update(set(aa_ids))

            pre_budget_count = len(activated)
            activated = [ac for ac in activated if ac.concept_id in allowed_ids]
            budget_trimmed = pre_budget_count - len(activated)

            budget_allocation_response = alloc.to_dict()

            if budget_trimmed > 0:
                logger.info(
                    f"GOV-W2.25: Budget trimmed {budget_trimmed} concepts (overflow: {len(alloc.overflow_summaries)})"
                )
            else:
                logger.info(
                    f"GOV-W2.25: All {len(activated)} concepts within budget "
                    f"(T1:{len(alloc.tiers.get('guaranteed', []))} "
                    f"T2:{len(alloc.tiers.get('priority', []))} "
                    f"T3:{len(alloc.tiers.get('fill', []))})"
                )
        except _BudgetSkip:
            pass
        except Exception as e:
            logger.warning(f"GOV-W2.25: Budget governance failed (non-fatal): {e}")

        # --- GOV-W2.5: Constraint assembly (budget: 3ms) ---
        # Extracts high-authority concepts as behavioral constraints with anti-terms.
        # Returned to client for pre-generation awareness + post-generation validation.
        constraint_set_response = None
        try:
            # P4-PREREQ: Feature-flagged constraint assembly
            from app.config import FEATURE_FLAGS as _ca_ff

            if not _ca_ff.get("CONSTRAINT_ASSEMBLY_ENABLED", True):
                raise RuntimeError("CONSTRAINT_ASSEMBLY_ENABLED=False, skipping")
            if gov_ctx:
                # P4-PREREQ: Promoted to REQUIRED — constraint_set is critical for P4a validation.
                # Budget is 3ms (assembly is fast); was OPTIONAL and always skipped due to budget exhaustion.
                gov_ctx.check_latency_budget("constraint_assembly", 3.0, PhasePriority.REQUIRED)
            from app.prediction_error import assemble_constraint_set, constraint_set_to_dict

            constraint_candidates = []
            for ac in activated:
                concept = _concept_cache.get(ac.concept_id)
                if concept:
                    # Use authority_score if computed, else fall back to confidence
                    auth = concept.authority_score if concept.authority_score is not None else concept.confidence
                    # P4-PREREQ: pass concept_type for type gating in constraint assembly
                    ct = getattr(concept, "concept_type", None) or "observation"
                    # P4-PREREQ: pass epistemic_network for epistemic cap application
                    # BUG FIX: Do NOT default to "assessment" — apply_epistemic_cap has a
                    # safety valve that skips capping for legacy concepts (network=None).
                    # Defaulting to "assessment" forces all legacy concepts into 0.40 cap,
                    # which drops them below the 0.55 threshold → 0 constraints.
                    en = getattr(concept, "epistemic_network", None)  # None = legacy, uncapped
                    vs = getattr(concept, "verification_status", None)
                    constraint_candidates.append(
                        {
                            "concept_id": ac.concept_id,
                            "summary": ac.summary,
                            "authority_score": auth or 0.0,
                            "concept_type": ct,
                            "epistemic_network": en,
                            "verification_status": vs,
                        }
                    )

            cs = assemble_constraint_set(constraint_candidates, conn=_get_connection())
            if cs.constraint_count > 0:
                constraint_set_response = constraint_set_to_dict(cs)
                logger.info(
                    f"GOV-W2.5: {cs.constraint_count} constraints, "
                    f"{cs.total_anti_terms} anti-terms in {cs.assembly_time_ms:.1f}ms"
                )
        except Exception as e:
            logger.warning(f"GOV-W2.5: Constraint assembly failed (non-fatal): {e}")

        # --- CCL §3c.1: Persist constraint_set for next-turn validation ---
        # Store in session state so CCL step 0 can validate on next turn.
        # In-memory only — on session restart, _previous_constraint_set is None
        # and validation gracefully skips.
        if constraint_set_response:
            try:
                from app.prediction_error import _extract_terms

                self._previous_constraint_set = {
                    "constraints": constraint_set_response.get("constraints", []),
                    "timestamp": _utc_now_iso(),
                    "topic_terms": _extract_terms(request.message or "")[:20],
                }
            except Exception as e:
                logger.warning(f"CCL §3c.1: Constraint persistence failed (non-fatal): {e}")

        # Compute graph density: associations / concepts ratio
        total_concepts = retrieval_engine.index.document_count or 1
        total_assocs = len(edges) if top_results else count_associations()
        graph_density = round(total_assocs / max(total_concepts, 1), 3)

        # --- S0: First-call detection (for is_first_call flag only) ---
        # Boundary detection still sets is_first_call for protocol metadata,
        # but orientation is NO LONGER gated on it.
        CONVERSATION_BOUNDARY_SECONDS = 120  # 2 minutes
        now_mono = time.perf_counter()
        if (
            self._conversation_turn_called
            and self._last_conversation_turn_at is not None
            and (now_mono - self._last_conversation_turn_at) > CONVERSATION_BOUNDARY_SECONDS
        ):
            logger.info(
                f"S0: Conversation boundary detected ({now_mono - self._last_conversation_turn_at:.0f}s gap). Resetting first-call flag."
            )
            self._conversation_turn_called = False

        is_first_call = not self._conversation_turn_called
        self._conversation_turn_called = True
        self._last_conversation_turn_at = now_mono

        # --- B5.1: Resumption detection ---
        is_resumption = self._detect_resumption()

        # --- RC-B: Resume Context injection (v1.1) ---
        resume_context = None
        resume_context_tier = None
        resume_context_suppressed = False
        if is_first_call and is_resumption:
            resume_context, resume_context_tier, resume_context_suppressed = self._inject_resume_context(request)

            # --- RC Phase 2: Observability metrics ---
            try:
                from app.metrics import metrics as _rc_metrics

                _rc_metrics.record(
                    "resume_context_injection",
                    1,
                    {
                        "tier": resume_context_tier or "NONE",
                        "suppressed": str(resume_context_suppressed),
                        "has_context": str(resume_context is not None),
                    },
                )
                if resume_context_tier and resume_context_tier != "EXPIRED":
                    _rc_metrics.record("resume_context_tier", 1, {"tier": resume_context_tier})
                if resume_context_suppressed:
                    _rc_metrics.record("resume_context_drift_suppressed", 1)
            except Exception:
                pass  # Metrics are best-effort

            # --- RC Phase 2: Governance event logging ---
            if gov_ctx:
                try:  # noqa: SIM105
                    gov_ctx.log_event(
                        GOV_EVENT_RESUME_CONTEXT_INJECTION,
                        None,
                        {
                            "tier": resume_context_tier,
                            "suppressed": resume_context_suppressed,
                            "context_length": len(resume_context) if resume_context else 0,
                        },
                    )
                except Exception:
                    pass

        # --- CTX S-0.5b: Compaction re-injection (when detected) ---
        # If compaction was detected earlier, re-inject critical context
        # from the rolling snapshot. This OVERRIDES the normal resume_context
        # and orientation to provide compaction-specific recovery.
        if compaction_was_detected and not is_first_call:
            try:
                comp_resume, comp_orient, comp_hint, comp_quality = self._handle_compaction_reinjection(request)
                if comp_resume:
                    resume_context = comp_resume
                    resume_context_tier = "COMPACTION_RECOVERY"
                if gov_ctx:
                    gov_ctx.log_event(
                        GOV_EVENT_COMPACTION_REINJECTION,
                        None,
                        {
                            "has_resume": comp_resume is not None,
                            "turn_count": turn_count,
                            "recovery_quality": comp_quality,
                        },
                    )
            except Exception as comp_inj_err:
                logger.warning(f"CTX S-0.5b: Re-injection failed (non-fatal): {comp_inj_err}")

        # --- S6: ALWAYS-SERVE orientation (budget: ~10ms) ---
        # Previous design gated orientation on is_first_call + boundary
        # heuristics (L1-L4). Every heuristic had edge cases that caused
        # orientation to return null on new conversations. The fundamental
        # fix: orientation is cheap (~10ms), so always build and serve it.
        # The client decides whether to use it. No heuristic can be wrong
        # if there is no heuristic.
        orientation_summary, greeting_hint = self._build_temporal_context(request, is_resumption=is_resumption)

        # CKPT-005: Resume signal in greeting_hint (not orientation — per conv_2269a0763534)
        checkpoint_resume_available = False
        if is_first_call and is_resumption:
            try:
                from app.storage import load_checkpoint
                _resume_cp = load_checkpoint(max_age_hours=24)
                if _resume_cp and _resume_cp["status"] in ("paused", "active", "planning"):
                    checkpoint_resume_available = True
                    if greeting_hint:
                        greeting_hint += " A recent checkpoint is available in working_context — consider offering to resume."
            except Exception as e:
                logger.debug(f"CKPT-005: Checkpoint resume signal failed: {e}")

        # CTX: If compaction was detected, override orientation with recovery-specific hint
        if compaction_was_detected and not is_first_call:
            # CONTEXT-001: Greeting hint references working_context field
            greeting_hint = (
                "COMPACTION_RECOVERY. Your context was likely summarized. "
                "Critical operational context has been re-injected. "
                "Reference working_context for structured work state."
            )

        # --- RC §5.5: First-call budget enforcement ---
        # Enforce 1400-token ceiling across all injection sources on first call.
        # Uses aa_ids from S4.6/S4.7 to identify firmware/always-activate concepts.
        if is_first_call:
            try:
                try:
                    _aa_id_set = set(aa_ids)
                except NameError:
                    _aa_id_set = set()
                always_activate = [c for c in activated if c.concept_id in _aa_id_set]
                regular_activated = [c for c in activated if c.concept_id not in _aa_id_set]

                always_activate, resume_context, orientation_summary, regular_activated = (
                    self._enforce_first_call_budget(
                        always_activate_concepts=always_activate,
                        resume_context=resume_context,
                        orientation_summary=orientation_summary,
                        activated_concepts=regular_activated,
                    )
                )
                # Recombine: always-activate first, then regular
                activated = always_activate + regular_activated
            except Exception as budget_err:
                logger.warning(f"RC §5.5: Budget enforcement failed (non-fatal): {budget_err}")

        # --- PERF-024: Non-first-call response budget governor ---
        # First-call has its own budget (RC §5.5 above). For subsequent turns,
        # cap activated_concepts to prevent response bloat. Other fields
        # (constraint_set, working_context, governance_summary) are bounded
        # by design; activated_concepts is the only unbounded variable.
        if not is_first_call:
            try:
                budget = self.TURN_TOKEN_BUDGET
                # Estimate tokens for non-concept fields (constraint_set, working_context, etc.)
                # These are bounded by design; use conservative fixed estimate.
                _overhead_tokens = 400  # constraint_set + working_context + governance + metadata
                concept_budget = budget - _overhead_tokens

                total_concept_tokens = 0
                trimmed_activated = []
                for c in activated:
                    c_tokens = len((getattr(c, "summary", "") or "").split()) + 5
                    if total_concept_tokens + c_tokens > concept_budget:
                        break
                    trimmed_activated.append(c)
                    total_concept_tokens += c_tokens

                if len(trimmed_activated) < len(activated):
                    logger.info(
                        "PERF-024: Response budget governor trimmed activated_concepts "
                        f"from {len(activated)} to {len(trimmed_activated)} "
                        f"(budget={concept_budget} tokens)"
                    )
                    if gov_ctx:
                        gov_ctx.log_event(
                            "RESPONSE_BUDGET_TRIMMED",
                            None,
                            {
                                "original_count": len(activated),
                                "trimmed_count": len(trimmed_activated),
                                "concept_budget": concept_budget,
                                "estimated_tokens": total_concept_tokens,
                            },
                        )
                    activated = trimmed_activated
            except Exception as perf024_err:
                logger.warning(f"PERF-024: Budget governor failed (non-fatal): {perf024_err}")

        # --- RETRIEVAL-037b v4: Conflict pre-filter (subject dedup + chain prune) ---
        # Phase 1: Same-subject dedup (keeps highest serial_order per subject key).
        # Phase 2: Chain-aware orphan pruning (removes downstream fragments of destroyed chains).
        # Applied AFTER budget governor so relevance-ranked concepts are selected first,
        # then duplicates are removed from the budgeted set. This matches the prototype's
        # pipeline ordering (client-side prefilter ran after server returned concepts).
        _conflict_prefilter_enabled = os.environ.get("PITH_CONFLICT_PREFILTER", "").lower() in ("true", "1")
        if _conflict_prefilter_enabled:
            try:
                _pre_conflict = len(activated)
                activated_filtered = _conflict_prefilter(activated)
                _post_conflict = len(activated_filtered)
                _destroyed = [c for c in activated if c not in activated_filtered]
                if _pre_conflict != _post_conflict:
                    logger.info(
                        f"RETRIEVAL-037b: Conflict pre-filter reduced {_pre_conflict} → {_post_conflict} "
                        f"concepts ({_pre_conflict - _post_conflict} same-subject duplicates removed)"
                    )

                # Phase 2: Chain-aware orphan pruning (gated separately)
                _chain_prune_env = os.environ.get("PITH_CHAIN_PREFILTER", "").lower() in ("true", "1")
                # RETRIEVAL-CHAIN-GATE-001: Only chain-prune on multihop queries.
                # SH queries need wide recall — chain prune amputates context they need.
                # _multihop_used is set at ~line 2690-2710 by RETRIEVAL-060 router.
                _chain_prune_enabled = _chain_prune_env and _multihop_used
                if _chain_prune_enabled and _destroyed:
                    activated_filtered = _chain_aware_prune(activated_filtered, _destroyed)
                elif _chain_prune_env and not _multihop_used and _destroyed:
                    logger.debug(
                        "RETRIEVAL-CHAIN-GATE-001: Chain prune skipped (non-multihop query)"
                    )

                activated = activated_filtered
            except Exception as e:
                logger.warning(f"RETRIEVAL-037b: Conflict pre-filter failed (non-fatal): {e}")

        t_end = time.perf_counter()
        elapsed_ms = round((t_end - t0) * 1000, 2)

        # FED-013: Update session registry heartbeat + working context
        try:
            from app.federation import get_registry

            _fed_registry = get_registry()
            _wc = {
                "activated_domains": activated_domain_ids or [],
                "top_knowledge_areas": list(
                    {
                        getattr(r, "knowledge_area", None)
                        for r in (search_results or [])[:10]
                        if getattr(r, "knowledge_area", None)
                    }
                )[:5],
                "message_keywords": (request.message or "")[:200].split()[:20],
                "recent_concept_ids": [c.concept_id for c in (activated or [])[:5]],
            }
            _fed_registry.update_heartbeat(
                session_id=self.current_session.session_id if self.current_session else None,
                working_context=_wc,
            )
        except Exception as e:
            logger.debug(f"FED-013: Heartbeat hook failed (non-fatal): {e}")

        # WS2: Metric 1 — conversation_turn_latency_ms
        try:
            from app.metrics import metrics

            metrics.record("conversation_turn_latency_ms", elapsed_ms)
        except Exception:
            pass  # Metrics are best-effort

        # PERF-016: Per-phase timing metrics
        try:
            from app.metrics import metrics as _phase_metrics

            _phases = {
                "ct_phase_autolearn_ms": (t_autolearn - t0),
                "ct_phase_health_ms": (t_health - t_autolearn),
                "ct_phase_correction_ms": (t_correction - t_health),
                "ct_phase_search_lightweight_ms": (_t_search_lw_end - _t_search_lw_start),  # PERF-017
                "ct_phase_retrieval_ms": (t_retrieval - t_correction),
                "ct_phase_graph_ms": (t_graph - t_retrieval),
                "ct_phase_injection_ms": (t_injection - t_graph),
                "ct_phase_evolution_ms": (t_evolution - t_injection),
                "ct_phase_contradiction_ms": (t_contradiction - t_evolution),
                "ct_phase_assembly_ms": (t_end - t_contradiction),
            }
            for metric_name, duration_s in _phases.items():
                _phase_metrics.record(metric_name, round(duration_s * 1000, 2))
        except Exception:
            pass  # Metrics are best-effort

        # Attack 6: Phase timing for performance monitoring
        logger.info(
            f"conversation_turn timing: "
            f"autolearn={round((t_autolearn - t0) * 1000)}ms "
            f"health={round((t_health - t_autolearn) * 1000)}ms "
            f"correction={round((t_correction - t_health) * 1000)}ms "
            f"retrieval={round((t_retrieval - t_correction) * 1000)}ms "
            f"graph={round((t_graph - t_retrieval) * 1000)}ms "
            f"injection={round((t_injection - t_graph) * 1000)}ms "
            f"evolution={round((t_evolution - t_injection) * 1000)}ms "
            f"contradiction={round((t_contradiction - t_evolution) * 1000)}ms "
            f"assembly={round((t_end - t_contradiction) * 1000)}ms "
            f"total={round(elapsed_ms)}ms"
        )

        # MEASURE-001: Docker context pollution rate — log % of activated concepts
        # that are docker/container-related before full relevance decay takes effect
        try:
            if activated:
                _docker_kws = ("docker", "container", "devops", "kubernetes", "k8s")
                _docker_count = sum(
                    1 for c in activated
                    if any(kw in (getattr(c, "knowledge_area", "") or "").lower() or
                           kw in (getattr(c, "summary", "") or "").lower()[:80]
                           for kw in _docker_kws)
                )
                _docker_pct = _docker_count / len(activated)
                from app.metrics import metrics as _m001
                _m001.record("docker_context_pollution_rate", _docker_pct)
                if _docker_pct > 0.10:
                    logger.info(
                        f"MEASURE-001: Docker pollution {_docker_pct:.0%} "
                        f"({_docker_count}/{len(activated)} activated concepts)"
                    )
        except Exception:
            pass  # Best-effort

        # Build auto_learned summary if learning happened
        # PERF-FORT-2/A1: When background mode active, auto_learn_result is None.
        # Use previous turn's cached result instead. Pricing/metering moved to background method.
        auto_learned = None
        budget_warnings = []
        upgrade_nudge = None
        recall_gap_attribution = None  # PRICING-007: Set before response construction
        if auto_learn_result:
            # Synchronous path (feature flag OFF) — original behavior preserved
            budget_warnings = auto_learn_result.budget_warnings or []
            if auto_learn_result.learning_events > 0:
                auto_learned = {
                    "events": auto_learn_result.learning_events,
                    "concepts_created": [c.concept_id for c in auto_learn_result.concepts_created],
                    "concepts_evolved": [c.concept_id for c in auto_learn_result.concepts_evolved],
                    "budget_warnings": budget_warnings,
                }
                try:
                    from app.pricing import conversation_meter
                    remaining = conversation_meter.consume_turn()
                    if remaining == 0:
                        budget_warnings.append(
                            f"turn_budget_exhausted: 0 remaining of {conversation_meter._daily_limit}/day"
                        )
                    elif remaining > 0 and remaining <= conversation_meter._daily_limit * 0.1:
                        budget_warnings.append(
                            f"turn_budget_low: {remaining} remaining of {conversation_meter._daily_limit}/day"
                        )
                    upgrade_nudge = conversation_meter.get_upgrade_nudge()
                    if upgrade_nudge:
                        logger.info("MONITOR-026: upgrade_nudge activated: %s", upgrade_nudge.get("reason", "unknown"))
                except Exception as pricing_err:
                    logger.warning(f"PRICING-002: Turn metering failed (non-fatal): {pricing_err}")
        else:
            # Background path — use snapshot (race-safe, captured before dispatch)
            try:
                auto_learned = _bg_snapshot_auto_learned
                budget_warnings = _bg_snapshot_budget_warnings or []
            except NameError:
                # No background dispatch happened (prev_response too short)
                auto_learned = getattr(self, '_last_autolearn_result', None)
                budget_warnings = getattr(self, '_last_autolearn_budget_warnings', []) or []

        # --- RETRO-001: Automated retrospective nudge ---
        # Check if Pith has accumulated many observations without
        # extracting higher-order abstractions (principles, methods, etc.)
        # This detects the meta-learning gap where Pith remembers
        # WHAT happened but never learns HOW to think better.
        retrospective_nudge = self._check_retrospective_needed()

        # --- T1: Retroactive reflection on orphaned sessions (conversation_turn path) ---
        # Covers the case where session_start wasn't called but conversation_turn is
        # (e.g., auto-created sessions from context compaction)
        retroactive_reflection = None
        if is_resumption and is_first_call:
            try:
                from app.auto_reflection import (
                    check_orphaned_sessions_for_reflection,
                    mark_session_reflected,
                    record_reflection_event,
                )
                from app.storage import _db

                with _db() as conn:
                    orphan_rows = conn.execute(
                        """SELECT id, started_at, ended_at, status,
                                  learning_event_count, data
                           FROM sessions
                           WHERE status IN ('interrupted', 'recovered')
                           ORDER BY ended_at DESC LIMIT 5"""
                    ).fetchall()
                orphan_sessions = [
                    {
                        "id": r[0],
                        "started_at": r[1],
                        "ended_at": r[2],
                        "status": r[3],
                        "learning_event_count": r[4],
                        "data": r[5],
                    }
                    for r in orphan_rows
                ]
                retro = check_orphaned_sessions_for_reflection(orphan_sessions)
                if retro:
                    retroactive_reflection = retro
                    sid = self.current_session.session_id if self.current_session else "unknown"
                    record_reflection_event(
                        session_id=sid,
                        trigger_type="T1_retroactive",
                        prompts_sent=len(retro.get("prompts", [])),
                        prompt_data=retro.get("prompts"),
                    )
                    mark_session_reflected(retro["orphaned_session_id"])
                    logger.info("T1 retroactive reflection attached to conversation_turn")
            except Exception as e:
                logger.warning(f"T1 retroactive reflection in conversation_turn failed: {e}")

        # --- T2: In-flight reflection bookmarks ---
        reflection_bookmarks_response = None
        if self.current_session:
            self.current_session.reflection_turn_counter += 1
            # Track concepts created by auto-learn this turn
            if auto_learn_result and auto_learn_result.concepts_created:
                for lc in auto_learn_result.concepts_created:
                    self.current_session.concepts_since_last_bookmark.append(lc.concept_id)

            from app.auto_reflection import T2_TURN_INTERVAL

            if self.current_session.reflection_turn_counter >= T2_TURN_INTERVAL:
                try:
                    from app.auto_reflection import check_inflight_reflection, record_reflection_event

                    bookmark = check_inflight_reflection(
                        session_concepts_since_last_bookmark=self.current_session.concepts_since_last_bookmark,
                        existing_bookmarks=self.current_session.reflection_bookmarks,
                    )
                    if bookmark:
                        self.current_session.reflection_bookmarks.append(bookmark)
                        self.current_session.concepts_since_last_bookmark = []
                        reflection_bookmarks_response = [bookmark]
                        sid = self.current_session.session_id
                        record_reflection_event(
                            session_id=sid,
                            trigger_type="T2_bookmark",
                            prompts_sent=1,
                            prompt_data=[bookmark],
                        )
                        logger.info(f"T2 bookmark generated: {bookmark.get('hint', '')}")
                    self.current_session.reflection_turn_counter = 0
                except Exception as e:
                    logger.warning(f"T2 in-flight bookmark failed (non-fatal): {e}")

        # --- GOV: Finalize governance context ---
        governance_summary = None
        if gov_ctx:
            try:
                gov_ctx.log_event(
                    GOV_EVENT_CONVERSATION_TURN_COMPLETE,
                    None,
                    {
                        "activated_count": len(activated),
                        "activated_concept_ids": [ac.concept_id for ac in activated] if activated else [],
                        "staleness_filtered": staleness_filtered_count,
                        "shadow_expanded": len(shadow_expanded),
                        "maturity_filtered_count": maturity_filtered_count,
                        "maturity_gate_bypassed": maturity_gate_bypassed,
                        "elapsed_ms": elapsed_ms,
                    },
                )
                governance_summary = gov_ctx.finalize()

                # WS2: Metric 6 — budget_overrun_ms (when latency budget exceeded)
                try:
                    from app.metrics import metrics as _m6

                    remaining = governance_summary.get("latency_remaining_ms", 0)
                    if remaining < 0:
                        _m6.record("budget_overrun_ms", abs(remaining))
                except Exception:
                    pass

                # W3: Inject maturity gate stats into governance summary
                if governance_summary and isinstance(governance_summary, dict):
                    governance_summary["maturity_filtered_count"] = maturity_filtered_count
                    governance_summary["maturity_gate_bypassed"] = maturity_gate_bypassed
                    # W5: Inject contradiction detection stats
                    if contradiction_result:
                        governance_summary["contradictions_found"] = len(contradiction_result.pairs)
                        governance_summary["concepts_suppressed"] = len(contradiction_result.suppressed_ids)
                    else:
                        governance_summary["contradictions_found"] = 0
                        governance_summary["concepts_suppressed"] = 0

                # Flush governance events to DB so benchmarks can query them.
                # Without this, events only exist in-memory and governance_events
                # table stays empty (root cause of 12/13 adversarial failure).
                try:
                    from app.storage import _db

                    with _db() as ev_conn:
                        # CTX-008: pass session_id for analytics attribution
                        _sid = self.current_session.session_id if self.current_session else None
                        flushed = gov_ctx.flush_events_to_db(ev_conn, session_id=_sid)
                        if flushed:
                            logger.info(f"GOV: Flushed {flushed} governance events to DB")
                except Exception as flush_err:
                    logger.warning(f"GOV: Event flush failed (non-fatal): {flush_err}")

            except Exception as e:
                logger.warning(f"GOV: finalize failed (non-fatal): {e}")

        # GOV-W2: Track activated concept IDs + embeddings for correction detection on next turn
        self._last_activated_concept_ids = [ac.concept_id for ac in activated]
        # Cache concept dicts with embeddings for Layer 4 drift detection
        self._last_activated_concept_dicts = []
        try:
            from app.embedding import embedding_engine

            for ac in activated[:5]:  # Cap at 5 to limit memory
                emb = None
                pos = embedding_engine._id_to_pos.get(ac.concept_id)
                if pos is not None and embedding_engine._index_matrix is not None:
                    emb = embedding_engine._index_matrix[pos]
                self._last_activated_concept_dicts.append(
                    {
                        "concept_id": ac.concept_id,
                        "embedding": emb,
                    }
                )
        except Exception:
            pass  # Embedding cache is best-effort

        # --- CTX Phase 1: Context Priority Hints ---
        context_priority_hints = None
        try:
            from app.config import FEATURE_FLAGS as _ctx_hints_ff

            if _ctx_hints_ff.get("CONTEXT_PRIORITY_HINTS_ENABLED", False):
                try:
                    _hint_aa_ids = set(aa_ids)
                except NameError:
                    _hint_aa_ids = set()
                context_priority_hints = self._build_context_priority_hints(activated, aa_ids=_hint_aa_ids)

                # CTX Phase 3: Apply survival formatting to critical concepts
                if _ctx_hints_ff.get("COMPACTION_SURVIVAL_FORMAT", False) and context_priority_hints:
                    crit_set = set(context_priority_hints.get("critical_ids", []))
                    for ac in activated:
                        if ac.concept_id in crit_set:
                            c = load_concept(ac.concept_id, track_access=False)
                            if c:
                                ac.summary = self._format_for_compaction_survival(
                                    ac.concept_id, ac.summary, c.concept_type
                                )
        except Exception as hints_err:
            logger.warning(f"CTX Phase 1: Priority hints failed (non-fatal): {hints_err}")

        # ARCH-001: Model-agnostic skill routing
        _recommended_skills: list[str] = []
        try:
            from app.skill_index import recommend_skills

            _recommended_skills = recommend_skills(request.message, max_results=3)
            if _recommended_skills:
                logger.info(f"ARCH-001: Recommended {len(_recommended_skills)} skills")
        except Exception as skill_err:
            logger.debug(f"ARCH-001: Skill routing failed (non-fatal): {skill_err}")

        # --- EXP-025: Demand-side analogy detection ---
        _analogy_suggestions = None
        from app.config import BENCHMARK as _bm_analogy
        try:
            if _bm_analogy.skip_analogies:
                raise Exception("BENCHMARK-004: skipped in benchmark mode")
            from app.experiments import detect_demand_side_analogies

            # A1: Build concept_type mapping from concept cache
            _concept_types = {}
            for _ac in activated:
                _c = _concept_cache.get(_ac.concept_id)
                if _c:
                    _concept_types[_ac.concept_id] = getattr(_c, "concept_type", "observation")

            _analogy_suggestions = detect_demand_side_analogies(
                activated, concept_types=_concept_types
            ) or None
            if _analogy_suggestions:
                logger.info(
                    "EXP-025: %d analogy suggestion(s), top score=%.3f",
                    len(_analogy_suggestions),
                    _analogy_suggestions[0]["score"],
                )
                # MONITOR-051: Track analogy suggestion rate in metrics
                from app.metrics import metrics as _analogy_metrics
                _analogy_metrics.record("analogy_suggestions_count", len(_analogy_suggestions))
        except Exception as analogy_err:
            logger.debug(f"EXP-025: Demand-side analogy detection failed (non-fatal): {analogy_err}")

        # STABILITY-012: Factual freshness flagging (runs once per session, first turn only)
        _freshness_warnings: list[dict] = []
        if is_first_call:
            try:
                from app.staleness import scan_factual_freshness

                _freshness_warnings = scan_factual_freshness(limit=3)  # MONITOR-021: cap at 3
                if _freshness_warnings:
                    logger.info(f"STABILITY-012: {len(_freshness_warnings)} freshness warnings")
            except Exception as fresh_err:
                logger.debug(f"STABILITY-012: Freshness scan failed (non-fatal): {fresh_err}")

        # PRICING-007: Recall gap attribution
        try:
            from app.pricing import conversation_meter as _pricing_meter

            recall_gap_attribution = _pricing_meter.get_recall_gap_attribution()
        except Exception:
            pass  # Non-fatal — recall gap is informational

        # --- RETRIEVAL-040: Per-hop concept scoring for enriched chain_hint ---
        _per_hop_concepts: dict | None = None
        if _multihop_used and _multihop_clauses and len(_multihop_clauses) > 1:
            try:
                # NAMING-001: Renamed from PITH_CHAIN_REASONING (which means something
                # completely different in benchmarks — LLM decomposition engine).
                # Backward-compatible: checks new name first, falls back to old.
                _chain_flag = (
                    os.environ.get("PITH_CHAIN_HINT_ENRICHMENT", "").lower() in ("true", "1")
                    or os.environ.get("PITH_CHAIN_REASONING", "").lower() in ("true", "1")
                )
                _CHAIN_ENRICHMENT_POOL = int(os.environ.get("PITH_CHAIN_ENRICHMENT_MAX_POOL", "50"))
                if _chain_flag and _mh_retriever is not None:
                    from app.retrieval_multihop import ProductionMultiHopRetriever
                    _per_hop_concepts = ProductionMultiHopRetriever.score_concepts_per_hop(
                        _multihop_clauses,
                        activated,  # ActivatedConcept list, built earlier in conversation_turn
                        min_similarity=0.25,
                        max_pool_size=_CHAIN_ENRICHMENT_POOL,
                    )
                    logger.info(
                        f"RETRIEVAL-040: Chain reasoning scored "
                        f"{sum(len(v) for v in _per_hop_concepts.values())} concept-hop pairs "
                        f"across {len(_per_hop_concepts)} steps"
                    )
            except Exception as _cr_e:
                logger.warning(f"RETRIEVAL-040: Per-hop scoring failed (non-fatal): {_cr_e}")
                _per_hop_concepts = None

        # --- CONTEXT-001: Build working_context (returned every turn) ---
        working_context = None
        try:
            working_context = self._build_working_context_block(request)
        except Exception as wc_err:
            logger.warning(f"CONTEXT-001: working_context build failed (non-fatal): {wc_err}")

        # PERF-FORT-3 + OPT-1a: Build load_pressure notification for degradation visibility.
        # OPT-1a: Background auto-learn is an optimization, not degradation — don't count
        # it toward the level. Level reflects governance phase skips only.
        _load_pressure = None
        _phases_deferred = []
        _bg_autolearn_active = (
            auto_learn_result is None
            and request.previous_response
            and len(request.previous_response) >= 30
        )
        # OPT-1a: Governance-skipped phases are real degradation.
        _gov_skipped = governance_summary.get("phases_skipped", []) if governance_summary else []
        if _gov_skipped:
            _phases_deferred.extend(_gov_skipped)
        if _phases_deferred or _bg_autolearn_active:
            # Level based on governance skips only (not auto-learn):
            # 0 skips = "normal" (auto-learn only), 1-2 = "elevated", 3+ = "critical"
            if len(_phases_deferred) == 0:
                _level = "normal"
            elif len(_phases_deferred) <= 2:
                _level = "elevated"
            else:
                _level = "critical"
            # Include auto-learn in list for visibility, but tagged as background
            _deferred_list = _phases_deferred.copy()
            if _bg_autolearn_active:
                _deferred_list.insert(0, "auto_learn(background)")
            _load_pressure = {
                "level": _level,
                "phases_deferred": _deferred_list,
                "message": (
                    f"{len(_phases_deferred)} governance phase(s) skipped."
                    if _phases_deferred
                    else "Learning deferred to background — results appear next turn."
                ),
            }

        # MONITOR-OPT1: Persist load_pressure to governance_events for trend analysis.
        # Only log when governance phases are actually skipped (not just auto-learn background).
        if _load_pressure and _gov_skipped:
            try:
                import json as _lp_json
                from app.storage import _db as _lp_db
                with _lp_db() as _lp_conn:
                    _lp_conn.execute(
                        """INSERT INTO governance_events
                           (event_type, session_id, concept_id, details, created_at)
                           VALUES (?, ?, ?, ?, ?)""",
                        (
                            "LOAD_PRESSURE",
                            self.current_session.session_id if self.current_session else None,
                            None,
                            _lp_json.dumps({
                                "level": _load_pressure["level"],
                                "phases_skipped": _gov_skipped,
                                "governance_elapsed_ms": governance_summary.get("total_elapsed_ms") if governance_summary else None,
                                "budget_ms": governance_summary.get("latency_remaining_ms", 0) + governance_summary.get("total_elapsed_ms", 0) if governance_summary else None,
                            }),
                            _utc_now_iso(),
                        ),
                    )
            except Exception:
                pass  # Best-effort monitoring — don't affect the hot path

        # --- C1: Engine-side per-hop chain answering ---
        # When PITH_LLM_CHAIN_REASONING=true, decompose multihop questions
        # and answer each hop via LLM, chaining intermediate results.
        # Returns answer string or None (runner falls back to generate_answer).
        _chain_answer: str | None = None
        try:
            from app.chain_reasoning import engine_chain_answer
            _chain_answer = engine_chain_answer(
                question=request.message or search_query,
                activated_concepts=activated,
            )
            if _chain_answer:
                logger.info(
                    f"C1-CHAIN: Engine produced answer: "
                    f"{_chain_answer[:60]}"
                )
        except Exception as _c1_e:
            logger.warning(f"C1-CHAIN: Failed (non-fatal): {_c1_e}")
            _chain_answer = None

        response = ConversationTurnResponse(
            activated_concepts=activated,
            activation_count=len(activated),
            predictions=[],
            graph_density=graph_density,
            processing_time_ms=elapsed_ms,
            staleness_filtered_count=staleness_filtered_count,
            shadow_expanded_count=len(shadow_expanded),
            is_first_call=is_first_call,
            is_resumption=is_resumption,
            orientation_summary=orientation_summary,
            greeting_hint=greeting_hint,
            auto_learned=auto_learned,
            load_pressure=_load_pressure,
            budget_warnings=budget_warnings,
            extraction_request=extraction_request,
            retrospective_nudge=retrospective_nudge,
            retroactive_reflection=retroactive_reflection,
            reflection_bookmarks=reflection_bookmarks_response,
            governance_summary=governance_summary,
            constraint_set=constraint_set_response,
            correction_signals=correction_signals_response,
            coverage_confidence=coverage_confidence,
            coverage_score=coverage_score,  # QUALITY-002
            abstention_signal=abstention_signal,  # PRODUCT-003
            checkpoint_resume_available=checkpoint_resume_available,  # CKPT-005
            blind_spot_match=blind_spot_match,
            directives=[
                {
                    "directive_id": d["directive_id"],
                    "category": d["category"],
                    "content": d["content"],
                    "priority": d["priority"],
                }
                for d in directives_response.get("directives", [])
            ]
            or None,
            directive_budget_warning=directives_response.get("budget_warning"),
            activated_domains=activated_domain_ids or None,
            # ARCH-001: Model-agnostic skill routing
            recommended_skills=_recommended_skills,
            # EXP-025: Demand-side analogy suggestions
            analogy_suggestions=_analogy_suggestions,
            # STABILITY-012: Factual freshness warnings
            freshness_warnings=_freshness_warnings or None,
            # Resume Context v1.1
            resume_context=resume_context,
            resume_context_tier=resume_context_tier,
            resume_context_suppressed=resume_context_suppressed,
            # Context Management Integration
            compaction_detected=compaction_was_detected,
            context_priority_hints=context_priority_hints,
            # PRICING-003: Upgrade nudge
            upgrade_nudge=upgrade_nudge,
            # PRICING-007: Recall gap attribution
            recall_gap_attribution=recall_gap_attribution,
            # TEMPORAL_AWARENESS v2.4
            server_time_utc=_utc_now().isoformat(),
            # CONTEXT-001: Structured working context
            working_context=working_context,
            # RETRIEVAL-037d: Chain hint from multihop decomposition
            chain_hint=self._build_chain_hint(_multihop_used, _multihop_clauses, _per_hop_concepts),
            # SAL V0: Structured summary (None when SAL disabled)
            structured_summary=_sal_result,
            sal_context=_sal_context,
            chain_answer=_chain_answer,  # C1: Per-hop answer (None if disabled/failed)
        )

        # --- CONTEXT-001 Fix 12: Token dedup + payload metrics ---
        # When working_context carries pinned concepts, signal to client
        if working_context and working_context.get("pinned_concepts"):
            pinned_ids = {p["id"] for p in working_context["pinned_concepts"] if "id" in p}
            if pinned_ids and response.context_priority_hints:
                response.context_priority_hints["working_context_covers"] = list(pinned_ids)

        # Payload size metric
        if working_context:
            import json as _wc_metric_json
            _wc_size = len(_wc_metric_json.dumps(working_context))
            from app.metrics import metrics as _wc_metrics
            _wc_metrics.record("working_context_payload_bytes", float(_wc_size))

        # --- RC-A: Capture rolling snapshot AFTER response assembly ---
        # Best-effort, non-blocking. Failures logged, not raised.
        try:
            self._capture_rolling_snapshot(request)
        except Exception as snap_err:
            logger.warning(f"RC-A: Post-assembly snapshot failed: {snap_err}")

        # --- B5: Context pressure monitoring (CTX-003) ---
        try:
            from app.config import (
                CTX_PRESSURE_THRESHOLD_CRITICAL,
                CTX_PRESSURE_THRESHOLD_SUGGEST,
                CTX_PRESSURE_THRESHOLD_URGE,
            )

            session = self.current_session
            lec = getattr(session, "learning_event_count", 0) if session else 0

            # CTX-003: Compute composite pressure score
            pressure = self._compute_context_pressure(lec)

            # Client override: if client reports actual context utilization, use that instead
            if request.context_pressure is not None:
                pressure = max(0.0, min(1.0, request.context_pressure))

            if pressure >= CTX_PRESSURE_THRESHOLD_SUGGEST:
                response.checkpoint_suggested = True
                if pressure >= CTX_PRESSURE_THRESHOLD_CRITICAL:
                    response.checkpoint_reason = (
                        f"\U0001f534 CRITICAL: Context pressure {pressure:.0%}. "
                        "Save checkpoint IMMEDIATELY — compaction imminent."
                    )
                elif pressure >= CTX_PRESSURE_THRESHOLD_URGE:
                    response.checkpoint_reason = (
                        f"\u26a0\ufe0f CONTEXT PRESSURE {pressure:.0%}. Checkpoint now to prevent data loss."
                    )
                else:
                    response.checkpoint_reason = f"Session pressure at {pressure:.0%}. Consider checkpointing."

                # At URGE+, pre-compose checkpoint payload to reduce friction
                if pressure >= CTX_PRESSURE_THRESHOLD_URGE:
                    response.checkpoint_payload = {
                        "done": list(self._session_concept_ids)[:50],
                        "active": getattr(session, "context_hint", "") if session else "",
                        "next": [],
                        "context": {
                            "turn_count": self._episode_turn_counter,
                            "elapsed_min": round(self._get_session_elapsed_min(), 1),
                            "learning_events": lec if isinstance(lec, int) else 0,
                            "pressure_score": round(pressure, 3),
                        },
                    }

                # CKPT-008: Track nudge events for compliance measurement
                try:
                    from app.storage import _db as _nudge_db

                    with _nudge_db() as _nudge_conn:
                        _nudge_conn.execute(
                            "INSERT INTO governance_events (session_id, event_type, details, created_at) "
                            "VALUES (?, ?, ?, ?)",
                            (
                                self.current_session.session_id if self.current_session else None,
                                "checkpoint_nudge_fired",
                                json.dumps({
                                    "pressure": round(pressure, 3),
                                    "level": "critical" if pressure >= CTX_PRESSURE_THRESHOLD_CRITICAL
                                             else "urge" if pressure >= CTX_PRESSURE_THRESHOLD_URGE
                                             else "suggest",
                                    "has_payload": response.checkpoint_payload is not None,
                                }),
                                _utc_now_iso(),
                            ),
                        )
                except Exception:
                    pass  # Telemetry — never block response

            # MONITOR-001: Persist pressure_score to sessions table for trend analysis
            if self.current_session:
                try:
                    from app.storage import _db

                    with _db() as _p_conn:
                        _p_conn.execute(
                            "UPDATE sessions SET pressure_score = ? WHERE id = ?",
                            (round(pressure, 4), self.current_session.session_id),
                        )
                except Exception:
                    pass  # Non-fatal — column may not exist yet

            logger.info(
                "CTX-003: pressure=%.3f turns=%d bytes=%d lec=%d",
                pressure,
                self._episode_turn_counter,
                self._cumulative_response_bytes,
                lec if isinstance(lec, int) else 0,
            )
        except Exception as e:
            logger.debug(f"CTX-003: Pressure computation failed (non-fatal): {e}")

        # PERF-FORT-2: Dispatch deferred background auto-learn AFTER all main-path
        # DB writes complete. This prevents "database is locked" contention between
        # the background thread's session_learn and the main path's governance flush.
        try:
            _dal = locals().get('_deferred_autolearn_args')
            if _dal is not None:
                self._learn_executor.submit(self._background_autolearn, *_dal)
                logger.info("S-1: Auto-learn dispatched to background (deferred)")
        except Exception as e:
            logger.warning(f"S-1: Deferred auto-learn dispatch failed (non-fatal): {e}")

        # ARCH-D05: Periodic KA promotion (every 30 min, piggybacked on conversation_turn)
        # promote_knowledge_areas() only fires on session_end, but Cowork sessions rarely
        # end cleanly. This periodic check ensures KA promotions happen reliably.
        try:
            from app.taxonomy import _should_run_promotion, promote_knowledge_areas, _record_promotion_run
            from app.config import KA_PROMOTION_INTERVAL_MINUTES
            if _should_run_promotion(KA_PROMOTION_INTERVAL_MINUTES):
                def _bg_ka_promote():
                    try:
                        transitions = promote_knowledge_areas()
                        _record_promotion_run()
                        if transitions:
                            logger.info(f"ARCH-D05: Periodic KA promotion: {len(transitions)} transitions")
                        else:
                            logger.debug("ARCH-D05: Periodic KA promotion: no transitions needed")
                    except Exception as e:
                        logger.error(f"ARCH-D05: Periodic KA promotion failed: {e}")
                self._learn_executor.submit(_bg_ka_promote)
        except Exception as e:
            logger.debug(f"ARCH-D05: KA promotion check failed (non-fatal): {e}")

        return response

    def _get_session_elapsed_min(self) -> float:
        """CTX-003: Get elapsed minutes since session start."""
        session = self.current_session
        if session and session.started_at:
            try:
                from datetime import datetime

                started = datetime.fromisoformat(session.started_at.replace("Z", "+00:00"))
                return max(0.0, (datetime.now(UTC) - started).total_seconds() / 60.0)
            except (ValueError, TypeError):
                return 0.0
        return 0.0

    def _compute_context_pressure(self, learning_event_count: int) -> float:
        """CTX-003: Compute composite context pressure score (0.0-1.0).

        Combines 4 signals: turn count, elapsed time, cumulative response bytes,
        and learning events. Each normalized to 0.0-1.0, then weighted.
        """
        from app.config import (
            CTX_PRESSURE_BYTES_MAX,
            CTX_PRESSURE_LEARNS_MAX,
            CTX_PRESSURE_TIME_MAX,
            CTX_PRESSURE_TURNS_MAX,
            CTX_PRESSURE_WEIGHT_BYTES,
            CTX_PRESSURE_WEIGHT_LEARNS,
            CTX_PRESSURE_WEIGHT_TIME,
            CTX_PRESSURE_WEIGHT_TURNS,
        )

        # Signal 1: Turn count
        turns = min(1.0, self._episode_turn_counter / CTX_PRESSURE_TURNS_MAX)

        # Signal 2: Elapsed time (minutes)
        elapsed = min(1.0, self._get_session_elapsed_min() / CTX_PRESSURE_TIME_MAX)

        # Signal 3: Cumulative previous_response bytes
        bytes_norm = min(1.0, self._cumulative_response_bytes / CTX_PRESSURE_BYTES_MAX)

        # Signal 4: Learning events
        lec = learning_event_count if isinstance(learning_event_count, int) else 0
        learns = min(1.0, lec / CTX_PRESSURE_LEARNS_MAX)

        return (
            CTX_PRESSURE_WEIGHT_TURNS * turns
            + CTX_PRESSURE_WEIGHT_TIME * elapsed
            + CTX_PRESSURE_WEIGHT_BYTES * bytes_norm
            + CTX_PRESSURE_WEIGHT_LEARNS * learns
        )

    def _check_retrospective_needed(self) -> dict | None:
        """RETRO-001: Check if Pith needs a retrospective nudge.

        Detects when observation-to-abstraction ratio is poor, indicating
        Pith remembers WHAT happened but hasn't learned HOW to think better.

        Gating:
        - Only checks once per session (uses instance flag)
        - Escalating cooldown: 2h → 6h → 24h (tracks consecutive nudges)
        - Returns actionable protocol, not just an alert

        Returns nudge dict if retrospective needed, None otherwise.
        Budget: <10ms (one SQL query + metadata check), runs at most once/session
        """
        from app.storage import count_concepts_by_type_tier, get_metadata, set_metadata

        try:
            # Gate: only check once per session
            if getattr(self, "_retro_checked_this_session", False):
                return None
            self._retro_checked_this_session = True

            # Escalating cooldown: 2h, 6h, 24h based on consecutive nudge count
            COOLDOWN_HOURS = [2, 6, 24]
            nudge_count_str = get_metadata("retro_consecutive_nudges") or "0"
            nudge_count = int(nudge_count_str) if nudge_count_str.isdigit() else 0
            cooldown_idx = min(nudge_count, len(COOLDOWN_HOURS) - 1)
            cooldown_hours = COOLDOWN_HOURS[cooldown_idx]

            last_nudge = get_metadata("last_retrospective_nudge")
            if last_nudge:
                try:
                    last_dt = datetime.fromisoformat(last_nudge)
                    if _utc_now() - _ensure_aware(last_dt) < timedelta(hours=cooldown_hours):
                        return None
                except (ValueError, TypeError):
                    pass

            # Count concepts from the last 7 days
            since = (_utc_now() - timedelta(days=7)).isoformat()
            tier_counts = count_concepts_by_type_tier(since_iso=since)

            total = tier_counts.get("total", 0)
            l3 = tier_counts.get("L3_abstractions", 0)
            l1 = tier_counts.get("L1_observations", 0)
            ratio = tier_counts.get("ratio", 0.0)

            # Thresholds: nudge if we have enough observations but too few abstractions
            # Minimum 15 concepts before we judge, and L3 ratio below 15%
            if total < 15:
                return None
            if ratio >= 0.15:
                # Ratio improved — reset consecutive nudge counter
                if nudge_count > 0:
                    set_metadata("retro_consecutive_nudges", "0")
                return None

            # Record nudge time + increment consecutive counter
            set_metadata("last_retrospective_nudge", _utc_now_iso())
            set_metadata("retro_consecutive_nudges", str(nudge_count + 1))

            next_cooldown = COOLDOWN_HOURS[min(nudge_count + 1, len(COOLDOWN_HOURS) - 1)]

            return {
                "type": "retrospective_needed",
                "message": (
                    f"The pith has {l1} observations but only {l3} abstractions "
                    f"(ratio: {ratio:.1%}). A retrospective would help extract "
                    f"principles, methods, and heuristics from recent work."
                ),
                "L1_observations": l1,
                "L3_abstractions": l3,
                "ratio": ratio,
                "total": total,
                "cooldown_hours": next_cooldown,
                "consecutive_nudges": nudge_count + 1,
                "action_protocol": {
                    "description": "To address this, include higher-order concepts in extracted_concepts_json",
                    "steps": [
                        "Review recent work for recurring patterns and lessons learned",
                        "Extract 2-3 principles (reusable rules) or methods (repeatable processes)",
                        "Include them in extracted_concepts_json with concept_type: 'principle', 'method', or 'heuristic'",
                        "Each concept needs confidence >= 0.5 and evidence with verification markers",
                    ],
                    "target_ratio": 0.15,
                    "current_ratio": ratio,
                    "abstractions_needed": max(1, int(total * 0.15) - l3),
                },
            }
        except Exception as e:
            logger.warning(f"RETRO-001: retrospective check failed: {e}")
            return None

    # --- FIX 1a: Coverage confidence metric ---
    def _compute_coverage_confidence(self, activated: list, query_text: str) -> dict | None:
        """4-signal LLM coverage validator (COVERAGE-001).

        Two layers:
        1. Basic structural checks (always run, <2ms) — backward compatible
        2. LLM signal checks (when COVERAGE_LLM_ENABLED=True, ~200-600ms)

        Returns None if coverage is adequate, or a structured warning if
        retrieval results are incomplete, irrelevant, or at wrong abstraction.

        Spec ref: COVERAGE_001_SPEC v1.2
        """
        try:
            from app.config import COVERAGE_RELEVANCE_THRESHOLD
        except ImportError:
            COVERAGE_RELEVANCE_THRESHOLD = 0.30

        # --- Layer 1: Basic structural checks (backward compatible, <2ms) ---
        if not activated:
            return {"level": "no_results", "message": "No concepts matched this query"}

        contextual = [c for c in activated if (c.get("relevance_score") or 0) > 0]
        if not contextual:
            return {
                "level": "no_strong_match",
                "message": "All activated concepts are fixed injections (AA/firmware), none matched query",
                "top_score": 0.0,
            }

        relevant = [c for c in contextual if c.get("relevance_score", 0) > COVERAGE_RELEVANCE_THRESHOLD]

        if len(relevant) == 0:
            top_score = max(c.get("relevance_score", 0) for c in contextual)
            return {
                "level": "no_strong_match",
                "message": f"Retrieved {len(contextual)} concepts but none scored above {COVERAGE_RELEVANCE_THRESHOLD} relevance",
                "top_score": round(top_score, 4),
            }

        if len(relevant) < 3:
            top_score = max(c.get("relevance_score", 0) for c in contextual)
            return {
                "level": "sparse_coverage",
                "message": f"Only {len(relevant)} concept(s) with relevance > {COVERAGE_RELEVANCE_THRESHOLD}",
                "top_score": round(top_score, 4),
            }

        # --- Layer 2: LLM signal checks (COVERAGE-001) ---
        from app.config import get_feature_flag

        if not get_feature_flag("COVERAGE_LLM_ENABLED", False):
            return None  # Basic checks passed, LLM disabled — adequate

        # Skip for short messages (greetings, confirmations)
        if len(query_text.split()) < 5:
            return None

        # COVERAGE-001 v1.1: Coverage runs in benchmark mode too — it's a quality signal
        # that benchmarks should measure. ~200ms per query is acceptable benchmark cost.
        # Only skip if explicitly disabled via PITH_SKIP_COVERAGE_LLM=true.
        if os.environ.get("PITH_SKIP_COVERAGE_LLM", "false").lower() == "true":
            return None

        # LLM classification (SYNC — 2s hard timeout)
        try:
            signals = self._classify_coverage_signals(query_text)
        except Exception as e:
            logger.debug(f"COVERAGE-001: LLM classification failed (fail-open): {e}")
            return None

        if signals is None:
            return None

        advisories = []
        confidence = 0.8

        # Signal 1: Completeness — query expects ALL items, not just top-N
        if signals.get("completeness"):
            if len(contextual) < 5:
                advisories.append(
                    f"Query expects a complete list but only {len(contextual)} concepts matched. "
                    f"Results are likely incomplete — supplement with targeted search."
                )
                confidence = min(confidence, 0.3)

        # Signal 2: Specificity + abstraction mismatch (PRODUCT-001 detection)
        if signals.get("specificity") and contextual:
            abstract_markers = {"pattern", "principle", "method", "heuristic",
                               "decision", "constraint", "trend", "approach"}
            abstract_count = sum(
                1 for c in contextual[:5]
                if any(m in c.get("summary", "").lower() for m in abstract_markers)
            )
            if abstract_count > 0:
                advisories.append(
                    f"Query needs exact facts but {abstract_count}/{min(5, len(contextual))} "
                    f"top concepts are abstract patterns. Pith may have stored this knowledge "
                    f"as patterns rather than preserving specific details."
                )
                confidence = min(confidence, 0.4)

        # Signal 3: Named entity presence
        if signals.get("entity"):
            entity_name = signals.get("entity_value", "")
            if entity_name:
                all_text = " ".join(c.get("summary", "") for c in contextual).upper()
                if entity_name.upper() not in all_text:
                    advisories.append(
                        f"Query references '{entity_name}' but no retrieved concept mentions it. "
                        f"Use targeted search: pith_search('{entity_name}')."
                    )
                    confidence = min(confidence, 0.3)

        # Signal 4: Temporal precision (Phase 2 — placeholder)
        if signals.get("temporal_precision") and contextual:
            pass  # Phase 2: add timestamp checking

        if not advisories:
            return None  # Basic and LLM checks both passed — adequate

        # Track coverage advisory metric
        try:
            from app.metrics import metrics
            metrics.record("coverage_advisory_fired", 1)
        except Exception:
            pass

        return {
            "level": "incomplete" if confidence < 0.4 else "uncertain",
            "confidence": round(confidence, 2),
            "advisories": advisories,
            "signals": signals,
            "n_contextual": len(contextual),
        }

    def _classify_coverage_signals(self, query: str) -> dict | None:
        """SYNC function — single LLM call extracting 4 binary coverage signals.

        Uses gpt-4o-mini via OpenRouter. OpenRouter is the ONLY provider
        (no Anthropic fallback — avoids burning API credits for coverage).
        Timeout: 2 seconds. Returns safe defaults on any failure.

        COVERAGE-001 spec §5 Fix 2.
        """
        client = _get_coverage_client()
        if not client:
            return None  # No API key — skip coverage

        import json as _json

        prompt = (
            'Analyze this query and answer 4 yes/no questions.\n\n'
            '1. COMPLETENESS: Does the user want a COMPLETE LIST of items?\n'
            '   YES: "What books have I read?", "List all projects", "What tools do we use?"\n'
            '   NO: "Why did X fail?", "How does X work?", "What\'s my dog\'s name?"\n\n'
            '2. SPECIFICITY: Does the user need EXACT FACTS (numbers, dates, amounts)?\n'
            '   YES: "How many bass did I catch?", "What time is my appointment?", "What\'s the total cost?"\n'
            '   NO: "What are my hobbies?", "How do I feel about X?"\n'
            '   NO: Yes/no existence checks ("Does X exist?")\n\n'
            '3. ENTITY: Is the query ABOUT a specific named person, project, or ID that must be found?\n'
            '   YES: "Status of MEASURE-028" (looking up MEASURE-028), "What did Sarah say?" (looking up Sarah)\n'
            '   NO: "How many bass at Lake Michigan?" (Lake Michigan is context, not the lookup target)\n'
            '   NO: "What\'s Sarah\'s phone number?" (looking up a phone number, Sarah is context)\n'
            '   NO: Generic categories: "medications", "tools", "expenses", "subscriptions"\n'
            '   If YES, include ONLY the lookup target entity name.\n\n'
            '4. TEMPORAL: Does the query constrain results to a specific time window?\n'
            '   YES: "What happened last week?", "Events between Jan 1-15", "What shipped this sprint?"\n'
            '   NO: "What are my hobbies?", "How does X work?"\n\n'
            f'Query: "{query}"\n\n'
            'Reply ONLY with JSON: {{"completeness": true/false, "specificity": true/false, '
            '"entity": true/false, "entity_value": "name or empty", "temporal_precision": true/false}}'
        )

        try:
            resp = client.chat.completions.create(
                model="openai/gpt-4o-mini", max_tokens=60, temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.choices[0].message.content.strip()

            import re
            # Extract JSON — try targeted pattern first, then greedy fallback
            m = re.search(r'\{[^{}]*"completeness"[^{}]*\}', raw)
            if not m:
                m = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                result = _json.loads(m.group())
                # Track success metric
                try:
                    from app.metrics import metrics
                    metrics.record("coverage_llm_call_count", 1)
                except Exception:
                    pass
                return result
        except Exception as e:
            logger.debug(f"COVERAGE-001: LLM call failed (fail-open): {e}")
            # Track failure metric
            try:
                from app.metrics import metrics
                metrics.record("coverage_llm_failure", 1)
            except Exception:
                pass

        return None  # Fail-open: skip coverage on any error

    # --- PRODUCT-003: Confidence-gated abstention ---
    @staticmethod
    def _compute_abstention_signal(
        coverage_confidence: dict | None,
        coverage_score: float | None,
    ) -> dict | None:
        """Synthesize coverage signals into an explicit abstention recommendation.

        Two-tier decision:
        - Hard abstain: no_results or no_strong_match → high confidence abstention
        - Soft abstain: sparse_coverage + low coverage_score → moderate confidence
        - No abstain: adequate coverage → returns None

        Returns None when pith has sufficient knowledge to respond.

        Note: The 0.30 boundary for soft abstention is intentionally 0.05 below
        COVERAGE_RELEVANCE_THRESHOLD (0.35 in config). This creates a conservative
        "uncertain" band where coverage_confidence reports "sparse coverage"
        but soft abstention hasn't triggered. Soft threshold is 0.40 (calibrated
        via PCB-SIM: sparse_coverage + cs=0.35-0.38 indicates topically adjacent
        but factually irrelevant concepts). Hard threshold boundary remains at
        coverage_confidence level (no_results/no_strong_match).
        """
        if coverage_confidence is None and (coverage_score is None or coverage_score >= 0.40):
            return None  # Coverage adequate — no abstention

        level = coverage_confidence.get("level") if coverage_confidence else None

        # Hard abstain: nothing relevant found
        if level in ("no_results", "no_strong_match"):
            top_score = coverage_confidence.get("top_score", 0.0) if coverage_confidence else 0.0
            return {
                "should_abstain": True,
                "confidence": round(0.90 + (0.10 * (1.0 - min(top_score / 0.30, 1.0))), 4),
                "reason": coverage_confidence.get("message", "No relevant knowledge found"),
                "level": "hard",
            }

        # Soft abstain: sparse coverage AND low/marginal relevance score
        # Threshold raised from 0.30 to 0.40 based on PCB-SIM calibration:
        # sparse_coverage + cs=0.35-0.38 indicates topically adjacent but
        # factually irrelevant concepts (e.g., health concepts for "blood type"
        # query when blood type was never ingested). The 0.40 boundary captures
        # the "barely relevant" band that the 0.30 threshold missed.
        SOFT_ABSTAIN_SCORE_THRESHOLD = 0.40
        if level == "sparse_coverage" and coverage_score is not None and coverage_score < SOFT_ABSTAIN_SCORE_THRESHOLD:
            return {
                "should_abstain": True,
                "confidence": round(0.50 + (0.20 * (1.0 - coverage_score / SOFT_ABSTAIN_SCORE_THRESHOLD)), 4),
                "reason": f"Sparse coverage (score={coverage_score}) — knowledge may be incomplete",
                "level": "soft",
            }

        # MEASURE-032: Structurally adequate but low mean relevance.
        # Post-BENCH-017, adversarial queries (e.g., "blood type?" when never ingested)
        # activate ≥3 topically adjacent concepts above COVERAGE_RELEVANCE_THRESHOLD,
        # so coverage_confidence returns None (adequate). But mean relevance 0.30-0.40
        # indicates the concepts are tangentially related, not factually relevant.
        # This closes the gap where coverage_confidence=None + coverage_score<0.40
        # fell through all checks without triggering abstention.
        ADEQUATE_BUT_WEAK_THRESHOLD = 0.40
        if coverage_confidence is None and coverage_score is not None and coverage_score < ADEQUATE_BUT_WEAK_THRESHOLD:
            return {
                "should_abstain": True,
                "confidence": round(0.45 + (0.20 * (1.0 - coverage_score / ADEQUATE_BUT_WEAK_THRESHOLD)), 4),
                "reason": f"Adequate concept count but low mean relevance ({coverage_score:.4f}) — concepts are topically adjacent but may not contain the answer",
                "level": "soft",
            }

        # Edge case: no coverage_confidence but very low score
        if coverage_score is not None and coverage_score < 0.15:
            return {
                "should_abstain": True,
                "confidence": 0.65,
                "reason": f"Very low coverage score ({coverage_score}) with no structural signal",
                "level": "soft",
            }

        return None  # Not enough signal to recommend abstention

    # --- FIX 1b: Blind spot cross-reference ---
    def _check_blind_spot_relevance(self, query_text: str, coverage: dict | None) -> dict | None:
        """Check if query touches a known blind spot area.

        Only runs if coverage_confidence indicates sparse/no coverage.
        Adversarial F3/F10: This is a BONUS signal — coverage_confidence (1a)
        is the primary signal and works independently. Blind spots may be empty
        on cold start (before first reflection run).
        Budget: <3ms (cached blind spots, string operations only)
        """
        if coverage is None:
            return None  # Coverage is fine, no need to check blind spots

        from app.self_model import SelfModelManager

        manager = SelfModelManager()
        blind_spots = manager.get_blind_spots()
        if not blind_spots:
            return None  # Cold start or no blind spots computed

        query_lower = query_text.lower()
        query_words = set(query_lower.split())

        for bs in blind_spots:
            bs_desc = bs.description if isinstance(bs.description, str) else str(bs)
            bs_words = set(bs_desc.lower().split())
            overlap = len(bs_words & query_words)
            if overlap >= 2:
                return {
                    "blind_spot_match": bs_desc,
                    "severity": getattr(bs, "severity", "moderate"),
                    "advisory": "Knowledge in this area is sparse — treat retrieved concepts with lower confidence",
                }

        return None

    # --- FIX 2: Topic shift detection ---
    STOP_WORDS = frozenset(
        {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "shall",
            "can",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "out",
            "off",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "both",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "about",
            "up",
            "it",
            "its",
            "this",
            "that",
            "these",
            "those",
            "i",
            "me",
            "my",
            "we",
            "our",
            "you",
            "your",
            "he",
            "him",
            "his",
            "she",
            "her",
            "they",
            "them",
            "what",
            "which",
            "who",
            "whom",
            "and",
            "but",
            "if",
            "or",
            "because",
            "while",
            "although",
            "let",
            "s",
            "t",
            "don",
            "re",
            "ve",
            "ll",
        }
    )

    def _detect_topic_shift(self, current_query: str, conversation_context: str | None) -> bool:
        """Detect if current query represents a topic shift from session context.

        Spec ref: RETRIEVAL_ARCHITECTURE_SPEC v1.1, Fix 2
        Adversarial F1: Addresses BOTH anchoring mechanisms:
          (a) conversation_context string → cleared by caller setting effective_context = None
          (b) spreading activation → cleared by caller calling reset_activations()
        Adversarial F9: Phase 1 uses explicit phrases + keyword overlap.
          Phase 2 upgrade path: embedding cosine distance.
        Budget: <1ms (string operations only)
        """
        if not conversation_context:
            return False

        query_lower = current_query.lower()

        # Explicit shift signals (robust, low false-positive)
        shift_phrases = [
            "forget",
            "actually,",
            "different topic",
            "switching to",
            "change of subject",
            "moving on",
            "nevermind",
            "new question",
            "forget the",
            "forget about",
            "instead,",
            "on another note",
            "unrelated,",
            "separate question",
        ]
        if any(phrase in query_lower for phrase in shift_phrases):
            logger.info("TOPIC-SHIFT: Explicit phrase detected in query")
            return True

        # Keyword overlap check (Phase 1 heuristic)
        context_words = set(conversation_context.lower().split()) - self.STOP_WORDS
        query_words = set(query_lower.split()) - self.STOP_WORDS

        if not context_words or not query_words:
            return False

        overlap = len(context_words & query_words) / max(len(query_words), 1)
        if overlap < 0.15:
            logger.info(f"TOPIC-SHIFT: Low keyword overlap ({overlap:.2f}) detected")
            return True

        return False

    def _generate_extraction_request(
        self,
        auto_learn_result: Optional["SessionLearnResponse"],
        previous_text: str,
        current_message: str,
    ) -> dict | None:
        """B1: Analyze learning gaps and generate targeted extraction prompts.

        Runs AFTER auto-learn. Compares what was captured vs what was likely
        discussed. Returns structured request for the AI to fill gaps.

        Adversarial hardening:
        - Attack 2: Require 2+ markers per gap type (not just 1)
        - Attack 5: Session-level suppression for rejected-after-request gaps
        - Attack 8: Use concept_type (not knowledge_area) for captured_types

        Budget: <5ms (text scanning only, no disk I/O)
        """
        if not previous_text or len(previous_text) < 100:
            return None

        request_items = []
        prev_lower = previous_text.lower()

        # Attack 8 fix: Track concept_type (not knowledge_area) from created concepts
        captured_types = set()
        if auto_learn_result and auto_learn_result.concepts_created:
            for c in auto_learn_result.concepts_created:
                captured_types.add(c.concept_type)

        # --- Gap 1: Decision language without decision concept ---
        DECISION_MARKERS = [
            "decided",
            "decision",
            "going with",
            "chose",
            "opted for",
            "we'll use",
            "let's go with",
            "the fix is",
            "the approach is",
            "recommendation:",
            "verdict:",
            "conclusion:",
        ]
        # Attack 2: require 2+ markers
        decision_hits = sum(1 for m in DECISION_MARKERS if m in prev_lower)
        has_decision_language = decision_hits >= 2
        has_decision_captured = "decision" in captured_types

        if has_decision_language and not has_decision_captured:
            request_items.append(
                {
                    "type": "decision",
                    "prompt": "A decision was discussed in your previous response. Extract: what was decided, why, and what alternatives were considered.",
                    "priority": "high",
                }
            )

        # --- Gap 2: Principle/method language without abstract concept ---
        PRINCIPLE_MARKERS = [
            "always",
            "never",
            "the rule is",
            "the principle",
            "the pattern",
            "whenever you",
            "the key insight",
            "the fundamental",
            "design for",
            "the approach should be",
            "best practice",
        ]
        principle_hits = sum(1 for m in PRINCIPLE_MARKERS if m in prev_lower)
        has_principle_language = principle_hits >= 2  # Attack 2: require 2+
        has_abstract = any(t in captured_types for t in {"principle", "method", "heuristic", "cognitive_strategy"})

        if has_principle_language and not has_abstract:
            request_items.append(
                {
                    "type": "principle",
                    "prompt": "A reusable principle, method, or pattern was discussed. Extract the general rule that could apply beyond this specific situation.",
                    "priority": "medium",
                }
            )

        # --- Gap 3: Architecture/design discussion without pattern ---
        ARCH_MARKERS = [
            "architecture",
            "design pattern",
            "data model",
            "schema",
            "pipeline",
            "module",
            "component",
            "interface",
            "protocol",
            "layer",
            "service",
            "endpoint",
        ]
        has_arch_language = sum(1 for m in ARCH_MARKERS if m in prev_lower) >= 2
        has_pattern = "pattern" in captured_types

        if has_arch_language and not has_pattern:
            request_items.append(
                {
                    "type": "pattern",
                    "prompt": "An architecture or design pattern was discussed. Extract the structural insight.",
                    "priority": "medium",
                }
            )

        # --- Gap 4: Substantive text with zero extraction ---
        if len(previous_text) > 500 and auto_learn_result and auto_learn_result.learning_events == 0:
            request_items.append(
                {
                    "type": "any",
                    "prompt": "Your previous response was substantive but no knowledge was captured. What were the 1-3 key insights?",
                    "priority": "high",
                }
            )

        # --- Gap 5: Metacognitive process signals without method/heuristic ---
        # Detects when the LLM described HOW it worked through a problem
        # but didn't extract the reusable process as a method or heuristic.
        PROCESS_MARKERS = [
            "first i",
            "then i",
            "next i",
            "my approach",
            "the way i",
            "i checked",
            "i verified",
            "i traced",
            "i grepped",
            "step 1",
            "step 2",
            "the process",
            "the workflow",
            "i started by",
            "i noticed that",
            "which led me to",
        ]
        process_hits = sum(1 for m in PROCESS_MARKERS if m in prev_lower)
        has_method = any(t in captured_types for t in {"method", "heuristic", "cognitive_strategy"})

        if process_hits >= 3 and not has_method:
            request_items.append(
                {
                    "type": "method",
                    "prompt": "You described a multi-step process or investigation approach in your previous response. Extract the REUSABLE METHOD — what steps would a future session follow to solve a similar problem?",
                    "priority": "medium",
                }
            )

        # --- Gap 6: Lesson/learning language without principle extraction ---
        # Detects when the conversation discussed what was learned but the
        # takeaway wasn't captured as a reusable principle.
        LESSON_MARKERS = [
            "the lesson",
            "what we learned",
            "takeaway",
            "in hindsight",
            "the real issue was",
            "root cause",
            "the fix is",
            "going forward",
            "next time",
            "the mistake was",
            "turns out",
            "the key was",
        ]
        lesson_hits = sum(1 for m in LESSON_MARKERS if m in prev_lower)
        has_principle = "principle" in captured_types

        if lesson_hits >= 2 and not has_principle:
            request_items.append(
                {
                    "type": "principle",
                    "prompt": "Your previous response contained a lesson or retrospective insight. Extract the PRINCIPLE — what general rule applies beyond this specific case?",
                    "priority": "medium",
                }
            )

        # --- B1-Gap 7: Preference language without preference concept ---
        PREFERENCE_MARKERS = [
            "i prefer",
            "i like to",
            "i don't like",
            "i always want",
            "i never want",
            "my preference",
            "my style",
            "i'd rather",
            "don't ever",
            "always use",
            "never use",
        ]
        preference_hits = sum(1 for m in PREFERENCE_MARKERS if m in prev_lower)
        has_preference = "preference" in captured_types

        if preference_hits >= 2 and not has_preference:
            request_items.append(
                {
                    "type": "preference",
                    "prompt": "The user stated a behavioral preference in the conversation. Extract: what they prefer, the context, and any reasoning given.",
                    "priority": "medium",
                }
            )

        # Attack 2: Filter out types already requested last turn (anti-nagging)
        request_items = [item for item in request_items if item["type"] not in self._last_extraction_request_types]

        # Attack 5: Filter out session-level suppressed gaps
        request_items = [item for item in request_items if item["type"] not in self._suppressed_gap_types]

        # Update tracking for next turn
        self._last_extraction_request_types = {item["type"] for item in request_items}

        if not request_items:
            return None

        return {
            "gaps_detected": len(request_items),
            "items": request_items[:3],  # Cap at 3 requests
            "instruction": "Address these gaps by including matching concepts in extracted_concepts_json on your NEXT conversation_turn call.",
        }

    @staticmethod
    def _build_chain_hint(
        multihop_used: bool,
        clauses: list[str],
        per_hop_concepts: dict[int, list[tuple[str, str, float]]] | None = None,
    ) -> str | None:
        """RETRIEVAL-037d + RETRIEVAL-040: Build enriched reasoning chain hint.

        When multihop fires and produces >1 clause, generates a step-by-step
        reasoning chain. RETRIEVAL-040 enriches each step with per-hop concept
        snippets so the downstream LLM uses stored facts instead of parametric knowledge.
        """
        if not multihop_used or not clauses or len(clauses) <= 1:
            return None
        steps = []
        ordered_clauses = list(reversed(clauses))
        for i, clause in enumerate(ordered_clauses):
            step_num = i + 1
            clause_clean = clause.strip().rstrip(',').strip()
            if step_num == 1:
                step_line = f"Step {step_num}: Find {clause_clean}"
            else:
                step_line = f"Step {step_num}: Using the result from Step {step_num - 1}, find {clause_clean}"

            # RETRIEVAL-040: Attach per-hop stored facts
            if per_hop_concepts and step_num in per_hop_concepts:
                hop_facts = per_hop_concepts[step_num]
                if hop_facts:
                    fact_lines = []
                    for _cid, snippet, _score in hop_facts[:3]:
                        fact_lines.append(f"    → {snippet}")
                    step_line += "\n  Relevant stored facts:\n" + "\n".join(fact_lines)

            steps.append(step_line)
        return "REASONING CHAIN (follow these steps using ONLY the stored facts below each step):\n" + "\n".join(steps)

    def _extract_top_evidence(self, evidence_list, limit: int = 2) -> list[str]:
        """Extract top evidence items as strings, handling mixed formats.

        Handles: str, dict (stored Evidence), Evidence objects.
        Returns plain text strings, capped at limit.
        """
        items = []
        for e in evidence_list:
            if isinstance(e, str):
                items.append(e)
            elif isinstance(e, dict):
                content = e.get("content", "")
                if content:
                    items.append(content)
                else:
                    # Fallback to source_reference
                    items.append(e.get("source_reference", str(e)))
            elif hasattr(e, "content"):
                items.append(e.content)
            if len(items) >= limit:
                break
        return items

    # ============================================================
    # Resume Context v1.1 — Cross-Session Continuity
    # Spec: RESUME_CONTEXT_SPEC.md v1.1
    # ============================================================

    # ============================================================
    # Context Management Integration — Compaction Detection (Phase 2)
    # Spec: CONTEXT_MANAGEMENT_INTEGRATION_SPEC.md §4.2
    # Gauntlet amendments: CTX-2 (cooldown), CTX-3 (is_first_call),
    #   CTX-5 (S-0.5 position), CTX-9 (baseline measurement)
    # ============================================================

    @staticmethod
    def _format_for_compaction_survival(concept_id: str, summary: str, concept_type: str) -> str:
        """CTX Phase 3: Format critical concept for maximum survival through
        context summarization.

        Uses structured markers that LLM summarizers tend to preserve at
        higher fidelity than prose. Gated behind COMPACTION_SURVIVAL_FORMAT flag.

        Budget: ~10-20 extra tokens per critical concept.
        """
        if concept_type not in ("constraint", "decision", "principle"):
            return summary

        lines = [f'[CRITICAL-CONTEXT id="{concept_id}"]']
        # Preserve the most important 200 chars
        trimmed = summary[:200]
        lines.append(trimmed)
        lines.append("[/CRITICAL-CONTEXT]")
        return "\n".join(lines)

    def _build_context_priority_hints(
        self,
        activated: list[ActivatedConcept],
        aa_ids: set[str] | None = None,
    ) -> dict | None:
        """CTX Phase 1: Build priority hints for activated concepts.

        Classifies each activated concept as critical/high/normal/low
        based on concept_type, always-activate status, and governance scores.
        Returns a dict with critical_ids, ephemeral_ids, ttl_seconds, and
        total_critical_tokens estimate.

        Budget: ~2-5ms (loads concept_type per concept via load_concept).
        """
        from app.config import (
            CTX_TTL_ACTIVATED,
            CTX_TTL_CONSTRAINT,
            CTX_TTL_DECISION,
            CTX_TTL_FIRMWARE,
        )

        if not activated:
            return None

        aa_set = aa_ids or set()
        critical_ids = []
        ephemeral_ids = []
        ttl_seconds = {}
        total_critical_tokens = 0

        for ac in activated:
            cid = ac.concept_id

            # Always-activate / firmware → CRITICAL, no TTL
            if cid in aa_set:
                critical_ids.append(cid)
                ttl_seconds[cid] = CTX_TTL_FIRMWARE
                total_critical_tokens += 50  # ~50 tokens per firmware concept
                continue

            # Load concept type from knowledge for classification
            concept = None
            try:  # noqa: SIM105
                concept = load_concept(cid, track_access=False)
            except Exception:
                pass

            ctype = concept.concept_type if concept else "observation"

            if ctype in ("constraint", "firmware"):
                critical_ids.append(cid)
                ttl_seconds[cid] = CTX_TTL_CONSTRAINT
                total_critical_tokens += 50
            elif ctype == "decision":
                ttl_seconds[cid] = CTX_TTL_DECISION
                # Decisions with high authority are high-priority
                if concept and getattr(concept, "authority_score", 0) and concept.authority_score >= 0.70:
                    critical_ids.append(cid)
                    total_critical_tokens += 50
            elif ctype in ("principle", "method", "heuristic", "cognitive_strategy"):
                ttl_seconds[cid] = CTX_TTL_DECISION  # Same TTL as decisions
            elif ctype in ("observation", "pattern"):
                ttl_seconds[cid] = CTX_TTL_ACTIVATED
                ephemeral_ids.append(cid)
            else:
                ttl_seconds[cid] = CTX_TTL_ACTIVATED
                ephemeral_ids.append(cid)

        if not critical_ids and not ephemeral_ids:
            return None

        return {
            "critical_ids": critical_ids,
            "ephemeral_ids": ephemeral_ids,
            "ttl_seconds": ttl_seconds,
            "total_critical_tokens": total_critical_tokens,
        }

    def _detect_compaction(self, request: ConversationTurnRequest) -> bool:
        """Detect likely context compaction event. Budget: <1ms.

        Uses heuristic signals with a two-signal rule (same pattern as
        CCL correction detection). Returns True if 2+ signals fire.

        Guards:
        - Skip on is_first_call (CTX-3: Resume Context handles that)
        - Cooldown: max 1 detection per COMPACTION_COOLDOWN_SECONDS (CTX-2)
        - Session circuit breaker: disable after COMPACTION_FALSE_POSITIVE_LIMIT (CTX-2)
        """
        from app.config import (
            COMPACTION_AMNESIA_MIN_LENGTH,
            COMPACTION_CONTEXT_AMNESIA_MIN_TURNS,
            COMPACTION_COOLDOWN_SECONDS,
            COMPACTION_EMPTY_EXTRACTIONS_THRESHOLD,
            COMPACTION_FALSE_POSITIVE_LIMIT,
            COMPACTION_MIN_TURNS_FOR_DETECTION,
            COMPACTION_SIGNALS_REQUIRED,
            COMPACTION_TEMPORAL_GAP_SECONDS,
        )

        # Explicit client signal (Phase 4 future-proofing)
        if getattr(request, "compaction_detected", None) is True:
            self._last_compaction_detected_at = time.perf_counter()
            return True

        # Guard: no session or no prior turns
        if not self.current_session or not self._last_conversation_turn_at:
            return False

        turn_count = self.current_session.learning_event_count

        # Guard: CTX-3 — skip on first call (Resume Context handles it)
        if not self._conversation_turn_called:
            return False

        # Guard: too few turns for meaningful detection
        if turn_count < COMPACTION_MIN_TURNS_FOR_DETECTION:
            return False

        # Guard: CTX-2 — session circuit breaker (too many false positives)
        if self._compaction_false_positive_count >= COMPACTION_FALSE_POSITIVE_LIMIT:
            return False

        # Guard: CTX-2 — cooldown (max 1 detection per interval)
        now = time.perf_counter()
        if (
            self._last_compaction_detected_at is not None
            and (now - self._last_compaction_detected_at) < COMPACTION_COOLDOWN_SECONDS
        ):
            return False

        # --- Track consecutive empty extractions ---
        if request.extracted_concepts_json in (None, "", "[]"):
            self._consecutive_empty_extractions += 1
        else:
            self._consecutive_empty_extractions = 0

        # --- Signal 1: Temporal gap after active session ---
        gap = now - self._last_conversation_turn_at
        temporal_gap = gap > COMPACTION_TEMPORAL_GAP_SECONDS and turn_count >= COMPACTION_MIN_TURNS_FOR_DETECTION

        # --- Signal 2: Missing previous_response after established session ---
        prev_resp = request.previous_response or ""
        context_amnesia = (
            turn_count >= COMPACTION_CONTEXT_AMNESIA_MIN_TURNS and len(prev_resp) < COMPACTION_AMNESIA_MIN_LENGTH
        )

        # --- Signal 3: Consecutive empty extractions ---
        behavioral_regression = (
            turn_count >= COMPACTION_CONTEXT_AMNESIA_MIN_TURNS
            and self._consecutive_empty_extractions >= COMPACTION_EMPTY_EXTRACTIONS_THRESHOLD
        )

        # --- Two-signal rule ---
        signals = [temporal_gap, context_amnesia, behavioral_regression]
        signal_names = ["temporal_gap", "context_amnesia", "behavioral_regression"]
        fired = [n for n, s in zip(signal_names, signals, strict=False) if s]

        if len(fired) >= COMPACTION_SIGNALS_REQUIRED:
            logger.info(
                f"CTX S-0.5: Compaction detected — signals fired: {fired}, "
                f"gap={gap:.0f}s, prev_resp_len={len(prev_resp)}, "
                f"consecutive_empty={self._consecutive_empty_extractions}"
            )
            self._last_compaction_detected_at = now
            self._consecutive_empty_extractions = 0  # Reset after detection
            return True

        return False


    def _build_working_context_block(self, request) -> dict | None:
        """CONTEXT-001: Build structured working_context returned every turn.

        5 priority layers with 400-token budget:
        L1: Checkpoint state (highest priority — never trimmed)
        L2: Active task + domain
        L3: Session metadata
        L4: Tools used
        L5: Pinned concepts (lowest priority — trimmed first)

        Gated behind WORKING_CONTEXT_ENABLED feature flag.
        """
        import json as _wc_json
        from app.config import FEATURE_FLAGS as _wc_ff

        if not _wc_ff.get("WORKING_CONTEXT_ENABLED", False):
            return None

        from app.config import WORKING_CONTEXT_MAX_TOKENS

        wc: dict = {}

        # Hoisted checkpoint load — shared by L1, L2, L4
        _cp = None
        try:
            from app.storage import load_checkpoint

            session_id = None
            if self.current_session:
                session_id = self.current_session.session_id
            _cp = load_checkpoint(session_id=session_id)
        except Exception:
            pass

        # L1: Checkpoint state (highest priority)
        if _cp:
            # CKPT-004: Consumer-friendly checkpoint presentation
            done_items = _cp.get("done") or []
            next_items = _cp.get("next") or []
            total_items = len(done_items) + len(next_items)
            completion_pct = round(len(done_items) / total_items * 100) if total_items > 0 else 0

            # Time since last update
            time_since = ""
            try:
                updated = _ensure_aware(datetime.fromisoformat(_cp.get("updated_at", "")))
                delta = _utc_now() - updated
                if delta.total_seconds() < 3600:
                    time_since = f"{int(delta.total_seconds() / 60)}m ago"
                elif delta.total_seconds() < 86400:
                    time_since = f"{int(delta.total_seconds() / 3600)}h ago"
                else:
                    time_since = f"{delta.days}d ago"
            except Exception:
                time_since = "unknown"

            # Resume hint — consumer-facing one-liner
            desc = (_cp.get("description") or "")[:60]
            active = (_cp.get("active") or "")[:40]
            if active:
                resume_hint = f"You were working on: {active} ({desc})"
            elif desc:
                resume_hint = f"Pick up where you left off: {desc}"
            else:
                resume_hint = None

            wc["checkpoint"] = {
                "task_id": _cp.get("task_id"),
                "description": (_cp.get("description") or "")[:80],
                "status": _cp.get("status"),
                "active": (_cp.get("active") or "")[:60],
                "done_count": len(done_items),
                "next_count": len(next_items),
                "completion_pct": completion_pct,
                "time_since_update": time_since,
                "resume_hint": resume_hint,
            }

        # L2: Active task + domain
        active_task = self._extract_active_task(
            request.message if hasattr(request, "message") else "",
            _cached_checkpoint=_cp,
        )
        if active_task:
            task_domain = None
            if self._last_activated_concept_ids:
                try:
                    from app.storage import load_concept
                    top_c = load_concept(self._last_activated_concept_ids[0], track_access=False)
                    if top_c:
                        task_domain = top_c.knowledge_area
                except Exception:
                    pass
            wc["task"] = {
                "active_task": active_task,
                "task_domain": task_domain,
            }

        # L3: Session metadata
        session = self.current_session
        if session:
            import time as _wc_time
            elapsed = 0.0
            if session.started_at:
                try:
                    from datetime import datetime, timezone
                    started = datetime.fromisoformat(session.started_at)
                    if started.tzinfo is None:
                        started = started.replace(tzinfo=timezone.utc)
                    elapsed = (_wc_time.time() - started.timestamp()) / 60.0
                except Exception:
                    pass
            wc["session"] = {
                "session_id": session.session_id[:12],  # Truncate for token budget
                "turn_count": getattr(self, "_episode_turn_counter", 0),
                "learning_events": session.learning_event_count,
                "elapsed_minutes": round(elapsed, 1),
            }

        # L4: Tools used (from checkpoint context)
        if _cp and _cp.get("context", {}).get("tools"):
            wc["tools"] = _cp["context"]["tools"][:5]

        # L5: Pinned concepts (cached per turn)
        pinned = self._select_pinned_concepts()
        if pinned:
            wc["pinned_concepts"] = pinned

        # Token budget enforcement — priority trimming L5->L4->L3->L2 (keep L1)
        estimated_tokens = len(_wc_json.dumps(wc)) // 4
        if estimated_tokens > WORKING_CONTEXT_MAX_TOKENS:
            for trim_key in ["pinned_concepts", "tools", "session", "task"]:
                if trim_key in wc:
                    del wc[trim_key]
                if len(_wc_json.dumps(wc)) // 4 <= WORKING_CONTEXT_MAX_TOKENS:
                    break

        return wc if wc else None

    def _handle_compaction_reinjection(
        self, request: ConversationTurnRequest
    ) -> tuple[str | None, str | None, str | None, float]:
        """Re-inject critical context after compaction detection.

        Returns: (resume_context, orientation_summary, greeting_hint, recovery_quality)
        Loads the latest rolling snapshot (shared with Resume Context v1.1)
        and re-serves orientation.
        recovery_quality: 0.0–1.0 score reflecting how substantive the recovery was.
          0.4 base for having a snapshot, +0.2 each for active_task, pinned_concepts, gist.
        """
        resume_context = None
        orientation_summary = None
        greeting_hint = None
        recovery_quality = 0.0  # CTX-005: quality score for observability

        try:
            snapshot = load_resume_snapshot()
            if snapshot:
                # Build re-injection from snapshot (same format as Resume Context)
                active_task = snapshot.get("active_task", "")
                task_domain = snapshot.get("task_domain", "")
                gist = snapshot.get("last_exchange_gist", "")
                pinned = snapshot.get("pinned_concepts", [])
                pinned_summaries = [p.get("summary", "") for p in pinned if p.get("summary")]

                # CTX-005: compute recovery quality from snapshot contents
                recovery_quality = self.COMPACTION_QUALITY_HAS_SNAPSHOT
                if active_task:
                    recovery_quality += self.COMPACTION_QUALITY_HAS_TASK
                if pinned_summaries:
                    recovery_quality += self.COMPACTION_QUALITY_HAS_PINNED
                if gist:
                    recovery_quality += self.COMPACTION_QUALITY_HAS_GIST

                # CONTEXT-001: Prose fallback for resume_context (working_context carries structured data)
                parts = ["COMPACTION_RECOVERY: Your context was likely summarized."]
                parts.append("Check working_context field for full structured state.")
                if active_task:
                    parts.append(f"You were working on: {active_task}.")
                if task_domain:
                    parts.append(f"Domain: {task_domain}.")
                if pinned_summaries:
                    parts.append(f"Key context: [{', '.join(pinned_summaries[:5])}].")
                if gist:
                    parts.append(f"Last exchange touched: {gist}.")
                resume_context = " ".join(parts)

            # Re-serve orientation (TEMPORAL_AWARENESS v2.4)
            orientation_summary, greeting_hint = self._build_temporal_context(request, is_resumption=False)
            # Override greeting hint for compaction recovery
            greeting_hint = (
                "COMPACTION_RECOVERY. Your context was likely summarized. "
                "Critical operational context has been re-injected. "
                "Reference the resume_context for work state."
            )
        except Exception as e:
            logger.warning(f"CTX: Compaction re-injection failed (non-fatal): {e}")

        return resume_context, orientation_summary, greeting_hint, recovery_quality

    # Configurable thresholds (v1.1 Root Cause 5)
    RESUME_TIER_FRESH_HOURS = 2
    RESUME_TIER_RECENT_HOURS = 24
    RESUME_TIER_STALE_DAYS = 7
    RESUME_TOKEN_FRESH = 200
    RESUME_TOKEN_RECENT = 120
    RESUME_TOKEN_STALE = 60
    RESUME_DRIFT_THRESHOLD = 0.08  # v1.1: suppress injection if similarity below this

    def _capture_rolling_snapshot(self, request: ConversationTurnRequest) -> None:
        """RC-A: Capture rolling snapshot at end of conversation_turn.

        Runs after response assembly. Best-effort — failures logged, not raised.
        v1.1: Caches concept summaries at write time (not IDs).
        v1.1: Time-decay access scoring to prevent gaming.
        CONTEXT-001: Hoisted checkpoint load shared by active_task + tools + checkpoint_summary.
        """
        try:
            session = self.current_session
            if not session:
                return

            session_id = session.session_id

            # CONTEXT-001: Hoist checkpoint load — shared by active_task, tools_used, checkpoint_summary
            _cached_cp = None
            try:
                from app.storage import load_checkpoint
                _cached_cp = load_checkpoint()
            except Exception:
                pass

            # Extract active_task from user message via simple keyword extraction
            active_task = self._extract_active_task(request.message, _cached_checkpoint=_cached_cp)

            # Determine task_domain from most-activated concept's knowledge_area
            task_domain = None
            if self._last_activated_concept_ids:
                try:
                    top_concept = load_concept(self._last_activated_concept_ids[0], track_access=False)
                    if top_concept:
                        task_domain = top_concept.knowledge_area
                except Exception:
                    pass

            # Build pinned concepts (v1.1: cached summaries, time-decay scoring)
            pinned_concepts = self._select_pinned_concepts()

            # Extract gist via keyword extraction
            gist_text = (request.message or "")[-100:]
            if request.previous_response:
                gist_text += " " + (request.previous_response or "")[-200:]
            last_exchange_gist = self._extract_gist(gist_text)

            # Session metadata
            turn_count = getattr(session, "reflection_turn_counter", 0) + 1
            learning_events_count = session.learning_event_count

            # Tools used (inferred from checkpoint context) — CONTEXT-001: uses hoisted _cached_cp
            tools_used = []
            if _cached_cp and _cached_cp.get("context", {}).get("tools"):
                tools_used = _cached_cp["context"]["tools"][:5]

            # CONTEXT-001: Extract checkpoint summary for working_context L1
            checkpoint_summary = None
            if _cached_cp:
                checkpoint_summary = {
                    "task_id": _cached_cp.get("task_id"),
                    "description": (_cached_cp.get("description") or "")[:80],
                    "status": _cached_cp.get("status"),
                    "active": (_cached_cp.get("active") or "")[:60],
                    "done_count": len(_cached_cp.get("done") or []),
                    "next_count": len(_cached_cp.get("next") or []),
                }

            save_resume_snapshot(
                session_id=session_id,
                active_task=active_task,
                task_domain=task_domain,
                pinned_concepts=pinned_concepts,
                last_exchange_gist=last_exchange_gist,
                turn_count=turn_count,
                learning_events=learning_events_count,
                tools_used=tools_used,
                checkpoint_summary=checkpoint_summary,  # CONTEXT-001
            )
        except Exception as e:
            logger.warning(f"RC-A: Snapshot capture failed (non-fatal): {e}")

    def _extract_active_task(self, message: str, _cached_checkpoint: dict | None = None) -> str | None:
        """Extract active task description from user message.

        Uses simple term frequency — top terms from message, capped at 80 chars.
        Falls back to checkpoint description if active checkpoint exists.
        CONTEXT-001: Accepts _cached_checkpoint to avoid redundant load.
        """
        if not message or len(message.strip()) < 5:
            return None

        # Check for active checkpoint first
        # CONTEXT-001: Use cached checkpoint if provided
        cp = _cached_checkpoint
        if cp is None:
            try:
                from app.storage import load_checkpoint

                cp = load_checkpoint()
            except Exception:
                pass
        if cp and cp.get("status") in ("active", "planning") and cp.get("description"):
            return cp["description"][:80]

        # Simple extraction: remove common stopwords, take top terms
        import re

        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "shall",
            "can",
            "need",
            "dare",
            "ought",
            "used",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "out",
            "off",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "every",
            "both",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "because",
            "but",
            "and",
            "or",
            "if",
            "while",
            "that",
            "this",
            "what",
            "which",
            "who",
            "whom",
            "these",
            "those",
            "i",
            "me",
            "my",
            "we",
            "our",
            "you",
            "your",
            "he",
            "him",
            "his",
            "she",
            "her",
            "it",
            "its",
            "they",
            "them",
            "their",
            "let",
            "lets",
            "let's",
            "hey",
            "hi",
            "hello",
            "please",
            "thanks",
            "yeah",
            "yes",
            "ok",
            "okay",
        }
        words = re.findall(r"\b[a-zA-Z_]{3,}\b", message.lower())
        terms = [w for w in words if w not in stopwords]

        if not terms:
            return None

        # Frequency count, take top 5 terms
        from collections import Counter

        freq = Counter(terms)
        top_terms = [t for t, _ in freq.most_common(5)]
        result = ", ".join(top_terms)
        return result[:80]

    def _select_pinned_concepts(self) -> list[dict]:
        """Select top 3 pinned concepts by time-decayed access frequency.

        v1.1: Returns cached {id, summary} dicts, not just IDs.
        v1.1: Time-decay scoring prevents artificial boosting.
        CONTEXT-001: Turn-scoped cache — avoids redundant recomputation within same turn.
        """
        # CONTEXT-001: Return cached result if same turn
        current_turn = getattr(self, "_episode_turn_counter", 0)
        if (
            self._cached_pinned_concepts is not None
            and self._cached_pinned_concepts_turn == current_turn
        ):
            return self._cached_pinned_concepts

        if not self._last_activated_concept_ids:
            return []

        pinned = []
        seen = set()
        for cid in self._last_activated_concept_ids[:10]:  # Check top 10
            if cid in seen:
                continue
            seen.add(cid)
            try:
                concept = load_concept(cid, track_access=False)
                if concept and concept.confidence >= 0.1:
                    pinned.append(
                        {
                            "id": cid,
                            "summary": (concept.summary or "")[:40],
                        }
                    )
                    if len(pinned) >= 3:
                        break
            except Exception:
                continue

        # CONTEXT-001: Cache for this turn
        self._cached_pinned_concepts = pinned
        self._cached_pinned_concepts_turn = current_turn
        return pinned

    def _extract_gist(self, text: str) -> str:
        """Extract keyword gist from text via simple term frequency.

        v1.1: Sanitizes output — strips IPs, emails, tokens.
        Returns comma-separated top terms, max 120 chars.
        """
        import re

        if not text or len(text.strip()) < 5:
            return ""

        # v1.1: Sanitize — remove structured data that could leak
        text = re.sub(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "", text)  # IPs
        text = re.sub(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", "", text)  # emails
        text = re.sub(r"\b[a-fA-F0-9]{32,}\b", "", text)  # hex tokens
        text = re.sub(r"https?://\S+", "", text)  # URLs

        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "and",
            "or",
            "but",
            "not",
            "this",
            "that",
            "it",
            "its",
            "i",
            "we",
            "you",
            "they",
            "my",
            "our",
            "your",
            "their",
            "what",
            "which",
            "who",
            "how",
            "all",
            "each",
            "just",
            "also",
            "being",
            "into",
            "more",
        }
        words = re.findall(r"\b[a-zA-Z_]{3,}\b", text.lower())
        terms = [w for w in words if w not in stopwords]

        if not terms:
            return ""

        from collections import Counter

        freq = Counter(terms)
        top_terms = [t for t, _ in freq.most_common(7)]
        result = ", ".join(top_terms)
        return result[:120]

    def _inject_resume_context(self, request: ConversationTurnRequest) -> tuple[str | None, str | None, bool]:
        """RC-B: Inject resume context on first call of resumed session.

        Returns: (resume_context_str, tier, was_suppressed)
        v1.1: Topic drift detection — suppresses injection if user started new work.
        """
        try:
            # Find prior session's snapshot
            snapshot = load_resume_snapshot()
            if not snapshot:
                return None, None, False

            # Don't inject our own session's snapshot
            current_id = self.current_session.session_id if self.current_session else None
            if snapshot["session_id"] == current_id:
                return None, None, False

            # Determine tier based on age
            from datetime import datetime as dt

            captured_at = _ensure_aware(dt.fromisoformat(snapshot["captured_at"]))
            age_hours = (_utc_now() - captured_at).total_seconds() / 3600

            if age_hours > self.RESUME_TIER_STALE_DAYS * 24:
                return None, "EXPIRED", False
            elif age_hours > self.RESUME_TIER_RECENT_HOURS:
                tier = "STALE"
                budget = self.RESUME_TOKEN_STALE
            elif age_hours > self.RESUME_TIER_FRESH_HOURS:
                tier = "RECENT"
                budget = self.RESUME_TOKEN_RECENT
            else:
                tier = "FRESH"
                budget = self.RESUME_TOKEN_FRESH

            # v1.1: Topic drift detection
            if request.message:
                drift_score = self._compute_drift_score(request.message, snapshot)
                if drift_score < self.RESUME_DRIFT_THRESHOLD:
                    logger.info(
                        f"RC-B: Drift detected (score={drift_score:.3f} < {self.RESUME_DRIFT_THRESHOLD}), suppressing injection"
                    )
                    return None, tier, True  # suppressed

            # Build injection text
            active_task = snapshot.get("active_task", "")
            task_domain = snapshot.get("task_domain", "")
            gist = snapshot.get("last_exchange_gist", "")
            pinned = snapshot.get("pinned_concepts", [])
            turn_count = snapshot.get("turn_count", 0)
            learning_events = snapshot.get("learning_events", 0)
            tools = snapshot.get("tools_used", [])

            pinned_summaries = [p.get("summary", "") for p in pinned if p.get("summary")]

            if tier == "FRESH":
                parts = [f"RESUME: You were working on: {active_task}."]
                if task_domain:
                    parts.append(f"Domain: {task_domain}.")
                if pinned_summaries:
                    parts.append(f"Key context: [{', '.join(pinned_summaries)}].")
                if gist:
                    parts.append(f"Last exchange touched: {gist}.")
                if turn_count > 0:
                    parts.append(f"Session had {min(turn_count, 100)}+ turns, {learning_events} concepts learned.")
                if tools:
                    parts.append(f"Tools active: {', '.join(tools)}.")
                resume_text = " ".join(parts)

            elif tier == "RECENT":
                parts = [f"RESUME: Prior session worked on: {active_task}."]
                if task_domain:
                    parts.append(f"Domain: {task_domain}.")
                if pinned_summaries:
                    parts.append(f"Key context: [{', '.join(pinned_summaries)}].")
                resume_text = " ".join(parts)

            else:  # STALE
                parts = [f"RESUME: Last active domain: {task_domain or 'general'}."]
                if gist:
                    parts.append(f"Topics: {gist}.")
                resume_text = " ".join(parts)

            # Enforce per-tier token budget (word count proxy)
            words = resume_text.split()
            if len(words) > budget:
                resume_text = " ".join(words[:budget])

            logger.info(f"RC-B: Injecting resume context tier={tier} tokens={len(words)} task={active_task}")
            return resume_text, tier, False

        except Exception as e:
            logger.warning(f"RC-B: Resume injection failed (non-fatal): {e}")
            return None, None, False

    def _compute_drift_score(self, message: str, snapshot: dict) -> float:
        """Compute TF-IDF-like similarity between current message and snapshot context.

        Returns 0.0 (no overlap) to 1.0 (perfect match).
        Used for v1.1 drift detection — suppress injection if user started new work.
        """
        import re

        # Build snapshot text from active_task + gist
        snapshot_text = (snapshot.get("active_task", "") or "") + " " + (snapshot.get("last_exchange_gist", "") or "")

        # Tokenize both
        def tokenize(text):
            return set(re.findall(r"\b[a-zA-Z_]{3,}\b", text.lower()))

        msg_tokens = tokenize(message)
        snap_tokens = tokenize(snapshot_text)

        if not msg_tokens or not snap_tokens:
            return 0.0

        # Jaccard similarity (simple, fast, no dependencies)
        intersection = msg_tokens & snap_tokens
        union = msg_tokens | snap_tokens
        return len(intersection) / len(union) if union else 0.0

    # ============================================================
    # RC §5.5: First-Call Budget Enforcement
    # Total ceiling: 1400 tokens across all injection sources.
    # Priority (highest first): always-activate > resume > orientation > retrieved
    # ============================================================

    FIRST_CALL_TOKEN_BUDGET = 1400  # Total token ceiling for first-call injection
    TURN_TOKEN_BUDGET = 2500  # Token ceiling for non-first-call turns (PERF-024)

    def _enforce_first_call_budget(
        self,
        always_activate_concepts: list,
        resume_context: str | None,
        orientation_summary: str | None,
        activated_concepts: list,
    ) -> tuple[list, str | None, str | None, list]:
        """RC §5.5: Enforce 1400-token ceiling across all first-call injection sources.

        Priority order (highest to lowest):
        1. Always-activate concepts (never trimmed — they're firmware)
        2. Resume context (tier-budgeted already, trimmed only if catastrophic)
        3. Orientation summary (trimmed to fit remaining budget)
        4. Retrieved concepts (trimmed from tail)

        Returns: (always_activate, resume_context, orientation, activated) — all possibly trimmed.
        Token estimation: word count as proxy (1 word ≈ 1.3 tokens, but word count is close enough).
        """
        budget = self.FIRST_CALL_TOKEN_BUDGET

        def estimate_tokens(text: str | None) -> int:
            """Word-count proxy for token estimation."""
            if not text:
                return 0
            return len(text.split())

        def estimate_concept_tokens(concepts: list) -> int:
            """Estimate tokens for a list of activated concepts."""
            total = 0
            for c in concepts:
                summary = getattr(c, "summary", "") or ""
                total += len(summary.split()) + 5  # +5 for metadata overhead
            return total

        def trim_text_to_budget(text: str, max_tokens: int) -> str:
            """Trim text to fit within token budget (word-count proxy)."""
            words = text.split()
            if len(words) <= max_tokens:
                return text
            return " ".join(words[:max_tokens])

        # Phase 1: Always-activate (never trimmed — these are firmware)
        aa_tokens = estimate_concept_tokens(always_activate_concepts)
        budget -= aa_tokens

        if budget <= 0:
            # Extreme edge case: firmware alone exceeds budget.
            # Still serve firmware but zero everything else.
            logger.warning(
                f"RC §5.5: Always-activate concepts alone consume {aa_tokens} tokens "
                f"(budget={self.FIRST_CALL_TOKEN_BUDGET}). All other sources zeroed."
            )
            return always_activate_concepts, None, None, []

        # Phase 2: Resume context (already tier-budgeted, trim only if needed)
        rc_tokens = estimate_tokens(resume_context)
        if rc_tokens > budget:
            resume_context = trim_text_to_budget(resume_context, budget)
            rc_tokens = budget
            logger.info(f"RC §5.5: Resume context trimmed to {budget} tokens")
        budget -= rc_tokens

        # Phase 3: Orientation (trim to remaining budget, min 30 tokens)
        orient_tokens = estimate_tokens(orientation_summary)
        if orient_tokens > budget:
            min_orient = min(30, budget)
            orientation_summary = trim_text_to_budget(orientation_summary, max(min_orient, budget))
            orient_tokens = max(min_orient, budget)
            logger.info(f"RC §5.5: Orientation trimmed to {orient_tokens} tokens")
        budget -= min(orient_tokens, budget)

        # Phase 4: Retrieved concepts (trim from tail — least relevant first)
        concept_tokens = estimate_concept_tokens(activated_concepts)
        if concept_tokens > budget and budget > 0:
            # Keep concepts from head until budget exhausted
            trimmed = []
            running = 0
            for c in activated_concepts:
                c_tokens = len((getattr(c, "summary", "") or "").split()) + 5
                if running + c_tokens > budget:
                    break
                trimmed.append(c)
                running += c_tokens
            activated_concepts = trimmed
            logger.info(
                f"RC §5.5: Retrieved concepts trimmed from {len(activated_concepts)} "
                f"to {len(trimmed)} (budget remaining: {budget})"
            )
        elif budget <= 0:
            activated_concepts = []
            logger.info("RC §5.5: No budget remaining for retrieved concepts")

        return always_activate_concepts, resume_context, orientation_summary, activated_concepts

    def _detect_resumption(self) -> bool:
        """B5.1: Detect if this is a resumption (prior session within 24h).

        Returns True if at least one ended session exists within the last 24 hours.
        Gracefully returns False on any error.
        """
        try:
            from app.storage import list_sessions

            cutoff_24h = (_utc_now() - timedelta(hours=24)).isoformat()
            recent = list_sessions(limit=5, since=cutoff_24h)
            # Filter to sessions that actually ended (not the current one)
            # Note: list_sessions returns "id" column, not "session_id"
            current_id = self.current_session.session_id if self.current_session else None
            prior_sessions = [s for s in recent if s.get("status") == "ended" and s.get("id") != current_id]
            if prior_sessions:
                logger.info(f"B5.1: Resumption detected — {len(prior_sessions)} prior session(s) in 24h")
                return True
            return False
        except Exception as e:
            logger.warning(f"B5.1: Resumption detection failed (non-fatal): {e}")
            return False

    @staticmethod
    def _truncate_at_boundary(text: str, max_chars: int) -> str:
        """Truncate text at the last natural boundary before max_chars.

        S7.1 Fix 1: Replaces hard [:N] char slices that cut mid-word.
        Boundaries searched in priority order: sentence end (". "),
        semicolon ("; "), em dash (" — "), comma (", ").
        Falls back to last space if no boundary found (gauntlet A-1).
        """
        if len(text) <= max_chars:
            return text

        boundaries = [". ", "; ", " — ", ", "]
        truncated = text[:max_chars]

        for boundary in boundaries:
            idx = truncated.rfind(boundary)
            if idx > max_chars * 0.5:  # Don't truncate below 50%
                return truncated[: idx + len(boundary)].strip()

        # Fallback: last space (gauntlet finding A-1)
        space_idx = truncated.rfind(" ")
        if space_idx > max_chars * 0.5:
            return truncated[:space_idx].strip()

        # Ultimate fallback: hard cut (same as status quo)
        return truncated.strip()

    @staticmethod
    def _is_orientation_worthy(concept: dict) -> bool:
        """Check if a concept is quality enough for orientation display.

        S7.1 Fix 2: Simplified per gauntlet (MR-3). Only checks length
        and deletion markers. Tuned for LOW false positive rate (L-1).

        CONCEPT_LIFECYCLE_SPEC L1d: Belt-and-suspenders currency check.
        Filters concepts with non-ACTIVE currency_status even if the
        storage-layer query didn't filter them.
        """
        summary = concept.get("summary", "")

        # Minimum length — very short summaries are usually artifacts
        if len(summary) < 20:
            return False

        # Deletion markers — concepts flagged for removal
        summary_lower = summary.lower()
        for pattern in ORIENTATION_EXCLUDE_PATTERNS:
            if pattern in summary_lower:
                return False

        # CONCEPT_LIFECYCLE_SPEC L1d: Currency status guard
        currency = concept.get("currency_status", "ACTIVE")
        if currency in ("STALE", "SUPERSEDED"):
            return False

        # ORIENTATION_V2: Resolved-state text detection (uses module-level _RESOLVED_PATTERNS)
        # Defense-in-depth for frontier layer (P4: content signals > metadata)
        if _RESOLVED_PATTERNS.search(summary):  # noqa: SIM103
            return False

        return True

    def _refresh_orientation_currency(self):
        """CONCEPT_LIFECYCLE_SPEC L3: Targeted currency refresh before orientation.

        Recomputes currency_status for the ~30 most recent concepts (48h window)
        so that orientation queries filter on fresh data, not stale cached scores.

        Critical for resumption turns: if previous session ended softly (no
        end_session call), cached currency_status values may be outdated. This
        inline refresh ensures the first orientation of a new session is accurate.

        Cost: ~30 concepts × currency computation ≈ 20-40ms. Acceptable since
        orientation build already takes 100-200ms.
        """
        try:
            from app.currency import batch_compute_currency
            from app.storage import _db

            cutoff = (_utc_now() - timedelta(hours=48)).isoformat()

            with _db() as conn:
                rows = conn.execute(
                    """
                    SELECT id FROM concepts
                    WHERE status = 'active'
                      AND created_at >= ?
                      AND confidence >= 0.35
                    ORDER BY created_at DESC
                    LIMIT 30
                """,
                    (cutoff,),
                ).fetchall()

                if rows:
                    concept_ids = [row["id"] for row in rows]
                    updated = batch_compute_currency(conn, concept_ids)
                    if updated > 0:
                        logger.debug(
                            f"LIFECYCLE L3: Orientation currency refresh — "
                            f"{updated}/{len(concept_ids)} concepts updated"
                        )
        except Exception as e:
            logger.warning(f"LIFECYCLE L3: Orientation currency refresh failed: {e}")

    def _build_temporal_context(
        self, request: ConversationTurnRequest, is_resumption: bool = False
    ) -> tuple[str | None, str | None]:
        """TEMPORAL_AWARENESS v2.4: Lightweight temporal summary replacing 418-line orientation.

        Returns (orientation_summary, greeting_hint) for interface compatibility.
        orientation_summary: Single sentence about learning recency.
        greeting_hint: Type-aware behavioral directive for the LLM.
        """
        try:
            now = _utc_now()
            conn = _get_connection()

            # Most recent concept learned
            row = conn.execute(
                "SELECT created_at FROM concepts WHERE is_current=1 ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            last_learned = _ensure_aware(datetime.fromisoformat(row[0])) if row else None
            last_learned_ago = round((now - last_learned).total_seconds() / 3600, 1) if last_learned else None

            # Learning velocity (24h)
            count_24h = conn.execute(
                "SELECT COUNT(*) FROM concepts WHERE is_current=1 AND created_at > ?",
                ((now - timedelta(hours=24)).isoformat(),),
            ).fetchone()[0]

            # Build advisory
            parts = []
            if last_learned_ago is not None:
                if last_learned_ago < 1:
                    parts.append(f"Active learning session — concepts are current. {count_24h} concepts in 24h.")
                elif last_learned_ago < 6:
                    parts.append(f"Last learning {last_learned_ago:.0f}h ago. {count_24h} concepts in 24h.")
                else:
                    parts.append(f"Last learning {last_learned_ago:.0f}h ago. Retrieved concepts may be outdated.")

            # DEBT-206: Render experiment_summary in orientation
            try:
                from app.experiments import load_experiments

                _active_exps = load_experiments(status=["reasoning"], limit=5)
                if _active_exps:
                    _exp_types = set(e.experiment_type for e in _active_exps)
                    parts.append(f"{len(_active_exps)} active experiment(s): {', '.join(_exp_types)}.")
            except Exception:
                pass  # Experiment summary is enrichment, not critical path

            orientation_summary = " ".join(parts) if parts else None

            # Amendment 1: Type-aware behavioral directive
            greeting_hint = (
                "Concepts include age_minutes and freshness_label fields. "
                "Calibrate confidence by age AND type: "
                "Observations and patterns age fast — treat >1440 min (24h) as potentially outdated. "
                "Principles, constraints, and methods age slowly — a 2-week-old principle may still hold. "
                "Decisions age at medium speed — verify if context has changed. "
                "When multiple concepts conflict, prefer the newer one."
            )

            # Resumption-specific hint
            if is_resumption:
                greeting_hint = (
                    "RETURNING USER. " + greeting_hint + " "
                    "Lead with synthesis of current work context, not a generic greeting."
                )

            return orientation_summary, greeting_hint
        except Exception as e:
            logger.warning(f"TEMPORAL_AWARENESS: _build_temporal_context failed: {e}")
            return None, None

    # P0.2: Rate limiting (S7) — in-memory counter, resets on restart
    _rate_counter: dict[str, int] = {}
    SESSION_LEARN_RATE_LIMIT = int(os.environ.get("PITH_SESSION_LEARN_RATE_LIMIT", 20))  # max calls per 10-min window
    _BASE_RATE_LIMIT: int = int(os.environ.get("PITH_SESSION_LEARN_RATE_LIMIT", 20))  # INGEST-025: original default

    # INGEST-025: Bulk ingestion auto-detection state
    _bulk_call_timestamps: list[float] = []  # recent session_learn call times
    _BULK_DETECT_THRESHOLD: int = 10  # calls within window to trigger bulk mode
    _BULK_DETECT_WINDOW_S: float = 60.0  # detection window in seconds
    _BULK_ELEVATED_LIMIT: int = int(os.environ.get("PITH_BULK_ELEVATED_LIMIT", 500))  # INGEST-042: raised 200→500 for large-context ingestion; was capped at base limit, causing 35% fact drop at 64k
    _BULK_DECAY_S: float = 120.0  # seconds of quiet before reverting to base limit
    _bulk_mode_active: bool = False

    # P0.2: Daily learning budget (S4) — PRICING-001: configurable per profile tier.
    # Override via PITH_DAILY_BUDGET env var. Tier constants in config.py.
    DAILY_BUDGET = int(os.environ.get("PITH_DAILY_BUDGET", 150))
    _daily_budget_key: str = ""
    _daily_budget_count: int = 0

    def _detect_bulk_pattern(self) -> None:
        """INGEST-025: Auto-detect bulk ingestion and elevate rate limits.

        If >10 session_learn calls arrive within 60s, transparently switch to
        bulk mode (rate_limit=200). Reverts after 120s of quiet. Zero consumer config.
        """
        now = time.monotonic()
        self._bulk_call_timestamps.append(now)

        # Prune timestamps older than detection window
        cutoff = now - self._BULK_DETECT_WINDOW_S
        self._bulk_call_timestamps = [t for t in self._bulk_call_timestamps if t >= cutoff]

        if not self._bulk_mode_active:
            # Check if bulk pattern detected
            if len(self._bulk_call_timestamps) >= self._BULK_DETECT_THRESHOLD:
                self._bulk_mode_active = True
                self.SESSION_LEARN_RATE_LIMIT = max(self.SESSION_LEARN_RATE_LIMIT, self._BULK_ELEVATED_LIMIT)
                logger.info(
                    f"INGEST-025: Bulk ingestion detected ({len(self._bulk_call_timestamps)} calls in "
                    f"{self._BULK_DETECT_WINDOW_S}s). Rate limit elevated to {self.SESSION_LEARN_RATE_LIMIT}."
                )
        else:
            # Check if we should decay back to normal
            if len(self._bulk_call_timestamps) <= 1:
                # Only the current call in window — quiet period, revert
                self._bulk_mode_active = False
                self.SESSION_LEARN_RATE_LIMIT = self._BASE_RATE_LIMIT
                logger.info(
                    f"INGEST-025: Bulk mode ended (quiet >{self._BULK_DECAY_S}s). "
                    f"Rate limit restored to {self._BASE_RATE_LIMIT}."
                )

    def _check_rate_limit(self) -> tuple:
        """S7: Simple sliding-window rate limit on session_learn calls."""
        # INGEST-025: Auto-detect bulk pattern before checking limit
        self._detect_bulk_pattern()

        window_key = _utc_now().strftime("%Y%m%d%H%M")[:-1]  # 10-min window
        current = self._rate_counter.get(window_key, 0)
        if current >= self.SESSION_LEARN_RATE_LIMIT:
            return (False, 300)
        self._rate_counter[window_key] = current + 1
        # Clean old keys
        for k in list(self._rate_counter.keys()):
            if k != window_key:
                del self._rate_counter[k]
        return (True, 0)

    def _check_daily_budget(self) -> int:
        """S4: Check remaining daily concept creation budget."""
        from app.pricing import dev_mode_active

        if dev_mode_active():
            return 999999  # PRICING-005: unlimited in dev mode
        today = _utc_now().strftime("%Y%m%d")
        if self._daily_budget_key != today:
            self._daily_budget_key = today
            self._daily_budget_count = 0
        return self.DAILY_BUDGET - self._daily_budget_count

    def _consume_budget(self, knowledge_area: str = "unknown"):
        """S4: Consume one unit of daily budget."""
        self._daily_budget_count += 1
        # MONITOR-001: Emit per-concept budget consumption metric with KA label
        try:
            from app.metrics import metrics as _cb_metrics
            _cb_metrics.record("learn_concept_created", 1.0, {
                "ka": knowledge_area,
                "budget_remaining": self._check_daily_budget(),
            })
        except Exception:
            pass  # Metrics are best-effort

    def session_learn(self, request: SessionLearnRequest) -> SessionLearnResponse:
        """Post-response concept extraction. Target <200ms synchronous.

        Pipeline (P0.2 extended):
          Step 1: Rate limit + text preparation
          Step 2: Tier 2 processing (client-extracted) — parse, garbage detect, validate
          Step 3: Tier 1 processing (heuristic extraction)
          Step 4: Cross-tier dedup — TF-IDF cosine >=0.50, prefer Tier 2
          Step 5: Quality ranking + combined cap at 7
          Step 6: Existing-concept dedup — 3-zone: skip/evolve/create
          Step 7: Store + bookkeeping + response
        """
        t0 = time.perf_counter()

        from app.retrieval import retrieval_engine
        from app.retrospective import ConversationProcessor

        concepts_created: list[LearnedConcept] = []
        concepts_evolved: list[EvolvedConcept] = []
        associations_created = 0
        duplicates_skipped = 0
        concepts_skipped = 0
        errors = 0
        garbage_rejected = 0
        rejection_details: list[dict] = []  # per-concept feedback
        budget_warnings: list[str] = []  # proactive budget limit signals
        concepts_superseded = 0  # S3.5
        supersession_details: list[dict] = []  # S3.5
        source_breakdown = {"heuristic": 0, "client": 0}
        evolved_this_call: set = set()  # S3: per-call evidence cap

        # --- Step 0: Session boundary check ---
        # EC12 finding: session_learn succeeds after session_end, but counters
        # don't update (self.current_session is None). We warn but still process —
        # losing concepts is worse than a counter mismatch. The warning signals
        # the client to call session_start.
        # TODO(KTA-EC12): Consider stricter session boundary enforcement —
        #   currently we process anyway to avoid concept loss, but this means
        #   session counters can drift from reality.
        session_warning = None
        if not self.current_session:
            session_warning = "no_active_session: session_learn called without active session. Concepts will be processed but session counters won't update. Call session_start first."
            logger.warning(f"session_learn: {session_warning}")
        elif request.session_id and request.session_id != self.current_session.session_id:
            session_warning = f"session_id_mismatch: request has {request.session_id} but active session is {self.current_session.session_id}"
            logger.warning(f"session_learn: {session_warning}")

        # --- Step 1: Rate limit + text preparation ---
        allowed, retry_after = self._check_rate_limit()
        if not allowed:
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            logger.warning(
                f"session_learn: RATE_LIMIT_EXCEEDED — concepts silently dropped "
                f"(limit={self.SESSION_LEARN_RATE_LIMIT}/10min, retry_after={retry_after}s). "
                f"Concepts in this call: {len(request.extracted_concepts or [])}. "
                f"Raise PITH_SESSION_LEARN_RATE_LIMIT env var for bulk-ingest scenarios."
            )
            return SessionLearnResponse(
                concepts_created=[],
                concepts_evolved=[],
                associations_created=0,
                duplicates_skipped=0,
                concepts_skipped=0,
                errors=1,
                processing_time_ms=elapsed_ms,
                learning_events=0,
                extraction_source_breakdown=source_breakdown,
                learning_budget_remaining=self._check_daily_budget(),
                garbage_rejected=0,
                budget_warnings=[f"rate_limit_exceeded: retry after {retry_after}s"],
                concepts_superseded=0,
                supersession_details=[],
            )

        combined_text = self._prepare_text(request.user_message, request.assistant_response)
        budget_remaining = self._check_daily_budget()

        # --- Step 2: Tier 2 processing (client-extracted concepts) ---
        tier2_insights = []
        if request.extracted_concepts:
            from app.extraction import ExtractedConcept, GarbageDetector
            # DEBT-030: normalize_knowledge_area hoisted to module-level import

            # Parse and validate
            valid_concepts = []
            # BENCHMARK-002: Cap is configurable via PITH_MAX_INSIGHTS_PER_CALL (default 7).
            # Benchmark sends 20-concept batches; set env var to 30 to pass all through.
            _client_cap = int(os.environ.get("PITH_MAX_INSIGHTS_PER_CALL", "7"))
            from app.config import BENCHMARK as _bm_cap
            if _bm_cap.cap_debug_logging:
                logger.warning(
                    f"BENCHMARK-CAP-DEBUG: PITH_MAX_INSIGHTS_PER_CALL="
                    f"{os.environ.get('PITH_MAX_INSIGHTS_PER_CALL', 'NOT_SET')} → cap={_client_cap} "
                    f"concepts_sent={len(request.extracted_concepts)}"
                )
            for i, raw in enumerate(request.extracted_concepts[:_client_cap]):
                try:
                    ec = ExtractedConcept(**raw) if isinstance(raw, dict) else raw
                    valid_concepts.append(ec)
                except Exception as e:
                    logger.warning(f"session_learn: invalid extracted concept: {e}")
                    garbage_rejected += 1
                    # Extract summary preview from raw data for diagnostics
                    preview = ""
                    if isinstance(raw, dict):
                        preview = str(raw.get("summary", ""))[:80]
                    rejection_details.append(
                        {"index": i, "reason": str(e), "summary_preview": preview, "stage": "validation"}
                    )

            # Garbage detection
            # BENCHMARK-001: Skip garbage detector in benchmark mode.
            # GarbageDetector caps grounded concepts at max(5, ceil(words/200)).
            # Benchmark conversations are tiny (~35 words) → max_grounded=5 always,
            # silently discarding 15 of every 20 batch-ingested facts.
            # Facts are pre-validated by pith_agent; garbage detection is harmful here.
            from app.config import BENCHMARK as _bm_gc
            if valid_concepts:
                if _bm_gc.skip_garbage_detection:
                    survivors, rejections = valid_concepts, []
                else:
                    survivors, rejections = GarbageDetector.detect_batch(valid_concepts, combined_text)
                garbage_rejected += len(rejections)
                for r in rejections:
                    logger.info(f"session_learn: garbage rejected: {r['reason']} — {r['summary_preview']}")
                    rejection_details.append(
                        {
                            "index": r["index"],
                            "reason": r["reason"],
                            "summary_preview": r["summary_preview"],
                            "stage": "garbage_detection",
                        }
                    )

                # Emit budget_warnings for per-call limit hits
                for r in rejections:
                    reason = r.get("reason", "")
                    if "abstract_count_exceeded" in reason:
                        budget_warnings.append(f"per_call_abstract_limit: {reason}")
                    elif "grounded_count_exceeded" in reason:
                        budget_warnings.append(f"per_call_grounded_limit: {reason}")

                # Batch suspicion: if >50% failed, lower survivor confidence
                batch_suspicion = len(rejections) > len(valid_concepts) / 2

                # Convert survivors to insight dicts with extraction_source
                # DEBT-030: infer_knowledge_area hoisted to module-level import
                for ec in survivors:
                    normalized_area, _ = normalize_knowledge_area(ec.knowledge_area or "general", strict=False)
                    # KA-001: When client sends empty/general, infer from summary
                    if normalized_area == "general" and ec.summary:
                        inferred = infer_knowledge_area(ec.summary)
                        if inferred:
                            logger.info(f"KA-001 inference (client): '{ec.summary[:60]}' → {inferred} (was general)")
                            normalized_area = inferred
                    conf = ec.confidence or 0.50
                    if batch_suspicion:
                        conf = min(conf, 0.40)
                    tier2_insights.append(
                        {
                            "summary": ec.summary,
                            "confidence": conf,
                            "type": ec.concept_type or "observation",
                            "signals": ec.signals or [],
                            "evidence": ec.evidence or [],
                            "knowledge_area": normalized_area,
                            "extraction_source": "client",
                            "was_untyped": ec.concept_type is None,
                            "supersedes": ec.supersedes,  # EXPLICIT_SUPERSESSION_SPEC v1.1
                        }
                    )

        # --- Step 3: Tier 1 processing (heuristic extraction) ---
        tier1_insights = []
        if len(combined_text) >= 50:
            processor = ConversationProcessor()
            raw_insights = self._extract_insights(
                processor, combined_text,
                assistant_text=request.assistant_response or None,
            )
            for ins in raw_insights:
                ins.setdefault("extraction_source", "heuristic")  # DATA-068: preserve factual_scan
                # INGEST-008: Infer knowledge_area for heuristic insights.
                # Tier 2 (client) insights get KA via normalize + infer at ~line 5925.
                # Tier 1 (heuristic) insights lack KA, defaulting to "general" at
                # _process_single_insight:6769, which triggers the cross-KA guard
                # against any specific-KA match. Fix: infer KA from summary text.
                if not ins.get("knowledge_area") or ins.get("knowledge_area") == "general":
                    inferred_ka = infer_knowledge_area(ins.get("summary", ""))
                    if inferred_ka:
                        ins["knowledge_area"] = inferred_ka
                        logger.info(f"INGEST-008: KA inferred for heuristic insight: '{ins['summary'][:50]}' → {inferred_ka}")
                    else:
                        ins["knowledge_area"] = "general"
            tier1_insights = raw_insights

        # EXTRACT-C2: Run demographic safety net on combined Tier 1+2 before dedup
        all_pre_dedup = list(tier2_insights) + tier1_insights
        user_msg = request.user_message or ""
        asst_msg = request.assistant_response or ""
        if user_msg or asst_msg:
            enriched = self._ensure_demographic_facts(all_pre_dedup, user_msg, asst_msg)
            # Any newly injected concepts go into tier1 bucket
            new_demographic = [c for c in enriched if c.get("_source", "").startswith("EXTRACT-C2")]
            for nd in new_demographic:
                nd.setdefault("extraction_source", "heuristic")
                if not nd.get("knowledge_area") or nd["knowledge_area"] == "general":
                    nd["knowledge_area"] = "personal"
            tier1_insights.extend(new_demographic)

        # --- Step 4: Cross-tier dedup ---
        # If both tiers produced insights, remove Tier 1 duplicates of Tier 2
        merged_insights = list(tier2_insights)  # Tier 2 first (preferred)
        if tier1_insights and tier2_insights:
            from app.incremental_tfidf import IncrementalTfidfIndex

            dedup_index = IncrementalTfidfIndex()
            # Index Tier 2 summaries
            for i, t2 in enumerate(tier2_insights):
                dedup_index.add_concept(f"t2_{i}", t2["summary"])
            # Check each Tier 1 against Tier 2
            for t1 in tier1_insights:
                scores = dedup_index.search(t1["summary"], top_k=1)
                if scores and scores[0][1] >= 0.50:
                    logger.debug(f"session_learn: cross-tier dedup removed T1: {t1['summary'][:50]}")
                    continue
                merged_insights.append(t1)
        elif tier1_insights:
            merged_insights = tier1_insights

        # --- Step 5: Quality ranking + combined cap at 7 ---
        def quality_score(insight):
            score = insight.get("confidence", 0.40)
            if insight.get("evidence"):
                score += 0.1 * min(len(insight["evidence"]), 3)
            if insight.get("extraction_source") == "client":
                score += 0.05  # Slight preference for client extraction
            return score

        merged_insights.sort(key=quality_score, reverse=True)
        # BENCHMARK-002: Allow higher throughput during bulk ingestion.
        # Production default is 7 to bound per-call latency.
        _max_insights_per_call = int(os.environ.get("PITH_MAX_INSIGHTS_PER_CALL", "7"))
        merged_insights = merged_insights[:_max_insights_per_call]

        # --- INGEST-037 Phase 2a: Verbatim fragment auto-extraction ---
        # Detect high-info fragments (code, SQL, formulas, quotes) in the raw text
        # and map them to their corresponding insights. Fragments are attached to
        # each insight dict as _verbatim_fragments for downstream storage in
        # _create_new_concept.
        try:
            from app.config import get_feature_flag
            if get_feature_flag("VERBATIM_AUTO_EXTRACT_ENABLED", False):
                from app.verbatim_detect import (
                    detect_verbatim_fragments,
                    match_fragments_to_insights,
                )
                _vf_fragments = detect_verbatim_fragments(combined_text)
                if _vf_fragments and merged_insights:
                    _vf_mapping = match_fragments_to_insights(
                        _vf_fragments, merged_insights
                    )
                    for idx, frags in _vf_mapping.items():
                        if 0 <= idx < len(merged_insights):
                            merged_insights[idx]["_verbatim_fragments"] = frags
                    _vf_coverage = sum(
                        1 for i in merged_insights if i.get("_verbatim_fragments")
                    )
                    logger.info(
                        "INGEST-037: Detected %d verbatim fragments, "
                        "mapped to %d/%d insights (coverage %.0f%%)",
                        len(_vf_fragments),
                        len(_vf_mapping),
                        len(merged_insights),
                        (_vf_coverage / len(merged_insights) * 100) if merged_insights else 0,
                    )
        except Exception as _vf_err:
            logger.warning(
                "INGEST-037: Verbatim auto-extraction failed (non-fatal): %s",
                _vf_err,
            )

        if not merged_insights:
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            return SessionLearnResponse(
                concepts_created=[],
                concepts_evolved=[],
                associations_created=0,
                duplicates_skipped=0,
                concepts_skipped=0,
                errors=errors,
                processing_time_ms=elapsed_ms,
                learning_events=0,
                extraction_source_breakdown=source_breakdown,
                learning_budget_remaining=budget_remaining,
                garbage_rejected=garbage_rejected,
                rejection_details=rejection_details,
                budget_warnings=list(dict.fromkeys(budget_warnings)),
                session_warning=session_warning,
                concepts_superseded=0,
                supersession_details=[],
            )

        # --- PERF-021/036: Pre-compute batch dedup candidates (DB I/O reduction) ---
        # PERF-021: TF-IDF batch (original). PERF-036: Embedding batch (extends to embedding path).
        # Replaces N sequential encode+search+DB calls with 1 batch encode + 1 WHERE IN query.
        # Falls back to per-call search if batch fails (graceful degradation).
        _batch_dedup: list[list[dict]] | None = None
        try:
            from app.config import FEATURE_FLAGS as _perf021_ff

            _perf021_use_embedding = _perf021_ff.get("EMBEDDING_DEDUP_ENABLED", False)
            if len(merged_insights) > 1:
                _batch_summaries = [i.get("summary", "") for i in merged_insights]
                if _perf021_use_embedding:
                    # PERF-036: Batch embedding dedup
                    _batch_dedup = retrieval_engine.search_for_dedup_embedding_batch(
                        _batch_summaries, top_k=3
                    )
                else:
                    _batch_dedup = retrieval_engine.search_for_dedup_tfidf_batch(
                        _batch_summaries, top_k=3
                    )
        except Exception as _perf021_e:
            logger.warning(f"PERF-021/036: batch dedup pre-compute failed, falling back to per-call: {_perf021_e}")
            _batch_dedup = None

        # --- Step 6-7: Process each insight through dedup + create/evolve ---
        # PERF-038: Separate overhead timer from per-insight timer.
        # Pipeline overhead (extraction, dedup, batch precompute) should not count
        # against the per-insight budget. For 8k+ concept brains, overhead alone
        # consumed 1500-3000ms of the flat 5000ms budget, causing silent concept loss.
        t_insights = time.perf_counter()
        _overhead_ms = (t_insights - t0) * 1000

        from app.config import AUTOLEARN_BUDGET_MS as _learn_budget
        from app.config import AUTOLEARN_PER_INSIGHT_BUDGET_MS as _per_insight_budget
        from app.config import AUTOLEARN_MAX_BUDGET_MS as _max_budget

        # PERF-038: Scale budget by insight count, capped at max.
        _effective_budget = min(
            _learn_budget + _per_insight_budget * max(0, len(merged_insights) - 1),
            _max_budget,
        )
        logger.info(
            f"PERF-038: autolearn budget: overhead={_overhead_ms:.0f}ms "
            f"effective_budget={_effective_budget}ms (base={_learn_budget} + "
            f"{_per_insight_budget}ms * {max(0, len(merged_insights) - 1)} insights, "
            f"cap={_max_budget}ms)"
        )

        explicit_supersession_total = 0  # M1: aggregate cap (EXPLICIT_SUPERSESSION_SPEC v1.1)
        _budget_exhausted = False
        _benchmark_no_budget = os.environ.get("PITH_BENCHMARK_MODE", "").lower() in ("true", "1")
        for idx, insight in enumerate(merged_insights):
            # PERF-038: Time budget check — uses insight-only timer (excludes pipeline overhead).
            # Always process first insight regardless of budget.
            _insight_elapsed_ms = (time.perf_counter() - t_insights) * 1000
            if idx > 0 and _insight_elapsed_ms > _effective_budget and not _benchmark_no_budget:
                _skipped_count = len(merged_insights) - idx
                logger.warning(
                    f"session_learn: PERF-038 time budget exhausted "
                    f"(insight_time={_insight_elapsed_ms:.0f}ms > budget={_effective_budget}ms, "
                    f"overhead={_overhead_ms:.0f}ms) after "
                    f"{idx}/{len(merged_insights)} insights, skipping {_skipped_count}"
                )
                concepts_skipped += _skipped_count
                _budget_exhausted = True
                break
            try:
                ext_source = insight.get("extraction_source", "heuristic")

                # M1: Strip supersedes if aggregate cap reached (EXPLICIT_SUPERSESSION_SPEC v1.1)
                if explicit_supersession_total >= 10 and insight.get("supersedes"):
                    logger.info("EXPLICIT_SUPERSESSION: aggregate cap (10) reached, stripping supersedes")
                    insight["supersedes"] = None

                result = self._process_single_insight(
                    insight=insight,
                    request=request,
                    retrieval_engine=retrieval_engine,
                    extraction_source=ext_source,
                    evolved_this_call=evolved_this_call,
                    budget_remaining=self._check_daily_budget(),
                    precomputed_dedup=(
                        _batch_dedup[idx] if _batch_dedup and idx < len(_batch_dedup) else None
                    ),  # PERF-021
                )
                if result["action"] == "created":
                    concepts_created.append(result["learned_concept"])
                    associations_created += result.get("associations", 0)
                    source_breakdown[ext_source] = source_breakdown.get(ext_source, 0) + 1
                    self._consume_budget(knowledge_area=insight.get("knowledge_area", "unknown"))
                    # SESSION-005: Invalidate batch dedup cache after creation.
                    # Newly created concept is in the live index (L6 in _create_new_concept),
                    # but pre-computed batch results (PERF-021) don't see it.
                    # Force subsequent insights to use fresh live search.
                    _batch_dedup = None
                    # S3.5: Track supersessions
                    if "superseded" in result:
                        concepts_superseded += 1
                        supersession_details.append(result["superseded"])
                    # M1: Track explicit supersessions for aggregate cap
                    if "explicit_supersessions" in result:
                        explicit_supersession_total += result["explicit_supersessions"]
                        concepts_superseded += result["explicit_supersessions"]
                elif result["action"] == "evolved":
                    concepts_evolved.append(result["evolved_concept"])
                    associations_created += result.get("associations", 0)
                    source_breakdown[ext_source] = source_breakdown.get(ext_source, 0) + 1
                elif result["action"] == "skipped_duplicate":
                    duplicates_skipped += 1
                elif result["action"] == "skipped_per_call_cap":
                    duplicates_skipped += 1
                elif result["action"] == "skipped_confidence":
                    concepts_skipped += 1
                elif result["action"] == "skipped_saturated":
                    duplicates_skipped += 1
                elif result["action"] == "skipped_budget":
                    concepts_skipped += 1
                    remaining = self._check_daily_budget()
                    budget_warnings.append(f"daily_budget_exhausted: {remaining} remaining of {self.DAILY_BUDGET}/day")
            except Exception as e:
                logger.error(f"session_learn: insight processing failed: {e}")
                errors += 1

        # --- INGEST-034: Fire background event extraction ---
        from app.config import EE_ENABLED, EE_MIN_CONVERSATION_LENGTH

        if (
            EE_ENABLED
            and concepts_created
            and len(combined_text) >= EE_MIN_CONVERSATION_LENGTH
        ):
            _ee_concept_ids = [lc.concept_id for lc in concepts_created]
            import concurrent.futures as _cf_ee
            if not hasattr(self, '_event_executor') or self._event_executor is None:
                self._event_executor = _cf_ee.ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="event_extract"
                )
            self._event_executor.submit(
                self._extract_events,
                combined_text,
                _ee_concept_ids,
                request.session_id,
            )
            logger.info(
                f"INGEST-034: Event extraction fired for {len(_ee_concept_ids)} concepts "
                f"(text_len={len(combined_text)})"
            )

        # --- INGEST-038: Capture raw conversation text as verbatim fragments ---
        # CRITICAL: Use request.user_message / request.assistant_response (raw client input),
        # NOT combined_text (which may be preprocessed). Lossless capture requires raw source.
        _elapsed_total_ms = (time.perf_counter() - t0) * 1000
        _capture_budget_ms = _max_budget * 1.5  # F11: Allow 50% over insight budget for capture
        _benchmark_skip_cap = os.environ.get("PITH_BENCHMARK_MODE", "").lower() in ("true", "1")
        if _elapsed_total_ms < _capture_budget_ms or _benchmark_skip_cap:
            try:
                from app.config import get_feature_flag as _ingest038_ff
                if _ingest038_ff("VERBATIM_CONVERSATION_CAPTURE_ENABLED", True):
                    from app.verbatim_detect import capture_conversation_verbatim
                    _created_ids = [lc.concept_id for lc in concepts_created]
                    if _created_ids and request.user_message:
                        _conv_frag_ids = capture_conversation_verbatim(
                            user_message=request.user_message,
                            assistant_response=request.assistant_response or "",
                            concept_ids=_created_ids,
                            concept_versions={
                                c.concept_id: getattr(c, "version", None)
                                for c in concepts_created
                            },
                        )
                        if _conv_frag_ids:
                            logger.info(
                                "INGEST-038: Captured %d conversation fragments "
                                "for %d concepts (%.0fms elapsed)",
                                len(_conv_frag_ids), len(_created_ids), _elapsed_total_ms,
                            )
            except Exception as _conv_err:
                logger.warning(
                    "INGEST-038: Conversation capture failed (non-fatal): %s",
                    _conv_err,
                )
        else:
            logger.info(
                "INGEST-038: Skipping conversation capture (elapsed %.0fms > cap %.0fms)",
                _elapsed_total_ms, _capture_budget_ms,
            )

        # --- S6: Source-tagged logging ---
        learning_events = len(concepts_created) + len(concepts_evolved)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        # OBS-03: emit session_learn latency to metrics DB (mirrors conversation_turn pattern)
        try:
            from app.metrics import metrics as _sl_metrics

            _sl_metrics.record("session_learn_latency_ms", elapsed_ms)
        except Exception:
            pass  # Metrics are best-effort

        # Proactive daily budget warning (at 10% remaining)
        final_budget = self._check_daily_budget()
        if final_budget <= self.DAILY_BUDGET * 0.1 and final_budget > 0:
            budget_warnings.append(f"daily_budget_low: {final_budget} remaining of {self.DAILY_BUDGET}/day (under 10%)")

        # Deduplicate budget_warnings
        budget_warnings = list(dict.fromkeys(budget_warnings))

        # --- Self-awareness: update session performance counters ---
        if self.current_session and learning_events > 0:
            self.current_session.concepts_created += len(concepts_created)
            self.current_session.concepts_evolved += len(concepts_evolved)
            update_session(
                self.current_session.session_id,
                concepts_created=self.current_session.concepts_created,
                concepts_evolved=self.current_session.concepts_evolved,
            )

        untyped_count = sum(1 for i in merged_insights if i.get("was_untyped", False))

        # --- Wave 4b: Create learning event trace + set source_trace_id [X2] ---
        try:
            from app.traces import create_trace

            if concepts_created or concepts_evolved:
                concept_ref_ids = [c.concept_id for c in concepts_created] + [c.concept_id for c in concepts_evolved]
                sid = request.session_id or (self.current_session.session_id if self.current_session else "unknown")
                trace = create_trace(
                    session_id=sid,
                    trigger_type="learning_event",
                    situation=f"session_learn processing {len(merged_insights)} insights",
                    intent="Extract and persist knowledge from conversation",
                    assessment=f"Created {len(concepts_created)}, evolved {len(concepts_evolved)}",
                    justification=f"Tier1={len(tier1_insights)}, Tier2={len(tier2_insights)}",
                    concept_refs=concept_ref_ids,
                )
                # Set source_trace_id on newly created concepts [X2]
                for lc in concepts_created:
                    try:
                        from app.storage import load_concept as _lc
                        from app.storage import save_concept as _sc

                        c = _lc(lc.concept_id, track_access=False)
                        if c and not c.source_trace_id:
                            c.source_trace_id = trace.id
                            _sc(c)
                    except Exception:
                        pass  # Best-effort linkage
        except Exception as e:
            logger.debug(f"Wave 4b: trace creation skipped: {e}")

        # --- Wave 4b: Resolve predictions for evolved concepts [FIX I1] ---
        try:
            from app.traces import resolve_predictions_for_concept

            for ec in concepts_evolved:
                resolve_predictions_for_concept(ec.concept_id, outcome="revised", outcome_source="evolution")
        except Exception as e:
            logger.debug(f"Wave 4b: prediction resolution skipped: {e}")

        # --- Wave 5: Auto-link new concepts to active threads ---
        try:
            from app.threads import auto_link_candidates, link_concept_to_thread, load_threads

            active_threads = load_threads(status="active")
            if active_threads and concepts_created:
                for lc in concepts_created:
                    concept = _lc(lc.concept_id, track_access=False)
                    if concept:
                        thread_ids = auto_link_candidates(concept, active_threads)
                        for tid in thread_ids:
                            # LIFECYCLE-001: Auto-classify role based on concept_type
                            from app.threads import classify_thread_role

                            role = classify_thread_role(concept.id, concept.concept_type, tid)
                            link_concept_to_thread(tid, concept.id, role=role, added_by="auto")
        except Exception as e:
            logger.debug(f"Wave 5: thread auto-link skipped: {e}")

        logger.info(
            f"session_learn_pipeline: "
            f"tier1={len(tier1_insights)} tier2={len(tier2_insights)} "
            f"merged={len(merged_insights)} created={len(concepts_created)} "
            f"evolved={len(concepts_evolved)} superseded={concepts_superseded} "
            f"garbage={garbage_rejected} untyped={untyped_count} "
            f"budget={self._check_daily_budget()} time={elapsed_ms}ms "
            f"sources={source_breakdown} budget_exhausted={_budget_exhausted}"
        )

        # MONITOR-001: Emit pipeline-level observability metrics
        try:
            from app.metrics import metrics as _lo_metrics
            _lo_metrics.record("learn_pipeline_created", float(len(concepts_created)))
            _lo_metrics.record("learn_pipeline_evolved", float(len(concepts_evolved)))
            _lo_metrics.record("learn_pipeline_skipped", float(concepts_skipped + garbage_rejected))
            _lo_metrics.record("learn_pipeline_latency_ms", elapsed_ms)
            _lo_metrics.record("learn_budget_remaining", float(self._check_daily_budget()))
        except Exception:
            pass  # Metrics are best-effort

        # PERF-038: Record budget exhaustion metric for monitoring
        if _budget_exhausted:
            try:
                from app.metrics import metrics as _d03_metrics
                _d03_metrics.record("autolearn_budget_exhausted", 1.0, {"skipped": concepts_skipped})
            except Exception:
                pass
            budget_warnings.append(
                f"autolearn_time_budget: insight_time={_insight_elapsed_ms:.0f}ms exceeded "
                f"effective_budget={_effective_budget}ms (overhead={_overhead_ms:.0f}ms), "
                f"{concepts_skipped} insights skipped"
            )

        return SessionLearnResponse(
            concepts_created=concepts_created,
            concepts_evolved=concepts_evolved,
            associations_created=associations_created,
            duplicates_skipped=duplicates_skipped,
            concepts_skipped=concepts_skipped,
            errors=errors,
            processing_time_ms=elapsed_ms,
            learning_events=learning_events,
            extraction_source_breakdown=source_breakdown,
            learning_budget_remaining=self._check_daily_budget(),
            garbage_rejected=garbage_rejected,
            rejection_details=rejection_details,
            budget_warnings=budget_warnings,
            session_warning=session_warning,
            concepts_superseded=concepts_superseded,
            supersession_details=supersession_details,
        )

    def _prepare_text(self, user_message: str, assistant_response: str) -> str:
        """L1: Combine and clean text for insight extraction.

        Strips common boilerplate, greetings, and filler phrases.
        Returns cleaned combined text.
        """
        # Strip common AI boilerplate from assistant response
        boilerplate = [
            "Sure, ",
            "Of course, ",
            "Absolutely, ",
            "Great question! ",
            "I'd be happy to ",
            "Let me ",
            "Here's ",
            "I think ",
            "That's a great ",
            "Good question, ",
        ]
        cleaned_response = assistant_response
        for prefix in boilerplate:
            if cleaned_response.startswith(prefix):
                cleaned_response = cleaned_response[len(prefix) :]
                break

        # Strip user greetings
        greetings = ["hi ", "hey ", "hello ", "thanks ", "thank you "]
        cleaned_user = user_message
        lower_user = cleaned_user.lower()
        for g in greetings:
            if lower_user.startswith(g):
                cleaned_user = cleaned_user[len(g) :]
                break

        return f"{cleaned_user.strip()} {cleaned_response.strip()}"

    # EXTRACT-C2: Post-extraction demographic fact safety net

    @staticmethod
    def _ensure_demographic_facts(concepts: list[dict],
                                  user_msg: str, assistant_msg: str) -> list[dict]:
        """EXTRACT-C2: Post-extraction safety net for critical demographic facts.

        Scans raw conversation text for explicit age/birthday mentions and injects
        a synthetic concept if the LLM extraction missed it. This prevents
        extraction non-determinism from losing high-value retrieval anchors.

        Evidence: LongMemEval 157a136e regressed True→False when gpt-4o-mini
        non-deterministically dropped the user's age — the only concept carrying
        that signal. Without it, cross-session arithmetic was unsolvable.
        """
        import re as _re

        combined = f"{user_msg}\n{assistant_msg}".lower()
        existing_summaries = " ".join(
            c.get("summary", "").lower() for c in concepts if isinstance(c, dict)
        )

        injected = 0

        # Pattern 1: Explicit age — "I'm 32", "I am 25", "turned 28"
        # (?!\s*%) prevents false positive on "I'm 100% sure"
        age_patterns = [
            _re.compile(r"\bi['\u2019]?m\s+(\d{1,3})(?!\s*%)\b", _re.IGNORECASE),
            _re.compile(r"\bi am\s+(\d{1,3})\s*(?:years?\s*old)?(?!\s*%)\b", _re.IGNORECASE),
            _re.compile(r"\b(?:just\s+)?turned\s+(\d{1,3})\b", _re.IGNORECASE),
            _re.compile(r"\bmy\s+(\d{1,3})(?:st|nd|rd|th)\s+birthday\b", _re.IGNORECASE),
            _re.compile(r"\bat\s+(\d{1,3})[,]\s+(?:you|i|we)\b", _re.IGNORECASE),
        ]
        for pattern in age_patterns:
            m = pattern.search(combined)
            if m:
                age = int(m.group(1))
                if 16 <= age <= 95:  # plausible human age (tighter than benchmark)
                    age_str = str(age)
                    if age_str not in existing_summaries:
                        concepts.append({
                            "summary": f"The user is {age} years old",
                            "confidence": 0.8,
                            "knowledge_area": "personal",
                            "concept_type": "observation",
                            "evidence": [m.group(0)[:80]],
                            "is_factual": True,
                            "temporal_category": "identity",
                            "_source": "EXTRACT-C2-demographic-safety-net",
                        })
                        injected += 1
                        logger.info(f"EXTRACT-C2: Injected missing age fact: user is {age} "
                                    f"(pattern: {pattern.pattern})")
                    break  # One age fact is enough

        # Pattern 2: Decade of life — "in my 30s", "being in my 20s"
        if injected == 0:
            decade_match = _re.search(
                r"\b(?:in\s+)?my\s+(\d0)s\b", combined, _re.IGNORECASE
            )
            if decade_match:
                decade = decade_match.group(1)
                if decade + "s" not in existing_summaries:
                    concepts.append({
                        "summary": f"The user is in their {decade}s",
                        "confidence": 0.7,
                        "knowledge_area": "personal",
                        "concept_type": "observation",
                        "evidence": [decade_match.group(0)[:80]],
                        "is_factual": True,
                        "temporal_category": "identity",
                        "_source": "EXTRACT-C2-demographic-safety-net",
                    })
                    injected += 1
                    logger.info(f"EXTRACT-C2: Injected missing decade fact: user in {decade}s")

        # Pattern 3: Birth year — "born in 1992", "birth year is 1990"
        birth_match = _re.search(
            r"\bborn\s+(?:in\s+)?(\d{4})\b", combined, _re.IGNORECASE
        )
        if birth_match:
            year = birth_match.group(1)
            if year not in existing_summaries:
                concepts.append({
                    "summary": f"The user was born in {year}",
                    "confidence": 0.8,
                    "knowledge_area": "personal",
                    "concept_type": "observation",
                    "evidence": [birth_match.group(0)[:80]],
                    "is_factual": True,
                    "temporal_category": "identity",
                    "_source": "EXTRACT-C2-demographic-safety-net",
                })
                injected += 1
                logger.info(f"EXTRACT-C2: Injected missing birth year: {year}")

        if injected:
            logger.info(f"EXTRACT-C2: Safety net injected {injected} demographic fact(s)")
        return concepts

    # PERF-001: Tier 3 LLM Extraction (background, non-blocking)

    async def _tier3_llm_extraction(
        self,
        user_message: str,
        assistant_response: str,
        existing_insights: list[dict],
        request: "SessionLearnRequest",
    ) -> None:
        """PERF-001: Background Tier 3 LLM extraction.

        Runs async after conversation_turn returns. Calls Haiku to extract
        additional concepts from the conversation that Tier 1+2 missed.
        Results are processed through the standard session_learn pipeline.
        """
        import os
        import time
        from datetime import datetime

        from app.config import (
            TIER3_DAILY_BUDGET,
            TIER3_LLM_MODEL,
            TIER3_MAX_CONCEPTS_PER_CALL,
            TIER3_MAX_OUTPUT_TOKENS,
            TIER3_MIN_CONVERSATION_LENGTH,
        )

        t0 = time.perf_counter()

        # Gate 1: Minimum conversation length
        combined_len = len(user_message or "") + len(assistant_response or "")
        if combined_len < TIER3_MIN_CONVERSATION_LENGTH:
            logger.debug("PERF-001: Skipping Tier 3 — conversation too short")
            return

        # Gate 2: API key available
        if not os.environ.get("ANTHROPIC_API_KEY"):
            logger.debug("PERF-001: Skipping Tier 3 — no ANTHROPIC_API_KEY")
            return

        # Gate 3: Daily budget check
        if not hasattr(self, "_tier3_calls_today"):
            self._tier3_calls_today = 0
            self._tier3_day = datetime.now(UTC).date()

        current_day = datetime.now(UTC).date()
        if current_day != self._tier3_day:
            self._tier3_calls_today = 0
            self._tier3_day = current_day

        if self._tier3_calls_today >= TIER3_DAILY_BUDGET:
            logger.info(f"PERF-001: Tier 3 daily budget exhausted ({TIER3_DAILY_BUDGET})")
            return

        # Gate 4: Cooldown check
        if hasattr(self, "_tier3_last_call"):
            from app.config import TIER3_COOLDOWN_SECONDS

            elapsed = time.perf_counter() - self._tier3_last_call
            if elapsed < TIER3_COOLDOWN_SECONDS:
                logger.debug(f"PERF-001: Tier 3 cooldown ({elapsed:.1f}s < {TIER3_COOLDOWN_SECONDS}s)")
                return

        try:
            from app.extraction import build_tier3_prompt, parse_tier3_response
            from app.taxonomy import get_ka_hints

            # KA-INJECT-001: Fetch user's KA vocabulary for extraction guidance
            try:
                ka_hint_list = get_ka_hints(max_hints=12)
            except Exception:
                ka_hint_list = None  # Fallback handled in build_tier3_prompt

            # Build prompt
            prompt = build_tier3_prompt(
                user_message=user_message,
                assistant_response=assistant_response,
                existing_concepts=existing_insights,
                max_concepts=TIER3_MAX_CONCEPTS_PER_CALL,
                session_date=getattr(self, '_session_date', None),  # RAGAS-DIAG-001 Fix 3c
                ka_hints=ka_hint_list,  # KA-INJECT-001
            )

            # Call Haiku
            import anthropic

            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            client = anthropic.AsyncAnthropic(api_key=api_key)
            response = await client.messages.create(
                model=TIER3_LLM_MODEL,
                max_tokens=TIER3_MAX_OUTPUT_TOKENS,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )

            raw_text = response.content[0].text if response.content else ""
            self._tier3_calls_today += 1
            self._tier3_last_call = time.perf_counter()

            # Record cost metric
            from app.metrics import metrics as _t3_metrics

            _t3_metrics.record(
                "tier3_llm_call",
                1.0,
                {
                    "model": TIER3_LLM_MODEL,
                    "input_tokens": response.usage.input_tokens if response.usage else 0,
                    "output_tokens": response.usage.output_tokens if response.usage else 0,
                },
            )

            # Parse response
            tier3_concepts = parse_tier3_response(raw_text, TIER3_MAX_CONCEPTS_PER_CALL)

            # EXTRACT-C2: Post-extraction demographic fact safety net
            tier3_concepts = self._ensure_demographic_facts(
                tier3_concepts, user_message, assistant_response
            )

            if not tier3_concepts:
                logger.info("PERF-001: Tier 3 extracted 0 additional concepts")
                return

            logger.info(f"PERF-001: Tier 3 extracted {len(tier3_concepts)} concepts, processing...")

            # Process through standard session_learn pipeline
            from app.retrieval import retrieval_engine

            evolved_this_call = set()  # Gauntlet A1: shared across batch
            for insight in tier3_concepts:
                try:
                    result = self._process_single_insight(
                        insight=insight,
                        request=request,
                        retrieval_engine=retrieval_engine,
                        extraction_source="llm_tier3",
                        evolved_this_call=evolved_this_call,
                        budget_remaining=TIER3_MAX_CONCEPTS_PER_CALL,  # Gauntlet A2: independent of main budget
                    )
                    if result["action"] in ("created", "evolved"):
                        logger.info(f"PERF-001: Tier 3 {result['action']}: {insight['summary'][:80]}")
                except Exception as e:
                    logger.warning(f"PERF-001: Tier 3 insight processing failed: {e}")

            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            logger.info(f"PERF-001: Tier 3 completed in {elapsed_ms:.0f}ms")

        except Exception as e:
            logger.error(f"PERF-001: Tier 3 extraction failed: {e}")

    # INGEST-022: Temporal anchor detection regex for enriching extracted concepts.
    # Matches dates, relative time references, durations, and time-of-day patterns.
    _TEMPORAL_ANCHOR_RE = _re.compile(
        r'\b(?:'
        r'\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*'
        r'|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2}(?:st|nd|rd|th)?'
        r'|\d{4}[-/]\d{2}[-/]\d{2}'
        r'|(?:last|this|next)\s+(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun|week|month|year)\w*'
        r'|yesterday|today|tomorrow'
        r'|(?:about|around|roughly|approximately)\s+\d+\s+(?:days?|weeks?|months?|years?)\s+ago'
        r'|\d+\s+(?:days?|weeks?|months?|years?)\s+ago'
        r'|(?:in|on|since|before|after)\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*(?:\s+\d{4})?'
        r'|\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?'
        r')\b',
        _re.IGNORECASE
    )

    @staticmethod
    def _enrich_temporal_anchors(summary: str, source_sentence: str) -> str:
        """INGEST-022: Append temporal markers from source that are missing from summary."""
        markers = SessionManager._TEMPORAL_ANCHOR_RE.findall(source_sentence)
        if not markers:
            return summary
        # Check which markers are already in the summary
        summary_lower = summary.lower()
        missing = [m.strip() for m in markers if m.strip().lower() not in summary_lower]
        if not missing:
            return summary
        # Append temporal tag (cap at 2 markers to avoid bloat)
        tag = ", ".join(missing[:2])
        return f"{summary} [temporal: {tag}]"

    def _extract_insights(
        self,
        processor: "ConversationProcessor",
        text: str,
        assistant_text: str | None = None,
    ) -> list[dict]:
        """L2: Extract insights using extended heuristic patterns.

        Returns list of dicts: {summary, confidence, type, signals}.
        Uses ConversationProcessor as base, then applies additional
        conversation-specific patterns for richer extraction.

        INGEST-020: assistant_text enables role-aware factual extraction —
        declarative facts stated by the assistant (thresholds, defaults,
        return values) are scanned separately from combined user+assistant text.
        """
        insights = []

        # Pattern set for conversation-specific insight extraction
        # These catch common knowledge-sharing patterns in AI conversations
        # Types mapped to valid CONCEPT_TYPES (Knowledge Hierarchy L1-L6)
        conversation_patterns = [
            # Decisions and choices made → L2: decision
            {
                "markers": ["decided to", "decision is", "we chose", "going with", "opted for"],
                "type": "decision",
                "base_confidence": 0.55,
            },
            # Technical discoveries or findings → L1: observation
            {
                "markers": ["found that", "discovered that", "turns out", "realized that", "root cause"],
                "type": "observation",
                "base_confidence": 0.50,
            },
            # Architecture and design patterns → L1: pattern
            {
                "markers": ["architecture", "design pattern", "schema", "data model", "pipeline"],
                "type": "pattern",
                "base_confidence": 0.45,
            },
            # Process or methodology insights → L4: method
            {
                "markers": ["workflow", "best practice", "methodology", "approach", "strategy"],
                "type": "method",
                "base_confidence": 0.45,
            },
            # Performance insights → L1: observation
            {
                "markers": ["performance", "latency", "benchmark", "optimization", "bottleneck"],
                "type": "observation",
                "base_confidence": 0.50,
            },
            # Problem-solution pairs → L2: decision
            {
                "markers": ["the fix", "solution is", "solved by", "resolved by", "workaround"],
                "type": "decision",
                "base_confidence": 0.55,
            },
            # Requirements and constraints → L2: constraint
            {
                "markers": ["requirement", "constraint", "must be", "non-negotiable", "critical that"],
                "type": "constraint",
                "base_confidence": 0.50,
            },
            # Tradeoffs, comparisons → L1: observation
            {
                "markers": [
                    "tradeoff",
                    "trade-off",
                    "versus",
                    " vs ",
                    "compared to",
                    "distinction between",
                    "difference between",
                    "advantage",
                    "disadvantage",
                ],
                "type": "observation",
                "base_confidence": 0.45,
            },
            # Key insights and explanations → L1: observation
            {
                "markers": [
                    "the key ",
                    "the core ",
                    "the main ",
                    "essential ",
                    "the important ",
                    "the critical ",
                    "the fundamental ",
                ],
                "type": "observation",
                "base_confidence": 0.45,
            },
            # Causal reasoning → L1: pattern (recurring causal pattern)
            {
                "markers": ["because ", "the reason ", "this causes", "leads to ", "results in ", "due to "],
                "type": "pattern",
                "base_confidence": 0.45,
            },
            # Recommendations and guidance → L5: heuristic
            {
                "markers": ["recommend", "should use", "you should", "better to ", "prefer ", "avoid ", "don't use"],
                "type": "heuristic",
                "base_confidence": 0.50,
            },
            # User-stated preferences → L1.5: preference
            {
                "markers": ["i prefer ", "i like to ", "i don't like", "i always want", "i never want", "my preference", "i'd rather "],
                "type": "preference",
                "base_confidence": 0.50,
            },
            # INGEST-015: Personal identity facts → observation (is_factual, identity)
            {
                "markers": ["my name is", "i'm called ", "i am called "],
                "type": "observation",
                "base_confidence": 0.65,
                "is_factual": True,
                "temporal_category": "identity",
            },
            # EXTRACT-C2: Demographic age/birthday facts → observation (is_factual, identity)
            {
                "markers": ["i'm ", "i am ", "years old", "just turned ", "my birthday",
                            "born in ", "birth year", "in my 20s", "in my 30s",
                            "in my 40s", "in my 50s", "in my 60s"],
                "type": "observation",
                "base_confidence": 0.65,
                "is_factual": True,
                "temporal_category": "identity",
            },
            # INGEST-015: Role/employment facts → observation (is_factual, role)
            {
                "markers": ["i work at ", "i work for ", "i work as ", "my job is", "my role is", "my title is", "i'm employed at"],
                "type": "observation",
                "base_confidence": 0.65,
                "is_factual": True,
                "temporal_category": "role",
            },
            # INGEST-015: Relational facts → observation (is_factual, relational)
            {
                "markers": ["my partner ", "my wife ", "my husband ", "my girlfriend ", "my boyfriend ", "my manager ", "my boss ", "my colleague ", "my cofounder "],
                "type": "observation",
                "base_confidence": 0.60,
                "is_factual": True,
                "temporal_category": "relational",
            },
            # INGEST-022: Temporal activity facts → observation (is_factual, activity)
            # Catches statements with temporal anchors that no other pattern would match.
            {
                "markers": [" ago ", "last week", "last month", "last year",
                            "yesterday ", "this morning", "this afternoon",
                            "this evening", "last night"],
                "type": "observation",
                "base_confidence": 0.50,
                "is_factual": True,
                "temporal_category": "activity",
            },
            # PRODUCT-002: Quantitative episodic facts → observation (is_factual, episodic)
            # Catches "caught 10 bass", "spent $500", "finished 3 projects"
            {
                "markers": ["caught ", "spent ", "earned ", "saved ", "lost ",
                            "bought ", "sold ", "paid ", "scored ", "gained ",
                            "completed ", "finished ", "visited ", "traveled ",
                            "drove ", "walked ", "ran ", "swam "],
                "type": "observation",
                "base_confidence": 0.55,
                "is_factual": True,
                "temporal_category": "episodic",
            },
            # PRODUCT-002: Possessive count facts → observation (is_factual, count)
            # Catches "I have 3 dogs", "we own 2 cars"
            {
                "markers": ["i have ", "i own ", "we have ", "we own ",
                            "i've got ", "we've got "],
                "type": "observation",
                "base_confidence": 0.55,
                "is_factual": True,
                "temporal_category": "count",
            },
            # EXTRACT-C5: Possession/acquisition facts → observation (is_factual, episodic)
            # Catches "bought their EP", "got my vinyl signed", "picked up a guitar"
            {
                "markers": ["i bought ", "i purchased ", "i picked up ",
                            "i ordered ", "i downloaded ", "i subscribed ",
                            "got my ", "got a ", "got the "],
                "type": "observation",
                "base_confidence": 0.55,
                "is_factual": True,
                "temporal_category": "episodic",
            },
        ]

        text_lower = text.lower()
        sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]

        # Scan sentences for pattern matches
        for sentence in sentences:
            if len(sentence) < 30:
                continue
            sent_lower = sentence.lower()

            for pattern in conversation_patterns:
                matches = sum(1 for m in pattern["markers"] if m in sent_lower)
                if matches >= 1:
                    # Build summary: use the matching sentence, cap at 300 chars
                    summary = sentence[:300].strip()
                    if len(sentence) > 300:
                        summary += "..."

                    confidence = min(0.70, pattern["base_confidence"] + (matches - 1) * 0.10)

                    insight_dict = {
                        "summary": summary,
                        "confidence": confidence,
                        "type": pattern["type"],
                        "signals": [pattern["type"]],
                    }
                    # INGEST-015: Forward factual metadata from pattern definition
                    if pattern.get("is_factual"):
                        insight_dict["is_factual"] = True
                        insight_dict["temporal_category"] = pattern.get("temporal_category")
                    # INGEST-022: Enrich summary with temporal anchors from source
                    insight_dict["summary"] = self._enrich_temporal_anchors(
                        insight_dict["summary"], sentence
                    )
                    insights.append(insight_dict)
                    break  # One pattern per sentence

        # Also try ConversationProcessor heuristic for additional extraction
        heuristic = processor.extract_insight_heuristic(text)
        if heuristic and heuristic.get("summary"):
            # Check it's not a near-duplicate of already extracted insights
            is_dup = any(self._text_overlap(heuristic["summary"], i["summary"]) > 0.7 for i in insights)
            if not is_dup:
                insights.append(heuristic)

        # INGEST-020: Role-aware factual scan — assistant_text only.
        # Declarative facts from the assistant (thresholds, defaults, config values)
        # are more reliably extracted from the assistant turn alone, without
        # user-turn noise. Capped at 3 to avoid flooding with low-confidence observations.
        if assistant_text and len(assistant_text) >= 30:
            FACTUAL_MARKERS = [
                "the default is", "the value is", "the threshold is",
                "is set to", "is stored in", "is defined in",
                "must be ", "cannot be ", "requires ",
                "returns ", "raises ", "expects ",
            ]
            # INGEST-021: Split on newlines first (captures list items, bullet points),
            # then split each line on periods (captures multi-sentence lines).
            # Strip list prefixes BEFORE splitting so bullets/numbers don't leak through.
            _LIST_PREFIX_RE = _re.compile(r'^\s*(?:[-*•]|\d+[.)]) *')
            raw_lines = assistant_text.replace("!", ".").replace("?", ".").split("\n")
            asst_sentences = []
            for line in raw_lines:
                # Strip list prefix first (bullets, numbers)
                clean_line = _LIST_PREFIX_RE.sub('', line).strip()
                if not clean_line or len(clean_line) < 30:
                    continue
                # Split on periods for multi-sentence lines
                if "." in clean_line:
                    parts = [p.strip() for p in clean_line.split(".") if p.strip() and len(p.strip()) >= 30]
                    if parts:
                        asst_sentences.extend(parts)
                    else:
                        # Periods present but splits too short — use whole line
                        asst_sentences.append(clean_line)
                else:
                    # No periods — use whole cleaned line
                    asst_sentences.append(clean_line)
            factual_count = 0
            for sentence in asst_sentences:
                if factual_count >= 3:
                    break
                sent_lower = sentence.lower()
                if any(m in sent_lower for m in FACTUAL_MARKERS):
                    is_dup = any(self._text_overlap(sentence, i["summary"]) > 0.6 for i in insights)
                    if not is_dup:
                        # INGEST-022: Enrich factual scan with temporal anchors
                        enriched_summary = self._enrich_temporal_anchors(
                            sentence[:300], sentence
                        )
                        insights.append({
                            "summary": enriched_summary,
                            "confidence": 0.55,  # DATA-068: raised from 0.45 to survive conservation zone (floor=0.50)
                            "type": "observation",
                            "signals": ["factual_marker"],
                            "extraction_source": "factual_scan",
                        })
                        factual_count += 1

        # Cap at 7 insights per call (INGEST-020: raised from 5)
        return insights[:7]

    def _text_overlap(self, a: str, b: str) -> float:
        """Simple word overlap ratio between two strings."""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        return len(intersection) / min(len(words_a), len(words_b))

    # --- S3.5: Contradiction detection ---
    # LIFECYCLE-001: STATUS_TRANSITIONS moved to app/constants.py for shared access
    # Used by write-time detection (here). Serve-time Phase 1b removed per LIFECYCLE-001 cleanup.
    # SUPER-009: Self-correction detection handled by Layer 1.5 in _detect_contradiction
    # (NOT as a STATUS_TRANSITIONS entry — "never"/"actually" are too common as standalone
    # markers, causing false positives on unrelated sentences. Layer 1.5 has 2-word topic
    # overlap guard that prevents cross-topic matches.)

    def _detect_contradiction(self, existing_summary: str, new_summary: str) -> str | None:
        """S3.5: Detect if new insight contradicts/supersedes an existing concept.

        Uses STATUS_TRANSITIONS marker pairs + opposing assertion detection.
        Returns reason string if contradiction detected, None otherwise.

        Bug 4 fix: Also detects opposing claims where two concepts discuss the
        same topic but assert mutually exclusive specifics (e.g., "uses Python"
        vs "uses Go"). Uses the contradiction engine's negation detection.
        """
        old_lower = existing_summary.lower()
        new_lower = new_summary.lower()

        # MEASURE-018: Layer-level instrumentation for contradiction detection analysis
        _layers_checked: list[str] = []

        # Layer 1: STATUS_TRANSITIONS marker pairs (original behavior)
        # LIFECYCLE-001: Imported from constants.py (lazy import to avoid circular deps)
        from app.constants import STATUS_TRANSITIONS

        _layers_checked.append("L1_status_transitions")
        for before_markers, after_markers, reason in STATUS_TRANSITIONS:
            old_matches = any(m in old_lower for m in before_markers)
            new_matches = any(m in new_lower for m in after_markers)
            if old_matches and new_matches:
                logger.debug(f"MEASURE-018: Contradiction detected at L1 — {reason}")
                return reason

        # Layer 1.5 (SELFCORR-001): Detect same-entity opposing conclusions
        # Catches: "X never fired" vs "X fired 110 times" where STATUS_TRANSITIONS
        # markers don't match because summaries use analytical language.
        negation_markers = {"never", "zero", "no ", "not ", "none", "0 ", "doesn't", "cannot", "impossible"}
        affirmation_markers = {"actually", "confirmed", "verified", "found that", "turns out", "in fact", "does exist"}

        old_has_negation = any(m in old_lower for m in negation_markers)
        new_has_affirmation = any(m in new_lower for m in affirmation_markers)

        _layers_checked.append("L1.5_self_correction")
        if old_has_negation and new_has_affirmation:
            old_words = set(_re.findall(r"\b\w{4,}\b", old_lower))
            new_words = set(_re.findall(r"\b\w{4,}\b", new_lower))
            overlap = old_words & new_words
            if len(overlap) >= 2:
                logger.debug(f"MEASURE-018: Contradiction detected at L1.5 — self-correction")
                return f"Self-correction: negation in old + affirmation in new (shared terms: {', '.join(list(overlap)[:5])})"

        # Layer 2 (Bug 4 fix): Opposing assertion detection via contradiction engine
        # Only runs if summaries have meaningful overlap (same topic area) but also
        # have clear differences (not just paraphrases)
        _layers_checked.append("L2_opposing_assertions")
        try:
            from app.contradiction import ScoredConcept, detect_retrieval_contradictions

            existing_sc = ScoredConcept(
                concept_id="__existing__",
                summary=existing_summary,
                knowledge_area="dedup_check",
                authority_score=0.5,
                currency_score=1.0,
            )
            new_sc = ScoredConcept(
                concept_id="__new__",
                summary=new_summary,
                knowledge_area="dedup_check",
                authority_score=0.5,
                currency_score=1.0,
            )
            result = detect_retrieval_contradictions([existing_sc, new_sc])
            if result.pairs:
                pair = result.pairs[0]
                logger.debug(f"MEASURE-018: Contradiction detected at L2 — {pair.contradiction_type.value}")
                return f"Opposing assertions detected: {pair.contradiction_type.value} ({pair.reason or 'semantic conflict'})"
        except Exception as e:
            logger.debug(f"S3.5: Opposing assertion check failed (non-fatal): {e}")

        # Layer 3 (Phase 3 v1.1): LLM Tier 2 for ambiguous cases
        _layers_checked.append("L3_llm_tier2")
        try:
            from app.config import get_feature_flag

            if get_feature_flag("LLM_CONTRADICTION_TIER2_ENABLED"):
                from app.contradiction import _compute_keyword_overlap_score
                from app.contradiction_llm import detect_contradiction_llm_sync, is_tier2_candidate

                tier1_score = _compute_keyword_overlap_score(existing_summary, new_summary)
                if is_tier2_candidate(tier1_score):
                    llm_result = detect_contradiction_llm_sync(
                        new_summary,
                        existing_summary,
                        session_id=getattr(self, "_current_session_id", ""),
                    )
                    if llm_result.score > 0.7:
                        logger.debug(f"MEASURE-018: Contradiction detected at L3 — LLM Tier 2")
                        return f"LLM Tier 2: {llm_result.reason[:200]}"
        except Exception as e:
            logger.debug(f"S3.5: LLM Tier 2 check failed (non-fatal): {e}")

        # MEASURE-018: No contradiction detected across any layer
        logger.debug(f"MEASURE-018: No contradiction detected (layers checked: {_layers_checked})")
        return None

    def _supersede_concept(self, old_concept_id: str, new_concept_id: str, reason: str) -> bool:
        """S3.5: Execute supersession via unified path (SUPER-012).

        Delegates to execute_supersession() which handles ALL supersession behaviors:
        currency_status, superseded_by, confidence reduction, anti-terms, edge creation,
        association transfer, evidence addition, governance event.

        Replaces the previous independent implementation that was inconsistent with
        the supersession.py path (see SUPERSESSION_COMPOUND_GAUNTLET.md, finding G-01).
        """
        try:
            from app.storage import db_immediate
            from app.supersession import execute_supersession

            with db_immediate() as conn:
                result = execute_supersession(
                    old_concept_id=old_concept_id,
                    new_concept_id=new_concept_id,
                    reason=reason,
                    conn=conn,
                )
            return result.superseded

        except Exception as e:
            logger.error(
                "SUPER-012: Unified supersession failed %s→%s: %s",
                old_concept_id,
                new_concept_id,
                e,
            )
            return False

    def _process_single_insight(
        self,
        insight: dict,
        request: SessionLearnRequest,
        retrieval_engine,
        extraction_source: str = "heuristic",
        evolved_this_call: set = None,
        budget_remaining: int = 50,
        precomputed_dedup: list[dict] | None = None,  # PERF-021: batch pre-computed dedup results
    ) -> dict:
        """Process a single extracted insight through dedup, creation, and association.

        Returns dict with action taken and result details.
        Implements S1 (source tagging), S2 (self-corroboration), S3 (per-call cap),
        S5 (HHI confidence cap).
        """
        if evolved_this_call is None:
            evolved_this_call = set()

        summary = insight["summary"]
        confidence = insight.get("confidence", 0.40)

        # --- Quality gate: confidence floor (PRICING-006: budget-aware) ---
        _budget_zone_value = "unknown"
        try:
            from app.config import BUDGET_ZONE_THRESHOLDS
            from app.pricing import conversation_meter

            budget_zone = conversation_meter.get_budget_zone()
            _budget_zone_value = budget_zone.value
            zone_thresholds = BUDGET_ZONE_THRESHOLDS.get(
                budget_zone.value,
                BUDGET_ZONE_THRESHOLDS["normal"],  # Fallback to default
            )
            client_floor = zone_thresholds["client"]
            heuristic_floor = zone_thresholds["heuristic"]
        except Exception:
            # Fallback to Sprint B defaults if pricing module unavailable
            client_floor = 0.35
            heuristic_floor = 0.45

        # INGEST-001 + PRICING-006: Budget-aware confidence floors
        if extraction_source == "heuristic" and confidence < heuristic_floor:
            return {"action": "skipped_confidence_heuristic", "budget_zone": _budget_zone_value}
        elif confidence < client_floor:
            return {"action": "skipped_confidence", "budget_zone": _budget_zone_value}

        # INGEST-001: Minimum summary quality
        # MEASURE-026 §18: Benchmark facts are complete declarative sentences (5-7 words)
        # that are valid despite being short. Client-extracted concepts with evidence
        # use a lower floor (4 words) since they've already been validated upstream.
        summary_words = len(summary.split())
        _min_words = 4 if (extraction_source == "client" and insight.get("evidence")) else 8
        if summary_words < _min_words:
            return {"action": "skipped_short_summary"}

        # INGEST-001: Evidence requirement for client-extracted concepts
        evidence = insight.get("evidence", [])
        if extraction_source == "client" and not evidence:
            return {"action": "skipped_no_evidence"}

        # --- Deduplication via cosine similarity ---
        # MATURITY-003: Use embedding search when available (handles paraphrases).
        # TF-IDF dead zone: 99.1% of scores fall below 0.50, preventing evolution.
        # Embedding search produces continuous distribution in the EVOLVE zone.
        from app.config import (
            EMBEDDING_EVOLVE_THRESHOLD,
            EMBEDDING_SKIP_THRESHOLD,
            FEATURE_FLAGS,
        )

        _use_embedding = FEATURE_FLAGS.get("EMBEDDING_DEDUP_ENABLED", False)
        if _use_embedding:
            dedup_results = retrieval_engine.search_for_dedup_embedding(summary, top_k=3)
            _skip_threshold = EMBEDDING_SKIP_THRESHOLD  # 0.85 (calibrated)
            _evolve_threshold = EMBEDDING_EVOLVE_THRESHOLD  # 0.55 (calibrated)
        elif precomputed_dedup is not None:
            # PERF-021: use batch pre-computed results (avoids redundant DB round-trip)
            dedup_results = precomputed_dedup
            _skip_threshold = float(os.environ.get("PITH_TFIDF_SKIP_THRESHOLD", "0.85"))
            _evolve_threshold = float(os.environ.get("PITH_TFIDF_EVOLVE_THRESHOLD", "0.50"))
        else:
            dedup_results = retrieval_engine.search_for_dedup_tfidf(summary, top_k=3)
            _skip_threshold = float(os.environ.get("PITH_TFIDF_SKIP_THRESHOLD", "0.85"))
            _evolve_threshold = float(os.environ.get("PITH_TFIDF_EVOLVE_THRESHOLD", "0.50"))

        top_cosine = dedup_results[0]["cosine_score"] if dedup_results else 0.0
        top_match = dedup_results[0] if dedup_results else None

        # RETRIEVAL-021: Activation bias — lower evolve threshold for activated concepts
        from app.config import ACTIVATION_EVOLVE_BIAS
        _activated_ids = getattr(request, "activated_concept_ids", None) or []
        _activation_bias = 0.0
        if top_match and top_match.get("concept_id") in _activated_ids:
            _activation_bias = min(ACTIVATION_EVOLVE_BIAS, 0.20)  # Clamp to prevent aggressive merging
            logger.info(
                f"RETRIEVAL-021: activation bias for {top_match['concept_id']} "
                f"(cosine={top_cosine:.3f}, effective_threshold={_evolve_threshold - _activation_bias:.2f})"
            )
        _effective_evolve_threshold = _evolve_threshold - _activation_bias

        # INGEST-007: Cross-KA merge guard
        from app.config import CROSS_KA_EVOLVE_THRESHOLD, ka_groups_match
        _incoming_ka = insight.get("knowledge_area", "general") or "general"
        _match_ka = top_match.get("knowledge_area", "") if top_match else ""
        _ka_match = ka_groups_match(_incoming_ka, _match_ka)
        if not _ka_match:
            _effective_evolve_threshold = max(_effective_evolve_threshold, CROSS_KA_EVOLVE_THRESHOLD)

        # BENCHMARK-001: In benchmark mode, force all concepts to CREATE zone —
        # bypass dedup entirely. FactConsolidation facts share template structure
        # ("X is famous for Y") so TF-IDF similarity can exceed 0.85 skip threshold
        # even when subject/object differ. Pre-deduplicated by pith_agent conflict
        # resolution (highest serial wins), so dedup here is harmful not helpful.
        # BENCH-INFRA-002: Dedup bypass via BenchmarkIngestionMode config.
        # BENCHMARK-001b: Separate control for dedup bypass vs GarbageDetector bypass.
        from app.config import BENCHMARK as _bm_dedup
        _dedup_bypass = _bm_dedup.skip_dedup

        # DATA-055: Classify dedup zone for structured logging
        if _dedup_bypass:
            _dedup_zone = "CREATE"
        elif top_cosine >= _skip_threshold:
            _dedup_zone = "SKIP"
        elif top_cosine >= _effective_evolve_threshold and top_match:
            _dedup_zone = "EVOLVE"
        else:
            _dedup_zone = "CREATE"

        # INGEST-007: Log KA guard decision
        if not _ka_match and top_cosine >= EMBEDDING_EVOLVE_THRESHOLD:
            logger.info(
                f"INGEST-007: Cross-KA guard activated — "
                f"incoming={_incoming_ka} match={_match_ka} "
                f"cosine={top_cosine:.4f} effective_thresh={_effective_evolve_threshold:.2f} "
                f"zone={_dedup_zone}"
            )
            # MONITOR-056: Track cross-KA guard activation rate
            from app.metrics import metrics as _ka_metrics
            _ka_metrics.record("cross_ka_guard_activations", 1)

        # DATA-055: Structured dedup outcome log
        _dedup_method = "embedding" if _use_embedding else "tfidf"
        _match_id = top_match["concept_id"] if top_match else None
        logger.info(
            f"DEDUP_DECISION: zone={_dedup_zone} cosine={top_cosine:.4f} "
            f"match={_match_id} method={_dedup_method} "
            f"skip_thresh={_skip_threshold} evolve_thresh={_effective_evolve_threshold:.2f} "
            f"activation_bias={_activation_bias:.2f} "
            f"same_call_dups={len(evolved_this_call) if evolved_this_call else 0} "
            f"summary_hash={hashlib.sha256(summary.encode()).hexdigest()[:12]}"
        )

        # MEASURE-026 §15: SKIP-zone divergence detection.
        # When cosine >= skip_threshold, check if incoming and existing concepts
        # share most tokens (same subject) but differ on content words (different value).
        # If so, override SKIP→EVOLVE to handle temporal supersession at the SKIP boundary.
        if _dedup_zone == "SKIP" and top_match and _use_embedding:
            try:
                _div_existing = load_concept(top_match["concept_id"], track_access=False)
                if _div_existing and _div_existing.summary:
                    _div_incoming_tok = set(summary.lower().split())
                    _div_existing_tok = set(_div_existing.summary.lower().split())
                    _div_intersection = _div_incoming_tok & _div_existing_tok
                    _div_union = _div_incoming_tok | _div_existing_tok
                    _div_jaccard = len(_div_intersection) / len(_div_union) if _div_union else 1.0
                    _div_stopwords = {
                        'the', 'a', 'an', 'is', 'was', 'are', 'were', 'be', 'been',
                        'of', 'in', 'to', 'for', 'by', 'and', 'or', 'with', 'that',
                        'this', 'it', 'its', 'has', 'had', 'have', 'not', 'on', 'at',
                    }
                    _div_only_incoming = _div_incoming_tok - _div_existing_tok - _div_stopwords
                    _div_only_existing = _div_existing_tok - _div_incoming_tok - _div_stopwords
                    _div_content = _div_only_incoming | _div_only_existing
                    _div_threshold = float(os.environ.get("PITH_SKIP_DIVERGENCE_THRESHOLD", "0.6"))
                    if _div_jaccard >= _div_threshold and len(_div_content) >= 1:
                        _dedup_zone = "EVOLVE"
                        logger.info(
                            f"SKIP-OVERRIDE: divergence detected — cosine={top_cosine:.4f} "
                            f"jaccard={_div_jaccard:.3f} divergent_tokens={_div_content} "
                            f"reclassified SKIP→EVOLVE for match={top_match['concept_id']}"
                        )
            except Exception as _div_err:
                logger.warning(f"SKIP-OVERRIDE: divergence check failed: {_div_err}")

        # MEASURE-026 §18: Entity-aware dedup guard.
        # Template-similar facts about DIFFERENT subjects (e.g. "X is married to A"
        # vs "Y is married to B") produce high cosine due to shared template tokens.
        # Extract leading subject entity from both concepts; if subjects differ,
        # override EVOLVE/SKIP → CREATE to prevent cross-entity merging.
        if _dedup_zone in ("EVOLVE", "SKIP") and top_match:
            try:
                _eg_existing = (
                    _div_existing if '_div_existing' in dir() and _div_existing
                    else load_concept(top_match["concept_id"], track_access=False)
                )
                if _eg_existing and _eg_existing.summary:
                    def _extract_subject(text: str) -> str:
                        """Extract leading proper-noun span as subject entity.
                        Returns lowercased subject for comparison.
                        Handles: 'Thomas Kyd was born in...' → 'thomas kyd'
                                 'The capital of France is...' → 'the capital of france'
                                 'Windows Phone was developed by...' → 'windows phone'
                        """
                        _copulas = {
                            'is', 'was', 'are', 'were', 'has', 'had',
                            'plays', 'speaks', 'works', 'worked', 'died',
                        }
                        words = text.split()
                        subject_words = []
                        for w in words:
                            # Stop at copula/verb boundary
                            if w.lower().rstrip('.,;:') in _copulas:
                                break
                            subject_words.append(w)
                        # Fallback: if no copula found, take first 3 words
                        if not subject_words:
                            subject_words = words[:3]
                        return " ".join(subject_words).lower().strip('.,;:')

                    _eg_incoming_subj = _extract_subject(summary)
                    _eg_existing_subj = _extract_subject(_eg_existing.summary)

                    if _eg_incoming_subj and _eg_existing_subj and _eg_incoming_subj != _eg_existing_subj:
                        _eg_original_zone = _dedup_zone
                        _dedup_zone = "CREATE"
                        logger.info(
                            f"ENTITY-GUARD: subject mismatch — "
                            f"incoming='{_eg_incoming_subj}' existing='{_eg_existing_subj}' "
                            f"cosine={top_cosine:.4f} overriding {_eg_original_zone}→CREATE "
                            f"for match={top_match['concept_id']}"
                        )
            except Exception as _eg_err:
                logger.warning(f"ENTITY-GUARD: check failed: {_eg_err}")

        # PRODUCT-002: Factual-into-abstract guard.
        # Episodic facts (numbers, dates, amounts) must not be absorbed into
        # pattern/principle/heuristic concepts. Force CREATE to preserve specifics.
        if _dedup_zone == "EVOLVE" and top_match:
            try:
                from app.config import get_feature_flag
                if get_feature_flag("EPISODIC_GRANULARITY_GUARD_ENABLED", False):
                    from app.fact_classifier import classify_concept as _fc_classify
                    _incoming_cls = _fc_classify(
                        summary=summary,
                        concept_type=insight.get("type", "observation"),
                        knowledge_area=insight.get("knowledge_area", "general") or "general",
                    )
                    _incoming_factual = _incoming_cls.get("is_factual", False)
                    _incoming_score = _incoming_cls.get("factual_score", 0)

                    if _incoming_factual and _incoming_score >= 2.0:
                        _eg_for_p002 = (
                            _div_existing if '_div_existing' in dir() and _div_existing
                            else load_concept(top_match["concept_id"], track_access=False)
                        )
                        if _eg_for_p002:
                            _existing_type = _eg_for_p002.concept_type or "observation"
                            _ABSTRACT_TYPES = {"pattern", "principle", "method",
                                               "heuristic", "cognitive_strategy"}
                            if _existing_type in _ABSTRACT_TYPES:
                                _dedup_zone = "CREATE"
                                logger.info(
                                    f"PRODUCT-002: Factual-into-abstract guard — "
                                    f"incoming factual (score={_incoming_score:.1f}) "
                                    f"vs existing {_existing_type} "
                                    f"cosine={top_cosine:.4f} overriding EVOLVE→CREATE "
                                    f"for match={top_match['concept_id']}"
                                )
                                try:
                                    from app.metrics import metrics as _p002_metrics
                                    _p002_metrics.record("product002_guard_activations", 1)
                                except Exception:
                                    pass
            except Exception as _p002_err:
                logger.debug(f"PRODUCT-002: Guard check failed (non-fatal): {_p002_err}")

        # --- STALE-003: Metric-conflict write-time check ---
        # If incoming concept contains a recognizable metric pattern (X%, N/M)
        # and the nearest match contains a DIFFERENT value for the same metric,
        # force EVOLVE to update rather than creating a contradictory duplicate.
        if _dedup_zone == "CREATE" and top_match and top_cosine >= 0.35:
            _stale_metric_override = os.environ.get("PITH_METRIC_CONFLICT_CHECK", "true").lower() == "true"
            if _stale_metric_override:
                import re as _mc_re
                _METRIC_RE = _mc_re.compile(
                    r'(\d+\.?\d*)\s*%'       # percentage: "73.2%"
                    r'|(\d+)\s*/\s*(\d+)'     # fraction: "60/71"
                    r'|(\d+\.?\d*)\s*pp'      # percentage points: "+4.2pp"
                )
                _incoming_metrics = set(_METRIC_RE.findall(summary))
                if _incoming_metrics and top_match:
                    _existing_summary = top_match.get("summary", "")
                    _existing_metrics = set(_METRIC_RE.findall(_existing_summary))
                    if _incoming_metrics and _existing_metrics and _incoming_metrics != _existing_metrics:
                        _dedup_zone = "EVOLVE"
                        logger.info(
                            f"STALE-003: metric-conflict override CREATE→EVOLVE — "
                            f"incoming_metrics={_incoming_metrics} "
                            f"existing_metrics={_existing_metrics} "
                            f"match={top_match.get('concept_id', 'unknown')}"
                        )

        # Three-zone dedup logic (thresholds adapt to search method)
        if _dedup_zone == "SKIP":
            return {"action": "skipped_duplicate", "dedup_zone": "SKIP",
                    "cosine": round(top_cosine, 4), "match_id": _match_id, "method": _dedup_method}

        if not _dedup_bypass and _dedup_zone == "EVOLVE" and _effective_evolve_threshold <= top_cosine < _skip_threshold and top_match:
            # S3.5: Contradiction detection — check if new insight supersedes old
            existing_concept = load_concept(top_match["concept_id"], track_access=False)
            if existing_concept:
                contradiction_reason = self._detect_contradiction(existing_concept.summary, summary)
                # MEASURE-018: Instrument contradiction detection for false-negative analysis
                try:
                    import json as _m18_json
                    from datetime import datetime as _m18_dt

                    from app.storage import db_immediate

                    with db_immediate() as _m18_conn:
                        _m18_conn.execute(
                            """INSERT INTO governance_events
                               (session_id, event_type, concept_id, details, created_at)
                               VALUES (?, ?, ?, ?, ?)""",
                            (
                                self.current_session.session_id if self.current_session else None,
                                "contradiction_check",
                                existing_concept.id,
                                _m18_json.dumps({
                                    "detected": contradiction_reason is not None,
                                    "reason": contradiction_reason[:200] if contradiction_reason else None,
                                    "cosine": round(top_cosine, 4),
                                    "existing_summary": existing_concept.summary[:100],
                                    "new_summary": summary[:100],
                                    "dedup_zone": "EVOLVE",
                                }),
                                _m18_dt.now(UTC).isoformat(),
                            ),
                        )
                except Exception:
                    pass  # Instrumentation is best-effort
                if contradiction_reason:
                    # P3-2: Log contradiction detection governance event
                    try:
                        import json as _json
                        from datetime import datetime as _dt

                        from app.storage import db_immediate

                        with db_immediate() as _gov_conn:
                            _gov_conn.execute(
                                """INSERT INTO governance_events
                                   (event_type, concept_id, details, created_at)
                                   VALUES (?, ?, ?, ?)""",
                                (
                                    GOV_EVENT_CONTRADICTION_REVIEW,
                                    top_match["concept_id"],
                                    _json.dumps(
                                        {
                                            "new_insight_summary": summary[:200],
                                            "contradiction_reason": contradiction_reason,
                                            "session_id": getattr(self, "_current_session_id", ""),
                                        }
                                    ),
                                    _dt.now(UTC).isoformat(),
                                ),
                            )
                    except Exception:
                        pass

                    # This is a supersession, not an evolution — create new, mark old
                    logger.info(
                        f"S3.5: Contradiction detected: '{top_match['concept_id']}' "
                        f"→ new insight. Reason: {contradiction_reason}"
                    )
                    # Create the new concept (treat as novel)
                    # Bug 6 fix: skip_write_contradiction=True prevents double-catch —
                    # we already know there's a contradiction (that's why we're superseding),
                    # so the write-time check would either HARD_REJECT or error on the
                    # same conflict we already detected.
                    search_results = retrieval_engine.search_lightweight(
                        summary,
                        top_k=3,
                        min_confidence=0.0,
                    )
                    result = self._create_new_concept(
                        insight,
                        request,
                        retrieval_engine,
                        search_results,
                        extraction_source=extraction_source,
                        skip_write_contradiction=True,
                    )
                    # If creation succeeded, supersede the old concept
                    if result.get("action") == "created":
                        new_id = result["learned_concept"].concept_id
                        self._supersede_concept(top_match["concept_id"], new_id, contradiction_reason)
                        result["superseded"] = {
                            "old_id": top_match["concept_id"],
                            "new_id": new_id,
                            "reason": contradiction_reason,
                        }
                    return result

            # EXPLICIT_SUPERSESSION_SPEC v1.1 Amendment A1:
            # If LLM explicitly declared supersession of the dedup match,
            # override evolution — create new concept + supersede old.
            explicit_supersedes = insight.get("supersedes") or []
            if top_match["concept_id"] in explicit_supersedes:
                logger.info(
                    f"EXPLICIT_SUPERSESSION: overriding dedup evolution for "
                    f"'{top_match['concept_id']}' (cosine={top_cosine:.2f})"
                )
                search_results = retrieval_engine.search_lightweight(summary, top_k=3, min_confidence=0.0)
                result = self._create_new_concept(
                    insight,
                    request,
                    retrieval_engine,
                    search_results,
                    extraction_source=extraction_source,
                    skip_write_contradiction=True,
                )
                if result.get("action") == "created":
                    new_id = result["learned_concept"].concept_id
                    self._supersede_concept(
                        top_match["concept_id"], new_id, "Explicit supersession declared at extraction time"
                    )
                    result["superseded"] = {
                        "old_id": top_match["concept_id"],
                        "new_id": new_id,
                        "reason": "explicit_supersession_override_dedup",
                    }
                return result

            # S3: Per-call evidence cap — max 1 evolution per concept per call
            concept_id = top_match["concept_id"]

            # BENCH-EVOLVE-001: Gate evolve on PITH_DISABLE_EVOLVE.
            # When set (benchmark ingestion), skip evolve and fall through to
            # novel creation — lets CF facts create separate concepts instead of
            # being absorbed into their real-world predecessors.
            import os as _evo_os
            _evolve_disabled = _evo_os.environ.get("PITH_DISABLE_EVOLVE", "").lower() in ("true", "1")

            if _evolve_disabled:
                logger.info(
                    f"BENCH-EVOLVE-001: Evolve disabled (PITH_DISABLE_EVOLVE=true), "
                    f"skipping dedup evolve for cosine={top_cosine:.4f} match={concept_id}"
                )
                # Fall through past this block to S4 budget check → novel creation
            else:
                if concept_id in evolved_this_call:
                    logger.info(
                        f"DEDUP_DECISION: zone=EVOLVE_CAPPED cosine={top_cosine:.4f} "
                        f"match={concept_id} method={_dedup_method} reason=per_call_cap"
                    )
                    return {"action": "skipped_per_call_cap", "dedup_zone": "EVOLVE_CAPPED",
                            "cosine": round(top_cosine, 4), "match_id": concept_id, "method": _dedup_method}
                evolved_this_call.add(concept_id)
                _evolve_result = self._evolve_existing_from_dedup(top_match, insight, request, extraction_source=extraction_source)
                _evolve_result["dedup_zone"] = "EVOLVE"
                _evolve_result["cosine"] = round(top_cosine, 4)
                _evolve_result["match_id"] = concept_id
                _evolve_result["method"] = _dedup_method
                return _evolve_result

        # S4: Budget check for new concept creation
        if budget_remaining <= 0:
            logger.info("session_learn: S4 daily budget exhausted, skipping creation")
            return {"action": "skipped_budget"}

        # Novel: CREATE new concept (cosine < 0.50)
        search_results = retrieval_engine.search_lightweight(
            summary,
            top_k=3,
            min_confidence=0.0,
        )
        result = self._create_new_concept(
            insight, request, retrieval_engine, search_results, extraction_source=extraction_source
        )

        # --- Trigger 1: Staleness check on embedding neighbors ---
        # The dedup above used TF-IDF (cosine < 0.50), but embedding search
        # can find same-topic concepts that TF-IDF missed (empirically validated:
        # TF-IDF gives 0.04 where embedding gives 0.42 on status-transition pairs).
        if result.get("action") == "created" and result.get("learned_concept"):
            try:
                from app.staleness import check_for_stale_relatives

                new_id = result["learned_concept"].concept_id
                staleness_result = check_for_stale_relatives(
                    new_concept_id=new_id,
                    new_summary=summary,
                    retrieval_engine=retrieval_engine,
                    supersede_fn=self._supersede_concept,
                )
                if staleness_result.concepts_superseded > 0:
                    result["staleness_t1"] = {
                        "superseded": staleness_result.concepts_superseded,
                        "details": staleness_result.details,
                        "time_ms": staleness_result.time_ms,
                    }
            except Exception as e:
                logger.warning(f"Staleness T1 check failed (non-fatal): {e}")

        # --- RETRIEVAL-020b: Evolution supersession check on novel concepts ---
        # Fires after concept creation to detect type-progression pairs
        # (e.g., observation → principle) in the EVOLUTION ZONE (cosine 0.50-0.82).
        if result.get("action") == "created" and result.get("learned_concept"):
            try:
                from app.evolution import check_evolution_supersession
                from app.config import EVOLUTION_CANARY_MODE

                lc = result["learned_concept"]
                evo_result = check_evolution_supersession(
                    new_concept_id=lc.concept_id,
                    new_concept_type=getattr(lc, "concept_type", "observation"),
                    new_authority=getattr(lc, "authority_score", None) or 0.5,
                    retrieval_engine=retrieval_engine,
                    concept_loader=load_concept,
                    supersede_fn=self._supersede_concept,
                    canary_mode=EVOLUTION_CANARY_MODE,
                )
                if evo_result.pair_detected:
                    result["evolution_t3"] = {
                        "pair_detected": evo_result.pair_detected,
                        "older_concept_id": evo_result.older_concept_id,
                        "newer_concept_id": evo_result.newer_concept_id,
                        "composite_score": evo_result.composite_score,
                        "type_progression": evo_result.type_progression,
                        "action_taken": evo_result.action_taken,
                        "time_ms": evo_result.time_ms,
                        "canary_mode": EVOLUTION_CANARY_MODE,
                    }
                    logger.info(
                        f"RETRIEVAL-020b: Evolution pair detected — "
                        f"{evo_result.type_progression} "
                        f"(composite={evo_result.composite_score:.3f}, "
                        f"action={evo_result.action_taken}, "
                        f"canary={EVOLUTION_CANARY_MODE})"
                    )
            except Exception as evo_err:
                logger.warning(f"RETRIEVAL-020b: Evolution check failed (non-fatal): {evo_err}")

        # --- EXPLICIT_SUPERSESSION_SPEC v1.1: Declared supersession from extraction ---
        if result.get("action") == "created" and result.get("learned_concept"):
            supersede_ids = insight.get("supersedes")
            if supersede_ids:
                new_id = result["learned_concept"].concept_id
                explicit_count = 0
                for old_id in supersede_ids[:5]:  # Hard cap per concept
                    if old_id == new_id:  # A2: self-referential guard
                        logger.warning(f"EXPLICIT_SUPERSESSION: skipping self-reference {old_id}")
                        continue
                    success = self._supersede_concept(
                        old_id, new_id, "Explicit supersession declared at extraction time"
                    )
                    if success:
                        explicit_count += 1
                        logger.info(f"EXPLICIT_SUPERSESSION: '{old_id}' → '{new_id}'")
                if explicit_count > 0:
                    result["explicit_supersessions"] = explicit_count

        # L3: Log when supersedes declared but concept not created
        if insight.get("supersedes") and result.get("action") != "created":
            logger.warning(
                f"EXPLICIT_SUPERSESSION: supersedes declared but concept not created "
                f"(action={result.get('action')}). Targets: {insight['supersedes']}"
            )

        return result

    def _maybe_promote_maturity(self, concept_id: str) -> None:
        """W7: Check if a concept qualifies for maturity promotion.

        Promotion rules:
          QUARANTINED + evidence_count >= QUARANTINE_PROMOTION_MIN_EVIDENCE → PROVISIONAL
          PROVISIONAL + evidence_count >= 1 + access_count >= 5 → ESTABLISHED
          PROVISIONAL + reinforcement >= 8 → ESTABLISHED

        Guards (ARCH-D05):
          - Superseded concepts are never promoted
          - Concepts with confidence < 0.25 are not promoted to ESTABLISHED
        """
        # BENCH-INFRA-008: Skip maturity promotion in benchmark readonly mode.
        # Prevents PROVISIONAL→ESTABLISHED cascades that cause ±4% EM noise.
        import os as _os
        if _os.environ.get("PITH_BENCHMARK_READONLY", "").lower() in ("true", "1"):
            return

        from app.config import (
            FEATURE_FLAGS,
            PROVISIONAL_PROMOTION_MIN_ACCESS,
            PROVISIONAL_PROMOTION_MIN_EVIDENCE,
            QUARANTINE_PROMOTION_MIN_EVIDENCE,
            REINFORCEMENT_PROMOTION_THRESHOLD,
        )

        if not FEATURE_FLAGS.get("INGESTION_VALIDATION_ENABLED", False):
            return

        concept = load_concept(concept_id, track_access=False)
        if not concept:
            return

        # ARCH-D05: Guard against promoting superseded concepts
        if getattr(concept, "superseded_by", None):
            return

        maturity = getattr(concept, "maturity", "ESTABLISHED")
        evidence_count = len(concept.evidence) if concept.evidence else 0
        access_count = getattr(concept, "access_count", 0)
        reinforcement = getattr(concept, "reinforcement_count", 0)
        confidence = getattr(concept, "confidence", 0.0)
        new_maturity = maturity

        # MAINT-003: Use config constant instead of hardcoded 3
        if maturity == "QUARANTINED" and evidence_count >= QUARANTINE_PROMOTION_MIN_EVIDENCE:
            new_maturity = "PROVISIONAL"
        elif (
            maturity == "PROVISIONAL"
            and confidence >= 0.25  # ARCH-D05: confidence floor
            and (
                (evidence_count >= PROVISIONAL_PROMOTION_MIN_EVIDENCE
                 and access_count >= PROVISIONAL_PROMOTION_MIN_ACCESS)
                or reinforcement >= REINFORCEMENT_PROMOTION_THRESHOLD
            )
        ):
            new_maturity = "ESTABLISHED"

        if new_maturity != maturity:
            concept.maturity = new_maturity
            concept.maturity_promoted_at = _utc_now_iso()
            concept.maturity_promotion_evidence = f"Auto-promoted: evidence={evidence_count}, access={access_count}"
            save_concept(concept)
            logger.info(
                f"W7: Maturity promotion {concept_id}: {maturity} → {new_maturity} "
                f"(evidence={evidence_count}, access={access_count})"
            )

    def _evolve_existing_from_dedup(
        self, match: dict, insight: dict, request: SessionLearnRequest, extraction_source: str = "heuristic"
    ) -> dict:
        """Evolve an existing concept with new evidence from conversation.

        Takes dedup result dict {concept_id, cosine_score, knowledge_area, evidence_count}.
        Implements S1 (source tagging), S2 (self-corroboration guard), S5 (HHI cap).
        """
        from app.learning import evolve_concept

        concept_id = match["concept_id"]

        # Evidence saturation check
        if match.get("evidence_count", 0) >= 10:
            logger.debug(f"session_learn: evidence saturated for {concept_id}")
            return {"action": "skipped_saturated"}

        # S2: Self-corroboration guard — check if existing concept has same source
        existing = load_concept(concept_id, track_access=False)
        corroboration_type = None
        confidence_boost = 0.02
        if existing and existing.metadata:
            existing_source = existing.metadata.get("extraction_source", "heuristic")
            if existing_source == extraction_source:
                corroboration_type = "same_source"
                confidence_boost = 0.01  # S2: cap boost for same-source
            else:
                corroboration_type = "cross_source"

        # S5: HHI confidence cap — single-source concepts capped at 0.75
        if existing and corroboration_type == "same_source":
            if existing.confidence >= 0.75:
                confidence_boost = 0.0  # Already at cap

        # FIX-2 (EVOLUTION_CHAIN_BREAK): Type-aware merge routing.
        # When incoming concept has a higher TYPE_RANK than existing, upgrade the type.
        # When same rank, use incoming summary if more specific (>1.2x longer).
        # Never downgrade type rank.
        from app.supersession import TYPE_RANK

        new_concept_type = None
        new_summary = None
        incoming_type = insight.get("type", "observation")
        existing_type = existing.concept_type if existing else "observation"
        incoming_rank = TYPE_RANK.get(incoming_type, 0)
        existing_rank = TYPE_RANK.get(existing_type, 0)

        if incoming_rank > existing_rank:
            # TYPE UPGRADE
            new_concept_type = incoming_type
            new_summary = insight["summary"]
            confidence_boost = max(confidence_boost, 0.05)
            logger.info(
                "FIX-2: Type upgrade for %s: %s (rank %d) -> %s (rank %d)",
                concept_id,
                existing_type,
                existing_rank,
                incoming_type,
                incoming_rank,
            )
        elif incoming_rank == existing_rank and existing:
            # INGEST-027: Same-rank concepts always prefer incoming summary (newest wins).
            # Previous 1.2x length gate blocked same-length corrections:
            #   "User lives in NYC" → "User lives in SF" kept NYC (same length).
            # FC_mh_6k RCA: 13/14 evolved concepts kept stale summaries.
            # Safety: P0-PRECISION guard (downstream) still blocks generic→specific regression.
            new_summary = insight["summary"]
            confidence_boost = max(confidence_boost, 0.03)
        # incoming_rank < existing_rank: DO NOT downgrade.

        # P0-PRECISION: Specificity guard — don't replace a specific summary with a generic one.
        # Evolution re-lossification confirmed 2026-03-17: type upgrades and 1.2x length triggers
        # can replace "Pilsner or Lager for Seco de Cordero" with "beer for lamb dish".
        # Guard: if old summary has named entities/specifics and new doesn't, keep old summary.
        if new_summary and existing and existing.summary:
            old_has_specifics = _has_named_entities(existing.summary)
            new_has_specifics = _has_named_entities(new_summary)
            if old_has_specifics and not new_has_specifics:
                global _PRECISION_GUARD_BLOCKS
                _PRECISION_GUARD_BLOCKS += 1
                logger.info(
                    "P0-PRECISION: Blocking summary replacement — old has named entities, new doesn't. "
                    "old='%s' new='%s' (total_blocks=%d)",
                    existing.summary[:80],
                    new_summary[:80],
                    _PRECISION_GUARD_BLOCKS,
                )
                new_summary = None  # Keep old summary, still add new evidence + type upgrade

        # S1: Build evidence with extraction_source tag
        evidence_entry = {
            "source_type": "conversation",
            "content": f"Extracted from conversation: {insight['summary'][:200]}",
            "source_reference": f"session:{request.session_id}" if request.session_id else None,
            "reliability_weight": 0.7,
            "directness": 0.6,
            "consistency": 0.8,  # MAINT-009: Deprecated — not used in formula. Kept for backward compat.
            "extraction_source": extraction_source,
            "corroboration_type": corroboration_type,
            "model_origin": getattr(request, "model_id", "unknown"),  # FEDERATION L1.5
            "timestamp": _utc_now_iso(),
        }

        evolution = ConceptEvolution(
            concept_id=concept_id,
            new_evidence=[evidence_entry],
            new_signals=insight.get("signals", []),
            confidence_change=confidence_boost,
            new_concept_type=new_concept_type,  # FIX-2: Set when incoming has higher TYPE_RANK
            new_summary=new_summary,  # FIX-2: Set when incoming is more specific
            session_id=request.session_id,  # CASCADE-001 A1.2: Enable reinforcement independence check
            raw_evidence_count=len(insight.get("evidence", [])),  # A1.5: Layer 1 count for cascade
        )

        result = evolve_concept(evolution)
        if result:
            self.record_learning_event()
            # FEDERATION L2: Emit event for cross-pith bridging
            self._emit_federation_event(
                "concept_evolved",
                concept_id,
                {
                    "summary": match.get("summary", "")[:500],
                    "new_confidence": getattr(result, "confidence", 0),
                    "knowledge_area": match.get("knowledge_area", "general"),
                },
                model_id=getattr(request, "model_id", "unknown"),
            )
            # FED-015: Write-time cross-session conflict detection (post-evolve)
            try:
                from app.federation import detect_write_conflict

                detect_write_conflict(
                    new_concept_data={
                        "id": concept_id,
                        "summary": match.get("summary", ""),
                        "knowledge_area": match.get("knowledge_area", "general"),
                        "authority_score": getattr(result, "authority_score", None),
                        "currency_score": getattr(result, "currency_score", None),
                        "embedding": getattr(result, "embedding", None),
                    },
                    source_session_id=request.session_id or "",
                )
            except Exception as e:
                logger.debug(f"FED-015: Evolve conflict check failed (non-fatal): {e}")

            # W7: Maturity promotion lifecycle check after evolution
            try:
                self._maybe_promote_maturity(concept_id)
            except Exception as e:
                logger.warning(f"W7: Maturity promotion check failed for {concept_id}: {e}")
            return {
                "action": "evolved",
                "evolved_concept": EvolvedConcept(
                    concept_id=concept_id,
                    version=result.version,
                    change=f"New evidence ({extraction_source}): {insight.get('type', 'insight')}",
                ),
                "associations": 0,
            }

        return {"action": "skipped_duplicate"}

    def _create_new_concept(
        self,
        insight: dict,
        request: SessionLearnRequest,
        retrieval_engine,
        search_results,
        extraction_source: str = "heuristic",
        skip_write_contradiction: bool = False,
    ) -> dict:
        """Create a new concept with PROVISIONAL maturity and content-hash ID.

        Includes quality gates, knowledge area assignment, auto-association,
        and S1 extraction source tagging.

        Args:
            skip_write_contradiction: Bug 6 fix — when True, skips the write-time
                contradiction check. Used when creating a concept via the supersession
                path, where _detect_contradiction already confirmed the conflict.
        """
        summary = insight["summary"]
        concept_type = insight.get("type", "observation")

        # ORIENTATION_V2 Fix A4: Content-type consistency gate at ingestion
        # Demotes misclassified types (e.g., backlog labeled "decision", impl detail labeled "principle")
        # Gauntlet 3.2 fix: trusts explicit PRINCIPLE: prefix
        concept_type = _validate_concept_type(summary, concept_type)
        insight["type"] = concept_type  # TUNE-EXTRACTION fix: propagate validated type to all consumers

        # Type-aware confidence defaults:
        # Abstract types (principles, methods, strategies) start LOWER —
        # they must earn confidence through citation, not assertion.
        from app.models import ABSTRACT_CONCEPT_TYPES

        if concept_type in ABSTRACT_CONCEPT_TYPES:
            confidence = max(insight.get("confidence", 0.35), 0.35)
            confidence = min(confidence, 0.55)  # Cap: principles earn trust, not assert it
        else:
            confidence = max(insight.get("confidence", 0.40), 0.35)

        # --- Content-hash concept ID ---
        content_hash = hashlib.sha256(summary.encode()).hexdigest()[:12]
        concept_id = f"conv_{content_hash}"

        # Check if concept already exists
        existing = load_concept(concept_id, track_access=False)
        if existing:
            logger.debug(f"session_learn: concept {concept_id} already exists, skipping")
            return {"action": "skipped_duplicate"}

        # --- Knowledge area resolution ---
        # DEBT-030: normalize_knowledge_area + infer_knowledge_area hoisted to module-level import

        # For client extractions, use the provided knowledge_area if available
        if extraction_source == "client" and insight.get("knowledge_area"):
            raw_area = insight["knowledge_area"]
            # KA-007: Client KA was already normalized in Tier 2 (strict=False).
            # Use strict=False here to preserve novel client KAs instead of
            # double-normalizing with strict=True which drops them to "unclassified".
            knowledge_area, ka_source, ka_confidence = classify_knowledge_area(
                summary=summary, raw_area=raw_area, strict=False
            )
        else:
            raw_area = self._resolve_knowledge_area(request, search_results)
            # DEBT-108/KA-003: Shared multi-tier classification (keyword → embedding)
            knowledge_area, ka_source, ka_confidence = classify_knowledge_area(
                summary=summary, raw_area=raw_area, strict=True
            )

        # S1: Build evidence with extraction_source tag
        evidence_entry = {
            "source_type": "conversation",
            "content": f"Extracted from conversation: {summary[:200]}",
            "source_reference": f"session:{request.session_id}" if request.session_id else None,
            "reliability_weight": 0.7,
            "directness": 0.6,
            "consistency": 0.8,  # MAINT-009: Deprecated — not used in formula. Kept for backward compat.
            "extraction_source": extraction_source,
            "model_origin": getattr(request, "model_id", "unknown"),  # FEDERATION L1.5
            "timestamp": _utc_now_iso(),
        }

        # For client extractions, include provided evidence strings
        client_evidence = insight.get("evidence", [])
        if client_evidence and extraction_source == "client":
            evidence_entry["content"] = f"Client evidence: {'; '.join(client_evidence[:3])}"[:200]

        # Memory Integrity §5.2.3: Evidence method anti-spoofing
        try:
            from app.evidence_method import sanitize_evidence

            sanitize_evidence([evidence_entry], source_type=extraction_source)
        except Exception as e:
            logger.warning(f"Evidence anti-spoofing failed (non-fatal): {e}")

        now = _utc_now_iso()

        # INGEST-017: Structural fact classification (overrides markers + LLM)
        _is_factual = insight.get("is_factual", False)
        _temporal_category = insight.get("temporal_category", None)
        _factual_score = None
        _signals_fired = None

        try:
            from app.config import get_feature_flag
            if get_feature_flag("STRUCTURAL_CONCEPT_CLASSIFIER_ENABLED", True):
                from app.fact_classifier import classify_concept
                _cls = classify_concept(
                    summary=insight["summary"],
                    concept_type=insight.get("type", "observation"),
                    knowledge_area=knowledge_area or "general",
                )
                _is_factual = _cls["is_factual"]
                _temporal_category = _cls["temporal_category"]
                _factual_score = _cls["factual_score"]
                _signals_fired = _cls["signals_fired"]
        except Exception:
            logger.debug("INGEST-017: structural classifier unavailable, using fallback")

        # TEMPORAL-002: Extract temporal reference from summary+evidence text
        try:
            from app.temporal import extract_temporal_reference
            _temporal_text = summary + ' ' + ' '.join(str(e) for e in insight.get('evidence', []))
            _original_date = extract_temporal_reference(_temporal_text)
        except Exception:
            _original_date = None

        new_concept = Concept(
            id=concept_id,
            version="v1",
            created_at=now,
            concept_type=insight.get("type", "observation"),
            summary=summary,
            evidence=[evidence_entry],
            signals=insight.get("signals", []),
            confidence=confidence,
            stability=0.5,  # STABILITY-001 Component A: align with learning.py and schema default
            maturity="PROVISIONAL",
            content_hash=content_hash,
            knowledge_area=knowledge_area,  # KA-001: Set directly so save_concept writes it
            original_date=_original_date,  # TEMPORAL-002
            session_id=request.session_id if request.session_id else None,  # AGENT-004
            metadata={
                "knowledge_area": knowledge_area,
                "knowledge_area_source": ka_source,
                "ka_confidence": ka_confidence,  # Float or None. Used by async reclass + trust gating.
                "extraction_source": extraction_source,
                "created_by": "session_learn",
                "source_session": request.session_id,
                "was_untyped": insight.get("was_untyped", False),
                # INGEST-017: Structural fact classification (canonical)
                "is_factual": _is_factual,
                "temporal_category": _temporal_category,
                "factual_score": _factual_score,
                "signals_fired": _signals_fired,
                # INGEST-017: Preserve marker/LLM values for comparison
                "marker_is_factual": insight.get("is_factual", False),
                "llm_is_factual": insight.get("llm_is_factual", None),
                # AGENT-001: request > session > default precedence
                "agent_id": self._resolve_agent_id(request),  # DEBT-019
            },
        )

        # Memory Integrity §5.1.5: Write-time contradiction check
        # Bug 6 fix: Skip when called from supersession path (already detected)
        # BENCHMARK-003: Skip contradiction check when dedup bypass is active —
        # benchmark facts are intentionally counter-factual and should not be rejected.
        from app.config import BENCHMARK as _bm_wcontra
        _skip_contra_for_benchmark = _bm_wcontra.skip_write_contradictions
        if not skip_write_contradiction and not _skip_contra_for_benchmark:
            try:
                from app.contradiction import detect_write_contradiction

                contra_result = detect_write_contradiction(
                    new_summary=summary,
                    new_knowledge_area=knowledge_area,
                    concept_id=concept_id,
                )
                if contra_result.action == "HARD_REJECT":
                    logger.info(
                        f"session_learn: HARD_REJECT concept {concept_id} — "
                        f"contradicts {contra_result.contradicting_concept_id} "
                        f"(score={contra_result.max_score:.3f})"
                    )
                    return {"action": "rejected_contradiction", "reason": contra_result.reason}
                elif contra_result.action == "QUARANTINE":
                    new_concept.maturity = "QUARANTINED"
                    # STABILITY-026: M3 ceiling guard — cap confidence at ingest time
                    from app.config import PSIS_QUARANTINE_CONFIDENCE_CAP
                    if new_concept.confidence > PSIS_QUARANTINE_CONFIDENCE_CAP:
                        logger.info(
                            "STABILITY-026: Capped quarantined concept %s confidence %.3f → %.1f",
                            concept_id, new_concept.confidence, PSIS_QUARANTINE_CONFIDENCE_CAP,
                        )
                        new_concept.confidence = PSIS_QUARANTINE_CONFIDENCE_CAP
                    logger.info(
                        f"session_learn: quarantined {concept_id} — "
                        f"soft contradiction with {contra_result.contradicting_concept_id} "
                        f"(score={contra_result.max_score:.3f}, phase={getattr(contra_result, 'phase', 'unknown')})"
                    )
                    # EVIDENCE_QUARANTINE_SPEC Fix 5: Log governance event for quarantine tracking
                    try:
                        import json as _q_json

                        from app.storage import _db  # BUG-019: was missing, caused NameError

                        with _db() as _gov_conn:
                            _gov_conn.execute(
                                """INSERT INTO governance_events
                                   (event_type, concept_id, details, created_at)
                                   VALUES (?, ?, ?, ?)""",
                                (
                                    "CONCEPT_QUARANTINED",
                                    concept_id,
                                    _q_json.dumps(
                                        {
                                            "contradicting_concept_id": contra_result.contradicting_concept_id,
                                            "max_score": round(contra_result.max_score, 4),
                                            "phase": getattr(contra_result, "phase", None),
                                            "reason": getattr(contra_result, "reason", None),
                                        }
                                    ),
                                    _utc_now_iso(),
                                ),
                            )
                    except Exception:
                        logger.debug("Non-fatal: quarantine governance event logging failed", exc_info=True)
            except Exception as e:
                logger.warning(f"session_learn: contradiction check failed (non-fatal): {e}")
        else:
            logger.info(
                f"session_learn: skipping write-time contradiction check for {concept_id} "
                f"(supersession path — contradiction already confirmed)"
            )

        # Retrieval Defense W2: Epistemic classification before storage
        try:
            from app.epistemic import classify_and_annotate_concept

            classified = classify_and_annotate_concept(new_concept)
            if classified:
                logger.info(
                    f"W2: Epistemic classification applied to {concept_id}: "
                    f"network={new_concept.epistemic_network}, "
                    f"verification={new_concept.verification_status}"
                )
        except Exception as e:
            logger.warning(f"W2: Epistemic classification failed for {concept_id}: {e}")

        # STABILITY-027: M3 compliance — cap confidence for PSIS-quarantined concepts at ingest
        from app.config import PSIS_QUARANTINE_CONFIDENCE_CAP, PSIS_QUARANTINE_EVIDENCE_MARKER
        if PSIS_QUARANTINE_EVIDENCE_MARKER in (new_concept.evidence or []):
            new_concept.confidence = min(new_concept.confidence, PSIS_QUARANTINE_CONFIDENCE_CAP)

        save_concept(new_concept)

        # INGEST-037 Phase 2a: Save auto-extracted verbatim fragments
        # Fragments were attached to the insight dict by session_learn's
        # verbatim detection hook as insight["_verbatim_fragments"].
        _vf_list = insight.get("_verbatim_fragments", [])
        if _vf_list:
            try:
                from app.storage import save_verbatim_fragment

                _vf_saved = 0
                for _vf in _vf_list:
                    _vf_id = save_verbatim_fragment(
                        concept_id=concept_id,
                        fragment_type=getattr(_vf, "fragment_type", "text"),
                        content=getattr(_vf, "content", None),
                        pointer_uri=getattr(_vf, "pointer_uri", None),
                        pointer_meta=getattr(_vf, "pointer_meta", None),
                        concept_version=new_concept.version,
                    )
                    if _vf_id:
                        _vf_saved += 1
                if _vf_saved:
                    logger.info(
                        "INGEST-037: Saved %d/%d verbatim fragments for %s",
                        _vf_saved, len(_vf_list), concept_id,
                    )
            except Exception as _vf_save_err:
                logger.warning(
                    "INGEST-037: Verbatim fragment save failed for %s (non-fatal): %s",
                    concept_id, _vf_save_err,
                )

        # RETRIEVAL-010: Compute governance scores for newly created concepts
        try:
            from app.learning import _recompute_governance_scores

            _recompute_governance_scores(concept_id)
        except Exception as e:
            logger.warning(f"Governance score recompute failed for {concept_id}: {e}")

        # CONTRA-ACTIVATE-001: Write-time supersession check.
        # Previously only in learning.py:learn_concept() which has zero callers —
        # session.py is the actual ingestion path. Wiring added after L2 live
        # testing proved the code path was never reached (0 supersession events).
        _ss_result_data = None
        # BENCH-EVOLVE-001: Skip LLM-based supersession when evolve is disabled.
        # But ALWAYS run deterministic subject-key dedup (RETRIEVAL-072).
        import os as _ss_os
        _supersession_disabled = _ss_os.environ.get("PITH_DISABLE_EVOLVE", "").lower() in ("true", "1")

        # RETRIEVAL-072: Deterministic write-time subject-key supersession.
        # Runs even when PITH_DISABLE_EVOLVE=true. Uses structured pattern
        # matching (same as conflict prefilter) to detect duplicate subject keys.
        # Supersedes the OLDER concept, keeping the newly-written one.
        # No LLM call — pure string matching. Validated: +4 EM on SH 32k.
        try:
            from app.storage import db_immediate
            _new_summary = summary
            _new_key = _extract_subject_key(_new_summary)
            if _new_key:
                with db_immediate() as _sk_conn:
                    # Find active concepts with same subject key
                    _sk_candidates = _sk_conn.execute(
                        "SELECT id, summary FROM concepts "
                        "WHERE superseded_by IS NULL AND id != ?",
                        (concept_id,),
                    ).fetchall()
                    for _sk_cid, _sk_summary in _sk_candidates:
                        _sk_existing_key = _extract_subject_key(_sk_summary or "")
                        if _sk_existing_key == _new_key:
                            # Same subject key — supersede the old one
                            _sk_conn.execute(
                                "UPDATE concepts SET superseded_by = ?, "
                                "supersession_reason = 'RETRIEVAL-072: subject-key dedup', "
                                "is_current = 0 WHERE id = ?",
                                (concept_id, _sk_cid),
                            )
                            logger.info(
                                "RETRIEVAL-072: Subject-key supersession: %s superseded %s "
                                "(key='%s')",
                                concept_id, _sk_cid, _new_key[:60],
                            )
                            break  # One supersession per write
        except Exception as _sk_err:
            logger.warning(
                "RETRIEVAL-072: Subject-key dedup failed for %s (non-fatal): %s",
                concept_id, _sk_err,
            )

        if _supersession_disabled:
            logger.debug("BENCH-EVOLVE-001: LLM supersession disabled (PITH_DISABLE_EVOLVE=true), skipping for %s", concept_id)
        else:
            try:
                from app.storage import db_immediate
                from app.supersession import check_supersession_on_write

                with db_immediate() as _ss_conn:
                    _ss_result = check_supersession_on_write(concept_id, _ss_conn)
                    if _ss_result and _ss_result.superseded:
                        logger.info(
                            "CONTRA-ACTIVATE-001: Write-time supersession: %s superseded %s (%s)",
                            concept_id,
                            _ss_result.old_concept_id,
                            _ss_result.reason,
                        )
                        _ss_result_data = {
                            "old_id": _ss_result.old_concept_id,
                            "new_id": concept_id,
                            "reason": _ss_result.reason,
                        }
            except Exception as _ss_err:
                logger.warning(
                    "CONTRA-ACTIVATE-001: Supersession check failed for %s (non-fatal): %s",
                    concept_id,
                    _ss_err,
                )

        # FEDERATION L2: Emit event for cross-pith bridging
        self._emit_federation_event(
            "concept_proposed",
            concept_id,
            {
                "summary": summary[:500],
                "confidence": confidence,
                "knowledge_area": knowledge_area,
                "concept_type": insight.get("type", "observation"),
            },
            model_id=getattr(request, "model_id", "unknown"),
        )

        # FED-015: Write-time cross-session conflict detection
        try:
            from app.federation import detect_write_conflict

            detect_write_conflict(
                new_concept_data={
                    "id": concept_id,
                    "summary": summary,
                    "knowledge_area": knowledge_area,
                    "authority_score": getattr(new_concept, "authority_score", None),
                    "currency_score": getattr(new_concept, "currency_score", None),
                    "embedding": getattr(new_concept, "embedding", None),
                },
                source_session_id=request.session_id or "",
            )
        except Exception as e:
            logger.debug(f"FED-015: Propose conflict check failed (non-fatal): {e}")

        # CONCEPT_LIFECYCLE_SPEC L4: Track session-created concepts for end-of-session refresh
        self._session_concept_ids.add(concept_id)

        # --- L5: Auto-association (budget: 35ms) ---
        assoc_count = 0
        if request.auto_associate:
            assoc_count = self._auto_associate(concept_id, search_results, retrieval_engine)

        # --- L6: Incremental index update ---
        try:
            retrieval_engine.add_concept(concept_id)
        except Exception as e:
            logger.warning(f"session_learn: index update failed for {concept_id}: {e}")

        # --- L6.5: Prospective indexing (RETRIEVAL-057) ---
        from app.config import PROSPECTIVE_INDEXING_ENABLED
        if PROSPECTIVE_INDEXING_ENABLED:
            try:
                _evidence_strs_pi = []
                for _e in insight.get("evidence", []):
                    if isinstance(_e, str):
                        _evidence_strs_pi.append(_e)
                    elif isinstance(_e, dict):
                        _evidence_strs_pi.append(_e.get("content", ""))

                # Fire-and-forget via dedicated executor (non-blocking)
                import concurrent.futures as _cf_pi
                if not hasattr(self, '_pi_executor') or self._pi_executor is None:
                    self._pi_executor = _cf_pi.ThreadPoolExecutor(
                        max_workers=1, thread_name_prefix="prospective_idx"
                    )
                self._pi_executor.submit(
                    self._generate_implications,
                    concept_id=concept_id,
                    summary=summary,
                    knowledge_area=knowledge_area,
                    concept_type=insight.get("type", "observation"),
                    evidence=_evidence_strs_pi[:3],
                )
                logger.debug(f"RETRIEVAL-057: Queued implications generation for {concept_id}")
            except Exception as e:
                logger.debug(f"RETRIEVAL-057: Failed to queue implications: {e}")

        # Record learning event
        self.record_learning_event()

        result = {
            "action": "created",
            "learned_concept": LearnedConcept(
                concept_id=concept_id,
                summary=summary,
                confidence=confidence,
                knowledge_area=knowledge_area,
                concept_type=insight.get("type", "observation"),
            ),
            "associations": assoc_count,
        }
        # CONTRA-ACTIVATE-001: Surface supersession in result for pipeline summary
        if _ss_result_data:
            result["superseded"] = _ss_result_data
        return result

    # PERF-006: Cache federation_events table existence (won't disappear mid-session)
    _federation_table_exists: bool | None = None

    @classmethod
    def _reset_federation_cache(cls) -> None:
        """PERF-007: Invalidate federation table existence cache after migrations."""
        cls._federation_table_exists = None

    def _extract_events(
        self,
        combined_text: str,
        concept_ids: list[str],
        session_id: str | None = None,
    ) -> None:
        """INGEST-034: Background event extraction via LLM. Runs in _event_executor.

        Extracts structured {action, cause, consequence, actors} tuples from
        conversation text, then attaches them to the specified concepts via
        update_concept_data. Fire-and-forget — failure is logged, never blocks.
        """
        import anthropic
        from app.config import EE_LLM_MODEL, EE_MAX_OUTPUT_TOKENS, EE_TIMEOUT_SECONDS, EE_MAX_INPUT_CHARS
        from app.extraction import build_event_extraction_prompt, parse_event_response

        try:
            text = combined_text[:EE_MAX_INPUT_CHARS]
            prompt = build_event_extraction_prompt(text)

            client = anthropic.Anthropic(timeout=EE_TIMEOUT_SECONDS)
            response = client.messages.create(
                model=EE_LLM_MODEL,
                max_tokens=EE_MAX_OUTPUT_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )

            response_text = response.content[0].text if response.content else ""
            events = parse_event_response(response_text)

            if not events:
                logger.debug("INGEST-034: No events extracted from conversation")
                return

            event_dicts = [e.to_dict() for e in events]
            event_texts = [e.to_searchable_text() for e in events]

            logger.info(
                f"INGEST-034: Extracted {len(events)} events for {len(concept_ids)} concepts "
                f"(session={session_id})"
            )

            from app.storage import _db, update_concept_data
            import json

            for concept_id in concept_ids:
                try:
                    with _db() as conn:
                        row = conn.execute(
                            "SELECT data FROM concepts WHERE id = ? AND is_current = 1",
                            (concept_id,),
                        ).fetchone()
                        if not row:
                            continue

                        data = json.loads(row[0]) if row[0] else {}
                        data["events"] = event_dicts

                        if "metadata" not in data:
                            data["metadata"] = {}
                        data["metadata"]["events"] = event_dicts
                        data["metadata"]["event_texts"] = event_texts

                        update_concept_data(conn, concept_id, data)
                        # commit handled by _db() context manager

                    # Re-index to update embedding with event text
                    try:
                        from app.retrieval import retrieval_engine
                        retrieval_engine.add_concept(concept_id)
                    except Exception as reindex_err:
                        logger.warning(
                            f"INGEST-034: Re-index failed for {concept_id} after event attach: {reindex_err}"
                        )

                except Exception as attach_err:
                    logger.warning(f"INGEST-034: Failed to attach events to {concept_id}: {attach_err}")

        except anthropic.APITimeoutError:
            logger.warning("INGEST-034: Event extraction LLM call timed out")
        except anthropic.APIError as api_err:
            logger.warning(f"INGEST-034: Event extraction API error: {api_err}")
        except Exception as e:
            logger.error(f"INGEST-034: Event extraction failed: {e}", exc_info=True)

    def _generate_implications(
        self,
        concept_id: str,
        summary: str,
        knowledge_area: str,
        concept_type: str,
        evidence: list[str],
    ) -> None:
        """RETRIEVAL-057: Background prospective indexing (sync, runs in executor).

        Generates hypothetical future retrieval scenarios for a newly created
        concept. Updates the concept's data JSON with an 'implications' field
        and re-indexes embedding.
        """
        import os
        import time
        from datetime import datetime, UTC

        from app.config import (
            PI_COOLDOWN_SECONDS,
            PI_DAILY_BUDGET,
            PI_LLM_MODEL,
            PI_MAX_IMPLICATIONS,
            PI_MAX_OUTPUT_TOKENS,
            PI_MIN_SUMMARY_LENGTH,
        )

        t0 = time.perf_counter()

        # Gate 1: Minimum summary length
        if len(summary) < PI_MIN_SUMMARY_LENGTH:
            logger.debug(f"RETRIEVAL-057: Skipping implications — summary too short ({len(summary)} chars)")
            return

        # Gate 2: API key available
        if not os.environ.get("ANTHROPIC_API_KEY"):
            logger.debug("RETRIEVAL-057: Skipping implications — no ANTHROPIC_API_KEY")
            return

        # Gate 3: Daily budget check
        if not hasattr(self, "_pi_calls_today"):
            self._pi_calls_today = 0
            self._pi_day = datetime.now(UTC).date()

        current_day = datetime.now(UTC).date()
        if current_day != self._pi_day:
            self._pi_calls_today = 0
            self._pi_day = current_day

        if self._pi_calls_today >= PI_DAILY_BUDGET:
            logger.info(f"RETRIEVAL-057: PI daily budget exhausted ({PI_DAILY_BUDGET})")
            return

        # Gate 4: Cooldown check
        if hasattr(self, "_pi_last_call"):
            elapsed = time.perf_counter() - self._pi_last_call
            if elapsed < PI_COOLDOWN_SECONDS:
                logger.debug(f"RETRIEVAL-057: PI cooldown ({elapsed:.1f}s < {PI_COOLDOWN_SECONDS}s)")
                return

        try:
            from app.extraction import build_implications_prompt, parse_implications_response

            prompt = build_implications_prompt(
                summary=summary,
                knowledge_area=knowledge_area,
                concept_type=concept_type,
                evidence=evidence,
                max_implications=PI_MAX_IMPLICATIONS,
            )

            import anthropic

            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=PI_LLM_MODEL,
                max_tokens=PI_MAX_OUTPUT_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )

            raw_text = response.content[0].text if response.content else ""
            self._pi_calls_today += 1
            self._pi_last_call = time.perf_counter()

            # Record cost metric
            try:
                from app.metrics import metrics as _pi_metrics
                _pi_metrics.record(
                    "prospective_indexing_llm_call",
                    1.0,
                    {
                        "model": PI_LLM_MODEL,
                        "input_tokens": response.usage.input_tokens if response.usage else 0,
                        "output_tokens": response.usage.output_tokens if response.usage else 0,
                        "concept_id": concept_id,
                    },
                )
            except Exception:
                pass  # Metrics are best-effort

            implications = parse_implications_response(raw_text, PI_MAX_IMPLICATIONS)

            if not implications:
                logger.info(f"RETRIEVAL-057: No implications parsed for {concept_id}")
                return

            # Update concept data JSON with implications (dual-storage pattern)
            from app.storage import _db, update_concept_data
            import json

            with _db() as conn:
                row = conn.execute(
                    "SELECT data FROM concepts WHERE id = ? AND is_current = 1",
                    (concept_id,),
                ).fetchone()
                if not row:
                    logger.warning(f"RETRIEVAL-057: Concept {concept_id} not found for implications update")
                    return

                current_data = json.loads(row[0]) if row[0] else {}
                current_data["implications"] = implications
                current_data["implications_model"] = PI_LLM_MODEL
                current_data["implications_generated_at"] = _utc_now_iso()

                # Also store in metadata for Pydantic-loaded access path
                meta = current_data.get("metadata", {})
                if not isinstance(meta, dict):
                    meta = {}
                meta["implications"] = implications
                current_data["metadata"] = meta

                update_concept_data(conn, concept_id, current_data)
                # commit handled by _db() context manager

            # Re-index with implications included in searchable text
            from app.retrieval import retrieval_engine
            retrieval_engine.add_concept(concept_id)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.info(
                f"RETRIEVAL-057: Generated {len(implications)} implications for {concept_id} "
                f"in {elapsed_ms:.0f}ms"
            )

        except Exception as e:
            logger.warning(f"RETRIEVAL-057: Implications generation failed for {concept_id}: {e}")

    def _emit_federation_event(
        self, event_type: str, concept_id: str, payload: dict, model_id: str = "unknown"
    ) -> None:
        """Emit a federation event for cross-pith bridging.

        Non-critical — failure is logged but doesn't block concept creation.
        Only emits if FEDERATION_EVENTS_ENABLED and federation_events table exists.
        """
        try:
            from app.config import FEATURE_FLAGS

            if not FEATURE_FLAGS.get("FEDERATION_EVENTS_ENABLED", False):
                return

            from app.storage import _db

            with _db() as conn:
                # PERF-006: Check cache first, query only on first call
                if self._federation_table_exists is None:
                    tables = [
                        r[0]
                        for r in conn.execute(
                            "SELECT name FROM sqlite_master WHERE type='table' AND name='federation_events'"
                        ).fetchall()
                    ]
                    self.__class__._federation_table_exists = "federation_events" in tables
                if not self._federation_table_exists:
                    return  # Pre-migration — silently skip

                conn.execute(
                    """INSERT INTO federation_events
                       (event_type, concept_id, source_session_id, source_model_id,
                        source_agent_id, payload, origin_brain, bridge_depth, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, NULL, 0, datetime('now'))""",
                    (
                        event_type,
                        concept_id,
                        self.current_session.session_id if self.current_session else None,
                        model_id,
                        getattr(self, "_current_agent_id", "default"),
                        json.dumps(payload),
                    ),
                )
        except Exception as e:
            logger.debug(f"Federation event emission failed (non-fatal): {e}")

    def _resolve_agent_id(self, request) -> str:
        """DEBT-019: Agent ID precedence: request > session > default."""
        req_aid = getattr(request, "agent_id", "default")
        if req_aid and req_aid != "default":
            return req_aid
        if self.current_session and getattr(self.current_session, "agent_id", "default") != "default":
            return self.current_session.agent_id
        return "default"

    def _resolve_knowledge_area(self, request: SessionLearnRequest, search_results) -> str:
        """3-tier knowledge area fallback (design gap §11.7).

        Tier 1: Explicit from request
        Tier 2: Inherit from nearest TF-IDF match (0.30-0.49)
        Tier 3: Default "conversation"
        """
        # Tier 1: Explicit override
        if request.knowledge_area and request.knowledge_area != "conversation":
            return request.knowledge_area

        # Tier 2: Nearest match inference
        if search_results:
            for result in search_results:
                if 0.30 <= result.relevance_score < 0.50:
                    if result.knowledge_area:
                        return result.knowledge_area

        # Tier 3: Default
        return "conversation"

    def _auto_associate(self, concept_id: str, search_results, retrieval_engine) -> int:
        """L5: Create associations with related concepts.

        Delegates to shared auto_associate_single pipeline which uses raw
        TF-IDF cosine similarity (consistent with batch pipeline).
        Returns count of associations created.
        """
        from app.association import auto_associate_single
        from app.models import AutoAssociateSingleRequest

        request = AutoAssociateSingleRequest(threshold=0.12, max_edges=3)
        try:
            result = auto_associate_single(concept_id, request)
            return result.edges_created
        except Exception as e:
            logger.warning(f"session_learn: auto_associate failed for {concept_id}: {e}")
            return 0

    def _session_duration(self) -> float:
        """Compute session duration in seconds."""
        if not self.current_session or not self.current_session.started_at:
            return 0.0
        try:
            start = _ensure_aware(datetime.fromisoformat(self.current_session.started_at))
            end = _utc_now()
            return round((end - start).total_seconds(), 1)
        except (ValueError, TypeError):
            return 0.0


# Singleton instance
session_manager = SessionManager()
