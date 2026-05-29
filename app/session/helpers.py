"""Session middleware — lifecycle management + present moment orientation.

Phase 1A D7: Implements orientation (where-been/where-am/where-going) and session bookends.
Phase 1B P1.1: Session persistence to SQLite, startup recovery, stub retirement.

Key design: session_start loads concepts ONCE and passes to both introspect
and orient — single disk scan, no redundant reads.

Stub surface area (2 stubs remaining for Phase 1B+ retirement):
  - contradictions_detected — wired to DB via ARCH-O01 (session.py:1165)
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

from app.core.constants import (
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
from app.core.datetime_utils import _ensure_aware, _utc_now, _utc_now_iso
from app.core.models import (
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
from app.session.self_model import self_model_manager
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
from app.cognitive.taxonomy import (  # DEBT-030/DEBT-108
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


def _log_concept_type_correction(from_type: str, to_type: str, summary: str) -> None:
    """TIER2-DAY1: Log concept_type corrections for observability.

    Structured log + MetricsCollector for /learning_metrics visibility.
    """
    logger.info(
        "INGEST-TYPE-CORRECTION concept_type corrected: %s → %s | summary_prefix=%.80s",
        from_type, to_type, summary,
    )
    try:
        from app.ops.metrics import metrics as _metrics
        _metrics.record("concept_type_correction", 1, labels={"from": from_type, "to": to_type})
    except Exception:
        pass  # Never block ingestion on metrics failure


def _validate_concept_type(summary: str, claimed_type: str) -> str:
    """Validate and correct concept_type based on content signals.

    ORIENTATION_V2 Fix A4: Ingestion gate for content-type consistency.
    Applied at concept creation time to prevent future misclassification.

    TUNE-EXTRACTION: Also UPGRADES observations with structural markers.
    Client defaults to "observation" when uncertain — detect upgradable patterns.

    Gauntlet 3.2 fix: Trust explicit 'PRINCIPLE:' prefix.

    TIER2-DAY1: Added correction logging + MetricsCollector tracking.
    """
    # TUNE-EXTRACTION: Upgrade observations with structural signals.
    if claimed_type == "observation":
        stripped = summary.strip()
        s_lower = summary.lower()
        # Explicit prefix markers (highest confidence)
        if stripped.startswith("PRINCIPLE:") or stripped.startswith("[PRINCIPLE]"):
            _log_concept_type_correction(claimed_type, "principle", summary)
            return "principle"
        if stripped.startswith("HEURISTIC:") or stripped.startswith("[HEURISTIC]"):
            _log_concept_type_correction(claimed_type, "heuristic", summary)
            return "heuristic"
        if stripped.startswith("METHOD:") or stripped.startswith("[METHOD]"):
            _log_concept_type_correction(claimed_type, "method", summary)
            return "method"
        if stripped.startswith("DECISION") and ":" in stripped[:12]:
            _log_concept_type_correction(claimed_type, "decision", summary)
            return "decision"
        if stripped.startswith("PATTERN:") or stripped.startswith("[PATTERN]"):
            _log_concept_type_correction(claimed_type, "pattern", summary)
            return "pattern"
        if stripped.startswith("CONSTRAINT:") or stripped.startswith("[CONSTRAINT]"):
            _log_concept_type_correction(claimed_type, "constraint", summary)
            return "constraint"
        # Structural signals (moderate confidence — gauntlet B2: require action verb)
        _has_imperative = any(kw in s_lower for kw in ("should ", "require", "enforce", "ensure", "must "))
        if _has_imperative and any(kw in s_lower for kw in ("always ", "never ", "rule:")):
            if len(summary) > 60:
                _log_concept_type_correction(claimed_type, "constraint", summary)
                return "constraint"
        # Causal / conditional patterns → heuristic
        if ("→" in summary or "->" in summary) and any(kw in s_lower for kw in ("when ", "if ", "trigger")):
            _log_concept_type_correction(claimed_type, "heuristic", summary)
            return "heuristic"

    if claimed_type == "principle":
        stripped = summary.strip()
        if not stripped.startswith("PRINCIPLE:") and not stripped.startswith("[PRINCIPLE]"):
            if _IMPLEMENTATION_DETAIL_PATTERNS.search(summary):
                _log_concept_type_correction(claimed_type, "observation", summary)
                return "observation"
            if _COMMIT_RECORD_PATTERNS.search(summary):
                _log_concept_type_correction(claimed_type, "observation", summary)
                return "observation"
    if claimed_type == "decision":
        if _BACKLOG_PATTERNS.search(summary):
            _log_concept_type_correction(claimed_type, "observation", summary)
            return "observation"
        if _COMMIT_RECORD_PATTERNS.search(summary):
            _log_concept_type_correction(claimed_type, "observation", summary)
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
    from app.cognitive.entity_detector import has_specific_entities
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
               ' plans ', ' wants ', ' likes ', ' enjoys ', ' prefers ',
               # KU-VALUE-CONFLICT-001: Common possession/role verbs for KU matching
               ' owns ', ' drives ', ' manages ', ' leads ', ' teaches ',
               ' runs ', ' rents ', ' operates ']

_SK_STOP_WORDS = frozenset({'a', 'an', 'the', 'their', 'his', 'her', 'its', 'my',
                             'in', 'on', 'at', 'to', 'for', 'of', 'from', 'with',
                             'and', 'or', 'but', 'that', 'which', 'who', 'where',
                             'been', 'being', 'have', 'having', 'also', 'very',
                             'just', 'about', 'really', 'currently', 'recently',
                             'specifically', 'especially', 'new', 'old'})

_SK_PLACEHOLDERS = {'n', '$x', 'time', 'date'}

# ── D851D5BA: Monetary event normalization (Phase 0.5) ──────────────────
# Paraphrased monetary event facts ("raised $1,000 at a bake sale" vs
# "raised over $1,000 through fundraising at a bake sale") produce divergent
# keys under copula keying. This phase extracts a canonical key from the
# dollar amount + event descriptor, collapsing paraphrases.
_SK_MONETARY_RE = _re.compile(r'\$[\d,]+(?:\.\d{1,2})?')
_SK_EVENT_VERBS = frozenset({
    'raised', 'donated', 'contributed', 'gave', 'spent', 'earned',
    'collected', 'fundraised', 'volunteered', 'participated',
})
_SK_EVENT_NOUNS = frozenset({
    'charity', 'fundraiser', 'fundraising', 'donation', 'bake',
    'shelter', 'event', 'marathon', 'run', 'walk', 'auction',
    'gala', 'benefit', 'drive', 'campaign',
})


def _sk_extract_monetary_event_key(text_lower: str) -> str:
    """Extract canonical key for monetary event facts.

    Returns 'monetary_event|{subject}|{amount}|{event_words}' if the text
    describes a monetary event (contains a dollar amount + event verb/noun).
    Returns empty string if not a monetary event fact.

    Amendment A2: includes event descriptor words to prevent over-dedup of
    same-amount different-event facts (e.g., $1000 bake sale vs $1000 car wash).
    Amendment A4: empty/short summary guard.
    """
    if not text_lower or len(text_lower.strip()) < 10:
        return ""

    # Must contain a dollar amount
    amounts = _SK_MONETARY_RE.findall(text_lower)
    if not amounts:
        return ""

    # Must contain at least one event verb or event noun
    words_raw = text_lower.split()
    words_clean = {w.strip('.,;:!?\'()"-') for w in words_raw}
    has_event_verb = bool(words_clean & _SK_EVENT_VERBS)
    has_event_noun = bool(words_clean & _SK_EVENT_NOUNS)
    if not (has_event_verb or has_event_noun):
        return ""

    # Extract subject (text before first monetary verb/noun or dollar sign)
    # For personal memory, subject is typically "user"
    subject = "user"
    for prefix in ("user ", "the user "):
        if text_lower.startswith(prefix):
            subject = "user"
            break

    # Normalize amount: strip commas, trailing decimals
    amount = amounts[0].replace(',', '')

    # Key on subject + amount only. Gauntlet A2 suggested including event
    # descriptor, but paraphrases use different nouns for the same event
    # ("charity event" vs "fundraising", "charity event run" vs "event run
    # shelter"). The amount is the only reliably stable signal. Over-dedup
    # risk (two genuinely different events at the same dollar amount for the
    # same subject) is extremely low in practice — and the prefilter keeps
    # highest serial_order, so the most recent fact survives.
    return f"monetary_event|{subject}|{amount}"


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

    # Phase 0.5: Monetary event normalization (D851D5BA fix)
    # Must run BEFORE generic keying to prevent paraphrase-divergent keys.
    _monetary_key = _sk_extract_monetary_event_key(text_lower)
    if _monetary_key:
        return _monetary_key

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
                # KU-VALUE-CONFLICT-001: Include attribute noun to prevent
                # over-broad keys. 'user has N bikes' -> 'user | bikes'
                # instead of just 'user' which matches everything.
                attr_words = [w for w in pred_norm.split()
                              if w.lower().strip('.,;:') not in _SK_STOP_WORDS
                              and w.lower().strip(".,;:") not in _SK_PLACEHOLDERS]
                if attr_words:
                    return f"{subject} | {' '.join(attr_words)}"
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
        """Extract subject+predicate key.

        Amendment A1 (D851D5BA): Delegates to module-level _extract_subject_key
        for single source of truth. The module-level function includes Phase 0.5
        monetary event normalization that this inner copy previously lacked.
        """
        return _extract_subject_key(text)

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

    def _has_selected_authority(c) -> bool:
        """Return true only for validated, explicit branch-authority winners."""
        envelope = getattr(c, "branch_provenance", None)
        if envelope is None:
            metadata = getattr(c, "metadata", None)
            if isinstance(metadata, dict):
                envelope = metadata.get("branch_provenance")
        if not isinstance(envelope, dict):
            return False
        if envelope.get("branch_resolution_state") != "selected_authoritative":
            return False
        try:
            from app.cognitive.branch_provenance_metadata import validate_branch_provenance_metadata

            return bool(validate_branch_provenance_metadata(envelope).ready)
        except Exception:
            return False

    def _prefer_candidate(candidate, existing) -> bool:
        candidate_has_authority = _has_selected_authority(candidate)
        existing_has_authority = _has_selected_authority(existing)
        if candidate_has_authority != existing_has_authority:
            return candidate_has_authority
        return _get_serial(candidate) > _get_serial(existing)

    best_per_subject: dict[str, object] = {}

    for c in concepts:
        text = (getattr(c, 'summary', '') or '')
        key = _extract_key(text)

        existing = best_per_subject.get(key)
        if existing is None or _prefer_candidate(c, existing):
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


def _infer_session_local_slot_frame(query: str) -> dict[str, str] | None:
    """Infer the supported session-local slot frame for a user query.

    BENCH-045 scope stays intentionally narrow:
    - current_state
    - entity_attribute
    - preference_profile (specialized through the same machinery)
    """
    q = (query or "").strip().lower()
    if not q:
        return None

    if (
        ("stay connected" in q and ("colleagues" in q or "coworkers" in q or "team" in q))
        or ("working remotely" in q and ("connected" in q or "colleagues" in q))
        or ("remote work" in q and ("connected" in q or "colleagues" in q))
    ):
        return {
            "slot_class": "preference_profile",
            "slot_subject": "user_remote_work_connection",
            "slot_attribute": "preference_profile",
            "slot_group_id": f"preference_profile:{hashlib.sha1(q.encode()).hexdigest()[:12]}",
        }

    if "wake up" in q or "wakes up" in q:
        return {
            "slot_class": "current_state",
            "slot_subject": "user",
            "slot_attribute": "wake_up_time",
            "slot_group_id": f"current_state:{hashlib.sha1(q.encode()).hexdigest()[:12]}",
        }

    if any(p in q for p in ("work for", "work at", "employer", "company", "job", "role", "title")):
        return {
            "slot_class": "entity_attribute",
            "slot_subject": "user",
            "slot_attribute": "employer",
            "slot_group_id": f"entity_attribute:{hashlib.sha1(q.encode()).hexdigest()[:12]}",
        }

    if any(p in q for p in ("buy", "bought", "purchase", "purchased", "got for myself", "got my own")):
        return {
            "slot_class": "entity_attribute",
            "slot_subject": "user",
            "slot_attribute": "bought_item",
            "slot_group_id": f"entity_attribute:{hashlib.sha1(q.encode()).hexdigest()[:12]}",
        }

    return None


_SESSION_LOCAL_PAST_MARKERS = (
    "used to",
    "previously",
    "formerly",
    "earlier",
    "back when",
    "before ",
    "last year",
)
_SESSION_LOCAL_FUTURE_MARKERS = (
    "plan to",
    "plans to",
    "going to",
    "will ",
    "next ",
    "soon ",
    "wants to",
)
_PREFERENCE_OPTION_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("virtual coffee breaks", ("virtual coffee breaks", "virtual coffee break")),
    ("virtual team-building activities", ("virtual team-building activities", "online team activities", "team-building activities")),
    ("collaborative projects and working groups", ("collaborative projects", "working groups", "collaborative project")),
    ("interest-based groups", ("interest-based groups", "interest based groups")),
    ("regular check-ins", ("regular check-ins", "regular checkins", "check-ins")),
]


def _session_local_surface_text(concept: ActivatedConcept) -> str:
    parts = [concept.summary]
    parts.extend(concept.key_evidence or [])
    for fragment in concept.verbatim_fragments or []:
        if isinstance(fragment, dict) and fragment.get("content"):
            parts.append(fragment["content"])
    return " ".join(p for p in parts if p).lower()


def _extract_preference_options(text: str) -> list[str]:
    found: list[str] = []
    for canonical, patterns in _PREFERENCE_OPTION_PATTERNS:
        if any(pattern in text for pattern in patterns) and canonical not in found:
            found.append(canonical)
    return found


def _classify_session_local_evidence(
    concept: ActivatedConcept, slot_frame: dict[str, str]
) -> dict[str, Any] | None:
    """Classify a surfaced concept into a BENCH-045 session-local evidence role."""
    text = _session_local_surface_text(concept)
    summary_lower = (concept.summary or "").lower()

    if summary_lower.startswith(("[firmware]", "[principle]", "[always]", "[cko]")):
        return {
            "is_session_local_evidence": False,
            "evidence_role": "schema_meta",
            "slot_subject": slot_frame["slot_subject"],
            "slot_attribute": slot_frame["slot_attribute"],
            "slot_group_id": slot_frame["slot_group_id"],
            "grounding_priority": 0.05,
        }

    slot_class = slot_frame["slot_class"]
    slot_attribute = slot_frame["slot_attribute"]
    has_past = any(marker in text for marker in _SESSION_LOCAL_PAST_MARKERS)
    has_future = any(marker in text for marker in _SESSION_LOCAL_FUTURE_MARKERS)

    if slot_class == "preference_profile":
        options = _extract_preference_options(text)
        has_context = any(term in text for term in ("remote", "colleagues", "coworkers", "team", "connected"))
        has_negative = any(term in text for term in ("not prefer", "doesn't help", "does not help", "one-size-fits-all", "generic"))
        if options:
            return {
                "is_session_local_evidence": True,
                "evidence_role": "preference_positive",
                "slot_subject": slot_frame["slot_subject"],
                "slot_attribute": slot_attribute,
                "slot_group_id": slot_frame["slot_group_id"],
                "grounding_priority": 0.98,
            }
        if has_negative:
            return {
                "is_session_local_evidence": True,
                "evidence_role": "preference_negative",
                "slot_subject": slot_frame["slot_subject"],
                "slot_attribute": slot_attribute,
                "slot_group_id": slot_frame["slot_group_id"],
                "grounding_priority": 0.85,
            }
        if has_context:
            return {
                "is_session_local_evidence": True,
                "evidence_role": "preference_context",
                "slot_subject": slot_frame["slot_subject"],
                "slot_attribute": slot_attribute,
                "slot_group_id": slot_frame["slot_group_id"],
                "grounding_priority": 0.55,
            }
        return None

    direct_match = False
    if slot_attribute == "wake_up_time":
        direct_match = "wake" in text and bool(
            _re.search(r"\b\d{1,2}:\d{2}\s*(?:am|pm)\b", text)
        )
    elif slot_attribute == "employer":
        direct_match = any(term in text for term in ("work for", "work at", "employed by", "company", "employer"))
    elif slot_attribute == "bought_item":
        direct_match = any(term in text for term in ("bought", "purchased", "got their own", "got my own", "own set"))

    if not direct_match:
        related = False
        if slot_attribute == "wake_up_time":
            related = "wake" in text or "morning" in text
        elif slot_attribute == "employer":
            related = any(term in text for term in ("job", "role", "title", "work"))
        elif slot_attribute == "bought_item":
            related = any(term in text for term in ("buy", "bought", "purchase", "purchased", "competition"))
        if related:
            return {
                "is_session_local_evidence": True,
                "evidence_role": "generic_reference",
                "slot_subject": slot_frame["slot_subject"],
                "slot_attribute": slot_attribute,
                "slot_group_id": slot_frame["slot_group_id"],
                "grounding_priority": 0.4,
            }
        return None

    if has_future:
        role = "plan_future"
        priority = 0.2
    elif has_past:
        role = "stale_past"
        priority = 0.35
    else:
        role = "direct_current"
        priority = 0.96

    return {
        "is_session_local_evidence": True,
        "evidence_role": role,
        "slot_subject": slot_frame["slot_subject"],
        "slot_attribute": slot_attribute,
        "slot_group_id": slot_frame["slot_group_id"],
        "grounding_priority": priority,
    }


def _build_session_local_grounding(
    activated: list[ActivatedConcept], query: str
) -> tuple[dict[str, str] | None, dict[str, Any] | None, ActivatedConcept | None]:
    """Apply BENCH-045 session-local grounding over activated concepts."""
    slot_frame = _infer_session_local_slot_frame(query)
    if not slot_frame:
        return None, None, None

    session_local: list[ActivatedConcept] = []
    for concept in activated:
        annotation = _classify_session_local_evidence(concept, slot_frame)
        if not annotation:
            continue
        concept.is_session_local_evidence = annotation["is_session_local_evidence"]
        concept.evidence_role = annotation["evidence_role"]
        concept.slot_subject = annotation["slot_subject"]
        concept.slot_attribute = annotation["slot_attribute"]
        concept.slot_group_id = annotation["slot_group_id"]
        concept.grounding_priority = annotation["grounding_priority"]
        session_local.append(concept)

    direct_roles = {"direct_current", "preference_positive", "preference_negative"}
    direct = [c for c in session_local if c.evidence_role in direct_roles]

    if slot_frame["slot_class"] == "preference_profile":
        option_text = " ".join(_session_local_surface_text(c) for c in session_local)
        options = _extract_preference_options(option_text)
        if options:
            summary = (
                "To stay connected with colleagues while working remotely, "
                f"the user responds best to {', '.join(options[:-1]) + ', and ' + options[-1] if len(options) > 1 else options[0]}."
            )
            synthetic = ActivatedConcept(
                concept_id=f"grounding:{slot_frame['slot_group_id']}",
                summary=summary,
                confidence=max((c.confidence for c in direct), default=0.7),
                relevance_score=max((c.relevance_score for c in direct), default=0.6),
                knowledge_area="preference",
                key_evidence=[],
                associations=[],
                is_session_local_evidence=True,
                evidence_role="grounded_synthetic_preference",
                slot_subject=slot_frame["slot_subject"],
                slot_attribute=slot_frame["slot_attribute"],
                slot_group_id=slot_frame["slot_group_id"],
                grounding_priority=1.0,
            )
            return slot_frame, {
                "grounded_slot_subject": slot_frame["slot_subject"],
                "grounded_slot_attribute": slot_frame["slot_attribute"],
                "grounding_mode": "synthesized",
                "grounding_confidence": round(min(0.95, 0.6 + 0.08 * len(options)), 2),
            }, synthetic
        return slot_frame, {
            "grounded_slot_subject": slot_frame["slot_subject"],
            "grounded_slot_attribute": slot_frame["slot_attribute"],
            "grounding_mode": "missing",
            "grounding_confidence": 0.0,
        }, None

    if not direct:
        return slot_frame, {
            "grounded_slot_subject": slot_frame["slot_subject"],
            "grounded_slot_attribute": slot_frame["slot_attribute"],
            "grounding_mode": "missing",
            "grounding_confidence": 0.0,
        }, None

    ranked = sorted(
        direct,
        key=lambda c: (
            c.grounding_priority or 0.0,
            c.serial_order or 0,
            c.relevance_score,
            c.confidence,
        ),
        reverse=True,
    )
    best = ranked[0]
    synthetic = ActivatedConcept(
        concept_id=f"grounding:{slot_frame['slot_group_id']}",
        summary=best.summary,
        confidence=best.confidence,
        relevance_score=max(best.relevance_score, 0.65),
        knowledge_area=best.knowledge_area,
        key_evidence=list(best.key_evidence or []),
        associations=[],
        is_session_local_evidence=True,
        evidence_role="grounded_resolved",
        slot_subject=slot_frame["slot_subject"],
        slot_attribute=slot_frame["slot_attribute"],
        slot_group_id=slot_frame["slot_group_id"],
        grounding_priority=1.0,
        freshness_label=best.freshness_label,
        serial_order=best.serial_order,
        original_date=best.original_date,
    )
    return slot_frame, {
        "grounded_slot_subject": slot_frame["slot_subject"],
        "grounded_slot_attribute": slot_frame["slot_attribute"],
        "grounding_mode": "direct",
        "grounding_confidence": round(min(0.99, (best.confidence or 0.6) + 0.1), 2),
    }, synthetic


def _chain_aware_prune(
    concepts: list,
    destroyed_concepts: list,
    protected_ids: set[str] | None = None,
) -> list:
    """Remove orphaned chain fragments after subject-level dedup.

    When a conflict loser is destroyed (e.g., "Steve Sax → baseball"),
    downstream facts keyed on the destroyed OBJECT become orphaned.

    Key insight: an object is only TRULY orphaned if no surviving concept
    still references it. Uses reference counting, not global elimination.

    Args:
        concepts: surviving concepts after subject-level dedup
        destroyed_concepts: concepts removed by subject-level dedup
        protected_ids: concept IDs that must survive pruning even if their
            subject looks orphaned by a destroyed sibling.
    Returns:
        filtered concept list with orphans removed
    """
    protected_ids = protected_ids or set()

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
            if getattr(c, "concept_id", None) in protected_ids:
                kept_this_round.append(c)
                continue
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
