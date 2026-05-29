"""Staleness Detection — CONCEPT_STALENESS_FIX_SPEC v1.1

Two-trigger system for detecting and resolving stale concepts:
  Trigger 1 (in-session): After new concept creation, check embedding neighbors
  Trigger 2 (session-end): Cross-reference checkpoints against concepts

Uses embedding search (NOT TF-IDF) — empirical validation showed TF-IDF cosine
of 0.04 between same-topic status-transition pairs vs embedding cosine of 0.42.
"""

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from app.core.datetime_utils import _ensure_aware, _utc_now

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

# Embedding cosine thresholds (NOT TF-IDF — validated empirically)
STALE_RELATIVE_EMBEDDING_MIN = 0.30  # Catches all 4 validated pairs
STALE_RELATIVE_EMBEDDING_MAX = 0.85  # Above this, dedup already handles it

# Topic overlap gate (Amendment A2)
TOPIC_OVERLAP_MIN_TERMS = 2
SIGNIFICANT_TERM_MIN_LENGTH = 4

# Caps
MAX_STALE_RESOLUTIONS_PER_TURN = 3
MAX_STALE_RESOLUTIONS_PER_SESSION_END = 5
RECENTLY_UPDATED_HOURS = 1

# Concept types eligible for staleness marking
# CURRENCY-ACTUATOR: All types eligible — previous 4-type gate excluded 41.1% of
# CONTRADICTED concepts (method, principle, heuristic, constraint).
STALENESS_ELIGIBLE_TYPES = None  # None = all types eligible

# Stop words for topic overlap
STOP_WORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "that",
        "this",
        "with",
        "from",
        "are",
        "was",
        "been",
        "have",
        "has",
        "had",
        "will",
        "would",
        "could",
        "should",
        "not",
        "but",
        "also",
        "into",
        "about",
        "than",
        "then",
        "when",
        "which",
        "their",
        "there",
        "these",
        "those",
        "some",
        "other",
    }
)

# Word-boundary matching indicators (Amendment A5)
STALE_INDICATORS = [
    "written at",
    "spec'd",
    "proposed",
    "planned",
    "pending",
    "not started",
    "in progress",
    "to be implemented",
    "awaiting",
    "needs implementation",
    "will be",
    "should be",
    "draft",
    "investigating",
    "exploring",
    "attempting",
    # STALE-PREMISE-001: Feature-state gap indicators
    "has no",
    "critical gap",
    "missing",
    "not yet",
    "no signal",
]

COMPLETION_INDICATORS = [
    "committed as",
    "implemented",
    "complete",
    "done",
    "shipped",
    "deployed",
    "merged",
    "resolved",
    "fixed",
    "closed",
    "working",
    "live",
    "passed",
    "verified",
]

# STATUS_TRANSITIONS — same as session.py S3.5 but used on wider band
STATUS_TRANSITIONS = [
    (
        ["plan to", "will ", "going to", "intend to", "proposed"],
        ["implemented", "deployed", "built", "completed", "shipped", "launched", "done"],
        "Plan superseded by implementation",
    ),
    (
        ["investigating", "exploring", "looking into", "researching", "analyzing"],
        ["found that", "root cause", "discovered", "turns out", "the issue was", "resolved"],
        "Investigation superseded by finding",
    ),
    (
        ["trying", "attempting", "experimenting with", "testing"],
        ["decided to", "going with", "chose", "opted for", "switched to"],
        "Experiment superseded by decision",
    ),
    (
        ["broken", "failing", "bug", "error", "doesn't work", "not working"],
        ["fixed", "resolved", "patched", "working now", "the fix"],
        "Bug report superseded by fix",
    ),
    (
        ["v1", "initial", "first version", "prototype", "draft"],
        ["v2", "rewrite", "redesign", "refactored", "upgraded", "replaced"],
        "Earlier version superseded by newer version",
    ),
    (
        ["written at", "spec written", "spec'd", "designed"],
        ["committed as", "implemented", "all fixes", "deployed", "shipped"],
        "Spec superseded by implementation",
    ),
    # RETRIEVAL-024 / DATA-057: Technology-elimination patterns
    (
        ["docker", "dockerfile", "docker-compose", "docker compose",
         "node.js wrapper", "node.js bridge", "server.js bridge"],
        ["eliminated", "removed", "replaced", "no longer", "dropped", "python-only",
         "without docker", "without node", "native python",
         "migrated to", "switched to", "moved to"],
        "Technology reference superseded by elimination decision",
    ),
    (
        ["setup.sh", "install.sh", "installer script"],
        ["rewritten", "replaced", "new installer", "updated to",
         "restructured", "overhauled"],
        "Installation tooling superseded by updated version",
    ),
]


# =============================================================================
# Data classes
# =============================================================================


@dataclass
class StalenessSignal:
    """Result of staleness detection for a single concept."""

    is_stale: bool = False
    concept_id: str = ""
    reason: str = ""
    stale_indicator: str = ""
    completion_indicator: str = ""
    detection_method: str = ""  # "status_transition" or "keyword"


@dataclass
class StalenessResult:
    """Result of a staleness check run."""

    concepts_superseded: int = 0
    concepts_staled: int = 0
    details: list = field(default_factory=list)
    time_ms: float = 0.0
    errors: int = 0


# =============================================================================
# Topic overlap gate (Amendment A2)
# =============================================================================


def _tokenize_for_overlap(text: str) -> set:
    """Tokenize text for topic overlap, splitting on underscores/hyphens/slashes too."""
    # Split on word boundaries AND common code separators
    raw_tokens = re.findall(r"[A-Za-z]+", text)
    return {w.lower() for w in raw_tokens if len(w) >= SIGNIFICANT_TERM_MIN_LENGTH and w.lower() not in STOP_WORDS}


def has_topic_overlap(summary_a: str, summary_b: str) -> bool:
    """Check if two summaries share enough significant terms to be same-topic."""
    terms_a = _tokenize_for_overlap(summary_a)
    terms_b = _tokenize_for_overlap(summary_b)
    shared = terms_a & terms_b
    return len(shared) >= TOPIC_OVERLAP_MIN_TERMS


# =============================================================================
# Keyword staleness detection (Amendment A5: word-boundary matching)
# =============================================================================


def _matches_indicator(text: str, indicators: list) -> str | None:
    """Check if text matches any indicator with word boundaries."""
    text_lower = text.lower()
    for ind in indicators:
        if " " in ind:
            # Phrase: use plain substring match
            if ind in text_lower:
                return ind
        else:
            # Single word: use word boundary
            if re.search(rf"\b{re.escape(ind)}\b", text_lower):
                return ind
    return None


def detect_keyword_staleness(existing_summary: str, new_summary: str) -> StalenessSignal | None:
    """Check if existing concept has stale indicators and new has completion."""
    stale_match = _matches_indicator(existing_summary, STALE_INDICATORS)
    completion_match = _matches_indicator(new_summary, COMPLETION_INDICATORS)

    if stale_match and completion_match:
        return StalenessSignal(
            is_stale=True,
            reason=f"'{stale_match}' in existing, '{completion_match}' in new",
            stale_indicator=stale_match,
            completion_indicator=completion_match,
            detection_method="keyword",
        )
    return None


def detect_status_transition(existing_summary: str, new_summary: str) -> str | None:
    """Check STATUS_TRANSITIONS pairs between existing and new summaries."""
    old_lower = existing_summary.lower()
    new_lower = new_summary.lower()

    for before_markers, after_markers, reason in STATUS_TRANSITIONS:
        old_matches = any(m in old_lower for m in before_markers)
        new_matches = any(m in new_lower for m in after_markers)
        if old_matches and new_matches:
            return reason
    return None


# =============================================================================
# Value Change Detection (STALE-PREMISE-001)
# =============================================================================

# Patterns that extract (old_value, new_value) pairs from decision summaries
_VALUE_CHANGE_PATTERNS = [
    re.compile(r"(?:from|was)\s+([\d.]+)\s*(?:to|→|->)\s*([\d.]+)", re.IGNORECASE),
    re.compile(r"([\d.]+)\s*→\s*([\d.]+)"),
    re.compile(
        r"(?:changed|zeroed|set|updated|reduced|increased)\s+\S+\s+(?:from\s+)?([\d.]+)\s*(?:to|→)\s*([\d.]+)",
        re.IGNORECASE,
    ),
]

# Context patterns: old value must appear near an operator or parameter-like word
_VALUE_CONTEXT_RE = re.compile(r"[*=:]\s*{val}|{val}\s*[*=:]|weight[_\s]*{val}")

# Diagnostic context: skip concepts that already know the value is old
_DIAGNOSTIC_CONTEXT = re.compile(
    r"\b(?:stale|old|previous|former|was|changed from|superseded)\b", re.IGNORECASE
)

# STALE-PREMISE-001-HARDEN: Subject extraction for value change attribution
_SUBJECT_EXTRACTORS = [
    re.compile(
        r"(?:zeroed|changed|set|updated|reduced|increased)\s+(.+?)\s+(?:from\s+)?\d",
        re.IGNORECASE,
    ),
    re.compile(r"(\S+(?:\s+\S+)?)\s+(?:changed\s+)?from\s+[\d.]", re.IGNORECASE),
]

# Qualifying nouns that indicate what a numeric value represents.
# If the existing concept qualifies the value with one of these AND it doesn't
# match the change subject, the value likely means something different.
_VALUE_QUALIFIERS = frozenset({
    "threshold", "accuracy", "target", "benchmark", "latency", "percentile",
    "error", "rate", "score", "weight", "coefficient", "factor", "contribution",
    "limit", "boundary", "cutoff", "minimum", "maximum", "ratio", "percentage",
    "throughput", "cache",
})


def _extract_change_subjects(new_summary: str) -> tuple[list[str], set[str]]:
    """Extract the subject of a value change (e.g., 'confidence weight')."""
    subjects: list[str] = []
    for pat in _SUBJECT_EXTRACTORS:
        for m in pat.finditer(new_summary):
            subj = m.group(1).strip().lower()
            subj = re.sub(r"^(?:retrieval-\d+:\s*)", "", subj)
            if len(subj) >= 3 and subj not in ("the", "its", "from"):
                subjects.append(subj)
    words = set()
    for s in set(subjects):
        words.update(w for w in s.split() if len(w) >= 4)
    return list(set(subjects)), words


def _has_anti_subject(
    existing_summary: str, new_summary: str, old_val: str
) -> bool:
    """Check if old_val in existing is qualified by a different domain than the change.

    Returns True if the local context around old_val contains a qualifier noun
    (like 'threshold', 'accuracy', 'cache') that does NOT appear in the change
    subject words. This prevents superseding 'p99=0.55s' when the change is
    about 'similarity weight 0.55'.
    """
    _, subject_words = _extract_change_subjects(new_summary)
    if not subject_words:
        return False
    for m in re.finditer(re.escape(old_val), existing_summary):
        start = max(0, m.start() - 40)
        end = min(len(existing_summary), m.end() + 40)
        ctx = existing_summary[start:end].lower()
        ctx_words = set(re.findall(r"\b\w{4,}\b", ctx))
        qualifiers_present = ctx_words & _VALUE_QUALIFIERS
        if qualifiers_present:
            matching = qualifiers_present & subject_words
            non_matching = qualifiers_present - subject_words
            if non_matching and not matching:
                return True
    return False


def detect_value_change(
    existing_summary: str, new_summary: str
) -> "StalenessSignal | None":
    """STALE-PREMISE-001: Detect value override patterns.

    When new_summary says "changed X from A to B" and existing_summary
    contains X*A or X=A, the existing concept is stale.

    Gauntlet amendments:
      - Requires contextual match (operator or shared keyword proximity)
      - Excludes diagnostic contexts (concepts already aware value changed)
      - Minimum old_value length of 2 chars to avoid single-digit noise
    """
    # Skip if existing concept already discusses the change diagnostically
    if _DIAGNOSTIC_CONTEXT.search(existing_summary):
        return None

    # Extract value change pairs from the new (decision) concept
    changes: list[tuple[str, str]] = []
    for pattern in _VALUE_CHANGE_PATTERNS:
        for match in pattern.finditer(new_summary):
            old_val, new_val = match.group(1), match.group(2)
            if len(old_val) >= 2 and old_val != new_val:
                changes.append((old_val, new_val))

    if not changes:
        return None

    # Extract shared keywords (words ≥4 chars appearing in both summaries)
    existing_words = set(re.findall(r"\b\w{4,}\b", existing_summary.lower()))
    new_words = set(re.findall(r"\b\w{4,}\b", new_summary.lower()))
    shared_keywords = existing_words & new_words - STOP_WORDS

    for old_val, new_val in changes:
        if old_val not in existing_summary:
            continue

        # --- STALE-PREMISE-001-HARDEN: Tiered gate system ---
        # Anti-subject check: does the value live in a different domain?
        anti_subject = _has_anti_subject(existing_summary, new_summary, old_val)

        # Gate 1A: Strong operator (*) — formula context, high confidence
        strong_re = re.compile(
            rf"\*\s*{re.escape(old_val)}|{re.escape(old_val)}\s*\*",
            re.IGNORECASE,
        )
        has_strong_op = bool(strong_re.search(existing_summary))

        # Gate 1B: Weak operator (=, :, "weight", array brackets)
        weak_re = re.compile(
            rf"[=:]\s*{re.escape(old_val)}|{re.escape(old_val)}\s*[=:]"
            rf"|weight\S*\s*{re.escape(old_val)}"
            rf"|{re.escape(old_val)}\s*\S*weight"
            rf"|\[\s*(?:[\d.,\s]*,\s*)?{re.escape(old_val)}"
            rf"|{re.escape(old_val)}\s*[,\]]",
            re.IGNORECASE,
        )
        has_weak_op = bool(weak_re.search(existing_summary))

        # Gate 2: Shared keyword proximity — old value within 50 chars of shared keyword
        has_keyword_ctx = False
        if shared_keywords:
            val_positions = [m.start() for m in re.finditer(re.escape(old_val), existing_summary)]
            for kw in shared_keywords:
                kw_positions = [m.start() for m in re.finditer(re.escape(kw), existing_summary.lower())]
                for vp in val_positions:
                    for kp in kw_positions:
                        if abs(vp - kp) <= 50:
                            has_keyword_ctx = True
                            break
                    if has_keyword_ctx:
                        break
                if has_keyword_ctx:
                    break

        # Gate 3: Subject binding — change subject words appear in existing
        _, subject_words = _extract_change_subjects(new_summary)
        existing_lower = existing_summary.lower()
        subj_matches = (
            sum(1 for w in subject_words if w in existing_lower)
            if subject_words
            else -1
        )
        has_subject = subj_matches >= 1
        no_subject_info = subj_matches == -1

        # --- Tiered decision ---
        tier = None
        # TIER A: Strong operator (*) + no anti-subject → formula, high confidence
        if has_strong_op and not anti_subject:
            tier = "TIER_A"
        # TIER B: Weak operator + subject binding + no anti-subject
        elif has_weak_op and has_subject and not anti_subject:
            tier = "TIER_B"
        # TIER C: Keyword proximity + subject binding + no anti-subject
        elif has_keyword_ctx and has_subject and not anti_subject:
            tier = "TIER_C"
        # TIER D: Weak op + keyword (no subject extractable) — conservative fallback
        elif has_weak_op and has_keyword_ctx and no_subject_info:
            tier = "TIER_D"

        if tier:
            return StalenessSignal(
                is_stale=True,
                reason=f"Value override ({tier}): '{old_val}' → '{new_val}'",
                stale_indicator=old_val,
                completion_indicator=new_val,
                detection_method="value_change",
            )

    return None


# =============================================================================
# Trigger 1: In-session stale relative detection
# =============================================================================


def check_for_stale_relatives(
    new_concept_id: str,
    new_summary: str,
    retrieval_engine,
    supersede_fn,
) -> StalenessResult:
    """Trigger 1: After new concept creation, search embedding neighbors for stale relatives.

    Called from _process_single_insight() after a concept is created in the <0.50 zone.
    Uses embedding search (NOT TF-IDF) to find concepts in the 0.30–0.85 band that
    the dedup system missed because TF-IDF scored them <0.50.

    Args:
        new_concept_id: ID of the just-created concept
        new_summary: Summary text of the new concept
        retrieval_engine: RetrievalEngine instance (has .search() with embedding)
        supersede_fn: Callable(old_id, new_id, reason) -> bool — typically _supersede_concept

    Returns:
        StalenessResult with counts and details of any resolutions
    """
    from app.storage import load_concept

    t0 = time.perf_counter()
    result = StalenessResult()

    try:
        # Step 1: Embedding search for neighbors
        # PERF FIX: Use search_lightweight instead of full search().
        # Full search() runs predictive_activation.preload_for_query() which
        # does O(N) load_concept calls across ALL concepts (N+1 query pattern).
        # With 2700 concepts this takes ~13s. Lightweight skips preload,
        # SAL multipliers, and governance scoring — staleness only needs
        # embedding similarity scores.
        neighbors = retrieval_engine.search_lightweight(new_summary, top_k=10, min_confidence=0.0)

        resolutions = 0
        for neighbor in neighbors:
            # Skip self
            if neighbor.concept_id == new_concept_id:
                continue

            # Step 2: Filter to embedding band 0.30–0.85
            # relevance_score is the embedding-based score from search()
            if neighbor.relevance_score < STALE_RELATIVE_EMBEDDING_MIN:
                continue
            if neighbor.relevance_score >= STALE_RELATIVE_EMBEDDING_MAX:
                # Above 0.85 — dedup should have caught this, skip
                continue

            # Step 3: Cap resolutions per turn
            if resolutions >= MAX_STALE_RESOLUTIONS_PER_TURN:
                logger.info(f"Staleness T1: Hit per-turn cap ({MAX_STALE_RESOLUTIONS_PER_TURN}), stopping early")
                break

            # Step 4: Load full concept for checks
            existing = load_concept(neighbor.concept_id, track_access=False)
            if not existing:
                continue

            # Skip already-superseded concepts
            if existing.summary.startswith("[SUPERSEDED]"):
                continue

            # Step 5: Check concept type eligibility
            concept_type = getattr(existing, "concept_type", None) or "observation"
            if STALENESS_ELIGIBLE_TYPES is not None and concept_type not in STALENESS_ELIGIBLE_TYPES:
                continue

            # Step 6: Recently-updated guard — don't supersede concepts
            # that were just updated (avoids churn)
            # DATA-031: Use content_updated_at (actual content change) with fallback
            # to updated_at. Prevents concepts that were merely accessed/touched from
            # bypassing staleness detection.
            updated_at = (
                getattr(existing, "content_updated_at", None)
                or getattr(existing, "updated_at", None)
                or getattr(existing, "created_at", "")
            )
            if updated_at:
                try:
                    ts = updated_at.replace("Z", "+00:00")
                    updated_dt = datetime.fromisoformat(ts)
                    cutoff = _utc_now() - timedelta(hours=RECENTLY_UPDATED_HOURS)
                    if _ensure_aware(updated_dt) > cutoff:
                        logger.debug(
                            f"Staleness T1: Skipping {neighbor.concept_id} — "
                            f"updated within last {RECENTLY_UPDATED_HOURS}h"
                        )
                        continue
                except (ValueError, TypeError):
                    pass  # Can't parse timestamp, proceed with checks

            # Step 7: Run detection — status transition first (higher signal)
            # Status transition does NOT require topic overlap gate because
            # the transition markers themselves establish topical connection
            # (empirically validated: real failing pair has 0 vocab overlap
            # but clear status transition "written at" → "committed as")
            transition_reason = detect_status_transition(existing.summary, new_summary)
            if transition_reason:
                logger.info(
                    f"Staleness T1: STATUS_TRANSITION — superseding "
                    f"'{neighbor.concept_id}' (emb={neighbor.relevance_score:.2f}): "
                    f"{transition_reason}"
                )
                success = supersede_fn(neighbor.concept_id, new_concept_id, transition_reason)
                if success:
                    result.concepts_superseded += 1
                    resolutions += 1
                    result.details.append(
                        {
                            "action": "superseded",
                            "old_id": neighbor.concept_id,
                            "new_id": new_concept_id,
                            "reason": transition_reason,
                            "method": "status_transition",
                            "embedding_score": neighbor.relevance_score,
                        }
                    )
                continue

            # Step 8: Fall back to keyword staleness — REQUIRES topic overlap gate
            # Keyword detection is the weaker signal, so we gate it with
            # topic overlap (Amendment A2) to prevent false positives
            if not has_topic_overlap(existing.summary, new_summary):
                continue

            keyword_signal = detect_keyword_staleness(existing.summary, new_summary)
            if keyword_signal and keyword_signal.is_stale:
                reason = f"Keyword staleness: {keyword_signal.reason}"
                logger.info(
                    f"Staleness T1: KEYWORD — superseding "
                    f"'{neighbor.concept_id}' (emb={neighbor.relevance_score:.2f}): "
                    f"{reason}"
                )
                success = supersede_fn(neighbor.concept_id, new_concept_id, reason)
                if success:
                    result.concepts_superseded += 1
                    resolutions += 1
                    result.details.append(
                        {
                            "action": "superseded",
                            "old_id": neighbor.concept_id,
                            "new_id": new_concept_id,
                            "reason": reason,
                            "method": "keyword",
                            "embedding_score": neighbor.relevance_score,
                        }
                    )
                continue

            # Step 9: STALE-PREMISE-001 — Value change detection
            # Catches "X=old_value" in existing when new says "changed X from old to new"
            value_signal = detect_value_change(existing.summary, new_summary)
            if value_signal and value_signal.is_stale:
                reason = f"Value change: {value_signal.reason}"
                logger.info(
                    f"Staleness T1: VALUE_CHANGE — superseding "
                    f"'{neighbor.concept_id}' (emb={neighbor.relevance_score:.2f}): "
                    f"{reason}"
                )
                success = supersede_fn(neighbor.concept_id, new_concept_id, reason)
                if success:
                    result.concepts_superseded += 1
                    resolutions += 1
                    result.details.append(
                        {
                            "action": "superseded",
                            "old_id": neighbor.concept_id,
                            "new_id": new_concept_id,
                            "reason": reason,
                            "method": "value_change",
                            "embedding_score": neighbor.relevance_score,
                        }
                    )

    except Exception as e:
        logger.error(f"Staleness T1: Error during check: {e}", exc_info=True)
        result.errors += 1

    result.time_ms = (time.perf_counter() - t0) * 1000
    if result.concepts_superseded > 0:
        logger.info(f"Staleness T1: Resolved {result.concepts_superseded} stale concepts in {result.time_ms:.1f}ms")
    return result


# =============================================================================
# Trigger 2: Session-end checkpoint reconciliation
# =============================================================================


def reconcile_checkpoint_concepts(
    retrieval_engine,
    supersede_fn,
) -> StalenessResult:
    """Trigger 2: At session end, cross-reference checkpoint done[] items against concepts.

    Loads active checkpoints, builds a synthetic summary from done[] items,
    searches for related concepts, and checks if any are stale relative to
    checkpoint progress.

    Called from end_session() between T3 reflection and checkpoint auto-save.

    Args:
        retrieval_engine: RetrievalEngine instance
        supersede_fn: Callable(old_id, new_id, reason) -> bool

    Returns:
        StalenessResult with counts and details
    """
    from app.storage import list_checkpoints, load_concept

    t0 = time.perf_counter()
    result = StalenessResult()

    try:
        # Step 1: Load active checkpoints
        checkpoints = list_checkpoints()
        active_cps = [
            cp for cp in checkpoints if cp.get("status") in ("active", "planning", "paused") and cp.get("done")
        ]

        if not active_cps:
            result.time_ms = (time.perf_counter() - t0) * 1000
            return result

        resolutions = 0

        for cp in active_cps:
            if resolutions >= MAX_STALE_RESOLUTIONS_PER_SESSION_END:
                break

            done_items = cp.get("done", [])
            if not done_items:
                continue

            # Step 2: Build synthetic completion summary from done items
            # This represents "what has been accomplished"
            completion_summary = "Completed: " + "; ".join(done_items[:10])

            # Step 3: Search for concepts related to this checkpoint's work
            # PERF FIX: Use search_lightweight to avoid O(N) preload_for_query.
            # Staleness only needs embedding similarity, not governance scoring.
            neighbors = retrieval_engine.search_lightweight(completion_summary, top_k=10, min_confidence=0.0)

            for neighbor in neighbors:
                if resolutions >= MAX_STALE_RESOLUTIONS_PER_SESSION_END:
                    break

                # Filter to embedding band
                if neighbor.relevance_score < STALE_RELATIVE_EMBEDDING_MIN:
                    continue
                if neighbor.relevance_score >= STALE_RELATIVE_EMBEDDING_MAX:
                    continue

                # Load full concept
                existing = load_concept(neighbor.concept_id, track_access=False)
                if not existing:
                    continue

                # Skip already-superseded
                if existing.summary.startswith("[SUPERSEDED]"):
                    continue

                # Topic overlap gate
                if not has_topic_overlap(existing.summary, completion_summary):
                    continue

                # Concept type check
                concept_type = getattr(existing, "concept_type", None) or "observation"
                if STALENESS_ELIGIBLE_TYPES is not None and concept_type not in STALENESS_ELIGIBLE_TYPES:
                    continue

                # Step 4: Check if existing concept has stale indicators
                # while the checkpoint done[] shows completion
                stale_match = _matches_indicator(existing.summary, STALE_INDICATORS)
                if not stale_match:
                    # Also try status transition against done items
                    transition_reason = detect_status_transition(existing.summary, completion_summary)
                    if transition_reason:
                        # For checkpoint reconciliation, we don't have a "new concept"
                        # to supersede with — we evolve the existing concept instead
                        reason = f"Checkpoint reconciliation: {transition_reason} (done: {', '.join(done_items[:3])})"
                        logger.info(f"Staleness T2: CHECKPOINT_TRANSITION — evolving '{neighbor.concept_id}': {reason}")
                        success = _evolve_stale_concept(neighbor.concept_id, existing.summary, done_items, reason)
                        if success:
                            result.concepts_staled += 1
                            resolutions += 1
                            result.details.append(
                                {
                                    "action": "evolved",
                                    "concept_id": neighbor.concept_id,
                                    "reason": reason,
                                    "method": "checkpoint_transition",
                                    "checkpoint_task_id": cp.get("task_id", "unknown"),
                                }
                            )
                    continue

                # Existing has stale indicators — check if checkpoint shows completion
                completion_match = None
                for done_item in done_items:
                    completion_match = _matches_indicator(done_item, COMPLETION_INDICATORS)
                    if completion_match:
                        break

                if not completion_match:
                    # Check if any done item itself implies completion of the existing concept
                    if has_topic_overlap(existing.summary, completion_summary):
                        # Done items overlap with the stale concept — treat as implicit completion
                        completion_match = "checkpoint_done_overlap"

                if completion_match:
                    reason = (
                        f"Checkpoint reconciliation: '{stale_match}' in concept, "
                        f"'{completion_match}' in done items "
                        f"(task: {cp.get('task_id', 'unknown')})"
                    )
                    logger.info(f"Staleness T2: CHECKPOINT_KEYWORD — evolving '{neighbor.concept_id}': {reason}")
                    success = _evolve_stale_concept(neighbor.concept_id, existing.summary, done_items, reason)
                    if success:
                        result.concepts_staled += 1
                        resolutions += 1
                        result.details.append(
                            {
                                "action": "evolved",
                                "concept_id": neighbor.concept_id,
                                "reason": reason,
                                "method": "checkpoint_keyword",
                                "checkpoint_task_id": cp.get("task_id", "unknown"),
                            }
                        )

    except Exception as e:
        logger.error(f"Staleness T2: Error during reconciliation: {e}", exc_info=True)
        result.errors += 1

    # --- Checkpoint auto-completion evaluation ---
    # GAUNTLET AMENDMENTS:
    #   FP1: Only auto-complete "active" or "paused" (not "planning")
    #   S2: Load full checkpoint for concept_refs before completing
    #   FP2: save_count >= 2 (strict, ensures at least one update cycle)
    try:
        from app.storage import complete_checkpoint
        from app.storage import load_checkpoint as _load_cp

        one_hour_ago = (_utc_now() - timedelta(hours=1)).isoformat()

        for cp in active_cps:
            cp_status = cp.get("status", "")
            if cp_status not in ("active", "paused"):
                continue  # FP1: skip "planning" — may not have next[] yet

            next_items = cp.get("next", [])
            active_item = cp.get("active", "")
            done_items = cp.get("done", [])
            save_count = cp.get("save_count", 1)
            updated_at = cp.get("updated_at", "")
            task_id = cp.get("task_id", "")

            if not next_items and not active_item and done_items and save_count >= 2 and updated_at < one_hour_ago:
                # S2: Load full checkpoint for concept_refs before completing
                cp_full = _load_cp(task_id=task_id)
                concept_refs = []
                if cp_full:
                    concept_refs = cp_full.get("concept_refs", [])

                complete_checkpoint(task_id)

                # Audit trace for auto-completion
                try:
                    from app.core.metrics_facade import create_trace

                    create_trace(
                        session_id="system_auto_complete",
                        trigger_type="checkpoint_auto_complete",
                        situation=f"Checkpoint '{task_id}' had empty next/active with {len(done_items)} done items",
                        intent="Auto-complete stale checkpoint at session end",
                        assessment=f"Completed: save_count={save_count}, done={done_items[:5]}",
                        justification="Heuristic: status in (active,paused), "
                        "next=[], active='', done non-empty, "
                        "save_count>=2, age>1h",
                        concept_refs=concept_refs,
                    )
                except Exception as trace_err:
                    logger.debug(f"Trace for auto-complete skipped: {trace_err}")

                result.details.append(
                    {
                        "action": "auto_completed",
                        "checkpoint_task_id": task_id,
                        "reason": f"next=[], active='', done has {len(done_items)} "
                        f"items, save_count={save_count}, age>1h",
                        "method": "checkpoint_auto_complete",
                    }
                )
                logger.info(f"Staleness T2: AUTO-COMPLETED checkpoint '{task_id}'")
    except Exception as ac_err:
        logger.debug(f"Checkpoint auto-completion skipped: {ac_err}")

    result.time_ms = (time.perf_counter() - t0) * 1000
    if result.concepts_staled > 0:
        logger.info(f"Staleness T2: Evolved {result.concepts_staled} stale concepts in {result.time_ms:.1f}ms")
    return result


# =============================================================================
# Helper: Evolve a stale concept with updated status (Trigger 2)
# =============================================================================


def _evolve_stale_concept(
    concept_id: str,
    existing_summary: str,
    done_items: list,
    reason: str,
) -> bool:
    """Evolve a stale concept to reflect checkpoint completion.

    Instead of superseding (which requires a new concept), this evolves the
    existing concept's summary to reflect that the work is now done.
    Used by Trigger 2 where we don't have a new concept to supersede with.
    """
    try:
        from app.cognitive.learning import evolve_concept
        from app.core.models import ConceptEvolution

        # Build updated summary: prepend [STALE-EVOLVED] and append status
        # Keep original summary for context but add completion note
        done_snippet = "; ".join(done_items[:3])
        new_summary = f"{existing_summary} [STATUS: Completed — {done_snippet}]"

        evolution = ConceptEvolution(
            concept_id=concept_id,
            new_summary=new_summary,
            confidence_change=-0.1,  # Slight reduction — concept is outdated
            new_evidence=[
                {
                    "source_type": "system",
                    "content": f"Staleness T2: {reason}",
                    "reliability_weight": 0.8,
                }
            ],
        )
        evolve_concept(evolution)
        logger.info(f"Staleness T2: Evolved stale concept '{concept_id}'")
        return True

    except Exception as e:
        logger.error(f"Staleness T2: Failed to evolve '{concept_id}': {e}")
        return False


# =============================================================================
# Trigger 2b: Session-end concept reconciliation (no checkpoint dependency)
# CONCEPT_LIFECYCLE_SPEC Layer 2
# =============================================================================


def reconcile_session_concepts(
    session_start_iso: str,
    retrieval_engine,
    supersede_fn,
) -> StalenessResult:
    """Trigger 2b: Session-end concept reconciliation without checkpoint dependency.

    Compares concepts created during this session to detect in-session
    status transitions (planned → implemented, proposed → committed, etc.).

    Does NOT require checkpoint done[] items. Uses session-internal
    concept sequencing within knowledge_areas as the signal.

    Args:
        session_start_iso: ISO datetime of session start
        retrieval_engine: RetrievalEngine instance
        supersede_fn: Callable(old_id, new_id, reason) -> bool

    Returns:
        StalenessResult with counts and details
    """
    from app.storage import load_recent_concepts

    t0 = time.perf_counter()
    result = StalenessResult()

    try:
        # Load ALL concepts created this session
        session_concepts = load_recent_concepts(
            since_iso=session_start_iso,
            limit=50,
            min_confidence=0.0,
        )

        if len(session_concepts) < 2:
            result.time_ms = (time.perf_counter() - t0) * 1000
            return result

        # Group by knowledge_area
        by_area: dict = {}
        for c in session_concepts:
            area = c.get("knowledge_area", "general")
            by_area.setdefault(area, []).append(c)

        resolutions = 0

        for area, concepts in by_area.items():
            if resolutions >= MAX_STALE_RESOLUTIONS_PER_SESSION_END:
                break
            if len(concepts) < 2:
                continue

            # Sort by created_at ascending (oldest first)
            concepts.sort(key=lambda c: c.get("created_at", ""))

            # Check each pair: older concept stale? newer concept completes it?
            for i, older in enumerate(concepts[:-1]):
                if resolutions >= MAX_STALE_RESOLUTIONS_PER_SESSION_END:
                    break

                older_summary = older.get("summary", "")
                # Skip if already superseded
                if older_summary.startswith("[SUPERSEDED]"):
                    continue

                # Concept type check
                concept_type = older.get("concept_type", "observation")
                if STALENESS_ELIGIBLE_TYPES is not None and concept_type not in STALENESS_ELIGIBLE_TYPES:
                    continue

                stale_match = _matches_indicator(older_summary, STALE_INDICATORS)
                if not stale_match:
                    continue

                # Check if ANY later concept in same area shows completion
                for newer in concepts[i + 1 :]:
                    newer_summary = newer.get("summary", "")
                    completion_match = _matches_indicator(newer_summary, COMPLETION_INDICATORS)
                    if completion_match and has_topic_overlap(older_summary, newer_summary):
                        reason = f"Session reconciliation: '{stale_match}' → '{completion_match}' within session"
                        success = supersede_fn(
                            older["concept_id"],
                            newer["concept_id"],
                            reason,
                        )
                        if success:
                            result.concepts_staled += 1
                            resolutions += 1
                            result.details.append(
                                {
                                    "action": "superseded",
                                    "old_id": older["concept_id"],
                                    "new_id": newer["concept_id"],
                                    "reason": reason,
                                    "method": "session_reconciliation",
                                }
                            )
                            logger.info(
                                f"Staleness T2b: SESSION_RECONCILIATION — "
                                f"superseded '{older['concept_id']}' with "
                                f"'{newer['concept_id']}': {reason}"
                            )
                        break  # Move to next older concept

        result.time_ms = (time.perf_counter() - t0) * 1000
        return result

    except Exception as e:
        logger.error(f"Staleness T2b: Session reconciliation failed: {e}")
        result.time_ms = (time.perf_counter() - t0) * 1000
        return result


# =============================================================================
# STABILITY-012: Factual Freshness Flagging
# =============================================================================

# Regex patterns for concrete factual references that can go stale.
_FACTUAL_PATTERNS = [
    ("file_path", re.compile(r"[~/][\w/.-]{3,}\.\w{1,10}")),
    ("version_string", re.compile(r"v\d+\.\d+(?:\.\d+)?")),
    ("semver", re.compile(r"\b\d+\.\d+\.\d+\b")),
    ("url", re.compile(r"https?://\S{5,}")),
    ("db_config_ref", re.compile(r"\b\w+\.(?:db|json|yaml|yml|toml|conf)\b")),
    (
        "numeric_count",
        re.compile(
            r"\b(\d{2,}(?:,\d{3})*)\s+(?:concepts?|items?|files?|entries|records?|rows?|events?|pairs?)\b",
            re.IGNORECASE,
        ),
    ),
]

# Exclude patterns that look like parameters/thresholds (not stale facts)
_FACTUAL_EXCLUDES = re.compile(
    r"(?:confidence|threshold|weight|factor|score|ratio|probability|default)\s*"
    r"(?:=|:|\bis\b|of)\s*\d",
    re.IGNORECASE,
)

FRESHNESS_STALE_DAYS = 14


def extract_factual_reference_hits(summary: str) -> list[dict]:
    """Return factual-looking references that may encode time-sensitive claims."""
    if not summary:
        return []

    hits = []
    for pattern_name, pattern in _FACTUAL_PATTERNS:
        matches = pattern.findall(summary)
        for match in matches:
            match_str = match if isinstance(match, str) else str(match)
            context_start = max(0, summary.find(match_str) - 30)
            context_end = min(len(summary), summary.find(match_str) + len(match_str) + 30)
            context = summary[context_start:context_end]
            if _FACTUAL_EXCLUDES.search(context):
                continue
            hits.append({"type": pattern_name, "value": match_str})
    return hits


def scan_factual_freshness(
    min_confidence: float = 0.7,
    stale_days: int = FRESHNESS_STALE_DAYS,
    include_always_activate: bool = True,
    limit: int = 50,
) -> list[dict]:
    """STABILITY-012: Scan high-confidence and AA concepts for stale factual references."""
    from app.storage import _db

    results = []
    cutoff = (_utc_now() - timedelta(days=stale_days)).isoformat()

    try:
        with _db() as conn:
            # DATA-020: Use content_updated_at for accurate freshness detection.
            # updated_at changes on every access/touch; content_updated_at only
            # changes when summary actually changes.
            query = """
                SELECT id, summary, confidence, updated_at, created_at,
                       always_activate, version, content_updated_at
                FROM concepts
                WHERE status = 'active'
                  AND (confidence >= ? OR always_activate = 1)
                  AND COALESCE(content_updated_at, updated_at, created_at) < ?
                ORDER BY confidence DESC
                LIMIT ?
            """
            rows = conn.execute(query, (min_confidence, cutoff, limit * 3)).fetchall()

        for row in rows:
            cid, summary, confidence, updated_at, created_at, is_aa, version, content_updated_at = row
            if not summary:
                continue
            if not is_aa and confidence < min_confidence:
                continue
            if not include_always_activate and is_aa and confidence < min_confidence:
                continue

            stale_refs = extract_factual_reference_hits(summary)
            if not stale_refs:
                continue

            # DATA-031: Use content_updated_at for accurate freshness display
            # (matches the COALESCE in the WHERE clause)
            last_change = content_updated_at or updated_at or created_at
            try:
                last_dt = _ensure_aware(
                    datetime.fromisoformat(
                        last_change.replace("Z", "+00:00") if isinstance(last_change, str) else str(last_change)
                    )
                )
                days_since = (_utc_now() - last_dt).total_seconds() / 86400
            except (ValueError, TypeError):
                days_since = stale_days + 1

            results.append(
                {
                    "concept_id": cid,
                    "summary_snippet": summary[:120],
                    "stale_refs": stale_refs[:5],
                    "days_since_evolution": round(days_since, 1),
                    "is_aa": bool(is_aa),
                    "confidence": confidence,
                    "version": version or "v1",
                }
            )

            if len(results) >= limit:
                break

    except Exception as e:
        logger.error(f"STABILITY-012: Factual freshness scan failed: {e}")

    if results:
        logger.info(f"STABILITY-012: Found {len(results)} concepts with potentially stale factual refs")

    return results



# --- DATA-057: Targeted stale technology sweep ---

def sweep_stale_technology_refs(
    dry_run: bool = True,
    technology_patterns: list[tuple[str, str]] | None = None,
) -> dict:
    """One-time + periodic sweep for concepts referencing eliminated technologies.

    Scans active concepts for technology references that are known to be outdated,
    and marks them as SUPERSEDED with a system-generated reason.

    Args:
        dry_run: If True, report but don't modify.
        technology_patterns: List of (search_term, supersession_reason) pairs.
            Defaults to known eliminated technologies.

    Returns:
        dict with counts and details.
    """
    from app.storage import _db

    if technology_patterns is None:
        technology_patterns = [
            ("Docker container", "Docker eliminated from Pith architecture"),
            ("docker-compose", "Docker eliminated from Pith architecture"),
            ("Dockerfile", "Docker eliminated from Pith architecture"),
            ("Node.js wrapper", "Node.js wrapper replaced by Python MCP SDK"),
            ("Node.js bridge", "Node.js wrapper replaced by Python MCP SDK"),
            ("server.js bridge", "server.js replaced by pith_mcp.py"),
        ]

    results = {
        "scanned": 0, "matched": 0, "superseded": 0,
        "skipped": 0, "details": [],
    }

    with _db() as conn:
        for pattern, reason in technology_patterns:
            rows = conn.execute(
                """SELECT id, summary, confidence, currency_status
                   FROM concepts
                   WHERE status = 'active'
                   AND currency_status NOT IN ('SUPERSEDED', 'STALE')
                   AND summary LIKE ?""",
                (f"%{pattern}%",),
            ).fetchall()

            for row in rows:
                cid, summary, conf, currency = row
                results["scanned"] += 1
                results["matched"] += 1

                # Skip if already contradicted (handled by contradiction system)
                if currency == "CONTRADICTED":
                    results["skipped"] += 1
                    continue

                if not dry_run:
                    conn.execute(
                        """UPDATE concepts
                           SET currency_status = 'SUPERSEDED',
                               updated_at = datetime('now')
                           WHERE id = ?""",
                        (cid,),
                    )
                    results["superseded"] += 1

                results["details"].append({
                    "id": cid,
                    "confidence": conf,
                    "pattern": pattern,
                    "reason": reason,
                    "action": "superseded" if not dry_run else "would_supersede",
                })

    logger.info(
        f"sweep_stale_technology_refs: scanned={results['scanned']}, "
        f"matched={results['matched']}, superseded={results['superseded']}, "
        f"dry_run={dry_run}"
    )
    return results
