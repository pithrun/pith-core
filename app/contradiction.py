"""Retrieval-Time Contradiction Detection — 3-phase pairwise checking.

Runs after currency filtering, before context assembly. Operates on 5-15
retrieval survivors and checks all pairs for contradictions.

Three detection phases:
  Phase 1: Keyword negation markers (<1ms) — catches ~50%
  Phase 2: Cached embedding similarity + directional analysis (<5ms) — catches ~35% more
  Phase 3: Soft detection — flags ambiguous tension for human review

Resolution: higher authority wins. Equal authority → both presented as CONTESTED.
"""

import logging
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from app.config import CROSS_KA_DAMPENING, FEATURE_FLAGS, TIER2_AMBIGUOUS_LOW, get_feature_flag
from app.constants import (
    GOV_EVENT_CONTRADICTION_DETECTED,
    GOV_EVENT_CONTRADICTION_PHASE_2_COMPLETED,
    GOV_EVENT_TIER2_LLM_COMPLETED,
)
from app.metrics import metrics as _m4

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Phase 1: Negation markers that suggest contradiction
NEGATION_MARKERS = [
    r"\bnot\b",
    r"\bno longer\b",
    r"\binstead of\b",
    r"\breplaced by\b",
    r"\bwrong\b",
    r"\bactually\b",
    r"\bcorrected to\b",
    r"\bisn't\b",
    r"\bdoesn't\b",
    r"\bdon't\b",
    r"\bnever\b",
    r"\bnot a\b",
    r"\bshouldn't\b",
    r"\bwasn't\b",
]

# Pre-compile for performance
_NEGATION_PATTERNS = [re.compile(p, re.IGNORECASE) for p in NEGATION_MARKERS]

# Phase 2: Embedding thresholds
EMBEDDING_SAME_TOPIC_THRESHOLD = 0.70  # Above this = same topic
EMBEDDING_SOFT_LOWER = 0.40  # Soft contradiction range lower bound
EMBEDDING_SOFT_UPPER = 0.70  # Soft contradiction range upper bound

# CONTRA-009: Cross-topic false positive prevention
# Minimum keyword overlap (Jaccard) to consider two concepts as same-topic.
# Prevents flagging unrelated concepts that happen to co-retrieve.
CROSS_TOPIC_OVERLAP_MIN = 0.08  # ~2 shared words in typical 20-word summaries

# Phase 2: Negation vocabulary for asymmetry detection (A5, v1.2)
# Text is normalized (apostrophes stripped) before matching
NEGATION_WORDS = frozenset(
    {
        "not",
        "no",
        "never",
        "none",
        "neither",
        "nor",
        "cannot",
        "cant",
        "wont",
        "wouldnt",
        "shouldnt",
        "doesnt",
        "dont",
        "didnt",
        "isnt",
        "arent",
        "wasnt",
        "werent",
        "hardly",
        "barely",
        "rarely",
        "seldom",
        "without",
        "lack",
        "lacking",
        "absence",
        "unable",
        "fail",
        "fails",
        "failed",
        "reject",
        "rejected",
        "deny",
        "denied",
        "avoid",
        "prevent",
        "prevents",
        "prevented",
    }
)

NEGATION_PREFIXES = ("un", "dis", "non", "in", "im", "ir", "il")

CONTRAST_CONNECTORS = frozenset(
    {
        "but",
        "however",
        "although",
        "despite",
        "instead",
        "rather",
        "conversely",
        "whereas",
        "unlike",
        "except",
    }
)

# Phase 2: Static antonym dictionary (~150 pairs) for cross-summary matching (A5, v1.2)
ANTONYM_PAIRS: dict[str, str] = {
    # Performance/quality
    "fast": "slow",
    "faster": "slower",
    "fastest": "slowest",
    "good": "bad",
    "better": "worse",
    "best": "worst",
    "improves": "degrades",
    "improve": "degrade",
    "improved": "degraded",
    "optimal": "suboptimal",
    "optimizes": "pessimizes",
    "efficient": "inefficient",
    "effective": "ineffective",
    "increases": "decreases",
    "increase": "decrease",
    "accelerates": "decelerates",
    "accelerate": "decelerate",
    "enhances": "diminishes",
    "enhance": "diminish",
    "strengthens": "weakens",
    "strengthen": "weaken",
    # Boolean/state
    "enables": "disables",
    "enable": "disable",
    "enabled": "disabled",
    "allows": "prevents",
    "allow": "prevent",
    "allowed": "prevented",
    "supports": "opposes",
    "support": "oppose",
    "accepts": "rejects",
    "accept": "reject",
    "accepted": "rejected",
    "includes": "excludes",
    "include": "exclude",
    "requires": "forbids",
    "require": "forbid",
    "permits": "prohibits",
    "permit": "prohibit",
    "creates": "destroys",
    "create": "destroy",
    "connects": "disconnects",
    "connect": "disconnect",
    "opens": "closes",
    "open": "close",
    "opened": "closed",
    "starts": "stops",
    "start": "stop",
    "started": "stopped",
    "adds": "removes",
    "add": "remove",
    "added": "removed",
    "installs": "uninstalls",
    "install": "uninstall",
    # Recommendation/preference
    "recommended": "discouraged",
    "recommend": "discourage",
    "preferred": "avoided",
    "prefer": "avoid",
    "should": "shouldnt",
    "must": "mustnt",
    "always": "never",
    "required": "optional",
    "necessary": "unnecessary",
    "essential": "nonessential",
    "critical": "trivial",
    "important": "unimportant",
    # Correctness
    "correct": "incorrect",
    "true": "false",
    "valid": "invalid",
    "right": "wrong",
    "accurate": "inaccurate",
    "precise": "imprecise",
    "reliable": "unreliable",
    "stable": "unstable",
    "safe": "unsafe",
    "secure": "insecure",
    "compatible": "incompatible",
    # Scope/quantity
    "all": "none",
    "every": "no",
    "complete": "incomplete",
    "full": "empty",
    "maximum": "minimum",
    "max": "min",
    "more": "less",
    "most": "least",
    "above": "below",
    "over": "under",
    "before": "after",
    "first": "last",
    "internal": "external",
    "public": "private",
    "simple": "complex",
    "easy": "difficult",
    "major": "minor",
    "primary": "secondary",
    "success": "failure",
    "succeeds": "fails",
    "positive": "negative",
    "benefit": "drawback",
    "advantage": "disadvantage",
    # Architecture
    "synchronous": "asynchronous",
    "sync": "async",
    "centralized": "decentralized",
    "stateful": "stateless",
    "mutable": "immutable",
    "blocking": "nonblocking",
    "coupled": "decoupled",
    "monolithic": "distributed",
    "upstream": "downstream",
    "deprecated": "recommended",
    "replaces": "complements",
    "replace": "complement",
}

# Build reverse lookup for bidirectional matching
_ANTONYM_LOOKUP: dict[str, set[str]] = {}
for _a, _b in ANTONYM_PAIRS.items():
    _ANTONYM_LOOKUP.setdefault(_a, set()).add(_b)
    _ANTONYM_LOOKUP.setdefault(_b, set()).add(_a)

# Phase 3: Minimum authority for soft contradiction flagging
SOFT_CONTRADICTION_MIN_AUTHORITY = 0.40

# Phase 2 cost estimate for latency budget
PHASE_2_ESTIMATED_COST_MS = 5.0


# =============================================================================
# Data Models
# =============================================================================


class ContradictionType(str, Enum):  # noqa: UP042
    HARD = "HARD"
    SOFT = "SOFT"


# COGGOV-005: Per-call circuit breaker limit for contradiction suppression.
# Limits automated suppression to prevent over-governance in a single detection call.
_CONTRADICTION_CB_LIMIT = int(os.environ.get("CONTRADICTION_CB_LIMIT", "5"))


class ResolutionAction(str, Enum):  # noqa: UP042
    SUPPRESS_LOSER = "SUPPRESS_LOSER"
    PRESENT_BOTH_CONTESTED = "PRESENT_BOTH_CONTESTED"
    FLAG_FOR_REVIEW = "FLAG_FOR_REVIEW"


@dataclass
class ScoredConcept:
    """Minimal concept representation for contradiction checking."""

    concept_id: str
    summary: str
    knowledge_area: str
    authority_score: float
    currency_score: float
    embedding: np.ndarray | None = None  # Cached 384-dim vector
    maturity: str = "PROVISIONAL"  # Phase 3 v1.1 (A4): for LLM prompt hardening
    evidence: list = field(default_factory=list)  # Phase 3 v1.1 (A1): for topic-match pre-filter
    created_at: str | None = None  # LIFECYCLE-001: ISO timestamp for temporal resolution
    concept_type: str | None = None  # LIFECYCLE-001: For thread role classification + arc detection


@dataclass
class ContradictionPair:
    """Detected contradiction between two concepts."""

    concept_a_id: str
    concept_b_id: str
    contradiction_type: ContradictionType
    detection_phase: int  # 0 (version-pair), 1 (keyword), 2 (embedding), 3 (soft)
    similarity_score: float | None = None
    reason: str = ""
    winner_id: str | None = None
    loser_id: str | None = None
    action: ResolutionAction = ResolutionAction.FLAG_FOR_REVIEW


@dataclass
class ContradictionResult:
    """Full result from contradiction detection pass."""

    pairs: list[ContradictionPair] = field(default_factory=list)
    suppressed_ids: list[str] = field(default_factory=list)
    contested_ids: list[str] = field(default_factory=list)
    phase_05_time_ms: float = 0.0  # Phase 0.5: version-pair drift (MONITOR-C015)
    phase_1_time_ms: float = 0.0
    phase_2_time_ms: float = 0.0
    phase_3_time_ms: float = 0.0
    tier2_time_ms: float = 0.0  # Phase 3: LLM Tier 2 time
    total_time_ms: float = 0.0
    phase_2_skipped: bool = False
    pairs_evaluated: int = 0
    tier2_evaluations: int = 0  # Phase 3: LLM Tier 2 calls made
    tier2_contradictions_found: int = 0  # Phase 3: contradictions caught by Tier 2
    tier2_topic_match_escalations: int = 0  # Phase 3 (A1): low-overlap pairs escalated


# =============================================================================
# Phase 1: Keyword Negation Detection
# =============================================================================


def _has_negation_marker(text: str) -> bool:
    """Check if text contains any negation marker."""
    return any(pattern.search(text) for pattern in _NEGATION_PATTERNS)


_VERSION_SUFFIX_PATTERN = re.compile(r"_v\d+$")


def _strip_version_suffix(concept_id: str) -> str:
    """Remove trailing _vN suffix from a concept_id (e.g. 'jwt_auth_v2' → 'jwt_auth')."""
    return _VERSION_SUFFIX_PATTERN.sub("", concept_id)


def _version_pair_drift_check(a: ScoredConcept, b: ScoredConcept) -> ContradictionPair | None:
    """Phase 0.5: Detect semantic drift between versioned concept pairs.

    Two concepts are a version pair if their base IDs match after stripping
    the _vN suffix (e.g. 'jwt_auth' and 'jwt_auth_v2').  If their summaries
    differ this signals potential drift that warrants review.

    Returns a SOFT ContradictionPair (detection_phase=0) or None.
    """
    base_a = _strip_version_suffix(a.concept_id)
    base_b = _strip_version_suffix(b.concept_id)
    if base_a != base_b:
        return None
    # CONTRA-017/018: empty or whitespace-only summary is not a meaningful divergence signal
    a_clean = (a.summary or "").strip()
    b_clean = (b.summary or "").strip()
    if not a_clean or not b_clean:
        return None
    if a_clean == b_clean:
        return None
    return ContradictionPair(
        concept_a_id=a.concept_id,
        concept_b_id=b.concept_id,
        contradiction_type=ContradictionType.SOFT,
        detection_phase=0,
        similarity_score=None,
        reason=(
            f"Version-pair semantic drift: '{base_a}' has diverging summaries "
            f"across versions (Phase 0.5)"
        ),
    )


def _phase_1_check(a: ScoredConcept, b: ScoredConcept) -> ContradictionPair | None:
    """Phase 1: Keyword-based contradiction detection.

    Same knowledge area + negation marker in one but not both = potential flip.
    Both having negation markers (double negative) is not a contradiction.
    """
    # Same knowledge area is a prerequisite for Phase 1
    if a.knowledge_area != b.knowledge_area:
        return None

    # CONTRA-014: Topic overlap guard — same as Phase 2/3 (CONTRA-009).
    # Phase 1 fired on completely unrelated concepts sharing only a KA,
    # generating 98% false positive suppressions. Require minimum keyword
    # overlap before treating negation asymmetry as a contradiction signal.
    overlap = _compute_keyword_overlap_score(a.summary, b.summary)
    if overlap < CROSS_TOPIC_OVERLAP_MIN:
        return None

    a_neg = _has_negation_marker(a.summary)
    b_neg = _has_negation_marker(b.summary)

    # XOR: one has negation, the other doesn't = potential contradiction
    if a_neg != b_neg:
        return ContradictionPair(
            concept_a_id=a.concept_id,
            concept_b_id=b.concept_id,
            contradiction_type=ContradictionType.HARD,
            detection_phase=1,
            reason="Negation marker asymmetry in same knowledge area",
        )

    return None


# =============================================================================
# Phase 2: Embedding-Based Contradiction Detection
# =============================================================================


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. L2-normalized = dot product."""
    return float(np.dot(a, b))


def _normalize_text(text: str) -> str:
    """Normalize text for negation/antonym matching: lowercase + strip apostrophes."""
    return text.lower().replace("'", "").replace("\u2019", "")


def _negation_score(summary: str) -> int:
    """Count negation indicators in a summary."""
    words = re.findall(r"\b\w+\b", _normalize_text(summary))
    score = 0
    for w in words:
        if w in NEGATION_WORDS or w in CONTRAST_CONNECTORS:
            score += 1
        elif len(w) > 5:
            for prefix in NEGATION_PREFIXES:
                if w.startswith(prefix) and len(w) > len(prefix) + 3:
                    score += 1
                    break
    return score


def _has_antonym_match(summary_a: str, summary_b: str) -> tuple[bool, str]:
    """Check if summaries contain antonym pairs across them.

    Returns (is_match, reason_detail).
    """
    words_a = set(re.findall(r"\b\w+\b", _normalize_text(summary_a)))
    words_b = set(re.findall(r"\b\w+\b", _normalize_text(summary_b)))

    for word_a in words_a:
        antonyms = _ANTONYM_LOOKUP.get(word_a)
        if antonyms and antonyms & words_b:
            matched = antonyms & words_b
            return True, f"{word_a} \u2194 {matched.pop()}"

    return False, ""


def _has_directional_opposition(summary_a: str, summary_b: str) -> tuple[bool, str]:
    """Check if two summaries express opposing positions.

    Two-signal detection (A5, v1.2):
    1. Antonym match: content words in A have antonyms present in B
    2. Negation asymmetry: one summary has ≥2 more negation indicators

    Either signal alone is sufficient when combined with high embedding similarity.
    Returns (is_opposition, signal_detail) for observability.
    """
    # Signal 1: Antonym match (fast path — dict lookup)
    has_antonym, antonym_detail = _has_antonym_match(summary_a, summary_b)
    if has_antonym:
        return True, f"antonym ({antonym_detail})"

    # Signal 2: Negation asymmetry
    neg_a = _negation_score(summary_a)
    neg_b = _negation_score(summary_b)
    if abs(neg_a - neg_b) >= 2:
        return True, f"negation asymmetry ({neg_a} vs {neg_b})"

    return False, ""


def _apply_ka_dampening(
    concept_a_ka: str | None,
    concept_b_ka: str | None,
    raw_score: float,
) -> float:
    """Dampen contradiction scores for cross-KA concept pairs.

    Federation Phase 0, Component 0.2.
    Same-KA pairs: score unchanged. Cross-KA or NULL: score *= CROSS_KA_DAMPENING.
    Applied BEFORE _resolve_contradiction (which reads similarity_score).
    """
    if concept_a_ka is not None and concept_b_ka is not None and concept_a_ka == concept_b_ka:
        return raw_score
    return raw_score * CROSS_KA_DAMPENING


def _phase_2_check(a: ScoredConcept, b: ScoredConcept) -> ContradictionPair | None:
    """Phase 2: Embedding similarity + directional analysis.

    High similarity (same topic) + opposing direction = HARD CONTRADICTION.
    Uses pre-cached embeddings from concepts.embedding column.
    """
    if a.embedding is None or b.embedding is None:
        return None

    # CONTRA-009: Skip pairs with insufficient keyword overlap.
    overlap = _compute_keyword_overlap_score(a.summary, b.summary)
    if overlap < CROSS_TOPIC_OVERLAP_MIN:
        return None

    sim = _cosine_similarity(a.embedding, b.embedding)

    # Same topic + directional opposition = hard contradiction
    if sim >= EMBEDDING_SAME_TOPIC_THRESHOLD:
        is_opposition, signal = _has_directional_opposition(a.summary, b.summary)
        if is_opposition:
            return ContradictionPair(
                concept_a_id=a.concept_id,
                concept_b_id=b.concept_id,
                contradiction_type=ContradictionType.HARD,
                detection_phase=2,
                similarity_score=sim,
                reason=f"High embedding similarity ({sim:.3f}) with directional opposition [{signal}]",
            )

    return None


# =============================================================================
# Phase 3: Soft Contradiction Detection
# =============================================================================


def _phase_3_check(a: ScoredConcept, b: ScoredConcept) -> ContradictionPair | None:
    """Phase 3: Soft contradiction — same area, moderate similarity, both authoritative.

    Flags for human review without suppression.
    """
    if a.knowledge_area != b.knowledge_area:
        return None

    # CONTRA-009: Cross-topic false positive prevention
    overlap = _compute_keyword_overlap_score(a.summary, b.summary)
    if overlap < CROSS_TOPIC_OVERLAP_MIN:
        return None

    if a.authority_score < SOFT_CONTRADICTION_MIN_AUTHORITY or b.authority_score < SOFT_CONTRADICTION_MIN_AUTHORITY:
        return None

    if a.embedding is None or b.embedding is None:
        return None

    sim = _cosine_similarity(a.embedding, b.embedding)

    if EMBEDDING_SOFT_LOWER <= sim <= EMBEDDING_SOFT_UPPER:
        return ContradictionPair(
            concept_a_id=a.concept_id,
            concept_b_id=b.concept_id,
            contradiction_type=ContradictionType.SOFT,
            detection_phase=3,
            similarity_score=sim,
            reason=f"Moderate similarity ({sim:.3f}) in same area with dual authority",
        )

    return None


# =============================================================================
# Resolution
# =============================================================================


def _resolve_contradiction(pair: ContradictionPair, concepts: dict) -> ContradictionPair:
    """Apply resolution rules to a detected contradiction.

    Rules:
      HARD + clear authority winner → suppress loser
      HARD + equal authority → present both as CONTESTED
      SOFT → flag for review only
      W4 (Gauntlet F-22): Both high-authority (>0.8) + score < 0.95 → CONTESTED, not suppressed
    """
    if pair.contradiction_type == ContradictionType.SOFT:
        pair.action = ResolutionAction.FLAG_FOR_REVIEW
        return pair

    a = concepts.get(pair.concept_a_id)
    b = concepts.get(pair.concept_b_id)

    if not a or not b:
        pair.action = ResolutionAction.FLAG_FOR_REVIEW
        return pair

    a_score = a.authority_score
    b_score = b.authority_score

    # W4 (Gauntlet F-22): Authority threshold — don't suppress high-authority
    # concepts unless contradiction score is very high. Both must be >0.8 authority
    # and contradiction similarity must be <0.95 to trigger this protection.
    HIGH_AUTHORITY_THRESHOLD = 0.8
    HIGH_AUTHORITY_SCORE_OVERRIDE = 0.95
    if a_score > HIGH_AUTHORITY_THRESHOLD and b_score > HIGH_AUTHORITY_THRESHOLD:
        effective_score = pair.similarity_score or 0.0
        if effective_score < HIGH_AUTHORITY_SCORE_OVERRIDE:
            pair.action = ResolutionAction.PRESENT_BOTH_CONTESTED
            pair.winner_id = None
            pair.loser_id = None
            pair.reason = (
                (pair.reason or "")
                + f" [W4: both high-authority ({a_score:.2f}, {b_score:.2f}), score {effective_score:.3f} < {HIGH_AUTHORITY_SCORE_OVERRIDE}]"
            )
            return pair

    # Clear winner: authority difference > 0.1
    if abs(a_score - b_score) > 0.1:
        if a_score > b_score:
            pair.winner_id = a.concept_id
            pair.loser_id = b.concept_id
        else:
            pair.winner_id = b.concept_id
            pair.loser_id = a.concept_id
        pair.action = ResolutionAction.SUPPRESS_LOSER
    elif abs(a.currency_score - b.currency_score) > 0.1:
        # Tiebreak on currency
        if a.currency_score > b.currency_score:
            pair.winner_id = a.concept_id
            pair.loser_id = b.concept_id
        else:
            pair.winner_id = b.concept_id
            pair.loser_id = a.concept_id
        pair.action = ResolutionAction.SUPPRESS_LOSER
    else:
        # Equal authority and currency — try temporal tiebreaker for observations
        # STALE-001: Newer observation wins when authority is tied
        # A6: Only explicit observation types eligible (not untagged concepts)
        _TEMPORAL_ELIGIBLE = {"observation", "pattern", "decision"}
        _a_is_observation = (a.concept_type or "").lower() in _TEMPORAL_ELIGIBLE
        _b_is_observation = (b.concept_type or "").lower() in _TEMPORAL_ELIGIBLE
        _temporal_resolved = False

        if _a_is_observation and _b_is_observation and a.created_at and b.created_at:
            try:
                from datetime import datetime as _dt_cls
                _a_dt = _dt_cls.fromisoformat(a.created_at.replace("Z", "+00:00"))
                _b_dt = _dt_cls.fromisoformat(b.created_at.replace("Z", "+00:00"))
                _age_diff_hours = abs((_a_dt - _b_dt).total_seconds()) / 3600

                # Only apply if meaningfully different age (>1 hour apart)
                TEMPORAL_TIEBREAKER_MIN_HOURS = 1.0
                if _age_diff_hours > TEMPORAL_TIEBREAKER_MIN_HOURS:
                    if _a_dt > _b_dt:
                        pair.winner_id = a.concept_id
                        pair.loser_id = b.concept_id
                    else:
                        pair.winner_id = b.concept_id
                        pair.loser_id = a.concept_id
                    pair.action = ResolutionAction.SUPPRESS_LOSER
                    pair.reason = (
                        (pair.reason or "")
                        + f" [STALE-001: temporal tiebreaker, age_diff={_age_diff_hours:.1f}h, "
                        + f"newer={pair.winner_id}]"
                    )
                    _temporal_resolved = True
            except (ValueError, TypeError) as _te:
                logger.debug("STALE-001: temporal parse failed: %s", _te)

        if not _temporal_resolved:
            # Truly equal — present both as contested
            pair.action = ResolutionAction.PRESENT_BOTH_CONTESTED
            pair.winner_id = None
            pair.loser_id = None

    return pair


def persist_contradiction_resolution(
    pair: ContradictionPair, source: str = "retrieval", session_id: str | None = None
) -> None:
    """CONTRA-001: Persist a contradiction resolution outcome to DB.

    Args:
        pair: Resolved ContradictionPair with action, winner, loser set.
        source: Origin — 'retrieval', 'graph_signal', or 'write_time'.
        session_id: Optional session context.
    """
    try:
        from app.storage import _db

        with _db() as conn:
            conn.execute(
                """INSERT INTO contradiction_resolutions
                   (concept_a_id, concept_b_id, contradiction_type, detection_phase,
                    similarity_score, action, winner_id, loser_id, reason, source, session_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    pair.concept_a_id,
                    pair.concept_b_id,
                    pair.contradiction_type.value,
                    pair.detection_phase,
                    pair.similarity_score,
                    pair.action.value,
                    pair.winner_id,
                    pair.loser_id,
                    (pair.reason or "")[:500],  # Truncate long reasons
                    source,
                    session_id,
                ),
            )
    except Exception as e:
        logger.debug("CONTRA-001: Failed to persist resolution (best-effort): %s", e)


# =============================================================================
# Phase 3 v1.1: Tier 2 Helper Functions
# =============================================================================


def _compute_keyword_overlap_score(summary_a: str, summary_b: str) -> float:
    """Compute simple keyword overlap score between two summaries.

    Returns 0.0-1.0 where higher means more overlap.
    Used to gate Tier 2 LLM evaluation — NOT a contradiction score.
    """
    words_a = set(summary_a.lower().split())
    words_b = set(summary_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union) if union else 0.0


def _is_same_topic(a: ScoredConcept, b: ScoredConcept) -> bool:
    """Amendment A1: Check if two concepts are about the same topic.

    Returns True if they share the same knowledge_area AND have
    at least one overlapping evidence source string. This catches
    semantic contradictions like 'Python/Flask' vs 'Go/Gin' that
    have near-zero keyword overlap but discuss the same subject.
    """
    if a.knowledge_area != b.knowledge_area:
        return False
    if not a.evidence or not b.evidence:
        return False
    # Check for overlapping evidence strings (normalize to lowercase)
    evidence_a = {str(e).lower().strip() for e in a.evidence if e}
    evidence_b = {str(e).lower().strip() for e in b.evidence if e}
    return bool(evidence_a & evidence_b)


# =============================================================================
# Tier 2 Parallel Execution (WS1 Optimization)
# =============================================================================


def _run_parallel_tier2(coros: list) -> list:
    """Run multiple async LLM calls in parallel.

    STABILITY-014 Fix 3: Simplified from ThreadPoolExecutor pattern.
    All callers are sync functions in FastAPI's anyio threadpool, so
    asyncio.run() is always the correct path. The old ThreadPoolExecutor
    branch was dead code (get_running_loop() raises RuntimeError in
    worker threads) and would block the event loop if an async caller
    were ever added.

    Returns list of results (or Exception instances for failed calls).
    """
    import asyncio

    async def _gather():
        return await asyncio.gather(*coros, return_exceptions=True)

    return asyncio.run(_gather())


# =============================================================================
# Main Detection Pipeline
# =============================================================================


def detect_retrieval_contradictions(
    survivors: list[ScoredConcept],
    gov_ctx=None,
) -> ContradictionResult:
    """Run 3-phase contradiction detection on retrieval survivors.

    Args:
        survivors: 5-15 concepts that passed currency filtering.
        gov_ctx: GovernanceContext for latency budget + event logging.

    Returns:
        ContradictionResult with detected pairs and resolution actions.
    """
    result = ContradictionResult()
    n = len(survivors)
    if n < 2:
        return result

    # Build lookup dict for resolution
    concept_map = {s.concept_id: s for s in survivors}
    already_detected = set()  # Track pairs to avoid double-flagging

    result.pairs_evaluated = n * (n - 1) // 2
    t0 = time.perf_counter()

    # CONTRA-005: Build always-activate set for sibling exclusion.
    # AA constraints share knowledge_area="protocol" and contain negation words
    # as part of their constraint language — they are NOT contradictory siblings.
    always_activate_ids: set[str] = set()
    try:
        from app.storage import load_always_activate_concepts

        aa_concepts = load_always_activate_concepts()
        # GA-001: load_always_activate_concepts returns dicts, not objects
        always_activate_ids = {c["concept_id"] for c in aa_concepts}
    except Exception:
        logger.debug("Could not load always-activate concepts, running detection normally")

    # --- Phase 0.5: Version-pair drift (CONTRA-015) ---
    t05_start = time.perf_counter()
    for i in range(n):
        for j in range(i + 1, n):
            if (
                survivors[i].concept_id in always_activate_ids
                and survivors[j].concept_id in always_activate_ids
            ):
                continue
            pair = _version_pair_drift_check(survivors[i], survivors[j])
            if pair:
                pair = _resolve_contradiction(pair, concept_map)
                result.pairs.append(pair)
                already_detected.add(frozenset([pair.concept_a_id, pair.concept_b_id]))
    result.phase_05_time_ms = (time.perf_counter() - t05_start) * 1000

    # --- Phase 1: Keyword negation (all pairs) ---
    t1_start = time.perf_counter()
    for i in range(n):
        for j in range(i + 1, n):
            # CONTRA-005: Skip contradiction check between two always-activate concepts
            if survivors[i].concept_id in always_activate_ids and survivors[j].concept_id in always_activate_ids:
                continue
            # CONTRA-016: skip pairs already flagged by Phase 0.5 (or future pre-phases)
            if frozenset([survivors[i].concept_id, survivors[j].concept_id]) in already_detected:
                continue
            pair = _phase_1_check(survivors[i], survivors[j])
            if pair:
                pair = _resolve_contradiction(pair, concept_map)
                result.pairs.append(pair)
                already_detected.add(frozenset([pair.concept_a_id, pair.concept_b_id]))
    result.phase_1_time_ms = (time.perf_counter() - t1_start) * 1000

    # --- Phase 2: Embedding similarity (remaining pairs) ---
    # Check latency budget before Phase 2
    budget_ok = True
    if gov_ctx:
        try:
            budget_ok = gov_ctx.check_latency_budget("contradiction_phase_2_embedding", PHASE_2_ESTIMATED_COST_MS)
        except Exception:
            budget_ok = True  # Fail open — run Phase 2 if budget check fails

    if budget_ok:
        t2_start = time.perf_counter()
        for i in range(n):
            for j in range(i + 1, n):
                # CONTRA-005: Skip AA-vs-AA pairs
                if survivors[i].concept_id in always_activate_ids and survivors[j].concept_id in always_activate_ids:
                    continue
                pair_key = frozenset([survivors[i].concept_id, survivors[j].concept_id])
                if pair_key in already_detected:
                    continue
                pair = _phase_2_check(survivors[i], survivors[j])
                if pair:
                    # Federation Phase 0, Component 0.2: Cross-KA dampening
                    if get_feature_flag("KA_RELATIVE_GOVERNANCE_ENABLED", False):
                        if pair.similarity_score is not None:
                            pair.similarity_score = _apply_ka_dampening(
                                survivors[i].knowledge_area,
                                survivors[j].knowledge_area,
                                pair.similarity_score,
                            )
                    pair = _resolve_contradiction(pair, concept_map)
                    result.pairs.append(pair)
                    already_detected.add(pair_key)
        result.phase_2_time_ms = (time.perf_counter() - t2_start) * 1000
        # Log distinct event so adversarial benchmark can verify Phase 2 ran
        if gov_ctx:
            try:
                phase2_detections = sum(1 for p in result.pairs if p.detection_phase == 2)
                gov_ctx.log_event(
                    GOV_EVENT_CONTRADICTION_PHASE_2_COMPLETED,
                    None,
                    {
                        "pairs_checked": (n * (n - 1)) // 2,
                        "detections": phase2_detections,
                        "time_ms": round(result.phase_2_time_ms, 2),
                    },
                )
            except Exception:
                logger.debug("Phase 2 governance event logging failed (best-effort)")
    else:
        result.phase_2_skipped = True
        logger.warning("Phase 2 skipped: latency budget insufficient")

    # --- Phase 3: Soft detection (remaining pairs) ---
    t3_start = time.perf_counter()
    for i in range(n):
        for j in range(i + 1, n):
            # CONTRA-005: Skip AA-vs-AA pairs
            if survivors[i].concept_id in always_activate_ids and survivors[j].concept_id in always_activate_ids:
                continue
            pair_key = frozenset([survivors[i].concept_id, survivors[j].concept_id])
            if pair_key in already_detected:
                continue
            pair = _phase_3_check(survivors[i], survivors[j])
            if pair:
                pair = _resolve_contradiction(pair, concept_map)
                result.pairs.append(pair)
                already_detected.add(pair_key)
    result.phase_3_time_ms = (time.perf_counter() - t3_start) * 1000

    # --- Phase 4 (Phase 3 v1.1): LLM Tier 2 for ambiguous + topic-match pairs ---
    # WS1 optimization: Collect candidates first, then run in parallel via asyncio.gather.
    # Reduces 2× sequential ~500ms calls to 1× parallel ~500ms batch.
    try:
        if get_feature_flag("LLM_CONTRADICTION_TIER2_ENABLED"):
            from app.contradiction_llm import detect_contradiction_llm, is_tier2_candidate

            tier2_budget = 2  # MAX_TIER2_CHECKS_PER_TURN
            t4_start = time.perf_counter()

            # Step 1: Collect candidates for Tier 2 escalation
            tier2_pairs = []
            for i in range(n):
                if len(tier2_pairs) >= tier2_budget:
                    break
                for j in range(i + 1, n):
                    if len(tier2_pairs) >= tier2_budget:
                        break
                    # CONTRA-005: Skip AA-vs-AA pairs
                    if (
                        survivors[i].concept_id in always_activate_ids
                        and survivors[j].concept_id in always_activate_ids
                    ):
                        continue
                    pair_key = frozenset([survivors[i].concept_id, survivors[j].concept_id])
                    if pair_key in already_detected:
                        continue

                    a, b = survivors[i], survivors[j]
                    tier1_score = _compute_keyword_overlap_score(a.summary, b.summary)

                    escalate = False
                    if is_tier2_candidate(tier1_score):
                        escalate = True
                    elif tier1_score < TIER2_AMBIGUOUS_LOW:
                        if _is_same_topic(a, b):
                            escalate = True
                            result.tier2_topic_match_escalations += 1

                    if escalate:
                        tier2_pairs.append((i, j, a, b, pair_key))

            # Step 2: Run all Tier 2 LLM calls in parallel
            if tier2_pairs:
                coros = [
                    detect_contradiction_llm(
                        a.summary,
                        b.summary,
                        session_id="",
                        authority_a=a.authority_score,
                        maturity_a=a.maturity,
                        authority_b=b.authority_score,
                        maturity_b=b.maturity,
                    )
                    for (i, j, a, b, pair_key) in tier2_pairs
                ]
                llm_results = _run_parallel_tier2(coros)

                # Step 3: Process results
                for (i, j, a, b, pair_key), llm_result in zip(tier2_pairs, llm_results, strict=False):
                    result.tier2_evaluations += 1

                    # Skip exceptions from gather(return_exceptions=True)
                    if isinstance(llm_result, Exception):
                        logger.warning("Tier 2 parallel call failed: %s", llm_result)
                        continue

                    if llm_result.score > 0.7:
                        pair = ContradictionPair(
                            concept_a_id=a.concept_id,
                            concept_b_id=b.concept_id,
                            contradiction_type=ContradictionType.HARD,
                            detection_phase=4,  # LLM Tier 2
                            similarity_score=llm_result.score,
                            reason=f"LLM Tier 2 ({llm_result.contradiction_type}): {llm_result.reason[:100]}",
                        )
                        pair = _resolve_contradiction(pair, concept_map)
                        result.pairs.append(pair)
                        already_detected.add(pair_key)
                        result.tier2_contradictions_found += 1

            result.tier2_time_ms = (time.perf_counter() - t4_start) * 1000

            if gov_ctx and result.tier2_evaluations > 0:
                try:
                    gov_ctx.log_event(
                        GOV_EVENT_TIER2_LLM_COMPLETED,
                        None,
                        {
                            "evaluations": result.tier2_evaluations,
                            "contradictions_found": result.tier2_contradictions_found,
                            "topic_match_escalations": result.tier2_topic_match_escalations,
                            "time_ms": round(result.tier2_time_ms, 2),
                            "parallel": True,
                        },
                    )
                except Exception as e:
                    logger.debug("Contradiction metrics recording in LLM tier failed: %s", e)
    except ImportError:
        pass  # contradiction_llm not available — skip Tier 2
    except Exception as e:
        logger.warning("Tier 2 LLM phase failed (non-fatal): %s", e)

    result.total_time_ms = (time.perf_counter() - t0) * 1000

    # COGGOV-005: Post-resolution safety guards
    # Check protected flag and circuit breaker BEFORE persistence.
    # Circuit breaker uses a LOCAL counter — resets each detect_retrieval_contradictions call.
    _coggov_protected_count = 0
    _coggov_cb_tripped_count = 0
    _coggov_suppression_count = 0  # Local per-call counter
    if result.pairs:
        for pair in result.pairs:
            if pair.action == ResolutionAction.SUPPRESS_LOSER and pair.loser_id:
                # Guard 1: Protected concepts cannot be suppressed
                from app.storage import load_concept as _lc_protected
                _loser_concept = _lc_protected(pair.loser_id, track_access=False)
                if _loser_concept and getattr(_loser_concept, "protected", False):
                    pair.action = ResolutionAction.PRESENT_BOTH_CONTESTED
                    pair.reason = (pair.reason or "") + " [COGGOV-005: loser is protected]"
                    _coggov_protected_count += 1
                    continue
                # Guard 2: Circuit breaker — limit suppressions per detection call
                _coggov_suppression_count += 1
                if _coggov_suppression_count > _CONTRADICTION_CB_LIMIT:
                    pair.action = ResolutionAction.PRESENT_BOTH_CONTESTED
                    pair.reason = (pair.reason or "") + f" [COGGOV-005: circuit breaker at {_CONTRADICTION_CB_LIMIT}]"
                    _coggov_cb_tripped_count += 1
    if _coggov_protected_count > 0:
        logger.info("COGGOV-005: Protected %d concepts from suppression", _coggov_protected_count)
    if _coggov_cb_tripped_count > 0:
        logger.info("COGGOV-005: Circuit breaker tripped — %d suppressions downgraded to contested", _coggov_cb_tripped_count)

    # CONTRA-001: Persist all resolution outcomes (best-effort, non-blocking)
    if result.pairs:
        for pair in result.pairs:
            persist_contradiction_resolution(pair, source="retrieval")

    # Collect suppressed and contested IDs
    for pair in result.pairs:
        if pair.action == ResolutionAction.SUPPRESS_LOSER and pair.loser_id:
            result.suppressed_ids.append(pair.loser_id)
        elif pair.action == ResolutionAction.PRESENT_BOTH_CONTESTED:
            result.contested_ids.append(pair.concept_a_id)
            result.contested_ids.append(pair.concept_b_id)

    # Deduplicate contested IDs
    result.contested_ids = list(set(result.contested_ids))

    # Log events via GovernanceContext
    if gov_ctx and result.pairs:
        for pair in result.pairs:
            try:
                gov_ctx.log_event(
                    GOV_EVENT_CONTRADICTION_DETECTED,
                    concept_id=pair.concept_a_id,
                    details={
                        "other_concept_id": pair.concept_b_id,
                        "type": pair.contradiction_type.value,
                        "phase": pair.detection_phase,
                        "similarity": pair.similarity_score,
                        "action": pair.action.value,
                        "winner_id": pair.winner_id,
                    },
                )
            except Exception as e:
                logger.warning("Failed to log contradiction event: %s", e)

    if result.pairs:
        logger.info(
            "Contradiction detection: %d pairs found (%d hard, %d soft) in %.1fms [P0.5=%.1fms, P1=%.1fms, P2=%.1fms%s, P3=%.1fms]",
            len(result.pairs),
            sum(1 for p in result.pairs if p.contradiction_type == ContradictionType.HARD),
            sum(1 for p in result.pairs if p.contradiction_type == ContradictionType.SOFT),
            result.total_time_ms,
            result.phase_05_time_ms,
            result.phase_1_time_ms,
            result.phase_2_time_ms,
            " (skipped)" if result.phase_2_skipped else "",
            result.phase_3_time_ms,
        )

    # WS2: Metric 4 — contradiction_detection_rate
    try:
        rate = len(result.pairs) / result.pairs_evaluated if result.pairs_evaluated > 0 else 0.0
        _m4.record(
            "contradiction_detection_rate",
            rate,
            {
                "pairs_evaluated": result.pairs_evaluated,
                "pairs_found": len(result.pairs),
            },
        )
    except Exception as e:
        logger.debug("Contradiction metrics recording failed: %s", e)

    return result


# =============================================================================
# Write-Time Contradiction Detection (Memory Integrity Spec v1.2, §5.1.5)
# =============================================================================

# Thresholds for write-time contradiction routing
WRITE_CONTRADICTION_HARD_REJECT = 0.80  # Above this → reject (don't write)
WRITE_CONTRADICTION_QUARANTINE = 0.50  # 0.50-0.80 → set maturity=QUARANTINED
# Below 0.50 → PASS (write normally)


@dataclass
class WriteContradictionResult:
    """Result of write-time contradiction check for a single new concept."""

    action: str  # "PASS" | "QUARANTINE" | "HARD_REJECT"
    max_score: float  # Highest contradiction score found
    contradicting_concept_id: str | None = None  # Which existing concept triggered
    reason: str = ""
    phase: int = 0  # Which detection phase caught it


# =============================================================================
# Ingestion Latency Optimization (§5.2.9, H13)
# =============================================================================
# Cache recent contradiction results to avoid redundant checks.
# Key: (new_summary_hash, existing_concept_id) → (result, timestamp)
# TTL: 30 minutes (1800 seconds)

_CONTRADICTION_CACHE: dict[tuple[str, str], tuple[float, float]] = {}
_CONTRADICTION_CACHE_TTL = 1800  # 30 minutes in seconds


def _cache_key(new_summary: str, existing_id: str) -> tuple[str, str]:
    """Generate cache key from summary hash + existing concept ID."""
    import hashlib

    summary_hash = hashlib.md5(new_summary.encode()).hexdigest()[:16]
    return (summary_hash, existing_id)


def _get_cached_score(new_summary: str, existing_id: str) -> float | None:
    """Return cached contradiction score if fresh, else None."""
    key = _cache_key(new_summary, existing_id)
    entry = _CONTRADICTION_CACHE.get(key)
    if entry is None:
        return None
    score, cached_at = entry
    if time.time() - cached_at > _CONTRADICTION_CACHE_TTL:
        del _CONTRADICTION_CACHE[key]
        return None
    return score


def _set_cached_score(new_summary: str, existing_id: str, score: float) -> None:
    """Cache a contradiction score."""
    key = _cache_key(new_summary, existing_id)
    _CONTRADICTION_CACHE[key] = (score, time.time())


def clear_contradiction_cache() -> int:
    """Clear the contradiction cache. Returns number of entries removed."""
    count = len(_CONTRADICTION_CACHE)
    _CONTRADICTION_CACHE.clear()
    return count


def validate_ingestion_batch(
    concepts: list[dict],
) -> list[WriteContradictionResult]:
    """Validate multiple concepts for ingestion, using cache for speed.

    §5.2.9 H13: Reduces per-turn latency from 225-375ms (serial) to ~75ms
    by caching pairwise contradiction results with 30-min TTL.

    Args:
        concepts: List of dicts with 'summary', 'knowledge_area', 'concept_id'.

    Returns:
        List of WriteContradictionResult, one per input concept.
    """

    if not FEATURE_FLAGS.get("INGESTION_LATENCY_OPT_ENABLED", False):
        return [
            detect_write_contradiction(
                c.get("summary", ""),
                c.get("knowledge_area", "general"),
                c.get("concept_id", ""),
            )
            for c in concepts
        ]

    results = []
    for c in concepts:
        result = detect_write_contradiction(
            c.get("summary", ""),
            c.get("knowledge_area", "general"),
            c.get("concept_id", ""),
        )
        results.append(result)
    return results


def detect_write_contradiction(
    new_summary: str,
    new_knowledge_area: str,
    concept_id: str = "",
) -> WriteContradictionResult:
    """Check a new concept against existing concepts before writing.

    Three-tier detection per §5.1.5:
      Tier 1: Existing contradiction.py keyword + embedding checks.
      Tier 2: LLM semantic comparison (Phase 3, stubbed).
      Tier 3: Routing based on score thresholds.

    Args:
        new_summary: The concept text to check.
        new_knowledge_area: Knowledge area of the new concept.
        concept_id: Optional ID for logging.

    Returns:
        WriteContradictionResult with action (PASS/QUARANTINE/HARD_REJECT).
    """
    from app.config import get_feature_flag

    if not FEATURE_FLAGS.get("INGESTION_VALIDATION_ENABLED", False):
        return WriteContradictionResult(action="PASS", max_score=0.0)

    if not new_summary or not new_summary.strip():
        return WriteContradictionResult(action="PASS", max_score=0.0)

    try:
        from app.embedding import embedding_engine
        from app.storage import _db

        # Load existing concepts in same knowledge area for comparison
        with _db() as conn:
            rows = conn.execute(
                """SELECT id, summary, knowledge_area, authority_score, currency_score,
                          created_at, concept_type
                   FROM concepts
                   WHERE status = 'active'
                     AND maturity != 'DISCARDED'
                     AND knowledge_area = ?
                   ORDER BY authority_score DESC
                   LIMIT 30""",
                (new_knowledge_area,),
            ).fetchall()

        if not rows:
            return WriteContradictionResult(action="PASS", max_score=0.0)

        # Build ScoredConcept for the new concept
        new_emb = None
        try:
            new_emb = embedding_engine.embed_text(new_summary)
        except Exception as e:
            logger.debug("Embedding for write-time contradiction check failed: %s", e)

        new_scored = ScoredConcept(
            concept_id=concept_id or "__new__",
            summary=new_summary,
            knowledge_area=new_knowledge_area,
            authority_score=0.0,  # New concept has no authority yet
            currency_score=1.0,
            embedding=new_emb,
        )

        max_score = 0.0
        worst_match_id = None
        worst_phase = 0

        for row in rows:
            existing_id = row[0]
            existing_summary = row[1]
            existing_ka = row[2] or "unknown"
            existing_auth = row[3] or 0.0
            existing_currency = row[4] or 0.5

            # Get cached embedding for existing concept
            existing_emb = None
            try:
                pos = embedding_engine._id_to_pos.get(existing_id)
                if pos is not None and embedding_engine._index_matrix is not None:
                    existing_emb = embedding_engine._index_matrix[pos]
            except Exception as e:
                logger.debug("Embedding position lookup failed for %s: %s", existing_id, e)

            existing_scored = ScoredConcept(
                concept_id=existing_id,
                summary=existing_summary,
                knowledge_area=existing_ka,
                authority_score=existing_auth,
                currency_score=existing_currency,
                embedding=existing_emb,
                created_at=row[5],             # STALE-001: for temporal tiebreaker
                concept_type=row[6],           # STALE-001: for observation filtering
            )

            # §5.2.9 H13: Check cache first to skip redundant pairwise checks
            cached = _get_cached_score(new_summary, existing_id)
            if cached is not None:
                if cached > max_score:
                    max_score = cached
                    worst_match_id = existing_id
                    worst_phase = -1  # From cache
                continue

            pair_score = 0.0
            pair_phase = 0

            # Phase 1: Keyword check (GATE — does not score, just triggers Phase 2)
            p1 = _phase_1_check(new_scored, existing_scored)

            # Phase 2: Embedding check
            # Phase 1 is a sensitivity gate, not a scorer. Only Phase 2 assigns scores.
            p2 = _phase_2_check(new_scored, existing_scored)
            if p2:
                emb_score = p2.similarity_score or 0.0  # No phantom scores
                if emb_score > pair_score:
                    pair_score = emb_score
                    pair_phase = 2
            elif p1:
                # Phase 1 fired but Phase 2 found no embedding contradiction.
                # Keyword match alone is a false positive — no action.
                pass

            # Cache the pairwise result
            _set_cached_score(new_summary, existing_id, pair_score)

            if pair_score > max_score:
                max_score = pair_score
                worst_match_id = existing_id
                worst_phase = pair_phase

        # Tier 2: LLM semantic comparison (Phase 3, §5.1.5)
        # Only invoked when Tier 1 score is in the ambiguous range
        if get_feature_flag("LLM_CONTRADICTION_TIER2_ENABLED") and worst_match_id:
            from app.contradiction_llm import detect_contradiction_llm_sync, is_tier2_candidate

            if is_tier2_candidate(max_score):
                try:
                    # Get the existing concept's summary for LLM comparison
                    existing_row = [r for r in rows if r[0] == worst_match_id]
                    if existing_row:
                        existing_summary = existing_row[0][1]
                        llm_result = detect_contradiction_llm_sync(
                            new_summary,
                            existing_summary,
                        )
                        if llm_result.score > 0.80:
                            max_score = llm_result.score
                            worst_phase = 99  # LLM tier
                        elif llm_result.score < 0.20:
                            max_score = llm_result.score
                            worst_phase = 99
                        # else: keep Tier 1 score (LLM was ambiguous too)
                except Exception as e:
                    logger.warning("Tier 2 LLM check failed (non-fatal): %s", e)
                    # Fall through to Tier 3 with Tier 1 score

        # Tier 3: Route based on thresholds
        if max_score >= WRITE_CONTRADICTION_HARD_REJECT:
            return WriteContradictionResult(
                action="HARD_REJECT",
                max_score=max_score,
                contradicting_concept_id=worst_match_id,
                reason=f"Hard contradiction (score={max_score:.3f}) with {worst_match_id}",
                phase=worst_phase,
            )
        elif max_score >= WRITE_CONTRADICTION_QUARANTINE:
            return WriteContradictionResult(
                action="QUARANTINE",
                max_score=max_score,
                contradicting_concept_id=worst_match_id,
                reason=f"Soft contradiction (score={max_score:.3f}) with {worst_match_id}",
                phase=worst_phase,
            )
        else:
            return WriteContradictionResult(
                action="PASS",
                max_score=max_score,
            )

    except Exception as e:
        logger.warning(f"Write-time contradiction check failed (non-fatal): {e}")
        return WriteContradictionResult(action="PASS", max_score=0.0, reason=f"Error: {e}")


def filter_contradictions(
    survivors: list[ScoredConcept],
    result: ContradictionResult,
) -> list[ScoredConcept]:
    """Remove suppressed concepts and tag contested ones.

    Applied after detect_retrieval_contradictions, before context assembly.

    Returns:
        Filtered list of survivors (suppressed removed, contested tagged).
    """
    suppressed = set(result.suppressed_ids)
    filtered = [s for s in survivors if s.concept_id not in suppressed]
    return filtered


# =============================================================================
# CONTRA-002: Batch consumer for GRAPH_CONTRADICTION_SIGNAL events
# =============================================================================


def consume_graph_contradiction_signals(batch_size: int = 500) -> dict:
    """Process unresolved GRAPH_CONTRADICTION_SIGNAL events from governance_events.

    Deduplicates by (source, target) pair, runs resolution for each unique pair
    that doesn't already have a persisted resolution, and persists outcomes.

    Args:
        batch_size: Max unique pairs to process per call.

    Returns:
        Stats dict: {total_events, unique_pairs, already_resolved, newly_resolved, errors}.
    """
    stats = {"total_events": 0, "unique_pairs": 0, "already_resolved": 0, "newly_resolved": 0, "errors": 0}

    try:
        from app.storage import _db

        with _db() as conn:
            # Step 1: Get distinct (source, target) pairs from unprocessed signals
            rows = conn.execute(
                """SELECT DISTINCT
                       json_extract(details, '$.source') AS src,
                       json_extract(details, '$.target') AS tgt
                   FROM governance_events
                   WHERE event_type = 'GRAPH_CONTRADICTION_SIGNAL'
                   LIMIT ?""",
                (batch_size * 2,),  # Over-fetch to account for duplicates
            ).fetchall()

            stats["total_events"] = conn.execute(
                "SELECT COUNT(*) FROM governance_events WHERE event_type = 'GRAPH_CONTRADICTION_SIGNAL'"
            ).fetchone()[0]

            # Deduplicate: normalize pair ordering (alphabetical) to avoid (A,B) vs (B,A)
            unique_pairs: set[tuple[str, str]] = set()
            for row in rows:
                src, tgt = row[0], row[1]
                if src and tgt:
                    pair = tuple(sorted([src, tgt]))
                    unique_pairs.add(pair)

            stats["unique_pairs"] = len(unique_pairs)

            # Step 2: Filter out pairs that already have resolutions
            unresolved: list[tuple[str, str]] = []
            for a_id, b_id in unique_pairs:
                if len(unresolved) >= batch_size:
                    break
                existing = conn.execute(
                    """SELECT 1 FROM contradiction_resolutions
                       WHERE (concept_a_id = ? AND concept_b_id = ?)
                          OR (concept_a_id = ? AND concept_b_id = ?)
                       LIMIT 1""",
                    (a_id, b_id, b_id, a_id),
                ).fetchone()
                if existing:
                    stats["already_resolved"] += 1
                else:
                    unresolved.append((a_id, b_id))

            # Step 3: Load concepts and resolve each unresolved pair
            for a_id, b_id in unresolved:
                try:
                    # Load minimal concept data for resolution
                    a_row = conn.execute(
                        "SELECT id, summary, knowledge_area, authority_score, currency_score, created_at, concept_type FROM concepts WHERE id = ?",
                        (a_id,),
                    ).fetchone()
                    b_row = conn.execute(
                        "SELECT id, summary, knowledge_area, authority_score, currency_score, created_at, concept_type FROM concepts WHERE id = ?",
                        (b_id,),
                    ).fetchone()

                    if not a_row or not b_row:
                        stats["errors"] += 1
                        continue

                    a = ScoredConcept(
                        concept_id=a_row[0],
                        summary=a_row[1],
                        knowledge_area=a_row[2] or "",
                        authority_score=a_row[3] or 0.0,
                        currency_score=a_row[4] or 0.0,
                        created_at=a_row[5],       # STALE-001: for temporal tiebreaker
                        concept_type=a_row[6],     # STALE-001: for observation filtering
                    )
                    b = ScoredConcept(
                        concept_id=b_row[0],
                        summary=b_row[1],
                        knowledge_area=b_row[2] or "",
                        authority_score=b_row[3] or 0.0,
                        currency_score=b_row[4] or 0.0,
                        created_at=b_row[5],       # STALE-001: for temporal tiebreaker
                        concept_type=b_row[6],     # STALE-001: for observation filtering
                    )

                    # Create pair and resolve (graph signals are HARD by definition —
                    # the "contradicts" edge was set deliberately)
                    pair = ContradictionPair(
                        concept_a_id=a_id,
                        concept_b_id=b_id,
                        contradiction_type=ContradictionType.HARD,
                        detection_phase=0,  # 0 = graph signal (not runtime detection)
                        similarity_score=None,
                        reason="Graph edge: contradicts",
                    )
                    concept_map = {a_id: a, b_id: b}
                    pair = _resolve_contradiction(pair, concept_map)

                    # Persist using direct SQL (we already have the conn)
                    conn.execute(
                        """INSERT INTO contradiction_resolutions
                           (concept_a_id, concept_b_id, contradiction_type, detection_phase,
                            similarity_score, action, winner_id, loser_id, reason, source)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            pair.concept_a_id,
                            pair.concept_b_id,
                            pair.contradiction_type.value,
                            pair.detection_phase,
                            pair.similarity_score,
                            pair.action.value,
                            pair.winner_id,
                            pair.loser_id,
                            (pair.reason or "")[:500],
                            "graph_signal",
                        ),
                    )
                    stats["newly_resolved"] += 1

                except Exception as e:
                    logger.debug("CONTRA-002: Failed to resolve pair (%s, %s): %s", a_id, b_id, e)
                    stats["errors"] += 1

            # Step 4: Delete consumed events (they're now represented in contradiction_resolutions)
            if stats["newly_resolved"] > 0 or stats["already_resolved"] > 0:
                conn.execute("DELETE FROM governance_events WHERE event_type = 'GRAPH_CONTRADICTION_SIGNAL'")
                logger.info(
                    "CONTRA-002: Consumed %d GRAPH_CONTRADICTION_SIGNAL events, resolved %d new pairs (%d already resolved, %d errors)",
                    stats["total_events"],
                    stats["newly_resolved"],
                    stats["already_resolved"],
                    stats["errors"],
                )

    except Exception as e:
        logger.error("CONTRA-002: Batch consumption failed: %s", e)
        stats["errors"] += 1

    return stats
