"""Correction Capture Protocol — detecting and recording user corrections.

Detects when users correct the agent using 4-layer heuristics with a
two-signal rule and 0.60 confidence threshold. Records corrections,
identifies affected concepts, and triggers governance recomputation.

Sync steps 1-5 run within-turn (5ms budget).
Async steps 6-7 are post-session (skill extraction, self-model update).
"""

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from app.core.constants import GOV_EVENT_CORRECTION_RECORDED
from app.core.datetime_utils import _utc_now_iso

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Detection confidence threshold — must exceed to record
CORRECTION_CONFIDENCE_THRESHOLD = 0.60

# Two-signal rule: need >= 2 matches, OR 1 match with confidence > this
HIGH_CONFIDENCE_SINGLE_SIGNAL = 0.80

# Layer 4: Activated concept comparison threshold
ACTIVATED_CONCEPT_DISTANCE_THRESHOLD = 0.60

# Positive sentiment words that negate correction detection
POSITIVE_SENTIMENT_WORDS = frozenset(
    [
        "fine",
        "good",
        "correct",
        "right",
        "great",
        "perfect",
        "exactly",
        "agreed",
        "yes",
        "true",
        "accurate",
        "nice",
        "awesome",
        "excellent",
        "ok",
        "okay",
        "yep",
        "yup",
    ]
)


# =============================================================================
# Error Classification Taxonomy
# =============================================================================


class ErrorCause(StrEnum):
    AUTHORITY_VIOLATION = "AUTHORITY_VIOLATION"
    RECENCY_BIAS = "RECENCY_BIAS"
    FRAMING_DRIFT = "FRAMING_DRIFT"
    SCOPE_CREEP = "SCOPE_CREEP"
    STALE_RETRIEVAL = "STALE_RETRIEVAL"
    MISSING_RETRIEVAL = "MISSING_RETRIEVAL"
    FABRICATION = "FABRICATION"
    UNCLASSIFIED = "UNCLASSIFIED"  # A2: fallback — excluded from benchmark ratio


class CorrectionType(StrEnum):
    FACTUAL = "factual"
    FRAMING = "framing"
    BEHAVIORAL = "behavioral"
    SCOPE = "scope"


# =============================================================================
# Detection Patterns
# =============================================================================

# Layer 1: Explicit negation patterns (confidence: 0.85)
EXPLICIT_NEGATION_PATTERNS = [
    (r"that's wrong", 0.85),
    (r"that is wrong", 0.85),
    (r"no,?\s+it's\b", 0.85),
    (r"no,?\s+it is\b", 0.85),
    (r"not\s+\w+\s+but\s+\w+", 0.85),
    (r"\bincorrect\b", 0.85),
    (r"you're wrong", 0.85),
    (r"that's not right", 0.85),
    (r"that's not correct", 0.85),
    (r"wrong[.!]", 0.85),
    (r"that's not how", 0.85),
    (r"that's not what", 0.85),
]

# Layer 2: Contradiction markers (confidence: 0.70)
CONTRADICTION_MARKERS = [
    (r"i already told you", 0.70),
    (r"we decided", 0.70),
    (r"remember that", 0.70),
    (r"stop saying", 0.70),
    (r"don't call it", 0.70),
    (r"it's not a\b", 0.70),
    (r"i said\b", 0.70),
    (r"as i mentioned", 0.70),
    (r"i've told you", 0.70),
    (r"how many times", 0.70),
]

# Layer 3: Frustration signals (confidence: 0.40, needs corroboration)
FRUSTRATION_SIGNALS = [
    (r"you should know", 0.40),
    (r"by now", 0.40),
    (r"again\??[!.]*$", 0.40),
    (r"sigh", 0.40),
]

# Pre-compile all patterns
_COMPILED_NEGATION = [(re.compile(p, re.IGNORECASE), c) for p, c in EXPLICIT_NEGATION_PATTERNS]
_COMPILED_CONTRADICTION = [(re.compile(p, re.IGNORECASE), c) for p, c in CONTRADICTION_MARKERS]
_COMPILED_FRUSTRATION = [(re.compile(p, re.IGNORECASE), c) for p, c in FRUSTRATION_SIGNALS]


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class DetectionSignal:
    """Single detection signal from a heuristic layer."""

    layer: int  # 1-4
    pattern: str
    confidence: float
    matched_text: str = ""


@dataclass
class CorrectionEvent:
    """Detected correction event before recording."""

    detection_confidence: float
    signals: list[DetectionSignal]
    corrected_claim: str = ""
    correct_claim: str = ""
    user_message: str = ""
    correction_type: CorrectionType = CorrectionType.FACTUAL
    error_cause: ErrorCause | None = None


@dataclass
class CorrectionRecord:
    """Recorded correction in the database."""

    id: str
    session_id: str
    correction_type: str
    error_cause: str | None
    corrected_claim: str
    correct_claim: str
    affected_concept_ids: list[str]
    detection_confidence: float
    created_at: str
    skill_extracted: bool = False


# =============================================================================
# Layer 1-3: Pattern-Based Detection
# =============================================================================


def _check_positive_sentiment_override(text: str, match_end: int) -> bool:
    """Check if text after a negation pattern contains positive sentiment.

    'no, it's fine' → True (not a correction)
    'no, it's wrong' → False (is a correction)
    """
    remaining = text[match_end:].strip().lower()
    words = remaining.split()[:5]  # Check next 5 words
    return any(w.strip(".,!?") in POSITIVE_SENTIMENT_WORDS for w in words)


def _detect_pattern_signals(message: str) -> list[DetectionSignal]:
    """Run layers 1-3 pattern detection on a user message."""
    signals = []
    msg_lower = message.lower()

    # Check for ALL CAPS (frustration indicator)
    has_caps = len(message) > 5 and message.upper() == message

    # Layer 1: Explicit negation
    for pattern, confidence in _COMPILED_NEGATION:
        match = pattern.search(msg_lower)
        if match:
            # Check positive sentiment override
            if not _check_positive_sentiment_override(msg_lower, match.end()):
                signals.append(
                    DetectionSignal(
                        layer=1,
                        pattern=pattern.pattern,
                        confidence=confidence,
                        matched_text=match.group(),
                    )
                )

    # Layer 2: Contradiction markers
    for pattern, confidence in _COMPILED_CONTRADICTION:
        match = pattern.search(msg_lower)
        if match:
            signals.append(
                DetectionSignal(
                    layer=2,
                    pattern=pattern.pattern,
                    confidence=confidence,
                    matched_text=match.group(),
                )
            )

    # Layer 3: Frustration signals
    for pattern, confidence in _COMPILED_FRUSTRATION:
        match = pattern.search(msg_lower)
        if match:
            conf = confidence
            if has_caps:
                conf = min(1.0, conf + 0.20)  # Caps boost
            signals.append(
                DetectionSignal(
                    layer=3,
                    pattern=pattern.pattern,
                    confidence=conf,
                    matched_text=match.group(),
                )
            )

    return signals


# =============================================================================
# Layer 4: Activated Concept Comparison
# =============================================================================


def _detect_activated_concept_drift(
    message: str,
    activated_concepts: list[dict[str, Any]],
    embedding_engine=None,
) -> DetectionSignal | None:
    """Layer 4: Compare user message against previously activated concepts.

    If the user's message in domain X has high embedding distance from
    concepts activated in the previous turn, this suggests implicit correction.

    This is an improvement over pure semantic drift — it compares against
    what was ACTUALLY used in the response, not all concepts in the area.

    Args:
        message: Current user message
        activated_concepts: Concepts activated in the previous turn
            (each dict should have 'concept_id', 'summary', 'embedding')
        embedding_engine: EmbeddingEngine instance for encoding

    Returns:
        DetectionSignal if drift detected, None otherwise
    """
    if not activated_concepts or embedding_engine is None:
        return None

    try:
        msg_embedding = embedding_engine.embed_text(message)
    except Exception:
        return None

    # Compare against each activated concept
    max_distance = 0.0
    most_distant_id = None

    for ac in activated_concepts:
        emb = ac.get("embedding")
        if emb is None:
            continue

        if isinstance(emb, bytes):
            emb = np.frombuffer(emb, dtype=np.float32)

        # Cosine similarity (dot product on L2-normalized)
        sim = float(np.dot(msg_embedding, emb))
        distance = 1.0 - sim

        if distance > max_distance:
            max_distance = distance
            most_distant_id = ac.get("concept_id", "unknown")

    if max_distance > ACTIVATED_CONCEPT_DISTANCE_THRESHOLD:
        return DetectionSignal(
            layer=4,
            pattern="activated_concept_drift",
            confidence=0.50,
            matched_text=f"drift={max_distance:.3f} from {most_distant_id}",
        )

    return None



# =============================================================================
# Layer 5: AI Self-Correction Detection (COGGOV-006)
# =============================================================================

# AI self-correction patterns in previous_response
# At 0.65-0.70 confidence, none pass two-signal rule solo (all < HIGH_CONFIDENCE_SINGLE_SIGNAL 0.80).
# Layer 5 signals CORROBORATE Layers 1-4, they don't fire independently.
AI_SELF_CORRECTION_PATTERNS = [
    (r"(?:I |we |my )(?:was|were|made a?) (?:wrong|incorrect|mistake)", 0.65),
    (r"root cause (?:is|was) (?:actually|not what)", 0.65),
    (r"(?:previous|earlier|original)\s+(?:analysis|conclusion|finding)\s+(?:was|is)\s+(?:wrong|incorrect|flawed)", 0.70),
    (r"(?:I|we) (?:incorrectly|erroneously|mistakenly)\s+(?:said|stated|claimed|concluded)", 0.70),
    (r"(?:revising|updating|correcting)\s+(?:my |the )?(?:earlier|previous|original)", 0.65),
]

_COMPILED_AI_SELF_CORRECTION = [
    (re.compile(p, re.IGNORECASE), c) for p, c in AI_SELF_CORRECTION_PATTERNS
]

# Max chars of previous_response to scan (budget protection)
AI_SELF_CORRECTION_SCAN_LIMIT = 2000


def _detect_ai_self_correction(previous_response: str) -> list[DetectionSignal]:
    """Layer 5: Detect AI self-correction patterns in previous_response.

    Scans the AI's last response for language indicating it discovered
    and corrected its own error (e.g., through tool use, RCA, debugging).

    Budget: <0.5ms for 2000 chars (regex only, no embedding).

    Args:
        previous_response: The AI's previous response text

    Returns:
        List of DetectionSignal from Layer 5 matches
    """
    if not previous_response or len(previous_response.strip()) < 20:
        return []

    from app.core.config import get_feature_flag
    if not get_feature_flag("COGGOV_006_AI_SELF_CORRECTION", True):
        return []

    # Scan first N chars only (budget protection)
    text = previous_response[:AI_SELF_CORRECTION_SCAN_LIMIT]
    signals = []

    for pattern, confidence in _COMPILED_AI_SELF_CORRECTION:
        match = pattern.search(text)
        if match:
            signals.append(
                DetectionSignal(
                    layer=5,
                    pattern=pattern.pattern,
                    confidence=confidence,
                    matched_text=match.group(),
                )
            )

    if signals:
        try:
            from app.core.metrics_facade import metrics
            metrics.record("coggov006_layer5_fires", 1)
            for s in signals:
                logger.info(
                    "COGGOV-006: Layer 5 signal: pattern=%s confidence=%.2f matched=%s",
                    s.pattern[:40], s.confidence, s.matched_text[:50],
                )
        except Exception:
            pass

    return signals


# =============================================================================
# Main Detection Pipeline
# =============================================================================


def detect_correction(
    message: str,
    activated_concepts: list[dict[str, Any]] | None = None,
    embedding_engine=None,
    previous_response: str | None = None,
) -> CorrectionEvent | None:
    """Detect if a user message contains a correction.

    Applies the two-signal rule:
      - Need >= 2 signals, OR
      - 1 signal with confidence > 0.80

    Final confidence must exceed 0.60 threshold.

    Args:
        message: The user's current message
        activated_concepts: Concepts from previous turn (for layer 4)
        embedding_engine: EmbeddingEngine for layer 4

    Returns:
        CorrectionEvent if correction detected, None otherwise
    """
    if not message or len(message.strip()) < 3:
        return None

    # Collect signals from all layers
    signals = _detect_pattern_signals(message)

    # Layer 4: Activated concept drift
    if activated_concepts:
        drift_signal = _detect_activated_concept_drift(message, activated_concepts, embedding_engine)
        if drift_signal:
            signals.append(drift_signal)

    # Layer 5: AI self-correction in previous_response (COGGOV-006)
    if previous_response:
        ai_signals = _detect_ai_self_correction(previous_response)
        signals.extend(ai_signals)

    if not signals:
        return None

    # Apply two-signal rule
    max_confidence = max(s.confidence for s in signals)
    signal_count = len(signals)

    passes_rule = signal_count >= 2 or max_confidence >= HIGH_CONFIDENCE_SINGLE_SIGNAL

    if not passes_rule:
        return None

    # Compute composite confidence
    # Use max confidence, boosted by additional signals
    composite = max_confidence
    if signal_count > 1:
        # Each additional signal adds diminishing boost
        for i, s in enumerate(sorted(signals, key=lambda x: x.confidence, reverse=True)):
            if i == 0:
                continue
            composite += s.confidence * (0.1 / i)
    composite = min(1.0, composite)

    if composite < CORRECTION_CONFIDENCE_THRESHOLD:
        return None

    return CorrectionEvent(
        detection_confidence=round(composite, 4),
        signals=signals,
        user_message=message,
    )


# =============================================================================
# Affected Concept Identification
# =============================================================================


def identify_affected_concepts(
    correction: CorrectionEvent,
    recent_activated: list[str],
    conn=None,
    embedding_engine=None,
) -> list[str]:
    """Find which concepts led to the error being corrected.

    Strategy:
      1. Start with recently activated concepts (most likely culprits)
      2. If correction mentions a specific topic, search by knowledge_area
      3. Embedding similarity between correction text and concept summaries

    Args:
        correction: The detected correction event
        recent_activated: Concept IDs activated in the previous turn
        conn: SQLite connection for concept lookups
        embedding_engine: For semantic matching

    Returns:
        List of affected concept IDs (typically 1-3)
    """
    affected = []

    # Strategy 1: Recent activated concepts are the most likely culprits
    if recent_activated:
        affected.extend(recent_activated[:3])  # Cap at 3 most relevant

    # If we have enough from activation context, don't over-search
    if len(affected) >= 2:
        return affected[:3]

    # Strategy 2: Embedding similarity against recent concepts
    if conn and embedding_engine and correction.user_message:
        try:
            msg_emb = embedding_engine.embed_text(correction.user_message)
            # Get recent concepts with embeddings
            rows = conn.execute(
                """SELECT id, embedding FROM concepts
                   WHERE embedding IS NOT NULL AND status != 'deleted'
                   ORDER BY updated_at DESC LIMIT 20"""
            ).fetchall()

            scored = []
            for row in rows:
                cid, emb_blob = row
                if cid in affected or emb_blob is None:
                    continue
                emb = np.frombuffer(emb_blob, dtype=np.float32)
                sim = float(np.dot(msg_emb, emb))
                scored.append((cid, sim))

            scored.sort(key=lambda x: x[1], reverse=True)
            for cid, sim in scored[:2]:
                if sim > 0.5 and cid not in affected:
                    affected.append(cid)
        except Exception as e:
            logger.warning("Embedding search for affected concepts failed: %s", e)

    return affected[:5]  # Hard cap at 5


# =============================================================================
# Error Cause Classification (§3.2 + Amendment A2)
# =============================================================================


def classify_error_cause(
    correction: "CorrectionEvent",
    affected_concept_ids: list[str],
    conn=None,
) -> ErrorCause:
    """Heuristic classifier for error cause. No LLM call.

    Runs BEFORE recording. Checks signals in priority order:
    1. STALE_RETRIEVAL: affected concept has low currency
    2. MISSING_RETRIEVAL: correction introduces concept not in activated set
    3. AUTHORITY_VIOLATION: affected concept below median authority
    4. SCOPE_CREEP: affected concept's knowledge_area differs from turn domain
    5. FABRICATION: no activated concept similar to corrected claim (guard: must not be in activated set)
    6. UNCLASSIFIED: fallback (excluded from benchmark ratio per A2)
    """
    if not conn or not affected_concept_ids:
        return ErrorCause.UNCLASSIFIED

    try:
        # Check affected concepts for stale currency
        for cid in affected_concept_ids[:3]:
            row = conn.execute(
                """SELECT currency_score, currency_status, authority_score, knowledge_area
                   FROM concepts WHERE id = ? AND is_current = 1""",
                (cid,),
            ).fetchone()
            if not row:
                continue

            currency_score = row[0] if row[0] is not None else 0.5
            currency_status = row[1] or "ACTIVE"
            authority_score = row[2] if row[2] is not None else 0.5

            # Signal 1: STALE_RETRIEVAL — low currency or explicitly STALE
            if currency_score < 0.30 or currency_status == "STALE":
                return ErrorCause.STALE_RETRIEVAL

            # Signal 3: AUTHORITY_VIOLATION — below median authority of activated set
            try:
                median_auth = conn.execute(
                    """SELECT AVG(authority_score) FROM concepts
                       WHERE is_current = 1 AND authority_score IS NOT NULL
                       AND updated_at > datetime('now', '-1 day')"""
                ).fetchone()[0]
                if median_auth and authority_score < median_auth * 0.7:
                    return ErrorCause.AUTHORITY_VIOLATION
            except Exception:
                pass

            # Signal 4: SCOPE_CREEP — knowledge_area mismatch
            concept_ka = row[3] or ""
            if concept_ka and correction.user_message:
                # Simple heuristic: if message mentions a domain keyword not in concept's area
                msg_lower = correction.user_message.lower()
                if concept_ka.lower() not in msg_lower and len(concept_ka) > 3:
                    # Check if there's a different, more relevant area
                    try:
                        better_match = conn.execute(
                            """SELECT DISTINCT knowledge_area FROM concepts
                               WHERE is_current = 1 AND knowledge_area != ?
                               AND summary LIKE ?
                               LIMIT 1""",
                            (concept_ka, f"%{msg_lower[:30]}%"),
                        ).fetchone()
                        if better_match:
                            return ErrorCause.SCOPE_CREEP
                    except Exception:
                        pass

        # Signal 5: FABRICATION — no activated concept matches corrected claim
        # Guard: only if we have a corrected_claim to check against
        if correction.corrected_claim and len(correction.corrected_claim) > 10:
            # Check if any affected concept has summary similar to the claim
            has_backing = False
            claim_words = set(correction.corrected_claim.lower().split())
            for cid in affected_concept_ids[:3]:
                try:
                    srow = conn.execute(
                        "SELECT summary FROM concepts WHERE id = ? AND is_current = 1",
                        (cid,),
                    ).fetchone()
                    if srow and srow[0]:
                        summary_words = set(srow[0].lower().split())
                        overlap = len(claim_words & summary_words) / max(len(claim_words), 1)
                        if overlap > 0.3:
                            has_backing = True
                            break
                except Exception:
                    pass
            if not has_backing:
                return ErrorCause.FABRICATION

    except Exception as e:
        logger.warning("Error cause classification failed: %s", e)

    # P3-5: Keyword fallback — examine correction text when heuristic signals miss
    try:
        claim_text = f"{correction.corrected_claim or ''} {correction.correct_claim or ''}".lower()
        KEYWORD_SIGNALS = {
            ErrorCause.STALE_RETRIEVAL: ["outdated", "stale", "old", "superseded", "deprecated", "no longer"],
            ErrorCause.FABRICATION: ["wrong", "incorrect", "inaccurate", "false", "fabricat", "hallucin", "made up"],
            ErrorCause.MISSING_RETRIEVAL: ["missing", "not found", "didn't know", "unaware", "omitted"],
            ErrorCause.SCOPE_CREEP: ["irrelevant", "off topic", "wrong context", "misframed", "scope"],
        }
        for cause, keywords in KEYWORD_SIGNALS.items():
            if any(kw in claim_text for kw in keywords):
                return cause
    except Exception:
        pass

    return ErrorCause.UNCLASSIFIED


# =============================================================================
# Correction Recording (Sync Steps 1-5)
# =============================================================================


def record_correction(
    correction: CorrectionEvent,
    affected_concept_ids: list[str],
    session_id: str,
    conn=None,
    gov_ctx=None,
    previous_response: str | None = None,
) -> CorrectionRecord | None:
    """Record a correction and trigger governance recomputation.

    Sync steps (within turn, 5ms budget):
      1. Create CorrectionRecord in corrections table
      2. Evolve affected concepts (set CONTESTED, add correction evidence)
      3. Recompute authority_score for affected concepts
      4. Recompute currency_score for affected concepts
      5. Update anti-terms for affected decisions (feeds Layer 3)

    Args:
        correction: Detected correction event
        affected_concept_ids: Concepts that led to the error
        session_id: Current session ID
        conn: SQLite connection
        gov_ctx: GovernanceContext for event logging

    Returns:
        CorrectionRecord if successfully recorded, None otherwise
    """
    if not conn:
        logger.warning("No DB connection — correction not recorded")
        return None

    t0 = time.perf_counter()
    now = _utc_now_iso()
    record_id = str(uuid.uuid4())[:12]

    # §3.2: Auto-classify error cause before recording
    if correction.error_cause is None:
        correction.error_cause = classify_error_cause(correction, affected_concept_ids, conn)

    # COGGOV-012: Claim extraction removed (0% hit rate on 16 production corrections).
    # user_message is stored directly for future backtesting (Path D+a).

    record = CorrectionRecord(
        id=record_id,
        session_id=session_id,
        correction_type=correction.correction_type.value,
        error_cause=correction.error_cause.value if correction.error_cause else None,
        corrected_claim=correction.corrected_claim,
        correct_claim=correction.correct_claim,
        affected_concept_ids=affected_concept_ids,
        detection_confidence=correction.detection_confidence,
        created_at=now,
    )

    try:
        # Step 1: Insert correction record
        # FIX-1a: Let AUTOINCREMENT handle id. Add required concept_id column.
        concept_id = affected_concept_ids[0] if affected_concept_ids else "unknown"
        cursor = conn.execute(
            """INSERT INTO corrections
               (concept_id, session_id, correction_type, error_cause, corrected_claim,
                correct_claim, affected_concept_ids, detection_confidence,
                created_at, skill_extracted, user_message)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                concept_id,
                record.session_id,
                record.correction_type,
                record.error_cause,
                record.corrected_claim,
                record.correct_claim,
                json.dumps(record.affected_concept_ids),
                record.detection_confidence,
                record.created_at,
                False,
                correction.user_message[:1000] if correction.user_message else "",
            ),
        )
        # FIX-1a(A1): Read back AUTOINCREMENT id so callers get valid correction_id
        record.id = cursor.lastrowid

        # Step 1.5: COGGOV-013 — Correction-triggered supersession (BEFORE Step 2)
        # Must run before Step 2 because Step 2 sets updated_at, which triggers
        # the 1-second rate limiter in evolve_concept_nonlossy (Gauntlet F1).
        _coggov013_supersession_result = False
        from app.core.config import get_feature_flag as _get_ff
        if _get_ff("COGGOV_013_CORRECTION_SUPERSESSION", False):
            try:
                _coggov013_supersession_result = _execute_correction_supersession(
                    correction=correction,
                    affected_concept_ids=affected_concept_ids,
                    record_id=record.id,
                    session_id=session_id,
                    conn=conn,
                )
            except Exception as _sup_err:
                logger.warning("COGGOV-013: Supersession failed (non-fatal): %s", _sup_err)

        # Step 2: Evolve affected concepts — mark CONTESTED
        # Note: if COGGOV-013 superseded the concept, it's now is_current=0.
        # Step 2 will still run but the UPDATE will affect 0 rows (concept not found
        # with is_current=1), which is the correct behavior.
        # FIX-1b(A2): Write change_type/change_reason into data JSON blob,
        # not as SQL columns (which don't exist on concepts table).
        # authority.py reads change_type from data JSON at line 116.
        for cid in affected_concept_ids:
            try:
                # Read current data JSON to merge change_type into it
                _row = conn.execute("SELECT data FROM concepts WHERE id = ? AND is_current = 1", (cid,)).fetchone()
                if _row:
                    try:
                        _cdata = json.loads(_row[0]) if _row[0] else {}
                    except (json.JSONDecodeError, TypeError):
                        _cdata = {}
                    _cdata["change_type"] = "contradiction_flag"
                    _cdata["change_reason"] = (
                        f"User correction detected (confidence={correction.detection_confidence:.2f})"
                    )
                    # KA-006: Route through write gateway for column sync
                    from app.storage import update_concept_data

                    update_concept_data(
                        conn, cid, _cdata, extra_sets="currency_status = ?", extra_params=("CONTESTED",)
                    )
                else:
                    # Concept not found — just try setting currency_status
                    conn.execute(
                        """UPDATE concepts
                           SET currency_status = 'CONTESTED',
                               updated_at = ?
                           WHERE id = ?""",
                        (now, cid),
                    )
            except Exception as e:
                logger.warning("Failed to mark concept %s as CONTESTED: %s", cid, e)

        # Step 3: Recompute authority for affected concepts
        try:
            from app.authority import batch_compute_authority

            batch_compute_authority(conn, affected_concept_ids)
        except Exception as e:
            logger.warning("Authority recompute failed for corrections: %s", e)

        # Step 4: Recompute currency for affected concepts
        try:
            from app.governance.currency import batch_compute_currency

            batch_compute_currency(conn, affected_concept_ids)
        except Exception as e:
            logger.warning("Currency recompute failed for corrections: %s", e)

        # Step 5: Anti-term generation is handled by prediction_error.py
        # which reads from the corrections table. No action needed here.

        # Step 6: COGGOV-013 — Mark cascade_complete if supersession succeeded
        from app.core.config import get_feature_flag
        if get_feature_flag("COGGOV_013_CORRECTION_SUPERSESSION", False):
            if _coggov013_supersession_result:
                conn.execute(
                    "UPDATE corrections SET cascade_complete = 1 WHERE id = ?",
                    (record.id,),
                )

        conn.commit()

        # COGGOV-007/012: Queue async evolution with user_message (Path D+a: zero-extraction)
        try:
            queue_correction_evolution(
                affected_concept_ids=affected_concept_ids,
                user_message=correction.user_message or "",
                correction_id=record.id,
                session_id=session_id,
            )
        except Exception as _evo_err:
            logger.warning("COGGOV-007: Evolution queue failed (non-fatal): %s", _evo_err)

        # Wave 4b: Set has_correction=True on affected concepts [FIX F3]
        for cid in affected_concept_ids:
            try:
                conn.execute(
                    "UPDATE concepts SET data = json_set(data, '$.has_correction', 1) WHERE id = ?",
                    (cid,),
                )
            except Exception:
                pass  # Best-effort

        # Wave 4b: Create correction trace + resolve predictions
        try:
            from app.core.metrics_facade import create_trace, resolve_predictions_for_concept

            create_trace(
                session_id=session_id,
                trigger_type="correction",
                situation=f"User correction detected: {correction.corrected_claim[:100] if correction.corrected_claim else ''}",
                intent="Record and apply user correction",
                assessment=f"Affected {len(affected_concept_ids)} concept(s)",
                justification=f"Correction type: {correction.correction_type.value}",
                concept_refs=affected_concept_ids,
            )
            for cid in affected_concept_ids:
                resolve_predictions_for_concept(cid, outcome="corrected", outcome_source="correction")
        except Exception as e:
            logger.debug(f"Wave 4b: correction trace/prediction skipped: {e}")

        if conn:
            try:
                conn.commit()
            except Exception:
                pass

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Correction recorded: id=%s, confidence=%.2f, affected=%d concepts, %.1fms",
            record.id,
            record.detection_confidence,
            len(affected_concept_ids),
            elapsed_ms,
        )

        # Log governance event
        if gov_ctx:
            try:
                gov_ctx.log_event(
                    GOV_EVENT_CORRECTION_RECORDED,
                    concept_id=affected_concept_ids[0] if affected_concept_ids else None,
                    details={
                        "correction_id": record.id,
                        "detection_confidence": record.detection_confidence,
                        "affected_count": len(affected_concept_ids),
                        "correction_type": record.correction_type,
                        "sync_time_ms": round(elapsed_ms, 2),
                    },
                )
            except Exception:
                pass

        return record

    except Exception as e:
        logger.error("Failed to record correction: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass
        return None


# =============================================================================

# =============================================================================
# Claim Extraction (COGGOV-007)
# =============================================================================

# Patterns to extract what was wrong and what's correct from correction context
_CLAIM_EXTRACTION_PATTERNS = [
    # Pattern 1: "X is actually Y" -> correct=Y
    # COGGOV-011/A1: tightened prefix from .{5,40} to (?:the |that )\w[\w\s]{3,35}
    re.compile(
        r"(?:it(?:'s| is| was)|this (?:is|was)|(?:the |that )\w[\w\s]{3,35} (?:is|was)) (?:actually|not .+? but) ([^.!?\n]{10,200})",
        re.IGNORECASE,
    ),
    # Pattern 2: "not X, it's Y" -> corrected=X, correct=Y
    # COGGOV-011: expanded separators to include dashes/em-dashes and "but"
    re.compile(
        r"not ([^.!?\n]{5,100})[,;\u2014\u2013\-]\s*(?:it(?:'s| is)|rather|instead|but) ([^.!?\n]{5,200})",
        re.IGNORECASE,
    ),
    # Pattern 3: "the root cause is X (not Y)"
    re.compile(
        r"root cause (?:is|was) ([^.!?\n]{10,200})(?:\s*\(not (.{5,100})\))?",
        re.IGNORECASE,
    ),
]


def _extract_correction_claims(
    message: str,
    previous_response: str | None = None,
) -> tuple[str, str]:
    """Extract corrected_claim and correct_claim from correction context.

    Heuristic-only. Scans user message and AI previous_response for
    structured correction language.

    Returns:
        (corrected_claim, correct_claim) -- either or both may be empty
    """
    corrected = ""
    correct = ""

    # Combine sources, preferring previous_response (AI corrections are more structured)
    texts = []
    if previous_response:
        texts.append(previous_response[:2000])
    if message:
        texts.append(message[:500])

    for text in texts:
        for pattern in _CLAIM_EXTRACTION_PATTERNS:
            match = pattern.search(text)
            if match:
                groups = match.groups()
                if len(groups) >= 2 and groups[1]:
                    corrected = groups[0].strip()[:200]  # groups[0] = after "not" = wrong thing
                    correct = groups[1].strip()[:200]    # groups[1] = after "rather" = right thing
                elif len(groups) >= 1:
                    correct = groups[0].strip()[:200]
                if correct:
                    break
        if correct:
            break

    return corrected, correct


# =============================================================================
# COGGOV-013: Correction-Triggered Supersession
# =============================================================================

# F8: Module-level constants for correction supersession
CORRECTION_CONFIDENCE_FLOOR = 0.4  # F2: Don't let corrected version rank below typical concepts
MAX_CORRECTION_SUPERSESSIONS = 3  # Match COGGOV-007 limit


def _execute_correction_supersession(
    correction: CorrectionEvent,
    affected_concept_ids: list[str],
    record_id: int | str,
    session_id: str,
    conn,
) -> bool:
    """COGGOV-013: Supersede stale concepts after user correction.

    Creates a new version of each affected concept via nonlossy evolution,
    with the correction evidence merged. Old version gets is_current=0,
    superseded_by=new_version_id. No LLM call — uses the user's correction
    text directly.

    Returns True if at least one concept was successfully superseded.
    """
    if not affected_concept_ids:
        return False

    # CI-029: importlib lazy loader — avoids governance→cognitive static import (Contract 4).
    # Same pattern as DEBT-240 quarantine→contradiction precedent.
    import importlib
    evolve_concept_nonlossy = importlib.import_module("app.cognitive.nonlossy").evolve_concept_nonlossy

    # F5: Metrics for observability
    try:
        from app.core.metrics_facade import metrics as _metrics
    except Exception:
        _metrics = None

    any_success = False
    now = _utc_now_iso()

    for cid in affected_concept_ids[:MAX_CORRECTION_SUPERSESSIONS]:
        try:
            row = conn.execute(
                "SELECT id, summary, data, confidence "
                "FROM concepts WHERE id = ? AND is_current = 1",
                (cid,),
            ).fetchone()
            if not row:
                logger.debug("COGGOV-013: Concept %s not found or not current, skipping", cid)
                continue

            old_id, old_summary, old_data_raw, old_confidence = row

            try:
                old_data = json.loads(old_data_raw) if old_data_raw else {}
            except (json.JSONDecodeError, TypeError):
                old_data = {}

            # Build correction annotation for the summary
            correction_note = ""
            if correction.correct_claim:
                correction_note = f" [CORRECTED {now[:10]}: {correction.correct_claim[:200]}]"
            elif correction.user_message:
                correction_note = f" [CORRECTED {now[:10]}: per user correction]"

            new_summary = old_summary + correction_note
            if len(new_summary) > 800:
                new_summary = old_summary[:600] + "..." + correction_note

            # Build new_data dict for evolve_concept_nonlossy API:
            # Signature: evolve_concept_nonlossy(concept_id, new_data, conn) -> str | None
            new_confidence = max(old_confidence if old_confidence else 0.5, CORRECTION_CONFIDENCE_FLOOR)

            evolution_data = {
                "summary": new_summary,
                "confidence": new_confidence,
                "new_evidence": [
                    f"User correction ({now[:10]}): "
                    f"{correction.correct_claim[:300] if correction.correct_claim else 'corrected by user'}"
                ],
                "maturity": "ESTABLISHED",  # F3: bypass W9 content guard for corrections
                "correction_supersession": {
                    "correction_id": str(record_id),
                    "original_concept_id": old_id,
                    "correction_type": correction.correction_type.value if correction.correction_type else "factual",
                    "superseded_at": now,
                    "session_id": session_id,
                },
            }

            # F5: Record supersession attempt metric
            if _metrics:
                _metrics.record("coggov013_supersession_attempted", 1.0)

            # Call nonlossy evolution — preserves old version as is_current=0
            evolve_result = evolve_concept_nonlossy(
                concept_id=old_id,
                new_data=evolution_data,
                conn=conn,
            )

            if evolve_result:
                any_success = True
                if _metrics:
                    _metrics.record("coggov013_supersession_succeeded", 1.0)
                logger.info(
                    "COGGOV-013: Superseded %s to new version (correction_id=%s)",
                    old_id, record_id,
                )
            else:
                logger.warning(
                    "COGGOV-013: evolve_concept_nonlossy returned None for %s "
                    "(may be rate-limited or flag-gated)",
                    cid,
                )

        except Exception as e:
            logger.warning("COGGOV-013: Failed to supersede concept %s: %s", cid, e)

    return any_success


# =============================================================================
# Async Correction Evolution (COGGOV-007)
# =============================================================================


def queue_correction_evolution(
    affected_concept_ids: list[str],
    user_message: str,
    correction_id: int | str,
    session_id: str,
) -> None:
    """Queue non-destructive evidence-append for CONTESTED concepts.

    COGGOV-012 Path D+a: Zero-extraction, no relevance gate.
      - NEVER rewrites summary during correction
      - NEVER re-indexes embeddings
      - Appends user_message snippet as evidence to data JSON
      - No claim extraction (0% hit rate on production corrections)
      - No relevance gate (user_message embeddings have ~0.1-0.2 cosine to concepts;
        identify_affected_concepts already provides relevance filtering)
      - Stores raw user_message for future backtesting

    Future: Reflection pipeline (separate spec) handles summary revision.
    Note: COGGOV-013 correction-triggered supersession is now handled in
    record_correction() Step 1.5 via _execute_correction_supersession().
    This function continues as evidence-append only.

    Args:
        affected_concept_ids: Concepts to evolve (capped at 3)
        user_message: Raw correction message from user
        correction_id: ID of the correction record
        session_id: Current session ID
    """
    # MONITOR-132: correction evolution funnel counters
    try:
        from app.core.metrics_facade import metrics as _evo_m
    except Exception:
        _evo_m = None

    if not affected_concept_ids:
        if _evo_m:
            _evo_m.record("coggov007_evolution_skipped", 1.0, {"reason": "no_concept_ids"})
        return

    from app.core.config import get_feature_flag
    if not get_feature_flag("COGGOV_007_CORRECTION_EVOLUTION", False):
        if _evo_m:
            _evo_m.record("coggov007_evolution_skipped", 1.0, {"reason": "feature_flag_off"})
        return

    if _evo_m:
        _evo_m.record("coggov007_evolution_attempted", float(len(affected_concept_ids[:3])))

    try:
        from app.storage import db_immediate

        with db_immediate() as conn:
            for cid in affected_concept_ids[:3]:  # Cap at 3
                try:
                    row = conn.execute(
                        "SELECT data FROM concepts WHERE id = ? AND is_current = 1",
                        (cid,),
                    ).fetchone()
                    if not row:
                        continue

                    import json as _json
                    try:
                        data = _json.loads(row[0]) if row[0] else {}
                    except (TypeError, _json.JSONDecodeError):
                        data = {}

                    corrections_list = data.get("correction_evidence", [])

                    # Idempotency: skip if this correction_id already recorded
                    if any(e.get("correction_id") == str(correction_id) for e in corrections_list):
                        logger.debug(
                            "COGGOV-012: Skipping duplicate correction_id=%s for concept %s",
                            correction_id, cid,
                        )
                        continue

                    # COGGOV-012 D+a: Evidence append with user_message snippet
                    corrections_list.append({
                        "correction_id": str(correction_id),
                        "user_message": user_message[:500],
                        "session_id": session_id,
                        "timestamp": _utc_now_iso(),
                    })
                    data["correction_evidence"] = corrections_list[-5:]  # Keep last 5

                    # NO summary rewrite, NO embedding re-index
                    from app.storage import update_concept_data
                    update_concept_data(conn, cid, data)

                    logger.info(
                        "COGGOV-012: Appended evidence to concept %s (correction_id=%s)",
                        cid, correction_id,
                    )
                except Exception as e:
                    logger.warning("COGGOV-012: Failed to evolve concept %s: %s", cid, e)

        # MONITOR-128: record evidence appends to metrics table
        _evidence_appends = len(affected_concept_ids[:3])
        try:
            from app.core.metrics_facade import metrics as _corr_m
            if _evidence_appends > 0:
                _corr_m.record("coggov012_evidence_appends", float(_evidence_appends))
                # MONITOR-132: evolution success counter
                _corr_m.record("coggov007_evolution_succeeded", float(_evidence_appends))
        except Exception:
            pass

    except Exception as e:
        logger.warning("COGGOV-012: Correction evolution failed: %s", e)

