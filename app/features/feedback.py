"""
FEEDBACK-001: L1 Retrieval Utility Signal

Measures whether activated concepts were actually used in the LLM's response.
Heuristic-only (no LLM call), target <10ms for 14 concepts.

Scoring algorithm (v1):
  1. Keyword overlap (0-0.4): TF-IDF top keywords from concept summary vs response
  2. Knowledge area mention (0-0.2): KA terms appear in response
  3. Concept ID/summary reference (0-0.2): Direct substring match
  4. Position signal (0-0.2): Higher-ranked concepts that appear get bonus

Thresholds:
  >= 0.4 → USED
  >= 0.15 → PARTIAL
  < 0.15 → UNUSED
"""

import logging
import re
import time
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)

# --- Constants ---
UTIL_THRESHOLD_USED = 0.4
UTIL_THRESHOLD_PARTIAL = 0.15

# Minimum keyword length to avoid matching common words
MIN_KEYWORD_LEN = 4

# Stop words for keyword extraction (lightweight, no NLTK needed)
_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "this", "that", "these",
    "those", "it", "its", "they", "them", "their", "we", "our", "you",
    "your", "he", "she", "his", "her", "and", "but", "or", "nor", "not",
    "no", "so", "if", "then", "than", "too", "very", "just", "about",
    "above", "after", "again", "all", "also", "any", "because", "before",
    "between", "both", "by", "down", "each", "few", "for", "from",
    "get", "got", "here", "how", "in", "into", "more", "most", "new",
    "now", "of", "off", "on", "one", "only", "other", "out", "over",
    "own", "per", "put", "said", "same", "see", "some", "such", "take",
    "tell", "there", "to", "two", "up", "use", "used", "using", "what",
    "when", "where", "which", "while", "who", "whom", "why", "with",
    "concept", "pith", "none", "null", "true", "false", "default",
})

_FEEDBACK_LOCK_TIMEOUT_COUNTS: Counter[str] = Counter()


def get_feedback_lock_timeout_counts() -> dict[str, int]:
    """Return in-process optional feedback DB lock timeout counts by operation."""
    return dict(_FEEDBACK_LOCK_TIMEOUT_COUNTS)


def _record_feedback_lock_timeout(operation: str) -> int:
    """Increment the in-process lock-timeout counter for optional feedback paths."""
    _FEEDBACK_LOCK_TIMEOUT_COUNTS[operation] += 1
    return _FEEDBACK_LOCK_TIMEOUT_COUNTS[operation]


def _feedback_db(operation: str):
    """Return a short-budget DB context for optional feedback writes."""
    from app.core.config import get_feedback_db_lock_timeout_s
    import app.storage as storage

    return storage._db(timeout_s=get_feedback_db_lock_timeout_s(), operation=operation)


def _extract_keywords(text: str, top_n: int = 5) -> list[str]:
    """Extract top N keywords from text using simple frequency heuristic."""
    words = re.findall(r'[a-z_][a-z0-9_]{3,}', text.lower())
    words = [w for w in words if w not in _STOP_WORDS and len(w) >= MIN_KEYWORD_LEN]

    # Frequency count, prefer longer words as tiebreaker
    freq: dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1

    sorted_words = sorted(freq.items(), key=lambda x: (x[1], len(x[0])), reverse=True)
    return [w for w, _ in sorted_words[:top_n]]


def _score_single_concept(
    concept_summary: str,
    concept_ka: str,
    concept_id: str,
    response_lower: str,
    response_words: set[str],
    activation_rank: int,
    total_activated: int,
) -> dict[str, Any]:
    """Score a single concept's utilization against the response."""
    score = 0.0
    components: dict[str, float] = {}

    # 1. Keyword overlap (0-0.4)
    keywords = _extract_keywords(concept_summary, top_n=5)
    if keywords:
        matches = sum(1 for kw in keywords if kw in response_words or kw in response_lower)
        keyword_score = (matches / len(keywords)) * 0.4
    else:
        keyword_score = 0.0
    components["keyword_overlap"] = round(keyword_score, 4)
    score += keyword_score

    # 2. Knowledge area mention (0-0.2)
    ka_terms = set(re.findall(r'[a-z_]{4,}', concept_ka.lower().replace("_", " ")))
    ka_terms -= _STOP_WORDS
    ka_match = 1 if any(t in response_lower for t in ka_terms if len(t) >= 4) else 0
    ka_score = 0.2 if ka_match else 0.0
    components["ka_match"] = ka_match
    score += ka_score

    # 3. Concept ID / summary substring reference (0-0.2)
    id_ref = 0
    # Check if concept_id appears in response
    if concept_id.lower() in response_lower:
        id_ref = 1
    else:
        # Check for significant summary phrases (>= 8 char substrings)
        summary_phrases = re.findall(r'[a-z][a-z\s]{6,}[a-z]', concept_summary.lower())
        for phrase in summary_phrases[:5]:  # Cap iteration
            clean = phrase.strip()
            if len(clean) >= 8 and clean in response_lower:
                id_ref = 1
                break
    id_score = 0.2 if id_ref else 0.0
    components["id_reference"] = id_ref
    score += id_score

    # 4. Position signal (0-0.2)
    if total_activated > 1 and (keyword_score > 0 or id_ref):
        position_score = 0.2 * (1 - activation_rank / total_activated)
    else:
        position_score = 0.0
    components["position_signal"] = round(position_score, 4)
    score += position_score

    # Classify
    total = round(score, 4)
    if total >= UTIL_THRESHOLD_USED:
        util_class = "USED"
    elif total >= UTIL_THRESHOLD_PARTIAL:
        util_class = "PARTIAL"
    else:
        util_class = "UNUSED"

    return {
        "concept_id": concept_id,
        "activation_rank": activation_rank,
        "utilization_score": total,
        "class": util_class,
        **components,
    }


def score_retrieval_utility(
    activated_concept_ids: list[str],
    previous_response: str,
    session_id: str | None = None,
    turn_number: int | None = None,
) -> list[dict[str, Any]]:
    """
    Score how much each activated concept was utilized in the response.

    Returns list of score dicts, one per concept. Also persists to
    retrieval_feedback table if session_id is provided.

    Target: <10ms for 14 concepts.
    """
    t0 = time.perf_counter()

    if not activated_concept_ids or not previous_response or len(previous_response) < 30:
        return []

    # Pre-process response once
    response_lower = previous_response.lower()
    response_words = set(re.findall(r'[a-z_][a-z0-9_]{3,}', response_lower))

    # FEEDBACK-002: Batch concept loading — single SELECT WHERE IN replaces N sequential load_concept calls
    from app.storage import read_snapshot_db

    scores = []
    total = len(activated_concept_ids)

    try:
        with read_snapshot_db("feedback_score_concepts") as conn:
            placeholders = ",".join("?" for _ in activated_concept_ids)
            rows = conn.execute(
                f"SELECT id, summary, knowledge_area FROM concepts WHERE id IN ({placeholders})",
                activated_concept_ids,
            ).fetchall()
            concept_map = {row[0]: (row[1] or "", row[2] or "general") for row in rows}
    except Exception as e:
        logger.debug(f"FEEDBACK-002: Batch load failed, falling back: {e}")
        concept_map = {}

    for rank, cid in enumerate(activated_concept_ids):
        try:
            if cid in concept_map:
                summary, ka = concept_map[cid]
            else:
                # Fallback for concepts not found in batch query
                continue
            result = _score_single_concept(
                concept_summary=summary,
                concept_ka=ka,
                concept_id=cid,
                response_lower=response_lower,
                response_words=response_words,
                activation_rank=rank,
                total_activated=total,
            )
            scores.append(result)
        except Exception as e:
            logger.debug(f"FEEDBACK-001: Score failed for {cid}: {e}")
            continue

    # Persist to DB (non-blocking)
    if session_id and scores:
        try:
            _persist_feedback(session_id, turn_number or 0, scores)
        except Exception as e:
            from app.core.config import get_feedback_db_lock_timeout_s
            _timeout_count = _record_feedback_lock_timeout("feedback_persist")
            logger.warning(
                "FEEDBACK-001: Persist failed (non-fatal): operation=feedback_persist "
                "timeout_ms=%.1f timeout_count=%d error=%s",
                get_feedback_db_lock_timeout_s() * 1000,
                _timeout_count,
                e,
            )

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"FEEDBACK-001: L1 scored {len(scores)} concepts in {elapsed_ms:.1f}ms")

    return scores


def _persist_feedback(session_id: str, turn_number: int, scores: list[dict]) -> None:
    """Write L1 scores to retrieval_feedback table."""
    from app.core.config import get_feedback_db_slow_log_ms

    t0 = time.perf_counter()
    with _feedback_db("feedback_persist") as conn:

        for s in scores:
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO retrieval_feedback
                       (session_id, turn_number, concept_id, activation_rank,
                        utilization_score, utilization_class, keyword_overlap,
                        ka_match, id_reference, position_signal)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        session_id,
                        turn_number,
                        s["concept_id"],
                        s["activation_rank"],
                        s["utilization_score"],
                        s["class"],
                        s.get("keyword_overlap", 0.0),
                        s.get("ka_match", 0),
                        s.get("id_reference", 0),
                        s.get("position_signal", 0.0),
                    ),
                )
            except Exception as e:
                logger.debug(f"FEEDBACK-001: Row insert failed for {s['concept_id']}: {e}")

    elapsed_ms = (time.perf_counter() - t0) * 1000
    if elapsed_ms > get_feedback_db_slow_log_ms():
        logger.warning(
            "FEEDBACK-001: feedback_persist slow path %.1fms for %d score(s)",
            elapsed_ms,
            len(scores),
        )


def update_concept_utility(scores: list[dict[str, Any]]) -> dict[str, Any]:
    """RETRIEVAL-080: Update rolling utility for concepts based on feedback scores.

    Uses exponential moving average (EMA) with asymmetric weighting:
    - USED:    α = 0.15, target = 1.0 (strong positive signal, fast update)
    - PARTIAL: α = 0.08, target = 0.5 (moderate signal)
    - UNUSED:  α = 0.03, target = 0.0 (weak negative signal, slow decay)

    Formula: utility_new = (1 - α) × utility_old + α × target

    Uses classification-mapped targets {USED→1.0, PARTIAL→0.5, UNUSED→0.0},
    NOT raw utilization_score. Gauntlet G4 finding: raw scores too compressed
    for meaningful EMA differentiation.

    Safety caps:
    - Clamped to [0.1, 0.9] — no permanent death or invulnerability
    - Cold-start protection: utility can't drop below current confidence
      until MIN_UTILITY_SAMPLES reached (gauntlet G1 finding)
    - Baseline date: ignores feedback from before FEEDBACK_BASELINE_DATE
    - Structural floor: firmware/constraints/always-activate get min 0.7

    Returns: dict with update stats
    """
    from app.core.config import (
        FEEDBACK_BASELINE_DATE,
        MIN_UTILITY_SAMPLES,
        UTILITY_COLD_START,
        UTILITY_EMA_ALPHA_PARTIAL,
        UTILITY_EMA_ALPHA_UNUSED,
        UTILITY_EMA_ALPHA_USED,
        UTILITY_SCORE_MAX,
        UTILITY_SCORE_MIN,
        UTILITY_STRUCTURAL_FLOOR,
        UTILITY_TARGET_PARTIAL,
        UTILITY_TARGET_UNUSED,
        UTILITY_TARGET_USED,
        get_feedback_db_lock_timeout_s,
        get_feedback_db_slow_log_ms,
        get_feature_flag,
    )

    if not get_feature_flag("FEEDBACK_LOOP_ENABLED", True):
        return {"status": "disabled", "updated": 0}

    if not scores:
        return {"status": "no_scores", "updated": 0}

    # Map classification to (alpha, target)
    ema_params = {
        "USED": (UTILITY_EMA_ALPHA_USED, UTILITY_TARGET_USED),
        "PARTIAL": (UTILITY_EMA_ALPHA_PARTIAL, UTILITY_TARGET_PARTIAL),
        "UNUSED": (UTILITY_EMA_ALPHA_UNUSED, UTILITY_TARGET_UNUSED),
    }

    updated = 0
    errors = 0
    now_iso = _utc_now_iso()
    t0 = time.perf_counter()

    try:
        with _feedback_db("feedback_utility") as conn:
            for score_entry in scores:
                concept_id = score_entry.get("concept_id")
                util_class = score_entry.get("class", "UNUSED")

                if not concept_id or util_class not in ema_params:
                    continue

                alpha, target = ema_params[util_class]

                try:
                    # Read current utility state
                    row = conn.execute(
                        """SELECT utility_score, utility_samples, confidence,
                                  always_activate, concept_type
                           FROM concepts WHERE id = ?""",
                        (concept_id,),
                    ).fetchone()

                    if row is None:
                        continue

                    current_utility = row[0] if row[0] is not None else UTILITY_COLD_START
                    current_samples = row[1] if row[1] is not None else 0
                    current_confidence = row[2] if row[2] is not None else 0.3
                    is_always_activate = bool(row[3]) if row[3] is not None else False
                    concept_type = row[4] or "observation"

                    # EMA update
                    new_utility = (1 - alpha) * current_utility + alpha * target
                    new_samples = current_samples + 1

                    # Safety cap: clamp to [min, max]
                    new_utility = max(UTILITY_SCORE_MIN, min(UTILITY_SCORE_MAX, new_utility))

                    # Cold-start protection (gauntlet G1): utility can't drop below
                    # current confidence until we have enough samples
                    if new_samples < MIN_UTILITY_SAMPLES and new_utility < current_confidence:
                        new_utility = current_confidence

                    # Structural floor: always-activate / firmware concepts
                    if is_always_activate:
                        new_utility = max(new_utility, UTILITY_STRUCTURAL_FLOOR)

                    # Write back
                    conn.execute(
                        """UPDATE concepts
                           SET utility_score = ?, utility_samples = ?, utility_updated = ?
                           WHERE id = ?""",
                        (round(new_utility, 6), new_samples, now_iso, concept_id),
                    )
                    updated += 1

                except Exception as e:
                    logger.debug(f"RETRIEVAL-080: Utility update failed for {concept_id}: {e}")
                    errors += 1
    except RuntimeError as e:
        _timeout_count = _record_feedback_lock_timeout("feedback_utility")
        logger.warning(
            "RETRIEVAL-080: Utility update skipped (non-fatal): "
            "operation=feedback_utility timeout_ms=%.1f timeout_count=%d error=%s",
            get_feedback_db_lock_timeout_s() * 1000,
            _timeout_count,
            e,
        )
        return {
            "status": "skipped_lock_timeout",
            "updated": 0,
            "errors": 1,
            "operation": "feedback_utility",
            "timeout_count": _timeout_count,
        }

    elapsed_ms = (time.perf_counter() - t0) * 1000
    if elapsed_ms > get_feedback_db_slow_log_ms():
        logger.warning(
            "RETRIEVAL-080: feedback_utility slow path %.1fms for %d score(s)",
            elapsed_ms,
            len(scores),
        )

    logger.info(f"RETRIEVAL-080: Updated utility for {updated} concepts ({errors} errors)")
    return {"status": "ok", "updated": updated, "errors": errors}


def _utc_now_iso() -> str:
    """Return current UTC time as ISO string."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_utilization_stats(concept_id: str | None = None, days: int = 7) -> dict:
    """Query aggregate L1 stats for a concept or globally."""
    import app.storage as storage

    with storage._db() as conn:
        # Gracefully handle missing table
        try:
            conn.execute("SELECT 1 FROM retrieval_feedback LIMIT 1")
        except Exception:
            return {"USED": 0, "PARTIAL": 0, "UNUSED": 0, "total": 0, "avg_score": 0.0, "used_ratio": 0.0}
        if concept_id:
            rows = conn.execute(
                """SELECT utilization_class, COUNT(*), AVG(utilization_score)
                   FROM retrieval_feedback
                   WHERE concept_id = ? AND created_at > datetime('now', ?)
                   GROUP BY utilization_class""",
                (concept_id, f"-{days} days"),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT utilization_class, COUNT(*), AVG(utilization_score)
                   FROM retrieval_feedback
                   WHERE created_at > datetime('now', ?)
                   GROUP BY utilization_class""",
                (f"-{days} days",),
            ).fetchall()

    stats = {"USED": 0, "PARTIAL": 0, "UNUSED": 0, "total": 0, "avg_score": 0.0}
    total_score = 0.0
    total_count = 0
    for cls, count, avg in rows:
        stats[cls] = count
        total_count += count
        total_score += avg * count
    stats["total"] = total_count
    stats["avg_score"] = round(total_score / total_count, 4) if total_count > 0 else 0.0
    stats["used_ratio"] = round(stats["USED"] / total_count, 4) if total_count > 0 else 0.0

    return stats
