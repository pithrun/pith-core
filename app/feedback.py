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

    # Load concept summaries (batch for efficiency)
    from app.storage import load_concept

    scores = []
    total = len(activated_concept_ids)

    for rank, cid in enumerate(activated_concept_ids):
        try:
            concept = load_concept(cid, track_access=False)
            if concept is None:
                continue
            result = _score_single_concept(
                concept_summary=concept.summary,
                concept_ka=concept.knowledge_area,
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
            logger.warning(f"FEEDBACK-001: Persist failed (non-fatal): {e}")

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"FEEDBACK-001: L1 scored {len(scores)} concepts in {elapsed_ms:.1f}ms")

    return scores


def _persist_feedback(session_id: str, turn_number: int, scores: list[dict]) -> None:
    """Write L1 scores to retrieval_feedback table."""
    from app.storage import _db

    with _db() as conn:
        # Ensure table exists (idempotent)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS retrieval_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                turn_number INTEGER NOT NULL,
                concept_id TEXT NOT NULL,
                activation_rank INTEGER NOT NULL,
                relevance_score REAL,
                utilization_score REAL NOT NULL,
                utilization_class TEXT NOT NULL,
                keyword_overlap REAL DEFAULT 0.0,
                ka_match INTEGER DEFAULT 0,
                id_reference INTEGER DEFAULT 0,
                position_signal REAL DEFAULT 0.0,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(session_id, turn_number, concept_id)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rf_concept ON retrieval_feedback(concept_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rf_session ON retrieval_feedback(session_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rf_util_class ON retrieval_feedback(utilization_class)"
        )

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


def get_utilization_stats(concept_id: str | None = None, days: int = 7) -> dict:
    """Query aggregate L1 stats for a concept or globally."""
    from app.storage import _db

    with _db() as conn:
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
