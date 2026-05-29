"""Automatic Reflection — post-session metacognitive extraction.

Three trigger points for L1→L3+ concept synthesis:
  T1: Retroactive reflection on orphaned sessions (at session_start/conversation_turn)
  T2: In-flight reflection bookmarks (during conversation_turn, every N turns)
  T3: Full session-end reflection (at end_session)

Design doc: docs/design/AUTOMATIC_REFLECTION_DESIGN.md
"""

import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime, timedelta

from app.core.datetime_utils import _ensure_aware, _utc_now, _utc_now_iso
from app.storage import load_concept

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants (§10 of spec)
# ---------------------------------------------------------------------------

# T1: Retroactive reflection on orphaned sessions
T1_MIN_LEARNING_EVENTS = 3
T1_STALENESS_MIN_HOURS = 1
T1_STALENESS_MAX_DAYS = 7
T1_MAX_PROMPTS = 3

# T2: In-flight bookmarks
T2_TURN_INTERVAL = 10
T2_MIN_L1_SINCE_LAST = 4
T2_MIN_SHARED_AREA = 2
T2_MAX_BOOKMARKS_PER_SESSION = 5

# T3: Session-end full reflection
T3_MIN_LEARNING_EVENTS = 5
T3_MIN_DURATION_SECONDS = 300
T3_L1_RATIO_THRESHOLD = 0.80
T3_MAX_PROMPTS = 3

# Concept type classification
L1_TYPES = {"observation", "pattern", "goal", "constraint", "hypothesis"}
L3_TYPES = {"principle", "method", "heuristic", "cognitive_strategy", "system_model"}


# ---------------------------------------------------------------------------
# Cluster pattern → suggested synthesis type (§7.3 of spec)
# ---------------------------------------------------------------------------

CLUSTER_TYPE_HINTS = {
    "how_to": {
        "keywords": ["how", "process", "step", "workflow", "pipeline", "setup", "configure"],
        "target_type": "method",
        "question_template": "Is there a reusable METHOD for {theme}?",
    },
    "causal": {
        "keywords": ["caused", "because", "reason", "why", "root cause", "led to", "resulted"],
        "target_type": "principle",
        "question_template": "Is there a general PRINCIPLE underlying {theme}?",
    },
    "conditional": {
        "keywords": ["when", "if", "should", "avoid", "prefer", "instead", "fallback"],
        "target_type": "heuristic",
        "question_template": "Is there a reusable HEURISTIC (when X, do Y) for {theme}?",
    },
    "meta": {
        "keywords": ["review", "verify", "check", "debug", "diagnose", "think", "approach"],
        "target_type": "cognitive_strategy",
        "question_template": "Is there a COGNITIVE STRATEGY for {theme}?",
    },
}


# ---------------------------------------------------------------------------
# Core: Prompt Generation Engine (§7 of spec)
# ---------------------------------------------------------------------------


def _classify_cluster_type(summaries: list[str]) -> tuple[str, str]:
    """Determine suggested concept type + question template for a cluster.

    Scans summaries for keyword patterns. Returns (target_type, question_template).
    Falls back to 'principle' if no strong signal.
    """
    combined = " ".join(s.lower() for s in summaries)

    best_match = None
    best_score = 0

    for hint_id, hint in CLUSTER_TYPE_HINTS.items():
        score = sum(1 for kw in hint["keywords"] if kw in combined)
        if score > best_score:
            best_score = score
            best_match = hint

    if best_match and best_score >= 2:
        return best_match["target_type"], best_match["question_template"]

    # Default: principle
    return "principle", "Is there a general PRINCIPLE underlying {theme}?"


def _extract_theme(summaries: list[str], knowledge_area: str) -> str:
    """Extract a short theme description from a cluster of summaries.

    Uses simple word frequency to find the dominant topic.
    """
    # Common stopwords
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
        "and",
        "but",
        "or",
        "nor",
        "not",
        "so",
        "yet",
        "both",
        "either",
        "neither",
        "each",
        "every",
        "all",
        "any",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "only",
        "own",
        "same",
        "than",
        "too",
        "very",
        "that",
        "this",
        "it",
        "its",
        "new",
        "also",
        "when",
        "if",
        "then",
        "else",
        "what",
    }

    word_freq: dict[str, int] = defaultdict(int)
    for s in summaries:
        for word in s.lower().split():
            cleaned = word.strip(".,;:!?()[]{}\"'`—-")
            if len(cleaned) > 3 and cleaned not in stopwords:
                word_freq[cleaned] += 1

    # Top 3 words as theme
    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:3]
    if top_words:
        return f"{knowledge_area}: {', '.join(w[0] for w in top_words)}"
    return knowledge_area


def generate_reflection_prompts(
    concept_ids: list[str],
    max_prompts: int = 3,
    include_bookmarks: list[dict] | None = None,
) -> list[dict]:
    """Generate targeted reflection prompts from a set of concepts.

    Algorithm (§7.1):
    1. Load concepts, filter to L1 types
    2. Cluster by knowledge_area
    3. For each cluster with ≥2 concepts, generate synthesis prompt
    4. Rank clusters by size × avg confidence
    5. Cap at max_prompts

    Returns list of prompt dicts.
    """
    # Step 1: Load and filter
    l1_concepts = []
    for cid in concept_ids:
        concept = load_concept(cid, track_access=False)
        if not concept:
            continue
        ctype = concept.concept_type or "observation"
        if ctype in L1_TYPES:
            ka = "general"
            if concept.metadata and concept.metadata.get("knowledge_area"):
                ka = concept.metadata["knowledge_area"]
            l1_concepts.append(
                {
                    "id": concept.id,
                    "summary": concept.summary,
                    "confidence": concept.confidence,
                    "knowledge_area": ka,
                    "concept_type": ctype,
                }
            )

    if len(l1_concepts) < 2:
        return []

    # Step 2: Cluster by knowledge_area
    clusters: dict[str, list[dict]] = defaultdict(list)
    for c in l1_concepts:
        clusters[c["knowledge_area"]].append(c)

    # Step 3: Generate prompts for clusters with ≥2 concepts
    prompts = []
    for area, concepts in clusters.items():
        if len(concepts) < 2:
            continue

        summaries = [c["summary"] for c in concepts]
        concept_ids_in_cluster = [c["id"] for c in concepts]
        avg_confidence = sum(c["confidence"] for c in concepts) / len(concepts)

        # Determine synthesis type
        target_type, question_template = _classify_cluster_type(summaries)
        theme = _extract_theme(summaries, area)

        # Build the targeted prompt
        concept_refs = ", ".join(f"`{cid}`" for cid in concept_ids_in_cluster[:5])
        suggested_synthesis = question_template.format(theme=theme)

        prompt = {
            "observation_cluster": concept_ids_in_cluster,
            "observation_summaries": summaries[:5],  # Cap for payload size
            "suggested_synthesis": suggested_synthesis,
            "target_concept_type": target_type,
            "knowledge_area": area,
            "concept_refs_display": concept_refs,
            "cluster_size": len(concepts),
            "avg_confidence": round(avg_confidence, 3),
        }

        # Attach any unprocessed bookmarks for this area
        if include_bookmarks:
            area_bookmarks = [b for b in include_bookmarks if b.get("knowledge_area") == area]
            if area_bookmarks:
                prompt["unprocessed_bookmarks"] = area_bookmarks

        prompts.append(prompt)

    # Step 4: Rank by cluster_size × avg_confidence
    prompts.sort(key=lambda p: p["cluster_size"] * p["avg_confidence"], reverse=True)

    # Step 5: Cap
    return prompts[:max_prompts]


# ---------------------------------------------------------------------------
# T1: Retroactive Reflection on Orphaned Sessions (§4 of spec)
# ---------------------------------------------------------------------------


def check_orphaned_sessions_for_reflection(
    orphaned_sessions: list[dict],
) -> dict | None:
    """Check orphaned sessions for reflection eligibility and generate prompts.

    Args:
        orphaned_sessions: List of session dicts from recover_interrupted_sessions
            Each has: id, started_at, ended_at, status, learning_event_count, data

    Returns:
        Retroactive reflection payload or None if no eligible sessions.
    """
    # REFLECT-021: Feature flag gate — T1 had 0/13 productive cycles in production
    from app.core.config import FEATURE_FLAGS
    if not FEATURE_FLAGS.get("T1_RETROACTIVE_REFLECTION_ENABLED", False):
        return None

    now = _utc_now()

    for session in orphaned_sessions:
        # Gate 1: Minimum learning events
        events = session.get("learning_event_count", 0)
        if events < T1_MIN_LEARNING_EVENTS:
            continue

        # Gate 2: Staleness window
        ended_at_str = session.get("ended_at")
        if not ended_at_str:
            continue
        try:
            ended_at = _ensure_aware(datetime.fromisoformat(ended_at_str))
        except (ValueError, TypeError):
            continue

        age = now - ended_at
        if age < timedelta(hours=T1_STALENESS_MIN_HOURS):
            continue  # Too fresh — might be a pause
        if age > timedelta(days=T1_STALENESS_MAX_DAYS):
            continue  # Too stale

        # Gate 3: Not already reflected
        session_data = session.get("data")
        if session_data:
            try:
                data = json.loads(session_data) if isinstance(session_data, str) else session_data
                if data.get("reflection_completed"):
                    continue
            except (json.JSONDecodeError, TypeError):
                pass

        # Find concepts created during this session
        session_id = session.get("id", "")
        session_concept_ids = _find_session_concepts(session_id)

        if len(session_concept_ids) < 2:
            continue

        # Generate prompts
        prompts = generate_reflection_prompts(session_concept_ids, max_prompts=T1_MAX_PROMPTS)

        if not prompts:
            continue

        # Build session stats
        stats = _compute_session_stats(session_concept_ids)

        logger.info(
            f"T1 retroactive reflection: orphaned session {session_id}, "
            f"{len(session_concept_ids)} concepts, {len(prompts)} prompts generated"
        )

        return {
            "type": "retroactive_reflection",
            "orphaned_session_id": session_id,
            "prompts": prompts,
            "session_stats": stats,
        }

    return None


# ---------------------------------------------------------------------------
# T2: In-Flight Reflection Bookmarks (§5 of spec)
# ---------------------------------------------------------------------------


def check_inflight_reflection(
    session_concepts_since_last_bookmark: list[str],
    existing_bookmarks: list[dict],
) -> dict | None:
    """Check if an in-flight bookmark should be generated.

    Args:
        session_concepts_since_last_bookmark: concept_ids created since last bookmark
        existing_bookmarks: already-generated bookmarks this session

    Returns:
        ReflectionBookmark dict or None.
    """
    # Gate 1: Max bookmarks per session
    if len(existing_bookmarks) >= T2_MAX_BOOKMARKS_PER_SESSION:
        return None

    # Gate 2: Minimum L1 concepts since last bookmark
    if len(session_concepts_since_last_bookmark) < T2_MIN_L1_SINCE_LAST:
        return None

    # Load and filter to L1
    l1_by_area: dict[str, list[dict]] = defaultdict(list)
    for cid in session_concepts_since_last_bookmark:
        concept = load_concept(cid, track_access=False)
        if not concept:
            continue
        ctype = concept.concept_type or "observation"
        if ctype not in L1_TYPES:
            continue
        ka = "general"
        if concept.metadata and concept.metadata.get("knowledge_area"):
            ka = concept.metadata["knowledge_area"]
        l1_by_area[ka].append(
            {
                "id": concept.id,
                "summary": concept.summary,
            }
        )

    # Gate 3: At least T2_MIN_SHARED_AREA concepts sharing an area
    best_area = None
    best_count = 0
    for area, concepts in l1_by_area.items():
        if len(concepts) > best_count:
            best_count = len(concepts)
            best_area = area

    if best_count < T2_MIN_SHARED_AREA:
        return None

    # Generate lightweight bookmark
    area_concepts = l1_by_area[best_area]
    summaries = [c["summary"] for c in area_concepts]
    concept_ids = [c["id"] for c in area_concepts]
    theme = _extract_theme(summaries, best_area)

    bookmark = {
        "bookmark_id": str(uuid.uuid4())[:8],
        "related_observations": concept_ids,
        "observation_summaries": summaries[:3],
        "hint": f"{len(area_concepts)} observations about {theme} — possible pattern",
        "knowledge_area": best_area,
        "created_at": _utc_now_iso(),
    }

    logger.info(f"T2 bookmark generated: {bookmark['hint']}")
    return bookmark


# ---------------------------------------------------------------------------
# T3: Full Session-End Reflection (§6 of spec)
# ---------------------------------------------------------------------------


def generate_session_end_reflection(
    session_concept_ids: list[str],
    learning_event_count: int,
    session_duration_seconds: float,
    unprocessed_bookmarks: list[dict] | None = None,
) -> dict | None:
    """Generate full reflection prompts at session end.

    Args:
        session_concept_ids: All concept_ids created this session
        learning_event_count: Total learning events this session
        session_duration_seconds: How long the session lasted
        unprocessed_bookmarks: T2 bookmarks that weren't addressed

    Returns:
        Full reflection payload or None if criteria not met.
    """
    # REFLECT-021: Feature flag gate — T3 had 0/21 productive cycles in production
    from app.core.config import FEATURE_FLAGS
    if not FEATURE_FLAGS.get("T3_SESSION_END_REFLECTION_ENABLED", False):
        return None

    # Gate 1: Minimum learning events
    if learning_event_count < T3_MIN_LEARNING_EVENTS:
        return None

    # Gate 2: Minimum duration
    if session_duration_seconds < T3_MIN_DURATION_SECONDS:
        return None

    # Gate 3: L1:L3 ratio check
    stats = _compute_session_stats(session_concept_ids)
    total = stats.get("total_concepts", 0)
    if total == 0:
        return None

    l1_ratio = stats.get("l1_count", 0) / total
    if l1_ratio < T3_L1_RATIO_THRESHOLD:
        # Session already has enough L3+ concepts — reflection less needed
        return None

    # Generate prompts
    prompts = generate_reflection_prompts(
        session_concept_ids,
        max_prompts=T3_MAX_PROMPTS,
        include_bookmarks=unprocessed_bookmarks,
    )

    if not prompts:
        return None

    logger.info(
        f"T3 session-end reflection: {len(prompts)} prompts, L1 ratio={l1_ratio:.2f}, {learning_event_count} events"
    )

    return {
        "type": "session_end_reflection",
        "prompts": prompts,
        "session_summary": stats,
        "unprocessed_bookmark_count": len(unprocessed_bookmarks or []),
    }


# ---------------------------------------------------------------------------
# Compliance Tracking (§8 of spec)
# ---------------------------------------------------------------------------


def record_reflection_event(
    session_id: str,
    trigger_type: str,
    prompts_sent: int,
    prompt_data: list[dict] | None = None,
) -> None:
    """Record that reflection prompts were sent. Called by triggers T1/T2/T3.

    Completion is tracked later when concepts with reflection_source tags arrive.
    """
    try:
        from app.storage import _db

        now = _utc_now_iso()
        with _db() as conn:
            conn.execute(
                """INSERT INTO reflection_tracking
                   (session_id, trigger_type, prompts_sent, prompt_data, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (session_id, trigger_type, prompts_sent, json.dumps(prompt_data) if prompt_data else None, now),
            )
        logger.debug(f"Reflection event recorded: {trigger_type}, {prompts_sent} prompts")
    except Exception as e:
        logger.warning(f"Failed to record reflection event (non-fatal): {e}")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _find_session_concepts(session_id: str) -> list[str]:
    """Find all concept IDs created during a given session.

    Uses SQL JSON extraction instead of N+1 load_concept calls.
    source_session is stored in data->'metadata'->'source_session'.
    """
    from app.storage import _get_connection

    conn = _get_connection()
    rows = conn.execute(
        """SELECT id FROM concepts
           WHERE json_extract(data, '$.metadata.source_session') = ?""",
        (session_id,),
    ).fetchall()
    return [row[0] for row in rows]


def _compute_session_stats(concept_ids: list[str]) -> dict:
    """Compute concept type distribution stats for a set of concepts."""
    by_type: dict[str, int] = defaultdict(int)
    by_area: dict[str, int] = defaultdict(int)
    l1_count = 0
    l3_count = 0
    total = 0

    for cid in concept_ids:
        concept = load_concept(cid, track_access=False)
        if not concept:
            continue
        total += 1
        ctype = concept.concept_type or "observation"
        by_type[ctype] += 1

        if ctype in L1_TYPES:
            l1_count += 1
        elif ctype in L3_TYPES:
            l3_count += 1

        ka = "general"
        if concept.metadata and concept.metadata.get("knowledge_area"):
            ka = concept.metadata["knowledge_area"]
        by_area[ka] += 1

    return {
        "total_concepts": total,
        "l1_count": l1_count,
        "l3_count": l3_count,
        "l1_l3_ratio": round(l1_count / total, 3) if total > 0 else 0.0,
        "by_type": dict(by_type),
        "by_area": dict(by_area),
        "knowledge_areas": list(by_area.keys()),
    }


def mark_session_reflected(session_id: str) -> None:
    """Mark a session as having completed reflection.

    Prevents T1 from re-reflecting on the same orphaned session.
    """
    try:
        from app.storage import _db

        with _db() as conn:
            # Update session data JSON to include reflection_completed flag
            row = conn.execute("SELECT data FROM sessions WHERE id = ?", (session_id,)).fetchone()

            data = {}
            if row and row[0]:
                try:
                    data = json.loads(row[0]) if isinstance(row[0], str) else row[0]
                except (json.JSONDecodeError, TypeError):
                    data = {}

            data["reflection_completed"] = True
            data["reflection_completed_at"] = _utc_now_iso()

            conn.execute(
                "UPDATE sessions SET data = ? WHERE id = ?",
                (json.dumps(data), session_id),
            )
        logger.debug(f"Session {session_id} marked as reflection-completed")
    except Exception as e:
        logger.warning(f"Failed to mark session reflected (non-fatal): {e}")


# ---------------------------------------------------------------------------
# Wave 4a §4a.3: Stale Salience Alerts
# ---------------------------------------------------------------------------
