"""Wave 5 — Narrative Threads.

Thread lifecycle management, concept-thread linkage, staleness detection,
auto-linking, and intent-based trace retrieval.

Threads are experiential records — they organize work streams, not knowledge.
They answer "what am I working on?" not "what do I know?"
"""

import json
import logging
import uuid
from datetime import datetime, timedelta

from app.datetime_utils import _ensure_aware, _utc_now, _utc_now_iso
from app.models import NarrativeThread, ThreadConceptLink, ThreadSummary

logger = logging.getLogger(__name__)


# =============================================================================
# Thread CRUD
# =============================================================================


def create_thread(
    title: str,
    description: str = "",
    urgency: str = "normal",
    goal_ids: list[str] | None = None,
    knowledge_areas: list[str] | None = None,
    agent_id: str = "default",
) -> NarrativeThread:
    """Create a new narrative thread."""
    from app.storage import _db

    # THREAD-002: Normalize KAs through canonical taxonomy before constructing thread
    if knowledge_areas:
        from app.taxonomy import normalize_knowledge_area

        normalized_kas = []
        for ka in knowledge_areas:
            normalized, _ = normalize_knowledge_area(
                ka, strict=True
            )  # DEBT-021: reject unknown KAs at creation boundary
            if normalized not in normalized_kas:
                normalized_kas.append(normalized)
        knowledge_areas = normalized_kas

    now = _utc_now_iso()
    thread = NarrativeThread(
        id=str(uuid.uuid4()),
        title=title[:500],  # [ST-2 length limit]
        description=description[:500],
        status="active",
        created_at=now,
        updated_at=now,
        last_activity_at=now,
        urgency=urgency if urgency in ("low", "normal", "high") else "normal",
        agent_id=agent_id,
        goal_ids=goal_ids or [],
        knowledge_areas=knowledge_areas or [],
    )

    data = thread.model_dump()
    with _db() as conn:
        conn.execute(
            """INSERT INTO threads (id, title, description, status, created_at,
               updated_at, last_activity_at, completed_at, urgency, agent_id, data)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                thread.id,
                thread.title,
                thread.description,
                thread.status,
                thread.created_at,
                thread.updated_at,
                thread.last_activity_at,
                thread.completed_at,
                thread.urgency,
                thread.agent_id,
                json.dumps(data),
            ),
        )

    return thread


def load_thread(thread_id: str) -> NarrativeThread | None:
    """Load a single thread by ID."""
    from app.storage import _db

    with _db() as conn:
        row = conn.execute("SELECT data FROM threads WHERE id = ?", (thread_id,)).fetchone()
    if not row:
        return None
    return NarrativeThread(**json.loads(row[0]))


def save_thread(thread: NarrativeThread, conn=None) -> None:
    """Persist thread state. Optionally use existing connection for transactions."""
    data = json.dumps(thread.model_dump())

    def _do_save(c):
        c.execute(
            """UPDATE threads SET title=?, description=?, status=?,
               updated_at=?, last_activity_at=?, completed_at=?,
               urgency=?, data=? WHERE id=?""",
            (
                thread.title,
                thread.description,
                thread.status,
                thread.updated_at,
                thread.last_activity_at,
                thread.completed_at,
                thread.urgency,
                data,
                thread.id,
            ),
        )

    if conn:
        _do_save(conn)
    else:
        from app.storage import _db

        with _db() as c:
            _do_save(c)


def load_threads(status: str | None = None, agent_id: str = "default") -> list[NarrativeThread]:
    """Load threads with optional status filter."""
    from app.storage import _db

    with _db() as conn:
        if status:
            rows = conn.execute(
                "SELECT data FROM threads WHERE status = ? AND agent_id = ? ORDER BY last_activity_at DESC",
                (status, agent_id),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT data FROM threads WHERE agent_id = ? ORDER BY last_activity_at DESC",
                (agent_id,),
            ).fetchall()

    return [NarrativeThread(**json.loads(r[0])) for r in rows]


# =============================================================================
# Thread Lifecycle (§5.2)
# =============================================================================

VALID_TRANSITIONS = {
    "active": {"paused", "completed", "abandoned"},
    "paused": {"active", "completed", "abandoned"},
    "completed": {"active"},  # [OR-1] reactivation
    "abandoned": {"active"},  # [OR-1] reactivation
}


def update_thread_status(thread_id: str, new_status: str, reason: str = "user") -> NarrativeThread:
    """Transition thread status with audit trail. [FIX I2]"""
    thread = load_thread(thread_id)
    if not thread:
        raise ValueError(f"Thread {thread_id} not found")

    old_status = thread.status
    if new_status not in VALID_TRANSITIONS.get(old_status, set()):
        raise ValueError(f"Invalid transition: {old_status} → {new_status}")

    thread.status_history.append(
        {
            "from": old_status,
            "to": new_status,
            "reason": reason,
            "at": _utc_now_iso(),
        }
    )
    thread.status = new_status
    thread.updated_at = _utc_now_iso()
    if new_status in ("completed", "abandoned"):
        thread.completed_at = _utc_now_iso()
    save_thread(thread)
    return thread


def record_thread_activity(thread_id: str, source: str = "user") -> None:
    """Update thread activity timestamp. [FIX T1: user vs auto activity]

    source="user" or "session_learn" → updates last_activity_at (resets staleness)
    source="auto" or "reflection" → updates last_auto_activity_at only
    """
    thread = load_thread(thread_id)
    if not thread:
        return
    now = _utc_now_iso()
    thread.updated_at = now
    if source in ("user", "session_learn"):
        thread.last_activity_at = now
    else:
        thread.last_auto_activity_at = now
    save_thread(thread)


# =============================================================================
# Staleness Detection (§5.2) [FIX Q3]
# =============================================================================


def detect_stale_threads() -> list[dict]:
    """Run during reflection. Returns actions taken."""
    from app.config import STALENESS_TIERS

    now = _utc_now()
    actions = []

    for status_filter in ["active", "paused"]:
        threads = load_threads(status=status_filter)
        for t in threads:
            tier = STALENESS_TIERS.get(t.urgency, STALENESS_TIERS["normal"])
            try:
                last_dt = _ensure_aware(datetime.fromisoformat(t.last_activity_at.replace("Z", "")))
                days_idle = (now - last_dt).days
            except (ValueError, TypeError):
                days_idle = 0

            if status_filter == "active" and days_idle >= tier["auto_pause"]:
                update_thread_status(t.id, "paused", reason="staleness_auto_pause")
                actions.append({"thread_id": t.id, "action": "auto_paused", "days_idle": days_idle})
            elif status_filter == "paused" and days_idle >= tier["auto_abandon"]:
                update_thread_status(t.id, "abandoned", reason="staleness_auto_abandon")
                actions.append({"thread_id": t.id, "action": "auto_abandoned", "days_idle": days_idle})

    return actions


# =============================================================================
# Concept-Thread Linkage (§5.3)
# =============================================================================


def link_concept_to_thread(
    thread_id: str,
    concept_id: str,
    role: str = "member",
    added_by: str = "system",
) -> ThreadConceptLink | None:
    """Add concept to thread. Transactional. [FIX F1]"""
    from app.storage import _db

    now = _utc_now_iso()
    link = ThreadConceptLink(
        thread_id=thread_id,
        concept_id=concept_id,
        role=role,
        added_at=now,
        added_by=added_by,
    )

    with _db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO thread_concept_links VALUES (?, ?, ?, ?, ?)",
            (thread_id, concept_id, role, now, added_by),
        )
        # [FIX: deadlock] Inline thread load — load_thread() calls _db() which
        # re-acquires the non-reentrant threading.Lock, causing deadlock.
        row = conn.execute("SELECT data FROM threads WHERE id = ?", (thread_id,)).fetchone()
        if row:
            thread = NarrativeThread(**json.loads(row[0]))
            if concept_id not in thread.concept_ids:
                thread.concept_ids.append(concept_id)
                save_thread(thread, conn=conn)

    activity_source = "user" if added_by == "user" else ("auto" if added_by == "auto" else "session_learn")
    record_thread_activity(thread_id, source=activity_source)
    return link


def classify_thread_role(
    concept_id: str,
    concept_type: str | None,
    thread_id: str,
) -> str:
    """LIFECYCLE-001: Auto-classify thread role based on concept_type and temporal ordering.

    Rules:
    - First concept linked to a thread → initiator
    - concept_type in (decision, principle, method) → conclusion
    - concept_type = constraint → blocker
    - concept_type in (observation, pattern) → evidence
    - Default → member

    Temporal check: "first" is determined by existing link count for this thread.
    If thread has 0 links, this concept is the initiator.
    """
    from app.storage import _db

    # Check if this is the first concept in the thread
    with _db() as conn:
        existing_count = conn.execute(
            "SELECT COUNT(*) FROM thread_concept_links WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()[0]

    if existing_count == 0:
        return "initiator"

    ct = (concept_type or "").lower()

    if ct in ("decision", "principle", "method"):
        return "conclusion"

    if ct == "constraint":
        return "blocker"

    if ct in ("observation", "pattern"):
        return "evidence"

    return "member"


def unlink_concept_from_thread(thread_id: str, concept_id: str) -> None:
    """Remove concept from thread. Transactional. [FIX F1]"""
    from app.storage import _db

    with _db() as conn:
        conn.execute(
            "DELETE FROM thread_concept_links WHERE thread_id=? AND concept_id=?",
            (thread_id, concept_id),
        )
        # [FIX: deadlock] Inline thread load — load_thread() calls _db() which
        # re-acquires the non-reentrant threading.Lock, causing deadlock.
        row = conn.execute("SELECT data FROM threads WHERE id = ?", (thread_id,)).fetchone()
        if row:
            thread = NarrativeThread(**json.loads(row[0]))
            if concept_id in thread.concept_ids:
                thread.concept_ids.remove(concept_id)
                save_thread(thread, conn=conn)


def get_threads_for_concept(concept_id: str) -> list[NarrativeThread]:
    """What threads is this concept in?"""
    from app.storage import _db

    with _db() as conn:
        rows = conn.execute(
            """SELECT t.data FROM threads t
               JOIN thread_concept_links tcl ON t.id = tcl.thread_id
               WHERE tcl.concept_id = ?""",
            (concept_id,),
        ).fetchall()
    return [NarrativeThread(**json.loads(r[0])) for r in rows]


def get_concepts_for_thread(thread_id: str, role: str | None = None) -> list[ThreadConceptLink]:
    """What concepts are in this thread? Optionally filtered by role."""
    from app.storage import _db

    with _db() as conn:
        if role:
            rows = conn.execute(
                "SELECT * FROM thread_concept_links WHERE thread_id=? AND role=?",
                (thread_id, role),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM thread_concept_links WHERE thread_id=?",
                (thread_id,),
            ).fetchall()

    return [ThreadConceptLink(thread_id=r[0], concept_id=r[1], role=r[2], added_at=r[3], added_by=r[4]) for r in rows]


# =============================================================================
# Hybrid Auto-Linking (§5.3) [FIX Q2]
# =============================================================================


def _ka_overlaps(concept_ka: str, thread_kas: list) -> bool:
    """Check if concept KA matches any thread KA, with fuzzy normalization.

    THREAD-002: Replaces exact `in` check that always failed because thread KAs
    use non-canonical names (e.g., 'pith_infrastructure') that don't match any
    concept KAs in the canonical taxonomy.

    Min length guard: substrings < 3 chars produce false positive matches.
    """
    concept_ka_lower = concept_ka.lower()
    if len(concept_ka_lower) < 3:
        return False
    for tka in thread_kas:
        tka_lower = tka.lower()
        if len(tka_lower) < 3:
            continue
        if concept_ka_lower == tka_lower:
            return True
        if concept_ka_lower in tka_lower or tka_lower in concept_ka_lower:
            return True
    return False


def auto_link_candidates(concept, active_threads: list[NarrativeThread]) -> list[str]:
    """Return thread IDs this concept should be auto-linked to.

    Gate 1: Knowledge area overlap (fuzzy, via _ka_overlaps)
    Gate 1b: Title similarity bypass (if KA doesn't match, check title)
    Gate 2: TF-IDF confirmation (rejects false positives from Gate 1/1b)
    Gate 3: Trace linkage (deterministic, always trusted)

    THREAD-002: Added Gate 1b + _ka_overlaps to fix 0-link problem caused by
    non-canonical thread KAs that never matched concept KAs.
    """
    from app.config import AUTO_LINK_TFIDF_THRESHOLD, AUTO_LINK_TITLE_SIMILARITY_THRESHOLD

    # DEBT-024: Hoisted from Gate 1b + Gate 2 (was imported twice)
    try:
        from app.retrieval import retrieval_engine

        _has_retrieval = True
    except (ImportError, Exception):
        _has_retrieval = False

    candidates = []
    concept_ka = getattr(concept, "knowledge_area", "") or ""
    concept_text = f"{getattr(concept, 'summary', '')} {concept_ka}"
    source_trace_id = getattr(concept, "source_trace_id", None)

    for thread in active_threads:
        # Gate 3: Trace linkage — deterministic
        if source_trace_id and source_trace_id in (thread.trace_ids or []):
            candidates.append(thread.id)
            continue

        # Gate 1: KA overlap (fuzzy via _ka_overlaps)
        # DEBT-023: Cache result to avoid duplicate call at Gate 2 branch
        ka_match_direct = concept_ka and _ka_overlaps(concept_ka, thread.knowledge_areas or [])
        ka_match = ka_match_direct

        # Gate 1b: Title similarity bypass — if KA doesn't match,
        # check if concept text is similar enough to thread title
        if not ka_match and _has_retrieval:
            # DEBT-024: uses hoisted retrieval_engine import
            try:
                thread_title_text = f"{thread.title} {thread.description}"
                title_sim = retrieval_engine.index.compute_similarity(concept_text, thread_title_text)
                if title_sim >= AUTO_LINK_TITLE_SIMILARITY_THRESHOLD:
                    ka_match = True  # Promote to Gate 2
            except Exception:
                pass

        if ka_match:
            # Gate 1 KA match is sufficient — thread KA overlap is a strong signal.
            # Gate 2 TF-IDF confirmation is only required when entry was via Gate 1b
            # (title similarity bypass), since title similarity is a weaker signal.
            if ka_match_direct:  # DEBT-023: use cached result
                # Direct KA match — trust it
                candidates.append(thread.id)
            elif _has_retrieval:
                # Entered via Gate 1b — require Gate 2 TF-IDF confirmation
                # DEBT-024: uses hoisted retrieval_engine import
                try:
                    thread_text = f"{thread.title} {thread.description}"
                    similarity = retrieval_engine.index.compute_similarity(concept_text, thread_text)
                    if similarity >= AUTO_LINK_TFIDF_THRESHOLD:
                        candidates.append(thread.id)
                except Exception:
                    pass  # Gate 1b + failed Gate 2 = no link

    return candidates


# =============================================================================
# Intent-Based Episode Retrieval (§5.4)
# =============================================================================


def load_traces_since(cutoff_iso: str, limit: int = 500) -> list:
    """Load traces created after cutoff. Returns TraceRecord list."""
    from app.models import TraceRecord
    from app.storage import _db

    with _db() as conn:
        rows = conn.execute(
            "SELECT data FROM traces WHERE created_at >= ? ORDER BY created_at DESC LIMIT ?",
            (cutoff_iso, limit),
        ).fetchall()

    results = []
    for r in rows:
        try:
            data = json.loads(r[0])
            # TraceRecord fields are stored in the data JSON
            results.append(TraceRecord(**data))
        except Exception:
            continue
    return results


def retrieve_similar_traces(
    situation: str,
    intent: str | None = None,
    limit: int = 5,
    min_similarity: float = 0.4,
    scan_window_days: int = 90,
) -> list[dict]:
    """Find past traces with similar situation/intent. [FIX S1: temporal window]"""
    from app.config import TRACE_RETRIEVAL_SCAN_LIMIT

    query_text = f"{situation} {intent}" if intent else situation
    cutoff = (_utc_now() - timedelta(days=scan_window_days)).isoformat()
    traces = load_traces_since(cutoff, limit=TRACE_RETRIEVAL_SCAN_LIMIT)

    if not traces:
        return []

    # Simple term-overlap scoring (traces are short docs)
    query_terms = _extract_terms_simple(query_text)
    if not query_terms:
        return []

    scored = []
    for trace in traces:
        trace_text = f"{trace.situation} {trace.intent} {trace.assessment}"
        trace_terms = set(_extract_terms_simple(trace_text))
        if not trace_terms:
            continue
        hits = sum(1 for t in query_terms if t in trace_terms)
        similarity = hits / len(query_terms)
        if similarity >= min_similarity:
            scored.append(
                {
                    "trace": trace.model_dump(),
                    "similarity": round(similarity, 3),
                    "linked_concepts": trace.concept_refs,
                }
            )

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:limit]


def _extract_terms_simple(text: str) -> list[str]:
    """Lightweight term extraction for trace search. Shared with §A.5."""
    import re

    STOP_WORDS = {
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
        "just",
        "because",
        "if",
        "when",
        "that",
        "this",
        "it",
        "its",
        "they",
        "them",
        "their",
        "we",
        "our",
        "you",
    }
    words = re.findall(r"[a-z][a-z0-9_]+", text.lower())
    seen = set()
    terms = []
    for w in words:
        if w not in STOP_WORDS and w not in seen:
            seen.add(w)
            terms.append(w)
    return terms


# =============================================================================
# Thread Membership for Salience (§5.6)
# =============================================================================


def preload_thread_membership_cache() -> dict:
    """Batch preload active thread membership counts for salience."""
    from app.storage import _db

    with _db() as conn:
        rows = conn.execute("""
            SELECT tcl.concept_id, COUNT(*) as active_count
            FROM thread_concept_links tcl
            JOIN threads t ON tcl.thread_id = t.id
            WHERE t.status = 'active'
            GROUP BY tcl.concept_id
        """).fetchall()
    return {row[0]: row[1] for row in rows}


def compute_thread_membership(concept_id: str, cache: dict | None = None) -> float:
    """Compute thread membership signal for salience. min(active_count / 3, 1.0)."""
    from app.config import THREAD_MEMBERSHIP_SALIENCE_DIVISOR

    if cache is not None:
        active_count = cache.get(concept_id, 0)
    else:
        threads = get_threads_for_concept(concept_id)
        active_count = sum(1 for t in threads if t.status == "active")
    return min(active_count / THREAD_MEMBERSHIP_SALIENCE_DIVISOR, 1.0)


# =============================================================================
# Thread Orientation (§5.5)
# =============================================================================


def compute_open_threads() -> list[ThreadSummary]:
    """Compute thread summaries for orientation display."""
    from app.config import STALENESS_TIERS, THREAD_MAX_ORIENTATION_DISPLAY

    now = _utc_now()
    summaries = []

    for status in ["active", "paused"]:  # [FIX P1]
        threads = load_threads(status=status)
        for t in threads:
            # [FIX F2] Exclude empty threads older than 24h
            if len(t.concept_ids) == 0 and len(t.trace_ids) == 0 and t.created_at:
                try:
                    created_dt = _ensure_aware(datetime.fromisoformat(t.created_at.replace("Z", "")))
                    if (now - created_dt).days >= 1:
                        continue
                except (ValueError, TypeError):
                    pass

            try:
                last_dt = _ensure_aware(datetime.fromisoformat(t.last_activity_at.replace("Z", "")))
                days_idle = (now - last_dt).days
            except (ValueError, TypeError):
                days_idle = 0

            tier = STALENESS_TIERS.get(t.urgency, STALENESS_TIERS["normal"])
            summaries.append(
                ThreadSummary(
                    thread_id=t.id,
                    title=t.title,
                    status=t.status,
                    urgency=t.urgency,
                    days_since_activity=days_idle,
                    concept_count=len(t.concept_ids),
                    goal_ids=t.goal_ids,
                    staleness_warning=days_idle >= tier["warning"],
                )
            )

    summaries.sort(key=lambda s: (s.status != "active", s.days_since_activity))
    return summaries[:THREAD_MAX_ORIENTATION_DISPLAY]


# =============================================================================
# Thread Stats for pith_stats (§5.8)
# =============================================================================


def get_thread_stats() -> dict:
    """Thread health metrics for pith_stats endpoint."""
    from app.storage import _db

    with _db() as conn:
        counts = {}
        for st in ["active", "paused", "completed", "abandoned"]:
            row = conn.execute("SELECT COUNT(*) FROM threads WHERE status = ?", (st,)).fetchone()
            counts[f"{st}_threads"] = row[0] if row else 0

        total_links = conn.execute("SELECT COUNT(*) FROM thread_concept_links").fetchone()
        counts["total_thread_concept_links"] = total_links[0] if total_links else 0

        avg_row = conn.execute("""
            SELECT AVG(cnt) FROM (
                SELECT COUNT(*) as cnt FROM thread_concept_links tcl
                JOIN threads t ON tcl.thread_id = t.id
                WHERE t.status = 'active'
                GROUP BY tcl.thread_id
            )
        """).fetchone()
        counts["avg_concepts_per_active_thread"] = round(avg_row[0], 1) if avg_row and avg_row[0] else 0.0

        auto_links = conn.execute("SELECT COUNT(*) FROM thread_concept_links WHERE added_by = 'auto'").fetchone()
        counts["auto_link_count"] = auto_links[0] if auto_links else 0

    return counts


def get_active_thread_concept_ids() -> set[str]:
    """THREAD-001: Return concept IDs linked to active threads.

    Used by retrieval scoring to apply a small relevance boost to concepts
    that are part of the user's active work streams.

    Returns:
        Set of concept_id strings from all active threads.
    """
    from app.storage import _db

    with _db() as conn:
        rows = conn.execute(
            """SELECT DISTINCT tcl.concept_id
               FROM thread_concept_links tcl
               JOIN threads t ON tcl.thread_id = t.id
               WHERE t.status = 'active'""",
        ).fetchall()
    return {r[0] for r in rows}
