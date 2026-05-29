"""Wave 5 — Narrative Threads.

Thread lifecycle management, concept-thread linkage, staleness detection,
auto-linking, and intent-based trace retrieval.

Threads are experiential records — they organize work streams, not knowledge.
They answer "what am I working on?" not "what do I know?"
"""

import hashlib
import json
import logging
import re
import time
import uuid
from collections import Counter
from datetime import UTC, datetime, timedelta

import numpy as np

from app.core.datetime_utils import _ensure_aware, _utc_now, _utc_now_iso
from app.core.models import (
    ORIGIN_ID_RE,
    NarrativeThread,
    ThreadConceptLink,
    ThreadSummary,
    WorkstreamDiscoveryState,
    WorkstreamMetadata,
)

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
        from app.cognitive.taxonomy import normalize_knowledge_area

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


def load_threads(
    status: str | None = None,
    agent_id: str = "default",
    limit: int | None = None,
) -> list[NarrativeThread]:
    """Load threads with optional status filter and positive row limit."""
    from app.storage import _db

    params: list[object]
    with _db() as conn:
        if status:
            query = "SELECT data FROM threads WHERE status = ? AND agent_id = ? ORDER BY last_activity_at DESC"
            params = [status, agent_id]
        else:
            query = "SELECT data FROM threads WHERE agent_id = ? ORDER BY last_activity_at DESC"
            params = [agent_id]
        if limit is not None:
            try:
                normalized_limit = int(limit)
            except (TypeError, ValueError):
                normalized_limit = 0
            if normalized_limit > 0:
                query += " LIMIT ?"
                params.append(normalized_limit)
        rows = conn.execute(query, tuple(params)).fetchall()

    return [NarrativeThread(**json.loads(r[0])) for r in rows]


def summarize_thread_for_list(thread: NarrativeThread) -> dict:
    """Return a compact thread summary for list responses."""
    return {
        "id": thread.id,
        "title": thread.title,
        "description": thread.description,
        "status": thread.status,
        "created_at": thread.created_at,
        "updated_at": thread.updated_at,
        "last_activity_at": thread.last_activity_at,
        "last_auto_activity_at": thread.last_auto_activity_at,
        "completed_at": thread.completed_at,
        "urgency": thread.urgency,
        "agent_id": thread.agent_id,
        "goal_ids": thread.goal_ids,
        "knowledge_areas": thread.knowledge_areas,
        "predecessor_id": thread.predecessor_id,
        "concept_count": len(thread.concept_ids),
        "trace_count": len(thread.trace_ids),
        "status_history_count": len(thread.status_history),
    }


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
    from app.core.config import STALENESS_TIERS

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
# Workstreams Phase 0/1: Read-only classification and context blocks
# =============================================================================

WORKSTREAM_MAINTENANCE_PREFIX = "Residual Review:"
WORKSTREAM_REF_LIMIT_DEFAULT = 10
WORKSTREAM_REF_LIMIT_MAX = 20
WORKSTREAM_CLASSIFIER_LIMIT_MAX = 100
WORKSTREAM_TEXT_MAX = 1000
WORKSTREAM_BLOCKERS_MAX = 20
WORKSTREAM_BLOCKER_TEXT_MAX = 500
WORKSTREAM_KNOWLEDGE_AREAS_MAX = 10
WORKSTREAM_KNOWLEDGE_AREA_TEXT_MAX = 100
WORKSTREAM_KNOWLEDGE_AREA_RE = re.compile(r"^[a-z0-9_]+$")
WORKSTREAM_QUALITY_STATES = {"ok", "needs_review", "blocked"}
WORKSTREAM_RELATIONSHIPS = {"child", "related"}
WORKSTREAM_DISCOVERY_TIERS = {
    "curated_candidate",
    "recent_auto_advisory",
    "stale_auto_debug",
    "needs_hygiene_review",
    "proof_or_maintenance",
    "terminal_archive",
}
WORKSTREAM_DISCOVERY_HIDDEN_TIERS = {
    "stale_auto_debug",
    "needs_hygiene_review",
    "proof_or_maintenance",
    "terminal_archive",
}
WORKSTREAM_DISCOVERY_DEFAULT_TTL_DAYS = 7
WORKSTREAM_ACTIVATION_SKIP_PREFIX = "workstream-skip:"
WORKSTREAM_ACTIVATION_BUCKET_LIMIT = 5
WORKSTREAM_ACTIVATION_TASK_ID_MAX = 256
WORKSTREAM_ACTIVATION_SKIP_REASON_MAX = 500
WORKSTREAM_ACTIVATION_SKIP_EXCEPTION_KINDS = {
    "trivial",
    "proof_only",
    "maintenance_only",
    "operator_declined",
    "ambiguous_candidates",
    "other",
}
WORKSTREAM_ACTIVATION_DEFAULT_SKIP_EXCEPTION_KIND = "other"
_WORKSTREAM_PROOF_TERMS = (
    "proof",
    "control window",
    "controlled proof",
    "render proof",
    "fallback/http",
    "fallback http",
    "gauntlet rerun",
)
_WORKSTREAM_ROLE_PRIORITY = {
    "initiator": 0,
    "blocker": 1,
    "conclusion": 2,
    "evidence": 3,
    "member": 4,
}


def _row_to_dict(row) -> dict:
    """Convert sqlite Row or tuple-like rows into a plain dict."""
    if row is None:
        return {}
    if hasattr(row, "keys"):
        return {key: row[key] for key in row.keys()}
    return dict(row)


def _coerce_nonempty_string(value, default: str = "default", max_len: int = 128) -> str:
    text = str(value if value is not None else default).strip()
    if not text:
        text = default
    return text[:max_len]


def _json_count(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _clamp_positive_int(value, default: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(1, min(parsed, maximum))


def _normalize_workstream_text(value: object, max_len: int = WORKSTREAM_TEXT_MAX) -> str:
    return str(value or "").strip()[:max_len]


def _normalize_workstream_blockers(value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("blockers_must_be_list")
    blockers = []
    for item in value[:WORKSTREAM_BLOCKERS_MAX]:
        text = _normalize_workstream_text(item, max_len=WORKSTREAM_BLOCKER_TEXT_MAX)
        if text:
            blockers.append(text)
    return blockers


def _normalize_workstream_knowledge_areas(value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("knowledge_areas_must_be_list")
    areas = []
    for item in value[:WORKSTREAM_KNOWLEDGE_AREAS_MAX]:
        area = str(item or "").strip().lower()[:WORKSTREAM_KNOWLEDGE_AREA_TEXT_MAX]
        if not area:
            continue
        if not WORKSTREAM_KNOWLEDGE_AREA_RE.fullmatch(area):
            raise ValueError("invalid_knowledge_area")
        if area not in areas:
            areas.append(area)
    return areas


def _normalize_workstream_quality_state(value: object) -> str:
    state = str(value or "ok").strip().lower()
    if state not in WORKSTREAM_QUALITY_STATES:
        raise ValueError("invalid_quality_state")
    return state


def _normalize_discovery_reason_codes(value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("discovery_reason_codes_must_be_list")
    codes: list[str] = []
    for item in value[:16]:
        code = str(item or "").strip().lower()[:80]
        if code and re.fullmatch(r"[a-z0-9_:-]+", code) and code not in codes:
            codes.append(code)
    return codes


def _normalize_workstream_discovery_state(value: object) -> WorkstreamDiscoveryState | None:
    if value is None:
        return None
    if isinstance(value, WorkstreamDiscoveryState):
        return value
    if not isinstance(value, dict):
        raise ValueError("discovery_state_must_be_object")
    tier = str(value.get("tier") or "needs_hygiene_review").strip().lower()
    if tier not in WORKSTREAM_DISCOVERY_TIERS:
        raise ValueError("invalid_discovery_tier")
    previous_tier = _normalize_workstream_text(value.get("previous_tier"), max_len=80) or None
    if previous_tier and previous_tier not in WORKSTREAM_DISCOVERY_TIERS:
        previous_tier = None
    return WorkstreamDiscoveryState(
        tier=tier,
        reason_codes=_normalize_discovery_reason_codes(value.get("reason_codes")),
        source=_normalize_workstream_text(value.get("source"), max_len=128) or "manual",
        run_id=_normalize_workstream_text(value.get("run_id"), max_len=128) or None,
        last_evaluated_at=_normalize_workstream_text(value.get("last_evaluated_at"), max_len=64) or None,
        eligible_until=_normalize_workstream_text(value.get("eligible_until"), max_len=64) or None,
        previous_tier=previous_tier,
        promoted_by=_normalize_workstream_text(value.get("promoted_by"), max_len=128) or None,
        promoted_at=_normalize_workstream_text(value.get("promoted_at"), max_len=64) or None,
        promotion_reason=_normalize_workstream_text(value.get("promotion_reason"), max_len=500) or None,
    )


def _normalize_workstream_relationship(value: object) -> str | None:
    relationship = str(value or "").strip().lower()
    if not relationship:
        return None
    if relationship not in WORKSTREAM_RELATIONSHIPS:
        return None
    return relationship


def _normalize_parent_workstream_id(value: object) -> str | None:
    parent_id = _normalize_workstream_text(value, max_len=128)
    return parent_id or None


def _build_workstream_metadata(
    current_objective: object = "",
    current_summary: object = "",
    next_action: object = "",
    blockers: object = None,
    quality_state: object = "ok",
    created_by: object = "user",
    updated_by: object = "user",
    parent_workstream_id: object = None,
    parent_title: object = None,
    relationship: object = None,
    discovery_state: object = None,
) -> WorkstreamMetadata:
    return WorkstreamMetadata(
        current_objective=_normalize_workstream_text(current_objective),
        current_summary=_normalize_workstream_text(current_summary),
        next_action=_normalize_workstream_text(next_action),
        blockers=_normalize_workstream_blockers(blockers),
        quality_state=_normalize_workstream_quality_state(quality_state),
        created_by=_normalize_workstream_text(created_by, max_len=128) or "user",
        updated_by=_normalize_workstream_text(updated_by, max_len=128) or "user",
        parent_workstream_id=_normalize_parent_workstream_id(parent_workstream_id),
        parent_title=_normalize_workstream_text(parent_title, max_len=500) or None,
        relationship=_normalize_workstream_relationship(relationship),
        discovery_state=_normalize_workstream_discovery_state(discovery_state),
    )


def _metadata_updates_from_payload(payload: dict) -> dict:
    updates = {}
    if "current_objective" in payload:
        updates["current_objective"] = _normalize_workstream_text(payload.get("current_objective"))
    if "current_summary" in payload:
        updates["current_summary"] = _normalize_workstream_text(payload.get("current_summary"))
    if "next_action" in payload:
        updates["next_action"] = _normalize_workstream_text(payload.get("next_action"))
    if "blockers" in payload:
        updates["blockers"] = _normalize_workstream_blockers(payload.get("blockers"))
    if "quality_state" in payload:
        updates["quality_state"] = _normalize_workstream_quality_state(payload.get("quality_state"))
    if "updated_by" in payload:
        updates["updated_by"] = _normalize_workstream_text(payload.get("updated_by"), max_len=128) or "user"
    if "parent_workstream_id" in payload:
        updates["parent_workstream_id"] = _normalize_parent_workstream_id(payload.get("parent_workstream_id"))
    if "parent_title" in payload:
        updates["parent_title"] = _normalize_workstream_text(payload.get("parent_title"), max_len=500) or None
    if "relationship" in payload:
        updates["relationship"] = _normalize_workstream_relationship(payload.get("relationship"))
    return updates


def _normalize_binding_authority(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    if not ORIGIN_ID_RE.fullmatch(normalized):
        raise ValueError(f"invalid_{field_name}")
    return normalized


def _workstream_binding_task_id(thread_id: str, origin_id: str | None = None, session_id: str | None = None) -> str:
    authority = _normalize_binding_authority(origin_id, "origin_id")
    if authority:
        return f"workstream:{authority}:{thread_id}"
    authority = _normalize_binding_authority(session_id, "session_id")
    if authority:
        return f"workstream:{authority}:{thread_id}"
    raise ValueError("binding_authority_required")


def _workstream_binding_authority(origin_id: str | None = None, session_id: str | None = None) -> tuple[str, str]:
    normalized_origin = _normalize_binding_authority(origin_id, "origin_id")
    if normalized_origin:
        return "origin_id", normalized_origin
    normalized_session = _normalize_binding_authority(session_id, "session_id")
    if normalized_session:
        return "session_id", normalized_session
    raise ValueError("binding_authority_required")


def _workstream_activation_authority(origin_id: str | None = None, session_id: str | None = None) -> tuple[str, str]:
    return _workstream_binding_authority(origin_id=origin_id, session_id=session_id)


def _normalize_activation_task_id(value: object) -> str:
    text = str(value or "").strip()[:WORKSTREAM_ACTIVATION_TASK_ID_MAX]
    if not text:
        raise ValueError("current_task_id_required")
    return text


def _current_task_hash(current_task_id: str | None) -> str:
    normalized_task = _normalize_activation_task_id(current_task_id)
    return hashlib.sha256(normalized_task.encode("utf-8")).hexdigest()[:16]


def _workstream_composite_task_id(thread_id: str, current_task_id: str | None) -> str:
    return f"workstream-composite:{thread_id}:{_current_task_hash(current_task_id)}"


def _workstream_session_task_id(thread_id: str, current_task_id: str | None) -> str:
    return f"workstream-session-task:{thread_id}:{_current_task_hash(current_task_id)}"


def _workstream_skip_task_id(authority_type: str, authority_value: str, current_task_id: str) -> str:
    normalized_task = _normalize_activation_task_id(current_task_id)
    task_hash = hashlib.sha256(normalized_task.encode("utf-8")).hexdigest()[:16]
    return f"{WORKSTREAM_ACTIVATION_SKIP_PREFIX}{authority_type}:{authority_value}:{task_hash}"


def _normalize_skip_exception_kind(value: object) -> str:
    kind = str(value or WORKSTREAM_ACTIVATION_DEFAULT_SKIP_EXCEPTION_KIND).strip().lower()
    if kind not in WORKSTREAM_ACTIVATION_SKIP_EXCEPTION_KINDS:
        return WORKSTREAM_ACTIVATION_DEFAULT_SKIP_EXCEPTION_KIND
    return kind


def _activation_create_metadata(current_task_id: str | None) -> dict:
    task_label = _normalize_workstream_text(current_task_id or "durable work", max_len=120) or "durable work"
    return {
        "read_only": True,
        "suggestion_authority": "non_authoritative_draft",
        "title": f"Workstream for {task_label}",
        "current_objective": f"Track durable work for {task_label}",
        "next_action": "Bind this session and continue the work",
    }


def _activation_decision_options() -> list[dict[str, str]]:
    return [
        {
            "action": "bind_existing",
            "write_mode": "bind_existing",
            "label": "Continue existing Workstream",
        },
        {
            "action": "create_child",
            "write_mode": "create_and_bind",
            "label": "Create narrower child Workstream",
        },
        {
            "action": "create_new",
            "write_mode": "create_and_bind",
            "label": "Create unrelated new Workstream",
        },
        {
            "action": "skip",
            "write_mode": "skip",
            "label": "Record typed skip exception",
        },
    ]


def _activation_child_metadata(current_task_id: str | None, recommended_rows: list[dict] | None) -> dict | None:
    strong_rows = [
        row
        for row in recommended_rows or []
        if row.get("candidate_evidence_quality") == "strong" and row.get("thread_id")
    ]
    if len(strong_rows) != 1:
        return None
    parent = strong_rows[0]
    task_label = _normalize_workstream_text(current_task_id or "durable work", max_len=120) or "durable work"
    return {
        "read_only": True,
        "suggestion_authority": "non_authoritative_draft",
        "parent_workstream_id": parent.get("thread_id"),
        "parent_title": parent.get("title"),
        "relationship": "child",
        "title": f"Workstream for {task_label}",
        "current_objective": f"Track narrower durable work for {task_label}",
        "next_action": "Bind this session and continue the child work",
    }


def _activation_parent_choice_state(recommended_rows: list[dict] | None) -> str:
    strong_count = sum(1 for row in recommended_rows or [] if row.get("candidate_evidence_quality") == "strong")
    if strong_count == 0:
        return "none"
    if strong_count == 1:
        return "single_strong_parent"
    return "ambiguous"


_WORKSTREAM_ACTIVATION_TOPIC_STOPWORDS = {
    "active",
    "and",
    "audit",
    "check",
    "continue",
    "current",
    "for",
    "from",
    "into",
    "lane",
    "next",
    "session",
    "status",
    "task",
    "the",
    "this",
    "with",
    "workstream",
    "workstreams",
}


_WORKSTREAM_ACTIVATION_WEAK_TOPIC_TOKENS = {
    "action",
    "actions",
    "amend",
    "adversarial",
    "beta",
    "binding",
    "contract",
    "cognitive",
    "decision",
    "design",
    "dev",
    "environment",
    "gauntlet",
    "health",
    "implement",
    "implementation",
    "official",
    "output",
    "pipeline",
    "pith",
    "plan",
    "planning",
    "project",
    "quality",
    "rca",
    "review",
    "run",
    "spec",
    "status",
    "turn",
    "work",
    "workflow",
    "write",
}


def _activation_topic_tokens(*values: object) -> set[str]:
    text = " ".join(str(value or "") for value in values).lower()
    raw = re.findall(r"[a-z0-9]{3,}", text.replace("_", " ").replace("-", " "))
    return {token for token in raw if token not in _WORKSTREAM_ACTIVATION_TOPIC_STOPWORDS}


def _activation_candidate_tokens(candidate: dict) -> set[str]:
    return _activation_topic_tokens(
        candidate.get("title"),
        candidate.get("current_objective"),
        candidate.get("next_action"),
        candidate.get("status"),
    )


def _activation_has_topic_overlap(candidate: dict, query_tokens: set[str]) -> bool:
    if not query_tokens:
        return False
    overlap = query_tokens & _activation_candidate_tokens(candidate)
    strong_overlap = overlap - _WORKSTREAM_ACTIVATION_WEAK_TOPIC_TOKENS
    return len(strong_overlap) >= 2 or (len(strong_overlap) >= 1 and len(overlap) >= 2)


def _non_exact_workstream_recommendations_enabled() -> bool:
    try:
        from app.core.config import get_feature_flag

        return bool(get_feature_flag("WORKSTREAMS_NON_EXACT_RECOMMENDATIONS_ENABLED", False))
    except Exception:
        logger.debug("WORKSTREAMS: failed to read non-exact recommendation flag", exc_info=True)
        return False


def _active_binding_relatedness(thread_id: str | None, query_tokens: set[str]) -> bool | None:
    if not thread_id or not query_tokens:
        return None
    thread = load_thread(thread_id)
    if not thread:
        return None
    workstream = thread.workstream
    candidate = {
        "title": thread.title,
        "current_objective": workstream.current_objective if workstream else "",
        "next_action": workstream.next_action if workstream else "",
        "status": thread.status,
    }
    return _activation_has_topic_overlap(candidate, query_tokens)


def _build_workstream_activation_decision(
    *,
    active_binding: dict | None,
    explicit_skip: dict | None,
    recommended_rows: list[dict] | None = None,
    recommended_count: int,
    possible_match_count: int = 0,
    advisory_candidate_count: int = 0,
    proof_or_maintenance_count: int,
    needs_review_count: int,
    current_task_id: str | None,
    active_binding_related: bool | None = None,
) -> dict:
    current_task_present = bool(str(current_task_id or "").strip())
    base = {
        "read_only": True,
        "current_task_id_present": current_task_present,
        "active_binding_related": active_binding_related,
        "recommended_count": recommended_count,
        "possible_match_count": possible_match_count,
        "advisory_candidate_count": advisory_candidate_count,
        "proof_or_maintenance_count": proof_or_maintenance_count,
        "needs_review_count": needs_review_count,
        "skip_allowed": True,
        "skip_requires_reason": True,
        "skip_exception_kinds": sorted(WORKSTREAM_ACTIVATION_SKIP_EXCEPTION_KINDS),
        "suggested_create_metadata": _activation_create_metadata(current_task_id),
        "decision_options": _activation_decision_options(),
    }
    child_metadata = _activation_child_metadata(current_task_id, recommended_rows)
    parent_choice_state = _activation_parent_choice_state(recommended_rows)
    if not current_task_present:
        return {**base, "decision_kind": "not_required", "required_action": "none"}
    if active_binding:
        if active_binding_related is True:
            return {**base, "decision_kind": "active_binding_related", "required_action": "none"}
        if active_binding_related is False:
            return {
                **base,
                "decision_kind": "active_binding_unrelated",
                "required_action": "choose_bind_or_create",
                "recommended_next_action": "operator_choose",
            }
        return {
            **base,
            "decision_kind": "active_binding_unknown",
            "required_action": "confirm_active_binding_or_create",
        }
    if explicit_skip:
        context = explicit_skip.get("context") or {}
        return {
            **base,
            "decision_kind": "explicit_skip_exception",
            "required_action": "none",
            "skip_exception_kind": _normalize_skip_exception_kind(context.get("skip_exception_kind")),
        }
    if recommended_count > 0:
        decision = {
            **base,
            "decision_kind": "bind_or_create_required",
            "required_action": "choose_bind_or_create",
            "recommended_next_action": "operator_choose",
            "parent_choice_state": parent_choice_state,
        }
        if child_metadata is not None:
            decision["suggested_child_metadata"] = child_metadata
        return decision
    if advisory_candidate_count > 0:
        return {
            **base,
            "decision_kind": "operator_review_required",
            "required_action": "create_or_skip_or_confirm_candidate",
            "recommended_next_action": "operator_choose",
        }
    return {**base, "decision_kind": "create_required", "required_action": "create_and_bind_workstream"}


def _record_workstream_metric(name: str, tags: dict[str, object]) -> None:
    try:
        from app.ops.metrics import metrics

        metrics.record(name, 1.0, {key: str(value) for key, value in tags.items()})
    except Exception:
        pass


def _checkpoint_row_to_dict(row, *, selection_source: str, selection_authority: str) -> dict | None:
    if not row:
        return None
    return {
        "task_id": row["task_id"],
        "session_id": row["session_id"],
        "origin_id": row["origin_id"] if "origin_id" in row.keys() else None,
        "selection_source": selection_source,
        "selection_authority": selection_authority,
        "status": row["status"],
        "description": row["description"],
        "done": json.loads(row["done"]) if row["done"] else [],
        "active": row["active"] or "",
        "next": json.loads(row["next"]) if row["next"] else [],
        "blockers": json.loads(row["blockers"]) if row["blockers"] else [],
        "context": json.loads(row["context"]) if row["context"] else {},
        "concept_refs": json.loads(row["concept_refs"]) if row["concept_refs"] else [],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "expires_at": row["expires_at"],
        "save_count": row["save_count"],
    }


def _thread_metadata_dict(thread: NarrativeThread) -> dict | None:
    if thread.workstream is None:
        return None
    return thread.workstream.model_dump()


def create_workstream(
    title: str,
    description: str = "",
    urgency: str = "normal",
    goal_ids: list[str] | None = None,
    knowledge_areas: list[str] | None = None,
    agent_id: str = "default",
    current_objective: str = "",
    current_summary: str = "",
    next_action: str = "",
    blockers: list[str] | None = None,
    quality_state: str = "ok",
    created_by: str = "user",
    parent_workstream_id: str | None = None,
    parent_title: str | None = None,
    relationship: str | None = None,
) -> NarrativeThread:
    """Create an explicit user-curated Workstream on top of a narrative thread."""
    workstream_knowledge_areas = _normalize_workstream_knowledge_areas(knowledge_areas)
    thread = create_thread(
        title=title,
        description=description,
        urgency=urgency,
        goal_ids=goal_ids,
        knowledge_areas=None,
        agent_id=agent_id,
    )
    thread.knowledge_areas = workstream_knowledge_areas
    thread.workstream = _build_workstream_metadata(
        current_objective=current_objective,
        current_summary=current_summary,
        next_action=next_action,
        blockers=blockers,
        quality_state=quality_state,
        created_by=created_by,
        updated_by=created_by,
        parent_workstream_id=parent_workstream_id,
        parent_title=parent_title,
        relationship=relationship,
    )
    thread.updated_at = _utc_now_iso()
    save_thread(thread)
    logger.info("workstream_created thread_id=%s status=ok", thread.id)
    return thread


def promote_thread_to_workstream(
    thread_id: str,
    metadata: dict | None = None,
    operator_mode: bool = False,
) -> NarrativeThread:
    """Attach Workstream metadata to an existing non-maintenance thread."""
    thread = load_thread(thread_id)
    if not thread:
        raise ValueError("thread_not_found")
    classification = _single_thread_classification(thread.id, agent_id=thread.agent_id)
    if thread.title.startswith(WORKSTREAM_MAINTENANCE_PREFIX) or (
        classification and classification["class"] == "maintenance_cluster"
    ):
        raise ValueError("maintenance_cluster_not_promotable")

    payload = metadata or {}
    if thread.workstream is None:
        thread.workstream = _build_workstream_metadata(
            current_objective=payload.get("current_objective", ""),
            current_summary=payload.get("current_summary", ""),
            next_action=payload.get("next_action", ""),
            blockers=payload.get("blockers"),
            quality_state=payload.get("quality_state", "ok"),
            created_by=payload.get("created_by", "user"),
            updated_by=payload.get("updated_by", payload.get("created_by", "user")),
            parent_workstream_id=payload.get("parent_workstream_id"),
            parent_title=payload.get("parent_title"),
            relationship=payload.get("relationship"),
        )
    else:
        updates = _metadata_updates_from_payload(payload)
        for key, value in updates.items():
            setattr(thread.workstream, key, value)

    thread.updated_at = _utc_now_iso()
    save_thread(thread)
    logger.info("workstream_promoted thread_id=%s status=ok", thread.id)
    return thread


def update_workstream_metadata(thread_id: str, updates: dict) -> NarrativeThread:
    """Edit explicit Workstream metadata while preserving the underlying thread."""
    thread = load_thread(thread_id)
    if not thread:
        raise ValueError("thread_not_found")
    if thread.workstream is None:
        raise ValueError("not_workstream")

    metadata_updates = _metadata_updates_from_payload(updates)
    for key, value in metadata_updates.items():
        setattr(thread.workstream, key, value)
    if updates.get("urgency"):
        urgency = str(updates["urgency"]).strip()
        thread.urgency = urgency if urgency in ("low", "normal", "high") else "normal"
    thread.updated_at = _utc_now_iso()
    save_thread(thread)
    logger.info("workstream_updated thread_id=%s status=ok", thread.id)
    return thread


def bind_workstream_checkpoint(
    thread_id: str,
    origin_id: str | None = None,
    session_id: str | None = None,
    current_task_id: str | None = None,
    authority_mode: str = "auto",
    op_id: int | None = None,
    payload_hash: str | None = None,
) -> dict:
    """Persist an origin/session-scoped active Workstream binding via checkpoints."""
    thread = load_thread(thread_id)
    if not thread:
        raise ValueError("thread_not_found")
    if thread.workstream is None:
        raise ValueError("not_workstream")

    normalized_authority_mode = str(authority_mode or "auto").strip().lower()
    if normalized_authority_mode not in {"legacy", "auto", "composite", "session_task"}:
        raise ValueError("invalid_authority_mode")

    normalized_task = _normalize_activation_task_id(current_task_id) if current_task_id else None
    task_hash = _current_task_hash(normalized_task) if normalized_task else None
    normalized_origin = _normalize_binding_authority(origin_id, "origin_id")
    normalized_session = _normalize_binding_authority(session_id, "session_id")

    if normalized_authority_mode == "auto":
        if normalized_task and normalized_origin and normalized_session:
            binding_authority = "composite"
        elif normalized_task and normalized_session:
            binding_authority = "session_task"
        else:
            binding_authority, binding_value = _workstream_binding_authority(
                origin_id=origin_id,
                session_id=session_id,
            )
    else:
        binding_authority = normalized_authority_mode

    if binding_authority == "composite":
        if not normalized_origin or not normalized_session or not normalized_task or not task_hash:
            raise ValueError("composite_authority_requires_origin_session_task")
        binding_value = f"{normalized_origin}|{normalized_session}|{task_hash}"
        task_id = _workstream_composite_task_id(thread.id, normalized_task)
    elif binding_authority == "session_task":
        if not normalized_session or not normalized_task or not task_hash:
            raise ValueError("session_task_authority_requires_session_task")
        binding_value = f"{normalized_session}|{task_hash}"
        task_id = _workstream_session_task_id(thread.id, normalized_task)
    else:
        binding_authority, binding_value = _workstream_binding_authority(origin_id=origin_id, session_id=session_id)
        task_id = _workstream_binding_task_id(thread.id, origin_id=origin_id, session_id=session_id)

    active = thread.workstream.next_action or thread.workstream.current_objective or thread.title
    context = {
        "workstream_thread_id": thread.id,
        "binding_source": "user",
        "binding_authority": binding_authority,
        "binding_value": binding_value,
    }
    if normalized_task and task_hash:
        context.update(
            {
                "current_task_id": normalized_task,
                "current_task_hash": task_hash,
                "authority_components": {
                    "origin_id_present": bool(normalized_origin),
                    "session_id_present": bool(normalized_session),
                    "current_task_id_present": True,
                },
            }
        )
    from app.storage import save_checkpoint

    if binding_authority in {"composite", "session_task"}:
        previous = load_exact_workstream_binding_checkpoint(
            origin_id=origin_id,
            session_id=session_id,
            current_task_id=normalized_task,
        )
        previous_context = previous.get("context", {}) if previous else {}
        previous_thread_id = previous_context.get("workstream_thread_id")
        if previous and previous_thread_id and previous_thread_id != thread.id:
            from app.storage import complete_checkpoint

            complete_checkpoint(previous["task_id"])

    checkpoint_op_id = op_id if binding_authority == "origin_id" else None
    checkpoint_payload_hash = payload_hash if binding_authority == "origin_id" else None

    result = save_checkpoint(
        task_id=task_id,
        description=f"Active Workstream binding: {thread.title}",
        status="active",
        active=active,
        next_items=[],
        blockers=thread.workstream.blockers,
        context=context,
        concept_refs=thread.concept_ids[:WORKSTREAM_REF_LIMIT_MAX],
        session_id=session_id,
        origin_id=origin_id,
        op_id=checkpoint_op_id,
        payload_hash=checkpoint_payload_hash,
    )
    result["context"] = context
    result["origin_id"] = origin_id
    result["session_id"] = session_id
    logger.info("workstream_bound thread_id=%s authority=%s status=ok", thread.id, binding_authority)
    return {"status": "ok", "binding_status": binding_authority, "thread_id": thread.id, "checkpoint": result}


def load_exact_workstream_binding_checkpoint(
    origin_id: str | None = None,
    session_id: str | None = None,
    current_task_id: str | None = None,
    max_age_hours: int = 24,
) -> dict | None:
    """Load task-scoped Workstream binding checkpoints before broad origin/session fallbacks."""
    if not current_task_id or not session_id:
        return None
    normalized_task = _normalize_activation_task_id(current_task_id)
    task_hash = _current_task_hash(normalized_task)
    max_age_hours = max_age_hours or 24
    cutoff = (_utc_now() - timedelta(hours=max_age_hours)).isoformat()
    from app.storage.connection import read_snapshot_db

    normalized_origin = _normalize_binding_authority(origin_id, "origin_id") if origin_id else None
    normalized_session = _normalize_binding_authority(session_id, "session_id")
    with read_snapshot_db("load_exact_workstream_binding_checkpoint") as conn:
        if normalized_origin:
            row = conn.execute(
                """
                SELECT * FROM checkpoints
                WHERE status NOT IN ('complete', 'archived')
                  AND expires_at > ?
                  AND updated_at > ?
                  AND task_id LIKE ?
                  AND origin_id = ?
                  AND session_id = ?
                  AND json_extract(context, '$.binding_authority') = 'composite'
                  AND json_extract(context, '$.current_task_hash') = ?
                  AND json_extract(context, '$.workstream_thread_id') IS NOT NULL
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (_utc_now_iso(), cutoff, f"workstream-composite:%:{task_hash}", normalized_origin, normalized_session, task_hash),
            ).fetchone()
            if row:
                return _checkpoint_row_to_dict(
                    row,
                    selection_source="composite_task",
                    selection_authority="authoritative",
                )

        row = conn.execute(
            """
            SELECT * FROM checkpoints
            WHERE status NOT IN ('complete', 'archived')
              AND expires_at > ?
              AND updated_at > ?
              AND task_id LIKE ?
              AND session_id = ?
              AND json_extract(context, '$.binding_authority') = 'session_task'
              AND json_extract(context, '$.current_task_hash') = ?
              AND json_extract(context, '$.workstream_thread_id') IS NOT NULL
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (_utc_now_iso(), cutoff, f"workstream-session-task:%:{task_hash}", normalized_session, task_hash),
        ).fetchone()
    if row:
        return _checkpoint_row_to_dict(
            row,
            selection_source="session_task",
            selection_authority="authoritative",
        )
    return None


def load_active_workstream_binding_checkpoint(
    origin_id: str | None = None,
    session_id: str | None = None,
    max_age_hours: int = 24,
) -> dict | None:
    """Load only active Workstream binding checkpoints for an origin/session authority."""
    authority_type, authority_value = _workstream_activation_authority(origin_id=origin_id, session_id=session_id)
    max_age_hours = max_age_hours or 24
    cutoff = (_utc_now() - timedelta(hours=max_age_hours)).isoformat()
    from app.storage.connection import read_snapshot_db

    authority_predicate = "origin_id = ?" if authority_type == "origin_id" else "session_id = ?"
    with read_snapshot_db("load_active_workstream_binding_checkpoint") as conn:
        row = conn.execute(
            f"""
            SELECT * FROM checkpoints
            WHERE status NOT IN ('complete', 'archived')
              AND expires_at > ?
              AND updated_at > ?
              AND task_id LIKE 'workstream:%'
              AND json_extract(context, '$.workstream_thread_id') IS NOT NULL
              AND {authority_predicate}
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (_utc_now_iso(), cutoff, authority_value),
        ).fetchone()
    return _checkpoint_row_to_dict(row, selection_source=authority_type, selection_authority="authoritative")


def clear_workstream_binding(
    thread_id: str | None = None,
    origin_id: str | None = None,
    session_id: str | None = None,
    current_task_id: str | None = None,
) -> dict:
    """Complete the matching binding checkpoint without mutating the Workstream."""
    if current_task_id and session_id:
        checkpoint = load_exact_workstream_binding_checkpoint(
            origin_id=origin_id,
            session_id=session_id,
            current_task_id=current_task_id,
        )
        if checkpoint:
            from app.storage import complete_checkpoint

            context = checkpoint.get("context") or {}
            result = complete_checkpoint(checkpoint["task_id"])
            if result:
                target_thread_id = context.get("workstream_thread_id")
                binding_source = checkpoint.get("selection_source") or "exact_task"
                logger.info("workstream_binding_cleared thread_id=%s authority=%s status=ok", target_thread_id, binding_source)
                return {
                    "status": "ok",
                    "binding_status": "cleared",
                    "thread_id": target_thread_id,
                    "checkpoint": result,
                }

    binding_authority, _ = _workstream_binding_authority(origin_id=origin_id, session_id=session_id)
    target_thread_id = str(thread_id).strip() if thread_id else None
    if target_thread_id is None:
        checkpoint = load_active_workstream_binding_checkpoint(origin_id=origin_id, session_id=session_id)
        context = checkpoint.get("context", {}) if checkpoint else {}
        target_thread_id = context.get("workstream_thread_id")
    if not target_thread_id:
        return {"status": "not_found", "binding_status": "not_found"}

    task_id = _workstream_binding_task_id(target_thread_id, origin_id=origin_id, session_id=session_id)
    from app.storage import complete_checkpoint

    result = complete_checkpoint(task_id)
    if not result:
        return {"status": "not_found", "binding_status": "not_found", "thread_id": target_thread_id}
    logger.info("workstream_binding_cleared thread_id=%s authority=%s status=ok", target_thread_id, binding_authority)
    return {"status": "ok", "binding_status": "cleared", "thread_id": target_thread_id, "checkpoint": result}


def _load_workstream_skip(
    origin_id: str | None = None,
    session_id: str | None = None,
    current_task_id: str | None = None,
) -> dict | None:
    try:
        authority_type, authority_value = _workstream_activation_authority(origin_id=origin_id, session_id=session_id)
    except ValueError:
        return None
    task_pattern = f"{WORKSTREAM_ACTIVATION_SKIP_PREFIX}{authority_type}:{authority_value}:%"
    params: list[object] = [_utc_now_iso(), task_pattern]
    current_task_hash = None
    if current_task_id:
        normalized_task = _normalize_activation_task_id(current_task_id)
        current_task_hash = hashlib.sha256(normalized_task.encode("utf-8")).hexdigest()[:16]
    from app.storage.connection import read_snapshot_db

    with read_snapshot_db("load_workstream_activation_skip") as conn:
        rows = conn.execute(
            """
            SELECT * FROM checkpoints
            WHERE status NOT IN ('complete', 'archived')
              AND expires_at > ?
              AND task_id LIKE ?
              AND json_extract(context, '$.workstream_activation_skip') = 1
            ORDER BY updated_at DESC
            LIMIT 20
            """,
            tuple(params),
        ).fetchall()
    for row in rows:
        checkpoint = _checkpoint_row_to_dict(row, selection_source=authority_type, selection_authority="authoritative")
        if current_task_hash is None or str(checkpoint["task_id"]).endswith(f":{current_task_hash}"):
            return checkpoint
    return None


def build_workstream_activation_hint(
    origin_id: str | None = None,
    session_id: str | None = None,
    current_task_id: str | None = None,
) -> dict:
    """Return compact read-only Workstream activation state for conversation_turn."""
    from app.core.config import get_feature_flag

    if not get_feature_flag("WORKSTREAMS_READ_ENABLED", True):
        return {
            "status": "disabled",
            "activation_state": "disabled",
            "reason": "WORKSTREAMS_READ_ENABLED is false",
            "read_only": True,
            "decision_needed": False,
            "activation_decision": {
                "read_only": True,
                "decision_kind": "unavailable",
                "required_action": "retry_or_run_pith_api_workstreams_candidate",
            },
        }

    if not origin_id and not session_id:
        return {
            "status": "ok",
            "activation_state": "unavailable",
            "reason": "authority_required",
            "read_only": True,
            "decision_needed": False,
            "origin_id_present": False,
            "session_id_present": False,
            "current_task_id_present": bool(current_task_id),
            "activation_decision": {
                "read_only": True,
                "decision_kind": "authority_required",
                "required_action": "provide_origin_id_or_session_id",
            },
        }

    try:
        active_checkpoint = load_exact_workstream_binding_checkpoint(
            origin_id=origin_id,
            session_id=session_id,
            current_task_id=current_task_id,
        )
        if active_checkpoint is None:
            active_checkpoint = load_active_workstream_binding_checkpoint(
                origin_id=origin_id,
                session_id=session_id,
            )
    except ValueError as exc:
        return {
            "status": "ok",
            "activation_state": "unavailable",
            "reason": str(exc),
            "read_only": True,
            "decision_needed": False,
            "activation_decision": {
                "read_only": True,
                "decision_kind": "unavailable",
                "required_action": "provide_origin_id_or_session_id",
            },
        }

    if active_checkpoint:
        context = active_checkpoint.get("context", {})
        binding_source = active_checkpoint.get("selection_source")
        return {
            "status": "ok",
            "activation_state": "active_binding",
            "read_only": True,
            "active_binding": {
                "status": active_checkpoint.get("status"),
                "binding_source": binding_source,
                "binding_status": binding_source,
                "thread_id": context.get("workstream_thread_id"),
                "checkpoint_task_id": active_checkpoint.get("task_id"),
            },
            "explicit_skip": None,
            "decision_needed": False,
            "origin_id_present": bool(origin_id),
            "session_id_present": bool(session_id),
            "current_task_id_present": bool(current_task_id),
            "activation_decision": _build_workstream_activation_decision(
                active_binding={"thread_id": context.get("workstream_thread_id")},
                explicit_skip=None,
                recommended_count=0,
                proof_or_maintenance_count=0,
                needs_review_count=0,
                current_task_id=current_task_id,
                active_binding_related=True if binding_source in {"composite_task", "session_task"} else None,
            ),
        }

    explicit_skip = _load_workstream_skip(
        origin_id=origin_id,
        session_id=session_id,
        current_task_id=current_task_id,
    )
    if explicit_skip:
        skip_context = explicit_skip.get("context") or {}
        return {
            "status": "ok",
            "activation_state": "explicit_skip",
            "read_only": True,
            "active_binding": None,
            "explicit_skip": {
                "status": explicit_skip.get("status"),
                "checkpoint_task_id": explicit_skip.get("task_id"),
                "current_task_id": skip_context.get("current_task_id"),
                "skip_exception_kind": _normalize_skip_exception_kind(skip_context.get("skip_exception_kind")),
            },
            "decision_needed": False,
            "origin_id_present": bool(origin_id),
            "session_id_present": bool(session_id),
            "current_task_id_present": bool(current_task_id),
            "activation_decision": _build_workstream_activation_decision(
                active_binding=None,
                explicit_skip=explicit_skip,
                recommended_count=0,
                proof_or_maintenance_count=0,
                needs_review_count=0,
                current_task_id=current_task_id,
            ),
        }

    return {
        "status": "ok",
        "activation_state": "decision_needed",
        "read_only": True,
        "active_binding": None,
        "explicit_skip": None,
        "decision_needed": True,
        "candidate_detail_available": True,
        "origin_id_present": bool(origin_id),
        "session_id_present": bool(session_id),
        "current_task_id_present": bool(current_task_id),
        "activation_decision": _build_workstream_activation_decision(
            active_binding=None,
            explicit_skip=None,
            recommended_count=0,
            proof_or_maintenance_count=0,
            needs_review_count=0,
            current_task_id=current_task_id,
        ),
    }


def _record_workstream_skip(
    origin_id: str | None = None,
    session_id: str | None = None,
    current_task_id: str | None = None,
    skip_reason: str | None = None,
    skip_exception_kind: str | None = None,
    op_id: int | None = None,
    payload_hash: str | None = None,
) -> dict:
    authority_type, authority_value = _workstream_activation_authority(origin_id=origin_id, session_id=session_id)
    normalized_task = _normalize_activation_task_id(current_task_id)
    reason = _normalize_workstream_text(skip_reason, max_len=WORKSTREAM_ACTIVATION_SKIP_REASON_MAX)
    if not reason:
        raise ValueError("skip_reason_required")
    task_id = _workstream_skip_task_id(authority_type, authority_value, normalized_task)
    context = {
        "workstream_activation_skip": True,
        "skip_reason": reason,
        "skip_exception_kind": _normalize_skip_exception_kind(skip_exception_kind),
        "binding_authority": authority_type,
        "binding_value": authority_value,
        "current_task_id": normalized_task,
    }
    from app.storage import save_checkpoint

    checkpoint = save_checkpoint(
        task_id=task_id,
        description=f"Workstream activation skipped: {reason[:120]}",
        status="active",
        active="Workstream activation explicitly skipped",
        next_items=[],
        blockers=[],
        context=context,
        concept_refs=[],
        session_id=session_id,
        origin_id=origin_id,
        op_id=op_id if authority_type == "origin_id" else None,
        payload_hash=payload_hash if authority_type == "origin_id" else None,
    )
    checkpoint["context"] = context
    return {"status": "ok", "skip_status": "recorded", "checkpoint": checkpoint}


def _candidate_evidence_quality(candidate: dict, query_tokens: set[str]) -> tuple[str, str]:
    tier = str(candidate.get("effective_discovery_tier") or candidate.get("discovery_tier") or "")
    if tier in WORKSTREAM_DISCOVERY_HIDDEN_TIERS:
        return "polluted", f"discovery_tier:{tier}"
    quality_flags = set(candidate.get("quality_flags") or [])
    candidate_class = candidate.get("class")
    if _is_proof_or_maintenance_workstream(candidate) or candidate_class != "workstream_candidate":
        return "polluted", "proof_or_maintenance_or_non_workstream_class"
    if {"high_volume", "insufficient_signal", "migration_suspect"} & quality_flags:
        return "polluted", "quality_flags_require_review"
    has_overlap = _activation_has_topic_overlap(candidate, query_tokens)
    link_count = int(candidate.get("link_count") or 0)
    user_links = int(candidate.get("user_links") or 0)
    if has_overlap and (user_links > 0 or link_count > 0):
        return "strong", "linked_evidence_with_title_objective_overlap"
    if has_overlap:
        return "weak", "title_objective_overlap_without_links"
    return "unknown", "no_reliable_overlap_evidence"


def _redact_activation_candidate(candidate: dict, *, query_tokens: set[str] | None = None) -> dict:
    allowed = (
        "thread_id",
        "title",
        "status",
        "urgency",
        "agent_id",
        "updated_at",
        "last_activity_at",
        "class",
        "quality_flags",
        "link_count",
        "user_links",
        "auto_links",
        "current_objective",
        "next_action",
        "legacy_class",
        "discovery_tier",
        "effective_discovery_tier",
        "discovery_reason_codes",
        "binding_visibility_warning",
        "read_only",
    )
    redacted = {key: candidate.get(key) for key in allowed if key in candidate}
    quality, reason = _candidate_evidence_quality(candidate, query_tokens or set())
    redacted["candidate_evidence_quality"] = quality
    redacted["candidate_evidence_reason"] = reason
    redacted["read_only"] = True
    return redacted


def _is_proof_or_maintenance_workstream(candidate: dict) -> bool:
    if candidate.get("class") == "maintenance_cluster":
        return True
    text = " ".join(
        str(candidate.get(key) or "")
        for key in ("title", "description", "current_objective", "current_summary", "next_action")
    ).lower()
    return any(term in text for term in _WORKSTREAM_PROOF_TERMS)


def _split_workstream_activation_candidates(
    candidates: list[dict],
    *,
    current_task_id: str | None = None,
    situation: str | None = None,
    origin_id: str | None = None,
    session_id: str | None = None,
) -> dict:
    buckets = {
        "recommended": [],
        "advisory_candidates": [],
        "possible_matches": [],
        "proof_or_maintenance": [],
        "needs_review": [],
    }
    query_tokens = _activation_topic_tokens(current_task_id, situation)
    query_text_present = bool(str(current_task_id or "").strip() or str(situation or "").strip())
    non_exact_recommendations_enabled = _non_exact_workstream_recommendations_enabled()
    for candidate in candidates:
        redacted = _redact_activation_candidate(candidate, query_tokens=query_tokens)
        _record_workstream_metric(
            "workstream_candidate_evidence_quality",
            {
                "selection_source": "candidate",
                "decision_kind": redacted.get("candidate_evidence_quality"),
                "origin_id_present": bool(origin_id),
                "session_id_present": bool(session_id),
                "current_task_id_present": bool(current_task_id),
                "read_only": True,
            },
        )
        tier = candidate.get("effective_discovery_tier") or candidate.get("discovery_tier")
        if tier == "proof_or_maintenance" or _is_proof_or_maintenance_workstream(candidate):
            buckets["proof_or_maintenance"].append(redacted)
        elif tier in {"terminal_archive", "stale_auto_debug", "needs_hygiene_review"}:
            buckets["needs_review"].append(redacted)
        elif tier == "recent_auto_advisory":
            if not query_text_present or _activation_has_topic_overlap(candidate, query_tokens):
                buckets["advisory_candidates"].append(redacted)
            else:
                buckets["possible_matches"].append(redacted)
        elif tier == "curated_candidate":
            if not query_text_present or _activation_has_topic_overlap(candidate, query_tokens):
                if non_exact_recommendations_enabled:
                    buckets["recommended"].append(redacted)
                else:
                    buckets["advisory_candidates"].append(redacted)
            else:
                buckets["possible_matches"].append(redacted)
        else:
            candidate_class = candidate.get("class")
            quality_flags = set(candidate.get("quality_flags") or [])
            if candidate_class != "workstream_candidate" or (
                {"high_volume", "insufficient_signal", "migration_suspect"} & quality_flags
            ):
                buckets["needs_review"].append(redacted)
            elif not query_text_present or _activation_has_topic_overlap(candidate, query_tokens):
                if non_exact_recommendations_enabled:
                    buckets["recommended"].append(redacted)
                else:
                    buckets["advisory_candidates"].append(redacted)
            else:
                buckets["possible_matches"].append(redacted)
    counts = {name: len(rows) for name, rows in buckets.items()}
    capped = {name: rows[:WORKSTREAM_ACTIVATION_BUCKET_LIMIT] for name, rows in buckets.items()}
    return {**capped, "counts": counts}


def ensure_workstream_activation(
    mode: str,
    origin_id: str | None = None,
    session_id: str | None = None,
    current_task_id: str | None = None,
    situation: str | None = None,
    thread_id: str | None = None,
    metadata: dict | None = None,
    skip_reason: str | None = None,
    skip_exception_kind: str | None = None,
    operator_confirmed: bool = False,
    include_proof_candidates: bool = False,
    op_id: int | None = None,
    payload_hash: str | None = None,
) -> dict:
    """Explicit operator bridge for candidate, bind, create-and-bind, or skip decisions."""
    normalized_mode = str(mode or "candidate").strip().lower()
    if normalized_mode not in {"candidate", "bind_existing", "create_and_bind", "skip"}:
        return {"status": "rejected", "mode": normalized_mode, "read_only": True, "reason": "invalid_mode"}

    if normalized_mode == "candidate":
        active = None
        explicit_skip = None
        try:
            if origin_id or session_id:
                active_checkpoint = load_exact_workstream_binding_checkpoint(
                    origin_id=origin_id,
                    session_id=session_id,
                    current_task_id=current_task_id,
                )
                if active_checkpoint is None:
                    active_checkpoint = load_active_workstream_binding_checkpoint(
                        origin_id=origin_id,
                        session_id=session_id,
                    )
                if active_checkpoint:
                    context = active_checkpoint.get("context", {})
                    active_thread_id = context.get("workstream_thread_id")
                    active_classification = (
                        _single_thread_classification(active_thread_id) if active_thread_id else None
                    )
                    active = {
                        "status": active_checkpoint.get("status"),
                        "binding_source": active_checkpoint.get("selection_source"),
                        "binding_status": active_checkpoint.get("selection_source"),
                        "thread_id": active_thread_id,
                        "checkpoint_task_id": active_checkpoint.get("task_id"),
                        "effective_discovery_tier": (
                            active_classification.get("effective_discovery_tier") if active_classification else None
                        ),
                        "discovery_reason_codes": (
                            active_classification.get("discovery_reason_codes") if active_classification else []
                        ),
                        "binding_visibility_warning": _binding_visibility_warning(active_classification),
                    }
                explicit_skip = _load_workstream_skip(
                    origin_id=origin_id,
                    session_id=session_id,
                    current_task_id=current_task_id,
                )
        except ValueError as exc:
            return {"status": "rejected", "mode": normalized_mode, "read_only": True, "reason": str(exc)}
        classified = classify_workstream_threads(include_maintenance=True, limit=WORKSTREAM_CLASSIFIER_LIMIT_MAX)
        split = _split_workstream_activation_candidates(
            classified.get("threads") or [],
            current_task_id=current_task_id,
            situation=situation,
            origin_id=origin_id,
            session_id=session_id,
        )
        proof_rows = split["proof_or_maintenance"] if include_proof_candidates else split["proof_or_maintenance"]
        counts = split["counts"]
        active_related = _active_binding_relatedness(
            active.get("thread_id") if active else None,
            _activation_topic_tokens(current_task_id, situation),
        )
        if active and active.get("binding_source") in {"composite_task", "session_task"}:
            active_related = True
        activation_decision = _build_workstream_activation_decision(
            active_binding=active,
            explicit_skip=explicit_skip,
            recommended_rows=split["recommended"],
            recommended_count=counts["recommended"],
            possible_match_count=counts["possible_matches"],
            advisory_candidate_count=counts["advisory_candidates"],
            proof_or_maintenance_count=counts["proof_or_maintenance"],
            needs_review_count=counts["needs_review"],
            current_task_id=current_task_id,
            active_binding_related=active_related,
        )
        _record_workstream_metric(
            "workstream_activation_decision",
            {
                "selection_source": active.get("binding_source") if active else "none",
                "decision_kind": activation_decision.get("decision_kind"),
                "required_action": activation_decision.get("required_action"),
                "recommended_next_action": activation_decision.get("recommended_next_action"),
                "parent_choice_state": activation_decision.get("parent_choice_state"),
                "advisory_candidate_count": activation_decision.get("advisory_candidate_count"),
                "origin_id_present": bool(origin_id),
                "session_id_present": bool(session_id),
                "current_task_id_present": bool(current_task_id),
                "read_only": True,
            },
        )
        return {
            "status": "ok",
            "mode": "candidate",
            "read_only": True,
            "active_binding": active,
            "explicit_skip": (
                {
                    "status": explicit_skip.get("status"),
                    "checkpoint_task_id": explicit_skip.get("task_id"),
                    "skip_reason": (explicit_skip.get("context") or {}).get("skip_reason"),
                    "skip_exception_kind": _normalize_skip_exception_kind(
                        (explicit_skip.get("context") or {}).get("skip_exception_kind")
                    ),
                    "current_task_id": (explicit_skip.get("context") or {}).get("current_task_id"),
                }
                if explicit_skip
                else None
            ),
            "recommended": split["recommended"],
            "advisory_candidates": split["advisory_candidates"],
            "possible_matches": split["possible_matches"],
            "proof_or_maintenance": proof_rows,
            "needs_review": split["needs_review"],
            "counts": counts,
            "activation_decision": activation_decision,
        }

    if operator_confirmed is not True:
        return {
            "status": "rejected",
            "mode": normalized_mode,
            "read_only": True,
            "reason": "operator_confirmation_required",
        }

    try:
        authority_type, authority_value = _workstream_activation_authority(origin_id=origin_id, session_id=session_id)
    except ValueError as exc:
        return {"status": "rejected", "mode": normalized_mode, "read_only": True, "reason": str(exc)}

    if normalized_mode == "skip":
        try:
            result = _record_workstream_skip(
                origin_id=origin_id,
                session_id=session_id,
                current_task_id=current_task_id,
                skip_reason=skip_reason,
                skip_exception_kind=skip_exception_kind,
                op_id=op_id,
                payload_hash=payload_hash,
            )
        except ValueError as exc:
            return {"status": "rejected", "mode": normalized_mode, "read_only": True, "reason": str(exc)}
        checkpoint = result["checkpoint"]
        return {
            "status": "ok",
            "mode": "skip",
            "read_only": False,
            "skip_source": authority_type,
            "binding_authority": authority_type,
            "binding_value": authority_value,
            "checkpoint_task_id": checkpoint.get("task_id"),
            "checkpoint": checkpoint,
        }

    if normalized_mode == "bind_existing":
        normalized_thread_id = str(thread_id or "").strip()
        if not normalized_thread_id:
            return {"status": "rejected", "mode": normalized_mode, "read_only": True, "reason": "thread_id_required"}
        thread = load_thread(normalized_thread_id)
        if not thread:
            return {"status": "not_found", "mode": normalized_mode, "read_only": True}
        if thread.workstream is None:
            return {"status": "rejected", "mode": normalized_mode, "read_only": True, "reason": "not_workstream"}
        previous = load_exact_workstream_binding_checkpoint(
            origin_id=origin_id,
            session_id=session_id,
            current_task_id=current_task_id,
        )
        if previous is None:
            previous = load_active_workstream_binding_checkpoint(origin_id=origin_id, session_id=session_id)
        previous_context = previous.get("context", {}) if previous else {}
        previous_thread_id = previous_context.get("workstream_thread_id")
        if previous and previous_thread_id and previous_thread_id != normalized_thread_id:
            from app.storage import complete_checkpoint

            complete_checkpoint(previous["task_id"])
        bound = bind_workstream_checkpoint(
            normalized_thread_id,
            origin_id=origin_id,
            session_id=session_id,
            current_task_id=current_task_id,
            op_id=op_id,
            payload_hash=payload_hash,
        )
        bound_context = bound["checkpoint"].get("context") or {}
        bound_authority = bound.get("binding_status") or authority_type
        return {
            "status": "ok",
            "mode": "bind_existing",
            "read_only": False,
            "binding_source": bound_authority,
            "binding_authority": bound_authority,
            "binding_value": bound_context.get("binding_value", authority_value),
            "thread_id": normalized_thread_id,
            "checkpoint_task_id": bound["checkpoint"].get("task_id"),
            "checkpoint": bound["checkpoint"],
            "previous_binding": (
                {"thread_id": previous_thread_id, "checkpoint_task_id": previous.get("task_id")}
                if previous and previous_thread_id != normalized_thread_id
                else None
            ),
        }

    payload = metadata or {}
    if not isinstance(payload, dict):
        return {
            "status": "rejected",
            "mode": normalized_mode,
            "read_only": True,
            "reason": "metadata_must_be_object",
            "field": "metadata",
            "required_action": "send_metadata_object",
        }
    title = _normalize_workstream_text(payload.get("title"), max_len=500)
    if not title:
        return {
            "status": "rejected",
            "mode": normalized_mode,
            "read_only": True,
            "reason": "title_required",
            "field": "metadata.title",
            "required_action": "send_metadata_title",
        }
    try:
        thread = create_workstream(
            title=title,
            description=payload.get("description", ""),
            urgency=payload.get("urgency", "normal"),
            goal_ids=payload.get("goal_ids"),
            knowledge_areas=payload.get("knowledge_areas"),
            agent_id=payload.get("agent_id", "default"),
            current_objective=payload.get("current_objective", ""),
            current_summary=payload.get("current_summary", ""),
            next_action=payload.get("next_action", ""),
            blockers=payload.get("blockers"),
            quality_state=payload.get("quality_state", "ok"),
            created_by=payload.get("created_by", "user"),
            parent_workstream_id=payload.get("parent_workstream_id"),
            parent_title=payload.get("parent_title"),
            relationship=payload.get("relationship"),
        )
        bound = bind_workstream_checkpoint(
            thread.id,
            origin_id=origin_id,
            session_id=session_id,
            current_task_id=current_task_id,
            op_id=op_id,
            payload_hash=payload_hash,
        )
    except Exception as exc:
        if "thread" in locals():
            try:
                update_thread_status(thread.id, "abandoned", reason="activation_bind_failed")
            except Exception:
                logger.warning("workstream_activation_partial_cleanup_failed thread_id=%s", getattr(thread, "id", None))
            return {
                "status": "partial_created_unbound",
                "mode": "create_and_bind",
                "read_only": False,
                "thread_id": thread.id,
                "reason": type(exc).__name__,
            }
        if isinstance(exc, ValueError):
            return {"status": "rejected", "mode": normalized_mode, "read_only": True, "reason": str(exc)}
        raise
    bound_context = bound["checkpoint"].get("context") or {}
    bound_authority = bound.get("binding_status") or authority_type
    return {
        "status": "ok",
        "mode": "create_and_bind",
        "read_only": False,
        "binding_source": bound_authority,
        "binding_authority": bound_authority,
        "binding_value": bound_context.get("binding_value", authority_value),
        "thread": thread.model_dump(),
        "thread_id": thread.id,
        "checkpoint_task_id": bound["checkpoint"].get("task_id"),
        "checkpoint": bound["checkpoint"],
    }


def resolve_active_workstream(
    thread_id: str | None = None,
    origin_id: str | None = None,
    session_id: str | None = None,
    current_task_id: str | None = None,
    operator_mode: bool = False,
    max_refs: int = WORKSTREAM_REF_LIMIT_DEFAULT,
    include_concept_summaries: bool = True,
) -> dict:
    """Resolve explicit active Workstream binding, falling back to suggestions only."""
    if thread_id:
        result = build_workstream_context_block(
            thread_id=thread_id,
            operator_mode=operator_mode,
            max_refs=max_refs,
            include_concept_summaries=include_concept_summaries,
        )
        result["binding_status"] = "request"
        result["binding_source"] = "request"
        if result.get("status") == "ok":
            result["thread_id"] = thread_id
        return result

    checkpoint = None
    binding_status = "none"
    if session_id and current_task_id:
        checkpoint = load_exact_workstream_binding_checkpoint(
            origin_id=origin_id,
            session_id=session_id,
            current_task_id=current_task_id,
        )
        if checkpoint:
            binding_status = checkpoint.get("selection_source") or "exact_task"
    if origin_id:
        if checkpoint is None:
            checkpoint = load_active_workstream_binding_checkpoint(origin_id=origin_id)
            binding_status = "origin_id"
    if checkpoint is None and session_id:
        checkpoint = load_active_workstream_binding_checkpoint(session_id=session_id)
        binding_status = "session_id"

    context = checkpoint.get("context", {}) if checkpoint else {}
    bound_thread_id = context.get("workstream_thread_id")
    if not bound_thread_id:
        result = build_workstream_context_block(
            thread_id=None,
            operator_mode=operator_mode,
            max_refs=max_refs,
            include_concept_summaries=include_concept_summaries,
        )
        result["binding_source"] = "none"
        return result

    result = build_workstream_context_block(
        thread_id=bound_thread_id,
        operator_mode=operator_mode,
        max_refs=max_refs,
        include_concept_summaries=include_concept_summaries,
    )
    result["binding_status"] = binding_status
    result["binding_source"] = binding_status
    result["thread_id"] = bound_thread_id
    result["checkpoint_task_id"] = checkpoint.get("task_id")
    result["source_trace"] = ["checkpoints", *result.get("source_trace", [])]
    _record_workstream_metric(
        "workstream_binding_authority_resolved",
        {
            "selection_source": binding_status,
            "decision_kind": "active_binding_resolved",
            "origin_id_present": bool(origin_id),
            "session_id_present": bool(session_id),
            "current_task_id_present": bool(current_task_id),
            "read_only": True,
        },
    )
    logger.info("active_workstream_resolved thread_id=%s binding_status=%s", bound_thread_id, binding_status)
    return result


def _fetch_workstream_rows(agent_id: str = "default", include_non_workstreams: bool = False) -> list[dict]:
    """Fetch thread rows with aggregate link counts without mutating state."""
    from app.storage.connection import read_snapshot_db

    normalized_agent = _coerce_nonempty_string(agent_id)
    include_debug_rows = 1 if include_non_workstreams else 0
    with read_snapshot_db("fetch_workstream_rows") as conn:
        rows = conn.execute(
            """
            WITH link_counts AS (
                SELECT thread_id,
                       COUNT(*) AS link_count,
                       SUM(CASE WHEN added_by = 'auto' THEN 1 ELSE 0 END) AS auto_links,
                       SUM(CASE WHEN added_by = 'user' THEN 1 ELSE 0 END) AS user_links
                FROM thread_concept_links
                GROUP BY thread_id
            )
            SELECT t.id,
                   t.title,
                   t.description,
                   t.status,
                   t.urgency,
                   t.agent_id,
                   t.updated_at,
                   t.last_activity_at,
                   json_extract(t.data, '$.workstream.kind') AS workstream_kind,
                   json_extract(t.data, '$.workstream.current_objective') AS current_objective,
                   json_extract(t.data, '$.workstream.current_summary') AS current_summary,
                   json_extract(t.data, '$.workstream.next_action') AS next_action,
                   json_extract(t.data, '$.workstream.quality_state') AS quality_state,
                   json_extract(t.data, '$.workstream.discovery_state') AS discovery_state_json,
                   json_extract(t.data, '$.workstream.discovery_state.tier') AS discovery_tier,
                   json_extract(t.data, '$.workstream.discovery_state.eligible_until') AS discovery_eligible_until,
                   COALESCE(l.link_count, 0) AS link_count,
                   COALESCE(l.auto_links, 0) AS auto_links,
                   COALESCE(l.user_links, 0) AS user_links,
                   COALESCE(json_array_length(json_extract(t.data, '$.concept_ids')), 0) AS json_concept_count,
                   COALESCE(json_array_length(json_extract(t.data, '$.trace_ids')), 0) AS trace_count,
                   COALESCE(json_array_length(json_extract(t.data, '$.goal_ids')), 0) AS goal_count,
                   COALESCE(json_array_length(json_extract(t.data, '$.knowledge_areas')), 0) AS knowledge_area_count
            FROM threads t
            LEFT JOIN link_counts l ON l.thread_id = t.id
            WHERE t.agent_id = ?
              AND (? OR json_extract(t.data, '$.workstream.kind') = 'workstream')
            ORDER BY t.last_activity_at DESC
            """,
            (normalized_agent, include_debug_rows),
        ).fetchall()
    return [_row_to_dict(row) for row in rows]


def _effective_workstream_discovery_tier(candidate: dict, *, now: datetime | None = None) -> tuple[str, list[str]]:
    tier = str(candidate.get("discovery_tier") or "").strip().lower()
    if tier in WORKSTREAM_DISCOVERY_TIERS:
        eligible_until = str(candidate.get("discovery_eligible_until") or "").strip()
        if eligible_until:
            try:
                expires = _ensure_aware(datetime.fromisoformat(eligible_until.replace("Z", "+00:00")))
                current = _ensure_aware(now or datetime.now(UTC))
                if expires < current:
                    return "stale_auto_debug", ["expired_discovery_state", f"stored_tier:{tier}"]
            except (TypeError, ValueError):
                return "needs_hygiene_review", ["invalid_eligible_until", f"stored_tier:{tier}"]
        return tier, []
    return "needs_hygiene_review", ["missing_discovery_state"]


def compute_workstream_discovery_state(
    row: dict,
    *,
    evaluated_at: datetime | None = None,
    run_id: str | None = None,
) -> dict:
    current = _ensure_aware(evaluated_at or datetime.now(UTC))
    status = str(row.get("status") or "")
    link_count = _json_count(row.get("link_count"))
    auto_links = _json_count(row.get("auto_links"))
    user_links = _json_count(row.get("user_links"))
    existing_tier = str(row.get("discovery_tier") or "").strip().lower() or None
    reason_codes: list[str] = []

    if status in {"completed", "abandoned"}:
        tier = "terminal_archive"
        reason_codes.append(f"status:{status}")
    elif _is_proof_or_maintenance_workstream({**row, "class": "workstream_candidate"}):
        tier = "proof_or_maintenance"
        reason_codes.append("proof_or_maintenance_terms")
    elif existing_tier == "curated_candidate":
        tier = "curated_candidate"
        reason_codes.append("operator_promoted")
    elif user_links > 0:
        tier = "recent_auto_advisory"
        reason_codes.append("user_links_present_but_not_curated")
    elif auto_links > 0 and link_count <= 250:
        tier = "recent_auto_advisory"
        reason_codes.append("auto_only_links")
    elif link_count == 0:
        tier = "needs_hygiene_review"
        reason_codes.append("no_links")
    else:
        tier = "needs_hygiene_review"
        reason_codes.append("high_volume_or_unknown_links")

    return {
        "tier": tier,
        "reason_codes": sorted(set(reason_codes)),
        "source": "hygiene_v1",
        "run_id": run_id,
        "last_evaluated_at": current.isoformat(),
        "eligible_until": (current + timedelta(days=WORKSTREAM_DISCOVERY_DEFAULT_TTL_DAYS)).isoformat(),
        "previous_tier": existing_tier,
    }


def classify_thread_for_workstream(row: dict) -> dict:
    """Classify one thread row as a Workstream candidate or maintenance row."""
    data = dict(row or {})
    title = str(data.get("title") or "")
    status = str(data.get("status") or "")
    link_count = _json_count(data.get("link_count"))
    auto_links = _json_count(data.get("auto_links"))
    user_links = _json_count(data.get("user_links"))
    json_concept_count = _json_count(data.get("json_concept_count"))
    quality_flags: list[str] = []
    workstream_kind = str(data.get("workstream_kind") or "").strip()

    if json_concept_count != link_count:
        quality_flags.append("migration_suspect")

    if (workstream_kind and workstream_kind != "workstream") or (
        data.get("workstream_kind") is None and "workstream_kind" in data
    ):
        thread_class = "needs_review"
        quality_flags.append("non_workstream_thread")
    elif title.startswith(WORKSTREAM_MAINTENANCE_PREFIX):
        thread_class = "maintenance_cluster"
        quality_flags.append("maintenance_only")
    elif status in ("completed", "abandoned") and link_count == 0:
        thread_class = "archive_candidate"
        quality_flags.append("empty_terminal")
    elif user_links > 0:
        thread_class = "workstream_candidate"
        quality_flags.append("human_anchored")
        if link_count > 250:
            quality_flags.append("high_volume")
    elif status in ("active", "paused") and 1 <= link_count <= 250:
        thread_class = "workstream_candidate"
        quality_flags.append("inferred_active")
    elif status in ("active", "paused") and link_count > 250:
        thread_class = "needs_review"
        quality_flags.append("high_volume")
    else:
        thread_class = "needs_review"
        quality_flags.append("insufficient_signal")
    effective_tier, tier_reason_codes = _effective_workstream_discovery_tier(data)
    legacy_class = thread_class

    return {
        "thread_id": data.get("id"),
        "title": title,
        "description": data.get("description") or "",
        "status": status,
        "urgency": data.get("urgency"),
        "agent_id": data.get("agent_id"),
        "updated_at": data.get("updated_at"),
        "last_activity_at": data.get("last_activity_at"),
        "current_objective": data.get("current_objective") or "",
        "current_summary": data.get("current_summary") or "",
        "next_action": data.get("next_action") or "",
        "class": thread_class,
        "legacy_class": legacy_class,
        "quality_flags": sorted(set(quality_flags)),
        "link_count": link_count,
        "auto_links": auto_links,
        "user_links": user_links,
        "workstream_kind": workstream_kind,
        "discovery_state": data.get("discovery_state_json"),
        "discovery_tier": data.get("discovery_tier"),
        "effective_discovery_tier": effective_tier,
        "discovery_reason_codes": tier_reason_codes,
        "json_concept_count": json_concept_count,
        "trace_count": _json_count(data.get("trace_count")),
        "goal_count": _json_count(data.get("goal_count")),
        "knowledge_area_count": _json_count(data.get("knowledge_area_count")),
        "read_only": True,
    }


def classify_workstream_threads(
    agent_id: str = "default",
    include_maintenance: bool = True,
    limit: int = WORKSTREAM_CLASSIFIER_LIMIT_MAX,
    include_non_workstreams: bool = False,
) -> dict:
    """Classify existing threads for Workstream readiness without writes."""
    start = time.perf_counter()
    effective_limit = _clamp_positive_int(limit, WORKSTREAM_CLASSIFIER_LIMIT_MAX, WORKSTREAM_CLASSIFIER_LIMIT_MAX)
    rows = _fetch_workstream_rows(agent_id=agent_id, include_non_workstreams=include_non_workstreams)
    classifications = [classify_thread_for_workstream(row) for row in rows]
    if not include_maintenance:
        classifications = [c for c in classifications if c["class"] != "maintenance_cluster"]

    counts = Counter(c["class"] for c in classifications)
    tier_counts = Counter(c["effective_discovery_tier"] for c in classifications)
    visible = classifications[:effective_limit]
    truncated = len(classifications) > effective_limit
    logger.info(
        "workstream_classifier_completed agent_id=%s rows=%d elapsed_ms=%.1f",
        _coerce_nonempty_string(agent_id),
        len(classifications),
        (time.perf_counter() - start) * 1000,
    )
    return {
        "status": "ok",
        "generated_at": _utc_now_iso(),
        "classes": dict(sorted(counts.items())),
        "discovery_tiers": dict(sorted(tier_counts.items())),
        "total": len(classifications),
        "limit": effective_limit,
        "truncated": truncated,
        "threads": visible,
        "read_only": True,
    }


def _workstream_hygiene_fingerprint(row: dict) -> str:
    payload = {
        "thread_id": row.get("id"),
        "status": row.get("status"),
        "last_activity_at": row.get("last_activity_at"),
        "current_objective": row.get("current_objective") or "",
        "next_action": row.get("next_action") or "",
        "quality_state": row.get("quality_state") or "",
        "discovery_state_json": row.get("discovery_state_json") or "",
        "link_count": _json_count(row.get("link_count")),
        "auto_links": _json_count(row.get("auto_links")),
        "user_links": _json_count(row.get("user_links")),
        "json_concept_count": _json_count(row.get("json_concept_count")),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def dry_run_workstream_hygiene(agent_id: str = "default", limit: int = WORKSTREAM_CLASSIFIER_LIMIT_MAX) -> dict:
    """Compute proposed Workstream discovery state without writes."""
    start = time.perf_counter()
    effective_limit = _clamp_positive_int(limit, WORKSTREAM_CLASSIFIER_LIMIT_MAX, WORKSTREAM_CLASSIFIER_LIMIT_MAX)
    evaluated_at_dt = datetime.now(UTC)
    run_id = f"workstream-hygiene-{evaluated_at_dt.strftime('%Y%m%dT%H%M%SZ')}"
    rows = _fetch_workstream_rows(agent_id=agent_id, include_non_workstreams=False)
    fingerprints: dict[str, str] = {}
    proposed_states: dict[str, dict] = {}
    proposed_rows: list[dict] = []
    for row in rows:
        thread_id = str(row.get("id") or "")
        if not thread_id:
            continue
        state = compute_workstream_discovery_state(row, evaluated_at=evaluated_at_dt, run_id=run_id)
        fingerprint = _workstream_hygiene_fingerprint(row)
        fingerprints[thread_id] = fingerprint
        proposed_states[thread_id] = state
        proposed_rows.append(
            {
                "thread_id": thread_id,
                "title": row.get("title"),
                "status": row.get("status"),
                "current_tier": row.get("discovery_tier"),
                "proposed_tier": state["tier"],
                "reason_codes": state["reason_codes"],
                "fingerprint": fingerprint,
            }
        )
    tier_counts = Counter(state["tier"] for state in proposed_states.values())
    elapsed_ms = (time.perf_counter() - start) * 1000
    _record_workstream_metric(
        "workstream_hygiene_dry_run_completed",
        {"rows": len(proposed_states), "elapsed_ms": round(elapsed_ms, 2), "read_only": True},
    )
    return {
        "status": "ok",
        "run_id": run_id,
        "evaluated_at": evaluated_at_dt.isoformat(),
        "read_only": True,
        "rollback_required": False,
        "total": len(proposed_states),
        "tier_counts": dict(sorted(tier_counts.items())),
        "rows": proposed_rows[:effective_limit],
        "limit": effective_limit,
        "truncated": len(proposed_rows) > effective_limit,
        "fingerprints": fingerprints,
        "proposed_states": proposed_states,
    }


def _parse_hygiene_evaluated_at(value: str) -> datetime:
    if not value or not isinstance(value, str):
        raise ValueError("evaluated_at_required")
    try:
        return _ensure_aware(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError as exc:
        raise ValueError("invalid_evaluated_at") from exc


def _hygiene_rows_by_id(agent_id: str = "default") -> dict[str, dict]:
    return {str(row.get("id")): row for row in _fetch_workstream_rows(agent_id=agent_id)}


def apply_workstream_hygiene(
    run_id: str,
    evaluated_at: str,
    fingerprints: dict[str, str],
    proposed_states: dict[str, dict],
    operator_confirmed: bool = False,
) -> dict:
    """Apply a previously reviewed hygiene dry-run if fingerprints still match."""
    if operator_confirmed is not True:
        return {"status": "rejected", "read_only": True, "reason": "operator_confirmation_required"}
    normalized_run_id = _normalize_workstream_text(run_id, max_len=128)
    if not normalized_run_id:
        return {"status": "rejected", "read_only": True, "reason": "run_id_required"}
    if not isinstance(fingerprints, dict) or not isinstance(proposed_states, dict) or not proposed_states:
        return {"status": "rejected", "read_only": True, "reason": "dry_run_payload_required"}
    try:
        evaluated_at_dt = _parse_hygiene_evaluated_at(evaluated_at)
    except ValueError as exc:
        return {"status": "rejected", "read_only": True, "reason": str(exc)}

    rows = _hygiene_rows_by_id()
    stale_thread_ids: list[str] = []
    mismatched_state_ids: list[str] = []
    previous_states: dict[str, dict | None] = {}
    normalized_new_states: dict[str, dict] = {}
    for thread_id, proposed_state in proposed_states.items():
        row = rows.get(str(thread_id))
        if row is None or _workstream_hygiene_fingerprint(row) != fingerprints.get(thread_id):
            stale_thread_ids.append(str(thread_id))
            continue
        recomputed = compute_workstream_discovery_state(row, evaluated_at=evaluated_at_dt, run_id=normalized_run_id)
        if recomputed != proposed_state:
            mismatched_state_ids.append(str(thread_id))
            continue
        thread = load_thread(str(thread_id))
        previous = _normalize_workstream_discovery_state(
            thread.workstream.discovery_state if thread and thread.workstream else None
        )
        previous_states[str(thread_id)] = previous.model_dump() if previous else None
        normalized_new_states[str(thread_id)] = _normalize_workstream_discovery_state(proposed_state).model_dump()

    if stale_thread_ids or mismatched_state_ids:
        return {
            "status": "rejected",
            "read_only": True,
            "reason": "dry_run_payload_stale",
            "stale_thread_ids": stale_thread_ids,
            "mismatched_state_ids": mismatched_state_ids,
        }

    from app.storage import save_checkpoint

    rollback_task_id = f"workstream-hygiene-rollback:{normalized_run_id}"
    checkpoint = save_checkpoint(
        task_id=rollback_task_id,
        description=f"Workstream hygiene rollback payload: {normalized_run_id}",
        status="active",
        active="Rollback Workstream hygiene apply if needed",
        next_items=[],
        blockers=[],
        context={
            "run_id": normalized_run_id,
            "previous_states": previous_states,
            "new_states": normalized_new_states,
        },
        concept_refs=[],
        ttl_days=14,
    )
    if not checkpoint or checkpoint.get("error"):
        return {"status": "rejected", "read_only": True, "reason": "rollback_checkpoint_failed"}

    changed_thread_ids: list[str] = []
    failed_thread_ids: list[str] = []
    for thread_id, new_state in normalized_new_states.items():
        try:
            thread = load_thread(thread_id)
            if not thread or not thread.workstream:
                failed_thread_ids.append(thread_id)
                continue
            thread.workstream.discovery_state = _normalize_workstream_discovery_state(new_state)
            save_thread(thread)
            changed_thread_ids.append(thread_id)
        except Exception:
            logger.exception("workstream_hygiene_apply_failed thread_id=%s run_id=%s", thread_id, normalized_run_id)
            failed_thread_ids.append(thread_id)

    status = "partial_apply" if failed_thread_ids else "ok"
    _record_workstream_metric(
        "workstream_hygiene_apply_completed",
        {"changed_rows": len(changed_thread_ids), "failed_rows": len(failed_thread_ids), "status": status},
    )
    return {
        "status": status,
        "read_only": False,
        "run_id": normalized_run_id,
        "changed_thread_ids": changed_thread_ids,
        "failed_thread_ids": failed_thread_ids,
        "rollback_task_id": rollback_task_id,
    }


def rollback_workstream_hygiene(run_id: str, operator_confirmed: bool = False) -> dict:
    """Restore discovery states from a hygiene rollback checkpoint."""
    if operator_confirmed is not True:
        return {"status": "rejected", "read_only": True, "reason": "operator_confirmation_required"}
    normalized_run_id = _normalize_workstream_text(run_id, max_len=128)
    if not normalized_run_id:
        return {"status": "rejected", "read_only": True, "reason": "run_id_required"}
    from app.storage import load_checkpoint

    rollback_task_id = f"workstream-hygiene-rollback:{normalized_run_id}"
    checkpoint = load_checkpoint(task_id=rollback_task_id)
    if not checkpoint:
        return {"status": "not_found", "read_only": True, "reason": "rollback_checkpoint_not_found"}
    previous_states = (checkpoint.get("context") or {}).get("previous_states") or {}
    restored_thread_ids: list[str] = []
    failed_thread_ids: list[str] = []
    for thread_id, previous_state in previous_states.items():
        try:
            thread = load_thread(thread_id)
            if not thread or not thread.workstream:
                failed_thread_ids.append(thread_id)
                continue
            thread.workstream.discovery_state = _normalize_workstream_discovery_state(previous_state)
            save_thread(thread)
            restored_thread_ids.append(thread_id)
        except Exception:
            logger.exception("workstream_hygiene_rollback_failed thread_id=%s run_id=%s", thread_id, normalized_run_id)
            failed_thread_ids.append(thread_id)
    return {
        "status": "partial_rollback" if failed_thread_ids else "ok",
        "read_only": False,
        "run_id": normalized_run_id,
        "restored_thread_ids": restored_thread_ids,
        "failed_thread_ids": failed_thread_ids,
        "rollback_task_id": rollback_task_id,
    }


def promote_workstream_discovery_candidate(
    thread_id: str,
    promotion_reason: str,
    promoted_by: str = "operator",
    operator_confirmed: bool = False,
) -> dict:
    if operator_confirmed is not True:
        return {"status": "rejected", "read_only": True, "reason": "operator_confirmation_required"}
    reason = _normalize_workstream_text(promotion_reason, max_len=500)
    if not reason:
        return {"status": "rejected", "read_only": True, "reason": "promotion_reason_required"}
    thread = load_thread(str(thread_id or "").strip())
    if not thread:
        return {"status": "not_found", "read_only": True}
    if thread.workstream is None:
        return {"status": "rejected", "read_only": True, "reason": "not_workstream"}
    previous_tier = thread.workstream.discovery_state.tier if thread.workstream.discovery_state else None
    thread.workstream.discovery_state = WorkstreamDiscoveryState(
        tier="curated_candidate",
        reason_codes=["operator_promoted"],
        source="operator",
        last_evaluated_at=_utc_now_iso(),
        eligible_until=(_utc_now() + timedelta(days=WORKSTREAM_DISCOVERY_DEFAULT_TTL_DAYS)).isoformat(),
        previous_tier=previous_tier,
        promoted_by=_normalize_workstream_text(promoted_by, max_len=128) or "operator",
        promoted_at=_utc_now_iso(),
        promotion_reason=reason,
    )
    save_thread(thread)
    return {"status": "ok", "read_only": False, "thread": thread.model_dump()}


def demote_workstream_discovery_candidate(
    thread_id: str,
    reason: str,
    demoted_by: str = "operator",
    operator_confirmed: bool = False,
) -> dict:
    if operator_confirmed is not True:
        return {"status": "rejected", "read_only": True, "reason": "operator_confirmation_required"}
    demotion_reason = _normalize_workstream_text(reason, max_len=500)
    if not demotion_reason:
        return {"status": "rejected", "read_only": True, "reason": "demotion_reason_required"}
    thread = load_thread(str(thread_id or "").strip())
    if not thread:
        return {"status": "not_found", "read_only": True}
    if thread.workstream is None:
        return {"status": "rejected", "read_only": True, "reason": "not_workstream"}
    previous_tier = thread.workstream.discovery_state.tier if thread.workstream.discovery_state else None
    thread.workstream.discovery_state = WorkstreamDiscoveryState(
        tier="needs_hygiene_review",
        reason_codes=["operator_demoted"],
        source="operator",
        last_evaluated_at=_utc_now_iso(),
        previous_tier=previous_tier,
        promoted_by=_normalize_workstream_text(demoted_by, max_len=128) or "operator",
        promotion_reason=demotion_reason,
    )
    save_thread(thread)
    return {"status": "ok", "read_only": False, "thread": thread.model_dump()}


def _clamp_workstream_ref_limit(max_refs: object) -> int:
    """Clamp Workstream context references to the Phase 1 hard cap."""
    return _clamp_positive_int(max_refs, WORKSTREAM_REF_LIMIT_DEFAULT, WORKSTREAM_REF_LIMIT_MAX)


def _rank_workstream_links(links: list[ThreadConceptLink]) -> list[ThreadConceptLink]:
    """Rank links by role importance, human curation, then recency."""

    def _key(link: ThreadConceptLink):
        role_rank = _WORKSTREAM_ROLE_PRIORITY.get(link.role, 99)
        added_by_rank = 0 if link.added_by == "user" else 1
        try:
            added_ts = datetime.fromisoformat(str(link.added_at or "").replace("Z", "+00:00")).timestamp()
        except (TypeError, ValueError):
            added_ts = 0.0
        return (role_rank, added_by_rank, -added_ts)

    return sorted(links, key=_key)


def _single_thread_classification(thread_id: str, agent_id: str = "default") -> dict | None:
    for row in _fetch_workstream_rows(agent_id=agent_id):
        if row.get("id") == thread_id:
            return classify_thread_for_workstream(row)
    for row in _fetch_workstream_rows(agent_id=agent_id, include_non_workstreams=True):
        if row.get("id") == thread_id:
            return classify_thread_for_workstream(row)
    return None


def _reference_reason(link: ThreadConceptLink) -> tuple[str, float, str]:
    if link.added_by == "user":
        return "user_linked", 0.80, "accepted"
    if link.role == "initiator":
        return "thread_initiator", 0.70, "proposed"
    if link.role == "blocker":
        return "blocker_reference", 0.65, "proposed"
    if link.role == "conclusion":
        return "decision_reference", 0.65, "proposed"
    return "auto_thread_link", 0.45, "proposed"


def _binding_visibility_warning(classification: dict | None) -> dict | None:
    if not classification:
        return None
    tier = classification.get("effective_discovery_tier") or classification.get("discovery_tier")
    if tier not in WORKSTREAM_DISCOVERY_HIDDEN_TIERS:
        return None
    return {
        "tier": tier,
        "reason_codes": classification.get("discovery_reason_codes") or [],
        "message": "Exact-bound Workstream is hidden from ordinary discovery but remains available by binding.",
    }


def build_workstream_context_block(
    thread_id: str | None = None,
    operator_mode: bool = False,
    max_refs: int = WORKSTREAM_REF_LIMIT_DEFAULT,
    include_concept_summaries: bool = True,
) -> dict:
    """Build a bounded, typed Workstream context block for an explicit thread."""
    start = time.perf_counter()
    if thread_id is None or not str(thread_id).strip():
        candidates = [
            c
            for c in classify_workstream_threads(include_maintenance=False, limit=5)["threads"]
            if c["class"] == "workstream_candidate"
        ][:5]
        logger.info("workstream_context_no_binding candidate_count=%d", len(candidates))
        return {
            "status": "ok",
            "binding_status": "none",
            "binding_candidates": candidates,
            "context_block": None,
            "maintenance_filtered": False,
            "read_only": True,
        }

    normalized_thread_id = str(thread_id).strip()
    if len(normalized_thread_id) > 128:
        return {
            "status": "invalid",
            "binding_status": "invalid",
            "reason": "thread_id_too_long",
            "context_block": None,
            "read_only": True,
        }

    thread = load_thread(normalized_thread_id)
    if not thread:
        return {"status": "not_found", "binding_status": "not_found", "context_block": None, "read_only": True}

    classification = _single_thread_classification(normalized_thread_id, agent_id=thread.agent_id)
    if classification is None:
        return {"status": "not_found", "binding_status": "not_found", "context_block": None, "read_only": True}

    workstream = {
        "thread_id": thread.id,
        "title": thread.title,
        "status": thread.status,
        "class": classification["class"],
        "legacy_class": classification.get("legacy_class"),
        "effective_discovery_tier": classification.get("effective_discovery_tier"),
        "discovery_state": (
            thread.workstream.discovery_state.model_dump()
            if thread.workstream and thread.workstream.discovery_state
            else None
        ),
        "quality_flags": classification["quality_flags"],
        "metadata": _thread_metadata_dict(thread),
    }
    visibility_warning = _binding_visibility_warning(classification)

    if classification["class"] == "maintenance_cluster" and not operator_mode:
        logger.info("workstream_context_filtered thread_id=%s reason=maintenance_cluster", normalized_thread_id)
        return {
            "status": "filtered",
            "binding_status": "explicit",
            "workstream": workstream,
            "binding_visibility_warning": visibility_warning,
            "context_block": None,
            "maintenance_filtered": True,
            "read_only": True,
        }

    effective_limit = _clamp_workstream_ref_limit(max_refs)
    ranked_links = _rank_workstream_links(get_concepts_for_thread(normalized_thread_id))
    emitted_links = ranked_links[:effective_limit]
    concept_map = {}
    if include_concept_summaries and emitted_links:
        from app.storage.concepts import load_concepts_batch

        concept_map = load_concepts_batch([link.concept_id for link in emitted_links])

    refs = []
    for link in emitted_links:
        reason_code, confidence_hint, review_state = _reference_reason(link)
        ref = {
            "concept_id": link.concept_id,
            "role": link.role,
            "added_by": link.added_by,
            "added_at": link.added_at,
            "source_table": "thread_concept_links",
            "source_thread_id": link.thread_id,
            "reason_code": reason_code,
            "confidence_hint": confidence_hint,
            "review_state": review_state,
        }
        concept = concept_map.get(link.concept_id)
        if concept is not None:
            ref.update(
                {
                    "summary": concept.summary,
                    "knowledge_area": concept.knowledge_area,
                    "concept_type": concept.concept_type,
                    "confidence": concept.confidence,
                    "status": getattr(concept, "status", None),
                }
            )
        refs.append(ref)

    logger.info(
        "workstream_context_completed thread_id=%s class=%s refs=%d elapsed_ms=%.1f status=ok",
        normalized_thread_id,
        classification["class"],
        len(refs),
        (time.perf_counter() - start) * 1000,
    )
    return {
        "status": "ok",
        "binding_status": "explicit",
        "workstream": workstream,
        "binding_visibility_warning": visibility_warning,
        "context_block": {
            "generated_at": _utc_now_iso(),
            "max_refs": effective_limit,
            "refs": refs,
        },
        "maintenance_filtered": False,
        "source_trace": ["threads", "thread_concept_links", "concepts"],
        "read_only": True,
    }


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


def _thread_member_count(thread: NarrativeThread) -> int:
    return len(thread.concept_ids or [])


def _thread_session_overlap(concept_session_id: str | None, thread_session_counts: Counter) -> float:
    if not concept_session_id:
        return 0.0
    total = sum(thread_session_counts.values())
    if total <= 0:
        return 0.0
    return thread_session_counts.get(concept_session_id, 0) / total


def _projected_ka_purity(thread_ka_counts: Counter, incoming_ka: str) -> float:
    projected = Counter(thread_ka_counts)
    if incoming_ka:
        projected[incoming_ka] += 1
    total = sum(projected.values())
    if total <= 0:
        return 0.0
    return max(projected.values()) / total


def _embedding_vector_for_concept(concept_id: str):
    from app.storage.embedding import embedding_engine

    if getattr(embedding_engine, "_index_matrix", None) is None:
        return None
    pos = getattr(embedding_engine, "_id_to_pos", {}).get(concept_id)
    if pos is None:
        return None
    return embedding_engine._index_matrix[pos]


def _centroid_for_concept_ids(concept_ids: list[str]):
    vectors = []
    for concept_id in concept_ids:
        vec = _embedding_vector_for_concept(concept_id)
        if vec is not None:
            vectors.append(vec)
    if not vectors:
        return None
    centroid = np.mean(np.stack(vectors), axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm <= 0:
        return None
    return centroid / norm


def _cosine_similarity(vec_a, vec_b) -> float | None:
    if vec_a is None or vec_b is None:
        return None
    return float(vec_a @ vec_b)


def _top3_association_strength(concept_id: str, thread_concept_ids: list[str]) -> float:
    from app.storage import get_db_connection

    other_ids = [cid for cid in thread_concept_ids if cid != concept_id]
    if not other_ids:
        return 0.0
    placeholders = ",".join("?" for _ in other_ids)
    conn = get_db_connection()
    try:
        rows = conn.execute(
            f"""
            SELECT strength FROM associations
            WHERE (source = ? AND target IN ({placeholders}))
               OR (target = ? AND source IN ({placeholders}))
            ORDER BY strength DESC
            LIMIT 3
            """,
            (concept_id, *other_ids, concept_id, *other_ids),
        ).fetchall()
    finally:
        conn.close()
    strengths = [float(row[0] or 0.0) for row in rows]
    if not strengths:
        return 0.0
    return sum(strengths) / len(strengths)


def build_thread_guardrail_cache(active_threads: list[NarrativeThread]) -> dict:
    from app.storage import load_concepts_batch

    cache = {"threads": {}, "membership_counts": Counter()}
    concept_ids: list[str] = []
    seen_concept_ids: set[str] = set()
    for thread in active_threads:
        for concept_id in thread.concept_ids or []:
            if concept_id in seen_concept_ids:
                continue
            seen_concept_ids.add(concept_id)
            concept_ids.append(concept_id)
    concepts_by_id = load_concepts_batch(concept_ids) if concept_ids else {}

    for thread in active_threads:
        ka_counts: Counter = Counter()
        session_counts: Counter = Counter()
        for concept_id in thread.concept_ids or []:
            cache["membership_counts"][concept_id] += 1
            concept = concepts_by_id.get(concept_id)
            if not concept:
                continue
            knowledge_area = getattr(concept, "knowledge_area", "") or ""
            session_id = getattr(concept, "session_id", None)
            if knowledge_area:
                ka_counts[knowledge_area] += 1
            if session_id:
                session_counts[session_id] += 1

        cache["threads"][thread.id] = {
            "member_count": _thread_member_count(thread),
            "ka_counts": ka_counts,
            "session_counts": session_counts,
            "centroid": _centroid_for_concept_ids(thread.concept_ids or []),
            "concept_ids": list(thread.concept_ids or []),
        }
    return cache


def _legacy_auto_link_candidates(concept, active_threads: list[NarrativeThread]) -> list[dict]:
    from app.core.config import AUTO_LINK_TFIDF_THRESHOLD, AUTO_LINK_TITLE_SIMILARITY_THRESHOLD

    try:
        from app.retrieval import retrieval_engine

        has_retrieval = True
    except (ImportError, Exception):
        has_retrieval = False

    decisions = []
    concept_ka = getattr(concept, "knowledge_area", "") or ""
    concept_text = f"{getattr(concept, 'summary', '')} {concept_ka}"
    source_trace_id = getattr(concept, "source_trace_id", None)

    for thread in active_threads:
        if source_trace_id and source_trace_id in (thread.trace_ids or []):
            decisions.append({"thread_id": thread.id, "admit": True, "reason_code": "admit_trace_linkage"})
            continue

        ka_match_direct = concept_ka and _ka_overlaps(concept_ka, thread.knowledge_areas or [])
        ka_match = ka_match_direct

        if not ka_match and has_retrieval:
            try:
                title_sim = retrieval_engine.index.compute_similarity(
                    concept_text,
                    f"{thread.title} {thread.description}",
                )
                if title_sim >= AUTO_LINK_TITLE_SIMILARITY_THRESHOLD:
                    ka_match = True
            except Exception:
                pass

        if not ka_match:
            continue

        if ka_match_direct:
            decisions.append({"thread_id": thread.id, "admit": True, "reason_code": "admit_legacy_match"})
            continue

        if has_retrieval:
            try:
                similarity = retrieval_engine.index.compute_similarity(
                    concept_text,
                    f"{thread.title} {thread.description}",
                )
                if similarity >= AUTO_LINK_TFIDF_THRESHOLD:
                    decisions.append({"thread_id": thread.id, "admit": True, "reason_code": "admit_legacy_match"})
            except Exception:
                pass

    return decisions


def _queue_reorg_seed(concept, reason_code: str, notes: dict | None = None) -> None:
    try:
        from app.ops.metrics import metrics
        from app.ops.thread_reorg import queue_seed_candidate

        queue_seed_candidate(concept, reason=reason_code.replace("reject_", ""), notes=notes or {})
        metrics.record("thread_reorg_seed_candidate_queued", 1.0, {"reason": reason_code})
    except Exception as exc:
        logger.debug("THREAD-004: seed queue skipped: %s", exc)


def auto_link_candidates(
    concept,
    active_threads: list[NarrativeThread],
    guardrail_cache: dict | None = None,
) -> list[dict]:
    """Return structured auto-link decisions for a concept.

    Guardrails are feature-flagged. With the flag off, this preserves the
    historical permissive behavior but returns the modern decision payload.
    """
    from app.core.config import (
        THREAD_REORG_ASSOC_FLOOR,
        THREAD_REORG_GUARDRAILS_ENABLED,
        THREAD_REORG_KA_PURITY_FLOOR,
        THREAD_REORG_MAX_LINKS_PER_CONCEPT,
        THREAD_REORG_SEMANTIC_FLOOR,
        THREAD_REORG_THREAD_SOFT_CAP,
    )

    if not THREAD_REORG_GUARDRAILS_ENABLED:
        return _legacy_auto_link_candidates(concept, active_threads)

    if not active_threads:
        return []

    concept_id = getattr(concept, "id", "")
    concept_ka = getattr(concept, "knowledge_area", "") or ""
    concept_session_id = getattr(concept, "session_id", None)
    concept_summary = getattr(concept, "summary", "") or ""
    concept_trace_id = getattr(concept, "source_trace_id", None)

    guardrail_cache = guardrail_cache or build_thread_guardrail_cache(active_threads)
    existing_memberships = guardrail_cache.get("membership_counts", Counter()).get(concept_id, 0)
    if existing_memberships >= THREAD_REORG_MAX_LINKS_PER_CONCEPT:
        return [{"thread_id": None, "admit": False, "reason_code": "reject_membership_cap"}]

    concept_vec = _embedding_vector_for_concept(concept_id)
    decisions: list[dict] = []
    admitted = False
    semantic_unavailable = False

    for thread in active_threads:
        if concept_id in (thread.concept_ids or []):
            continue

        if concept_trace_id and concept_trace_id in (thread.trace_ids or []):
            decisions.append(
                {
                    "thread_id": thread.id,
                    "admit": True,
                    "reason_code": "admit_trace_linkage",
                    "assoc_raw": None,
                    "semantic_raw": None,
                    "temporal_raw": None,
                    "projected_ka_purity": None,
                }
            )
            admitted = True
            continue

        if not _ka_overlaps(concept_ka, thread.knowledge_areas or []):
            continue

        thread_cache = guardrail_cache["threads"].get(thread.id, {})
        if thread_cache.get("member_count", 0) >= THREAD_REORG_THREAD_SOFT_CAP:
            decisions.append({"thread_id": thread.id, "admit": False, "reason_code": "reject_thread_soft_cap"})
            continue

        assoc_raw = _top3_association_strength(concept_id, thread_cache.get("concept_ids", []))
        if assoc_raw < THREAD_REORG_ASSOC_FLOOR:
            decisions.append(
                {
                    "thread_id": thread.id,
                    "admit": False,
                    "reason_code": "reject_assoc_floor",
                    "assoc_raw": round(assoc_raw, 4),
                }
            )
            continue

        semantic_raw = _cosine_similarity(concept_vec, thread_cache.get("centroid"))
        if semantic_raw is None:
            semantic_unavailable = True
            decisions.append({"thread_id": thread.id, "admit": False, "reason_code": "reject_semantic_unavailable"})
            continue
        if semantic_raw < THREAD_REORG_SEMANTIC_FLOOR:
            decisions.append(
                {
                    "thread_id": thread.id,
                    "admit": False,
                    "reason_code": "reject_semantic_floor",
                    "semantic_raw": round(semantic_raw, 4),
                }
            )
            continue

        projected_ka_purity = _projected_ka_purity(thread_cache.get("ka_counts", Counter()), concept_ka)
        if projected_ka_purity < THREAD_REORG_KA_PURITY_FLOOR:
            decisions.append(
                {
                    "thread_id": thread.id,
                    "admit": False,
                    "reason_code": "reject_ka_purity",
                    "projected_ka_purity": round(projected_ka_purity, 4),
                }
            )
            continue

        temporal_raw = _thread_session_overlap(concept_session_id, thread_cache.get("session_counts", Counter()))
        decisions.append(
            {
                "thread_id": thread.id,
                "admit": True,
                "reason_code": "admit_guardrailed_match",
                "assoc_raw": round(assoc_raw, 4),
                "semantic_raw": round(semantic_raw, 4),
                "temporal_raw": round(temporal_raw, 4),
                "projected_ka_purity": round(projected_ka_purity, 4),
            }
        )
        admitted = True

    if admitted:
        return decisions

    if semantic_unavailable:
        try:
            from app.ops.metrics import metrics

            metrics.record("thread_reorg_embedding_unavailable", 1.0)
        except Exception:
            pass
        _queue_reorg_seed(
            concept,
            "reject_semantic_unavailable",
            {"summary": concept_summary[:200]},
        )
        decisions.append({"thread_id": None, "admit": False, "reason_code": "reject_semantic_unavailable"})
    else:
        _queue_reorg_seed(
            concept,
            "reject_no_guardrailed_match",
            {"summary": concept_summary[:200]},
        )
        decisions.append({"thread_id": None, "admit": False, "reason_code": "reject_no_guardrailed_match"})

    return decisions


# =============================================================================
# Intent-Based Episode Retrieval (§5.4)
# =============================================================================


def load_traces_since(cutoff_iso: str, limit: int = 500) -> list:
    """Load traces created after cutoff. Returns TraceRecord list."""
    from app.core.models import TraceRecord
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
    from app.core.config import TRACE_RETRIEVAL_SCAN_LIMIT

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
    from app.core.config import THREAD_MEMBERSHIP_SALIENCE_DIVISOR

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
    from app.core.config import STALENESS_TIERS, THREAD_MAX_ORIENTATION_DISPLAY

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
