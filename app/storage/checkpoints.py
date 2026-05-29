"""Storage sub-module: checkpoints.

Checkpoint CRUD, resume snapshots, checkpoint analytics.
Extracted from storage/__init__.py during Item 2b decomposition.
"""
import json
import logging
import re
from datetime import timedelta

import app.storage.connection as _conn
from app.core.datetime_utils import _utc_now, _utc_now_iso
from app.storage.connection import read_snapshot_db

logger = logging.getLogger(__name__)

MAX_CHECKPOINTS = 50
DEFAULT_TTL_DAYS = 7
STALE_CHECKPOINT_HOURS = 48  # CKPT-001: Archive checkpoints with no update in this many hours
COMPLETED_TTL_DAYS = 1
ORIGIN_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def _normalize_origin_id(origin_id: str | None) -> str | None:
    if origin_id is None:
        return None
    if not isinstance(origin_id, str):
        raise ValueError("origin_id must be a string")
    normalized = origin_id.strip()
    if not normalized:
        return None
    if not ORIGIN_ID_PATTERN.fullmatch(normalized):
        raise ValueError("origin_id must match ^[A-Za-z0-9._:-]{1,128}$")
    return normalized

def save_checkpoint(
    task_id: str,
    description: str,
    status: str = "active",
    done: list = None,
    active: str = "",
    next_items: list = None,
    blockers: list = None,
    context: dict = None,
    concept_refs: list = None,
    session_id: str = None,
    ttl_days: int = None,
    origin_id: str = None,
    op_id: int = None,
    payload_hash: str = None,
) -> dict:
    """Upsert checkpoint by task_id. done[] is union-merged (append-only)."""
    origin_id = _normalize_origin_id(origin_id)
    if op_id is not None and origin_id is None:
        raise ValueError("op_id requires origin_id")

    now = _utc_now_iso()
    ttl = ttl_days or DEFAULT_TTL_DAYS
    expires_at = (_utc_now() + timedelta(days=ttl)).isoformat()

    with _conn._db_immediate() as conn:
        sync_state = None
        if origin_id is not None and op_id is not None:
            sync_state = conn.execute(
                """
                SELECT last_op_id, payload_hash, checkpoint_updated_at, updated_at
                FROM checkpoint_sync_state
                WHERE task_id = ? AND origin_id = ?
                """,
                (task_id, origin_id),
            ).fetchone()
            if sync_state and op_id <= sync_state["last_op_id"]:
                row = conn.execute("SELECT * FROM checkpoints WHERE task_id = ?", (task_id,)).fetchone()
                current = {
                    "task_id": task_id,
                    "status": row["status"] if row else "not_found",
                    "description": row["description"] if row else description,
                    "done": json.loads(row["done"]) if row and row["done"] else [],
                    "active": row["active"] if row else "",
                    "next": json.loads(row["next"]) if row and row["next"] else [],
                    "blockers": json.loads(row["blockers"]) if row and row["blockers"] else [],
                    "save_count": row["save_count"] if row else 0,
                    "created_at": row["created_at"] if row else None,
                    "updated_at": row["updated_at"] if row else None,
                    "expires_at": row["expires_at"] if row else None,
                }
                reason = "duplicate_op" if op_id == sync_state["last_op_id"] else "stale_op"
                current["sync"] = {
                    "origin_id": origin_id,
                    "op_id": op_id,
                    "payload_hash": payload_hash,
                    "applied": False,
                    "reason": reason,
                    "last_applied_op_id": sync_state["last_op_id"],
                    "checkpoint_updated_at": sync_state["checkpoint_updated_at"],
                    "server_updated_at": sync_state["updated_at"],
                }
                logger.info(
                    "Checkpoint replay ignored: task_id=%s origin_id=%s op_id=%s reason=%s last_applied=%s",
                    task_id,
                    origin_id,
                    op_id,
                    reason,
                    sync_state["last_op_id"],
                )
                return current

        existing = conn.execute(
            "SELECT done, save_count, created_at, status FROM checkpoints WHERE task_id = ?", (task_id,)
        ).fetchone()

        # CKPT-001: Lifecycle validation — reject saves to terminal states
        # Allow reopen if caller explicitly passes an active status
        if existing and existing["status"] in ("complete", "archived") and status not in (
            "complete",
            "archived",
            "active",
            "planning",
        ):
            logger.warning(
                f"CKPT-001: Rejected save to {task_id} — status '{existing['status']}' is terminal. "
                f"Pass explicit status='active' to reopen."
            )
            return load_checkpoint(task_id=task_id) or {
                "task_id": task_id,
                "status": existing["status"],
                "error": "terminal_state",
            }

        if existing:
            # Union-merge done[] (never remove items)
            old_done = json.loads(existing["done"]) if existing["done"] else []
            new_done = list(set(old_done + (done or [])))
            save_count = (existing["save_count"] or 0) + 1
            created_at = existing["created_at"]
        else:
            new_done = done or []
            save_count = 1
            created_at = now

        conn.execute(
            """
            INSERT OR REPLACE INTO checkpoints
            (task_id, session_id, origin_id, status, description, done, active, next,
             blockers, context, concept_refs, created_at, updated_at, expires_at, save_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                task_id,
                session_id,
                origin_id,
                status,
                description,
                json.dumps(new_done),
                active or "",
                json.dumps(next_items or []),
                json.dumps(blockers or []),
                json.dumps(context or {}),
                json.dumps(concept_refs or []),
                created_at,
                now,
                expires_at,
                save_count,
            ),
        )

        if origin_id is not None and op_id is not None:
            created_sync_at = sync_state["updated_at"] if sync_state else now
            conn.execute(
                """
                INSERT INTO checkpoint_sync_state
                (task_id, origin_id, last_op_id, payload_hash, checkpoint_updated_at, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id, origin_id) DO UPDATE SET
                    last_op_id = excluded.last_op_id,
                    payload_hash = excluded.payload_hash,
                    checkpoint_updated_at = excluded.checkpoint_updated_at,
                    updated_at = excluded.updated_at
                """,
                (
                    task_id,
                    origin_id,
                    op_id,
                    payload_hash,
                    now,
                    created_sync_at,
                    now,
                ),
            )

        # FIFO eviction if over max
        count = conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
        if count > MAX_CHECKPOINTS:
            excess = count - MAX_CHECKPOINTS
            evicted_rows = conn.execute(
                """
                SELECT task_id FROM checkpoints
                WHERE status NOT IN ('active', 'blocked')
                ORDER BY updated_at ASC LIMIT ?
                """,
                (excess,),
            ).fetchall()
            evicted_task_ids = [row["task_id"] for row in evicted_rows]
            conn.execute(
                """
                DELETE FROM checkpoints WHERE task_id IN (
                    SELECT task_id FROM checkpoints
                    WHERE status NOT IN ('active', 'blocked')
                    ORDER BY updated_at ASC LIMIT ?
                )
            """,
                (excess,),
            )
            if evicted_task_ids:
                placeholders = ",".join("?" for _ in evicted_task_ids)
                conn.execute(
                    f"DELETE FROM checkpoint_sync_state WHERE task_id IN ({placeholders})",
                    evicted_task_ids,
                )

    logger.info(f"Checkpoint saved: {task_id} status={status} save_count={save_count}")
    result = {
        "task_id": task_id,
        "origin_id": origin_id,
        "status": status,
        "description": description,
        "done": new_done,
        "active": active or "",
        "next": next_items or [],
        "blockers": blockers or [],
        "save_count": save_count,
        "created_at": created_at,
        "updated_at": now,
        "expires_at": expires_at,
    }
    if origin_id is not None and op_id is not None:
        result["sync"] = {
            "origin_id": origin_id,
            "op_id": op_id,
            "payload_hash": payload_hash,
            "applied": True,
            "reason": "applied",
            "last_applied_op_id": op_id,
            "checkpoint_updated_at": now,
            "server_updated_at": now,
        }
    return result


def load_checkpoint(
    task_id: str = None,
    max_age_hours: int = 24,
    session_id: str = None,
    origin_id: str = None,
) -> dict | None:
    """Load checkpoint by authoritative task/origin, then candidate session/global recency."""
    origin_id = _normalize_origin_id(origin_id)
    max_age_hours = max_age_hours or 24  # Guard against None from MCP wrapper
    cutoff = (_utc_now() - timedelta(hours=max_age_hours)).isoformat()
    selection_source = None
    selection_authority = None

    with read_snapshot_db("load_checkpoint") as conn:
        if task_id:
            row = conn.execute(
                "SELECT * FROM checkpoints WHERE task_id = ? AND expires_at > ?", (task_id, _utc_now_iso())
            ).fetchone()
            selection_source = "task_id"
            selection_authority = "authoritative"
        elif origin_id:
            row = conn.execute(
                """
                SELECT * FROM checkpoints
                WHERE status NOT IN ('complete', 'archived') AND expires_at > ? AND updated_at > ?
                AND origin_id = ?
                ORDER BY updated_at DESC LIMIT 1
            """,
                (_utc_now_iso(), cutoff, origin_id),
            ).fetchone()
            selection_source = "origin_id"
            selection_authority = "authoritative"
        elif session_id:
            # CONTEXT-001 Fix 9: Session-scoped checkpoint — find checkpoint created by this session
            # SESSION-004 Fix 1: Added AND session_id = ? (was returning any recent checkpoint)
            row = conn.execute(
                """
                SELECT * FROM checkpoints
                WHERE status NOT IN ('complete', 'archived') AND expires_at > ? AND updated_at > ?
                AND session_id = ?
                ORDER BY updated_at DESC LIMIT 1
            """,
                (_utc_now_iso(), cutoff, session_id),
            ).fetchone()
            selection_source = "session_id"
            selection_authority = "candidate"
        else:
            row = conn.execute(
                """
                SELECT * FROM checkpoints
                WHERE status NOT IN ('complete', 'archived') AND expires_at > ? AND updated_at > ?
                ORDER BY updated_at DESC LIMIT 1
            """,
                (_utc_now_iso(), cutoff),
            ).fetchone()
            selection_source = "global_recent"
            selection_authority = "candidate"

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


def list_checkpoints() -> list:
    """List all non-expired checkpoints, newest first.

    Returns full checkpoint data including next[], blockers[], active
    for orientation enrichment and anticipation (S6.3).
    """
    now = _utc_now_iso()
    with read_snapshot_db("list_checkpoints") as conn:
        rows = conn.execute(
            """
            SELECT task_id, status, description, active, next, blockers,
                   done, updated_at, save_count, origin_id
            FROM checkpoints WHERE expires_at > ?
            ORDER BY updated_at DESC
        """,
            (now,),
        ).fetchall()
    results = []
    for r in rows:
        results.append(
            {
                "task_id": r["task_id"],
                "origin_id": r["origin_id"] if "origin_id" in r.keys() else None,
                "status": r["status"],
                "description": r["description"],
                "active": r["active"] or "",
                "next": json.loads(r["next"]) if r["next"] else [],
                "blockers": json.loads(r["blockers"]) if r["blockers"] else [],
                "done": json.loads(r["done"]) if r["done"] else [],
                "updated_at": r["updated_at"],
                "save_count": r["save_count"],
            }
        )
    return results


def complete_checkpoint(task_id: str) -> dict | None:
    """Mark checkpoint complete, set short TTL."""
    now = _utc_now_iso()
    short_ttl = (_utc_now() + timedelta(days=COMPLETED_TTL_DAYS)).isoformat()

    with _conn._db() as conn:
        row = conn.execute("SELECT active, done FROM checkpoints WHERE task_id = ?", (task_id,)).fetchone()
        if not row:
            return None

        # Move active to done
        done = json.loads(row["done"]) if row["done"] else []
        if row["active"] and row["active"] not in done:
            done.append(row["active"])

        conn.execute(
            """
            UPDATE checkpoints SET status = 'complete', active = '',
                next = '[]', done = ?, updated_at = ?, expires_at = ?
            WHERE task_id = ?
        """,
            (json.dumps(done), now, short_ttl, task_id),
        )

    logger.info(f"Checkpoint completed: {task_id}")
    return load_checkpoint(task_id)


def touch_checkpoint(task_id: str, ttl_days: int = DEFAULT_TTL_DAYS) -> dict | None:
    """Extend TTL without changing content."""
    now = _utc_now_iso()
    new_expires = (_utc_now() + timedelta(days=ttl_days)).isoformat()

    with _conn._db() as conn:
        cursor = conn.execute(
            """
            UPDATE checkpoints SET expires_at = ?, updated_at = ?
            WHERE task_id = ?
        """,
            (new_expires, now, task_id),
        )
        if cursor.rowcount == 0:
            return None

    return load_checkpoint(task_id)


def cleanup_expired_checkpoints() -> int:
    """Delete expired checkpoints regardless of status."""
    now = _utc_now_iso()
    with _conn._db() as conn:
        # Log what we're about to delete for auditability
        zombies = conn.execute("SELECT task_id, status FROM checkpoints WHERE expires_at < ?", (now,)).fetchall()
        for z in zombies:
            logger.info(f"Cleaning expired checkpoint: {z[0]} (status={z[1]})")

        zombie_task_ids = [z[0] for z in zombies]
        cursor = conn.execute("DELETE FROM checkpoints WHERE expires_at < ?", (now,))
        if zombie_task_ids:
            placeholders = ",".join("?" for _ in zombie_task_ids)
            conn.execute(
                f"DELETE FROM checkpoint_sync_state WHERE task_id IN ({placeholders})",
                zombie_task_ids,
            )
    deleted = cursor.rowcount
    if deleted:
        logger.info(f"Cleaned up {deleted} expired checkpoint(s)")
    return deleted


def archive_stale_checkpoints(max_age_hours: int = STALE_CHECKPOINT_HOURS, exclude_session_id: str = None) -> int:
    """Archive checkpoints that haven't been updated in max_age_hours.

    CKPT-001: Stale checkpoints (no update in 48h) get archived rather than
    completed, because a stale checkpoint may represent paused work — not
    finished work. Archived checkpoints are excluded from load_checkpoint()
    and working_context but remain in DB for audit.

    NOTE: Auto-COMPLETE is a separate mechanism in staleness.py:587-611 with
    stricter guards (save_count>=2, non-empty done, empty next+active, >1h old).
    This function only ARCHIVES (soft-delete for stale items).
    """
    if max_age_hours < 1:
        max_age_hours = STALE_CHECKPOINT_HOURS  # Clamp invalid input

    cutoff = (_utc_now() - timedelta(hours=max_age_hours)).isoformat()

    with _conn._db() as conn:
        query = """
            UPDATE checkpoints SET status = 'archived', updated_at = ?
            WHERE status IN ('active', 'paused', 'planning')
            AND updated_at < ?
        """
        params = [_utc_now_iso(), cutoff]

        if exclude_session_id:
            query += " AND (session_id IS NULL OR session_id != ?)"
            params.append(exclude_session_id)

        cursor = conn.execute(query, params)

    archived = cursor.rowcount
    if archived:
        logger.info(f"CKPT-001: Archived {archived} stale checkpoint(s) (cutoff={max_age_hours}h)")
    return archived

MEDIUM_TRUNCATE_CHARS = 100
MEDIUM_MAX_LIST_ITEMS = 10

def get_checkpoint_effectiveness() -> dict:
    """CKPT-007: Legacy wrapper — calls get_checkpoint_dashboard() for backward compat."""
    dashboard = get_checkpoint_dashboard()
    return dashboard.get("checkpoint_lifecycle", {})


def get_checkpoint_dashboard() -> dict:
    """MEASURE-020: Comprehensive checkpoint & coverage measurement dashboard.

    Returns 4 metric categories:
    1. checkpoint_lifecycle — status distribution, stale/completion rates, nudge compliance
    2. compaction_recovery — event count, avg recovery_quality, quality distribution
    3. coverage_distribution — score histogram, threshold analysis (BENCH-015)
    4. session_health — drop rate, learning event distribution

    All queries are cold-path (maintenance/API). Not called during conversation_turn.
    """
    import json as _dj

    with read_snapshot_db("get_checkpoint_dashboard") as conn:
        # --- Category 1: Checkpoint lifecycle ---
        total = conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
        status_dist = {}
        for row in conn.execute("SELECT status, COUNT(*) FROM checkpoints GROUP BY status").fetchall():
            status_dist[row[0]] = row[1]
        archived = status_dist.get("archived", 0)
        complete = status_dist.get("complete", 0)

        nudge_count = conn.execute(
            "SELECT COUNT(*) FROM governance_events WHERE event_type='checkpoint_nudge_fired'"
        ).fetchone()[0]
        save_count = conn.execute(
            "SELECT COUNT(*) FROM governance_events WHERE event_type='checkpoint_save'"
        ).fetchone()[0]

        avg_saves = conn.execute(
            "SELECT ROUND(AVG(save_count), 1) FROM checkpoints WHERE save_count > 0"
        ).fetchone()[0] or 0

        checkpoint_lifecycle = {
            "total_checkpoints": total,
            "status_distribution": status_dist,
            "stale_rate": round(archived / total * 100, 1) if total > 0 else 0,
            "completion_rate": round(complete / total * 100, 1) if total > 0 else 0,
            "nudge_events": nudge_count,
            "save_events": save_count,
            "nudge_compliance": round(save_count / nudge_count * 100, 1) if nudge_count > 0 else None,
            "avg_saves_per_checkpoint": avg_saves,
        }

        # --- Category 2: Compaction recovery ---
        comp_events = conn.execute(
            "SELECT details FROM governance_events WHERE event_type='compaction_reinjection'"
        ).fetchall()
        comp_count = len(comp_events)
        comp_qualities = []
        for row in comp_events:
            try:
                d = _dj.loads(row[0])
                q = d.get("recovery_quality")
                if q is not None:
                    comp_qualities.append(q)
            except Exception:
                pass

        compaction_recovery = {
            "total_events": comp_count,
            "avg_recovery_quality": round(sum(comp_qualities) / len(comp_qualities), 3) if comp_qualities else None,
            "quality_distribution": {
                "high_0.8_plus": len([q for q in comp_qualities if q >= 0.8]),
                "medium_0.5_0.8": len([q for q in comp_qualities if 0.5 <= q < 0.8]),
                "low_below_0.5": len([q for q in comp_qualities if q < 0.5]),
            },
            "has_resume_rate": round(
                len([r for r in comp_events if '"has_resume": true' in (r[0] or "")]) / comp_count * 100, 1
            ) if comp_count > 0 else None,
        }

        # --- Category 3: Coverage distribution (BENCH-015) ---
        cov_rows = conn.execute(
            "SELECT details FROM governance_events WHERE event_type='coverage_score_recorded' "
            "ORDER BY created_at DESC LIMIT 1000"
        ).fetchall()
        cov_scores = []
        above_threshold_counts = []
        for row in cov_rows:
            try:
                d = _dj.loads(row[0])
                cs = d.get("coverage_score")
                if cs is not None:
                    cov_scores.append(cs)
                at = d.get("above_threshold")
                if at is not None:
                    above_threshold_counts.append(at)
            except Exception:
                pass

        cov_total = len(cov_scores)
        coverage_distribution = {
            "total_recorded": cov_total,
            "histogram": {
                "0.00-0.15": len([s for s in cov_scores if s < 0.15]),
                "0.15-0.30": len([s for s in cov_scores if 0.15 <= s < 0.30]),
                "0.30-0.35": len([s for s in cov_scores if 0.30 <= s < 0.35]),
                "0.35-0.45": len([s for s in cov_scores if 0.35 <= s < 0.45]),
                "0.45-0.60": len([s for s in cov_scores if 0.45 <= s < 0.60]),
                "0.60+": len([s for s in cov_scores if s >= 0.60]),
            } if cov_total > 0 else {},
            "mean_score": round(sum(cov_scores) / cov_total, 4) if cov_total > 0 else None,
            "median_score": round(sorted(cov_scores)[cov_total // 2], 4) if cov_total > 0 else None,
            "threshold_analysis": {
                "current_threshold": 0.35,
                "pct_above_threshold": round(
                    len([s for s in cov_scores if s >= 0.35]) / cov_total * 100, 1
                ) if cov_total > 0 else None,
                "mean_above_threshold_count": round(
                    sum(above_threshold_counts) / len(above_threshold_counts), 1
                ) if above_threshold_counts else None,
            },
        }

        # --- Category 4: Session health ---
        total_sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        short_sessions = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE learning_event_count <= 2"
        ).fetchone()[0]

        session_health = {
            "total_sessions": total_sessions,
            "drop_rate": round(short_sessions / total_sessions * 100, 1) if total_sessions > 0 else 0,
            "learning_distribution": {},
        }
        for row in conn.execute("""
            SELECT
                CASE
                    WHEN learning_event_count <= 1 THEN '0-1'
                    WHEN learning_event_count <= 5 THEN '2-5'
                    WHEN learning_event_count <= 20 THEN '6-20'
                    ELSE '20+'
                END as bucket,
                COUNT(*)
            FROM sessions GROUP BY bucket ORDER BY bucket
        """).fetchall():
            session_health["learning_distribution"][row[0]] = row[1]

    return {
        "checkpoint_lifecycle": checkpoint_lifecycle,
        "compaction_recovery": compaction_recovery,
        "coverage_distribution": coverage_distribution,
        "session_health": session_health,
        "generated_at": _utc_now_iso(),
    }

def compress_checkpoint(checkpoint: dict) -> dict:
    """CKPT-002: Compress checkpoint content using TTL tier classification.

    DURABLE fields: kept verbatim (task_id, description, done, status)
    MEDIUM fields: truncated to first 100 chars each item (active, next, blockers, concept_refs)
    PERISHABLE fields: stripped entirely (context)

    Returns a new dict with compressed content. Does not mutate input.
    """
    from app.core.models import CHECKPOINT_FIELD_TTL, CheckpointTTLTier

    compressed = {}
    for field, value in checkpoint.items():
        tier = CHECKPOINT_FIELD_TTL.get(field)

        if tier == CheckpointTTLTier.DURABLE:
            compressed[field] = value  # Keep verbatim
        elif tier == CheckpointTTLTier.MEDIUM:
            # Truncate: strings to 100 chars, lists to first 10 items with truncated strings
            if isinstance(value, str):
                compressed[field] = value[:MEDIUM_TRUNCATE_CHARS]
            elif isinstance(value, list):
                compressed[field] = [
                    (item[:MEDIUM_TRUNCATE_CHARS] if isinstance(item, str) else item)
                    for item in value[:MEDIUM_MAX_LIST_ITEMS]
                ]
            else:
                compressed[field] = value
        elif tier == CheckpointTTLTier.PERISHABLE:
            # Strip entirely — replace with empty equivalent
            if isinstance(value, dict):
                compressed[field] = {}
            elif isinstance(value, list):
                compressed[field] = []
            elif isinstance(value, str):
                compressed[field] = ""
            else:
                compressed[field] = None
        else:
            # Unknown field — keep as-is (forward compatibility)
            compressed[field] = value

    return compressed

RESUME_SNAPSHOT_TTL_DAYS = 7

def save_resume_snapshot(
    session_id: str,
    active_task: str | None = None,
    task_domain: str | None = None,
    pinned_concepts: list | None = None,
    last_exchange_gist: str | None = None,
    turn_count: int = 0,
    learning_events: int = 0,
    tools_used: list | None = None,
    checkpoint_summary: dict | None = None,  # CONTEXT-001: Checkpoint state for working_context
    topic_keywords: str | None = None,  # SESSION-012: cross-session topic signal
) -> dict:
    """Upsert rolling snapshot for a session. Replaces prior snapshot entirely."""
    now = _utc_now_iso()
    expires_at = (_utc_now() + timedelta(days=RESUME_SNAPSHOT_TTL_DAYS)).isoformat()

    with _conn._db() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO resume_snapshots
            (session_id, captured_at, active_task, task_domain, pinned_concepts,
             last_exchange_gist, turn_count, learning_events, tools_used,
             checkpoint_summary, expires_at, topic_keywords)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                session_id,
                now,
                (active_task or "")[:80],  # v1.1: enforce 80 char cap
                task_domain,
                json.dumps(pinned_concepts or []),
                (last_exchange_gist or "")[:120],  # v1.1: enforce 120 char cap
                turn_count,
                learning_events,
                json.dumps((tools_used or [])[:5]),  # v1.1: cap at 5 tools
                json.dumps(checkpoint_summary or {}),  # CONTEXT-001
                expires_at,
                (topic_keywords or "")[:200],  # SESSION-012: cap at 200 chars
            ),
        )

    logger.debug(f"Resume snapshot saved: session={session_id} task={active_task}")
    return {
        "session_id": session_id,
        "captured_at": now,
        "active_task": active_task,
        "task_domain": task_domain,
        "pinned_concepts": pinned_concepts or [],
        "last_exchange_gist": last_exchange_gist,
        "turn_count": turn_count,
        "learning_events": learning_events,
        "tools_used": tools_used or [],
        "checkpoint_summary": checkpoint_summary or {},  # CONTEXT-001
        "expires_at": expires_at,
        "topic_keywords": topic_keywords or "",  # SESSION-012
    }


def load_resume_snapshot(prior_session_id: str | None = None) -> dict | None:
    """Load resume snapshot for injection.

    If prior_session_id given, load that session's snapshot.
    Otherwise, load the most recent non-expired snapshot.
    """
    now = _utc_now_iso()

    with read_snapshot_db("load_resume_snapshot") as conn:
        if prior_session_id:
            row = conn.execute(
                "SELECT * FROM resume_snapshots WHERE session_id = ? AND expires_at > ?", (prior_session_id, now)
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT * FROM resume_snapshots
                WHERE expires_at > ?
                ORDER BY captured_at DESC LIMIT 1
            """,
                (now,),
            ).fetchone()

    if not row:
        return None

    # CONTEXT-001: Defensive — checkpoint_summary column may not exist pre-migration
    _cs_raw = None
    try:
        _cs_raw = row["checkpoint_summary"]
    except (KeyError, IndexError):
        pass

    return {
        "session_id": row["session_id"],
        "captured_at": row["captured_at"],
        "active_task": row["active_task"],
        "task_domain": row["task_domain"],
        "pinned_concepts": json.loads(row["pinned_concepts"]) if row["pinned_concepts"] else [],
        "last_exchange_gist": row["last_exchange_gist"],
        "turn_count": row["turn_count"],
        "learning_events": row["learning_events"],
        "tools_used": json.loads(row["tools_used"]) if row["tools_used"] else [],
        "checkpoint_summary": json.loads(_cs_raw) if _cs_raw else {},  # CONTEXT-001
        "expires_at": row["expires_at"],
    }


def cleanup_expired_snapshots() -> int:
    """Delete expired resume snapshots. Called alongside checkpoint cleanup."""
    now = _utc_now_iso()
    with _conn._db() as conn:
        cursor = conn.execute("DELETE FROM resume_snapshots WHERE expires_at < ?", (now,))
    deleted = cursor.rowcount
    if deleted:
        logger.info(f"Cleaned up {deleted} expired resume snapshot(s)")
    return deleted
