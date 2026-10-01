"""Storage sub-module: sessions.

Session CRUD operations.
Extracted from storage/__init__.py during Item 2b decomposition.
"""
import json
import logging
import sqlite3

import app.storage.connection as _conn
from app.core.datetime_utils import _utc_now_iso
from app.storage.connection import read_snapshot_db
from app.storage.utils import validate_agent_id

logger = logging.getLogger(__name__)

_MANAGED_MUTABLE_FIELDS = frozenset(
    {
        "concepts_created",
        "concepts_evolved",
        "data",
        "last_learning_at",
        "last_previous_response",
        "learning_event_count",
    }
)
_MANAGED_PROTECTED_FIELDS = frozenset(
    {
        "binding_generation",
        "binding_hash",
        "last_activity_at",
        "lifecycle_phase",
    }
)
_MANAGED_DENIAL_REASONS = frozenset(
    {
        "field_denied",
        "generation_mismatch",
        "permit_missing",
    }
)


def _deny_managed_write(reason: str) -> None:
    """Raise a typed denial and emit only bounded-cardinality telemetry."""

    from app.storage.session_bindings import ManagedSessionWriteDenied

    bounded_reason = reason if reason in _MANAGED_DENIAL_REASONS else "field_denied"
    try:
        from app.core.metrics_facade import metrics

        metrics.record(
            "managed_session_write_denied",
            1.0,
            labels={"reason": bounded_reason},
        )
    except Exception:
        logger.debug("managed session write denial metric failed", exc_info=True)
    raise ManagedSessionWriteDenied(f"managed session write denied: {bounded_reason}")


def _authorize_managed_helper_update(
    *,
    binding_hash: str,
    binding_generation: int,
    fields: set[str],
) -> None:
    """Require exact episode authority for allowlisted non-lifecycle fields."""

    from app.storage.session_bindings import current_managed_session_write_permit

    if not fields <= _MANAGED_MUTABLE_FIELDS:
        _deny_managed_write("field_denied")
    permit = current_managed_session_write_permit()
    if permit is None:
        _deny_managed_write("permit_missing")
    if (
        permit.binding_hash != binding_hash
        or permit.binding_generation != binding_generation
    ):
        _deny_managed_write("generation_mismatch")


def _load_session_authority(
    conn: sqlite3.Connection,
    session_id: str,
) -> tuple[object | None, bool]:
    """Load managed authority, tolerating pre-migration legacy-only schemas."""

    try:
        return (
            conn.execute(
                "SELECT binding_hash, binding_generation FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone(),
            True,
        )
    except sqlite3.OperationalError as exc:
        if "no such column: binding_" not in str(exc).lower():
            raise
        return (
            conn.execute(
                "SELECT NULL, NULL FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone(),
            False,
        )


def save_session(
    session_id: str,
    started_at: str,
    status: str = "active",
    context_hint: str = "",
    learning_event_count: int = 0,
    agent_id: str = "default",
    model_id: str = "unknown",
    platform_hint: str = "unknown",
    surface_id: str = "unknown",
    origin_id: str | None = None,
) -> None:
    """Insert a new session row."""
    validated_aid = validate_agent_id(agent_id)
    with _conn._db() as conn:
        conn.execute(
            """INSERT INTO sessions (id, started_at, status, context_hint,
               learning_event_count, agent_id, data, model_id, platform_hint, surface_id, origin_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                started_at,
                status,
                context_hint or "",
                learning_event_count,
                validated_aid,
                json.dumps({"session_id": session_id}),
                model_id or "unknown",
                platform_hint or "unknown",
                surface_id or "unknown",
                origin_id,
            ),
        )
    logger.info(f"Session saved: {session_id} status={status} agent_id={validated_aid}")


def update_session(session_id: str, **kwargs) -> bool:
    """Update specific session fields. Returns True if row was updated."""
    allowed = {
        "ended_at",
        "status",
        "learning_event_count",
        "context_hint",
        "last_learning_at",
        "concepts_created",
        "concepts_evolved",
        "data",
        "model_id",
        "platform_hint",
        "surface_id",
        "origin_id",
        "last_heartbeat",
        "pressure_score",
        "working_context_json",
        "last_previous_response",  # SESSION-009: dropout recovery
    }
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    protected_fields = set(kwargs) & _MANAGED_PROTECTED_FIELDS
    if not updates and not protected_fields:
        return False

    set_clause = ", ".join(f"{k} = ?" for k in updates)

    with _conn._db() as conn:
        row, managed_schema = _load_session_authority(conn, session_id)
        if row is None:
            return False
        binding_hash = row[0]
        binding_generation = row[1]
        if binding_hash is None:
            if not updates:
                return False
            legacy_predicate = " AND binding_hash IS NULL" if managed_schema else ""
            cursor = conn.execute(
                f"UPDATE sessions SET {set_clause} WHERE id = ?{legacy_predicate}",
                [*updates.values(), session_id],
            )
        else:
            try:
                generation = int(binding_generation)
            except (TypeError, ValueError):
                _deny_managed_write("generation_mismatch")
            _authorize_managed_helper_update(
                binding_hash=str(binding_hash),
                binding_generation=generation,
                fields=set(updates) | protected_fields,
            )
            cursor = conn.execute(
                f"""UPDATE sessions SET {set_clause}
                    WHERE id = ? AND binding_hash = ? AND binding_generation = ?""",
                [*updates.values(), session_id, binding_hash, generation],
            )
    return cursor.rowcount > 0


def mark_session_reflection_completed(session_id: str, *, reflected_at: str) -> bool:
    """Atomically merge the reflection marker under exact managed authority."""

    with _conn._db() as conn:
        authority_row, managed_schema = _load_session_authority(conn, session_id)
        if authority_row is None:
            return False
        binding_hash = authority_row[0]
        binding_generation = authority_row[1]
        row = conn.execute(
            "SELECT data FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        data: dict = {}
        if row is not None and row[0]:
            try:
                loaded = json.loads(row[0]) if isinstance(row[0], str) else row[0]
                if isinstance(loaded, dict):
                    data = loaded
            except (json.JSONDecodeError, TypeError):
                data = {}
        data["reflection_completed"] = True
        data["reflection_completed_at"] = reflected_at
        serialized = json.dumps(data)

        if binding_hash is None:
            legacy_predicate = " AND binding_hash IS NULL" if managed_schema else ""
            cursor = conn.execute(
                f"UPDATE sessions SET data = ? WHERE id = ?{legacy_predicate}",
                (serialized, session_id),
            )
        else:
            try:
                generation = int(binding_generation)
            except (TypeError, ValueError):
                _deny_managed_write("generation_mismatch")
            _authorize_managed_helper_update(
                binding_hash=str(binding_hash),
                binding_generation=generation,
                fields={"data"},
            )
            cursor = conn.execute(
                """UPDATE sessions SET data = ?
                   WHERE id = ? AND binding_hash = ? AND binding_generation = ?""",
                (serialized, session_id, binding_hash, generation),
            )
    return cursor.rowcount > 0


def get_session_counts(session_id: str) -> dict:
    """Fetch current concept counters for a session.

    SESSION-LEARN-MISMATCH-001: Used when counter attribution targets a session
    different from self.current_session (whose in-memory counters are stale).

    Returns dict with 'concepts_created', 'concepts_evolved', and
    'learning_event_count' keys.
    Returns empty dict if session not found.
    """
    with _conn._db() as conn:
        row = conn.execute(
            "SELECT concepts_created, concepts_evolved, learning_event_count FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
    if row:
        return {
            "concepts_created": int(row[0] or 0),
            "concepts_evolved": int(row[1] or 0),
            "learning_event_count": int(row[2] or 0),
        }
    return {}


def record_session_learning_commit(
    session_id: str,
    *,
    concepts_created_delta: int,
    concepts_evolved_delta: int,
    learning_event_delta: int | None = None,
    learned_at: str | None = None,
    require_active: bool = True,
) -> dict | None:
    """Atomically record successful learning output for a session."""
    if not session_id:
        return None
    if concepts_created_delta < 0 or concepts_evolved_delta < 0:
        raise ValueError("learning counter deltas must be non-negative")
    if learning_event_delta is not None and learning_event_delta < 0:
        raise ValueError("learning counter deltas must be non-negative")

    if (
        concepts_created_delta == 0
        and concepts_evolved_delta == 0
        and (learning_event_delta is None or learning_event_delta == 0)
    ):
        return get_session_counts(session_id) or None

    learned_at = learned_at or _utc_now_iso()
    if learning_event_delta is None:
        learning_event_delta = concepts_created_delta + concepts_evolved_delta
    with _conn._db() as conn:
        authority_row, _managed_schema = _load_session_authority(conn, session_id)
        if authority_row is None:
            return None
        binding_hash = authority_row[0]
        binding_generation = authority_row[1]
        where_clause = "WHERE id = ? AND status = 'active'" if require_active else "WHERE id = ?"
        where_values: list[object] = [session_id]
        if binding_hash is not None:
            try:
                generation = int(binding_generation)
            except (TypeError, ValueError):
                _deny_managed_write("generation_mismatch")
            _authorize_managed_helper_update(
                binding_hash=str(binding_hash),
                binding_generation=generation,
                fields={
                    "concepts_created",
                    "concepts_evolved",
                    "learning_event_count",
                    "last_learning_at",
                },
            )
            where_clause += " AND binding_hash = ? AND binding_generation = ?"
            where_values.extend((binding_hash, generation))
        cursor = conn.execute(
            f"""UPDATE sessions
               SET concepts_created = COALESCE(concepts_created, 0) + ?,
                   concepts_evolved = COALESCE(concepts_evolved, 0) + ?,
                   learning_event_count = COALESCE(learning_event_count, 0) + ?,
                   last_learning_at = ?
               {where_clause}""",
            (
                concepts_created_delta,
                concepts_evolved_delta,
                learning_event_delta,
                learned_at,
                *where_values,
            ),
        )
        if cursor.rowcount <= 0:
            return None
        row = conn.execute(
            "SELECT concepts_created, concepts_evolved, learning_event_count, last_learning_at FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
    if not row:
        return None
    return {
        "concepts_created": int(row[0] or 0),
        "concepts_evolved": int(row[1] or 0),
        "learning_event_count": int(row[2] or 0),
        "last_learning_at": row[3],
    }


def load_session(session_id: str) -> dict | None:
    """Load a single session by ID."""
    with read_snapshot_db("load_session") as conn:
        row = conn.execute(
            "SELECT id, started_at, ended_at, status, learning_event_count, "
            "context_hint, last_learning_at, last_heartbeat, working_context_json, "
            "agent_id, model_id, platform_hint, surface_id, origin_id "
            "FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
    if not row:
        return None
    return dict(row)


def load_active_sessions_by_origin(origin_id: str) -> list[dict]:
    """Load active session rows for a stable client/thread origin."""
    with read_snapshot_db("load_active_sessions_by_origin") as conn:
        rows = conn.execute(
            "SELECT id, started_at, ended_at, status, learning_event_count, "
            "context_hint, last_learning_at, last_heartbeat, working_context_json, "
            "agent_id, model_id, platform_hint, surface_id, origin_id "
            "FROM sessions WHERE origin_id = ? AND status = 'active' "
            "ORDER BY started_at DESC",
            (origin_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def list_sessions(status: str = None, limit: int = 50, since: str = None) -> list[dict]:
    """Query sessions with optional filters. Returns newest-first."""
    query = (
        "SELECT id, started_at, ended_at, status, learning_event_count, context_hint, last_learning_at FROM sessions"
    )
    conditions = []
    params = []

    if status:
        conditions.append("status = ?")
        params.append(status)
    if since:
        conditions.append("started_at >= ?")
        params.append(since)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY started_at DESC LIMIT ?"
    params.append(limit)

    with read_snapshot_db("list_sessions") as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def count_sessions(since: str = None) -> int:
    """Count sessions, optionally since a cutoff datetime."""
    if since:
        query = "SELECT COUNT(*) FROM sessions WHERE started_at >= ?"
        params = (since,)
    else:
        query = "SELECT COUNT(*) FROM sessions"
        params = ()

    with read_snapshot_db("count_sessions") as conn:
        row = conn.execute(query, params).fetchone()
    return row[0] if row else 0

def recover_interrupted_sessions(started_before: str | None = None) -> int:
    """Mark orphaned active sessions as 'interrupted'. Returns count fixed.

    SESSION-011: scope blast radius to sessions started before a cutoff.
    Pass started_before (ISO timestamp) to protect concurrent sessions from
    the current process. Defaults to None (marks ALL active sessions — legacy).

    Args:
        started_before: Only interrupt sessions whose started_at < this ISO
            timestamp. Use server startup time to avoid touching live sessions.
    """
    now = _utc_now_iso()
    with _conn._db() as conn:
        if started_before is not None:
            cursor = conn.execute(
                "UPDATE sessions SET status = 'interrupted', ended_at = ? "
                "WHERE status = 'active' AND binding_hash IS NULL AND started_at < ?",
                (now, started_before),
            )
        else:
            # Legacy path: interrupt ALL active sessions (backward compat)
            cursor = conn.execute(
                "UPDATE sessions SET status = 'interrupted', ended_at = ? "
                "WHERE status = 'active' AND binding_hash IS NULL",
                (now,),
            )
    count = cursor.rowcount
    if count > 0:
        logger.warning(
            f"Recovered {count} interrupted session(s) from previous run"
            + (f" (started before {started_before})" if started_before else "")
        )
    return count

def load_session_velocity(cutoff_iso: str, prior_cutoff_iso: str = None) -> dict:
    """Load session performance data for cognitive velocity computation.

    Returns aggregate stats for sessions in the window (cutoff → now),
    and optionally for the prior window (prior_cutoff → cutoff) for trend comparison.
    """

    def _aggregate(since: str, until: str = None) -> dict:
        with read_snapshot_db("load_session_velocity") as conn:
            where = "WHERE status IN ('ended', 'active') AND started_at >= ?"
            params = [since]
            if until:
                where += " AND started_at < ?"
                params.append(until)
            row = conn.execute(
                f"""
                SELECT
                    COUNT(*) as session_count,
                    COALESCE(SUM(learning_event_count), 0) as total_learning_events,
                    COALESCE(SUM(concepts_created), 0) as total_concepts_created,
                    COALESCE(SUM(concepts_evolved), 0) as total_concepts_evolved
                FROM sessions {where}
            """,
                params,
            ).fetchone()
        return {
            "session_count": row["session_count"],
            "total_learning_events": row["total_learning_events"],
            "total_concepts_created": row["total_concepts_created"],
            "total_concepts_evolved": row["total_concepts_evolved"],
        }

    current = _aggregate(cutoff_iso)
    prior = _aggregate(prior_cutoff_iso, cutoff_iso) if prior_cutoff_iso else None
    return {"current": current, "prior": prior}
