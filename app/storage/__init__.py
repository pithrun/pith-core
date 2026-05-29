"""SQLite storage backend for Pith.

Replaces YAML filesystem storage with single SQLite database.
All function signatures preserved for consumer compatibility except:
  - save_concept() returns None (no callers used returned Path)
  - knowledge_area CRUD removed (derived via DISTINCT query)
  - AccessTracker replaced with compat shim (direct DB writes in load_concept)

Phase 1B P0.2: Migration from YAML to SQLite for 10-20x performance improvement.

Item 2b decomposition: This __init__.py is a backward-compatible re-export shim.
All implementation lives in sub-modules. Every name previously available via
``from app.storage import X`` or ``mock.patch("app.storage.X")`` still works.
"""
# ruff: noqa: I001

# --- utils first: runs _migrate_db_filename() at import time ---
from app.storage.utils import (  # noqa: F401
    _AGENT_ID_PATTERN,
    CONCEPTS_DIR,
    DATA_DIR,
    DB_PATH,
    INDEX_DIR,
    SCHEMA_DDL,
    _clamp_score,
    _migrate_db_filename,
    _safe_json_loads,
    validate_agent_id,
)

# --- connection: DB handles, backend shims ---
from app.storage.connection import (  # noqa: F401
    _AccessTrackerShim,
    _KA_SENTINELS,
    access_tracker,
    db_immediate,
    diagnostic_read_db,
    diagnostic_snapshot_db,
    get_db_boundary_observability,
    get_db_connection,
    managed_write_db,
    open_owned_connection,
    owned_connection,
    read_snapshot_db,
    request_db_scope,
    required_context_read_db,
    reset_db_boundary_observability,
)

# _db, _db_immediate, _get_connection resolved dynamically via __getattr__
# so that mock.patch("app.storage.connection._db") is seen by lazy importers.

# --- concepts: CRUD, associations, FTS5 sync ---
from app.storage.concepts import (  # noqa: F401
    AssociationIndexes,
    AssociationIndexLoadResult,
    LifecycleTransitionError,
    _MAX_EDGES_PER_ENTITY,
    _TYPED_ASSOC_DEFAULT_STRENGTH,
    _invalidate_associations_cache,
    _resolve_knowledge_area,
    _sync_fts5,
    add_association,
    add_associations_bulk,
    add_typed_association,
    apply_lifecycle_transition_conn,
    archive_concept,
    count_associations,
    count_orphan_concepts,
    get_adjacency_graph,
    get_all_association_triples,
    get_association_triples_for_pairs,
    get_knowledge_area_map,
    get_next_version,
    get_next_version_conn,
    get_related_concepts,
    get_typed_association_count,
    get_typed_associations,
    list_archived_concepts,
    list_concepts,
    list_concepts_for_indexing,
    list_concepts_full,
    list_concepts_modified_since,
    load_all_versions,
    load_association_indexes,
    load_association_indexes_budgeted,
    load_associations,
    load_concept,
    load_concept_conn,
    load_concepts_batch,
    load_self_model,
    restore_concept,
    save_concept,
    save_concept_conn,
    shutdown_association_index_refresh,
    update_stale_risk_fields,
    save_self_model,
    update_concept_data,
)

# --- sessions ---
from app.storage.sessions import (  # noqa: F401
    count_sessions,
    list_sessions,
    load_active_sessions_by_origin,
    load_session,
    load_session_velocity,
    recover_interrupted_sessions,
    save_session,
    update_session,
)

# --- stats ---
from app.storage.stats import (  # noqa: F401
    analyze_coverage_threshold,
    analyze_session_drops,
    get_distribution_report,
    get_memory_projection_data,
    get_pith_health_fast,
    get_pith_stats_aggregates,
    get_pith_stats_fast,
)

# --- checkpoints ---
from app.storage.checkpoints import (  # noqa: F401
    DEFAULT_TTL_DAYS,
    STALE_CHECKPOINT_HOURS,
    archive_stale_checkpoints,
    cleanup_expired_checkpoints,
    cleanup_expired_snapshots,
    complete_checkpoint,
    compress_checkpoint,
    get_checkpoint_dashboard,
    get_checkpoint_effectiveness,
    list_checkpoints,
    load_checkpoint,
    load_resume_snapshot,
    save_checkpoint,
    save_resume_snapshot,
    touch_checkpoint,
)

# --- queries ---
from app.storage.queries import (  # noqa: F401
    count_concepts_by_type_tier,
    get_high_authority_concepts_by_ka,
    get_metadata,
    list_concepts_for_knowledge_area,
    list_knowledge_area_summaries,
    load_always_activate_concepts,
    load_concepts_by_type,
    load_firmware,
    load_recent_concepts,
    load_recent_concepts_by_types,
    save_firmware,
    set_always_activate,
    set_metadata,
)

# --- verbatim ---
from app.storage.verbatim import (  # noqa: F401
    _recompute_fragment_keywords,
    _sync_fts5_verbatim,
    delete_verbatim_fragment,
    delete_verbatim_fragments_for_concept,
    extract_fragment_keywords,
    get_verbatim_fragments,
    get_verbatim_stats,
    repair_fts_verbatim,
    save_verbatim_fragment,
    search_verbatim_fts5,
    search_verbatim_fts5_dual,
)

# --- tokens ---
from app.storage.tokens import (  # noqa: F401
    create_agent_token,
    list_agent_tokens,
    resolve_agent_token,
    revoke_agent_token,
)

# --- Re-export datetime utils (some callers import via app.storage) ---
from app.core.datetime_utils import _utc_now, _utc_now_iso  # noqa: F401

# --- Existing sub-modules (not part of Item 2b decomposition) ---
# app.storage.backend    -- StorageBackend / SQLiteBackend
# app.storage.embedding  -- EmbeddingStore
# app.storage.migration  -- schema migration logic
# app.storage.sql_compat -- SQL compatibility helpers
#
# These are imported by consumers directly (e.g. from app.storage.migration import ...)
# and do not need re-export here.


# --- Proxy DB handles ---
# Explicit proxy functions so _db/_db_immediate/_get_connection live in
# app.storage.__dict__ as real callables. This means:
#   1. mock.patch("app.storage._db", mock) replaces the proxy directly
#   2. mock.patch("app.storage.connection._db", mock) is picked up at call
#      time via the lazy import inside the proxy body
# Both patch targets work; single-patch at either level is sufficient.
from contextlib import contextmanager as _cm  # noqa: E402
import importlib as _importlib  # noqa: E402
import json as _json  # noqa: E402
import os as _os  # noqa: E402


@_cm
def _db(*, timeout_s: float = 30.0, operation: str = "db"):  # noqa: F811
    import app.storage.connection as _c

    with _c._db(timeout_s=timeout_s, operation=operation) as conn:
        yield conn


@_cm
def _db_immediate(*, timeout_s: float = 30.0, operation: str = "db_immediate"):  # noqa: F811
    import app.storage.connection as _c

    with _c._db_immediate(timeout_s=timeout_s, operation=operation) as conn:
        yield conn


def _get_connection():  # noqa: F811
    import app.storage.connection as _c

    return _c._get_connection()


def record_governance_event(
    event_type: str,
    *,
    details: dict | None = None,
    session_id: str | None = None,
    concept_id: str | None = None,
    latency_remaining_ms: float | None = None,
    created_at: str | None = None,
) -> None:
    """Public storage-layer helper for non-critical governance event writes."""
    import json

    timestamp = created_at or _utc_now_iso()
    with _db() as conn:
        conn.execute(
            """INSERT INTO governance_events
               (session_id, event_type, concept_id, details, latency_remaining_ms, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                event_type,
                concept_id,
                json.dumps(details) if details is not None else None,
                latency_remaining_ms,
                timestamp,
            ),
        )


def _env_flag_enabled(name: str | None) -> bool:
    return bool(name) and _os.environ.get(name, "").lower() in ("true", "1")


def run_storage_migration(module_name: str) -> dict:
    """Run an idempotent migration module inside the storage transaction boundary."""
    mod = _importlib.import_module(module_name)
    migration_id = getattr(mod, "MIGRATION_ID", None)
    description = getattr(mod, "DESCRIPTION", module_name)
    force_env_var = getattr(mod, "FORCE_ENV_VAR", None)
    readonly_skip_env = getattr(mod, "READONLY_SKIP_ENV", None)
    needs_migration = getattr(mod, "needs_migration", None)

    if _env_flag_enabled(readonly_skip_env):
        return {
            "status": "skipped",
            "reason": "readonly",
            "migration_id": migration_id,
        }

    forced = _env_flag_enabled(force_env_var)

    with _db() as conn:
        if migration_id:
            from app.storage.migration import _is_migration_applied, _record_migration

            if not forced and _is_migration_applied(conn, migration_id):
                return {
                    "status": "skipped",
                    "reason": "checkpoint",
                    "migration_id": migration_id,
                }

            if not forced and callable(needs_migration) and not needs_migration(conn):
                _record_migration(conn, migration_id, description)
                return {
                    "status": "skipped",
                    "reason": "no_work",
                    "migration_id": migration_id,
                }

            result = mod.migrate(conn)
            if not (isinstance(result, dict) and result.get("status") == "skipped"):
                _record_migration(conn, migration_id, description)
            return result if isinstance(result, dict) else {"status": "success", "migration_id": migration_id}

        result = mod.migrate(conn)
        return result if isinstance(result, dict) else {"status": "success", "migration_id": None}


def _write_replay_row_to_dict(row) -> dict:
    if hasattr(row, "keys"):
        return {key: row[key] for key in row.keys()}
    return dict(row)


def load_write_request_replay(endpoint: str, profile: str, request_id: str) -> dict | None:
    """Return replay metadata for a prior idempotent write, if any."""
    with _db() as conn:
        row = conn.execute(
            "SELECT status, response_json, request_json, attempt_count, last_error, "
            "lease_owner, lease_expires_at, next_retry_at, updated_at FROM write_request_replays "
            "WHERE endpoint=? AND profile=? AND request_id=?",
            (endpoint, profile, request_id),
        ).fetchone()
    if not row:
        return None
    data = _write_replay_row_to_dict(row)
    return {
        "status": data["status"],
        "response": _json.loads(data["response_json"]) if data.get("response_json") else None,
        "request": _json.loads(data["request_json"]) if data.get("request_json") else None,
        "request_json": data.get("request_json"),
        "attempt_count": int(data.get("attempt_count") or 0),
        "last_error": data.get("last_error"),
        "lease_owner": data.get("lease_owner"),
        "lease_expires_at": data.get("lease_expires_at"),
        "next_retry_at": data.get("next_retry_at"),
        "updated_at": data["updated_at"],
    }


def insert_write_request_processing(
    endpoint: str,
    profile: str,
    request_id: str,
    now: str,
    request_payload: dict | None = None,
) -> None:
    request_json = _json.dumps(request_payload) if request_payload is not None else None
    with _db() as conn:
        conn.execute(
            "INSERT INTO write_request_replays(endpoint, profile, request_id, status, response_json, request_json, "
            "attempt_count, last_error, lease_owner, lease_expires_at, next_retry_at, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (endpoint, profile, request_id, "processing", None, request_json, 0, None, None, None, None, now, now),
        )


def mark_write_request_processing(
    endpoint: str,
    profile: str,
    request_id: str,
    now: str,
    request_payload: dict | None = None,
) -> None:
    request_json = _json.dumps(request_payload) if request_payload is not None else None
    with _db() as conn:
        if request_json is None:
            conn.execute(
                "UPDATE write_request_replays SET status=?, response_json=NULL, lease_owner=NULL, lease_expires_at=NULL, updated_at=? "
                "WHERE endpoint=? AND profile=? AND request_id=?",
                ("processing", now, endpoint, profile, request_id),
            )
        else:
            conn.execute(
                "UPDATE write_request_replays SET status=?, response_json=NULL, request_json=COALESCE(request_json, ?), "
                "lease_owner=NULL, lease_expires_at=NULL, updated_at=? "
                "WHERE endpoint=? AND profile=? AND request_id=?",
                ("processing", request_json, now, endpoint, profile, request_id),
            )


def commit_write_request_replay(
    endpoint: str,
    profile: str,
    request_id: str,
    response: dict,
    now: str,
) -> None:
    with _db() as conn:
        conn.execute(
            "UPDATE write_request_replays SET status=?, response_json=?, request_json=NULL, "
            "last_error=NULL, lease_owner=NULL, lease_expires_at=NULL, next_retry_at=NULL, updated_at=? "
            "WHERE endpoint=? AND profile=? AND request_id=?",
            ("committed", _json.dumps(response), now, endpoint, profile, request_id),
        )


def fail_write_request_replay(
    endpoint: str,
    profile: str,
    request_id: str,
    response: dict,
    now: str,
    error_class: str,
) -> int:
    payload = dict(response)
    payload.setdefault("status", "failed")
    payload.setdefault("persistence_state", "failed")
    payload.setdefault("request_id", request_id)
    payload.setdefault("error_class", error_class)
    with _db() as conn:
        cur = conn.execute(
            "UPDATE write_request_replays SET status=?, response_json=?, request_json=NULL, "
            "last_error=?, lease_owner=NULL, lease_expires_at=NULL, next_retry_at=NULL, updated_at=? "
            "WHERE endpoint=? AND profile=? AND request_id=?",
            ("failed", _json.dumps(payload), error_class[:1000], now, endpoint, profile, request_id),
        )
        return int(cur.rowcount or 0)


def delete_processing_write_request(endpoint: str, profile: str, request_id: str) -> None:
    with _db() as conn:
        conn.execute(
            "DELETE FROM write_request_replays WHERE endpoint=? AND profile=? AND request_id=? AND status=?",
            (endpoint, profile, request_id, "processing"),
        )


def fail_unrecoverable_stale_write_requests(
    endpoint: str,
    profile: str,
    stale_before_iso: str,
    now: str,
    error_class: str,
    limit: int,
) -> int:
    if limit <= 0:
        return 0
    failed = 0
    with _db_immediate() as conn:
        rows = conn.execute(
            """SELECT request_id
               FROM write_request_replays
               WHERE endpoint=? AND profile=? AND status='processing'
                 AND updated_at < ?
                 AND request_json IS NULL
               ORDER BY updated_at
               LIMIT ?""",
            (endpoint, profile, stale_before_iso, limit),
        ).fetchall()
        for row in rows:
            data = _write_replay_row_to_dict(row)
            request_id = data["request_id"]
            response = {
                "status": "failed",
                "persistence_state": "failed",
                "request_id": request_id,
                "error_class": error_class,
            }
            cur = conn.execute(
                "UPDATE write_request_replays SET status=?, response_json=?, request_json=NULL, "
                "last_error=?, lease_owner=NULL, lease_expires_at=NULL, next_retry_at=NULL, updated_at=? "
                "WHERE endpoint=? AND profile=? AND request_id=? AND status='processing' "
                "AND updated_at < ? AND request_json IS NULL",
                (
                    "failed",
                    _json.dumps(response),
                    error_class[:1000],
                    now,
                    endpoint,
                    profile,
                    request_id,
                    stale_before_iso,
                ),
            )
            failed += int(cur.rowcount or 0)
    return failed


def record_checkpoint_save_event(session_id: str | None, task_id: str) -> None:
    with _db(timeout_s=0.05, operation="checkpoint_measurement") as conn:
        conn.execute(
            "INSERT INTO governance_events (session_id, event_type, details, created_at) VALUES (?, ?, ?, ?)",
            (
                session_id,
                "checkpoint_save",
                _json.dumps({"task_id": task_id, "action": "save", "source": "api"}),
                _utc_now_iso(),
            ),
        )


def summarize_write_request_processing(endpoint: str, profile: str, stale_before_iso: str, max_attempts: int = 3) -> dict:
    with _db() as conn:
        rows = conn.execute(
            """SELECT request_id, updated_at, request_json, attempt_count
               FROM write_request_replays
               WHERE endpoint=? AND profile=? AND status=?
               ORDER BY updated_at""",
            (endpoint, profile, "processing"),
        ).fetchall()
    processing_count = len(rows)
    stale_rows = []
    recoverable = 0
    unrecoverable = 0
    max_attempts_exhausted = 0
    for row in rows:
        data = _write_replay_row_to_dict(row)
        is_stale = data["updated_at"] < stale_before_iso
        if not is_stale:
            continue
        stale_rows.append(data)
        if data.get("request_json"):
            recoverable += 1
        else:
            unrecoverable += 1
        if int(data.get("attempt_count") or 0) >= max_attempts:
            max_attempts_exhausted += 1
    oldest = rows[0] if rows else None
    oldest_stale = stale_rows[0] if stale_rows else None
    return {
        "processing_count": processing_count,
        "stale_processing_count": len(stale_rows),
        "recoverable_stale_count": recoverable,
        "unrecoverable_stale_count": unrecoverable,
        "max_attempts_exhausted_count": max_attempts_exhausted,
        "oldest_processing_updated_at": _write_replay_row_to_dict(oldest)["updated_at"] if oldest else None,
        "oldest_stale_request_id": oldest_stale["request_id"] if oldest_stale else None,
        "oldest_stale_updated_at": oldest_stale["updated_at"] if oldest_stale else None,
    }


def claim_stale_write_requests(
    endpoint: str,
    profile: str,
    stale_before_iso: str,
    lease_owner: str,
    lease_expires_at: str,
    limit: int,
    now: str,
    max_attempts: int,
) -> list[dict]:
    if limit <= 0:
        return []
    claimed: list[dict] = []
    with _db_immediate() as conn:
        rows = conn.execute(
            """SELECT endpoint, profile, request_id, request_json, attempt_count, updated_at
               FROM write_request_replays
               WHERE endpoint=? AND profile=? AND status='processing'
                 AND updated_at < ?
                 AND request_json IS NOT NULL
                 AND (next_retry_at IS NULL OR next_retry_at <= ?)
                 AND (lease_expires_at IS NULL OR lease_expires_at <= ?)
                 AND COALESCE(attempt_count, 0) < ?
               ORDER BY updated_at
               LIMIT ?""",
            (endpoint, profile, stale_before_iso, now, now, max_attempts, limit),
        ).fetchall()
        for row in rows:
            data = _write_replay_row_to_dict(row)
            cur = conn.execute(
                """UPDATE write_request_replays
                   SET attempt_count=COALESCE(attempt_count, 0)+1,
                       last_error=NULL,
                       lease_owner=?,
                       lease_expires_at=?,
                       updated_at=?
                   WHERE endpoint=? AND profile=? AND request_id=?
                     AND status='processing'
                     AND updated_at < ?
                     AND request_json IS NOT NULL
                     AND (next_retry_at IS NULL OR next_retry_at <= ?)
                     AND (lease_expires_at IS NULL OR lease_expires_at <= ?)
                     AND COALESCE(attempt_count, 0) < ?""",
                (
                    lease_owner,
                    lease_expires_at,
                    now,
                    endpoint,
                    profile,
                    data["request_id"],
                    stale_before_iso,
                    now,
                    now,
                    max_attempts,
                ),
            )
            if cur.rowcount:
                data["request"] = _json.loads(data["request_json"])
                data["lease_owner"] = lease_owner
                data["lease_expires_at"] = lease_expires_at
                data["attempt_count"] = int(data.get("attempt_count") or 0) + 1
                claimed.append(data)
    return claimed


def mark_write_request_reclaim_failed(
    endpoint: str,
    profile: str,
    request_id: str,
    last_error: str,
    next_retry_at: str,
    now: str,
) -> None:
    with _db() as conn:
        conn.execute(
            """UPDATE write_request_replays
               SET last_error=?, next_retry_at=?, lease_owner=NULL, lease_expires_at=NULL, updated_at=?
               WHERE endpoint=? AND profile=? AND request_id=? AND status='processing'""",
            (last_error[:1000], next_retry_at, now, endpoint, profile, request_id),
        )
