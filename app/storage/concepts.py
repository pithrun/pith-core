"""Storage sub-module: concepts.

Concept CRUD, associations, FTS5 sync, self-model.
Extracted from storage/__init__.py during Item 2b decomposition.
"""
import json
import logging
import math
import re
import sqlite3
import threading
import time as _time_mod
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, timedelta
from pathlib import Path
from typing import Optional

import app.storage.connection as _conn
from app.core.config import BENCHMARK_READONLY
from app.core.datetime_utils import _utc_now, _utc_now_iso
from app.core.models import Concept
from app.storage.connection import access_tracker, read_snapshot_db
from app.storage.utils import DB_PATH, _clamp_score, _safe_json_loads, validate_agent_id


@dataclass(frozen=True)
class AssociationIndexes:
    """Derived association indexes built from one association edge snapshot."""

    edges: list[dict]
    adjacency: dict[str, list[str]]
    edge_relations: dict[tuple[str, str], str]
    edge_strength: dict[tuple[str, str], float]
    contradiction_signals: list[tuple[str, str]]


@dataclass(frozen=True)
class AssociationIndexLoadResult:
    """Deadline-aware association index load result."""

    indexes: AssociationIndexes | None
    state: str
    age_ms: float | None = None
    refresh_scheduled: bool = False
    refresh_in_flight: bool = False


# PERF-016: Module-level association cache
_associations_cache_lock = threading.RLock()
_associations_cache: dict | None = None
_associations_cache_ts: float = 0.0
_ASSOCIATIONS_CACHE_TTL_S: float = 60.0  # 60-second TTL
_association_indexes_cache: AssociationIndexes | None = None
_association_indexes_cache_ts: float = 0.0
_association_index_refresh_executor: ThreadPoolExecutor | None = None
_association_index_refresh_future: Future | None = None

# MONITOR-031/042: Cache hit/miss counters (module-level, reset on process restart)
_assoc_cache_hits: int = 0
_assoc_cache_misses: int = 0
_adjacency_cache_hits: int = 0
_adjacency_cache_misses: int = 0
_assoc_index_cache_hits: int = 0
_assoc_index_cache_misses: int = 0

# PERF-023: Adjacency graph cache (derived from association list, same TTL)
_adjacency_graph_cache: dict[str, dict[str, float]] | None = None
_adjacency_graph_cache_ts: float = 0.0

_ASSOCIATION_EDGE_TYPE_MULTIPLIER = {
    "structural_analogy": 1.5,
    "supports": 1.2,
    "part_of": 1.1,
    "derived_from": 1.0,
    "enables": 1.0,
    "constrains": 0.8,
    "related_to": 0.7,
}

logger = logging.getLogger(__name__)

# DATA-028: advisory lock for restore_concept
_restore_concept_lock = threading.Lock()

_KA_SENTINELS = {None, "", "general", "unclassified", "unknown"}
_STALE_RISK_RESET_COLUMNS = (
    "staleness_state",
    "staleness_score",
    "staleness_reason",
    "staleness_evaluated_at",
    "staleness_detector_version",
)


class LifecycleTransitionError(ValueError):
    """Raised when a caller asks for an invalid concept lifecycle transition."""


def _json_set_status_expr(*paths: tuple[str, str]) -> str:
    """Build a json_set expression with literal JSON paths and bound values."""
    if not paths:
        return "COALESCE(data, '{}')"
    args = ", ".join(f"'{path}', :{param}" for path, param in paths)
    return f"json_set(COALESCE(data, '{{}}'), {args})"


def _concepts_has_column(conn: sqlite3.Connection, column: str) -> bool:
    try:
        return any(row[1] == column for row in conn.execute("PRAGMA table_info(concepts)").fetchall())
    except Exception:
        return False


_CONCEPT_SQL_HYDRATED_FIELDS = (
    "created_at",
    "valid_from",
    "content_updated_at",
    "session_id",
    "original_date",
)


def _hydrate_concept_sql_columns(data: dict, row: sqlite3.Row, fields: tuple[str, ...] = _CONCEPT_SQL_HYDRATED_FIELDS) -> None:
    """Prefer canonical concepts-table columns over stale JSON blob values."""
    for field in fields:
        try:
            value = row[field]
        except (IndexError, KeyError):
            continue
        if value is not None:
            data[field] = value


def _data_update_sql(
    conn: sqlite3.Connection,
    *paths: tuple[str, str],
    remove_paths: tuple[str, ...] = (),
) -> str:
    """Return a guarded JSON mirror assignment for schemas that have data."""
    if not _concepts_has_column(conn, "data"):
        return ""
    expr = _json_set_status_expr(*paths)
    if remove_paths:
        removes = ", ".join(f"'{path}'" for path in remove_paths)
        expr = f"json_remove({expr}, {removes})"
    return f", data = {expr}"


def apply_lifecycle_transition_conn(
    conn: sqlite3.Connection,
    concept_id: str,
    transition: str,
    *,
    superseded_by: str | None = None,
    reason: str | None = None,
    confidence: float | None = None,
    valid_until: str | None = None,
    on_archived: "callable | None" = None,
) -> int:
    """Apply one canonical lifecycle transition to SQL columns and JSON mirrors.

    DATA-070: status, is_current, currency_status, superseded_by, and their JSON
    mirrors must move atomically through this gateway.
    """
    now = valid_until if transition == "supersede" and valid_until is not None else _utc_now_iso()
    params: dict[str, object] = {
        "concept_id": concept_id,
        "now": now,
        "reason": reason,
        "superseded_by": superseded_by,
    }

    if transition == "archive":
        sql = f"""UPDATE concepts
                  SET status = 'archived',
                      is_current = 0,
                      updated_at = :now
                      {_data_update_sql(conn, ('$.status', 'status'))}
                  WHERE id = :concept_id AND status = 'active'"""
        params["status"] = "archived"
    elif transition == "restore_archive":
        reason_sql = "supersession_reason = NULL," if _concepts_has_column(conn, "supersession_reason") else ""
        sql = f"""UPDATE concepts
                  SET status = 'active',
                      is_current = 1,
                      currency_status = 'ACTIVE',
                      superseded_by = NULL,
                      superseded_at = NULL,
                      {reason_sql}
                      updated_at = :now
                      {_data_update_sql(
                          conn,
                          ('$.status', 'status'),
                          ('$.currency_status', 'currency_status'),
                          remove_paths=('$.superseded_by',),
                      )}
                  WHERE id = :concept_id AND status = 'archived'"""
        params["status"] = "active"
        params["currency_status"] = "ACTIVE"
    elif transition == "supersede":
        if not superseded_by:
            raise LifecycleTransitionError("supersede transition requires superseded_by")
        params["currency_status"] = "SUPERSEDED"
        params["status"] = "superseded"
        params["confidence"] = confidence
        params["valid_until"] = valid_until
        confidence_sql = ""
        confidence_json: list[tuple[str, str]] = []
        if confidence is not None:
            confidence_sql = ", confidence = :confidence"
            confidence_json.append(("$.confidence", "confidence"))
        valid_until_sql = (
            ", valid_until = :valid_until"
            if valid_until is not None and _concepts_has_column(conn, "valid_until")
            else ""
        )
        reason_sql = (
            ", supersession_reason = COALESCE(:reason, supersession_reason)"
            if _concepts_has_column(conn, "supersession_reason")
            else ""
        )
        sql = f"""UPDATE concepts
                  SET status = 'superseded',
                      is_current = 0,
                      currency_status = 'SUPERSEDED',
                      superseded_by = :superseded_by,
                      superseded_at = :now,
                      updated_at = :now
                      {reason_sql}
                      {confidence_sql}
                      {valid_until_sql}
                      {_data_update_sql(
                          conn,
                          ('$.status', 'status'),
                          ('$.currency_status', 'currency_status'),
                          ('$.superseded_by', 'superseded_by'),
                          *confidence_json,
                      )}
                  WHERE id = :concept_id
                    AND (superseded_by IS NULL OR superseded_by = :superseded_by)"""
    elif transition == "rollback_supersession":
        if not superseded_by:
            raise LifecycleTransitionError("rollback_supersession transition requires superseded_by")
        reason_sql = ", supersession_reason = NULL" if _concepts_has_column(conn, "supersession_reason") else ""
        sql = f"""UPDATE concepts
                  SET status = 'active',
                      is_current = 1,
                      currency_status = 'ACTIVE',
                      superseded_by = NULL,
                      superseded_at = NULL,
                      updated_at = :now
                      {reason_sql}
                      {_data_update_sql(
                          conn,
                          ('$.status', 'status'),
                          ('$.currency_status', 'currency_status'),
                          remove_paths=('$.superseded_by',),
                      )}
                  WHERE id = :concept_id AND superseded_by = :superseded_by"""
        params["status"] = "active"
        params["currency_status"] = "ACTIVE"
    else:
        raise LifecycleTransitionError(f"unknown lifecycle transition: {transition}")

    cursor = conn.execute(sql, params)
    if cursor.rowcount <= 0:
        return 0

    if transition in {"archive", "supersede"}:
        _sync_fts5(conn, concept_id, delete=True)
        try:
            conn.execute("DELETE FROM fts_verbatim WHERE concept_id = ?", (concept_id,))
        except Exception as exc:
            logger.warning("FTS verbatim delete failed for lifecycle transition %s: %s", concept_id, exc)

    if transition == "archive":
        orphaned = conn.execute(
            "DELETE FROM associations WHERE source = ? OR target = ?",
            (concept_id, concept_id),
        ).rowcount
        if orphaned > 0:
            logger.info("Removed %d orphaned edges for archived concept %s", orphaned, concept_id)
            _invalidate_associations_cache()
        if on_archived:
            try:
                on_archived(concept_id)
                logger.debug("on_archived callback invoked for %s", concept_id)
            except Exception as cb_err:
                logger.warning("on_archived callback failed for %s (non-fatal): %s", concept_id, cb_err)

    return cursor.rowcount


def clear_stale_risk_metadata(data: dict) -> dict:
    """Reset stale-risk lifecycle metadata for content-changing writes."""
    if not isinstance(data, dict):
        return {}
    for field in _STALE_RISK_RESET_COLUMNS:
        data[field] = None
    data["staleness_consecutive_hits"] = 0
    return data


def _stale_risk_field_values(data: dict) -> tuple:
    """Return stale-risk field tuple in DB column order."""
    return (
        data.get("staleness_state"),
        _clamp_score(data.get("staleness_score")),
        data.get("staleness_reason"),
        data.get("staleness_evaluated_at"),
        data.get("staleness_detector_version"),
        int(data.get("staleness_consecutive_hits") or 0),
    )


def _apply_summary_change_reset(conn, concept_id: str, data: dict, new_summary: str | None) -> tuple[bool, str | None]:
    """Clear stale-risk lifecycle metadata when summary text changes."""
    if new_summary is None:
        return False, None
    old_row = conn.execute("SELECT summary FROM concepts WHERE id = ?", (concept_id,)).fetchone()
    summary_changed = bool(old_row and old_row[0] != new_summary)
    if summary_changed:
        clear_stale_risk_metadata(data)
    return summary_changed, old_row[0] if old_row else None

def _resolve_knowledge_area(concept, meta: dict) -> str:
    """Resolve knowledge_area with sentinel-awareness + taxonomy normalization.

    KA-005 fix: The old or-chain `concept.knowledge_area or meta.get(...) or 'general'`
    treated "general" as truthy, preventing metadata fallback from ever being reached.
    This helper treats sentinel values (None, '', 'general', 'unclassified', 'unknown')
    as "not classified" and falls through to metadata, which often contains the correct KA
    from reclassification or ingestion.

    KA-004: All resolved KAs are normalized through the taxonomy before returning.
    Uses permissive mode (strict=False) — novel KAs pass through with WARNING log.
    Wrapped in try-except so taxonomy failures never break concept writes.
    """
    # Priority 1: concept-level KA if it's a real classification
    concept_ka = getattr(concept, "knowledge_area", None)
    if concept_ka and concept_ka not in _KA_SENTINELS:
        resolved = concept_ka
    # Priority 2: metadata KA (often set by reclassification)
    elif (meta_ka := (meta.get("knowledge_area") if meta else None)) and meta_ka not in _KA_SENTINELS:
        resolved = meta_ka
    # Priority 3: nested metadata inside concept data
    elif hasattr(concept, "metadata") and isinstance(concept.metadata, dict):
        nested_ka = concept.metadata.get("knowledge_area")
        if nested_ka and nested_ka not in _KA_SENTINELS:
            resolved = nested_ka
        else:
            resolved = concept_ka if concept_ka else "general"
    else:
        # Fallback: preserve existing non-sentinel or default
        resolved = concept_ka if concept_ka else "general"

    # KA-004: Enforce taxonomy normalization at storage layer (single chokepoint)
    try:
        from app.core.taxonomy_utils import normalize_knowledge_area  # DEBT-234

        normalized, source = normalize_knowledge_area(resolved, strict=False)
        if source == "novel":
            logger.warning("KA-004: Novel KA '%s' accepted at storage layer (no canonical match)", resolved)
        return normalized
    except Exception as e:
        logger.error("KA-004: Taxonomy normalization failed — writing raw KA '%s': %s", resolved, e)
        return resolved


def update_concept_data(
    conn, concept_id: str, data: dict, *, extra_sets: str = "", extra_params: tuple = (), require_current: bool = True
) -> int:
    """Gateway for writing concept JSON blobs with automatic column sync.

    KA-006: Any code that writes `SET data = ?` on the concepts table should
    use this function instead of raw SQL.  It ensures knowledge_area, confidence,
    maturity and other dual-tracked fields stay in sync between the SQL columns
    and the JSON blob, preventing the class of desync bugs that KA-005 fixed
    in save_concept.

    Args:
        conn: SQLite connection (caller manages transaction / commit).
        concept_id: The concept UUID to update.
        data: The full JSON-serialisable dict to write as the blob.
        extra_sets: Optional additional SET clause fragments (e.g. "currency_status = ?").
        extra_params: Parameters corresponding to extra_sets placeholders.

    Returns:
        Number of rows updated (0 or 1).
    """
    # --- KA sentinel-aware resolution (mirrors _resolve_knowledge_area logic) ---
    meta = data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {}
    blob_ka = data.get("knowledge_area")
    meta_ka = meta.get("knowledge_area")

    if blob_ka in _KA_SENTINELS and meta_ka and meta_ka not in _KA_SENTINELS:
        data["knowledge_area"] = meta_ka
        resolved_ka = meta_ka
    elif blob_ka and blob_ka not in _KA_SENTINELS:
        resolved_ka = blob_ka
    else:
        resolved_ka = blob_ka or "general"

    # DATA-020 + COGGOV-014: content changes refresh content_updated_at and clear stale-risk lifecycle state
    new_summary = data.get("summary")
    summary_changed, _ = _apply_summary_change_reset(conn, concept_id, data, new_summary)

    # --- Sync dual-tracked fields from blob → columns ---
    now = _utc_now_iso()
    set_parts = ["data = ?", "knowledge_area = ?", "updated_at = ?"]
    params: list = [json.dumps(data), resolved_ka, now]

    if summary_changed:
        set_parts.extend([
            "content_updated_at = ?",
            "staleness_state = ?",
            "staleness_score = ?",
            "staleness_reason = ?",
            "staleness_evaluated_at = ?",
            "staleness_detector_version = ?",
            "staleness_consecutive_hits = ?",
        ])
        params.extend([now, *_stale_risk_field_values(data)])

    confidence = data.get("confidence")
    if confidence is not None:
        set_parts.append("confidence = ?")
        params.append(round(float(confidence), 6))

    maturity = data.get("maturity")
    if maturity is not None:
        set_parts.append("maturity = ?")
        params.append(maturity)

    if extra_sets:
        set_parts.append(extra_sets)
        params.extend(extra_params)

    params.append(concept_id)
    set_clause = ", ".join(set_parts)
    where = "WHERE id = ? AND is_current = 1" if require_current else "WHERE id = ?"
    cursor = conn.execute(
        f"UPDATE concepts SET {set_clause} {where}",
        tuple(params),
    )
    return cursor.rowcount


def update_stale_risk_fields(
    conn,
    concept_id: str,
    *,
    state: str | None = None,
    score: float | None = None,
    reason: str | None = None,
    evaluated_at: str | None = None,
    detector_version: str | None = None,
    consecutive_hits: int = 0,
    clear: bool = False,
    require_current: bool = True,
) -> int:
    """Update stale-risk lifecycle fields without mutating freshness timestamps."""
    where = "WHERE id = ? AND is_current = 1" if require_current else "WHERE id = ?"
    row = conn.execute(f"SELECT data FROM concepts {where}", (concept_id,)).fetchone()
    if not row:
        return 0

    data = _safe_json_loads(row["data"], context=f"update_stale_risk_fields({concept_id})") or {}
    if clear:
        clear_stale_risk_metadata(data)
    else:
        data["staleness_state"] = state
        data["staleness_score"] = _clamp_score(score)
        data["staleness_reason"] = reason
        data["staleness_evaluated_at"] = evaluated_at
        data["staleness_detector_version"] = detector_version
        data["staleness_consecutive_hits"] = int(consecutive_hits or 0)

    values = _stale_risk_field_values(data)
    params = [*values, json.dumps(data), concept_id]
    cursor = conn.execute(
        f"""UPDATE concepts SET
                staleness_state = ?,
                staleness_score = ?,
                staleness_reason = ?,
                staleness_evaluated_at = ?,
                staleness_detector_version = ?,
                staleness_consecutive_hits = ?,
                data = ?
            {where}""",
        tuple(params),
    )
    return cursor.rowcount

def _sync_fts5(conn, concept_id: str, summary: str | None = None, delete: bool = False):
    """Sync a single concept to the FTS5 full-text index.

    Called by save_concept (upsert) and delete paths. Non-fatal on failure.
    RETRIEVAL-042 upgrade: keeps FTS5 index in sync with concepts table.
    INGEST-037-L4 Amendment A1: Self-reads fragment_keywords from concepts table
    so ALL callers automatically produce enriched FTS5 entries.
    """
    try:
        if delete:
            conn.execute("DELETE FROM fts_concepts WHERE concept_id = ?", (concept_id,))
        elif summary:
            # INGEST-037 L4: Self-read fragment keywords from DB
            enriched = summary
            try:
                fk_row = conn.execute(
                    "SELECT fragment_keywords FROM concepts WHERE id = ?",
                    (concept_id,),
                ).fetchone()
                fk = fk_row[0] if fk_row and fk_row[0] else None
                if fk:
                    enriched = f"{summary} [frag: {fk}]"
            except Exception:
                pass  # Column may not exist yet (pre-migration) — degrade gracefully

            # Upsert: delete old entry then insert new
            conn.execute("DELETE FROM fts_concepts WHERE concept_id = ?", (concept_id,))
            conn.execute(
                "INSERT INTO fts_concepts(concept_id, summary) VALUES (?, ?)",
                (concept_id, enriched),
            )
    except Exception as e:
        logger.warning(f"FTS5 sync failed for {concept_id}: {e}")

def save_concept(concept: Concept) -> bool | None:
    """Save concept to SQLite. Writes to both concepts (latest) and concept_versions.

    Uses INSERT for new concepts (inherits column defaults like always_activate=0)
    and UPDATE for existing concepts (preserves column-level flags like always_activate).

    BUG FIX: Previously used INSERT OR REPLACE which deletes-then-inserts, wiping
    any columns not in the INSERT list (e.g., always_activate) back to DEFAULT.
    """
    data = concept.model_dump()
    meta = data.get("metadata", {})
    now = _utc_now_iso()
    content_updated_at = getattr(concept, "content_updated_at", None) or now

    # DEBT-185: Resolve KA once, sync to blob before serialization.
    # Previously the column got the resolved KA but json.dumps(data) kept the stale value.
    resolved_ka = _resolve_knowledge_area(concept, meta)
    data["knowledge_area"] = resolved_ka

    # GATE-BENCHMARKS: Skip benchmark concepts outside benchmark mode.
    # RETRIEVAL-061 excluded them at retrieval; this gates at ingestion.
    if resolved_ka == "pith_benchmarks":
        from app.core.config import BENCHMARK as _bm_gate
        if not _bm_gate.enabled:
            logger.info(
                "GATE-BENCHMARKS: Skipping benchmark concept %s (not in benchmark mode)",
                concept.id,
            )
            return False  # Don't write — benchmark data excluded at ingestion

    # FIX-M3: Enforce M3 confidence ceiling at write time.
    # STABILITY-026/027 only guard ingest. evolve_concept legacy fallback bypasses them.
    # nonlossy.py:242 has its own M3 check; this is defense-in-depth for ALL write paths.
    from app.core.config import PSIS_QUARANTINE_CONFIDENCE_CAP, PSIS_QUARANTINE_EVIDENCE_MARKER
    _concept_evidence = data.get("evidence", [])
    if isinstance(_concept_evidence, list) and PSIS_QUARANTINE_EVIDENCE_MARKER in _concept_evidence:
        if concept.confidence > PSIS_QUARANTINE_CONFIDENCE_CAP:
            logger.info(
                "FIX-M3: Clamping PSIS concept %s confidence %.3f → %.1f at write time",
                concept.id, concept.confidence, PSIS_QUARANTINE_CONFIDENCE_CAP,
            )
            concept.confidence = PSIS_QUARANTINE_CONFIDENCE_CAP
            data["confidence"] = PSIS_QUARANTINE_CONFIDENCE_CAP

    with _conn._db() as conn:
        # Check if concept already exists
        exists = conn.execute("SELECT 1 FROM concepts WHERE id = ?", (concept.id,)).fetchone()

        if exists:
            # UPDATE existing — preserves always_activate and other flag columns
            summary_changed, _ = _apply_summary_change_reset(conn, concept.id, data, concept.summary)
            # FIX-1 (EVOLUTION_CHAIN_BREAK): Prevent in-memory model from overwriting
            # DB superseded_by back to NULL. If DB has a non-NULL superseded_by but
            # the in-memory model has None (loaded before supersession), preserve DB value.
            _superseded_by_val = getattr(concept, "superseded_by", None)
            if _superseded_by_val is None:
                _db_superseded = conn.execute(
                    "SELECT superseded_by FROM concepts WHERE id = ?", (concept.id,)
                ).fetchone()
                if _db_superseded and _db_superseded[0] is not None:
                    _superseded_by_val = _db_superseded[0]
                    logger.info(
                        "FIX-1: Preserving DB superseded_by=%s for %s (in-memory was None)",
                        _superseded_by_val,
                        concept.id,
                    )

            # AGENT-004: Include session_id only if concept has one
            # (don't overwrite existing session_id with NULL on evolution)
            session_id_val = getattr(concept, "session_id", None)
            if session_id_val:
                conn.execute(
                    """
                    UPDATE concepts SET
                        version = ?, summary = ?, confidence = ?, stability = ?,
                        knowledge_area = ?, concept_type = ?, status = ?,
                        salience = ?, salience_source = ?, maturity = ?,
                        updated_at = ?, last_accessed = ?, access_count = ?,
                        session_id = ?,
                        authority_score = ?, effective_authority = ?,
                        currency_score = ?, currency_status = ?,
                        staleness_state = ?, staleness_score = ?, staleness_reason = ?,
                        staleness_evaluated_at = ?, staleness_detector_version = ?,
                        staleness_consecutive_hits = ?,
                        superseded_by = ?, epistemic_network = ?,
                        reinforcement_count = ?,
                        original_date = ?,
                        edit_provenance = ?,
                        subject_key = ?,
                        content_updated_at = COALESCE(?, content_updated_at),
                        data = ?
                    WHERE id = ?
                """,
                    (
                        concept.version,
                        concept.summary,
                        concept.confidence,
                        concept.stability,
                        resolved_ka,  # DEBT-185: use pre-resolved KA (synced to blob)
                        getattr(concept, "concept_type", "insight"),
                        concept.status,
                        getattr(concept, "salience", 0.5),
                        getattr(concept, "salience_source", "system"),
                        getattr(concept, "maturity", "ESTABLISHED"),
                        now,
                        getattr(concept, "last_accessed", None),
                        getattr(concept, "access_count", 0),
                        session_id_val,
                        _clamp_score(getattr(concept, "authority_score", None)),  # DEBT-182
                        _clamp_score(getattr(concept, "effective_authority", None)),  # DEBT-182
                        _clamp_score(getattr(concept, "currency_score", None)),  # DEBT-182
                        getattr(concept, "currency_status", None),
                        *_stale_risk_field_values(data),
                        _superseded_by_val,  # FIX-1: Use guarded value
                        getattr(concept, "epistemic_network", None),
                        getattr(concept, "reinforcement_count", None),
                        getattr(concept, "original_date", None),  # TEMPORAL-002
                        getattr(concept, "edit_provenance", None),  # RETRIEVAL-104
                        getattr(concept, "subject_key", None),  # EUNOMIA-040 Fix 3
                        content_updated_at if summary_changed else getattr(concept, "content_updated_at", None),
                        json.dumps(data),
                        concept.id,
                    ),
                )
            else:
                conn.execute(
                    """
                    UPDATE concepts SET
                        version = ?, summary = ?, confidence = ?, stability = ?,
                        knowledge_area = ?, concept_type = ?, status = ?,
                        salience = ?, salience_source = ?, maturity = ?,
                        updated_at = ?, last_accessed = ?, access_count = ?,
                        authority_score = ?, effective_authority = ?,
                        currency_score = ?, currency_status = ?,
                        staleness_state = ?, staleness_score = ?, staleness_reason = ?,
                        staleness_evaluated_at = ?, staleness_detector_version = ?,
                        staleness_consecutive_hits = ?,
                        superseded_by = ?, epistemic_network = ?,
                        reinforcement_count = ?,
                        original_date = ?,
                        edit_provenance = ?,
                        subject_key = ?,
                        content_updated_at = COALESCE(?, content_updated_at),
                        data = ?
                    WHERE id = ?
                """,
                    (
                        concept.version,
                        concept.summary,
                        concept.confidence,
                        concept.stability,
                        resolved_ka,  # DEBT-185: use pre-resolved KA (synced to blob)
                        getattr(concept, "concept_type", "insight"),
                        concept.status,
                        getattr(concept, "salience", 0.5),
                        getattr(concept, "salience_source", "system"),
                        getattr(concept, "maturity", "ESTABLISHED"),
                        now,
                        getattr(concept, "last_accessed", None),
                        getattr(concept, "access_count", 0),
                        _clamp_score(getattr(concept, "authority_score", None)),  # DEBT-182
                        _clamp_score(getattr(concept, "effective_authority", None)),  # DEBT-182
                        _clamp_score(getattr(concept, "currency_score", None)),  # DEBT-182
                        getattr(concept, "currency_status", None),
                        *_stale_risk_field_values(data),
                        _superseded_by_val,  # FIX-1: Use guarded value
                        getattr(concept, "epistemic_network", None),
                        getattr(concept, "reinforcement_count", None),
                        getattr(concept, "original_date", None),  # TEMPORAL-002
                        getattr(concept, "edit_provenance", None),  # RETRIEVAL-104
                        getattr(concept, "subject_key", None),  # EUNOMIA-040 Fix 3
                        content_updated_at if summary_changed else getattr(concept, "content_updated_at", None),
                        json.dumps(data),
                        concept.id,
                    ),
                )
        else:
            # INSERT new — column defaults (always_activate=0) apply correctly
            validated_aid = validate_agent_id(meta.get("agent_id", "default"))
            conn.execute(
                """
                INSERT INTO concepts
                (id, version, summary, confidence, stability, knowledge_area,
                 concept_type, status, salience, salience_source, maturity,
                 created_at, updated_at, last_accessed, access_count, agent_id,
                 session_id, authority_score, effective_authority,
                 currency_score, currency_status, staleness_state, staleness_score,
                 staleness_reason, staleness_evaluated_at, staleness_detector_version,
                 staleness_consecutive_hits, superseded_by,
                 epistemic_network, reinforcement_count, content_updated_at,
                 valid_from, original_date, edit_provenance, subject_key, data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    concept.id,
                    concept.version,
                    concept.summary,
                    concept.confidence,
                    concept.stability,
                    resolved_ka,  # DEBT-185: use pre-resolved KA (synced to blob)
                    getattr(concept, "concept_type", "insight"),
                    concept.status,
                    getattr(concept, "salience", 0.5),
                    getattr(concept, "salience_source", "system"),
                    getattr(concept, "maturity", "ESTABLISHED"),
                    concept.created_at,
                    now,
                    getattr(concept, "last_accessed", None),
                    getattr(concept, "access_count", 0),
                    validated_aid,  # 16: agent_id
                    getattr(concept, "session_id", None),  # 17: session_id
                    _clamp_score(getattr(concept, "authority_score", None)),  # 18: DEBT-137b + DEBT-182
                    _clamp_score(getattr(concept, "effective_authority", None)),  # 19: DEBT-137b + DEBT-182
                    _clamp_score(getattr(concept, "currency_score", None)),  # 20: DEBT-137b + DEBT-182
                    getattr(concept, "currency_status", None),  # 21: DEBT-137b
                    *_stale_risk_field_values(data),  # 22-27: COGGOV-014 stale-risk lifecycle
                    getattr(concept, "superseded_by", None),  # 28: DEBT-137b
                    getattr(concept, "epistemic_network", None),  # 29: DEBT-137b
                    getattr(concept, "reinforcement_count", None),  # 30: DEBT-137b
                    content_updated_at,  # 31: DATA-020 content_updated_at
                    concept.created_at,  # 32: INGEST-016 valid_from = created_at
                    getattr(concept, "original_date", None),  # 33: TEMPORAL-002
                    getattr(concept, "edit_provenance", None),  # 34: RETRIEVAL-104
                    getattr(concept, "subject_key", None),  # 35: EUNOMIA-040 Fix 3
                    json.dumps(data),  # 36: data (always last)
                ),
            )

        # RETRIEVAL-042 upgrade: Sync FTS5 full-text index
        _sync_fts5(conn, concept.id, concept.summary)

        # Insert version record (append-only history)
        conn.execute(
            """
            INSERT OR IGNORE INTO concept_versions (id, version, data, created_at)
            VALUES (?, ?, ?, ?)
        """,
            (concept.id, concept.version, json.dumps(data), concept.created_at),
        )


# --- Conn-aware helpers for atomic evolve (§5.2.2) ---


def load_concept_conn(conn, concept_id: str) -> Concept | None:
    """Load latest concept using an existing connection (no separate transaction).

    Used inside _conn._db_immediate() blocks for atomic read-modify-write cycles.
    Does NOT track access (internal operation).
    """
    row = conn.execute(
        """SELECT data, authority_score, currency_score, currency_status, staleness_state,
                  staleness_score, staleness_reason, staleness_evaluated_at,
                  staleness_detector_version, staleness_consecutive_hits, knowledge_area,
                  access_count, effective_authority, reinforcement_count, last_accessed,
                  last_organic_access, ka_relative_authority, status, superseded_by,
                  maturity, created_at, valid_from, content_updated_at, session_id,
                  original_date, protected
           FROM concepts WHERE id = ?""",
        (concept_id,),
    ).fetchone()
    if not row:
        return None
    data = _safe_json_loads(row["data"], context=f"load_concept_conn({concept_id})")
    if data is None:
        return None
    try:
        data["authority_score"] = row["authority_score"]
        data["currency_score"] = row["currency_score"]
        data["currency_status"] = row["currency_status"] or "ACTIVE"
        data["staleness_state"] = row["staleness_state"]
        data["staleness_score"] = row["staleness_score"]
        data["staleness_reason"] = row["staleness_reason"]
        data["staleness_evaluated_at"] = row["staleness_evaluated_at"]
        data["staleness_detector_version"] = row["staleness_detector_version"]
        data["staleness_consecutive_hits"] = row["staleness_consecutive_hits"] or 0
        data["access_count"] = row["access_count"] or 0
        data["effective_authority"] = row["effective_authority"]
        data["reinforcement_count"] = row["reinforcement_count"] or 0
        data["ka_relative_authority"] = row["ka_relative_authority"]
        # MATURITY-006: Inject maturity from DB column (canonical source).
        # Old concepts lack maturity in JSON blob, causing Pydantic to default to
        # ESTABLISHED — masking true PROVISIONAL state from the promotion sweep.
        data["maturity"] = row["maturity"] or "PROVISIONAL"
        # CURRENCY-001: Inject last_accessed from SQL to prevent desync.
        # Pre-RETRIEVAL-012, load_concept wrote last_accessed to SQL only (not JSON).
        if row["last_accessed"]:
            data["last_accessed"] = row["last_accessed"]
        # DATA-065: Inject last_organic_access from SQL column.
        if row["last_organic_access"]:
            data["last_organic_access"] = row["last_organic_access"]
        # DATA-018: Inject status from SQL column (not stored in JSON blob).
        data["status"] = row["status"] or "active"
        _hydrate_concept_sql_columns(data, row)
        # MAINT-030: Hydrate superseded_by from DB column
        # Eliminates FIX-1 per-save DB reads during maintenance
        _sup_by = row["superseded_by"]
        if _sup_by is not None:
            data["superseded_by"] = _sup_by
        # COGGOV-005: Hydrate protected flag from DB column
        try:
            data["protected"] = bool(row["protected"]) if row["protected"] is not None else False
        except (IndexError, KeyError):
            pass
    except (IndexError, KeyError):
        pass
    if "knowledge_area" not in data or data["knowledge_area"] is None:
        meta = data.get("metadata", {})
        if meta.get("knowledge_area"):
            data["knowledge_area"] = meta["knowledge_area"]
        else:
            try:
                if row["knowledge_area"]:
                    data["knowledge_area"] = row["knowledge_area"]
            except (IndexError, KeyError):
                pass
    # FIX-2(A3): Inject defaults for missing required Pydantic fields
    _required_defaults = {
        "id": concept_id,
        "version": "v1",
        "created_at": _utc_now_iso(),
        "summary": "",
        "confidence": 0.5,
    }
    for _field, _default in _required_defaults.items():
        if _field not in data or data[_field] is None:
            data[_field] = _default
    try:
        return Concept(**data)
    except Exception as e:
        logger.error("load_concept_conn(%s) Pydantic error after defaults: %s", concept_id, e)
        return None


def load_concept(concept_id: str, version: str = "latest", track_access: bool = True) -> Concept | None:
    """Load concept from SQLite.

    Args:
        concept_id: The concept identifier.
        version: 'latest' for current, 'all' for all versions, or specific 'v1', 'v2'.
        track_access: If True, increments access_count and updates last_accessed.
            Internal scans (reflection, decay) pass False to avoid inflating metrics.
    """
    if version == "all":
        return load_all_versions(concept_id)

    if BENCHMARK_READONLY:
        track_access = False

    db_context = _conn._db() if track_access else read_snapshot_db("load_concept_snapshot")
    with db_context as conn:
        if version == "latest":
            row = conn.execute(
                """SELECT data, authority_score, currency_score, currency_status, staleness_state,
                          staleness_score, staleness_reason, staleness_evaluated_at,
                          staleness_detector_version, staleness_consecutive_hits, knowledge_area,
                          access_count, effective_authority, reinforcement_count, last_accessed,
                          last_organic_access, ka_relative_authority, status, superseded_by,
                          maturity, created_at, valid_from, content_updated_at, session_id,
                          original_date, protected, utility_score, utility_samples, utility_updated
                   FROM concepts WHERE id = ?""",
                (concept_id,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT data FROM concept_versions WHERE id = ? AND version = ?", (concept_id, version)
            ).fetchone()

        if not row:
            return None

        # Direct DB access tracking (replaces AccessTracker batching)
        # IMPORTANT: This is the SOLE organic access write path.
        # last_organic_access is updated ONLY here — NOT in save_concept().
        # If adding a new caller with track_access=True, ensure it represents
        # genuine user-initiated retrieval, not batch/maintenance operations.
        #
        # RETRIEVAL-096 FIX: Suppress access tracking in benchmark readonly mode.
        # access_count/last_accessed/reinforcement_count writes were mutating
        # the brain DB ~30 times per question (once per retrieved concept),
        # causing progressive retrieval drift via currency scoring changes.
        # Canary detected 3.3% overlap (1/30) at Q25 with co-activation fix
        # alone — this was the remaining mutation path.
        if track_access:
            # RETRIEVAL-012 + CURRENCY-003: Update last_accessed ONLY on retrieval
            # activation (pith_conversation_turn), NOT on evolution (learning.py).
            # This is the sole path that refreshes last_accessed, keeping
            # access_recency_score meaningful for currency computation.
            # DATA-065: Also update last_organic_access for freshness discrimination.
            _now = _utc_now_iso()
            conn.execute(
                """
                UPDATE concepts SET access_count = access_count + 1,
                    reinforcement_count = reinforcement_count + 1,
                    last_accessed = ?,
                    last_organic_access = ?,
                    data = json_set(data,
                        '$.last_accessed', ?,
                        '$.last_organic_access', ?,
                        '$.access_count', access_count + 1,
                        '$.reinforcement_count', reinforcement_count + 1)
                WHERE id = ?
            """,
                (_now, _now, _now, _now, concept_id),
            )

        data = _safe_json_loads(row["data"], context=f"load_concept({concept_id}, {version})")
        if data is None:
            return None
        # Inject governance scores from DB columns (not stored in JSON blob)
        if version == "latest":
            try:
                data["authority_score"] = row["authority_score"]
                data["currency_score"] = row["currency_score"]
                data["currency_status"] = row["currency_status"] or "ACTIVE"
                data["staleness_state"] = row["staleness_state"]
                data["staleness_score"] = row["staleness_score"]
                data["staleness_reason"] = row["staleness_reason"]
                data["staleness_evaluated_at"] = row["staleness_evaluated_at"]
                data["staleness_detector_version"] = row["staleness_detector_version"]
                data["staleness_consecutive_hits"] = row["staleness_consecutive_hits"] or 0
                data["access_count"] = row["access_count"] or 0
                data["effective_authority"] = row["effective_authority"]
                data["reinforcement_count"] = row["reinforcement_count"] or 0
                data["ka_relative_authority"] = row["ka_relative_authority"]
                # CURRENCY-001: Inject last_accessed from SQL to prevent desync.
                if row["last_accessed"]:
                    data["last_accessed"] = row["last_accessed"]
                # DATA-065: Inject last_organic_access from SQL column.
                if row["last_organic_access"]:
                    data["last_organic_access"] = row["last_organic_access"]
                # DATA-018: Inject status from SQL column (not stored in JSON blob).
                data["status"] = row["status"] or "active"
                # MATURITY-006: Inject maturity from DB column (canonical source).
                # Old concepts lack maturity in JSON blob, causing Pydantic to default to
                # ESTABLISHED — masking true PROVISIONAL state from the promotion sweep.
                data["maturity"] = row["maturity"] or "PROVISIONAL"
                # MAINT-030: Hydrate superseded_by from DB column
                # Eliminates FIX-1 per-save DB reads during maintenance
                _sup_by = row["superseded_by"]
                if _sup_by is not None:
                    data["superseded_by"] = _sup_by
                _hydrate_concept_sql_columns(data, row)
                # COGGOV-005: Hydrate protected flag from DB column
                try:
                    data["protected"] = bool(row["protected"]) if row["protected"] is not None else False
                except (IndexError, KeyError):
                    pass
                # RETRIEVAL-080: Hydrate utility fields from DB columns
                try:
                    if row["utility_score"] is not None:
                        data["utility_score"] = row["utility_score"]
                    if row["utility_samples"] is not None:
                        data["utility_samples"] = row["utility_samples"]
                    if row["utility_updated"] is not None:
                        data["utility_updated"] = row["utility_updated"]
                except (IndexError, KeyError):
                    pass  # Utility columns not yet migrated
            except (IndexError, KeyError):
                pass  # Governance columns not yet migrated
        # Inject knowledge_area from DB column if not already in JSON blob
        if "knowledge_area" not in data or data["knowledge_area"] is None:
            # Try metadata dict first (where save_concept writes it from)
            meta = data.get("metadata", {})
            if meta.get("knowledge_area"):
                data["knowledge_area"] = meta["knowledge_area"]
            elif version == "latest":
                try:
                    if row["knowledge_area"]:
                        data["knowledge_area"] = row["knowledge_area"]
                except (IndexError, KeyError):
                    pass
        # FIX-2(A3): Inject defaults for missing required Pydantic fields
        # to prevent ValidationError crashes across all 40+ callers.
        _required_defaults = {
            "id": concept_id,
            "version": "v1",
            "created_at": _utc_now_iso(),
            "summary": "",
            "confidence": 0.5,
        }
        for _field, _default in _required_defaults.items():
            if _field not in data or data[_field] is None:
                data[_field] = _default
        try:
            return Concept(**data)
        except Exception as e:
            logger.error("load_concept(%s) Pydantic error after defaults: %s", concept_id, e)
            return None


def load_concepts_batch(concept_ids: list[str]) -> dict[str, "Concept"]:
    """Batch-load concepts by ID. Returns {concept_id: Concept} dict.

    PERF-076: Replaces N individual load_concept() calls with a single
    WHERE id IN (...) query. Used by search_lightweight where track_access=False.

    Does NOT update access_count/last_accessed (equivalent to track_access=False).
    """
    if not concept_ids:
        return {}

    from app.core.models import Concept as ConceptModel

    result: dict[str, "Concept"] = {}
    # SQLite parameter limit is 999; chunk if needed
    _CHUNK = 900
    for chunk_start in range(0, len(concept_ids), _CHUNK):
        chunk = concept_ids[chunk_start : chunk_start + _CHUNK]
        placeholders = ",".join("?" * len(chunk))
        with read_snapshot_db("load_concepts_batch") as conn:
            rows = conn.execute(
                f"""SELECT id, data, authority_score, currency_score, currency_status,
                           staleness_state, staleness_score, staleness_reason,
                           staleness_evaluated_at, staleness_detector_version,
                           staleness_consecutive_hits, knowledge_area, access_count,
                           effective_authority, reinforcement_count, last_accessed,
                           last_organic_access, ka_relative_authority, status,
                           superseded_by, maturity, protected,
                           created_at, valid_from, content_updated_at, session_id,
                           original_date, utility_score, utility_samples, utility_updated
                    FROM concepts WHERE id IN ({placeholders})""",
                chunk,
            ).fetchall()

        for row in rows:
            data = _safe_json_loads(row["data"], context=f"load_concepts_batch({row['id']})")
            if data is None:
                continue
            # Inject governance scores from DB columns (same as load_concept)
            try:
                data["authority_score"] = row["authority_score"]
                data["currency_score"] = row["currency_score"]
                data["currency_status"] = row["currency_status"] or "ACTIVE"
                data["staleness_state"] = row["staleness_state"]
                data["staleness_score"] = row["staleness_score"]
                data["staleness_reason"] = row["staleness_reason"]
                data["staleness_evaluated_at"] = row["staleness_evaluated_at"]
                data["staleness_detector_version"] = row["staleness_detector_version"]
                data["staleness_consecutive_hits"] = row["staleness_consecutive_hits"] or 0
                data["access_count"] = row["access_count"] or 0
                data["effective_authority"] = row["effective_authority"]
                data["reinforcement_count"] = row["reinforcement_count"] or 0
                data["ka_relative_authority"] = row["ka_relative_authority"]
                if row["last_accessed"]:
                    data["last_accessed"] = row["last_accessed"]
                if row["last_organic_access"]:
                    data["last_organic_access"] = row["last_organic_access"]
                data["status"] = row["status"] or "active"
                data["maturity"] = row["maturity"] or "PROVISIONAL"
                _sup_by = row["superseded_by"]
                if _sup_by is not None:
                    data["superseded_by"] = _sup_by
                _hydrate_concept_sql_columns(data, row)
                try:
                    data["protected"] = bool(row["protected"]) if row["protected"] is not None else False
                except (IndexError, KeyError):
                    pass
                try:
                    if row["utility_score"] is not None:
                        data["utility_score"] = row["utility_score"]
                    if row["utility_samples"] is not None:
                        data["utility_samples"] = row["utility_samples"]
                    if row["utility_updated"] is not None:
                        data["utility_updated"] = row["utility_updated"]
                except (IndexError, KeyError):
                    pass
            except (IndexError, KeyError):
                pass

            # Inject knowledge_area from DB column
            if "knowledge_area" not in data or data["knowledge_area"] is None:
                meta = data.get("metadata", {})
                if meta.get("knowledge_area"):
                    data["knowledge_area"] = meta["knowledge_area"]
                elif row["knowledge_area"]:
                    data["knowledge_area"] = row["knowledge_area"]

            # GAUNTLET A1: Apply required defaults (same as load_concept lines 954-960)
            _required_defaults = {
                "id": row["id"],
                "version": "v1",
                "created_at": _utc_now_iso(),
                "summary": "",
                "confidence": 0.5,
            }
            for _field, _default in _required_defaults.items():
                if _field not in data or data[_field] is None:
                    data[_field] = _default

            try:
                concept = ConceptModel(**data)
                result[row["id"]] = concept
            except Exception as e:
                logger.warning("load_concepts_batch: Failed to hydrate %s: %s", row["id"], e)

    return result


def load_all_versions(concept_id: str) -> list[Concept]:
    """Load all versions of a concept, ordered by version."""
    with read_snapshot_db("load_all_versions") as conn:
        rows = conn.execute("SELECT data FROM concept_versions WHERE id = ? ORDER BY version", (concept_id,)).fetchall()
    results = []
    for r in rows:
        d = _safe_json_loads(r["data"], context=f"load_concept_versions({concept_id})")
        if d is not None:
            results.append(Concept(**d))
    return results


def list_concepts() -> list[str]:
    """List all active, current concept IDs.

    CURRENCY-008: Added is_current=1 filter. Previously returned 5894 concepts
    including 3530 superseded versions (is_current=0) that polluted health score
    aggregation, recalibration, and all reflection operations.
    """
    with read_snapshot_db("list_concepts") as conn:
        rows = conn.execute("SELECT id FROM concepts WHERE status = 'active' AND is_current = 1 ORDER BY id").fetchall()
    return [r["id"] for r in rows]


def list_concepts_modified_since(cutoff_iso: str) -> list[str]:
    """List concept IDs modified or created since a cutoff timestamp.

    REFLECT-021: Used by _merge_duplicates() to narrow scan to recently-changed
    concepts instead of iterating the full population. Falls back to full scan
    if cutoff is None.

    Args:
        cutoff_iso: ISO timestamp string. Returns concepts where
            content_updated_at > cutoff OR created_at > cutoff.
    """
    with read_snapshot_db("list_concepts_modified_since") as conn:
        rows = conn.execute(
            """SELECT id FROM concepts
               WHERE status = 'active' AND is_current = 1
               AND (content_updated_at > ? OR created_at > ?)
               ORDER BY id""",
            (cutoff_iso, cutoff_iso),
        ).fetchall()
    return [r["id"] for r in rows]


def list_concepts_full(conn: sqlite3.Connection | None = None) -> list[Concept]:
    """List all active concepts as full Concept objects in a single query.

    Used by session_start and other operations needing full models.
    Does NOT increment access_count (bulk read, not individual access).
    """
    context = nullcontext(conn) if conn is not None else read_snapshot_db("list_concepts_full")
    with context as conn:
        rows = conn.execute(
            """SELECT data, authority_score, currency_score, currency_status,
                      utility_score, utility_samples, utility_updated
               FROM concepts WHERE status = 'active' AND is_current = 1 ORDER BY id"""
        ).fetchall()
    concepts = []
    for r in rows:
        try:
            data = _safe_json_loads(r["data"], context="list_concepts_full")
            if data is None:
                continue
            # Inject governance scores from DB columns
            try:
                data["authority_score"] = r["authority_score"]
                data["currency_score"] = r["currency_score"]
                data["currency_status"] = r["currency_status"] or "ACTIVE"
            except (IndexError, KeyError):
                pass
            # RETRIEVAL-080: Inject utility fields from DB columns
            try:
                if r["utility_score"] is not None:
                    data["utility_score"] = r["utility_score"]
                if r["utility_samples"] is not None:
                    data["utility_samples"] = r["utility_samples"]
                if r["utility_updated"] is not None:
                    data["utility_updated"] = r["utility_updated"]
            except (IndexError, KeyError):
                pass  # Utility columns not yet migrated
            concepts.append(Concept(**data))
        except Exception as e:
            logger.warning(f"list_concepts_full: failed to parse concept: {e}")
    return concepts


def list_concepts_for_indexing() -> list[dict]:
    """Lightweight bulk query for index building — returns raw dicts, not Concept models.

    Skips Pydantic construction (4ms/concept overhead) by returning parsed JSON
    dicts. Used exclusively by build_index where full model validation is unnecessary.
    """
    with read_snapshot_db("list_concepts_for_indexing") as conn:
        rows = conn.execute(
            "SELECT id, data FROM concepts WHERE status = 'active' AND is_current = 1 ORDER BY id"
        ).fetchall()
    results = []
    for r in rows:
        try:
            data = _safe_json_loads(r["data"], context=f"list_concepts_for_indexing({r['id']})")
            if data is None:
                continue
            data["_id"] = r["id"]
            results.append(data)
        except Exception as e:
            logger.warning(f"list_concepts_for_indexing: failed to parse: {e}")
    return results


def get_next_version(concept_id: str) -> str:
    """Get next version number for a concept."""
    with _conn._db() as conn:
        row = conn.execute(
            "SELECT version FROM concept_versions WHERE id = ? ORDER BY version DESC LIMIT 1", (concept_id,)
        ).fetchone()
    if not row:
        return "v1"
    # Extract number from "v3" -> 3, return "v4"
    try:
        num = int(row["version"][1:])
        return f"v{num + 1}"
    except (ValueError, IndexError):
        return "v1"

def _invalidate_associations_cache() -> None:
    """Invalidate the associations cache. Call after any write to associations table."""
    global _associations_cache, _associations_cache_ts, _adjacency_graph_cache, _adjacency_graph_cache_ts
    global _association_indexes_cache, _association_indexes_cache_ts
    with _associations_cache_lock:
        _associations_cache = None
        _associations_cache_ts = 0.0
        _adjacency_graph_cache = None  # PERF-023
        _adjacency_graph_cache_ts = 0.0  # PERF-023
        _association_indexes_cache = None  # PERF-080
        _association_indexes_cache_ts = 0.0  # PERF-080


def load_associations() -> dict:
    """Load association graph with module-level TTL cache (PERF-016).

    Returns dict with 'associations' list and 'metadata'.
    Cache is invalidated on writes via _invalidate_associations_cache()
    and expires after _ASSOCIATIONS_CACHE_TTL_S seconds.
    """
    global _associations_cache, _associations_cache_ts, _assoc_cache_hits, _assoc_cache_misses

    now = _time_mod.monotonic()
    with _associations_cache_lock:
        if _associations_cache is not None and (now - _associations_cache_ts) < _ASSOCIATIONS_CACHE_TTL_S:
            _assoc_cache_hits += 1
            return _associations_cache
        _assoc_cache_misses += 1

    with read_snapshot_db("load_associations") as conn:
        rows = conn.execute("SELECT source, target, relation, strength, created_at FROM associations").fetchall()

    edges = [
        {
            "source": r["source"],
            "target": r["target"],
            "relation": r["relation"],
            "strength": r["strength"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]

    result = {"associations": edges, "metadata": {"storage": "sqlite"}}
    with _associations_cache_lock:
        _associations_cache = result
        _associations_cache_ts = now
    return result


def get_adjacency_graph() -> dict[str, dict[str, float]]:
    """Return adjacency dict mapping concept_id → {neighbor_id: strength, ...} (PERF-023).

    Builds from load_associations() and caches independently with the same
    60-second TTL. Invalidated alongside the association cache via
    _invalidate_associations_cache() on any write.

    Callers (get_related_concepts, _spread_activation) pay <0.01ms on cache hit
    instead of 37ms for a full DB scan + adjacency rebuild.

    ARCH-O02: Changed from list[str] to dict[str, float] to preserve per-edge
    strength. Callers iterating neighbors get dict keys (same IDs as before).
    Callers needing strength use graph[src][tgt].
    """
    global _adjacency_graph_cache, _adjacency_graph_cache_ts, _adjacency_cache_hits, _adjacency_cache_misses

    now = _time_mod.monotonic()
    with _associations_cache_lock:
        if _adjacency_graph_cache is not None and (now - _adjacency_graph_cache_ts) < _ASSOCIATIONS_CACHE_TTL_S:
            _adjacency_cache_hits += 1
            return _adjacency_graph_cache
        _adjacency_cache_misses += 1

    assoc_data = load_associations()
    graph: dict[str, dict[str, float]] = {}
    for edge in assoc_data["associations"]:
        src, tgt = edge["source"], edge["target"]
        strength = edge.get("strength", 0.5)
        graph.setdefault(src, {})[tgt] = strength
        graph.setdefault(tgt, {})[src] = strength

    with _associations_cache_lock:
        _adjacency_graph_cache = graph
        _adjacency_graph_cache_ts = now
    return graph


def load_association_indexes() -> AssociationIndexes:
    """Return derived association indexes with shared TTL/invalidation (PERF-080)."""
    global _association_indexes_cache, _association_indexes_cache_ts
    global _assoc_index_cache_hits, _assoc_index_cache_misses

    now = _time_mod.monotonic()
    with _associations_cache_lock:
        if (
            _association_indexes_cache is not None
            and (now - _association_indexes_cache_ts) < _ASSOCIATIONS_CACHE_TTL_S
        ):
            _assoc_index_cache_hits += 1
            return _association_indexes_cache
        _assoc_index_cache_misses += 1

    assoc_data = load_associations()
    edges = assoc_data.get("associations", [])
    adjacency: dict[str, list[str]] = {}
    edge_relations: dict[tuple[str, str], str] = {}
    edge_strength: dict[tuple[str, str], float] = {}
    contradiction_signals: list[tuple[str, str]] = []

    for edge in edges:
        src, tgt = edge["source"], edge["target"]
        relation = edge.get("relation", "related_to")
        if relation == "contradicts":
            contradiction_signals.append((src, tgt))
            continue

        raw_strength = edge.get("strength")
        base_strength = raw_strength if raw_strength is not None else 0.5
        effective_strength = base_strength * _ASSOCIATION_EDGE_TYPE_MULTIPLIER.get(relation, 0.7)

        adjacency.setdefault(src, []).append(tgt)
        adjacency.setdefault(tgt, []).append(src)
        edge_relations[(src, tgt)] = relation
        edge_relations[(tgt, src)] = relation
        edge_strength[(src, tgt)] = effective_strength
        edge_strength[(tgt, src)] = effective_strength

    indexes = AssociationIndexes(
        edges=edges,
        adjacency=adjacency,
        edge_relations=edge_relations,
        edge_strength=edge_strength,
        contradiction_signals=contradiction_signals,
    )
    with _associations_cache_lock:
        _association_indexes_cache = indexes
        _association_indexes_cache_ts = now
    return indexes


def _on_association_index_refresh_done(future: Future) -> None:
    """Clear the single-flight background association-index refresh handle."""
    global _association_index_refresh_future
    try:
        exc = None if future.cancelled() else future.exception()
        if exc is not None:
            logger.warning("Association index background refresh failed: %s", exc)
    except Exception as exc:
        logger.warning("Association index background refresh status failed: %s", exc)
    finally:
        with _associations_cache_lock:
            if _association_index_refresh_future is future:
                _association_index_refresh_future = None


def _schedule_association_index_refresh_locked() -> bool:
    """Schedule one background association-index rebuild while lock is held."""
    global _association_index_refresh_executor, _association_index_refresh_future
    if (
        _association_index_refresh_future is not None
        and not _association_index_refresh_future.done()
    ):
        return False
    if _association_index_refresh_executor is None:
        _association_index_refresh_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="pith-association-index-refresh",
        )
    future = _association_index_refresh_executor.submit(load_association_indexes)
    _association_index_refresh_future = future
    future.add_done_callback(_on_association_index_refresh_done)
    return True


def shutdown_association_index_refresh(*, wait: bool = True) -> None:
    """Stop background association-index refresh work during server shutdown."""
    global _association_index_refresh_executor, _association_index_refresh_future
    with _associations_cache_lock:
        executor = _association_index_refresh_executor
        _association_index_refresh_executor = None
        _association_index_refresh_future = None
    if executor is not None:
        executor.shutdown(wait=wait, cancel_futures=True)


def load_association_indexes_budgeted(
    *,
    allow_foreground_rebuild: bool,
    prefer_stale: bool = True,
    max_stale_ms: float = 300000.0,
    background_refresh: bool = True,
) -> AssociationIndexLoadResult:
    """Return association indexes without forcing a foreground rebuild.

    The legacy load_association_indexes() API remains the strict path. This
    budgeted variant lets deadline-enabled request paths use fresh or bounded
    stale graph indexes and kick a single background refresh when rebuilding
    would risk blowing the caller's turn budget.
    """
    now = _time_mod.monotonic()
    with _associations_cache_lock:
        indexes = _association_indexes_cache
        cache_ts = _association_indexes_cache_ts
        age_ms = ((now - cache_ts) * 1000.0) if indexes is not None and cache_ts else None
        refresh_in_flight = (
            _association_index_refresh_future is not None
            and not _association_index_refresh_future.done()
        )
        if indexes is not None and age_ms is not None:
            if age_ms <= (_ASSOCIATIONS_CACHE_TTL_S * 1000.0):
                return AssociationIndexLoadResult(
                    indexes=indexes,
                    state="fresh_hit",
                    age_ms=round(age_ms, 2),
                    refresh_in_flight=refresh_in_flight,
                )
            if prefer_stale and age_ms <= max_stale_ms:
                scheduled = (
                    _schedule_association_index_refresh_locked()
                    if background_refresh
                    else False
                )
                return AssociationIndexLoadResult(
                    indexes=indexes,
                    state="stale_hit",
                    age_ms=round(age_ms, 2),
                    refresh_scheduled=scheduled,
                    refresh_in_flight=refresh_in_flight and not scheduled,
                )

        if not allow_foreground_rebuild:
            scheduled = (
                _schedule_association_index_refresh_locked()
                if background_refresh
                else False
            )
            return AssociationIndexLoadResult(
                indexes=None,
                state="miss",
                age_ms=round(age_ms, 2) if age_ms is not None else None,
                refresh_scheduled=scheduled,
                refresh_in_flight=refresh_in_flight and not scheduled,
            )

    return AssociationIndexLoadResult(
        indexes=load_association_indexes(),
        state="foreground_rebuild",
        age_ms=0.0,
    )


def count_associations(conn: sqlite3.Connection | None = None) -> int:
    """Count total association edges. Internal utility — not exposed via API."""
    context = nullcontext(conn) if conn is not None else read_snapshot_db("count_associations")
    with context as conn:
        row = conn.execute("SELECT COUNT(*) as cnt FROM associations").fetchone()
    return row["cnt"] if row else 0


def count_orphan_concepts() -> int:
    """Count active concepts with no association edges. Internal — used in pith_stats()."""
    with read_snapshot_db("count_orphan_concepts") as conn:
        row = conn.execute("""
            SELECT COUNT(*) as cnt FROM concepts
            WHERE status = 'active'
            AND id NOT IN (
                SELECT source FROM associations
                UNION
                SELECT target FROM associations
            )
        """).fetchone()
    return row["cnt"] if row else 0


def add_association(concept_a: str, concept_b: str, relation: str, strength: float = 0.5) -> None:
    """Add a single association edge (idempotent).

    Direction is normalized: source < target alphabetically. This ensures
    that A→B and B→A produce the same row, preventing duplicate edges
    regardless of argument order.
    """
    source, target = sorted([concept_a, concept_b])
    with _conn._db() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO associations (source, target, relation, strength, created_at) VALUES (?, ?, ?, ?, ?)",
            (source, target, relation, strength, _utc_now_iso()),
        )
    _invalidate_associations_cache()  # PERF-016


def add_associations_bulk(
    edges: list[tuple[str, str, str, float]],
    *,
    invalidate_cache: bool = True,
) -> int:
    """Add association edges in one transaction, invalidating caches at most once."""
    normalized: list[tuple[str, str, str, float]] = []
    seen: set[tuple[str, str, str]] = set()
    for concept_a, concept_b, relation, strength in edges:
        if not concept_a or not concept_b or not relation:
            continue
        source, target = sorted([concept_a, concept_b])
        key = (source, target, relation)
        if key in seen:
            continue
        seen.add(key)
        normalized.append((source, target, relation, strength))

    if not normalized:
        return 0

    inserted = 0
    created_at = _utc_now_iso()
    with _conn._db() as conn:
        for source, target, relation, strength in normalized:
            cur = conn.execute(
                "INSERT OR IGNORE INTO associations (source, target, relation, strength, created_at) VALUES (?, ?, ?, ?, ?)",
                (source, target, relation, strength, created_at),
            )
            if cur.rowcount and cur.rowcount > 0:
                inserted += cur.rowcount
    if inserted > 0 and invalidate_cache:
        _invalidate_associations_cache()
    return inserted


# --- INGEST-044: Typed association functions ---

_TYPED_ASSOC_DEFAULT_STRENGTH = 0.8

# INGEST-053: Maximum edges per entity. Prevents hub explosion from
# regex artifacts that survive the blocklist in triple_extractor.py.
_MAX_EDGES_PER_ENTITY = 20


def add_typed_association(
    source_entity: str,
    target_entity: str,
    relation: str,
    concept_id: str,
    strength: float = _TYPED_ASSOC_DEFAULT_STRENGTH,
) -> bool:
    """INGEST-044: Add a directional typed association (subject→predicate→object).

    Unlike add_association(), this preserves source→target ordering (no alphabetical sort).
    Uses direction='forward' to distinguish from bidirectional related_to edges.
    Uses chain_id to store the source concept_id for provenance.

    Returns True if inserted, False if duplicate (INSERT OR IGNORE).
    """
    source_lower = source_entity.lower()
    target_lower = target_entity.lower()

    with _conn._db() as conn:
        # INGEST-053: Per-entity cap — reject if source already has too many edges
        source_count = conn.execute(
            "SELECT COUNT(*) FROM associations "
            "WHERE source = ? AND direction = 'forward'",
            (source_lower,),
        ).fetchone()[0]
        if source_count >= _MAX_EDGES_PER_ENTITY:
            logger.warning(
                "INGEST-053: Rejected edge %s→%s: source entity has %d edges (cap=%d)",
                source_lower, target_lower, source_count, _MAX_EDGES_PER_ENTITY,
            )
            return False

        cursor = conn.execute(
            "INSERT OR IGNORE INTO associations "
            "(source, target, relation, strength, created_at, mechanism, direction, chain_id) "
            "VALUES (?, ?, ?, ?, ?, 'triple_extraction', 'forward', ?)",
            (source_lower, target_lower, relation, strength,
             _utc_now_iso(), concept_id),
        )
    _invalidate_associations_cache()
    return cursor.rowcount > 0


def get_typed_associations(
    entity: str,
    relation: Optional[str] = None,
    direction: str = 'outgoing',
) -> list[dict]:
    """INGEST-044: Query typed associations for an entity.

    Args:
        entity: Entity name to search for (case-insensitive via pre-lowered storage)
        relation: Optional predicate type filter (e.g., 'birthplace', 'performer')
        direction: 'outgoing' (entity is source), 'incoming' (entity is target), 'both'

    Returns list of dicts with keys: source, target, relation, strength, concept_id
    """
    entity_lower = entity.lower()
    with read_snapshot_db("get_typed_associations") as conn:
        if direction == 'outgoing':
            query = ("SELECT source, target, relation, strength, chain_id as concept_id "
                     "FROM associations WHERE source = ? AND direction = 'forward'")
            params: list = [entity_lower]
        elif direction == 'incoming':
            query = ("SELECT source, target, relation, strength, chain_id as concept_id "
                     "FROM associations WHERE target = ? AND direction = 'forward'")
            params = [entity_lower]
        else:  # 'both'
            query = ("SELECT source, target, relation, strength, chain_id as concept_id "
                     "FROM associations WHERE (source = ? OR target = ?) "
                     "AND direction = 'forward'")
            params = [entity_lower, entity_lower]

        if relation:
            query += " AND relation = ?"
            params.append(relation)

        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def get_typed_association_count() -> int:
    """INGEST-044: Count typed associations (for monitoring)."""
    with read_snapshot_db("get_typed_association_count") as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM associations WHERE direction = 'forward'"
        ).fetchone()
    return row[0] if row else 0


# --- End INGEST-044 ---


def get_all_association_triples() -> set:
    """Return all existing association edges as a set of (source, target, relation) tuples.

    Used by the auto-association pipeline for efficient O(1) duplicate checking
    before bulk edge insertion.
    """
    with read_snapshot_db("get_all_association_triples") as conn:
        rows = conn.execute("SELECT source, target, relation FROM associations").fetchall()
    return {(r["source"], r["target"], r["relation"]) for r in rows}


def get_association_triples_for_pairs(
    pairs: Iterable[tuple[str, str, str]],
    *,
    chunk_size: int = 100,
) -> set[tuple[str, str, str]]:
    """Return existing association triples for a bounded set of candidate pairs."""
    normalized: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for concept_a, concept_b, relation in pairs:
        if not concept_a or not concept_b or not relation:
            continue
        source, target = sorted([concept_a, concept_b])
        triple = (source, target, relation)
        if triple in seen:
            continue
        seen.add(triple)
        normalized.append(triple)

    if not normalized:
        return set()

    existing: set[tuple[str, str, str]] = set()
    safe_chunk_size = max(1, chunk_size)
    with read_snapshot_db("get_association_triples_for_pairs") as conn:
        for idx in range(0, len(normalized), safe_chunk_size):
            chunk = normalized[idx: idx + safe_chunk_size]
            predicates = " OR ".join(["(source = ? AND target = ? AND relation = ?)"] * len(chunk))
            params = [value for triple in chunk for value in triple]
            rows = conn.execute(
                f"SELECT source, target, relation FROM associations WHERE {predicates}",
                params,
            ).fetchall()
            existing.update((r["source"], r["target"], r["relation"]) for r in rows)
    return existing


def get_knowledge_area_map() -> dict:
    """Return a dict of concept_id → knowledge_area for all active concepts.

    Lightweight query for the auto-association pipeline's Tier 2 domain matching.
    """
    with read_snapshot_db("get_knowledge_area_map") as conn:
        rows = conn.execute(
            "SELECT id, knowledge_area FROM concepts WHERE status = 'active' AND is_current = 1"
        ).fetchall()
    return {r["id"]: r["knowledge_area"] for r in rows}


def get_related_concepts(concept_id: str, max_depth: int = 2) -> list[str]:
    """Get concepts related to a given concept via BFS edge traversal (PERF-023).

    Uses get_adjacency_graph() — cached, <0.01ms on hit vs 37ms DB scan.
    BFS replaces recursive DFS: cleaner, avoids stack pressure on large graphs.
    """
    graph = get_adjacency_graph()
    if concept_id not in graph:
        return []

    related: set[str] = set()
    frontier: set[str] = {concept_id}

    for _ in range(max_depth):
        next_frontier: set[str] = set()
        for cid in frontier:
            for neighbor in graph.get(cid, {}):  # ARCH-O02: dict default
                if neighbor not in related and neighbor != concept_id:
                    related.add(neighbor)
                    next_frontier.add(neighbor)
        frontier = next_frontier
        if not frontier:
            break

    return list(related)


# --- Archive / Restore (status-based soft delete) ---


def archive_concept(concept_id: str, on_archived: "callable | None" = None) -> bool:
    """Archive a concept by setting status to 'archived'.

    Amendment 6: Also cleans up orphaned edges pointing to/from the archived concept.

    Args:
        on_archived: Optional callback(concept_id) invoked after successful archive.
            Used by callers to notify retrieval index, breaking the storage→retrieval
            circular dependency (Item 2b).
    """
    with _conn._db() as conn:
        row = conn.execute("SELECT id FROM concepts WHERE id = ? AND status = 'active'", (concept_id,)).fetchone()
        if not row:
            logger.warning(f"Cannot archive {concept_id}: not found or already archived")
            return False

        changed = apply_lifecycle_transition_conn(
            conn,
            concept_id,
            "archive",
            on_archived=on_archived,
        )
        if changed:
            logger.info(f"Archived concept: {concept_id}")
        return changed > 0


def restore_concept(concept_id: str) -> bool:
    """Restore a concept from archived back to active.

    DATA-028: _restore_concept_lock prevents concurrent restore races where two
    callers could both pass the archived-check before either commits the UPDATE.
    """
    with _restore_concept_lock:
        with _conn._db() as conn:
            row = conn.execute(
                "SELECT id FROM concepts WHERE id = ? AND status = 'archived'", (concept_id,)
            ).fetchone()
            if not row:
                logger.warning(f"Cannot restore {concept_id}: not in archive")
                return False

            changed = apply_lifecycle_transition_conn(
                conn,
                concept_id,
                "restore_archive",
            )
            if changed:
                logger.info(f"Restored concept: {concept_id}")
            return changed > 0


def list_archived_concepts() -> list[str]:
    """List all archived concept IDs."""
    with read_snapshot_db("list_archived_concepts") as conn:
        rows = conn.execute("SELECT id FROM concepts WHERE status = 'archived' ORDER BY id").fetchall()
    return [r["id"] for r in rows]


# --- SelfModel Persistence ---


def save_self_model(model_data: dict) -> Path:
    """Save SelfModel to SQLite. Returns a Path for compat (unused by callers)."""
    version = model_data.get("version", 1)
    now = _utc_now_iso()
    data_json = json.dumps(model_data)

    with _conn._db() as conn:
        # Upsert current
        conn.execute(
            "INSERT OR REPLACE INTO self_model (id, version, data, updated_at) VALUES (?, ?, ?, ?)",
            ("current", version, data_json, now),
        )
        # Append to version history
        conn.execute(
            "INSERT OR IGNORE INTO self_model_versions (version, data, created_at) VALUES (?, ?, ?)",
            (version, data_json, model_data.get("generated_at", now)),
        )

    logger.info(f"SelfModel saved: v{version}")
    # Return a Path for backward compat (no callers use this)
    return DB_PATH


def load_self_model() -> dict | None:
    """Load current SelfModel from SQLite."""
    with read_snapshot_db("load_self_model") as conn:
        row = conn.execute("SELECT data FROM self_model WHERE id = 'current'").fetchone()
    if not row:
        return None
    return _safe_json_loads(row["data"], context="load_self_model")

# --- Conn-aware helpers for atomic evolve ---
def load_concept_conn(conn, concept_id: str) -> Concept | None:
    """Load latest concept using an existing connection (no separate transaction).

    Used inside _conn._db_immediate() blocks for atomic read-modify-write cycles.
    Does NOT track access (internal operation).
    """
    row = conn.execute(
        """SELECT data, authority_score, currency_score, currency_status, knowledge_area, access_count, effective_authority, reinforcement_count, last_accessed, last_organic_access, ka_relative_authority, status, superseded_by, maturity, created_at, valid_from, content_updated_at, session_id, original_date, protected
           FROM concepts WHERE id = ?""",
        (concept_id,),
    ).fetchone()
    if not row:
        return None
    data = _safe_json_loads(row["data"], context=f"load_concept_conn({concept_id})")
    if data is None:
        return None
    try:
        data["authority_score"] = row["authority_score"]
        data["currency_score"] = row["currency_score"]
        data["currency_status"] = row["currency_status"] or "ACTIVE"
        data["access_count"] = row["access_count"] or 0
        data["effective_authority"] = row["effective_authority"]
        data["reinforcement_count"] = row["reinforcement_count"] or 0
        data["ka_relative_authority"] = row["ka_relative_authority"]
        # MATURITY-006: Inject maturity from DB column (canonical source).
        # Old concepts lack maturity in JSON blob, causing Pydantic to default to
        # ESTABLISHED — masking true PROVISIONAL state from the promotion sweep.
        data["maturity"] = row["maturity"] or "PROVISIONAL"
        # CURRENCY-001: Inject last_accessed from SQL to prevent desync.
        # Pre-RETRIEVAL-012, load_concept wrote last_accessed to SQL only (not JSON).
        if row["last_accessed"]:
            data["last_accessed"] = row["last_accessed"]
        # DATA-065: Inject last_organic_access from SQL column.
        if row["last_organic_access"]:
            data["last_organic_access"] = row["last_organic_access"]
        # DATA-018: Inject status from SQL column (not stored in JSON blob).
        data["status"] = row["status"] or "active"
        _hydrate_concept_sql_columns(data, row)
        # MAINT-030: Hydrate superseded_by from DB column
        # Eliminates FIX-1 per-save DB reads during maintenance
        _sup_by = row["superseded_by"]
        if _sup_by is not None:
            data["superseded_by"] = _sup_by
        # COGGOV-005: Hydrate protected flag from DB column
        try:
            data["protected"] = bool(row["protected"]) if row["protected"] is not None else False
        except (IndexError, KeyError):
            pass
    except (IndexError, KeyError):
        pass
    if "knowledge_area" not in data or data["knowledge_area"] is None:
        meta = data.get("metadata", {})
        if meta.get("knowledge_area"):
            data["knowledge_area"] = meta["knowledge_area"]
        else:
            try:
                if row["knowledge_area"]:
                    data["knowledge_area"] = row["knowledge_area"]
            except (IndexError, KeyError):
                pass
    # FIX-2(A3): Inject defaults for missing required Pydantic fields
    _required_defaults = {
        "id": concept_id,
        "version": "v1",
        "created_at": _utc_now_iso(),
        "summary": "",
        "confidence": 0.5,
    }
    for _field, _default in _required_defaults.items():
        if _field not in data or data[_field] is None:
            data[_field] = _default
    try:
        return Concept(**data)
    except Exception as e:
        logger.error("load_concept_conn(%s) Pydantic error after defaults: %s", concept_id, e)
        return None


def get_next_version_conn(conn, concept_id: str) -> str:
    """Get next version number using an existing connection."""
    row = conn.execute(
        "SELECT version FROM concept_versions WHERE id = ? ORDER BY version DESC LIMIT 1", (concept_id,)
    ).fetchone()
    if not row:
        return "v1"
    try:
        num = int(row["version"][1:])
        return f"v{num + 1}"
    except (ValueError, IndexError):
        return "v1"


def save_concept_conn(conn, concept: "Concept") -> None:
    """Save concept using an existing connection (no separate transaction).

    Same logic as save_concept() but operates on a provided connection
    for use within _conn._db_immediate() atomic blocks.
    """
    data = concept.model_dump()
    meta = data.get("metadata", {})
    now = _utc_now_iso()

    # DEBT-185: Resolve KA once, sync to blob before serialization.
    resolved_ka = _resolve_knowledge_area(concept, meta)
    data["knowledge_area"] = resolved_ka

    exists = conn.execute("SELECT 1 FROM concepts WHERE id = ?", (concept.id,)).fetchone()

    if exists:
        summary_changed, _ = _apply_summary_change_reset(conn, concept.id, data, concept.summary)
        # AGENT-004: Include session_id only if concept has one
        # (don't overwrite existing session_id with NULL on evolution)
        session_id_val = getattr(concept, "session_id", None)
        if session_id_val:
            conn.execute(
                """
                UPDATE concepts SET
                    version = ?, summary = ?, confidence = ?, stability = ?,
                    knowledge_area = ?, concept_type = ?, status = ?,
                    salience = ?, salience_source = ?, maturity = ?,
                    updated_at = ?, last_accessed = ?, access_count = ?,
                    session_id = ?,
                    authority_score = ?, effective_authority = ?,
                    currency_score = ?, currency_status = ?,
                    staleness_state = ?, staleness_score = ?, staleness_reason = ?,
                    staleness_evaluated_at = ?, staleness_detector_version = ?,
                    staleness_consecutive_hits = ?,
                    superseded_by = ?, epistemic_network = ?,
                    reinforcement_count = ?,
                    original_date = ?,
                    edit_provenance = ?,
                    subject_key = ?,
                    content_updated_at = COALESCE(?, content_updated_at),
                    data = ?
                WHERE id = ?
            """,
                (
                    concept.version,
                    concept.summary,
                    concept.confidence,
                    concept.stability,
                    resolved_ka,  # DEBT-185: use pre-resolved KA (synced to blob)
                    getattr(concept, "concept_type", "insight"),
                    concept.status,
                    getattr(concept, "salience", 0.5),
                    getattr(concept, "salience_source", "system"),
                    getattr(concept, "maturity", "ESTABLISHED"),
                    now,
                    getattr(concept, "last_accessed", None),
                    getattr(concept, "access_count", 0),
                    session_id_val,
                    _clamp_score(getattr(concept, "authority_score", None)),  # DEBT-187
                    _clamp_score(getattr(concept, "effective_authority", None)),  # DEBT-187
                    _clamp_score(getattr(concept, "currency_score", None)),  # DEBT-187
                    getattr(concept, "currency_status", None),
                    *_stale_risk_field_values(data),
                    getattr(concept, "superseded_by", None),
                    getattr(concept, "epistemic_network", None),
                    getattr(concept, "reinforcement_count", None),
                    getattr(concept, "original_date", None),  # TEMPORAL-002
                    getattr(concept, "edit_provenance", None),  # RETRIEVAL-104
                    getattr(concept, "subject_key", None),  # EUNOMIA-040 Fix 3
                    now if summary_changed else None,
                    json.dumps(data),
                    concept.id,
                ),
            )
        else:
            conn.execute(
                """
                UPDATE concepts SET
                    version = ?, summary = ?, confidence = ?, stability = ?,
                    knowledge_area = ?, concept_type = ?, status = ?,
                    salience = ?, salience_source = ?, maturity = ?,
                    updated_at = ?, last_accessed = ?, access_count = ?,
                    authority_score = ?, effective_authority = ?,
                    currency_score = ?, currency_status = ?,
                    staleness_state = ?, staleness_score = ?, staleness_reason = ?,
                    staleness_evaluated_at = ?, staleness_detector_version = ?,
                    staleness_consecutive_hits = ?,
                    superseded_by = ?, epistemic_network = ?,
                    reinforcement_count = ?,
                    original_date = ?,
                    edit_provenance = ?,
                    subject_key = ?,
                    content_updated_at = COALESCE(?, content_updated_at),
                    data = ?
                WHERE id = ?
            """,
                (
                    concept.version,
                    concept.summary,
                    concept.confidence,
                    concept.stability,
                    resolved_ka,  # DEBT-185: use pre-resolved KA (synced to blob)
                    getattr(concept, "concept_type", "insight"),
                    concept.status,
                    getattr(concept, "salience", 0.5),
                    getattr(concept, "salience_source", "system"),
                    getattr(concept, "maturity", "ESTABLISHED"),
                    now,
                    getattr(concept, "last_accessed", None),
                    getattr(concept, "access_count", 0),
                    _clamp_score(getattr(concept, "authority_score", None)),  # DEBT-187
                    _clamp_score(getattr(concept, "effective_authority", None)),  # DEBT-187
                    _clamp_score(getattr(concept, "currency_score", None)),  # DEBT-187
                    getattr(concept, "currency_status", None),
                    *_stale_risk_field_values(data),
                    getattr(concept, "superseded_by", None),
                    getattr(concept, "epistemic_network", None),
                    getattr(concept, "reinforcement_count", None),
                    getattr(concept, "original_date", None),  # TEMPORAL-002
                    getattr(concept, "edit_provenance", None),  # RETRIEVAL-104
                    getattr(concept, "subject_key", None),  # EUNOMIA-040 Fix 3
                    now if summary_changed else None,
                    json.dumps(data),
                    concept.id,
                ),
            )
    else:
        validated_aid = validate_agent_id(meta.get("agent_id", "default"))
        conn.execute(
            """
            INSERT INTO concepts
            (id, version, summary, confidence, stability, knowledge_area,
             concept_type, status, salience, salience_source, maturity,
             created_at, updated_at, last_accessed, access_count, agent_id,
             session_id, authority_score, effective_authority,
             currency_score, currency_status, staleness_state, staleness_score,
             staleness_reason, staleness_evaluated_at, staleness_detector_version,
             staleness_consecutive_hits, superseded_by, epistemic_network,
             reinforcement_count, original_date, edit_provenance, subject_key, data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                concept.id,
                concept.version,
                concept.summary,
                concept.confidence,
                concept.stability,
                resolved_ka,  # DEBT-185: use pre-resolved KA (synced to blob)
                getattr(concept, "concept_type", "insight"),
                concept.status,
                getattr(concept, "salience", 0.5),
                getattr(concept, "salience_source", "system"),
                getattr(concept, "maturity", "ESTABLISHED"),
                concept.created_at,
                now,
                getattr(concept, "last_accessed", None),
                getattr(concept, "access_count", 0),
                validated_aid,  # 16: agent_id
                getattr(concept, "session_id", None),  # 17: session_id
                _clamp_score(getattr(concept, "authority_score", None)),  # 18: DEBT-137b + DEBT-187
                _clamp_score(getattr(concept, "effective_authority", None)),  # 19: DEBT-137b + DEBT-187
                _clamp_score(getattr(concept, "currency_score", None)),  # 20: DEBT-137b + DEBT-187
                getattr(concept, "currency_status", None),  # 21: DEBT-137b
                *_stale_risk_field_values(data),  # 22-27: COGGOV-014 stale-risk lifecycle
                getattr(concept, "superseded_by", None),  # 28: DEBT-137b
                getattr(concept, "epistemic_network", None),  # 29: DEBT-137b
                getattr(concept, "reinforcement_count", None),  # 30: DEBT-137b
                getattr(concept, "original_date", None),  # 31: TEMPORAL-002
                getattr(concept, "edit_provenance", None),  # 32: RETRIEVAL-104
                getattr(concept, "subject_key", None),  # 33: EUNOMIA-040 Fix 3
                json.dumps(data),  # 34: data (always last)
            ),
        )
    if not exists:
        # New concept: content_updated_at = now
        conn.execute(
            "UPDATE concepts SET content_updated_at = ? WHERE id = ?",
            (now, concept.id),
        )

    # RETRIEVAL-042 upgrade: Sync FTS5 full-text index
    _sync_fts5(conn, concept.id, concept.summary)

    conn.execute(
        """
        INSERT OR IGNORE INTO concept_versions (id, version, data, created_at)
        VALUES (?, ?, ?, ?)
    """,
        (concept.id, concept.version, json.dumps(data), concept.created_at),
    )
