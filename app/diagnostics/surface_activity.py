"""Surface activity diagnostics for lifecycle attribution audits."""

from __future__ import annotations

import os
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from app.storage.connection import diagnostic_read_db

CLAIM_BOUNDARY = "diagnostic_source_labeled_not_semantic_summary"
SURFACE_ACTIVITY_COVERAGE_SCHEMA_VERSION = "surface_activity_coverage.v1"
DEFAULT_WINDOW_HOURS = 24
DEFAULT_MIN_CONFIDENCE = 0.5
DEFAULT_MAX_SAMPLES_PER_SURFACE = 3
MAX_SAMPLES_PER_SURFACE_LIMIT = 10
MAX_REQUESTED_SURFACES = 32
QUALITY_CURRENCY_STATUSES = ("ACTIVE", "CONTESTED", "RESOLVED")
CODEX_STATE_PATH = Path("~/.codex/state_5.sqlite").expanduser()
CODEX_THREAD_COLUMNS = {"id", "source", "thread_source", "updated_at", "updated_at_ms"}


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso_or_default(value: str | None, default: datetime) -> str:
    if not value:
        return default.isoformat()
    return value


def _sqlite_datetime_expr(column_ms: str = "updated_at_ms", column_s: str = "updated_at") -> str:
    return f"datetime(coalesce({column_ms}, {column_s} * 1000) / 1000, 'unixepoch')"


def _row_dicts(rows) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _clamp_float(value: float | None, default: float, min_value: float, max_value: float) -> float:
    if value is None:
        return default
    return max(min_value, min(max_value, float(value)))


def _clamp_int(value: int | None, default: int, min_value: int, max_value: int) -> int:
    if value is None:
        return default
    return max(min_value, min(max_value, int(value)))


def _parse_requested_surfaces(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        return []
    raw_items = value if isinstance(value, (list, tuple)) else str(value).split(",")
    seen: set[str] = set()
    parsed: list[str] = []
    for item in raw_items:
        surface = str(item).strip().lower()
        if not surface or surface in seen:
            continue
        seen.add(surface)
        parsed.append(surface)
        if len(parsed) >= MAX_REQUESTED_SURFACES:
            break
    return parsed


def _requested_surface_token_count(value: str | list[str] | tuple[str, ...] | None) -> int:
    if value is None:
        return 0
    if isinstance(value, (list, tuple)):
        return len(value)
    return len([item for item in str(value).split(",") if item.strip()])


def _surface_verdict(session_count: int, concept_count: int, quality_concept_count: int) -> str:
    if quality_concept_count > 0:
        return "covered"
    if session_count > 0 or concept_count > 0:
        return "sparse"
    return "absent"


def _overall_coverage_verdict(surface_rows: list[dict[str, Any]]) -> str:
    if not surface_rows:
        return "unsupported"
    verdicts = [str(row["verdict"]) for row in surface_rows]
    if all(verdict == "unsupported" for verdict in verdicts):
        return "unsupported"
    if all(verdict == "covered" for verdict in verdicts):
        return "covered"
    if any(verdict == "covered" for verdict in verdicts):
        return "partial"
    return "missing"


def _build_agent_context_block(coverage: dict[str, Any]) -> dict[str, Any]:
    surfaces = coverage.get("surfaces", [])
    if not surfaces:
        return {
            "status": "unavailable",
            "fallback_artifacts_needed": True,
            "text": "No requested surface coverage was computed.",
        }
    fragments = [
        f"{row['surface_id']}={row['verdict']} "
        f"(quality={row['quality_concept_count']}, concepts={row['concept_count']})"
        for row in surfaces
    ]
    fallback_needed = any(row["verdict"] != "covered" for row in surfaces)
    text = (
        "Surface activity coverage: "
        + "; ".join(fragments)
        + ". This is source coverage evidence, not a semantic answer."
    )
    if fallback_needed:
        text += " Use durable fallback artifacts for sparse, absent, or unsupported surfaces."
    return {
        "status": "available",
        "fallback_artifacts_needed": fallback_needed,
        "text": text,
    }


def _build_pith_lifecycle(since: str, until: str) -> dict[str, Any]:
    with diagnostic_read_db("surface_activity_diagnostic") as conn:
        sessions_by_surface = _row_dicts(
            conn.execute(
                """SELECT COALESCE(surface_id, 'unknown') AS surface_id,
                          COALESCE(platform_hint, 'unknown') AS platform_hint,
                          COUNT(*) AS count
                   FROM sessions
                   WHERE started_at >= ? AND started_at < ?
                   GROUP BY COALESCE(surface_id, 'unknown'), COALESCE(platform_hint, 'unknown')
                   ORDER BY count DESC, surface_id, platform_hint""",
                (since, until),
            ).fetchall()
        )
        raw_payload_completeness = _row_dicts(
            conn.execute(
                """SELECT COALESCE(payload_completeness, 'unknown') AS payload_completeness,
                          COUNT(*) AS count
                   FROM raw_turn_payloads
                   WHERE captured_at >= ? AND captured_at < ?
                   GROUP BY COALESCE(payload_completeness, 'unknown')
                   ORDER BY count DESC, payload_completeness""",
                (since, until),
            ).fetchall()
        )
        raw_payloads_by_surface = _row_dicts(
            conn.execute(
                """SELECT COALESCE(s.surface_id, 'unknown') AS surface_id,
                          COALESCE(s.platform_hint, 'unknown') AS platform_hint,
                          COALESCE(r.payload_completeness, 'unknown') AS payload_completeness,
                          COUNT(*) AS count
                   FROM raw_turn_payloads r
                   LEFT JOIN sessions s ON s.id = r.session_id
                   WHERE r.captured_at >= ? AND r.captured_at < ?
                   GROUP BY COALESCE(s.surface_id, 'unknown'),
                            COALESCE(s.platform_hint, 'unknown'),
                            COALESCE(r.payload_completeness, 'unknown')
                   ORDER BY count DESC, surface_id, platform_hint, payload_completeness""",
                (since, until),
            ).fetchall()
        )
    return {
        "sessions_by_surface": sessions_by_surface,
        "raw_payload_completeness": raw_payload_completeness,
        "raw_payloads_by_surface": raw_payloads_by_surface,
    }


def _build_codex_local_threads(since: str, until: str) -> dict[str, Any]:
    path = Path(os.environ.get("PITH_CODEX_STATE_DB", str(CODEX_STATE_PATH))).expanduser()
    if not path.exists():
        return {"available": False, "error": f"codex_state_db_missing:{path}"}

    try:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        try:
            columns = _table_columns(conn, "threads")
            missing = CODEX_THREAD_COLUMNS - columns
            if missing:
                return {
                    "available": False,
                    "error": "codex_threads_schema_missing:" + ",".join(sorted(missing)),
                }
            updated_expr = _sqlite_datetime_expr()
            source_distribution = _row_dicts(
                conn.execute(
                    f"""WITH t AS (
                            SELECT source, thread_source, {updated_expr} AS updated_utc
                            FROM threads
                        )
                        SELECT COALESCE(source, 'unknown') AS source,
                               COALESCE(thread_source, 'unknown') AS thread_source,
                               COUNT(*) AS count
                        FROM t
                        WHERE updated_utc >= ? AND updated_utc < ?
                        GROUP BY COALESCE(source, 'unknown'), COALESCE(thread_source, 'unknown')
                        ORDER BY count DESC, source, thread_source""",
                    (since.replace("T", " ")[:19], until.replace("T", " ")[:19]),
                ).fetchall()
            )
            records = _row_dicts(
                conn.execute(
                    f"""WITH t AS (
                            SELECT id, source, thread_source, {updated_expr} AS updated_utc
                            FROM threads
                        )
                        SELECT id,
                               COALESCE(source, 'unknown') AS source,
                               COALESCE(thread_source, 'unknown') AS thread_source,
                               updated_utc
                        FROM t
                        WHERE updated_utc >= ? AND updated_utc < ?
                        ORDER BY updated_utc DESC
                        LIMIT 100""",
                    (since.replace("T", " ")[:19], until.replace("T", " ")[:19]),
                ).fetchall()
            )
            return {
                "available": True,
                "path": str(path),
                "source_distribution": source_distribution,
                "records": records,
                "record_limit": 100,
            }
        finally:
            conn.close()
    except Exception as exc:
        return {"available": False, "path": str(path), "error": f"{type(exc).__name__}: {exc}"}


def build_requested_surface_coverage(
    *,
    since: str,
    until: str,
    requested_surfaces: str | list[str] | tuple[str, ...] | None,
    include_concept_samples: bool = False,
    min_confidence: float | None = None,
    max_samples_per_surface: int | None = None,
) -> dict[str, Any]:
    from app.core.models import SURFACE_ID_VALUES

    requested = _parse_requested_surfaces(requested_surfaces)
    raw_requested_count = _requested_surface_token_count(requested_surfaces)
    min_conf = _clamp_float(min_confidence, DEFAULT_MIN_CONFIDENCE, 0.0, 1.0)
    sample_limit = _clamp_int(
        max_samples_per_surface,
        DEFAULT_MAX_SAMPLES_PER_SURFACE,
        0,
        MAX_SAMPLES_PER_SURFACE_LIMIT,
    )
    known_surfaces = set(SURFACE_ID_VALUES)
    valid_requested = [surface for surface in requested if surface in known_surfaces]
    unsupported = [surface for surface in requested if surface not in known_surfaces]

    rows_by_surface: dict[str, dict[str, Any]] = {}
    samples_by_surface: dict[str, list[dict[str, Any]]] = {surface: [] for surface in valid_requested}

    if valid_requested:
        surface_placeholders = ",".join("?" for _ in valid_requested)
        currency_placeholders = ",".join("?" for _ in QUALITY_CURRENCY_STATUSES)
        with diagnostic_read_db("surface_activity_coverage") as conn:
            aggregate_rows = conn.execute(
                f"""SELECT COALESCE(s.surface_id, 'unknown') AS surface_id,
                          COUNT(DISTINCT s.id) AS session_count,
                          COUNT(c.id) AS concept_count,
                          SUM(CASE WHEN c.id IS NOT NULL
                                    AND c.status = 'active'
                                    AND COALESCE(c.currency_status, 'ACTIVE') IN ({currency_placeholders})
                                    AND c.confidence >= ?
                                    AND length(trim(c.summary)) > 0
                                   THEN 1 ELSE 0 END) AS quality_concept_count,
                          MAX(c.created_at) AS latest_concept_at
                   FROM sessions s
                   LEFT JOIN concepts c ON c.session_id = s.id
                   WHERE s.started_at >= ? AND s.started_at < ?
                     AND s.surface_id IN ({surface_placeholders})
                   GROUP BY COALESCE(s.surface_id, 'unknown')""",
                (*QUALITY_CURRENCY_STATUSES, min_conf, since, until, *valid_requested),
            ).fetchall()
            rows_by_surface = {str(row["surface_id"]): dict(row) for row in aggregate_rows}

            if include_concept_samples and sample_limit > 0:
                sample_rows = conn.execute(
                    f"""SELECT s.surface_id AS surface_id,
                              c.id AS concept_id,
                              c.summary AS summary,
                              c.knowledge_area AS knowledge_area,
                              c.confidence AS confidence,
                              c.status AS status,
                              COALESCE(c.currency_status, 'ACTIVE') AS currency_status,
                              c.created_at AS created_at,
                              c.session_id AS session_id
                       FROM concepts c
                       JOIN sessions s ON s.id = c.session_id
                       WHERE s.started_at >= ? AND s.started_at < ?
                         AND s.surface_id IN ({surface_placeholders})
                         AND c.status = 'active'
                         AND COALESCE(c.currency_status, 'ACTIVE') IN ({currency_placeholders})
                         AND c.confidence >= ?
                         AND length(trim(c.summary)) > 0
                       ORDER BY s.surface_id, c.created_at DESC""",
                    (since, until, *valid_requested, *QUALITY_CURRENCY_STATUSES, min_conf),
                ).fetchall()
                for row in sample_rows:
                    surface_id = str(row["surface_id"])
                    bucket = samples_by_surface.setdefault(surface_id, [])
                    if len(bucket) < sample_limit:
                        bucket.append(dict(row))

    surfaces: list[dict[str, Any]] = []
    for surface in valid_requested:
        row = rows_by_surface.get(surface, {})
        session_count = int(row.get("session_count") or 0)
        concept_count = int(row.get("concept_count") or 0)
        quality_count = int(row.get("quality_concept_count") or 0)
        surfaces.append(
            {
                "surface_id": surface,
                "verdict": _surface_verdict(session_count, concept_count, quality_count),
                "session_count": session_count,
                "concept_count": concept_count,
                "quality_concept_count": quality_count,
                "latest_concept_at": row.get("latest_concept_at"),
                "sample_concepts": samples_by_surface.get(surface, []) if include_concept_samples else [],
            }
        )
    for surface in unsupported:
        surfaces.append(
            {
                "surface_id": surface,
                "verdict": "unsupported",
                "session_count": 0,
                "concept_count": 0,
                "quality_concept_count": 0,
                "latest_concept_at": None,
                "sample_concepts": [],
            }
        )

    limitations = [
        "diagnostic coverage evidence only; does not prove semantic answer sufficiency",
        "V1 does not group related surfaces; request each surface_id explicitly",
        "unknown is an attribution bucket, not proof of a named consumer surface",
    ]
    if raw_requested_count > MAX_REQUESTED_SURFACES:
        limitations.append(f"requested_surfaces truncated to first {MAX_REQUESTED_SURFACES} unique entries")

    return {
        "schema_version": SURFACE_ACTIVITY_COVERAGE_SCHEMA_VERSION,
        "requested_surfaces": requested,
        "window": {"since": since, "until": until, "timezone": "UTC"},
        "quality_floor": {
            "min_confidence": min_conf,
            "status": "active",
            "currency_statuses": list(QUALITY_CURRENCY_STATUSES),
        },
        "surfaces": surfaces,
        "overall_verdict": _overall_coverage_verdict(surfaces),
        "limitations": limitations,
    }


def build_surface_activity_diagnostic(
    *,
    since: str | None = None,
    until: str | None = None,
    include_codex_local: bool = False,
    requested_surfaces: str | list[str] | tuple[str, ...] | None = None,
    include_concept_samples: bool = False,
    min_confidence: float | None = None,
    max_samples_per_surface: int | None = None,
) -> dict[str, Any]:
    now = _utc_now()
    since_iso = _iso_or_default(since, now - timedelta(hours=DEFAULT_WINDOW_HOURS))
    until_iso = _iso_or_default(until, now)
    payload = {
        "period": {"since": since_iso, "until": until_iso, "timezone": "UTC"},
        "pith_lifecycle": _build_pith_lifecycle(since_iso, until_iso),
        "codex_local_threads": {"available": False, "skipped": not include_codex_local},
        "claim_boundary": CLAIM_BOUNDARY,
    }
    if include_codex_local:
        payload["codex_local_threads"] = _build_codex_local_threads(since_iso, until_iso)
    if _parse_requested_surfaces(requested_surfaces):
        coverage = build_requested_surface_coverage(
            since=since_iso,
            until=until_iso,
            requested_surfaces=requested_surfaces,
            include_concept_samples=include_concept_samples,
            min_confidence=min_confidence,
            max_samples_per_surface=max_samples_per_surface,
        )
        payload["requested_surface_coverage"] = coverage
        payload["agent_context_block"] = _build_agent_context_block(coverage)
    return payload
