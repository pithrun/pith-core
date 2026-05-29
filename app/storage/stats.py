"""Storage sub-module: stats.

Statistics aggregation and analytics.
Extracted from storage/__init__.py during Item 2b decomposition.
"""
import logging
import math
import sqlite3
from contextlib import nullcontext
from datetime import UTC

from app.core.datetime_utils import _utc_now_iso
from app.storage import concepts as _concepts_mod
from app.storage.connection import diagnostic_read_db, read_snapshot_db

logger = logging.getLogger(__name__)

def get_pith_stats_fast(conn: sqlite3.Connection | None = None) -> dict:
    """Bounded stats summary for default health/status surfaces.

    This intentionally avoids the full monitoring aggregate helper. Keep this
    path limited to cheap read-only SQL that is safe for MCP and report defaults.
    """
    context = nullcontext(conn) if conn is not None else diagnostic_read_db("pith_stats_fast")
    with context as conn:
        row = conn.execute("""
            SELECT
                COUNT(*) as total_concepts,
                COALESCE(AVG(confidence), 0.0) as avg_confidence,
                COALESCE(AVG(stability), 0.0) as avg_stability,
                COUNT(DISTINCT knowledge_area) as knowledge_areas
            FROM concepts
            WHERE status = 'active'
        """).fetchone()
        assoc_row = conn.execute("SELECT COUNT(*) as cnt FROM associations").fetchone()
        data_quality_row = conn.execute("""
            SELECT
                SUM(CASE WHEN last_accessed IS NULL OR last_accessed = ''
                    THEN 1 ELSE 0 END) as null_timestamps
            FROM concepts WHERE status = 'active'
        """).fetchone()

    return {
        "total_concepts": row["total_concepts"] if row else 0,
        "avg_confidence": round(row["avg_confidence"] or 0.0, 4) if row else 0.0,
        "avg_stability": round(row["avg_stability"] or 0.0, 4) if row else 0.0,
        "knowledge_areas": row["knowledge_areas"] if row else 0,
        "associations": assoc_row["cnt"] if assoc_row else 0,
        "data_quality": {
            "null_timestamps": data_quality_row["null_timestamps"] or 0 if data_quality_row else 0,
            "bad_json": 0,
            "bad_json_checked": False,
        },
    }


def get_pith_health_fast(conn: sqlite3.Connection | None = None) -> dict:
    """Bounded cognitive health summary for default status surfaces."""
    context = nullcontext(conn) if conn is not None else diagnostic_read_db("pith_health_fast")
    with context as conn:
        row = conn.execute("""
            SELECT
                COUNT(*) as total_concepts,
                COALESCE(AVG(confidence), 0.0) as avg_confidence,
                COALESCE(AVG(stability), 0.0) as avg_stability,
                SUM(CASE WHEN maturity = 'ESTABLISHED' THEN 1 ELSE 0 END) as established_count
            FROM concepts
            WHERE status = 'active'
        """).fetchone()

    total = row["total_concepts"] if row else 0
    avg_confidence = float(row["avg_confidence"] or 0.0) if row else 0.0
    avg_stability = float(row["avg_stability"] or 0.0) if row else 0.0
    established = int(row["established_count"] or 0) if row else 0
    maturity_health = established / max(total, 1)
    health_score = round(0.4 * avg_confidence + 0.4 * avg_stability + 0.2 * maturity_health, 4)

    return {
        "status": "healthy" if total else "empty",
        "score_model": "fast_v1",
        "health_score": health_score,
        "stability_score": health_score,
        "total_concepts": total,
        "avg_confidence": round(avg_confidence, 4),
        "avg_stability": round(avg_stability, 4),
        "established_concepts": established,
        "maturity_health": round(maturity_health, 4),
    }


def get_pith_stats_aggregates(conn: sqlite3.Connection | None = None) -> dict:
    """Aggregate pith stats (concept counts, avg confidence, KA breakdown, orphans).

    Called by pith_stats() MCP endpoint. Use this, not count_* individually,
    to avoid N+1 queries on the stats path.
    """
    # MONITOR-CI032A-01: post-init migration integrity audit. Uses the public
    # get_backend() accessor (singleton, re-entrant safe via _backend_lock).
    # Wrapped in try/except so a stats query never fails on an audit glitch.
    try:
        from app.storage.backend import get_backend
        _backend = get_backend()
        with read_snapshot_db("get_pith_stats_aggregates") as _integ_conn:
            migration_integrity = _backend._check_migration_integrity(_integ_conn)
    except Exception as _integ_err:
        migration_integrity = {
            "status": "UNKNOWN",
            "expected_count": 0,
            "present_count": 0,
            "missing": [],
            "error": str(_integ_err),
        }

    context = nullcontext(conn) if conn is not None else read_snapshot_db("get_pith_stats_aggregates")
    with context as conn:
        row = conn.execute("""
            SELECT
                COUNT(*) as total_concepts,
                COALESCE(AVG(confidence), 0.0) as avg_confidence,
                COALESCE(AVG(stability), 0.0) as avg_stability,
                COUNT(DISTINCT knowledge_area) as knowledge_areas
            FROM concepts
            WHERE status = 'active'
        """).fetchone()

        versions_row = conn.execute("SELECT COUNT(*) as total FROM concept_versions").fetchone()

        # DEBT-006: Per-knowledge-area breakdown
        ka_rows = conn.execute("""
            SELECT
                COALESCE(knowledge_area, 'unknown') as ka,
                COUNT(*) as count,
                ROUND(AVG(confidence), 4) as avg_conf,
                ROUND(AVG(stability), 4) as avg_stab
            FROM concepts
            WHERE status = 'active'
            GROUP BY knowledge_area
            ORDER BY count DESC
        """).fetchall()
        ka_breakdown = [
            {
                "knowledge_area": r["ka"],
                "count": r["count"],
                "avg_confidence": r["avg_conf"],
                "avg_stability": r["avg_stab"],
            }
            for r in ka_rows
        ]

        # DEBT-003: Orphan concept count
        orphan_row = conn.execute("""
            SELECT COUNT(*) as cnt FROM concepts c
            WHERE c.status = 'active'
            AND c.id NOT IN (SELECT source FROM associations)
            AND c.id NOT IN (SELECT target FROM associations)
        """).fetchone()

        # DEBT-007: Evidence provenance summary
        # HEALTH-005: json_valid guard prevents 500 on malformed data blobs
        evidence_row = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN json_valid(data)
                    AND json_array_length(json_extract(data, '$.evidence')) > 0
                    THEN 1 ELSE 0 END) as with_evidence,
                SUM(CASE WHEN NOT json_valid(data)
                    OR json_array_length(json_extract(data, '$.evidence')) = 0
                    OR json_extract(data, '$.evidence') IS NULL
                    THEN 1 ELSE 0 END) as without_evidence,
                ROUND(AVG(CASE WHEN json_valid(data)
                    THEN json_array_length(json_extract(data, '$.evidence'))
                    ELSE 0 END), 2) as avg_evidence_count
            FROM concepts
            WHERE status = 'active'
            AND json_valid(data) = 1
        """).fetchone()

        # HEALTH-005: data quality metrics for observability
        data_quality_row = conn.execute("""
            SELECT
                SUM(CASE WHEN last_accessed IS NULL OR last_accessed = ''
                    THEN 1 ELSE 0 END) as null_timestamps,
                SUM(CASE WHEN json_valid(data) = 0
                    THEN 1 ELSE 0 END) as bad_json
            FROM concepts WHERE status = 'active'
        """).fetchone()

        # MONITOR-003: reinforcement_count distribution
        rc_rows = conn.execute("""
            SELECT CASE
                WHEN reinforcement_count IS NULL OR reinforcement_count = 0 THEN '0'
                WHEN reinforcement_count BETWEEN 1 AND 3 THEN '1-3'
                WHEN reinforcement_count BETWEEN 4 AND 10 THEN '4-10'
                ELSE '10+'
            END as bucket, COUNT(*) as cnt
            FROM concepts WHERE status = 'active' GROUP BY bucket
        """).fetchall()
        rc_dist = {r["bucket"]: r["cnt"] for r in rc_rows}

        # MONITOR-003: access_count distribution
        ac_rows = conn.execute("""
            SELECT CASE
                WHEN access_count IS NULL OR access_count = 0 THEN '0'
                WHEN access_count BETWEEN 1 AND 5 THEN '1-5'
                WHEN access_count BETWEEN 6 AND 20 THEN '6-20'
                ELSE '20+'
            END as bucket, COUNT(*) as cnt
            FROM concepts WHERE status = 'active' GROUP BY bucket
        """).fetchall()
        ac_dist = {r["bucket"]: r["cnt"] for r in ac_rows}

        # MONITOR-004: last_accessed temporal distribution
        la_rows = conn.execute("""
            SELECT CASE
                WHEN last_accessed IS NULL THEN 'never'
                WHEN last_accessed > datetime('now', '-1 day') THEN 'last_24h'
                WHEN last_accessed > datetime('now', '-7 days') THEN 'last_7d'
                WHEN last_accessed > datetime('now', '-30 days') THEN 'last_30d'
                ELSE 'older'
            END as recency, COUNT(*) as cnt
            FROM concepts WHERE status = 'active' GROUP BY recency
        """).fetchall()
        la_dist = {r["recency"]: r["cnt"] for r in la_rows}

        # MONITOR-029: Governance sweep zero-count alert
        gov_24h_row = conn.execute("""
            SELECT
                SUM(CASE WHEN event_type = 'MATURITY_PROMOTED' THEN 1 ELSE 0 END) as promotions_24h,
                SUM(CASE WHEN event_type LIKE '%QUARANTINE%' THEN 1 ELSE 0 END) as quarantine_24h,
                SUM(CASE WHEN event_type LIKE '%BACKFILL%' THEN 1 ELSE 0 END) as backfill_24h,
                COUNT(*) as total_gov_events_24h
            FROM governance_events
            WHERE created_at > datetime('now', '-1 day')
        """).fetchone()
        _active_concepts = row["total_concepts"] or 0
        _total_gov_24h = gov_24h_row["total_gov_events_24h"] or 0
        _gov_sweep_alert = _active_concepts > 1000 and _total_gov_24h == 0

        # MEASURE-008: Experiment concept efficacy tracking
        exp_efficacy_row = conn.execute("""
            SELECT
                COUNT(*) as total,
                COALESCE(AVG(access_count), 0) as avg_access,
                COALESCE(AVG(confidence), 0) as avg_conf,
                SUM(CASE WHEN access_count > 0 THEN 1 ELSE 0 END) as retrieved
            FROM concepts
            WHERE status = 'active' AND json_valid(data) = 1
            AND json_extract(data, '$.evidence') LIKE '%experiment:%'
        """).fetchone()
        regular_access_row = conn.execute("""
            SELECT COALESCE(AVG(access_count), 0) as avg_access
            FROM concepts WHERE status = 'active'
            AND (NOT json_valid(data)
                 OR json_extract(data, '$.evidence') NOT LIKE '%experiment:%')
        """).fetchone()

        # MONITOR-036: score-range validation — flag out-of-[0,1] scores
        oor_row = conn.execute("""
            SELECT
                SUM(CASE WHEN authority_score IS NOT NULL
                    AND (authority_score < 0.0 OR authority_score > 1.0) THEN 1 ELSE 0 END) as auth_oor,
                SUM(CASE WHEN effective_authority IS NOT NULL
                    AND (effective_authority < 0.0 OR effective_authority > 1.0) THEN 1 ELSE 0 END) as eff_auth_oor,
                SUM(CASE WHEN confidence IS NOT NULL
                    AND (confidence < 0.0 OR confidence > 1.0) THEN 1 ELSE 0 END) as conf_oor
            FROM concepts WHERE status = 'active'
        """).fetchone()

        # MONITOR-038: always-activate concept count
        always_activate_row = conn.execute(
            "SELECT COUNT(*) as cnt FROM concepts WHERE always_activate = 1 AND status = 'active'"
        ).fetchone()

        # MONITOR-033: ka_relative_authority coverage
        ka_ra_row = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN ka_relative_authority IS NOT NULL THEN 1 ELSE 0 END) as with_ka_ra
            FROM concepts WHERE status = 'active'
        """).fetchone()

        # MONITOR-024: reflection_tracking table analytics
        rt_rows = conn.execute("""
            SELECT trigger_type,
                   COUNT(*) as cnt,
                   SUM(CASE WHEN reflection_quality = 'timeout' THEN 1 ELSE 0 END) as timeouts,
                   AVG(concepts_returned) as avg_returned,
                   AVG(prompts_sent) as avg_prompts
            FROM reflection_tracking
            GROUP BY trigger_type
        """).fetchall()
        rt_total_row = conn.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN reflection_quality = 'timeout' THEN 1 ELSE 0 END) as total_timeouts,
                   SUM(CASE WHEN reflection_quality = 'auto_closed' THEN 1 ELSE 0 END) as total_auto_closed,
                   AVG(concepts_returned) as avg_concepts_returned
            FROM reflection_tracking
        """).fetchone()
        _rt_by_trigger = {
            r["trigger_type"]: {
                "count": r["cnt"],
                "timeouts": r["timeouts"],
                "avg_concepts_returned": round(r["avg_returned"] or 0, 2),
                "avg_prompts_sent": round(r["avg_prompts"] or 0, 2),
            }
            for r in rt_rows
        }

        # MONITOR-032: zombie concepts (is_current=1, not archived, no associations)
        zombie_row = conn.execute("""
            SELECT COUNT(*) as cnt FROM concepts
            WHERE is_current = 1
              AND status != 'archived'
              AND id NOT IN (
                  SELECT source FROM associations
                  UNION
                  SELECT target FROM associations
              )
        """).fetchone()

        # MONITOR-011: Index consistency — active vs superseded concept counts
        index_consistency_row = conn.execute("""
            SELECT
                SUM(CASE WHEN is_current = 1 AND status = 'active' THEN 1 ELSE 0 END) as active_current,
                SUM(CASE WHEN is_current = 0 AND status = 'active' THEN 1 ELSE 0 END) as active_superseded,
                COUNT(*) as total_all
            FROM concepts
        """).fetchone()

        # MONITOR-013: FIX-1 effectiveness — concepts with is_current=1 AND superseded_by set
        # (resurrection guard bypass: if count > 0, FIX-1 missed a resurrection)
        fix1_zombie_row = conn.execute("""
            SELECT COUNT(*) as cnt FROM concepts
            WHERE is_current = 1
              AND superseded_by IS NOT NULL
              AND superseded_by NOT IN ('', '__orphaned_supersession__')
              AND status = 'active'
        """).fetchone()

        # MONITOR-049: CTX-007 compaction survival format monitoring
        # Liveness probe: sample one eligible concept, confirm formatter produces [CRITICAL-CONTEXT].
        from app.core.config import FEATURE_FLAGS as _ff
        _csf_enabled = _ff.get("COMPACTION_SURVIVAL_FORMAT", False)
        _csf_row = conn.execute("""
            SELECT COUNT(*) as eligible
            FROM concepts
            WHERE status = 'active'
              AND concept_type IN ('constraint', 'decision', 'principle')
        """).fetchone()
        _csf_probe_ok = False
        if _csf_enabled and (_csf_row["eligible"] or 0) > 0:
            _probe = conn.execute("""
                SELECT id, summary, concept_type FROM concepts
                WHERE status = 'active' AND concept_type IN ('constraint', 'decision', 'principle')
                LIMIT 1
            """).fetchone()
            if _probe:
                from app.core.format_helpers import format_for_compaction_survival
                _formatted = format_for_compaction_survival(
                    _probe["id"], _probe["summary"], _probe["concept_type"]
                )
                _csf_probe_ok = "[CRITICAL-CONTEXT" in _formatted

        # ARGUS-S23-F1: Token budget monitoring for COMPACTION_SURVIVAL_FORMAT
        # Estimate token overhead: avg summary chars of eligible concepts × ~4 chars/token + tag overhead
        _csf_avg_chars = 0
        _csf_est_tokens = 0
        if _csf_enabled and (_csf_row["eligible"] or 0) > 0:
            _csf_len_row = conn.execute("""
                SELECT ROUND(AVG(LENGTH(COALESCE(summary, ''))), 0) as avg_chars
                FROM concepts
                WHERE status = 'active'
                  AND concept_type IN ('constraint', 'decision', 'principle')
            """).fetchone()
            _csf_avg_chars = int(_csf_len_row["avg_chars"] or 0)
            # ~4 chars/token + 20 tokens tag overhead per concept; cap at 10 (always-activate pool)
            _concepts_in_budget = min(10, _csf_row["eligible"] or 0)
            _csf_est_tokens = round((_csf_avg_chars / 4 + 20) * _concepts_in_budget)

        # MONITOR-035: currency health alert
        _curr_row = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN currency_status = 'CONTRADICTED' THEN 1 ELSE 0 END) as contradicted,
                AVG(currency_score) as mean_score
            FROM concepts
            WHERE is_current = 1 AND status != 'archived'
        """).fetchone()

        # MONITOR-069: Factual coverage metrics
        factual_row = conn.execute("""
            SELECT
                COUNT(*) as total_active,
                SUM(CASE WHEN json_valid(data) AND json_extract(data, '$.metadata.is_factual') = 1
                    THEN 1 ELSE 0 END) as factual_count,
                SUM(CASE WHEN valid_from IS NOT NULL AND valid_from != ''
                    THEN 1 ELSE 0 END) as has_valid_from
            FROM concepts
            WHERE status = 'active' AND is_current = 1
        """).fetchone()
        _factual_total = factual_row["total_active"] or 1
        _factual_count = factual_row["factual_count"] or 0
        _has_valid_from = factual_row["has_valid_from"] or 0
        _factual_rate = round(_factual_count / _factual_total * 100, 1)

        # ARGUS-S25-F2: experiment_generation task health monitoring
        # Apply MONITOR-049 liveness pattern: last run recency + status.
        # Defensive: async_task_runs may not exist if ensure_async_tables hasn't run yet.
        try:
            _exp_gen_row = conn.execute("""
                SELECT status, started_at, completed_at, duration_ms, items_processed
                FROM async_task_runs
                WHERE task_type = 'experiment_generation'
                ORDER BY started_at DESC
                LIMIT 1
            """).fetchone()
        except Exception:
            _exp_gen_row = None
        if _exp_gen_row:
            _exp_gen_age_row = conn.execute(
                "SELECT ROUND((julianday('now') - julianday(?)) * 24, 1) as hours_ago",
                (_exp_gen_row["started_at"],),
            ).fetchone()
            _exp_gen_hours = _exp_gen_age_row["hours_ago"] if _exp_gen_age_row else None
            _exp_gen_status = (
                "healthy"
                if (_exp_gen_row["status"] == "success" and _exp_gen_hours is not None and _exp_gen_hours <= 48.0)
                else "stale"
                if (_exp_gen_hours is not None and _exp_gen_hours > 48.0)
                else "degraded"
            )
            _exp_gen_info: dict = {
                "last_status": _exp_gen_row["status"],
                "last_run_hours_ago": _exp_gen_hours,
                "last_duration_ms": _exp_gen_row["duration_ms"],
                "items_processed": _exp_gen_row["items_processed"],
                "health": _exp_gen_status,
            }
        else:
            _exp_gen_info = {
                "health": "never_run",
                "last_status": None,
                "last_run_hours_ago": None,
                "last_duration_ms": None,
                "items_processed": None,
            }


        # MONITOR-034: Stuck PROVISIONAL concepts (evidence<1 OR access<5 AND reinforcement<8)
        _stuck_prov_row = conn.execute("""
            SELECT
                COUNT(*) as stuck,
                (SELECT COUNT(*) FROM concepts
                 WHERE status='active' AND maturity='PROVISIONAL') as total_prov
            FROM concepts
            WHERE status='active' AND maturity='PROVISIONAL'
            AND (
                COALESCE(json_array_length(json_extract(data,'$.evidence')), 0) < 1
                OR (access_count < 5 AND reinforcement_count < 8)
            )
        """).fetchone()
        _stuck_prov = _stuck_prov_row["stuck"] if _stuck_prov_row else 0
        _total_prov = _stuck_prov_row["total_prov"] if _stuck_prov_row else 0

        # MONITOR-018: Stale session buildup (no ended_at, started >24h ago)
        _stale_sess_row = conn.execute("""
            SELECT COUNT(*) as cnt FROM sessions
            WHERE ended_at IS NULL
            AND started_at < datetime('now', '-24 hours')
        """).fetchone()
        _stale_sessions = _stale_sess_row["cnt"] if _stale_sess_row else 0

        # MONITOR-009: Context pressure trend (sessions.pressure_score)
        _pressure_row = conn.execute(
            """
            SELECT
                ROUND(AVG(CASE WHEN started_at > datetime('now','-1 day')
                               THEN pressure_score END), 3) AS avg_24h,
                COUNT(CASE WHEN started_at > datetime('now','-1 day')
                            AND pressure_score > 0.7 THEN 1 END) AS high_pressure_24h,
                ROUND(AVG(CASE WHEN started_at BETWEEN datetime('now','-7 days')
                               AND datetime('now','-1 day')
                               THEN pressure_score END), 3) AS avg_prior_7d
            FROM sessions
            WHERE pressure_score IS NOT NULL
            """
        ).fetchone()
        _p_avg_24h = (
            float(_pressure_row["avg_24h"])
            if _pressure_row and _pressure_row["avg_24h"] is not None
            else None
        )
        _p_high_24h = int(_pressure_row["high_pressure_24h"] or 0) if _pressure_row else 0
        _p_avg_7d = (
            float(_pressure_row["avg_prior_7d"])
            if _pressure_row and _pressure_row["avg_prior_7d"] is not None
            else None
        )
        if _p_avg_24h is not None and _p_avg_7d is not None:
            _p_trend = "rising" if _p_avg_24h > _p_avg_7d + 0.05 else (
                "falling" if _p_avg_24h < _p_avg_7d - 0.05 else "stable"
            )
        else:
            _p_trend = "unknown"

        # MONITOR-056: Cross-KA guard activation rate (24h from metrics)
        _cross_ka_row = conn.execute("""
            SELECT COALESCE(SUM(value), 0) as total_24h
            FROM metrics
            WHERE metric = 'cross_ka_guard_activations'
            AND timestamp > datetime('now', '-1 day')
        """).fetchone()
        _cross_ka_24h = int(_cross_ka_row["total_24h"] or 0)

        # MONITOR-058: Episode count for health monitoring
        try:
            _ep_row = conn.execute("SELECT COUNT(*) as cnt FROM episodes").fetchone()
            _episode_count = _ep_row["cnt"] if _ep_row else 0
        except Exception:
            _episode_count = -1  # Table may not exist in all environments

        # MONITOR-044: PSIS M3 compliance — quarantined concepts over confidence cap
        _psis_overcap_row = conn.execute("""
            SELECT COUNT(*) as over_cap FROM concepts
            WHERE maturity = 'QUARANTINED' AND confidence > 0.4
        """).fetchone()
        _psis_overcap = _psis_overcap_row["over_cap"] if _psis_overcap_row else 0

        # MONITOR-128: Correction pipeline stats — gated count, evidence appends, pattern match rate
        _corr_events_row = conn.execute("""
            SELECT COUNT(*) as total,
                   COALESCE(AVG(CAST(json_extract(details, '$.detection_confidence') AS REAL)), 0) as avg_conf,
                   COALESCE(SUM(CAST(json_extract(details, '$.affected_count') AS INTEGER)), 0) as total_affected
            FROM governance_events
            WHERE event_type = 'CORRECTION_RECORDED'
        """).fetchone()
        _corr_total = int(_corr_events_row["total"] or 0)
        _corr_avg_conf = round(float(_corr_events_row["avg_conf"] or 0.0), 3)
        _corr_total_affected = int(_corr_events_row["total_affected"] or 0)

        _corr_metrics_row = conn.execute("""
            SELECT
                COALESCE(SUM(CASE WHEN metric = 'coggov009_gated_count' THEN value ELSE 0 END), 0) as gated,
                COALESCE(SUM(CASE WHEN metric = 'coggov009_evidence_appends' THEN value ELSE 0 END), 0) as appends,
                COALESCE(SUM(CASE WHEN metric = 'coggov006_layer5_fires' THEN value ELSE 0 END), 0) as l5_fires
            FROM metrics
            WHERE metric IN ('coggov009_gated_count', 'coggov009_evidence_appends', 'coggov006_layer5_fires')
        """).fetchone()
        _corr_gated = int(_corr_metrics_row["gated"] or 0)
        _corr_appends = int(_corr_metrics_row["appends"] or 0)
        _corr_l5_fires = int(_corr_metrics_row["l5_fires"] or 0)
        _corr_pattern_rate = round(_corr_total / max(_corr_l5_fires, 1) * 100, 1) if _corr_l5_fires else None

        # MONITOR-051: Analogy suggestion rate — 24h count from metrics table
        _analogy_metric_row = conn.execute("""
            SELECT COALESCE(SUM(value), 0) as total_24h,
                   COUNT(*) as turns_24h
            FROM metrics
            WHERE metric = 'analogy_suggestions_count'
            AND timestamp > datetime('now', '-1 day')
        """).fetchone()
        _analogy_total_24h = int(_analogy_metric_row["total_24h"] or 0)
        _analogy_turns_24h = int(_analogy_metric_row["turns_24h"] or 0)

        # MONITOR-124: Recency score distribution — age distribution for post-processing
        _recency_age_rows = conn.execute("""
            SELECT (julianday('now') - julianday(created_at)) as age_days
            FROM concepts
            WHERE status = 'active' AND created_at IS NOT NULL
            ORDER BY age_days
        """).fetchall()

        # MONITOR-126: platform_hint distribution from sessions
        _platform_rows = conn.execute("""
            SELECT COALESCE(platform_hint, 'unknown') as ph, COUNT(*) as cnt
            FROM sessions
            GROUP BY ph
            ORDER BY cnt DESC
        """).fetchall()
        _platform_dist = {r["ph"]: r["cnt"] for r in _platform_rows}

        # MONITOR-070: Decay distribution by is_factual split
        _decay_rows = conn.execute("""
            SELECT
                CASE WHEN json_valid(data) AND json_extract(data, '$.metadata.is_factual') = 1
                     THEN 'factual' ELSE 'non_factual' END as kind,
                COUNT(*) as cnt,
                ROUND(AVG(COALESCE(currency_score, 0.0)), 4) as avg_score,
                ROUND(MIN(COALESCE(currency_score, 0.0)), 4) as min_score,
                ROUND(MAX(COALESCE(currency_score, 0.0)), 4) as max_score
            FROM concepts
            WHERE status = 'active' AND is_current = 1 AND currency_score IS NOT NULL
            GROUP BY kind
        """).fetchall()
        _decay_dist = {
            r["kind"]: {
                "count": r["cnt"],
                "avg_currency_score": r["avg_score"],
                "min_currency_score": r["min_score"],
                "max_currency_score": r["max_score"],
            }
            for r in _decay_rows
        }

    # MONITOR-041: canary window elapsed check
    import datetime as _dt

    import app.core.config as _cfg
    _canary_start = _dt.date.fromisoformat(getattr(_cfg, "EVOLUTION_CANARY_START_DATE", "2026-03-13"))
    _canary_elapsed = (_dt.date.today() - _canary_start).days
    _canary_window_passed = _canary_elapsed >= _cfg.EVOLUTION_CANARY_DURATION_DAYS

    # MONITOR-035: derive currency health alert status
    _curr_total = _curr_row["total"] or 0
    _contradicted = _curr_row["contradicted"] or 0
    _mean_score = _curr_row["mean_score"] or 0.0
    if _curr_total == 0:
        _curr_alert = "UNKNOWN"
        _contradicted_pct = 0.0
    else:
        _contradicted_pct = round(_contradicted / _curr_total * 100, 2)
        if _contradicted_pct > 50.0 or _mean_score < 0.5:
            _curr_alert = "CRITICAL"
        elif _contradicted_pct > 35.0 or _mean_score < 0.7:
            _curr_alert = "DEGRADED"
        else:
            _curr_alert = "HEALTHY"

    # MONITOR-124: compute recency score percentiles from age distribution
    import math as _math

    from app.core.config import RETRIEVAL_RECENCY_HALF_LIFE_DAYS as _hl_days
    _hl = max(0.1, _hl_days)
    _ages = [r["age_days"] for r in _recency_age_rows if r["age_days"] is not None and r["age_days"] >= 0]
    if _ages:
        _recency_scores_sorted = sorted([_math.exp(-_math.log(2) / _hl * a) for a in _ages])
        _n_rec = len(_recency_scores_sorted)
        _rec_mean = round(sum(_recency_scores_sorted) / _n_rec, 4)
        _rec_p50 = round(_recency_scores_sorted[_n_rec // 2], 4)
        _rec_p95 = round(_recency_scores_sorted[min(int(_n_rec * 0.95), _n_rec - 1)], 4)
    else:
        _rec_mean = _rec_p50 = _rec_p95 = None

    # MONITOR-073: KA canonical drift — count active concepts with non-canonical knowledge_area
    # KA-006: Use get_canonical_areas() (seed+established+mature from DB) instead of the
    # hardcoded _CANONICAL_KA_DESCRIPTIONS dict, which only covers embedding-classified KAs
    # and was missing architecture_gaps, ip_protection, product_operations, pith_benchmarks.
    from app.core.taxonomy_utils import get_canonical_areas as _get_canonical_areas  # DEBT-234
    _canonical_kas = _get_canonical_areas()
    _placeholders = ",".join("?" * len(_canonical_kas))
    with read_snapshot_db("get_pith_stats_aggregates") as _ka_conn:
        _ka_drift_row = _ka_conn.execute(
            f"""
            SELECT COUNT(*) as cnt FROM concepts
            WHERE is_current = 1
              AND status = 'active'
              AND knowledge_area NOT IN ({_placeholders})
            """,
            list(_canonical_kas),
        ).fetchone()
    _ka_non_canonical_count = _ka_drift_row["cnt"] if _ka_drift_row else 0

    # MONITOR-SESSION010: Compaction detection event count + tier distribution
    import json as _json
    with read_snapshot_db("get_pith_stats_aggregates") as _cmp_conn:
        _cmp_rows = _cmp_conn.execute(
            """SELECT details FROM governance_events
               WHERE event_type = 'compaction_reinjection'
               ORDER BY created_at DESC LIMIT 500"""
        ).fetchall()
    _cmp_total = len(_cmp_rows)
    _cmp_tier_dist: dict[str, int] = {}
    for _cmp_row in _cmp_rows:
        try:
            _cmp_det = _json.loads(_cmp_row["details"] or "{}")
            _tier = _cmp_det.get("detection_tier") or "pre_faf5f25"  # MONITOR-130: pre-bootstrap events lack tier
        except Exception:
            _tier = "pre_faf5f25"
        _cmp_tier_dist[_tier] = _cmp_tier_dist.get(_tier, 0) + 1

    return {
        "total_concepts": row["total_concepts"],
        "avg_confidence": round(row["avg_confidence"], 4),
        "avg_stability": round(row["avg_stability"], 4),
        "knowledge_areas": row["knowledge_areas"],
        "total_versions": versions_row["total"],
        "ka_breakdown": ka_breakdown,
        "orphan_concepts": orphan_row["cnt"],
        "evidence_stats": {
            "with_evidence": evidence_row["with_evidence"],
            "without_evidence": evidence_row["without_evidence"],
            "avg_evidence_per_concept": evidence_row["avg_evidence_count"],
        },
        # HEALTH-005: surface data quality in stats
        "data_quality": {
            "null_timestamps": data_quality_row["null_timestamps"],
            "bad_json": data_quality_row["bad_json"],
        },
        # MONITOR-CI032A-01: post-init migration integrity check. Alerts on
        # status == "CRITICAL" — a CRITICAL here means a column listed in
        # _COLUMN_MIGRATIONS did not land in the live schema (exactly the
        # silent-skip class of bug CI-032a fixed).
        "migration_integrity": migration_integrity,
        # MONITOR-003: engagement distribution histograms
        "reinforcement_count_distribution": rc_dist,
        "access_count_distribution": ac_dist,
        # MONITOR-004: temporal access recency distribution
        "last_accessed_distribution": la_dist,
        # MONITOR-029: governance sweep health
        "governance_sweep_24h": {
            "promotions": gov_24h_row["promotions_24h"] or 0,
            "quarantine_events": gov_24h_row["quarantine_24h"] or 0,
            "backfill_events": gov_24h_row["backfill_24h"] or 0,
            "total_events": _total_gov_24h,
            "zero_count_alert": _gov_sweep_alert,
        },
        # MEASURE-008: experiment concept efficacy
        "experiment_efficacy": {
            "experiment_concepts": exp_efficacy_row["total"] or 0,
            "avg_access_experiment": round(exp_efficacy_row["avg_access"] or 0, 2),
            "avg_access_regular": round(regular_access_row["avg_access"] or 0, 2),
            "avg_confidence": round(exp_efficacy_row["avg_conf"] or 0, 4),
            "retrieved_at_least_once": exp_efficacy_row["retrieved"] or 0,
            "retrieval_rate": round((exp_efficacy_row["retrieved"] or 0) / max(exp_efficacy_row["total"] or 0, 1), 4),
        },
        # MONITOR-036: score-range validation
        "score_range_validation": {
            "authority_out_of_range": oor_row["auth_oor"] or 0,
            "effective_authority_out_of_range": oor_row["eff_auth_oor"] or 0,
            "confidence_out_of_range": oor_row["conf_oor"] or 0,
        },
        # MONITOR-038: always-activate concept count
        "always_activate_count": always_activate_row["cnt"] or 0,
        # MONITOR-033: ka_relative_authority coverage
        "ka_relative_authority_coverage": {
            "total_active": ka_ra_row["total"] or 0,
            "with_ka_ra": ka_ra_row["with_ka_ra"] or 0,
            "coverage_pct": round((ka_ra_row["with_ka_ra"] or 0) / max(ka_ra_row["total"] or 0, 1) * 100, 1),
        },
        # MONITOR-041: canary window elapsed
        "evolution_canary": {
            "mode": getattr(_cfg, "EVOLUTION_CANARY_MODE", True),
            "elapsed_days": _canary_elapsed,
            "window_days": _cfg.EVOLUTION_CANARY_DURATION_DAYS,
            "window_passed": _canary_window_passed,
        },
        # MONITOR-024: reflection cycle analytics
        "reflection_tracking": {
            "total_cycles": rt_total_row["total"] or 0,
            "total_timeouts": rt_total_row["total_timeouts"] or 0,
            "total_auto_closed": rt_total_row["total_auto_closed"] or 0,
            "avg_concepts_returned": round(rt_total_row["avg_concepts_returned"] or 0, 2),
            "timeout_rate": round(
                (rt_total_row["total_timeouts"] or 0) / max(rt_total_row["total"] or 0, 1), 4
            ),
            "by_trigger_type": _rt_by_trigger,
        },
        # MONITOR-032: zombie concept count
        "zombie_count": zombie_row["cnt"] if zombie_row else 0,
        # MONITOR-011: index consistency — active vs superseded breakdown
        # MONITOR-060: superseded_pct alert thresholds (warn>70%, critical>85%)
        "index_consistency": {
            "active_current": index_consistency_row["active_current"] or 0,
            "active_superseded": index_consistency_row["active_superseded"] or 0,
            "total_all": index_consistency_row["total_all"] or 0,
            "superseded_pct": round(
                (index_consistency_row["active_superseded"] or 0)
                / max(index_consistency_row["total_all"] or 0, 1)
                * 100,
                1,
            ),
            "alert_level": (
                "critical"
                if round(
                    (index_consistency_row["active_superseded"] or 0)
                    / max(index_consistency_row["total_all"] or 0, 1) * 100, 1
                ) > 85.0
                else "warn"
                if round(
                    (index_consistency_row["active_superseded"] or 0)
                    / max(index_consistency_row["total_all"] or 0, 1) * 100, 1
                ) > 70.0
                else "ok"
            ),
        },
        # MONITOR-013: FIX-1 effectiveness — resurrected zombie count (should be 0)
        "fix1_zombie_alert": {
            "resurrected_count": fix1_zombie_row["cnt"] if fix1_zombie_row else 0,
            "alert": (fix1_zombie_row["cnt"] or 0) > 0,
        },
        # MONITOR-035: currency health alert
        "currency_health": {
            "status": _curr_alert,
            "total_is_current": _curr_total,
            "contradicted_pct": _contradicted_pct,
            "mean_currency_score": round(_mean_score, 4),
            "thresholds": {
                "degraded_if_contradicted_pct_above": 35.0,
                "critical_if_contradicted_pct_above": 50.0,
                "degraded_if_mean_score_below": 0.7,
                "critical_if_mean_score_below": 0.5,
            },
        },        # MONITOR-069: Factual coverage metrics
        "factual_coverage": {
            "total_active": _factual_total,
            "factual_count": _factual_count,
            "factual_rate_pct": _factual_rate,
            "has_valid_from": _has_valid_from,
            "valid_from_coverage_pct": round(_has_valid_from / max(_factual_count, 1) * 100, 1),
            "status": "healthy" if 20 <= _factual_rate <= 40 else ("low" if _factual_rate < 20 else "high"),
        },

        # MONITOR-049: CTX-007 compaction survival format health
        # MONITOR-049 + ARGUS-S23-F1: CSF health + token budget metrics
        "compaction_survival": {
            "flag_enabled": _csf_enabled,
            "eligible_concepts": _csf_row["eligible"] or 0,
            "formatter_live": _csf_probe_ok,
            "avg_summary_chars": _csf_avg_chars,
            "estimated_tokens_overhead": _csf_est_tokens,
            "status": "disabled" if not _csf_enabled else (
                "healthy" if _csf_probe_ok else (
                    "no_eligible_concepts" if (_csf_row["eligible"] or 0) == 0
                    else "formatter_broken"
                )
            ),
        },
        # ARGUS-S25-F2: experiment_generation task health
        "experiment_generation": _exp_gen_info,
        # MONITOR-056: Cross-KA guard activation rate (24h)
        "cross_ka_guard_rate": {
            "activations_24h": _cross_ka_24h,
        },
        # MONITOR-034: Stuck PROVISIONAL monitoring
        "stuck_provisional": {
            "stuck_count": _stuck_prov,
            "total_provisional": _total_prov,
            "stuck_pct": round(_stuck_prov / max(_total_prov, 1) * 100, 1),
            "alert": _stuck_prov > 200,
        },
        # MONITOR-018: Stale session buildup
        "stale_sessions": {
            "count": _stale_sessions,
            "alert": _stale_sessions > 10,
        },
        # MONITOR-009: Context pressure trend
        "pressure_trend": {
            "avg_24h": _p_avg_24h,
            "high_pressure_sessions_24h": _p_high_24h,
            "avg_prior_7d": _p_avg_7d,
            "trend": _p_trend,
            "alert": (_p_avg_24h is not None and _p_avg_24h > 0.6) or _p_high_24h > 3,
        },
        # MONITOR-058: Episode count
        "episode_count": _episode_count,
        # MONITOR-031/042: Association and adjacency cache hit/miss counters
        "association_cache_stats": {
            "hits": _concepts_mod._assoc_cache_hits,
            "misses": _concepts_mod._assoc_cache_misses,
            "hit_rate_pct": round(
                _concepts_mod._assoc_cache_hits / max(_concepts_mod._assoc_cache_hits + _concepts_mod._assoc_cache_misses, 1) * 100, 1
            ),
        },
        "adjacency_cache_stats": {
            "hits": _concepts_mod._adjacency_cache_hits,
            "misses": _concepts_mod._adjacency_cache_misses,
            "hit_rate_pct": round(
                _concepts_mod._adjacency_cache_hits / max(_concepts_mod._adjacency_cache_hits + _concepts_mod._adjacency_cache_misses, 1) * 100, 1
            ),
        },
        # MONITOR-044: PSIS M3 compliance alert
        "psis_m3_compliance": {
            "quarantined_over_cap": _psis_overcap,
            "cap_threshold": 0.4,
            "alert": _psis_overcap > 0,
        },
        # MONITOR-051: Analogy suggestion rate (24h window from metrics table)
        "analogy_suggestion_rate": {
            "total_suggestions_24h": _analogy_total_24h,
            "turns_with_suggestions_24h": _analogy_turns_24h,
        },
        # MONITOR-070: Currency score decay distribution by is_factual
        "decay_distribution": _decay_dist,
        # MONITOR-124: recency score distribution (exponential decay from created_at age)
        "recency_score_distribution": {
            "mean": _rec_mean,
            "p50": _rec_p50,
            "p95": _rec_p95,
            "half_life_days": _hl,
            "sample_count": len(_ages) if _ages else 0,
        },
        # MONITOR-126: platform_hint distribution from sessions
        "platform_hint_distribution": _platform_dist,
        # MONITOR-073: KA canonical drift — active concepts with non-canonical knowledge_area
        "ka_canonical_drift": {
            "non_canonical_count": _ka_non_canonical_count,
            "alert": _ka_non_canonical_count > 0,
        },
        # MONITOR-SESSION010: Compaction detection events (count + tier distribution)
        "compaction_detection": {
            "total_events": _cmp_total,
            "tier_distribution": _cmp_tier_dist,
        },
        # MONITOR-128: Correction pipeline stats
        "correction_pipeline": {
            "total_corrections": _corr_total,
            "avg_detection_confidence": _corr_avg_conf,
            "total_affected_concepts": _corr_total_affected,
            "evidence_appends": _corr_appends,
            "gated_count": _corr_gated,
            "pattern_match_rate_pct": _corr_pattern_rate,
        },
    }


def get_memory_projection_data() -> dict:
    """Compute growth velocity, per-KA health, and capacity projection.

    HEALTH-002: Answers "what will be" vs pith_stats "what is".

    Returns dict with:
      - growth_velocity: concepts/day over recent windows (7d, 14d, 30d)
      - ka_velocity: per-KA growth/decay rates (last 7d vs previous 7d)
      - maturity_flow: maturity transitions
      - capacity_projection: at current rate, when do we hit capacity thresholds
      - retrieval_activity: conversation turns per day (last 14d)
    """
    with read_snapshot_db("get_memory_projection_data") as conn:
        c = conn.cursor()
        result = {}

        # Growth velocity: concepts created per day over windows
        for window_label, days in [("7d", 7), ("14d", 14), ("30d", 30)]:
            c.execute(
                "SELECT COUNT(*) FROM concepts WHERE created_at > datetime('now', ?)",
                (f"-{days} days",),
            )
            count = c.fetchone()[0]
            result.setdefault("growth_velocity", {})[window_label] = {
                "total_created": count,
                "per_day": round(count / days, 1),
            }

        # Per-KA velocity: compare last 7d vs previous 7d
        c.execute("""
            SELECT knowledge_area,
                   SUM(CASE WHEN created_at > datetime('now', '-7 days') THEN 1 ELSE 0 END) as recent,
                   SUM(CASE WHEN created_at BETWEEN datetime('now', '-14 days')
                       AND datetime('now', '-7 days') THEN 1 ELSE 0 END) as previous
            FROM concepts
            GROUP BY knowledge_area
            HAVING recent > 0 OR previous > 0
            ORDER BY recent DESC
        """)
        ka_velocity = []
        for row in c.fetchall():
            ka, recent, previous = row[0], row[1], row[2]
            delta = recent - previous
            direction = "growing" if delta > 0 else "shrinking" if delta < 0 else "stable"
            ka_velocity.append(
                {
                    "knowledge_area": ka,
                    "last_7d": recent,
                    "prev_7d": previous,
                    "delta": delta,
                    "direction": direction,
                }
            )
        result["ka_velocity"] = ka_velocity

        # Maturity distribution + recent changes
        c.execute("SELECT maturity, COUNT(*) FROM concepts GROUP BY maturity")
        result["maturity_distribution"] = {row[0]: row[1] for row in c.fetchall()}

        c.execute("""
            SELECT maturity, COUNT(*) FROM concepts
            WHERE updated_at > datetime('now', '-7 days')
              AND maturity IN ('ESTABLISHED', 'PROVISIONAL')
            GROUP BY maturity
        """)
        result["recent_maturity_changes"] = {row[0]: row[1] for row in c.fetchall()}

        # Capacity projection: linear extrapolation
        c.execute("SELECT COUNT(*) FROM concepts")
        total = c.fetchone()[0]
        daily_rate = result["growth_velocity"]["7d"]["per_day"]

        CAPACITY_THRESHOLDS = [5000, 10000, 25000, 50000]
        projections = {}
        for threshold in CAPACITY_THRESHOLDS:
            if total >= threshold:
                projections[str(threshold)] = "already_reached"
            elif daily_rate > 0:
                days_to_reach = round((threshold - total) / daily_rate, 0)
                from datetime import datetime, timedelta

                est_date = (datetime.now(UTC) + timedelta(days=days_to_reach)).strftime("%Y-%m-%d")
                projections[str(threshold)] = {
                    "days_from_now": int(days_to_reach),
                    "estimated_date": est_date,  # DEBT-097: computed here instead of deferred
                }
            else:
                projections[str(threshold)] = "no_growth"
        result["capacity_projection"] = projections

        # Retrieval pressure: conversation turns per day
        c.execute("""
            SELECT DATE(created_at) as d, COUNT(*) FROM governance_events
            WHERE event_type = 'conversation_turn_complete'
              AND created_at > datetime('now', '-14 days')
            GROUP BY d ORDER BY d
        """)
        result["retrieval_activity"] = [{"date": row[0], "turns": row[1]} for row in c.fetchall()]

        return result


def get_distribution_report() -> dict:
    """Compute distribution statistics for all 7 retrieval blend factors.

    MEASURE-005 diagnostic: tracks whether inputs are discriminative.
    Returns per-factor: histogram (10 buckets), mean, stddev, % in dominant bucket.
    """
    with read_snapshot_db("get_distribution_report") as conn:
        c = conn.cursor()
        report = {}

        def _column_dist(col_name: str, where: str = "") -> dict:
            where_clause = f"WHERE {where}" if where else ""
            # Use CAST to bucket into 0.1 ranges
            c.execute(f"""
                SELECT CAST({col_name} * 10 AS INTEGER) / 10.0 as bucket, COUNT(*)
                FROM concepts {where_clause}
                GROUP BY bucket ORDER BY bucket
            """)
            rows = c.fetchall()
            if not rows:
                return {"histogram": {}, "mean": 0, "stddev": 0, "dominant_bucket_pct": 0, "discriminative": False}

            histogram = {}
            values = []
            for val, cnt in rows:
                bucket_key = f"{val:.1f}" if val is not None else "NULL"
                histogram[bucket_key] = cnt
                if val is not None:
                    values.extend([float(val)] * cnt)

            total = sum(histogram.values())
            dominant_pct = max(histogram.values()) / total * 100 if total > 0 else 0
            mean = sum(values) / len(values) if values else 0
            variance = sum((v - mean) ** 2 for v in values) / len(values) if values else 0

            return {
                "histogram": histogram,
                "count": total,
                "mean": round(mean, 4),
                "stddev": round(math.sqrt(variance), 4),
                "dominant_bucket_pct": round(dominant_pct, 1),
                "discriminative": dominant_pct < 50,
            }

        report["confidence"] = _column_dist("confidence")
        report["stability"] = _column_dist("stability")
        report["authority_score"] = _column_dist("authority_score", where="authority_score IS NOT NULL")
        report["currency_score"] = _column_dist("currency_score", where="currency_score IS NOT NULL")

        # Summary
        discriminative_count = sum(1 for f in report.values() if isinstance(f, dict) and f.get("discriminative", False))
        report["summary"] = {
            "factors_measured": 4,
            "factors_discriminative": discriminative_count,
            "factors_collapsed": 4 - discriminative_count,
            "note": "context_boost and goal_boost are query-time; emb_score varies per query",
        }

        return report

def analyze_session_drops() -> dict:
    """Analyze session drop rate with full taxonomy and pressure correlation.

    Returns mutually exclusive session categories that sum to total sessions,
    plus pressure × drop-rate cross-tabulation for trend tracking.

    Cold-path only — called via checkpoint endpoint, not conversation_turn.
    """
    with read_snapshot_db("analyze_session_drops") as conn:
        total = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        if total == 0:
            return {"total_sessions": 0, "taxonomy": {}, "pressure_correlation": {}}

        # --- Taxonomy (mutually exclusive, exhaustive) ---
        taxonomy = {}

        taxonomy["quick_lookup"] = conn.execute(
            "SELECT COUNT(*) FROM sessions "
            "WHERE status = 'ended' AND learning_event_count = 0 "
            "AND (julianday(ended_at) - julianday(started_at)) * 1440 < 1"
        ).fetchone()[0]

        taxonomy["warmup_abandoned"] = conn.execute(
            "SELECT COUNT(*) FROM sessions "
            "WHERE status = 'ended' AND learning_event_count = 0 "
            "AND (julianday(ended_at) - julianday(started_at)) * 1440 BETWEEN 1 AND 10"
        ).fetchone()[0]

        taxonomy["brief_exchange"] = conn.execute(
            "SELECT COUNT(*) FROM sessions "
            "WHERE status = 'ended' AND learning_event_count BETWEEN 1 AND 2 "
            "AND (julianday(ended_at) - julianday(started_at)) * 1440 < 5"
        ).fetchone()[0]

        taxonomy["engaged_low_learn"] = conn.execute(
            "SELECT COUNT(*) FROM sessions "
            "WHERE status = 'ended' AND learning_event_count BETWEEN 1 AND 2 "
            "AND (julianday(ended_at) - julianday(started_at)) * 1440 BETWEEN 5 AND 10"
        ).fetchone()[0]

        taxonomy["lost_work"] = conn.execute(
            "SELECT COUNT(*) FROM sessions "
            "WHERE status = 'ended' AND learning_event_count <= 2 "
            "AND (julianday(ended_at) - julianday(started_at)) * 1440 > 10"
        ).fetchone()[0]

        taxonomy["interrupted"] = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE status = 'interrupted'"
        ).fetchone()[0]

        taxonomy["healthy"] = conn.execute(
            "SELECT COUNT(*) FROM sessions "
            "WHERE status = 'ended' AND learning_event_count > 2"
        ).fetchone()[0]

        taxonomy["active"] = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE status = 'active'"
        ).fetchone()[0]

        # Derived metrics
        dropped = sum(v for k, v in taxonomy.items() if k not in ("healthy", "active"))
        checkpoint_beneficiaries = taxonomy["lost_work"] + taxonomy["interrupted"]

        # --- Pressure × drop correlation ---
        pressure_corr = {}
        for label, lo, hi in [
            ("high", 0.25, 999.0),
            ("medium", 0.10, 0.25),
            ("low", 0.0, 0.10),
        ]:
            row = conn.execute(
                "SELECT "
                "  SUM(CASE WHEN learning_event_count <= 2 THEN 1 ELSE 0 END), "
                "  SUM(CASE WHEN learning_event_count > 2 THEN 1 ELSE 0 END) "
                "FROM sessions "
                "WHERE pressure_score IS NOT NULL "
                "AND pressure_score >= ? AND pressure_score < ?",
                (lo, hi),
            ).fetchone()
            d, a = row[0] or 0, row[1] or 0
            pressure_corr[label] = {
                "dropped": d,
                "active": a,
                "drop_rate": round(d / (d + a) * 100, 1) if (d + a) > 0 else None,
            }

    return {
        "total_sessions": total,
        "taxonomy": taxonomy,
        "drop_rate_pct": round(dropped / total * 100, 1),
        "checkpoint_beneficiaries": checkpoint_beneficiaries,
        "checkpoint_beneficiary_pct": round(checkpoint_beneficiaries / total * 100, 1),
        "benign_drop_pct": round((dropped - checkpoint_beneficiaries) / total * 100, 1),
        "pressure_correlation": pressure_corr,
        "generated_at": _utc_now_iso(),
    }


def analyze_coverage_threshold(candidate_thresholds: list[float] | None = None) -> dict:
    """BENCH-015: Analyze coverage_score distribution against candidate thresholds.

    Runs an eligibility sweep: for each threshold, computes the percentage of
    queries that would pass/fail. Helps calibrate COVERAGE_RELEVANCE_THRESHOLD.

    Args:
        candidate_thresholds: List of thresholds to test. Defaults to
            [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    """
    import json as _at_json

    if candidate_thresholds is None:
        candidate_thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    with read_snapshot_db("analyze_coverage_threshold") as conn:
        rows = conn.execute(
            "SELECT details FROM governance_events WHERE event_type='coverage_score_recorded'"
        ).fetchall()

    scores = []
    for row in rows:
        try:
            d = _at_json.loads(row[0])
            cs = d.get("coverage_score")
            if cs is not None:
                scores.append(cs)
        except Exception:
            pass

    if not scores:
        return {"error": "No coverage_score data recorded yet. Run some conversation_turns first."}

    total = len(scores)
    mean = sum(scores) / total
    results = {}
    for thresh in candidate_thresholds:
        above = len([s for s in scores if s >= thresh])
        results[str(thresh)] = {
            "above_count": above,
            "above_pct": round(above / total * 100, 1),
            "below_count": total - above,
            "below_pct": round((total - above) / total * 100, 1),
        }

    return {
        "total_samples": total,
        "mean": round(mean, 4),
        "median": round(sorted(scores)[total // 2], 4),
        "std_dev": round((sum((s - mean) ** 2 for s in scores) / total) ** 0.5, 4),
        "threshold_sweep": results,
        "current_threshold": 0.35,
        "recommendation": None,  # Populated after sufficient data (>100 samples)
    }
