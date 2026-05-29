"""COGGOV-014: Criteria-based stale-risk detector."""

from __future__ import annotations

import importlib
import json
import logging
import re
from datetime import datetime


# CI-029: Lazy loaders for cognitive + retrieval dependencies.
# Avoids governance→cognitive (Contract 4) and governance→retrieval (Contract 4)
# static imports. Same pattern as DEBT-237..240 precedents.
def _get_extract_factual_reference_hits():
    """Lazy loader — avoids governance→cognitive static import (Contract 4)."""
    return importlib.import_module("app.cognitive.staleness").extract_factual_reference_hits


def _get_classify_evidence_source():
    """Lazy loader — avoids governance→retrieval static import (Contract 4)."""
    return importlib.import_module("app.retrieval.provenance").classify_evidence_source


from app.core.config import (
    STALE_RISK_CONSECUTIVE_REVIEW_HITS,
    STALE_RISK_DETECTOR_ENABLED,
    STALE_RISK_DETECTOR_VERSION,
    STALE_RISK_HOT_ACCESS_DAYS,
    STALE_RISK_MAX_PROMOTIONS_PER_RUN,
    STALE_RISK_MIN_ACCESS_COUNT,
    STALE_RISK_THRESHOLD_AGING,
    STALE_RISK_THRESHOLD_REVIEW,
    STALE_RISK_TYPE_WINDOWS,
)
from app.core.constants import (
    GOV_EVENT_STALE_RISK_CLEARED,
    GOV_EVENT_STALE_RISK_RUN_COMPLETE,
    GOV_EVENT_STALE_RISK_STATE_CHANGED,
)
from app.core.datetime_utils import _ensure_aware, _utc_now, _utc_now_iso
from app.storage.concepts import update_stale_risk_fields

logger = logging.getLogger(__name__)

_STATE_ORDER = {None: 0, "AGING": 1, "REVIEW": 2}
_HIGH_RISK_KNOWLEDGE_AREAS = frozenset(
    {"project_status", "implementation", "operations", "product_strategy", "competitive_analysis"}
)
_DURABLE_KNOWLEDGE_AREAS = frozenset({"architecture", "process", "testing"})
_FRAMEWORK_HEAVY_KNOWLEDGE_AREAS = frozenset(
    {"architecture", "process", "testing", "review_methodology"}
)
_GUIDANCE_HEAVY_KNOWLEDGE_AREAS = frozenset(
    {"architecture", "specification", "review_methodology", "testing", "security"}
)
_TEMPORAL_STATE_MARKERS = (
    "phase",
    "sprint",
    "baseline",
    "milestone",
    "priority",
    "roadmap",
    "rollout",
    "migration",
    "transition",
    "next move",
    "next step",
    "current status",
    "in progress",
    "planned",
    "pending",
    "complete",
    "completed",
    "shipped",
    "launched",
    "as of",
)
_STRONG_TEMPORAL_STATE_MARKERS = (
    "phase",
    "sprint",
    "baseline",
    "milestone",
    "priority",
    "priority stack",
    "roadmap",
    "rollout",
    "migration",
    "transition",
    "next move",
    "next step",
    "current status",
    "as of",
    "gated by",
)
_TEMPORAL_STATE_REGEXES = (
    re.compile(r"\bphase[\s_-]*[0-9a-z]+\b", re.I),
    re.compile(r"\bsprint[\s_-]*\d+\b", re.I),
    re.compile(r"\bq[1-4]\b", re.I),
)
_DATE_REFERENCE_REGEXES = (
    re.compile(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b", re.I),
    re.compile(r"\b20\d{2}\b"),
)
_DOC_FILENAME_REGEX = re.compile(r"\b[A-Z0-9][A-Z0-9_.-]*\.(?:md|doc|docx|pdf)\b")
_DURABLE_CLAIM_MARKERS = (
    "architecture",
    "protocol",
    "taxonomy",
    "support matrix",
    "intel mac",
    "supports",
    "supported",
    "requires",
    "pins",
    "install",
    "installer",
    "resolution",
    "implements",
)
_DURABLE_FRAMEWORK_MARKERS = (
    "principle:",
    "principle ",
    "universal principle",
    "framework",
    "methodology",
    "meta-pattern",
    "pattern:",
    "best practice",
    "best practices",
    "role separation",
    "review depth",
    "roi quantified",
)
_DATED_SNAPSHOT_MARKERS = (
    "baseline",
    "competitive window",
    "competitive landscape",
    "readiness assessment",
    "final assessment",
    "gap analysis",
    "retrospective",
    "strategic positions",
    "positioning re-evaluation",
    "snapshot",
)
_DESIGN_GUIDANCE_MARKERS = (
    "analytical lens",
    "backlog",
    "delivery chain",
    "design claim",
    "design review",
    "gauntlet",
    "guide",
    "matrix",
    "must be updated",
    "policy",
    "priority dependency",
    "requirements",
    "semantic versioning",
    "threshold",
    "verification plan",
    "versioning",
)
_HISTORICAL_ARTIFACT_MARKERS = (
    "design document",
    "specification",
    "spec ",
    "strategic document suite",
    "migration guide",
    "single source of truth",
    "approved for implementation",
    "companion to",
    "post-adversarial review",
    "files changed",
    "insertions",
    "tests passing",
    "committed to",
)
_ARTIFACT_PROVENANCE_MARKERS = _HISTORICAL_ARTIFACT_MARKERS + (
    "approved",
    "revision",
    "archive",
    "archived",
    "guide",
    "docs/",
    "specs/",
    ".md",
    ".doc",
    ".docx",
    ".pdf",
)
_CODE_ANCHOR_PATH_MARKERS = (
    "app/",
    "tests/",
    "scripts/",
    "server.js",
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
)
_RUNTIME_INCIDENT_MARKERS = (
    "broke",
    "broken",
    "built successfully",
    "error",
    "failed",
    "failure",
    "ghost",
    "incident",
    "migration integrity gap",
    "outage",
    "regression",
    "unreachable",
    "wrong import",
)
_RUNTIME_STATE_EXEMPT_KNOWLEDGE_AREAS = frozenset(
    {"project_status", "implementation", "operations", "debugging"}
)


def _days_since(ts: str | None) -> float:
    if not ts:
        return 0.0
    try:
        parsed = _ensure_aware(datetime.fromisoformat(str(ts).replace("Z", "+00:00")))
        return max(0.0, (_utc_now() - parsed).total_seconds() / 86400.0)
    except Exception:
        return 0.0


def _json_details(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True)


def _record_event(conn, event_type: str, concept_id: str | None, details: dict) -> None:
    conn.execute(
        """INSERT INTO governance_events
           (event_type, concept_id, details, created_at)
           VALUES (?, ?, ?, ?)""",
        (event_type, concept_id, json.dumps(details, sort_keys=True), _utc_now_iso()),
    )


def _has_temporal_state_signal(summary: str, knowledge_area: str) -> bool:
    summary_lower = summary.lower()
    has_marker = any(marker in summary_lower for marker in _TEMPORAL_STATE_MARKERS)
    has_strong_marker = any(marker in summary_lower for marker in _STRONG_TEMPORAL_STATE_MARKERS)
    has_phase_like_regex = any(pattern.search(summary) for pattern in _TEMPORAL_STATE_REGEXES)
    has_date_reference = any(pattern.search(summary) for pattern in _DATE_REFERENCE_REGEXES)

    if knowledge_area in _DURABLE_KNOWLEDGE_AREAS:
        return has_strong_marker or has_phase_like_regex
    if has_marker or has_phase_like_regex:
        return True
    return has_date_reference and has_strong_marker


def _has_durable_claim_signal(summary: str, knowledge_area: str) -> bool:
    summary_lower = summary.lower()
    if knowledge_area in _DURABLE_KNOWLEDGE_AREAS:
        return True
    return any(marker in summary_lower for marker in _DURABLE_CLAIM_MARKERS)


def _has_historical_artifact_signal(summary: str) -> bool:
    summary_lower = summary.lower()
    return any(marker in summary_lower for marker in _HISTORICAL_ARTIFACT_MARKERS) or bool(
        _DOC_FILENAME_REGEX.search(summary)
    )


def _has_durable_framework_signal(summary: str, knowledge_area: str) -> bool:
    summary_lower = summary.lower()
    if knowledge_area not in _FRAMEWORK_HEAVY_KNOWLEDGE_AREAS:
        return False
    return any(marker in summary_lower for marker in _DURABLE_FRAMEWORK_MARKERS)


def _has_dated_snapshot_signal(summary: str, knowledge_area: str) -> bool:
    summary_lower = summary.lower()
    if knowledge_area not in {
        "architecture",
        "competitive_analysis",
        "product_strategy",
        "business_strategy",
        "operations",
        "process",
        "specification",
        "system_quality",
    }:
        return False
    has_snapshot_marker = any(marker in summary_lower for marker in _DATED_SNAPSHOT_MARKERS)
    has_date_reference = any(pattern.search(summary) for pattern in _DATE_REFERENCE_REGEXES)
    return has_snapshot_marker and has_date_reference


def _has_design_guidance_signal(summary: str, knowledge_area: str) -> bool:
    summary_lower = summary.lower()
    if knowledge_area not in _GUIDANCE_HEAVY_KNOWLEDGE_AREAS:
        return False
    if any(marker in summary_lower for marker in _RUNTIME_INCIDENT_MARKERS):
        return False
    return any(marker in summary_lower for marker in _DESIGN_GUIDANCE_MARKERS)


def _semantic_class(
    *,
    knowledge_area: str,
    temporal_state_signal: bool,
    durable_framework_signal: bool,
    historical_artifact_signal: bool,
    dated_snapshot_signal: bool,
    design_guidance_signal: bool,
) -> str:
    if durable_framework_signal and not temporal_state_signal:
        return "durable_framework"
    if historical_artifact_signal and not dated_snapshot_signal and knowledge_area not in _RUNTIME_STATE_EXEMPT_KNOWLEDGE_AREAS:
        return "historical_artifact"
    if dated_snapshot_signal:
        return "dated_snapshot"
    if design_guidance_signal:
        return "design_guidance"
    if temporal_state_signal:
        return "stale_state"
    return "generic_old_hot"


def _load_concept_data(raw_data: str | None) -> dict:
    if not raw_data:
        return {}
    try:
        data = json.loads(raw_data)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _has_code_anchor(evidence_item: dict) -> bool:
    joined = " ".join(
        str(evidence_item.get(field, "") or "")
        for field in ("file_path", "source_reference")
    ).lower()
    return any(marker in joined for marker in _CODE_ANCHOR_PATH_MARKERS)


def _has_artifact_anchor(text: str) -> bool:
    text_lower = text.lower()
    return any(marker in text_lower for marker in _ARTIFACT_PROVENANCE_MARKERS)


def _extract_provenance_features(raw_data: str | None) -> dict:
    data = _load_concept_data(raw_data)
    evidence = data.get("evidence", []) if isinstance(data.get("evidence"), list) else []
    metadata = data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {}

    classified_sources = []
    document_like_evidence = 0
    artifact_like_evidence = 0
    code_anchored_evidence = 0

    _classify_evidence_source = _get_classify_evidence_source()  # CI-029: cached lazy resolution
    for item in evidence:
        if not isinstance(item, dict):
            continue
        classified = _classify_evidence_source(item)
        classified_sources.append(classified)
        joined = " ".join(
            str(item.get(field, "") or "")
            for field in ("content", "file_path", "source_reference")
        )
        if classified == "document_extracted":
            document_like_evidence += 1
        if _has_artifact_anchor(joined):
            artifact_like_evidence += 1
        if _has_code_anchor(item):
            code_anchored_evidence += 1

    metadata_extraction_source = metadata.get("extraction_source")
    provenance_available = bool(evidence or metadata_extraction_source)
    provenance_artifact_signal = document_like_evidence > 0 and artifact_like_evidence > 0

    return {
        "provenance_available": provenance_available,
        "metadata_extraction_source": metadata_extraction_source,
        "classified_sources": [source for source in classified_sources if source != "unclassified"],
        "document_like_evidence_count": document_like_evidence,
        "artifact_like_evidence_count": artifact_like_evidence,
        "code_anchored_evidence_count": code_anchored_evidence,
        "provenance_artifact_signal": provenance_artifact_signal,
        "code_anchor_signal": code_anchored_evidence > 0,
    }


def _score_candidate(row) -> tuple[float, dict]:
    knowledge_area = row["knowledge_area"] or ""
    summary = row["summary"] or ""
    concept_type = row["concept_type"]
    type_window_days = STALE_RISK_TYPE_WINDOWS[concept_type]
    content_age_days = _days_since(
        row["content_updated_at"] or row["updated_at"] or row["created_at"]
    )
    organic_access_age_days = _days_since(
        row["last_organic_access"] or row["last_accessed"] or row["created_at"]
    )
    factual_hits = _get_extract_factual_reference_hits()(summary)  # CI-029: lazy inline resolution
    temporal_state_signal = _has_temporal_state_signal(summary, knowledge_area)
    durable_claim_signal = _has_durable_claim_signal(summary, knowledge_area)
    durable_framework_signal = _has_durable_framework_signal(summary, knowledge_area)
    provenance_features = _extract_provenance_features(row["data"])
    summary_artifact_signal = _has_historical_artifact_signal(summary)
    dated_snapshot_signal = _has_dated_snapshot_signal(summary, knowledge_area)
    design_guidance_signal = _has_design_guidance_signal(summary, knowledge_area)
    historical_artifact_signal = provenance_features["provenance_artifact_signal"] or (
        summary_artifact_signal and not provenance_features["code_anchor_signal"]
    )
    semantic_class = _semantic_class(
        knowledge_area=knowledge_area,
        temporal_state_signal=temporal_state_signal,
        durable_framework_signal=durable_framework_signal,
        historical_artifact_signal=historical_artifact_signal,
        dated_snapshot_signal=dated_snapshot_signal,
        design_guidance_signal=design_guidance_signal,
    )
    generic_old_hot_reinforced = bool(
        factual_hits
        or provenance_features["code_anchor_signal"]
        or row["currency_status"] == "CONTESTED"
    )

    age_component = max(0.0, (content_age_days - type_window_days) / type_window_days)
    heat_component = 1.0 if (
        organic_access_age_days <= STALE_RISK_HOT_ACCESS_DAYS
        and (row["access_count"] or 0) >= STALE_RISK_MIN_ACCESS_COUNT
        and content_age_days > type_window_days * 1.25
    ) else 0.0
    factual_component = 1.0 if factual_hits else 0.0
    status_component = 0.25 if row["currency_status"] == "CONTESTED" else 0.0

    base_score = min(
        1.0,
        0.55 * min(age_component, 1.0)
        + 0.15 * heat_component
        + 0.20 * factual_component
        + 0.10 * status_component,
    )
    score = base_score

    # Durable principles/frameworks and doc artifacts should not share a
    # stale-state lane with dated snapshots and stale execution state.
    if semantic_class == "durable_framework":
        score *= 0.25
    elif semantic_class == "historical_artifact":
        score *= 0.40
    elif semantic_class == "design_guidance":
        score *= 0.40
    elif semantic_class == "generic_old_hot" and not generic_old_hot_reinforced:
        score *= 0.80
    elif knowledge_area in _DURABLE_KNOWLEDGE_AREAS and not temporal_state_signal:
        score *= 0.45
    elif durable_claim_signal and not temporal_state_signal:
        score *= 0.65

    features = {
        "knowledge_area": knowledge_area or "unknown",
        "type_window_days": type_window_days,
        "content_age_days": round(content_age_days, 2),
        "organic_access_age_days": round(organic_access_age_days, 2),
        "access_count": row["access_count"] or 0,
        "factual_hit_count": len(factual_hits),
        "temporal_state_signal": temporal_state_signal,
        "durable_claim_signal": durable_claim_signal,
        "durable_framework_signal": durable_framework_signal,
        "design_guidance_signal": design_guidance_signal,
        "dated_snapshot_signal": dated_snapshot_signal,
        "summary_artifact_signal": summary_artifact_signal,
        "historical_artifact_signal": historical_artifact_signal,
        "semantic_class": semantic_class,
        "generic_old_hot_reinforced": generic_old_hot_reinforced,
        "provenance_available": provenance_features["provenance_available"],
        "provenance_artifact_signal": provenance_features["provenance_artifact_signal"],
        "code_anchor_signal": provenance_features["code_anchor_signal"],
        "document_like_evidence_count": provenance_features["document_like_evidence_count"],
        "artifact_like_evidence_count": provenance_features["artifact_like_evidence_count"],
        "classified_sources": provenance_features["classified_sources"],
        "metadata_extraction_source": provenance_features["metadata_extraction_source"],
        "currency_status": row["currency_status"] or "ACTIVE",
        "score": round(score, 4),
    }
    return score, features


def _supports_review_promotion(semantic_class: str) -> bool:
    return semantic_class in {"stale_state", "dated_snapshot"}


def _target_state(
    score: float,
    current_state: str | None,
    current_hits: int,
    semantic_class: str,
) -> tuple[str | None, int]:
    if score < STALE_RISK_THRESHOLD_AGING:
        return None, 0
    if score >= STALE_RISK_THRESHOLD_REVIEW:
        return "REVIEW", max(1, current_hits if current_state == "REVIEW" else current_hits + 1)

    aging_hits = current_hits + 1 if current_state == "AGING" else 1
    if (
        current_state == "AGING"
        and aging_hits >= STALE_RISK_CONSECUTIVE_REVIEW_HITS
        and _supports_review_promotion(semantic_class)
    ):
        return "REVIEW", aging_hits
    return "AGING", aging_hits


def _eligible_query(last_id: str | None, page_size: int) -> tuple[str, tuple]:
    sql = """
        SELECT id, summary, concept_type, currency_status, access_count,
               knowledge_area, created_at, updated_at, content_updated_at, last_accessed,
               last_organic_access, staleness_state, staleness_score, data,
               staleness_reason, staleness_consecutive_hits
        FROM concepts
        WHERE is_current = 1
          AND status = 'active'
          AND (currency_status IN ('ACTIVE', 'CONTESTED') OR currency_status IS NULL)
          AND COALESCE(protected, 0) = 0
          AND COALESCE(always_activate, 0) = 0
          AND concept_type IN ('observation', 'decision')
    """
    params: list = []
    if last_id:
        sql += " AND id > ?"
        params.append(last_id)
    sql += " ORDER BY id LIMIT ?"
    params.append(page_size)
    return sql, tuple(params)


def _cleanup_ineligible(conn) -> int:
    rows = conn.execute(
        """
        SELECT id FROM concepts
        WHERE staleness_state IS NOT NULL
          AND (
            is_current != 1
            OR status != 'active'
            OR COALESCE(protected, 0) = 1
            OR COALESCE(always_activate, 0) = 1
            OR (currency_status NOT IN ('ACTIVE', 'CONTESTED') AND currency_status IS NOT NULL)
            OR concept_type NOT IN ('observation', 'decision')
          )
        """
    ).fetchall()
    cleared = 0
    for row in rows:
        if update_stale_risk_fields(conn, row["id"], clear=True, require_current=False):
            _record_event(
                conn,
                GOV_EVENT_STALE_RISK_CLEARED,
                row["id"],
                {"reason": "became_ineligible", "detector_version": STALE_RISK_DETECTOR_VERSION},
            )
            cleared += 1
    return cleared


def run_criteria_staleness_detector(conn, page_size: int = 500) -> dict:
    """Evaluate stale-risk lifecycle state over the full eligible pool."""
    if not STALE_RISK_DETECTOR_ENABLED:
        return {"processed": 0, "state_changes": 0, "cleared": 0, "promotions_capped": 0}

    processed = 0
    state_changes = 0
    cleared = _cleanup_ineligible(conn)
    promotions_capped = 0
    upward_changes = []
    last_id = None
    evaluated_at = _utc_now_iso()

    while True:
        sql, params = _eligible_query(last_id, page_size)
        rows = conn.execute(sql, params).fetchall()
        if not rows:
            break
        for row in rows:
            processed += 1
            last_id = row["id"]
            score, features = _score_candidate(row)
            current_state = row["staleness_state"]
            current_hits = int(row["staleness_consecutive_hits"] or 0)
            target_state, next_hits = _target_state(
                score,
                current_state,
                current_hits,
                features["semantic_class"],
            )
            reason = _json_details(features)

            if target_state is None:
                if current_state is not None:
                    if update_stale_risk_fields(conn, row["id"], clear=True):
                        _record_event(
                            conn,
                            GOV_EVENT_STALE_RISK_CLEARED,
                            row["id"],
                            {"previous_state": current_state, "score": round(score, 4), **features},
                        )
                        cleared += 1
                else:
                    update_stale_risk_fields(
                        conn,
                        row["id"],
                        state=None,
                        score=score,
                        reason=reason,
                        evaluated_at=evaluated_at,
                        detector_version=STALE_RISK_DETECTOR_VERSION,
                        consecutive_hits=0,
                    )
                continue

            severity_delta = _STATE_ORDER[target_state] - _STATE_ORDER.get(current_state, 0)
            payload = {
                "concept_id": row["id"],
                "target_state": target_state,
                "previous_state": current_state,
                "score": score,
                "reason": reason,
                "evaluated_at": evaluated_at,
                "consecutive_hits": next_hits,
                "features": features,
            }

            if severity_delta > 0:
                upward_changes.append(payload)
                continue

            update_stale_risk_fields(
                conn,
                row["id"],
                state=target_state,
                score=score,
                reason=reason,
                evaluated_at=evaluated_at,
                detector_version=STALE_RISK_DETECTOR_VERSION,
                consecutive_hits=next_hits,
            )
            if target_state != current_state:
                _record_event(
                    conn,
                    GOV_EVENT_STALE_RISK_STATE_CHANGED,
                    row["id"],
                    {
                        "detector_version": STALE_RISK_DETECTOR_VERSION,
                        "previous_state": current_state,
                        "new_state": target_state,
                        **features,
                    },
                )
                state_changes += 1
        if len(rows) < page_size:
            break

    upward_changes.sort(
        key=lambda item: (
            -item["score"],
            -(item["features"]["content_age_days"]),
            -(item["features"]["access_count"]),
            item["concept_id"],
        )
    )

    for idx, item in enumerate(upward_changes):
        if idx >= STALE_RISK_MAX_PROMOTIONS_PER_RUN:
            promotions_capped += 1
            continue
        update_stale_risk_fields(
            conn,
            item["concept_id"],
            state=item["target_state"],
            score=item["score"],
            reason=item["reason"],
            evaluated_at=item["evaluated_at"],
            detector_version=STALE_RISK_DETECTOR_VERSION,
            consecutive_hits=item["consecutive_hits"],
        )
        _record_event(
            conn,
            GOV_EVENT_STALE_RISK_STATE_CHANGED,
            item["concept_id"],
            {
                "detector_version": STALE_RISK_DETECTOR_VERSION,
                "previous_state": item["previous_state"],
                "new_state": item["target_state"],
                **item["features"],
            },
        )
        state_changes += 1

    _record_event(
        conn,
        GOV_EVENT_STALE_RISK_RUN_COMPLETE,
        None,
        {
            "detector_version": STALE_RISK_DETECTOR_VERSION,
            "processed": processed,
            "state_changes": state_changes,
            "cleared": cleared,
            "promotions_capped": promotions_capped,
            "page_size": page_size,
        },
    )
    conn.commit()
    logger.info(
        "COGGOV-014: processed=%d state_changes=%d cleared=%d promotions_capped=%d",
        processed,
        state_changes,
        cleared,
        promotions_capped,
    )
    return {
        "processed": processed,
        "state_changes": state_changes,
        "cleared": cleared,
        "promotions_capped": promotions_capped,
    }
