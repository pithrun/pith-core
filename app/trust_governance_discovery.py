"""Observe-mode discovery for trust-governance candidate signals."""

from __future__ import annotations

import json
import re
import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DISCOVERY_SCHEMA_VERSION = "trust_governance_discovery.v1.1"
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]*")
AUTHORITY_TERMS = frozenset(
    {
        "approved",
        "authority",
        "authoritative",
        "current",
        "final",
        "govern",
        "governing",
        "governs",
        "landed",
        "official",
        "policy",
        "source-of-truth",
        "truth",
    }
)
SUPERSESSION_TERMS = frozenset(
    {
        "archived",
        "deprecated",
        "historical",
        "no-longer-current",
        "replaced",
        "replacement",
        "supersede",
        "superseded",
        "supersedes",
    }
)
STRONG_SOURCE_KEYS = frozenset(
    {
        "canonical_path",
        "canonical_url",
        "commit_sha",
        "git_commit",
        "repo_path",
        "source_commit",
        "source_path",
        "source_reference",
        "source_ref",
        "source_session",
        "source_trace_id",
        "source_url",
    }
)
WEAK_SOURCE_KEYS = frozenset(
    {
        "extraction_source",
        "ka_admission_source",
        "knowledge_area_source",
        "salience_source",
        "source_type",
    }
)
SOURCE_KEYS = STRONG_SOURCE_KEYS | WEAK_SOURCE_KEYS
ARTIFACT_SOURCE_KEYS = STRONG_SOURCE_KEYS - {"source_session", "source_trace_id"}
RAW_BODY_SOURCE_REF_KEYS = frozenset(
    {
        "assistant",
        "assistant_response",
        "content",
        "message",
        "prompt",
        "response",
        "summary",
        "text",
        "user",
        "user_message",
    }
)
SOURCE_REF_VALUE_MAX_CHARS = 240
COMMON_AUTHORITY_TERMS = frozenset({"current", "final", "policy", "truth"})
NEGATIVE_STATES = frozenset({"contradicted", "contested", "stale", "superseded", "archived"})
MAX_EVIDENCE_ITEMS = 8
CURRENT_CANDIDATE_THRESHOLD = 0.62
HISTORICAL_CANDIDATE_THRESHOLD = 0.45
CANDIDATE_PREFILTER_SQL = """(
    authority_score >= 0.7
    OR superseded_by IS NOT NULL
    OR lower(coalesce(staleness_state, '')) IN ('contradicted', 'contested', 'stale', 'superseded', 'archived')
    OR lower(coalesce(status, '')) IN ('archived', 'deleted', 'inactive')
    OR lower(summary) LIKE '%authority%'
    OR lower(summary) LIKE '%authoritative%'
    OR lower(summary) LIKE '%approved%'
    OR lower(summary) LIKE '%governing%'
    OR lower(summary) LIKE '%governs%'
    OR lower(summary) LIKE '%official%'
    OR lower(summary) LIKE '%source-of-truth%'
    OR lower(summary) LIKE '%superseded%'
    OR lower(summary) LIKE '%replaced%'
    OR lower(summary) LIKE '%deprecated%'
)"""


@dataclass(frozen=True)
class DiscoveryEvidence:
    kind: str
    value: str
    weight: float
    source: str


@dataclass(frozen=True)
class AuthorityCandidate:
    concept_id: str
    summary: str
    knowledge_area: str | None
    subject_key: str | None
    concept_type: str | None
    proposed_state: str
    confidence: float
    risk_flags: tuple[str, ...]
    evidence: tuple[DiscoveryEvidence, ...]
    source_refs: tuple[dict[str, str], ...]
    lifecycle: dict[str, Any]


@dataclass(frozen=True)
class DiscoveryReport:
    schema_version: str
    db_path: str
    generated_at: str
    scanned_concepts: int
    candidate_count: int
    metrics: dict[str, Any]
    candidates: tuple[AuthorityCandidate, ...]
    warnings: tuple[str, ...]


def _tokens(text: str | None) -> set[str]:
    lowered = (text or "").casefold()
    boundary_split = re.sub(r"[-/]+", " ", lowered)
    return set(TOKEN_RE.findall(lowered)) | set(TOKEN_RE.findall(boundary_split))


def _safe_json_loads(raw: Any) -> tuple[dict[str, Any], bool]:
    if raw is None:
        return {}, False
    if isinstance(raw, dict):
        return raw, False
    try:
        decoded = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}, True
    return (decoded, False) if isinstance(decoded, dict) else ({}, True)


def _stringify_json_values(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(_stringify_json_values(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(_stringify_json_values(item) for item in value)
    return str(value or "")


def _source_ref_strength(key: str) -> str | None:
    normalized = key.casefold()
    if normalized in STRONG_SOURCE_KEYS:
        return "strong"
    if normalized in WEAK_SOURCE_KEYS:
        return "weak"
    return None


def _truncate_source_ref_value(value: Any) -> str:
    text = str(value)
    if len(text) <= SOURCE_REF_VALUE_MAX_CHARS:
        return text
    return text[: SOURCE_REF_VALUE_MAX_CHARS - 3] + "..."


def _compact_source_ref_value(value: Any) -> str | None:
    if isinstance(value, (str, int, float)):
        return _truncate_source_ref_value(value)
    if isinstance(value, dict):
        compact = {
            str(key): _truncate_source_ref_value(item)
            for key, item in value.items()
            if str(key).casefold() not in RAW_BODY_SOURCE_REF_KEYS
            and isinstance(item, (str, int, float))
        }
        if compact:
            return _truncate_source_ref_value(json.dumps(compact, sort_keys=True))
    return None


def _walk_source_refs(value: Any, *, max_items: int = MAX_EVIDENCE_ITEMS) -> tuple[dict[str, str], ...]:
    refs: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()

    def add_ref(key: str, item: Any, path: str) -> None:
        normalized_key = key.casefold()
        if normalized_key in RAW_BODY_SOURCE_REF_KEYS:
            return
        strength = _source_ref_strength(normalized_key)
        if strength is None:
            return
        compact_value = _compact_source_ref_value(item)
        if not compact_value:
            return
        dedupe_key = (strength, normalized_key, path, compact_value)
        if dedupe_key in seen:
            return
        seen.add(dedupe_key)
        refs.append(
            {
                "key": normalized_key,
                "path": path,
                "strength": strength,
                "value": compact_value,
            }
        )

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, item in node.items():
                key_text = str(key)
                normalized_key = key_text.casefold()
                child_path = f"{path}.{normalized_key}" if path else normalized_key
                add_ref(normalized_key, item, child_path)
                if isinstance(item, list) and _source_ref_strength(normalized_key):
                    for index, element in enumerate(item):
                        add_ref(normalized_key, element, f"{child_path}[{index}]")
                walk(item, child_path)
        elif isinstance(node, list):
            for item in node:
                walk(item, f"{path}[]" if path else "[]")

    walk(value, "")
    refs.sort(key=lambda item: 0 if item["strength"] == "strong" else 1)
    return tuple(refs[:max_items])


def _is_artifact_source_ref(ref: dict[str, str]) -> bool:
    return ref.get("strength") == "strong" and ref.get("key") in ARTIFACT_SOURCE_KEYS


def _row_value(row: sqlite3.Row, key: str, default: Any = None) -> Any:
    try:
        return row[key]
    except (KeyError, IndexError):
        return default


def _as_bool_int(value: Any) -> bool | None:
    if value is None:
        return None
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return None


def _infer_state(row: sqlite3.Row, data: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    risk_flags: list[str] = []
    superseded_by = str(_row_value(row, "superseded_by", "") or "").strip()
    is_current = _as_bool_int(_row_value(row, "is_current"))
    status = str(_row_value(row, "status", "") or "").casefold()
    currency_status = str(_row_value(row, "currency_status", "") or "").casefold()
    staleness_state = str(_row_value(row, "staleness_state", "") or "").casefold()
    data_state = str(data.get("state") or data.get("status") or "").casefold()

    if superseded_by:
        risk_flags.append("superseded_by")
        return "superseded", tuple(risk_flags)
    if is_current is False:
        risk_flags.append("not_current")
        return "historical", tuple(risk_flags)
    if status in {"archived", "deleted", "inactive"}:
        risk_flags.append(f"status:{status}")
        return "historical", tuple(risk_flags)
    if currency_status in NEGATIVE_STATES:
        risk_flags.append(f"currency_status:{currency_status}")
        return "historical", tuple(risk_flags)
    if staleness_state in NEGATIVE_STATES:
        risk_flags.append(f"staleness_state:{staleness_state}")
        return "historical", tuple(risk_flags)
    if data_state in {"historical", "superseded", "archived", "deprecated"}:
        risk_flags.append(f"data_state:{data_state}")
        return "historical", tuple(risk_flags)
    return "current_candidate", tuple(risk_flags)


def _evidence_from_row(
    row: sqlite3.Row,
    data: dict[str, Any],
    source_refs: tuple[dict[str, str], ...],
) -> tuple[DiscoveryEvidence, ...]:
    summary = str(_row_value(row, "summary", "") or "")
    data_text = _stringify_json_values(data)[:2000]
    token_set = _tokens(f"{summary} {data_text}")
    evidence: list[DiscoveryEvidence] = []

    for term in sorted(AUTHORITY_TERMS & token_set):
        evidence.append(DiscoveryEvidence("authority_term", term, 0.12, "summary_or_data"))
    for term in sorted(SUPERSESSION_TERMS & token_set):
        evidence.append(DiscoveryEvidence("supersession_term", term, 0.10, "summary_or_data"))

    artifact_source_count = sum(1 for ref in source_refs if _is_artifact_source_ref(ref))
    if artifact_source_count:
        evidence.append(DiscoveryEvidence("source_ref", str(artifact_source_count), 0.0, "data"))
    elif source_refs:
        evidence.append(DiscoveryEvidence("weak_source_ref", str(len(source_refs)), 0.0, "data"))

    authority_score = _row_value(row, "authority_score")
    if authority_score is not None:
        try:
            authority_value = float(authority_score)
        except (TypeError, ValueError):
            authority_value = 0.0
        if authority_value >= 0.7:
            evidence.append(DiscoveryEvidence("authority_score", f"{authority_value:.3f}", 0.18, "column"))
        elif authority_value > 0.0:
            evidence.append(DiscoveryEvidence("authority_score", f"{authority_value:.3f}", 0.08, "column"))

    verification_status = str(_row_value(row, "verification_status", "") or "").casefold()
    if verification_status in {"verified", "approved", "source_verified"}:
        evidence.append(DiscoveryEvidence("verification_status", verification_status, 0.14, "column"))

    currency_status = str(_row_value(row, "currency_status", "") or "").casefold()
    if currency_status in {"active", "current"}:
        evidence.append(DiscoveryEvidence("currency_status", currency_status, 0.08, "column"))

    if str(_row_value(row, "subject_key", "") or "").strip():
        evidence.append(DiscoveryEvidence("subject_key", str(_row_value(row, "subject_key")), 0.10, "column"))

    if str(_row_value(row, "superseded_by", "") or "").strip():
        evidence.append(DiscoveryEvidence("superseded_by", str(_row_value(row, "superseded_by")), -0.18, "column"))

    return tuple(evidence)


def _score_candidate(evidence: tuple[DiscoveryEvidence, ...], risk_flags: tuple[str, ...]) -> float:
    score = 0.18 + sum(item.weight for item in evidence)
    if "malformed_data_json" in risk_flags:
        score -= 0.12
    if any(flag.startswith(("staleness_state:", "currency_status:", "status:")) for flag in risk_flags):
        score -= 0.14
    if "not_current" in risk_flags:
        score -= 0.18
    if "superseded_by" in risk_flags:
        score -= 0.08
    return max(0.0, min(1.0, score))


def iter_concept_rows(conn: sqlite3.Connection, *, limit: int | None = None) -> Iterable[sqlite3.Row]:
    sql = """SELECT id, summary, knowledge_area, concept_type, status, data,
                   authority_score, currency_status, staleness_state, staleness_score,
                   staleness_reason, superseded_by, supersession_reason,
                   verification_status, is_current, session_id, subject_key, provenance,
                   created_at, updated_at
            FROM concepts
            WHERE """ + CANDIDATE_PREFILTER_SQL + """
            ORDER BY updated_at DESC"""
    params: tuple[Any, ...] = ()
    if limit is not None:
        sql += " LIMIT ?"
        params = (int(limit),)
    yield from conn.execute(sql, params)


def build_candidate(row: sqlite3.Row) -> AuthorityCandidate | None:
    data, malformed_json = _safe_json_loads(_row_value(row, "data"))
    proposed_state, state_risks = _infer_state(row, data)
    risk_flags = list(state_risks)
    if malformed_json:
        risk_flags.append("malformed_data_json")

    source_refs = _walk_source_refs(data)
    evidence = _evidence_from_row(row, data, source_refs)
    evidence_kinds = {item.kind for item in evidence}
    evidence_terms = {item.value for item in evidence if item.kind == "authority_term"}
    has_subject = bool(str(_row_value(row, "subject_key", "") or "").strip())
    has_lifecycle = bool(
        evidence_kinds
        & {
            "authority_score",
            "currency_status",
            "verification_status",
            "superseded_by",
        }
    )
    common_only = bool(evidence_terms) and evidence_terms <= COMMON_AUTHORITY_TERMS
    has_strong_authority = any(
        item.kind in {"authority_score", "verification_status"} and item.weight >= 0.14
        for item in evidence
    ) or bool(evidence_terms - COMMON_AUTHORITY_TERMS)

    if proposed_state == "current_candidate" and common_only and not has_subject:
        risk_flags.append("common_term_without_subject_or_source")
        return None
    if proposed_state == "current_candidate" and common_only and not has_strong_authority:
        risk_flags.append("common_term_without_strong_authority")
        return None

    confidence = _score_candidate(evidence, tuple(risk_flags))
    if proposed_state == "current_candidate":
        if not (has_lifecycle and has_subject and "authority_term" in evidence_kinds):
            return None
        if confidence < CURRENT_CANDIDATE_THRESHOLD:
            return None
    elif confidence < HISTORICAL_CANDIDATE_THRESHOLD:
        return None

    lifecycle = {
        "authority_score": _row_value(row, "authority_score"),
        "currency_status": _row_value(row, "currency_status"),
        "staleness_state": _row_value(row, "staleness_state"),
        "staleness_score": _row_value(row, "staleness_score"),
        "staleness_reason": _row_value(row, "staleness_reason"),
        "superseded_by": _row_value(row, "superseded_by"),
        "supersession_reason": _row_value(row, "supersession_reason"),
        "verification_status": _row_value(row, "verification_status"),
        "is_current": _row_value(row, "is_current"),
        "status": _row_value(row, "status"),
    }
    return AuthorityCandidate(
        concept_id=str(_row_value(row, "id")),
        summary=str(_row_value(row, "summary", "") or ""),
        knowledge_area=_optional_string(_row_value(row, "knowledge_area")),
        subject_key=_optional_string(_row_value(row, "subject_key")),
        concept_type=_optional_string(_row_value(row, "concept_type")),
        proposed_state=proposed_state,
        confidence=round(confidence, 3),
        risk_flags=tuple(dict.fromkeys(risk_flags)),
        evidence=evidence,
        source_refs=source_refs,
        lifecycle=lifecycle,
    )


def build_discovery_report(
    db_path: str | Path,
    *,
    limit: int | None = None,
    include_candidates: bool = True,
) -> DiscoveryReport:
    started = time.perf_counter()
    path = Path(db_path)
    candidates: list[AuthorityCandidate] = []
    warnings: list[str] = []
    scanned = 0
    malformed = 0
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        _require_concepts_table(conn)
        for row in iter_concept_rows(conn, limit=limit):
            scanned += 1
            candidate = build_candidate(row)
            if candidate is None:
                data, malformed_json = _safe_json_loads(_row_value(row, "data"))
                if malformed_json:
                    malformed += 1
                continue
            if "malformed_data_json" in candidate.risk_flags:
                malformed += 1
            candidates.append(candidate)

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if malformed:
        warnings.append(f"malformed_data_json_rows={malformed}")
    strong_source_count = sum(
        1 for item in candidates if any(ref.get("strength") == "strong" for ref in item.source_refs)
    )
    weak_only_source_count = sum(
        1
        for item in candidates
        if item.source_refs and not any(ref.get("strength") == "strong" for ref in item.source_refs)
    )
    no_source_count = sum(1 for item in candidates if not item.source_refs)
    metrics = {
        "elapsed_ms": round(elapsed_ms, 3),
        "candidate_density": round(len(candidates) / scanned, 4) if scanned else 0.0,
        "current_candidate_count": sum(1 for item in candidates if item.proposed_state == "current_candidate"),
        "historical_candidate_count": sum(1 for item in candidates if item.proposed_state != "current_candidate"),
        "source_evidence_coverage": round(
            sum(1 for item in candidates if item.source_refs) / len(candidates), 4
        )
        if candidates
        else 1.0,
        "strong_source_evidence_coverage": round(strong_source_count / len(candidates), 4) if candidates else 1.0,
        "weak_only_source_evidence_coverage": round(weak_only_source_count / len(candidates), 4)
        if candidates
        else 1.0,
        "no_source_candidate_count": no_source_count,
    }
    return DiscoveryReport(
        schema_version=DISCOVERY_SCHEMA_VERSION,
        db_path=str(path),
        generated_at=datetime.now(UTC).isoformat(),
        scanned_concepts=scanned,
        candidate_count=len(candidates),
        metrics=metrics,
        candidates=tuple(candidates if include_candidates else ()),
        warnings=tuple(warnings),
    )


def report_to_dict(report: DiscoveryReport) -> dict[str, Any]:
    return {
        "schema_version": report.schema_version,
        "db_path": report.db_path,
        "generated_at": report.generated_at,
        "scanned_concepts": report.scanned_concepts,
        "candidate_count": report.candidate_count,
        "metrics": report.metrics,
        "warnings": list(report.warnings),
        "candidates": [_candidate_to_dict(candidate) for candidate in report.candidates],
    }


def _candidate_to_dict(candidate: AuthorityCandidate) -> dict[str, Any]:
    return {
        "concept_id": candidate.concept_id,
        "summary": candidate.summary,
        "knowledge_area": candidate.knowledge_area,
        "subject_key": candidate.subject_key,
        "concept_type": candidate.concept_type,
        "proposed_state": candidate.proposed_state,
        "confidence": candidate.confidence,
        "risk_flags": list(candidate.risk_flags),
        "evidence": [evidence.__dict__ for evidence in candidate.evidence],
        "source_refs": list(candidate.source_refs),
        "lifecycle": candidate.lifecycle,
    }


def _optional_string(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _require_concepts_table(conn: sqlite3.Connection) -> None:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='concepts'"
    ).fetchone()
    if row is None:
        raise RuntimeError("database is missing required concepts table")
