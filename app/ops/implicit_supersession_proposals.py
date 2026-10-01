"""Read-only implicit supersession proposal discovery.

This module intentionally proposes human-review candidates only. It never
mutates concept lifecycle state.
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

SCHEMA_VERSION = "implicit_supersession_proposals.v1"
REVIEW_SCHEMA_VERSION = "implicit_supersession_review_labels.v1"
CLAIM_BOUNDARY = (
    "Read-only implicit supersession proposal discovery; candidates are not "
    "authority and do not mutate memory."
)
REVIEW_DECISIONS = {
    "approve_supersession",
    "reject_not_supersession",
    "needs_more_context",
    "partial_retain",
}
RETENTION_MODES = {"replace", "keep_both", "partial_retain"}

REQUIRED_COLUMNS = {
    "id",
    "summary",
    "knowledge_area",
    "created_at",
    "updated_at",
    "data",
    "superseded_by",
    "subject_key",
    "is_current",
    "status",
    "provenance",
    "edit_provenance",
}

PROVISIONAL_MARKERS = {
    "draft",
    "pending",
    "planned",
    "not yet",
    "in progress",
    "missing",
    "blocked",
    "awaiting",
    "should be",
    "proposal",
    "tentative",
}
RESOLUTION_MARKERS = {
    "final",
    "approved",
    "landed",
    "implemented",
    "fixed",
    "resolved",
    "shipped",
    "complete",
    "verified",
    "done",
    "now",
}
REPLACEMENT_MARKERS = {
    "supersedes",
    "replaces",
    "instead of",
    "no longer",
    "source of truth",
    "governs",
    "landed on",
}
SEQUENCE_GUARD_MARKERS = {
    "next move",
    "next moves",
    "recommended next",
    "next step",
    "next steps",
    "proceed with",
    "proceed to",
    "next lane",
    "next implementation move",
}
GENERIC_SEQUENCE_SUBJECT_MARKERS = {
    "next recommended move",
    "the next recommended move",
    "the correct next step",
    "next step",
    "next gate",
    "next lane",
    "active lane",
    "proceed with",
    "lets proceed",
    "let's proceed",
    "resolve the blocked",
}
CONTENT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "by",
    "for",
    "from",
    "gate",
    "in",
    "is",
    "it",
    "lane",
    "move",
    "moves",
    "next",
    "of",
    "on",
    "or",
    "recommended",
    "step",
    "steps",
    "the",
    "then",
    "this",
    "to",
    "with",
}


@dataclass(frozen=True)
class ConceptRow:
    id: str
    summary: str
    knowledge_area: str | None
    created_at: str | None
    updated_at: str | None
    subject_key: str | None
    superseded_by: str | None
    is_current: int | None
    status: str | None
    provenance: str | None
    edit_provenance: str | None
    data: dict[str, Any]

    @classmethod
    def from_mapping(cls, row: sqlite3.Row | dict[str, Any]) -> ConceptRow:
        values = dict(row)
        return cls(
            id=str(values.get("id") or ""),
            summary=str(values.get("summary") or ""),
            knowledge_area=values.get("knowledge_area"),
            created_at=values.get("created_at"),
            updated_at=values.get("updated_at"),
            subject_key=values.get("subject_key"),
            superseded_by=values.get("superseded_by"),
            is_current=values.get("is_current"),
            status=values.get("status"),
            provenance=values.get("provenance"),
            edit_provenance=values.get("edit_provenance"),
            data=_parse_data(values.get("data")),
        )


def _parse_data(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def open_readonly_connection(db_path: Path) -> sqlite3.Connection:
    """Open SQLite in immutable read-only mode for proposal discovery."""
    resolved = db_path.expanduser().resolve()
    uri = f"file:{quote(str(resolved))}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def validate_schema(conn: sqlite3.Connection) -> None:
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(concepts)")}
    missing = sorted(REQUIRED_COLUMNS - columns)
    if missing:
        raise ValueError(f"concepts table missing required columns: {', '.join(missing)}")


def discover_proposals(
    conn: sqlite3.Connection,
    *,
    max_groups: int = 200,
    max_group_size: int = 8,
    max_candidates: int = 200,
) -> dict[str, Any]:
    """Discover bounded review candidates from same-subject temporal groups."""
    started = datetime.now(UTC)
    validate_schema(conn)
    group_rows = conn.execute(
        """
        SELECT subject_key, COUNT(*) AS n, MAX(created_at) AS newest_created_at
        FROM concepts
        WHERE subject_key IS NOT NULL
          AND TRIM(subject_key) != ''
          AND COALESCE(is_current, 1) = 1
          AND superseded_by IS NULL
          AND COALESCE(status, 'active') != 'deleted'
        GROUP BY subject_key
        HAVING n BETWEEN 2 AND ?
        ORDER BY newest_created_at DESC
        LIMIT ?
        """,
        (int(max_group_size), int(max_groups)),
    ).fetchall()

    candidates: list[dict[str, Any]] = []
    disposition_counts: dict[str, int] = {}
    review_groups: dict[str, int] = {}
    groups_scanned = 0
    pairs_scored = 0

    for group in group_rows:
        subject_key = group["subject_key"]
        rows = [
            ConceptRow.from_mapping(row)
            for row in conn.execute(
                """
                SELECT id, summary, knowledge_area, created_at, updated_at, data,
                       superseded_by, subject_key, is_current, status, provenance,
                       edit_provenance
                FROM concepts
                WHERE subject_key = ?
                  AND COALESCE(is_current, 1) = 1
                  AND superseded_by IS NULL
                  AND COALESCE(status, 'active') != 'deleted'
                ORDER BY datetime(created_at), created_at, id
                LIMIT ?
                """,
                (subject_key, int(max_group_size)),
            ).fetchall()
        ]
        if len(rows) < 2:
            continue
        groups_scanned += 1

        pair_indexes: set[tuple[int, int]] = set()
        for index in range(len(rows) - 1):
            pair_indexes.add((index, index + 1))
        pair_indexes.add((0, len(rows) - 1))

        for old_index, new_index in sorted(pair_indexes):
            if len(candidates) >= int(max_candidates):
                break
            pairs_scored += 1
            result = classify_pair(rows[old_index], rows[new_index])
            disposition_counts[result["disposition"]] = (
                disposition_counts.get(result["disposition"], 0) + 1
            )
            if result["proposal"]:
                candidates.append(result)
                group_key = str(result.get("review_group_key") or "")
                if group_key:
                    review_groups[group_key] = review_groups.get(group_key, 0) + 1
        if len(candidates) >= int(max_candidates):
            break

    elapsed_ms = (datetime.now(UTC) - started).total_seconds() * 1000.0
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "generated_at": datetime.now(UTC).isoformat(),
        "claim_boundary": CLAIM_BOUNDARY,
        "mode": "read_only_proposal_discovery",
        "mutation_count": 0,
        "parameters": {
            "max_groups": int(max_groups),
            "max_group_size": int(max_group_size),
            "max_candidates": int(max_candidates),
        },
        "summary": {
            "groups_considered": len(group_rows),
            "groups_scanned": groups_scanned,
            "pairs_scored": pairs_scored,
            "candidate_count": len(candidates),
            "review_group_count": len(review_groups),
            "review_groups": dict(sorted(review_groups.items())),
            "review_recommended_count": sum(
                1 for row in candidates if row["disposition"] == "review_recommended"
            ),
            "needs_context_count": sum(
                1 for row in candidates if row["disposition"] == "needs_context"
            ),
            "candidate_count_by_disposition": {
                key: sum(1 for row in candidates if row["disposition"] == key)
                for key in sorted({row["disposition"] for row in candidates})
            },
            "disposition_counts": disposition_counts,
            "elapsed_ms": elapsed_ms,
        },
        "candidates": candidates,
    }


def classify_pair(old: ConceptRow | dict[str, Any], new: ConceptRow | dict[str, Any]) -> dict[str, Any]:
    old_row = old if isinstance(old, ConceptRow) else ConceptRow.from_mapping(old)
    new_row = new if isinstance(new, ConceptRow) else ConceptRow.from_mapping(new)
    old_text = old_row.summary.lower()
    new_text = new_row.summary.lower()
    joined_text = f"{old_text}\n{new_text}"
    signals: list[str] = []

    temporal_ok = bool(_parse_time(old_row.created_at) and _parse_time(new_row.created_at))
    if temporal_ok and _parse_time(new_row.created_at) > _parse_time(old_row.created_at):
        signals.append("newer_candidate")
    elif temporal_ok:
        return _result(old_row, new_row, "not_supersession", 0.0, ["candidate_not_newer"])

    if not old_row.subject_key or old_row.subject_key != new_row.subject_key:
        return _result(old_row, new_row, "not_supersession", 0.0, ["subject_mismatch"])

    if old_row.superseded_by == new_row.id:
        return _result(old_row, new_row, "already_governed", 1.0, ["explicit_supersession_exists"])

    old_provisional = _contains_any(old_text, PROVISIONAL_MARKERS)
    new_resolved = _contains_any(new_text, RESOLUTION_MARKERS)
    replacement_signal = _contains_any(joined_text, REPLACEMENT_MARKERS)
    strong_replacement_signal = _contains_any(
        joined_text,
        REPLACEMENT_MARKERS - {"current"},
    )
    sequence_guard = _has_sequence_guard(joined_text)

    if old_provisional:
        signals.append("old_provisional")
    if new_resolved:
        signals.append("new_resolved")
    if replacement_signal:
        signals.append("replacement_language")
    if sequence_guard:
        signals.append("sequence_guard")

    has_source = _has_source_evidence(old_row) and _has_source_evidence(new_row)
    evidence_complete = has_source and temporal_ok
    if _is_generic_sequence_subject(old_row.subject_key) and not _has_meaningful_overlap(
        old_text, new_text
    ):
        return _result(
            old_row,
            new_row,
            "not_supersession",
            0.05,
            signals + ["generic_sequence_subject_scope_mismatch"],
            evidence_complete=evidence_complete,
        )

    if sequence_guard and not strong_replacement_signal:
        return _result(
            old_row,
            new_row,
            "not_supersession",
            0.05,
            signals + ["sequence_status_without_explicit_replacement"],
            evidence_complete=evidence_complete,
        )

    if _is_lossy_successor(old_text, new_text):
        return _result(
            old_row,
            new_row,
            "not_supersession",
            0.05,
            signals + ["successor_drops_material_detail"],
            evidence_complete=evidence_complete,
        )

    score = 0.0
    if temporal_ok:
        score += 0.20
    if old_provisional and new_resolved:
        score += 0.45
    if replacement_signal:
        score += 0.35
    if has_source:
        score += 0.05
    score = min(score, 1.0)

    if score >= 0.65 and evidence_complete:
        return _result(
            old_row,
            new_row,
            "review_recommended",
            score,
            signals,
            evidence_complete=evidence_complete,
        )
    if score >= 0.45 and (old_provisional or new_resolved or replacement_signal):
        return _result(
            old_row,
            new_row,
            "needs_context",
            score,
            signals + ["incomplete_or_ambiguous_evidence"],
            proposal=True,
            evidence_complete=evidence_complete,
        )
    return _result(
        old_row,
        new_row,
        "not_supersession",
        score,
        signals or ["weak_implicit_signal"],
        proposal=False,
        evidence_complete=evidence_complete,
    )


def write_report_artifacts(report: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "implicit_supersession_proposals.json"
    md_path = out_dir / "implicit_supersession_proposals.md"
    tsv_path = out_dir / "implicit_supersession_proposals.tsv"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    tsv_path.write_text(render_tsv(report), encoding="utf-8")
    return {"json_path": str(json_path), "markdown_path": str(md_path), "tsv_path": str(tsv_path)}


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    lines = [
        "# Implicit Supersession Proposal Packet",
        "",
        f"- Generated: {report.get('generated_at')}",
        f"- Boundary: {report.get('claim_boundary')}",
        f"- Mutation count: {report.get('mutation_count', 0)}",
        f"- Groups scanned: {summary.get('groups_scanned', 0)}",
        f"- Pairs scored: {summary.get('pairs_scored', 0)}",
        f"- Review candidates: {summary.get('candidate_count', 0)}",
        f"- Review groups: {summary.get('review_group_count', 0)}",
        "",
        "## Review Queue",
        "",
    ]
    candidates = report.get("candidates") or []
    if not candidates:
        lines.append("No implicit supersession candidates were found in this bounded pass.")
        lines.append("")
        return "\n".join(lines)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        grouped.setdefault(str(candidate.get("review_group_key") or "ungrouped"), []).append(candidate)

    item_index = 1
    for group_key, group_candidates in grouped.items():
        lines.extend([f"### Group: `{group_key}`", ""])
        if len(group_candidates) > 1:
            lines.extend([f"- Candidates in group: {len(group_candidates)}", ""])
        for candidate in group_candidates:
            lines.extend(
                [
                    f"#### {item_index}. {candidate['old_id']} -> {candidate['new_id']}",
                    "",
                    f"- Subject: `{candidate.get('subject_key') or 'unknown'}`",
                    f"- Disposition: `{candidate['disposition']}`",
                    f"- Score: {candidate['score']:.2f}",
                    f"- Old created: {candidate.get('old_created_at') or 'unknown'}",
                    f"- New created: {candidate.get('new_created_at') or 'unknown'}",
                    f"- Signals: {', '.join(candidate.get('signals') or [])}",
                    f"- Evidence complete: {candidate.get('evidence_complete')}",
                    f"- Review schema: `{candidate.get('review_schema_version')}`",
                    f"- Review decision: `{candidate.get('review_decision') or '<fill one: approve_supersession | reject_not_supersession | needs_more_context | partial_retain>'}`",
                    f"- Retention mode: `{candidate.get('retention_mode') or '<fill one: replace | keep_both | partial_retain>'}`",
                    "",
                    f"Old: {candidate.get('old_summary')}",
                    "",
                    f"New: {candidate.get('new_summary')}",
                    "",
                    "Rationale: <required for approve_supersession or partial_retain>",
                    "",
                ]
            )
            item_index += 1
    return "\n".join(lines)


def render_tsv(report: dict[str, Any]) -> str:
    fields = [
        "old_id",
        "new_id",
        "subject_key",
        "disposition",
        "score",
        "old_created_at",
        "new_created_at",
        "signals",
        "evidence_complete",
        "old_summary",
        "new_summary",
        "review_schema_version",
        "review_group_key",
        "review_decision",
        "retention_mode",
        "reviewer",
        "reviewed_at",
        "review_rationale",
        "review_source",
        "expected_disposition",
    ]
    rows: list[str] = []
    output = _StringWriter()
    writer = csv.DictWriter(output, fieldnames=fields, delimiter="\t", extrasaction="ignore")
    writer.writeheader()
    rows.append(output.pop())
    for candidate in report.get("candidates") or []:
        output = _StringWriter()
        row = dict(candidate)
        row["signals"] = ",".join(candidate.get("signals") or [])
        writer = csv.DictWriter(output, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writerow(row)
        rows.append(output.pop())
    return "".join(rows)


def _result(
    old: ConceptRow,
    new: ConceptRow,
    disposition: str,
    score: float,
    signals: list[str],
    *,
    proposal: bool | None = None,
    evidence_complete: bool | None = None,
) -> dict[str, Any]:
    if proposal is None:
        proposal = disposition in {"review_recommended", "needs_context"}
    if evidence_complete is None:
        evidence_complete = _has_source_evidence(old) and _has_source_evidence(new) and bool(
            _parse_time(old.created_at) and _parse_time(new.created_at)
        )
    return {
        "old_id": old.id,
        "new_id": new.id,
        "subject_key": old.subject_key,
        "disposition": disposition,
        "proposal": bool(proposal),
        "score": round(float(score), 4),
        "signals": sorted(set(signals)),
        "evidence_complete": bool(evidence_complete),
        "old_created_at": old.created_at,
        "new_created_at": new.created_at,
        "old_summary": old.summary,
        "new_summary": new.summary,
        "old_source": _source_evidence(old),
        "new_source": _source_evidence(new),
        "review_schema_version": REVIEW_SCHEMA_VERSION,
        "review_group_key": _review_group_key(old),
        "review_decision": "",
        "retention_mode": "",
        "reviewer": "",
        "reviewed_at": "",
        "review_rationale": "",
        "review_source": "",
        "expected_disposition": _expected_disposition_for_review(disposition),
    }


def _review_group_key(row: ConceptRow) -> str:
    subject = (row.subject_key or "unknown").strip() or "unknown"
    return f"{subject}::{row.id}"


def _expected_disposition_for_review(disposition: str) -> str:
    if disposition == "review_recommended":
        return "review_recommended"
    if disposition == "needs_context":
        return "needs_context"
    return disposition


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _contains_any(text: str, markers: set[str]) -> bool:
    return any(_contains_marker(text, marker) for marker in markers)


def _contains_marker(text: str, marker: str) -> bool:
    escaped = re.escape(marker.lower())
    return bool(re.search(rf"(?<![a-z0-9-]){escaped}(?![a-z0-9-])", text.lower()))


def _has_sequence_guard(text: str) -> bool:
    if _contains_any(text, SEQUENCE_GUARD_MARKERS):
        return True
    return bool(
        re.search(r"\bnext\s+(?:recommended\s+)?[a-z0-9_-]{1,40}\s+move\s+is\b", text)
        or re.search(r"\bnext\s+(?:recommended\s+)?[a-z0-9_-]{1,40}\s+lane\s+is\b", text)
        or re.search(r"\bnext\s+(?:recommended\s+)?[a-z0-9_-]{1,40}\s+step\s+is\b", text)
        or re.search(r"\bnext\s+implementation\s+target\s+is\b", text)
        or re.search(r"\bthe\s+next\s+recommended\s+move\s+is\b", text)
        or re.search(r"\bproceed\s+as\s+suggested\b", text)
        or re.search(r"\blet'?s\s+proceed\b", text)
    )


def _is_generic_sequence_subject(subject_key: str | None) -> bool:
    subject = (subject_key or "").strip().lower()
    return any(subject.startswith(marker) for marker in GENERIC_SEQUENCE_SUBJECT_MARKERS)


def _content_tokens(text: str) -> set[str]:
    tokens = {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", text.lower())
        if token not in CONTENT_STOPWORDS
    }
    return tokens


def _has_meaningful_overlap(old_text: str, new_text: str) -> bool:
    old_tokens = _content_tokens(old_text)
    new_tokens = _content_tokens(new_text)
    if not old_tokens or not new_tokens:
        return False
    overlap = len(old_tokens & new_tokens)
    return overlap >= 3 or overlap / max(1, min(len(old_tokens), len(new_tokens))) >= 0.35


def _is_lossy_successor(old_text: str, new_text: str) -> bool:
    old_tokens = _content_tokens(old_text)
    new_tokens = _content_tokens(new_text)
    if len(old_tokens) < 6 or not new_tokens:
        return False
    if len(new_tokens) > max(3, int(len(old_tokens) * 0.70)):
        return False
    overlap_ratio = len(old_tokens & new_tokens) / max(1, len(new_tokens))
    return overlap_ratio >= 0.65


def _source_evidence(row: ConceptRow) -> dict[str, Any]:
    source: dict[str, Any] = {}
    for key in (
        "source_path",
        "source_commit",
        "source_url",
        "url",
        "commit",
        "path",
        "evidence",
        "authority_source",
    ):
        value = row.data.get(key)
        if value:
            source[key] = value
    if row.provenance:
        source["provenance"] = row.provenance
    if row.edit_provenance:
        source["edit_provenance"] = row.edit_provenance
    return source


def _has_source_evidence(row: ConceptRow) -> bool:
    source = _source_evidence(row)
    return any(key in source for key in ("source_path", "source_commit", "source_url", "url", "evidence"))


class _StringWriter:
    def __init__(self) -> None:
        self.parts: list[str] = []

    def write(self, value: str) -> int:
        self.parts.append(value)
        return len(value)

    def pop(self) -> str:
        value = "".join(self.parts)
        self.parts = []
        return value
