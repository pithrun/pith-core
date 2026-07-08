"""Diagnostic current-state arbitration for conversation_turn context.

This module compares bounded client-supplied current-state evidence against
already-assembled context surfaces. It is intentionally read-only and performs
no filesystem, network, DB, git, backlog, PR, or CI inspection.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable
from typing import Any

STATUS_ALIASES = {
    "open": "open",
    "active": "open",
    "pending": "open",
    "branch-local": "open",
    "done": "done",
    "complete": "done",
    "completed": "done",
    "closed": "done",
    "merged": "done",
    "passed": "passed",
    "failed": "failed",
}

CONTRADICTIONS = {
    ("open", "done"),
    ("done", "open"),
    ("passed", "failed"),
    ("failed", "passed"),
}

MAX_SNIPPET_CHARS = 512
MAX_CANDIDATES = 50
TRUSTED_AUTHORITIES = {"live", "operator_confirmed"}

_TOKEN_BOUNDARY_LEFT = r"(?<![A-Za-z0-9_-])"
_TOKEN_BOUNDARY_RIGHT = r"(?![A-Za-z0-9_-])"
_STATUS_PATTERN = re.compile(
    _TOKEN_BOUNDARY_LEFT
    + "("
    + "|".join(re.escape(token) for token in sorted(STATUS_ALIASES, key=len, reverse=True))
    + ")"
    + _TOKEN_BOUNDARY_RIGHT,
    re.IGNORECASE,
)
_NON_STATUS_OPEN_FOLLOWERS = {
    "a",
    "an",
    "the",
    "file",
    "folder",
    "link",
    "url",
    "page",
    "browser",
    "terminal",
}


def _record_metric(name: str, value: float, labels: dict[str, str] | None = None) -> None:
    try:
        from app.ops.metrics import metrics

        metrics.record(name, value, labels or {})
    except Exception:
        return


def _bounded_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text[:MAX_SNIPPET_CHARS]


def _exact_token_present(text: str, token: str) -> bool:
    if not text or not token:
        return False
    pattern = re.compile(
        _TOKEN_BOUNDARY_LEFT + re.escape(token) + _TOKEN_BOUNDARY_RIGHT,
        re.IGNORECASE,
    )
    return bool(pattern.search(text))


def _status_tokens(text: str) -> list[str]:
    statuses: list[str] = []
    for match in _STATUS_PATTERN.finditer(text):
        token = match.group(1).lower()
        if token == "open" and _looks_like_open_verb(text, match.end()):
            continue
        normalized = STATUS_ALIASES.get(token)
        if normalized and normalized not in statuses:
            statuses.append(normalized)
    return statuses


def _looks_like_open_verb(text: str, end_pos: int) -> bool:
    remainder = text[end_pos:].lstrip()
    next_match = re.match(r"([A-Za-z0-9_-]+)", remainder)
    if not next_match:
        return False
    return next_match.group(1).lower() in _NON_STATUS_OPEN_FOLLOWERS


def _evidence_get(evidence: Any, key: str, default: Any = None) -> Any:
    if isinstance(evidence, dict):
        return evidence.get(key, default)
    return getattr(evidence, key, default)


def _iter_text_values(value: Any) -> Iterable[str]:
    if value is None:
        return
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, dict):
        for nested in value.values():
            yield from _iter_text_values(nested)
        return
    if isinstance(value, (list, tuple)):
        for nested in value:
            yield from _iter_text_values(nested)
        return
    if isinstance(value, (int, float, bool)):
        yield str(value)


def _add_candidate(candidates: list[dict[str, str]], source: str, text: Any) -> None:
    if len(candidates) >= MAX_CANDIDATES:
        return
    snippet = _bounded_text(text)
    if snippet:
        candidates.append({"source": source, "snippet": snippet})


def _concept_field(concept: Any, field: str) -> Any:
    if isinstance(concept, dict):
        return concept.get(field)
    return getattr(concept, field, None)


def _collect_candidates(
    *,
    activated_concepts: list,
    resume_context: str | None,
    working_context: dict | None,
    active_workstream: dict | None,
    workstream_activation: dict | None,
) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for concept in activated_concepts or []:
        _add_candidate(candidates, "activated_concept.summary", _concept_field(concept, "summary"))
        key_evidence = _concept_field(concept, "key_evidence")
        for item in key_evidence or []:
            _add_candidate(candidates, "activated_concept.key_evidence", item)
            if len(candidates) >= MAX_CANDIDATES:
                return candidates
        if len(candidates) >= MAX_CANDIDATES:
            return candidates

    _add_candidate(candidates, "resume_context", resume_context)

    checkpoint = working_context.get("checkpoint") if isinstance(working_context, dict) else None
    if isinstance(checkpoint, dict):
        for key in ("description", "active", "status"):
            _add_candidate(candidates, f"working_context.checkpoint.{key}", checkpoint.get(key))

    for key in ("title", "objective", "current_objective", "next_action", "status"):
        if isinstance(active_workstream, dict):
            _add_candidate(candidates, f"active_workstream.{key}", active_workstream.get(key))
    if isinstance(active_workstream, dict):
        workstream = active_workstream.get("workstream")
        if isinstance(workstream, dict):
            for key in ("title", "objective", "current_objective", "next_action", "status"):
                _add_candidate(candidates, f"active_workstream.workstream.{key}", workstream.get(key))

    if isinstance(workstream_activation, dict):
        for key in ("title", "objective", "current_objective", "next_action", "status"):
            _add_candidate(candidates, f"workstream_activation.{key}", workstream_activation.get(key))
        for text in _iter_text_values(workstream_activation.get("candidates")):
            _add_candidate(candidates, "workstream_activation.candidates", text)
            if len(candidates) >= MAX_CANDIDATES:
                return candidates

    return candidates[:MAX_CANDIDATES]


def _normalize_status(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in STATUS_ALIASES:
        return STATUS_ALIASES[text]
    statuses = _status_tokens(text)
    return statuses[0] if len(statuses) == 1 else None


def _conflict_for_candidate(evidence: Any, candidate: dict[str, str]) -> dict[str, Any] | None:
    subject = str(_evidence_get(evidence, "subject", "") or "").strip()
    evidence_status = _normalize_status(_evidence_get(evidence, "value"))
    if not subject or not evidence_status:
        return None

    snippet = candidate["snippet"]
    if not _exact_token_present(snippet, subject):
        return None

    for candidate_status in _status_tokens(snippet):
        if (evidence_status, candidate_status) in CONTRADICTIONS:
            verified = bool(_evidence_get(evidence, "verified", False))
            authority = str(_evidence_get(evidence, "authority", "") or "")
            trust = (
                "prefer_live_evidence"
                if verified and authority in TRUSTED_AUTHORITIES
                else "advisory_conflict_only"
            )
            return {
                "subject": subject,
                "attribute": _evidence_get(evidence, "attribute", "status"),
                "evidence_value": _evidence_get(evidence, "value"),
                "evidence_normalized": evidence_status,
                "context_value": candidate_status,
                "context_source": candidate["source"],
                "context_snippet": snippet,
                "evidence_source": _evidence_get(evidence, "source"),
                "authority": authority,
                "verified": verified,
                "verification_method": _evidence_get(evidence, "verification_method"),
                "evidence_ref": _evidence_get(evidence, "evidence_ref"),
                "trust": trust,
            }
    return None


def arbitrate_current_state(
    *,
    message: str,
    activated_concepts: list,
    resume_context: str | None,
    working_context: dict | None,
    active_workstream: dict | None,
    workstream_activation: dict | None,
    current_state_evidence: list | None,
) -> dict:
    """Compare current-state evidence against context surfaces."""

    del message  # The hook keeps this parameter for future trace expansion.
    started = time.perf_counter()
    evidence_items = list(current_state_evidence or [])
    if not evidence_items:
        return {
            "conflicts": [],
            "decision": None,
            "trace": {"evidence_count": 0, "candidate_claim_count": 0, "elapsed_ms": 0.0},
        }

    candidates = _collect_candidates(
        activated_concepts=activated_concepts,
        resume_context=resume_context,
        working_context=working_context,
        active_workstream=active_workstream,
        workstream_activation=workstream_activation,
    )
    conflicts: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    invalid_evidence_count = 0

    for evidence in evidence_items[:10]:
        if _normalize_status(_evidence_get(evidence, "value")) is None:
            invalid_evidence_count += 1
            continue
        for candidate in candidates:
            conflict = _conflict_for_candidate(evidence, candidate)
            if not conflict:
                continue
            key = (
                str(conflict["subject"]),
                str(conflict["evidence_normalized"]),
                str(conflict["context_value"]),
                str(conflict["context_source"]),
            )
            if key not in seen:
                seen.add(key)
                conflicts.append(conflict)

    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    trusted_conflicts = [item for item in conflicts if item.get("trust") == "prefer_live_evidence"]
    if trusted_conflicts:
        decision = {
            "decision": "used_with_caution",
            "trust": "prefer_live_evidence",
            "reason": "verified current-state evidence contradicts activated context",
            "conflict_count": len(conflicts),
        }
    elif conflicts:
        decision = {
            "decision": "used_with_caution",
            "trust": "advisory_conflict_only",
            "reason": "unverified current-state evidence contradicts activated context",
            "conflict_count": len(conflicts),
        }
    else:
        decision = None

    _record_metric("current_state_arbitration.conflict_count", float(len(conflicts)))
    _record_metric("current_state_arbitration.invalid_evidence_count", float(invalid_evidence_count))
    _record_metric("current_state_arbitration.elapsed_ms", elapsed_ms)

    return {
        "conflicts": conflicts,
        "decision": decision,
        "trace": {
            "evidence_count": len(evidence_items[:10]),
            "candidate_claim_count": len(candidates),
            "elapsed_ms": elapsed_ms,
        },
    }
