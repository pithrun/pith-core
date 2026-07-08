"""Exact supersession-chain current-head rescue for conversation retrieval."""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from app.core.models import ActivatedConcept, SearchResult
from app.retrieval.temporal import walk_to_chain_head
from app.session.context_freshness import classify_freshness_intent

_CONCEPT_ID_TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_:.-]{2,127}\b")
_BLOCKED_HEAD_STATUSES = {"archived", "deleted", "superseded"}
_BLOCKED_HEAD_MATURITIES = {"QUARANTINED", "DISCARDED"}
_CURRENT_CURRENCY = {"ACTIVE"}
_SUPERSEDED_CURRENCY = {"SUPERSEDED"}
_SUBJECT_RESCUE_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SUBJECT_RESCUE_STOPWORDS = {
    "about",
    "after",
    "also",
    "before",
    "being",
    "current",
    "decision",
    "does",
    "from",
    "have",
    "into",
    "that",
    "their",
    "there",
    "this",
    "what",
    "when",
    "where",
    "which",
    "with",
    "your",
}
_SUBJECT_RESCUE_MAX_FTS_TERMS = 3
_SUBJECT_RESCUE_MIN_SCORE = 3.0
_SUBJECT_RESCUE_MIN_DIMENSIONS = 2
_SUBJECT_CHAIN_FALLBACK_MAX_QUERY_TERMS = 5
_SUBJECT_CHAIN_FALLBACK_MIN_SCORE = 8.0
_SUBJECT_CHAIN_FALLBACK_MIN_DIMENSIONS = 3
_SUBJECT_CHAIN_FALLBACK_AMBIGUITY_MARGIN = 1.0


def _subject_token_stem(token: str) -> str:
    if token.endswith("ing") and len(token) > 6:
        return token[:-3]
    if token.endswith("ies") and len(token) > 5:
        return f"{token[:-3]}y"
    if token.endswith("s") and len(token) > 4:
        return token[:-1]
    return token


def _tokenize_subject_text(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw in _SUBJECT_RESCUE_TOKEN_RE.findall((text or "").casefold().replace("_", " ")):
        if len(raw) < 4 or raw in _SUBJECT_RESCUE_STOPWORDS:
            continue
        tokens.add(raw)
        tokens.add(_subject_token_stem(raw))
    return {token for token in tokens if token}


def _fts_term_hit_count(conn: Any, term: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM fts_concepts WHERE fts_concepts MATCH ?",
        (term,),
    ).fetchone()
    return int(_row_value(row, "COUNT(*)", 0) or 0) if row is not None else 0


def _subject_rescue_terms(
    message: str,
    *,
    conn: Any,
    max_query_terms: int,
    max_term_hits: int,
    max_fts_terms: int | None = None,
) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for token in _SUBJECT_RESCUE_TOKEN_RE.findall((message or "").casefold()):
        term = _subject_token_stem(token)
        if len(term) < 4 or term in _SUBJECT_RESCUE_STOPWORDS or term in seen:
            continue
        seen.add(term)
        try:
            hit_count = _fts_term_hit_count(conn, term)
        except Exception:
            continue
        if hit_count <= 0 or hit_count > max_term_hits:
            continue
        terms.append(term)
        if len(terms) >= max_query_terms:
            break
    term_cap = _SUBJECT_RESCUE_MAX_FTS_TERMS if max_fts_terms is None else max_fts_terms
    return terms[: max(0, int(term_cap))]


def _row_mapping(row: Any) -> dict[str, Any]:
    if hasattr(row, "keys"):
        return {key: row[key] for key in row.keys()}
    keys = ("id", "status", "currency_status", "superseded_by", "is_current", "maturity", "knowledge_area", "subject_key", "summary")
    return {key: row[index] if index < len(row) else None for index, key in enumerate(keys)}


def _score_subject_chain_row(row: Mapping[str, Any], query_terms: set[str]) -> tuple[float, set[str]]:
    dimensions: set[str] = set()
    score = 0.0
    id_tokens = _tokenize_subject_text(str(row.get("id") or ""))
    summary_tokens = _tokenize_subject_text(str(row.get("summary") or ""))
    ka_tokens = _tokenize_subject_text(str(row.get("knowledge_area") or ""))
    subject_tokens = _tokenize_subject_text(str(row.get("subject_key") or ""))

    if query_terms & id_tokens:
        dimensions.add("id")
        score += 2.0
    if query_terms & summary_tokens:
        dimensions.add("summary")
        score += 1.5 * len(query_terms & summary_tokens)
    if query_terms & ka_tokens:
        dimensions.add("knowledge_area")
        score += 2.0
    if query_terms & subject_tokens:
        dimensions.add("subject_key")
        score += 2.0
    return score, dimensions


def _is_current_head_row(row: Mapping[str, Any]) -> bool:
    return (
        int(row.get("is_current") or 0) == 1
        and str(row.get("currency_status") or "").upper() in _CURRENT_CURRENCY
        and not row.get("superseded_by")
    )


def _score_subject_chain_rows(rows: list[Mapping[str, Any]], query_terms: set[str]) -> tuple[float, set[str]]:
    dimensions: set[str] = set()
    score = 0.0
    current_summary_tokens: set[str] = set()
    tail_id_tokens: set[str] = set()
    tail_summary_tokens: set[str] = set()
    knowledge_area_tokens: set[str] = set()
    subject_key_tokens: set[str] = set()

    for row in rows:
        id_tokens = _tokenize_subject_text(str(row.get("id") or ""))
        summary_tokens = _tokenize_subject_text(str(row.get("summary") or ""))
        knowledge_area_tokens |= _tokenize_subject_text(str(row.get("knowledge_area") or ""))
        subject_key_tokens |= _tokenize_subject_text(str(row.get("subject_key") or ""))
        if _is_current_head_row(row):
            current_summary_tokens |= summary_tokens
        else:
            tail_id_tokens |= id_tokens
            tail_summary_tokens |= summary_tokens

    current_summary_overlap = query_terms & current_summary_tokens
    tail_id_overlap = query_terms & tail_id_tokens
    tail_summary_overlap = query_terms & tail_summary_tokens
    knowledge_area_overlap = query_terms & knowledge_area_tokens
    subject_key_overlap = query_terms & subject_key_tokens

    if current_summary_overlap:
        dimensions.add("current_summary")
        score += 2.0 * len(current_summary_overlap)
    if tail_id_overlap:
        dimensions.add("tail_id")
        score += 2.5 * len(tail_id_overlap)
    if tail_summary_overlap:
        dimensions.add("tail_summary")
        score += 1.5 * len(tail_summary_overlap)
    if knowledge_area_overlap:
        dimensions.add("knowledge_area")
        score += 1.5 * len(knowledge_area_overlap)
    if subject_key_overlap:
        dimensions.add("subject_key")
        score += 2.0 * len(subject_key_overlap)
    if len(current_summary_overlap) >= 2:
        dimensions.add("current_summary_multi")
        score += 1.0
    chain_tokens = current_summary_tokens | tail_id_tokens | tail_summary_tokens | knowledge_area_tokens | subject_key_tokens
    if len(query_terms & chain_tokens) >= 3:
        dimensions.add("chain_coverage")
        score += 1.0
    return score, dimensions


def _aliased_concept_row(row: Any, prefix: str) -> dict[str, Any]:
    return {
        "id": _row_value(row, f"{prefix}_id", 0),
        "status": _row_value(row, f"{prefix}_status", 1),
        "currency_status": _row_value(row, f"{prefix}_currency_status", 2),
        "superseded_by": _row_value(row, f"{prefix}_superseded_by", 3),
        "is_current": _row_value(row, f"{prefix}_is_current", 4),
        "maturity": _row_value(row, f"{prefix}_maturity", 5),
        "knowledge_area": _row_value(row, f"{prefix}_knowledge_area", 6),
        "subject_key": _row_value(row, f"{prefix}_subject_key", 7),
        "summary": _row_value(row, f"{prefix}_summary", 8),
    }


@dataclass(slots=True)
class SupersessionChainRescueResult:
    attempted: bool = False
    matched_tokens: list[str] = field(default_factory=list)
    intent: str = "neutral"
    source: str | None = None
    tail_id: str | None = None
    head_id: str | None = None
    chain_depth: int = 0
    head_result: SearchResult | None = None
    tail_result: SearchResult | None = None
    protected_results: dict[str, SearchResult] = field(default_factory=dict)
    protected_ids: set[str] = field(default_factory=set)
    historical_tail_admitted: bool = False
    candidate_head_count: int = 0
    predecessor_count: int = 0
    skipped_reason: str | None = None
    rejected: list[dict[str, str]] = field(default_factory=list)
    latency_ms: float = 0.0
    subject_fallback_attempted: bool = False
    subject_fallback_admitted: bool = False
    subject_fallback_reason: str | None = None
    subject_fallback_candidate_count: int = 0
    subject_fallback_best_score: float | None = None
    subject_fallback_runner_up_delta: float | None = None

    def to_trace(self) -> dict[str, Any]:
        return {
            "schema_version": "supersession_chain_rescue.v2",
            "attempted": self.attempted,
            "matched_token_count": len(self.matched_tokens),
            "intent": self.intent,
            "source": self.source,
            "tail_id": self.tail_id,
            "head_id": self.head_id,
            "chain_depth": self.chain_depth,
            "head_admitted": self.head_result is not None,
            "historical_tail_admitted": self.historical_tail_admitted,
            "candidate_head_count": self.candidate_head_count,
            "predecessor_count": self.predecessor_count,
            "protected_ids": sorted(self.protected_ids)[:5],
            "rejected": self.rejected[:5],
            "skipped_reason": self.skipped_reason,
            "latency_ms": round(float(self.latency_ms or 0.0), 4),
            "subject_fallback_attempted": self.subject_fallback_attempted,
            "subject_fallback_admitted": self.subject_fallback_admitted,
            "subject_fallback_reason": self.subject_fallback_reason,
            "subject_fallback_candidate_count": self.subject_fallback_candidate_count,
            "subject_fallback_best_score": (
                round(float(self.subject_fallback_best_score), 4)
                if self.subject_fallback_best_score is not None
                else None
            ),
            "subject_fallback_runner_up_delta": (
                round(float(self.subject_fallback_runner_up_delta), 4)
                if self.subject_fallback_runner_up_delta is not None
                else None
            ),
        }


def _supersession_chain_rescue_metric_events(
    result: SupersessionChainRescueResult,
) -> list[tuple[str, float, dict[str, str] | None]]:
    events: list[tuple[str, float, dict[str, str] | None]] = [
        ("supersession_chain_rescue_attempted_total", 1.0, None),
    ]
    if result.source == "subject_chain":
        events.append(("supersession_chain_rescue_subject_attempted_total", 1.0, None))

    if result.head_result is not None:
        events.append(("supersession_chain_rescue_head_admitted_total", 1.0, None))
        if result.source == "subject_chain":
            events.append(("supersession_chain_rescue_subject_head_admitted_total", 1.0, None))
        if result.historical_tail_admitted:
            events.append(("supersession_chain_rescue_historical_tail_admitted_total", 1.0, None))
        elif result.intent == "current_state":
            events.append(("supersession_chain_rescue_current_head_only_total", 1.0, None))
        if result.source == "candidate_chain":
            events.append(("supersession_chain_rescue_candidate_chain_total", 1.0, None))
    elif result.rejected:
        events.append(("supersession_chain_rescue_rejected_total", 1.0, None))
        if result.source == "subject_chain":
            events.append(("supersession_chain_rescue_subject_rejected_total", 1.0, None))
        if result.source == "candidate_chain":
            events.append(("supersession_chain_rescue_candidate_chain_rejected_total", 1.0, None))
    elif result.skipped_reason:
        reason = str(result.skipped_reason or "unknown")
        source = str(result.source or "unknown")
        events.append(("supersession_chain_rescue_skipped_total", 1.0, {"source": source, "reason": reason}))
        if result.source == "subject_chain":
            events.append(("supersession_chain_rescue_subject_skipped_total", 1.0, {"reason": reason}))
    else:
        events.append(("supersession_chain_rescue_no_head_total", 1.0, None))

    events.append(
        (
            "supersession_chain_rescue_latency_ms",
            round(float(result.latency_ms or 0.0), 4),
            None,
        )
    )
    if result.source == "subject_chain":
        events.append(
            (
                "supersession_chain_rescue_subject_latency_ms",
                round(float(result.latency_ms or 0.0), 4),
                None,
            )
        )
    return events


def _looks_like_concept_id(token: str) -> bool:
    return "_" in token or ":" in token or "-" in token or "." in token or any(char.isdigit() for char in token)


def extract_exact_concept_id_tokens(message: str, *, max_tokens: int = 5) -> list[str]:
    if not message:
        return []
    tokens: list[str] = []
    for match in _CONCEPT_ID_TOKEN_RE.finditer(message):
        token = match.group(0).strip(".,;:!?()[]{}\"'")
        if token and _looks_like_concept_id(token) and token not in tokens:
            tokens.append(token)
        if len(tokens) >= max_tokens:
            break
    return tokens


def _row_value(row: Any, key: str, index: int) -> Any:
    if hasattr(row, "keys"):
        return row[key]
    return row[index]


def _valid_current_head(concept: Any) -> bool:
    if concept is None:
        return False
    head_status = str(getattr(concept, "status", "active") or "active").lower()
    head_currency = str(getattr(concept, "currency_status", "ACTIVE") or "ACTIVE").upper()
    head_maturity = str(getattr(concept, "maturity", "ESTABLISHED") or "ESTABLISHED").upper()
    if head_status in _BLOCKED_HEAD_STATUSES:
        return False
    if head_currency not in _CURRENT_CURRENCY:
        return False
    if getattr(concept, "is_current", 1) != 1:
        return False
    return head_maturity not in _BLOCKED_HEAD_MATURITIES


def _concept_to_search_result(
    concept: Any,
    *,
    relevance_score: float,
    rescue_reason: str,
    tail_id: str | None = None,
    head_id: str | None = None,
    intent: str = "neutral",
) -> SearchResult:
    metadata = {
        "rescue_reason": rescue_reason,
        "intent": intent,
    }
    if tail_id:
        metadata["tail_id"] = tail_id
    if head_id:
        metadata["head_id"] = head_id
    return SearchResult(
        concept_id=getattr(concept, "id", None) or concept.concept_id,
        version=getattr(concept, "version", "v1") or "v1",
        summary=getattr(concept, "summary", "") or "",
        confidence=float(getattr(concept, "confidence", 0.0) or 0.0),
        relevance_score=relevance_score,
        knowledge_area=getattr(concept, "knowledge_area", None),
        ka_relative_authority=getattr(concept, "ka_relative_authority", None),
        maturity=getattr(concept, "maturity", None),
        created_at=getattr(concept, "created_at", None),
        metadata=metadata,
    )


def _add_protected_result(result: SupersessionChainRescueResult, search_result: SearchResult) -> None:
    result.protected_results[search_result.concept_id] = search_result
    result.protected_ids.add(search_result.concept_id)


def resolve_exact_supersession_head(
    message: str,
    *,
    conn: Any,
    load_concept_fn: Callable[..., Any],
    enabled: bool,
    max_tokens: int = 5,
    max_depth: int = 8,
) -> SupersessionChainRescueResult:
    started = time.perf_counter()
    result = SupersessionChainRescueResult(attempted=bool(enabled))
    try:
        if not enabled:
            result.skipped_reason = "disabled"
            return result

        result.intent = classify_freshness_intent(message)
        result.source = "exact_id"
        result.matched_tokens = extract_exact_concept_id_tokens(message, max_tokens=max_tokens)
        if not result.matched_tokens:
            result.skipped_reason = "no_exact_tokens"
            return result
        for token in result.matched_tokens:
            row = conn.execute(
                """
                SELECT id, status, currency_status, superseded_by, is_current, maturity
                FROM concepts
                WHERE id = ?
                """,
                (token,),
            ).fetchone()
            if row is None:
                result.rejected.append({"token": token, "reason": "unknown_token"})
                continue

            tail_status = str(_row_value(row, "status", 1) or "").lower()
            tail_currency = str(_row_value(row, "currency_status", 2) or "").upper()
            tail_superseded_by = _row_value(row, "superseded_by", 3) or ""
            is_superseded_tail = (
                tail_status == "superseded" or tail_currency in _SUPERSEDED_CURRENCY or bool(tail_superseded_by)
            )
            if not is_superseded_tail:
                result.rejected.append({"token": token, "reason": "not_superseded"})
                continue

            head_id, chain_depth = walk_to_chain_head(token, conn, max_depth=max_depth)
            if not head_id:
                result.rejected.append({"token": token, "reason": "head_missing"})
                continue

            head = load_concept_fn(head_id, track_access=False)
            if head is None:
                result.rejected.append({"token": token, "reason": "head_load_failed"})
                continue

            if not _valid_current_head(head):
                head_status = str(getattr(head, "status", "active") or "active").lower()
                head_currency = str(getattr(head, "currency_status", "ACTIVE") or "ACTIVE").upper()
                head_maturity = str(getattr(head, "maturity", "ESTABLISHED") or "ESTABLISHED").upper()
                if head_status in _BLOCKED_HEAD_STATUSES:
                    result.rejected.append({"token": token, "reason": "head_status_rejected"})
                    continue
                if head_currency not in _CURRENT_CURRENCY:
                    result.rejected.append({"token": token, "reason": "head_currency_rejected"})
                    continue
                if getattr(head, "is_current", 1) != 1:
                    result.rejected.append({"token": token, "reason": "head_not_current"})
                    continue
                if head_maturity in _BLOCKED_HEAD_MATURITIES:
                    result.rejected.append({"token": token, "reason": "head_maturity_rejected"})
                    continue
                result.rejected.append({"token": token, "reason": "head_status_rejected"})
                continue

            result.tail_id = token
            result.head_id = head_id
            result.chain_depth = int(chain_depth or 0)
            result.head_result = _concept_to_search_result(
                head,
                relevance_score=1.0,
                rescue_reason="supersession_chain_current_head",
                tail_id=token,
                head_id=head_id,
                intent=result.intent,
            )
            _add_protected_result(result, result.head_result)
            if result.intent == "historical":
                tail = load_concept_fn(token, track_access=False)
                if tail is None:
                    result.rejected.append({"token": token, "reason": "tail_load_failed"})
                else:
                    result.tail_result = _concept_to_search_result(
                        tail,
                        relevance_score=0.999,
                        rescue_reason="supersession_chain_historical_tail",
                        tail_id=token,
                        head_id=head_id,
                        intent=result.intent,
                    )
                    _add_protected_result(result, result.tail_result)
                    result.historical_tail_admitted = True
            break
        if not result.protected_results and not result.skipped_reason:
            result.skipped_reason = "no_valid_supersession_head"
        return result
    finally:
        result.latency_ms = (time.perf_counter() - started) * 1000.0


def restore_protected_search_results(
    results: list[SearchResult],
    protected_results: dict[str, SearchResult],
    *,
    max_items: int | None = None,
) -> tuple[list[SearchResult], list[str]]:
    if not protected_results:
        return results, []

    by_id = {result.concept_id: result for result in results}
    restored = [protected for concept_id, protected in protected_results.items() if concept_id not in by_id]
    if not restored:
        return results, []

    merged = list(results) + restored
    merged.sort(key=lambda result: getattr(result, "relevance_score", 0.0) or 0.0, reverse=True)
    if max_items is not None and len(merged) > max_items:
        protected_ids = set(protected_results)
        protected = [result for result in merged if result.concept_id in protected_ids]
        ordinary = [result for result in merged if result.concept_id not in protected_ids]
        merged = (protected + ordinary)[:max_items]
    return merged, [result.concept_id for result in restored]


def resolve_candidate_supersession_chains(
    message: str,
    search_results: list[SearchResult],
    *,
    conn: Any,
    load_concept_fn: Callable[..., Any],
    enabled: bool,
    max_candidates: int = 3,
    max_predecessors_per_head: int = 2,
) -> SupersessionChainRescueResult:
    started = time.perf_counter()
    result = SupersessionChainRescueResult(
        attempted=bool(enabled),
        intent=classify_freshness_intent(message),
        source="candidate_chain",
    )
    try:
        if not enabled:
            result.skipped_reason = "disabled"
            return result
        if result.intent == "neutral":
            result.skipped_reason = "neutral_intent"
            return result
        if not search_results:
            result.skipped_reason = "no_search_results"
            return result

        for search_result in list(search_results)[: max(0, max_candidates)]:
            head_id = getattr(search_result, "concept_id", None)
            if not head_id:
                continue
            head = load_concept_fn(head_id, track_access=False)
            if not _valid_current_head(head):
                result.rejected.append({"token": str(head_id), "reason": "candidate_not_current_head"})
                continue

            predecessor_rows = conn.execute(
                """
                SELECT id
                FROM concepts
                WHERE superseded_by = ?
                  AND (status = 'superseded' OR currency_status = 'SUPERSEDED' OR is_current = 0)
                LIMIT ?
                """,
                (head_id, int(max_predecessors_per_head)),
            ).fetchall()
            if not predecessor_rows:
                result.rejected.append({"token": str(head_id), "reason": "candidate_no_predecessors"})
                continue

            result.candidate_head_count += 1
            result.head_id = str(head_id)
            result.head_result = _concept_to_search_result(
                head,
                relevance_score=max(float(getattr(search_result, "relevance_score", 0.0) or 0.0), 1.0),
                rescue_reason="supersession_chain_candidate_head",
                head_id=str(head_id),
                intent=result.intent,
            )
            _add_protected_result(result, result.head_result)
            if result.intent == "current_state":
                break

            for row in predecessor_rows:
                predecessor_id = str(_row_value(row, "id", 0))
                tail = load_concept_fn(predecessor_id, track_access=False)
                if tail is None:
                    result.rejected.append({"token": predecessor_id, "reason": "predecessor_load_failed"})
                    continue
                tail_result = _concept_to_search_result(
                    tail,
                    relevance_score=0.999,
                    rescue_reason="supersession_chain_candidate_historical_tail",
                    tail_id=predecessor_id,
                    head_id=str(head_id),
                    intent=result.intent,
                )
                if result.tail_result is None:
                    result.tail_result = tail_result
                    result.tail_id = predecessor_id
                result.predecessor_count += 1
                result.historical_tail_admitted = True
                _add_protected_result(result, tail_result)
            if result.protected_results:
                break

        if not result.protected_results and not result.skipped_reason:
            result.skipped_reason = "no_candidate_chain"
        return result
    finally:
        result.latency_ms = (time.perf_counter() - started) * 1000.0


def _attempt_subject_chain_fallback(
    message: str,
    *,
    result: SupersessionChainRescueResult,
    conn: Any,
    load_concept_fn: Callable[..., Any],
    max_query_terms: int,
    max_term_hits: int,
) -> bool:
    result.subject_fallback_attempted = True
    terms = _subject_rescue_terms(
        message,
        conn=conn,
        max_query_terms=max_query_terms,
        max_term_hits=max_term_hits,
        max_fts_terms=_SUBJECT_CHAIN_FALLBACK_MAX_QUERY_TERMS,
    )
    if len(terms) < 2:
        result.subject_fallback_reason = "subject_chain_fallback_terms_too_sparse"
        result.skipped_reason = result.subject_fallback_reason
        return False

    result.matched_tokens = terms
    query_terms = set(terms)
    term_filter_clauses: list[str] = []
    term_filter_params: list[str] = []
    for term in terms:
        like_term = f"%{term}%"
        for column in (
            "h.summary",
            "h.subject_key",
            "p.id",
            "p.summary",
            "p.subject_key",
        ):
            term_filter_clauses.append(f"COALESCE({column}, '') LIKE ?")
            term_filter_params.append(like_term)
    term_filter_sql = " OR ".join(term_filter_clauses)
    rows = conn.execute(
        f"""
        SELECT
            h.id AS head_id,
            h.status AS head_status,
            h.currency_status AS head_currency_status,
            h.superseded_by AS head_superseded_by,
            h.is_current AS head_is_current,
            h.maturity AS head_maturity,
            h.knowledge_area AS head_knowledge_area,
            h.subject_key AS head_subject_key,
            h.summary AS head_summary,
            p.id AS predecessor_id,
            p.status AS predecessor_status,
            p.currency_status AS predecessor_currency_status,
            p.superseded_by AS predecessor_superseded_by,
            p.is_current AS predecessor_is_current,
            p.maturity AS predecessor_maturity,
            p.knowledge_area AS predecessor_knowledge_area,
            p.subject_key AS predecessor_subject_key,
            p.summary AS predecessor_summary
        FROM concepts h
        JOIN concepts p ON p.superseded_by = h.id
        WHERE h.is_current = 1
          AND h.currency_status = 'ACTIVE'
          AND h.superseded_by IS NULL
          AND (
            p.status = 'superseded'
            OR p.currency_status = 'SUPERSEDED'
            OR p.is_current = 0
          )
          AND ({term_filter_sql})
        """,
        tuple(term_filter_params),
    ).fetchall()
    if not rows:
        result.subject_fallback_reason = "subject_chain_fallback_no_candidates"
        result.skipped_reason = result.subject_fallback_reason
        return False

    chains: dict[str, list[dict[str, Any]]] = {}
    predecessor_ids: dict[str, list[str]] = {}
    for raw_row in rows:
        head_row = _aliased_concept_row(raw_row, "head")
        predecessor_row = _aliased_concept_row(raw_row, "predecessor")
        head_id = str(head_row.get("id") or "")
        predecessor_id = str(predecessor_row.get("id") or "")
        if not head_id or not predecessor_id:
            continue
        chain_rows = chains.setdefault(head_id, [head_row])
        chain_rows.append(predecessor_row)
        predecessor_ids.setdefault(head_id, []).append(predecessor_id)

    scored_chains: list[tuple[float, str, set[str]]] = []
    for head_id, chain_rows in chains.items():
        score, dimensions = _score_subject_chain_rows(chain_rows, query_terms)
        if score < _SUBJECT_CHAIN_FALLBACK_MIN_SCORE or len(dimensions) < _SUBJECT_CHAIN_FALLBACK_MIN_DIMENSIONS:
            continue
        scored_chains.append((score, head_id, dimensions))

    scored_chains.sort(key=lambda item: (-item[0], item[1]))
    result.subject_fallback_candidate_count = len(scored_chains)
    if not scored_chains:
        result.subject_fallback_reason = "subject_chain_fallback_no_qualifying_chain"
        result.skipped_reason = result.subject_fallback_reason
        return False

    best_score, best_head_id, _best_dimensions = scored_chains[0]
    result.subject_fallback_best_score = best_score
    if len(scored_chains) > 1:
        runner_up_delta = best_score - scored_chains[1][0]
        result.subject_fallback_runner_up_delta = runner_up_delta
        if runner_up_delta < _SUBJECT_CHAIN_FALLBACK_AMBIGUITY_MARGIN:
            result.subject_fallback_reason = "subject_chain_fallback_ambiguous"
            result.skipped_reason = result.subject_fallback_reason
            result.rejected.append({"token": best_head_id, "reason": "subject_chain_fallback_ambiguous"})
            return False

    head = load_concept_fn(best_head_id, track_access=False)
    if not _valid_current_head(head):
        result.subject_fallback_reason = "subject_chain_fallback_head_rejected"
        result.skipped_reason = result.subject_fallback_reason
        result.rejected.append({"token": best_head_id, "reason": "head_rejected"})
        return False

    first_predecessor = predecessor_ids.get(best_head_id, [None])[0]
    result.head_id = best_head_id
    result.tail_id = first_predecessor
    result.chain_depth = 1
    result.predecessor_count = len(predecessor_ids.get(best_head_id, []))
    result.head_result = _concept_to_search_result(
        head,
        relevance_score=1.0,
        rescue_reason="supersession_chain_subject_fallback_head",
        tail_id=first_predecessor,
        head_id=best_head_id,
        intent=result.intent,
    )
    _add_protected_result(result, result.head_result)
    result.candidate_head_count += 1
    result.subject_fallback_admitted = True
    result.subject_fallback_reason = "subject_chain_fallback_admitted"
    result.skipped_reason = None
    return True


def resolve_subject_supersession_chain_candidates(
    message: str,
    *,
    conn: Any,
    load_concept_fn: Callable[..., Any],
    enabled: bool,
    max_query_terms: int = 8,
    max_term_hits: int = 800,
    max_rows: int = 25,
    max_heads: int = 3,
) -> SupersessionChainRescueResult:
    started = time.perf_counter()
    result = SupersessionChainRescueResult(
        attempted=bool(enabled),
        intent=classify_freshness_intent(message),
        source="subject_chain",
    )
    try:
        if not enabled:
            result.skipped_reason = "disabled"
            return result
        if result.intent != "current_state":
            result.skipped_reason = "non_current_intent" if result.intent == "historical" else "neutral_intent"
            return result

        terms = _subject_rescue_terms(
            message,
            conn=conn,
            max_query_terms=max_query_terms,
            max_term_hits=max_term_hits,
        )
        result.matched_tokens = terms
        if not terms:
            result.skipped_reason = "no_subject_terms"
            return result
        if len(terms) < 2:
            result.skipped_reason = "subject_terms_too_sparse"
            return result

        query_terms = set(terms)
        match_query = " OR ".join(terms)
        fts_rows = conn.execute(
            """
            SELECT concept_id
            FROM fts_concepts
            WHERE fts_concepts MATCH ?
            ORDER BY bm25(fts_concepts)
            LIMIT ?
            """,
            (match_query, int(max_rows)),
        ).fetchall()
        concept_ids: list[str] = []
        seen_concept_ids: set[str] = set()
        for fts_row in fts_rows:
            concept_id = str(_row_value(fts_row, "concept_id", 0) or "")
            if concept_id and concept_id not in seen_concept_ids:
                concept_ids.append(concept_id)
                seen_concept_ids.add(concept_id)
        if not concept_ids:
            if _attempt_subject_chain_fallback(
                message,
                result=result,
                conn=conn,
                load_concept_fn=load_concept_fn,
                max_query_terms=max_query_terms,
                max_term_hits=max_term_hits,
            ):
                return result
            if not result.skipped_reason:
                result.skipped_reason = "no_subject_candidates"
            return result

        placeholders = ",".join("?" for _ in concept_ids)
        concept_rows = conn.execute(
            f"""
            SELECT
                c.id,
                c.status,
                c.currency_status,
                c.superseded_by,
                c.is_current,
                c.maturity,
                c.knowledge_area,
                c.subject_key,
                c.summary
            FROM concepts c
            WHERE c.id IN ({placeholders})
            """,
            tuple(concept_ids),
        ).fetchall()
        concept_rows_by_id = {_row_value(row, "id", 0): row for row in concept_rows}
        rows = []
        for concept_id in concept_ids:
            raw_row = concept_rows_by_id.get(concept_id)
            if raw_row is None:
                continue
            row_superseded_by = _row_value(raw_row, "superseded_by", 3)
            row_status = str(_row_value(raw_row, "status", 1) or "").lower()
            row_currency = str(_row_value(raw_row, "currency_status", 2) or "").upper()
            row_is_current = int(_row_value(raw_row, "is_current", 4) or 0) == 1
            is_superseded_tail = bool(row_superseded_by) and (
                row_status == "superseded" or row_currency in _SUPERSEDED_CURRENCY or not row_is_current
            )
            is_current_head = row_is_current and row_currency in _CURRENT_CURRENCY and not row_superseded_by
            has_predecessor = False
            if is_current_head:
                has_predecessor = (
                    conn.execute(
                        "SELECT 1 FROM concepts WHERE superseded_by = ? LIMIT 1",
                        (concept_id,),
                    ).fetchone()
                    is not None
                )
            if is_superseded_tail or has_predecessor:
                rows.append(raw_row)
        if not rows:
            if _attempt_subject_chain_fallback(
                message,
                result=result,
                conn=conn,
                load_concept_fn=load_concept_fn,
                max_query_terms=max_query_terms,
                max_term_hits=max_term_hits,
            ):
                return result
            if not result.skipped_reason:
                result.skipped_reason = "no_subject_candidates"
            return result

        scored_rows: list[tuple[float, dict[str, Any]]] = []
        for raw_row in rows:
            row = _row_mapping(raw_row)
            score, dimensions = _score_subject_chain_row(row, query_terms)
            if score < _SUBJECT_RESCUE_MIN_SCORE or len(dimensions) < _SUBJECT_RESCUE_MIN_DIMENSIONS:
                result.rejected.append({"token": str(row.get("id") or ""), "reason": "subject_score_too_low"})
                continue
            scored_rows.append((score, row))

        scored_rows.sort(key=lambda item: item[0], reverse=True)
        admitted_heads = 0
        admitted_head_ids: set[str] = set()
        for _score, row in scored_rows:
            row_id = str(row.get("id") or "")
            if not row_id:
                continue
            row_superseded_by = row.get("superseded_by")
            row_is_current = int(row.get("is_current") or 0) == 1
            row_currency = str(row.get("currency_status") or "").upper()
            is_head_candidate = row_is_current and row_currency in _CURRENT_CURRENCY and not row_superseded_by
            tail_id = None
            head_id = None
            chain_depth = 0
            if is_head_candidate:
                head_id = row_id
            else:
                tail_id = row_id
                head_id, chain_depth = walk_to_chain_head(tail_id, conn, max_depth=8)
            if not head_id:
                result.rejected.append({"token": row_id, "reason": "head_missing"})
                continue
            if str(head_id) in admitted_head_ids:
                continue

            head = load_concept_fn(head_id, track_access=False)
            if not _valid_current_head(head):
                result.rejected.append({"token": row_id, "reason": "head_rejected"})
                continue

            result.head_id = str(head_id)
            result.chain_depth = int(chain_depth or 0)
            result.head_result = _concept_to_search_result(
                head,
                relevance_score=1.0,
                rescue_reason="supersession_chain_subject_head",
                tail_id=tail_id,
                head_id=str(head_id),
                intent=result.intent,
            )
            _add_protected_result(result, result.head_result)
            result.candidate_head_count += 1
            admitted_heads += 1
            admitted_head_ids.add(str(head_id))

            if admitted_heads >= max_heads or result.intent == "current_state":
                break

        if not result.protected_results:
            _attempt_subject_chain_fallback(
                message,
                result=result,
                conn=conn,
                load_concept_fn=load_concept_fn,
                max_query_terms=max_query_terms,
                max_term_hits=max_term_hits,
            )

        if not result.protected_results and not result.skipped_reason:
            result.skipped_reason = "no_subject_chain"
        return result
    finally:
        result.latency_ms = (time.perf_counter() - started) * 1000.0


def restore_protected_activated_concepts(
    activated: list[ActivatedConcept],
    protected_activated: dict[str, ActivatedConcept],
    *,
    max_items: int,
) -> tuple[list[ActivatedConcept], list[str]]:
    if not protected_activated:
        return activated, []

    active_ids = {concept.concept_id for concept in activated}
    restored = [protected for concept_id, protected in protected_activated.items() if concept_id not in active_ids]
    if not restored:
        return activated, []

    protected_ids = set(protected_activated)
    protected = [concept for concept in activated if concept.concept_id in protected_ids] + restored
    ordinary = [concept for concept in activated if concept.concept_id not in protected_ids]
    return (protected + ordinary)[:max_items], [concept.concept_id for concept in restored]
