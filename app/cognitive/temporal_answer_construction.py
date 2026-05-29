"""Evidence-bound temporal answer construction candidates.

This module is deliberately pure: no storage, network, LLM, benchmark, session,
or runtime-policy imports belong here.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

STATUS_ACCEPTED = "accepted"
STATUS_FALLBACK = "fallback"

REASON_DIRECT_DURATION = "direct_duration"
REASON_DEICTIC_SOURCE_DATE = "deictic_source_date"
REASON_ENDPOINT_RANGE = "endpoint_range"
REASON_MISSING_PROVENANCE = "missing_provenance"
REASON_INVALID_PROVENANCE = "invalid_provenance"
REASON_SOURCE_DATE_CONFLICT = "source_date_conflict"
REASON_AMBIGUOUS_CANDIDATES = "ambiguous_candidates"
REASON_NO_EVENT_OVERLAP = "no_event_overlap"
REASON_NON_TEMPORAL_QUESTION = "non_temporal_question"
REASON_UNSUPPORTED_RELATIVE_DATE = "unsupported_relative_date"
REASON_NO_SUPPORTED_TEMPORAL_EVIDENCE = "no_supported_temporal_evidence"

_VALID_PROVENANCE = {"explicit"}
_INVALID_PROVENANCE = {"", "filename", "benchmark_source_key", "inferred", "unknown"}

_MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}
_MONTH_PATTERN = "|".join(sorted(_MONTHS, key=len, reverse=True))
_DATE_RE = re.compile(
    rf"\b(?P<month>{_MONTH_PATTERN})\.?\s+(?P<day>\d{{1,2}})(?:,\s*(?P<year>\d{{4}}))?\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
_DURATION_UNIT_PATTERN = r"days?|weeks?|months?|years?"
_DURATION_SPAN_PATTERN = rf"(?P<value>\d{{1,4}})\s+(?P<unit>{_DURATION_UNIT_PATTERN})"
_DIRECT_DURATION_RE = re.compile(
    r"\b(?:completed|finished|done|resolved|wrapped|shipped|implemented|closed)"
    r"\b[^.\n!?]{0,80}?\b(?:in|after|within|took)\s+"
    + _DURATION_SPAN_PATTERN
    + r"\b",
    re.IGNORECASE,
)
_REVERSED_DIRECT_DURATION_RE = re.compile(
    r"\b(?:in|after|within|took)\s+"
    + _DURATION_SPAN_PATTERN
    + r"\b[^.\n!?]{0,80}?\b(?:completed|finished|done|resolved|wrapped|shipped|implemented|closed)\b",
    re.IGNORECASE,
)
_STATE_DIRECT_DURATION_RE = re.compile(
    r"\b(?:has|have|had|is|was|were|been)\b[^.\n!?]{0,80}?\bfor\s+"
    + _DURATION_SPAN_PATTERN
    + r"\b",
    re.IGNORECASE,
)
_ALREADY_DIRECT_DURATION_RE = re.compile(
    r"\b(?:for\s+)?"
    + _DURATION_SPAN_PATTERN
    + r"\s+already\b",
    re.IGNORECASE,
)
_UNRELATED_DURATION_RE = re.compile(
    r"\b(?:waited|pause|paused|delay|delayed|planning|research|unrelated)\b"
    rf"[^.\n!?]{{0,80}}?\b\d{{1,4}}\s+(?:{_DURATION_UNIT_PATTERN})\b",
    re.IGNORECASE,
)
_TEMPORAL_QUESTION_TERMS = (
    "how long",
    "how many days",
    "how much time",
    "duration",
    "elapsed",
    "date range",
    "time period",
)
_RELATIVE_DATE_RE = re.compile(
    rf"\b(?:the\s+)?(?:day|week|month|year)s?\s+(?:before|after)\s+(?=(?:the\s+)?(?:{_MONTH_PATTERN}|\d))"
    r"|\b(?:yesterday|today|tomorrow)\b",
    re.IGNORECASE,
)
_CONVERSATION_HEADER_DATE_RE = re.compile(
    rf"\[conversation on [^\]]* on (?P<day>\d{{1,2}})\s+"
    rf"(?P<month>{_MONTH_PATTERN}),?\s+(?P<year>\d{{4}})\]",
    re.IGNORECASE,
)
_DEICTIC_WHEN_QUESTION_RE = re.compile(
    r"^\s*when\s+did\s+(?P<subject>[a-z][a-z' -]{0,60}?)\s+"
    r"(?P<verb>go(?:\s+(?:to|on))?|join|encounter|pass|attend|give(?:\s+a)?|"
    r"gave(?:\s+a)?|meet(?:\s+up(?:\s+with)?)?|met(?:\s+up(?:\s+with)?)?|"
    r"apply(?:\s+to)?|applied(?:\s+to)?)\s+"
    r"(?P<event>.+?)\??\s*$",
    re.IGNORECASE,
)
_LAST_FRIDAY_RE = re.compile(r"\blast\s+fri(?:day)?\b", re.IGNORECASE)
_LAST_TUESDAY_RE = re.compile(r"\blast\s+tues(?:day)?\b", re.IGNORECASE)
_LAST_WEEKEND_RE = re.compile(r"\b(?:last|this\s+past)\s+weekend\b", re.IGNORECASE)
_THIS_WEEK_RE = re.compile(r"\bthis\s+week\b", re.IGNORECASE)
_LAST_WEEK_RE = re.compile(r"\blast\s+week\b", re.IGNORECASE)
_DEICTIC_EVENT_STOPWORDS = frozenset(
    {
        "apply",
        "applied",
        "attend",
        "encounter",
        "gave",
        "give",
        "join",
        "meet",
        "pass",
        "with",
        "during",
        "spring",
        "summer",
        "autumn",
        "winter",
    }
)
_MONTH_TERMS = frozenset(_MONTHS)
_START_ROLE_RE = re.compile(
    r"\b(start|starts|started|began|kicked off|launched|opened|from|since|onboarded)\b",
    re.IGNORECASE,
)
_END_ROLE_RE = re.compile(
    r"\b(deadline|due by|due on|by|until|through|finished on|completed on|ended on|target date)\b",
    re.IGNORECASE,
)
_QUESTION_STOPWORDS = frozenset(
    {
        "after",
        "before",
        "been",
        "days",
        "did",
        "does",
        "has",
        "had",
        "have",
        "how",
        "long",
        "many",
        "much",
        "take",
        "time",
        "what",
        "when",
        "was",
        "were",
        "with",
    }
)


@dataclass(frozen=True)
class TemporalSupport:
    support_id: str
    text: str
    source_date: date | None = None
    source_year: int | None = None
    provenance: str = "explicit"


@dataclass(frozen=True)
class TemporalConstructionDecision:
    status: Literal["accepted", "fallback"]
    answer: str | None
    reason: str
    support_ids: tuple[str, ...]
    strategy: str | None = None
    diagnostics: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class _Endpoint:
    role: str
    text: str
    value: date
    support_id: str
    provenance_used: str


@dataclass(frozen=True)
class _DeicticQuestion:
    subject: str
    event_terms: tuple[str, ...]
    month_terms: tuple[str, ...]
    required_phrases: tuple[str, ...] = ()


@dataclass(frozen=True)
class _DeicticCandidate:
    answer: str
    support_id: str
    source_date: date
    deictic_phrase: str
    relative_relation: str
    matched_event_terms: tuple[str, ...]
    event_terms: tuple[str, ...]
    provenance_used: str
    resolved_date: date | None = None


def construct_temporal_answer_candidate(
    question: str,
    supports: Sequence[TemporalSupport],
) -> TemporalConstructionDecision:
    """Construct a temporal answer only for narrow, evidence-bound shapes."""
    if not _is_temporal_question(question):
        return _fallback(REASON_NON_TEMPORAL_QUESTION)

    valid_supports, invalid_support_ids = _partition_valid_supports(supports)
    if not valid_supports:
        reason = REASON_INVALID_PROVENANCE if invalid_support_ids else REASON_NO_SUPPORTED_TEMPORAL_EVIDENCE
        return _fallback(reason, support_ids=tuple(invalid_support_ids))

    relative_support_ids = tuple(
        support.support_id for support in valid_supports if _RELATIVE_DATE_RE.search(support.text or "")
    )
    deictic_decision = _select_deictic_source_date(question, valid_supports)
    if deictic_decision is not None:
        return deictic_decision

    if relative_support_ids:
        return _fallback(REASON_UNSUPPORTED_RELATIVE_DATE, support_ids=relative_support_ids)

    if _deictic_when_question_parts(question) is not None:
        return _fallback(REASON_NO_SUPPORTED_TEMPORAL_EVIDENCE)

    direct_duration = _select_direct_duration(question, valid_supports)
    if direct_duration is not None:
        duration, support_id = direct_duration
        return _accepted(
            duration,
            REASON_DIRECT_DURATION,
            support_ids=(support_id,),
            candidate_count=1,
            provenance_used=("support_text",),
        )

    range_decision = _select_endpoint_range(question, valid_supports)
    if range_decision is not None:
        return range_decision

    if _has_missing_year_endpoint(valid_supports):
        return _fallback(REASON_MISSING_PROVENANCE, support_ids=tuple(s.support_id for s in valid_supports))
    return _fallback(REASON_NO_SUPPORTED_TEMPORAL_EVIDENCE)


def _partition_valid_supports(
    supports: Sequence[TemporalSupport],
) -> tuple[list[TemporalSupport], list[str]]:
    valid: list[TemporalSupport] = []
    invalid_ids: list[str] = []
    for support in supports:
        provenance = (support.provenance or "").strip().lower()
        if provenance in _INVALID_PROVENANCE or provenance not in _VALID_PROVENANCE:
            invalid_ids.append(support.support_id)
            continue
        valid.append(support)
    return valid, invalid_ids


def _is_temporal_question(question: str) -> bool:
    text = (question or "").lower()
    return any(term in text for term in _TEMPORAL_QUESTION_TERMS) or (
        _deictic_when_question_parts(text) is not None
    )


def _select_deictic_source_date(
    question: str,
    supports: Sequence[TemporalSupport],
) -> TemporalConstructionDecision | None:
    parts = _deictic_when_question_parts(question)
    if parts is None:
        return None

    candidates: list[_DeicticCandidate] = []
    source_date_conflict_ids: list[str] = []
    for support in supports:
        if not _subject_matches_deictic_support(parts.subject, support.text):
            continue
        if not _required_phrases_match(parts.required_phrases, support.text):
            continue
        matched_terms = _matched_event_terms(support.text, parts.event_terms)
        min_matches = 2 if len(parts.event_terms) > 1 else 1
        if len(matched_terms) < min_matches:
            continue
        if len(parts.event_terms) > 1 and len(matched_terms) / len(parts.event_terms) < 0.6:
            continue
        support_terms = _material_terms(support.text)
        if parts.month_terms and not all(term in support_terms for term in parts.month_terms):
            continue
        source_date, provenance_used, conflict = _deictic_support_source_date(support)
        if conflict:
            source_date_conflict_ids.append(support.support_id)
            continue
        if source_date is None:
            continue
        deictic = _deictic_cue(support.text, source_date)
        if deictic is None:
            continue
        deictic_phrase, relative_relation, resolved_date, answer = deictic
        candidates.append(
            _DeicticCandidate(
                answer=answer,
                support_id=support.support_id,
                source_date=source_date,
                deictic_phrase=deictic_phrase,
                relative_relation=relative_relation,
                resolved_date=resolved_date,
                matched_event_terms=tuple(matched_terms),
                event_terms=parts.event_terms,
                provenance_used=provenance_used,
            )
        )

    if source_date_conflict_ids:
        return _fallback(REASON_SOURCE_DATE_CONFLICT, support_ids=tuple(source_date_conflict_ids))
    if not candidates:
        return None

    by_answer: dict[str, _DeicticCandidate] = {}
    for candidate in candidates:
        by_answer.setdefault(_normalize_temporal_surface(candidate.answer), candidate)
    candidate = candidates[0]
    return _accepted(
        candidate.answer,
        REASON_DEICTIC_SOURCE_DATE,
        support_ids=(candidate.support_id,),
        candidate_count=len(candidates),
        provenance_used=(candidate.provenance_used,),
        extra_diagnostics={
            "source_date": _format_date(candidate.source_date),
            "deictic_phrase": candidate.deictic_phrase,
            "relative_relation": candidate.relative_relation,
            "resolved_date": _format_date(candidate.resolved_date) if candidate.resolved_date else None,
            "event_terms": candidate.event_terms,
            "matched_event_terms": candidate.matched_event_terms,
        },
    )


def _deictic_when_question_parts(question: str) -> _DeicticQuestion | None:
    match = _DEICTIC_WHEN_QUESTION_RE.match((question or "").lower().strip())
    if not match:
        return None
    subject = re.sub(r"\s+", " ", match.group("subject")).strip()
    if not subject or " and " in subject:
        return None
    event_phrase = re.sub(r"\s+", " ", f"{match.group('verb')} {match.group('event')}").strip()
    event_terms = tuple(
        term for term in _material_terms(event_phrase) if term not in _DEICTIC_EVENT_STOPWORDS
    )
    if not event_terms:
        return None
    month_terms = tuple(term for term in event_terms if term in _MONTH_TERMS)
    required_phrases = ("support group",) if "support group" in event_phrase else ()
    return _DeicticQuestion(
        subject=subject,
        event_terms=event_terms,
        month_terms=month_terms,
        required_phrases=required_phrases,
    )


def _deictic_support_source_date(support: TemporalSupport) -> tuple[date | None, str, bool]:
    header_date = _conversation_header_date(support.text)
    if support.source_date is not None:
        if header_date is not None and abs((support.source_date - header_date).days) > 1:
            return None, REASON_SOURCE_DATE_CONFLICT, True
        return support.source_date, "source_date", False
    if header_date is not None:
        return header_date, "conversation_header", False
    return None, REASON_MISSING_PROVENANCE, False


def _conversation_header_date(text: str) -> date | None:
    match = _CONVERSATION_HEADER_DATE_RE.search(text or "")
    if not match:
        return None
    month = _MONTHS[match.group("month").lower().rstrip(".")]
    return _safe_date(int(match.group("year")), month, int(match.group("day")))


def _subject_matches_deictic_support(subject: str, text: str) -> bool:
    normalized_subject = (subject or "").lower().strip()
    normalized_text = (text or "").lower()
    if not normalized_subject or not normalized_text:
        return False
    if normalized_subject in normalized_text:
        return True
    family_match = re.fullmatch(r"([a-z][a-z' -]{0,40}?)'s\s+(family|kids)", normalized_subject)
    if family_match:
        possessor_terms = _material_terms(family_match.group(1))
        if not possessor_terms or not possessor_terms <= _material_terms(text):
            return False
        return bool(re.search(r"\b(?:children|daughter|family|kids|son)\b", normalized_text))
    subject_terms = _material_terms(normalized_subject)
    return bool(subject_terms) and subject_terms <= _material_terms(text)


def _required_phrases_match(required_phrases: tuple[str, ...], text: str) -> bool:
    normalized_text = re.sub(r"\s+", " ", (text or "").lower())
    return all(phrase in normalized_text for phrase in required_phrases)


def _matched_event_terms(text: str, event_terms: tuple[str, ...]) -> list[str]:
    support_terms = _material_terms(text)
    if {"bad", "upset", "bugged"}.intersection(support_terms):
        support_terms.add("negative")
    if "gang" in support_terms:
        support_terms.update({"friend", "friends"})
    return [term for term in event_terms if term in support_terms]


def _deictic_cue(text: str, source_date: date) -> tuple[str, str, date | None, str] | None:
    if _LAST_FRIDAY_RE.search(text or ""):
        return _weekday_deictic("last Friday", "Friday before source date", 4, source_date)
    if _LAST_TUESDAY_RE.search(text or ""):
        return _weekday_deictic("last Tuesday", "Tuesday before source date", 1, source_date)
    if _LAST_WEEKEND_RE.search(text or ""):
        return (
            "last weekend",
            "weekend before source date",
            None,
            f"The weekend before {_format_date(source_date)}",
        )
    if _THIS_WEEK_RE.search(text or ""):
        return (
            "this week",
            "week of source date",
            None,
            f"The week of {_format_date(source_date)}",
        )
    if _LAST_WEEK_RE.search(text or ""):
        return (
            "last week",
            "week before source date",
            None,
            f"The week before {_format_date(source_date)}",
        )
    return None


def _weekday_deictic(
    phrase: str,
    relation: str,
    weekday: int,
    source_date: date,
) -> tuple[str, str, date, str]:
    days_back = (source_date.weekday() - weekday) % 7
    if days_back == 0:
        days_back = 7
    resolved_date = date.fromordinal(source_date.toordinal() - days_back)
    weekday_name = "Friday" if weekday == 4 else "Tuesday"
    return (
        phrase,
        relation,
        resolved_date,
        f"The {weekday_name} before {_format_date(source_date)}",
    )


def _select_direct_duration(
    question: str, supports: Sequence[TemporalSupport]
) -> tuple[str, str] | None:
    question_terms = _material_terms(question)
    candidates: list[tuple[int, str, str]] = []
    for support in supports:
        for sentence in _sentences(support.text):
            if _UNRELATED_DURATION_RE.search(sentence):
                continue
            match = (
                _DIRECT_DURATION_RE.search(sentence)
                or _REVERSED_DIRECT_DURATION_RE.search(sentence)
                or _STATE_DIRECT_DURATION_RE.search(sentence)
                or _ALREADY_DIRECT_DURATION_RE.search(sentence)
            )
            if not match:
                continue
            overlap = len(question_terms & _material_terms(sentence))
            if overlap <= 0:
                continue
            duration = _normalize_duration(match.group("value"), match.group("unit"))
            candidates.append((overlap, duration, support.support_id))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    if len(candidates) > 1 and candidates[1][0] == candidates[0][0]:
        return None
    return candidates[0][1], candidates[0][2]


def _select_endpoint_range(
    question: str, supports: Sequence[TemporalSupport]
) -> TemporalConstructionDecision | None:
    question_terms = _material_terms(question)
    candidates: list[tuple[int, _Endpoint, _Endpoint]] = []
    ambiguous = False
    for support in supports:
        for sentence in _sentences(support.text):
            endpoints = _extract_endpoints(sentence, support)
            starts = [endpoint for endpoint in endpoints if endpoint.role == "start"]
            ends = [endpoint for endpoint in endpoints if endpoint.role == "end"]
            if len(starts) > 1 or len(ends) > 1:
                ambiguous = True
                continue
            if len(starts) != 1 or len(ends) != 1:
                continue
            start, end = starts[0], ends[0]
            if end.value <= start.value:
                continue
            overlap = len(question_terms & _material_terms(sentence))
            if overlap <= 0:
                continue
            candidates.append((overlap, start, end))
    if not candidates:
        if ambiguous:
            return _fallback(REASON_AMBIGUOUS_CANDIDATES)
        if _has_endpoint_without_event_overlap(question, supports):
            return _fallback(REASON_NO_EVENT_OVERLAP)
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    if len(candidates) > 1 and candidates[1][0] == candidates[0][0]:
        return _fallback(REASON_AMBIGUOUS_CANDIDATES)
    _overlap, start, end = candidates[0]
    elapsed_days = (end.value - start.value).days
    return _accepted(
        f"From {start.text} to {end.text}: {elapsed_days} days.",
        REASON_ENDPOINT_RANGE,
        support_ids=(start.support_id, end.support_id),
        candidate_count=len(candidates),
        provenance_used=(start.provenance_used, end.provenance_used),
    )


def _extract_endpoints(sentence: str, support: TemporalSupport) -> list[_Endpoint]:
    endpoints: list[_Endpoint] = []
    explicit_years = {int(match.group(1)) for match in _YEAR_RE.finditer(sentence or "")}
    single_context_year = next(iter(explicit_years)) if len(explicit_years) == 1 else None
    for match in _DATE_RE.finditer(sentence or ""):
        year_text = match.group("year")
        year: int | None
        provenance_used: str
        if year_text:
            year = int(year_text)
            provenance_used = "support_text_year"
        elif single_context_year is not None:
            year = single_context_year
            provenance_used = "support_text_year"
        elif support.source_date is not None:
            year = support.source_date.year
            provenance_used = "source_date"
        elif support.source_year is not None:
            year = support.source_year
            provenance_used = "source_year"
        else:
            year = None
            provenance_used = REASON_MISSING_PROVENANCE
        if year is None:
            continue
        month = _MONTHS[match.group("month").lower().rstrip(".")]
        day = int(match.group("day"))
        value = _safe_date(year, month, day)
        if value is None:
            continue
        role = _endpoint_role_near(sentence, match.start(), match.end())
        if role is None:
            continue
        endpoints.append(
            _Endpoint(
                role=role,
                text=match.group(0),
                value=value,
                support_id=support.support_id,
                provenance_used=provenance_used,
            )
        )
    return endpoints


def _has_missing_year_endpoint(supports: Sequence[TemporalSupport]) -> bool:
    for support in supports:
        if support.source_date is not None or support.source_year is not None:
            continue
        for match in _DATE_RE.finditer(support.text or ""):
            if not match.group("year"):
                return True
    return False


def _has_endpoint_without_event_overlap(question: str, supports: Sequence[TemporalSupport]) -> bool:
    question_terms = _material_terms(question)
    for support in supports:
        for sentence in _sentences(support.text):
            endpoints = _extract_endpoints(sentence, support)
            if any(endpoint.role == "start" for endpoint in endpoints) and any(
                endpoint.role == "end" for endpoint in endpoints
            ):
                if len(question_terms & _material_terms(sentence)) <= 0:
                    return True
    return False


def _endpoint_role_near(text: str, start_index: int, end_index: int) -> str | None:
    before = text[max(0, start_index - 48) : start_index]
    after = text[end_index : min(len(text), end_index + 48)]
    start_distance = _nearest_role_distance(before, after, _START_ROLE_RE)
    end_distance = _nearest_role_distance(before, after, _END_ROLE_RE)
    if start_distance is not None and (end_distance is None or start_distance < end_distance):
        return "start"
    if end_distance is not None and (start_distance is None or end_distance < start_distance):
        return "end"
    return None


def _nearest_role_distance(before: str, after: str, pattern: re.Pattern[str]) -> int | None:
    distances: list[int] = []
    for match in pattern.finditer(before):
        distances.append(len(before) - match.end())
    for match in pattern.finditer(after):
        distances.append(match.start())
    return min(distances) if distances else None


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text or "") if part.strip()]


def _material_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for token in re.findall(r"[a-z][a-z0-9]+", (text or "").lower()):
        if len(token) < 4 or token in _QUESTION_STOPWORDS:
            continue
        terms.add(token)
        if token.endswith("ies") and len(token) > 5:
            terms.add(f"{token[:-3]}y")
        if token.endswith("s") and len(token) > 4:
            terms.add(token[:-1])
        if token.endswith("ing") and len(token) > 6:
            stem = token[:-3]
            terms.add(stem)
            if stem.endswith("k"):
                terms.add(f"{stem}e")
        if token == "roadtrip":
            terms.update({"road", "trip"})
    return terms


def _normalize_duration(value: str, unit: str) -> str:
    amount = int(value)
    base_unit = (unit or "").lower().rstrip("s")
    if amount == 1:
        return f"{amount} {base_unit}"
    return f"{amount} {base_unit}s"


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _format_date(value: date) -> str:
    return f"{value.day} {value.strftime('%B')} {value.year}"


def _normalize_temporal_surface(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def _accepted(
    answer: str,
    reason: str,
    *,
    support_ids: tuple[str, ...],
    candidate_count: int,
    provenance_used: tuple[str, ...],
    extra_diagnostics: Mapping[str, object] | None = None,
) -> TemporalConstructionDecision:
    diagnostics = {
        "candidate_count": candidate_count,
        "fallback_reason": None,
        "strategy": reason,
        "support_ids": support_ids,
        "provenance_used": provenance_used,
        "constructor_llm_calls": 0,
    }
    if extra_diagnostics:
        diagnostics.update(dict(extra_diagnostics))
    return TemporalConstructionDecision(
        status=STATUS_ACCEPTED,
        answer=answer,
        reason=reason,
        support_ids=support_ids,
        strategy=reason,
        diagnostics=diagnostics,
    )


def _fallback(
    reason: str,
    *,
    support_ids: tuple[str, ...] = (),
    candidate_count: int = 0,
) -> TemporalConstructionDecision:
    return TemporalConstructionDecision(
        status=STATUS_FALLBACK,
        answer=None,
        reason=reason,
        support_ids=support_ids,
        strategy=None,
        diagnostics={
            "candidate_count": candidate_count,
            "fallback_reason": reason,
            "strategy": None,
            "support_ids": support_ids,
            "provenance_used": (),
            "constructor_llm_calls": 0,
        },
    )
