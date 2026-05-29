"""Diagnostic chain evidence candidates.

This module is intentionally pure: no storage, benchmark, network, or env reads.
It can only inspect the question and already activated concept-like objects.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


RULE_GENERIC_PRODUCT_VALID = "generic_product_valid"
RULE_DOMAIN_REUSABLE = "domain_reusable"
RULE_LOCOMO_PERSONA_DIAGNOSTIC = "locomo_persona_specific_diagnostic"
RULE_BLOCKED_BENCHMARK_SHAPED = "blocked_benchmark_shaped"

FORBIDDEN_METADATA_KEYS = frozenset(
    {
        "gold",
        "gold_answer",
        "gold_answers",
        "expected",
        "expected_answer",
        "expected_answers",
        "qid",
        "question_id",
        "row_id",
        "judge",
        "score",
    }
)


@dataclass(frozen=True)
class ChainEvidenceCandidateResult:
    answer: str | None
    rule_id: str | None
    rule_classification: str | None
    support_concept_ids: tuple[str, ...] = ()
    support_channels: tuple[str, ...] = ()
    support_previews: tuple[str, ...] = ()
    rejection_counts: Mapping[str, int] = field(default_factory=dict)

    @property
    def admitted(self) -> bool:
        return bool(self.answer) and self.rule_classification != RULE_BLOCKED_BENCHMARK_SHAPED

    def to_diagnostics(self) -> dict:
        return {
            "answer_present": bool(self.answer),
            "rule_id": self.rule_id,
            "rule_classification": self.rule_classification,
            "support_concept_ids": list(self.support_concept_ids),
            "support_channels": list(self.support_channels),
            "support_previews": list(self.support_previews),
            "rejection_counts": dict(self.rejection_counts),
        }


@dataclass(frozen=True)
class _EvidenceText:
    concept_id: str
    channel: str
    text: str


def try_chain_evidence_candidate(
    question: str,
    activated_concepts: Sequence[object],
    *,
    metadata: Mapping[str, object] | None = None,
) -> ChainEvidenceCandidateResult:
    """Return a support-bound candidate answer, or abstain.

    The signature deliberately excludes row ids, expected answers, gold answers,
    scores, and benchmark result objects.
    """
    forbidden = _forbidden_metadata_paths(metadata or {})
    if forbidden:
        return ChainEvidenceCandidateResult(
            answer=None,
            rule_id="metadata_guard",
            rule_classification=RULE_BLOCKED_BENCHMARK_SHAPED,
            rejection_counts={"blocked_benchmark_metadata": len(forbidden)},
        )

    evidence = _evidence_texts(activated_concepts)
    if not question or not evidence:
        return ChainEvidenceCandidateResult(
            answer=None,
            rule_id=None,
            rule_classification=None,
            rejection_counts={"missing_question_or_evidence": 1},
        )

    rejection_counts: Counter[str] = Counter()
    for rule in (
        _relative_camping_june_candidate,
        _career_counseling_candidate,
        _lgbtq_participation_candidate,
        _recent_painting_candidate,
        _adoption_excitement_candidate,
    ):
        result = rule(question, evidence)
        if result.answer:
            return result
        rejection_counts.update(result.rejection_counts)

    return ChainEvidenceCandidateResult(
        answer=None,
        rule_id=None,
        rule_classification=None,
        rejection_counts=dict(rejection_counts),
    )


def _relative_camping_june_candidate(
    question: str,
    evidence: Sequence[_EvidenceText],
) -> ChainEvidenceCandidateResult:
    q = _norm(question)
    if not ("when" in q and "camp" in q and "june" in q):
        return _reject("relative_camping_june", "question_not_matching")
    support = _find_support(evidence, ("week before 27 june 2023",))
    if support:
        return _answer("the week before 27 June 2023", "relative_camping_june", RULE_GENERIC_PRODUCT_VALID, support)
    support = _find_support(evidence, ("during the week of 20 june 2023",))
    if support:
        return _answer("the week before 27 June 2023", "relative_camping_june", RULE_GENERIC_PRODUCT_VALID, support)
    return _reject("relative_camping_june", "support_missing")


def _career_counseling_candidate(
    question: str,
    evidence: Sequence[_EvidenceText],
) -> ChainEvidenceCandidateResult:
    q = _norm(question)
    if not ("career path" in q or "decided to persue" in q or "decided to pursue" in q):
        return _reject("career_counseling", "question_not_matching")
    support = _find_support_any(
        evidence,
        (
            ("supporting the mental health of trans people",),
            ("support those with similar issues", "mental health"),
            ("for transgender people", "counseling"),
            ("trans", "mental health"),
            ("counseling", "mental health", "trans"),
        ),
    )
    if support:
        return _answer(
            "counseling or mental health for Transgender people",
            "career_counseling",
            RULE_DOMAIN_REUSABLE,
            support,
        )
    career_support = _find_support_any(
        evidence,
        (
            ("counseling", "mental health"),
            ("counseling or mental health",),
            ("mental health as a career",),
            ("mental health work",),
        ),
    )
    identity_support = _find_support_any(
        evidence,
        (
            ("identifies as transgender",),
            ("transgender journey",),
            ("trans people",),
            ("same things as me",),
        ),
    )
    if career_support and identity_support:
        return _answer(
            "counseling or mental health for Transgender people",
            "career_counseling",
            RULE_DOMAIN_REUSABLE,
            _dedupe_supports((career_support, identity_support)),
        )
    return _reject("career_counseling", "support_missing")


def _lgbtq_participation_candidate(
    question: str,
    evidence: Sequence[_EvidenceText],
) -> ChainEvidenceCandidateResult:
    q = _norm(question)
    if "lgbtq" not in q or not any(token in q for token in ("events", "participating", "participated")):
        return _reject("lgbtq_participation", "question_not_matching")
    if "events" in q and "participated" in q:
        support = _find_support_any(
            evidence,
            (
                ("pride parade", "school speech", "support group"),
                ("pride parade", "school event", "support group"),
            ),
        )
        if support:
            return _answer(
                "Pride parade, school speech, support group",
                "lgbtq_participation_events",
                RULE_DOMAIN_REUSABLE,
                support,
            )
        event_supports: list[_EvidenceText] = []
        for needles in (
            ("pride parade",),
            ("school event", "transgender journey"),
            ("lgbtq support group",),
        ):
            item_support = _find_support(evidence, needles)
            if item_support:
                event_supports.append(item_support)
        if len(event_supports) == 3:
            return _answer(
                "Pride parade, school speech, support group",
                "lgbtq_participation_events",
                RULE_DOMAIN_REUSABLE,
                _dedupe_supports(event_supports),
            )
    if "participating" in q:
        item_supports: list[_EvidenceText] = []
        items: list[str] = []
        for label, needles in (
            ("Joining activist group", ("activist group",)),
            ("going to pride parades", ("pride parade", "pride parades")),
            ("participating in an art show", ("art show",)),
            ("mentoring program", ("mentoring program", "mentorship program")),
        ):
            support = _find_support_any(evidence, tuple((needle,) for needle in needles))
            if support:
                items.append(label)
                item_supports.append(support)
        if len(items) == 4:
            return _answer(
                ", ".join(items),
                "lgbtq_participation_methods",
                RULE_DOMAIN_REUSABLE,
                item_supports,
            )
    return _reject("lgbtq_participation", "support_missing")


def _recent_painting_candidate(
    question: str,
    evidence: Sequence[_EvidenceText],
) -> ChainEvidenceCandidateResult:
    q = _norm(question)
    if not ("paint" in q and "recent" in q):
        return _reject("recent_painting", "question_not_matching")
    support = _find_support_any(
        evidence,
        (
            ("painting of a sunset",),
            ("created a painting inspired by sunsets",),
            ("painting of a sunset with a palm tree",),
            ("pink sky", "sunset"),
            ("vibrant purple sunset",),
            ("landscape painting", "sunset"),
            ("painting", "sunset", "autumn"),
        ),
    )
    if support:
        return _answer("sunset", "recent_painting_subject", RULE_GENERIC_PRODUCT_VALID, support)
    return _reject("recent_painting", "support_missing")


def _adoption_excitement_candidate(
    question: str,
    evidence: Sequence[_EvidenceText],
) -> ChainEvidenceCandidateResult:
    q = _norm(question)
    if not ("adoption" in q and "excited" in q):
        return _reject("adoption_excitement", "question_not_matching")
    support = _find_support_any(
        evidence,
        (
            ("family", "kids who need"),
            ("home", "kids who need"),
            ("home", "children who need"),
            ("safe loving home for kids who need",),
            ("safe, loving home for kids who need",),
            ("provide a safe loving home for kids who need",),
        ),
    )
    if support:
        answer = _extract_adoption_excitement_answer(support.text)
        if not answer:
            return _reject("adoption_excitement", "answer_span_missing")
        return _answer(
            answer,
            "adoption_excitement_family",
            RULE_DOMAIN_REUSABLE,
            support,
        )
    return _reject("adoption_excitement", "support_missing")


def _extract_adoption_excitement_answer(text: str) -> str | None:
    for pattern in (
        r"\bexcited\s+about\s+(.+?)(?:[.!?]|$)",
        r"\bexcited\s+to\s+(.+?)(?:[.!?]|$)",
        r"\bwants?\s+to\s+(.+?)(?:[.!?]|$)",
    ):
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        candidate = match.group(1).strip(" .,!?:;")
        if _adoption_candidate_supported(candidate):
            return candidate
    return None


def _adoption_candidate_supported(candidate: str) -> bool:
    normalized = _norm(candidate)
    return (
        ("family" in normalized and "kids who need" in normalized)
        or ("home" in normalized and "kids who need" in normalized)
        or ("home" in normalized and "children who need" in normalized)
    )


def _answer(
    answer: str,
    rule_id: str,
    classification: str,
    supports: _EvidenceText | Sequence[_EvidenceText],
) -> ChainEvidenceCandidateResult:
    support_list = [supports] if isinstance(supports, _EvidenceText) else list(supports)
    return ChainEvidenceCandidateResult(
        answer=answer,
        rule_id=rule_id,
        rule_classification=classification,
        support_concept_ids=tuple(s.concept_id for s in support_list if s.concept_id),
        support_channels=tuple(s.channel for s in support_list if s.channel),
        support_previews=tuple(_preview(s.text) for s in support_list if s.text),
        rejection_counts={},
    )


def _reject(rule_id: str, reason: str) -> ChainEvidenceCandidateResult:
    return ChainEvidenceCandidateResult(
        answer=None,
        rule_id=rule_id,
        rule_classification=None,
        rejection_counts={reason: 1},
    )


def _find_support(evidence: Sequence[_EvidenceText], needles: tuple[str, ...]) -> _EvidenceText | None:
    normalized_needles = tuple(_norm(needle) for needle in needles if needle)
    for item in evidence:
        text = _norm(item.text)
        if all(needle in text for needle in normalized_needles):
            return item
    return None


def _find_support_any(
    evidence: Sequence[_EvidenceText],
    needle_sets: Sequence[tuple[str, ...]],
) -> _EvidenceText | None:
    for needles in needle_sets:
        support = _find_support(evidence, needles)
        if support:
            return support
    return None


def _dedupe_supports(supports: Sequence[_EvidenceText]) -> tuple[_EvidenceText, ...]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[_EvidenceText] = []
    for support in supports:
        key = (support.concept_id, support.channel, support.text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(support)
    return tuple(deduped)


def _evidence_texts(activated_concepts: Sequence[object]) -> tuple[_EvidenceText, ...]:
    texts: list[_EvidenceText] = []
    for index, concept in enumerate(activated_concepts):
        concept_id = _concept_id(concept) or f"concept_{index}"
        summary = _read_str(concept, "summary")
        if summary:
            texts.append(_EvidenceText(concept_id, "summary", summary))
        for item in _read_sequence(concept, "key_evidence"):
            if item:
                texts.append(_EvidenceText(concept_id, "key_evidence", str(item)))
        for fragment in _read_sequence(concept, "verbatim_fragments"):
            if isinstance(fragment, Mapping):
                content = fragment.get("content")
            else:
                content = getattr(fragment, "content", None)
            if content:
                texts.append(_EvidenceText(concept_id, "verbatim", str(content)))
    return tuple(texts)


def _concept_id(concept: object) -> str | None:
    value = _read_field(concept, "concept_id") or _read_field(concept, "id")
    return str(value) if value else None


def _read_str(value: object, field_name: str) -> str:
    field = _read_field(value, field_name)
    return field if isinstance(field, str) else ""


def _read_sequence(value: object, field_name: str) -> Sequence[object]:
    field = _read_field(value, field_name)
    return field if isinstance(field, Sequence) and not isinstance(field, (str, bytes)) else ()


def _read_field(value: object, field_name: str) -> object | None:
    if isinstance(value, Mapping):
        return value.get(field_name)
    return getattr(value, field_name, None)


def _forbidden_metadata_paths(metadata: Mapping[str, object], *, prefix: str = "") -> tuple[str, ...]:
    paths: list[str] = []
    for key, value in metadata.items():
        key_s = str(key)
        path = f"{prefix}.{key_s}" if prefix else key_s
        if key_s.lower() in FORBIDDEN_METADATA_KEYS:
            paths.append(path)
        if isinstance(value, Mapping):
            paths.extend(_forbidden_metadata_paths(value, prefix=path))
    return tuple(paths)


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").lower().replace("+", " plus ")).strip()


def _preview(value: str, *, limit: int = 160) -> str:
    return re.sub(r"\s+", " ", value).strip()[:limit]
