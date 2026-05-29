"""Unsupported-attribute refusal planning.

Pure, deterministic helper for detecting when prompt-visible memories do not
support the qualitative attribute requested by a question. This module performs
no retrieval, storage access, model calls, or runtime wiring by itself.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

SupportStatus = Literal["supported", "unsupported", "uncertain"]

_ATTRIBUTE_FAMILIES: dict[str, tuple[str, ...]] = {
    "atmosphere": ("atmosphere", "mood", "tone", "vibe", "ambiance", "ambience", "energy"),
    "feeling": ("feel", "feeling", "felt", "experience", "impression"),
    "reaction": ("reaction", "react", "received", "response"),
}

_ADJACENT_EVENT_RE = re.compile(
    r"\b("
    r"happened|held|hosted|attended|attendees|attendance|joined|participated|"
    r"successful|success|went well|completed|finished|scheduled|planned|discussed|"
    r"decided|agreed|approved|launched|met|meeting|session|event"
    r")\b",
    re.IGNORECASE,
)

_DIRECT_SUPPORT_RE = re.compile(
    r"\b("
    r"atmosphere|mood|tone|vibe|ambiance|ambience|energy|felt|feeling|"
    r"experience|impression|reaction|reacted|received|response|enthusiastic|"
    r"warm|tense|calm|excited|positive|negative|awkward|comfortable|lively|"
    r"quiet|energetic|collaborative|supportive|hostile|friendly|welcoming"
    r")\b",
    re.IGNORECASE,
)

_EXCLUDED_QUESTION_TYPES = frozenset(
    {
        "event_ordering",
        "information_extraction",
        "multi_session_reasoning",
        "count",
        "counting",
        "structured_count",
    }
)

_GENERIC_SUMMARY_RE = re.compile(r"\b(summarize|summary|recap|overview|what happened|plans? change)\b", re.I)
_EXTRACTION_RE = re.compile(r"\b(which|what|who|when|where|how many|list|name|mention only|only and only)\b", re.I)

_FORBIDDEN_KEY_PARTS = (
    ("expected", "answer"),
    ("expected", "answers"),
    ("expected", "source", "ref"),
    ("expected", "source", "refs"),
    ("expected", "sources"),
    ("gold",),
    ("gold", "answer"),
    ("ground", "truth"),
    ("ground", "truth", "answer"),
    ("judge", "rubric"),
    ("judge", "rubrics"),
    ("nugget",),
    ("nugget", "scores"),
    ("rubric",),
    ("score",),
    ("source", "chat", "ids"),
)
_FORBIDDEN_KEYS = frozenset("_".join(parts) for parts in _FORBIDDEN_KEY_PARTS)


@dataclass(frozen=True)
class UnsupportedAttributeRefusalPlan:
    attempted: bool
    would_inject: bool
    requested_attribute_family: str | None
    support_status: SupportStatus
    reason: str
    instruction: str | None
    runtime_effect: bool = False
    score_claim_allowed: bool = False
    extra_llm_calls: int = 0
    token_delta_estimate: int = 0

    def as_debug_dict(self) -> dict[str, Any]:
        return asdict(self)


def assert_no_forbidden_refusal_material(value: object, *, path: str = "payload") -> None:
    bad_paths = _forbidden_paths(value, path=path)
    if bad_paths:
        joined = ", ".join(bad_paths[:8])
        raise ValueError(f"forbidden unsupported-attribute material present: {joined}")


def build_unsupported_attribute_refusal_plan(
    question: str,
    retrieved_context: str,
    *,
    question_type: str | None = None,
) -> UnsupportedAttributeRefusalPlan:
    assert_no_forbidden_refusal_material({"question": question, "retrieved_context": retrieved_context})
    question_text = _clean(question)
    context_text = _clean(retrieved_context)
    family = _requested_attribute_family(question_text)

    if not family:
        return _plan(False, False, None, "uncertain", "no_attribute_query", None)
    if _excluded_by_type_or_shape(question_text, question_type):
        return _plan(True, False, family, "uncertain", "excluded_question_type", None)
    if not context_text:
        return _plan(True, False, family, "uncertain", "uncertain_support", None)
    if _has_direct_support(context_text, family):
        return _plan(True, False, family, "supported", "supported_attribute_evidence_present", None)
    if _has_adjacent_event_facts(context_text):
        instruction = _instruction_for(family)
        return _plan(
            True,
            True,
            family,
            "unsupported",
            "unsupported_attribute",
            instruction,
            token_delta_estimate=_estimate_tokens(instruction),
        )
    return _plan(True, False, family, "uncertain", "uncertain_support", None)


def _plan(
    attempted: bool,
    would_inject: bool,
    family: str | None,
    support_status: SupportStatus,
    reason: str,
    instruction: str | None,
    *,
    token_delta_estimate: int = 0,
) -> UnsupportedAttributeRefusalPlan:
    return UnsupportedAttributeRefusalPlan(
        attempted=attempted,
        would_inject=would_inject,
        requested_attribute_family=family,
        support_status=support_status,
        reason=reason,
        instruction=instruction,
        token_delta_estimate=token_delta_estimate,
    )


def _requested_attribute_family(question: str) -> str | None:
    lower = question.lower()
    asks_quality = bool(re.search(r"\b(what|how)\b", lower)) and bool(
        re.search(r"\b(like|feel|felt|was|were|seem|seemed|received|react)\b", lower)
    )
    if not asks_quality:
        return None
    for family, terms in _ATTRIBUTE_FAMILIES.items():
        if any(re.search(rf"\b{re.escape(term)}\b", lower) for term in terms):
            return family
    return None


def _excluded_by_type_or_shape(question: str, question_type: str | None) -> bool:
    kind = (question_type or "").strip().lower()
    if kind in _EXCLUDED_QUESTION_TYPES:
        return True
    if kind == "summarization" and _GENERIC_SUMMARY_RE.search(question):
        return True
    return bool(_EXTRACTION_RE.search(question) and not _requested_attribute_family(question))


def _has_direct_support(context: str, family: str) -> bool:
    lower = context.lower()
    family_terms = _ATTRIBUTE_FAMILIES.get(family, ())
    if any(re.search(rf"\b{re.escape(term)}\b", lower) for term in family_terms):
        return True
    return bool(_DIRECT_SUPPORT_RE.search(context))


def _has_adjacent_event_facts(context: str) -> bool:
    return bool(_ADJACENT_EVENT_RE.search(context))


def _instruction_for(family: str) -> str:
    label = family.replace("_", " ")
    return (
        "UNSUPPORTED ATTRIBUTE BOUNDARY:\n"
        f"- The question asks for a {label} attribute.\n"
        "- Do not infer that attribute from event existence, attendance, success, scheduling, or adjacent outcomes.\n"
        "- Answer that the retrieved memories do not provide information about the requested attribute unless the memories directly support it."
    )


def _estimate_tokens(text: str | None) -> int:
    if not text:
        return 0
    return max(1, round(len(text.split()) * 1.25))


def _clean(value: str | None) -> str:
    return " ".join(str(value or "").split())


def _forbidden_paths(value: object, *, path: str) -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            if key_text.lower() in _FORBIDDEN_KEYS:
                paths.append(child_path)
            paths.extend(_forbidden_paths(item, path=child_path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            paths.extend(_forbidden_paths(item, path=f"{path}[{index}]"))
    return paths
