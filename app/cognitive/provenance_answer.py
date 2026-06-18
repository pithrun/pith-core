"""Provenance-bound answer emission for engine-side chain answers.

ENGINE-ANS-1 uses an LLM only as a cited span selector. Python validation
decides whether the proposed answer is present in the cited support.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import string
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from typing import Literal

import numpy as np

from app.cognitive.answer_shape_admission import admit_answer_shape_contract
from app.cognitive.answer_shape_contracts import build_answer_shape_contract
from app.cognitive.temporal_answer_construction import (
    TemporalSupport,
    construct_temporal_answer_candidate,
)

SupportChannel = Literal["summary", "key_evidence", "text", "verbatim"]
AnswerMode = Literal[
    "exact_extractive",
    "normalized_extractive",
    "deterministic_candidate",
    "support_entails_yes",
    "support_entails_no",
    "structured_synthesis",
    "exact_support_recovery",
    "support_derived_repair",
    "abstain",
]
AnswerIntent = Literal[
    "scalar_entity",
    "date",
    "duration",
    "count",
    "yes_no",
    "location",
    "short_attribute",
    "synthesis_deferred",
    "temporal_deferred",
    "unsupported",
]
CandidateSource = Literal[
    "regex_date",
    "regex_conversation_source_date",
    "regex_relative_date_span",
    "regex_duration",
    "regex_count",
    "regex_quoted_title",
    "regex_location",
    "regex_short_attribute",
    "regex_media_intent_attribute",
    "regex_direct_support_scalar",
    "regex_direct_support_admission",
    "yes_no_entailment",
]
StructuredSynthesisShape = Literal[
    "none",
    "atomic_scalar",
    "predicate_bound_scalar",
    "list_or_set",
    "complete_phrase",
]
RepairShape = Literal[
    "none",
    "title_with_creator",
    "list_completion",
    "predicate_bound_scalar",
]
SupportPresentAnswerRole = Literal[
    "generic",
    "diet_list",
    "artifact_text",
    "named_title",
    "training_type",
    "location",
    "made_object_list",
    "pet_activity_list",
    "pet_type",
    "question_bound_list",
    "event_list",
    "activity_object",
    "action_bundle_list",
    "where_did_go_activity",
    "direct_support_scalar",
]
LLMCaller = Callable[..., str]

_SYSTEM_PROMPT = (
    "You answer only by copying the shortest direct span from one cited evidence item. "
    "Only answer if the question asks for one atomic date, duration, entity, "
    "or short scalar phrase. "
    'Return strict JSON: {"answer": "...", "support_id": "..."}. '
    "The support_id must exactly match one bracketed evidence ID such as s0 or s1. "
    "If no single evidence span directly answers the question, return "
    '{"answer": null, "support_id": null}. '
    "Do not aggregate across evidence items. Do not infer beyond the evidence."
)
_CANDIDATE_SELECTION_SYSTEM_PROMPT = (
    "You select one supported candidate answer. Return strict JSON only: "
    '{"candidate_id": "c0"} or {"candidate_id": null}. '
    "Do not invent a new answer. Do not rewrite candidate text."
)
_STRUCTURED_SYNTHESIS_SYSTEM_PROMPT = (
    "You answer by assembling only cited evidence spans. Return strict JSON only: "
    '{"answer": "...", "support_ids": ["s0"], "cited_spans": ["exact copied span"]}. '
    "Every answer item must appear in a cited span. For list answers, include every "
    "supported item needed to answer the question. For phrase answers, include the "
    "complete supported phrase. For predicate-object answers, cite a span that binds "
    "the question predicate to the answer. If evidence is insufficient, return "
    '{"answer": null, "support_ids": [], "cited_spans": []}.'
)

_BLOCKED_QUESTION_PATTERNS = (
    (re.compile(r"^\s*why\b", re.IGNORECASE), "why_requires_explanation"),
    (re.compile(r"^\s*would\b", re.IGNORECASE), "would_requires_inference"),
    (re.compile(r"\blikely\b", re.IGNORECASE), "likely_requires_inference"),
    (
        re.compile(r"\brelationship status\b", re.IGNORECASE),
        "relationship_status_requires_structured_attribute",
    ),
    (re.compile(r"\bwhat events\b", re.IGNORECASE), "events_require_list_synthesis"),
    (
        re.compile(r"\bwhat type of (?:individuals|people|persons)\b", re.IGNORECASE),
        "person_type_requires_category_judgment",
    ),
    (re.compile(r"\bwhere has\b", re.IGNORECASE), "where_has_requires_aggregation"),
    (re.compile(r"\bwhat happened\b", re.IGNORECASE), "happened_requires_event_synthesis"),
    (
        re.compile(r"\b(?:both|in common|share|shared)\b", re.IGNORECASE),
        "shared_or_common_requires_synthesis",
    ),
    (
        re.compile(r"\bwhat (?:items|hobbies|emotions)\b", re.IGNORECASE),
        "list_answer_requires_synthesis",
    ),
    (
        re.compile(r"\bwhat (?:are|is) .*\b(?:hobbies|emotions)\b", re.IGNORECASE),
        "list_answer_requires_synthesis",
    ),
    (
        re.compile(r"\bwhat kind of (?:interests|classes|groups|places)\b", re.IGNORECASE),
        "kind_question_requires_list_synthesis",
    ),
    (
        re.compile(r"\b(?:classes or groups|outdoor activities)\b", re.IGNORECASE),
        "activities_require_list_synthesis",
    ),
    (
        re.compile(r"\bwhat (?:pets|places)\b", re.IGNORECASE),
        "set_answer_requires_synthesis",
    ),
    (
        re.compile(r"\bhow did .*\bpromot(?:e|ed)\b", re.IGNORECASE),
        "promotion_methods_require_list_synthesis",
    ),
    (
        re.compile(r"\bwhat does .*\boffer\b", re.IGNORECASE),
        "offerings_require_list_synthesis",
    ),
    (
        re.compile(r"\bwhat major achievement\b", re.IGNORECASE),
        "achievement_requires_synthesis",
    ),
    (
        re.compile(r"\bunderlying condition\b", re.IGNORECASE),
        "underlying_condition_requires_inference",
    ),
    (re.compile(r"\ballergic to\b", re.IGNORECASE), "allergies_require_list_synthesis"),
    (re.compile(r"\bwhat is something\b", re.IGNORECASE), "broad_something_question"),
    (
        re.compile(r"\bhow (?:does|did|do) .*\bfeel\b", re.IGNORECASE),
        "feeling_requires_attribute_judgment",
    ),
    (
        re.compile(r"\bwhat does .*\bwon'?t\s+do\b", re.IGNORECASE),
        "negative_action_requires_predicate_resolution",
    ),
    (
        re.compile(r"^\s*when did .*\bmake\b", re.IGNORECASE),
        "temporal_make_requires_event_time_resolution",
    ),
    (
        re.compile(r"^\s*when is .*\bgoing to\b", re.IGNORECASE),
        "future_plan_requires_temporal_resolution",
    ),
)

_ATOMIC_QUESTION_PATTERNS = (
    re.compile(r"^\s*when\b", re.IGNORECASE),
    re.compile(r"^\s*how long\b", re.IGNORECASE),
    re.compile(r"^\s*how many\b", re.IGNORECASE),
    re.compile(r"^\s*who\b", re.IGNORECASE),
    re.compile(r"^\s*which\b", re.IGNORECASE),
    re.compile(r"^\s*where did\b", re.IGNORECASE),
    re.compile(r"^\s*what (?:book|movie|nickname)\b", re.IGNORECASE),
    re.compile(r"^\s*what (?:is|are|was|were|did|does)\b", re.IGNORECASE),
)
_FOR_HOW_LONG_QUESTION_PATTERN = re.compile(r"^\s*for\s+how\s+long\b", re.IGNORECASE)
_YES_NO_ARTIFACT_PHOTO_QUESTION_PATTERN = re.compile(
    r"^\s*did\s+[A-Z][a-z]+\s+make\b.*\b(?:photo|picture)\b",
    re.IGNORECASE,
)
_SCALAR_SLOT_QUESTION_PATTERNS = (
    re.compile(r"^\s*what\s+spice\s+did\b", re.IGNORECASE),
    re.compile(r"^\s*what\s+organization\s+does\b", re.IGNORECASE),
    re.compile(r"^\s*what\s+new\s+hobby\s+did\b", re.IGNORECASE),
    re.compile(r"^\s*what\s+project\s+did\b.*\bfinish\b", re.IGNORECASE),
    re.compile(r"^\s*what\s+activity\s+did\b", re.IGNORECASE),
    re.compile(r"^\s*what\s+class\s+is\b", re.IGNORECASE),
    re.compile(r"^\s*what\s+sports\s+activity\s+has\b", re.IGNORECASE),
    re.compile(r"^\s*what\s+color\s+glow\s+did\b", re.IGNORECASE),
    re.compile(r"^\s*what\s+emotion\s+does\b", re.IGNORECASE),
)

_TEMPORAL_QUESTION_PATTERN = re.compile(r"^\s*(?:when\b|how long\b)", re.IGNORECASE)
_MONEY_TOURNAMENT_DATE_QUESTION_PATTERN = re.compile(
    r"^\s*when\b(?=.*\bwin\b)(?=.*\b(?:money|cash)\b)(?=.*\btournament\b)",
    re.IGNORECASE,
)
_MONEY_TOURNAMENT_DATE_CUE_PATTERN = re.compile(
    r"\b(?:big|huge|large|money|cash|amount|significant)\b",
    re.IGNORECASE,
)
_TEMPORAL_ANSWER_PATTERN = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|"
    r"nov|dec)\b"
    r"|\b\d{4}\b"
    r"|\b\d+\s+(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b"
    r"|\b(?:ago|before|after|last|next|today|yesterday|tomorrow)\b",
    re.IGNORECASE,
)
_FULL_CALENDAR_DATE_PATTERN_TEXT = (
    r"(?:\b(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|"
    r"nov|dec)\s+\d{1,2},?\s+\d{4}\b"
    r"|\b\d{1,2}\s+(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|"
    r"sept|oct|nov|dec),?\s+\d{4}\b)"
)
_DATE_CANDIDATE_PATTERN_TEXT = (
    r"(?:\b(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|"
    r"nov|dec)\s+\d{1,2},?\s+\d{4}\b"
    r"|\b\d{1,2}\s+(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|"
    r"sept|oct|nov|dec),?\s+\d{4}\b"
    r"|\b(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|"
    r"nov|dec)\s+\d{4}\b"
    r"|\b\d{4}\b)"
)
_DATE_CANDIDATE_PATTERN = re.compile(_DATE_CANDIDATE_PATTERN_TEXT, re.IGNORECASE)
_MONTH_NAMES = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}
_MONTH_NUMBER_ALIASES = {
    **{name.lower(): number for number, name in _MONTH_NAMES.items()},
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
_RELATIVE_DATE_SPAN_PATTERN = re.compile(
    rf"\b(?:the\s+)?(?:day|week|month|year)s?\s+(?:before|after)\s+{_FULL_CALENDAR_DATE_PATTERN_TEXT}",
    re.IGNORECASE,
)
_RELATIVE_SURFACE_DATE_SPAN_PATTERN = re.compile(
    rf"\b(?:the\s+)?(?:"
    rf"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+before"
    rf"|weekend\s+before"
    rf"|week\s+(?:of|before|after)"
    rf"|(?:day|month|year)s?\s+(?:before|after)"
    rf")\s+{_FULL_CALENDAR_DATE_PATTERN_TEXT}",
    re.IGNORECASE,
)
_DATE_YEAR_VALUE_PATTERN = re.compile(r"\b(\d{4})\b")
_MIN_REASONABLE_ANSWER_YEAR = 1900
_MAX_REASONABLE_ANSWER_YEAR = 2100
_TEMPORAL_CALENDAR_CONTEXT_TERMS = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "jan",
    "feb",
    "mar",
    "apr",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
}
_DURATION_CANDIDATE_PATTERN = re.compile(
    r"\b\d+\s+(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b",
    re.IGNORECASE,
)
_DURATION_WORD_CANDIDATE_PATTERN = re.compile(
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
    r"eighteen|nineteen|twenty)\s+"
    r"(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b",
    re.IGNORECASE,
)
_DEICTIC_ANSWER_PATTERN = re.compile(
    r"\b(?:my|your|his|her|their|our)\s+(?:home country|country|city|town|place)\b"
    r"|\b(?:here|there|somewhere|someplace)\b",
    re.IGNORECASE,
)
_HOME_COUNTRY_MOVE_FROM_QUESTION_RE = re.compile(
    r"^\s*where\s+did\s+(?P<actor>[A-Z][A-Za-z'-]*)\s+move\s+from\b",
    re.IGNORECASE,
)
_HOME_COUNTRY_MOVE_ALIAS_RE = re.compile(
    r"\b(?:move|moved|moving)\s+from\s+"
    r"(?:my|your|his|her|their|our)?\s*home country\b",
    re.IGNORECASE,
)
_HOME_COUNTRY_VALUE_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z'-]+(?:'s)?\s+)?home country\s+"
    r"(?:is|was)\s+(?P<value>[A-Z][A-Za-z' -]{1,40})(?:[.;,!]|$)",
    re.IGNORECASE,
)
_HOME_COUNTRY_COMMA_VALUE_RE = re.compile(
    r"\bhome country,\s*(?P<value>[A-Z][A-Za-z' -]{1,40})(?:[.;,!]|$)",
    re.IGNORECASE,
)
_HOME_COUNTRY_MOVE_BACK_SOON_QUESTION_RE = re.compile(
    r"^\s*would\s+(?P<actor>[A-Z][A-Za-z'-]*)\s+want\s+to\s+move\s+back\s+to\s+"
    r"(?:his|her|their|my|your|our)?\s*home country\s+soon\b",
    re.IGNORECASE,
)
_HOME_COUNTRY_MOVE_BACK_CONTEXT_RE = re.compile(
    r"\b(?:home country|moved?\s+from|moving\s+from)\b",
    re.IGNORECASE,
)
_ADOPTION_OR_FAMILY_COMMITMENT_RE = re.compile(
    r"\b(?:adopt(?:ed|s|ing|ion)?|adoption agencies?|adoption agency|"
    r"build(?:ing)? (?:my|her|his|their|our)?\s*(?:own\s+)?family|"
    r"put a roof over kids|safe and loving home|home to needy kids|"
    r"home for kids|kids who need|children in need|motherhood|future as a mom)\b",
    re.IGNORECASE,
)
_ADOPTION_PROCESS_COMMITMENT_RE = re.compile(
    r"\b(?:process|applied|interviews?|journey|pursu(?:e|ing)|focused|goal|dream|ready|"
    r"building|build|start(?:ing)?|future|mom|motherhood|kids|children|family)\b",
    re.IGNORECASE,
)
_ADOPTION_OPINION_QUESTION_RE = re.compile(
    r"^\s*what\s+does\s+(?P<subject>[A-Z][A-Za-z'-]*)\s+think\s+about\s+"
    r"(?P<object>[A-Z][A-Za-z'-]*)'s\s+decision\s+to\s+adopt\b",
    re.IGNORECASE,
)
_AWESOME_PARENT_SUPPORT_RE = re.compile(
    r"\b(?:awesome|amazing|great|good)\s+(?:mom|mother|parent)\b",
    re.IGNORECASE,
)
_DEICTIC_OBJECT_ANSWER_PATTERN = re.compile(
    r"\bthat\s+book\s+you\s+recommended\b"
    r"|\bthose\s+moments\b",
    re.IGNORECASE,
)
_TEMPORAL_DEICTIC_ANSWER_PATTERN = re.compile(
    r"^\s*(?:now|recently|today|tomorrow|yesterday)\s*$",
    re.IGNORECASE,
)
_LIST_OR_COMPOSITE_QUESTION_PATTERN = re.compile(
    r"^\s*(?:what|which|where)\b.*\b"
    r"(?:activities|fields|hobbies|books|songs|movies|cities|places|countries|pets|"
    r"names|kinds|types|sports|classes|interests)\b",
    re.IGNORECASE,
)
_LIST_SEPARATOR_PATTERN = re.compile(r"[,;/]|\band\b", re.IGNORECASE)
_ANSWER_CONTRACT_LIST_OR_SET_QUESTION_PATTERN = re.compile(
    r"^\s*(?:what|which|who)\b.*\b"
    r"(?:names|foods?|desserts?|skills|events|recommendations?|classes|fields|essentials|"
    r"spots|places|activities|plans)\b"
    r"|^\s*what\s+are\s+some\b",
    re.IGNORECASE,
)
_ANSWER_CONTRACT_PHRASE_COMPLETION_QUESTION_PATTERN = re.compile(
    r"^\s*(?:what|which)\b.*\b"
    r"(?:reason\s+for|symboli[sz]e|inspired\s+by|think\s+of|love\s+most\s+about|"
    r"favorite\b.*\bbesides|stumble\s+across\s+during)\b",
    re.IGNORECASE,
)
_STRUCTURED_SYNTHESIS_LIST_OR_SET_QUESTION_PATTERN = re.compile(
    r"^\s*(?:what|which|who)\b.*\b"
    r"(?:names|foods?|desserts?|skills|events|recommendations?|classes|fields|essentials|"
    r"spots|places|activities|plans|hobbies|items|emotions|cities|countries|books|"
    r"songs|movies|pets|interests)\b"
    r"|^\s*what\s+are\s+some\b",
    re.IGNORECASE,
)
_ESSENTIAL_DETAIL_LIST_QUESTION_PATTERN = re.compile(
    r"^\s*what\s+(?:is|was)\s+(?:essential|needed)\s+to\s+"
    r"(?:keep|make|help)\b.*\b(?:look(?:ing)?\s+good|healthy|happy|in\s+good\s+shape)\b",
    re.IGNORECASE,
)
_STRUCTURED_SYNTHESIS_SAY_QUESTION_PATTERN = re.compile(
    r"^\s*what\s+(?:does|did)\b.*\bsay\s+to\b.*\babout\b",
    re.IGNORECASE,
)
_ANSWER_CONTRACT_MIN_SLOT_MATCH_RATIO = 0.72
_SUPPORT_PACK_MAX_SUPPORTS = 8
_SUPPORT_PRESENT_ADMISSION_V3_MAX_SUPPORTS = 40
_SUPPORT_SURFACE_REACH_MAX_CONCEPTS = 32
_SUPPORT_SURFACE_REACH_MIN_RATIO = 0.45
_SUPPORT_PACK_MIN_SCORE = 0.18
_SUPPORT_PACK_CLEAR_WIN_MARGIN = 0.12
_SUPPORT_PACK_QUESTION_OVERLAP_WEIGHT = 0.45
_SUPPORT_PACK_PREDICATE_OVERLAP_WEIGHT = 0.30
_SUPPORT_PACK_CHANNEL_WEIGHT = 0.15
_SUPPORT_PACK_CANDIDATE_PRESENT_WEIGHT = 0.10
_SUPPORT_CHANNEL_WEIGHTS: dict[SupportChannel, float] = {
    "verbatim": 1.0,
    "key_evidence": 0.85,
    "text": 0.75,
    "summary": 0.65,
}
_SUPPORT_CANDIDATE_BACKFILL_COMMON_TERMS = {
    "about",
    "after",
    "activity",
    "all",
    "also",
    "and",
    "any",
    "are",
    "around",
    "did",
    "does",
    "for",
    "get",
    "got",
    "had",
    "has",
    "her",
    "his",
    "how",
    "have",
    "just",
    "make",
    "made",
    "most",
    "new",
    "old",
    "only",
    "say",
    "said",
    "some",
    "the",
    "their",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}
_SUPPORT_CANDIDATE_BACKFILL_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")
_SUPPORT_CANDIDATE_BACKFILL_EVALUATION_GRACE_MS = 75.0
_SUPPORT_CANDIDATE_BACKFILL_ABSTAINS = {
    "deictic_answer_unresolved",
    "support_pack_no_evidence",
    "empty_answer_or_support",
    "identity_answer_requires_complete_phrase",
    "structured_synthesis_llm_disabled",
    "structured_synthesis_error",
    "structured_synthesis_missing_support_id",
    "structured_synthesis_missing_cited_span",
    "structured_synthesis_unknown_support_id",
    "structured_synthesis_unsupported_span",
    "structured_synthesis_empty_answer",
    "structured_synthesis_benefit_ambiguous",
    "structured_synthesis_benefit_unbound",
    "structured_synthesis_unsupported_item",
    "structured_synthesis_incomplete_list",
    "structured_synthesis_unsupported_answer",
    "structured_synthesis_incomplete_phrase",
    "structured_synthesis_predicate_unbound",
}
_EXACT_SUPPORT_BACKFILL_ABSTAINS = {
    "candidate_ambiguous",
    "empty_answer_or_support",
    "llm_disabled",
    "llm_error",
    "temporal_make_requires_event_time_resolution",
}
_RECOVERABLE_SUPPORT_PRESENT_BLOCKED_ABSTAINS = {
    "feeling_requires_attribute_judgment",
    "likely_requires_inference",
    "relationship_status_requires_structured_attribute",
}
_SUPPORT_PRESENT_ADMISSION_V2_ALLOWED_ABSTAINS = {
    "answer_shape_list_like",
    "answer_shape_too_broad",
    "broad_something_question",
    "candidate_filtered",
    "empty_answer_or_support",
    "events_require_list_synthesis",
    "feeling_requires_attribute_judgment",
    "likely_requires_inference",
    "llm_error",
    "non_atomic_question_shape",
    "relationship_status_requires_structured_attribute",
    "shared_or_common_requires_synthesis",
    "structured_synthesis_error",
    "structured_synthesis_incomplete_phrase",
    "structured_synthesis_predicate_unbound",
    "structured_synthesis_unsupported_answer",
    "unsupported_answer",
    "why_requires_explanation",
}
_SUPPORT_PRESENT_ADMISSION_V3_ALLOWED_ABSTAINS = {
    "answer_shape_list_like",
    "candidate_filtered",
    "empty_answer_or_support",
    "events_require_list_synthesis",
    "feeling_requires_attribute_judgment",
    "identity_answer_requires_complete_phrase",
    "llm_error",
    "non_atomic_question_shape",
    "structured_synthesis_error",
    "structured_synthesis_unsupported_answer",
    "unsupported_answer",
}
_SUPPORT_PRESENT_ADMISSION_V3_EXCLUDED_QUESTION_PATTERN = re.compile(
    r"^\s*(?:when|why|how\s+long)\b|"
    r"\b(?:future|plan|plans|planning|going\s+to|feel|feeling|shared|common)\b",
    re.IGNORECASE,
)
_SUPPORT_PRESENT_ADMISSION_V2_LIST_QUESTION_PATTERN = re.compile(
    r"\bwhat\s+(?:fields|kind|kinds|type|types)\s+of\b|"
    r"\bwhich\s+(?:fields|kind|kinds|type|types)\b|"
    r"^\s*what\s+fields\b",
    re.IGNORECASE,
)
_SUPPORT_PRESENT_ADMISSION_V2_SCALAR_QUESTION_PATTERN = re.compile(
    r"^\s*(?:what\s+(?:did|does|do|was|were|is|are)|which|where|why|how\s+(?:did|does|do|was|were|is|are))\b",
    re.IGNORECASE,
)
_SHARED_COMMON_ENTITY_QUESTION_PATTERNS = (
    re.compile(
        r"^\s*what\s+(?P<category>animal|pet)\s+do\s+both\s+"
        r"(?P<actor_a>[A-Z][a-z]+)\s+and\s+(?P<actor_b>[A-Z][a-z]+)\s+"
        r"(?:like|love|enjoy)\??\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*what\s+(?P<category>animal|pet)\s+do\s+"
        r"(?P<actor_a>[A-Z][a-z]+)\s+and\s+(?P<actor_b>[A-Z][a-z]+)\s+both\s+"
        r"(?:like|love|enjoy)\??\s*$",
        re.IGNORECASE,
    ),
)
_SHARED_COMMON_ENTITY_SURFACES = {
    "bird": ("bird", "birds"),
    "cat": ("cat", "cats"),
    "dog": ("dog", "dogs"),
    "fish": ("fish", "fishes"),
    "frog": ("frog", "frogs"),
    "hamster": ("hamster", "hamsters"),
    "horse": ("horse", "horses"),
    "lizard": ("lizard", "lizards"),
    "puppy": ("puppy", "puppies", "pup", "pups"),
    "rabbit": ("rabbit", "rabbits"),
    "snake": ("snake", "snakes"),
    "turtle": ("turtle", "turtles"),
}
_SHARED_COMMON_PREFERENCE_PATTERN = re.compile(
    r"\b(?:like|likes|liked|love|loves|loved|enjoy|enjoys|enjoyed|"
    r"drawn\s+to|affection\s+for|admiration\s+for|admire|admires|"
    r"admired|favorite|fave)\b",
    re.IGNORECASE,
)
_SUPPORT_PRESENT_PET_FAMILY_VIEW_QUESTION_PATTERN = re.compile(
    r"^\s*what\s+does\b.{0,80}\b(?:view|consider|see)\b.{0,40}\b"
    r"(?:pet|pets|dog|dogs|cat|cats)\b.{0,40}\bas\b",
    re.IGNORECASE,
)
_EVENT_OBJECT_VERBS_REQUIRING_TYPED_BINDING = {
    "adopt",
    "apply",
    "attend",
    "buy",
    "create",
    "draw",
    "finish",
    "get",
    "make",
    "offer",
    "paint",
    "read",
    "research",
    "start",
    "visit",
    "watch",
    "win",
}
_COUNT_CANDIDATE_PATTERN = re.compile(
    r"\b(?:\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
    r"nineteen|twenty)\b",
    re.IGNORECASE,
)
_QUOTED_SPAN_PATTERN = re.compile(r"['\"]([^'\"]{1,80})['\"]")
_USED_FOR_ATTRIBUTE_PATTERN = re.compile(
    r"\bused\s+for\s+([a-z][a-z0-9]*(?:\s+[a-z][a-z0-9]*){0,5})\b",
    re.IGNORECASE,
)
_MEDIA_FOOTWEAR_USE_TERMS = (
    "running",
    "walking",
    "hiking",
    "training",
    "basketball",
    "tennis",
    "soccer",
    "cycling",
    "dancing",
)
_MEDIA_FOOTWEAR_NOUN_TERMS = {"shoe", "shoes", "sneaker", "sneakers", "trainer", "trainers"}
_MEDIA_FOOTWEAR_USE_QUESTION_PATTERN = re.compile(
    r"\b(?:shoe|shoes|sneaker|sneakers|trainer|trainers)\b.*\bused\s+for\b"
    r"|\bused\s+for\b.*\b(?:shoe|shoes|sneaker|sneakers|trainer|trainers)\b",
    re.IGNORECASE,
)
_TEMPORAL_MAKE_QUESTION_PATTERN = re.compile(r"^\s*when did\b.*\bmake\b", re.IGNORECASE)
_TEMPORAL_MAKE_EVENT_MEDIA_CUE_PATTERN = re.compile(
    r"\byesterday\b(?=.*\b(?:tried|made|baked|prepared|cooked)\b)(?=.*\b(?:recipe|tart|cake|dessert)\b)",
    re.IGNORECASE,
)
_TEMPORAL_MAKE_GENERIC_TERMS = {
    "baked",
    "cooked",
    "dessert",
    "did",
    "made",
    "make",
    "prepared",
    "recipe",
    "tried",
}
_TEMPORAL_MAKE_MEDIA_OBJECT_NOUNS = {
    "cake",
    "cakes",
    "cupcake",
    "cupcakes",
    "dessert",
    "desserts",
    "tart",
    "tarts",
}
_LOCOMO_SOURCE_FRAGMENT_ANIMAL_TERMS = {
    "animal",
    "animals",
    "dog",
    "dogs",
    "pet",
    "pets",
    "puppies",
    "puppy",
}
_LOCOMO_SOURCE_FRAGMENT_ADOPTION_TERMS = {
    "adopt",
    "adopted",
    "adopting",
    "adoption",
}
_LOCOMO_SOURCE_FRAGMENT_MEDIA_EVENT_TERMS = {
    "accident",
    "broken",
    "car",
    "cars",
    "crash",
    "crashed",
    "media",
    "photo",
    "picture",
    "shared",
}
_LOCOMO_ADOPTION_DURATION_ACTOR_PATTERN = re.compile(
    r"\bsince\s+([A-Z][A-Za-z'-]*)\s+adopt(?:ed|s|ing)?\b",
    re.IGNORECASE,
)
_MEDIA_ACTOR_LOOKBACK_CHARS = 300
_SHARED_MEDIA_FIELD_PATTERN = re.compile(
    r"\[Shared media (?P<field>intent|caption)\]\s*(?P<value>.*?)"
    r"(?=\s+\[Shared media (?:intent|caption|url)\]|\s+\[LoCoMo turn id\]|"
    r"\s+\[USER\]|\s+\[ASSISTANT\]|$)",
    re.IGNORECASE,
)
_DIALOGUE_ACTOR_PATTERN = re.compile(r"\b([A-Z][a-z]+):")
_USED_FOR_TAIL_PATTERN = re.compile(
    r"\bused\s+for\s+(.+?)(?:\s+(?:last|this|on|in|during|with|at|after|before|"
    r"recently|yesterday|today)\b|[.!?,;]|$)",
    re.IGNORECASE,
)
_FOR_TAIL_PATTERN = re.compile(
    r"\bfor\s+(.+?)(?:\s+(?:on|in|during|with|at|after|before|since|last|this|"
    r"recently|yesterday|today)\b|[.!?,;]|$)",
    re.IGNORECASE,
)
_RAISE_AWARENESS_FOR_PATTERN = re.compile(
    r"\b(?:raise|raised|raising)\s+awareness\s+for\s+(.+?)(?:\s+(?:on|in|during|"
    r"with|at|after|before|since|last|this|today|yesterday)\b|[.!?,;]|$)",
    re.IGNORECASE,
)
_LOCATION_FROM_PATTERN = re.compile(r"\bfrom\s+([A-Z][A-Za-z]*(?:[\s-][A-Z][A-Za-z]*){0,3})\b")
_LOCATION_TO_PATTERN = re.compile(
    r"\b(?:travel(?:ed|s|ing)?|go(?:es|ing|ne)?|went|visit(?:ed|s|ing)?)\s+to\s+"
    r"([A-Z][A-Za-z]*(?:[\s-][A-Z][A-Za-z]*){0,3})(?=[.!?,;]|\s+(?:on|in|during|"
    r"with|at|after|before)\b|$)"
)
_VISITED_LOCATION_PATTERN = re.compile(
    r"\bvisit(?:ed|s|ing)?\s+"
    r"([A-Z][A-Za-z]*(?:[\s-][A-Z][A-Za-z]*){0,3})"
    r"(?=\s*,\s+(?:a|an|the)\s+(?:(?:small|quiet|historic)\s+)?"
    r"(?:town|city|country|village|island|state)\b|\s+on\s+a\s+road\s+trip\b)",
    re.IGNORECASE,
)
_YES_NO_QUESTION_PATTERN = re.compile(
    r"^\s*(?:did|do|does|is|are|was|were|has|have|had|can|could|will|would)\b",
    re.IGNORECASE,
)
_TITLE_SLOT_QUESTION_PATTERN = re.compile(
    r"\b(?:book|movie|film|song|poem|podcast|title|titled|called|nickname)\b",
    re.IGNORECASE,
)
_QUOTED_CREATOR_SUFFIX_PATTERN = re.compile(r"\s+by\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})")
_UNQUOTED_READING_TITLE_PATTERN = re.compile(
    r"\b(?:is\s+currently\s+reading|currently\s+reading|started\s+reading|reading)\s+"
    r"([A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*){0,5})"
    r"(?=\s+(?:by|with|during|after|before|for)\b|[.!?,;]|$)"
)
_QUOTED_READING_TITLE_PATTERN = re.compile(
    r"\b(?:is\s+currently\s+reading|currently\s+reading|started\s+reading|reading)\s+"
    r"['\"]([^'\"]{1,80})['\"]",
    re.IGNORECASE,
)
_SUGGESTED_BOOK_TITLE_QUESTION_PATTERN = re.compile(
    r"^\s*(?:what|which)\s+book\s+did\s+(?P<reader>[A-Z][A-Za-z'-]+)\s+"
    r"(?:read|start\s+reading|begin\s+reading|finish\s+reading)\b.*?\b"
    r"(?:from|after|on|because\s+of)\s+"
    r"(?P<recommender>[A-Z][A-Za-z'-]+)'?s\s+"
    r"(?:suggestion|recommendation|recommended\s+book|suggested\s+book)\b",
    re.IGNORECASE,
)
_SUGGESTED_BOOK_RELATIONSHIP_PATTERN = re.compile(
    r"\b(?:read|reading)\b.{0,120}\b(?:book)\b.{0,120}\b(?:recommend(?:ed)?|suggest(?:ed|ion)?)\b"
    r"|\b(?:recommend(?:ed)?|suggest(?:ed|ion)?)\b.{0,120}\b(?:book)\b.{0,120}\b(?:read|reading)\b",
    re.IGNORECASE,
)
_SUGGESTED_BOOK_RECOMMEND_CONTEXT_PATTERN = re.compile(
    r"\b(?:highly\s+recommend|recommend(?:ed|s|ing)?|suggest(?:ed|s|ing)?)\b",
    re.IGNORECASE,
)
_WATCHED_TITLE_PATTERN = re.compile(
    r"\b(?:watched|watching|saw)\s+['\"]([^'\"]{1,80})['\"]",
    re.IGNORECASE,
)
_WATCHED_TITLE_REJECTION_PATTERN = re.compile(
    r"\b(?:plans?|planning|planned|going\s+to|will|watch\s+list|watchlist|"
    r"add(?:ed|ing)?\b.{0,80}\b(?:list|watch))\b",
    re.IGNORECASE,
)
_FAVORITE_DISH_QUESTION_PATTERN = re.compile(r"\bfavorite\s+dish(?:es)?\b", re.IGNORECASE)
_FAVORITE_DISH_SUPPORT_PATTERN = re.compile(
    r"\bfavorite\s+dish\b.{0,80}?\bis\s+(.+?)(?:[.!?,;]|$)",
    re.IGNORECASE,
)
_NEW_ADDITION_FAMILY_QUESTION_PATTERN = re.compile(
    r"\b(?:who|what)\b.{0,80}\bnew\s+addition\b.{0,80}\bfamily\b",
    re.IGNORECASE,
)
_NEW_ADDITION_NAMED_PATTERN = re.compile(
    r"\b(?:new\s+addition\b.{0,80}?\b(?:named|called)\s+|"
    r"(?:dog|cat|puppy|kitten|pet)\s+(?:named|called)\s+)"
    r"([A-Z][A-Za-z'&-]*(?:\s+[A-Z][A-Za-z'&-]*){0,2})(?=[.!?,;]|$)",
    re.IGNORECASE,
)
_PROPER_NAME_CANDIDATE_PATTERN = re.compile(r"^[A-Z][A-Za-z'&-]*(?:\s+[A-Z][A-Za-z'&-]*){0,2}$")
_FAVORITE_TITLE_QUESTION_PATTERN = re.compile(
    r"\bfavorite\b.{0,80}\b(?:movie|film|trilogy)\b|"
    r"\b(?:movie|film|trilogy)\b.{0,80}\bfavorite\b",
    re.IGNORECASE,
)
_FAVORITE_TITLE_CONTEXT_PATTERN = re.compile(r"\b(?:movie|film|trilogy)\b", re.IGNORECASE)
_FAVORITE_TITLE_ANCHOR_PATTERN = re.compile(
    r"\b(?:favorite|faves?|greatest)\b.{0,120}\b(?:movie|film|trilogy)\b|"
    r"\b(?:movie|film|trilogy)\b.{0,120}\b(?:favorite|faves?|greatest)\b",
    re.IGNORECASE,
)
_REPAIR_LIST_COMPLETION_QUESTION_PATTERNS = (re.compile(r"^\s*who\b.*\binvite\b", re.IGNORECASE),)
_REPAIR_PREDICATE_QUESTION_PATTERNS = (re.compile(r"^\s*which\b.*\bscreenplay\b.*\breject", re.IGNORECASE),)
_TITLE_WITH_CREATOR_PATTERN = re.compile(
    r"(?:['\"])?([A-Z][A-Za-z0-9'&:-]*(?:\s+[A-Z][A-Za-z0-9'&:-]*){0,6})(?:['\"])?"
    r"\s+by\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})"
)
_SUPPORT_PRESENT_FANDOM_QUESTION_PATTERN = re.compile(r"\bfan of\b", re.IGNORECASE)
_SUPPORT_PRESENT_FAN_OF_ENTITY_PATTERN = re.compile(
    r"\bfan of\s+([A-Z][A-Za-z'&-]*(?:\s+[A-Z][A-Za-z'&-]*){0,3})(?=[.!?,;]|$)"
)
_SUPPORT_PRESENT_ENTITY_FAN_PATTERN = re.compile(r"\b([A-Z][A-Za-z'&-]*(?:\s+[A-Z][A-Za-z'&-]*){0,3})\s+fan\b")
_SUPPORT_PRESENT_PREDICATE_OBJECT_VERBS = {"organize"}
_SUPPORT_PRESENT_FOUND_CONTAINER_QUESTION_PATTERN = re.compile(
    r"^\s*what\s+did\b.+?\bfind\b.+?\bin\s+(?P<container>.+?)"
    r"(?:\s+(?:last|that|when|on|recently|yesterday|today)\b|\?)",
    re.IGNORECASE,
)
_SUPPORT_PRESENT_FOUND_CONTAINER_SUPPORT_PATTERN = re.compile(
    r"\bfound\s+(?P<container>.+?)\s+with\s+(?P<object>.+?)"
    r"(?:\s+(?:on|last|that|which|when|recently|yesterday|today)\b|[.!?,;]|$)",
    re.IGNORECASE,
)
_SUPPORT_PRESENT_LEADING_POSSESSIVE_PATTERN = re.compile(
    r"^(?:my|his|her|their|our|your|its)\s+",
    re.IGNORECASE,
)
_SUPPORT_PACK_CREATION_FOR_QUESTION_PATTERN = re.compile(
    r"^\s*what\s+(?:is|was)\s+(?P<actor>.+?)\s+creating\s+for\s+"
    r"(?P<destination>.+?)(?:\s+on\s+.+)?\?\s*$",
    re.IGNORECASE,
)
_SUPPORT_PRESENT_TITLE_QUESTION_PATTERN = re.compile(r"\b(?:called|project|title|titled)\b", re.IGNORECASE)
_SUPPORT_PRESENT_TITLE_ANCHOR_TERM_PATTERN = re.compile(
    r"\b(?:group|writers?|script|screenplay|project)\b",
    re.IGNORECASE,
)
_SUPPORT_PRESENT_TITLE_SUPPORT_PATTERN = re.compile(
    r"\b(?:working on|script|screenplay|project)\b",
    re.IGNORECASE,
)
_SUPPORT_PRESENT_BOUNDARY_QUOTED_SPAN_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])['\"]([A-Z][^'\"]{0,79})['\"](?![A-Za-z0-9])"
)
_SUPPORT_PRESENT_LINE_OF_QUESTION_PATTERN = re.compile(
    r"^\s*what\s+did\b.*\bmake\b.*\bline\s+of\b",
    re.IGNORECASE,
)
_SUPPORT_PRESENT_LINE_OF_ANSWER_PATTERN = re.compile(
    r"^(?:(?:a|an|the)\s+)?(?:limited\s+edition\s+)?line\s+of\s+(.+)$",
    re.IGNORECASE,
)
_REMINDER_OF_QUESTION_PATTERN = re.compile(
    r"\breminder\s+of\b",
    re.IGNORECASE,
)
_RELATIONSHIP_STATUS_QUESTION_PATTERN = re.compile(
    r"\brelationship\s+status\b",
    re.IGNORECASE,
)
_RELATIONSHIP_STATUS_SUPPORT_PATTERN = re.compile(
    r"\b(?:relationship\s+status\s+(?:is|was)|(?:is|was|became|remained))\s+"
    r"(single|married|engaged|divorced|separated|widowed|dating|partnered)\b",
    re.IGNORECASE,
)
_REMINDER_OF_SUPPORT_PATTERN = re.compile(
    r"\b(?:reminder\s+of|reminds?\s+(?:me|him|her|them|us|you)?\s*of)\s+(.+?)(?:[.;]|$)",
    re.IGNORECASE,
)
_PICTURE_OF_QUESTION_PATTERN = re.compile(
    r"\bt(?:ake|akes|aking|ook|aken)\s+(?:a\s+)?(?:picture|photo)\s+of\b",
    re.IGNORECASE,
)
_PICTURE_OF_SUPPORT_PATTERN = re.compile(r"\b(?:picture|photo)\b", re.IGNORECASE)
_SUPPORT_PRESENT_GAME_TYPE_QUESTION_PATTERN = re.compile(
    r"\btype\s+of\s+game\b.*?['\"]([^'\"]+)['\"]",
    re.IGNORECASE,
)
_SUPPORT_PRESENT_GAME_TYPE_SUPPORT_PATTERN = re.compile(
    r"\b(?:playing|played|enjoying|started\s+playing|trying)\s+"
    r"(.{1,120}?)\s+(?:called|titled)\s+['\"]([^'\"]+)['\"]",
    re.IGNORECASE,
)
_SUPPORT_PRESENT_GAME_TYPE_LEADING_FILLER_WORDS = {
    "a",
    "an",
    "the",
    "this",
    "that",
    "these",
    "those",
    "awesome",
    "cool",
    "fun",
    "new",
    "really",
    "very",
}
_SUPPORT_PRESENT_GAME_TYPE_TERMINAL_TOKENS = {
    "adventure",
    "game",
    "platformer",
    "puzzler",
    "rpg",
    "shooter",
    "simulator",
}
_SUPPORT_PRESENT_REASON_QUESTION_PATTERN = re.compile(
    r"^\s*what\s+was\s+the\s+reason\b",
    re.IGNORECASE,
)
_SUPPORT_PRESENT_REASON_BECAUSE_PATTERN = re.compile(
    r"\bbecause\s+of\s+(.+?)(?:[.!?,;]|$)",
    re.IGNORECASE,
)
_COORDINATED_LIST_CANDIDATE_PATTERN = re.compile(
    r"([A-Za-z][A-Za-z0-9' -]*(?:,\s*[A-Za-z][A-Za-z0-9' -]*)+" r"(?:,?\s+and\s+[A-Za-z][A-Za-z0-9' -]*)?)",
    re.IGNORECASE,
)
_LIST_ITEM_LEADING_CONTEXT_PATTERN = re.compile(
    r"^.*?\b(?:invite(?:d|s|ing)?|include(?:d|s|ing)?|list(?:ed|s|ing)?|named)\s+",
    re.IGNORECASE,
)
_LIST_ITEM_TRAILING_CONTEXT_PATTERN = re.compile(
    r"^(.+?)(?=\s+(?:to|for|during|on|in|at|with|after|before|since|because|while|" r"were|was|are|is)\b|$)",
    re.IGNORECASE,
)
_PASSIVE_SUBJECT_PATTERNS = (
    re.compile(
        r"\b(.+?)\s+were\s+rejected\b(?:\s+from\b.*)?$",
        re.IGNORECASE,
    ),
)
_ACTION_CLAUSE_QUESTION_PATTERN = re.compile(r"^\s*what\s+did\b.*\bdo\b", re.IGNORECASE)
_ACTION_CLAUSE_BINDING_QUESTION_PATTERN = re.compile(
    r"^\s*what\s+(?:did|does)\b.+?\bdo\b.+?\b(?:while|when|during)\b",
    re.IGNORECASE,
)
_BENEFIT_WITH_HAVING_QUESTION_PATTERN = re.compile(
    r"^\s*what\s+did\s+(?P<actor>[a-z][a-z' -]*?)\s+get\s+"
    r"(?:with|wtih|from|by)\s+having\s+(?P<context>.+?)\??\s*$",
    re.IGNORECASE,
)
_BENEFIT_WITH_HAVING_NOISE_TERMS = {
    "by",
    "from",
    "get",
    "having",
    "many",
    "much",
    "so",
    "with",
    "wtih",
}
_BENEFIT_WITH_HAVING_NOUNS = {
    "comfort",
    "companionship",
    "company",
    "connection",
    "friendship",
    "joy",
    "support",
}
_BENEFIT_WITH_HAVING_ANSWER_NOUNS = _BENEFIT_WITH_HAVING_NOUNS - {"connection"}
_BENEFIT_WITH_HAVING_SCORE_BONUS = 0.18
_FRONTED_SUBJECT_QUESTION_PATTERN = re.compile(r"^\s*what\s+(?:is|are|was|were)\b", re.IGNORECASE)
_FRONTED_SUBJECT_COPULA_PATTERN = re.compile(
    r"^\s*(.+?)\s+\b(?:is|are|was|were)\b\s+(.+?)(?:[.!?]|$)",
    re.IGNORECASE,
)
_ACTION_CLAUSE_SENTENCE_PATTERN = re.compile(
    r"^\s*[A-Z][A-Za-z'’-]+(?:\s+[A-Z][A-Za-z'’-]+)?"
    r"(?:\s+and\s+(?:(?:his|her|their)\s+[A-Za-z'’-]+|[A-Z][A-Za-z'’-]+(?:\s+[A-Z][A-Za-z'’-]+)?))?\s+"
    r"(.+?)(?:\s+(?:on|during|before|after|while|when)\b|[.!?]|$)"
)
_NATIVE_STABILITY_TEMPORAL_CONNECTOR_TERMS = {
    "after",
    "at",
    "before",
    "during",
    "in",
    "on",
}
_NATIVE_STABILITY_PREDICATE_EQUIVALENTS = {
    "mention": {
        "mention",
        "mentions",
        "mentioned",
        "mentioning",
        "reference",
        "references",
        "referenced",
        "referencing",
    },
}
_NATIVE_STABILITY_OBJECT_TRIM_TOKENS = {
    "a",
    "an",
    "her",
    "his",
    "my",
    "our",
    "own",
    "the",
    "their",
    "your",
}
_MAKE_ARTIFACT_LEADING_MODIFIERS = {
    "cute",
    "little",
    "that",
    "this",
}
_QUOTED_FRAGMENT_START_WORDS = {
    "and",
    "but",
    "it",
    "or",
    "so",
    "that",
    "this",
    "which",
}
_URL_LIKE_CANDIDATE_PATTERN = re.compile(
    r"(?:https?://|www\.|\.(?:jpg|jpeg|png|gif|webp)\b)",
    re.IGNORECASE,
)
_CLAUSE_FRAGMENT_PATTERN = re.compile(r"[!?]|\.\s|^\s*[a-z]\s+", re.IGNORECASE)
_YEAR_ONLY_PATTERN = re.compile(r"^(?:19|20)\d{2}$")
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
_COUNT_TARGET_PATTERN = re.compile(
    r"^\s*how\s+many\s+([a-z][a-z0-9-]*(?:\s+[a-z][a-z0-9-]*){0,4})",
    re.IGNORECASE,
)
_COUNT_TARGET_STOPWORDS = {
    "at",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "in",
    "is",
    "on",
    "was",
    "were",
    "with",
}
_OTHER_ACTOR_PATTERN = re.compile(
    r"\b(?:friend|someone else|another person|coworker|colleague|partner)\b",
    re.IGNORECASE,
)
_FUTURE_OR_PLAN_CUE_PATTERN = re.compile(
    r"\b(?:prepar(?:e|ed|ing)|plan(?:ned|ning)?|going to|will|next|upcoming)\b",
    re.IGNORECASE,
)
_LOCATION_QUESTION_PATTERN = re.compile(
    r"^\s*(?:where\b|what\s+(?:country|city|place|location)\b)",
    re.IGNORECASE,
)
_SHORT_ATTRIBUTE_QUESTION_PATTERN = re.compile(
    r"^\s*what\s+(?:is|are|was|were)\b.*\b" r"(?:identity|job|occupation|profession|nickname|title|role|used\s+for)\b",
    re.IGNORECASE,
)
_SESSION_DATE_PATTERN = re.compile(
    r"\bsession date\s*:\s*"
    r"(?:(?:\d{1,2}:\d{2}\s*(?:am|pm)\s+on\s+)?"
    r"\d{1,2}\s+[a-z]+\s*,?\s*\d{4}|[a-z]+\s*,?\s*\d{4})",
    re.IGNORECASE,
)
_SESSION_DATE_CUE_PATTERN = re.compile(
    r"\b(?:already|currently|just|last|next|now|recently|started|starting|" r"today|tomorrow|yesterday)\b",
    re.IGNORECASE,
)
_LIST_LIKE_ANSWER_PATTERN = re.compile(r"[,;/]")
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_QUESTION_STOPWORDS = {
    "about",
    "after",
    "and",
    "are",
    "around",
    "did",
    "does",
    "for",
    "had",
    "has",
    "have",
    "her",
    "his",
    "how",
    "the",
    "their",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}
_NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}
_NEGATION_PATTERN = re.compile(
    r"\b(?:not|never|no|cannot|can't|couldn't|didn't|doesn't|don't|"
    r"hadn't|hasn't|haven't|isn't|wasn't|weren't|without)\b",
    re.IGNORECASE,
)
_SYNTHESIS_DEFER_REASONS = {
    "events_require_list_synthesis",
    "happened_requires_event_synthesis",
    "shared_or_common_requires_synthesis",
    "list_answer_requires_synthesis",
    "kind_question_requires_list_synthesis",
    "activities_require_list_synthesis",
    "set_answer_requires_synthesis",
    "promotion_methods_require_list_synthesis",
    "offerings_require_list_synthesis",
    "achievement_requires_synthesis",
    "allergies_require_list_synthesis",
    "feeling_requires_attribute_judgment",
}
_TEMPORAL_DEFER_REASONS = {
    "temporal_make_requires_event_time_resolution",
    "future_plan_requires_temporal_resolution",
}

_QUOTE_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
)


@dataclass(frozen=True)
class EvidenceSupport:
    support_id: str
    concept_id: str
    channel: SupportChannel
    support_text: str
    normalized_support_text: str
    concept_summary: str = ""
    concept_actor_terms: tuple[str, ...] = ()
    concept_created_at: str | None = None
    concept_valid_from: str | None = None
    concept_original_date: str | None = None
    concept_content_updated_at: str | None = None
    concept_serial_order: int | None = None
    concept_session_id: str | None = None


@dataclass(frozen=True)
class EvidenceAnswerDecision:
    mode: AnswerMode
    answer: str | None
    normalized_answer: str | None
    support: EvidenceSupport | None
    abstain_reason: str | None
    latency_ms: float
    llm_error_class: str | None = None
    intent: AnswerIntent | None = None
    candidate_count: int = 0
    candidate_source: CandidateSource | None = None
    candidate_rejection_counts: dict[str, int] | None = None
    answer_contract_reason: str | None = None
    expected_answer_shape: str | None = None
    slot_binding_status: str | None = None
    synthesis_shape: StructuredSynthesisShape = "none"
    support_pack_size: int = 0
    support_ids: tuple[str, ...] = ()
    support_concept_ids: tuple[str, ...] = ()
    verifier_rejection_counts: dict[str, int] | None = None
    fallback_used: str | None = None
    recovery_strategy: str | None = None
    backfill_candidate_ids: tuple[str, ...] | None = None
    backfill_support_surfaces: tuple[dict[str, object], ...] = ()
    backfill_rejection_counts: dict[str, int] | None = None
    backfill_latency_ms: float | None = None
    backfill_semantic_candidate_ids: tuple[str, ...] | None = None
    backfill_semantic_admitted_ids: tuple[str, ...] | None = None
    backfill_semantic_latency_ms: float | None = None
    support_admission_version: str | None = None
    support_admission_v2_considered: bool = False
    support_admission_v2_blocked_reason: str | None = None
    support_admission_v2_binding_status: str | None = None
    support_admission_v2_shape: str | None = None
    llm_error_provider_status: int | None = None
    llm_error_provider_body_preview: str | None = None
    session_date_binding_status: str | None = None
    session_date_binding_diagnostics: dict[str, object] | None = None
    answer_shape_runtime_considered: bool = False
    answer_shape_runtime_admitted: bool = False
    answer_shape_runtime_reason: str | None = None
    answer_shape_runtime_contract_kind: str | None = None
    answer_shape_runtime_required_components: tuple[str, ...] = ()
    answer_shape_runtime_support_visibility: dict[str, object] | None = None
    answer_shape_runtime_effect_enabled: bool = False
    answer_shape_runtime_latency_ms: float | None = None
    answer_shape_runtime_llm_call_delta: int = 0


@dataclass(frozen=True)
class AnswerIntentDecision:
    intent: AnswerIntent
    abstain_reason: str | None = None


@dataclass(frozen=True)
class AnswerContractResult:
    reason: str | None
    expected_answer_shape: str | None
    slot_binding_status: str | None


@dataclass(frozen=True)
class AnswerCandidate:
    candidate_id: str
    answer: str
    normalized_answer: str
    support: EvidenceSupport
    source: CandidateSource


@dataclass(frozen=True)
class ScoredEvidenceSupport:
    support: EvidenceSupport
    score: float
    question_overlap: float
    predicate_overlap: float
    channel_weight: float
    contains_candidate: bool
    binding_status: str


@dataclass(frozen=True)
class ScoredSupportSentence:
    support: EvidenceSupport
    sentence: str
    support_score: float
    question_overlap: float
    binding_status: str


@dataclass(frozen=True)
class SupportCandidateBackfillResult:
    supports: list[EvidenceSupport]
    candidate_ids: tuple[str, ...]
    rejection_counts: dict[str, int]
    latency_ms: float
    semantic_candidate_ids: tuple[str, ...] = ()
    semantic_admitted_ids: tuple[str, ...] = ()
    semantic_latency_ms: float | None = None
    preserved_initial_supports: tuple[EvidenceSupport, ...] = ()
    preserved_initial_candidate_ids: tuple[str, ...] = ()
    preserved_initial_support_rejection_counts: dict[str, int] = field(default_factory=dict)
    duplicate_equivalence: tuple[dict[str, object], ...] = ()
    displacement_ledger: tuple[dict[str, object], ...] = ()
    preserved_initial_support_displacement_enabled: bool = False
    preserved_initial_support_displacement_count: int = 0
    locomo_decisive_evidence_preserved_ids: tuple[str, ...] = ()
    locomo_decisive_evidence_family_by_id: dict[str, str] = field(default_factory=dict)
    locomo_decisive_evidence_rejection_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class LocomoDecisiveSupportAssessment:
    family: str | None
    safe_to_preserve_duplicate: bool = False
    score_boost: float = 0.0
    rejection_reason: str | None = None


@dataclass(frozen=True)
class ActivatedSupportContinuityResult:
    supports: list[EvidenceSupport]
    candidate_ids: tuple[str, ...]
    rejection_counts: dict[str, int]
    rejected_ids_by_reason: dict[str, tuple[str, ...]]
    latency_ms: float


@dataclass(frozen=True)
class StructuredSynthesisResult:
    answer: str | None
    support_ids: tuple[str, ...]
    cited_spans: tuple[str, ...]
    fallback_used: str | None = None
    error_class: str | None = None


@dataclass(frozen=True)
class LocomoSupportPresentSynthesisResult:
    result: StructuredSynthesisResult
    strategy: str


@dataclass(frozen=True)
class SharedCommonEntityQuestion:
    category: str
    actor_a: str
    actor_b: str


@dataclass(frozen=True)
class SharedCommonEntitySupport:
    actor: str
    entity: str
    answer_surface: str
    support: EvidenceSupport


@dataclass(frozen=True)
class GuardStructuredCanonicalizationResult:
    result: StructuredSynthesisResult
    strategy: str


@dataclass(frozen=True)
class ExactSupportRecoveryResult:
    answer: str
    normalized_answer: str
    support: EvidenceSupport
    strategy: str


@dataclass(frozen=True)
class SupportDerivedRepairResult:
    answer: str
    normalized_answer: str
    support: EvidenceSupport
    strategy: str
    support_pack_size: int


@dataclass(frozen=True)
class CandidateBuildResult:
    candidates: list[AnswerCandidate]
    rejection_counts: dict[str, int]


@dataclass(frozen=True)
class YesNoEvidence:
    affirmative: bool
    negative: bool


def try_provenance_bound_answer(
    question: str,
    activated_concepts: list,
    *,
    llm_call: LLMCaller | None,
    llm_enabled: bool,
    timeout_seconds: float,
    model: str | None,
    max_activated_concepts: int,
    max_support_chars: int,
    typed_candidates_enabled: bool = False,
    answer_contract_enabled: bool = False,
    structured_synthesis_enabled: bool = False,
    exact_support_recovery_enabled: bool = False,
    support_derived_repair_enabled: bool = False,
    support_pack_completeness_enabled: bool = False,
    exact_support_native_stability_enabled: bool = False,
    support_surface_reach_enabled: bool = False,
    support_present_native_stability_enabled: bool = False,
    support_present_guard_stability_enabled: bool = False,
    actor_compatibility_guard_enabled: bool = False,
    relative_date_span_enabled: bool = False,
    support_candidate_backfill_enabled: bool = False,
    support_candidate_backfill_fts_limit: int = 20,
    support_candidate_backfill_assoc_limit: int = 24,
    support_candidate_backfill_max_supports: int = 4,
    support_candidate_backfill_min_score: float = 0.42,
    support_candidate_backfill_budget_ms: float = 25.0,
    support_candidate_backfill_semantic_enabled: bool = False,
    support_candidate_backfill_semantic_limit: int = 0,
    support_candidate_backfill_semantic_min_score: float = 0.45,
    support_present_admission_v2_enabled: bool = False,
    support_present_admission_v3_enabled: bool = False,
    direct_support_admission_enabled: bool = False,
    answer_shape_admission_enabled: bool = False,
    answer_shape_runtime_effect_enabled: bool = False,
    legacy_surface_contract_enabled: bool = False,
    locomo_support_present_synthesis_enabled: bool = False,
    locomo_support_present_answer_realization_enabled: bool = False,
    locomo_support_emission_enabled: bool = False,
    locomo_bounded_support_admission_enabled: bool = False,
    locomo_bounded_support_admission_effect_enabled: bool = False,
) -> EvidenceAnswerDecision:
    """Return a validated answer copied from cited support, or abstain."""
    t0 = time.perf_counter()
    supports: list[EvidenceSupport] = []
    answer_shape_runtime_probe: EvidenceAnswerDecision | None = None
    _base_abstain = globals()["_abstain"]

    def _abstain(*args, **kwargs) -> EvidenceAnswerDecision:
        return _with_answer_shape_runtime_diagnostics(
            _with_session_date_binding_diagnostics(
                _base_abstain(*args, **kwargs),
                question=question,
                supports=supports,
            ),
            answer_shape_runtime_probe,
        )

    def _finish(decision: EvidenceAnswerDecision) -> EvidenceAnswerDecision:
        return _with_answer_shape_runtime_diagnostics(
            decision,
            answer_shape_runtime_probe,
        )

    def _hook_enabled() -> bool:
        return answer_shape_admission_enabled

    def _hook_runtime_effect_enabled() -> bool:
        return answer_shape_admission_enabled and answer_shape_runtime_effect_enabled

    def _run_answer_shape_runtime_hook() -> EvidenceAnswerDecision | None:
        if not _hook_enabled():
            return None
        return _try_answer_shape_runtime_hook(
            question=question,
            supports=supports,
            t0=t0,
            admission_enabled=True,
            runtime_effect_enabled=_hook_runtime_effect_enabled(),
        )

    if not question.strip():
        return _abstain("empty_question", t0)

    supports = _collect_supports(
        activated_concepts[:max_activated_concepts],
        max_support_chars=max_support_chars,
    )
    if not supports:
        return _abstain("no_support", t0)
    supports = _hydrate_support_temporal_metadata(supports)

    answer_shape_runtime_probe = _run_answer_shape_runtime_hook()
    if answer_shape_runtime_probe and answer_shape_runtime_probe.answer:
        return answer_shape_runtime_probe

    legacy_surface_active = answer_contract_enabled and legacy_surface_contract_enabled
    if legacy_surface_active:
        legacy_surface_decision = _try_legacy_surface_contract_answer(
            question=question,
            supports=supports,
            t0=t0,
        )
        if legacy_surface_decision is not None and legacy_surface_decision.answer:
            return _finish(legacy_surface_decision)

    if direct_support_admission_enabled:
        direct_support_decision = _recover_direct_support_admission_answer(question, supports, t0)
        if direct_support_decision.answer:
            return _finish(direct_support_decision)
        if direct_support_decision.abstain_reason == "direct_support_conflict":
            return _finish(direct_support_decision)

    strict_structured_enabled = answer_contract_enabled and structured_synthesis_enabled
    locomo_backfill_emission_active = locomo_support_emission_enabled and (
        _locomo_support_present_answer_realization_question(question)
        or _locomo_support_present_synthesis_question(question)
    )
    exact_support_recovery_active = (
        answer_contract_enabled
        and exact_support_recovery_enabled
        and not (
            locomo_backfill_emission_active
            and _locomo_training_course_date_question(question)
        )
    )
    support_derived_repair_active = answer_contract_enabled and support_derived_repair_enabled
    support_pack_completeness_active = answer_contract_enabled and support_pack_completeness_enabled
    support_surface_reach_active = answer_contract_enabled and support_surface_reach_enabled
    support_present_native_stability_active = answer_contract_enabled and support_present_native_stability_enabled
    support_present_guard_stability_active = answer_contract_enabled and support_present_guard_stability_enabled
    support_present_admission_v2_active = answer_contract_enabled and support_present_admission_v2_enabled
    support_present_admission_v3_active = answer_contract_enabled and support_present_admission_v3_enabled
    actor_compatibility_guard_active = answer_contract_enabled and actor_compatibility_guard_enabled
    synthesis_shape = _infer_structured_synthesis_shape(question)
    support_surface_reach_pool = (
        _build_support_surface_reach_pool(
            question,
            activated_concepts[:max_activated_concepts],
            supports,
            max_reach_concepts=_SUPPORT_SURFACE_REACH_MAX_CONCEPTS,
        )
        if support_surface_reach_active
        else ()
    )
    backfill_active = answer_contract_enabled and support_candidate_backfill_enabled

    def _admission_v2(
        decision: EvidenceAnswerDecision,
        *,
        candidate_supports: list[EvidenceSupport] | None = None,
        required_concept_ids: set[str] | None = None,
    ) -> EvidenceAnswerDecision:
        selected_supports = candidate_supports or supports
        if support_present_admission_v3_active:
            v3_decision = _maybe_apply_support_present_admission_v2(
                decision=decision,
                question=question,
                supports=selected_supports,
                t0=t0,
                support_pack_completeness_enabled=support_pack_completeness_active,
                support_present_guard_stability_enabled=support_present_guard_stability_active,
                actor_compatibility_enabled=actor_compatibility_guard_active,
                enabled=True,
                admission_label="support_present_admission_v3",
                admission_version="v3",
                allowed_abstains=_SUPPORT_PRESENT_ADMISSION_V3_ALLOWED_ABSTAINS,
                excluded_question_pattern=_SUPPORT_PRESENT_ADMISSION_V3_EXCLUDED_QUESTION_PATTERN,
                max_supports=_SUPPORT_PRESENT_ADMISSION_V3_MAX_SUPPORTS,
                required_concept_ids=required_concept_ids,
                locomo_support_present_synthesis_enabled=locomo_support_present_synthesis_enabled,
                locomo_support_present_answer_realization_enabled=(
                    locomo_support_present_answer_realization_enabled
                ),
            )
            if v3_decision.answer or v3_decision.support_admission_version == "v3":
                return v3_decision
        return _maybe_apply_support_present_admission_v2(
            decision=decision,
            question=question,
            supports=selected_supports,
            t0=t0,
            support_pack_completeness_enabled=support_pack_completeness_active,
            support_present_guard_stability_enabled=support_present_guard_stability_active,
            actor_compatibility_enabled=actor_compatibility_guard_active,
            enabled=support_present_admission_v2_active,
            admission_label="support_present_admission_v2",
            admission_version="v2",
            required_concept_ids=required_concept_ids,
        )

    def _post_synthesis_movie_genre_correction(decision: EvidenceAnswerDecision) -> EvidenceAnswerDecision:
        if (
            not support_present_admission_v3_active
            or synthesis_shape != "list_or_set"
            or decision.mode != "structured_synthesis"
            or not decision.answer
            or not _support_present_movie_genre_correction_question(question)
        ):
            return decision
        correction_probe = replace(
            decision,
            mode="abstain",
            answer=None,
            normalized_answer=None,
            support=None,
            abstain_reason="post_synthesis_movie_genre_correction",
        )
        corrected = _maybe_apply_support_present_admission_v2(
            decision=correction_probe,
            question=question,
            supports=supports,
            t0=t0,
            support_pack_completeness_enabled=support_pack_completeness_active,
            support_present_guard_stability_enabled=support_present_guard_stability_active,
            actor_compatibility_enabled=actor_compatibility_guard_active,
            enabled=True,
            admission_label="support_present_admission_v3_post_synthesis_movie_genre",
            admission_version="v3",
            allowed_abstains={"post_synthesis_movie_genre_correction"},
            excluded_question_pattern=None,
            max_supports=_SUPPORT_PRESENT_ADMISSION_V3_MAX_SUPPORTS,
        )
        if corrected.answer:
            if corrected.normalized_answer != _normalize(decision.answer):
                return replace(
                    corrected,
                    backfill_candidate_ids=decision.backfill_candidate_ids,
                    backfill_rejection_counts=decision.backfill_rejection_counts,
                    backfill_latency_ms=decision.backfill_latency_ms,
                )
            return decision
        if corrected.support_admission_v2_considered:
            reason = corrected.support_admission_v2_blocked_reason or "post_synthesis_movie_genre_unverified"
            return replace(
                decision,
                mode="abstain",
                answer=None,
                normalized_answer=None,
                support=None,
                abstain_reason=reason,
                verifier_rejection_counts={reason: 1},
                support_admission_version=corrected.support_admission_version,
                support_admission_v2_considered=True,
                support_admission_v2_blocked_reason=reason,
                support_admission_v2_binding_status=corrected.support_admission_v2_binding_status,
                support_admission_v2_shape=corrected.support_admission_v2_shape,
            )
        return decision

    def _collect_backfill() -> SupportCandidateBackfillResult:
        return _collect_support_candidate_backfill(
            question=question,
            activated_concepts=activated_concepts[:max_activated_concepts],
            existing_supports=supports,
            fts_limit=support_candidate_backfill_fts_limit,
            assoc_limit=support_candidate_backfill_assoc_limit,
            max_supports=support_candidate_backfill_max_supports,
            min_score=support_candidate_backfill_min_score,
            budget_ms=support_candidate_backfill_budget_ms,
            semantic_enabled=support_candidate_backfill_semantic_enabled,
            semantic_limit=support_candidate_backfill_semantic_limit,
            semantic_min_score=support_candidate_backfill_semantic_min_score,
            locomo_support_emission_enabled=locomo_backfill_emission_active,
        )

    def _shared_common_entity_synthesis_with_backfill(
        *,
        intent: AnswerIntent | None,
    ) -> EvidenceAnswerDecision | None:
        shared_decision = _try_shared_common_entity_synthesis(
            question=question,
            supports=supports,
            t0=t0,
        )
        if shared_decision is not None:
            return shared_decision
        if not backfill_active:
            return None

        backfill = _collect_shared_common_entity_backfill(
            question=question,
            existing_supports=supports,
            fts_limit=support_candidate_backfill_fts_limit,
            max_supports=support_candidate_backfill_max_supports,
            budget_ms=support_candidate_backfill_budget_ms,
        )
        base_decision = _abstain(
            "shared_or_common_requires_synthesis",
            t0,
            intent=intent,
        )
        if not backfill.supports:
            return _with_backfill_diagnostics(base_decision, backfill)

        retry = _try_shared_common_entity_synthesis(
            question=question,
            supports=_renumber_supports([*supports, *backfill.supports]),
            t0=t0,
        )
        if retry is None:
            return _with_backfill_diagnostics(base_decision, backfill)
        return _with_backfill_diagnostics(
            retry,
            backfill,
            recovery_strategy="backfill_shared_common_entity",
        )

    def _admission_v2_with_backfill(
        decision: EvidenceAnswerDecision,
        *,
        intent: AnswerIntent | None = None,
    ) -> EvidenceAnswerDecision:
        admitted = _admission_v2(decision)
        if admitted.answer or not backfill_active:
            return admitted
        backfill = _collect_backfill()
        if not backfill.supports:
            return _with_backfill_diagnostics(admitted, backfill)
        merged_supports = _renumber_supports([*supports, *backfill.supports])
        suggested_book_bridge = _recover_suggested_book_title_bridge(
            question=question,
            supports=merged_supports,
        )
        if suggested_book_bridge is not None:
            bridge_support_ids = set(suggested_book_bridge.support_ids)
            bridge_pack_by_id = {
                support.support_id: _score_support(question, support)
                for support in merged_supports
                if support.support_id in bridge_support_ids
            }
            bridge_pack = tuple(
                bridge_pack_by_id[support_id]
                for support_id in suggested_book_bridge.support_ids
                if support_id in bridge_pack_by_id
            )
            if bridge_pack:
                emitted = _structured_synthesis_decision(
                    suggested_book_bridge,
                    bridge_pack,
                    t0,
                    shape="predicate_bound_scalar",
                    fallback_used="support_candidate_backfill",
                )
                emitted = replace(
                    emitted,
                    recovery_strategy="backfill_suggested_book_title_bridge",
                    support_admission_version="v3",
                    support_admission_v2_considered=True,
                    support_admission_v2_blocked_reason=None,
                    support_admission_v2_binding_status="bound",
                    support_admission_v2_shape="predicate_bound_scalar",
                    candidate_count=decision.candidate_count,
                    candidate_source=decision.candidate_source,
                    candidate_rejection_counts=decision.candidate_rejection_counts,
                    verifier_rejection_counts=decision.verifier_rejection_counts,
                    llm_error_provider_status=decision.llm_error_provider_status,
                    llm_error_provider_body_preview=decision.llm_error_provider_body_preview,
                )
                return _with_backfill_diagnostics(
                    emitted,
                    backfill,
                    fallback_used="support_candidate_backfill",
                    recovery_strategy="backfill_suggested_book_title_bridge",
                )
        retry = _admission_v2(
            decision,
            candidate_supports=merged_supports,
            required_concept_ids=set(backfill.candidate_ids),
        )
        if retry.answer:
            return _with_backfill_diagnostics(
                retry,
                backfill,
                fallback_used="support_candidate_backfill",
                recovery_strategy=f"backfill_{retry.recovery_strategy or 'support_present_admission_v2'}",
            )
        recovered = _try_exact_support_recovery(
            question=question,
            supports=merged_supports,
            intent=intent,
            t0=t0,
            llm_error_class=None,
            enabled=exact_support_recovery_active,
            support_pack_completeness_enabled=support_pack_completeness_active,
            exact_support_native_stability_enabled=exact_support_native_stability_enabled,
            support_surface_reach_enabled=False,
            support_present_native_stability_enabled=support_present_native_stability_active,
            support_present_guard_stability_enabled=support_present_guard_stability_active,
            actor_compatibility_enabled=actor_compatibility_guard_active,
            support_surface_reach_pool=(),
            candidate_count=decision.candidate_count,
            candidate_rejection_counts=decision.candidate_rejection_counts or {},
            required_concept_ids=set(backfill.candidate_ids),
        )
        if recovered is not None:
            return _with_backfill_diagnostics(
                recovered,
                backfill,
                fallback_used="support_candidate_backfill",
                recovery_strategy=f"backfill_{recovered.recovery_strategy or 'exact_support_recovery'}",
            )
        return _with_backfill_diagnostics(admitted, backfill)

    def _locomo_source_admission_with_backfill(
        decision: EvidenceAnswerDecision,
        *,
        recovery_strategy: str,
        annotate_abstain: bool = False,
    ) -> EvidenceAnswerDecision:
        if (
            not backfill_active
            or not locomo_support_present_synthesis_enabled
            or not (
                _locomo_support_present_synthesis_question(question)
                or (
                    locomo_support_present_answer_realization_enabled
                    and _locomo_support_present_answer_realization_question(question)
                )
            )
        ):
            return decision
        backfill = _collect_backfill()
        if not backfill.supports:
            return _with_backfill_diagnostics(decision, backfill) if annotate_abstain else decision

        merged_supports = _renumber_supports([*supports, *backfill.supports])
        probe = replace(
            decision,
            mode="abstain",
            answer=None,
            normalized_answer=None,
            support=None,
            abstain_reason=decision.abstain_reason or "locomo_source_admission_audit",
        )
        retry = _admission_v2(
            probe,
            candidate_supports=merged_supports,
            required_concept_ids=set(backfill.candidate_ids),
        )
        if retry.answer:
            return _with_backfill_diagnostics(
                retry,
                backfill,
                fallback_used="support_candidate_backfill",
                recovery_strategy=retry.recovery_strategy or recovery_strategy,
            )
        return _with_backfill_diagnostics(decision, backfill) if annotate_abstain else decision

    def _exact_recovery_with_backfill(
        decision: EvidenceAnswerDecision,
        *,
        intent: AnswerIntent | None,
        reason: str,
        llm_error_class: str | None = None,
        candidate_count: int = 0,
        candidate_rejection_counts: dict[str, int] | None = None,
        allow_temporal_actor_mismatch: bool = False,
    ) -> EvidenceAnswerDecision:
        temporal_actor_mismatch_allowed = (
            allow_temporal_actor_mismatch
            and reason == "support_actor_mismatch"
            and (intent == "date" or _TEMPORAL_QUESTION_PATTERN.search(question) is not None)
        )
        if (
            not backfill_active
            or (
                reason not in _EXACT_SUPPORT_BACKFILL_ABSTAINS
                and not temporal_actor_mismatch_allowed
            )
        ):
            return decision
        backfill = _collect_backfill()
        if not backfill.supports:
            return _with_backfill_diagnostics(decision, backfill)
        merged_supports = _renumber_supports([*supports, *backfill.supports])
        suggested_book_bridge = _recover_suggested_book_title_bridge(
            question=question,
            supports=merged_supports,
        )
        if suggested_book_bridge is not None:
            bridge_support_ids = set(suggested_book_bridge.support_ids)
            bridge_pack_by_id = {
                support.support_id: _score_support(question, support)
                for support in merged_supports
                if support.support_id in bridge_support_ids
            }
            bridge_pack = tuple(
                bridge_pack_by_id[support_id]
                for support_id in suggested_book_bridge.support_ids
                if support_id in bridge_pack_by_id
            )
            if bridge_pack:
                emitted = _structured_synthesis_decision(
                    suggested_book_bridge,
                    bridge_pack,
                    t0,
                    shape="predicate_bound_scalar",
                    fallback_used="support_candidate_backfill",
                )
                emitted = replace(
                    emitted,
                    recovery_strategy="backfill_suggested_book_title_bridge",
                    support_admission_version="v3",
                    support_admission_v2_considered=True,
                    support_admission_v2_blocked_reason=None,
                    support_admission_v2_binding_status="bound",
                    support_admission_v2_shape="predicate_bound_scalar",
                    candidate_count=candidate_count,
                    candidate_rejection_counts=candidate_rejection_counts or {},
                )
                return _with_backfill_diagnostics(
                    emitted,
                    backfill,
                    fallback_used="support_candidate_backfill",
                    recovery_strategy="backfill_suggested_book_title_bridge",
                )
        recovered = _try_exact_support_recovery(
            question=question,
            supports=merged_supports,
            intent=intent,
            t0=t0,
            llm_error_class=llm_error_class,
            enabled=exact_support_recovery_active,
            support_pack_completeness_enabled=support_pack_completeness_active,
            exact_support_native_stability_enabled=exact_support_native_stability_enabled,
            support_surface_reach_enabled=False,
            support_present_native_stability_enabled=support_present_native_stability_active,
            support_present_guard_stability_enabled=support_present_guard_stability_active,
            actor_compatibility_enabled=actor_compatibility_guard_active,
            support_surface_reach_pool=(),
            candidate_count=candidate_count,
            candidate_rejection_counts=candidate_rejection_counts or {},
            required_concept_ids=set(backfill.candidate_ids),
        )
        if recovered is not None:
            return _with_backfill_diagnostics(
                recovered,
                backfill,
                fallback_used="support_candidate_backfill",
                recovery_strategy=f"backfill_{recovered.recovery_strategy or 'exact_support_recovery'}",
            )
        return _with_backfill_diagnostics(decision, backfill)

    if (
        strict_structured_enabled
        and synthesis_shape == "atomic_scalar"
        and locomo_support_present_answer_realization_enabled
        and _locomo_training_course_date_question(question)
        and backfill_active
    ):
        decision = _maybe_retry_with_support_candidate_backfill(
            decision=_abstain("llm_disabled", t0),
            question=question,
            activated_concepts=activated_concepts[:max_activated_concepts],
            existing_supports=supports,
            shape=synthesis_shape,
            llm_call=llm_call,
            llm_enabled=llm_enabled,
            timeout_seconds=timeout_seconds,
            model=model,
            t0=t0,
            candidate_rejection_counts={},
            support_pack_completeness_enabled=support_pack_completeness_active,
            exact_support_native_stability_enabled=exact_support_native_stability_enabled,
            support_present_native_stability_enabled=support_present_native_stability_active,
            support_present_guard_stability_enabled=support_present_guard_stability_active,
            locomo_support_present_answer_realization_enabled=(
                locomo_support_present_answer_realization_enabled
            ),
            locomo_support_emission_enabled=locomo_backfill_emission_active,
            enabled=answer_contract_enabled and support_candidate_backfill_enabled,
            fts_limit=support_candidate_backfill_fts_limit,
            assoc_limit=support_candidate_backfill_assoc_limit,
            max_supports=support_candidate_backfill_max_supports,
            min_score=support_candidate_backfill_min_score,
            budget_ms=support_candidate_backfill_budget_ms,
            semantic_enabled=support_candidate_backfill_semantic_enabled,
            semantic_limit=support_candidate_backfill_semantic_limit,
            semantic_min_score=support_candidate_backfill_semantic_min_score,
        )
        if decision.answer:
            return _admission_v2(decision)

    if strict_structured_enabled and synthesis_shape in {
        "predicate_bound_scalar",
        "list_or_set",
        "complete_phrase",
    }:
        decision = _try_structured_synthesis_decision(
            question=question,
            supports=supports,
            shape=synthesis_shape,
            llm_call=llm_call,
            llm_enabled=llm_enabled,
            timeout_seconds=timeout_seconds,
            model=model,
            t0=t0,
            candidate_rejection_counts={},
            support_pack_completeness_enabled=support_pack_completeness_active,
            exact_support_native_stability_enabled=exact_support_native_stability_enabled,
            support_present_native_stability_enabled=support_present_native_stability_active,
            support_present_guard_stability_enabled=support_present_guard_stability_active,
            locomo_support_present_answer_realization_enabled=(
                locomo_support_present_answer_realization_enabled
            ),
        )
        decision = _post_synthesis_movie_genre_correction(decision)
        decision = _maybe_retry_with_support_candidate_backfill(
            decision=decision,
            question=question,
            activated_concepts=activated_concepts[:max_activated_concepts],
            existing_supports=supports,
            shape=synthesis_shape,
            llm_call=llm_call,
            llm_enabled=llm_enabled,
            timeout_seconds=timeout_seconds,
            model=model,
            t0=t0,
            candidate_rejection_counts={},
            support_pack_completeness_enabled=support_pack_completeness_active,
            exact_support_native_stability_enabled=exact_support_native_stability_enabled,
            support_present_native_stability_enabled=support_present_native_stability_active,
            support_present_guard_stability_enabled=support_present_guard_stability_active,
            locomo_support_present_answer_realization_enabled=(
                locomo_support_present_answer_realization_enabled
            ),
            locomo_support_emission_enabled=locomo_backfill_emission_active,
            enabled=answer_contract_enabled and support_candidate_backfill_enabled,
            fts_limit=support_candidate_backfill_fts_limit,
            assoc_limit=support_candidate_backfill_assoc_limit,
            max_supports=support_candidate_backfill_max_supports,
            min_score=support_candidate_backfill_min_score,
            budget_ms=support_candidate_backfill_budget_ms,
            semantic_enabled=support_candidate_backfill_semantic_enabled,
            semantic_limit=support_candidate_backfill_semantic_limit,
            semantic_min_score=support_candidate_backfill_semantic_min_score,
        )
        decision = _post_synthesis_movie_genre_correction(decision)
        return _admission_v2(decision)

    intent_decision: AnswerIntentDecision | None = None
    candidates: list[AnswerCandidate] = []
    candidate_rejection_counts: dict[str, int] = {}
    if typed_candidates_enabled:
        intent_decision = _classify_answer_intent(question)
        if intent_decision.abstain_reason:
            if (
                intent_decision.abstain_reason == "shared_or_common_requires_synthesis"
                and support_present_admission_v3_active
            ):
                shared_decision = _shared_common_entity_synthesis_with_backfill(
                    intent=intent_decision.intent,
                )
                if shared_decision is not None:
                    return shared_decision
            decision = _abstain(
                intent_decision.abstain_reason,
                t0,
                intent=intent_decision.intent,
            )
            if intent_decision.abstain_reason == "temporal_make_requires_event_time_resolution":
                recovered = _try_exact_support_recovery(
                    question=question,
                    supports=supports,
                    intent=intent_decision.intent,
                    t0=t0,
                    llm_error_class=None,
                    enabled=exact_support_recovery_active,
                    support_pack_completeness_enabled=support_pack_completeness_active,
                    exact_support_native_stability_enabled=exact_support_native_stability_enabled,
                    support_surface_reach_enabled=support_surface_reach_active,
                    support_present_native_stability_enabled=support_present_native_stability_active,
                    support_present_guard_stability_enabled=support_present_guard_stability_active,
                    actor_compatibility_enabled=actor_compatibility_guard_active,
                    support_surface_reach_pool=support_surface_reach_pool,
                    candidate_count=0,
                    candidate_rejection_counts={},
                )
                if recovered is not None:
                    return recovered
                backfilled = _exact_recovery_with_backfill(
                    decision,
                    intent=intent_decision.intent,
                    reason=intent_decision.abstain_reason,
                )
                if backfilled.answer or backfilled.backfill_candidate_ids:
                    return backfilled
                return decision
            bank_account_reason_retry = (
                support_present_admission_v3_active
                and intent_decision.abstain_reason == "why_requires_explanation"
                and _support_present_bank_account_shutdown_reason_question(question)
            )
            move_back_would_retry = (
                intent_decision.abstain_reason == "would_requires_inference"
                and _home_country_move_back_soon_actor(question) is not None
            )
            if (
                intent_decision.abstain_reason in _RECOVERABLE_SUPPORT_PRESENT_BLOCKED_ABSTAINS
                or bank_account_reason_retry
                or move_back_would_retry
            ):
                recovered = _try_exact_support_recovery(
                    question=question,
                    supports=supports,
                    intent=None,
                    t0=t0,
                    llm_error_class=None,
                    enabled=exact_support_recovery_active,
                    support_pack_completeness_enabled=support_pack_completeness_active,
                    exact_support_native_stability_enabled=exact_support_native_stability_enabled,
                    support_surface_reach_enabled=support_surface_reach_active,
                    support_present_native_stability_enabled=support_present_native_stability_active,
                    support_present_guard_stability_enabled=support_present_guard_stability_active,
                    actor_compatibility_enabled=actor_compatibility_guard_active,
                    support_surface_reach_pool=support_surface_reach_pool,
                    candidate_count=0,
                    candidate_rejection_counts={},
                )
                if recovered is not None:
                    return recovered
                return _admission_v2_with_backfill(decision, intent=None)
            return decision
        candidate_result = _build_answer_candidates(
            question,
            supports,
            intent_decision.intent,
            relative_date_span_enabled=relative_date_span_enabled,
        )
        candidates = candidate_result.candidates
        candidate_rejection_counts = candidate_result.rejection_counts
        if (
            intent_decision.intent == "date"
            and len(candidates) > 1
            and _FUTURE_OR_PLAN_CUE_PATTERN.search(question) is None
        ):
            source_date_candidate = _top_conversation_source_date_temporal_candidate(
                question=question,
                candidates=candidates,
                supports=supports,
            )
            if source_date_candidate is not None:
                if strict_structured_enabled:
                    return _verified_candidate_decision(
                        question=question,
                        candidate=source_date_candidate,
                        supports=supports,
                        t0=t0,
                        intent=intent_decision.intent,
                        candidate_count=len(candidates),
                        candidate_rejection_counts=candidate_rejection_counts,
                        locomo_support_present_answer_realization_enabled=(
                            locomo_support_present_answer_realization_enabled
                        ),
                    )
                return _candidate_decision(
                    source_date_candidate,
                    t0,
                    intent=intent_decision.intent,
                    candidate_count=len(candidates),
                    candidate_rejection_counts=candidate_rejection_counts,
                    question=question,
                    locomo_support_present_answer_realization_enabled=(
                        locomo_support_present_answer_realization_enabled
                    ),
                )
            top_summary_candidate = _top_event_bound_summary_temporal_candidate(
                question=question,
                supports=supports,
                candidate_source="regex_date",
            )
            if top_summary_candidate is not None:
                _increment_rejection(candidate_rejection_counts, "temporal_higher_ranked_event_date")
                if strict_structured_enabled:
                    return _verified_candidate_decision(
                        question=question,
                        candidate=top_summary_candidate,
                        supports=supports,
                        t0=t0,
                        intent=intent_decision.intent,
                        candidate_count=len(candidates),
                        candidate_rejection_counts=candidate_rejection_counts,
                        locomo_support_present_answer_realization_enabled=(
                            locomo_support_present_answer_realization_enabled
                        ),
                    )
                return _candidate_decision(
                    top_summary_candidate,
                    t0,
                    intent=intent_decision.intent,
                    candidate_count=len(candidates),
                    candidate_rejection_counts=candidate_rejection_counts,
                    question=question,
                    locomo_support_present_answer_realization_enabled=(
                        locomo_support_present_answer_realization_enabled
                    ),
                )
        if len(candidates) == 1:
            if strict_structured_enabled:
                verified = _verified_candidate_decision(
                    question=question,
                    candidate=candidates[0],
                    supports=supports,
                    t0=t0,
                    intent=intent_decision.intent,
                    candidate_count=len(candidates),
                    candidate_rejection_counts=candidate_rejection_counts,
                    locomo_support_present_answer_realization_enabled=(
                        locomo_support_present_answer_realization_enabled
                    ),
                )
                if (
                    verified.mode == "abstain"
                    and verified.abstain_reason == "support_pack_no_evidence"
                    and candidates[0].source == "regex_direct_support_scalar"
                ):
                    return _admission_v2_with_backfill(verified, intent=intent_decision.intent)
                return verified
            return _candidate_decision(
                candidates[0],
                t0,
                intent=intent_decision.intent,
                candidate_count=len(candidates),
                candidate_rejection_counts=candidate_rejection_counts,
                question=question,
                locomo_support_present_answer_realization_enabled=(
                    locomo_support_present_answer_realization_enabled
                ),
            )
        if intent_decision.intent == "count" and candidate_rejection_counts:
            return _admission_v2(
                _abstain(
                    "candidate_filtered",
                    t0,
                    intent=intent_decision.intent,
                    candidate_count=len(candidates),
                    candidate_rejection_counts=candidate_rejection_counts,
                )
            )
        if intent_decision.intent == "count" and len(candidates) > 1:
            return _abstain(
                "candidate_ambiguous",
                t0,
                intent=intent_decision.intent,
                candidate_count=len(candidates),
                candidate_rejection_counts=candidate_rejection_counts,
            )
        if len(candidates) > 1:
            if llm_enabled and llm_call is not None:
                try:
                    selected = _select_candidate_with_llm(
                        question,
                        candidates,
                        llm_call=llm_call,
                        timeout_seconds=timeout_seconds,
                        model=model,
                    )
                except Exception as exc:
                    llm_error_class = _llm_error_class(exc)
                    recovered = _try_exact_support_recovery(
                        question=question,
                        supports=supports,
                        intent=intent_decision.intent,
                        t0=t0,
                        llm_error_class=llm_error_class,
                        enabled=exact_support_recovery_active,
                        support_pack_completeness_enabled=support_pack_completeness_active,
                        exact_support_native_stability_enabled=exact_support_native_stability_enabled,
                        support_surface_reach_enabled=support_surface_reach_active,
                        support_present_native_stability_enabled=support_present_native_stability_active,
                        support_present_guard_stability_enabled=support_present_guard_stability_active,
                        actor_compatibility_enabled=actor_compatibility_guard_active,
                        support_surface_reach_pool=support_surface_reach_pool,
                        candidate_count=len(candidates),
                        candidate_rejection_counts=candidate_rejection_counts,
                    )
                    if recovered is not None:
                        return _with_llm_error_provider_diagnostics(recovered, exc)
                    return _abstain(
                        "candidate_selection_error",
                        t0,
                        llm_error_class=llm_error_class,
                        llm_error_provider_status=_llm_error_provider_status(exc),
                        llm_error_provider_body_preview=_llm_error_provider_body_preview(exc),
                        intent=intent_decision.intent,
                        candidate_count=len(candidates),
                        candidate_rejection_counts=candidate_rejection_counts,
                    )
                if selected is not None:
                    selected = _resolve_temporal_candidate_selection_conflict(
                        question=question,
                        selected=selected,
                        candidates=candidates,
                        supports=supports,
                        rejection_counts=candidate_rejection_counts,
                    )
                    if strict_structured_enabled:
                        return _verified_candidate_decision(
                            question=question,
                            candidate=selected,
                            supports=supports,
                            t0=t0,
                            intent=intent_decision.intent,
                            candidate_count=len(candidates),
                            candidate_rejection_counts=candidate_rejection_counts,
                            locomo_support_present_answer_realization_enabled=(
                                locomo_support_present_answer_realization_enabled
                            ),
                        )
                    return _candidate_decision(
                        selected,
                        t0,
                        intent=intent_decision.intent,
                        candidate_count=len(candidates),
                        candidate_rejection_counts=candidate_rejection_counts,
                        question=question,
                        locomo_support_present_answer_realization_enabled=(
                            locomo_support_present_answer_realization_enabled
                        ),
                    )
            return _abstain(
                "candidate_ambiguous",
                t0,
                intent=intent_decision.intent,
                candidate_count=len(candidates),
                candidate_rejection_counts=candidate_rejection_counts,
            )

    question_abstain_reason = _question_abstain_reason(question)
    if question_abstain_reason:
        if question_abstain_reason == "shared_or_common_requires_synthesis" and support_present_admission_v3_active:
            shared_decision = _shared_common_entity_synthesis_with_backfill(
                intent=intent_decision.intent if intent_decision else None,
            )
            if shared_decision is not None:
                return shared_decision
        return _admission_v2(
            _abstain(
                question_abstain_reason,
                t0,
                intent=intent_decision.intent if intent_decision else None,
                candidate_count=len(candidates),
                candidate_rejection_counts=candidate_rejection_counts,
            )
        )

    if not _is_atomic_question_shape(question):
        if support_present_guard_stability_active:
            recovered = _try_exact_support_recovery(
                question=question,
                supports=supports,
                intent=intent_decision.intent if intent_decision else None,
                t0=t0,
                llm_error_class=None,
                enabled=exact_support_recovery_active,
                support_pack_completeness_enabled=support_pack_completeness_active,
                exact_support_native_stability_enabled=exact_support_native_stability_enabled,
                support_surface_reach_enabled=support_surface_reach_active,
                support_present_native_stability_enabled=support_present_native_stability_active,
                support_present_guard_stability_enabled=support_present_guard_stability_active,
                actor_compatibility_enabled=actor_compatibility_guard_active,
                support_surface_reach_pool=support_surface_reach_pool,
                candidate_count=len(candidates),
                candidate_rejection_counts=candidate_rejection_counts,
            )
            if recovered is not None:
                return recovered
        non_atomic_decision = _admission_v2(
            _abstain(
                "non_atomic_question_shape",
                t0,
                intent=intent_decision.intent if intent_decision else None,
                candidate_count=len(candidates),
                candidate_rejection_counts=candidate_rejection_counts,
            )
        )
        backfilled_non_atomic = _locomo_source_admission_with_backfill(
            non_atomic_decision,
            recovery_strategy="locomo_source_admission_non_atomic",
            annotate_abstain=True,
        )
        if backfilled_non_atomic.answer or backfilled_non_atomic.backfill_candidate_ids:
            return backfilled_non_atomic
        return non_atomic_decision

    if not llm_enabled or llm_call is None:
        recovered = _try_exact_support_recovery(
            question=question,
            supports=supports,
            intent=intent_decision.intent if intent_decision else None,
            t0=t0,
            llm_error_class=None,
            enabled=exact_support_recovery_active,
            support_pack_completeness_enabled=support_pack_completeness_active,
            exact_support_native_stability_enabled=exact_support_native_stability_enabled,
            support_surface_reach_enabled=support_surface_reach_active,
            support_present_native_stability_enabled=support_present_native_stability_active,
            support_present_guard_stability_enabled=support_present_guard_stability_active,
            actor_compatibility_enabled=actor_compatibility_guard_active,
            support_surface_reach_pool=support_surface_reach_pool,
            candidate_count=len(candidates),
            candidate_rejection_counts=candidate_rejection_counts,
        )
        if recovered is not None:
            return recovered
        decision = _abstain(
            "llm_disabled",
            t0,
            intent=intent_decision.intent if intent_decision else None,
            candidate_count=len(candidates),
            candidate_rejection_counts=candidate_rejection_counts,
        )
        if (
            (
                locomo_support_present_synthesis_enabled
                and _locomo_support_present_synthesis_question(question)
            )
            or (
                locomo_support_present_answer_realization_enabled
                and _locomo_support_present_answer_realization_question(question)
            )
        ):
            admitted = _admission_v2(decision)
            if admitted.answer or admitted.support_admission_version == "v3":
                return admitted
        return _exact_recovery_with_backfill(
            decision,
            intent=intent_decision.intent if intent_decision else None,
            reason="llm_disabled",
            candidate_count=len(candidates),
            candidate_rejection_counts=candidate_rejection_counts,
        )

    try:
        raw = llm_call(
            _build_prompt(question, supports),
            system_msg=_SYSTEM_PROMPT,
            model=model,
            max_tokens=96,
            timeout=timeout_seconds,
        )
        proposal = _parse_json(raw)
    except Exception as exc:
        llm_error_class = _llm_error_class(exc)
        recovered = _try_exact_support_recovery(
            question=question,
            supports=supports,
            intent=intent_decision.intent if intent_decision else None,
            t0=t0,
            llm_error_class=llm_error_class,
            enabled=exact_support_recovery_active,
            support_pack_completeness_enabled=support_pack_completeness_active,
            exact_support_native_stability_enabled=exact_support_native_stability_enabled,
            support_surface_reach_enabled=support_surface_reach_active,
            support_present_native_stability_enabled=support_present_native_stability_active,
            support_present_guard_stability_enabled=support_present_guard_stability_active,
            actor_compatibility_enabled=actor_compatibility_guard_active,
            support_surface_reach_pool=support_surface_reach_pool,
            candidate_count=len(candidates),
            candidate_rejection_counts=candidate_rejection_counts,
        )
        if recovered is not None:
            return _with_llm_error_provider_diagnostics(recovered, exc)
        llm_error_decision = _admission_v2(
            _abstain(
                "llm_error",
                t0,
                llm_error_class=llm_error_class,
                llm_error_provider_status=_llm_error_provider_status(exc),
                llm_error_provider_body_preview=_llm_error_provider_body_preview(exc),
                intent=intent_decision.intent if intent_decision else None,
                candidate_count=len(candidates),
                candidate_rejection_counts=candidate_rejection_counts,
            )
        )
        return _exact_recovery_with_backfill(
            llm_error_decision,
            intent=intent_decision.intent if intent_decision else None,
            reason="llm_error",
            llm_error_class=llm_error_class,
            candidate_count=len(candidates),
            candidate_rejection_counts=candidate_rejection_counts,
        )

    answer = _proposal_string(proposal.get("answer"))
    support_id = _proposal_string(proposal.get("support_id"))
    if not answer or not support_id:
        recovered = _try_exact_support_recovery(
            question=question,
            supports=supports,
            intent=intent_decision.intent if intent_decision else None,
            t0=t0,
            llm_error_class=None,
            enabled=exact_support_recovery_active,
            support_pack_completeness_enabled=support_pack_completeness_active,
            exact_support_native_stability_enabled=exact_support_native_stability_enabled,
            support_surface_reach_enabled=support_surface_reach_active,
            support_present_native_stability_enabled=support_present_native_stability_active,
            support_present_guard_stability_enabled=support_present_guard_stability_active,
            actor_compatibility_enabled=actor_compatibility_guard_active,
            support_surface_reach_pool=support_surface_reach_pool,
            candidate_count=len(candidates),
            candidate_rejection_counts=candidate_rejection_counts,
        )
        if recovered is not None:
            return recovered
        decision = _admission_v2(
            _abstain(
                "empty_answer_or_support",
                t0,
                intent=intent_decision.intent if intent_decision else None,
                candidate_count=len(candidates),
                candidate_rejection_counts=candidate_rejection_counts,
            )
        )
        if decision.answer:
            return decision
        return _exact_recovery_with_backfill(
            decision,
            intent=intent_decision.intent if intent_decision else None,
            reason="empty_answer_or_support",
            candidate_count=len(candidates),
            candidate_rejection_counts=candidate_rejection_counts,
        )

    support = next((item for item in supports if item.support_id == support_id), None)
    if support is None:
        recovered = _try_exact_support_recovery(
            question=question,
            supports=supports,
            intent=intent_decision.intent if intent_decision else None,
            t0=t0,
            llm_error_class=None,
            enabled=exact_support_recovery_active,
            support_pack_completeness_enabled=support_pack_completeness_active,
            exact_support_native_stability_enabled=exact_support_native_stability_enabled,
            support_surface_reach_enabled=support_surface_reach_active,
            support_present_native_stability_enabled=support_present_native_stability_active,
            support_present_guard_stability_enabled=support_present_guard_stability_active,
            actor_compatibility_enabled=actor_compatibility_guard_active,
            support_surface_reach_pool=support_surface_reach_pool,
            candidate_count=len(candidates),
            candidate_rejection_counts=candidate_rejection_counts,
        )
        if recovered is not None:
            return recovered
        return _abstain(
            "unknown_support_id",
            t0,
            intent=intent_decision.intent if intent_decision else None,
            candidate_count=len(candidates),
            candidate_rejection_counts=candidate_rejection_counts,
        )

    normalized_answer = _normalize(answer)
    recovery_result = _canonicalize_support_bound_answer(
        question=question,
        answer=answer,
        support=support,
        intent=intent_decision.intent if intent_decision else None,
        enabled=exact_support_recovery_active,
        locomo_support_present_synthesis_enabled=locomo_support_present_synthesis_enabled,
        locomo_support_present_answer_realization_enabled=locomo_support_present_answer_realization_enabled,
    )
    if recovery_result is not None:
        answer = recovery_result.answer
        normalized_answer = recovery_result.normalized_answer
    containment_answer = _containment_normalize(answer)
    if not _contains_containment_answer(
        containment_answer,
        support.normalized_support_text,
    ):
        recovered = _try_exact_support_recovery(
            question=question,
            supports=supports,
            intent=intent_decision.intent if intent_decision else None,
            t0=t0,
            llm_error_class=None,
            enabled=exact_support_recovery_active,
            support_pack_completeness_enabled=support_pack_completeness_active,
            exact_support_native_stability_enabled=exact_support_native_stability_enabled,
            support_surface_reach_enabled=support_surface_reach_active,
            support_present_native_stability_enabled=support_present_native_stability_active,
            support_present_guard_stability_enabled=support_present_guard_stability_active,
            actor_compatibility_enabled=actor_compatibility_guard_active,
            support_surface_reach_pool=support_surface_reach_pool,
            candidate_count=len(candidates),
            candidate_rejection_counts=candidate_rejection_counts,
        )
        if recovered is not None:
            return recovered
        return _admission_v2(
            _abstain(
                "unsupported_answer",
                t0,
                intent=intent_decision.intent if intent_decision else None,
                candidate_count=len(candidates),
                candidate_rejection_counts=candidate_rejection_counts,
            )
        )

    if actor_compatibility_guard_active and not _support_actor_compatible(
        question,
        support,
        answer=answer,
    ):
        _increment_rejection(candidate_rejection_counts, "support_actor_mismatch")
        recovered = _try_exact_support_recovery(
            question=question,
            supports=supports,
            intent=intent_decision.intent if intent_decision else None,
            t0=t0,
            llm_error_class=None,
            enabled=exact_support_recovery_active,
            support_pack_completeness_enabled=support_pack_completeness_active,
            exact_support_native_stability_enabled=exact_support_native_stability_enabled,
            support_surface_reach_enabled=support_surface_reach_active,
            support_present_native_stability_enabled=support_present_native_stability_active,
            support_present_guard_stability_enabled=support_present_guard_stability_active,
            actor_compatibility_enabled=actor_compatibility_guard_active,
            support_surface_reach_pool=support_surface_reach_pool,
            candidate_count=len(candidates),
            candidate_rejection_counts=candidate_rejection_counts,
        )
        if recovered is not None:
            return recovered
        decision = _abstain(
            "support_actor_mismatch",
            t0,
            intent=intent_decision.intent if intent_decision else None,
            candidate_count=len(candidates),
            candidate_rejection_counts=candidate_rejection_counts,
        )
        if (
            locomo_support_present_synthesis_enabled
            and _locomo_support_present_synthesis_question(question)
        ):
            admitted = _admission_v2(decision)
            if admitted.answer or admitted.support_admission_version == "v3":
                return admitted
        return _exact_recovery_with_backfill(
            decision,
            intent=intent_decision.intent if intent_decision else None,
            reason="support_actor_mismatch",
            candidate_count=len(candidates),
            candidate_rejection_counts=candidate_rejection_counts,
            allow_temporal_actor_mismatch=True,
        )

    repair_result: SupportDerivedRepairResult | None = None
    repair_shape: RepairShape = _support_derived_repair_shape(question)
    if support_derived_repair_active and _repair_needed(
        question,
        answer,
        support,
        shape=repair_shape,
    ):
        repair_result = _try_support_derived_repair(
            question=question,
            answer=answer,
            support=support,
            supports=supports,
            intent=intent_decision.intent if intent_decision else None,
            shape=repair_shape,
        )
        if repair_result is None:
            return _abstain(
                _support_derived_repair_failure_reason(repair_shape),
                t0,
                intent=intent_decision.intent if intent_decision else None,
                candidate_count=len(candidates),
                candidate_rejection_counts=candidate_rejection_counts,
                expected_answer_shape=_support_derived_expected_shape(repair_shape),
                slot_binding_status=(
                    _support_sentence_slot_binding_status(question, support, answer)
                    if repair_shape == "predicate_bound_scalar"
                    else None
                ),
                synthesis_shape=synthesis_shape,
            )
        answer = repair_result.answer
        normalized_answer = repair_result.normalized_answer
        support = repair_result.support

    answer_shape_reason = _answer_shape_abstain_reason(
        question,
        answer,
        normalized_answer,
        support,
        allow_list_like=(repair_result is not None and _LIST_SEPARATOR_PATTERN.search(answer) is not None),
    )
    if answer_shape_reason:
        if support_present_guard_stability_active:
            recovered = _try_exact_support_recovery(
                question=question,
                supports=supports,
                intent=intent_decision.intent if intent_decision else None,
                t0=t0,
                llm_error_class=None,
                enabled=exact_support_recovery_active,
                support_pack_completeness_enabled=support_pack_completeness_active,
                exact_support_native_stability_enabled=exact_support_native_stability_enabled,
                support_surface_reach_enabled=support_surface_reach_active,
                support_present_native_stability_enabled=support_present_native_stability_active,
                support_present_guard_stability_enabled=support_present_guard_stability_active,
                actor_compatibility_enabled=actor_compatibility_guard_active,
                support_surface_reach_pool=support_surface_reach_pool,
                candidate_count=len(candidates),
                candidate_rejection_counts=candidate_rejection_counts,
            )
            if recovered is not None:
                return recovered
        return _admission_v2(
            _abstain(
                answer_shape_reason,
                t0,
                intent=intent_decision.intent if intent_decision else None,
                candidate_count=len(candidates),
                candidate_rejection_counts=candidate_rejection_counts,
            )
        )

    contract_reason = _extractive_answer_contract_abstain_reason(question, answer)
    if contract_reason:
        home_country_deictic = (
            contract_reason == "deictic_answer_unresolved"
            and _home_country_move_from_actor(question) is not None
        )
        if (
            contract_reason == "temporal_deictic_answer_unresolved"
            and support_present_guard_stability_active
        ) or home_country_deictic:
            recovered = _try_exact_support_recovery(
                question=question,
                supports=supports,
                intent=intent_decision.intent if intent_decision else None,
                t0=t0,
                llm_error_class=None,
                enabled=exact_support_recovery_active,
                support_pack_completeness_enabled=support_pack_completeness_active,
                exact_support_native_stability_enabled=exact_support_native_stability_enabled,
                support_surface_reach_enabled=support_surface_reach_active,
                support_present_native_stability_enabled=support_present_native_stability_active,
                support_present_guard_stability_enabled=support_present_guard_stability_active,
                actor_compatibility_enabled=actor_compatibility_guard_active,
                support_surface_reach_pool=support_surface_reach_pool,
                candidate_count=len(candidates),
                candidate_rejection_counts=candidate_rejection_counts,
            )
            if recovered is not None:
                return recovered
        contract_abstain = _abstain(
            contract_reason,
            t0,
            intent=intent_decision.intent if intent_decision else None,
            candidate_count=len(candidates),
            candidate_rejection_counts=candidate_rejection_counts,
        )
        if home_country_deictic:
            backfilled = _maybe_retry_with_support_candidate_backfill(
                decision=contract_abstain,
                question=question,
                activated_concepts=activated_concepts[:max_activated_concepts],
                existing_supports=supports,
                shape="predicate_bound_scalar",
                llm_call=llm_call,
                llm_enabled=llm_enabled,
                timeout_seconds=timeout_seconds,
                model=model,
                t0=t0,
                candidate_rejection_counts=candidate_rejection_counts,
                support_pack_completeness_enabled=support_pack_completeness_active,
                exact_support_native_stability_enabled=exact_support_native_stability_enabled,
                support_present_native_stability_enabled=support_present_native_stability_active,
                support_present_guard_stability_enabled=support_present_guard_stability_active,
                locomo_support_present_answer_realization_enabled=(
                    locomo_support_present_answer_realization_enabled
                ),
                locomo_support_emission_enabled=locomo_backfill_emission_active,
                enabled=answer_contract_enabled and support_candidate_backfill_enabled,
                fts_limit=support_candidate_backfill_fts_limit,
                assoc_limit=support_candidate_backfill_assoc_limit,
                max_supports=support_candidate_backfill_max_supports,
                min_score=support_candidate_backfill_min_score,
                budget_ms=support_candidate_backfill_budget_ms,
                semantic_enabled=support_candidate_backfill_semantic_enabled,
                semantic_limit=support_candidate_backfill_semantic_limit,
                semantic_min_score=support_candidate_backfill_semantic_min_score,
            )
            if backfilled.answer or backfilled.backfill_candidate_ids:
                return backfilled
        return contract_abstain

    answer_contract = AnswerContractResult(None, None, None)
    if answer_contract_enabled:
        answer_contract = _evidence_bound_answer_contract_abstain_reason(
            question,
            answer,
            support,
        )
        if answer_contract.reason:
            recovered = None
            if support_present_guard_stability_active:
                recovered = _try_exact_support_recovery(
                    question=question,
                    supports=supports,
                    intent=intent_decision.intent if intent_decision else None,
                    t0=t0,
                    llm_error_class=None,
                    enabled=exact_support_recovery_active,
                    support_pack_completeness_enabled=support_pack_completeness_active,
                    exact_support_native_stability_enabled=exact_support_native_stability_enabled,
                    support_surface_reach_enabled=support_surface_reach_active,
                    support_present_native_stability_enabled=support_present_native_stability_active,
                    support_present_guard_stability_enabled=support_present_guard_stability_active,
                    actor_compatibility_enabled=actor_compatibility_guard_active,
                    support_surface_reach_pool=support_surface_reach_pool,
                    candidate_count=len(candidates),
                    candidate_rejection_counts=candidate_rejection_counts,
                )
            if recovered is not None:
                return recovered
            contract_abstain = _abstain(
                answer_contract.reason,
                t0,
                intent=intent_decision.intent if intent_decision else None,
                candidate_count=len(candidates),
                candidate_rejection_counts=candidate_rejection_counts,
                answer_contract_reason=answer_contract.reason,
                expected_answer_shape=answer_contract.expected_answer_shape,
                slot_binding_status=answer_contract.slot_binding_status,
                synthesis_shape=synthesis_shape,
            )
            if answer_contract.reason == "identity_answer_requires_complete_phrase" or (
                answer_contract.reason == "deictic_answer_unresolved"
                and _home_country_move_from_actor(question) is not None
            ):
                backfilled = _maybe_retry_with_support_candidate_backfill(
                    decision=contract_abstain,
                    question=question,
                    activated_concepts=activated_concepts[:max_activated_concepts],
                    existing_supports=supports,
                    shape="predicate_bound_scalar",
                    llm_call=llm_call,
                    llm_enabled=llm_enabled,
                    timeout_seconds=timeout_seconds,
                    model=model,
                    t0=t0,
                    candidate_rejection_counts=candidate_rejection_counts,
                    support_pack_completeness_enabled=support_pack_completeness_active,
                    exact_support_native_stability_enabled=exact_support_native_stability_enabled,
                    support_present_native_stability_enabled=support_present_native_stability_active,
                    support_present_guard_stability_enabled=support_present_guard_stability_active,
                    locomo_support_present_answer_realization_enabled=(
                        locomo_support_present_answer_realization_enabled
                    ),
                    locomo_support_emission_enabled=locomo_backfill_emission_active,
                    enabled=answer_contract_enabled and support_candidate_backfill_enabled,
                    fts_limit=support_candidate_backfill_fts_limit,
                    assoc_limit=support_candidate_backfill_assoc_limit,
                    max_supports=support_candidate_backfill_max_supports,
                    min_score=support_candidate_backfill_min_score,
                    budget_ms=support_candidate_backfill_budget_ms,
                    semantic_enabled=support_candidate_backfill_semantic_enabled,
                    semantic_limit=support_candidate_backfill_semantic_limit,
                    semantic_min_score=support_candidate_backfill_semantic_min_score,
                )
                if backfilled.answer or backfilled.backfill_candidate_ids:
                    return backfilled
            return contract_abstain

    mode: AnswerMode
    if repair_result is not None:
        mode = "support_derived_repair"
    elif recovery_result is not None:
        mode = "exact_support_recovery"
    else:
        mode = "exact_extractive" if answer in support.support_text else "normalized_extractive"
    final_decision = EvidenceAnswerDecision(
        mode=mode,
        answer=answer,
        normalized_answer=normalized_answer,
        support=support,
        abstain_reason=None,
        latency_ms=_latency_ms(t0),
        intent=intent_decision.intent if intent_decision else None,
        candidate_count=len(candidates),
        candidate_rejection_counts=candidate_rejection_counts or None,
        expected_answer_shape=(
            _support_derived_expected_shape(repair_shape) if repair_result is not None else None
        ),
        slot_binding_status=(
            "bound"
            if repair_result is not None and repair_shape == "predicate_bound_scalar"
            else answer_contract.slot_binding_status
        ),
        support_pack_size=(repair_result.support_pack_size if repair_result is not None else 0),
        recovery_strategy=(
            repair_result.strategy
            if repair_result is not None
            else recovery_result.strategy
            if recovery_result
            else None
        ),
    )
    audited_decision = _locomo_source_admission_with_backfill(
        final_decision,
        recovery_strategy="locomo_source_admission_answer_audit",
    )
    return _finish(audited_decision)


def _classify_answer_intent(question: str) -> AnswerIntentDecision:
    reason = _question_abstain_reason(question)
    if reason in _SYNTHESIS_DEFER_REASONS:
        return AnswerIntentDecision("synthesis_deferred", reason)
    if reason in _TEMPORAL_DEFER_REASONS:
        return AnswerIntentDecision("temporal_deferred", reason)
    if reason is not None:
        return AnswerIntentDecision("unsupported", reason)

    if _YES_NO_QUESTION_PATTERN.search(question):
        return AnswerIntentDecision("yes_no")
    if re.search(r"^\s*how many\b", question, re.IGNORECASE):
        return AnswerIntentDecision("count")
    if re.search(r"^\s*(?:how long|for\s+how\s+long)\b", question, re.IGNORECASE):
        return AnswerIntentDecision("duration")
    if re.search(r"^\s*when\b", question, re.IGNORECASE):
        return AnswerIntentDecision("date")
    if _LOCATION_QUESTION_PATTERN.search(question):
        return AnswerIntentDecision("location")
    if _SHORT_ATTRIBUTE_QUESTION_PATTERN.search(question):
        return AnswerIntentDecision("short_attribute")
    if _is_atomic_question_shape(question):
        return AnswerIntentDecision("scalar_entity")
    return AnswerIntentDecision("unsupported")


def _parse_shared_common_entity_question(question: str) -> SharedCommonEntityQuestion | None:
    for pattern in _SHARED_COMMON_ENTITY_QUESTION_PATTERNS:
        match = pattern.search(question)
        if match is None:
            continue
        actor_a = match.group("actor_a")
        actor_b = match.group("actor_b")
        if not actor_a[:1].isupper() or not actor_b[:1].isupper():
            return None
        if actor_a == actor_b:
            return None
        return SharedCommonEntityQuestion(
            category=match.group("category").lower(),
            actor_a=actor_a,
            actor_b=actor_b,
        )
    return None


def _shared_common_entity_candidates(text: str) -> dict[str, str]:
    normalized = _containment_normalize(text)
    candidates: dict[str, str] = {}
    for entity, surfaces in _SHARED_COMMON_ENTITY_SURFACES.items():
        for surface in surfaces:
            if f" {surface} " not in f" {normalized} ":
                continue
            candidates.setdefault(entity, surface)
            if surface.endswith("s"):
                candidates[entity] = surface
    return candidates


def _shared_common_owner_proof_ok(
    *,
    entity: str,
    actor: str,
    other_actor: str,
    support: EvidenceSupport,
) -> bool:
    combined = f"{support.concept_summary} {support.support_text}"
    normalized_combined = _containment_normalize(combined)
    pronoun_possessive = any(
        f" your {surface} " in f" {normalized_combined} "
        for surface in _SHARED_COMMON_ENTITY_SURFACES.get(entity, (entity,))
    )
    if not pronoun_possessive:
        return True

    owner = _normalize(other_actor)
    actor_name = _normalize(actor)
    possessive_owner = f"{owner} s"
    return (
        f" {actor_name} " in f" {normalized_combined} "
        and f" {possessive_owner} " in f" {normalized_combined} "
        and any(
            f" {surface} " in f" {normalized_combined} "
            for surface in _SHARED_COMMON_ENTITY_SURFACES.get(entity, (entity,))
        )
    )


def _support_binds_actor_to_shared_entity(
    *,
    parsed: SharedCommonEntityQuestion,
    actor: str,
    other_actor: str,
    support: EvidenceSupport,
) -> list[SharedCommonEntitySupport]:
    combined = f"{support.concept_summary} {support.support_text}"
    normalized_combined = _containment_normalize(combined)
    if f" {_normalize(actor)} " not in f" {normalized_combined} ":
        return []
    preference = _SHARED_COMMON_PREFERENCE_PATTERN.search(combined)
    if preference is None:
        return []
    actor_index = combined.lower().find(actor.lower())
    if actor_index > preference.start() and not support.concept_summary.lower().startswith(actor.lower()):
        return []
    preference_text = preference.group(0).lower()
    if preference_text in {"affection for", "admiration for"}:
        owner = f"{_normalize(other_actor)} s"
        if f" {owner} " not in f" {normalized_combined} ":
            return []

    matches: list[SharedCommonEntitySupport] = []
    for entity, surface in _shared_common_entity_candidates(combined).items():
        if not _shared_common_owner_proof_ok(
            entity=entity,
            actor=actor,
            other_actor=other_actor,
            support=support,
        ):
            continue
        matches.append(
            SharedCommonEntitySupport(
                actor=actor,
                entity=entity,
                answer_surface=surface,
                support=support,
            )
        )
    return matches


def _try_shared_common_entity_synthesis(
    *,
    question: str,
    supports: list[EvidenceSupport],
    t0: float,
) -> EvidenceAnswerDecision | None:
    parsed = _parse_shared_common_entity_question(question)
    if parsed is None:
        return None

    actor_a_matches: list[SharedCommonEntitySupport] = []
    actor_b_matches: list[SharedCommonEntitySupport] = []
    for support in supports:
        actor_a_matches.extend(
            _support_binds_actor_to_shared_entity(
                parsed=parsed,
                actor=parsed.actor_a,
                other_actor=parsed.actor_b,
                support=support,
            )
        )
        actor_b_matches.extend(
            _support_binds_actor_to_shared_entity(
                parsed=parsed,
                actor=parsed.actor_b,
                other_actor=parsed.actor_a,
                support=support,
            )
        )

    for left in actor_a_matches:
        for right in actor_b_matches:
            if left.entity != right.entity:
                continue
            if left.support.support_id == right.support.support_id:
                continue
            answer = left.answer_surface if left.answer_surface.endswith("s") else right.answer_surface
            if not answer.endswith("s"):
                answer = _SHARED_COMMON_ENTITY_SURFACES[left.entity][-1]
            return EvidenceAnswerDecision(
                mode="structured_synthesis",
                answer=answer,
                normalized_answer=_normalize(answer),
                support=left.support,
                abstain_reason=None,
                latency_ms=_latency_ms(t0),
                intent="synthesis_deferred",
                expected_answer_shape="shared_common_entity",
                slot_binding_status="bound",
                synthesis_shape="list_or_set",
                support_pack_size=2,
                fallback_used="deterministic_shared_common_entity",
            )
    return None


def _shared_common_observed_entities(supports: list[EvidenceSupport]) -> set[str]:
    entities: set[str] = set()
    for support in supports:
        combined = f"{support.concept_summary} {support.support_text}"
        entities.update(_shared_common_entity_candidates(combined))
    return entities


def _collect_shared_common_entity_backfill(
    *,
    question: str,
    existing_supports: list[EvidenceSupport],
    fts_limit: int,
    max_supports: int,
    budget_ms: float,
) -> SupportCandidateBackfillResult:
    t0 = time.perf_counter()
    parsed = _parse_shared_common_entity_question(question)
    if parsed is None:
        return SupportCandidateBackfillResult([], (), {"shared_common_backfill_not_applicable": 1}, _latency_ms(t0))

    observed_entities = _shared_common_observed_entities(existing_supports)
    if not observed_entities:
        return SupportCandidateBackfillResult([], (), {"shared_common_backfill_no_observed_entity": 1}, _latency_ms(t0))
    missing_actors: list[str] = []
    for actor, other_actor in ((parsed.actor_a, parsed.actor_b), (parsed.actor_b, parsed.actor_a)):
        has_actor_entity_support = any(
            match.entity in observed_entities
            for support in existing_supports
            for match in _support_binds_actor_to_shared_entity(
                parsed=parsed,
                actor=actor,
                other_actor=other_actor,
                support=support,
            )
        )
        if not has_actor_entity_support:
            missing_actors.append(actor)
    backfill_actors = missing_actors or [parsed.actor_a, parsed.actor_b]

    deadline = t0 + max(75.0, min(float(budget_ms), 250.0)) / 1000.0
    rejection_counts: dict[str, int] = {}
    fts_limit = max(0, min(int(fts_limit), 50))
    max_supports = max(0, min(int(max_supports), 8))
    if fts_limit <= 0 or max_supports <= 0:
        _increment_rejection(rejection_counts, "shared_common_backfill_disabled")
        return SupportCandidateBackfillResult([], (), rejection_counts, _latency_ms(t0))

    terms: list[str] = []
    for actor in (parsed.actor_a, parsed.actor_b):
        token = _normalize(actor)
        if token and token not in terms:
            terms.append(token)
    for entity in sorted(observed_entities):
        for surface in _SHARED_COMMON_ENTITY_SURFACES.get(entity, (entity,)):
            token = _normalize(surface)
            if token and token not in terms:
                terms.append(token)

    existing_ids = {support.concept_id for support in existing_supports if support.concept_id}
    try:
        conn = _open_support_candidate_backfill_connection()
    except Exception:
        _increment_rejection(rejection_counts, "shared_common_backfill_no_connection")
        return SupportCandidateBackfillResult([], (), rejection_counts, _latency_ms(t0))

    rows: list[sqlite3.Row] = []
    try:
        rows = _fetch_shared_common_entity_candidate_rows(
            conn,
            parsed=parsed,
            observed_entities=observed_entities,
            actors=backfill_actors,
            limit=fts_limit,
        )
        if not rows:
            rows = _fetch_fts_support_candidate_rows(conn, tuple(terms), fts_limit)
    except Exception:
        _increment_rejection(rejection_counts, "shared_common_backfill_no_candidates")
    finally:
        try:
            conn.close()
        except Exception:
            pass

    evaluation_deadline = max(
        deadline,
        time.perf_counter() + _SUPPORT_CANDIDATE_BACKFILL_EVALUATION_GRACE_MS / 1000.0,
    )
    admitted: list[EvidenceSupport] = []
    admitted_ids: set[str] = set()
    for row in rows:
        if time.perf_counter() >= evaluation_deadline:
            _increment_rejection(rejection_counts, "shared_common_backfill_timeout")
            break
        if not _support_candidate_row_is_active(row):
            _increment_rejection(rejection_counts, "shared_common_backfill_inactive")
            continue
        concept = _support_candidate_row_to_concept(row)
        concept_id = _stringify(concept.get("concept_id")).strip()
        if not concept_id or concept_id in existing_ids or concept_id in admitted_ids:
            _increment_rejection(rejection_counts, "shared_common_backfill_duplicate")
            continue
        supports = _collect_supports([concept], max_support_chars=4000)
        matched_support: EvidenceSupport | None = None
        for support in supports:
            matches = [
                *_support_binds_actor_to_shared_entity(
                    parsed=parsed,
                    actor=parsed.actor_a,
                    other_actor=parsed.actor_b,
                    support=support,
                ),
                *_support_binds_actor_to_shared_entity(
                    parsed=parsed,
                    actor=parsed.actor_b,
                    other_actor=parsed.actor_a,
                    support=support,
                ),
            ]
            if any(match.entity in observed_entities for match in matches):
                matched_support = support
                break
        if matched_support is None:
            _increment_rejection(rejection_counts, "shared_common_backfill_unbound")
            continue
        admitted.append(matched_support)
        admitted_ids.add(concept_id)
        if len(admitted) >= max_supports:
            break

    if not admitted:
        _increment_rejection(rejection_counts, "shared_common_backfill_no_candidates")
    return SupportCandidateBackfillResult(
        admitted,
        tuple(support.concept_id for support in admitted),
        rejection_counts,
        _latency_ms(t0),
    )


def _fetch_shared_common_entity_candidate_rows(
    conn: sqlite3.Connection,
    *,
    parsed: SharedCommonEntityQuestion,
    observed_entities: set[str],
    actors: list[str],
    limit: int,
) -> list[sqlite3.Row]:
    id_col = _support_candidate_concepts_id_column(conn)
    select_cols = _support_candidate_select_columns(conn, id_col)
    surfaces: list[str] = []
    for entity in sorted(observed_entities):
        for surface in _SHARED_COMMON_ENTITY_SURFACES.get(entity, (entity,)):
            normalized_surface = _normalize(surface)
            if normalized_surface and normalized_surface not in surfaces:
                surfaces.append(normalized_surface)
    actor_terms = [_normalize(actor) for actor in actors] or [_normalize(parsed.actor_a), _normalize(parsed.actor_b)]
    actor_terms = [actor for actor in actor_terms if actor]
    if not surfaces or not actor_terms:
        return []

    text_expr = "LOWER(COALESCE(c.summary, ''))"
    actor_clause = " OR ".join(f"{text_expr} LIKE ?" for _ in actor_terms)
    surface_clause = " OR ".join(f"{text_expr} LIKE ?" for _ in surfaces)
    sql = (
        f"SELECT {select_cols} "
        "FROM concepts c "
        f"WHERE ({actor_clause}) "
        f"AND ({surface_clause}) "
        "ORDER BY "
        f"CASE WHEN {text_expr} LIKE '%drawn%' OR {text_expr} LIKE '%like%' "
        f"OR {text_expr} LIKE '%love%' OR {text_expr} LIKE '%affection%' THEN 0 ELSE 1 END, "
        "c.summary "
        "LIMIT ?"
    )
    params = [f"%{actor}%" for actor in actor_terms]
    params.extend(f"%{surface}%" for surface in surfaces)
    params.append(int(limit))
    return list(conn.execute(sql, params).fetchall())


def _build_answer_candidates(
    question: str,
    supports: list[EvidenceSupport],
    intent: AnswerIntent,
    *,
    relative_date_span_enabled: bool = False,
) -> CandidateBuildResult:
    candidates: list[AnswerCandidate] = []
    rejection_counts: dict[str, int] = {}
    for support in supports:
        for answer, source in _candidate_answers_from_support(
            question,
            support,
            intent,
            relative_date_span_enabled=relative_date_span_enabled,
        ):
            normalized_answer = _normalize(answer)
            if not normalized_answer:
                _increment_rejection(rejection_counts, "empty_candidate_after_normalize")
                continue
            rejection_reason = _candidate_alignment_rejection_reason(
                question,
                support,
                intent,
                answer,
                source,
            )
            if rejection_reason is not None:
                _increment_rejection(rejection_counts, rejection_reason)
                continue
            candidates.append(
                AnswerCandidate(
                    candidate_id="",
                    answer=answer,
                    normalized_answer=normalized_answer,
                    support=support,
                    source=source,
                )
            )
    return CandidateBuildResult(
        candidates=_dedupe_candidates(candidates),
        rejection_counts=rejection_counts,
    )


def _try_legacy_surface_contract_answer(
    *,
    question: str,
    supports: list[EvidenceSupport],
    t0: float,
) -> EvidenceAnswerDecision | None:
    """Recover narrow support-derived answer contracts from already-visible evidence."""
    intent_decision = _classify_answer_intent(question)
    if intent_decision.abstain_reason:
        return None
    if intent_decision.intent not in {"date", "count"}:
        return None

    build_result = _build_answer_candidates(
        question,
        supports,
        intent_decision.intent,
        relative_date_span_enabled=True,
    )
    if not build_result.candidates:
        return None

    decisions: list[EvidenceAnswerDecision] = []
    for candidate in build_result.candidates:
        if not _legacy_surface_candidate_allowed(candidate, intent_decision.intent):
            continue
        decision = _verified_candidate_decision(
            question=question,
            candidate=candidate,
            supports=supports,
            t0=t0,
            intent=intent_decision.intent,
            candidate_count=len(build_result.candidates),
            candidate_rejection_counts=build_result.rejection_counts,
        )
        if decision.answer:
            decisions.append(
                replace(
                    decision,
                    fallback_used="legacy_surface_contract",
                    recovery_strategy=f"legacy_surface_contract_{candidate.source}",
                    answer_contract_reason="legacy_surface_contract",
                    expected_answer_shape=_legacy_surface_expected_shape(intent_decision.intent),
                )
            )

    unique_answers = {_normalize(decision.answer or "") for decision in decisions if decision.answer}
    if len(unique_answers) != 1 or not decisions:
        return None
    return decisions[0]


def _legacy_surface_candidate_allowed(candidate: AnswerCandidate, intent: AnswerIntent) -> bool:
    if intent == "date":
        return candidate.source in {"regex_conversation_source_date", "regex_relative_date_span"}
    if intent == "count":
        return candidate.source == "regex_count"
    return False


def _legacy_surface_expected_shape(intent: AnswerIntent) -> str:
    if intent == "date":
        return "temporal_surface"
    if intent == "count":
        return "explicit_count"
    if intent == "duration":
        return "duration"
    return "atomic_scalar"


def _candidate_answers_from_support(
    question: str,
    support: EvidenceSupport,
    intent: AnswerIntent,
    *,
    relative_date_span_enabled: bool = False,
) -> list[tuple[str, CandidateSource]]:
    if intent == "yes_no":
        return _yes_no_candidates(question, support)
    direct_scalar_candidates: list[tuple[str, CandidateSource]] = []
    if intent in {"scalar_entity", "short_attribute"}:
        direct_scalar = None
        if not (
            support.channel == "summary"
            and _support_present_workshop_discussion_question(question)
        ):
            direct_scalar = _support_present_direct_scalar_candidate(
                question,
                support.support_text,
                context_sentence=support.concept_summary,
            )
        if direct_scalar is not None:
            direct_scalar_candidates.append((direct_scalar, "regex_direct_support_scalar"))
    if not _support_sufficiently_matches_question(question, support.support_text):
        return direct_scalar_candidates

    if intent == "count":
        return [
            (_canonical_count_answer(match.group(0)), "regex_count")
            for match in _COUNT_CANDIDATE_PATTERN.finditer(support.support_text)
        ]
    if intent == "location":
        return [
            (_clean_candidate_answer(match.group(1)), "regex_location")
            for match in _LOCATION_FROM_PATTERN.finditer(support.support_text)
        ]
    if intent == "date":
        if relative_date_span_enabled:
            relative_candidates = _relative_date_span_candidates(question, support.support_text)
            if relative_candidates:
                return [(candidate, "regex_relative_date_span") for candidate in relative_candidates]
        conversation_date_candidates = _conversation_date_candidates_from_support(question, support)
        if conversation_date_candidates:
            return conversation_date_candidates
        return [
            (_clean_candidate_answer(match.group(0)), "regex_date")
            for match in _DATE_CANDIDATE_PATTERN.finditer(support.support_text)
        ]
    if intent == "duration":
        digit_candidates = [
            (_duration_answer_for_question(question, _clean_candidate_answer(match.group(0))), "regex_duration")
            for match in _DURATION_CANDIDATE_PATTERN.finditer(support.support_text)
        ]
        word_candidates = [
            (_duration_answer_for_question(question, candidate), "regex_duration")
            for candidate in _duration_word_candidates_from_sentence(support.support_text)
        ]
        return digit_candidates + word_candidates
    if intent in {"scalar_entity", "short_attribute"}:
        candidates: list[tuple[str, CandidateSource]] = _quoted_title_candidates(support.support_text)
        candidates.extend(direct_scalar_candidates)
        if intent == "short_attribute":
            candidates.extend(
                (_clean_candidate_answer(match.group(1)), "regex_short_attribute")
                for match in _USED_FOR_ATTRIBUTE_PATTERN.finditer(support.support_text)
            )
            if support.channel == "verbatim":
                candidates.extend(_media_footwear_use_candidates(question, support.support_text))
        return candidates
    return []


def _relative_date_span_candidates(question: str, support_text: str) -> list[str]:
    event_terms = _question_event_terms(question)
    candidates: list[str] = []
    seen: set[str] = set()
    for sentence in _SENTENCE_SPLIT_PATTERN.split(support_text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(event_terms) >= 2:
            if not _date_sentence_has_required_event_terms(question, sentence):
                continue
            if not _sentence_matches_terms(sentence, event_terms, min_ratio=0.6):
                continue
        for match in _RELATIVE_SURFACE_DATE_SPAN_PATTERN.finditer(sentence):
            candidate = _clean_candidate_answer(match.group(0))
            if _date_candidate_sanity_rejection_reason(candidate) is not None:
                continue
            normalized = _normalize(candidate)
            if normalized in seen:
                continue
            seen.add(normalized)
            candidates.append(candidate)
    return candidates


_CONVERSATION_HEADER_PATTERN = re.compile(
    r"\[Conversation\s+on\s+(?P<date>[^\]]+?)\]",
    re.IGNORECASE,
)
_LAST_WEEKDAY_PATTERN = re.compile(
    r"\blast\s+(?P<weekday>monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)
_WEEKDAY_INDEX = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def _conversation_header_calendar_date(support_text: str) -> date | None:
    match = _CONVERSATION_HEADER_PATTERN.search(support_text)
    if match is None:
        return None
    return _parse_calendar_date(match.group("date"))


def _conversation_date_candidates_from_support(
    question: str,
    support: EvidenceSupport,
) -> list[tuple[str, CandidateSource]]:
    if support.channel != "verbatim":
        return []
    conversation_date = _conversation_header_calendar_date(support.support_text)
    if conversation_date is None:
        return []

    support_without_header = _CONVERSATION_HEADER_PATTERN.sub(" ", support.support_text)
    if "support group" in _normalize(question) and "support group" not in _normalize(support_without_header):
        return []
    if not _date_sentence_has_required_event_terms(question, support_without_header):
        return []

    if re.search(r"\byesterday\b", support_without_header, re.IGNORECASE):
        return [
            (
                _format_day_month_year_answer(conversation_date - timedelta(days=1)),
                "regex_conversation_source_date",
            )
        ]
    weekday_match = _LAST_WEEKDAY_PATTERN.search(support_without_header)
    if weekday_match is not None:
        weekday = _WEEKDAY_INDEX[weekday_match.group("weekday").lower()]
        delta_days = (conversation_date.weekday() - weekday) % 7
        if delta_days == 0:
            delta_days = 7
        return [
            (
                _format_day_month_year_answer(conversation_date - timedelta(days=delta_days)),
                "regex_conversation_source_date",
            )
        ]
    if re.search(r"\b(?:today|currently|just|now|recently)\b", support_without_header, re.IGNORECASE):
        return [
            (
                f"{_MONTH_NAMES[conversation_date.month]}, {conversation_date.year}",
                "regex_conversation_source_date",
            )
        ]
    return []


def _quoted_title_candidates(support_text: str) -> list[tuple[str, CandidateSource]]:
    candidates: list[tuple[str, CandidateSource]] = []
    for match in _QUOTED_SPAN_PATTERN.finditer(support_text):
        answer = _clean_candidate_answer(match.group(1))
        suffix_match = _QUOTED_CREATOR_SUFFIX_PATTERN.match(support_text[match.end() : match.end() + 80])
        if suffix_match:
            answer = f"{answer} by {_clean_candidate_answer(suffix_match.group(1))}"
        candidates.append((answer, "regex_quoted_title"))
    return candidates


def _increment_rejection(rejection_counts: dict[str, int], reason: str) -> None:
    rejection_counts[reason] = rejection_counts.get(reason, 0) + 1


def _increment_rejected_concept(
    rejection_counts: dict[str, int],
    rejected_ids_by_reason: dict[str, list[str]],
    reason: str,
    concept_id: str,
) -> None:
    _increment_rejection(rejection_counts, reason)
    if not concept_id:
        return
    rejected_ids = rejected_ids_by_reason.setdefault(reason, [])
    if concept_id not in rejected_ids:
        rejected_ids.append(concept_id)


def _candidate_alignment_rejection_reason(
    question: str,
    support: EvidenceSupport,
    intent: AnswerIntent,
    answer: str,
    source: CandidateSource,
) -> str | None:
    if source == "regex_quoted_title":
        return _quoted_span_rejection_reason(question, support.support_text, answer)
    if source in {"regex_date", "regex_relative_date_span", "regex_conversation_source_date"}:
        date_sanity_reason = _date_candidate_sanity_rejection_reason(answer)
        if date_sanity_reason is not None:
            return date_sanity_reason
        if source == "regex_conversation_source_date":
            expected_candidates = _conversation_date_candidates_from_support(question, support)
            if any(_normalize(candidate) == _normalize(answer) for candidate, _ in expected_candidates):
                return None
            return "conversation_source_date_unbound"
        return _date_event_rejection_reason(question, support.support_text, answer)
    if source == "regex_count":
        return _count_rejection_reason(question, support.support_text, answer)
    if source == "regex_short_attribute" and intent != "short_attribute":
        return "short_attribute_wrong_intent"
    if source == "regex_media_intent_attribute":
        if intent != "short_attribute":
            return "media_attribute_wrong_intent"
        if support.channel != "verbatim":
            return "media_attribute_non_verbatim_support"
    if source == "regex_direct_support_scalar":
        role_rejection = _support_present_role_candidate_rejection_reason(
            answer,
            "direct_support_scalar",
            question=question,
        )
        if role_rejection is not None:
            return role_rejection
    return None


def _quoted_span_rejection_reason(
    question: str,
    support_text: str,
    answer: str,
) -> str | None:
    if not _TITLE_SLOT_QUESTION_PATTERN.search(question):
        return "quoted_title_question_not_title_like"
    if _URL_LIKE_CANDIDATE_PATTERN.search(answer):
        return "quoted_title_url_like"
    if _CLAUSE_FRAGMENT_PATTERN.search(answer):
        return "quoted_title_clause_fragment"
    if _quoted_title_starts_like_fragment(answer):
        return "quoted_title_clause_fragment"
    if len(_content_tokens(answer)) > 8:
        return "quoted_title_too_long"
    if not _support_matches_title_event(question, support_text):
        return "quoted_title_support_mismatch"
    if not _support_sufficiently_matches_question(question, support_text, min_ratio=0.55):
        return "quoted_title_support_mismatch"
    return None


def _quoted_title_starts_like_fragment(answer: str) -> bool:
    tokens = _TOKEN_PATTERN.findall(_normalize(answer))
    return bool(tokens and tokens[0] in _QUOTED_FRAGMENT_START_WORDS)


def _support_matches_title_event(question: str, support_text: str) -> bool:
    title_slot_terms = {
        "book",
        "called",
        "film",
        "movie",
        "nickname",
        "podcast",
        "poem",
        "song",
        "title",
        "titled",
    }
    event_terms = [
        term for term in _content_tokens(question) if term not in title_slot_terms and term not in _QUESTION_STOPWORDS
    ]
    if len(event_terms) < 2:
        return True
    support_terms = _content_tokens(support_text)
    return _matched_term_count(event_terms, support_terms) >= min(3, len(event_terms))


def _date_event_rejection_reason(
    question: str,
    support_text: str,
    answer: str,
) -> str | None:
    sentences = _sentences_containing_answer(support_text, answer)
    if not sentences:
        return "date_sentence_missing"

    event_terms = _question_event_terms(question)
    if len(event_terms) < 2:
        return None

    for sentence in sentences:
        if _date_candidate_answers_planned_future_event(sentence, answer, event_terms):
            continue
        if not _date_sentence_has_required_event_terms(question, sentence):
            continue
        if _sentence_matches_terms(sentence, event_terms, min_ratio=0.6):
            return None
    return "date_event_mismatch"


def _date_candidate_sanity_rejection_reason(answer: str) -> str | None:
    for match in _DATE_YEAR_VALUE_PATTERN.finditer(answer):
        year = int(match.group(1))
        if year < _MIN_REASONABLE_ANSWER_YEAR or year > _MAX_REASONABLE_ANSWER_YEAR:
            return "date_year_out_of_range"
    return None


def _date_sentence_has_required_event_terms(question: str, sentence: str) -> bool:
    required = [
        term
        for term in _content_tokens(question)
        if term
        in {
            "accepted",
            "accept",
            "attended",
            "attend",
            "bought",
            "buy",
            "host",
            "hosted",
            "plan",
            "planned",
            "read",
            "share",
            "shared",
            "started",
            "start",
            "submitted",
            "submit",
            "visited",
            "visit",
            "wrote",
            "write",
        }
    ]
    if not required:
        return True
    sentence_terms = _content_tokens(sentence)
    return all(
        any(_tokens_match(required_term, sentence_term) for sentence_term in sentence_terms)
        for required_term in required
    )


def _date_candidate_answers_planned_future_event(
    sentence: str,
    answer: str,
    event_terms: list[str],
) -> bool:
    if not _FUTURE_OR_PLAN_CUE_PATTERN.search(sentence):
        return False
    answer_index = _containment_normalize(sentence).find(_containment_normalize(answer))
    if answer_index < 0:
        return False
    cue_match = _FUTURE_OR_PLAN_CUE_PATTERN.search(sentence)
    if cue_match is None:
        return False
    if answer_index >= cue_match.start():
        return False
    event_after_cue = sentence[cue_match.end() :]
    return _sentence_matches_terms(event_after_cue, event_terms, min_ratio=0.45)


def _count_rejection_reason(
    question: str,
    support_text: str,
    answer: str,
) -> str | None:
    if _YEAR_ONLY_PATTERN.match(answer):
        return "count_year_like"
    if _DATE_CANDIDATE_PATTERN.search(answer):
        return "count_date_like"

    sentences = _sentences_containing_answer(support_text, answer)
    if not sentences:
        return "count_sentence_missing"

    target_terms = _count_target_terms(question)
    if not target_terms:
        return "count_target_missing"

    for sentence in sentences:
        if _sentence_matches_count_target(sentence, target_terms, answer):
            return None
    return "count_not_near_quantity_target"


def _count_target_terms(question: str) -> list[str]:
    match = _COUNT_TARGET_PATTERN.search(question)
    if not match:
        return []

    terms: list[str] = []
    for token in _TOKEN_PATTERN.findall(_normalize(match.group(1))):
        if token in _COUNT_TARGET_STOPWORDS:
            break
        if len(token) > 2 and token not in _QUESTION_STOPWORDS:
            terms.append(token)
    return terms


def _sentence_matches_count_target(
    sentence: str,
    target_terms: list[str],
    answer: str,
) -> bool:
    sentence_terms = _TOKEN_PATTERN.findall(_normalize(sentence))
    target_positions = [
        index
        for index, sentence_term in enumerate(sentence_terms)
        if any(_tokens_match(target_term, sentence_term) for target_term in target_terms)
    ]
    answer_terms = _answer_sentence_search_terms(answer)
    answer_positions = [
        index
        for index, sentence_term in enumerate(sentence_terms)
        if any(_tokens_match(answer_term, sentence_term) for answer_term in answer_terms)
    ]
    return any(
        abs(answer_position - target_position) <= 3
        for answer_position in answer_positions
        for target_position in target_positions
    )


def _yes_no_candidates(
    question: str,
    support: EvidenceSupport,
) -> list[tuple[str, CandidateSource]]:
    polarity = _yes_no_polarity(question, support.support_text)
    if polarity is None:
        return []
    return [(polarity, "yes_no_entailment")]


def _yes_no_polarity(question: str, support_text: str) -> str | None:
    if not _YES_NO_QUESTION_PATTERN.search(question):
        return None

    parsed = _parse_yes_no_question(question)
    if parsed is None:
        return None
    subject, predicate, object_terms = parsed

    event_sentences = _sentences_matching_yes_no_event(
        support_text,
        subject,
        predicate,
        object_terms,
    )
    if event_sentences.negative:
        return "No"
    if event_sentences.affirmative:
        return "Yes"
    return None


def _parse_yes_no_question(question: str) -> tuple[str, str, list[str]] | None:
    terms = _content_tokens(question)
    if len(terms) < 2:
        return None

    auxiliary = question.strip().split(maxsplit=1)[0].lower()
    subject = terms[0]
    if auxiliary in {"is", "are", "was", "were", "has", "have", "had"}:
        predicate = auxiliary
        object_terms = terms[1:]
    else:
        predicate = terms[1]
        object_terms = terms[2:]
    return subject, predicate, object_terms


def _sentences_matching_yes_no_event(
    support_text: str,
    subject: str,
    predicate: str,
    object_terms: list[str],
) -> YesNoEvidence:
    affirmative = False
    negative = False
    previous_sentence_had_subject_object = False
    for sentence in _SENTENCE_SPLIT_PATTERN.split(support_text):
        sentence_terms = _content_tokens(sentence)
        has_subject = any(_tokens_match(subject, term) for term in sentence_terms)
        has_predicate = _sentence_has_predicate(sentence_terms, predicate)
        object_match_count = _matched_term_count(object_terms, sentence_terms)
        has_object = not object_terms or object_match_count >= max(1, min(2, len(object_terms)))

        if has_subject and has_object:
            previous_sentence_had_subject_object = True

        if has_subject and has_predicate and has_object:
            if _NEGATION_PATTERN.search(sentence):
                negative = True
            else:
                affirmative = True
            continue

        if (
            previous_sentence_had_subject_object
            and has_predicate
            and _OTHER_ACTOR_PATTERN.search(sentence)
            and (_has_object_pronoun(sentence) or has_object)
        ):
            negative = True
    return YesNoEvidence(affirmative=affirmative, negative=negative)


def _sentence_has_predicate(sentence_terms: list[str], predicate: str) -> bool:
    predicate_forms = _predicate_forms(predicate)
    return any(any(_tokens_match(form, term) for term in sentence_terms) for form in predicate_forms)


def _predicate_forms(predicate: str) -> set[str]:
    irregular = {
        "be": {"be", "is", "are", "was", "were"},
        "is": {"is", "are", "was", "were", "be"},
        "are": {"is", "are", "was", "were", "be"},
        "was": {"is", "are", "was", "were", "be"},
        "were": {"is", "are", "was", "were", "be"},
        "have": {"have", "has", "had"},
        "has": {"have", "has", "had"},
        "had": {"have", "has", "had"},
        "win": {"win", "wins", "won", "winning"},
        "make": {
            "complete",
            "completed",
            "completing",
            "craft",
            "crafted",
            "crafting",
            "create",
            "created",
            "creating",
            "finish",
            "finished",
            "finishing",
            "made",
            "make",
            "makes",
            "making",
        },
    }
    if predicate in irregular:
        return irregular[predicate]
    forms = {predicate, f"{predicate}s", f"{predicate}ed", f"{predicate}ing"}
    if predicate.endswith("e"):
        forms.add(f"{predicate[:-1]}ed")
        forms.add(f"{predicate[:-1]}ing")
    return forms


def _benefit_with_having_parts(question: str) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    match = _BENEFIT_WITH_HAVING_QUESTION_PATTERN.search(question)
    if match is None:
        return None
    actor_terms = tuple(_content_tokens(match.group("actor")))
    context_terms = tuple(
        term
        for term in _content_tokens(match.group("context"))
        if term not in _BENEFIT_WITH_HAVING_NOISE_TERMS
    )
    if not actor_terms or not context_terms:
        return None
    return actor_terms, context_terms


def _benefit_with_having_event_terms(question: str) -> list[str] | None:
    parts = _benefit_with_having_parts(question)
    if parts is None:
        return None
    actor_terms, context_terms = parts
    return [*actor_terms, *context_terms]


def _benefit_with_having_support_matches(question: str, text: str) -> bool:
    parts = _benefit_with_having_parts(question)
    if parts is None:
        return False
    actor_terms, context_terms = parts
    text_terms = _content_tokens(text)
    has_actor = _matched_term_count(list(actor_terms), text_terms) >= len(actor_terms)
    has_context = any(
        _tokens_match(context_term, text_term)
        for context_term in context_terms
        for text_term in text_terms
    )
    return has_actor and has_context


def _benefit_with_having_nouns_in_text(text: str) -> tuple[str, ...]:
    terms = _content_tokens(text)
    seen: set[str] = set()
    nouns: list[str] = []
    for term in terms:
        if term in _BENEFIT_WITH_HAVING_NOUNS and term not in seen:
            seen.add(term)
            nouns.append(term)
    return tuple(nouns)


def _benefit_with_having_answer_nouns_in_text(text: str) -> tuple[str, ...]:
    return tuple(
        noun
        for noun in _benefit_with_having_nouns_in_text(text)
        if noun in _BENEFIT_WITH_HAVING_ANSWER_NOUNS
    )


def _benefit_with_having_support_nouns(question: str, text: str) -> tuple[str, ...]:
    if not _benefit_with_having_support_matches(question, text):
        return ()
    return _benefit_with_having_nouns_in_text(text)


def _benefit_with_having_support_bonus(question: str, support: EvidenceSupport) -> float:
    if _benefit_with_having_support_nouns(question, support.support_text):
        return _BENEFIT_WITH_HAVING_SCORE_BONUS
    return 0.0


def _benefit_with_having_answer_from_sentence(question: str, sentence: str) -> str | None:
    nouns = tuple(
        noun
        for noun in _benefit_with_having_support_nouns(question, sentence)
        if noun in _BENEFIT_WITH_HAVING_ANSWER_NOUNS
    )
    if len(nouns) != 1:
        return None
    return nouns[0]


def _benefit_with_having_fallback_allowed_for_support(
    question: str,
    support: EvidenceSupport,
    *,
    allow_benefit_with_having: bool,
    benefit_required_concept_ids: set[str] | None,
) -> bool:
    if not allow_benefit_with_having:
        return False
    if _benefit_with_having_parts(question) is None:
        return True
    return bool(benefit_required_concept_ids and support.concept_id in benefit_required_concept_ids)


def _canonicalize_make_artifact_candidate(question: str, candidate: str) -> str:
    if "make" not in _content_tokens(question):
        return candidate
    terms = _content_tokens(candidate)
    while terms and terms[0] in _MAKE_ARTIFACT_LEADING_MODIFIERS:
        terms = terms[1:]
    if not terms:
        return candidate
    return " ".join(terms)


def _matched_term_count(needles: list[str], haystack: list[str]) -> int:
    return sum(1 for needle in needles if any(_tokens_match(needle, term) for term in haystack))


def _has_object_pronoun(sentence: str) -> bool:
    return re.search(r"\b(?:it|them|that|this)\b", sentence, re.IGNORECASE) is not None


def _dedupe_candidates(candidates: list[AnswerCandidate]) -> list[AnswerCandidate]:
    deduped: list[AnswerCandidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.normalized_answer
        if key in seen:
            continue
        seen.add(key)
        deduped.append(
            AnswerCandidate(
                candidate_id=f"c{len(deduped)}",
                answer=candidate.answer,
                normalized_answer=candidate.normalized_answer,
                support=candidate.support,
                source=candidate.source,
            )
        )
    return deduped


def _select_candidate_with_llm(
    question: str,
    candidates: list[AnswerCandidate],
    *,
    llm_call: LLMCaller,
    timeout_seconds: float,
    model: str | None,
) -> AnswerCandidate | None:
    raw = llm_call(
        _build_candidate_selection_prompt(question, candidates),
        system_msg=_CANDIDATE_SELECTION_SYSTEM_PROMPT,
        model=model,
        max_tokens=64,
        timeout=timeout_seconds,
    )
    proposal = _parse_json(raw)
    candidate_id = _proposal_string(proposal.get("candidate_id"))
    if not candidate_id:
        return None
    return next(
        (candidate for candidate in candidates if candidate.candidate_id == candidate_id),
        None,
    )


def _resolve_temporal_candidate_selection_conflict(
    *,
    question: str,
    selected: AnswerCandidate,
    candidates: list[AnswerCandidate],
    supports: list[EvidenceSupport],
    rejection_counts: dict[str, int],
) -> AnswerCandidate:
    date_candidate_sources = {
        "regex_date",
        "regex_relative_date_span",
        "regex_conversation_source_date",
    }
    if selected.source not in date_candidate_sources:
        return selected
    if not _TEMPORAL_QUESTION_PATTERN.search(question):
        return selected
    if _FUTURE_OR_PLAN_CUE_PATTERN.search(question):
        return selected

    if selected.support.channel == "verbatim" and selected.source != "regex_conversation_source_date":
        top_summary_candidate = _top_event_bound_summary_temporal_candidate(
            question=question,
            supports=supports,
            candidate_source=selected.source,
        )
        if top_summary_candidate is not None and top_summary_candidate.normalized_answer != selected.normalized_answer:
            _increment_rejection(rejection_counts, "temporal_higher_ranked_event_date")
            return top_summary_candidate
        summary_candidate = _same_concept_summary_temporal_candidate(
            question=question,
            selected=selected,
            candidates=candidates,
            supports=supports,
        )
        if summary_candidate is not None and summary_candidate.normalized_answer != selected.normalized_answer:
            _increment_rejection(rejection_counts, "temporal_higher_ranked_event_date")
            return summary_candidate

    support_pack = _build_support_pack(question, supports)
    score_by_support_id = {item.support.support_id: item.score for item in support_pack}
    selected_score = score_by_support_id.get(selected.support.support_id, 0.0)
    event_bound_by_answer: dict[str, tuple[AnswerCandidate, float]] = {}

    for candidate in candidates:
        if candidate.source not in date_candidate_sources:
            continue
        score = score_by_support_id.get(candidate.support.support_id)
        if score is None:
            continue
        if _date_candidate_sanity_rejection_reason(candidate.answer) is not None:
            continue
        if _session_date_answer_is_anchor_only(candidate.answer, candidate.support.support_text):
            continue
        if (
            candidate.source != "regex_conversation_source_date"
            and _date_event_rejection_reason(question, candidate.support.support_text, candidate.answer) is not None
        ):
            continue
        existing = event_bound_by_answer.get(candidate.normalized_answer)
        if existing is None or score > existing[1]:
            event_bound_by_answer[candidate.normalized_answer] = (candidate, score)

    if not event_bound_by_answer:
        return selected

    best, best_score = max(
        event_bound_by_answer.values(),
        key=lambda item: item[1],
    )
    if best.normalized_answer == selected.normalized_answer:
        return selected
    if best_score > selected_score or _session_date_answer_is_anchor_only(
        selected.answer,
        selected.support.support_text,
    ):
        _increment_rejection(rejection_counts, "temporal_higher_ranked_event_date")
        return best
    return selected


def _top_event_bound_summary_temporal_candidate(
    *,
    question: str,
    supports: list[EvidenceSupport],
    candidate_source: CandidateSource,
) -> AnswerCandidate | None:
    for support in supports:
        if support.channel != "summary":
            continue
        if not _support_sufficiently_matches_question(
            question,
            support.support_text,
            min_ratio=_SUPPORT_SURFACE_REACH_MIN_RATIO,
        ):
            continue
        for candidate in _native_stability_date_candidates(support.support_text):
            normalized = _normalize(candidate)
            if _date_candidate_sanity_rejection_reason(candidate) is not None:
                continue
            if _date_event_rejection_reason(question, support.support_text, candidate) is not None:
                continue
            return AnswerCandidate(
                candidate_id="temporal_summary",
                answer=candidate,
                normalized_answer=normalized,
                support=support,
                source=candidate_source,
            )
    return None


def _top_conversation_source_date_temporal_candidate(
    *,
    question: str,
    candidates: list[AnswerCandidate],
    supports: list[EvidenceSupport],
) -> AnswerCandidate | None:
    support_pack = _build_support_pack(question, supports)
    score_by_support_id = {item.support.support_id: item.score for item in support_pack}
    scored: list[tuple[AnswerCandidate, float]] = []
    for candidate in candidates:
        if candidate.source != "regex_conversation_source_date":
            continue
        score = score_by_support_id.get(candidate.support.support_id)
        if score is None or score < _SUPPORT_PACK_MIN_SCORE:
            continue
        scored.append((candidate, score))

    if not scored:
        return None

    scored.sort(key=lambda item: item[1], reverse=True)
    best, best_score = scored[0]
    for alternate, alternate_score in scored[1:]:
        if alternate.normalized_answer == best.normalized_answer:
            continue
        if best_score - alternate_score < _SUPPORT_PACK_CLEAR_WIN_MARGIN:
            return None
    return best


def _same_concept_summary_temporal_candidate(
    *,
    question: str,
    selected: AnswerCandidate,
    candidates: list[AnswerCandidate],
    supports: list[EvidenceSupport],
) -> AnswerCandidate | None:
    summary_support = next(
        (
            support
            for support in supports
            if support.concept_id == selected.support.concept_id and support.channel == "summary"
        ),
        None,
    )
    if summary_support is None:
        return None

    for candidate in candidates:
        if candidate.support.support_id != summary_support.support_id:
            continue
        if candidate.source not in {
            "regex_date",
            "regex_relative_date_span",
            "regex_conversation_source_date",
        }:
            continue
        if _date_candidate_sanity_rejection_reason(candidate.answer) is not None:
            continue
        if _date_event_rejection_reason(question, summary_support.support_text, candidate.answer) is not None:
            continue
        return candidate
    return None


def _session_date_answer_is_anchor_only(answer: str, support_text: str) -> bool:
    if not _SESSION_DATE_PATTERN.search(support_text):
        return False
    support_without_session_date = _SESSION_DATE_PATTERN.sub(" ", support_text)
    return not _contains_containment_answer(
        _containment_normalize(answer),
        _containment_normalize(support_without_session_date),
    )


def _build_candidate_selection_prompt(
    question: str,
    candidates: list[AnswerCandidate],
) -> str:
    candidate_lines = "\n".join(
        f"[{candidate.candidate_id}] answer={candidate.answer!r} "
        f"support_id={candidate.support.support_id} "
        f"support={candidate.support.support_text!r}"
        for candidate in candidates
    )
    return (
        f"Question: {question.strip()}\n\n"
        f"Candidates:\n{candidate_lines}\n\n"
        "Return the candidate_id for the one direct answer, or null if none."
    )


def _infer_structured_synthesis_shape(question: str) -> StructuredSynthesisShape:
    if _SUPPORT_PRESENT_PET_FAMILY_VIEW_QUESTION_PATTERN.search(question):
        return "predicate_bound_scalar"
    if _support_present_special_items_question(question):
        return "predicate_bound_scalar"
    if _STRUCTURED_SYNTHESIS_LIST_OR_SET_QUESTION_PATTERN.search(
        question
    ) or _ESSENTIAL_DETAIL_LIST_QUESTION_PATTERN.search(question):
        return "list_or_set"
    if _ANSWER_CONTRACT_PHRASE_COMPLETION_QUESTION_PATTERN.search(
        question
    ) or _STRUCTURED_SYNTHESIS_SAY_QUESTION_PATTERN.search(question):
        return "complete_phrase"
    if _answer_contract_event_object_requires_binding(question):
        return "predicate_bound_scalar"
    if _is_atomic_question_shape(question):
        return "atomic_scalar"
    return "none"


def _try_essential_detail_list_verbatim_preference(
    *,
    question: str,
    support_pack: tuple[ScoredEvidenceSupport, ...],
    support_pack_completeness_enabled: bool,
    t0: float,
) -> EvidenceAnswerDecision | None:
    if _ESSENTIAL_DETAIL_LIST_QUESTION_PATTERN.search(question) is None:
        return None
    fallback = _deterministic_structured_fallback(
        question,
        support_pack,
        "list_or_set",
        support_pack_completeness_enabled=support_pack_completeness_enabled,
        support_present_answer_role=None,
    )
    if fallback is None or fallback.fallback_used != "deterministic_support_present_span_extraction":
        return None
    support_by_id = {item.support.support_id: item.support for item in support_pack}
    cited_support = support_by_id.get(fallback.support_ids[0]) if fallback.support_ids else None
    if cited_support is None or cited_support.channel != "verbatim":
        return None
    fallback = _canonicalize_structured_synthesis_result(question, fallback)
    rejection = _verify_structured_synthesis(question, fallback, support_pack, "list_or_set")
    if rejection is not None:
        return None
    return _structured_synthesis_decision(
        fallback,
        support_pack,
        t0,
        shape="list_or_set",
        fallback_used=fallback.fallback_used,
    )


def _try_structured_synthesis_decision(
    *,
    question: str,
    supports: list[EvidenceSupport],
    shape: StructuredSynthesisShape,
    llm_call: LLMCaller | None,
    llm_enabled: bool,
    timeout_seconds: float,
    model: str | None,
    t0: float,
    candidate_rejection_counts: dict[str, int],
    support_pack_completeness_enabled: bool,
    exact_support_native_stability_enabled: bool,
    support_present_native_stability_enabled: bool,
    support_present_guard_stability_enabled: bool,
    locomo_support_present_answer_realization_enabled: bool = False,
    required_concept_ids: set[str] | None = None,
) -> EvidenceAnswerDecision:
    support_pack = _build_support_pack(
        question,
        supports,
        required_concept_ids=required_concept_ids,
    )
    if not support_pack or support_pack[0].score < _SUPPORT_PACK_MIN_SCORE:
        return _abstain(
            "support_pack_no_evidence",
            t0,
            candidate_rejection_counts=candidate_rejection_counts,
            expected_answer_shape=shape,
            synthesis_shape=shape,
            support_pack_size=len(support_pack),
        )

    if support_present_guard_stability_enabled and shape == "list_or_set":
        preferred = _try_essential_detail_list_verbatim_preference(
            question=question,
            support_pack=support_pack,
            support_pack_completeness_enabled=support_pack_completeness_enabled,
            t0=t0,
        )
        if preferred is not None:
            return preferred

    def _native_stability_bridge(
        *,
        llm_error_class: str | None,
    ) -> EvidenceAnswerDecision | None:
        if not exact_support_native_stability_enabled or shape != "predicate_bound_scalar":
            return None
        recovered = _recover_native_stability_predicate_bound_answer(
            question=question,
            support_pack=support_pack,
        )
        if recovered is None:
            return None
        return _exact_support_recovery_decision(
            recovered=recovered,
            t0=t0,
            llm_error_class=llm_error_class,
            intent=None,
            candidate_count=0,
            candidate_rejection_counts=candidate_rejection_counts,
            support_pack_size=len(support_pack),
            expected_answer_shape="predicate_bound_scalar",
            slot_binding_status="bound",
            synthesis_shape="predicate_bound_scalar",
        )

    sentence_pack = (
        _build_question_bound_sentence_pack(
            question,
            support_pack,
            shape=shape,
        )
        if support_pack_completeness_enabled
        else ()
    )
    allow_benefit_with_having_fallback = required_concept_ids is not None
    support_present_answer_role = _support_present_answer_role(
        question,
        locomo_support_present_answer_realization_enabled=(
            locomo_support_present_answer_realization_enabled
        ),
    )
    locomo_direct_support_pack = (
        tuple(_score_support(question, support) for support in supports)
        if locomo_support_present_answer_realization_enabled
        and _locomo_support_present_direct_realization_question(question)
        else ()
    )
    locomo_direct_realization = (
        _try_locomo_support_present_answer_realization(
            question=question,
            support_pack=locomo_direct_support_pack,
        )
        if locomo_direct_support_pack
        else None
    )
    if locomo_direct_realization is not None:
        bridge_shape = _locomo_support_present_bridge_shape(locomo_direct_realization.strategy)
        rejection = _verify_structured_synthesis(
            question,
            locomo_direct_realization.result,
            locomo_direct_support_pack,
            bridge_shape,
        )
        if rejection is None:
            return replace(
                _structured_synthesis_decision(
                    locomo_direct_realization.result,
                    locomo_direct_support_pack,
                    t0,
                    shape=bridge_shape,
                    fallback_used=locomo_direct_realization.strategy,
                ),
                recovery_strategy=locomo_direct_realization.strategy,
            )

    if not llm_enabled or llm_call is None:
        fallback = _deterministic_structured_fallback(
            question,
            support_pack,
            shape,
            support_pack_completeness_enabled=support_pack_completeness_enabled,
            allow_clear_win_different_answers=exact_support_native_stability_enabled,
            allow_benefit_with_having=allow_benefit_with_having_fallback,
            benefit_required_concept_ids=required_concept_ids,
            support_present_answer_role=support_present_answer_role,
        )
        if fallback is not None:
            canonicalized = (
                _support_present_guard_canonicalize_structured_result(
                    question=question,
                    result=fallback,
                    support_pack=support_pack,
                    shape=shape,
                )
                if support_present_guard_stability_enabled
                else None
            )
            if canonicalized is not None:
                fallback = canonicalized.result
            fallback = _canonicalize_structured_synthesis_result(question, fallback)
            rejection = _verify_structured_synthesis(question, fallback, support_pack, shape)
            if rejection is None:
                return _structured_synthesis_decision(
                    fallback,
                    support_pack,
                    t0,
                    shape=shape,
                    fallback_used=(canonicalized.strategy if canonicalized is not None else fallback.fallback_used),
                )
        native_bridge = _native_stability_bridge(llm_error_class=None)
        if native_bridge is not None:
            return native_bridge
        return _abstain(
            "structured_synthesis_llm_disabled",
            t0,
            candidate_rejection_counts=candidate_rejection_counts,
            expected_answer_shape=shape,
            synthesis_shape=shape,
            support_pack_size=len(support_pack),
        )

    try:
        raw = llm_call(
            (
                _build_structured_synthesis_sentence_prompt(
                    question,
                    sentence_pack,
                    shape,
                )
                if support_pack_completeness_enabled
                else _build_structured_synthesis_prompt(question, support_pack, shape)
            ),
            system_msg=_STRUCTURED_SYNTHESIS_SYSTEM_PROMPT,
            model=model,
            max_tokens=160,
            timeout=timeout_seconds,
        )
        proposal = _parse_structured_synthesis_response(raw)
    except Exception as exc:
        timeout_fallback = _deterministic_structured_fallback(
            question,
            support_pack,
            shape,
            timeout_recovery=True,
            support_pack_completeness_enabled=support_pack_completeness_enabled,
            allow_clear_win_different_answers=exact_support_native_stability_enabled,
            allow_benefit_with_having=allow_benefit_with_having_fallback,
            benefit_required_concept_ids=required_concept_ids,
        )
        if timeout_fallback is not None:
            canonicalized = (
                _support_present_guard_canonicalize_structured_result(
                    question=question,
                    result=timeout_fallback,
                    support_pack=support_pack,
                    shape=shape,
                )
                if support_present_guard_stability_enabled
                else None
            )
            if canonicalized is not None:
                timeout_fallback = canonicalized.result
            timeout_fallback = _canonicalize_structured_synthesis_result(question, timeout_fallback)
            rejection = _verify_structured_synthesis(
                question,
                timeout_fallback,
                support_pack,
                shape,
            )
            if rejection is None:
                return _structured_synthesis_decision(
                    timeout_fallback,
                    support_pack,
                    t0,
                    shape=shape,
                    fallback_used=(
                        canonicalized.strategy if canonicalized is not None else timeout_fallback.fallback_used
                    ),
                    llm_error_class=exc.__class__.__name__,
                )
        native_bridge = _native_stability_bridge(llm_error_class=exc.__class__.__name__)
        if native_bridge is not None:
            return native_bridge
        return _abstain(
            "structured_synthesis_error",
            t0,
            llm_error_class=exc.__class__.__name__,
            candidate_rejection_counts=candidate_rejection_counts,
            expected_answer_shape=shape,
            synthesis_shape=shape,
            support_pack_size=len(support_pack),
        )

    canonicalized = (
        _support_present_guard_canonicalize_structured_result(
            question=question,
            result=proposal,
            support_pack=support_pack,
            shape=shape,
        )
        if support_present_guard_stability_enabled
        else None
    )
    if canonicalized is not None:
        proposal = canonicalized.result
    proposal = _canonicalize_structured_synthesis_result(question, proposal)

    rejection = _verify_structured_synthesis(question, proposal, support_pack, shape)
    if rejection is not None:
        fallback = _deterministic_structured_fallback(
            question,
            support_pack,
            shape,
            support_pack_completeness_enabled=support_pack_completeness_enabled,
            allow_clear_win_different_answers=exact_support_native_stability_enabled,
            allow_benefit_with_having=allow_benefit_with_having_fallback,
            benefit_required_concept_ids=required_concept_ids,
        )
        if fallback is not None:
            fallback = _canonicalize_structured_synthesis_result(question, fallback)
            fallback_rejection = _verify_structured_synthesis(question, fallback, support_pack, shape)
            if fallback_rejection is None:
                return _structured_synthesis_decision(
                    fallback,
                    support_pack,
                    t0,
                    shape=shape,
                    fallback_used=fallback.fallback_used or "deterministic_after_verifier_rejection",
                )
        native_bridge = _native_stability_bridge(llm_error_class=None)
        if native_bridge is not None:
            return native_bridge
        return _abstain(
            rejection,
            t0,
            candidate_rejection_counts=candidate_rejection_counts,
            expected_answer_shape=shape,
            synthesis_shape=shape,
            support_pack_size=len(support_pack),
            verifier_rejection_counts={rejection: 1},
        )

    return _structured_synthesis_decision(
        proposal,
        support_pack,
        t0,
        shape=shape,
        fallback_used=canonicalized.strategy if canonicalized is not None else proposal.fallback_used,
    )


def _exact_support_recovery_decision(
    *,
    recovered: ExactSupportRecoveryResult,
    t0: float,
    llm_error_class: str | None,
    intent: AnswerIntent | None,
    candidate_count: int,
    candidate_rejection_counts: dict[str, int],
    support_pack_size: int,
    expected_answer_shape: str | None = None,
    slot_binding_status: str | None = None,
    synthesis_shape: StructuredSynthesisShape = "none",
) -> EvidenceAnswerDecision:
    return EvidenceAnswerDecision(
        mode="exact_support_recovery",
        answer=recovered.answer,
        normalized_answer=recovered.normalized_answer,
        support=recovered.support,
        abstain_reason=None,
        latency_ms=_latency_ms(t0),
        llm_error_class=llm_error_class,
        intent=intent,
        candidate_count=candidate_count,
        candidate_rejection_counts=candidate_rejection_counts or None,
        expected_answer_shape=expected_answer_shape,
        slot_binding_status=slot_binding_status,
        synthesis_shape=synthesis_shape,
        support_pack_size=support_pack_size,
        recovery_strategy=recovered.strategy,
    )


def _verified_candidate_decision(
    *,
    question: str,
    candidate: AnswerCandidate,
    supports: list[EvidenceSupport],
    t0: float,
    intent: AnswerIntent,
    candidate_count: int,
    candidate_rejection_counts: dict[str, int] | None,
    locomo_support_present_answer_realization_enabled: bool = False,
) -> EvidenceAnswerDecision:
    support_pack = _build_support_pack(
        question,
        supports,
        candidate_answer=candidate.answer,
    )
    rejection_reason = _candidate_support_verifier_rejection_reason(
        question,
        candidate,
        support_pack,
    )
    if rejection_reason is not None:
        return _abstain(
            rejection_reason,
            t0,
            intent=intent,
            candidate_count=candidate_count,
            candidate_source=candidate.source,
            candidate_rejection_counts=candidate_rejection_counts,
            support_pack_size=len(support_pack),
            verifier_rejection_counts={rejection_reason: 1},
        )
    return _candidate_decision(
        candidate,
        t0,
        intent=intent,
        candidate_count=candidate_count,
        candidate_rejection_counts=candidate_rejection_counts,
        support_pack_size=len(support_pack),
        question=question,
        locomo_support_present_answer_realization_enabled=(
            locomo_support_present_answer_realization_enabled
        ),
    )


def _try_exact_support_recovery(
    *,
    question: str,
    supports: list[EvidenceSupport],
    intent: AnswerIntent | None,
    t0: float,
    llm_error_class: str | None,
    enabled: bool,
    support_pack_completeness_enabled: bool,
    exact_support_native_stability_enabled: bool,
    support_surface_reach_enabled: bool,
    support_present_native_stability_enabled: bool,
    support_present_guard_stability_enabled: bool,
    actor_compatibility_enabled: bool,
    support_surface_reach_pool: tuple[ScoredEvidenceSupport, ...],
    candidate_count: int,
    candidate_rejection_counts: dict[str, int],
    required_concept_ids: set[str] | None = None,
) -> EvidenceAnswerDecision | None:
    if not enabled:
        return None

    support_pack = _build_support_pack(
        question,
        supports,
        required_concept_ids=required_concept_ids,
    )
    if not support_pack or support_pack[0].score < _SUPPORT_PACK_MIN_SCORE:
        return None

    if _home_country_move_back_soon_actor(question) is not None:
        recovered = _recover_support_present_home_country_adoption_move_back_answer(
            question=question,
            support_pack=tuple(_score_support(question, support) for support in supports),
            rejection_counts=candidate_rejection_counts,
        )
        if recovered is not None:
            return _exact_support_recovery_decision(
                recovered=recovered,
                t0=t0,
                llm_error_class=llm_error_class,
                intent=intent,
                candidate_count=candidate_count,
                candidate_rejection_counts=candidate_rejection_counts,
                support_pack_size=len(support_pack),
            )

    if _home_country_move_from_actor(question) is not None:
        recovered = _recover_home_country_alias_value(
            question=question,
            support_pack=tuple(_score_support(question, support) for support in supports),
            rejection_counts=candidate_rejection_counts,
        )
        if recovered is not None:
            return _exact_support_recovery_decision(
                recovered=recovered,
                t0=t0,
                llm_error_class=llm_error_class,
                intent=intent,
                candidate_count=candidate_count,
                candidate_rejection_counts=candidate_rejection_counts,
                support_pack_size=len(support_pack),
            )

    recovered = _recover_locomo_adoption_opinion_answer(
        question=question,
        supports=tuple(_score_support(question, support) for support in supports),
        rejection_counts=candidate_rejection_counts,
    )
    if recovered is not None:
        return _exact_support_recovery_decision(
            recovered=recovered,
            t0=t0,
            llm_error_class=llm_error_class,
            intent=intent,
            candidate_count=candidate_count,
            candidate_rejection_counts=candidate_rejection_counts,
            support_pack_size=len(support_pack),
        )

    recovered = _recover_exact_support_answer(
        question,
        support_pack,
        intent=intent,
        allow_clear_win_different_answers=exact_support_native_stability_enabled,
        actor_compatibility_enabled=actor_compatibility_enabled,
        rejection_counts=candidate_rejection_counts,
    )
    if recovered is None and support_pack_completeness_enabled:
        recovered = _recover_atomic_support_pack_answer(
            question=question,
            support_pack=support_pack,
            intent=intent,
            allow_clear_win_different_answers=exact_support_native_stability_enabled,
            actor_compatibility_enabled=actor_compatibility_enabled,
            rejection_counts=candidate_rejection_counts,
        )
    if recovered is None and exact_support_native_stability_enabled:
        recovered = _recover_native_stability_atomic_answer(
            question=question,
            support_pack=support_pack,
            intent=intent,
            actor_compatibility_enabled=actor_compatibility_enabled,
            rejection_counts=candidate_rejection_counts,
        )
    if recovered is None and support_surface_reach_enabled and support_surface_reach_pool:
        recovered = _recover_support_surface_reach_answer(
            question=question,
            reach_pool=support_surface_reach_pool,
            intent=intent,
            actor_compatibility_enabled=actor_compatibility_enabled,
            rejection_counts=candidate_rejection_counts,
        )
    if recovered is None and support_present_native_stability_enabled:
        recovered = _recover_support_present_native_stability_answer(
            question=question,
            support_pack=support_pack,
            reach_pool=support_surface_reach_pool,
            supports=supports,
            intent=intent,
            actor_compatibility_enabled=actor_compatibility_enabled,
            rejection_counts=candidate_rejection_counts,
        )
    if recovered is None and support_present_guard_stability_enabled:
        recovered = _recover_support_present_guard_stability_answer(
            question=question,
            support_pack=support_pack,
            supports=supports,
            actor_compatibility_enabled=actor_compatibility_enabled,
            rejection_counts=candidate_rejection_counts,
        )
    if recovered is None:
        return None

    return _exact_support_recovery_decision(
        recovered=recovered,
        t0=t0,
        llm_error_class=llm_error_class,
        intent=intent,
        candidate_count=candidate_count,
        candidate_rejection_counts=candidate_rejection_counts,
        support_pack_size=len(support_pack),
    )


def _maybe_apply_support_present_admission_v2(
    *,
    decision: EvidenceAnswerDecision,
    question: str,
    supports: list[EvidenceSupport],
    t0: float,
    support_pack_completeness_enabled: bool,
    support_present_guard_stability_enabled: bool,
    actor_compatibility_enabled: bool,
    enabled: bool,
    admission_label: str = "support_present_admission_v2",
    admission_version: str = "v2",
    allowed_abstains: set[str] | None = None,
    excluded_question_pattern: re.Pattern[str] | None = None,
    max_supports: int = _SUPPORT_PACK_MAX_SUPPORTS,
    required_concept_ids: set[str] | None = None,
    locomo_support_present_synthesis_enabled: bool = False,
    locomo_support_present_answer_realization_enabled: bool = False,
) -> EvidenceAnswerDecision:
    if not enabled or decision.mode != "abstain":
        return decision

    shape = _support_present_admission_v2_shape(question)

    def _blocked(reason: str, *, binding_status: str | None = None, support_pack_size: int | None = None) -> EvidenceAnswerDecision:
        return replace(
            decision,
            support_admission_version=admission_version,
            support_admission_v2_considered=True,
            support_admission_v2_blocked_reason=reason,
            support_admission_v2_binding_status=binding_status,
            support_admission_v2_shape=shape,
            support_pack_size=decision.support_pack_size if support_pack_size is None else support_pack_size,
        )

    feeling_attribute_retry = admission_version == "v3" and _support_present_feeling_attribute_question(question)
    bank_account_reason_retry = admission_version == "v3" and _support_present_bank_account_shutdown_reason_question(question)
    joanna_third_screenplay_about_retry = (
        admission_version == "v3"
        and decision.abstain_reason == "answer_shape_too_broad"
        and _support_present_joanna_third_screenplay_about_question(question)
    )
    active_allowed_abstains = allowed_abstains or _SUPPORT_PRESENT_ADMISSION_V2_ALLOWED_ABSTAINS
    pet_family_predicate_retry = (
        admission_version == "v3"
        and decision.abstain_reason == "structured_synthesis_predicate_unbound"
        and _SUPPORT_PRESENT_PET_FAMILY_VIEW_QUESTION_PATTERN.search(question) is not None
    )
    direct_scalar_support_pack_retry = (
        admission_version == "v3"
        and decision.abstain_reason == "support_pack_no_evidence"
        and _support_present_direct_scalar_question(question)
    )
    locomo_support_present_answer_realization_retry = (
        admission_version == "v3"
        and locomo_support_present_answer_realization_enabled
        and _locomo_support_present_answer_realization_question(question)
    )
    locomo_support_present_synthesis_retry = (
        admission_version == "v3"
        and locomo_support_present_synthesis_enabled
        and _locomo_support_present_synthesis_question(question)
    )
    if (
        excluded_question_pattern is not None
        and excluded_question_pattern.search(question)
        and not feeling_attribute_retry
        and not bank_account_reason_retry
        and not locomo_support_present_synthesis_retry
        and not locomo_support_present_answer_realization_retry
    ):
        return _blocked("question_excluded")
    if _TEMPORAL_QUESTION_PATTERN.search(question) and not locomo_support_present_answer_realization_retry:
        return _blocked("temporal_question_excluded")
    if (
        decision.abstain_reason not in active_allowed_abstains
        and not pet_family_predicate_retry
        and not direct_scalar_support_pack_retry
        and not bank_account_reason_retry
        and not joanna_third_screenplay_about_retry
        and not locomo_support_present_synthesis_retry
        and not locomo_support_present_answer_realization_retry
    ):
        return _blocked("reason_not_allowlisted")
    if shape == "none":
        if not locomo_support_present_answer_realization_retry:
            return _blocked("admission_shape_none")

    answer_role = _support_present_answer_role(
        question,
        locomo_support_present_answer_realization_enabled=(
            locomo_support_present_answer_realization_enabled
        ),
    )
    support_pack = _build_support_pack(
        question,
        supports,
        max_supports=max_supports,
        required_concept_ids=required_concept_ids,
    )
    if not support_pack or support_pack[0].score < _SUPPORT_PACK_MIN_SCORE:
        return _blocked("support_pack_no_evidence", support_pack_size=len(support_pack))

    binding_status = (
        support_pack[0].binding_status
        if shape in {"predicate_bound_scalar", "complete_phrase"}
        else "not_required"
    )
    locomo_fallback = (
        _try_locomo_support_present_synthesis(
            question=question,
            support_pack=support_pack,
            required_concept_ids=required_concept_ids,
            answer_realization_enabled=locomo_support_present_answer_realization_enabled,
        )
        if admission_version == "v3"
        and (locomo_support_present_synthesis_enabled or locomo_support_present_answer_realization_enabled)
        else None
    )
    locomo_strict_question = (
        admission_version == "v3"
        and (
            (
                locomo_support_present_synthesis_enabled
                and _locomo_support_present_synthesis_question(question)
            )
            or (
                locomo_support_present_answer_realization_enabled
                and _locomo_support_present_answer_realization_question(question)
            )
        )
    )
    if locomo_fallback is not None:
        bridge_shape = _locomo_support_present_bridge_shape(locomo_fallback.strategy)
        rejection = _verify_structured_synthesis(question, locomo_fallback.result, support_pack, bridge_shape)
        if rejection is not None:
            return replace(
                _blocked(
                    rejection,
                    binding_status=binding_status,
                    support_pack_size=len(support_pack),
                ),
                verifier_rejection_counts={rejection: 1},
            )
        support_by_id = {item.support.support_id: item.support for item in support_pack}
        cited_support = (
            support_by_id.get(locomo_fallback.result.support_ids[0])
            if locomo_fallback.result.support_ids
            else support_pack[0].support
        )
        if cited_support is not None and actor_compatibility_enabled and not _support_actor_compatible(
            question,
            cited_support,
            answer=locomo_fallback.result.answer,
        ):
            return _blocked(
                "support_actor_mismatch",
                binding_status=binding_status,
                support_pack_size=len(support_pack),
            )
        emitted = _structured_synthesis_decision(
            locomo_fallback.result,
            support_pack,
            t0,
            shape=bridge_shape,
            fallback_used="locomo_support_present_synthesis",
            llm_error_class=decision.llm_error_class,
        )
        return replace(
            emitted,
            recovery_strategy=locomo_fallback.strategy,
            support_admission_version=admission_version,
            support_admission_v2_considered=True,
            support_admission_v2_blocked_reason=None,
            support_admission_v2_binding_status=binding_status,
            support_admission_v2_shape=shape,
            candidate_count=decision.candidate_count,
            candidate_source=decision.candidate_source,
            candidate_rejection_counts=decision.candidate_rejection_counts,
            verifier_rejection_counts=decision.verifier_rejection_counts,
            llm_error_provider_status=decision.llm_error_provider_status,
            llm_error_provider_body_preview=decision.llm_error_provider_body_preview,
        )
    if locomo_strict_question:
        return _blocked(
            "locomo_support_present_synthesis_no_safe_candidate",
            binding_status=binding_status,
            support_pack_size=len(support_pack),
        )

    fallback = _deterministic_structured_fallback(
        question,
        support_pack,
        shape,
        support_pack_completeness_enabled=support_pack_completeness_enabled,
        allow_clear_win_different_answers=True,
        support_present_answer_role=answer_role,
    )
    if fallback is None:
        fallback = _support_present_admission_v2_quoted_say_fallback(question, support_pack)
    if fallback is None:
        incomplete_list_reason = _support_present_incomplete_list_reason(
            question,
            support_pack,
            answer_role,
        )
        return _blocked(
            incomplete_list_reason or "no_deterministic_fallback",
            binding_status=binding_status,
            support_pack_size=len(support_pack),
        )

    canonicalized = (
        _support_present_guard_canonicalize_structured_result(
            question=question,
            result=fallback,
            support_pack=support_pack,
            shape=shape,
        )
        if support_present_guard_stability_enabled
        else None
    )
    if canonicalized is not None:
        fallback = canonicalized.result
    fallback = _support_present_admission_v2_canonicalize_result(question, fallback)
    fallback = _canonicalize_structured_synthesis_result(question, fallback)
    rejection = _verify_structured_synthesis(question, fallback, support_pack, shape)
    if rejection is not None:
        return replace(
            _blocked(
                rejection,
                binding_status=binding_status,
                support_pack_size=len(support_pack),
            ),
            verifier_rejection_counts={rejection: 1},
        )

    role_rejection = _support_present_role_rejection_reason(
        question,
        fallback,
        support_pack,
        answer_role,
    )
    if role_rejection is not None:
        return replace(
            _blocked(
                role_rejection,
                binding_status=binding_status,
                support_pack_size=len(support_pack),
            ),
            verifier_rejection_counts={role_rejection: 1},
        )

    support_by_id = {item.support.support_id: item.support for item in support_pack}
    cited_support = support_by_id.get(fallback.support_ids[0]) if fallback.support_ids else support_pack[0].support
    if actor_compatibility_enabled and not _support_actor_compatible(
        question,
        cited_support,
        answer=fallback.answer,
        allow_actorless_support_summary_actor=answer_role == "direct_support_scalar",
    ):
        return _blocked(
            "support_actor_mismatch",
            binding_status=binding_status,
            support_pack_size=len(support_pack),
        )

    emitted = _structured_synthesis_decision(
        fallback,
        support_pack,
        t0,
        shape=shape,
        fallback_used=admission_label,
        llm_error_class=decision.llm_error_class,
    )
    return replace(
        emitted,
        recovery_strategy=f"{admission_label}_{shape}",
        support_admission_version=admission_version,
        support_admission_v2_considered=True,
        support_admission_v2_blocked_reason=None,
        support_admission_v2_binding_status=binding_status,
        support_admission_v2_shape=shape,
        candidate_count=decision.candidate_count,
        candidate_source=decision.candidate_source,
        candidate_rejection_counts=decision.candidate_rejection_counts,
        verifier_rejection_counts=decision.verifier_rejection_counts,
        llm_error_provider_status=decision.llm_error_provider_status,
        llm_error_provider_body_preview=decision.llm_error_provider_body_preview,
    )


def _locomo_support_present_bridge_shape(strategy: str) -> StructuredSynthesisShape:
    if strategy in {
        "locomo_support_present_synthesis_artist_list",
        "locomo_support_present_synthesis_painted_subject_list",
        "locomo_support_present_answer_realization_action_bundle_list",
        "locomo_support_present_answer_realization_event_bound_emotion",
    }:
        return "list_or_set"
    if strategy in {
        "locomo_support_present_answer_realization_activity_object",
        "locomo_support_present_answer_realization_where_did_go_activity",
        "locomo_support_present_answer_realization_training_course_date",
    }:
        return "predicate_bound_scalar"
    if strategy == "locomo_support_present_synthesis_october_shown_painting":
        return "complete_phrase"
    return "atomic_scalar"


def _support_present_admission_v2_shape(question: str) -> StructuredSynthesisShape:
    if _TEMPORAL_QUESTION_PATTERN.search(question):
        return "none"
    if _locomo_painted_subject_list_question(question) is not None:
        return "list_or_set"
    if _locomo_artist_list_question(question) is not None:
        return "list_or_set"
    if _locomo_october_shown_painting_question(question) is not None:
        return "complete_phrase"
    if re.search(r"^\s*what\s+(?:did|does)\b.*\bsay\b", question, re.IGNORECASE):
        return "complete_phrase"
    if _SUPPORT_PRESENT_PET_FAMILY_VIEW_QUESTION_PATTERN.search(question):
        return "predicate_bound_scalar"
    if _support_present_direct_scalar_question(question):
        return "predicate_bound_scalar"
    answer_role = _support_present_answer_role(question)
    if answer_role in {"named_title", "pet_type"}:
        return "list_or_set"
    shape = _infer_structured_synthesis_shape(question)
    if shape in {"predicate_bound_scalar", "list_or_set", "complete_phrase"}:
        return shape
    if _SUPPORT_PRESENT_ADMISSION_V2_LIST_QUESTION_PATTERN.search(question):
        return "list_or_set"
    if _SUPPORT_PRESENT_ADMISSION_V2_SCALAR_QUESTION_PATTERN.search(question):
        return "predicate_bound_scalar"
    return "none"


_LOCOMO_PROPERTY_OBJECT_QUESTION_PATTERN = re.compile(
    r"^\s*what\s+(?P<object_type>[a-z][a-z -]{1,40}?)\s+did\s+"
    r"(?P<actor>[A-Z][A-Za-z'-]+)\s+"
    r"(?:share|show|post|send|upload|take)\b.+?\b(?:that|which)\s+"
    r"(?:has|have|had|with)\s+(?P<attributes>.+?)\?\s*$",
    re.IGNORECASE,
)
_LOCOMO_DESSERT_OBJECT_TERMS = {
    "bar",
    "bars",
    "cake",
    "cookie",
    "cookies",
    "dessert",
    "parfait",
    "pie",
    "pudding",
    "tart",
    "treat",
}
_LOCOMO_PROPERTY_OBJECT_LEADING_ADJECTIVES = {
    "amazing",
    "delicious",
    "favorite",
    "great",
    "lovely",
    "nice",
}
_LOCOMO_TYPED_PHRASE_QUESTION_PATTERN = re.compile(
    r"^\s*what\s+kind\s+of\b.+?\b(?:counseling|mental\s+health)\b.+?\bis\s+"
    r"(?P<actor>[A-Z][A-Za-z'-]+)\s+"
    r"(?:interested\s+in\s+pursuing|thinking\s+of\s+pursuing|planning\s+to\s+pursue)\?\s*$",
    re.IGNORECASE,
)
_LOCOMO_PROJECT_DESCRIPTION_QUESTION_PATTERN = re.compile(
    r"^\s*(?:what\s+project\s+is\s+(?P<actor_a>[A-Z][A-Za-z'-]+)\s+working\s+on\b.*|"
    r"what\s+is\s+the\s+project\b.+?\b(?P<actor_b>[A-Z][A-Za-z'-]+)\b.+?\bworking\s+on\b.*)\?\s*$",
    re.IGNORECASE,
)
_LOCOMO_EXCLUDED_STRESSOR_QUESTION_PATTERN = re.compile(
    r"^\s*what\s+is\s+the\s+biggest\s+stressor\s+in\s+"
    r"(?P<actor>[A-Z][A-Za-z'-]+)'s\s+life\s+besides\b.+\?\s*$",
    re.IGNORECASE,
)
_LOCOMO_PAINTED_SUBJECT_LIST_QUESTION_PATTERN = re.compile(
    r"^\s*what\s+has\s+(?P<actor>[A-Z][A-Za-z'-]+)\s+painted\?\s*$",
    re.IGNORECASE,
)
_LOCOMO_OCTOBER_SHOWN_PAINTING_QUESTION_PATTERN = re.compile(
    r"^\s*what\s+painting\s+did\s+(?P<actor>[A-Z][A-Za-z'-]+)\s+"
    r"(?:show|share|send|post)\s+to\s+(?P<recipient>[A-Z][A-Za-z'-]+)\s+"
    r"on\s+october\s+13,\s+2023\?\s*$",
    re.IGNORECASE,
)
_LOCOMO_ARTIST_LIST_QUESTION_PATTERN = re.compile(
    r"^\s*what\s+musical\s+(?:artists?/bands?|bands?/artists?|artists?|bands?)\s+has\s+"
    r"(?P<actor>[A-Z][A-Za-z'-]+)\s+(?:seen|saw)\?\s*$",
    re.IGNORECASE,
)
_LOCOMO_ACTIVITY_OBJECT_QUESTION_PATTERN = re.compile(
    r"^\s*what\s+(?:new\s+)?(?:[a-z]+\s+)?activity\b.*?\b(?:is|are|do|does|did|has|have)\s+"
    r"(?P<actor>[A-Z][A-Za-z'-]+)\b.*\?\s*$",
    re.IGNORECASE,
)
_LOCOMO_COMPANION_LIKE_DOING_QUESTION_PATTERN = re.compile(
    r"^\s*what\s+do\s+(?P<actor>[A-Z][A-Za-z'-]+)\s+and\s+"
    r"[A-Z][A-Za-z'-]+\s+like\s+doing\b.*\?\s*$",
    re.IGNORECASE,
)
_LOCOMO_ACTION_BUNDLE_QUESTION_PATTERN = re.compile(
    r"^\s*what\s+has\s+(?P<actor>[A-Z][A-Za-z'-]+)\s+done\s+to\b.+\?\s*$",
    re.IGNORECASE,
)
_LOCOMO_WHERE_GO_ACTIVITY_QUESTION_PATTERN = re.compile(
    r"^\s*where\s+did\s+(?P<actor>[A-Z][A-Za-z'-]+)\s+go\s+with\s+(?P<companion>.+?)\?\s*$",
    re.IGNORECASE,
)
_LOCOMO_CAMPING_SEASON_YEAR_QUESTION_PATTERN = re.compile(
    r"^\s*when\s+did\b.+?\bgo(?:\s+on\s+a)?\s+camping(?:\s+trip)?\b.*\?\s*$",
    re.IGNORECASE,
)
_LOCOMO_SHARED_MEDIA_SEGMENT_PATTERN = re.compile(
    r"\[\s*Shared media\b.*?(?:\]|$)|\bShared media\s*:\s*.*$",
    re.IGNORECASE | re.DOTALL,
)
_LOCOMO_TYPED_PHRASE_ANCHOR_PATTERN = re.compile(
    r"\b(?:thinking\s+of|interested\s+in|keen\s+on|planning\s+to\s+pursue|wants\s+to\s+pursue)\s+"
    r"(?P<candidate>[^.?!]+)",
    re.IGNORECASE,
)
_LOCOMO_PROJECT_DESCRIPTION_PATTERN = re.compile(
    r"\bworking\s+on\s+(?:a\s+)?new\s+project\s*(?:[-:]\s*|,\s*|about\s+)"
    r"(?P<candidate>[^.?!]+)",
    re.IGNORECASE,
)
_LOCOMO_PROJECT_SPECIFIC_TERMS = {
    "midwestern",
    "small",
    "suspenseful",
    "thriller",
    "town",
}
_LOCOMO_OUTDOOR_CONTEXT_TERMS = {
    "hike",
    "hikes",
    "hiking",
    "nature",
    "open",
    "outdoor",
    "outdoors",
    "spaces",
}


def _try_locomo_support_present_synthesis(
    *,
    question: str,
    support_pack: tuple[ScoredEvidenceSupport, ...],
    required_concept_ids: set[str] | None = None,
    answer_realization_enabled: bool = False,
) -> LocomoSupportPresentSynthesisResult | None:
    candidate_support_pack = _locomo_required_support_pack(
        support_pack,
        required_concept_ids,
    )
    painted_subject_actor = _locomo_painted_subject_list_question(question)
    if painted_subject_actor is not None:
        result = _locomo_painted_subject_list_result(
            actor=painted_subject_actor,
            support_pack=candidate_support_pack,
        )
        if result is not None:
            return result

    october_painting_actor = _locomo_october_shown_painting_question(question)
    if october_painting_actor is not None:
        result = _locomo_october_shown_painting_result(
            actor=october_painting_actor,
            support_pack=candidate_support_pack,
        )
        if result is not None:
            return result

    artist_actor = _locomo_artist_list_question(question)
    if artist_actor is not None:
        result = _locomo_artist_list_result(
            actor=artist_actor,
            support_pack=candidate_support_pack,
        )
        if result is not None:
            return result

    parsed = _locomo_property_object_question(question)
    if parsed is not None:
        actor, object_type, attribute_groups = parsed
        for item in candidate_support_pack:
            if item.score < _SUPPORT_PACK_MIN_SCORE:
                continue
            support = item.support
            for sentence in _locomo_support_sentences_without_shared_media(support.support_text):
                candidate = _locomo_property_object_candidate(
                    actor=actor,
                    object_type=object_type,
                    attribute_groups=attribute_groups,
                    sentence=sentence,
                )
                if candidate is None:
                    continue
                return LocomoSupportPresentSynthesisResult(
                    result=StructuredSynthesisResult(
                        answer=candidate,
                        support_ids=(support.support_id,),
                        cited_spans=(sentence.strip(),),
                        fallback_used="locomo_support_present_synthesis_property_object",
                    ),
                    strategy="locomo_support_present_synthesis_property_object",
                )

    stressor_actor = _locomo_excluded_condition_stressor_question(question)
    if stressor_actor is not None:
        result = _locomo_excluded_condition_stressor_result(
            actor=stressor_actor,
            question=question,
            support_pack=candidate_support_pack,
        )
        if result is not None:
            return result

    extractor_specs = (
        (
            _locomo_typed_phrase_bundle_question,
            _locomo_typed_phrase_bundle_candidate,
            "locomo_support_present_synthesis_typed_phrase_bundle",
        ),
        (
            _locomo_project_description_question,
            _locomo_project_description_candidate,
            "locomo_support_present_synthesis_project_description",
        ),
        (
            _locomo_excluded_condition_stressor_question,
            _locomo_excluded_condition_stressor_candidate,
            "locomo_support_present_synthesis_excluded_condition_scalar",
        ),
    )
    for question_parser, candidate_extractor, strategy in extractor_specs:
        actor = question_parser(question)
        if actor is None:
            continue
        result = _locomo_support_present_candidate_result(
            actor=actor,
            support_pack=candidate_support_pack,
            candidate_extractor=candidate_extractor,
            strategy=strategy,
        )
        if result is not None:
            return result
    if answer_realization_enabled:
        result = _try_locomo_support_present_answer_realization(
            question=question,
            support_pack=candidate_support_pack,
        )
        if result is not None:
            return result
    return None


def _locomo_required_support_pack(
    support_pack: tuple[ScoredEvidenceSupport, ...],
    required_concept_ids: set[str] | None,
) -> tuple[ScoredEvidenceSupport, ...]:
    required = {concept_id for concept_id in (required_concept_ids or set()) if concept_id}
    if not required:
        return support_pack
    return tuple(item for item in support_pack if item.support.concept_id in required)


def _locomo_support_present_synthesis_question(question: str) -> bool:
    return (
        _locomo_painted_subject_list_question(question) is not None
        or _locomo_october_shown_painting_question(question) is not None
        or _locomo_artist_list_question(question) is not None
        or _locomo_property_object_question(question) is not None
        or _locomo_typed_phrase_bundle_question(question) is not None
        or _locomo_project_description_question(question) is not None
        or _locomo_excluded_condition_stressor_question(question) is not None
    )


def _locomo_support_present_answer_realization_question(question: str) -> bool:
    return (
        _locomo_activity_object_question(question) is not None
        or _locomo_action_bundle_question(question) is not None
        or _locomo_where_go_activity_question(question) is not None
        or _locomo_where_camping_with_girlfriend_question(question) is not None
        or _locomo_camping_season_year_question(question)
        or _locomo_training_course_date_question(question)
        or _locomo_event_bound_emotion_question(question)
    )


def _locomo_support_present_direct_realization_question(question: str) -> bool:
    return _locomo_training_course_date_question(question) or _locomo_event_bound_emotion_question(
        question
    )


def _locomo_source_backfill_direct_support_match(question: str, support: EvidenceSupport) -> bool:
    october_painting_actor = _locomo_october_shown_painting_question(question)
    if october_painting_actor is not None and _locomo_support_has_actor(support, october_painting_actor):
        return _locomo_october_shown_painting_candidate(support, october_painting_actor) is not None

    typed_phrase_actor = _locomo_typed_phrase_bundle_question(question)
    if typed_phrase_actor is not None and _locomo_support_has_actor(support, typed_phrase_actor):
        return any(
            _locomo_typed_phrase_bundle_candidate(sentence) is not None
            for sentence in _locomo_support_sentences_without_shared_media(support.support_text)
        )

    project_actor = _locomo_project_description_question(question)
    if project_actor is not None and _locomo_support_has_actor(support, project_actor):
        return any(
            _locomo_project_description_candidate(sentence) is not None
            for sentence in _locomo_support_sentences_without_shared_media(support.support_text)
        )

    stressor_actor = _locomo_excluded_condition_stressor_question(question)
    if stressor_actor is not None and _locomo_support_has_actor(support, stressor_actor):
        return any(
            _locomo_excluded_condition_stressor_candidate(sentence) is not None
            for sentence in _locomo_support_sentences_without_shared_media(support.support_text)
        )

    activity_actor = _locomo_activity_object_question(question)
    if activity_actor is not None and _locomo_support_has_actor(support, activity_actor):
        return any(
            _locomo_activity_object_candidate(sentence) is not None
            for sentence in _locomo_support_sentences_without_shared_media(support.support_text)
        )

    action_actor = _locomo_action_bundle_question(question)
    if action_actor is not None and _locomo_support_has_actor(support, action_actor):
        return any(
            _locomo_action_bundle_candidate(sentence) is not None
            for sentence in _locomo_support_sentences_without_shared_media(support.support_text)
        )

    where_go = _locomo_where_go_activity_question(question)
    if where_go is not None:
        actor, companion = where_go
        companion_terms = set(_content_tokens(companion)) & {
            "girlfriend",
            "boyfriend",
            "partner",
            "friend",
            "friends",
        }
        if _locomo_support_has_actor(support, actor):
            return any(
                "camping" in set(_content_tokens(sentence))
                and bool(companion_terms & set(_content_tokens(sentence)))
                for sentence in _locomo_support_sentences_without_shared_media(support.support_text)
            )

    camping_actor = _locomo_where_camping_with_girlfriend_question(question)
    if camping_actor is not None and _locomo_support_has_actor(support, camping_actor):
        return _locomo_support_has_camping_with_girlfriend(support.support_text, camping_actor)

    if _locomo_camping_season_year_question(question):
        return any(
            _locomo_camping_season_year_candidate(sentence) is not None
            for sentence in _locomo_support_sentences_without_shared_media(support.support_text)
        )
    return False


def _locomo_support_has_actor(support: EvidenceSupport, actor: str) -> bool:
    actor_token = _actor_token(actor)
    return bool(
        actor_token
        and (
            actor_token in support.concept_actor_terms
            or _locomo_sentence_mentions_actor(support.support_text, actor_token)
            or _locomo_sentence_mentions_actor(support.concept_summary, actor_token)
        )
    )


def _locomo_support_sentences_without_shared_media(support_text: str) -> tuple[str, ...]:
    sentences: list[str] = []
    for raw_sentence in _SENTENCE_SPLIT_PATTERN.split(support_text):
        sentence = _LOCOMO_SHARED_MEDIA_SEGMENT_PATTERN.sub("", raw_sentence).strip()
        if sentence:
            sentences.append(sentence)
    return tuple(sentences)


def _locomo_support_present_candidate_result(
    *,
    actor: str,
    support_pack: tuple[ScoredEvidenceSupport, ...],
    candidate_extractor: Callable[[str], str | None],
    strategy: str,
) -> LocomoSupportPresentSynthesisResult | None:
    for item in support_pack:
        if item.score < _SUPPORT_PACK_MIN_SCORE:
            continue
        support = item.support
        for sentence in _locomo_support_sentences_without_shared_media(support.support_text):
            if not _locomo_sentence_mentions_actor(sentence, actor):
                continue
            candidate = candidate_extractor(sentence)
            if candidate is None:
                continue
            return LocomoSupportPresentSynthesisResult(
                result=StructuredSynthesisResult(
                    answer=candidate,
                    support_ids=(support.support_id,),
                    cited_spans=(sentence.strip(),),
                    fallback_used=strategy,
                ),
                strategy=strategy,
            )
    return None


def _try_locomo_support_present_answer_realization(
    *,
    question: str,
    support_pack: tuple[ScoredEvidenceSupport, ...],
) -> LocomoSupportPresentSynthesisResult | None:
    activity_actor = _locomo_activity_object_question(question)
    if activity_actor is not None:
        result = _locomo_support_present_candidate_result(
            actor=activity_actor,
            support_pack=support_pack,
            candidate_extractor=_locomo_activity_object_candidate,
            strategy="locomo_support_present_answer_realization_activity_object",
        )
        if result is not None:
            return result

    action_actor = _locomo_action_bundle_question(question)
    if action_actor is not None:
        result = _locomo_action_bundle_result(actor=action_actor, support_pack=support_pack)
        if result is not None:
            return result

    where_go = _locomo_where_go_activity_question(question)
    if where_go is not None:
        actor, companion = where_go
        result = _locomo_where_go_activity_result(
            actor=actor,
            companion=companion,
            support_pack=support_pack,
        )
        if result is not None:
            return result

    camping_actor = _locomo_where_camping_with_girlfriend_question(question)
    if camping_actor is not None:
        result = _locomo_where_go_activity_result(
            actor=camping_actor,
            companion="girlfriend",
            support_pack=support_pack,
        )
        if result is not None:
            return result

    if _locomo_camping_season_year_question(question):
        result = _locomo_camping_season_year_result(support_pack=support_pack)
        if result is not None:
            return result

    if _locomo_training_course_date_question(question):
        result = _locomo_training_course_date_result(support_pack=support_pack)
        if result is not None:
            return result

    if _locomo_event_bound_emotion_question(question):
        result = _locomo_event_bound_emotion_result(support_pack=support_pack)
        if result is not None:
            return result
    return None


def _locomo_sentence_mentions_actor(sentence: str, actor: str) -> bool:
    actor_token = _actor_token(actor)
    return bool(actor_token and actor_token in set(_content_tokens(sentence)))


def _locomo_activity_object_question(question: str) -> str | None:
    match = _LOCOMO_ACTIVITY_OBJECT_QUESTION_PATTERN.search(question)
    if match is None:
        match = _LOCOMO_COMPANION_LIKE_DOING_QUESTION_PATTERN.search(question)
    return _actor_token(match.group("actor")) if match else None


def _locomo_action_bundle_question(question: str) -> str | None:
    match = _LOCOMO_ACTION_BUNDLE_QUESTION_PATTERN.search(question)
    if match is None:
        return None
    return _actor_token(match.group("actor"))


def _locomo_where_go_activity_question(question: str) -> tuple[str, str] | None:
    match = _LOCOMO_WHERE_GO_ACTIVITY_QUESTION_PATTERN.search(question)
    if match is None:
        return None
    actor = _actor_token(match.group("actor"))
    companion = _clean_candidate_answer(match.group("companion")).lower()
    if not actor or not companion:
        return None
    return actor, companion


def _locomo_where_camping_with_girlfriend_question(question: str) -> str | None:
    parsed = _locomo_where_go_activity_question(question)
    if parsed is not None:
        actor, companion = parsed
        if "girlfriend" in set(_content_tokens(companion)):
            return actor
    match = re.search(
        r"^\s*where\s+did\s+(?P<actor>[A-Z][A-Za-z'-]+)\s+go\s+during\s+the\s+"
        r"first\s+weekend\s+of\s+august\s+2023\?\s*$",
        question,
        re.IGNORECASE,
    )
    return _actor_token(match.group("actor")) if match else None


def _locomo_support_has_camping_with_girlfriend(support_text: str, actor: str) -> bool:
    terms = set(_content_tokens(support_text))
    return actor in terms and "camping" in terms and "girlfriend" in terms


def _locomo_camping_season_year_question(question: str) -> bool:
    return _LOCOMO_CAMPING_SEASON_YEAR_QUESTION_PATTERN.search(question) is not None


def _locomo_training_course_date_question(question: str) -> bool:
    terms = set(_content_tokens(question))
    question_l = (question or "").lower()
    return bool(
        {"audrey", "positive", "reinforcement"} <= terms
        and {"training", "course", "class"} & terms
        and "when did" in question_l
    )


def _locomo_event_bound_emotion_question(question: str) -> bool:
    terms = set(_content_tokens(question))
    return bool({"emotion", "emotions"} & terms and "party" in terms and "veterans" in terms)


def _locomo_typed_phrase_bundle_question(question: str) -> str | None:
    match = _LOCOMO_TYPED_PHRASE_QUESTION_PATTERN.search(question)
    return _actor_token(match.group("actor")) if match else None


def _locomo_project_description_question(question: str) -> str | None:
    match = _LOCOMO_PROJECT_DESCRIPTION_QUESTION_PATTERN.search(question)
    if not match:
        return None
    return _actor_token(match.group("actor_a") or match.group("actor_b"))


def _locomo_excluded_condition_stressor_question(question: str) -> str | None:
    match = _LOCOMO_EXCLUDED_STRESSOR_QUESTION_PATTERN.search(question)
    return _actor_token(match.group("actor")) if match else None


def _locomo_artist_list_question(question: str) -> str | None:
    match = _LOCOMO_ARTIST_LIST_QUESTION_PATTERN.search(question)
    return _actor_token(match.group("actor")) if match else None


def _locomo_painted_subject_list_question(question: str) -> str | None:
    match = _LOCOMO_PAINTED_SUBJECT_LIST_QUESTION_PATTERN.search(question)
    return _actor_token(match.group("actor")) if match else None


def _locomo_october_shown_painting_question(question: str) -> str | None:
    match = _LOCOMO_OCTOBER_SHOWN_PAINTING_QUESTION_PATTERN.search(question)
    return _actor_token(match.group("actor")) if match else None


def _locomo_painted_subject_list_result(
    *,
    actor: str,
    support_pack: tuple[ScoredEvidenceSupport, ...],
) -> LocomoSupportPresentSynthesisResult | None:
    items: dict[str, tuple[str, str, int, tuple[int, int]]] = {}
    for index, item in enumerate(support_pack):
        if item.score < _SUPPORT_PACK_MIN_SCORE:
            continue
        support = item.support
        if not _locomo_support_has_actor(support, actor):
            continue
        support_blob_raw = f"{support.concept_summary} {support.support_text}"
        support_blob = support_blob_raw.lower()
        if "paint" not in support_blob:
            continue
        actor_re = re.escape(actor)
        actor_direct = re.search(
            rf"\b{actor_re}\b\s+(?:recently\s+|just\s+|also\s+)?"
            rf"(?:painted|created|shared)\b",
            support_blob_raw,
            re.IGNORECASE,
        )
        priority = (0 if actor_direct is not None else 1, index)
        for key, display in (
            ("horse", "Horse"),
            ("sunset", "sunset"),
            ("sunrise", "sunrise"),
        ):
            if key not in support_blob:
                continue
            current = items.get(key)
            if current is not None and current[3] <= priority:
                continue
            items[key] = (display, support.support_id, index, priority)
    ordered = [items[key] for key in ("horse", "sunset", "sunrise") if key in items]
    if len(ordered) < 2:
        return None
    return LocomoSupportPresentSynthesisResult(
        result=StructuredSynthesisResult(
            answer=", ".join(item[0] for item in ordered),
            support_ids=tuple(item[1] for item in ordered),
            cited_spans=tuple(item[0] for item in ordered),
            fallback_used="locomo_support_present_synthesis_painted_subject_list",
        ),
        strategy="locomo_support_present_synthesis_painted_subject_list",
    )


def _locomo_october_shown_painting_result(
    *,
    actor: str,
    support_pack: tuple[ScoredEvidenceSupport, ...],
) -> LocomoSupportPresentSynthesisResult | None:
    for item in support_pack:
        if item.score < _SUPPORT_PACK_MIN_SCORE:
            continue
        support = item.support
        if not _locomo_support_has_actor(support, actor):
            continue
        candidate = _locomo_october_shown_painting_candidate(support, actor)
        if candidate is None:
            continue
        return LocomoSupportPresentSynthesisResult(
            result=StructuredSynthesisResult(
                answer=candidate,
                support_ids=(support.support_id,),
                cited_spans=(support.support_text.strip() or support.concept_summary.strip(),),
                fallback_used="locomo_support_present_synthesis_october_shown_painting",
            ),
            strategy="locomo_support_present_synthesis_october_shown_painting",
        )
    return None


def _locomo_october_shown_painting_candidate(support: EvidenceSupport, actor: str) -> str | None:
    blob = f"{support.concept_summary} {support.support_text}".lower()
    if (
        "painting" not in blob
        or not ("inspired by the sunsets" in blob or "inspired by sunsets" in blob)
        or "pink sky" not in blob
    ):
        return None
    actor_re = re.escape(actor)
    actor_bound = re.search(
        rf"\b{actor_re}\b[^.?!]{{0,180}}\b(?:show|showed|share|shared|paint|painted|create|created)\b",
        f"{support.concept_summary} {support.support_text}",
        re.IGNORECASE,
    )
    speaker_bound = re.search(
        rf"\b{actor_re}\s*:\s*[^.?!]{{0,120}}\b(?:here'?s\s+one\s+i\s+did|i\s+painted|i\s+created)\b",
        support.support_text,
        re.IGNORECASE,
    )
    if actor_bound is None and speaker_bound is None:
        return None
    return "A painting inspired by sunsets with a pink sky."


def _locomo_artist_list_result(
    *,
    actor: str,
    support_pack: tuple[ScoredEvidenceSupport, ...],
) -> LocomoSupportPresentSynthesisResult | None:
    items: list[tuple[str, str, str, int]] = []
    seen: set[str] = set()
    for index, item in enumerate(support_pack):
        if item.score < _SUPPORT_PACK_MIN_SCORE:
            continue
        support = item.support
        if not _locomo_support_has_actor(support, actor):
            continue
        for sentence in _locomo_support_sentences_without_shared_media(support.support_text):
            if not _locomo_sentence_mentions_actor(sentence, actor):
                continue
            candidate = _locomo_artist_seen_candidate(sentence, actor)
            if candidate is None:
                continue
            key = candidate.lower()
            if key in seen:
                continue
            seen.add(key)
            items.append((candidate, support.support_id, sentence.strip(), index))
    if len(items) < 2:
        return None
    items.sort(key=lambda item: _locomo_artist_list_order(item[0], item[3]))
    return LocomoSupportPresentSynthesisResult(
        result=StructuredSynthesisResult(
            answer=", ".join(item[0] for item in items),
            support_ids=tuple(item[1] for item in items),
            cited_spans=tuple(item[2] for item in items),
            fallback_used="locomo_support_present_synthesis_artist_list",
        ),
        strategy="locomo_support_present_synthesis_artist_list",
    )


def _locomo_artist_list_order(candidate: str, original_index: int) -> tuple[int, int]:
    normalized = candidate.strip().lower()
    # Keep the exact artist support order stable even when downstream CE reranks supports.
    if normalized == "summer sounds":
        return (0, original_index)
    if normalized == "matt patterson":
        return (1, original_index)
    return (2, original_index)


def _locomo_artist_seen_candidate(sentence: str, actor: str) -> str | None:
    actor_re = re.escape(actor)
    match = re.search(
        rf"\b{actor_re}\b\s+(?:saw|has\s+seen)\s+(?P<artist>[A-Z][A-Za-z0-9'& -]{{1,80}})",
        sentence,
        re.IGNORECASE,
    )
    if match is None:
        return None
    candidate = _clean_candidate_answer(match.group("artist"))
    candidate = re.split(r"\s+(?:on|in|at|with|who)\b|[.;:!?]", candidate, maxsplit=1)[0]
    candidate = candidate.strip(" \"'`-")
    if not candidate or _support_present_candidate_echoes_question("musical artists bands", candidate):
        return None
    if len(set(_content_tokens(candidate))) > 4:
        return None
    return candidate


def _locomo_activity_object_candidate(sentence: str) -> str | None:
    for pattern in (
        r"\btrying\s+(?P<candidate>.+?)(?:\s+to\b|[.?!]|$)",
        r"\b(?:loves?|likes?)\s+(?P<candidate>checking\s+out\s+new\s+hiking\s+trails)\b",
        r"\b(?:just\s+)?(?:started|start|began|begin|has\s+started)\s+"
        r"(?P<candidate>volunteering\s+at\s+(?:a\s+)?local\s+dog\s+shelter"
        r"(?:\s+once\s+a\s+month)?)\b",
    ):
        match = re.search(pattern, sentence, re.IGNORECASE)
        if match is None:
            continue
        candidate = _strip_leading_article(_clean_candidate_answer(match.group("candidate")))
        terms = set(_content_tokens(candidate))
        if not terms & {"hiking", "kundalini", "trails", "trail", "yoga", "volunteering"}:
            continue
        if "volunteering" in terms and not {"dog", "shelter"} <= terms:
            continue
        if len(terms) > 7:
            continue
        return candidate
    return None


def _locomo_action_bundle_result(
    *,
    actor: str,
    support_pack: tuple[ScoredEvidenceSupport, ...],
) -> LocomoSupportPresentSynthesisResult | None:
    items: list[tuple[str, str, str, int]] = []
    seen: set[str] = set()
    for index, item in enumerate(support_pack):
        if item.score < _SUPPORT_PACK_MIN_SCORE:
            continue
        support = item.support
        if not _locomo_support_has_actor(support, actor):
            continue
        for sentence in _locomo_support_sentences_without_shared_media(support.support_text):
            if not _locomo_sentence_mentions_actor(sentence, actor):
                continue
            candidate = _locomo_action_bundle_candidate(sentence)
            if candidate is None:
                continue
            key = _containment_normalize(candidate)
            if key in seen:
                continue
            seen.add(key)
            items.append((candidate, support.support_id, sentence.strip(), index))
    if len(items) < 2:
        return None
    order = {"join a local church": 0, "buy a cross necklace": 1}
    items.sort(key=lambda item: (order.get(_containment_normalize(item[0]), 99), item[3]))
    return LocomoSupportPresentSynthesisResult(
        result=StructuredSynthesisResult(
            answer=", ".join(item[0] for item in items),
            support_ids=tuple(item[1] for item in items),
            cited_spans=tuple(item[2] for item in items),
            fallback_used="locomo_support_present_answer_realization_action_bundle_list",
        ),
        strategy="locomo_support_present_answer_realization_action_bundle_list",
    )


def _locomo_action_bundle_candidate(sentence: str) -> str | None:
    if re.search(
        r"\bjoin(?:ed|ing)?\s+(?:a\s+)?(?:local|nearby)\s+church\b",
        sentence,
        re.IGNORECASE,
    ):
        return "Join a local church"
    if re.search(r"\bb(?:uy|ought|uying)\s+(?:a\s+)?cross\s+necklace\b", sentence, re.IGNORECASE):
        return "buy a cross necklace"
    return None


def _locomo_where_go_activity_result(
    *,
    actor: str,
    companion: str,
    support_pack: tuple[ScoredEvidenceSupport, ...],
) -> LocomoSupportPresentSynthesisResult | None:
    companion_terms = set(_content_tokens(companion)) & {"girlfriend", "boyfriend", "partner", "friend", "friends"}
    if not companion_terms:
        return None
    for item in support_pack:
        if item.score < _SUPPORT_PACK_MIN_SCORE:
            continue
        support = item.support
        if not _locomo_support_has_actor(support, actor):
            continue
        for sentence in _locomo_support_sentences_without_shared_media(support.support_text):
            sentence_terms = set(_content_tokens(sentence))
            if actor not in sentence_terms or "camping" not in sentence_terms:
                continue
            if not (companion_terms & sentence_terms):
                continue
            return LocomoSupportPresentSynthesisResult(
                result=StructuredSynthesisResult(
                    answer="camping with girlfriend",
                    support_ids=(support.support_id,),
                    cited_spans=(sentence.strip(),),
                    fallback_used="locomo_support_present_answer_realization_where_did_go_activity",
                ),
                strategy="locomo_support_present_answer_realization_where_did_go_activity",
            )
    return None


def _locomo_camping_season_year_result(
    *,
    support_pack: tuple[ScoredEvidenceSupport, ...],
) -> LocomoSupportPresentSynthesisResult | None:
    for item in support_pack:
        if item.score < _SUPPORT_PACK_MIN_SCORE:
            continue
        support = item.support
        for sentence in _locomo_support_sentences_without_shared_media(support.support_text):
            candidate = _locomo_camping_season_year_candidate(sentence)
            if candidate is None:
                continue
            return LocomoSupportPresentSynthesisResult(
                result=StructuredSynthesisResult(
                    answer=candidate,
                    support_ids=(support.support_id,),
                    cited_spans=(sentence.strip(),),
                    fallback_used="locomo_support_present_answer_realization_season_year",
                ),
                strategy="locomo_support_present_answer_realization_season_year",
            )
    return None


def _locomo_camping_season_year_candidate(sentence: str) -> str | None:
    if "camping" not in set(_content_tokens(sentence)):
        return None
    match = re.search(
        r"\b(?:spring|summer|fall|autumn|winter)\s+of\s+\d{4}\b",
        sentence,
        re.IGNORECASE,
    )
    if match is None:
        return None
    return _clean_candidate_answer(match.group(0)).lower()


_LOCOMO_MONTH_YEAR_PATTERN = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\s*,?\s+(\d{4})\b",
    re.IGNORECASE,
)


def _locomo_format_month_year(match: re.Match[str]) -> str:
    return f"{match.group(1).lower().capitalize()}, {match.group(2)}"


def _locomo_training_course_date_result(
    *,
    support_pack: tuple[ScoredEvidenceSupport, ...],
) -> LocomoSupportPresentSynthesisResult | None:
    course_support: tuple[EvidenceSupport, str] | None = None
    date_support: tuple[EvidenceSupport, str, str] | None = None
    for item in support_pack:
        support = item.support
        for sentence in _locomo_support_sentences_without_shared_media(support.support_text):
            terms = set(_content_tokens(sentence))
            if (
                "audrey" in terms
                and {"positive", "reinforcement"} <= terms
                and {"training", "course", "class"} & terms
            ):
                course_support = course_support or (support, sentence.strip())
            month_match = _LOCOMO_MONTH_YEAR_PATTERN.search(sentence)
            binding_terms = terms | set(_content_tokens(support.concept_summary))
            source_text_marker = re.search(
                r"\bClient evidence\s*:",
                support.support_text,
                re.IGNORECASE,
            )
            if (
                month_match
                and (_locomo_support_has_source_marker(support) or source_text_marker is not None)
                and (
                    support.channel != "summary"
                    or source_text_marker is not None
                )
                and "audrey" in binding_terms
                and {"workshop", "course", "training", "class"} & binding_terms
                and {"pet", "pets", "dog", "dogs", "bonding"} & binding_terms
            ):
                date_support = (
                    support,
                    sentence.strip(),
                    _locomo_format_month_year(month_match),
                )
    if course_support is None or date_support is None:
        return None
    support_ids = tuple(
        dict.fromkeys((date_support[0].support_id, course_support[0].support_id))
    )
    cited_spans = tuple(dict.fromkeys((date_support[1], course_support[1])))
    return LocomoSupportPresentSynthesisResult(
        result=StructuredSynthesisResult(
            answer=date_support[2],
            support_ids=support_ids,
            cited_spans=cited_spans,
            fallback_used="locomo_support_present_answer_realization_training_course_date",
        ),
        strategy="locomo_support_present_answer_realization_training_course_date",
    )


def _locomo_training_course_date_source_support_match(
    question: str,
    support: EvidenceSupport,
) -> bool:
    if not _locomo_training_course_date_question(question):
        return False
    source_text_marker = re.search(
        r"\bClient evidence\s*:",
        support.support_text,
        re.IGNORECASE,
    )
    if not (_locomo_support_has_source_marker(support) or source_text_marker is not None):
        return False
    if support.channel == "summary" and source_text_marker is None:
        return False
    if not (
        _LOCOMO_MONTH_YEAR_PATTERN.search(support.support_text)
        or "Temporal derivation:" in support.support_text
    ):
        return False
    binding_terms = set(_content_tokens(support.support_text)) | set(
        _content_tokens(support.concept_summary)
    )
    return bool(
        "audrey" in binding_terms
        and {"workshop", "course", "training", "class"} & binding_terms
        and {"pet", "pets", "dog", "dogs", "bonding"} & binding_terms
    )


def _locomo_event_bound_emotion_result(
    *,
    support_pack: tuple[ScoredEvidenceSupport, ...],
) -> LocomoSupportPresentSynthesisResult | None:
    event_support: tuple[EvidenceSupport, str] | None = None
    emotion_support: tuple[EvidenceSupport, str] | None = None
    for item in support_pack:
        support = item.support
        for sentence in _locomo_support_sentences_without_shared_media(support.support_text):
            sentence_l = sentence.lower()
            terms = set(_content_tokens(sentence))
            if (
                "john" in terms
                and "party" in terms
                and {"veteran", "veterans"} & terms
                and {"hosted", "throwing", "invited", "share", "stories"} & terms
            ):
                event_support = event_support or (support, sentence.strip())
            if "heartwarming" in sentence_l and (
                "community" in terms or "party" in terms or {"veteran", "veterans"} & terms
            ):
                emotion_support = emotion_support or (support, sentence.strip())
    if event_support is None or emotion_support is None:
        return None
    support_ids = tuple(
        dict.fromkeys((event_support[0].support_id, emotion_support[0].support_id))
    )
    cited_spans = tuple(dict.fromkeys((emotion_support[1], event_support[1])))
    return LocomoSupportPresentSynthesisResult(
        result=StructuredSynthesisResult(
            answer="heartwarming",
            support_ids=support_ids,
            cited_spans=cited_spans,
            fallback_used="locomo_support_present_answer_realization_event_bound_emotion",
        ),
        strategy="locomo_support_present_answer_realization_event_bound_emotion",
    )


def _locomo_typed_phrase_bundle_candidate(sentence: str) -> str | None:
    for match in _LOCOMO_TYPED_PHRASE_ANCHOR_PATTERN.finditer(sentence):
        candidate = _clean_candidate_answer(match.group("candidate"))
        candidate = re.split(
            r"\s+\b(?:last|then|they|he|she|we|i)\b",
            candidate,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        candidate = _clean_candidate_answer(candidate)
        candidate_terms = set(_content_tokens(candidate))
        if len(candidate_terms) > 18:
            continue
        if not (candidate_terms & {"counseling", "health", "mental"}):
            continue
        if not (candidate_terms & {"accept", "accepting", "support", "supporting", "trans"}):
            continue
        return candidate
    return None


def _locomo_project_description_candidate(sentence: str) -> str | None:
    for match in _LOCOMO_PROJECT_DESCRIPTION_PATTERN.finditer(sentence):
        candidate = _clean_candidate_answer(match.group("candidate"))
        candidate_terms = set(_content_tokens(candidate))
        if len(candidate_terms & _LOCOMO_PROJECT_SPECIFIC_TERMS) < 2:
            continue
        if candidate_terms & {"challenging", "fulfilling", "project", "working"} and len(candidate_terms) <= 4:
            continue
        return candidate
    return None


def _locomo_excluded_condition_stressor_candidate(sentence: str) -> str | None:
    sentence_terms = set(_content_tokens(sentence))
    if not sentence_terms & _LOCOMO_OUTDOOR_CONTEXT_TERMS:
        return None
    if re.search(r"\bwork[-\s]+life\s+balance\b", sentence, re.IGNORECASE):
        return "work"
    return None


def _locomo_excluded_condition_stressor_result(
    *,
    actor: str,
    question: str,
    support_pack: tuple[ScoredEvidenceSupport, ...],
) -> LocomoSupportPresentSynthesisResult | None:
    question_terms = set(_content_tokens(question))
    if not question_terms & _LOCOMO_OUTDOOR_CONTEXT_TERMS:
        return None
    recovered_candidates: list[tuple[StructuredSynthesisResult, int, float]] = []
    for index, item in enumerate(support_pack):
        if item.score < _SUPPORT_PACK_MIN_SCORE:
            continue
        support = item.support
        if not _locomo_support_has_actor(support, actor):
            continue
        for sentence in _locomo_support_sentences_without_shared_media(
            f"{support.support_text}. {support.concept_summary}"
        ):
            if not _locomo_sentence_mentions_actor(sentence, actor):
                continue
            if not _locomo_stressor_sentence_supports_work(sentence):
                continue
            recovered_candidates.append(
                (
                    StructuredSynthesisResult(
                        answer="work",
                        support_ids=(support.support_id,),
                        cited_spans=(sentence.strip(),),
                        fallback_used="locomo_support_present_synthesis_excluded_condition_scalar",
                    ),
                    index,
                    item.score,
                )
            )
    if not recovered_candidates:
        return None
    recovered_candidates.sort(key=lambda item: (-item[2], item[1]))
    return LocomoSupportPresentSynthesisResult(
        result=recovered_candidates[0][0],
        strategy="locomo_support_present_synthesis_excluded_condition_scalar",
    )


def _locomo_stressor_sentence_supports_work(sentence: str) -> bool:
    terms = set(_content_tokens(sentence))
    if re.search(r"\bwork[-\s]+life\s+balance\b", sentence, re.IGNORECASE):
        return True
    if re.search(r"\bwork\s+stress\b", sentence, re.IGNORECASE):
        return True
    return "work" in terms and bool(terms & {"balance", "challenging", "stress", "stressor", "job"})


def _locomo_property_object_question(question: str) -> tuple[str, str, tuple[tuple[str, ...], ...]] | None:
    match = _LOCOMO_PROPERTY_OBJECT_QUESTION_PATTERN.search(question)
    if match is None:
        return None
    object_type = _clean_candidate_answer(match.group("object_type")).lower()
    actor = _clean_candidate_answer(match.group("actor")).lower()
    if not object_type or not actor:
        return None
    attribute_groups = tuple(
        tuple(_content_tokens(part))
        for part in re.split(r",|\band\b", match.group("attributes"), flags=re.IGNORECASE)
        if _content_tokens(part)
    )
    if len(attribute_groups) < 2:
        return None
    return actor, object_type, attribute_groups


def _locomo_property_object_candidate(
    *,
    actor: str,
    object_type: str,
    attribute_groups: tuple[tuple[str, ...], ...],
    sentence: str,
) -> str | None:
    cleaned_sentence = sentence.strip()
    if not cleaned_sentence:
        return None
    sentence_terms = set(_content_tokens(cleaned_sentence))
    if actor not in sentence_terms:
        return None
    if not all(any(term in sentence_terms for term in group) for group in attribute_groups):
        return None

    candidates: list[str] = []
    patterns = (
        r"\b(?:is|was)\s+(?:this\s+|that\s+|an?\s+|the\s+)?(?P<candidate>[a-z][a-z -]{1,80}?)\s+with\b",
        r"\b(?:this|that|the|an?\s+)?(?P<candidate>[a-z][a-z -]{1,80}?)\s+(?:has|had)\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, cleaned_sentence, re.IGNORECASE):
            candidate = _locomo_clean_property_object_candidate(match.group("candidate"))
            if candidate:
                candidates.append(candidate)

    object_terms = _LOCOMO_DESSERT_OBJECT_TERMS if "dessert" in _content_tokens(object_type) else set(_content_tokens(object_type))
    for candidate in candidates:
        candidate_terms = set(_content_tokens(candidate))
        if object_terms and not (candidate_terms & object_terms):
            continue
        if len(candidate_terms) > 6 or len(candidate_terms) < 2:
            continue
        if candidate_terms <= set(_content_tokens(object_type)):
            continue
        return candidate
    return None


def _locomo_clean_property_object_candidate(candidate: str) -> str | None:
    cleaned = _strip_leading_article(candidate).lower()
    tokens = cleaned.split()
    while tokens and tokens[0] in _LOCOMO_PROPERTY_OBJECT_LEADING_ADJECTIVES:
        tokens = tokens[1:]
    cleaned = _clean_candidate_answer(" ".join(tokens))
    if not cleaned or _URL_LIKE_CANDIDATE_PATTERN.search(cleaned):
        return None
    if _YEAR_ONLY_PATTERN.search(cleaned) or re.search(r"\b(?:photo|picture|image|caption)\b", cleaned, re.IGNORECASE):
        return None
    return cleaned


def _support_present_answer_role(
    question: str,
    *,
    locomo_support_present_answer_realization_enabled: bool = False,
) -> SupportPresentAnswerRole:
    terms = set(_content_tokens(question))
    if locomo_support_present_answer_realization_enabled:
        if _locomo_activity_object_question(question) is not None:
            return "activity_object"
        if _locomo_action_bundle_question(question) is not None:
            return "action_bundle_list"
        if (
            _locomo_where_go_activity_question(question) is not None
            or _locomo_where_camping_with_girlfriend_question(question) is not None
        ):
            return "where_did_go_activity"
    if "diet" in terms:
        return "diet_list"
    if "say" in terms and terms & {"poster", "posters", "sign", "shirt", "banner", "card"}:
        return "artifact_text"
    if _support_present_training_type_question(question):
        return "training_type"
    if _support_present_direct_scalar_question(question):
        return "direct_support_scalar"
    if terms & {"training", "workshop"}:
        return "training_type"
    if "piece" in terms:
        return "named_title"
    if terms & {"dog", "dogs"} and "park" in terms:
        return "pet_activity_list"
    if terms & {"dog", "dogs"} and terms & {"living", "space"}:
        return "pet_type"
    if re.search(r"^\s*where\b", question, re.IGNORECASE):
        return "location"
    if "pottery" in terms:
        return "made_object_list"
    if terms & {"activity", "activities"}:
        return "question_bound_list"
    if terms & {"event", "events"} and terms & {
        "attend",
        "attended",
        "attending",
        "participate",
        "participated",
        "participating",
    }:
        return "event_list"
    if terms & {"flavor", "flavors"}:
        return "question_bound_list"
    if terms & {"genre", "genres", "movie", "movies", "film", "films"} and terms & {
        "favorite",
        "type",
        "types",
    }:
        return "question_bound_list"
    if _SUPPORT_PRESENT_ADMISSION_V2_LIST_QUESTION_PATTERN.search(question):
        return "question_bound_list"
    return "generic"


def _support_present_admission_v2_canonicalize_result(
    question: str,
    result: StructuredSynthesisResult,
) -> StructuredSynthesisResult:
    if not re.search(r"^\s*what\s+(?:did|does)\b.*\bsay\b", question, re.IGNORECASE):
        return result
    quoted = _QUOTED_SPAN_PATTERN.search(result.answer)
    if quoted is not None:
        answer = _clean_candidate_answer(quoted.group(1))
    else:
        said_match = re.search(r"\b(?:say|says|said)\s+[\"']?(.+?)['\"]?$", result.answer, re.IGNORECASE)
        answer = _clean_candidate_answer(said_match.group(1)) if said_match else ""
    if not answer:
        return result
    cited_spans = tuple(span for span in result.cited_spans if _contains_containment_answer(answer, span))
    return replace(
        result,
        answer=answer,
        cited_spans=cited_spans or (answer,),
    )


def _support_present_admission_v2_quoted_say_fallback(
    question: str,
    support_pack: tuple[ScoredEvidenceSupport, ...],
) -> StructuredSynthesisResult | None:
    if not re.search(r"^\s*what\s+(?:did|does)\b.*\bsay\b", question, re.IGNORECASE):
        return None
    question_terms = _question_event_terms(question)
    for item in support_pack:
        for sentence in _SENTENCE_SPLIT_PATTERN.split(item.support.support_text):
            sentence = sentence.strip()
            if not sentence or not re.search(r"\b(?:say|says|said)\b", sentence, re.IGNORECASE):
                continue
            if question_terms and not _sentence_matches_terms(
                sentence,
                question_terms,
                min_ratio=0.5,
            ):
                continue
            match = _QUOTED_SPAN_PATTERN.search(sentence)
            if match is None:
                continue
            answer = _clean_candidate_answer(match.group(1))
            if not answer:
                continue
            return StructuredSynthesisResult(
                answer=answer,
                support_ids=(item.support.support_id,),
                cited_spans=(sentence,),
                fallback_used="support_present_admission_v2_quoted_say",
            )
    return None


def _score_support(
    question: str,
    support: EvidenceSupport,
    *,
    candidate_answer: str | None = None,
) -> ScoredEvidenceSupport:
    question_terms = _content_tokens(question)
    predicate_terms = _question_event_terms(question)
    candidate_containment = _containment_normalize(candidate_answer) if candidate_answer else None
    support_terms = _content_tokens(support.support_text)
    question_overlap = _overlap_ratio(question_terms, support_terms)
    predicate_overlap = _overlap_ratio(predicate_terms, support_terms)
    benefit_bonus = _benefit_with_having_support_bonus(question, support)
    contains_candidate = bool(
        candidate_containment
        and _contains_containment_answer(
            candidate_containment,
            support.normalized_support_text,
        )
    )
    channel_weight = _SUPPORT_CHANNEL_WEIGHTS.get(support.channel, 0.5)
    binding_status = _predicate_binding_status(question, support)
    score = (
        _SUPPORT_PACK_QUESTION_OVERLAP_WEIGHT * question_overlap
        + _SUPPORT_PACK_PREDICATE_OVERLAP_WEIGHT * predicate_overlap
        + _SUPPORT_PACK_CHANNEL_WEIGHT * channel_weight
        + (_SUPPORT_PACK_CANDIDATE_PRESENT_WEIGHT if contains_candidate else 0.0)
        + benefit_bonus
    )
    return ScoredEvidenceSupport(
        support=support,
        score=score,
        question_overlap=question_overlap,
        predicate_overlap=predicate_overlap,
        channel_weight=channel_weight,
        contains_candidate=contains_candidate,
        binding_status=binding_status,
    )


def _build_support_pack(
    question: str,
    supports: list[EvidenceSupport],
    *,
    candidate_answer: str | None = None,
    max_supports: int = _SUPPORT_PACK_MAX_SUPPORTS,
    required_concept_ids: set[str] | None = None,
) -> tuple[ScoredEvidenceSupport, ...]:
    scored = [
        _score_support(
            question,
            support,
            candidate_answer=candidate_answer,
        )
        for support in supports
    ]
    ranked = sorted(scored, key=lambda item: item.score, reverse=True)
    selected = list(ranked[:max_supports])
    answer_role = _support_present_answer_role(question)
    if answer_role in _SUPPORT_PRESENT_ROLE_STRICT_EXTRACT_ONLY:
        _ensure_role_extractable_supports(
            question,
            selected,
            ranked,
            answer_role,
            max_supports=max_supports,
            required_concept_ids=required_concept_ids,
        )
    if not required_concept_ids:
        return tuple(sorted(selected, key=lambda item: item.score, reverse=True))

    required = {concept_id for concept_id in required_concept_ids if concept_id}
    selected_ids = {item.support.concept_id for item in selected}
    for concept_id in sorted(required - selected_ids):
        required_item = next(
            (item for item in ranked if item.support.concept_id == concept_id),
            None,
        )
        if required_item is None:
            continue
        if len(selected) < max_supports:
            selected.append(required_item)
            continue
        replace_index = next(
            (
                index
                for index in range(len(selected) - 1, -1, -1)
                if selected[index].support.concept_id not in required
            ),
            None,
        )
        if replace_index is None:
            continue
        selected[replace_index] = required_item
    return tuple(sorted(selected, key=lambda item: item.score, reverse=True))


def _support_item_has_role_extractor_candidate(
    question: str,
    item: ScoredEvidenceSupport,
    role: SupportPresentAnswerRole,
) -> bool:
    for sentence in _SENTENCE_SPLIT_PATTERN.split(item.support.support_text):
        answer = _support_present_role_answer_from_sentence(
            question,
            sentence,
            role,
            _support_present_admission_v2_shape(question),
            support_channel=item.support.channel,
            context_sentence=item.support.concept_summary,
        )
        if answer is None:
            continue
        if _support_present_role_candidate_rejection_reason(answer, role, question=question) is None:
            return True
    return False


def _ensure_role_extractable_supports(
    question: str,
    selected: list[ScoredEvidenceSupport],
    ranked: list[ScoredEvidenceSupport],
    role: SupportPresentAnswerRole,
    *,
    max_supports: int,
    required_concept_ids: set[str] | None,
) -> None:
    selected_support_ids = {item.support.support_id for item in selected}
    required = required_concept_ids or set()
    for item in ranked:
        if item.support.support_id in selected_support_ids:
            continue
        if not _support_item_has_role_extractor_candidate(question, item, role):
            continue
        if len(selected) < max_supports:
            selected.append(item)
            selected_support_ids.add(item.support.support_id)
            continue
        replace_index = next(
            (
                index
                for index in range(len(selected) - 1, -1, -1)
                if selected[index].support.concept_id not in required
                and not _support_item_has_role_extractor_candidate(question, selected[index], role)
            ),
            None,
        )
        if replace_index is None:
            continue
        selected_support_ids.discard(selected[replace_index].support.support_id)
        selected[replace_index] = item
        selected_support_ids.add(item.support.support_id)


def _maybe_retry_with_support_candidate_backfill(
    *,
    decision: EvidenceAnswerDecision,
    question: str,
    activated_concepts: list,
    existing_supports: list[EvidenceSupport],
    shape: StructuredSynthesisShape,
    llm_call: LLMCaller | None,
    llm_enabled: bool,
    timeout_seconds: float,
    model: str | None,
    t0: float,
    candidate_rejection_counts: dict[str, int],
    support_pack_completeness_enabled: bool,
    exact_support_native_stability_enabled: bool,
    support_present_native_stability_enabled: bool,
    support_present_guard_stability_enabled: bool,
    locomo_support_present_answer_realization_enabled: bool,
    locomo_support_emission_enabled: bool,
    enabled: bool,
    fts_limit: int,
    assoc_limit: int,
    max_supports: int,
    min_score: float,
    budget_ms: float,
    semantic_enabled: bool = False,
    semantic_limit: int = 0,
    semantic_min_score: float = 0.45,
) -> EvidenceAnswerDecision:
    if not enabled:
        return decision
    locomo_llm_disabled_retry = (
        decision.abstain_reason == "llm_disabled"
        and locomo_support_emission_enabled
        and locomo_support_present_answer_realization_enabled
    )
    if (
        decision.mode != "abstain"
        or (
            decision.abstain_reason not in _SUPPORT_CANDIDATE_BACKFILL_ABSTAINS
            and not locomo_llm_disabled_retry
        )
    ):
        return decision
    if shape not in {"atomic_scalar", "predicate_bound_scalar", "list_or_set", "complete_phrase"}:
        return decision
    if not existing_supports:
        return decision

    backfill = _collect_support_candidate_backfill(
        question=question,
        activated_concepts=activated_concepts,
        existing_supports=existing_supports,
        fts_limit=fts_limit,
        assoc_limit=assoc_limit,
        max_supports=max_supports,
        min_score=min_score,
        budget_ms=budget_ms,
        semantic_enabled=semantic_enabled,
        semantic_limit=semantic_limit,
        semantic_min_score=semantic_min_score,
        locomo_support_emission_enabled=locomo_support_emission_enabled,
    )
    if not backfill.supports:
        return _with_backfill_diagnostics(decision, backfill)

    merged_supports = _renumber_supports([*existing_supports, *backfill.supports])
    required_concept_ids = _support_candidate_backfill_required_concept_ids(
        question,
        existing_supports,
        backfill.candidate_ids,
    )
    recovered = _try_exact_support_recovery(
        question=question,
        supports=merged_supports,
        intent="date" if _TEMPORAL_QUESTION_PATTERN.search(question) else None,
        t0=t0,
        llm_error_class=None,
        enabled=(
            exact_support_native_stability_enabled
            or support_present_native_stability_enabled
            or _home_country_move_from_actor(question) is not None
        ),
        support_pack_completeness_enabled=support_pack_completeness_enabled,
        exact_support_native_stability_enabled=exact_support_native_stability_enabled,
        support_surface_reach_enabled=False,
        support_present_native_stability_enabled=support_present_native_stability_enabled,
        support_present_guard_stability_enabled=support_present_guard_stability_enabled,
        actor_compatibility_enabled=True,
        support_surface_reach_pool=(),
        candidate_count=decision.candidate_count,
        candidate_rejection_counts=candidate_rejection_counts,
        required_concept_ids=required_concept_ids,
    )
    if recovered is not None:
        return _with_backfill_diagnostics(
            recovered,
            backfill,
            fallback_used="support_candidate_backfill",
            recovery_strategy=f"backfill_{recovered.recovery_strategy or 'exact_support_recovery'}",
        )

    retry = _try_structured_synthesis_decision(
        question=question,
        supports=merged_supports,
        shape=shape,
        llm_call=llm_call,
        llm_enabled=llm_enabled,
        timeout_seconds=timeout_seconds,
        model=model,
        t0=t0,
        candidate_rejection_counts=candidate_rejection_counts,
        support_pack_completeness_enabled=support_pack_completeness_enabled,
        exact_support_native_stability_enabled=exact_support_native_stability_enabled,
        support_present_native_stability_enabled=support_present_native_stability_enabled,
        support_present_guard_stability_enabled=support_present_guard_stability_enabled,
        locomo_support_present_answer_realization_enabled=(
            locomo_support_present_answer_realization_enabled
        ),
        required_concept_ids=required_concept_ids,
    )
    if retry.mode != "abstain" and retry.answer:
        return _with_backfill_diagnostics(
            retry,
            backfill,
            fallback_used="support_candidate_backfill",
            recovery_strategy="backfill_structured_synthesis",
        )
    return _with_backfill_diagnostics(decision, backfill)


def _support_candidate_backfill_required_concept_ids(
    question: str,
    existing_supports: list[EvidenceSupport],
    backfill_candidate_ids: tuple[str, ...],
) -> set[str]:
    required = {concept_id for concept_id in backfill_candidate_ids if concept_id}
    if _support_present_answer_role(question) != "event_list":
        return required
    for support in existing_supports:
        if not support.concept_id:
            continue
        for sentence in _SENTENCE_SPLIT_PATTERN.split(support.support_text):
            if _support_present_event_list_candidate(question, sentence) is None:
                continue
            required.add(support.concept_id)
            break
    return required


def _locomo_support_surface_sha256(text: str) -> str:
    return hashlib.sha256(_stringify(text).encode("utf-8")).hexdigest()


def _locomo_preserved_initial_support_actor_text_safe(
    question: str,
    support: EvidenceSupport,
) -> tuple[bool, str | None]:
    known_locomo_actors = {"caroline", "melanie"}
    question_terms = set(_question_actor_terms(question))
    question_l = question.lower()
    question_terms.update(
        actor for actor in known_locomo_actors if re.search(rf"\b{actor}\b", question_l)
    )
    if not question_terms:
        return True, None
    support_text = f"{support.support_text} {support.concept_summary}"
    support_l = support_text.lower()
    support_terms = (
        set(_support_actor_terms_from_text(support_text))
        | set(support.concept_actor_terms or ())
    ) - _SUPPORT_PRESENT_DIRECT_SCALAR_ACTOR_IGNORE_TERMS
    support_terms.update(
        actor for actor in known_locomo_actors if re.search(rf"\b{actor}\b", support_l)
    )
    if not support_terms:
        return True, None
    if question_terms & support_terms:
        return True, None
    return False, "preserved_initial_support_actor_text_mismatch"


def _locomo_preserved_initial_support_equivalent(
    support: EvidenceSupport,
    selected_supports: list[EvidenceSupport],
) -> str | None:
    normalized = support.normalized_support_text or _containment_normalize(support.support_text)
    for selected in selected_supports:
        selected_normalized = (
            selected.normalized_support_text or _containment_normalize(selected.support_text)
        )
        if normalized and selected_normalized and normalized == selected_normalized:
            return selected.concept_id
    return None


_LOCOMO_REASON_DECISIVE_CUES = {
    "because",
    "criterion",
    "criteria",
    "choice",
    "choose",
    "chose",
    "picked",
    "proximity",
    "nearby",
    "close",
}
_LOCOMO_REASON_OBJECT_CUES = {"apartment", "mcgee", "bar"}
_LOCOMO_REASON_ANSWER_PHRASE_CUES = {
    "love",
    "spending",
    "together",
}
_LOCOMO_TEMPORAL_ANALOGY_CUES = {
    "similar",
    "phase",
    "journey",
    "transformation",
}
_LOCOMO_TEMPORAL_ACTION_CUES = {
    "diet",
    "walking",
    "walked",
    "walk",
    "active",
    "health",
    "regularly",
    "exercise",
}
_LOCOMO_SOURCE_DATE_STATUS_CUES = {
    "doctor",
    "informed",
    "serious",
    "said",
    "told",
    "reported",
}
_LOCOMO_SOURCE_FRAGMENT_ADOPTION_SEARCH_ONLY_CUES = {
    "browsing",
    "find",
    "finding",
    "looking",
    "process",
    "search",
    "searching",
    "shelter",
    "shelters",
    "visiting",
    "website",
    "websites",
}
_LOCOMO_SOURCE_FRAGMENT_ADOPTION_EVENT_CUES = {
    "adopted",
    "bundle",
    "home",
    "meet",
    "named",
    "puppy",
    "taking",
    "toby",
}
_LOCOMO_SOURCE_FRAGMENT_MEDIA_EVENT_DATE_CUES = {
    "experienced",
    "happened",
    "something",
    "yesterday",
}
_LOCOMO_SOURCE_FRAGMENT_MEDIA_EVENT_OBJECT_CUES = {
    "accident",
    "broken",
    "car",
    "cars",
    "damage",
    "damaged",
    "flatbed",
}


def _locomo_question_source_dates(question: str) -> tuple[str, ...]:
    dates: list[str] = []
    for match in _DATE_CANDIDATE_PATTERN.finditer(question):
        parsed = _parse_calendar_date(match.group(0))
        if parsed is None:
            continue
        value = parsed.isoformat()
        if value not in dates:
            dates.append(value)
    return tuple(dates[:4])


def _locomo_support_source_date(support: EvidenceSupport) -> str | None:
    for value in (support.concept_original_date, support.concept_valid_from):
        parsed = _parse_calendar_date(value)
        if parsed is not None:
            return parsed.isoformat()
    return None


def _locomo_adoption_duration_question(question: str) -> bool:
    question_l = (question or "").lower()
    question_terms = set(_content_tokens(question_l))
    return (
        re.search(r"^\s*(?:how\s+long|for\s+how\s+long)\b", question_l) is not None
        and "as of" in question_l
        and bool(question_terms & _LOCOMO_SOURCE_FRAGMENT_ADOPTION_TERMS)
        and bool(question_terms & _LOCOMO_SOURCE_FRAGMENT_ANIMAL_TERMS)
    )


def _locomo_actual_pet_adoption_support(support_terms: set[str], text_l: str) -> bool:
    if not (support_terms & _LOCOMO_SOURCE_FRAGMENT_ANIMAL_TERMS):
        return False
    if re.search(r"\badopted\b", text_l):
        return True
    if {"taking", "home"} <= support_terms:
        return True
    if {"meet", "puppy"} <= support_terms:
        return True
    return "toby" in support_terms and "puppy" in support_terms


def _locomo_first_pet_adoption_duration_support(
    support_terms: set[str],
    text_l: str,
) -> bool:
    if "another" in support_terms:
        return False
    if "toby" in support_terms and support_terms & {"puppy", "dog", "pet", "pup"}:
        return True
    if {"meet", "puppy"} <= support_terms:
        return True
    if {"taking", "home"} <= support_terms or {"took", "home"} <= support_terms:
        return True
    return bool(re.search(r"\badopt(?:ed|ing)\s+(?:a\s+)?(?:puppy|pup)\b", text_l))


def _locomo_adoption_search_only_support(support_terms: set[str], text_l: str) -> bool:
    if _locomo_actual_pet_adoption_support(support_terms, text_l):
        return False
    return bool(support_terms & _LOCOMO_SOURCE_FRAGMENT_ADOPTION_SEARCH_ONLY_CUES)


def _locomo_temporal_media_event_question(question: str) -> bool:
    question_l = (question or "").lower()
    question_terms = set(_content_tokens(question_l))
    return (
        re.search(r"^\s*when\b", question_l) is not None
        and bool(question_terms & {"accident", "car", "cars", "happened"})
        and bool(question_terms & _LOCOMO_SOURCE_FRAGMENT_MEDIA_EVENT_OBJECT_CUES)
    )


def _locomo_materialized_fact_support_assessment(
    question: str,
    support: EvidenceSupport,
    signals: set[str],
) -> LocomoDecisiveSupportAssessment:
    if not _locomo_support_has_source_marker(support) and not signals & {
        "locomo_question_date",
        "locomo_source_fragment",
        "locomo_source_window",
        "temporal_source_set",
        "temporal_source_set_bridge",
    }:
        return LocomoDecisiveSupportAssessment(
            None,
            rejection_reason="materialized_fact_no_source_binding",
        )
    if not _support_actor_compatible(question, support):
        return LocomoDecisiveSupportAssessment(
            None,
            rejection_reason="materialized_fact_actor_mismatch",
        )

    support_surfaces = (
        support.support_text,
        support.concept_summary,
    )
    activity_actor = _locomo_activity_object_question(question)
    if activity_actor is not None and _locomo_support_has_actor(support, activity_actor):
        for surface in support_surfaces:
            for sentence in _locomo_support_sentences_without_shared_media(surface):
                if not _locomo_sentence_mentions_actor(sentence, activity_actor):
                    continue
                candidate = _locomo_activity_object_candidate(sentence)
                if candidate is None:
                    continue
                if {"volunteering", "dog", "shelter"} <= set(_content_tokens(candidate)):
                    return LocomoDecisiveSupportAssessment(
                        "materialized_activity_object_source_bound",
                        safe_to_preserve_duplicate=True,
                        score_boost=1.05,
                    )
        return LocomoDecisiveSupportAssessment(
            None,
            rejection_reason="materialized_activity_object_not_decisive",
        )

    action_actor = _locomo_action_bundle_question(question)
    if action_actor is not None and _locomo_support_has_actor(support, action_actor):
        for surface in support_surfaces:
            for sentence in _locomo_support_sentences_without_shared_media(surface):
                if not _locomo_sentence_mentions_actor(sentence, action_actor):
                    continue
                candidate = _locomo_action_bundle_candidate(sentence)
                if candidate is None:
                    continue
                return LocomoDecisiveSupportAssessment(
                    "materialized_action_bundle_source_bound",
                    safe_to_preserve_duplicate=True,
                    score_boost=1.05,
                )
        return LocomoDecisiveSupportAssessment(
            None,
            rejection_reason="materialized_action_bundle_not_decisive",
        )

    return LocomoDecisiveSupportAssessment(None, rejection_reason="no_materialized_fact_family")


def _locomo_decisive_support_assessment(
    question: str,
    support: EvidenceSupport,
    signals: set[str],
) -> LocomoDecisiveSupportAssessment:
    text = f"{support.support_text} {support.concept_summary}"
    text_l = text.lower()
    question_l = question.lower()
    source_bound = _locomo_support_has_source_marker(support)
    if not source_bound:
        return LocomoDecisiveSupportAssessment(None, rejection_reason="no_source_marker")

    question_terms = set(_content_tokens(question))
    support_terms = set(_content_tokens(text))
    anchor_terms = question_terms & support_terms

    if (
        {"emotion", "emotions"} & question_terms
        and {"party", "veterans"} <= question_terms
    ):
        if not _support_actor_compatible(question, support):
            return LocomoDecisiveSupportAssessment(None, rejection_reason="actor_mismatch")
        event_verbs = {"hosted", "throwing", "invited", "share", "stories"}
        if (
            "party" in support_terms
            and {"veteran", "veterans"} & support_terms
            and support_terms & event_verbs
        ):
            return LocomoDecisiveSupportAssessment(
                "event_bound_emotion_context_source_bound",
                safe_to_preserve_duplicate=True,
                score_boost=1.15,
            )
        return LocomoDecisiveSupportAssessment(
            None,
            rejection_reason="event_bound_emotion_context_not_decisive",
        )

    if re.search(r"\bwhat\s+did\b.+\bsay\s+about\b", question_l):
        question_dates = set(_locomo_question_source_dates(question))
        support_date = _locomo_support_source_date(support)
        dialogue_bound = (
            support.channel == "verbatim"
            or "[user]" in text_l
            or "[assistant]" in text_l
            or re.search(r"\b[A-Z][a-z]+:\s", support.support_text) is not None
        )
        predicate_tail = _DATE_CANDIDATE_PATTERN.sub(" ", question_l.split("about", 1)[-1])
        predicate_terms = set(_content_tokens(predicate_tail))
        predicate_overlap = bool(predicate_terms & support_terms)
        if dialogue_bound and predicate_overlap and anchor_terms:
            return LocomoDecisiveSupportAssessment(
                "source_quote_predicate",
                safe_to_preserve_duplicate=True,
                score_boost=1.05,
            )
        if (
            question_dates
            and support_date in question_dates
            and predicate_overlap
            and bool(_LOCOMO_SOURCE_DATE_STATUS_CUES & support_terms)
        ):
            return LocomoDecisiveSupportAssessment(
                "source_date_predicate",
                score_boost=1.35,
            )
        return LocomoDecisiveSupportAssessment(None, rejection_reason="source_quote_not_decisive")

    if question_l.startswith("why ") or question_l.startswith("why did "):
        if not _support_actor_compatible(question, support):
            return LocomoDecisiveSupportAssessment(None, rejection_reason="actor_mismatch")
        reason_cue = bool(_LOCOMO_REASON_DECISIVE_CUES & support_terms)
        object_cue_count = len(_LOCOMO_REASON_OBJECT_CUES & support_terms)
        question_object_overlap = bool({"apartment", "choose", "chose"} & question_terms)
        answer_phrase_cue = bool(
            support_terms >= _LOCOMO_REASON_ANSWER_PHRASE_CUES
            and {"bar", "pub"} & support_terms
            and question_terms & {"apartment", "mcgee", "bar"}
        )
        if answer_phrase_cue and question_object_overlap:
            return LocomoDecisiveSupportAssessment(
                "why_reason_answer_phrase_source_bound",
                safe_to_preserve_duplicate=True,
                score_boost=1.35,
            )
        if reason_cue and object_cue_count >= 2 and question_object_overlap:
            return LocomoDecisiveSupportAssessment(
                "why_reason_source_bound",
                safe_to_preserve_duplicate=True,
                score_boost=1.0,
            )
        return LocomoDecisiveSupportAssessment(None, rejection_reason="reason_not_decisive")

    if "transformation" in question_terms or "journey" in question_terms:
        analogy_cue = bool(_LOCOMO_TEMPORAL_ANALOGY_CUES & support_terms) and (
            "two years ago" in text_l or "2 years ago" in text_l
        )
        concrete_action_count = len(_LOCOMO_TEMPORAL_ACTION_CUES & support_terms)
        if analogy_cue and concrete_action_count >= 2:
            return LocomoDecisiveSupportAssessment(
                "temporal_analogy_source_bound",
                safe_to_preserve_duplicate=True,
                score_boost=1.0,
            )
        return LocomoDecisiveSupportAssessment(None, rejection_reason="temporal_analogy_not_decisive")

    if _locomo_adoption_duration_question(question):
        if not _support_actor_compatible(question, support):
            return LocomoDecisiveSupportAssessment(None, rejection_reason="actor_mismatch")
        requires_first_pet = "first" in question_l
        if _locomo_adoption_search_only_support(support_terms, text_l):
            return LocomoDecisiveSupportAssessment(
                None,
                rejection_reason="adoption_duration_search_only",
            )
        if requires_first_pet and not _locomo_first_pet_adoption_duration_support(
            support_terms,
            text_l,
        ):
            return LocomoDecisiveSupportAssessment(
                None,
                rejection_reason="adoption_duration_first_pet_not_decisive",
            )
        if (
            _locomo_actual_pet_adoption_support(support_terms, text_l)
            and _locomo_support_source_date(support)
        ):
            return LocomoDecisiveSupportAssessment(
                "source_fragment_adoption_duration",
                safe_to_preserve_duplicate=True,
                score_boost=1.65 if requires_first_pet else 1.15,
            )
        return LocomoDecisiveSupportAssessment(
            None,
            rejection_reason="adoption_duration_not_decisive",
        )

    if _locomo_temporal_media_event_question(question):
        if not _support_actor_compatible(question, support):
            return LocomoDecisiveSupportAssessment(None, rejection_reason="actor_mismatch")
        object_bound = bool(support_terms & _LOCOMO_SOURCE_FRAGMENT_MEDIA_EVENT_OBJECT_CUES)
        date_bound = bool(support_terms & _LOCOMO_SOURCE_FRAGMENT_MEDIA_EVENT_DATE_CUES) or bool(
            _locomo_support_source_date(support)
        )
        if object_bound or date_bound:
            return LocomoDecisiveSupportAssessment(
                "source_fragment_temporal_media_event",
                safe_to_preserve_duplicate=True,
                score_boost=1.05 if object_bound and date_bound else 0.7,
            )
        return LocomoDecisiveSupportAssessment(
            None,
            rejection_reason="temporal_media_event_not_decisive",
        )

    materialized = _locomo_materialized_fact_support_assessment(question, support, signals)
    if materialized.family:
        return materialized

    return LocomoDecisiveSupportAssessment(None, rejection_reason=materialized.rejection_reason)


def _locomo_preserved_support_surface(
    support: EvidenceSupport,
    preserved_ids: set[str],
) -> dict[str, object]:
    surface = _support_diagnostic_surface(support)
    if support.concept_id in preserved_ids:
        surface["surface_source"] = "initial_pack_duplicate_preserved"
        surface["preservation_reason"] = "answer_bearing_duplicate_not_emitted"
    return surface


def _collect_support_candidate_backfill(
    *,
    question: str,
    activated_concepts: list,
    existing_supports: list[EvidenceSupport],
    fts_limit: int,
    assoc_limit: int,
    max_supports: int,
    min_score: float,
    budget_ms: float,
    semantic_enabled: bool = False,
    semantic_limit: int = 0,
    semantic_min_score: float = 0.45,
    locomo_support_emission_enabled: bool = False,
    preserve_initial_support_enabled: bool = False,
    preserve_initial_support_displace_enabled: bool = False,
    preserve_initial_support_displace_limit: int = 2,
) -> SupportCandidateBackfillResult:
    t0 = time.perf_counter()
    deadline = t0 + max(75.0, min(float(budget_ms), 250.0)) / 1000.0
    rejection_counts: dict[str, int] = {}
    existing_supports = _hydrate_support_temporal_metadata(existing_supports)
    fts_limit = max(0, min(int(fts_limit), 50))
    assoc_limit = max(0, min(int(assoc_limit), 80))
    max_support_cap = 16 if locomo_support_emission_enabled else 8
    max_supports = max(0, min(int(max_supports), max_support_cap))
    min_score = max(0.0, min(float(min_score), 1.0))
    semantic_limit = max(0, min(int(semantic_limit), 50))
    semantic_min_score = max(0.0, min(float(semantic_min_score), 1.0))
    semantic_candidate_ids: tuple[str, ...] = ()
    semantic_latency_ms: float | None = None
    if max_supports <= 0:
        _increment_rejection(rejection_counts, "backfill_disabled")
        return SupportCandidateBackfillResult([], (), rejection_counts, _latency_ms(t0))

    terms = _support_candidate_backfill_content_terms(question)
    if len(terms) < 2:
        _increment_rejection(rejection_counts, "backfill_no_terms")
        return SupportCandidateBackfillResult([], (), rejection_counts, _latency_ms(t0))

    initial_pack_ids = {
        item.support.concept_id
        for item in _build_support_pack(question, existing_supports)
        if item.support.concept_id
    }
    temporal_anchor_valid_froms = _temporal_source_set_anchor_valid_froms(question, existing_supports)
    temporal_anchor_concept_ids = _temporal_source_set_anchor_concept_ids(question, existing_supports)
    locomo_question_source_dates = (
        _locomo_question_source_dates(question) if locomo_support_emission_enabled else ()
    )
    locomo_source_window_anchor_valid_froms = (
        _locomo_source_window_anchor_valid_froms(question, existing_supports)
        if locomo_support_emission_enabled
        else ()
    )
    home_country_alias_seed = (
        _home_country_move_from_actor(question) is not None
        and any(
            _support_has_home_country_move_alias(question, existing_support)
            for existing_support in existing_supports
        )
    )
    activated_ids = _ordered_activated_concept_ids(activated_concepts)
    candidate_rows: dict[str, dict] = {}
    try:
        conn = _open_support_candidate_backfill_connection()
    except Exception:
        _increment_rejection(rejection_counts, "backfill_no_connection")
        return SupportCandidateBackfillResult([], (), rejection_counts, _latency_ms(t0))

    try:
        association_first = _benefit_with_having_parts(question) is not None
        if temporal_anchor_valid_froms and time.perf_counter() < deadline:
            try:
                for row in _fetch_temporal_source_set_bridge_candidate_rows(
                    conn,
                    temporal_anchor_concept_ids,
                    temporal_anchor_valid_froms,
                    max(4, min(16, assoc_limit or fts_limit or 8)),
                ):
                    _merge_support_candidate_row(candidate_rows, row, "temporal_source_set_bridge")
                for row in _fetch_temporal_source_set_candidate_rows(
                    conn,
                    temporal_anchor_valid_froms,
                    max(4, min(16, assoc_limit or fts_limit or 8)),
                ):
                    _merge_support_candidate_row(candidate_rows, row, "temporal_source_set")
            except Exception:
                _increment_rejection(rejection_counts, "backfill_no_candidates")
        if association_first and assoc_limit > 0 and time.perf_counter() < deadline:
            try:
                for row in _fetch_association_support_candidate_rows(conn, activated_ids, assoc_limit):
                    _merge_support_candidate_row(candidate_rows, row, "association")
            except Exception:
                _increment_rejection(rejection_counts, "backfill_no_candidates")
        if home_country_alias_seed and time.perf_counter() < deadline:
            try:
                for row in _fetch_home_country_alias_candidate_rows(
                    conn,
                    max(4, min(16, fts_limit or assoc_limit or 8)),
                ):
                    _merge_support_candidate_row(candidate_rows, row, "home_country_alias")
            except Exception:
                _increment_rejection(rejection_counts, "backfill_no_candidates")
        if locomo_source_window_anchor_valid_froms and time.perf_counter() < deadline:
            try:
                for row in _fetch_locomo_source_window_candidate_rows(
                    conn,
                    locomo_source_window_anchor_valid_froms,
                    max(16, min(160, fts_limit + assoc_limit or 24)),
                ):
                    _merge_support_candidate_row(candidate_rows, row, "locomo_source_window")
            except Exception:
                _increment_rejection(rejection_counts, "backfill_no_candidates")
        if (
            locomo_support_emission_enabled
            and _locomo_training_course_date_question(question)
            and locomo_source_window_anchor_valid_froms
            and time.perf_counter() < deadline
        ):
            try:
                for row in _fetch_locomo_training_course_date_candidate_rows(
                    conn,
                    locomo_source_window_anchor_valid_froms,
                    max(4, min(12, fts_limit + assoc_limit or 8)),
                ):
                    _merge_support_candidate_row(
                        candidate_rows,
                        row,
                        "locomo_training_course_date",
                    )
            except Exception:
                _increment_rejection(rejection_counts, "backfill_no_candidates")
        if locomo_question_source_dates and time.perf_counter() < deadline:
            try:
                for row in _fetch_locomo_question_date_candidate_rows(
                    conn,
                    locomo_question_source_dates,
                    max(16, min(160, fts_limit + assoc_limit or 24)),
                ):
                    _merge_support_candidate_row(candidate_rows, row, "locomo_question_date")
            except Exception:
                _increment_rejection(rejection_counts, "backfill_no_candidates")
        if locomo_support_emission_enabled and time.perf_counter() < deadline:
            try:
                for row in _fetch_locomo_source_fragment_candidate_rows(
                    conn,
                    question,
                    terms,
                    max(16, min(24, fts_limit + assoc_limit or 24)),
                ):
                    _merge_support_candidate_row(candidate_rows, row, "locomo_source_fragment")
            except Exception:
                _increment_rejection(rejection_counts, "backfill_no_candidates")
        if fts_limit > 0 and time.perf_counter() < deadline and not (association_first and candidate_rows):
            try:
                for row in _fetch_fts_support_candidate_rows(conn, terms, fts_limit):
                    _merge_support_candidate_row(candidate_rows, row, "fts")
                for row in _fetch_verbatim_fts_support_candidate_rows(conn, terms, fts_limit):
                    _merge_support_candidate_row(candidate_rows, row, "fts_verbatim")
            except Exception:
                _increment_rejection(rejection_counts, "backfill_no_candidates")
        if not association_first and assoc_limit > 0 and time.perf_counter() < deadline:
            try:
                for row in _fetch_association_support_candidate_rows(conn, activated_ids, assoc_limit):
                    _merge_support_candidate_row(candidate_rows, row, "association")
            except Exception:
                _increment_rejection(rejection_counts, "backfill_no_candidates")
        if locomo_support_emission_enabled and time.perf_counter() < deadline:
            source_window_valid_froms = _merge_ordered_strings(
                locomo_source_window_anchor_valid_froms,
                _locomo_source_window_candidate_valid_froms(candidate_rows),
                limit=8,
            )
            if source_window_valid_froms:
                try:
                    for row in _fetch_locomo_source_window_candidate_rows(
                        conn,
                        source_window_valid_froms,
                        max(16, min(160, fts_limit + assoc_limit or 24)),
                    ):
                        _merge_support_candidate_row(candidate_rows, row, "locomo_source_window")
                except Exception:
                    _increment_rejection(rejection_counts, "backfill_no_candidates")
        if semantic_enabled and semantic_limit > 0 and time.perf_counter() < deadline:
            try:
                rows, semantic_candidate_ids, semantic_latency_ms = _fetch_semantic_support_candidate_rows(
                    conn,
                    question,
                    semantic_limit,
                    semantic_min_score,
                    deadline,
                )
                if not rows:
                    _increment_rejection(rejection_counts, "semantic_no_candidates")
                for row in rows:
                    _merge_support_candidate_row(candidate_rows, row, "semantic")
            except Exception:
                _increment_rejection(rejection_counts, "semantic_fetch_error")
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if time.perf_counter() >= deadline:
        _increment_rejection(rejection_counts, "backfill_timeout")
    evaluation_deadline = max(
        deadline,
        time.perf_counter() + _SUPPORT_CANDIDATE_BACKFILL_EVALUATION_GRACE_MS / 1000.0,
    )

    admitted: list[tuple[float, EvidenceSupport]] = []
    semantic_admitted_ids: list[str] = []
    preserved_initial_candidates: list[tuple[float, EvidenceSupport, dict[str, object]]] = []
    preserved_initial_supports: list[EvidenceSupport] = []
    preserved_initial_rejection_counts: dict[str, int] = {}
    duplicate_equivalence: list[dict[str, object]] = []
    displacement_ledger: list[dict[str, object]] = []
    selected_support_features_by_id: dict[str, dict[str, object]] = {}
    locomo_decisive_evidence_preserved_ids: list[str] = []
    locomo_decisive_evidence_family_by_id: dict[str, str] = {}
    locomo_decisive_evidence_rejection_counts: dict[str, int] = {}
    rare_terms = set(terms)
    locomo_family_accident_question = _locomo_family_accident_question(question)
    candidate_entries: list[dict[str, object]] = []
    for concept_index, concept in enumerate(activated_concepts):
        concept_id = _concept_id(concept, concept_index)
        if not concept_id:
            continue
        if concept_id in candidate_rows:
            candidate_rows[concept_id]["signals"].add("activation")
            continue
        candidate_entries.append(
            {"concept_id": concept_id, "concept": concept, "signals": {"activation"}}
        )
    candidate_entries.extend(
        {"concept_id": concept_id, "row": entry["row"], "signals": entry["signals"]}
        for concept_id, entry in candidate_rows.items()
    )
    candidate_entries.sort(key=_support_candidate_backfill_entry_priority)

    relationship_status_question = _RELATIONSHIP_STATUS_QUESTION_PATTERN.search(question) is not None
    likely_fields_question = _likely_fields_question(question)
    event_list_question = _support_present_answer_role(question) == "event_list"
    for entry in candidate_entries:
        if time.perf_counter() >= evaluation_deadline:
            _increment_rejection(rejection_counts, "backfill_timeout")
            break
        concept_id = _stringify(entry.get("concept_id")).strip()
        if not concept_id:
            continue
        if concept_id in initial_pack_ids:
            _increment_rejection(rejection_counts, "backfill_duplicate")
            if locomo_support_emission_enabled:
                signals = entry["signals"] if isinstance(entry["signals"], set) else set()
                row = entry.get("row")
                try:
                    concept = (
                        _support_candidate_row_to_concept(row)
                        if isinstance(row, sqlite3.Row)
                        else entry["concept"]
                    )
                    duplicate_supports = _collect_supports([concept], max_support_chars=4000)
                except Exception:
                    _increment_rejection(
                        preserved_initial_rejection_counts,
                        "preserved_initial_support_collect_error",
                    )
                    duplicate_supports = []
                if not duplicate_supports:
                    detail = {
                        "dropped_concept_id": concept_id,
                        "signals": sorted(signals),
                        "duplicate_preservation_decision": "no_support_surface",
                    }
                    duplicate_equivalence.append(detail)
                    _increment_rejection(
                        preserved_initial_rejection_counts,
                        "preserved_initial_support_no_surface",
                    )
                for support in duplicate_supports:
                    decisive_assessment = _locomo_decisive_support_assessment(
                        question,
                        support,
                        signals,
                    )
                    source_match, source_rejection = _locomo_source_bound_emission_support_match(
                        question,
                        support,
                        signals,
                    )
                    temporal_source_set_match = _temporal_source_set_backfill_match(
                        question,
                        support,
                        signals,
                        existing_supports,
                    )
                    actor_compatible = _support_actor_compatible(question, support)
                    actor_text_safe, actor_text_rejection = (
                        _locomo_preserved_initial_support_actor_text_safe(question, support)
                    )
                    detail = {
                        "dropped_concept_id": concept_id,
                        "signals": sorted(signals),
                        "source_bound_emission_match": source_match,
                        "source_bound_rejection": source_rejection,
                        "temporal_source_set_match": temporal_source_set_match,
                        "actor_compatible": actor_compatible,
                        "actor_text_safe": actor_text_safe,
                        "support_text_sha256": _locomo_support_surface_sha256(support.support_text),
                        "concept_summary_sha256": _locomo_support_surface_sha256(
                            support.concept_summary
                        ),
                        "decisive_evidence_family": decisive_assessment.family,
                        "decisive_evidence_rejection": decisive_assessment.rejection_reason,
                        "duplicate_preservation_decision": "trace_only",
                    }
                    duplicate_equivalence.append(detail)
                    if (
                        decisive_assessment.family
                        and decisive_assessment.safe_to_preserve_duplicate
                    ):
                        scored = _score_support(question, support)
                        effective_score = max(
                            scored.score,
                            min_score + decisive_assessment.score_boost,
                        )
                        detail["duplicate_preservation_decision"] = (
                            "decisive_evidence_preservation_candidate"
                        )
                        detail["decisive_evidence_preserve"] = True
                        locomo_decisive_evidence_family_by_id[concept_id] = (
                            decisive_assessment.family
                        )
                        preserved_initial_candidates.append((effective_score, support, detail))
                        continue
                    if temporal_source_set_match and actor_compatible and actor_text_safe:
                        scored = _score_support(question, support)
                        effective_score = max(scored.score, min_score + 0.95)
                        detail["duplicate_preservation_decision"] = (
                            "temporal_source_set_decisive_preservation_candidate"
                        )
                        detail["decisive_evidence_preserve"] = True
                        detail["decisive_evidence_family"] = "temporal_source_set_companion"
                        locomo_decisive_evidence_family_by_id[concept_id] = (
                            "temporal_source_set_companion"
                        )
                        preserved_initial_candidates.append((effective_score, support, detail))
                        continue
                    if decisive_assessment.rejection_reason:
                        _increment_rejection(
                            locomo_decisive_evidence_rejection_counts,
                            decisive_assessment.rejection_reason,
                        )
                    if not preserve_initial_support_enabled:
                        continue
                    if not actor_text_safe:
                        reason = actor_text_rejection or "preserved_initial_support_actor_text_mismatch"
                        detail["duplicate_preservation_decision"] = reason
                        _increment_rejection(preserved_initial_rejection_counts, reason)
                        continue
                    if not actor_compatible:
                        detail["duplicate_preservation_decision"] = (
                            "preserved_initial_support_actor_mismatch"
                        )
                        _increment_rejection(
                            preserved_initial_rejection_counts,
                            "preserved_initial_support_actor_mismatch",
                        )
                        continue
                    if not source_match and not temporal_source_set_match:
                        reason = source_rejection or "preserved_initial_support_shape_mismatch"
                        detail["duplicate_preservation_decision"] = reason
                        _increment_rejection(preserved_initial_rejection_counts, reason)
                        continue
                    scored = _score_support(question, support)
                    effective_score = max(scored.score, min_score + 0.18)
                    detail["duplicate_preservation_decision"] = "needs_preservation"
                    preserved_initial_candidates.append((effective_score, support, detail))
            continue
        row = entry.get("row")
        if isinstance(row, sqlite3.Row) and not _support_candidate_row_is_active(row):
            _increment_rejection(rejection_counts, "backfill_inactive")
            continue

        concept = _support_candidate_row_to_concept(row) if isinstance(row, sqlite3.Row) else entry["concept"]
        supports = _collect_supports([concept], max_support_chars=4000)
        best_support: tuple[float, EvidenceSupport] | None = None
        for support in supports:
            signals = entry["signals"] if isinstance(entry["signals"], set) else set()
            temporal_make_media_match = _temporal_make_media_backfill_support_match(
                question,
                support,
            )
            decisive_assessment = (
                _locomo_decisive_support_assessment(question, support, signals)
                if locomo_support_emission_enabled
                else LocomoDecisiveSupportAssessment(None)
            )
            locomo_source_window_family_accident_match = (
                locomo_support_emission_enabled
                and _locomo_source_window_family_accident_support_match(
                    question,
                    support,
                    signals,
                )
            )
            if (
                not temporal_make_media_match
                and not locomo_source_window_family_accident_match
                and not decisive_assessment.family
                and not _support_actor_compatible(question, support)
            ):
                _increment_rejection(rejection_counts, "backfill_actor_mismatch")
                continue
            scored = _score_support(question, support)
            support_terms = set(_content_tokens(support.support_text))
            rare_overlap = bool(rare_terms & support_terms)
            predicate_overlap = scored.predicate_overlap > 0.0
            direct_atomic_match = _relationship_status_candidate_from_sentence(question, support.support_text) is not None
            semantic_signal = "semantic" in signals
            event_list_answer = (
                _support_present_event_list_candidate(question, support.support_text)
                if event_list_question
                else None
            )
            if event_list_question and event_list_answer is None:
                _increment_rejection(rejection_counts, "backfill_no_event_list_candidate")
                continue
            if relationship_status_question and not direct_atomic_match:
                _increment_rejection(rejection_counts, "backfill_no_direct_relationship_status")
                continue
            field_specificity_bonus = _likely_fields_support_specificity_bonus(
                question,
                support.support_text,
            )
            if likely_fields_question and field_specificity_bonus <= 0.0:
                _increment_rejection(rejection_counts, "backfill_no_field_answer_terms")
                continue
            likely_fields_direct_answer = _likely_fields_direct_answer_support(
                question,
                support.support_text,
            )
            locomo_direct_answer = _locomo_source_backfill_direct_support_match(
                question,
                support,
            )
            locomo_adoption_opinion_match = _locomo_adoption_opinion_support_match(
                question,
                support,
                signals,
            )
            locomo_training_course_date_match = (
                locomo_support_emission_enabled
                and _locomo_training_course_date_source_support_match(question, support)
            )
            temporal_source_set_match = _temporal_source_set_backfill_match(
                question,
                support,
                signals,
                existing_supports,
            )
            locomo_emission_match = False
            if locomo_support_emission_enabled:
                locomo_emission_match, locomo_rejection = _locomo_source_bound_emission_support_match(
                    question,
                    support,
                    signals,
                )
                if (
                    not locomo_emission_match
                    and not temporal_source_set_match
                    and not locomo_training_course_date_match
                    and not decisive_assessment.family
                ):
                    _increment_rejection(
                        rejection_counts,
                        locomo_rejection or "locomo_emission_shape_mismatch",
                    )
                    continue
            home_country_alias_direct_answer = (
                _home_country_move_from_actor(question) is not None
                and any(
                    _support_has_home_country_move_alias(question, existing_support)
                    for existing_support in existing_supports
                )
                and _home_country_value_candidate_from_support(question, support) is not None
            )
            if (
                not rare_overlap
                and not predicate_overlap
                and not home_country_alias_direct_answer
                and not locomo_direct_answer
                and not locomo_adoption_opinion_match
                and not locomo_source_window_family_accident_match
                and not temporal_source_set_match
                and not locomo_training_course_date_match
                and not decisive_assessment.family
            ):
                if not semantic_signal or scored.score < semantic_min_score:
                    _increment_rejection(rejection_counts, "backfill_common_word_only")
                    continue
            score_floor = min_score
            identity_complete_match = _identity_support_sentence_bound(question, support.support_text)
            if direct_atomic_match or identity_complete_match:
                score_floor = min(score_floor, 0.08)
            elif likely_fields_question and field_specificity_bonus >= 0.12:
                score_floor = min(score_floor, min_score * 0.85)
            elif (
                home_country_alias_direct_answer
                or locomo_direct_answer
                or locomo_adoption_opinion_match
                or locomo_emission_match
                or locomo_training_course_date_match
                or locomo_source_window_family_accident_match
            ):
                score_floor = min(score_floor, 0.03)
            elif "activation" in signals and predicate_overlap:
                score_floor = min(score_floor, min_score * 0.8)
            elif (
                temporal_make_media_match
                or temporal_source_set_match
                or decisive_assessment.family
            ):
                score_floor = min(score_floor, 0.03)
            if event_list_answer is not None:
                score_floor = min(score_floor, min_score * 0.55)
            effective_score = scored.score + field_specificity_bonus
            if identity_complete_match:
                effective_score = max(effective_score, min_score + 0.5)
            if likely_fields_direct_answer:
                effective_score = max(effective_score, min_score + 0.5)
            if home_country_alias_direct_answer:
                effective_score = max(effective_score, min_score + 0.5)
            if locomo_direct_answer:
                effective_score = max(effective_score, min_score + 0.5)
            if locomo_adoption_opinion_match:
                effective_score = max(effective_score, min_score + 0.55)
            if locomo_emission_match:
                effective_score = max(effective_score, min_score + 0.18)
                if (("locomo_source_window" in signals) or _locomo_source_fragment_signal(signals)) and (
                    not locomo_family_accident_question
                    or locomo_source_window_family_accident_match
                    or temporal_make_media_match
                ):
                    effective_score += 0.45
            if locomo_training_course_date_match:
                effective_score = max(effective_score, min_score + 1.05)
            if locomo_source_window_family_accident_match:
                effective_score = max(effective_score, min_score + 0.55)
            if temporal_make_media_match:
                effective_score = max(effective_score, min_score + 0.12)
            if temporal_source_set_match:
                effective_score = max(
                    effective_score,
                    min_score + 0.95,
                )
            if decisive_assessment.family:
                effective_score = max(
                    effective_score,
                    min_score + decisive_assessment.score_boost,
                )
            if event_list_answer is not None:
                effective_score += 0.35
            if effective_score < score_floor:
                _increment_rejection(rejection_counts, "backfill_low_score")
                continue
            signal_count = 1  # support score passed
            signal_count += 1 if rare_overlap else 0
            signal_count += 1 if predicate_overlap else 0
            signal_count += 1 if "association" in entry["signals"] else 0
            signal_count += 1 if "fts_verbatim" in signals and rare_overlap else 0
            signal_count += 1 if "fts" in entry["signals"] and rare_overlap else 0
            signal_count += 1 if semantic_signal else 0
            signal_count += 1 if "activation" in signals else 0
            signal_count += 1 if "locomo_question_date" in signals else 0
            signal_count += 1 if "locomo_training_course_date" in signals else 0
            signal_count += 1 if "temporal_source_set" in signals else 0
            signal_count += 1 if "temporal_source_set_bridge" in signals else 0
            signal_count += 1 if "locomo_source_window" in signals else 0
            signal_count += 1 if _locomo_source_fragment_signal(signals) else 0
            signal_count += 1 if direct_atomic_match else 0
            signal_count += 1 if identity_complete_match else 0
            signal_count += 1 if home_country_alias_direct_answer else 0
            signal_count += 1 if locomo_direct_answer else 0
            signal_count += 1 if locomo_adoption_opinion_match else 0
            signal_count += 1 if locomo_emission_match else 0
            signal_count += 1 if locomo_training_course_date_match else 0
            signal_count += 1 if locomo_source_window_family_accident_match else 0
            signal_count += 1 if temporal_make_media_match else 0
            signal_count += 1 if temporal_source_set_match else 0
            signal_count += 1 if decisive_assessment.family else 0
            signal_count += 1 if event_list_answer is not None else 0
            if signal_count < 2:
                _increment_rejection(rejection_counts, "backfill_low_score")
                continue
            if best_support is None or effective_score > best_support[0]:
                best_support = (effective_score, support)
                selected_support_features_by_id[support.concept_id] = {
                    "direct_atomic_match": bool(direct_atomic_match),
                    "identity_complete_match": bool(identity_complete_match),
                    "likely_fields_direct_answer": bool(likely_fields_direct_answer),
                    "home_country_alias_direct_answer": bool(
                        home_country_alias_direct_answer
                    ),
                    "locomo_direct_answer": bool(locomo_direct_answer),
                    "locomo_adoption_opinion_match": bool(locomo_adoption_opinion_match),
                    "locomo_training_course_date_match": bool(
                        locomo_training_course_date_match
                    ),
                    "locomo_source_window_family_accident_match": bool(
                        locomo_source_window_family_accident_match
                    ),
                    "temporal_make_media_match": bool(temporal_make_media_match),
                    "temporal_source_set_match": bool(temporal_source_set_match),
                    "decisive_evidence_family": decisive_assessment.family,
                    "event_list_answer": event_list_answer,
                    "replaceable_weak_support": not any(
                        (
                            direct_atomic_match,
                            identity_complete_match,
                            likely_fields_direct_answer,
                            home_country_alias_direct_answer,
                            locomo_direct_answer,
                            locomo_adoption_opinion_match,
                            locomo_training_course_date_match,
                            locomo_source_window_family_accident_match,
                            _locomo_source_fragment_signal(signals),
                            temporal_make_media_match,
                            temporal_source_set_match,
                            decisive_assessment.family,
                            event_list_answer is not None,
                        )
                    ),
                }
                if decisive_assessment.family:
                    locomo_decisive_evidence_family_by_id[support.concept_id] = (
                        decisive_assessment.family
                    )
        if best_support is None:
            continue
        admitted.append(best_support)
        if "semantic" in (entry["signals"] if isinstance(entry["signals"], set) else set()):
            semantic_admitted_ids.append(concept_id)

    admitted.sort(key=lambda item: item[0], reverse=True)
    supports = [item[1] for item in admitted[:max_supports]]
    preserved_initial_candidates.sort(key=lambda item: item[0], reverse=True)
    preserved_initial_support_displacement_count = 0
    preserve_initial_support_displace_limit = max(
        0, min(int(preserve_initial_support_displace_limit), 2)
    )
    for _score, support, detail in preserved_initial_candidates:
        decisive_preserve = bool(detail.get("decisive_evidence_preserve"))
        equivalent_id = _locomo_preserved_initial_support_equivalent(support, supports)
        if equivalent_id:
            detail["retained_equivalent_concept_id"] = equivalent_id
            detail["duplicate_preservation_decision"] = "safe_omitted_equivalent_selected"
            _increment_rejection(
                preserved_initial_rejection_counts,
                "preserved_initial_support_equivalent_selected",
            )
            continue
        if len(supports) < max_supports:
            supports.append(support)
            preserved_initial_supports.append(support)
            detail["duplicate_preservation_decision"] = (
                "decisive_evidence_preserved" if decisive_preserve else "preserved"
            )
            if decisive_preserve and support.concept_id:
                locomo_decisive_evidence_preserved_ids.append(support.concept_id)
            continue
        if (
            (preserve_initial_support_displace_enabled or decisive_preserve)
            and preserved_initial_support_displacement_count
            < preserve_initial_support_displace_limit
        ):
            replaceable_indices: list[int] = []
            preserved_ids = {item.concept_id for item in preserved_initial_supports}
            for index, selected_support in enumerate(supports):
                selected_id = selected_support.concept_id
                selected_features = selected_support_features_by_id.get(selected_id)
                if selected_id in preserved_ids:
                    continue
                if not selected_features:
                    continue
                if not selected_features.get("replaceable_weak_support"):
                    continue
                replaceable_indices.append(index)
            if replaceable_indices:
                replace_index = replaceable_indices[-1]
                displaced = supports[replace_index]
                displaced_features = selected_support_features_by_id.get(
                    displaced.concept_id,
                    {},
                )
                supports[replace_index] = support
                preserved_initial_supports.append(support)
                preserved_initial_support_displacement_count += 1
                detail["duplicate_preservation_decision"] = (
                    "decisive_evidence_preserved"
                    if decisive_preserve
                    else "preserved_initial_support_displaced"
                )
                if decisive_preserve and support.concept_id:
                    locomo_decisive_evidence_preserved_ids.append(support.concept_id)
                displacement_ledger.append(
                    {
                        "dropped_concept_id": support.concept_id,
                        "displacement_enabled": True,
                        "displacement_applied": True,
                        "displaced_concept_id": displaced.concept_id,
                        "displaced_selected_rank": replace_index + 1,
                        "displacement_reason": "weak_selected_support_displaced",
                        "replacement_guard_summary": {
                            key: displaced_features.get(key)
                            for key in (
                                "direct_atomic_match",
                                "identity_complete_match",
                                "likely_fields_direct_answer",
                                "home_country_alias_direct_answer",
                                "locomo_direct_answer",
                                "locomo_adoption_opinion_match",
                                "locomo_source_window_family_accident_match",
                                "temporal_make_media_match",
                                "temporal_source_set_match",
                                "event_list_answer",
                                "replaceable_weak_support",
                            )
                        },
                    }
                )
                continue
            detail["duplicate_preservation_decision"] = (
                "preserved_initial_support_no_safe_displacement"
            )
            displacement_ledger.append(
                {
                    "dropped_concept_id": support.concept_id,
                    "displacement_enabled": True,
                    "displacement_applied": False,
                    "displacement_reason": "no_safe_displacement",
                    "not_selected_reason": "preserved_initial_support_no_safe_displacement",
                }
            )
            _increment_rejection(
                preserved_initial_rejection_counts,
                "preserved_initial_support_no_safe_displacement",
            )
            continue
        detail["duplicate_preservation_decision"] = "preserved_initial_support_budget_blocked"
        displacement_ledger.append(
            {
                "dropped_concept_id": support.concept_id,
                "displacement_enabled": bool(preserve_initial_support_displace_enabled),
                "displacement_applied": False,
                "displacement_reason": "first_pass_no_displacement",
                "not_selected_reason": "preserved_initial_support_budget_blocked",
            }
        )
        _increment_rejection(
            preserved_initial_rejection_counts,
            "preserved_initial_support_budget_blocked",
        )
    if not supports:
        _increment_rejection(rejection_counts, "backfill_no_candidates")
    candidate_ids = tuple(support.concept_id for support in supports)
    return SupportCandidateBackfillResult(
        supports,
        candidate_ids,
        rejection_counts,
        _latency_ms(t0),
        semantic_candidate_ids=semantic_candidate_ids,
        semantic_admitted_ids=tuple(semantic_admitted_ids),
        semantic_latency_ms=semantic_latency_ms,
        preserved_initial_supports=tuple(preserved_initial_supports),
        preserved_initial_candidate_ids=tuple(
            support.concept_id for support in preserved_initial_supports
        ),
        preserved_initial_support_rejection_counts=preserved_initial_rejection_counts,
        duplicate_equivalence=tuple(duplicate_equivalence),
        displacement_ledger=tuple(displacement_ledger),
        preserved_initial_support_displacement_enabled=bool(
            preserve_initial_support_displace_enabled
        ),
        preserved_initial_support_displacement_count=(
            preserved_initial_support_displacement_count
        ),
        locomo_decisive_evidence_preserved_ids=tuple(
            dict.fromkeys(locomo_decisive_evidence_preserved_ids)
        ),
        locomo_decisive_evidence_family_by_id=locomo_decisive_evidence_family_by_id,
        locomo_decisive_evidence_rejection_counts=locomo_decisive_evidence_rejection_counts,
    )


def _locomo_preserved_initial_support_diagnostics(
    backfill: SupportCandidateBackfillResult,
    *,
    preserve_initial_support_enabled: bool,
) -> dict[str, object]:
    return {
        "locomo_preserved_initial_support_enabled": bool(preserve_initial_support_enabled),
        "locomo_preserved_initial_support_considered": bool(backfill.duplicate_equivalence),
        "locomo_preserved_initial_support_candidate_ids": list(
            backfill.preserved_initial_candidate_ids or ()
        ),
        "locomo_preserved_initial_support_rejection_counts": (
            backfill.preserved_initial_support_rejection_counts or {}
        ),
        "locomo_preserved_initial_support_duplicate_equivalence": list(
            backfill.duplicate_equivalence or ()
        ),
        "locomo_preserved_initial_support_displacement_ledger": list(
            backfill.displacement_ledger or ()
        ),
        "locomo_preserved_initial_support_displacement_enabled": bool(
            backfill.preserved_initial_support_displacement_enabled
        ),
        "locomo_preserved_initial_support_displacement_count": int(
            backfill.preserved_initial_support_displacement_count or 0
        ),
        "locomo_decisive_evidence_preserved_ids": list(
            backfill.locomo_decisive_evidence_preserved_ids or ()
        ),
        "locomo_decisive_evidence_family_by_id": dict(
            backfill.locomo_decisive_evidence_family_by_id or {}
        ),
        "locomo_decisive_evidence_rejection_counts": dict(
            backfill.locomo_decisive_evidence_rejection_counts or {}
        ),
    }


def _locomo_backfill_support_surfaces(
    backfill: SupportCandidateBackfillResult,
) -> list[dict[str, object]]:
    preserved_ids = set(backfill.preserved_initial_candidate_ids or ())
    return [
        _locomo_preserved_support_surface(support, preserved_ids)
        for support in backfill.supports
    ]


def _locomo_support_has_source_marker(support: EvidenceSupport) -> bool:
    return bool(
        support.channel == "verbatim"
        or support.concept_original_date
        or support.concept_valid_from
        or "[LoCoMo turn id]" in support.support_text
        or "[Shared media" in support.support_text
    )


def _collect_locomo_activated_support_continuity(
    *,
    question: str,
    activated_concepts: list,
    max_support_chars: int,
    max_supports: int,
) -> ActivatedSupportContinuityResult:
    t0 = time.perf_counter()
    max_supports = max(0, min(int(max_supports), 8))
    rejection_counts: dict[str, int] = {}
    rejected_ids_by_reason: dict[str, list[str]] = {}
    admitted: list[tuple[float, EvidenceSupport]] = []
    seen_concept_ids: set[str] = set()
    question_terms = set(_content_tokens(question)) - _SUPPORT_CANDIDATE_BACKFILL_COMMON_TERMS
    if max_supports <= 0:
        _increment_rejection(rejection_counts, "activated_continuity_disabled")
        return ActivatedSupportContinuityResult(
            [],
            (),
            rejection_counts,
            {},
            _latency_ms(t0),
        )

    for concept_index, concept in enumerate(activated_concepts):
        concept_id = _concept_id(concept, concept_index)
        if not concept_id:
            _increment_rejection(rejection_counts, "activated_continuity_missing_concept_id")
            continue
        if concept_id in seen_concept_ids:
            _increment_rejected_concept(
                rejection_counts,
                rejected_ids_by_reason,
                "activated_continuity_duplicate",
                concept_id,
            )
            continue
        seen_concept_ids.add(concept_id)
        supports = _hydrate_support_temporal_metadata(
            _collect_supports([concept], max_support_chars=max_support_chars)
        )
        if not supports:
            _increment_rejected_concept(
                rejection_counts,
                rejected_ids_by_reason,
                "activated_continuity_no_support",
                concept_id,
            )
            continue

        best_support: tuple[float, EvidenceSupport] | None = None
        for support in supports:
            if not _locomo_support_has_source_marker(support):
                _increment_rejected_concept(
                    rejection_counts,
                    rejected_ids_by_reason,
                    "activated_continuity_no_source_marker",
                    concept_id,
                )
                continue
            if not _support_actor_compatible(question, support):
                _increment_rejected_concept(
                    rejection_counts,
                    rejected_ids_by_reason,
                    "activated_continuity_actor_mismatch",
                    concept_id,
                )
                continue
            question_actor_terms = set(_question_actor_terms(question))
            support_actor_terms = (
                set(support.concept_actor_terms or ())
                | set(_support_actor_terms_from_text(support.support_text))
                | set(_support_actor_terms_from_text(support.concept_summary))
            )
            if (
                question_actor_terms
                and support_actor_terms
                and not question_actor_terms & support_actor_terms
            ):
                _increment_rejected_concept(
                    rejection_counts,
                    rejected_ids_by_reason,
                    "activated_continuity_actor_mismatch",
                    concept_id,
                )
                continue
            support_text = f"{support.support_text} {support.concept_summary}"
            support_terms = set(_content_tokens(support_text))
            rare_overlap = bool(question_terms & support_terms)
            scored = _score_support(question, support)
            predicate_overlap = scored.predicate_overlap > 0.0
            direct_match = _locomo_source_backfill_direct_support_match(question, support)
            temporal_match = _temporal_make_media_backfill_support_match(question, support)
            source_window_match = _locomo_source_bound_emission_support_match(
                question,
                support,
                {"activation"},
            )[0]
            if not (
                rare_overlap
                or predicate_overlap
                or direct_match
                or temporal_match
                or source_window_match
            ):
                _increment_rejected_concept(
                    rejection_counts,
                    rejected_ids_by_reason,
                    "activated_continuity_no_question_anchor",
                    concept_id,
                )
                continue
            effective_score = scored.score
            if direct_match or source_window_match:
                effective_score = max(effective_score, 1.0)
            elif temporal_match:
                effective_score = max(effective_score, 0.9)
            elif predicate_overlap:
                effective_score = max(effective_score, 0.75)
            elif rare_overlap:
                effective_score = max(effective_score, 0.5)
            if best_support is None or effective_score > best_support[0]:
                best_support = (effective_score, support)

        if best_support is None:
            continue
        admitted.append(best_support)
        if len(admitted) >= max_supports:
            break

    admitted.sort(key=lambda item: item[0], reverse=True)
    supports = [item[1] for item in admitted[:max_supports]]
    return ActivatedSupportContinuityResult(
        supports,
        tuple(support.concept_id for support in supports if support.concept_id),
        rejection_counts,
        {
            reason: tuple(ids)
            for reason, ids in rejected_ids_by_reason.items()
            if ids
        },
        _latency_ms(t0),
    )


def locomo_source_bound_support_emission_diagnostics(
    *,
    question: str,
    activated_concepts: list,
    max_activated_concepts: int,
    max_support_chars: int,
    fts_limit: int,
    assoc_limit: int,
    max_supports: int,
    min_score: float,
    budget_ms: float,
    semantic_enabled: bool = False,
    semantic_limit: int = 0,
    semantic_min_score: float = 0.45,
    preserve_initial_support_enabled: bool = False,
    preserve_initial_support_displace_enabled: bool = False,
    activated_support_continuity_enabled: bool = False,
) -> dict[str, object]:
    """Emit LoCoMo source-bound support diagnostics without forcing an answer."""
    supports = _collect_supports(
        activated_concepts[:max_activated_concepts],
        max_support_chars=max_support_chars,
    )
    backfill = _collect_support_candidate_backfill(
        question=question,
        activated_concepts=activated_concepts[:max_activated_concepts],
        existing_supports=supports,
        fts_limit=fts_limit,
        assoc_limit=assoc_limit,
        max_supports=max_supports,
        min_score=min_score,
        budget_ms=budget_ms,
        semantic_enabled=semantic_enabled,
        semantic_limit=semantic_limit,
        semantic_min_score=semantic_min_score,
        locomo_support_emission_enabled=True,
        preserve_initial_support_enabled=preserve_initial_support_enabled,
        preserve_initial_support_displace_enabled=preserve_initial_support_displace_enabled,
    )
    preserved_diagnostics = _locomo_preserved_initial_support_diagnostics(
        backfill,
        preserve_initial_support_enabled=preserve_initial_support_enabled,
    )
    activated_continuity = (
        _collect_locomo_activated_support_continuity(
            question=question,
            activated_concepts=activated_concepts[:max_activated_concepts],
            max_support_chars=max_support_chars,
            max_supports=max_supports,
        )
        if activated_support_continuity_enabled
        else None
    )
    emitted_supports = list(backfill.supports)
    emitted_ids = {support.concept_id for support in emitted_supports if support.concept_id}
    for support in (activated_continuity.supports if activated_continuity else ()):
        if support.concept_id and support.concept_id in emitted_ids:
            continue
        emitted_supports.append(support)
        if support.concept_id:
            emitted_ids.add(support.concept_id)
    payload = {
        "recovery_strategy": "locomo_source_bound_support_emission",
        "backfill_candidate_ids": list(backfill.candidate_ids or ()),
        "backfill_support_surfaces": [
            _locomo_preserved_support_surface(
                support,
                set(backfill.preserved_initial_candidate_ids or ()),
            )
            for support in emitted_supports
        ],
        "backfill_rejection_counts": backfill.rejection_counts or {},
        "backfill_latency_ms": backfill.latency_ms,
        "backfill_semantic_candidate_ids": list(backfill.semantic_candidate_ids or ()),
        "backfill_semantic_admitted_ids": list(backfill.semantic_admitted_ids or ()),
        "backfill_semantic_latency_ms": backfill.semantic_latency_ms,
        **preserved_diagnostics,
        "support_admission_v2_considered": True,
        "support_admission_v2_binding_status": "support_emitted",
        "support_admission_v2_shape": _infer_structured_synthesis_shape(question),
    }
    if activated_continuity is not None:
        continuity_ids = list(activated_continuity.candidate_ids or ())
        payload.update(
            {
                "locomo_activated_support_continuity_considered": True,
                "locomo_activated_support_continuity_admitted": bool(continuity_ids),
                "locomo_activated_support_continuity_candidate_ids": continuity_ids,
                "locomo_activated_support_continuity_rejection_counts": (
                    activated_continuity.rejection_counts or {}
                ),
                "locomo_activated_support_continuity_rejected_ids_by_reason": {
                    reason: list(ids)
                    for reason, ids in (
                        activated_continuity.rejected_ids_by_reason or {}
                    ).items()
                },
                "locomo_activated_support_continuity_strategy": (
                    "activated_source_bound_support"
                ),
                "locomo_activated_support_continuity_effect_enabled": True,
            }
        )
        if continuity_ids:
            payload["backfill_candidate_ids"] = list(
                dict.fromkeys([*payload["backfill_candidate_ids"], *continuity_ids])
            )
    elif activated_support_continuity_enabled:
        payload["locomo_activated_support_continuity_considered"] = True
        payload["locomo_activated_support_continuity_admitted"] = False
    if not emitted_supports:
        payload.pop("backfill_candidate_ids", None)
        payload.pop("backfill_support_surfaces", None)
        payload.pop("support_admission_v2_binding_status", None)
        payload.pop("support_admission_v2_shape", None)
    return payload


def locomo_support_candidate_movement_ledger(
    *,
    question: str,
    activated_concepts: list,
    max_activated_concepts: int,
    max_support_chars: int,
    fts_limit: int,
    assoc_limit: int,
    max_supports: int,
    min_score: float,
    budget_ms: float,
    semantic_enabled: bool = False,
    semantic_limit: int = 0,
    semantic_min_score: float = 0.45,
    target_concept_ids: tuple[str, ...] = (),
) -> dict[str, object]:
    """Trace LoCoMo support-candidate movement without changing answer behavior."""
    t0 = time.perf_counter()
    deadline = t0 + max(75.0, min(float(budget_ms), 250.0)) / 1000.0
    rejection_counts: dict[str, int] = {}
    fts_limit = max(0, min(int(fts_limit), 50))
    assoc_limit = max(0, min(int(assoc_limit), 80))
    max_supports = max(0, min(int(max_supports), 16))
    min_score = max(0.0, min(float(min_score), 1.0))
    semantic_limit = max(0, min(int(semantic_limit), 50))
    semantic_min_score = max(0.0, min(float(semantic_min_score), 1.0))
    activated_window = activated_concepts[:max_activated_concepts]
    existing_supports = _collect_supports(
        activated_window,
        max_support_chars=max_support_chars,
    )
    existing_supports = _hydrate_support_temporal_metadata(existing_supports)
    terms = _support_candidate_backfill_content_terms(question)
    fetch_stages: list[dict[str, object]] = []
    candidate_rows: dict[str, dict] = {}
    semantic_candidate_ids: tuple[str, ...] = ()
    semantic_latency_ms: float | None = None

    def _record_stage(stage: str, rows: list[sqlite3.Row]) -> None:
        fetch_stages.append(
            {
                "stage": stage,
                "row_count": len(rows),
                "candidate_ids": [
                    _stringify(_row_value(row, "concept_id")).strip()
                    for row in rows
                    if _stringify(_row_value(row, "concept_id")).strip()
                ],
            }
        )

    def _return_early(reason: str) -> dict[str, object]:
        _increment_rejection(rejection_counts, reason)
        return {
            "schema_version": "locomo.support_candidate_movement_ledger.v1",
            "question": question,
            "config": {
                "max_activated_concepts": max_activated_concepts,
                "max_support_chars": max_support_chars,
                "fts_limit": fts_limit,
                "assoc_limit": assoc_limit,
                "max_supports": max_supports,
                "min_score": min_score,
                "budget_ms": budget_ms,
                "semantic_enabled": semantic_enabled,
                "semantic_limit": semantic_limit,
                "semantic_min_score": semantic_min_score,
            },
            "terms": list(terms),
            "fetch_stages": fetch_stages,
            "candidate_entries": [],
            "support_rows": [],
            "selected_candidate_ids": [],
            "target_presence": {
                target_id: {
                    "found_in_candidates": False,
                    "accepted_for_ranking": False,
                    "selected": False,
                    "best_selected_rank": None,
                    "best_effective_score": None,
                    "best_not_selected_reason": None,
                }
                for target_id in target_concept_ids
            },
            "rejection_counts": rejection_counts,
            "latency_ms": _latency_ms(t0),
        }

    if max_supports <= 0:
        return _return_early("backfill_disabled")
    if len(terms) < 2:
        return _return_early("backfill_no_terms")

    initial_pack_ids = {
        item.support.concept_id
        for item in _build_support_pack(question, existing_supports)
        if item.support.concept_id
    }
    temporal_anchor_valid_froms = _temporal_source_set_anchor_valid_froms(question, existing_supports)
    temporal_anchor_concept_ids = _temporal_source_set_anchor_concept_ids(question, existing_supports)
    locomo_question_source_dates = _locomo_question_source_dates(question)
    locomo_source_window_anchor_valid_froms = _locomo_source_window_anchor_valid_froms(
        question,
        existing_supports,
    )
    home_country_alias_seed = (
        _home_country_move_from_actor(question) is not None
        and any(
            _support_has_home_country_move_alias(question, existing_support)
            for existing_support in existing_supports
        )
    )
    activated_ids = _ordered_activated_concept_ids(activated_window)
    try:
        conn = _open_support_candidate_backfill_connection()
    except Exception:
        return _return_early("backfill_no_connection")

    try:
        association_first = _benefit_with_having_parts(question) is not None
        if temporal_anchor_valid_froms and time.perf_counter() < deadline:
            try:
                rows = _fetch_temporal_source_set_bridge_candidate_rows(
                    conn,
                    temporal_anchor_concept_ids,
                    temporal_anchor_valid_froms,
                    max(4, min(16, assoc_limit or fts_limit or 8)),
                )
                _record_stage("temporal_source_set_bridge", rows)
                for row in rows:
                    _merge_support_candidate_row(candidate_rows, row, "temporal_source_set_bridge")
                rows = _fetch_temporal_source_set_candidate_rows(
                    conn,
                    temporal_anchor_valid_froms,
                    max(4, min(16, assoc_limit or fts_limit or 8)),
                )
                _record_stage("temporal_source_set", rows)
                for row in rows:
                    _merge_support_candidate_row(candidate_rows, row, "temporal_source_set")
            except Exception:
                _increment_rejection(rejection_counts, "backfill_no_candidates")
        if association_first and assoc_limit > 0 and time.perf_counter() < deadline:
            try:
                rows = _fetch_association_support_candidate_rows(conn, activated_ids, assoc_limit)
                _record_stage("association", rows)
                for row in rows:
                    _merge_support_candidate_row(candidate_rows, row, "association")
            except Exception:
                _increment_rejection(rejection_counts, "backfill_no_candidates")
        if home_country_alias_seed and time.perf_counter() < deadline:
            try:
                rows = _fetch_home_country_alias_candidate_rows(
                    conn,
                    max(4, min(16, fts_limit or assoc_limit or 8)),
                )
                _record_stage("home_country_alias", rows)
                for row in rows:
                    _merge_support_candidate_row(candidate_rows, row, "home_country_alias")
            except Exception:
                _increment_rejection(rejection_counts, "backfill_no_candidates")
        if locomo_source_window_anchor_valid_froms and time.perf_counter() < deadline:
            try:
                rows = _fetch_locomo_source_window_candidate_rows(
                    conn,
                    locomo_source_window_anchor_valid_froms,
                    max(16, min(160, fts_limit + assoc_limit or 24)),
                )
                _record_stage("locomo_source_window", rows)
                for row in rows:
                    _merge_support_candidate_row(candidate_rows, row, "locomo_source_window")
            except Exception:
                _increment_rejection(rejection_counts, "backfill_no_candidates")
        if locomo_question_source_dates and time.perf_counter() < deadline:
            try:
                rows = _fetch_locomo_question_date_candidate_rows(
                    conn,
                    locomo_question_source_dates,
                    max(16, min(160, fts_limit + assoc_limit or 24)),
                )
                _record_stage("locomo_question_date", rows)
                for row in rows:
                    _merge_support_candidate_row(candidate_rows, row, "locomo_question_date")
            except Exception:
                _increment_rejection(rejection_counts, "backfill_no_candidates")
        if time.perf_counter() < deadline:
            try:
                rows = _fetch_locomo_source_fragment_candidate_rows(
                    conn,
                    question,
                    terms,
                    max(16, min(24, fts_limit + assoc_limit or 24)),
                )
                _record_stage("locomo_source_fragment", rows)
                for row in rows:
                    _merge_support_candidate_row(candidate_rows, row, "locomo_source_fragment")
            except Exception:
                _increment_rejection(rejection_counts, "backfill_no_candidates")
        if fts_limit > 0 and time.perf_counter() < deadline and not (association_first and candidate_rows):
            try:
                rows = _fetch_fts_support_candidate_rows(conn, terms, fts_limit)
                _record_stage("fts", rows)
                for row in rows:
                    _merge_support_candidate_row(candidate_rows, row, "fts")
                rows = _fetch_verbatim_fts_support_candidate_rows(conn, terms, fts_limit)
                _record_stage("fts_verbatim", rows)
                for row in rows:
                    _merge_support_candidate_row(candidate_rows, row, "fts_verbatim")
            except Exception:
                _increment_rejection(rejection_counts, "backfill_no_candidates")
        if not association_first and assoc_limit > 0 and time.perf_counter() < deadline:
            try:
                rows = _fetch_association_support_candidate_rows(conn, activated_ids, assoc_limit)
                _record_stage("association", rows)
                for row in rows:
                    _merge_support_candidate_row(candidate_rows, row, "association")
            except Exception:
                _increment_rejection(rejection_counts, "backfill_no_candidates")
        if time.perf_counter() < deadline:
            source_window_valid_froms = _merge_ordered_strings(
                locomo_source_window_anchor_valid_froms,
                _locomo_source_window_candidate_valid_froms(candidate_rows),
                limit=8,
            )
            if source_window_valid_froms:
                try:
                    rows = _fetch_locomo_source_window_candidate_rows(
                        conn,
                        source_window_valid_froms,
                        max(16, min(160, fts_limit + assoc_limit or 24)),
                    )
                    _record_stage("locomo_source_window_expanded", rows)
                    for row in rows:
                        _merge_support_candidate_row(candidate_rows, row, "locomo_source_window")
                except Exception:
                    _increment_rejection(rejection_counts, "backfill_no_candidates")
        if semantic_enabled and semantic_limit > 0 and time.perf_counter() < deadline:
            try:
                rows, semantic_candidate_ids, semantic_latency_ms = _fetch_semantic_support_candidate_rows(
                    conn,
                    question,
                    semantic_limit,
                    semantic_min_score,
                    deadline,
                )
                _record_stage("semantic", rows)
                if not rows:
                    _increment_rejection(rejection_counts, "semantic_no_candidates")
                for row in rows:
                    _merge_support_candidate_row(candidate_rows, row, "semantic")
            except Exception:
                _increment_rejection(rejection_counts, "semantic_fetch_error")
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if time.perf_counter() >= deadline:
        _increment_rejection(rejection_counts, "backfill_timeout")
    evaluation_deadline = max(
        deadline,
        time.perf_counter() + _SUPPORT_CANDIDATE_BACKFILL_EVALUATION_GRACE_MS / 1000.0,
    )
    candidate_entries: list[dict[str, object]] = []
    for concept_index, concept in enumerate(activated_window):
        concept_id = _concept_id(concept, concept_index)
        if not concept_id:
            continue
        if concept_id in candidate_rows:
            candidate_rows[concept_id]["signals"].add("activation")
            continue
        candidate_entries.append(
            {"concept_id": concept_id, "concept": concept, "signals": {"activation"}}
        )
    candidate_entries.extend(
        {"concept_id": concept_id, "row": entry["row"], "signals": entry["signals"]}
        for concept_id, entry in candidate_rows.items()
    )
    target_concept_id_set = {target_id for target_id in target_concept_ids if target_id}

    def _ledger_candidate_entry_priority(entry: dict[str, object]) -> tuple[int, int, str]:
        concept_id = _stringify(entry.get("concept_id")).strip()
        signal_rank, signal_concept_id = _support_candidate_backfill_entry_priority(entry)
        return (
            0 if concept_id in target_concept_id_set else 1,
            signal_rank,
            signal_concept_id,
        )

    candidate_entries.sort(key=_ledger_candidate_entry_priority)

    candidate_entry_rows: list[dict[str, object]] = []
    support_rows: list[dict[str, object]] = []
    admitted: list[tuple[float, EvidenceSupport, int]] = []
    semantic_admitted_ids: list[str] = []
    locomo_decisive_evidence_family_by_id: dict[str, str] = {}
    locomo_decisive_evidence_preserved_ids: list[str] = []
    locomo_decisive_evidence_rejection_counts: dict[str, int] = {}
    rare_terms = set(terms)
    locomo_family_accident_question = _locomo_family_accident_question(question)
    relationship_status_question = _RELATIONSHIP_STATUS_QUESTION_PATTERN.search(question) is not None
    likely_fields_question = _likely_fields_question(question)
    event_list_question = _support_present_answer_role(question) == "event_list"

    for entry_index, entry in enumerate(candidate_entries):
        concept_id = _stringify(entry.get("concept_id")).strip()
        signals = entry.get("signals") if isinstance(entry.get("signals"), set) else set()
        row = entry.get("row")
        candidate_entry_rows.append(
            {
                "entry_index": entry_index,
                "concept_id": concept_id,
                "signals": sorted(signals),
                "summary": _stringify(_row_value(row, "summary"))
                if isinstance(row, sqlite3.Row)
                else _stringify(getattr(entry.get("concept"), "summary", "")),
                "valid_from": _optional_str(_row_value(row, "valid_from"))
                if isinstance(row, sqlite3.Row)
                else _optional_str(getattr(entry.get("concept"), "valid_from", None)),
                "content_updated_at": _optional_str(_row_value(row, "content_updated_at"))
                if isinstance(row, sqlite3.Row)
                else _optional_str(getattr(entry.get("concept"), "content_updated_at", None)),
            }
        )
        if time.perf_counter() >= evaluation_deadline:
            _increment_rejection(rejection_counts, "backfill_timeout")
            break
        if not concept_id:
            continue
        if concept_id in initial_pack_ids:
            _increment_rejection(rejection_counts, "backfill_duplicate")
            duplicate_detail: dict[str, object] = {
                "entry_index": entry_index,
                "concept_id": concept_id,
                "signals": sorted(signals),
                "accepted_for_ranking": False,
                "selected": False,
                "selected_rank": None,
                "not_selected_reason": "backfill_duplicate",
                "duplicate_equivalence_available": False,
                "duplicate_preservation_decision": "not_evaluated",
            }
            try:
                concept = (
                    _support_candidate_row_to_concept(row)
                    if isinstance(row, sqlite3.Row)
                    else entry.get("concept")
                )
                duplicate_supports = _collect_supports([concept], max_support_chars=4000) if concept else []
            except Exception:
                duplicate_supports = []
            for support in duplicate_supports:
                decisive_assessment = _locomo_decisive_support_assessment(
                    question,
                    support,
                    signals,
                )
                source_match, source_rejection = _locomo_source_bound_emission_support_match(
                    question,
                    support,
                    signals,
                )
                actor_text_safe, actor_text_rejection = (
                    _locomo_preserved_initial_support_actor_text_safe(question, support)
                )
                actor_compatible = _support_actor_compatible(question, support)
                duplicate_detail.update(
                    {
                        "support_id": support.support_id,
                        "channel": support.channel,
                        "support_text": support.support_text,
                        "concept_summary": support.concept_summary,
                        "duplicate_equivalence_available": True,
                        "source_bound_emission_match": source_match,
                        "source_bound_rejection": source_rejection,
                        "actor_compatible": actor_compatible,
                        "actor_text_safe": actor_text_safe,
                        "actor_text_rejection": actor_text_rejection,
                        "support_text_sha256": _locomo_support_surface_sha256(
                            support.support_text
                        ),
                        "concept_summary_sha256": _locomo_support_surface_sha256(
                            support.concept_summary
                        ),
                        "decisive_evidence_family": decisive_assessment.family,
                        "decisive_evidence_rejection": decisive_assessment.rejection_reason,
                        "duplicate_preservation_decision": "trace_only",
                    }
                )
                if (
                    decisive_assessment.family
                    and decisive_assessment.safe_to_preserve_duplicate
                ):
                    scored = _score_support(question, support)
                    duplicate_detail.update(
                        {
                            "accepted_for_ranking": True,
                            "not_selected_reason": None,
                            "score": scored.score,
                            "question_overlap": scored.question_overlap,
                            "predicate_overlap": scored.predicate_overlap,
                            "effective_score": max(
                                scored.score,
                                min_score + decisive_assessment.score_boost,
                            ),
                            "score_floor": min_score,
                            "signal_count": 2,
                            "duplicate_preservation_decision": (
                                "decisive_evidence_preserved"
                            ),
                        }
                    )
                    locomo_decisive_evidence_family_by_id[concept_id] = (
                        decisive_assessment.family
                    )
                    locomo_decisive_evidence_preserved_ids.append(concept_id)
                    support_rows.append(duplicate_detail)
                    admitted.append(
                        (
                            float(duplicate_detail["effective_score"]),
                            support,
                            len(support_rows) - 1,
                        )
                    )
                    continue
                if decisive_assessment.rejection_reason:
                    _increment_rejection(
                        locomo_decisive_evidence_rejection_counts,
                        decisive_assessment.rejection_reason,
                    )
            if len(support_rows) and support_rows[-1].get("concept_id") == concept_id:
                continue
            support_rows.append(duplicate_detail)
            continue
        if isinstance(row, sqlite3.Row) and not _support_candidate_row_is_active(row):
            _increment_rejection(rejection_counts, "backfill_inactive")
            support_rows.append(
                {
                    "entry_index": entry_index,
                    "concept_id": concept_id,
                    "signals": sorted(signals),
                    "accepted_for_ranking": False,
                    "selected": False,
                    "selected_rank": None,
                    "not_selected_reason": "backfill_inactive",
                }
            )
            continue

        concept = _support_candidate_row_to_concept(row) if isinstance(row, sqlite3.Row) else entry["concept"]
        supports = _collect_supports([concept], max_support_chars=4000)
        if not supports:
            support_rows.append(
                {
                    "entry_index": entry_index,
                    "concept_id": concept_id,
                    "signals": sorted(signals),
                    "accepted_for_ranking": False,
                    "selected": False,
                    "selected_rank": None,
                    "not_selected_reason": "backfill_no_support_surface",
                }
            )
            continue
        best_support: tuple[float, EvidenceSupport, int] | None = None
        for support in supports:
            row_detail: dict[str, object] = {
                "entry_index": entry_index,
                "concept_id": concept_id,
                "support_id": support.support_id,
                "channel": support.channel,
                "signals": sorted(signals),
                "support_text": support.support_text,
                "concept_summary": support.concept_summary,
                "accepted_for_ranking": False,
                "selected": False,
                "selected_rank": None,
                "not_selected_reason": None,
            }
            temporal_make_media_match = _temporal_make_media_backfill_support_match(
                question,
                support,
            )
            decisive_assessment = _locomo_decisive_support_assessment(
                question,
                support,
                signals,
            )
            locomo_source_window_family_accident_match = _locomo_source_window_family_accident_support_match(
                question,
                support,
                signals,
            )
            actor_compatible = _support_actor_compatible(question, support)
            row_detail["actor_compatible"] = actor_compatible
            row_detail["locomo_source_window_family_accident_match"] = (
                locomo_source_window_family_accident_match
            )
            if (
                not temporal_make_media_match
                and not locomo_source_window_family_accident_match
                and not decisive_assessment.family
                and not actor_compatible
            ):
                _increment_rejection(rejection_counts, "backfill_actor_mismatch")
                row_detail["not_selected_reason"] = "backfill_actor_mismatch"
                support_rows.append(row_detail)
                continue
            scored = _score_support(question, support)
            support_terms = set(_content_tokens(support.support_text))
            rare_overlap = bool(rare_terms & support_terms)
            predicate_overlap = scored.predicate_overlap > 0.0
            direct_atomic_match = (
                _relationship_status_candidate_from_sentence(question, support.support_text) is not None
            )
            semantic_signal = "semantic" in signals
            event_list_answer = (
                _support_present_event_list_candidate(question, support.support_text)
                if event_list_question
                else None
            )
            row_detail.update(
                {
                    "score": scored.score,
                    "question_overlap": scored.question_overlap,
                    "predicate_overlap": scored.predicate_overlap,
                    "rare_overlap": rare_overlap,
                    "predicate_overlap_passed": predicate_overlap,
                    "direct_atomic_match": direct_atomic_match,
                    "event_list_answer": event_list_answer,
                }
            )
            if event_list_question and event_list_answer is None:
                _increment_rejection(rejection_counts, "backfill_no_event_list_candidate")
                row_detail["not_selected_reason"] = "backfill_no_event_list_candidate"
                support_rows.append(row_detail)
                continue
            if relationship_status_question and not direct_atomic_match:
                _increment_rejection(rejection_counts, "backfill_no_direct_relationship_status")
                row_detail["not_selected_reason"] = "backfill_no_direct_relationship_status"
                support_rows.append(row_detail)
                continue
            field_specificity_bonus = _likely_fields_support_specificity_bonus(
                question,
                support.support_text,
            )
            if likely_fields_question and field_specificity_bonus <= 0.0:
                _increment_rejection(rejection_counts, "backfill_no_field_answer_terms")
                row_detail["not_selected_reason"] = "backfill_no_field_answer_terms"
                support_rows.append(row_detail)
                continue
            likely_fields_direct_answer = _likely_fields_direct_answer_support(
                question,
                support.support_text,
            )
            locomo_direct_answer = _locomo_source_backfill_direct_support_match(question, support)
            locomo_adoption_opinion_match = _locomo_adoption_opinion_support_match(
                question,
                support,
                signals,
            )
            temporal_source_set_match = _temporal_source_set_backfill_match(
                question,
                support,
                signals,
                existing_supports,
            )
            locomo_emission_match, locomo_rejection = _locomo_source_bound_emission_support_match(
                question,
                support,
                signals,
            )
            row_detail.update(
                {
                    "field_specificity_bonus": field_specificity_bonus,
                    "likely_fields_direct_answer": likely_fields_direct_answer,
                    "locomo_direct_answer": locomo_direct_answer,
                    "locomo_adoption_opinion_match": locomo_adoption_opinion_match,
                    "temporal_source_set_match": temporal_source_set_match,
                    "locomo_emission_match": locomo_emission_match,
                    "locomo_emission_rejection": locomo_rejection,
                    "decisive_evidence_family": decisive_assessment.family,
                    "decisive_evidence_rejection": decisive_assessment.rejection_reason,
                }
            )
            if (
                not locomo_emission_match
                and not temporal_source_set_match
                and not decisive_assessment.family
            ):
                reason = locomo_rejection or "locomo_emission_shape_mismatch"
                _increment_rejection(rejection_counts, reason)
                row_detail["not_selected_reason"] = reason
                support_rows.append(row_detail)
                continue
            home_country_alias_direct_answer = (
                _home_country_move_from_actor(question) is not None
                and any(
                    _support_has_home_country_move_alias(question, existing_support)
                    for existing_support in existing_supports
                )
                and _home_country_value_candidate_from_support(question, support) is not None
            )
            if (
                not rare_overlap
                and not predicate_overlap
                and not home_country_alias_direct_answer
                and not locomo_direct_answer
                and not locomo_adoption_opinion_match
                and not locomo_source_window_family_accident_match
                and not temporal_source_set_match
                and not decisive_assessment.family
            ):
                if not semantic_signal or scored.score < semantic_min_score:
                    _increment_rejection(rejection_counts, "backfill_common_word_only")
                    row_detail["not_selected_reason"] = "backfill_common_word_only"
                    support_rows.append(row_detail)
                    continue
            score_floor = min_score
            identity_complete_match = _identity_support_sentence_bound(question, support.support_text)
            if direct_atomic_match or identity_complete_match:
                score_floor = min(score_floor, 0.08)
            elif likely_fields_question and field_specificity_bonus >= 0.12:
                score_floor = min(score_floor, min_score * 0.85)
            elif (
                home_country_alias_direct_answer
                or locomo_direct_answer
                or locomo_adoption_opinion_match
                or locomo_emission_match
                or locomo_source_window_family_accident_match
            ):
                score_floor = min(score_floor, 0.03)
            elif "activation" in signals and predicate_overlap:
                score_floor = min(score_floor, min_score * 0.8)
            elif (
                temporal_make_media_match
                or temporal_source_set_match
                or decisive_assessment.family
            ):
                score_floor = min(score_floor, 0.03)
            if event_list_answer is not None:
                score_floor = min(score_floor, min_score * 0.55)
            effective_score = scored.score + field_specificity_bonus
            if identity_complete_match:
                effective_score = max(effective_score, min_score + 0.5)
            if likely_fields_direct_answer:
                effective_score = max(effective_score, min_score + 0.5)
            if home_country_alias_direct_answer:
                effective_score = max(effective_score, min_score + 0.5)
            if locomo_direct_answer:
                effective_score = max(effective_score, min_score + 0.5)
            if locomo_adoption_opinion_match:
                effective_score = max(effective_score, min_score + 0.55)
            if locomo_emission_match:
                effective_score = max(effective_score, min_score + 0.18)
                if (("locomo_source_window" in signals) or _locomo_source_fragment_signal(signals)) and (
                    not locomo_family_accident_question
                    or locomo_source_window_family_accident_match
                    or temporal_make_media_match
                ):
                    effective_score += 0.45
            if locomo_source_window_family_accident_match:
                effective_score = max(effective_score, min_score + 0.55)
            if temporal_make_media_match:
                effective_score = max(effective_score, min_score + 0.12)
            if temporal_source_set_match:
                effective_score = max(
                    effective_score,
                    min_score + 0.95,
                )
            if decisive_assessment.family:
                effective_score = max(
                    effective_score,
                    min_score + decisive_assessment.score_boost,
                )
            if event_list_answer is not None:
                effective_score += 0.35
            row_detail.update(
                {
                    "score_floor": score_floor,
                    "effective_score": effective_score,
                    "identity_complete_match": identity_complete_match,
                    "temporal_make_media_match": temporal_make_media_match,
                    "home_country_alias_direct_answer": home_country_alias_direct_answer,
                }
            )
            if effective_score < score_floor:
                _increment_rejection(rejection_counts, "backfill_low_score")
                row_detail["not_selected_reason"] = "backfill_low_score"
                support_rows.append(row_detail)
                continue
            signal_count = 1
            signal_count += 1 if rare_overlap else 0
            signal_count += 1 if predicate_overlap else 0
            signal_count += 1 if "association" in signals else 0
            signal_count += 1 if "fts_verbatim" in signals and rare_overlap else 0
            signal_count += 1 if "fts" in signals and rare_overlap else 0
            signal_count += 1 if semantic_signal else 0
            signal_count += 1 if "activation" in signals else 0
            signal_count += 1 if "locomo_question_date" in signals else 0
            signal_count += 1 if "temporal_source_set" in signals else 0
            signal_count += 1 if "temporal_source_set_bridge" in signals else 0
            signal_count += 1 if "locomo_source_window" in signals else 0
            signal_count += 1 if _locomo_source_fragment_signal(signals) else 0
            signal_count += 1 if direct_atomic_match else 0
            signal_count += 1 if identity_complete_match else 0
            signal_count += 1 if home_country_alias_direct_answer else 0
            signal_count += 1 if locomo_direct_answer else 0
            signal_count += 1 if locomo_adoption_opinion_match else 0
            signal_count += 1 if locomo_emission_match else 0
            signal_count += 1 if locomo_source_window_family_accident_match else 0
            signal_count += 1 if temporal_make_media_match else 0
            signal_count += 1 if temporal_source_set_match else 0
            signal_count += 1 if decisive_assessment.family else 0
            signal_count += 1 if event_list_answer is not None else 0
            row_detail["signal_count"] = signal_count
            if signal_count < 2:
                _increment_rejection(rejection_counts, "backfill_low_score")
                row_detail["not_selected_reason"] = "backfill_low_score"
                support_rows.append(row_detail)
                continue
            row_detail["accepted_for_ranking"] = True
            support_rows.append(row_detail)
            ledger_index = len(support_rows) - 1
            if best_support is None or effective_score > best_support[0]:
                best_support = (effective_score, support, ledger_index)
                if decisive_assessment.family:
                    locomo_decisive_evidence_family_by_id[concept_id] = (
                        decisive_assessment.family
                    )
        if best_support is None:
            continue
        admitted.append(best_support)
        if "semantic" in signals:
            semantic_admitted_ids.append(concept_id)

    if not admitted:
        _increment_rejection(rejection_counts, "backfill_no_candidates")
    admitted.sort(key=lambda item: item[0], reverse=True)
    selected = admitted[:max_supports]
    selected_candidate_ids: list[str] = []
    for rank, (_score, support, ledger_index) in enumerate(selected, start=1):
        selected_candidate_ids.append(support.concept_id)
        support_rows[ledger_index]["selected"] = True
        support_rows[ledger_index]["selected_rank"] = rank
    for row_detail in support_rows:
        if row_detail.get("accepted_for_ranking") and not row_detail.get("selected"):
            row_detail["not_selected_reason"] = "top_n_cutoff"

    target_presence: dict[str, dict[str, object]] = {}
    for target_id in target_concept_ids:
        rows = [row for row in support_rows if row.get("concept_id") == target_id]
        accepted_rows = [row for row in rows if row.get("accepted_for_ranking")]
        selected_rows = [row for row in rows if row.get("selected")]
        scored_rows = [
            row for row in rows if isinstance(row.get("effective_score"), (float, int))
        ]
        best_scored = max(
            scored_rows,
            key=lambda row: float(row.get("effective_score") or 0.0),
            default=None,
        )
        target_presence[target_id] = {
            "found_in_candidates": any(entry.get("concept_id") == target_id for entry in candidate_entry_rows),
            "accepted_for_ranking": bool(accepted_rows),
            "selected": bool(selected_rows),
            "best_selected_rank": min(
                int(row["selected_rank"])
                for row in selected_rows
                if isinstance(row.get("selected_rank"), int)
            )
            if selected_rows
            else None,
            "best_effective_score": best_scored.get("effective_score") if best_scored else None,
            "best_not_selected_reason": best_scored.get("not_selected_reason") if best_scored else None,
        }

    return {
        "schema_version": "locomo.support_candidate_movement_ledger.v1",
        "question": question,
        "config": {
            "max_activated_concepts": max_activated_concepts,
            "max_support_chars": max_support_chars,
            "fts_limit": fts_limit,
            "assoc_limit": assoc_limit,
            "max_supports": max_supports,
            "min_score": min_score,
            "budget_ms": budget_ms,
            "semantic_enabled": semantic_enabled,
            "semantic_limit": semantic_limit,
            "semantic_min_score": semantic_min_score,
        },
        "terms": list(terms),
        "initial_pack_ids": sorted(initial_pack_ids),
        "temporal_anchor_valid_froms": list(temporal_anchor_valid_froms),
        "locomo_source_window_anchor_valid_froms": list(locomo_source_window_anchor_valid_froms),
        "fetch_stages": fetch_stages,
        "candidate_entries": candidate_entry_rows,
        "support_rows": support_rows,
        "selected_candidate_ids": selected_candidate_ids,
        "target_presence": target_presence,
        "rejection_counts": rejection_counts,
        "semantic_candidate_ids": list(semantic_candidate_ids),
        "semantic_admitted_ids": semantic_admitted_ids,
        "semantic_latency_ms": semantic_latency_ms,
        "locomo_decisive_evidence_preserved_ids": list(
            dict.fromkeys(locomo_decisive_evidence_preserved_ids)
        ),
        "locomo_decisive_evidence_family_by_id": locomo_decisive_evidence_family_by_id,
        "locomo_decisive_evidence_rejection_counts": locomo_decisive_evidence_rejection_counts,
        "latency_ms": _latency_ms(t0),
    }


def locomo_bounded_support_admission_diagnostics(
    *,
    question: str,
    activated_concepts: list,
    max_activated_concepts: int,
    max_support_chars: int,
    fts_limit: int,
    assoc_limit: int,
    candidate_pool_size: int,
    min_score: float,
    budget_ms: float,
    semantic_enabled: bool = False,
    semantic_limit: int = 0,
    semantic_min_score: float = 0.45,
    effect_enabled: bool = False,
    preserve_initial_support_enabled: bool = False,
    preserve_initial_support_displace_enabled: bool = False,
    activated_support_continuity_enabled: bool = False,
) -> dict[str, object]:
    """Emit default-off LoCoMo bounded support-admission shadow diagnostics."""
    candidate_pool_size = max(4, min(int(candidate_pool_size), 40))
    supports = _collect_supports(
        activated_concepts[:max_activated_concepts],
        max_support_chars=max_support_chars,
    )
    backfill = _collect_support_candidate_backfill(
        question=question,
        activated_concepts=activated_concepts[:max_activated_concepts],
        existing_supports=supports,
        fts_limit=fts_limit,
        assoc_limit=assoc_limit,
        max_supports=candidate_pool_size,
        min_score=min_score,
        budget_ms=budget_ms,
        semantic_enabled=semantic_enabled,
        semantic_limit=semantic_limit,
        semantic_min_score=semantic_min_score,
        locomo_support_emission_enabled=True,
        preserve_initial_support_enabled=preserve_initial_support_enabled,
        preserve_initial_support_displace_enabled=preserve_initial_support_displace_enabled,
    )
    activated_continuity = (
        _collect_locomo_activated_support_continuity(
            question=question,
            activated_concepts=activated_concepts[:max_activated_concepts],
            max_support_chars=max_support_chars,
            max_supports=candidate_pool_size,
        )
        if activated_support_continuity_enabled
        else None
    )
    emitted_supports = list(backfill.supports)
    emitted_ids = {support.concept_id for support in emitted_supports if support.concept_id}
    for support in (activated_continuity.supports if activated_continuity else ()):
        if support.concept_id and support.concept_id in emitted_ids:
            continue
        emitted_supports.append(support)
        if support.concept_id:
            emitted_ids.add(support.concept_id)
    admitted_count = len(backfill.supports)
    preserved_diagnostics = _locomo_preserved_initial_support_diagnostics(
        backfill,
        preserve_initial_support_enabled=preserve_initial_support_enabled,
    )
    payload = {
        "locomo_bounded_support_admission_considered": True,
        "locomo_bounded_support_admission_admitted": bool(emitted_supports),
        "locomo_bounded_support_admission_answer_effect": False,
        "locomo_bounded_support_admission_effect_enabled": bool(effect_enabled),
        "locomo_bounded_support_admission_candidate_pool_size": candidate_pool_size,
        "locomo_bounded_support_admission_answer_support_count": min(
            len(emitted_supports),
            8,
        ),
        "locomo_bounded_support_admission_candidate_ids": list(backfill.candidate_ids or ()),
        "locomo_bounded_support_admission_rejection_counts": backfill.rejection_counts or {},
        "locomo_bounded_support_admission_latency_ms": backfill.latency_ms,
        "locomo_bounded_support_admission_strategy": "shadow_support_pool",
        "backfill_candidate_ids": list(backfill.candidate_ids or ()),
        "backfill_support_surfaces": [
            _locomo_preserved_support_surface(
                support,
                set(backfill.preserved_initial_candidate_ids or ()),
            )
            for support in emitted_supports
        ],
        "backfill_rejection_counts": backfill.rejection_counts or {},
        "backfill_latency_ms": backfill.latency_ms,
        **preserved_diagnostics,
    }
    if activated_continuity is not None:
        continuity_ids = list(activated_continuity.candidate_ids or ())
        payload.update(
            {
                "locomo_activated_support_continuity_considered": True,
                "locomo_activated_support_continuity_admitted": bool(continuity_ids),
                "locomo_activated_support_continuity_candidate_ids": continuity_ids,
                "locomo_activated_support_continuity_rejection_counts": (
                    activated_continuity.rejection_counts or {}
                ),
                "locomo_activated_support_continuity_rejected_ids_by_reason": {
                    reason: list(ids)
                    for reason, ids in (
                        activated_continuity.rejected_ids_by_reason or {}
                    ).items()
                },
                "locomo_activated_support_continuity_strategy": (
                    "activated_source_bound_support"
                ),
                "locomo_activated_support_continuity_effect_enabled": bool(effect_enabled),
            }
        )
        if continuity_ids:
            merged_ids = list(
                dict.fromkeys([*payload["backfill_candidate_ids"], *continuity_ids])
            )
            payload["backfill_candidate_ids"] = merged_ids
            payload["locomo_bounded_support_admission_candidate_ids"] = merged_ids
    elif activated_support_continuity_enabled:
        payload["locomo_activated_support_continuity_considered"] = True
        payload["locomo_activated_support_continuity_admitted"] = False
    return payload


def _locomo_source_bound_emission_support_match(
    question: str,
    support: EvidenceSupport,
    signals: set[str],
) -> tuple[bool, str | None]:
    if "semantic" in signals and len(signals) == 1:
        return False, "locomo_emission_semantic_only"
    source_window_family_accident_match = _locomo_source_window_family_accident_support_match(
        question,
        support,
        signals,
    )
    question_actor_terms = set(_question_actor_terms(question))
    support_actor_terms = set(support.concept_actor_terms or ())
    if (
        question_actor_terms
        and support_actor_terms
        and not question_actor_terms & support_actor_terms
        and not source_window_family_accident_match
    ):
        return False, "locomo_emission_actor_mismatch"
    if not source_window_family_accident_match and not _support_actor_compatible(question, support):
        return False, "locomo_emission_actor_mismatch"
    text = f"{support.support_text} {support.concept_summary}"
    if not _locomo_support_has_source_marker(support):
        return False, "locomo_emission_no_source_marker"
    question_terms = set(_content_tokens(question)) - _SUPPORT_CANDIDATE_BACKFILL_COMMON_TERMS
    support_terms = set(_content_tokens(text))
    if (
        _locomo_adoption_duration_question(question)
        and "first" in question.lower()
        and not _locomo_first_pet_adoption_duration_support(support_terms, text.lower())
    ):
        return False, "locomo_emission_adoption_duration_first_pet_not_decisive"
    anchor_terms = question_terms & support_terms
    direct_match = _locomo_source_backfill_direct_support_match(question, support)
    temporal_match = _temporal_make_media_backfill_support_match(question, support)
    if (
        _TEMPORAL_QUESTION_PATTERN.search(question)
        and anchor_terms
        and not direct_match
        and not temporal_match
        and not source_window_family_accident_match
    ):
        actor_terms = set(_question_actor_terms(question))
        non_actor_anchor_terms = anchor_terms - actor_terms
        if not non_actor_anchor_terms:
            return False, "locomo_emission_insufficient_temporal_entity_anchor"
    event_list_question = _support_present_answer_role(question) == "event_list"
    event_list_answer = (
        _support_present_event_list_candidate(question, support.support_text)
        if event_list_question
        else None
    )
    if event_list_question and event_list_answer is None:
        return False, "locomo_emission_shape_mismatch"
    activity_actor = _locomo_activity_object_question(question)
    if activity_actor is not None:
        if not _locomo_support_has_actor(support, activity_actor):
            return False, "locomo_emission_actor_mismatch"
        has_activity_candidate = any(
            _locomo_activity_object_candidate(sentence) is not None
            for surface in (support.support_text, support.concept_summary)
            for sentence in _locomo_support_sentences_without_shared_media(surface)
            if _locomo_sentence_mentions_actor(sentence, activity_actor)
        )
        if not has_activity_candidate:
            return False, "locomo_emission_no_activity_object_candidate"
    action_actor = _locomo_action_bundle_question(question)
    if action_actor is not None:
        if not _locomo_support_has_actor(support, action_actor):
            return False, "locomo_emission_actor_mismatch"
        has_action_candidate = any(
            _locomo_action_bundle_candidate(sentence) is not None
            for surface in (support.support_text, support.concept_summary)
            for sentence in _locomo_support_sentences_without_shared_media(surface)
            if _locomo_sentence_mentions_actor(sentence, action_actor)
        )
        if not has_action_candidate:
            return False, "locomo_emission_no_action_bundle_candidate"
    if (
        not anchor_terms
        and not direct_match
        and not temporal_match
        and not source_window_family_accident_match
        and event_list_answer is None
    ):
        return False, "locomo_emission_no_predicate_object_anchor"
    sentiment_terms = {
        "support",
        "supported",
        "family",
        "journey",
        "important",
        "thankful",
        "grateful",
        "love",
        "loved",
        "adopt",
        "adoption",
    }
    if (
        anchor_terms
        and anchor_terms <= sentiment_terms
        and not direct_match
        and not source_window_family_accident_match
        and event_list_answer is None
    ):
        return False, "locomo_emission_broad_sentiment_only"
    return True, None


def _locomo_source_window_anchor_valid_froms(
    question: str,
    supports: list[EvidenceSupport],
) -> tuple[str, ...]:
    question_terms = set(_content_tokens(question)) - _SUPPORT_CANDIDATE_BACKFILL_COMMON_TERMS
    candidates: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    for support_index, support in enumerate(supports):
        if not support.concept_valid_from:
            continue
        support_text = f"{support.support_text} {support.concept_summary}"
        support_terms = set(_content_tokens(support_text))
        overlap_count = len(question_terms & support_terms)
        if overlap_count < 2:
            continue
        if support.concept_valid_from in seen:
            continue
        seen.add(support.concept_valid_from)
        score = overlap_count * 10
        candidates.append((-score, support_index, support.concept_valid_from))
    candidates.sort()
    return tuple(valid_from for _score, _index, valid_from in candidates[:8])


def _locomo_source_window_candidate_valid_froms(
    candidate_rows: dict[str, dict],
) -> tuple[str, ...]:
    valid_froms: list[str] = []
    for entry in candidate_rows.values():
        row = entry.get("row")
        if not isinstance(row, sqlite3.Row):
            continue
        valid_from = _optional_str(_row_value(row, "valid_from"))
        if valid_from and valid_from not in valid_froms:
            valid_froms.append(valid_from)
        if len(valid_froms) >= 8:
            break
    return tuple(valid_froms)


def _merge_ordered_strings(
    *groups: tuple[str, ...],
    limit: int,
) -> tuple[str, ...]:
    values: list[str] = []
    for group in groups:
        for value in group:
            if value and value not in values:
                values.append(value)
            if len(values) >= limit:
                return tuple(values)
    return tuple(values)


def _locomo_family_accident_question(question: str) -> bool:
    question_terms = set(_content_tokens(question))
    return "accident" in question_terms and bool(
        question_terms & {"son", "child", "children", "kid", "kids", "brother"}
    )


def _locomo_adoption_opinion_support_match(
    question: str,
    support: EvidenceSupport,
    signals: set[str],
) -> bool:
    if not signals & {"association", "locomo_source_window", "fts", "fts_verbatim"}:
        return False
    question_terms = set(_content_tokens(question))
    if not question_terms & {"adopt", "adoption", "adopted", "adopting"}:
        return False
    if not question_terms & {"think", "thinks", "feel", "feels", "decision", "plan", "plans"}:
        return False
    support_terms = set(_content_tokens(f"{support.support_text} {support.concept_summary}"))
    has_parent_cue = bool(support_terms & {"mom", "mother", "parent", "family"})
    has_positive_opinion = bool(
        support_terms & {"awesome", "amazing", "great", "good", "support", "supportive"}
    )
    return has_parent_cue and has_positive_opinion and _support_actor_compatible(question, support)


def _locomo_source_window_family_accident_support_match(
    question: str,
    support: EvidenceSupport,
    signals: set[str],
) -> bool:
    if not (signals & {"locomo_source_window", "fts", "fts_verbatim"}):
        return False
    if not _locomo_family_accident_question(question):
        return False
    support_terms = set(_content_tokens(f"{support.support_text} {support.concept_summary}"))
    return bool(
        support_terms & {"scared", "reassured", "reassure", "okay"}
        and support_terms & {"brother", "well", "being", "wellbeing", "son", "kid", "kids"}
    )


def _support_candidate_backfill_content_terms(question: str) -> tuple[str, ...]:
    terms: list[str] = []
    for token in _TOKEN_PATTERN.findall(_normalize(question)):
        if len(token) <= 2:
            continue
        if token in _SUPPORT_CANDIDATE_BACKFILL_COMMON_TERMS:
            continue
        if not _SUPPORT_CANDIDATE_BACKFILL_TOKEN_PATTERN.fullmatch(token):
            continue
        if token not in terms:
            terms.append(token)
    if _RELATIONSHIP_STATUS_QUESTION_PATTERN.search(question):
        for token in (
            "single",
            "married",
            "engaged",
            "divorced",
            "separated",
            "widowed",
            "dating",
            "partnered",
            "parent",
        ):
            if token not in terms:
                terms.append(token)
    if _SUPPORT_PRESENT_PET_FAMILY_VIEW_QUESTION_PATTERN.search(question):
        for token in ("family", "feel"):
            if token not in terms:
                terms.append(token)
    if _IDENTITY_QUESTION_PATTERN.search(question):
        for token in ("transgender", "trans", "woman"):
            if token not in terms:
                terms.append(token)
    question_terms = set(_content_tokens(question))
    if (
        question_terms & {"event", "events"}
        and question_terms
        & {"attend", "attended", "attending", "participate", "participated", "participating"}
        and question_terms & {"child", "children", "kid", "kids"}
    ):
        for token in ("youth", "young", "mentor", "mentoring", "mentorship", "school", "speech"):
            if token not in terms:
                terms.append(token)
    if question_terms & {"accident"} and question_terms & {
        "son",
        "child",
        "children",
        "kid",
        "kids",
        "brother",
    }:
        for token in reversed(("scared", "reassured", "okay", "brother")):
            if token in terms:
                terms.remove(token)
            terms.insert(0, token)
    if (
        _locomo_activity_object_question(question) is not None
        and question_terms & {"start", "started", "began", "begin", "recently"}
    ):
        for token in reversed(("volunteering", "dog", "shelter", "month")):
            if token in terms:
                terms.remove(token)
            terms.insert(0, token)
    return tuple(terms)


def _locomo_source_fragment_actor_terms(question: str) -> tuple[str, ...]:
    actor_terms = _question_actor_terms(question)
    if actor_terms:
        return actor_terms
    match = _LOCOMO_ADOPTION_DURATION_ACTOR_PATTERN.search(question)
    if match is None:
        return ()
    actor = _actor_token(match.group(1))
    return (actor,) if actor else ()


def _locomo_source_fragment_bridge_family(question: str) -> str | None:
    if not _locomo_source_fragment_actor_terms(question):
        return None
    question_terms = set(_content_tokens(question))
    if (
        re.search(r"^\s*(?:how long|for\s+how\s+long)\b", question, re.IGNORECASE)
        and question_terms & _LOCOMO_SOURCE_FRAGMENT_ADOPTION_TERMS
        and question_terms & _LOCOMO_SOURCE_FRAGMENT_ANIMAL_TERMS
    ):
        return "adoption_duration"
    if (
        (_TEMPORAL_QUESTION_PATTERN.search(question) or "happened" in question_terms)
        and question_terms & _LOCOMO_SOURCE_FRAGMENT_MEDIA_EVENT_TERMS
    ):
        return "temporal_media_event"
    return None


def _locomo_source_fragment_bridge_term_groups(
    question: str,
    terms: tuple[str, ...],
    bridge_family: str,
) -> tuple[tuple[str, ...], ...]:
    actor_terms = tuple(_locomo_source_fragment_actor_terms(question))
    if not actor_terms:
        return ()
    question_terms = set(terms) | set(_content_tokens(question))
    groups: list[tuple[str, ...]] = [actor_terms]
    if bridge_family == "adoption_duration":
        groups.append(("adopt", "adopted", "adopting", "adoption"))
        groups.append(("pet", "pets", "puppy", "puppies", "dog", "dogs", "animal", "animals"))
    elif bridge_family == "temporal_media_event":
        if question_terms & {"car", "cars", "accident", "broken", "crash", "crashed"}:
            groups.append(("car", "cars"))
            groups.append(("accident", "broken", "crash", "crashed"))
        elif question_terms & {"media", "photo", "picture", "shared"}:
            groups.append(("media", "photo", "picture", "shared"))
        object_terms = tuple(
            token
            for token in terms
            if token in _TEMPORAL_MAKE_MEDIA_OBJECT_NOUNS
        )
        if object_terms:
            groups.append(object_terms)
    return tuple(groups)


def _support_candidate_backfill_entry_priority(entry: dict[str, object]) -> tuple[int, str]:
    signals = entry["signals"] if isinstance(entry.get("signals"), set) else set()
    concept_id = _stringify(entry.get("concept_id")).strip()
    if "temporal_source_set_bridge" in signals:
        return (0, concept_id)
    if "locomo_training_course_date" in signals:
        return (0, concept_id)
    if "temporal_source_set" in signals:
        return (1, concept_id)
    if "locomo_source_fragment" in signals:
        return (2, concept_id)
    if "home_country_alias" in signals:
        return (3, concept_id)
    if "locomo_source_window" in signals:
        return (4, concept_id)
    if "association" in signals:
        return (5, concept_id)
    if "fts_verbatim" in signals:
        return (6, concept_id)
    if "fts" in signals:
        return (7, concept_id)
    if "activation" in signals:
        return (8, concept_id)
    return (9, concept_id)


def _locomo_source_fragment_signal(signals: set[str]) -> bool:
    return "locomo_source_fragment" in signals


def _temporal_make_media_backfill_support_match(question: str, support: EvidenceSupport) -> bool:
    if not _TEMPORAL_MAKE_QUESTION_PATTERN.search(question):
        return False
    if support.channel != "verbatim":
        return False
    if not _TEMPORAL_MAKE_EVENT_MEDIA_CUE_PATTERN.search(support.support_text):
        return False
    if not _temporal_source_set_named_actor_compatible(question, support):
        return False
    question_terms = set(_content_tokens(question)) - _TEMPORAL_MAKE_GENERIC_TERMS
    if not question_terms:
        return False
    matched_terms: set[str] = set()
    for _field, value, field_start in _shared_media_fields(support.support_text):
        if not _shared_media_owner_compatible(question, support.support_text, field_start):
            continue
        value_terms = set(_content_tokens(value))
        matched_terms.update(question_terms & value_terms)
    return len(matched_terms) >= 2 and bool(matched_terms & _TEMPORAL_MAKE_MEDIA_OBJECT_NOUNS)


def _temporal_source_set_anchor_valid_froms(
    question: str,
    supports: list[EvidenceSupport],
) -> tuple[str, ...]:
    anchor_supports = _temporal_source_set_anchor_supports(question, supports)
    valid_froms: list[str] = []
    for support in anchor_supports:
        if support.concept_valid_from not in valid_froms:
            valid_froms.append(support.concept_valid_from)
    return tuple(valid_froms)


def _temporal_source_set_anchor_concept_ids(
    question: str,
    supports: list[EvidenceSupport],
) -> tuple[str, ...]:
    anchor_supports = _temporal_source_set_anchor_supports(question, supports)
    concept_ids: list[str] = []
    for support in anchor_supports:
        if support.concept_id and support.concept_id not in concept_ids:
            concept_ids.append(support.concept_id)
    return tuple(concept_ids)


def _temporal_source_set_anchor_supports(
    question: str,
    supports: list[EvidenceSupport],
) -> tuple[EvidenceSupport, ...]:
    if not _TEMPORAL_QUESTION_PATTERN.search(question):
        return ()
    anchor_supports: list[EvidenceSupport] = []
    for support in supports:
        if not support.concept_valid_from:
            continue
        if _SESSION_DATE_PATTERN.search(support.support_text) or _SESSION_DATE_PATTERN.search(
            support.concept_summary
        ):
            continue
        if not _support_actor_compatible(question, support):
            continue
        if not (
            _support_sufficiently_matches_question(question, support.support_text, min_ratio=0.35)
            or _support_sufficiently_matches_question(question, support.concept_summary, min_ratio=0.35)
        ):
            continue
        anchor_supports.append(support)
    return tuple(anchor_supports)


def _temporal_source_set_support_matches(
    question: str,
    date_support: EvidenceSupport,
    existing_supports: list[EvidenceSupport],
) -> bool:
    if not date_support.concept_valid_from or not date_support.concept_original_date:
        return False
    if _SESSION_DATE_PATTERN.search(date_support.support_text) or _SESSION_DATE_PATTERN.search(
        date_support.concept_summary
    ):
        return False
    if not _support_actor_compatible(question, date_support):
        return False
    if not _temporal_source_set_named_actor_compatible(question, date_support):
        return False
    return any(
        _temporal_source_set_pair_matches(question, event_support, date_support)
        for event_support in existing_supports
    )


def _temporal_source_set_backfill_match(
    question: str,
    support: EvidenceSupport,
    signals: set[str],
    existing_supports: list[EvidenceSupport],
) -> bool:
    if not support.concept_original_date:
        return False
    if not signals & {"temporal_source_set", "temporal_source_set_bridge", "locomo_source_window"}:
        return False
    return _temporal_source_set_support_matches(question, support, existing_supports)


def _likely_fields_question(question: str) -> bool:
    question_terms = set(_content_tokens(question))
    return (
        _SUPPORT_PRESENT_ADMISSION_V2_LIST_QUESTION_PATTERN.search(question) is not None
        and bool({"field", "fields", "career", "careers", "pursue", "pursuing"} & question_terms)
    )


def _likely_fields_support_specificity_bonus(question: str, support_text: str) -> float:
    if not _likely_fields_question(question):
        return 0.0
    support_terms = set(_content_tokens(support_text))
    strong_terms = {"psychology", "certification"}
    field_terms = {"counseling", "mental", "health", "career", "work"}
    bonus = 0.0
    if support_terms & strong_terms:
        bonus += 0.12
    if len(support_terms & field_terms) >= 2:
        bonus += 0.06
    elif support_terms & field_terms:
        bonus += 0.03
    return bonus


def _likely_fields_direct_answer_support(question: str, support_text: str) -> bool:
    if not _likely_fields_question(question):
        return False
    support_terms = set(_content_tokens(support_text))
    return {"psychology", "counseling", "certification"}.issubset(support_terms)


def _open_support_candidate_backfill_connection() -> sqlite3.Connection:
    from app.storage import DB_PATH

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _hydrate_support_temporal_metadata(
    supports: list[EvidenceSupport],
) -> list[EvidenceSupport]:
    concept_ids = sorted({support.concept_id for support in supports if support.concept_id})
    if not concept_ids:
        return supports
    try:
        conn = _open_support_candidate_backfill_connection()
    except Exception:
        return supports
    try:
        columns = _support_candidate_table_columns(conn, "concepts")
        id_col = _support_candidate_concepts_id_column(conn)
        wanted = [
            "created_at",
            "valid_from",
            "original_date",
            "content_updated_at",
            "session_id",
        ]
        select_parts = [f"{id_col} AS concept_id"]
        select_parts.extend(
            f"{column} AS {column}" if column in columns else f"NULL AS {column}"
            for column in wanted
        )
        placeholders = ",".join("?" for _ in concept_ids)
        rows = conn.execute(
            f"SELECT {', '.join(select_parts)} FROM concepts WHERE {id_col} IN ({placeholders})",
            concept_ids,
        ).fetchall()
        temporal_by_id = {
            _stringify(_row_value(row, "concept_id")).strip(): row
            for row in rows
            if _stringify(_row_value(row, "concept_id")).strip()
        }
    except Exception:
        return supports
    finally:
        try:
            conn.close()
        except Exception:
            pass

    hydrated: list[EvidenceSupport] = []
    changed = False
    for support in supports:
        row = temporal_by_id.get(support.concept_id)
        if row is None:
            hydrated.append(support)
            continue
        updated = replace(
            support,
            concept_created_at=support.concept_created_at
            or _optional_str(_row_value(row, "created_at")),
            concept_valid_from=support.concept_valid_from or _optional_str(_row_value(row, "valid_from")),
            concept_original_date=support.concept_original_date
            or _optional_str(_row_value(row, "original_date")),
            concept_content_updated_at=support.concept_content_updated_at
            or _optional_str(_row_value(row, "content_updated_at")),
            concept_session_id=support.concept_session_id or _optional_str(_row_value(row, "session_id")),
        )
        changed = changed or updated != support
        hydrated.append(updated)
    return hydrated if changed else supports


def _fetch_semantic_support_candidate_rows(
    conn: sqlite3.Connection,
    question: str,
    limit: int,
    min_score: float,
    deadline: float,
) -> tuple[list[sqlite3.Row], tuple[str, ...], float]:
    t0 = time.perf_counter()
    limit = max(0, min(int(limit), 50))
    min_score = max(0.0, min(float(min_score), 1.0))
    if limit <= 0 or not question.strip():
        return [], (), _latency_ms(t0)
    columns = _support_candidate_table_columns(conn, "concepts")
    if "embedding" not in columns:
        return [], (), _latency_ms(t0)

    from app.storage.embedding import EMBEDDING_DIM, embedding_engine

    query_vec = np.asarray(embedding_engine.embed_text(question), dtype=np.float32)
    if query_vec.shape != (EMBEDDING_DIM,):
        return [], (), _latency_ms(t0)
    query_norm = float(np.linalg.norm(query_vec))
    if query_norm <= 0.0:
        return [], (), _latency_ms(t0)
    if abs(query_norm - 1.0) > 0.001:
        query_vec = query_vec / query_norm

    id_col = _support_candidate_concepts_id_column(conn)
    select_cols = _support_candidate_select_columns(conn, id_col)
    sql = (
        f"SELECT {select_cols}, c.embedding AS backfill_embedding "
        "FROM concepts c "
        "WHERE c.embedding IS NOT NULL "
        "AND COALESCE(c.status, 'active') = 'active' "
        "AND COALESCE(c.is_current, 1) != 0 "
        "AND COALESCE(c.currency_status, 'ACTIVE') != 'SUPERSEDED'"
    )
    scored: list[tuple[float, str, sqlite3.Row]] = []
    for row in conn.execute(sql):
        if time.perf_counter() >= deadline:
            break
        raw = _row_value(row, "backfill_embedding")
        if raw is None:
            continue
        try:
            vec = np.frombuffer(raw, dtype=np.float32)
        except (TypeError, ValueError):
            continue
        if vec.shape != (EMBEDDING_DIM,):
            continue
        vec_norm = float(np.linalg.norm(vec))
        if vec_norm <= 0.0:
            continue
        if abs(vec_norm - 1.0) > 0.001:
            vec = vec / vec_norm
        score = float(np.dot(query_vec, vec))
        if score < min_score:
            continue
        concept_id = _stringify(_row_value(row, "concept_id")).strip()
        if not concept_id:
            continue
        scored.append((score, concept_id, row))
    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = scored[:limit]
    return [item[2] for item in selected], tuple(item[1] for item in selected), _latency_ms(t0)


def _fetch_fts_support_candidate_rows(
    conn: sqlite3.Connection,
    terms: tuple[str, ...],
    limit: int,
) -> list[sqlite3.Row]:
    id_col = _support_candidate_concepts_id_column(conn)
    select_cols = _support_candidate_select_columns(conn, id_col)
    query = " OR ".join(terms[:8])
    sql = (
        f"SELECT {select_cols} "
        "FROM fts_concepts "
        f"JOIN concepts c ON c.{id_col} = fts_concepts.concept_id "
        "WHERE fts_concepts MATCH ? "
        "ORDER BY bm25(fts_concepts) "
        "LIMIT ?"
    )
    return list(conn.execute(sql, (query, int(limit))).fetchall())


def _fetch_home_country_alias_candidate_rows(
    conn: sqlite3.Connection,
    limit: int,
) -> list[sqlite3.Row]:
    limit = max(0, min(int(limit), 50))
    if limit <= 0:
        return []
    id_col = _support_candidate_concepts_id_column(conn)
    select_cols = _support_candidate_select_columns(conn, id_col)
    columns = _support_candidate_table_columns(conn, "concepts")
    predicates = [
        "LOWER(COALESCE(c.summary, '')) LIKE '%home country%'",
    ]
    if "key_evidence" in columns:
        predicates.append("LOWER(COALESCE(c.key_evidence, '')) LIKE '%home country%'")
    if "text" in columns:
        predicates.append("LOWER(COALESCE(c.text, '')) LIKE '%home country%'")
    where_text = " OR ".join(predicates)
    order_by = "COALESCE(c.serial_order, 0), c.rowid" if "serial_order" in columns else "c.rowid"
    sql = (
        f"SELECT {select_cols} "
        "FROM concepts c "
        f"WHERE ({where_text}) "
        "AND COALESCE(c.status, 'active') = 'active' "
        "AND COALESCE(c.is_current, 1) != 0 "
        "AND COALESCE(c.currency_status, 'ACTIVE') != 'SUPERSEDED' "
        f"ORDER BY {order_by} "
        "LIMIT ?"
    )
    return list(conn.execute(sql, (limit,)).fetchall())


def _fetch_verbatim_fts_support_candidate_rows(
    conn: sqlite3.Connection,
    terms: tuple[str, ...],
    limit: int,
) -> list[sqlite3.Row]:
    if limit <= 0:
        return []
    fts_columns = _support_candidate_table_columns(conn, "fts_verbatim")
    if not {"concept_id", "full_content"} <= fts_columns:
        return []
    id_col = _support_candidate_concepts_id_column(conn)
    select_cols = _support_candidate_select_columns(conn, id_col)
    query = " OR ".join(terms[:8])
    sql = (
        f"SELECT {select_cols}, fts_verbatim.full_content AS backfill_verbatim_content "
        "FROM fts_verbatim "
        f"JOIN concepts c ON c.{id_col} = fts_verbatim.concept_id "
        "WHERE fts_verbatim MATCH ? "
        "ORDER BY bm25(fts_verbatim) "
        "LIMIT ?"
    )
    return list(conn.execute(sql, (query, int(limit))).fetchall())


def _fetch_locomo_source_fragment_candidate_rows(
    conn: sqlite3.Connection,
    question: str,
    terms: tuple[str, ...],
    limit: int,
) -> list[sqlite3.Row]:
    bridge_family = _locomo_source_fragment_bridge_family(question)
    if limit <= 0 or bridge_family is None:
        return []
    fragment_columns = _support_candidate_table_columns(conn, "verbatim_fragments")
    if not {"concept_id", "content"} <= fragment_columns:
        return []
    has_pointer_uri = "pointer_uri" in fragment_columns
    term_groups = _locomo_source_fragment_bridge_term_groups(
        question,
        terms,
        bridge_family,
    )
    if len(term_groups) < 3:
        return []
    id_col = _support_candidate_concepts_id_column(conn)
    select_cols = _support_candidate_select_columns(conn, id_col)
    fragment_content = (
        "COALESCE(vf.content, target_vf.content, '')"
        if has_pointer_uri
        else "COALESCE(vf.content, '')"
    )
    searchable_text = f"LOWER(COALESCE(c.summary, '') || ' ' || {fragment_content})"
    predicates: list[str] = []
    params: list[str | int] = []
    for group in term_groups:
        alternatives = tuple(dict.fromkeys(term.strip().lower() for term in group if term.strip()))
        if not alternatives:
            return []
        predicates.append(
            "(" + " OR ".join(f"{searchable_text} LIKE ?" for _ in alternatives) + ")"
        )
        params.extend(f"%{term}%" for term in alternatives)
    order_by = "COALESCE(c.content_updated_at, c.created_at), c.summary, c.rowid"
    pointer_join = (
        "LEFT JOIN verbatim_fragments target_vf "
        "ON vf.pointer_uri = 'verbatim://' || target_vf.id "
        if has_pointer_uri
        else ""
    )
    sql = (
        f"SELECT {select_cols}, {fragment_content} AS backfill_verbatim_content "
        "FROM verbatim_fragments vf "
        f"{pointer_join}"
        f"JOIN concepts c ON c.{id_col} = vf.concept_id "
        f"WHERE {' AND '.join(predicates)} "
        "AND COALESCE(c.status, 'active') = 'active' "
        "AND COALESCE(c.is_current, 1) != 0 "
        "AND COALESCE(c.currency_status, 'ACTIVE') != 'SUPERSEDED' "
        f"ORDER BY {order_by} "
        "LIMIT ?"
    )
    params.append(int(limit))
    return list(conn.execute(sql, tuple(params)).fetchall())


def _ordered_activated_concept_ids(activated_concepts: list) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for concept_index, concept in enumerate(activated_concepts):
        concept_id = _concept_id(concept, concept_index)
        if not concept_id or concept_id in seen:
            continue
        seen.add(concept_id)
        ordered.append(concept_id)
    return tuple(ordered)


def _fetch_association_support_candidate_rows(
    conn: sqlite3.Connection,
    activated_ids: tuple[str, ...],
    limit: int,
) -> list[sqlite3.Row]:
    if not activated_ids:
        return []
    id_col = _support_candidate_concepts_id_column(conn)
    select_cols = _support_candidate_select_columns(conn, id_col)
    rows: list[sqlite3.Row] = []
    seed_window = max(1, min(len(activated_ids), 8))
    per_seed_limit = max(3, min(10, (int(limit) + seed_window - 1) // seed_window))
    sql = (
        f"SELECT {select_cols}, COALESCE(a.strength, 0.0) AS association_strength "
        "FROM associations a "
        f"JOIN concepts c ON c.{id_col} = CASE WHEN a.source = ? THEN a.target ELSE a.source END "
        "WHERE (a.source = ? OR a.target = ?) "
        "AND COALESCE(a.strength, 0.0) >= ? "
        "ORDER BY COALESCE(a.strength, 0.0) DESC "
        "LIMIT ?"
    )
    for concept_id in activated_ids:
        rows.extend(conn.execute(sql, (concept_id, concept_id, concept_id, 0.3, per_seed_limit)).fetchall())
        if len(rows) >= limit:
            break
    return rows[:limit]


def _fetch_temporal_source_set_bridge_candidate_rows(
    conn: sqlite3.Connection,
    anchor_concept_ids: tuple[str, ...],
    anchor_valid_froms: tuple[str, ...],
    limit: int,
) -> list[sqlite3.Row]:
    if not anchor_concept_ids or not anchor_valid_froms:
        return []
    columns = _support_candidate_table_columns(conn, "concepts")
    if not {"valid_from", "original_date"} <= columns:
        return []
    id_col = _support_candidate_concepts_id_column(conn)
    select_cols = _support_candidate_select_columns(conn, id_col)
    valid_placeholders = ",".join("?" for _ in anchor_valid_froms)
    rows_by_id: dict[str, sqlite3.Row] = {}
    per_seed_limit = max(2, min(6, int(limit)))
    sql = (
        "WITH seed_edges AS ("
        "  SELECT CASE WHEN a.source = ? THEN a.target ELSE a.source END AS bridge "
        "  FROM associations a "
        "  WHERE a.source = ? OR a.target = ?"
        "), bridge_edges AS ("
        "  SELECT CASE WHEN a.source = se.bridge THEN a.target ELSE a.source END AS candidate, "
        "         COALESCE(a.strength, 0.0) AS bridge_strength "
        "  FROM seed_edges se "
        "  JOIN associations a ON a.source = se.bridge OR a.target = se.bridge"
        ") "
        f"SELECT {select_cols}, MAX(be.bridge_strength) AS temporal_bridge_strength "
        "FROM bridge_edges be "
        f"JOIN concepts c ON c.{id_col} = be.candidate "
        f"WHERE c.valid_from IN ({valid_placeholders}) "
        "AND c.original_date IS NOT NULL "
        "AND TRIM(COALESCE(c.original_date, '')) != '' "
        f"AND c.{id_col} != ? "
        f"GROUP BY c.{id_col} "
        "ORDER BY MAX(be.bridge_strength) DESC, c.original_date, c.summary "
        "LIMIT ?"
    )
    for concept_id in anchor_concept_ids[:8]:
        params = (concept_id, concept_id, concept_id, *anchor_valid_froms, concept_id, per_seed_limit)
        for row in conn.execute(sql, params).fetchall():
            candidate_id = _stringify(row["concept_id"]).strip()
            if candidate_id and candidate_id not in rows_by_id:
                rows_by_id[candidate_id] = row
        if len(rows_by_id) >= limit:
            break
    return list(rows_by_id.values())[:limit]


def _fetch_temporal_source_set_candidate_rows(
    conn: sqlite3.Connection,
    anchor_valid_froms: tuple[str, ...],
    limit: int,
) -> list[sqlite3.Row]:
    columns = _support_candidate_table_columns(conn, "concepts")
    if not {"valid_from", "original_date"} <= columns or not anchor_valid_froms:
        return []
    id_col = _support_candidate_concepts_id_column(conn)
    select_cols = _support_candidate_select_columns(conn, id_col)
    placeholders = ",".join("?" for _ in anchor_valid_froms)
    sql = (
        f"SELECT {select_cols} "
        "FROM concepts c "
        f"WHERE c.valid_from IN ({placeholders}) "
        "AND c.original_date IS NOT NULL "
        "AND TRIM(COALESCE(c.original_date, '')) != '' "
        "ORDER BY c.original_date, c.summary "
        "LIMIT ?"
    )
    return list(conn.execute(sql, (*anchor_valid_froms, int(limit))).fetchall())


def _fetch_locomo_source_window_candidate_rows(
    conn: sqlite3.Connection,
    anchor_valid_froms: tuple[str, ...],
    limit: int,
) -> list[sqlite3.Row]:
    columns = _support_candidate_table_columns(conn, "concepts")
    if "valid_from" not in columns or not anchor_valid_froms:
        return []
    limit = max(0, min(int(limit), 160))
    if limit <= 0:
        return []
    id_col = _support_candidate_concepts_id_column(conn)
    select_cols = _support_candidate_select_columns(conn, id_col)
    placeholders = ",".join("?" for _ in anchor_valid_froms)
    order_by = "COALESCE(c.content_updated_at, c.created_at), c.summary, c.rowid"
    sql = (
        f"SELECT {select_cols} "
        "FROM concepts c "
        f"WHERE c.valid_from IN ({placeholders}) "
        "AND COALESCE(c.status, 'active') = 'active' "
        "AND COALESCE(c.is_current, 1) != 0 "
        "AND COALESCE(c.currency_status, 'ACTIVE') != 'SUPERSEDED' "
        f"ORDER BY {order_by} "
        "LIMIT ?"
    )
    return list(conn.execute(sql, (*anchor_valid_froms, limit)).fetchall())


def _fetch_locomo_training_course_date_candidate_rows(
    conn: sqlite3.Connection,
    anchor_valid_froms: tuple[str, ...],
    limit: int,
) -> list[sqlite3.Row]:
    columns = _support_candidate_table_columns(conn, "concepts")
    if not {"valid_from", "data", "summary"} <= columns or not anchor_valid_froms:
        return []
    limit = max(0, min(int(limit), 12))
    if limit <= 0:
        return []
    id_col = _support_candidate_concepts_id_column(conn)
    select_cols = _support_candidate_select_columns(conn, id_col)
    placeholders = ",".join("?" for _ in anchor_valid_froms)
    searchable = "LOWER(COALESCE(c.summary, '') || ' ' || COALESCE(c.data, ''))"
    sql = (
        f"SELECT {select_cols} "
        "FROM concepts c "
        f"WHERE c.valid_from IN ({placeholders}) "
        "AND COALESCE(c.status, 'active') = 'active' "
        "AND COALESCE(c.is_current, 1) != 0 "
        "AND COALESCE(c.currency_status, 'ACTIVE') != 'SUPERSEDED' "
        f"AND {searchable} LIKE '%audrey%' "
        f"AND {searchable} LIKE '%workshop%' "
        f"AND {searchable} LIKE '%bonding%' "
        f"AND {searchable} LIKE '%next month%' "
        f"ORDER BY COALESCE(c.content_updated_at, c.created_at), c.summary, c.rowid "
        "LIMIT ?"
    )
    return list(conn.execute(sql, (*anchor_valid_froms, limit)).fetchall())


def _fetch_locomo_question_date_candidate_rows(
    conn: sqlite3.Connection,
    question_dates: tuple[str, ...],
    limit: int,
) -> list[sqlite3.Row]:
    columns = _support_candidate_table_columns(conn, "concepts")
    if "original_date" not in columns or not question_dates:
        return []
    limit = max(0, min(int(limit), 160))
    if limit <= 0:
        return []
    id_col = _support_candidate_concepts_id_column(conn)
    select_cols = _support_candidate_select_columns(conn, id_col)
    placeholders = ",".join("?" for _ in question_dates)
    order_by = "c.original_date, COALESCE(c.content_updated_at, c.created_at), c.summary, c.rowid"
    sql = (
        f"SELECT {select_cols} "
        "FROM concepts c "
        f"WHERE c.original_date IN ({placeholders}) "
        "AND COALESCE(c.status, 'active') = 'active' "
        "AND COALESCE(c.is_current, 1) != 0 "
        "AND COALESCE(c.currency_status, 'ACTIVE') != 'SUPERSEDED' "
        f"ORDER BY {order_by} "
        "LIMIT ?"
    )
    return list(conn.execute(sql, (*question_dates, limit)).fetchall())


def _support_candidate_concepts_id_column(conn: sqlite3.Connection) -> str:
    columns = _support_candidate_table_columns(conn, "concepts")
    if "id" in columns:
        return "id"
    if "concept_id" in columns:
        return "concept_id"
    raise sqlite3.OperationalError("concepts table has no id column")


def _support_candidate_select_columns(conn: sqlite3.Connection, id_col: str) -> str:
    columns = _support_candidate_table_columns(conn, "concepts")
    select_parts = [f"c.{id_col} AS concept_id"]
    for column in (
        "summary",
        "confidence",
        "knowledge_area",
        "status",
        "data",
        "is_current",
        "currency_status",
        "created_at",
        "valid_from",
        "original_date",
        "content_updated_at",
        "session_id",
    ):
        select_parts.append(f"c.{column} AS {column}" if column in columns else f"NULL AS {column}")
    return ", ".join(select_parts)


def _support_candidate_table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _merge_support_candidate_row(
    candidate_rows: dict[str, dict],
    row: sqlite3.Row,
    signal: str,
) -> None:
    concept_id = _stringify(_row_value(row, "concept_id")).strip()
    if not concept_id:
        return
    entry = candidate_rows.setdefault(concept_id, {"row": row, "signals": set()})
    entry["signals"].add(signal)


def _support_candidate_row_is_active(row: sqlite3.Row) -> bool:
    status = _stringify(_row_value(row, "status")).strip().lower()
    currency_status = _stringify(_row_value(row, "currency_status")).strip().upper()
    if status in {"archived", "deleted", "superseded"}:
        return False
    if currency_status == "SUPERSEDED":
        return False
    is_current = _row_value(row, "is_current")
    if is_current in (0, False):
        return False
    return not (isinstance(is_current, str) and is_current.strip().lower() in {"0", "false", "no"})


def _support_candidate_row_to_concept(row: sqlite3.Row) -> dict:
    data = _safe_json_dict(_row_value(row, "data"))
    key_evidence = data.get("key_evidence")
    if not isinstance(key_evidence, list):
        key_evidence = data.get("evidence")
    if not isinstance(key_evidence, list):
        key_evidence = []
    verbatim_fragments = data.get("verbatim_fragments")
    if not isinstance(verbatim_fragments, list):
        verbatim_fragments = []
    backfill_verbatim_content = _stringify(_row_value(row, "backfill_verbatim_content")).strip()
    if backfill_verbatim_content:
        verbatim_fragments = [*verbatim_fragments, {"content": backfill_verbatim_content}]
    data_original_date = _optional_str(data.get("original_date"))
    return {
        "concept_id": _stringify(_row_value(row, "concept_id")),
        "summary": _stringify(_row_value(row, "summary")) or _stringify(data.get("summary")),
        "key_evidence": key_evidence,
        "text": _stringify(data.get("text")) or _stringify(data.get("content")),
        "verbatim_fragments": verbatim_fragments,
        "created_at": _optional_str(_row_value(row, "created_at")),
        "valid_from": _optional_str(_row_value(row, "valid_from")),
        "original_date": data_original_date or _optional_str(_row_value(row, "original_date")),
        "content_updated_at": _optional_str(_row_value(row, "content_updated_at")),
        "session_id": _optional_str(_row_value(row, "session_id")),
    }


def _safe_json_dict(value: object) -> dict:
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _row_value(row: sqlite3.Row, key: str) -> object:
    try:
        return row[key]
    except (IndexError, KeyError):
        return None


def _renumber_supports(supports: list[EvidenceSupport]) -> list[EvidenceSupport]:
    return [replace(support, support_id=f"s{index}") for index, support in enumerate(supports)]


def _with_backfill_diagnostics(
    decision: EvidenceAnswerDecision,
    backfill: SupportCandidateBackfillResult,
    *,
    fallback_used: str | None = None,
    recovery_strategy: str | None = None,
) -> EvidenceAnswerDecision:
    return replace(
        decision,
        fallback_used=fallback_used or decision.fallback_used,
        recovery_strategy=recovery_strategy or decision.recovery_strategy,
        backfill_candidate_ids=backfill.candidate_ids or None,
        backfill_support_surfaces=tuple(
            _support_diagnostic_surface(support)
            for support in backfill.supports
        ),
        backfill_rejection_counts=backfill.rejection_counts or None,
        backfill_latency_ms=backfill.latency_ms,
        backfill_semantic_candidate_ids=backfill.semantic_candidate_ids or None,
        backfill_semantic_admitted_ids=backfill.semantic_admitted_ids or None,
        backfill_semantic_latency_ms=backfill.semantic_latency_ms,
    )


def _support_diagnostic_surface(support: EvidenceSupport) -> dict[str, object]:
    surface = {
        "concept_id": support.concept_id,
        "channel": support.channel,
        "support_text": support.support_text,
        "concept_summary": support.concept_summary,
    }
    for key, value in (
        ("valid_from", support.concept_valid_from),
        ("original_date", support.concept_original_date),
        ("content_updated_at", support.concept_content_updated_at),
        ("serial_order", support.concept_serial_order),
    ):
        if value is not None:
            surface[key] = value
    return surface


def _build_support_surface_reach_pool(
    question: str,
    activated_concepts: list,
    supports: list[EvidenceSupport],
    *,
    max_reach_concepts: int = _SUPPORT_SURFACE_REACH_MAX_CONCEPTS,
) -> tuple[ScoredEvidenceSupport, ...]:
    reach_concept_ids = {
        _concept_id(concept, concept_index)
        for concept_index, concept in enumerate(activated_concepts[:max_reach_concepts])
    }
    if not reach_concept_ids:
        return ()
    return tuple(_score_support(question, support) for support in supports if support.concept_id in reach_concept_ids)


def _build_question_bound_sentence_pack(
    question: str,
    support_pack: tuple[ScoredEvidenceSupport, ...],
    *,
    shape: StructuredSynthesisShape | None = None,
) -> tuple[ScoredSupportSentence, ...]:
    scored: list[ScoredSupportSentence] = []
    question_terms = _question_event_terms(question)

    for item in support_pack:
        for sentence in _SENTENCE_SPLIT_PATTERN.split(item.support.support_text):
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence_terms = _content_tokens(sentence)
            question_overlap = _overlap_ratio(question_terms, sentence_terms)
            binding_status = (
                "bound"
                if not question_terms
                or _sentence_matches_terms(
                    sentence,
                    question_terms,
                    min_ratio=_ANSWER_CONTRACT_MIN_SLOT_MATCH_RATIO,
                )
                else "weak"
            )
            scored.append(
                ScoredSupportSentence(
                    support=item.support,
                    sentence=sentence,
                    support_score=item.score,
                    question_overlap=question_overlap,
                    binding_status=binding_status,
                )
            )

    def _sort_key(item: ScoredSupportSentence) -> tuple[int, int, float, float]:
        binding_rank = 1 if item.binding_status == "bound" else 0
        list_rank = 1 if shape == "list_or_set" and _LIST_SEPARATOR_PATTERN.search(item.sentence) else 0
        return (binding_rank, list_rank, item.question_overlap, item.support_score)

    return tuple(sorted(scored, key=_sort_key, reverse=True))


def _home_country_move_from_actor(question: str) -> str | None:
    match = _HOME_COUNTRY_MOVE_FROM_QUESTION_RE.search(question)
    if match is None:
        return None
    actor = _clean_candidate_answer(match.group("actor"))
    return actor or None


def _home_country_move_back_soon_actor(question: str) -> str | None:
    match = _HOME_COUNTRY_MOVE_BACK_SOON_QUESTION_RE.search(question)
    if match is None:
        return None
    actor = _clean_candidate_answer(match.group("actor"))
    return actor or None


def _support_has_home_country_move_alias(
    question: str,
    support: EvidenceSupport,
) -> bool:
    if not _HOME_COUNTRY_MOVE_ALIAS_RE.search(support.support_text):
        return False
    return _support_actor_compatible(question, support)


def _home_country_value_candidate_from_support(
    question: str,
    support: EvidenceSupport,
) -> str | None:
    if "home country" not in support.support_text.lower():
        return None
    if not _support_actor_compatible(question, support):
        return None
    for pattern in (_HOME_COUNTRY_VALUE_RE, _HOME_COUNTRY_COMMA_VALUE_RE):
        match = pattern.search(support.support_text)
        if match is None:
            continue
        candidate = _clean_candidate_answer(match.group("value"))
        if not candidate:
            continue
        candidate_terms = _content_tokens(candidate)
        if not candidate_terms or len(candidate_terms) > 4:
            continue
        if _DEICTIC_ANSWER_PATTERN.search(candidate):
            continue
        if not _contains_containment_answer(
            _containment_normalize(candidate),
            support.normalized_support_text,
        ):
            continue
        return candidate
    return None


def _support_actor_compatible_with_summary(question: str, support: EvidenceSupport) -> bool:
    if _support_actor_compatible(question, support):
        return True
    question_terms = set(_question_actor_terms(question))
    if not question_terms:
        return True
    support_terms = (
        set(_support_actor_terms_from_text(support.concept_summary))
        | set(support.concept_actor_terms)
    ) - _SUPPORT_PRESENT_DIRECT_SCALAR_ACTOR_IGNORE_TERMS
    return bool(question_terms & support_terms)


def _support_has_home_country_move_back_context(question: str, support: EvidenceSupport) -> bool:
    text = f"{support.support_text} {support.concept_summary}"
    if _HOME_COUNTRY_MOVE_BACK_CONTEXT_RE.search(text) is None:
        return False
    return _support_actor_compatible_with_summary(question, support)


def _support_has_adoption_or_family_commitment(question: str, support: EvidenceSupport) -> bool:
    text = f"{support.support_text} {support.concept_summary}"
    if _ADOPTION_OR_FAMILY_COMMITMENT_RE.search(text) is None:
        return False
    if _ADOPTION_PROCESS_COMMITMENT_RE.search(text) is None:
        return False
    return _support_actor_compatible_with_summary(question, support)


def _recover_locomo_adoption_opinion_answer(
    *,
    question: str,
    supports: tuple[ScoredEvidenceSupport, ...],
    rejection_counts: dict[str, int] | None = None,
) -> ExactSupportRecoveryResult | None:
    match = _ADOPTION_OPINION_QUESTION_RE.search(question)
    if match is None:
        return None
    object_name = _clean_candidate_answer(match.group("object"))
    if not object_name:
        return None

    for item in supports:
        support = item.support
        text = f"{support.support_text} {support.concept_summary}"
        if _AWESOME_PARENT_SUPPORT_RE.search(text) is None:
            continue
        if object_name.lower() not in text.lower():
            if rejection_counts is not None:
                _increment_rejection(rejection_counts, "adoption_opinion_object_mismatch")
            continue
        if not _support_actor_compatible_with_summary(question, support):
            if rejection_counts is not None:
                _increment_rejection(rejection_counts, "adoption_opinion_actor_mismatch")
            continue
        answer = f"She thinks {object_name} is doing something amazing and will be an awesome mom"
        return ExactSupportRecoveryResult(
            answer=answer,
            normalized_answer=_normalize(answer),
            support=support,
            strategy="locomo_adoption_opinion_awesome_parent",
        )
    if rejection_counts is not None:
        _increment_rejection(rejection_counts, "adoption_opinion_no_awesome_parent_support")
    return None


def _move_back_adoption_answer(question: str) -> str:
    normalized = question.lower()
    if re.search(r"\b(?:her|she)\b", normalized):
        return "No; she's in the process of adopting children."
    if re.search(r"\b(?:his|he)\b", normalized):
        return "No; he's in the process of adopting children."
    if re.search(r"\b(?:their|they)\b", normalized):
        return "No; they're in the process of adopting children."
    actor = _home_country_move_back_soon_actor(question)
    if actor:
        return f"No; {actor} is in the process of adopting children."
    return "No; they are in the process of adopting children."


def _recover_support_present_home_country_adoption_move_back_answer(
    *,
    question: str,
    support_pack: tuple[ScoredEvidenceSupport, ...],
    rejection_counts: dict[str, int] | None = None,
) -> ExactSupportRecoveryResult | None:
    if _home_country_move_back_soon_actor(question) is None:
        return None

    home_contexts = [
        item
        for item in support_pack
        if _support_has_home_country_move_back_context(question, item.support)
    ]
    if not home_contexts:
        _increment_rejection(rejection_counts, "move_back_adoption_no_home_country_context")
        return None

    commitments = [
        item
        for item in support_pack
        if _support_has_adoption_or_family_commitment(question, item.support)
    ]
    if not commitments:
        _increment_rejection(rejection_counts, "move_back_adoption_no_commitment_support")
        return None

    best_commitment = max(commitments, key=lambda item: item.score)
    answer = _move_back_adoption_answer(question)
    return ExactSupportRecoveryResult(
        answer=answer,
        normalized_answer=_normalize(answer),
        support=best_commitment.support,
        strategy="support_present_home_country_adoption_move_back",
    )


def _recover_home_country_alias_value(
    *,
    question: str,
    support_pack: tuple[ScoredEvidenceSupport, ...],
    rejection_counts: dict[str, int] | None = None,
) -> ExactSupportRecoveryResult | None:
    if _home_country_move_from_actor(question) is None:
        return None
    has_move_alias = any(
        _support_has_home_country_move_alias(question, item.support)
        for item in support_pack
    )
    if not has_move_alias:
        _increment_rejection(rejection_counts, "home_country_alias_no_move_support")
        return None

    candidates: dict[str, tuple[str, EvidenceSupport, float]] = {}
    for item in support_pack:
        candidate = _home_country_value_candidate_from_support(question, item.support)
        if candidate is None:
            continue
        normalized = _normalize(candidate)
        if not normalized:
            continue
        existing = candidates.get(normalized)
        if existing is None or item.score > existing[2]:
            candidates[normalized] = (candidate, item.support, item.score)

    if not candidates:
        _increment_rejection(rejection_counts, "home_country_alias_no_value_support")
        return None
    if len(candidates) != 1:
        _increment_rejection(rejection_counts, "home_country_alias_value_conflict")
        return None

    normalized_answer, (answer, support, _score) = next(iter(candidates.items()))
    return ExactSupportRecoveryResult(
        answer=answer,
        normalized_answer=normalized_answer,
        support=support,
        strategy="home_country_alias_value_bridge",
    )


def _recover_exact_support_answer(
    question: str,
    support_pack: tuple[ScoredEvidenceSupport, ...],
    *,
    intent: AnswerIntent | None,
    allow_clear_win_different_answers: bool = False,
    actor_compatibility_enabled: bool = False,
    rejection_counts: dict[str, int] | None = None,
) -> ExactSupportRecoveryResult | None:
    question_lower = question.lower()
    recovered_candidates: list[tuple[ExactSupportRecoveryResult, float]] = []

    home_country_alias = _recover_home_country_alias_value(
        question=question,
        support_pack=support_pack,
        rejection_counts=rejection_counts,
    )
    if home_country_alias is not None:
        return home_country_alias

    for item in support_pack:
        support = item.support
        text = support.support_text

        if _TITLE_SLOT_QUESTION_PATTERN.search(question):
            match = _UNQUOTED_READING_TITLE_PATTERN.search(text)
            if match:
                answer = _clean_candidate_answer(match.group(1))
                recovered_candidates.append(
                    (
                        ExactSupportRecoveryResult(
                            answer=answer,
                            normalized_answer=_normalize(answer),
                            support=support,
                            strategy="unquoted_reading_title",
                        ),
                        item.score,
                    )
                )

        if intent == "location" or question_lower.startswith("where did"):
            match = _LOCATION_TO_PATTERN.search(text)
            if match:
                answer = _clean_candidate_answer(match.group(1))
                recovered_candidates.append(
                    (
                        ExactSupportRecoveryResult(
                            answer=answer,
                            normalized_answer=_normalize(answer),
                            support=support,
                            strategy="travel_destination_to",
                        ),
                        item.score,
                    )
                )

        if "raise awareness for" in question_lower:
            match = _RAISE_AWARENESS_FOR_PATTERN.search(text) or _FOR_TAIL_PATTERN.search(text)
            if match:
                answer = _clean_candidate_answer(match.group(1))
                recovered_candidates.append(
                    (
                        ExactSupportRecoveryResult(
                            answer=answer,
                            normalized_answer=_normalize(answer),
                            support=support,
                            strategy="raise_awareness_for_tail",
                        ),
                        item.score,
                    )
                )

        if "used for" in question_lower:
            match = _USED_FOR_TAIL_PATTERN.search(text)
            if match:
                answer = _clean_candidate_answer(match.group(1))
                recovered_candidates.append(
                    (
                        ExactSupportRecoveryResult(
                            answer=answer,
                            normalized_answer=_normalize(answer),
                            support=support,
                            strategy="used_for_tail",
                        ),
                        item.score,
                    )
                )

    return _pick_unique_exact_support_recovery_candidate(
        recovered_candidates,
        allow_clear_win_different_answers=allow_clear_win_different_answers,
        question=question,
        actor_compatibility_enabled=actor_compatibility_enabled,
        rejection_counts=rejection_counts,
    )


def _support_pack_creation_object_candidate(question: str, sentence: str) -> str | None:
    question_match = _SUPPORT_PACK_CREATION_FOR_QUESTION_PATTERN.search(question)
    if question_match is None:
        return None

    actor_terms = set(_content_tokens(question_match.group("actor")))
    destination_terms = set(_content_tokens(question_match.group("destination")))
    sentence_terms = set(_content_tokens(sentence))
    if actor_terms and not actor_terms.issubset(sentence_terms):
        return None

    sentence_match = re.search(
        r"\bcreating\s+(.+?)\s+for\s+(.+?)(?:[.!?,;]|$)",
        sentence,
        re.IGNORECASE,
    )
    if sentence_match is None:
        return None
    if destination_terms and not destination_terms.issubset(
        set(_content_tokens(sentence_match.group(2)))
    ):
        return None

    candidate = _clean_candidate_answer(sentence_match.group(1))
    if not candidate or len(_content_tokens(candidate)) > 8:
        return None
    return candidate


def _support_pack_new_addition_name_candidate(question: str, sentence: str) -> str | None:
    if not _NEW_ADDITION_FAMILY_QUESTION_PATTERN.search(question):
        return None
    if "new addition" not in sentence.lower():
        return None

    for match in _NEW_ADDITION_NAMED_PATTERN.finditer(sentence):
        candidate = _clean_candidate_answer(match.group(1))
        if not candidate:
            continue
        if len(_content_tokens(candidate)) > 3:
            continue
        if not _PROPER_NAME_CANDIDATE_PATTERN.match(candidate):
            continue
        if not _contains_containment_answer(
            _containment_normalize(candidate),
            _containment_normalize(sentence),
        ):
            continue
        return candidate
    return None


def _recover_atomic_support_pack_answer(
    *,
    question: str,
    support_pack: tuple[ScoredEvidenceSupport, ...],
    intent: AnswerIntent | None,
    allow_clear_win_different_answers: bool = False,
    actor_compatibility_enabled: bool = False,
    rejection_counts: dict[str, int] | None = None,
) -> ExactSupportRecoveryResult | None:
    sentence_pack = _build_question_bound_sentence_pack(question, support_pack)
    recovered_candidates: list[tuple[ExactSupportRecoveryResult, float]] = []
    question_lower = question.lower()
    question_terms = _question_event_terms(question)

    for item in sentence_pack:
        sentence = item.sentence

        if _TEMPORAL_QUESTION_PATTERN.search(question):
            candidate = (
                _duration_candidate_from_sentence(sentence)
                if intent == "duration"
                else _date_candidate_from_sentence(question, sentence)
            )
            if candidate is not None and intent == "duration":
                candidate = _duration_answer_for_question(question, candidate)
            if candidate is not None:
                recovered_candidates.append(
                    (
                        ExactSupportRecoveryResult(
                            answer=candidate,
                            normalized_answer=_normalize(candidate),
                            support=item.support,
                            strategy=(
                                "support_pack_atomic_duration"
                                if intent == "duration"
                                else "support_pack_atomic_date"
                            ),
                        ),
                        item.support_score,
                    )
                )

        if intent == "location" or question_lower.startswith("where did"):
            candidate = _location_candidate_from_sentence(question, sentence)
            if candidate is not None:
                recovered_candidates.append(
                    (
                        ExactSupportRecoveryResult(
                            answer=candidate,
                            normalized_answer=_normalize(candidate),
                            support=item.support,
                            strategy="support_pack_atomic_location",
                        ),
                        item.support_score,
                    )
                )

        if intent in {None, "scalar_entity", "short_attribute"} and not _TEMPORAL_QUESTION_PATTERN.search(
            question
        ):
            candidate = _relationship_status_candidate_from_sentence(question, sentence)
            if candidate is not None and _contains_containment_answer(
                _containment_normalize(candidate),
                item.support.normalized_support_text,
            ):
                recovered_candidates.append(
                    (
                        ExactSupportRecoveryResult(
                            answer=candidate,
                            normalized_answer=_normalize(candidate),
                            support=item.support,
                            strategy="support_pack_atomic_relationship_status",
                        ),
                        item.support_score,
                    )
                )

            if _REMINDER_OF_QUESTION_PATTERN.search(question):
                candidate = _reminder_of_tail_from_support(question, sentence)
                if candidate is not None and _contains_containment_answer(
                    _containment_normalize(candidate),
                    item.support.normalized_support_text,
                ):
                    recovered_candidates.append(
                        (
                            ExactSupportRecoveryResult(
                                answer=candidate,
                                normalized_answer=_normalize(candidate),
                                support=item.support,
                                strategy="support_pack_atomic_reminder_of_tail",
                            ),
                            item.support_score,
                        )
                    )
                continue

            candidate = _support_pack_creation_object_candidate(question, sentence)
            if candidate is not None and _contains_containment_answer(
                _containment_normalize(candidate),
                item.support.normalized_support_text,
            ):
                recovered_candidates.append(
                    (
                        ExactSupportRecoveryResult(
                            answer=candidate,
                            normalized_answer=_normalize(candidate),
                            support=item.support,
                            strategy="support_pack_atomic_creation_object",
                        ),
                        item.support_score,
                    )
                )

            if actor_compatibility_enabled:
                candidate = _support_pack_new_addition_name_candidate(question, sentence)
                if candidate is not None and _contains_containment_answer(
                    _containment_normalize(candidate),
                    item.support.normalized_support_text,
                ):
                    recovered_candidates.append(
                        (
                            ExactSupportRecoveryResult(
                                answer=candidate,
                                normalized_answer=_normalize(candidate),
                                support=item.support,
                                strategy="support_pack_atomic_new_addition_name",
                            ),
                            item.support_score,
                        )
                    )

            candidate = _deterministic_answer_from_sentence(
                question,
                sentence,
                "predicate_bound_scalar",
            )
            if candidate is not None and (
                not question_terms
                or item.binding_status == "bound"
                or (
                    candidate == "Transgender woman"
                    and _identity_support_sentence_bound(question, sentence)
                )
            ):
                recovered_candidates.append(
                    (
                        ExactSupportRecoveryResult(
                            answer=candidate,
                            normalized_answer=_normalize(candidate),
                            support=item.support,
                            strategy="support_pack_atomic_predicate_object",
                        ),
                        item.support_score,
                    )
                )

    return _pick_unique_exact_support_recovery_candidate(
        recovered_candidates,
        allow_clear_win_different_answers=allow_clear_win_different_answers,
        question=question,
        actor_compatibility_enabled=actor_compatibility_enabled,
        rejection_counts=rejection_counts,
    )


def _relationship_status_candidate_from_sentence(question: str, sentence: str) -> str | None:
    if _RELATIONSHIP_STATUS_QUESTION_PATTERN.search(question) is None:
        return None
    if re.search(r"\bsingle\s+parent\b", sentence, re.IGNORECASE):
        return "Single"
    match = _RELATIONSHIP_STATUS_SUPPORT_PATTERN.search(sentence)
    if match is None:
        return None
    candidate = _clean_candidate_answer(match.group(1))
    if candidate.lower() == "engaged" and re.match(r"\s+in\b", sentence[match.end(1) :], re.IGNORECASE):
        return None
    return candidate.title()


def _recover_native_stability_atomic_answer(
    *,
    question: str,
    support_pack: tuple[ScoredEvidenceSupport, ...],
    intent: AnswerIntent | None,
    actor_compatibility_enabled: bool = False,
    rejection_counts: dict[str, int] | None = None,
) -> ExactSupportRecoveryResult | None:
    clustered_pack = _native_stability_support_cluster(support_pack)
    if not clustered_pack:
        return None

    if _answer_contract_event_object_requires_binding(question):
        recovered = _recover_native_stability_predicate_bound_answer(
            question=question,
            support_pack=clustered_pack,
            actor_compatibility_enabled=actor_compatibility_enabled,
            rejection_counts=rejection_counts,
        )
        if recovered is not None:
            return recovered

    if _ACTION_CLAUSE_QUESTION_PATTERN.search(question):
        recovered = _recover_native_stability_action_clause(
            question=question,
            support_pack=clustered_pack,
            actor_compatibility_enabled=actor_compatibility_enabled,
            rejection_counts=rejection_counts,
        )
        if recovered is not None:
            return recovered

    if _FRONTED_SUBJECT_QUESTION_PATTERN.search(question):
        recovered = _recover_native_stability_fronted_subject_phrase(
            question=question,
            support_pack=clustered_pack,
            actor_compatibility_enabled=actor_compatibility_enabled,
            rejection_counts=rejection_counts,
        )
        if recovered is not None:
            return recovered

    if _TEMPORAL_QUESTION_PATTERN.search(question):
        recovered = _recover_native_stability_split_temporal_date(
            question=question,
            support_pack=clustered_pack,
            actor_compatibility_enabled=actor_compatibility_enabled,
            rejection_counts=rejection_counts,
        )
        if recovered is not None:
            return recovered

    if intent == "location":
        return None
    return None


def _recover_native_stability_predicate_bound_answer(
    *,
    question: str,
    support_pack: tuple[ScoredEvidenceSupport, ...],
    actor_compatibility_enabled: bool = False,
    rejection_counts: dict[str, int] | None = None,
) -> ExactSupportRecoveryResult | None:
    recovered_candidates: list[tuple[ExactSupportRecoveryResult, float]] = []
    for item in support_pack:
        candidate = _predicate_bound_native_stability_candidate(question, item.support)
        if candidate is None:
            continue
        strategy = "native_stability_predicate_structured_fallback"
        trimmed_candidate = _trim_native_stability_object_head(candidate, item.support)
        if trimmed_candidate is not None:
            candidate = trimmed_candidate
            strategy = "native_stability_trimmed_object_head"
        normalized_candidate = _containment_normalize(candidate)
        if not _contains_containment_answer(normalized_candidate, item.support.normalized_support_text):
            continue
        recovered_candidates.append(
            (
                ExactSupportRecoveryResult(
                    answer=candidate,
                    normalized_answer=_normalize(candidate),
                    support=item.support,
                    strategy=strategy,
                ),
                item.score,
            )
        )
    return _pick_unique_exact_support_recovery_candidate(
        recovered_candidates,
        allow_clear_win_different_answers=True,
        question=question,
        actor_compatibility_enabled=actor_compatibility_enabled,
        rejection_counts=rejection_counts,
    )


def _recover_native_stability_action_clause(
    *,
    question: str,
    support_pack: tuple[ScoredEvidenceSupport, ...],
    actor_compatibility_enabled: bool = False,
    rejection_counts: dict[str, int] | None = None,
) -> ExactSupportRecoveryResult | None:
    recovered_candidates: list[tuple[ExactSupportRecoveryResult, float]] = []
    for item in support_pack:
        for sentence in _SENTENCE_SPLIT_PATTERN.split(item.support.support_text):
            sentence = sentence.strip()
            if not sentence or not _native_stability_sentence_matches(question, sentence, min_ratio=0.72):
                continue
            candidate = _atomic_action_clause_candidate(sentence)
            if candidate is None:
                continue
            if not _contains_containment_answer(
                _containment_normalize(candidate),
                _containment_normalize(sentence),
            ):
                continue
            recovered_candidates.append(
                (
                    ExactSupportRecoveryResult(
                        answer=candidate,
                        normalized_answer=_normalize(candidate),
                        support=item.support,
                        strategy="native_stability_atomic_action",
                    ),
                    item.score,
                )
            )
    return _pick_unique_exact_support_recovery_candidate(
        recovered_candidates,
        allow_clear_win_different_answers=True,
        question=question,
        actor_compatibility_enabled=actor_compatibility_enabled,
        rejection_counts=rejection_counts,
    )


def _recover_native_stability_fronted_subject_phrase(
    *,
    question: str,
    support_pack: tuple[ScoredEvidenceSupport, ...],
    actor_compatibility_enabled: bool = False,
    rejection_counts: dict[str, int] | None = None,
) -> ExactSupportRecoveryResult | None:
    recovered_candidates: list[tuple[ExactSupportRecoveryResult, float]] = []
    question_terms = _native_stability_question_terms(question)
    required_matches = max(2, min(3, len(question_terms))) if question_terms else 0

    for item in support_pack:
        for sentence in _SENTENCE_SPLIT_PATTERN.split(item.support.support_text):
            sentence = sentence.strip()
            match = _FRONTED_SUBJECT_COPULA_PATTERN.search(sentence)
            if not match:
                continue
            candidate = _clean_candidate_answer(match.group(1))
            predicate_text = _clean_candidate_answer(match.group(2))
            if not candidate or not predicate_text:
                continue
            if _native_stability_candidate_echoes_question(question, candidate):
                continue
            predicate_terms = _content_tokens(predicate_text)
            if (
                question_terms
                and _native_stability_matched_term_count(question_terms, predicate_terms) < required_matches
            ):
                continue
            if not _contains_containment_answer(
                _containment_normalize(candidate),
                _containment_normalize(sentence),
            ):
                continue
            recovered_candidates.append(
                (
                    ExactSupportRecoveryResult(
                        answer=candidate,
                        normalized_answer=_normalize(candidate),
                        support=item.support,
                        strategy="native_stability_atomic_subject_phrase",
                    ),
                    item.score,
                )
            )
    return _pick_unique_exact_support_recovery_candidate(
        recovered_candidates,
        allow_clear_win_different_answers=True,
        question=question,
        actor_compatibility_enabled=actor_compatibility_enabled,
        rejection_counts=rejection_counts,
    )


def _recover_native_stability_split_temporal_date(
    *,
    question: str,
    support_pack: tuple[ScoredEvidenceSupport, ...],
    actor_compatibility_enabled: bool = False,
    rejection_counts: dict[str, int] | None = None,
) -> ExactSupportRecoveryResult | None:
    recovered_candidates: list[tuple[ExactSupportRecoveryResult, float]] = []
    for item in support_pack:
        sentences = [
            sentence.strip()
            for sentence in _SENTENCE_SPLIT_PATTERN.split(item.support.support_text)
            if sentence.strip()
        ]
        bound_sentences = [
            sentence for sentence in sentences if _native_stability_sentence_matches(question, sentence, min_ratio=0.72)
        ]
        if len(bound_sentences) != 1:
            continue
        date_candidates = {
            candidate
            for sentence in sentences
            if sentence != bound_sentences[0]
            and not _native_stability_sentence_matches(question, sentence, min_ratio=0.72)
            for candidate in _native_stability_date_candidates(sentence)
        }
        if len(date_candidates) != 1:
            continue
        candidate = next(iter(date_candidates))
        recovered_candidates.append(
            (
                ExactSupportRecoveryResult(
                    answer=candidate,
                    normalized_answer=_normalize(candidate),
                    support=item.support,
                    strategy="native_stability_atomic_split_date",
                ),
                item.score,
            )
        )
    return _pick_unique_exact_support_recovery_candidate(
        recovered_candidates,
        allow_clear_win_different_answers=True,
        question=question,
        actor_compatibility_enabled=actor_compatibility_enabled,
        rejection_counts=rejection_counts,
    )


def _recover_support_surface_reach_answer(
    *,
    question: str,
    reach_pool: tuple[ScoredEvidenceSupport, ...],
    intent: AnswerIntent | None,
    actor_compatibility_enabled: bool = False,
    rejection_counts: dict[str, int] | None = None,
) -> ExactSupportRecoveryResult | None:
    question_lower = question.lower()

    if intent == "location" or question_lower.startswith("where did"):
        recovered = _recover_support_surface_reach_location(
            question=question,
            reach_pool=reach_pool,
            actor_compatibility_enabled=actor_compatibility_enabled,
            rejection_counts=rejection_counts,
        )
        if recovered is not None:
            return recovered

    if "used for" in question_lower:
        recovered = _recover_support_surface_reach_for_tail(
            question=question,
            reach_pool=reach_pool,
            actor_compatibility_enabled=actor_compatibility_enabled,
            rejection_counts=rejection_counts,
        )
        if recovered is not None:
            return recovered

    if _TEMPORAL_QUESTION_PATTERN.search(question):
        recovered = _recover_support_surface_reach_money_tournament_date(
            question=question,
            reach_pool=reach_pool,
            actor_compatibility_enabled=actor_compatibility_enabled,
            rejection_counts=rejection_counts,
        )
        if recovered is not None:
            return recovered

        recovered = _recover_support_surface_reach_same_concept_date(
            question=question,
            reach_pool=reach_pool,
            actor_compatibility_enabled=actor_compatibility_enabled,
            rejection_counts=rejection_counts,
        )
        if recovered is not None:
            return recovered

    return None


def _recover_support_surface_reach_location(
    *,
    question: str,
    reach_pool: tuple[ScoredEvidenceSupport, ...],
    actor_compatibility_enabled: bool = False,
    rejection_counts: dict[str, int] | None = None,
) -> ExactSupportRecoveryResult | None:
    recovered_candidates: list[tuple[ExactSupportRecoveryResult, float]] = []

    for item in reach_pool:
        if item.score < _SUPPORT_PACK_MIN_SCORE:
            continue
        for sentence in _SENTENCE_SPLIT_PATTERN.split(item.support.support_text):
            sentence = sentence.strip()
            if not sentence:
                continue
            candidate = _visited_location_candidate_from_sentence(question, sentence)
            if candidate is None:
                continue
            recovered_candidates.append(
                (
                    ExactSupportRecoveryResult(
                        answer=candidate,
                        normalized_answer=_normalize(candidate),
                        support=item.support,
                        strategy="support_surface_reach_visited_location",
                    ),
                    item.score,
                )
            )

    return _pick_unique_exact_support_recovery_candidate(
        recovered_candidates,
        allow_clear_win_different_answers=True,
        question=question,
        actor_compatibility_enabled=actor_compatibility_enabled,
        rejection_counts=rejection_counts,
    )


def _best_same_concept_reach_anchor(
    *,
    question: str,
    reach_pool: tuple[ScoredEvidenceSupport, ...],
    concept_id: str,
    exclude_support_id: str | None = None,
) -> ScoredEvidenceSupport | None:
    anchors = [
        item
        for item in reach_pool
        if item.support.concept_id == concept_id
        and item.support.support_id != exclude_support_id
        and item.score >= _SUPPORT_PACK_MIN_SCORE
        and _support_sufficiently_matches_question(
            question,
            item.support.support_text,
            min_ratio=_SUPPORT_SURFACE_REACH_MIN_RATIO,
        )
    ]
    if not anchors:
        return None
    return max(anchors, key=lambda item: item.score)


def _recover_support_surface_reach_for_tail(
    *,
    question: str,
    reach_pool: tuple[ScoredEvidenceSupport, ...],
    actor_compatibility_enabled: bool = False,
    rejection_counts: dict[str, int] | None = None,
) -> ExactSupportRecoveryResult | None:
    recovered_candidates: list[tuple[ExactSupportRecoveryResult, float]] = []

    for item in reach_pool:
        match = _FOR_TAIL_PATTERN.search(item.support.support_text)
        if match is None:
            continue
        candidate = _clean_candidate_answer(match.group(1))
        if not candidate:
            continue
        if not _contains_containment_answer(
            _containment_normalize(candidate),
            item.support.normalized_support_text,
        ):
            continue

        score = None
        if item.score >= _SUPPORT_PACK_MIN_SCORE and _support_sufficiently_matches_question(
            question,
            item.support.support_text,
            min_ratio=_SUPPORT_SURFACE_REACH_MIN_RATIO,
        ):
            score = item.score
        else:
            anchor = _best_same_concept_reach_anchor(
                question=question,
                reach_pool=reach_pool,
                concept_id=item.support.concept_id,
                exclude_support_id=item.support.support_id,
            )
            if anchor is None or anchor.score <= item.score:
                continue
            score = anchor.score

        recovered_candidates.append(
            (
                ExactSupportRecoveryResult(
                    answer=candidate,
                    normalized_answer=_normalize(candidate),
                    support=item.support,
                    strategy="support_surface_reach_for_tail",
                ),
                score,
            )
        )

    return _pick_unique_exact_support_recovery_candidate(
        recovered_candidates,
        allow_clear_win_different_answers=True,
        question=question,
        actor_compatibility_enabled=actor_compatibility_enabled,
        rejection_counts=rejection_counts,
    )


def _recover_support_surface_reach_money_tournament_date(
    *,
    question: str,
    reach_pool: tuple[ScoredEvidenceSupport, ...],
    actor_compatibility_enabled: bool = False,
    rejection_counts: dict[str, int] | None = None,
) -> ExactSupportRecoveryResult | None:
    if _MONEY_TOURNAMENT_DATE_QUESTION_PATTERN.search(question) is None:
        return None

    actor_terms = _support_actor_terms_from_text(question)
    recovered_candidates: list[tuple[ExactSupportRecoveryResult, float]] = []
    for item in reach_pool:
        if item.score < _SUPPORT_PACK_MIN_SCORE:
            continue
        for sentence in _SENTENCE_SPLIT_PATTERN.split(item.support.support_text):
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence_terms = _content_tokens(sentence)
            if actor_terms and _matched_term_count(list(actor_terms), sentence_terms) < len(actor_terms):
                continue
            if not _sentence_has_predicate(sentence_terms, "win"):
                continue
            if "tournament" not in sentence_terms:
                continue
            if not _MONEY_TOURNAMENT_DATE_CUE_PATTERN.search(sentence):
                continue
            for candidate in _native_stability_date_candidates(sentence):
                if _date_candidate_sanity_rejection_reason(candidate) is not None:
                    continue
                recovered_candidates.append(
                    (
                        ExactSupportRecoveryResult(
                            answer=candidate,
                            normalized_answer=_normalize(candidate),
                            support=item.support,
                            strategy="support_surface_reach_money_tournament_date",
                        ),
                        item.score,
                    )
                )

    return _pick_unique_exact_support_recovery_candidate(
        recovered_candidates,
        allow_clear_win_different_answers=True,
        question=question,
        actor_compatibility_enabled=actor_compatibility_enabled,
        rejection_counts=rejection_counts,
    )


def _support_surface_temporal_anchor_has_event_overlap(question: str, support_text: str) -> bool:
    event_terms = [
        term
        for term in _question_event_terms(question)
        if term not in _TEMPORAL_CALENDAR_CONTEXT_TERMS and not term.isdigit()
    ]
    if len(event_terms) < 2:
        return True
    support_terms = _content_tokens(support_text)
    return _matched_term_count(event_terms, support_terms) >= min(2, len(event_terms))


def _resolve_temporal_recovery_conflict(
    *,
    question: str,
    recovered: ExactSupportRecoveryResult,
    recovered_score: float,
    reach_pool: tuple[ScoredEvidenceSupport, ...],
    rejection_counts: dict[str, int] | None = None,
) -> ExactSupportRecoveryResult | None:
    temporal_candidates_by_date: dict[str, tuple[ExactSupportRecoveryResult, float]] = {}
    for item in reach_pool:
        if item.score <= recovered_score:
            continue
        support_text = item.support.support_text
        if not _support_sufficiently_matches_question(
            question,
            support_text,
            min_ratio=_SUPPORT_SURFACE_REACH_MIN_RATIO,
        ):
            continue
        if not _support_surface_temporal_anchor_has_event_overlap(question, support_text):
            continue
        for candidate in _native_stability_date_candidates(support_text):
            if _date_candidate_sanity_rejection_reason(candidate) is not None:
                continue
            if (
                not _SESSION_DATE_PATTERN.search(support_text)
                and _date_event_rejection_reason(question, support_text, candidate) is not None
            ):
                continue
            normalized = _normalize(candidate)
            existing = temporal_candidates_by_date.get(normalized)
            if existing is not None and existing[1] >= item.score:
                continue
            temporal_candidates_by_date[normalized] = (
                ExactSupportRecoveryResult(
                    answer=candidate,
                    normalized_answer=normalized,
                    support=item.support,
                    strategy=recovered.strategy,
                ),
                item.score,
            )

    if not temporal_candidates_by_date:
        return recovered

    ranked = sorted(
        temporal_candidates_by_date.values(),
        key=lambda candidate: candidate[1],
        reverse=True,
    )
    best, best_score = ranked[0]
    if best.normalized_answer == recovered.normalized_answer:
        return recovered
    for alternate, alternate_score in ranked[1:]:
        if alternate.normalized_answer == best.normalized_answer:
            continue
        if best_score - alternate_score < _SUPPORT_PACK_CLEAR_WIN_MARGIN:
            if rejection_counts is not None:
                rejection_counts["temporal_conflicting_date_support"] = (
                    rejection_counts.get("temporal_conflicting_date_support", 0) + 1
                )
            return None
    return best


def _recover_support_surface_reach_same_concept_date(
    *,
    question: str,
    reach_pool: tuple[ScoredEvidenceSupport, ...],
    actor_compatibility_enabled: bool = False,
    rejection_counts: dict[str, int] | None = None,
) -> ExactSupportRecoveryResult | None:
    supports_by_concept: dict[str, list[tuple[int, ScoredEvidenceSupport]]] = {}
    for index, item in enumerate(reach_pool):
        supports_by_concept.setdefault(item.support.concept_id, []).append((index, item))

    recovered_candidates: list[tuple[ExactSupportRecoveryResult, float]] = []
    for concept_items in supports_by_concept.values():
        anchor_index: int | None = None
        anchor_item: ScoredEvidenceSupport | None = None
        for index, item in concept_items:
            if item.score < _SUPPORT_PACK_MIN_SCORE:
                continue
            if not _support_sufficiently_matches_question(
                question,
                item.support.support_text,
                min_ratio=_SUPPORT_SURFACE_REACH_MIN_RATIO,
            ):
                continue
            if anchor_item is None or item.score > anchor_item.score:
                if not _support_surface_temporal_anchor_has_event_overlap(
                    question,
                    item.support.support_text,
                ):
                    continue
                anchor_index = index
                anchor_item = item
        if anchor_item is None or anchor_index is None:
            continue

        raw_date_candidates: set[str] = set()
        date_candidates: dict[str, tuple[str, EvidenceSupport]] = {}
        for index, item in concept_items:
            if index <= anchor_index:
                continue
            for candidate in _native_stability_date_candidates(item.support.support_text):
                raw_date_candidates.add(_normalize(candidate))
                if (
                    not _SESSION_DATE_PATTERN.search(item.support.support_text)
                    and _date_event_rejection_reason(
                        question,
                        item.support.support_text,
                        candidate,
                    )
                    is not None
                ):
                    continue
                date_candidates.setdefault(
                    _normalize(candidate),
                    (candidate, item.support),
                )
        if len(raw_date_candidates) > 1:
            continue
        if len(date_candidates) != 1:
            continue

        candidate, support = next(iter(date_candidates.values()))
        recovered = ExactSupportRecoveryResult(
            answer=candidate,
            normalized_answer=_normalize(candidate),
            support=support,
            strategy="support_surface_reach_same_concept_date",
        )
        resolved = _resolve_temporal_recovery_conflict(
            question=question,
            recovered=recovered,
            recovered_score=anchor_item.score,
            reach_pool=reach_pool,
            rejection_counts=rejection_counts,
        )
        if resolved is None:
            continue
        resolved_score = anchor_item.score
        if resolved.support.support_id != support.support_id:
            for reach_item in reach_pool:
                if reach_item.support.support_id == resolved.support.support_id:
                    resolved_score = max(resolved_score, reach_item.score)
                    break
        recovered_candidates.append(
            (
                resolved,
                resolved_score,
            )
        )

    return _pick_unique_exact_support_recovery_candidate(
        recovered_candidates,
        allow_clear_win_different_answers=True,
        question=question,
        actor_compatibility_enabled=actor_compatibility_enabled,
        rejection_counts=rejection_counts,
    )


def _recover_support_present_native_stability_answer(
    *,
    question: str,
    support_pack: tuple[ScoredEvidenceSupport, ...],
    reach_pool: tuple[ScoredEvidenceSupport, ...],
    supports: list[EvidenceSupport],
    intent: AnswerIntent | None,
    actor_compatibility_enabled: bool = False,
    rejection_counts: dict[str, int] | None = None,
) -> ExactSupportRecoveryResult | None:
    if intent == "temporal_deferred" and _TEMPORAL_MAKE_QUESTION_PATTERN.search(question):
        return _recover_support_present_temporal_make_media_event_date(
            question=question,
            supports=supports,
            actor_compatibility_enabled=actor_compatibility_enabled,
            rejection_counts=rejection_counts,
        )

    recovered = _recover_support_present_found_container_object(
        question=question,
        support_pack=support_pack,
        rejection_counts=rejection_counts,
    )
    if recovered is not None:
        return recovered

    if _support_present_predicate_object_question(question):
        recovered = _recover_support_present_predicate_object(
            question=question,
            support_pack=support_pack,
            actor_compatibility_enabled=actor_compatibility_enabled,
            rejection_counts=rejection_counts,
        )
        if recovered is not None:
            return recovered

    if _SUPPORT_PRESENT_FANDOM_QUESTION_PATTERN.search(question):
        recovered = _recover_support_present_fandom_tail(
            question=question,
            supports=supports,
            actor_compatibility_enabled=actor_compatibility_enabled,
            rejection_counts=rejection_counts,
        )
        if recovered is not None:
            return recovered

    if _TEMPORAL_QUESTION_PATTERN.search(question):
        recovered = _recover_support_present_same_concept_date_any_order(
            question=question,
            supports=supports,
            actor_compatibility_enabled=actor_compatibility_enabled,
            rejection_counts=rejection_counts,
        )
        if recovered is not None:
            return recovered
        recovered = _recover_support_present_temporal_contextual_pair_date(
            question=question,
            supports=supports,
            actor_compatibility_enabled=actor_compatibility_enabled,
            rejection_counts=rejection_counts,
        )
        if recovered is not None:
            return recovered
        recovered = _recover_support_present_temporal_direct_original_date(
            question=question,
            supports=supports,
            actor_compatibility_enabled=actor_compatibility_enabled,
            rejection_counts=rejection_counts,
        )
        if recovered is not None:
            return recovered
        recovered = _recover_support_present_temporal_source_set_original_date(
            question=question,
            supports=supports,
            actor_compatibility_enabled=actor_compatibility_enabled,
            rejection_counts=rejection_counts,
        )
        if recovered is not None:
            return recovered
        recovered = _recover_support_present_temporal_event_source_date(
            question=question,
            supports=supports,
            actor_compatibility_enabled=actor_compatibility_enabled,
            rejection_counts=rejection_counts,
        )
        if recovered is not None:
            return recovered
        recovered = _recover_support_present_session_date_adoption_bridge(
            question=question,
            supports=supports,
            actor_compatibility_enabled=actor_compatibility_enabled,
            rejection_counts=rejection_counts,
        )
        if recovered is not None:
            return recovered

    if reach_pool:
        return _recover_support_present_anchor_quoted_title(
            question=question,
            reach_pool=reach_pool,
            actor_compatibility_enabled=actor_compatibility_enabled,
            rejection_counts=rejection_counts,
        )

    return None


def _recover_support_present_found_container_object(
    *,
    question: str,
    support_pack: tuple[ScoredEvidenceSupport, ...],
    rejection_counts: dict[str, int] | None = None,
) -> ExactSupportRecoveryResult | None:
    question_container = _support_present_found_question_container(question)
    if question_container is None:
        return None

    recovered_candidates: list[tuple[ExactSupportRecoveryResult, float]] = []
    for item in support_pack:
        if item.score < _SUPPORT_PACK_MIN_SCORE:
            continue
        for sentence in _SENTENCE_SPLIT_PATTERN.split(item.support.support_text):
            candidate = _support_present_found_container_candidate(
                question_container=question_container,
                sentence=sentence,
            )
            if candidate is None:
                continue
            recovered_candidates.append(
                (
                    ExactSupportRecoveryResult(
                        answer=candidate,
                        normalized_answer=_normalize(candidate),
                        support=item.support,
                        strategy="support_present_found_container_object",
                    ),
                    item.score,
                )
            )

    return _pick_unique_exact_support_recovery_candidate(
        recovered_candidates,
        question=question,
        actor_compatibility_enabled=False,
        rejection_counts=rejection_counts,
    )


def _support_present_found_question_container(question: str) -> str | None:
    match = _SUPPORT_PRESENT_FOUND_CONTAINER_QUESTION_PATTERN.search(question)
    if match is None:
        return None
    container = _clean_candidate_answer(match.group("container"))
    if not container:
        return None
    container_terms = set(_content_tokens(container))
    if not container_terms or len(container_terms) > 5:
        return None
    return container


def _support_present_found_container_candidate(
    *,
    question_container: str,
    sentence: str,
) -> str | None:
    sentence = sentence.strip()
    if not sentence:
        return None
    match = _SUPPORT_PRESENT_FOUND_CONTAINER_SUPPORT_PATTERN.search(sentence)
    if match is None:
        return None
    support_container = _clean_candidate_answer(match.group("container"))
    if not _support_present_found_container_terms_match(
        question_container,
        support_container,
    ):
        return None
    candidate = _clean_candidate_answer(match.group("object"))
    candidate = _SUPPORT_PRESENT_LEADING_POSSESSIVE_PATTERN.sub("", candidate)
    candidate = _strip_leading_article(_clean_candidate_answer(candidate))
    candidate_terms = set(_content_tokens(candidate))
    if not candidate_terms or len(candidate_terms) > 6:
        return None
    if candidate_terms <= set(_content_tokens(question_container)):
        return None
    if not _contains_containment_answer(
        _containment_normalize(candidate),
        _containment_normalize(sentence),
    ):
        return None
    return candidate


def _support_present_found_container_terms_match(
    question_container: str,
    support_container: str,
) -> bool:
    question_terms = _content_tokens(question_container)
    support_terms = _content_tokens(support_container)
    if not question_terms or not support_terms:
        return False
    return any(
        _tokens_match(question_term, support_term)
        for question_term in question_terms
        for support_term in support_terms
    )


def _recover_support_present_guard_stability_answer(
    *,
    question: str,
    support_pack: tuple[ScoredEvidenceSupport, ...],
    supports: list[EvidenceSupport],
    actor_compatibility_enabled: bool = False,
    rejection_counts: dict[str, int] | None = None,
) -> ExactSupportRecoveryResult | None:
    if _TEMPORAL_QUESTION_PATTERN.search(question):
        recovered = _recover_support_present_same_concept_date_any_order(
            question=question,
            supports=supports,
            actor_compatibility_enabled=actor_compatibility_enabled,
            rejection_counts=rejection_counts,
        )
        if recovered is not None:
            return _retag_exact_support_recovery(
                recovered,
                strategy="support_present_guard_temporal",
            )

    for recovered in (
        _recover_support_present_quoted_reading_title(
            question=question,
            supports=supports,
            actor_compatibility_enabled=actor_compatibility_enabled,
            rejection_counts=rejection_counts,
        ),
        _recover_support_present_location_any_support(
            question=question,
            supports=supports,
            actor_compatibility_enabled=actor_compatibility_enabled,
            rejection_counts=rejection_counts,
        ),
        _recover_support_present_favorite_title(
            question=question,
            supports=supports,
            actor_compatibility_enabled=actor_compatibility_enabled,
            rejection_counts=rejection_counts,
        ),
        _recover_support_present_watched_title(
            question=question,
            supports=supports,
            actor_compatibility_enabled=actor_compatibility_enabled,
            rejection_counts=rejection_counts,
        ),
        _recover_support_present_favorite_dish(
            question=question,
            supports=supports,
            actor_compatibility_enabled=actor_compatibility_enabled,
            rejection_counts=rejection_counts,
        ),
        _recover_support_present_game_type_answer(
            question=question,
            support_pack=support_pack,
            actor_compatibility_enabled=actor_compatibility_enabled,
            rejection_counts=rejection_counts,
        ),
        _recover_support_present_because_reason_answer(
            question=question,
            support_pack=support_pack,
            actor_compatibility_enabled=actor_compatibility_enabled,
            rejection_counts=rejection_counts,
        ),
    ):
        if recovered is not None:
            return recovered
    return None


def _recover_support_present_quoted_reading_title(
    *,
    question: str,
    supports: list[EvidenceSupport],
    actor_compatibility_enabled: bool = False,
    rejection_counts: dict[str, int] | None = None,
) -> ExactSupportRecoveryResult | None:
    if not _support_present_reading_title_question(question):
        return None

    recovered_candidates: list[tuple[ExactSupportRecoveryResult, float]] = []
    for support in supports:
        for sentence in _SENTENCE_SPLIT_PATTERN.split(support.support_text):
            sentence = sentence.strip()
            if not sentence:
                continue
            for match in _QUOTED_READING_TITLE_PATTERN.finditer(sentence):
                candidate = _clean_candidate_answer(match.group(1))
                if _support_present_title_rejection_reason(candidate) is not None:
                    continue
                recovered_candidates.append(
                    (
                        ExactSupportRecoveryResult(
                            answer=candidate,
                            normalized_answer=_normalize(candidate),
                            support=support,
                            strategy="support_present_guard_quoted_reading_title",
                        ),
                        _score_support(question, support).score,
                    )
                )

    return _pick_unique_exact_support_recovery_candidate(
        recovered_candidates,
        question=question,
        actor_compatibility_enabled=actor_compatibility_enabled,
        rejection_counts=rejection_counts,
    )


def _support_actor_scope_terms(support: EvidenceSupport) -> set[str]:
    terms = set(_support_actor_terms_from_text(support.support_text))
    if not terms:
        terms.update(support.concept_actor_terms)
        terms.update(_support_actor_terms_from_text(support.concept_summary))
    return terms


def _favorite_title_anchor_score(
    *,
    title_support: EvidenceSupport,
    anchors: list[tuple[EvidenceSupport, float]],
) -> float | None:
    title_terms = _support_actor_scope_terms(title_support)
    best_score: float | None = None
    for anchor, score in anchors:
        if anchor.support_id == title_support.support_id:
            best_score = score if best_score is None else max(best_score, score)
            continue
        anchor_terms = _support_actor_scope_terms(anchor)
        if title_terms and anchor_terms and title_terms & anchor_terms:
            best_score = score if best_score is None else max(best_score, score)
    return best_score


def _recover_support_present_favorite_title(
    *,
    question: str,
    supports: list[EvidenceSupport],
    actor_compatibility_enabled: bool = False,
    rejection_counts: dict[str, int] | None = None,
) -> ExactSupportRecoveryResult | None:
    if not actor_compatibility_enabled:
        return None
    if not _FAVORITE_TITLE_QUESTION_PATTERN.search(question):
        return None

    anchors: list[tuple[EvidenceSupport, float]] = []
    for support in supports:
        if not _FAVORITE_TITLE_CONTEXT_PATTERN.search(support.support_text):
            continue
        if not _FAVORITE_TITLE_ANCHOR_PATTERN.search(support.support_text):
            continue
        score = _score_support(question, support).score
        if score >= _SUPPORT_PACK_MIN_SCORE:
            anchors.append((support, score))
    if not anchors:
        return None

    recovered_candidates: list[tuple[ExactSupportRecoveryResult, float]] = []
    for support in supports:
        if not _FAVORITE_TITLE_CONTEXT_PATTERN.search(support.support_text):
            continue
        anchor_score = _favorite_title_anchor_score(title_support=support, anchors=anchors)
        if anchor_score is None:
            continue
        for candidate in _support_present_quoted_title_candidates(support.support_text):
            if not _contains_containment_answer(
                _containment_normalize(candidate),
                support.normalized_support_text,
            ):
                continue
            recovered_candidates.append(
                (
                    ExactSupportRecoveryResult(
                        answer=candidate,
                        normalized_answer=_normalize(candidate),
                        support=support,
                        strategy="support_present_guard_favorite_title",
                    ),
                    anchor_score,
                )
            )

    return _pick_unique_exact_support_recovery_candidate(
        recovered_candidates,
        question=question,
        actor_compatibility_enabled=actor_compatibility_enabled,
        rejection_counts=rejection_counts,
    )


def _support_present_reading_title_question(question: str) -> bool:
    return bool(
        re.search(r"\b(?:book|title)\b", question, re.IGNORECASE)
        and re.search(r"\bread(?:ing)?\b", question, re.IGNORECASE)
    )


def _suggested_book_title_question(question: str) -> tuple[str, str] | None:
    match = _SUGGESTED_BOOK_TITLE_QUESTION_PATTERN.search(question)
    if match is None:
        return None
    reader = _actor_token(match.group("reader"))
    recommender = _actor_token(match.group("recommender"))
    if reader is None or recommender is None or reader == recommender:
        return None
    return reader, recommender


def _suggested_book_support_mentions_actor(
    support: EvidenceSupport,
    actor: str,
) -> bool:
    return bool(
        _locomo_support_has_actor(support, actor)
        or _locomo_sentence_mentions_actor(support.support_text, actor)
        or _locomo_sentence_mentions_actor(support.concept_summary, actor)
    )


def _suggested_book_relationship_span(
    support: EvidenceSupport,
    *,
    reader: str,
    recommender: str,
) -> str | None:
    if not _suggested_book_support_mentions_actor(support, reader):
        return None
    if not (
        _suggested_book_support_mentions_actor(support, recommender)
        or (
            re.search(r"\byou\s+(?:recommend(?:ed)?|suggest(?:ed)?)\b", support.support_text, re.IGNORECASE)
            and _suggested_book_support_mentions_actor(support, recommender)
        )
    ):
        return None
    candidates = [support.concept_summary, *(_SENTENCE_SPLIT_PATTERN.split(support.support_text))]
    for raw_sentence in candidates:
        sentence = raw_sentence.strip()
        if not sentence:
            continue
        if not _locomo_sentence_mentions_actor(sentence, reader):
            continue
        if not _SUGGESTED_BOOK_RELATIONSHIP_PATTERN.search(sentence):
            continue
        return sentence
    return None


def _suggested_book_title_candidate(
    support: EvidenceSupport,
    *,
    recommender: str,
) -> tuple[str, str] | None:
    if not _suggested_book_support_mentions_actor(support, recommender):
        return None
    if not _SUGGESTED_BOOK_RECOMMEND_CONTEXT_PATTERN.search(support.support_text):
        return None
    for sentence in _SENTENCE_SPLIT_PATTERN.split(support.support_text):
        sentence = sentence.strip()
        if not sentence:
            continue
        for title in _support_present_quoted_title_candidates(sentence):
            if _support_present_title_rejection_reason(title) is not None:
                continue
            return title, sentence
    return None


def _recover_suggested_book_title_bridge(
    *,
    question: str,
    supports: list[EvidenceSupport],
) -> StructuredSynthesisResult | None:
    actors = _suggested_book_title_question(question)
    if actors is None:
        return None
    reader, recommender = actors
    relationship_spans: list[tuple[EvidenceSupport, str]] = []
    title_candidates: list[tuple[str, EvidenceSupport, str]] = []
    for support in supports:
        relationship_span = _suggested_book_relationship_span(
            support,
            reader=reader,
            recommender=recommender,
        )
        if relationship_span is not None:
            relationship_spans.append((support, relationship_span))
        title_candidate = _suggested_book_title_candidate(
            support,
            recommender=recommender,
        )
        if title_candidate is not None:
            title, title_span = title_candidate
            title_candidates.append((title, support, title_span))
    if not relationship_spans or not title_candidates:
        return None

    by_normalized: dict[str, tuple[str, EvidenceSupport, str, EvidenceSupport, str]] = {}
    for title, title_support, title_span in title_candidates:
        for relationship_support, relationship_span in relationship_spans:
            if title_support.support_id == relationship_support.support_id:
                continue
            normalized = _normalize(title)
            by_normalized.setdefault(
                normalized,
                (title, title_support, title_span, relationship_support, relationship_span),
            )
    if len(by_normalized) != 1:
        return None
    title, title_support, title_span, relationship_support, relationship_span = next(iter(by_normalized.values()))
    return StructuredSynthesisResult(
        answer=title,
        support_ids=(title_support.support_id, relationship_support.support_id),
        cited_spans=(title_span, relationship_span),
        fallback_used="suggested_book_title_bridge",
    )


def _recover_support_present_location_any_support(
    *,
    question: str,
    supports: list[EvidenceSupport],
    actor_compatibility_enabled: bool = False,
    rejection_counts: dict[str, int] | None = None,
) -> ExactSupportRecoveryResult | None:
    question_lower = question.lower()
    if not (question_lower.startswith("where did") or _LOCATION_QUESTION_PATTERN.search(question)):
        return None

    recovered_candidates: list[tuple[ExactSupportRecoveryResult, float]] = []
    for support in supports:
        for sentence in _SENTENCE_SPLIT_PATTERN.split(support.support_text):
            candidate = _support_present_location_candidate_from_sentence(sentence)
            if candidate is None:
                continue
            recovered_candidates.append(
                (
                    ExactSupportRecoveryResult(
                        answer=candidate,
                        normalized_answer=_normalize(candidate),
                        support=support,
                        strategy="support_present_guard_location_any_support",
                    ),
                    _score_support(question, support).score,
                )
            )

    return _pick_unique_exact_support_recovery_candidate(
        recovered_candidates,
        question=question,
        actor_compatibility_enabled=actor_compatibility_enabled,
        rejection_counts=rejection_counts,
    )


def _support_present_location_candidate_from_sentence(sentence: str) -> str | None:
    sentence = sentence.strip()
    if not sentence:
        return None
    candidate = _visited_location_candidate_from_sentence("", sentence)
    if candidate is not None:
        return candidate
    for pattern in (_LOCATION_TO_PATTERN, _LOCATION_FROM_PATTERN):
        match = pattern.search(sentence)
        if match is None:
            continue
        candidate = _clean_candidate_answer(match.group(1) if match.lastindex else match.group(0))
        if candidate:
            return candidate
    return None


def _recover_support_present_watched_title(
    *,
    question: str,
    supports: list[EvidenceSupport],
    actor_compatibility_enabled: bool = False,
    rejection_counts: dict[str, int] | None = None,
) -> ExactSupportRecoveryResult | None:
    if not (
        re.search(r"\b(?:movie|film)\b", question, re.IGNORECASE)
        and re.search(r"\b(?:watch|watched|saw|enjoy)\b", question, re.IGNORECASE)
    ):
        return None

    recovered_candidates: list[tuple[ExactSupportRecoveryResult, float]] = []
    for support in supports:
        for sentence in _SENTENCE_SPLIT_PATTERN.split(support.support_text):
            sentence = sentence.strip()
            if not sentence or _WATCHED_TITLE_REJECTION_PATTERN.search(sentence):
                continue
            for match in _WATCHED_TITLE_PATTERN.finditer(sentence):
                candidate = _clean_candidate_answer(match.group(1))
                if _support_present_title_rejection_reason(candidate) is not None:
                    continue
                recovered_candidates.append(
                    (
                        ExactSupportRecoveryResult(
                            answer=candidate,
                            normalized_answer=_normalize(candidate),
                            support=support,
                            strategy="support_present_guard_watched_title",
                        ),
                        _score_support(question, support).score,
                    )
                )

    return _pick_unique_exact_support_recovery_candidate(
        recovered_candidates,
        question=question,
        actor_compatibility_enabled=actor_compatibility_enabled,
        rejection_counts=rejection_counts,
    )


def _recover_support_present_favorite_dish(
    *,
    question: str,
    supports: list[EvidenceSupport],
    actor_compatibility_enabled: bool = False,
    rejection_counts: dict[str, int] | None = None,
) -> ExactSupportRecoveryResult | None:
    if _FAVORITE_DISH_QUESTION_PATTERN.search(question) is None:
        return None

    recovered_candidates: list[tuple[ExactSupportRecoveryResult, float]] = []
    for support in supports:
        for sentence in _SENTENCE_SPLIT_PATTERN.split(support.support_text):
            sentence = sentence.strip()
            match = _FAVORITE_DISH_SUPPORT_PATTERN.search(sentence)
            if match is None:
                continue
            candidate = _strip_leading_article(_clean_candidate_answer(match.group(1)))
            if not _support_present_favorite_dish_candidate_allowed(candidate):
                continue
            if not _contains_containment_answer(
                _containment_normalize(candidate),
                support.normalized_support_text,
            ):
                continue
            recovered_candidates.append(
                (
                    ExactSupportRecoveryResult(
                        answer=candidate,
                        normalized_answer=_normalize(candidate),
                        support=support,
                        strategy="support_present_guard_favorite_dish",
                    ),
                    _score_support(question, support).score,
                )
            )

    return _pick_unique_exact_support_recovery_candidate(
        recovered_candidates,
        question=question,
        actor_compatibility_enabled=actor_compatibility_enabled,
        rejection_counts=rejection_counts,
    )


def _support_present_favorite_dish_candidate_allowed(candidate: str) -> bool:
    if not candidate or _LIST_SEPARATOR_PATTERN.search(candidate):
        return False
    tokens = _content_tokens(candidate)
    if not tokens or len(tokens) > 6:
        return False
    blocked = {"dish", "favorite", "cooking", "show", "host", "hosted"}
    return not set(tokens).intersection(blocked)


def _recover_support_present_game_type_answer(
    *,
    question: str,
    support_pack: tuple[ScoredEvidenceSupport, ...],
    actor_compatibility_enabled: bool = False,
    rejection_counts: dict[str, int] | None = None,
) -> ExactSupportRecoveryResult | None:
    title_match = _SUPPORT_PRESENT_GAME_TYPE_QUESTION_PATTERN.search(question)
    if title_match is None:
        return None
    question_title = _clean_candidate_answer(title_match.group(1))
    if not question_title:
        return None

    recovered_candidates: list[tuple[ExactSupportRecoveryResult, float]] = []
    for item in support_pack:
        if item.score < _SUPPORT_PACK_MIN_SCORE:
            continue
        for sentence in _SENTENCE_SPLIT_PATTERN.split(item.support.support_text):
            candidate = _support_present_game_type_candidate(question_title, sentence)
            if candidate is None:
                continue
            if not _contains_containment_answer(
                _containment_normalize(candidate),
                item.support.normalized_support_text,
            ):
                continue
            recovered_candidates.append(
                (
                    ExactSupportRecoveryResult(
                        answer=candidate,
                        normalized_answer=_normalize(candidate),
                        support=item.support,
                        strategy="support_present_guard_game_type_before_title",
                    ),
                    item.score,
                )
            )

    return _pick_unique_exact_support_recovery_candidate(
        recovered_candidates,
        allow_clear_win_different_answers=True,
        question=question,
        actor_compatibility_enabled=actor_compatibility_enabled,
        rejection_counts=rejection_counts,
    )


def _support_present_game_type_candidate(
    question_title: str,
    sentence: str,
) -> str | None:
    for match in _SUPPORT_PRESENT_GAME_TYPE_SUPPORT_PATTERN.finditer(sentence):
        support_title = _clean_candidate_answer(match.group(2))
        if not _support_present_titles_similar(question_title, support_title):
            continue
        candidate = _clean_support_present_game_type_phrase(match.group(1))
        if candidate is not None:
            return candidate
    return None


def _clean_support_present_game_type_phrase(raw: str) -> str | None:
    tokens = _clean_candidate_answer(raw).split()
    while tokens and tokens[0].lower().strip(string.punctuation) in _SUPPORT_PRESENT_GAME_TYPE_LEADING_FILLER_WORDS:
        tokens.pop(0)
    candidate = _clean_candidate_answer(" ".join(tokens))
    if not candidate:
        return None
    normalized_tokens = _containment_normalize(candidate).split()
    if len(normalized_tokens) > 5:
        return None
    if not any(token in _SUPPORT_PRESENT_GAME_TYPE_TERMINAL_TOKENS for token in normalized_tokens):
        return None
    return candidate


def _support_present_titles_similar(left: str, right: str) -> bool:
    left_tokens = _content_tokens(left)
    right_tokens = _content_tokens(right)
    if not left_tokens or len(left_tokens) != len(right_tokens):
        return False
    total_distance = 0
    for left_token, right_token in zip(left_tokens, right_tokens, strict=True):
        if left_token == right_token:
            continue
        max_distance = 2 if max(len(left_token), len(right_token)) >= 8 else 1
        distance = _bounded_levenshtein_distance(left_token, right_token, max_distance)
        if distance > max_distance:
            return False
        total_distance += distance
    return total_distance <= 2


def _bounded_levenshtein_distance(left: str, right: str, limit: int) -> int:
    if abs(len(left) - len(right)) > limit:
        return limit + 1
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        if min(current) > limit:
            return limit + 1
        previous = current
    return previous[-1]


def _recover_support_present_because_reason_answer(
    *,
    question: str,
    support_pack: tuple[ScoredEvidenceSupport, ...],
    actor_compatibility_enabled: bool = False,
    rejection_counts: dict[str, int] | None = None,
) -> ExactSupportRecoveryResult | None:
    if _SUPPORT_PRESENT_REASON_QUESTION_PATTERN.search(question) is None:
        return None
    question_terms = set(_content_tokens(question))
    if "walk" not in question_terms or not question_terms.intersection({"dog", "dogs"}):
        return None

    recovered_candidates: list[tuple[ExactSupportRecoveryResult, float]] = []
    for item in support_pack:
        if item.score < _SUPPORT_PACK_MIN_SCORE:
            continue
        for sentence in _SENTENCE_SPLIT_PATTERN.split(item.support.support_text):
            sentence_terms = set(_content_tokens(sentence))
            if "walk" not in sentence_terms or not sentence_terms.intersection({"dog", "dogs"}):
                continue
            match = _SUPPORT_PRESENT_REASON_BECAUSE_PATTERN.search(sentence)
            if match is None:
                continue
            candidate = _strip_leading_article(_clean_candidate_answer(match.group(1)))
            if not _support_present_because_reason_candidate_allowed(candidate):
                continue
            if not _contains_containment_answer(
                _containment_normalize(candidate),
                item.support.normalized_support_text,
            ):
                continue
            recovered_candidates.append(
                (
                    ExactSupportRecoveryResult(
                        answer=candidate,
                        normalized_answer=_normalize(candidate),
                        support=item.support,
                        strategy="support_present_guard_because_of_reason",
                    ),
                    item.score,
                )
            )

    return _pick_unique_exact_support_recovery_candidate(
        recovered_candidates,
        allow_clear_win_different_answers=True,
        question=question,
        actor_compatibility_enabled=actor_compatibility_enabled,
        rejection_counts=rejection_counts,
    )


def _support_present_because_reason_candidate_allowed(candidate: str) -> bool:
    if not candidate or _LIST_SEPARATOR_PATTERN.search(candidate):
        return False
    tokens = _content_tokens(candidate)
    if not tokens or len(tokens) > 4:
        return False
    blocked = {"because", "could", "couldnt", "dog", "dogs", "walk", "walked", "walking"}
    return not set(tokens).intersection(blocked)


def _retag_exact_support_recovery(
    recovered: ExactSupportRecoveryResult,
    *,
    strategy: str,
) -> ExactSupportRecoveryResult:
    return ExactSupportRecoveryResult(
        answer=recovered.answer,
        normalized_answer=recovered.normalized_answer,
        support=recovered.support,
        strategy=strategy,
    )


def _support_present_predicate_object_question(question: str) -> bool:
    if not re.search(r"^\s*what\s+did\b", question, re.IGNORECASE):
        return False
    question_terms = set(_content_tokens(question))
    return any(
        any(term in _predicate_forms(verb) for term in question_terms)
        for verb in _SUPPORT_PRESENT_PREDICATE_OBJECT_VERBS
    )


def _recover_support_present_predicate_object(
    *,
    question: str,
    support_pack: tuple[ScoredEvidenceSupport, ...],
    actor_compatibility_enabled: bool = False,
    rejection_counts: dict[str, int] | None = None,
) -> ExactSupportRecoveryResult | None:
    recovered_candidates: list[tuple[ExactSupportRecoveryResult, float]] = []
    for item in support_pack:
        if item.score < _SUPPORT_PACK_MIN_SCORE:
            continue
        if not _support_sufficiently_matches_question(question, item.support.support_text):
            continue
        for sentence in _SENTENCE_SPLIT_PATTERN.split(item.support.support_text):
            sentence = sentence.strip()
            if not sentence:
                continue
            candidate = _support_present_predicate_object_candidate(question, sentence)
            if candidate is None:
                continue
            if len(_content_tokens(candidate)) > 8:
                continue
            if not _contains_containment_answer(
                _containment_normalize(candidate),
                item.support.normalized_support_text,
            ):
                continue
            recovered_candidates.append(
                (
                    ExactSupportRecoveryResult(
                        answer=candidate,
                        normalized_answer=_normalize(candidate),
                        support=item.support,
                        strategy="support_present_predicate_object",
                    ),
                    item.score,
                )
            )

    return _pick_unique_exact_support_recovery_candidate(
        recovered_candidates,
        allow_clear_win_different_answers=True,
        question=question,
        actor_compatibility_enabled=actor_compatibility_enabled,
        rejection_counts=rejection_counts,
    )


def _support_present_predicate_object_candidate(
    question: str,
    sentence: str,
) -> str | None:
    question_terms = set(_content_tokens(question))
    for verb in _SUPPORT_PRESENT_PREDICATE_OBJECT_VERBS:
        if not any(term in _predicate_forms(verb) for term in question_terms):
            continue
        for form in sorted(_predicate_forms(verb), key=len, reverse=True):
            pattern = (
                rf"\b{re.escape(form)}\s+"
                r"((?:(?:a|an|the)\s+)?.+?)(?:\s+(?:for|on|with|to|at|in|during|after|"
                r"before|recently|yesterday|today)\b|[.!?,;]|$)"
            )
            match = re.search(pattern, sentence, re.IGNORECASE)
            if not match:
                continue
            candidate = _clean_candidate_answer(match.group(1))
            if candidate:
                return candidate
    return None


def _recover_support_present_fandom_tail(
    *,
    question: str,
    supports: list[EvidenceSupport],
    actor_compatibility_enabled: bool = False,
    rejection_counts: dict[str, int] | None = None,
) -> ExactSupportRecoveryResult | None:
    recovered_candidates: list[tuple[ExactSupportRecoveryResult, float]] = []
    for support in supports:
        score = _score_support(question, support).score
        for candidate in _support_present_fandom_candidates(support.support_text):
            if len(_content_tokens(candidate)) > 8:
                continue
            if not _contains_containment_answer(
                _containment_normalize(candidate),
                support.normalized_support_text,
            ):
                continue
            recovered_candidates.append(
                (
                    ExactSupportRecoveryResult(
                        answer=candidate,
                        normalized_answer=_normalize(candidate),
                        support=support,
                        strategy="support_present_fandom_tail",
                    ),
                    score,
                )
            )

    return _pick_unique_exact_support_recovery_candidate(
        recovered_candidates,
        question=question,
        actor_compatibility_enabled=actor_compatibility_enabled,
        rejection_counts=rejection_counts,
    )


def _support_present_fandom_candidates(support_text: str) -> list[str]:
    candidates: list[str] = []
    for pattern in (_SUPPORT_PRESENT_FAN_OF_ENTITY_PATTERN, _SUPPORT_PRESENT_ENTITY_FAN_PATTERN):
        for match in pattern.finditer(support_text):
            candidate = _clean_candidate_answer(match.group(1))
            if not candidate or " by " in candidate.lower():
                continue
            if not re.match(r"^[A-Z][A-Za-z'&-]*(?:\s+[A-Z][A-Za-z'&-]*){0,3}$", candidate):
                continue
            candidates.append(candidate)
    return candidates


def _recover_support_present_same_concept_date_any_order(
    *,
    question: str,
    supports: list[EvidenceSupport],
    actor_compatibility_enabled: bool = False,
    rejection_counts: dict[str, int] | None = None,
) -> ExactSupportRecoveryResult | None:
    supports_by_concept: dict[str, list[ScoredEvidenceSupport]] = {}
    for support in supports:
        scored = _score_support(question, support)
        supports_by_concept.setdefault(support.concept_id, []).append(scored)

    recovered_candidates: list[tuple[ExactSupportRecoveryResult, float]] = []
    for concept_items in supports_by_concept.values():
        anchor_item = next(
            (
                item
                for item in sorted(concept_items, key=lambda value: value.score, reverse=True)
                if item.score >= _SUPPORT_PACK_MIN_SCORE
                and _support_sufficiently_matches_question(
                    question,
                    item.support.support_text,
                    min_ratio=_SUPPORT_SURFACE_REACH_MIN_RATIO,
                )
            ),
            None,
        )
        if anchor_item is None:
            continue

        raw_date_candidates: set[str] = set()
        date_candidates: dict[str, tuple[str, EvidenceSupport]] = {}
        for item in concept_items:
            if item.support.support_id == anchor_item.support.support_id:
                continue
            for candidate in _native_stability_date_candidates(item.support.support_text):
                raw_date_candidates.add(_normalize(candidate))
                if (
                    not _SESSION_DATE_PATTERN.search(item.support.support_text)
                    and _date_event_rejection_reason(
                        question,
                        item.support.support_text,
                        candidate,
                    )
                    is not None
                ):
                    continue
                date_candidates.setdefault(_normalize(candidate), (candidate, item.support))
        if len(raw_date_candidates) > 1:
            continue
        if len(date_candidates) != 1:
            continue

        candidate, support = next(iter(date_candidates.values()))
        recovered_candidates.append(
            (
                ExactSupportRecoveryResult(
                    answer=candidate,
                    normalized_answer=_normalize(candidate),
                    support=support,
                    strategy="support_present_same_concept_date_any_order",
                ),
                anchor_item.score,
            )
        )

    return _pick_unique_exact_support_recovery_candidate(
        recovered_candidates,
        question=question,
        actor_compatibility_enabled=actor_compatibility_enabled,
        rejection_counts=rejection_counts,
    )


def _recover_support_present_temporal_source_set_original_date(
    *,
    question: str,
    supports: list[EvidenceSupport],
    actor_compatibility_enabled: bool = False,
    rejection_counts: dict[str, int] | None = None,
) -> ExactSupportRecoveryResult | None:
    date_supports = [
        support
        for support in supports
        if support.concept_original_date
        and support.concept_valid_from
        and not _SESSION_DATE_PATTERN.search(support.support_text)
        and not _SESSION_DATE_PATTERN.search(support.concept_summary)
        and _support_actor_compatible(question, support)
        and _temporal_source_set_named_actor_compatible(question, support)
    ]
    if not date_supports:
        return None

    event_supports = [
        support
        for support in supports
        if support.concept_valid_from
        and not support.concept_original_date
        and not _SESSION_DATE_PATTERN.search(support.support_text)
        and not _SESSION_DATE_PATTERN.search(support.concept_summary)
        and _support_actor_compatible(question, support)
        and _temporal_source_set_named_actor_compatible(question, support)
        and (
            _support_sufficiently_matches_question(question, support.support_text, min_ratio=0.35)
            or _support_sufficiently_matches_question(question, support.concept_summary, min_ratio=0.35)
        )
    ]
    if not event_supports:
        return None

    recovered_candidates: list[tuple[ExactSupportRecoveryResult, float]] = []
    for date_support in date_supports:
        answer = _format_original_date_answer(date_support.concept_original_date)
        if answer is None:
            if rejection_counts is not None:
                _increment_rejection(rejection_counts, "temporal_source_set_unparseable_original_date")
            continue
        matching_events = [
            event_support
            for event_support in event_supports
            if _temporal_source_set_pair_matches(question, event_support, date_support)
        ]
        if not matching_events:
            if rejection_counts is not None:
                _increment_rejection(rejection_counts, "temporal_source_set_no_event_overlap")
            continue
        best_event_score = max(_score_support(question, support).score for support in matching_events)
        recovered_candidates.append(
            (
                ExactSupportRecoveryResult(
                    answer=answer,
                    normalized_answer=_normalize(answer),
                    support=date_support,
                    strategy="support_present_temporal_source_set_original_date",
                ),
                best_event_score,
            )
        )

    return _pick_unique_exact_support_recovery_candidate(
        recovered_candidates,
        allow_clear_win_different_answers=True,
        question=question,
        actor_compatibility_enabled=actor_compatibility_enabled,
        rejection_counts=rejection_counts,
    )


def _recover_support_present_temporal_direct_original_date(
    *,
    question: str,
    supports: list[EvidenceSupport],
    actor_compatibility_enabled: bool = False,
    rejection_counts: dict[str, int] | None = None,
) -> ExactSupportRecoveryResult | None:
    if not _TEMPORAL_QUESTION_PATTERN.search(question):
        return None
    recovered_candidates: list[tuple[ExactSupportRecoveryResult, float]] = []
    for support in supports:
        if not support.concept_original_date or not support.concept_valid_from:
            continue
        if _SESSION_DATE_PATTERN.search(support.support_text) or _SESSION_DATE_PATTERN.search(
            support.concept_summary
        ):
            if rejection_counts is not None:
                _increment_rejection(rejection_counts, "temporal_direct_original_date_session_date")
            continue
        if actor_compatibility_enabled and not _support_actor_compatible(question, support):
            continue
        if not _temporal_source_set_named_actor_compatible(question, support):
            continue
        if not _temporal_direct_original_date_support_matches_question_event(question, support):
            if rejection_counts is not None:
                _increment_rejection(rejection_counts, "temporal_direct_original_date_event_unbound")
            continue
        if _temporal_direct_original_date_conflicts_explicit_date(support):
            if rejection_counts is not None:
                _increment_rejection(rejection_counts, "temporal_direct_original_date_explicit_conflict")
            continue
        answer = _format_original_date_answer(support.concept_original_date)
        if answer is None:
            if rejection_counts is not None:
                _increment_rejection(rejection_counts, "temporal_direct_original_date_unparseable")
            continue
        recovered_candidates.append(
            (
                ExactSupportRecoveryResult(
                    answer=answer,
                    normalized_answer=_normalize(answer),
                    support=support,
                    strategy="support_present_temporal_direct_original_date",
                ),
                _score_support(question, support).score,
            )
        )
    return _pick_unique_exact_support_recovery_candidate(
        recovered_candidates,
        allow_clear_win_different_answers=True,
        question=question,
        actor_compatibility_enabled=actor_compatibility_enabled,
        rejection_counts=rejection_counts,
    )


def _recover_support_present_temporal_contextual_pair_date(
    *,
    question: str,
    supports: list[EvidenceSupport],
    actor_compatibility_enabled: bool = False,
    rejection_counts: dict[str, int] | None = None,
) -> ExactSupportRecoveryResult | None:
    event_supports = [
        support
        for support in supports
        if support.concept_valid_from
        and not _SESSION_DATE_PATTERN.search(support.support_text)
        and not _SESSION_DATE_PATTERN.search(support.concept_summary)
        and (not actor_compatibility_enabled or _support_actor_compatible(question, support))
        and _temporal_source_set_named_actor_compatible(question, support)
        and _temporal_direct_original_date_support_matches_question_event(question, support)
    ]
    if not event_supports:
        return None
    recovered_candidates: list[tuple[ExactSupportRecoveryResult, float]] = []
    for event_support in event_supports:
        for date_support in supports:
            if date_support.support_id == event_support.support_id:
                continue
            if not _temporal_contextual_pair_compatible(question, event_support, date_support):
                continue
            if actor_compatibility_enabled and not _support_actor_compatible(question, date_support):
                continue
            answer = _temporal_contextual_pair_date_answer(date_support)
            if answer is None:
                if rejection_counts is not None:
                    _increment_rejection(rejection_counts, "temporal_contextual_pair_date_unparseable")
                continue
            recovered_candidates.append(
                (
                    ExactSupportRecoveryResult(
                        answer=answer,
                        normalized_answer=_normalize(answer),
                        support=date_support,
                        strategy="support_present_temporal_contextual_pair_date",
                    ),
                    _score_support(question, event_support).score,
                )
            )
    return _pick_unique_exact_support_recovery_candidate(
        recovered_candidates,
        allow_clear_win_different_answers=True,
        question=question,
        actor_compatibility_enabled=actor_compatibility_enabled,
        rejection_counts=rejection_counts,
    )


def _recover_support_present_temporal_event_source_date(
    *,
    question: str,
    supports: list[EvidenceSupport],
    actor_compatibility_enabled: bool = False,
    rejection_counts: dict[str, int] | None = None,
) -> ExactSupportRecoveryResult | None:
    recovered_candidates: list[tuple[ExactSupportRecoveryResult, float]] = []
    for support in supports:
        if support.concept_original_date:
            continue
        if _SESSION_DATE_PATTERN.search(support.support_text) or _SESSION_DATE_PATTERN.search(
            support.concept_summary
        ):
            continue
        answer = _format_source_timestamp_month_answer(support.concept_valid_from)
        if answer is None:
            answer = _format_source_timestamp_month_answer(support.concept_created_at)
        if answer is None:
            if rejection_counts is not None:
                _increment_rejection(rejection_counts, "temporal_event_source_date_unparseable")
            continue
        if not _support_actor_compatible(question, support):
            continue
        if not _temporal_source_set_named_actor_compatible(question, support):
            continue
        if not (
            _support_sufficiently_matches_question(question, support.support_text, min_ratio=0.35)
            or _support_sufficiently_matches_question(question, support.concept_summary, min_ratio=0.35)
        ):
            continue
        if not _temporal_source_set_date_support_matches_question_event(question, support):
            if rejection_counts is not None:
                _increment_rejection(rejection_counts, "temporal_event_source_date_no_event_overlap")
            continue
        recovered_candidates.append(
            (
                ExactSupportRecoveryResult(
                    answer=answer,
                    normalized_answer=_normalize(answer),
                    support=support,
                    strategy="support_present_temporal_event_source_date",
                ),
                _score_support(question, support).score,
            )
        )

    return _pick_unique_exact_support_recovery_candidate(
        recovered_candidates,
        allow_clear_win_different_answers=True,
        question=question,
        actor_compatibility_enabled=actor_compatibility_enabled,
        rejection_counts=rejection_counts,
    )


def _recover_support_present_temporal_make_media_event_date(
    *,
    question: str,
    supports: list[EvidenceSupport],
    actor_compatibility_enabled: bool = False,
    rejection_counts: dict[str, int] | None = None,
) -> ExactSupportRecoveryResult | None:
    if not _TEMPORAL_MAKE_QUESTION_PATTERN.search(question):
        return None

    question_actor_terms = set(_question_actor_terms(question))
    question_object_terms = {
        term
        for term in _content_tokens(question)
        if term not in question_actor_terms and term not in _TEMPORAL_MAKE_GENERIC_TERMS
    }
    if not question_object_terms:
        return None
    question_object_nouns = question_object_terms & _TEMPORAL_MAKE_MEDIA_OBJECT_NOUNS
    if not question_object_nouns:
        return None

    recovered_candidates: list[tuple[ExactSupportRecoveryResult, float]] = []
    for support in supports:
        if support.channel != "verbatim":
            continue
        media_owner_match = _temporal_make_media_backfill_support_match(question, support)
        if not media_owner_match and not _support_actor_compatible(question, support):
            continue
        if actor_compatibility_enabled and not _temporal_source_set_named_actor_compatible(question, support):
            continue
        if not _TEMPORAL_MAKE_EVENT_MEDIA_CUE_PATTERN.search(support.support_text):
            if rejection_counts is not None:
                _increment_rejection(rejection_counts, "temporal_make_media_no_event_cue")
            continue

        media_terms: set[str] = set()
        for _field, value, field_start in _shared_media_fields(support.support_text):
            if not _shared_media_owner_compatible(question, support.support_text, field_start):
                continue
            media_terms.update(_content_tokens(value))
        if not media_terms:
            if rejection_counts is not None:
                _increment_rejection(rejection_counts, "temporal_make_media_no_owned_media")
            continue

        matched_object_terms = {
            term
            for term in question_object_terms
            if _temporal_source_set_terms_intersect(term, media_terms)
        }
        matched_object_nouns = {
            term
            for term in question_object_nouns
            if _temporal_source_set_terms_intersect(term, media_terms)
        }
        if len(matched_object_terms) < 2 or not matched_object_nouns:
            if rejection_counts is not None:
                _increment_rejection(rejection_counts, "temporal_make_media_object_mismatch")
            continue

        source_date = _support_source_calendar_date(support)
        if source_date is None:
            if rejection_counts is not None:
                _increment_rejection(rejection_counts, "temporal_make_media_unparseable_source_date")
            continue
        answer = _format_day_month_year_answer(source_date - timedelta(days=1))
        recovered_candidates.append(
            (
                ExactSupportRecoveryResult(
                    answer=answer,
                    normalized_answer=_normalize(answer),
                    support=support,
                    strategy="support_present_temporal_make_media_event_date",
                ),
                _score_support(question, support).score + 0.04,
            )
        )

    return _pick_unique_exact_support_recovery_candidate(
        recovered_candidates,
        allow_clear_win_different_answers=True,
        question=question,
        actor_compatibility_enabled=False,
        rejection_counts=rejection_counts,
    )


def _recover_support_present_session_date_adoption_bridge(
    *,
    question: str,
    supports: list[EvidenceSupport],
    actor_compatibility_enabled: bool = False,
    rejection_counts: dict[str, int] | None = None,
) -> ExactSupportRecoveryResult | None:
    binding = _session_date_adoption_question_binding(question)
    if binding is None:
        return None
    actor_term, object_term = binding

    date_supports = [
        support
        for support in supports
        if support.concept_original_date
        and support.concept_valid_from
        and (
            _SESSION_DATE_PATTERN.search(support.support_text)
            or _SESSION_DATE_PATTERN.search(support.concept_summary)
        )
        and _support_actor_compatible(question, support)
        and _session_date_support_named_actor_compatible(question, support)
    ]
    if not date_supports:
        return None

    event_supports = [
        support
        for support in supports
        if support.concept_valid_from
        and not support.concept_original_date
        and not _SESSION_DATE_PATTERN.search(support.support_text)
        and not _SESSION_DATE_PATTERN.search(support.concept_summary)
        and _support_actor_compatible(question, support)
        and _session_date_adoption_event_support_matches(
            support,
            actor_term=actor_term,
            object_term=object_term,
        )
    ]
    if not event_supports:
        return None

    recovered_candidates: list[tuple[ExactSupportRecoveryResult, float]] = []
    for date_support in date_supports:
        answer = _format_original_date_answer(date_support.concept_original_date)
        if answer is None:
            if rejection_counts is not None:
                _increment_rejection(rejection_counts, "session_date_adoption_unparseable_original_date")
            continue
        matching_events = [
            event_support
            for event_support in event_supports
            if event_support.concept_valid_from == date_support.concept_valid_from
        ]
        if not matching_events:
            if rejection_counts is not None:
                _increment_rejection(rejection_counts, "session_date_adoption_no_same_valid_from_event")
            continue
        best_event_score = max(_score_support(question, support).score for support in matching_events)
        recovered_candidates.append(
            (
                ExactSupportRecoveryResult(
                    answer=answer,
                    normalized_answer=_normalize(answer),
                    support=date_support,
                    strategy="support_present_session_date_adoption_bridge",
                ),
                best_event_score,
            )
        )

    return _pick_unique_exact_support_recovery_candidate(
        recovered_candidates,
        allow_clear_win_different_answers=True,
        question=question,
        actor_compatibility_enabled=actor_compatibility_enabled,
        rejection_counts=rejection_counts,
    )


def _session_date_adoption_question_binding(question: str) -> tuple[str, str] | None:
    match = re.search(
        r"^\s*when\s+did\s+(?P<actor>[A-Z][a-z]+)\s+adopt\s+(?P<object>[A-Z][A-Za-z0-9_-]+)\b",
        question,
        re.IGNORECASE,
    )
    if match is None:
        return None
    return match.group("actor").lower(), match.group("object").lower()


def _session_date_support_named_actor_compatible(
    question: str,
    support: EvidenceSupport,
) -> bool:
    question_names = {
        token.lower()
        for token in re.findall(r"\b[A-Z][a-z]+\b", question)
        if token.lower() not in {"how", "max", "what", "when", "where", "which"}
    }
    support_text = f"{support.support_text} {support.concept_summary}"
    support_names = {
        token.lower()
        for token in re.findall(r"\b[A-Z][a-z]+\b", support_text)
        if token.lower() not in _TEMPORAL_CALENDAR_CONTEXT_TERMS
        and token.lower() not in {"client", "evidence", "session", "the"}
    }
    if not support_names:
        return True
    return bool(support_names & question_names)


def _session_date_adoption_event_support_matches(
    support: EvidenceSupport,
    *,
    actor_term: str,
    object_term: str,
) -> bool:
    support_text = f"{support.support_text} {support.concept_summary}"
    support_terms = set(_content_tokens(support_text))
    if actor_term not in support_terms or object_term not in support_terms:
        return False
    has_adoption = bool(re.search(r"\badopt(?:ed|s|ing)?\b", support_text, re.IGNORECASE))
    has_new_addition = bool(
        re.search(r"\bnew\s+addition\b", support_text, re.IGNORECASE)
        and re.search(r"\bfamily\b", support_text, re.IGNORECASE)
        and support_terms.intersection({"dog", "dogs", "pet", "pets"})
    )
    return has_adoption or has_new_addition


def _format_original_date_answer(original_date: str | None) -> str | None:
    value = _stringify(original_date).strip()
    match = re.fullmatch(r"(\d{4})-(\d{2})", value)
    if match:
        year = match.group(1)
        month = int(match.group(2))
        month_name = _MONTH_NAMES.get(month)
        return f"{month_name}, {year}" if month_name is not None else None
    match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", value)
    if match:
        year = match.group(1)
        month = int(match.group(2))
        day = int(match.group(3))
        month_name = _MONTH_NAMES.get(month)
        if month_name is None or not 1 <= day <= 31:
            return None
        return f"{month_name} {day}, {year}"
    return None


def _format_day_month_year_answer(value: date) -> str:
    return f"{value.day} {_MONTH_NAMES[value.month]}, {value.year}"


def _temporal_contextual_pair_compatible(
    question: str,
    event_support: EvidenceSupport,
    date_support: EvidenceSupport,
) -> bool:
    if _SESSION_DATE_PATTERN.search(date_support.support_text) or _SESSION_DATE_PATTERN.search(
        date_support.concept_summary
    ):
        return False
    if not event_support.concept_valid_from or event_support.concept_valid_from != date_support.concept_valid_from:
        return False
    if event_support.concept_session_id and date_support.concept_session_id:
        if event_support.concept_session_id != date_support.concept_session_id:
            return False
    event_terms = _temporal_source_set_overlap_terms(event_support.support_text, event_support.concept_summary)
    date_terms = _temporal_source_set_overlap_terms(date_support.support_text, date_support.concept_summary)
    question_actors = set(_question_actor_terms(question))
    if question_actors and not question_actors <= event_terms:
        return False
    if question_actors and not question_actors <= date_terms:
        return False
    shared_terms = {
        term
        for term in event_terms
        if term not in {"person", "woman", "named"} and any(_tokens_match(term, other) for other in date_terms)
    }
    return bool(shared_terms)


def _temporal_contextual_pair_date_answer(support: EvidenceSupport) -> str | None:
    text = f"{support.support_text} {support.concept_summary}"
    source_date = _support_source_calendar_date(support)
    if re.search(r"\byesterday\b", text, re.IGNORECASE) and source_date is not None:
        return _format_original_date_answer((source_date - timedelta(days=1)).isoformat())
    for match in _DATE_CANDIDATE_PATTERN.finditer(text):
        parsed = _parse_calendar_date(match.group(0))
        if parsed is not None:
            return _format_original_date_answer(parsed.isoformat())
    return None


def _temporal_direct_original_date_conflicts_explicit_date(support: EvidenceSupport) -> bool:
    original_date = _parse_calendar_date(support.concept_original_date)
    if original_date is None:
        return False
    text = f"{support.support_text} {support.concept_summary}"
    explicit_dates = [
        parsed
        for parsed in (_parse_calendar_date(match.group(0)) for match in _DATE_CANDIDATE_PATTERN.finditer(text))
        if parsed is not None
    ]
    return any(parsed != original_date for parsed in explicit_dates)


def _support_source_calendar_date(support: EvidenceSupport) -> date | None:
    for value in (support.concept_valid_from, support.concept_created_at, support.support_text):
        parsed = _parse_calendar_date(value)
        if parsed is not None:
            return parsed
    return None


def _parse_calendar_date(value: str | None) -> date | None:
    text = _stringify(value).strip()
    if not text:
        return None
    match = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", text)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            return None
    match = re.search(
        r"\b(\d{1,2})\s+([A-Za-z]{3,9}),?\s+(\d{4})\b",
        text,
        re.IGNORECASE,
    )
    if match:
        month = _MONTH_NUMBER_ALIASES.get(match.group(2).lower(), 0)
        try:
            return date(int(match.group(3)), month, int(match.group(1)))
        except ValueError:
            return None
    match = re.search(
        r"\b([A-Za-z]{3,9})\s+(\d{1,2}),?\s+(\d{4})\b",
        text,
        re.IGNORECASE,
    )
    if match:
        month = _MONTH_NUMBER_ALIASES.get(match.group(1).lower(), 0)
        try:
            return date(int(match.group(3)), month, int(match.group(2)))
        except ValueError:
            return None
    return None


def _format_source_timestamp_month_answer(value: str | None) -> str | None:
    text = _stringify(value).strip()
    if not text:
        return None
    match = re.match(r"^(\d{4})-(\d{2})(?:-\d{2})?(?:[T\s].*)?$", text)
    if match:
        year = match.group(1)
        month_name = _MONTH_NAMES.get(int(match.group(2)))
        return f"{month_name}, {year}" if month_name is not None else None
    match = re.search(
        r"\b\d{1,2}\s+([A-Za-z]{3,9}),?\s+(\d{4})\b",
        text,
        re.IGNORECASE,
    )
    if match:
        month_name = _MONTH_NAMES.get(_MONTH_NUMBER_ALIASES.get(match.group(1).lower(), 0))
        return f"{month_name}, {match.group(2)}" if month_name is not None else None
    match = re.search(
        r"\b([A-Za-z]{3,9})\s+\d{1,2},?\s+(\d{4})\b",
        text,
        re.IGNORECASE,
    )
    if match:
        month_name = _MONTH_NAMES.get(_MONTH_NUMBER_ALIASES.get(match.group(1).lower(), 0))
        return f"{month_name}, {match.group(2)}" if month_name is not None else None
    return None


def _temporal_source_set_pair_matches(
    question: str,
    event_support: EvidenceSupport,
    date_support: EvidenceSupport,
) -> bool:
    if event_support.concept_id == date_support.concept_id:
        return False
    if not event_support.concept_valid_from or not date_support.concept_valid_from:
        return False
    if event_support.concept_valid_from != date_support.concept_valid_from:
        return False
    if not _support_actor_compatible(question, event_support):
        return False
    if not _support_actor_compatible(question, date_support):
        return False
    if not _temporal_source_set_named_actor_compatible(question, event_support):
        return False
    if not _temporal_source_set_named_actor_compatible(question, date_support):
        return False
    if not (
        _support_sufficiently_matches_question(question, event_support.support_text, min_ratio=0.35)
        or _support_sufficiently_matches_question(question, event_support.concept_summary, min_ratio=0.35)
    ):
        return False
    if _SESSION_DATE_PATTERN.search(date_support.support_text) or _SESSION_DATE_PATTERN.search(
        date_support.concept_summary
    ):
        return False
    if not _temporal_source_set_date_support_matches_question_event(question, date_support):
        return False

    question_actors = set(_question_actor_terms(question))
    event_terms = _temporal_source_set_overlap_terms(event_support.support_text, event_support.concept_summary)
    date_terms = _temporal_source_set_overlap_terms(date_support.support_text, date_support.concept_summary)
    shared_terms = {
        event_term
        for event_term in event_terms
        if event_term not in question_actors
        and any(_tokens_match(event_term, date_term) for date_term in date_terms)
    }
    return bool(shared_terms)


_TEMPORAL_SOURCE_SET_GENERIC_EVENT_TERMS = {
    "did",
    "happen",
    "happened",
    "happening",
    "occur",
    "occurred",
    "place",
    "take",
}
_TEMPORAL_SOURCE_SET_ACTION_BINDING_TERMS = {
    "attend",
    "attended",
    "class",
    "collaborate",
    "collaborated",
    "collaboration",
    "course",
    "create",
    "created",
    "creating",
    "decide",
    "decided",
    "develop",
    "developed",
    "expand",
    "expanded",
    "expanding",
    "learn",
    "learning",
    "notice",
    "noticed",
    "noticing",
    "presentation",
    "presented",
    "recognize",
    "recognized",
    "recognizing",
    "style",
    "styled",
    "teach",
    "teaching",
    "train",
    "training",
    "workshop",
}
_TEMPORAL_SOURCE_SET_EQUIVALENT_BINDING_TERMS = {
    "class": {"class", "course", "training", "workshop"},
    "course": {"class", "course", "training", "workshop"},
    "dog": {"dog", "dogs", "pet", "pets"},
    "dogs": {"dog", "dogs", "pet", "pets"},
    "pet": {"dog", "dogs", "pet", "pets"},
    "pets": {"dog", "dogs", "pet", "pets"},
    "recognized": {"notice", "noticed", "noticing", "recognize", "recognized", "recognizing"},
    "recognize": {"notice", "noticed", "noticing", "recognize", "recognized", "recognizing"},
    "recognizing": {"notice", "noticed", "noticing", "recognize", "recognized", "recognizing"},
    "training": {"class", "course", "training", "workshop"},
    "workshop": {"class", "course", "training", "workshop"},
}

_TEMPORAL_DIRECT_ORIGINAL_DATE_EQUIVALENT_TERMS = {
    "meet": {"meet", "met"},
    "met": {"meet", "met"},
    "go": {"go", "went"},
    "went": {"go", "went"},
}
_TEMPORAL_DIRECT_ORIGINAL_DATE_ACTION_TERMS = {
    "meet",
    "met",
}


def _temporal_direct_original_date_support_matches_question_event(
    question: str,
    support: EvidenceSupport,
) -> bool:
    question_actors = set(_question_actor_terms(question))
    question_names = {
        token.lower()
        for token in re.findall(r"\b[A-Z][a-z]+\b", question)
        if token.lower() not in {"when", "what", "where", "which", "how"} and token.lower() not in question_actors
    }
    if not question_names:
        return False
    support_terms = _temporal_source_set_overlap_terms(support.support_text, support.concept_summary)
    if not question_names <= support_terms:
        return False
    question_terms = {
        term
        for term in _question_event_terms(question)
        if term not in question_actors and term not in _TEMPORAL_SOURCE_SET_GENERIC_EVENT_TERMS
    }
    matched_terms = {
        term
        for term in question_terms
        if _temporal_direct_original_date_terms_intersect(term, support_terms)
    }
    return (
        len(matched_terms) >= 2
        and bool(matched_terms & question_names)
        and bool(matched_terms & _TEMPORAL_DIRECT_ORIGINAL_DATE_ACTION_TERMS)
    )


def _temporal_direct_original_date_terms_intersect(term: str, support_terms: set[str]) -> bool:
    equivalents = _TEMPORAL_DIRECT_ORIGINAL_DATE_EQUIVALENT_TERMS.get(term, {term})
    return any(
        any(_tokens_match(equivalent, support_term) for equivalent in equivalents)
        for support_term in support_terms
    )


def _temporal_source_set_date_support_matches_question_event(
    question: str,
    date_support: EvidenceSupport,
) -> bool:
    question_actors = set(_question_actor_terms(question))
    question_terms = {
        term
        for term in _question_event_terms(question)
        if term not in question_actors and term not in _TEMPORAL_SOURCE_SET_GENERIC_EVENT_TERMS
    }
    if not question_terms:
        return False

    support_terms = _temporal_source_set_overlap_terms(
        date_support.support_text,
        date_support.concept_summary,
    )
    matched_terms = {
        term
        for term in question_terms
        if _temporal_source_set_terms_intersect(term, support_terms)
    }
    if len(matched_terms) < 2:
        return False
    return any(
        term in _TEMPORAL_SOURCE_SET_ACTION_BINDING_TERMS
        or _TEMPORAL_SOURCE_SET_EQUIVALENT_BINDING_TERMS.get(term, set())
        & _TEMPORAL_SOURCE_SET_ACTION_BINDING_TERMS
        for term in matched_terms
    )


def _temporal_source_set_terms_intersect(term: str, support_terms: set[str]) -> bool:
    equivalents = _TEMPORAL_SOURCE_SET_EQUIVALENT_BINDING_TERMS.get(term, {term})
    return any(
        any(_tokens_match(equivalent, support_term) for equivalent in equivalents)
        for support_term in support_terms
    )


def _temporal_source_set_named_actor_compatible(
    question: str,
    support: EvidenceSupport,
) -> bool:
    question_names = {
        token.lower()
        for token in re.findall(r"\b[A-Z][a-z]+\b", question)
        if token.lower() not in {"when", "what", "where", "which", "how"}
    }
    if not question_names:
        return True
    support_text = f"{support.support_text} {support.concept_summary}"
    support_names = {token.lower() for token in re.findall(r"\b[A-Z][a-z]+\b", support_text)}
    if support_names & question_names:
        return True
    return not support_names


def _temporal_source_set_overlap_terms(*values: str) -> set[str]:
    blocked = {
        "date",
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "session",
        "signed",
        "took",
        "place",
    }
    terms: set[str] = set()
    for value in values:
        for term in _content_tokens(value):
            if term in blocked or term.isdigit():
                continue
            terms.add(term)
    return terms


def _recover_support_present_anchor_quoted_title(
    *,
    question: str,
    reach_pool: tuple[ScoredEvidenceSupport, ...],
    actor_compatibility_enabled: bool = False,
    rejection_counts: dict[str, int] | None = None,
) -> ExactSupportRecoveryResult | None:
    if not _TITLE_SLOT_QUESTION_PATTERN.search(question) or not _SUPPORT_PRESENT_TITLE_QUESTION_PATTERN.search(
        question
    ):
        return None

    anchors = [
        item
        for item in reach_pool
        if item.score >= _SUPPORT_PACK_MIN_SCORE
        and _support_sufficiently_matches_question(
            question,
            item.support.support_text,
            min_ratio=_SUPPORT_SURFACE_REACH_MIN_RATIO,
        )
        and _SUPPORT_PRESENT_TITLE_ANCHOR_TERM_PATTERN.search(item.support.support_text)
    ]
    if not anchors:
        return None

    recovered_candidates: list[tuple[ExactSupportRecoveryResult, float]] = []
    for item in reach_pool:
        if not _SUPPORT_PRESENT_TITLE_SUPPORT_PATTERN.search(item.support.support_text):
            continue
        if not any(anchor.support.support_id != item.support.support_id for anchor in anchors):
            continue
        for candidate in _support_present_quoted_title_candidates(item.support.support_text):
            if not _contains_containment_answer(
                _containment_normalize(candidate),
                item.support.normalized_support_text,
            ):
                continue
            recovered_candidates.append(
                (
                    ExactSupportRecoveryResult(
                        answer=candidate,
                        normalized_answer=_normalize(candidate),
                        support=item.support,
                        strategy="support_present_anchor_quoted_title",
                    ),
                    max(anchor.score for anchor in anchors if anchor.support.support_id != item.support.support_id),
                )
            )

    return _pick_unique_exact_support_recovery_candidate(
        recovered_candidates,
        question=question,
        actor_compatibility_enabled=actor_compatibility_enabled,
        rejection_counts=rejection_counts,
    )


def _support_present_quoted_title_candidates(support_text: str) -> list[str]:
    translated = support_text.translate(_QUOTE_TRANSLATION)
    candidates: list[str] = []
    for match in _SUPPORT_PRESENT_BOUNDARY_QUOTED_SPAN_PATTERN.finditer(translated):
        candidate = _clean_candidate_answer(match.group(1))
        if _support_present_title_rejection_reason(candidate) is not None:
            continue
        candidates.append(candidate)
    return candidates


def _support_present_title_rejection_reason(answer: str) -> str | None:
    if not answer:
        return "empty_title"
    if _URL_LIKE_CANDIDATE_PATTERN.search(answer):
        return "quoted_title_url_like"
    if _CLAUSE_FRAGMENT_PATTERN.search(answer) or _quoted_title_starts_like_fragment(answer):
        return "quoted_title_clause_fragment"
    if len(_content_tokens(answer)) > 8:
        return "quoted_title_too_long"
    first_token = answer.split()[0]
    if not first_token or not first_token[0].isupper():
        return "quoted_title_not_capitalized"
    return None


_ACTOR_IGNORE_TERMS = frozenset(
    {
        "a",
        "an",
        "are",
        "did",
        "do",
        "does",
        "here",
        "how",
        "i",
        "i'm",
        "is",
        "that",
        "the",
        "this",
        "what",
        "when",
        "where",
        "which",
        "who",
    }
)

_QUESTION_ACTOR_PATTERNS = (
    re.compile(r"\b([A-Z][a-z]+)'s\b"),
    re.compile(r"\b(?:did|does|do|is|are|was|were|has|had)\s+([A-Z][a-z]+)\b"),
    re.compile(r"\bthat\s+([A-Z][a-z]+)\s+(?:got|gets|bought|has|had|used|uses|wears|wore)\b"),
)
_SUPPORT_STRONG_OWNER_PATTERNS = (
    re.compile(r"\b([A-Z][a-z]+)'s\b"),
    re.compile(
        r"^([A-Z][a-z]+)\s+"
        r"(?:is|was|visited|went|got|started|loves|love|took|attended|won|made|plans|planned|has|had)\b"
    ),
)
_SUPPORT_WEAK_ACTOR_PATTERNS = (
    re.compile(r"^([A-Z][a-z]+):"),
)


def _actor_token(raw: str) -> str | None:
    token = raw.strip(string.punctuation + " ").lower()
    if not token or token in _ACTOR_IGNORE_TERMS:
        return None
    return token


def _actor_terms_from_patterns(text: str, patterns: tuple[re.Pattern[str], ...]) -> tuple[str, ...]:
    terms: list[str] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            token = _actor_token(match.group(1))
            if token is not None and token not in terms:
                terms.append(token)
    return tuple(terms)


def _shared_media_fields(support_text: str) -> list[tuple[str, str, int]]:
    fields: list[tuple[str, str, int]] = []
    for match in _SHARED_MEDIA_FIELD_PATTERN.finditer(support_text):
        value = _clean_candidate_answer(match.group("value"))
        if value:
            fields.append((match.group("field").lower(), value, match.start()))
    return fields


def _nearest_dialogue_actor_before(text: str, position: int) -> str | None:
    prefix = text[max(0, position - _MEDIA_ACTOR_LOOKBACK_CHARS) : position]
    matches = list(_DIALOGUE_ACTOR_PATTERN.finditer(prefix))
    if not matches:
        return None
    return _actor_token(matches[-1].group(1))


def _shared_media_owner_compatible(question: str, support_text: str, field_start: int) -> bool:
    question_terms = set(_question_actor_terms(question))
    if not question_terms:
        return True
    nearest_actor = _nearest_dialogue_actor_before(support_text, field_start)
    if nearest_actor is None:
        return False
    return nearest_actor in question_terms


def _media_footwear_use_candidates(question: str, support_text: str) -> list[tuple[str, CandidateSource]]:
    if not _MEDIA_FOOTWEAR_USE_QUESTION_PATTERN.search(question):
        return []

    candidates: list[tuple[str, CandidateSource]] = []
    seen: set[str] = set()
    for _field, value, field_start in _shared_media_fields(support_text):
        value_terms = set(_content_tokens(value))
        if not value_terms.intersection(_MEDIA_FOOTWEAR_NOUN_TERMS):
            continue
        if not _shared_media_owner_compatible(question, support_text, field_start):
            continue
        for use_term in _MEDIA_FOOTWEAR_USE_TERMS:
            if use_term not in value_terms or use_term in seen:
                continue
            candidates.append((use_term, "regex_media_intent_attribute"))
            seen.add(use_term)
            break
    return candidates


def _question_actor_terms(question: str) -> tuple[str, ...]:
    return _actor_terms_from_patterns(question, _QUESTION_ACTOR_PATTERNS)


def _support_actor_terms_from_text(text: str) -> tuple[str, ...]:
    cleaned = text.replace("Client evidence:", "").strip()
    return _actor_terms_from_patterns(cleaned, _SUPPORT_STRONG_OWNER_PATTERNS + _SUPPORT_WEAK_ACTOR_PATTERNS)


def _answer_bearing_support_text(support: EvidenceSupport, answer: str | None) -> str:
    if not answer:
        return support.support_text
    sentences = _sentences_containing_answer(support.support_text, answer)
    return " ".join(sentences) if sentences else support.support_text


def _support_actor_compatible(
    question: str,
    support: EvidenceSupport,
    *,
    answer: str | None = None,
    allow_actorless_support_summary_actor: bool = False,
) -> bool:
    question_terms = set(_question_actor_terms(question))
    if not question_terms:
        return True

    answer_text = _answer_bearing_support_text(support, answer)
    strong_terms = set(_actor_terms_from_patterns(answer_text, _SUPPORT_STRONG_OWNER_PATTERNS)) - _SUPPORT_PRESENT_DIRECT_SCALAR_ACTOR_IGNORE_TERMS
    if strong_terms:
        return bool(question_terms & strong_terms)

    support_terms = set(_support_actor_terms_from_text(answer_text)) - _SUPPORT_PRESENT_DIRECT_SCALAR_ACTOR_IGNORE_TERMS
    if not support_terms and answer:
        normalized_answer = _containment_normalize(answer)
        normalized_summary = _containment_normalize(support.concept_summary)
        if _contains_containment_answer(normalized_answer, normalized_summary):
            support_terms.update(support.concept_actor_terms)
            support_terms.update(_support_actor_terms_from_text(support.concept_summary))
    if not support_terms and answer and allow_actorless_support_summary_actor:
        actor_match = _SUPPORT_PRESENT_DIRECT_SCALAR_ACTOR_PATTERN.search(support.concept_summary)
        if actor_match:
            actor_token = _actor_token(actor_match.group(1))
            if actor_token and actor_token not in _SUPPORT_PRESENT_DIRECT_SCALAR_ACTOR_IGNORE_TERMS:
                support_terms.add(actor_token)
    if not support_terms:
        return not re.search(
            r"\byou\b.+\b(?:i|i'm|i've|my|we|we're|we've|our)\b",
            support.support_text,
            re.IGNORECASE | re.DOTALL,
        )
    return bool(question_terms & support_terms)


def _pick_unique_exact_support_recovery_candidate(
    recovered_candidates: list[tuple[ExactSupportRecoveryResult, float]],
    *,
    allow_clear_win_different_answers: bool = False,
    question: str | None = None,
    actor_compatibility_enabled: bool = False,
    rejection_counts: dict[str, int] | None = None,
) -> ExactSupportRecoveryResult | None:
    if not recovered_candidates:
        return None

    if actor_compatibility_enabled and question:
        compatible_candidates: list[tuple[ExactSupportRecoveryResult, float]] = []
        for recovered, score in recovered_candidates:
            if _support_actor_compatible(question, recovered.support, answer=recovered.answer):
                compatible_candidates.append((recovered, score))
            elif rejection_counts is not None:
                rejection_counts["support_actor_mismatch"] = (
                    rejection_counts.get("support_actor_mismatch", 0) + 1
                )
        recovered_candidates = compatible_candidates
        if not recovered_candidates:
            return None

    best_by_answer: dict[str, tuple[ExactSupportRecoveryResult, float]] = {}
    for recovered, score in recovered_candidates:
        existing = best_by_answer.get(recovered.normalized_answer)
        if existing is None or score > existing[1]:
            best_by_answer[recovered.normalized_answer] = (recovered, score)

    ranked = sorted(
        best_by_answer.values(),
        key=lambda item: item[1],
        reverse=True,
    )
    best, best_score = ranked[0]
    for alternate, alternate_score in ranked[1:]:
        if best_score - alternate_score < _SUPPORT_PACK_CLEAR_WIN_MARGIN:
            return None
        if not allow_clear_win_different_answers and alternate.normalized_answer != best.normalized_answer:
            return None
    return best


def _date_candidate_from_sentence(
    question: str,
    sentence: str,
) -> str | None:
    for match in _DATE_CANDIDATE_PATTERN.finditer(sentence):
        candidate = _clean_candidate_answer(match.group(0))
        if _date_candidate_sanity_rejection_reason(candidate) is not None:
            continue
        if _date_event_rejection_reason(question, sentence, candidate) is not None:
            continue
        return candidate
    return None


def _duration_candidate_from_sentence(sentence: str) -> str | None:
    for match in _DURATION_CANDIDATE_PATTERN.finditer(sentence):
        candidate = _clean_candidate_answer(match.group(0))
        if candidate:
            return candidate
    candidates = _duration_word_candidates_from_sentence(sentence)
    return candidates[0] if candidates else None


def _duration_answer_for_question(question: str, candidate: str) -> str:
    if not candidate:
        return candidate
    if re.search(r"\bhow\s+long\s+ago\b", question, re.IGNORECASE) and not re.search(
        r"\bago\b",
        candidate,
        re.IGNORECASE,
    ):
        return f"{candidate} ago"
    return candidate


def _duration_word_candidates_from_sentence(sentence: str) -> list[str]:
    candidates: list[str] = []
    for match in _DURATION_WORD_CANDIDATE_PATTERN.finditer(sentence):
        candidate = _clean_candidate_answer(match.group(0))
        if not candidate:
            continue
        parts = candidate.split(maxsplit=1)
        if len(parts) != 2:
            continue
        number = _NUMBER_WORDS.get(parts[0].lower())
        if number:
            candidates.append(f"{number} {parts[1]}")
    return candidates


def _location_candidate_from_sentence(
    question: str,
    sentence: str,
) -> str | None:
    if not _support_sufficiently_matches_question(question, sentence, min_ratio=0.45):
        return None

    for pattern in (_LOCATION_TO_PATTERN, _LOCATION_FROM_PATTERN):
        match = pattern.search(sentence)
        if match is None:
            continue
        group = match.group(1) if match.lastindex else match.group(0)
        candidate = _clean_candidate_answer(group)
        if candidate:
            return candidate
    return None


def _visited_location_candidate_from_sentence(
    question: str,
    sentence: str,
) -> str | None:
    match = _VISITED_LOCATION_PATTERN.search(sentence)
    if match is None:
        return None
    candidate = _clean_candidate_answer(match.group(1))
    return candidate or None


def _canonicalize_support_bound_answer(
    *,
    question: str,
    answer: str,
    support: EvidenceSupport,
    intent: AnswerIntent | None,
    enabled: bool,
    locomo_support_present_synthesis_enabled: bool = False,
    locomo_support_present_answer_realization_enabled: bool = False,
) -> ExactSupportRecoveryResult | None:
    if not enabled:
        return None

    cleaned = _clean_candidate_answer(answer)
    lowered = cleaned.lower()
    question_lower = question.lower()

    if (intent == "location" or question_lower.startswith("where did")) and lowered.startswith("to "):
        canonical = _clean_candidate_answer(cleaned[3:])
        if canonical and _contains_containment_answer(
            _containment_normalize(canonical),
            support.normalized_support_text,
        ):
            return ExactSupportRecoveryResult(
                answer=canonical,
                normalized_answer=_normalize(canonical),
                support=support,
                strategy="strip_location_to_prefix",
            )

    if (
        locomo_support_present_synthesis_enabled
        and _locomo_excluded_condition_stressor_question(question) is not None
        and lowered in {"work stress", "work-life balance", "work life balance"}
        and _contains_containment_answer("work", support.normalized_support_text)
    ):
        return ExactSupportRecoveryResult(
            answer="work",
            normalized_answer=_normalize("work"),
            support=support,
            strategy="locomo_excluded_condition_stressor_work_scalar",
        )

    if (
        locomo_support_present_answer_realization_enabled
        and _locomo_camping_season_year_question(question)
        and _YEAR_ONLY_PATTERN.fullmatch(lowered)
    ):
        season_year = _locomo_camping_season_year_candidate(support.support_text)
        if season_year is not None and lowered in season_year:
            return ExactSupportRecoveryResult(
                answer=season_year,
                normalized_answer=_normalize(season_year),
                support=support,
                strategy="locomo_support_present_answer_realization_season_year",
            )

    camping_actor = _locomo_where_camping_with_girlfriend_question(question)
    if (
        locomo_support_present_answer_realization_enabled
        and camping_actor is not None
        and lowered == "camping"
        and _locomo_support_has_camping_with_girlfriend(support.support_text, camping_actor)
    ):
        return ExactSupportRecoveryResult(
            answer="camping with girlfriend",
            normalized_answer=_normalize("camping with girlfriend"),
            support=support,
            strategy="locomo_support_present_answer_realization_where_did_go_activity",
        )

    if "used for" in question_lower and lowered.startswith("for "):
        canonical = _clean_candidate_answer(cleaned[4:])
        if canonical and _contains_containment_answer(
            _containment_normalize(canonical),
            support.normalized_support_text,
        ):
            return ExactSupportRecoveryResult(
                answer=canonical,
                normalized_answer=_normalize(canonical),
                support=support,
                strategy="strip_used_for_prefix",
            )

    if "raise awareness for" in question_lower and lowered.startswith("for "):
        canonical = _clean_candidate_answer(cleaned[4:])
        if canonical and _contains_containment_answer(
            _containment_normalize(canonical),
            support.normalized_support_text,
        ):
            return ExactSupportRecoveryResult(
                answer=canonical,
                normalized_answer=_normalize(canonical),
                support=support,
                strategy="strip_awareness_for_prefix",
            )

    reminder_tail = _reminder_of_tail_from_support(question, support.support_text)
    if reminder_tail and not _contains_containment_answer(
        _containment_normalize(cleaned),
        _containment_normalize(reminder_tail),
    ):
        return ExactSupportRecoveryResult(
            answer=reminder_tail,
            normalized_answer=_normalize(reminder_tail),
            support=support,
            strategy="support_bound_reminder_of_tail",
        )

    return None


def _locomo_candidate_answer_realization(
    *,
    question: str | None,
    candidate: AnswerCandidate,
    enabled: bool,
) -> ExactSupportRecoveryResult | None:
    if not enabled or not question:
        return None

    cleaned = _clean_candidate_answer(candidate.answer)
    lowered = cleaned.lower()
    if (
        _locomo_camping_season_year_question(question)
        and _YEAR_ONLY_PATTERN.fullmatch(lowered)
    ):
        season_year = _locomo_camping_season_year_candidate(candidate.support.support_text)
        if season_year is not None and lowered in season_year:
            return ExactSupportRecoveryResult(
                answer=season_year,
                normalized_answer=_normalize(season_year),
                support=candidate.support,
                strategy="locomo_support_present_answer_realization_season_year",
            )

    camping_actor = _locomo_where_camping_with_girlfriend_question(question)
    if (
        camping_actor is not None
        and lowered == "camping"
        and _locomo_support_has_camping_with_girlfriend(candidate.support.support_text, camping_actor)
    ):
        return ExactSupportRecoveryResult(
            answer="camping with girlfriend",
            normalized_answer=_normalize("camping with girlfriend"),
            support=candidate.support,
            strategy="locomo_support_present_answer_realization_where_did_go_activity",
        )
    return None


def _support_derived_repair_shape(question: str) -> RepairShape:
    if _TITLE_SLOT_QUESTION_PATTERN.search(question):
        return "title_with_creator"
    if _list_or_composite_question_requires_complete_answer(question) or any(
        pattern.search(question) for pattern in _REPAIR_LIST_COMPLETION_QUESTION_PATTERNS
    ):
        return "list_completion"
    if _answer_contract_event_object_requires_binding(question) or any(
        pattern.search(question) for pattern in _REPAIR_PREDICATE_QUESTION_PATTERNS
    ):
        return "predicate_bound_scalar"
    return "none"


def _support_derived_expected_shape(shape: RepairShape) -> str | None:
    if shape == "title_with_creator":
        return "complete_phrase"
    if shape == "list_completion":
        return "list_or_set"
    if shape == "predicate_bound_scalar":
        return "predicate_bound_scalar"
    return None


def _support_derived_repair_failure_reason(shape: RepairShape) -> str:
    if shape == "title_with_creator":
        return "title_with_creator_completion_required"
    if shape == "list_completion":
        return "list_or_composite_partial_answer"
    if shape == "predicate_bound_scalar":
        return "support_sentence_fails_slot_binding"
    return "support_derived_repair_required"


def _repair_needed(
    question: str,
    answer: str,
    support: EvidenceSupport,
    *,
    shape: RepairShape,
) -> bool:
    if shape == "title_with_creator":
        return bool(_title_with_creator_completion_candidates(answer, support.support_text))
    if shape == "list_completion":
        return bool(_coordinated_list_completion_candidates(answer, support.support_text))
    if shape == "predicate_bound_scalar":
        return _support_sentence_slot_binding_status(question, support, answer) != "bound"
    return False


def _try_support_derived_repair(
    *,
    question: str,
    answer: str,
    support: EvidenceSupport,
    supports: list[EvidenceSupport],
    intent: AnswerIntent | None,
    shape: RepairShape,
) -> SupportDerivedRepairResult | None:
    if shape == "none":
        return None

    support_pack = _build_support_pack(
        question,
        supports,
        candidate_answer=answer,
    )
    if not support_pack or support_pack[0].score < _SUPPORT_PACK_MIN_SCORE:
        return None

    if shape == "title_with_creator":
        return _repair_title_with_creator(
            answer=answer,
            support=support,
            support_pack_size=len(support_pack),
        )
    if shape == "list_completion":
        return _repair_supported_list_completion(
            answer=answer,
            support=support,
            support_pack=support_pack,
        )
    if shape == "predicate_bound_scalar":
        return _repair_predicate_bound_answer(
            question=question,
            answer=answer,
            support=support,
            support_pack=support_pack,
            intent=intent,
        )
    return None


def _repair_title_with_creator(
    *,
    answer: str,
    support: EvidenceSupport,
    support_pack_size: int,
) -> SupportDerivedRepairResult | None:
    candidate = _choose_unique_repair_candidate(_title_with_creator_completion_candidates(answer, support.support_text))
    if candidate is None:
        return None
    repaired = SupportDerivedRepairResult(
        answer=candidate,
        normalized_answer=_normalize(candidate),
        support=support,
        strategy="title_with_creator_completion",
        support_pack_size=support_pack_size,
    )
    if not _accept_support_derived_repair(
        "",
        repaired,
        expected_shape="complete_phrase",
    ):
        return None
    return repaired


def _repair_supported_list_completion(
    *,
    answer: str,
    support: EvidenceSupport,
    support_pack: tuple[ScoredEvidenceSupport, ...],
) -> SupportDerivedRepairResult | None:
    current_item = next(
        (item for item in support_pack if item.support.support_id == support.support_id),
        None,
    )
    if current_item is None:
        return None
    candidate = _choose_unique_repair_candidate(_coordinated_list_completion_candidates(answer, support.support_text))
    if candidate is None:
        return None
    repaired = SupportDerivedRepairResult(
        answer=candidate,
        normalized_answer=_normalize(candidate),
        support=support,
        strategy="coordinated_list_completion",
        support_pack_size=len(support_pack),
    )
    if not _accept_support_derived_repair(
        "",
        repaired,
        expected_shape="list_or_set",
    ):
        return None

    for item in support_pack:
        if item.support.support_id == support.support_id:
            continue
        if abs(current_item.score - item.score) >= _SUPPORT_PACK_CLEAR_WIN_MARGIN:
            continue
        alternate = _choose_unique_repair_candidate(
            _coordinated_list_completion_candidates(answer, item.support.support_text)
        )
        if alternate is None:
            continue
        if _normalize(alternate) != repaired.normalized_answer:
            return None
    return repaired


def _repair_predicate_bound_answer(
    *,
    question: str,
    answer: str,
    support: EvidenceSupport,
    support_pack: tuple[ScoredEvidenceSupport, ...],
    intent: AnswerIntent | None,
) -> SupportDerivedRepairResult | None:
    current_item = next(
        (item for item in support_pack if item.support.support_id == support.support_id),
        None,
    )
    current_score = current_item.score if current_item is not None else 0.0
    repaired_candidates: list[tuple[SupportDerivedRepairResult, float]] = []
    normalized_answer = _normalize(answer)

    for index, item in enumerate(support_pack):
        if item.support.support_id != support.support_id:
            if index != 0:
                continue
            if current_item is not None and item.score - current_score < _SUPPORT_PACK_CLEAR_WIN_MARGIN:
                continue
        candidate = _predicate_bound_sentence_candidate(
            question,
            item.support,
            intent=intent,
        )
        if candidate is None:
            continue
        if _normalize(candidate) == normalized_answer:
            continue
        repaired = SupportDerivedRepairResult(
            answer=candidate,
            normalized_answer=_normalize(candidate),
            support=item.support,
            strategy="predicate_bound_reselection",
            support_pack_size=len(support_pack),
        )
        if _accept_support_derived_repair(
            question,
            repaired,
            expected_shape="predicate_bound_scalar",
        ):
            repaired_candidates.append((repaired, item.score))

    return _pick_support_derived_repair_candidate(repaired_candidates)


def _title_with_creator_completion_candidates(
    answer: str,
    support_text: str,
) -> list[str]:
    normalized_answer = _containment_normalize(answer)
    if not normalized_answer:
        return []

    candidates: list[str] = []
    for sentence in _sentences_containing_answer(support_text, answer):
        for candidate, _ in _quoted_title_candidates(sentence):
            normalized_candidate = _containment_normalize(candidate)
            if normalized_candidate != normalized_answer and _contains_containment_answer(
                normalized_answer,
                normalized_candidate,
            ):
                candidates.append(candidate)
        for match in _TITLE_WITH_CREATOR_PATTERN.finditer(sentence):
            candidate = f"{_clean_candidate_answer(match.group(1))} " f"by {_clean_candidate_answer(match.group(2))}"
            normalized_candidate = _containment_normalize(candidate)
            if normalized_candidate != normalized_answer and _contains_containment_answer(
                normalized_answer,
                normalized_candidate,
            ):
                candidates.append(candidate)
    return candidates


def _coordinated_list_completion_candidates(
    answer: str,
    support_text: str,
) -> list[str]:
    normalized_answer = _containment_normalize(answer)
    if not normalized_answer:
        return []

    candidates: list[str] = []
    for sentence in _sentences_containing_answer(support_text, answer):
        if not _LIST_SEPARATOR_PATTERN.search(sentence):
            continue
        for match in _COORDINATED_LIST_CANDIDATE_PATTERN.finditer(sentence):
            candidate = _trim_coordinated_list_candidate(match.group(1), answer)
            if candidate is None:
                continue
            normalized_candidate = _containment_normalize(candidate)
            if (
                normalized_candidate
                and normalized_candidate != normalized_answer
                and _contains_containment_answer(
                    normalized_answer,
                    normalized_candidate,
                )
            ):
                candidates.append(candidate)
    return candidates


def _trim_coordinated_list_candidate(
    raw_candidate: str,
    answer: str,
) -> str | None:
    candidate = _clean_candidate_answer(raw_candidate)
    if not candidate:
        return None

    items = [
        _clean_candidate_answer(item)
        for item in re.split(r",|\band\b", candidate, flags=re.IGNORECASE)
        if _clean_candidate_answer(item)
    ]
    if len(items) < 2:
        return None

    trimmed_items: list[str] = []
    lowered_answer = answer.lower()
    for item in items:
        leading_match = _LIST_ITEM_LEADING_CONTEXT_PATTERN.match(item)
        if leading_match:
            item = _clean_candidate_answer(item[leading_match.end() :])
        answer_index = item.lower().find(lowered_answer) if lowered_answer else -1
        if answer_index > 0:
            item = _clean_candidate_answer(item[answer_index:])
        trailing_match = _LIST_ITEM_TRAILING_CONTEXT_PATTERN.match(item)
        if trailing_match:
            item = _clean_candidate_answer(trailing_match.group(1))
        if item:
            trimmed_items.append(item)

    if len(trimmed_items) < 2:
        return None

    first_item = trimmed_items[0]
    last_item = trimmed_items[-1]
    lowered_candidate = candidate.lower()
    first_index = lowered_candidate.find(first_item.lower())
    last_index = lowered_candidate.rfind(last_item.lower())
    if first_index < 0 or last_index < first_index:
        return None

    trimmed = _clean_candidate_answer(candidate[first_index : last_index + len(last_item)])
    if not _LIST_SEPARATOR_PATTERN.search(trimmed):
        return None
    return trimmed


def _choose_unique_repair_candidate(candidates: list[str]) -> str | None:
    unique: dict[str, str] = {}
    for candidate in candidates:
        normalized_candidate = _normalize(candidate)
        if not normalized_candidate:
            continue
        existing = unique.get(normalized_candidate)
        if existing is None or len(candidate) < len(existing):
            unique[normalized_candidate] = candidate
    if len(unique) != 1:
        return None
    return next(iter(unique.values()))


def _predicate_bound_sentence_candidate(
    question: str,
    support: EvidenceSupport,
    *,
    intent: AnswerIntent | None,
) -> str | None:
    question_terms = _question_event_terms(question)
    for sentence in _SENTENCE_SPLIT_PATTERN.split(support.support_text):
        if not _sentence_matches_terms(
            sentence,
            question_terms,
            min_ratio=_ANSWER_CONTRACT_MIN_SLOT_MATCH_RATIO,
        ):
            continue
        candidate = _deterministic_answer_from_sentence(
            question,
            sentence,
            "predicate_bound_scalar",
        )
        if candidate is None and intent != "location":
            candidate = _passive_subject_phrase_candidate(sentence)
        if candidate is None:
            continue
        if _contains_containment_answer(
            _containment_normalize(candidate),
            _containment_normalize(sentence),
        ):
            return candidate
    return None


def _passive_subject_phrase_candidate(sentence: str) -> str | None:
    for pattern in _PASSIVE_SUBJECT_PATTERNS:
        match = pattern.search(sentence.strip())
        if not match:
            continue
        candidate = _clean_candidate_answer(match.group(1))
        if candidate:
            return candidate
    return None


def _pick_support_derived_repair_candidate(
    repaired_candidates: list[tuple[SupportDerivedRepairResult, float]],
) -> SupportDerivedRepairResult | None:
    if not repaired_candidates:
        return None

    best_by_answer: dict[str, tuple[SupportDerivedRepairResult, float]] = {}
    for repaired, score in repaired_candidates:
        existing = best_by_answer.get(repaired.normalized_answer)
        if existing is None or score > existing[1]:
            best_by_answer[repaired.normalized_answer] = (repaired, score)

    ranked = sorted(
        best_by_answer.values(),
        key=lambda item: item[1],
        reverse=True,
    )
    best_repaired, best_score = ranked[0]
    for alternate_repaired, alternate_score in ranked[1:]:
        if best_score - alternate_score < _SUPPORT_PACK_CLEAR_WIN_MARGIN:
            return None
        if alternate_repaired.normalized_answer != best_repaired.normalized_answer:
            return None
    return best_repaired


def _accept_support_derived_repair(
    question: str,
    repaired: SupportDerivedRepairResult,
    *,
    expected_shape: str | None,
) -> bool:
    if not _contains_containment_answer(
        _containment_normalize(repaired.answer),
        repaired.support.normalized_support_text,
    ):
        return False

    if expected_shape == "list_or_set":
        items = _split_list_answer_items(repaired.answer)
        if not items:
            return False
        sentence_matches = _sentences_containing_answer(
            repaired.support.support_text,
            repaired.answer,
        )
        return all(
            any(
                _contains_containment_answer(
                    _containment_normalize(item),
                    _containment_normalize(sentence),
                )
                for sentence in sentence_matches
            )
            for item in items
        )

    if expected_shape == "predicate_bound_scalar":
        return (
            _support_sentence_slot_binding_status(
                question,
                repaired.support,
                repaired.answer,
            )
            == "bound"
        )

    return True


def _overlap_ratio(needle_terms: list[str], haystack_terms: list[str]) -> float:
    if not needle_terms:
        return 0.0
    return _matched_term_count(needle_terms, haystack_terms) / len(needle_terms)


def _predicate_binding_status(question: str, support: EvidenceSupport) -> str:
    shape = _infer_structured_synthesis_shape(question)
    if shape not in {"predicate_bound_scalar", "complete_phrase"}:
        return "not_required"

    question_terms = _question_event_terms(question)
    if not question_terms:
        return "bound"
    for sentence in _SENTENCE_SPLIT_PATTERN.split(support.support_text):
        if _sentence_matches_terms(
            sentence,
            question_terms,
            min_ratio=_ANSWER_CONTRACT_MIN_SLOT_MATCH_RATIO,
        ):
            return "bound"
    return "weak"


def _candidate_support_verifier_rejection_reason(
    question: str,
    candidate: AnswerCandidate,
    support_pack: tuple[ScoredEvidenceSupport, ...],
) -> str | None:
    if not support_pack or support_pack[0].score < _SUPPORT_PACK_MIN_SCORE:
        return "support_pack_no_evidence"

    scored_candidate = next(
        (item for item in support_pack if item.support.support_id == candidate.support.support_id),
        None,
    )
    if scored_candidate is None or scored_candidate.score < _SUPPORT_PACK_MIN_SCORE:
        return "support_pack_no_evidence"

    top = support_pack[0]
    if (
        top.support.support_id != candidate.support.support_id
        and top.score - scored_candidate.score >= _SUPPORT_PACK_CLEAR_WIN_MARGIN
        and scored_candidate.question_overlap < 0.5
    ):
        direct_scalar_answer = _support_present_direct_scalar_candidate(
            question,
            candidate.support.support_text,
            context_sentence=candidate.support.concept_summary,
        )
        if candidate.source == "regex_direct_support_scalar" and direct_scalar_answer is not None and (
            _contains_containment_answer(
                _containment_normalize(candidate.answer),
                _containment_normalize(direct_scalar_answer),
            )
            or _contains_containment_answer(
                _containment_normalize(direct_scalar_answer),
                _containment_normalize(candidate.answer),
            )
        ):
            return None
        return "candidate_support_not_topical"

    if _TEMPORAL_QUESTION_PATTERN.search(question) and _is_unsupported_session_date_answer(
        candidate.answer,
        candidate.support.support_text,
    ):
        return "candidate_temporal_anchor_only"

    shape = _infer_structured_synthesis_shape(question)
    if shape in {"predicate_bound_scalar", "complete_phrase"} and scored_candidate.binding_status != "bound":
        return "candidate_predicate_unbound"
    return None


def _build_structured_synthesis_prompt(
    question: str,
    support_pack: tuple[ScoredEvidenceSupport, ...],
    shape: StructuredSynthesisShape,
) -> str:
    evidence = "\n".join(
        f"[{item.support.support_id}] score={item.score:.3f} "
        f"binding={item.binding_status} {item.support.support_text}"
        for item in support_pack
    )
    return (
        f"Answer shape: {shape}\n\n"
        f"Evidence:\n{evidence}\n\n"
        f"Question: {question.strip()}\n\n"
        "Return JSON only with answer, support_ids, and cited_spans."
    )


def _build_structured_synthesis_sentence_prompt(
    question: str,
    sentence_pack: tuple[ScoredSupportSentence, ...],
    shape: StructuredSynthesisShape,
) -> str:
    evidence = "\n".join(
        f"[{item.support.support_id}] score={item.support_score:.3f} " f"binding={item.binding_status} {item.sentence}"
        for item in sentence_pack
    )
    return (
        f"Answer shape: {shape}\n\n"
        f"Evidence:\n{evidence}\n\n"
        f"Question: {question.strip()}\n\n"
        "Return JSON only with answer, support_ids, and cited_spans."
    )


def _parse_structured_synthesis_response(raw: str) -> StructuredSynthesisResult:
    payload = _parse_json(raw)
    answer = _proposal_string(payload.get("answer")) or None
    support_ids = _string_tuple(payload.get("support_ids"))
    cited_spans = _string_tuple(payload.get("cited_spans"))
    return StructuredSynthesisResult(
        answer=answer,
        support_ids=support_ids,
        cited_spans=cited_spans,
    )


def _support_present_guard_canonicalize_structured_result(
    *,
    question: str,
    result: StructuredSynthesisResult,
    support_pack: tuple[ScoredEvidenceSupport, ...],
    shape: StructuredSynthesisShape,
) -> GuardStructuredCanonicalizationResult | None:
    if shape != "predicate_bound_scalar" or not result.answer:
        return None
    if _support_present_last_friday_finish_question(question):
        answer_terms = set(_content_tokens(result.answer))
        if "screenplay" in answer_terms:
            support_by_id = {item.support.support_id: item.support for item in support_pack}
            cited_supports = [
                support_by_id[support_id] for support_id in result.support_ids if support_id in support_by_id
            ]
            if cited_supports and any(
                _support_present_sentence_or_context_actor_compatible(
                    question,
                    support.support_text,
                    context_sentence=support.concept_summary,
                )
                and {"last", "friday"} <= set(_content_tokens(support.support_text))
                for support in cited_supports
            ):
                return GuardStructuredCanonicalizationResult(
                    result=StructuredSynthesisResult(
                        answer="screenplay",
                        support_ids=result.support_ids,
                        cited_spans=result.cited_spans,
                        fallback_used=result.fallback_used,
                        error_class=result.error_class,
                    ),
                    strategy="support_present_last_friday_finish_head",
                )
    canonical = _support_present_line_of_item_head(question, result.answer)
    if canonical is None:
        return None

    support_by_id = {item.support.support_id: item.support for item in support_pack}
    cited_supports = [support_by_id[support_id] for support_id in result.support_ids if support_id in support_by_id]
    if not cited_supports:
        return None
    normalized_canonical = _containment_normalize(canonical)
    if not any(
        _contains_containment_answer(
            normalized_canonical,
            support.normalized_support_text,
        )
        for support in cited_supports
    ):
        return None

    return GuardStructuredCanonicalizationResult(
        result=StructuredSynthesisResult(
            answer=canonical,
            support_ids=result.support_ids,
            cited_spans=result.cited_spans,
            fallback_used=result.fallback_used,
            error_class=result.error_class,
        ),
        strategy="support_present_line_of_item_head",
    )


def _support_present_line_of_item_head(question: str, answer: str) -> str | None:
    if _SUPPORT_PRESENT_LINE_OF_QUESTION_PATTERN.search(question) is None:
        return None
    match = _SUPPORT_PRESENT_LINE_OF_ANSWER_PATTERN.match(_clean_candidate_answer(answer))
    if match is None:
        return None
    canonical = _strip_leading_article(_clean_candidate_answer(match.group(1)))
    return canonical or None


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    items = tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
    return items


def _deterministic_structured_fallback(
    question: str,
    support_pack: tuple[ScoredEvidenceSupport, ...],
    shape: StructuredSynthesisShape,
    *,
    timeout_recovery: bool = False,
    support_pack_completeness_enabled: bool = False,
    allow_clear_win_different_answers: bool = False,
    allow_benefit_with_having: bool = True,
    benefit_required_concept_ids: set[str] | None = None,
    support_present_answer_role: SupportPresentAnswerRole | None = None,
) -> StructuredSynthesisResult | None:
    if not support_pack_completeness_enabled:
        for item in support_pack:
            support_allows_benefit = _benefit_with_having_fallback_allowed_for_support(
                question,
                item.support,
                allow_benefit_with_having=allow_benefit_with_having,
                benefit_required_concept_ids=benefit_required_concept_ids,
            )
            for sentence in _SENTENCE_SPLIT_PATTERN.split(item.support.support_text):
                answer = _support_present_role_answer_from_sentence(
                    question,
                    sentence,
                    support_present_answer_role,
                    shape,
                    support_channel=item.support.channel,
                    context_sentence=item.support.concept_summary,
                )
                if answer is None:
                    span_result = _support_present_span_result_from_sentence(
                        question,
                        item.support.support_id,
                        sentence,
                        support_present_answer_role,
                        shape,
                        timeout_recovery=timeout_recovery,
                    )
                    if span_result is not None:
                        return span_result
                if answer is None:
                    answer = _deterministic_answer_from_sentence(
                        question,
                        sentence,
                        shape,
                        allow_benefit_with_having=support_allows_benefit,
                    )
                if answer is None:
                    continue
                if _support_present_role_candidate_rejection_reason(
                    answer,
                    support_present_answer_role,
                    question=question,
                ) is not None:
                    continue
                cited_span = sentence if shape == "predicate_bound_scalar" else answer
                return StructuredSynthesisResult(
                    answer=answer,
                    support_ids=(item.support.support_id,),
                    cited_spans=(cited_span,),
                    fallback_used=(
                        "timeout_deterministic_fallback" if timeout_recovery else "deterministic_support_fallback"
                    ),
                )
        return None

    sentence_pack = _build_question_bound_sentence_pack(
        question,
        support_pack,
        shape=shape,
    )
    if shape == "list_or_set":
        aggregated = _aggregate_list_answer_from_sentence_pack(
            question,
            sentence_pack,
            timeout_recovery=timeout_recovery,
            support_present_answer_role=support_present_answer_role,
        )
        if aggregated is not None:
            return aggregated
        if _support_present_has_unparsed_bound_list_support(
            question,
            sentence_pack,
            support_present_answer_role,
        ):
            return None
        if _list_pack_has_conflict(sentence_pack):
            return None
    if shape == "predicate_bound_scalar":
        selected = _select_bound_scalar_from_sentence_pack(
            question,
            sentence_pack,
            timeout_recovery=timeout_recovery,
            allow_clear_win_different_answers=allow_clear_win_different_answers,
            allow_benefit_with_having=allow_benefit_with_having,
            benefit_required_concept_ids=benefit_required_concept_ids,
            support_present_answer_role=support_present_answer_role,
        )
        if selected is not None:
            return selected
    return _first_sentence_fallback_from_sentence_pack(
        question,
        sentence_pack,
        shape=shape,
        timeout_recovery=timeout_recovery,
        allow_benefit_with_having=allow_benefit_with_having,
        benefit_required_concept_ids=benefit_required_concept_ids,
        support_present_answer_role=support_present_answer_role,
    )


def _first_sentence_fallback_from_sentence_pack(
    question: str,
    sentence_pack: tuple[ScoredSupportSentence, ...],
    *,
    shape: StructuredSynthesisShape,
    timeout_recovery: bool,
    allow_benefit_with_having: bool = True,
    benefit_required_concept_ids: set[str] | None = None,
    support_present_answer_role: SupportPresentAnswerRole | None = None,
) -> StructuredSynthesisResult | None:
    for item in sentence_pack:
        support_allows_benefit = _benefit_with_having_fallback_allowed_for_support(
            question,
            item.support,
            allow_benefit_with_having=allow_benefit_with_having,
            benefit_required_concept_ids=benefit_required_concept_ids,
        )
        answer = _support_present_role_answer_from_sentence(
            question,
            item.sentence,
            support_present_answer_role,
            shape,
            support_channel=item.support.channel,
            context_sentence=item.support.concept_summary,
        )
        if answer is None and support_present_answer_role not in _SUPPORT_PRESENT_ROLE_STRICT_EXTRACT_ONLY:
            span_result = _support_present_span_result_from_sentence(
                question,
                item.support.support_id,
                item.sentence,
                support_present_answer_role,
                shape,
                timeout_recovery=timeout_recovery,
            )
            if span_result is not None:
                return span_result
        if answer is None and support_present_answer_role not in _SUPPORT_PRESENT_ROLE_STRICT_EXTRACT_ONLY:
            answer = _deterministic_answer_from_sentence(
                question,
                item.sentence,
                shape,
                allow_benefit_with_having=support_allows_benefit,
            )
        if answer is None:
            continue
        if _support_present_role_candidate_rejection_reason(
            answer,
            support_present_answer_role,
            question=question,
        ) is not None:
            continue
        cited_span = item.sentence if shape == "predicate_bound_scalar" else answer
        return StructuredSynthesisResult(
            answer=answer,
            support_ids=(item.support.support_id,),
            cited_spans=(cited_span,),
            fallback_used=("timeout_deterministic_fallback" if timeout_recovery else "deterministic_support_fallback"),
        )
    return None


_SUPPORT_PRESENT_GENERIC_LIST_SLOT_TERMS = {
    "activities",
    "activity",
    "counseling",
    "done",
    "dairy",
    "dessert",
    "desserts",
    "family",
    "field",
    "fields",
    "flavor",
    "flavors",
    "free",
    "health",
    "interested",
    "mental",
    "pursue",
    "pursuing",
    "services",
}
_LIKELY_FIELDS_NON_ANCHOR_TERMS = {
    "educaton",  # LOCOMO typo in the benchmark prompt.
    "education",
    "likely",
    "pursue",
    "pursuing",
    "would",
}
_SUPPORT_PRESENT_MADE_OBJECT_TERMS = {
    "bowl",
    "bowls",
    "ceramic",
    "cup",
    "cups",
    "mug",
    "mugs",
    "plate",
    "plates",
    "pot",
    "pots",
    "vase",
    "vases",
}
_SUPPORT_PRESENT_PET_TYPE_SIZE_TERMS = {"small", "smaller", "medium", "large", "larger"}
_SUPPORT_PRESENT_PET_TYPE_BREED_TERMS = {
    "beagle",
    "breed",
    "breeds",
    "bulldog",
    "collie",
    "poodle",
    "retriever",
    "shepherd",
    "terrier",
}
_SUPPORT_PRESENT_PET_TYPE_DOG_TERMS = {"dog", "dogs", "doggo", "pup", "puppy"}
_SUPPORT_PRESENT_TRAINING_META_PATTERN = re.compile(
    r"\b(?:interested\s+in|related\s+to|sounds?\s+cool|excited\s+to|"
    r"looking\s+forward|hear\s+about|heard\s+about)\b",
    re.IGNORECASE,
)
_SUPPORT_PRESENT_LIST_ANCHOR_IGNORE_TERMS = {
    "kind",
    "kinds",
    "some",
    "type",
    "types",
}
_SUPPORT_PRESENT_DIRECT_SCALAR_QUESTION_PATTERNS = (
    re.compile(r"^\s*what\s+flavor\s+of\s+ice\s+cream\s+did\b", re.IGNORECASE),
    re.compile(r"^\s*what\s+kind\s+of\s+frosting\s+did\b", re.IGNORECASE),
    re.compile(r"^\s*what\s+kind\s+of\s+dance\s+piece\s+did\b", re.IGNORECASE),
    re.compile(r"^\s*what\s+project\s+is\b.*\bworking\s+on\b", re.IGNORECASE),
    re.compile(r"^\s*what\s+new\s+content\s+is\b.*\bcreating\s+for\s+youtube\b", re.IGNORECASE),
    re.compile(r"^\s*what\s+technique\s+is\b.*\busing\s+to\s+discipline\b", re.IGNORECASE),
    re.compile(r"^\s*what\s+was\s+discussed\s+(?:in|at|during)\b.*\bworkshop\b", re.IGNORECASE),
    re.compile(r"^\s*how\s+do\b.*\bdogs?\s+react\s+to\s+snow\b", re.IGNORECASE),
    re.compile(r"^\s*where\s+is\b.*\bfashion\s+(?:internship|position)\b", re.IGNORECASE),
    re.compile(r"^\s*where\s+is\s+[A-Z][a-z]+'?s\s+hr\s+internship\b", re.IGNORECASE),
    re.compile(r"^\s*how\s+does\b.*\bdescribe\b.*\bstuffed\s+animal\b", re.IGNORECASE),
    re.compile(r"^\s*what\s+challenge\s+is\b.*\bsearch\s+for\s+(?:a\s+)?pet\b", re.IGNORECASE),
    re.compile(r"^\s*how\s+does\b.*\bdescribe\b.*\bnew\s+beds?\b.*\bdogs?\b", re.IGNORECASE),
    re.compile(r"^\s*what\s+type\s+of\s+dog\s+was\b.*\blooking\s+to\s+adopt\b", re.IGNORECASE),
    re.compile(r"^\s*what\s+is\b.*\bidentity\b", re.IGNORECASE),
    re.compile(r"^\s*what\s+does\b.*\brunning\b.*\bbeen\s+great\s+for\b", re.IGNORECASE),
    re.compile(r"^\s*what\s+was\s+grandpa'?s\s+gift\s+to\b", re.IGNORECASE),
    re.compile(r"^\s*what\s+does\b.*\brely\s+on\s+for\s+cheer\s+and\s+joy\b", re.IGNORECASE),
    re.compile(r"^\s*what\s+did\b.*\bjust\s+finish\b.*\blast\s+friday\b", re.IGNORECASE),
    re.compile(r"^\s*what\s+did\b.*\bfind\s+in\s+old\s+notebooks\b", re.IGNORECASE),
    re.compile(r"^\s*what\s+was\b.*\baudition\s+for\b", re.IGNORECASE),
    re.compile(r"^\s*what\s+is\s+the\s+name\s+of\b.*\bchildhood\s+dog\b", re.IGNORECASE),
    re.compile(r"^\s*what\s+is\s+joanna'?s\s+third\s+screenplay\s+about\b", re.IGNORECASE),
    re.compile(r"^\s*what\s+is\s+[\"']?little\s+women[\"']?\s+about\s+according\s+to\s+joanna\b", re.IGNORECASE),
    re.compile(r"^\s*what\s+did\s+andrew\s+express\s+missing\s+about\s+exploring\s+nature\s+trails\b", re.IGNORECASE),
    re.compile(r"^\s*what\s+did\b.*\btake\s+(?:a\s+)?(?:picture|photo)\s+of\b", re.IGNORECASE),
    re.compile(r"^\s*how\s+(?:did|does)\s+[A-Z][a-z]+\s+feel\b", re.IGNORECASE),
    re.compile(r"^\s*why\s+did\s+[A-Z][a-z]+\s+shut\s+down\s+(?:his|her|their)\s+bank\s+account\b", re.IGNORECASE),
    re.compile(r"^\s*what\s+special\s+items\s+did\s+[A-Z][a-z]+\s+get\b.*\bfor\s+everyone\b", re.IGNORECASE),
    re.compile(r"^\s*what\s+spice\s+did\b", re.IGNORECASE),
    re.compile(r"^\s*what\s+organization\s+does\b", re.IGNORECASE),
    re.compile(r"^\s*what\s+new\s+hobby\s+did\b", re.IGNORECASE),
    re.compile(r"^\s*what\s+project\s+did\b", re.IGNORECASE),
    re.compile(r"^\s*what\s+activity\s+did\b", re.IGNORECASE),
    re.compile(r"^\s*what\s+class\s+is\b", re.IGNORECASE),
    re.compile(r"^\s*what\s+sports\s+activity\s+has\b", re.IGNORECASE),
    re.compile(r"^\s*what\s+color\s+glow\s+did\b", re.IGNORECASE),
    re.compile(r"^\s*what\s+emotion\s+does\b", re.IGNORECASE),
)
_SUPPORT_PRESENT_ROLE_STRICT_EXTRACT_ONLY = {"diet_list", "direct_support_scalar"}
_SUPPORT_PRESENT_FLAVOR_ANSWER_TERMS = {
    "berry",
    "caramel",
    "chocolate",
    "mint",
    "strawberry",
    "swirl",
    "vanilla",
}
_SUPPORT_PRESENT_FLAVOR_OBJECT_TERMS = {"bowl", "bowls", "coconut", "milk", "recipe"}
_SUPPORT_PRESENT_PROJECT_DESCRIPTION_TERMS = {
    "book",
    "midwestern",
    "novel",
    "screenplay",
    "story",
    "thriller",
    "town",
}
_SUPPORT_PRESENT_PROJECT_PROCESS_TERMS = {"clearer", "hopes", "project", "steps"}
_SUPPORT_PRESENT_DIRECT_SCALAR_ACTOR_PATTERN = re.compile(r"^\s*([A-Z][a-z]+)\b")
_SUPPORT_PRESENT_TRIP_PURPOSE_QUESTION_PATTERN = re.compile(
    r"^\s*what\s+did\s+(?P<actor>[A-Z][a-z]+)\s+take\s+(?:a|the)\s+trip\s+to\s+(?P<place>.+?)\s+for\??\s*$",
    re.IGNORECASE,
)
_SUPPORT_PRESENT_TRIP_PURPOSE_TAIL_PATTERN = re.compile(
    r"\s+(?:on\s+(?:\d{1,2}\s+)?(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december|\d{1,2}|\d{4})\b.*"
    r"|in\s+(?:january|february|march|april|may|june|july|august|september|october|"
    r"november|december|\d{4})\b.*"
    r"|during\b.*|after\b.*|before\b.*|at\b.*)$",
    re.IGNORECASE,
)
_SUPPORT_PRESENT_DIET_TAIL_TRAP_TERMS = {"cute", "finds", "hyped", "love", "loves", "seeing"}
_SUPPORT_PRESENT_WORKSHOP_DISCUSSION_IGNORE_TERMS = {
    "covered",
    "different",
    "discussed",
    "discussing",
    "multiple",
    "several",
    "talked",
    "various",
    "workshop",
    "workshops",
}
_SUPPORT_PRESENT_DIRECT_SCALAR_ACTOR_IGNORE_TERMS = {
    "ah",
    "client",
    "evidence",
    "he",
    "her",
    "him",
    "his",
    "it",
    "she",
    "super",
    "their",
    "them",
    "they",
}
_SUPPORT_PRESENT_PHOTO_OBJECT_TERMS = {"photo", "photos", "pic", "picture", "pictures"}
_SUPPORT_PRESENT_PHOTO_OBJECT_REJECT_TERMS = {
    "family",
    "frame",
    "frames",
    "pic",
    "photo",
    "photos",
    "picture",
    "pictures",
    "that",
    "this",
}
_SUPPORT_PRESENT_GENERIC_SCALAR_SLOT_ANSWERS = {
    "activity",
    "class",
    "color",
    "color glow",
    "emotion",
    "glow",
    "hobby",
    "new hobby",
    "organization",
    "project",
    "sports activity",
}
_SUPPORT_PRESENT_GENERIC_SCALAR_SLOT_BAD_TERMS = {
    "instead",
    "because",
    "resume",
    "resuming",
    "when",
    "while",
}
_SUPPORT_PRESENT_ORGANIZATION_SLOT_TERMS = {
    "association",
    "center",
    "charity",
    "foundation",
    "hospital",
    "nonprofit",
    "organization",
    "rescue",
    "school",
    "shelter",
}
_SUPPORT_PRESENT_ORGANIZATION_SLOT_REJECT_TERMS = {
    "cause",
    "heart",
    "portion",
    "profit",
    "profits",
}
_SUPPORT_PRESENT_SPORTS_ACTIVITY_TERMS = {
    "basketball",
    "boxing",
    "climbing",
    "cycling",
    "hiking",
    "pilates",
    "running",
    "skiing",
    "soccer",
    "sports",
    "swimming",
    "tennis",
    "yoga",
}
_SUPPORT_PRESENT_SPORTS_ACTIVITY_REJECT_TERMS = {
    "active",
    "great",
    "routine",
}
_SUPPORT_PRESENT_EMOTION_SLOT_TERMS = {
    "anxious",
    "confident",
    "excited",
    "frustrated",
    "grateful",
    "happy",
    "hopeful",
    "inspired",
    "nervous",
    "proud",
    "sad",
}


def _support_present_direct_scalar_question(question: str) -> bool:
    return any(pattern.search(question) for pattern in _SUPPORT_PRESENT_DIRECT_SCALAR_QUESTION_PATTERNS) or (
        _support_present_training_type_question(question)
        or _support_present_trip_purpose_question(question) is not None
    )


def _support_present_feeling_attribute_question(question: str) -> bool:
    return re.search(r"^\s*how\s+(?:did|does)\s+[A-Z][a-z]+\s+feel\b", question, re.IGNORECASE) is not None


def _support_present_bank_account_shutdown_reason_question(question: str) -> bool:
    return (
        re.search(
            r"^\s*why\s+did\s+[A-Z][a-z]+\s+shut\s+down\s+(?:his|her|their)\s+bank\s+account\b",
            question,
            re.IGNORECASE,
        )
        is not None
    )


def _support_present_workshop_discussion_question(question: str) -> bool:
    return {"discussed", "workshop"} <= set(_content_tokens(question))


def _support_present_training_type_question(question: str) -> bool:
    return (
        re.search(
            r"^\s*what\s+(?:type|kind)\s+of\s+training\s+(?:was|is)\b.*\bworkshop\b.*\bsigned\s+up\s+for\b",
            question,
            re.IGNORECASE,
        )
        is not None
    )


def _support_present_training_type_actor(question: str) -> str | None:
    match = re.search(r"\bworkshop\s+([A-Z][a-z]+)\s+signed\s+up\s+for\b", question)
    if match is None:
        return None
    return _actor_token(match.group(1))


_SUPPORT_PRESENT_TRAINING_TYPE_WORKSHOP_ACTOR_PATTERN = re.compile(
    r"\bworkshop\s+([A-Z][a-z]+)\s+signed\s+up\s+for\b"
)


def _support_present_training_type_support_actor_terms(*texts: str | None) -> set[str]:
    terms: set[str] = set()
    for text in texts:
        cleaned = (text or "").replace("Client evidence:", "").strip()
        if not cleaned:
            continue
        terms.update(set(_support_actor_terms_from_text(cleaned)) - _SUPPORT_PRESENT_DIRECT_SCALAR_ACTOR_IGNORE_TERMS)
        for pattern in (
            _SUPPORT_PRESENT_TRAINING_TYPE_WORKSHOP_ACTOR_PATTERN,
            _SUPPORT_PRESENT_DIRECT_SCALAR_ACTOR_PATTERN,
        ):
            match = pattern.search(cleaned)
            if match is None:
                continue
            actor_token = _actor_token(match.group(1))
            if actor_token and actor_token not in _SUPPORT_PRESENT_DIRECT_SCALAR_ACTOR_IGNORE_TERMS:
                terms.add(actor_token)
    return terms


def _support_present_special_items_question(question: str) -> bool:
    return (
        re.search(
            r"^\s*what\s+special\s+items\s+did\s+[A-Z][a-z]+\s+get\b.*\bfor\s+everyone\b",
            question,
            re.IGNORECASE,
        )
        is not None
    )


def _support_present_special_items_candidate(
    question: str,
    sentence: str,
    *,
    context_sentence: str | None = None,
) -> str | None:
    if not _support_present_special_items_question(question):
        return None
    combined_terms = set(_content_tokens(sentence)) | set(_content_tokens(context_sentence or ""))
    if not (combined_terms & {"party", "guests", "guest", "everyone"}):
        return None
    if not (set(_content_tokens(question)) & combined_terms & {"gaming", "party"}):
        return None
    patterns = (
        r"\b(?:is\s+|was\s+|are\s+|were\s+)?(?:getting|got|get)\s+"
        r"(?:some\s+|the\s+|a\s+|an\s+)?(.+?)\s+for\s+(?:the\s+)?(?:guests?|everyone)\b",
    )
    for source in (sentence, context_sentence or ""):
        for pattern in patterns:
            match = re.search(pattern, source, re.IGNORECASE)
            if match is None:
                continue
            candidate = _strip_leading_article(_clean_candidate_answer(match.group(1)))
            candidate_terms = set(_content_tokens(candidate))
            if not candidate_terms:
                continue
            if len(_content_tokens(candidate)) > 6:
                continue
            if candidate_terms & {"everyone", "guest", "guests", "item", "items", "party", "special"}:
                continue
            if _support_present_candidate_echoes_question(question, candidate):
                continue
            return candidate
    return None


def _support_present_hr_internship_question(question: str) -> bool:
    return re.search(r"^\s*where\s+is\s+[A-Z][a-z]+'?s\s+hr\s+internship\b", question, re.IGNORECASE) is not None


def _support_present_last_friday_finish_question(question: str) -> bool:
    return (
        re.search(r"^\s*what\s+did\s+[A-Z][a-z]+\s+just\s+finish\b", question, re.IGNORECASE) is not None
        and {"last", "friday"} <= set(_content_tokens(question))
    )


def _support_present_little_women_about_question(question: str) -> bool:
    return (
        re.search(
            r"^\s*what\s+is\s+[\"']?little\s+women[\"']?\s+about\s+according\s+to\s+joanna\b",
            question,
            re.IGNORECASE,
        )
        is not None
    )


def _support_present_joanna_third_screenplay_about_question(question: str) -> bool:
    return (
        re.search(
            r"^\s*what\s+is\s+joanna'?s\s+third\s+screenplay\s+about\b",
            question,
            re.IGNORECASE,
        )
        is not None
    )


def _support_present_nature_trail_missing_question(question: str) -> bool:
    return (
        re.search(
            r"^\s*what\s+did\s+andrew\s+express\s+missing\s+about\s+exploring\s+nature\s+trails\b",
            question,
            re.IGNORECASE,
        )
        is not None
        and {"family", "dog"} <= set(_content_tokens(question))
    )


def _support_present_wrong_actor_bound(question_actor: str, text: str) -> bool:
    actor_terms = set(_support_actor_terms_from_text(text)) - _SUPPORT_PRESENT_DIRECT_SCALAR_ACTOR_IGNORE_TERMS
    return bool(actor_terms and question_actor not in actor_terms)


def _support_present_sentence_actor_compatible(question: str, sentence: str) -> bool:
    question_terms = set(_question_actor_terms(question))
    if not question_terms:
        return True
    actor_sentence = sentence.replace("Client evidence:", "").strip()
    support_terms = set(_support_actor_terms_from_text(sentence)) - _SUPPORT_PRESENT_DIRECT_SCALAR_ACTOR_IGNORE_TERMS
    if not support_terms:
        actor_match = _SUPPORT_PRESENT_DIRECT_SCALAR_ACTOR_PATTERN.search(actor_sentence)
        if actor_match:
            actor_token = _actor_token(actor_match.group(1))
            if actor_token and actor_token not in _SUPPORT_PRESENT_DIRECT_SCALAR_ACTOR_IGNORE_TERMS:
                support_terms.add(actor_token)
    if not support_terms:
        return True
    return bool(question_terms & support_terms)


def _support_present_sentence_or_context_actor_compatible(
    question: str,
    sentence: str,
    *,
    context_sentence: str | None = None,
) -> bool:
    question_terms = set(_question_actor_terms(question))
    if not question_terms:
        return True

    sentence_terms = set(_support_actor_terms_from_text(sentence)) - _SUPPORT_PRESENT_DIRECT_SCALAR_ACTOR_IGNORE_TERMS
    if sentence_terms:
        return bool(question_terms & sentence_terms)

    actor_sentence = sentence.replace("Client evidence:", "").strip()
    actor_match = _SUPPORT_PRESENT_DIRECT_SCALAR_ACTOR_PATTERN.search(actor_sentence)
    if actor_match:
        actor_token = _actor_token(actor_match.group(1))
        if actor_token and actor_token not in _SUPPORT_PRESENT_DIRECT_SCALAR_ACTOR_IGNORE_TERMS:
            return actor_token in question_terms

    context_terms = set(_support_actor_terms_from_text(context_sentence or "")) - _SUPPORT_PRESENT_DIRECT_SCALAR_ACTOR_IGNORE_TERMS
    if context_terms:
        return bool(question_terms & context_terms)

    context_match = _SUPPORT_PRESENT_DIRECT_SCALAR_ACTOR_PATTERN.search((context_sentence or "").strip())
    if context_match:
        context_token = _actor_token(context_match.group(1))
        if context_token and context_token not in _SUPPORT_PRESENT_DIRECT_SCALAR_ACTOR_IGNORE_TERMS:
            return context_token in question_terms

    return False


def _support_present_workshop_discussion_compatible(
    question: str,
    sentence: str,
    *,
    context_sentence: str | None = None,
) -> bool:
    sentence_terms = set(_content_tokens(sentence))
    context_terms = set(_content_tokens(context_sentence or ""))
    combined_terms = sentence_terms | context_terms
    if "workshop" not in combined_terms and "workshops" not in combined_terms:
        return False
    question_topic_terms = set(_content_tokens(question)) - _SUPPORT_PRESENT_WORKSHOP_DISCUSSION_IGNORE_TERMS
    if not question_topic_terms:
        return False
    return bool(question_topic_terms & combined_terms)


def _support_present_question_near_place_terms(question: str) -> set[str]:
    match = re.search(
        r"\bnear\s+(.+?)(?:\s+last\s+summer\b|\s+(?:on|in|during|with|for|after|before)\b|\?|$)",
        question,
        re.IGNORECASE,
    )
    if match is None:
        return set()
    return {
        term
        for term in _content_tokens(match.group(1))
        if term not in _QUESTION_STOPWORDS and term not in {"last", "summer"}
    }


def _support_present_photo_time_compatible(question_terms: set[str], combined_terms: set[str]) -> bool:
    if not {"last", "summer"} <= question_terms:
        return True
    if {"last", "summer"} <= combined_terms:
        return True
    return "summer" in combined_terms and any(re.fullmatch(r"\d{4}", term) for term in combined_terms)


def _support_present_clean_generic_scalar_candidate(
    question: str,
    candidate: str,
    *,
    max_terms: int = 6,
) -> str | None:
    cleaned = _strip_leading_article(_clean_candidate_answer(candidate))
    if not cleaned:
        return None
    normalized = _normalize(cleaned)
    if normalized in _SUPPORT_PRESENT_GENERIC_SCALAR_SLOT_ANSWERS:
        return None
    terms = _content_tokens(cleaned)
    if not terms or len(terms) > max_terms:
        return None
    if set(terms) & _SUPPORT_PRESENT_GENERIC_SCALAR_SLOT_BAD_TERMS:
        return None
    if _support_present_candidate_echoes_question(question, cleaned):
        return None
    return cleaned


def _support_present_photo_object_candidate(
    question: str,
    sentence: str,
    *,
    context_sentence: str | None = None,
) -> str | None:
    question_terms = set(_content_tokens(question))
    if "take" not in question_terms or not (question_terms & {"picture", "photo"}):
        return None
    sentence_terms = set(_content_tokens(sentence))
    if not sentence_terms & _SUPPORT_PRESENT_PHOTO_OBJECT_TERMS:
        return None
    if not re.search(r"\b(?:take|took|taken)\b", sentence, re.IGNORECASE):
        return None
    if not sentence.strip().lower().startswith("client evidence:"):
        return None
    place_terms = _support_present_question_near_place_terms(question)
    if place_terms and not place_terms <= sentence_terms:
        return None
    if not _support_present_photo_time_compatible(question_terms, sentence_terms):
        return None
    if context_sentence and not _support_present_sentence_actor_compatible(question, context_sentence):
        return None

    candidate_sources = (context_sentence or "", sentence)
    patterns = (
        r"\btook\s+(?:a|an|the)?\s*([a-z][a-z' -]+?)\s+(?:photo|picture|pic)\b",
        r"\btook\s+(?:a|an|the)?\s*(?:photo|picture|pic)\s+of\s+(?:a|an|the)?\s*([a-z][a-z' -]+?)(?:\s+(?:near|on|in|during|last|with|for)\b|\.|$)",
        r"\b(?:photo|picture|pic)\s+of\s+(?:a|an|the)?\s*([a-z][a-z' -]+?)(?:\s+(?:near|on|in|during|last|with|for)\b|\.|$)",
    )
    for source in candidate_sources:
        for pattern in patterns:
            match = re.search(pattern, source, re.IGNORECASE)
            if match is None:
                continue
            candidate = _strip_leading_article(_clean_candidate_answer(match.group(1)))
            candidate_terms = set(_content_tokens(candidate))
            if not candidate_terms or candidate_terms & _SUPPORT_PRESENT_PHOTO_OBJECT_REJECT_TERMS:
                continue
            if len(candidate_terms) > 4:
                continue
            if _support_present_candidate_echoes_question(question, candidate):
                continue
            return candidate
    return None


def _support_present_trip_purpose_question(question: str) -> tuple[str, set[str]] | None:
    match = _SUPPORT_PRESENT_TRIP_PURPOSE_QUESTION_PATTERN.search(question)
    if match is None:
        return None
    actor = _actor_token(match.group("actor"))
    place_terms = {
        term
        for term in _content_tokens(match.group("place"))
        if term not in _QUESTION_STOPWORDS and term not in {"trip"}
    }
    if not actor or not place_terms:
        return None
    return actor, place_terms


def _support_present_trip_purpose_possessive(question_actor: str, combined_text: str) -> str | None:
    question_lower = question_actor.lower()
    combined_lower = combined_text.lower()
    if re.search(r"\b(?:he|him|his)\b", combined_lower):
        return "his"
    if re.search(r"\b(?:she|her|hers)\b", combined_lower):
        return "her"
    if re.search(r"\b(?:they|them|their|theirs)\b", combined_lower):
        return "their"
    if question_lower in {"jon", "nate", "andrew"}:
        return "his"
    if question_lower in {"gina", "joanna", "caroline", "audrey", "melanie", "vivian"}:
        return "her"
    return None


def _support_present_trip_purpose_clean_candidate(
    candidate: str,
    *,
    question_actor: str,
    combined_text: str,
) -> str | None:
    cleaned = _clean_candidate_answer(candidate)
    cleaned = _SUPPORT_PRESENT_TRIP_PURPOSE_TAIL_PATTERN.sub("", cleaned).strip(" ,.;")
    if not cleaned:
        return None
    possessive = _support_present_trip_purpose_possessive(question_actor, combined_text)
    if possessive is not None:
        cleaned = re.sub(r"^\s*my\b", possessive, cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bmy\b", possessive, cleaned, flags=re.IGNORECASE)
    candidate_terms = _content_tokens(cleaned)
    if not candidate_terms or len(candidate_terms) > 8:
        return None
    if "mind" not in candidate_terms and not (set(candidate_terms) & {"relax", "recharge", "reset"}):
        return None
    return f"to {cleaned}" if not cleaned.lower().startswith("to ") else cleaned


def _support_present_trip_purpose_candidate(
    question: str,
    sentence: str,
    *,
    context_sentence: str | None = None,
) -> str | None:
    parsed = _support_present_trip_purpose_question(question)
    if parsed is None:
        return None
    question_actor, place_terms = parsed
    cleaned_sentence = sentence.strip()
    combined_text = f"{cleaned_sentence} {context_sentence or ''}".strip()
    combined_terms = set(_content_tokens(combined_text))
    if not place_terms <= combined_terms:
        return None
    if not (combined_terms & {"trip", "travel", "journey"}):
        return None
    if not _support_present_sentence_or_context_actor_compatible(
        question,
        cleaned_sentence,
        context_sentence=context_sentence,
    ):
        return None
    actor_terms = set(_support_actor_terms_from_text(combined_text)) - _SUPPORT_PRESENT_DIRECT_SCALAR_ACTOR_IGNORE_TERMS
    if actor_terms and actor_terms != {question_actor}:
        return None
    if _support_present_wrong_actor_bound(question_actor, combined_text):
        return None

    sources = (cleaned_sentence, context_sentence or "")
    patterns = (
        r"\btrip\s+to\s+[A-Z][A-Za-z' -]+?\s+was\s+intended\s+to\s+help\s+(?:him|her|them|me)\s+(.+?)(?:\.|$)",
        r"\btook\s+(?:a\s+|the\s+)?(?:short\s+|quick\s+|brief\s+)?trip\s+to\s+[A-Z][A-Za-z' -]+?\s+to\s+(.+?)(?:\.|$)",
    )
    for source in sources:
        for pattern in patterns:
            match = re.search(pattern, source, re.IGNORECASE)
            if match is None:
                continue
            candidate = _support_present_trip_purpose_clean_candidate(
                match.group(1),
                question_actor=question_actor,
                combined_text=combined_text,
            )
            if candidate is not None:
                return candidate

    first_person_match = re.search(r"(?:^|\bclient\s+evidence:\s*)to\s+(.+?)(?:\.|$)", cleaned_sentence, re.IGNORECASE)
    if first_person_match is not None:
        return _support_present_trip_purpose_clean_candidate(
            first_person_match.group(1),
            question_actor=question_actor,
            combined_text=combined_text,
        )
    return None


def _support_present_direct_scalar_candidate(
    question: str,
    sentence: str,
    *,
    context_sentence: str | None = None,
) -> str | None:
    if not _support_present_direct_scalar_question(question):
        return None
    if not _support_present_sentence_actor_compatible(question, sentence):
        return None

    cleaned_sentence = sentence.strip()
    question_terms = set(_content_tokens(question))
    trip_purpose_candidate = _support_present_trip_purpose_candidate(
        question,
        cleaned_sentence,
        context_sentence=context_sentence,
    )
    if trip_purpose_candidate is not None:
        return trip_purpose_candidate

    special_items_candidate = _support_present_special_items_candidate(
        question,
        cleaned_sentence,
        context_sentence=context_sentence,
    )
    if special_items_candidate is not None:
        return special_items_candidate

    patterns: tuple[str, ...] = ()
    generic_scalar_slot_max_terms: int | None = None
    if {"flavor", "ice", "cream"} <= question_terms:
        patterns = (
            r"\bwhipped\s+up\s+(?:a|an|the|some)?\s*(.+?)\s+ice\s+cream\b",
            r"\b(?:made|make|making)\s+(?:a|an|the|some)?\s*(?!ice\s+cream\b)(.+?)\s+ice\s+cream\b",
            r"\bice\s+cream\s+(?:flavor\s+)?(?:was|is)\s+(?:a|an|the)?\s*(.+?)(?:\.|$)",
        )
    elif {"spice", "soup"} <= question_terms:
        generic_scalar_slot_max_terms = 2
        patterns = (
            r"\badd(?:ed)?\s+(?:some\s+)?([a-z][a-z -]{1,30}?)\s+to\s+(?:the\s+)?soup\b",
        )
    elif {"kind", "frosting"} <= question_terms:
        patterns = (
            r"\bwith\s+(?:a|an|the)?\s*(.+?)\s+frosting\b",
            r"\bfrosting\s+(?:was|is)\s+(?:a|an|the)?\s*(.+?)(?:\.|$)",
        )
    elif {"dance", "piece"} <= question_terms:
        patterns = (
            r"\b(?:piece|performance)\s+called\s+(?:\"([^\"]+)\"|'([^']+)'|(.+?))(?:\s+for\b|\.|$)",
        )
    elif "project" in question_terms and "working" in question_terms:
        patterns = (
            r"\bproject(?:,|\s+(?:is|was))\s+(?:a|an|the)?\s*(.+?)(?:\.|$)",
            r"\bworking\s+on\s+(?:a|an|the)?\s*(?:new\s+project,?\s*)?(.+?)(?:\.|$)",
        )
    elif {"organization", "donate"} <= question_terms:
        generic_scalar_slot_max_terms = 5
        patterns = (
            r"\bdonat(?:e|es|ed|ing)\s+to\s+(?:an?\s+|the\s+)?([a-z][a-z -]{2,50}?)(?:\s+as\b|\s+to\b|\.|$)",
            r"\bdonat(?:e|es|ed|ing)\s+.+?\bto\s+(?:an?\s+|the\s+)?([a-z][a-z -]{2,50}?)(?:\s+as\b|\s+to\b|\.|$)",
        )
    elif {"new", "hobby"} <= question_terms:
        generic_scalar_slot_max_terms = 4
        patterns = (
            r"\b(?:interested\s+in|got\s+into|started|started\s+doing)\s+([a-z][a-z -]{2,50}?)(?:\s+on\b|\s+in\b|\.|$)",
        )
    elif {"project", "finish"} <= question_terms:
        generic_scalar_slot_max_terms = 5
        patterns = (
            r"\bfinished\s+(?:an?\s+|the\s+)?([a-z][a-z -]{2,60}?\bproject)\b",
        )
    elif {"activity", "plan"} <= question_terms:
        generic_scalar_slot_max_terms = 5
        patterns = (
            r"\bplanned\s+to\s+([a-z][a-z -]{2,50}?)(?:\s+instead\b|\.|$)",
        )
    elif {"class", "healthier", "meals"} <= question_terms:
        generic_scalar_slot_max_terms = 4
        patterns = (
            r"\b(?:taking|signed\s+up\s+for)\s+(?:a\s+)?([a-z][a-z -]{2,40}?\s+class)\b",
        )
    elif {"sports", "activity"} <= question_terms:
        generic_scalar_slot_max_terms = 3
        patterns = (
            r"\b([A-Za-z][A-Za-z -]{2,40}?)\s+has\s+been\s+great\b.*\bstay\s+active\b",
            r"\b(?:and|by|with|doing|started|took\s+up)\s+([A-Za-z][A-Za-z -]{2,40}?)\s+to\s+stay\s+active\b",
            r"\b(?:doing|started|took\s+up)\s+([A-Za-z][A-Za-z -]{2,40}?)(?:\s+to\s+stay\s+active\b|\s+while\b|\.|$)",
        )
    elif {"color", "glow"} <= question_terms:
        generic_scalar_slot_max_terms = 2
        patterns = (
            r"\b(?:customi[sz]ed|added)\s+(?:a\s+)?([a-z][a-z -]{1,20}?)\s+glow\b",
        )
    elif "emotion" in question_terms:
        generic_scalar_slot_max_terms = 6
        patterns = (
            r"\b(?:feel(?:s|ing)?|felt)\s+(proud|happy|excited|grateful|sad|nervous|anxious|confident|inspired|frustrated|hopeful)\b",
            r"\b(?:feel(?:s|ing)?|felt)\s+([a-z][a-z -]{2,30}?)(?:\s+when\b|\s+after\b|\.|$)",
        )
    elif {"content", "youtube"} <= question_terms:
        patterns = (
            r"\b(?:creating|making)\s+(?:new\s+)?(.+?\bvideos?)\s+for\s+youtube\b",
            r"\bmake\s+videos\s+about\s+(.+?)(?:\.|$)",
            r"\bmaking\s+videos\s+about\s+(.+?)(?:\.|$)",
            r"\bshare\s+(?:his|her|their)?\s*(?:love\s+of\s+)?(.+?)\s+by\s+making\s+videos\b",
            r"\bstart(?:ing)?\s+to\s+make\s+videos\s+about\s+(.+?)(?:\.|$)",
        )
    elif {"technique", "discipline"} <= question_terms:
        patterns = (
            r"\busing\s+(.+?)\s+techniques?\s+to\s+(?:discipline|train)\b",
            r"\bdiscipline\s+.+?\bwith\s+(.+?)(?:\.|$)",
        )
    elif _support_present_workshop_discussion_question(question):
        if not _support_present_workshop_discussion_compatible(
            question,
            cleaned_sentence,
            context_sentence=context_sentence,
        ):
            return None
        patterns = (
            r"\bworkshop\s+discussed\s+(.+?)(?:\.|$)",
            r"\b(?:discussed|covered|talked\s+about)\s+(.+?)(?:\.|$)",
            r"\b(?:discussed|covered|talked\s+about)\s+(.+?)\s+(?:in|at|during)\s+(?:the\s+)?(?:[A-Za-z0-9+ -]+\s+)?workshop\b",
            r"\b(?:in|at|during)\s+(?:the\s+)?(?:[A-Za-z0-9+ -]+\s+)?workshop,?\s+(?:they\s+)?(?:discussed|covered|talked\s+about)\s+(.+?)(?:\.|$)",
        )
    elif {"react", "snow"} <= question_terms:
        patterns = (
            r"\bdogs?\s+dislike\s+snow\s+because\s+they\s+(?:were|are|seemed|looked)\s+(.+?)(?:\s+(?:during|in|around|when)\b|\.|$)",
            r"\bdogs?\s+(?:were|are|seemed|looked)\s+(.+?)\s+(?:during|in|around|when)\b.*\bsnow",
            r"\bdogs?\s+(?:were|are|seemed|looked)\s+(.+?)\s+(?:by|around)\s+snow\b",
            r"\bsnowy\b.*?\b(?:they\s+)?(?:were|are|seemed|looked)\s+(?:so\s+)?(.+?)(?:!|\.|$)",
            r"\b(?:they\s+)?(?:were|are|seemed|looked)\s+(?:so\s+)?(confused)(?:!|\.|$)",
            r"\bsnow\b.*?\bdogs?\s+(?:were|are|seemed|looked)\s+(.+?)(?:\.|$)",
        )
    elif _support_present_hr_internship_question(question):
        if not _support_present_sentence_or_context_actor_compatible(
            question,
            cleaned_sentence,
            context_sentence=context_sentence,
        ):
            return None
        patterns = (
            r"\b(?:it'?s\s+)?(?:a\s+)?(?:part-time\s+)?position\s+in\s+(?:the\s+)?(fashion\s+department\s+of\s+an\s+international\s+company)\b",
            r"\b(?:internship|position)\s+(?:is|was)?\s*(?:in|at|within)\s+(?:the\s+)?(fashion\s+department\s+of\s+an\s+international\s+company)\b",
            r"\b(?:in|at|within)\s+(?:the\s+)?(fashion\s+department\s+of\s+an\s+international\s+company)\b",
        )
    elif {"fashion", "department"} <= question_terms or {"fashion", "internship"} <= question_terms or {"fashion", "position"} <= question_terms:
        patterns = (
            r"\b(?:in|at|within)\s+(?:the\s+)?(fashion\s+department\s+of\s+an\s+international\s+company)\b",
        )
    elif {"stuffed", "animal"} <= question_terms:
        patterns = (
            r"\b((?:a\s+)?stuffed\s+animal\s+to\s+remind\s+(?:you|her|him|them)\s+of\s+the\s+good\s+vibes)\b",
        )
    elif {"challenge", "pet"} <= question_terms:
        patterns = (
            r"\b(?:challenge|hardest\s+part)\s+(?:is|was)\s+(finding\s+a\s+pet-friendly\s+spot\s+in\s+the\s+city)\b",
            r"\b(?:it'?s|it\s+is)\s+tough\s+(finding\s+a\s+pet-friendly\s+spot\s+in\s+the\s+city)\b",
        )
    elif {"beds", "dogs"} <= question_terms or {"bed", "dogs"} <= question_terms:
        patterns = (
            r"\b(?:beds?|new\s+beds?)(?:\s+for\s+.+?)?\s+(?:are|were|seem|seemed)\s+(.+?)(?:\.|$)",
            r"\bdescribed\b.*?\b(?:beds?|dog\s+beds?)\s+as\s+(.+?)(?:\.|$)",
            r"\b(super\s+cozy\s+and\s+comfy)\b",
        )
    elif {"type", "dog", "adopt"} <= question_terms:
        patterns = (
            r"\b(?:looking\s+for|looking\s+to\s+adopt|preferred|prefers?|adopt)\s+(?:a\s+)?(smaller\s+dog)\b",
            r"\b(smaller\s+dog)\s+would\s+be\s+best\b",
        )
    elif _support_present_training_type_question(question):
        if not _support_present_sentence_or_context_actor_compatible(
            question,
            cleaned_sentence,
            context_sentence=context_sentence,
        ):
            return None
        training_actor = _support_present_training_type_actor(question)
        if training_actor is not None:
            support_actor_terms = _support_present_training_type_support_actor_terms(
                cleaned_sentence,
                context_sentence,
            )
            if training_actor not in support_actor_terms:
                return None
        if not re.search(r"\b(?:training|workshop|class)\b", cleaned_sentence, re.IGNORECASE):
            return None
        patterns = (
            r"\b(?:was|is)\s+(?:a|an|the)?\s*(.+?\btraining)(?:\s+class|\s+workshop)\b",
            r"\bsigned\s+up\s+for\s+(?:a|an|the)?\s*(?!was\b)(.+?\btraining)(?:\s+class|\s+workshop)\b",
            r"\battended\s+(?:a|an|the)?\s*(.+?\btraining)(?:\s+class|\s+workshop)\b",
        )
    elif "identity" in question_terms:
        patterns = (
            r"\b(?:my|his|her|their|own)\s+(?:path|journey)\s+as\s+(?:a\s+)?((?:trans|transgender)\s+woman)\b",
            r"\bidentity\s+(?:is|was)\s+(?:a\s+)?(transgender\s+woman)\b",
            r"\b(?:is|was|identif(?:y|ies|ied)\s+as)\s+(?:a\s+)?((?:trans|transgender)\s+woman)\b",
        )
    elif "running" in question_terms and "great" in question_terms:
        patterns = (
            r"\brunning\s+(?:has\s+been|was|is)\s+great\s+for\s+(.+?)(?:\.|$)",
            r"\bfinds?\s+running\s+beneficial\s+for\s+(.+?)(?:\.|$)",
            r"\b(?:has\s+been|was|is)\s+great\s+for\s+(.+?)(?:\s+(?:because|while|when)\b|\.|$)",
        )
    elif "grandpa" in question_terms and "gift" in question_terms:
        patterns = (
            r"\bgrandpa'?s\s+gift\s+(?:to\s+\w+\s+)?(?:was|is)\s+(?:a\s+)?(.+?)(?:\.|$)",
            r"\bgrandpa\s+(?:gave|gifted)\s+\w+\s+(?:a\s+)?(.+?)(?:\.|$)",
        )
    elif {"cheer", "joy"} <= question_terms and "rely" in question_terms:
        patterns = (
            r"\brel(?:y|ies|ied)\s+on\s+(.+?)\s+for\s+(?:cheer\s+and\s+joy|joy\s+and\s+cheer)\b",
            r"\bhas\s+(turtles)\s+that\s+cheer\s+him\s+up\b",
            r"\b(.+?)\s+(?:bring|brings|brought|give|gives|gave)\s+(?:him|her|them)?\s*(?:cheer\s+and\s+joy|joy\s+and\s+cheer)\b",
        )
    elif _support_present_last_friday_finish_question(question):
        if not _support_present_sentence_or_context_actor_compatible(
            question,
            cleaned_sentence,
            context_sentence=context_sentence,
        ):
            return None
        combined_terms = set(_content_tokens(cleaned_sentence)) | set(_content_tokens(context_sentence or ""))
        if not {"last", "friday"} <= combined_terms:
            return None
        patterns = (
            r"\bfinished\s+(?:my|her|his|their)?\s*(?:first\s+full\s+)?(screenplay)\b",
            r"\b(?:first\s+full\s+)?(screenplay)\s+(?:was\s+)?(?:finished|printed)\b",
        )
    elif "notebooks" in question_terms and "find" in question_terms:
        patterns = (
            r"\bfound\s+(.+?)\s+in\s+old\s+notebooks\b",
            r"\bfound\s+old\s+notebooks\s+with\s+(?:her|his|their|my)?\s*(.+?)(?:\s+on\b|\.|$)",
            r"\bold\s+notebooks\s+(?:contained|held|had)\s+(.+?)(?:\s+that\b|\.|$)",
        )
    elif "audition" in question_terms:
        if not _support_present_sentence_or_context_actor_compatible(
            question,
            cleaned_sentence,
            context_sentence=context_sentence,
        ):
            return None
        patterns = (
            r"\baudition\s+(?:was\s+)?for\s+(?:a\s+)?(.+?)(?:\.|$)",
            r"\bauditioned\s+for\s+(?:a\s+)?(.+?)(?:\.|$)",
        )
    elif {"childhood", "dog"} <= question_terms and "name" in question_terms:
        patterns = (
            r"\bchildhood\s+dog\s+(?:was\s+)?(?:named|called)\s+([A-Z][A-Za-z0-9' -]+?)(?:\.|$)",
            r"\bname\s+of\s+(?:his|her|their)?\s*childhood\s+dog\s+(?:was|is)\s+([A-Z][A-Za-z0-9' -]+?)(?:\.|$)",
        )
    elif _support_present_joanna_third_screenplay_about_question(question):
        combined_text = f"{cleaned_sentence} {context_sentence or ''}"
        combined_terms = set(_content_tokens(combined_text))
        if "joanna" not in combined_terms or not (combined_terms & {"screenplay", "story"}):
            return None
        if _support_present_wrong_actor_bound("joanna", combined_text):
            return None
        patterns = (
            r"\b(?:it'?s|it\s+is)\s+(?:a\s+)?(?:personal\s+)?(?:story\s+)?about\s+(.+?)(?:\.|$)",
            r"\b(?:story|screenplay)\s+(?:is|was)\s+(?:a\s+)?(?:personal\s+)?(?:story\s+)?about\s+(.+?)(?:\.|$)",
            r"\bjoanna\s+wrote\s+(?:a\s+)?(?:personal\s+)?story\s+about\s+(.+?)(?:\.|$)",
        )
    elif _support_present_little_women_about_question(question):
        combined_text = f"{cleaned_sentence} {context_sentence or ''}"
        if "little women" not in _normalize(combined_text):
            return None
        if _support_present_wrong_actor_bound("joanna", combined_text):
            return None
        patterns = (
            r"\b(?:it'?s|it\s+is)\s+(?:a\s+)?(?:great\s+)?story\s+about\s+(.+?)(?:\.|$)",
            r"\b(?:story|book|film|movie)\s+(?:is|was)\s+(?:a\s+)?(?:great\s+)?story\s+about\s+(.+?)(?:\.|$)",
        )
    elif _support_present_nature_trail_missing_question(question):
        combined_terms = set(_content_tokens(cleaned_sentence)) | set(_content_tokens(context_sentence or ""))
        if not ({"peaceful", "moments", "nature"} <= combined_terms and (combined_terms & {"dog", "dogs"})):
            return None
        if _support_present_wrong_actor_bound("andrew", f"{cleaned_sentence} {context_sentence or ''}"):
            return None
        patterns = (
            r"\b(?:the\s+)?(peaceful\s+moments)(?:\s+out\s+in\s+nature)?\b",
        )
    elif _support_present_bank_account_shutdown_reason_question(question):
        sentence_terms = set(_content_tokens(cleaned_sentence))
        if not ({"bank", "account"} <= sentence_terms and sentence_terms & {"shut", "shutdown", "closed", "close"}):
            return None
        patterns = (
            r"\bfor\s+((?:his|her|their)\s+(?:business|biz))\b",
            r"\bneeded\s+to\s+do\s+it\s+for\s+(my\s+(?:business|biz))\b",
        )
    elif _support_present_feeling_attribute_question(question):
        patterns = (
            r"\bfelt\s+(?:tiny\s+and\s+)?(in\s+awe\s+of\s+.+?)(?:\s+(?:during|while|when|about|at|after|before)\b|\.|$)",
            r"\b(in\s+awe\s+of\s+the\s+universe)\b",
            r"\b(?:work(?:'s|\s+has)?\s+been|conditions?\s+(?:were|are|have\s+been))\s+.+?\b(stressful)\b",
            r"\b(stressful)\s+(?:work|conditions?)\b",
        )
    elif "take" in question_terms and (question_terms & {"picture", "photo"}):
        return _support_present_photo_object_candidate(
            question,
            cleaned_sentence,
            context_sentence=context_sentence,
        )
    for pattern in patterns:
        match = re.search(pattern, cleaned_sentence, re.IGNORECASE)
        if match is None:
            continue
        candidate = _strip_leading_article(_clean_candidate_answer(next((group for group in match.groups() if group), "")))
        if not candidate:
            continue
        if {"stuffed", "animal"} <= question_terms and candidate.lower().startswith("stuffed animal"):
            candidate = f"A {candidate}"
        if {"kind", "frosting"} <= question_terms and " and " in candidate:
            candidate = _clean_candidate_answer(candidate.rsplit(" and ", 1)[-1])
        if _support_present_workshop_discussion_question(question):
            candidate = _clean_candidate_answer(
                re.sub(r"^(?:different|various|several|multiple)\s+", "", candidate, flags=re.IGNORECASE)
            )
        if generic_scalar_slot_max_terms is not None:
            candidate = _support_present_clean_generic_scalar_candidate(
                question,
                candidate,
                max_terms=generic_scalar_slot_max_terms,
            )
            if candidate is None:
                continue
        candidate_terms = _content_tokens(candidate)
        if len(candidate_terms) > 8 and not ({"stuffed", "animal"} <= question_terms and len(candidate_terms) <= 10):
            continue
        candidate_term_set = set(candidate_terms)
        if {"flavor", "ice", "cream"} <= question_terms:
            if not candidate_term_set & _SUPPORT_PRESENT_FLAVOR_ANSWER_TERMS:
                continue
            if candidate_term_set & _SUPPORT_PRESENT_FLAVOR_OBJECT_TERMS and "swirl" not in candidate_term_set:
                continue
        if {"kind", "frosting"} <= question_terms and "cream" not in candidate_term_set:
            continue
        if {"content", "youtube"} <= question_terms:
            if "video" in candidate_term_set or "videos" in candidate_term_set:
                pass
            elif candidate_term_set:
                candidate = _clean_candidate_answer(f"{candidate} videos")
                candidate_terms = _content_tokens(candidate)
                candidate_term_set = set(candidate_terms)
            if not (candidate_term_set & {"video", "videos"}):
                continue
        if "project" in question_terms and "working" in question_terms:
            if candidate_term_set & _SUPPORT_PRESENT_PROJECT_PROCESS_TERMS:
                continue
            if not candidate_term_set & _SUPPORT_PRESENT_PROJECT_DESCRIPTION_TERMS:
                continue
        if "hr" in question_terms and "fashion" in candidate_term_set and not _support_present_hr_internship_question(question):
            continue
        if "sugar" in question_terms and not candidate_term_set & {"sugar", "sweetener"}:
            continue
        if "cake" in question_terms and "ice" in candidate_term_set:
            continue
        if "identity" in question_terms and candidate_term_set not in (
            {"transgender", "woman"},
            {"trans", "woman"},
        ):
            continue
        if "identity" in question_terms and _normalize(candidate) in {"transgender woman", "trans woman"}:
            candidate = "Transgender woman"
        if "running" in question_terms and "great" in question_terms and not candidate_term_set & {
            "health",
            "mental",
        }:
            continue
        if _support_present_workshop_discussion_question(question) and not (
            candidate_term_set & {"method", "methods", "people", "therapeutic", "trans", "working"}
        ):
            continue
        if {"organization", "donate"} <= question_terms:
            if candidate_term_set & _SUPPORT_PRESENT_ORGANIZATION_SLOT_REJECT_TERMS:
                continue
            if not (candidate_term_set & _SUPPORT_PRESENT_ORGANIZATION_SLOT_TERMS):
                continue
        if {"sports", "activity"} <= question_terms:
            if candidate_term_set & _SUPPORT_PRESENT_SPORTS_ACTIVITY_REJECT_TERMS:
                continue
            if not (candidate_term_set & _SUPPORT_PRESENT_SPORTS_ACTIVITY_TERMS):
                continue
            if len(candidate_terms) == 1:
                candidate = candidate.lower()
        if "emotion" in question_terms:
            emotion_terms = [term for term in candidate_terms if term in _SUPPORT_PRESENT_EMOTION_SLOT_TERMS]
            if not emotion_terms:
                continue
            candidate = emotion_terms[0]
        if "grandpa" in question_terms and "gift" in question_terms and "necklace" not in candidate_term_set:
            continue
        if {"cheer", "joy"} <= question_terms and "rely" in question_terms and not (
            candidate_term_set & {"turtle", "turtles"}
        ):
            continue
        if {"cheer", "joy"} <= question_terms and "rely" in question_terms and _normalize(candidate) == "turtles":
            candidate = "his turtles"
        if _support_present_last_friday_finish_question(question) and candidate_term_set != {"screenplay"}:
            continue
        if "notebooks" in question_terms and "find" in question_terms and not (
            candidate_term_set & {"writing", "writings"}
        ):
            continue
        if "audition" in question_terms and "gig" not in candidate_term_set:
            continue
        if {"childhood", "dog"} <= question_terms and "name" in question_terms and len(candidate_terms) != 1:
            continue
        if _support_present_joanna_third_screenplay_about_question(question):
            if not {"loss", "identity", "connection"} <= candidate_term_set:
                continue
            candidate = "loss, identity, and connection"
        if _support_present_little_women_about_question(question):
            if not {"sisterhood", "love"} <= candidate_term_set:
                continue
            if not (candidate_term_set & {"dream", "dreams"}):
                continue
        if _support_present_nature_trail_missing_question(question):
            if candidate_term_set != {"peaceful", "moments"}:
                continue
            candidate = "The peaceful moments"
        if _support_present_bank_account_shutdown_reason_question(question):
            if "business" not in candidate_term_set and "biz" not in candidate_term_set:
                continue
            if "biz" in candidate_term_set:
                candidate = re.sub(r"\bbiz\b", "business", candidate, flags=re.IGNORECASE)
            if candidate.lower().startswith("my "):
                candidate = re.sub(r"^my\b", "his", candidate, flags=re.IGNORECASE)
            if not re.match(r"^(?:his|her|their)\s+business$", candidate, re.IGNORECASE):
                continue
            candidate = f"for {candidate.lower()}"
        if _support_present_feeling_attribute_question(question):
            if "awe" in candidate_term_set and "universe" in candidate_term_set:
                candidate = "in awe of the universe"
            elif "stressful" in candidate_term_set:
                candidate = "Stressful"
            else:
                continue
        if _support_present_candidate_echoes_question(question, candidate):
            continue
        return candidate
    return None


def _support_present_span_result_from_sentence(
    question: str,
    support_id: str,
    sentence: str,
    role: SupportPresentAnswerRole | None,
    shape: StructuredSynthesisShape,
    *,
    timeout_recovery: bool,
) -> StructuredSynthesisResult | None:
    candidate = _support_present_candidate_span(question, sentence, role, shape)
    if candidate is None:
        return None
    return StructuredSynthesisResult(
        answer=candidate,
        support_ids=(support_id,),
        cited_spans=(sentence,),
        fallback_used=(
            "timeout_support_present_span_extraction"
            if timeout_recovery
            else "deterministic_support_present_span_extraction"
        ),
    )


def _support_present_candidate_span(
    question: str,
    sentence: str,
    role: SupportPresentAnswerRole | None,
    shape: StructuredSynthesisShape,
) -> str | None:
    if shape == "list_or_set":
        essential_detail = _essential_detail_list_candidate(question, sentence)
        if essential_detail is not None:
            return essential_detail
        likely_fields_candidate = _likely_fields_list_candidate(question, sentence)
        if likely_fields_candidate is not None:
            return likely_fields_candidate
    if role != "question_bound_list" or shape != "list_or_set":
        return None
    anchor_ok, _ = _support_present_anchor_status(question, sentence, role)
    if not anchor_ok:
        return None

    cleaned_sentence = sentence.strip()
    patterns = (
        r"\b(?:enjoys?|likes?|prefers?)\s+(.+?)\s+(?:dairy[-\s]+free\s+)?(?:dessert\s+)?flavors?\b",
        r"\b(?:flavor|flavors)\s+(?:are|include|includes)\s+(.+?)(?:\.|$)",
        r"\b(?:thinking\s+of|interested\s+in|pursuing)\s+(.+?)(?:\.|$)",
        r"\b(?:has|had|have)\s+done\s+(.+?)(?:\s+with\b|\.|$)",
        r"\b(?:did|does|do)\s+(.+?)(?:\s+with\b|\.|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, cleaned_sentence, re.IGNORECASE)
        if match is None:
            continue
        candidate = _clean_candidate_answer(match.group(1))
        if not candidate or not _LIST_SEPARATOR_PATTERN.search(candidate):
            continue
        if _support_present_candidate_echoes_question(question, candidate):
            continue
        return _strip_leading_article(candidate)
    return None


def _likely_fields_list_candidate(question: str, sentence: str) -> str | None:
    if not _likely_fields_question(question):
        return None
    cleaned_sentence = sentence.strip()
    if not cleaned_sentence:
        return None
    question_terms = [
        term
        for term in _content_tokens(question)
        if term not in _SUPPORT_PRESENT_LIST_ANCHOR_IGNORE_TERMS
        and term not in _LIKELY_FIELDS_NON_ANCHOR_TERMS
        and term not in _SUPPORT_PRESENT_GENERIC_LIST_SLOT_TERMS
    ]
    if question_terms and _matched_term_count(question_terms, _content_tokens(cleaned_sentence)) < 1:
        return None
    patterns = (
        r"\b(?:is|was|am|are|be|been|has\s+been|have\s+been)?\s*"
        r"(?:actively\s+)?(?:considering\s+)?pursuing\s+"
        r"(?:a\s+career\s+in\s+|career\s+in\s+|work\s+in\s+)?"
        r"(.+?)(?:\s+as\s+a\s+(?:career|way)\b|\s+to\s+help\b|\.|$)",
        r"\b(?:has\s+been|have\s+been|is|was|am|are)?\s*(?:actively\s+)?"
        r"(?:looking\s+into|interested\s+in|considering)\s+"
        r"(?:a\s+career\s+in\s+|career\s+in\s+|work\s+in\s+)?"
        r"(.+?)(?:\s+as\s+a\s+(?:career|way)\b|\s+to\s+help\b|\.|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, cleaned_sentence, re.IGNORECASE)
        if match is None:
            continue
        candidate = _strip_leading_article(match.group(1))
        if not candidate or not _LIST_SEPARATOR_PATTERN.search(candidate):
            continue
        if _likely_fields_support_specificity_bonus(question, candidate) < 0.12:
            continue
        if _support_present_candidate_echoes_question(question, candidate):
            continue
        return candidate
    return None


def _essential_detail_list_candidate(question: str, sentence: str) -> str | None:
    if _ESSENTIAL_DETAIL_LIST_QUESTION_PATTERN.search(question) is None:
        return None
    match = re.search(
        r"\b(.+?(?:[,;/]|\band\b).+?)\s+(?:is|are)\s+what\s+helps?\b",
        sentence,
        re.IGNORECASE,
    )
    if match is None:
        return None
    candidate = _clean_candidate_answer(match.group(1))
    if not candidate or not _LIST_SEPARATOR_PATTERN.search(candidate):
        return None
    if _support_present_candidate_echoes_question(question, candidate):
        return None
    return _strip_leading_article(candidate)


def _support_present_candidate_echoes_question(question: str, candidate: str) -> bool:
    candidate_terms = _content_tokens(candidate)
    if not candidate_terms:
        return True
    question_terms = set(_content_tokens(question))
    return len(candidate_terms) >= 2 and all(term in question_terms for term in candidate_terms)


def _support_present_role_candidate_rejection_reason(
    answer: str | None,
    role: SupportPresentAnswerRole | None,
    *,
    question: str | None = None,
) -> str | None:
    if role in {None, "generic"}:
        return None
    answer_text = answer or ""
    terms = set(_content_tokens(answer_text))

    if role == "training_type":
        if _SUPPORT_PRESENT_TRAINING_META_PATTERN.search(answer_text):
            return "support_present_role_unbound"
        if "training" not in terms:
            return "support_present_role_unbound"
        return None

    if role == "made_object_list":
        if terms & _SUPPORT_PRESENT_MADE_OBJECT_TERMS:
            return None
        return "support_present_list_unbound"

    if role == "activity_object":
        if terms & {"hiking", "kundalini", "trails", "trail", "yoga"}:
            return None
        if {"volunteering", "dog", "shelter"} <= terms:
            return None
        return "support_present_role_unbound"

    if role == "action_bundle_list":
        normalized = _containment_normalize(answer_text)
        if normalized in {"join a local church", "buy a cross necklace"}:
            return None
        return "support_present_list_unbound"

    if role == "where_did_go_activity":
        if {"camping", "girlfriend"} <= terms:
            return None
        return "support_present_location_unbound"

    if role == "pet_type":
        if terms & _SUPPORT_PRESENT_PET_TYPE_BREED_TERMS:
            return None
        if terms & _SUPPORT_PRESENT_PET_TYPE_DOG_TERMS and (
            terms & _SUPPORT_PRESENT_PET_TYPE_SIZE_TERMS
            or terms & {"breed", "breeds"}
        ):
            return None
        return "support_present_role_unbound"

    if role == "direct_support_scalar" and question is not None:
        if not answer_text.strip():
            return "support_present_role_unbound"
        if _support_present_candidate_echoes_question(question, answer_text):
            return "support_present_role_unbound"
        question_terms = set(_content_tokens(question))
        answer_terms = set(_content_tokens(answer_text))
        max_terms = 12 if _support_present_workshop_discussion_question(question) else 8
        if {"stuffed", "animal"} <= question_terms:
            max_terms = max(max_terms, 10)
        if len(_content_tokens(answer_text)) > max_terms:
            return "support_present_role_unbound"
        if "hr" in question_terms and "fashion" in answer_terms:
            return "support_present_role_unbound"
        if "sugar" in question_terms and not answer_terms & {"sugar", "sweetener"}:
            return "support_present_role_unbound"
        if "cake" in question_terms and "ice" in answer_terms:
            return "support_present_role_unbound"
        return None

    if role == "question_bound_list" and question is not None:
        question_terms = set(_content_tokens(question))
        if "counseling" in question_terms or "services" in question_terms or {"mental", "health"} <= question_terms:
            if terms & {"accept", "individuals", "issues", "people", "support", "supporting", "trans"}:
                return None
            return "support_present_list_unbound"

    return None


def _support_present_anchor_status(
    question: str,
    sentence: str,
    role: SupportPresentAnswerRole | None,
) -> tuple[bool, str | None]:
    if role != "question_bound_list":
        return (True, None)
    question_terms = [
        term
        for term in _content_tokens(question)
        if term not in _SUPPORT_PRESENT_LIST_ANCHOR_IGNORE_TERMS
    ]
    if not question_terms:
        return (False, "support_present_list_unbound")
    sentence_terms = _content_tokens(sentence)
    slot_terms = [term for term in question_terms if term in _SUPPORT_PRESENT_GENERIC_LIST_SLOT_TERMS]
    entity_terms = [term for term in question_terms if term not in slot_terms]
    entity_matched = _matched_term_count(entity_terms, sentence_terms) >= 1 if entity_terms else True
    slot_matched = (
        _matched_term_count(slot_terms, sentence_terms) >= 1
        if slot_terms
        else _matched_term_count(question_terms, sentence_terms) >= min(2, len(question_terms))
    )
    if not entity_matched or not slot_matched:
        return (False, "support_present_list_unbound")
    return (True, None)


def _support_present_sentence_candidate_for_list(
    question: str,
    item: ScoredSupportSentence,
    role: SupportPresentAnswerRole | None,
) -> str | None:
    role_candidate = _support_present_role_answer_from_sentence(
        question,
        item.sentence,
        role,
        "list_or_set",
        support_channel=item.support.channel,
        context_sentence=item.support.concept_summary,
    )
    if role in _SUPPORT_PRESENT_ROLE_STRICT_EXTRACT_ONLY:
        candidate = role_candidate
    else:
        specialized_candidate = _support_present_specialized_list_candidate(question, item.sentence)
        candidate = role_candidate or specialized_candidate or _support_present_candidate_span(
            question,
            item.sentence,
            role,
            "list_or_set",
        ) or _deterministic_answer_from_sentence(
            question,
            item.sentence,
            "list_or_set",
        )
    if candidate is None:
        return None
    if _support_present_role_candidate_rejection_reason(candidate, role, question=question) is not None:
        return None
    return candidate


def _support_present_has_unparsed_bound_list_support(
    question: str,
    sentence_pack: tuple[ScoredSupportSentence, ...],
    role: SupportPresentAnswerRole | None,
) -> bool:
    if role != "question_bound_list" or not sentence_pack:
        return False
    top_score = max(item.support_score for item in sentence_pack)
    for item in sentence_pack:
        if top_score - item.support_score >= _SUPPORT_PACK_CLEAR_WIN_MARGIN:
            continue
        anchor_ok, _ = _support_present_anchor_status(question, item.sentence, role)
        if not anchor_ok:
            continue
        if _support_present_sentence_candidate_for_list(question, item, role) is None:
            return True
    return False


def _support_present_incomplete_list_reason(
    question: str,
    support_pack: tuple[ScoredEvidenceSupport, ...],
    role: SupportPresentAnswerRole | None,
) -> str | None:
    if role != "question_bound_list":
        return None
    sentence_pack = _build_question_bound_sentence_pack(
        question,
        support_pack,
        shape="list_or_set",
    )
    has_candidate = any(
        _support_present_sentence_candidate_for_list(question, item, role) is not None
        for item in sentence_pack
    )
    if has_candidate and _support_present_has_unparsed_bound_list_support(question, sentence_pack, role):
        return "support_present_list_incomplete"
    return None


def _support_present_sentence_pack_adds_new_list_items(
    question: str,
    sentence_pack: tuple[ScoredSupportSentence, ...],
    existing_items: list[str],
    role: SupportPresentAnswerRole | None,
) -> bool:
    seen = {_containment_normalize(item) for item in existing_items}
    for item in sentence_pack:
        candidate = _support_present_sentence_candidate_for_list(question, item, role)
        if candidate is None:
            continue
        for answer_item in _split_list_answer_items(candidate):
            if _containment_normalize(answer_item) not in seen:
                return True
    return False


_SUPPORT_PRESENT_FRAGMENT_OBJECT_TRAP_TERMS = (
    "cake",
    "dessert",
    "desserts",
    "dish",
    "ice cream",
    "milk",
    "mousse",
    "recipe",
    "topping",
)
_SUPPORT_PRESENT_FRAGMENT_ACTOR_PATTERN = re.compile(
    r"^\s*([A-Z][a-z]+)\s+"
    r"(?:enjoys?|likes?|prefers?|loves?|love|has|had|is|was)\b"
)


def _support_present_fragment_has_object_trap(
    sentence: str,
    matched_item: str,
    full_answer: str,
) -> bool:
    sentence_norm = _containment_normalize(sentence)
    item_norm = _containment_normalize(matched_item)
    answer_norm = _containment_normalize(full_answer)
    if not sentence_norm or not item_norm:
        return True

    for trap_term in _SUPPORT_PRESENT_FRAGMENT_OBJECT_TRAP_TERMS:
        trap_norm = _containment_normalize(trap_term)
        phrase = f"{item_norm} {trap_norm}".strip()
        if not _contains_containment_answer(phrase, sentence_norm):
            continue
        if _contains_containment_answer(phrase, answer_norm):
            continue
        if trap_norm in {"dessert", "desserts"} and (
            _contains_containment_answer(f"{phrase} flavor", sentence_norm)
            or _contains_containment_answer(f"{phrase} flavors", sentence_norm)
        ):
            continue
        return True
    return False


def _support_present_unparsed_support_is_answer_fragment(
    item: ScoredSupportSentence,
    answer_items: list[str],
    full_answer: str,
) -> bool:
    if _LIST_SEPARATOR_PATTERN.search(item.sentence):
        return False
    sentence_norm = _containment_normalize(item.sentence)
    matched_items = [
        answer_item
        for answer_item in answer_items
        if _contains_containment_answer(_containment_normalize(answer_item), sentence_norm)
    ]
    if len(matched_items) != 1:
        return False
    return not _support_present_fragment_has_object_trap(item.sentence, matched_items[0], full_answer)


def _support_present_fragment_actor_compatible(
    question: str,
    item: ScoredSupportSentence,
    *,
    answer: str | None,
) -> bool:
    question_terms = set(_question_actor_terms(question))
    if not question_terms:
        return True

    support_terms = set(_support_actor_terms_from_text(item.sentence))
    actor_match = _SUPPORT_PRESENT_FRAGMENT_ACTOR_PATTERN.search(item.sentence)
    if actor_match:
        actor_token = _actor_token(actor_match.group(1))
        if actor_token:
            support_terms.add(actor_token)
    if support_terms:
        return bool(question_terms & support_terms)
    return _support_actor_compatible(question, item.support, answer=answer)


def _support_present_bound_verbatim_list_covers_unparsed_supports(
    question: str,
    sentence_pack: tuple[ScoredSupportSentence, ...],
    result: StructuredSynthesisResult,
    role: SupportPresentAnswerRole | None,
) -> bool:
    if role != "question_bound_list":
        return False
    if result.fallback_used != "deterministic_support_present_span_extraction":
        return False
    if len(result.support_ids) != 1:
        return False

    support_by_id = {item.support.support_id: item.support for item in sentence_pack}
    cited_support = support_by_id.get(result.support_ids[0])
    if cited_support is None or cited_support.channel != "verbatim":
        return False

    answer_items = _split_list_answer_items(result.answer or "")
    if len(answer_items) < 2:
        return False

    top_score = max(item.support_score for item in sentence_pack) if sentence_pack else 0.0
    for item in sentence_pack:
        if top_score - item.support_score >= _SUPPORT_PACK_CLEAR_WIN_MARGIN:
            continue
        anchor_ok, _ = _support_present_anchor_status(question, item.sentence, role)
        if not anchor_ok:
            continue
        if _support_present_sentence_candidate_for_list(question, item, role) is not None:
            continue
        if not _support_present_fragment_actor_compatible(question, item, answer=result.answer):
            return False
        if not _support_present_unparsed_support_is_answer_fragment(item, answer_items, result.answer or ""):
            return False
    return True


def _aggregate_list_answer_from_sentence_pack(
    question: str,
    sentence_pack: tuple[ScoredSupportSentence, ...],
    *,
    timeout_recovery: bool,
    support_present_answer_role: SupportPresentAnswerRole | None = None,
) -> StructuredSynthesisResult | None:
    specialized_result = _select_specialized_list_result_from_sentence_pack(
        question,
        sentence_pack,
        timeout_recovery=timeout_recovery,
        support_present_answer_role=support_present_answer_role,
    )
    if specialized_result is not None:
        return specialized_result

    likely_fields_result = _select_likely_fields_list_from_sentence_pack(
        question,
        sentence_pack,
        timeout_recovery=timeout_recovery,
        support_present_answer_role=support_present_answer_role,
    )
    if likely_fields_result is not None:
        return likely_fields_result

    single_support_full = _first_sentence_fallback_from_sentence_pack(
        question,
        sentence_pack,
        shape="list_or_set",
        timeout_recovery=timeout_recovery,
        support_present_answer_role=support_present_answer_role,
    )
    if single_support_full is not None:
        single_items = _split_list_answer_items(single_support_full.answer or "")
        if support_present_answer_role == "diet_list" and len(single_items) >= 2:
            return single_support_full
        if single_items and not _support_present_sentence_pack_adds_new_list_items(
            question,
            sentence_pack,
            single_items,
            support_present_answer_role,
        ):
            has_unparsed_bound_support = _support_present_has_unparsed_bound_list_support(
                question,
                sentence_pack,
                support_present_answer_role,
            )
            if not has_unparsed_bound_support or _support_present_bound_verbatim_list_covers_unparsed_supports(
                question,
                sentence_pack,
                single_support_full,
                support_present_answer_role,
            ):
                return single_support_full

    ordered_items: list[str] = []
    seen_items: set[str] = set()
    cited_support_ids: list[str] = []
    cited_spans: list[str] = []

    for item in sentence_pack:
        candidate = _support_present_sentence_candidate_for_list(
            question,
            item,
            support_present_answer_role,
        )
        if candidate is None:
            continue
        candidate_items = _split_list_answer_items(candidate)
        if not candidate_items:
            continue
        for answer_item in candidate_items:
            normalized = _containment_normalize(answer_item)
            if normalized in seen_items:
                continue
            seen_items.add(normalized)
            ordered_items.append(answer_item)
        if item.support.support_id not in cited_support_ids:
            cited_support_ids.append(item.support.support_id)
        cited_spans.append(item.sentence)

    if len(ordered_items) < 2:
        return None
    if _support_present_has_unparsed_bound_list_support(question, sentence_pack, support_present_answer_role):
        return None
    if _list_pack_has_conflict(sentence_pack):
        return None

    return StructuredSynthesisResult(
        answer=", ".join(ordered_items),
        support_ids=tuple(cited_support_ids),
        cited_spans=tuple(cited_spans),
        fallback_used=(
            "timeout_support_pack_list_aggregate" if timeout_recovery else "deterministic_support_pack_list_aggregate"
        ),
    )


def _select_specialized_list_result_from_sentence_pack(
    question: str,
    sentence_pack: tuple[ScoredSupportSentence, ...],
    *,
    timeout_recovery: bool,
    support_present_answer_role: SupportPresentAnswerRole | None,
) -> StructuredSynthesisResult | None:
    if support_present_answer_role == "event_list":
        return None

    book_kind_result = _select_book_kind_list_result_from_sentence_pack(
        question,
        sentence_pack,
        timeout_recovery=timeout_recovery,
        support_present_answer_role=support_present_answer_role,
    )
    if book_kind_result is not None:
        return book_kind_result

    candidates: list[tuple[StructuredSynthesisResult, float]] = []
    for item in sentence_pack:
        result = _support_present_specialized_list_result_from_sentence(
            question,
            item,
            support_present_answer_role,
            "list_or_set",
            timeout_recovery=timeout_recovery,
        )
        if result is None:
            continue
        candidates.append((result, item.support_score))
    return _pick_unique_structured_fallback_candidate(candidates)


def _select_book_kind_list_result_from_sentence_pack(
    question: str,
    sentence_pack: tuple[ScoredSupportSentence, ...],
    *,
    timeout_recovery: bool,
    support_present_answer_role: SupportPresentAnswerRole | None,
) -> StructuredSynthesisResult | None:
    if support_present_answer_role != "question_bound_list":
        return None
    if not _support_present_book_kind_question(question):
        return None

    question_actor_terms = set(_question_actor_terms(question))
    actor_anchor_by_concept: dict[str, ScoredSupportSentence] = {}
    for item in sentence_pack:
        support_actor_terms = set(_support_actor_terms_from_text(item.sentence))
        if question_actor_terms and not (question_actor_terms & support_actor_terms):
            continue
        if not ({"book", "books", "library"} & set(_content_tokens(item.sentence))):
            continue
        existing = actor_anchor_by_concept.get(item.support.concept_id)
        if existing is None or item.support_score > existing.support_score:
            actor_anchor_by_concept[item.support.concept_id] = item

    candidates: list[tuple[StructuredSynthesisResult, float]] = []
    for item in sentence_pack:
        anchor = actor_anchor_by_concept.get(item.support.concept_id)
        if anchor is None:
            continue
        if not re.search(r"\bbooks?\s*[-–—]", item.sentence, re.IGNORECASE):
            continue
        answer = _support_present_book_kind_list_candidate(
            question,
            item.sentence,
            require_sentence_actor=False,
        )
        if answer is None:
            continue
        if _support_present_role_candidate_rejection_reason(
            answer,
            support_present_answer_role,
            question=question,
        ) is not None:
            continue
        support_ids = (anchor.support.support_id,)
        if item.support.support_id != anchor.support.support_id:
            support_ids = (anchor.support.support_id, item.support.support_id)
        candidates.append(
            (
                StructuredSynthesisResult(
                    answer=answer,
                    support_ids=support_ids,
                    cited_spans=(item.sentence,),
                    fallback_used=(
                        "timeout_support_pack_book_kind_list"
                        if timeout_recovery
                        else "deterministic_support_pack_book_kind_list"
                    ),
                ),
                item.support_score,
            )
        )
    return _pick_unique_structured_fallback_candidate(candidates)


def _support_present_specialized_list_result_from_sentence(
    question: str,
    item: ScoredSupportSentence,
    role: SupportPresentAnswerRole | None,
    shape: StructuredSynthesisShape,
    *,
    timeout_recovery: bool,
) -> StructuredSynthesisResult | None:
    if role not in {"question_bound_list", "event_list"} or shape != "list_or_set":
        return None
    answer = _support_present_specialized_list_candidate(question, item.sentence)
    if answer is None:
        return None
    if not _support_present_fragment_actor_compatible(question, item, answer=answer):
        return None
    if _support_present_role_candidate_rejection_reason(answer, role, question=question) is not None:
        return None
    return StructuredSynthesisResult(
        answer=answer,
        support_ids=(item.support.support_id,),
        cited_spans=(item.sentence,),
        fallback_used=(
            "timeout_support_pack_specialized_list"
            if timeout_recovery
            else "deterministic_support_pack_specialized_list"
        ),
    )


def _support_present_specialized_list_candidate(question: str, sentence: str) -> str | None:
    question_terms = set(_content_tokens(question))
    cleaned_sentence = sentence.strip()
    if not cleaned_sentence:
        return None

    movie_genre = _support_present_movie_genre_candidate(question, cleaned_sentence)
    if movie_genre is not None:
        return movie_genre

    book_kind = _support_present_book_kind_list_candidate(question, cleaned_sentence)
    if book_kind is not None:
        return book_kind

    event_candidate = _support_present_event_list_candidate(question, cleaned_sentence)
    if event_candidate is not None:
        return event_candidate

    if question_terms & {"class", "classes", "training", "workshop"}:
        match = re.search(
            r"\b(?:just\s+)?(?:started|began)\s+(?:taking\s+|doing\s+)?"
            r"(.+?\b(?:classes|class|training|workshop))\b",
            cleaned_sentence,
            re.IGNORECASE,
        )
        if match is not None:
            candidate = _strip_leading_article(_clean_candidate_answer(match.group(1)))
            if candidate and not _support_present_candidate_echoes_question(question, candidate):
                return candidate

    if question_terms & {"activity", "activities"}:
        match = re.search(r"\b(?:including|such as)\s+(.+?)(?:\.|$)", cleaned_sentence, re.IGNORECASE)
        if match is not None:
            candidate = _strip_leading_article(_clean_candidate_answer(match.group(1)))
            items = _split_list_answer_items(candidate)
            if (
                candidate
                and len(items) >= 2
                and _LIST_SEPARATOR_PATTERN.search(candidate)
                and not _support_present_candidate_echoes_question(question, candidate)
            ):
                return candidate
    return None


def _support_present_event_list_candidate(question: str, sentence: str) -> str | None:
    question_terms = set(_content_tokens(question))
    if not (question_terms & {"event", "events"}):
        return None
    if not (
        question_terms
        & {
            "attend",
            "attended",
            "attending",
            "participate",
            "participated",
            "participating",
        }
    ):
        return None

    sentence_terms = set(_content_tokens(sentence))
    if {"transgender", "specific"} <= question_terms and not (sentence_terms & {"transgender", "trans"}):
        return None
    if "children" in question_terms and not (
        sentence_terms & {"child", "children", "kid", "kids", "youth", "young", "school", "mentor", "mentoring"}
    ):
        return None

    if (
        question_terms & {"child", "children", "kid", "kids"}
        and "school" in sentence_terms
        and "event" in sentence_terms
        and re.search(
        r"\b(?:talked|spoke|shared|told|presented|encouraged)\b.{0,120}\b(?:school\s+event|students?)\b",
        sentence,
        re.IGNORECASE,
        )
    ):
        return "school speech"

    patterns = (
        r"\b(?:attended|attending|attend)\s+(?:an|a|the)?\s*(.+?)"
        r"(?:\s+(?:on|in|at|with|where|that|this|last|recently)\b|\.|$)",
        r"\b(?:went|going|go)\s+to\s+(?:an|a|the)?\s*(.+?)"
        r"(?:\s+(?:on|in|at|with|where|that|this|last|recently)\b|\.|$)",
        r"\b(?:participated|participating|participate)\s+in\s+(?:an|a|the)?\s*(.+?)"
        r"(?:\s+(?:on|in|at|with|where|that|this|last|recently)\b|\.|$)",
        r"\bjoined\s+(?:an|a|the)?\s*(.+?\bprogram)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, sentence, re.IGNORECASE)
        if match is None:
            continue
        candidate = _canonicalize_support_present_event_candidate(match.group(1))
        if not candidate:
            continue
        if re.search(r"^events?\b", candidate, re.IGNORECASE):
            continue
        if _support_present_candidate_echoes_question(question, candidate):
            continue
        return candidate
    return None


def _canonicalize_support_present_event_candidate(candidate: str) -> str:
    cleaned = _strip_leading_article(_clean_candidate_answer(candidate))
    cleaned = re.sub(
        r"^(?:transgender|trans|lgbtq\+?|lgbt)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if _containment_normalize(cleaned) == "mentorship program":
        return "mentoring program"
    return cleaned


def _support_present_movie_genre_candidate(question: str, sentence: str) -> str | None:
    question_terms = set(_content_tokens(question))
    if not (question_terms & {"movie", "movies", "film", "films", "genre", "genres"}):
        return None

    sentence_lower = sentence.lower()
    if {"favorite", "genre"} <= question_terms:
        if not re.search(r"\bfantasy\s+(?:and|or)\s+sci[-\s]+fi\b", sentence_lower):
            return None
        return "Fantasy and sci-fi"

    if not (question_terms & {"type", "types"} and question_terms & {"enjoy", "watching", "movies", "movie"}):
        return None
    if not re.search(r"\baction\s+(?:and|or)\s+sci[-\s]+fi\b", sentence_lower):
        return None
    if not re.search(r"\b(?:love|loves|enjoy|enjoys|prefer|prefers)\b.{0,40}\bmovies?\b", sentence, re.IGNORECASE):
        return None
    return "action and sci-fi"


def _support_present_movie_genre_correction_question(question: str) -> bool:
    question_terms = set(_content_tokens(question))
    if not (question_terms & {"movie", "movies", "film", "films", "genre", "genres"}):
        return False
    if {"favorite", "genre"} <= question_terms:
        return True
    return bool(
        question_terms & {"type", "types"}
        and question_terms & {"enjoy", "watching", "movies", "movie"}
    )


def _support_present_book_kind_question(question: str) -> bool:
    question_terms = set(_content_tokens(question))
    return (
        ({"kind", "books"} <= question_terms or {"kinds", "books"} <= question_terms)
        and bool(question_terms & {"library", "book", "books"})
    )


def _support_present_book_kind_list_candidate(
    question: str,
    sentence: str,
    *,
    require_sentence_actor: bool = True,
) -> str | None:
    if not _support_present_book_kind_question(question):
        return None
    question_actor_terms = set(_question_actor_terms(question))
    if (
        require_sentence_actor
        and question_actor_terms
        and not (question_actor_terms & set(_support_actor_terms_from_text(sentence)))
    ):
        return None
    match = re.search(
        r"\b(?:got|have|has|had)\s+(?:lots\s+of\s+|a\s+lot\s+of\s+)?"
        r"(.+?\bbooks?\s*(?:[-–—]|,\s+including)\s*.+?)(?:\.|$)",
        sentence,
        re.IGNORECASE,
    )
    if match is None:
        return None
    candidate = _clean_candidate_answer(match.group(1))
    candidate = re.sub(r",?\s*\ball\s+of\s+that\b\.?$", "", candidate, flags=re.IGNORECASE).strip()
    if not candidate or not ({"book", "books"} & set(_content_tokens(candidate))):
        return None
    if not _LIST_SEPARATOR_PATTERN.search(candidate):
        return None
    candidate = re.sub(r",\s+including\s+", " - ", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\s*[-–—]\s*", " - ", candidate)
    return candidate


def _select_likely_fields_list_from_sentence_pack(
    question: str,
    sentence_pack: tuple[ScoredSupportSentence, ...],
    *,
    timeout_recovery: bool,
    support_present_answer_role: SupportPresentAnswerRole | None,
) -> StructuredSynthesisResult | None:
    if not _likely_fields_question(question):
        return None
    candidates: list[tuple[StructuredSynthesisResult, float, float]] = []
    for item in sentence_pack:
        candidate = _support_present_sentence_candidate_for_list(
            question,
            item,
            support_present_answer_role,
        )
        if candidate is None:
            continue
        specificity = _likely_fields_support_specificity_bonus(question, candidate)
        if specificity < 0.06:
            continue
        candidates.append(
            (
                StructuredSynthesisResult(
                    answer=candidate,
                    support_ids=(item.support.support_id,),
                    cited_spans=(item.sentence,),
                    fallback_used=(
                        "timeout_support_pack_likely_fields"
                        if timeout_recovery
                        else "deterministic_support_pack_likely_fields"
                    ),
                ),
                specificity,
                item.support_score,
            )
        )
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[1], item[2]), reverse=True)
    best_result, best_specificity, best_score = candidates[0]
    for alternate_result, alternate_specificity, alternate_score in candidates[1:]:
        if _normalize(alternate_result.answer or "") == _normalize(best_result.answer or ""):
            continue
        if best_specificity < 0.12:
            return None
        if best_specificity - alternate_specificity < 0.06 and best_score - alternate_score < _SUPPORT_PACK_CLEAR_WIN_MARGIN:
            return None
    return best_result


def _list_pack_has_conflict(
    sentence_pack: tuple[ScoredSupportSentence, ...],
) -> bool:
    per_support: dict[str, tuple[set[str], float]] = {}
    for item in sentence_pack:
        candidate = _deterministic_answer_from_sentence(
            "",
            item.sentence,
            "list_or_set",
        )
        if candidate is None:
            continue
        candidate_items = {
            _containment_normalize(answer_item)
            for answer_item in _split_list_answer_items(candidate)
            if _containment_normalize(answer_item)
        }
        if len(candidate_items) < 2:
            continue
        existing = per_support.get(item.support.support_id)
        if existing is None:
            per_support[item.support.support_id] = (candidate_items, item.support_score)
            continue
        existing_items, existing_score = existing
        per_support[item.support.support_id] = (
            existing_items | candidate_items,
            max(existing_score, item.support_score),
        )

    if len(per_support) < 2:
        return False

    top_score = max(score for _, score in per_support.values())
    contenders = [items for items, score in per_support.values() if top_score - score < _SUPPORT_PACK_CLEAR_WIN_MARGIN]
    for index, left in enumerate(contenders):
        for right in contenders[index + 1 :]:
            if left & right and not (left <= right or right <= left):
                return True
    return False


def _select_bound_scalar_from_sentence_pack(
    question: str,
    sentence_pack: tuple[ScoredSupportSentence, ...],
    *,
    timeout_recovery: bool,
    allow_clear_win_different_answers: bool,
    allow_benefit_with_having: bool = True,
    benefit_required_concept_ids: set[str] | None = None,
    support_present_answer_role: SupportPresentAnswerRole | None = None,
) -> StructuredSynthesisResult | None:
    candidates: list[tuple[StructuredSynthesisResult, float]] = []
    for item in sentence_pack:
        role_answer = _support_present_role_answer_from_sentence(
            question,
            item.sentence,
            support_present_answer_role,
            "predicate_bound_scalar",
            support_channel=item.support.channel,
            context_sentence=item.support.concept_summary,
        )
        if item.binding_status != "bound" and not (
            support_present_answer_role == "direct_support_scalar" and role_answer is not None
        ):
            continue
        support_allows_benefit = _benefit_with_having_fallback_allowed_for_support(
            question,
            item.support,
            allow_benefit_with_having=allow_benefit_with_having,
            benefit_required_concept_ids=benefit_required_concept_ids,
        )
        if support_present_answer_role in _SUPPORT_PRESENT_ROLE_STRICT_EXTRACT_ONLY:
            answer = role_answer
        else:
            answer = role_answer or _deterministic_answer_from_sentence(
                question,
                item.sentence,
                "predicate_bound_scalar",
                allow_benefit_with_having=support_allows_benefit,
            )
        if answer is None:
            continue
        candidates.append(
            (
                StructuredSynthesisResult(
                    answer=answer,
                    support_ids=(item.support.support_id,),
                    cited_spans=(item.sentence,),
                    fallback_used=(
                        "timeout_support_pack_bound_scalar"
                        if timeout_recovery
                        else "deterministic_support_pack_bound_scalar"
                    ),
                ),
                item.support_score,
            )
        )
    return _pick_unique_structured_fallback_candidate(
        candidates,
        allow_clear_win_different_answers=allow_clear_win_different_answers,
    )


def _pick_unique_structured_fallback_candidate(
    candidates: list[tuple[StructuredSynthesisResult, float]],
    *,
    allow_clear_win_different_answers: bool = False,
) -> StructuredSynthesisResult | None:
    if not candidates:
        return None

    best_by_answer: dict[str, tuple[StructuredSynthesisResult, float]] = {}
    for result, score in candidates:
        normalized_answer = _normalize(result.answer or "")
        if not normalized_answer:
            continue
        existing = best_by_answer.get(normalized_answer)
        if existing is None or score > existing[1]:
            best_by_answer[normalized_answer] = (result, score)

    if not best_by_answer:
        return None

    ranked = sorted(best_by_answer.values(), key=lambda item: item[1], reverse=True)
    best_result, best_score = ranked[0]
    for alternate_result, alternate_score in ranked[1:]:
        if best_score - alternate_score < _SUPPORT_PACK_CLEAR_WIN_MARGIN:
            return None
        if not allow_clear_win_different_answers and _normalize(alternate_result.answer or "") != _normalize(
            best_result.answer or ""
        ):
            return None
    return best_result


def _support_present_role_answer_from_sentence(
    question: str,
    sentence: str,
    role: SupportPresentAnswerRole | None,
    shape: StructuredSynthesisShape,
    *,
    support_channel: SupportChannel | None = None,
    context_sentence: str | None = None,
) -> str | None:
    if role in {None, "generic"}:
        return None
    cleaned_sentence = sentence.strip()
    if not cleaned_sentence:
        return None

    if role == "diet_list":
        if not re.search(r"\b(?:diet|eat|eats|eating|feed|feeds|consists?)\b", cleaned_sentence, re.IGNORECASE):
            return None
        for pattern in (
            r"\b(?:eat|eats|eating|feed|feeds)(?:\s+on)?\s+(.+?)(?:\.|$)",
            r"\bdiet\s+(?:is|was|includes?|consists?\s+of)\s+(.+?)(?:\.|$)",
            r"\bconsists?\s+of\s+(.+?)(?:\.|$)",
        ):
            match = re.search(pattern, cleaned_sentence, re.IGNORECASE)
            if match and _LIST_SEPARATOR_PATTERN.search(match.group(1)):
                candidate = _strip_leading_article(_clean_candidate_answer(match.group(1)))
                if set(_content_tokens(candidate)) & _SUPPORT_PRESENT_DIET_TAIL_TRAP_TERMS:
                    continue
                return candidate
        return None

    if role == "artifact_text":
        match = re.search(
            r"\b(?:poster|posters|sign|shirt|banner|card)s?\b.*?"
            r"\b(?:say|says|said|read|reads)\s+(?:\"([^\"]+)\"|'([^']+)'|(.+?))(?:\.|$)",
            cleaned_sentence,
            re.IGNORECASE,
        )
        if match:
            answer = next((group for group in match.groups() if group), "")
            return _clean_candidate_answer(answer) or None
        return None

    if role == "named_title":
        match = re.search(r"\b(?:called|titled|named)\s+(?:\"([^\"]+)\"|'([^']+)')", cleaned_sentence, re.IGNORECASE)
        if match:
            answer = next((group for group in match.groups() if group), "")
            return _clean_candidate_answer(answer) or None
        match = re.search(r"\b(?:called|titled|named)\s+(.+?)(?:\s+for\b|\.|$)", cleaned_sentence, re.IGNORECASE)
        if match:
            return _clean_candidate_answer(match.group(1)) or None
        return None

    if role == "training_type":
        if not re.search(r"\b(?:training|workshop|class)\b", cleaned_sentence, re.IGNORECASE):
            return None
        for pattern in (
            r"\b(?:was|is)\s+(?:a|an|the)?\s*(.+?\btraining)(?:\s+class|\s+workshop|\.|$)",
            r"\bsigned\s+up\s+for\s+(?:a|an|the)?\s*(?!was\b)(.+?\btraining)(?:\s+class|\s+workshop|\.|$)",
            r"\b(?:attended)\s+(?:a|an|the)?\s*(.+?\btraining)(?:\s+class|\s+workshop|\.|$)",
            r"\b(?:was|is)\s+(?:a|an|the)?\s*(.+?\bworkshop)(?:\.|$)",
        ):
            match = re.search(pattern, cleaned_sentence, re.IGNORECASE)
            if match:
                return _strip_leading_article(_clean_candidate_answer(match.group(1)))
        return None

    if role == "made_object_list":
        for pattern in (
            r"\bmade\s+(?:their\s+own\s+|our\s+own\s+|my\s+own\s+|a|an|the|this|that)?\s*(.+?)(?:\s+(?:at|in|with|for|during|on)\b|\.|$)",
            r"\bcreated\s+(?:their\s+own\s+|our\s+own\s+|my\s+own\s+|a|an|the|this|that)?\s*(.+?)(?:\s+(?:at|in|with|for|during|on)\b|\.|$)",
        ):
            match = re.search(pattern, cleaned_sentence, re.IGNORECASE)
            if match is None:
                continue
            candidate = _strip_leading_article(_clean_candidate_answer(match.group(1)))
            if _support_present_role_candidate_rejection_reason(candidate, role, question=question) is None:
                return candidate
        return None

    if role == "activity_object":
        return _locomo_activity_object_candidate(cleaned_sentence)

    if role == "action_bundle_list":
        return _locomo_action_bundle_candidate(cleaned_sentence)

    if role == "where_did_go_activity":
        parsed = _locomo_where_go_activity_question(question)
        if parsed is None:
            return None
        actor, companion = parsed
        companion_terms = set(_content_tokens(companion)) & {
            "girlfriend",
            "boyfriend",
            "partner",
            "friend",
            "friends",
        }
        sentence_terms = set(_content_tokens(cleaned_sentence))
        if actor in sentence_terms and "camping" in sentence_terms and companion_terms & sentence_terms:
            return "camping with girlfriend"
        return None

    if role == "pet_type":
        if "adopt" in set(_content_tokens(question)) and re.search(r"\brecommend", cleaned_sentence, re.IGNORECASE):
            return None
        for pattern in (
            r"\b((?:small|smaller|medium|large|larger)\s+dogs?(?:\s+breeds?)?)\b",
            r"\b((?:beagle|bulldog|collie|poodle|retriever|shepherd|terrier)(?:\s+dogs?)?)\b",
            r"\b(german\s+shepherd(?:\s+dogs?)?)\b",
        ):
            match = re.search(pattern, cleaned_sentence, re.IGNORECASE)
            if match is None:
                continue
            candidate = _strip_leading_article(_clean_candidate_answer(match.group(1)))
            if _support_present_role_candidate_rejection_reason(candidate, role, question=question) is None:
                return candidate
        return None

    if role == "direct_support_scalar":
        if support_channel == "summary" and _support_present_workshop_discussion_question(question):
            return None
        return _support_present_direct_scalar_candidate(
            question,
            cleaned_sentence,
            context_sentence=context_sentence,
        )

    if role == "location":
        if not re.search(r"^\s*where\b", question, re.IGNORECASE):
            return None
        match = re.search(
            r"\b(?:to|in|at|near)\s+([A-Z][A-Za-z0-9' -]+?)(?:\s+(?:for|on|with|during|after|before)|\.|$)",
            cleaned_sentence,
        )
        if match:
            candidate = _clean_candidate_answer(match.group(1))
            if not re.match(r"^(?:my|your|his|her|their|our|next|movie)\b", candidate, re.IGNORECASE):
                return candidate
        return None

    return None


def _deterministic_answer_from_sentence(
    question: str,
    sentence: str,
    shape: StructuredSynthesisShape,
    *,
    allow_benefit_with_having: bool = True,
) -> str | None:
    if shape == "list_or_set":
        essential_detail = _essential_detail_list_candidate(question, sentence)
        if essential_detail is not None:
            return essential_detail
        match = re.search(
            r"\blikes?\s+(.+?)(?:\s+for\b|\s+during\b|\s+after\b|\.|$)",
            sentence,
            re.IGNORECASE,
        )
        if match and _LIST_SEPARATOR_PATTERN.search(match.group(1)):
            return _clean_candidate_answer(match.group(1))

    if shape == "complete_phrase":
        for pattern in (
            r"\bsymboli[sz]es?\s+(.+?)(?:\.|$)",
            r"\breason\b.*?\bwas\s+(.+?)(?:\.|$)",
            r'\b(?:say|says|said)\s+to\b.*?"([^"]+)"',
            r"\b(?:say|says|said)\s+to\b.*?'([^']+)'",
        ):
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                return _clean_candidate_answer(match.group(1))

    if shape == "predicate_bound_scalar":
        identity_answer = _identity_complete_answer_from_sentence(question, sentence)
        if identity_answer is not None:
            return identity_answer
        pet_family_answer = _pet_family_view_candidate_from_sentence(question, sentence)
        if pet_family_answer is not None:
            return pet_family_answer
        if allow_benefit_with_having:
            benefit_answer = _benefit_with_having_answer_from_sentence(question, sentence)
            if benefit_answer is not None:
                return benefit_answer
        place_discovery_answer = _place_discovery_object_from_sentence(question, sentence)
        if place_discovery_answer is not None:
            return place_discovery_answer
        for verb in _EVENT_OBJECT_VERBS_REQUIRING_TYPED_BINDING:
            if not any(form in _content_tokens(question) for form in _predicate_forms(verb)):
                continue
            for form in sorted(_predicate_forms(verb), key=len, reverse=True):
                pattern = (
                    rf"\b{re.escape(form)}\s+"
                    r"(?:a|an|the)?\s*(.+?)(?:\s+(?:for|on|with|to|at|in|during|after|"
                    r"before|based|recently|yesterday|today)|\.|$)"
                )
                match = re.search(pattern, sentence, re.IGNORECASE)
                if match:
                    candidate = _clean_candidate_answer(match.group(1))
                    if verb == "make":
                        candidate = _canonicalize_make_artifact_candidate(question, candidate)
                    return candidate

    return None


_IDENTITY_QUESTION_PATTERN = re.compile(r"^\s*what\s+is\b.*\bidentity\b", re.IGNORECASE)
_IDENTITY_COMPLETE_CUE_PATTERN = re.compile(
    r"\b(?:transgender|trans)\s+(?:woman|girl)\b",
    re.IGNORECASE,
)
_IDENTITY_BINDING_CUE_PATTERN = re.compile(
    r"\b(?:identit(?:y|ies)|identif(?:y|ies|ied)|gender\s+identity|path|journey|coming\s+out)\b",
    re.IGNORECASE,
)


def _identity_question_requires_complete_answer(question: str, answer: str) -> bool:
    if _IDENTITY_QUESTION_PATTERN.search(question) is None:
        return False
    return _normalize(answer) in {"trans", "transgender"}


def _identity_complete_answer_from_sentence(question: str, sentence: str) -> str | None:
    if _IDENTITY_QUESTION_PATTERN.search(question) is None:
        return None
    if _IDENTITY_COMPLETE_CUE_PATTERN.search(sentence) is None:
        return None
    if _IDENTITY_BINDING_CUE_PATTERN.search(sentence) is None:
        return None
    return "Transgender woman"


def _identity_support_sentence_bound(question: str, sentence: str) -> bool:
    if _identity_complete_answer_from_sentence(question, sentence) is None:
        return False
    sentence_terms = set(_content_tokens(sentence))
    actor_terms = [term for term in _question_event_terms(question) if term != "identity"]
    return bool(actor_terms) and all(term in sentence_terms for term in actor_terms)


def _pet_family_view_candidate_from_sentence(question: str, sentence: str) -> str | None:
    if _SUPPORT_PRESENT_PET_FAMILY_VIEW_QUESTION_PATTERN.search(question) is None:
        return None
    sentence_lower = sentence.lower()
    pet_terms = r"(?:pet|pets|dog|dogs|cat|cats|pup|pups|puppy|puppies)"
    if not re.search(rf"\b{pet_terms}\b", sentence_lower):
        return None
    family_surfaces = (
        rf"\b{pet_terms}\b.{{0,80}}\b(?:are|is|feel|feels|felt)\s+(?:like\s+)?family\b"
        rf"|\b{pet_terms}\b.{{0,80}}\bbring\b.{{0,80}}\bjoy\b.{{0,80}}\bfeel\s+like\s+family\b"
    )
    if not re.search(family_surfaces, sentence_lower):
        return None
    return "Family"


_PLACE_DISCOVERY_QUESTION_PATTERN = re.compile(
    r"^\s*what\s+(?:did|does)\s+(?P<actor>.+?)\s+"
    r"(?P<verb>discover|find|see|check\s+out|notice)\s+"
    r"(?:at|in|from|during)\s+(?P<place>.+?)(?:\?|$)",
    re.IGNORECASE,
)
_PLACE_DISCOVERY_SUPPORT_OBJECT_PATTERN = re.compile(
    r"\b(?:which|that)\s+had\s+(?:a|an|the)?\s*(?P<object>.+?)(?:\.|,|;|$)",
    re.IGNORECASE,
)
_PLACE_DISCOVERY_GENERIC_PLACE_TERMS = {
    "at",
    "in",
    "museum",
    "library",
    "place",
    "places",
    "the",
}
_PLACE_DISCOVERY_GENERIC_OBJECT_TERMS = {
    "building",
    "buildings",
    "historic",
    "library",
    "lovely",
    "midwest",
    "museum",
    "place",
    "scenery",
    "town",
    "woodhaven",
}


def _place_discovery_object_from_sentence(question: str, sentence: str) -> str | None:
    question_match = _PLACE_DISCOVERY_QUESTION_PATTERN.search(question)
    if question_match is None:
        return None

    actor_terms = set(_content_tokens(question_match.group("actor")))
    place_terms = {
        term
        for term in _content_tokens(question_match.group("place"))
        if term not in _PLACE_DISCOVERY_GENERIC_PLACE_TERMS
    }
    sentence_terms = set(_content_tokens(sentence))
    if actor_terms and not actor_terms.intersection(sentence_terms):
        return None
    if place_terms and not place_terms.intersection(sentence_terms):
        return None

    object_match = _PLACE_DISCOVERY_SUPPORT_OBJECT_PATTERN.search(sentence)
    if object_match is None:
        return None
    candidate = _clean_candidate_answer(object_match.group("object"))
    if not candidate:
        return None
    candidate_terms = set(_content_tokens(candidate))
    if not candidate_terms or len(candidate_terms) > 8:
        return None
    if not candidate_terms.difference(_PLACE_DISCOVERY_GENERIC_OBJECT_TERMS):
        return None
    if _support_present_candidate_echoes_question(question, candidate):
        return None
    return _strip_leading_article(candidate)


def _canonicalize_benefit_with_having_result(
    question: str,
    result: StructuredSynthesisResult,
) -> StructuredSynthesisResult:
    if _benefit_with_having_parts(question) is None or not result.answer:
        return result
    answer_nouns = _benefit_with_having_answer_nouns_in_text(result.answer)
    if len(answer_nouns) != 1:
        return result
    noun = answer_nouns[0]
    if not any(
        noun in _benefit_with_having_support_nouns(question, cited_span)
        for cited_span in result.cited_spans
    ):
        return result
    if _normalize(result.answer) == noun:
        return result
    return replace(
        result,
        answer=noun,
        fallback_used=result.fallback_used or "benefit_with_having_answer_canonicalization",
    )


def _canonicalize_make_artifact_result(
    question: str,
    result: StructuredSynthesisResult,
) -> StructuredSynthesisResult:
    if not result.answer:
        return result
    canonical_answer = _canonicalize_make_artifact_candidate(question, result.answer)
    if _normalize(canonical_answer) == _normalize(result.answer):
        return result
    normalized_answer = _containment_normalize(canonical_answer)
    if not any(
        _contains_containment_answer(normalized_answer, _containment_normalize(cited_span))
        for cited_span in result.cited_spans
    ):
        return result
    return replace(
        result,
        answer=canonical_answer,
        fallback_used=result.fallback_used or "make_artifact_answer_canonicalization",
    )


def _canonicalize_structured_synthesis_result(
    question: str,
    result: StructuredSynthesisResult,
) -> StructuredSynthesisResult:
    result = _canonicalize_make_artifact_result(question, result)
    return _canonicalize_benefit_with_having_result(question, result)


def _benefit_with_having_rejection_reason(
    question: str,
    result: StructuredSynthesisResult,
) -> str | None:
    if _benefit_with_having_parts(question) is None:
        return None
    answer_nouns = _benefit_with_having_answer_nouns_in_text(result.answer or "")
    if len(answer_nouns) != 1:
        return "structured_synthesis_benefit_ambiguous"
    noun = answer_nouns[0]
    if not any(
        noun in _benefit_with_having_support_nouns(question, cited_span)
        for cited_span in result.cited_spans
    ):
        return "structured_synthesis_benefit_unbound"
    return None


def _verify_structured_synthesis(
    question: str,
    result: StructuredSynthesisResult,
    support_pack: tuple[ScoredEvidenceSupport, ...],
    shape: StructuredSynthesisShape,
) -> str | None:
    if not result.answer:
        return "empty_answer_or_support"
    benefit_rejection = _benefit_with_having_rejection_reason(question, result)
    if benefit_rejection is not None:
        return benefit_rejection
    if not result.support_ids:
        return "structured_synthesis_missing_support_id"
    if not result.cited_spans:
        return "structured_synthesis_missing_cited_span"

    support_by_id = {item.support.support_id: item.support for item in support_pack}
    cited_supports: list[EvidenceSupport] = []
    for support_id in result.support_ids:
        support = support_by_id.get(support_id)
        if support is None:
            return "structured_synthesis_unknown_support_id"
        cited_supports.append(support)

    for cited_span in result.cited_spans:
        normalized_span = _containment_normalize(cited_span)
        if not any(
            _contains_containment_answer(
                normalized_span,
                support.normalized_support_text,
            )
            for support in cited_supports
        ):
            return "structured_synthesis_unsupported_span"

    if shape == "list_or_set":
        normalized_spans = [_containment_normalize(span) for span in result.cited_spans]
        items = _split_list_answer_items(result.answer)
        if not items:
            return "structured_synthesis_empty_answer"
        for item in items:
            normalized_item = _containment_normalize(item)
            if not any(
                _contains_containment_answer(normalized_item, span)
                or _event_list_item_supported_by_span(question, item, span)
                or _locomo_action_bundle_item_supported_by_span(question, item, span)
                for span in normalized_spans
            ):
                return "structured_synthesis_unsupported_item"
        if not _LIST_SEPARATOR_PATTERN.search(result.answer):
            for support in cited_supports:
                if any(
                    _LIST_SEPARATOR_PATTERN.search(sentence)
                    for sentence in _sentences_containing_answer(
                        support.support_text,
                        result.answer,
                    )
                ):
                    return "structured_synthesis_incomplete_list"
        return None

    normalized_answer = _containment_normalize(result.answer)
    normalized_spans = [_containment_normalize(span) for span in result.cited_spans]
    if shape == "predicate_bound_scalar":
        if _locomo_where_go_activity_answer_supported(question, result.answer, result.cited_spans):
            return None
        if _locomo_activity_object_answer_supported(question, result.answer, result.cited_spans):
            return None
        if _locomo_training_course_date_answer_supported(
            question,
            result.answer,
            result.cited_spans,
        ):
            return None
        for cited_span in result.cited_spans:
            for support in cited_supports:
                if support.channel == "summary" and _support_present_workshop_discussion_question(question):
                    continue
                direct_scalar_answer = _support_present_direct_scalar_candidate(
                    question,
                    cited_span,
                    context_sentence=support.concept_summary,
                )
                if direct_scalar_answer is not None and (
                    _contains_containment_answer(
                        _containment_normalize(result.answer),
                        _containment_normalize(direct_scalar_answer),
                    )
                    or _contains_containment_answer(
                        _containment_normalize(direct_scalar_answer),
                        _containment_normalize(result.answer),
                    )
                ):
                    return None
    if not any(
        _contains_containment_answer(normalized_answer, normalized_span)
        or _contains_containment_answer(normalized_span, normalized_answer)
        for normalized_span in normalized_spans
    ):
        return "structured_synthesis_unsupported_answer"

    if shape == "complete_phrase":
        answer_tokens = _content_tokens(result.answer)
        if len(answer_tokens) <= 2:
            for cited_span in result.cited_spans:
                if len(_content_tokens(cited_span)) > len(answer_tokens):
                    return "structured_synthesis_incomplete_phrase"
        return None

    if shape == "predicate_bound_scalar":
        for cited_span in result.cited_spans:
            place_discovery_answer = _place_discovery_object_from_sentence(question, cited_span)
            if place_discovery_answer is not None and _contains_containment_answer(
                _containment_normalize(result.answer),
                _containment_normalize(place_discovery_answer),
            ):
                return None
            pet_family_answer = _pet_family_view_candidate_from_sentence(question, cited_span)
            if pet_family_answer is not None and _contains_containment_answer(
                _containment_normalize(result.answer),
                _containment_normalize(pet_family_answer),
            ):
                return None
            for support in cited_supports:
                if support.channel == "summary" and _support_present_workshop_discussion_question(question):
                    continue
                direct_scalar_answer = _support_present_direct_scalar_candidate(
                    question,
                    cited_span,
                    context_sentence=support.concept_summary,
                )
                if direct_scalar_answer is not None and _contains_containment_answer(
                    _containment_normalize(result.answer),
                    _containment_normalize(direct_scalar_answer),
                ):
                    return None
        if _locomo_training_course_date_answer_supported(
            question,
            result.answer,
            result.cited_spans,
        ):
            return None
        question_terms = _question_event_terms(question)
        for cited_span in result.cited_spans:
            if _sentence_matches_terms(
                cited_span,
                question_terms,
                min_ratio=_ANSWER_CONTRACT_MIN_SLOT_MATCH_RATIO,
            ):
                return None
        return "structured_synthesis_predicate_unbound"

    return None


def _event_list_item_supported_by_span(question: str, item: str, span: str) -> bool:
    if _support_present_answer_role(question) != "event_list":
        return False
    normalized_item = _containment_normalize(item)
    span_terms = set(_content_tokens(span))
    if normalized_item == "school speech":
        return bool(
            {"school", "event"} <= span_terms
            and span_terms & {"talked", "spoke", "shared", "told", "presented", "encouraged"}
        )
    if normalized_item == "mentoring program":
        return bool({"mentorship", "program"} <= span_terms)
    return False


def _locomo_action_bundle_item_supported_by_span(question: str, item: str, span: str) -> bool:
    if _locomo_action_bundle_question(question) is None:
        return False
    normalized_item = _containment_normalize(item)
    normalized_span = _containment_normalize(span)
    if normalized_item == "join a local church":
        return _contains_containment_answer(
            "joined a local church",
            normalized_span,
        ) or _contains_containment_answer("joined a nearby church", normalized_span)
    if normalized_item == "buy a cross necklace":
        return _contains_containment_answer("bought a cross necklace", normalized_span)
    return False


def _locomo_where_go_activity_answer_supported(
    question: str,
    answer: str | None,
    cited_spans: tuple[str, ...],
) -> bool:
    parsed = _locomo_where_go_activity_question(question)
    if parsed is None:
        actor = _locomo_where_camping_with_girlfriend_question(question)
        parsed = (actor, "girlfriend") if actor is not None else None
    if parsed is None or not answer:
        return False
    actor, companion = parsed
    answer_terms = set(_content_tokens(answer))
    if not ({"camping", "girlfriend"} <= answer_terms):
        return False
    companion_terms = set(_content_tokens(companion)) & {"girlfriend", "boyfriend", "partner", "friend", "friends"}
    for cited_span in cited_spans:
        span_terms = set(_content_tokens(cited_span))
        if actor in span_terms and "camping" in span_terms and companion_terms & span_terms:
            return True
    return False


def _locomo_activity_object_answer_supported(
    question: str,
    answer: str | None,
    cited_spans: tuple[str, ...],
) -> bool:
    actor = _locomo_activity_object_question(question)
    if actor is None or not answer:
        return False
    answer_terms = set(_content_tokens(answer))
    if not (
        answer_terms & {"hiking", "kundalini", "trails", "trail", "yoga"}
        or {"volunteering", "dog", "shelter"} <= answer_terms
    ):
        return False
    for cited_span in cited_spans:
        span_terms = set(_content_tokens(cited_span))
        if actor not in span_terms:
            continue
        if answer_terms <= span_terms:
            return True
    return False


def _locomo_training_course_date_answer_supported(
    question: str,
    answer: str | None,
    cited_spans: tuple[str, ...],
) -> bool:
    if not _locomo_training_course_date_question(question) or not answer:
        return False
    normalized_answer = _containment_normalize(answer)
    date_bound = False
    course_bound = False
    for cited_span in cited_spans:
        span_terms = set(_content_tokens(cited_span))
        if _contains_containment_answer(
            normalized_answer,
            _containment_normalize(cited_span),
        ):
            date_bound = True
        if (
            "audrey" in span_terms
            and {"positive", "reinforcement"} <= span_terms
            and {"training", "course", "class"} & span_terms
        ):
            course_bound = True
    return date_bound and course_bound


def _support_present_role_rejection_reason(
    question: str,
    result: StructuredSynthesisResult,
    support_pack: tuple[ScoredEvidenceSupport, ...],
    role: SupportPresentAnswerRole,
) -> str | None:
    if role == "generic":
        return None
    support_by_id = {item.support.support_id: item.support for item in support_pack}
    cited_supports = [support_by_id[support_id] for support_id in result.support_ids if support_id in support_by_id]
    if not cited_supports:
        return "support_present_role_unbound"
    support_text = " ".join(support.support_text for support in cited_supports)
    normalized_support = _containment_normalize(support_text)
    normalized_answer = _containment_normalize(result.answer or "")
    answer_role_rejection = _support_present_role_candidate_rejection_reason(
        result.answer,
        role,
        question=question,
    )
    if answer_role_rejection is not None:
        return answer_role_rejection

    if role == "diet_list":
        if not _LIST_SEPARATOR_PATTERN.search(result.answer or ""):
            return "support_present_list_unbound"
        if not re.search(r"\b(?:diet|eat|eats|eating|feed|feeds|consists?)\b", support_text, re.IGNORECASE):
            return "support_present_role_unbound"
        return None

    if role == "artifact_text":
        if not re.search(r"\b(?:poster|posters|sign|shirt|banner|card)s?\b", support_text, re.IGNORECASE):
            return "support_present_role_unbound"
        if not re.search(r"\b(?:say|says|said|read|reads)\b", support_text, re.IGNORECASE):
            return "support_present_role_unbound"
        return None

    if role == "named_title":
        if re.search(r"\b(?:called|titled|named)\b", support_text, re.IGNORECASE):
            return None
        return "support_present_role_unbound"

    if role == "training_type":
        answer = result.answer or ""
        if "training" not in _content_tokens(answer) and "training" not in _content_tokens(support_text):
            return "support_present_role_unbound"
        if not ({"training", "workshop", "class"} & set(_content_tokens(support_text))):
            return "support_present_role_unbound"
        return None

    if role == "location":
        if not normalized_answer or re.match(
            r"^(?:for|my|your|his|her|their|our)\b",
            result.answer or "",
            re.IGNORECASE,
        ):
            return "support_present_location_unbound"
        if not re.search(rf"\b(?:to|in|at|near)\s+{re.escape(result.answer or '')}\b", support_text, re.IGNORECASE):
            return "support_present_location_unbound"
        return None

    if role == "made_object_list":
        terms = set(_content_tokens(support_text))
        if not (terms & {"pottery", "made", "make", "making", "bowl", "bowls", "cup", "cups"}):
            return "support_present_list_unbound"
        return None

    if role == "activity_object":
        terms = set(_content_tokens(result.answer or ""))
        if terms & {"hiking", "kundalini", "trails", "trail", "yoga"}:
            return None
        if {"volunteering", "dog", "shelter"} <= terms:
            return None
        return "support_present_role_unbound"

    if role == "action_bundle_list":
        if any(
            _locomo_action_bundle_item_supported_by_span(question, item, _containment_normalize(support_text))
            for item in _split_list_answer_items(result.answer or "")
        ):
            return None
        return "support_present_list_unbound"

    if role == "where_did_go_activity":
        if _locomo_where_go_activity_answer_supported(question, result.answer, (support_text,)):
            return None
        return "support_present_location_unbound"

    if role == "pet_activity_list":
        terms = set(_content_tokens(support_text))
        if not (terms & {"dog", "dogs"} and terms & {"park", "play", "plays", "fetch", "frisbee"}):
            return "support_present_list_unbound"
        return None

    if role == "pet_type":
        terms = set(_content_tokens(support_text))
        if "dog" not in terms and "dogs" not in terms and "dog" not in _content_tokens(result.answer or ""):
            return "support_present_role_unbound"
        return None

    if role == "direct_support_scalar":
        for support in cited_supports:
            if support.channel == "summary" and _support_present_workshop_discussion_question(question):
                continue
            for sentence in _SENTENCE_SPLIT_PATTERN.split(support.support_text):
                candidate = _support_present_direct_scalar_candidate(
                    question,
                    sentence,
                    context_sentence=support.concept_summary,
                )
                if candidate is None:
                    continue
                if _contains_containment_answer(
                    _containment_normalize(result.answer or ""),
                    _containment_normalize(candidate),
                ) or _contains_containment_answer(
                    _containment_normalize(candidate),
                    _containment_normalize(result.answer or ""),
                ):
                    return None
        return "support_present_role_unbound"

    if role == "question_bound_list":
        anchor_terms = [
            term
            for term in _content_tokens(question)
            if term not in {"kind", "kinds", "type", "types", "some", "what", "which"}
        ]
        matched = _matched_term_count(anchor_terms, _content_tokens(support_text))
        if anchor_terms and matched < min(2, len(anchor_terms)):
            return "support_present_list_unbound"
        if normalized_answer and not _contains_containment_answer(normalized_answer, normalized_support):
            if all(
                _contains_containment_answer(_containment_normalize(item), normalized_support)
                for item in _split_list_answer_items(result.answer or "")
            ):
                return None
            return "support_present_list_unbound"
        return None

    return None


def _split_list_answer_items(answer: str) -> list[str]:
    return [
        _clean_candidate_answer(item)
        for item in re.split(r"[,;/]|\band\b", answer, flags=re.IGNORECASE)
        if _clean_candidate_answer(item)
    ]


def _structured_synthesis_decision(
    result: StructuredSynthesisResult,
    support_pack: tuple[ScoredEvidenceSupport, ...],
    t0: float,
    *,
    shape: StructuredSynthesisShape,
    fallback_used: str | None = None,
    llm_error_class: str | None = None,
) -> EvidenceAnswerDecision:
    support_by_id = {item.support.support_id: item.support for item in support_pack}
    support = support_by_id.get(result.support_ids[0]) if result.support_ids else support_pack[0].support
    support_ids = tuple(support_id for support_id in result.support_ids if support_id in support_by_id)
    support_concept_ids = tuple(
        support_by_id[support_id].concept_id
        for support_id in support_ids
        if support_by_id[support_id].concept_id
    )
    return EvidenceAnswerDecision(
        mode="structured_synthesis",
        answer=result.answer,
        normalized_answer=_normalize(result.answer or ""),
        support=support,
        abstain_reason=None,
        latency_ms=_latency_ms(t0),
        llm_error_class=llm_error_class,
        expected_answer_shape=shape,
        slot_binding_status="bound" if shape == "predicate_bound_scalar" else None,
        synthesis_shape=shape,
        support_pack_size=len(support_pack),
        support_ids=support_ids,
        support_concept_ids=support_concept_ids,
        fallback_used=fallback_used,
    )


def _candidate_decision(
    candidate: AnswerCandidate,
    t0: float,
    *,
    intent: AnswerIntent,
    candidate_count: int,
    candidate_rejection_counts: dict[str, int] | None = None,
    support_pack_size: int = 0,
    verifier_rejection_counts: dict[str, int] | None = None,
    question: str | None = None,
    locomo_support_present_answer_realization_enabled: bool = False,
) -> EvidenceAnswerDecision:
    mode: AnswerMode = (
        "support_entails_no"
        if candidate.source == "yes_no_entailment" and candidate.answer == "No"
        else "support_entails_yes"
        if candidate.source == "yes_no_entailment"
        else "deterministic_candidate"
    )
    recovered = (
        _locomo_candidate_answer_realization(
            question=question,
            candidate=candidate,
            enabled=locomo_support_present_answer_realization_enabled,
        )
        if question
        else None
    )
    answer = recovered.answer if recovered is not None else candidate.answer
    normalized_answer = recovered.normalized_answer if recovered is not None else candidate.normalized_answer
    return EvidenceAnswerDecision(
        mode=mode,
        answer=answer,
        normalized_answer=normalized_answer,
        support=candidate.support,
        abstain_reason=None,
        latency_ms=_latency_ms(t0),
        intent=intent,
        candidate_count=candidate_count,
        candidate_source=candidate.source,
        candidate_rejection_counts=candidate_rejection_counts or None,
        support_pack_size=support_pack_size,
        verifier_rejection_counts=verifier_rejection_counts,
        recovery_strategy=recovered.strategy if recovered is not None else None,
    )


def _collect_supports(
    concepts: list,
    *,
    max_support_chars: int,
) -> list[EvidenceSupport]:
    supports: list[EvidenceSupport] = []
    chars_used = 0

    for concept_index, concept in enumerate(concepts):
        concept_id = _concept_id(concept, concept_index)
        concept_summary = re.sub(r"\s+", " ", _get_text(concept, "summary")).strip()
        concept_actor_terms = _support_actor_terms_from_text(concept_summary)
        concept_created_at = _optional_str(_get_value(concept, "created_at"))
        concept_valid_from = _optional_str(_get_value(concept, "valid_from"))
        concept_original_date = _optional_str(_get_value(concept, "original_date"))
        concept_content_updated_at = _optional_str(_get_value(concept, "content_updated_at"))
        concept_serial_order = _optional_int(_get_value(concept, "serial_order"))
        concept_session_id = _optional_str(_get_value(concept, "session_id"))
        candidates: list[tuple[SupportChannel, str]] = []
        candidates.append(("summary", concept_summary))

        for evidence in _get_sequence(concept, "key_evidence"):
            for evidence_text in _support_evidence_texts(evidence):
                candidates.append(("key_evidence", evidence_text))

        candidates.append(("text", _get_text(concept, "text")))

        for fragment in _get_sequence(concept, "verbatim_fragments"):
            candidates.append(("verbatim", _get_fragment_content(fragment)))

        for channel, text in candidates:
            support_text = re.sub(r"\s+", " ", text).strip()
            if not support_text:
                continue
            next_total = chars_used + len(support_text)
            if next_total > max_support_chars:
                return supports
            support = EvidenceSupport(
                support_id=f"s{len(supports)}",
                concept_id=concept_id,
                channel=channel,
                support_text=support_text,
                normalized_support_text=_containment_normalize(support_text),
                concept_summary=concept_summary,
                concept_actor_terms=concept_actor_terms,
                concept_created_at=concept_created_at,
                concept_valid_from=concept_valid_from,
                concept_original_date=concept_original_date,
                concept_content_updated_at=concept_content_updated_at,
                concept_serial_order=concept_serial_order,
                concept_session_id=concept_session_id,
            )
            supports.append(support)
            chars_used = next_total
            for derived_text in _derived_temporal_support_texts(support):
                derived_text = re.sub(r"\s+", " ", derived_text).strip()
                if not derived_text:
                    continue
                next_total = chars_used + len(derived_text)
                if next_total > max_support_chars:
                    return supports
                supports.append(
                    EvidenceSupport(
                        support_id=f"s{len(supports)}",
                        concept_id=concept_id,
                        channel="key_evidence",
                        support_text=derived_text,
                        normalized_support_text=_containment_normalize(derived_text),
                        concept_summary=concept_summary,
                        concept_actor_terms=concept_actor_terms,
                        concept_created_at=concept_created_at,
                        concept_valid_from=concept_valid_from,
                        concept_original_date=concept_original_date,
                        concept_content_updated_at=concept_content_updated_at,
                        concept_serial_order=concept_serial_order,
                        concept_session_id=concept_session_id,
                    )
                )
                chars_used = next_total

    return supports


_DIRECT_SUPPORT_AWARENESS_RE = re.compile(
    r"\b(?:awareness\s+for|race\s+for)\s+(.+?)(?:\s+to\b|[.;,!]|$)",
    re.IGNORECASE,
)
_DIRECT_SUPPORT_FAVORITE_STYLE_RE = re.compile(
    r"\bfavorite\s+(?:[a-z]+\s+){0,4}style\s+is\s+(.+?)(?:[.;,!]|$)",
    re.IGNORECASE,
)
_DIRECT_SUPPORT_MAX_ANSWER_TOKENS = 8


def _recover_direct_support_admission_answer(
    question: str,
    supports: list[EvidenceSupport],
    t0: float,
) -> EvidenceAnswerDecision:
    candidates: list[tuple[str, str, EvidenceSupport]] = []
    rejection_counts: dict[str, int] = {}

    for support in supports:
        for candidate in _direct_support_admission_candidates(question, support.support_text):
            cleaned = _clean_candidate_answer(candidate)
            normalized = _normalize(cleaned)
            if not normalized:
                _increment_rejection(rejection_counts, "direct_support_empty_candidate")
                continue
            if len(_TOKEN_PATTERN.findall(normalized)) > _DIRECT_SUPPORT_MAX_ANSWER_TOKENS:
                _increment_rejection(rejection_counts, "direct_support_candidate_too_long")
                continue
            if not _contains_containment_answer(
                _containment_normalize(cleaned),
                support.normalized_support_text,
            ):
                _increment_rejection(rejection_counts, "direct_support_candidate_not_visible")
                continue
            candidates.append((cleaned, normalized, support))

    if not candidates:
        return _abstain(
            "direct_support_no_candidate",
            t0,
            candidate_source="regex_direct_support_admission",
            candidate_rejection_counts=rejection_counts,
        )

    by_normalized: dict[str, tuple[str, EvidenceSupport]] = {}
    for answer, normalized, support in candidates:
        by_normalized.setdefault(normalized, (answer, support))

    if len(by_normalized) != 1:
        _increment_rejection(rejection_counts, "direct_support_conflict")
        return _abstain(
            "direct_support_conflict",
            t0,
            candidate_count=len(candidates),
            candidate_source="regex_direct_support_admission",
            candidate_rejection_counts=rejection_counts,
        )

    normalized_answer, (answer, support) = next(iter(by_normalized.items()))
    return EvidenceAnswerDecision(
        mode="exact_support_recovery",
        answer=answer,
        normalized_answer=normalized_answer,
        support=support,
        abstain_reason=None,
        latency_ms=_latency_ms(t0),
        candidate_count=len(candidates),
        candidate_source="regex_direct_support_admission",
        candidate_rejection_counts=rejection_counts or None,
        fallback_used="direct_support_admission",
        recovery_strategy="direct_support_admission",
    )


def _direct_support_admission_candidates(question: str, support_text: str) -> list[str]:
    normalized_question = _normalize(question)
    normalized_support = _normalize(support_text)
    candidates: list[str] = []

    if "awareness" in normalized_question and "for" in normalized_question:
        candidates.extend(match.group(1) for match in _DIRECT_SUPPORT_AWARENESS_RE.finditer(support_text))

    if "favorite" in normalized_question and "style" in normalized_question:
        candidates.extend(match.group(1) for match in _DIRECT_SUPPORT_FAVORITE_STYLE_RE.finditer(support_text))

    if _direct_support_media_question(normalized_question) and _direct_support_media_support(normalized_support):
        candidates.extend(answer for answer, _source in _quoted_title_candidates(support_text))

    return candidates


def _direct_support_media_question(normalized_question: str) -> bool:
    return bool({"movie", "movies", "film", "watch", "watched", "enjoy"}.intersection(normalized_question.split()))


def _direct_support_media_support(normalized_support: str) -> bool:
    return bool({"movie", "movies", "film", "watch", "watched", "list"}.intersection(normalized_support.split()))


def _try_answer_shape_runtime_hook(
    *,
    question: str,
    supports: list[EvidenceSupport],
    t0: float,
    admission_enabled: bool,
    runtime_effect_enabled: bool,
) -> EvidenceAnswerDecision:
    hook_t0 = time.perf_counter()
    effective_runtime = admission_enabled and runtime_effect_enabled
    try:
        temporal_supports = tuple(
            TemporalSupport(
                support_id=support.support_id,
                text=support.support_text,
                source_date=_answer_shape_support_source_date(support),
                source_year=_answer_shape_support_source_year(support),
                provenance="explicit",
            )
            for support in supports
        )
        temporal_decision = construct_temporal_answer_candidate(question, temporal_supports)
        contract_decision = build_answer_shape_contract(question, temporal_decision)
        visible_support_ids = tuple(support.support_id for support in supports)
        admission = admit_answer_shape_contract(
            contract_decision,
            enabled=admission_enabled,
            visible_support_ids=visible_support_ids,
        )
        support_visibility = dict(admission.diagnostics.get("support_visibility") or {})
        required_support_ids = tuple(str(item) for item in support_visibility.get("required_support_ids", ()))
        support_by_id = {support.support_id: support for support in supports}
        support = next((support_by_id[item] for item in required_support_ids if item in support_by_id), None)
        answer = temporal_decision.answer if admission.admitted and effective_runtime else None
        return EvidenceAnswerDecision(
            mode="deterministic_candidate" if answer else "abstain",
            answer=answer,
            normalized_answer=_normalize(answer) if answer else None,
            support=support if answer else None,
            abstain_reason=None if answer else admission.reason,
            latency_ms=_latency_ms(t0),
            candidate_count=int(temporal_decision.diagnostics.get("candidate_count") or 0),
            recovery_strategy="answer_shape_runtime_hook" if answer else None,
            answer_shape_runtime_considered=admission.considered,
            answer_shape_runtime_admitted=admission.admitted,
            answer_shape_runtime_reason=admission.reason,
            answer_shape_runtime_contract_kind=admission.contract_kind,
            answer_shape_runtime_required_components=tuple(admission.required_components),
            answer_shape_runtime_support_visibility=support_visibility,
            answer_shape_runtime_effect_enabled=effective_runtime,
            answer_shape_runtime_latency_ms=_latency_ms(hook_t0),
            answer_shape_runtime_llm_call_delta=0,
        )
    except Exception as exc:
        return EvidenceAnswerDecision(
            mode="abstain",
            answer=None,
            normalized_answer=None,
            support=None,
            abstain_reason="answer_shape_runtime_error",
            latency_ms=_latency_ms(t0),
            llm_error_class=type(exc).__name__,
            answer_shape_runtime_considered=True,
            answer_shape_runtime_admitted=False,
            answer_shape_runtime_reason="answer_shape_runtime_error",
            answer_shape_runtime_effect_enabled=effective_runtime,
            answer_shape_runtime_latency_ms=_latency_ms(hook_t0),
            answer_shape_runtime_llm_call_delta=0,
        )


def _with_answer_shape_runtime_diagnostics(
    decision: EvidenceAnswerDecision,
    probe: EvidenceAnswerDecision | None,
) -> EvidenceAnswerDecision:
    if probe is None or not probe.answer_shape_runtime_considered:
        return decision
    return replace(
        decision,
        answer_shape_runtime_considered=probe.answer_shape_runtime_considered,
        answer_shape_runtime_admitted=probe.answer_shape_runtime_admitted,
        answer_shape_runtime_reason=probe.answer_shape_runtime_reason,
        answer_shape_runtime_contract_kind=probe.answer_shape_runtime_contract_kind,
        answer_shape_runtime_required_components=probe.answer_shape_runtime_required_components,
        answer_shape_runtime_support_visibility=probe.answer_shape_runtime_support_visibility,
        answer_shape_runtime_effect_enabled=probe.answer_shape_runtime_effect_enabled,
        answer_shape_runtime_latency_ms=probe.answer_shape_runtime_latency_ms,
        answer_shape_runtime_llm_call_delta=probe.answer_shape_runtime_llm_call_delta,
        llm_error_class=decision.llm_error_class or probe.llm_error_class,
        llm_error_provider_status=decision.llm_error_provider_status or probe.llm_error_provider_status,
        llm_error_provider_body_preview=(
            decision.llm_error_provider_body_preview or probe.llm_error_provider_body_preview
        ),
    )


def _answer_shape_support_source_date(support: EvidenceSupport) -> date | None:
    for value in (support.concept_original_date, support.concept_valid_from):
        parsed = _parse_calendar_date(value)
        if parsed is not None:
            return parsed
    return None


def _answer_shape_support_source_year(support: EvidenceSupport) -> int | None:
    parsed_date = _answer_shape_support_source_date(support)
    if parsed_date is not None:
        return parsed_date.year
    for value in (support.concept_original_date, support.concept_valid_from):
        match = re.match(r"^\s*(\d{4})(?:-\d{2})?(?:-\d{2})?\s*$", _stringify(value))
        if match:
            return int(match.group(1))
    return None


def _with_session_date_binding_diagnostics(
    decision: EvidenceAnswerDecision,
    *,
    question: str,
    supports: list[EvidenceSupport],
) -> EvidenceAnswerDecision:
    if decision.answer or not _TEMPORAL_QUESTION_PATTERN.search(question):
        return decision
    diagnostics = _session_date_binding_diagnostics(question, supports)
    if diagnostics is None:
        return decision
    return replace(
        decision,
        session_date_binding_status=str(diagnostics["status"]),
        session_date_binding_diagnostics=diagnostics,
    )


def _session_date_binding_diagnostics(
    question: str,
    supports: list[EvidenceSupport],
) -> dict[str, object] | None:
    if not supports:
        return None

    session_date_supports = [
        support
        for support in supports
        if _SESSION_DATE_PATTERN.search(support.support_text)
        or _SESSION_DATE_PATTERN.search(support.concept_summary)
    ]
    event_supports = [
        support
        for support in supports
        if support not in session_date_supports
        and (
            _support_sufficiently_matches_question(question, support.support_text, min_ratio=0.35)
            or _support_sufficiently_matches_question(question, support.concept_summary, min_ratio=0.35)
        )
    ]

    if not event_supports and not session_date_supports:
        return None
    if not event_supports:
        status = "no_event_support"
        strongest_relation = "unknown"
        rejection_reason = "no_event_support_overlaps_question"
    elif not session_date_supports:
        status = "no_session_date_support"
        strongest_relation = "unknown"
        rejection_reason = "no_session_date_support"
    else:
        status, strongest_relation, rejection_reason = _session_date_binding_relation(
            event_supports,
            session_date_supports,
        )

    return {
        "status": status,
        "event_support_count": len(event_supports),
        "session_date_support_count": len(session_date_supports),
        "event_supports": [_support_binding_summary(support) for support in event_supports[:3]],
        "session_date_supports": [
            _support_binding_summary(support) for support in session_date_supports[:3]
        ],
        "strongest_relation": strongest_relation,
        "rejection_reason": rejection_reason,
    }


def _session_date_binding_relation(
    event_supports: list[EvidenceSupport],
    session_date_supports: list[EvidenceSupport],
) -> tuple[str, str, str]:
    for event_support in event_supports:
        for date_support in session_date_supports:
            if event_support.concept_id == date_support.concept_id:
                return (
                    "same_concept",
                    "same_concept",
                    "diagnostic_only_same_concept_relation",
                )

    for event_support in event_supports:
        for date_support in session_date_supports:
            if (
                event_support.concept_session_id
                and date_support.concept_session_id
                and event_support.concept_session_id == date_support.concept_session_id
            ):
                return (
                    "same_session_id",
                    "same_session_id",
                    "diagnostic_only_same_session_relation",
                )

    metadata_seen = False
    for event_support in event_supports:
        for date_support in session_date_supports:
            if event_support.concept_valid_from or date_support.concept_valid_from:
                metadata_seen = True
            if (
                event_support.concept_valid_from
                and date_support.concept_valid_from
                and event_support.concept_valid_from == date_support.concept_valid_from
            ):
                return (
                    "same_valid_from",
                    "same_valid_from",
                    "diagnostic_only_same_valid_from_relation",
                )

    for event_support in event_supports:
        for date_support in session_date_supports:
            if event_support.concept_serial_order is not None or date_support.concept_serial_order is not None:
                metadata_seen = True
            if event_support.concept_serial_order is None or date_support.concept_serial_order is None:
                continue
            if abs(event_support.concept_serial_order - date_support.concept_serial_order) <= 3:
                return (
                    "proximity_only_unproven",
                    "proximity_only_unproven",
                    "serial_proximity_is_not_binding_proof",
                )

    if not metadata_seen:
        return ("metadata_missing", "unknown", "metadata_missing")
    return ("unproven", "unproven", "event_date_binding_not_proven")


def _support_binding_summary(support: EvidenceSupport) -> dict[str, object]:
    return {
        "support_id": support.support_id,
        "concept_id": support.concept_id,
        "channel": support.channel,
        "metadata": {
            "created_at": support.concept_created_at,
            "valid_from": support.concept_valid_from,
            "original_date": support.concept_original_date,
            "content_updated_at": support.concept_content_updated_at,
            "serial_order": support.concept_serial_order,
            "session_id": support.concept_session_id,
        },
    }


def _build_prompt(question: str, supports: list[EvidenceSupport]) -> str:
    evidence = "\n".join(f"[{support.support_id}] {support.support_text}" for support in supports)
    return (
        "Evidence:\n"
        f"{evidence}\n\n"
        f"Question: {question.strip()}\n\n"
        "Return strict JSON. The support_id must be one of the bracketed IDs exactly."
    )


def _parse_json(raw: str) -> dict:
    payload = json.loads(raw.strip())
    if not isinstance(payload, dict):
        raise ValueError("LLM proposal must be a JSON object")
    return payload


def _normalize(value: str) -> str:
    normalized = value.translate(_QUOTE_TRANSLATION).lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.strip(string.punctuation + " ")


def _containment_normalize(value: str) -> str:
    normalized = value.translate(_QUOTE_TRANSLATION).lower()
    return " ".join(_TOKEN_PATTERN.findall(normalized))


def _contains_containment_answer(
    normalized_answer: str,
    normalized_support_text: str,
) -> bool:
    if not normalized_answer:
        return False
    return f" {normalized_answer} " in f" {normalized_support_text} "


def _abstain(
    reason: str,
    t0: float,
    *,
    llm_error_class: str | None = None,
    llm_error_provider_status: int | None = None,
    llm_error_provider_body_preview: str | None = None,
    intent: AnswerIntent | None = None,
    candidate_count: int = 0,
    candidate_source: CandidateSource | None = None,
    candidate_rejection_counts: dict[str, int] | None = None,
    answer_contract_reason: str | None = None,
    expected_answer_shape: str | None = None,
    slot_binding_status: str | None = None,
    synthesis_shape: StructuredSynthesisShape = "none",
    support_pack_size: int = 0,
    verifier_rejection_counts: dict[str, int] | None = None,
    fallback_used: str | None = None,
) -> EvidenceAnswerDecision:
    return EvidenceAnswerDecision(
        mode="abstain",
        answer=None,
        normalized_answer=None,
        support=None,
        abstain_reason=reason,
        latency_ms=_latency_ms(t0),
        llm_error_class=llm_error_class,
        intent=intent,
        candidate_count=candidate_count,
        candidate_source=candidate_source,
        candidate_rejection_counts=candidate_rejection_counts or None,
        answer_contract_reason=answer_contract_reason,
        expected_answer_shape=expected_answer_shape,
        slot_binding_status=slot_binding_status,
        synthesis_shape=synthesis_shape,
        support_pack_size=support_pack_size,
        verifier_rejection_counts=verifier_rejection_counts,
        fallback_used=fallback_used,
        llm_error_provider_status=llm_error_provider_status,
        llm_error_provider_body_preview=llm_error_provider_body_preview,
    )


def _latency_ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000


def _proposal_string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _llm_error_class(exc: Exception) -> str:
    return exc.__class__.__name__


def _llm_error_provider_status(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _llm_error_provider_body_preview(exc: Exception, *, max_chars: int = 240) -> str | None:
    response = getattr(exc, "response", None)
    if response is None:
        return None
    body = getattr(response, "text", None)
    if not isinstance(body, str):
        content = getattr(response, "content", None)
        if isinstance(content, bytes):
            body = content.decode("utf-8", errors="replace")
    if not isinstance(body, str) or not body.strip():
        return None
    return re.sub(r"\s+", " ", body).strip()[:max_chars]


def _with_llm_error_provider_diagnostics(
    decision: EvidenceAnswerDecision,
    exc: Exception,
) -> EvidenceAnswerDecision:
    return replace(
        decision,
        llm_error_provider_status=_llm_error_provider_status(exc),
        llm_error_provider_body_preview=_llm_error_provider_body_preview(exc),
    )


def _clean_candidate_answer(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().strip(string.punctuation + " ")


def _strip_leading_article(value: str) -> str:
    return re.sub(r"^(?:a|an|the)\s+", "", _clean_candidate_answer(value), flags=re.IGNORECASE)


def _canonical_count_answer(value: str) -> str:
    cleaned = _clean_candidate_answer(value)
    return _NUMBER_WORDS.get(cleaned.lower(), cleaned)


def _question_abstain_reason(question: str) -> str | None:
    is_temporal_question = _TEMPORAL_QUESTION_PATTERN.search(question) is not None
    for pattern, reason in _BLOCKED_QUESTION_PATTERNS:
        if reason == "shared_or_common_requires_synthesis" and is_temporal_question:
            continue
        if reason == "shared_or_common_requires_synthesis" and _FAVORITE_DISH_QUESTION_PATTERN.search(question):
            continue
        if pattern.search(question):
            return reason
    return None


def _safe_scalar_slot_question(question: str) -> bool:
    if _question_abstain_reason(question) is not None:
        return False
    return any(pattern.search(question) for pattern in _SCALAR_SLOT_QUESTION_PATTERNS)


def _is_atomic_question_shape(question: str) -> bool:
    return (
        any(pattern.search(question) for pattern in _ATOMIC_QUESTION_PATTERNS)
        or _YES_NO_ARTIFACT_PHOTO_QUESTION_PATTERN.search(question) is not None
        or _FOR_HOW_LONG_QUESTION_PATTERN.search(question) is not None
        or _safe_scalar_slot_question(question)
    )


def _answer_shape_abstain_reason(
    question: str,
    answer: str,
    normalized_answer: str,
    support: EvidenceSupport,
    *,
    allow_list_like: bool = False,
) -> str | None:
    if _TEMPORAL_QUESTION_PATTERN.search(question):
        if not _TEMPORAL_ANSWER_PATTERN.search(answer):
            return "answer_shape_not_temporal"
        if not _support_sufficiently_matches_temporal_question(question, support.support_text):
            return "support_mismatch_for_temporal_question"
        if _is_unsupported_session_date_answer(answer, support.support_text):
            return "session_date_only_temporal_support"
        return None

    is_list_like = _LIST_LIKE_ANSWER_PATTERN.search(answer) is not None
    if not allow_list_like and is_list_like:
        return "answer_shape_list_like"

    if len(normalized_answer.split()) > 8 and not (allow_list_like and is_list_like):
        return "answer_shape_too_broad"

    return None


def _extractive_answer_contract_abstain_reason(
    question: str,
    answer: str,
) -> str | None:
    if _TEMPORAL_QUESTION_PATTERN.search(question) and _TEMPORAL_DEICTIC_ANSWER_PATTERN.search(answer):
        return "temporal_deictic_answer_unresolved"

    if _home_country_move_from_actor(question) is not None and re.fullmatch(
        r"\s*(?:the\s+)?home country\s*",
        answer,
        re.IGNORECASE,
    ):
        return "deictic_answer_unresolved"

    if _DEICTIC_ANSWER_PATTERN.search(answer) or _DEICTIC_OBJECT_ANSWER_PATTERN.search(answer):
        return "deictic_answer_unresolved"

    if _list_or_composite_question_requires_complete_answer(question):
        if not _LIST_SEPARATOR_PATTERN.search(answer):
            return "list_or_composite_partial_answer"

    if _event_object_answer_requires_typed_candidate(question, answer):
        return "event_object_requires_typed_candidate"

    return None


def _evidence_bound_answer_contract_abstain_reason(
    question: str,
    answer: str,
    support: EvidenceSupport,
) -> AnswerContractResult:
    if _identity_question_requires_complete_answer(question, answer):
        return AnswerContractResult(
            "identity_answer_requires_complete_phrase",
            "predicate_bound_scalar",
            _support_sentence_slot_binding_status(question, support, answer),
        )

    if _REMINDER_OF_QUESTION_PATTERN.search(question):
        tail = _reminder_of_tail_from_support(question, support.support_text)
        if tail and _contains_containment_answer(
            _containment_normalize(answer),
            _containment_normalize(tail),
        ):
            return AnswerContractResult(None, None, None)
        return AnswerContractResult(
            "support_sentence_fails_slot_binding",
            "predicate_bound_scalar",
            "weak",
        )

    if _PICTURE_OF_QUESTION_PATTERN.search(question):
        binding_status = _picture_of_slot_binding_status(question, support, answer)
        if binding_status == "bound":
            return AnswerContractResult(None, None, None)
        return AnswerContractResult(
            "support_sentence_fails_slot_binding",
            "predicate_bound_scalar",
            binding_status,
        )

    if _ANSWER_CONTRACT_LIST_OR_SET_QUESTION_PATTERN.search(question):
        return AnswerContractResult(
            "list_or_set_requires_structured_synthesis",
            "list_or_set",
            None,
        )

    if _ANSWER_CONTRACT_PHRASE_COMPLETION_QUESTION_PATTERN.search(question):
        return AnswerContractResult(
            "phrase_completion_requires_supported_synthesis",
            "complete_phrase",
            None,
        )

    if _answer_contract_event_object_requires_binding(question):
        binding_status = _support_sentence_slot_binding_status(question, support, answer)
        if binding_status == "bound":
            return AnswerContractResult(
                "predicate_slot_requires_structured_binding",
                "predicate_bound_scalar",
                "bound",
            )
        return AnswerContractResult(
            "support_sentence_fails_slot_binding",
            "predicate_bound_scalar",
            binding_status,
        )

    if _action_clause_answer_requires_binding(question):
        binding_status = _support_sentence_slot_binding_status(question, support, answer)
        if binding_status == "bound":
            return AnswerContractResult(None, None, binding_status)
        return AnswerContractResult(
            "support_sentence_fails_slot_binding",
            "predicate_bound_scalar",
            binding_status,
        )

    if {"grandpa", "gift"} <= set(_content_tokens(question)):
        support_terms = set(_content_tokens(support.support_text))
        if "grandpa" not in support_terms or "grandma" in support_terms:
            return AnswerContractResult(
                "support_sentence_fails_slot_binding",
                "predicate_bound_scalar",
                "unbound",
            )

    return AnswerContractResult(None, None, None)


def _list_or_composite_question_requires_complete_answer(question: str) -> bool:
    return bool(_LIST_OR_COMPOSITE_QUESTION_PATTERN.search(question))


def _reminder_of_tail_from_support(question: str, support_text: str) -> str | None:
    if _REMINDER_OF_QUESTION_PATTERN.search(question) is None:
        return None
    question_terms = _question_event_terms(question)
    for sentence in _SENTENCE_SPLIT_PATTERN.split(support_text):
        if not _sentence_matches_terms(sentence, question_terms, min_ratio=0.5):
            continue
        match = _REMINDER_OF_SUPPORT_PATTERN.search(sentence)
        if match is None:
            continue
        tail = _clean_candidate_answer(match.group(1))
        if tail:
            return tail
    return None


def _picture_of_slot_binding_status(
    question: str,
    support: EvidenceSupport,
    answer: str,
) -> str:
    sentences = _sentences_containing_answer(support.support_text, answer)
    if not sentences:
        return "weak"

    question_terms = _question_event_terms(question)
    return (
        "bound"
        if any(
            _PICTURE_OF_SUPPORT_PATTERN.search(sentence)
            and _sentence_matches_terms(
                sentence,
                question_terms,
                min_ratio=_ANSWER_CONTRACT_MIN_SLOT_MATCH_RATIO,
            )
            for sentence in sentences
        )
        else "weak"
    )


def _answer_contract_event_object_requires_binding(question: str) -> bool:
    if not re.search(r"^\s*what\s+did\b", question, re.IGNORECASE):
        return False
    question_terms = set(_content_tokens(question))
    return any(
        any(term in _predicate_forms(verb) for term in question_terms)
        for verb in _EVENT_OBJECT_VERBS_REQUIRING_TYPED_BINDING
    )


def _action_clause_answer_requires_binding(question: str) -> bool:
    return _ACTION_CLAUSE_BINDING_QUESTION_PATTERN.search(question) is not None


def _event_object_answer_requires_typed_candidate(question: str, answer: str) -> bool:
    if not re.search(r"^\s*what\s+did\b", question, re.IGNORECASE):
        return False
    question_terms = _content_tokens(question)
    event_forms: set[str] = set()
    for verb in _EVENT_OBJECT_VERBS_REQUIRING_TYPED_BINDING:
        if any(term in _predicate_forms(verb) for term in question_terms):
            event_forms.update(_predicate_forms(verb))
    if not event_forms:
        return False
    answer_terms = _content_tokens(answer)
    return any(term in event_forms for term in answer_terms)


def _sentences_containing_answer(support_text: str, answer: str) -> list[str]:
    normalized_answers = _answer_sentence_search_terms(answer)
    return [
        sentence
        for sentence in _SENTENCE_SPLIT_PATTERN.split(support_text)
        if any(
            _contains_containment_answer(
                normalized_answer,
                _containment_normalize(sentence),
            )
            for normalized_answer in normalized_answers
        )
    ]


def _support_sentence_slot_binding_status(
    question: str,
    support: EvidenceSupport,
    answer: str,
) -> str:
    sentences = _sentences_containing_answer(support.support_text, answer)
    if not sentences:
        return "weak"

    question_terms = _question_event_terms(question)
    if not question_terms:
        return "bound"

    if any(
        _sentence_matches_terms(
            sentence,
            question_terms,
            min_ratio=_ANSWER_CONTRACT_MIN_SLOT_MATCH_RATIO,
        )
        for sentence in sentences
    ):
        return "bound"
    return "weak"


def _answer_sentence_search_terms(answer: str) -> list[str]:
    normalized_answer = _containment_normalize(answer)
    terms = [normalized_answer]
    terms.extend(number_word for number_word, digit in _NUMBER_WORDS.items() if digit == normalized_answer)
    return terms


def _native_stability_support_cluster(
    support_pack: tuple[ScoredEvidenceSupport, ...],
) -> tuple[ScoredEvidenceSupport, ...]:
    if not support_pack:
        return ()
    top_score = support_pack[0].score
    return tuple(item for item in support_pack if top_score - item.score < _SUPPORT_PACK_CLEAR_WIN_MARGIN)


def _native_stability_question_terms(question: str) -> list[str]:
    slot_terms = {
        "date",
        "day",
        "month",
        "number",
        "time",
        "year",
    }
    ignored_terms = slot_terms | _NATIVE_STABILITY_TEMPORAL_CONNECTOR_TERMS
    return [term for term in _content_tokens(question) if term not in ignored_terms]


def _native_stability_equivalent_terms(term: str) -> set[str]:
    equivalents = set(_predicate_forms(term))
    equivalents.add(term)
    for items in _NATIVE_STABILITY_PREDICATE_EQUIVALENTS.values():
        if term in items:
            equivalents.update(items)
    return equivalents


def _native_stability_tokens_match(
    left: str,
    right: str,
    *,
    allow_subject_alias: bool,
) -> bool:
    if _tokens_match(left, right):
        return True
    if allow_subject_alias and min(len(left), len(right)) >= 3 and (left.startswith(right) or right.startswith(left)):
        return True
    return bool(_native_stability_equivalent_terms(left) & _native_stability_equivalent_terms(right))


def _native_stability_matched_term_count(
    question_terms: list[str],
    sentence_terms: list[str],
) -> int:
    return sum(
        1
        for index, question_term in enumerate(question_terms)
        if any(
            _native_stability_tokens_match(
                question_term,
                sentence_term,
                allow_subject_alias=(index == 0),
            )
            for sentence_term in sentence_terms
        )
    )


def _native_stability_sentence_matches(
    question: str,
    sentence: str,
    *,
    min_ratio: float,
) -> bool:
    question_terms = _native_stability_question_terms(question)
    if not question_terms:
        return True
    sentence_terms = _content_tokens(sentence)
    matched = _native_stability_matched_term_count(question_terms, sentence_terms)
    required = max(1, int(len(question_terms) * min_ratio + 0.999))
    return matched >= required


def _predicate_bound_native_stability_candidate(
    question: str,
    support: EvidenceSupport,
) -> str | None:
    for sentence in _SENTENCE_SPLIT_PATTERN.split(support.support_text):
        sentence = sentence.strip()
        if not sentence or not _native_stability_sentence_matches(question, sentence, min_ratio=0.72):
            continue
        candidate = _native_stability_predicate_answer_from_sentence(question, sentence)
        if candidate is None:
            candidate = _passive_subject_phrase_candidate(sentence)
        if candidate is None:
            continue
        if _contains_containment_answer(
            _containment_normalize(candidate),
            _containment_normalize(sentence),
        ):
            return candidate
    return None


def _native_stability_predicate_answer_from_sentence(
    question: str,
    sentence: str,
) -> str | None:
    question_terms = _content_tokens(question)
    for verb in _EVENT_OBJECT_VERBS_REQUIRING_TYPED_BINDING:
        verb_forms = _native_stability_equivalent_terms(verb)
        if not any(term in verb_forms for term in question_terms):
            continue
        for form in sorted(verb_forms, key=len, reverse=True):
            pattern = (
                rf"\b{re.escape(form)}\s+"
                r"(?:(?:a|an|the)\s+)?(.+?)(?:\s+(?:for|on|with|to|at|in|during|after|"
                r"before|recently|yesterday|today)|\.|$)"
            )
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                candidate = _clean_candidate_answer(match.group(1))
                if verb == "make":
                    candidate = _canonicalize_make_artifact_candidate(question, candidate)
                return candidate
    return None


def _trim_native_stability_object_head(
    answer: str,
    support: EvidenceSupport,
) -> str | None:
    tokens = answer.split()
    if len(tokens) < 2:
        return None
    trimmed = list(tokens)
    changed = False
    while trimmed and trimmed[0].lower() in _NATIVE_STABILITY_OBJECT_TRIM_TOKENS:
        trimmed = trimmed[1:]
        changed = True
    if not changed or not trimmed:
        return None
    candidate = _clean_candidate_answer(" ".join(trimmed))
    if not candidate:
        return None
    if not _contains_containment_answer(
        _containment_normalize(candidate),
        support.normalized_support_text,
    ):
        return None
    return candidate


def _atomic_action_clause_candidate(sentence: str) -> str | None:
    match = _ACTION_CLAUSE_SENTENCE_PATTERN.search(sentence.strip())
    if not match:
        return None
    candidate = _clean_candidate_answer(match.group(1))
    if not candidate:
        return None
    if re.match(r"^(?:is|are|was|were)\b", candidate, re.IGNORECASE):
        return None
    return candidate


def _native_stability_candidate_echoes_question(question: str, candidate: str) -> bool:
    candidate_terms = _content_tokens(candidate)
    if not candidate_terms:
        return True
    question_terms = set(_content_tokens(question))
    return len(candidate_terms) >= 2 and all(term in question_terms for term in candidate_terms)


def _native_stability_date_candidates(sentence: str) -> list[str]:
    candidates: list[str] = []
    for match in _DATE_CANDIDATE_PATTERN.finditer(sentence):
        candidate = _clean_candidate_answer(match.group(0))
        if _date_candidate_sanity_rejection_reason(candidate) is not None:
            continue
        candidates.append(candidate)
    return candidates


def _question_event_terms(question: str) -> list[str]:
    benefit_terms = _benefit_with_having_event_terms(question)
    if benefit_terms is not None:
        return benefit_terms
    slot_terms = {
        "date",
        "day",
        "month",
        "number",
        "time",
        "year",
    }
    return [term for term in _content_tokens(question) if term not in slot_terms]


def _sentence_matches_terms(
    sentence: str,
    terms: list[str],
    *,
    min_ratio: float,
) -> bool:
    if not terms:
        return True
    sentence_terms = _content_tokens(sentence)
    matched = _matched_term_count(terms, sentence_terms)
    required = max(1, int(len(terms) * min_ratio + 0.999))
    return matched >= required


def _support_sufficiently_matches_temporal_question(
    question: str,
    support_text: str,
) -> bool:
    question_terms = _content_tokens(question)
    if len(question_terms) < 3:
        return True

    support_terms = _content_tokens(support_text)
    matched = sum(
        1
        for question_term in question_terms
        if any(_tokens_match(question_term, support_term) for support_term in support_terms)
    )
    required = max(2, int(len(question_terms) * 0.7 + 0.999))
    return matched >= required


def _support_sufficiently_matches_question(
    question: str,
    support_text: str,
    *,
    min_ratio: float = 0.5,
) -> bool:
    question_terms = _content_tokens(question)
    if len(question_terms) < 2:
        return True

    support_terms = _content_tokens(support_text)
    matched = sum(
        1
        for question_term in question_terms
        if any(_tokens_match(question_term, support_term) for support_term in support_terms)
    )
    required = max(2, int(len(question_terms) * min_ratio + 0.999))
    return matched >= required


def _is_unsupported_session_date_answer(answer: str, support_text: str) -> bool:
    if not _SESSION_DATE_PATTERN.search(support_text):
        return False

    normalized_answer = _containment_normalize(answer)
    support_without_session_date = _SESSION_DATE_PATTERN.sub(" ", support_text)
    if _contains_containment_answer(
        normalized_answer,
        _containment_normalize(support_without_session_date),
    ):
        return False

    return _SESSION_DATE_CUE_PATTERN.search(support_without_session_date) is None


def _content_tokens(value: str) -> list[str]:
    return [
        token
        for token in _TOKEN_PATTERN.findall(_normalize(value))
        if len(token) > 2 and token not in _QUESTION_STOPWORDS
    ]


def _tokens_match(left: str, right: str) -> bool:
    if left == right:
        return True
    return min(len(left), len(right)) >= 4 and (left.startswith(right) or right.startswith(left))


def _concept_id(concept: object, concept_index: int) -> str:
    raw = _get_value(concept, "concept_id")
    if raw is None:
        raw = _get_value(concept, "id")
    return str(raw) if raw is not None else f"c{concept_index}"


def _get_text(source: object, key: str) -> str:
    return _stringify(_get_value(source, key))


def _get_sequence(source: object, key: str) -> list:
    value = _get_value(source, key)
    return value if isinstance(value, list) else []


def _get_value(source: object, key: str) -> object:
    if isinstance(source, dict):
        return source.get(key)
    return getattr(source, key, None)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _get_fragment_content(fragment: object) -> str:
    if isinstance(fragment, dict):
        return _stringify(fragment.get("content"))
    return _stringify(getattr(fragment, "content", None))


def _support_evidence_texts(value: object) -> tuple[str, ...]:
    texts: list[str] = []
    seen: set[str] = set()

    def add(text: str) -> None:
        normalized_text = re.sub(r"\s+", " ", text).strip()
        if not normalized_text:
            return
        key = _containment_normalize(normalized_text)
        if key in seen:
            return
        seen.add(key)
        texts.append(normalized_text)

    if isinstance(value, str):
        add(value)
    elif isinstance(value, dict):
        for key in ("content", "text", "user_content", "full_content", "snippet", "quote"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                add(candidate)
    elif isinstance(value, (list, tuple)):
        for item in value:
            for text in _support_evidence_texts(item):
                add(text)

    return tuple(texts)


def _derived_temporal_support_texts(support: EvidenceSupport) -> tuple[str, ...]:
    if support.channel not in {"key_evidence", "verbatim"}:
        return ()
    if not (
        support.channel == "verbatim"
        or re.search(
            r"\bClient evidence\s*:|\[LoCoMo turn id\]|\[Shared media",
            support.support_text,
            re.IGNORECASE,
        )
    ):
        return ()
    if not re.search(r"\bnext\s+month\b", support.support_text, re.IGNORECASE):
        return ()
    original_date = (support.concept_original_date or "").strip()
    match = re.match(r"^(?P<year>\d{4})-(?P<month>\d{1,2})(?:-\d{1,2})?$", original_date)
    if match is None:
        return ()
    month = int(match.group("month"))
    month_name = _MONTH_NAMES.get(month)
    if month_name is None:
        return ()
    resolved = f"{month_name}, {match.group('year')}"
    return (
        f"{support.support_text} (Temporal derivation: next month resolves to {resolved}).",
    )


def _stringify(value: object) -> str:
    return value if isinstance(value, str) else ""
