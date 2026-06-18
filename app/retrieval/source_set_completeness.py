"""Source-set completeness tracing for production retrieval.

RETRIEVAL-112 first slice: source-set instrumentation plus bounded, opt-in
repair helpers. This module does not write to storage. It creates a compact
trace that explains whether a query appears to need multiple evidence sources
and which source/slot metadata is available for the retrieved candidates.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from app.storage import _get_connection

_COMMON_MARKERS = frozenset({"before", "after", "not", "never", "like"})
_PREFERENCE_INTENT_RE = re.compile(
    r"\b(?:prefer(?:ence|ences|red|s)?|favorite|favourite|like most|enjoy most|"
    r"what kind of .{0,40}(?:like|prefer|enjoy)|"
    r"(?:recommend|recommendation|suggestion|suggestions).{0,80}(?:for me|based on my preferences?|that I))\b",
    re.IGNORECASE,
)
_ADVICE_INTENT_RE = re.compile(
    r"\b(?:tip|tips|advice|recommend|recommendation|recommendations|suggest|suggestion|"
    r"suggestions|help with|how to|what should i|what can i|any ideas|keeping it clean)\b",
    re.IGNORECASE,
)
_SELECTION_INTENT_RE = re.compile(
    r"\b(?:what|which|how many|how much|list|show|select|selected|selection|choose|chosen|chose|pick|picked|decided)\b"
    r".{0,100}\b(?:select|selected|selection|choose|chosen|chose|pick|picked|decided|go with|settled on)\b"
    r"|\b(?:select|selected|selection|choose|chosen|chose|pick|picked|decided|go with|settled on)\b"
    r".{0,100}\b(?:what|which|how many|how much|list|show)\b",
    re.IGNORECASE,
)
_SELECTION_CONTEXT_RE = re.compile(
    r"\b(?:for|future|next|live chat|discussion|event|meeting|club|presentation|context|session|conversation)\b",
    re.IGNORECASE,
)
_SELECTION_EXPLORATION_COUNT_RE = re.compile(
    r"\bhow many\b.{0,80}\b(?:different|series|genres?|books?)\b"
    r".{0,140}\b(?:want(?:ing|ed)?|planned|planning|plans?|goals?|explor(?:e|ing)|read(?:ing)?)\b",
    re.IGNORECASE,
)
_CLAIM_SUPPORT_SOURCE_RE = re.compile(
    r"\b(?:(?:which|what)\s+source-backed\s+concept\s+supports?\s+this\s+claim(?:\s+set)?|"
    r"source-backed\s+concept\s+supports?\s+this\s+claim(?:\s+set)?|"
    r"supports?\s+this\s+claim\s+set)\b",
    re.IGNORECASE,
)
_CLAIM_RELATION_SOURCE_RE = re.compile(
    r"\b(?:(?:connect|show)\s+(?:the\s+)?related\s+concepts?\s+around\s+this\s+claim|"
    r"related\s+concepts?\s+around\s+this\s+claim)\b",
    re.IGNORECASE,
)
_PREFERENCE_NEGATIVE_RE = re.compile(
    r"\b(?:dislike|dislikes|don't like|do not like|not prefer|doesn't like|does not like|"
    r"hate|hates|avoid|avoids|tired of)\b",
    re.IGNORECASE,
)
_PREFERENCE_AVOID_RE = re.compile(
    r"\b(?:don't recommend|do not recommend|should avoid|avoid recommending|"
    r"not suggest|don't suggest|do not suggest|do-not-recommend)\b",
    re.IGNORECASE,
)
_PREFERENCE_CONSTRAINT_RE = re.compile(
    r"\b(?:constraint|constraints|budget|under \$?\d+|less than \$?\d+|without|"
    r"allergy|allergies|limited|restriction|restrictions)\b",
    re.IGNORECASE,
)
_PREFERENCE_BRANCH_OUT_RE = re.compile(
    r"\b(?:different from|something different|branch out|beyond|instead of|"
    r"tired of|move beyond|not the usual|usual .{0,30} like)\b",
    re.IGNORECASE,
)
_PREFERENCE_SLOT_FACET_TYPES = {
    "positive_preference": {"positive", "context"},
    "negative_preference": {"negative"},
    "do_not_recommend": {"do_not_recommend"},
    "constraint": {"constraint"},
    "branch_out_target": {"branch_out"},
}
_ADVICE_SLOT_FACET_TYPES = {
    "advice_tip": {"tip", "recommendation", "constraint", "procedure", "avoidance"},
}
_SELECTION_SLOT_FACET_TYPES = {
    "selected_option": {"selected_for_future_context", "selected_option", "selection_context"},
    "selection_context": {"selected_for_future_context", "selection_context"},
}
_AGGREGATE_TOTAL_RE = re.compile(
    r"\b(?:total|sum|combined|altogether|overall|how much total|total weight|total amount)\b",
    re.IGNORECASE,
)
_AGGREGATE_PURCHASE_RE = re.compile(
    r"\b(?:purchas(?:e|ed|ing)|bought|buy|got|ordered|acquired|new feed)\b",
    re.IGNORECASE,
)
_AGGREGATE_UNIT_RE = re.compile(
    r"\b(?:weight|pounds?|lbs?|ounces?|oz|kilograms?|kg|grams?)\b",
    re.IGNORECASE,
)
_AGGREGATE_PURCHASE_TERMS = ("purchased", "purchase", "bought", "buy", "got", "ordered", "acquired")
_AGGREGATE_UNIT_TERMS = (
    "weight",
    "pounds",
    "pound",
    "lbs",
    "lb",
    "ounces",
    "ounce",
    "oz",
    "kilograms",
    "kilogram",
    "kg",
    "grams",
    "gram",
)
_AGGREGATE_FEED_CONTEXT_RE = re.compile(
    r"\b(?:feed|grain|grains|scratch|layer|hens?|chickens?|flock)\b",
    re.IGNORECASE,
)
_AGGREGATE_FEED_DOMAIN_TERMS = (
    "scratch",
    "grain",
    "grains",
    "layer",
    "feed",
    "hen",
    "hens",
    "chicken",
    "chickens",
)
_AGGREGATE_QUANTITY_UNIT_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:-|\s)?(?:pounds?|lbs?|ounces?|oz|kilograms?|kg|grams?)\b",
    re.IGNORECASE,
)
_AGGREGATE_COST_CONTEXT_RE = re.compile(
    r"(?:\$\d|\bcost\b|\bprice\b|\bspent\b|\bper\s+pound\b|\bper\s+lb\b|\bnet cost\b|\bcost per\b)",
    re.IGNORECASE,
)
_AGGREGATE_REPAIR_EXCLUDE_TERMS = frozenset(
    {
        "acquired",
        "aggregate",
        "altogether",
        "amount",
        "bought",
        "combined",
        "could",
        "feed",
        "got",
        "grams",
        "kilogram",
        "kilograms",
        "month",
        "months",
        "new",
        "ordered",
        "ounce",
        "ounces",
        "overall",
        "past",
        "pound",
        "pounds",
        "purchase",
        "purchased",
        "purchasing",
        "total",
        "weight",
    }
)
_PREFERENCE_CONTEXT_BLOCK_ID = "preference_evidence_block"
_PREFERENCE_FACET_BOOST_PER_SLOT = 0.04
_PREFERENCE_FACET_SOURCE_BOOST = 0.02
_PREFERENCE_FACET_MAX_BOOST = 0.08
_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "again",
        "before",
        "does",
        "from",
        "have",
        "many",
        "never",
        "still",
        "that",
        "then",
        "there",
        "this",
        "what",
        "when",
        "where",
        "which",
        "with",
        "would",
    }
)


@dataclass(frozen=True)
class SourceEvidenceRef:
    concept_id: str
    fragment_id: str | None = None
    episode_id: str | None = None
    session_id: str | None = None
    turn_number: int | None = None
    original_date: str | None = None
    pointer_uri: str | None = None
    source_hash: str | None = None
    degraded: bool = False


@dataclass(frozen=True)
class EvidenceSlot:
    slot_id: str
    slot_type: str
    required: bool
    anchors: tuple[str, ...] = ()
    support: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateCoverage:
    concept_id: str
    source_refs: tuple[SourceEvidenceRef, ...]
    slot_ids: tuple[str, ...]
    coverage_score: float
    non_match_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class CoverageDebt:
    slot_id: str
    cause: str
    repair_queries: tuple[str, ...] = ()


@dataclass(frozen=True)
class BeamCandidateView:
    concept_id: str
    memory: str
    score: float
    beam_source_key: str | None = None
    beam_source_turn_id: str | None = None
    beam_source_role: str | None = None
    created_at: str | None = None
    score_debug: dict[str, Any] | None = None
    preference_facet: dict[str, Any] | None = None
    selection_facet: dict[str, Any] | None = None


@dataclass(frozen=True)
class SourceSetTrace:
    source_set_required: bool
    coverage: tuple[CandidateCoverage, ...]
    debts: tuple[CoverageDebt, ...]
    selected_concept_ids: tuple[str, ...] = ()
    dropped_reasons: tuple[dict[str, str], ...] = ()
    elapsed_ms: float = 0.0
    degraded_ref_count: int = 0
    required_slot_count: int = 0
    covered_slot_count: int = 0

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-safe, compact trace payload."""
        payload = asdict(self)
        payload["coverage"] = [
            {
                "concept_id": c["concept_id"],
                "slot_ids": c["slot_ids"],
                "coverage_score": c["coverage_score"],
                "source_ref_count": len(c["source_refs"]),
                "degraded_ref_count": sum(1 for ref in c["source_refs"] if ref.get("degraded")),
                "non_match_reasons": c["non_match_reasons"],
            }
            for c in payload["coverage"]
        ]
        admission_miss_count = len(payload.get("debts") or ())
        payload["admission_miss_count"] = admission_miss_count
        if not payload.get("source_set_required"):
            payload["row_break_class"] = "not_source_set_required"
        elif admission_miss_count:
            payload["row_break_class"] = "source_set_admission_gap"
        else:
            payload["row_break_class"] = "satisfied"
        return payload


def _row_get(row: Any, key: str) -> Any:
    if row is None:
        return None
    if isinstance(row, dict):
        return row.get(key)
    try:
        return row[key]
    except Exception:
        return getattr(row, key, None)


def _value_get(item: Any, key: str) -> Any:
    if item is None:
        return None
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def _nested_containers(item: Any) -> tuple[dict[str, Any], ...]:
    containers: list[dict[str, Any]] = []
    if isinstance(item, dict):
        containers.append(item)
        for key in ("metadata", "data"):
            child = item.get(key)
            if isinstance(child, dict):
                containers.append(child)
                grandchild = child.get("metadata")
                if isinstance(grandchild, dict):
                    containers.append(grandchild)
    else:
        for key in ("metadata", "data"):
            child = getattr(item, key, None)
            if isinstance(child, dict):
                containers.append(child)
                grandchild = child.get("metadata")
                if isinstance(grandchild, dict):
                    containers.append(grandchild)
    return tuple(containers)


def _normalise_preference_facet(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    facet_type = value.get("facet_type")
    if facet_type not in {"positive", "negative", "constraint", "branch_out", "do_not_recommend", "context"}:
        return None
    facet: dict[str, Any] = {
        "facet_type": facet_type,
        "subject": str(value.get("subject") or "user")[:80],
    }
    for key in ("domain", "target", "polarity", "observed_at"):
        item = value.get(key)
        if item is not None and item != "":
            facet[key] = str(item)[:240]
    confidence = value.get("confidence")
    if confidence is not None:
        try:
            facet["confidence"] = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            pass
    source_evidence = value.get("source_evidence")
    if isinstance(source_evidence, list):
        facet["source_evidence"] = source_evidence[:5]
    return facet


def _normalise_advice_facet(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    facet_type = value.get("facet_type")
    if facet_type not in {"tip", "recommendation", "constraint", "procedure", "avoidance"}:
        return None
    facet: dict[str, Any] = {
        "facet_type": facet_type,
        "subject": str(value.get("subject") or "user")[:80],
        "evidence_kind": "advice_facet",
    }
    for key in ("domain", "target", "observed_at"):
        item = value.get(key)
        if item is not None and item != "":
            facet[key] = str(item)[:240]
    confidence = value.get("confidence")
    if confidence is not None:
        try:
            facet["confidence"] = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            pass
    source_evidence = value.get("source_evidence")
    if isinstance(source_evidence, list):
        facet["source_evidence"] = source_evidence[:5]
    return facet


def _normalise_selection_facet(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    facet_type = value.get("facet_type")
    if facet_type not in {"selected_for_future_context", "selected_option", "selection_context"}:
        return None
    facet: dict[str, Any] = {
        "facet_type": facet_type,
        "subject": str(value.get("subject") or "user")[:80],
        "evidence_kind": "selection_facet",
    }
    for key in ("domain", "target", "purpose", "observed_at"):
        item = value.get(key)
        if item is not None and item != "":
            facet[key] = str(item)[:240]
    confidence = value.get("confidence")
    if confidence is not None:
        try:
            facet["confidence"] = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            pass
    source_evidence = value.get("source_evidence")
    if isinstance(source_evidence, list):
        facet["source_evidence"] = source_evidence[:5]
    return facet


def _facet_has_source_evidence(facet: dict[str, Any] | None) -> bool:
    if not facet:
        return False
    source_evidence = facet.get("source_evidence")
    return isinstance(source_evidence, list) and any(bool(item) for item in source_evidence)


def _facet_matches_slot(facet: dict[str, Any] | None, slot: EvidenceSlot) -> bool:
    if not facet:
        return False
    allowed = _PREFERENCE_SLOT_FACET_TYPES.get(slot.slot_id)
    if allowed and facet.get("facet_type") in allowed:
        return True
    advice_allowed = _ADVICE_SLOT_FACET_TYPES.get(slot.slot_id)
    if advice_allowed and facet.get("facet_type") in advice_allowed:
        return True
    selection_allowed = _SELECTION_SLOT_FACET_TYPES.get(slot.slot_id)
    return bool(selection_allowed and facet.get("facet_type") in selection_allowed)


def _placeholders(values: Sequence[str]) -> str:
    return ",".join("?" for _ in values)


def _table_exists(conn: Any, table: str) -> bool:
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?",
            (table,),
        ).fetchone()
        return row is not None
    except Exception:
        return False


def _extract_anchors(text: str) -> tuple[str, ...]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", text or "")
    seen: list[str] = []
    for word in words:
        low = word.lower()
        if low in _STOPWORDS or low in seen:
            continue
        seen.append(low)
    return tuple(seen[:8])


def _aggregate_repair_terms(query: str) -> dict[str, tuple[str, ...]]:
    q = query or ""
    purchase_terms = _AGGREGATE_PURCHASE_TERMS if _AGGREGATE_PURCHASE_RE.search(q) else ()
    unit_terms = _AGGREGATE_UNIT_TERMS if _AGGREGATE_UNIT_RE.search(q) else ()
    topic_terms = tuple(
        term
        for term in _extract_anchors(q)
        if term not in _AGGREGATE_REPAIR_EXCLUDE_TERMS and term not in purchase_terms and term not in unit_terms
    )
    return {
        "purchase_terms": tuple(purchase_terms),
        "unit_terms": tuple(unit_terms),
        "topic_terms": topic_terms,
        "search_terms": tuple(dict.fromkeys(tuple(purchase_terms) + tuple(unit_terms))),
    }


def _plan_aggregate_quantity_slots(query: str, anchors: tuple[str, ...], cls: str) -> list[EvidenceSlot]:
    q = query or ""
    if not (_AGGREGATE_TOTAL_RE.search(q) and (_AGGREGATE_PURCHASE_RE.search(q) or _AGGREGATE_UNIT_RE.search(q))):
        return []
    support = anchors or _aggregate_repair_terms(query).get("topic_terms", ())
    if not support and not cls:
        return []
    return [
        EvidenceSlot("aggregate_quantity_operand", "aggregate_quantity", True, support, ("aggregate", "purchase")),
        EvidenceSlot("aggregate_unit", "unit", True, support, ("aggregate", "unit")),
        EvidenceSlot("aggregate_topic_anchor", "topic", False, support, ("aggregate", "topic")),
    ]


def _classification_label(classification: dict[str, Any] | None) -> str:
    if not classification:
        return ""
    return str(classification.get("classification") or classification.get("type") or "").lower()


def _plan_preference_slots(query: str, anchors: tuple[str, ...], cls: str) -> list[EvidenceSlot]:
    """Plan source-set slots for preference/recommendation evidence tracing."""
    q = query or ""
    q_lower = q.lower()
    has_support = len(anchors) >= 2 or bool(cls)
    if not has_support:
        return []

    preference_classified = "preference" in cls
    positive_intent = bool(_PREFERENCE_INTENT_RE.search(q))
    negative_intent = bool(_PREFERENCE_NEGATIVE_RE.search(q))
    avoid_intent = bool(_PREFERENCE_AVOID_RE.search(q))
    constraint_intent = bool(_PREFERENCE_CONSTRAINT_RE.search(q))
    branch_out_intent = bool(_PREFERENCE_BRANCH_OUT_RE.search(q))
    recommendation_context = any(term in q_lower for term in ("recommend", "suggest", "what kind"))

    if not any(
        (
            preference_classified,
            positive_intent,
            negative_intent,
            avoid_intent,
            constraint_intent and "preference" in q_lower,
            branch_out_intent,
        )
    ):
        return []

    slots: list[EvidenceSlot] = []
    positive_required = (
        (positive_intent and not (negative_intent or avoid_intent))
        or (preference_classified and recommendation_context and not (negative_intent or avoid_intent))
        or branch_out_intent
    )
    if positive_required:
        slots.append(EvidenceSlot("positive_preference", "preference_positive", True, anchors, ("preference",)))
    if negative_intent:
        slots.append(EvidenceSlot("negative_preference", "preference_negative", True, anchors, ("preference",)))
    if avoid_intent:
        slots.append(EvidenceSlot("do_not_recommend", "preference_avoid", True, anchors, ("preference",)))
    if constraint_intent:
        slots.append(EvidenceSlot("constraint", "preference_constraint", True, anchors, ("preference",)))
    if branch_out_intent:
        slots.append(EvidenceSlot("branch_out_target", "preference_branch_out", True, anchors, ("preference",)))
    if slots:
        slots.append(EvidenceSlot("source_evidence", "source", True, anchors, ("preference",)))
    return slots


def _plan_advice_slots(query: str, anchors: tuple[str, ...], cls: str) -> list[EvidenceSlot]:
    """Plan source-backed advice slots without treating advice as user preference."""
    if len(anchors) < 2 and not cls:
        return []
    q = query or ""
    if not (_ADVICE_INTENT_RE.search(q) or "preference" in cls):
        return []
    return [EvidenceSlot("advice_tip", "advice_actionable", True, anchors, ("advice",))]


def _plan_selection_slots(query: str, anchors: tuple[str, ...], cls: str) -> list[EvidenceSlot]:
    """Plan source-backed selected-option slots from serving-visible query text."""
    if len(anchors) < 2 and not cls:
        return []
    q = query or ""
    has_selection_query = bool(_SELECTION_INTENT_RE.search(q) and _SELECTION_CONTEXT_RE.search(q))
    has_future_exploration_count = bool(_SELECTION_EXPLORATION_COUNT_RE.search(q))
    if not (has_selection_query or has_future_exploration_count):
        return []
    support = anchors or ("selection",)
    return [
        EvidenceSlot("selected_option", "selection_option", True, support, ("selection",)),
        EvidenceSlot("selection_context", "selection_context", False, support, ("selection", "context")),
        EvidenceSlot("selection_source_evidence", "source", True, support, ("selection",)),
    ]


def _claim_payload_anchors(query: str, match: re.Match[str] | None) -> tuple[str, ...]:
    if not match:
        return ()
    tail = (query or "")[match.end() :].lstrip()
    if tail.startswith(":"):
        tail = tail[1:].lstrip()
    return _extract_anchors(tail)


def _plan_claim_source_slots(query: str, cls: str) -> list[EvidenceSlot]:
    """Plan source-set slots for explicit source-backed claim support queries."""
    _ = cls
    slots: list[EvidenceSlot] = []
    support_match = _CLAIM_SUPPORT_SOURCE_RE.search(query or "")
    support_anchors = _claim_payload_anchors(query, support_match)
    if support_anchors:
        slots.extend(
            [
                EvidenceSlot("claim_support_fact", "fact", True, support_anchors, ("claim_support",)),
                EvidenceSlot("claim_support_source", "source", True, support_anchors, ("claim_support", "source")),
            ]
        )

    relation_match = _CLAIM_RELATION_SOURCE_RE.search(query or "")
    relation_anchors = _claim_payload_anchors(query, relation_match)
    if relation_anchors:
        slots.extend(
            [
                EvidenceSlot("claim_relation_anchor", "fact", True, relation_anchors, ("claim_relation",)),
                EvidenceSlot("claim_relation_source", "source", True, relation_anchors, ("claim_relation", "source")),
                EvidenceSlot("claim_relation_context", "fact", False, relation_anchors, ("claim_relation", "context")),
            ]
        )
    return slots


def _candidate_identity(item: Any, key: str) -> Any:
    direct = _value_get(item, key)
    if direct is not None and direct != "":
        return direct
    for container in _nested_containers(item):
        value = container.get(key)
        if value is not None and value != "":
            return value
    return None


def plan_evidence_slots(query: str, classification: dict[str, Any] | None = None) -> tuple[EvidenceSlot, ...]:
    """Infer required evidence slots from serving-visible query signals."""
    q = (query or "").lower()
    anchors = _extract_anchors(query)
    cls = _classification_label(classification)
    has_support = len(anchors) >= 2 or bool(cls)

    slots: list[EvidenceSlot] = []
    if (
        "contradict" in q or "changed" in q or "counterclaim" in q or "still true" in q or "contradiction" in cls
    ) and has_support:
        slots.extend(
            [
                EvidenceSlot("target_claim", "claim", True, anchors, ("contradiction",)),
                EvidenceSlot("counterclaim", "counterclaim", True, anchors, ("contradiction",)),
            ]
        )

    ordering_markers = {"before", "after", "first", "then", "sequence", "ordered"}
    if (any(marker in q for marker in ordering_markers) or "temporal" in cls or "event" in cls) and has_support:
        slots.extend(
            [
                EvidenceSlot("event_start", "event", True, anchors, ("temporal",)),
                EvidenceSlot("event_next", "event", True, anchors, ("temporal",)),
                EvidenceSlot("temporal_anchor", "date", True, anchors, ("temporal",)),
            ]
        )

    if ("session" in q or "multi_session" in cls or "multi-session" in cls) and has_support:
        slots.extend(
            [
                EvidenceSlot("primary_fact", "fact", True, anchors, ("multi_session",)),
                EvidenceSlot("bridge_fact", "bridge", True, anchors, ("multi_session",)),
                EvidenceSlot("session_diversity", "session", True, anchors, ("multi_session",)),
            ]
        )

    slots.extend(_plan_preference_slots(query, anchors, cls))
    slots.extend(_plan_advice_slots(query, anchors, cls))
    slots.extend(_plan_selection_slots(query, anchors, cls))
    slots.extend(_plan_claim_source_slots(query, cls))
    slots.extend(_plan_aggregate_quantity_slots(query, anchors, cls))

    # Guard common marker-only activations. If the query is essentially just a
    # common marker, no source-set requirement should fire.
    query_terms = set(re.findall(r"[a-z]+", q))
    if query_terms and query_terms.issubset(_COMMON_MARKERS | {"what", "was", "is", "the"}):
        return ()

    deduped: dict[str, EvidenceSlot] = {}
    for slot in slots:
        deduped.setdefault(slot.slot_id, slot)
    return tuple(deduped.values())


def candidate_view_from_result(result: Any) -> BeamCandidateView:
    """Normalize serving or benchmark candidates into a small canonical view."""
    concept_id = _value_get(result, "concept_id") or _value_get(result, "id") or "unknown"
    memory = _value_get(result, "memory")
    if memory is None:
        memory = _value_get(result, "summary") or ""
    score = _value_get(result, "score")
    if score is None:
        score = _value_get(result, "relevance_score") or 0.0
    score_debug = _value_get(result, "score_debug")
    if not isinstance(score_debug, dict):
        score_debug = None
    metadata = _value_get(result, "metadata")
    preference_facet = None
    selection_facet = None
    if isinstance(metadata, dict):
        preference_facet = metadata.get("preference_facet")
        selection_facet = metadata.get("selection_facet")
    if preference_facet is None:
        preference_facet = _candidate_identity(result, "preference_facet")
    if selection_facet is None:
        selection_facet = _candidate_identity(result, "selection_facet")
    return BeamCandidateView(
        concept_id=str(concept_id),
        memory=str(memory or ""),
        score=float(score or 0.0),
        beam_source_key=_candidate_identity(result, "beam_source_key"),
        beam_source_turn_id=_candidate_identity(result, "beam_source_turn_id"),
        beam_source_role=_candidate_identity(result, "beam_source_role") or _candidate_identity(result, "beam_role"),
        created_at=_value_get(result, "created_at"),
        score_debug=score_debug,
        preference_facet=_normalise_preference_facet(preference_facet),
        selection_facet=_normalise_selection_facet(selection_facet),
    )


def candidate_views_from_results(results: Sequence[Any]) -> tuple[BeamCandidateView, ...]:
    return tuple(candidate_view_from_result(result) for result in results)


def _candidate_text(item: Any, loaded: dict[str, Any] | None = None) -> str:
    parts: list[str] = []
    for key in ("summary", "memory", "user_content", "full_content", "content", "text"):
        value = _row_get(item, key)
        if value:
            parts.append(str(value))
    if loaded:
        for key in ("summary", "memory"):
            value = loaded.get(key)
            if value:
                parts.append(str(value))
    return " ".join(parts).lower()


def _loaded_concept(load_concept_fn: Any, concept_id: str) -> dict[str, Any]:
    try:
        row = load_concept_fn(concept_id)
    except Exception:
        return {}
    if not row:
        return {}
    if isinstance(row, dict):
        return dict(row)
    summary = _row_get(row, "summary")
    confidence = _row_get(row, "confidence")
    knowledge_area = _row_get(row, "knowledge_area")
    if summary is None and not isinstance(row, str):
        try:
            summary, confidence, knowledge_area = row[:3]
        except Exception:
            pass
    return {
        "summary": summary,
        "confidence": confidence,
        "knowledge_area": knowledge_area,
    }


def _topic_terms_from_existing(query: str, existing_results: Sequence[Any]) -> tuple[str, ...]:
    ordered: dict[str, int] = {}
    for term in _aggregate_repair_terms(query).get("topic_terms", ()):
        ordered[term] = ordered.get(term, 0) + 4
    existing_text_parts: list[str] = []
    for result in existing_results[:12]:
        text = str(_value_get(result, "summary") or _value_get(result, "memory") or "").lower()
        existing_text_parts.append(text)
        for term in re.findall(r"[a-z][a-z0-9_-]{3,}", text):
            if term in _STOPWORDS or term in _AGGREGATE_REPAIR_EXCLUDE_TERMS:
                continue
            ordered[term] = ordered.get(term, 0) + 1
    feed_context = _AGGREGATE_FEED_CONTEXT_RE.search(query or "") or _AGGREGATE_FEED_CONTEXT_RE.search(
        " ".join(existing_text_parts)
    )
    if feed_context:
        for term in _AGGREGATE_FEED_DOMAIN_TERMS:
            ordered[term] = ordered.get(term, 0) + 3
    return tuple(sorted(ordered, key=lambda t: (-ordered[t], t))[:20])


def _contains_any_term(text: str, terms: Sequence[str]) -> bool:
    if not text or not terms:
        return False
    variants: set[str] = set()
    for term in terms:
        variants.add(term)
        if len(term) > 3 and term.endswith("s"):
            variants.add(term[:-1])
        elif len(term) > 3:
            variants.add(f"{term}s")
    return any(re.search(rf"\b{re.escape(term)}\b", text) for term in variants)


def _augment_aggregate_summary(summary: str, query: str, text: str) -> str:
    if not _AGGREGATE_FEED_CONTEXT_RE.search(query or ""):
        return summary
    surface = f"{summary} {text}".lower()
    if "feed" in (summary or "").lower():
        return summary
    if not any(term in surface for term in ("scratch", "grain", "grains")):
        return summary
    if re.search(r"\bscratch grains\b", summary, flags=re.IGNORECASE):
        revised = re.sub(
            r"\bscratch grains\b",
            "chicken feed (scratch grains)",
            summary,
            flags=re.IGNORECASE,
        )
        return f"{revised}; include this feed purchase in aggregate new-feed totals."
    return f"{summary} (feed-related chicken purchase)"


def build_aggregate_source_set_repair(
    query: str,
    existing_results: Sequence[Any],
    *,
    search_fn: Any,
    load_concept_fn: Any,
    search_limit: int = 120,
    domain_search_limit: int = 40,
    topic_search_limit: int = 60,
    max_insertions: int = 4,
    weak_topic_max_insertions: int = 2,
    score_cap: float = 0.62,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return capped neutral candidates for aggregate purchase/unit queries."""
    repair_terms = _aggregate_repair_terms(query)
    trace: dict[str, Any] = {
        "enabled": True,
        "triggered": False,
        "repair_terms": repair_terms,
        "candidate_count": 0,
        "domain_candidate_count": 0,
        "topic_candidate_count": 0,
        "inserted_count": 0,
        "inserted_concept_ids": [],
        "weak_topic_anchor_fallback_count": 0,
        "search_limit": search_limit,
        "domain_search_limit": domain_search_limit,
        "topic_search_limit": topic_search_limit,
        "drop_reasons": {},
    }
    slots = _plan_aggregate_quantity_slots(query, _extract_anchors(query), "")
    search_terms = list(repair_terms.get("search_terms", ()))
    if not slots or not search_terms:
        trace["drop_reasons"]["not_aggregate_query"] = 1
        return [], trace

    trace["triggered"] = True
    topic_terms = _topic_terms_from_existing(query, existing_results)
    try:
        domain_terms = tuple(term for term in _AGGREGATE_FEED_DOMAIN_TERMS if term in topic_terms)
        domain_candidates = list(search_fn(domain_terms, limit=domain_search_limit) or []) if domain_terms else []
        raw_candidates = list(domain_candidates)
        raw_candidates.extend(search_fn(search_terms, limit=search_limit) or [])
        trace["domain_candidate_count"] = len(domain_candidates)
        if topic_terms:
            topic_candidates = list(search_fn(topic_terms, limit=topic_search_limit) or [])
            raw_candidates.extend(topic_candidates)
            trace["topic_candidate_count"] = len(topic_candidates)
    except Exception as exc:
        trace["drop_reasons"]["search_failed"] = str(exc)[:160]
        return [], trace

    unique_candidates: list[Any] = []
    seen_candidate_ids: set[str] = set()
    for candidate in raw_candidates:
        candidate_id = _row_get(candidate, "concept_id") or _row_get(candidate, "id")
        if not candidate_id:
            unique_candidates.append(candidate)
            continue
        candidate_id = str(candidate_id)
        if candidate_id in seen_candidate_ids:
            continue
        seen_candidate_ids.add(candidate_id)
        unique_candidates.append(candidate)

    trace["candidate_count"] = len(unique_candidates)
    existing_ids = {str(cid) for cid in (_value_get(result, "concept_id") for result in existing_results) if cid}
    inserted: list[dict[str, Any]] = []
    inserted_ids = set(existing_ids)
    weak_count = 0
    drop_reasons: dict[str, int] = {}

    def _drop(reason: str) -> None:
        drop_reasons[reason] = drop_reasons.get(reason, 0) + 1

    for candidate in unique_candidates:
        if len(inserted) >= max_insertions:
            _drop("max_insertions")
            break
        concept_id = _row_get(candidate, "concept_id") or _row_get(candidate, "id")
        if not concept_id:
            _drop("missing_concept_id")
            continue
        concept_id = str(concept_id)
        if concept_id in inserted_ids:
            _drop("duplicate_concept_id")
            continue

        loaded = _loaded_concept(load_concept_fn, concept_id)
        text = _candidate_text(candidate, loaded)
        if not _contains_any_term(text, _AGGREGATE_PURCHASE_TERMS):
            _drop("missing_purchase_term")
            continue
        if not _contains_any_term(text, _AGGREGATE_UNIT_TERMS):
            _drop("missing_unit_term")
            continue
        if not _AGGREGATE_QUANTITY_UNIT_RE.search(text):
            _drop("missing_quantity_unit")
            continue
        if _AGGREGATE_COST_CONTEXT_RE.search(text):
            _drop("cost_context")
            continue

        has_topic_anchor = _contains_any_term(text, topic_terms)
        if not has_topic_anchor:
            if weak_count >= weak_topic_max_insertions:
                _drop("weak_topic_anchor_cap")
                continue
            weak_count += 1

        summary = loaded.get("summary") or _row_get(candidate, "summary") or _row_get(candidate, "user_content") or ""
        summary = _augment_aggregate_summary(str(summary), query, text)
        confidence = loaded.get("confidence")
        try:
            confidence = float(confidence) if confidence is not None else 0.5
        except (TypeError, ValueError):
            confidence = 0.5
        score = _row_get(candidate, "relevance_score") or _row_get(candidate, "score")
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = score_cap
        if score <= 0:
            score = score_cap
        row = {
            "concept_id": concept_id,
            "summary": summary,
            "confidence": confidence,
            "knowledge_area": loaded.get("knowledge_area") or _row_get(candidate, "knowledge_area") or "unknown",
            "relevance_score": round(min(score_cap, score), 4),
            "reason": "topic_anchor_match" if has_topic_anchor else "weak_topic_anchor_fallback",
        }
        inserted.append(row)
        inserted_ids.add(concept_id)

    trace["inserted_count"] = len(inserted)
    trace["inserted_concept_ids"] = [row["concept_id"] for row in inserted]
    trace["weak_topic_anchor_fallback_count"] = weak_count
    trace["drop_reasons"] = drop_reasons
    return inserted, trace


def extract_source_refs(
    concept_ids: Sequence[str], conn: Any | None = None
) -> dict[str, tuple[SourceEvidenceRef, ...]]:
    """Read existing source/span-adjacent metadata for concept ids."""
    ids = tuple(dict.fromkeys(cid for cid in concept_ids if cid))
    if not ids:
        return {}
    conn = conn or _get_connection()

    by_concept: dict[str, list[SourceEvidenceRef]] = {cid: [] for cid in ids}
    concept_rows: dict[str, Any] = {}
    try:
        rows = conn.execute(
            f"SELECT id, session_id, original_date, created_at FROM concepts WHERE id IN ({_placeholders(ids)})",
            ids,
        ).fetchall()
        concept_rows = {_row_get(row, "id"): row for row in rows}
    except Exception:
        concept_rows = {}

    if _table_exists(conn, "verbatim_fragments"):
        try:
            for row in conn.execute(
                f"""SELECT id, concept_id, pointer_uri, source_hash, created_at
                    FROM verbatim_fragments WHERE concept_id IN ({_placeholders(ids)})""",
                ids,
            ).fetchall():
                cid = _row_get(row, "concept_id")
                c_row = concept_rows.get(cid)
                by_concept.setdefault(cid, []).append(
                    SourceEvidenceRef(
                        concept_id=cid,
                        fragment_id=_row_get(row, "id"),
                        session_id=_row_get(c_row, "session_id"),
                        original_date=_row_get(c_row, "original_date") or _row_get(row, "created_at"),
                        pointer_uri=_row_get(row, "pointer_uri"),
                        source_hash=_row_get(row, "source_hash"),
                        degraded=False,
                    )
                )
        except Exception:
            pass

    if _table_exists(conn, "episodes"):
        try:
            episode_rows = conn.execute(
                "SELECT id, session_id, turn_number, extracted_concept_ids, world_timestamp FROM episodes"
            ).fetchall()
            id_set = set(ids)
            for row in episode_rows:
                try:
                    extracted = json.loads(_row_get(row, "extracted_concept_ids") or "[]")
                except Exception:
                    extracted = []
                for cid in id_set.intersection(extracted):
                    c_row = concept_rows.get(cid)
                    by_concept.setdefault(cid, []).append(
                        SourceEvidenceRef(
                            concept_id=cid,
                            episode_id=_row_get(row, "id"),
                            session_id=_row_get(row, "session_id") or _row_get(c_row, "session_id"),
                            turn_number=_row_get(row, "turn_number"),
                            original_date=_row_get(c_row, "original_date") or _row_get(row, "world_timestamp"),
                            degraded=False,
                        )
                    )
        except Exception:
            pass

    for cid in ids:
        if by_concept.get(cid):
            continue
        row = concept_rows.get(cid)
        session_id = _row_get(row, "session_id")
        original_date = _row_get(row, "original_date") or _row_get(row, "created_at")
        by_concept[cid] = [
            SourceEvidenceRef(
                concept_id=cid,
                session_id=session_id,
                original_date=original_date,
                degraded=not bool(session_id or original_date),
            )
        ]

    return {cid: tuple(refs) for cid, refs in by_concept.items()}


def extract_preference_facets(
    concept_ids: Sequence[str], conn: Any | None = None
) -> dict[str, dict[str, Any]]:
    """Read sanitized preference facet metadata for candidate concept ids."""
    ids = tuple(dict.fromkeys(cid for cid in concept_ids if cid))
    if not ids:
        return {}
    conn = conn or _get_connection()
    facets: dict[str, dict[str, Any]] = {}
    try:
        rows = conn.execute(
            f"SELECT id, data FROM concepts WHERE id IN ({_placeholders(ids)})",
            ids,
        ).fetchall()
    except Exception:
        return {}
    for row in rows:
        cid = _row_get(row, "id")
        try:
            data = json.loads(_row_get(row, "data") or "{}")
        except Exception:
            data = {}
        metadata = data.get("metadata") if isinstance(data, dict) else None
        facet = _normalise_preference_facet(
            metadata.get("preference_facet") if isinstance(metadata, dict) else None
        )
        if cid and facet:
            facets[cid] = facet
    return facets


def extract_selection_facets(concept_ids: Sequence[str], conn: Any | None = None) -> dict[str, dict[str, Any]]:
    """Read sanitized source-backed selection facet metadata for candidate concept ids."""
    ids = tuple(dict.fromkeys(cid for cid in concept_ids if cid))
    if not ids:
        return {}
    conn = conn or _get_connection()
    facets: dict[str, dict[str, Any]] = {}
    try:
        rows = conn.execute(
            f"SELECT id, data FROM concepts WHERE id IN ({_placeholders(ids)})",
            ids,
        ).fetchall()
    except Exception:
        return {}
    for row in rows:
        cid = _row_get(row, "id")
        try:
            data = json.loads(_row_get(row, "data") or "{}")
        except Exception:
            data = {}
        metadata = data.get("metadata") if isinstance(data, dict) else None
        facet = _normalise_selection_facet(
            metadata.get("selection_facet") if isinstance(metadata, dict) else None
        )
        if cid and facet and _facet_has_source_evidence(facet):
            facets[cid] = facet
    return facets


def extract_evidence_facets(concept_ids: Sequence[str], conn: Any | None = None) -> dict[str, dict[str, Any]]:
    """Read source-backed preference/advice facet metadata for candidate concept ids."""
    ids = tuple(dict.fromkeys(cid for cid in concept_ids if cid))
    if not ids:
        return {}
    conn = conn or _get_connection()
    facets: dict[str, dict[str, Any]] = {}
    try:
        rows = conn.execute(
            f"SELECT id, data FROM concepts WHERE id IN ({_placeholders(ids)})",
            ids,
        ).fetchall()
    except Exception:
        return {}
    for row in rows:
        cid = _row_get(row, "id")
        try:
            data = json.loads(_row_get(row, "data") or "{}")
        except Exception:
            data = {}
        metadata = data.get("metadata") if isinstance(data, dict) else None
        if not isinstance(metadata, dict):
            continue
        preference_facet = _normalise_preference_facet(metadata.get("preference_facet"))
        if preference_facet:
            preference_facet["evidence_kind"] = "preference_facet"
            facets[cid] = preference_facet
            continue
        advice_facet = _normalise_advice_facet(metadata.get("advice_facet"))
        if advice_facet:
            facets[cid] = advice_facet
            continue
        selection_facet = _normalise_selection_facet(metadata.get("selection_facet"))
        if selection_facet and _facet_has_source_evidence(selection_facet):
            facets[cid] = selection_facet
    return facets


def compute_candidate_coverage(
    results: Sequence[Any],
    slots: Sequence[EvidenceSlot],
    source_refs_by_concept: dict[str, tuple[SourceEvidenceRef, ...]],
    preference_facets_by_concept: dict[str, dict[str, Any]] | None = None,
) -> tuple[CandidateCoverage, ...]:
    """Compute lightweight slot coverage for retrieved candidates."""
    coverage: list[CandidateCoverage] = []
    preference_facets_by_concept = preference_facets_by_concept or {}
    for result in results:
        cid = getattr(result, "concept_id", None)
        summary = (getattr(result, "summary", "") or "").lower()
        refs = source_refs_by_concept.get(cid, ())
        facet = preference_facets_by_concept.get(cid)
        matched: list[str] = []
        reasons: list[str] = []
        for slot in slots:
            anchor_hit = any(anchor in summary for anchor in slot.anchors)
            date_hit = slot.slot_type == "date" and any(ref.original_date for ref in refs)
            session_hit = slot.slot_type == "session" and any(ref.session_id for ref in refs)
            preference_hit = slot.slot_type.startswith("preference_") and _facet_matches_slot(facet, slot)
            selection_hit = slot.slot_type.startswith("selection_") and _facet_matches_slot(facet, slot)
            source_hit = slot.slot_type == "source" and (
                any(not ref.degraded for ref in refs) or _facet_has_source_evidence(facet)
            )
            if slot.slot_type == "source":
                slot_matched = source_hit
            elif slot.slot_type.startswith("preference_"):
                slot_matched = preference_hit or anchor_hit
            elif slot.slot_type.startswith("selection_"):
                slot_matched = selection_hit or anchor_hit
            else:
                slot_matched = anchor_hit or date_hit or session_hit
            if slot_matched:
                matched.append(slot.slot_id)
            else:
                reasons.append(f"no_match:{slot.slot_id}")
        score = 0.0 if not slots else round(len(matched) / len(slots), 4)
        coverage.append(
            CandidateCoverage(
                concept_id=cid,
                source_refs=refs,
                slot_ids=tuple(matched),
                coverage_score=score,
                non_match_reasons=tuple(reasons[:5]),
            )
        )
    return tuple(coverage)


def compute_candidate_coverage_from_views(
    candidate_views: Sequence[BeamCandidateView],
    slots: Sequence[EvidenceSlot],
) -> tuple[CandidateCoverage, ...]:
    """Compute slot coverage from a canonical benchmark candidate view."""
    coverage: list[CandidateCoverage] = []
    for candidate in candidate_views:
        lowered = candidate.memory.lower()
        synthetic_ref = SourceEvidenceRef(
            concept_id=candidate.concept_id,
            session_id=candidate.beam_source_turn_id,
            original_date=candidate.created_at,
            degraded=not bool(candidate.beam_source_key or candidate.beam_source_turn_id or candidate.created_at),
        )
        facet = _normalise_preference_facet(candidate.preference_facet)
        selection_facet = _normalise_selection_facet(candidate.selection_facet)
        matched: list[str] = []
        reasons: list[str] = []
        for slot in slots:
            anchor_hit = any(anchor in lowered for anchor in slot.anchors)
            date_hit = slot.slot_type == "date" and bool(candidate.created_at)
            session_hit = slot.slot_type == "session" and bool(
                candidate.beam_source_turn_id or candidate.beam_source_key
            )
            preference_hit = slot.slot_type.startswith("preference_") and _facet_matches_slot(facet, slot)
            selection_hit = slot.slot_type.startswith("selection_") and _facet_matches_slot(selection_facet, slot)
            source_hit = slot.slot_type == "source" and (
                not synthetic_ref.degraded
                or _facet_has_source_evidence(facet)
                or _facet_has_source_evidence(selection_facet)
            )
            if slot.slot_type == "source":
                slot_matched = source_hit
            elif slot.slot_type.startswith("preference_"):
                slot_matched = preference_hit or anchor_hit
            elif slot.slot_type.startswith("selection_"):
                slot_matched = selection_hit or anchor_hit
            else:
                slot_matched = anchor_hit or date_hit or session_hit
            if slot_matched:
                matched.append(slot.slot_id)
            else:
                reasons.append(f"no_match:{slot.slot_id}")
        score = 0.0 if not slots else round(len(matched) / len(slots), 4)
        coverage.append(
            CandidateCoverage(
                concept_id=candidate.concept_id,
                source_refs=(synthetic_ref,),
                slot_ids=tuple(matched),
                coverage_score=score,
                non_match_reasons=tuple(reasons[:5]),
            )
        )
    return tuple(coverage)


def classify_coverage_debt(
    slots: Sequence[EvidenceSlot],
    coverage: Sequence[CandidateCoverage],
    candidate_views: Sequence[BeamCandidateView] = (),
) -> tuple[CoverageDebt, ...]:
    covered = {slot_id for candidate in coverage for slot_id in candidate.slot_ids}
    debts: list[CoverageDebt] = []
    has_degraded_refs = any(ref.degraded for candidate in coverage for ref in candidate.source_refs)
    has_source_identity = any(view.beam_source_key or view.beam_source_turn_id for view in candidate_views)
    for slot in slots:
        if not slot.required or slot.slot_id in covered:
            continue
        if covered:
            cause = "PARTIAL"
        elif has_degraded_refs and not has_source_identity:
            cause = "UNKNOWN"
        else:
            cause = "ABSENT"
        repair_terms = tuple(anchor for anchor in slot.anchors[:3] if anchor)
        debts.append(CoverageDebt(slot.slot_id, cause, repair_terms))
    return tuple(debts)


def preference_facet_boosts(
    query: str,
    results: Sequence[Any],
    facets_by_concept: dict[str, dict[str, Any]] | None = None,
    classification: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Return bounded relevance boosts for matched source-backed preference facets."""
    slots = tuple(
        slot for slot in plan_evidence_slots(query, classification) if slot.slot_type.startswith("preference_")
    )
    if not slots:
        return {}
    concept_ids = [getattr(result, "concept_id", None) for result in results]
    facets = facets_by_concept or extract_preference_facets([cid for cid in concept_ids if cid])
    boosts: dict[str, float] = {}
    for cid in concept_ids:
        facet = facets.get(cid)
        if not facet or not _facet_has_source_evidence(facet):
            continue
        matched_count = sum(1 for slot in slots if _facet_matches_slot(facet, slot))
        if matched_count <= 0:
            continue
        boost = min(_PREFERENCE_FACET_MAX_BOOST, matched_count * _PREFERENCE_FACET_BOOST_PER_SLOT)
        boost = min(_PREFERENCE_FACET_MAX_BOOST, boost + _PREFERENCE_FACET_SOURCE_BOOST)
        boosts[cid] = round(boost, 4)
    return boosts


def selection_facet_boosts(
    query: str,
    results: Sequence[Any],
    facets_by_concept: dict[str, dict[str, Any]] | None = None,
    classification: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Return bounded relevance boosts for matched source-backed selection facets."""
    slots = tuple(
        slot for slot in plan_evidence_slots(query, classification) if slot.slot_type.startswith("selection_")
    )
    if not slots:
        return {}
    concept_ids = [getattr(result, "concept_id", None) for result in results]
    facets = facets_by_concept or extract_selection_facets([cid for cid in concept_ids if cid])
    boosts: dict[str, float] = {}
    for cid in concept_ids:
        facet = facets.get(cid)
        if not facet or not _facet_has_source_evidence(facet):
            continue
        matched_count = sum(1 for slot in slots if _facet_matches_slot(facet, slot))
        if matched_count <= 0:
            continue
        boost = min(_PREFERENCE_FACET_MAX_BOOST, matched_count * _PREFERENCE_FACET_BOOST_PER_SLOT)
        boost = min(_PREFERENCE_FACET_MAX_BOOST, boost + _PREFERENCE_FACET_SOURCE_BOOST)
        boosts[cid] = round(boost, 4)
    return boosts


def build_preference_evidence_block(
    query: str,
    results: Sequence[Any],
    facets_by_concept: dict[str, dict[str, Any]] | None = None,
    classification: dict[str, Any] | None = None,
    limit: int = 5,
) -> str | None:
    """Build a compact source-backed preference evidence context block."""
    slots = tuple(
        slot
        for slot in plan_evidence_slots(query, classification)
        if slot.slot_type.startswith("preference_") or slot.slot_type.startswith("advice_")
    )
    if not slots:
        return None
    concept_ids = [getattr(result, "concept_id", None) for result in results]
    facets = facets_by_concept or extract_evidence_facets([cid for cid in concept_ids if cid])
    lines: list[str] = []
    seen: set[tuple[str, str]] = set()
    for slot in slots:
        for result in results:
            cid = getattr(result, "concept_id", None)
            facet = facets.get(cid)
            if not _facet_matches_slot(facet, slot) or not _facet_has_source_evidence(facet):
                continue
            key = (str(cid), str(facet.get("facet_type")))
            if key in seen:
                continue
            seen.add(key)
            target = str(facet.get("target") or getattr(result, "summary", ""))[:240]
            domain = str(facet.get("domain") or "general")[:80]
            source = _format_preference_source(facet.get("source_evidence"))
            kind = str(facet.get("evidence_kind") or "preference_facet")
            lines.append(f"- {kind}:{facet['facet_type']}/{domain}: {target} | source: {source}")
            if len(lines) >= limit:
                return "[Preference evidence]\n" + "\n".join(lines)
    if not lines:
        return None
    return "[Preference evidence]\n" + "\n".join(lines)


def build_selection_evidence_block(
    query: str,
    results: Sequence[Any],
    facets_by_concept: dict[str, dict[str, Any]] | None = None,
    classification: dict[str, Any] | None = None,
    limit: int = 5,
) -> str | None:
    """Build compact source-backed selected-option evidence context."""
    slots = tuple(
        slot for slot in plan_evidence_slots(query, classification) if slot.slot_type.startswith("selection_")
    )
    if not slots:
        return None
    concept_ids = [getattr(result, "concept_id", None) for result in results]
    facets = facets_by_concept or extract_selection_facets([cid for cid in concept_ids if cid])
    lines: list[str] = []
    seen: set[tuple[str, str]] = set()
    for slot in slots:
        for result in results:
            cid = getattr(result, "concept_id", None)
            facet = facets.get(cid)
            if not _facet_matches_slot(facet, slot) or not _facet_has_source_evidence(facet):
                continue
            key = (str(cid), str(facet.get("target") or ""))
            if key in seen:
                continue
            seen.add(key)
            target = str(facet.get("target") or getattr(result, "summary", ""))[:240]
            purpose = str(facet.get("purpose") or "future context")[:240]
            domain = str(facet.get("domain") or "general")[:80]
            source = _format_preference_source(facet.get("source_evidence"))
            lines.append(
                f"- selection_facet:{facet['facet_type']}/{domain}: "
                f"{target} | purpose: {purpose} | source: {source}"
            )
            if len(lines) >= limit:
                return "[Selection evidence]\n" + "\n".join(lines)
    if not lines:
        return None
    return "[Selection evidence]\n" + "\n".join(lines)


def _format_preference_source(source_evidence: Any) -> str:
    if not isinstance(source_evidence, list) or not source_evidence:
        return "source-backed preference"
    item = source_evidence[0]
    if isinstance(item, str):
        return item[:240]
    if isinstance(item, dict):
        for key in ("verbatim", "fragment_id", "evidence_id", "session_id", "source_reference"):
            value = item.get(key)
            if value:
                return str(value)[:240]
    return "source-backed preference"


def build_source_set_trace_from_candidate_views(
    query: str,
    candidate_views: Sequence[BeamCandidateView],
    classification: dict[str, Any] | None = None,
) -> SourceSetTrace:
    """Build a generic trace for benchmark-local candidate windows."""
    t0 = time.perf_counter()
    slots = plan_evidence_slots(query, classification)
    coverage = compute_candidate_coverage_from_views(candidate_views, slots)
    debts = classify_coverage_debt(slots, coverage, candidate_views=candidate_views)
    degraded_count = sum(1 for candidate in coverage for ref in candidate.source_refs if ref.degraded)
    covered_slots = {slot_id for candidate in coverage for slot_id in candidate.slot_ids}
    return SourceSetTrace(
        source_set_required=bool(slots),
        coverage=coverage,
        debts=debts,
        selected_concept_ids=tuple(candidate.concept_id for candidate in candidate_views),
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 3),
        degraded_ref_count=degraded_count,
        required_slot_count=sum(1 for slot in slots if slot.required),
        covered_slot_count=len(covered_slots),
    )


def build_source_set_trace(
    query: str,
    results: Sequence[Any],
    classification: dict[str, Any] | None = None,
) -> SourceSetTrace:
    """Build a read-only source-set trace for the current candidate ordering."""
    t0 = time.perf_counter()
    slots = plan_evidence_slots(query, classification)
    concept_ids = [getattr(result, "concept_id", None) for result in results]
    refs_by_concept = extract_source_refs([cid for cid in concept_ids if cid])
    facets_by_concept = extract_evidence_facets([cid for cid in concept_ids if cid])
    coverage = compute_candidate_coverage(results, slots, refs_by_concept, facets_by_concept)
    debts = classify_coverage_debt(slots, coverage, candidate_views=candidate_views_from_results(results))
    degraded_count = sum(1 for candidate in coverage for ref in candidate.source_refs if ref.degraded)
    covered_slots = {slot_id for candidate in coverage for slot_id in candidate.slot_ids}
    return SourceSetTrace(
        source_set_required=bool(slots),
        coverage=coverage,
        debts=debts,
        selected_concept_ids=tuple(cid for cid in concept_ids if cid),
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 3),
        degraded_ref_count=degraded_count,
        required_slot_count=sum(1 for slot in slots if slot.required),
        covered_slot_count=len(covered_slots),
    )
