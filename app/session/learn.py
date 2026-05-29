"""SessionLearnMixin — session_learn pipeline + insight processing.

Extracted from session/__init__.py lines 10194-13436 per ARCH-009.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
import re as _re
from datetime import UTC, datetime, timedelta
from typing import Any, Optional

from app.core.constants import (
    FRESHNESS_EARLIER_TODAY_UPPER,
    FRESHNESS_HOURS_AGO_UPPER,
    FRESHNESS_JUST_NOW_MINS,
    FRESHNESS_MINUTES_AGO_UPPER,
    FRESHNESS_ONE_HOUR_UPPER,
    FRESHNESS_YESTERDAY_UPPER,
    GOV_EVENT_CCL_VIOLATIONS_DETECTED,
    GOV_EVENT_CIRCUIT_BREAKER_TRIPPED,
    GOV_EVENT_COMPACTION_REINJECTION,
    GOV_EVENT_CONTRADICTION_REVIEW,
    GOV_EVENT_CONVERSATION_TURN_COMPLETE,
    GOV_EVENT_GRAPH_CONTRADICTION_SIGNAL,
    GOV_EVENT_RESUME_CONTEXT_INJECTION,
    MINUTES_PER_HOUR,
)
from app.core.datetime_utils import _ensure_aware, _utc_now, _utc_now_iso
from app.core.models import (
    ActivatedConcept,
    ActiveDirectionality,
    ActiveUncertainty,
    AreaStrength,
    Concept,
    ConceptEvolution,
    ConceptEvolutionRecord,
    ConversationTurnRequest,
    ConversationTurnResponse,
    CuriosityFrontierItem,
    CurrentStateAssessment,
    EvolvedConcept,
    GoalSummary,
    LearnedConcept,
    PendingQuestionSummary,
    PresentMomentOrientation,
    RecentConceptSummary,
    RecentEvolutionSummary,
    SearchResult,
    SessionEndRequest,
    SessionInfo,
    SessionLearnRequest,
    SessionLearnResponse,
    SessionStartResponse,
)
from app.cognitive.branch_provenance_population import preserve_or_build_branch_provenance
from app.session.self_model import self_model_manager
from app.storage import (
    _get_connection,
    cleanup_expired_snapshots,
    count_associations,
    count_sessions,
    get_related_concepts,
    list_concepts,
    load_associations,
    load_concept,
    load_recent_concepts,
    load_resume_snapshot,
    load_session,
    load_session_velocity,
    recover_interrupted_sessions,
    save_concept,
    # Resume Context v1.1
    save_resume_snapshot,
    save_session,
    update_session,
)
from app.cognitive.taxonomy import (  # DEBT-030/DEBT-108
    classify_knowledge_area,
    infer_knowledge_area,
    normalize_knowledge_area_boundary,
)

from app.session.helpers import _extract_subject_key, _has_named_entities, _validate_concept_type
from app.session.grouped_count_packet_metadata import normalise_grouped_count_packet_metadata

logger = logging.getLogger(__name__)

# Module-level counter for precision guard observability (used via `global` in methods)
_PRECISION_GUARD_BLOCKS: int = 0


logger = logging.getLogger(__name__)

# Module-level counter for precision guard observability (used via `global` in methods)
_PRECISION_GUARD_BLOCKS: int = 0


_PREFERENCE_FACET_ALLOWED_KEYS = frozenset(
    {
        "subject",
        "domain",
        "facet_type",
        "target",
        "polarity",
        "source_evidence",
        "confidence",
        "observed_at",
    }
)
_PREFERENCE_SOURCE_EVIDENCE_KEYS = frozenset(
    {"evidence_id", "fragment_id", "session_id", "verbatim", "source_reference"}
)
_SELECTION_SOURCE_EVIDENCE_KEYS = _PREFERENCE_SOURCE_EVIDENCE_KEYS | {"turn_id"}
_PREFERENCE_FACET_TYPES = frozenset(
    {"positive", "negative", "constraint", "branch_out", "do_not_recommend", "context"}
)
_SELECTION_FACET_ALLOWED_KEYS = frozenset(
    {
        "subject",
        "domain",
        "facet_type",
        "target",
        "purpose",
        "source_evidence",
        "confidence",
        "observed_at",
    }
)
_SELECTION_FACET_TYPES = frozenset(
    {"selected_for_future_context", "selected_option", "selection_context"}
)
_FORBIDDEN_SELECTION_FACET_KEYS = frozenset(
    {
        "expected_answer",
        "expected_source_ref",
        "rubric",
        "source_chat_ids",
        "gold_id",
        "answer_string",
    }
)
_GROUNDING_METADATA_KEYS = frozenset(
    {
        "evidence_role",
        "slot_subject",
        "slot_attribute",
        "slot_group_id",
        "grounding_priority",
    }
)
_GROUNDING_EVIDENCE_ROLES = frozenset(
    {
        "instruction_obligation",
        "summary_milestone",
        "exact_detail",
        "correction_update",
        "contradiction_side",
    }
)
_FORBIDDEN_GROUNDING_METADATA_KEYS = frozenset(
    {
        "answer_key",
        "answer_string",
        "expected_answer",
        "expected_source_ref",
        "expected_source_refs",
        "gold_answer",
        "gold_id",
        "judge_score",
        "pass_fail_label",
        "question_id",
        "row_id",
        "rubric",
        "score_label",
        "source_chat_ids",
    }
)


def _contains_forbidden_grounding_marker(value: Any) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).lower() in _FORBIDDEN_GROUNDING_METADATA_KEYS:
                return True
            if _contains_forbidden_grounding_marker(nested):
                return True
        return False
    if isinstance(value, (list, tuple, set)):
        return any(_contains_forbidden_grounding_marker(item) for item in value)
    return False


def _normalise_grounding_metadata(insight: dict) -> dict[str, Any]:
    client_metadata = insight.get("metadata") if isinstance(insight.get("metadata"), dict) else {}
    if not any(key in client_metadata for key in _GROUNDING_METADATA_KEYS):
        return {}

    candidate = {key: client_metadata.get(key) for key in _GROUNDING_METADATA_KEYS}
    if any(value is None or value == "" for value in candidate.values()):
        return {}
    if any(_contains_forbidden_grounding_marker(value) for value in candidate.values()):
        return {}

    role = candidate["evidence_role"]
    if not isinstance(role, str) or role not in _GROUNDING_EVIDENCE_ROLES:
        return {}

    normalized: dict[str, Any] = {"evidence_role": role}
    for key in ("slot_subject", "slot_attribute", "slot_group_id"):
        value = candidate[key]
        if not isinstance(value, str):
            return {}
        stripped = value.strip()
        if not stripped:
            return {}
        normalized[key] = stripped[:240]

    priority = candidate["grounding_priority"]
    if isinstance(priority, bool) or not isinstance(priority, (int, float)):
        return {}
    priority_value = float(priority)
    if priority_value < 0.0 or priority_value > 1.0:
        return {}
    normalized["grounding_priority"] = priority_value
    return normalized


def _session_learn_floor_breach_labels(
    *,
    input_count: int,
    output_count: int,
    created_count: int,
    evolved_count: int,
    duplicates_skipped: int,
    concepts_skipped: int,
    verbatim_attach_source: str,
    verbatim_attachment_count: int,
    verbatim_fragment_count: int,
) -> dict[str, object]:
    fallback_preserved = (
        output_count == 0
        and verbatim_attach_source == "session_fallback"
        and verbatim_attachment_count > 0
        and verbatim_fragment_count > 0
    )
    return {
        "input_count": input_count,
        "output_count": output_count,
        "created_count": created_count,
        "evolved_count": evolved_count,
        "duplicates_skipped": duplicates_skipped,
        "concepts_skipped": concepts_skipped,
        "verbatim_attach_source": verbatim_attach_source,
        "verbatim_attachment_count": verbatim_attachment_count,
        "verbatim_fragment_count": verbatim_fragment_count,
        "fallback_preserved": fallback_preserved,
    }
_PREFERENCE_POLARITIES = frozenset({"positive", "negative", "avoid", "neutral"})
_ADVICE_FACET_TYPES = frozenset({"tip", "recommendation", "constraint", "procedure", "avoidance"})
_USER_MESSAGE_PREFERENCE_SIGNAL_RE = _re.compile(
    r"\b(?:i|me|my|mine|we|our)\b.{0,80}\b(?:prefer|preference|like|dislike|enjoy|favorite|favourite|"
    r"avoid|don't like|do not like|don't recommend|do not recommend|need|want|budget|constraint|"
    r"restriction|allergy|different from|branch out|tired of)\b"
    r"|\b(?:don't recommend|do not recommend|avoid recommending|should avoid|under \$?\d+|less than \$?\d+)\b",
    _re.IGNORECASE,
)
_USER_STATED_PREFERENCE_TEXT_RE = _re.compile(
    r"\b(?:user|client|i|me|my)\b.{0,80}\b(?:prefer|preference|like|dislike|enjoy|favorite|favourite|"
    r"avoid|don't like|do not like|don't recommend|do not recommend|need|budget|constraint|"
    r"restriction|allergy|different from|branch out|tired of)\b",
    _re.IGNORECASE,
)
_PREFERENCE_NEGATIVE_SIGNAL_RE = _re.compile(
    r"\b(?:dislike|don't like|do not like|not prefer|avoid|hate|tired of)\b",
    _re.IGNORECASE,
)
_PREFERENCE_DO_NOT_RECOMMEND_SIGNAL_RE = _re.compile(
    r"\b(?:don't recommend|do not recommend|avoid recommending|should avoid|not suggest|don't suggest|do not suggest)\b",
    _re.IGNORECASE,
)
_PREFERENCE_CONSTRAINT_SIGNAL_RE = _re.compile(
    r"\b(?:constraint|constraints|budget|under \$?\d+|less than \$?\d+|without|allergy|allergies|restriction|restrictions)\b",
    _re.IGNORECASE,
)
_PREFERENCE_BRANCH_OUT_SIGNAL_RE = _re.compile(
    r"\b(?:different from|something different|branch out|beyond|instead of|tired of|not the usual)\b",
    _re.IGNORECASE,
)
_USER_ADVICE_REQUEST_RE = _re.compile(
    r"\b(?:tip|tips|recommend|recommendation|recommendations|suggest|suggestion|suggestions|"
    r"advice|help with|how to|what should i|what can i|any ideas|can you give me)\b",
    _re.IGNORECASE,
)
_ADVICE_ACTION_SIGNAL_RE = _re.compile(
    r"\b(?:tip|tips|recommend|suggest|try|use|keep|clean|organize|avoid|remember to|"
    r"step|steps|procedure|recipe|store|reheat|check|make sure|consider)\b",
    _re.IGNORECASE,
)
_SELECTION_VERB_SIGNAL_RE = _re.compile(
    r"\b(?:selected|selecting|select|chose|choose|picked|pick|decided|decided on|"
    r"go with|going with|settled on|settle on)\b",
    _re.IGNORECASE,
)
_SELECTION_PURPOSE_SIGNAL_RE = _re.compile(
    r"\b(?:for|live chat|discussion|event|meeting|club|presentation|next|future|"
    r"context|session|conversation|topic)\b",
    _re.IGNORECASE,
)
_SELECTION_TARGET_RE = _re.compile(
    r"\b(?:selected|selecting|chose|choose|picked|pick|decided on|go with|going with|settled on)\s+"
    r"(?:the\s+)?(?P<target>[^.;\n]{1,160}?)(?:\s+for\b|\s+to\b|[.;,]|$)",
    _re.IGNORECASE,
)


def _normalise_preference_facet_metadata(insight: dict, request: SessionLearnRequest) -> dict[str, Any]:
    """Return sanitized preference metadata derived only from user-stated signals."""
    if (insight.get("type") or "") != "preference":
        return {}

    evidence_text = " ".join(str(item) for item in insight.get("evidence", []) if item)
    summary = str(insight.get("summary") or "")
    user_message = request.user_message if isinstance(request.user_message, str) else ""
    user_signal = bool(_USER_MESSAGE_PREFERENCE_SIGNAL_RE.search(user_message))
    stated_signal = bool(_USER_STATED_PREFERENCE_TEXT_RE.search(f"{summary} {evidence_text}"))
    if not (user_signal or stated_signal):
        return {}

    client_metadata = insight.get("metadata") if isinstance(insight.get("metadata"), dict) else {}
    raw_facet = client_metadata.get("preference_facet")
    facet = _sanitize_preference_facet(raw_facet) if isinstance(raw_facet, dict) else {}
    if not facet:
        facet = _derive_preference_facet(summary, evidence_text, user_message, insight.get("confidence"))
    return {"preference_facet": facet} if facet else {}


def _normalise_selection_facet_metadata(insight: dict, request: SessionLearnRequest) -> dict[str, Any]:
    """Return sanitized metadata for a user-stated selected option and future purpose."""
    user_message = request.user_message if isinstance(request.user_message, str) else ""
    summary = str(insight.get("summary") or "")
    evidence_text = " ".join(str(item) for item in insight.get("evidence", []) if item)
    candidate_text = f"{user_message} {summary} {evidence_text}"
    if not (_SELECTION_VERB_SIGNAL_RE.search(candidate_text) and _SELECTION_PURPOSE_SIGNAL_RE.search(candidate_text)):
        return {}

    client_metadata = insight.get("metadata") if isinstance(insight.get("metadata"), dict) else {}
    raw_facet = client_metadata.get("selection_facet")
    facet = _sanitize_selection_facet(raw_facet) if isinstance(raw_facet, dict) else {}
    if not facet:
        facet = _derive_selection_facet(summary, evidence_text, user_message, insight.get("confidence"))
    return {"selection_facet": facet} if facet else {}


def _normalise_advice_facet_metadata(insight: dict, request: SessionLearnRequest) -> dict[str, Any]:
    """Return source-backed advice metadata without relabeling advice as preference."""
    user_message = request.user_message if isinstance(request.user_message, str) else ""
    if not _USER_ADVICE_REQUEST_RE.search(user_message):
        return {}

    summary = str(insight.get("summary") or "")
    evidence_text = " ".join(str(item) for item in insight.get("evidence", []) if item)
    assistant_response = request.assistant_response if isinstance(request.assistant_response, str) else ""
    candidate_text = f"{summary} {evidence_text}"
    if not _ADVICE_ACTION_SIGNAL_RE.search(candidate_text):
        return {}
    if not assistant_response.strip():
        return {}

    return {
        "advice_facet": _derive_advice_facet(
            summary=summary,
            evidence_text=evidence_text,
            user_message=user_message,
            assistant_response=assistant_response,
            confidence=insight.get("confidence"),
        )
    }


def _normalise_branch_provenance_metadata(insight: dict, request: SessionLearnRequest) -> dict[str, Any]:
    client_metadata = insight.get("metadata") if isinstance(insight.get("metadata"), dict) else {}
    result = preserve_or_build_branch_provenance(
        insight,
        request_metadata={
            "session_id": request.session_id,
            "agent_id": getattr(request, "agent_id", None),
            "trigger_path": getattr(request, "trigger_path", None),
        },
        source_index=client_metadata,
    )
    return result.metadata_patch


def _sanitize_preference_facet(raw_facet: dict[str, Any]) -> dict[str, Any]:
    facet: dict[str, Any] = {}
    for key, value in raw_facet.items():
        if key not in _PREFERENCE_FACET_ALLOWED_KEYS or value is None or value == "":
            continue
        if key in {"subject", "domain", "target", "observed_at"}:
            facet[key] = str(value)[:240]
        elif key == "facet_type" and str(value) in _PREFERENCE_FACET_TYPES:
            facet[key] = str(value)
        elif key == "polarity" and str(value) in _PREFERENCE_POLARITIES:
            facet[key] = str(value)
        elif key == "confidence":
            try:
                facet[key] = max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                continue
        elif key == "source_evidence" and isinstance(value, list):
            facet[key] = _sanitize_preference_source_evidence(value)
    return facet


def _sanitize_preference_source_evidence(items: list[Any]) -> list[Any]:
    sanitized: list[Any] = []
    for item in items[:5]:
        if isinstance(item, str) and item.strip():
            sanitized.append(item.strip()[:500])
        elif isinstance(item, dict):
            entry = {
                key: str(value)[:500]
                for key, value in item.items()
                if key in _PREFERENCE_SOURCE_EVIDENCE_KEYS and value is not None and value != ""
            }
            if entry:
                sanitized.append(entry)
    return sanitized


def _sanitize_selection_facet(raw_facet: dict[str, Any]) -> dict[str, Any]:
    if any(key in raw_facet for key in _FORBIDDEN_SELECTION_FACET_KEYS):
        return {}
    facet: dict[str, Any] = {}
    for key, value in raw_facet.items():
        if key not in _SELECTION_FACET_ALLOWED_KEYS or value is None or value == "":
            continue
        if key in {"subject", "domain", "target", "purpose", "observed_at"}:
            facet[key] = str(value)[:240]
        elif key == "facet_type" and str(value) in _SELECTION_FACET_TYPES:
            facet[key] = str(value)
        elif key == "confidence":
            try:
                facet[key] = max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                continue
        elif key == "source_evidence" and isinstance(value, list):
            facet[key] = _sanitize_selection_source_evidence(value)
    if "facet_type" not in facet:
        facet["facet_type"] = "selected_for_future_context"
    return facet if facet.get("target") and facet.get("source_evidence") else {}


def _sanitize_selection_source_evidence(items: list[Any]) -> list[Any]:
    sanitized: list[Any] = []
    for item in items[:5]:
        if isinstance(item, str) and item.strip():
            sanitized.append(item.strip()[:500])
        elif isinstance(item, dict):
            entry = {
                key: str(value)[:500]
                for key, value in item.items()
                if key in _SELECTION_SOURCE_EVIDENCE_KEYS and value is not None and value != ""
            }
            if entry:
                sanitized.append(entry)
    return sanitized


def _derive_selection_facet(
    summary: str,
    evidence_text: str,
    user_message: str,
    confidence: Any,
) -> dict[str, Any]:
    text = f"{user_message} {summary} {evidence_text}"
    target_match = _SELECTION_TARGET_RE.search(text)
    target = target_match.group("target").strip() if target_match else summary.strip()
    target = _re.sub(r"^[\s\-:;,.]+|[\s\-:;,.]+$", "", _re.sub(r"\s+", " ", target))[:240]
    if not target:
        return {}
    purpose_match = _re.search(
        r"\bfor\s+(?P<purpose>[^.;\n]{1,120})",
        text,
        _re.IGNORECASE,
    )
    purpose = (
        _re.sub(r"^[\s\-:;,.]+|[\s\-:;,.]+$", "", purpose_match.group("purpose"))[:240]
        if purpose_match
        else "future context"
    )
    try:
        confidence_value = max(0.0, min(1.0, float(confidence or 0.5)))
    except (TypeError, ValueError):
        confidence_value = 0.5
    domain_text = text.lower()
    domain = "general"
    if any(term in domain_text for term in ("live chat", "discussion", "conversation", "meeting", "club")):
        domain = "conversation"
    elif any(term in domain_text for term in ("presentation", "event", "session")):
        domain = "event"
    source_evidence = []
    if evidence_text:
        source_evidence.append(evidence_text[:500])
    elif user_message:
        source_evidence.append({"verbatim": user_message[:500]})
    if not source_evidence:
        return {}
    return {
        "subject": "user",
        "facet_type": "selected_for_future_context",
        "domain": domain,
        "target": target,
        "purpose": purpose,
        "confidence": confidence_value,
        "source_evidence": source_evidence,
    }


def _derive_preference_facet(
    summary: str,
    evidence_text: str,
    user_message: str,
    confidence: Any,
) -> dict[str, Any]:
    text = f"{user_message} {summary} {evidence_text}"
    facet_type = "positive"
    polarity = "positive"
    if _PREFERENCE_DO_NOT_RECOMMEND_SIGNAL_RE.search(text):
        facet_type = "do_not_recommend"
        polarity = "avoid"
    elif _PREFERENCE_BRANCH_OUT_SIGNAL_RE.search(text):
        facet_type = "branch_out"
        polarity = "neutral"
    elif _PREFERENCE_CONSTRAINT_SIGNAL_RE.search(text):
        facet_type = "constraint"
        polarity = "neutral"
    elif _PREFERENCE_NEGATIVE_SIGNAL_RE.search(text):
        facet_type = "negative"
        polarity = "negative"

    try:
        confidence_value = max(0.0, min(1.0, float(confidence or 0.5)))
    except (TypeError, ValueError):
        confidence_value = 0.5

    source_evidence = []
    if evidence_text:
        source_evidence.append(evidence_text[:500])
    elif user_message:
        source_evidence.append({"verbatim": user_message[:500]})

    return {
        "subject": "user",
        "facet_type": facet_type,
        "polarity": polarity,
        "target": summary[:240],
        "confidence": confidence_value,
        "source_evidence": source_evidence,
    }


def _derive_advice_facet(
    summary: str,
    evidence_text: str,
    user_message: str,
    assistant_response: str,
    confidence: Any,
) -> dict[str, Any]:
    text = f"{user_message} {summary} {evidence_text} {assistant_response}"
    facet_type = "tip"
    if _PREFERENCE_DO_NOT_RECOMMEND_SIGNAL_RE.search(text) or _PREFERENCE_NEGATIVE_SIGNAL_RE.search(text):
        facet_type = "avoidance"
    elif _PREFERENCE_CONSTRAINT_SIGNAL_RE.search(text):
        facet_type = "constraint"
    elif _re.search(r"\b(?:step|steps|procedure|how to|recipe)\b", text, _re.IGNORECASE):
        facet_type = "procedure"
    elif _re.search(r"\b(?:recommend|suggest)\b", text, _re.IGNORECASE):
        facet_type = "recommendation"

    try:
        confidence_value = max(0.0, min(1.0, float(confidence or 0.5)))
    except (TypeError, ValueError):
        confidence_value = 0.5

    domain = "general"
    domain_text = f"{user_message} {summary}".lower()
    if any(term in domain_text for term in ("kitchen", "countertop", "utensil", "faucet", "garbage disposal")):
        domain = "kitchen"
    elif any(term in domain_text for term in ("recipe", "dinner", "food", "sauce", "yogurt")):
        domain = "food"
    elif any(term in domain_text for term in ("hotel", "travel", "trip", "flight")):
        domain = "travel"

    return {
        "subject": "user",
        "facet_type": facet_type,
        "domain": domain,
        "target": summary[:240],
        "confidence": confidence_value,
        "requested_by_user": True,
        "source_evidence": [{"verbatim": assistant_response[:500], "role": "assistant"}],
    }


def _session_learn_benchmark_mode_active() -> bool:
    return os.environ.get("PITH_BENCHMARK_MODE", "").lower() in ("true", "1")


def _normalise_session_learn_observation_timestamp(request: SessionLearnRequest) -> str | None:
    """Return UTC ISO source-observation time supplied by benchmark ingestion."""
    raw_observation = getattr(request, "observation_date", None)
    raw_timestamp = getattr(request, "timestamp", None)
    benchmark_mode = _session_learn_benchmark_mode_active()

    def _parse(value: str | int | float | None) -> datetime | None:
        if value is None or value == "":
            return None
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=UTC)
        text = str(value).strip()
        if not text:
            return None
        try:
            return datetime.fromtimestamp(float(text), tz=UTC)
        except ValueError:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC)

    try:
        observation_dt = _parse(raw_observation)
        timestamp_dt = _parse(raw_timestamp)
    except (TypeError, ValueError) as exc:
        if benchmark_mode:
            raise ValueError("invalid session_learn observation timestamp") from exc
        return None

    if observation_dt and timestamp_dt:
        delta = abs((observation_dt - timestamp_dt).total_seconds())
        if delta > 86400 and benchmark_mode:
            raise ValueError("observation_date and timestamp disagree by more than 24h")

    chosen = observation_dt or timestamp_dt
    return chosen.isoformat() if chosen else None


class SessionLearnMixin:
    """Mixin providing sessionlearn methods for SessionManager."""

    def prepare_session_learn_binding(self, request: SessionLearnRequest) -> dict[str, Any] | None:
        """Bind explicit session_learn session_id to request-local session state."""
        requested_session_id = (request.session_id or "").strip() or None
        if not requested_session_id:
            return None

        session = self._in_memory_session(requested_session_id)
        if session is None:
            row = load_session(requested_session_id)
            session = self._session_info_from_row(row)
        if session is None or session.status != "active":
            return None

        session_token, active_token = self._push_request_session(session)
        return {
            "session_token": session_token,
            "active_token": active_token,
            "binding_source": "explicit_session_id",
            "resolved_session_id": session.session_id,
        }

    def _resolve_session_learn_request_session(self, request: SessionLearnRequest) -> bool:
        """Resolve missing session_learn session_id from active session when safe."""
        if request.session_id or not self.current_session:
            return False

        active_session_id = self.current_session.session_id
        active_agent = (getattr(self.current_session, "agent_id", "default") or "default").strip()
        request_agent = (getattr(request, "agent_id", "default") or "default").strip()
        explicit_active = active_agent not in {"", "default", "unknown"}
        explicit_request = request_agent not in {"", "default", "unknown"}

        if explicit_active and explicit_request and active_agent != request_agent:
            logger.warning(
                "session_learn: not resolving missing session_id from active session "
                "%s because request agent_id=%s differs from active agent_id=%s",
                active_session_id,
                request_agent,
                active_agent,
            )
            return False

        request.session_id = active_session_id
        logger.info(
            "session_learn: resolved missing request.session_id from active session %s",
            active_session_id,
        )
        return True

    def session_learn(self, request: SessionLearnRequest) -> SessionLearnResponse:
        """Post-response concept extraction. Target <200ms synchronous.

        Pipeline (P0.2 extended):
          Step 1: Rate limit + text preparation
          Step 2: Tier 2 processing (client-extracted) — parse, garbage detect, validate
          Step 3: Tier 1 processing (heuristic extraction)
          Step 4: Cross-tier dedup — TF-IDF cosine >=0.50, prefer Tier 2
          Step 5: Quality ranking + combined cap at 7
          Step 6: Existing-concept dedup — 3-zone: skip/evolve/create
          Step 7: Store + bookkeeping + response
        """
        t0 = time.perf_counter()

        from app.retrieval import retrieval_engine
        from app.cognitive.retrospective import ConversationProcessor

        concepts_created: list[LearnedConcept] = []
        concepts_evolved: list[EvolvedConcept] = []
        associations_created = 0
        duplicates_skipped = 0
        concepts_skipped = 0
        errors = 0
        garbage_rejected = 0
        rejection_details: list[dict] = []  # per-concept feedback
        budget_warnings: list[str] = []  # proactive budget limit signals
        concepts_superseded = 0  # S3.5
        supersession_details: list[dict] = []  # S3.5
        source_breakdown = {"heuristic": 0, "client": 0}
        _verbatim_attach_source = "none"
        _verbatim_attachment_count = 0
        _verbatim_fragment_count = 0
        evolved_this_call: set = set()  # S3: per-call evidence cap

        # --- Step 0: Session boundary check ---
        # EC12 finding: session_learn succeeds after session_end, but counters
        # don't update (self.current_session is None). We warn but still process —
        # losing concepts is worse than a counter mismatch. The warning signals
        # the client to call session_start.
        # TODO(KTA-EC12): Consider stricter session boundary enforcement —
        #   currently we process anyway to avoid concept loss, but this means
        #   session counters can drift from reality.
        session_warning = None
        if not self.current_session:
            session_warning = "no_active_session: session_learn called without active session. Concepts will be processed but session counters won't update. Call session_start first."
            logger.warning(f"session_learn: {session_warning}")
        elif request.session_id and request.session_id != self.current_session.session_id:
            _trigger = getattr(request, 'trigger_path', 'unknown')
            session_warning = (
                f"session_id_mismatch: request has {request.session_id} "
                f"but active session is {self.current_session.session_id} "
                f"[trigger={_trigger}]"
            )
            logger.warning(f"session_learn: {session_warning}")
        else:
            self._resolve_session_learn_request_session(request)

        # --- Step 1: Rate limit + text preparation ---
        allowed, retry_after = self._check_rate_limit()
        if not allowed:
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            logger.warning(
                f"session_learn: RATE_LIMIT_EXCEEDED — concepts silently dropped "
                f"(limit={self.SESSION_LEARN_RATE_LIMIT}/10min, retry_after={retry_after}s). "
                f"Concepts in this call: {len(request.extracted_concepts or [])}. "
                f"Raise PITH_SESSION_LEARN_RATE_LIMIT env var for bulk-ingest scenarios."
            )
            return SessionLearnResponse(
                concepts_created=[],
                concepts_evolved=[],
                associations_created=0,
                duplicates_skipped=0,
                concepts_skipped=0,
                errors=1,
                processing_time_ms=elapsed_ms,
                learning_events=0,
                extraction_source_breakdown=source_breakdown,
                learning_budget_remaining=self._check_daily_budget(),
                garbage_rejected=0,
                budget_warnings=[f"rate_limit_exceeded: retry after {retry_after}s"],
                concepts_superseded=0,
                supersession_details=[],
            )

        combined_text = self._prepare_text(request.user_message, request.assistant_response)
        _benchmark_mode = _session_learn_benchmark_mode_active()
        if _benchmark_mode:
            _normalise_session_learn_observation_timestamp(request)
        _benchmark_fastpath = _benchmark_mode and bool(getattr(request, "benchmark_ingest_fastpath", False))
        budget_remaining = self._check_daily_budget()

        # --- Step 2: Tier 2 processing (client-extracted concepts) ---
        tier2_insights = []
        if request.extracted_concepts:
            from app.cognitive.extraction import ExtractedConcept, GarbageDetector
            # DEBT-030: normalize_knowledge_area hoisted to module-level import

            # Parse and validate
            valid_concepts = []
            # BENCHMARK-002: Cap is configurable via PITH_MAX_INSIGHTS_PER_CALL (default 7).
            # Benchmark sends 20-concept batches; set env var to 30 to pass all through.
            _client_cap = int(os.environ.get("PITH_MAX_INSIGHTS_PER_CALL", "7"))
            from app.core.config import BENCHMARK as _bm_cap
            if _bm_cap.cap_debug_logging:
                logger.warning(
                    f"BENCHMARK-CAP-DEBUG: PITH_MAX_INSIGHTS_PER_CALL="
                    f"{os.environ.get('PITH_MAX_INSIGHTS_PER_CALL', 'NOT_SET')} → cap={_client_cap} "
                    f"concepts_sent={len(request.extracted_concepts)}"
                )
            for i, raw in enumerate(request.extracted_concepts[:_client_cap]):
                try:
                    ec = ExtractedConcept(**raw) if isinstance(raw, dict) else raw
                    valid_concepts.append(ec)
                except Exception as e:
                    logger.warning(f"session_learn: invalid extracted concept: {e}")
                    garbage_rejected += 1
                    # Extract summary preview from raw data for diagnostics
                    preview = ""
                    if isinstance(raw, dict):
                        preview = str(raw.get("summary", ""))[:80]
                    rejection_details.append(
                        {"index": i, "reason": str(e), "summary_preview": preview, "stage": "validation"}
                    )

            # Garbage detection
            # BENCHMARK-001: Skip garbage detector in benchmark mode.
            # GarbageDetector caps grounded concepts at max(5, ceil(words/200)).
            # Benchmark conversations are tiny (~35 words) → max_grounded=5 always,
            # silently discarding 15 of every 20 batch-ingested facts.
            # Facts are pre-validated by pith_agent; garbage detection is harmful here.
            from app.core.config import BENCHMARK as _bm_gc
            if valid_concepts:
                if _bm_gc.skip_garbage_detection:
                    survivors, rejections = valid_concepts, []
                else:
                    survivors, rejections = GarbageDetector.detect_batch(valid_concepts, combined_text)
                garbage_rejected += len(rejections)
                for r in rejections:
                    logger.info(f"session_learn: garbage rejected: {r['reason']} — {r['summary_preview']}")
                    rejection_details.append(
                        {
                            "index": r["index"],
                            "reason": r["reason"],
                            "summary_preview": r["summary_preview"],
                            "stage": "garbage_detection",
                        }
                    )

                # Emit budget_warnings for per-call limit hits
                for r in rejections:
                    reason = r.get("reason", "")
                    if "abstract_count_exceeded" in reason:
                        budget_warnings.append(f"per_call_abstract_limit: {reason}")
                    elif "grounded_count_exceeded" in reason:
                        budget_warnings.append(f"per_call_grounded_limit: {reason}")

                # Batch suspicion: if >50% failed, lower survivor confidence
                batch_suspicion = len(rejections) > len(valid_concepts) / 2

                # Convert survivors to insight dicts with extraction_source
                # DEBT-030: infer_knowledge_area hoisted to module-level import
                for ec in survivors:
                    client_metadata = dict(ec.metadata or {})
                    for key in (
                        "beam_source_key",
                        "beam_source_turn_id",
                        "beam_source_turn_index",
                        "beam_source_batch_idx",
                        "beam_source_role",
                        "beam_role",
                    ):
                        value = getattr(ec, key, None)
                        if value is not None and value != "":
                            client_metadata[key] = value
                    boundary = normalize_knowledge_area_boundary(
                        ec.knowledge_area,
                        summary=ec.summary,
                        concept_type=ec.concept_type,
                        strict=False,
                    )
                    normalized_area = boundary.canonical_knowledge_area
                    if boundary.raw_knowledge_area:
                        client_metadata.setdefault("raw_knowledge_area", boundary.raw_knowledge_area)
                    client_metadata.setdefault("knowledge_area_label_kind", boundary.label_kind)
                    if boundary.facet:
                        client_metadata.setdefault("knowledge_area_facet", boundary.facet)
                    conf = ec.confidence or 0.50
                    if batch_suspicion:
                        conf = min(conf, 0.40)
                    tier2_insights.append(
                        {
                            "summary": ec.summary,
                            "confidence": conf,
                            "type": ec.concept_type or "observation",
                            "signals": ec.signals or [],
                            "evidence": ec.evidence or [],
                            "knowledge_area": normalized_area,
                            "extraction_source": "client",
                            "was_untyped": ec.concept_type is None,
                            "supersedes": ec.supersedes,  # EXPLICIT_SUPERSESSION_SPEC v1.1
                            "edit_provenance": ec.edit_provenance,  # RETRIEVAL-104
                            "metadata": client_metadata,
                        }
                    )

        # --- Step 3: Tier 1 processing (heuristic extraction) ---
        tier1_insights = []
        if _benchmark_fastpath:
            logger.info("BENCH-FASTPATH: skipping Tier 1 heuristic extraction during diagnostic benchmark ingest")
        elif len(combined_text) >= 50:
            processor = ConversationProcessor()
            raw_insights = self._extract_insights(
                processor, combined_text,
                assistant_text=request.assistant_response or None,
            )
            for ins in raw_insights:
                ins.setdefault("extraction_source", "heuristic")  # DATA-068: preserve factual_scan
                # INGEST-008: Infer knowledge_area for heuristic insights.
                # Tier 2 (client) insights get KA via normalize + infer at ~line 5925.
                # Tier 1 (heuristic) insights lack KA, defaulting to "general" at
                # _process_single_insight:6769, which triggers the cross-KA guard
                # against any specific-KA match. Fix: infer KA from summary text.
                if not ins.get("knowledge_area") or ins.get("knowledge_area") == "general":
                    inferred_ka = infer_knowledge_area(ins.get("summary", ""))
                    if inferred_ka:
                        ins["knowledge_area"] = inferred_ka
                        logger.info(f"INGEST-008: KA inferred for heuristic insight: '{ins['summary'][:50]}' → {inferred_ka}")
                    else:
                        ins["knowledge_area"] = "general"
            tier1_insights = raw_insights

        # EXTRACT-C2: Run demographic safety net on combined Tier 1+2 before dedup
        all_pre_dedup = list(tier2_insights) + tier1_insights
        user_msg = request.user_message or ""
        asst_msg = request.assistant_response or ""
        if _benchmark_fastpath:
            logger.info("BENCH-FASTPATH: skipping demographic safety net during diagnostic benchmark ingest")
        elif user_msg or asst_msg:
            enriched = self._ensure_demographic_facts(all_pre_dedup, user_msg, asst_msg)
            # Any newly injected concepts go into tier1 bucket
            new_demographic = [c for c in enriched if c.get("_source", "").startswith("EXTRACT-C2")]
            for nd in new_demographic:
                nd.setdefault("extraction_source", "heuristic")
                if not nd.get("knowledge_area") or nd["knowledge_area"] == "general":
                    nd["knowledge_area"] = "personal"
            tier1_insights.extend(new_demographic)

        # --- Step 4: Cross-tier dedup ---
        # If both tiers produced insights, remove Tier 1 duplicates of Tier 2
        merged_insights = list(tier2_insights)  # Tier 2 first (preferred)
        if tier1_insights and tier2_insights:
            from app.retrieval.incremental_tfidf import IncrementalTfidfIndex

            dedup_index = IncrementalTfidfIndex()
            # Index Tier 2 summaries
            for i, t2 in enumerate(tier2_insights):
                dedup_index.add_concept(f"t2_{i}", t2["summary"])
            # Check each Tier 1 against Tier 2
            for t1 in tier1_insights:
                scores = dedup_index.search(t1["summary"], top_k=1)
                if scores and scores[0][1] >= 0.50:
                    logger.debug(f"session_learn: cross-tier dedup removed T1: {t1['summary'][:50]}")
                    continue
                merged_insights.append(t1)
        elif tier1_insights:
            merged_insights = tier1_insights

        # --- Step 5: Quality ranking + combined cap at 7 ---
        def quality_score(insight):
            score = insight.get("confidence", 0.40)
            if insight.get("evidence"):
                score += 0.1 * min(len(insight["evidence"]), 3)
            if insight.get("extraction_source") == "client":
                score += 0.05  # Slight preference for client extraction
            return score

        merged_insights.sort(key=quality_score, reverse=True)
        # BENCHMARK-002: Allow higher throughput during bulk ingestion.
        # Production default is 7 to bound per-call latency.
        _max_insights_per_call = int(os.environ.get("PITH_MAX_INSIGHTS_PER_CALL", "7"))
        merged_insights = merged_insights[:_max_insights_per_call]

        # --- INGEST-037 Phase 2a: Verbatim fragment auto-extraction ---
        # Detect high-info fragments (code, SQL, formulas, quotes) in the raw text
        # and map them to their corresponding insights. Fragments are attached to
        # each insight dict as _verbatim_fragments for downstream storage in
        # _create_new_concept.
        try:
            from app.core.config import get_feature_flag
            if get_feature_flag("VERBATIM_AUTO_EXTRACT_ENABLED", False):
                from app.cognitive.verbatim_detect import (
                    detect_verbatim_fragments,
                    match_fragments_to_insights,
                )
                _vf_fragments = detect_verbatim_fragments(combined_text)
                if _vf_fragments and merged_insights:
                    _vf_mapping = match_fragments_to_insights(
                        _vf_fragments, merged_insights
                    )
                    for idx, frags in _vf_mapping.items():
                        if 0 <= idx < len(merged_insights):
                            merged_insights[idx]["_verbatim_fragments"] = frags
                    _vf_coverage = sum(
                        1 for i in merged_insights if i.get("_verbatim_fragments")
                    )
                    logger.info(
                        "INGEST-037: Detected %d verbatim fragments, "
                        "mapped to %d/%d insights (coverage %.0f%%)",
                        len(_vf_fragments),
                        len(_vf_mapping),
                        len(merged_insights),
                        (_vf_coverage / len(merged_insights) * 100) if merged_insights else 0,
                    )
        except Exception as _vf_err:
            logger.warning(
                "INGEST-037: Verbatim auto-extraction failed (non-fatal): %s",
                _vf_err,
            )

        if not merged_insights:
            # INGEST-043: Capture verbatim even when no insights extracted.
            # Without this, conversations with no extractable concepts (e.g.,
            # numbered lists, continuation pairs) lose their raw text permanently.
            _benchmark_skip_cap = os.environ.get("PITH_BENCHMARK_MODE", "").lower() in ("true", "1")
            _elapsed_early_ms = (time.perf_counter() - t0) * 1000
            if request.user_message and request.assistant_response:
                try:
                    from app.core.config import get_feature_flag as _i043_ff
                    if _i043_ff("VERBATIM_CONVERSATION_CAPTURE_ENABLED", True):
                        # Try session concepts first, then orphan fallback
                        _early_ids = []
                        if hasattr(self, '_session_concept_ids') and self._session_concept_ids:
                            _early_ids = list(self._session_concept_ids)[-3:]
                        if _early_ids:
                            from app.cognitive.verbatim_detect import capture_conversation_verbatim
                            _conv_frag_ids = capture_conversation_verbatim(
                                user_message=request.user_message,
                                assistant_response=request.assistant_response,
                                concept_ids=_early_ids,
                                concept_versions={},
                            )
                            if _conv_frag_ids:
                                logger.info(
                                    "INGEST-043: Early-return verbatim captured %d fragments "
                                    "for %d session concepts (%.0fms elapsed)",
                                    len(_conv_frag_ids), len(_early_ids), _elapsed_early_ms,
                                )
                        else:
                            # No concept IDs at all — write directly to fts_verbatim
                            import uuid as _uuid043e
                            _orphan_fid = f"vf_orphan_{_uuid043e.uuid4().hex[:12]}"
                            _orphan_content = (
                                "[USER] " + request.user_message
                                + "\n\n[ASSISTANT] " + request.assistant_response
                            )
                            from app.storage import _db as _i043e_db
                            with _i043e_db() as _oe_conn:
                                _oe_conn.execute(
                                    "DELETE FROM fts_verbatim WHERE fragment_id = ?",
                                    (_orphan_fid,),
                                )
                                _oe_conn.execute(
                                    "INSERT INTO fts_verbatim"
                                    "(fragment_id, concept_id, user_content, full_content) "
                                    "VALUES (?, ?, ?, ?)",
                                    (_orphan_fid, "orphan_verbatim",
                                     request.user_message, _orphan_content),
                                )
                            logger.info(
                                "INGEST-043: Early-return orphan verbatim written to fts_verbatim "
                                "(%d chars, %.0fms elapsed)",
                                len(_orphan_content), _elapsed_early_ms,
                            )
                except Exception as _early_err:
                    logger.warning("INGEST-043: Early-return verbatim failed: %s", _early_err)
            elif not request.user_message and request.assistant_response:
                # INGEST-050: Log when verbatim skipped due to empty user_message.
                # Primary cause: conversation_turn auto-learn with missing previous_message.
                logger.warning(
                    "INGEST-050: Verbatim capture SKIPPED — user_message empty. "
                    "assistant_response present (%d chars) but ungated.",
                    len(request.assistant_response),
                )

            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            return SessionLearnResponse(
                concepts_created=[],
                concepts_evolved=[],
                associations_created=0,
                duplicates_skipped=0,
                concepts_skipped=0,
                errors=errors,
                processing_time_ms=elapsed_ms,
                learning_events=0,
                extraction_source_breakdown=source_breakdown,
                learning_budget_remaining=budget_remaining,
                garbage_rejected=garbage_rejected,
                rejection_details=rejection_details,
                budget_warnings=list(dict.fromkeys(budget_warnings)),
                session_warning=session_warning,
                concepts_superseded=0,
                supersession_details=[],
            )

        # --- PERF-021/036: Pre-compute batch dedup candidates (DB I/O reduction) ---
        # PERF-021: TF-IDF batch (original). PERF-036: Embedding batch (extends to embedding path).
        # Replaces N sequential encode+search+DB calls with 1 batch encode + 1 WHERE IN query.
        # Falls back to per-call search if batch fails (graceful degradation).
        _batch_dedup: list[list[dict]] | None = None
        try:
            from app.core.config import FEATURE_FLAGS as _perf021_ff

            _perf021_use_embedding = _perf021_ff.get("EMBEDDING_DEDUP_ENABLED", False)
            if len(merged_insights) > 1:
                _batch_summaries = [i.get("summary", "") for i in merged_insights]
                if _perf021_use_embedding:
                    # PERF-036: Batch embedding dedup
                    _batch_dedup = retrieval_engine.search_for_dedup_embedding_batch(
                        _batch_summaries, top_k=3
                    )
                else:
                    _batch_dedup = retrieval_engine.search_for_dedup_tfidf_batch(
                        _batch_summaries, top_k=3
                    )
        except Exception as _perf021_e:
            logger.warning(f"PERF-021/036: batch dedup pre-compute failed, falling back to per-call: {_perf021_e}")
            _batch_dedup = None

        # --- Step 6-7: Process each insight through dedup + create/evolve ---
        # PERF-038: Separate overhead timer from per-insight timer.
        # Pipeline overhead (extraction, dedup, batch precompute) should not count
        # against the per-insight budget. For 8k+ concept brains, overhead alone
        # consumed 1500-3000ms of the flat 5000ms budget, causing silent concept loss.
        t_insights = time.perf_counter()
        _overhead_ms = (t_insights - t0) * 1000

        from app.core.config import AUTOLEARN_BUDGET_MS as _learn_budget
        from app.core.config import AUTOLEARN_PER_INSIGHT_BUDGET_MS as _per_insight_budget
        from app.core.config import AUTOLEARN_MAX_BUDGET_MS as _max_budget

        # PERF-038: Scale budget by insight count, capped at max.
        _effective_budget = min(
            _learn_budget + _per_insight_budget * max(0, len(merged_insights) - 1),
            _max_budget,
        )
        logger.info(
            f"PERF-038: autolearn budget: overhead={_overhead_ms:.0f}ms "
            f"effective_budget={_effective_budget}ms (base={_learn_budget} + "
            f"{_per_insight_budget}ms * {max(0, len(merged_insights) - 1)} insights, "
            f"cap={_max_budget}ms)"
        )

        explicit_supersession_total = 0  # M1: aggregate cap (EXPLICIT_SUPERSESSION_SPEC v1.1)
        _budget_exhausted = False

        # Cache only triples observed or created during this pipeline call.
        # auto_associate_single performs bounded DB duplicate checks per candidate set.
        self._cached_association_triples = set()
        _benchmark_no_budget = os.environ.get("PITH_BENCHMARK_MODE", "").lower() in ("true", "1")
        for idx, insight in enumerate(merged_insights):
            # PERF-038: Time budget check — uses insight-only timer (excludes pipeline overhead).
            # Always process first insight regardless of budget.
            _insight_elapsed_ms = (time.perf_counter() - t_insights) * 1000
            if idx > 0 and _insight_elapsed_ms > _effective_budget and not _benchmark_no_budget:
                _skipped_count = len(merged_insights) - idx
                logger.warning(
                    f"session_learn: PERF-038 time budget exhausted "
                    f"(insight_time={_insight_elapsed_ms:.0f}ms > budget={_effective_budget}ms, "
                    f"overhead={_overhead_ms:.0f}ms) after "
                    f"{idx}/{len(merged_insights)} insights, skipping {_skipped_count}"
                )
                concepts_skipped += _skipped_count
                _budget_exhausted = True
                break
            try:
                ext_source = insight.get("extraction_source", "heuristic")

                # M1: Strip supersedes if aggregate cap reached (EXPLICIT_SUPERSESSION_SPEC v1.1)
                if explicit_supersession_total >= 10 and insight.get("supersedes"):
                    logger.info("EXPLICIT_SUPERSESSION: aggregate cap (10) reached, stripping supersedes")
                    insight["supersedes"] = None

                result = self._process_single_insight(
                    insight=insight,
                    request=request,
                    retrieval_engine=retrieval_engine,
                    extraction_source=ext_source,
                    evolved_this_call=evolved_this_call,
                    budget_remaining=self._check_daily_budget(),
                    precomputed_dedup=(
                        _batch_dedup[idx] if _batch_dedup and idx < len(_batch_dedup) else None
                    ),  # PERF-021
                )
                if result["action"] == "created":
                    concepts_created.append(result["learned_concept"])
                    associations_created += result.get("associations", 0)
                    source_breakdown[ext_source] = source_breakdown.get(ext_source, 0) + 1
                    self._consume_budget(knowledge_area=insight.get("knowledge_area", "unknown"))
                    # SESSION-005: Invalidate batch dedup cache after creation.
                    # Newly created concept is in the live index (L6 in _create_new_concept),
                    # but pre-computed batch results (PERF-021) don't see it.
                    # Force subsequent insights to use fresh live search.
                    _batch_dedup = None
                    # S3.5: Track supersessions
                    if "superseded" in result:
                        concepts_superseded += 1
                        supersession_details.append(result["superseded"])
                    # M1: Track explicit supersessions for aggregate cap
                    if "explicit_supersessions" in result:
                        explicit_supersession_total += result["explicit_supersessions"]
                        concepts_superseded += result["explicit_supersessions"]
                elif result["action"] == "evolved":
                    concepts_evolved.append(result["evolved_concept"])
                    associations_created += result.get("associations", 0)
                    source_breakdown[ext_source] = source_breakdown.get(ext_source, 0) + 1
                elif result["action"] == "skipped_duplicate":
                    duplicates_skipped += 1
                elif result["action"] == "skipped_per_call_cap":
                    duplicates_skipped += 1
                elif result["action"] == "skipped_confidence":
                    concepts_skipped += 1
                elif result["action"] == "skipped_saturated":
                    duplicates_skipped += 1
            except Exception as e:
                logger.error(f"session_learn: insight processing failed: {e}")
                errors += 1

        # --- INGEST-034: Fire background event extraction ---
        from app.core.config import EE_ENABLED, EE_MIN_CONVERSATION_LENGTH

        if (
            not _benchmark_mode
            and EE_ENABLED
            and concepts_created
            and len(combined_text) >= EE_MIN_CONVERSATION_LENGTH
        ):
            _ee_concept_ids = [lc.concept_id for lc in concepts_created]
            import concurrent.futures as _cf_ee
            if not hasattr(self, '_event_executor') or self._event_executor is None:
                self._event_executor = _cf_ee.ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="event_extract"
                )
            self._event_executor.submit(
                self._extract_events,
                combined_text,
                _ee_concept_ids,
                request.session_id,
            )
            logger.info(
                f"INGEST-034: Event extraction fired for {len(_ee_concept_ids)} concepts "
                f"(text_len={len(combined_text)})"
            )
        elif _benchmark_mode:
            logger.info("INGEST-034: Skipped event extraction in benchmark mode")

        # --- INGEST-038: Capture raw conversation text as verbatim fragments ---
        # CRITICAL: Use request.user_message / request.assistant_response (raw client input),
        # NOT combined_text (which may be preprocessed). Lossless capture requires raw source.
        # EUNOMIA-040 Fix 5: Budget gate REMOVED — verbatim always captures.
        # When BACKGROUND_AUTOLEARN_ENABLED=True (default), session_learn runs post-response
        # via _background_autolearn, so no latency impact. For sync fallback, 50-200ms is
        # acceptable vs permanent text loss (15 events observed with old gate).
        _benchmark_skip_cap = os.environ.get("PITH_BENCHMARK_MODE", "").lower() in ("true", "1")
        _elapsed_total_ms = round((time.perf_counter() - t0) * 1000, 1)  # Sprint-1: fix NameError at lines 675,704
        if True or _benchmark_skip_cap:  # EUNOMIA-040: always execute (gate removed)
            try:
                from app.core.config import get_feature_flag as _ingest038_ff
                if _ingest038_ff("VERBATIM_CONVERSATION_CAPTURE_ENABLED", True):
                    from app.cognitive.verbatim_detect import capture_conversation_verbatim
                    # INGEST-043: Use created + evolved IDs; fall back to session concepts
                    _created_ids = [lc.concept_id for lc in concepts_created]
                    _evolved_ids = [ec.concept_id for ec in concepts_evolved]
                    _attach_ids = _created_ids + _evolved_ids
                    _attach_source = "created+evolved"
                    if not _attach_ids and hasattr(self, '_session_concept_ids') and self._session_concept_ids:
                        # Fallback: attach to most recent session concept when dedup skipped all insights
                        _attach_ids = list(self._session_concept_ids)[-3:]  # Last 3 as attachment points
                        _attach_source = "session_fallback"
                        logger.info(
                            "INGEST-043: No created/evolved concepts; using %d session concept(s) as fallback",
                            len(_attach_ids),
                        )
                    _verbatim_attach_source = _attach_source if _attach_ids else "none"
                    _verbatim_attachment_count = len(_attach_ids)
                    if _attach_ids and request.user_message:
                        _conv_frag_ids = capture_conversation_verbatim(
                            user_message=request.user_message,
                            assistant_response=request.assistant_response or "",
                            concept_ids=_attach_ids,
                            concept_versions={
                                c.concept_id: getattr(c, "version", None)
                                for c in concepts_created
                            },
                        )
                        if _conv_frag_ids:
                            _verbatim_fragment_count = len(_conv_frag_ids)
                            _elapsed_total_ms = round((time.perf_counter() - t0) * 1000, 2)
                            logger.info(
                                "INGEST-038: Captured %d conversation fragments "
                                "for %d concepts (%s, %.0fms elapsed)",
                                len(_conv_frag_ids),
                                len(_attach_ids),
                                _attach_source,
                                _elapsed_total_ms,
                            )
                    elif not _attach_ids and request.user_message and request.assistant_response:
                        # INGEST-043: No concept IDs at all (first call, dedup skipped).
                        # Write directly to fts_verbatim for R080 keyword retrieval.
                        # No verbatim_fragments row — this is retrieval-only, not concept-linked.
                        import uuid as _uuid043
                        _orphan_fid = f"vf_orphan_{_uuid043.uuid4().hex[:12]}"
                        _orphan_content = (
                            "[USER] " + request.user_message
                            + "\n\n[ASSISTANT] " + request.assistant_response
                        )
                        try:
                            from app.storage import _db as _ingest043_db
                            with _ingest043_db() as _o_conn:
                                _o_user = request.user_message
                                _o_conn.execute(
                                    "DELETE FROM fts_verbatim WHERE fragment_id = ?",
                                    (_orphan_fid,),
                                )
                                _o_conn.execute(
                                    "INSERT INTO fts_verbatim"
                                    "(fragment_id, concept_id, user_content, full_content) "
                                    "VALUES (?, ?, ?, ?)",
                                    (_orphan_fid, "orphan_verbatim", _o_user, _orphan_content),
                                )
                            _elapsed_total_ms = round((time.perf_counter() - t0) * 1000, 2)
                            _verbatim_attach_source = "orphan"
                            _verbatim_attachment_count = 0
                            _verbatim_fragment_count = 1
                            logger.info(
                                "INGEST-043: Orphan verbatim written to fts_verbatim "
                                "(%d chars, %.0fms elapsed)",
                                len(_orphan_content),
                                _elapsed_total_ms,
                            )
                        except Exception as _o_err:
                            logger.warning("INGEST-043: Orphan verbatim failed: %s", _o_err)
            except Exception as _conv_err:
                logger.warning(
                    "INGEST-038: Conversation capture failed (non-fatal): %s",
                    _conv_err,
                )
        # EUNOMIA-040 Fix 5: Budget-gated skip path removed.
        # Verbatim capture always runs. No more permanent text loss.

        # --- S6: Source-tagged logging ---
        learning_events = len(concepts_created) + len(concepts_evolved)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        # OBS-03: emit session_learn latency to metrics DB (mirrors conversation_turn pattern)
        try:
            from app.ops.metrics import metrics as _sl_metrics

            _sl_metrics.record("session_learn_latency_ms", elapsed_ms)
        except Exception:
            pass  # Metrics are best-effort

        # Deduplicate budget_warnings
        budget_warnings = list(dict.fromkeys(budget_warnings))

        # --- Self-awareness: update session performance counters ---
        # SESSION-LEARN-MISMATCH-001: Use request.session_id (dispatch-time) for counter
        # attribution, not self.current_session (execution-time). Background auto-learn
        # (PERF-FORT-2) and client cache staleness can both cause self.current_session
        # to differ from the session that generated this learn request.
        _counter_sid = request.session_id or (
            self.current_session.session_id if self.current_session else None
        )
        if _counter_sid and learning_events > 0:
            # Fetch current counters from DB to avoid stale in-memory state
            from app.storage.sessions import get_session_counts
            _current_counts = get_session_counts(_counter_sid)
            _new_created = (_current_counts.get("concepts_created", 0) or 0) + len(concepts_created)
            _new_evolved = (_current_counts.get("concepts_evolved", 0) or 0) + len(concepts_evolved)
            update_session(
                _counter_sid,
                concepts_created=_new_created,
                concepts_evolved=_new_evolved,
            )
            # Keep in-memory state consistent IF this is the current session
            if self.current_session and self.current_session.session_id == _counter_sid:
                self.current_session.concepts_created = _new_created
                self.current_session.concepts_evolved = _new_evolved
            # SESSION-LEARN-MISMATCH-001 F3.1: Log when counter attribution is redirected
            elif self.current_session:
                logger.info(
                    f"session_learn: counter attribution redirected "
                    f"{self.current_session.session_id} → {_counter_sid}"
                )

        untyped_count = sum(1 for i in merged_insights if i.get("was_untyped", False))

        # --- Wave 4b: Create learning event trace + set source_trace_id [X2] ---
        try:
            from app.ops.traces import create_trace

            if concepts_created or concepts_evolved:
                concept_ref_ids = [c.concept_id for c in concepts_created] + [c.concept_id for c in concepts_evolved]
                sid = request.session_id or (self.current_session.session_id if self.current_session else "unknown")
                trace = create_trace(
                    session_id=sid,
                    trigger_type="learning_event",
                    situation=f"session_learn processing {len(merged_insights)} insights",
                    intent="Extract and persist knowledge from conversation",
                    assessment=f"Created {len(concepts_created)}, evolved {len(concepts_evolved)}",
                    justification=f"Tier1={len(tier1_insights)}, Tier2={len(tier2_insights)}",
                    concept_refs=concept_ref_ids,
                )
                # Set source_trace_id on newly created concepts [X2]
                for lc in concepts_created:
                    try:
                        from app.storage import load_concept as _lc
                        from app.storage import save_concept as _sc

                        c = _lc(lc.concept_id, track_access=False)
                        if c and not c.source_trace_id:
                            c.source_trace_id = trace.id
                            _sc(c)
                    except Exception:
                        pass  # Best-effort linkage
        except Exception as e:
            logger.debug(f"Wave 4b: trace creation skipped: {e}")

        # --- Wave 4b: Resolve predictions for evolved concepts [FIX I1] ---
        try:
            from app.ops.traces import resolve_predictions_for_concept

            for ec in concepts_evolved:
                resolve_predictions_for_concept(ec.concept_id, outcome="revised", outcome_source="evolution")
        except Exception as e:
            logger.debug(f"Wave 4b: prediction resolution skipped: {e}")

        # --- Wave 5: Auto-link new concepts to active threads ---
        try:
            from app.features.threads import (
                auto_link_candidates,
                build_thread_guardrail_cache,
                link_concept_to_thread,
                load_threads,
            )
            from app.ops.metrics import metrics as _thread_metrics

            active_threads = load_threads(status="active") if concepts_created else []
            guardrail_cache = build_thread_guardrail_cache(active_threads) if active_threads else {}
            if active_threads:
                for lc in concepts_created:
                    concept = _lc(lc.concept_id, track_access=False)
                    if concept:
                        decisions = auto_link_candidates(
                            concept,
                            active_threads,
                            guardrail_cache=guardrail_cache,
                        )
                        for decision in decisions:
                            if not decision.get("admit"):
                                _thread_metrics.record("thread_autolink_rejected", 1.0)
                                _thread_metrics.record(
                                    "thread_autolink_reject_reason",
                                    1.0,
                                    {"reason": decision.get("reason_code", "unknown")},
                                )
                                continue
                            # LIFECYCLE-001: Auto-classify role based on concept_type
                            from app.features.threads import classify_thread_role

                            role = classify_thread_role(
                                concept.id,
                                concept.concept_type,
                                decision["thread_id"],
                            )
                            link_concept_to_thread(
                                decision["thread_id"],
                                concept.id,
                                role=role,
                                added_by="auto",
                            )
                            _thread_metrics.record(
                                "thread_autolink_admitted",
                                1.0,
                                {"reason": decision.get("reason_code", "unknown")},
                            )
        except Exception as e:
            logger.debug(f"Wave 5: thread auto-link skipped: {e}")

        logger.info(
            f"session_learn_pipeline: "
            f"tier1={len(tier1_insights)} tier2={len(tier2_insights)} "
            f"merged={len(merged_insights)} created={len(concepts_created)} "
            f"evolved={len(concepts_evolved)} superseded={concepts_superseded} "
            f"garbage={garbage_rejected} untyped={untyped_count} "
            f"budget={self._check_daily_budget()} time={elapsed_ms}ms "
            f"sources={source_breakdown} budget_exhausted={_budget_exhausted}"
        )

        # MONITOR-001: Emit pipeline-level observability metrics
        try:
            from app.ops.metrics import metrics as _lo_metrics
            _lo_metrics.record("learn_pipeline_created", float(len(concepts_created)))
            _lo_metrics.record("learn_pipeline_evolved", float(len(concepts_evolved)))
            _lo_metrics.record("learn_pipeline_skipped", float(concepts_skipped + garbage_rejected))
            _lo_metrics.record("learn_pipeline_latency_ms", elapsed_ms)
            _lo_metrics.record("learn_budget_remaining", float(self._check_daily_budget()))
        except Exception:
            pass  # Metrics are best-effort

        # MONITOR-133: session_learn pipeline floor alarm
        _input_count = len(request.extracted_concepts or []) + len(tier1_insights)
        _output_count = len(concepts_created) + len(concepts_evolved)
        if _input_count >= 3 and _output_count == 0:
            logger.warning(
                "MONITOR-133: session_learn pipeline floor breach — "
                "%d input concepts/insights produced 0 output concepts. "
                "Possible extraction regression or over-aggressive dedup.",
                _input_count,
            )
            try:
                from app.ops.metrics import metrics as _floor_m
                _floor_m.record(
                    "session_learn_floor_breach",
                    1.0,
                    _session_learn_floor_breach_labels(
                        input_count=_input_count,
                        output_count=_output_count,
                        created_count=len(concepts_created),
                        evolved_count=len(concepts_evolved),
                        duplicates_skipped=duplicates_skipped,
                        concepts_skipped=concepts_skipped,
                        verbatim_attach_source=_verbatim_attach_source,
                        verbatim_attachment_count=_verbatim_attachment_count,
                        verbatim_fragment_count=_verbatim_fragment_count,
                    ),
                )
            except Exception:
                pass

        # PERF-038: Record budget exhaustion metric for monitoring
        if _budget_exhausted:
            try:
                from app.ops.metrics import metrics as _d03_metrics
                _d03_metrics.record("autolearn_budget_exhausted", 1.0, {"skipped": concepts_skipped})
            except Exception:
                pass
            budget_warnings.append(
                f"autolearn_time_budget: insight_time={_insight_elapsed_ms:.0f}ms exceeded "
                f"effective_budget={_effective_budget}ms (overhead={_overhead_ms:.0f}ms), "
                f"{concepts_skipped} insights skipped"
            )

        return SessionLearnResponse(
            concepts_created=concepts_created,
            concepts_evolved=concepts_evolved,
            associations_created=associations_created,
            duplicates_skipped=duplicates_skipped,
            concepts_skipped=concepts_skipped,
            errors=errors,
            processing_time_ms=elapsed_ms,
            learning_events=learning_events,
            extraction_source_breakdown=source_breakdown,
            learning_budget_remaining=self._check_daily_budget(),
            garbage_rejected=garbage_rejected,
            rejection_details=rejection_details,
            budget_warnings=budget_warnings,
            session_warning=session_warning,
            concepts_superseded=concepts_superseded,
            supersession_details=supersession_details,
        )

    def _prepare_text(self, user_message: str, assistant_response: str) -> str:
        """L1: Combine and clean text for insight extraction.

        Strips common boilerplate, greetings, and filler phrases.
        Returns cleaned combined text.
        """
        # Strip common AI boilerplate from assistant response
        boilerplate = [
            "Sure, ",
            "Of course, ",
            "Absolutely, ",
            "Great question! ",
            "I'd be happy to ",
            "Let me ",
            "Here's ",
            "I think ",
            "That's a great ",
            "Good question, ",
        ]
        cleaned_response = assistant_response
        for prefix in boilerplate:
            if cleaned_response.startswith(prefix):
                cleaned_response = cleaned_response[len(prefix) :]
                break

        # Strip user greetings
        greetings = ["hi ", "hey ", "hello ", "thanks ", "thank you "]
        cleaned_user = user_message
        lower_user = cleaned_user.lower()
        for g in greetings:
            if lower_user.startswith(g):
                cleaned_user = cleaned_user[len(g) :]
                break

        return f"{cleaned_user.strip()} {cleaned_response.strip()}"

    # EXTRACT-C2: Post-extraction demographic fact safety net

    @staticmethod
    def _ensure_demographic_facts(concepts: list[dict],
                                  user_msg: str, assistant_msg: str) -> list[dict]:
        """EXTRACT-C2: Post-extraction safety net for critical demographic facts.

        Scans raw conversation text for explicit age/birthday mentions and injects
        a synthetic concept if the LLM extraction missed it. This prevents
        extraction non-determinism from losing high-value retrieval anchors.

        Evidence: LongMemEval 157a136e regressed True→False when gpt-4o-mini
        non-deterministically dropped the user's age — the only concept carrying
        that signal. Without it, cross-session arithmetic was unsolvable.
        """
        import re as _re

        combined = f"{user_msg}\n{assistant_msg}".lower()
        existing_summaries = " ".join(
            c.get("summary", "").lower() for c in concepts if isinstance(c, dict)
        )

        injected = 0

        # Pattern 1: Explicit age — "I'm 32", "I am 25", "turned 28"
        # (?!\s*%) prevents false positive on "I'm 100% sure"
        age_patterns = [
            _re.compile(r"\bi['\u2019]?m\s+(\d{1,3})(?!\s*%)\b", _re.IGNORECASE),
            _re.compile(r"\bi am\s+(\d{1,3})\s*(?:years?\s*old)?(?!\s*%)\b", _re.IGNORECASE),
            _re.compile(r"\b(?:just\s+)?turned\s+(\d{1,3})\b", _re.IGNORECASE),
            _re.compile(r"\bmy\s+(\d{1,3})(?:st|nd|rd|th)\s+birthday\b", _re.IGNORECASE),
            _re.compile(r"\bat\s+(\d{1,3})[,]\s+(?:you|i|we)\b", _re.IGNORECASE),
        ]
        for pattern in age_patterns:
            m = pattern.search(combined)
            if m:
                age = int(m.group(1))
                if 16 <= age <= 95:  # plausible human age (tighter than benchmark)
                    age_str = str(age)
                    if age_str not in existing_summaries:
                        concepts.append({
                            "summary": f"The user is {age} years old",
                            "confidence": 0.8,
                            "knowledge_area": "personal",
                            "concept_type": "observation",
                            "evidence": [m.group(0)[:80]],
                            "is_factual": True,
                            "temporal_category": "identity",
                            "_source": "EXTRACT-C2-demographic-safety-net",
                        })
                        injected += 1
                        logger.info(f"EXTRACT-C2: Injected missing age fact: user is {age} "
                                    f"(pattern: {pattern.pattern})")
                    break  # One age fact is enough

        # Pattern 2: Decade of life — "in my 30s", "being in my 20s"
        if injected == 0:
            decade_match = _re.search(
                r"\b(?:in\s+)?my\s+(\d0)s\b", combined, _re.IGNORECASE
            )
            if decade_match:
                decade = decade_match.group(1)
                if decade + "s" not in existing_summaries:
                    concepts.append({
                        "summary": f"The user is in their {decade}s",
                        "confidence": 0.7,
                        "knowledge_area": "personal",
                        "concept_type": "observation",
                        "evidence": [decade_match.group(0)[:80]],
                        "is_factual": True,
                        "temporal_category": "identity",
                        "_source": "EXTRACT-C2-demographic-safety-net",
                    })
                    injected += 1
                    logger.info(f"EXTRACT-C2: Injected missing decade fact: user in {decade}s")

        # Pattern 3: Birth year — "born in 1992", "birth year is 1990"
        birth_match = _re.search(
            r"\bborn\s+(?:in\s+)?(\d{4})\b", combined, _re.IGNORECASE
        )
        if birth_match:
            year = birth_match.group(1)
            if year not in existing_summaries:
                concepts.append({
                    "summary": f"The user was born in {year}",
                    "confidence": 0.8,
                    "knowledge_area": "personal",
                    "concept_type": "observation",
                    "evidence": [birth_match.group(0)[:80]],
                    "is_factual": True,
                    "temporal_category": "identity",
                    "_source": "EXTRACT-C2-demographic-safety-net",
                })
                injected += 1
                logger.info(f"EXTRACT-C2: Injected missing birth year: {year}")

        if injected:
            logger.info(f"EXTRACT-C2: Safety net injected {injected} demographic fact(s)")
        return concepts

    # PERF-001: Tier 3 LLM Extraction (background, non-blocking)

    async def _tier3_llm_extraction(
        self,
        user_message: str,
        assistant_response: str,
        existing_insights: list[dict],
        request: "SessionLearnRequest",
    ) -> None:
        """PERF-001: Background Tier 3 LLM extraction.

        Runs async after conversation_turn returns. Calls Haiku to extract
        additional concepts from the conversation that Tier 1+2 missed.
        Results are processed through the standard session_learn pipeline.
        """
        import os
        import time
        from datetime import datetime

        from app.core.config import (
            TIER3_DAILY_BUDGET,
            TIER3_LLM_MODEL,
            TIER3_MAX_CONCEPTS_PER_CALL,
            TIER3_MAX_OUTPUT_TOKENS,
            TIER3_MIN_CONVERSATION_LENGTH,
        )

        t0 = time.perf_counter()

        # Gate 1: Minimum conversation length
        combined_len = len(user_message or "") + len(assistant_response or "")
        if combined_len < TIER3_MIN_CONVERSATION_LENGTH:
            logger.debug("PERF-001: Skipping Tier 3 — conversation too short")
            return

        # Gate 2: API key available
        if not os.environ.get("OPENROUTER_API_KEY"):
            logger.debug("PERF-001: Skipping Tier 3 — no OPENROUTER_API_KEY")
            return

        # Gate 3: Daily budget check
        if not hasattr(self, "_tier3_calls_today"):
            self._tier3_calls_today = 0
            self._tier3_day = datetime.now(UTC).date()

        current_day = datetime.now(UTC).date()
        if current_day != self._tier3_day:
            self._tier3_calls_today = 0
            self._tier3_day = current_day

        if self._tier3_calls_today >= TIER3_DAILY_BUDGET:
            logger.info(f"PERF-001: Tier 3 daily budget exhausted ({TIER3_DAILY_BUDGET})")
            return

        # Gate 4: Cooldown check
        if hasattr(self, "_tier3_last_call"):
            from app.core.config import TIER3_COOLDOWN_SECONDS

            elapsed = time.perf_counter() - self._tier3_last_call
            if elapsed < TIER3_COOLDOWN_SECONDS:
                logger.debug(f"PERF-001: Tier 3 cooldown ({elapsed:.1f}s < {TIER3_COOLDOWN_SECONDS}s)")
                return

        try:
            from app.cognitive.extraction import build_tier3_prompt, parse_tier3_response
            from app.cognitive.taxonomy import get_ka_hints

            # KA-INJECT-001: Fetch user's KA vocabulary for extraction guidance
            try:
                ka_hint_list = get_ka_hints(max_hints=12)
            except Exception:
                ka_hint_list = None  # Fallback handled in build_tier3_prompt

            # Build prompt
            prompt = build_tier3_prompt(
                user_message=user_message,
                assistant_response=assistant_response,
                existing_concepts=existing_insights,
                max_concepts=TIER3_MAX_CONCEPTS_PER_CALL,
                session_date=getattr(self, '_session_date', None),  # RAGAS-DIAG-001 Fix 3c
                ka_hints=ka_hint_list,  # KA-INJECT-001
            )

            # Call LLM via OpenRouter (COST-001: switched from Anthropic direct billing to OpenRouter)
            from openai import AsyncOpenAI as _AsyncOAI

            _or_key = os.environ.get("OPENROUTER_API_KEY", "")
            client = _AsyncOAI(base_url="https://openrouter.ai/api/v1", api_key=_or_key)
            response = await client.chat.completions.create(
                model=TIER3_LLM_MODEL,
                max_tokens=TIER3_MAX_OUTPUT_TOKENS,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )

            raw_text = response.choices[0].message.content or ""
            self._tier3_calls_today += 1
            self._tier3_last_call = time.perf_counter()

            # Record cost metric
            from app.ops.metrics import metrics as _t3_metrics

            _t3_metrics.record(
                "tier3_llm_call",
                1.0,
                {
                    "model": TIER3_LLM_MODEL,
                    "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "output_tokens": response.usage.completion_tokens if response.usage else 0,
                },
            )

            # Parse response
            tier3_concepts = parse_tier3_response(raw_text, TIER3_MAX_CONCEPTS_PER_CALL)

            # EXTRACT-C2: Post-extraction demographic fact safety net
            tier3_concepts = self._ensure_demographic_facts(
                tier3_concepts, user_message, assistant_response
            )

            if not tier3_concepts:
                logger.info("PERF-001: Tier 3 extracted 0 additional concepts")
                return

            logger.info(f"PERF-001: Tier 3 extracted {len(tier3_concepts)} concepts, processing...")

            # Process through standard session_learn pipeline
            from app.retrieval import retrieval_engine

            evolved_this_call = set()  # Gauntlet A1: shared across batch
            for insight in tier3_concepts:
                try:
                    result = self._process_single_insight(
                        insight=insight,
                        request=request,
                        retrieval_engine=retrieval_engine,
                        extraction_source="llm_tier3",
                        evolved_this_call=evolved_this_call,
                        budget_remaining=TIER3_MAX_CONCEPTS_PER_CALL,  # Gauntlet A2: independent of main budget
                    )
                    if result["action"] in ("created", "evolved"):
                        logger.info(f"PERF-001: Tier 3 {result['action']}: {insight['summary'][:80]}")
                except Exception as e:
                    logger.warning(f"PERF-001: Tier 3 insight processing failed: {e}")

            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            logger.info(f"PERF-001: Tier 3 completed in {elapsed_ms:.0f}ms")

        except Exception as e:
            logger.error(f"PERF-001: Tier 3 extraction failed: {e}")
            from app.ops.metrics import metrics as _t3_fail_metrics
            _t3_fail_metrics.record("extraction_llm_failure", 1.0, {"source": "tier3", "error": str(e)[:120]})

    # INGEST-022: Temporal anchor detection regex for enriching extracted concepts.
    # Matches dates, relative time references, durations, and time-of-day patterns.
    _TEMPORAL_ANCHOR_RE = _re.compile(
        r'\b(?:'
        r'\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*'
        r'|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2}(?:st|nd|rd|th)?'
        r'|\d{4}[-/]\d{2}[-/]\d{2}'
        r'|(?:last|this|next)\s+(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun|week|month|year)\w*'
        r'|yesterday|today|tomorrow'
        r'|(?:about|around|roughly|approximately)\s+\d+\s+(?:days?|weeks?|months?|years?)\s+ago'
        r'|\d+\s+(?:days?|weeks?|months?|years?)\s+ago'
        r'|(?:in|on|since|before|after)\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*(?:\s+\d{4})?'
        r'|\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?'
        r')\b',
        _re.IGNORECASE
    )

    @staticmethod
    def _enrich_temporal_anchors(summary: str, source_sentence: str) -> str:
        """INGEST-022: Append temporal markers from source that are missing from summary."""
        markers = SessionLearnMixin._TEMPORAL_ANCHOR_RE.findall(source_sentence)
        if not markers:
            return summary
        # Check which markers are already in the summary
        summary_lower = summary.lower()
        missing = [m.strip() for m in markers if m.strip().lower() not in summary_lower]
        if not missing:
            return summary
        # Append temporal tag (cap at 2 markers to avoid bloat)
        tag = ", ".join(missing[:2])
        return f"{summary} [temporal: {tag}]"

    def _extract_insights(
        self,
        processor: "ConversationProcessor",
        text: str,
        assistant_text: str | None = None,
    ) -> list[dict]:
        """L2: Extract insights using extended heuristic patterns.

        Returns list of dicts: {summary, confidence, type, signals}.
        Uses ConversationProcessor as base, then applies additional
        conversation-specific patterns for richer extraction.

        INGEST-020: assistant_text enables role-aware factual extraction —
        declarative facts stated by the assistant (thresholds, defaults,
        return values) are scanned separately from combined user+assistant text.
        """
        insights = []

        # Pattern set for conversation-specific insight extraction
        # These catch common knowledge-sharing patterns in AI conversations
        # Types mapped to valid CONCEPT_TYPES (Knowledge Hierarchy L1-L6)
        conversation_patterns = [
            # Decisions and choices made → L2: decision
            {
                "markers": ["decided to", "decision is", "we chose", "going with", "opted for"],
                "type": "decision",
                "base_confidence": 0.55,
            },
            # Technical discoveries or findings → L1: observation
            {
                "markers": ["found that", "discovered that", "turns out", "realized that", "root cause"],
                "type": "observation",
                "base_confidence": 0.50,
            },
            # Architecture and design patterns → L1: pattern
            {
                "markers": ["architecture", "design pattern", "schema", "data model", "pipeline"],
                "type": "pattern",
                "base_confidence": 0.45,
            },
            # Process or methodology insights → L4: method
            {
                "markers": ["workflow", "best practice", "methodology", "approach", "strategy"],
                "type": "method",
                "base_confidence": 0.45,
            },
            # Performance insights → L1: observation
            {
                "markers": ["performance", "latency", "benchmark", "optimization", "bottleneck"],
                "type": "observation",
                "base_confidence": 0.50,
            },
            # Problem-solution pairs → L2: decision
            {
                "markers": ["the fix", "solution is", "solved by", "resolved by", "workaround"],
                "type": "decision",
                "base_confidence": 0.55,
            },
            # Requirements and constraints → L2: constraint
            {
                "markers": ["requirement", "constraint", "must be", "non-negotiable", "critical that"],
                "type": "constraint",
                "base_confidence": 0.50,
            },
            # Tradeoffs, comparisons → L1: observation
            {
                "markers": [
                    "tradeoff",
                    "trade-off",
                    "versus",
                    " vs ",
                    "compared to",
                    "distinction between",
                    "difference between",
                    "advantage",
                    "disadvantage",
                ],
                "type": "observation",
                "base_confidence": 0.45,
            },
            # Key insights and explanations → L1: observation
            {
                "markers": [
                    "the key ",
                    "the core ",
                    "the main ",
                    "essential ",
                    "the important ",
                    "the critical ",
                    "the fundamental ",
                ],
                "type": "observation",
                "base_confidence": 0.45,
            },
            # Causal reasoning → L1: pattern (recurring causal pattern)
            {
                "markers": ["because ", "the reason ", "this causes", "leads to ", "results in ", "due to "],
                "type": "pattern",
                "base_confidence": 0.45,
            },
            # Recommendations and guidance → L5: heuristic
            {
                "markers": ["recommend", "should use", "you should", "better to ", "prefer ", "avoid ", "don't use"],
                "type": "heuristic",
                "base_confidence": 0.50,
            },
            # User-stated preferences → L1.5: preference
            {
                "markers": ["i prefer ", "i like to ", "i don't like", "i always want", "i never want", "my preference", "i'd rather "],
                "type": "preference",
                "base_confidence": 0.50,
            },
            # INGEST-015: Personal identity facts → observation (is_factual, identity)
            {
                "markers": ["my name is", "i'm called ", "i am called "],
                "type": "observation",
                "base_confidence": 0.65,
                "is_factual": True,
                "temporal_category": "identity",
            },
            # EXTRACT-C2: Demographic age/birthday facts → observation (is_factual, identity)
            {
                "markers": ["i'm ", "i am ", "years old", "just turned ", "my birthday",
                            "born in ", "birth year", "in my 20s", "in my 30s",
                            "in my 40s", "in my 50s", "in my 60s"],
                "type": "observation",
                "base_confidence": 0.65,
                "is_factual": True,
                "temporal_category": "identity",
            },
            # INGEST-015: Role/employment facts → observation (is_factual, role)
            {
                "markers": ["i work at ", "i work for ", "i work as ", "my job is", "my role is", "my title is", "i'm employed at"],
                "type": "observation",
                "base_confidence": 0.65,
                "is_factual": True,
                "temporal_category": "role",
            },
            # INGEST-015: Relational facts → observation (is_factual, relational)
            {
                "markers": ["my partner ", "my wife ", "my husband ", "my girlfriend ", "my boyfriend ", "my manager ", "my boss ", "my colleague ", "my cofounder "],
                "type": "observation",
                "base_confidence": 0.60,
                "is_factual": True,
                "temporal_category": "relational",
            },
            # INGEST-022: Temporal activity facts → observation (is_factual, activity)
            # Catches statements with temporal anchors that no other pattern would match.
            {
                "markers": [" ago ", "last week", "last month", "last year",
                            "yesterday ", "this morning", "this afternoon",
                            "this evening", "last night"],
                "type": "observation",
                "base_confidence": 0.50,
                "is_factual": True,
                "temporal_category": "activity",
            },
            # PRODUCT-002: Quantitative episodic facts → observation (is_factual, episodic)
            # Catches "caught 10 bass", "spent $500", "finished 3 projects"
            {
                "markers": ["caught ", "spent ", "earned ", "saved ", "lost ",
                            "bought ", "sold ", "paid ", "scored ", "gained ",
                            "completed ", "finished ", "visited ", "traveled ",
                            "drove ", "walked ", "ran ", "swam "],
                "type": "observation",
                "base_confidence": 0.55,
                "is_factual": True,
                "temporal_category": "episodic",
            },
            # PRODUCT-002: Possessive count facts → observation (is_factual, count)
            # Catches "I have 3 dogs", "we own 2 cars"
            {
                "markers": ["i have ", "i own ", "we have ", "we own ",
                            "i've got ", "we've got "],
                "type": "observation",
                "base_confidence": 0.55,
                "is_factual": True,
                "temporal_category": "count",
            },
            # EXTRACT-C5: Possession/acquisition facts → observation (is_factual, episodic)
            # Catches "bought their EP", "got my vinyl signed", "picked up a guitar"
            {
                "markers": ["i bought ", "i purchased ", "i picked up ",
                            "i ordered ", "i downloaded ", "i subscribed ",
                            "got my ", "got a ", "got the "],
                "type": "observation",
                "base_confidence": 0.55,
                "is_factual": True,
                "temporal_category": "episodic",
            },
        ]

        text_lower = text.lower()
        sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]

        # Scan sentences for pattern matches
        for sentence in sentences:
            if len(sentence) < 30:
                continue
            sent_lower = sentence.lower()

            for pattern in conversation_patterns:
                matches = sum(1 for m in pattern["markers"] if m in sent_lower)
                if matches >= 1:
                    # Build summary: use the matching sentence, cap at 300 chars
                    summary = sentence[:300].strip()
                    if len(sentence) > 300:
                        summary += "..."

                    confidence = min(0.70, pattern["base_confidence"] + (matches - 1) * 0.10)

                    insight_dict = {
                        "summary": summary,
                        "confidence": confidence,
                        "type": pattern["type"],
                        "signals": [pattern["type"]],
                    }
                    # INGEST-015: Forward factual metadata from pattern definition
                    if pattern.get("is_factual"):
                        insight_dict["is_factual"] = True
                        insight_dict["temporal_category"] = pattern.get("temporal_category")
                    # INGEST-022: Enrich summary with temporal anchors from source
                    insight_dict["summary"] = self._enrich_temporal_anchors(
                        insight_dict["summary"], sentence
                    )
                    insights.append(insight_dict)
                    break  # One pattern per sentence

        # Also try ConversationProcessor heuristic for additional extraction
        heuristic = processor.extract_insight_heuristic(text)
        if heuristic and heuristic.get("summary"):
            # Check it's not a near-duplicate of already extracted insights
            is_dup = any(self._text_overlap(heuristic["summary"], i["summary"]) > 0.7 for i in insights)
            if not is_dup:
                insights.append(heuristic)

        # INGEST-020: Role-aware factual scan — assistant_text only.
        # Declarative facts from the assistant (thresholds, defaults, config values)
        # are more reliably extracted from the assistant turn alone, without
        # user-turn noise. Capped at 3 to avoid flooding with low-confidence observations.
        if assistant_text and len(assistant_text) >= 30:
            FACTUAL_MARKERS = [
                "the default is", "the value is", "the threshold is",
                "is set to", "is stored in", "is defined in",
                "must be ", "cannot be ", "requires ",
                "returns ", "raises ", "expects ",
            ]
            # INGEST-021: Split on newlines first (captures list items, bullet points),
            # then split each line on periods (captures multi-sentence lines).
            # Strip list prefixes BEFORE splitting so bullets/numbers don't leak through.
            _LIST_PREFIX_RE = _re.compile(r'^\s*(?:[-*•]|\d+[.)]) *')
            raw_lines = assistant_text.replace("!", ".").replace("?", ".").split("\n")
            asst_sentences = []
            for line in raw_lines:
                # Strip list prefix first (bullets, numbers)
                clean_line = _LIST_PREFIX_RE.sub('', line).strip()
                if not clean_line or len(clean_line) < 30:
                    continue
                # Split on periods for multi-sentence lines
                if "." in clean_line:
                    parts = [p.strip() for p in clean_line.split(".") if p.strip() and len(p.strip()) >= 30]
                    if parts:
                        asst_sentences.extend(parts)
                    else:
                        # Periods present but splits too short — use whole line
                        asst_sentences.append(clean_line)
                else:
                    # No periods — use whole cleaned line
                    asst_sentences.append(clean_line)
            factual_count = 0
            for sentence in asst_sentences:
                if factual_count >= 3:
                    break
                sent_lower = sentence.lower()
                if any(m in sent_lower for m in FACTUAL_MARKERS):
                    is_dup = any(self._text_overlap(sentence, i["summary"]) > 0.6 for i in insights)
                    if not is_dup:
                        # INGEST-022: Enrich factual scan with temporal anchors
                        enriched_summary = self._enrich_temporal_anchors(
                            sentence[:300], sentence
                        )
                        insights.append({
                            "summary": enriched_summary,
                            "confidence": 0.55,  # DATA-068: raised from 0.45 to survive conservation zone (floor=0.50)
                            "type": "observation",
                            "signals": ["factual_marker"],
                            "extraction_source": "factual_scan",
                        })
                        factual_count += 1

        # Cap at 7 insights per call (INGEST-020: raised from 5)
        return insights[:7]

    def _text_overlap(self, a: str, b: str) -> float:
        """Simple word overlap ratio between two strings."""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        return len(intersection) / min(len(words_a), len(words_b))

    # --- S3.5: Contradiction detection ---
    # LIFECYCLE-001: STATUS_TRANSITIONS moved to app/constants.py for shared access
    # Used by write-time detection (here). Serve-time Phase 1b removed per LIFECYCLE-001 cleanup.
    # SUPER-009: Self-correction detection handled by Layer 1.5 in _detect_contradiction
    # (NOT as a STATUS_TRANSITIONS entry — "never"/"actually" are too common as standalone
    # markers, causing false positives on unrelated sentences. Layer 1.5 has 2-word topic
    # overlap guard that prevents cross-topic matches.)

    def _detect_contradiction(self, existing_summary: str, new_summary: str, new_ka: str = "") -> str | None:
        """S3.5: Detect if new insight contradicts/supersedes an existing concept.

        Uses STATUS_TRANSITIONS marker pairs + opposing assertion detection.
        Returns reason string if contradiction detected, None otherwise.

        Bug 4 fix: Also detects opposing claims where two concepts discuss the
        same topic but assert mutually exclusive specifics (e.g., "uses Python"
        vs "uses Go"). Uses the contradiction engine's negation detection.
        """
        old_lower = existing_summary.lower()
        new_lower = new_summary.lower()

        # MEASURE-018: Layer-level instrumentation for contradiction detection analysis
        _layers_checked: list[str] = []

        # Layer 1: STATUS_TRANSITIONS marker pairs (original behavior)
        # LIFECYCLE-001: Imported from constants.py (lazy import to avoid circular deps)
        from app.core.constants import STATUS_TRANSITIONS

        _layers_checked.append("L1_status_transitions")
        for before_markers, after_markers, reason in STATUS_TRANSITIONS:
            old_matches = any(m in old_lower for m in before_markers)
            new_matches = any(m in new_lower for m in after_markers)
            if old_matches and new_matches:
                logger.debug(f"MEASURE-018: Contradiction detected at L1 — {reason}")
                return reason

        # Layer 1.5 (SELFCORR-001): Detect same-entity opposing conclusions
        # Catches: "X never fired" vs "X fired 110 times" where STATUS_TRANSITIONS
        # markers don't match because summaries use analytical language.
        # PHASE1-FP-001: Tightened negation markers — removed "no " and "not " which cause
        # false positives on decision concepts ("not a bug", "no longer needed", etc.)
        negation_markers = {"never ", "zero ", "none of", "0 ", "doesn't ", "cannot ", "impossible"}
        affirmation_markers = {"actually", "confirmed", "verified", "found that", "turns out", "in fact", "does exist"}

        old_has_negation = any(m in old_lower for m in negation_markers)
        new_has_affirmation = any(m in new_lower for m in affirmation_markers)

        _layers_checked.append("L1.5_self_correction")
        if old_has_negation and new_has_affirmation:
            old_words = set(_re.findall(r"\b\w{4,}\b", old_lower))
            new_words = set(_re.findall(r"\b\w{4,}\b", new_lower))
            overlap = old_words & new_words
            if len(overlap) >= 2:
                logger.debug(f"MEASURE-018: Contradiction detected at L1.5 — self-correction")
                return f"Self-correction: negation in old + affirmation in new (shared terms: {', '.join(list(overlap)[:5])})"

        # Layer 1.7 (KU-VALUE-CONFLICT-001): Same-entity value-conflict detection.
        # Catches: "user has 3 bikes" -> "user now has 4 bikes" where the entity
        # and attribute are the same but the value changed.
        # Zero LLM calls — pure regex + string comparison. <1ms.
        _layers_checked.append("L1.7_value_conflict")
        try:
            from app.cognitive.value_conflict import detect_value_conflict
            vc_result = detect_value_conflict(old_lower, new_lower)
            if vc_result:
                logger.debug(f"MEASURE-018: Contradiction detected at L1.7 — {vc_result}")
                return f"Value conflict: {vc_result}"
        except Exception as _vc_err:
            logger.debug(f"L1.7: Value conflict check failed (non-fatal): {_vc_err}")

        # Layer 1.8 (CONTRA-018): Embedding pre-check at ingestion time.
        # Replaces Phase 1 keyword negation with Phase 2 semantic detection.
        # Searches embedding index for similar concepts, then runs
        # _has_directional_opposition() — the same logic Phase 2 uses at retrieval time.
        # MEASURE-018: Full metrics for effectiveness measurement.
        _layers_checked.append("L1.8_embedding_pre_check")
        _l18_t0 = time.perf_counter() if 'time' in dir() else __import__('time').perf_counter()
        _l18_candidates_checked = 0
        _l18_filtered_cosine = 0
        _l18_filtered_ka = 0
        _l18_filtered_overlap = 0
        try:
            import time as _l18_time
            _l18_t0 = _l18_time.perf_counter()

            from app.retrieval import retrieval_engine
            from app.cognitive.contradiction import (
                _has_directional_opposition,
                EMBEDDING_SAME_TOPIC_THRESHOLD,
                CROSS_TOPIC_OVERLAP_MIN,
                _compute_keyword_overlap_score,
            )

            candidates = retrieval_engine.search_for_dedup_embedding(
                query_text=new_summary,
                top_k=5,
            )

            for candidate in candidates:
                _l18_candidates_checked += 1

                # Filter 1: Cosine similarity threshold
                if candidate["cosine_score"] < EMBEDDING_SAME_TOPIC_THRESHOLD:
                    _l18_filtered_cosine += 1
                    continue

                # Filter 2: Same knowledge area
                if new_ka and candidate["knowledge_area"] != new_ka:
                    _l18_filtered_ka += 1
                    continue

                # Filter 3: Keyword overlap guard (prevents cross-topic false positives)
                cand_summary = candidate.get("summary", "")
                if not cand_summary:
                    continue
                overlap = _compute_keyword_overlap_score(new_summary, cand_summary)
                if overlap < CROSS_TOPIC_OVERLAP_MIN:
                    _l18_filtered_overlap += 1
                    continue

                # Core check: directional opposition (same logic as Phase 2)
                is_opposition, signal = _has_directional_opposition(new_summary, cand_summary)
                if is_opposition:
                    _l18_elapsed_ms = (_l18_time.perf_counter() - _l18_t0) * 1000
                    logger.debug(
                        f"MEASURE-018: Contradiction detected at L1.8 — "
                        f"sim={candidate['cosine_score']:.3f}, signal={signal}, "
                        f"candidate={candidate['concept_id']}, "
                        f"checked={_l18_candidates_checked}, elapsed_ms={_l18_elapsed_ms:.1f}"
                    )
                    # CONTRA-018-METRICS: Log detection event for effectiveness tracking
                    try:
                        from app.storage import db_immediate
                        with db_immediate() as _l18_conn:
                            _l18_conn.execute(
                                """INSERT INTO governance_events
                                   (event_type, concept_id, details, created_at)
                                   VALUES (?, ?, ?, datetime('now'))""",
                                (
                                    "contra_018_l18_detection",
                                    candidate["concept_id"],
                                    f"cosine={candidate['cosine_score']:.3f}|signal={signal}|overlap={overlap:.3f}|"
                                    f"candidates={_l18_candidates_checked}|elapsed_ms={_l18_elapsed_ms:.1f}|"
                                    f"filtered_cosine={_l18_filtered_cosine}|filtered_ka={_l18_filtered_ka}|"
                                    f"filtered_overlap={_l18_filtered_overlap}",
                                ),
                            )
                    except Exception:
                        pass  # Non-fatal — metrics logging must not block ingestion

                    return f"Embedding pre-check: cosine={candidate['cosine_score']:.3f}, opposition={signal}"

            # CONTRA-018-METRICS: Log L1.8 miss for false-negative analysis
            _l18_elapsed_ms = (_l18_time.perf_counter() - _l18_t0) * 1000
            logger.debug(
                f"CONTRA-018: L1.8 no contradiction found — "
                f"checked={_l18_candidates_checked}, elapsed_ms={_l18_elapsed_ms:.1f}, "
                f"filtered: cosine={_l18_filtered_cosine}, ka={_l18_filtered_ka}, overlap={_l18_filtered_overlap}"
            )
        except Exception as _l18_err:
            logger.debug(f"L1.8: Embedding pre-check failed (non-fatal): {_l18_err}")

        # Layer 2 (Bug 4 fix): Opposing assertion detection via contradiction engine
        # Only runs if summaries have meaningful overlap (same topic area) but also
        # have clear differences (not just paraphrases)
        _layers_checked.append("L2_opposing_assertions")
        try:
            from app.cognitive.contradiction import ScoredConcept, detect_retrieval_contradictions

            existing_sc = ScoredConcept(
                concept_id="__existing__",
                summary=existing_summary,
                knowledge_area="dedup_check",
                authority_score=0.5,
                currency_score=1.0,
            )
            new_sc = ScoredConcept(
                concept_id="__new__",
                summary=new_summary,
                knowledge_area="dedup_check",
                authority_score=0.5,
                currency_score=1.0,
            )
            result = detect_retrieval_contradictions([existing_sc, new_sc])
            if result.pairs:
                pair = result.pairs[0]
                logger.debug(f"MEASURE-018: Contradiction detected at L2 — {pair.contradiction_type.value}")
                return f"Opposing assertions detected: {pair.contradiction_type.value} ({pair.reason or 'semantic conflict'})"
        except Exception as e:
            logger.debug(f"S3.5: Opposing assertion check failed (non-fatal): {e}")

        # Layer 3 (Phase 3 v1.1): LLM Tier 2 for ambiguous cases
        _layers_checked.append("L3_llm_tier2")
        try:
            from app.core.config import get_feature_flag

            if get_feature_flag("LLM_CONTRADICTION_TIER2_ENABLED"):
                from app.cognitive.contradiction import _compute_keyword_overlap_score
                from app.cognitive.contradiction_llm import detect_contradiction_llm_sync, is_tier2_candidate

                tier1_score = _compute_keyword_overlap_score(existing_summary, new_summary)
                if is_tier2_candidate(tier1_score):
                    llm_result = detect_contradiction_llm_sync(
                        new_summary,
                        existing_summary,
                        session_id=getattr(self, "_current_session_id", ""),
                    )
                    if llm_result.score > 0.7:
                        logger.debug(f"MEASURE-018: Contradiction detected at L3 — LLM Tier 2")
                        return f"LLM Tier 2: {llm_result.reason[:200]}"
        except Exception as e:
            logger.debug(f"S3.5: LLM Tier 2 check failed (non-fatal): {e}")

        # MEASURE-018: No contradiction detected across any layer
        logger.debug(f"MEASURE-018: No contradiction detected (layers checked: {_layers_checked})")
        return None

    def _supersede_concept(self, old_concept_id: str, new_concept_id: str, reason: str) -> bool:
        """S3.5: Execute supersession via unified path (SUPER-012).

        Delegates to execute_supersession() which handles ALL supersession behaviors:
        currency_status, superseded_by, confidence reduction, anti-terms, edge creation,
        association transfer, evidence addition, governance event.

        Replaces the previous independent implementation that was inconsistent with
        the supersession.py path (see SUPERSESSION_COMPOUND_GAUNTLET.md, finding G-01).
        """
        try:
            from app.storage import db_immediate
            from app.cognitive.supersession import execute_supersession

            with db_immediate() as conn:
                result = execute_supersession(
                    old_concept_id=old_concept_id,
                    new_concept_id=new_concept_id,
                    reason=reason,
                    conn=conn,
                )
            return result.superseded

        except Exception as e:
            logger.error(
                "SUPER-012: Unified supersession failed %s→%s: %s",
                old_concept_id,
                new_concept_id,
                e,
            )
            return False

    def _process_single_insight(
        self,
        insight: dict,
        request: SessionLearnRequest,
        retrieval_engine,
        extraction_source: str = "heuristic",
        evolved_this_call: set = None,
        budget_remaining: int = 50,
        precomputed_dedup: list[dict] | None = None,  # PERF-021: batch pre-computed dedup results
    ) -> dict:
        """Process a single extracted insight through dedup, creation, and association.

        Returns dict with action taken and result details.
        Implements S1 (source tagging), S2 (self-corroboration), S3 (per-call cap),
        S5 (HHI confidence cap).
        """
        if evolved_this_call is None:
            evolved_this_call = set()

        summary = insight["summary"]
        confidence = insight.get("confidence", 0.40)

        # --- Quality gate: confidence floor (PRICING-006: budget-aware) ---
        _budget_zone_value = "unknown"
        try:
            from app.core.config import BUDGET_ZONE_THRESHOLDS
            from app.api.pricing import conversation_meter

            budget_zone = conversation_meter.get_budget_zone()
            _budget_zone_value = budget_zone.value
            zone_thresholds = BUDGET_ZONE_THRESHOLDS.get(
                budget_zone.value,
                BUDGET_ZONE_THRESHOLDS["normal"],  # Fallback to default
            )
            client_floor = zone_thresholds["client"]
            heuristic_floor = zone_thresholds["heuristic"]
        except Exception:
            # Fallback to Sprint B defaults if pricing module unavailable
            client_floor = 0.35
            heuristic_floor = 0.45

        # INGEST-001 + PRICING-006: Budget-aware confidence floors
        if extraction_source == "heuristic" and confidence < heuristic_floor:
            return {"action": "skipped_confidence_heuristic", "budget_zone": _budget_zone_value}
        elif confidence < client_floor:
            return {"action": "skipped_confidence", "budget_zone": _budget_zone_value}

        # INGEST-001: Minimum summary quality
        # MEASURE-026 §18: Benchmark facts are complete declarative sentences (5-7 words)
        # that are valid despite being short. Client-extracted concepts with evidence
        # use a lower floor (4 words) since they've already been validated upstream.
        summary_words = len(summary.split())
        _min_words = 4 if (extraction_source == "client" and insight.get("evidence")) else 8
        if summary_words < _min_words:
            return {"action": "skipped_short_summary"}

        # INGEST-001: Evidence requirement for client-extracted concepts
        evidence = insight.get("evidence", [])
        if extraction_source == "client" and not evidence:
            return {"action": "skipped_no_evidence"}

        # --- Deduplication via cosine similarity ---
        # MATURITY-003: Use embedding search when available (handles paraphrases).
        # TF-IDF dead zone: 99.1% of scores fall below 0.50, preventing evolution.
        # Embedding search produces continuous distribution in the EVOLVE zone.
        from app.core.config import (
            EMBEDDING_EVOLVE_THRESHOLD,
            EMBEDDING_SKIP_THRESHOLD,
            FEATURE_FLAGS,
        )

        _use_embedding = FEATURE_FLAGS.get("EMBEDDING_DEDUP_ENABLED", False)
        if _use_embedding:
            dedup_results = retrieval_engine.search_for_dedup_embedding(summary, top_k=3)
            _skip_threshold = EMBEDDING_SKIP_THRESHOLD  # 0.85 (calibrated)
            _evolve_threshold = EMBEDDING_EVOLVE_THRESHOLD  # 0.55 (calibrated)
        elif precomputed_dedup is not None:
            # PERF-021: use batch pre-computed results (avoids redundant DB round-trip)
            dedup_results = precomputed_dedup
            _skip_threshold = float(os.environ.get("PITH_TFIDF_SKIP_THRESHOLD", "0.85"))
            _evolve_threshold = float(os.environ.get("PITH_TFIDF_EVOLVE_THRESHOLD", "0.50"))
        else:
            dedup_results = retrieval_engine.search_for_dedup_tfidf(summary, top_k=3)
            _skip_threshold = float(os.environ.get("PITH_TFIDF_SKIP_THRESHOLD", "0.85"))
            _evolve_threshold = float(os.environ.get("PITH_TFIDF_EVOLVE_THRESHOLD", "0.50"))

        top_cosine = dedup_results[0]["cosine_score"] if dedup_results else 0.0
        top_match = dedup_results[0] if dedup_results else None

        # RETRIEVAL-021: Activation bias — lower evolve threshold for activated concepts
        from app.core.config import ACTIVATION_EVOLVE_BIAS
        _activated_ids = getattr(request, "activated_concept_ids", None) or []
        _activation_bias = 0.0
        if top_match and top_match.get("concept_id") in _activated_ids:
            _activation_bias = min(ACTIVATION_EVOLVE_BIAS, 0.20)  # Clamp to prevent aggressive merging
            logger.info(
                f"RETRIEVAL-021: activation bias for {top_match['concept_id']} "
                f"(cosine={top_cosine:.3f}, effective_threshold={_evolve_threshold - _activation_bias:.2f})"
            )
        _effective_evolve_threshold = _evolve_threshold - _activation_bias

        # INGEST-007: Cross-KA merge guard
        from app.core.config import CROSS_KA_EVOLVE_THRESHOLD, ka_groups_match
        _incoming_ka = insight.get("knowledge_area", "general") or "general"
        _match_ka = top_match.get("knowledge_area", "") if top_match else ""
        _ka_match = ka_groups_match(_incoming_ka, _match_ka)
        if not _ka_match:
            _effective_evolve_threshold = max(_effective_evolve_threshold, CROSS_KA_EVOLVE_THRESHOLD)

        # BENCHMARK-001: In benchmark mode, force all concepts to CREATE zone —
        # bypass dedup entirely. FactConsolidation facts share template structure
        # ("X is famous for Y") so TF-IDF similarity can exceed 0.85 skip threshold
        # even when subject/object differ. Pre-deduplicated by pith_agent conflict
        # resolution (highest serial wins), so dedup here is harmful not helpful.
        # BENCH-INFRA-002: Dedup bypass via BenchmarkIngestionMode config.
        # BENCHMARK-001b: Separate control for dedup bypass vs GarbageDetector bypass.
        from app.core.config import BENCHMARK as _bm_dedup
        _dedup_bypass = _bm_dedup.skip_dedup

        # DATA-055: Classify dedup zone for structured logging
        if _dedup_bypass:
            _dedup_zone = "CREATE"
        elif top_cosine >= _skip_threshold:
            _dedup_zone = "SKIP"
        elif top_cosine >= _effective_evolve_threshold and top_match:
            _dedup_zone = "EVOLVE"
        else:
            _dedup_zone = "CREATE"

        # INGEST-007: Log KA guard decision
        if not _ka_match and top_cosine >= EMBEDDING_EVOLVE_THRESHOLD:
            logger.info(
                f"INGEST-007: Cross-KA guard activated — "
                f"incoming={_incoming_ka} match={_match_ka} "
                f"cosine={top_cosine:.4f} effective_thresh={_effective_evolve_threshold:.2f} "
                f"zone={_dedup_zone}"
            )
            # MONITOR-056: Track cross-KA guard activation rate
            from app.ops.metrics import metrics as _ka_metrics
            _ka_metrics.record("cross_ka_guard_activations", 1)

        # DATA-055: Structured dedup outcome log
        _dedup_method = "embedding" if _use_embedding else "tfidf"
        _match_id = top_match["concept_id"] if top_match else None
        logger.info(
            f"DEDUP_DECISION: zone={_dedup_zone} cosine={top_cosine:.4f} "
            f"match={_match_id} method={_dedup_method} "
            f"skip_thresh={_skip_threshold} evolve_thresh={_effective_evolve_threshold:.2f} "
            f"activation_bias={_activation_bias:.2f} "
            f"same_call_dups={len(evolved_this_call) if evolved_this_call else 0} "
            f"summary_hash={hashlib.sha256(summary.encode()).hexdigest()[:12]}"
        )

        # MEASURE-026 §15: SKIP-zone divergence detection.
        # When cosine >= skip_threshold, check if incoming and existing concepts
        # share most tokens (same subject) but differ on content words (different value).
        # If so, override SKIP→EVOLVE to handle temporal supersession at the SKIP boundary.
        if _dedup_zone == "SKIP" and top_match and _use_embedding:
            try:
                _div_existing = load_concept(top_match["concept_id"], track_access=False)
                if _div_existing and _div_existing.summary:
                    _div_incoming_tok = set(summary.lower().split())
                    _div_existing_tok = set(_div_existing.summary.lower().split())
                    _div_intersection = _div_incoming_tok & _div_existing_tok
                    _div_union = _div_incoming_tok | _div_existing_tok
                    _div_jaccard = len(_div_intersection) / len(_div_union) if _div_union else 1.0
                    _div_stopwords = {
                        'the', 'a', 'an', 'is', 'was', 'are', 'were', 'be', 'been',
                        'of', 'in', 'to', 'for', 'by', 'and', 'or', 'with', 'that',
                        'this', 'it', 'its', 'has', 'had', 'have', 'not', 'on', 'at',
                    }
                    _div_only_incoming = _div_incoming_tok - _div_existing_tok - _div_stopwords
                    _div_only_existing = _div_existing_tok - _div_incoming_tok - _div_stopwords
                    _div_content = _div_only_incoming | _div_only_existing
                    _div_threshold = float(os.environ.get("PITH_SKIP_DIVERGENCE_THRESHOLD", "0.6"))
                    if _div_jaccard >= _div_threshold and len(_div_content) >= 1:
                        _dedup_zone = "EVOLVE"
                        logger.info(
                            f"SKIP-OVERRIDE: divergence detected — cosine={top_cosine:.4f} "
                            f"jaccard={_div_jaccard:.3f} divergent_tokens={_div_content} "
                            f"reclassified SKIP→EVOLVE for match={top_match['concept_id']}"
                        )
            except Exception as _div_err:
                logger.warning(f"SKIP-OVERRIDE: divergence check failed: {_div_err}")

        # MEASURE-026 §18: Entity-aware dedup guard.
        # Template-similar facts about DIFFERENT subjects (e.g. "X is married to A"
        # vs "Y is married to B") produce high cosine due to shared template tokens.
        # Extract leading subject entity from both concepts; if subjects differ,
        # override EVOLVE/SKIP → CREATE to prevent cross-entity merging.
        if _dedup_zone in ("EVOLVE", "SKIP") and top_match:
            try:
                _eg_existing = (
                    _div_existing if '_div_existing' in dir() and _div_existing
                    else load_concept(top_match["concept_id"], track_access=False)
                )
                if _eg_existing and _eg_existing.summary:
                    def _extract_subject(text: str) -> str:
                        """Extract leading proper-noun span as subject entity.
                        Returns lowercased subject for comparison.
                        Handles: 'Thomas Kyd was born in...' → 'thomas kyd'
                                 'The capital of France is...' → 'the capital of france'
                                 'Windows Phone was developed by...' → 'windows phone'
                        """
                        _copulas = {
                            'is', 'was', 'are', 'were', 'has', 'had',
                            'plays', 'speaks', 'works', 'worked', 'died',
                        }
                        words = text.split()
                        subject_words = []
                        for w in words:
                            # Stop at copula/verb boundary
                            if w.lower().rstrip('.,;:') in _copulas:
                                break
                            subject_words.append(w)
                        # Fallback: if no copula found, take first 3 words
                        if not subject_words:
                            subject_words = words[:3]
                        return " ".join(subject_words).lower().strip('.,;:')

                    _eg_incoming_subj = _extract_subject(summary)
                    _eg_existing_subj = _extract_subject(_eg_existing.summary)

                    if _eg_incoming_subj and _eg_existing_subj and _eg_incoming_subj != _eg_existing_subj:
                        _eg_original_zone = _dedup_zone
                        _dedup_zone = "CREATE"
                        logger.info(
                            f"ENTITY-GUARD: subject mismatch — "
                            f"incoming='{_eg_incoming_subj}' existing='{_eg_existing_subj}' "
                            f"cosine={top_cosine:.4f} overriding {_eg_original_zone}→CREATE "
                            f"for match={top_match['concept_id']}"
                        )
            except Exception as _eg_err:
                logger.warning(f"ENTITY-GUARD: check failed: {_eg_err}")

        # PRODUCT-002: Factual-into-abstract guard.
        # Episodic facts (numbers, dates, amounts) must not be absorbed into
        # pattern/principle/heuristic concepts. Force CREATE to preserve specifics.
        if _dedup_zone == "EVOLVE" and top_match:
            try:
                from app.core.config import get_feature_flag
                if get_feature_flag("EPISODIC_GRANULARITY_GUARD_ENABLED", False):
                    from app.cognitive.fact_classifier import classify_concept as _fc_classify
                    _incoming_cls = _fc_classify(
                        summary=summary,
                        concept_type=insight.get("type", "observation"),
                        knowledge_area=insight.get("knowledge_area", "general") or "general",
                    )
                    _incoming_factual = _incoming_cls.get("is_factual", False)
                    _incoming_score = _incoming_cls.get("factual_score", 0)

                    if _incoming_factual and _incoming_score >= 2.0:
                        _eg_for_p002 = (
                            _div_existing if '_div_existing' in dir() and _div_existing
                            else load_concept(top_match["concept_id"], track_access=False)
                        )
                        if _eg_for_p002:
                            _existing_type = _eg_for_p002.concept_type or "observation"
                            _ABSTRACT_TYPES = {"pattern", "principle", "method",
                                               "heuristic", "cognitive_strategy"}
                            if _existing_type in _ABSTRACT_TYPES:
                                _dedup_zone = "CREATE"
                                logger.info(
                                    f"PRODUCT-002: Factual-into-abstract guard — "
                                    f"incoming factual (score={_incoming_score:.1f}) "
                                    f"vs existing {_existing_type} "
                                    f"cosine={top_cosine:.4f} overriding EVOLVE→CREATE "
                                    f"for match={top_match['concept_id']}"
                                )
                                try:
                                    from app.ops.metrics import metrics as _p002_metrics
                                    _p002_metrics.record("product002_guard_activations", 1)
                                except Exception:
                                    pass
            except Exception as _p002_err:
                logger.debug(f"PRODUCT-002: Guard check failed (non-fatal): {_p002_err}")

        # --- STALE-003: Metric-conflict write-time check ---
        # If incoming concept contains a recognizable metric pattern (X%, N/M)
        # and the nearest match contains a DIFFERENT value for the same metric,
        # force EVOLVE to update rather than creating a contradictory duplicate.
        if _dedup_zone == "CREATE" and top_match and top_cosine >= 0.35:
            _stale_metric_override = os.environ.get("PITH_METRIC_CONFLICT_CHECK", "true").lower() == "true"
            if _stale_metric_override:
                import re as _mc_re
                _METRIC_RE = _mc_re.compile(
                    r'(\d+\.?\d*)\s*%'       # percentage: "73.2%"
                    r'|(\d+)\s*/\s*(\d+)'     # fraction: "60/71"
                    r'|(\d+\.?\d*)\s*pp'      # percentage points: "+4.2pp"
                )
                _incoming_metrics = set(_METRIC_RE.findall(summary))
                if _incoming_metrics and top_match:
                    _existing_summary = top_match.get("summary", "")
                    _existing_metrics = set(_METRIC_RE.findall(_existing_summary))
                    if _incoming_metrics and _existing_metrics and _incoming_metrics != _existing_metrics:
                        _dedup_zone = "EVOLVE"
                        logger.info(
                            f"STALE-003: metric-conflict override CREATE→EVOLVE — "
                            f"incoming_metrics={_incoming_metrics} "
                            f"existing_metrics={_existing_metrics} "
                            f"match={top_match.get('concept_id', 'unknown')}"
                        )

        # Three-zone dedup logic (thresholds adapt to search method)
        if _dedup_zone == "SKIP":
            return {"action": "skipped_duplicate", "dedup_zone": "SKIP",
                    "cosine": round(top_cosine, 4), "match_id": _match_id, "method": _dedup_method}

        if not _dedup_bypass and _dedup_zone == "EVOLVE" and _effective_evolve_threshold <= top_cosine < _skip_threshold and top_match:
            # S3.5: Contradiction detection — check if new insight supersedes old
            existing_concept = load_concept(top_match["concept_id"], track_access=False)
            if existing_concept:
                contradiction_reason = self._detect_contradiction(existing_concept.summary, summary, new_ka=_incoming_ka)
                # MEASURE-018: Instrument contradiction detection for false-negative analysis
                try:
                    import json as _m18_json
                    from datetime import datetime as _m18_dt

                    from app.storage import db_immediate

                    with db_immediate() as _m18_conn:
                        _m18_conn.execute(
                            """INSERT INTO governance_events
                               (session_id, event_type, concept_id, details, created_at)
                               VALUES (?, ?, ?, ?, ?)""",
                            (
                                self.current_session.session_id if self.current_session else None,
                                "contradiction_check",
                                existing_concept.id,
                                _m18_json.dumps({
                                    "detected": contradiction_reason is not None,
                                    "reason": contradiction_reason[:200] if contradiction_reason else None,
                                    "cosine": round(top_cosine, 4),
                                    "existing_summary": existing_concept.summary[:100],
                                    "new_summary": summary[:100],
                                    "dedup_zone": "EVOLVE",
                                }),
                                _m18_dt.now(UTC).isoformat(),
                            ),
                        )
                except Exception:
                    pass  # Instrumentation is best-effort
                if contradiction_reason:
                    # P3-2: Log contradiction detection governance event
                    try:
                        import json as _json
                        from datetime import datetime as _dt

                        from app.storage import db_immediate

                        with db_immediate() as _gov_conn:
                            _gov_conn.execute(
                                """INSERT INTO governance_events
                                   (event_type, concept_id, details, created_at)
                                   VALUES (?, ?, ?, ?)""",
                                (
                                    GOV_EVENT_CONTRADICTION_REVIEW,
                                    top_match["concept_id"],
                                    _json.dumps(
                                        {
                                            "new_insight_summary": summary[:200],
                                            "contradiction_reason": contradiction_reason,
                                            "session_id": getattr(self, "_current_session_id", ""),
                                        }
                                    ),
                                    _dt.now(UTC).isoformat(),
                                ),
                            )
                    except Exception:
                        pass

                    # This is a supersession, not an evolution — create new, mark old
                    logger.info(
                        f"S3.5: Contradiction detected: '{top_match['concept_id']}' "
                        f"→ new insight. Reason: {contradiction_reason}"
                    )
                    # Create the new concept (treat as novel)
                    # Bug 6 fix: skip_write_contradiction=True prevents double-catch —
                    # we already know there's a contradiction (that's why we're superseding),
                    # so the write-time check would either HARD_REJECT or error on the
                    # same conflict we already detected.
                    search_results = retrieval_engine.search_lightweight(
                        summary,
                        top_k=3,
                        min_confidence=0.0,
                    )
                    result = self._create_new_concept(
                        insight,
                        request,
                        retrieval_engine,
                        search_results,
                        extraction_source=extraction_source,
                        skip_write_contradiction=True,
                    )
                    # If creation succeeded, supersede the old concept
                    if result.get("action") == "created":
                        new_id = result["learned_concept"].concept_id
                        self._supersede_concept(top_match["concept_id"], new_id, contradiction_reason)
                        result["superseded"] = {
                            "old_id": top_match["concept_id"],
                            "new_id": new_id,
                            "reason": contradiction_reason,
                        }
                    return result

            # EXPLICIT_SUPERSESSION_SPEC v1.1 Amendment A1:
            # If LLM explicitly declared supersession of the dedup match,
            # override evolution — create new concept + supersede old.
            explicit_supersedes = insight.get("supersedes") or []
            if top_match["concept_id"] in explicit_supersedes:
                logger.info(
                    f"EXPLICIT_SUPERSESSION: overriding dedup evolution for "
                    f"'{top_match['concept_id']}' (cosine={top_cosine:.2f})"
                )
                search_results = retrieval_engine.search_lightweight(summary, top_k=3, min_confidence=0.0)
                result = self._create_new_concept(
                    insight,
                    request,
                    retrieval_engine,
                    search_results,
                    extraction_source=extraction_source,
                    skip_write_contradiction=True,
                )
                if result.get("action") == "created":
                    new_id = result["learned_concept"].concept_id
                    self._supersede_concept(
                        top_match["concept_id"], new_id, "Explicit supersession declared at extraction time"
                    )
                    result["superseded"] = {
                        "old_id": top_match["concept_id"],
                        "new_id": new_id,
                        "reason": "explicit_supersession_override_dedup",
                    }
                return result

            # S3: Per-call evidence cap — max 1 evolution per concept per call
            concept_id = top_match["concept_id"]

            # BENCH-EVOLVE-001: Gate evolve on PITH_DISABLE_EVOLVE.
            # When set (benchmark ingestion), skip evolve and fall through to
            # novel creation — lets CF facts create separate concepts instead of
            # being absorbed into their real-world predecessors.
            import os as _evo_os
            _evolve_disabled = _evo_os.environ.get("PITH_DISABLE_EVOLVE", "").lower() in ("true", "1")

            if _evolve_disabled:
                logger.info(
                    f"BENCH-EVOLVE-001: Evolve disabled (PITH_DISABLE_EVOLVE=true), "
                    f"skipping dedup evolve for cosine={top_cosine:.4f} match={concept_id}"
                )
                # Fall through past this block to S4 budget check → novel creation
            else:
                if concept_id in evolved_this_call:
                    logger.info(
                        f"DEDUP_DECISION: zone=EVOLVE_CAPPED cosine={top_cosine:.4f} "
                        f"match={concept_id} method={_dedup_method} reason=per_call_cap"
                    )
                    return {"action": "skipped_per_call_cap", "dedup_zone": "EVOLVE_CAPPED",
                            "cosine": round(top_cosine, 4), "match_id": concept_id, "method": _dedup_method}
                evolved_this_call.add(concept_id)
                _evolve_result = self._evolve_existing_from_dedup(top_match, insight, request, extraction_source=extraction_source)
                _evolve_result["dedup_zone"] = "EVOLVE"
                _evolve_result["cosine"] = round(top_cosine, 4)
                _evolve_result["match_id"] = concept_id
                _evolve_result["method"] = _dedup_method
                return _evolve_result

        # Novel: CREATE new concept (cosine < 0.50)
        search_results = retrieval_engine.search_lightweight(
            summary,
            top_k=3,
            min_confidence=0.0,
        )
        result = self._create_new_concept(
            insight, request, retrieval_engine, search_results, extraction_source=extraction_source
        )

        # --- Trigger 1: Staleness check on embedding neighbors ---
        # The dedup above used TF-IDF (cosine < 0.50), but embedding search
        # can find same-topic concepts that TF-IDF missed (empirically validated:
        # TF-IDF gives 0.04 where embedding gives 0.42 on status-transition pairs).
        if result.get("action") == "created" and result.get("learned_concept"):
            try:
                from app.cognitive.staleness import check_for_stale_relatives

                new_id = result["learned_concept"].concept_id
                staleness_result = check_for_stale_relatives(
                    new_concept_id=new_id,
                    new_summary=summary,
                    retrieval_engine=retrieval_engine,
                    supersede_fn=self._supersede_concept,
                )
                if staleness_result.concepts_superseded > 0:
                    result["staleness_t1"] = {
                        "superseded": staleness_result.concepts_superseded,
                        "details": staleness_result.details,
                        "time_ms": staleness_result.time_ms,
                    }
            except Exception as e:
                logger.warning(f"Staleness T1 check failed (non-fatal): {e}")

        # --- RETRIEVAL-020b: Evolution supersession check on novel concepts ---
        # Fires after concept creation to detect type-progression pairs
        # (e.g., observation → principle) in the EVOLUTION ZONE (cosine 0.50-0.82).
        if result.get("action") == "created" and result.get("learned_concept"):
            try:
                from app.cognitive.evolution import check_evolution_supersession
                from app.core.config import EVOLUTION_CANARY_MODE

                lc = result["learned_concept"]
                evo_result = check_evolution_supersession(
                    new_concept_id=lc.concept_id,
                    new_concept_type=getattr(lc, "concept_type", "observation"),
                    new_authority=getattr(lc, "authority_score", None) or 0.5,
                    retrieval_engine=retrieval_engine,
                    concept_loader=load_concept,
                    supersede_fn=self._supersede_concept,
                    canary_mode=EVOLUTION_CANARY_MODE,
                )
                if evo_result.pair_detected:
                    result["evolution_t3"] = {
                        "pair_detected": evo_result.pair_detected,
                        "older_concept_id": evo_result.older_concept_id,
                        "newer_concept_id": evo_result.newer_concept_id,
                        "composite_score": evo_result.composite_score,
                        "type_progression": evo_result.type_progression,
                        "action_taken": evo_result.action_taken,
                        "time_ms": evo_result.time_ms,
                        "canary_mode": EVOLUTION_CANARY_MODE,
                    }
                    logger.info(
                        f"RETRIEVAL-020b: Evolution pair detected — "
                        f"{evo_result.type_progression} "
                        f"(composite={evo_result.composite_score:.3f}, "
                        f"action={evo_result.action_taken}, "
                        f"canary={EVOLUTION_CANARY_MODE})"
                    )
            except Exception as evo_err:
                logger.warning(f"RETRIEVAL-020b: Evolution check failed (non-fatal): {evo_err}")

        # --- EXPLICIT_SUPERSESSION_SPEC v1.1: Declared supersession from extraction ---
        if result.get("action") == "created" and result.get("learned_concept"):
            supersede_ids = insight.get("supersedes")
            if supersede_ids:
                new_id = result["learned_concept"].concept_id
                explicit_count = 0
                for old_id in supersede_ids[:5]:  # Hard cap per concept
                    if old_id == new_id:  # A2: self-referential guard
                        logger.warning(f"EXPLICIT_SUPERSESSION: skipping self-reference {old_id}")
                        continue
                    success = self._supersede_concept(
                        old_id, new_id, "Explicit supersession declared at extraction time"
                    )
                    if success:
                        explicit_count += 1
                        logger.info(f"EXPLICIT_SUPERSESSION: '{old_id}' → '{new_id}'")
                if explicit_count > 0:
                    result["explicit_supersessions"] = explicit_count

        # L3: Log when supersedes declared but concept not created
        if insight.get("supersedes") and result.get("action") != "created":
            logger.warning(
                f"EXPLICIT_SUPERSESSION: supersedes declared but concept not created "
                f"(action={result.get('action')}). Targets: {insight['supersedes']}"
            )

        return result

    def _maybe_promote_maturity(self, concept_id: str) -> None:
        """W7: Check if a concept qualifies for maturity promotion.

        Promotion rules:
          QUARANTINED + evidence_count >= QUARANTINE_PROMOTION_MIN_EVIDENCE → PROVISIONAL
          PROVISIONAL + evidence_count >= 1 + access_count >= 5 → ESTABLISHED
          PROVISIONAL + reinforcement >= 8 → ESTABLISHED

        Guards (ARCH-D05):
          - Superseded concepts are never promoted
          - Concepts with confidence < 0.25 are not promoted to ESTABLISHED
        """
        # BENCH-INFRA-008: Skip maturity promotion in benchmark readonly mode.
        # Prevents PROVISIONAL→ESTABLISHED cascades that cause ±4% EM noise.
        import os as _os
        if _os.environ.get("PITH_BENCHMARK_READONLY", "").lower() in ("true", "1"):
            return

        from app.core.config import (
            FEATURE_FLAGS,
            PROVISIONAL_PROMOTION_MIN_ACCESS,
            PROVISIONAL_PROMOTION_MIN_EVIDENCE,
            QUARANTINE_PROMOTION_MIN_EVIDENCE,
            REINFORCEMENT_PROMOTION_THRESHOLD,
        )

        if not FEATURE_FLAGS.get("INGESTION_VALIDATION_ENABLED", False):
            return

        concept = load_concept(concept_id, track_access=False)
        if not concept:
            return

        # ARCH-D05: Guard against promoting superseded concepts
        if getattr(concept, "superseded_by", None):
            return

        maturity = getattr(concept, "maturity", "ESTABLISHED")
        evidence_count = len(concept.evidence) if concept.evidence else 0
        access_count = getattr(concept, "access_count", 0)
        reinforcement = getattr(concept, "reinforcement_count", 0)
        confidence = getattr(concept, "confidence", 0.0)
        new_maturity = maturity

        # MAINT-003: Use config constant instead of hardcoded 3
        if maturity == "QUARANTINED" and evidence_count >= QUARANTINE_PROMOTION_MIN_EVIDENCE:
            new_maturity = "PROVISIONAL"
        elif (
            maturity == "PROVISIONAL"
            and confidence >= 0.25  # ARCH-D05: confidence floor
            and (
                (evidence_count >= PROVISIONAL_PROMOTION_MIN_EVIDENCE
                 and access_count >= PROVISIONAL_PROMOTION_MIN_ACCESS)
                or reinforcement >= REINFORCEMENT_PROMOTION_THRESHOLD
            )
        ):
            new_maturity = "ESTABLISHED"

        if new_maturity != maturity:
            concept.maturity = new_maturity
            concept.maturity_promoted_at = _utc_now_iso()
            concept.maturity_promotion_evidence = f"Auto-promoted: evidence={evidence_count}, access={access_count}"
            save_concept(concept)
            logger.info(
                f"W7: Maturity promotion {concept_id}: {maturity} → {new_maturity} "
                f"(evidence={evidence_count}, access={access_count})"
            )

    def _evolve_existing_from_dedup(
        self, match: dict, insight: dict, request: SessionLearnRequest, extraction_source: str = "heuristic"
    ) -> dict:
        """Evolve an existing concept with new evidence from conversation.

        Takes dedup result dict {concept_id, cosine_score, knowledge_area, evidence_count}.
        Implements S1 (source tagging), S2 (self-corroboration guard), S5 (HHI cap).
        """
        from app.cognitive.learning import evolve_concept

        concept_id = match["concept_id"]

        # Evidence saturation check
        if match.get("evidence_count", 0) >= 10:
            logger.debug(f"session_learn: evidence saturated for {concept_id}")
            return {"action": "skipped_saturated"}

        # S2: Self-corroboration guard — check if existing concept has same source
        existing = load_concept(concept_id, track_access=False)
        corroboration_type = None
        confidence_boost = 0.02
        if existing and existing.metadata:
            existing_source = existing.metadata.get("extraction_source", "heuristic")
            if existing_source == extraction_source:
                corroboration_type = "same_source"
                confidence_boost = 0.01  # S2: cap boost for same-source
            else:
                corroboration_type = "cross_source"

        # S5: HHI confidence cap — single-source concepts capped at 0.75
        if existing and corroboration_type == "same_source":
            if existing.confidence >= 0.75:
                confidence_boost = 0.0  # Already at cap

        # FIX-2 (EVOLUTION_CHAIN_BREAK): Type-aware merge routing.
        # When incoming concept has a higher TYPE_RANK than existing, upgrade the type.
        # When same rank, use incoming summary if more specific (>1.2x longer).
        # Never downgrade type rank.
        from app.cognitive.supersession import TYPE_RANK

        new_concept_type = None
        new_summary = None
        incoming_type = insight.get("type", "observation")
        existing_type = existing.concept_type if existing else "observation"
        incoming_rank = TYPE_RANK.get(incoming_type, 0)
        existing_rank = TYPE_RANK.get(existing_type, 0)

        if incoming_rank > existing_rank:
            # TYPE UPGRADE
            new_concept_type = incoming_type
            new_summary = insight["summary"]
            confidence_boost = max(confidence_boost, 0.05)
            logger.info(
                "FIX-2: Type upgrade for %s: %s (rank %d) -> %s (rank %d)",
                concept_id,
                existing_type,
                existing_rank,
                incoming_type,
                incoming_rank,
            )
        elif incoming_rank == existing_rank and existing:
            # INGEST-057: Cosine-gated replacement policy (replaces INGEST-027 "newest wins").
            # Tier 1 (cosine >= 0.80): full replacement authority.
            # Tier 2 (0.70 <= cosine < 0.80): replace only if incoming is longer or more specific.
            # Tier 3 (0.55 <= cosine < 0.70): evidence-only evolution, summary preserved.
            # Feature flag: REPLACEMENT_GATE_ENABLED (OFF = fallback to INGEST-027 newest-wins).
            from app.core.config import (
                get_feature_flag,
                REPLACEMENT_GATE_STRONG,
                REPLACEMENT_GATE_MODERATE,
                REPLACEMENT_GATE_COSINE_EPSILON,
                MAX_EVIDENCE_PER_CONCEPT,
            )

            _cosine = match.get("cosine_score", 0.0)

            if not get_feature_flag("REPLACEMENT_GATE_ENABLED", False):
                # INGEST-027 fallback: newest wins (original behavior).
                new_summary = insight["summary"]
                confidence_boost = max(confidence_boost, 0.03)
            elif _cosine >= REPLACEMENT_GATE_STRONG - REPLACEMENT_GATE_COSINE_EPSILON:
                # Tier 1: Strong match — full replacement authority.
                new_summary = insight["summary"]
                confidence_boost = max(confidence_boost, 0.05)
                logger.info(
                    "INGEST-057 Tier1: REPLACE concept=%s cosine=%.4f",
                    concept_id, _cosine,
                )
            elif _cosine >= REPLACEMENT_GATE_MODERATE - REPLACEMENT_GATE_COSINE_EPSILON:
                # Tier 2: Moderate match — replace only if incoming is longer or more specific.
                try:
                    _incoming_longer = len(insight["summary"]) > len(existing.summary or "") * 1.15
                    _incoming_entities = _has_named_entities(insight["summary"])
                    _existing_entities = _has_named_entities(existing.summary or "")
                    _incoming_more_specific = _incoming_entities and not _existing_entities
                except Exception as _nlp_err:
                    logger.warning("INGEST-057 Tier2: NLP extraction failed: %s", _nlp_err)
                    _incoming_longer = False
                    _incoming_more_specific = False

                if _incoming_longer or _incoming_more_specific:
                    new_summary = insight["summary"]
                    confidence_boost = max(confidence_boost, 0.03)
                    logger.info(
                        "INGEST-057 Tier2: CONDITIONAL_REPLACE concept=%s cosine=%.4f reason=%s",
                        concept_id, _cosine,
                        "longer" if _incoming_longer else "more_specific",
                    )
                else:
                    # Tier 2 condition failed — evidence-only (same as Tier 3).
                    confidence_boost = max(confidence_boost, 0.02)
                    logger.info(
                        "INGEST-057 Tier2: EVIDENCE_ONLY concept=%s cosine=%.4f (condition_failed)",
                        concept_id, _cosine,
                    )
            else:
                # Tier 3: Weak match — evidence-only evolution, summary preserved.
                # Cap evidence to prevent unbounded pile-up.
                _existing_evidence_count = match.get("evidence_count", 0)
                if _existing_evidence_count >= MAX_EVIDENCE_PER_CONCEPT:
                    logger.info(
                        "INGEST-057 Tier3: EVIDENCE_CAP_REACHED concept=%s count=%d, skipping",
                        concept_id, _existing_evidence_count,
                    )
                    return {"action": "skipped_evidence_cap"}
                confidence_boost = max(confidence_boost, 0.01)
                logger.info(
                    "INGEST-057 Tier3: EVIDENCE_ONLY concept=%s cosine=%.4f",
                    concept_id, _cosine,
                )
        # incoming_rank < existing_rank: DO NOT downgrade.

        # P0-PRECISION: Specificity guard — don't replace a specific summary with a generic one.
        # Evolution re-lossification confirmed 2026-03-17: type upgrades and 1.2x length triggers
        # can replace "Pilsner or Lager for Seco de Cordero" with "beer for lamb dish".
        # Guard: if old summary has named entities/specifics and new doesn't, keep old summary.
        if new_summary and existing and existing.summary:
            old_has_specifics = _has_named_entities(existing.summary)
            new_has_specifics = _has_named_entities(new_summary)
            if old_has_specifics and not new_has_specifics:
                global _PRECISION_GUARD_BLOCKS
                _PRECISION_GUARD_BLOCKS += 1
                logger.info(
                    "P0-PRECISION: Blocking summary replacement — old has named entities, new doesn't. "
                    "old='%s' new='%s' (total_blocks=%d)",
                    existing.summary[:80],
                    new_summary[:80],
                    _PRECISION_GUARD_BLOCKS,
                )
                new_summary = None  # Keep old summary, still add new evidence + type upgrade

        # S1: Build evidence with extraction_source tag
        evidence_entry = {
            "source_type": "conversation",
            "content": f"Extracted from conversation: {insight['summary'][:200]}",
            "source_reference": f"session:{request.session_id}" if request.session_id else None,
            "reliability_weight": 0.7,
            "directness": 0.6,
            "consistency": 0.8,  # MAINT-009: Deprecated — not used in formula. Kept for backward compat.
            "extraction_source": extraction_source,
            "corroboration_type": corroboration_type,
            "model_origin": getattr(request, "model_id", "unknown"),  # FEDERATION L1.5
            "timestamp": _utc_now_iso(),
        }

        facet_metadata = {
            **_normalise_preference_facet_metadata(insight, request),
            **_normalise_advice_facet_metadata(insight, request),
            **_normalise_selection_facet_metadata(insight, request),
            **normalise_grouped_count_packet_metadata(insight, request),
        }
        branch_provenance_metadata = _normalise_branch_provenance_metadata(insight, request)
        grounding_metadata = _normalise_grounding_metadata(insight)
        client_metadata = insight.get("metadata") if isinstance(insight.get("metadata"), dict) else {}
        boundary_metadata = {
            key: value
            for key, value in {
                "raw_knowledge_area": client_metadata.get("raw_knowledge_area"),
                "knowledge_area_label_kind": client_metadata.get("knowledge_area_label_kind"),
                "knowledge_area_facet": client_metadata.get("knowledge_area_facet"),
            }.items()
            if value is not None
        }

        evolution = ConceptEvolution(
            concept_id=concept_id,
            new_evidence=[evidence_entry],
            new_signals=insight.get("signals", []),
            confidence_change=confidence_boost,
            new_concept_type=new_concept_type,  # FIX-2: Set when incoming has higher TYPE_RANK
            new_summary=new_summary,  # FIX-2: Set when incoming is more specific
            new_metadata={
                **facet_metadata,
                **branch_provenance_metadata,
                **grounding_metadata,
            },
            session_id=request.session_id,  # CASCADE-001 A1.2: Enable reinforcement independence check
            raw_evidence_count=len(insight.get("evidence", [])),  # A1.5: Layer 1 count for cascade
        )

        result = evolve_concept(evolution)
        if result:
            self.record_learning_event()
            # FEDERATION L2: Emit event for cross-pith bridging
            self._emit_federation_event(
                "concept_evolved",
                concept_id,
                {
                    "summary": match.get("summary", "")[:500],
                    "new_confidence": getattr(result, "confidence", 0),
                    "knowledge_area": match.get("knowledge_area", "general"),
                },
                model_id=getattr(request, "model_id", "unknown"),
            )
            # FED-015: Write-time cross-session conflict detection (post-evolve)
            try:
                from app.features.federation import detect_write_conflict

                detect_write_conflict(
                    new_concept_data={
                        "id": concept_id,
                        "summary": match.get("summary", ""),
                        "knowledge_area": match.get("knowledge_area", "general"),
                        "authority_score": getattr(result, "authority_score", None),
                        "currency_score": getattr(result, "currency_score", None),
                        "embedding": getattr(result, "embedding", None),
                    },
                    source_session_id=request.session_id or "",
                )
            except Exception as e:
                logger.debug(f"FED-015: Evolve conflict check failed (non-fatal): {e}")

            # W7: Maturity promotion lifecycle check after evolution
            try:
                self._maybe_promote_maturity(concept_id)
            except Exception as e:
                logger.warning(f"W7: Maturity promotion check failed for {concept_id}: {e}")
            return {
                "action": "evolved",
                "evolved_concept": EvolvedConcept(
                    concept_id=concept_id,
                    version=result.version,
                    change=f"New evidence ({extraction_source}): {insight.get('type', 'insight')}",
                ),
                "associations": 0,
            }

        return {"action": "skipped_duplicate"}

    def _create_new_concept(
        self,
        insight: dict,
        request: SessionLearnRequest,
        retrieval_engine,
        search_results,
        extraction_source: str = "heuristic",
        skip_write_contradiction: bool = False,
    ) -> dict:
        """Create a new concept with PROVISIONAL maturity and content-hash ID.

        Includes quality gates, knowledge area assignment, auto-association,
        and S1 extraction source tagging.

        Args:
            skip_write_contradiction: Bug 6 fix — when True, skips the write-time
                contradiction check. Used when creating a concept via the supersession
                path, where _detect_contradiction already confirmed the conflict.
        """
        summary = insight["summary"]
        concept_type = insight.get("type", "observation")

        # ORIENTATION_V2 Fix A4: Content-type consistency gate at ingestion
        # Demotes misclassified types (e.g., backlog labeled "decision", impl detail labeled "principle")
        # Gauntlet 3.2 fix: trusts explicit PRINCIPLE: prefix
        concept_type = _validate_concept_type(summary, concept_type)
        insight["type"] = concept_type  # TUNE-EXTRACTION fix: propagate validated type to all consumers

        # Type-aware confidence defaults:
        # Abstract types (principles, methods, strategies) start LOWER —
        # they must earn confidence through citation, not assertion.
        from app.core.models import ABSTRACT_CONCEPT_TYPES

        if concept_type in ABSTRACT_CONCEPT_TYPES:
            confidence = max(insight.get("confidence", 0.35), 0.35)
            confidence = min(confidence, 0.55)  # Cap: principles earn trust, not assert it
        else:
            confidence = max(insight.get("confidence", 0.40), 0.35)

        # --- Content-hash concept ID ---
        content_hash = hashlib.sha256(summary.encode()).hexdigest()[:12]
        concept_id = f"conv_{content_hash}"

        # Check if concept already exists
        existing = load_concept(concept_id, track_access=False)
        if existing:
            logger.debug(f"session_learn: concept {concept_id} already exists, skipping")
            return {"action": "skipped_duplicate"}

        # --- Knowledge area resolution ---
        # DEBT-030: normalize_knowledge_area + infer_knowledge_area hoisted to module-level import

        # For client extractions, use the provided knowledge_area if available
        if extraction_source == "client" and insight.get("knowledge_area"):
            raw_area = insight["knowledge_area"]
            # KA-007: Client KA was already normalized in Tier 2 (strict=False).
            # Use strict=False here to preserve novel client KAs instead of
            # double-normalizing with strict=True which drops them to "unclassified".
            knowledge_area, ka_source, ka_confidence = classify_knowledge_area(
                summary=summary, raw_area=raw_area, strict=False
            )
        else:
            raw_area = self._resolve_knowledge_area(request, search_results)
            # DEBT-108/KA-003: Shared multi-tier classification (keyword → embedding)
            knowledge_area, ka_source, ka_confidence = classify_knowledge_area(
                summary=summary, raw_area=raw_area, strict=True
            )

        # S1: Build evidence with extraction_source tag
        evidence_entry = {
            "source_type": "conversation",
            "content": f"Extracted from conversation: {summary[:200]}",
            "source_reference": f"session:{request.session_id}" if request.session_id else None,
            "reliability_weight": 0.7,
            "directness": 0.6,
            "consistency": 0.8,  # MAINT-009: Deprecated — not used in formula. Kept for backward compat.
            "extraction_source": extraction_source,
            "model_origin": getattr(request, "model_id", "unknown"),  # FEDERATION L1.5
            "timestamp": _utc_now_iso(),
        }

        # For client extractions, include provided evidence strings
        client_evidence = insight.get("evidence", [])
        if client_evidence and extraction_source == "client":
            evidence_entry["content"] = f"Client evidence: {'; '.join(client_evidence[:3])}"[:200]

        # Memory Integrity §5.2.3: Evidence method anti-spoofing
        try:
            from app.governance.evidence_method import sanitize_evidence

            sanitize_evidence([evidence_entry], source_type=extraction_source)
        except Exception as e:
            logger.warning(f"Evidence anti-spoofing failed (non-fatal): {e}")

        now = _utc_now_iso()

        # INGEST-017: Structural fact classification (overrides markers + LLM)
        _is_factual = insight.get("is_factual", False)
        _temporal_category = insight.get("temporal_category", None)
        _factual_score = None
        _signals_fired = None

        try:
            from app.core.config import get_feature_flag
            if get_feature_flag("STRUCTURAL_CONCEPT_CLASSIFIER_ENABLED", True):
                from app.cognitive.fact_classifier import classify_concept
                _cls = classify_concept(
                    summary=insight["summary"],
                    concept_type=insight.get("type", "observation"),
                    knowledge_area=knowledge_area or "general",
                )
                _is_factual = _cls["is_factual"]
                _temporal_category = _cls["temporal_category"]
                _factual_score = _cls["factual_score"]
                _signals_fired = _cls["signals_fired"]
        except Exception:
            logger.debug("INGEST-017: structural classifier unavailable, using fallback")

        # TEMPORAL-002: Extract temporal reference from summary+evidence text
        try:
            from app.retrieval.temporal import extract_temporal_reference
            _temporal_text = summary + ' ' + ' '.join(str(e) for e in insight.get('evidence', []))
            _original_date = extract_temporal_reference(_temporal_text)
        except Exception:
            _original_date = None

        _source_observation_ts = _normalise_session_learn_observation_timestamp(request)
        _benchmark_temporal_override = _source_observation_ts if _session_learn_benchmark_mode_active() else None
        _created_at = _benchmark_temporal_override or now
        _original_date = _benchmark_temporal_override or _original_date
        client_metadata = insight.get("metadata") if isinstance(insight.get("metadata"), dict) else {}
        benchmark_source_metadata = {
            key: value
            for key, value in client_metadata.items()
            if key
            in {
                "beam_source_message",
                "beam_role",
                "beam_source_key",
                "beam_source_turn_id",
                "beam_source_turn_index",
                "beam_source_batch_idx",
                "beam_source_role",
                "benchmark_observation_date",
                "benchmark_timestamp",
            }
            and value is not None
            and value != ""
        }
        facet_metadata = {
            **_normalise_preference_facet_metadata(insight, request),
            **_normalise_advice_facet_metadata(insight, request),
            **_normalise_selection_facet_metadata(insight, request),
            **normalise_grouped_count_packet_metadata(insight, request),
        }
        branch_provenance_metadata = _normalise_branch_provenance_metadata(insight, request)
        grounding_metadata = _normalise_grounding_metadata(insight)
        boundary_metadata = {
            key: value
            for key, value in {
                "raw_knowledge_area": client_metadata.get("raw_knowledge_area"),
                "knowledge_area_label_kind": client_metadata.get("knowledge_area_label_kind"),
                "knowledge_area_facet": client_metadata.get("knowledge_area_facet"),
            }.items()
            if value is not None
        }

        new_concept = Concept(
            id=concept_id,
            version="v1",
            created_at=_created_at,
            concept_type=insight.get("type", "observation"),
            summary=summary,
            evidence=[evidence_entry],
            signals=insight.get("signals", []),
            confidence=confidence,
            stability=0.5,  # STABILITY-001 Component A: align with learning.py and schema default
            maturity="PROVISIONAL",
            content_hash=content_hash,
            knowledge_area=knowledge_area,  # KA-001: Set directly so save_concept writes it
            valid_from=_benchmark_temporal_override,
            content_updated_at=_benchmark_temporal_override,
            original_date=_original_date,  # TEMPORAL-002
            edit_provenance=insight.get("edit_provenance"),  # RETRIEVAL-104
            session_id=request.session_id if request.session_id else None,  # AGENT-004
            metadata={
                "knowledge_area": knowledge_area,
                "knowledge_area_source": ka_source,
                "ka_confidence": ka_confidence,  # Float or None. Used by async reclass + trust gating.
                "extraction_source": extraction_source,
                "created_by": "session_learn",
                "source_session": request.session_id,
                "was_untyped": insight.get("was_untyped", False),
                # INGEST-017: Structural fact classification (canonical)
                "is_factual": _is_factual,
                "temporal_category": _temporal_category,
                "factual_score": _factual_score,
                "signals_fired": _signals_fired,
                # INGEST-017: Preserve marker/LLM values for comparison
                "marker_is_factual": insight.get("is_factual", False),
                "llm_is_factual": insight.get("llm_is_factual", None),
                # AGENT-001: request > session > default precedence
                "agent_id": self._resolve_agent_id(request),  # DEBT-019
                **benchmark_source_metadata,
                **facet_metadata,
                **branch_provenance_metadata,
                **grounding_metadata,
                **boundary_metadata,
            },
        )

        # Memory Integrity §5.1.5: Write-time contradiction check
        # Bug 6 fix: Skip when called from supersession path (already detected)
        # BENCHMARK-003: Skip contradiction check when dedup bypass is active —
        # benchmark facts are intentionally counter-factual and should not be rejected.
        from app.core.config import BENCHMARK as _bm_wcontra
        _skip_contra_for_benchmark = _bm_wcontra.skip_write_contradictions
        if not skip_write_contradiction and not _skip_contra_for_benchmark:
            try:
                from app.cognitive.contradiction import detect_write_contradiction

                contra_result = detect_write_contradiction(
                    new_summary=summary,
                    new_knowledge_area=knowledge_area,
                    concept_id=concept_id,
                )
                if contra_result.action == "HARD_REJECT":
                    logger.info(
                        f"session_learn: HARD_REJECT concept {concept_id} — "
                        f"contradicts {contra_result.contradicting_concept_id} "
                        f"(score={contra_result.max_score:.3f})"
                    )
                    return {"action": "rejected_contradiction", "reason": contra_result.reason}
                elif contra_result.action == "QUARANTINE":
                    new_concept.maturity = "QUARANTINED"
                    # STABILITY-026: M3 ceiling guard — cap confidence at ingest time
                    from app.core.config import PSIS_QUARANTINE_CONFIDENCE_CAP
                    if new_concept.confidence > PSIS_QUARANTINE_CONFIDENCE_CAP:
                        logger.info(
                            "STABILITY-026: Capped quarantined concept %s confidence %.3f → %.1f",
                            concept_id, new_concept.confidence, PSIS_QUARANTINE_CONFIDENCE_CAP,
                        )
                        new_concept.confidence = PSIS_QUARANTINE_CONFIDENCE_CAP
                    logger.info(
                        f"session_learn: quarantined {concept_id} — "
                        f"soft contradiction with {contra_result.contradicting_concept_id} "
                        f"(score={contra_result.max_score:.3f}, phase={getattr(contra_result, 'phase', 'unknown')})"
                    )
                    # EVIDENCE_QUARANTINE_SPEC Fix 5: Log governance event for quarantine tracking
                    try:
                        import json as _q_json

                        from app.storage import _db  # BUG-019: was missing, caused NameError

                        with _db() as _gov_conn:
                            _gov_conn.execute(
                                """INSERT INTO governance_events
                                   (event_type, concept_id, details, created_at)
                                   VALUES (?, ?, ?, ?)""",
                                (
                                    "CONCEPT_QUARANTINED",
                                    concept_id,
                                    _q_json.dumps(
                                        {
                                            "contradicting_concept_id": contra_result.contradicting_concept_id,
                                            "max_score": round(contra_result.max_score, 4),
                                            "phase": getattr(contra_result, "phase", None),
                                            "reason": getattr(contra_result, "reason", None),
                                        }
                                    ),
                                    _utc_now_iso(),
                                ),
                            )
                    except Exception:
                        logger.debug("Non-fatal: quarantine governance event logging failed", exc_info=True)
            except Exception as e:
                logger.warning(f"session_learn: contradiction check failed (non-fatal): {e}")
        else:
            logger.info(
                f"session_learn: skipping write-time contradiction check for {concept_id} "
                f"(supersession path — contradiction already confirmed)"
            )

        # Retrieval Defense W2: Epistemic classification before storage
        try:
            from app.governance.epistemic import classify_and_annotate_concept

            classified = classify_and_annotate_concept(new_concept)
            if classified:
                logger.info(
                    f"W2: Epistemic classification applied to {concept_id}: "
                    f"network={new_concept.epistemic_network}, "
                    f"verification={new_concept.verification_status}"
                )
        except Exception as e:
            logger.warning(f"W2: Epistemic classification failed for {concept_id}: {e}")

        # STABILITY-027: M3 compliance — cap confidence for PSIS-quarantined concepts at ingest
        from app.core.config import PSIS_QUARANTINE_CONFIDENCE_CAP, PSIS_QUARANTINE_EVIDENCE_MARKER
        if PSIS_QUARANTINE_EVIDENCE_MARKER in (new_concept.evidence or []):
            new_concept.confidence = min(new_concept.confidence, PSIS_QUARANTINE_CONFIDENCE_CAP)

        # EUNOMIA-040 Fix 3: Pre-compute subject_key for indexed RETRIEVAL-072 lookup
        try:
            new_concept.subject_key = _extract_subject_key(summary)
        except Exception:
            new_concept.subject_key = None

        _saved = save_concept(new_concept)

        # OPS-500-FIX: If concept was gated (e.g., GATE-BENCHMARKS), skip all downstream work
        if _saved is False:
            logger.debug("session_learn: concept %s gated at save_concept, skipping downstream", concept_id)
            return {
                "action": "gated",
                "concept_id": concept_id,
                "knowledge_area": knowledge_area,
            }

        # INGEST-037 Phase 2a: Save auto-extracted verbatim fragments
        # Fragments were attached to the insight dict by session_learn's
        # verbatim detection hook as insight["_verbatim_fragments"].
        _vf_list = insight.get("_verbatim_fragments", [])
        if _vf_list:
            try:
                from app.storage import save_verbatim_fragment

                _vf_saved = 0
                for _vf in _vf_list:
                    _vf_id = save_verbatim_fragment(
                        concept_id=concept_id,
                        fragment_type=getattr(_vf, "fragment_type", "text"),
                        content=getattr(_vf, "content", None),
                        pointer_uri=getattr(_vf, "pointer_uri", None),
                        pointer_meta=getattr(_vf, "pointer_meta", None),
                        concept_version=new_concept.version,
                    )
                    if _vf_id:
                        _vf_saved += 1
                if _vf_saved:
                    logger.info(
                        "INGEST-037: Saved %d/%d verbatim fragments for %s",
                        _vf_saved, len(_vf_list), concept_id,
                    )
            except Exception as _vf_save_err:
                logger.warning(
                    "INGEST-037: Verbatim fragment save failed for %s (non-fatal): %s",
                    concept_id, _vf_save_err,
                )

        # STABILITY-045: Queue expensive governance/similarity maintenance instead
        # of running it inline on the autolearn thread.
        _ss_result_data = None
        try:
            from app.session.autolearn_maintenance import (
                enqueue_autolearn_maintenance,
                enqueue_subject_key_supersession,
                kick_autolearn_maintenance_drain,
            )

            enqueue_autolearn_maintenance(
                concept_id,
                new_concept.version,
                source="session_learn",
                include_similarity=True,
            )
        except Exception as _maint_err:
            logger.warning(
                "STABILITY-045: Autolearn maintenance enqueue failed for %s (non-fatal): %s",
                concept_id,
                _maint_err,
            )

        # RETRIEVAL-072: Deterministic write-time subject-key supersession.
        # Runs even when PITH_DISABLE_EVOLVE=true. Uses structured pattern
        # matching (same as conflict prefilter) to detect duplicate subject keys.
        # Supersedes the OLDER concept, keeping the newly-written one.
        # No LLM call — pure string matching. Validated: +4 EM on SH 32k.
        #
        # Explicit supersession declarations are stronger product authority than
        # subject-key latest-write dedup. Let the explicit path below call the
        # unified execute_supersession() writer so branch authority metadata can
        # be populated from ready provenance envelopes.
        _subject_key_deferred = False
        try:
            from app.core.config import get_autolearn_subject_key_timeout_s
            from app.storage import apply_lifecycle_transition_conn, db_immediate
            _new_summary = summary
            _new_key = _extract_subject_key(_new_summary)
            _explicit_supersedes_declared = bool(insight.get("supersedes"))
            if _new_key and not _explicit_supersedes_declared:
                with db_immediate(
                    timeout_s=get_autolearn_subject_key_timeout_s(),
                    operation="autolearn_subject_key_supersession",
                ) as _sk_conn:
                    # EUNOMIA-040 Fix 3: Index-backed subject-key lookup
                    # instead of full-table scan + Python _extract_subject_key per row
                    _sk_candidates = _sk_conn.execute(
                        "SELECT id FROM concepts "
                        "WHERE subject_key = ? AND superseded_by IS NULL AND id != ?",
                        (_new_key, concept_id),
                    ).fetchall()
                    for (_sk_cid,) in _sk_candidates:
                        # Same subject key — supersede the old one
                        apply_lifecycle_transition_conn(
                            _sk_conn,
                            _sk_cid,
                            "supersede",
                            superseded_by=concept_id,
                            reason="RETRIEVAL-072: subject-key dedup",
                        )
                        logger.info(
                            "RETRIEVAL-072: Subject-key supersession: %s superseded %s "
                            "(key='%s')",
                            concept_id, _sk_cid, _new_key[:60],
                        )
                        break  # One supersession per write
            elif _new_key and _explicit_supersedes_declared:
                logger.info(
                    "RETRIEVAL-072: Subject-key supersession skipped for %s because explicit supersedes is declared",
                    concept_id,
                )
        except Exception as _sk_err:
            _subject_key_deferred = True
            logger.warning(
                "RETRIEVAL-072: Subject-key dedup failed for %s (non-fatal): %s",
                concept_id, _sk_err,
            )
            try:
                enqueue_subject_key_supersession(
                    concept_id,
                    new_concept.version,
                    source="session_learn_subject_key_fallback",
                )
            except Exception:
                pass
        finally:
            try:
                kick_autolearn_maintenance_drain()
            except Exception as _kick_err:
                logger.debug("STABILITY-045: maintenance drain kick skipped for %s: %s", concept_id, _kick_err)

        # FEDERATION L2: Emit event for cross-pith bridging
        self._emit_federation_event(
            "concept_proposed",
            concept_id,
            {
                "summary": summary[:500],
                "confidence": confidence,
                "knowledge_area": knowledge_area,
                "concept_type": insight.get("type", "observation"),
            },
            model_id=getattr(request, "model_id", "unknown"),
        )

        # FED-015: Write-time cross-session conflict detection
        try:
            from app.features.federation import detect_write_conflict

            detect_write_conflict(
                new_concept_data={
                    "id": concept_id,
                    "summary": summary,
                    "knowledge_area": knowledge_area,
                    "authority_score": getattr(new_concept, "authority_score", None),
                    "currency_score": getattr(new_concept, "currency_score", None),
                    "embedding": getattr(new_concept, "embedding", None),
                },
                source_session_id=request.session_id or "",
            )
        except Exception as e:
            logger.debug(f"FED-015: Propose conflict check failed (non-fatal): {e}")

        # CONCEPT_LIFECYCLE_SPEC L4: Track session-created concepts for end-of-session refresh
        self._session_concept_ids.add(concept_id)

        # --- L5: Auto-association (budget: 35ms) ---
        assoc_count = 0
        if request.auto_associate:
            assoc_count = self._auto_associate(concept_id, search_results, retrieval_engine,
                                                     cached_triples=self._cached_association_triples)

        # --- L6: Incremental index update ---
        try:
            retrieval_engine.add_concept(concept_id)
        except Exception as e:
            logger.warning(f"session_learn: index update failed for {concept_id}: {e}")

        # --- L6.5: Prospective indexing (RETRIEVAL-057) ---
        from app.core.config import PROSPECTIVE_INDEXING_ENABLED
        _benchmark_mode = os.environ.get("PITH_BENCHMARK_MODE", "").lower() in ("true", "1")
        if PROSPECTIVE_INDEXING_ENABLED and not _benchmark_mode:
            try:
                _evidence_strs_pi = []
                for _e in insight.get("evidence", []):
                    if isinstance(_e, str):
                        _evidence_strs_pi.append(_e)
                    elif isinstance(_e, dict):
                        _evidence_strs_pi.append(_e.get("content", ""))

                # Fire-and-forget via dedicated executor (non-blocking)
                import concurrent.futures as _cf_pi
                if not hasattr(self, '_pi_executor') or self._pi_executor is None:
                    self._pi_executor = _cf_pi.ThreadPoolExecutor(
                        max_workers=1, thread_name_prefix="prospective_idx"
                    )
                self._pi_executor.submit(
                    self._generate_implications,
                    concept_id=concept_id,
                    summary=summary,
                    knowledge_area=knowledge_area,
                    concept_type=insight.get("type", "observation"),
                    evidence=_evidence_strs_pi[:3],
                )
                logger.debug(f"RETRIEVAL-057: Queued implications generation for {concept_id}")
            except Exception as e:
                logger.debug(f"RETRIEVAL-057: Failed to queue implications: {e}")
        elif PROSPECTIVE_INDEXING_ENABLED and _benchmark_mode:
            logger.debug("RETRIEVAL-057: Skipped prospective indexing in benchmark mode")

        # Record learning event
        self.record_learning_event()

        result = {
            "action": "created",
            "learned_concept": LearnedConcept(
                concept_id=concept_id,
                summary=summary,
                confidence=confidence,
                knowledge_area=knowledge_area,
                concept_type=insight.get("type", "observation"),
            ),
            "associations": assoc_count,
        }
        # CONTRA-ACTIVATE-001: Surface supersession in result for pipeline summary
        if _ss_result_data:
            result["superseded"] = _ss_result_data
        return result

    # PERF-006: Cache federation_events table existence (won't disappear mid-session)
    _federation_table_exists: bool | None = None

    @classmethod
    def _reset_federation_cache(cls) -> None:
        """PERF-007: Invalidate federation table existence cache after migrations."""
        cls._federation_table_exists = None

    def _extract_events(
        self,
        combined_text: str,
        concept_ids: list[str],
        session_id: str | None = None,
    ) -> None:
        """INGEST-034: Background event extraction via LLM. Runs in _event_executor.

        Extracts structured {action, cause, consequence, actors} tuples from
        conversation text, then attaches them to the specified concepts via
        update_concept_data. Fire-and-forget — failure is logged, never blocks.
        """
        from openai import OpenAI as _SyncOAI
        from app.core.config import EE_LLM_MODEL, EE_MAX_OUTPUT_TOKENS, EE_TIMEOUT_SECONDS, EE_MAX_INPUT_CHARS
        from app.cognitive.extraction import build_event_extraction_prompt, parse_event_response

        try:
            text = combined_text[:EE_MAX_INPUT_CHARS]
            prompt = build_event_extraction_prompt(text)

            _or_key = os.environ.get("OPENROUTER_API_KEY", "")
            client = _SyncOAI(base_url="https://openrouter.ai/api/v1", api_key=_or_key, timeout=EE_TIMEOUT_SECONDS)
            response = client.chat.completions.create(
                model=EE_LLM_MODEL,
                max_tokens=EE_MAX_OUTPUT_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )

            response_text = response.choices[0].message.content or ""
            events = parse_event_response(response_text)

            if not events:
                logger.debug("INGEST-034: No events extracted from conversation")
                return

            event_dicts = [e.to_dict() for e in events]
            event_texts = [e.to_searchable_text() for e in events]

            logger.info(
                f"INGEST-034: Extracted {len(events)} events for {len(concept_ids)} concepts "
                f"(session={session_id})"
            )

            from app.storage import _db, update_concept_data
            import json

            for concept_id in concept_ids:
                try:
                    with _db() as conn:
                        row = conn.execute(
                            "SELECT data FROM concepts WHERE id = ? AND is_current = 1",
                            (concept_id,),
                        ).fetchone()
                        if not row:
                            continue

                        data = json.loads(row[0]) if row[0] else {}
                        data["events"] = event_dicts

                        if "metadata" not in data:
                            data["metadata"] = {}
                        data["metadata"]["events"] = event_dicts
                        data["metadata"]["event_texts"] = event_texts

                        update_concept_data(conn, concept_id, data)
                        # commit handled by _db() context manager

                    # Re-index to update embedding with event text
                    try:
                        from app.retrieval import retrieval_engine
                        retrieval_engine.add_concept(concept_id)
                    except Exception as reindex_err:
                        logger.warning(
                            f"INGEST-034: Re-index failed for {concept_id} after event attach: {reindex_err}"
                        )

                except Exception as attach_err:
                    logger.warning(f"INGEST-034: Failed to attach events to {concept_id}: {attach_err}")

        except Exception as _ee_err:
            if "timeout" in str(_ee_err).lower() or "timed out" in str(_ee_err).lower():
                logger.warning("INGEST-034: Event extraction LLM call timed out")
            else:
                logger.warning(f"INGEST-034: Event extraction API error: {_ee_err}")
            from app.ops.metrics import metrics as _ee_fail_metrics
            _ee_fail_metrics.record("extraction_llm_failure", 1.0, {"source": "event_extraction", "error": str(_ee_err)[:120]})

    def _generate_implications(
        self,
        concept_id: str,
        summary: str,
        knowledge_area: str,
        concept_type: str,
        evidence: list[str],
    ) -> None:
        """RETRIEVAL-057: Background prospective indexing (sync, runs in executor).

        Generates hypothetical future retrieval scenarios for a newly created
        concept. Updates the concept's data JSON with an 'implications' field
        and re-indexes embedding.
        """
        import os
        import time
        from datetime import datetime, UTC

        from app.core.config import (
            PI_COOLDOWN_SECONDS,
            PI_DAILY_BUDGET,
            PI_LLM_MODEL,
            PI_MAX_IMPLICATIONS,
            PI_MAX_OUTPUT_TOKENS,
            PI_MIN_SUMMARY_LENGTH,
        )

        t0 = time.perf_counter()

        # Gate 1: Minimum summary length
        if len(summary) < PI_MIN_SUMMARY_LENGTH:
            logger.debug(f"RETRIEVAL-057: Skipping implications — summary too short ({len(summary)} chars)")
            return

        # Gate 2: API key available
        if not os.environ.get("OPENROUTER_API_KEY"):
            logger.debug("RETRIEVAL-057: Skipping implications — no OPENROUTER_API_KEY")
            return

        # Gate 3: Daily budget check
        if not hasattr(self, "_pi_calls_today"):
            self._pi_calls_today = 0
            self._pi_day = datetime.now(UTC).date()

        current_day = datetime.now(UTC).date()
        if current_day != self._pi_day:
            self._pi_calls_today = 0
            self._pi_day = current_day

        if self._pi_calls_today >= PI_DAILY_BUDGET:
            logger.info(f"RETRIEVAL-057: PI daily budget exhausted ({PI_DAILY_BUDGET})")
            return

        # Gate 4: Cooldown check
        if hasattr(self, "_pi_last_call"):
            elapsed = time.perf_counter() - self._pi_last_call
            if elapsed < PI_COOLDOWN_SECONDS:
                logger.debug(f"RETRIEVAL-057: PI cooldown ({elapsed:.1f}s < {PI_COOLDOWN_SECONDS}s)")
                return

        try:
            from app.cognitive.extraction import build_implications_prompt, parse_implications_response

            prompt = build_implications_prompt(
                summary=summary,
                knowledge_area=knowledge_area,
                concept_type=concept_type,
                evidence=evidence,
                max_implications=PI_MAX_IMPLICATIONS,
            )

            from openai import OpenAI as _SyncOAI

            _or_key = os.environ.get("OPENROUTER_API_KEY", "")
            client = _SyncOAI(base_url="https://openrouter.ai/api/v1", api_key=_or_key)
            response = client.chat.completions.create(
                model=PI_LLM_MODEL,
                max_tokens=PI_MAX_OUTPUT_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )

            raw_text = response.choices[0].message.content or ""
            self._pi_calls_today += 1
            self._pi_last_call = time.perf_counter()

            # Record cost metric
            try:
                from app.ops.metrics import metrics as _pi_metrics
                _pi_metrics.record(
                    "prospective_indexing_llm_call",
                    1.0,
                    {
                        "model": PI_LLM_MODEL,
                        "input_tokens": response.usage.input_tokens if response.usage else 0,
                        "output_tokens": response.usage.output_tokens if response.usage else 0,
                        "concept_id": concept_id,
                    },
                )
            except Exception:
                pass  # Metrics are best-effort

            implications = parse_implications_response(raw_text, PI_MAX_IMPLICATIONS)

            if not implications:
                logger.info(f"RETRIEVAL-057: No implications parsed for {concept_id}")
                return

            # Update concept data JSON with implications (dual-storage pattern)
            from app.storage import _db, update_concept_data
            import json

            with _db() as conn:
                row = conn.execute(
                    "SELECT data FROM concepts WHERE id = ? AND is_current = 1",
                    (concept_id,),
                ).fetchone()
                if not row:
                    logger.warning(f"RETRIEVAL-057: Concept {concept_id} not found for implications update")
                    return

                current_data = json.loads(row[0]) if row[0] else {}
                current_data["implications"] = implications
                current_data["implications_model"] = PI_LLM_MODEL
                current_data["implications_generated_at"] = _utc_now_iso()

                # Also store in metadata for Pydantic-loaded access path
                meta = current_data.get("metadata", {})
                if not isinstance(meta, dict):
                    meta = {}
                meta["implications"] = implications
                current_data["metadata"] = meta

                update_concept_data(conn, concept_id, current_data)
                # commit handled by _db() context manager

            # Re-index with implications included in searchable text
            from app.retrieval import retrieval_engine
            retrieval_engine.add_concept(concept_id)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.info(
                f"RETRIEVAL-057: Generated {len(implications)} implications for {concept_id} "
                f"in {elapsed_ms:.0f}ms"
            )

        except Exception as e:
            logger.warning(f"RETRIEVAL-057: Implications generation failed for {concept_id}: {e}")

    def _emit_federation_event(
        self, event_type: str, concept_id: str, payload: dict, model_id: str = "unknown"
    ) -> None:
        """Emit a federation event for cross-pith bridging.

        Non-critical — failure is logged but doesn't block concept creation.
        Only emits if FEDERATION_EVENTS_ENABLED and federation_events table exists.
        """
        try:
            from app.core.config import FEATURE_FLAGS

            if not FEATURE_FLAGS.get("FEDERATION_EVENTS_ENABLED", False):
                return

            from app.storage import _db

            with _db() as conn:
                # PERF-006: Check cache first, query only on first call
                if self._federation_table_exists is None:
                    tables = [
                        r[0]
                        for r in conn.execute(
                            "SELECT name FROM sqlite_master WHERE type='table' AND name='federation_events'"
                        ).fetchall()
                    ]
                    self.__class__._federation_table_exists = "federation_events" in tables
                if not self._federation_table_exists:
                    return  # Pre-migration — silently skip

                conn.execute(
                    """INSERT INTO federation_events
                       (event_type, concept_id, source_session_id, source_model_id,
                        source_agent_id, payload, origin_brain, bridge_depth, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, NULL, 0, datetime('now'))""",
                    (
                        event_type,
                        concept_id,
                        self.current_session.session_id if self.current_session else None,
                        model_id,
                        getattr(self, "_current_agent_id", "default"),
                        json.dumps(payload),
                    ),
                )
        except Exception as e:
            logger.debug(f"Federation event emission failed (non-fatal): {e}")

    def _resolve_agent_id(self, request) -> str:
        """DEBT-019: Agent ID precedence: request > session > default."""
        req_aid = getattr(request, "agent_id", "default")
        if req_aid and req_aid != "default":
            return req_aid
        if self.current_session and getattr(self.current_session, "agent_id", "default") != "default":
            return self.current_session.agent_id
        return "default"

    def _resolve_knowledge_area(self, request: SessionLearnRequest, search_results) -> str:
        """3-tier knowledge area fallback (design gap §11.7).

        Tier 1: Explicit from request
        Tier 2: Inherit from nearest TF-IDF match (0.30-0.49)
        Tier 3: Default "conversation"
        """
        # Tier 1: Explicit override
        if request.knowledge_area and request.knowledge_area != "conversation":
            return request.knowledge_area

        # Tier 2: Nearest match inference
        if search_results:
            for result in search_results:
                if 0.30 <= result.relevance_score < 0.50:
                    if result.knowledge_area:
                        return result.knowledge_area

        # Tier 3: Default
        return "conversation"

    def _auto_associate(self, concept_id: str, search_results, retrieval_engine, cached_triples=None) -> int:
        """L5: Create associations with related concepts.

        Delegates to shared auto_associate_single pipeline which uses raw
        TF-IDF cosine similarity (consistent with batch pipeline).
        Returns count of associations created.
        """
        from app.cognitive.association import auto_associate_single
        from app.core.models import AutoAssociateSingleRequest

        request = AutoAssociateSingleRequest(threshold=0.12, max_edges=3)
        try:
            result = auto_associate_single(concept_id, request, cached_triples=cached_triples)
            return result.edges_created
        except Exception as e:
            logger.warning(f"session_learn: auto_associate failed for {concept_id}: {e}")
            return 0

    def _session_duration(self) -> float:
        """Compute session duration in seconds."""
        if not self.current_session or not self.current_session.started_at:
            return 0.0
        try:
            start = _ensure_aware(datetime.fromisoformat(self.current_session.started_at))
            end = _utc_now()
            return round((end - start).total_seconds(), 1)
        except (ValueError, TypeError):
            return 0.0
