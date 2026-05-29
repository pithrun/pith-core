"""Pure answer-shape contract candidates.

This module is deliberately isolated from runtime answer construction. It has no
storage, network, LLM, benchmark, session, or runtime-policy imports.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

STATUS_ACCEPTED = "accepted"
STATUS_FALLBACK = "fallback"
STATUS_INELIGIBLE = "ineligible"
STATUS_ERROR = "error"

CONTRACT_TEMPORAL_DIRECT_DURATION = "temporal_direct_duration"
CONTRACT_TEMPORAL_DEICTIC_SOURCE_DATE = "temporal_deictic_source_date"
CONTRACT_TEMPORAL_ENDPOINT_RANGE = "temporal_endpoint_range"
CONTRACT_COUNT_LIST_GROUPED = "count_list_grouped"
CONTRACT_SOURCE_PROCEDURE_MODULE_PAIR = "source_procedure_module_pair"
CONTRACT_SOURCE_UI_DEFAULT_STATE = "source_ui_default_state"
CONTRACT_SOURCE_GOTCHA_MODULE = "source_gotcha_module"

SOURCE_BASIS_RETRIEVED_MEMORY_CANDIDATE = "retrieved_memory_candidate"
SOURCE_BASIS_SOURCE_AUTHORITY_AUDIT = "source_authority_audit"
SELECTION_SCOPE_DIAGNOSTIC_ONLY = "diagnostic_only"
INSTRUCTION_PRESERVE_REQUIRED_COMPONENTS = "preserve_required_components"

REASON_ACCEPTED_DIRECT_DURATION = "accepted_direct_duration"
REASON_ACCEPTED_DEICTIC_SOURCE_DATE = "accepted_deictic_source_date"
REASON_ACCEPTED_ENDPOINT_RANGE = "accepted_endpoint_range"
REASON_NON_TEMPORAL_QUESTION = "non_temporal_question"
REASON_MISSING_REQUIRED_COMPONENT = "missing_required_component"
REASON_AMBIGUOUS_CANDIDATES = "ambiguous_candidates"
REASON_UNSUPPORTED_RELATIVE_DATE = "unsupported_relative_date"
REASON_NO_EVENT_OVERLAP = "no_event_overlap"
REASON_INVALID_TEMPORAL_DECISION = "invalid_temporal_decision"
REASON_ACCEPTED_COUNT_LIST_GROUPED = "accepted_count_list_grouped"
REASON_NON_COUNT_LIST_QUESTION = "non_count_list_question"
REASON_NO_PRIMARY_COUNT_PACKETS = "no_primary_count_packets"
REASON_INVALID_COUNT_PACKET_PLAN = "invalid_count_packet_plan"
REASON_ACCEPTED_SOURCE_AUTHORITY = "accepted_source_authority"
REASON_INVALID_SOURCE_AUTHORITY = "invalid_source_authority"
REASON_SOURCE_AUTHORITY_NOT_FOUND = "source_authority_not_found"
REASON_SOURCE_AUTHORITY_CONFLICT = "source_authority_conflict"
REASON_UNSUPPORTED_SOURCE_CONTRACT_FAMILY = "unsupported_source_contract_family"
REASON_FORBIDDEN_SOURCE_AUTHORITY_METADATA = "forbidden_source_authority_metadata"
REASON_MISSING_SOURCE_AUTHORITY_COMPONENT = "missing_source_authority_component"

_TEMPORAL_DIRECT_DURATION_REASON = "direct_duration"
_TEMPORAL_DEICTIC_SOURCE_DATE_REASON = "deictic_source_date"
_TEMPORAL_ENDPOINT_RANGE_REASON = "endpoint_range"
_PASS_THROUGH_FALLBACK_REASONS = {
    REASON_NON_TEMPORAL_QUESTION,
    "missing_provenance",
    "invalid_provenance",
    REASON_AMBIGUOUS_CANDIDATES,
    REASON_NO_EVENT_OVERLAP,
    REASON_UNSUPPORTED_RELATIVE_DATE,
    "no_supported_temporal_evidence",
}
_DURATION_RE = re.compile(
    r"\b(?P<value>\d{1,4})\s+(?P<unit>days?|weeks?|months?|years?)\b",
    re.IGNORECASE,
)
_ENDPOINT_RANGE_RE = re.compile(
    r"^\s*from\s+(?P<start>.+?)\s+to\s+(?P<end>.+?)\s*:\s*"
    r"(?P<duration>\d{1,4})\s+days?\.?\s*$",
    re.IGNORECASE,
)
_GROUPED_COUNT_DERIVATION_METHODS = frozenset(
    {
        "client_provided_grouped_count_v1",
        "source_text_grouped_count_v1",
        "user_selected_grouped_set_v1",
    }
)
_FORBIDDEN_COUNT_PACKET_KEYS = frozenset(
    {
        "answer",
        "answer_string",
        "benchmark_private",
        "candidate_answer",
        "expected_answer",
        "expected_source_ref",
        "expected_source_refs",
        "expected_sources",
        "gold_answer",
        "gold_id",
        "gold_ids",
        "judge_rubric",
        "qid",
        "question_id",
        "row_id",
        "rubric",
        "short_id",
        "source_chat_ids",
        "source_ref",
    }
)
_FORBIDDEN_SOURCE_AUTHORITY_KEYS = frozenset(
    {
        "answer",
        "answer_string",
        "candidate_answer",
        "expected_answer",
        "expected_source_ref",
        "expected_source_refs",
        "expected_sources",
        "gold_answer",
        "gold_id",
        "gold_ids",
        "judge_rubric",
        "qid",
        "question_id",
        "row_id",
        "rubric",
        "short_id",
    }
)
_SOURCE_AUTHORITY_CONTRACT_KIND_BY_FAMILY = {
    "enterprise_procedure_module_pair": CONTRACT_SOURCE_PROCEDURE_MODULE_PAIR,
    "enterprise_ui_default_state_mc": CONTRACT_SOURCE_UI_DEFAULT_STATE,
    "enterprise_gotcha_module": CONTRACT_SOURCE_GOTCHA_MODULE,
}


@dataclass(frozen=True)
class AnswerShapeContract:
    contract_kind: str
    required_components: tuple[str, ...]
    component_values: Mapping[str, str] = field(default_factory=dict)
    source_basis: str = SOURCE_BASIS_RETRIEVED_MEMORY_CANDIDATE
    runtime_effect: bool = False


@dataclass(frozen=True)
class AnswerShapeContractDecision:
    status: Literal["accepted", "fallback", "ineligible", "error"]
    contract: AnswerShapeContract | None
    reason: str
    prompt_plan: Mapping[str, str] | None = None


def build_answer_shape_contract(
    question: str,
    temporal_decision: object,
) -> AnswerShapeContractDecision:
    """Build a diagnostic-only answer-shape contract from a temporal decision."""
    try:
        status = _as_str(_read_field(temporal_decision, "status"))
        reason = _as_str(_read_field(temporal_decision, "reason"))
        answer = _as_optional_str(_read_field(temporal_decision, "answer"))
        diagnostics = _as_mapping(_read_field(temporal_decision, "diagnostics"))
    except Exception:
        return _fallback(REASON_INVALID_TEMPORAL_DECISION)

    if status != STATUS_ACCEPTED:
        return _fallback(_fallback_reason(reason))
    if not answer:
        return _fallback(REASON_MISSING_REQUIRED_COMPONENT)

    if reason == _TEMPORAL_DIRECT_DURATION_REASON:
        return _direct_duration_contract(answer, diagnostics)
    if reason == _TEMPORAL_DEICTIC_SOURCE_DATE_REASON:
        return _deictic_source_date_contract(answer, diagnostics)
    if reason == _TEMPORAL_ENDPOINT_RANGE_REASON:
        return _endpoint_range_contract(answer, diagnostics)
    return _fallback(REASON_INVALID_TEMPORAL_DECISION)


def classify_count_list_packets(
    question: str,
    candidates: Sequence[object],
    packet_metadata_by_concept: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    """Classify product-safe grouped-count packets for count/list questions.

    The classifier intentionally keys on sanitized ``grouped_count_packet``
    metadata, not benchmark row ids or packet concept-id prefixes.
    """
    query_intent = _classify_count_list_intent(question)
    primary_packets: list[Mapping[str, object]] = []
    support_ids: list[str] = []
    rejected: dict[str, str] = {}
    forbidden_material_detected = False
    packet_metadata = packet_metadata_by_concept or {}

    for index, candidate in enumerate(candidates):
        candidate_id = _candidate_id(candidate, index=index)
        metadata = _candidate_metadata(candidate)
        external_metadata = _metadata_for_candidate(packet_metadata, candidate_id)
        packet = _grouped_count_packet(metadata) or _grouped_count_packet(external_metadata)
        if not packet:
            support_ids.append(candidate_id)
            continue
        rejection_reason = _grouped_count_packet_rejection_reason(packet)
        if rejection_reason:
            rejected[candidate_id] = rejection_reason
            if rejection_reason == "forbidden_packet_metadata":
                forbidden_material_detected = True
            support_ids.append(candidate_id)
            continue
        primary_packets.append(
            {
                "candidate_id": candidate_id,
                "group_label": _clean_component(str(packet.get("group_label") or "")),
                "count": int(packet.get("count") or 0),
                "members": tuple(
                    _clean_component(str(member))
                    for member in packet.get("members", ())
                    if _clean_component(str(member))
                ),
                "derivation_method": str(packet.get("derivation_method") or ""),
            }
        )

    active = query_intent == "count_list"
    groups = tuple(primary_packets) if active else ()
    return {
        "query_intent": query_intent,
        "contract_active": active,
        "primary_count_packet_ids": tuple(packet["candidate_id"] for packet in groups),
        "support_memory_ids": tuple(support_ids),
        "rejected_packet_ids": tuple(rejected),
        "rejection_reasons": rejected,
        "forbidden_material_detected": forbidden_material_detected,
        "groups": groups,
        "total_count": sum(int(packet["count"]) for packet in groups),
    }


def build_count_list_answer_shape_contract(
    question: str,
    packet_plan: object,
) -> AnswerShapeContractDecision:
    """Build a diagnostic-only grouped count/list answer-shape contract."""
    plan = _as_mapping(packet_plan)
    if not plan:
        return _fallback(REASON_INVALID_COUNT_PACKET_PLAN)
    if plan.get("query_intent") != "count_list":
        return _fallback(REASON_NON_COUNT_LIST_QUESTION)
    groups = _as_sequence(plan.get("groups"))
    if not groups:
        return _fallback(REASON_NO_PRIMARY_COUNT_PACKETS)

    group_labels: list[str] = []
    group_member_blocks: list[str] = []
    support_ids: list[str] = []
    total_count = 0
    for group in groups:
        group_mapping = _as_mapping(group)
        label = _clean_component(_optional_component(group_mapping.get("group_label")))
        members = tuple(
            _clean_component(str(member))
            for member in _as_sequence(group_mapping.get("members"))
            if _clean_component(str(member))
        )
        try:
            count = int(group_mapping.get("count") or 0)
        except (TypeError, ValueError):
            count = 0
        candidate_id = _optional_component(group_mapping.get("candidate_id"))
        if not label or not members or count != len(members) or not candidate_id:
            return _fallback(REASON_MISSING_REQUIRED_COMPONENT)
        total_count += count
        group_labels.append(label)
        group_member_blocks.append(f"{label}: {', '.join(members)}")
        support_ids.append(candidate_id)

    if total_count < 1:
        return _fallback(REASON_MISSING_REQUIRED_COMPONENT)

    return _accepted(
        contract_kind=CONTRACT_COUNT_LIST_GROUPED,
        required_components=("total_count", "group_labels", "group_members"),
        component_values={
            "total_count": str(total_count),
            "group_count": str(len(groups)),
            "group_labels": " | ".join(group_labels),
            "group_members": " | ".join(group_member_blocks),
            "support_ids": ",".join(support_ids),
        },
        reason=REASON_ACCEPTED_COUNT_LIST_GROUPED,
    )


def build_source_authority_answer_shape_contract(source_authority: object) -> AnswerShapeContractDecision:
    """Build a diagnostic-only answer-shape contract from sanitized source authority."""
    authority = _as_mapping(source_authority)
    if not authority:
        return _fallback(REASON_INVALID_SOURCE_AUTHORITY)
    if _contains_forbidden_source_authority_material(authority):
        return _fallback(REASON_FORBIDDEN_SOURCE_AUTHORITY_METADATA)

    status = _optional_component(authority.get("status"))
    if status == "conflict":
        return _fallback(REASON_SOURCE_AUTHORITY_CONFLICT)
    if status != "source_contract_found":
        return _fallback(REASON_SOURCE_AUTHORITY_NOT_FOUND)

    contract_kind = _SOURCE_AUTHORITY_CONTRACT_KIND_BY_FAMILY.get(_optional_component(authority.get("contract_family")))
    if not contract_kind:
        return _fallback(REASON_UNSUPPORTED_SOURCE_CONTRACT_FAMILY)

    candidate_value = _clean_component(_optional_component(authority.get("candidate")))
    source_trajectory_id = _clean_component(_optional_component(authority.get("source_trajectory_id")))
    proof_state_indices = tuple(str(item).strip() for item in _as_sequence(authority.get("proof_state_indices")) if str(item).strip())
    proof_snippets = tuple(
        _clean_component(str(item))
        for item in _as_sequence(authority.get("proof_snippets"))
        if _clean_component(str(item))
    )
    proof = _clean_component(_optional_component(authority.get("proof")))
    if not candidate_value or not source_trajectory_id or not proof_state_indices or not (proof or proof_snippets):
        return _fallback(REASON_MISSING_SOURCE_AUTHORITY_COMPONENT)

    return _accepted(
        contract_kind=contract_kind,
        required_components=("candidate_value", "source_trajectory_id", "proof_state_indices"),
        component_values={
            "candidate_value": candidate_value,
            "source_trajectory_id": source_trajectory_id,
            "proof_state_indices": ",".join(proof_state_indices),
            "support_ids": source_trajectory_id,
        },
        reason=REASON_ACCEPTED_SOURCE_AUTHORITY,
        source_basis=SOURCE_BASIS_SOURCE_AUTHORITY_AUDIT,
    )


def _direct_duration_contract(
    answer: str,
    diagnostics: Mapping[str, object],
) -> AnswerShapeContractDecision:
    match = _DURATION_RE.search(answer)
    if not match:
        return _fallback(REASON_MISSING_REQUIRED_COMPONENT)
    amount = int(match.group("value"))
    unit = match.group("unit").lower().rstrip("s")
    duration = f"{amount} {unit if amount == 1 else unit + 's'}"
    return _accepted(
        contract_kind=CONTRACT_TEMPORAL_DIRECT_DURATION,
        required_components=("elapsed_duration",),
        component_values={
            "elapsed_duration": duration,
            "support_ids": _support_ids_value(diagnostics),
            "provenance_used": _provenance_used_value(diagnostics),
        },
        reason=REASON_ACCEPTED_DIRECT_DURATION,
    )


def _endpoint_range_contract(
    answer: str,
    diagnostics: Mapping[str, object],
) -> AnswerShapeContractDecision:
    match = _ENDPOINT_RANGE_RE.match(answer)
    if not match:
        return _fallback(REASON_MISSING_REQUIRED_COMPONENT)
    return _accepted(
        contract_kind=CONTRACT_TEMPORAL_ENDPOINT_RANGE,
        required_components=("start_endpoint", "end_endpoint", "elapsed_duration"),
        component_values={
            "start_endpoint": _clean_component(match.group("start")),
            "end_endpoint": _clean_component(match.group("end")),
            "elapsed_duration": f"{int(match.group('duration'))} days",
            "support_ids": _support_ids_value(diagnostics),
            "provenance_used": _provenance_used_value(diagnostics),
        },
        reason=REASON_ACCEPTED_ENDPOINT_RANGE,
    )


def _deictic_source_date_contract(
    answer: str,
    diagnostics: Mapping[str, object],
) -> AnswerShapeContractDecision:
    source_date = _clean_component(_optional_component(diagnostics.get("source_date")))
    deictic_phrase = _clean_component(_optional_component(diagnostics.get("deictic_phrase")))
    relative_relation = _clean_component(_optional_component(diagnostics.get("relative_relation")))
    if not answer or not source_date or not deictic_phrase or not relative_relation:
        return _fallback(REASON_MISSING_REQUIRED_COMPONENT)
    return _accepted(
        contract_kind=CONTRACT_TEMPORAL_DEICTIC_SOURCE_DATE,
        required_components=("source_date", "deictic_phrase", "relative_relation"),
        component_values={
            "answer_surface": _clean_component(answer),
            "source_date": source_date,
            "deictic_phrase": deictic_phrase,
            "relative_relation": relative_relation,
            "resolved_date": _optional_component(diagnostics.get("resolved_date")),
            "support_ids": _support_ids_value(diagnostics),
            "provenance_used": _provenance_used_value(diagnostics),
        },
        reason=REASON_ACCEPTED_DEICTIC_SOURCE_DATE,
    )


def _accepted(
    *,
    contract_kind: str,
    required_components: tuple[str, ...],
    component_values: Mapping[str, str],
    reason: str,
    source_basis: str = SOURCE_BASIS_RETRIEVED_MEMORY_CANDIDATE,
) -> AnswerShapeContractDecision:
    contract = AnswerShapeContract(
        contract_kind=contract_kind,
        required_components=required_components,
        component_values=dict(component_values),
        source_basis=source_basis,
        runtime_effect=False,
    )
    return AnswerShapeContractDecision(
        status=STATUS_ACCEPTED,
        contract=contract,
        reason=reason,
        prompt_plan={
            "instruction_kind": INSTRUCTION_PRESERVE_REQUIRED_COMPONENTS,
            "selection_scope": SELECTION_SCOPE_DIAGNOSTIC_ONLY,
            "required_components": ",".join(required_components),
            "source_basis": source_basis,
        },
    )


def _fallback(reason: str) -> AnswerShapeContractDecision:
    return AnswerShapeContractDecision(
        status=STATUS_FALLBACK,
        contract=None,
        reason=reason,
        prompt_plan=None,
    )


def _fallback_reason(reason: str) -> str:
    return reason if reason in _PASS_THROUGH_FALLBACK_REASONS else REASON_INVALID_TEMPORAL_DECISION


def _read_field(value: object, field_name: str) -> object:
    if isinstance(value, Mapping):
        return value.get(field_name)
    return getattr(value, field_name)


def _as_str(value: object) -> str:
    return value if isinstance(value, str) else ""


def _as_optional_str(value: object) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) else ""


def _as_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _as_sequence(value: object) -> Sequence[object]:
    return value if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)) else ()


def _clean_component(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().rstrip(".")


def _optional_component(value: object) -> str:
    return value if isinstance(value, str) else ""


def _support_ids_value(diagnostics: Mapping[str, object]) -> str:
    return _tupleish_value(diagnostics.get("support_ids"))


def _provenance_used_value(diagnostics: Mapping[str, object]) -> str:
    return _tupleish_value(diagnostics.get("provenance_used"))


def _tupleish_value(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (tuple, list)):
        return ",".join(str(item) for item in value if item is not None)
    return ""


def _classify_count_list_intent(question: str) -> str:
    normalized = question.lower()
    count_signal = any(signal in normalized for signal in ("how many", "number of", "count", "total"))
    list_signal = any(signal in normalized for signal in ("different", "which", "what"))
    entity_signal = any(signal in normalized for signal in ("series", "books", "items", "titles", "entities"))
    return "count_list" if count_signal and (list_signal or entity_signal) else "not_count_list"


def _candidate_id(candidate: object, *, index: int) -> str:
    for field_name in ("concept_id", "id", "support_id"):
        value = _read_optional_field(candidate, field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"candidate:{index}"


def _candidate_metadata(candidate: object) -> Mapping[str, object]:
    metadata = _read_optional_field(candidate, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _metadata_for_candidate(packet_metadata: Mapping[str, object], candidate_id: str) -> Mapping[str, object]:
    value = packet_metadata.get(candidate_id)
    if isinstance(value, Mapping) and "grouped_count_packet" in value:
        return value
    if isinstance(value, Mapping):
        return {"grouped_count_packet": value}
    return {}


def _grouped_count_packet(metadata: Mapping[str, object]) -> Mapping[str, object]:
    packet = metadata.get("grouped_count_packet")
    return packet if isinstance(packet, Mapping) else {}


def _grouped_count_packet_rejection_reason(packet: Mapping[str, object]) -> str | None:
    if _contains_forbidden_count_packet_material(packet):
        return "forbidden_packet_metadata"
    if packet.get("packet_type") != "grouped_count":
        return "unsupported_packet_type"
    if str(packet.get("derivation_method") or "") not in _GROUPED_COUNT_DERIVATION_METHODS:
        return "unsupported_derivation_method"
    label = _clean_component(str(packet.get("group_label") or ""))
    members = _as_sequence(packet.get("members"))
    try:
        count = int(packet.get("count") or 0)
    except (TypeError, ValueError):
        count = 0
    clean_members = tuple(_clean_component(str(member)) for member in members if _clean_component(str(member)))
    if not label or not clean_members or count != len(clean_members):
        return "count_member_mismatch"
    if not _as_sequence(packet.get("source_evidence")):
        return "missing_source_evidence"
    return None


def _contains_forbidden_count_packet_material(value: object) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).casefold() in _FORBIDDEN_COUNT_PACKET_KEYS:
                return True
            if _contains_forbidden_count_packet_material(nested):
                return True
        return False
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return any(_contains_forbidden_count_packet_material(item) for item in value)
    return False


def _contains_forbidden_source_authority_material(value: object) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).casefold() in _FORBIDDEN_SOURCE_AUTHORITY_KEYS:
                return True
            if _contains_forbidden_source_authority_material(nested):
                return True
        return False
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return any(_contains_forbidden_source_authority_material(item) for item in value)
    return False


def _read_optional_field(value: object, field_name: str) -> object:
    if isinstance(value, Mapping):
        return value.get(field_name)
    return getattr(value, field_name, None)
