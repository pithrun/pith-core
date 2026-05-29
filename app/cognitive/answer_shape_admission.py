"""Observe-only admission guard for answer-shape contracts.

This module is deliberately pure: no storage, network, LLM, benchmark, session,
or runtime-policy imports belong here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from app.cognitive.answer_shape_contracts import (
    CONTRACT_COUNT_LIST_GROUPED,
    CONTRACT_SOURCE_GOTCHA_MODULE,
    CONTRACT_SOURCE_PROCEDURE_MODULE_PAIR,
    CONTRACT_SOURCE_UI_DEFAULT_STATE,
    CONTRACT_TEMPORAL_DEICTIC_SOURCE_DATE,
    CONTRACT_TEMPORAL_DIRECT_DURATION,
    CONTRACT_TEMPORAL_ENDPOINT_RANGE,
    STATUS_ACCEPTED,
    AnswerShapeContract,
)

REASON_ACCEPTED_SUPPORTED_CONTRACT = "accepted_supported_contract"
REASON_DISABLED = "disabled"
REASON_CONTRACT_NOT_ACCEPTED = "contract_not_accepted"
REASON_UNSUPPORTED_CONTRACT_KIND = "unsupported_contract_kind"
REASON_RUNTIME_EFFECT_NOT_ALLOWED = "runtime_effect_not_allowed"
REASON_MISSING_REQUIRED_COMPONENT = "missing_required_component"
REASON_SUPPORT_NOT_VISIBLE = "support_not_visible"
REASON_FORBIDDEN_METADATA_PRESENT = "forbidden_metadata_present"
REASON_INVALID_CONTRACT_DECISION = "invalid_contract_decision"

SUPPORTED_CONTRACT_KINDS = frozenset(
    {
        CONTRACT_TEMPORAL_DIRECT_DURATION,
        CONTRACT_TEMPORAL_DEICTIC_SOURCE_DATE,
        CONTRACT_TEMPORAL_ENDPOINT_RANGE,
        CONTRACT_COUNT_LIST_GROUPED,
        CONTRACT_SOURCE_PROCEDURE_MODULE_PAIR,
        CONTRACT_SOURCE_UI_DEFAULT_STATE,
        CONTRACT_SOURCE_GOTCHA_MODULE,
    }
)
FORBIDDEN_METADATA_KEYS = frozenset(
    {
        "answer",
        "candidate_answer",
        "expected_answer",
        "expected_source_ref",
        "expected_sources",
        "gold_answer",
        "judge_rubric",
        "question_id",
        "row_id",
        "short_id",
    }
)


@dataclass(frozen=True)
class AnswerShapeAdmissionDecision:
    considered: bool
    admitted: bool
    reason: str
    contract_kind: str | None = None
    required_components: tuple[str, ...] = ()
    runtime_effect: bool = False
    diagnostics: Mapping[str, object] = field(default_factory=dict)


def admit_answer_shape_contract(
    decision: object,
    *,
    enabled: bool,
    visible_support_ids: Sequence[str] = (),
    metadata: Mapping[str, object] | None = None,
) -> AnswerShapeAdmissionDecision:
    """Return an observe-only admission decision for an answer-shape contract."""
    if not enabled:
        return _decision(
            considered=False,
            admitted=False,
            reason=REASON_DISABLED,
            enabled=False,
            visible_support_ids=visible_support_ids,
        )

    forbidden_paths = _forbidden_paths(metadata or {})
    if isinstance(decision, Mapping):
        forbidden_paths.extend(_forbidden_paths(decision, prefix="decision"))
    if forbidden_paths:
        return _decision(
            considered=True,
            admitted=False,
            reason=REASON_FORBIDDEN_METADATA_PRESENT,
            enabled=True,
            visible_support_ids=visible_support_ids,
            forbidden_paths=tuple(forbidden_paths),
        )

    try:
        status = _read_str(decision, "status")
        contract = _read_field(decision, "contract")
    except Exception:
        return _invalid(visible_support_ids=visible_support_ids)

    if status != STATUS_ACCEPTED:
        return _decision(
            considered=True,
            admitted=False,
            reason=REASON_CONTRACT_NOT_ACCEPTED,
            enabled=True,
            visible_support_ids=visible_support_ids,
        )
    if not isinstance(contract, AnswerShapeContract):
        return _invalid(visible_support_ids=visible_support_ids)

    component_forbidden_paths = _forbidden_paths(
        _as_mapping(contract.component_values), prefix="contract.component_values"
    )
    if component_forbidden_paths:
        return _decision(
            considered=True,
            admitted=False,
            reason=REASON_FORBIDDEN_METADATA_PRESENT,
            enabled=True,
            contract_kind=contract.contract_kind,
            required_components=tuple(contract.required_components),
            runtime_effect=bool(contract.runtime_effect),
            visible_support_ids=visible_support_ids,
            forbidden_paths=tuple(component_forbidden_paths),
        )

    if contract.contract_kind not in SUPPORTED_CONTRACT_KINDS:
        return _decision(
            considered=True,
            admitted=False,
            reason=REASON_UNSUPPORTED_CONTRACT_KIND,
            enabled=True,
            contract_kind=contract.contract_kind,
            required_components=tuple(contract.required_components),
            runtime_effect=bool(contract.runtime_effect),
            visible_support_ids=visible_support_ids,
        )
    if contract.runtime_effect:
        return _decision(
            considered=True,
            admitted=False,
            reason=REASON_RUNTIME_EFFECT_NOT_ALLOWED,
            enabled=True,
            contract_kind=contract.contract_kind,
            required_components=tuple(contract.required_components),
            runtime_effect=True,
            visible_support_ids=visible_support_ids,
        )

    component_values = _as_mapping(contract.component_values)
    required_components = tuple(str(component) for component in contract.required_components if component)
    missing_components = tuple(
        component for component in required_components if not _non_empty_component(component_values.get(component))
    )
    if not required_components or missing_components:
        return _decision(
            considered=True,
            admitted=False,
            reason=REASON_MISSING_REQUIRED_COMPONENT,
            enabled=True,
            contract_kind=contract.contract_kind,
            required_components=required_components,
            missing_components=missing_components,
            visible_support_ids=visible_support_ids,
        )

    required_support_ids = _support_ids(component_values.get("support_ids"))
    visible_support_set = {str(item) for item in visible_support_ids if str(item)}
    missing_support_ids = tuple(item for item in required_support_ids if item not in visible_support_set)
    if missing_support_ids:
        return _decision(
            considered=True,
            admitted=False,
            reason=REASON_SUPPORT_NOT_VISIBLE,
            enabled=True,
            contract_kind=contract.contract_kind,
            required_components=required_components,
            visible_support_ids=visible_support_ids,
            required_support_ids=required_support_ids,
            missing_support_ids=missing_support_ids,
        )

    return _decision(
        considered=True,
        admitted=True,
        reason=REASON_ACCEPTED_SUPPORTED_CONTRACT,
        enabled=True,
        contract_kind=contract.contract_kind,
        required_components=required_components,
        visible_support_ids=visible_support_ids,
        required_support_ids=required_support_ids,
    )


def _invalid(*, visible_support_ids: Sequence[str]) -> AnswerShapeAdmissionDecision:
    return _decision(
        considered=True,
        admitted=False,
        reason=REASON_INVALID_CONTRACT_DECISION,
        enabled=True,
        visible_support_ids=visible_support_ids,
    )


def _decision(
    *,
    considered: bool,
    admitted: bool,
    reason: str,
    enabled: bool,
    contract_kind: str | None = None,
    required_components: tuple[str, ...] = (),
    missing_components: tuple[str, ...] = (),
    runtime_effect: bool = False,
    visible_support_ids: Sequence[str] = (),
    required_support_ids: tuple[str, ...] = (),
    missing_support_ids: tuple[str, ...] = (),
    forbidden_paths: tuple[str, ...] = (),
) -> AnswerShapeAdmissionDecision:
    support_visibility = {
        "required_support_ids": required_support_ids,
        "visible_support_ids": tuple(str(item) for item in visible_support_ids if str(item)),
        "missing_support_ids": missing_support_ids,
    }
    diagnostics = {
        "enabled": enabled,
        "considered": considered,
        "admitted": admitted,
        "reason": reason,
        "contract_kind": contract_kind,
        "required_components": required_components,
        "missing_components": missing_components,
        "support_visibility": support_visibility,
        "runtime_effect": runtime_effect,
        "llm_call_delta": 0,
        "forbidden_metadata_paths": forbidden_paths,
    }
    return AnswerShapeAdmissionDecision(
        considered=considered,
        admitted=admitted,
        reason=reason,
        contract_kind=contract_kind,
        required_components=required_components,
        runtime_effect=runtime_effect,
        diagnostics=diagnostics,
    )


def _read_field(value: object, field_name: str) -> object:
    if isinstance(value, Mapping):
        return value.get(field_name)
    return getattr(value, field_name)


def _read_str(value: object, field_name: str) -> str:
    field_value = _read_field(value, field_name)
    return field_value if isinstance(field_value, str) else ""


def _as_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _non_empty_component(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _support_ids(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def _forbidden_paths(value: object, *, prefix: str = "metadata") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            if key_text.lower() in FORBIDDEN_METADATA_KEYS:
                paths.append(path)
            paths.extend(_forbidden_paths(item, prefix=path))
        return paths
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        for index, item in enumerate(value):
            paths.extend(_forbidden_paths(item, prefix=f"{prefix}[{index}]"))
    return paths
