"""Producer admission gateway for STABILITY-048 sidecar command appends."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from app.core.config import get_feature_flag
from app.storage.command_admission import command_admission_observer
from app.storage.command_log import CommandLog
from app.storage.operation_classes import CommandStatus, FreshnessClass, OperationClass, OperationCommand

DEFAULT_MAX_PAYLOAD_BYTES = 4096
DEFAULT_MAX_APPEND_MS = 250


class AdmissionFailureMode(StrEnum):
    FATAL = "fatal"
    DEGRADE = "degrade"
    DROP = "drop"


class DeadLetterReason(StrEnum):
    UNKNOWN_PRODUCER = "unknown_producer"
    PRODUCER_DISABLED = "producer_disabled"
    FOREGROUND_APPEND_DISABLED = "foreground_append_disabled"
    UNSUPPORTED_PAYLOAD_TYPE = "unsupported_payload_type"
    OPERATION_CLASS_NOT_ALLOWED = "operation_class_not_allowed"
    FRESHNESS_CLASS_NOT_ALLOWED = "freshness_class_not_allowed"
    IDEMPOTENCY_KEY_INVALID = "idempotency_key_invalid"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    APPEND_TIMEOUT = "append_timeout"
    SIDECAR_UNAVAILABLE = "sidecar_unavailable"
    BEST_EFFORT_PRESSURE_DROP = "best_effort_pressure_drop"


@dataclass(frozen=True)
class ProducerDefinition:
    name: str
    payload_types: frozenset[str]
    op_class: OperationClass
    freshness_class: FreshnessClass
    feature_flag: str
    idempotency_prefix: str
    failure_mode: AdmissionFailureMode = AdmissionFailureMode.DROP
    may_run_foreground: bool = False
    max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES
    max_append_ms: int = DEFAULT_MAX_APPEND_MS
    max_attempts: int = 3


@dataclass(frozen=True)
class AdmissionContext:
    foreground: bool = False


@dataclass(frozen=True)
class AdmissionResult:
    accepted: bool
    producer: str
    reason: str | None = None
    failure_mode: str | None = None
    denied_layer: str | None = None
    command_id: str | None = None
    status: str | None = None
    created: bool = False
    append_latency_ms: float | None = None
    over_budget: bool = False


PRODUCERS: dict[str, ProducerDefinition] = {
    "synthetic_test": ProducerDefinition(
        name="synthetic_test",
        payload_types=frozenset({"test.noop"}),
        op_class=OperationClass.OBSERVABILITY,
        freshness_class=FreshnessClass.BEST_EFFORT,
        feature_flag="SYNTHETIC_COMMAND_PRODUCER_ENABLED",
        idempotency_prefix="synthetic:",
        failure_mode=AdmissionFailureMode.DROP,
    ),
    "observability_shadow": ProducerDefinition(
        name="observability_shadow",
        payload_types=frozenset({"observability.noop"}),
        op_class=OperationClass.OBSERVABILITY,
        freshness_class=FreshnessClass.BEST_EFFORT,
        feature_flag="OBSERVABILITY_COMMAND_PRODUCER_ENABLED",
        idempotency_prefix="observability:",
        failure_mode=AdmissionFailureMode.DROP,
    ),
}


def append_if_admitted(
    command_log: CommandLog,
    command: OperationCommand,
    *,
    producer_name: str,
    context: AdmissionContext | None = None,
) -> AdmissionResult:
    context = context or AdmissionContext()
    producer = PRODUCERS.get(producer_name)
    if not get_feature_flag("COMMAND_PRODUCER_ADMISSION_ENABLED"):
        return _deny(producer_name, DeadLetterReason.PRODUCER_DISABLED, AdmissionFailureMode.DROP, "global")
    if producer is None:
        return _deny(producer_name, DeadLetterReason.UNKNOWN_PRODUCER, AdmissionFailureMode.DROP)
    if not get_feature_flag(producer.feature_flag):
        return _deny(producer_name, DeadLetterReason.PRODUCER_DISABLED, producer.failure_mode, "producer")
    if context.foreground and not get_feature_flag("COMMAND_PRODUCER_FOREGROUND_APPEND_ENABLED"):
        return _deny(producer_name, DeadLetterReason.FOREGROUND_APPEND_DISABLED, producer.failure_mode, "foreground")

    reason = _validate_command(command, producer)
    if reason is not None:
        return _deny(producer_name, reason, producer.failure_mode)
    if command_log.busy_timeout_ms > producer.max_append_ms:
        return _deny(producer_name, DeadLetterReason.APPEND_TIMEOUT, producer.failure_mode)

    try:
        result = command_log.append(command, producer=producer.name, max_attempts=producer.max_attempts)
    except sqlite3.OperationalError as exc:
        text = str(exc).lower()
        reason = DeadLetterReason.APPEND_TIMEOUT if "locked" in text or "busy" in text else DeadLetterReason.SIDECAR_UNAVAILABLE
        return _deny(producer_name, reason, producer.failure_mode)
    except OSError:
        return _deny(producer_name, DeadLetterReason.SIDECAR_UNAVAILABLE, producer.failure_mode)

    over_budget = bool(result.append_latency_ms is not None and result.append_latency_ms > producer.max_append_ms)
    if over_budget:
        command_admission_observer.record(
            "command_append_over_budget",
            labels={"producer": producer.name, "max_append_ms": producer.max_append_ms},
        )
    return AdmissionResult(
        accepted=True,
        producer=producer.name,
        reason=DeadLetterReason.APPEND_TIMEOUT.value if over_budget else None,
        command_id=result.command_id,
        status=result.status.value,
        created=result.created,
        append_latency_ms=result.append_latency_ms,
        over_budget=over_budget,
    )


def _validate_command(command: OperationCommand, producer: ProducerDefinition) -> DeadLetterReason | None:
    if command.payload_type not in producer.payload_types:
        return DeadLetterReason.UNSUPPORTED_PAYLOAD_TYPE
    if command.op_class != producer.op_class:
        return DeadLetterReason.OPERATION_CLASS_NOT_ALLOWED
    if command.freshness_class != producer.freshness_class:
        return DeadLetterReason.FRESHNESS_CLASS_NOT_ALLOWED
    if not command.idempotency_key.startswith(producer.idempotency_prefix):
        return DeadLetterReason.IDEMPOTENCY_KEY_INVALID
    if len(command.payload_text().encode("utf-8")) > producer.max_payload_bytes:
        return DeadLetterReason.PAYLOAD_TOO_LARGE
    return None


def _deny(
    producer_name: str,
    reason: DeadLetterReason,
    failure_mode: AdmissionFailureMode,
    denied_layer: str | None = None,
) -> AdmissionResult:
    command_admission_observer.record(
        "command_admission_denied",
        labels={"producer": producer_name, "reason": reason.value, "denied_layer": denied_layer or ""},
    )
    status = CommandStatus.DISCARDED.value if failure_mode == AdmissionFailureMode.DROP else None
    return AdmissionResult(
        accepted=False,
        producer=producer_name,
        reason=reason.value,
        failure_mode=failure_mode.value,
        denied_layer=denied_layer,
        status=status,
    )
