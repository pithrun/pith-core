"""Operator-triggered observability command producer for STABILITY-048."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any

from app.core.config import get_feature_flag
from app.storage.command_log import DEFAULT_BUSY_TIMEOUT_MS, CommandLog
from app.storage.command_producers import AdmissionContext, AdmissionResult, append_if_admitted
from app.storage.command_writer import CommandWriter
from app.storage.operation_classes import FreshnessClass, OperationClass, OperationCommand

DEFAULT_OBSERVABILITY_SCOPE = "default"


def build_observability_noop_command(
    *,
    scope: str = DEFAULT_OBSERVABILITY_SCOPE,
    suffix: str | None = None,
    origin_id: str | None = None,
    session_id: str | None = None,
    agent_id: str | None = None,
) -> OperationCommand:
    scope_token = _token(scope or DEFAULT_OBSERVABILITY_SCOPE)
    suffix_token = _token(suffix) if suffix else None
    idempotency_key = f"observability:stage3a:{scope_token}"
    if suffix_token:
        idempotency_key = f"{idempotency_key}:{suffix_token}"
    return OperationCommand(
        idempotency_key=idempotency_key,
        origin_id=origin_id,
        session_id=session_id,
        agent_id=agent_id,
        op_class=OperationClass.OBSERVABILITY,
        freshness_class=FreshnessClass.BEST_EFFORT,
        payload_type="observability.noop",
        payload_version=1,
        payload_json={"scope": scope_token, "stage": "stability048-stage3a"},
    )


def append_observability_noop(
    *,
    path: Path | str | None = None,
    command_log: CommandLog | None = None,
    scope: str = DEFAULT_OBSERVABILITY_SCOPE,
    suffix: str | None = None,
    foreground: bool = False,
    drain_once: bool = False,
    origin_id: str | None = None,
    session_id: str | None = None,
    agent_id: str | None = None,
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
) -> dict[str, Any]:
    command = build_observability_noop_command(
        scope=scope,
        suffix=suffix,
        origin_id=origin_id,
        session_id=session_id,
        agent_id=agent_id,
    )
    log = command_log or _command_log_for_admission(
        path=path,
        foreground=foreground,
        busy_timeout_ms=busy_timeout_ms,
    )
    result = append_if_admitted(
        log,
        command,
        producer_name="observability_shadow",
        context=AdmissionContext(foreground=foreground),
    )
    drain_result = None
    if drain_once and result.accepted:
        drain_result = asdict(
            CommandWriter(log).drain_once(
                max_commands=1,
                op_classes=[OperationClass.OBSERVABILITY],
            )
        )
    return {
        "admission": _admission_dict(result),
        "command": {
            "idempotency_key": command.idempotency_key,
            "payload_type": command.payload_type,
            "op_class": command.op_class.value,
            "freshness_class": command.freshness_class.value,
        },
        "drain": drain_result,
        "health": _health_if_available(log, result.accepted),
        "path": str(log.path),
    }


def _command_log_for_admission(
    *,
    path: Path | str | None,
    foreground: bool,
    busy_timeout_ms: int,
) -> CommandLog:
    initialize = _append_may_be_attempted(foreground=foreground)
    try:
        return CommandLog(path, busy_timeout_ms=busy_timeout_ms, initialize=initialize)
    except OSError:
        if not initialize:
            raise
        return CommandLog(path, busy_timeout_ms=busy_timeout_ms, initialize=False)


def _append_may_be_attempted(*, foreground: bool) -> bool:
    if not get_feature_flag("COMMAND_PRODUCER_ADMISSION_ENABLED"):
        return False
    if not get_feature_flag("OBSERVABILITY_COMMAND_PRODUCER_ENABLED"):
        return False
    if foreground and not get_feature_flag("COMMAND_PRODUCER_FOREGROUND_APPEND_ENABLED"):
        return False
    return True


def _health_if_available(command_log: CommandLog, accepted: bool) -> dict[str, Any]:
    if not accepted and not command_log.path.exists():
        return {"available": False, "path": str(command_log.path), "reason": "not_initialized"}
    try:
        return command_log.health()
    except (OSError, sqlite3.Error) as exc:
        return {"available": False, "path": str(command_log.path), "reason": type(exc).__name__}


def _admission_dict(result: AdmissionResult) -> dict[str, Any]:
    return {
        "accepted": result.accepted,
        "producer": result.producer,
        "reason": result.reason,
        "failure_mode": result.failure_mode,
        "denied_layer": result.denied_layer,
        "command_id": result.command_id,
        "status": result.status,
        "created": result.created,
        "append_latency_ms": result.append_latency_ms,
        "over_budget": result.over_budget,
    }


def _token(value: str | None) -> str:
    raw = (value or DEFAULT_OBSERVABILITY_SCOPE).strip()
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in raw)
    return cleaned.strip("-") or DEFAULT_OBSERVABILITY_SCOPE
