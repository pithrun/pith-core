"""STABILITY-048 operation-class command envelope primitives.

This module is intentionally storage-backend neutral. Segment 1 defines the
contract only; it does not route existing concept or retrieval writes through it.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class OperationClass(StrEnum):
    FOREGROUND_READ = "foreground_read"
    FOREGROUND_USER_WRITE = "foreground_user_write"
    POST_RESPONSE_LEARN = "post_response_learn"
    MAINTENANCE = "maintenance"
    OBSERVABILITY = "observability"
    ADMIN_BENCHMARK = "admin_benchmark"


class FreshnessClass(StrEnum):
    IMMEDIATE = "immediate"
    NEXT_TURN = "next_turn"
    EVENTUAL = "eventual"
    BEST_EFFORT = "best_effort"


class CommandStatus(StrEnum):
    QUEUED = "queued"
    CLAIMED = "claimed"
    APPLIED = "applied"
    FAILED = "failed"
    DEFERRED = "deferred"
    DISCARDED = "discarded"


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class OperationCommand:
    idempotency_key: str
    op_class: OperationClass
    freshness_class: FreshnessClass
    payload_type: str
    payload_version: int
    payload_json: dict[str, Any]
    priority: int = 0
    command_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    origin_id: str | None = None
    session_id: str | None = None
    agent_id: str | None = None
    causal_turn_id: str | None = None
    deadline_ms: int | None = None
    created_at: str = field(default_factory=utc_now_iso)
    not_before: str | None = None

    def __post_init__(self) -> None:
        if not self.command_id:
            raise ValueError("command_id is required")
        if not self.idempotency_key:
            raise ValueError("idempotency_key is required")
        if not self.payload_type:
            raise ValueError("payload_type is required")
        if int(self.payload_version) < 1:
            raise ValueError("payload_version must be >= 1")
        if self.deadline_ms is not None and int(self.deadline_ms) < 0:
            raise ValueError("deadline_ms must be >= 0")

    def payload_text(self) -> str:
        return json.dumps(self.payload_json, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "OperationCommand":
        payload_raw = row.get("payload_json") or "{}"
        payload = json.loads(payload_raw) if isinstance(payload_raw, str) else dict(payload_raw)
        return cls(
            command_id=str(row["command_id"]),
            idempotency_key=str(row["idempotency_key"]),
            origin_id=row.get("origin_id"),
            session_id=row.get("session_id"),
            agent_id=row.get("agent_id"),
            causal_turn_id=row.get("causal_turn_id"),
            op_class=OperationClass(str(row["op_class"])),
            priority=int(row["priority"]),
            freshness_class=FreshnessClass(str(row["freshness_class"])),
            deadline_ms=row.get("deadline_ms"),
            payload_type=str(row["payload_type"]),
            payload_version=int(row["payload_version"]),
            payload_json=payload,
            created_at=str(row["created_at"]),
            not_before=row.get("not_before"),
        )
