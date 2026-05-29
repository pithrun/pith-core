"""STABILITY-048 writer coordinator skeleton for operation commands."""

from __future__ import annotations

import socket
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from app.storage.command_log import CommandLog
from app.storage.operation_classes import OperationClass, utc_now_iso

NOOP_PAYLOAD_TYPES = {"noop", "test.noop", "observability.noop"}


@dataclass(frozen=True)
class DrainResult:
    claimed: int
    applied: int
    deferred: int
    failed: int


class CommandWriter:
    """Claim and apply sidecar operation commands.

    Segment 1 only applies synthetic no-op commands. Real concept, FTS,
    retrieval, raw-capture, and maintenance writes remain on existing paths.
    """

    def __init__(self, command_log: CommandLog, *, lease_owner: str | None = None) -> None:
        self.command_log = command_log
        self.lease_owner = lease_owner or f"command-writer:{socket.gethostname()}"

    def drain_once(
        self,
        *,
        max_commands: int = 10,
        op_classes: list[OperationClass] | None = None,
    ) -> DrainResult:
        rows = self.command_log.claim(
            lease_owner=self.lease_owner,
            op_classes=op_classes,
            limit=max_commands,
        )
        applied = deferred = failed = 0
        for row in rows:
            command_id = str(row["command_id"])
            payload_type = str(row["payload_type"])
            try:
                if payload_type in NOOP_PAYLOAD_TYPES:
                    self.command_log.mark_applied(command_id, result={"operation": "noop"})
                    applied += 1
                else:
                    self.command_log.mark_failed(
                        command_id,
                        f"unsupported payload_type in Segment 1: {payload_type}",
                        error_code="unsupported_payload_type",
                    )
                    failed += 1
            except Exception as exc:
                attempt_count = int(row.get("attempt_count") or 0)
                max_attempts = max(1, int(row.get("max_attempts") or 3))
                if attempt_count >= max_attempts:
                    self.command_log.mark_failed(command_id, repr(exc), error_code="max_attempts_exceeded")
                    failed += 1
                else:
                    self.command_log.mark_failed(
                        command_id,
                        repr(exc),
                        retry_at=_retry_at(),
                        error_code="writer_exception",
                    )
                    deferred += 1
        return DrainResult(claimed=len(rows), applied=applied, deferred=deferred, failed=failed)

    def health(self) -> dict[str, Any]:
        payload = self.command_log.health()
        payload["writer"] = {"lease_owner": self.lease_owner, "segment": "1-shadow"}
        return payload


def _retry_at() -> str:
    return (datetime.fromisoformat(utc_now_iso()) + timedelta(seconds=30)).isoformat()
