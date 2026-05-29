"""Helpers for idempotent replay of consumer write requests."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from fastapi import HTTPException

from app.core.config import WRITE_STALE_MINUTES
from app.core.datetime_utils import _utc_now_iso
from app.core.profile import get_active_profile
from app.storage import (
    commit_write_request_replay,
    delete_processing_write_request,
    fail_write_request_replay,
    insert_write_request_processing,
    load_write_request_replay,
    mark_write_request_processing,
)

STALE_PROCESSING_TIMEOUT = timedelta(minutes=WRITE_STALE_MINUTES)

# MONITOR-135: write-durability telemetry
try:
    from app.core.metrics_facade import metrics as _wd_metrics
except Exception:
    _wd_metrics = None


@dataclass
class WriteReplayState:
    replay: dict | None = None
    request_id: str | None = None


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed
    except Exception:
        return None


def _timer(metric_name: str, endpoint: str, labels: dict | None = None):
    if not _wd_metrics or not hasattr(_wd_metrics, "timer"):
        return nullcontext()
    metric_labels = {"endpoint": endpoint}
    if labels:
        metric_labels.update(labels)
    return _wd_metrics.timer(metric_name, metric_labels)


def _flush_metrics() -> None:
    if not _wd_metrics or not hasattr(_wd_metrics, "flush"):
        return
    _wd_metrics.flush()


def begin_write_request(
    endpoint: str,
    request_id: str | None,
    *,
    request_payload: dict | None = None,
) -> WriteReplayState:
    if not request_id:
        return WriteReplayState(replay=None, request_id=None)

    profile = get_active_profile()
    now = _utc_now_iso()
    with _timer("write_request_begin_latency_ms", endpoint):
        row = load_write_request_replay(endpoint, profile, request_id)
        if row:
            if row["status"] == "committed" and row["response"]:
                payload = dict(row["response"])
                payload.setdefault("persistence_state", "committed")
                return WriteReplayState(replay=payload, request_id=request_id)
            if row["status"] == "failed":
                if row["response"]:
                    payload = dict(row["response"])
                    payload.setdefault("status", "failed")
                    payload.setdefault("persistence_state", "failed")
                    payload.setdefault("request_id", request_id)
                    return WriteReplayState(replay=payload, request_id=request_id)
                raise HTTPException(status_code=409, detail="Prior write request failed without replay payload")
            updated_at = _parse_timestamp(row["updated_at"])
            if (
                row["status"] == "processing"
                and updated_at is not None
                and datetime.now(UTC) - updated_at < STALE_PROCESSING_TIMEOUT
            ):
                if _wd_metrics:
                    _wd_metrics.record("write_durability_blocked_409", 1.0, {"endpoint": endpoint})
                    _flush_metrics()
                raise HTTPException(status_code=409, detail="Duplicate write request is already processing")
            # MONITOR-135: stale processing reclaim
            if row.get("request") is not None and request_payload is not None and row["request"] != request_payload:
                if _wd_metrics:
                    _wd_metrics.record("write_durability_payload_mismatch_409", 1.0, {"endpoint": endpoint})
                    _flush_metrics()
                raise HTTPException(status_code=409, detail="Duplicate write request payload differs from stored processing payload")
            if _wd_metrics:
                _wd_metrics.record("write_durability_stale_reclaim", 1.0, {"endpoint": endpoint})
            mark_write_request_processing(endpoint, profile, request_id, now, request_payload=request_payload)
            return WriteReplayState(replay=None, request_id=request_id)

        insert_write_request_processing(endpoint, profile, request_id, now, request_payload=request_payload)
    return WriteReplayState(replay=None, request_id=request_id)


def commit_write_request(endpoint: str, request_id: str | None, response: dict) -> dict:
    response.setdefault("persistence_state", "committed")
    if not request_id:
        return response

    profile = get_active_profile()
    now = _utc_now_iso()
    with _timer("write_request_commit_latency_ms", endpoint):
        commit_write_request_replay(endpoint, profile, request_id, response, now)
    _flush_metrics()
    return response


def abandon_write_request(endpoint: str, request_id: str | None, *, error_class: str = "unknown") -> None:
    if not request_id:
        return

    profile = get_active_profile()
    if _wd_metrics:
        _wd_metrics.record("write_durability_abandoned", 1.0, {"endpoint": endpoint, "error_class": error_class})
    with _timer("write_request_abandon_latency_ms", endpoint, {"error_class": error_class}):
        delete_processing_write_request(endpoint, profile, request_id)
    _flush_metrics()


def fail_write_request(endpoint: str, request_id: str | None, response: dict, *, error_class: str) -> dict:
    payload = dict(response)
    payload.setdefault("status", "failed")
    payload.setdefault("persistence_state", "failed")
    payload.setdefault("error_class", error_class)
    if not request_id:
        return payload

    payload.setdefault("request_id", request_id)
    profile = get_active_profile()
    now = _utc_now_iso()
    with _timer("write_request_fail_latency_ms", endpoint, {"error_class": error_class}):
        fail_write_request_replay(endpoint, profile, request_id, payload, now, error_class)
    if _wd_metrics:
        _wd_metrics.record("write_request_failed_terminal", 1.0, {"endpoint": endpoint, "error_class": error_class})
    _flush_metrics()
    return payload
