"""Helpers for idempotent replay of consumer write requests."""

from __future__ import annotations

import copy
import hashlib
import json
import secrets
import sqlite3
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from fastapi import HTTPException

from app.core.config import WRITE_STALE_MINUTES
from app.core.datetime_utils import _utc_now_iso
from app.core.profile import get_active_profile
from app.core.request_identity import normalize_optional_request_id
from app.storage import (
    commit_write_request_replay,
    delete_processing_write_request,
    fail_write_request_replay,
    insert_write_request_processing,
    load_managed_write_request_replays_for_episode,
    load_write_request_replay,
    mark_write_request_processing,
)

STALE_PROCESSING_TIMEOUT = timedelta(minutes=WRITE_STALE_MINUTES)
STATUS_ENDPOINT_ALLOWLIST = frozenset({"session_learn", "session_end", "checkpoint"})
STATUS_SUMMARY_FIELDS = (
    "learning_events",
    "concepts_created",
    "concepts_evolved",
    "accepted_learning_events",
    "learning_capture_state",
    "session_linkage_state",
    "processing_state",
    "persistence_state",
    "status",
)

# MONITOR-135: write-durability telemetry
try:
    from app.core.metrics_facade import metrics as _wd_metrics
except Exception:
    _wd_metrics = None


@dataclass
class WriteReplayState:
    replay: dict | None = None
    request_id: str | None = None
    storage_request_id: str | None = None
    request_hash: str | None = None
    binding_hash: str | None = None
    session_id: str | None = None
    claim_token: str | None = None


@dataclass(frozen=True)
class ManagedReplayAuthority:
    """Redacted authority used to namespace a managed replay row."""

    profile: str
    binding_hash: str
    session_id: str


def _managed_storage_request_id(
    endpoint: str,
    external_request_id: str,
    authority: ManagedReplayAuthority,
) -> str:
    raw = f"v1\0{authority.profile}\0{authority.binding_hash}\0{endpoint}\0{external_request_id}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _managed_request_payload(
    request_payload: dict | None,
    *,
    binding_hash: str,
) -> dict | None:
    if request_payload is None:
        return None
    payload = copy.deepcopy(request_payload)
    binding = payload.get("binding")
    if isinstance(binding, dict):
        binding.pop("capability", None)
        binding["binding_hash"] = binding_hash
    return payload


def _canonical_request_hash(payload: dict | None) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _managed_state(
    *,
    replay: dict | None,
    external_request_id: str,
    storage_request_id: str,
    request_hash: str,
    authority: ManagedReplayAuthority,
    claim_token: str | None,
) -> WriteReplayState:
    return WriteReplayState(
        replay=replay,
        request_id=external_request_id,
        storage_request_id=storage_request_id,
        request_hash=request_hash,
        binding_hash=authority.binding_hash,
        session_id=authority.session_id,
        claim_token=claim_token,
    )


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


def _load_write_request_replay_conn(
    conn: sqlite3.Connection,
    endpoint: str,
    profile: str,
    request_id: str,
) -> dict | None:
    row = conn.execute(
        "SELECT status, response_json, request_json, attempt_count, last_error, "
        "lease_owner, lease_expires_at, next_retry_at, binding_hash, session_id, "
        "external_request_id, request_hash, claim_token, updated_at "
        "FROM write_request_replays WHERE endpoint=? AND profile=? AND request_id=?",
        (endpoint, profile, request_id),
    ).fetchone()
    if row is None:
        return None
    data = {key: row[key] for key in row.keys()} if hasattr(row, "keys") else dict(row)
    return {
        "status": data["status"],
        "response": json.loads(data["response_json"]) if data.get("response_json") else None,
        "request": json.loads(data["request_json"]) if data.get("request_json") else None,
        "request_json": data.get("request_json"),
        "attempt_count": int(data.get("attempt_count") or 0),
        "last_error": data.get("last_error"),
        "lease_owner": data.get("lease_owner"),
        "lease_expires_at": data.get("lease_expires_at"),
        "next_retry_at": data.get("next_retry_at"),
        "binding_hash": data.get("binding_hash"),
        "session_id": data.get("session_id"),
        "external_request_id": data.get("external_request_id"),
        "request_hash": data.get("request_hash"),
        "claim_token": data.get("claim_token"),
        "updated_at": data["updated_at"],
    }


def _insert_write_request_processing_conn(
    conn: sqlite3.Connection,
    *,
    endpoint: str,
    profile: str,
    storage_request_id: str,
    now: str,
    request_payload: dict | None,
    authority: ManagedReplayAuthority,
    external_request_id: str,
    request_hash: str,
    claim_token: str,
) -> int:
    from app.storage.session_bindings import ManagedQueuePermit, managed_queue_write_permit

    request_json = json.dumps(request_payload) if request_payload is not None else None
    with managed_queue_write_permit(
        ManagedQueuePermit(
            table_name="write_request_replays",
            binding_hash=authority.binding_hash,
        )
    ):
        cursor = conn.execute(
            "INSERT INTO write_request_replays(endpoint, profile, request_id, status, response_json, request_json, "
            "attempt_count, last_error, lease_owner, lease_expires_at, next_retry_at, binding_hash, session_id, "
            "external_request_id, request_hash, claim_token, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                endpoint,
                profile,
                storage_request_id,
                "processing",
                None,
                request_json,
                0,
                None,
                None,
                None,
                None,
                authority.binding_hash,
                authority.session_id,
                external_request_id,
                request_hash,
                claim_token,
                now,
                now,
            ),
        )
    return int(cursor.rowcount or 0)


def _mark_write_request_processing_conn(
    conn: sqlite3.Connection,
    *,
    endpoint: str,
    profile: str,
    storage_request_id: str,
    now: str,
    request_payload: dict | None,
    authority: ManagedReplayAuthority,
    expected_claim_token: str,
    new_claim_token: str,
) -> int:
    from app.storage.session_bindings import ManagedQueuePermit, managed_queue_write_permit

    request_json = json.dumps(request_payload) if request_payload is not None else None
    with managed_queue_write_permit(
        ManagedQueuePermit(
            table_name="write_request_replays",
            binding_hash=authority.binding_hash,
            claim_token=expected_claim_token,
        )
    ):
        cursor = conn.execute(
            "UPDATE write_request_replays SET status='processing', response_json=NULL, "
            "request_json=COALESCE(request_json, ?), lease_owner=NULL, lease_expires_at=NULL, "
            "claim_token=?, updated_at=? WHERE endpoint=? AND profile=? AND request_id=? "
            "AND binding_hash=? AND claim_token=?",
            (
                request_json,
                new_claim_token,
                now,
                endpoint,
                profile,
                storage_request_id,
                authority.binding_hash,
                expected_claim_token,
            ),
        )
    return int(cursor.rowcount or 0)


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
    authority: ManagedReplayAuthority | None = None,
    conn: sqlite3.Connection | None = None,
) -> WriteReplayState:
    if not request_id:
        return WriteReplayState(replay=None, request_id=None)
    if conn is not None and authority is None:
        raise ValueError("connection-aware replay begin requires managed authority")

    profile = get_active_profile()
    external_request_id = request_id
    storage_request_id = request_id
    request_hash = None
    claim_token = None
    if authority is not None:
        if authority.profile != profile:
            raise ValueError("managed replay profile does not match active profile")
        storage_request_id = _managed_storage_request_id(
            endpoint,
            external_request_id,
            authority,
        )
        request_payload = _managed_request_payload(
            request_payload,
            binding_hash=authority.binding_hash,
        )
        request_hash = _canonical_request_hash(request_payload)
        claim_token = secrets.token_hex(32)
    now = _utc_now_iso()
    timer = nullcontext() if conn is not None else _timer("write_request_begin_latency_ms", endpoint)
    with timer:
        row = (
            _load_write_request_replay_conn(conn, endpoint, profile, storage_request_id)
            if conn is not None
            else load_write_request_replay(endpoint, profile, storage_request_id)
        )
        if row:
            if authority is not None:
                if (
                    row.get("binding_hash") != authority.binding_hash
                    or row.get("session_id") != authority.session_id
                    or row.get("external_request_id") != external_request_id
                ):
                    raise HTTPException(
                        status_code=409,
                        detail={"error": "binding_mismatch"},
                    )
                if row.get("request_hash") != request_hash:
                    raise HTTPException(
                        status_code=409,
                        detail={"error": "request_payload_conflict"},
                    )
            if row["status"] == "committed" and row["response"]:
                payload = dict(row["response"])
                payload.setdefault("persistence_state", "committed")
                if authority is not None:
                    payload["request_id"] = external_request_id
                    return _managed_state(
                        replay=payload,
                        external_request_id=external_request_id,
                        storage_request_id=storage_request_id,
                        request_hash=request_hash,
                        authority=authority,
                        claim_token=row.get("claim_token"),
                    )
                return WriteReplayState(replay=payload, request_id=request_id)
            if row["status"] == "failed":
                if row["response"]:
                    payload = dict(row["response"])
                    payload.setdefault("status", "failed")
                    payload.setdefault("persistence_state", "failed")
                    payload.setdefault("request_id", external_request_id)
                    if authority is not None:
                        return _managed_state(
                            replay=payload,
                            external_request_id=external_request_id,
                            storage_request_id=storage_request_id,
                            request_hash=request_hash,
                            authority=authority,
                            claim_token=row.get("claim_token"),
                        )
                    return WriteReplayState(replay=payload, request_id=request_id)
                raise HTTPException(status_code=409, detail="Prior write request failed without replay payload")
            updated_at = _parse_timestamp(row["updated_at"])
            if (
                row["status"] == "processing"
                and updated_at is not None
                and datetime.now(UTC) - updated_at < STALE_PROCESSING_TIMEOUT
            ):
                if _wd_metrics and conn is None:
                    _wd_metrics.record("write_durability_blocked_409", 1.0, {"endpoint": endpoint})
                    _flush_metrics()
                raise HTTPException(status_code=409, detail="Duplicate write request is already processing")
            # MONITOR-135: stale processing reclaim
            if row.get("request") is not None and request_payload is not None and row["request"] != request_payload:
                if _wd_metrics and conn is None:
                    _wd_metrics.record("write_durability_payload_mismatch_409", 1.0, {"endpoint": endpoint})
                    _flush_metrics()
                raise HTTPException(
                    status_code=409, detail="Duplicate write request payload differs from stored processing payload"
                )
            if _wd_metrics and conn is None:
                _wd_metrics.record("write_durability_stale_reclaim", 1.0, {"endpoint": endpoint})
            if authority is not None:
                expected_claim_token = row.get("claim_token")
                if not expected_claim_token:
                    raise HTTPException(
                        status_code=409,
                        detail={"error": "managed_replay_claim_missing"},
                    )
                if conn is not None:
                    updated = _mark_write_request_processing_conn(
                        conn,
                        endpoint=endpoint,
                        profile=profile,
                        storage_request_id=storage_request_id,
                        now=now,
                        request_payload=request_payload,
                        authority=authority,
                        expected_claim_token=expected_claim_token,
                        new_claim_token=claim_token,
                    )
                else:
                    updated = mark_write_request_processing(
                        endpoint,
                        profile,
                        storage_request_id,
                        now,
                        request_payload=request_payload,
                        binding_hash=authority.binding_hash,
                        expected_claim_token=expected_claim_token,
                        new_claim_token=claim_token,
                    )
                if updated != 1:
                    raise HTTPException(
                        status_code=409,
                        detail={"error": "managed_replay_claim_lost"},
                    )
                return _managed_state(
                    replay=None,
                    external_request_id=external_request_id,
                    storage_request_id=storage_request_id,
                    request_hash=request_hash,
                    authority=authority,
                    claim_token=claim_token,
                )
            mark_write_request_processing(endpoint, profile, request_id, now, request_payload=request_payload)
            return WriteReplayState(replay=None, request_id=request_id)

        if authority is not None:
            try:
                if conn is not None:
                    _insert_write_request_processing_conn(
                        conn,
                        endpoint=endpoint,
                        profile=profile,
                        storage_request_id=storage_request_id,
                        now=now,
                        request_payload=request_payload,
                        authority=authority,
                        external_request_id=external_request_id,
                        request_hash=request_hash,
                        claim_token=claim_token,
                    )
                else:
                    insert_write_request_processing(
                        endpoint,
                        profile,
                        storage_request_id,
                        now,
                        request_payload=request_payload,
                        binding_hash=authority.binding_hash,
                        session_id=authority.session_id,
                        external_request_id=external_request_id,
                        request_hash=request_hash,
                        claim_token=claim_token,
                    )
            except sqlite3.IntegrityError as exc:
                winner = (
                    _load_write_request_replay_conn(conn, endpoint, profile, storage_request_id)
                    if conn is not None
                    else load_write_request_replay(endpoint, profile, storage_request_id)
                )
                if winner and winner.get("request_hash") == request_hash:
                    raise HTTPException(
                        status_code=409,
                        detail={"error": "managed_replay_already_processing"},
                    ) from exc
                raise HTTPException(
                    status_code=409,
                    detail={"error": "request_payload_conflict"},
                ) from exc
            return _managed_state(
                replay=None,
                external_request_id=external_request_id,
                storage_request_id=storage_request_id,
                request_hash=request_hash,
                authority=authority,
                claim_token=claim_token,
            )
        insert_write_request_processing(endpoint, profile, request_id, now, request_payload=request_payload)
    return WriteReplayState(replay=None, request_id=request_id)


def commit_write_request(
    endpoint: str,
    request_id: str | None,
    response: dict,
    *,
    replay_state: WriteReplayState | None = None,
) -> dict:
    response.setdefault("persistence_state", "committed")
    if not request_id:
        return response

    profile = get_active_profile()
    now = _utc_now_iso()
    storage_request_id = replay_state.storage_request_id if replay_state else request_id
    with _timer("write_request_commit_latency_ms", endpoint):
        updated = commit_write_request_replay(
            endpoint,
            profile,
            storage_request_id,
            response,
            now,
            binding_hash=replay_state.binding_hash if replay_state else None,
            expected_claim_token=replay_state.claim_token if replay_state else None,
        )
    if replay_state and replay_state.binding_hash is not None and updated != 1:
        raise HTTPException(
            status_code=409,
            detail={"error": "managed_replay_claim_lost"},
        )
    _flush_metrics()
    return response


def commit_managed_write_requests_for_episode(
    endpoint: str,
    authority: ManagedReplayAuthority,
    response: dict,
) -> int:
    """Commit every accepted replay request for one terminal managed episode."""

    profile = get_active_profile()
    if authority.profile != profile:
        raise ValueError("managed replay profile does not match active profile")
    rows = load_managed_write_request_replays_for_episode(
        endpoint,
        profile,
        authority.binding_hash,
        authority.session_id,
        status="processing",
    )
    committed = 0
    for row in rows:
        external_request_id = str(row.get("external_request_id") or "")
        storage_request_id = str(row.get("request_id") or "")
        claim_token = str(row.get("claim_token") or "")
        request_hash = str(row.get("request_hash") or "")
        if not all((external_request_id, storage_request_id, claim_token, request_hash)):
            raise RuntimeError("managed replay authority is incomplete")
        replay_state = WriteReplayState(
            request_id=external_request_id,
            storage_request_id=storage_request_id,
            request_hash=request_hash,
            binding_hash=authority.binding_hash,
            session_id=authority.session_id,
            claim_token=claim_token,
        )
        payload = dict(response)
        payload["request_id"] = external_request_id
        commit_write_request(
            endpoint,
            external_request_id,
            payload,
            replay_state=replay_state,
        )
        committed += 1
    return committed


def _redacted_response_summary(response: dict | None) -> dict:
    if not isinstance(response, dict):
        return {}
    return {field: response[field] for field in STATUS_SUMMARY_FIELDS if field in response}


def get_write_request_status(
    endpoint: str,
    request_id: str | None,
    *,
    authority: ManagedReplayAuthority | None = None,
) -> dict:
    """Return redacted status for an idempotent write request in the active profile."""
    if endpoint not in STATUS_ENDPOINT_ALLOWLIST:
        raise ValueError("unsupported write request endpoint")
    request_id = normalize_optional_request_id(request_id)
    if request_id is None:
        raise ValueError("request_id is required")

    profile = get_active_profile()
    external_request_id = request_id
    storage_request_id = request_id
    if authority is not None:
        if authority.profile != profile:
            raise ValueError("managed replay profile does not match active profile")
        storage_request_id = _managed_storage_request_id(endpoint, external_request_id, authority)
    row = load_write_request_replay(endpoint, profile, storage_request_id)
    if row and authority is not None and (
        row.get("binding_hash") != authority.binding_hash
        or row.get("session_id") != authority.session_id
        or row.get("external_request_id") != external_request_id
    ):
        row = None
    if not row:
        return {
            "endpoint": endpoint,
            "request_id": external_request_id,
            "status": "unknown",
            "processing_state": "unknown",
        }

    status = str(row.get("status") or "unknown")
    response = row.get("response")
    payload = {
        "endpoint": endpoint,
        "request_id": external_request_id,
        "status": status,
        "processing_state": status,
        "updated_at": row.get("updated_at"),
        "attempt_count": int(row.get("attempt_count") or 0),
    }
    if row.get("last_error"):
        payload["error_class"] = row.get("last_error")
    summary = _redacted_response_summary(response)
    if summary:
        payload["summary"] = summary
    return payload


def abandon_write_request(
    endpoint: str,
    request_id: str | None,
    *,
    error_class: str = "unknown",
    replay_state: WriteReplayState | None = None,
) -> None:
    if not request_id:
        return

    if replay_state and replay_state.binding_hash is not None:
        fail_write_request(
            endpoint,
            request_id,
            {"status": "failed", "request_id": request_id},
            error_class=error_class,
            replay_state=replay_state,
        )
        return

    profile = get_active_profile()
    if _wd_metrics:
        _wd_metrics.record("write_durability_abandoned", 1.0, {"endpoint": endpoint, "error_class": error_class})
    with _timer("write_request_abandon_latency_ms", endpoint, {"error_class": error_class}):
        delete_processing_write_request(endpoint, profile, request_id)
    _flush_metrics()


def fail_write_request(
    endpoint: str,
    request_id: str | None,
    response: dict,
    *,
    error_class: str,
    replay_state: WriteReplayState | None = None,
) -> dict:
    payload = dict(response)
    payload.setdefault("status", "failed")
    payload.setdefault("persistence_state", "failed")
    payload.setdefault("error_class", error_class)
    if not request_id:
        return payload

    payload.setdefault("request_id", request_id)
    profile = get_active_profile()
    now = _utc_now_iso()
    storage_request_id = replay_state.storage_request_id if replay_state else request_id
    with _timer("write_request_fail_latency_ms", endpoint, {"error_class": error_class}):
        updated = fail_write_request_replay(
            endpoint,
            profile,
            storage_request_id,
            payload,
            now,
            error_class,
            binding_hash=replay_state.binding_hash if replay_state else None,
            expected_claim_token=replay_state.claim_token if replay_state else None,
        )
    if replay_state and replay_state.binding_hash is not None and updated != 1:
        raise HTTPException(
            status_code=409,
            detail={"error": "managed_replay_claim_lost"},
        )
    if _wd_metrics:
        _wd_metrics.record("write_request_failed_terminal", 1.0, {"endpoint": endpoint, "error_class": error_class})
    _flush_metrics()
    return payload
