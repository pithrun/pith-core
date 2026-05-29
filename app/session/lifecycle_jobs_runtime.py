"""Runtime helpers for draining durable lifecycle jobs."""

from __future__ import annotations

import concurrent.futures
import os
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from app.core.config import (
    LIFECYCLE_DRAIN_STUCK_SECONDS,
    LIFECYCLE_JOB_LEASE_SECONDS,
    LIFECYCLE_JOB_MAX_ATTEMPTS,
    LIFECYCLE_JOB_RETRY_SECONDS,
)
from app.core.profile import get_active_profile
from app.storage.lifecycle_jobs import (
    claim_lifecycle_jobs,
    commit_lifecycle_job,
    enqueue_lifecycle_job,
    fail_lifecycle_job,
    retry_lifecycle_job,
)

_ALL_SOURCES_KEY = "__all__"
_EXECUTORS: dict[str, concurrent.futures.ThreadPoolExecutor] = {}
_EXECUTOR_LOCK = threading.Lock()
_DRAIN_FUTURES: dict[str, concurrent.futures.Future] = {}
_DRAIN_SUBMITTED_AT: dict[str, float] = {}
_DRAIN_LOCK = threading.Lock()
_DRAIN_EXECUTION_LOCK = threading.Lock()


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _source_key(source: str | None) -> str:
    return source or _ALL_SOURCES_KEY


def _get_executor(source_key: str) -> concurrent.futures.ThreadPoolExecutor:
    with _EXECUTOR_LOCK:
        executor = _EXECUTORS.get(source_key)
        if executor is None:
            safe_key = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in source_key)
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"lifecycle_jobs_{safe_key}",
            )
            _EXECUTORS[source_key] = executor
        return executor


def _is_active_future(source_key: str) -> bool:
    future = _DRAIN_FUTURES.get(source_key)
    return future is not None and not future.done()


def _has_conflicting_drain(source_key: str) -> bool:
    if source_key == _ALL_SOURCES_KEY:
        return any(future is not None and not future.done() for future in _DRAIN_FUTURES.values())
    return _is_active_future(source_key) or _is_active_future(_ALL_SOURCES_KEY)


def _record_stale_future_metrics(now: float) -> None:
    for source_key, future in list(_DRAIN_FUTURES.items()):
        if future is None or future.done():
            continue
        submitted_at = _DRAIN_SUBMITTED_AT.get(source_key)
        if submitted_at is None:
            continue
        age_seconds = max(0.0, now - submitted_at)
        if age_seconds >= LIFECYCLE_DRAIN_STUCK_SECONDS:
            _record_metric("lifecycle_drain_future_stale", 1.0, {"source": source_key})
            _record_metric("lifecycle_drain_future_age_seconds", age_seconds, {"source": source_key})


def _shutdown_lifecycle_executors_for_tests() -> None:
    """Reset source-keyed executors for tests."""
    global _EXECUTORS, _DRAIN_FUTURES, _DRAIN_SUBMITTED_AT
    with _DRAIN_LOCK:
        futures = list(_DRAIN_FUTURES.values())
        executors = list(_EXECUTORS.values())
        _DRAIN_FUTURES = {}
        _DRAIN_SUBMITTED_AT = {}
        _EXECUTORS = {}
    for future in futures:
        future.cancel()
    for executor in executors:
        executor.shutdown(wait=False, cancel_futures=True)


def _record_metric(metric: str, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
    try:
        from app.ops.metrics import metrics

        metrics.record(metric, value, labels or {})
        metrics.flush()
    except Exception:
        pass


def enqueue_conversation_autolearn_job(
    *,
    learn_request: Any,
    extracted: list | None,
    request_message: str,
    prev_msg: str,
    prev_response: str,
    bound_session: Any | None,
    raw_capture_ref: dict | None,
    idempotency_key: str,
    priority: int = 50,
) -> dict[str, Any]:
    """Persist a conversation-turn autolearn job without exposing payload in logs."""
    payload = {
        "learn_request": learn_request.model_dump(mode="json")
        if hasattr(learn_request, "model_dump")
        else dict(learn_request),
        "extracted": extracted,
        "request_message": request_message,
        "prev_msg": prev_msg,
        "prev_response": prev_response,
        "bound_session": bound_session.model_dump(mode="json")
        if hasattr(bound_session, "model_dump")
        else bound_session,
        "raw_capture_ref": raw_capture_ref,
    }
    job = enqueue_lifecycle_job(
        profile=get_active_profile(),
        source="conversation_turn",
        idempotency_key=idempotency_key,
        stage="learn",
        payload=payload,
        priority=priority,
    )
    _record_metric("lifecycle_job_enqueued", 1.0, {"source": "conversation_turn", "stage": "learn"})
    return job


def enqueue_session_learn_job(
    *,
    learn_request: Any,
    request_id: str,
    priority: int = 60,
) -> dict[str, Any]:
    """Persist an explicit session_learn job for durable post-response processing."""
    payload = {
        "learn_request": learn_request.model_dump(mode="json")
        if hasattr(learn_request, "model_dump")
        else dict(learn_request),
    }
    job = enqueue_lifecycle_job(
        profile=get_active_profile(),
        source="session_learn",
        idempotency_key=request_id,
        stage="learn",
        payload=payload,
        priority=priority,
    )
    _record_metric("lifecycle_job_enqueued", 1.0, {"source": "session_learn", "stage": "learn"})
    _record_metric("session_learn_lifecycle_enqueued", 1.0)
    return job


def run_lifecycle_drain_once(
    *,
    run_job: Callable[[dict[str, Any]], Any],
    reason: str,
    limit: int = 1,
    source: str | None = None,
) -> dict[str, int | str]:
    """Claim and run lifecycle jobs until the queue is empty for this pass.

    Source-specific drain workers may be submitted concurrently, but the job
    bodies share the same SQLite writer. Serialize the drain body so one
    lifecycle worker cannot hold the DB lock while another lifecycle worker is
    trying to begin or replay a write request.
    """
    profile = get_active_profile()
    result = {"claimed": 0, "committed": 0, "retried": 0, "failed": 0, "reason": reason}
    with _DRAIN_EXECUTION_LOCK:
        while True:
            now = datetime.now(UTC)
            lease_owner = f"{os.getpid()}:{reason}"
            lease_expires_at = (now + timedelta(seconds=LIFECYCLE_JOB_LEASE_SECONDS)).isoformat()
            jobs = claim_lifecycle_jobs(
                profile=profile,
                lease_owner=lease_owner,
                lease_expires_at=lease_expires_at,
                limit=limit,
                now=now.isoformat(),
                max_attempts=LIFECYCLE_JOB_MAX_ATTEMPTS,
                source=source,
            )
            if not jobs:
                break
            result["claimed"] += len(jobs)
            for job in jobs:
                _record_metric("lifecycle_job_claimed", 1.0, {"source": job["source"], "stage": job["stage"]})
                try:
                    run_result = run_job(job)
                    commit_lifecycle_job(
                        profile=profile,
                        job_id=job["job_id"],
                        result={"status": "ok", "result": str(run_result)},
                    )
                    _record_metric("lifecycle_job_committed", 1.0, {"source": job["source"], "stage": job["stage"]})
                    result["committed"] += 1
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    if int(job.get("attempts") or 0) >= LIFECYCLE_JOB_MAX_ATTEMPTS:
                        fail_lifecycle_job(profile=profile, job_id=job["job_id"], error=error)
                        _record_metric("lifecycle_job_failed", 1.0, {"source": job["source"], "stage": job["stage"]})
                        result["failed"] += 1
                    else:
                        retry_at = (datetime.now(UTC) + timedelta(seconds=LIFECYCLE_JOB_RETRY_SECONDS)).isoformat()
                        retry_lifecycle_job(profile=profile, job_id=job["job_id"], error=error, next_retry_at=retry_at)
                        _record_metric("lifecycle_job_retry", 1.0, {"source": job["source"], "stage": job["stage"]})
                        result["retried"] += 1
    return result


def submit_lifecycle_drain(
    *,
    run_job: Callable[[dict[str, Any]], Any],
    reason: str,
    limit: int = 1,
    source: str | None = None,
) -> bool:
    """Submit one drain task if no drain is already running."""
    source_key = _source_key(source)
    with _DRAIN_LOCK:
        now = time.monotonic()
        _record_stale_future_metrics(now)
        if _has_conflicting_drain(source_key):
            _record_metric("lifecycle_drain_submit_blocked", 1.0, {"source": source_key})
            return False
        _DRAIN_FUTURES[source_key] = _get_executor(source_key).submit(
            run_lifecycle_drain_once,
            run_job=run_job,
            reason=reason,
            limit=limit,
            source=source,
        )
        _DRAIN_SUBMITTED_AT[source_key] = now
        return True
