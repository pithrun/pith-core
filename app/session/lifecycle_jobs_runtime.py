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
    LIFECYCLE_DRAIN_WALL_BUDGET_SECONDS,
    LIFECYCLE_JOB_LEASE_SECONDS,
    LIFECYCLE_JOB_MAX_ATTEMPTS,
    LIFECYCLE_JOB_RETRY_SECONDS,
)
from app.core.profile import get_active_profile
from app.storage.lifecycle_jobs import (
    VALID_SOURCES,
    claim_lifecycle_jobs,
    commit_lifecycle_job,
    defer_lifecycle_job,
    enqueue_lifecycle_job,
    fail_lifecycle_job,
    retry_lifecycle_job,
    summarize_lifecycle_jobs,
    summarize_lifecycle_jobs_by_source,
)

_ALL_SOURCES_KEY = "__all__"
_EXECUTORS: dict[str, concurrent.futures.ThreadPoolExecutor] = {}
_EXECUTOR_LOCK = threading.Lock()
_DRAIN_FUTURES: dict[str, concurrent.futures.Future] = {}
_DRAIN_SUBMITTED_AT: dict[str, float] = {}
_DRAIN_LOCK = threading.Lock()
_DRAIN_EXECUTION_LOCK = threading.Lock()
_LIFECYCLE_JOB_RUNNERS: dict[str, Callable[[dict[str, Any]], Any]] = {}
_RUNNER_LOCK = threading.Lock()
_SUPERVISOR_LOCK = threading.Lock()
_SUPERVISOR_STOP: threading.Event | None = None
_SUPERVISOR_THREAD: threading.Thread | None = None


class LifecycleJobDeferred(RuntimeError):
    """Raised when a claimed lifecycle job did no work and should wait."""

    def __init__(self, reason: str, *, retry_after_seconds: float | None = None) -> None:
        super().__init__(reason)
        self.retry_after_seconds = retry_after_seconds


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


def _clear_lifecycle_job_runners_for_tests() -> None:
    with _RUNNER_LOCK:
        _LIFECYCLE_JOB_RUNNERS.clear()


def shutdown_lifecycle_runtime(*, wait: bool = True) -> None:
    """Stop lifecycle supervisor and source-keyed drain executors."""
    stop_lifecycle_supervisor(wait=wait)
    _shutdown_lifecycle_executors_for_tests()


def _record_metric(metric: str, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
    try:
        from app.ops.metrics import metrics

        metrics.record(metric, value, labels or {})
        metrics.flush()
    except Exception:
        pass


def _pressure_backpressure_active() -> tuple[bool, str, str]:
    try:
        from app.ops.pressure_policy import foreground_pressure_mode, should_defer_background_maintenance
        from app.ops.pressure_state import build_pressure_state

        state = build_pressure_state(use_cache=True)
        mode = foreground_pressure_mode(state)
        if should_defer_background_maintenance(state):
            return True, mode, str(state.pressure_level)
        return False, mode, str(state.pressure_level)
    except Exception:
        return False, "unknown", "unknown"


def _record_pressure_defer(source: str | None, reason: str, mode: str, level: str) -> None:
    _record_metric(
        "lifecycle_drain_pressure_deferred_total",
        1.0,
        {"source": _source_key(source), "reason": reason, "mode": mode, "pressure_level": level},
    )


def _parse_iso_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    try:
        text = str(value)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except Exception:
        return None


def _age_seconds(value: Any, *, now: datetime) -> float | None:
    parsed = _parse_iso_datetime(value)
    if parsed is None:
        return None
    return max(0.0, (now - parsed).total_seconds())


def _should_override_pressure_for_starvation(source: str | None, threshold_seconds: float | None) -> bool:
    if threshold_seconds is None:
        return False
    threshold = max(0.0, float(threshold_seconds))
    profile = get_active_profile()
    now = datetime.now(UTC)
    stale_before = (now - timedelta(seconds=LIFECYCLE_JOB_LEASE_SECONDS)).isoformat()
    try:
        if source:
            summary = summarize_lifecycle_jobs_by_source(
                profile=profile,
                source=source,
                stale_before_iso=stale_before,
            )
        else:
            summary = summarize_lifecycle_jobs(profile=profile, stale_before_iso=stale_before)
    except Exception:
        return False
    queued_age = _age_seconds(summary.get("oldest_queued_updated_at"), now=now)
    running_age = (
        _age_seconds(summary.get("oldest_running_updated_at"), now=now)
        if int(summary.get("stale_running_count") or 0) > 0
        else None
    )
    starved = (queued_age is not None and queued_age >= threshold) or (
        running_age is not None and running_age >= threshold
    )
    if starved:
        _record_metric(
            "lifecycle_drain_pressure_starvation_override_total",
            1.0,
            {"source": _source_key(source), "threshold_seconds": str(int(threshold))},
        )
    return starved


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
    active_binding_snapshot: dict | None = None,
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
        "active_binding_snapshot": active_binding_snapshot,
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


def _load_turn_ingestion_diagnostic(raw_capture_ref: dict[str, Any] | None) -> dict[str, Any]:
    """Load canonical turn-learning status for lifecycle diagnostics."""
    from app.storage.turn_ingestion import _get_turn_ingestion_diagnostic_default_db

    return _get_turn_ingestion_diagnostic_default_db(raw_capture_ref)


def _build_lifecycle_commit_result(job: dict[str, Any], run_result: Any) -> dict[str, Any]:
    """Build a backward-compatible lifecycle result payload."""
    result: dict[str, Any] = (
        {"status": "ok", **run_result}
        if isinstance(run_result, dict)
        else {"status": "ok", "result": str(run_result)}
    )
    if job.get("source") != "conversation_turn":
        return result
    payload = job.get("payload") or {}
    raw_capture_ref = payload.get("raw_capture_ref")
    if raw_capture_ref:
        result["raw_capture_ref"] = {
            "session_id": raw_capture_ref.get("session_id"),
            "turn_id": raw_capture_ref.get("turn_id"),
            "source": raw_capture_ref.get("source", "conversation_turn"),
        }
    result.update(_load_turn_ingestion_diagnostic(raw_capture_ref))
    if isinstance(run_result, dict):
        for key in ("learning_status", "concepts_extracted", "skip_reason", "fallback_status", "sync_handled"):
            if key in run_result and run_result[key] is not None:
                result[key] = run_result[key]
    return result


def run_lifecycle_drain_once(
    *,
    run_job: Callable[[dict[str, Any]], Any],
    reason: str,
    limit: int = 1,
    source: str | None = None,
    pressure_starvation_seconds: float | None = None,
) -> dict[str, int | str]:
    """Claim and run lifecycle jobs until the queue is empty for this pass.

    Source-specific drain workers may be submitted concurrently, but the job
    bodies share the same SQLite writer. Serialize the drain body so one
    lifecycle worker cannot hold the DB lock while another lifecycle worker is
    trying to begin or replay a write request.
    """
    profile = get_active_profile()
    start_monotonic = time.monotonic()
    result = {"claimed": 0, "committed": 0, "retried": 0, "deferred": 0, "failed": 0, "reason": reason}
    max_jobs = max(0, int(limit or 0))
    with _DRAIN_EXECUTION_LOCK:
        while int(result["claimed"]) < max_jobs:
            pressure_defer, pressure_mode, pressure_level = _pressure_backpressure_active()
            if pressure_defer:
                if _should_override_pressure_for_starvation(source, pressure_starvation_seconds):
                    result["pressure_starvation_override"] = 1
                    result["pressure_mode"] = pressure_mode
                    result["pressure_level"] = pressure_level
                else:
                    result["pressure_deferred"] = 1
                    result["pressure_mode"] = pressure_mode
                    result["pressure_level"] = pressure_level
                    _record_pressure_defer(source, reason, pressure_mode, pressure_level)
                    break
            elapsed_seconds = time.monotonic() - start_monotonic
            if elapsed_seconds >= LIFECYCLE_DRAIN_WALL_BUDGET_SECONDS:
                result["wall_budget_exhausted"] = 1
                _record_metric(
                    "lifecycle_drain_wall_budget_exhausted",
                    1.0,
                    {"source": _source_key(source), "reason": reason},
                )
                break
            now = datetime.now(UTC)
            lease_owner = f"{os.getpid()}:{reason}"
            lease_expires_at = (now + timedelta(seconds=LIFECYCLE_JOB_LEASE_SECONDS)).isoformat()
            jobs = claim_lifecycle_jobs(
                profile=profile,
                lease_owner=lease_owner,
                lease_expires_at=lease_expires_at,
                limit=1,
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
                        result=_build_lifecycle_commit_result(job, run_result),
                    )
                    _record_metric("lifecycle_job_committed", 1.0, {"source": job["source"], "stage": job["stage"]})
                    result["committed"] += 1
                except LifecycleJobDeferred as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    retry_after = (
                        float(exc.retry_after_seconds)
                        if exc.retry_after_seconds is not None
                        else float(LIFECYCLE_JOB_RETRY_SECONDS)
                    )
                    retry_at = (datetime.now(UTC) + timedelta(seconds=max(0.0, retry_after))).isoformat()
                    defer_lifecycle_job(profile=profile, job_id=job["job_id"], error=error, next_retry_at=retry_at)
                    _record_metric("lifecycle_job_deferred", 1.0, {"source": job["source"], "stage": job["stage"]})
                    result["deferred"] += 1
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
                pressure_defer, pressure_mode, pressure_level = _pressure_backpressure_active()
                if pressure_defer:
                    if _should_override_pressure_for_starvation(source, pressure_starvation_seconds):
                        result["pressure_starvation_override"] = 1
                        result["pressure_mode"] = pressure_mode
                        result["pressure_level"] = pressure_level
                    else:
                        result["pressure_deferred"] = 1
                        result["pressure_mode"] = pressure_mode
                        result["pressure_level"] = pressure_level
                        _record_pressure_defer(source, reason, pressure_mode, pressure_level)
                        break
    return result


def submit_lifecycle_drain(
    *,
    run_job: Callable[[dict[str, Any]], Any],
    reason: str,
    limit: int = 1,
    source: str | None = None,
    pressure_starvation_seconds: float | None = None,
) -> bool:
    """Submit one drain task if no drain is already running."""
    source_key = _source_key(source)
    with _DRAIN_LOCK:
        now = time.monotonic()
        _record_stale_future_metrics(now)
        pressure_defer, pressure_mode, pressure_level = _pressure_backpressure_active()
        if pressure_defer:
            if not _should_override_pressure_for_starvation(source, pressure_starvation_seconds):
                _record_pressure_defer(source, reason, pressure_mode, pressure_level)
                return False
        if _has_conflicting_drain(source_key):
            _record_metric("lifecycle_drain_submit_blocked", 1.0, {"source": source_key})
            return False
        _DRAIN_FUTURES[source_key] = _get_executor(source_key).submit(
            run_lifecycle_drain_once,
            run_job=run_job,
            reason=reason,
            limit=limit,
            source=source,
            pressure_starvation_seconds=pressure_starvation_seconds,
        )
        _DRAIN_SUBMITTED_AT[source_key] = now
        return True


def register_lifecycle_job_runner(source: str, run_job: Callable[[dict[str, Any]], Any]) -> None:
    """Register a source-specific lifecycle job runner for supervisor drains."""
    if source not in VALID_SOURCES:
        raise ValueError(f"invalid lifecycle job source: {source}")
    with _RUNNER_LOCK:
        _LIFECYCLE_JOB_RUNNERS[source] = run_job


def run_lifecycle_supervisor_once(
    *,
    reason: str = "lifecycle_supervisor",
    batch_size: int = 5,
    max_wall_seconds: float = 10.0,
    starvation_seconds: float = 600.0,
) -> dict[str, Any]:
    """Drain registered lifecycle sources once, with bounded pressure-aware catch-up."""
    started = time.monotonic()
    result: dict[str, Any] = {
        "reason": reason,
        "sources": {},
        "claimed": 0,
        "committed": 0,
        "retried": 0,
        "deferred": 0,
        "failed": 0,
    }
    with _RUNNER_LOCK:
        runners = list(_LIFECYCLE_JOB_RUNNERS.items())
    pressure_defer, _, _ = _pressure_backpressure_active()
    per_source_limit = 1 if pressure_defer else max(1, int(batch_size or 1))
    if pressure_defer:
        runners.sort(key=lambda item: 0 if item[0] == "session_learn" else 1)
    for source, run_job in runners:
        if time.monotonic() - started >= max(0.0, float(max_wall_seconds)):
            result["wall_budget_exhausted"] = 1
            break
        source_result = run_lifecycle_drain_once(
            run_job=run_job,
            reason=reason,
            limit=per_source_limit,
            source=source,
            pressure_starvation_seconds=starvation_seconds,
        )
        result["sources"][source] = source_result
        for key in ("claimed", "committed", "retried", "deferred", "failed"):
            result[key] = int(result[key]) + int(source_result.get(key, 0) or 0)
    _record_metric("lifecycle_supervisor_pass_total", 1.0, {"reason": reason})
    if int(result["claimed"]) > 0:
        _record_metric("lifecycle_supervisor_claimed_total", float(result["claimed"]), {"reason": reason})
    return result


def _lifecycle_supervisor_loop(
    *,
    stop_event: threading.Event,
    interval_seconds: float,
    batch_size: int,
    max_wall_seconds: float,
    starvation_seconds: float,
) -> None:
    while not stop_event.is_set():
        try:
            run_lifecycle_supervisor_once(
                batch_size=batch_size,
                max_wall_seconds=max_wall_seconds,
                starvation_seconds=starvation_seconds,
            )
        except Exception as exc:
            _record_metric(
                "lifecycle_supervisor_error_total",
                1.0,
                {"error_class": type(exc).__name__},
            )
        stop_event.wait(max(1.0, float(interval_seconds)))


def start_lifecycle_supervisor(
    *,
    interval_seconds: float = 30.0,
    batch_size: int = 5,
    max_wall_seconds: float = 10.0,
    starvation_seconds: float = 600.0,
) -> bool:
    """Start a daemon lifecycle supervisor thread if it is not already running."""
    global _SUPERVISOR_STOP, _SUPERVISOR_THREAD
    with _SUPERVISOR_LOCK:
        if _SUPERVISOR_THREAD is not None and _SUPERVISOR_THREAD.is_alive():
            return False
        stop_event = threading.Event()
        thread = threading.Thread(
            target=_lifecycle_supervisor_loop,
            kwargs={
                "stop_event": stop_event,
                "interval_seconds": interval_seconds,
                "batch_size": max(1, int(batch_size or 1)),
                "max_wall_seconds": max(0.1, float(max_wall_seconds)),
                "starvation_seconds": max(0.0, float(starvation_seconds)),
            },
            name="pith-lifecycle-supervisor",
            daemon=True,
        )
        _SUPERVISOR_STOP = stop_event
        _SUPERVISOR_THREAD = thread
        thread.start()
        return True


def stop_lifecycle_supervisor(*, wait: bool = True) -> None:
    """Stop the lifecycle supervisor thread if it is running."""
    global _SUPERVISOR_STOP, _SUPERVISOR_THREAD
    with _SUPERVISOR_LOCK:
        stop_event = _SUPERVISOR_STOP
        thread = _SUPERVISOR_THREAD
        _SUPERVISOR_STOP = None
        _SUPERVISOR_THREAD = None
    if stop_event is not None:
        stop_event.set()
    if wait and thread is not None and thread.is_alive():
        thread.join(timeout=5)
