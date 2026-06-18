"""Shared execution contract for reflection runs.

Reflection can be invoked by public API calls and scheduled maintenance. This
runner centralizes admission, process/profile locking, deadline setup, and
structured result mapping so callers do not execute full reflection directly.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from app.cognitive.reflection import reflection_engine
from app.core.datetime_utils import _utc_now_iso
from app.core.fork_safety import should_suppress_optional_subprocess
from app.core.profile import resolve_data_dir
from app.ops.metrics import metrics

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_FULL_TIMEOUT_SECONDS = 420
_DEFAULT_RETRY_AFTER_SECONDS = 30
_DEFAULT_MIN_FREE_BYTES = 1024 * 1024 * 1024
_REFLECTION_FULL_SOURCE = "reflection_full"
_REFLECTION_FULL_STAGE = "reflect"


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        raw = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        raw = default
    return max(minimum, min(maximum, raw))


def _full_timeout_seconds() -> int:
    return _env_int(
        "PITH_REFLECTION_API_FULL_TIMEOUT_SECONDS",
        _DEFAULT_FULL_TIMEOUT_SECONDS,
        minimum=30,
        maximum=600,
    )


def _retry_after_seconds() -> int:
    return _env_int(
        "PITH_REFLECTION_RETRY_AFTER_SECONDS",
        _DEFAULT_RETRY_AFTER_SECONDS,
        minimum=1,
        maximum=3600,
    )


def _min_free_bytes() -> int:
    return _env_int(
        "PITH_REFLECTION_MIN_FREE_BYTES",
        _DEFAULT_MIN_FREE_BYTES,
        minimum=0,
        maximum=10 * 1024 * 1024 * 1024,
    )


def _background_full_enabled() -> bool:
    raw = os.environ.get("PITH_REFLECTION_FULL_BACKGROUND_ENABLED", "1")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _safety_margin_seconds(timeout_seconds: int | float) -> float:
    return min(30.0, max(5.0, float(timeout_seconds) * 0.20))


def _long_step_min_budget_seconds(timeout_seconds: int | float) -> float:
    return min(45.0, max(10.0, float(timeout_seconds) * 0.20))


@dataclass(frozen=True)
class ReflectionRunRequest:
    mode: str = "incremental"
    source: str = "api"
    verbose: bool = False
    require_ready: bool = False
    allow_degraded: bool = False
    timeout_seconds: int | None = None
    ready_state: dict[str, Any] | None = None
    reflection_engine_override: Any | None = field(default=None, repr=False, compare=False)
    cancel_event: threading.Event | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class ReflectionAdmission:
    accepted: bool
    reason: str | None = None
    http_status: int = 200
    retry_after_seconds: int | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReflectionRunResult:
    run_id: str
    mode: str
    source: str
    status: str
    started_at: str
    completed_at: str
    duration_ms: float
    summary: Any | None = None
    admission: ReflectionAdmission | None = None
    error: str | None = None
    active_snapshot: dict[str, Any] | None = None

    @property
    def response_fields(self) -> dict[str, Any]:
        response = {
            "run_id": self.run_id,
            "mode": self.mode,
            "source": self.source,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms,
        }
        if self.admission is not None and not self.admission.accepted:
            response["reason"] = self.admission.reason
            response["details"] = self.admission.details
            if self.admission.retry_after_seconds is not None:
                response["retry_after_seconds"] = self.admission.retry_after_seconds
        if self.error:
            response["error"] = self.error
        if self.active_snapshot:
            response["active_reflection"] = self.active_snapshot
        return response


class ReflectionRunner:
    def __init__(self) -> None:
        self._thread_lock = threading.Lock()
        self._active_lock = threading.Lock()
        self._active_run: dict[str, Any] | None = None

    def get_active_run(self) -> dict[str, Any] | None:
        with self._active_lock:
            return dict(self._active_run) if self._active_run else None

    def run(self, request: ReflectionRunRequest) -> ReflectionRunResult:
        run_id = str(uuid.uuid4())
        started_at = _utc_now_iso()
        started_monotonic = time.monotonic()
        admission = self._admit(request)
        if not admission.accepted:
            return self._finish_rejected(run_id, request, started_at, started_monotonic, admission)

        if not self._thread_lock.acquire(blocking=False):
            admission = ReflectionAdmission(
                accepted=False,
                reason="reflection_already_running",
                http_status=409,
                retry_after_seconds=_retry_after_seconds(),
            )
            return self._finish_rejected(run_id, request, started_at, started_monotonic, admission)

        try:
            with self._profile_file_lock() as lock_admission:
                if lock_admission is not None:
                    return self._finish_rejected(run_id, request, started_at, started_monotonic, lock_admission)
                active_snapshot = self._set_active_run(run_id, request, started_at)
                try:
                    summary = self._execute(request)
                    status = self._status_from_summary(summary)
                    self._record_metric("reflection_run_completed", request, status=status)
                    return self._finish_success(
                        run_id,
                        request,
                        started_at,
                        started_monotonic,
                        status=status,
                        summary=summary,
                        active_snapshot=active_snapshot,
                    )
                except Exception as exc:
                    logger.exception("Reflection run failed: run_id=%s mode=%s source=%s", run_id, request.mode, request.source)
                    self._record_metric("reflection_run_failed", request, status="failed")
                    return self._finish_failed(run_id, request, started_at, started_monotonic, exc, active_snapshot)
                finally:
                    self._clear_active_run(run_id)
        finally:
            self._thread_lock.release()

    def submit_background(self, request: ReflectionRunRequest) -> ReflectionRunResult:
        """Submit full reflection to a separate worker process and return quickly."""
        if request.mode != "full" or not _background_full_enabled():
            return self.run(request)

        if _durable_reflection_jobs_enabled():
            return enqueue_and_submit_reflection_job(request)

        run_id = str(uuid.uuid4())
        started_at = _utc_now_iso()
        started_monotonic = time.monotonic()
        admission = self._admit(request)
        if not admission.accepted:
            if admission.http_status == 202:
                return self._finish_deferred(run_id, request, started_at, started_monotonic, admission)
            return self._finish_rejected(run_id, request, started_at, started_monotonic, admission)
        pressure_admission = self._check_pressure_budget()
        if not pressure_admission.accepted:
            return self._finish_deferred(run_id, request, started_at, started_monotonic, pressure_admission)
        fork_safety_admission = self._check_worker_fork_safety()
        if not fork_safety_admission.accepted:
            return self._finish_deferred(run_id, request, started_at, started_monotonic, fork_safety_admission)

        if not self._thread_lock.acquire(blocking=False):
            admission = ReflectionAdmission(
                accepted=False,
                reason="reflection_already_running",
                http_status=409,
                retry_after_seconds=_retry_after_seconds(),
            )
            return self._finish_rejected(run_id, request, started_at, started_monotonic, admission)

        active_snapshot = self._set_active_run(run_id, request, started_at)
        state_dir = Path(resolve_data_dir()) / "reflection_runs"
        state_dir.mkdir(parents=True, exist_ok=True)
        output_path = state_dir / f"{run_id}.json"
        timeout_seconds = self._timeout_for_request(request)
        command = [
            sys.executable,
            "-m",
            "app.ops.reflection_worker",
            "--mode",
            request.mode,
            "--source",
            request.source,
            "--timeout-seconds",
            str(timeout_seconds),
            "--output",
            str(output_path),
        ]
        env = os.environ.copy()
        env["PITH_REFLECTION_WORKER_RUN_ID"] = run_id
        try:
            process = subprocess.Popen(
                command,
                cwd=str(_REPO_ROOT),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception as exc:
            self._clear_active_run(run_id)
            self._thread_lock.release()
            logger.exception("Reflection worker launch failed: run_id=%s", run_id)
            self._record_metric("reflection_run_failed", request, status="worker_launch_failed")
            return self._finish_failed(run_id, request, started_at, started_monotonic, exc, active_snapshot)

        active_snapshot = self._update_active_run(
            run_id,
            {
                "pid": process.pid,
                "runner_kind": "subprocess",
                "output_path": str(output_path),
                "status": "running",
            },
        )
        monitor = threading.Thread(
            target=self._monitor_background_process,
            args=(run_id, request, process, output_path),
            name=f"reflection-worker-{run_id[:8]}",
            daemon=True,
        )
        monitor.start()
        self._record_metric("reflection_run_submitted", request, status="accepted")
        return ReflectionRunResult(
            run_id=run_id,
            mode=request.mode,
            source=request.source,
            status="accepted",
            started_at=started_at,
            completed_at=_utc_now_iso(),
            duration_ms=round((time.monotonic() - started_monotonic) * 1000, 2),
            active_snapshot=active_snapshot,
        )

    def run_worker_blocking(self, request: ReflectionRunRequest, *, run_id: str | None = None) -> ReflectionRunResult:
        """Run full reflection in a worker process and wait for durable completion."""
        if request.mode != "full" or not _background_full_enabled():
            return self.run(request)

        selected_run_id = run_id or str(uuid.uuid4())
        started_at = _utc_now_iso()
        started_monotonic = time.monotonic()
        admission = self._admit(request)
        if not admission.accepted:
            if admission.http_status == 202:
                return self._finish_deferred(selected_run_id, request, started_at, started_monotonic, admission)
            return self._finish_rejected(selected_run_id, request, started_at, started_monotonic, admission)
        pressure_admission = self._check_pressure_budget()
        if not pressure_admission.accepted:
            return self._finish_deferred(selected_run_id, request, started_at, started_monotonic, pressure_admission)
        fork_safety_admission = self._check_worker_fork_safety()
        if not fork_safety_admission.accepted:
            return self._finish_deferred(
                selected_run_id,
                request,
                started_at,
                started_monotonic,
                fork_safety_admission,
            )

        if not self._thread_lock.acquire(blocking=False):
            admission = ReflectionAdmission(
                accepted=False,
                reason="reflection_already_running",
                http_status=409,
                retry_after_seconds=_retry_after_seconds(),
            )
            return self._finish_rejected(selected_run_id, request, started_at, started_monotonic, admission)

        active_snapshot = self._set_active_run(selected_run_id, request, started_at)
        state_dir = Path(resolve_data_dir()) / "reflection_runs"
        state_dir.mkdir(parents=True, exist_ok=True)
        output_path = state_dir / f"{selected_run_id}.json"
        timeout_seconds = self._timeout_for_request(request)
        command = [
            sys.executable,
            "-m",
            "app.ops.reflection_worker",
            "--mode",
            request.mode,
            "--source",
            request.source,
            "--timeout-seconds",
            str(timeout_seconds),
            "--output",
            str(output_path),
        ]
        env = os.environ.copy()
        env["PITH_REFLECTION_WORKER_RUN_ID"] = selected_run_id
        process: subprocess.Popen | None = None
        try:
            process = subprocess.Popen(
                command,
                cwd=str(_REPO_ROOT),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            active_snapshot = self._update_active_run(
                selected_run_id,
                {
                    "pid": process.pid,
                    "runner_kind": "subprocess",
                    "output_path": str(output_path),
                    "status": "running",
                },
            )
            self._record_metric("reflection_run_submitted", request, status="durable")
            try:
                exit_code = process.wait(timeout=timeout_seconds + 30)
            except TypeError:
                exit_code = process.wait()
            except subprocess.TimeoutExpired as exc:
                with contextlib.suppress(Exception):
                    process.kill()
                with contextlib.suppress(Exception):
                    process.wait(timeout=10)
                raise TimeoutError(f"reflection_worker_timeout_after_{timeout_seconds}s") from exc

            payload: dict[str, Any] = {}
            if output_path.exists():
                try:
                    payload = json.loads(output_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    payload = {"status": "failed", "error": f"worker_output_invalid: {exc}"}
            status = str(payload.get("status") or ("completed" if exit_code == 0 else "failed"))
            if exit_code != 0 and status == "completed":
                status = "failed"
            self._update_active_run(
                selected_run_id,
                {
                    "status": status,
                    "completed_at": _utc_now_iso(),
                    "exit_code": exit_code,
                    "worker_status": payload.get("status"),
                },
            )
            self._record_metric("reflection_run_worker_completed", request, status=status)
            return ReflectionRunResult(
                run_id=selected_run_id,
                mode=request.mode,
                source=request.source,
                status=status,
                started_at=started_at,
                completed_at=_utc_now_iso(),
                duration_ms=round((time.monotonic() - started_monotonic) * 1000, 2),
                summary=payload.get("summary"),
                error=payload.get("error"),
                active_snapshot=active_snapshot,
            )
        except Exception as exc:
            logger.exception("Durable reflection worker failed: run_id=%s", selected_run_id)
            self._record_metric("reflection_run_failed", request, status="durable_worker_failed")
            return self._finish_failed(selected_run_id, request, started_at, started_monotonic, exc, active_snapshot)
        finally:
            self._clear_active_run(selected_run_id)
            self._thread_lock.release()

    async def run_async(self, request: ReflectionRunRequest) -> ReflectionRunResult:
        loop = asyncio.get_running_loop()
        cancel_event = request.cancel_event
        if request.mode == "full" and cancel_event is None:
            cancel_event = threading.Event()
            request = replace(request, cancel_event=cancel_event)
        future = loop.run_in_executor(None, self.run, request)
        try:
            return await future
        except asyncio.CancelledError:
            if cancel_event is not None:
                cancel_event.set()
            raise

    def _admit(self, request: ReflectionRunRequest) -> ReflectionAdmission:
        if request.mode not in {"incremental", "full"}:
            return ReflectionAdmission(
                accepted=False,
                reason="invalid_mode",
                http_status=400,
                details={"allowed_modes": ["incremental", "full"], "mode": request.mode},
            )
        if request.mode == "full":
            disk_admission = self._check_disk_budget()
            if not disk_admission.accepted:
                return disk_admission
        if request.require_ready:
            ready = request.ready_state or {}
            process_state = ready.get("process_state")
            retrieval_state = ready.get("retrieval_state")
            if process_state != "running":
                return ReflectionAdmission(
                    accepted=False,
                    reason="process_not_running",
                    http_status=503,
                    retry_after_seconds=_retry_after_seconds(),
                    details={"process_state": process_state},
                )
            if retrieval_state != "ready":
                return ReflectionAdmission(
                    accepted=False,
                    reason="retrieval_not_ready",
                    http_status=503,
                    retry_after_seconds=_retry_after_seconds(),
                    details={"retrieval_state": retrieval_state},
                )
            if ready.get("git_commit_matches_head") is False:
                return ReflectionAdmission(
                    accepted=False,
                    reason="runtime_git_mismatch",
                    http_status=503,
                    retry_after_seconds=_retry_after_seconds(),
                    details={
                        "git_commit": ready.get("git_commit"),
                        "git_head_commit": ready.get("git_head_commit"),
                    },
                )
            degraded_reason = ready.get("degraded_reason")
            if degraded_reason and not request.allow_degraded:
                return ReflectionAdmission(
                    accepted=False,
                    reason="runtime_degraded",
                    http_status=503,
                    retry_after_seconds=_retry_after_seconds(),
                    details={"degraded_reason": degraded_reason},
                )
        return ReflectionAdmission(accepted=True)

    def _check_pressure_budget(self) -> ReflectionAdmission:
        try:
            from app.ops.pressure_policy import foreground_pressure_mode
            from app.ops.pressure_state import build_pressure_state

            pressure_state = build_pressure_state(active_reflection=self.get_active_run(), use_cache=True)
            mode = foreground_pressure_mode(pressure_state)
        except Exception as exc:
            logger.debug("Reflection pressure check unavailable: %s", exc)
            return ReflectionAdmission(accepted=True)
        if mode not in {"protected", "critical"}:
            return ReflectionAdmission(accepted=True)
        return ReflectionAdmission(
            accepted=False,
            reason="pressure_protected_reflection_deferred",
            http_status=202,
            retry_after_seconds=_retry_after_seconds(),
            details={"pressure_mode": mode},
        )

    def _check_worker_fork_safety(self) -> ReflectionAdmission:
        if not should_suppress_optional_subprocess("reflection_worker"):
            return ReflectionAdmission(accepted=True)
        return ReflectionAdmission(
            accepted=False,
            reason="reflection_worker_subprocess_suppressed_fork_safety",
            http_status=202,
            retry_after_seconds=_retry_after_seconds(),
            details={"runner_kind": "subprocess"},
        )

    def _check_disk_budget(self) -> ReflectionAdmission:
        data_dir = Path(resolve_data_dir())
        min_free = _min_free_bytes()
        try:
            free = shutil.disk_usage(data_dir).free
        except Exception as exc:
            return ReflectionAdmission(
                accepted=False,
                reason="disk_budget_unavailable",
                http_status=503,
                retry_after_seconds=_retry_after_seconds(),
                details={"error": str(exc)[:200]},
            )
        if free < min_free:
            return ReflectionAdmission(
                accepted=False,
                reason="low_disk",
                http_status=503,
                retry_after_seconds=_retry_after_seconds(),
                details={"free_bytes": free, "min_free_bytes": min_free, "data_dir": str(data_dir)},
            )
        return ReflectionAdmission(accepted=True)

    @contextlib.contextmanager
    def _profile_file_lock(self):
        lock_dir = Path(resolve_data_dir()) / "locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = lock_dir / "reflection.lock"
        handle = lock_path.open("a+", encoding="utf-8")
        try:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                yield ReflectionAdmission(
                    accepted=False,
                    reason="reflection_already_running",
                    http_status=409,
                    retry_after_seconds=_retry_after_seconds(),
                    details={"lock_path": str(lock_path)},
                )
                return
            yield None
        finally:
            with contextlib.suppress(Exception):
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    def _set_active_run(self, run_id: str, request: ReflectionRunRequest, started_at: str) -> dict[str, Any]:
        snapshot = {
            "run_id": run_id,
            "mode": request.mode,
            "source": request.source,
            "started_at": started_at,
            "timeout_seconds": self._timeout_for_request(request),
        }
        with self._active_lock:
            self._active_run = dict(snapshot)
        return snapshot

    def _clear_active_run(self, run_id: str) -> None:
        with self._active_lock:
            if self._active_run and self._active_run.get("run_id") == run_id:
                self._active_run = None

    def _update_active_run(self, run_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        with self._active_lock:
            if self._active_run and self._active_run.get("run_id") == run_id:
                self._active_run.update(updates)
                return dict(self._active_run)
            return dict(updates)

    def _monitor_background_process(
        self,
        run_id: str,
        request: ReflectionRunRequest,
        process: subprocess.Popen,
        output_path: Path,
    ) -> None:
        status = "completed"
        try:
            exit_code = process.wait()
            payload: dict[str, Any] = {}
            if output_path.exists():
                try:
                    payload = json.loads(output_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    payload = {"status": "failed", "error": f"worker_output_invalid: {exc}"}
            if exit_code != 0:
                status = "failed"
            else:
                status = str(payload.get("status") or "completed")
            self._update_active_run(
                run_id,
                {
                    "status": status,
                    "completed_at": _utc_now_iso(),
                    "exit_code": exit_code,
                    "worker_status": payload.get("status"),
                },
            )
            self._record_metric("reflection_run_worker_completed", request, status=status)
        except Exception:
            logger.exception("Reflection worker monitor failed: run_id=%s", run_id)
            self._record_metric("reflection_run_worker_failed", request, status="monitor_failed")
        finally:
            self._clear_active_run(run_id)
            self._thread_lock.release()

    def _execute(self, request: ReflectionRunRequest):
        engine = request.reflection_engine_override or reflection_engine
        if request.mode != "full":
            return engine.reflect(request.mode)
        timeout_seconds = self._timeout_for_request(request)
        cancel_event = request.cancel_event or threading.Event()
        deadline_monotonic = time.monotonic() + max(1.0, timeout_seconds - _safety_margin_seconds(timeout_seconds))
        return engine.reflect(
            mode="full",
            cancel_event=cancel_event,
            deadline_monotonic=deadline_monotonic,
            long_step_min_remaining_seconds=_long_step_min_budget_seconds(timeout_seconds),
        )

    def _timeout_for_request(self, request: ReflectionRunRequest) -> int:
        if request.timeout_seconds is not None:
            return max(30, min(600, int(request.timeout_seconds)))
        return _full_timeout_seconds()

    def _status_from_summary(self, summary: Any | None) -> str:
        if summary is None:
            return "skipped"
        if getattr(summary, "aborted", False):
            return "deferred" if getattr(summary, "budget_status", None) == "deferred" else "aborted"
        return "completed"

    def _finish_rejected(
        self,
        run_id: str,
        request: ReflectionRunRequest,
        started_at: str,
        started_monotonic: float,
        admission: ReflectionAdmission,
    ) -> ReflectionRunResult:
        self._record_metric("reflection_run_rejected", request, status=admission.reason or "rejected")
        return ReflectionRunResult(
            run_id=run_id,
            mode=request.mode,
            source=request.source,
            status="rejected",
            started_at=started_at,
            completed_at=_utc_now_iso(),
            duration_ms=round((time.monotonic() - started_monotonic) * 1000, 2),
            admission=admission,
        )

    def _finish_deferred(
        self,
        run_id: str,
        request: ReflectionRunRequest,
        started_at: str,
        started_monotonic: float,
        admission: ReflectionAdmission,
    ) -> ReflectionRunResult:
        self._record_metric("reflection_run_deferred", request, status=admission.reason or "deferred")
        return ReflectionRunResult(
            run_id=run_id,
            mode=request.mode,
            source=request.source,
            status="deferred",
            started_at=started_at,
            completed_at=_utc_now_iso(),
            duration_ms=round((time.monotonic() - started_monotonic) * 1000, 2),
            admission=admission,
        )

    def _finish_success(
        self,
        run_id: str,
        request: ReflectionRunRequest,
        started_at: str,
        started_monotonic: float,
        *,
        status: str,
        summary: Any | None,
        active_snapshot: dict[str, Any],
    ) -> ReflectionRunResult:
        return ReflectionRunResult(
            run_id=run_id,
            mode=request.mode,
            source=request.source,
            status=status,
            started_at=started_at,
            completed_at=_utc_now_iso(),
            duration_ms=round((time.monotonic() - started_monotonic) * 1000, 2),
            summary=summary,
            active_snapshot=active_snapshot,
        )

    def _finish_failed(
        self,
        run_id: str,
        request: ReflectionRunRequest,
        started_at: str,
        started_monotonic: float,
        exc: Exception,
        active_snapshot: dict[str, Any],
    ) -> ReflectionRunResult:
        return ReflectionRunResult(
            run_id=run_id,
            mode=request.mode,
            source=request.source,
            status="failed",
            started_at=started_at,
            completed_at=_utc_now_iso(),
            duration_ms=round((time.monotonic() - started_monotonic) * 1000, 2),
            error=str(exc)[:500],
            active_snapshot=active_snapshot,
        )

    def _record_metric(self, name: str, request: ReflectionRunRequest, *, status: str) -> None:
        try:
            metrics.record(name, 1.0, {"mode": request.mode, "source": request.source, "status": status})
        except Exception:
            logger.debug("Reflection metric record failed: %s", name, exc_info=True)


reflection_runner = ReflectionRunner()


def _durable_reflection_jobs_enabled() -> bool:
    try:
        from app.core.config import REFLECTION_DURABLE_JOBS_ENABLED

        return bool(REFLECTION_DURABLE_JOBS_ENABLED)
    except Exception:
        return False


def _reflection_payload(request: ReflectionRunRequest) -> dict[str, Any]:
    return {
        "mode": request.mode,
        "source": request.source,
        "verbose": bool(request.verbose),
        "timeout_seconds": request.timeout_seconds,
    }


def enqueue_reflection_full_job(request: ReflectionRunRequest) -> dict[str, Any]:
    """Enqueue or reuse one open durable full-reflection lifecycle job."""
    from app.core.profile import get_active_profile
    from app.storage.lifecycle_jobs import enqueue_lifecycle_job, load_open_lifecycle_job

    profile = get_active_profile()
    open_job = load_open_lifecycle_job(
        profile=profile,
        source=_REFLECTION_FULL_SOURCE,
        stage=_REFLECTION_FULL_STAGE,
    )
    if open_job:
        open_job["reused_open_job"] = True
        return open_job

    job = enqueue_lifecycle_job(
        profile=profile,
        source=_REFLECTION_FULL_SOURCE,
        idempotency_key=f"reflection_full:{uuid.uuid4().hex}",
        stage=_REFLECTION_FULL_STAGE,
        payload=_reflection_payload(request),
        priority=90,
    )
    job["reused_open_job"] = False
    try:
        metrics.record("reflection_lifecycle_job_enqueued", 1.0, {"source": request.source})
    except Exception:
        logger.debug("Reflection lifecycle enqueue metric failed", exc_info=True)
    return job


def run_reflection_lifecycle_job(job: dict[str, Any]) -> dict[str, Any]:
    """Execute a claimed durable full-reflection lifecycle job."""
    payload = job.get("payload") or {}
    request = ReflectionRunRequest(
        mode=str(payload.get("mode") or "full"),
        source=str(payload.get("source") or "lifecycle_job"),
        verbose=bool(payload.get("verbose") or False),
        require_ready=False,
        allow_degraded=True,
        timeout_seconds=payload.get("timeout_seconds"),
    )
    result = reflection_runner.run_worker_blocking(request, run_id=str(job.get("job_id") or uuid.uuid4()))
    response = result.response_fields
    if result.summary is not None:
        response["summary"] = result.summary
    if result.status == "deferred":
        from app.session.lifecycle_jobs_runtime import LifecycleJobDeferred

        reason = result.admission.reason if result.admission else result.error or "reflection_deferred"
        retry_after = result.admission.retry_after_seconds if result.admission else None
        raise LifecycleJobDeferred(str(reason), retry_after_seconds=retry_after)
    if result.status in {"failed", "rejected"}:
        reason = result.error or (result.admission.reason if result.admission else None) or result.status
        raise RuntimeError(reason)
    if result.status != "completed":
        raise RuntimeError(f"reflection_worker_not_completed: {result.status}")
    return response


def enqueue_and_submit_reflection_job(request: ReflectionRunRequest) -> ReflectionRunResult:
    """Persist a full-reflection request and opportunistically start a drain."""
    started_at = _utc_now_iso()
    started_monotonic = time.monotonic()
    job = enqueue_reflection_full_job(request)
    submitted = False
    try:
        from app.core.config import LIFECYCLE_SUPERVISOR_STARVATION_SECONDS
        from app.session.lifecycle_jobs_runtime import submit_lifecycle_drain

        submitted = submit_lifecycle_drain(
            run_job=run_reflection_lifecycle_job,
            reason="reflection_api",
            limit=1,
            source=_REFLECTION_FULL_SOURCE,
            pressure_starvation_seconds=LIFECYCLE_SUPERVISOR_STARVATION_SECONDS,
        )
    except Exception:
        logger.debug("Reflection lifecycle drain submission failed", exc_info=True)

    status = "accepted" if submitted else "queued"
    active_snapshot = {
        "job_id": job.get("job_id"),
        "job_status": job.get("status"),
        "runner_kind": "lifecycle_job",
        "drain_submitted": submitted,
        "reused_open_job": bool(job.get("reused_open_job")),
    }
    return ReflectionRunResult(
        run_id=str(job.get("job_id") or uuid.uuid4()),
        mode=request.mode,
        source=request.source,
        status=status,
        started_at=started_at,
        completed_at=_utc_now_iso(),
        duration_ms=round((time.monotonic() - started_monotonic) * 1000, 2),
        active_snapshot=active_snapshot,
    )


def run_reflection(request: ReflectionRunRequest) -> ReflectionRunResult:
    return reflection_runner.run(request)


async def run_reflection_async(request: ReflectionRunRequest) -> ReflectionRunResult:
    return await reflection_runner.run_async(request)


def submit_reflection_background(request: ReflectionRunRequest) -> ReflectionRunResult:
    return reflection_runner.submit_background(request)
