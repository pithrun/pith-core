"""FastAPI server exposing Pith MCP tools."""

import asyncio
import concurrent.futures
import copy
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent


# OPS-051: Capture git commit at startup for deployed-commit verification.
def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _get_git_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _get_git_dirty() -> bool | None:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    return bool(output.strip())


def _resolve_git_dir(repo_root: Path = _REPO_ROOT) -> Path | None:
    """Resolve this worktree's git directory without spawning git."""
    git_path = repo_root / ".git"
    try:
        if git_path.is_dir():
            return git_path
        if git_path.is_file():
            content = git_path.read_text(encoding="utf-8", errors="replace").strip()
            if content.startswith("gitdir:"):
                raw_path = content.split(":", 1)[1].strip()
                path = Path(raw_path)
                return path if path.is_absolute() else (repo_root / path).resolve()
    except Exception:
        return None
    return None


def _resolve_common_git_dir(git_dir: Path) -> Path:
    """Resolve shared git metadata for linked worktrees."""
    try:
        common_dir = (git_dir / "commondir").read_text(encoding="utf-8", errors="replace").strip()
        if common_dir:
            path = Path(common_dir)
            return path if path.is_absolute() else (git_dir / path).resolve()
    except Exception:
        pass
    return git_dir


def _read_git_head_commit(repo_root: Path = _REPO_ROOT) -> str:
    """Read current HEAD cheaply for health checks.

    /health and /readyz are hot operational paths. Spawning git here can block
    indefinitely when the worktree/gitdir is contended, which makes deep health
    fail while /healthz remains healthy.
    """
    git_dir = _resolve_git_dir(repo_root)
    if git_dir is None:
        return "unknown"
    try:
        common_git_dir = _resolve_common_git_dir(git_dir)
        head = (git_dir / "HEAD").read_text(encoding="utf-8", errors="replace").strip()
        if not head:
            return "unknown"
        if not head.startswith("ref:"):
            return head[:7]
        ref_name = head.split(" ", 1)[1].strip()
        for ref_path in (git_dir / ref_name, common_git_dir / ref_name):
            if ref_path.exists():
                commit = ref_path.read_text(encoding="utf-8", errors="replace").strip()
                return commit[:7] if commit else "unknown"
        for packed_refs in (git_dir / "packed-refs", common_git_dir / "packed-refs"):
            if not packed_refs.exists():
                continue
            for line in packed_refs.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line or line.startswith(("#", "^")):
                    continue
                try:
                    commit, packed_ref = line.split(" ", 1)
                except ValueError:
                    continue
                if packed_ref.strip() == ref_name:
                    return commit[:7]
    except Exception:
        return "unknown"
    return "unknown"


_GIT_COMMIT = _get_git_commit()
_GIT_BRANCH = _get_git_branch()
_GIT_DIRTY = _get_git_dirty()
_GIT_MISMATCH_REASON = "runtime_git_commit_mismatch"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        logger.warning("Invalid integer env var %s=%r; using %s", name, os.environ.get(name), default)
        return default


_FAST_STATS_CACHE_TTL_S = float(os.environ.get("PITH_FAST_STATS_CACHE_TTL_S", "5"))
_FAST_STATS_COLD_BUDGET_S = float(os.environ.get("PITH_FAST_STATS_COLD_BUDGET_S", "0.5"))
_FAST_STATS_MAX_STALE_S = float(os.environ.get("PITH_FAST_STATS_MAX_STALE_S", "60"))
_FAST_STATS_CACHE_LOCK = threading.Lock()
_FAST_STATS_CACHE: tuple[float, dict] | None = None
_FAST_STATS_REFRESH_LOCK = threading.Lock()
_FAST_STATS_REFRESH_FUTURE: concurrent.futures.Future | None = None
_FAST_STATS_REFRESH_ERROR: str | None = None
_FAST_STATS_OBSERVABILITY: dict[str, int | float | str | None] = {
    "refresh_success_count": 0,
    "refresh_failure_count": 0,
    "stale_served_count": 0,
    "last_refresh_duration_ms": None,
    "last_refresh_error": None,
}
_FAST_HEALTH_CACHE_TTL_S = float(os.environ.get("PITH_FAST_HEALTH_CACHE_TTL_S", "5"))
_FAST_HEALTH_CACHE_LOCK = threading.Lock()
_FAST_HEALTH_CACHE: tuple[float, dict] | None = None
_KNOWLEDGE_AREAS_CACHE_TTL_S = float(os.environ.get("PITH_KNOWLEDGE_AREAS_CACHE_TTL_S", "5"))
_KNOWLEDGE_AREAS_CACHE_LOCK = threading.Lock()
_KNOWLEDGE_AREAS_CACHE: tuple[float, dict] | None = None
_DIAGNOSTIC_ENDPOINT_SEMAPHORE = threading.Semaphore(max(1, _env_int("PITH_DIAGNOSTIC_ENDPOINT_SLOTS", 1)))
_DIAGNOSTIC_ENDPOINT_SLOT_TIMEOUT_S = float(os.environ.get("PITH_DIAGNOSTIC_ENDPOINT_SLOT_TIMEOUT_S", "0.05"))
_DIAGNOSTIC_REFRESH_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="pith-diagnostic-refresh",
)


def _build_git_provenance() -> dict:
    """Compare imported runtime code against the current worktree HEAD."""
    current_head = _read_git_head_commit()
    matches_head = None
    if _GIT_COMMIT != "unknown" and current_head != "unknown":
        matches_head = current_head == _GIT_COMMIT
    return {
        "git_commit": _GIT_COMMIT,
        "git_branch": _GIT_BRANCH,
        "git_dirty": _GIT_DIRTY,
        "git_current_head": current_head,
        "git_commit_matches_head": matches_head,
    }

from app.api.write_durability import (
    abandon_write_request,
    begin_write_request,
    commit_write_request,
    fail_write_request,
)
from app.core.brain_lock import BrainLockError, acquire_brain_lock, release_brain_lock
from app.core.datetime_utils import _ensure_aware, _utc_now, _utc_now_iso
from app.governance.runtime_install_guard import RuntimeInstallGuardError, ensure_safe_installed_runtime

# Load .env for dev API key persistence (e.g., ANTHROPIC_API_KEY for Tier 2 LLM).
# python-dotenv won't override existing env vars, so explicit exports still win.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # python-dotenv optional in production

# RETRIEVAL-057/INGEST-034: Re-compute feature flags that may have been cached
# before load_dotenv() ran (app/__init__.py import chain loads config.py early)
import app.core.config as _cfg

_cfg.EE_ENABLED = _cfg.os.environ.get("PITH_EVENT_EXTRACTION", "").lower() in ("1", "true", "yes")
_cfg.PROSPECTIVE_INDEXING_ENABLED = _cfg.os.environ.get("PITH_PROSPECTIVE_INDEXING", "0") == "1"
_cfg.SESSION_LEARN_SYNC_WAIT_SECONDS = float(
    os.environ.get("PITH_SESSION_LEARN_SYNC_WAIT_SECONDS", str(_cfg.SESSION_LEARN_SYNC_WAIT_SECONDS))
)
_cfg.SESSION_LEARN_PROCESSING_RETRY_AFTER_SECONDS = float(
    os.environ.get(
        "PITH_SESSION_LEARN_PROCESSING_RETRY_AFTER_SECONDS",
        str(_cfg.SESSION_LEARN_PROCESSING_RETRY_AFTER_SECONDS),
    )
)
_cfg.SESSION_LEARN_EXECUTOR_WORKERS = int(
    os.environ.get("PITH_SESSION_LEARN_EXECUTOR_WORKERS", str(_cfg.SESSION_LEARN_EXECUTOR_WORKERS))
)
_cfg.SESSION_LEARN_LIFECYCLE_JOBS_ENABLED = os.environ.get(
    "PITH_SESSION_LEARN_LIFECYCLE_JOBS_ENABLED",
    str(_cfg.SESSION_LEARN_LIFECYCLE_JOBS_ENABLED).lower(),
).lower() in ("1", "true", "yes")
_cfg.LIFECYCLE_JOBS_ENABLED = os.environ.get(
    "PITH_LIFECYCLE_JOBS_ENABLED",
    str(_cfg.LIFECYCLE_JOBS_ENABLED).lower(),
).lower() in ("1", "true", "yes")
_cfg.LIFECYCLE_JOBS_FALLBACK_DIRECT = os.environ.get(
    "PITH_LIFECYCLE_JOBS_FALLBACK_DIRECT",
    str(_cfg.LIFECYCLE_JOBS_FALLBACK_DIRECT).lower(),
).lower() in ("1", "true", "yes")
_cfg.LIFECYCLE_WORKERS = max(1, int(os.environ.get("PITH_LIFECYCLE_WORKERS", str(_cfg.LIFECYCLE_WORKERS))))
_cfg.LIFECYCLE_JOB_LEASE_SECONDS = int(
    os.environ.get("PITH_LIFECYCLE_JOB_LEASE_SECONDS", str(_cfg.LIFECYCLE_JOB_LEASE_SECONDS))
)
_cfg.LIFECYCLE_DRAIN_STUCK_SECONDS = int(
    os.environ.get("PITH_LIFECYCLE_DRAIN_STUCK_SECONDS", str(_cfg.LIFECYCLE_DRAIN_STUCK_SECONDS))
)
_cfg.LIFECYCLE_JOB_MAX_ATTEMPTS = int(
    os.environ.get("PITH_LIFECYCLE_JOB_MAX_ATTEMPTS", str(_cfg.LIFECYCLE_JOB_MAX_ATTEMPTS))
)
_cfg.LIFECYCLE_JOB_RETRY_SECONDS = int(
    os.environ.get("PITH_LIFECYCLE_JOB_RETRY_SECONDS", str(_cfg.LIFECYCLE_JOB_RETRY_SECONDS))
)
_cfg.LIFECYCLE_JOB_CLEANUP_DAYS = int(
    os.environ.get("PITH_LIFECYCLE_JOB_CLEANUP_DAYS", str(_cfg.LIFECYCLE_JOB_CLEANUP_DAYS))
)


import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib import import_module
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import (
    BaseModel as PydanticBaseModel,
)
from pydantic import (
    Field,
    StrictBool,
    StrictInt,
    ValidationError,
)

from app.cognitive.learning import create_concept, evolve_concept, validate_proposal
from app.core.logging_config import setup_logging
from app.core.models import (
    Association,
    AutoAssociateBatchRequest,
    AutoAssociateSingleRequest,
    ConceptEvolution,
    ConceptProposal,
    ConversationTurnRequest,
    SearchQuery,
    SessionEndRequest,
    SessionLearnRequest,
    SessionLearnResponse,
)
from app.storage import (
    add_association,
    count_associations,
    get_distribution_report,
    get_memory_projection_data,
    get_related_concepts,
    list_archived_concepts,
    list_concepts_full,
    list_sessions,
    load_concept,
    load_write_request_replay,
    restore_concept,
    run_storage_migration,
    save_concept,
)
from app.storage.connection import read_snapshot_db, request_db_scope

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)


class _LazyImportProxy:
    """Defer heavyweight imports until first real use.

    This keeps module import time low enough for liveness startup work while
    preserving the existing call sites throughout the server module.
    """

    def __init__(self, loader: Callable[[], Any]):
        self._loader = loader
        self._value: Any | None = None

    def _load(self) -> Any:
        if self._value is None:
            self._value = self._loader()
        return self._value

    def __getattr__(self, name: str) -> Any:
        return getattr(self._load(), name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._load()(*args, **kwargs)


causal = _LazyImportProxy(lambda: import_module("app.retrieval.causal"))
question_queue = _LazyImportProxy(lambda: import_module("app.features.question_queue"))
temporal = _LazyImportProxy(lambda: import_module("app.retrieval.temporal"))
curiosity_engine = _LazyImportProxy(lambda: import_module("app.features.curiosity").curiosity_engine)
goal_directed = _LazyImportProxy(lambda: import_module("app.features.goal_directed").goal_directed)
predictive_activation = _LazyImportProxy(lambda: import_module("app.retrieval.predictive").predictive_activation)
reflection_engine = _LazyImportProxy(lambda: import_module("app.cognitive.reflection").reflection_engine)
retrieval_engine = _LazyImportProxy(lambda: import_module("app.retrieval").retrieval_engine)
conversation_processor = _LazyImportProxy(lambda: import_module("app.cognitive.retrospective").conversation_processor)
self_model_manager = _LazyImportProxy(lambda: import_module("app.session.self_model").self_model_manager)
session_manager = _LazyImportProxy(lambda: import_module("app.session").session_manager)


def auto_associate_batch(*args: Any, **kwargs: Any) -> Any:
    return import_module("app.cognitive.association").auto_associate_batch(*args, **kwargs)


def auto_associate_single(*args: Any, **kwargs: Any) -> Any:
    return import_module("app.cognitive.association").auto_associate_single(*args, **kwargs)


# MATURITY-001: Maturities blocked from external API results
_BLOCKED_MATURITIES = {"QUARANTINED", "DISCARDED"}

app = FastAPI(
    title="Pith Server", version="1.1.0", description="AI Learning Architecture with versioned conceptual memory"
)

_PITH_ORIENT_CACHE_LOCK = threading.Lock()
_PITH_ORIENT_CACHE: dict[tuple[str], dict[str, Any]] = {}
_PITH_ORIENT_DEFAULT_CACHE_TTL_SECONDS = 15.0


def _pith_orient_cache_ttl_seconds() -> float:
    raw_value = os.environ.get("PITH_ORIENT_CACHE_TTL_SECONDS")
    if raw_value is None:
        return _PITH_ORIENT_DEFAULT_CACHE_TTL_SECONDS
    try:
        return max(0.0, float(raw_value))
    except ValueError:
        return _PITH_ORIENT_DEFAULT_CACHE_TTL_SECONDS


def _pith_orient_payload_hash(payload: dict[str, Any]) -> str:
    data = copy.deepcopy(payload)
    data.pop("generated_at", None)
    data.pop("orientation_hash", None)
    if data.get("workstreams") is None:
        data.pop("workstreams", None)
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


def _compute_pith_orient_base_payload(time_window: str) -> dict[str, Any]:
    concepts = session_manager._load_all_concepts()
    orientation = session_manager.orient(
        concepts=concepts,
        time_window=time_window,
        include_workstreams=False,
    )
    payload = orientation.model_dump()
    payload.pop("workstreams", None)
    return payload


def _pith_orient_base_payload(time_window: str, *, force_refresh: bool = False) -> dict[str, Any]:
    ttl_seconds = _pith_orient_cache_ttl_seconds()
    if ttl_seconds <= 0:
        return _compute_pith_orient_base_payload(time_window)

    cache_key = (time_window,)
    now = time.monotonic()
    if not force_refresh:
        with _PITH_ORIENT_CACHE_LOCK:
            cached = _PITH_ORIENT_CACHE.get(cache_key)
            if cached and float(cached["expires_at"]) > now:
                return copy.deepcopy(cached["payload"])

    payload = _compute_pith_orient_base_payload(time_window)
    with _PITH_ORIENT_CACHE_LOCK:
        _PITH_ORIENT_CACHE[cache_key] = {
            "expires_at": time.monotonic() + ttl_seconds,
            "payload": copy.deepcopy(payload),
        }
    return payload


def _pith_orient_payload_with_workstreams(
    base_payload: dict[str, Any],
    *,
    workstream_limit: int,
    origin_id: str | None,
    session_id: str | None,
) -> dict[str, Any]:
    payload = copy.deepcopy(base_payload)
    payload["workstreams"] = session_manager._compute_workstreams_status(
        origin_id=origin_id,
        session_id=session_id,
        limit=workstream_limit,
    )
    payload["orientation_hash"] = _pith_orient_payload_hash(payload)
    return payload

app.state.process_state = "starting"
app.state.write_state = "queued"
app.state.retrieval_state = "recovering"
app.state.maintenance_state = "disabled"
app.state.degraded_reason = None
app.state.last_successful_write_at = None
app.state.outbox_depth = 0
app.state.startup_warnings = []
app.state.startup_task = None

# OPS-074: Configurable host/port with sane defaults
PITH_HOST = os.environ.get("PITH_HOST", "127.0.0.1").strip() or "127.0.0.1"
# PERF-FORT-1: Concurrency limiter for heavyweight endpoints.
# Prevents threadpool starvation under concurrent MCP tool calls.
# Value 2 = at most 2 heavy requests process simultaneously;
# others queue (with timeout) rather than crash the server.
import threading as _threading_fort

HEAVY_SEMAPHORE_LIMIT = 2
HEAVY_ENDPOINT_TIMEOUT_S = 30
_HEAVY_ENDPOINT_SEMAPHORE = _threading_fort.Semaphore(HEAVY_SEMAPHORE_LIMIT)
_SESSION_LEARN_SYNC_WAIT_SECONDS = max(0.0, min(_cfg.SESSION_LEARN_SYNC_WAIT_SECONDS, 15.0))
_SESSION_LEARN_PROCESSING_RETRY_AFTER_SECONDS = max(0.0, _cfg.SESSION_LEARN_PROCESSING_RETRY_AFTER_SECONDS)
_SESSION_LEARN_EXECUTOR_WORKERS = max(1, _cfg.SESSION_LEARN_EXECUTOR_WORKERS)
_SESSION_LEARN_LIFECYCLE_JOBS_ENABLED = bool(_cfg.SESSION_LEARN_LIFECYCLE_JOBS_ENABLED)
_SESSION_LEARN_LIFECYCLE_PRIORITY = int(os.environ.get("PITH_SESSION_LEARN_LIFECYCLE_PRIORITY", "60"))
_SESSION_LEARN_RECLAIMER_BATCH_SIZE = int(os.environ.get("PITH_SESSION_LEARN_RECLAIMER_BATCH_SIZE", "5"))
_SESSION_LEARN_RECLAIMER_MAX_ATTEMPTS = int(os.environ.get("PITH_SESSION_LEARN_RECLAIMER_MAX_ATTEMPTS", "3"))
_SESSION_LEARN_RECLAIMER_LEASE_SECONDS = int(os.environ.get("PITH_SESSION_LEARN_RECLAIMER_LEASE_SECONDS", "300"))
_SESSION_LEARN_RECLAIMER_RETRY_SECONDS = int(os.environ.get("PITH_SESSION_LEARN_RECLAIMER_RETRY_SECONDS", "60"))
_SESSION_LEARN_EXECUTOR: concurrent.futures.ThreadPoolExecutor | None = None
_SESSION_LEARN_EXECUTOR_LOCK = threading.Lock()
_SESSION_LEARN_RECLAIMER_LOCK = threading.Lock()
_SESSION_LEARN_RECLAIMER_FUTURE: concurrent.futures.Future | None = None
_WRITE_REPLAY_RECLAIMER_LOCK = threading.Lock()
_WRITE_REPLAY_RECLAIMER_FUTURES: dict[str, concurrent.futures.Future] = {}

# ARGUS-C3-F1: Fail-fast with clear message on invalid port
_raw_port = os.environ.get("PITH_PORT", "8000")
try:
    PITH_PORT = int(_raw_port)
except (ValueError, TypeError):
    print(f"FATAL: PITH_PORT must be an integer, got {_raw_port!r}", file=sys.stderr)
    sys.exit(1)
if not (1 <= PITH_PORT <= 65535):
    print(f"FATAL: PITH_PORT must be 1-65535, got {PITH_PORT}", file=sys.stderr)
    sys.exit(1)

# OPS-074: CORS origins derived from configured port — no hardcoded port numbers
_cors_origins = [
    f"http://localhost:{PITH_PORT}",
    f"http://127.0.0.1:{PITH_PORT}",
    "http://localhost:3000",  # Dev frontend
    "http://127.0.0.1:3000",  # Dev frontend
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Route Naming Normalization (ROUTE-COMPAT) ---
# Backward-compatible middleware: rewrites deprecated paths to canonical pith_ paths.
# Old callers (server.js, tests) keep working while we migrate incrementally.
from app.api.route_compat import install_route_compat_middleware

install_route_compat_middleware(app)

# --- API Key Authentication for Write Endpoints (OPS-072) ---


def _ensure_api_key() -> str:
    """Ensure an API key exists — read from env, or auto-generate and persist.

    Priority: PITH_API_KEY env > BRAIN_API_KEY env > auto-generate.
    A3: If persistence to ~/.pith/.env fails, still sets os.environ (session-only)
    and logs a warning so operators know the key won't survive restart.
    """
    import secrets

    key = os.environ.get("PITH_API_KEY") or os.environ.get("BRAIN_API_KEY", "")
    if key:
        return key

    # Auto-generate a key for consumer installs that didn't set one
    key = secrets.token_hex(32)
    os.environ["PITH_API_KEY"] = key

    # Attempt to persist to ~/.pith/.env
    try:
        pith_dir = Path(os.path.expanduser("~/.pith"))
        pith_dir.mkdir(parents=True, exist_ok=True)
        env_file = pith_dir / ".env"

        # Read existing content, append PITH_API_KEY
        existing = env_file.read_text() if env_file.exists() else ""
        if "PITH_API_KEY" not in existing:
            with open(env_file, "a") as f:
                f.write(f"\nPITH_API_KEY={key}\n")
            # A3: Secure file permissions
            os.chmod(env_file, 0o600)
            logger.info(f"Auto-generated PITH_API_KEY persisted to {env_file}")
        else:
            # Read the persisted key from file (was written on a prior start)
            for _line in existing.splitlines():
                if _line.startswith("PITH_API_KEY="):
                    _persisted = _line.split("=", 1)[1].strip()
                    if _persisted:
                        os.environ["PITH_API_KEY"] = _persisted
                        logger.info(
                            "MONITOR-134: Loaded existing PITH_API_KEY from ~/.pith/.env (file fallback active — env var not set)"
                        )
                        try:
                            from app.core.metrics_facade import metrics as _auth_m

                            _auth_m.record("api_key_file_fallback", 1.0)
                        except Exception:
                            pass
                        return _persisted
            # Entry present but empty/malformed — fall through with generated key
            logger.info("PITH_API_KEY in ~/.pith/.env was empty — using generated key")
    except Exception as e:
        # A3: Partial persistence fallback — key works this session but won't survive restart
        logger.warning(
            f"Auto-generated PITH_API_KEY set in env but FAILED to persist to ~/.pith/.env: {e}. "
            f"Key is session-only — will be regenerated on next restart."
        )

    return key


API_KEY = _ensure_api_key()


async def verify_api_key(request: Request):
    """Verify API key on write endpoints. No dev-mode bypass (OPS-072)."""
    key = request.headers.get("X-API-Key", "")
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


_LOCAL_OPERATOR_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}


class AnswerPathPolicyUpdateRequest(PydanticBaseModel):
    """Strict runtime answer-path policy update payload."""

    observe_only: StrictBool
    enforcement_enabled: StrictBool
    ttl_seconds: StrictInt = Field(..., ge=1)
    enforce_modes: list[str] | None = None
    source: str = Field(default="runtime_api", min_length=1, max_length=40)


class AnswerPathPolicyResponse(PydanticBaseModel):
    """Runtime answer-path policy response payload."""

    observe_only: bool
    enforcement_enabled: bool
    source: str
    state: str
    generation: int
    enforce_modes: list[str] = Field(default_factory=list)
    expires_at: str | None = None
    runtime_active: bool
    max_ttl_seconds: int


class AnswerPathPolicyResetRequest(PydanticBaseModel):
    """Runtime answer-path policy reset payload."""

    source: str = Field(default="runtime_api", min_length=1, max_length=40)


def _is_local_request(request: Request) -> bool:
    """Return whether the request originated from the local operator surface."""
    direct_host = getattr(getattr(request, "client", None), "host", None)
    forwarded_for = request.headers.get("X-Forwarded-For", "")
    if forwarded_for:
        first_hop = forwarded_for.split(",", 1)[0].strip()
        if first_hop and first_hop not in _LOCAL_OPERATOR_HOSTS:
            return False
    return direct_host in _LOCAL_OPERATOR_HOSTS


def _require_local_operator(request: Request) -> None:
    """Require local operator access in addition to API-key authentication."""
    if not _is_local_request(request):
        raise HTTPException(
            status_code=403,
            detail="Runtime policy updates require local operator access",
        )


def _record_answer_path_policy_metric(name: str, labels: dict[str, str]) -> None:
    """Record low-cardinality answer-path policy metrics best-effort."""
    try:
        from app.ops.metrics import metrics as _policy_metrics

        _policy_metrics.record(name, 1.0, labels)
    except Exception:
        pass


def _answer_path_policy_response(snapshot) -> AnswerPathPolicyResponse:
    from app.session.answer_path_policy import get_answer_path_policy

    return AnswerPathPolicyResponse(
        **asdict(snapshot),
        max_ttl_seconds=get_answer_path_policy().max_ttl_seconds(),
    )


# MONITOR-076: Sanitize error responses — never leak internal paths or tracebacks.
# In production (PITH_DEBUG != "1"), return generic message; in dev, expose str(e).
_PITH_DEBUG = os.environ.get("PITH_DEBUG", "0") == "1"


# OPS-500-FIX: Retry helper for transient DB lock errors
import sqlite3 as _sqlite3_retry
import time as _time_retry
from datetime import UTC


def _with_db_retry(fn, max_retries=2, backoff=0.5):
    """Execute fn with retry on transient DB lock errors."""
    last_err = None
    for attempt in range(max_retries):
        try:
            return fn()
        except (RuntimeError, _sqlite3_retry.OperationalError) as e:
            err_str = str(e).lower()
            if "deadlock" in err_str or "database is locked" in err_str or "lock" in err_str:
                last_err = e
                if attempt < max_retries - 1:
                    logger.warning(
                        "OPS-500-FIX: Transient DB error on attempt %d/%d, retrying in %.1fs: %s",
                        attempt + 1,
                        max_retries,
                        backoff,
                        e,
                    )
                    _time_retry.sleep(backoff)
                    backoff *= 2
                    continue
            raise
    raise last_err


def _safe_error(exc: Exception) -> str:
    """Return safe error detail: full message in debug mode, generic in production."""
    if _PITH_DEBUG:
        return str(exc)
    return "Internal server error"


def _record_successful_write() -> None:
    app.state.last_successful_write_at = _utc_now_iso()
    app.state.write_state = "accepting"


def _set_degraded_reason(reason: str | None) -> None:
    app.state.degraded_reason = reason
    if reason:
        app.state.retrieval_state = "degraded"


def _build_auth_status(request: Request | None = None) -> dict:
    caller_key = request.headers.get("X-API-Key", "") if request else ""
    key_configured = bool(API_KEY)
    caller_authenticated = key_configured and caller_key == API_KEY
    return {
        "key_configured": key_configured,
        "caller_authenticated": caller_authenticated,
        "write_path": "ok" if caller_authenticated else ("no_key_configured" if not key_configured else "invalid_key"),
    }


def _build_ready_state() -> dict:
    process_state = getattr(app.state, "process_state", "starting")
    write_state = getattr(app.state, "write_state", "accepting")
    retrieval_state = getattr(app.state, "retrieval_state", "recovering")
    maintenance_state = getattr(app.state, "maintenance_state", "disabled")
    degraded_reason = getattr(app.state, "degraded_reason", None)
    warnings = list(getattr(app.state, "startup_warnings", []) or [])
    git_provenance = _build_git_provenance()
    commit_matches_head = git_provenance["git_commit_matches_head"]
    if commit_matches_head is False:
        if degraded_reason is None:
            degraded_reason = _GIT_MISMATCH_REASON
        warning = (
            f"{_GIT_MISMATCH_REASON}: startup={git_provenance['git_commit']} "
            f"current_head={git_provenance['git_current_head']}"
        )
        if warning not in warnings:
            warnings.append(warning)
    mode = "ready"
    if process_state == "blocked" or write_state == "blocked":
        mode = "blocked"
    elif process_state != "running":
        mode = "recovering"
    elif degraded_reason or retrieval_state in {"recovering", "degraded"} or commit_matches_head is False:
        mode = "degraded"
    return {
        "status": "healthy" if process_state in {"starting", "running"} else "stopping",
        "service": "pith",
        "timestamp": _utc_now_iso(),
        "version": "1.1.0",
        **git_provenance,
        "mode": mode,
        "process_state": process_state,
        "write_state": write_state,
        "retrieval_state": retrieval_state,
        "maintenance_state": maintenance_state,
        "degraded_reason": degraded_reason,
        "last_successful_write_at": getattr(app.state, "last_successful_write_at", None),
        "outbox_depth": getattr(app.state, "outbox_depth", 0),
        "startup_warnings": warnings,
    }


def _external_maintenance_freshness_threshold_hours(scheduler_installed: bool) -> float:
    """Return stale threshold for the configured external maintenance scheduler."""
    return 36.0 if scheduler_installed else 12.0


def _build_external_maintenance_health() -> dict[str, Any]:
    """Summarize external maintenance health without assuming built-in scheduling."""
    import json as _json
    from datetime import datetime as _dt

    from app.core.profile import resolve_data_dir

    heartbeat_path = Path(resolve_data_dir()) / "maintenance_heartbeat.json"
    scheduler_path = Path.home() / "Library" / "LaunchAgents" / "com.pith.maintenance.plist"
    scheduler_installed = scheduler_path.exists()

    base = {
        "scheduler_backend": "launchd",
        "scheduler_installed": scheduler_installed,
        "scheduler_path": str(scheduler_path),
        "heartbeat_path": str(heartbeat_path),
        "external_launchd_scheduler": {
            "backend": "launchd",
            "installed": scheduler_installed,
            "optional": True,
            "path": str(scheduler_path),
            "schedule": "daily 3:00 AM",
        },
    }

    if not heartbeat_path.exists():
        scheduler_state = "scheduled" if scheduler_installed else "not_installed"
        message = (
            "External launchd maintenance is scheduled but no heartbeat exists yet."
            if scheduler_installed
            else "External launchd maintenance is not installed (optional)."
        )
        return {
            **base,
            "status": "never_run" if scheduler_installed else "not_installed",
            "scheduler_state": scheduler_state,
            "alert": False,
            "message": message,
            "heartbeat": {
                "exists": False,
                "path": str(heartbeat_path),
                "source": None,
            },
            "external_launchd_scheduler": {
                **base["external_launchd_scheduler"],
                "state": scheduler_state,
            },
        }

    try:
        data = _json.loads(heartbeat_path.read_text())
        last_run_raw = data.get("last_run") or data.get("timestamp")
        if not last_run_raw:
            return {
                **base,
                **data,
                "status": "error",
                "scheduler_state": "error",
                "alert": True,
                "message": "maintenance heartbeat missing last_run/timestamp",
            }
        last_run = _ensure_aware(_dt.fromisoformat(last_run_raw))
        hours_since = (_utc_now() - last_run).total_seconds() / 3600
        status = data.get("status", "unknown")
        source = data.get("source") or ("built_in" if data.get("scheduler") == "builtin" else "unknown")
        freshness_threshold_hours = _external_maintenance_freshness_threshold_hours(scheduler_installed)
        freshness_state = "stale" if hours_since > freshness_threshold_hours else "fresh"
        alert = freshness_state == "stale" or status in {"error", "errors", "circuit_open"}
        return {
            **base,
            **data,
            "source": source,
            "hours_since_last_run": round(hours_since, 1),
            "freshness_threshold_hours": freshness_threshold_hours,
            "freshness_state": freshness_state,
            "scheduler_state": "degraded" if alert else "healthy",
            "alert": alert,
            "heartbeat": {
                "exists": True,
                "path": str(heartbeat_path),
                "source": source,
                "status": status,
                "last_run": last_run_raw,
                "hours_since_last_run": round(hours_since, 1),
                "freshness_threshold_hours": freshness_threshold_hours,
                "freshness_state": freshness_state,
            },
            "external_launchd_scheduler": {
                **base["external_launchd_scheduler"],
                "state": "installed" if scheduler_installed else "not_installed",
            },
        }
    except Exception as e:
        return {
            **base,
            "status": "error",
            "scheduler_state": "error",
            "alert": True,
            "message": _safe_error(e),
            "heartbeat": {
                "exists": heartbeat_path.exists(),
                "path": str(heartbeat_path),
                "source": "unknown",
            },
            "external_launchd_scheduler": {
                **base["external_launchd_scheduler"],
                "state": "error",
            },
        }


def _build_maintenance_health() -> dict[str, Any]:
    """Build user-facing maintenance health with effective, built-in, and external sections."""
    external_health = _build_external_maintenance_health()
    built_in_state = getattr(app.state, "maintenance_state", "disabled")
    effective_state, maintenance_mode = _effective_maintenance_state(built_in_state, external_health)

    external_absence_only = (
        built_in_state != "disabled"
        and not external_health.get("scheduler_installed")
        and external_health.get("status") == "not_installed"
    )
    if external_absence_only:
        external_health["alert"] = False
        external_health["message"] = "External launchd scheduler is not installed (optional; built-in maintenance is running)."

    heartbeat = external_health.get("heartbeat", {})
    external_scheduler = external_health.get("external_launchd_scheduler", {})
    return {
        **external_health,
        "effective": {
            "state": effective_state,
            "mode": maintenance_mode,
            "alert": effective_state in {"degraded", "disabled"} and not external_absence_only,
        },
        "built_in_scheduler": {
            "state": built_in_state,
            "default": True,
        },
        "external_launchd_scheduler": {
            **external_scheduler,
            "optional": True,
        },
        "heartbeat": heartbeat,
    }


def _effective_maintenance_state(
    built_in_state: str,
    external_health: dict[str, Any],
) -> tuple[str, str]:
    """Collapse built-in and external states into an honest overall mode/state."""
    if built_in_state != "disabled":
        return built_in_state, "built_in"

    external_state = external_health.get("scheduler_state", "not_installed")
    if external_state in {"healthy", "scheduled"}:
        return "external", "external"
    if external_state in {"degraded", "error"}:
        return "degraded", "external"
    if external_state == "never_run":
        return "scheduled", "external"
    return "disabled", "none"
def _require_thread_reorg_ready(*, require_write: bool = False, require_retrieval_ready: bool = False) -> None:
    """Reject THREAD-004 requests when startup/recovery state makes them unsafe."""
    ready = _build_ready_state()
    if ready["process_state"] != "running":
        raise HTTPException(
            status_code=503,
            detail="THREAD-004 unavailable while server startup is still in progress",
            headers={"Retry-After": "10"},
        )
    if require_write and ready["write_state"] != "accepting":
        raise HTTPException(
            status_code=503,
            detail="THREAD-004 write path not ready yet",
            headers={"Retry-After": "10"},
        )
    if require_retrieval_ready and ready["retrieval_state"] == "recovering":
        raise HTTPException(
            status_code=503,
            detail="THREAD-004 mining unavailable while retrieval recovery is still in progress",
            headers={"Retry-After": "10"},
        )


def _require_retrieval_ready(endpoint_name: str) -> None:
    """Reject latency-sensitive reads while deferred retrieval startup is still warming."""
    ready = _build_ready_state()
    if ready["process_state"] != "running":
        raise HTTPException(
            status_code=503,
            detail=f"{endpoint_name} unavailable while server startup is still in progress",
            headers={"Retry-After": "10"},
        )
    if ready["retrieval_state"] == "recovering":
        raise HTTPException(
            status_code=503,
            detail=f"{endpoint_name} unavailable while retrieval initialization is still in progress",
            headers={"Retry-After": "10"},
        )


def _acquire_thread_reorg_slot() -> None:
    """Reuse the heavy-endpoint limiter for THREAD-004 operations."""
    acquired = _HEAVY_ENDPOINT_SEMAPHORE.acquire(timeout=HEAVY_ENDPOINT_TIMEOUT_S)
    if not acquired:
        raise HTTPException(
            status_code=503,
            detail="Server under heavy load — try again in a few seconds",
            headers={"Retry-After": "3"},
        )


def select_ambient_concepts(
    concept_id: str,
    query: str = "",
    exclude_ids: set = None,
    max_results: int = 2,
) -> list[dict[str, Any]]:
    """C2: Select up to 2 related concepts for ambient_context enrichment.

    Primary: Association graph neighbors sorted by strength. O(1) per concept.
    Fallback: TF-IDF similarity search if graph returns < max_results.
    Hard cap at max_results. Deduplicates against exclude_ids.
    """
    if exclude_ids is None:
        exclude_ids = set()
    exclude_ids.add(concept_id)

    results = []

    # Primary: association graph neighbors
    try:
        neighbors = get_related_concepts(concept_id, max_depth=1)
        for nid in neighbors:
            if nid in exclude_ids:
                continue
            neighbor = load_concept(nid, track_access=False)
            if neighbor:
                results.append(
                    {
                        "concept_id": nid,
                        "summary": (neighbor.summary[:150] + "...")
                        if len(neighbor.summary) > 150
                        else neighbor.summary,
                        "confidence": round(neighbor.confidence, 2),
                        "relation": "associated",
                    }
                )
                exclude_ids.add(nid)
            if len(results) >= max_results:
                break
    except Exception as e:
        logger.debug(f"C2: Association lookup failed for {concept_id}: {e}")

    # Fallback: TF-IDF similarity if < max_results from graph
    if len(results) < max_results and query:
        try:
            search_query = SearchQuery(query=query, max_results=max_results + 2)
            search_results = retrieval_engine.search(search_query)
            for sr in search_results:
                if sr.concept_id in exclude_ids:
                    continue
                results.append(
                    {
                        "concept_id": sr.concept_id,
                        "summary": (sr.summary[:150] + "...") if len(sr.summary) > 150 else sr.summary,
                        "confidence": round(sr.confidence, 2),
                        "relation": "semantically_similar",
                    }
                )
                exclude_ids.add(sr.concept_id)
                if len(results) >= max_results:
                    break
        except Exception as e:
            logger.debug(f"C2: TF-IDF fallback failed: {e}")

    return results[:max_results]


async def _deferred_index_repair(integrity: dict):
    """STABILITY-030: Run index repair in background after startup completes.

    Deferred from startup_event() so health endpoint is available immediately.
    Uses asyncio.to_thread() because repair_index_drift() is synchronous and
    CPU/IO-bound (disk saves every 10 ghost removals).
    """
    import asyncio

    try:
        logger.info(f"Background index repair starting — {integrity['ghosts']} ghosts, {integrity['orphans']} orphans")
        repair = await asyncio.to_thread(retrieval_engine.repair_index_drift, integrity=integrity)
        logger.info(f"Background index repair complete — {repair}")
    except asyncio.CancelledError:
        logger.warning("Background index repair cancelled (shutdown)")
    except Exception as e:
        logger.error(f"Background index repair failed: {e}", exc_info=True)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _stage3b_required_context_prewarm_enabled() -> bool:
    raw = os.environ.get("PITH_STAGE3B_REQUIRED_CONTEXT_PREWARM")
    if raw is not None:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    stage3b_mode = os.environ.get("PITH_STAGE3B_LATENCY_MODE", "shadow").strip().lower()
    return stage3b_mode in {"shadow", "enforce"}


async def _maybe_prewarm_required_context() -> None:
    """Populate required context before retrieval is marked ready."""
    from app.ops.metrics import metrics as _metrics

    if not _stage3b_required_context_prewarm_enabled():
        _metrics.record(
            "ct_stage3b_required_context_prewarm_total",
            1.0,
            {"result": "disabled", "state": "disabled"},
        )
        _metrics.flush()
        logger.info("Startup: Required context prewarm disabled")
        return

    started = time.perf_counter()
    try:
        from app.session.required_context_cache import prewarm_required_context

        stats = await asyncio.to_thread(prewarm_required_context)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        result = "success" if stats.state != "shutdown" else "shutdown"
        _metrics.record("ct_stage3b_required_context_prewarm_ms", round(elapsed_ms, 2))
        _metrics.record(
            "ct_stage3b_required_context_prewarm_total",
            1.0,
            {"result": result, "state": stats.state},
        )
        _metrics.flush()
        logger.info(
            "Startup: Required context prewarmed state=%s refresh_ms=%.2f always_activate_ms=%.2f firmware_ms=%.2f directives_ms=%.2f",
            stats.state,
            stats.refresh_ms,
            stats.always_activate_ms,
            stats.firmware_ms,
            stats.directives_ms,
        )
    except Exception as e:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        _metrics.record("ct_stage3b_required_context_prewarm_ms", round(elapsed_ms, 2))
        _metrics.record(
            "ct_stage3b_required_context_prewarm_total",
            1.0,
            {"result": "error", "state": type(e).__name__},
        )
        _metrics.flush()
        severity = (
            "critical"
            if _env_flag("PITH_STAGE3B_REQUIRED_CONTEXT_PREWARM_REQUIRED", False)
            else "degraded"
        )
        app.state.startup_warnings.append(
            {
                "component": "required_context_prewarm",
                "severity": severity,
                "message": f"Required context prewarm failed: {e}",
            }
        )
        if severity == "critical":
            _set_degraded_reason("required_context_prewarm_failed")
            raise
        logger.warning("Startup: Required context prewarm failed (degraded): %s", e, exc_info=True)


async def _warm_embeddings_for_startup() -> None:
    """Warm semantic embeddings without blocking retrieval readiness."""
    try:
        from app.storage.embedding import embedding_engine

        await asyncio.to_thread(embedding_engine.load_model)
        logger.info("Startup: embedding model pre-loaded (PERF-035)")
        await asyncio.to_thread(retrieval_engine._init_embeddings)
        logger.info(
            "Startup: semantic embedding index warmed: %s concepts (PERF-076)",
            embedding_engine.index_size,
        )
        if embedding_engine.index_size > 0:
            await asyncio.to_thread(
                embedding_engine.search,
                "startup semantic retrieval warmup",
                1,
            )
            logger.info("Startup: semantic retrieval query path warmed (PERF-076)")
    except asyncio.CancelledError:
        logger.info("Startup: semantic embedding warmup cancelled")
        raise
    except Exception as e:
        logger.warning(f"Startup: embedding warmup failed (non-fatal): {e}")
        app.state.startup_warnings.append(
            {"component": "embedding_warmup", "severity": "degraded", "message": str(e)}
        )


async def _warm_reranker_for_startup() -> None:
    """Warm cross-encoder reranker without blocking retrieval readiness."""
    try:
        import importlib

        reranker = await asyncio.to_thread(importlib.import_module, "app.reranker")
        _reranker_warmup_ok = await asyncio.to_thread(reranker.warmup)
        if _reranker_warmup_ok:
            logger.info("Startup: reranker model pre-loaded (PERF-040)")
        else:
            logger.warning("Startup: reranker warmup failed; CE paths degraded")
            app.state.startup_warnings.append(
                {
                    "component": "reranker_warmup",
                    "severity": "degraded",
                    "message": "Reranker warmup failed; cross-encoder paths unavailable",
                }
            )
    except asyncio.CancelledError:
        logger.info("Startup: reranker warmup cancelled")
        raise
    except Exception as e:
        logger.warning(f"Startup: reranker warmup failed (non-fatal): {e}")
        app.state.startup_warnings.append(
            {"component": "reranker_warmup", "severity": "degraded", "message": str(e)}
        )


async def _run_deferred_startup_warmups(
    *,
    warm_embeddings: bool,
    warm_reranker: bool,
) -> None:
    """Let startup finish, then serialize optional ML warmups."""
    delay_s = float(os.environ.get("PITH_STARTUP_BACKGROUND_WARMUP_DELAY_S", "5"))
    if delay_s > 0:
        await asyncio.sleep(delay_s)
    if warm_embeddings:
        logger.info("Startup: semantic embedding warmup beginning in background")
        await _warm_embeddings_for_startup()
    if warm_reranker:
        logger.info("Startup: reranker warmup beginning in background")
        await _warm_reranker_for_startup()


async def _complete_startup_initialization():
    """Finish non-critical startup work after the API is live.

    Stage 3 requires liveness to come up before retrieval warmup completes.
    Keep the write path gated via readyz.write_state until storage init finishes,
    then let retrieval/index work continue in the background.
    """
    try:
        _defer_embedding_warmup = False
        _defer_reranker_warmup = False
        # Minimal write-path dependency: storage backend + schema availability.
        try:
            from app.storage.backend import get_backend

            await asyncio.to_thread(get_backend)
            app.state.write_state = "accepting"
            _schedule_all_write_replay_reclaimers("startup")
        except Exception as e:
            app.state.process_state = "blocked"
            app.state.write_state = "blocked"
            _set_degraded_reason("storage_backend_init_failed")
            logger.error(f"CRITICAL: Storage backend init failed during deferred startup: {e}", exc_info=True)
            app.state.startup_warnings.append(
                {
                    "component": "storage_backend",
                    "severity": "critical",
                    "message": f"Storage backend init failed: {e}",
                }
            )
            return

        # SYSTEMIC_FIXES_SPEC v1.1 Fix 4: Environment validation
        _in_venv = sys.prefix != sys.base_prefix
        if not _in_venv:
            logger.warning("STARTUP WARNING: Running outside a virtual environment (sys.prefix == sys.base_prefix)")
        try:
            if _env_flag("PITH_STARTUP_BLOCK_ON_EMBEDDING_WARMUP", False):
                import importlib

                await asyncio.to_thread(importlib.import_module, "sentence_transformers")

                logger.info(
                    f"Startup: sentence_transformers available (Python {sys.version_info.major}.{sys.version_info.minor})"
                )
                app.state.embeddings_available = True
                await _warm_embeddings_for_startup()
            else:
                app.state.embeddings_available = True
                _defer_embedding_warmup = True

            _reranker_enabled = os.environ.get("PITH_RERANKER", "").lower() in ("true", "1")
            if _reranker_enabled:
                if _env_flag("PITH_STARTUP_BLOCK_ON_RERANKER_WARMUP", False):
                    await _warm_reranker_for_startup()
                else:
                    _defer_reranker_warmup = True

        except ImportError:
            logger.warning("Startup: sentence_transformers not available — TF-IDF fallback active")
            app.state.embeddings_available = False
            _set_degraded_reason("tfidf_fallback_active")

        _pith_profile = os.environ.get("PITH_PROFILE")
        if _pith_profile and not API_KEY:
            logger.warning(f"STARTUP WARNING: PITH_PROFILE={_pith_profile} but PITH_API_KEY not set")

        # Run schema migration 002 (temporal/causal) — idempotent
        try:
            import importlib

            mod = importlib.import_module("migrations.002_temporal_causal")
            await asyncio.to_thread(mod.run_migration)
            logger.info("Startup: Schema migration 002 (temporal/causal) complete")
        except ImportError:
            logger.debug("Startup: Migration 002 not found, skipping")
        except Exception as e:
            logger.warning(f"Startup: Migration 002 failed (non-fatal): {e}")

        if not API_KEY:
            logger.warning("PITH_API_KEY not set — write endpoints unprotected")

        # Run schema migration 003 (domains/directives) — idempotent
        try:
            import importlib

            mod003 = importlib.import_module("migrations.003_domains_directives")
            await asyncio.to_thread(mod003.run_migration)
            logger.info("Startup: Schema migration 003 (domains_directives) complete")
        except ImportError:
            logger.debug("Startup: Migration 003 not found, skipping")
        except Exception as e:
            logger.warning(f"Startup: Migration 003 failed (non-fatal): {e}")

        # P0-7: Seed firmware before anything else (idempotent)
        _benchmark_mode = os.environ.get("PITH_BENCHMARK_MODE", "").lower() in ("true", "1")
        try:
            if _benchmark_mode:
                logger.info("Firmware seed: SKIPPED (PITH_BENCHMARK_MODE=true)")
                result = {"action": "skipped", "version": "benchmark"}
            else:
                from app.ops.seed_firmware import seed_firmware

                result = await asyncio.to_thread(seed_firmware)
            logger.info(f"Firmware seed: {result.get('action')} (v{result.get('version')})")
        except Exception as e:
            logger.error(f"CRITICAL: Firmware seed failed: {e}", exc_info=True)
            app.state.startup_warnings.append(
                {
                    "component": "firmware_seed",
                    "severity": "critical",
                    "message": f"Firmware seed failed: {e}",
                }
            )

        try:
            from app.ops.seed_domains import seed_domains

            result = await asyncio.to_thread(seed_domains)
            logger.info(f"Domain seed: {result.get('action')} (v{result.get('version', 'n/a')})")
        except Exception as e:
            logger.warning(f"Startup: Domain seed failed (degraded): {e}", exc_info=True)
            app.state.startup_warnings.append(
                {
                    "component": "domain_seed",
                    "severity": "degraded",
                    "message": f"Domain seed failed: {e}",
                }
            )

        await _maybe_prewarm_required_context()

        try:
            if retrieval_engine.index.document_count > 0:
                logger.info(f"Index loaded from disk: {retrieval_engine.index.document_count} concepts already indexed")
            else:
                await asyncio.to_thread(retrieval_engine.build_index)
                concept_count = retrieval_engine.index.document_count
                logger.info(f"Index built successfully: {concept_count} concepts indexed")
        except Exception as e:
            logger.error(f"CRITICAL: Failed to build index on startup: {e}", exc_info=True)
            _set_degraded_reason("index_build_failed")
            app.state.startup_warnings.append(
                {
                    "component": "index_build",
                    "severity": "critical",
                    "message": f"Index build failed: {e}",
                }
            )

        try:
            integrity = await asyncio.to_thread(retrieval_engine.verify_index_integrity)
            if not integrity["is_healthy"]:
                logger.warning(
                    f"Startup: Index drift detected — {integrity['ghosts']} ghosts, {integrity['orphans']} orphans. Deferring repair to background..."
                )
                app.state.index_repair_task = asyncio.create_task(_deferred_index_repair(integrity))
            else:
                logger.info("Startup: Index integrity check passed")
        except Exception as e:
            logger.warning(f"Startup: Index integrity check failed (non-fatal): {e}")

        try:
            from app.ops.metrics import metrics as _metrics_instance

            await asyncio.to_thread(_metrics_instance.startup_health_check)
        except Exception as e:
            logger.warning(f"Startup: Metrics health check failed (non-fatal): {e}")

        try:
            run_storage_migration("migrations.005_content_updated_at")
            logger.info("Startup: Migration 005 (content_updated_at) complete")
        except ImportError:
            logger.debug("Startup: Migration 005 not found, skipping")
        except Exception as e:
            logger.warning(f"Startup: Migration 005 failed (non-fatal): {e}")

        try:
            run_storage_migration("migrations.006_junk_type_cleanup")
            logger.info("Startup: Migration 006 (junk type cleanup) complete")
        except ImportError:
            logger.debug("Startup: Migration 006 not found, skipping")
        except Exception as e:
            logger.warning(f"Startup: Migration 006 failed (non-fatal): {e}")

        try:
            run_storage_migration("migrations.013_original_date")
            logger.info("Startup: Migration 013 (original_date) complete")
        except ImportError:
            logger.debug("Startup: Migration 013 not found, skipping")
        except Exception as e:
            logger.warning(f"Startup: Migration 013 failed (non-fatal): {e}")

        _benchmark_readonly_startup = os.environ.get("PITH_BENCHMARK_READONLY", "").lower() in ("true", "1")
        _scheduler_disabled = os.environ.get("PITH_DISABLE_BUILTIN_SCHEDULER", "0").lower() in ("1", "true", "yes")
        if _benchmark_readonly_startup or _scheduler_disabled:
            app.state.maintenance_state = "disabled"
            logger.info(
                "Startup: Maintenance scheduler SKIPPED (%s)",
                "PITH_BENCHMARK_READONLY" if _benchmark_readonly_startup else "PITH_DISABLE_BUILTIN_SCHEDULER",
            )
        else:
            try:
                from app.ops.maintenance_scheduler import start_maintenance_scheduler

                _sched_task = await start_maintenance_scheduler()
                if _sched_task is not None:
                    app.state.maintenance_scheduler = _sched_task
                    app.state.maintenance_state = "running"
                    logger.info("Startup: Built-in maintenance scheduler started")
                else:
                    app.state.maintenance_state = "disabled"
                    logger.info("Startup: Built-in maintenance scheduler disabled")
            except Exception as e:
                app.state.maintenance_state = "degraded"
                _set_degraded_reason(app.state.degraded_reason or "maintenance_scheduler_failed")
                logger.warning(f"Startup: Maintenance scheduler failed (degraded): {e}")
                app.state.startup_warnings.append(
                    {
                        "component": "maintenance_scheduler",
                        "severity": "degraded",
                        "message": f"Scheduler failed: {e}",
                    }
                )

        if getattr(app.state, "retrieval_state", "recovering") != "degraded":
            app.state.retrieval_state = "ready"
        if app.state.write_state != "blocked":
            app.state.write_state = "accepting"
        if _defer_embedding_warmup or _defer_reranker_warmup:
            app.state.ml_warmup_task = asyncio.create_task(
                _run_deferred_startup_warmups(
                    warm_embeddings=_defer_embedding_warmup,
                    warm_reranker=_defer_reranker_warmup,
                )
            )
            if _defer_embedding_warmup:
                logger.info("Startup: semantic embedding warmup scheduled after readiness (PERF-076)")
            if _defer_reranker_warmup:
                logger.info("Startup: reranker warmup scheduled after readiness (PERF-040)")
    except asyncio.CancelledError:
        logger.info("Startup: Deferred initialization cancelled")
        raise
    except Exception as e:
        _set_degraded_reason(app.state.degraded_reason or "deferred_startup_failed")
        logger.error(f"Deferred startup failed: {e}", exc_info=True)
        app.state.startup_warnings.append(
            {
                "component": "deferred_startup",
                "severity": "critical",
                "message": f"Deferred startup failed: {e}",
            }
        )


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    logger.info("Pith Server starting up...")
    logger.info("Server version: 1.1.0")
    app.state.process_state = "starting"
    app.state.write_state = "queued"
    app.state.retrieval_state = "recovering"
    app.state.maintenance_state = "disabled"
    app.state.degraded_reason = None

    try:
        ensure_safe_installed_runtime(invocation_path=os.getcwd())
    except RuntimeInstallGuardError as exc:
        logger.critical(f"STARTUP BLOCKED: {exc}")
        raise RuntimeError(str(exc)) from exc

    try:
        ensure_safe_installed_runtime(invocation_path=os.getcwd())
    except RuntimeInstallGuardError as exc:
        logger.critical(f"STARTUP BLOCKED: {exc}")
        raise RuntimeError(str(exc)) from exc

    # STABILITY-021: Track degraded startup state
    app.state.startup_warnings = []

    try:
        ensure_safe_installed_runtime(invocation_path=os.getcwd())
    except RuntimeInstallGuardError as exc:
        app.state.process_state = "blocked"
        app.state.write_state = "blocked"
        _set_degraded_reason("runtime_install_guard_blocked")
        logger.critical(f"STARTUP BLOCKED: {exc}")
        raise RuntimeError(str(exc)) from exc

    # PROFILE_FIX_SPEC Fix 3: Explicit profile/data dir logging on startup
    from app.core.profile import get_active_profile, resolve_data_dir

    _active_profile = get_active_profile()
    _data_dir = resolve_data_dir()
    logger.info(f"STARTUP: Profile={_active_profile}, DataDir={_data_dir}")

    # STABILITY-044 Fix 2b: Early exit if disabled flag is set.
    # Prevents KeepAlive respawns from doing expensive startup work
    # when the circuit breaker has fired.
    _pith_home = os.environ.get("PITH_HOME", os.path.expanduser("~/.pith"))
    _disabled_path = os.path.join(_pith_home, "server.disabled")
    if os.path.exists(_disabled_path):
        _reason = "unknown"
        try:
            with open(_disabled_path) as _df:
                _reason = _df.read().strip()[:200]
        except OSError:
            pass
        logger.warning(
            "STABILITY-044: server.disabled flag present (%s). "
            "Exiting early. Remove flag and restart to recover.",
            _reason,
        )
        raise RuntimeError(f"server.disabled: {_reason}")

    # BRAIN_LOCK_SPEC: Exclusive data directory lock — prevents rogue servers
    try:
        acquire_brain_lock(data_dir=str(_data_dir), port=PITH_PORT, pid=os.getpid())
    except BrainLockError as exc:
        logger.critical(f"STARTUP BLOCKED: {exc}")
        raise RuntimeError(str(exc)) from exc

    app.state.process_state = "running"
    import time as _startup_time

    app.state._start_monotonic = _startup_time.monotonic()  # STABILITY-037: uptime tracking
    _register_signal_handlers()
    app.state.startup_task = asyncio.create_task(_complete_startup_initialization())
    logger.info("Startup: Deferred initialization scheduled")


@app.on_event("shutdown")
async def shutdown_event():
    """Graceful shutdown — stop scheduler, wait for background tasks, log."""
    app.state.process_state = "stopping"

    # STABILITY-037 Fix 3a: Signal backend shutdown flag FIRST
    try:
        from app.storage.backend import get_backend

        backend = get_backend()
        if hasattr(backend, "begin_shutdown"):
            backend.begin_shutdown()
    except Exception as e:
        logger.warning(f"Shutdown: Backend shutdown signal failed: {e}")

    try:
        from app.session.required_context_cache import shutdown_required_context_cache

        shutdown_required_context_cache(wait=True)
        logger.info("Shutdown: Required context cache refresh executor stopped")
    except Exception as e:
        logger.warning(f"Shutdown: Required context cache shutdown failed: {e}")

    try:
        from app.storage import shutdown_association_index_refresh

        shutdown_association_index_refresh(wait=True)
        logger.info("Shutdown: Association index refresh executor stopped")
    except Exception as e:
        logger.warning(f"Shutdown: Association index refresh shutdown failed: {e}")

    # STABILITY-037 Fix 3b: Wait for autolearn executor to finish
    try:
        _executor = getattr(session_manager, "_learn_executor", None)
        if _executor is not None:
            logger.info("Shutdown: Waiting for autolearn executor (max 10s)...")
            _executor.shutdown(wait=True, cancel_futures=True)
            logger.info("Shutdown: Autolearn executor stopped")
    except Exception as e:
        logger.warning(f"Shutdown: Autolearn executor shutdown failed: {e}")

    _startup_task = getattr(app.state, "startup_task", None)
    if _startup_task and not _startup_task.done():
        _startup_task.cancel()
        logger.info("Shutdown: Cancelled deferred startup task")
    # STABILITY-030: Cancel in-flight index repair
    _repair_task = getattr(app.state, "index_repair_task", None)
    if _repair_task and not _repair_task.done():
        _repair_task.cancel()
        logger.info("Shutdown: Cancelled in-flight index repair")

    # MAINT-033: Stop maintenance scheduler
    try:
        from app.ops.maintenance_scheduler import stop_maintenance_scheduler

        await stop_maintenance_scheduler()
    except Exception as e:
        logger.warning(f"Shutdown: Scheduler stop failed: {e}")
    release_brain_lock()
    logger.info("Pith Server shutting down...")


def _register_signal_handlers():
    """OPS-075: Register SIGTERM/SIGINT handlers without overriding uvicorn shutdown."""
    import signal

    def _make_shutdown_handler(previous_handler):
        def _handle_shutdown_signal(signum, frame):
            import time as _sig_time
            import traceback as _tb

            sig_name = signal.Signals(signum).name
            caller_info = ""
            if frame is not None:
                caller_info = f" frame={frame.f_code.co_filename}:{frame.f_lineno} in {frame.f_code.co_name}"

            _uptime_s = _sig_time.monotonic() - getattr(app.state, "_start_monotonic", _sig_time.monotonic())
            _ppid = os.getppid()
            _source = "unknown"
            if _ppid == 1:
                _source = "launchd"
            elif _ppid == os.getpid():
                _source = "self"
            else:
                try:
                    import subprocess

                    _pname = subprocess.check_output(["ps", "-p", str(_ppid), "-o", "comm="], timeout=1).decode().strip()
                    _source = f"parent:{_pname}(pid={_ppid})"
                except Exception:
                    _source = f"pid={_ppid}"

            logger.warning(
                "OPS-075/STABILITY-037: Received %s (pid=%d, ppid=%d, source=%s, "
                "uptime=%.1fs%s) — initiating graceful shutdown",
                sig_name,
                os.getpid(),
                _ppid,
                _source,
                _uptime_s,
                caller_info,
            )
            try:
                stack = "".join(_tb.format_stack(frame, limit=5))
                logger.warning("OPS-075/T1-3: Stack at %s:\n%s", sig_name, stack)
            except Exception:
                pass
            app.state.process_state = "stopping"
            if callable(previous_handler):
                return previous_handler(signum, frame)
            return None

        _handle_shutdown_signal._pith_shutdown_handler = True
        _handle_shutdown_signal._pith_previous_handler = previous_handler
        return _handle_shutdown_signal

    def _install(signum):
        current_handler = signal.getsignal(signum)
        if getattr(current_handler, "_pith_shutdown_handler", False):
            return
        sig_name = signal.Signals(signum).name
        signal.signal(signum, _make_shutdown_handler(current_handler))
        logger.info("OPS-075: Signal handler registered (%s)", sig_name)

    # Only register in main thread (uvicorn workers may fork)
    import threading

    if threading.current_thread() is threading.main_thread():
        _install(signal.SIGTERM)
        _install(signal.SIGINT)


def _get_feature_flags() -> dict:
    """TOOLING-034: Expose feature flag state for health/diagnostics.
    OPS-106: Now returns runtime-resolved values (env var overrides),
    not just config.py defaults. Uses get_feature_flag() per flag."""
    from app.core.config import FEATURE_FLAGS, get_feature_flag

    return {name: get_feature_flag(name, default) for name, default in FEATURE_FLAGS.items()}


def _read_only_aggregates_enabled() -> bool:
    from app.core.config import get_feature_flag

    return get_feature_flag("READ_ONLY_AGGREGATES_ENABLED", False)


def _read_only_aggregates_fallback_allowed() -> bool:
    """Parent spec §15 safety flag: if True, a failing read_snapshot_db call
    falls back to the legacy connection path instead of surfacing HTTP 500.
    Default True so a transient SQLite hiccup does not degrade /pith_stats
    or /pith_search_suggestions. Set False only in hardening/assertion mode.
    """
    from app.core.config import get_feature_flag

    return get_feature_flag("READ_ONLY_AGGREGATES_FALLBACK_ALLOWED", True)


def _cascade_alert_threshold() -> int:
    """NITS-001: Lazy-load configurable cascade alert threshold."""
    from app.core.config import CASCADE_ALERT_THRESHOLD

    return CASCADE_ALERT_THRESHOLD


def _circuit_breaker_alert_threshold() -> int:
    """MONITOR-072: Lazy-load configurable circuit breaker alert threshold."""
    from app.core.config import CIRCUIT_BREAKER_ALERT_THRESHOLD

    return CIRCUIT_BREAKER_ALERT_THRESHOLD


def _get_contradiction_signal_backlog(conn=None) -> dict:
    """MONITOR-027: Check for unprocessed GRAPH_CONTRADICTION_SIGNAL events."""
    try:
        if conn is not None:
            count = conn.execute(
                "SELECT COUNT(*) FROM governance_events WHERE event_type = 'GRAPH_CONTRADICTION_SIGNAL'"
            ).fetchone()[0]
            return {"unprocessed_count": count, "alert": count > 1000}

        from app.storage.connection import read_snapshot_db

        with read_snapshot_db("health_contradiction_signal_backlog", allow_fallback=False) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM governance_events WHERE event_type = 'GRAPH_CONTRADICTION_SIGNAL'"
            ).fetchone()[0]
        return {"unprocessed_count": count, "alert": count > 1000}
    except Exception as exc:
        return {
            "unprocessed_count": -1,
            "alert": False,
            "degraded": True,
            "reason": _safe_error(exc),
        }


def _get_pricing_health() -> dict:
    """MONITOR-028: Pricing metrics for /health endpoint."""
    try:
        from app.api.pricing import conversation_meter

        status = conversation_meter.get_status()
        return {
            "budget_zone": status.get("budget_zone", "unknown"),
            "turns_used": status.get("turns_used", 0),
            "daily_limit": status.get("daily_limit", 0),
            "capped_at": status.get("capped_at"),
        }
    except Exception:
        return {"error": "pricing_unavailable"}


@app.get("/healthz")
def healthz() -> dict:
    from app.core.profile import resolve_data_dir

    data_dir = Path(resolve_data_dir())
    return {
        "status": "healthy"
        if getattr(app.state, "process_state", "starting") in {"starting", "running"}
        else "stopping",
        "service": "pith",
        "timestamp": _utc_now_iso(),
        "version": "1.1.0",
        "git_commit": _GIT_COMMIT,
        "git_branch": _GIT_BRANCH,
        "git_dirty": _GIT_DIRTY,
        "benchmark_mode": os.environ.get("PITH_BENCHMARK_MODE", "").lower() in ("true", "1"),
        "benchmark_readonly": os.environ.get("PITH_BENCHMARK_READONLY", "").lower() in ("true", "1"),
        "data_dir": str(data_dir),
        "db_path": str(data_dir / "pith.db"),
        "process_state": getattr(app.state, "process_state", "starting"),
    }


@app.get("/readyz")
def readyz(request: Request) -> dict:
    ready = _build_ready_state()
    ready["auth_status"] = _build_auth_status(request)
    return ready


@app.get("/runtime/answer_path_policy", dependencies=[Depends(verify_api_key)])
def answer_path_policy_status() -> AnswerPathPolicyResponse:
    from app.session.answer_path_policy import get_answer_path_policy

    snapshot = get_answer_path_policy().snapshot()
    return _answer_path_policy_response(snapshot)


@app.post("/runtime/answer_path_policy", dependencies=[Depends(verify_api_key)])
def update_answer_path_policy(
    request: Request,
    body: AnswerPathPolicyUpdateRequest,
) -> AnswerPathPolicyResponse:
    from app.session.answer_path_policy import (
        AnswerPathPolicyError,
        get_answer_path_policy,
    )

    _require_local_operator(request)
    ready = _build_ready_state()
    if ready["process_state"] != "running" or ready["retrieval_state"] == "recovering":
        reason = (
            "process_not_running"
            if ready["process_state"] != "running"
            else "retrieval_recovering"
        )
        _record_answer_path_policy_metric(
            "answer_path_policy_reject_total",
            {"reason": reason, "source": "runtime_api"},
        )
        raise HTTPException(
            status_code=503,
            detail="Answer-path runtime policy updates require a running process and initialized retrieval",
            headers={"Retry-After": "10"},
        )

    try:
        snapshot = get_answer_path_policy().set_runtime(
            observe_only=body.observe_only,
            enforcement_enabled=body.enforcement_enabled,
            ttl_seconds=body.ttl_seconds,
            enforce_modes=body.enforce_modes,
            source=body.source,
        )
    except AnswerPathPolicyError as exc:
        _record_answer_path_policy_metric(
            "answer_path_policy_reject_total",
            {"reason": "invalid_request", "source": "runtime_api"},
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _record_answer_path_policy_metric(
        "answer_path_policy_update_total",
        {"state": snapshot.state, "source": snapshot.source},
    )
    logger.info(
        "ANSWER-PATH-POLICY: update state=%s source=%s generation=%s expires_at=%s",
        snapshot.state,
        snapshot.source,
        snapshot.generation,
        snapshot.expires_at,
    )
    return _answer_path_policy_response(snapshot)


@app.post("/runtime/answer_path_policy/reset", dependencies=[Depends(verify_api_key)])
def reset_answer_path_policy(
    request: Request,
    body: AnswerPathPolicyResetRequest | None = None,
) -> AnswerPathPolicyResponse:
    from app.session.answer_path_policy import get_answer_path_policy

    _require_local_operator(request)
    source = body.source if body is not None else "runtime_api"
    snapshot = get_answer_path_policy().reset(source=source)
    _record_answer_path_policy_metric(
        "answer_path_policy_reset_total",
        {"state": snapshot.state, "source": snapshot.source},
    )
    logger.info(
        "ANSWER-PATH-POLICY: reset state=%s source=%s generation=%s",
        snapshot.state,
        snapshot.source,
        snapshot.generation,
    )
    return _answer_path_policy_response(snapshot)


@app.get("/health")
def health_check(request: Request):
    with request_db_scope("health"):
        try:
            from app.core.profile import get_active_profile, resolve_data_dir

            ready = _build_ready_state()
            external_maintenance = _build_external_maintenance_health()
            built_in_state = ready["maintenance_state"]
            effective_state, maintenance_mode = _effective_maintenance_state(
                built_in_state,
                external_maintenance,
            )
            ready["built_in_maintenance_state"] = built_in_state
            ready["maintenance_state"] = effective_state
            ready["maintenance_mode"] = maintenance_mode
            ready["external_maintenance"] = external_maintenance
            ready["profile"] = get_active_profile()
            data_dir = Path(resolve_data_dir())
            ready["data_dir"] = str(data_dir)
            ready["db_path"] = str(data_dir / "pith.db")
            ready["benchmark_mode"] = os.environ.get("PITH_BENCHMARK_MODE", "").lower() in ("true", "1")
            ready["benchmark_readonly"] = os.environ.get("PITH_BENCHMARK_READONLY", "").lower() in ("true", "1")
            ready["auth_status"] = _build_auth_status(request)
            required_context_cache = _build_required_context_cache_health()
            defer_db_metrics = _should_defer_health_db_metrics(ready)
            defer_reason = _health_metrics_defer_reason(ready) if defer_db_metrics else None
            conversation_turn_latency = (
                _build_unavailable_health_metric(defer_reason)
                if defer_db_metrics and defer_reason is not None
                else _build_conversation_turn_latency_health()
            )
            lifecycle_jobs = (
                _build_unavailable_health_metric(defer_reason)
                if defer_db_metrics and defer_reason is not None
                else _build_lifecycle_jobs_health()
            )
            ready["components"] = {
                "storage": "ok",
                "retrieval_index": ready["retrieval_state"],
                "activation_engine": "ok",
                "goal_engine": "ok",
                "curiosity_engine": "ok",
                "maintenance_scheduler": effective_state,
                "maintenance_scheduler_builtin": built_in_state,
                "maintenance_scheduler_external": external_maintenance["scheduler_state"],
                "required_context_cache": required_context_cache.get("status", "unknown"),
                "conversation_turn_latency": conversation_turn_latency.get("status", "unknown"),
                "lifecycle_jobs": lifecycle_jobs.get("status", "unknown"),
            }
            if defer_db_metrics:
                contradiction_signals = _build_unavailable_health_metric(defer_reason)
                indexed_concepts = None
                session_learn_queue = _build_deferred_session_learn_queue_health(defer_reason)
            else:
                if _read_only_aggregates_enabled():
                    try:
                        with read_snapshot_db("health") as conn:
                            contradiction_signals = _get_contradiction_signal_backlog(conn=conn)
                    except Exception as snap_err:
                        if not _read_only_aggregates_fallback_allowed():
                            raise
                        logger.warning(
                            "read-only snapshot failed for /health, falling back to legacy path: %s",
                            snap_err,
                        )
                        contradiction_signals = _get_contradiction_signal_backlog()
                else:
                    contradiction_signals = _get_contradiction_signal_backlog()
                indexed_concepts = retrieval_engine.index.document_count
                session_learn_queue = _build_session_learn_queue_health()
            ready["metrics"] = {
                "indexed_concepts": indexed_concepts,
                "pricing": _get_pricing_health(),
                "contradiction_signals": contradiction_signals,
                "required_context_cache": required_context_cache,
                "conversation_turn_latency": conversation_turn_latency,
            }
            ready["metrics"]["session_learn_queue"] = session_learn_queue
            ready["metrics"]["lifecycle_jobs"] = lifecycle_jobs
            ready["components"]["session_learn_queue"] = session_learn_queue.get("status", "unknown")
            ready["feature_flags"] = _get_feature_flags()
            return ready

        except Exception as e:
            logger.error(f"Health check failed: {e}", exc_info=True)
            return {
                "status": "unhealthy",
                "service": "pith",
                "timestamp": _utc_now_iso(),
                "version": "1.1.0",
                "error": _safe_error(e),
            }


@app.get("/pith_stats")
def pith_stats(detail: str = "fast"):
    """Get overall pith statistics via aggregate SQL (no N+1 loop)."""
    with request_db_scope("pith_stats"):
        if detail not in {"fast", "full"}:
            raise HTTPException(status_code=400, detail="detail must be 'fast' or 'full'")
        if detail == "fast":
            return _build_pith_stats_fast_cached()

        from app.storage import get_pith_stats_aggregates

        # DEBT-145: Forward full aggregate dict instead of cherry-picking fields.
        # New fields from get_pith_stats_aggregates() are auto-surfaced.
        if _read_only_aggregates_enabled():
            try:
                with read_snapshot_db("pith_stats") as conn:
                    agg = get_pith_stats_aggregates(conn=conn)
                    agg["associations"] = count_associations(conn=conn)
            except Exception as snap_err:
                if not _read_only_aggregates_fallback_allowed():
                    raise
                logger.warning(
                    "read-only snapshot failed for /pith_stats, falling back to legacy path: %s",
                    snap_err,
                )
                agg = get_pith_stats_aggregates()
                agg["associations"] = count_associations()
        else:
            agg = get_pith_stats_aggregates()
            agg["associations"] = count_associations()
        agg["pending_questions"] = len(question_queue.get_questions(limit=1000))
        # BLIND-001: Surface computed blind spots in stats dashboard
        try:
            from app.session.self_model import SelfModelManager

            _blind_spots = SelfModelManager().get_blind_spots()
            agg["blind_spots"] = [
                {
                    "description": bs.description if isinstance(bs.description, str) else str(bs),
                    "severity": getattr(bs, "severity", "moderate"),
                }
                for bs in _blind_spots
            ]
        except Exception:
            agg["blind_spots"] = []
        agg.setdefault("mode", "full")
        agg.setdefault("generated_at", _utc_now_iso())
        agg.setdefault("cache_age_ms", 0)
        agg.setdefault("partial", False)
        agg.setdefault("section_errors", {})
        agg.setdefault("section_timings_ms", {})
        return agg


def _build_pith_stats_fast_payload() -> dict:
    """Build the fast stats payload and leave cache policy to the caller."""
    from app.storage import get_pith_stats_fast

    section_errors: dict[str, str] = {}
    payload = get_pith_stats_fast()
    try:
        payload["pending_questions"] = len(question_queue.get_questions(limit=1000))
    except Exception as err:
        payload["pending_questions"] = 0
        section_errors["pending_questions"] = _safe_error(err)
    payload.update(
        {
            "mode": "fast",
            "generated_at": _utc_now_iso(),
            "cache_age_ms": 0,
            "partial": bool(section_errors),
            "stale": False,
            "section_errors": section_errors,
        }
    )
    return _annotate_fast_stats_freshness(payload, "fresh", "ok", "cache_fresh", cache_age_ms=0)


def _fast_stats_max_stale_ms() -> int:
    return int(max(0.0, _FAST_STATS_MAX_STALE_S) * 1000)


def _fast_stats_observability_snapshot() -> dict:
    with _FAST_STATS_REFRESH_LOCK:
        return dict(_FAST_STATS_OBSERVABILITY)


def _annotate_fast_stats_freshness(
    payload: dict,
    freshness_state: str,
    freshness_status: str,
    freshness_reason: str,
    *,
    cache_age_ms: int | None = None,
) -> dict:
    annotated = dict(payload)
    if cache_age_ms is not None:
        annotated["cache_age_ms"] = cache_age_ms
    annotated.update(
        {
            "freshness_state": freshness_state,
            "freshness_status": freshness_status,
            "freshness_reason": freshness_reason,
            "max_stale_ms": _fast_stats_max_stale_ms(),
            "refresh_observability": _fast_stats_observability_snapshot(),
        }
    )
    return annotated


def _classify_stale_fast_stats(cache_age_ms: int, refresh_error: str | None) -> tuple[str, str, str]:
    if refresh_error:
        return "refresh_failed", "degraded", refresh_error
    if cache_age_ms > _fast_stats_max_stale_ms():
        return "stale_beyond_slo", "degraded", "cache_age_exceeded_max_stale"
    return "refreshing_bounded", "warning", "refresh_in_progress"


def _record_fast_stats_refresh_observation(success: bool, duration_ms: float, error: str | None = None) -> None:
    global _FAST_STATS_REFRESH_ERROR
    with _FAST_STATS_REFRESH_LOCK:
        if success:
            _FAST_STATS_OBSERVABILITY["refresh_success_count"] = (
                int(_FAST_STATS_OBSERVABILITY["refresh_success_count"] or 0) + 1
            )
            _FAST_STATS_OBSERVABILITY["last_refresh_error"] = None
            _FAST_STATS_REFRESH_ERROR = None
        else:
            _FAST_STATS_OBSERVABILITY["refresh_failure_count"] = (
                int(_FAST_STATS_OBSERVABILITY["refresh_failure_count"] or 0) + 1
            )
            _FAST_STATS_OBSERVABILITY["last_refresh_error"] = error or "unknown_refresh_error"
            _FAST_STATS_REFRESH_ERROR = error or "unknown_refresh_error"
        _FAST_STATS_OBSERVABILITY["last_refresh_duration_ms"] = round(duration_ms, 2)


def _record_fast_stats_stale_served() -> None:
    with _FAST_STATS_REFRESH_LOCK:
        _FAST_STATS_OBSERVABILITY["stale_served_count"] = (
            int(_FAST_STATS_OBSERVABILITY["stale_served_count"] or 0) + 1
        )


def _refresh_fast_stats_cache() -> dict:
    """Refresh fast stats cache from a worker so callers can stay bounded."""
    global _FAST_STATS_CACHE, _FAST_STATS_REFRESH_ERROR
    started = time.perf_counter()
    acquired = _DIAGNOSTIC_ENDPOINT_SEMAPHORE.acquire(timeout=_DIAGNOSTIC_ENDPOINT_SLOT_TIMEOUT_S)
    if not acquired:
        duration_ms = (time.perf_counter() - started) * 1000
        _record_fast_stats_refresh_observation(False, duration_ms, "diagnostic_slot_unavailable")
        raise TimeoutError("diagnostic_slot_unavailable")
    try:
        payload = _build_pith_stats_fast_payload()
        duration_ms = (time.perf_counter() - started) * 1000
        _record_fast_stats_refresh_observation(True, duration_ms)
        payload = _annotate_fast_stats_freshness(payload, "fresh", "ok", "cache_fresh", cache_age_ms=0)
        with _FAST_STATS_CACHE_LOCK:
            _FAST_STATS_CACHE = (time.monotonic(), dict(payload))
        return payload
    except Exception as err:
        duration_ms = (time.perf_counter() - started) * 1000
        _record_fast_stats_refresh_observation(False, duration_ms, _safe_error(err))
        raise
    finally:
        _DIAGNOSTIC_ENDPOINT_SEMAPHORE.release()


def _record_fast_stats_refresh_result(future: concurrent.futures.Future) -> None:
    global _FAST_STATS_REFRESH_ERROR
    try:
        exc = future.exception()
    except concurrent.futures.CancelledError:
        exc = None
    if exc is not None:
        with _FAST_STATS_REFRESH_LOCK:
            _FAST_STATS_REFRESH_ERROR = _safe_error(exc)


def _ensure_fast_stats_refresh_started() -> concurrent.futures.Future:
    global _FAST_STATS_REFRESH_FUTURE
    add_callback = False
    with _FAST_STATS_REFRESH_LOCK:
        if _FAST_STATS_REFRESH_FUTURE is None or _FAST_STATS_REFRESH_FUTURE.done():
            _FAST_STATS_REFRESH_FUTURE = _DIAGNOSTIC_REFRESH_EXECUTOR.submit(_refresh_fast_stats_cache)
            add_callback = True
        future = _FAST_STATS_REFRESH_FUTURE
    if add_callback:
        future.add_done_callback(_record_fast_stats_refresh_result)
    return future


def _build_pith_stats_fast_cached() -> dict:
    """Build/coalesce default stats so status surfaces do not stampede SQLite."""
    now = time.monotonic()
    with _FAST_STATS_CACHE_LOCK:
        if _FAST_STATS_CACHE is not None:
            cached_at, cached_payload = _FAST_STATS_CACHE
            cache_age_ms = int((now - cached_at) * 1000)
            if cache_age_ms <= int(_FAST_STATS_CACHE_TTL_S * 1000):
                payload = dict(cached_payload)
                return _annotate_fast_stats_freshness(payload, "fresh", "ok", "cache_fresh", cache_age_ms=cache_age_ms)
            stale_cache = (cached_at, dict(cached_payload))
        else:
            stale_cache = None

    refresh_future = _ensure_fast_stats_refresh_started()
    if stale_cache is not None:
        with _FAST_STATS_REFRESH_LOCK:
            refresh_error = _FAST_STATS_REFRESH_ERROR
        return _build_stale_diagnostic_payload(stale_cache, "fast_stats", refresh_error or "refresh_in_progress")

    try:
        return refresh_future.result(timeout=max(0.001, _FAST_STATS_COLD_BUDGET_S))
    except concurrent.futures.TimeoutError:
        return _build_empty_fast_stats_payload("budget_exceeded")
    except Exception as err:
        return _build_empty_fast_stats_payload(_safe_error(err))


def _build_stale_diagnostic_payload(cache_entry: tuple[float, dict], section: str, error: str) -> dict:
    cached_at, cached_payload = cache_entry
    payload = dict(cached_payload)
    errors = dict(payload.get("section_errors") or {})
    errors[section] = error
    cache_age_ms = int((time.monotonic() - cached_at) * 1000)
    state, status, reason = _classify_stale_fast_stats(cache_age_ms, error if error != "refresh_in_progress" else None)
    _record_fast_stats_stale_served()
    payload.update(
        {
            "cache_age_ms": cache_age_ms,
            "partial": True,
            "stale": True,
            "section_errors": errors,
        }
    )
    return _annotate_fast_stats_freshness(payload, state, status, reason, cache_age_ms=cache_age_ms)


def _build_empty_fast_stats_payload(error: str) -> dict:
    if error == "budget_exceeded":
        state, status, reason = "cold_budget_exceeded", "warning", "budget_exceeded"
    else:
        state, status, reason = "cold_failed", "degraded", error
    payload = {
        "mode": "fast",
        "generated_at": _utc_now_iso(),
        "cache_age_ms": 0,
        "partial": True,
        "stale": False,
        "section_errors": {"fast_stats": error},
        "total_concepts": 0,
        "avg_confidence": 0.0,
        "avg_stability": 0.0,
        "knowledge_areas": 0,
        "associations": 0,
        "data_quality": {"null_timestamps": 0, "bad_json": 0},
        "pending_questions": 0,
    }
    return _annotate_fast_stats_freshness(payload, state, status, reason, cache_age_ms=0)


def _build_empty_fast_health_payload(error: str) -> dict:
    return {
        "mode": "fast",
        "generated_at": _utc_now_iso(),
        "cache_age_ms": 0,
        "partial": True,
        "stale": False,
        "section_errors": {"fast_health": error},
        "status": "partial",
        "health_score": 0.0,
        "total_concepts": 0,
        "avg_confidence": 0.0,
        "avg_stability": 0.0,
    }


def _build_pith_health_fast_cached() -> dict:
    """Build/coalesce default health so diagnostic readers remain bounded."""
    global _FAST_HEALTH_CACHE
    now = time.monotonic()
    with _FAST_HEALTH_CACHE_LOCK:
        if _FAST_HEALTH_CACHE is not None:
            cached_at, cached_payload = _FAST_HEALTH_CACHE
            cache_age_ms = int((now - cached_at) * 1000)
            if cache_age_ms <= int(_FAST_HEALTH_CACHE_TTL_S * 1000):
                payload = dict(cached_payload)
                payload["cache_age_ms"] = cache_age_ms
                return payload
            stale_cache = (cached_at, dict(cached_payload))
        else:
            stale_cache = None

    acquired = _DIAGNOSTIC_ENDPOINT_SEMAPHORE.acquire(timeout=_DIAGNOSTIC_ENDPOINT_SLOT_TIMEOUT_S)
    if not acquired:
        if stale_cache is not None:
            return _build_stale_diagnostic_payload(stale_cache, "fast_health", "diagnostic_slot_unavailable")
        return _build_empty_fast_health_payload("diagnostic_slot_unavailable")

    try:
        from app.storage import get_pith_health_fast

        health = get_pith_health_fast()
        health.update(
            {
                "mode": "fast",
                "generated_at": _utc_now_iso(),
                "cache_age_ms": 0,
                "partial": False,
                "stale": False,
                "section_errors": {},
            }
        )
        with _FAST_HEALTH_CACHE_LOCK:
            _FAST_HEALTH_CACHE = (time.monotonic(), dict(health))
        return health
    except Exception as err:
        logger.error("Fast pith health failed: %s", err, exc_info=True)
        if stale_cache is not None:
            return _build_stale_diagnostic_payload(stale_cache, "fast_health", _safe_error(err))
        return _build_empty_fast_health_payload(_safe_error(err))
    finally:
        _DIAGNOSTIC_ENDPOINT_SEMAPHORE.release()


def _build_empty_knowledge_areas_payload(error: str) -> dict:
    return {
        "total": 0,
        "areas": [],
        "mode": "fast",
        "generated_at": _utc_now_iso(),
        "cache_age_ms": 0,
        "partial": True,
        "stale": False,
        "section_errors": {"knowledge_areas": error},
    }


def _build_knowledge_areas_cached() -> dict:
    """Return knowledge-area summaries without materializing every concept."""
    global _KNOWLEDGE_AREAS_CACHE
    now = time.monotonic()
    with _KNOWLEDGE_AREAS_CACHE_LOCK:
        if _KNOWLEDGE_AREAS_CACHE is not None:
            cached_at, cached_payload = _KNOWLEDGE_AREAS_CACHE
            cache_age_ms = int((now - cached_at) * 1000)
            if cache_age_ms <= int(_KNOWLEDGE_AREAS_CACHE_TTL_S * 1000):
                payload = dict(cached_payload)
                payload["cache_age_ms"] = cache_age_ms
                return payload
            stale_cache = (cached_at, dict(cached_payload))
        else:
            stale_cache = None

    acquired = _DIAGNOSTIC_ENDPOINT_SEMAPHORE.acquire(timeout=_DIAGNOSTIC_ENDPOINT_SLOT_TIMEOUT_S)
    if not acquired:
        if stale_cache is not None:
            return _build_stale_diagnostic_payload(
                stale_cache,
                "knowledge_areas",
                "diagnostic_slot_unavailable",
            )
        return _build_empty_knowledge_areas_payload("diagnostic_slot_unavailable")

    try:
        from app.storage import list_knowledge_area_summaries

        areas = list_knowledge_area_summaries()
        payload = {
            "total": len(areas),
            "areas": areas,
            "mode": "fast",
            "generated_at": _utc_now_iso(),
            "cache_age_ms": 0,
            "partial": False,
            "stale": False,
            "section_errors": {},
        }
        with _KNOWLEDGE_AREAS_CACHE_LOCK:
            _KNOWLEDGE_AREAS_CACHE = (time.monotonic(), dict(payload))
        return payload
    except Exception as err:
        logger.error("Fast knowledge area listing failed: %s", err, exc_info=True)
        if stale_cache is not None:
            return _build_stale_diagnostic_payload(stale_cache, "knowledge_areas", _safe_error(err))
        return _build_empty_knowledge_areas_payload(_safe_error(err))
    finally:
        _DIAGNOSTIC_ENDPOINT_SEMAPHORE.release()


def _reset_fast_stats_cache_for_tests() -> None:
    global _FAST_HEALTH_CACHE, _FAST_STATS_CACHE, _FAST_STATS_REFRESH_ERROR, _FAST_STATS_REFRESH_FUTURE, _KNOWLEDGE_AREAS_CACHE
    with _FAST_STATS_CACHE_LOCK:
        _FAST_STATS_CACHE = None
    with _FAST_STATS_REFRESH_LOCK:
        if _FAST_STATS_REFRESH_FUTURE is not None and not _FAST_STATS_REFRESH_FUTURE.done():
            _FAST_STATS_REFRESH_FUTURE.cancel()
        _FAST_STATS_REFRESH_FUTURE = None
        _FAST_STATS_REFRESH_ERROR = None
        _FAST_STATS_OBSERVABILITY.update(
            {
                "refresh_success_count": 0,
                "refresh_failure_count": 0,
                "stale_served_count": 0,
                "last_refresh_duration_ms": None,
                "last_refresh_error": None,
            }
        )
    with _FAST_HEALTH_CACHE_LOCK:
        _FAST_HEALTH_CACHE = None
    with _KNOWLEDGE_AREAS_CACHE_LOCK:
        _KNOWLEDGE_AREAS_CACHE = None


@app.get("/learning_metrics")
def learning_metrics():
    """Learning performance dashboard — monitors extraction pipeline health.

    Key metrics for routine monitoring:
    - type_distribution: L3+ vs L1 ratio (target: >20% L3+)
    - daily_throughput: concepts created per day (last 7 days)
    - budget_utilization: daily budget usage pattern
    - pipeline_health: rejection rates, dedup rates
    """
    from app.storage import _db

    # B7 fix: Use _db() context manager instead of raw sqlite3.connect().
    # The old code opened a second connection bypassing _get_connection(),
    # which meant: (a) no _db_lock serialization, (b) default synchronous=FULL
    # instead of matching the app, (c) a concurrent reader outside the
    # managed connection — all amplifying WAL corruption risk.
    with _db() as conn:
        c = conn.cursor()

        # Type distribution (all time)
        c.execute("SELECT concept_type, COUNT(*) FROM concepts GROUP BY concept_type ORDER BY COUNT(*) DESC")
        type_dist = {row[0] or "untyped": row[1] for row in c.fetchall()}
        total = sum(type_dist.values())
        abstract_types = {"principle", "method", "heuristic", "cognitive_strategy", "system_model"}
        l3_count = sum(v for k, v in type_dist.items() if k in abstract_types)

        # Daily throughput (last 7 days)
        c.execute("""SELECT DATE(created_at) as day, COUNT(*),
                     SUM(CASE WHEN concept_type IN ('principle','method','heuristic','cognitive_strategy','system_model') THEN 1 ELSE 0 END) as l3_count
                     FROM concepts WHERE created_at > datetime('now', '-7 days')
                     GROUP BY day ORDER BY day""")
        daily = [{"date": r[0], "total": r[1], "l3_plus": r[2]} for r in c.fetchall()]

        # Recent 24h type breakdown
        c.execute("""SELECT concept_type, COUNT(*) FROM concepts
                     WHERE created_at > datetime('now', '-1 day')
                     GROUP BY concept_type ORDER BY COUNT(*) DESC""")
        last_24h = {row[0] or "untyped": row[1] for row in c.fetchall()}
        total_24h = sum(last_24h.values())
        l3_24h = sum(v for k, v in last_24h.items() if k in abstract_types)

        # Session manager budget info
        budget_remaining = session_manager._check_daily_budget()
        budget_total = session_manager.DAILY_BUDGET
        try:
            from app.storage.turn_ingestion import get_ingestion_capture_summary

            ingestion_capture = get_ingestion_capture_summary(conn)
            ingestion_capture_error = None
        except Exception as e:
            ingestion_capture = None
            ingestion_capture_error = str(e)

    # MONITOR-001: Derived observability signals from MetricsCollector
    import json
    from datetime import timedelta

    from app.api.pricing import conversation_meter
    from app.ops.metrics import metrics as _lm_metrics

    _1h_ago = (_utc_now() - timedelta(hours=1)).isoformat()
    _24h_ago = (_utc_now() - timedelta(hours=24)).isoformat()

    # Learning velocity (concepts created per minute, last hour)
    velocity = _lm_metrics.query_rate("learn_pipeline_created", window_minutes=60)

    # Dedup efficiency (what % of pipeline input becomes new concepts?)
    _created_1h = _lm_metrics.query_count("learn_pipeline_created", since=_1h_ago)
    _skipped_1h = _lm_metrics.query_count("learn_pipeline_skipped", since=_1h_ago)
    _evolved_1h = _lm_metrics.query_count("learn_pipeline_evolved", since=_1h_ago)
    _total_1h = _created_1h + _skipped_1h + _evolved_1h
    dedup_efficiency = round(_created_1h / _total_1h * 100, 1) if _total_1h else 0

    # KA growth distribution (from learn_concept_created labels)
    _ka_rows = _lm_metrics.query("learn_concept_created", since=_1h_ago, limit=5000)
    ka_dist: dict[str, int] = {}
    for row in _ka_rows:
        _labels_raw = row.get("labels", "{}")
        _labels = json.loads(_labels_raw) if isinstance(_labels_raw, str) else (_labels_raw or {})
        ka = _labels.get("ka", "unknown")
        ka_dist[ka] = ka_dist.get(ka, 0) + int(row.get("value", 0))
    ka_dist = dict(sorted(ka_dist.items(), key=lambda x: -x[1]))

    # Budget zone info (from ConversationMeter)
    meter_status = conversation_meter.get_status()

    # Zone transitions (last 24h)
    _transitions = _lm_metrics.query("budget_zone_transition", since=_24h_ago, limit=100)

    # Pipeline health (latency)
    pipeline_latency = _lm_metrics.query_aggregate("learn_pipeline_latency_ms", since=_1h_ago)
    budget_trend = _lm_metrics.query("learn_budget_remaining", since=_1h_ago, limit=100)

    # --- TIER2-DAY1: concept_type correction observability ---
    from app.core.models import CONCEPT_TYPES_SET

    _ct_correction_1h = _lm_metrics.query_count("concept_type_correction", since=_1h_ago)
    _ct_correction_24h = _lm_metrics.query_count("concept_type_correction", since=_24h_ago)
    # Correction breakdown by from→to labels
    _ct_correction_rows = _lm_metrics.query("concept_type_correction", since=_24h_ago, limit=5000)
    _ct_corrections_by_pair: dict[str, int] = {}
    for _ctr in _ct_correction_rows:
        _ctr_labels_raw = _ctr.get("labels", "{}")
        _ctr_labels = json.loads(_ctr_labels_raw) if isinstance(_ctr_labels_raw, str) else (_ctr_labels_raw or {})
        _pair_key = f"{_ctr_labels.get('from', '?')}→{_ctr_labels.get('to', '?')}"
        _ct_corrections_by_pair[_pair_key] = _ct_corrections_by_pair.get(_pair_key, 0) + 1
    _leakage_count = sum(v for k, v in type_dist.items() if k not in CONCEPT_TYPES_SET and k != "untyped")

    return {
        "type_distribution": {
            "all_time": type_dist,
            "total": total,
            "l3_plus_count": l3_count,
            "l3_plus_pct": round(l3_count / total * 100, 1) if total else 0,
            "target_pct": 20.0,
            "status": "healthy" if l3_count / total * 100 >= 20 else "below_target",
        },
        # TIER2-DAY1: concept_type correction rates and leakage detection
        "concept_type_health": {
            "correction_rate_1h": _ct_correction_1h,
            "correction_rate_24h": _ct_correction_24h,
            "corrections_by_pair_24h": _ct_corrections_by_pair,
            "leakage_count": _leakage_count,
            "leakage_status": "clean" if _leakage_count == 0 else f"ALERT: {_leakage_count} non-canonical types",
        },
        "last_24h": {
            "types": last_24h,
            "total": total_24h,
            "l3_plus_count": l3_24h,
            "l3_plus_pct": round(l3_24h / total_24h * 100, 1) if total_24h else 0,
        },
        "daily_throughput": daily,
        "budget": {
            "daily_limit": budget_total,
            "remaining_today": budget_remaining,
            "used_today": budget_total - budget_remaining,
            "utilization_pct": round((budget_total - budget_remaining) / budget_total * 100, 1),
        },
        # --- MONITOR-001: New observability signals ---
        "velocity": {
            "concepts_per_min": velocity,
            "dedup_efficiency_pct": dedup_efficiency,
            "created_1h": _created_1h,
            "evolved_1h": _evolved_1h,
            "skipped_1h": _skipped_1h,
        },
        "ka_growth": {
            "last_hour": ka_dist,
            "top_ka": next(iter(ka_dist), None),
            "ka_count": len(ka_dist),
        },
        "budget_health": {
            "zone": meter_status["budget_zone"],
            "tier": meter_status["tier"],
            "capped_at": meter_status.get("capped_at"),
            "zone_transitions_24h": len(_transitions),
            "transitions": _transitions[:10],
            "budget_trend": [{"time": r["timestamp"], "remaining": r["value"]} for r in budget_trend[-20:]],
        },
        "pipeline_health": {
            "latency_ms": pipeline_latency,
            "status": "healthy" if pipeline_latency.get("p95", 0) < 5000 else "degraded",
        },
        "ingestion_capture": ingestion_capture,
        "ingestion_capture_error": ingestion_capture_error,
    }


@app.post("/pith_search")
def pith_search(query: SearchQuery):
    """Search for concepts using RAG retrieval."""
    try:
        results = retrieval_engine.search(query)

        # MATURITY-001: Filter quarantined/discarded from external API results
        results = [r for r in results if (r.maturity or "ESTABLISHED") not in _BLOCKED_MATURITIES]
        # C2: Enrich with ambient_context from top result's associations
        ambient = []
        if results:
            top_id = results[0].concept_id
            result_ids = {r.concept_id for r in results}
            ambient = select_ambient_concepts(top_id, query=query.query, exclude_ids=result_ids)

        return {
            "results": [r.model_dump() if hasattr(r, "model_dump") else r for r in results],
            "ambient_context": {"related": ambient} if ambient else {},
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=_safe_error(e))


@app.get("/pith_get_concept")
def pith_get_concept(concept_id: str, version: str = "latest"):
    """Get concept by ID and version."""
    concept = load_concept(concept_id, version)
    if not concept:
        raise HTTPException(status_code=404, detail=f"Concept {concept_id} not found")

    if version == "all":
        return {"concepts": concept}  # Returns list

    # C2: Enrich with ambient_context from association neighbors
    ambient = select_ambient_concepts(concept_id, query=concept.summary)

    result = concept.model_dump() if hasattr(concept, "model_dump") else concept
    if ambient:
        result["ambient_context"] = {"related": ambient}
    return result


@app.post("/pith_propose_concept", dependencies=[Depends(verify_api_key)])
def pith_propose_concept(proposal: ConceptProposal):
    """Propose new concept."""
    # PERF-FORT-1: Semaphore prevents threadpool starvation under concurrent load
    acquired = _HEAVY_ENDPOINT_SEMAPHORE.acquire(timeout=HEAVY_ENDPOINT_TIMEOUT_S)
    if not acquired:
        raise HTTPException(
            status_code=503,
            detail="Server under heavy load — try again in a few seconds",
            headers={"Retry-After": "3"},
        )
    try:
        return _pith_propose_concept_inner(proposal)
    finally:
        _HEAVY_ENDPOINT_SEMAPHORE.release()


def _pith_propose_concept_inner(proposal: ConceptProposal):
    """Inner logic for propose_concept — separated for semaphore wrapping."""
    # Validate proposal
    valid, message = validate_proposal(proposal)
    if not valid:
        raise HTTPException(status_code=400, detail=message)

    # §5.8.4 H18: Write-scoped governance context for event tracing
    from app.governance.governance_context import write_governance_context

    _gov_ctx_mgr = write_governance_context("propose_concept")
    _gov_ctx = _gov_ctx_mgr.__enter__()

    # Memory Integrity §5.1.5: Write-time contradiction check
    contra_result = None
    try:
        from app.cognitive.contradiction import detect_write_contradiction

        contra_result = detect_write_contradiction(
            new_summary=proposal.summary,
            new_knowledge_area=getattr(proposal, "knowledge_area", "general") or "general",
            concept_id=proposal.concept_id,
        )
        # Log ingestion validation event through GovernanceContext
        if _gov_ctx:
            _gov_ctx.log_ingestion_validation(
                concept_id=proposal.concept_id,
                validation_result=contra_result.action,
                reason=contra_result.reason or "",
                contradiction_score=getattr(contra_result, "contradiction_score", 0.0),
                tier_used=getattr(contra_result, "tier_used", 0),
            )
        if contra_result.action == "HARD_REJECT":
            from app.governance.policy_engine import PolicyViolation, get_policy_engine

            try:
                engine = get_policy_engine()
                engine._log_violation(
                    PolicyViolation(
                        rule_id="write_contradiction_hard_reject",
                        severity="BLOCK",
                        concept_id=proposal.concept_id,
                        detail=contra_result.reason,
                        caller_context="pith_propose_concept",
                    )
                )
            except Exception:
                pass
            # Flush governance context before raising
            try:
                _gov_ctx_mgr.__exit__(None, None, None)
            except Exception:
                pass
            raise HTTPException(
                status_code=409, detail=f"Concept contradicts existing knowledge: {contra_result.reason}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"propose_concept: contradiction check failed (non-fatal): {e}")

    # Memory Integrity §Gap 2 / A4-H1: Dedup at ingestion
    # INGEST-005: Use embedding dedup (matches session_learn path) with config thresholds
    try:
        from app.core.config import (
            EMBEDDING_EVOLVE_THRESHOLD,
            EMBEDDING_SKIP_THRESHOLD,
            FEATURE_FLAGS,
        )

        if FEATURE_FLAGS.get("DEDUP_AT_INGESTION_ENABLED", False):
            _use_embedding = FEATURE_FLAGS.get("EMBEDDING_DEDUP_ENABLED", False)
            if _use_embedding:
                dedup_results = retrieval_engine.search_for_dedup_embedding(proposal.summary, top_k=3)
                _skip_threshold = EMBEDDING_SKIP_THRESHOLD
                _evolve_threshold = EMBEDDING_EVOLVE_THRESHOLD
            else:
                dedup_results = retrieval_engine.search_for_dedup_tfidf(proposal.summary, top_k=3)
                _skip_threshold = 0.85
                _evolve_threshold = 0.50

            top_cosine = dedup_results[0]["cosine_score"] if dedup_results else 0.0
            top_match = dedup_results[0] if dedup_results else None

            # INGEST-007: Cross-KA merge guard
            from app.core.config import CROSS_KA_EVOLVE_THRESHOLD, ka_groups_match

            _incoming_ka = getattr(proposal, "knowledge_area", "general") or "general"
            _match_ka = top_match.get("knowledge_area", "") if top_match else ""
            _ka_match = ka_groups_match(_incoming_ka, _match_ka)
            _effective_evolve = _evolve_threshold if _ka_match else CROSS_KA_EVOLVE_THRESHOLD

            # Classify dedup zone (DATA-055 parity)
            if top_cosine >= _skip_threshold:
                _dedup_zone = "SKIP"
            elif top_cosine >= _effective_evolve and top_match:
                _dedup_zone = "EVOLVE"
            else:
                _dedup_zone = "CREATE"

            # INGEST-007: Log KA guard decision
            if not _ka_match and top_cosine >= _evolve_threshold:
                logger.info(
                    f"INGEST-007: Cross-KA guard activated — "
                    f"incoming={_incoming_ka} match={_match_ka} "
                    f"cosine={top_cosine:.4f} effective_thresh={_effective_evolve:.2f} "
                    f"zone={_dedup_zone}"
                )

            # INGEST-005: Structured dedup log (matches session_learn DATA-055 format)
            _dedup_method = "embedding" if _use_embedding else "tfidf"
            _match_id = top_match["concept_id"] if top_match else None
            logger.info(
                f"DEDUP_DECISION: zone={_dedup_zone} cosine={top_cosine:.4f} "
                f"match={_match_id} method={_dedup_method} "
                f"skip_thresh={_skip_threshold} evolve_thresh={_evolve_threshold:.2f} "
                f"caller=propose_concept "
                f"summary_hash={hashlib.sha256(proposal.summary.encode()).hexdigest()[:12]}"
            )

            if _dedup_zone == "SKIP":
                return {
                    "status": "skipped_duplicate",
                    "existing_concept_id": top_match["concept_id"],
                    "cosine_score": top_cosine,
                    "message": f"Near-duplicate of {top_match['concept_id']} (cosine={top_cosine:.3f})",
                }
            elif _dedup_zone == "EVOLVE":
                try:
                    from app.cognitive.learning import evolve_concept
                    from app.core.models import ConceptEvolution

                    evo = ConceptEvolution(
                        concept_id=top_match["concept_id"],
                        new_summary=proposal.summary,
                        new_evidence=getattr(proposal, "evidence", None),
                    )
                    evolved = evolve_concept(evo)
                    if evolved:
                        retrieval_engine.add_concept(evolved.id)
                        return {
                            "status": "merged_into_existing",
                            "existing_concept_id": top_match["concept_id"],
                            "cosine_score": top_cosine,
                            "evolved_version": evolved.version,
                            "message": f"Merged into {top_match['concept_id']} (cosine={top_cosine:.3f})",
                        }
                except Exception as e:
                    logger.warning(f"propose_concept: dedup merge failed, creating new: {e}")
    except Exception as e:
        logger.warning(f"propose_concept: dedup check failed (non-fatal): {e}")

    # Create concept
    try:
        concept = create_concept(proposal)
        # If QUARANTINE, override maturity
        try:
            if contra_result.action == "QUARANTINE":
                from app.storage import _db

                with _db() as conn:
                    conn.execute(
                        "UPDATE concepts SET maturity = 'QUARANTINED', "
                        "data = json_set(data, '$.maturity', 'QUARANTINED') "
                        "WHERE id = ?",
                        (concept.id,),
                    )
                logger.info(
                    f"propose_concept: quarantined {concept.id} due to contradiction with {contra_result.contradicting_concept_id}"
                )
        except Exception as e:
            logger.warning(f"propose_concept: quarantine update failed: {e}")

        # FED-005: Emit federation event for propose_concept
        try:
            session_manager._emit_federation_event(
                event_type="concept_proposed",
                concept_id=concept.id,
                payload={
                    "summary": concept.summary,
                    "confidence": concept.confidence,
                    "knowledge_area": getattr(concept, "knowledge_area", "general"),
                    "concept_type": getattr(concept, "concept_type", "observation"),
                    "original_confidence": concept.confidence,
                },
                model_id="unknown",
            )
        except Exception as e:
            logger.debug(f"propose_concept: federation event failed (non-fatal): {e}")

        # Update index
        retrieval_engine.add_concept(concept.id)

        # Auto-associate with existing concepts
        from app.cognitive.association import auto_associate_single
        from app.core.models import AutoAssociateSingleRequest

        assoc_request = AutoAssociateSingleRequest(threshold=0.12, max_edges=3)
        try:
            assoc_result = auto_associate_single(concept.id, assoc_request)
            assoc_count = assoc_result.edges_created
        except Exception as e:
            logger.warning(f"propose_concept: auto_associate failed for {concept.id}: {e}")
            assoc_count = 0

        # C3: Implicit learning event + consume budget
        session_manager._consume_budget()
        session_manager.register_implicit_learning_event(
            event_type="concept_proposed",
            concept_id=concept.id,
            summary=concept.summary,
        )

        # C2: Enrich with similar existing concepts (dedup value)
        ambient = select_ambient_concepts(concept.id, query=concept.summary)

        # P1-1: Handle always_activate flag if provided in request body
        aa_skipped = False
        raw_body = proposal.model_dump()
        if raw_body.get("always_activate"):
            from app.core.config import MAX_ALWAYS_ACTIVATE
            from app.storage import load_always_activate_concepts, set_always_activate

            current_aa = load_always_activate_concepts()
            if len(current_aa) >= MAX_ALWAYS_ACTIVATE:
                logger.warning(
                    f"propose_concept: always_activate skipped for {concept.id} — "
                    f"cap reached ({len(current_aa)}/{MAX_ALWAYS_ACTIVATE})"
                )
                aa_skipped = True
            else:
                set_always_activate(concept.id, True)

        response = {
            "status": "created",
            "concept_id": concept.id,
            "version": concept.version,
            "message": "Concept created successfully",
            "associations_created": assoc_count,
        }
        if aa_skipped:
            response["always_activate_warning"] = (
                f"always_activate flag not set — cap reached "
                f"({MAX_ALWAYS_ACTIVATE}/{MAX_ALWAYS_ACTIVATE}). Unset one first."
            )
        if ambient:
            response["ambient_context"] = {"related": ambient}
        # §5.8.4 H18: Flush write-scoped governance context
        try:
            _gov_ctx_mgr.__exit__(None, None, None)
        except Exception:
            pass
        return response
    except Exception as e:
        # Flush governance context on error path too
        try:
            _gov_ctx_mgr.__exit__(None, None, None)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=_safe_error(e))


@app.post("/pith_evolve_concept", dependencies=[Depends(verify_api_key)])
def pith_evolve_concept_endpoint(evolution: ConceptEvolution):
    """Evolve existing concept."""
    # §5.8.4 H18: Write-scoped governance context
    from app.governance.governance_context import write_governance_context

    _gov_ctx_mgr = write_governance_context("evolve_concept")
    _gov_ctx = _gov_ctx_mgr.__enter__()

    concept = evolve_concept(evolution)

    if not concept:
        try:
            _gov_ctx_mgr.__exit__(None, None, None)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail="Evolution not warranted or concept not found")

    # Update index
    retrieval_engine.add_concept(concept.id)

    # Memory Integrity A4-H1: Post-evolution dedup check
    dedup_warning = None
    try:
        from app.core.config import FEATURE_FLAGS

        if FEATURE_FLAGS.get("DEDUP_AT_INGESTION_ENABLED", False):
            dedup_results = retrieval_engine.search_for_dedup_tfidf(concept.summary, top_k=3)
            for match in dedup_results:
                if match["concept_id"] != concept.id and match["concept_id"] != evolution.concept_id:
                    if match["cosine_score"] >= 0.85:
                        dedup_warning = {
                            "duplicate_of": match["concept_id"],
                            "cosine_score": match["cosine_score"],
                            "message": f"Post-evolution: now duplicates {match['concept_id']}",
                        }
                        logger.warning(
                            f"evolve_concept: post-evolution dedup alert — "
                            f"{concept.id} now duplicates {match['concept_id']} "
                            f"(cosine={match['cosine_score']:.3f})"
                        )
                        break
    except Exception as e:
        logger.warning(f"evolve_concept: post-evolution dedup check failed: {e}")

    # C3: Implicit learning event
    session_manager.register_implicit_learning_event(
        event_type="concept_evolved",
        concept_id=concept.id,
        summary=concept.summary,
    )

    # P1-1: Handle always_activate flag if explicitly set
    if evolution.always_activate is not None:
        if evolution.always_activate:
            from app.core.config import MAX_ALWAYS_ACTIVATE
            from app.storage import load_always_activate_concepts, set_always_activate

            current_aa = load_always_activate_concepts()
            if len(current_aa) >= MAX_ALWAYS_ACTIVATE:
                logger.warning(
                    f"evolve_concept: always_activate skipped for {concept.id} — "
                    f"cap reached ({len(current_aa)}/{MAX_ALWAYS_ACTIVATE})"
                )
            else:
                set_always_activate(concept.id, True)
        else:
            from app.storage import set_always_activate

            set_always_activate(concept.id, False)

    # C2: Enrich with affected concepts via associations
    ambient = select_ambient_concepts(concept.id, query=concept.summary)

    response = {
        "status": "evolved",
        "concept_id": concept.id,
        "version": concept.version,
        "previous_version": concept.supersedes,
        "message": "Concept evolved successfully",
    }
    if ambient:
        response["ambient_context"] = {"related": ambient}
    if dedup_warning:
        response["dedup_warning"] = dedup_warning
    # §5.8.4 H18: Flush write-scoped governance context
    try:
        _gov_ctx_mgr.__exit__(None, None, None)
    except Exception:
        pass
    return response


class AlwaysActivateRequest(PydanticBaseModel):
    """P1-1: Request to set/unset always-activate flag."""

    concept_id: str
    value: bool = True


@app.post("/pith_set_always_activate", dependencies=[Depends(verify_api_key)])
def pith_set_always_activate(request: AlwaysActivateRequest):
    """Set or unset always_activate flag on a concept. P1-1."""
    from app.core.config import MAX_ALWAYS_ACTIVATE
    from app.storage import load_always_activate_concepts, set_always_activate

    # Governance: cap at MAX_ALWAYS_ACTIVATE concepts
    if request.value:
        current = load_always_activate_concepts()
        if len(current) >= MAX_ALWAYS_ACTIVATE:
            raise HTTPException(
                status_code=400,
                detail=f"Maximum {MAX_ALWAYS_ACTIVATE} always-activate concepts allowed. Currently {len(current)} flagged. Unset one first.",
            )
    updated = set_always_activate(request.concept_id, request.value)
    if not updated:
        raise HTTPException(status_code=404, detail=f"Concept {request.concept_id} not found")
    return {"status": "updated", "concept_id": request.concept_id, "always_activate": request.value}


@app.get("/pith_list_always_activate", dependencies=[Depends(verify_api_key)])
def pith_list_always_activate():
    """List all concepts with always_activate flag. P1-1."""
    from app.core.config import MAX_ALWAYS_ACTIVATE
    from app.storage import load_always_activate_concepts

    concepts = load_always_activate_concepts()
    return {"count": len(concepts), "max": MAX_ALWAYS_ACTIVATE, "concepts": concepts}


# =============================================================================
# Quarantine Management — §5.2.5 Gap 5 (Memory Integrity Spec v1.2)
# =============================================================================


@app.get("/pith/quarantine", dependencies=[Depends(verify_api_key)])
def list_quarantined(limit: int = 50):
    """List concepts with maturity='QUARANTINED'.

    Returns concept_id, summary, confidence, quarantine_entered, evidence count.
    Feature-gated by QUARANTINE_ENDPOINTS_ENABLED.
    """
    from app.core.config import FEATURE_FLAGS

    if not FEATURE_FLAGS.get("QUARANTINE_ENDPOINTS_ENABLED", False):
        raise HTTPException(status_code=503, detail="Quarantine endpoints not enabled")

    from app.storage import get_db_connection

    conn = get_db_connection()
    rows = conn.execute(
        """
        SELECT id, summary, confidence, maturity, data
        FROM concepts
        WHERE maturity = 'QUARANTINED'
        ORDER BY updated_at DESC
        LIMIT ?
    """,
        (min(limit, 200),),
    ).fetchall()

    results = []
    for row in rows:
        import json as _json

        data = _json.loads(row["data"]) if row["data"] else {}
        results.append(
            {
                "concept_id": row["id"],
                "summary": row["summary"],
                "confidence": row["confidence"],
                "quarantine_entered": data.get("quarantine_entered"),
                "evidence_count": len(data.get("evidence", [])),
                "knowledge_area": data.get("knowledge_area", ""),
            }
        )
    return {"count": len(results), "quarantined": results}


class QuarantineActionRequest(PydanticBaseModel):
    """Request body for quarantine promote/discard actions."""

    reason: str = ""


@app.post("/pith/quarantine/{concept_id}/promote", dependencies=[Depends(verify_api_key)])
def promote_from_quarantine(concept_id: str, request: QuarantineActionRequest = QuarantineActionRequest()):
    """Promote a quarantined concept to PROVISIONAL.

    Updates maturity, maturity_promoted_at, maturity_promotion_evidence.
    Logs to policy_violations for audit trail.
    """
    from app.core.config import FEATURE_FLAGS

    if not FEATURE_FLAGS.get("QUARANTINE_ENDPOINTS_ENABLED", False):
        raise HTTPException(status_code=503, detail="Quarantine endpoints not enabled")

    concept = load_concept(concept_id, track_access=False)
    if not concept:
        raise HTTPException(status_code=404, detail=f"Concept {concept_id} not found")
    if concept.maturity != "QUARANTINED":
        raise HTTPException(status_code=409, detail=f"Concept {concept_id} is {concept.maturity}, not QUARANTINED")

    now = _utc_now_iso()
    concept.maturity = "PROVISIONAL"
    concept.maturity_promoted_at = now
    concept.maturity_promotion_evidence = request.reason or "manual_promotion"

    save_concept(concept)

    # Audit log
    from app.governance.policy_engine import log_policy_event

    log_policy_event(
        rule_id="quarantine_promote",
        severity="LOG",
        concept_id=concept_id,
        detail=f"Promoted from QUARANTINED to PROVISIONAL: {request.reason}",
        caller_context="quarantine_endpoint",
    )

    return {
        "status": "promoted",
        "concept_id": concept_id,
        "new_maturity": "PROVISIONAL",
        "promoted_at": now,
    }


@app.post("/pith/quarantine/{concept_id}/discard", dependencies=[Depends(verify_api_key)])
def discard_quarantined(concept_id: str, request: QuarantineActionRequest = QuarantineActionRequest()):
    """Discard a quarantined concept (set maturity to DISCARDED).

    Version history preserved in concept_versions for forensics.
    Logs to policy_violations for audit trail.
    """
    from app.core.config import FEATURE_FLAGS

    if not FEATURE_FLAGS.get("QUARANTINE_ENDPOINTS_ENABLED", False):
        raise HTTPException(status_code=503, detail="Quarantine endpoints not enabled")

    concept = load_concept(concept_id, track_access=False)
    if not concept:
        raise HTTPException(status_code=404, detail=f"Concept {concept_id} not found")
    if concept.maturity != "QUARANTINED":
        raise HTTPException(status_code=409, detail=f"Concept {concept_id} is {concept.maturity}, not QUARANTINED")

    concept.maturity = "DISCARDED"
    save_concept(concept)

    # Audit log
    from app.governance.policy_engine import log_policy_event

    log_policy_event(
        rule_id="quarantine_discard",
        severity="LOG",
        concept_id=concept_id,
        detail=f"Discarded from quarantine: {request.reason}",
        caller_context="quarantine_endpoint",
    )

    return {
        "status": "discarded",
        "concept_id": concept_id,
        "new_maturity": "DISCARDED",
    }


# =============================================================================
# Rejection Visibility — §5.2.10 H14 (Memory Integrity Spec v1.2)
# =============================================================================


@app.get("/pith/policy/rejections", dependencies=[Depends(verify_api_key)])
def get_policy_rejections(
    since: str | None = None,
    severity: str | None = None,
    limit: int = 100,
):
    """Get filtered rejection log with rate statistics.

    §5.2.10 H14: Operators need visibility into why concepts were rejected.
    Returns rejections from policy_violations table plus hourly rate stats.
    Feature-gated by REJECTION_VISIBILITY_ENABLED.

    Query params:
        since: ISO datetime filter (e.g. '2026-02-25T00:00:00')
        severity: Filter by severity ('BLOCK', 'WARN')
        limit: Max results (default 100)
    """
    from app.core.config import FEATURE_FLAGS

    if not FEATURE_FLAGS.get("REJECTION_VISIBILITY_ENABLED", False):
        raise HTTPException(
            status_code=503,
            detail="Rejection visibility endpoint not enabled (REJECTION_VISIBILITY_ENABLED=False)",
        )
    from app.governance.policy_engine import get_rejections

    return get_rejections(since=since, severity=severity, limit=limit)


# =============================================================================
# Behavioral Directives — CRUD (DOMAINS_AND_DIRECTIVES_SPEC.md Section 3.6)
# =============================================================================


class DirectiveRequest(PydanticBaseModel):
    """Create or update a behavioral directive."""

    directive_id: str
    category: str
    content: str
    priority: int = 100


@app.post("/directives", dependencies=[Depends(verify_api_key)])
def create_or_update_directive(request: DirectiveRequest):
    """Create or update a directive (upsert by directive_id). S4.8."""
    from app.governance.directives import DirectiveValidationError, save_directive

    try:
        result = save_directive(
            directive_id=request.directive_id,
            category=request.category,
            content=request.content,
            priority=request.priority,
        )
        return result
    except DirectiveValidationError as e:
        raise HTTPException(status_code=400, detail={"error": e.error_code, "message": e.detail})


@app.get("/directives")
def list_directives(category: str | None = None, active: bool | None = None):
    """List all directives with optional filters."""
    from app.governance.directives import load_directives

    all_directives = load_directives(active_only=active if active is not None else True)
    if category:
        all_directives = [d for d in all_directives if d["category"] == category]
    return {"count": len(all_directives), "directives": all_directives}


@app.get("/directive/{directive_id}")
def get_directive_detail(directive_id: str):
    """Get a single directive with version history."""
    from app.governance.directives import get_directive

    result = get_directive(directive_id, include_versions=True)
    if not result:
        raise HTTPException(status_code=404, detail=f"Directive {directive_id} not found")
    return result


@app.delete("/directive/{directive_id}", dependencies=[Depends(verify_api_key)])
def delete_directive_endpoint(directive_id: str):
    """Soft-delete a directive (sets active=false)."""
    from app.governance.directives import delete_directive

    if delete_directive(directive_id):
        return {"status": "deactivated", "directive_id": directive_id}
    raise HTTPException(status_code=404, detail=f"Directive {directive_id} not found")


# =============================================================================
# Wave 4a — pith_salience Tool (§4a.5)
# =============================================================================


class SalienceRequest(PydanticBaseModel):
    concept_id: str | None = None
    mode: str = "get"  # get | set | recompute | bulk_recompute
    salience: float | None = None  # For mode="set"
    reason: str | None = None  # For mode="set"


@app.post("/pith_salience", dependencies=[Depends(verify_api_key)])
def pith_salience(request: SalienceRequest):
    """Get or set concept salience.

    Modes:
    - get: Return current salience + breakdown for concept_id
    - set: Manually set salience (salience_source="user")
    - recompute: Trigger system recomputation for concept_id
    - bulk_recompute: Recompute all concepts (reflection-time only)
    """
    from app.retrieval.salience import recompute_salience

    if request.mode == "get":
        if not request.concept_id:
            raise HTTPException(status_code=400, detail="concept_id required for get mode")
        concept = load_concept(request.concept_id)
        if not concept:
            raise HTTPException(status_code=404, detail=f"Concept {request.concept_id} not found")
        return {
            "concept_id": request.concept_id,
            "salience": concept.salience,
            "salience_source": concept.salience_source,
            "salience_set_at": concept.salience_set_at,
            "salience_reason": concept.salience_reason,
        }

    elif request.mode == "set":
        if not request.concept_id:
            raise HTTPException(status_code=400, detail="concept_id required for set mode")
        if request.salience is None:
            raise HTTPException(status_code=400, detail="salience value required for set mode")
        concept = load_concept(request.concept_id, track_access=False)
        if not concept:
            raise HTTPException(status_code=404, detail=f"Concept {request.concept_id} not found")
        concept.salience = max(0.0, min(1.0, request.salience))
        concept.salience_source = "user"
        concept.salience_set_at = _utc_now_iso()
        concept.salience_reason = request.reason or "Manually set by user"
        from app.storage import save_concept

        save_concept(concept)
        return {
            "status": "updated",
            "concept_id": request.concept_id,
            "salience": concept.salience,
            "salience_source": "user",
        }

    elif request.mode == "recompute":
        if not request.concept_id:
            raise HTTPException(status_code=400, detail="concept_id required for recompute mode")
        result = recompute_salience(concept_id=request.concept_id)
        return result

    elif request.mode == "bulk_recompute":
        result = recompute_salience(concept_id=None)
        return result

    else:
        raise HTTPException(status_code=400, detail=f"Unknown mode: {request.mode}")


@app.post("/pith_link_concepts", dependencies=[Depends(verify_api_key)])
def pith_link_concepts(association: Association):
    """Create association between concepts."""
    # Verify both concepts exist
    concept_a = load_concept(association.concept_a, track_access=False)
    concept_b = load_concept(association.concept_b, track_access=False)

    if not concept_a:
        raise HTTPException(status_code=404, detail=f"Concept {association.concept_a} not found")
    if not concept_b:
        raise HTTPException(status_code=404, detail=f"Concept {association.concept_b} not found")

    # Add association
    add_association(association.concept_a, association.concept_b, association.relation, association.strength)

    # C3: Implicit learning event
    session_manager.register_implicit_learning_event(
        event_type="concepts_linked",
        concept_id=f"{association.concept_a}<->{association.concept_b}",
        summary=f"Linked {association.concept_a} to {association.concept_b} via {association.relation}",
    )

    return {
        "status": "linked",
        "concept_a": association.concept_a,
        "concept_b": association.concept_b,
        "relation": association.relation,
        "message": "Concepts linked successfully",
    }


@app.get("/pith_related_concepts")
def pith_related_concepts(concept_id: str, max_depth: int = 2) -> list[str]:
    """Get concepts related to given concept."""
    concept = load_concept(concept_id)
    if not concept:
        raise HTTPException(status_code=404, detail=f"Concept {concept_id} not found")

    related = get_related_concepts(concept_id, max_depth)
    # MATURITY-001: Filter quarantined/discarded from external results
    filtered = []
    for cid in related:
        c = load_concept(cid, track_access=False)
        if c and getattr(c, "maturity", "ESTABLISHED") not in _BLOCKED_MATURITIES:
            filtered.append(cid)
    return filtered


@app.post("/pith_curiosity", dependencies=[Depends(verify_api_key)])
def pith_curiosity():
    """Generate questions for weak concepts."""
    questions = curiosity_engine.generate_questions()

    # Add to queue
    question_queue.add_questions(questions)

    return {"generated": len(questions), "questions": [q.model_dump() for q in questions]}


@app.get("/pith_questions")
def pith_questions(limit: int = 10) -> list[dict]:
    """Get questions from curiosity queue."""
    questions = question_queue.get_questions(limit)
    return questions


@app.get("/pith_curiosity/experiment_frontier", dependencies=[Depends(verify_api_key)])
def pith_curiosity_experiment_frontier(limit: int = 20, baseline_limit: int = 20) -> dict:
    """Preview read-only Curiosity questions generated from Experiment outputs."""
    bounded_limit = max(1, min(int(limit), 50))
    bounded_baseline_limit = max(0, min(int(baseline_limit), 50))
    payload = curiosity_engine.generate_experiment_frontier_questions(
        limit=bounded_limit,
        baseline_limit=bounded_baseline_limit,
    )
    payload["operator_surface"] = {
        "read_only": True,
        "writes_to_queue": False,
        "writes_to_concepts": False,
        "writes_to_experiments": False,
        "limit": bounded_limit,
        "baseline_limit": bounded_baseline_limit,
    }
    return payload


@app.post("/pith_reindex", dependencies=[Depends(verify_api_key)])
def pith_reindex():
    """Rebuild the entire search index, purging stale entries."""
    try:
        # Compact first to purge ghost entries from deleted concepts
        retrieval_engine.index._compact_matrix()
        # Then rebuild to pick up any missing concepts
        retrieval_engine.build_index()
        indexed_count = len(retrieval_engine.index.concept_ids) - len(retrieval_engine.index.deleted_indices)
        return {
            "status": "success",
            "message": "Index compacted and rebuilt successfully",
            "concepts_indexed": indexed_count,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=_safe_error(e))


@app.get("/pith_integrity")
def pith_integrity(repair: bool = False):
    """Check index integrity and optionally auto-repair drift.

    SYSTEMIC_FIXES_SPEC v1.1 Fix 2: Dedicated integrity endpoint.

    Query params:
        repair: If true, auto-repair any drift found (default: false).
    """
    try:
        integrity = retrieval_engine.verify_index_integrity()
        result = {**integrity}

        if repair and not integrity["is_healthy"]:
            repair_result = retrieval_engine.repair_index_drift(integrity=integrity)
            result["repair"] = repair_result

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=_safe_error(e))


@app.delete("/pith_question/{concept_id}", dependencies=[Depends(verify_api_key)])
def remove_question(concept_id: str):
    """Remove question for concept from queue."""
    question_queue.remove_question(concept_id)
    return {"status": "removed", "concept_id": concept_id}


@app.post("/pith_reflect", dependencies=[Depends(verify_api_key)])
def pith_reflect(mode: str = "incremental", verbose: bool = False):
    """Run reflection/consolidation cycle.

    Args:
        mode: 'incremental' (default) or 'full'
        verbose: If True, include phase_timings and evidence_cv breakdowns.
                 If False (default), return key counts + reflection_summary only.

    The reflection_summary field is always present regardless of verbose setting.
    """
    try:
        summary = reflection_engine.reflect(mode)

        # OPS-046: Build human-readable reflection_summary (always present)
        parts = []
        if summary.concepts_consolidated:
            parts.append(f"{summary.concepts_consolidated} consolidated")
        if summary.concepts_decayed:
            parts.append(f"{summary.concepts_decayed} decayed")
        if summary.concepts_archived:
            parts.append(f"{summary.concepts_archived} archived")
        if summary.concepts_recalibrated:
            parts.append(f"{summary.concepts_recalibrated} recalibrated")
        if summary.concepts_promoted:
            parts.append(f"{summary.concepts_promoted} promoted")
        if summary.concepts_time_matured:
            parts.append(f"{summary.concepts_time_matured} matured")
        if summary.associations_updated:
            parts.append(f"{summary.associations_updated} associations updated")
        if summary.questions_generated:
            parts.append(f"{summary.questions_generated} questions generated")

        if summary.aborted:
            abort_location = getattr(summary, "abort_stage", None) or summary.last_completed_step or "startup"
            reflection_summary = (
                f"{mode.capitalize()} reflection aborted: "
                f"{summary.abort_reason or 'unknown'} after {abort_location}"
            )
        else:
            reflection_summary = f"{mode.capitalize()} reflection: " + (", ".join(parts) if parts else "no changes")

        # Core fields always returned
        response = {
            "reflection_summary": reflection_summary,
            "mode": mode,
            "concepts_consolidated": summary.concepts_consolidated,
            "concepts_decayed": summary.concepts_decayed,
            "concepts_archived": summary.concepts_archived,
            "concepts_recalibrated": summary.concepts_recalibrated,
            "concepts_promoted": summary.concepts_promoted,
            "concepts_time_matured": summary.concepts_time_matured,
            "associations_updated": summary.associations_updated,
            "questions_generated": summary.questions_generated,
            "gc_queue_remaining": summary.gc_queue_remaining,
            "timestamp": summary.timestamp,
            "aborted": summary.aborted,
            "abort_reason": summary.abort_reason,
            "last_completed_step": summary.last_completed_step,
            "abort_stage": getattr(summary, "abort_stage", None),
        }

        # Verbose fields: internal scoring details omitted by default (OPS-048 alignment)
        if verbose:
            response["phase_timings"] = summary.phase_timings
            response["evidence_cv"] = {
                "composite": summary.evidence_cv_composite,
                "reliability": summary.evidence_cv_reliability,
                "directness": summary.evidence_cv_directness,
                "consistency": summary.evidence_cv_consistency,
                "corroboration": summary.evidence_cv_corroboration,
                "recency": summary.evidence_cv_recency,
            }

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=_safe_error(e))


@app.get("/pith_health")
def pith_health(detail: str = "fast"):
    """Get pith health analysis."""
    if detail not in {"fast", "full"}:
        raise HTTPException(status_code=400, detail="detail must be 'fast' or 'full'")
    ready = _build_ready_state()
    if _should_defer_health_db_metrics(ready):
        reason = _health_metrics_defer_reason(ready)
        return {
            "mode": detail,
            "generated_at": _utc_now_iso(),
            "cache_age_ms": 0,
            "partial": True,
            "section_errors": {"startup": reason},
            "status": "recovering",
            "health_score": 0.0,
            "total_concepts": 0,
            "avg_confidence": 0.0,
            "avg_stability": 0.0,
            "ready_state": {
                "mode": ready.get("mode"),
                "process_state": ready.get("process_state"),
                "write_state": ready.get("write_state"),
                "retrieval_state": ready.get("retrieval_state"),
            },
        }
    if detail == "fast":
        return _build_pith_health_fast_cached()

    try:
        health = reflection_engine.analyze_stability()

        # FEDERATION L1.5: Add model diversity stats
        try:
            from app.storage import _get_connection

            conn = _get_connection()
            model_rows = conn.execute(
                "SELECT model_id, COUNT(*) as session_count "
                "FROM sessions WHERE model_id IS NOT NULL AND model_id != 'unknown' "
                "GROUP BY model_id ORDER BY session_count DESC"
            ).fetchall()
            health["model_stats"] = {
                "models_seen": [{"model_id": r[0], "session_count": r[1]} for r in model_rows],
                "unique_model_count": len(model_rows),
            }
        except Exception as e:
            health["model_stats"] = {"error": _safe_error(e)}

        # FEDERATION L2: Add federation status (A4.2)
        try:
            from app.storage import _db

            with _db() as conn:
                tables = [
                    r[0]
                    for r in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='federation_events'"
                    ).fetchall()
                ]
                if "federation_events" in tables:
                    recent_events = conn.execute(
                        "SELECT COUNT(*) FROM federation_events WHERE created_at > datetime('now', '-24 hours')"
                    ).fetchone()[0]
                    unconsumed = conn.execute("SELECT COUNT(*) FROM federation_events WHERE consumed = 0").fetchone()[0]
                    total_events = conn.execute("SELECT COUNT(*) FROM federation_events").fetchone()[0]
                    # MAINT-015: Accurate bridge status
                    if total_events > 0 and unconsumed == total_events:
                        bridge_status = "no_consumer"
                    elif unconsumed >= 1000:
                        bridge_status = "backpressure"
                    elif unconsumed > 0:
                        bridge_status = "lagging"
                    else:
                        bridge_status = "healthy"
                    health["federation_status"] = {
                        "events_emitted_24h": recent_events,
                        "events_unconsumed": unconsumed,
                        "events_total": total_events,
                        "bridge_status": bridge_status,
                        "bridge_healthy": bridge_status in ("healthy", "lagging"),
                    }
        except Exception as e:
            health["federation_status"] = {"error": _safe_error(e)}

        # VERBATIM-SURFACE A4: FTS5 parity check
        try:
            from app.storage import _db as _vs_db

            with _vs_db() as _vs_conn:
                _fts_count = _vs_conn.execute(
                    "SELECT COUNT(DISTINCT fc.c0) "
                    "FROM fts_verbatim_content fc "
                    "JOIN verbatim_fragments vf ON vf.id = fc.c0 "
                    "JOIN concepts c ON c.id = vf.concept_id "
                    "WHERE vf.fragment_type='conversation' "
                    "AND vf.content IS NOT NULL "
                    "AND c.status='active'"
                ).fetchone()[0]
                _canonical_count = _vs_conn.execute(
                    "SELECT COUNT(*) FROM verbatim_fragments vf "
                    "JOIN concepts c ON c.id = vf.concept_id "
                    "WHERE vf.fragment_type='conversation' "
                    "AND vf.content IS NOT NULL "
                    "AND c.status='active'"
                ).fetchone()[0]
                _fts_drift = abs(_fts_count - _canonical_count) / max(_canonical_count, 1)
                health["fts_verbatim_parity"] = {
                    "fts_count": _fts_count,
                    "canonical_count": _canonical_count,
                    "drift_pct": round(_fts_drift * 100, 1),
                    "status": "ok" if _fts_drift <= 0.05 else "degraded",
                }
        except Exception as _vs_err:
            health["fts_verbatim_parity"] = {"error": str(_vs_err)}

        # STABILITY-021: Include startup warnings in health
        warnings = getattr(app.state, "startup_warnings", [])
        if warnings:
            health["startup_warnings"] = warnings
            health["startup_degraded"] = any(w["severity"] == "critical" for w in warnings)

        health.setdefault("mode", "full")
        health.setdefault("generated_at", _utc_now_iso())
        health.setdefault("cache_age_ms", 0)
        health.setdefault("partial", False)
        health.setdefault("section_errors", {})
        health.setdefault("section_timings_ms", {})
        return health
    except Exception as e:
        raise HTTPException(status_code=500, detail=_safe_error(e))


@app.get("/memory_projection")
def memory_projection():
    """HEALTH-002: Predictive memory growth projection."""
    try:
        return get_memory_projection_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=_safe_error(e))


@app.get("/retrieval_distribution")
def retrieval_distribution():
    """MEASURE-005: Retrieval score distribution report."""
    try:
        return get_distribution_report()
    except Exception as e:
        raise HTTPException(status_code=500, detail=_safe_error(e))


@app.get("/health/maintenance")
def maintenance_health():
    """Amendment 3: Check maintenance scheduler health via heartbeat file."""
    result = _build_maintenance_health()
    result["session_learn_queue"] = _build_session_learn_queue_health()
    return result


@app.get("/health/backup")
def backup_health():
    """OPS-153: Backup health status — structured JSON for monitoring dashboards.

    Returns age, size, concept count, integrity, and a clear status:
      healthy: backup exists and is <12h old
      warning: backup exists but is 12-24h old
      critical: backup missing or >24h old or integrity failure
    """
    import sqlite3 as _sqlite3
    from datetime import UTC
    from datetime import datetime as _dt

    from app.core.profile import resolve_data_dir

    data_dir = resolve_data_dir()
    backup_path = data_dir / "pith_backup.db"

    if not backup_path.exists():
        return {
            "status": "critical",
            "reason": "no_backup",
            "message": "No backup file found. Run maintenance or wait for next 6h cycle.",
            "backup_path": str(backup_path),
            "timestamp": _utc_now_iso(),
        }

    try:
        stat = backup_path.stat()
        backup_age_hours = (_utc_now() - _ensure_aware(_dt.fromtimestamp(stat.st_mtime, tz=UTC))).total_seconds() / 3600
        backup_size_mb = round(stat.st_size / 1024 / 1024, 1)

        # Read-only connection to backup for integrity + concept count
        verify_conn = _sqlite3.connect(f"file:{backup_path}?mode=ro", uri=True)
        try:
            integrity = verify_conn.execute("PRAGMA quick_check").fetchone()[0]
            concept_count = verify_conn.execute("SELECT COUNT(*) FROM concepts WHERE is_current = 1").fetchone()[0]
        finally:
            verify_conn.close()

        # Determine status
        if integrity != "ok":
            status = "critical"
            reason = "integrity_failure"
        elif backup_age_hours > 24:
            status = "critical"
            reason = "stale_backup"
        elif backup_age_hours > 12:
            status = "warning"
            reason = "aging_backup"
        else:
            status = "healthy"
            reason = "ok"

        return {
            "status": status,
            "reason": reason,
            "backup_age_hours": round(backup_age_hours, 1),
            "backup_size_mb": backup_size_mb,
            "concept_count": concept_count,
            "integrity": integrity,
            "backup_path": str(backup_path),
            "timestamp": _utc_now_iso(),
        }
    except Exception as e:
        return {
            "status": "critical",
            "reason": "read_error",
            "message": _safe_error(e),
            "backup_path": str(backup_path),
            "timestamp": _utc_now_iso(),
        }


@app.post("/pith/benchmark", dependencies=[Depends(verify_api_key)])
async def benchmark_endpoint(mode: str = "full"):
    """AF-05: Run CogGov-Bench behavioral governance benchmark.

    The benchmark logic lives in app/coggov_bench.py (992+ lines).
    Amendment 7: 5-minute timeout to prevent server hangs.
    """
    import asyncio as _asyncio

    if mode not in ("light", "full"):
        raise HTTPException(status_code=400, detail="mode must be 'light' or 'full'")

    import threading as _threading

    async def _wait_for_worker_shutdown(reason: str) -> None:
        if bench_task is None:
            return
        try:
            await _asyncio.wait_for(bench_task, timeout=5)
        except CogGovBenchCancelled:
            logger.info("Benchmark worker acknowledged %s cancellation", reason)
        except Exception as exc:
            logger.warning(
                "Benchmark worker stopped after %s with %s: %s",
                reason,
                type(exc).__name__,
                exc,
            )

    cancel_event = _threading.Event()
    bench_task = None
    try:
        from app.ops.coggov_bench import CogGovBenchCancelled, run_coggov_bench

        bench_task = _asyncio.create_task(_asyncio.to_thread(run_coggov_bench, mode=mode, cancel_event=cancel_event))
        done, _ = await _asyncio.wait({bench_task}, timeout=300)  # 5 minute max without cancelling worker
        if not done:
            raise TimeoutError
        result = bench_task.result()
        return result.to_dict() if hasattr(result, "to_dict") else result
    except TimeoutError:
        cancel_event.set()
        await _wait_for_worker_shutdown("timeout")
        raise HTTPException(status_code=504, detail=f"Benchmark timed out after 300s in {mode} mode")
    except _asyncio.CancelledError:
        cancel_event.set()
        await _wait_for_worker_shutdown("request")
        raise
    except CogGovBenchCancelled:
        raise HTTPException(status_code=503, detail="Benchmark cancelled before completion")
    except Exception as e:
        raise HTTPException(status_code=500, detail=_safe_error(e))


# --- Profile Management ---


@app.get("/profiles")
def get_profiles():
    """List available data profiles under ~/pith-data/."""
    from app.core.profile import get_active_profile, list_profiles

    return {
        "active": get_active_profile(),
        "profiles": list_profiles(),
    }


@app.get("/profile")
def get_profile():
    """Return active profile name and resolved data path."""
    from app.core.profile import get_active_profile, resolve_data_dir

    return {
        "profile": get_active_profile(),
        "data_dir": str(resolve_data_dir()),
    }


# --- Maintenance / Autonomous Cognition ---


@app.post("/maintenance", dependencies=[Depends(verify_api_key)])
async def maintenance_endpoint(request: Request, body: dict = {}):
    """Run autonomous maintenance cycle (sleeptime compute).

    Body params:
      phases: list[int] — which phases to run (default: all 1-5)
      dry_run: bool — preview without executing (default: false)

    Phases:
      1: Scheduled async tasks (currency scan, authority recal, etc.)
      2: Reflection cycle (decay, forgetting, strengthening, merging)
      3: Experiment generation (synthesis, hypothesis, counterfactual, analogy)
      4: Curiosity — question generation for weak concepts
      5: Health report + degradation alerts
    """
    from app.ops.maintenance import run_maintenance
    from app.ops.maintenance_scheduler import maintenance_lock  # A2: Shared mutex

    phases = body.get("phases")
    dry_run = body.get("dry_run", False)
    async with maintenance_lock:
        report = await run_maintenance(phases=phases, dry_run=dry_run)
    return report.to_dict()


@app.get("/maintenance/status", dependencies=[Depends(verify_api_key)])
def maintenance_status():
    """Get status of all async tasks and degradation alerts."""
    from app.ops.async_tasks import task_runner
    from app.storage import get_db_connection

    conn = get_db_connection()
    return task_runner.get_status(conn)


# --- OPS-080: Consumer Status Endpoint ---


@app.get("/status")
def status_endpoint():
    """Public status endpoint showing tier, features, maintenance, and auth state.

    No auth required — returns non-sensitive operational metadata only.
    """
    from app.core.config import (
        EDGE_LLM_RECLASSIFICATION_ENABLED,
        FEATURE_FLAGS,
        KA_LLM_RECLASSIFICATION_ENABLED,
        LLM_TIER,
    )
    from app.ops.maintenance_scheduler import (
        MAINTENANCE_INTERVAL_SECONDS,
        _circuit_open,
        _consecutive_failures,
        _scheduler_task,
    )

    # Read heartbeat if available
    scheduler_status = "unknown"
    try:
        from app.core.profile import resolve_data_dir

        heartbeat_path = Path(resolve_data_dir()) / "maintenance_heartbeat.json"
        if heartbeat_path.exists():
            import json

            hb = json.loads(heartbeat_path.read_text())
            scheduler_status = hb.get("status", "unknown")
    except Exception:
        pass

    # LLM feature summary
    llm_features = {
        "tier3_extraction": FEATURE_FLAGS.get("TIER3_LLM_EXTRACTION_ENABLED", False),
        "experiment_resolution": FEATURE_FLAGS.get("LLM_EXPERIMENT_RESOLUTION_ENABLED", False),
        "contradiction_tier2": FEATURE_FLAGS.get("LLM_CONTRADICTION_TIER2_ENABLED", False),
        "edge_reclassification": EDGE_LLM_RECLASSIFICATION_ENABLED,
        "ka_reclassification": KA_LLM_RECLASSIFICATION_ENABLED,
    }

    return {
        "llm_tier": LLM_TIER,
        "llm_tier_label": {0: "offline", 1: "commodity", 2: "frontier"}.get(LLM_TIER, "unknown"),
        "llm_features": llm_features,
        "auth": {
            "api_key_set": bool(API_KEY),
            "auto_generated": "PITH_API_KEY" not in (os.environ.get("_PITH_ORIGINAL_ENV", "")),
        },
        "maintenance": {
            "scheduler_active": _scheduler_task is not None and not _scheduler_task.done()
            if _scheduler_task
            else False,
            "scheduler_status": scheduler_status,
            "interval_seconds": MAINTENANCE_INTERVAL_SECONDS,
            "circuit_open": _circuit_open,
            "consecutive_failures": _consecutive_failures,
        },
        "server_version": "1.1.0",
    }


# --- RETRIEVAL-019: Evolution Backfill Endpoints ---


@app.post("/backfill/run", dependencies=[Depends(verify_api_key)])
async def backfill_run(request: Request, body: dict = {}):
    """Run evolution backfill for a knowledge area.

    Body params:
        knowledge_area (str): Target KA to process (required)
        dry_run (bool): If true, evaluate but don't commit (default: false)
        window_days (int): Max age gap between pairs (default: 14)
        auto_commit (bool): Auto-commit approved pairs (default: true)
    """
    from app.ops.backfill import run_backfill

    knowledge_area = body.get("knowledge_area")
    if not knowledge_area:
        raise HTTPException(status_code=400, detail="knowledge_area is required")

    result = run_backfill(
        knowledge_area=knowledge_area,
        dry_run=body.get("dry_run", False),
        window_days=body.get("window_days", 14),
        auto_commit=body.get("auto_commit", True),
    )

    return {
        "batch_id": result.batch_id,
        "knowledge_area": result.knowledge_area,
        "phase": result.phase,
        "candidates_generated": result.candidates_generated,
        "pairs_evaluated": result.pairs_evaluated,
        "auto_approved": result.auto_approved,
        "manual_review": result.manual_review,
        "auto_rejected": result.auto_rejected,
        "committed": result.committed,
        "exec_rejected": result.exec_rejected,
        "duration_ms": round(result.duration_ms, 1),
        "errors": result.errors,
    }


@app.get("/backfill/status", dependencies=[Depends(verify_api_key)])
def backfill_status(batch_id: str):
    """Get status summary for a backfill batch."""
    from app.ops.backfill import get_batch_status

    return get_batch_status(batch_id)


@app.post("/backfill/rollback", dependencies=[Depends(verify_api_key)])
async def backfill_rollback(request: Request, body: dict = {}):
    """Rollback a committed backfill batch."""
    from app.ops.backfill import rollback_batch

    batch_id = body.get("batch_id")
    if not batch_id:
        raise HTTPException(status_code=400, detail="batch_id is required")

    count = rollback_batch(batch_id)
    return {"batch_id": batch_id, "rolled_back": count}


# --- THREAD-004: Thread Reorganization Endpoints ---


@app.post("/thread_reorg/mine", dependencies=[Depends(verify_api_key)])
def thread_reorg_mine(body: dict = {}):
    from app.ops.thread_reorg import mine_active_thread

    _require_thread_reorg_ready(require_retrieval_ready=True)
    _acquire_thread_reorg_slot()
    try:
        result = mine_active_thread(body.get("source_thread_id"))
        include_merge_decisions = bool(body.get("include_merge_decisions", False))
        return {
            "status": "ok",
            "source_thread_id": result.source_thread_id,
            "source_thread_title": result.source_thread_title,
            "resolved_count": result.resolved_count,
            "stale_concept_ids": result.stale_concept_ids,
            "cluster_count": result.cluster_count,
            "clusters": result.clusters,
            "candidate_stats": result.candidate_stats,
            "deferred_singletons": result.deferred_singletons,
            "percentile_snapshot": result.percentile_snapshot,
            "merge_decision_count": len(result.merge_decisions),
            "merge_decisions_truncated": not include_merge_decisions,
            "merge_decisions": result.merge_decisions if include_merge_decisions else [],
        }
    finally:
        _HEAVY_ENDPOINT_SEMAPHORE.release()


@app.post("/thread_reorg/batch/preview", dependencies=[Depends(verify_api_key)])
def thread_reorg_batch_preview(body: dict):
    from app.ops.thread_reorg import preview_batch

    _require_thread_reorg_ready(require_write=True)
    _acquire_thread_reorg_slot()
    try:
        if not body.get("source_thread_id"):
            raise HTTPException(400, "source_thread_id is required")
        if not body.get("clusters"):
            raise HTTPException(400, "clusters are required")
        payload = preview_batch(
            source_thread_id=body["source_thread_id"],
            clusters=body["clusters"],
            evaluation_set_id=body.get("evaluation_set_id"),
            max_batch_size=body.get("max_batch_size"),
        )
        payload["batch_status"] = payload.pop("status")
        return {
            "status": "ok",
            **payload,
        }
    finally:
        _HEAVY_ENDPOINT_SEMAPHORE.release()


@app.post("/thread_reorg/batch/preview_residual", dependencies=[Depends(verify_api_key)])
def thread_reorg_batch_preview_residual(body: dict):
    from app.ops.thread_reorg import preview_deferred_review_batch

    _require_thread_reorg_ready(require_write=True, require_retrieval_ready=True)
    _acquire_thread_reorg_slot()
    try:
        if not body.get("source_thread_id"):
            raise HTTPException(400, "source_thread_id is required")
        try:
            payload = preview_deferred_review_batch(
                source_thread_id=body["source_thread_id"],
                bucket_ids=body.get("bucket_ids"),
                evaluation_set_id=body.get("evaluation_set_id"),
                max_batch_size=body.get("max_batch_size"),
            )
        except ValueError as exc:
            detail = str(exc)
            if detail in {
                "No deferred singleton buckets available for review",
                "No deferred bucket concepts passed the residual review quality gates",
            }:
                raise HTTPException(409, detail) from exc
            raise HTTPException(400, detail) from exc
        payload["batch_status"] = payload.pop("status")
        return {
            "status": "ok",
            **payload,
        }
    finally:
        _HEAVY_ENDPOINT_SEMAPHORE.release()


@app.post("/thread_reorg/batch/commit", dependencies=[Depends(verify_api_key)])
def thread_reorg_batch_commit(body: dict):
    from app.ops.thread_reorg import commit_batch

    _require_thread_reorg_ready(require_write=True)
    _acquire_thread_reorg_slot()
    try:
        batch_id = body.get("batch_id")
        if not batch_id:
            raise HTTPException(400, "batch_id is required")
        try:
            payload = commit_batch(batch_id)
        except ValueError as exc:
            if str(exc) == "THREAD_REORG_BATCH_WRITE_ENABLED is disabled":
                raise HTTPException(409, "thread reorg batch commit is disabled on this runtime") from exc
            raise
        payload["batch_status"] = payload.pop("status")
        return {"status": "ok", **payload}
    finally:
        _HEAVY_ENDPOINT_SEMAPHORE.release()


@app.get("/thread_reorg/batch/status", dependencies=[Depends(verify_api_key)])
def thread_reorg_batch_status(batch_id: str, include_members: bool = False, member_limit: int = 100):
    from app.ops.thread_reorg import get_batch_status

    payload = get_batch_status(batch_id, include_members=include_members, member_limit=member_limit)
    payload["batch_status"] = payload.pop("status")
    return {"status": "ok", **payload}


@app.post("/thread_reorg/batch/rollback", dependencies=[Depends(verify_api_key)])
def thread_reorg_batch_rollback(body: dict):
    from app.ops.thread_reorg import rollback_batch

    _require_thread_reorg_ready(require_write=True)
    _acquire_thread_reorg_slot()
    try:
        batch_id = body.get("batch_id")
        if not batch_id:
            raise HTTPException(400, "batch_id is required")
        payload = rollback_batch(batch_id)
        payload["batch_status"] = payload.pop("status")
        return {"status": "ok", **payload}
    finally:
        _HEAVY_ENDPOINT_SEMAPHORE.release()


# --- Phase 1A: Forgetting Recovery (Spec deviation — not in any spec) ---


@app.post("/pith_recover", dependencies=[Depends(verify_api_key)])
def pith_recover(concept_id: str):
    """Restore an archived (forgotten) concept back to active graph.

    Forgetting archives concepts, but recovery must always be possible.
    Archived concepts are never deleted.
    """
    if restore_concept(concept_id):
        retrieval_engine.build_index()  # Re-include in search
        return {
            "status": "restored",
            "concept_id": concept_id,
            "message": f"Concept {concept_id} restored from archive to active graph",
        }
    raise HTTPException(status_code=404, detail=f"Concept {concept_id} not found in archive")


@app.get("/pith_archived")
def pith_list_archived():
    """List all archived (forgotten) concepts."""
    archived = list_archived_concepts()
    return {"archived_count": len(archived), "concept_ids": archived}


@app.get("/pith_introspect")
def pith_introspect(mode: str = "summary", update: bool = False):
    """Return cognitive self-assessment.

    Modes:
      summary          — identity, health, top_strengths, weakest_areas, recent_errors
      full             — complete SelfModel object
      capability_check — CognitiveCapabilityInventory only
      epistemic_check  — EpistemicProfile only

    Latency targets:
      update=false: <200ms (cached read)
      update=true:  <500ms (recompute from live data, any mode)

    Phase 1A deviations:
      - 'weakest_areas' replaces 'top_gaps' (blind_spots is stubbed)
      - 'recent_errors' returns empty (error_history is stubbed)
    """
    if mode not in ("summary", "full", "capability_check", "epistemic_check"):
        raise HTTPException(
            status_code=400, detail=f"Invalid mode '{mode}'. Use: summary, full, capability_check, epistemic_check"
        )
    try:
        # PERF-077: Pre-load concepts via read_snapshot_db to avoid O(N) writer-lock
        # contention in introspect fallback path. Only needed when update=True
        # triggers SelfModel regeneration from live concept data.
        from app.storage.concepts import list_concepts_full

        concepts = list_concepts_full() if update else None
        return self_model_manager.introspect(mode=mode, update=update, concepts=concepts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=_safe_error(e))


@app.post("/pith_activate_context", dependencies=[Depends(verify_api_key)])
def pith_activate_context(context: str, boost: float = 0.5):
    """
    Activate concepts based on current context.

    Pre-loads related concepts for faster retrieval.
    """
    try:
        predictive_activation.activate_from_context(context, boost)
        active = predictive_activation.get_active_concepts()
        return {"activated": len(active), "concepts": [{"concept_id": cid, "activation": act} for cid, act in active]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=_safe_error(e))


@app.get("/pith_activation_state")
def pith_activation_state():
    """Get current activation state."""
    try:
        state = predictive_activation.get_activation_state()
        return state
    except Exception as e:
        raise HTTPException(status_code=500, detail=_safe_error(e))


@app.post("/pith_set_goal", dependencies=[Depends(verify_api_key)])
def pith_set_goal(goal: str, context: dict = None):
    """
    Set current goal for goal-directed retrieval.

    Goals: improve_process, solve_problem, understand_system,
           make_decision, learn_topic, plan_project, etc.
    """
    try:
        goal_directed.set_goal(goal, context or {})
        return {"status": "goal_set", "goal": goal, "message": f"Goal '{goal}' activated for retrieval boosting"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=_safe_error(e))


@app.post("/pith_clear_goal", dependencies=[Depends(verify_api_key)])
def pith_clear_goal():
    """Clear the current goal."""
    try:
        goal_directed.clear_goal()
        return {"status": "goal_cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=_safe_error(e))


@app.get("/pith_predict_next")
def pith_predict_next(current_concepts: list[str], num_predictions: int = 5):
    """
    Predict which concepts are likely to be needed next.

    Based on activation patterns and associations.
    """
    try:
        predictions = predictive_activation.predict_next_concepts(current_concepts, num_predictions)
        return {"predictions": predictions, "count": len(predictions)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=_safe_error(e))


@app.post("/pith_reset_activation", dependencies=[Depends(verify_api_key)])
def pith_reset_activation():
    """Reset all concept activations (new session/context)."""
    try:
        predictive_activation.reset()
        return {"status": "reset", "message": "All activations cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=_safe_error(e))


@app.post("/pith_import_conversation", dependencies=[Depends(verify_api_key)])
def pith_import_conversation(
    conversation_text: str, source_id: str = "manual_import", knowledge_area: str = "imported", chunk_size: int = 200
):
    """
    Import and learn from a historical conversation.

    Processes conversation text and extracts learnable concepts.
    """
    try:
        result = conversation_processor.process_conversation(
            conversation_text=conversation_text,
            source_id=source_id,
            knowledge_area=knowledge_area,
            chunk_size=chunk_size,
        )

        # Rebuild index after import
        retrieval_engine.build_index()

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=_safe_error(e))


@app.post("/pith_import_batch", dependencies=[Depends(verify_api_key)])
def pith_import_batch(conversations: list[dict], knowledge_area: str = "imported"):
    """
    Import multiple conversations in batch.

    Uses conversation_processor.process_batch() which writes directly to storage
    via create_concept() — does NOT go through session_learn, so rate limits and
    daily budgets do not apply. INGEST-042 audit confirmed the original INGEST-023
    rate elevation here was dead code (process_batch never calls session_learn).

    Args:
        conversations: List of {"text": str, "source_id": str} dicts
        knowledge_area: Knowledge area for imported concepts

    Returns:
        Batch processing summary
    """
    try:
        logger.info(f"pith_import_batch: processing {len(conversations)} conversations")

        result = conversation_processor.process_batch(conversations=conversations, knowledge_area=knowledge_area)

        # Rebuild index after batch import
        retrieval_engine.build_index()

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=_safe_error(e))


@app.get("/pith_import_stats")
def pith_import_stats():
    """Get retrospective import statistics."""
    try:
        stats = conversation_processor.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=_safe_error(e))


# ============================================
# Conversation Import Pipeline (v2)
# ============================================


@app.post("/pith_import_file", dependencies=[Depends(verify_api_key)])
def pith_import_file(
    file_path: str,
    source: str,
    skip_report: bool = False,
    resume: bool = False,
):
    """Import conversation history from ChatGPT/Claude export files.

    Full pipeline: parse → normalize → bulk import → report generation.
    Spec: CONVERSATION_IMPORT_PIPELINE_v2.md (gauntlet 8.5/10 PASS)
    """
    from pathlib import Path as _Path

    from app.ops.import_pipeline import run_import_pipeline

    try:
        from app.core.profile import resolve_data_dir

        checkpoint_dir = _Path(resolve_data_dir()) / "import_checkpoints"
        result = run_import_pipeline(
            file_path=file_path,
            source=source,
            skip_report=skip_report,
            resume=resume,
            checkpoint_dir=checkpoint_dir,
        )
        # Rebuild index after import
        retrieval_engine.build_index()
        return result
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"pith_import_file error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_safe_error(e))


@app.post("/pith_import_cancel", dependencies=[Depends(verify_api_key)])
def pith_import_cancel():
    """Cancel an in-progress import."""
    from app.ops.import_pipeline import cancel_import

    cancel_import()
    return {"status": "cancel_requested"}


# ============================================
# KA-ARCH-001: Dynamic KA Hints Endpoint
# ============================================


@app.get("/pith_ka_hints")
def pith_ka_hints(max_hints: int = 12) -> dict:
    """Get dynamic KA vocabulary hints for extraction prompts.

    Returns the user's most active knowledge areas, ordered by maturity
    and concept count. Use these hints in extraction prompts to align
    vocabulary with the user's established domain language.

    KA-ARCH-001 Fix 9: Adaptive extraction primitive.
    """
    try:
        from app.cognitive.taxonomy import get_ka_hints

        hints = get_ka_hints(max_hints)
        return {"ka_hints": hints, "count": len(hints)}
    except Exception as e:
        logger.error(f"pith_ka_hints failed: {e}")
        return {"ka_hints": ["knowledge", "workflow", "relationships", "context", "goals", "observations"], "count": 6}


# ============================================
# PITH INTEGRATION ENDPOINTS
# ============================================


@app.get("/knowledge_areas")
def list_knowledge_areas_endpoint() -> dict[str, Any]:
    """List all knowledge areas with concept counts."""
    return _build_knowledge_areas_cached()


@app.get("/knowledge_areas/{area_name}")
def get_knowledge_area_concepts(area_name: str) -> dict[str, Any]:
    """Get all concepts in a specific knowledge area."""
    from app.storage import list_concepts_for_knowledge_area

    area_concepts = list_concepts_for_knowledge_area(area_name)
    if not area_concepts:
        raise HTTPException(status_code=404, detail=f"Knowledge area '{area_name}' not found")

    return {"area": area_name, "concept_count": len(area_concepts), "concepts": area_concepts}


@app.post("/pith_suggest_associations", dependencies=[Depends(verify_api_key)])
def suggest_associations(concept_id: str, max_suggestions: int = 5) -> list[dict]:
    """Suggest potential associations based on semantic similarity."""
    concept = load_concept(concept_id, track_access=False)
    if not concept:
        raise HTTPException(status_code=404, detail=f"Concept {concept_id} not found")

    query = SearchQuery(query=concept.summary, max_results=max_suggestions + 1)
    results = retrieval_engine.search(query)

    suggestions = []
    for result in results:
        if result.concept_id != concept_id:
            source_ka = concept.metadata.get("knowledge_area", "unknown")
            target_ka = result.knowledge_area or "unknown"

            if source_ka == target_ka:
                relation = "related_to"
                reasoning = f"Both in {source_ka} knowledge area"
            else:
                relation = "related_to"
                reasoning = f"Cross-domain: {source_ka} <-> {target_ka}"

            suggestions.append(
                {
                    "concept_id": result.concept_id,
                    "similarity_score": round(result.relevance_score, 2),
                    "suggested_relation": relation,
                    "reasoning": reasoning,
                }
            )

    return suggestions[:max_suggestions]


@app.get("/pith_concept_timeline")
def concept_timeline(limit: int = 20) -> list[dict]:
    """Get recent concept creation timeline."""
    all_concepts = list_concepts_full()  # Single bulk query — no N+1
    timeline = []

    for concept in all_concepts:
        timeline.append(
            {
                "id": concept.id,
                "created_at": concept.created_at,
                "knowledge_area": concept.metadata.get("knowledge_area", "unknown"),
                "confidence": concept.confidence,
                "summary": concept.summary[:100] + "..." if len(concept.summary) > 100 else concept.summary,
            }
        )

    timeline.sort(key=lambda x: x["created_at"], reverse=True)
    return timeline[:limit]


@app.get("/pith_search_suggestions")
def search_suggestions(query: str, limit: int = 5) -> list[str]:
    """Get search query suggestions based on existing concepts."""
    with request_db_scope("pith_search_suggestions"):
        if _read_only_aggregates_enabled():
            try:
                with read_snapshot_db("pith_search_suggestions") as conn:
                    all_concepts = list_concepts_full(conn=conn)
            except Exception as snap_err:
                if not _read_only_aggregates_fallback_allowed():
                    raise
                logger.warning(
                    "read-only snapshot failed for /pith_search_suggestions, falling back: %s",
                    snap_err,
                )
                all_concepts = list_concepts_full()
        else:
            all_concepts = list_concepts_full()  # Single bulk query — no N+1
        suggestions = set()
        query_lower = query.lower()

        for concept in all_concepts:
            ka = concept.metadata.get("knowledge_area", "")
            if query_lower in ka.lower():
                suggestions.add(ka)
            if query_lower in concept.id.lower():
                suggestions.add(concept.id.replace("_", " "))
            if len(suggestions) >= limit * 2:
                break

        return sorted(list(suggestions))[:limit]


# ============================================================
# Auto-Association Endpoints (Phase 1.3)
# ============================================================


@app.post("/auto_associate_batch", dependencies=[Depends(verify_api_key)])
def auto_associate_batch_endpoint(request: AutoAssociateBatchRequest = None):
    """Run batch auto-association pipeline across all active concepts.

    Two-tier strategy:
      Tier 1 — cosine similarity above tier1_threshold (default 0.12)
      Tier 2 — lower cosine + same knowledge_area for remaining orphans

    All parameters optional with sensible defaults. Use dry_run=true to preview.
    """
    if request is None:
        request = AutoAssociateBatchRequest()
    try:
        result = auto_associate_batch(request)
        return result.model_dump()
    except Exception as e:
        logger.error(f"auto_associate_batch error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_safe_error(e))


@app.post("/auto_associate/{concept_id}", dependencies=[Depends(verify_api_key)])
def auto_associate_single_endpoint(concept_id: str, request: AutoAssociateSingleRequest = None):
    """Auto-associate a single concept with its nearest neighbors.

    Finds similar concepts via TF-IDF cosine similarity and creates
    'related_to' edges for matches above the threshold.
    """
    if request is None:
        request = AutoAssociateSingleRequest()
    try:
        result = auto_associate_single(concept_id, request)
        if not result.matches and result.edges_created == 0:
            concept = load_concept(concept_id, track_access=False)
            if not concept:
                raise HTTPException(status_code=404, detail=f"Concept {concept_id} not found")
        return result.model_dump()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"auto_associate_single error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_safe_error(e))


# ============================================================
# Session Middleware (Phase 1A D7)
# ============================================================


@app.get("/pith_orient")
def pith_orient(
    time_window: str = "7_days",
    include_threads: bool = True,
    include_recommendations: bool = True,
    force_refresh: bool = False,
    include_workstreams: bool = False,
    workstream_limit: int = 8,
    origin_id: str | None = None,
    session_id: str | None = None,
):
    """Generate present moment orientation.

    Returns where-been, where-am, where-going for session bootstrap.
    """
    valid_windows = {"1_day", "7_days", "30_days", "all"}
    if time_window not in valid_windows:
        raise HTTPException(status_code=400, detail=f"Invalid time_window. Must be one of: {valid_windows}")

    payload = _pith_orient_base_payload(
        time_window=time_window,
        force_refresh=force_refresh,
    )
    if include_workstreams:
        payload = _pith_orient_payload_with_workstreams(
            payload,
            workstream_limit=workstream_limit,
            origin_id=origin_id,
            session_id=session_id,
        )
    else:
        payload.pop("workstreams", None)
    return payload


@app.post("/session_start", dependencies=[Depends(verify_api_key)])
def session_start(body: dict = None):
    """Bootstrap session with single concept load.

    Returns introspect summary + orientation in one call.
    """
    context_hint = ""
    agent_id = "default"
    platform_hint = "unknown"  # SESSION-012 v0.3
    if body and isinstance(body, dict):
        context_hint = body.get("context_hint", "")
        agent_id = body.get("agent_id", "default")
        platform_hint = body.get("platform_hint", "unknown")

    result = session_manager.start_session(context_hint=context_hint, agent_id=agent_id, platform_hint=platform_hint)
    # start_session now returns dict (with optional active_checkpoint attached)
    if hasattr(result, "model_dump"):
        return result.model_dump()
    return result


@app.post("/session_end", dependencies=[Depends(verify_api_key)])
async def session_end(request: Request):
    """End current session. Optionally accepts last-exchange data for flush.
    Flushes access tracker, triggers reflection if learning_event_count >= threshold."""
    from app.ops.metrics import metrics as _se_metrics

    with _se_metrics.timer("session_end_latency_ms"):
        start = time.perf_counter()
        body = None
        end_request = None
        request_id = None
        try:
            body = await request.json()
        except Exception:
            body = None

        recognized_fields = {
            "request_id",
            "session_id",
            "origin_id",
            "previous_response",
            "previous_message",
            "extracted_concepts_json",
            "agent_id",
        }
        if body and isinstance(body, dict):
            request_id = body.get("request_id")
            if any(key in body for key in recognized_fields):
                try:
                    end_request = SessionEndRequest(**body)
                except ValidationError as exc:
                    raise HTTPException(
                        status_code=400,
                        detail=exc.errors(include_context=False),
                    ) from exc

        request_id = _ensure_session_end_request_id(request_id)
        if end_request is not None and end_request.request_id != request_id:
            end_request = end_request.model_copy(update={"request_id": request_id})
        request_payload = dict(body) if isinstance(body, dict) else {}
        request_payload["request_id"] = request_id

        try:
            replay_state = begin_write_request("session_end", request_id, request_payload=request_payload)
        except HTTPException as exc:
            if exc.status_code == 409:
                return _session_end_processing_payload(
                    request_id,
                    processing_time_ms=(time.perf_counter() - start) * 1000,
                )
            raise
        if replay_state.replay is not None:
            return replay_state.replay

        future = _get_session_learn_executor().submit(lambda: session_manager.end_session(end_request))
        try:
            result = future.result(timeout=_remaining_session_learn_sync_wait(start))
            return _commit_session_end_result(request_id, result)
        except concurrent.futures.TimeoutError:
            future.add_done_callback(lambda done_future: _finalize_deferred_session_end(done_future, request_id))
            _record_session_learn_contract_metric("session_end_deferred")
            _schedule_write_replay_reclaimer("session_end", "deferred_session_end")
            return _session_end_processing_payload(
                request_id,
                processing_time_ms=(time.perf_counter() - start) * 1000,
            )
        except Exception as exc:
            abandon_write_request("session_end", request_id, error_class=type(exc).__name__)
            raise


@app.post("/conversation_turn", dependencies=[Depends(verify_api_key)])
def conversation_turn_endpoint(
    request: ConversationTurnRequest,
    background_tasks: BackgroundTasks,
    x_pith_transport: str | None = Header(None),
):
    """Pre-response context activation. Given a user message, find and return
    the most relevant existing knowledge. Read-only. Target: <50ms."""
    if getattr(app.state, "startup_task", None) is not None:
        _require_retrieval_ready("conversation_turn")
    # PERF-FORT-1: Semaphore prevents threadpool starvation under concurrent load
    acquired = _HEAVY_ENDPOINT_SEMAPHORE.acquire(timeout=HEAVY_ENDPOINT_TIMEOUT_S)
    if not acquired:
        raise HTTPException(
            status_code=503,
            detail="Server under heavy load — try again in a few seconds",
            headers={"Retry-After": "3"},
        )
    binding = None
    try:
        request.transport_mode = x_pith_transport
        binding = session_manager.prepare_conversation_turn_binding(request)
        result = session_manager.conversation_turn(request)
        result.bind_status = binding["bind_status"]
        result.binding_source = binding["binding_source"]
        result.resolved_session_id = binding["resolved_session_id"]
        if binding["bind_status"] == "unbound":
            result.working_context = None
            result.checkpoint_resume_available = False
        # EUNOMIA-039 Fix 3: Dispatch autolearn AFTER response is sent to client
        background_tasks.add_task(session_manager.dispatch_post_response_tasks, result)
        return result.model_dump()
    except Exception as e:
        from app.governance.repo_hygiene_policy import RepoHygienePolicyError

        if isinstance(e, RepoHygienePolicyError):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": e.error_code,
                    "message": e.detail,
                    "workspace_context": e.workspace_context,
                },
            )
        if e.__class__.__name__ == "InvalidSessionBindingError":
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "invalid_session_id",
                    "message": str(e),
                    "session_id": getattr(e, "session_id", None),
                },
            ) from e
        logger.error(f"conversation_turn error: {e}")
        raise HTTPException(status_code=500, detail=_safe_error(e))
    finally:
        if binding is not None:
            session_manager._pop_request_session(
                binding["session_token"],
                binding["active_token"],
            )
        _HEAVY_ENDPOINT_SEMAPHORE.release()


def _get_session_learn_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Return the bounded executor used for direct session_learn writes."""
    global _SESSION_LEARN_EXECUTOR
    if _SESSION_LEARN_EXECUTOR is None:
        with _SESSION_LEARN_EXECUTOR_LOCK:
            if _SESSION_LEARN_EXECUTOR is None:
                _SESSION_LEARN_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
                    max_workers=_SESSION_LEARN_EXECUTOR_WORKERS,
                    thread_name_prefix="session_learn",
                )
    return _SESSION_LEARN_EXECUTOR


def _record_session_learn_contract_metric(metric: str) -> None:
    try:
        from app.ops.metrics import metrics as _sl_contract_metrics

        _sl_contract_metrics.record(metric, 1.0)
        _sl_contract_metrics.flush()
    except Exception:
        pass


def _record_session_learn_initial_response_latency(start: float, state: str) -> None:
    try:
        from app.ops.metrics import metrics as _sl_contract_metrics

        _sl_contract_metrics.record(
            "session_learn_initial_response_latency_ms",
            (time.perf_counter() - start) * 1000,
            {"state": state},
        )
        _sl_contract_metrics.flush()
    except Exception:
        pass


def _record_session_learn_lifecycle_latency(metric: str, start: float, labels: dict | None = None) -> None:
    try:
        from app.ops.metrics import metrics as _sl_lifecycle_metrics

        _sl_lifecycle_metrics.record(
            metric,
            (time.perf_counter() - start) * 1000,
            labels or {},
        )
        _sl_lifecycle_metrics.flush()
    except Exception:
        pass


def _remaining_session_learn_sync_wait(start: float) -> float:
    return max(0.0, _SESSION_LEARN_SYNC_WAIT_SECONDS - (time.perf_counter() - start))


def _ensure_session_learn_request_id(request: SessionLearnRequest) -> tuple[SessionLearnRequest, str]:
    request_id = request.request_id or f"sl_srv_{uuid.uuid4().hex[:16]}"
    if request.request_id == request_id:
        return request, request_id
    try:
        request.request_id = request_id
        return request, request_id
    except Exception:
        return request.model_copy(update={"request_id": request_id}), request_id


def _session_learn_processing_payload(request_id: str, *, processing_time_ms: float = 0.0) -> dict:
    response = SessionLearnResponse(
        concepts_created=[],
        concepts_evolved=[],
        associations_created=0,
        duplicates_skipped=0,
        concepts_skipped=0,
        errors=0,
        processing_time_ms=round(processing_time_ms, 2),
        learning_events=0,
        extraction_source_breakdown={},
        budget_warnings=[
            "session_learn_processing: learning is still running; retry with the same request_id"
        ],
        persistence_state="processing",
        processing_state="processing",
        request_id=request_id,
        retry_after_seconds=_SESSION_LEARN_PROCESSING_RETRY_AFTER_SECONDS,
    )
    return response.model_dump()


def _enqueue_session_learn_lifecycle_job(request: SessionLearnRequest, request_id: str) -> dict:
    from app.session.lifecycle_jobs_runtime import enqueue_session_learn_job, submit_lifecycle_drain

    enqueue_start = time.perf_counter()
    job = enqueue_session_learn_job(
        learn_request=request,
        request_id=request_id,
        priority=_SESSION_LEARN_LIFECYCLE_PRIORITY,
    )
    _record_session_learn_lifecycle_latency(
        "session_learn_lifecycle_enqueue_latency_ms",
        enqueue_start,
        {"status": str(job.get("status", "unknown"))},
    )
    submitted = submit_lifecycle_drain(
        run_job=_run_session_learn_lifecycle_job,
        reason="explicit_session_learn",
        limit=1,
        source="session_learn",
    )
    if submitted:
        _record_session_learn_contract_metric("session_learn_lifecycle_drain_submitted")
    else:
        _record_session_learn_contract_metric("session_learn_lifecycle_drain_already_running")
    return job


def _ensure_session_end_request_id(request_id: str | None) -> str:
    return request_id or f"se_srv_{uuid.uuid4().hex[:16]}"


def _session_end_processing_payload(request_id: str, *, processing_time_ms: float = 0.0) -> dict:
    return {
        "status": "processing",
        "persistence_state": "processing",
        "processing_state": "processing",
        "request_id": request_id,
        "retry_after_seconds": _SESSION_LEARN_PROCESSING_RETRY_AFTER_SECONDS,
        "final_learning_state": "processing",
        "processing_time_ms": round(processing_time_ms, 2),
        "budget_warnings": [
            "session_end_processing: closeout is still running; retry with the same request_id"
        ],
    }


def _session_learn_stale_cutoff_iso() -> str:
    from app.api.write_durability import STALE_PROCESSING_TIMEOUT

    return (_utc_now() - STALE_PROCESSING_TIMEOUT).isoformat()


def _iso_age_seconds(value: str | None) -> int | None:
    if not value:
        return None
    try:
        from datetime import datetime

        return max(0, int((_utc_now() - _ensure_aware(datetime.fromisoformat(value))).total_seconds()))
    except Exception:
        return None


def _build_session_learn_queue_health() -> dict:
    try:
        from app.core.profile import get_active_profile
        from app.storage import summarize_write_request_processing

        stale_cutoff = _session_learn_stale_cutoff_iso()
        summary = summarize_write_request_processing(
            "session_learn",
            get_active_profile(),
            stale_cutoff,
            max_attempts=_SESSION_LEARN_RECLAIMER_MAX_ATTEMPTS,
        )
        stale_count = int(summary["stale_processing_count"])
        status = "warning" if stale_count else "ok"
        oldest_age = _iso_age_seconds(summary.get("oldest_processing_updated_at"))
        stale_age = _iso_age_seconds(summary.get("oldest_stale_updated_at"))
        return {
            "status": status,
            "alert": stale_count > 0,
            "stale_threshold_seconds": int(_cfg.WRITE_STALE_MINUTES * 60),
            "processing_count": int(summary["processing_count"]),
            "stale_processing_count": stale_count,
            "recoverable_stale_count": int(summary["recoverable_stale_count"]),
            "unrecoverable_stale_count": int(summary["unrecoverable_stale_count"]),
            "oldest_processing_age_seconds": oldest_age,
            "oldest_stale_age_seconds": stale_age,
            "oldest_stale_request_id": summary.get("oldest_stale_request_id"),
            "max_attempts_exhausted_count": int(summary["max_attempts_exhausted_count"]),
        }
    except Exception as exc:
        return {
            "status": "unknown",
            "alert": True,
            "reason": "session_learn_queue_health_failed",
            "error": _safe_error(exc),
        }


def _build_unavailable_health_metric(reason: str) -> dict:
    return {
        "status": "unknown",
        "alert": False,
        "reason": reason,
    }


def _build_deferred_session_learn_queue_health(reason: str) -> dict:
    return {
        **_build_unavailable_health_metric(reason),
        "stale_threshold_seconds": int(_cfg.WRITE_STALE_MINUTES * 60),
        "processing_count": None,
        "stale_processing_count": None,
        "recoverable_stale_count": None,
        "unrecoverable_stale_count": None,
        "oldest_processing_age_seconds": None,
        "oldest_stale_age_seconds": None,
        "oldest_stale_request_id": None,
        "max_attempts_exhausted_count": None,
    }


def _build_lifecycle_jobs_health() -> dict:
    try:
        from datetime import timedelta

        from app.core.profile import get_active_profile
        from app.storage.lifecycle_jobs import summarize_lifecycle_jobs, summarize_lifecycle_jobs_by_source

        stale_cutoff = (_utc_now() - timedelta(seconds=_cfg.LIFECYCLE_JOB_LEASE_SECONDS)).isoformat()
        summary = summarize_lifecycle_jobs(
            profile=get_active_profile(),
            stale_before_iso=stale_cutoff,
        )
        session_learn_summary = summarize_lifecycle_jobs_by_source(
            profile=get_active_profile(),
            source="session_learn",
            stale_before_iso=stale_cutoff,
        )
        queued_count = int(summary["queued_count"]) + int(summary["retry_count"])
        stale_running_count = int(summary["stale_running_count"])
        oldest_queued_age = _iso_age_seconds(summary.get("oldest_queued_updated_at"))
        oldest_running_age = _iso_age_seconds(summary.get("oldest_running_updated_at"))
        if queued_count > 50 or (oldest_queued_age or 0) > 600 or stale_running_count > 0:
            status = "critical"
        elif queued_count > 10 or (oldest_queued_age or 0) > 120:
            status = "warning"
        else:
            status = "ok"
        return {
            "status": status,
            "alert": status in {"warning", "critical"},
            "enabled": bool(_cfg.LIFECYCLE_JOBS_ENABLED),
            "queued_count": queued_count,
            "running_count": int(summary["running_count"]),
            "retry_count": int(summary["retry_count"]),
            "failed_count": int(summary["failed_count"]),
            "committed_count": int(summary["committed_count"]),
            "skipped_count": int(summary["skipped_count"]),
            "stale_running_count": stale_running_count,
            "oldest_queued_age_seconds": oldest_queued_age,
            "oldest_running_age_seconds": oldest_running_age,
            "warning_queued_count": 10,
            "critical_queued_count": 50,
            "warning_oldest_queued_age_seconds": 120,
            "critical_oldest_queued_age_seconds": 600,
            "by_source": {
                "session_learn": {
                    "enabled": bool(_SESSION_LEARN_LIFECYCLE_JOBS_ENABLED),
                    "queued_count": int(session_learn_summary["queued_count"]),
                    "running_count": int(session_learn_summary["running_count"]),
                    "retry_count": int(session_learn_summary["retry_count"]),
                    "failed_count": int(session_learn_summary["failed_count"]),
                    "committed_count": int(session_learn_summary["committed_count"]),
                    "skipped_count": int(session_learn_summary["skipped_count"]),
                    "stale_running_count": int(session_learn_summary["stale_running_count"]),
                    "oldest_queued_age_seconds": _iso_age_seconds(
                        session_learn_summary.get("oldest_queued_updated_at")
                    ),
                    "oldest_running_age_seconds": _iso_age_seconds(
                        session_learn_summary.get("oldest_running_updated_at")
                    ),
                }
            },
        }
    except Exception as exc:
        return {
            "status": "unknown",
            "alert": True,
            "reason": "lifecycle_jobs_health_failed",
            "error": _safe_error(exc),
        }


def _build_required_context_cache_health() -> dict[str, Any]:
    """Summarize required-context cache readiness for bounded turn latency."""
    try:
        from app.session.required_context_cache import required_context_cache_status

        max_age_ms = float(os.environ.get("PITH_REQUIRED_CONTEXT_CACHE_HEALTH_MAX_AGE_MS", "60000"))
        status = required_context_cache_status(health_max_age_ms=max_age_ms)
        component_status = {
            "empty": "degraded",
            "fresh": "ok",
            "stale_but_servable": "warning",
            "cold_degraded": "degraded",
        }.get(status.state, "unknown")
        alert = status.state in {"empty", "cold_degraded"}
        return {
            "status": component_status,
            "alert": alert,
            "warm": status.state == "fresh",
            "state": status.state,
            "age_ms": status.age_ms,
            "servable": status.servable,
            "max_age_ms": max_age_ms,
            "health_max_age_ms": status.health_max_age_ms,
            "serving_max_stale_ms": status.serving_max_stale_ms,
            "refresh_after_ms": status.refresh_after_ms,
            "refresh_in_flight": status.refresh_in_flight,
            "stale_first_enabled": os.environ.get(
                "PITH_STAGE3B_REQUIRED_CONTEXT_STALE_FIRST", "true"
            ).lower() in ("true", "1"),
        }
    except Exception as exc:
        return {
            "status": "unknown",
            "alert": True,
            "reason": "required_context_cache_health_failed",
            "error": _safe_error(exc),
        }


def _build_conversation_turn_latency_health() -> dict[str, Any]:
    """Summarize recent conversation_turn latency from metrics."""
    try:
        from app.ops.metrics import metrics

        aggregate = metrics.query_aggregate("conversation_turn_latency_ms")
        p95_ms = float(aggregate.get("p95", 0.0) or 0.0)
        max_ms = float(aggregate.get("max", 0.0) or 0.0)
        if max_ms > 15000.0:
            status = "critical"
        elif p95_ms > 3500.0:
            status = "degraded"
        else:
            status = "ok"
        return {
            "status": status,
            "alert": status in {"critical", "degraded"},
            "p95_ms": p95_ms,
            "max_ms": max_ms,
            "count": int(aggregate.get("count", 0) or 0),
            "threshold_p95_ms": 3500.0,
            "threshold_critical_max_ms": 15000.0,
        }
    except Exception as exc:
        return {
            "status": "unknown",
            "alert": True,
            "reason": "conversation_turn_latency_health_failed",
            "error": _safe_error(exc),
        }


def _should_defer_health_db_metrics(ready: dict) -> bool:
    return ready.get("process_state", "running") != "running" or ready.get("retrieval_state") != "ready"


def _health_metrics_defer_reason(ready: dict) -> str:
    if ready.get("process_state", "running") != "running":
        return "process_not_running"
    if ready.get("retrieval_state") == "recovering":
        return "startup_recovering"
    return "retrieval_not_ready"


@dataclass(frozen=True)
class WriteReplayRecoverySpec:
    endpoint: str
    run: Callable[[dict], object]
    commit: Callable[[str, object], dict]


def _record_write_replay_metric(metric: str, endpoint: str, labels: dict | None = None) -> None:
    try:
        from app.ops.metrics import metrics as _wr_metrics

        metric_labels = {"endpoint": endpoint}
        if labels:
            metric_labels.update(labels)
        _wr_metrics.record(metric, 1.0, metric_labels)
        _wr_metrics.flush()
    except Exception:
        pass


def _run_session_learn_replay_payload(payload: dict) -> object:
    request = SessionLearnRequest(**payload)
    return _with_db_retry(lambda: session_manager.session_learn(request))


def _load_committed_session_learn_replay(request_id: str) -> dict | None:
    from app.core.profile import get_active_profile

    row = load_write_request_replay("session_learn", get_active_profile(), request_id)
    if not row or row.get("status") != "committed" or not row.get("response"):
        return None
    payload = dict(row["response"])
    payload.setdefault("request_id", request_id)
    payload.setdefault("persistence_state", "committed")
    return payload


def _run_session_learn_lifecycle_job(job: dict) -> dict:
    request_id = job.get("idempotency_key") or job.get("request_id")
    if not request_id:
        raise ValueError("session_learn lifecycle job missing idempotency_key")
    committed = _load_committed_session_learn_replay(str(request_id))
    if committed is not None:
        _record_session_learn_contract_metric("session_learn_lifecycle_replay_already_committed")
        return {"status": "replay_already_committed", "request_id": request_id}

    payload = job.get("payload") or {}
    request_payload = payload.get("learn_request") if isinstance(payload, dict) else None
    if not isinstance(request_payload, dict):
        raise ValueError("session_learn lifecycle job missing learn_request payload")
    request_payload.setdefault("request_id", request_id)
    request = SessionLearnRequest(**request_payload)
    result = _run_session_learn_replay_payload(request.model_dump(mode="json"))
    _commit_session_learn_result(str(request_id), result)
    _record_session_learn_contract_metric("session_learn_lifecycle_committed")
    return {"status": "committed", "request_id": request_id}


def _run_session_end_replay_payload(payload: dict) -> object:
    request_id = _ensure_session_end_request_id(payload.get("request_id"))
    request_payload = dict(payload)
    request_payload["request_id"] = request_id
    request = SessionEndRequest(**request_payload)
    return _with_db_retry(lambda: session_manager.end_session(request))


def _run_checkpoint_replay_payload(payload: dict) -> dict:
    return _with_db_retry(lambda: _execute_checkpoint_action(dict(payload)))


def _commit_checkpoint_result(request_id: str, result) -> dict:
    payload = dict(result) if isinstance(result, dict) else {"status": str(result)}
    payload = commit_write_request("checkpoint", request_id, payload)
    _record_successful_write()
    return payload


def _write_replay_recovery_specs() -> dict[str, WriteReplayRecoverySpec]:
    return {
        "session_learn": WriteReplayRecoverySpec("session_learn", _run_session_learn_replay_payload, _commit_session_learn_result),
        "session_end": WriteReplayRecoverySpec("session_end", _run_session_end_replay_payload, _commit_session_end_result),
        "checkpoint": WriteReplayRecoverySpec("checkpoint", _run_checkpoint_replay_payload, _commit_checkpoint_result),
    }


def _clear_write_replay_reclaimer_future(endpoint: str, done_future: concurrent.futures.Future) -> None:
    with _WRITE_REPLAY_RECLAIMER_LOCK:
        if _WRITE_REPLAY_RECLAIMER_FUTURES.get(endpoint) is done_future:
            _WRITE_REPLAY_RECLAIMER_FUTURES.pop(endpoint, None)


def _schedule_write_replay_reclaimer(endpoint: str, reason: str) -> bool:
    if endpoint not in _write_replay_recovery_specs():
        return False
    with _WRITE_REPLAY_RECLAIMER_LOCK:
        existing = _WRITE_REPLAY_RECLAIMER_FUTURES.get(endpoint)
        if existing is not None and not existing.done():
            return False
        future = _get_session_learn_executor().submit(_run_write_replay_reclaimer, endpoint, reason)
        _WRITE_REPLAY_RECLAIMER_FUTURES[endpoint] = future
    future.add_done_callback(
        lambda done_future, endpoint=endpoint: _clear_write_replay_reclaimer_future(endpoint, done_future)
    )
    return True


def _schedule_all_write_replay_reclaimers(reason: str) -> dict[str, bool]:
    return {
        endpoint: _schedule_write_replay_reclaimer(endpoint, reason)
        for endpoint in _write_replay_recovery_specs()
    }


def _schedule_session_learn_reclaimer(reason: str) -> bool:
    return _schedule_write_replay_reclaimer("session_learn", reason)


def _run_session_learn_reclaimer(reason: str, batch_size: int | None = None) -> dict:
    return _run_write_replay_reclaimer("session_learn", reason, batch_size=batch_size)


def _run_write_replay_reclaimer(endpoint: str, reason: str, batch_size: int | None = None) -> dict:
    from datetime import timedelta

    from app.core.profile import get_active_profile
    from app.storage import (
        claim_stale_write_requests,
        fail_unrecoverable_stale_write_requests,
        mark_write_request_reclaim_failed,
    )

    specs = _write_replay_recovery_specs()
    if endpoint not in specs:
        return {"endpoint": endpoint, "reason": reason, "claimed": 0, "committed": 0, "failed": 0, "retry_scheduled": 0, "unrecoverable_failed": 0}
    spec = specs[endpoint]
    profile = get_active_profile()
    now_dt = _utc_now()
    now = now_dt.isoformat()
    stale_cutoff = _session_learn_stale_cutoff_iso()
    lease_expires_at = (now_dt + timedelta(seconds=_SESSION_LEARN_RECLAIMER_LEASE_SECONDS)).isoformat()
    batch_limit = batch_size or _SESSION_LEARN_RECLAIMER_BATCH_SIZE
    rows = claim_stale_write_requests(
        endpoint,
        profile,
        stale_cutoff,
        f"{os.getpid()}:{id(app)}:{reason}:{endpoint}",
        lease_expires_at,
        batch_limit,
        now,
        _SESSION_LEARN_RECLAIMER_MAX_ATTEMPTS,
    )
    result = {
        "endpoint": endpoint,
        "reason": reason,
        "claimed": len(rows),
        "committed": 0,
        "failed": 0,
        "retry_scheduled": 0,
        "unrecoverable_failed": 0,
        "skipped_active_lifecycle": 0,
    }
    if rows:
        _record_write_replay_metric("write_request_reclaimer_claimed", endpoint)
        if endpoint == "session_learn":
            _record_session_learn_contract_metric("session_learn_reclaimer_claimed")
    for row in rows:
        request_id = row["request_id"]
        if endpoint == "session_learn" and _SESSION_LEARN_LIFECYCLE_JOBS_ENABLED:
            from app.storage.lifecycle_jobs import load_lifecycle_job_by_identity

            lifecycle_job = load_lifecycle_job_by_identity(
                profile=profile,
                source="session_learn",
                idempotency_key=request_id,
            )
            if lifecycle_job and lifecycle_job.get("status") in {"queued", "running", "retry"}:
                _record_session_learn_contract_metric("session_learn_reclaimer_skipped_active_lifecycle")
                result["skipped_active_lifecycle"] += 1
                continue
        try:
            request_payload = dict(row["request"])
            request_payload.setdefault("request_id", request_id)
            replay_result = spec.run(request_payload)
            spec.commit(request_id, replay_result)
            _record_write_replay_metric("write_request_reclaimer_committed", endpoint)
            if endpoint == "session_learn":
                _record_session_learn_contract_metric("session_learn_reclaimer_committed")
            result["committed"] += 1
        except (ValidationError, ValueError) as exc:
            fail_write_request(
                endpoint,
                request_id,
                {
                    "status": "failed",
                    "persistence_state": "failed",
                    "request_id": request_id,
                    "error_class": type(exc).__name__,
                    "error": _safe_error(exc),
                },
                error_class=type(exc).__name__,
            )
            _record_write_replay_metric("write_request_reclaimer_failed", endpoint, {"error_class": type(exc).__name__})
            result["failed"] += 1
        except Exception as exc:
            retry_at = (_utc_now() + timedelta(seconds=_SESSION_LEARN_RECLAIMER_RETRY_SECONDS)).isoformat()
            mark_write_request_reclaim_failed(
                endpoint,
                profile,
                request_id,
                f"{type(exc).__name__}: {_safe_error(exc)}",
                retry_at,
                _utc_now_iso(),
            )
            _record_write_replay_metric("write_request_reclaimer_retry_scheduled", endpoint, {"error_class": type(exc).__name__})
            if endpoint == "session_learn":
                _record_session_learn_contract_metric("session_learn_reclaimer_failed")
            logger.warning("%s reclaimer failed for request_id=%s: %s", endpoint, request_id, exc, exc_info=True)
            result["retry_scheduled"] += 1
    unrecoverable_failed = fail_unrecoverable_stale_write_requests(
        endpoint,
        profile,
        stale_cutoff,
        _utc_now_iso(),
        "UnrecoverableReplayMissingPayload",
        batch_limit,
    )
    if unrecoverable_failed:
        _record_write_replay_metric("write_request_unrecoverable_failed", endpoint)
        result["failed"] += unrecoverable_failed
        result["unrecoverable_failed"] = unrecoverable_failed
    return result


def _commit_session_learn_result(request_id: str, result) -> dict:
    payload = result.model_dump()
    if not payload.get("request_id"):
        payload["request_id"] = request_id
    if not payload.get("processing_state"):
        payload["processing_state"] = "committed"
    payload = commit_write_request("session_learn", request_id, payload)
    _record_successful_write()
    return payload


def _coerce_session_end_payload(request_id: str, result) -> dict:
    if hasattr(result, "model_dump"):
        payload = result.model_dump()
    elif isinstance(result, dict):
        payload = dict(result)
    else:
        payload = {"status": str(result)}
    payload.setdefault("request_id", request_id)
    payload.setdefault("processing_state", "committed")
    payload.setdefault("final_learning_state", "committed")
    return payload


def _commit_session_end_result(request_id: str, result) -> dict:
    payload = _coerce_session_end_payload(request_id, result)
    payload = commit_write_request("session_end", request_id, payload)
    _record_successful_write()
    return payload


def _finalize_deferred_session_learn(future: concurrent.futures.Future, request_id: str) -> None:
    try:
        result = future.result()
        _commit_session_learn_result(request_id, result)
        _record_session_learn_contract_metric("session_learn_background_committed")
    except Exception as exc:
        abandon_write_request("session_learn", request_id, error_class=type(exc).__name__)
        _record_session_learn_contract_metric("session_learn_background_failed")
        logger.error("session_learn background error for request_id=%s: %s", request_id, exc, exc_info=True)


def _finalize_deferred_session_end(future: concurrent.futures.Future, request_id: str) -> None:
    try:
        result = future.result()
        _commit_session_end_result(request_id, result)
        _record_session_learn_contract_metric("session_end_background_committed")
    except Exception as exc:
        abandon_write_request("session_end", request_id, error_class=type(exc).__name__)
        _record_session_learn_contract_metric("session_end_background_failed")
        logger.error("session_end background error for request_id=%s: %s", request_id, exc, exc_info=True)


@app.post("/session_learn", dependencies=[Depends(verify_api_key)])
def session_learn_endpoint(request: SessionLearnRequest):
    """Post-response concept extraction. Given a completed exchange, extract
    new knowledge, evolve existing concepts, build associations.
    Fast requests return synchronously; slow requests return replayable processing state."""
    start = time.perf_counter()
    request, request_id = _ensure_session_learn_request_id(request)
    try:
        replay_state = begin_write_request(
            "session_learn",
            request_id,
            request_payload=request.model_dump(mode="json"),
        )
    except HTTPException as exc:
        if exc.status_code == 409:
            _record_session_learn_initial_response_latency(start, "processing_duplicate")
            return _session_learn_processing_payload(
                request_id,
                processing_time_ms=(time.perf_counter() - start) * 1000,
            )
        _record_session_learn_initial_response_latency(start, "error")
        raise
    if replay_state.replay is not None:
        _record_session_learn_initial_response_latency(start, "replay")
        return replay_state.replay

    if _SESSION_LEARN_LIFECYCLE_JOBS_ENABLED:
        try:
            _enqueue_session_learn_lifecycle_job(request, request_id)
        except Exception as e:
            abandon_write_request("session_learn", request_id, error_class=type(e).__name__)
            _record_session_learn_initial_response_latency(start, "error")
            logger.error("session_learn lifecycle enqueue error: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail=_safe_error(e))
        _record_session_learn_contract_metric("session_learn_lifecycle_deferred")
        _record_session_learn_lifecycle_latency("session_learn_lifecycle_ack_latency_ms", start)
        _record_session_learn_initial_response_latency(start, "lifecycle_processing")
        return _session_learn_processing_payload(
            request_id,
            processing_time_ms=(time.perf_counter() - start) * 1000,
        )

    def _do_session_learn():
        binding = None
        prepare_binding = getattr(session_manager, "prepare_session_learn_binding", None)
        if callable(prepare_binding):
            binding = prepare_binding(request)
        try:
            return _with_db_retry(lambda: session_manager.session_learn(request))
        finally:
            if binding is not None:
                session_manager._pop_request_session(
                    binding["session_token"],
                    binding["active_token"],
                )

    future = _get_session_learn_executor().submit(_do_session_learn)
    try:
        result = future.result(timeout=_remaining_session_learn_sync_wait(start))
        _record_session_learn_initial_response_latency(start, "committed")
        return _commit_session_learn_result(request_id, result)
    except concurrent.futures.TimeoutError:
        future.add_done_callback(lambda done_future: _finalize_deferred_session_learn(done_future, request_id))
        _record_session_learn_contract_metric("session_learn_deferred")
        _schedule_session_learn_reclaimer("deferred_session_learn")
        _record_session_learn_initial_response_latency(start, "processing_deferred")
        return _session_learn_processing_payload(
            request_id,
            processing_time_ms=(time.perf_counter() - start) * 1000,
        )
    except Exception as e:
        abandon_write_request("session_learn", request_id, error_class=type(e).__name__)
        _record_session_learn_initial_response_latency(start, "error")
        logger.error(f"session_learn error: {e}")
        raise HTTPException(status_code=500, detail=_safe_error(e))


@app.get("/sessions_list")
def sessions_list_endpoint(status: str = None, limit: int = 20, since: str = None):
    """Query session history. Optional filters: status, limit, since (ISO datetime)."""
    return list_sessions(status=status, limit=limit, since=since)


def _record_checkpoint_save_telemetry(body: dict) -> None:
    try:
        from app.storage import record_checkpoint_save_event

        record_checkpoint_save_event(body.get("session_id"), body["task_id"])
    except Exception as _ckpt_metric_err:
        logger.warning(
            "CKPT-008: checkpoint_save telemetry skipped under DB contention: %s",
            _safe_error(_ckpt_metric_err),
        )


def _execute_checkpoint_action(body: dict) -> dict:
    from app.storage import complete_checkpoint, save_checkpoint, touch_checkpoint

    action = body.get("action", "save")
    if action == "save":
        if not body.get("task_id") or not body.get("description"):
            raise ValueError("task_id and description required for save")

        def _do_save_checkpoint():
            return save_checkpoint(
                task_id=body["task_id"],
                description=body["description"],
                status=body.get("status", "active"),
                done=body.get("done", []),
                active=body.get("active", ""),
                next_items=body.get("next", []),
                blockers=body.get("blockers", []),
                context=body.get("context", {}),
                concept_refs=body.get("concept_refs", []),
                session_id=body.get("session_id"),
                ttl_days=body.get("ttl_days"),
                origin_id=body.get("origin_id"),
                op_id=body.get("op_id"),
                payload_hash=body.get("payload_hash"),
            )

        result = _do_save_checkpoint()
        _record_checkpoint_save_telemetry(body)
        return result
    if action == "complete":
        if not body.get("task_id"):
            raise ValueError("task_id required for complete")
        return complete_checkpoint(body["task_id"]) or {"status": "not_found"}
    if action == "touch":
        if not body.get("task_id"):
            raise ValueError("task_id required for touch")
        return touch_checkpoint(body["task_id"], ttl_days=body.get("ttl_days", 7)) or {"status": "not_found"}
    raise ValueError(f"Unsupported checkpoint replay action: {action}")


@app.post("/checkpoint", dependencies=[Depends(verify_api_key)])
def checkpoint_endpoint(body: dict):
    """Execution checkpoint CRUD — ephemeral resumption state, NOT concepts."""
    from app.storage import (
        list_checkpoints,
        load_checkpoint,
    )

    action = body.get("action", "save")
    request_id = body.get("request_id")
    mutating_actions = {"save", "touch", "complete"}
    replay_state = (
        begin_write_request("checkpoint", request_id, request_payload=dict(body))
        if action in mutating_actions
        else None
    )
    if replay_state and replay_state.replay is not None:
        return replay_state.replay

    if action in mutating_actions:
        try:
            result = _with_db_retry(lambda: _execute_checkpoint_action(body))
        except ValueError as exc:
            abandon_write_request("checkpoint", request_id, error_class="ValueError")
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            abandon_write_request("checkpoint", request_id, error_class=type(exc).__name__)
            logger.error("checkpoint %s error: %s", action, exc)
            raise HTTPException(503, f"Transient error during checkpoint {action}: {_safe_error(exc)}") from exc
        result = _commit_checkpoint_result(request_id, result)
        return result
    if action == "load":
        result = load_checkpoint(
            task_id=body.get("task_id"),
            max_age_hours=body.get("max_age_hours") or 24,
            session_id=body.get("session_id"),
            origin_id=body.get("origin_id"),
        )
        return result or {"status": "no_checkpoint_found"}
    elif action == "list":
        return {"checkpoints": list_checkpoints()}
    elif action == "dashboard":
        from app.storage import get_checkpoint_dashboard

        return get_checkpoint_dashboard()
    elif action == "threshold_analysis":
        from app.storage import analyze_coverage_threshold

        thresholds = body.get("thresholds")
        return analyze_coverage_threshold(thresholds)
    elif action == "session_drops":
        from app.storage import analyze_session_drops

        return analyze_session_drops()
    else:
        raise HTTPException(400, f"Unknown action: {action}")


# =============================================================================
# Wave 5: pith_threads MCP tool (§5.7)
# =============================================================================


def _parse_workstream_bool(value: object, default: bool = False) -> bool:
    """Parse Workstreams booleans without Python's bool('false') trap."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "1", "yes", "on"):
            return True
        if normalized in ("false", "0", "no", "off"):
            return False
        raise HTTPException(400, f"Invalid boolean value: {value}")
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise HTTPException(400, f"Invalid boolean value: {value}")


_WORKSTREAM_ACTIVATION_CREATE_METADATA_FIELDS = (
    "title",
    "description",
    "urgency",
    "goal_ids",
    "knowledge_areas",
    "agent_id",
    "current_objective",
    "current_summary",
    "next_action",
    "blockers",
    "quality_state",
    "created_by",
)


def _workstream_activation_metadata_from_body(body: dict) -> object:
    metadata = body.get("metadata")
    if metadata is not None:
        return metadata
    return {key: body[key] for key in _WORKSTREAM_ACTIVATION_CREATE_METADATA_FIELDS if key in body}


@app.post("/pith_threads", dependencies=[Depends(verify_api_key)])
def pith_threads_endpoint(body: dict):
    """Manage narrative threads — ongoing work streams and topics."""
    from app.features.threads import (
        apply_workstream_hygiene,
        bind_workstream_checkpoint,
        build_workstream_context_block,
        classify_workstream_threads,
        clear_workstream_binding,
        create_thread,
        create_workstream,
        demote_workstream_discovery_candidate,
        dry_run_workstream_hygiene,
        ensure_workstream_activation,
        get_concepts_for_thread,
        get_thread_stats,
        link_concept_to_thread,
        load_thread,
        load_threads,
        promote_thread_to_workstream,
        promote_workstream_discovery_candidate,
        resolve_active_workstream,
        retrieve_similar_traces,
        rollback_workstream_hygiene,
        save_thread,
        summarize_thread_for_list,
        unlink_concept_from_thread,
        update_thread_status,
        update_workstream_metadata,
    )

    action = body.get("action", "list")

    from app.core.config import get_feature_flag

    if action in (
        "classify_workstreams",
        "workstream_context",
        "active_workstream",
        "workstream_hygiene_dry_run",
    ) and not get_feature_flag("WORKSTREAMS_READ_ENABLED", True):
        return {"status": "disabled", "reason": "WORKSTREAMS_READ_ENABLED is false"}
    if action == "ensure_workstream_activation":
        activation_mode = str(body.get("mode", "candidate") or "candidate").strip().lower()
        if activation_mode == "candidate" and not get_feature_flag("WORKSTREAMS_READ_ENABLED", True):
            return {"status": "disabled", "reason": "WORKSTREAMS_READ_ENABLED is false"}
        if activation_mode != "candidate":
            broad_write_enabled = get_feature_flag("WORKSTREAMS_WRITE_ENABLED", False)
            activation_write_enabled = get_feature_flag("WORKSTREAMS_ACTIVATION_WRITE_ENABLED", False)
            if not (broad_write_enabled or activation_write_enabled):
                return {"status": "disabled", "reason": "WORKSTREAMS_ACTIVATION_WRITE_ENABLED is false"}
    if action in (
        "create_workstream",
        "promote_workstream",
        "update_workstream",
        "bind_workstream",
        "clear_workstream_binding",
        "workstream_hygiene_apply",
        "workstream_hygiene_rollback",
        "workstream_promote_discovery_candidate",
        "workstream_demote_discovery_candidate",
    ) and not get_feature_flag("WORKSTREAMS_WRITE_ENABLED", False):
        return {"status": "disabled", "reason": "WORKSTREAMS_WRITE_ENABLED is false"}

    if action == "create":
        if not body.get("title"):
            raise HTTPException(400, "title required for create")
        thread = create_thread(
            title=body["title"],
            description=body.get("description", ""),
            urgency=body.get("urgency", "normal"),
            goal_ids=body.get("goal_ids"),
            knowledge_areas=body.get("knowledge_areas"),
        )
        return {"status": "ok", "thread": thread.model_dump()}

    elif action == "get":
        if not body.get("thread_id"):
            raise HTTPException(400, "thread_id required for get")
        thread = load_thread(body["thread_id"])
        if not thread:
            return {"status": "not_found"}
        links = get_concepts_for_thread(body["thread_id"])
        return {
            "status": "ok",
            "thread": thread.model_dump(),
            "concept_links": [l.model_dump() for l in links],
        }

    elif action == "list":
        from app.core.config import THREAD_LIST_LIMIT_DEFAULT, THREAD_LIST_LIMIT_MAX

        status_filter = body.get("status")
        try:
            requested_limit = int(body.get("limit", THREAD_LIST_LIMIT_DEFAULT))
        except (TypeError, ValueError):
            requested_limit = THREAD_LIST_LIMIT_DEFAULT
        effective_limit = max(1, min(requested_limit, THREAD_LIST_LIMIT_MAX))
        threads = load_threads(status=status_filter, limit=effective_limit + 1)
        truncated = len(threads) > effective_limit
        visible_threads = threads[:effective_limit]
        return {
            "status": "ok",
            "threads": [summarize_thread_for_list(t) for t in visible_threads],
            "count": len(visible_threads),
            "limit": effective_limit,
            "truncated": truncated,
        }

    elif action == "update":
        if not body.get("thread_id"):
            raise HTTPException(400, "thread_id required for update")
        thread = load_thread(body["thread_id"])
        if not thread:
            return {"status": "not_found"}
        if body.get("title"):
            thread.title = body["title"][:500]
        if body.get("description") is not None:
            thread.description = body["description"][:500]
        if body.get("urgency"):
            thread.urgency = body["urgency"]
        if body.get("goal_ids") is not None:
            thread.goal_ids = body["goal_ids"]
        if body.get("knowledge_areas") is not None:
            thread.knowledge_areas = body["knowledge_areas"]
        thread.updated_at = _utc_now_iso()
        save_thread(thread)
        return {"status": "ok", "thread": thread.model_dump()}

    elif action == "close":
        if not body.get("thread_id"):
            raise HTTPException(400, "thread_id required for close")
        new_status = body.get("status", "completed")
        if new_status not in ("completed", "abandoned"):
            new_status = "completed"
        thread = update_thread_status(body["thread_id"], new_status, reason="user")
        return {"status": "ok", "thread": thread.model_dump()}

    elif action == "reactivate":
        if not body.get("thread_id"):
            raise HTTPException(400, "thread_id required for reactivate")
        thread = update_thread_status(body["thread_id"], "active", reason="user_reactivated")
        return {"status": "ok", "thread": thread.model_dump()}

    elif action == "link":
        if not body.get("thread_id") or not body.get("concept_id"):
            raise HTTPException(400, "thread_id and concept_id required for link")
        link = link_concept_to_thread(
            body["thread_id"],
            body["concept_id"],
            role=body.get("role", "member"),
            added_by=body.get("added_by", "user"),
        )
        return {"status": "ok", "link": link.model_dump() if link else None}

    elif action == "unlink":
        if not body.get("thread_id") or not body.get("concept_id"):
            raise HTTPException(400, "thread_id and concept_id required for unlink")
        unlink_concept_from_thread(body["thread_id"], body["concept_id"])
        return {"status": "ok"}

    elif action == "similar":
        if not body.get("situation"):
            raise HTTPException(400, "situation required for similar")
        results = retrieve_similar_traces(
            situation=body["situation"],
            intent=body.get("intent"),
            limit=body.get("limit", 5),
        )
        return {"status": "ok", "traces": results, "count": len(results)}

    elif action == "stats":
        return {"status": "ok", **get_thread_stats()}

    elif action == "classify_workstreams":
        return classify_workstream_threads(
            agent_id=body.get("agent_id", "default"),
            include_maintenance=_parse_workstream_bool(body.get("include_maintenance", True), default=True),
            limit=body.get("limit", 100),
            include_non_workstreams=_parse_workstream_bool(body.get("include_non_workstreams", False), default=False),
        )

    elif action == "workstream_hygiene_dry_run":
        return dry_run_workstream_hygiene(
            agent_id=body.get("agent_id", "default"),
            limit=body.get("limit", 100),
        )

    elif action == "workstream_hygiene_apply":
        return apply_workstream_hygiene(
            run_id=body.get("run_id"),
            evaluated_at=body.get("evaluated_at"),
            fingerprints=body.get("fingerprints") or {},
            proposed_states=body.get("proposed_states") or {},
            operator_confirmed=body.get("operator_confirmed", False),
        )

    elif action == "workstream_hygiene_rollback":
        return rollback_workstream_hygiene(
            run_id=body.get("run_id"),
            operator_confirmed=body.get("operator_confirmed", False),
        )

    elif action == "workstream_promote_discovery_candidate":
        return promote_workstream_discovery_candidate(
            thread_id=body.get("thread_id"),
            promotion_reason=body.get("promotion_reason", ""),
            promoted_by=body.get("promoted_by", "operator"),
            operator_confirmed=body.get("operator_confirmed", False),
        )

    elif action == "workstream_demote_discovery_candidate":
        return demote_workstream_discovery_candidate(
            thread_id=body.get("thread_id"),
            reason=body.get("reason", ""),
            demoted_by=body.get("demoted_by", "operator"),
            operator_confirmed=body.get("operator_confirmed", False),
        )

    elif action == "workstream_context":
        return build_workstream_context_block(
            thread_id=body.get("thread_id"),
            operator_mode=_parse_workstream_bool(body.get("operator_mode", False), default=False),
            max_refs=body.get("max_refs", 10),
            include_concept_summaries=_parse_workstream_bool(
                body.get("include_concept_summaries", True),
                default=True,
            ),
        )

    elif action == "create_workstream":
        if not body.get("title"):
            raise HTTPException(400, "title required for create_workstream")
        try:
            thread = create_workstream(
                title=body["title"],
                description=body.get("description", ""),
                urgency=body.get("urgency", "normal"),
                goal_ids=body.get("goal_ids"),
                knowledge_areas=body.get("knowledge_areas"),
                agent_id=body.get("agent_id", "default"),
                current_objective=body.get("current_objective", ""),
                current_summary=body.get("current_summary", ""),
                next_action=body.get("next_action", ""),
                blockers=body.get("blockers"),
                quality_state=body.get("quality_state", "ok"),
                created_by=body.get("created_by", "user"),
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"status": "ok", "thread": thread.model_dump()}

    elif action == "promote_workstream":
        if not body.get("thread_id"):
            raise HTTPException(400, "thread_id required for promote_workstream")
        metadata = {
            key: body[key]
            for key in (
                "current_objective",
                "current_summary",
                "next_action",
                "blockers",
                "quality_state",
                "created_by",
                "updated_by",
            )
            if key in body
        }
        try:
            thread = promote_thread_to_workstream(
                body["thread_id"],
                metadata=metadata,
                operator_mode=_parse_workstream_bool(body.get("operator_mode", False), default=False),
            )
        except ValueError as exc:
            if str(exc) == "thread_not_found":
                return {"status": "not_found"}
            if str(exc) == "maintenance_cluster_not_promotable":
                raise HTTPException(409, str(exc)) from exc
            raise HTTPException(400, str(exc)) from exc
        return {"status": "ok", "thread": thread.model_dump()}

    elif action == "update_workstream":
        if not body.get("thread_id"):
            raise HTTPException(400, "thread_id required for update_workstream")
        try:
            thread = update_workstream_metadata(body["thread_id"], body)
        except ValueError as exc:
            if str(exc) == "thread_not_found":
                return {"status": "not_found"}
            if str(exc) == "not_workstream":
                raise HTTPException(409, str(exc)) from exc
            raise HTTPException(400, str(exc)) from exc
        return {"status": "ok", "thread": thread.model_dump()}

    elif action == "bind_workstream":
        if not body.get("thread_id"):
            raise HTTPException(400, "thread_id required for bind_workstream")
        try:
            return bind_workstream_checkpoint(
                thread_id=body["thread_id"],
                origin_id=body.get("origin_id"),
                session_id=body.get("session_id"),
                current_task_id=body.get("current_task_id"),
                op_id=body.get("op_id"),
                payload_hash=body.get("payload_hash"),
            )
        except ValueError as exc:
            if str(exc) == "thread_not_found":
                return {"status": "not_found"}
            if str(exc) == "not_workstream":
                raise HTTPException(409, str(exc)) from exc
            raise HTTPException(400, str(exc)) from exc

    elif action == "clear_workstream_binding":
        try:
            return clear_workstream_binding(
                thread_id=body.get("thread_id"),
                origin_id=body.get("origin_id"),
                session_id=body.get("session_id"),
                current_task_id=body.get("current_task_id"),
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    elif action == "active_workstream":
        try:
            return resolve_active_workstream(
                thread_id=body.get("thread_id"),
                origin_id=body.get("origin_id"),
                session_id=body.get("session_id"),
                current_task_id=body.get("current_task_id"),
                operator_mode=_parse_workstream_bool(body.get("operator_mode", False), default=False),
                max_refs=body.get("max_refs", 10),
                include_concept_summaries=_parse_workstream_bool(
                    body.get("include_concept_summaries", True),
                    default=True,
                ),
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    elif action == "ensure_workstream_activation":
        result = ensure_workstream_activation(
            mode=body.get("mode", "candidate"),
            origin_id=body.get("origin_id"),
            session_id=body.get("session_id"),
            current_task_id=body.get("current_task_id"),
            situation=body.get("situation"),
            thread_id=body.get("thread_id"),
            metadata=_workstream_activation_metadata_from_body(body),
            skip_reason=body.get("skip_reason"),
            skip_exception_kind=body.get("skip_exception_kind"),
            operator_confirmed=body.get("operator_confirmed", False),
            include_proof_candidates=_parse_workstream_bool(
                body.get("include_proof_candidates", False),
                default=False,
            ),
            op_id=body.get("op_id"),
            payload_hash=body.get("payload_hash"),
        )
        if result.get("status") == "rejected":
            return result
        return result

    else:
        raise HTTPException(400, f"Unknown action: {action}")


# =============================================================================
# A.10: pith_traces MCP tool (cross-session search)
# =============================================================================


@app.post("/pith_traces", dependencies=[Depends(verify_api_key)])
def pith_traces_endpoint(body: dict):
    """Search and retrieve cognitive traces across sessions."""
    from app.core.config import TRACES_SEARCH_LIMIT_DEFAULT, TRACES_SEARCH_LIMIT_MAX
    from app.features.threads import _extract_terms_simple
    from app.ops.traces import load_trace

    action = body.get("action", "list")
    limit = min(body.get("limit", TRACES_SEARCH_LIMIT_DEFAULT), TRACES_SEARCH_LIMIT_MAX)
    offset = body.get("offset", 0)
    include_data = body.get("include_data", True)

    if action == "get":
        if not body.get("trace_id"):
            raise HTTPException(400, "trace_id required for get")
        trace = load_trace(body["trace_id"])
        if not trace:
            return {"status": "not_found", "message": f"Trace {body['trace_id']} not found"}
        # Enrich with linked concept summaries
        linked_concepts = []
        for cid in trace.concept_refs:
            concept = load_concept(cid, track_access=False)
            if concept:
                linked_concepts.append(
                    {
                        "id": concept.id,
                        "summary": concept.summary[:100],
                        "confidence": concept.confidence,
                        "concept_type": concept.concept_type,
                    }
                )
        return {
            "status": "ok",
            "trace": _format_trace(trace, 0.0, True),
            "linked_concepts": linked_concepts,
        }

    elif action == "list":
        traces = _load_filtered_traces(body, limit, offset)
        total = len(traces)  # Approximate; exact count would need separate query
        return {
            "status": "ok",
            "total": total,
            "returned": len(traces[:limit]),
            "offset": offset,
            "traces": [_format_trace(t, 0.0, include_data) for t in traces[:limit]],
        }

    elif action == "search":
        if not body.get("query"):
            raise HTTPException(400, "query required for search")
        query_terms = _extract_terms_simple(body["query"])
        if not query_terms:
            return {"status": "error", "message": "Query produced no searchable terms"}

        all_traces = _load_filtered_traces(body, limit=500, offset=0)
        if not all_traces:
            return {
                "status": "insufficient_data",
                "feature": "pith_traces search",
                "requirement": "at least 1 trace matching filters",
                "current": "0 traces found",
            }

        scored = []
        for trace in all_traces:
            primary_text = f"{trace.situation} {trace.assessment} {trace.justification}"
            secondary_text = f"{trace.intent} {trace.reflection}"
            primary_terms = set(_extract_terms_simple(primary_text))
            secondary_terms = set(_extract_terms_simple(secondary_text))
            primary_hits = sum(1 for t in query_terms if t in primary_terms)
            secondary_hits = sum(1 for t in query_terms if t in secondary_terms)
            score = (primary_hits + secondary_hits * 0.5) / len(query_terms)
            if score > 0:
                scored.append((trace, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        results = scored[offset : offset + limit]
        return {
            "status": "ok",
            "total_matches": len(scored),
            "returned": len(results),
            "offset": offset,
            "traces": [_format_trace(t, s, include_data) for t, s in results],
        }

    else:
        raise HTTPException(400, f"Unknown action: {action}")


def _format_trace(trace, score: float, include_data: bool) -> dict:
    """Format a TraceRecord for API response."""
    result = {
        "id": trace.id,
        "session_id": trace.session_id,
        "created_at": trace.created_at,
        "trigger_type": trace.trigger_type,
        "relevance_score": round(score, 3),
        "concept_count": len(trace.concept_refs),
    }
    if include_data:
        result.update(
            {
                "situation": trace.situation,
                "intent": trace.intent,
                "assessment": trace.assessment,
                "justification": trace.justification,
                "reflection": trace.reflection,
                "concept_refs": trace.concept_refs,
            }
        )
    return result


def _load_filtered_traces(body: dict, limit: int, offset: int) -> list:
    """Load traces with optional filters from request body."""

    from app.storage import _db

    clauses = []
    params = []
    if body.get("session_id"):
        clauses.append("session_id = ?")
        params.append(body["session_id"])
    if body.get("trigger_type"):
        clauses.append("trigger_type = ?")
        params.append(body["trigger_type"])
    if body.get("concept_id"):
        clauses.append("concept_refs LIKE ?")
        params.append(f"%{body['concept_id']}%")

    where = " AND ".join(clauses) if clauses else "1=1"
    sql = f"SELECT id, session_id, created_at, trigger_type, concept_refs, agent_id, data FROM traces WHERE {where} ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    with _db() as conn:
        rows = conn.execute(sql, params).fetchall()

    from app.ops.traces import row_to_trace

    results = []
    for r in rows:
        trace = row_to_trace(r)
        if trace is not None:
            results.append(trace)
    return results


# =============================================================================
# Wave 6: pith_experiment — Cognitive experiment engine
# =============================================================================


@app.post("/pith_experiment")
async def handle_pith_experiment(request: Request):
    """Generate, process, list, get, or archive cognitive experiments."""
    body = await request.json()
    action = body.get("action", "")

    if action == "generate":
        from app.features.experiments import (
            EXPERIMENT_VALID_TYPES,
            TFIDFCache,
            generate_experiment,
        )
        from app.storage import list_concepts, load_associations

        experiment_type = body.get("experiment_type", "")
        if experiment_type not in EXPERIMENT_VALID_TYPES:
            raise HTTPException(
                400, f"Invalid experiment_type '{experiment_type}'. Valid: {sorted(EXPERIMENT_VALID_TYPES)}"
            )

        # Load concepts (optionally scoped to thread)
        thread_id = body.get("thread_id")
        if thread_id:
            from app.features.threads import get_concepts_for_thread

            concept_ids = get_concepts_for_thread(thread_id)
        else:
            concept_ids = list_concepts()
        concepts = [load_concept(cid, track_access=False) for cid in concept_ids]
        concepts = [c for c in concepts if c is not None]

        # Load associations for counterfactual
        associations = []
        if experiment_type == "counterfactual":
            raw_assocs = load_associations()
            assoc_list = raw_assocs.get("associations", [])
            associations = [
                {"source": a["source"], "target": a["target"], "relation": a["relation"]} for a in assoc_list
            ]

        # Build assoc_counts and salience_ranks for analogy
        assoc_counts = {}
        salience_ranks = {}
        if experiment_type == "analogy_detection":
            # count_associations() returns global count — compute per-concept from loaded associations
            if not associations:
                raw_assocs = load_associations()
                assoc_list = raw_assocs.get("associations", [])
            else:
                assoc_list = associations  # Already loaded above (unlikely path: counterfactual + analogy)
            # Build per-concept association counts from the association list
            from collections import Counter

            source_counts = Counter(a["source"] if isinstance(a, dict) else a.get("source", "") for a in assoc_list)
            target_counts = Counter(a["target"] if isinstance(a, dict) else a.get("target", "") for a in assoc_list)
            for c in concepts:
                assoc_counts[c.id] = source_counts.get(c.id, 0) + target_counts.get(c.id, 0)
                salience_ranks[c.id] = getattr(c, "salience", 0.5)

        tfidf_cache = TFIDFCache()
        experiment = generate_experiment(
            experiment_type=experiment_type,
            concepts=concepts,
            associations=associations,
            assoc_counts=assoc_counts,
            salience_ranks=salience_ranks,
            direction=body.get("direction"),
            max_concept_age_days=body.get("max_concept_age_days"),
            thread_id=thread_id,
            tfidf_cache=tfidf_cache,
        )

        return {
            "status": experiment.status,
            "experiment_id": experiment.id,
            "experiment_type": experiment.experiment_type,
            "candidates_count": len(experiment.candidates),
            "generation_time_ms": experiment.generation_time_ms,
            "metadata": experiment.metadata,
        }

    elif action == "process_results":
        import json as _json

        from app.core.models import ExperimentResult
        from app.features.experiments import process_experiment_results

        experiment_id = body.get("experiment_id")
        if not experiment_id:
            raise HTTPException(400, "experiment_id required")

        result_json = body.get("result_json", "{}")
        if isinstance(result_json, str):
            result_data = _json.loads(result_json)
        else:
            result_data = result_json

        result = ExperimentResult(**result_data)
        outcome = process_experiment_results(experiment_id, result)
        return {"status": "ok", **outcome}

    elif action == "list":
        from app.features.experiments import load_experiments

        include_archived = body.get("include_archived", False)
        experiments = load_experiments(include_archived=include_archived, limit=50)
        return {
            "status": "ok",
            "count": len(experiments),
            "experiments": [
                {
                    "id": e.id,
                    "experiment_type": e.experiment_type,
                    "status": e.status,
                    "created_at": e.created_at,
                    "updated_at": e.updated_at,
                    "candidates_count": len(e.candidates),
                    "concepts_produced": len(e.concept_ids_produced),
                    "thread_id": e.thread_id,
                }
                for e in experiments
            ],
        }

    elif action == "get":
        from app.features.experiments import load_experiment

        experiment_id = body.get("experiment_id")
        if not experiment_id:
            raise HTTPException(400, "experiment_id required")

        experiment = load_experiment(experiment_id)
        if not experiment:
            raise HTTPException(404, f"Experiment {experiment_id} not found")

        return {
            "status": "ok",
            "experiment": experiment.model_dump(),
        }

    elif action == "archive":
        from app.features.experiments import archive_experiment

        experiment_id = body.get("experiment_id")
        if not experiment_id:
            raise HTTPException(400, "experiment_id required")

        experiment = archive_experiment(experiment_id)
        if not experiment:
            raise HTTPException(404, f"Experiment {experiment_id} not found")

        return {
            "status": "ok",
            "experiment_id": experiment.id,
            "new_status": "archived",
        }

    elif action == "results":
        # [EXP-020] Return completed experiment insights with full synthesis content
        from app.features.experiments import SOURCE_IDS_MAX, load_experiments

        try:
            min_confidence = max(0.0, min(1.0, float(body.get("min_confidence", 0.0))))
        except (ValueError, TypeError):
            raise HTTPException(400, "min_confidence must be a number between 0.0 and 1.0")
        experiment_type = body.get("experiment_type")  # optional filter
        include_not_meaningful = body.get("include_not_meaningful", False)
        try:
            limit = max(1, min(200, int(body.get("limit", 50))))
        except (ValueError, TypeError):
            raise HTTPException(400, "limit must be an integer between 1 and 200")

        experiments = load_experiments(
            status=["completed"],
            experiment_type=experiment_type,
            limit=limit,
        )

        results_list = []
        for e in experiments:
            if not e.result:
                continue
            conf = e.result.confidence
            is_not_meaningful = conf == 0.0
            if is_not_meaningful and not include_not_meaningful:
                continue
            if conf < min_confidence:
                continue

            # Extract top source concept IDs from top candidate
            source_concept_ids: list[str] = []
            if e.candidates:
                top_cand = max(e.candidates, key=lambda c: c.score)
                source_concept_ids = top_cand.concept_ids[:SOURCE_IDS_MAX]

            results_list.append(
                {
                    "id": e.id,
                    "experiment_type": e.experiment_type,
                    "synthesis": e.result.synthesis,
                    "confidence": conf,
                    "reasoning_trace": e.result.reasoning_trace,
                    "concept_ids_produced": e.concept_ids_produced,
                    "source_concept_ids": source_concept_ids,
                    "created_at": e.created_at,
                    "updated_at": e.updated_at,
                }
            )

        # Sort by confidence descending
        results_list.sort(key=lambda r: r["confidence"], reverse=True)

        return {
            "status": "ok",
            "count": len(results_list),
            "min_confidence": min_confidence,
            "experiment_type": experiment_type,
            "results": results_list,
        }

    else:
        raise HTTPException(400, f"Unknown action: {action}")


# --- Temporal & Causal Endpoints ---


@app.post("/pith_timeline")
async def pith_timeline_endpoint(request: Request, body: dict = {}):
    """Returns concepts and activity within a time window."""
    try:
        result = temporal.pith_timeline(
            since=body.get("since"),
            until=body.get("until"),
            event_types=body.get("event_types"),
            knowledge_area=body.get("knowledge_area"),
            concept_type=body.get("concept_type"),
            limit=body.get("limit", 100),
            group_by=body.get("group_by"),
        )
        return result
    except Exception as e:
        logger.error(f"pith_timeline error: {e}")
        raise HTTPException(500, _safe_error(e))


@app.post("/pith_knowledge_at")
async def pith_knowledge_at_endpoint(request: Request, body: dict = {}):
    """Returns concepts valid at a specific point in time."""
    point_in_time = body.get("point_in_time")
    if not point_in_time:
        raise HTTPException(400, "point_in_time is required (ISO 8601 datetime)")
    try:
        result = temporal.pith_knowledge_at(
            point_in_time=point_in_time,
            knowledge_area=body.get("knowledge_area"),
            concept_type=body.get("concept_type"),
            limit=body.get("limit", 100),
        )
        return result
    except Exception as e:
        logger.error(f"pith_knowledge_at error: {e}")
        raise HTTPException(500, _safe_error(e))


@app.post("/pith_evolution_of")
async def pith_evolution_of_endpoint(request: Request, body: dict = {}):
    """Walks the supersession chain bidirectionally for a concept."""
    concept_id = body.get("concept_id")
    if not concept_id:
        raise HTTPException(400, "concept_id is required")
    try:
        result = temporal.pith_evolution_of(concept_id=concept_id)
        return result
    except Exception as e:
        logger.error(f"pith_evolution_of error: {e}")
        raise HTTPException(500, _safe_error(e))


@app.post("/pith_trace_cause")
async def pith_trace_cause_endpoint(request: Request, body: dict = {}):
    """Traverse causal DAG to find root causes or consequences."""
    concept_id = body.get("concept_id")
    if not concept_id:
        raise HTTPException(400, "concept_id is required")
    try:
        result = causal.pith_trace_cause(
            concept_id=concept_id,
            direction=body.get("direction", "root_cause"),
            max_depth=body.get("max_depth", 5),
            chain_id=body.get("chain_id"),
        )
        return result
    except Exception as e:
        logger.error(f"pith_trace_cause error: {e}")
        raise HTTPException(500, _safe_error(e))


@app.post("/pith_find_path")
async def pith_find_path_endpoint(request: Request, body: dict = {}):
    """Find typed shortest path between two concepts."""
    from_concept = body.get("from_concept")
    to_concept = body.get("to_concept")
    if not from_concept or not to_concept:
        raise HTTPException(400, "from_concept and to_concept are required")
    try:
        result = causal.pith_find_path(
            from_concept=from_concept,
            to_concept=to_concept,
            max_depth=body.get("max_depth", 5),
            relation_types=body.get("relation_types"),
        )
        return result
    except Exception as e:
        logger.error(f"pith_find_path error: {e}")
        raise HTTPException(500, _safe_error(e))


# --- DATA-057: Targeted stale technology sweep ---


@app.post("/pith_sweep_stale_tech", dependencies=[Depends(verify_api_key)])
async def pith_sweep_stale_tech(dry_run: bool = True):
    """Sweep and supersede concepts referencing eliminated technologies."""
    from app.cognitive.staleness import sweep_stale_technology_refs

    return sweep_stale_technology_refs(dry_run=dry_run)


# =============================================================================
# WS2: Metrics Dashboard
# =============================================================================


@app.get("/metrics/dashboard")
async def metrics_dashboard(since: str | None = None):
    """Return aggregated metrics for the Critical 8 + summary stats.

    Query param:
        since: ISO timestamp lower bound (default: last hour).

    Returns JSON matching Phase 4 WS2 §3.6 response shape.
    """
    try:
        from app.ops.metrics import metrics

        turn_latency = metrics.query_aggregate("conversation_turn_latency_ms", since=since)
        tier2_calls = metrics.query_count("tier2_llm_cost_calls", since=since)
        tier2_latency = metrics.query_aggregate("tier2_llm_latency_ms", since=since)
        contradiction_rate = metrics.query_aggregate("contradiction_detection_rate", since=since)
        cascade_count = metrics.query_count("cascade_propagation_count", since=since)
        cb_trips = metrics.query_count("circuit_breaker_trip_count", since=since)
        retrieval_latency = metrics.query_aggregate("retrieval_search_latency_ms", since=since)

        # Budget overruns — return recent individual events
        budget_overruns_raw = metrics.query("budget_overrun_ms", since=since, limit=50)
        budget_overruns = [
            {"overrun_ms": round(e["value"], 2), "timestamp": e["timestamp"]} for e in budget_overruns_raw
        ]

        return {
            "period": "last_hour" if since is None else f"since_{since}",
            "conversation_turn_latency_ms": {
                "p50": turn_latency["p50"],
                "p95": turn_latency["p95"],
                "p99": turn_latency["p99"],
                "count": turn_latency["count"],
                "avg": turn_latency["avg"],
            },
            "tier2_calls_total": tier2_calls,
            "tier2_latency_ms": {
                "p50": tier2_latency["p50"],
                "p95": tier2_latency["p95"],
                "avg": tier2_latency["avg"],
            },
            "contradiction_detection_rate": {
                "avg": contradiction_rate["avg"],
                "count": contradiction_rate["count"],
            },
            "cascade_propagations": cascade_count,
            "cascade_alert": cascade_count > _cascade_alert_threshold(),  # NITS-001: configurable
            "circuit_breaker_trips": cb_trips,
            "circuit_breaker_alert": cb_trips > _circuit_breaker_alert_threshold(),  # MONITOR-072
            "retrieval_search_latency_ms": {
                "p50": retrieval_latency["p50"],
                "p95": retrieval_latency["p95"],
                "avg": retrieval_latency["avg"],
                "count": retrieval_latency["count"],
            },
            "budget_overruns": budget_overruns,
        }
    except Exception as e:
        logger.error(f"Metrics dashboard error: {e}")
        raise HTTPException(500, _safe_error(e))


@app.get("/metrics/bg_tasks")
async def metrics_bg_tasks(since: str | None = None):
    """STATS-004: Background task success/failure/cancelled rates by task name."""
    try:
        from datetime import timedelta

        from app.ops.metrics import metrics

        if since is None:
            since = (_utc_now() - timedelta(hours=24)).isoformat()
        metrics.flush()
        from app.storage import _db

        with _db() as conn:
            rows = conn.execute(
                """SELECT metric, json_extract(labels, '$.task') as task_name,
                          SUM(value) as total
                   FROM metrics
                   WHERE metric IN ('bg_task_success', 'bg_task_failure', 'bg_task_cancelled')
                     AND timestamp >= ?
                   GROUP BY metric, task_name
                   ORDER BY task_name, metric""",
                (since,),
            ).fetchall()
        tasks = {}
        for metric, task_name, total in rows:
            task_name = task_name or "unknown"
            if task_name not in tasks:
                tasks[task_name] = {"success": 0, "failure": 0, "cancelled": 0}
            kind = metric.replace("bg_task_", "")
            tasks[task_name][kind] = int(total)
        for counts in tasks.values():
            total = counts["success"] + counts["failure"] + counts["cancelled"]
            counts["total"] = total
            counts["failure_rate"] = round(counts["failure"] / max(total, 1), 3)
        return {"since": since, "tasks": tasks}
    except Exception as e:
        raise HTTPException(500, _safe_error(e))


@app.get("/dashboard", response_class=HTMLResponse)
async def metrics_dashboard_html():
    """OBS-01: Human-readable HTML metrics dashboard with Chart.js visualizations."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Pith — Metrics Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  body { font-family: system-ui, sans-serif; background: #0f1117; color: #e2e8f0; margin: 0; padding: 20px; }
  h1 { color: #a78bfa; margin-bottom: 4px; }
  .subtitle { color: #64748b; font-size: 0.875rem; margin-bottom: 24px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 20px; }
  .card { background: #1e2130; border-radius: 12px; padding: 20px; border: 1px solid #2d3148; }
  .card h3 { margin: 0 0 12px; color: #94a3b8; font-size: 0.875rem; text-transform: uppercase; letter-spacing: 0.05em; }
  .stat-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #2d3148; font-size: 0.875rem; }
  .stat-row:last-child { border-bottom: none; }
  .stat-val { color: #a78bfa; font-weight: 600; }
  .chart-wrap { height: 200px; }
  .refresh { color: #64748b; font-size: 0.75rem; margin-top: 8px; }
  .status-ok { color: #34d399; }
  .status-warn { color: #fbbf24; }
  .error { color: #f87171; font-size: 0.875rem; padding: 8px; }
</style>
</head>
<body>
<h1>🧠 Pith — Metrics Dashboard</h1>
<div class="subtitle" id="refresh-ts">Loading...</div>
<div class="grid" id="dashboard-grid"><div class="card"><p>Loading metrics...</p></div></div>
<script>
const BASE = window.location.origin;

async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(r.statusText);
  return r.json();
}

function fmtMs(v) { return v == null ? '—' : (v >= 1000 ? (v/1000).toFixed(1)+'s' : v.toFixed(1)+'ms'); }
function fmtNum(v) { return v == null ? '—' : Number(v).toLocaleString(); }

function makeCard(title, rows) {
  const card = document.createElement('div');
  card.className = 'card';
  card.innerHTML = '<h3>' + title + '</h3>' +
    rows.map(([k, v]) => '<div class="stat-row"><span>' + k + '</span><span class="stat-val">' + v + '</span></div>').join('');
  return card;
}

function makeChartCard(title, labels, data, color) {
  const card = document.createElement('div');
  card.className = 'card';
  const cid = 'c' + Math.random().toString(36).slice(2);
  card.innerHTML = '<h3>' + title + '</h3><div class="chart-wrap"><canvas id="' + cid + '"></canvas></div>';
  setTimeout(() => {
    const ctx = document.getElementById(cid);
    if (!ctx) return;
    new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [{ data: data, borderColor: color || '#a78bfa', backgroundColor: (color || '#a78bfa') + '22',
          tension: 0.3, fill: true, pointRadius: 2 }]
      },
      options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } },
        scales: { x: { ticks: { color: '#64748b', maxTicksLimit: 6 } }, y: { ticks: { color: '#64748b' } } } }
    });
  }, 0);
  return card;
}

async function render() {
  const grid = document.getElementById('dashboard-grid');
  grid.innerHTML = '';
  try {
    const [dash, summary] = await Promise.all([
      fetchJSON(BASE + '/metrics/dashboard'),
      fetchJSON(BASE + '/metrics/summary')
    ]);

    // Summary stats card
    const metrics = summary.metrics || {};
    const metricNames = Object.keys(metrics);
    const summaryRows = metricNames.slice(0, 8).map(name => {
      const m = metrics[name];
      const isLatency = name.includes('latency') || name.includes('ms');
      return [name.replace(/_/g,' '), isLatency ? fmtMs(m.mean) : fmtNum(m.count)];
    });
    grid.appendChild(makeCard('Metric Summary (' + metricNames.length + ' metrics)', summaryRows));

    // Latency cards
    const latencyMetrics = ['conversation_turn_latency_ms', 'session_learn_latency_ms',
      'reflect_latency_ms', 'auto_associate_batch_latency_ms'];
    const latencyRows = latencyMetrics.map(name => {
      const m = metrics[name];
      return [name.replace(/_latency_ms/,'').replace(/_/g,' '),
        m ? fmtMs(m.mean) + ' avg / ' + fmtNum(m.count) + ' calls' : '—'];
    });
    grid.appendChild(makeCard('Hot-Path Latencies', latencyRows));

    // Time-series chart for conversation_turn if available
    const ctData = dash.recent_metrics || dash.metrics || [];
    const ctRows = ctData.filter(r => r.metric === 'conversation_turn_latency_ms').slice(-20);
    if (ctRows.length > 0) {
      const labels = ctRows.map(r => r.timestamp ? r.timestamp.slice(11,16) : '');
      const vals = ctRows.map(r => r.value);
      grid.appendChild(makeChartCard('conversation_turn latency (last 20)', labels, vals, '#a78bfa'));
    }

    // session_learn chart
    const slRows = ctData.filter(r => r.metric === 'session_learn_latency_ms').slice(-20);
    if (slRows.length > 0) {
      const labels = slRows.map(r => r.timestamp ? r.timestamp.slice(11,16) : '');
      const vals = slRows.map(r => r.value);
      grid.appendChild(makeChartCard('session_learn latency (last 20)', labels, vals, '#34d399'));
    }

    document.getElementById('refresh-ts').textContent =
      'Last updated: ' + new Date().toLocaleTimeString() + ' — auto-refreshes every 30s';
  } catch(e) {
    grid.innerHTML = '<div class="card"><p class="error">Error loading metrics: ' + e.message + '</p></div>';
  }
}

render();
setInterval(render, 30000);
</script>
</body>
</html>"""
    return HTMLResponse(content=html)


@app.get("/metrics/summary")
async def metrics_summary_endpoint(days: int = 7):
    """OBS-003: Aggregated metrics summary with per-metric stats and trends."""
    try:
        from app.ops.metrics import metrics

        metrics.flush()
        result = metrics.metrics_summary(days=days)
        return result
    except Exception as e:
        logger.error(f"metrics_summary error: {e}")
        return {"status": "error", "error": _safe_error(e)}


@app.get("/metrics/health_trend")
async def metrics_health_trend(days: int = 7):
    """STATS-005: Pith health score time series over N days."""
    try:
        from datetime import timedelta

        from app.ops.metrics import metrics

        since = (_utc_now() - timedelta(days=days)).isoformat()
        metrics.flush()
        from app.storage import _db

        with _db() as conn:
            rows = conn.execute(
                """SELECT date(timestamp) as day, metric, AVG(value) as avg_val
                   FROM metrics
                   WHERE metric IN ('pith_health_score', 'pith_maturity_score',
                                    'pith_connectivity_score', 'pith_confidence_avg',
                                    'pith_freshness_ratio')
                     AND timestamp >= ?
                   GROUP BY day, metric
                   ORDER BY day""",
                (since,),
            ).fetchall()
        days_data = {}
        for day, metric, avg_val in rows:
            if day not in days_data:
                days_data[day] = {"date": day}
            days_data[day][metric] = round(avg_val, 4)
        series = sorted(days_data.values(), key=lambda d: d["date"])
        degradation_alert = False
        if len(series) >= 2:
            latest = series[-1].get("pith_health_score", 0)
            prev = series[-2].get("pith_health_score", 0)
            if prev > 0 and (prev - latest) > 0.05:
                degradation_alert = True
        return {"days": days, "series": series, "degradation_alert": degradation_alert}
    except Exception as e:
        raise HTTPException(500, _safe_error(e))


# ---------------------------------------------------------------------------
# Unified Observability (TIER2 Developer Surface)
# ---------------------------------------------------------------------------


@app.get("/pith/observability")
async def pith_observability():
    """Single-call system health snapshot.

    Aggregates cognitive health, stats, learning metrics, performance,
    and background task status into one response. Each section is
    independently try/excepted so partial failures don't block the rest.
    """
    from datetime import datetime

    result: dict = {
        "timestamp": datetime.now(UTC).isoformat(),
        "status": "healthy",
    }

    # --- Cognitive health (from /pith_health internals) ---
    try:
        health = reflection_engine.analyze_stability()
        result["cognitive"] = {
            "stability_score": health.get("stability_score", health.get("score")),
            "total_concepts": health.get("total_concepts"),
            "knowledge_areas": health.get("knowledge_areas", {}),
        }
    except Exception as e:
        result["cognitive"] = {"error": _safe_error(e)}

    # --- Stats (from /pith_stats internals) ---
    try:
        from app.storage import get_pith_stats_aggregates

        agg = get_pith_stats_aggregates()
        agg["associations"] = count_associations()
        agg["pending_questions"] = len(question_queue.get_questions(limit=1000))
        result["stats"] = {
            "total_concepts": agg.get("total_concepts", 0),
            "associations": agg.get("associations", 0),
            "avg_confidence": agg.get("avg_confidence", 0),
            "pending_questions": agg.get("pending_questions", 0),
        }
    except Exception as e:
        result["stats"] = {"error": _safe_error(e)}

    # --- Learning metrics (subset of /learning_metrics) ---
    try:
        from app.storage import get_db_connection

        conn = get_db_connection()
        today_count = conn.execute("SELECT COUNT(*) FROM concepts WHERE created_at >= date('now')").fetchone()[0]
        result["learning"] = {
            "concepts_today": today_count,
            "total_concepts": result.get("stats", {}).get("total_concepts", 0),
        }
    except Exception as e:
        result["learning"] = {"error": _safe_error(e)}

    # --- Performance (subset of /metrics/dashboard) ---
    try:
        from app.ops.metrics import metrics

        turn_latency = metrics.query_aggregate("conversation_turn_latency_ms")
        retrieval_latency = metrics.query_aggregate("retrieval_search_latency_ms")
        cb_trips = metrics.query_count("circuit_breaker_trip_count")
        budget_overruns = metrics.query_count("budget_overrun_ms")
        result["performance"] = {
            "conversation_turn_latency_p95_ms": turn_latency.get("p95"),
            "retrieval_latency_p95_ms": retrieval_latency.get("p95"),
            "circuit_breaker_trips": cb_trips,
            "budget_overruns": budget_overruns,
        }
    except Exception as e:
        result["performance"] = {"error": _safe_error(e)}

    # --- Background tasks ---
    try:
        from app.ops.metrics import metrics

        bg = metrics.query_bg_tasks()
        result["background_tasks"] = {
            "running": bg.get("running", 0),
            "queued": bg.get("queued", 0),
        }
    except Exception as e:
        result["background_tasks"] = {"error": _safe_error(e)}

    # --- Derive overall status ---
    try:
        stability = result.get("cognitive", {}).get("stability_score") or 0
        cb = result.get("performance", {}).get("circuit_breaker_trips") or 0
        p95 = result.get("performance", {}).get("conversation_turn_latency_p95_ms") or 0
        if stability < 0.4 or cb >= 5:
            result["status"] = "unhealthy"
        elif stability < 0.7 or cb > 0 or (p95 and p95 > 5000):
            result["status"] = "degraded"
        else:
            result["status"] = "healthy"
    except Exception:
        result["status"] = "unknown"

    return result


@app.get("/metrics/compaction_summary")
def metrics_compaction_summary():
    """CTX-006: Compaction reinjection event analytics.

    Returns aggregate stats across all recorded compaction_reinjection
    governance events, including resume rate and average recovery quality.
    """
    from app.storage import _db

    with _db() as conn:
        row = conn.execute(
            """SELECT
                COUNT(*) as total_events,
                SUM(CASE WHEN json_extract(details, '$.has_resume') = 1 THEN 1 ELSE 0 END) as has_resume_count,
                AVG(json_extract(details, '$.turn_count')) as avg_turn_count,
                MAX(json_extract(details, '$.turn_count')) as max_turn_count,
                AVG(json_extract(details, '$.recovery_quality')) as avg_recovery_quality,
                MIN(created_at) as first_event_at,
                MAX(created_at) as most_recent_at
               FROM governance_events
               WHERE event_type = 'compaction_reinjection'"""
        ).fetchone()
        sessions_row = conn.execute(
            """SELECT COUNT(DISTINCT session_id) as sessions_affected
               FROM governance_events
               WHERE event_type = 'compaction_reinjection'"""
        ).fetchone()
    total = row["total_events"] or 0
    has_resume = row["has_resume_count"] or 0
    return {
        "total_events": total,
        "sessions_affected": sessions_row["sessions_affected"] or 0,
        "has_resume_rate": round(has_resume / total, 4) if total else 0.0,
        "avg_turn_count": round(row["avg_turn_count"] or 0, 1),
        "max_turn_count": row["max_turn_count"] or 0,
        "avg_recovery_quality": (
            round(row["avg_recovery_quality"], 4) if row["avg_recovery_quality"] is not None else None
        ),
        "first_event_at": row["first_event_at"],
        "most_recent_at": row["most_recent_at"],
    }


# =============================================================================
# OBS-009/010, MEASURE-002/015: Observability Endpoints — Sub-Sprint O-C
# =============================================================================

# MEASURE-015: Health threshold config — direction-aware evaluation
_HEALTH_THRESHOLDS = {
    "concept_count": {"green": 100, "amber": 50, "direction": "higher_better"},
    "session_count_7d": {"green": 5, "amber": 2, "direction": "higher_better"},
    "avg_confidence": {"green": 0.3, "amber": 0.2, "direction": "higher_better"},
    "quarantined_pct": {"green": 0.03, "amber": 0.05, "red": 0.10, "direction": "lower_better"},
    "superseded_pct": {"green": 0.10, "amber": 0.15, "red": 0.30, "direction": "lower_better"},
    "promotion_rate_7d": {"green": 1, "amber": 0, "direction": "higher_better"},
}


def _evaluate_metric(value, thresholds):
    """Evaluate a metric against green/amber/red thresholds."""
    direction = thresholds.get("direction", "higher_better")
    if direction == "lower_better":
        if "red" in thresholds and value >= thresholds["red"]:
            return "red"
        if "green" in thresholds and value <= thresholds["green"]:
            return "green"
        return "amber"
    else:
        if "green" in thresholds and value >= thresholds["green"]:
            return "green"
        if "amber" in thresholds and value >= thresholds["amber"]:
            return "amber"
        return "red"


@app.get("/metrics/governance_summary", dependencies=[Depends(verify_api_key)])
async def metrics_governance_summary(days: int = 7):
    """OBS-009: Governance event type distribution with trend."""
    try:
        from datetime import timedelta

        from app.storage import _db

        with _db() as conn:
            cutoff = (_utc_now() - timedelta(days=days)).isoformat()
            prior_cutoff = (_utc_now() - timedelta(days=days * 2)).isoformat()

            current = conn.execute(
                """SELECT event_type, COUNT(*) as cnt
                   FROM governance_events WHERE created_at >= ?
                   GROUP BY event_type ORDER BY cnt DESC LIMIT 25""",
                (cutoff,),
            ).fetchall()

            prior = conn.execute(
                """SELECT event_type, COUNT(*) as cnt
                   FROM governance_events WHERE created_at >= ? AND created_at < ?
                   GROUP BY event_type ORDER BY cnt DESC LIMIT 25""",
                (prior_cutoff, cutoff),
            ).fetchall()

            total_current = sum(r[1] for r in current)
            prior_map = {r[0]: r[1] for r in prior}
            total_prior = sum(r[1] for r in prior)

            event_types = []
            for event_type, cnt in current:
                prior_cnt = prior_map.get(event_type, 0)
                trend = ((cnt - prior_cnt) / prior_cnt * 100) if prior_cnt > 0 else None
                event_types.append(
                    {
                        "event_type": event_type,
                        "count": cnt,
                        "prior_count": prior_cnt,
                        "trend_pct": round(trend, 1) if trend is not None else None,
                    }
                )

            return {
                "period_days": days,
                "total_events": total_current,
                "prior_total": total_prior,
                "event_types": event_types,
            }
    except Exception as e:
        logger.error(f"governance_summary error: {e}")
        return {"status": "error", "error": _safe_error(e)}


@app.get("/metrics/session_activity", dependencies=[Depends(verify_api_key)])
async def metrics_session_activity(days: int = 7):
    """OBS-010: Session activity aggregates."""
    try:
        from datetime import timedelta

        from app.storage import _db

        with _db() as conn:
            cutoff = (_utc_now() - timedelta(days=days)).isoformat()

            stats = conn.execute(
                """SELECT
                     COUNT(*) as total_sessions,
                     AVG(learning_event_count) as avg_learning_events,
                     AVG(concepts_created) as avg_concepts_created,
                     AVG(concepts_evolved) as avg_concepts_evolved,
                     SUM(learning_event_count) as total_learning_events
                   FROM sessions WHERE started_at >= ?""",
                (cutoff,),
            ).fetchone()

            daily = conn.execute(
                """SELECT date(started_at) as day, COUNT(*) as cnt
                   FROM sessions WHERE started_at >= ?
                   GROUP BY day ORDER BY day""",
                (cutoff,),
            ).fetchall()

            models = conn.execute(
                """SELECT COALESCE(model_id, 'unknown') as model, COUNT(*) as cnt
                   FROM sessions WHERE started_at >= ?
                   GROUP BY model ORDER BY cnt DESC""",
                (cutoff,),
            ).fetchall()

            return {
                "period_days": days,
                "total_sessions": stats[0] or 0,
                "avg_learning_events": round(stats[1] or 0, 1),
                "avg_concepts_created": round(stats[2] or 0, 1),
                "avg_concepts_evolved": round(stats[3] or 0, 1),
                "total_learning_events": stats[4] or 0,
                "sessions_per_day": [{"date": r[0], "count": r[1]} for r in daily],
                "model_distribution": [{"model": r[0], "count": r[1]} for r in models],
            }
    except Exception as e:
        logger.error(f"session_activity error: {e}")
        return {"status": "error", "error": _safe_error(e)}


@app.get("/metrics/graduation_stats", dependencies=[Depends(verify_api_key)])
async def metrics_graduation_stats(days: int = 7):
    """MEASURE-002: Graduation pipeline metrics."""
    try:
        from datetime import timedelta

        from app.storage import _db

        with _db() as conn:
            cutoff = (_utc_now() - timedelta(days=days)).isoformat()

            maturity_dist = conn.execute(
                """SELECT maturity, COUNT(*) as cnt FROM concepts
                   WHERE is_current = 1
                   GROUP BY maturity ORDER BY cnt DESC"""
            ).fetchall()

            promotions = conn.execute(
                """SELECT COUNT(*) FROM governance_events
                   WHERE event_type = 'MATURITY_PROMOTED' AND created_at >= ?""",
                (cutoff,),
            ).fetchone()[0]

            daily_promo = conn.execute(
                """SELECT date(created_at) as day, COUNT(*) as cnt
                   FROM governance_events
                   WHERE event_type = 'MATURITY_PROMOTED' AND created_at >= ?
                   GROUP BY day ORDER BY day""",
                (cutoff,),
            ).fetchall()

            contradictions = conn.execute(
                """SELECT COUNT(*) FROM governance_events
                   WHERE event_type = 'CONTRADICTION_DETECTED' AND created_at >= ?""",
                (cutoff,),
            ).fetchone()[0]

            return {
                "period_days": days,
                "maturity_distribution": {r[0] or "NULL": r[1] for r in maturity_dist},
                "promotions_in_period": promotions,
                "contradictions_in_period": contradictions,
                "contradiction_to_promotion_ratio": (round(contradictions / promotions, 2) if promotions > 0 else None),
                "promotions_per_day": [{"date": r[0], "count": r[1]} for r in daily_promo],
            }
    except Exception as e:
        logger.error(f"graduation_stats error: {e}")
        return {"status": "error", "error": _safe_error(e)}


@app.get("/health/summary", dependencies=[Depends(verify_api_key)])
async def health_summary():
    """MEASURE-015: Structured health summary with per-metric status."""
    try:
        from datetime import timedelta

        from app.storage import _db

        with _db() as conn:
            cutoff_7d = (_utc_now() - timedelta(days=7)).isoformat()

            active = conn.execute(
                "SELECT COUNT(*) FROM concepts WHERE is_current = 1 AND status = 'active'"
            ).fetchone()[0]

            avg_conf = (
                conn.execute(
                    "SELECT AVG(confidence) FROM concepts WHERE is_current = 1 AND status = 'active'"
                ).fetchone()[0]
                or 0.0
            )

            quarantined = conn.execute("SELECT COUNT(*) FROM concepts WHERE maturity = 'QUARANTINED'").fetchone()[0]

            superseded = conn.execute("SELECT COUNT(*) FROM concepts WHERE is_current = 0").fetchone()[0]

            total = conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]

            sessions_7d = conn.execute("SELECT COUNT(*) FROM sessions WHERE started_at >= ?", (cutoff_7d,)).fetchone()[
                0
            ]

            promotions_7d = conn.execute(
                """SELECT COUNT(*) FROM governance_events
                   WHERE event_type = 'MATURITY_PROMOTED' AND created_at >= ?""",
                (cutoff_7d,),
            ).fetchone()[0]

            quar_pct = quarantined / active if active > 0 else 0
            super_pct = superseded / total if total > 0 else 0

            metrics_result = {
                "concept_count": {
                    "value": active,
                    "status": _evaluate_metric(active, _HEALTH_THRESHOLDS["concept_count"]),
                },
                "session_count_7d": {
                    "value": sessions_7d,
                    "status": _evaluate_metric(sessions_7d, _HEALTH_THRESHOLDS["session_count_7d"]),
                },
                "avg_confidence": {
                    "value": round(avg_conf, 3),
                    "status": _evaluate_metric(avg_conf, _HEALTH_THRESHOLDS["avg_confidence"]),
                },
                "quarantined_pct": {
                    "value": round(quar_pct, 3),
                    "status": _evaluate_metric(quar_pct, _HEALTH_THRESHOLDS["quarantined_pct"]),
                },
                "superseded_pct": {
                    "value": round(super_pct, 3),
                    "status": _evaluate_metric(super_pct, _HEALTH_THRESHOLDS["superseded_pct"]),
                },
                "promotion_rate_7d": {
                    "value": promotions_7d,
                    "status": _evaluate_metric(promotions_7d, _HEALTH_THRESHOLDS["promotion_rate_7d"]),
                },
            }

            statuses = [m["status"] for m in metrics_result.values()]
            if "red" in statuses:
                overall = "red"
            elif "amber" in statuses:
                overall = "amber"
            else:
                overall = "green"

            return {
                "overall": overall,
                "metrics": metrics_result,
                "thresholds": _HEALTH_THRESHOLDS,
            }
    except Exception as e:
        logger.error(f"health_summary error: {e}")
        return {"status": "error", "error": _safe_error(e)}


# =============================================================================


# =============================================================================
# EPISODE-001: Episode Query Endpoint
# =============================================================================


@app.get("/pith_episodes", dependencies=[Depends(verify_api_key)])
async def pith_episodes(
    session_id: str | None = None,
    query: str | None = None,
    limit: int = 50,
    offset: int = 0,
):
    """EPISODE-001: Query episodes with optional session filter and text search."""
    try:
        from app.storage import _db

        with _db() as conn:
            conditions = []
            params = []

            if session_id:
                conditions.append("session_id = ?")
                params.append(session_id)

            if query:
                conditions.append("(intent_summary LIKE ? OR classification LIKE ?)")
                params.extend([f"%{query}%", f"%{query}%"])

            where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

            # Count total
            total = conn.execute(f"SELECT COUNT(*) FROM episodes {where}", params).fetchone()[0]

            # Fetch page
            rows = conn.execute(
                f"""SELECT id, session_id, turn_number, extracted_concept_ids,
                       intent_summary, classification, world_timestamp, created_at
                   FROM episodes {where}
                   ORDER BY created_at DESC
                   LIMIT ? OFFSET ?""",
                params + [limit, offset],
            ).fetchall()

            episodes = []
            for r in rows:
                import json as _json

                concept_ids = []
                try:
                    concept_ids = _json.loads(r[3]) if r[3] else []
                except (ValueError, TypeError):
                    pass

                episodes.append(
                    {
                        "id": r[0],
                        "session_id": r[1],
                        "turn_number": r[2],
                        "concept_ids": concept_ids,
                        "intent_summary": r[4],
                        "classification": r[5],
                        "timestamp": r[6],
                        "created_at": r[7],
                    }
                )

            return {
                "total": total,
                "limit": limit,
                "offset": offset,
                "episodes": episodes,
            }
    except Exception as e:
        logger.error(f"pith_episodes error: {e}")
        return {"status": "error", "error": _safe_error(e)}


# CKO (Compound Knowledge Objects) Endpoints — Layer 4
# =============================================================================


class CKOCreateRequest(PydanticBaseModel):
    """Request body for creating a CKO."""

    title: str
    concept_ids: list[str]
    synthesis: str
    knowledge_area: str = "general"
    cko_type: str = "analysis"


class CKOUpdateRequest(PydanticBaseModel):
    """Request body for updating a CKO."""

    concept_ids: list[str] | None = None
    synthesis: str | None = None


@app.post("/pith/cko", dependencies=[Depends(verify_api_key)])
def cko_create(request: CKOCreateRequest):
    """Create a new Compound Knowledge Object."""
    from app.features.cko import create_cko
    from app.storage import managed_write_db

    try:
        with managed_write_db(operation="cko_create") as conn:
            cko = create_cko(
                conn=conn,
                title=request.title,
                concept_ids=request.concept_ids,
                synthesis=request.synthesis,
                knowledge_area=request.knowledge_area,
                cko_type=request.cko_type,
            )
        return cko.to_dict()
    except Exception as e:
        logger.error(f"CKO create failed: {e}")
        raise HTTPException(500, _safe_error(e))


@app.get("/pith/cko/{cko_id}")
def cko_get(cko_id: str):
    """Load a single CKO by ID."""
    from app.features.cko import load_cko
    from app.storage import read_snapshot_db

    with read_snapshot_db("cko_get") as conn:
        cko = load_cko(conn, cko_id, ensure_table=False)
    if not cko:
        raise HTTPException(404, f"CKO {cko_id} not found")
    return cko.to_dict()


@app.get("/pith/cko")
def cko_list(
    status: str | None = None,
    knowledge_area: str | None = None,
    limit: int = 50,
):
    """List CKOs with optional filters."""
    from app.features.cko import list_ckos
    from app.storage import read_snapshot_db

    with read_snapshot_db("cko_list") as conn:
        ckos = list_ckos(conn, status=status, knowledge_area=knowledge_area, limit=limit, ensure_table=False)
    return {"count": len(ckos), "ckos": [c.to_dict() for c in ckos]}


@app.post("/pith/cko/search")
def cko_search(query_area: str | None = None, max_results: int = 3):
    """Search CKOs for context assembly."""
    from app.features.cko import search_ckos
    from app.storage import managed_write_db

    with managed_write_db(operation="cko_search") as conn:
        ckos = search_ckos(conn, query_area=query_area, max_results=max_results)
    return {"count": len(ckos), "ckos": [c.to_dict() for c in ckos]}


@app.put("/pith/cko/{cko_id}", dependencies=[Depends(verify_api_key)])
def cko_update(cko_id: str, request: CKOUpdateRequest):
    """Update a CKO's synthesis and/or constituents."""
    from app.features.cko import load_cko, update_cko_synthesis
    from app.storage import managed_write_db

    try:
        with managed_write_db(operation="cko_update") as conn:
            existing = load_cko(conn, cko_id)
            if not existing:
                raise HTTPException(404, f"CKO {cko_id} not found")
            cko = update_cko_synthesis(
                conn=conn,
                cko_id=cko_id,
                new_synthesis=request.synthesis,
                new_concept_ids=request.concept_ids,
            )
        return cko.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CKO update failed: {e}")
        raise HTTPException(500, _safe_error(e))


@app.delete("/pith/cko/{cko_id}", dependencies=[Depends(verify_api_key)])
def cko_delete(cko_id: str):
    """Delete a CKO."""
    from app.features.cko import delete_cko
    from app.storage import managed_write_db

    with managed_write_db(operation="cko_delete") as conn:
        success = delete_cko(conn, cko_id)
    if not success:
        raise HTTPException(404, f"CKO {cko_id} not found")
    return {"deleted": cko_id}


@app.post("/pith/cko/lifecycle", dependencies=[Depends(verify_api_key)])
def cko_lifecycle():
    """Run CKO lifecycle management: refresh scores, archive stale, find merge candidates."""
    from app.features.cko import run_cko_lifecycle
    from app.storage import managed_write_db

    try:
        with managed_write_db(operation="cko_lifecycle") as conn:
            result = run_cko_lifecycle(conn)
        return result
    except Exception as e:
        logger.error(f"CKO lifecycle failed: {e}")
        raise HTTPException(500, _safe_error(e))


@app.post("/pith/cko/{cko_id}/refresh", dependencies=[Depends(verify_api_key)])
def cko_refresh(cko_id: str):
    """Refresh a single CKO's scores from current constituent state."""
    from app.features.cko import refresh_cko
    from app.storage import managed_write_db

    with managed_write_db(operation="cko_refresh") as conn:
        cko = refresh_cko(conn, cko_id)
    if not cko:
        raise HTTPException(404, f"CKO {cko_id} not found")
    return cko.to_dict()


# =============================================================================
# User Policies (Phase 3)
# =============================================================================


@app.post("/pith/policies", dependencies=[Depends(verify_api_key)])
def create_policy_endpoint(body: dict):
    """Create a new user policy."""
    from app.governance.user_policies import create_policy

    required = {"policy_type", "rule", "action"}
    missing = required - set(body.keys())
    if missing:
        raise HTTPException(400, f"Missing required fields: {missing}")
    try:
        policy = create_policy(
            policy_type=body["policy_type"],
            rule=body["rule"],
            action=body["action"],
            condition=body.get("condition"),
            priority=body.get("priority", 50),
        )
        return {"status": "created", "policy": asdict(policy)}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/pith/policies")
def list_policies_endpoint(policy_type: str = None, include_disabled: bool = False):
    """List user policies, optionally filtered by type."""
    from app.governance.user_policies import list_policies

    try:
        policies = list_policies(policy_type=policy_type, include_disabled=include_disabled)
        return {"policies": [asdict(p) for p in policies]}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/pith/policies/{policy_id}")
def get_policy_endpoint(policy_id: str):
    """Get a single policy by ID."""
    from app.governance.user_policies import get_policy

    policy = get_policy(policy_id)
    if not policy:
        raise HTTPException(404, f"Policy {policy_id} not found")
    return {"policy": asdict(policy)}


@app.put("/pith/policies/{policy_id}", dependencies=[Depends(verify_api_key)])
def update_policy_endpoint(policy_id: str, body: dict):
    """Update an existing policy."""
    from app.governance.user_policies import update_policy

    policy = update_policy(
        policy_id=policy_id,
        rule=body.get("rule"),
        action=body.get("action"),
        condition=body.get("condition"),
        priority=body.get("priority"),
        enabled=body.get("enabled"),
    )
    if not policy:
        raise HTTPException(404, f"Policy {policy_id} not found")
    return {"status": "updated", "policy": asdict(policy)}


@app.delete("/pith/policies/{policy_id}", dependencies=[Depends(verify_api_key)])
def delete_policy_endpoint(policy_id: str):
    """Soft-delete a policy (disable it)."""
    from app.governance.user_policies import delete_policy

    success = delete_policy(policy_id)
    if not success:
        raise HTTPException(404, f"Policy {policy_id} not found")
    return {"status": "deleted", "policy_id": policy_id}


# =============================================================================
# P4a: Post-Response Validation Endpoint
# =============================================================================


@app.post("/validate_response", dependencies=[Depends(verify_api_key)])
async def validate_response_endpoint(request: Request, body: dict = {}):
    """Validate a draft response against active constraints.

    P4a: Three-tier validation (negation + entity overlap + embedding escalation).
    Feature-flagged via POST_RESPONSE_VALIDATION_ENABLED.

    Body:
        response_text (str): The draft response to validate
        constraint_set (dict): The constraint_set from conversation_turn
        skip_validation (bool, optional): Skip all validation (opt-out)
    """
    from app.cognitive.prediction_error import validate_response

    response_text = body.get("response_text", "")
    constraint_set = body.get("constraint_set", {})
    skip_validation = body.get("skip_validation", False)

    if not response_text:
        return {"error": "response_text is required"}
    if not constraint_set:
        return {"error": "constraint_set is required"}

    result = validate_response(
        response_text=response_text,
        constraint_set=constraint_set,
        skip_validation=skip_validation,
    )
    return result


# ---------------------------------------------------------------------------
# P4b: Belief Diff
# ---------------------------------------------------------------------------


@app.post("/belief_diff", dependencies=[Depends(verify_api_key)])
async def belief_diff_endpoint(request: Request, body: dict = {}):
    """Compare pith belief state between two timestamps.

    Body params:
        t1 (str): ISO datetime for earlier state
        t2 (str): ISO datetime for later state
        knowledge_area (str, optional): Filter to specific domain
    """
    from app.cognitive.belief_diff import belief_diff

    t1 = body.get("t1", "")
    t2 = body.get("t2", "")
    knowledge_area = body.get("knowledge_area")

    if not t1 or not t2:
        return {"error": "Both t1 and t2 timestamps are required"}

    return belief_diff(t1=t1, t2=t2, knowledge_area=knowledge_area)


# ---------------------------------------------------------------------------
# P4c: Epistemic Network Migration
# ---------------------------------------------------------------------------


@app.post("/migrate_epistemic_networks", dependencies=[Depends(verify_api_key)])
async def migrate_epistemic_networks_endpoint(request: Request, body: dict = {}):
    """Migrate concepts to extended epistemic networks.

    Body params:
        dry_run (bool): If True (default), report changes without applying them
    """
    from app.governance.epistemic import migrate_epistemic_networks

    dry_run = body.get("dry_run", True)
    return migrate_epistemic_networks(dry_run=dry_run)


# =============================================================================
# AGENT-002: Agent Token Management
# =============================================================================


@app.post("/agent_tokens", dependencies=[Depends(verify_api_key)])
def create_token_endpoint(body: dict):
    """Create a new agent bearer token."""
    from app.storage import create_agent_token

    agent_id = body.get("agent_id")
    if not agent_id:
        raise HTTPException(400, "agent_id is required")
    label = body.get("label", "")
    result = create_agent_token(agent_id=agent_id, label=label)
    return result


@app.delete("/agent_tokens/{token}", dependencies=[Depends(verify_api_key)])
def revoke_token_endpoint(token: str):
    """Revoke an agent token."""
    from app.storage import revoke_agent_token

    success = revoke_agent_token(token)
    if not success:
        raise HTTPException(404, "Token not found or already revoked")
    return {"status": "revoked", "token_prefix": token[:9] + "..."}


@app.get("/agent_tokens", dependencies=[Depends(verify_api_key)])
def list_tokens_endpoint(agent_id: str = None):
    """List agent tokens (masked). Optionally filter by agent_id."""
    from app.storage import list_agent_tokens

    return {"tokens": list_agent_tokens(agent_id=agent_id)}


@app.get("/agent_tokens/resolve")
def resolve_token_endpoint(token: str):
    """Resolve a bearer token to agent_id. Used by MCP HTTP server.
    No API key required — the token itself IS the credential."""
    from app.storage import resolve_agent_token

    agent_id = resolve_agent_token(token)
    if not agent_id:
        raise HTTPException(401, "Invalid or revoked token")
    return {"agent_id": agent_id}


# --- INGEST-037: Verbatim fragment endpoints ---


@app.post("/verbatim/store", dependencies=[Depends(verify_api_key)])
def store_verbatim(
    concept_id: str,
    fragment_type: str = "text",
    content: str | None = None,
    pointer_uri: str | None = None,
    pointer_meta: dict | None = None,
    evidence_id: str | None = None,
):
    """Store a verbatim fragment for a concept."""
    from app.storage import load_concept, save_verbatim_fragment

    concept = load_concept(concept_id)
    if not concept:
        raise HTTPException(404, f"Concept {concept_id} not found")
    if not content and not pointer_uri:
        raise HTTPException(400, "Either content or pointer_uri is required")

    version = concept.get("version") if isinstance(concept, dict) else getattr(concept, "version", None)
    frag_id = save_verbatim_fragment(
        concept_id=concept_id,
        fragment_type=fragment_type,
        content=content,
        pointer_uri=pointer_uri,
        pointer_meta=pointer_meta,
        evidence_id=evidence_id,
        concept_version=version,
    )
    if frag_id is None:
        raise HTTPException(413, "Verbatim budget exceeded for this concept")
    return {"fragment_id": frag_id, "concept_id": concept_id}


@app.get("/verbatim/{concept_id}", dependencies=[Depends(verify_api_key)])
def get_verbatim(concept_id: str, limit: int = 10):
    """Get all verbatim fragments for a concept."""
    from app.storage import get_verbatim_fragments

    fragments = get_verbatim_fragments(concept_id, limit=limit)
    return {"concept_id": concept_id, "fragments": fragments, "count": len(fragments)}


@app.delete("/verbatim/{fragment_id}", dependencies=[Depends(verify_api_key)])
def delete_verbatim(fragment_id: str):
    """Delete a specific verbatim fragment."""
    from app.storage import delete_verbatim_fragment

    deleted = delete_verbatim_fragment(fragment_id)
    if not deleted:
        raise HTTPException(404, f"Fragment {fragment_id} not found")
    return {"deleted": fragment_id}


@app.get("/verbatim_stats", dependencies=[Depends(verify_api_key)])
def verbatim_stats():
    """Get aggregate verbatim fragment statistics."""
    from app.storage import get_verbatim_stats

    return get_verbatim_stats()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=PITH_HOST, port=PITH_PORT)
