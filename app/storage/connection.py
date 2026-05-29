"""Storage sub-module: connection.

Database connection management and backend shims.
Extracted from storage/__init__.py during Item 2b decomposition.
"""
import inspect
import logging
import os
import sqlite3
import threading
import time
from collections import Counter, deque
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from app.storage.backend import get_backend

logger = logging.getLogger(__name__)

_REQUEST_DB_ROOT: ContextVar[str | None] = ContextVar("request_db_root", default=None)
_REQUEST_DB_DEPTH: ContextVar[int] = ContextVar("request_db_depth", default=0)
_REQUEST_DB_STATE: ContextVar[dict | None] = ContextVar("request_db_state", default=None)

_AUDIT_LOCK = threading.Lock()
_AUDIT_TOTALS: Counter[str] = Counter()
_AUDIT_BY_ROOT: Counter[str] = Counter()
_AUDIT_BY_HELPER: Counter[str] = Counter()
_AUDIT_BY_CALLSITE: Counter[str] = Counter()
_AUDIT_RECENT = deque(maxlen=25)
_THIS_FILE = Path(__file__).resolve()
_APP_ROOT = _THIS_FILE.parents[1]


def _audit_enabled() -> bool:
    from app.core.config import get_feature_flag

    return get_feature_flag("DB_BOUNDARY_AUDIT_ENABLED", True)


def _guard_mode() -> str:
    from app.core import config as _cfg

    mode = _cfg.os.environ.get("PITH_DB_BOUNDARY_GUARD_MODE", "shadow").strip().lower()
    if mode not in {"off", "shadow", "enforce"}:
        return "shadow"
    return mode


def _record_metric(
    metric_name: str,
    *,
    value: float = 1.0,
    labels: dict[str, str | int | float] | None = None,
) -> None:
    # Route ops-metrics access through app.core.metrics_facade (DEBT-244 approved
    # single crossing point) so import-linter Contract 2 (Storage→Core-only) stays
    # green. Direct `from app.ops.metrics import ...` from this module is forbidden.
    # `value` and `labels` are keyword-only to prevent the regression where a
    # legacy caller passed a dict as the second positional arg (intended for
    # labels) and it silently became `value`, triggering buffered metric records
    # with dict values and cascading through flush() → _db() → _record_managed_entry,
    # which falsely bumped by_request_root audit counters. Caught in CI on PR #40.
    try:
        from app.core.metrics_facade import metrics as _metrics

        _metrics.record(metric_name, value, labels=labels)
    except Exception:
        logger.warning("db boundary metric %s failed", metric_name, exc_info=True)


def _snapshot_event(event: dict) -> None:
    with _AUDIT_LOCK:
        _AUDIT_RECENT.appendleft(event)


def _detect_callsite() -> str:
    for frame_info in inspect.stack()[2:]:
        frame_path = Path(frame_info.filename).resolve()
        if frame_path == _THIS_FILE:
            continue
        try:
            relative_path = frame_path.relative_to(_APP_ROOT.parent)
        except ValueError:
            relative_path = frame_path
        return f"{relative_path}:{frame_info.function}:{frame_info.lineno}"
    return "unknown"


def _record_boundary_hit(helper_name: str) -> None:
    request_root = _REQUEST_DB_ROOT.get()
    if not request_root or not _audit_enabled():
        return

    mode = _guard_mode()
    callsite = _detect_callsite()
    state = _REQUEST_DB_STATE.get() or {}
    state["raw_helper_hits"] = int(state.get("raw_helper_hits", 0)) + 1
    state.setdefault("raw_helpers", Counter())[helper_name] += 1
    state.setdefault("raw_callsites", Counter())[callsite] += 1

    with _AUDIT_LOCK:
        _AUDIT_TOTALS["raw_helper_hits"] += 1
        _AUDIT_BY_ROOT[request_root] += 1
        _AUDIT_BY_HELPER[helper_name] += 1
        _AUDIT_BY_CALLSITE[callsite] += 1
        hit_count = _AUDIT_TOTALS["raw_helper_hits"]

    event = {
        "event": "raw_helper_hit",
        "request_root": request_root,
        "helper": helper_name,
        "callsite": callsite,
        "mode": mode,
        "timestamp": time.time(),
    }
    _snapshot_event(event)
    _record_metric(
        "db_boundary_raw_helper_hit",
        labels={"request_root": request_root, "helper": helper_name, "callsite": callsite, "mode": mode},
    )

    if hit_count <= 5 or hit_count % 50 == 0:
        logger.warning(
            "DB boundary %s: raw helper %s used under request root %s via %s",
            mode.upper(),
            helper_name,
            request_root,
            callsite,
        )

    if mode == "enforce":
        with _AUDIT_LOCK:
            _AUDIT_TOTALS["raw_helper_blocks"] += 1
        _record_metric(
            "db_boundary_raw_helper_block",
            labels={"request_root": request_root, "helper": helper_name, "callsite": callsite},
        )
        raise RuntimeError(
            f"Raw DB helper {helper_name} is blocked for request root {request_root} in enforce mode ({callsite})"
        )


def _record_managed_entry(kind: str, elapsed_ms: float) -> None:
    request_root = _REQUEST_DB_ROOT.get()
    if not request_root or not _audit_enabled():
        return

    state = _REQUEST_DB_STATE.get() or {}
    key = {
        "db_immediate": "managed_immediate_entries",
        "read_snapshot": "managed_read_snapshot_entries",
    }.get(kind, "managed_db_entries")
    state[key] = int(state.get(key, 0)) + 1

    with _AUDIT_LOCK:
        _AUDIT_TOTALS[key] += 1
        _AUDIT_BY_ROOT[request_root] += 1

    _record_metric(
        "db_boundary_managed_entry",
        labels={"request_root": request_root, "kind": kind},
    )
    _record_metric(
        "db_boundary_managed_hold_ms",
        value=round(elapsed_ms, 2),
        labels={"request_root": request_root, "kind": kind},
    )


@contextmanager
def request_db_scope(request_root: str):
    """Label the current request root for DB-boundary observability."""
    current_depth = _REQUEST_DB_DEPTH.get()
    depth_token = _REQUEST_DB_DEPTH.set(current_depth + 1)
    root_token = None
    state_token = None

    if current_depth == 0:
        root_token = _REQUEST_DB_ROOT.set(request_root)
        state_token = _REQUEST_DB_STATE.set(
            {
                "request_root": request_root,
                "started_at": time.perf_counter(),
                "raw_helper_hits": 0,
                "managed_db_entries": 0,
                "managed_immediate_entries": 0,
                "managed_read_snapshot_entries": 0,
                "raw_helpers": Counter(),
                "raw_callsites": Counter(),
            }
        )
        if _audit_enabled():
            with _AUDIT_LOCK:
                _AUDIT_TOTALS["request_scopes"] += 1
            _record_metric("db_boundary_request_scope", labels={"request_root": request_root})

    try:
        yield
    finally:
        _REQUEST_DB_DEPTH.reset(depth_token)
        if current_depth == 0:
            state = _REQUEST_DB_STATE.get() or {}
            elapsed_ms = (time.perf_counter() - state.get("started_at", time.perf_counter())) * 1000
            event = {
                "event": "request_scope_summary",
                "request_root": request_root,
                "elapsed_ms": round(elapsed_ms, 2),
                "raw_helper_hits": int(state.get("raw_helper_hits", 0)),
                "managed_db_entries": int(state.get("managed_db_entries", 0)),
                "managed_immediate_entries": int(state.get("managed_immediate_entries", 0)),
                "managed_read_snapshot_entries": int(state.get("managed_read_snapshot_entries", 0)),
                "raw_helpers": dict(state.get("raw_helpers", {})),
                "raw_callsites": dict(state.get("raw_callsites", {})),
                "timestamp": time.time(),
            }
            _snapshot_event(event)
            _record_metric(
                "db_boundary_request_scope_duration_ms",
                value=round(elapsed_ms, 2),
                labels={"request_root": request_root},
            )
            if root_token is not None:
                _REQUEST_DB_ROOT.reset(root_token)
            if state_token is not None:
                _REQUEST_DB_STATE.reset(state_token)


def get_db_boundary_observability() -> dict:
    """Return shadow-mode DB-boundary counters for rollout and debugging."""
    with _AUDIT_LOCK:
        totals = dict(_AUDIT_TOTALS)
        by_root = dict(_AUDIT_BY_ROOT)
        by_helper = dict(_AUDIT_BY_HELPER)
        by_callsite = dict(_AUDIT_BY_CALLSITE)
        recent = list(_AUDIT_RECENT)
    return {
        "enabled": _audit_enabled(),
        "guard_mode": _guard_mode(),
        "totals": totals,
        "by_request_root": by_root,
        "by_helper": by_helper,
        "by_callsite": by_callsite,
        "recent_events": recent,
    }


def reset_db_boundary_observability() -> None:
    """Test helper: clear shadow-mode counters and recent events."""
    with _AUDIT_LOCK:
        _AUDIT_TOTALS.clear()
        _AUDIT_BY_ROOT.clear()
        _AUDIT_BY_HELPER.clear()
        _AUDIT_BY_CALLSITE.clear()
        _AUDIT_RECENT.clear()


def _get_connection() -> sqlite3.Connection:
    """Get persistent connection via storage backend.

    Phase 4.5 shim — delegates to get_backend().get_connection().
    All initialization (pragmas, DDL, migrations) handled by backend.
    """
    _record_boundary_hit("_get_connection")
    return get_backend().get_connection()


def get_db_connection() -> sqlite3.Connection:
    """Public wrapper for embedding engine and other direct callers."""
    _record_boundary_hit("get_db_connection")
    return get_backend().get_connection()


def open_owned_connection() -> sqlite3.Connection:
    """Open a fresh uncached connection for async/thread owners."""
    return get_backend().open_owned_connection()


def diagnostic_readonly_snapshots_enabled() -> bool:
    raw = os.environ.get("PITH_DIAGNOSTIC_READONLY_SNAPSHOTS_ENABLED", "1")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _diagnostic_readonly_busy_timeout_ms() -> int:
    raw = os.environ.get("PITH_DIAGNOSTIC_READONLY_BUSY_TIMEOUT_MS", "500")
    try:
        return max(0, min(int(raw), 5000))
    except (TypeError, ValueError):
        return 500


@contextmanager
def diagnostic_snapshot_db(label: str, *, busy_timeout_ms: int | None = None):
    """Yield a true read-only SQLite snapshot for diagnostic request paths.

    Unlike read_snapshot_db(), this does not use the storage backend's owned
    connection factory and does not execute PRAGMA journal_mode=WAL on open.
    Use this only for bounded diagnostics that can degrade on read contention.
    """
    started_at = time.perf_counter()
    backend = get_backend()
    if backend.backend_type != "sqlite":
        raise RuntimeError("diagnostic_snapshot_db is only available for sqlite backend")
    timeout_ms = _diagnostic_readonly_busy_timeout_ms() if busy_timeout_ms is None else max(0, int(busy_timeout_ms))
    conn = sqlite3.connect(
        f"file:{backend.db_path}?mode=ro",
        uri=True,
        check_same_thread=False,
        isolation_level=None,
    )
    conn.row_factory = sqlite3.Row
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace") if isinstance(b, bytes) else b
    try:
        conn.execute(f"PRAGMA busy_timeout={timeout_ms}")
        conn.execute("PRAGMA query_only = 1")
        conn.execute("BEGIN")
        yield conn
    finally:
        try:
            if getattr(conn, "in_transaction", False):
                conn.rollback()
        finally:
            conn.close()
            _record_managed_entry("diagnostic_snapshot_db", (time.perf_counter() - started_at) * 1000)


@contextmanager
def diagnostic_read_db(label: str):
    """Read context for fast diagnostics with a feature-flag rollback path."""
    if diagnostic_readonly_snapshots_enabled():
        with diagnostic_snapshot_db(label) as conn:
            yield conn
    else:
        with read_snapshot_db(label) as conn:
            yield conn


def required_context_readonly_snapshots_enabled() -> bool:
    raw = os.environ.get("PITH_REQUIRED_CONTEXT_READONLY_SNAPSHOTS_ENABLED", "1")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _required_context_readonly_busy_timeout_ms() -> int:
    raw = os.environ.get("PITH_REQUIRED_CONTEXT_READONLY_BUSY_TIMEOUT_MS", "500")
    try:
        return max(0, min(int(raw), 5000))
    except (TypeError, ValueError):
        return 500


@contextmanager
def required_context_read_db(label: str):
    """Read boundary for required conversation-turn context.

    Required context is user-facing but read-only. Keep it off the backend-owned
    snapshot path by default so refreshes do not execute connection PRAGMAs on
    every owned handle under write pressure. The flag provides a direct rollback
    to the managed snapshot path.
    """
    if required_context_readonly_snapshots_enabled():
        with diagnostic_snapshot_db(
            label,
            busy_timeout_ms=_required_context_readonly_busy_timeout_ms(),
        ) as conn:
            yield conn
    else:
        with read_snapshot_db(label) as conn:
            yield conn


@contextmanager
def owned_connection():
    """Yield a fresh connection owned and closed by the local caller."""
    conn = open_owned_connection()
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def _db(*, timeout_s: float = 30.0, operation: str = "db"):
    """Transaction context manager — delegates to backend.db()."""
    started_at = time.perf_counter()
    try:
        with get_backend().db(timeout_s=timeout_s, operation=operation) as conn:
            yield conn
    finally:
        _record_managed_entry("db", (time.perf_counter() - started_at) * 1000)


_ORIGINAL_DB_CONTEXT = _db


@contextmanager
def managed_write_db(*, timeout_s: float = 30.0, operation: str = "db"):
    """Public managed write boundary for non-storage modules."""
    with _db(timeout_s=timeout_s, operation=operation) as conn:
        yield conn


@contextmanager
def read_snapshot_db(label: str, *, allow_fallback: bool = True):
    """Yield a read-only WAL snapshot for verified pure-read request paths.

    Falls back to _db() when no backend is initialized (test environment).
    This ensures tests that only patch _db() still work after PERF-080+
    migrated reads from _db() to read_snapshot_db().
    """
    started_at = time.perf_counter()
    if allow_fallback and _db is not _ORIGINAL_DB_CONTEXT:
        with _db() as fallback_conn:
            yield fallback_conn
        return
    try:
        conn = open_owned_connection()
    except Exception:
        if not allow_fallback:
            raise
        # No backend initialized (test environment) — fall back to _db()
        with _db() as fallback_conn:
            yield fallback_conn
        return
    try:
        conn.execute("PRAGMA query_only = 1")
        conn.execute("BEGIN")
        yield conn
    finally:
        try:
            if getattr(conn, "in_transaction", False):
                conn.rollback()
        finally:
            conn.close()
            _record_managed_entry("read_snapshot", (time.perf_counter() - started_at) * 1000)


@contextmanager
def _db_immediate(*, timeout_s: float = 30.0, operation: str = "db_immediate"):
    """Serialized write transaction — delegates to backend.db_immediate()."""
    started_at = time.perf_counter()
    try:
        with get_backend().db_immediate(timeout_s=timeout_s, operation=operation) as conn:
            yield conn
    finally:
        _record_managed_entry("db_immediate", (time.perf_counter() - started_at) * 1000)


def db_immediate(*, timeout_s: float = 30.0, operation: str = "db_immediate"):
    """Public access to BEGIN IMMEDIATE transaction context manager."""
    return _db_immediate(timeout_s=timeout_s, operation=operation)


# Module-level compat: keep access_tracker as a no-op shim.
# Direct DB writes happen inside load_concept() now.


class _AccessTrackerShim:
    """Compatibility shim — access tracking is now direct DB writes."""

    def record_access(self, concept_id: str) -> None:
        pass  # No-op: access tracked directly in load_concept

    def flush(self) -> int:
        return 0  # No pending writes — all immediate

    @property
    def pending_count(self) -> int:
        return 0


access_tracker = _AccessTrackerShim()

_KA_SENTINELS = {None, "", "general", "unclassified", "unknown"}
