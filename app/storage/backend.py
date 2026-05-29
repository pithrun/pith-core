"""Storage Backend Abstraction — Phase 4.5 WS1.

Defines StorageBackend protocol and SQLiteBackend implementation.
All subsystems code against _db()/_get_connection() shims in storage.py,
which delegate to get_backend() here. Zero call-site changes required.

Usage:
    from app.storage.backend import get_backend

    backend = get_backend()
    with backend.db() as conn:
        conn.execute("SELECT ...")

Environment:
    PITH_STORAGE_BACKEND: "sqlite" (default) or "postgresql"
    PITH_PG_DSN: PostgreSQL connection string (required if backend=postgresql)
"""

import faulthandler
import logging
import os
import signal as _signal
import sqlite3
import sys
import threading
import traceback
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Deadlock diagnostic subsystem (ported from observer-fix runtime 2026-04-11)
# ---------------------------------------------------------------------------
_deadlock_diag_logger: logging.Logger | None = None
_deadlock_diag_init_lock = threading.Lock()


def _get_deadlock_diag_logger() -> logging.Logger:
    """Return the deadlock-diagnostic logger, lazily creating it on first use."""
    global _deadlock_diag_logger
    if _deadlock_diag_logger is not None:
        return _deadlock_diag_logger
    with _deadlock_diag_init_lock:
        if _deadlock_diag_logger is not None:
            return _deadlock_diag_logger
        diag = logging.getLogger("pith.deadlock_diagnostic")
        diag.setLevel(logging.CRITICAL)
        diag.propagate = False  # keep the dump out of pith.log
        try:
            from app.core.profile import resolve_data_dir

            log_dir = Path(os.environ.get("PITH_LOG_DIR") or (resolve_data_dir() / "logs"))
            log_dir.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(
                log_dir / "deadlock-diagnostic.log", mode="a"
            )
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(levelname)s - %(message)s"
                )
            )
            diag.addHandler(handler)
        except Exception as _init_err:
            logger.warning(
                "deadlock diagnostic logger init failed: %s", _init_err
            )
        _deadlock_diag_logger = diag
        return diag


def _capture_all_thread_stacks() -> str:
    """Capture stack traces for all live Python threads.

    Pure-Python implementation via sys._current_frames() + traceback.format_stack(),
    avoiding faulthandler.dump_traceback (which requires a real file descriptor).
    Per-frame try/except so one unformattable frame can't poison the whole dump.
    """
    try:
        frames = sys._current_frames()
        threads_by_id = {t.ident: t for t in threading.enumerate()}
        parts = []
        for tid, frame in frames.items():
            tobj = threads_by_id.get(tid)
            tname = tobj.name if tobj else f"<unknown tid={tid}>"
            daemon = getattr(tobj, "daemon", "?")
            try:
                lines = traceback.format_stack(frame)
                parts.append(
                    f"Thread {tname} (daemon={daemon}, tid={tid}):\n"
                    + "".join(lines)
                )
            except Exception as frame_err:
                parts.append(
                    f"Thread {tname} (tid={tid}): "
                    f"<format_stack failed: {frame_err}>"
                )
        return "\n".join(parts)
    except Exception as dump_err:
        return f"<stack dump failed: {dump_err}>"


# ---------------------------------------------------------------------------
# STABILITY-DIAG: faulthandler-based deadlock observer (3 layers)
#   L1  load-time SMOKE_TEST proving module loaded into the running PID
#   L2  dump_traceback_later() dead-man's-switch re-armed every 10s by daemon
#       thread. If blocked >60s, C-level timer dumps all thread stacks.
#   L3  SIGUSR1 handler (faulthandler.register) — external trigger from
#       check_server_health.sh when readiness is degraded.
# Kill switch: PITH_DEADLOCK_WATCHDOG_DISABLED=1
# ---------------------------------------------------------------------------
_faulthandler_installed = False
_faulthandler_file: Any = None
_watchdog_thread: "threading.Thread | None" = None
_WATCHDOG_REARM_INTERVAL_SECS = 10.0
_WATCHDOG_TIMEOUT_SECS = 60
_WATCHDOG_STARTUP_DELAY_SECS = 2.0
_WATCHDOG_CRASH_BACKOFF_SECS = 60.0


def _watchdog_loop() -> None:
    """Re-arm faulthandler dead-man's-switch every _WATCHDOG_REARM_INTERVAL_SECS.

    If this loop is blocked for more than _WATCHDOG_TIMEOUT_SECS, the C-level
    timer fires and dumps all thread stacks to _faulthandler_file.
    Tolerates one crash with a 60s backoff, gives up after a second crash.
    """
    import time as _time

    _time.sleep(_WATCHDOG_STARTUP_DELAY_SECS)
    crash_count = 0
    while True:
        try:
            while True:
                if _faulthandler_file is not None:
                    faulthandler.dump_traceback_later(
                        _WATCHDOG_TIMEOUT_SECS,
                        repeat=False,
                        file=_faulthandler_file,
                        exit=False,
                    )
                _time.sleep(_WATCHDOG_REARM_INTERVAL_SECS)
        except Exception as _watchdog_err:
            crash_count += 1
            logger.error(
                "deadlock watchdog loop crashed (count=%d): %s",
                crash_count,
                _watchdog_err,
                exc_info=True,
            )
            if crash_count >= 2:
                logger.error(
                    "deadlock watchdog loop crashed twice; giving up"
                )
                return
            _time.sleep(_WATCHDOG_CRASH_BACKOFF_SECS)


def _install_faulthandler_hooks() -> None:
    """Install the three-layer faulthandler observer.

    Idempotent. Safe to call at module import time. Installation failures
    are logged and the service continues without the observer.
    """
    global _faulthandler_installed, _faulthandler_file, _watchdog_thread
    if _faulthandler_installed:
        return
    if os.environ.get("PITH_DEADLOCK_WATCHDOG_DISABLED") == "1":
        logger.info(
            "deadlock watchdog disabled via PITH_DEADLOCK_WATCHDOG_DISABLED"
        )
        _faulthandler_installed = True
        return
    try:
        from app.core.profile import resolve_data_dir

        log_dir = Path(os.environ.get("PITH_LOG_DIR") or (resolve_data_dir() / "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        _faulthandler_file = open(  # noqa: SIM115 - long-lived process file
            log_dir / "deadlock-diagnostic-stacks.log",
            "a",
            buffering=1,
        )
        # Layer 3: SIGUSR1 external trigger — CRITICAL for T0-2.
        # Without this, `kill -USR1` from check_server_health.sh terminates
        # the server (SIG_DFL on macOS/Python = process termination).
        faulthandler.register(
            _signal.SIGUSR1,
            file=_faulthandler_file,
            all_threads=True,
            chain=False,
        )
        # Layer 2: daemon watchdog thread
        _watchdog_thread = threading.Thread(
            target=_watchdog_loop,
            name="pith-deadlock-watchdog",
            daemon=True,
        )
        _watchdog_thread.start()
        _faulthandler_installed = True
        # Layer 1: load-time SMOKE_TEST
        import datetime as _dt

        now_iso = _dt.datetime.now().isoformat(timespec="seconds")
        smoke_msg = (
            "SMOKE_TEST: backend.py faulthandler observer loaded "
            f"pid={os.getpid()} "
            "build=T0-2-SIGUSR1-PORT-2026-04-12 "
            f"watchdog_interval={_WATCHDOG_REARM_INTERVAL_SECS}s "
            f"watchdog_timeout={_WATCHDOG_TIMEOUT_SECS}s"
        )
        logger.critical(smoke_msg)
        try:
            _get_deadlock_diag_logger().critical(smoke_msg)
        except Exception:
            pass
        try:
            _faulthandler_file.write(f"{now_iso} SMOKE_TEST {smoke_msg}\n")
            _faulthandler_file.flush()
        except Exception:
            pass
    except Exception as _init_err:
        logger.error(
            "faulthandler observer install failed: %s",
            _init_err,
            exc_info=True,
        )


# Install observer at import time so every process that imports backend.py
# (including uvicorn workers spawned by launchd) is instrumented automatically.
# Skip during tests — the watchdog thread causes test timeout false positives.
if "pytest" not in sys.modules and os.environ.get("PYTEST_CURRENT_TEST") is None:
    _install_faulthandler_hooks()


@runtime_checkable
class StorageBackend(Protocol):
    """Abstract storage backend protocol.

    All subsystems code against this interface via shims in storage.py.
    Implementations: SQLiteBackend (production), PostgreSQLBackend (Phase 5).
    """

    def initialize(self) -> None:
        """Create tables, run migrations, set pragmas."""
        ...

    @contextmanager
    def db(
        self,
        *,
        timeout_s: float = 30.0,
        operation: str = "db",
    ) -> Generator[Any, None, None]:
        """Transaction context manager. Yields a connection-like object.
        Commits on clean exit, rolls back on exception."""
        ...

    @contextmanager
    def db_immediate(
        self,
        *,
        timeout_s: float = 30.0,
        operation: str = "db_immediate",
    ) -> Generator[Any, None, None]:
        """Serialized write transaction.
        SQLite: BEGIN IMMEDIATE. PostgreSQL: SERIALIZABLE."""
        ...

    def get_connection(self) -> Any:
        """Raw connection access (escape hatch for bulk operations).
        Prefer db() context manager for normal use."""
        ...

    def open_owned_connection(self) -> Any:
        """Fresh uncached connection for async/thread ownership."""
        ...

    @property
    def db_path(self) -> Path:
        """Filesystem path for SQLite diagnostics."""
        ...

    @property
    def backend_type(self) -> str:
        """Returns 'sqlite' or 'postgresql'."""
        ...


class SQLiteBackend:
    """SQLite storage backend — wraps current storage.py internals.

    Thread-safe via _lock (RLock — reentrant for nested _db() calls). Single persistent connection with DELETE
    journal mode (historically required for Docker VirtioFS bind mounts — B7 fix; retained for safety).
    """

    # MONITOR-CI032A-01: Single source of truth for inline ALTER migrations.
    # Used by _run_column_migrations (apply path) AND _check_migration_integrity
    # (audit path). Keeping them tied to one constant guarantees they can't drift.
    # Append new migrations here; _run_column_migrations and the audit pick them
    # up automatically.
    _COLUMN_MIGRATIONS: tuple[tuple[str, str, str], ...] = (
        # B5: session learning timestamp
        ("sessions", "last_learning_at", "TEXT"),
        # Self-awareness: session performance counters
        ("sessions", "concepts_created", "INTEGER DEFAULT 0"),
        ("sessions", "concepts_evolved", "INTEGER DEFAULT 0"),
        # Legacy benchmark DBs may have sessions without status; DDL indexes
        # sessions(status) before governance migrations run.
        ("sessions", "status", "TEXT NOT NULL DEFAULT 'active'"),
        ("sessions", "learning_event_count", "INTEGER DEFAULT 0"),
        ("sessions", "data", "JSON"),
        ("sessions", "last_heartbeat", "TEXT DEFAULT NULL"),
        ("metadata", "updated_at", "TEXT DEFAULT ''"),
        ("associations", "relation", "TEXT NOT NULL DEFAULT 'related'"),
        ("associations", "mechanism", "TEXT DEFAULT NULL"),
        ("associations", "direction", "TEXT DEFAULT 'bidirectional'"),
        ("associations", "chain_id", "TEXT DEFAULT NULL"),
        # P0.3: embedding columns for concepts
        ("concepts", "embedding", "BLOB"),
        ("concepts", "embedding_version", "INTEGER DEFAULT 0"),
        # P1-1: always-activate flag for pre-flight injection
        ("concepts", "always_activate", "INTEGER DEFAULT 0"),
        # AGENT-001: multi-agent scoping
        ("sessions", "agent_id", "TEXT NOT NULL DEFAULT 'default'"),
        ("concepts", "agent_id", "TEXT NOT NULL DEFAULT 'default'"),
        ("concepts", "session_id", "TEXT DEFAULT NULL"),
        # DATA-020: Track when summary actually changes (vs updated_at on every touch)
        ("concepts", "content_updated_at", "TEXT DEFAULT NULL"),
        # CONTEXT-001: Checkpoint summary in rolling snapshots
        ("resume_snapshots", "checkpoint_summary", "TEXT DEFAULT '{}'"),
        # RETRIEVAL-029: temporal filter outcome tracking on episodes
        ("episodes", "temporal_filter_outcome", "TEXT DEFAULT ''"),
        # SESSION-009: Dropout recovery — store last response for orphan flush safety net
        ("sessions", "last_previous_response", "TEXT DEFAULT NULL"),
        # RETRIEVAL-080: Utility accumulator — feedback loop closes retrieval→learning gap
        ("concepts", "utility_score", "REAL DEFAULT 0.5"),
        ("concepts", "utility_samples", "INTEGER DEFAULT 0"),
        ("concepts", "utility_updated", "TEXT DEFAULT NULL"),
        # SKILL-DEPLOY-001: Synthesis watermark — tracks which concepts have been evaluated
        ("concepts", "last_synthesis_evaluated_at", "TEXT DEFAULT NULL"),
        # SESSION-012: Cross-session awareness — topic keywords for peer hints
        ("resume_snapshots", "topic_keywords", "TEXT DEFAULT ''"),
        # SESSION-012 v0.3: Platform hint for session provenance
        ("sessions", "platform_hint", "TEXT NOT NULL DEFAULT 'unknown'"),
        # SESSION-014: stable client/thread origin for checkpoint authority
        ("checkpoints", "origin_id", "TEXT DEFAULT NULL"),
        # SESSION-015: stable client/thread origin for session closeout binding
        ("sessions", "origin_id", "TEXT DEFAULT NULL"),
        # SESSION-LEARN-DURABILITY: recoverable write replay queue metadata
        ("write_request_replays", "request_json", "TEXT DEFAULT NULL"),
        ("write_request_replays", "attempt_count", "INTEGER DEFAULT 0"),
        ("write_request_replays", "last_error", "TEXT DEFAULT NULL"),
        ("write_request_replays", "lease_owner", "TEXT DEFAULT NULL"),
        ("write_request_replays", "lease_expires_at", "TEXT DEFAULT NULL"),
        ("write_request_replays", "next_retry_at", "TEXT DEFAULT NULL"),
    )

    def __init__(self, db_path: Path, schema_ddl: str = ""):
        self._db_path = db_path
        self._schema_ddl = schema_ddl
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()  # RLock: allows nested _db() calls on same thread
        self._nesting_depth = 0  # Track transaction nesting depth (same thread via RLock)
        self._shutting_down = False  # STABILITY-037: shutdown coordination flag

    def initialize(self) -> None:
        """Create tables, set pragmas, run migrations on first connection.

        Mirrors the full initialization sequence from the original
        _get_connection() in storage.py — schema DDL, all ALTER TABLE
        migrations (B5, self-awareness, P0.3, P1-1, GOV), row_factory,
        and text_factory for UTF-8 error handling.
        """
        conn = self.get_connection()
        # row_factory for dict-like access (used by server.py dashboard etc.)
        conn.row_factory = sqlite3.Row
        # Bug 7b fix: handle corrupted UTF-8 in data columns
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace") if isinstance(b, bytes) else b
        # TIER4-002: Run column migrations BEFORE DDL indexes.
        # Existing DBs may lack columns (e.g. utility_score) that DDL indexes reference.
        # CREATE INDEX on a missing column is a hard error, so add the columns first.
        self._run_column_migrations(conn)
        self._run_legacy_compat_backfills(conn)

        # Create all tables + indexes (idempotent)
        if self._schema_ddl:
            conn.executescript(self._schema_ddl)
            logger.info("Schema DDL executed (CREATE TABLE IF NOT EXISTS)")

            # CI-032a: Second migration pass — now that DDL has created the
            # tables on a fresh DB, ALTER-added columns from the inline
            # migrations list actually land. On a fresh DB the first pass saw
            # "no such table" for every ALTER and silently skipped (the guard
            # at _run_column_migrations's inner try/except is intentional and
            # required by TIER4-002 multi-pass sequencing). This second pass
            # is idempotent for any column already present in SCHEMA_DDL via
            # the _column_exists guard at the top of _run_column_migrations.
            self._run_column_migrations(conn)
            logger.info("Column migrations re-run post-DDL (CI-032a)")

        # RETRIEVAL-042 upgrade: Ensure FTS5 table exists and populate on first run
        try:
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS fts_concepts
                USING fts5(concept_id UNINDEXED, summary, tokenize='porter ascii')
            """)
            _fts_count = conn.execute("SELECT COUNT(*) FROM fts_concepts").fetchone()[0]
            if _fts_count == 0:
                _concept_count = conn.execute(
                    "SELECT COUNT(*) FROM concepts WHERE status = 'active' AND is_current = 1"
                ).fetchone()[0]
                if _concept_count > 0:
                    conn.execute("""
                        INSERT INTO fts_concepts(concept_id, summary)
                        SELECT id, summary FROM concepts
                        WHERE status = 'active' AND is_current = 1
                    """)
                    conn.commit()
                    logger.info(f"FTS5: Populated {_concept_count} concepts into full-text index")
        except Exception as _fts_init_err:
            logger.warning(f"FTS5 init failed (non-fatal): {_fts_init_err}")

        # Run governance framework migrations (column migrations already ran above)
        self._run_governance_migrations(conn)

        # MONITOR-CI032A-01: Post-init audit that every _COLUMN_MIGRATIONS
        # column actually landed. Non-fatal by design — logs + metric only,
        # never raises. Surfaces via /pith_stats.migration_integrity so
        # silent-skip regressions (the CI-032a class of bug) can be alerted
        # on instead of hiding for weeks.
        try:
            self._check_migration_integrity(conn)
        except Exception as _integ_err:
            logger.warning(
                "MONITOR-CI032A-01: integrity check crashed (non-fatal): %s",
                _integ_err,
            )

    def _run_column_migrations(self, conn: sqlite3.Connection) -> None:
        """Run all idempotent ALTER TABLE migrations.

        MONITOR-CI032A-01: Iterates self._COLUMN_MIGRATIONS (class-level
        constant, single source of truth). The same constant is used by
        _check_migration_integrity() to audit post-init column presence.
        """
        def _column_exists(table: str, column: str) -> bool:
            return any(row[1] == column for row in conn.execute(f"PRAGMA table_info({table})"))

        for table, column, col_type in self._COLUMN_MIGRATIONS:
            if _column_exists(table, column):
                continue
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
                conn.commit()
                logger.info("Migration: added %s.%s", table, column)
            except (sqlite3.OperationalError, sqlite3.DatabaseError, SystemError) as e:
                # Multi-worker startup can race here: one worker adds the column
                # while another is still checking. Treat duplicate-column failures
                # as idempotent success, but re-raise anything else.
                _err = str(e).lower()
                if "duplicate column name" in _err and _column_exists(table, column):
                    logger.info("Migration race resolved: %s.%s already exists", table, column)
                    continue
                if "no such table" in _err:
                    # TIER4-002: Fresh DB — table doesn't exist yet.
                    # DDL executescript will create it with all columns. Skip gracefully.
                    continue
                raise

    def _run_legacy_compat_backfills(self, conn: sqlite3.Connection) -> None:
        """Backfill renamed legacy columns after early compatibility ALTERs."""
        def _column_exists(table: str, column: str) -> bool:
            return any(
                row[1] == column
                for row in conn.execute(f"PRAGMA table_info({table})")
            )

        if (
            _column_exists("associations", "association_type")
            and _column_exists("associations", "relation")
        ):
            conn.execute(
                """
                UPDATE associations
                   SET relation = COALESCE(NULLIF(association_type, ''), relation)
                 WHERE association_type IS NOT NULL
                   AND association_type != ''
                   AND (relation IS NULL OR relation = 'related')
                """
            )
            conn.commit()

    @classmethod
    def _audit_migration_columns(cls, conn: sqlite3.Connection) -> dict:
        """MONITOR-CI032A-02: Pure column-presence audit. No side effects.

        Iterates _COLUMN_MIGRATIONS, runs PRAGMA table_info for each unique
        table, and classifies every declared column as present or missing.
        On sqlite3.OperationalError for PRAGMA (table dropped, locked, etc.)
        the affected table's columns are all classified as missing.

        Returns a dict with keys: status, expected_count, present_count,
        missing (list of {table, column}), checked_at.

        This is the pure-function core of migration integrity verification.
        Declared as @classmethod because it only reads the class-level
        _COLUMN_MIGRATIONS constant — no instance state is needed. External
        callers (e.g. scripts/monitoring/run_monitor.py) can call this
        directly as SQLiteBackend._audit_migration_columns(conn) without
        constructing a backend instance, avoiding any risk of triggering
        metric buffering or logger side effects that are not safe in
        short-lived processes.
        """
        from app.core.datetime_utils import _utc_now_iso

        expected = len(cls._COLUMN_MIGRATIONS)
        tables_seen: set[str] = set()
        for _table, _col, _type in cls._COLUMN_MIGRATIONS:
            tables_seen.add(_table)

        table_columns: dict[str, set[str]] = {}
        for table in tables_seen:
            try:
                rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
                table_columns[table] = {row[1] for row in rows}
            except sqlite3.OperationalError:
                table_columns[table] = set()

        missing: list[dict] = []
        present = 0
        for table, column, _col_type in cls._COLUMN_MIGRATIONS:
            if column in table_columns.get(table, set()):
                present += 1
            else:
                missing.append({"table": table, "column": column})

        status = "HEALTHY" if not missing else "CRITICAL"
        return {
            "status": status,
            "expected_count": expected,
            "present_count": present,
            "missing": missing,
            "checked_at": _utc_now_iso(),
        }

    def _check_migration_integrity(self, conn: sqlite3.Connection) -> dict:
        """MONITOR-CI032A-01: Audit + emit metric + log.

        Wraps _audit_migration_columns with the side-effectful behavior that
        was previously inline: records a metric via the buffered metrics
        collector and logs HEALTHY/CRITICAL at the appropriate level. Public
        API unchanged for in-process callers (server init, stats aggregator).

        Non-fatal: metric emit and log writes never propagate exceptions so
        a stale schema can never brick startup. External callers that do NOT
        want buffered-metric side effects should use _audit_migration_columns
        directly (it is a @classmethod — no instance needed).
        """
        result = self._audit_migration_columns(conn)
        status = result["status"]
        expected = result["expected_count"]
        present = result["present_count"]
        missing = result["missing"]

        try:
            # MONITOR-CI032A-02 + Contract-2 compliance: route through
            # app.core.metrics_facade (the sanctioned single crossing point
            # for storage → ops metrics access) instead of importing
            # app.ops.metrics directly. See .importlinter Contract 2 and
            # DEBT-244 for rationale.
            from app.core.metrics_facade import metrics
            metrics.record(
                "migration_integrity_check",
                float(len(missing)),
                {
                    "status": status,
                    "expected": expected,
                    "present": present,
                    "backend": self.backend_type,
                },
            )
        except Exception as metric_err:
            logger.warning(
                "MONITOR-CI032A-01: metric record failed: %s", metric_err
            )

        if status == "CRITICAL":
            logger.error(
                "MONITOR-CI032A-01: migration integrity CRITICAL — %d/%d columns present, missing: %s",
                present,
                expected,
                ", ".join(f"{m['table']}.{m['column']}" for m in missing),
            )
        else:
            logger.info(
                "MONITOR-CI032A-01: migration integrity HEALTHY — %d/%d columns present",
                present,
                expected,
            )

        return result

    def _run_governance_migrations(self, conn: sqlite3.Connection) -> None:
        """Run governance framework migrations (GOV module)."""
        try:
            from app.storage.migration import run_governance_migrations

            gov_result = run_governance_migrations(conn)
            if gov_result.get("applied"):
                logger.info(
                    "Governance migrations applied: %s",
                    gov_result["applied"],
                )
        except Exception as e:
            logger.warning("Governance migrations failed (non-fatal): %s", e)

    @contextmanager
    def db(
        self,
        *,
        timeout_s: float = 30.0,
        operation: str = "db",
    ) -> Generator[sqlite3.Connection, None, None]:
        """Transaction context manager on persistent connection.

        Uses threading.RLock to prevent concurrent access from FastAPI's
        anyio threadpool while allowing nested _db() calls on the same thread.
        Tiered deadlock detection: WARNING at 5s, CRITICAL at 30s.
        T1-1: Lock holder tracking for contention diagnostics.
        """
        import time as _time
        import traceback as _tb

        _caller = ""
        try:
            frame = _tb.extract_stack(limit=3)
            if len(frame) >= 2:
                f = frame[-2]
                _caller = f"{f.filename.rsplit('/', 1)[-1]}:{f.lineno}:{f.name}"
        except Exception:
            pass

        _operation = operation or "db"
        try:
            _timeout_s = max(0.0, float(timeout_s))
        except (TypeError, ValueError):
            _timeout_s = 30.0

        _t0 = _time.monotonic()
        acquired = self._lock.acquire(timeout=_timeout_s)
        _elapsed = _time.monotonic() - _t0
        if acquired:
            self._lock_holder = f"{threading.current_thread().name}@{_caller}[{_operation}]"
            self._lock_acquired_at = _t0
        if _elapsed > 5.0 and acquired:
            logger.warning(
                "db() lock contention: acquired after %.1fs (thread=%s, "
                "caller=%s, operation=%s, prev_holder=%s)",
                _elapsed,
                threading.current_thread().name,
                _caller,
                _operation,
                getattr(self, "_prev_lock_holder", "unknown"),
            )
        if not acquired:
            _now = _time.monotonic()
            _holder_at = getattr(self, "_lock_acquired_at", None)
            _holder_age = (_now - _holder_at) if _holder_at is not None else -1.0
            logger.critical(
                "DEADLOCK DETECTED: db() lock not acquired in %.3fs "
                "(thread=%s, caller=%s, operation=%s, holder=%s, "
                "holder_age=%.3fs, prev_holder=%s, nesting_depth=%s)",
                _timeout_s,
                threading.current_thread().name,
                _caller,
                _operation,
                getattr(self, "_lock_holder", "unknown"),
                _holder_age,
                getattr(self, "_prev_lock_holder", "unknown"),
                self._nesting_depth,
            )
            raise RuntimeError(f"Database deadlock detected — db() lock held for >{_timeout_s:.3f}s")
        self._nesting_depth += 1
        is_outermost = self._nesting_depth == 1
        try:
            conn = self.get_connection()
            if is_outermost:
                conn.execute("BEGIN")
            try:
                yield conn
                if is_outermost:
                    if getattr(conn, "in_transaction", False):
                        conn.commit()
                    else:
                        logger.debug(
                            "db(): outer transaction already closed before context exit; skipping commit"
                        )
            except Exception:
                if is_outermost:
                    if getattr(conn, "in_transaction", False):
                        conn.rollback()
                raise
        finally:
            self._nesting_depth -= 1
            if self._nesting_depth == 0:
                import time as _time2
                _held = _time2.monotonic() - getattr(self, "_lock_acquired_at", _time2.monotonic())
                self._prev_lock_holder = f"{getattr(self, '_lock_holder', '?')} held={_held:.1f}s"
                if _held > 5.0:
                    logger.warning(
                        "db() lock held %.1fs by %s (operation=%s)",
                        _held, self._lock_holder, _operation,
                    )
            self._lock.release()

    @contextmanager
    def db_immediate(
        self,
        *,
        timeout_s: float = 30.0,
        operation: str = "db_immediate",
    ) -> Generator[sqlite3.Connection, None, None]:
        """Transaction context manager with BEGIN IMMEDIATE.

        Memory Integrity Spec v1.2, §5.2.2 — prevents version chain
        forking by acquiring SQLite's reserved lock at transaction start.
        Tiered deadlock detection: WARNING at 5s, CRITICAL at 30s.
        T1-1: Lock holder tracking for contention diagnostics.
        """
        import time as _time
        import traceback as _tb

        _caller = ""
        try:
            frame = _tb.extract_stack(limit=3)
            if len(frame) >= 2:
                f = frame[-2]
                _caller = f"{f.filename.rsplit('/', 1)[-1]}:{f.lineno}:{f.name}"
        except Exception:
            pass

        _operation = operation or "db_immediate"
        try:
            _timeout_s = max(0.0, float(timeout_s))
        except (TypeError, ValueError):
            _timeout_s = 30.0

        _t0 = _time.monotonic()
        acquired = self._lock.acquire(timeout=_timeout_s)
        _elapsed = _time.monotonic() - _t0
        if acquired:
            self._lock_holder = f"{threading.current_thread().name}@{_caller}[{_operation}]"
            self._lock_acquired_at = _t0
        if _elapsed > 5.0 and acquired:
            logger.warning(
                "db_immediate() lock contention: acquired after %.1fs (thread=%s, "
                "caller=%s, operation=%s, prev_holder=%s)",
                _elapsed,
                threading.current_thread().name,
                _caller,
                _operation,
                getattr(self, "_prev_lock_holder", "unknown"),
            )
        if not acquired:
            _now = _time.monotonic()
            _holder_at = getattr(self, "_lock_acquired_at", None)
            _holder_age = (_now - _holder_at) if _holder_at is not None else -1.0
            logger.critical(
                "DEADLOCK DETECTED: db_immediate() lock not acquired in %.3fs "
                "(thread=%s, caller=%s, operation=%s, holder=%s, "
                "holder_age=%.3fs, prev_holder=%s, nesting_depth=%s)",
                _timeout_s,
                threading.current_thread().name,
                _caller,
                _operation,
                getattr(self, "_lock_holder", "unknown"),
                _holder_age,
                getattr(self, "_prev_lock_holder", "unknown"),
                self._nesting_depth,
            )
            raise RuntimeError(f"Database deadlock detected — db_immediate() lock held for >{_timeout_s:.3f}s")
        self._nesting_depth += 1
        is_outermost = self._nesting_depth == 1
        try:
            conn = self.get_connection()
            if is_outermost:
                conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
                if is_outermost:
                    if getattr(conn, "in_transaction", False):
                        conn.commit()
                    else:
                        logger.debug(
                            "db_immediate(): outer transaction already closed before context exit; skipping commit"
                        )
            except Exception:
                if is_outermost:
                    if getattr(conn, "in_transaction", False):
                        conn.rollback()
                raise
        finally:
            self._nesting_depth -= 1
            if self._nesting_depth == 0:
                import time as _time2
                _held = _time2.monotonic() - getattr(self, "_lock_acquired_at", _time2.monotonic())
                self._prev_lock_holder = f"{getattr(self, '_lock_holder', '?')} held={_held:.1f}s"
                if _held > 5.0:
                    logger.warning(
                        "db_immediate() lock held %.1fs by %s (operation=%s)",
                        _held, self._lock_holder, _operation,
                    )
            self._lock.release()

    def get_connection(self) -> sqlite3.Connection:
        """Get persistent SQLite connection with WAL journal mode.

        WAL mode allows concurrent readers + one writer without blocking.
        Previously used DELETE mode for Docker VirtioFS compatibility,
        but native macOS execution supports WAL fully.
        synchronous=NORMAL is safe with WAL (WAL provides its own durability).
        busy_timeout=10000 gives writers 10s to acquire the lock before failing.

        STABILITY-014 Fix 2: Health check on cached connection to catch stale
        handles after unclean shutdown (e.g., SIGKILL from launchctl kickstart -k).
        """
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.execute("SELECT 1")
                except (sqlite3.ProgrammingError, sqlite3.OperationalError):
                    logger.warning("STABILITY-014: Stale DB connection detected — recycling")
                    try:
                        self._conn.close()
                    except Exception:
                        pass
                    self._conn = None
            if self._conn is None:
                self._conn = self._create_connection()
            return self._conn

    def open_owned_connection(self) -> sqlite3.Connection:
        """Open a fresh uncached SQLite handle for async/thread owners."""
        return self._create_connection()

    def _create_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            isolation_level=None,  # True autocommit — all transactions explicit
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=10000")
        # STABILITY-014 Fix 2b: Re-apply row_factory + text_factory on reconnect
        # and on any fresh owned handle so call sites get consistent sqlite3.Row
        # behavior plus UTF-8 replacement semantics.
        conn.row_factory = sqlite3.Row
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace") if isinstance(b, bytes) else b
        return conn

    @property
    def db_path(self) -> Path:
        return self._db_path

    @property
    def backend_type(self) -> str:
        return "sqlite"

    def close(self) -> None:
        """Close persistent connection (for testing teardown)."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def begin_shutdown(self) -> None:
        """STABILITY-037: Signal that shutdown is in progress. Background tasks should stop."""
        self._shutting_down = True
        logger.info("STABILITY-037: Backend shutdown flag set")

    @property
    def is_shutting_down(self) -> bool:
        return self._shutting_down


class PostgreSQLBackend:
    """PostgreSQL storage backend — Phase 5 stub.

    Implements StorageBackend protocol but raises NotImplementedError
    for all methods. Full implementation in Phase 5 with psycopg3.
    """

    def __init__(self, dsn: str):
        self._dsn = dsn

    def initialize(self) -> None:
        raise NotImplementedError("PostgreSQL backend is Phase 5 scope. Set PITH_STORAGE_BACKEND=sqlite (default).")

    @contextmanager
    def db(
        self,
        *,
        timeout_s: float = 30.0,
        operation: str = "db",
    ) -> Generator[Any, None, None]:
        raise NotImplementedError("PostgreSQL backend not yet implemented")
        yield  # Make it a generator for type checking

    @contextmanager
    def db_immediate(
        self,
        *,
        timeout_s: float = 30.0,
        operation: str = "db_immediate",
    ) -> Generator[Any, None, None]:
        raise NotImplementedError("PostgreSQL backend not yet implemented")
        yield

    def get_connection(self) -> Any:
        raise NotImplementedError("PostgreSQL backend not yet implemented")

    def open_owned_connection(self) -> Any:
        raise NotImplementedError("PostgreSQL backend not yet implemented")

    @property
    def db_path(self) -> Path:
        raise NotImplementedError("PostgreSQL backend does not expose a SQLite db_path")

    @property
    def backend_type(self) -> str:
        return "postgresql"


# --- Backend Factory ---

_backend: StorageBackend | None = None
_backend_lock = threading.Lock()  # STABILITY-014 Fix 1: thread-safe singleton init


def get_backend() -> StorageBackend:
    """Get or create the storage backend singleton.

    Backend type determined by PITH_STORAGE_BACKEND env var:
        "sqlite"     — SQLiteBackend (default)
        "postgresql" — PostgreSQLBackend (Phase 5 stub)

    The backend is lazy-initialized on first call and cached.
    Thread-safe via double-checked locking (STABILITY-014).
    """
    global _backend
    if _backend is not None:
        return _backend
    with _backend_lock:
        if _backend is not None:  # double-check after acquiring lock
            return _backend
        backend_type = os.environ.get("PITH_STORAGE_BACKEND", "sqlite")
        if backend_type == "postgresql":
            dsn = os.environ.get("PITH_PG_DSN", "")
            if not dsn:
                raise ValueError("PITH_PG_DSN environment variable required when PITH_STORAGE_BACKEND=postgresql")
            _backend = PostgreSQLBackend(dsn)
        else:
            # Import here to avoid circular dependency with storage.py
            from app.storage import DB_PATH, SCHEMA_DDL

            _backend = SQLiteBackend(DB_PATH, SCHEMA_DDL)
        _backend.initialize()
        logger.info("Storage backend initialized: %s", _backend.backend_type)
    return _backend


def reset_backend() -> None:
    """Reset backend singleton (for testing)."""
    global _backend
    if _backend is not None and hasattr(_backend, "close"):
        _backend.close()
    _backend = None
