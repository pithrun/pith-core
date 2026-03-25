"""Storage Backend Abstraction — Phase 4.5 WS1.

Defines StorageBackend protocol and SQLiteBackend implementation.
All subsystems code against _db()/_get_connection() shims in storage.py,
which delegate to get_backend() here. Zero call-site changes required.

Usage:
    from app.storage_backend import get_backend

    backend = get_backend()
    with backend.db() as conn:
        conn.execute("SELECT ...")

Environment:
    PITH_STORAGE_BACKEND: "sqlite" (default) or "postgresql"
    PITH_PG_DSN: PostgreSQL connection string (required if backend=postgresql)
"""

import logging
import os
import sqlite3
import threading
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


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
    def db(self) -> Generator[Any, None, None]:
        """Transaction context manager. Yields a connection-like object.
        Commits on clean exit, rolls back on exception."""
        ...

    @contextmanager
    def db_immediate(self) -> Generator[Any, None, None]:
        """Serialized write transaction.
        SQLite: BEGIN IMMEDIATE. PostgreSQL: SERIALIZABLE."""
        ...

    def get_connection(self) -> Any:
        """Raw connection access (escape hatch for bulk operations).
        Prefer db() context manager for normal use."""
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

    def __init__(self, db_path: Path, schema_ddl: str = ""):
        self._db_path = db_path
        self._schema_ddl = schema_ddl
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()  # RLock: allows nested _db() calls on same thread
        self._nesting_depth = 0  # Track transaction nesting depth (same thread via RLock)

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
        # Create all tables + indexes (idempotent)
        if self._schema_ddl:
            conn.executescript(self._schema_ddl)
            logger.info("Schema DDL executed (CREATE TABLE IF NOT EXISTS)")

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

        # Run all idempotent column migrations
        self._run_column_migrations(conn)
        # Run governance framework migrations
        self._run_governance_migrations(conn)

    def _run_column_migrations(self, conn: sqlite3.Connection) -> None:
        """Run all idempotent ALTER TABLE migrations."""
        migrations = [
            # B5: session learning timestamp
            ("sessions", "last_learning_at", "TEXT"),
            # Self-awareness: session performance counters
            ("sessions", "concepts_created", "INTEGER DEFAULT 0"),
            ("sessions", "concepts_evolved", "INTEGER DEFAULT 0"),
            # P0.3: embedding columns for concepts
            ("concepts", "embedding", "BLOB"),
            ("concepts", "embedding_version", "INTEGER DEFAULT 0"),
            # P1-1: always-activate flag for pre-flight injection
            ("concepts", "always_activate", "INTEGER DEFAULT 0"),
            # AGENT-001: multi-agent scoping
            ("sessions", "agent_id", "TEXT NOT NULL DEFAULT 'default'"),
            ("concepts", "agent_id", "TEXT NOT NULL DEFAULT 'default'"),
            # DATA-020: Track when summary actually changes (vs updated_at on every touch)
            ("concepts", "content_updated_at", "TEXT DEFAULT NULL"),
            # CONTEXT-001: Checkpoint summary in rolling snapshots
            ("resume_snapshots", "checkpoint_summary", "TEXT DEFAULT '{}'"),
            # RETRIEVAL-029: temporal filter outcome tracking on episodes
            ("episodes", "temporal_filter_outcome", "TEXT DEFAULT ''"),
            # SESSION-009: Dropout recovery — store last response for orphan flush safety net
            ("sessions", "last_previous_response", "TEXT DEFAULT NULL"),
        ]
        for table, column, col_type in migrations:
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
                conn.commit()
                logger.info("Migration: added %s.%s", table, column)
            except sqlite3.OperationalError:
                pass  # Column already exists — expected

    def _run_governance_migrations(self, conn: sqlite3.Connection) -> None:
        """Run governance framework migrations (GOV module)."""
        try:
            from app.migration import run_governance_migrations

            gov_result = run_governance_migrations(conn)
            if gov_result.get("applied"):
                logger.info(
                    "Governance migrations applied: %s",
                    gov_result["applied"],
                )
        except Exception as e:
            logger.warning("Governance migrations failed (non-fatal): %s", e)

    @contextmanager
    def db(self) -> Generator[sqlite3.Connection, None, None]:
        """Transaction context manager on persistent connection.

        Uses threading.RLock to prevent concurrent access from FastAPI's
        anyio threadpool while allowing nested _db() calls on the same thread.
        Tiered deadlock detection: WARNING at 5s, CRITICAL at 30s.
        """
        import time as _time

        _t0 = _time.monotonic()
        acquired = self._lock.acquire(timeout=30.0)
        _elapsed = _time.monotonic() - _t0
        if _elapsed > 5.0 and acquired:
            logger.warning(
                "db() lock contention: acquired after %.1fs (thread=%s)",
                _elapsed,
                threading.current_thread().name,
            )
        if not acquired:
            logger.critical(
                "DEADLOCK DETECTED: db() lock not acquired in 30s (thread=%s)",
                threading.current_thread().name,
            )
            raise RuntimeError("Database deadlock detected — db() lock held for >30s")
        self._nesting_depth += 1
        is_outermost = self._nesting_depth == 1
        try:
            conn = self.get_connection()
            if is_outermost:
                conn.execute("BEGIN")
            try:
                yield conn
                if is_outermost:
                    conn.commit()
            except Exception:
                if is_outermost:
                    conn.rollback()
                raise
        finally:
            self._nesting_depth -= 1
            self._lock.release()

    @contextmanager
    def db_immediate(self) -> Generator[sqlite3.Connection, None, None]:
        """Transaction context manager with BEGIN IMMEDIATE.

        Memory Integrity Spec v1.2, §5.2.2 — prevents version chain
        forking by acquiring SQLite's reserved lock at transaction start.
        Tiered deadlock detection: WARNING at 5s, CRITICAL at 30s.
        """
        import time as _time

        _t0 = _time.monotonic()
        acquired = self._lock.acquire(timeout=30.0)
        _elapsed = _time.monotonic() - _t0
        if _elapsed > 5.0 and acquired:
            logger.warning(
                "db_immediate() lock contention: acquired after %.1fs (thread=%s)",
                _elapsed,
                threading.current_thread().name,
            )
        if not acquired:
            logger.critical(
                "DEADLOCK DETECTED: db_immediate() lock not acquired in 30s (thread=%s)",
                threading.current_thread().name,
            )
            raise RuntimeError("Database deadlock detected — db_immediate() lock held for >30s")
        self._nesting_depth += 1
        is_outermost = self._nesting_depth == 1
        try:
            conn = self.get_connection()
            if is_outermost:
                conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
                if is_outermost:
                    conn.commit()
            except Exception:
                if is_outermost:
                    conn.rollback()
                raise
        finally:
            self._nesting_depth -= 1
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
            self._conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
                isolation_level=None,  # True autocommit — all transactions explicit
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=10000")
            # STABILITY-014 Fix 2b: Re-apply row_factory + text_factory on reconnect.
            # initialize() sets these once on first connection, but if get_connection()
            # recycles a stale handle (health check above), the new connection needs them
            # too — otherwise all queries return plain tuples instead of sqlite3.Row,
            # causing "tuple indices must be integers or slices, not str" on any
            # row["column"] access.
            self._conn.row_factory = sqlite3.Row
            self._conn.text_factory = lambda b: b.decode("utf-8", errors="replace") if isinstance(b, bytes) else b
        return self._conn

    @property
    def backend_type(self) -> str:
        return "sqlite"

    def close(self) -> None:
        """Close persistent connection (for testing teardown)."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None


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
    def db(self) -> Generator[Any, None, None]:
        raise NotImplementedError("PostgreSQL backend not yet implemented")
        yield  # Make it a generator for type checking

    @contextmanager
    def db_immediate(self) -> Generator[Any, None, None]:
        raise NotImplementedError("PostgreSQL backend not yet implemented")
        yield

    def get_connection(self) -> Any:
        raise NotImplementedError("PostgreSQL backend not yet implemented")

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
