"""Batch utility functions for out-of-server database operations.

Provides safe connection factory, lock checking, and WAL management
for batch scripts that operate outside the uvicorn server process.

These utilities mirror the journal mode and safety pragmas of
SQLiteBackend.get_connection() but create short-lived connections suitable
for batch iteration patterns.

For scripts in scripts/ directory, use scripts/lib/db_connect.py instead.
This module is for app/-level batch operations (e.g., run_ka_reclassification).

Created: 2026-03-09 (STABILITY-007)
Motivation: KA-003 DB corruption incident — batch scripts rolled their own
connections without WAL pragmas or lock checks.
"""

import logging
import os
import sqlite3
import subprocess
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# DEBT-116: Lazy DB path resolution — evaluated at call time, not import time.
# If PITH_PROFILE env var changes after import, functions pick up the new value.
def _resolve_default_db_path() -> str:
    """Resolve default DB path from current PITH_PROFILE env var."""
    profile = os.environ.get("PITH_PROFILE", "default")
    db_dir = os.path.expanduser(f"~/pith-data/{profile}")
    return os.path.join(db_dir, "pith.db")


def get_batch_connection(
    db_path: str | None = None, busy_timeout_ms: int = 30000
) -> sqlite3.Connection:
    """Create a safe SQLite connection for batch operations.

    Sets WAL journal mode, busy_timeout, synchronous=NORMAL, and
    isolation_level=None (autocommit) to match the pragma discipline
    of SQLiteBackend.get_connection().

    Unlike the server's persistent connection, this creates a NEW connection
    each call. Callers should close it when done, or use batch_connection()
    context manager for automatic lifecycle management.

    Args:
        db_path: Path to SQLite database. Defaults to profile-based path.
        busy_timeout_ms: How long to wait for locks (default 30s, vs server's 10s).
            Batch operations are more tolerant of delays than the hot path.

    Returns:
        sqlite3.Connection with safe pragmas applied.
    """
    path = db_path or _resolve_default_db_path()
    conn = sqlite3.connect(path, timeout=busy_timeout_ms / 1000, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def check_db_lock(db_path: str | None = None) -> tuple[bool, str]:
    """Pre-flight check: can we acquire a write lock on the database?

    Attempts BEGIN IMMEDIATE (which requires the write lock), then
    immediately rolls back. This is non-destructive.

    Returns:
        (is_available, message) — True if lock is acquirable, False if contention detected.
    """
    path = db_path or _resolve_default_db_path()
    if not os.path.exists(path):
        return False, f"Database file not found: {path}"
    conn = sqlite3.connect(path, timeout=2)
    conn.execute("PRAGMA busy_timeout=2000")
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("ROLLBACK")
        conn.close()
        return True, "Write lock acquired successfully"
    except sqlite3.OperationalError as e:
        conn.close()
        return False, f"Lock contention: {e}"


def checkpoint_wal(conn: sqlite3.Connection) -> tuple[int, int, int]:
    """Run WAL checkpoint (TRUNCATE mode) to flush WAL to main DB file.

    TRUNCATE mode flushes all WAL frames to the DB and resets WAL to zero size.
    This is critical in FUSE environments where large WAL files combined with
    SHM coordination issues can cause corruption.

    Returns:
        (busy, log, checkpointed) — SQLite checkpoint result tuple.
    """
    result = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    return result if result else (0, 0, 0)


def verify_no_server(db_path: str | None = None, port: int = 8000) -> tuple[bool, str]:
    """Check that no server process holds the DB file open.

    Uses lsof to detect other processes with file descriptors on the DB.
    Also checks if anything is listening on the server port.

    Returns:
        (is_clear, message) — True if no conflicting processes found.
    """
    path = db_path or _resolve_default_db_path()
    issues = []

    # Check for processes holding DB open
    try:
        result = subprocess.run(
            ["lsof", path],
            capture_output=True, text=True, timeout=5,
        )
        if result.stdout.strip():
            lines = result.stdout.strip().split("\n")
            # DEBT-118: Use exact PID match to avoid substring false positives
            # (e.g., PID 123 matching inside PID 1234)
            my_pid = str(os.getpid())
            procs = [l for l in lines[1:] if my_pid not in l.split()]
            if procs:
                issues.append(f"{len(procs)} other process(es) hold {path} open")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        issues.append("Could not run lsof to check DB file handles")

    # Check if server port is in use
    try:
        result = subprocess.run(
            ["lsof", "-i", f":{port}"],
            capture_output=True, text=True, timeout=5,
        )
        if result.stdout.strip():
            lines = result.stdout.strip().split("\n")
            if len(lines) > 1:
                issues.append(f"Port {port} is in use (server may be running)")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass  # Non-fatal — port check is best-effort

    if issues:
        return False, "; ".join(issues)
    return True, "No conflicting processes detected"


@contextmanager
def batch_connection(db_path: str | None = None, busy_timeout_ms: int = 30000):
    """Context manager wrapper for safe batch connections.

    Automatically closes the connection on exit (normal or exception).
    Runs WAL checkpoint before closing to flush writes.

    Usage:
        with batch_connection() as conn:
            run_ka_reclassification(conn, batch_size=100)
    """
    conn = get_batch_connection(db_path, busy_timeout_ms)
    try:
        yield conn
    finally:
        try:
            checkpoint_wal(conn)
        except Exception:
            pass  # Best-effort checkpoint on exit
        conn.close()
