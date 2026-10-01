"""Storage primitives for task-owned managed session lifecycles.

The schema is additive and inert by default. Persistent writer-barrier triggers
are installed only by an explicit control-plane call to
``install_writer_barrier``; normal database initialization merely registers the
SQLite authorization functions so an already-armed database remains writable
by protocol-aware code.
"""

from __future__ import annotations

import contextvars
import os
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.core.file_lock import lock_fd_exclusive, unlock_fd
from app.core.profile import resolve_data_dir

WRITER_PROTOCOL_VERSION = 1
DEFAULT_BINDING_PROTOCOL_VERSION = 1
DEFAULT_IDLE_SECONDS = 7200
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")

WRITER_BARRIER_TRIGGER_NAMES = (
    "trg_managed_sessions_insert",
    "trg_managed_sessions_update",
    "trg_managed_sessions_delete",
    "trg_managed_lifecycle_jobs_insert",
    "trg_managed_lifecycle_jobs_update",
    "trg_managed_lifecycle_jobs_delete",
    "trg_managed_write_replays_insert",
    "trg_managed_write_replays_update",
    "trg_managed_write_replays_delete",
)


class ManagedSessionWriteDenied(RuntimeError):
    """Raised when application code attempts a managed write without authority."""


class ManagedFileGuardBusyError(RuntimeError):
    """Raised when another local process owns a managed binding file guard."""


class ManagedBindingFileGuard:
    """Nonblocking advisory guard backed by a persistent binding lock file."""

    suffix = ".lock"

    def __init__(
        self,
        binding_hash: str,
        *,
        profile: str,
        data_dir: Path | None = None,
    ) -> None:
        if not isinstance(binding_hash, str) or not _HEX_64.fullmatch(binding_hash):
            raise ValueError("binding_hash must be 64 lowercase hex characters")
        base = Path(data_dir) if data_dir is not None else resolve_data_dir(profile)
        lock_dir = base / "locks" / "session-bindings" / binding_hash[:2]
        self._path = lock_dir / f"{binding_hash}{self.suffix}"
        self._fd: int | None = None

    @property
    def acquired(self) -> bool:
        return self._fd is not None

    def acquire(self):
        if self._fd is not None:
            raise RuntimeError("binding guard is already acquired")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            lock_fd_exclusive(fd, blocking=False)
        except BlockingIOError as exc:
            os.close(fd)
            raise ManagedFileGuardBusyError() from exc
        self._fd = fd
        return self

    def release(self) -> None:
        fd = self._fd
        if fd is None:
            return
        self._fd = None
        try:
            unlock_fd(fd)
        finally:
            os.close(fd)

    def __enter__(self):
        return self.acquire()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()


class ManagedEpisodeEffectFileGuard(ManagedBindingFileGuard):
    """Binding-scoped effect guard shared by workers and stale reclaimers."""

    suffix = ".effect.lock"


@dataclass(frozen=True)
class ManagedEpisodePermit:
    """Context-local authority for one immutable binding generation."""

    binding_hash: str
    binding_generation: int

    def __post_init__(self) -> None:
        if not isinstance(self.binding_hash, str) or not _HEX_64.fullmatch(self.binding_hash):
            raise ValueError("binding_hash must be 64 lowercase hex characters")
        if (
            isinstance(self.binding_generation, bool)
            or not isinstance(self.binding_generation, int)
            or self.binding_generation < 0
        ):
            raise ValueError("binding_generation must be non-negative")


@dataclass(frozen=True)
class ManagedQueuePermit:
    """Context-local authority for one managed queue/replay row family."""

    table_name: str
    binding_hash: str
    claim_token: str | None = None

    def __post_init__(self) -> None:
        if self.table_name not in {"lifecycle_jobs", "write_request_replays"}:
            raise ValueError("unsupported managed queue table")
        if not _HEX_64.fullmatch(self.binding_hash):
            raise ValueError("binding_hash must be 64 lowercase hex characters")
        if self.claim_token is not None and not _HEX_64.fullmatch(self.claim_token):
            raise ValueError("claim_token must be 64 lowercase hex characters")


_SESSION_PERMIT: contextvars.ContextVar[ManagedEpisodePermit | None] = contextvars.ContextVar(
    "pith_managed_session_permit",
    default=None,
)
_QUEUE_PERMIT: contextvars.ContextVar[ManagedQueuePermit | None] = contextvars.ContextVar(
    "pith_managed_queue_permit",
    default=None,
)


@contextmanager
def managed_session_write_permit(permit: ManagedEpisodePermit) -> Iterator[None]:
    """Authorize managed-session SQL in the current context only."""

    token = _SESSION_PERMIT.set(permit)
    try:
        yield
    finally:
        _SESSION_PERMIT.reset(token)


def current_managed_session_write_permit() -> ManagedEpisodePermit | None:
    """Return the exact managed episode authority active in this context."""

    return _SESSION_PERMIT.get()


@contextmanager
def managed_queue_write_permit(permit: ManagedQueuePermit) -> Iterator[None]:
    """Authorize managed queue/replay SQL in the current context only."""

    token = _QUEUE_PERMIT.set(permit)
    try:
        yield
    finally:
        _QUEUE_PERMIT.reset(token)


def _normalized_generation(value: object) -> int | None:
    if value is None:
        return None
    try:
        generation = int(value)
    except (TypeError, ValueError):
        return None
    return generation if generation >= 0 else None


def _managed_session_write_authorized(
    operation: object,
    old_binding_hash: object,
    old_generation: object,
    new_binding_hash: object,
    new_generation: object,
) -> int:
    op = str(operation or "").lower()
    old_hash = str(old_binding_hash) if old_binding_hash is not None else None
    new_hash = str(new_binding_hash) if new_binding_hash is not None else None

    if old_hash is None and new_hash is None:
        return 1
    if op == "delete":
        return 0
    if op == "update" and (old_hash != new_hash or old_generation != new_generation):
        return 0

    target_hash = new_hash if new_hash is not None else old_hash
    target_generation = _normalized_generation(new_generation if new_hash is not None else old_generation)
    permit = _SESSION_PERMIT.get()
    if permit is None or target_hash is None or target_generation is None:
        return 0
    return int(permit.binding_hash == target_hash and permit.binding_generation == target_generation)


def _managed_queue_write_authorized(
    table_name: object,
    operation: object,
    old_binding_hash: object,
    new_binding_hash: object,
) -> int:
    table = str(table_name or "")
    op = str(operation or "").lower()
    old_hash = str(old_binding_hash) if old_binding_hash is not None else None
    new_hash = str(new_binding_hash) if new_binding_hash is not None else None

    if old_hash is None and new_hash is None:
        return 1
    if op == "delete":
        return 0
    if op == "update" and old_hash != new_hash:
        return 0

    target_hash = new_hash if new_hash is not None else old_hash
    permit = _QUEUE_PERMIT.get()
    if permit is None or target_hash is None:
        return 0
    if op == "update" and permit.claim_token is None:
        return 0
    return int(permit.table_name == table and permit.binding_hash == target_hash)


def register_managed_write_udfs(
    conn: sqlite3.Connection,
    *,
    writer_protocol: int = WRITER_PROTOCOL_VERSION,
) -> None:
    """Register protocol-1 writer-barrier UDFs on a writable connection."""

    if int(writer_protocol) < WRITER_PROTOCOL_VERSION:
        raise ValueError("writer_protocol is below the supported minimum")
    conn.create_function(
        "pith_managed_session_write_authorized",
        5,
        _managed_session_write_authorized,
        deterministic=False,
    )
    conn.create_function(
        "pith_managed_queue_write_authorized",
        4,
        _managed_queue_write_authorized,
        deterministic=False,
    )


_WRITER_BARRIER_SQL = (
    """CREATE TRIGGER IF NOT EXISTS trg_managed_sessions_insert
       BEFORE INSERT ON sessions BEGIN
         SELECT CASE WHEN pith_managed_session_write_authorized(
           'insert', NULL, NULL, NEW.binding_hash, NEW.binding_generation
         ) = 1 THEN 1 ELSE RAISE(ABORT, 'managed session write denied') END;
       END""",
    """CREATE TRIGGER IF NOT EXISTS trg_managed_sessions_update
       BEFORE UPDATE ON sessions BEGIN
         SELECT CASE WHEN pith_managed_session_write_authorized(
           'update', OLD.binding_hash, OLD.binding_generation,
           NEW.binding_hash, NEW.binding_generation
         ) = 1 THEN 1 ELSE RAISE(ABORT, 'managed session write denied') END;
       END""",
    """CREATE TRIGGER IF NOT EXISTS trg_managed_sessions_delete
       BEFORE DELETE ON sessions BEGIN
         SELECT CASE WHEN pith_managed_session_write_authorized(
           'delete', OLD.binding_hash, OLD.binding_generation, NULL, NULL
         ) = 1 THEN 1 ELSE RAISE(ABORT, 'managed session write denied') END;
       END""",
    """CREATE TRIGGER IF NOT EXISTS trg_managed_lifecycle_jobs_insert
       BEFORE INSERT ON lifecycle_jobs BEGIN
         SELECT CASE WHEN pith_managed_queue_write_authorized(
           'lifecycle_jobs','insert',NULL,NEW.binding_hash
         ) = 1 THEN 1 ELSE RAISE(ABORT, 'lifecycle job write denied') END;
       END""",
    """CREATE TRIGGER IF NOT EXISTS trg_managed_lifecycle_jobs_update
       BEFORE UPDATE ON lifecycle_jobs BEGIN
         SELECT CASE WHEN pith_managed_queue_write_authorized(
           'lifecycle_jobs','update',OLD.binding_hash,NEW.binding_hash
         ) = 1 THEN 1 ELSE RAISE(ABORT, 'lifecycle job write denied') END;
       END""",
    """CREATE TRIGGER IF NOT EXISTS trg_managed_lifecycle_jobs_delete
       BEFORE DELETE ON lifecycle_jobs BEGIN
         SELECT CASE WHEN pith_managed_queue_write_authorized(
           'lifecycle_jobs','delete',OLD.binding_hash,NULL
         ) = 1 THEN 1 ELSE RAISE(ABORT, 'lifecycle job delete denied') END;
       END""",
    """CREATE TRIGGER IF NOT EXISTS trg_managed_write_replays_insert
       BEFORE INSERT ON write_request_replays BEGIN
         SELECT CASE WHEN pith_managed_queue_write_authorized(
           'write_request_replays','insert',NULL,NEW.binding_hash
         ) = 1 THEN 1 ELSE RAISE(ABORT, 'write replay write denied') END;
       END""",
    """CREATE TRIGGER IF NOT EXISTS trg_managed_write_replays_update
       BEFORE UPDATE ON write_request_replays BEGIN
         SELECT CASE WHEN pith_managed_queue_write_authorized(
           'write_request_replays','update',OLD.binding_hash,NEW.binding_hash
         ) = 1 THEN 1 ELSE RAISE(ABORT, 'write replay write denied') END;
       END""",
    """CREATE TRIGGER IF NOT EXISTS trg_managed_write_replays_delete
       BEFORE DELETE ON write_request_replays BEGIN
         SELECT CASE WHEN pith_managed_queue_write_authorized(
           'write_request_replays','delete',OLD.binding_hash,NULL
         ) = 1 THEN 1 ELSE RAISE(ABORT, 'write replay delete denied') END;
       END""",
)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _assert_barrier_schema(conn: sqlite3.Connection) -> None:
    required = {
        "sessions": {"binding_hash", "binding_generation"},
        "lifecycle_jobs": {"binding_hash", "claim_token"},
        "write_request_replays": {"binding_hash", "claim_token"},
        "session_binding_policy": {"writer_barrier_armed", "minimum_writer_protocol"},
    }
    for table, expected in required.items():
        actual = {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}
        missing = expected - actual
        if missing:
            raise RuntimeError(f"writer barrier schema missing {table}: {sorted(missing)}")


def install_writer_barrier(
    conn: sqlite3.Connection,
    *,
    profile: str,
    updated_by: str,
    now: str | None = None,
    writer_protocol: int = WRITER_PROTOCOL_VERSION,
) -> dict[str, object]:
    """Atomically install persistent writer triggers and arm one policy row.

    This function is intentionally never called by schema initialization.
    Production callers must sit behind the separate guarded operator command.
    """

    if not profile.strip():
        raise ValueError("profile is required")
    if not updated_by.strip():
        raise ValueError("updated_by is required")
    protocol = int(writer_protocol)
    if protocol < WRITER_PROTOCOL_VERSION:
        raise ValueError("writer_protocol is below the supported minimum")

    _assert_barrier_schema(conn)
    ts = now or _utc_now_iso()
    started_transaction = not conn.in_transaction
    if started_transaction:
        conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            """INSERT OR IGNORE INTO session_binding_policy
               (profile, mode, protocol_version, idle_seconds,
                minimum_writer_protocol, writer_barrier_armed,
                writer_barrier_armed_at, updated_at, updated_by)
               VALUES (?, 'legacy_disabled', ?, ?, ?, 0, NULL, ?, ?)""",
            (
                profile,
                DEFAULT_BINDING_PROTOCOL_VERSION,
                DEFAULT_IDLE_SECONDS,
                WRITER_PROTOCOL_VERSION,
                ts,
                updated_by,
            ),
        )
        row = conn.execute(
            """SELECT minimum_writer_protocol
               FROM session_binding_policy WHERE profile=?""",
            (profile,),
        ).fetchone()
        minimum = int(row[0]) if row is not None else WRITER_PROTOCOL_VERSION
        if protocol < minimum:
            raise RuntimeError("writer protocol is below persisted minimum")
        for statement in _WRITER_BARRIER_SQL:
            conn.execute(statement)
        conn.execute(
            """UPDATE session_binding_policy
               SET writer_barrier_armed=1,
                   writer_barrier_armed_at=COALESCE(writer_barrier_armed_at, ?),
                   updated_at=?,
                   updated_by=?
               WHERE profile=?""",
            (ts, ts, updated_by, profile),
        )
        if started_transaction:
            conn.commit()
    except Exception:
        if started_transaction and conn.in_transaction:
            conn.rollback()
        raise
    return writer_barrier_status(conn, profile=profile)


def writer_barrier_status(conn: sqlite3.Connection, *, profile: str) -> dict[str, object]:
    """Return policy and trigger state without mutating the database."""

    row = conn.execute(
        """SELECT profile, mode, protocol_version, idle_seconds,
                  minimum_writer_protocol, writer_barrier_armed,
                  writer_barrier_armed_at, updated_at, updated_by
           FROM session_binding_policy WHERE profile=?""",
        (profile,),
    ).fetchone()
    present = {
        str(trigger_row[0])
        for trigger_row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE 'trg_managed_%'"
        )
    }
    policy = dict(row) if row is not None and hasattr(row, "keys") else None
    if row is not None and policy is None:
        keys = (
            "profile",
            "mode",
            "protocol_version",
            "idle_seconds",
            "minimum_writer_protocol",
            "writer_barrier_armed",
            "writer_barrier_armed_at",
            "updated_at",
            "updated_by",
        )
        policy = dict(zip(keys, row, strict=True))
    return {
        "policy": policy,
        "triggers_present": sorted(present),
        "triggers_complete": set(WRITER_BARRIER_TRIGGER_NAMES).issubset(present),
    }
