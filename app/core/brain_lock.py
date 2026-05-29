"""BRAIN_LOCK_SPEC: Exclusive data-directory lock for Pith server.

Prevents multiple server processes from writing to the same SQLite brain
simultaneously. Uses kernel-level flock — model-agnostic, system-wide,
automatically released on process death.

Multi-worker aware: when uvicorn runs with --workers N, the first worker
acquires the lock. Sibling workers (same parent PID) are allowed through.

See the internal BRAIN_LOCK design notes.
"""

import fcntl
import json
import logging
import os
import signal
import time

logger = logging.getLogger(__name__)

_lock_fd = None


class BrainLockError(RuntimeError):
    """Raised when another server already holds the brain lock."""
    pass


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def ensure_non_server_prod_db_safe(
    *,
    tool_name: str,
    write_capable: bool,
    read_only: bool = False,
    database_path=None,
) -> None:
    """Guard non-server tools from accidental protected-profile writes."""
    from pathlib import Path

    from app.core.profile import PITH_DATA_ROOT, get_active_profile, resolve_data_dir

    data_dir = Path(resolve_data_dir()).resolve()
    protected_profile = os.environ.get("PITH_PROTECTED_PROFILE", "").strip()
    protected_dir = (
        (PITH_DATA_ROOT / protected_profile).resolve()
        if protected_profile
        else None
    )
    resolved_database_path = (
        Path(database_path).expanduser().resolve() if database_path else None
    )
    profile = get_active_profile()
    database_targets_protected = bool(
        protected_dir
        and
        resolved_database_path
        and (
            resolved_database_path == protected_dir
            or protected_dir in resolved_database_path.parents
        )
    )
    production_protected = bool(
        protected_profile
        and (
            profile == protected_profile
            or data_dir == protected_dir
            or database_targets_protected
        )
    )
    if not production_protected:
        return

    explicit_readonly = read_only or (
        not write_capable
        and (
            _truthy_env("PITH_BENCHMARK_READONLY")
            or _truthy_env("PITH_ALLOW_PROD_DB_READONLY")
        )
    )
    if not write_capable or explicit_readonly:
        logger.warning(
            "PROD-DB-GUARD: allowing read-only protected-profile tool "
            "%s pid=%s profile=%s data_dir=%s",
            tool_name,
            os.getpid(),
            profile,
            resolved_database_path or data_dir,
        )
        return

    override = os.environ.get("PITH_ALLOW_PROD_DB_WRITE_TOOL", "").strip()
    if override == tool_name:
        logger.critical(
            "PROD-DB-GUARD: explicit write override for %s "
            "pid=%s profile=%s data_dir=%s",
            tool_name,
            os.getpid(),
            profile,
            resolved_database_path or data_dir,
        )
        return

    raise BrainLockError(
        "PROD-DB-GUARD: refusing write-capable non-server tool "
        f"{tool_name!r} against protected production DB boundary "
        f"{resolved_database_path or data_dir}. "
        "Use isolated PITH_DATA_DIR, set PITH_BENCHMARK_READONLY=1 for "
        "read-only diagnostics, or set PITH_ALLOW_PROD_DB_WRITE_TOOL to "
        "the exact tool name only for an intentional maintenance override."
    )


def acquire_brain_lock(
    data_dir: str, port: int = 0, pid: int = 0, force_takeover: bool = False,
) -> None:
    """Acquire exclusive lock on the brain data directory.

    Multi-worker safe: sibling workers (same ppid) are allowed through.
    Raises BrainLockError for genuinely different server processes.

    STABILITY-043: When force_takeover=True (or PITH_BRAIN_LOCK_FORCE_TAKEOVER
    env var is set), sends SIGTERM to the holder PID and retries instead of
    raising BrainLockError. Prevents brain lock bounce loops during restarts.
    """
    force_takeover = force_takeover or os.environ.get(
        "PITH_BRAIN_LOCK_FORCE_TAKEOVER", ""
    ).lower() in ("1", "true", "yes")
    global _lock_fd

    lock_path = os.path.join(data_dir, "server.lock")
    my_pid = pid or os.getpid()
    my_ppid = os.getppid()

    # Open WITHOUT truncation — critical for multi-worker metadata reads
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    _lock_fd = os.fdopen(fd, "r+")

    try:
        fcntl.flock(_lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        # Lock held — read metadata from the file (written by lock holder)
        _lock_fd.seek(0)
        holder = {}

        # Retry reads: the lock holder may still be writing metadata
        for attempt in range(5):
            try:
                _lock_fd.seek(0)
                data = _lock_fd.read().strip()
                if data:
                    holder = json.loads(data)
                    break
            except Exception:
                pass
            time.sleep(0.2)

        holder_ppid = holder.get("ppid", -1)
        holder_pid = holder.get("pid", "?")

        if holder_ppid == my_ppid:
            _lock_fd.close()
            _lock_fd = None
            logger.info(
                "BRAIN-LOCK: Sibling worker detected (holder PID %s, "
                "our PID %s, shared parent %s) — skipping lock",
                holder_pid, my_pid, my_ppid,
            )
            return

        # T1-2: Stale-lock recovery — check if holder PID is still alive.
        # When the old server crashes (SIGKILL, OOM, etc.), flock SHOULD
        # release automatically. But if the fd is inherited by a child or
        # the process is a zombie, the lock can persist. Check liveness
        # before denying.
        holder_alive = True
        if isinstance(holder_pid, int) and holder_pid > 0:
            try:
                os.kill(holder_pid, 0)  # signal 0 = liveness check only
            except ProcessLookupError:
                holder_alive = False
            except PermissionError:
                holder_alive = True  # process exists but different user

        if not holder_alive:
            logger.warning(
                "BRAIN-LOCK: Stale lock detected — holder PID %s is dead. "
                "Stealing lock for PID %s (T1-2 recovery).",
                holder_pid, my_pid,
            )
            # Close stale fd and re-open fresh to avoid inheriting the
            # old flock state. Then acquire with blocking wait (brief).
            _lock_fd.close()
            _lock_fd = None
            fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
            _lock_fd = os.fdopen(fd, "r+")
            try:
                fcntl.flock(_lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                # Another process beat us to the steal — retry with brief wait
                try:
                    fcntl.flock(_lock_fd.fileno(), fcntl.LOCK_EX)
                except Exception as steal_err:
                    _lock_fd.close()
                    _lock_fd = None
                    raise BrainLockError(
                        f"BRAIN LOCK: Stale lock steal failed after "
                        f"dead PID {holder_pid}: {steal_err}"
                    ) from steal_err
        else:
            # STABILITY-043: Force takeover — SIGTERM the holder and retry
            if force_takeover and isinstance(holder_pid, int) and holder_pid > 0:
                logger.warning(
                    "BRAIN-LOCK: Force takeover — sending SIGTERM to holder "
                    "PID %s (STABILITY-043).",
                    holder_pid,
                )
                _lock_fd.close()
                _lock_fd = None
                try:
                    os.kill(holder_pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError) as e:
                    logger.warning(
                        "BRAIN-LOCK: Could not SIGTERM PID %s: %s", holder_pid, e
                    )

                # Wait for holder to release (up to 10s)
                for _wait in range(20):
                    time.sleep(0.5)
                    try:
                        os.kill(holder_pid, 0)
                    except ProcessLookupError:
                        break  # Dead — lock should be free
                else:
                    # Still alive after 10s — escalate to SIGKILL
                    logger.warning(
                        "BRAIN-LOCK: PID %s did not exit after SIGTERM, "
                        "sending SIGKILL.",
                        holder_pid,
                    )
                    try:
                        os.kill(holder_pid, signal.SIGKILL)
                        time.sleep(1.0)
                    except (ProcessLookupError, PermissionError):
                        pass

                # Retry lock acquisition
                fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
                _lock_fd = os.fdopen(fd, "r+")
                try:
                    fcntl.flock(_lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    _lock_fd.close()
                    _lock_fd = None
                    raise BrainLockError(
                        f"BRAIN LOCK: Force takeover failed — PID {holder_pid} "
                        f"was killed but lock still held (child process inheritance?)"
                    )
                logger.info(
                    "BRAIN-LOCK: Force takeover succeeded — acquired lock "
                    "after killing PID %s.",
                    holder_pid,
                )
            else:
                _lock_fd.close()
                _lock_fd = None

                # STABILITY-044: Circuit breaker — after MAX consecutive denials,
                # write stuck + disabled flags so KeepAlive respawns exit fast.
                _DENIAL_COUNT_FILE = os.path.join(data_dir, "brainlock_denial_count")
                _MAX_DENIALS = int(os.environ.get("PITH_BRAIN_LOCK_MAX_DENIALS", "10"))

                consecutive = 0
                try:
                    if os.path.exists(_DENIAL_COUNT_FILE):
                        with open(_DENIAL_COUNT_FILE) as f:
                            consecutive = int(f.read().strip())
                except (ValueError, OSError):
                    consecutive = 0

                consecutive += 1

                try:
                    with open(_DENIAL_COUNT_FILE, "w") as f:
                        f.write(str(consecutive))
                except OSError:
                    pass

                if consecutive >= _MAX_DENIALS:
                    # Write diagnostic flag
                    stuck_flag = os.path.join(data_dir, "server.brainlock_stuck")
                    try:
                        with open(stuck_flag, "w") as f:
                            f.write(
                                f"holder_pid={holder_pid} "
                                f"denials={consecutive} "
                                f"since={time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
                            )
                    except OSError:
                        pass

                    # Write server.disabled flag — KeepAlive=true (boolean) respawns
                    # on ALL exits regardless of exit code, so sys.exit() won't stop
                    # the cascade. Instead, server.py checks this flag early in startup
                    # and exits in <100ms (Fix 2b). The health guard also respects it.
                    pith_home = os.environ.get(
                        "PITH_HOME", os.path.expanduser("~/.pith")
                    )
                    disabled_flag = os.path.join(pith_home, "server.disabled")
                    try:
                        with open(disabled_flag, "w") as f:
                            f.write(
                                f"brainlock_circuit_breaker "
                                f"denials={consecutive} "
                                f"holder_pid={holder_pid}\n"
                            )
                    except OSError:
                        pass

                    logger.critical(
                        "STABILITY-044 CIRCUIT BREAKER: %d consecutive brain lock "
                        "denials by PID %s. Wrote server.disabled + "
                        "server.brainlock_stuck flags. Recovery: kill PID %s, "
                        "then rm server.disabled && pith start.",
                        consecutive, holder_pid, holder_pid,
                    )

                msg = (
                    f"BRAIN LOCK DENIED: Another server "
                    f"(PID {holder_pid}, port {holder.get('port', '?')}, "
                    f"started {holder.get('started', '?')}) "
                    f"already holds exclusive access to {data_dir}. "
                    f"Kill that process first or use a different PITH_DATA_DIR. "
                    f"(denial {consecutive}/{_MAX_DENIALS} before circuit breaker)"
                )
                logger.critical(msg)
                raise BrainLockError(msg)
    except OSError as e:
        logger.warning(f"Brain lock acquisition failed (non-fatal): {e}")
        return

    # Got the lock — clean up any circuit breaker state from prior cascade
    for _cleanup_file in (
        os.path.join(data_dir, "brainlock_denial_count"),
        os.path.join(data_dir, "server.brainlock_stuck"),
    ):
        try:
            os.unlink(_cleanup_file)
        except FileNotFoundError:
            pass

    # Got the lock — write metadata (truncate first, then write)
    _lock_fd.seek(0)
    _lock_fd.truncate()
    holder_info = {
        "pid": my_pid,
        "ppid": my_ppid,
        "port": port,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "data_dir": data_dir,
    }
    _lock_fd.write(json.dumps(holder_info))
    _lock_fd.flush()
    os.fsync(_lock_fd.fileno())

    logger.info(
        "BRAIN-LOCK: Exclusive lock acquired on %s (PID %s, port %s)",
        data_dir, my_pid, port,
    )


def release_brain_lock() -> None:
    """Release the brain lock. Safe to call multiple times."""
    global _lock_fd
    if _lock_fd is not None:
        try:
            fcntl.flock(_lock_fd.fileno(), fcntl.LOCK_UN)
            _lock_fd.close()
        except Exception:
            pass
        _lock_fd = None
