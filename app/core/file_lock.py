"""Cross-platform advisory file locking helpers."""

from __future__ import annotations

import os

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised by Windows VM proof.
    fcntl = None
    import msvcrt
else:
    msvcrt = None


def lock_fd_exclusive(fd: int, *, blocking: bool = False) -> None:
    """Acquire an exclusive advisory lock for a file descriptor."""
    if fcntl is not None:
        flags = fcntl.LOCK_EX
        if not blocking:
            flags |= fcntl.LOCK_NB
        fcntl.flock(fd, flags)
        return

    os.lseek(fd, 0, os.SEEK_SET)
    try:
        mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
        msvcrt.locking(fd, mode, 1)
    except OSError as exc:
        raise BlockingIOError(str(exc)) from exc


def unlock_fd(fd: int) -> None:
    """Release an advisory lock for a file descriptor."""
    if fcntl is not None:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return

    os.lseek(fd, 0, os.SEEK_SET)
    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)


def lock_file_exclusive(file_obj, *, blocking: bool = False) -> None:
    """Acquire an exclusive advisory lock for an open file object."""
    lock_fd_exclusive(file_obj.fileno(), blocking=blocking)


def unlock_file(file_obj) -> None:
    """Release an advisory lock for an open file object."""
    unlock_fd(file_obj.fileno())
