"""DEBT-001: Timezone-aware datetime utilities.

Centralizes UTC datetime operations for the codebase-wide migration
from deprecated _utc_now() to timezone-aware alternatives.

Three utilities:
  _utc_now()     — Returns datetime.now(timezone.utc) for arithmetic/comparisons
  _utc_now_iso() — Returns UTC ISO string WITHOUT +00:00 suffix (SQLite sort compat)
  _ensure_aware(dt) — Makes a possibly-naive datetime timezone-aware (assumes UTC)
"""

from datetime import UTC, datetime


def _utc_now() -> datetime:
    """Return current UTC time as timezone-aware datetime.

    Replaces deprecated _utc_now() which returns naive datetime.
    Use this for datetime arithmetic and comparisons.
    """
    return datetime.now(UTC)


def _utc_now_iso() -> str:
    """Return current UTC time as ISO string without timezone suffix.

    Produces '2026-03-06T12:00:00' format (no +00:00 or Z suffix).
    This preserves SQLite string sort order compatibility with existing
    naive timestamp strings in the database.

    Use this for all DB timestamp writes (created_at, updated_at, etc.).
    """
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")


def _ensure_aware(dt: datetime) -> datetime:
    """Ensure a datetime is timezone-aware, assuming UTC if naive.

    Handles mixed-state during migration: old DB rows have naive timestamps
    from fromisoformat(), new rows may have aware timestamps.

    Args:
        dt: A datetime that may or may not have tzinfo.

    Returns:
        The same datetime with timezone.utc attached if it was naive.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt
