"""
Phase 4.5 WS3: SQL Compatibility Helpers

Abstracts SQLite vs PostgreSQL syntax differences so subsystems can write
backend-agnostic SQL.  All helpers accept a `backend_type` string ("sqlite"
or "postgresql") and return the appropriate SQL fragment.

AUDIT SUMMARY (Phase 4.5 WS3):
  datetime('now')    — 14 sites  (migration, seed_domains, async_tasks)
  INSERT OR REPLACE  — 10 sites  (experiments, threads, contradiction_llm,
                                   skills, causal, storage)
  INSERT OR IGNORE   —  9 sites  (migration, episodes, storage, nonlossy)
  AUTOINCREMENT      —  9 sites  (migration, correction, storage, async_tasks)
  PRAGMA             —  5 sites  (storage_backend — already backend-specific)
  GROUP_CONCAT       —  2 sites  (skills)
  lastrowid          —  2 sites  (correction, async_tasks)
  json_extract       —  1 site   (auto_reflection)

Usage:
    from app.storage.sql_compat import sql
    # sql is a module-level helper bound to the active backend
    expr = sql.now()          # "datetime('now')" or "NOW()"
    expr = sql.json_get(col, key)
    ddl  = sql.auto_id()     # "INTEGER PRIMARY KEY AUTOINCREMENT" or "SERIAL PRIMARY KEY"
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class SQLCompat:
    """Backend-aware SQL fragment generator.

    Instantiated once per backend type.  Thread-safe (stateless after init).
    """

    def __init__(self, backend_type: str = "sqlite"):
        self._backend = backend_type

    @property
    def backend_type(self) -> str:
        return self._backend

    # ------------------------------------------------------------------
    # Date/time
    # ------------------------------------------------------------------

    def now(self) -> str:
        """SQL expression for current UTC timestamp."""
        if self._backend == "postgresql":
            return "NOW()"
        return "datetime('now')"

    # ------------------------------------------------------------------
    # JSON access
    # ------------------------------------------------------------------

    def json_get(self, column: str, key: str) -> str:
        """Extract a top-level key from a JSON column.

        SQLite:      json_extract(column, '$.key')
        PostgreSQL:  column->>'key'
        """
        if self._backend == "postgresql":
            return f"{column}->>'{key}'"
        return f"json_extract({column}, '$.{key}')"

    # ------------------------------------------------------------------
    # UPSERT helpers
    # ------------------------------------------------------------------

    def upsert_replace(self, table: str, columns: list[str], conflict_col: str) -> str:
        """Generate an UPSERT that replaces on conflict.

        SQLite:      INSERT OR REPLACE INTO ...
        PostgreSQL:  INSERT INTO ... ON CONFLICT (col) DO UPDATE SET ...

        Returns the full INSERT statement with ? placeholders.
        """
        placeholders = ", ".join(["?"] * len(columns))
        cols = ", ".join(columns)

        if self._backend == "postgresql":
            # Build SET clause for all non-conflict columns
            set_parts = [f"{c} = EXCLUDED.{c}" for c in columns if c != conflict_col]
            set_clause = ", ".join(set_parts) if set_parts else f"{columns[0]} = EXCLUDED.{columns[0]}"
            return (
                f"INSERT INTO {table} ({cols}) VALUES ({placeholders}) "
                f"ON CONFLICT ({conflict_col}) DO UPDATE SET {set_clause}"
            )
        return f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({placeholders})"

    def upsert_ignore(self, table: str, columns: list[str], conflict_col: str | None = None) -> str:
        """Generate an INSERT that silently skips on conflict.

        SQLite:      INSERT OR IGNORE INTO ...
        PostgreSQL:  INSERT INTO ... ON CONFLICT DO NOTHING

        Returns the full INSERT statement with ? placeholders.
        """
        placeholders = ", ".join(["?"] * len(columns))
        cols = ", ".join(columns)

        if self._backend == "postgresql":
            conflict = f" ({conflict_col})" if conflict_col else ""
            return f"INSERT INTO {table} ({cols}) VALUES ({placeholders}) ON CONFLICT{conflict} DO NOTHING"
        return f"INSERT OR IGNORE INTO {table} ({cols}) VALUES ({placeholders})"

    # ------------------------------------------------------------------
    # DDL helpers
    # ------------------------------------------------------------------

    def auto_id(self, column: str = "id") -> str:
        """Auto-incrementing primary key DDL fragment.

        SQLite:      id INTEGER PRIMARY KEY AUTOINCREMENT
        PostgreSQL:  id SERIAL PRIMARY KEY
        """
        if self._backend == "postgresql":
            return f"{column} SERIAL PRIMARY KEY"
        return f"{column} INTEGER PRIMARY KEY AUTOINCREMENT"

    def text_type(self) -> str:
        """Standard text column type (same for both, included for clarity)."""
        return "TEXT"

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def group_concat(self, column: str, separator: str = ",") -> str:
        """Aggregate strings into a delimited list.

        SQLite:      GROUP_CONCAT(column, separator)
        PostgreSQL:  STRING_AGG(column, separator)
        """
        if self._backend == "postgresql":
            return f"STRING_AGG({column}, '{separator}')"
        return f"GROUP_CONCAT({column}, '{separator}')"

    # ------------------------------------------------------------------
    # Last inserted ID
    # ------------------------------------------------------------------

    def returning_id(self) -> str:
        """Suffix for INSERT to return the generated ID.

        SQLite:      '' (use cursor.lastrowid instead)
        PostgreSQL:  ' RETURNING id'

        For SQLite callers, continue using cursor.lastrowid.
        For PostgreSQL, append this to INSERT and fetchone()[0].
        """
        if self._backend == "postgresql":
            return " RETURNING id"
        return ""

    def get_last_id(self, cursor) -> int:
        """Extract last inserted auto-increment ID from a cursor.

        SQLite:      cursor.lastrowid
        PostgreSQL:  cursor.fetchone()[0]  (requires RETURNING id)
        """
        if self._backend == "postgresql":
            row = cursor.fetchone()
            return row[0] if row else 0
        return cursor.lastrowid

    # ------------------------------------------------------------------
    # Placeholder style
    # ------------------------------------------------------------------

    @property
    def param_style(self) -> str:
        """Parameter placeholder character.

        SQLite:      ?
        PostgreSQL:  %s
        """
        if self._backend == "postgresql":
            return "%s"
        return "?"


# ---------------------------------------------------------------------------
# Module-level singleton — bound to the active backend at first access
# ---------------------------------------------------------------------------

_sql: SQLCompat | None = None


def get_sql() -> SQLCompat:
    """Get the SQL compatibility helper for the active backend.

    Lazy-initializes from get_backend().backend_type on first call.
    """
    global _sql
    if _sql is None:
        try:
            from app.storage.backend import get_backend

            _sql = SQLCompat(get_backend().backend_type)
        except Exception:
            # Fallback to sqlite if backend not yet initialized
            _sql = SQLCompat("sqlite")
    return _sql


def reset_sql() -> None:
    """Reset the singleton (for testing or backend switch)."""
    global _sql
    _sql = None


# Convenience alias
sql = property(lambda self: get_sql())
