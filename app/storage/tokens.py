"""Storage sub-module: tokens.

Agent token CRUD operations.
Extracted from storage/__init__.py during Item 2b decomposition.
"""
import logging
import secrets

from app.core.datetime_utils import _utc_now_iso
import app.storage.connection as _conn
from app.storage.connection import read_snapshot_db
from app.storage.utils import validate_agent_id

logger = logging.getLogger(__name__)

def create_agent_token(agent_id: str, label: str = "") -> dict:
    """Create a new bearer token for an agent."""
    import secrets

    agent_id = validate_agent_id(agent_id)
    if agent_id == "default":
        raise ValueError("Cannot create token for 'default' agent_id — provide a real agent_id")
    token = f"pith_{secrets.token_urlsafe(32)}"
    now = _utc_now_iso()
    with _conn._db() as conn:
        conn.execute(
            "INSERT INTO agent_tokens (token, agent_id, label, created_at) VALUES (?, ?, ?, ?)",
            (token, agent_id, label, now),
        )
    logger.info(f"Agent token created for agent_id={agent_id} label={label!r}")
    return {"token": token, "agent_id": agent_id, "label": label, "created_at": now}


def resolve_agent_token(token: str) -> str | None:
    """Resolve a bearer token to an agent_id. Returns None if invalid/revoked."""
    if not token or not isinstance(token, str) or not token.startswith("pith_"):
        return None
    with _conn._db() as conn:
        row = conn.execute(
            "SELECT agent_id FROM agent_tokens WHERE token = ? AND revoked_at IS NULL",
            (token,),
        ).fetchone()
        if row:
            conn.execute(
                "UPDATE agent_tokens SET last_used_at = ? WHERE token = ?",
                (_utc_now_iso(), token),
            )
            return row[0]
    return None


def revoke_agent_token(token: str) -> bool:
    """Revoke a token. Returns True if token existed and was revoked."""
    with _conn._db() as conn:
        cursor = conn.execute(
            "UPDATE agent_tokens SET revoked_at = ? WHERE token = ? AND revoked_at IS NULL",
            (_utc_now_iso(), token),
        )
        return cursor.rowcount > 0


def list_agent_tokens(agent_id: str = None) -> list:
    """List tokens, optionally filtered by agent_id. Tokens are masked in output."""
    with read_snapshot_db("list_agent_tokens") as conn:
        if agent_id:
            rows = conn.execute(
                "SELECT token, agent_id, label, created_at, revoked_at, last_used_at "
                "FROM agent_tokens WHERE agent_id = ? ORDER BY created_at DESC",
                (agent_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT token, agent_id, label, created_at, revoked_at, last_used_at "
                "FROM agent_tokens ORDER BY created_at DESC"
            ).fetchall()
    return [
        {
            "token_prefix": r[0][:9] + "...",
            "agent_id": r[1],
            "label": r[2],
            "created_at": r[3],
            "revoked_at": r[4],
            "last_used_at": r[5],
        }
        for r in rows
    ]
