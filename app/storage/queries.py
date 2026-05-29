"""Storage sub-module: queries.

Concept query helpers, firmware CRUD, metadata.
Extracted from storage/__init__.py during Item 2b decomposition.
"""
import logging
from contextlib import nullcontext

import app.storage.connection as _conn
from app.core.datetime_utils import _utc_now_iso
from app.storage.connection import diagnostic_read_db, read_snapshot_db, required_context_read_db

logger = logging.getLogger(__name__)

def load_concepts_by_type(concept_types: list, limit: int = 20, min_confidence: float = 0.0) -> list:
    """Load active concepts filtered by concept_type.

    Used for ambient principle retrieval — surfaces principles, methods,
    and strategies regardless of keyword match, ordered by confidence desc.
    """
    if not concept_types:
        return []
    placeholders = ",".join("?" * len(concept_types))
    with read_snapshot_db("load_concepts_by_type") as conn:
        rows = conn.execute(
            f"""
            SELECT id, summary, confidence, concept_type, knowledge_area, data
            FROM concepts
            WHERE status = 'active'
              AND concept_type IN ({placeholders})
              AND confidence >= ?
            ORDER BY confidence DESC
            LIMIT ?
        """,
            (*concept_types, min_confidence, limit),
        ).fetchall()

    results = []
    for row in rows:
        results.append(
            {
                "concept_id": row["id"],
                "summary": row["summary"],
                "confidence": row["confidence"],
                "concept_type": row["concept_type"],
                "knowledge_area": row["knowledge_area"],
            }
        )
    return results


def load_recent_concepts_by_types(
    concept_types: list,
    since_iso: str = None,
    limit: int = 5,
    min_confidence: float = 0.40,
    order_by: str = "created_at DESC",
    require_active_currency: bool = False,
    exclude_quarantined: bool = False,
) -> list:
    """Load concepts filtered by concept_type, ordered by recency or confidence.

    Used for orientation enrichment — surfaces decisions, principles,
    and findings to include in resumption briefings.

    S7.1: since_iso is now optional. When None, queries across ALL time
    (for strategic context that transcends recency). order_by allows
    sorting by confidence DESC for importance-based retrieval.
    """
    if not concept_types:
        return []
    # Validate order_by to prevent SQL injection
    allowed_orders = {"created_at DESC", "confidence DESC", "created_at ASC"}
    if order_by not in allowed_orders:
        order_by = "created_at DESC"

    placeholders = ",".join("?" * len(concept_types))
    conditions = [
        "status = 'active'",
        f"concept_type IN ({placeholders})",
        "confidence >= ?",
    ]
    params = list(concept_types) + [min_confidence]

    if since_iso is not None:
        conditions.append("created_at >= ?")
        params.append(since_iso)
    if require_active_currency:
        conditions.append("currency_status = 'ACTIVE'")
    if exclude_quarantined:
        conditions.append("maturity != 'QUARANTINED'")

    params.append(limit)
    where_clause = " AND ".join(conditions)

    with read_snapshot_db("load_recent_concepts_by_types") as conn:
        rows = conn.execute(
            f"""
            SELECT id, summary, confidence, concept_type, knowledge_area
            FROM concepts
            WHERE {where_clause}
            ORDER BY {order_by}
            LIMIT ?
        """,
            tuple(params),
        ).fetchall()

    results = []
    for row in rows:
        results.append(
            {
                "concept_id": row["id"],
                "summary": row["summary"],
                "confidence": row["confidence"],
                "concept_type": row["concept_type"],
                "knowledge_area": row["knowledge_area"],
            }
        )
    return results


def load_recent_concepts(
    since_iso: str, limit: int = 10, min_confidence: float = 0.35, exclude_stale: bool = False
) -> list:
    """Load recent concepts of ALL types, ordered by recency.

    Used for orientation: sources WHERE BEEN and WHERE NOW from knowledge
    layer instead of stale checkpoint data. Returns the most recently
    created concepts regardless of concept_type.

    CONCEPT_LIFECYCLE_SPEC L1: exclude_stale filters out concepts with
    non-ACTIVE currency_status (STALE, SUPERSEDED, etc.).
    """
    conditions = [
        "status = 'active'",
        "confidence >= ?",
        "created_at >= ?",
    ]
    params = [min_confidence, since_iso]

    if exclude_stale:
        conditions.append("currency_status = 'ACTIVE'")

    where_clause = " AND ".join(conditions)

    with read_snapshot_db("load_recent_concepts") as conn:
        rows = conn.execute(
            f"""
            SELECT id, summary, confidence, concept_type, knowledge_area,
                   created_at, maturity, currency_status
            FROM concepts
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT ?
        """,
            tuple(params) + (limit,),
        ).fetchall()

    results = []
    for row in rows:
        results.append(
            {
                "concept_id": row["id"],
                "summary": row["summary"],
                "confidence": row["confidence"],
                "concept_type": row["concept_type"],
                "knowledge_area": row["knowledge_area"],
                "created_at": row["created_at"],
                "maturity": row["maturity"] if "maturity" in row.keys() else "ESTABLISHED",
                "currency_status": row["currency_status"] if "currency_status" in row.keys() else "ACTIVE",
            }
        )
    return results


def load_always_activate_concepts() -> list:
    """Load all concepts flagged as always_activate.

    These concepts are injected into EVERY conversation_turn response
    regardless of topic or search relevance. Used for operational constraints
    that must fire at tool-selection time (e.g., 'use Desktop Commander for host paths').

    P1-1: Always-Activate concept tags.
    GOVERNANCE: Capped at MAX_ALWAYS_ACTIVATE (config.py) to prevent budget creep.
    """
    from app.core.config import MAX_ALWAYS_ACTIVATE

    with required_context_read_db("load_always_activate_concepts") as conn:
        rows = conn.execute("""
            SELECT id, summary, confidence, concept_type, knowledge_area
            FROM concepts
            WHERE status = 'active'
              AND always_activate = 1
            ORDER BY confidence DESC
            LIMIT ?
        """, (MAX_ALWAYS_ACTIVATE,)).fetchall()

    return [
        {
            "concept_id": row["id"],
            "summary": row["summary"],
            "confidence": row["confidence"],
            "concept_type": row["concept_type"],
            "knowledge_area": row["knowledge_area"],
        }
        for row in rows
    ]


def list_knowledge_area_summaries(conn=None) -> list[dict]:
    """Return column-backed knowledge-area counts without full Concept loading."""
    context = nullcontext(conn) if conn is not None else diagnostic_read_db("knowledge_area_summaries")
    with context as conn:
        rows = conn.execute(
            """
            SELECT
                COALESCE(knowledge_area, 'unknown') AS name,
                COUNT(*) AS concept_count,
                COALESCE(AVG(confidence), 0.0) AS avg_confidence
            FROM concepts
            WHERE status = 'active' AND is_current = 1
            GROUP BY COALESCE(knowledge_area, 'unknown')
            ORDER BY concept_count DESC
            """
        ).fetchall()
    return [
        {
            "name": row["name"],
            "concept_count": row["concept_count"],
            "avg_confidence": round(row["avg_confidence"] or 0.0, 2),
        }
        for row in rows
    ]


def list_concepts_for_knowledge_area(area_name: str, conn=None) -> list[dict]:
    """Return active-current concepts for a column-backed knowledge area."""
    context = nullcontext(conn) if conn is not None else diagnostic_read_db("knowledge_area_concepts")
    with context as conn:
        rows = conn.execute(
            """
            SELECT id, version, confidence, summary, created_at
            FROM concepts
            WHERE status = 'active'
              AND is_current = 1
              AND COALESCE(knowledge_area, 'unknown') = ?
            ORDER BY confidence DESC
            """,
            (area_name,),
        ).fetchall()
    return [
        {
            "id": row["id"],
            "version": row["version"],
            "confidence": row["confidence"],
            "summary": row["summary"][:200] + "..." if len(row["summary"]) > 200 else row["summary"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def set_always_activate(concept_id: str, value: bool) -> bool:
    """Set or unset always_activate flag on a concept.

    Returns True if concept was found and updated, False otherwise.
    Raises ValueError if enabling would exceed MAX_ALWAYS_ACTIVATE cap.

    GOVERNANCE: Write-side guard prevents AA budget creep.
    Without this, always_activate flags accumulate unbounded and consume
    contextual retrieval slots (each AA concept costs one slot from
    CONTEXT_BUDGET_MAIN on every conversation turn).
    """
    from app.core.config import MAX_ALWAYS_ACTIVATE

    with _conn._db() as conn:
        if value:
            # Write-side cap enforcement: refuse to flag beyond MAX_ALWAYS_ACTIVATE
            current_count = conn.execute(
                "SELECT COUNT(*) FROM concepts WHERE always_activate = 1 AND status = 'active' AND id != ?",
                (concept_id,)
            ).fetchone()[0]
            if current_count >= MAX_ALWAYS_ACTIVATE:
                raise ValueError(
                    f"Cannot set always_activate: already at cap ({current_count}/{MAX_ALWAYS_ACTIVATE}). "
                    f"Unset an existing always-activate concept first."
                )
        cursor = conn.execute("UPDATE concepts SET always_activate = ? WHERE id = ?", (1 if value else 0, concept_id))
        return cursor.rowcount > 0


# --- Firmware (P0-5) ---


def load_firmware() -> list:
    """Load all firmware entries.

    Returns list of dicts with id, summary, category, firmware_version.
    Called by conversation_turn to inject static operational knowledge.
    """
    with required_context_read_db("load_firmware") as conn:
        rows = conn.execute("""
            SELECT id, summary, category, firmware_version
            FROM firmware
            ORDER BY category, id
        """).fetchall()

    return [
        {
            "id": row["id"],
            "summary": row["summary"],
            "category": row["category"],
            "firmware_version": row["firmware_version"],
        }
        for row in rows
    ]


def save_firmware(firmware_id: str, summary: str, category: str, firmware_version: str) -> None:
    """Upsert a firmware entry. Called only by seed_firmware.py on server startup."""
    now = _utc_now_iso()
    with _conn._db() as conn:
        conn.execute(
            """
            INSERT INTO firmware (id, summary, category, firmware_version, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                summary = excluded.summary,
                category = excluded.category,
                firmware_version = excluded.firmware_version,
                updated_at = excluded.updated_at
        """,
            (firmware_id, summary, category, firmware_version, now, now),
        )


def count_concepts_by_type_tier(since_iso: str = None) -> dict:
    """Count active concepts grouped by abstraction tier. Internal analytics.

    Returns: {
        'L1_observations': int,  (observation, pattern, goal, constraint)
        'L3_abstractions': int,  (principle, method, heuristic, cognitive_strategy)
        'L2_decisions': int,     (decision)
        'total': int,
        'ratio': float,          (L3 / total, 0.0 if total=0)
    }

    If since_iso is provided, only counts concepts created after that date.
    RETRO-001: Used to detect when retrospective is needed.
    """
    L1_TYPES = ("observation", "pattern", "goal", "constraint")
    L2_TYPES = ("decision",)
    L3_TYPES = ("principle", "method", "heuristic", "cognitive_strategy")

    where_clause = "WHERE status = 'active'"
    params = []
    if since_iso:
        where_clause += " AND created_at >= ?"
        params.append(since_iso)

    with read_snapshot_db("count_concepts_by_type_tier") as conn:
        rows = conn.execute(
            f"""
            SELECT concept_type, COUNT(*) as cnt
            FROM concepts
            {where_clause}
            GROUP BY concept_type
        """,
            params,
        ).fetchall()

    counts = {row["concept_type"]: row["cnt"] for row in rows}

    l1 = sum(counts.get(t, 0) for t in L1_TYPES)
    l2 = sum(counts.get(t, 0) for t in L2_TYPES)
    l3 = sum(counts.get(t, 0) for t in L3_TYPES)
    total = l1 + l2 + l3

    return {
        "L1_observations": l1,
        "L2_decisions": l2,
        "L3_abstractions": l3,
        "total": total,
        "ratio": round(l3 / max(total, 1), 3),
    }


def get_metadata(key: str) -> str | None:
    """Get a metadata value by key."""
    with read_snapshot_db("get_metadata") as conn:
        row = conn.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_metadata(key: str, value: str) -> None:
    """Set a metadata value (upsert)."""
    with _conn._db() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO metadata (key, value, updated_at)
            VALUES (?, ?, ?)
        """,
            (key, value, _utc_now_iso()),
        )

def get_high_authority_concepts_by_ka(knowledge_area: str, limit: int = 3) -> list[dict]:
    """Get highest-authority active concepts for a knowledge area.

    Used by S1.7 cross-domain query expansion and S4.2 cross-domain injection.
    Returns concepts ordered by authority_score descending, excluding
    SUPERSEDED/STALE/CONTRADICTED concepts.

    Args:
        knowledge_area: The knowledge area to query.
        limit: Maximum number of concepts to return.

    Returns:
        List of dicts with 'id' and 'summary' keys.
    """
    with read_snapshot_db("get_high_authority_concepts_by_ka") as conn:
        rows = conn.execute(
            """SELECT id, summary FROM concepts
               WHERE status = 'active' AND knowledge_area = ?
               AND currency_status NOT IN ('SUPERSEDED', 'STALE', 'CONTRADICTED')
               AND confidence >= 0.5
               ORDER BY authority_score DESC NULLS LAST
               LIMIT ?""",
            (knowledge_area, limit),
        ).fetchall()
    return [{"id": r[0], "summary": r[1]} for r in rows]

VERBATIM_BUDGET_PER_CONCEPT = 10_000  # chars (~2.5K tokens)
VERBATIM_BUDGET_TOTAL = 50_000_000  # 50MB total across all concepts
FRAGMENT_KEYWORD_CAP = 200  # max chars of keywords per concept
_SQL_STOPWORDS = frozenset({
    "SELECT", "FROM", "WHERE", "AND", "OR", "NOT", "IN", "IS", "NULL",
    "INSERT", "INTO", "VALUES", "UPDATE", "SET", "DELETE", "CREATE",
    "TABLE", "INDEX", "DROP", "ALTER", "ADD", "COLUMN", "PRIMARY",
    "KEY", "DEFAULT", "INTEGER", "TEXT", "REAL", "BLOB", "IF", "EXISTS",
    "JOIN", "LEFT", "RIGHT", "INNER", "OUTER", "ON", "AS", "ORDER",
    "BY", "GROUP", "HAVING", "LIMIT", "OFFSET", "UNION", "ALL",
    "CASE", "WHEN", "THEN", "ELSE", "END", "LIKE", "BETWEEN",
    "TRUE", "FALSE", "WITH", "DISTINCT", "COUNT", "SUM", "AVG",
    "MIN", "MAX", "ASC", "DESC", "CAST", "VARCHAR", "BOOLEAN",
    "THE", "FOR", "THIS", "THAT", "WAS", "ARE", "BUT", "HAS",
})
