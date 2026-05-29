"""Compound Knowledge Objects (CKOs) — Layer 4.

CKOs represent compound knowledge: a competitive analysis is 15 interconnected
concepts. Currently, retrieving "competitive analysis" surfaces fragments.
CKOs retrieve the coherent whole.

CKO Structure:
  - Ordered list of concept_ids (constituent concepts)
  - Synthesis: 500-2000 char coherent summary
  - Authority: MAX of constituent authorities (§4.3)
  - Currency: Weighted average by authority (§4.2)
  - Coherence bonus in retrieval scoring (§4.6)

Lifecycle:
  - active → degraded (1+ constituents stale) → stale (>50% stale) → archived
  - CKOs with <3 accesses in 30 days → archived
  - Overlapping CKOs (>60% shared constituents) → merge candidate
  - Max 10 active CKOs per knowledge_area

Reference: COGNITIVE_GOVERNANCE_ARCHITECTURE_v1.3.md §Layer 4
"""

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from app.core.datetime_utils import _utc_now, _utc_now_iso

logger = logging.getLogger(__name__)


# =============================================================================
# Schema — CKO storage table
# =============================================================================

CKO_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS compound_knowledge_objects (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    cko_type TEXT NOT NULL DEFAULT 'analysis',
    concept_ids TEXT NOT NULL,
    synthesis TEXT NOT NULL,
    knowledge_area TEXT NOT NULL DEFAULT 'general',
    confidence REAL DEFAULT 0.5,
    currency REAL DEFAULT 1.0,
    authority REAL DEFAULT 0.5,
    embedding_cache BLOB,
    status TEXT DEFAULT 'active',
    constituent_count INTEGER DEFAULT 0,
    degraded_constituents TEXT DEFAULT '[]',
    access_count INTEGER DEFAULT 0,
    last_accessed TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cko_status ON compound_knowledge_objects(status);
CREATE INDEX IF NOT EXISTS idx_cko_area ON compound_knowledge_objects(knowledge_area);
CREATE INDEX IF NOT EXISTS idx_cko_authority ON compound_knowledge_objects(authority);
"""

# Valid CKO types per spec
CKO_TYPES = {"analysis", "plan", "assessment", "investigation"}

# Max active CKOs per knowledge area
MAX_CKOS_PER_AREA = 10

# CKO budget in context window
CKO_CONTEXT_SLOTS = 3


# =============================================================================
# CKO Model
# =============================================================================


@dataclass
class CKO:
    """Compound Knowledge Object."""

    id: str
    title: str
    cko_type: str
    concept_ids: list[str]
    synthesis: str
    knowledge_area: str
    confidence: float = 0.5
    currency: float = 1.0
    authority: float = 0.5
    status: str = "active"
    constituent_count: int = 0
    degraded_constituents: list[str] = field(default_factory=list)
    access_count: int = 0
    last_accessed: str | None = None
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "cko_type": self.cko_type,
            "concept_ids": self.concept_ids,
            "synthesis": self.synthesis,
            "knowledge_area": self.knowledge_area,
            "confidence": round(self.confidence, 4),
            "currency": round(self.currency, 4),
            "authority": round(self.authority, 4),
            "status": self.status,
            "constituent_count": self.constituent_count,
            "degraded_constituents": self.degraded_constituents,
            "access_count": self.access_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


# =============================================================================
# CKO Creation
# =============================================================================


def ensure_cko_table(conn: sqlite3.Connection) -> None:
    """Create CKO table if not exists."""
    conn.executescript(CKO_TABLE_SQL)


def _is_missing_cko_table_error(exc: sqlite3.OperationalError) -> bool:
    return "no such table: compound_knowledge_objects" in str(exc)


def create_cko(
    conn: sqlite3.Connection,
    title: str,
    concept_ids: list[str],
    synthesis: str,
    knowledge_area: str = "general",
    cko_type: str = "analysis",
) -> CKO:
    """Create a new Compound Knowledge Object.

    Args:
        conn: DB connection
        title: CKO title
        concept_ids: Ordered list of constituent concept IDs
        synthesis: 500-2000 char coherent summary
        knowledge_area: Domain
        cko_type: One of analysis|plan|assessment|investigation

    Returns:
        Created CKO with computed authority/currency
    """
    ensure_cko_table(conn)

    if cko_type not in CKO_TYPES:
        cko_type = "analysis"

    # Enforce max CKOs per area
    active_in_area = conn.execute(
        """SELECT COUNT(*) FROM compound_knowledge_objects
           WHERE knowledge_area = ? AND status = 'active'""",
        (knowledge_area,),
    ).fetchone()[0]

    if active_in_area >= MAX_CKOS_PER_AREA:
        # Archive oldest by access
        conn.execute(
            """UPDATE compound_knowledge_objects SET status = 'archived', updated_at = ?
               WHERE id = (
                   SELECT id FROM compound_knowledge_objects
                   WHERE knowledge_area = ? AND status = 'active'
                   ORDER BY COALESCE(last_accessed, created_at) ASC LIMIT 1
               )""",
            (_utc_now_iso(), knowledge_area),
        )
        logger.info(f"CKO area limit reached for {knowledge_area}, archived oldest")

    # Compute authority (max) and currency (weighted avg) from constituents
    authority, currency, confidence = _compute_cko_scores(conn, concept_ids)
    degraded = _find_degraded_constituents(conn, concept_ids)

    # Determine initial status
    status = "active"
    if len(degraded) > len(concept_ids) * 0.5:
        status = "stale"
    elif len(degraded) > 0:
        status = "degraded"

    now = _utc_now_iso()
    cko_id = f"cko_{uuid.uuid4().hex[:12]}"

    cko = CKO(
        id=cko_id,
        title=title,
        cko_type=cko_type,
        concept_ids=concept_ids,
        synthesis=synthesis,
        knowledge_area=knowledge_area,
        confidence=confidence,
        currency=currency,
        authority=authority,
        status=status,
        constituent_count=len(concept_ids),
        degraded_constituents=degraded,
        access_count=0,
        created_at=now,
        updated_at=now,
    )

    conn.execute(
        """INSERT INTO compound_knowledge_objects
           (id, title, cko_type, concept_ids, synthesis, knowledge_area,
            confidence, currency, authority, status, constituent_count,
            degraded_constituents, access_count, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            cko.id,
            cko.title,
            cko.cko_type,
            json.dumps(cko.concept_ids),
            cko.synthesis,
            cko.knowledge_area,
            cko.confidence,
            cko.currency,
            cko.authority,
            cko.status,
            cko.constituent_count,
            json.dumps(cko.degraded_constituents),
            cko.access_count,
            cko.created_at,
            cko.updated_at,
        ),
    )
    conn.commit()

    logger.info(
        f"Created CKO {cko_id}: {title} ({len(concept_ids)} concepts, "
        f"auth={authority:.2f}, curr={currency:.2f}, status={status})"
    )
    return cko


# =============================================================================
# Score Computation (§4.2, §4.3)
# =============================================================================


def _compute_cko_scores(conn: sqlite3.Connection, concept_ids: list[str]) -> tuple[float, float, float]:
    """Compute CKO authority, currency, and confidence from constituents.

    Authority = MAX(constituent.authority)  [§4.3]
      Rationale: CKO with even one decision should present at that level.

    Currency = weighted_average(constituent.currency, weights=constituent.authority)  [§4.2]
      Rationale: One stale observation doesn't invalidate a mostly-fresh CKO.
      But one stale decision carries more weight (via authority weighting).

    Confidence = weighted_average(constituent.confidence, weights=constituent.authority)

    Returns: (authority, currency, confidence)
    """
    if not concept_ids:
        return 0.5, 1.0, 0.5

    placeholders = ",".join(["?"] * len(concept_ids))
    rows = conn.execute(
        f"""SELECT COALESCE(authority_score, confidence, 0.5) as auth,
                   COALESCE(currency_score, 1.0) as curr,
                   COALESCE(confidence, 0.5) as conf
            FROM concepts
            WHERE id IN ({placeholders}) AND status != 'deleted'""",
        concept_ids,
    ).fetchall()

    if not rows:
        return 0.5, 1.0, 0.5

    max_authority = max(r[0] for r in rows)

    # Weighted averages by authority
    total_weight = sum(r[0] for r in rows) or 1.0
    weighted_currency = sum(r[1] * r[0] for r in rows) / total_weight
    weighted_confidence = sum(r[2] * r[0] for r in rows) / total_weight

    return max_authority, weighted_currency, weighted_confidence


def _find_degraded_constituents(conn: sqlite3.Connection, concept_ids: list[str]) -> list[str]:
    """Find constituents that are STALE or SUPERSEDED."""
    if not concept_ids:
        return []

    placeholders = ",".join(["?"] * len(concept_ids))
    rows = conn.execute(
        f"""SELECT id FROM concepts
            WHERE id IN ({placeholders})
            AND (currency_status IN ('STALE', 'SUPERSEDED')
                 OR (currency_score IS NOT NULL AND currency_score < 0.30))
            AND status != 'deleted'""",
        concept_ids,
    ).fetchall()

    return [r[0] for r in rows]


# =============================================================================
# Row Deserialization Helper
# =============================================================================


def _row_to_cko(row: tuple) -> CKO:
    """Convert a DB row tuple to a CKO object. Single source of truth for deserialization."""
    return CKO(
        id=row[0],
        title=row[1],
        cko_type=row[2],
        concept_ids=json.loads(row[3]) if row[3] else [],
        synthesis=row[4],
        knowledge_area=row[5],
        confidence=row[6] or 0.5,
        currency=row[7] or 1.0,
        authority=row[8] or 0.5,
        status=row[9],
        constituent_count=row[10] or 0,
        degraded_constituents=json.loads(row[11]) if row[11] else [],
        access_count=row[12] or 0,
        last_accessed=row[13],
        created_at=row[14],
        updated_at=row[15],
    )


# =============================================================================
# CKO Retrieval (§4.6)
# =============================================================================


def search_ckos(
    conn: sqlite3.Connection,
    query_area: str | None = None,
    max_results: int = CKO_CONTEXT_SLOTS,
    *,
    record_access: bool = True,
    ensure_table: bool = True,
) -> list[CKO]:
    """Retrieve CKOs for context assembly.

    CKOs participate in retrieval with coherence bonus:
      cko_score = authority * 0.25 + currency * 0.20
                + confidence * 0.15 + coherence_bonus * 0.40

    Coherence bonus = constituent_count / 20 (more concepts = more coherent)

    Args:
        conn: DB connection
        query_area: Optional knowledge_area filter
        max_results: Max CKOs to return (default: 3 per spec)
        record_access: Whether to update access counters for returned CKOs.
        ensure_table: Whether to create the CKO table if missing.

    Returns:
        Scored and sorted list of CKOs
    """
    if ensure_table:
        ensure_cko_table(conn)

    where_clause = "WHERE status IN ('active', 'degraded')"
    params = []
    if query_area:
        where_clause += " AND knowledge_area = ?"
        params.append(query_area)

    try:
        rows = conn.execute(
            f"""SELECT id, title, cko_type, concept_ids, synthesis, knowledge_area,
                       confidence, currency, authority, status, constituent_count,
                       degraded_constituents, access_count, last_accessed,
                       created_at, updated_at
                FROM compound_knowledge_objects
                {where_clause}
                ORDER BY authority DESC
                LIMIT ?""",
            params + [max_results * 3],  # Over-fetch for scoring
        ).fetchall()
    except sqlite3.OperationalError as exc:
        if not ensure_table and _is_missing_cko_table_error(exc):
            return []
        raise

    ckos = [_row_to_cko(row) for row in rows]

    # Score and sort
    scored = []
    for cko in ckos:
        coherence_bonus = min(cko.constituent_count / 20.0, 1.0)
        score = cko.authority * 0.25 + cko.currency * 0.20 + cko.confidence * 0.15 + coherence_bonus * 0.40

        # Penalty for degraded status
        if cko.status == "degraded":
            score *= 0.85
        scored.append((score, cko))

    scored.sort(key=lambda x: x[0], reverse=True)

    if not record_access:
        return [cko for _, cko in scored[:max_results]]

    # Record access for lifecycle management
    now = _utc_now_iso()
    result_ckos = []
    for score, cko in scored[:max_results]:
        conn.execute(
            """UPDATE compound_knowledge_objects
               SET access_count = access_count + 1, last_accessed = ?
               WHERE id = ?""",
            (now, cko.id),
        )
        cko.access_count += 1
        cko.last_accessed = now
        result_ckos.append(cko)

    if result_ckos:
        conn.commit()

    return result_ckos


# =============================================================================
# CKO Loading & Refresh
# =============================================================================


def load_cko(conn: sqlite3.Connection, cko_id: str, *, ensure_table: bool = True) -> CKO | None:
    """Load a single CKO by ID. Returns None if not found."""
    if ensure_table:
        ensure_cko_table(conn)

    try:
        row = conn.execute(
            """SELECT id, title, cko_type, concept_ids, synthesis, knowledge_area,
                      confidence, currency, authority, status, constituent_count,
                      degraded_constituents, access_count, last_accessed,
                      created_at, updated_at
               FROM compound_knowledge_objects WHERE id = ?""",
            (cko_id,),
        ).fetchone()
    except sqlite3.OperationalError as exc:
        if not ensure_table and _is_missing_cko_table_error(exc):
            return None
        raise

    if not row:
        return None

    return _row_to_cko(row)


def refresh_cko(conn: sqlite3.Connection, cko_id: str) -> CKO | None:
    """Recompute CKO scores and status from current constituent state.

    Called after constituent concepts are mutated (evolved, decayed, etc.).
    Follows the lifecycle: active → degraded → stale → archived.
    """
    cko = load_cko(conn, cko_id)
    if not cko:
        return None

    # Recompute from current constituent scores
    authority, currency, confidence = _compute_cko_scores(conn, cko.concept_ids)
    degraded = _find_degraded_constituents(conn, cko.concept_ids)

    # Determine new status
    degraded_ratio = len(degraded) / max(len(cko.concept_ids), 1)
    if degraded_ratio > 0.5:
        new_status = "stale"
    elif len(degraded) > 0:
        new_status = "degraded"
    else:
        new_status = "active"

    # Don't resurrect archived CKOs
    if cko.status == "archived":
        new_status = "archived"

    now = _utc_now_iso()
    conn.execute(
        """UPDATE compound_knowledge_objects
           SET authority = ?, currency = ?, confidence = ?,
               degraded_constituents = ?, status = ?, updated_at = ?
           WHERE id = ?""",
        (authority, currency, confidence, json.dumps(degraded), new_status, now, cko_id),
    )
    conn.commit()

    cko.authority = authority
    cko.currency = currency
    cko.confidence = confidence
    cko.degraded_constituents = degraded
    cko.status = new_status
    cko.updated_at = now

    if cko.status != new_status:
        logger.info(
            f"CKO {cko_id} status: {cko.status} → {new_status} ({len(degraded)}/{len(cko.concept_ids)} degraded)"
        )

    return cko


def update_cko_synthesis(
    conn: sqlite3.Connection,
    cko_id: str,
    new_synthesis: str,
    new_concept_ids: list[str] | None = None,
) -> CKO | None:
    """Update a CKO's synthesis and optionally its constituent list.

    Used when concepts are added/removed from the CKO.
    """
    cko = load_cko(conn, cko_id)
    if not cko:
        return None

    concept_ids = new_concept_ids if new_concept_ids is not None else cko.concept_ids
    authority, currency, confidence = _compute_cko_scores(conn, concept_ids)
    degraded = _find_degraded_constituents(conn, concept_ids)

    degraded_ratio = len(degraded) / max(len(concept_ids), 1)
    status = "stale" if degraded_ratio > 0.5 else ("degraded" if degraded else "active")
    if cko.status == "archived":
        status = "archived"

    now = _utc_now_iso()
    conn.execute(
        """UPDATE compound_knowledge_objects
           SET synthesis = ?, concept_ids = ?, constituent_count = ?,
               authority = ?, currency = ?, confidence = ?,
               degraded_constituents = ?, status = ?, updated_at = ?
           WHERE id = ?""",
        (
            new_synthesis,
            json.dumps(concept_ids),
            len(concept_ids),
            authority,
            currency,
            confidence,
            json.dumps(degraded),
            status,
            now,
            cko_id,
        ),
    )
    conn.commit()

    cko.synthesis = new_synthesis
    cko.concept_ids = concept_ids
    cko.constituent_count = len(concept_ids)
    cko.authority = authority
    cko.currency = currency
    cko.confidence = confidence
    cko.degraded_constituents = degraded
    cko.status = status
    cko.updated_at = now

    return cko


# =============================================================================
# CKO Lifecycle Manager (§4.5)
# =============================================================================


def run_cko_lifecycle(conn: sqlite3.Connection) -> dict[str, Any]:
    """Run CKO lifecycle management.

    1. Refresh all active/degraded CKOs (recompute from constituent state)
    2. Archive CKOs with <3 accesses in 30 days
    3. Identify merge candidates (>60% shared constituents)

    Returns summary of actions taken.
    """
    ensure_cko_table(conn)
    actions = {"refreshed": 0, "archived": 0, "merge_candidates": []}

    # 1. Refresh all non-archived CKOs
    rows = conn.execute(
        """SELECT id FROM compound_knowledge_objects
           WHERE status IN ('active', 'degraded')"""
    ).fetchall()

    for (cko_id,) in rows:
        refresh_cko(conn, cko_id)
        actions["refreshed"] += 1

    # 2. Archive low-access CKOs
    cutoff = (_utc_now() - timedelta(days=30)).isoformat()
    low_access = conn.execute(
        """SELECT id, title, access_count FROM compound_knowledge_objects
           WHERE status IN ('active', 'degraded')
           AND access_count < 3
           AND (last_accessed IS NULL OR last_accessed < ?)""",
        (cutoff,),
    ).fetchall()

    now = _utc_now_iso()
    for cko_id, title, access_count in low_access:
        conn.execute(
            """UPDATE compound_knowledge_objects
               SET status = 'archived', updated_at = ?
               WHERE id = ?""",
            (now, cko_id),
        )
        actions["archived"] += 1
        logger.info(f"Archived CKO {cko_id} ({title}): only {access_count} accesses in 30d")

    # Also archive fully stale CKOs
    stale_ckos = conn.execute(
        """SELECT id, title FROM compound_knowledge_objects
           WHERE status = 'stale'"""
    ).fetchall()

    for cko_id, title in stale_ckos:
        conn.execute(
            """UPDATE compound_knowledge_objects
               SET status = 'archived', updated_at = ?
               WHERE id = ?""",
            (now, cko_id),
        )
        actions["archived"] += 1
        logger.info(f"Archived stale CKO {cko_id} ({title})")

    if actions["archived"] > 0:
        conn.commit()

    # 3. Identify merge candidates (>60% shared constituents)
    active_ckos = conn.execute(
        """SELECT id, concept_ids FROM compound_knowledge_objects
           WHERE status IN ('active', 'degraded')"""
    ).fetchall()

    cko_sets = []
    for cko_id, concept_ids_json in active_ckos:
        ids = set(json.loads(concept_ids_json)) if concept_ids_json else set()
        cko_sets.append((cko_id, ids))

    for i in range(len(cko_sets)):
        for j in range(i + 1, len(cko_sets)):
            id_a, set_a = cko_sets[i]
            id_b, set_b = cko_sets[j]
            if not set_a or not set_b:
                continue
            # Overlap uses min-containment (not Jaccard) deliberately:
            # if a small CKO (3 concepts) shares 2 with a large one (20 concepts),
            # min-containment = 2/3 = 67% (merge candidate — small is mostly redundant).
            # Jaccard would give 2/21 = 9.5% (no merge — misleading for subsumption).
            overlap = len(set_a & set_b) / min(len(set_a), len(set_b))
            if overlap > 0.60:
                actions["merge_candidates"].append(
                    {
                        "cko_a": id_a,
                        "cko_b": id_b,
                        "overlap_ratio": round(overlap, 2),
                        "shared_count": len(set_a & set_b),
                    }
                )
                logger.info(f"Merge candidate: {id_a} + {id_b} ({overlap:.0%} overlap)")

    return actions


def list_ckos(
    conn: sqlite3.Connection,
    status: str | None = None,
    knowledge_area: str | None = None,
    limit: int = 50,
    *,
    ensure_table: bool = True,
) -> list[CKO]:
    """List CKOs with optional filters."""
    if ensure_table:
        ensure_cko_table(conn)

    where_parts = []
    params = []
    if status:
        where_parts.append("status = ?")
        params.append(status)
    if knowledge_area:
        where_parts.append("knowledge_area = ?")
        params.append(knowledge_area)

    where_clause = "WHERE " + " AND ".join(where_parts) if where_parts else ""

    try:
        rows = conn.execute(
            f"""SELECT id, title, cko_type, concept_ids, synthesis, knowledge_area,
                       confidence, currency, authority, status, constituent_count,
                       degraded_constituents, access_count, last_accessed,
                       created_at, updated_at
                FROM compound_knowledge_objects
                {where_clause}
                ORDER BY authority DESC
                LIMIT ?""",
            params + [limit],
        ).fetchall()
    except sqlite3.OperationalError as exc:
        if not ensure_table and _is_missing_cko_table_error(exc):
            return []
        raise

    return [_row_to_cko(r) for r in rows]


def delete_cko(conn: sqlite3.Connection, cko_id: str) -> bool:
    """Permanently delete a CKO. Use archive via lifecycle for normal removal."""
    result = conn.execute("DELETE FROM compound_knowledge_objects WHERE id = ?", (cko_id,))
    conn.commit()
    return result.rowcount > 0
