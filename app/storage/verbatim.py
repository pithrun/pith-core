"""Storage sub-module: verbatim.

Verbatim fragment CRUD and FTS5 verbatim search.
Extracted from storage/__init__.py during Item 2b decomposition.
"""
import json
import logging
import os
import re
from contextlib import contextmanager

import app.storage.connection as _conn
from app.storage.connection import diagnostic_snapshot_db, read_snapshot_db

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

logger = logging.getLogger(__name__)


@contextmanager
def _verbatim_search_db(label: str, busy_timeout_ms: int | None = None):
    if busy_timeout_ms is None:
        with read_snapshot_db(label) as conn:
            yield conn
        return
    with diagnostic_snapshot_db(label, busy_timeout_ms=max(0, int(busy_timeout_ms))) as conn:
        yield conn


@contextmanager
def _fresh_immediate_connection(operation: str):
    """Yield a fresh write transaction for repair jobs that must see latest WAL state."""
    del operation
    conn = _conn.open_owned_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        yield conn
        if getattr(conn, "in_transaction", False):
            conn.commit()
    except Exception:
        if getattr(conn, "in_transaction", False):
            conn.rollback()
        raise
    finally:
        conn.close()


def _sync_fts5_verbatim(
    conn,
    fragment_id: str,
    concept_id: str,
    content: str | None = None,
    delete: bool = False,
) -> bool:
    """Sync a verbatim fragment to the FTS5 verbatim index.

    RETRIEVAL-070: Keeps fts_verbatim in sync with verbatim_fragments table.
    Extracts USER portion for user_content column, stores full text in full_content.
    """
    try:
        if delete:
            conn.execute("DELETE FROM fts_verbatim WHERE fragment_id = ?", (fragment_id,))
            return True
        elif content:
            # Extract USER portion from INGEST-038 format: [USER] msg\n\n[ASSISTANT] response
            user_content = content
            full_content = content
            if '[USER]' in content and '\n\n[ASSISTANT]' in content:
                user_content = content.split('\n\n[ASSISTANT]')[0].replace('[USER] ', '')

            # Upsert: delete old then insert
            conn.execute("DELETE FROM fts_verbatim WHERE fragment_id = ?", (fragment_id,))
            conn.execute(
                "INSERT INTO fts_verbatim(fragment_id, concept_id, user_content, full_content) "
                "VALUES (?, ?, ?, ?)",
                (fragment_id, concept_id, user_content, full_content),
            )
            return True
        return False
    except Exception as e:
        logger.warning(f"RETRIEVAL-070: FTS5 verbatim sync failed for {fragment_id}: {e}")
        return False

def repair_fts_verbatim(limit: int = 200) -> dict:
    """DATA-069: Find canonical conversation fragments missing from fts_verbatim and backfill.

    Runs during maintenance phase 9. Idempotent — safe to run repeatedly.
    Returns dict with counts for monitoring.
    """
    repaired = 0
    errors = 0
    scanned = 0
    with _fresh_immediate_connection(operation="repair_fts_verbatim") as conn:
        try:
            try:
                existing_ids = {
                    row[0]
                    for row in conn.execute(
                        "SELECT c0 FROM fts_verbatim_content WHERE c0 IS NOT NULL"
                    ).fetchall()
                }
            except Exception:
                existing_ids = {
                    row[0]
                    for row in conn.execute(
                        "SELECT fragment_id FROM fts_verbatim WHERE fragment_id IS NOT NULL"
                    ).fetchall()
                }
            candidates = conn.execute(
                """
                SELECT vf.id, vf.concept_id, vf.content
                FROM verbatim_fragments vf
                JOIN concepts c ON c.id = vf.concept_id
                WHERE vf.content IS NOT NULL
                  AND vf.char_count > 0
                  AND vf.fragment_type = 'conversation'
                  AND c.status = 'active'
                """,
            ).fetchall()
            missing = [
                (fragment_id, concept_id, content)
                for fragment_id, concept_id, content in candidates
                if fragment_id not in existing_ids
            ][:limit]
            scanned = len(missing)
            rows = []
            for fragment_id, concept_id, content in missing:
                user_content = content
                if "[USER]" in content and "\n\n[ASSISTANT]" in content:
                    user_content = content.split("\n\n[ASSISTANT]")[0].replace("[USER] ", "")
                rows.append((fragment_id, concept_id, user_content, content))
            conn.executemany(
                """
                INSERT INTO fts_verbatim(fragment_id, concept_id, user_content, full_content)
                VALUES (?, ?, ?, ?)
                """,
                rows,
            )
            repaired = scanned
        except Exception as e:
            logger.warning("DATA-069: FTS repair batch failed: %s", e)
            errors += 1

    logger.info(
        "DATA-069: FTS verbatim repair — %d repaired, %d errors",
        repaired, errors,
    )
    return {"repaired": repaired, "errors": errors, "scanned": scanned}


def search_verbatim_fts5(
    query_terms: list[str],
    limit: int = 30,
    column: str = 'user_content',
    busy_timeout_ms: int | None = None,
) -> list[dict]:
    """Search verbatim fragments via FTS5 keyword matching.

    RETRIEVAL-070: Returns matching verbatim fragments with BM25 scores.
    Used as PATH B in the two-path retrieval architecture.

    Args:
        query_terms: List of keyword terms to search for
        limit: Maximum results to return
        column: Which column to search ('user_content' or 'full_content')

    Returns:
        List of dicts with fragment_id, concept_id, content, bm25_score
    """
    # Sanitize terms for FTS5
    import re as _re
    safe_terms = []
    for term in query_terms:
        parts = term.split('-') if '-' in term else [term]
        for part in parts:
            cleaned = _re.sub(r'[^\w]', '', part)
            if cleaned and len(cleaned) > 2:
                safe_terms.append(cleaned)

    if not safe_terms:
        return []

    fts_query = " OR ".join(safe_terms)

    try:
        with _verbatim_search_db("search_verbatim_fts5", busy_timeout_ms=busy_timeout_ms) as conn:
            # Check if fts_verbatim table exists
            table_check = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='fts_verbatim'"
            ).fetchone()
            if not table_check:
                logger.debug("RETRIEVAL-070: fts_verbatim table not found, skipping verbatim search")
                return []

            rows = conn.execute(f"""
                SELECT fv.fragment_id, fv.concept_id, fv.{column}, fv.full_content,
                       bm25(fts_verbatim) as bm25_score
                FROM fts_verbatim fv
                WHERE fts_verbatim MATCH ?
                ORDER BY bm25(fts_verbatim)
                LIMIT ?
            """, (fts_query, limit)).fetchall()

            results = []
            seen_concepts = set()  # Deduplicate by concept_id
            for row in rows:
                fid, cid, user_text, full_text, score = row
                if cid in seen_concepts:
                    continue
                seen_concepts.add(cid)
                results.append({
                    'fragment_id': fid,
                    'concept_id': cid,
                    'user_content': user_text,
                    'full_content': full_text,
                    'bm25_score': score,
                })

        logger.info(f"RETRIEVAL-070: Verbatim FTS5 search '{fts_query}' → {len(results)} concepts "
                     f"({len(rows)} raw hits, deduplicated)")
        return results

    except Exception as e:
        logger.warning(f"RETRIEVAL-070: Verbatim FTS5 search failed (non-fatal): {e}")
        return []


def search_verbatim_fts5_dual(
    query_terms: list[str],
    limit: int = 30,
    w_user: float = 1.0,
    w_full: float = 0.7,
    busy_timeout_ms: int | None = None,
) -> list[dict]:
    """RETRIEVAL-080: Weighted dual-column FTS5 search over verbatim fragments.

    Searches BOTH user_content and full_content columns, returning results
    with column-aware weights. Fixes ASSISTANT_RECALL_MISS: facts originating
    in assistant responses (only in full_content) were invisible to R070's
    single-column search.

    Args:
        query_terms: List of keyword terms to search for
        limit: Maximum results to return
        w_user: Weight multiplier for user_content matches (default 1.0)
        w_full: Weight multiplier for full_content-only matches (default 0.7)

    Returns:
        List of dicts with fragment_id, concept_id, content, bm25_score,
        match_column ('user' | 'full' | 'both')
    """
    import re as _re
    safe_terms = []
    _ordinal_re = _re.compile(r'^(\d+)(?:st|nd|rd|th)$', _re.IGNORECASE)
    for term in query_terms:
        parts = term.split('-') if '-' in term else [term]
        for part in parts:
            cleaned = _re.sub(r'[^\w]', '', part)
            if cleaned and len(cleaned) > 2:
                safe_terms.append(cleaned)
                # INGEST-041 Change D: Expand ordinals — "27th" → also search "27"
                _ord_m = _ordinal_re.match(cleaned)
                if _ord_m:
                    safe_terms.append(_ord_m.group(1))

    if not safe_terms:
        return []

    fts_query = " OR ".join(safe_terms)

    try:
        with _verbatim_search_db("search_verbatim_fts5_dual", busy_timeout_ms=busy_timeout_ms) as conn:
            table_check = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='fts_verbatim'"
            ).fetchone()
            if not table_check:
                logger.debug("RETRIEVAL-080: fts_verbatim table not found")
                return []

            # Search user_content column
            user_rows = conn.execute("""
                SELECT fv.fragment_id, fv.concept_id, fv.user_content, fv.full_content,
                       bm25(fts_verbatim) as bm25_score
                FROM fts_verbatim fv
                WHERE fts_verbatim MATCH ?
                ORDER BY bm25(fts_verbatim)
                LIMIT ?
            """, (f"user_content:{fts_query}", limit)).fetchall()

            # Search full_content column
            full_rows = conn.execute("""
                SELECT fv.fragment_id, fv.concept_id, fv.user_content, fv.full_content,
                       bm25(fts_verbatim) as bm25_score
                FROM fts_verbatim fv
                WHERE fts_verbatim MATCH ?
                ORDER BY bm25(fts_verbatim)
                LIMIT ?
            """, (f"full_content:{fts_query}", limit)).fetchall()

        # Merge: per concept_id, pick best weighted score
        concept_best: dict[str, dict] = {}

        for row in user_rows:
            fid, cid, user_text, full_text, score = row
            weighted = abs(score) * w_user  # BM25 scores are negative in FTS5
            if cid not in concept_best or weighted > concept_best[cid]['_weighted']:
                concept_best[cid] = {
                    'fragment_id': fid,
                    'concept_id': cid,
                    'user_content': user_text,
                    'full_content': full_text,
                    'bm25_score': score,
                    'match_column': 'user',
                    '_weighted': weighted,
                }

        for row in full_rows:
            fid, cid, user_text, full_text, score = row
            weighted = abs(score) * w_full
            if cid not in concept_best:
                concept_best[cid] = {
                    'fragment_id': fid,
                    'concept_id': cid,
                    'user_content': user_text,
                    'full_content': full_text,
                    'bm25_score': score,
                    'match_column': 'full',
                    '_weighted': weighted,
                }
            else:
                # Concept found in BOTH columns — always mark 'both'
                concept_best[cid]['match_column'] = 'both'
                if weighted > concept_best[cid]['_weighted']:
                    concept_best[cid]['_weighted'] = weighted

        # Sort by weighted score descending, strip internal field
        results = sorted(concept_best.values(), key=lambda x: x['_weighted'], reverse=True)[:limit]
        for r in results:
            del r['_weighted']

        n_user_only = sum(1 for r in results if r['match_column'] == 'user')
        n_full_only = sum(1 for r in results if r['match_column'] == 'full')
        n_both = sum(1 for r in results if r['match_column'] == 'both')
        logger.info(
            f"RETRIEVAL-080: Dual-column FTS5 '{fts_query}' → {len(results)} concepts "
            f"(user={n_user_only}, full={n_full_only}, both={n_both})"
        )
        return results

    except Exception as e:
        logger.warning(f"RETRIEVAL-080: Dual-column FTS5 search failed (non-fatal): {e}")
        return []

def extract_fragment_keywords(content: str, fragment_type: str = "text") -> str:
    """Extract distinguishing technical keywords from fragment content.

    INGEST-037 Layer 4: Returns space-separated keyword string suitable for
    appending to concept searchable_text and FTS5 summary.

    Prioritizes: SQL function names, table/column identifiers, CamelCase,
    UPPER_CASE, and snake_case tokens. Filters binary content, stopwords,
    and common English words.

    Max output: FRAGMENT_KEYWORD_CAP chars.
    """

    if not content or not content.strip():
        return ""

    # Skip binary-looking content
    if "bytearray" in content or "\\x" in content[:100]:
        return ""

    tokens: list[str] = []

    # SQL function names: UPPER_CASE identifiers with optional parens
    tokens.extend(re.findall(r'\b([A-Z][A-Z_]{2,})\b', content))

    # Table/column names after SQL keywords
    for match in re.finditer(r'(?:FROM|JOIN|INTO|TABLE|UPDATE)\s+(\w+)', content, re.IGNORECASE):
        tok = match.group(1)
        if len(tok) >= 3:
            tokens.append(tok)

    # CamelCase identifiers
    tokens.extend(re.findall(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b', content))

    # snake_case identifiers (3+ chars, at least one underscore)
    tokens.extend(re.findall(r'\b([a-z][a-z0-9]*_[a-z0-9_]+)\b', content))

    # UPPER_SNAKE_CASE identifiers (like NOAA_GFS0P25)
    tokens.extend(re.findall(r'\b([A-Z][A-Z0-9]*_[A-Z0-9_]+)\b', content))

    # Filter: remove stopwords, short tokens, and numeric-only tokens
    filtered = []
    seen = set()
    for tok in tokens:
        upper = tok.upper()
        if upper in _SQL_STOPWORDS:
            continue
        if len(tok) < 3:
            continue
        # Skip purely numeric tokens or scientific notation
        if re.match(r'^[\d.eE+\-]+$', tok):
            continue
        if upper not in seen:
            seen.add(upper)
            filtered.append(tok)

    # Build keyword string, respecting cap
    kw_str = ""
    for kw in filtered:
        candidate = f"{kw_str} {kw}".strip() if kw_str else kw
        if len(candidate) > FRAGMENT_KEYWORD_CAP:
            break
        kw_str = candidate

    return kw_str


def _recompute_fragment_keywords(conn, concept_id: str) -> str | None:
    """Recompute fragment_keywords for a concept from all its fragments.

    INGEST-037 Layer 4: Called after fragment save/delete to keep keywords current.
    Returns the new keyword string (or None if no fragments).
    """
    # INGEST-038: Exclude conversation fragments — keyword enrichment is for code/SQL/config only
    rows = conn.execute(
        "SELECT content, fragment_type FROM verbatim_fragments WHERE concept_id = ? AND fragment_type != 'conversation' ORDER BY created_at ASC",
        (concept_id,),
    ).fetchall()

    if not rows:
        return None

    # Collect keywords from all fragments, deduplicate
    all_keywords: list[str] = []
    seen = set()
    for content, ftype in rows:
        kw = extract_fragment_keywords(content or "", ftype or "text")
        for tok in kw.split():
            upper = tok.upper()
            if upper not in seen:
                seen.add(upper)
                all_keywords.append(tok)

    # Build keyword string respecting cap
    kw_str = ""
    for kw in all_keywords:
        candidate = f"{kw_str} {kw}".strip() if kw_str else kw
        if len(candidate) > FRAGMENT_KEYWORD_CAP:
            break
        kw_str = candidate

    return kw_str if kw_str else None


def save_verbatim_fragment(
    concept_id: str,
    fragment_type: str = "text",
    content: str | None = None,
    pointer_uri: str | None = None,
    pointer_meta: dict | None = None,
    evidence_id: str | None = None,
    concept_version: str | None = None,
    inherited_from: str | None = None,
    skip_enrichment: bool = False,
) -> str | None:
    """Store a verbatim fragment for a concept. Returns fragment ID or None if budget exceeded."""
    import hashlib
    import uuid

    char_count = len(content) if content else 0

    # Budget check: per-concept
    with _conn._db() as conn:
        existing = conn.execute(
            "SELECT COALESCE(SUM(char_count), 0) FROM verbatim_fragments WHERE concept_id = ?",
            (concept_id,),
        ).fetchone()[0]
        if existing + char_count > VERBATIM_BUDGET_PER_CONCEPT and char_count > 0:
            logger.warning(
                "INGEST-037: Per-concept verbatim budget exceeded for %s (%d + %d > %d)",
                concept_id, existing, char_count, VERBATIM_BUDGET_PER_CONCEPT,
            )
            return None

        # Dedup via source_hash
        source_hash = None
        if content:
            source_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            dup = conn.execute(
                "SELECT id FROM verbatim_fragments WHERE concept_id = ? AND source_hash = ?",
                (concept_id, source_hash),
            ).fetchone()
            if dup:
                logger.debug("INGEST-037: Dedup — fragment already exists for %s (hash=%s)", concept_id, source_hash[:12])
                return dup[0]

        fragment_id = f"vf_{uuid.uuid4().hex[:16]}"
        pointer_meta_json = json.dumps(pointer_meta) if pointer_meta else None

        conn.execute(
            """INSERT INTO verbatim_fragments
               (id, concept_id, concept_version, evidence_id, fragment_type,
                content, pointer_uri, pointer_meta, char_count, source_hash, inherited_from)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (fragment_id, concept_id, concept_version, evidence_id, fragment_type,
             content, pointer_uri, pointer_meta_json, char_count, source_hash, inherited_from),
        )

        # RETRIEVAL-070: Sync to FTS5 verbatim index
        if content and fragment_type == 'conversation':
            _sync_fts5_verbatim(conn, fragment_id, concept_id, content)

    logger.debug("INGEST-037: Saved verbatim fragment %s for concept %s (%s, %d chars)",
                 fragment_id, concept_id, fragment_type, char_count)

    # INGEST-037 Layer 4: Recompute fragment keywords after successful save
    # INGEST-038: skip_enrichment=True for conversation fragments (avoids search index noise)
    if not skip_enrichment and os.environ.get("PITH_FRAGMENT_ENRICHMENT", "true").lower() != "false":
        try:
            with _conn._db() as kw_conn:
                new_kw = _recompute_fragment_keywords(kw_conn, concept_id)
                kw_conn.execute(
                    "UPDATE concepts SET fragment_keywords = ? WHERE id = ?",
                    (new_kw, concept_id),
                )
            n_kw = len(new_kw.split()) if new_kw else 0
            logger.info(
                "INGEST-037-L4: enriched concept %s with %d keywords (%d chars)",
                concept_id, n_kw, len(new_kw) if new_kw else 0,
            )
        except Exception as e:
            logger.warning("INGEST-037-L4: keyword enrichment failed for %s: %s", concept_id, e)

    return fragment_id


def _get_fragments_by_ids(fragment_ids: list[str]) -> dict[str, dict]:
    """INGEST-038: Batch-fetch verbatim fragments by ID. Returns {id: fragment_dict}."""
    import json as _json_batch

    if not fragment_ids:
        return {}
    with read_snapshot_db("_get_fragments_by_ids") as conn:
        placeholders = ",".join("?" for _ in fragment_ids)
        rows = conn.execute(
            f"""SELECT id, concept_id, concept_version, evidence_id, fragment_type,
                       content, pointer_uri, pointer_meta, char_count,
                       created_at, source_hash, inherited_from
                FROM verbatim_fragments WHERE id IN ({placeholders})""",
            fragment_ids,
        ).fetchall()
    result = {}
    for r in rows:
        meta = None
        if r[7]:
            try:
                meta = _json_batch.loads(r[7])
            except Exception:
                meta = r[7]
        result[r[0]] = {
            "id": r[0], "concept_id": r[1], "concept_version": r[2],
            "evidence_id": r[3], "fragment_type": r[4], "content": r[5],
            "pointer_uri": r[6], "pointer_meta": meta, "char_count": r[8],
            "created_at": r[9], "source_hash": r[10], "inherited_from": r[11],
        }
    return result


def get_verbatim_fragments(concept_id: str, limit: int = 10) -> list[dict]:
    """Get verbatim fragments for a concept, ordered by creation time."""

    with read_snapshot_db("get_verbatim_fragments") as conn:
        rows = conn.execute(
            """SELECT id, concept_version, evidence_id, fragment_type,
                      content, pointer_uri, pointer_meta, char_count,
                      created_at, source_hash, inherited_from
               FROM verbatim_fragments
               WHERE concept_id = ?
               ORDER BY created_at ASC
               LIMIT ?""",
            (concept_id, limit),
        ).fetchall()

    results = []
    for r in rows:
        meta = None
        if r[6]:
            try:
                meta = json.loads(r[6])
            except Exception:
                meta = r[6]
        results.append({
            "id": r[0],
            "concept_id": concept_id,
            "concept_version": r[1],
            "evidence_id": r[2],
            "fragment_type": r[3],
            "content": r[4],
            "pointer_uri": r[5],
            "pointer_meta": meta,
            "char_count": r[7],
            "created_at": r[8],
            "source_hash": r[9],
            "inherited_from": r[10],
        })

    # INGEST-038: Batch-resolve verbatim:// pointers
    _pointer_map = {}
    for _f in results:
        _uri = _f.get("pointer_uri") or ""
        if _uri.startswith("verbatim://"):
            _pointer_map[_f["id"]] = _uri[len("verbatim://"):]
    if _pointer_map:
        _canonical_ids = list(set(_pointer_map.values()))
        _canonicals = _get_fragments_by_ids(_canonical_ids)
        for _f in results:
            _cid = _pointer_map.get(_f["id"])
            if _cid and _cid in _canonicals:
                _f["content"] = _canonicals[_cid].get("content")
                _f["resolved_from"] = _cid
            elif _cid:
                # Dangling pointer — canonical was deleted or never saved
                _f["content"] = None
                _f["resolved_from"] = _cid
                _f["resolution_error"] = "canonical_not_found"

    return results


def delete_verbatim_fragment(fragment_id: str) -> bool:
    """Delete a specific verbatim fragment. Returns True if deleted."""
    with _conn._db() as conn:
        cursor = conn.execute(
            "DELETE FROM verbatim_fragments WHERE id = ?", (fragment_id,)
        )
    return cursor.rowcount > 0


def delete_verbatim_fragments_for_concept(concept_id: str) -> int:
    """Delete all verbatim fragments for a concept. Returns count deleted."""
    with _conn._db() as conn:
        cursor = conn.execute(
            "DELETE FROM verbatim_fragments WHERE concept_id = ?", (concept_id,)
        )
        deleted = cursor.rowcount
        # INGEST-037 Layer 4: Clear fragment keywords when all fragments removed
        if deleted > 0:
            conn.execute(
                "UPDATE concepts SET fragment_keywords = NULL WHERE id = ?",
                (concept_id,),
            )
    return deleted


def get_verbatim_stats() -> dict:
    """Get aggregate stats for verbatim fragments."""
    with read_snapshot_db("get_verbatim_stats") as conn:
        total = conn.execute("SELECT COUNT(*) FROM verbatim_fragments").fetchone()[0]
        total_chars = conn.execute("SELECT COALESCE(SUM(char_count), 0) FROM verbatim_fragments").fetchone()[0]
        concepts_with = conn.execute("SELECT COUNT(DISTINCT concept_id) FROM verbatim_fragments").fetchone()[0]
        by_type = conn.execute(
            "SELECT fragment_type, COUNT(*) FROM verbatim_fragments GROUP BY fragment_type"
        ).fetchall()
    return {
        "total_fragments": total,
        "total_chars": total_chars,
        "concepts_with_verbatim": concepts_with,
        "by_type": {r[0]: r[1] for r in by_type},
    }
