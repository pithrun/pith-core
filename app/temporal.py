"""TEMPORAL-002: Extract temporal references from concept text.

Extracts the date a piece of knowledge *refers to* (original_date),
which is distinct from created_at (when the concept was ingested).

Examples:
    "Andrew moved to SF in March 2025" → "2025-03"
    "Q3 2024 revenue was $5M"         → "2024-Q3"
    "Started new job on 2025-01-15"    → "2025-01-15"
    "The sky is blue"                  → None (no temporal reference)
"""

import re
from typing import Optional

# Priority-ordered patterns: most specific first
_TEMPORAL_PATTERNS = [
    # ISO date: 2025-01-15, 2025/01/15
    (re.compile(r"\b(20\d{2}[-/]\d{2}[-/]\d{2})\b"), lambda m: m.group(1).replace("/", "-")),
    # Month Year: January 2025, Jan 2025, March 2024
    (re.compile(
        r"\b(January|February|March|April|May|June|July|August|September|"
        r"October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        r"\s+(20\d{2})\b", re.IGNORECASE
    ), lambda m: f"{m.group(2)}-{_month_num(m.group(1))}"),
    # Quarter Year: Q1 2025, Q3 2024
    (re.compile(r"\b(Q[1-4])\s+(20\d{2})\b", re.IGNORECASE),
     lambda m: f"{m.group(2)}-{m.group(1).upper()}"),
    # Year Quarter: 2025 Q1
    (re.compile(r"\b(20\d{2})\s+(Q[1-4])\b", re.IGNORECASE),
     lambda m: f"{m.group(1)}-{m.group(2).upper()}"),
    # "in 2025", "since 2024", "during 2023" — year only with temporal preposition
    (re.compile(r"\b(?:in|since|during|from|around|by|after|before|until)\s+(20\d{2})\b", re.IGNORECASE),
     lambda m: m.group(1)),
]

_MONTH_MAP = {
    "january": "01", "jan": "01", "february": "02", "feb": "02",
    "march": "03", "mar": "03", "april": "04", "apr": "04",
    "may": "05", "june": "06", "jun": "06", "july": "07", "jul": "07",
    "august": "08", "aug": "08", "september": "09", "sep": "09",
    "october": "10", "oct": "10", "november": "11", "nov": "11",
    "december": "12", "dec": "12",
}


def _month_num(name: str) -> str:
    return _MONTH_MAP.get(name.lower(), "01")


# DEBT-188: Sentinel values that represent non-real concept IDs in supersession chains.
# Must match nonlossy.py's _SUPERSESSION_SENTINELS to prevent drift.
_SUPERSESSION_SENTINELS = ("", "__orphaned_supersession__")


def _is_sentinel(value: Optional[str]) -> bool:
    """Return True if value is a supersession sentinel (not a real concept ID)."""
    return value is None or value in _SUPERSESSION_SENTINELS


def _get_connection():
    """Get a database connection. Delegates to app.storage."""
    from app.storage import _get_connection as _storage_get_connection
    return _storage_get_connection()


def extract_temporal_reference(text: str) -> Optional[str]:
    """Extract the most specific temporal reference from text.

    Returns ISO-8601 partial date string or None.
    Scans summary + evidence text for date references.
    """
    if not text:
        return None
    for pattern, formatter in _TEMPORAL_PATTERNS:
        match = pattern.search(text)
        if match:
            return formatter(match)
    return None


def temporal_boost(timestamp: str) -> dict:
    """Calculate recency boost multiplier for a concept.
    
    Returns dict with 'status' and 'boost_multiplier'.
    Boost decays from 1.15 (now) to 1.0 (30+ days old).
    Step function at boundaries:
    - <=24h: 1.15
    - <=72h: 1.08
    - <=168h: 1.03
    - >720h: 1.0
    """
    from datetime import datetime, timezone
    
    try:
        # Parse ISO 8601 timestamp
        if timestamp.endswith('Z'):
            dt = datetime.fromisoformat(timestamp[:-1] + '+00:00')
        else:
            dt = datetime.fromisoformat(timestamp)
        
        now = datetime.now(timezone.utc)
        delta = now - dt
        hours = delta.total_seconds() / 3600
        
        # Clamp to [0, inf)
        hours = max(0, hours)
        
        if hours <= 24:
            boost = 1.15
        elif hours <= 72:
            boost = 1.08
        elif hours <= 168:
            boost = 1.03
        else:
            boost = 1.0
        
        return {"status": "success", "boost_multiplier": boost}
    except (ValueError, TypeError) as e:
        return {"status": "error", "message": f"Invalid date format: {timestamp}"}


def temporal_date_filters(since: str = None, until: str = None, temporal_field: str = 'updated_at') -> dict:
    """Build SQL WHERE clause for temporal filtering.
    
    Valid temporal_field values: "updated_at", "created_at", "last_accessed"
    Returns dict with 'status', 'where_clause', 'params', and 'field'.
    """
    from datetime import datetime
    
    # Validate temporal field
    valid_fields = {"updated_at", "created_at", "last_accessed"}
    if temporal_field not in valid_fields:
        return {
            "status": "error",
            "message": f"Invalid temporal_field: {temporal_field}. Must be one of {valid_fields}"
        }
    
    # Check that at least one filter is provided
    if since is None and until is None:
        return {
            "status": "error",
            "message": "At least one of 'since' or 'until' must be provided"
        }
    
    # Validate date formats
    try:
        if since is not None:
            if since.endswith('Z'):
                datetime.fromisoformat(since[:-1] + '+00:00')
            else:
                datetime.fromisoformat(since)
    except (ValueError, TypeError):
        return {
            "status": "error",
            "message": f"Invalid since date format: {since}"
        }
    
    try:
        if until is not None:
            if until.endswith('Z'):
                datetime.fromisoformat(until[:-1] + '+00:00')
            else:
                datetime.fromisoformat(until)
    except (ValueError, TypeError):
        return {
            "status": "error",
            "message": f"Invalid until date format: {until}"
        }
    
    # Build WHERE clause
    if since is not None and until is not None:
        where_clause = f"{temporal_field} BETWEEN ? AND ?"
        params = [since, until]
    elif since is not None:
        where_clause = f"{temporal_field} >= ?"
        params = [since]
    else:  # until is not None
        where_clause = f"{temporal_field} <= ?"
        params = [until]
    
    return {
        "status": "success",
        "where_clause": where_clause,
        "params": params,
        "field": temporal_field
    }


def pith_timeline(since: str, until: str, event_types=None, knowledge_area: str = None,
                  concept_type: str = None, limit: int = 100, group_by: str = None) -> dict:
    """Query concepts within a time window.
    
    Returns {"status": "empty"} when no concepts found.
    Returns {"status": "success", "concepts": [...]} when found.
    Returns {"status": "error", "message": "..."} for invalid dates or since > until.
    """
    from datetime import datetime
    # Validate date formats
    try:
        if since.endswith('Z'):
            since_dt = datetime.fromisoformat(since[:-1] + '+00:00')
        else:
            since_dt = datetime.fromisoformat(since)
    except (ValueError, TypeError):
        return {"status": "error", "message": f"Invalid since date: {since}"}
    
    try:
        if until.endswith('Z'):
            until_dt = datetime.fromisoformat(until[:-1] + '+00:00')
        else:
            until_dt = datetime.fromisoformat(until)
    except (ValueError, TypeError):
        return {"status": "error", "message": f"Invalid until date: {until}"}
    
    # Check since <= until
    if since_dt > until_dt:
        return {"status": "error", "message": "since must be <= until"}
    
    # Build query
    query = "SELECT * FROM concepts WHERE created_at BETWEEN ? AND ?"
    params = [since, until]
    
    if knowledge_area is not None:
        query += " AND knowledge_area = ?"
        params.append(knowledge_area)
    
    if concept_type is not None:
        query += " AND concept_type = ?"
        params.append(concept_type)
    
    query += f" ORDER BY created_at DESC LIMIT {limit}"
    
    try:
        conn = _get_connection()
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        
        if not rows:
            return {"status": "empty"}
        
        concepts = [dict(row) for row in rows]
        return {"status": "success", "concepts": concepts}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def pith_knowledge_at(point_in_time: str, knowledge_area: str = None,
                      concept_type: str = None, limit: int = 100) -> dict:
    """Query concepts valid at a specific point in time.
    
    Checks: valid_from <= point_in_time AND (valid_until IS NULL OR valid_until >= point_in_time)
    Returns {"status": "empty"} or {"status": "success", "concepts": [...]}.
    """
    from datetime import datetime
    
    # Validate date format
    try:
        if point_in_time.endswith('Z'):
            datetime.fromisoformat(point_in_time[:-1] + '+00:00')
        else:
            datetime.fromisoformat(point_in_time)
    except (ValueError, TypeError):
        return {"status": "error", "message": f"Invalid point_in_time: {point_in_time}"}
    
    # Build query
    query = """SELECT * FROM concepts 
               WHERE valid_from <= ? AND (valid_until IS NULL OR valid_until >= ?)"""
    params = [point_in_time, point_in_time]
    
    if knowledge_area is not None:
        query += " AND knowledge_area = ?"
        params.append(knowledge_area)
    
    if concept_type is not None:
        query += " AND concept_type = ?"
        params.append(concept_type)
    
    query += f" ORDER BY valid_from DESC LIMIT {limit}"
    
    try:
        conn = _get_connection()
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        
        if not rows:
            return {"status": "empty"}
        
        concepts = [dict(row) for row in rows]
        return {"status": "success", "concepts": concepts}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def pith_evolution_of(concept_id: str) -> dict:
    """Walk the supersession chain for a concept.
    
    Returns {"status": "empty"} if concept doesn't exist.
    Returns {"status": "success", "chain": [...], "current_version": "..."} with evolution chain.
    """
    try:
        conn = _get_connection()
        
        # Check if concept exists
        cursor = conn.execute("SELECT id FROM concepts WHERE id = ?", (concept_id,))
        if not cursor.fetchone():
            return {"status": "empty"}
        
        # Build supersession chain by walking backwards from concept_id
        chain = [concept_id]
        current = concept_id
        visited = {concept_id}
        
        while True:
            cursor = conn.execute(
                "SELECT superseded_by FROM concepts WHERE id = ?",
                (current,)
            )
            row = cursor.fetchone()
            if not row or not row[0] or row[0] in visited:
                break
            superseding = row[0]
            if _is_sentinel(superseding):
                break
            chain.append(superseding)
            visited.add(superseding)
            current = superseding
        
        return {
            "status": "success",
            "chain": chain,
            "current_version": current
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
