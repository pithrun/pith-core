"""Behavioral Directives — Provider-agnostic soul.

Implements DOMAINS_AND_DIRECTIVES_SPEC.md Sections 3, 6.3:
- CRUD with validation (DIRECTIVE_BLOCKLIST, character limits)
- Budget-aware delivery (8,000 char aggregate, priority-ordered truncation)
- Version history tracking
- Rate limiting (20 writes/hour)

Directives are Tier 2 in the injection hierarchy:
  Firmware > Directives > Always-Activate > Concepts
"""

import logging
import re
import time
from collections import defaultdict
from typing import Any

import app.storage as storage
from app.core.datetime_utils import _utc_now_iso
from app.storage import read_snapshot_db

logger = logging.getLogger(__name__)

# --- Constants (Section 6.3) ---
MAX_DIRECTIVE_CHARS = 2000  # Per-directive hard cap
AGGREGATE_BUDGET_CHARS = 8000  # Total active content budget
MAX_ACTIVE_DIRECTIVES = 50  # Max active directives
MAX_WRITES_PER_HOUR = 20  # Rate limit

VALID_CATEGORIES = frozenset(["persona", "workflow", "constraints", "formatting", "domain_rules"])

# --- Directive Blocklist (Section 3.8) ---
DIRECTIVE_BLOCKLIST = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|rules?|constraints?)",
        r"(reveal|show|display|output)\s+(the\s+)?(system\s+)?prompt",
        r"override\s+(safety|security|rules?|instructions?)",
        r"(jailbreak|DAN|do\s+anything\s+now)",
        r"you\s+are\s+now\s+(in\s+)?(developer|admin|debug|unrestricted)\s+mode",
        r"(disregard|forget|bypass)\s+(all\s+)?(previous|prior|safety)",
        r"pretend\s+(you\s+)?(are|have)\s+no\s+(restrictions?|rules?|limits?)",
        r"act\s+as\s+if\s+(you\s+)?(have|had)\s+no\s+(guidelines|restrictions?)",
    ]
]

# Provider-specific terms for soft warnings
_PROVIDER_TERMS = re.compile(r"\b(claude|chatgpt|gpt-4|gemini|copilot|llama)\b", re.IGNORECASE)

# Rate limit tracker: {api_key_or_ip: [(timestamp, ...)]}
_write_timestamps: dict[str, list] = defaultdict(list)


class DirectiveValidationError(Exception):
    """Raised when directive content fails validation."""

    def __init__(self, error_code: str, detail: str):
        self.error_code = error_code
        self.detail = detail
        super().__init__(detail)


def validate_content(content: str) -> str | None:
    """Validate directive content against blocklist and limits.

    Returns None if valid, or warning string for soft issues.
    Raises DirectiveValidationError for hard failures.
    """
    if not content or not content.strip():
        raise DirectiveValidationError("empty_content", "Directive content cannot be empty")

    if len(content) > MAX_DIRECTIVE_CHARS:
        raise DirectiveValidationError(
            "content_too_long", f"Content is {len(content)} chars, max is {MAX_DIRECTIVE_CHARS}"
        )

    # Check blocklist
    for pattern in DIRECTIVE_BLOCKLIST:
        if pattern.search(content):
            raise DirectiveValidationError("directive_content_blocked", "Content matches a blocked pattern")

    # Soft warning for provider-specific terms
    if _PROVIDER_TERMS.search(content):
        return "warning: content references a specific AI provider; directives should be provider-agnostic"

    return None


def check_rate_limit(caller_id: str = "default") -> bool:
    """Check if write rate limit is exceeded.

    Returns True if within limit, raises DirectiveValidationError if exceeded.
    """
    now = time.time()
    hour_ago = now - 3600

    # Clean old timestamps
    _write_timestamps[caller_id] = [t for t in _write_timestamps[caller_id] if t > hour_ago]

    if len(_write_timestamps[caller_id]) >= MAX_WRITES_PER_HOUR:
        raise DirectiveValidationError("rate_limit_exceeded", f"Max {MAX_WRITES_PER_HOUR} writes per hour")

    _write_timestamps[caller_id].append(now)
    return True


def save_directive(
    directive_id: str,
    category: str,
    content: str,
    priority: int = 100,
    caller_id: str = "default",
) -> dict[str, Any]:
    """Create or update a directive (upsert).

    Returns dict with action taken and any warnings.
    Raises DirectiveValidationError on validation failure.
    """
    # Validate
    if category not in VALID_CATEGORIES:
        raise DirectiveValidationError(
            "invalid_category", f"Category must be one of: {', '.join(sorted(VALID_CATEGORIES))}"
        )

    check_rate_limit(caller_id)
    warning = validate_content(content)

    now = _utc_now_iso()

    with storage._db() as conn:
        # Check if exists
        existing = conn.execute(
            "SELECT version, active FROM directives WHERE directive_id = ?", (directive_id,)
        ).fetchone()

        if existing:
            # Update
            new_version = existing["version"] + 1
            conn.execute(
                """
                UPDATE directives SET
                    category = ?, content = ?, priority = ?,
                    version = ?, updated_at = ?
                WHERE directive_id = ?
            """,
                (category, content, priority, new_version, now, directive_id),
            )

            # Record version history
            conn.execute(
                """
                INSERT INTO directive_versions
                    (directive_id, version, content, category, priority)
                VALUES (?, ?, ?, ?, ?)
            """,
                (directive_id, new_version, content, category, priority),
            )

            action = "updated"
            version = new_version
        else:
            # Check active count limit
            active_count = conn.execute("SELECT COUNT(*) as cnt FROM directives WHERE active = 1").fetchone()["cnt"]

            if active_count >= MAX_ACTIVE_DIRECTIVES:
                raise DirectiveValidationError(
                    "max_directives_exceeded", f"Max {MAX_ACTIVE_DIRECTIVES} active directives"
                )

            # Insert
            conn.execute(
                """
                INSERT INTO directives
                    (directive_id, category, content, priority, active, version, created_at, updated_at)
                VALUES (?, ?, ?, ?, 1, 1, ?, ?)
            """,
                (directive_id, category, content, priority, now, now),
            )

            # Record initial version
            conn.execute(
                """
                INSERT INTO directive_versions
                    (directive_id, version, content, category, priority)
                VALUES (?, 1, ?, ?, ?)
            """,
                (directive_id, content, category, priority),
            )

            action = "created"
            version = 1

    result = {
        "directive_id": directive_id,
        "action": action,
        "version": version,
    }
    if warning:
        result["warning"] = warning

    logger.info(f"Directive {action}: {directive_id} v{version}")
    return result


def load_directives(active_only: bool = True) -> list:
    """Load directives ordered by category then priority.

    Returns list of dicts with directive fields.
    """
    with read_snapshot_db("load_directives") as conn:
        where = "WHERE active = 1" if active_only else ""
        rows = conn.execute(f"""
            SELECT directive_id, category, content, priority, active, version,
                   created_at, updated_at
            FROM directives
            {where}
            ORDER BY category, priority ASC
        """).fetchall()

    return [dict(row) for row in rows]


def load_directives_budgeted() -> dict[str, Any]:
    """Load directives with aggregate budget enforcement (Section 6.3 Layer 2).

    Returns {directives: [...], budget_warning: str|None, total_chars: int, dropped: int}
    """
    all_directives = load_directives(active_only=True)

    budget_remaining = AGGREGATE_BUDGET_CHARS
    delivered = []
    dropped = 0

    for d in all_directives:
        cost = len(d["content"])
        if budget_remaining >= cost:
            delivered.append(d)
            budget_remaining -= cost
        else:
            dropped += 1

    total_chars = AGGREGATE_BUDGET_CHARS - budget_remaining
    warning = None
    if dropped > 0:
        warning = f"{dropped} directive(s) dropped due to aggregate budget ({AGGREGATE_BUDGET_CHARS} chars)"
        logger.warning(f"S4.8: Budget truncation — delivered {len(delivered)}, dropped {dropped}")

    return {
        "directives": delivered,
        "budget_warning": warning,
        "total_chars": total_chars,
        "dropped": dropped,
    }


def get_directive(directive_id: str, include_versions: bool = False) -> dict | None:
    """Get a single directive, optionally with version history."""
    with storage._db() as conn:
        row = conn.execute("SELECT * FROM directives WHERE directive_id = ?", (directive_id,)).fetchone()

        if not row:
            return None

        result = dict(row)

        if include_versions:
            versions = conn.execute(
                """
                SELECT version, content, category, priority, created_at
                FROM directive_versions
                WHERE directive_id = ?
                ORDER BY version ASC
            """,
                (directive_id,),
            ).fetchall()
            result["version_history"] = [dict(v) for v in versions]

    return result


def delete_directive(directive_id: str) -> bool:
    """Soft-delete a directive (set active=false). Preserves history.

    Returns True if found and deactivated, False if not found.
    """
    now = _utc_now_iso()
    with storage._db() as conn:
        cursor = conn.execute(
            "UPDATE directives SET active = 0, updated_at = ? WHERE directive_id = ?", (now, directive_id)
        )
        if cursor.rowcount > 0:
            logger.info(f"Directive soft-deleted: {directive_id}")
            return True
    return False
