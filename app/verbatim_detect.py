"""INGEST-037 Phase 2a: Verbatim fragment auto-extraction.

Post-extraction regex heuristics to detect high-information-density
patterns (code blocks, SQL, shell commands, config, quotes) in raw
conversation text and create VerbatimFragment objects.

This module is PURE — no side effects, no DB access, no imports from
session.py. Safe to import anywhere.
"""

from __future__ import annotations

import re

from app.models import VerbatimFragment

# --- Compiled regex patterns ---

# Fenced code blocks (```...```)
_FENCED_CODE_RE = re.compile(
    r"```(?:\w+)?\s*\n(.*?)```",
    re.DOTALL,
)

# SQL statements (SELECT, INSERT, UPDATE, DELETE, CREATE, ALTER, DROP)
_SQL_RE = re.compile(
    r"\b((?:SELECT|INSERT\s+INTO|UPDATE|DELETE\s+FROM|"
    r"CREATE\s+(?:OR\s+REPLACE\s+)?(?:TABLE|VIEW|INDEX|FUNCTION)|"
    r"ALTER\s+TABLE|DROP\s+(?:TABLE|VIEW|INDEX))\b[^;]{10,};?)",
    re.IGNORECASE | re.DOTALL,
)

# Indented code blocks (4+ spaces or tab, 2+ consecutive lines)
_INDENTED_CODE_RE = re.compile(
    r"(?:^(?:[ ]{4,}|\t).+\n?){2,}",
    re.MULTILINE,
)

# JSON/YAML config blocks — two-phase: fast single-line match + kv-count filter
# (Original nested-quantifier pattern caused catastrophic backtracking on AMA-scale text)
_CONFIG_JSON_RE = re.compile(
    r"\{[^{}\n]{20,500}\}",
)
_CONFIG_KV_RE = re.compile(r'\"[^\"]+\"\s*:')

# Exact quotes ("..." at least 20 chars)
_EXACT_QUOTE_RE = re.compile(
    r"\"([^\"]{20,}?)\"",
)

# Shell/SQL commands with output: common in agent trajectories
# Includes execute_snowflake_sql for AMA text2sql episodes
_SHELL_CMD_RE = re.compile(
    r"(?:execute_bash|execute_sql|execute_snowflake_sql|run_command):\s*(.{20,}?)(?:\n|$)",
    re.IGNORECASE,
)

# Minimum fragment size — skip trivially short matches
MIN_FRAGMENT_CHARS = 20
# Maximum fragment size — truncate extremely long matches
MAX_FRAGMENT_CHARS = 5000


def detect_verbatim_fragments(text: str) -> list[VerbatimFragment]:
    """Scan text for high-information-density patterns.

    Returns deduplicated list of VerbatimFragment objects, ordered by
    position in the source text. Each fragment is classified by type.

    This function is PURE — no side effects, no DB access.
    """
    if not text or len(text) < MIN_FRAGMENT_CHARS:
        return []

    fragments: list[tuple[int, VerbatimFragment]] = []  # (position, fragment)
    seen_hashes: set[str] = set()  # dedup by content prefix

    def _add(pos: int, content: str, ftype: str) -> None:
        content = content.strip()
        if len(content) < MIN_FRAGMENT_CHARS:
            return
        if len(content) > MAX_FRAGMENT_CHARS:
            content = content[:MAX_FRAGMENT_CHARS]
        # Simple dedup: first 200 chars as key
        h = content[:200]
        if h in seen_hashes:
            return
        seen_hashes.add(h)
        fragments.append((pos, VerbatimFragment(
            fragment_type=ftype,
            content=content,
        )))

    # 1. Fenced code blocks (highest priority)
    for m in _FENCED_CODE_RE.finditer(text):
        _add(m.start(), m.group(1), "code")

    # 2. SQL statements
    for m in _SQL_RE.finditer(text):
        # Skip if already captured inside a fenced code block
        if any(f[1].content and m.group(1).strip()[:50] in f[1].content for f in fragments):
            continue
        _add(m.start(), m.group(1), "code")

    # 3. Shell commands (agent trajectories)
    for m in _SHELL_CMD_RE.finditer(text):
        _add(m.start(), m.group(1), "code")

    # 4. Indented code blocks
    for m in _INDENTED_CODE_RE.finditer(text):
        if any(f[1].content and m.group(0).strip()[:50] in f[1].content for f in fragments):
            continue
        _add(m.start(), m.group(0), "code")

    # 5. JSON/YAML config blocks (two-phase: regex + kv-count filter)
    for m in _CONFIG_JSON_RE.finditer(text):
        block = m.group(0)
        if len(_CONFIG_KV_RE.findall(block)) < 2:
            continue
        if any(f[1].content and block.strip()[:50] in f[1].content for f in fragments):
            continue
        _add(m.start(), block, "code")

    # 6. Exact quotes (lowest priority — skip if overlaps with above)
    for m in _EXACT_QUOTE_RE.finditer(text):
        if any(f[1].content and m.group(1).strip()[:30] in f[1].content for f in fragments):
            continue
        _add(m.start(), m.group(1), "text")

    # Sort by position, return fragments only
    fragments.sort(key=lambda x: x[0])
    return [f[1] for f in fragments]


def match_fragments_to_insights(
    fragments: list[VerbatimFragment],
    insights: list[dict],
) -> dict[int, list[VerbatimFragment]]:
    """Map detected fragments to their originating insights.

    Args:
        fragments: Output of detect_verbatim_fragments().
        insights: The merged_insights list from session_learn (list of dicts
                  with 'summary', 'evidence', etc.).

    Returns:
        Dict mapping insight index -> list of matched VerbatimFragments.
        Orphan fragments are attached to the first insight as a fallback.
    """
    if not fragments or not insights:
        return {}

    result: dict[int, list[VerbatimFragment]] = {}
    orphans: list[VerbatimFragment] = []

    for frag in fragments:
        matched = False
        frag_text = (frag.content or "")[:200].lower()
        if not frag_text:
            continue

        for idx, insight in enumerate(insights):
            summary = (insight.get("summary", "") or "").lower()
            evidence_strs = insight.get("evidence", []) or []
            evidence_text = " ".join(
                str(e).lower() for e in evidence_strs[:5]
            )

            frag_prefix = frag_text[:80]
            summary_prefix = summary[:60]

            if (
                (frag_prefix and frag_prefix in summary + " " + evidence_text)
                or (summary_prefix and len(summary_prefix) > 20 and summary_prefix in frag_text)
            ):
                result.setdefault(idx, []).append(frag)
                matched = True
                break  # First match wins

        if not matched:
            orphans.append(frag)

    # Attach orphans to first insight (fallback)
    if orphans and insights:
        result.setdefault(0, []).extend(orphans)

    return result


# --- INGEST-038: Raw conversation text capture ---


def capture_conversation_verbatim(
    user_message: str,
    assistant_response: str,
    concept_ids: list[str],
    concept_versions: dict[str, str] | None = None,
) -> list[str]:
    """Capture raw conversation text as verbatim fragments.

    INGEST-038: Stores one canonical fragment (full text) on the first concept
    that accepts it (budget-aware fallback per gauntlet F2). Creates pointer
    fragments on remaining concepts referencing the canonical via verbatim:// URI.

    All saves use skip_enrichment=True (gauntlet F1/F6) — conversation text
    should not pollute the keyword search index.

    Args:
        user_message: Raw user message (from request, NOT combined_text)
        assistant_response: Raw assistant response
        concept_ids: Concept IDs created from this turn (ordered by relevance)
        concept_versions: Optional {concept_id: version} for version tagging

    Returns:
        List of created fragment IDs (canonical + pointers)
    """
    import logging

    logger = logging.getLogger(__name__)

    # Format content with role markers for downstream parsing
    content = "[USER] " + user_message + "\n\n[ASSISTANT] " + assistant_response
    if len(content) < MIN_FRAGMENT_CHARS:
        return []

    from app.storage import save_verbatim_fragment

    created_ids: list[str] = []
    canonical_id: str | None = None
    canonical_concept_idx: int = -1

    # Gauntlet F2: Try each concept as canonical until one accepts the budget
    for idx, cid in enumerate(concept_ids):
        version = (concept_versions or {}).get(cid)
        frag_id = save_verbatim_fragment(
            concept_id=cid,
            fragment_type="conversation",
            content=content,
            concept_version=version,
            skip_enrichment=True,  # F1/F6: no keyword pollution from conversation text
        )
        if frag_id:
            canonical_id = frag_id
            canonical_concept_idx = idx
            created_ids.append(frag_id)
            break  # Found a home for the canonical

    if not canonical_id:
        logger.warning(
            "INGEST-038: All %d concepts over budget, conversation capture skipped",
            len(concept_ids),
        )
        return []

    # Create pointer fragments on remaining concepts
    for idx, cid in enumerate(concept_ids):
        if idx == canonical_concept_idx:
            continue
        version = (concept_versions or {}).get(cid)
        ptr_id = save_verbatim_fragment(
            concept_id=cid,
            fragment_type="conversation",
            content=None,
            pointer_uri=f"verbatim://{canonical_id}",
            concept_version=version,
            skip_enrichment=True,  # F1/F6: pointers must never trigger recompute
        )
        if ptr_id:
            created_ids.append(ptr_id)

    logger.debug(
        "INGEST-038: Captured conversation verbatim — %d fragments "
        "(1 canonical on concept %s, %d pointers)",
        len(created_ids),
        concept_ids[canonical_concept_idx],
        len(created_ids) - 1,
    )

    return created_ids
