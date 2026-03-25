"""Cognitive Domains — Layer 3 retrieval boost.

Implements domain activation from DOMAINS_AND_DIRECTIVES_SPEC.md Section 2.3:
- Step 1: Word boundary trigger scanning (set intersection, no substring)
- Step 2: Domain scoring (strategic_priority × log2(match_count + 1))
- Step 3: Domain cap (MAX_ACTIVE_DOMAINS = 2)
- Step 4: Area boost computation (max semantics, BOOST_CAP = 0.15)
"""

import json
import logging
import math
import re

from app.storage import _db

logger = logging.getLogger(__name__)

# --- Constants (Section 2.3) ---
BOOST_CAP = 0.15  # Max domain boost (~30% of typical top semantic score)
MAX_ACTIVE_DOMAINS = 2  # Max domains activated per turn

# Word splitter: splits on whitespace and punctuation
_WORD_SPLIT = re.compile(r"[\s\W]+")


def load_all_domains() -> list:
    """Load all active domains with their area mappings.

    Returns list of dicts: {domain_id, name, strategic_priority,
                            activation_triggers: set, area_mappings: {area: weight}}
    """
    with _db() as conn:
        domains = conn.execute("""
            SELECT domain_id, name, strategic_priority, activation_triggers
            FROM cognitive_domains
            WHERE active = 1
        """).fetchall()

        if not domains:
            return []

        # Load all mappings in one query
        mappings = conn.execute("""
            SELECT domain_id, knowledge_area, activation_weight
            FROM domain_area_mapping
        """).fetchall()

    # Build mapping dict: domain_id -> {area: weight}
    area_map = {}
    for m in mappings:
        did = m["domain_id"]
        if did not in area_map:
            area_map[did] = {}
        area_map[did][m["knowledge_area"]] = m["activation_weight"]

    result = []
    for d in domains:
        triggers_raw = d["activation_triggers"]
        try:
            triggers = set(t.lower() for t in json.loads(triggers_raw))
        except (json.JSONDecodeError, TypeError):
            triggers = set()

        result.append(
            {
                "domain_id": d["domain_id"],
                "name": d["name"],
                "strategic_priority": d["strategic_priority"] or 0.5,
                "activation_triggers": triggers,
                "area_mappings": area_map.get(d["domain_id"], {}),
            }
        )

    return result


def detect_active_domains(message: str) -> list[dict]:
    """Detect which domains to activate for a message.

    Implements Section 2.3 Steps 1-3:
    1. Word boundary trigger scan
    2. Domain scoring
    3. Cap at MAX_ACTIVE_DOMAINS

    Returns list of activated domain dicts (max MAX_ACTIVE_DOMAINS).
    """
    if not message or not message.strip():
        return []

    # Step 1: Word boundary matching
    message_words = set(_WORD_SPLIT.split(message.lower()))
    message_words.discard("")  # Remove empty strings from split

    all_domains = load_all_domains()
    if not all_domains:
        return []

    # Step 2: Score each domain
    scored = []
    for domain in all_domains:
        triggers = domain["activation_triggers"]
        if not triggers:
            continue
        match_count = len(triggers & message_words)
        if match_count > 0:
            score = domain["strategic_priority"] * math.log2(match_count + 1)
            scored.append((score, domain["domain_id"], domain))

    if not scored:
        return []

    # Step 3: Cap at top-N, tiebreak by domain_id (deterministic)
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [d for _, _, d in scored[:MAX_ACTIVE_DOMAINS]]


def compute_domain_boosts(active_domains: list[dict]) -> dict[str, float]:
    """Compute per-area boost values from activated domains.

    Implements Section 2.3 Step 4:
    - Max semantics for areas in multiple domains
    - BOOST_CAP ceiling

    Returns {knowledge_area: capped_boost_value}
    """
    if not active_domains:
        return {}

    boosts = {}
    for domain in active_domains:
        priority = domain["strategic_priority"]
        for area, weight in domain["area_mappings"].items():
            raw_boost = weight * priority
            boosts[area] = min(max(boosts.get(area, 0.0), raw_boost), BOOST_CAP)

    return boosts


def apply_domain_boost(message: str) -> tuple[dict[str, float], list[str]]:
    """Full S1.5 pipeline: detect domains, compute boosts.

    Returns (boost_map, activated_domain_ids) for logging/response.
    """
    active = detect_active_domains(message)
    if not active:
        return {}, []

    boosts = compute_domain_boosts(active)
    domain_ids = [d["domain_id"] for d in active]

    logger.info(
        f"S1.5: Domains activated: {domain_ids}, "
        f"areas boosted: {len(boosts)}, "
        f"max boost: {max(boosts.values()) if boosts else 0:.3f}"
    )

    return boosts, domain_ids
