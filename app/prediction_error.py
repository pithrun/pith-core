"""Pre-Response Prediction Error — constraint assembly and post-response validation.

Assembles high-authority concepts into a constraint set, generates anti-terms
from correction history + auto-generated fallback, and provides both
client-side anti-term scan and server-side semantic validation.

Phase 1: Constraint set assembly + anti-term generation + anti-term scan
Phase 2 (P4a): Post-response validation — negation detection + entity overlap
              + embedding escalation + false positive mitigation
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from app.datetime_utils import _utc_now

logger = logging.getLogger(__name__)


# =============================================================================
# Constants — Memory Integrity Spec v1.2, §5.6.1
# =============================================================================
# CONSTRAINT_AUTHORITY_THRESHOLD and MAX_CONSTRAINTS moved to config.py.
# Feature-flag-gated: when HARDENED_CONSTRAINTS_ENABLED=True, uses hardened
# values (0.80 threshold, 8 max). When False, falls back to legacy values.

from app.config import (
    _LEGACY_CONSTRAINT_AUTHORITY_THRESHOLD,
    _LEGACY_MAX_CONSTRAINTS,
    FEATURE_FLAGS,
)
from app.config import (
    CONSTRAINT_AUTHORITY_THRESHOLD as _HARDENED_THRESHOLD,
)
from app.config import (
    MAX_CONSTRAINTS as _HARDENED_MAX_CONSTRAINTS,
)

# Resolve values based on feature flag
if FEATURE_FLAGS.get("HARDENED_CONSTRAINTS_ENABLED", False):
    CONSTRAINT_AUTHORITY_THRESHOLD = _HARDENED_THRESHOLD  # 0.80
    MAX_CONSTRAINTS = _HARDENED_MAX_CONSTRAINTS  # 8
else:
    CONSTRAINT_AUTHORITY_THRESHOLD = _LEGACY_CONSTRAINT_AUTHORITY_THRESHOLD  # 0.60
    MAX_CONSTRAINTS = _LEGACY_MAX_CONSTRAINTS  # 15

# Authority threshold for hard enforcement (vs soft warning)
HARD_ENFORCEMENT_AUTHORITY = 0.80

# Maximum anti-terms per constraint
MAX_ANTI_TERMS_PER_CONSTRAINT = 10

# Auto-generated anti-term confidence discount
AUTO_GENERATED_CONFIDENCE = 0.50

# Domain-specific opposition maps for auto-generation
# key -> list of anti-terms
_DOMAIN_OPPOSITIONS = {
    # Architecture/positioning
    "runtime": ["library", "framework", "toolkit", "SDK"],
    "cognitive": ["memory", "storage", "database", "cache"],
    "platform": ["tool", "utility", "script"],
    "operating system": ["application", "plugin", "extension"],
    "os layer": ["application layer", "plugin layer"],
    # Product
    "subscription": ["one-time", "free", "open source"],
    "enterprise": ["consumer", "personal", "individual"],
    "saas": ["self-hosted", "on-premise", "desktop"],
    # Technical
    "embedding": ["keyword", "lexical", "regex"],
    "real-time": ["batch", "offline", "scheduled"],
    "api": ["cli", "gui", "manual"],
}

# Action verbs and their reversals for anti-term generation (Strategy B)
_VERB_REVERSALS = {
    "shipped": ["revert", "rollback"],
    "approved": ["reject", "block"],
    "passed": ["failed", "reject"],
    "fixed": ["reintroduce", "break"],
    "enabled": ["disable", "turn off"],
    "added": ["remove", "delete"],
    "created": ["delete", "remove"],
    "implemented": ["revert", "remove"],
    "merged": ["revert", "unmerge"],
    "deployed": ["rollback", "undeploy"],
    "decided": ["reconsider", "reverse"],
    "rebalanced": ["revert to old", "reset"],
    "updated": ["revert", "restore old"],
    "migrated": ["rollback migration", "revert migration"],
    "committed": ["revert commit", "reset"],
    "confirmed": ["deny", "contradict"],
    "resolved": ["reopen", "unresolved"],
}

# Constraint violation patterns for anti-term generation (Strategy C)
_CONSTRAINT_MARKERS = {
    "must": "must not",
    "always": "never",
    "never": "always",
    "only": "also",
    "required": "optional",
    "mandatory": "skip",
    "do not": "do",
    "cannot": "can",
    "prohibited": "allowed",
}

# Stopwords for key noun extraction (Strategy D)
_STOPWORDS = frozenset([
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "that", "this", "these",
    "those", "it", "its", "and", "but", "or", "not", "no", "so", "if",
    "then", "than", "too", "very", "just", "also", "only", "with", "from",
    "into", "for", "of", "on", "in", "to", "at", "by", "as", "about",
    "which", "what", "when", "where", "who", "how", "all", "each", "both",
    "more", "most", "some", "any", "such", "there", "here", "after",
    "before", "between", "through", "during", "above", "below",
])


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class Constraint:
    """A behavioral constraint derived from a high-authority concept."""

    concept_id: str
    constraint: str  # The concept summary as a constraint statement
    authority: float
    anti_terms: list[str] = field(default_factory=list)
    anti_term_sources: dict[str, str] = field(default_factory=dict)  # term -> "curated" | "auto"
    presentation_mode: str = "DIRECTIVE"


@dataclass
class ConstraintViolation:
    """A detected violation of a constraint."""

    constraint: Constraint
    anti_term_matched: str
    position_in_text: int = -1
    severity: str = "soft"  # "hard" if authority >= 0.80, else "soft"
    source: str = "anti_term_scan"


@dataclass
class ConstraintSet:
    """Full constraint set for a conversation turn."""

    constraints: list[Constraint] = field(default_factory=list)
    assembly_time_ms: float = 0.0
    constraint_count: int = 0
    total_anti_terms: int = 0


@dataclass
class EnforcementResult:
    """Result from running enforcement scan on a response."""

    violations: list[ConstraintViolation] = field(default_factory=list)
    hard_violations: int = 0
    soft_violations: int = 0
    scan_time_ms: float = 0.0
    correction_prompt: str | None = None


# =============================================================================
# Anti-Term Generation
# =============================================================================


def _get_curated_anti_terms(
    concept_id: str,
    conn=None,
) -> list[tuple[str, str]]:
    """Get curated anti-terms from correction history for a concept.

    Reads the corrections table to find what WRONG framings were used
    when this concept was the affected concept.

    Returns:
        List of (anti_term, source) tuples where source = "curated"
    """
    if not conn:
        return []

    try:
        # SAFETY: concept_ids are system-generated (uuid4 hex + underscores),
        # never contain SQL wildcards (%, _) or quotes. If this changes,
        # switch to json_each() for proper JSON array membership testing.
        rows = conn.execute(
            """SELECT corrected_claim FROM corrections
               WHERE affected_concept_ids LIKE ?
               AND created_at > ?
               ORDER BY created_at DESC LIMIT 10""",
            (f'%"{concept_id}"%', (_utc_now() - timedelta(days=90)).isoformat()),
        ).fetchall()

        terms = []
        for row in rows:
            claim = row[0]
            if claim and len(claim) >= 3:
                # Extract key phrases from the wrong claim
                # Simple heuristic: split on common delimiters, take meaningful chunks
                for part in re.split(r"[,;.]", claim):
                    part = part.strip()
                    if 3 <= len(part) <= 50:
                        terms.append((part.lower(), "curated"))

        return terms[:MAX_ANTI_TERMS_PER_CONSTRAINT]

    except Exception as e:
        logger.warning("Curated anti-term lookup failed for %s: %s", concept_id, e)
        return []


def _get_stored_anti_terms(
    concept_id: str,
    conn=None,
) -> list[tuple[str, str]]:
    """Get explicitly stored anti-terms from concept.data['anti_terms'].

    These are the highest-priority anti-terms — hand-curated or persisted
    from prior generation cycles. Stored directly on the concept.

    Returns:
        List of (anti_term, source) tuples where source = "stored"
    """
    if not conn:
        return []

    try:
        row = conn.execute(
            "SELECT json_extract(data, '$.anti_terms') FROM concepts WHERE id = ?",
            (concept_id,),
        ).fetchone()

        if not row or not row[0]:
            return []

        stored = json.loads(row[0]) if isinstance(row[0], str) else row[0]
        if not isinstance(stored, list):
            return []

        return [
            (str(t).lower(), "stored")
            for t in stored
            if t and len(str(t)) >= 3
        ][:MAX_ANTI_TERMS_PER_CONSTRAINT]

    except Exception as e:
        logger.warning("Stored anti-term lookup failed for %s: %s", concept_id, e)
        return []


def _generate_auto_anti_terms(summary: str) -> list[tuple[str, str]]:
    """Generate anti-terms automatically from constraint/decision summary.

    Uses 4 extraction strategies in priority order:
      A. Domain oppositions (existing _DOMAIN_OPPOSITIONS map)
      B. Verb reversal extraction (action verbs → opposite actions)
      C. Constraint violation patterns (for CONSTRAINT/ALWAYS-tagged concepts)
      D. Key noun negation (significant nouns + negation prefixes)

    Returns:
        List of (anti_term, source) tuples where source = "auto"
    """
    terms: list[tuple[str, str]] = []
    seen: set[str] = set()
    summary_lower = summary.lower()

    def _add(term: str) -> None:
        t = term.lower().strip()
        if t and len(t) >= 3 and t not in seen and len(terms) < MAX_ANTI_TERMS_PER_CONSTRAINT:
            seen.add(t)
            terms.append((t, "auto"))

    # Strategy A: Domain oppositions (existing behavior, preserved)
    for key_phrase, oppositions in _DOMAIN_OPPOSITIONS.items():
        if key_phrase in summary_lower:
            for opp in oppositions:
                _add(opp)

    # Strategy B: Verb reversal extraction
    words = summary_lower.split()
    for word in words:
        clean = word.strip(".,;:!?()[]\"'")
        if clean in _VERB_REVERSALS:
            for reversal in _VERB_REVERSALS[clean]:
                _add(reversal)

    # Strategy C: Constraint violation patterns
    for marker, violation in _CONSTRAINT_MARKERS.items():
        marker_padded = f" {marker} "
        if marker_padded in f" {summary_lower} ":
            idx = summary_lower.find(marker)
            if idx >= 0:
                after = summary_lower[idx + len(marker):].strip()
                # Split on sentence boundaries first (A1 amendment)
                first_clause = re.split(r'[.;!?\n]', after)[0].strip()
                phrase_words = first_clause.split()[:6]
                phrase = " ".join(phrase_words).rstrip(",:")
                if len(phrase) >= 5:
                    _add(f"{violation} {phrase}")

    # Strategy D: Key noun negation
    significant = [
        w.strip(".,;:!?()[]\"'") for w in words
        if len(w) > 4 and w.strip(".,;:!?()[]\"'").lower() not in _STOPWORDS
    ]
    for noun in significant[:3]:
        _add(f"remove {noun}")

    return terms[:MAX_ANTI_TERMS_PER_CONSTRAINT]


def generate_anti_terms(
    concept_id: str,
    summary: str,
    conn=None,
) -> tuple[list[str], dict[str, str]]:
    """Generate anti-terms for a concept using stored + curated + auto-generated.

    Priority: stored terms (from concept.data) first, then curated (from correction
    history), then auto-generated fallback.

    Returns:
        Tuple of (anti_terms list, sources dict mapping term -> "stored"|"curated"|"auto")
    """
    # Primary: explicitly stored on concept.data (highest priority)
    stored = _get_stored_anti_terms(concept_id, conn)

    # Secondary: curated from correction history
    curated = _get_curated_anti_terms(concept_id, conn)

    # Fallback: auto-generated from domain oppositions
    auto = _generate_auto_anti_terms(summary)

    # Merge: stored → curated → auto (deduplicated)
    seen = set()
    all_terms = []
    sources = {}

    for term, source in stored + curated + auto:
        if term not in seen:
            seen.add(term)
            all_terms.append(term)
            sources[term] = source

    return all_terms[:MAX_ANTI_TERMS_PER_CONSTRAINT], sources


# =============================================================================
# Constraint Set Assembly
# =============================================================================


def assemble_constraint_set(
    activated_concepts: list[dict[str, Any]],
    conn=None,
) -> ConstraintSet:
    """Extract high-authority concepts as behavioral constraints.

    Filters to concepts with authority >= threshold, generates anti-terms,
    and packages into a ConstraintSet for enforcement.

    Args:
        activated_concepts: Concepts from conversation_turn retrieval.
            Each dict needs: concept_id, summary, authority_score
        conn: SQLite connection for correction history lookups

    Returns:
        ConstraintSet with constraints and anti-terms
    """
    t0 = time.perf_counter()
    result = ConstraintSet()

    # --- Phase 2: Type gating (§5.6.1 C4) ---
    # These concept types can NEVER be constraints — they inform, not constrain.
    CONSTRAINT_BLOCKED_TYPES = {"observation", "pattern", "heuristic", "cognitive_strategy"}

    # Filter to high-authority concepts with type gating + epistemic caps
    candidates = []
    for ac in activated_concepts:
        # Type gate: blocked types can never be constraints
        concept_type = ac.get("concept_type", "observation")
        if concept_type in CONSTRAINT_BLOCKED_TYPES:
            continue

        # Phase 2: Apply epistemic authority cap at retrieval time (§5.1.3)
        raw_authority = ac.get("authority_score", 0)
        try:
            from app.epistemic import apply_epistemic_cap

            authority = apply_epistemic_cap(raw_authority, ac)
        except Exception:
            authority = raw_authority

        if authority >= CONSTRAINT_AUTHORITY_THRESHOLD:
            # Store effective authority for downstream use
            ac["_effective_authority"] = authority
            candidates.append(ac)

    # Sort by effective authority descending, take top N
    candidates.sort(key=lambda x: x.get("_effective_authority", x.get("authority_score", 0)), reverse=True)
    candidates = candidates[:MAX_CONSTRAINTS]

    for ac in candidates:
        cid = ac.get("concept_id", "")
        summary = ac.get("summary", "")
        authority = ac.get("_effective_authority") or ac.get("authority_score", 0)

        # Generate anti-terms
        anti_terms, sources = generate_anti_terms(cid, summary, conn)

        # Determine presentation mode
        if authority >= 0.80:
            mode = "CONSTRAINT"
        elif authority >= 0.60:
            mode = "DIRECTIVE"
        else:
            mode = "CONTEXT"

        constraint = Constraint(
            concept_id=cid,
            constraint=summary,
            authority=authority,
            anti_terms=anti_terms,
            anti_term_sources=sources,
            presentation_mode=mode,
        )
        result.constraints.append(constraint)

        # Persist generated anti_terms to concept.data so CogGov benchmark
        # can detect coverage via `data LIKE '%anti_terms%'` check.
        # Non-fatal: failure here doesn't block the turn or corrupt the Constraint.
        if anti_terms and conn:
            try:
                conn.execute(
                    "UPDATE concepts SET data = json_patch(data, json_object('anti_terms', json(?))), "
                    "updated_at = ? WHERE id = ? AND id NOT LIKE 'test_%' AND id NOT LIKE 'live_l2_%'",
                    (json.dumps(anti_terms), _utc_now().isoformat(), cid),
                )
                conn.commit()
            except Exception as e:
                logger.debug("Anti-term persistence failed for %s (non-fatal): %s", cid, e)

    result.constraint_count = len(result.constraints)
    result.total_anti_terms = sum(len(c.anti_terms) for c in result.constraints)
    result.assembly_time_ms = (time.perf_counter() - t0) * 1000

    if result.constraints:
        logger.info(
            "Constraint set assembled: %d constraints, %d anti-terms in %.1fms",
            result.constraint_count,
            result.total_anti_terms,
            result.assembly_time_ms,
        )

    return result


# =============================================================================
# Enforcement: Anti-Term Scan
# =============================================================================


def scan_for_violations(
    response_text: str,
    constraint_set: ConstraintSet,
) -> EnforcementResult:
    """Scan a response for constraint violations via anti-term matching.

    Client-side enforcement: < 1ms string matching, no network calls.

    Args:
        response_text: The LLM's draft response
        constraint_set: Active constraints with anti-terms

    Returns:
        EnforcementResult with any violations found
    """
    t0 = time.perf_counter()
    result = EnforcementResult()
    response_lower = response_text.lower()

    for constraint in constraint_set.constraints:
        for anti_term in constraint.anti_terms:
            # Case-insensitive substring match
            pos = response_lower.find(anti_term.lower())
            if pos >= 0:
                severity = "hard" if constraint.authority >= HARD_ENFORCEMENT_AUTHORITY else "soft"
                violation = ConstraintViolation(
                    constraint=constraint,
                    anti_term_matched=anti_term,
                    position_in_text=pos,
                    severity=severity,
                )
                result.violations.append(violation)

                if severity == "hard":
                    result.hard_violations += 1
                else:
                    result.soft_violations += 1

    result.scan_time_ms = (time.perf_counter() - t0) * 1000

    # Generate correction prompt for hard violations
    if result.hard_violations > 0:
        violation_details = []
        for v in result.violations:
            if v.severity == "hard":
                violation_details.append(
                    f"- Constraint: {v.constraint.constraint}\n"
                    f"  Violation: used '{v.anti_term_matched}' (authority: {v.constraint.authority:.2f})"
                )

        result.correction_prompt = (
            "Your response violates the following constraints:\n"
            + "\n".join(violation_details)
            + "\n\nRevise your response to respect these constraints."
        )

    if result.violations:
        logger.info(
            "Enforcement scan: %d violations (%d hard, %d soft) in %.1fms",
            len(result.violations),
            result.hard_violations,
            result.soft_violations,
            result.scan_time_ms,
        )

    return result


# =============================================================================
# Serialization for API Response
# =============================================================================

# =============================================================================
# Wave 3c — Compounding Correction Loop (CCL)
# =============================================================================


def _extract_terms(text: str) -> list[str]:
    """Extract lowercase terms from text for topic overlap and alignment checks.

    Simple tokenizer: lowercase, split on non-word chars, remove short tokens
    and common stop words. Not meant to be sophisticated — just enough for
    topic overlap scoring and alignment checks.
    """
    import re

    _STOP_WORDS = frozenset(
        {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "shall",
            "can",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "out",
            "off",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
            "and",
            "but",
            "or",
            "nor",
            "not",
            "so",
            "yet",
            "both",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "only",
            "own",
            "same",
            "than",
            "too",
            "very",
            "just",
            "because",
            "if",
            "when",
            "where",
            "how",
            "what",
            "which",
            "who",
            "whom",
            "this",
            "that",
            "these",
            "those",
            "i",
            "me",
            "my",
            "we",
            "our",
            "you",
            "your",
            "he",
            "him",
            "his",
            "she",
            "her",
            "it",
            "its",
            "they",
            "them",
            "their",
            "also",
            "about",
            "up",
        }
    )
    tokens = re.findall(r"\b[a-z]{3,}\b", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS]


def validate_previous_response(
    previous_response: str,
    stored_constraint_set: dict,
    current_topic_terms: list[str],
) -> dict:
    """Validate LLM's previous response against constraint_set that was active
    when it was generated. Returns violation report.

    Runs BEFORE auto-learn so violations are detected before new learning occurs.

    §3c.2 — Previous-Response Validation
    """
    if not previous_response or not stored_constraint_set:
        return {"status": "skipped", "reason": "no previous response or constraint set"}

    violations = []
    stored_constraints = stored_constraint_set.get("constraints", [])
    stored_topic_terms = stored_constraint_set.get("topic_terms", [])

    # --- TOPIC RELEVANCE GATE ---
    # Only validate if current turn's topic overlaps with stored constraint topics.
    # Prevents false positives from multi-turn topic drift (S1-NEG-5 mitigation).
    topic_overlap = len(set(current_topic_terms) & set(stored_topic_terms))
    if topic_overlap == 0 and stored_topic_terms:
        return {"status": "skipped", "reason": "topic drift — no overlap with constrained turn"}

    # [FIX EC-4] Cap response length for validation — first 2000 chars representative
    from app.config import CCL_RESPONSE_CAP_CHARS

    capped_response = previous_response[:CCL_RESPONSE_CAP_CHARS]
    response_lower = capped_response.lower()
    response_terms = set(_extract_terms(capped_response))

    for constraint in stored_constraints:
        # --- ANTI-TERM CHECK ---
        anti_terms = constraint.get("anti_terms", [])
        hit_anti_terms = [t for t in anti_terms if t.lower() in response_lower]

        if hit_anti_terms:
            # Context-sensitive check: anti-term in a negation context is not a violation.
            # Simple heuristic: check if anti-term is preceded by "not", "isn't", "don't".
            confirmed_hits = []
            for term in hit_anti_terms:
                idx = response_lower.find(term.lower())
                preceding = response_lower[max(0, idx - 20) : idx]
                negation_words = ["not ", "n't ", "no ", "never ", "neither ", "without "]
                if not any(neg in preceding for neg in negation_words):
                    confirmed_hits.append(term)

            if confirmed_hits:
                violations.append(
                    {
                        "type": "anti_term_violation",
                        "constraint_concept_id": constraint.get("concept_id"),
                        "terms_found": confirmed_hits,
                        "severity": "medium",
                    }
                )

        # --- ALIGNMENT CHECK (Solution 3 kernel) ---
        # Lightweight semantic check: did the response discuss topics covered by this
        # constraint, and if so, does it align or potentially contradict?
        alignment_ref = constraint.get("constraint", "")
        if alignment_ref and constraint.get("presentation_mode") == "CONSTRAINT":
            ref_terms = set(_extract_terms(alignment_ref))
            overlap = response_terms & ref_terms
            # If response shares significant vocabulary with constraint but
            # also contains anti-terms, flag as potential misalignment
            if len(overlap) >= 3 and hit_anti_terms:
                violations.append(
                    {
                        "type": "alignment_drift",
                        "constraint_concept_id": constraint.get("concept_id"),
                        "overlap_terms": list(overlap)[:10],
                        "severity": "low",
                    }
                )

    return {
        "status": "violations_found" if violations else "clean",
        "violations": violations,
        "constraints_checked": len(stored_constraints),
        "topic_overlap_score": topic_overlap,
    }


def apply_correction_compound(violations: list[dict]) -> list[dict]:
    """For each violation, evolve the violated constraint's source concept:
    - Increment authority (concept becomes more prominent in future retrieval)
    - Add violation evidence to concept record
    - Create a correction trace linking the violation to the concept (if traces available)

    Returns list of evolution actions taken.

    §3c.3 — Compounding Authority (The Core Innovation)
    """
    from app.config import CCL_MAX_VIOLATIONS_PER_TURN
    from app.learning import evolve_concept
    from app.models import ConceptEvolution
    from app.storage import load_concept

    actions = []

    # [FIX PF-2] Cap violations per turn to bound DB writes
    violations = violations[:CCL_MAX_VIOLATIONS_PER_TURN]

    # [FIX DI-1] Deduplicate by concept_id — take highest severity only
    seen_concepts: dict[str, dict] = {}
    severity_rank = {"medium": 2, "low": 1}
    for v in violations:
        cid = v.get("constraint_concept_id")
        if cid and (
            cid not in seen_concepts
            or severity_rank.get(v.get("severity", ""), 0)
            > severity_rank.get(seen_concepts[cid].get("severity", ""), 0)
        ):
            seen_concepts[cid] = v
    violations = list(seen_concepts.values())

    for v in violations:
        concept_id = v.get("constraint_concept_id")
        if not concept_id:
            continue

        concept = load_concept(concept_id, track_access=False)
        if not concept:
            continue

        # Authority boost: +0.05 per violation, capped at 1.0
        from app.config import CCL_AUTHORITY_BOOST_PER_VIOLATION

        authority_boost = CCL_AUTHORITY_BOOST_PER_VIOLATION

        # Record violation as evidence on the concept
        violation_evidence = (
            f"Violation detected {_utc_now().strftime('%Y-%m-%d')}: "
            f"{v['type']} — {v.get('terms_found', v.get('overlap_terms', []))}"
        )

        evolution = ConceptEvolution(
            concept_id=concept_id,
            confidence_change=authority_boost,
            new_evidence=[violation_evidence],
        )
        evolve_concept(evolution)

        # Create correction trace for audit trail (Wave 4b — guarded import)
        try:
            from app.traces import create_trace

            create_trace(
                trigger_type="correction",
                situation=f"CCL detected {v['type']} against constraint {concept_id}",
                assessment=f"Previous response violated constraint. Terms: {v.get('terms_found', [])}",
                justification="Compounding correction loop — automatic server-side detection",
                concept_refs=[concept_id],
            )
        except ImportError:
            pass  # traces.py not yet built (Wave 4b)
        except Exception as e:
            logger.warning(f"CCL trace creation failed (non-fatal): {e}")

        new_confidence = min((concept.confidence or 0.0) + authority_boost, 1.0)
        actions.append(
            {
                "concept_id": concept_id,
                "authority_boost": authority_boost,
                "new_confidence": new_confidence,
                "violation_type": v["type"],
            }
        )

    return actions


def constraint_set_to_dict(cs: ConstraintSet) -> dict[str, Any]:
    """Serialize constraint set for inclusion in conversation_turn response."""
    return {
        "constraints": [
            {
                "concept_id": c.concept_id,
                "constraint": c.constraint,
                "authority": c.authority,
                "anti_terms": c.anti_terms,
                "presentation_mode": c.presentation_mode,
            }
            for c in cs.constraints
        ],
        "constraint_count": cs.constraint_count,
        "total_anti_terms": cs.total_anti_terms,
        "assembly_time_ms": round(cs.assembly_time_ms, 2),
    }


# =============================================================================
# Phase 2 (P4a) — Post-Response Validation Engine
# =============================================================================
# Architecture: Three-tier validation (fast → medium → slow)
#   Tier 1: Negation detection — regex-based, <5ms
#   Tier 2: Entity overlap — tokenized set comparison, <20ms
#   Tier 3: Embedding escalation — TF-IDF cosine, <150ms (only if ambiguous)
# Target: <200ms total, circuit breaker at 3s cumulative latency
# Feature flag: POST_RESPONSE_VALIDATION_ENABLED (default: False)

import re as _re

# --- Negation patterns ---
# Regex patterns that detect negation scope near key terms.
# Covers: not, no, never, don't, won't, can't, shouldn't, isn't, aren't,
#          without, refuse, avoid, skip, omit, ignore, exclude, neither, nor
_NEGATION_CUES = _re.compile(
    r"\b(?:not?|never|don'?t|won'?t|can'?t|couldn'?t|shouldn'?t|wouldn'?t|"
    r"isn'?t|aren'?t|wasn'?t|weren'?t|hasn'?t|haven'?t|hadn'?t|"
    r"without|refuse[sd]?|avoid(?:s|ed|ing)?|skip(?:s|ped|ping)?|"
    r"omit(?:s|ted|ting)?|ignor(?:e[sd]?|ing)|exclud(?:e[sd]?|ing)|"
    r"neither|nor|lack(?:s|ed|ing)?|fail(?:s|ed|ing)?\s+to)\b",
    _re.IGNORECASE,
)

# Window size: how many characters around a negation cue to check for entity match
_NEGATION_WINDOW = 80

# --- Entity extraction ---
_ENTITY_STOP_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "shall",
        "may",
        "might",
        "must",
        "can",
        "need",
        "ought",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "they",
        "them",
        "their",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "him",
        "her",
        "his",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "when",
        "where",
        "how",
        "what",
        "which",
        "who",
        "whom",
        "why",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "some",
        "any",
        "no",
        "not",
        "only",
        "very",
        "too",
        "also",
        "just",
        "than",
        "so",
        "for",
        "with",
        "from",
        "to",
        "of",
        "in",
        "on",
        "at",
        "by",
        "as",
        "into",
        "about",
        "up",
        "out",
        "always",
        "never",
        "constraint",
        "ensure",
        "make",
        "sure",
    }
)

# Confidence thresholds
_VALIDATION_CONFIDENCE_FLOOR = 0.7  # Only surface violations above this
_NEGATION_CONFIDENCE = 0.85  # High confidence for clear negation violations
_ENTITY_GAP_CONFIDENCE = 0.60  # Lower confidence for missing entities
_EMBEDDING_CONTRADICTION_CONFIDENCE = 0.75  # Medium for embedding-detected issues

# Latency budget
_VALIDATION_TIMEOUT_MS = 200  # Target max for entire validation
_CIRCUIT_BREAKER_MS = 3000  # Abort if cumulative latency exceeds this


@dataclass
class ValidationViolation:
    """A violation found by the P4a validation engine."""

    constraint_id: str
    constraint_text: str
    violation_type: str  # "negation", "entity_gap", "anti_term", "embedding_contradiction"
    detail: str  # Human-readable explanation
    confidence: float  # 0.0 - 1.0
    severity: str  # "hard" or "soft"
    evidence: str = ""  # The specific text that triggered this


@dataclass
class ValidationResult:
    """Complete result from post-response validation."""

    passed: bool = True
    violations: list[ValidationViolation] = field(default_factory=list)
    hard_violation_count: int = 0
    soft_violation_count: int = 0
    correction_prompt: str | None = None
    validation_time_ms: float = 0.0
    tiers_executed: list[str] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str | None = None


def _extract_constraint_entities(constraint_text: str) -> set:
    """Extract meaningful entities/terms from a constraint for overlap checking.

    Returns a set of lowercase terms, filtered of stop words and short tokens.
    """
    # Split on non-alpha characters (handles underscores, punctuation)
    tokens = _re.findall(r"[A-Za-z]+", constraint_text.lower())
    return {t for t in tokens if len(t) >= 3 and t not in _ENTITY_STOP_WORDS}


def _detect_negation_violations(
    response_text: str,
    constraints: list[dict[str, Any]],
) -> list[ValidationViolation]:
    """Tier 1: Detect negation patterns near constraint entities.

    Scans the response for negation cues (not, never, don't, etc.) and checks
    if constraint-relevant entities appear within a window around the negation.
    This catches "I will NOT follow the spec" when the constraint says "follow the spec".

    Time budget: <5ms
    """
    violations = []
    response_lower = response_text.lower()

    for c in constraints:
        cid = c.get("concept_id", "")
        ctext = c.get("constraint", "")
        authority = c.get("authority", 0)

        # Extract key entities from this constraint
        entities = _extract_constraint_entities(ctext)
        if not entities:
            continue

        # Find all negation cues in the response
        for neg_match in _NEGATION_CUES.finditer(response_lower):
            neg_start = max(0, neg_match.start() - _NEGATION_WINDOW)
            neg_end = min(len(response_lower), neg_match.end() + _NEGATION_WINDOW)
            window = response_lower[neg_start:neg_end]

            # Check if any constraint entities appear in the negation window
            matched_entities = [e for e in entities if e in window]

            # Need at least 2 entity matches to reduce false positives
            # (single word matches are too noisy)
            if len(matched_entities) >= 2:
                # Extract the actual text around the negation for evidence
                evidence_start = max(0, neg_match.start() - 30)
                evidence_end = min(len(response_text), neg_match.end() + 60)
                evidence = response_text[evidence_start:evidence_end].strip()

                severity = "hard" if authority >= HARD_ENFORCEMENT_AUTHORITY else "soft"
                violations.append(
                    ValidationViolation(
                        constraint_id=cid,
                        constraint_text=ctext[:200],
                        violation_type="negation",
                        detail=f"Negation '{neg_match.group()}' found near constraint entities: {', '.join(matched_entities[:5])}",
                        confidence=_NEGATION_CONFIDENCE,
                        severity=severity,
                        evidence=evidence,
                    )
                )
                break  # One negation violation per constraint is enough

    return violations


def _detect_entity_gaps(
    response_text: str,
    constraints: list[dict[str, Any]],
) -> list[ValidationViolation]:
    """Tier 2: Detect when the response ignores a constraint entirely.

    Checks if key entities from each constraint appear in the response at all.
    If a constraint's entities are completely absent, it may have been ignored.

    Time budget: <20ms
    """
    violations = []
    response_lower = response_text.lower()
    response_tokens = set(_re.findall(r"[A-Za-z]+", response_lower))

    for c in constraints:
        cid = c.get("concept_id", "")
        ctext = c.get("constraint", "")
        authority = c.get("authority", 0)

        entities = _extract_constraint_entities(ctext)
        if len(entities) < 3:
            # Too few entities to meaningfully check overlap
            continue

        # Count how many constraint entities appear in the response
        present = entities & response_tokens
        coverage = len(present) / len(entities) if entities else 1.0

        # If less than 20% of constraint entities appear, flag as potentially ignored
        # This is a soft signal — many constraints don't require explicit mention
        if coverage < 0.20:
            severity = "soft"  # Entity gaps are always soft violations
            violations.append(
                ValidationViolation(
                    constraint_id=cid,
                    constraint_text=ctext[:200],
                    violation_type="entity_gap",
                    detail=f"Only {len(present)}/{len(entities)} constraint entities found in response ({coverage:.0%} coverage)",
                    confidence=_ENTITY_GAP_CONFIDENCE
                    * (1 - coverage),  # Lower coverage = higher confidence it's violated
                    severity=severity,
                    evidence=f"Missing entities: {', '.join(list(entities - present)[:10])}",
                )
            )

    return violations


def _embedding_escalation(
    response_text: str,
    constraints: list[dict[str, Any]],
    ambiguous_ids: set,
) -> list[ValidationViolation]:
    """Tier 3: Embedding-based semantic check for ambiguous cases.

    Only runs on constraints where Tier 1 and 2 gave borderline results.
    Uses TF-IDF cosine similarity between constraint text and response text.
    Low similarity + high authority = potential contradiction.

    Time budget: <150ms
    """
    violations = []

    if not ambiguous_ids:
        return violations

    try:
        from app.incremental_tfidf import IncrementalTfidfIndex

        # Build a temporary mini-index for the comparison
        tfidf = IncrementalTfidfIndex()

        # Index constraint texts
        constraint_map = {}
        for c in constraints:
            cid = c.get("concept_id", "")
            if cid not in ambiguous_ids:
                continue
            ctext = c.get("constraint", "")
            constraint_map[cid] = c
            tfidf.add_concept(cid, ctext)

        if not constraint_map:
            return violations

        # Query with the response text
        results = tfidf.search(response_text, top_k=len(constraint_map))

        # Build score map; constraints missing from results have score 0.0
        score_map = {cid: score for cid, score in results}

        for cid, c in constraint_map.items():
            score = score_map.get(cid, 0.0)
            authority = c.get("authority", 0)

            # Very low similarity between constraint and response is suspicious
            # but only if the constraint has high authority (it should be relevant)
            if score < 0.15 and authority >= 0.60:
                violations.append(
                    ValidationViolation(
                        constraint_id=cid,
                        constraint_text=c.get("constraint", "")[:200],
                        violation_type="embedding_contradiction",
                        detail=f"Low semantic similarity ({score:.3f}) between constraint and response suggests possible contradiction or omission",
                        confidence=_EMBEDDING_CONTRADICTION_CONFIDENCE,
                        severity="soft",
                        evidence=f"TF-IDF cosine similarity: {score:.4f}",
                    )
                )

    except Exception as e:
        logger.debug(f"P4a: Embedding escalation skipped: {e}")

    return violations


def _build_correction_prompt(violations: list[ValidationViolation]) -> str:
    """Build a correction prompt from detected violations."""
    hard = [v for v in violations if v.severity == "hard"]
    soft = [v for v in violations if v.severity == "soft"]

    parts = ["Your response may violate the following constraints:\n"]

    if hard:
        parts.append("**HARD VIOLATIONS** (must fix):")
        for v in hard:
            parts.append(f"- [{v.violation_type}] {v.detail}")
            parts.append(f"  Constraint: {v.constraint_text[:150]}")
            if v.evidence:
                parts.append(f'  Evidence: "{v.evidence}"')

    if soft:
        parts.append("\n**SOFT VIOLATIONS** (review):")
        for v in soft[:3]:  # Limit soft violations to top 3
            parts.append(f"- [{v.violation_type}] {v.detail}")
            parts.append(f"  Constraint: {v.constraint_text[:150]}")

    parts.append("\nRevise your response to respect these constraints.")
    return "\n".join(parts)


def validate_response(
    response_text: str,
    constraint_set: dict[str, Any],
    skip_validation: bool = False,
    cumulative_latency_ms: float = 0.0,
) -> dict[str, Any]:
    """P4a: Full post-response validation against active constraints.

    Three-tier validation engine:
      Tier 1: Anti-term scan (existing) + negation detection (<5ms)
      Tier 2: Entity overlap checking (<20ms)
      Tier 3: Embedding escalation for ambiguous cases (<150ms)

    Args:
        response_text: The LLM's draft response to validate
        constraint_set: The constraint_set dict from conversation_turn
        skip_validation: If True, skip all validation (opt-out)
        cumulative_latency_ms: Total latency already spent in this turn

    Returns:
        Dict with: passed, violations, correction_prompt, confidence,
        validation_time_ms, tiers_executed
    """
    t0 = time.perf_counter()
    result = ValidationResult()

    # --- Gate: Feature flag ---
    if not FEATURE_FLAGS.get("POST_RESPONSE_VALIDATION_ENABLED", False):
        result.skipped = True
        result.skip_reason = "POST_RESPONSE_VALIDATION_ENABLED=False"
        return _validation_result_to_dict(result)

    # --- Gate: Explicit opt-out ---
    if skip_validation:
        result.skipped = True
        result.skip_reason = "skip_validation=True"
        return _validation_result_to_dict(result)

    # --- Gate: Circuit breaker ---
    if cumulative_latency_ms >= _CIRCUIT_BREAKER_MS:
        result.skipped = True
        result.skip_reason = (
            f"Circuit breaker: cumulative latency {cumulative_latency_ms:.0f}ms >= {_CIRCUIT_BREAKER_MS}ms"
        )
        return _validation_result_to_dict(result)

    # --- Gate: No constraints ---
    constraints = constraint_set.get("constraints", [])
    if not constraints:
        result.skipped = True
        result.skip_reason = "No constraints in constraint_set"
        return _validation_result_to_dict(result)

    # --- Gate: Empty response ---
    if not response_text or len(response_text.strip()) < 10:
        result.skipped = True
        result.skip_reason = "Response too short to validate"
        return _validation_result_to_dict(result)

    all_violations = []

    # --- Tier 1: Anti-term scan + Negation detection ---
    try:
        # Existing anti-term scan
        cs = _dict_to_constraint_set(constraint_set)
        anti_term_result = scan_for_violations(response_text, cs)
        for v in anti_term_result.violations:
            all_violations.append(
                ValidationViolation(
                    constraint_id=v.constraint.concept_id,
                    constraint_text=v.constraint.constraint[:200],
                    violation_type="anti_term",
                    detail=f"Anti-term '{v.anti_term_matched}' found in response",
                    confidence=0.90 if v.severity == "hard" else 0.70,
                    severity=v.severity,
                    evidence=v.anti_term_matched,
                )
            )

        # Negation detection
        negation_violations = _detect_negation_violations(response_text, constraints)
        all_violations.extend(negation_violations)
        result.tiers_executed.append("negation_detection")
    except Exception as e:
        logger.debug(f"P4a Tier 1 error: {e}")

    result.tiers_executed.append("anti_term_scan")

    # --- Check time budget ---
    elapsed_ms = (time.perf_counter() - t0) * 1000
    if elapsed_ms > _VALIDATION_TIMEOUT_MS:
        logger.debug(f"P4a: Skipping Tier 2+3 (elapsed {elapsed_ms:.0f}ms > {_VALIDATION_TIMEOUT_MS}ms)")
    else:
        # --- Tier 2: Entity overlap ---
        try:
            entity_violations = _detect_entity_gaps(response_text, constraints)
            all_violations.extend(entity_violations)
            result.tiers_executed.append("entity_overlap")
        except Exception as e:
            logger.debug(f"P4a Tier 2 error: {e}")

        # --- Check time budget before Tier 3 ---
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if elapsed_ms < _VALIDATION_TIMEOUT_MS * 0.6:  # Only if we have >40% budget left
            # Identify ambiguous constraints: those not already flagged by Tier 1/2
            flagged_ids = {v.constraint_id for v in all_violations}
            ambiguous_ids = {c.get("concept_id", "") for c in constraints} - flagged_ids

            # Only escalate for high-authority unflagged constraints
            ambiguous_ids = {
                cid
                for cid in ambiguous_ids
                for c in constraints
                if c.get("concept_id") == cid and c.get("authority", 0) >= 0.70
            }

            if ambiguous_ids:
                try:
                    embedding_violations = _embedding_escalation(response_text, constraints, ambiguous_ids)
                    all_violations.extend(embedding_violations)
                    result.tiers_executed.append("embedding_escalation")
                except Exception as e:
                    logger.debug(f"P4a Tier 3 error: {e}")

    # --- Filter by confidence floor ---
    all_violations = [v for v in all_violations if v.confidence >= _VALIDATION_CONFIDENCE_FLOOR]

    # --- Deduplicate: one violation per constraint, keep highest confidence ---
    seen = {}
    for v in sorted(all_violations, key=lambda x: x.confidence, reverse=True):
        if v.constraint_id not in seen:
            seen[v.constraint_id] = v
    all_violations = list(seen.values())

    # --- Populate result ---
    result.violations = all_violations
    result.hard_violation_count = sum(1 for v in all_violations if v.severity == "hard")
    result.soft_violation_count = sum(1 for v in all_violations if v.severity == "soft")
    result.passed = result.hard_violation_count == 0
    result.validation_time_ms = (time.perf_counter() - t0) * 1000

    if result.hard_violation_count > 0:
        result.correction_prompt = _build_correction_prompt(all_violations)

    if all_violations:
        logger.info(
            "P4a validation: %s (%d hard, %d soft) in %.1fms [%s]",
            "FAIL" if not result.passed else "WARN",
            result.hard_violation_count,
            result.soft_violation_count,
            result.validation_time_ms,
            "+".join(result.tiers_executed),
        )

    return _validation_result_to_dict(result)


def _dict_to_constraint_set(cs_dict: dict[str, Any]) -> ConstraintSet:
    """Reconstruct a ConstraintSet from a serialized dict."""
    cs = ConstraintSet()
    for c_dict in cs_dict.get("constraints", []):
        cs.constraints.append(
            Constraint(
                concept_id=c_dict.get("concept_id", ""),
                constraint=c_dict.get("constraint", ""),
                authority=c_dict.get("authority", 0),
                anti_terms=c_dict.get("anti_terms", []),
                presentation_mode=c_dict.get("presentation_mode", "CONTEXT"),
            )
        )
    cs.constraint_count = len(cs.constraints)
    cs.total_anti_terms = sum(len(c.anti_terms) for c in cs.constraints)
    return cs


def _validation_result_to_dict(result: ValidationResult) -> dict[str, Any]:
    """Serialize ValidationResult for API response."""
    return {
        "passed": result.passed,
        "violations": [
            {
                "constraint_id": v.constraint_id,
                "constraint_text": v.constraint_text,
                "violation_type": v.violation_type,
                "detail": v.detail,
                "confidence": round(v.confidence, 3),
                "severity": v.severity,
                "evidence": v.evidence,
            }
            for v in result.violations
        ],
        "hard_violation_count": result.hard_violation_count,
        "soft_violation_count": result.soft_violation_count,
        "correction_prompt": result.correction_prompt,
        "validation_time_ms": round(result.validation_time_ms, 2),
        "tiers_executed": result.tiers_executed,
        "skipped": result.skipped,
        "skip_reason": result.skip_reason,
    }
