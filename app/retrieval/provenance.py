"""Provenance-Weighted Trust (PWT) — Wave 4b.4

4-layer defense against knowledge poisoning:
  PWT-1: Evidence source classification (7 source types with weights)
  PWT-2: Corroboration requirement (high-confidence needs 2+ substantive sources)
  PWT-3: Velocity anomaly detection (flag suspicious fast-high-confidence)
  PWT-4: Provenance audit (DEFERRED to Wave 5 / reflection)

Key output: effective_confidence = min(raw, test_ceiling, provenance_score)

NEW file — no backward compat concerns.
"""

import logging
import re

from app.core.models import Concept

logger = logging.getLogger(__name__)


# =============================================================================
# PWT-1: Evidence Source Classification
# =============================================================================

# Source type weights — imported from config but defined here as fallback
try:
    from app.core.config import PWT_SOURCE_WEIGHTS, PWT_SUBSTANTIVE_SOURCE_TYPES
except ImportError:
    PWT_SOURCE_WEIGHTS = {
        "cross_corroborated": 0.95,
        "observed_behavior": 0.90,
        "user_repeated": 0.85,
        "document_extracted": 0.80,
        "user_explicit": 0.70,
        "unclassified": 0.50,
        "self_generated": 0.40,
    }
    PWT_SUBSTANTIVE_SOURCE_TYPES = {"user_explicit", "user_repeated", "document_extracted", "observed_behavior"}


def classify_evidence_source(evidence_item) -> str:
    """Classify a single evidence item into one of 7 source types.

    Works with both structured Evidence objects and legacy string evidence.

    Returns:
        Source type string from PWT_SOURCE_WEIGHTS keys.
    """
    # Structured evidence (has source_type + extraction_source)
    if hasattr(evidence_item, "source_type"):
        st = evidence_item.source_type
        es = getattr(evidence_item, "extraction_source", "heuristic")
        ct = getattr(evidence_item, "corroboration_type", None)
        content = getattr(evidence_item, "content", "") or ""
        source_reference = getattr(evidence_item, "source_reference", "") or ""
        file_path = getattr(evidence_item, "file_path", "") or ""

        if ct == "cross_source":
            return "cross_corroborated"
        if st == "observation":
            return "observed_behavior"
        if st == "document":
            return "document_extracted"
        if st == "external_data":
            return "document_extracted"
        if st == "conversation":
            # Check if client-extracted (user stated) vs heuristic (inferred)
            if es == "client":
                return "user_explicit"
            return "self_generated"
        if st == "inference":
            return "self_generated"
        if st and st not in {"legacy", "unknown"}:
            return "unclassified"
        fallback = _classify_legacy_evidence(" ".join(part for part in (content, source_reference, file_path) if part))
        if fallback != "unclassified":
            return fallback
        return "unclassified"

    # Dict evidence (from JSON)
    if isinstance(evidence_item, dict):
        st = evidence_item.get("source_type", "")
        es = evidence_item.get("extraction_source", "heuristic")
        ct = evidence_item.get("corroboration_type", None)
        content = evidence_item.get("content", "") or ""
        source_reference = evidence_item.get("source_reference", "") or ""
        file_path = evidence_item.get("file_path", "") or ""

        if ct == "cross_source":
            return "cross_corroborated"
        if st == "observation":
            return "observed_behavior"
        if st in ("document", "external_data"):
            return "document_extracted"
        if st == "conversation" and es == "client":
            return "user_explicit"
        if st in ("conversation", "inference"):
            return "self_generated"
        fallback = _classify_legacy_evidence(" ".join(part for part in (content, source_reference, file_path) if part))
        if fallback != "unclassified":
            return fallback
        return "unclassified"

    # Legacy string evidence — use regex heuristics [FIX M1]
    if isinstance(evidence_item, str):
        return _classify_legacy_evidence(evidence_item)

    return "unclassified"


# Regex patterns for legacy evidence migration [FIX M1]
_LEGACY_PATTERNS = [
    (re.compile(r"(?:user|client)\s+(?:said|stated|mentioned|asked|requested)", re.I), "user_explicit"),
    (re.compile(r"(?:observed|behavior|outcome|result|measured)", re.I), "observed_behavior"),
    (re.compile(r"(?:document|file|spec|readme|doc|paper|article)", re.I), "document_extracted"),
    (re.compile(r"\b[a-z0-9_.-]+\.(?:md|mdx|rst|txt|doc|docx|pdf|html|yaml|yml|json)\b", re.I), "document_extracted"),
    (re.compile(r"(?:^|[\s/])(docs?|specs?|retros?|guides?|plans?|roadmaps?|changelog)(?:/|[\s:]|$)", re.I), "document_extracted"),
    (re.compile(r"(?:inferred|deduced|concluded|extracted|heuristic)", re.I), "self_generated"),
]


def _classify_legacy_evidence(text: str) -> str:
    """Classify legacy string evidence via regex heuristics [FIX M1]."""
    for pattern, source_type in _LEGACY_PATTERNS:
        if pattern.search(text):
            return source_type
    return "unclassified"


def compute_provenance_score(concept: Concept) -> float:
    """Compute provenance score as weighted average of evidence source weights.

    For pre-4b concepts (provenance_migrated=False), returns default 0.7 [FIX F1].

    Args:
        concept: Concept with evidence list

    Returns:
        Provenance score [0.0, 1.0]
    """
    # Migration grace [FIX F1]
    if not getattr(concept, "provenance_migrated", True):
        return 0.7

    evidence = concept.evidence
    if not evidence:
        return 0.5  # No evidence → default to unclassified weight

    weights = []
    for item in evidence:
        source_type = classify_evidence_source(item)
        weight = PWT_SOURCE_WEIGHTS.get(source_type, 0.5)
        weights.append(weight)

    return sum(weights) / len(weights)


# =============================================================================
# Test-Status Ceiling (4-tier) — §4b.1
# =============================================================================


def compute_test_status_ceiling(concept: Concept) -> float:
    """Compute test-status confidence ceiling (4-tier).

    | Status         | Ceiling | Criteria                                              |
    |----------------|---------|-------------------------------------------------------|
    | corrected      | 0.0     | has_correction=True                                   |
    | untested       | 0.5     | access_count <= 3, no evolutions                      |
    | lightly_tested | 0.75    | access_count > 3 OR has evolutions, no corrections    |
    | validated      | 1.0     | access_count > 3 AND has evolutions AND no corrections|
    """
    has_correction = getattr(concept, "has_correction", False)
    access_count = concept.access_count or 0
    version = concept.version or "v1"

    # Parse version number for evolution check
    try:
        version_num = int(version.lstrip("v")) if version.startswith("v") else int(version)
    except (ValueError, TypeError):
        version_num = 1
    has_evolutions = version_num > 1

    if has_correction:
        return 0.0  # corrected — concept had contradiction flagged

    if access_count > 3 and has_evolutions:
        return 1.0  # validated

    if access_count > 3 or has_evolutions:
        return 0.75  # lightly_tested

    return 0.5  # untested


def get_test_status_label(concept: Concept) -> str:
    """Return human-readable test status label."""
    ceiling = compute_test_status_ceiling(concept)
    if ceiling == 0.0:
        return "corrected"
    if ceiling == 0.5:
        return "untested"
    if ceiling == 0.75:
        return "lightly_tested"
    return "validated"


# =============================================================================
# PWT-2: Corroboration Requirement [FIX G2]
# =============================================================================


def check_corroboration(concept: Concept) -> bool:
    """Check if concept meets corroboration requirement.

    Concepts with confidence > 0.8 need 2+ independent substantive source types.
    Only user_explicit, user_repeated, document_extracted, observed_behavior count.

    Returns:
        True if corroborated (or below threshold), False if needs corroboration.
    """
    try:
        from app.core.config import PWT_CORROBORATION_CONFIDENCE_THRESHOLD
    except ImportError:
        PWT_CORROBORATION_CONFIDENCE_THRESHOLD = 0.8

    if concept.confidence <= PWT_CORROBORATION_CONFIDENCE_THRESHOLD:
        return True  # Below threshold — no corroboration needed

    if not concept.evidence:
        return False  # High confidence, no evidence → uncorroborated

    # Count unique substantive source types
    substantive_types = set()
    for item in concept.evidence:
        source_type = classify_evidence_source(item)
        if source_type in PWT_SUBSTANTIVE_SOURCE_TYPES:
            substantive_types.add(source_type)

    return len(substantive_types) >= 2


# =============================================================================
# PWT-3: Velocity Anomaly Detection [FIX O2]
# =============================================================================


def check_velocity_anomaly(concept: Concept) -> str | None:
    """Detect velocity anomaly: high confidence + single evidence + low-trust source.

    Flag concepts created with confidence >0.7, single evidence source,
    and a low-trust source type (self_generated or unclassified).

    Returns:
        Alert message string if anomaly detected, None otherwise.
    """
    try:
        from app.core.config import PWT_VELOCITY_CLAMP, PWT_VELOCITY_CONFIDENCE_THRESHOLD
    except ImportError:
        PWT_VELOCITY_CONFIDENCE_THRESHOLD = 0.7
        PWT_VELOCITY_CLAMP = 0.6

    if concept.confidence <= PWT_VELOCITY_CONFIDENCE_THRESHOLD:
        return None

    evidence = concept.evidence
    if not evidence or len(evidence) != 1:
        return None  # Multiple evidence or none — not a velocity anomaly

    # Check if single evidence is low-trust
    source_type = classify_evidence_source(evidence[0])
    low_trust_types = {"self_generated", "unclassified"}

    if source_type in low_trust_types:
        return (
            f"Velocity anomaly: concept '{concept.id}' has confidence "
            f"{concept.confidence:.2f} but only 1 evidence of type '{source_type}'. "
            f"Consider clamping to {PWT_VELOCITY_CLAMP}."
        )

    return None


def clamp_velocity_anomaly(concept: Concept) -> float:
    """If velocity anomaly detected, return clamped confidence, else original.

    Does NOT mutate concept — caller applies.
    """
    try:
        from app.core.config import PWT_VELOCITY_CLAMP
    except ImportError:
        PWT_VELOCITY_CLAMP = 0.6

    alert = check_velocity_anomaly(concept)
    if alert:
        return PWT_VELOCITY_CLAMP
    return concept.confidence


# =============================================================================
# Preference Utilities (§4b.3)
# =============================================================================


def apply_preference_floor(concept: Concept, computed_salience: float) -> float:
    """Apply salience floor for preference-type concepts.

    Preferences never drop below PREFERENCE_SALIENCE_FLOOR.
    """
    try:
        from app.core.config import PREFERENCE_SALIENCE_FLOOR
    except ImportError:
        PREFERENCE_SALIENCE_FLOOR = 0.4

    if concept.concept_type == "preference":
        return max(computed_salience, PREFERENCE_SALIENCE_FLOOR)
    return computed_salience


# Regex patterns for preference detection in session_learn
PREFERENCE_SIGNALS = re.compile(
    r"(?:I prefer|I always|my style|I like to|I want|I need|"
    r"I usually|I tend to|my preference|I favor|I opt for)",
    re.IGNORECASE,
)


def detect_preference(summary: str) -> bool:
    """Check if a concept summary indicates a user preference.

    Used in session_learn() to auto-set concept_type='preference'.
    """
    return bool(PREFERENCE_SIGNALS.search(summary))
