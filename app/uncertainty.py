"""Calibrated Uncertainty Signaling — qualifier generation for trust signals.

Generates trust qualifiers that explain WHY a concept might be less
trustworthy. Used during context assembly to annotate activated concepts.

Qualifiers:
  EVIDENCE_AGING: Last evidence > 30 days old
  LOW_CORROBORATION: Fewer than 2 evidence items
  CONTESTED: Currency status is CONTESTED
  STALE_RISK: High authority but low currency (drift risk)
  SINGLE_SOURCE: All evidence from same session
  HIGH_CONFIDENCE: Strong on all axes (positive qualifier)
"""

import logging
from datetime import datetime
from typing import Any

from app.datetime_utils import _ensure_aware, _utc_now

try:
    from app.authority import get_presentation_mode
except ImportError:
    # Fallback for standalone testing without full app import chain
    def get_presentation_mode(authority: float) -> str:
        if authority >= 0.80:
            return "CONSTRAINT"
        if authority >= 0.60:
            return "DIRECTIVE"
        if authority >= 0.40:
            return "CONTEXT"
        return "BACKGROUND"


logger = logging.getLogger(__name__)


# =============================================================================
# Qualifier Thresholds
# =============================================================================

EVIDENCE_AGING_DAYS = 30
LOW_CORROBORATION_THRESHOLD = 2
STALE_RISK_AUTHORITY_MIN = 0.60
STALE_RISK_CURRENCY_MAX = 0.50
HIGH_CONFIDENCE_AUTHORITY_MIN = 0.80
HIGH_CONFIDENCE_CURRENCY_MIN = 0.80
HIGH_CONFIDENCE_EVIDENCE_MIN = 3


# =============================================================================
# Qualifier Generation
# =============================================================================


def generate_qualifiers(
    concept_data: dict[str, Any],
    authority_score: float,
    currency_score: float,
    currency_status: str = "ACTIVE",
) -> list[str]:
    """Generate trust qualifiers for a concept.

    Args:
        concept_data: Full concept data dict (contains evidence, etc.)
        authority_score: Pre-computed authority score
        currency_score: Pre-computed currency score
        currency_status: Current currency status

    Returns:
        List of qualifier strings
    """
    qualifiers = []

    # Parse evidence
    evidence_list = concept_data.get("evidence", [])
    evidence_count = len(evidence_list)

    # Calculate days since last evidence
    days_since_last = _days_since_last_evidence(evidence_list, concept_data)

    # EVIDENCE_AGING: last evidence > 30 days old
    if days_since_last > EVIDENCE_AGING_DAYS:
        qualifiers.append("EVIDENCE_AGING")

    # LOW_CORROBORATION: fewer than 2 evidence items
    if evidence_count < LOW_CORROBORATION_THRESHOLD:
        qualifiers.append("LOW_CORROBORATION")

    # CONTESTED: currency status is CONTESTED
    if currency_status == "CONTESTED":
        qualifiers.append("CONTESTED")

    # RETRIEVAL-014 Layer 1d: SUPERSEDED qualifier.
    # Belt-and-suspenders: SUPERSEDED concepts should not reach the LLM
    # (Layer 1b + _governance_score hard filter), but if they do, the
    # qualifier signals reduced trust.
    if currency_status == "SUPERSEDED":
        qualifiers.append("SUPERSEDED")

    # STALE_RISK: high authority but low currency (drift risk)
    if authority_score >= STALE_RISK_AUTHORITY_MIN and currency_score < STALE_RISK_CURRENCY_MAX:
        qualifiers.append("STALE_RISK")

    # SINGLE_SOURCE: all evidence from same session (check session refs)
    if evidence_count >= 2 and _is_single_source(evidence_list):
        qualifiers.append("SINGLE_SOURCE")

    # HIGH_CONFIDENCE: strong on all axes (positive qualifier)
    if (
        authority_score >= HIGH_CONFIDENCE_AUTHORITY_MIN
        and currency_score >= HIGH_CONFIDENCE_CURRENCY_MIN
        and evidence_count >= HIGH_CONFIDENCE_EVIDENCE_MIN
    ):
        qualifiers.append("HIGH_CONFIDENCE")

    return qualifiers


def generate_trust_explanation(
    qualifiers: list[str],
    authority_score: float,
    currency_score: float,
) -> str:
    """Generate a human-readable trust explanation from qualifiers.

    Returns a concise string explaining the trust state.
    """
    if not qualifiers:
        mode = get_presentation_mode(authority_score)
        return f"{mode} concept (authority={authority_score:.2f}, currency={currency_score:.2f})"

    if "HIGH_CONFIDENCE" in qualifiers:
        return "High-confidence concept with strong evidence and recent validation"

    parts = []
    if "CONTESTED" in qualifiers:
        parts.append("under active dispute")
    if "STALE_RISK" in qualifiers:
        parts.append("high authority but may be outdated")
    if "EVIDENCE_AGING" in qualifiers:
        parts.append("evidence is aging (>30 days)")
    if "LOW_CORROBORATION" in qualifiers:
        parts.append("limited supporting evidence")
    if "SINGLE_SOURCE" in qualifiers:
        parts.append("all evidence from single source")

    return "; ".join(parts) if parts else "no specific concerns"


def build_trust_signal(
    concept_id: str,
    concept_data: dict[str, Any],
    authority_score: float,
    currency_score: float,
    currency_status: str = "ACTIVE",
    has_contradiction: bool = False,
) -> dict[str, Any]:
    """Build a complete trust signal for a concept.

    Returns a dict matching TrustSignal model fields.
    """
    qualifiers = generate_qualifiers(concept_data, authority_score, currency_score, currency_status)

    evidence_list = concept_data.get("evidence", [])
    days_since = _days_since_last_evidence(evidence_list, concept_data)

    explanation = generate_trust_explanation(qualifiers, authority_score, currency_score)
    mode = get_presentation_mode(authority_score)

    needs_reval = "STALE_RISK" in qualifiers or "EVIDENCE_AGING" in qualifiers or "CONTESTED" in qualifiers

    return {
        "concept_id": concept_id,
        "authority": authority_score,
        "currency": currency_score,
        "presentation_mode": mode,
        "qualifiers": qualifiers,
        "trust_explanation": explanation,
        "needs_revalidation": needs_reval,
        "has_contradiction": has_contradiction,
        "evidence_count": len(evidence_list),
        "days_since_last_evidence": int(days_since),
    }


# =============================================================================
# Helpers
# =============================================================================


def _days_since_last_evidence(evidence_list: list, concept_data: dict) -> float:
    """Calculate days since the most recent evidence."""
    newest_ts = None

    for ev in evidence_list:
        if isinstance(ev, dict):
            ts = ev.get("timestamp")
            if ts:
                try:
                    dt = _ensure_aware(datetime.fromisoformat(ts.replace("Z", "+00:00")))
                    if newest_ts is None or dt > newest_ts:
                        newest_ts = dt
                except (ValueError, TypeError):
                    pass

    # Fallback to concept updated_at
    if newest_ts is None:
        updated_at = concept_data.get("updated_at")
        if updated_at:
            try:
                newest_ts = _ensure_aware(datetime.fromisoformat(updated_at.replace("Z", "+00:00")))
            except (ValueError, TypeError):
                pass

    if newest_ts is None:
        return 365.0  # No timestamp → treat as very old

    delta = _utc_now() - newest_ts
    return max(0.0, delta.total_seconds() / 86400.0)


def _is_single_source(evidence_list: list) -> bool:
    """Check if all evidence comes from the same session/source."""
    sources = set()
    for ev in evidence_list:
        if isinstance(ev, dict):
            src = ev.get("source_reference") or ev.get("extraction_source", "unknown")
            sources.add(src)
        else:
            sources.add("string_evidence")

    return len(sources) <= 1
