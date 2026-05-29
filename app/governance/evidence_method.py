"""Evidence Method Anti-Spoofing — derived from source provenance, never LLM input.

Memory Integrity Spec v1.2, §5.2.3 [resolves C7]:
LLM can claim evidence_method="code_verified" for unverified assertions.
Solution: Never trust evidence_method from LLM. Auto-classify from provenance.

Promotion path: evidence_method can only be upgraded from 'llm_assertion' to
'code_verified' or 'search_verified' through explicit verification actions.
The LLM cannot self-promote.
"""

import logging

logger = logging.getLogger(__name__)

# Source type → allowed evidence_method (immutable derivation)
EVIDENCE_METHOD_DERIVATION = {
    "auto_learn":       "llm_assertion",      # Always — LLM auto-extracted
    "session_learn":    "llm_assertion",      # Always — LLM extracted from conversation
    "propose":          "llm_assertion",      # Default — can be overridden by verify
    "correction":       "llm_assertion",      # Corrections are LLM claims until verified
    "import":           "llm_assertion",      # Bulk imports are unverified
    "conversation":     "llm_assertion",      # Extracted from conversation context
    "user_stated":      "user_stated",        # User explicitly said it
    "code_verified":    "code_verified",      # ONLY set by verification workflow
    "search_verified":  "search_verified",    # ONLY set by search verification
}

# Evidence methods that can ONLY be set by system, never accepted from input
PRIVILEGED_METHODS = {"code_verified", "search_verified"}

# Valid evidence methods (for validation)
VALID_EVIDENCE_METHODS = set(EVIDENCE_METHOD_DERIVATION.values())


def derive_evidence_method(source_type: str) -> str:
    """Derive evidence method from source type. LLM input is IGNORED.

    Args:
        source_type: The provenance of the concept (e.g., "session_learn", "propose").

    Returns:
        The derived evidence_method string.
    """
    return EVIDENCE_METHOD_DERIVATION.get(source_type, "llm_assertion")


def sanitize_evidence(evidence_list: list, source_type: str) -> list:
    """Strip any spoofed evidence_method from evidence entries.

    Replaces any evidence_method in the list with the derived value
    based on source_type. This prevents LLM from claiming verification
    it hasn't performed.

    Feature-gated by EVIDENCE_ANTISPOOFING_ENABLED. When off, returns
    evidence_list unchanged.

    Args:
        evidence_list: List of evidence dicts from concept data.
        source_type: The provenance of this write operation.

    Returns:
        Sanitized evidence list with corrected evidence_method values.
    """
    from app.core.config import FEATURE_FLAGS
    if not FEATURE_FLAGS.get("EVIDENCE_ANTISPOOFING_ENABLED", False):
        return evidence_list

    derived_method = derive_evidence_method(source_type)

    for entry in evidence_list:
        if not isinstance(entry, dict):
            continue

        claimed_method = entry.get("evidence_method")
        if claimed_method and claimed_method in PRIVILEGED_METHODS:
            # LLM tried to claim a privileged method — override it
            logger.warning(
                f"Evidence anti-spoofing: stripped claimed '{claimed_method}' "
                f"→ '{derived_method}' (source_type={source_type})"
            )
            entry["evidence_method"] = derived_method
        elif claimed_method and claimed_method not in VALID_EVIDENCE_METHODS:
            # Unknown method — default to derived
            entry["evidence_method"] = derived_method
        elif not claimed_method:
            # No method specified — set derived
            entry["evidence_method"] = derived_method

    return evidence_list
