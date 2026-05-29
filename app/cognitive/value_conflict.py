"""
KU-VALUE-CONFLICT-001: Deterministic value-conflict detection.

Detects when two concept summaries describe the same entity+attribute
but with different values. Zero LLM calls, <1ms.

Examples:
  "user has 3 bikes" vs "user now has 4 bikes" -> CONFLICT (bikes: 3->4)
  "user lives in Boston" vs "user moved to Austin" -> CONFLICT (location: Boston->Austin)
  "user has 3 bikes" vs "user has a dog" -> NO CONFLICT (different attribute)
"""

import re
from typing import Optional

# Numeric value pattern: "has/owns/has N X" or "N X"
_NUMERIC_CLAIM = re.compile(
    r'\b(?:has|have|had|owns?|got|bought|earned|leads?|manages?|teaches?)\s+'
    r'(\d+)\s+'
    r'([a-z]+)\b',
    re.IGNORECASE,
)

# Location/state patterns: "lives in X", "moved to X", "works at X"
# GAUNTLET-F1 FIX: re.IGNORECASE — _detect_contradiction passes lowercased text.
# Capture group changed from [A-Z] to [a-z] since input is lowered.
_LOCATION_CLAIM = re.compile(
    r'\b(?:lives?\s+in|moved?\s+to|works?\s+(?:at|for)|located?\s+in|'
    r'based\s+in|relocated?\s+to|transferred?\s+to)\s+'
    r'([a-z][a-zA-Z\s]{1,30})',
    re.IGNORECASE,
)

# Name/identity patterns: "named X", "called X", "is X"
# GAUNTLET-F1 FIX: re.IGNORECASE — same reason as _LOCATION_CLAIM.
_NAME_CLAIM = re.compile(
    r'\b(?:named|called|nicknamed|renamed?\s+to)\s+'
    r'([a-z][a-zA-Z\s]{1,30})',
    re.IGNORECASE,
)


def _extract_numeric_claims(text: str) -> list[tuple[str, str]]:
    """Extract (value, attribute) pairs from text."""
    return [(m.group(1), m.group(2).lower().rstrip('s')) for m in _NUMERIC_CLAIM.finditer(text)]


def _extract_location_claims(text: str) -> list[str]:
    """Extract location values from text."""
    return [m.group(1).strip().rstrip('.,:;') for m in _LOCATION_CLAIM.finditer(text)]


def _extract_name_claims(text: str) -> list[str]:
    """Extract name/identity values from text."""
    return [m.group(1).strip().rstrip('.,:;') for m in _NAME_CLAIM.finditer(text)]


def detect_value_conflict(old_text: str, new_text: str) -> Optional[str]:
    """Detect if two summaries describe same entity with different value.

    Returns a reason string if conflict detected, None otherwise.
    Conservative: prefers false negatives over false positives.
    """
    if not old_text or not new_text:
        return None

    # --- Numeric value conflicts ---
    old_nums = _extract_numeric_claims(old_text)
    new_nums = _extract_numeric_claims(new_text)

    for new_val, new_attr in new_nums:
        for old_val, old_attr in old_nums:
            # Same attribute (singularized), different value
            if old_attr == new_attr and old_val != new_val:
                return f"numeric: {new_attr} changed {old_val}\u2192{new_val}"

    # --- Location/state conflicts ---
    old_locs = _extract_location_claims(old_text)
    new_locs = _extract_location_claims(new_text)

    if old_locs and new_locs:
        for old_loc in old_locs:
            for new_loc in new_locs:
                if old_loc.lower() != new_loc.lower():
                    return f"location: changed '{old_loc}'\u2192'{new_loc}'"

    # --- Name/identity conflicts ---
    old_names = _extract_name_claims(old_text)
    new_names = _extract_name_claims(new_text)

    if old_names and new_names:
        for old_name in old_names:
            for new_name in new_names:
                if old_name.lower() != new_name.lower():
                    return f"name: changed '{old_name}'\u2192'{new_name}'"

    return None
