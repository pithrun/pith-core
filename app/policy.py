"""Content policy validator — shared authority-claim detection and overlap utilities.

Retrieval Defense Spec v1.2, Fixes 4/7/8:
  - check_content_policy(): Regex-based detection of authority-claiming patterns
  - compute_summary_overlap(): Word-level overlap for rewrite detection
"""

import logging
import re

logger = logging.getLogger(__name__)

AUTHORITY_CLAIM_PATTERN = re.compile(
    r"(?i)(system\s*override|all\s*constraints?\s*suspended|"
    r"admin\s*access|emergency\s*protocol|override\s*safety|"
    r"ignore\s*(all\s*)?(previous\s*)?instructions|"
    r"you\s*must\s*(now\s*)?obey|sudo\s*mode|"
    r"developer\s*override|root\s*access|"
    r"bypass\s*(all\s*)?(security|safety|constraints?))"
)


def check_content_policy(text: str) -> bool:
    """Returns True if text contains authority-claiming patterns.

    Used by Fix 4 (W6), Fix 7 (W8), and Fix 8 (W9) to detect
    adversarial content in concept summaries and evidence.
    """
    if not text:
        return False
    return bool(AUTHORITY_CLAIM_PATTERN.search(text))


SUMMARY_OVERLAP_THRESHOLD = 0.5  # Below this = substantial rewrite


def compute_summary_overlap(old_summary: str, new_summary: str) -> float:
    """Word-level overlap ratio between old and new summary.

    Returns fraction of old_summary words present in new_summary.
    Used by Fix 8 (W9) to detect substantial rewrites in evolution.
    """
    if not old_summary:
        return 1.0  # No old summary = not a rewrite
    old_words = set(old_summary.lower().split())
    new_words = set(new_summary.lower().split())
    if not old_words:
        return 1.0
    return len(old_words & new_words) / len(old_words)
