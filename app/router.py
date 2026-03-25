"""
Cognitive Retrieval Router for Pith Pith Server

Implements the Cognitive Retrieval Router per TEMPORAL_RETRIEVAL_SPEC.md §15.
Provides fast, heuristic-based question classification and supplementary retrieval
dispatch based on question type and temporal context.

Key Features:
- Fast keyword heuristics (<1ms) with no LLM calls [H8]
- Temporal and causal analysis dispatch [S4.5]
- 100ms hard timeout on supplementary retrieval [H10]
- Comprehensive logging and debug support [M50, M52]
"""

import logging
import os
import re
import time
from datetime import datetime, timedelta
from typing import Any

from app import causal, temporal
from app.datetime_utils import _utc_now, _utc_now_iso
from app.storage import _get_connection

# Configure logging
logger = logging.getLogger(__name__)

# Feature flag: Enable/disable cognitive router [M36]
ENABLE_COGNITIVE_ROUTER = os.environ.get("ENABLE_COGNITIVE_ROUTER", "true").lower() == "true"

# Debug mode: Enable verbose logging [M50]
DEBUG_MODE = os.environ.get("COGNITIVE_ROUTER_DEBUG", "false").lower() == "true"

# Hard timeout for supplementary retrieval in milliseconds [H10]
SUPPLEMENTARY_TIMEOUT_MS = 100

# Minimum relevance threshold for supplementary retrieval [M43]
MIN_RELEVANCE_FOR_SUPPLEMENTARY = 0.4


def _log_classification(result: dict[str, Any], input_len: int, start_time: float) -> None:
    """F6 + A-H15: Structured classification telemetry."""
    elapsed_ms = (time.monotonic() - start_time) * 1000
    logger.info(
        f"S2.5_CLASSIFY: result={result.get('classification', 'unknown')}, "
        f"confidence={result.get('confidence', 0):.2f}, "
        f"source={result.get('input_source', 'unknown')}, "
        f"elapsed_ms={elapsed_ms:.1f}, input_len={input_len}"
    )


def classify_question(
    message: str, user_raw_message: str | None = None, force_classification: str | None = None
) -> dict[str, Any]:
    """
    Classify a question into semantic types using fast keyword heuristics.

    Implements S2.5 question classifier with regex patterns, no LLM calls,
    <1ms execution target [H8].

    Classification types:
    - temporal_activity: "what did we do", "yesterday", "recent"
    - temporal_state: "what did we believe", "at that point", "on [date]"
    - causal_backward: "why did", "what caused", "root cause"
    - causal_forward: "what happens if", "consequences", "impact of"
    - evolution: "how has X changed", "history of", "evolution"
    - compositional: "how are X and Y related", "connection between"
    - contradiction: "is that still true", "conflict", "contradicts"
    - counting: "how many", "count of", "total number"
    - general: everything else

    Args:
        message: The processed/normalized message text
        user_raw_message: Optional raw user message before processing [H13]
        force_classification: Override classification for testing [M41]

    Returns:
        Dict with keys:
        - classification: str, one of the types above
        - confidence: float, 0.0-1.0 confidence in classification
        - input_source: str, either "raw" or "processed"
    """
    _classify_start = time.monotonic()

    if force_classification:
        logger.debug(f"Classification forced to {force_classification}")
        result = {"classification": force_classification, "confidence": 1.0, "input_source": "forced"}
        _log_classification(result, 0, _classify_start)
        return result

    # Use raw message if available, fallback to processed [H13]
    text_to_classify = user_raw_message if user_raw_message else message
    input_source = "raw" if user_raw_message else "processed"

    # Default to general for very short messages [M44]
    if not text_to_classify or len(text_to_classify.split()) < 3:
        if DEBUG_MODE:
            logger.debug(f"Message too short, defaulting to general: '{text_to_classify}'")
        result = {"classification": "general", "confidence": 0.5, "input_source": input_source}
        _log_classification(result, len(text_to_classify or ""), _classify_start)
        return result

    # Normalize for pattern matching
    text_lower = text_to_classify.lower()

    # Temporal activity patterns: "what did we do", "yesterday", "recent"
    temporal_activity_patterns = [
        r"\b(what did|what did we|what have|what activities)\b",
        r"\b(yesterday|last\s+(week|month|year|night|day))\b",
        r"\b(recent|recently|a\s+few\s+days?\s+ago)\b",
        r"\b(\d+\s+(days?|weeks?|months?)\s+ago)\b",
        r"\b(this\s+(week|month|year))\b",
        # CLASSIFIER-001: 9 missing patterns from temporal audit (40% recall → ~70%)
        r"\b(earlier\s+today|earlier\s+this\s+(week|month))\b",
        r"\b(last\s+session|previous\s+session|prior\s+session)\b",
        r"\b(where\s+(did\s+)?we\s+le(ave|ft)\s+off)\b",
        r"\b(catch\s+me\s+up|bring\s+me\s+up\s+to\s+speed)\b",
        r"\b(since\s+(last|monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|april|may|june|july|august|september|october|november|december))\b",
        r"\b(up\s+to\s+speed|fill\s+me\s+in)\b",
        r"\b(last\s+few\s+(days|weeks|hours|sessions))\b",
        r"\b(what('s|\s+is)\s+(new|changed|happened|different))\b",
        r"\b(what\s+were\s+we\s+(discussing|talking|working))\b",
    ]

    # Negative patterns to reduce false positives [M23]
    # CLASSIFIER-001: Softened "planning to" — now only blocks when it's the
    # primary verb frame, not when embedded in temporal context like
    # "what were we planning to discuss last week?"
    temporal_activity_negative = [
        r"^(i\s+am\s+planning|we\s+should|i\s+will)\b",
        r"\b(general|always|usually)\b",
    ]

    if _matches_patterns(text_lower, temporal_activity_patterns, temporal_activity_negative):
        # CLASSIFIER-002: Cascade ordering fix — if message has an explicit date anchor,
        # prefer temporal_state over temporal_activity. "On March 5th what did we decide"
        # should route to temporal_state (point-in-time query), not temporal_activity.
        _explicit_date_re = re.compile(
            r"\b(on\s+(january|february|march|april|may|june|july|august|september|"
            r"october|november|december)\s+\d{1,2}(st|nd|rd|th)?|on\s+\d{1,2}[-/]\d{1,2})\b"
        )
        if _explicit_date_re.search(text_lower):
            return {"classification": "temporal_state", "confidence": 0.88, "input_source": input_source}
        return {"classification": "temporal_activity", "confidence": 0.85, "input_source": input_source}

    # Temporal state patterns: "what did we believe", "at that point", "on [date]"
    temporal_state_patterns = [
        r"\b(at\s+that\s+point|back\s+then|at\s+the\s+time)\b",
        r"\b(on\s+\d{1,2}[-/]\d{1,2}|on\s+\w+\s+\d{1,2})\b",
        r"\b(did\s+we\s+believe|was\s+our\s+position|our\s+understanding)\b",
        r"\b(before\s+\w+|after\s+\w+)\b",
    ]

    temporal_state_negative = [
        r"\b(will|future|planning)\b",
    ]

    if _matches_patterns(text_lower, temporal_state_patterns, temporal_state_negative):
        return {"classification": "temporal_state", "confidence": 0.80, "input_source": input_source}

    # Causal backward patterns: "why did", "what caused", "root cause"
    causal_backward_patterns = [
        r"\b(why\s+did|why\s+do|what\s+caused|root\s+cause)\b",
        r"\b(reason\s+for|because\s+of|due\s+to)\b",
        r"\b(what\s+led\s+to|what\s+triggered)\b",
    ]

    if _matches_patterns(text_lower, causal_backward_patterns):
        return {"classification": "causal_backward", "confidence": 0.90, "input_source": input_source}

    # Causal forward patterns: "what happens if", "consequences", "impact of"
    causal_forward_patterns = [
        r"\b(what\s+happens\s+if|what\s+if|what\s+would)\b",
        r"\b(consequences\s+of|impact\s+of|effects?\s+of)\b",
        r"\b(what\s+would\s+happen|lead\s+to|result\s+in)\b",
        r"\b(if\s+\w+\s+\w+)\b",  # Simple if-then pattern
    ]

    if _matches_patterns(text_lower, causal_forward_patterns):
        return {"classification": "causal_forward", "confidence": 0.88, "input_source": input_source}

    # Evolution patterns: "how has X changed", "history of", "evolution"
    evolution_patterns = [
        r"\b(how\s+has|how\s+did)\b.*\b(chang|evolv|develop|grow)\b",
        r"\b(history\s+of|evolution\s+of|development\s+of)\b",
        r"\b(over\s+time|through\s+time|as\s+time\s+goes)\b",
        r"\b(changed\s+from|evolved\s+from|transformed)\b",
    ]

    if _matches_patterns(text_lower, evolution_patterns):
        return {"classification": "evolution", "confidence": 0.82, "input_source": input_source}

    # Compositional patterns: "how are X and Y related", "connection between"
    compositional_patterns = [
        r"\b(how\s+are).*\b(related|connected|linked)\b",
        r"\b(connection\s+between|relationship\s+between)\b",
        r"\b(relate|connect|link).*\b(and|to)\b",
        r"\b(similar|comparison|compare)\b.*\b(and|to|with)\b",
    ]

    if _matches_patterns(text_lower, compositional_patterns):
        return {"classification": "compositional", "confidence": 0.78, "input_source": input_source}

    # Contradiction patterns: "is that still true", "conflict", "contradicts"
    contradiction_patterns = [
        r"\b(is\s+that\s+still\s+true|still\s+true|still\s+valid)\b",
        r"\b(conflict|contradicts?|contradictory)\b",
        r"\b(contradict|conflicts?|disagree)\b",
        r"\b(inconsistent|inconsistency)\b",
    ]

    if _matches_patterns(text_lower, contradiction_patterns):
        return {"classification": "contradiction", "confidence": 0.80, "input_source": input_source}

    # Counting patterns: "how many", "count of", "total number" (RETRIEVAL-026)
    counting_patterns = [
        r"\b(how\s+many)\b",
        r"\b(count\s+of|number\s+of)\b",
        r"\b(total\s+(concepts?|ideas?|things?|items?))\b",
        r"\b(how\s+much\s+do\s+(i|we)\s+know)\b",
    ]

    # Negative: "how many times" is a correction signal, not counting
    counting_negative = [
        r"\bhow\s+many\s+times\b",
    ]

    if _matches_patterns(text_lower, counting_patterns, counting_negative):
        return {"classification": "counting", "confidence": 0.85, "input_source": input_source}

    # Default classification
    if DEBUG_MODE:
        logger.debug(f"No pattern matched, defaulting to general for: '{text_to_classify}'")

    result = {"classification": "general", "confidence": 0.5, "input_source": input_source}
    _log_classification(result, len(text_to_classify), _classify_start)
    return result


def _matches_patterns(text: str, positive_patterns: list[str], negative_patterns: list[str] | None = None) -> bool:
    """
    Check if text matches positive patterns and avoids negative patterns.

    Args:
        text: Text to check (already lowercased)
        positive_patterns: List of regex patterns that should match
        negative_patterns: List of regex patterns that should NOT match [M23]

    Returns:
        True if any positive pattern matches and no negative patterns match
    """
    # Check positive patterns
    positive_match = any(re.search(pattern, text) for pattern in positive_patterns)

    if not positive_match:
        return False

    # Check negative patterns [M23]
    if negative_patterns:
        negative_match = any(re.search(pattern, text) for pattern in negative_patterns)
        if negative_match:
            return False

    return True


def infer_dates(message: str, user_timezone_offset: int = 0) -> dict[str, str | None]:
    """
    Extract date/time references from message text.

    Supports:
    - Relative patterns: "yesterday", "3 days ago", "last week", "this week"
    - ISO date parsing: "Feb 20", "2026-02-20"

    Args:
        message: Message text to parse for dates
        user_timezone_offset: UTC offset in hours for timezone handling

    Returns:
        Dict with keys:
        - since: ISO date string or None (start of date range)
        - until: ISO date string or None (end of date range)
    """
    text_lower = message.lower()
    today = _utc_now().date()

    # Handle relative date patterns
    if re.search(r"\byesterday\b", text_lower):
        since_date = today - timedelta(days=1)
        until_date = today
        return {"since": since_date.isoformat(), "until": until_date.isoformat()}

    if re.search(r"\btoday\b", text_lower):
        return {"since": today.isoformat(), "until": (today + timedelta(days=1)).isoformat()}

    # Last N days pattern: "3 days ago", "last 5 days"
    match = re.search(r"\b(?:last\s+)?(\d+)\s+days?\s+ago\b", text_lower)
    if match:
        num_days = int(match.group(1))
        since_date = today - timedelta(days=num_days)
        until_date = today
        return {"since": since_date.isoformat(), "until": until_date.isoformat()}

    # Last N weeks pattern
    match = re.search(r"\b(?:last\s+)?(\d+)\s+weeks?\s+ago\b", text_lower)
    if match:
        num_weeks = int(match.group(1))
        since_date = today - timedelta(weeks=num_weeks)
        until_date = today
        return {"since": since_date.isoformat(), "until": until_date.isoformat()}

    # "last week" pattern
    if re.search(r"\blast\s+week\b", text_lower):
        # Last 7 days
        since_date = today - timedelta(days=7)
        until_date = today
        return {"since": since_date.isoformat(), "until": until_date.isoformat()}

    # "this week" pattern
    if re.search(r"\bthis\s+week\b", text_lower):
        # Days since start of week (assuming Monday=0)
        days_since_monday = today.weekday()
        since_date = today - timedelta(days=days_since_monday)
        until_date = today
        return {"since": since_date.isoformat(), "until": until_date.isoformat()}

    # "last month" pattern
    if re.search(r"\blast\s+month\b", text_lower):
        since_date = today - timedelta(days=30)
        until_date = today
        return {"since": since_date.isoformat(), "until": until_date.isoformat()}

    # "recent" or "recently" pattern - default to last 7 days
    if re.search(r"\b(?:recent|recently)\b", text_lower):
        since_date = today - timedelta(days=7)
        until_date = today
        return {"since": since_date.isoformat(), "until": until_date.isoformat()}

    # Check for "since" prefix — changes semantics from point-in-time to range-to-now
    # Bug fix: "since feb 22" was returning until=feb 23 instead of until=now
    has_since_prefix = bool(re.search(r"\bsince\b", text_lower))

    # A-H8: Detect date ranges ("from X to Y"), log warning (full parsing is future work)
    has_range = bool(re.search(r"\b(from\s+\w+\s+to|between\s+\w+\s+and)\b", text_lower))
    if has_range:
        logger.info(f"infer_dates: Date range detected but not fully parsed: '{text_lower[:80]}'")

    # Try ISO date parsing: "2026-02-20"
    iso_match = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", text_lower)
    if iso_match:
        try:
            year, month, day = int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3))
            date_obj = datetime(year, month, day).date()
            if has_since_prefix:
                return {"since": date_obj.isoformat(), "until": (today + timedelta(days=1)).isoformat()}
            return {"since": date_obj.isoformat(), "until": (date_obj + timedelta(days=1)).isoformat()}
        except ValueError:
            pass

    # Try month/day pattern with optional year (A-H18): "Feb 20", "February 20", "Feb 20, 2025"
    # Bug fix: original used only %b which fails on full month names like "February"
    month_match = re.search(
        r"\b(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|"
        r"aug|august|sep|september|oct|october|nov|november|dec|december)"
        r"\s+(\d{1,2})(?:\s*,?\s*(\d{4}))?\b",
        text_lower,
    )
    if month_match:
        month_str = month_match.group(1)
        day = int(month_match.group(2))
        year = int(month_match.group(3)) if month_match.group(3) else today.year

        # Try both abbreviated and full month name formats
        date_obj = None
        for fmt in ["%b", "%B"]:
            try:
                parsed_month = datetime.strptime(month_str, fmt).month
                date_obj = datetime(year, parsed_month, day).date()
                break
            except ValueError:
                continue

        if date_obj:
            if has_since_prefix:
                # "since feb 22" → from that date to NOW (inclusive of today)
                return {"since": date_obj.isoformat(), "until": (today + timedelta(days=1)).isoformat()}
            else:
                # "on feb 22" or just "feb 22" → just that day
                return {"since": date_obj.isoformat(), "until": (date_obj + timedelta(days=1)).isoformat()}

    # No date found
    return {"since": None, "until": None}


def dispatch_supplementary(
    classification: str,
    dates: dict[str, str | None],
    best_concept_id: str | None,
    session_id: str,
    skip_supplementary: bool = False,
    epistemic_filter: str | None = None,
) -> list[dict[str, Any]]:
    """
    Dispatch to supplementary retrieval based on question classification.

    Implements S4.5 supplementary retrieval with 100ms hard timeout [H10].
    Calls temporal.py or causal.py functions based on classification type.

    Args:
        classification: Question classification from classify_question()
        dates: Date range from infer_dates()
        best_concept_id: Primary concept ID or None
        session_id: Current session ID for tracking
        skip_supplementary: Skip supplementary retrieval [M48]
        epistemic_filter: Optional epistemic network filter (§5.8.7, Phase 3).
            When set and ROUTER_EPISTEMIC_FILTER_ENABLED=True, supplementary
            results are filtered to concepts matching this epistemic class.

    Returns:
        List of concept dicts or empty list on timeout/skip
    """
    # Skip if flag is set [M48]
    if skip_supplementary:
        if DEBUG_MODE:
            logger.debug("Supplementary retrieval skipped via flag")
        return []

    # Skip supplementary if best_concept relevance < 0.4 [M43]
    if best_concept_id is None:
        if DEBUG_MODE:
            logger.debug("No best_concept_id, skipping supplementary retrieval")
        return []

    # Start timeout timer [H10]
    start_time = time.perf_counter()

    try:
        supplementary = []

        if classification == "temporal_activity":
            supplementary = _dispatch_temporal_activity(dates, best_concept_id, session_id, start_time)

        elif classification == "temporal_state":
            supplementary = _dispatch_temporal_state(dates, best_concept_id, session_id, start_time)

        elif classification == "causal_backward":
            supplementary = _dispatch_causal_backward(best_concept_id, session_id, start_time)

        elif classification == "causal_forward":
            supplementary = _dispatch_causal_forward(best_concept_id, session_id, start_time)

        elif classification == "evolution":
            supplementary = _dispatch_evolution(dates, best_concept_id, session_id, start_time)

        elif classification == "counting":
            supplementary = _dispatch_counting(best_concept_id, session_id, start_time)

        elif classification in ["compositional", "contradiction"]:
            supplementary = []

        else:  # general or unknown
            supplementary = []

        # Apply epistemic filter if specified (§5.8.7, feature-flagged)
        return _apply_epistemic_filter(supplementary, epistemic_filter)

    except Exception as e:
        logger.error(f"Error in supplementary dispatch: {e}")
        return []


def _apply_epistemic_filter(
    results: list[dict[str, Any]],
    epistemic_filter: str | None,
) -> list[dict[str, Any]]:
    """Apply epistemic network filter to supplementary results (§5.8.7).

    Feature-flagged via ROUTER_EPISTEMIC_FILTER_ENABLED. When OFF, returns
    results unchanged (no filtering overhead).

    Args:
        results: Supplementary retrieval results
        epistemic_filter: Epistemic network to filter by (e.g. "preference")

    Returns:
        Filtered results or original if flag is OFF or no filter specified.
    """
    if not epistemic_filter or not results:
        return results

    from app.config import FEATURE_FLAGS

    if not FEATURE_FLAGS.get("ROUTER_EPISTEMIC_FILTER_ENABLED", False):
        return results

    filtered = [r for r in results if r.get("epistemic_network", "assessment") == epistemic_filter]

    if DEBUG_MODE:
        logger.debug(
            "Epistemic filter '%s': %d → %d results",
            epistemic_filter,
            len(results),
            len(filtered),
        )

    # If filtering removes everything, return originals (fail-open)
    return filtered if filtered else results


def _check_timeout(start_time: float) -> bool:
    """
    Check if timeout has been exceeded.

    Args:
        start_time: Result from time.perf_counter() at start

    Returns:
        True if >= SUPPLEMENTARY_TIMEOUT_MS has elapsed
    """
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    return elapsed_ms >= SUPPLEMENTARY_TIMEOUT_MS


def _dispatch_temporal_activity(
    dates: dict[str, str | None], best_concept_id: str, session_id: str, start_time: float
) -> list[dict[str, Any]]:
    """Dispatch to temporal activity retrieval via pith_timeline."""
    if _check_timeout(start_time):
        return []

    try:
        since = dates.get("since")
        until = dates.get("until")

        result = temporal.pith_timeline(
            since=since,
            until=until,
        )
        if result.get("status") == "success":
            return result.get("concepts", [])

        return []
    except Exception as e:
        logger.error(f"Error in temporal_activity dispatch: {e}")
        return []


def _dispatch_temporal_state(
    dates: dict[str, str | None], best_concept_id: str, session_id: str, start_time: float
) -> list[dict[str, Any]]:
    """Dispatch to temporal state retrieval via pith_knowledge_at."""
    if _check_timeout(start_time):
        return []

    try:
        # Use the midpoint or 'since' as the point_in_time
        point_in_time = dates.get("since") or dates.get("until")
        if not point_in_time:
            return []

        result = temporal.pith_knowledge_at(
            point_in_time=point_in_time,
        )
        if result.get("status") == "success":
            return result.get("concepts", [])

        return []
    except Exception as e:
        logger.error(f"Error in temporal_state dispatch: {e}")
        return []


def _dispatch_causal_backward(best_concept_id: str, session_id: str, start_time: float) -> list[dict[str, Any]]:
    """Dispatch to causal backward (root cause) retrieval via pith_trace_cause."""
    if _check_timeout(start_time):
        return []

    try:
        result = causal.pith_trace_cause(
            concept_id=best_concept_id,
            direction="root_cause",
        )
        if result.get("success") and result.get("data"):
            # Convert nodes to concept-like dicts for merging
            nodes = result["data"].get("nodes", [])
            return [n for n in nodes if n.get("concept_id") != best_concept_id]
        return []
    except Exception as e:
        logger.error(f"Error in causal_backward dispatch: {e}")
        return []


def _dispatch_causal_forward(best_concept_id: str, session_id: str, start_time: float) -> list[dict[str, Any]]:
    """Dispatch to causal forward (consequences) retrieval via pith_trace_cause."""
    if _check_timeout(start_time):
        return []

    try:
        result = causal.pith_trace_cause(
            concept_id=best_concept_id,
            direction="consequences",
        )
        if result.get("success") and result.get("data"):
            nodes = result["data"].get("nodes", [])
            return [n for n in nodes if n.get("concept_id") != best_concept_id]
        return []
    except Exception as e:
        logger.error(f"Error in causal_forward dispatch: {e}")
        return []


def _dispatch_evolution(
    dates: dict[str, str | None], best_concept_id: str, session_id: str, start_time: float
) -> list[dict[str, Any]]:
    """Dispatch to evolution/history retrieval via pith_evolution_of."""
    if _check_timeout(start_time):
        return []

    try:
        result = temporal.pith_evolution_of(
            concept_id=best_concept_id,
        )
        if result.get("status") == "success":
            # Return versions as concept-like dicts for merging
            return result.get("versions", [])
        return []
    except Exception as e:
        logger.error(f"Error in evolution dispatch: {e}")
        return []


def _dispatch_counting(best_concept_id: str, session_id: str, start_time: float) -> list[dict[str, Any]]:
    """Dispatch counting query — returns synthetic result with aggregate count.

    Extracts the counting subject from the query context (via best_concept_id's
    knowledge_area), then runs SQL aggregate COUNT(*) with optional KA filter.
    Returns a single synthetic concept dict with the count in the summary.

    RETRIEVAL-026: Counting query classification and handler.
    """
    if _check_timeout(start_time):
        return []

    try:
        from app.storage import _db, load_concept

        # Get the best concept's KA to use as a filter hint
        best_concept = load_concept(best_concept_id) if best_concept_id else None
        target_ka = best_concept.knowledge_area if best_concept else None

        with _db() as conn:
            if target_ka:
                # Count concepts in the target knowledge area
                row = conn.execute(
                    "SELECT COUNT(*) FROM concepts WHERE status='active' AND knowledge_area=?",
                    (target_ka,),
                ).fetchone()
                ka_count = row[0] if row else 0

                # Also get total for context
                total_row = conn.execute(
                    "SELECT COUNT(*) FROM concepts WHERE status='active'"
                ).fetchone()
                total_count = total_row[0] if total_row else 0

                # Get type breakdown within the KA
                type_rows = conn.execute(
                    """SELECT concept_type, COUNT(*) as cnt
                       FROM concepts WHERE status='active' AND knowledge_area=?
                       GROUP BY concept_type ORDER BY cnt DESC LIMIT 5""",
                    (target_ka,),
                ).fetchall()
                type_breakdown = ", ".join(f"{r[0]}={r[1]}" for r in type_rows) if type_rows else "none"

                summary = (
                    f"COUNTING RESULT: {ka_count} active concepts in knowledge area '{target_ka}' "
                    f"(out of {total_count} total). Type breakdown: {type_breakdown}."
                )
            else:
                # No KA hint — return total count with KA breakdown
                total_row = conn.execute(
                    "SELECT COUNT(*) FROM concepts WHERE status='active'"
                ).fetchone()
                total_count = total_row[0] if total_row else 0

                ka_rows = conn.execute(
                    """SELECT knowledge_area, COUNT(*) as cnt
                       FROM concepts WHERE status='active'
                       GROUP BY knowledge_area ORDER BY cnt DESC LIMIT 8""",
                ).fetchall()
                ka_breakdown = ", ".join(
                    f"{r[0] or 'unknown'}={r[1]}" for r in ka_rows
                ) if ka_rows else "none"

                summary = (
                    f"COUNTING RESULT: {total_count} total active concepts. "
                    f"Top knowledge areas: {ka_breakdown}."
                )

        return [{
            "id": f"counting_result_{session_id}",
            "summary": summary,
            "confidence": 1.0,
            "relevance_score": 0.95,
            "knowledge_area": target_ka or "aggregate",
            "concept_type": "counting_result",
        }]

    except Exception as e:
        logger.error(f"Error in counting dispatch: {e}")
        return []


def log_classification(
    session_id: str,
    input_source: str,
    input_length: int,
    classification: str,
    confidence: float,
    was_overridden: bool,
    override_value: str | None = None,
) -> None:
    """
    Log classification to database for analysis.

    Writes to classification_log table [M52]. Records all classifications
    for quality monitoring and debugging.

    Args:
        session_id: Current session ID
        input_source: "raw", "processed", or "forced"
        input_length: Number of characters in input
        classification: The assigned classification
        confidence: Confidence score 0.0-1.0
        was_overridden: Whether classification was forced
        override_value: The forced value if was_overridden is True
    """
    try:
        conn = _get_connection()
        cursor = conn.cursor()

        timestamp = _utc_now_iso()

        cursor.execute(
            """
            INSERT INTO classification_log
            (session_id, timestamp, input_source, input_length, classification,
             confidence, was_overridden, override_value)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                session_id,
                timestamp,
                input_source,
                input_length,
                classification,
                confidence,
                was_overridden,
                override_value,
            ),
        )

        conn.commit()
        cursor.close()

        if DEBUG_MODE:
            logger.debug(
                f"Logged classification: {classification} (confidence={confidence:.2f}, source={input_source})"
            )

    except Exception as e:
        logger.error(f"Error logging classification: {e}")


def build_router_metadata(
    classification: str,
    confidence: float,
    supplementary_query: str | None = None,
    concepts_added: int = 0,
    concepts_trimmed: int = 0,
    error: str | None = None,
) -> dict[str, Any] | None:
    """
    Build router metadata dict for response.

    Per §15.6, constructs metadata dict capturing router state and decisions.
    Returns None when classification is general and no supplementary occurred [H14].

    Args:
        classification: The assigned classification
        confidence: Confidence score 0.0-1.0
        supplementary_query: Type of supplementary retrieval performed
        concepts_added: Number of concepts added via supplementary retrieval
        concepts_trimmed: Number of concepts trimmed by router
        error: Any error that occurred during routing

    Returns:
        Dict with router metadata or None [H14]
    """
    # Return None if general classification with no supplementary [H14]
    if classification == "general" and not supplementary_query:
        return None

    metadata = {
        "classification": classification,
        "confidence": confidence,
    }

    if supplementary_query:
        metadata["supplementary_query"] = supplementary_query

    if concepts_added > 0:
        metadata["concepts_added"] = concepts_added

    if concepts_trimmed > 0:
        metadata["concepts_trimmed"] = concepts_trimmed

    if error:
        metadata["error"] = error

    return metadata if metadata else None
