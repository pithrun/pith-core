"""LLM-Powered Contradiction Detection — Tier 2.

Memory Integrity Spec v1.2, §5.1.5, SL-B2, SL-E1:
Uses an LLM for semantic contradiction detection when Tier 1 (keyword + embedding)
returns ambiguous scores (0.50-0.80). More accurate than embedding similarity for:
  - Different surface forms of the same fact ("100 req/s" vs "20 requests per second")
  - Subtle contradictions ("uses JWT for sessions" vs "uses JWT for permanent access")
  - Negation detection ("is enabled" vs "is disabled")

Budget: ~200ms per comparison (fast model, short prompt).
Only invoked when embedding similarity is in the ambiguous range.

Feature-gated on LLM_CONTRADICTION_TIER2_ENABLED (default: False).
Rate-limited per session and per day (SL-B2).
Provider configurable (SL-E1): openai | anthropic | local.
"""

import logging
import time
from dataclasses import dataclass
from datetime import date

from app.config import (
    CONTRADICTION_LLM_CACHE_DAYS,
    CONTRADICTION_LLM_MODEL,
    CONTRADICTION_LLM_PROVIDER,
    CONTRADICTION_LLM_TIMEOUT_MS,
    FEATURE_FLAGS,
    MAX_TIER2_CHECKS_PER_DAY,
    MAX_TIER2_CHECKS_PER_SESSION,
    TIER2_AMBIGUOUS_HIGH,
    TIER2_AMBIGUOUS_LOW,
    TIER2_FALLBACK_ON_CAP,
)
from app.datetime_utils import _utc_now, _utc_now_iso

logger = logging.getLogger(__name__)


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class LLMContradictionResult:
    """Result of LLM-powered contradiction check."""

    score: float  # 0.0 (consistent) to 1.0 (contradictory)
    method: str = "llm"  # Detection method
    reason: str = ""  # LLM's explanation
    provider: str = ""  # Which LLM provider was used
    latency_ms: float = 0.0  # How long the call took
    from_cache: bool = False  # Whether this was a cached result
    capped: bool = False  # Whether rate limit was hit
    contradiction_type: str = ""  # Phase 3 v1.1: semantic|factual|temporal|unknown


# =============================================================================
# Rate limiting
# =============================================================================

# =============================================================================
# API Key Validation (PERF FIX: fast-fail on missing key)
# =============================================================================

_api_key_warned: bool = False


def _warn_no_api_key(provider: str) -> None:
    """Log a one-time warning when API key is missing."""
    global _api_key_warned
    if not _api_key_warned:
        logger.warning(
            "Tier 2 LLM contradiction check skipped: API key not set for provider '%s'. "
            "Set the appropriate environment variable to enable LLM-powered contradiction detection.",
            provider,
        )
        _api_key_warned = True


def _has_api_key(provider: str) -> bool:
    """Check if the API key is available for the configured provider.

    Provider-aware: anthropic → ANTHROPIC_API_KEY, openai → OPENAI_API_KEY,
    local → always True (no key needed).
    """
    import os

    if provider == "anthropic":
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    elif provider == "openai":
        return bool(os.environ.get("OPENAI_API_KEY"))
    elif provider == "local":
        return True  # Local endpoint doesn't need an API key
    return False  # Unknown provider — fail safe


# In-memory rate counters (reset on restart — acceptable for feature-flagged code)
_session_counts: dict[str, int] = {}  # session_id → count this session
_daily_count: int = 0
_daily_date: date | None = None


def _check_rate_limit(session_id: str = "") -> bool:
    """Check if we're within rate limits. Returns True if allowed."""
    global _daily_count, _daily_date

    today = date.today()
    if _daily_date != today:
        _daily_count = 0
        _daily_date = today

    if _daily_count >= MAX_TIER2_CHECKS_PER_DAY:
        logger.warning("Tier 2 daily cap reached (%d)", MAX_TIER2_CHECKS_PER_DAY)
        return False

    if session_id:
        session_count = _session_counts.get(session_id, 0)
        if session_count >= MAX_TIER2_CHECKS_PER_SESSION:
            logger.debug("Tier 2 session cap reached for %s", session_id)
            return False

    return True


def _increment_rate_counter(session_id: str = "") -> None:
    """Increment rate counters after a successful LLM call."""
    global _daily_count
    _daily_count += 1
    if session_id:
        _session_counts[session_id] = _session_counts.get(session_id, 0) + 1


def reset_rate_limits() -> None:
    """Reset all rate limits — for testing only."""
    global _daily_count, _daily_date, _session_counts
    _daily_count = 0
    _daily_date = None
    _session_counts = {}


# =============================================================================
# LLM Contradiction Cache
# =============================================================================

# Simple in-memory cache: (summary_a_hash, summary_b_hash) → LLMContradictionResult
_contradiction_cache: dict[tuple[str, str], tuple[LLMContradictionResult, float]] = {}


def _cache_key(summary_a: str, summary_b: str) -> tuple[str, str]:
    """Create a normalized cache key (order-independent)."""
    a = summary_a.strip().lower()
    b = summary_b.strip().lower()
    return (min(a, b), max(a, b))


def _get_cached_result(summary_a: str, summary_b: str) -> LLMContradictionResult | None:
    """Check cache for a previous LLM contradiction result.

    Two-tier lookup: in-memory dict (fast path) → SQLite tier2_cache (disk path).
    WS1: Disk cache survives server restarts, eliminating redundant API calls.
    """
    key = _cache_key(summary_a, summary_b)

    # Fast path: in-memory cache
    if key in _contradiction_cache:
        result, timestamp = _contradiction_cache[key]
        age_days = (time.time() - timestamp) / 86400
        if age_days <= CONTRADICTION_LLM_CACHE_DAYS:
            return LLMContradictionResult(
                score=result.score,
                method=result.method,
                reason=result.reason,
                provider=result.provider,
                latency_ms=0.0,
                from_cache=True,
                contradiction_type=getattr(result, "contradiction_type", ""),
            )
        else:
            del _contradiction_cache[key]

    # Disk path: SQLite tier2_cache table
    try:
        from app.storage import _db

        cache_key_str = f"{key[0]}||{key[1]}"
        with _db() as conn:
            row = conn.execute(
                "SELECT score, method, reason, provider, contradiction_type FROM tier2_cache "
                "WHERE cache_key = ? AND expires_at > ?",
                (cache_key_str, _utc_now_iso()),
            ).fetchone()
            if row:
                result = LLMContradictionResult(
                    score=row[0],
                    method=row[1],
                    reason=row[2],
                    provider=row[3],
                    from_cache=True,
                    latency_ms=0.0,
                    contradiction_type=row[4] or "",
                )
                # Warm in-memory cache from disk hit
                _contradiction_cache[key] = (result, time.time())
                return result
    except Exception:
        pass  # Disk cache failure is non-fatal

    return None


def _set_cached_result(summary_a: str, summary_b: str, result: LLMContradictionResult) -> None:
    """Cache an LLM contradiction result in memory + SQLite disk.

    WS1: Dual-write ensures disk persistence across server restarts.
    Disk write is best-effort — failure falls back to memory-only.
    """
    key = _cache_key(summary_a, summary_b)
    _contradiction_cache[key] = (result, time.time())

    # Persist to SQLite
    try:
        from datetime import timedelta

        from app.storage import _db

        cache_key_str = f"{key[0]}||{key[1]}"
        expires = (_utc_now() + timedelta(days=CONTRADICTION_LLM_CACHE_DAYS)).isoformat()
        with _db() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO tier2_cache "
                "(cache_key, score, method, reason, provider, contradiction_type, created_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    cache_key_str,
                    result.score,
                    result.method,
                    result.reason,
                    result.provider,
                    getattr(result, "contradiction_type", ""),
                    _utc_now_iso(),
                    expires,
                ),
            )
    except Exception:
        pass  # Disk cache failure is non-fatal


# =============================================================================
# LLM Provider Abstraction (SL-E1)
# =============================================================================

CONTRADICTION_PROMPT_TEMPLATE = """You are evaluating whether two knowledge claims contradict each other.

Claim A [{authority_label_a}]: {summary_a}
Claim B [{authority_label_b}]: {summary_b}
{context_line}
Consider:
1. Do they make incompatible factual assertions about the same topic?
2. Could both be true simultaneously?
3. Are they about the same subject but with different specific claims?
4. If one claim has significantly lower authority or is quarantined, bias toward CONTRADICTION.

Respond with ONLY one of:
- CONTRADICTION: [type: semantic|factual|temporal] [brief reason]
- CONSISTENT: [brief reason]
- UNRELATED: [brief reason]"""


def _build_prompt(
    summary_a: str,
    summary_b: str,
    context: str = "",
    authority_a: float = 0.5,
    maturity_a: str = "PROVISIONAL",
    authority_b: float = 0.5,
    maturity_b: str = "PROVISIONAL",
) -> str:
    """Build the contradiction detection prompt with authority/maturity metadata (A4)."""
    context_line = f"Context: {context}\n" if context else ""
    return CONTRADICTION_PROMPT_TEMPLATE.format(
        summary_a=summary_a,
        summary_b=summary_b,
        context_line=context_line,
        authority_label_a=f"authority: {authority_a:.1f}, {maturity_a}",
        authority_label_b=f"authority: {authority_b:.1f}, {maturity_b}",
    )


def _parse_llm_response(response: str) -> LLMContradictionResult:
    """Parse the LLM's response into a structured result.

    Phase 3 v1.1: Now parses contradiction type (semantic|factual|temporal)
    from the enhanced prompt format.
    """
    response = response.strip()
    if response.upper().startswith("CONTRADICTION"):
        # Extract contradiction type if present
        contradiction_type = "unknown"
        for ctype in ("semantic", "factual", "temporal"):
            if ctype in response.lower():
                contradiction_type = ctype
                break
        return LLMContradictionResult(
            score=0.95,
            method="llm",
            reason=response,
            contradiction_type=contradiction_type,
        )
    elif response.upper().startswith("CONSISTENT"):
        return LLMContradictionResult(score=0.05, method="llm", reason=response)
    elif response.upper().startswith("UNRELATED"):
        return LLMContradictionResult(score=0.30, method="llm", reason=response)
    else:
        # Couldn't parse — return ambiguous
        return LLMContradictionResult(
            score=0.50,
            method="llm",
            reason=f"Unparseable LLM response: {response[:100]}",
        )


async def _call_llm_provider(prompt: str) -> str:
    """Call the configured LLM provider.

    SL-E1: Provider is configurable via CONTRADICTION_LLM_PROVIDER.
    Fallback chain: configured provider → quarantine.
    """
    provider = CONTRADICTION_LLM_PROVIDER
    model = CONTRADICTION_LLM_MODEL

    if provider == "openai":
        return await _call_openai(prompt, model)
    elif provider == "anthropic":
        return await _call_anthropic(prompt, model)
    elif provider == "local":
        return await _call_local(prompt, model)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


async def _call_openai(prompt: str, model: str) -> str:
    """Call OpenAI API for contradiction detection."""
    try:
        import openai

        client = openai.AsyncOpenAI()
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.0,
            timeout=CONTRADICTION_LLM_TIMEOUT_MS / 1000,
        )
        return response.choices[0].message.content or ""
    except ImportError:
        raise RuntimeError("openai package not installed")
    except Exception as e:
        raise RuntimeError(f"OpenAI API call failed: {e}")


async def _call_anthropic(prompt: str, model: str) -> str:
    """Call Anthropic API for contradiction detection."""
    try:
        import os

        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        client = anthropic.AsyncAnthropic(
            api_key=api_key,
            timeout=CONTRADICTION_LLM_TIMEOUT_MS / 1000,
            max_retries=0,  # No retries — latency-sensitive Tier 2 path
        )
        response = await client.messages.create(
            model=model,
            max_tokens=50,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text if response.content else ""
    except ImportError:
        raise RuntimeError("anthropic package not installed")
    except Exception as e:
        raise RuntimeError(f"Anthropic API call failed: {e}")


async def _call_local(prompt: str, model: str) -> str:
    """Call a local LLM endpoint for contradiction detection.

    Expects a local server at http://localhost:8080/v1/chat/completions
    with OpenAI-compatible API.
    """
    try:
        import aiohttp

        async with (
            aiohttp.ClientSession() as session,
            session.post(
                "http://localhost:8080/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 50,
                    "temperature": 0.0,
                },
                timeout=aiohttp.ClientTimeout(total=CONTRADICTION_LLM_TIMEOUT_MS / 1000),
            ) as resp,
        ):
            data = await resp.json()
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"Local LLM call failed: {e}")


# =============================================================================
# Main API
# =============================================================================


async def detect_contradiction_llm(
    new_summary: str,
    existing_summary: str,
    context: str = "",
    session_id: str = "",
    authority_a: float = 0.5,
    maturity_a: str = "PROVISIONAL",
    authority_b: float = 0.5,
    maturity_b: str = "PROVISIONAL",
) -> LLMContradictionResult:
    """Use LLM to detect semantic contradiction between two concept summaries.

    Budget: ~500ms per comparison (Haiku, short prompt).
    Only called when Tier 1 returns a score in the ambiguous range (0.50-0.80),
    or when topic-match pre-filter (A1) escalates low-overlap same-topic pairs.

    Phase 3 v1.1: Includes authority/maturity metadata in prompt (A4) to resist
    poisoned concept manipulation.

    Args:
        new_summary: The new concept's summary text.
        existing_summary: The existing concept's summary text.
        context: Optional additional context for the comparison.
        session_id: Session ID for per-session rate limiting.
        authority_a: Authority score of concept A (0.0-1.0).
        maturity_a: Maturity status of concept A (PROVISIONAL/ESTABLISHED/QUARANTINED).
        authority_b: Authority score of concept B (0.0-1.0).
        maturity_b: Maturity status of concept B (PROVISIONAL/ESTABLISHED/QUARANTINED).

    Returns:
        LLMContradictionResult with score, method, reason, and contradiction_type.
    """
    if not FEATURE_FLAGS.get("LLM_CONTRADICTION_TIER2_ENABLED", False):
        return LLMContradictionResult(
            score=0.50,
            method="tier2_disabled",
            reason="LLM_CONTRADICTION_TIER2_ENABLED is False",
        )

    # Check cache first
    cached = _get_cached_result(new_summary, existing_summary)
    if cached:
        return cached

    # PERF FIX: Fast-fail if API key is not configured for the current provider.
    # Without this check, we spin up a ThreadPoolExecutor, create a new event loop,
    # cold-import the LLM SDK, init an HTTP client, then fail on auth — ~634ms
    # of wasted work. With this check: <1ms.
    provider = CONTRADICTION_LLM_PROVIDER
    if not _has_api_key(provider):
        _warn_no_api_key(provider)
        return LLMContradictionResult(
            score=0.50,
            method="no_api_key",
            reason=f"API key not set for provider '{provider}' — skipping Tier 2 LLM check",
        )

    # Check rate limits
    if not _check_rate_limit(session_id):
        return LLMContradictionResult(
            score=0.50,
            method="rate_limited",
            reason=f"Rate limit reached, fallback to {TIER2_FALLBACK_ON_CAP}",
            capped=True,
        )

    # Build prompt and call LLM (A4: includes authority/maturity metadata)
    prompt = _build_prompt(
        new_summary,
        existing_summary,
        context,
        authority_a=authority_a,
        maturity_a=maturity_a,
        authority_b=authority_b,
        maturity_b=maturity_b,
    )
    start_ms = time.time() * 1000

    try:
        response = await _call_llm_provider(prompt)
        latency_ms = time.time() * 1000 - start_ms

        result = _parse_llm_response(response)
        result.provider = CONTRADICTION_LLM_PROVIDER
        result.latency_ms = latency_ms

        # Increment rate counter
        _increment_rate_counter(session_id)

        # Cache the result
        _set_cached_result(new_summary, existing_summary, result)

        # WS2: Metric 2 — tier2_llm_latency_ms
        # WS2: Metric 3 — tier2_llm_cost_calls
        try:
            from app.metrics import metrics as _m23

            _m23.record("tier2_llm_latency_ms", latency_ms, {"provider": CONTRADICTION_LLM_PROVIDER})
            _m23.record("tier2_llm_cost_calls", 1, {"provider": CONTRADICTION_LLM_PROVIDER})
        except Exception:
            pass

        logger.info(
            "Tier 2 LLM contradiction check: score=%.2f, latency=%.0fms, reason=%s",
            result.score,
            latency_ms,
            result.reason[:100],
        )
        return result

    except Exception as e:
        latency_ms = time.time() * 1000 - start_ms
        logger.warning("Tier 2 LLM call failed (%.0fms): %s", latency_ms, e)
        # Fallback: return ambiguous score → quarantine
        return LLMContradictionResult(
            score=0.50,
            method="llm_error",
            reason=f"LLM call failed: {e}",
            latency_ms=latency_ms,
        )


def detect_contradiction_llm_sync(
    new_summary: str,
    existing_summary: str,
    context: str = "",
    session_id: str = "",
    authority_a: float = 0.5,
    maturity_a: str = "PROVISIONAL",
    authority_b: float = 0.5,
    maturity_b: str = "PROVISIONAL",
) -> LLMContradictionResult:
    """Synchronous wrapper for detect_contradiction_llm.

    For use in synchronous code paths like detect_write_contradiction.

    STABILITY-014 Fix 4: Simplified from ThreadPoolExecutor pattern.
    All callers (contradiction.py:1054, session.py:5216) are sync functions
    in FastAPI's anyio threadpool, so asyncio.run() is always correct.
    """
    import asyncio

    coro = detect_contradiction_llm(
        new_summary,
        existing_summary,
        context,
        session_id,
        authority_a=authority_a,
        maturity_a=maturity_a,
        authority_b=authority_b,
        maturity_b=maturity_b,
    )

    return asyncio.run(coro)


def is_tier2_candidate(tier1_score: float) -> bool:
    """Check if a Tier 1 score is in the ambiguous range that warrants Tier 2.

    Args:
        tier1_score: The contradiction score from Tier 1 (keyword + embedding).

    Returns:
        True if the score is in the ambiguous range (TIER2_AMBIGUOUS_LOW to TIER2_AMBIGUOUS_HIGH).
    """
    return TIER2_AMBIGUOUS_LOW <= tier1_score <= TIER2_AMBIGUOUS_HIGH
