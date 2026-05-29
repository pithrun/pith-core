"""EXP-003a: LLM Experiment Resolution Pipeline.

EXPERIMENT_RESOLUTION_SPEC v1.2 — LLM-powered synthesis for experiment resolution.
Evaluates experiment candidates via Haiku and produces real synthesis text.
Concept creation from synthesis results is handled by process_experiment_results() in experiments.py.
"""

import json
import logging
import os
import time

from app.core.models import Experiment, ExperimentCandidate, ExperimentResult

logger = logging.getLogger(__name__)

MODEL = "google/gemini-2.0-flash-001"
TIMEOUT_SECONDS = 10
MAX_TOKENS = 300
SUMMARY_TRUNCATE = 500  # [P-1] Max chars per concept summary in prompts
# ============================================================
# Valid concept types for response validation [CF-1]
# ============================================================

VALID_CONCEPT_TYPES = {
    "principle",
    "pattern",
    "heuristic",
    "observation",
    "method",
    "decision",
    "cognitive_strategy",
}

# [EXP-018] Circuit breaker: trips on first AuthenticationError, resets on process restart
_LLM_AUTH_FAILED: bool = False


def is_llm_auth_failed() -> bool:
    """[EXP-018] Return True if ANTHROPIC_API_KEY was rejected this run."""
    return _LLM_AUTH_FAILED


# ============================================================
# Prompt Templates [GAME-1: skepticism framing]
# ============================================================

PROMPT_HYPOTHESIS = """You are analyzing a pith's knowledge base. Be skeptical — most clusters \
are noise or duplicate phrasing. Only mark as meaningful if there is a genuine non-obvious insight.

These {n} observations cluster together with {mean_sim:.2f} mean similarity:

{numbered_summaries}

Is there a unifying principle that explains this cluster? If yes, synthesize \
it (30-200 chars). If the cluster is just noise or duplicate phrasing of \
the same thing, say so.

Respond in JSON only:
{{"meaningful": true/false, "synthesis": "...", "confidence": 0.0-1.0, \
"concept_type": "principle|pattern|heuristic", "knowledge_area": "...", \
"reason": "brief explanation"}}"""

PROMPT_SYNTHESIS = """You are analyzing a pith's knowledge base. Be skeptical — most cross-domain \
overlaps are superficial. Only mark as meaningful if the connection reveals a genuine reusable principle.

Two concepts from different domains share structural overlap:

Domain A ({ka_a}): {summary_a}
Domain B ({ka_b}): {summary_b}
Shared terms: {shared_terms}
Similarity: {similarity:.4f}

Is this cross-domain connection meaningful? If yes, synthesize a reusable \
principle (30-200 chars).

Respond in JSON only:
{{"meaningful": true/false, "synthesis": "...", "confidence": 0.0-1.0, \
"concept_type": "principle|pattern|heuristic", "knowledge_area": "...", \
"reason": "brief explanation"}}"""

PROMPT_ANALOGY = """You are analyzing a pith's knowledge base. Be skeptical — most structural \
similarities are coincidental. Only mark as meaningful if the analogy produces a transferable insight.

Two concepts from different domains have similar structure:

Concept A ({ka_a}): {summary_a}
Concept B ({ka_b}): {summary_b}
Structural score: {score:.4f}

Is this structural analogy meaningful? What does concept A teach us about \
concept B (or vice versa)? If meaningful, synthesize the insight (30-200 chars).

Respond in JSON only:
{{"meaningful": true/false, "synthesis": "...", "confidence": 0.0-1.0, \
"concept_type": "principle|pattern|heuristic", "knowledge_area": "...", \
"reason": "brief explanation"}}"""

PROMPT_COUNTERFACTUAL = """You are analyzing a pith's knowledge base. Be skeptical — most \
"what if" scenarios are trivial inversions. Only mark as meaningful if removing or changing \
the seed concept would produce a non-obvious cascade effect.

{dir_label} concept ({ka}): {seed_summary}

This concept has {downstream_count} {reach_label} connections in the knowledge graph \
(max chain depth: {max_depth}).

Connected concepts:
{connected_summaries}

Counterfactual question: What would change in Pith's knowledge if this \
{dir_label_lower} had gone differently or never happened? Would the downstream \
concepts still hold? Would Pith reach different conclusions?

If the counterfactual reveals a meaningful insight (a hidden dependency, a fragile \
assumption, or an alternative path), synthesize it (30-200 chars).

Respond in JSON only:
{{"meaningful": true/false, "synthesis": "...", "confidence": 0.0-1.0, \
"concept_type": "principle|pattern|heuristic", "knowledge_area": "...", \
"reason": "brief explanation"}}"""

# ============================================================
# Response Validation [CF-1]
# ============================================================


def _validate_llm_response(raw: dict) -> dict:
    """Validate and sanitize LLM JSON response."""
    result = {
        "meaningful": bool(raw.get("meaningful", False)),
        "synthesis": str(raw.get("synthesis", ""))[:500],
        "confidence": max(0.0, min(1.0, float(raw.get("confidence", 0.5)))),
        "concept_type": raw.get("concept_type", "observation"),
        "knowledge_area": str(raw.get("knowledge_area", "general"))[:50],
        "reason": str(raw.get("reason", ""))[:200],
    }
    if result["concept_type"] not in VALID_CONCEPT_TYPES:
        result["concept_type"] = "observation"
    return result


# ============================================================
# Prompt Building
# ============================================================


def _build_prompt(experiment: Experiment, top: ExperimentCandidate, concepts: list) -> str | None:
    """Build the appropriate prompt for the experiment type."""
    # Build concept lookup
    concept_map = {c.id: c for c in concepts} if concepts else {}

    if experiment.experiment_type == "hypothesis_generation":
        summaries = []
        for i, cid in enumerate(top.concept_ids[:10], 1):
            c = concept_map.get(cid)
            summary = c.summary[:SUMMARY_TRUNCATE] if c else f"[concept {cid[:8]}]"
            summaries.append(f"{i}. {summary}")
        return PROMPT_HYPOTHESIS.format(
            n=len(top.concept_ids),
            mean_sim=top.score_components.get("mean_similarity", top.score) if top.score_components else top.score,
            numbered_summaries="\n".join(summaries),
        )

    elif experiment.experiment_type == "cross_domain_synthesis":
        if len(top.concept_ids) < 2:
            return None
        ca = concept_map.get(top.concept_ids[0])
        cb = concept_map.get(top.concept_ids[1])
        if not ca or not cb:
            return None
        metadata = top.metadata or {}
        return PROMPT_SYNTHESIS.format(
            ka_a=getattr(ca, "knowledge_area", "general"),
            summary_a=ca.summary[:SUMMARY_TRUNCATE],
            ka_b=getattr(cb, "knowledge_area", "general"),
            summary_b=cb.summary[:SUMMARY_TRUNCATE],
            shared_terms=metadata.get("shared_terms", "unknown"),
            similarity=top.score,
        )
    elif experiment.experiment_type == "analogy_detection":
        if len(top.concept_ids) < 2:
            return None
        ca = concept_map.get(top.concept_ids[0])
        cb = concept_map.get(top.concept_ids[1])
        if not ca or not cb:
            return None
        return PROMPT_ANALOGY.format(
            ka_a=getattr(ca, "knowledge_area", "general"),
            summary_a=ca.summary[:SUMMARY_TRUNCATE],
            ka_b=getattr(cb, "knowledge_area", "general"),
            summary_b=cb.summary[:SUMMARY_TRUNCATE],
            score=top.score,
        )

    elif experiment.experiment_type == "counterfactual":
        if not top.concept_ids:
            return None
        seed = concept_map.get(top.concept_ids[0])
        if not seed:
            return None
        metadata = top.score_components or {}
        direction = metadata.get("direction") or "forward"
        is_forward = direction == "forward"
        # Build connected concept summaries (skip seed)
        connected = []
        for i, cid in enumerate(top.concept_ids[1:11], 1):
            c = concept_map.get(cid)
            summary = c.summary[:SUMMARY_TRUNCATE] if c else f"[concept {cid[:8]}]"
            connected.append(f"{i}. {summary}")
        return PROMPT_COUNTERFACTUAL.format(
            dir_label="Decision" if is_forward else "Outcome",
            dir_label_lower="decision" if is_forward else "outcome",
            ka=getattr(seed, "knowledge_area", "general"),
            seed_summary=seed.summary[:SUMMARY_TRUNCATE],
            downstream_count=metadata.get("downstream_count", metadata.get("upstream_count", len(top.concept_ids) - 1)),
            reach_label="downstream" if is_forward else "upstream",
            max_depth=metadata.get("max_depth_reached", 0),
            connected_summaries="\n".join(connected) if connected else "(no connected concepts available)",
        )

    else:
        logger.info("EXP-003a: No prompt template for type %s", experiment.experiment_type)
        return None


# ============================================================
# LLM API Call (mirrors contradiction_llm.py pattern)
# ============================================================


async def _call_anthropic(prompt: str) -> str:
    """Call LLM via OpenRouter (COST-001: switched from Anthropic direct billing). Returns raw text response."""
    global _LLM_AUTH_FAILED
    if _LLM_AUTH_FAILED:
        raise RuntimeError("EXP-018: LLM disabled — OPENROUTER_API_KEY invalid this run")

    try:
        from openai import AsyncOpenAI as _AsyncOAI
    except ImportError:
        raise RuntimeError("openai package not installed")

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    client = _AsyncOAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=TIMEOUT_SECONDS,
        max_retries=0,
    )
    try:
        response = await client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        err_str = str(e).lower()
        if "401" in err_str or "authentication" in err_str or "invalid api key" in err_str:
            _LLM_AUTH_FAILED = True
            logger.error("EXP-018: LLM disabled — API key rejected. Error: %s", e)
            raise
        if "403" in err_str or "permission" in err_str:
            _LLM_AUTH_FAILED = True
            logger.error("EXP-018: LLM disabled — API key lacks permissions. Error: %s", e)
            raise
        if "credit" in err_str or "billing" in err_str:
            _LLM_AUTH_FAILED = True
            logger.error("EXP-018: LLM disabled — API credits depleted. Error: %s", e)
        raise
    return response.choices[0].message.content or ""


# ============================================================
# Main Resolution Function (EXP-003a)
# ============================================================


async def resolve_experiment(
    experiment: Experiment,
    concepts: list | None = None,
) -> ExperimentResult | None:
    """Use LLM to evaluate experiment's top candidate and synthesize knowledge.

    Returns ExperimentResult with real synthesis, or None if resolution fails.
    Concept creation from these results is handled by the caller (process_experiment_results in experiments.py).

    Args:
        experiment: Experiment with candidates to evaluate.
        concepts: Optional list of Concept objects for summary lookup.
                  If None, summaries are not included in prompts.
    """
    from app.core.config import FEATURE_FLAGS as app_config

    if not experiment.candidates:
        return None

    # Pick top candidate by score
    top = max(experiment.candidates, key=lambda c: c.score)

    # Build prompt
    prompt = _build_prompt(experiment, top, concepts or [])
    if not prompt:
        logger.info("EXP-003a: No prompt for %s %s", experiment.experiment_type, experiment.id[:8])
        return None

    # Verbose logging [O-1]
    verbose = app_config.get("LLM_EXPERIMENT_VERBOSE_LOG", False)
    if verbose:
        logger.info("EXP-003a PROMPT [%s]: %s", experiment.id[:8], prompt[:500])

    t0 = time.monotonic()
    try:
        raw_text = await _call_anthropic(prompt)
    except Exception as e:
        logger.warning("EXP-003a: API call failed for %s: %s", experiment.id[:8], e)
        raise  # Caller handles revert to reasoning

    elapsed_ms = int((time.monotonic() - t0) * 1000)

    if verbose:
        logger.info("EXP-003a RAW [%s] (%dms): %s", experiment.id[:8], elapsed_ms, raw_text[:500])
    # Parse JSON response
    try:
        # Handle potential markdown code fences
        text = raw_text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("EXP-003a: JSON parse failed for %s: %s (raw: %s)", experiment.id[:8], e, raw_text[:200])
        raise RuntimeError(f"JSON parse failed: {e}")

    validated = _validate_llm_response(parsed)

    # Build result
    if validated["meaningful"]:
        result = ExperimentResult(
            synthesis=validated["synthesis"],
            confidence=round(validated["confidence"] * 0.85, 3),
            concepts_produced=[
                {
                    "summary": validated["synthesis"],
                    "confidence": validated["confidence"] * 0.85,
                    "knowledge_area": validated["knowledge_area"],
                    "concept_type": validated["concept_type"],
                    # [EXP-020] Evidence removed from spec — process_experiment_results
                    # is the authoritative evidence writer. Previously caused duplication.
                }
            ],
            cko_produced=None,
            reasoning_trace=(
                f"LLM resolution: model={MODEL}, type={experiment.experiment_type}, "
                f"candidate_score={top.score:.3f}, elapsed={elapsed_ms}ms, "
                f"reason={validated['reason'][:100]}"
            ),
        )
    else:
        result = ExperimentResult(
            synthesis=f"LLM: not meaningful — {validated.get('reason', 'no reason')}",
            confidence=0.0,
            concepts_produced=[],
            cko_produced=None,
            reasoning_trace=(
                f"LLM: not meaningful, model={MODEL}, "
                f"type={experiment.experiment_type}, elapsed={elapsed_ms}ms, "
                f"reason={validated['reason'][:100]}"
            ),
        )

    logger.info(
        "EXP-003a: %s %s → %s (conf=%.3f, %dms)",
        experiment.experiment_type,
        experiment.id[:8],
        "meaningful" if validated["meaningful"] else "not_meaningful",
        result.confidence,
        elapsed_ms,
    )
    return result


# ============================================================
# Health Check [SF-1, CPLX-1]
# ============================================================


def log_health_check(not_meaningful_count: int, total_count: int) -> None:
    """Warn if most resolutions are not meaningful. Warning-only, no pause."""
    if total_count >= 3 and not_meaningful_count / total_count > 0.8:
        logger.warning(
            "SF-1: >80%% not_meaningful (%d/%d) — review experiment quality",
            not_meaningful_count,
            total_count,
        )


# ============================================================
# Availability Check
# ============================================================


def check_llm_available() -> bool:
    """Return True if LLM experiment resolution is available."""
    from app.core.config import FEATURE_FLAGS as app_config

    if not app_config.get("LLM_EXPERIMENT_RESOLUTION_ENABLED", False):
        return False
    if not os.environ.get("OPENROUTER_API_KEY"):
        return False
    try:
        from openai import AsyncOpenAI  # noqa: F401

        return True
    except ImportError:
        return False
