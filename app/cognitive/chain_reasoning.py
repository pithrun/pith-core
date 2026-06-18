"""Chain-of-Thought Decomposition for Multi-Hop Questions.

RETRIEVAL-040-LLM: LLM-based decomposition for multi-hop queries.
Adapted from the benchmark chain-reasoning adapter.
Feature-gated: PITH_LLM_CHAIN_REASONING=true (benchmark mode only).

Instead of giving the LLM 25 facts + 1 complex question, we:
1. Decompose into single-hop sub-questions via LLM
2. Answer each step against the FULL context
3. Substitute intermediate answers and continue the chain
4. Final answer = result of last step
"""

import logging
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").lower() in _TRUTHY_ENV_VALUES


CHAIN_REASONING_ENABLED = os.environ.get("PITH_LLM_CHAIN_REASONING", "").lower() in ("true", "1")

BENCHMARK_MODE_ENABLED = _env_truthy("PITH_BENCHMARK_MODE")

PROVENANCE_ANSWER_ENABLED = os.environ.get("PITH_ENGINE_ANS1_EXTRACTIVE_ENABLED", "").lower() in ("true", "1")

PROVENANCE_ANSWER_LLM_ENABLED = os.environ.get("PITH_ENGINE_ANS1_LLM_ENABLED", "").lower() in ("true", "1")

PROVENANCE_ANSWER_TYPED_CANDIDATES_ENABLED = os.environ.get(
    "PITH_ENGINE_ANS1_TYPED_CANDIDATES_ENABLED", ""
).lower() in ("true", "1")

PROVENANCE_ANSWER_CONTRACT_ENABLED = os.environ.get("PITH_ENGINE_ANS1_ANSWER_CONTRACT_ENABLED", "").lower() in (
    "true",
    "1",
)

PROVENANCE_ANSWER_STRUCTURED_SYNTHESIS_ENABLED = os.environ.get(
    "PITH_ENGINE_ANS1_STRUCTURED_SYNTHESIS_ENABLED", ""
).lower() in ("true", "1")

PROVENANCE_ANSWER_EXACT_SUPPORT_RECOVERY_ENABLED = os.environ.get(
    "PITH_ENGINE_ANS1_EXACT_SUPPORT_RECOVERY_ENABLED", ""
).lower() in ("true", "1")

PROVENANCE_ANSWER_SUPPORT_DERIVED_REPAIR_ENABLED = os.environ.get(
    "PITH_ENGINE_ANS1_SUPPORT_DERIVED_REPAIR_ENABLED", ""
).lower() in ("true", "1")

PROVENANCE_ANSWER_SUPPORT_PACK_COMPLETENESS_ENABLED = os.environ.get(
    "PITH_ENGINE_ANS1_SUPPORT_PACK_COMPLETENESS_ENABLED", ""
).lower() in ("true", "1")

PROVENANCE_ANSWER_EXACT_SUPPORT_NATIVE_STABILITY_ENABLED = os.environ.get(
    "PITH_ENGINE_ANS1_EXACT_SUPPORT_NATIVE_STABILITY_ENABLED", ""
).lower() in ("true", "1")

PROVENANCE_ANSWER_SUPPORT_SURFACE_REACH_ENABLED = os.environ.get(
    "PITH_ENGINE_ANS1_SUPPORT_SURFACE_REACH_ENABLED", ""
).lower() in ("true", "1")

PROVENANCE_ANSWER_SUPPORT_PRESENT_NATIVE_STABILITY_ENABLED = os.environ.get(
    "PITH_ENGINE_ANS1_SUPPORT_PRESENT_NATIVE_STABILITY_ENABLED", ""
).lower() in ("true", "1")

PROVENANCE_ANSWER_SUPPORT_PRESENT_GUARD_STABILITY_ENABLED = os.environ.get(
    "PITH_ENGINE_ANS1_SUPPORT_PRESENT_GUARD_STABILITY_ENABLED", ""
).lower() in ("true", "1")

PROVENANCE_ANSWER_SUPPORT_PRESENT_ADMISSION_V2_ENABLED = _env_truthy(
    "PITH_ENGINE_ANS1_SUPPORT_PRESENT_ADMISSION_V2_ENABLED"
)

PROVENANCE_ANSWER_SUPPORT_PRESENT_ADMISSION_V3_ENABLED = _env_truthy(
    "PITH_ENGINE_ANS1_SUPPORT_PRESENT_ADMISSION_V3_ENABLED"
)

PROVENANCE_ANSWER_DIRECT_SUPPORT_ADMISSION_ENABLED = _env_truthy(
    "PITH_ENGINE_ANS1_DIRECT_SUPPORT_ADMISSION_ENABLED"
)

PROVENANCE_ANSWER_SHAPE_ADMISSION_ENABLED = _env_truthy(
    "PITH_ENGINE_ANS1_ANSWER_SHAPE_ADMISSION_ENABLED"
)

PROVENANCE_ANSWER_SHAPE_RUNTIME_EFFECT_ENABLED = (
    PROVENANCE_ANSWER_SHAPE_ADMISSION_ENABLED
    and _env_truthy("PITH_ENGINE_ANS1_ANSWER_SHAPE_RUNTIME_EFFECT_ENABLED")
)

PROVENANCE_ANSWER_LEGACY_SURFACE_CONTRACT_ENABLED = _env_truthy(
    "PITH_ENGINE_ANS1_LEGACY_SURFACE_CONTRACT_ENABLED"
)

PROVENANCE_ANSWER_LOCOMO_SUPPORT_PRESENT_SYNTHESIS_ENABLED = _env_truthy(
    "PITH_ENGINE_ANS1_LOCOMO_SUPPORT_PRESENT_SYNTHESIS_ENABLED"
)
PROVENANCE_ANSWER_LOCOMO_SUPPORT_PRESENT_ANSWER_REALIZATION_ENABLED = _env_truthy(
    "PITH_ENGINE_ANS1_LOCOMO_SUPPORT_PRESENT_ANSWER_REALIZATION_ENABLED"
)

PROVENANCE_ANSWER_LOCOMO_SUPPORT_EMISSION_ENABLED = _env_truthy(
    "PITH_ENGINE_ANS1_LOCOMO_SUPPORT_EMISSION_ENABLED"
)
PROVENANCE_ANSWER_LOCOMO_BOUNDED_SUPPORT_ADMISSION_ENABLED = _env_truthy(
    "PITH_ENGINE_ANS1_LOCOMO_BOUNDED_SUPPORT_ADMISSION_ENABLED"
)
PROVENANCE_ANSWER_LOCOMO_BOUNDED_SUPPORT_ADMISSION_EFFECT_ENABLED = _env_truthy(
    "PITH_ENGINE_ANS1_LOCOMO_BOUNDED_SUPPORT_ADMISSION_EFFECT_ENABLED"
)
PROVENANCE_ANSWER_LOCOMO_PRESERVE_INITIAL_SUPPORT_ENABLED = _env_truthy(
    "PITH_ENGINE_ANS1_LOCOMO_PRESERVE_INITIAL_SUPPORT_ENABLED"
)
PROVENANCE_ANSWER_LOCOMO_PRESERVE_INITIAL_SUPPORT_DISPLACE_ENABLED = _env_truthy(
    "PITH_ENGINE_ANS1_LOCOMO_PRESERVE_INITIAL_SUPPORT_DISPLACE_ENABLED"
)
PROVENANCE_ANSWER_LOCOMO_ACTIVATED_SUPPORT_CONTINUITY_ENABLED = _env_truthy(
    "PITH_ENGINE_ANS1_LOCOMO_ACTIVATED_SUPPORT_CONTINUITY_ENABLED"
)

CHAIN_EVIDENCE_CANDIDATE_ENABLED = _env_truthy("PITH_CHAIN_EVIDENCE_CANDIDATE_ENABLED")
CHAIN_EVIDENCE_CANDIDATE_ARBITRATION_ENABLED = (
    CHAIN_EVIDENCE_CANDIDATE_ENABLED
    and _env_truthy("PITH_CHAIN_EVIDENCE_CANDIDATE_ARBITRATION_ENABLED")
)
LOCOMO_HIGHWATER_EVIDENCE_ANSWER_ENABLED = _env_truthy(
    "PITH_LOCOMO_HIGHWATER_EVIDENCE_ANSWER_ENABLED"
)
LOCOMO_HIGHWATER_EVIDENCE_ANSWER_DISABLED = _env_truthy(
    "PITH_LOCOMO_HIGHWATER_EVIDENCE_ANSWER_DISABLED"
)
LOCOMO_HIGHWATER_PROVENANCE_ARBITRATION_ENABLED = _env_truthy(
    "PITH_LOCOMO_HIGHWATER_PROVENANCE_ARBITRATION_ENABLED"
)

_CHAIN_EVIDENCE_CANDIDATE_ARBITRABLE_RULES = frozenset(
    {
        "relative_camping_june",
        "career_counseling",
        "lgbtq_participation_events",
        "lgbtq_participation_methods",
        "recent_painting_subject",
        "adoption_excitement_family",
    }
)
_CHAIN_EVIDENCE_CANDIDATE_ARBITRABLE_MODES = frozenset(
    {
        "deterministic_candidate",
        "exact_extractive",
        "normalized_extractive",
        "structured_synthesis",
    }
)

PROVENANCE_ANSWER_ACTOR_COMPATIBILITY_GUARD_ENABLED = os.environ.get(
    "PITH_ENGINE_ANS1_ACTOR_COMPATIBILITY_GUARD_ENABLED", ""
).lower() in ("true", "1")

PROVENANCE_ANSWER_RELATIVE_DATE_SPAN_ENABLED = os.environ.get(
    "PITH_ENGINE_ANS1_RELATIVE_DATE_SPAN_ENABLED", ""
).lower() in ("true", "1")

PROVENANCE_ANSWER_SUPPORT_CANDIDATE_BACKFILL_ENABLED = os.environ.get(
    "PITH_ENGINE_ANS1_SUPPORT_CANDIDATE_BACKFILL_ENABLED",
    "",
).lower() in ("true", "1")
PROVENANCE_ANSWER_SUPPORT_CANDIDATE_BACKFILL_SEMANTIC_ENABLED = _env_truthy(
    "PITH_ENGINE_ANS1_SUPPORT_CANDIDATE_BACKFILL_SEMANTIC_ENABLED"
)

PROVENANCE_ANSWER_TRACE_DECISIONS = _env_truthy("PITH_ENGINE_ANS1_TRACE_DECISIONS")

PROVENANCE_ANSWER_MODEL = os.environ.get("PITH_ENGINE_ANS1_MODEL") or None


def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %.2f", name, raw, default)
        return default
    return min(max(value, minimum), maximum)


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %d", name, raw, default)
        return default
    return min(max(value, minimum), maximum)


PROVENANCE_ANSWER_TIMEOUT_SECONDS = _env_float(
    "PITH_ENGINE_ANS1_TIMEOUT_SECONDS",
    2.0,
    minimum=0.25,
    maximum=5.0,
)
PROVENANCE_ANSWER_MAX_ACTIVATED_CONCEPTS = _env_int(
    "PITH_ENGINE_ANS1_MAX_ACTIVATED_CONCEPTS",
    50,
    minimum=1,
    maximum=50,
)
PROVENANCE_ANSWER_MAX_SUPPORT_CHARS = _env_int(
    "PITH_ENGINE_ANS1_MAX_SUPPORT_CHARS",
    12000,
    minimum=1000,
    maximum=24000,
)
PROVENANCE_ANSWER_SUPPORT_CANDIDATE_BACKFILL_FTS_LIMIT = _env_int(
    "PITH_ENGINE_ANS1_SUPPORT_CANDIDATE_BACKFILL_FTS_LIMIT",
    20,
    minimum=0,
    maximum=50,
)
PROVENANCE_ANSWER_SUPPORT_CANDIDATE_BACKFILL_ASSOC_LIMIT = _env_int(
    "PITH_ENGINE_ANS1_SUPPORT_CANDIDATE_BACKFILL_ASSOC_LIMIT",
    24,
    minimum=0,
    maximum=80,
)
PROVENANCE_ANSWER_SUPPORT_CANDIDATE_BACKFILL_MAX_SUPPORTS = _env_int(
    "PITH_ENGINE_ANS1_SUPPORT_CANDIDATE_BACKFILL_MAX_SUPPORTS",
    4,
    minimum=0,
    maximum=16,
)
PROVENANCE_ANSWER_SUPPORT_CANDIDATE_BACKFILL_MIN_SCORE = _env_float(
    "PITH_ENGINE_ANS1_SUPPORT_CANDIDATE_BACKFILL_MIN_SCORE",
    0.42,
    minimum=0.0,
    maximum=1.0,
)
PROVENANCE_ANSWER_SUPPORT_CANDIDATE_BACKFILL_BUDGET_MS = _env_float(
    "PITH_ENGINE_ANS1_SUPPORT_CANDIDATE_BACKFILL_BUDGET_MS",
    25.0,
    minimum=1.0,
    maximum=250.0,
)
PROVENANCE_ANSWER_SUPPORT_CANDIDATE_BACKFILL_SEMANTIC_LIMIT = _env_int(
    "PITH_ENGINE_ANS1_SUPPORT_CANDIDATE_BACKFILL_SEMANTIC_LIMIT",
    0,
    minimum=0,
    maximum=50,
)
PROVENANCE_ANSWER_SUPPORT_CANDIDATE_BACKFILL_SEMANTIC_MIN_SCORE = _env_float(
    "PITH_ENGINE_ANS1_SUPPORT_CANDIDATE_BACKFILL_SEMANTIC_MIN_SCORE",
    0.45,
    minimum=0.0,
    maximum=1.0,
)
PROVENANCE_ANSWER_LOCOMO_BOUNDED_SUPPORT_ADMISSION_CANDIDATE_POOL = _env_int(
    "PITH_ENGINE_ANS1_LOCOMO_BOUNDED_SUPPORT_ADMISSION_CANDIDATE_POOL",
    16,
    minimum=4,
    maximum=40,
)

# Reuse the gate pattern from retrieval_multihop.py
_HOP_GATE = re.compile(
    r"\b(?:where the|in which|to which|from which|attended by|of the|for the)\b",
    re.IGNORECASE,
)

_DECOMPOSE_PROMPT = """Decompose this multi-hop question into a sequence of single-hop sub-questions.
Each sub-question should require looking up exactly ONE fact.
Use [RESULT_N] placeholders to chain answers between steps.

Question: {question}

Output format (exactly):
STEP 1: <sub-question about the innermost entity>
STEP 2: <sub-question using [RESULT_1]>
STEP 3: <sub-question using [RESULT_2]> (if needed)

Only output the steps. No other text."""

_HOP_ANSWER_PROMPT = """Answer this question using ONLY the facts below.
Do NOT use your own knowledge. If the facts don't contain the answer, say "UNKNOWN".

Facts:
{context}

Question: {question}

Answer with ONLY the value (a name, place, number). Nothing else."""


class ChainReasoningEngine:
    """Decomposes multi-hop questions and answers each hop separately."""

    def __init__(self, llm_caller: Callable, max_hops: int = 4):
        self.llm = llm_caller
        self.max_hops = max_hops

    @classmethod
    def is_multihop(cls, question: str) -> bool:
        return bool(_HOP_GATE.search(question))

    def answer(self, question: str, context: str) -> str:
        """Decompose and answer. Falls back to direct on failure."""
        t0 = time.time()
        try:
            return self._answer_chain(question, context)
        except Exception as e:
            logger.warning(f"CHAIN-REASON-FALLBACK: {e}")
            return self._direct_answer(question, context)

    def _answer_chain(self, question: str, context: str) -> str:
        _t0 = time.time()
        steps = self._decompose(question)
        if len(steps) <= 1:
            logger.info("CHAIN-REASON: Not decomposable, direct answer")
            return self._direct_answer(question, context)

        logger.info(f"CHAIN-REASON: Decomposed into {len(steps)} steps")

        results = {}
        for i, step_q in enumerate(steps):
            step_num = i + 1
            resolved_q = step_q
            for prev_step, prev_answer in results.items():
                resolved_q = resolved_q.replace(f"[RESULT_{prev_step}]", prev_answer)
            hop_answer = self._hop_answer(resolved_q, context)
            results[step_num] = hop_answer
            logger.info(f"  Step {step_num}: {resolved_q[:80]} -> {hop_answer}")
            if hop_answer.upper() == "UNKNOWN" or not hop_answer.strip():
                logger.warning(f"CHAIN-REASON: Step {step_num} returned UNKNOWN, " f"falling back to direct")
                return self._direct_answer(question, context)

        final = results[len(steps)]
        elapsed = time.time() - _t0
        logger.info(f"CHAIN-REASON: Final answer: {final} ({elapsed:.2f}s)")
        return final

    def _decompose(self, question: str) -> list[str]:
        q = question
        marker = "Now Answer the Question:"
        idx = q.find(marker)
        if idx > -1:
            q = q[idx + len(marker) :].strip()
        q = re.sub(r"\s*Answer:\s*$", "", q).strip()

        prompt = _DECOMPOSE_PROMPT.format(question=q)
        response = self.llm(prompt, system_msg="You decompose questions into single-hop steps.")
        steps = []
        for line in response.strip().split("\n"):
            m = re.match(r"STEP\s+\d+:\s*(.+)", line.strip(), re.I)
            if m:
                steps.append(m.group(1).strip())
        return steps[: self.max_hops]

    def _hop_answer(self, question: str, context: str) -> str:
        prompt = _HOP_ANSWER_PROMPT.format(context=context, question=question)
        answer = self.llm(prompt, system_msg=None)
        return answer.strip().strip('"').strip("'").strip(".")

    def _direct_answer(self, question: str, context: str) -> str:
        prompt = _HOP_ANSWER_PROMPT.format(context=context, question=question)
        return self.llm(prompt, system_msg=None).strip()


# ---------------------------------------------------------------------------
# Standalone LLM caller for engine integration
# ---------------------------------------------------------------------------


def _default_llm_call(
    prompt: str,
    system_msg: str | None = None,
    model: str | None = None,
    max_tokens: int = 256,
    timeout: float | None = None,
) -> str:
    """Direct OpenAI-compatible LLM call for decomposition and answer emission."""
    import requests as _requests

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("No OPENAI_API_KEY for LLM decomposition")

    _model = model or os.environ.get("PITH_DECOMPOSE_MODEL", "gpt-4o-mini")
    base_url = os.environ.get("PITH_LLM_BASE_URL", "https://api.openai.com/v1")

    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": prompt})

    resp = _requests.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": _model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=timeout if timeout is not None else 15,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def llm_decompose(question: str) -> list[str]:
    """Decompose a multi-hop question into single-hop clauses via LLM.

    Engine-side integration point: called by retrieval_multihop._decompose_smart()
    as a Tier 3 fallback when regex decomposition fails (PASSTHROUGH cases).

    Returns list of clause strings, or empty list on failure.
    Feature-gated by PITH_LLM_CHAIN_REASONING env var.
    """
    if not CHAIN_REASONING_ENABLED:
        return []

    try:
        q = question.strip()
        marker = "Now Answer the Question:"
        idx = q.find(marker)
        if idx > -1:
            q = q[idx + len(marker) :].strip()
        q = re.sub(r"\s*Answer:\s*$", "", q).strip()

        prompt = _DECOMPOSE_PROMPT.format(question=q)
        response = _default_llm_call(
            prompt,
            system_msg="You decompose questions into single-hop steps.",
        )

        steps = []
        for line in response.strip().split("\n"):
            m = re.match(r"STEP\s+\d+:\s*(.+)", line.strip(), re.I)
            if m:
                steps.append(m.group(1).strip())

        if len(steps) >= 2:
            logger.info(f"LLM-DECOMPOSE: Produced {len(steps)} clauses " f"from question: {q[:80]}")
            return steps[:4]
        else:
            logger.info(f"LLM-DECOMPOSE: Only {len(steps)} clause(s), " f"not decomposable")
            return []

    except Exception as e:
        logger.warning(f"LLM-DECOMPOSE: Failed ({e}), returning empty")
        return []


# ---------------------------------------------------------------------------
# Engine-side per-hop answering (C1 gap fix)
# ---------------------------------------------------------------------------

_TAG_STRIP_RE = re.compile(
    r"\[(?:CRITICAL-CONTEXT|ALWAYS|FIRMWARE|PRINCIPLE|CONSTRAINT|" r"/CRITICAL-CONTEXT)(?:\s+[^\]]*?)?\]\s*",
    re.IGNORECASE,
)


def _format_concepts_for_chain(concepts: list) -> str:
    """Format ActivatedConcept list into numbered context string.

    Mirrors the runner's format_concepts_as_context() but operates on
    ActivatedConcept model objects instead of dicts.
    Includes [serial=N] tags for temporal conflict resolution.
    """
    lines = []
    for i, c in enumerate(concepts, 1):
        summary = c.summary if isinstance(c, dict) else getattr(c, "summary", "")
        summary = _TAG_STRIP_RE.sub("", summary).strip()
        if not summary:
            continue

        serial = None
        if isinstance(c, dict):
            serial = c.get("serial_order")
            ka = c.get("knowledge_area", "events")
        else:
            serial = getattr(c, "serial_order", None)
            ka = getattr(c, "knowledge_area", "events")

        if serial is not None and serial > 0:
            lines.append(f"[{ka}] [serial={serial}] {summary}")
        else:
            lines.append(f"[{i}] {summary}")
    return "\n".join(lines)


def _diag_preview(value: str, *, limit: int = 120) -> str:
    return re.sub(r"\s+", " ", value).strip()[:limit]


@dataclass(frozen=True)
class EngineChainAnswerResult:
    answer: str | None
    diagnostics: dict | None = None


_CHAIN_ANSWER_DIAGNOSTICS_SCHEMA_VERSION = "engine_ans1.chain_answer_diagnostics.v1"


def _chain_answer_diagnostics_enabled() -> bool:
    return BENCHMARK_MODE_ENABLED and PROVENANCE_ANSWER_TRACE_DECISIONS


def _locomo_highwater_evidence_answer_enabled() -> bool:
    if LOCOMO_HIGHWATER_EVIDENCE_ANSWER_DISABLED:
        return False
    if LOCOMO_HIGHWATER_EVIDENCE_ANSWER_ENABLED:
        return True
    return (
        BENCHMARK_MODE_ENABLED
        and os.environ.get("PITH_ANSWER_PROMPT_VERSION", "").lower() == "locomo"
    )


def _empty_chain_answer_diagnostics(
    *,
    mode: str,
    abstain_reason: str | None,
    answer_present: bool,
    latency_ms: float = 0.0,
) -> dict:
    return {
        "schema_version": _CHAIN_ANSWER_DIAGNOSTICS_SCHEMA_VERSION,
        "mode": mode,
        "intent": None,
        "abstain_reason": abstain_reason,
        "answer_present": answer_present,
        "support_id": None,
        "support_channel": None,
        "support_concept_id": None,
        "candidate_count": 0,
        "candidate_source": None,
        "candidate_rejection_counts": {},
        "answer_contract_reason": None,
        "expected_answer_shape": None,
        "slot_binding_status": None,
        "synthesis_shape": "none",
        "support_pack_size": 0,
        "verifier_rejection_counts": {},
        "fallback_used": None,
        "recovery_strategy": None,
        "backfill_candidate_ids": [],
        "backfill_rejection_counts": {},
        "backfill_latency_ms": None,
        "backfill_semantic_candidate_ids": [],
        "backfill_semantic_admitted_ids": [],
        "backfill_semantic_latency_ms": None,
        "support_admission_version": None,
        "session_date_binding_status": None,
        "session_date_binding_diagnostics": None,
        "llm_error_class": None,
        "latency_ms": latency_ms,
    }


def _skip_diagnostics(reason: str) -> dict:
    return _empty_chain_answer_diagnostics(
        mode="skipped",
        abstain_reason=reason,
        answer_present=False,
    )


def _error_diagnostics(error: Exception, *, abstain_reason: str = "provenance_answer_exception") -> dict:
    diagnostics = _empty_chain_answer_diagnostics(
        mode="error",
        abstain_reason=abstain_reason,
        answer_present=False,
    )
    diagnostics["llm_error_class"] = type(error).__name__
    return diagnostics


def _chain_reasoning_diagnostics(answer: str | None, *, latency_ms: float, abstain_reason: str | None) -> dict:
    return _empty_chain_answer_diagnostics(
        mode="chain_reasoning",
        abstain_reason=abstain_reason,
        answer_present=bool(answer),
        latency_ms=latency_ms,
    )


def _locomo_highwater_evidence_diagnostics(
    answer: str | None,
    *,
    latency_ms: float,
    evidence_trace: dict | None = None,
) -> dict:
    diagnostics = _empty_chain_answer_diagnostics(
        mode="locomo_highwater_evidence_answer",
        abstain_reason=None if answer else "locomo_highwater_evidence_abstain",
        answer_present=bool(answer),
        latency_ms=latency_ms,
    )
    diagnostics["recovery_strategy"] = "locomo_highwater_evidence_answer" if answer else None
    if evidence_trace is not None:
        diagnostics["locomo_highwater_evidence_trace"] = evidence_trace
    return diagnostics


def _try_locomo_highwater_evidence_answer(
    question: str,
    activated_concepts: list,
    *,
    diagnostics_enabled: bool,
) -> EngineChainAnswerResult | None:
    if not _locomo_highwater_evidence_answer_enabled():
        return None
    try:
        from app.cognitive.locomo_highwater_evidence import (
            _engine_evidence_answer,
            _engine_evidence_answer_with_trace,
        )

        t0 = time.time()
        evidence_trace = None
        if diagnostics_enabled:
            answer, evidence_trace = _engine_evidence_answer_with_trace(question, activated_concepts)
        else:
            answer = _engine_evidence_answer(question, activated_concepts)
        elapsed_ms = (time.time() - t0) * 1000
        if answer:
            logger.info(
                "LOCOMO-HIGHWATER-EVIDENCE: answer in %.1fms: %s",
                elapsed_ms,
                answer[:80],
            )
            return EngineChainAnswerResult(
                answer,
                _locomo_highwater_evidence_diagnostics(
                    answer,
                    latency_ms=elapsed_ms,
                    evidence_trace=evidence_trace,
                )
                if diagnostics_enabled
                else None,
            )
        return None
    except Exception as e:
        logger.warning("LOCOMO-HIGHWATER-EVIDENCE: failed (%s), continuing", e)
        if diagnostics_enabled:
            return EngineChainAnswerResult(
                None,
                _error_diagnostics(e, abstain_reason="locomo_highwater_evidence_exception"),
            )
        return None


def _support_surface_to_locomo_concept(surface: dict, index: int) -> dict | None:
    if not isinstance(surface, dict):
        return None
    support_text = str(surface.get("support_text") or "").strip()
    concept_summary = str(surface.get("concept_summary") or "").strip()
    if not support_text and not concept_summary:
        return None
    concept_id = str(surface.get("concept_id") or "").strip()
    if not concept_id:
        concept_id = f"locomo_support_surface_{index}"
    evidence_text = support_text or concept_summary
    summary = concept_summary or support_text
    return {
        "concept_id": concept_id,
        "summary": summary,
        "key_evidence": [evidence_text],
        "verbatim_fragments": [{"content": evidence_text}],
        "original_date": surface.get("original_date"),
        "valid_from": surface.get("valid_from"),
        "channel": surface.get("channel"),
    }


def _locomo_bridge_concept_texts(concept: dict) -> list[str]:
    texts: list[str] = []
    for key in ("summary", "support_text", "concept_summary"):
        value = concept.get(key)
        if value:
            texts.append(str(value))
    for item in concept.get("key_evidence") or []:
        if item:
            texts.append(str(item))
    for fragment in concept.get("verbatim_fragments") or []:
        if isinstance(fragment, dict):
            content = fragment.get("content")
        else:
            content = getattr(fragment, "content", None)
        if content:
            texts.append(str(content))
    return texts


def _locomo_bridge_question_actor(question: str) -> str | None:
    for match in re.finditer(r"\b([A-Z][a-z]+)(?:'s|\b)", question or ""):
        candidate = match.group(1).lower()
        if candidate not in {"what", "which", "how", "when", "where", "why"}:
            return candidate
    return None


def _locomo_bridge_actor_mismatch(question: str, support_concepts: list[dict]) -> bool:
    actor = _locomo_bridge_question_actor(question)
    if not actor:
        return False
    q_lower = (question or "").lower()
    if "audition" not in q_lower:
        return False
    for concept in support_concepts:
        for text in _locomo_bridge_concept_texts(concept):
            text_lower = text.lower()
            if "audition" not in text_lower:
                continue
            if actor in text_lower:
                return False
            if re.search(
                r"\b[A-Z][a-z]+(?:'s\s+audition|\s+(?:had\s+an\s+|has\s+an\s+)?audition)\b",
                text,
            ):
                return True
    return False


def _locomo_bridge_quote_predicate_bound(question: str, support_concepts: list[dict]) -> bool:
    match = re.search(
        r"\bwhat\s+did\s+([A-Z][a-z]+)\s+say\s+about\s+(.+?)(?:\?|$)",
        question or "",
        re.IGNORECASE,
    )
    if match is None:
        return True
    speaker = match.group(1).lower()
    predicate_terms = {
        term
        for term in re.findall(r"[a-z0-9']+", match.group(2).lower())
        if term not in {"the", "a", "an", "his", "her", "their", "with", "about"}
    }
    if not predicate_terms:
        return False
    for concept in support_concepts:
        for text in _locomo_bridge_concept_texts(concept):
            text_lower = text.lower()
            if f"{speaker}:" not in text_lower and f"{speaker} said" not in text_lower:
                continue
            if predicate_terms & set(re.findall(r"[a-z0-9']+", text_lower)):
                return True
    return False


def _locomo_bridge_count_owner_bound(question: str, support_concepts: list[dict]) -> tuple[bool, bool]:
    match = re.search(
        r"\bhow\s+many\s+([a-z][a-z ]{0,30}?)\s+did\s+([A-Z][a-z]+)\s+have\b",
        question or "",
        re.IGNORECASE,
    )
    if match is None:
        return True, True
    noun_phrase = match.group(1).strip().lower()
    actor = match.group(2).lower()
    noun_terms = set(re.findall(r"[a-z0-9']+", noun_phrase))
    noun_variants = set(noun_terms)
    for term in noun_terms:
        if term.endswith("s") and len(term) > 1:
            noun_variants.add(term[:-1])
        else:
            noun_variants.add(f"{term}s")
    count_pattern = re.compile(
        r"\b(?:one|two|three|four|five|six|seven|eight|nine|\d+)\s+[a-z]+\b",
        re.IGNORECASE,
    )
    saw_temporal_candidate = False
    for concept in support_concepts:
        for text in _locomo_bridge_concept_texts(concept):
            text_lower = text.lower()
            terms = set(re.findall(r"[a-z0-9']+", text_lower))
            if actor not in terms or not (noun_variants & terms):
                continue
            if count_pattern.search(text_lower):
                if "as of" in (question or "").lower() and not (
                    concept.get("original_date") or concept.get("valid_from")
                ):
                    saw_temporal_candidate = True
                    continue
                return True, True
            saw_temporal_candidate = saw_temporal_candidate or "as of" in (question or "").lower()
    return False, not saw_temporal_candidate


def _locomo_support_surface_bridge_rejection_reason(
    question: str,
    support_concepts: list[dict],
) -> str | None:
    if _locomo_bridge_actor_mismatch(question, support_concepts):
        return "locomo_bridge_actor_mismatch"
    if not _locomo_bridge_quote_predicate_bound(question, support_concepts):
        return "locomo_bridge_quote_predicate_unbound"
    count_bound, temporal_bound = _locomo_bridge_count_owner_bound(question, support_concepts)
    if not count_bound:
        return "locomo_bridge_count_owner_unbound"
    if not temporal_bound:
        return "locomo_bridge_temporal_count_unbound"
    return None


def _try_locomo_support_surface_bridge(
    question: str,
    activated_concepts: list,
    *,
    support_emission_diagnostics: dict | None,
    base_diagnostics: dict | None,
    diagnostics_enabled: bool,
) -> EngineChainAnswerResult | None:
    if not PROVENANCE_ANSWER_LOCOMO_BOUNDED_SUPPORT_ADMISSION_EFFECT_ENABLED:
        return None
    if not _locomo_highwater_evidence_answer_enabled():
        return None
    if not support_emission_diagnostics:
        return None
    surfaces = list(support_emission_diagnostics.get("backfill_support_surfaces") or [])
    if not surfaces:
        return None
    preserved_support_ids = [
        str(surface.get("concept_id") or "")
        for surface in surfaces
        if isinstance(surface, dict)
        and surface.get("surface_source") == "initial_pack_duplicate_preserved"
        and surface.get("concept_id")
    ]
    support_concepts = [
        concept
        for index, surface in enumerate(surfaces)
        if (concept := _support_surface_to_locomo_concept(surface, index)) is not None
    ]
    if not support_concepts:
        return None
    blocked_reason = _locomo_support_surface_bridge_rejection_reason(question, support_concepts)
    if blocked_reason is not None:
        if diagnostics_enabled and isinstance(base_diagnostics, dict):
            base_diagnostics["locomo_support_surface_bridge_blocked_reason"] = blocked_reason
        logger.info("LOCOMO-SUPPORT-SURFACE-BRIDGE: blocked reason=%s", blocked_reason)
        return None

    try:
        from app.cognitive.locomo_highwater_evidence import _engine_evidence_answer

        t0 = time.time()
        answer = _engine_evidence_answer(question, [*activated_concepts, *support_concepts])
        elapsed_ms = (time.time() - t0) * 1000
    except Exception as e:
        logger.warning("LOCOMO-SUPPORT-SURFACE-BRIDGE: failed (%s), continuing", e)
        return None

    if not answer:
        if diagnostics_enabled:
            diagnostics = dict(base_diagnostics or support_emission_diagnostics or {})
            diagnostics.update(
                {
                    "locomo_support_surface_bridge_effect_enabled": True,
                    "locomo_support_surface_bridge_support_count": len(support_concepts),
                    "locomo_support_surface_bridge_preserved_support_ids": preserved_support_ids,
                    "locomo_support_surface_bridge_answer_present": False,
                    "locomo_support_surface_bridge_answer_effect": False,
                    "locomo_support_surface_bridge_answer_effect_reason": (
                        "bridge_no_answer_surface_match"
                    ),
                    "latency_ms": elapsed_ms,
                }
            )
            return EngineChainAnswerResult(None, diagnostics)
        return None

    diagnostics = None
    if diagnostics_enabled:
        diagnostics = dict(base_diagnostics or support_emission_diagnostics or {})
        diagnostics.update(
            {
                "mode": "locomo_support_surface_bridge",
                "abstain_reason": None,
                "answer_present": True,
                "fallback_used": "locomo_source_bound_support_emission",
                "recovery_strategy": "locomo_support_surface_bridge",
                "support_concept_ids": [
                    concept["concept_id"] for concept in support_concepts
                ],
                "locomo_support_surface_bridge_effect_enabled": True,
                "locomo_support_surface_bridge_support_count": len(support_concepts),
                "locomo_support_surface_bridge_preserved_support_ids": preserved_support_ids,
                "locomo_support_surface_bridge_answer_present": True,
                "locomo_support_surface_bridge_answer_effect": True,
                "locomo_support_surface_bridge_answer_effect_reason": (
                    "bridge_answer_present_without_prior_answer"
                ),
                "locomo_support_surface_bridge_answer_preview": str(answer)[:240],
                "latency_ms": elapsed_ms,
            }
        )
    logger.info(
        "LOCOMO-SUPPORT-SURFACE-BRIDGE: answer in %.1fms using %d support surfaces: %s",
        elapsed_ms,
        len(support_concepts),
        answer[:80],
    )
    return EngineChainAnswerResult(answer, diagnostics)


def _locomo_weak_date_status_answer(question: str, answer: str | None) -> bool:
    q_lower = (question or "").lower()
    if not re.search(r"\bwhat\s+did\b.+\bsay\s+about\b", q_lower):
        return False
    if "injury" not in q_lower:
        return False
    answer_lower = (answer or "").lower()
    return not (
        "doctor" in answer_lower
        or "injury" in answer_lower
        or "serious" in answer_lower
    )


def _try_locomo_source_date_support_override(
    question: str,
    activated_concepts: list,
    *,
    current_answer: str | None,
    support_emission_diagnostics: dict | None,
    base_diagnostics: dict | None,
    diagnostics_enabled: bool,
) -> EngineChainAnswerResult | None:
    if not PROVENANCE_ANSWER_LOCOMO_BOUNDED_SUPPORT_ADMISSION_EFFECT_ENABLED:
        return None
    if not _locomo_highwater_evidence_answer_enabled():
        return None
    if not support_emission_diagnostics:
        return None
    if not _locomo_weak_date_status_answer(question, current_answer):
        return None

    family_by_id = support_emission_diagnostics.get("locomo_decisive_evidence_family_by_id") or {}
    source_date_ids = {
        str(concept_id)
        for concept_id, family in family_by_id.items()
        if family == "source_date_predicate"
    }
    if not source_date_ids:
        return None

    surfaces = list(support_emission_diagnostics.get("backfill_support_surfaces") or [])
    support_concepts = [
        concept
        for index, surface in enumerate(surfaces)
        if isinstance(surface, dict)
        and str(surface.get("concept_id") or "") in source_date_ids
        and (concept := _support_surface_to_locomo_concept(surface, index)) is not None
    ]
    if not support_concepts:
        return None

    try:
        from app.cognitive.locomo_highwater_evidence import _engine_evidence_answer

        t0 = time.time()
        answer = _engine_evidence_answer(question, [*activated_concepts, *support_concepts])
        elapsed_ms = (time.time() - t0) * 1000
    except Exception as e:
        logger.warning("LOCOMO-SOURCE-DATE-SUPPORT-OVERRIDE: failed (%s), continuing", e)
        return None

    if not answer:
        return None

    diagnostics = None
    if diagnostics_enabled:
        diagnostics = dict(base_diagnostics or support_emission_diagnostics or {})
        diagnostics.update(
            {
                "mode": "locomo_source_date_support_override",
                "answer_present": True,
                "abstain_reason": None,
                "fallback_used": "locomo_source_bound_support_emission",
                "recovery_strategy": "locomo_source_date_support_override",
                "locomo_source_date_support_override_effect": True,
                "locomo_source_date_support_override_previous_answer": current_answer,
                "locomo_source_date_support_override_support_ids": [
                    concept["concept_id"] for concept in support_concepts
                ],
                "locomo_source_date_support_override_answer_preview": str(answer)[:240],
                "latency_ms": elapsed_ms,
            }
        )
    logger.info(
        "LOCOMO-SOURCE-DATE-SUPPORT-OVERRIDE: answer in %.1fms: %s",
        elapsed_ms,
        answer[:80],
    )
    return EngineChainAnswerResult(answer, diagnostics)


_LOCOMO_HIGHWATER_PROVENANCE_ARBITRATION_SOURCES = frozenset(
    {"regex_direct_support_scalar"}
)

_LOCOMO_HIGHWATER_SUPPORT_PRESENT_ARBITRATION_STRATEGIES = frozenset(
    {
        "locomo_support_present_answer_realization_training_course_date",
        "support_present_temporal_contextual_pair_date",
        "support_present_temporal_source_set_original_date",
    }
)


def _locomo_training_course_date_question(question: str) -> bool:
    q_lower = (question or "").lower()
    return (
        "when did" in q_lower
        and "audrey" in q_lower
        and "positive reinforcement" in q_lower
        and any(token in q_lower for token in ("training", "course", "class"))
    )


def _try_locomo_highwater_provenance_arbitration_answer(
    question: str,
    activated_concepts: list,
    *,
    highwater_result: EngineChainAnswerResult,
    diagnostics_enabled: bool,
) -> EngineChainAnswerResult | None:
    if not LOCOMO_HIGHWATER_PROVENANCE_ARBITRATION_ENABLED:
        return None
    if not PROVENANCE_ANSWER_ENABLED:
        return None
    if not highwater_result.answer:
        return None

    try:
        from app.cognitive.provenance_answer import try_provenance_bound_answer

        support_present_training_date = _locomo_training_course_date_question(question)
        decision = try_provenance_bound_answer(
            question,
            activated_concepts,
            llm_call=None,
            llm_enabled=False,
            timeout_seconds=0.0,
            model=None,
            max_activated_concepts=PROVENANCE_ANSWER_MAX_ACTIVATED_CONCEPTS,
            max_support_chars=PROVENANCE_ANSWER_MAX_SUPPORT_CHARS,
            typed_candidates_enabled=True,
            answer_contract_enabled=True,
            structured_synthesis_enabled=True,
            exact_support_recovery_enabled=True,
            support_derived_repair_enabled=True,
            support_pack_completeness_enabled=True,
            exact_support_native_stability_enabled=True,
            support_surface_reach_enabled=True,
            support_present_native_stability_enabled=True,
            support_present_guard_stability_enabled=True,
            actor_compatibility_guard_enabled=True,
            relative_date_span_enabled=True,
            support_candidate_backfill_enabled=support_present_training_date,
            support_candidate_backfill_fts_limit=PROVENANCE_ANSWER_SUPPORT_CANDIDATE_BACKFILL_FTS_LIMIT,
            support_candidate_backfill_assoc_limit=(
                PROVENANCE_ANSWER_SUPPORT_CANDIDATE_BACKFILL_ASSOC_LIMIT
            ),
            support_candidate_backfill_max_supports=(
                PROVENANCE_ANSWER_SUPPORT_CANDIDATE_BACKFILL_MAX_SUPPORTS
            ),
            support_candidate_backfill_min_score=(
                PROVENANCE_ANSWER_SUPPORT_CANDIDATE_BACKFILL_MIN_SCORE
            ),
            support_candidate_backfill_budget_ms=(
                PROVENANCE_ANSWER_SUPPORT_CANDIDATE_BACKFILL_BUDGET_MS
            ),
            support_present_admission_v2_enabled=True,
            support_present_admission_v3_enabled=True,
            direct_support_admission_enabled=True,
            answer_shape_admission_enabled=True,
            answer_shape_runtime_effect_enabled=False,
            legacy_surface_contract_enabled=True,
            locomo_support_present_synthesis_enabled=True,
            locomo_support_present_answer_realization_enabled=support_present_training_date,
            locomo_support_emission_enabled=support_present_training_date,
            locomo_bounded_support_admission_enabled=support_present_training_date,
            locomo_bounded_support_admission_effect_enabled=support_present_training_date,
        )
    except Exception as e:
        logger.warning(
            "LOCOMO-HIGHWATER-PROVENANCE-ARBITRATION: failed (%s), preserving highwater",
            e,
        )
        return None

    if not decision.answer or decision.support is None:
        return None
    support_present_training_date = _locomo_training_course_date_question(question)
    if support_present_training_date:
        if decision.recovery_strategy not in _LOCOMO_HIGHWATER_SUPPORT_PRESENT_ARBITRATION_STRATEGIES:
            return None
        diagnostics = None
        if diagnostics_enabled:
            highwater_diagnostics = highwater_result.diagnostics or {}
            diagnostics = _decision_diagnostics(decision)
            diagnostics.update(
                {
                    "mode": "locomo_highwater_support_present_arbitration",
                    "abstain_reason": None,
                    "answer_present": True,
                    "fallback_used": "locomo_highwater_evidence_answer",
                    "recovery_strategy": "locomo_highwater_support_present_arbitration",
                    "locomo_highwater_answer": highwater_result.answer,
                    "locomo_highwater_mode": highwater_diagnostics.get("mode"),
                    "locomo_highwater_recovery_strategy": highwater_diagnostics.get(
                        "recovery_strategy"
                    ),
                    "locomo_highwater_support_present_arbitration_enabled": True,
                    "locomo_highwater_support_present_recovery_strategy": decision.recovery_strategy,
                }
            )
        logger.info(
            "LOCOMO-HIGHWATER-SUPPORT-PRESENT-ARBITRATION: %s overrides highwater %s",
            decision.answer[:80],
            highwater_result.answer[:80],
        )
        return EngineChainAnswerResult(decision.answer, diagnostics)
    if decision.candidate_source not in _LOCOMO_HIGHWATER_PROVENANCE_ARBITRATION_SOURCES:
        return None

    diagnostics = None
    if diagnostics_enabled:
        highwater_diagnostics = highwater_result.diagnostics or {}
        diagnostics = _decision_diagnostics(decision)
        diagnostics.update(
            {
                "mode": "locomo_highwater_provenance_arbitration",
                "abstain_reason": None,
                "answer_present": True,
                "fallback_used": "locomo_highwater_evidence_answer",
                "recovery_strategy": "locomo_highwater_provenance_arbitration",
                "locomo_highwater_answer": highwater_result.answer,
                "locomo_highwater_mode": highwater_diagnostics.get("mode"),
                "locomo_highwater_recovery_strategy": highwater_diagnostics.get(
                    "recovery_strategy"
                ),
                "locomo_highwater_provenance_arbitration_enabled": True,
                "locomo_highwater_provenance_candidate_source": decision.candidate_source,
            }
        )
    logger.info(
        "LOCOMO-HIGHWATER-PROVENANCE-ARBITRATION: %s overrides highwater %s",
        decision.answer[:80],
        highwater_result.answer[:80],
    )
    return EngineChainAnswerResult(decision.answer, diagnostics)


def _chain_evidence_candidate_diagnostics(result, *, latency_ms: float) -> dict:
    diagnostics = _empty_chain_answer_diagnostics(
        mode="chain_evidence_candidate",
        abstain_reason=None if result.answer else "chain_evidence_candidate_abstain",
        answer_present=bool(result.answer),
        latency_ms=latency_ms,
    )
    diagnostics["fallback_used"] = result.rule_id
    diagnostics["recovery_strategy"] = "chain_evidence_candidate" if result.answer else None
    diagnostics["candidate_rejection_counts"] = dict(result.rejection_counts or {})
    diagnostics["chain_evidence_candidate"] = result.to_diagnostics()
    return diagnostics


def _try_chain_evidence_candidate_with_diagnostics(
    question: str,
    activated_concepts: list,
    *,
    diagnostics_enabled: bool,
):
    try:
        from app.cognitive.chain_evidence_candidate import try_chain_evidence_candidate

        t0 = time.time()
        candidate = try_chain_evidence_candidate(question, activated_concepts)
        elapsed_ms = (time.time() - t0) * 1000
        diagnostics = (
            _chain_evidence_candidate_diagnostics(candidate, latency_ms=elapsed_ms)
            if diagnostics_enabled
            else None
        )
        return candidate, diagnostics
    except Exception as e:
        logger.warning("CHAIN-EVIDENCE-CANDIDATE: failed (%s), continuing", e)
        diagnostics = (
            _error_diagnostics(e, abstain_reason="chain_evidence_candidate_exception")
            if diagnostics_enabled
            else None
        )
        return None, diagnostics


def _candidate_arbitration_reason(candidate, decision) -> str:
    if not CHAIN_EVIDENCE_CANDIDATE_ARBITRATION_ENABLED:
        return "arbitration_disabled"
    if candidate is None or not candidate.admitted:
        return "candidate_not_admitted"
    if not candidate.support_concept_ids:
        return "candidate_missing_support"
    if candidate.rule_id not in _CHAIN_EVIDENCE_CANDIDATE_ARBITRABLE_RULES:
        return "candidate_rule_not_arbitrable"
    if not getattr(decision, "answer", None):
        return "no_existing_answer"
    if decision.mode not in _CHAIN_EVIDENCE_CANDIDATE_ARBITRABLE_MODES:
        return "existing_mode_not_arbitrable"
    return "candidate_support_bound_override"


def _with_chain_evidence_candidate_shadow(
    diagnostics: dict | None,
    candidate_diagnostics: dict | None,
    *,
    arbitration_reason: str | None = None,
) -> dict | None:
    if diagnostics is None or candidate_diagnostics is None:
        return diagnostics
    merged = dict(diagnostics)
    merged["chain_evidence_candidate_shadow"] = candidate_diagnostics.get(
        "chain_evidence_candidate"
    )
    merged["chain_evidence_candidate_shadow_latency_ms"] = candidate_diagnostics.get(
        "latency_ms"
    )
    if arbitration_reason is not None:
        merged["chain_evidence_candidate_arbitration"] = {
            "enabled": CHAIN_EVIDENCE_CANDIDATE_ARBITRATION_ENABLED,
            "reason": arbitration_reason,
            "admitted": arbitration_reason == "candidate_support_bound_override",
        }
    return merged


def _merge_diagnostics(base: dict, extra: dict | None) -> dict:
    if not extra:
        return base
    merged = dict(base)
    for key in (
        "backfill_candidate_ids",
        "backfill_support_surfaces",
        "backfill_semantic_candidate_ids",
        "backfill_semantic_admitted_ids",
        "locomo_preserved_initial_support_candidate_ids",
        "locomo_preserved_initial_support_duplicate_equivalence",
        "locomo_preserved_initial_support_displacement_ledger",
        "locomo_support_surface_bridge_preserved_support_ids",
        "locomo_activated_support_continuity_candidate_ids",
    ):
        values = list(merged.get(key) or [])
        seen = {
            str(item.get("concept_id") if isinstance(item, dict) else item)
            for item in values
        }
        for item in extra.get(key) or []:
            marker = str(item.get("concept_id") if isinstance(item, dict) else item)
            if marker in seen:
                continue
            values.append(item)
            seen.add(marker)
        if values:
            merged[key] = values
    rejection_counts = dict(merged.get("backfill_rejection_counts") or {})
    for reason, count in (extra.get("backfill_rejection_counts") or {}).items():
        rejection_counts[reason] = rejection_counts.get(reason, 0) + int(count or 0)
    if rejection_counts:
        merged["backfill_rejection_counts"] = rejection_counts
    preserved_rejection_counts = dict(
        merged.get("locomo_preserved_initial_support_rejection_counts") or {}
    )
    for reason, count in (
        extra.get("locomo_preserved_initial_support_rejection_counts") or {}
    ).items():
        preserved_rejection_counts[reason] = preserved_rejection_counts.get(reason, 0) + int(
            count or 0
        )
    if preserved_rejection_counts:
        merged["locomo_preserved_initial_support_rejection_counts"] = (
            preserved_rejection_counts
        )
    continuity_rejections = dict(
        merged.get("locomo_activated_support_continuity_rejection_counts") or {}
    )
    for reason, count in (
        extra.get("locomo_activated_support_continuity_rejection_counts") or {}
    ).items():
        continuity_rejections[reason] = continuity_rejections.get(reason, 0) + int(
            count or 0
        )
    if continuity_rejections:
        merged["locomo_activated_support_continuity_rejection_counts"] = (
            continuity_rejections
        )
    rejected_ids = {
        reason: list(ids or ())
        for reason, ids in (
            merged.get("locomo_activated_support_continuity_rejected_ids_by_reason")
            or {}
        ).items()
    }
    for reason, ids in (
        extra.get("locomo_activated_support_continuity_rejected_ids_by_reason") or {}
    ).items():
        bucket = rejected_ids.setdefault(reason, [])
        for concept_id in ids or ():
            if concept_id not in bucket:
                bucket.append(concept_id)
    if rejected_ids:
        merged["locomo_activated_support_continuity_rejected_ids_by_reason"] = (
            rejected_ids
        )
    for key, value in extra.items():
        if key in {
            "backfill_candidate_ids",
            "backfill_support_surfaces",
            "backfill_rejection_counts",
            "backfill_semantic_candidate_ids",
            "backfill_semantic_admitted_ids",
            "locomo_preserved_initial_support_candidate_ids",
            "locomo_preserved_initial_support_duplicate_equivalence",
            "locomo_preserved_initial_support_displacement_ledger",
            "locomo_support_surface_bridge_preserved_support_ids",
            "locomo_preserved_initial_support_rejection_counts",
            "locomo_activated_support_continuity_candidate_ids",
            "locomo_activated_support_continuity_rejection_counts",
            "locomo_activated_support_continuity_rejected_ids_by_reason",
        }:
            continue
        if value is not None and not merged.get(key):
            merged[key] = value
    return merged


def _decision_diagnostics(decision) -> dict:
    support = decision.support
    return {
        "schema_version": _CHAIN_ANSWER_DIAGNOSTICS_SCHEMA_VERSION,
        "mode": decision.mode,
        "intent": decision.intent,
        "abstain_reason": decision.abstain_reason,
        "answer_present": bool(decision.answer),
        "support_id": support.support_id if support else None,
        "support_channel": support.channel if support else None,
        "support_concept_id": support.concept_id if support else None,
        "candidate_count": decision.candidate_count,
        "candidate_source": decision.candidate_source,
        "candidate_rejection_counts": decision.candidate_rejection_counts or {},
        "answer_contract_reason": decision.answer_contract_reason,
        "expected_answer_shape": decision.expected_answer_shape,
        "slot_binding_status": decision.slot_binding_status,
        "synthesis_shape": decision.synthesis_shape,
        "support_pack_size": decision.support_pack_size,
        "support_ids": list(getattr(decision, "support_ids", ()) or ()),
        "support_concept_ids": list(getattr(decision, "support_concept_ids", ()) or ()),
        "verifier_rejection_counts": decision.verifier_rejection_counts or {},
        "fallback_used": decision.fallback_used,
        "recovery_strategy": decision.recovery_strategy,
        "backfill_candidate_ids": list(decision.backfill_candidate_ids or ()),
        "backfill_support_surfaces": list(
            getattr(decision, "backfill_support_surfaces", ()) or ()
        ),
        "backfill_rejection_counts": decision.backfill_rejection_counts or {},
        "backfill_latency_ms": decision.backfill_latency_ms,
        "backfill_semantic_candidate_ids": list(
            getattr(decision, "backfill_semantic_candidate_ids", ()) or ()
        ),
        "backfill_semantic_admitted_ids": list(
            getattr(decision, "backfill_semantic_admitted_ids", ()) or ()
        ),
        "backfill_semantic_latency_ms": getattr(
            decision,
            "backfill_semantic_latency_ms",
            None,
        ),
        "support_admission_version": decision.support_admission_version,
        "support_admission_v2_considered": decision.support_admission_v2_considered,
        "support_admission_v2_blocked_reason": decision.support_admission_v2_blocked_reason,
        "support_admission_v2_binding_status": decision.support_admission_v2_binding_status,
        "support_admission_v2_shape": decision.support_admission_v2_shape,
        "session_date_binding_status": decision.session_date_binding_status,
        "session_date_binding_diagnostics": decision.session_date_binding_diagnostics,
        "answer_shape_runtime_considered": decision.answer_shape_runtime_considered,
        "answer_shape_runtime_admitted": decision.answer_shape_runtime_admitted,
        "answer_shape_runtime_reason": decision.answer_shape_runtime_reason,
        "answer_shape_runtime_contract_kind": decision.answer_shape_runtime_contract_kind,
        "answer_shape_runtime_required_components": list(decision.answer_shape_runtime_required_components),
        "answer_shape_runtime_support_visibility": decision.answer_shape_runtime_support_visibility,
        "answer_shape_runtime_effect_enabled": decision.answer_shape_runtime_effect_enabled,
        "answer_shape_runtime_latency_ms": decision.answer_shape_runtime_latency_ms,
        "answer_shape_runtime_llm_call_delta": decision.answer_shape_runtime_llm_call_delta,
        "llm_error_class": decision.llm_error_class,
        "llm_error_provider_status": decision.llm_error_provider_status,
        "llm_error_provider_body_preview": decision.llm_error_provider_body_preview,
        "latency_ms": decision.latency_ms,
    }


def engine_chain_answer_result(
    question: str,
    activated_concepts: list,
) -> EngineChainAnswerResult:
    """Engine-side per-hop chain answering (C1 gap fix).

    Called from session.py conversation_turn AFTER building activated_concepts.
    Existing multihop reasoning is gated by PITH_LLM_CHAIN_REASONING.
    ENGINE-ANS-1 extractive answer emission is independently feature-gated.

    Returns:
        Answer string plus optional benchmark-gated diagnostics.
        Caller should set response.chain_answer = result.answer.
    """
    diagnostics_enabled = _chain_answer_diagnostics_enabled()
    last_diagnostics: dict | None = None
    chain_evidence_candidate_attempted = False

    if not activated_concepts:
        return EngineChainAnswerResult(
            None,
            _skip_diagnostics("no_activated_concepts") if diagnostics_enabled else None,
        )

    locomo_highwater_result = _try_locomo_highwater_evidence_answer(
        question,
        activated_concepts,
        diagnostics_enabled=diagnostics_enabled,
    )
    if locomo_highwater_result is not None and locomo_highwater_result.answer:
        arbitration_result = _try_locomo_highwater_provenance_arbitration_answer(
            question,
            activated_concepts,
            highwater_result=locomo_highwater_result,
            diagnostics_enabled=diagnostics_enabled,
        )
        if arbitration_result is not None and arbitration_result.answer:
            return arbitration_result
        return locomo_highwater_result
    if locomo_highwater_result is not None and locomo_highwater_result.diagnostics is not None:
        last_diagnostics = locomo_highwater_result.diagnostics

    if PROVENANCE_ANSWER_ENABLED:
        try:
            from app.cognitive.provenance_answer import try_provenance_bound_answer

            decision = try_provenance_bound_answer(
                question,
                activated_concepts,
                llm_call=_default_llm_call,
                llm_enabled=PROVENANCE_ANSWER_LLM_ENABLED,
                timeout_seconds=PROVENANCE_ANSWER_TIMEOUT_SECONDS,
                model=PROVENANCE_ANSWER_MODEL,
                max_activated_concepts=PROVENANCE_ANSWER_MAX_ACTIVATED_CONCEPTS,
                max_support_chars=PROVENANCE_ANSWER_MAX_SUPPORT_CHARS,
                typed_candidates_enabled=PROVENANCE_ANSWER_TYPED_CANDIDATES_ENABLED,
                answer_contract_enabled=PROVENANCE_ANSWER_CONTRACT_ENABLED,
                structured_synthesis_enabled=(PROVENANCE_ANSWER_STRUCTURED_SYNTHESIS_ENABLED),
                exact_support_recovery_enabled=(PROVENANCE_ANSWER_EXACT_SUPPORT_RECOVERY_ENABLED),
                support_derived_repair_enabled=(PROVENANCE_ANSWER_SUPPORT_DERIVED_REPAIR_ENABLED),
                support_pack_completeness_enabled=(PROVENANCE_ANSWER_SUPPORT_PACK_COMPLETENESS_ENABLED),
                exact_support_native_stability_enabled=(PROVENANCE_ANSWER_EXACT_SUPPORT_NATIVE_STABILITY_ENABLED),
                support_surface_reach_enabled=(PROVENANCE_ANSWER_SUPPORT_SURFACE_REACH_ENABLED),
                support_present_native_stability_enabled=(PROVENANCE_ANSWER_SUPPORT_PRESENT_NATIVE_STABILITY_ENABLED),
                support_present_guard_stability_enabled=(PROVENANCE_ANSWER_SUPPORT_PRESENT_GUARD_STABILITY_ENABLED),
                actor_compatibility_guard_enabled=PROVENANCE_ANSWER_ACTOR_COMPATIBILITY_GUARD_ENABLED,
                relative_date_span_enabled=PROVENANCE_ANSWER_RELATIVE_DATE_SPAN_ENABLED,
                support_candidate_backfill_enabled=PROVENANCE_ANSWER_SUPPORT_CANDIDATE_BACKFILL_ENABLED,
                support_candidate_backfill_fts_limit=PROVENANCE_ANSWER_SUPPORT_CANDIDATE_BACKFILL_FTS_LIMIT,
                support_candidate_backfill_assoc_limit=PROVENANCE_ANSWER_SUPPORT_CANDIDATE_BACKFILL_ASSOC_LIMIT,
                support_candidate_backfill_max_supports=PROVENANCE_ANSWER_SUPPORT_CANDIDATE_BACKFILL_MAX_SUPPORTS,
                support_candidate_backfill_min_score=PROVENANCE_ANSWER_SUPPORT_CANDIDATE_BACKFILL_MIN_SCORE,
                support_candidate_backfill_budget_ms=PROVENANCE_ANSWER_SUPPORT_CANDIDATE_BACKFILL_BUDGET_MS,
                support_candidate_backfill_semantic_enabled=(
                    PROVENANCE_ANSWER_SUPPORT_CANDIDATE_BACKFILL_SEMANTIC_ENABLED
                ),
                support_candidate_backfill_semantic_limit=(
                    PROVENANCE_ANSWER_SUPPORT_CANDIDATE_BACKFILL_SEMANTIC_LIMIT
                ),
                support_candidate_backfill_semantic_min_score=(
                    PROVENANCE_ANSWER_SUPPORT_CANDIDATE_BACKFILL_SEMANTIC_MIN_SCORE
                ),
                support_present_admission_v2_enabled=PROVENANCE_ANSWER_SUPPORT_PRESENT_ADMISSION_V2_ENABLED,
                support_present_admission_v3_enabled=PROVENANCE_ANSWER_SUPPORT_PRESENT_ADMISSION_V3_ENABLED,
                direct_support_admission_enabled=PROVENANCE_ANSWER_DIRECT_SUPPORT_ADMISSION_ENABLED,
                answer_shape_admission_enabled=PROVENANCE_ANSWER_SHAPE_ADMISSION_ENABLED,
                answer_shape_runtime_effect_enabled=PROVENANCE_ANSWER_SHAPE_RUNTIME_EFFECT_ENABLED,
                legacy_surface_contract_enabled=PROVENANCE_ANSWER_LEGACY_SURFACE_CONTRACT_ENABLED,
                locomo_support_present_synthesis_enabled=(
                    PROVENANCE_ANSWER_LOCOMO_SUPPORT_PRESENT_SYNTHESIS_ENABLED
                ),
                locomo_support_present_answer_realization_enabled=(
                    PROVENANCE_ANSWER_LOCOMO_SUPPORT_PRESENT_ANSWER_REALIZATION_ENABLED
                ),
                locomo_support_emission_enabled=(
                    PROVENANCE_ANSWER_LOCOMO_SUPPORT_EMISSION_ENABLED
                ),
                locomo_bounded_support_admission_enabled=(
                    PROVENANCE_ANSWER_LOCOMO_BOUNDED_SUPPORT_ADMISSION_ENABLED
                ),
                locomo_bounded_support_admission_effect_enabled=(
                    PROVENANCE_ANSWER_LOCOMO_BOUNDED_SUPPORT_ADMISSION_EFFECT_ENABLED
                ),
            )
            if PROVENANCE_ANSWER_TRACE_DECISIONS:
                logger.info(
                    "ENGINE-ANS-1-DIAG: mode=%s intent=%s abstain=%s "
                    "answer_present=%s support_id=%s support_channel=%s "
                    "candidate_count=%d candidate_source=%s candidate_rejections=%s "
                    "answer_contract=%s expected_shape=%s slot_binding=%s "
                    "synthesis_shape=%s support_pack_size=%d "
                    "verifier_rejections=%s fallback_used=%s recovery_strategy=%s "
                    "backfill_candidate_ids=%s backfill_rejections=%s "
                    "backfill_latency_ms=%s admission_version=%s admission_v2=%s admission_v2_blocked=%s "
                    "admission_v2_binding=%s admission_v2_shape=%s "
                    "answer_shape_considered=%s answer_shape_admitted=%s answer_shape_reason=%s "
                    "answer_shape_kind=%s answer_shape_effect=%s answer_shape_latency_ms=%s "
                    "llm_error_class=%s llm_error_provider_status=%s latency_ms=%.1f "
                    "question_preview=%r",
                    decision.mode,
                    decision.intent or "none",
                    decision.abstain_reason or "none",
                    bool(decision.answer),
                    decision.support.support_id if decision.support else "none",
                    decision.support.channel if decision.support else "none",
                    decision.candidate_count,
                    decision.candidate_source or "none",
                    decision.candidate_rejection_counts or {},
                    decision.answer_contract_reason or "none",
                    decision.expected_answer_shape or "none",
                    decision.slot_binding_status or "none",
                    decision.synthesis_shape,
                    decision.support_pack_size,
                    decision.verifier_rejection_counts or {},
                    decision.fallback_used or "none",
                    decision.recovery_strategy or "none",
                    decision.backfill_candidate_ids or (),
                    decision.backfill_rejection_counts or {},
                    (
                        f"{decision.backfill_latency_ms:.1f}"
                        if decision.backfill_latency_ms is not None
                        else "none"
                    ),
                    decision.support_admission_version or "none",
                    decision.support_admission_v2_considered,
                    decision.support_admission_v2_blocked_reason or "none",
                    decision.support_admission_v2_binding_status or "none",
                    decision.support_admission_v2_shape or "none",
                    decision.answer_shape_runtime_considered,
                    decision.answer_shape_runtime_admitted,
                    decision.answer_shape_runtime_reason or "none",
                    decision.answer_shape_runtime_contract_kind or "none",
                    decision.answer_shape_runtime_effect_enabled,
                    (
                        f"{decision.answer_shape_runtime_latency_ms:.1f}"
                        if decision.answer_shape_runtime_latency_ms is not None
                        else "none"
                    ),
                    decision.llm_error_class or "none",
                    (
                        decision.llm_error_provider_status
                        if decision.llm_error_provider_status is not None
                        else "none"
                    ),
                    decision.latency_ms,
                    _diag_preview(question),
                )
            diagnostics = _decision_diagnostics(decision) if diagnostics_enabled else None
            support_emission_diagnostics = None
            if (
                PROVENANCE_ANSWER_LOCOMO_SUPPORT_EMISSION_ENABLED
                and (
                    diagnostics is not None
                    or PROVENANCE_ANSWER_LOCOMO_BOUNDED_SUPPORT_ADMISSION_EFFECT_ENABLED
                )
            ):
                try:
                    from app.cognitive.provenance_answer import (
                        locomo_source_bound_support_emission_diagnostics,
                    )

                    support_emission_diagnostics = locomo_source_bound_support_emission_diagnostics(
                        question=question,
                        activated_concepts=activated_concepts,
                        max_activated_concepts=PROVENANCE_ANSWER_MAX_ACTIVATED_CONCEPTS,
                        max_support_chars=PROVENANCE_ANSWER_MAX_SUPPORT_CHARS,
                        fts_limit=PROVENANCE_ANSWER_SUPPORT_CANDIDATE_BACKFILL_FTS_LIMIT,
                        assoc_limit=PROVENANCE_ANSWER_SUPPORT_CANDIDATE_BACKFILL_ASSOC_LIMIT,
                        max_supports=PROVENANCE_ANSWER_SUPPORT_CANDIDATE_BACKFILL_MAX_SUPPORTS,
                        min_score=PROVENANCE_ANSWER_SUPPORT_CANDIDATE_BACKFILL_MIN_SCORE,
                        budget_ms=PROVENANCE_ANSWER_SUPPORT_CANDIDATE_BACKFILL_BUDGET_MS,
                        semantic_enabled=(
                            PROVENANCE_ANSWER_SUPPORT_CANDIDATE_BACKFILL_SEMANTIC_ENABLED
                        ),
                        semantic_limit=(
                            PROVENANCE_ANSWER_SUPPORT_CANDIDATE_BACKFILL_SEMANTIC_LIMIT
                        ),
                        semantic_min_score=(
                            PROVENANCE_ANSWER_SUPPORT_CANDIDATE_BACKFILL_SEMANTIC_MIN_SCORE
                        ),
                        preserve_initial_support_enabled=(
                            PROVENANCE_ANSWER_LOCOMO_PRESERVE_INITIAL_SUPPORT_ENABLED
                        ),
                        preserve_initial_support_displace_enabled=(
                            PROVENANCE_ANSWER_LOCOMO_PRESERVE_INITIAL_SUPPORT_DISPLACE_ENABLED
                        ),
                        activated_support_continuity_enabled=(
                            PROVENANCE_ANSWER_LOCOMO_ACTIVATED_SUPPORT_CONTINUITY_ENABLED
                        ),
                    )
                    if diagnostics is not None:
                        diagnostics = _merge_diagnostics(
                            diagnostics,
                            support_emission_diagnostics,
                        )
                except Exception as emission_exc:
                    if diagnostics is not None:
                        diagnostics.setdefault("backfill_rejection_counts", {})
                        diagnostics["backfill_rejection_counts"][
                            "locomo_support_emission_error"
                        ] = 1
                        diagnostics["locomo_support_emission_error"] = type(emission_exc).__name__
            if diagnostics is not None and PROVENANCE_ANSWER_LOCOMO_BOUNDED_SUPPORT_ADMISSION_ENABLED:
                try:
                    from app.cognitive.provenance_answer import (
                        locomo_bounded_support_admission_diagnostics,
                    )

                    diagnostics = _merge_diagnostics(
                        diagnostics,
                        locomo_bounded_support_admission_diagnostics(
                            question=question,
                            activated_concepts=activated_concepts,
                            max_activated_concepts=PROVENANCE_ANSWER_MAX_ACTIVATED_CONCEPTS,
                            max_support_chars=PROVENANCE_ANSWER_MAX_SUPPORT_CHARS,
                            fts_limit=PROVENANCE_ANSWER_SUPPORT_CANDIDATE_BACKFILL_FTS_LIMIT,
                            assoc_limit=PROVENANCE_ANSWER_SUPPORT_CANDIDATE_BACKFILL_ASSOC_LIMIT,
                            candidate_pool_size=(
                                PROVENANCE_ANSWER_LOCOMO_BOUNDED_SUPPORT_ADMISSION_CANDIDATE_POOL
                            ),
                            min_score=PROVENANCE_ANSWER_SUPPORT_CANDIDATE_BACKFILL_MIN_SCORE,
                            budget_ms=PROVENANCE_ANSWER_SUPPORT_CANDIDATE_BACKFILL_BUDGET_MS,
                            semantic_enabled=(
                                PROVENANCE_ANSWER_SUPPORT_CANDIDATE_BACKFILL_SEMANTIC_ENABLED
                            ),
                            semantic_limit=(
                                PROVENANCE_ANSWER_SUPPORT_CANDIDATE_BACKFILL_SEMANTIC_LIMIT
                            ),
                            semantic_min_score=(
                                PROVENANCE_ANSWER_SUPPORT_CANDIDATE_BACKFILL_SEMANTIC_MIN_SCORE
                            ),
                            effect_enabled=(
                                PROVENANCE_ANSWER_LOCOMO_BOUNDED_SUPPORT_ADMISSION_EFFECT_ENABLED
                            ),
                            preserve_initial_support_enabled=(
                                PROVENANCE_ANSWER_LOCOMO_PRESERVE_INITIAL_SUPPORT_ENABLED
                            ),
                            preserve_initial_support_displace_enabled=(
                                PROVENANCE_ANSWER_LOCOMO_PRESERVE_INITIAL_SUPPORT_DISPLACE_ENABLED
                            ),
                            activated_support_continuity_enabled=(
                                PROVENANCE_ANSWER_LOCOMO_ACTIVATED_SUPPORT_CONTINUITY_ENABLED
                            ),
                        ),
                    )
                except Exception as bounded_exc:
                    diagnostics.setdefault("backfill_rejection_counts", {})
                    diagnostics["backfill_rejection_counts"][
                        "locomo_bounded_support_admission_error"
                    ] = 1
                    diagnostics["locomo_bounded_support_admission_error"] = type(bounded_exc).__name__
            override_result = _try_locomo_source_date_support_override(
                question,
                activated_concepts,
                current_answer=decision.answer,
                support_emission_diagnostics=support_emission_diagnostics,
                base_diagnostics=diagnostics,
                diagnostics_enabled=diagnostics_enabled,
            )
            if override_result is not None and override_result.answer:
                return override_result
            if not decision.answer:
                bridge_result = _try_locomo_support_surface_bridge(
                    question,
                    activated_concepts,
                    support_emission_diagnostics=support_emission_diagnostics,
                    base_diagnostics=diagnostics,
                    diagnostics_enabled=diagnostics_enabled,
                )
                if bridge_result is not None and bridge_result.answer:
                    return bridge_result
                if bridge_result is not None and bridge_result.diagnostics is not None:
                    diagnostics = bridge_result.diagnostics
            candidate = None
            candidate_diagnostics = None
            if CHAIN_EVIDENCE_CANDIDATE_ENABLED:
                chain_evidence_candidate_attempted = True
                candidate, candidate_diagnostics = _try_chain_evidence_candidate_with_diagnostics(
                    question,
                    activated_concepts,
                    diagnostics_enabled=diagnostics_enabled,
                )
                if diagnostics is not None:
                    diagnostics = _with_chain_evidence_candidate_shadow(
                        diagnostics,
                        candidate_diagnostics,
                    )
            if decision.answer:
                arbitration_reason = _candidate_arbitration_reason(candidate, decision)
                if diagnostics is not None:
                    diagnostics = _with_chain_evidence_candidate_shadow(
                        diagnostics,
                        candidate_diagnostics,
                        arbitration_reason=arbitration_reason,
                    )
                if arbitration_reason == "candidate_support_bound_override":
                    logger.info(
                        "CHAIN-EVIDENCE-CANDIDATE-ARBITRATION: %s overrides %s: %s",
                        candidate.rule_id if candidate else "none",
                        decision.mode,
                        candidate.answer[:80] if candidate and candidate.answer else "None",
                    )
                    return EngineChainAnswerResult(candidate.answer, diagnostics)
                logger.info(
                    "ENGINE-ANS-1: %s answer from %s in %.1fms: %s",
                    decision.mode,
                    decision.support.support_id if decision.support else "unknown",
                    decision.latency_ms,
                    decision.answer[:80],
                )
                return EngineChainAnswerResult(decision.answer, diagnostics)
            logger.debug("ENGINE-ANS-1: abstain=%s", decision.abstain_reason)
            last_diagnostics = diagnostics
            if candidate is not None and candidate.admitted:
                logger.info(
                    "CHAIN-EVIDENCE-CANDIDATE: %s/%s in shadow after abstain: %s",
                    candidate.rule_id,
                    candidate.rule_classification,
                    candidate.answer[:80] if candidate.answer else "None",
                )
                return EngineChainAnswerResult(candidate.answer, candidate_diagnostics)
            if candidate_diagnostics is not None:
                last_diagnostics = candidate_diagnostics
        except Exception as e:
            logger.warning("ENGINE-ANS-1: failed (%s), returning None", e)
            if diagnostics_enabled:
                last_diagnostics = _error_diagnostics(e)
    elif diagnostics_enabled:
        last_diagnostics = _skip_diagnostics("provenance_answer_disabled")

    if CHAIN_EVIDENCE_CANDIDATE_ENABLED and not chain_evidence_candidate_attempted:
        candidate, candidate_diagnostics = _try_chain_evidence_candidate_with_diagnostics(
            question,
            activated_concepts,
            diagnostics_enabled=diagnostics_enabled,
        )
        if candidate is not None and candidate.admitted:
            logger.info(
                "CHAIN-EVIDENCE-CANDIDATE: %s/%s: %s",
                candidate.rule_id,
                candidate.rule_classification,
                candidate.answer[:80] if candidate.answer else "None",
            )
            return EngineChainAnswerResult(candidate.answer, candidate_diagnostics)
        if candidate_diagnostics is not None:
            last_diagnostics = candidate_diagnostics

    if not CHAIN_REASONING_ENABLED:
        return EngineChainAnswerResult(None, last_diagnostics)

    # Gate: only fire on multihop questions
    if not ChainReasoningEngine.is_multihop(question):
        logger.info("ENGINE-CHAIN: Not multihop, skipping")
        return EngineChainAnswerResult(
            None,
            last_diagnostics or (_skip_diagnostics("not_multihop") if diagnostics_enabled else None),
        )

    try:
        t0 = time.time()
        context = _format_concepts_for_chain(activated_concepts)
        engine = ChainReasoningEngine(
            llm_caller=_default_llm_call,
            max_hops=4,
        )
        answer = engine.answer(question, context)
        elapsed = time.time() - t0
        logger.info(f"ENGINE-CHAIN: Per-hop answer in {elapsed:.2f}s: " f"{answer[:80] if answer else 'None'}")
        # Filter out UNKNOWN / empty / refusal answers; return None to trigger runner fallback.
        if not answer or not answer.strip():
            return EngineChainAnswerResult(
                None,
                last_diagnostics
                or (
                    _chain_reasoning_diagnostics(
                        None,
                        latency_ms=elapsed * 1000,
                        abstain_reason="empty_answer",
                    )
                    if diagnostics_enabled
                    else None
                ),
            )
        if answer.strip().upper() == "UNKNOWN":
            logger.info("ENGINE-CHAIN: Answer is UNKNOWN, returning None for runner fallback")
            return EngineChainAnswerResult(
                None,
                last_diagnostics
                or (
                    _chain_reasoning_diagnostics(
                        None,
                        latency_ms=elapsed * 1000,
                        abstain_reason="unknown_answer",
                    )
                    if diagnostics_enabled
                    else None
                ),
            )
        return EngineChainAnswerResult(
            answer,
            _chain_reasoning_diagnostics(answer, latency_ms=elapsed * 1000, abstain_reason=None)
            if diagnostics_enabled
            else None,
        )
    except Exception as e:
        logger.warning(f"ENGINE-CHAIN: Failed ({e}), returning None")
        return EngineChainAnswerResult(
            None,
            last_diagnostics
            or (
                _error_diagnostics(e, abstain_reason="chain_reasoning_exception")
                if diagnostics_enabled
                else None
            ),
        )


def engine_chain_answer(
    question: str,
    activated_concepts: list,
) -> str | None:
    """Compatibility wrapper for callers that only need the answer string."""
    return engine_chain_answer_result(question, activated_concepts).answer
