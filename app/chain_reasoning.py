"""Chain-of-Thought Decomposition for Multi-Hop Questions.

RETRIEVAL-040-LLM: LLM-based decomposition for multi-hop queries.
Ported from benchmarks/adapter/chain_reasoning.py (pith-internal).
Feature-gated: PITH_LLM_CHAIN_REASONING=true (benchmark mode only).

Instead of giving the LLM 25 facts + 1 complex question, we:
1. Decompose into single-hop sub-questions via LLM
2. Answer each step against the FULL context
3. Substitute intermediate answers and continue the chain
4. Final answer = result of last step
"""

import re
import os
import logging
import time
from typing import Optional, Callable

logger = logging.getLogger(__name__)

CHAIN_REASONING_ENABLED = os.environ.get(
    "PITH_LLM_CHAIN_REASONING", ""
).lower() in ("true", "1")

# Reuse the gate pattern from retrieval_multihop.py
_HOP_GATE = re.compile(
    r'\b(?:where the|in which|to which|from which|attended by|of the|for the)\b',
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
                resolved_q = resolved_q.replace(
                    f"[RESULT_{prev_step}]", prev_answer
                )
            hop_answer = self._hop_answer(resolved_q, context)
            results[step_num] = hop_answer
            logger.info(f"  Step {step_num}: {resolved_q[:80]} -> {hop_answer}")
            if hop_answer.upper() == "UNKNOWN" or not hop_answer.strip():
                logger.warning(
                    f"CHAIN-REASON: Step {step_num} returned UNKNOWN, "
                    f"falling back to direct"
                )
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
            q = q[idx + len(marker):].strip()
        q = re.sub(r'\s*Answer:\s*$', '', q).strip()

        prompt = _DECOMPOSE_PROMPT.format(question=q)
        response = self.llm(
            prompt, system_msg="You decompose questions into single-hop steps."
        )
        steps = []
        for line in response.strip().split('\n'):
            m = re.match(r'STEP\s+\d+:\s*(.+)', line.strip(), re.I)
            if m:
                steps.append(m.group(1).strip())
        return steps[:self.max_hops]

    def _hop_answer(self, question: str, context: str) -> str:
        prompt = _HOP_ANSWER_PROMPT.format(context=context, question=question)
        answer = self.llm(prompt, system_msg=None)
        return answer.strip().strip('"').strip("'").strip('.')

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
) -> str:
    """Direct OpenAI-compatible LLM call for decomposition."""
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
        timeout=15,
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
            q = q[idx + len(marker):].strip()
        q = re.sub(r'\s*Answer:\s*$', '', q).strip()

        prompt = _DECOMPOSE_PROMPT.format(question=q)
        response = _default_llm_call(
            prompt,
            system_msg="You decompose questions into single-hop steps.",
        )

        steps = []
        for line in response.strip().split('\n'):
            m = re.match(r'STEP\s+\d+:\s*(.+)', line.strip(), re.I)
            if m:
                steps.append(m.group(1).strip())

        if len(steps) >= 2:
            logger.info(
                f"LLM-DECOMPOSE: Produced {len(steps)} clauses "
                f"from question: {q[:80]}"
            )
            return steps[:4]
        else:
            logger.info(
                f"LLM-DECOMPOSE: Only {len(steps)} clause(s), "
                f"not decomposable"
            )
            return []

    except Exception as e:
        logger.warning(f"LLM-DECOMPOSE: Failed ({e}), returning empty")
        return []



# ---------------------------------------------------------------------------
# Engine-side per-hop answering (C1 gap fix)
# ---------------------------------------------------------------------------

_TAG_STRIP_RE = re.compile(
    r'\[(?:CRITICAL-CONTEXT|ALWAYS|FIRMWARE|PRINCIPLE|CONSTRAINT|'
    r'/CRITICAL-CONTEXT)(?:\s+[^\]]*?)?\]\s*',
    re.IGNORECASE
)


def _format_concepts_for_chain(concepts: list) -> str:
    """Format ActivatedConcept list into numbered context string.

    Mirrors the runner's format_concepts_as_context() but operates on
    ActivatedConcept model objects instead of dicts.
    Includes [serial=N] tags for temporal conflict resolution.
    """
    lines = []
    for i, c in enumerate(concepts, 1):
        summary = c.summary if isinstance(c, dict) else getattr(c, 'summary', '')
        summary = _TAG_STRIP_RE.sub('', summary).strip()
        if not summary:
            continue

        serial = None
        if isinstance(c, dict):
            serial = c.get('serial_order')
            ka = c.get('knowledge_area', 'events')
        else:
            serial = getattr(c, 'serial_order', None)
            ka = getattr(c, 'knowledge_area', 'events')

        if serial is not None and serial > 0:
            lines.append(f"[{ka}] [serial={serial}] {summary}")
        else:
            lines.append(f"[{i}] {summary}")
    return "\n".join(lines)


def engine_chain_answer(
    question: str,
    activated_concepts: list,
) -> str | None:
    """Engine-side per-hop chain answering (C1 gap fix).

    Called from session.py conversation_turn AFTER building activated_concepts.
    Feature-gated by PITH_LLM_CHAIN_REASONING env var.

    Returns:
        Answer string if chain reasoning succeeds, None otherwise.
        Caller should set response.chain_answer = result.
    """
    if not CHAIN_REASONING_ENABLED:
        return None

    if not activated_concepts:
        return None

    # Gate: only fire on multihop questions
    if not ChainReasoningEngine.is_multihop(question):
        logger.info("ENGINE-CHAIN: Not multihop, skipping")
        return None

    try:
        t0 = time.time()
        context = _format_concepts_for_chain(activated_concepts)
        engine = ChainReasoningEngine(
            llm_caller=_default_llm_call,
            max_hops=4,
        )
        answer = engine.answer(question, context)
        elapsed = time.time() - t0
        logger.info(
            f"ENGINE-CHAIN: Per-hop answer in {elapsed:.2f}s: "
            f"{answer[:80] if answer else 'None'}"
        )
        # Filter out UNKNOWN / empty / refusal answers — return None to trigger runner fallback
        if not answer or not answer.strip():
            return None
        if answer.strip().upper() == "UNKNOWN":
            logger.info("ENGINE-CHAIN: Answer is UNKNOWN, returning None for runner fallback")
            return None
        return answer
    except Exception as e:
        logger.warning(f"ENGINE-CHAIN: Failed ({e}), returning None")
        return None
