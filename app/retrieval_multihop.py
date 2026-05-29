"""Production Multi-Hop Retriever for conversation_turn S2.

RETRIEVAL-037a: Ported from benchmarks/adapter/multihop.py.
Key differences from benchmark version:
- Calls search_lightweight directly (no HTTP adapter wrapper)
- Has latency budget enforcement (PITH_MULTIHOP_BUDGET_MS)
- Integrated with production logging for observability
- A2 fallback on any exception

Feature-flagged via PITH_MULTIHOP_ENABLED.

Spec: RETRIEVAL_037 sprint design notes.
Source: benchmarks/adapter/multihop.py (HybridMultiHopRetriever)
"""

import re
import os
import time
import logging
from typing import Optional

log = logging.getLogger("pith.retrieval_multihop")

# RETRIEVAL-041: Two-tier hop boundary patterns.
# Tier 1 (strong): relative clauses and participial phrases — reliable boundaries
#   with no false positives inside noun phrases.
# Tier 2 (weak): possessive noun chains — used only when Tier 1 yields <2 clauses.
# Gate: union of both tiers for is_multihop_question() complexity gate.
_HOP_STRONG = re.compile(
    r'\b(?:where the|in which|to which|from which|attended by)\b',
    re.IGNORECASE,
)
_HOP_WEAK = re.compile(
    r'\b(?:of the|for the)\b',
    re.IGNORECASE,
)
_HOP_GATE = re.compile(
    r'\b(?:where the|in which|to which|from which|attended by|of the|for the)\b',
    re.IGNORECASE,
)


class ProductionMultiHopRetriever:
    """Multi-hop retrieval for complex questions in conversation_turn.

    Adapted from benchmarks/adapter/multihop.py HybridMultiHopRetriever.
    Key differences from benchmark version:
    - Calls search_lightweight directly (no HTTP adapter wrapper)
    - Has latency budget enforcement (PITH_MULTIHOP_BUDGET_MS)
    - A2 fallback on any exception
    """

    def __init__(
        self,
        retrieval_engine,
        max_hops: int = 3,
        per_hop_k: int = 10,
        min_relevance: float = 0.15,
        budget_ms: float = 150.0,
        clause_order: str = "reverse",
    ):
        self.engine = retrieval_engine
        self.max_hops = max_hops
        self.per_hop_k = per_hop_k
        self.min_relevance = min_relevance
        self.budget_ms = budget_ms
        self.clause_order = clause_order

        # Metrics (per-instance, reset on server restart)
        self.total_queries = 0
        self.total_fallbacks = 0
        self.total_hops_executed = 0
        self.total_hops_skipped = 0

    # ------------------------------------------------------------------
    # Complexity gate (static — usable before instantiation)
    # ------------------------------------------------------------------

    @classmethod
    def is_multihop_question(cls, query: str) -> bool:
        """Complexity gate: returns True if query has relational phrases."""
        return bool(_HOP_GATE.search(query))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self, query: str, top_k: int,
        agent_id: Optional[str] = None, scope: str = "global",
    ) -> list:
        """Main entry point. Decomposes query, retrieves per-hop, deduplicates.

        Returns list of SearchResult objects (same type as search_lightweight).
        On ANY exception, falls back to standard single-pass retrieval (A2).
        Also stores decomposed_clauses on self for chain hint generation.
        """
        self.total_queries += 1
        self.decomposed_clauses = []  # RETRIEVAL-037d: expose for chain hints
        self.hop_context_summaries = []  # RETRIEVAL-047: expose for entity extraction
        try:
            return self._retrieve_hybrid(query, top_k, agent_id, scope)
        except Exception as e:
            self.total_fallbacks += 1
            log.warning(f"MULTIHOP-FALLBACK: {e}")
            return self.engine.search_lightweight(
                query, top_k=top_k, min_confidence=0.0,
                agent_id=agent_id, scope=scope,
            )

    # ------------------------------------------------------------------
    # Query decomposition
    # ------------------------------------------------------------------

    def _decompose_smart(self, question: str) -> list[str]:
        """Two-pass query decomposer: strong hop patterns first, weak fallback.

        RETRIEVAL-041: Fixes Q30 (4 garbled clauses from double "of the" split)
        and Q70 (under-split missing "attended by" boundary) regressions from
        RETRIEVAL-040.

        Tier 1 (strong): relative clauses + participial phrases. These reliably
        mark semantic hop boundaries without splitting inside noun phrases. Use
        if they yield >=2 clauses.
        Tier 2 (weak): possessive noun chains ("of the", "for the"). Fallback
        only when Tier 1 finds no boundary. Avoids the false-positive splits
        that fragmented Q30/Q70.
        """
        q = question.strip()
        # Tier 1: strong patterns (relative clauses, participial phrases)
        parts = _HOP_STRONG.split(q)
        clauses = [p.strip().strip('?').strip() for p in parts if len(p.strip()) > 5]
        if len(clauses) >= 2:
            return clauses
        # Tier 2: weak fallback (possessive noun chains)
        parts = _HOP_WEAK.split(q)
        return [p.strip().strip('?').strip() for p in parts if len(p.strip()) > 5]

    # ------------------------------------------------------------------
    # Core hybrid pipeline
    # ------------------------------------------------------------------

    def _retrieve_hybrid(
        self, question: str, top_k: int,
        agent_id: Optional[str], scope: str,
    ) -> list:
        """Decompose query into clauses, then iteratively retrieve."""
        t0 = time.perf_counter()

        # Step 1: Decompose
        clauses = self._decompose_smart(question)
        self.decomposed_clauses = clauses  # RETRIEVAL-037d

        # Fall-through: if not decomposable, use standard retrieval
        if len(clauses) <= 1:
            log.info(
                f"MULTIHOP-PASSTHROUGH: Query not decomposable "
                f"({len(clauses)} clause), using standard retrieval"
            )
            return self.engine.search_lightweight(
                question, top_k=top_k, min_confidence=0.0,
                agent_id=agent_id, scope=scope,
            )

        log.info(f"MULTIHOP-DECOMPOSE: {len(clauses)} clauses, order={self.clause_order}")

        # Step 2: Iterative context-seeded retrieval
        all_results = self._retrieve_iterative(
            question, clauses, top_k, agent_id, scope, t0,
        )

        log.info(f"MULTIHOP-RESULT: {len(all_results)} concepts from {len(clauses)} hops")
        return all_results

    # ------------------------------------------------------------------
    # Iterative retrieval with budget enforcement
    # ------------------------------------------------------------------

    def _retrieve_iterative(
        self, original_question: str, clauses: list[str],
        top_k: int, agent_id: Optional[str], scope: str,
        t0: float,
    ) -> list:
        """For each clause, retrieve concepts seeded by previous hop context.

        Budget enforcement: if elapsed time exceeds budget_ms, stop iteration
        and return what we have (graceful degradation, not failure).
        """
        # Order clauses per config
        ordered = list(reversed(clauses)) if self.clause_order == "reverse" else list(clauses)
        ordered = ordered[:self.max_hops]  # Cap at max_hops

        accumulated_context = ""
        seen: dict[str, object] = {}  # concept_id -> best SearchResult (A5 dedup)

        for i, clause in enumerate(ordered):
            hop_index = i + 1
            self.total_hops_executed += 1

            # Budget check
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if elapsed_ms > self.budget_ms:
                log.warning(
                    f"MULTIHOP-BUDGET: Stopping at hop {hop_index}/{len(ordered)} "
                    f"({elapsed_ms:.1f}ms > {self.budget_ms}ms budget)"
                )
                break

            # Build combined query: clause + accumulated context
            if accumulated_context:
                combined_q = f"{clause} {accumulated_context}"
            else:
                combined_q = clause

            # Retrieve for this hop
            hop_results = self.engine.search_lightweight(
                combined_q, top_k=self.per_hop_k, min_confidence=0.0,
                agent_id=agent_id, scope=scope,
            )

            # A6: Context pollution guard
            above_threshold = [
                r for r in hop_results
                if r.relevance_score >= self.min_relevance
            ]
            if not above_threshold:
                self.total_hops_skipped += 1
                log.info(
                    f"MULTIHOP-SKIP: Hop {hop_index} skipped — "
                    f"0/{len(hop_results)} above min_relevance={self.min_relevance}"
                )
                continue

            # A5: Dedup — keep highest relevance per concept_id
            for r in above_threshold:
                existing = seen.get(r.concept_id)
                if existing is None or r.relevance_score > existing.relevance_score:
                    seen[r.concept_id] = r

            # Update accumulated context with top result summary
            if above_threshold:
                accumulated_context = above_threshold[0].summary
                self.hop_context_summaries.append(accumulated_context)  # RETRIEVAL-047

            log.debug(
                f"  hop {hop_index}: {len(above_threshold)} concepts above threshold, "
                f"context seed: {accumulated_context[:60]}..."
            )

        # Final pass: original query + accumulated context as anchor
        if accumulated_context:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if elapsed_ms <= self.budget_ms:
                final_results = self.engine.search_lightweight(
                    f"{original_question} {accumulated_context}",
                    top_k=self.per_hop_k, min_confidence=0.0,
                    agent_id=agent_id, scope=scope,
                )
                for r in final_results:
                    existing = seen.get(r.concept_id)
                    if existing is None or r.relevance_score > existing.relevance_score:
                        seen[r.concept_id] = r

        # Build final list sorted by relevance
        # RETRIEVAL-037b v4.2: Deterministic tiebreaker — concept_id breaks ties
        # when multiple concepts share identical governance scores, making the
        # budget cutoff stable across server restarts.
        result = sorted(seen.values(), key=lambda r: (-r.relevance_score, r.concept_id))
        return result[:top_k]  # Cap to requested top_k

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict:
        """Return retrieval metrics for logging."""
        return {
            "total_queries": self.total_queries,
            "total_fallbacks": self.total_fallbacks,
            "total_hops_executed": self.total_hops_executed,
            "total_hops_skipped": self.total_hops_skipped,
            "fallback_rate": (
                self.total_fallbacks / max(self.total_queries, 1) * 100
            ),
        }

    # ------------------------------------------------------------------
    # RETRIEVAL-040: Per-hop concept scoring
    # ------------------------------------------------------------------

    @staticmethod
    def score_concepts_per_hop(
        clauses: list[str],
        activated_concepts: list,
        min_similarity: float = 0.25,
        max_pool_size: int = 0,
    ) -> dict[int, list[tuple[str, str, float]]]:
        """Score each activated concept against each decomposed clause.

        Returns {step_num: [(concept_id, snippet, score), ...]} where
        step_num is 1-indexed and hits are sorted by descending score,
        capped at 5 per step.

        If max_pool_size > 0 and len(activated_concepts) > max_pool_size,
        pre-filters to top max_pool_size concepts by relevance_score before
        embedding. This prevents embedding model degradation at large pool
        sizes (see RETRIEVAL-033).
        """
        from app.storage.embedding import embedding_engine
        import numpy as np

        if not clauses or not activated_concepts:
            return {}

        # Pre-filter to top-K by retriever relevance if pool is too large
        pool = activated_concepts
        if max_pool_size > 0 and len(pool) > max_pool_size:
            pool = sorted(
                pool,
                key=lambda c: getattr(c, 'relevance_score', 0) or 0,
                reverse=True,
            )[:max_pool_size]

        # Collect concept summaries
        summaries: list[str] = []
        concept_ids: list[str] = []
        for ac in pool:
            s = getattr(ac, 'summary', '') or ''
            if len(s) > 10:
                summaries.append(s[:500])
                concept_ids.append(getattr(ac, 'concept_id', ''))
        if not summaries:
            return {}

        # Embed clauses and concepts
        try:
            clause_vecs = embedding_engine.embed_batch(clauses)
        except Exception:
            return _keyword_hop_scoring(clauses, activated_concepts, min_similarity)

        try:
            concept_vecs = embedding_engine.embed_batch(summaries)
        except Exception:
            return {}

        # Cosine similarity matrix (clause_vecs and concept_vecs are L2-normed)
        sim_matrix = clause_vecs @ concept_vecs.T

        result: dict[int, list[tuple[str, str, float]]] = {}
        for step_idx in range(len(clauses)):
            step_num = step_idx + 1
            scores = sim_matrix[step_idx]
            step_hits: list[tuple[str, str, float]] = []
            for j in range(len(concept_ids)):
                if scores[j] >= min_similarity:
                    snippet = summaries[j][:120].replace('\n', ' ')
                    step_hits.append((concept_ids[j], snippet, float(scores[j])))
            step_hits.sort(key=lambda x: x[2], reverse=True)
            if step_hits:
                result[step_num] = step_hits[:5]

        return result


def _keyword_hop_scoring(
    clauses: list[str],
    activated_concepts: list,
    min_similarity: float = 0.25,
) -> dict[int, list[tuple[str, str, float]]]:
    """Keyword-overlap fallback when embedding engine is unavailable."""
    result: dict[int, list[tuple[str, str, float]]] = {}
    for step_idx, clause in enumerate(clauses):
        step_num = step_idx + 1
        clause_words = set(clause.lower().split())
        hits: list[tuple[str, str, float]] = []
        for ac in activated_concepts:
            s = getattr(ac, 'summary', '') or ''
            if len(s) < 10:
                continue
            concept_words = set(s.lower().split())
            overlap = len(clause_words & concept_words)
            if overlap >= 2:
                score = overlap / max(len(clause_words), 1)
                if score >= min_similarity:
                    snippet = s[:120].replace('\n', ' ')
                    hits.append((getattr(ac, 'concept_id', ''), snippet, score))
        hits.sort(key=lambda x: x[2], reverse=True)
        if hits:
            result[step_num] = hits[:5]
    return result
