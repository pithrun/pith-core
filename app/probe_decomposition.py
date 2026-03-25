"""
Multi-probe query decomposition for SAL Mode A and Mode C.

Decomposes a query into K aspect probes, retrieves concepts per-probe,
and applies quality gates before passing to graph attention.

Config reads from SAL_* constants in app.config (env var overridable).
"""

import logging
from typing import Any

from app.config import (
    SAL_PROBE_COUNT,
    SAL_PROBE_OVERLAP_THRESHOLD,
    SAL_PROBE_EMPTY_MAX_RATIO,
    SAL_PROBE_MIN_QUALITY,
    SAL_MAX_ACTIVATION_SIZE,
)

logger = logging.getLogger(__name__)


def decompose_query(
    query: str,
    probe_count: int | None = None,
    strategy: str = "llm_decompose",
) -> list[str]:
    """
    Decompose a query into K aspect probes.

    V0 strategy: simple keyword/phrase extraction.
    Future: LLM-based decomposition for richer aspects.
    """
    k = probe_count if probe_count is not None else SAL_PROBE_COUNT

    if strategy == "llm_decompose":
        # V0 placeholder: split into noun phrases / key terms
        # In V1, this calls an LLM to decompose the query into aspects
        words = query.split()
        if len(words) <= k:
            return [query]  # Query too short to decompose

        # Chunk the query into roughly equal aspect phrases
        chunk_size = max(1, len(words) // k)
        aspects = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                aspects.append(chunk.strip())
        return aspects[:k]

    elif strategy == "taxonomy":
        # V0: return the query as-is (taxonomy probes require KA mapping)
        return [query]

    else:
        return [query]


def decompose_and_retrieve(
    query: str,
    enriched_concepts: list[dict],
) -> list[dict]:
    """
    Decompose query into aspects and filter enriched concepts per-aspect.

    Returns: [{"aspect": str, "concepts": [enriched_concept_dicts]}]
    """
    aspects = decompose_query(query, SAL_PROBE_COUNT)

    results = []
    for aspect in aspects:
        # Filter concepts by relevance to this aspect
        # V0: simple keyword matching against concept summaries
        aspect_lower = aspect.lower()
        aspect_words = set(aspect_lower.split())

        scored = []
        for concept in enriched_concepts:
            summary = concept.get("summary", "").lower()
            summary_words = set(summary.split())
            overlap = len(aspect_words & summary_words)
            if overlap > 0:
                scored.append((concept, overlap))

        # Sort by overlap score, take top concepts per probe
        scored.sort(key=lambda x: x[1], reverse=True)
        max_per_probe = SAL_MAX_ACTIVATION_SIZE // SAL_PROBE_COUNT
        probe_concepts = [c for c, _ in scored[:max_per_probe]]

        results.append({
            "aspect": aspect,
            "concepts": probe_concepts,
        })

    return results


def check_probe_quality(
    probe_results: list[dict],
) -> dict[str, Any]:
    """
    Quality gate for probe decomposition.
    Reads thresholds from SAL_* config constants.

    Returns: {"passed": bool, "reason": str, "score": float}
    """
    total_probes = len(probe_results)
    if total_probes == 0:
        return {"passed": False, "reason": "no_probes", "score": 0.0}

    # Check empty probes
    empty_count = sum(1 for pr in probe_results if len(pr["concepts"]) == 0)
    empty_ratio = empty_count / total_probes
    if empty_ratio > SAL_PROBE_EMPTY_MAX_RATIO:
        return {
            "passed": False,
            "reason": f"empty_probes:{empty_count}/{total_probes}",
            "score": 0.0,
        }

    # Check overlap between probes
    non_empty = [pr for pr in probe_results if pr["concepts"]]
    if len(non_empty) < 2:
        return {"passed": True, "reason": "single_probe", "score": 1.0}

    # Compute pairwise overlap (Jaccard)
    total_overlap = 0.0
    pair_count = 0
    for i in range(len(non_empty)):
        ids_i = {c["concept_id"] for c in non_empty[i]["concepts"]}
        for j in range(i + 1, len(non_empty)):
            ids_j = {c["concept_id"] for c in non_empty[j]["concepts"]}
            union = ids_i | ids_j
            intersection = ids_i & ids_j
            if union:
                total_overlap += len(intersection) / len(union)
            pair_count += 1

    avg_overlap = total_overlap / max(pair_count, 1)
    quality_score = 1.0 - avg_overlap

    if avg_overlap > SAL_PROBE_OVERLAP_THRESHOLD:
        return {
            "passed": False,
            "reason": f"high_overlap:{avg_overlap:.2f}",
            "score": quality_score,
        }

    if quality_score < SAL_PROBE_MIN_QUALITY:
        return {
            "passed": False,
            "reason": f"low_quality:{quality_score:.2f}",
            "score": quality_score,
        }

    return {"passed": True, "reason": "ok", "score": quality_score}
