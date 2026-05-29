"""
Structured Activation Layer V0 — Heuristic Graph Attention

Entry point: process_sal() — called from session.py conversation_turn.
Works with ActivatedConcept objects + _concept_cache for enrichment.
Uses get_adjacency_graph() (already cached in storage.py) for edges.

Three operating modes:
  A: multi_probe — query decomposition into aspect probes
  B: graph_attention — subgraph + attention + pooling
  C: multi_probe_then_attention — A then B on unified result (recommended)
"""

import math
import time
import logging
from typing import Any
from datetime import datetime, timezone

from app.core.config import (
    FEATURE_FLAGS, SAL_MODE, SAL_MIN_ACTIVATION_SIZE, SAL_MAX_ACTIVATION_SIZE,
    SAL_MAX_ASSOC_PER_CONCEPT, SAL_SIMILARITY_EXPONENT, SAL_ASSOCIATION_EXPONENT,
    SAL_CONFIDENCE_FLOOR, SAL_TEMPORAL_ENABLED, SAL_HALFLIFE_OBSERVATION,
    SAL_HALFLIFE_PATTERN, SAL_HALFLIFE_HEURISTIC, SAL_HALFLIFE_DECISION,
    SAL_HALFLIFE_PRINCIPLE, SAL_SURPRISE_BUFFER_ENABLED, SAL_SURPRISE_RELEVANCE_FLOOR,
    SAL_SURPRISE_CONN_CEILING, SAL_EXPLORATION_ENABLED, SAL_EXPLORATION_RATE,
    SAL_MONOPOLE_THRESHOLD, SAL_UNIFORMITY_THRESHOLD,
)

logger = logging.getLogger(__name__)

# Halflife lookup from config constants
_HALFLIFE_MAP = {
    "observation": SAL_HALFLIFE_OBSERVATION,
    "pattern": SAL_HALFLIFE_PATTERN,
    "heuristic": SAL_HALFLIFE_HEURISTIC,
    "cognitive_strategy": SAL_HALFLIFE_HEURISTIC,
    "decision": SAL_HALFLIFE_DECISION,
    "principle": SAL_HALFLIFE_PRINCIPLE,
    "method": SAL_HALFLIFE_PRINCIPLE,
}


# ─── Public Entry Point ─────────────────────────────────────────

def process_sal(
    activated_concepts: list,  # list[ActivatedConcept]
    concept_cache: dict,       # concept_id → Concept (full objects with concept_type, created_at)
    query: str,
    adjacency_graph: dict,     # From get_adjacency_graph(): concept_id → {neighbor: strength}
) -> dict | None:
    """
    Main SAL entry point. Called from session.py conversation_turn.

    Returns dict with SAL results, or None on failure.
    Falls back to None (flat retrieval) on any error — never worse than baseline.
    """
    start_time = time.perf_counter()
    mode = SAL_MODE
    n = len(activated_concepts)

    # Below minimum — not enough for meaningful attention
    if n < SAL_MIN_ACTIVATION_SIZE:
        return _flat_result(mode, start_time, "below_min_activation")

    # Pre-filter if too large
    if n > SAL_MAX_ACTIVATION_SIZE:
        activated_concepts = sorted(
            activated_concepts,
            key=lambda c: c.relevance_score,
            reverse=True,
        )[:SAL_MAX_ACTIVATION_SIZE]
        n = SAL_MAX_ACTIVATION_SIZE

    # Enrich ActivatedConcepts with data from concept_cache
    enriched = _enrich_concepts(activated_concepts, concept_cache)

    try:
        if mode == "multi_probe":
            return _mode_a(enriched, query, start_time)
        elif mode == "graph_attention":
            return _mode_b(enriched, query, adjacency_graph, start_time)
        elif mode == "multi_probe_then_attention":
            return _mode_c(enriched, query, adjacency_graph, start_time)
        else:
            logger.error(f"SAL: Unknown mode '{mode}'")
            return _flat_result(mode, start_time, f"unknown_mode:{mode}")
    except Exception as e:
        logger.error(f"SAL: Processing error: {e}", exc_info=True)
        return _flat_result(mode, start_time, f"error:{type(e).__name__}")


def _enrich_concepts(activated: list, concept_cache: dict) -> list[dict]:
    """
    Convert ActivatedConcept objects to enriched dicts with concept_type and created_at.
    These fields are on the full Concept model but NOT on ActivatedConcept.
    """
    enriched = []
    for ac in activated:
        d = {
            "concept_id": ac.concept_id,
            "summary": ac.summary,
            "confidence": ac.confidence,
            "relevance_score": ac.relevance_score,
            "knowledge_area": ac.knowledge_area,
            "associations_ids": ac.associations,  # 1-hop neighbor IDs
        }
        # Enrich from concept_cache
        full_concept = concept_cache.get(ac.concept_id)
        if full_concept:
            d["concept_type"] = getattr(full_concept, "concept_type", "observation")
            d["created_at"] = getattr(full_concept, "created_at", None)
        else:
            d["concept_type"] = "observation"  # Safe default
            d["created_at"] = None
        enriched.append(d)
    return enriched


# ─── Temporal Weighting ─────────────────────────────────────────

def compute_temporal_weight(
    concept_type: str,
    created_at: str | None,
    query_scope: str,
) -> float:
    """
    Temporal dual-signal: validity_decay × temporal_relevance.
    created_at is ISO string (pith convention), not datetime.
    """
    if not created_at:
        return 1.0  # No temporal data → no penalty

    now = datetime.now(timezone.utc)
    try:
        if isinstance(created_at, str):
            ca = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            if ca.tzinfo is None:
                ca = ca.replace(tzinfo=timezone.utc)
        else:
            ca = created_at
        age_hours = max((now - ca).total_seconds() / 3600, 0.001)
    except (ValueError, TypeError):
        return 1.0

    halflife = _HALFLIFE_MAP.get(concept_type, SAL_HALFLIFE_PATTERN)
    validity_decay = math.exp(-0.693 * age_hours / halflife)

    if query_scope == "narrow":
        temporal_relevance = math.exp(-0.693 * age_hours / 48)
    elif query_scope == "wide":
        temporal_relevance = 1.0
    else:
        temporal_relevance = math.exp(-0.693 * age_hours / 720)

    return validity_decay * temporal_relevance


def infer_query_scope(query: str) -> str:
    """Infer temporal scope from query text."""
    narrow_signals = ["current", "today", "this week", "recent", "latest", "now", "just"]
    wide_signals = ["always", "principle", "our approach", "historically", "generally", "overall"]
    query_lower = query.lower()
    narrow_count = sum(1 for s in narrow_signals if s in query_lower)
    wide_count = sum(1 for s in wide_signals if s in query_lower)
    if narrow_count > wide_count:
        return "narrow"
    elif wide_count > narrow_count:
        return "wide"
    return "neutral"


# ─── Heuristic Interaction Scoring ──────────────────────────────

def compute_interaction_score(
    semantic_similarity: float,
    association_strength: float,
    mutual_relevance: float,
    confidence_a: float,
    confidence_b: float,
) -> float:
    """
    V0 heuristic: (semantic_compat^S) × (assoc_weight^A) × mutual_relevance × confidence_product
    Exponents read from config constants.
    """
    semantic_term = max(semantic_similarity, 0.0) ** SAL_SIMILARITY_EXPONENT
    assoc_term = max(association_strength, 0.001) ** SAL_ASSOCIATION_EXPONENT
    return semantic_term * assoc_term * mutual_relevance * (confidence_a * confidence_b)


# ─── Subgraph Extraction (uses get_adjacency_graph) ─────────────

def extract_subgraph(
    enriched_concepts: list[dict],
    adjacency_graph: dict[str, dict[str, float]],
) -> dict[tuple, float]:
    """
    Extract intra-activation-set edges from pre-built adjacency graph.
    Applies per-concept ceiling (SAL_MAX_ASSOC_PER_CONCEPT).
    Uses median weight floor per B2 validation.

    Returns: {(concept_a_id, concept_b_id): strength}
    """
    concept_ids = {c["concept_id"] for c in enriched_concepts}
    edges: dict[tuple, float] = {}

    for c in enriched_concepts:
        cid = c["concept_id"]
        neighbors = adjacency_graph.get(cid, {})

        # Filter to intra-activation-set neighbors
        relevant = [(nid, strength) for nid, strength in neighbors.items() if nid in concept_ids]

        # Median weight floor (B2 safeguard)
        if relevant:
            strengths = sorted([s for _, s in relevant])
            median_s = strengths[len(strengths) // 2]
            relevant = [(nid, s) for nid, s in relevant if s >= median_s]

        # Per-concept ceiling
        relevant = sorted(relevant, key=lambda x: x[1], reverse=True)[:SAL_MAX_ASSOC_PER_CONCEPT]

        for nid, strength in relevant:
            edge_key = tuple(sorted([cid, nid]))
            if edge_key not in edges:
                edges[edge_key] = strength

    return edges


# ─── Embedding Helper ─────────────────────────────────────────

def _get_embeddings(enriched_concepts: list[dict]) -> dict[str, list[float]]:
    """Batch-embed concept summaries for pairwise semantic similarity."""
    try:
        from app.storage.embedding import embedding_engine  # [A1] Class singleton, not module fn
        summaries = [c["summary"] for c in enriched_concepts]
        if not summaries:
            return {}
        vectors = embedding_engine.embed_batch(summaries)  # np.ndarray(N, 384), L2-normalized
        return {
            c["concept_id"]: vectors[i].tolist()
            for i, c in enumerate(enriched_concepts)
        }
    except Exception as e:
        logger.warning(f"SAL: Embedding batch failed: {e}")
        return {}


# ─── Graph Attention ──────────────────────────────────────────

def compute_graph_attention(
    enriched_concepts: list[dict],
    subgraph_edges: dict[tuple, float],
    query: str,
    embeddings: dict[str, list[float]],
) -> dict:
    """
    Heuristic graph attention over activated concept subgraph.

    Steps:
    1. Compute pairwise interaction scores
    2. Apply temporal weighting
    3. Normalize to attention weights
    4. Detect degeneracy
    5. Build clusters
    6. Extract surprise buffer

    Returns dict with: attention_weights, clusters, confidence_envelope,
    degeneracy_detected, degeneracy_type, surprise_buffer, metrics.
    """
    n = len(enriched_concepts)
    concept_map = {c["concept_id"]: c for c in enriched_concepts}
    concept_ids = list(concept_map.keys())
    query_scope = infer_query_scope(query)

    # Step 1: Pairwise interaction scores
    raw_scores = {}
    for i, cid_a in enumerate(concept_ids):
        for j in range(i + 1, len(concept_ids)):
            cid_b = concept_ids[j]
            ca, cb = concept_map[cid_a], concept_map[cid_b]

            # Semantic similarity (cosine of embeddings)
            emb_a = embeddings.get(cid_a, [])
            emb_b = embeddings.get(cid_b, [])
            sem_sim = _cosine_similarity(emb_a, emb_b)

            # Association weight from subgraph (0 if no direct edge)
            edge_key = tuple(sorted([cid_a, cid_b]))
            assoc_w = subgraph_edges.get(edge_key, 0.0)

            # Mutual relevance (geometric mean)
            mutual_rel = math.sqrt(
                ca.get("relevance_score", 0.5) * cb.get("relevance_score", 0.5)
            )

            score = compute_interaction_score(
                sem_sim, assoc_w, mutual_rel,
                ca.get("confidence", 0.5), cb.get("confidence", 0.5),
            )
            raw_scores[(cid_a, cid_b)] = score

    # Step 2: Temporal weighting
    temporal_weights = {}
    if SAL_TEMPORAL_ENABLED:
        for cid, concept in concept_map.items():
            temporal_weights[cid] = compute_temporal_weight(
                concept.get("concept_type", "observation"),
                concept.get("created_at"),
                query_scope,
            )
    else:
        temporal_weights = {cid: 1.0 for cid in concept_ids}

    # Step 3: Aggregate to per-concept attention weights
    attention_weights = {}
    for cid in concept_ids:
        total = 0.0
        for (a, b), score in raw_scores.items():
            if a == cid:
                total += score * temporal_weights.get(b, 1.0)
            elif b == cid:
                total += score * temporal_weights.get(a, 1.0)
        attention_weights[cid] = total * temporal_weights.get(cid, 1.0)

    # Normalize
    weight_sum = sum(attention_weights.values())
    if weight_sum > 0:
        attention_weights = {k: v / weight_sum for k, v in attention_weights.items()}
    else:
        attention_weights = {k: 1.0 / n for k in concept_ids}

    # Step 4: Degeneracy detection
    metrics = {}
    degeneracy_detected = False
    degeneracy_type = None

    max_weight = max(attention_weights.values()) if attention_weights else 0
    if max_weight > SAL_MONOPOLE_THRESHOLD:
        degeneracy_detected = True
        degeneracy_type = "monopole"

    entropy = 0.0
    entropy_ratio = 0.0
    if not degeneracy_detected and n > 1:
        entropy = -sum(
            w * math.log(max(w, 1e-10)) for w in attention_weights.values()
        )
        max_entropy = math.log(n)
        entropy_ratio = entropy / max_entropy if max_entropy > 0 else 0
        if entropy_ratio > SAL_UNIFORMITY_THRESHOLD:
            degeneracy_detected = True
            degeneracy_type = "uniform"

    metrics["attention_entropy"] = entropy
    metrics["entropy_ratio"] = entropy_ratio
    metrics["max_attention_weight"] = max_weight
    metrics["degeneracy_detected"] = degeneracy_detected
    metrics["degeneracy_type"] = degeneracy_type

    # Concept type concentration check
    type_counts: dict[str, int] = {}
    for c in enriched_concepts:
        ct = c.get("concept_type", "unknown")
        type_counts[ct] = type_counts.get(ct, 0) + 1
    max_type_pct = max(type_counts.values()) / n if n > 0 else 0
    metrics["concept_type_concentration"] = max_type_pct
    if max_type_pct > 0.8:
        logger.warning(
            f"SAL: concept_type concentration {max_type_pct:.0%} — "
            f"possible misclassification or narrow query"
        )

    # Confidence envelope (weighted average of concept confidences)
    confidence_envelope = sum(
        attention_weights.get(c["concept_id"], 0) * c.get("confidence", 0.5)
        for c in enriched_concepts
    )

    # Step 5: Build clusters from attention weights
    sorted_concepts = sorted(
        attention_weights.items(), key=lambda x: x[1], reverse=True
    )
    clusters = []
    for cid, weight in sorted_concepts:
        if weight > 1.0 / (2 * n):  # Above half-uniform threshold
            clusters.append({
                "concept_id": cid,
                "summary": concept_map[cid].get("summary", ""),
                "attention_weight": weight,
                "knowledge_area": concept_map[cid].get("knowledge_area", ""),
            })

    # Step 6: Surprise buffer
    surprise_buffer = []
    if SAL_SURPRISE_BUFFER_ENABLED:
        relevance_scores = sorted(
            [c.get("relevance_score", 0) for c in enriched_concepts],
            reverse=True,
        )
        rel_threshold = (
            relevance_scores[max(0, int(len(relevance_scores) * 0.2) - 1)]
            if relevance_scores else 0
        )
        for c in enriched_concepts:
            cid = c["concept_id"]
            rel = c.get("relevance_score", 0)
            att = attention_weights.get(cid, 0)
            if (rel >= max(rel_threshold, SAL_SURPRISE_RELEVANCE_FLOOR)
                    and att <= SAL_SURPRISE_CONN_CEILING):
                surprise_buffer.append({
                    "concept_id": cid,
                    "summary": c.get("summary", ""),
                    "relevance": rel,
                    "attention_weight": att,
                    "flag": "high-relevance, low-connectivity",
                })

    metrics["surprise_buffer_size"] = len(surprise_buffer)
    above_uniform = len(
        [c for c in concept_ids if attention_weights.get(c, 0) > 1.0 / n]
    )
    metrics["compression_ratio"] = n / max(above_uniform, 1)

    return {
        "attention_weights": attention_weights,
        "clusters": clusters,
        "confidence_envelope": confidence_envelope,
        "degeneracy_detected": degeneracy_detected,
        "degeneracy_type": degeneracy_type,
        "surprise_buffer": surprise_buffer,
        "metrics": metrics,
    }


# ─── Exploration Injection ────────────────────────────────────

def _inject_exploration(enriched_concepts: list[dict]) -> list[dict]:
    """Inject random concepts from OUTSIDE the activation set.
    V0: placeholder — requires access to full concept store.
    Integration with storage.get_random_concepts() is a follow-up."""
    if not SAL_EXPLORATION_ENABLED:
        return []
    # V0: return empty. Exploration needs a get_random_concepts callable
    # that isn't available at this pipeline stage without additional plumbing.
    return []


# ─── Attention Cache (module-level dict) ──────────────────────

_attention_cache: dict[str, tuple[dict, float]] = {}


def _cache_get(key: str) -> dict | None:
    """Get cached attention result, or None if expired/missing."""
    from app.core.config import SAL_CACHE_ENABLED, SAL_CACHE_TTL
    if not SAL_CACHE_ENABLED or key not in _attention_cache:
        return None
    result, timestamp = _attention_cache[key]
    if time.monotonic() - timestamp > SAL_CACHE_TTL:
        del _attention_cache[key]
        return None
    return result


def _cache_put(key: str, result: dict) -> None:
    """Store attention result in cache."""
    from app.core.config import SAL_CACHE_ENABLED
    if SAL_CACHE_ENABLED:
        _attention_cache[key] = (result, time.monotonic())


def _cache_invalidate_concept(concept_id: str) -> None:
    """Remove all cached patterns containing a concept."""
    to_remove = [
        k for k, (result, _) in _attention_cache.items()
        if concept_id in result.get("attention_weights", {})
    ]
    for k in to_remove:
        del _attention_cache[k]


def cache_invalidate_all() -> None:
    """Clear entire attention cache."""
    _attention_cache.clear()


def handle_governance_event(
    event_type: str, concept_id: str, **kwargs
) -> None:
    """Handle governance events for cache invalidation."""
    if event_type in ("superseded", "quarantined"):
        _cache_invalidate_concept(concept_id)
    elif event_type == "confidence_change":
        delta = kwargs.get("delta", 0)
        if abs(delta) > 0.1:
            _cache_invalidate_concept(concept_id)
    elif event_type == "new_association":
        _cache_invalidate_concept(concept_id)
        target = kwargs.get("target_concept_id")
        if target:
            _cache_invalidate_concept(target)


# ─── Mode Orchestration ──────────────────────────────────────

def _mode_a(
    enriched: list[dict],
    query: str,
    start_time: float,
) -> dict:
    """Mode A: Multi-probe only (query decomposition into aspect probes)."""
    from app.retrieval.probe_decomposition import decompose_and_retrieve

    probe_results = decompose_and_retrieve(query, enriched)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return {
        "mode_used": "multi_probe",
        "fallback_used": False,
        "clusters": [],
        "attention_weights": {},
        "confidence_envelope": (
            sum(c.get("confidence", 0.5) for c in enriched)
            / max(len(enriched), 1)
        ),
        "surprise_buffer": [],
        "exploration": [],
        "probe_results": probe_results,
        "available_branches": [
            {
                "label": pr["aspect"],
                "concept_ids": [c["concept_id"] for c in pr["concepts"]],
            }
            for pr in probe_results
        ],
        "processing_time_ms": elapsed_ms,
        "metrics": {"probe_count": len(probe_results)},
    }


def _mode_b(
    enriched: list[dict],
    query: str,
    adjacency_graph: dict,
    start_time: float,
) -> dict:
    """Mode B: Graph attention only."""
    subgraph = extract_subgraph(enriched, adjacency_graph)
    embeddings = _get_embeddings(enriched)
    attention = compute_graph_attention(enriched, subgraph, query, embeddings)

    # Degeneracy → flat fallback
    if attention["degeneracy_detected"]:
        return _flat_result(
            "graph_attention", start_time,
            f"degeneracy:{attention['degeneracy_type']}",
        )

    exploration = _inject_exploration(enriched)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return {
        "mode_used": "graph_attention",
        "fallback_used": False,
        "clusters": attention["clusters"],
        "attention_weights": attention["attention_weights"],
        "confidence_envelope": attention["confidence_envelope"],
        "surprise_buffer": attention["surprise_buffer"],
        "exploration": exploration,
        "available_branches": [
            {
                "label": c["knowledge_area"],
                "concept_ids": [c["concept_id"]],
                "weight": c["attention_weight"],
            }
            for c in attention["clusters"]
        ],
        "processing_time_ms": elapsed_ms,
        "metrics": attention["metrics"],
    }


def _mode_c(
    enriched: list[dict],
    query: str,
    adjacency_graph: dict,
    start_time: float,
) -> dict:
    """Mode C: Multi-probe → quality gate → graph attention on unified set."""
    from app.retrieval.probe_decomposition import decompose_and_retrieve, check_probe_quality

    probe_results = decompose_and_retrieve(query, enriched)

    # Probe quality gate
    quality = check_probe_quality(probe_results)

    if not quality["passed"]:
        logger.info(
            f"SAL Mode C: probe quality failed ({quality['reason']}), "
            f"falling back to Mode B"
        )
        return _mode_b(enriched, query, adjacency_graph, start_time)

    # Union with dedup
    unified = {}
    for pr in probe_results:
        for c in pr["concepts"]:
            cid = c["concept_id"]
            if cid not in unified:
                unified[cid] = c
    unified_concepts = list(unified.values())

    if len(unified_concepts) < SAL_MIN_ACTIVATION_SIZE:
        return _flat_result(
            "multi_probe_then_attention", start_time, "unified_below_min"
        )

    # Run graph attention on unified (smaller, cleaner) set
    subgraph = extract_subgraph(unified_concepts, adjacency_graph)
    embeddings = _get_embeddings(unified_concepts)
    attention = compute_graph_attention(
        unified_concepts, subgraph, query, embeddings
    )

    if attention["degeneracy_detected"]:
        return _flat_result(
            "multi_probe_then_attention", start_time,
            f"degeneracy:{attention['degeneracy_type']}",
        )

    # Enrich clusters with probe aspect labels
    probe_labels: dict[str, list[str]] = {}
    for pr in probe_results:
        for c in pr["concepts"]:
            cid = c["concept_id"]
            if cid not in probe_labels:
                probe_labels[cid] = []
            probe_labels[cid].append(pr["aspect"])

    for cluster in attention["clusters"]:
        cluster["probe_aspects"] = probe_labels.get(
            cluster["concept_id"], []
        )

    exploration = _inject_exploration(enriched)
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    attention["metrics"]["probe_decomposition_quality"] = quality["score"]

    return {
        "mode_used": "multi_probe_then_attention",
        "fallback_used": False,
        "clusters": attention["clusters"],
        "attention_weights": attention["attention_weights"],
        "confidence_envelope": attention["confidence_envelope"],
        "surprise_buffer": attention["surprise_buffer"],
        "exploration": exploration,
        "probe_results": probe_results,
        "available_branches": [
            {
                "label": (
                    c.get("probe_aspects", [c["knowledge_area"]])[0]
                    if c.get("probe_aspects")
                    else c["knowledge_area"]
                ),
                "concept_ids": [c["concept_id"]],
                "weight": c["attention_weight"],
            }
            for c in attention["clusters"]
        ],
        "processing_time_ms": elapsed_ms,
        "metrics": attention["metrics"],
    }


# ─── Flat Result Helper ──────────────────────────────────────

def _flat_result(mode: str, start_time: float, reason: str) -> dict:
    """Return flat (no compression) result — fallback for errors/degeneracy."""
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.info(f"SAL: flat fallback (mode={mode}, reason={reason})")
    return {
        "mode_used": mode,
        "fallback_used": True,
        "fallback_reason": reason,
        "clusters": [],
        "attention_weights": {},
        "confidence_envelope": 0.0,
        "surprise_buffer": [],
        "exploration": [],
        "available_branches": [],
        "processing_time_ms": elapsed_ms,
        "metrics": {"fallback_reason": reason},
    }


# ─── Utility Functions ────────────────────────────────────────

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Pith embeddings are L2-normalized → dot product."""
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))
