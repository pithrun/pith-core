"""Wave 6 — Experiment Engine.

Cognitive experimentation: cross-domain synthesis, hypothesis generation,
counterfactual reasoning, and analogy detection.

Four experiment generators produce candidates from the concept corpus.
Results are persisted with full provenance and optional CKO production.
"""

import hashlib
import logging
import math
import re
import time
from collections import defaultdict
from datetime import timedelta
from itertools import combinations
from uuid import uuid4

from app.config import EXPERIMENT_VALID_TYPES, experiment_config
from app.datetime_utils import _utc_now, _utc_now_iso
from app.embedding import embedding_engine
from app.models import (
    Experiment,
    ExperimentCandidate,
    ExperimentResult,
)

# DEBT-104: Hoisted from process_experiment_results() body for discoverability
# EXP-012: Dedup threshold for experiment-produced concepts (empirically derived)
# Pairwise analysis: known dupes >= 0.735 cosine, non-dupes <= 0.718. Margin: 0.018.
# EXP-027: Raised from 0.78→0.88 to preserve more novel experiment concepts.
# Higher threshold = harder to match as duplicate = more concepts survive dedup.
EXP_CONCEPT_DEDUP_THRESHOLD = 0.88

# DEBT-104: Lazy one-shot for retrieval_engine import (avoid per-iteration import)
_retrieval_engine_initialized = False

logger = logging.getLogger(__name__)


# ============================================================
# Shared Utilities (§A.5)
# ============================================================

STOP_WORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "shall",
    "can",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "as",
    "into",
    "through",
    "during",
    "before",
    "after",
    "and",
    "but",
    "or",
    "nor",
    "not",
    "so",
    "yet",
    "both",
    "either",
    "neither",
    "each",
    "every",
    "all",
    "any",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "only",
    "own",
    "same",
    "than",
    "too",
    "very",
    "just",
    "because",
    "if",
    "when",
    "that",
    "this",
    "it",
    "its",
    "they",
    "them",
    "their",
    "we",
    "our",
    "you",
}


def _extract_terms(text: str) -> list[str]:
    """Lowercase word tokenization with stop word removal.
    Returns deduplicated terms preserving first-occurrence order."""
    words = re.findall(r"[a-z][a-z0-9_]+", text.lower())
    seen = set()
    terms = []
    for w in words:
        if w not in STOP_WORDS and w not in seen:
            seen.add(w)
            terms.append(w)
    return terms


def _compute_tfidf_vectors(
    concepts: list,
) -> dict[str, dict[str, float]]:
    """Compute sparse TF-IDF vectors for a list of concepts.
    Returns: {concept_id: {term: tfidf_weight, ...}, ...}
    """
    df: dict[str, int] = {}
    concept_terms: dict[str, list[str]] = {}
    for c in concepts:
        terms = _extract_terms(c.summary if hasattr(c, "summary") else str(c))
        cid = c.id if hasattr(c, "id") else str(c)
        concept_terms[cid] = terms
        for t in set(terms):
            df[t] = df.get(t, 0) + 1

    n = len(concepts)
    vectors: dict[str, dict[str, float]] = {}
    for c in concepts:
        cid = c.id if hasattr(c, "id") else str(c)
        terms = concept_terms[cid]
        if not terms:
            vectors[cid] = {}
            continue
        tf: dict[str, int] = {}
        for t in terms:
            tf[t] = tf.get(t, 0) + 1
        max_tf = max(tf.values())
        vec: dict[str, float] = {}
        for t, count in tf.items():
            idf = math.log((n + 1) / (df.get(t, 0) + 1)) + 1
            vec[t] = (count / max_tf) * idf
        vectors[cid] = vec
    return vectors


def _cosine_similarity(vec_a: dict, vec_b: dict) -> float:
    """Cosine similarity between two sparse TF-IDF vectors."""
    if not vec_a or not vec_b:
        return 0.0
    common = set(vec_a) & set(vec_b)
    if not common:
        return 0.0
    dot = sum(vec_a[t] * vec_b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ============================================================
# TF-IDF Cache (§6.11)
# ============================================================


class TFIDFCache:
    """Shared TF-IDF vector cache for batched experiment generation."""

    def __init__(self):
        self._vectors: dict[str, dict] = {}
        self._corpus_hash: str | None = None

    def get_or_compute(self, concepts: list) -> dict[str, dict]:
        """Return cached vectors if corpus unchanged, recompute otherwise."""
        current_hash = hashlib.md5(
            "|".join(
                sorted(
                    (c.id if hasattr(c, "id") else str(c)) + (c.summary if hasattr(c, "summary") else str(c))
                    for c in concepts
                )
            ).encode()
        ).hexdigest()

        if current_hash != self._corpus_hash:
            self._vectors = _compute_tfidf_vectors(concepts)
            self._corpus_hash = current_hash

        return self._vectors

    def invalidate(self):
        """Force cache invalidation."""
        self._corpus_hash = None
        self._vectors = {}


# ============================================================
# Cold Start Guard (§6.10 CS1)
# ============================================================


def _cold_start_check(concepts: list) -> dict | None:
    """Return error dict if insufficient data for experimentation."""
    min_concepts = experiment_config["general"]["min_concepts_required"]
    min_kas = experiment_config["general"]["min_knowledge_areas_required"]

    kas = set()
    for c in concepts:
        ka = c.knowledge_area if hasattr(c, "knowledge_area") else "general"
        kas.add(ka)

    if len(concepts) < min_concepts or len(kas) < min_kas:
        return {
            "status": "insufficient_data",
            "message": (
                f"Experiment engine requires ≥{min_concepts} concepts across "
                f"≥{min_kas} knowledge areas. Current: {len(concepts)} concepts, "
                f"{len(kas)} areas. Keep learning!"
            ),
        }
    return None


# ============================================================
# Generator 1: Cross-Domain Synthesis (§6.4)
# ============================================================


def generate_synthesis_candidates(
    concepts: list,
    tfidf_cache: TFIDFCache | None = None,
    max_concept_age_days: int | None = None,
) -> list[ExperimentCandidate]:
    """Find concept pairs across different KAs with moderate similarity."""
    cfg = experiment_config["synthesis"]
    floor = cfg["similarity_floor"]
    ceiling = cfg["similarity_ceiling"]
    max_candidates = cfg["max_candidates"]
    age_filter = max_concept_age_days or cfg.get("max_concept_age_days")

    # Filter by confidence and optional age
    filtered = [c for c in concepts if c.confidence >= 0.3]
    if age_filter:
        cutoff = (_utc_now() - timedelta(days=age_filter)).isoformat()
        filtered = [c for c in filtered if (c.updated_at if hasattr(c, "updated_at") else c.created_at) >= cutoff]

    if len(filtered) < 2:
        return []

    # Build inverted term index for O(n * avg_terms) pre-filter
    term_to_concepts: dict[str, set[str]] = defaultdict(set)
    concept_ka: dict[str, str] = {}
    for c in filtered:
        concept_ka[c.id] = c.knowledge_area if hasattr(c, "knowledge_area") else "general"
        for term in _extract_terms(c.summary):
            term_to_concepts[term].add(c.id)

    # Pre-filter: pairs sharing ≥1 term AND different KA
    pairs: set[tuple[str, str]] = set()
    for term, cids in term_to_concepts.items():
        for a, b in combinations(cids, 2):
            if concept_ka[a] != concept_ka[b]:
                pairs.add((min(a, b), max(a, b)))

    if not pairs:
        return []

    # Compute TF-IDF and score
    vectors = tfidf_cache.get_or_compute(filtered) if tfidf_cache else _compute_tfidf_vectors(filtered)

    candidates = []
    for a_id, b_id in pairs:
        sim = _cosine_similarity(vectors.get(a_id, {}), vectors.get(b_id, {}))
        if floor <= sim <= ceiling:
            # Count shared terms for score_components
            terms_a = set(_extract_terms(next((c.summary for c in filtered if c.id == a_id), "")))
            terms_b = set(_extract_terms(next((c.summary for c in filtered if c.id == b_id), "")))
            shared = terms_a & terms_b

            candidates.append(
                ExperimentCandidate(
                    candidate_id=str(uuid4()),
                    experiment_type="cross_domain_synthesis",
                    concept_ids=[a_id, b_id],
                    score=sim,  # ka_diversity_bonus = 1.0 (always different KAs)
                    score_components={
                        "term_sim": round(sim, 4),
                        "shared_terms": len(shared),
                        "ka_pair": f"{concept_ka[a_id]}→{concept_ka[b_id]}",
                    },
                    rationale=(
                        f"Cross-domain pair ({concept_ka[a_id]} ↔ {concept_ka[b_id]}) "
                        f"with {len(shared)} shared terms and {sim:.2f} similarity"
                    ),
                    metadata={"shared_terms": list(shared)[:10]},
                )
            )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:max_candidates]


# ============================================================
# Generator 2: Hypothesis Generation (§6.5)
# ============================================================


def _greedy_cluster(
    concepts: list,
    tfidf_cache: TFIDFCache | None = None,
    threshold: float = 0.25,
) -> list[list[str]]:
    """Greedy agglomerative clustering by TF-IDF similarity."""
    vectors = tfidf_cache.get_or_compute(concepts) if tfidf_cache else _compute_tfidf_vectors(concepts)
    assigned: set[str] = set()
    clusters: list[list[str]] = []

    sorted_concepts = sorted(concepts, key=lambda c: c.confidence, reverse=True)

    for concept in sorted_concepts:
        if concept.id in assigned:
            continue
        cluster = [concept.id]
        assigned.add(concept.id)

        for other in sorted_concepts:
            if other.id in assigned:
                continue
            sim = _cosine_similarity(vectors.get(concept.id, {}), vectors.get(other.id, {}))
            if sim >= threshold:
                cluster.append(other.id)
                assigned.add(other.id)

        clusters.append(cluster)

    return clusters


def generate_hypothesis_candidates(
    concepts: list,
    tfidf_cache: TFIDFCache | None = None,
) -> list[ExperimentCandidate]:
    """Cluster observation/pattern concepts to surface potential unifying patterns."""
    cfg = experiment_config["hypothesis"]
    threshold = cfg["cluster_threshold"]
    min_size = cfg["min_cluster_size"]
    max_candidates = cfg["max_candidates"]

    # Filter to observations and patterns with sufficient confidence
    filtered = [
        c
        for c in concepts
        if getattr(c, "concept_type", "observation") in ("observation", "pattern") and c.confidence >= 0.3
    ]

    if len(filtered) < min_size:
        return []

    vectors = tfidf_cache.get_or_compute(filtered) if tfidf_cache else _compute_tfidf_vectors(filtered)

    clusters = _greedy_cluster(filtered, tfidf_cache, threshold)

    # Only surface clusters with >= min_size members
    candidates = []
    for cluster_ids in clusters:
        if len(cluster_ids) < min_size:
            continue

        # Compute mean internal similarity
        sims = []
        for i, a in enumerate(cluster_ids):
            for b in cluster_ids[i + 1 :]:
                sims.append(_cosine_similarity(vectors.get(a, {}), vectors.get(b, {})))
        mean_sim = sum(sims) / len(sims) if sims else 0.0

        # Count concept types in cluster
        type_counts: dict[str, int] = {}
        for cid in cluster_ids:
            ct = next((c.concept_type for c in filtered if c.id == cid), "observation")
            type_counts[ct] = type_counts.get(ct, 0) + 1

        score = mean_sim  # EXP-002: Use cohesion quality, not quantity × quality
        candidates.append(
            ExperimentCandidate(
                candidate_id=str(uuid4()),
                experiment_type="hypothesis_generation",
                concept_ids=cluster_ids,
                score=round(score, 4),
                score_components={
                    "cluster_size": len(cluster_ids),
                    "mean_similarity": round(mean_sim, 4),
                    "concept_types": type_counts,
                },
                rationale=(
                    f"Cluster of {len(cluster_ids)} observations/patterns with "
                    f"{mean_sim:.2f} mean similarity — potential unifying hypothesis"
                ),
                metadata={"cluster_threshold": threshold},
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:max_candidates]


# ============================================================
# Generator 3: Counterfactual Reasoning (§6.6)
# ============================================================


def _build_adjacency(associations: list, direction: str) -> dict[str, set[str]]:
    """Build adjacency map from associations for graph walking."""
    adj: dict[str, set[str]] = defaultdict(set)
    for assoc in associations:
        if isinstance(assoc, dict):
            src = assoc.get("source", "")
            tgt = assoc.get("target", "")
        elif isinstance(assoc, list | tuple) and len(assoc) >= 2:
            src, tgt = assoc[0], assoc[1]
        else:
            src = getattr(assoc, "source", "")
            tgt = getattr(assoc, "target", "")
        if direction == "forward":
            adj[src].add(tgt)
        else:
            adj[tgt].add(src)
    return dict(adj)


def _walk_graph(seed_id: str, adj: dict[str, set[str]], max_depth: int) -> set[str]:
    """BFS walk from seed, return all reachable concept IDs (excluding seed)."""
    visited: set[str] = set()
    frontier = {seed_id}
    for _ in range(max_depth):
        next_frontier: set[str] = set()
        for node in frontier:
            for neighbor in adj.get(node, set()):
                if neighbor not in visited and neighbor != seed_id:
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
        frontier = next_frontier
        if not frontier:
            break
    return visited


def _max_chain_depth(seed_id: str, adj: dict[str, set[str]], max_depth: int) -> int:
    """Return maximum chain depth reachable from seed."""
    frontier = {seed_id}
    depth = 0
    visited: set[str] = {seed_id}
    for d in range(1, max_depth + 1):
        next_frontier: set[str] = set()
        for node in frontier:
            for neighbor in adj.get(node, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
        if next_frontier:
            depth = d
            frontier = next_frontier
        else:
            break
    return depth


def generate_counterfactual_candidates(
    concepts: list,
    associations: list,
    direction: str | None = None,
) -> list[ExperimentCandidate]:
    """Generate counterfactual candidates with bidirectional graph walk."""
    cfg = experiment_config["counterfactual"]
    max_depth = cfg["max_depth"]
    min_conf = cfg["min_confidence"]
    max_candidates = cfg["max_candidates"]
    direction = direction or cfg["default_direction"]

    if direction == "forward":
        seed_types = {"decision"}
    else:
        seed_types = {"observation", "pattern", "constraint"}

    seeds = [c for c in concepts if getattr(c, "concept_type", "") in seed_types and c.confidence >= min_conf]

    if not seeds:
        return []

    adj = _build_adjacency(associations, direction)

    # Compute max_impact for normalization
    all_impacts = []
    for seed in seeds:
        reachable = _walk_graph(seed.id, adj, max_depth)
        all_impacts.append((seed, reachable))
    max_impact = max((len(r) for _, r in all_impacts), default=1) or 1

    candidates = []
    for seed, reachable in all_impacts:
        impact = len(reachable)
        if impact == 0:
            continue

        depth = _max_chain_depth(seed.id, adj, max_depth)

        if direction == "forward":
            siblings = [
                c
                for c in concepts
                if getattr(c, "concept_type", "") == "decision"
                and getattr(c, "knowledge_area", "") == getattr(seed, "knowledge_area", "")
                and c.id != seed.id
            ]
            alt_score = min(len(siblings) / 5.0, 1.0)
            score = (impact / max_impact) * 0.6 + alt_score * 0.4
        else:
            score = (impact / max_impact) * 0.5 + (depth / max_depth) * 0.5

        summary = seed.summary[:50] if hasattr(seed, "summary") else str(seed.id)[:50]
        dir_label = "Decision" if direction == "forward" else "Outcome"
        reach_label = "downstream" if direction == "forward" else "upstream"

        candidates.append(
            ExperimentCandidate(
                candidate_id=str(uuid4()),
                experiment_type="counterfactual",
                concept_ids=[seed.id] + list(reachable),
                score=round(score, 4),
                score_components={
                    "direction": direction,
                    "downstream_count" if direction == "forward" else "upstream_count": impact,
                    "alternative_count": len(siblings) if direction == "forward" else 0,
                    "max_depth_reached": depth,
                },
                rationale=(f"{dir_label} '{summary}...' with {impact} {reach_label} connections"),
                metadata={"direction": direction, "seed_type": getattr(seed, "concept_type", "")},
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:max_candidates]


# ============================================================
# Generator 4: Analogy Detection (§6.7)
# ============================================================


def _load_embeddings_for_ids(concept_ids: list[str]) -> dict[str, "np.ndarray"]:
    """Load embedding vectors from DB for a set of concept IDs.

    Returns dict mapping concept_id → L2-normalized 384-dim numpy array.
    Concepts without embeddings are silently skipped.
    """
    import numpy as np

    from app.embedding import EMBEDDING_DIM, EMBEDDING_VERSION
    from app.storage import _db

    embeddings: dict[str, np.ndarray] = {}
    with _db() as conn:
        placeholders = ",".join("?" * len(concept_ids))
        rows = conn.execute(
            f"SELECT id, embedding, embedding_version FROM concepts "
            f"WHERE id IN ({placeholders}) AND embedding IS NOT NULL",
            concept_ids,
        ).fetchall()
        for row in rows:
            if row["embedding_version"] == EMBEDDING_VERSION:
                emb = np.frombuffer(row["embedding"], dtype=np.float32).copy()
                if emb.shape[0] == EMBEDDING_DIM:
                    embeddings[row["id"]] = emb
    return embeddings


def score_analogy_pair(
    emb_a: "np.ndarray",
    emb_b: "np.ndarray",
    type_a: str,
    type_b: str,
    tfidf_vectors: dict | None = None,
    id_a: str = "",
    id_b: str = "",
    min_emb_sim: float = 0.30,
    max_emb_sim: float = 0.50,
    bonus_same: float = 0.10,
    bonus_cross: float = 0.05,
) -> dict | None:
    """Score a single cross-KA concept pair for analogy potential.

    EXP-025: Extracted from generate_analogy_candidates for reuse in
    demand-side (conversation_turn) and supply-side (maintenance) paths.

    Returns dict with score + components, or None if pair is filtered out.
    """
    import numpy as np

    from app.models import ABSTRACT_CONCEPT_TYPES

    emb_sim = float(np.dot(emb_a, emb_b))

    if emb_sim < min_emb_sim or emb_sim > max_emb_sim:
        return None

    # TF-IDF surface overlap — continuous penalty, not binary
    tfidf_sim = 0.0
    if tfidf_vectors and id_a in tfidf_vectors and id_b in tfidf_vectors:
        tfidf_sim = _cosine_similarity(tfidf_vectors[id_a], tfidf_vectors[id_b])

    base_score = emb_sim * (1.0 - tfidf_sim)

    type_bonus = 0.0
    if type_a in ABSTRACT_CONCEPT_TYPES and type_b in ABSTRACT_CONCEPT_TYPES:
        if type_a == type_b:
            type_bonus = bonus_same
        else:
            type_bonus = bonus_cross

    final_score = base_score + type_bonus
    if final_score < 0.1:
        return None

    return {
        "score": round(final_score, 4),
        "embedding_similarity": round(emb_sim, 4),
        "tfidf_similarity": round(tfidf_sim, 4),
        "type_bonus": round(type_bonus, 4),
    }


# EXP-025: Demand-side analogy detection constants
DEMAND_ANALOGY_MIN_KAS = 3
DEMAND_ANALOGY_MAX_SUGGESTIONS = 2


def detect_demand_side_analogies(
    activated_concepts: list,
    concept_types: dict[str, str] | None = None,
    max_suggestions: int = DEMAND_ANALOGY_MAX_SUGGESTIONS,
) -> list[dict]:
    """EXP-025: Detect analogies in the activated concept set at conversation_turn time.

    Scans cross-KA pairs in the already-activated set (5-20 concepts) and scores
    them using the same heuristic as supply-side analogy detection. Returns top
    candidates as suggestions — no LLM call, no experiment creation.

    Returns list of dicts: [{concept_a, concept_b, ka_a, ka_b, score, components}]
    """
    import numpy as np

    from app.config import experiment_config
    from app.embedding import embedding_engine

    cfg = experiment_config["analogy"]
    min_emb_sim = cfg.get("min_embedding_sim", 0.30)
    max_emb_sim = cfg.get("max_embedding_sim", 0.50)
    bonus_same = cfg.get("type_bonus_same_abstract", 0.10)
    bonus_cross = cfg.get("type_bonus_cross_abstract", 0.05)

    # Extract KAs from activated concepts
    ka_set = set()
    concept_data = []  # (concept_id, summary, ka, concept_type, embedding)
    for ac in activated_concepts:
        ka = getattr(ac, "knowledge_area", "general") or "general"
        ka_set.add(ka)
        # Look up embedding from index (no DB I/O)
        pos = embedding_engine._id_to_pos.get(ac.concept_id)
        if pos is not None and embedding_engine._index_matrix is not None:
            emb = embedding_engine._index_matrix[pos]
            # A1: Use concept_types mapping if available, else default
            ct = (concept_types or {}).get(ac.concept_id, "observation")
            concept_data.append((ac.concept_id, getattr(ac, "summary", "")[:100], ka, ct, emb))

    # Gate: need 3+ KAs for meaningful cross-domain pairs
    if len(ka_set) < DEMAND_ANALOGY_MIN_KAS:
        return []

    if len(concept_data) < 2:
        return []

    # Score all cross-KA pairs
    scored = []
    for i, (id_a, sum_a, ka_a, type_a, emb_a) in enumerate(concept_data):
        for id_b, sum_b, ka_b, type_b, emb_b in concept_data[i + 1:]:
            if ka_a == ka_b:
                continue

            result = score_analogy_pair(
                emb_a=emb_a,
                emb_b=emb_b,
                type_a=type_a,
                type_b=type_b,
                min_emb_sim=min_emb_sim,
                max_emb_sim=max_emb_sim,
                bonus_same=bonus_same,
                bonus_cross=bonus_cross,
            )
            if result:
                scored.append({
                    "concept_a": {"id": id_a, "summary": sum_a, "ka": ka_a},
                    "concept_b": {"id": id_b, "summary": sum_b, "ka": ka_b},
                    "score": result["score"],
                    "components": result,
                })

    # Return top-N by score
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:max_suggestions]


def generate_analogy_candidates(
    concepts: list,
    assoc_counts: dict[str, int],
    salience_ranks: dict[str, float],
    tfidf_cache: TFIDFCache | None = None,
    embeddings: dict[str, "np.ndarray"] | None = None,
) -> list[ExperimentCandidate]:
    """Find semantically related concepts across different domains.

    EXP-016: Embedding-based scoring. Analogy = moderate embedding cosine +
    low TF-IDF overlap + cross-KA.
    EXP-022: Narrowed band [0.30, 0.50] + concept_type bonus for abstract types.
    """
    import numpy as np

    from app.models import ABSTRACT_CONCEPT_TYPES

    cfg = experiment_config["analogy"]
    max_candidates = cfg["max_candidates"]
    min_emb_sim = cfg.get("min_embedding_sim", 0.30)
    max_emb_sim = cfg.get("max_embedding_sim", 0.50)
    per_ka = cfg.get("max_pairwise_per_ka", 5)
    bonus_same = cfg.get("type_bonus_same_abstract", 0.10)
    bonus_cross = cfg.get("type_bonus_cross_abstract", 0.05)

    filtered = [c for c in concepts if c.confidence >= 0.3]
    if len(filtered) < 2:
        return []

    # PERF-025: Stratified corpus cap — top-per_ka per KA bounds O(N²) loop
    # and maximizes cross-KA diversity. Global salience cap rejected: 77% of
    # top-300 is product_strategy at current data distribution.
    if per_ka > 0 and len(filtered) > per_ka:
        ka_buckets: dict[str, list] = defaultdict(list)
        for c in sorted(filtered, key=lambda c: getattr(c, "salience", 0.5) or 0.5, reverse=True):
            ka = getattr(c, "knowledge_area", "general")
            if len(ka_buckets[ka]) < per_ka:
                ka_buckets[ka].append(c)
        original_len = len(filtered)
        filtered = [c for bucket in ka_buckets.values() for c in bucket]
        if len(filtered) < original_len:
            logger.info(
                "generate_analogy_candidates: stratified corpus cap %d→%d (%d KAs, %d/KA)",
                original_len,
                len(filtered),
                len(ka_buckets),
                per_ka,
            )

    # EXP-016: Load embeddings for filtered concepts (after cap, not before).
    # DB I/O — injectable via `embeddings` param for testing.
    if embeddings is None:
        embeddings = _load_embeddings_for_ids([c.id for c in filtered])

    # Defensive: filter to concepts that have embeddings
    filtered = [c for c in filtered if c.id in embeddings]
    if len(filtered) < 2:
        logger.warning("generate_analogy_candidates: <2 concepts with embeddings, skipping")
        return []

    vectors = tfidf_cache.get_or_compute(filtered) if tfidf_cache else _compute_tfidf_vectors(filtered)

    candidates = []
    for i, a in enumerate(filtered):
        ka_a = getattr(a, "knowledge_area", "general")
        type_a = getattr(a, "concept_type", "observation")
        emb_a = embeddings[a.id]
        for b in filtered[i + 1 :]:
            ka_b = getattr(b, "knowledge_area", "general")
            if ka_a == ka_b:
                continue  # Same KA = comparison, not analogy

            # EXP-025: Use shared scoring function
            type_b = getattr(b, "concept_type", "observation")
            emb_b = embeddings[b.id]
            pair_result = score_analogy_pair(
                emb_a=emb_a,
                emb_b=emb_b,
                type_a=type_a,
                type_b=type_b,
                tfidf_vectors=vectors,
                id_a=a.id,
                id_b=b.id,
                min_emb_sim=min_emb_sim,
                max_emb_sim=max_emb_sim,
                bonus_same=bonus_same,
                bonus_cross=bonus_cross,
            )
            if pair_result is None:
                continue

            final_score = pair_result["score"]
            emb_sim = pair_result["embedding_similarity"]
            tfidf_sim = pair_result["tfidf_similarity"]
            type_bonus = pair_result["type_bonus"]

            candidates.append(
                ExperimentCandidate(
                    candidate_id=str(uuid4()),
                    experiment_type="analogy_detection",
                    concept_ids=[a.id, b.id],
                    score=round(final_score, 4),
                    score_components={
                        "embedding_similarity": round(emb_sim, 4),
                        "tfidf_similarity": round(tfidf_sim, 4),
                        "vocabulary_divergence": round(1.0 - tfidf_sim, 4),
                        "concept_type_a": type_a,
                        "concept_type_b": type_b,
                        "type_bonus": round(type_bonus, 4),
                    },
                    rationale=(
                        f"Analogy: {ka_a} ↔ {ka_b}, "
                        f"emb={emb_sim:.3f}, tfidf={tfidf_sim:.3f}, "
                        f"type_bonus={type_bonus:.2f}, score={final_score:.3f}"
                    ),
                    metadata={
                        "ka_pair": f"{ka_a}→{ka_b}",
                        "embedding_similarity": round(emb_sim, 4),
                        "tfidf_similarity": round(tfidf_sim, 4),
                    },
                )
            )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:max_candidates]


# ============================================================
# Experiment CRUD & Persistence
# ============================================================


def save_experiment(experiment: Experiment) -> None:
    """Persist experiment to SQLite."""
    import json

    from app.storage import _db

    experiment.updated_at = _utc_now_iso()
    candidates_json = json.dumps([c.model_dump() for c in experiment.candidates])
    result_json = json.dumps(experiment.result.model_dump()) if experiment.result else None

    with _db() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO experiments
            (id, experiment_type, status, created_at, updated_at,
             candidates, result, concept_ids_produced, cko_ids_produced,
             thread_id, config_snapshot, generation_time_ms,
             processing_time_ms, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                experiment.id,
                experiment.experiment_type,
                experiment.status,
                experiment.created_at,
                experiment.updated_at,
                candidates_json,
                result_json,
                json.dumps(experiment.concept_ids_produced),
                json.dumps(experiment.cko_ids_produced),
                experiment.thread_id,
                json.dumps(experiment.config_snapshot),
                experiment.generation_time_ms,
                experiment.processing_time_ms,
                json.dumps(experiment.metadata),
            ),
        )


def load_experiment(experiment_id: str) -> Experiment | None:
    """Load experiment by ID."""
    import json

    from app.storage import _db

    with _db() as conn:
        row = conn.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,)).fetchone()

    if not row:
        return None

    candidates_raw = json.loads(row["candidates"]) if row["candidates"] else []
    result_raw = json.loads(row["result"]) if row["result"] else None

    return Experiment(
        id=row["id"],
        experiment_type=row["experiment_type"],
        status=row["status"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        candidates=[ExperimentCandidate(**c) for c in candidates_raw],
        result=ExperimentResult(**result_raw) if result_raw else None,
        concept_ids_produced=json.loads(row["concept_ids_produced"]) if row["concept_ids_produced"] else [],
        cko_ids_produced=json.loads(row["cko_ids_produced"]) if row["cko_ids_produced"] else [],
        thread_id=row["thread_id"],
        config_snapshot=json.loads(row["config_snapshot"]) if row["config_snapshot"] else {},
        generation_time_ms=row["generation_time_ms"],
        processing_time_ms=row["processing_time_ms"],
        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
    )


def load_experiments(
    status: list[str] | None = None,
    include_archived: bool = False,
    limit: int = 50,
    experiment_type: str | None = None,  # [CRC-1] EXP-001
) -> list[Experiment]:
    """Load experiments filtered by status and optionally by type."""
    import json

    from app.storage import _db

    with _db() as conn:
        if status:
            placeholders = ",".join("?" * len(status))
            query = f"SELECT * FROM experiments WHERE status IN ({placeholders})"
            params: list = list(status)
            if experiment_type:
                query += " AND experiment_type = ?"
                params.append(experiment_type)
            query += " ORDER BY updated_at DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(query, params).fetchall()
        elif include_archived:
            query = "SELECT * FROM experiments"
            params = []
            if experiment_type:
                query += " WHERE experiment_type = ?"
                params.append(experiment_type)
            query += " ORDER BY updated_at DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(query, params).fetchall()
        else:
            query = "SELECT * FROM experiments WHERE status != 'archived'"
            params = []
            if experiment_type:
                query += " AND experiment_type = ?"
                params.append(experiment_type)
            query += " ORDER BY updated_at DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(query, params).fetchall()

    results = []
    for row in rows:
        candidates_raw = json.loads(row["candidates"]) if row["candidates"] else []
        result_raw = json.loads(row["result"]) if row["result"] else None
        results.append(
            Experiment(
                id=row["id"],
                experiment_type=row["experiment_type"],
                status=row["status"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                candidates=[ExperimentCandidate(**c) for c in candidates_raw],
                result=ExperimentResult(**result_raw) if result_raw else None,
                concept_ids_produced=json.loads(row["concept_ids_produced"]) if row["concept_ids_produced"] else [],
                cko_ids_produced=json.loads(row["cko_ids_produced"]) if row["cko_ids_produced"] else [],
                thread_id=row["thread_id"],
                config_snapshot=json.loads(row["config_snapshot"]) if row["config_snapshot"] else {},
                generation_time_ms=row["generation_time_ms"],
                processing_time_ms=row["processing_time_ms"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            )
        )
    return results


# ============================================================
# Dedup Gate (EXP-001) — EXPERIMENT_RESOLUTION_SPEC v1.2
# ============================================================


def _is_duplicate_experiment(
    experiment_type: str,
    candidates: list,
    overlap_threshold: float = 0.6,
) -> bool:
    """Check if a substantially similar experiment already exists.

    Compares concept_ids of top 3 candidates against ALL existing experiments
    (cross-type). Returns True if any existing experiment has >=60%
    Jaccard overlap. EXP-005: Previously only checked same experiment_type,
    allowing near-duplicate experiments across different types.
    """
    if not candidates:
        return False

    new_concept_ids: set[str] = set()
    for c in candidates[:3]:
        new_concept_ids.update(c.concept_ids)

    if not new_concept_ids:
        return False

    # EXP-005: Cross-type dedup — check ALL experiment types, not just same type.
    # Previously filtered to experiment_type only, allowing near-duplicate
    # experiments across different types (e.g. synthesis vs hypothesis_generation).
    existing = load_experiments(
        status=["reasoning", "completed"],
        # experiment_type removed — check cross-type duplicates
        limit=100,
    )

    for exp in existing:
        if not exp.candidates:
            continue
        existing_ids: set[str] = set()
        for c in exp.candidates[:3]:
            existing_ids.update(c.concept_ids)
        if not existing_ids:
            continue

        overlap = len(new_concept_ids & existing_ids) / max(len(new_concept_ids | existing_ids), 1)
        if overlap >= overlap_threshold:
            return True

    return False


def _has_purge_already_run() -> bool:
    """Check if retroactive dedup purge already ran (survives restarts). [FRAG-1]"""
    from app.storage import _db

    with _db() as conn:
        row = conn.execute(
            "SELECT count(*) FROM experiments WHERE metadata LIKE '%retroactive_dedup_purge%'"
        ).fetchone()
        return row[0] > 0


def retroactive_dedup_purge() -> int:
    """Archive duplicate + bad-score experiments. [I-1, PD-1, DM-2, FRAG-1]

    Keeps the oldest of each unique experiment set per type.
    Also archives completed experiments with impossible confidence (>1.0).
    Uses individual save_experiment calls — idempotent, partial completion is safe.
    """
    from collections import defaultdict

    # [FRAG-1] Skip if already ran (persisted check)
    if _has_purge_already_run():
        return 0

    experiments = load_experiments(status=["reasoning"], limit=200)

    # [DM-2] Also archive completed experiments with bad scores
    bad_score_exps = load_experiments(status=["completed"], limit=200)
    bad_archived = 0
    for exp in bad_score_exps:
        if exp.result and exp.result.confidence > 1.0:
            exp.status = "archived"
            if not exp.metadata:
                exp.metadata = {}
            exp.metadata["archive_reason"] = "retroactive_dedup_purge_bad_score"
            exp.updated_at = _utc_now_iso()
            save_experiment(exp)
            bad_archived += 1

    # Group by type, then by concept_id overlap
    type_groups: dict[str, list] = defaultdict(list)
    for exp in experiments:
        all_cids: set[str] = set()
        for c in (exp.candidates or [])[:3]:
            all_cids.update(c.concept_ids)
        type_groups[exp.experiment_type].append((exp, all_cids))

    archived = 0
    for etype, group in type_groups.items():
        keep: list[tuple] = []
        for exp, cids in group:
            is_dup = False
            for kept_exp, kept_cids in keep:
                if not cids or not kept_cids:
                    continue
                overlap = len(cids & kept_cids) / max(len(cids | kept_cids), 1)
                if overlap >= 0.6:
                    is_dup = True
                    break
            if is_dup:
                exp.status = "archived"
                if not exp.metadata:
                    exp.metadata = {}
                exp.metadata["archive_reason"] = "retroactive_dedup_purge"
                exp.updated_at = _utc_now_iso()
                save_experiment(exp)
                archived += 1
            else:
                keep.append((exp, cids))

    return archived + bad_archived


# ============================================================
# Result Processing & CKO Production (§6.9)
# ============================================================

# [EXP-020] Confidence threshold for tagging a resolved experiment concept as novel
NOVEL_INSIGHT_THRESHOLD = 0.6
# [EXP-020] Maximum source concept IDs to include in evidence for traceability
SOURCE_IDS_MAX = 5


def process_experiment_results(experiment_id: str, result: ExperimentResult) -> dict:
    """Persist experiment results: create concepts, optionally produce CKO.

    Returns summary dict with concepts_created, cko_created, cko_id, processing_time_ms.
    """
    start = time.monotonic()
    experiment = load_experiment(experiment_id)
    if not experiment:
        return {"error": f"Experiment {experiment_id} not found"}

    cfg = experiment_config["general"]
    concept_ids: list[str] = []

    # CKO-001: Actually CREATE concepts here instead of returning specs.
    # Previously, concept specs were returned for server.py to create, but
    # server.py never did — so experiment concepts were never persisted.

    # [EXP-020] Build top source concept IDs from top candidate for provenance tracing
    _top_source_ids: list[str] = []
    if experiment.candidates:
        _top_cand = max(experiment.candidates, key=lambda c: c.score)
        _top_source_ids = [f"source:{cid}" for cid in _top_cand.concept_ids[:SOURCE_IDS_MAX]]

    for spec in result.concepts_produced:
        summary = spec.get("summary", "")
        # [EXP-020] Authoritative evidence: experiment provenance + top-5 source concept refs
        # Deliberately NOT using spec.get("evidence") — prevents duplication from LLM spec
        evidence = [f"experiment:{experiment_id}"] + _top_source_ids

        # MEASURE-019: Track dedup score for yield logging
        _yield_dedup_score = 0.0

        # EXP-012: Check for near-duplicate via embedding cosine before creating
        try:
            # DEBT-104: One-shot retrieval_engine init (was per-iteration import)
            global _retrieval_engine_initialized
            if embedding_engine.index_size == 0 and not _retrieval_engine_initialized:
                from app.retrieval import retrieval_engine

                retrieval_engine._init_embeddings()
                _retrieval_engine_initialized = True
            emb_hits = embedding_engine.search(summary, top_k=3)
            # emb_hits is List[(concept_id, cosine_score)]
            if emb_hits:
                top_id, top_score = emb_hits[0]
                _yield_dedup_score = top_score  # MEASURE-019: capture for yield log
                if top_score >= EXP_CONCEPT_DEDUP_THRESHOLD:
                    logger.info(
                        "EXP-012: Skipping duplicate exp concept for experiment %s (cosine=%.3f with %s: '%s')",
                        experiment_id[:8],
                        top_score,
                        top_id[:16],
                        summary[:60],
                    )
                    continue
        except Exception as e:
            logger.warning("EXP-012: Dedup check failed, proceeding with creation: %s", e)

        try:
            from app.learning import create_concept
            from app.models import ConceptProposal

            # [EXP-020] Build experiment-origin signals for tagging and retrieval
            _exp_signals = [f"experiment:{experiment.experiment_type}"]
            if result.confidence >= NOVEL_INSIGHT_THRESHOLD:
                _exp_signals.append("novel_insight")
            _exp_signals.extend(spec.get("signals", []))  # preserve any LLM-provided signals

            proposal = ConceptProposal(
                concept_id=spec.get("concept_id", f"exp_{experiment_id}_{len(concept_ids)}"),
                summary=summary,
                knowledge_area=spec.get("knowledge_area", "general"),
                evidence=evidence,
                signals=_exp_signals,
                confidence=result.confidence * 0.9,  # slight discount
                concept_type=spec.get("concept_type", "observation"),
            )
            created = create_concept(proposal)
            concept_ids.append(created.id)

            # MEASURE-019: Yield tracking structured log
            logger.info(
                "EXP_YIELD: experiment=%s type=%s concept_id=%s confidence=%.3f "
                "dedup_score=%.3f summary=%.60s",
                experiment_id[:8],
                experiment.experiment_type,
                created.id[:16],
                result.confidence,
                _yield_dedup_score,
                summary[:60],
            )
        except Exception as e:
            logger.warning(f"CKO-001: Failed to create concept from experiment {experiment_id}: {e}")

    # CKO production gate
    cko_id = None
    if (
        result.confidence >= cfg["cko_min_confidence"]
        and len(concept_ids) >= cfg["cko_min_concepts"]
        and result.cko_produced
    ):
        # CKO-001 / GA-003+GA-004: Create CKO with real persisted concept IDs
        try:
            from app.cko import create_cko
            from app.storage import _get_connection

            conn = _get_connection()
            try:
                cko = create_cko(
                    conn=conn,
                    title=result.cko_produced.get("title", "Experiment Result"),
                    concept_ids=concept_ids,
                    synthesis=result.cko_produced.get("synthesis", result.synthesis),
                    knowledge_area=result.cko_produced.get("knowledge_area", "general"),
                    cko_type="analysis",
                )
                cko_id = cko.id
                logger.info(
                    f"CKO-001: Created CKO {cko_id} from experiment {experiment_id} with {len(concept_ids)} concepts"
                )
            finally:
                conn.close()
        except Exception as e:
            logger.warning(f"CKO-001: Failed to create CKO from experiment {experiment_id}: {e}")

    # Update experiment record
    experiment.status = "completed"
    experiment.result = result
    experiment.concept_ids_produced = concept_ids  # EXP-009: was missing — concepts created but never tracked
    experiment.processing_time_ms = int((time.monotonic() - start) * 1000)
    experiment.updated_at = _utc_now_iso()
    save_experiment(experiment)

    return {
        "concepts_created": concept_ids,
        "cko_id": cko_id,
        "cko_gate_passed": cko_id is not None,
        "processing_time_ms": experiment.processing_time_ms,
    }


# ============================================================
# Archive & Lifecycle (§6.8)
# ============================================================


def archive_experiment(experiment_id: str) -> Experiment | None:
    """Manually archive a specific experiment."""
    experiment = load_experiment(experiment_id)
    if not experiment:
        return None
    experiment.status = "archived"
    experiment.updated_at = _utc_now_iso()
    save_experiment(experiment)
    return experiment


def archive_stale_experiments() -> int:
    """Auto-archive completed/insufficient_data experiments past their retention window."""
    from app.storage import _db

    archive_days = experiment_config["general"]["archive_days"]
    completed_cutoff = (_utc_now() - timedelta(days=archive_days)).isoformat()
    # EXP-006: insufficient_data experiments are dead-ends — archive after 3 days
    insufficient_cutoff = (_utc_now() - timedelta(days=3)).isoformat()
    now = _utc_now_iso()

    with _db() as conn:
        cursor = conn.execute(
            """
            UPDATE experiments
            SET status = 'archived', updated_at = ?
            WHERE (status = 'completed' AND updated_at < ?)
               OR (status = 'insufficient_data' AND updated_at < ?)
        """,
            (now, completed_cutoff, insufficient_cutoff),
        )
        count = cursor.rowcount

    if count:
        logger.info(f"Auto-archived {count} stale experiment(s)")
    return count


# ============================================================
# Main Orchestrator — generate_experiment()
# ============================================================

GENERATOR_MAP = {
    "cross_domain_synthesis": "synthesis",
    "hypothesis_generation": "hypothesis",
    "counterfactual": "counterfactual",
    "analogy_detection": "analogy",
}


def _load_experiment_corpus(concepts_only: bool = False) -> tuple:
    """Load and prepare corpus for experiment generation (blocking sync helper).

    Extracted from phase3_experiments so it can be run in a thread executor,
    making asyncio.wait_for's timeout effective (MAINT-026).

    Args:
        concepts_only: If True, skip association/TFIDFCache loading (used for dry_run).
            Preserves pre-fix dry_run behavior: corpus_size without full data load.

    Returns:
        (concepts, associations, assoc_counts, salience_ranks, tfidf_cache)
        When concepts_only=True: associations=[], assoc_counts={}, salience_ranks={}, tfidf_cache=None
    """
    from app.storage import list_concepts, load_associations, load_concept

    concept_ids = list_concepts()
    concepts = []
    for cid in concept_ids:
        c = load_concept(cid, track_access=False)
        if c:
            concepts.append(c)

    if concepts_only:
        return concepts, [], {}, {}, None

    from collections import Counter

    raw_assocs = load_associations()
    assoc_list = raw_assocs.get("associations", [])
    associations = [
        (a["source"], a["target"], a.get("relation", "related_to"), a.get("strength", 0.5))
        for a in assoc_list
        if isinstance(a, dict) and "source" in a and "target" in a
    ]

    assoc_counts: Counter = Counter()
    for s, t, *_ in associations:
        assoc_counts[s] += 1
        assoc_counts[t] += 1

    salience_ranks: dict[str, float] = {}
    sorted_by_salience = sorted(concepts, key=lambda c: getattr(c, "salience", 0.5), reverse=True)
    for rank, c in enumerate(sorted_by_salience):
        salience_ranks[c.id] = rank / max(len(sorted_by_salience) - 1, 1)

    tfidf_cache = TFIDFCache()

    return concepts, associations, dict(assoc_counts), salience_ranks, tfidf_cache


def generate_experiment(
    experiment_type: str,
    concepts: list,
    associations: list | None = None,
    assoc_counts: dict[str, int] | None = None,
    salience_ranks: dict[str, float] | None = None,
    direction: str | None = None,
    max_concept_age_days: int | None = None,
    thread_id: str | None = None,
    tfidf_cache: TFIDFCache | None = None,
) -> Experiment:
    """Generate experiment candidates for the given type.

    Returns a persisted Experiment record with candidates populated.
    """
    start = time.monotonic()

    # Validate type
    if experiment_type not in EXPERIMENT_VALID_TYPES:
        raise ValueError(f"Invalid experiment_type '{experiment_type}'. Valid: {sorted(EXPERIMENT_VALID_TYPES)}")

    # Cold start guard
    cold_check = _cold_start_check(concepts)
    if cold_check:
        experiment = Experiment(
            experiment_type=experiment_type,
            status="insufficient_data",
            thread_id=thread_id,
            config_snapshot=experiment_config.copy(),
            metadata=cold_check,
        )
        save_experiment(experiment)
        return experiment

    # Snapshot config for reproducibility
    config_snapshot = {k: dict(v) if isinstance(v, dict) else v for k, v in experiment_config.items()}

    # Generate candidates based on type
    candidates: list[ExperimentCandidate] = []

    if experiment_type == "cross_domain_synthesis":
        candidates = generate_synthesis_candidates(concepts, tfidf_cache, max_concept_age_days)

    elif experiment_type == "hypothesis_generation":
        candidates = generate_hypothesis_candidates(concepts, tfidf_cache)

    elif experiment_type == "counterfactual":
        candidates = generate_counterfactual_candidates(concepts, associations or [], direction)

    elif experiment_type == "analogy_detection":
        candidates = generate_analogy_candidates(
            concepts,
            assoc_counts or {},
            salience_ranks or {},
            tfidf_cache,
        )

    # Cap stored candidates [S1 fix]
    max_stored = experiment_config["general"]["max_stored_candidates"]
    stored_candidates = candidates[:max_stored]

    # Dedup gate (EXP-001) — skip if substantially similar experiment exists
    if _is_duplicate_experiment(experiment_type, stored_candidates):
        logger.info("EXP-001: Skipping duplicate %s experiment", experiment_type)
        return None  # Caller handles None gracefully

    gen_time = int((time.monotonic() - start) * 1000)

    experiment = Experiment(
        experiment_type=experiment_type,
        status="reasoning" if stored_candidates else "insufficient_data",
        candidates=stored_candidates,
        thread_id=thread_id,
        config_snapshot=config_snapshot,
        generation_time_ms=gen_time,
        metadata={
            "total_candidates_generated": len(candidates),
            "candidates_stored": len(stored_candidates),
            "direction": direction,
        },
    )
    save_experiment(experiment)

    logger.info(
        f"Experiment {experiment.id}: type={experiment_type}, candidates={len(stored_candidates)}, time={gen_time}ms"
    )
    return experiment
