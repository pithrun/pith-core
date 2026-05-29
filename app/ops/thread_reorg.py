"""THREAD-004: Thread reorganization substrate.

Offline mining, staged add-only batch preview/commit/rollback, and
seed-candidate queuing used to contain the active sink thread.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np

from app.core.config import (
    THREAD_REORG_ASSOC_FLOOR,
    THREAD_REORG_BATCH_WRITE_ENABLED,
    THREAD_REORG_CONTROL_REGRESSION_CAUTION,
    THREAD_REORG_EVAL_CONTROL_SIZE,
    THREAD_REORG_EVAL_PRIMARY_SIZE,
    THREAD_REORG_MAX_BATCH_SIZE,
)
from app.core.datetime_utils import _utc_now_iso
from app.core.models import NarrativeThread
from app.storage.backend import get_backend
from app.storage import get_db_connection, load_concept
from app.storage.embedding import embedding_engine

logger = logging.getLogger(__name__)

TARGET_CLUSTER_MIN = 20
TARGET_CLUSTER_MAX = 80
THREAD_PREVIEW_SIZE_CAP = 300
MIN_PREVIEW_CLUSTER_SIZE = 3
SHORTLIST_LIMIT_PER_SIGNAL = 25
MAX_SPLIT_DEPTH = 3
DEFERRED_SINGLETON_BUCKET_SIZE = 50
SEMANTIC_CANDIDATE_FLOOR = 0.55
TEMPORAL_CANDIDATE_FLOOR = 0.05
STANDARD_MERGE_FLOOR = 0.62
MICRO_SEMANTIC_FLOOR = 0.70
MICRO_TEMPORAL_FLOOR = 0.55
MICRO_KA_FLOOR = 0.70
MIN_EMBEDDING_COVERAGE = 0.80
DEFERRED_REVIEW_MIN_CONFIDENCE = 0.35
DEFERRED_REVIEW_MIN_SUMMARY_CHARS = 80
DEFERRED_REVIEW_SHAPED_BUCKET_CAP = 10
DEFERRED_REVIEW_CONVERSATIONAL_MARKERS = (
    "if you want, i can keep polling",
    "if you want, i'll keep this narrow",
    "stop/continue recommendation",
    "exact saved score/state",
)
DEFERRED_REVIEW_TOPIC_FAMILIES = {
    "architecture": {
        "retrieval": {"retrieval", "recall", "search", "query"},
        "temporal": {"temporal", "currency", "freshness", "recency", "stale", "aging"},
        "evolution": {"evolution", "supersession", "superseded", "drift", "version"},
        "path": {"path", "route", "routing", "endpoint", "listener"},
        "maintenance": {"maintenance", "reflect", "scheduler", "background", "monitor"},
        "embedding": {"embedding", "centroid", "semantic", "vector"},
        "governance": {"governance", "policy", "constraint", "authority"},
    },
    "product_strategy": {
        "pricing": {"pricing", "price", "tier", "budget", "monetization", "free", "paid"},
        "launch": {"launch", "launching", "ship", "rollout", "pilot", "release"},
        "positioning": {"positioning", "category", "messaging", "brand", "one-liner", "hero"},
        "distribution": {"distribution", "channel", "growth", "gtm", "acquisition", "reach"},
        "governance": {"governance", "trust", "safety", "reliability"},
        "install": {"install", "onboarding", "setup", "integration", "plugin", "mcp"},
        "outreach": {"reply", "replies", "outreach", "citation", "benchmark", "webinar", "hn"},
    },
}


@dataclass
class ThreadReorgMiningResult:
    source_thread_id: str
    source_thread_title: str
    resolved_count: int
    stale_concept_ids: list[str] = field(default_factory=list)
    cluster_count: int = 0
    clusters: list[dict[str, Any]] = field(default_factory=list)
    percentile_snapshot: dict[str, dict[str, float]] = field(default_factory=dict)
    merge_decisions: list[dict[str, Any]] = field(default_factory=list)
    candidate_stats: dict[str, Any] = field(default_factory=dict)
    deferred_singletons: list[dict[str, Any]] = field(default_factory=list)


@contextmanager
def _read_conn():
    with get_backend().db() as conn:
        yield conn


@contextmanager
def _write_conn():
    with get_backend().db_immediate() as conn:
        yield conn


def _normalized(vec: np.ndarray | None) -> np.ndarray | None:
    if vec is None:
        return None
    norm = float(np.linalg.norm(vec))
    if norm <= 0:
        return None
    return vec / norm


def _fetch_active_thread(source_thread_id: str | None = None) -> NarrativeThread:
    from app.features.threads import load_thread, load_threads

    if source_thread_id:
        thread = load_thread(source_thread_id)
        if not thread:
            raise ValueError(f"Thread {source_thread_id} not found")
        return thread

    active_threads = load_threads(status="active")
    if not active_threads:
        raise ValueError("No active thread available for thread reorg")
    return active_threads[0]


def _concept_record(concept_id: str):
    concept = load_concept(concept_id, track_access=False)
    if not concept:
        return None
    return {
        "id": concept.id,
        "summary": getattr(concept, "summary", "") or "",
        "confidence": getattr(concept, "confidence", None),
        "knowledge_area": getattr(concept, "knowledge_area", "") or "",
        "session_id": getattr(concept, "session_id", None),
        "source_trace_id": getattr(concept, "source_trace_id", None),
        "concept_type": getattr(concept, "concept_type", None),
    }


def _load_thread_concepts(thread: NarrativeThread) -> tuple[dict[str, dict[str, Any]], list[str]]:
    concepts: dict[str, dict[str, Any]] = {}
    stale: list[str] = []
    for concept_id in thread.concept_ids:
        record = _concept_record(concept_id)
        if record is None:
            stale.append(concept_id)
        else:
            concepts[concept_id] = record
    return concepts, stale


def _load_association_graph(concept_ids: set[str]) -> tuple[dict[str, dict[str, float]], set[tuple[str, str]]]:
    graph: dict[str, dict[str, float]] = {cid: {} for cid in concept_ids}
    cross_pairs: set[tuple[str, str]] = set()
    if not concept_ids:
        return graph, cross_pairs

    placeholders = ",".join("?" for _ in concept_ids)
    query = f"""
        SELECT source, target, strength
        FROM associations
        WHERE source IN ({placeholders}) AND target IN ({placeholders})
    """
    params = tuple(concept_ids) + tuple(concept_ids)
    with _read_conn() as conn:
        rows = conn.execute(query, params).fetchall()

    for source, target, strength in rows:
        if source == target:
            continue
        weight = float(strength or 0.0)
        graph[source][target] = max(weight, graph[source].get(target, 0.0))
        graph[target][source] = max(weight, graph[target].get(source, 0.0))
        cross_pairs.add(tuple(sorted((source, target))))
    return graph, cross_pairs


def _connected_components(nodes: list[str], graph: dict[str, dict[str, float]]) -> list[list[str]]:
    remaining = set(nodes)
    components: list[list[str]] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        component = [start]
        while stack:
            current = stack.pop()
            for neighbor in graph.get(current, {}):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
                    component.append(neighbor)
        components.append(sorted(component))
    return components


def _load_embedding_map(concept_ids: set[str]) -> dict[str, np.ndarray]:
    vectors: dict[str, np.ndarray] = {}
    if getattr(embedding_engine, "_index_matrix", None) is None:
        return vectors
    for concept_id in concept_ids:
        pos = getattr(embedding_engine, "_id_to_pos", {}).get(concept_id)
        if pos is None:
            continue
        vectors[concept_id] = embedding_engine._index_matrix[pos]
    return vectors


def _cluster_metadata(
    cluster_ids: list[str],
    concept_map: dict[str, dict[str, Any]],
    embedding_map: dict[str, np.ndarray],
) -> dict[str, Any]:
    ka_counts = Counter()
    session_ids = set()
    trace_ids = set()
    summaries = []
    centroid_inputs = []
    for concept_id in cluster_ids:
        record = concept_map[concept_id]
        if record["knowledge_area"]:
            ka_counts[record["knowledge_area"]] += 1
        if record["session_id"]:
            session_ids.add(record["session_id"])
        if record.get("source_trace_id"):
            trace_ids.add(record["source_trace_id"])
        if record["summary"]:
            summaries.append(record["summary"])
        if concept_id in embedding_map:
            centroid_inputs.append(embedding_map[concept_id])

    dominant_ka = ""
    dominant_share = 0.0
    if ka_counts:
        dominant_ka, dominant_count = ka_counts.most_common(1)[0]
        dominant_share = dominant_count / len(cluster_ids)

    centroid = None
    embedding_coverage = 0.0
    if centroid_inputs:
        embedding_coverage = len(centroid_inputs) / len(cluster_ids)
        centroid = _normalized(np.mean(np.stack(centroid_inputs), axis=0))

    return {
        "cluster_id": f"cluster-{uuid.uuid4().hex[:8]}",
        "concept_ids": sorted(cluster_ids),
        "size": len(cluster_ids),
        "session_ids": session_ids,
        "trace_ids": trace_ids,
        "ka_counts": dict(ka_counts),
        "dominant_ka": dominant_ka,
        "dominant_ka_share": round(dominant_share, 4),
        "summary_preview": summaries[:3],
        "embedding_coverage": round(embedding_coverage, 4),
        "_centroid": centroid,
    }


def _cross_association_strengths(
    cluster_a: dict[str, Any],
    cluster_b: dict[str, Any],
    graph: dict[str, dict[str, float]],
) -> list[float]:
    strengths: list[float] = []
    ids_b = set(cluster_b["concept_ids"])
    for concept_id in cluster_a["concept_ids"]:
        for other_id, strength in graph.get(concept_id, {}).items():
            if other_id in ids_b:
                strengths.append(float(strength or 0.0))
    return strengths


def _temporal_overlap(cluster_a: dict[str, Any], cluster_b: dict[str, Any]) -> float:
    union = cluster_a["session_ids"] | cluster_b["session_ids"]
    if not union:
        return 0.0
    overlap = cluster_a["session_ids"] & cluster_b["session_ids"]
    return len(overlap) / len(union)


def _semantic_similarity(cluster_a: dict[str, Any], cluster_b: dict[str, Any]) -> tuple[float | None, bool]:
    coverage_ok = (
        cluster_a["embedding_coverage"] >= MIN_EMBEDDING_COVERAGE
        and cluster_b["embedding_coverage"] >= MIN_EMBEDDING_COVERAGE
    )
    if not coverage_ok:
        return None, False
    centroid_a = cluster_a.get("_centroid")
    centroid_b = cluster_b.get("_centroid")
    if centroid_a is None or centroid_b is None:
        return None, False
    return float(centroid_a @ centroid_b), True


def _ka_raw(cluster_a: dict[str, Any], cluster_b: dict[str, Any]) -> float:
    combined = Counter(cluster_a["ka_counts"]) + Counter(cluster_b["ka_counts"])
    total = sum(combined.values())
    if total <= 0:
        return 0.0
    return max(combined.values()) / total


def _percentile_pair(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=float)
    return float(np.percentile(arr, 25)), float(np.percentile(arr, 75))


def _normalize_signal(value: float | None, p25: float, p75: float) -> float:
    if value is None:
        return 0.0
    if p75 <= p25:
        return 1.0 if value > p25 else 0.0
    return float(max(0.0, min(1.0, (value - p25) / (p75 - p25))))


def _is_micro_cluster(cluster: dict[str, Any]) -> bool:
    return cluster["size"] < MIN_PREVIEW_CLUSTER_SIZE


def _is_previewable_parent(cluster: dict[str, Any]) -> bool:
    return (
        cluster["size"] >= MIN_PREVIEW_CLUSTER_SIZE
        and cluster["size"] <= THREAD_PREVIEW_SIZE_CAP
        and cluster["dominant_ka_share"] >= MICRO_KA_FLOOR
    )


def _strict_parent_merge_decision(
    candidate: dict[str, Any],
    temporal_norm: float,
) -> dict[str, Any]:
    cluster_a = candidate["cluster_a"]
    cluster_b = candidate["cluster_b"]
    cluster_a_micro = _is_micro_cluster(cluster_a)
    cluster_b_micro = _is_micro_cluster(cluster_b)

    if not (cluster_a_micro or cluster_b_micro):
        return {
            "accepted": False,
            "reason": "not_micro_merge",
            "parent_cluster_id": None,
            "child_cluster_id": None,
            "micro_checks": 0,
            "flags": [],
            "projected_ka_share": _ka_raw(cluster_a, cluster_b),
        }

    if cluster_a_micro and cluster_b_micro:
        return {
            "accepted": False,
            "reason": "reject_singleton_chain",
            "parent_cluster_id": None,
            "child_cluster_id": None,
            "micro_checks": 0,
            "flags": ["reject_singleton_chain"],
            "projected_ka_share": _ka_raw(cluster_a, cluster_b),
        }

    child_cluster = cluster_a if cluster_a_micro else cluster_b
    parent_cluster = cluster_b if cluster_a_micro else cluster_a
    projected_ka_share = _ka_raw(cluster_a, cluster_b)
    flags: list[str] = []

    if not _is_previewable_parent(parent_cluster):
        flags.append("reject_parent_not_previewable")

    dominant_ka_match = bool(
        child_cluster.get("dominant_ka")
        and parent_cluster.get("dominant_ka")
        and child_cluster["dominant_ka"] == parent_cluster["dominant_ka"]
    )
    if not dominant_ka_match:
        flags.append("reject_dominant_ka_mismatch")

    micro_checks = sum(
        (
            1 if candidate["association_raw"] >= THREAD_REORG_ASSOC_FLOOR else 0,
            1 if (candidate["semantic_raw"] or 0.0) >= MICRO_SEMANTIC_FLOOR else 0,
            1 if temporal_norm >= MICRO_TEMPORAL_FLOOR else 0,
        )
    )
    trace_overlap = int(candidate.get("trace_overlap") or 0)
    trace_temporal_support = trace_overlap > 0 and candidate["temporal_raw"] >= TEMPORAL_CANDIDATE_FLOOR
    if micro_checks < 2 and not trace_temporal_support:
        flags.append("reject_insufficient_strong_signals")

    projected_size = cluster_a["size"] + cluster_b["size"]
    if projected_size > THREAD_PREVIEW_SIZE_CAP:
        flags.append("reject_preview_size_cap")
    if projected_ka_share < MICRO_KA_FLOOR:
        flags.append("reject_projected_ka_floor")

    accepted = not flags
    return {
        "accepted": accepted,
        "reason": (
            "accepted_trace_overlap_merge"
            if accepted and trace_temporal_support and micro_checks < 2
            else "accepted_strict_parent_merge"
            if accepted
            else flags[0]
        ),
        "parent_cluster_id": parent_cluster["cluster_id"],
        "child_cluster_id": child_cluster["cluster_id"],
        "micro_checks": micro_checks,
        "flags": flags,
        "projected_ka_share": projected_ka_share,
    }


def _cluster_pair_key(cluster_a_id: str, cluster_b_id: str) -> tuple[str, str]:
    return tuple(sorted((cluster_a_id, cluster_b_id)))


def _cluster_pair_metrics(
    clusters: list[dict[str, Any]],
    graph: dict[str, dict[str, float]],
) -> dict[tuple[str, str], dict[str, float]]:
    concept_to_cluster: dict[str, str] = {}
    for cluster in clusters:
        for concept_id in cluster["concept_ids"]:
            concept_to_cluster[concept_id] = cluster["cluster_id"]

    pair_metrics: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"association_sum": 0.0, "association_count": 0.0}
    )
    for source, neighbors in graph.items():
        cluster_a_id = concept_to_cluster.get(source)
        if cluster_a_id is None:
            continue
        for target, strength in neighbors.items():
            if source >= target:
                continue
            cluster_b_id = concept_to_cluster.get(target)
            if cluster_b_id is None or cluster_a_id == cluster_b_id:
                continue
            key = _cluster_pair_key(cluster_a_id, cluster_b_id)
            pair_metrics[key]["association_sum"] += float(strength or 0.0)
            pair_metrics[key]["association_count"] += 1.0
    return pair_metrics


def _prefilter_neighbor_sets(
    clusters: list[dict[str, Any]],
    pair_metrics: dict[tuple[str, str], dict[str, float]],
) -> dict[str, set[str]]:
    neighbors: dict[str, set[str]] = defaultdict(set)

    for cluster_a_id, cluster_b_id in pair_metrics:
        neighbors[cluster_a_id].add(cluster_b_id)
        neighbors[cluster_b_id].add(cluster_a_id)

    dominant_ka_groups: dict[str, list[str]] = defaultdict(list)
    session_groups: dict[str, set[str]] = defaultdict(set)
    for cluster in clusters:
        cluster_id = cluster["cluster_id"]
        dominant_ka = cluster.get("dominant_ka")
        if dominant_ka:
            dominant_ka_groups[dominant_ka].append(cluster_id)
        for session_id in cluster["session_ids"]:
            session_groups[session_id].add(cluster_id)

    for cluster_ids in dominant_ka_groups.values():
        for cluster_a_id, cluster_b_id in combinations(sorted(cluster_ids), 2):
            neighbors[cluster_a_id].add(cluster_b_id)
            neighbors[cluster_b_id].add(cluster_a_id)

    for cluster_ids in session_groups.values():
        ordered = sorted(cluster_ids)
        for cluster_a_id, cluster_b_id in combinations(ordered, 2):
            neighbors[cluster_a_id].add(cluster_b_id)
            neighbors[cluster_b_id].add(cluster_a_id)

    return neighbors


def _candidate_payload(
    cluster_a: dict[str, Any],
    cluster_b: dict[str, Any],
    pair_metrics: dict[tuple[str, str], dict[str, float]],
) -> dict[str, Any]:
    pair_key = _cluster_pair_key(cluster_a["cluster_id"], cluster_b["cluster_id"])
    pair_metric = pair_metrics.get(pair_key, {})
    association_edges = int(pair_metric.get("association_count", 0.0))
    association_raw = 0.0
    if association_edges > 0:
        association_raw = float(pair_metric["association_sum"]) / association_edges
    temporal_raw = _temporal_overlap(cluster_a, cluster_b)
    semantic_raw, semantic_available = _semantic_similarity(cluster_a, cluster_b)
    ka_raw = _ka_raw(cluster_a, cluster_b)
    trace_overlap = len(cluster_a.get("trace_ids", set()) & cluster_b.get("trace_ids", set()))
    return {
        "cluster_a": cluster_a,
        "cluster_b": cluster_b,
        "association_raw": round(association_raw, 4),
        "temporal_raw": round(temporal_raw, 4),
        "semantic_raw": round(semantic_raw, 4) if semantic_raw is not None else None,
        "semantic_available": semantic_available,
        "ka_raw": round(ka_raw, 4),
        "association_edges": association_edges,
        "trace_overlap": trace_overlap,
    }


def _rank_shortlist_pairs(
    base_cluster_id: str,
    neighbor_ids: set[str],
    candidate_cache: dict[tuple[str, str], dict[str, Any]],
) -> set[tuple[str, str]]:
    if not neighbor_ids:
        return set()

    pair_keys = [_cluster_pair_key(base_cluster_id, neighbor_id) for neighbor_id in neighbor_ids]

    def _semantic_sort_key(pair_key: tuple[str, str]) -> tuple[float, float, float, str]:
        candidate = candidate_cache[pair_key]
        semantic_raw = candidate["semantic_raw"]
        return (
            semantic_raw if semantic_raw is not None else -1.0,
            candidate["association_raw"],
            candidate["temporal_raw"],
            pair_key[1] if pair_key[0] == base_cluster_id else pair_key[0],
        )

    def _temporal_sort_key(pair_key: tuple[str, str]) -> tuple[float, float, float, str]:
        candidate = candidate_cache[pair_key]
        return (
            candidate["temporal_raw"],
            candidate["association_raw"],
            candidate["ka_raw"],
            pair_key[1] if pair_key[0] == base_cluster_id else pair_key[0],
        )

    ranked_by_semantic = sorted(pair_keys, key=_semantic_sort_key, reverse=True)[:SHORTLIST_LIMIT_PER_SIGNAL]
    ranked_by_temporal = sorted(pair_keys, key=_temporal_sort_key, reverse=True)[:SHORTLIST_LIMIT_PER_SIGNAL]
    return set(ranked_by_semantic) | set(ranked_by_temporal)


def _candidate_pairs(
    clusters: list[dict[str, Any]],
    graph: dict[str, dict[str, float]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]], dict[str, int]]:
    total_pairs = len(clusters) * (len(clusters) - 1) // 2
    if len(clusters) < 2:
        return [], {
            "association": {"p25": 0.0, "p75": 0.0},
            "temporal": {"p25": 0.0, "p75": 0.0},
            "semantic": {"p25": 0.0, "p75": 0.0},
        }, {
            "candidate_pairs_total": total_pairs,
            "candidate_pairs_prefiltered": 0,
            "candidate_pairs_shortlisted": 0,
            "candidate_pairs_pruned": total_pairs,
        }

    cluster_map = {cluster["cluster_id"]: cluster for cluster in clusters}
    pair_metrics = _cluster_pair_metrics(clusters, graph)
    neighbor_sets = _prefilter_neighbor_sets(clusters, pair_metrics)
    candidate_cache: dict[tuple[str, str], dict[str, Any]] = {}
    for cluster in clusters:
        cluster_id = cluster["cluster_id"]
        for neighbor_id in neighbor_sets.get(cluster_id, set()):
            pair_key = _cluster_pair_key(cluster_id, neighbor_id)
            if pair_key in candidate_cache:
                continue
            cluster_a = cluster_map[pair_key[0]]
            cluster_b = cluster_map[pair_key[1]]
            candidate_cache[pair_key] = _candidate_payload(cluster_a, cluster_b, pair_metrics)

    shortlisted_pairs: set[tuple[str, str]] = set()
    for cluster in clusters:
        shortlisted_pairs |= _rank_shortlist_pairs(
            cluster["cluster_id"],
            neighbor_sets.get(cluster["cluster_id"], set()),
            candidate_cache,
        )

    raw_candidates = [candidate_cache[pair_key] for pair_key in sorted(shortlisted_pairs)]

    percentiles = {
        "association": dict(zip(("p25", "p75"), _percentile_pair([c["association_raw"] for c in raw_candidates]), strict=False)),
        "temporal": dict(zip(("p25", "p75"), _percentile_pair([c["temporal_raw"] for c in raw_candidates]), strict=False)),
        "semantic": dict(
            zip(
                ("p25", "p75"),
                _percentile_pair([c["semantic_raw"] for c in raw_candidates if c["semantic_raw"] is not None]),
                strict=False,
                )
        ),
    }
    candidate_stats = {
        "candidate_pairs_total": total_pairs,
        "candidate_pairs_prefiltered": len(candidate_cache),
        "candidate_pairs_shortlisted": len(raw_candidates),
        "candidate_pairs_pruned": max(total_pairs - len(raw_candidates), 0),
    }
    return raw_candidates, percentiles, candidate_stats


def _scored_candidates(
    raw_candidates: list[dict[str, Any]],
    percentiles: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for candidate in raw_candidates:
        association_norm = _normalize_signal(
            candidate["association_raw"],
            percentiles["association"]["p25"],
            percentiles["association"]["p75"],
        )
        temporal_norm = _normalize_signal(
            candidate["temporal_raw"],
            percentiles["temporal"]["p25"],
            percentiles["temporal"]["p75"],
        )
        semantic_norm = _normalize_signal(
            candidate["semantic_raw"],
            percentiles["semantic"]["p25"],
            percentiles["semantic"]["p75"],
        )
        ka_norm = candidate["ka_raw"]

        score = (0.35 * association_norm) + (0.25 * temporal_norm) + (0.25 * semantic_norm) + (0.15 * ka_norm)
        micro_merge = _is_micro_cluster(candidate["cluster_a"]) or _is_micro_cluster(candidate["cluster_b"])
        strict_parent_merge = _strict_parent_merge_decision(candidate, temporal_norm)
        micro_checks = strict_parent_merge["micro_checks"]
        accepted = strict_parent_merge["accepted"] if micro_merge else score >= STANDARD_MERGE_FLOOR
        merged_size = candidate["cluster_a"]["size"] + candidate["cluster_b"]["size"]
        combined_ka_norm = strict_parent_merge["projected_ka_share"]
        flags = list(strict_parent_merge["flags"])
        if merged_size > 300:
            flags.append("split_review_size")
        if combined_ka_norm < 0.70:
            flags.append("split_review_ka")

        decisions.append(
            {
                **candidate,
                "association_norm": round(association_norm, 4),
                "temporal_norm": round(temporal_norm, 4),
                "semantic_norm": round(semantic_norm, 4),
                "ka_norm": round(ka_norm, 4),
                "hybrid_merge_score": round(score, 4),
                "micro_merge": micro_merge,
                "micro_checks": micro_checks,
                "merge_reason": strict_parent_merge["reason"] if micro_merge else (
                    "accepted_merge" if accepted else "rejected_merge_floor"
                ),
                "strict_parent_merge": micro_merge,
                "parent_cluster_id": strict_parent_merge["parent_cluster_id"],
                "child_cluster_id": strict_parent_merge["child_cluster_id"],
                "accepted": accepted,
                "flags": flags,
            }
        )
    decisions.sort(key=lambda item: item["hybrid_merge_score"], reverse=True)
    return decisions


def _merge_clusters(
    cluster_a: dict[str, Any],
    cluster_b: dict[str, Any],
    concept_map: dict[str, dict[str, Any]],
    embedding_map: dict[str, np.ndarray],
) -> dict[str, Any]:
    merged_ids = sorted(set(cluster_a["concept_ids"]) | set(cluster_b["concept_ids"]))
    merged = _cluster_metadata(merged_ids, concept_map, embedding_map)
    merged["merged_from"] = [cluster_a["cluster_id"], cluster_b["cluster_id"]]
    return merged


def _absorb_micro_clusters(
    clusters: list[dict[str, Any]],
    concept_map: dict[str, dict[str, Any]],
    embedding_map: dict[str, np.ndarray],
    graph: dict[str, dict[str, float]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    initial_micro_clusters = [cluster for cluster in clusters if _is_micro_cluster(cluster)]
    aggregate_stats = {
        "micro_clusters_initial": len(initial_micro_clusters),
        "micro_clusters_with_parent_candidate": 0,
        "micro_absorption_candidate_pairs": 0,
        "micro_absorption_accepted_merges": 0,
        "micro_trace_overlap_candidate_pairs": 0,
        "micro_trace_overlap_accepted_merges": 0,
        "micro_absorption_rejected_pairs": 0,
        "micro_absorption_passes": 0,
        "micro_clusters_absorbed": 0,
        "micro_clusters_remaining": len(initial_micro_clusters),
    }
    if not initial_micro_clusters:
        return clusters, [], aggregate_stats

    candidate_micro_cluster_ids: set[str] = set()
    absorption_decisions: list[dict[str, Any]] = []

    while True:
        pair_metrics = _cluster_pair_metrics(clusters, graph)
        parent_clusters = [cluster for cluster in clusters if _is_previewable_parent(cluster)]
        micro_clusters = [cluster for cluster in clusters if _is_micro_cluster(cluster)]
        if not parent_clusters or not micro_clusters:
            break

        raw_candidates: list[dict[str, Any]] = []
        round_candidate_micro_ids: set[str] = set()
        for child_cluster in sorted(micro_clusters, key=lambda item: (-item["size"], item["cluster_id"])):
            for parent_cluster in parent_clusters:
                if child_cluster["cluster_id"] == parent_cluster["cluster_id"]:
                    continue
                if not child_cluster.get("dominant_ka") or child_cluster["dominant_ka"] != parent_cluster.get("dominant_ka"):
                    continue
                raw_candidates.append(_candidate_payload(child_cluster, parent_cluster, pair_metrics))
                round_candidate_micro_ids.add(child_cluster["cluster_id"])

        if not raw_candidates:
            break

        aggregate_stats["micro_absorption_passes"] += 1
        aggregate_stats["micro_absorption_candidate_pairs"] += len(raw_candidates)
        aggregate_stats["micro_trace_overlap_candidate_pairs"] += sum(
            1 for candidate in raw_candidates if candidate["trace_overlap"] > 0
        )
        candidate_micro_cluster_ids |= round_candidate_micro_ids

        percentiles = {
            "association": dict(
                zip(("p25", "p75"), _percentile_pair([c["association_raw"] for c in raw_candidates]), strict=False)
            ),
            "temporal": dict(
                zip(("p25", "p75"), _percentile_pair([c["temporal_raw"] for c in raw_candidates]), strict=False)
            ),
            "semantic": dict(
                zip(
                    ("p25", "p75"),
                    _percentile_pair([c["semantic_raw"] for c in raw_candidates if c["semantic_raw"] is not None]),
                    strict=False,
                )
            ),
        }
        scored = _scored_candidates(raw_candidates, percentiles)
        aggregate_stats["micro_absorption_rejected_pairs"] += sum(1 for item in scored if not item["accepted"])
        aggregate_stats["micro_trace_overlap_accepted_merges"] += sum(
            1 for item in scored if item["merge_reason"] == "accepted_trace_overlap_merge"
        )
        absorption_decisions.extend(
            {
                "cluster_a": item["cluster_a"]["cluster_id"],
                "cluster_b": item["cluster_b"]["cluster_id"],
                "accepted": item["accepted"],
                "reason": item["merge_reason"],
                "hybrid_merge_score": item["hybrid_merge_score"],
                "micro_merge": item["micro_merge"],
                "micro_checks": item["micro_checks"],
                "strict_parent_merge": item["strict_parent_merge"],
                "parent_cluster_id": item["parent_cluster_id"],
                "child_cluster_id": item["child_cluster_id"],
                "flags": item["flags"],
                "association_raw": item["association_raw"],
                "temporal_raw": item["temporal_raw"],
                "semantic_raw": item["semantic_raw"],
                "trace_overlap": item["trace_overlap"],
                "ka_raw": item["ka_raw"],
                "phase": "micro_absorption",
            }
            for item in scored
        )

        best = next((item for item in scored if item["accepted"]), None)
        if best is None:
            break

        merged = _merge_clusters(best["cluster_a"], best["cluster_b"], concept_map, embedding_map)
        clusters = [
            cluster
            for cluster in clusters
            if cluster["cluster_id"] not in {best["cluster_a"]["cluster_id"], best["cluster_b"]["cluster_id"]}
        ]
        clusters.append(merged)
        aggregate_stats["micro_absorption_accepted_merges"] += 1

    aggregate_stats["micro_clusters_with_parent_candidate"] = len(candidate_micro_cluster_ids)
    aggregate_stats["micro_clusters_remaining"] = sum(1 for cluster in clusters if _is_micro_cluster(cluster))
    aggregate_stats["micro_clusters_absorbed"] = (
        aggregate_stats["micro_clusters_initial"] - aggregate_stats["micro_clusters_remaining"]
    )
    return clusters, absorption_decisions, aggregate_stats


def _merge_candidate_cluster_set(
    clusters: list[dict[str, Any]],
    concept_map: dict[str, dict[str, Any]],
    embedding_map: dict[str, np.ndarray],
    graph: dict[str, dict[str, float]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, float]], dict[str, int]]:
    merge_decisions: list[dict[str, Any]] = []
    percentile_snapshot: dict[str, dict[str, float]] = {
        "association": {"p25": 0.0, "p75": 0.0},
        "temporal": {"p25": 0.0, "p75": 0.0},
        "semantic": {"p25": 0.0, "p75": 0.0},
    }
    aggregate_candidate_stats = {
        "candidate_pairs_total": 0,
        "candidate_pairs_prefiltered": 0,
        "candidate_pairs_shortlisted": 0,
        "candidate_pairs_pruned": 0,
    }

    while clusters and len(clusters) > TARGET_CLUSTER_MAX:
        raw_candidates, percentile_snapshot, candidate_stats = _candidate_pairs(clusters, graph)
        for key, value in candidate_stats.items():
            aggregate_candidate_stats[key] += value

        scored = _scored_candidates(raw_candidates, percentile_snapshot)
        merge_decisions.extend(
            {
                "cluster_a": item["cluster_a"]["cluster_id"],
                "cluster_b": item["cluster_b"]["cluster_id"],
                "accepted": item["accepted"],
                "reason": item["merge_reason"],
                "hybrid_merge_score": item["hybrid_merge_score"],
                "micro_merge": item["micro_merge"],
                "micro_checks": item["micro_checks"],
                "strict_parent_merge": item["strict_parent_merge"],
                "parent_cluster_id": item["parent_cluster_id"],
                "child_cluster_id": item["child_cluster_id"],
                "flags": item["flags"],
                "association_raw": item["association_raw"],
                "temporal_raw": item["temporal_raw"],
                "semantic_raw": item["semantic_raw"],
                "trace_overlap": item["trace_overlap"],
                "ka_raw": item["ka_raw"],
            }
            for item in scored
        )
        best = next((item for item in scored if item["accepted"]), None)
        if best is None:
            break

        merged = _merge_clusters(best["cluster_a"], best["cluster_b"], concept_map, embedding_map)
        next_clusters = [
            cluster
            for cluster in clusters
            if cluster["cluster_id"] not in {best["cluster_a"]["cluster_id"], best["cluster_b"]["cluster_id"]}
        ]
        next_clusters.append(merged)
        clusters = next_clusters

        if TARGET_CLUSTER_MIN <= len(clusters) <= TARGET_CLUSTER_MAX:
            break

    return clusters, merge_decisions, percentile_snapshot, aggregate_candidate_stats


def _strong_internal_components(cluster: dict[str, Any], graph: dict[str, dict[str, float]]) -> list[list[str]]:
    concept_ids = cluster["concept_ids"]
    concept_id_set = set(concept_ids)
    internal_edges: list[float] = []
    strong_graph: dict[str, dict[str, float]] = {concept_id: {} for concept_id in concept_ids}
    for source in concept_ids:
        for target, strength in graph.get(source, {}).items():
            if target not in concept_id_set or source >= target:
                continue
            internal_edges.append(float(strength or 0.0))

    if len(internal_edges) < 2:
        return []

    threshold = float(np.percentile(np.asarray(internal_edges, dtype=float), 75))
    if threshold <= 0:
        return []

    for source in concept_ids:
        for target, strength in graph.get(source, {}).items():
            if target not in concept_id_set or strength < threshold:
                continue
            strong_graph[source][target] = float(strength)
            strong_graph[target][source] = float(strength)

    components = _connected_components(concept_ids, strong_graph)
    if len(components) <= 1:
        return []
    return [component for component in components if component]


def _partition_cluster(
    cluster: dict[str, Any],
    concept_map: dict[str, dict[str, Any]],
    graph: dict[str, dict[str, float]],
) -> tuple[list[list[str]], str | None]:
    strong_components = _strong_internal_components(cluster, graph)
    if strong_components:
        return strong_components, "strong_internal_components"

    ka_groups: dict[str, list[str]] = defaultdict(list)
    for concept_id in cluster["concept_ids"]:
        knowledge_area = concept_map[concept_id].get("knowledge_area") or "__unknown__"
        ka_groups[knowledge_area].append(concept_id)
    if len(ka_groups) > 1:
        return [sorted(group) for group in ka_groups.values() if group], "knowledge_area_partition"

    return [], None


def _needs_split(cluster: dict[str, Any]) -> bool:
    return cluster["size"] > THREAD_PREVIEW_SIZE_CAP or cluster["dominant_ka_share"] < MICRO_KA_FLOOR


def _deferred_singleton_record(cluster: dict[str, Any]) -> dict[str, Any]:
    return {
        "cluster_id": cluster["cluster_id"],
        "concept_ids": cluster["concept_ids"],
        "size": cluster["size"],
        "dominant_ka": cluster["dominant_ka"],
        "dominant_ka_share": cluster["dominant_ka_share"],
        "reason": "deferred_singleton_tail",
    }


def _deferred_review_summary_exclusion(summary: str) -> str | None:
    normalized = (summary or "").strip()
    if len(normalized) < DEFERRED_REVIEW_MIN_SUMMARY_CHARS:
        return "short_summary"
    lowered = normalized.lower()
    if any(marker in lowered for marker in DEFERRED_REVIEW_CONVERSATIONAL_MARKERS):
        return "conversational_summary"
    return None


def _deferred_review_tokenize(summary: str) -> list[str]:
    return re.findall(r"[a-z][a-z0-9_/-]{3,}", (summary or "").lower())


def _filter_deferred_bucket_for_review(
    bucket: dict[str, Any],
    concept_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    dominant_ka = (bucket.get("dominant_ka") or "").strip()
    original_concept_ids = list(bucket.get("concept_ids") or [])
    kept_concept_ids: list[str] = []
    excluded_counts: Counter[str] = Counter()
    knowledge_area_counts: Counter[str] = Counter()

    for concept_id in original_concept_ids:
        record = concept_map.get(concept_id)
        if not record:
            excluded_counts["missing_record"] += 1
            continue

        knowledge_area = (record.get("knowledge_area") or "").strip()
        knowledge_area_counts[knowledge_area or "__unknown__"] += 1
        if dominant_ka and knowledge_area and knowledge_area != dominant_ka:
            excluded_counts["knowledge_area_mismatch"] += 1
            continue

        confidence = record.get("confidence")
        if confidence is not None and float(confidence) < DEFERRED_REVIEW_MIN_CONFIDENCE:
            excluded_counts["low_confidence"] += 1
            continue

        summary_exclusion = _deferred_review_summary_exclusion(record.get("summary") or "")
        if summary_exclusion:
            excluded_counts[summary_exclusion] += 1
            continue

        kept_concept_ids.append(concept_id)

    return {
        **bucket,
        "concept_ids": kept_concept_ids,
        "concept_count": len(kept_concept_ids),
        "original_concept_count": len(original_concept_ids),
        "knowledge_area_counts": dict(sorted(knowledge_area_counts.items())),
        "review_filters": {
            "min_confidence": DEFERRED_REVIEW_MIN_CONFIDENCE,
            "min_summary_chars": DEFERRED_REVIEW_MIN_SUMMARY_CHARS,
            "excluded_counts": dict(sorted(excluded_counts.items())),
        },
    }


def _balanced_chunk_sizes(total: int, cap: int) -> list[int]:
    if total <= 0:
        return []
    chunk_count = max(1, (total + cap - 1) // cap)
    base, remainder = divmod(total, chunk_count)
    return [base + (1 if idx < remainder else 0) for idx in range(chunk_count)]


def _pack_shaped_bucket(
    *,
    concept_ids: list[str],
    concept_map: dict[str, dict[str, Any]],
    base_bucket: dict[str, Any],
    subgroup_label: str,
    group_reason: str,
    group_anchor: str | None,
) -> list[dict[str, Any]]:
    if len(concept_ids) < 2:
        return []

    anchor_fragment = ""
    if group_anchor:
        anchor_fragment = re.sub(r"[^a-z0-9]+", "-", group_anchor.lower()).strip("-")[:24]
    subgroup_key = subgroup_label if not anchor_fragment or anchor_fragment == subgroup_label else f"{subgroup_label}-{anchor_fragment}"

    ordered_ids = sorted(
        concept_ids,
        key=lambda concept_id: (
            -float(concept_map[concept_id].get("confidence") or 0.0),
            concept_id,
        ),
    )
    packed: list[dict[str, Any]] = []
    start = 0
    for chunk_index, size in enumerate(_balanced_chunk_sizes(len(ordered_ids), DEFERRED_REVIEW_SHAPED_BUCKET_CAP), start=1):
        chunk_ids = ordered_ids[start : start + size]
        start += size
        packed.append(
            {
                **base_bucket,
                "bucket_id": f"{base_bucket['bucket_id']}--{subgroup_key}-{chunk_index:02d}",
                "parent_bucket_id": base_bucket["bucket_id"],
                "shape_reason": group_reason,
                "shape_anchor": group_anchor,
                "concept_ids": chunk_ids,
                "concept_count": len(chunk_ids),
                "original_concept_count": len(chunk_ids),
                "source_cluster_count": len(chunk_ids),
                "source_cluster_ids": [],
            }
        )
    return packed


def _shape_filtered_bucket_for_review(
    bucket: dict[str, Any],
    concept_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    filtered_ids = list(bucket.get("concept_ids") or [])
    dominant_ka = (bucket.get("dominant_ka") or "").strip()
    if len(filtered_ids) <= DEFERRED_REVIEW_SHAPED_BUCKET_CAP:
        return {
            "parent_bucket_id": bucket["bucket_id"],
            "shaped_buckets": [bucket] if filtered_ids else [],
            "overflow_concept_ids": [],
            "shape_strategy": "none",
        }

    shaped_buckets: list[dict[str, Any]] = []
    assigned_ids: set[str] = set()

    # Provenance-first: shared trace id, then shared session id.
    trace_groups: dict[str, list[str]] = defaultdict(list)
    session_groups: dict[str, list[str]] = defaultdict(list)
    for concept_id in filtered_ids:
        record = concept_map[concept_id]
        trace_id = (record.get("source_trace_id") or "").strip()
        session_id = (record.get("session_id") or "").strip()
        if trace_id:
            trace_groups[trace_id].append(concept_id)
        if session_id:
            session_groups[session_id].append(concept_id)

    for trace_id in sorted(trace_groups):
        group_ids = [concept_id for concept_id in trace_groups[trace_id] if concept_id not in assigned_ids]
        if len(group_ids) < 2:
            continue
        shaped_buckets.extend(
            _pack_shaped_bucket(
                concept_ids=group_ids,
                concept_map=concept_map,
                base_bucket=bucket,
                subgroup_label="trace",
                group_reason="shared_source_trace_id",
                group_anchor=trace_id,
            )
        )
        assigned_ids.update(group_ids)

    for session_id in sorted(session_groups):
        group_ids = [concept_id for concept_id in session_groups[session_id] if concept_id not in assigned_ids]
        if len(group_ids) < 2:
            continue
        shaped_buckets.extend(
            _pack_shaped_bucket(
                concept_ids=group_ids,
                concept_map=concept_map,
                base_bucket=bucket,
                subgroup_label="session",
                group_reason="shared_session_id",
                group_anchor=session_id,
            )
        )
        assigned_ids.update(group_ids)

    family_definitions = DEFERRED_REVIEW_TOPIC_FAMILIES.get(dominant_ka, {})
    topic_groups: dict[str, list[str]] = defaultdict(list)
    for concept_id in filtered_ids:
        if concept_id in assigned_ids:
            continue
        tokens = _deferred_review_tokenize(concept_map[concept_id].get("summary") or "")
        family_scores: dict[str, int] = {}
        for family_name, family_tokens in family_definitions.items():
            score = sum(1 for token in tokens if token in family_tokens)
            if score > 0:
                family_scores[family_name] = score
        if not family_scores:
            continue
        ordered_families = sorted(family_scores.items(), key=lambda item: (-item[1], item[0]))
        if len(ordered_families) > 1 and ordered_families[0][1] == ordered_families[1][1]:
            continue
        topic_groups[ordered_families[0][0]].append(concept_id)

    for family_name in sorted(topic_groups):
        group_ids = [concept_id for concept_id in topic_groups[family_name] if concept_id not in assigned_ids]
        if len(group_ids) < 2:
            continue
        shaped_buckets.extend(
            _pack_shaped_bucket(
                concept_ids=group_ids,
                concept_map=concept_map,
                base_bucket=bucket,
                subgroup_label=family_name,
                group_reason="topic_signature",
                group_anchor=family_name,
            )
        )
        assigned_ids.update(group_ids)

    overflow_ids = sorted(concept_id for concept_id in filtered_ids if concept_id not in assigned_ids)
    return {
        "parent_bucket_id": bucket["bucket_id"],
        "shaped_buckets": shaped_buckets,
        "overflow_concept_ids": overflow_ids,
        "shape_strategy": "provenance_then_topic_signature",
    }


def _consolidate_deferred_singletons(deferred_singletons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not deferred_singletons:
        return []

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in deferred_singletons:
        grouped[record.get("dominant_ka") or "__unknown__"].append(record)

    buckets: list[dict[str, Any]] = []
    for dominant_ka, records in grouped.items():
        ordered = sorted(records, key=lambda item: (-item["size"], item["cluster_id"]))
        bucket_records: list[dict[str, Any]] = []
        bucket_concepts: list[str] = []
        bucket_index = 1
        for record in ordered:
            next_size = len(bucket_concepts) + len(record["concept_ids"])
            if bucket_records and next_size > DEFERRED_SINGLETON_BUCKET_SIZE:
                buckets.append(
                    {
                        "bucket_id": f"deferred-{dominant_ka}-{bucket_index:02d}",
                        "dominant_ka": dominant_ka if dominant_ka != "__unknown__" else "",
                        "concept_ids": sorted(bucket_concepts),
                        "concept_count": len(bucket_concepts),
                        "source_cluster_count": len(bucket_records),
                        "source_cluster_ids": [item["cluster_id"] for item in bucket_records],
                        "reason": "deferred_singleton_bucket",
                    }
                )
                bucket_index += 1
                bucket_records = []
                bucket_concepts = []

            bucket_records.append(record)
            bucket_concepts.extend(record["concept_ids"])

        if bucket_records:
            buckets.append(
                {
                    "bucket_id": f"deferred-{dominant_ka}-{bucket_index:02d}",
                    "dominant_ka": dominant_ka if dominant_ka != "__unknown__" else "",
                    "concept_ids": sorted(bucket_concepts),
                    "concept_count": len(bucket_concepts),
                    "source_cluster_count": len(bucket_records),
                    "source_cluster_ids": [item["cluster_id"] for item in bucket_records],
                    "reason": "deferred_singleton_bucket",
                }
            )

    buckets.sort(key=lambda item: (-item["concept_count"], item["bucket_id"]))
    return buckets


def _resolve_preview_clusters(
    clusters: list[dict[str, Any]],
    concept_map: dict[str, dict[str, Any]],
    embedding_map: dict[str, np.ndarray],
    graph: dict[str, dict[str, float]],
    *,
    depth: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    preview_clusters: list[dict[str, Any]] = []
    deferred_singletons: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []

    for cluster in sorted(clusters, key=lambda item: item["size"], reverse=True):
        if cluster["size"] < MIN_PREVIEW_CLUSTER_SIZE:
            deferred_singletons.append(_deferred_singleton_record(cluster))
            artifact_rows.append(
                {
                    "cluster_a": cluster["cluster_id"],
                    "cluster_b": None,
                    "accepted": False,
                    "reason": "deferred_singleton_tail",
                    "hybrid_merge_score": None,
                    "micro_merge": True,
                    "micro_checks": None,
                    "flags": ["deferred_singleton_tail"],
                    "association_raw": None,
                    "temporal_raw": None,
                    "semantic_raw": None,
                    "ka_raw": cluster["dominant_ka_share"],
                }
            )
            continue

        if _needs_split(cluster) and depth < MAX_SPLIT_DEPTH:
            partitions, partition_reason = _partition_cluster(cluster, concept_map, graph)
            if len(partitions) > 1:
                artifact_rows.append(
                    {
                        "cluster_a": cluster["cluster_id"],
                        "cluster_b": None,
                        "accepted": True,
                        "reason": "split_partition_applied",
                        "hybrid_merge_score": None,
                        "micro_merge": False,
                        "micro_checks": None,
                        "flags": [partition_reason] if partition_reason else [],
                        "association_raw": None,
                        "temporal_raw": None,
                        "semantic_raw": None,
                        "ka_raw": cluster["dominant_ka_share"],
                    }
                )
                partition_clusters = [
                    _cluster_metadata(sorted(partition), concept_map, embedding_map) for partition in partitions if partition
                ]
                for partition_cluster in partition_clusters:
                    partition_cluster["split_from"] = cluster["cluster_id"]
                split_preview, split_deferred, split_rows = _resolve_preview_clusters(
                    partition_clusters,
                    concept_map,
                    embedding_map,
                    graph,
                    depth=depth + 1,
                )
                preview_clusters.extend(split_preview)
                deferred_singletons.extend(split_deferred)
                artifact_rows.extend(split_rows)
                continue

        cluster["preview_eligible"] = not _needs_split(cluster)
        if not cluster["preview_eligible"]:
            cluster["residual_reason"] = (
                "split_unresolved_size"
                if cluster["size"] > THREAD_PREVIEW_SIZE_CAP
                else "split_unresolved_ka"
            )
            artifact_rows.append(
                {
                    "cluster_a": cluster["cluster_id"],
                    "cluster_b": None,
                    "accepted": False,
                    "reason": cluster["residual_reason"],
                    "hybrid_merge_score": None,
                    "micro_merge": False,
                    "micro_checks": None,
                    "flags": [cluster["residual_reason"]],
                    "association_raw": None,
                    "temporal_raw": None,
                    "semantic_raw": None,
                    "ka_raw": cluster["dominant_ka_share"],
                }
            )
        else:
            cluster["residual_reason"] = None
        preview_clusters.append(cluster)

    return preview_clusters, deferred_singletons, artifact_rows


def _augment_candidate_stats(
    candidate_stats: dict[str, Any],
    clusters: list[dict[str, Any]],
) -> dict[str, Any]:
    residual_clusters = [cluster for cluster in clusters if not cluster.get("preview_eligible", True)]
    previewable_clusters = [cluster for cluster in clusters if cluster.get("preview_eligible", True)]

    total_pairs = int(candidate_stats.get("candidate_pairs_total", 0) or 0)
    pruned_pairs = int(candidate_stats.get("candidate_pairs_pruned", 0) or 0)
    if total_pairs > 0:
        candidate_stats["candidate_pairs_pruning_ratio"] = round(pruned_pairs / total_pairs, 4)
    else:
        candidate_stats["candidate_pairs_pruning_ratio"] = 0.0

    candidate_stats["residual_non_previewable_cluster_count"] = len(residual_clusters)
    candidate_stats["residual_max_cluster_size"] = max((cluster["size"] for cluster in residual_clusters), default=0)
    candidate_stats["max_previewable_cluster_size"] = max((cluster["size"] for cluster in previewable_clusters), default=0)
    return candidate_stats


def mine_active_thread(source_thread_id: str | None = None) -> ThreadReorgMiningResult:
    """Mine the current active thread into reorg-ready cluster proposals."""
    thread = _fetch_active_thread(source_thread_id)
    concept_map, stale_concept_ids = _load_thread_concepts(thread)
    concept_ids = set(concept_map)
    graph, _ = _load_association_graph(concept_ids)
    embedding_map = _load_embedding_map(concept_ids)

    components = _connected_components(sorted(concept_ids), graph) if concept_ids else []
    clusters = [_cluster_metadata(component, concept_map, embedding_map) for component in components]
    clusters, merge_decisions, percentile_snapshot, candidate_stats = _merge_candidate_cluster_set(
        clusters,
        concept_map,
        embedding_map,
        graph,
    )
    clusters, absorption_decisions, absorption_stats = _absorb_micro_clusters(
        clusters,
        concept_map,
        embedding_map,
        graph,
    )
    merge_decisions.extend(absorption_decisions)
    candidate_stats.update(absorption_stats)
    clusters, deferred_singletons, split_rows = _resolve_preview_clusters(
        clusters,
        concept_map,
        embedding_map,
        graph,
    )
    merge_decisions.extend(split_rows)
    deferred_buckets = _consolidate_deferred_singletons(deferred_singletons)
    candidate_stats["raw_deferred_singletons"] = len(deferred_singletons)
    candidate_stats["deferred_bucket_count"] = len(deferred_buckets)
    candidate_stats = _augment_candidate_stats(candidate_stats, clusters)

    serializable_clusters = []
    for cluster in sorted(clusters, key=lambda item: item["size"], reverse=True):
        serializable_clusters.append(
            {
                "cluster_id": cluster["cluster_id"],
                "concept_ids": cluster["concept_ids"],
                "size": cluster["size"],
                "dominant_ka": cluster["dominant_ka"],
                "dominant_ka_share": cluster["dominant_ka_share"],
                "ka_counts": cluster["ka_counts"],
                "session_count": len(cluster["session_ids"]),
                "trace_count": len(cluster.get("trace_ids", set())),
                "summary_preview": cluster["summary_preview"],
                "embedding_coverage": cluster["embedding_coverage"],
                "merged_from": cluster.get("merged_from", []),
                "split_from": cluster.get("split_from"),
                "preview_eligible": cluster.get("preview_eligible", True),
                "residual_reason": cluster.get("residual_reason"),
            }
        )

    return ThreadReorgMiningResult(
        source_thread_id=thread.id,
        source_thread_title=thread.title,
        resolved_count=len(concept_map),
        stale_concept_ids=sorted(stale_concept_ids),
        cluster_count=len(serializable_clusters),
        clusters=serializable_clusters,
        percentile_snapshot=percentile_snapshot,
        merge_decisions=merge_decisions,
        candidate_stats=candidate_stats,
        deferred_singletons=deferred_buckets,
    )


def _create_paused_thread(
    conn,
    *,
    source_thread_id: str,
    title: str,
    description: str,
    knowledge_areas: list[str],
    concept_ids: list[str] | None = None,
) -> NarrativeThread:
    now = _utc_now_iso()
    thread = NarrativeThread(
        id=str(uuid.uuid4()),
        title=title[:500],
        description=description[:500],
        status="paused",
        created_at=now,
        updated_at=now,
        last_activity_at=now,
        predecessor_id=source_thread_id,
        knowledge_areas=knowledge_areas,
        concept_ids=list(concept_ids or []),
    )
    conn.execute(
        """INSERT INTO threads (
               id, title, description, status, created_at, updated_at,
               last_activity_at, completed_at, urgency, agent_id, data
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            thread.id,
            thread.title,
            thread.description,
            thread.status,
            thread.created_at,
            thread.updated_at,
            thread.last_activity_at,
            thread.completed_at,
            thread.urgency,
            thread.agent_id,
            json.dumps(thread.model_dump()),
        ),
    )
    return thread


def queue_seed_candidate(concept, reason: str, notes: dict[str, Any] | None = None) -> None:
    """Queue a concept for later thread reorg seeding instead of force-fitting it."""
    if concept is None:
        return
    notes = notes or {}
    with _write_conn() as conn:
        conn.execute(
            """
            INSERT INTO thread_reorg_seed_candidates (
                concept_id, source_session_id, source_trace_id, knowledge_area,
                status, reason, notes_json, created_at, promoted_thread_id
            ) VALUES (?, ?, ?, ?, 'queued', ?, ?, ?, NULL)
            ON CONFLICT(concept_id) DO UPDATE SET
                source_session_id = excluded.source_session_id,
                source_trace_id = excluded.source_trace_id,
                knowledge_area = excluded.knowledge_area,
                status = CASE
                    WHEN thread_reorg_seed_candidates.status = 'dismissed'
                    THEN thread_reorg_seed_candidates.status
                    ELSE 'queued'
                END,
                reason = excluded.reason,
                notes_json = excluded.notes_json,
                created_at = excluded.created_at
            """,
            (
                concept.id,
                getattr(concept, "session_id", None),
                getattr(concept, "source_trace_id", None),
                getattr(concept, "knowledge_area", None),
                reason,
                json.dumps(notes),
                _utc_now_iso(),
            ),
        )


def _mark_seed_candidates_promoted(conn, concept_ids: list[str], target_thread_id: str) -> None:
    if not concept_ids:
        return
    placeholders = ",".join("?" for _ in concept_ids)
    conn.execute(
        f"""
        UPDATE thread_reorg_seed_candidates
        SET status = 'promoted',
            promoted_thread_id = ?
        WHERE concept_id IN ({placeholders})
        """,
        (target_thread_id, *concept_ids),
    )


def preview_batch(
    *,
    source_thread_id: str,
    clusters: list[dict[str, Any]],
    evaluation_set_id: str | None = None,
    max_batch_size: int | None = None,
    notes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stage an add-only batch preview for operator-approved cluster proposals."""
    source_thread = _fetch_active_thread(source_thread_id)
    batch_id = f"trb-{uuid.uuid4().hex[:12]}"
    max_batch_size = min(max_batch_size or THREAD_REORG_MAX_BATCH_SIZE, THREAD_REORG_MAX_BATCH_SIZE)
    created_thread_ids: list[str] = []
    staged_rows: list[tuple[Any, ...]] = []
    staged_concepts = 0
    validated_clusters: list[tuple[dict[str, Any], list[str]]] = []

    # Resolve concept existence before opening the write transaction.
    # load_concept() starts its own backend transaction, which collides with the
    # outer BEGIN IMMEDIATE in the live server path.
    for cluster in clusters:
        concept_ids = cluster.get("concept_ids") or []
        resolved_concept_ids = [
            concept_id for concept_id in concept_ids if load_concept(concept_id, track_access=False) is not None
        ]
        validated_clusters.append((cluster, resolved_concept_ids))

    with _write_conn() as conn:
        for cluster, resolved_concept_ids in validated_clusters:
            title = (cluster.get("title") or "").strip()
            description = cluster.get("description") or ""
            knowledge_areas = cluster.get("knowledge_areas") or []
            target_thread_id = cluster.get("target_thread_id")
            if not title or not knowledge_areas or not resolved_concept_ids:
                raise ValueError("Cluster preview requires title, knowledge_areas, and concept_ids")

            if target_thread_id:
                with _read_conn() as read_conn:
                    row = read_conn.execute("SELECT id FROM threads WHERE id = ?", (target_thread_id,)).fetchone()
                if not row:
                    raise ValueError(f"Target thread {target_thread_id} not found")
            else:
                target_thread = _create_paused_thread(
                    conn,
                    source_thread_id=source_thread.id,
                    title=title,
                    description=description,
                    knowledge_areas=knowledge_areas,
                    concept_ids=resolved_concept_ids,
                )
                target_thread_id = target_thread.id
                created_thread_ids.append(target_thread_id)

            for concept_id in resolved_concept_ids:
                if staged_concepts >= max_batch_size:
                    break
                staged_rows.append(
                    (
                        batch_id,
                        concept_id,
                        target_thread_id,
                        "add",
                        "staged",
                        cluster.get("role", "member"),
                        1 if concept_id in source_thread.concept_ids else 0,
                        json.dumps(
                            {
                                "cluster_title": title,
                                "cluster_description": description,
                                "knowledge_areas": knowledge_areas,
                            }
                        ),
                        _utc_now_iso(),
                    )
                )
                staged_concepts += 1
            if staged_concepts >= max_batch_size:
                break

        conn.execute(
            """
            INSERT INTO thread_reorg_batches (
                batch_id, source_thread_id, target_mode, status, planned_count,
                committed_count, detached_count, evaluation_set_id, notes_json, created_at
            ) VALUES (?, ?, 'preview', 'previewed', ?, 0, 0, ?, ?, ?)
            """,
            (
                batch_id,
                source_thread.id,
                len(staged_rows),
                evaluation_set_id,
                json.dumps(
                    {
                        "created_thread_ids": created_thread_ids,
                        "evaluation_primary_size": THREAD_REORG_EVAL_PRIMARY_SIZE,
                        "evaluation_control_size": THREAD_REORG_EVAL_CONTROL_SIZE,
                        "control_regression_caution": THREAD_REORG_CONTROL_REGRESSION_CAUTION,
                        **(notes or {}),
                    }
                ),
                _utc_now_iso(),
            ),
        )
        if staged_rows:
            conn.executemany(
                """
                INSERT INTO thread_reorg_batch_members (
                    batch_id, concept_id, target_thread_id, action, status, role,
                    legacy_membership_before, rationale_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                staged_rows,
            )

    return {
        "batch_id": batch_id,
        "source_thread_id": source_thread.id,
        "planned_count": len(staged_rows),
        "created_thread_ids": created_thread_ids,
        "status": "previewed",
    }


def _deferred_bucket_preview_cluster(
    bucket: dict[str, Any],
    source_thread: NarrativeThread,
) -> dict[str, Any]:
    dominant_ka = (bucket.get("dominant_ka") or "").strip()
    knowledge_areas = [dominant_ka] if dominant_ka else list(source_thread.knowledge_areas or [])[:1] or ["unknown"]
    ka_label = dominant_ka if dominant_ka else "unknown"
    return {
        "title": f"Residual Review: {ka_label} {bucket['bucket_id']}",
        "description": (
            f"Deferred singleton review bucket {bucket['bucket_id']} with {bucket['concept_count']} concepts "
            f"from {bucket['source_cluster_count']} residual source clusters."
        ),
        "knowledge_areas": knowledge_areas,
        "concept_ids": list(bucket.get("concept_ids") or []),
        "role": "member",
        "bucket_id": bucket["bucket_id"],
    }


def preview_deferred_review_batch(
    *,
    source_thread_id: str,
    bucket_ids: list[str] | None = None,
    evaluation_set_id: str | None = None,
    max_batch_size: int | None = None,
) -> dict[str, Any]:
    """Stage a preview batch directly from deferred singleton buckets for operator review."""
    source_thread = _fetch_active_thread(source_thread_id)
    result = mine_active_thread(source_thread_id)
    concept_map, _ = _load_thread_concepts(source_thread)
    available_buckets = result.deferred_singletons
    if not available_buckets:
        raise ValueError("No deferred singleton buckets available for review")

    bucket_map = {bucket["bucket_id"]: bucket for bucket in available_buckets}
    if bucket_ids:
        missing = [bucket_id for bucket_id in bucket_ids if bucket_id not in bucket_map]
        if missing:
            raise ValueError(f"Deferred bucket ids not found: {', '.join(missing)}")
        ordered_buckets = [bucket_map[bucket_id] for bucket_id in bucket_ids]
    else:
        ordered_buckets = list(available_buckets)

    max_batch_size = min(max_batch_size or THREAD_REORG_MAX_BATCH_SIZE, THREAD_REORG_MAX_BATCH_SIZE)
    filtered_buckets = [_filter_deferred_bucket_for_review(bucket, concept_map) for bucket in ordered_buckets]
    skipped_bucket_ids = [bucket["bucket_id"] for bucket in filtered_buckets if int(bucket.get("concept_count") or 0) <= 0]
    selected_buckets: list[dict[str, Any]] = []
    shaped_bucket_ids: list[str] = []
    selected_parent_bucket_ids: list[str] = []
    selected_count = 0
    for bucket in filtered_buckets:
        if int(bucket.get("concept_count") or 0) <= 0:
            continue
        shaped = _shape_filtered_bucket_for_review(bucket, concept_map)
        for shaped_bucket in shaped["shaped_buckets"]:
            bucket_size = int(shaped_bucket.get("concept_count") or 0)
            if bucket_size <= 0:
                continue
            if bucket_size > max_batch_size and not selected_buckets:
                raise ValueError(
                    f"Deferred shaped bucket {shaped_bucket['bucket_id']} exceeds max_batch_size={max_batch_size} with {bucket_size} concepts"
                )
            if selected_buckets and (selected_count + bucket_size) > max_batch_size:
                break
            selected_buckets.append(shaped_bucket)
            shaped_bucket_ids.append(shaped_bucket["bucket_id"])
            parent_bucket_id = shaped_bucket.get("parent_bucket_id") or shaped_bucket["bucket_id"]
            if parent_bucket_id not in selected_parent_bucket_ids:
                selected_parent_bucket_ids.append(parent_bucket_id)
            selected_count += bucket_size
        if selected_buckets and selected_count >= max_batch_size:
            break

    if not selected_buckets:
        raise ValueError("No deferred bucket concepts passed the residual review quality gates")

    shaped_parents: dict[str, dict[str, Any]] = {}
    for bucket in filtered_buckets:
        if int(bucket.get("concept_count") or 0) <= 0:
            continue
        shaped_parents[bucket["bucket_id"]] = _shape_filtered_bucket_for_review(bucket, concept_map)

    selected_bucket_notes = []
    for bucket in filtered_buckets:
        if int(bucket.get("concept_count") or 0) <= 0:
            continue
        shaped = shaped_parents[bucket["bucket_id"]]
        selected_shaped_buckets = [
            shaped_bucket
            for shaped_bucket in shaped["shaped_buckets"]
            if shaped_bucket["bucket_id"] in shaped_bucket_ids
        ]
        if not selected_shaped_buckets:
            continue
        selected_bucket_notes.append(
            {
                "bucket_id": bucket["bucket_id"],
                "dominant_ka": bucket.get("dominant_ka") or "",
                "original_concept_count": int(bucket.get("original_concept_count") or 0),
                "filtered_concept_count": int(bucket.get("concept_count") or 0),
                "staged_concept_count": sum(int(shaped_bucket.get("concept_count") or 0) for shaped_bucket in selected_shaped_buckets),
                "source_cluster_count": int(bucket.get("source_cluster_count") or 0),
                "source_cluster_ids": list(bucket.get("source_cluster_ids") or []),
                "knowledge_area_counts": dict(bucket.get("knowledge_area_counts") or {}),
                "excluded_counts": dict(bucket.get("review_filters", {}).get("excluded_counts") or {}),
                "shape_strategy": shaped.get("shape_strategy"),
                "shaped_bucket_count": len(shaped["shaped_buckets"]),
                "selected_shaped_bucket_ids": [shaped_bucket["bucket_id"] for shaped_bucket in selected_shaped_buckets],
                "selected_shaped_bucket_count": len(selected_shaped_buckets),
                "overflow_concept_ids": list(shaped.get("overflow_concept_ids") or []),
                "staged_concept_ids": [
                    concept_id
                    for shaped_bucket in selected_shaped_buckets
                    for concept_id in list(shaped_bucket.get("concept_ids") or [])
                ],
            }
        )

    payload = preview_batch(
        source_thread_id=source_thread.id,
        clusters=[_deferred_bucket_preview_cluster(bucket, source_thread) for bucket in selected_buckets],
        evaluation_set_id=evaluation_set_id,
        max_batch_size=max_batch_size,
        notes={
            "selected_bucket_ids": selected_parent_bucket_ids,
            "selected_shaped_bucket_ids": shaped_bucket_ids,
            "selected_buckets": selected_bucket_notes,
            "skipped_bucket_ids": skipped_bucket_ids,
            "review_filter_policy": {
                "min_confidence": DEFERRED_REVIEW_MIN_CONFIDENCE,
                "min_summary_chars": DEFERRED_REVIEW_MIN_SUMMARY_CHARS,
                "conversational_markers": list(DEFERRED_REVIEW_CONVERSATIONAL_MARKERS),
            },
        },
    )
    payload["selected_bucket_ids"] = selected_parent_bucket_ids
    payload["selected_bucket_count"] = len(selected_parent_bucket_ids)
    payload["selected_shaped_bucket_ids"] = shaped_bucket_ids
    payload["selected_shaped_bucket_count"] = len(shaped_bucket_ids)
    payload["raw_deferred_singletons"] = int(result.candidate_stats.get("raw_deferred_singletons", 0) or 0)
    payload["deferred_bucket_count"] = len(available_buckets)
    return payload


def commit_batch(batch_id: str) -> dict[str, Any]:
    """Commit a staged add-only batch."""
    if not THREAD_REORG_BATCH_WRITE_ENABLED:
        raise ValueError("THREAD_REORG_BATCH_WRITE_ENABLED is disabled")

    from app.features.threads import link_concept_to_thread, load_thread, save_thread, unlink_concept_from_thread
    from app.ops.metrics import metrics

    with _write_conn() as conn:
        batch = conn.execute(
            "SELECT source_thread_id, status, notes_json FROM thread_reorg_batches WHERE batch_id = ?",
            (batch_id,),
        ).fetchone()
        if not batch:
            raise ValueError(f"Batch {batch_id} not found")
        if batch[1] not in {"previewed", "held"}:
            raise ValueError(f"Batch {batch_id} is not commit-ready (status={batch[1]})")

        rows = conn.execute(
            """
            SELECT concept_id, target_thread_id, role, legacy_membership_before
            FROM thread_reorg_batch_members
            WHERE batch_id = ? AND status = 'staged' AND action = 'add'
            """,
            (batch_id,),
        ).fetchall()
        source_thread_id = batch[0]
        notes = json.loads(batch[2] or "{}")
        created_thread_ids = notes.get("created_thread_ids", [])

    committed = 0
    detached = 0
    promoted_by_thread: dict[str, list[str]] = defaultdict(list)
    detached_concept_ids: list[str] = []
    for concept_id, target_thread_id, role, legacy_membership_before in rows:
        link_concept_to_thread(target_thread_id, concept_id, role=role, added_by="auto")
        committed += 1
        promoted_by_thread[target_thread_id].append(concept_id)
        if legacy_membership_before and source_thread_id and source_thread_id != target_thread_id:
            unlink_concept_from_thread(source_thread_id, concept_id)
            detached += 1
            detached_concept_ids.append(concept_id)
        with _write_conn() as conn:
            conn.execute(
                """
                UPDATE thread_reorg_batch_members
                SET status = 'committed', applied_at = ?
                WHERE batch_id = ? AND concept_id = ? AND target_thread_id = ? AND action = 'add'
                """,
                (_utc_now_iso(), batch_id, concept_id, target_thread_id),
            )

    for thread_id in created_thread_ids:
        thread = load_thread(thread_id)
        if thread and thread.status == "paused":
            thread.status = "active"
            thread.updated_at = _utc_now_iso()
            save_thread(thread)

    with _write_conn() as conn:
        for target_thread_id, concept_ids in promoted_by_thread.items():
            _mark_seed_candidates_promoted(conn, concept_ids, target_thread_id)
        notes["detached_concept_ids"] = detached_concept_ids
        conn.execute(
            """
            UPDATE thread_reorg_batches
            SET target_mode = 'commit',
                status = 'committed',
                committed_count = ?,
                detached_count = ?,
                notes_json = ?,
                committed_at = ?
            WHERE batch_id = ?
            """,
            (committed, detached, json.dumps(notes), _utc_now_iso(), batch_id),
        )

    metrics.record("thread_reorg_batch_committed", committed, {"batch_id": batch_id})
    return {"batch_id": batch_id, "committed": committed, "detached": detached, "status": "committed"}


def _cleanup_created_preview_threads(conn, created_thread_ids: list[str]) -> dict[str, Any]:
    removed_thread_ids: list[str] = []
    retained_thread_ids: list[str] = []
    for thread_id in created_thread_ids:
        row = conn.execute("SELECT data FROM threads WHERE id = ?", (thread_id,)).fetchone()
        if not row:
            continue

        thread = NarrativeThread(**json.loads(row[0]))
        link_count = conn.execute(
            "SELECT COUNT(*) FROM thread_concept_links WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()[0]

        # Preview-only threads now carry staged concept_ids for operator review,
        # but should still be deleted on rollback if nothing was actually linked.
        if link_count == 0 and not thread.trace_ids:
            conn.execute("DELETE FROM threads WHERE id = ?", (thread_id,))
            removed_thread_ids.append(thread_id)
            continue

        if thread.status != "paused":
            thread.status = "paused"
            thread.updated_at = _utc_now_iso()
            save_thread(thread, conn=conn)
        retained_thread_ids.append(thread_id)

    return {
        "removed_thread_ids": removed_thread_ids,
        "retained_thread_ids": retained_thread_ids,
        "removed_thread_count": len(removed_thread_ids),
        "retained_thread_count": len(retained_thread_ids),
    }


def rollback_batch(batch_id: str) -> dict[str, Any]:
    """Rollback a previewed or committed add-only batch."""
    from app.features.threads import link_concept_to_thread, unlink_concept_from_thread
    from app.ops.metrics import metrics

    with _write_conn() as conn:
        batch = conn.execute(
            "SELECT source_thread_id, status, notes_json FROM thread_reorg_batches WHERE batch_id = ?",
            (batch_id,),
        ).fetchone()
        if not batch:
            raise ValueError(f"Batch {batch_id} not found")
        if batch[1] not in {"previewed", "committed", "held"}:
            raise ValueError(f"Batch {batch_id} cannot be rolled back from status={batch[1]}")
        rows = conn.execute(
            """
            SELECT concept_id, target_thread_id, status, role, legacy_membership_before
            FROM thread_reorg_batch_members
            WHERE batch_id = ? AND action = 'add' AND status IN ('staged', 'committed')
            """,
            (batch_id,),
        ).fetchall()
        source_thread_id = batch[0]
        notes = json.loads(batch[2] or "{}")
        created_thread_ids = notes.get("created_thread_ids", [])

    rolled_back = 0
    restored = 0
    for concept_id, target_thread_id, member_status, role, legacy_membership_before in rows:
        if member_status == "committed":
            unlink_concept_from_thread(target_thread_id, concept_id)
            if legacy_membership_before and source_thread_id and source_thread_id != target_thread_id:
                link_concept_to_thread(source_thread_id, concept_id, role=role, added_by="auto")
                restored += 1
        rolled_back += 1
        with _write_conn() as conn:
            conn.execute(
                """
                UPDATE thread_reorg_batch_members
                SET status = 'rolled_back', rolled_back_at = ?
                WHERE batch_id = ? AND concept_id = ? AND target_thread_id = ? AND action = 'add'
                """,
                (_utc_now_iso(), batch_id, concept_id, target_thread_id),
            )
            conn.execute(
                """
                UPDATE thread_reorg_seed_candidates
                SET status = 'queued', promoted_thread_id = NULL
                WHERE concept_id = ? AND promoted_thread_id = ?
                """,
                (concept_id, target_thread_id),
            )

    with _write_conn() as conn:
        cleanup = _cleanup_created_preview_threads(conn, created_thread_ids)
        notes["rollback_cleanup"] = cleanup
        notes["restored_source_membership_count"] = restored

        conn.execute(
            """
            UPDATE thread_reorg_batches
            SET status = 'rolled_back',
                rolled_back_at = ?,
                notes_json = ?
            WHERE batch_id = ?
            """,
            (_utc_now_iso(), json.dumps(notes), batch_id),
        )

    metrics.record("thread_reorg_batch_rolled_back", rolled_back, {"batch_id": batch_id})
    return {
        "batch_id": batch_id,
        "rolled_back": rolled_back,
        "restored": restored,
        "status": "rolled_back",
        "cleanup": cleanup,
    }


def get_batch_status(
    batch_id: str,
    *,
    include_members: bool = False,
    member_limit: int = 100,
) -> dict[str, Any]:
    with _read_conn() as conn:
        batch = conn.execute(
            """
            SELECT batch_id, source_thread_id, target_mode, status, planned_count,
                   committed_count, detached_count, evaluation_set_id, notes_json,
                   created_at, committed_at, rolled_back_at
            FROM thread_reorg_batches
            WHERE batch_id = ?
            """,
            (batch_id,),
        ).fetchone()
        if not batch:
            raise ValueError(f"Batch {batch_id} not found")
        member_rows = conn.execute(
            """
            SELECT status, COUNT(*)
            FROM thread_reorg_batch_members
            WHERE batch_id = ?
            GROUP BY status
            """,
            (batch_id,),
        ).fetchall()
        notes = json.loads(batch[8] or "{}")

        created_threads = []
        created_thread_ids = notes.get("created_thread_ids", [])
        if created_thread_ids:
            placeholders = ",".join("?" for _ in created_thread_ids)
            created_rows = conn.execute(
                f"""
                SELECT id, data
                FROM threads
                WHERE id IN ({placeholders})
                """,
                created_thread_ids,
            ).fetchall()
            created_map = {thread_id: json.loads(data) for thread_id, data in created_rows}
            for thread_id in created_thread_ids:
                thread_data = created_map.get(thread_id)
                if thread_data:
                    created_threads.append(
                        {
                            "thread_id": thread_id,
                            "exists": True,
                            "status": thread_data.get("status"),
                            "title": thread_data.get("title"),
                            "knowledge_areas": thread_data.get("knowledge_areas") or [],
                            "concept_ids": thread_data.get("concept_ids") or [],
                            "concept_count": len(thread_data.get("concept_ids") or []),
                        }
                    )
                else:
                    created_threads.append(
                        {
                            "thread_id": thread_id,
                            "exists": False,
                            "status": None,
                            "title": None,
                            "knowledge_areas": [],
                            "concept_ids": [],
                            "concept_count": 0,
                        }
                    )

        member_limit = max(1, member_limit)
        members: list[dict[str, Any]] = []
        if include_members:
            member_detail_rows = conn.execute(
                """
                SELECT m.concept_id, m.target_thread_id, m.status, m.role, m.legacy_membership_before,
                       m.rationale_json, c.summary, c.knowledge_area, c.confidence
                FROM thread_reorg_batch_members m
                LEFT JOIN concepts c ON c.id = m.concept_id
                WHERE m.batch_id = ?
                ORDER BY m.created_at ASC, m.concept_id ASC
                LIMIT ?
                """,
                (batch_id, member_limit),
            ).fetchall()
            members = [
                {
                    "concept_id": row[0],
                    "target_thread_id": row[1],
                    "status": row[2],
                    "role": row[3],
                    "legacy_membership_before": bool(row[4]),
                    "rationale": json.loads(row[5] or "{}"),
                    "summary": row[6],
                    "knowledge_area": row[7],
                    "confidence": row[8],
                }
                for row in member_detail_rows
            ]

    member_counts = {status: count for status, count in member_rows}
    payload = {
        "batch_id": batch[0],
        "source_thread_id": batch[1],
        "target_mode": batch[2],
        "status": batch[3],
        "planned_count": batch[4],
        "committed_count": batch[5],
        "detached_count": batch[6],
        "evaluation_set_id": batch[7],
        "notes": notes,
        "created_at": batch[9],
        "committed_at": batch[10],
        "rolled_back_at": batch[11],
        "member_counts": member_counts,
        "created_threads": created_threads,
    }
    if include_members:
        payload["members"] = members
        payload["member_limit"] = member_limit
        payload["member_total"] = sum(member_counts.values())
    return payload
