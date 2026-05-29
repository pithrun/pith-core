"""Bounded decision-shadow expansion for conversation_turn S4.1b."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from app.core.deadline import TurnDeadline
from app.core.models import SearchResult


STRATEGIC_DECISION_KAS = frozenset({
    "product_strategy",
    "business_strategy",
    "strategic_recommendation",
    "strategy",
})

STOP_REASON_CODES: dict[str, int] = {
    "not_run": 0,
    "completed": 1,
    "deadline_before_start": 2,
    "child_budget_exhausted": 3,
    "frontier_cap_reached": 4,
    "scanned_edge_cap_reached": 5,
    "candidate_load_cap_reached": 6,
    "hop3_budget_unhealthy": 7,
    "added_limit_reached": 8,
    "error": 9,
}

CAP_STOP_REASONS = {
    "child_budget_exhausted",
    "frontier_cap_reached",
    "scanned_edge_cap_reached",
    "candidate_load_cap_reached",
    "hop3_budget_unhealthy",
    "error",
}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"true", "1", "yes", "on"}:
        return True
    if value in {"false", "0", "no", "off"}:
        return False
    return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    value = int(_env_float(name, float(default)))
    return max(minimum, min(maximum, value))


def _env_ms(name: str, default: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, _env_float(name, default)))


@dataclass(frozen=True)
class DecisionShadowConfig:
    enabled: bool = True
    limit: int = 2
    neighbor_scan_limit: int = 200
    hop2_candidate_limit: int = 500
    hop3_candidate_limit: int = 500
    child_budget_ms: float = 250.0
    min_remaining_ms: float = 1000.0
    max_scanned_edges: int = 2000
    max_frontier: int = 1000
    max_candidate_loads: int = 80
    check_interval_edges: int = 64
    hop3_enabled: bool = True
    hop3_min_remaining_ms: float = 1800.0
    hop3_child_budget_ms: float = 75.0
    hop3_max_roots: int = 2
    hop3_max_frontier: int = 120

    @classmethod
    def from_env(cls) -> "DecisionShadowConfig":
        return cls(
            enabled=_env_bool("DECISION_SHADOW_ENABLED", True),
            limit=_env_int("DECISION_SHADOW_LIMIT", 2, 0, 20),
            neighbor_scan_limit=_env_int("DECISION_SHADOW_NEIGHBOR_SCAN_LIMIT", 200, 1, 5000),
            hop2_candidate_limit=_env_int("DECISION_SHADOW_HOP2_CANDIDATE_LIMIT", 500, 1, 10000),
            hop3_candidate_limit=_env_int("DECISION_SHADOW_HOP3_CANDIDATE_LIMIT", 500, 1, 10000),
            child_budget_ms=_env_ms("DECISION_SHADOW_CHILD_BUDGET_MS", 250.0, 10.0, 2000.0),
            min_remaining_ms=_env_ms("DECISION_SHADOW_MIN_REMAINING_MS", 1000.0, 0.0, 5000.0),
            max_scanned_edges=_env_int("DECISION_SHADOW_MAX_SCANNED_EDGES", 2000, 100, 50000),
            max_frontier=_env_int("DECISION_SHADOW_MAX_FRONTIER", 1000, 10, 10000),
            max_candidate_loads=_env_int("DECISION_SHADOW_MAX_CANDIDATE_LOADS", 80, 1, 500),
            check_interval_edges=_env_int("DECISION_SHADOW_CHECK_INTERVAL_EDGES", 64, 1, 1024),
            hop3_enabled=_env_bool("DECISION_HOP3_ENABLED", True),
            hop3_min_remaining_ms=_env_ms("DECISION_HOP3_MIN_REMAINING_MS", 1800.0, 0.0, 5000.0),
            hop3_child_budget_ms=_env_ms("DECISION_HOP3_CHILD_BUDGET_MS", 75.0, 10.0, 1000.0),
            hop3_max_roots=_env_int("DECISION_HOP3_MAX_ROOTS", 2, 1, 5),
            hop3_max_frontier=_env_int("DECISION_HOP3_MAX_FRONTIER", 120, 1, 5000),
        )


@dataclass
class DecisionShadowTrace:
    stop_reason: str = "not_run"
    scanned_edge_count: int = 0
    loaded_candidate_count: int = 0
    hop1_candidate_count: int = 0
    hop2_candidate_count: int = 0
    hop3_candidate_count: int = 0
    added_ids: list[str] = field(default_factory=list)
    added_hop_depths: dict[str, int] = field(default_factory=dict)
    root_ids: dict[str, str] = field(default_factory=dict)
    parent_ids: dict[str, str] = field(default_factory=dict)
    final_included_ids: list[str] = field(default_factory=list)
    final_inclusion_state: str = "unknown_at_decision_shadow"
    elapsed_ms: float = 0.0

    @property
    def stop_reason_code(self) -> int:
        return STOP_REASON_CODES.get(self.stop_reason, STOP_REASON_CODES["error"])


@dataclass
class DecisionShadowResult:
    additions: list[SearchResult] = field(default_factory=list)
    association_entries: dict[str, list[str]] = field(default_factory=dict)
    trace: DecisionShadowTrace = field(default_factory=DecisionShadowTrace)


def expand_decision_shadow(
    *,
    top_results: list[SearchResult],
    adjacency: dict[str, list[str]],
    edge_strength: dict[tuple[str, str], float],
    shadow_min_strength: float,
    deadline: TurnDeadline | None,
    load_concept_fn: Callable[[str], Any],
    config: DecisionShadowConfig | None = None,
    now_fn: Callable[[], float] = time.perf_counter,
) -> DecisionShadowResult:
    """Return bounded strategic decision additions for the current activation set."""

    config = config or DecisionShadowConfig.from_env()
    trace = DecisionShadowTrace()
    result = DecisionShadowResult(trace=trace)
    start_s = now_fn()
    stop_requested = False

    def finish(reason: str | None = None) -> DecisionShadowResult:
        if reason:
            trace.stop_reason = reason
        elif trace.stop_reason == "not_run":
            trace.stop_reason = "completed"
        trace.elapsed_ms = max(0.0, (now_fn() - start_s) * 1000.0)
        return result

    if not config.enabled or config.limit <= 0 or not top_results or not adjacency:
        return finish("not_run")

    active_kas = {r.knowledge_area for r in top_results if r.knowledge_area}
    if active_kas & STRATEGIC_DECISION_KAS:
        return finish("not_run")

    child_budget_ms = config.child_budget_ms
    if deadline is not None:
        child_budget_ms = deadline.child_budget_ms(
            "graph.decision_shadow",
            config.child_budget_ms,
            min_remaining_ms=config.min_remaining_ms,
        )
    if child_budget_ms <= 0.0:
        if deadline is not None:
            deadline.skip(
                "graph.decision_shadow",
                "deadline_before_start",
                min_remaining_ms=config.min_remaining_ms,
            )
        return finish("deadline_before_start")

    child_deadline_s = start_s + (child_budget_ms / 1000.0)
    existing_ids = {r.concept_id for r in top_results}

    def elapsed_over_budget(active_deadline_s: float = child_deadline_s) -> bool:
        return now_fn() >= active_deadline_s

    def request_stop(reason: str) -> bool:
        nonlocal stop_requested
        if trace.stop_reason in {"not_run", "completed"}:
            trace.stop_reason = reason
        stop_requested = True
        return True

    def should_stop(active_deadline_s: float = child_deadline_s) -> bool:
        if stop_requested:
            return True
        if trace.scanned_edge_count >= config.max_scanned_edges:
            return request_stop("scanned_edge_cap_reached")
        if elapsed_over_budget(active_deadline_s):
            return request_stop("child_budget_exhausted")
        return False

    def count_edge(active_deadline_s: float = child_deadline_s) -> bool:
        trace.scanned_edge_count += 1
        if trace.scanned_edge_count % config.check_interval_edges == 0:
            return should_stop(active_deadline_s)
        return False

    def decision_neighbors(src_id: str) -> list[str]:
        neighbors = list(adjacency.get(src_id, []))
        if config.neighbor_scan_limit > 0 and len(neighbors) > config.neighbor_scan_limit:
            neighbors.sort(
                key=lambda cid: edge_strength.get((src_id, cid), 0.0),
                reverse=True,
            )
            return neighbors[:config.neighbor_scan_limit]
        return neighbors

    def candidate_items(
        candidates: dict[str, tuple[float, str, str]],
        limit: int,
    ) -> list[tuple[str, tuple[float, str, str]]]:
        sorted_items = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)
        if limit > 0:
            return sorted_items[:limit]
        return sorted_items

    def try_add_decision(cid: str, score: float, hop_depth: int, root_id: str, parent_id: str) -> bool:
        if len(result.additions) >= config.limit:
            request_stop("added_limit_reached")
            return False
        if cid in existing_ids:
            return False
        if trace.loaded_candidate_count >= config.max_candidate_loads:
            request_stop("candidate_load_cap_reached")
            return False
        if should_stop():
            return False
        trace.loaded_candidate_count += 1
        concept = load_concept_fn(cid)
        if should_stop():
            return False
        if not concept:
            return False
        metadata = getattr(concept, "metadata", None) or {}
        concept_ka = metadata.get("knowledge_area")
        if concept_ka not in STRATEGIC_DECISION_KAS:
            return False
        if getattr(concept, "concept_type", "") != "decision":
            return False
        if getattr(concept, "currency_status", "ACTIVE") in ("SUPERSEDED", "STALE"):
            return False
        discount = {1: 0.50, 2: 0.35, 3: 0.20}.get(hop_depth, 0.15)
        result.additions.append(SearchResult(
            concept_id=concept.id,
            version=getattr(concept, "version", "v1"),
            summary=getattr(concept, "summary", ""),
            confidence=getattr(concept, "confidence", 0.0),
            relevance_score=round(score * discount, 4),
            knowledge_area=concept_ka,
        ))
        result.association_entries[cid] = adjacency.get(cid, [])
        existing_ids.add(cid)
        trace.added_ids.append(cid)
        trace.added_hop_depths[cid] = hop_depth
        trace.root_ids[cid] = root_id
        trace.parent_ids[cid] = parent_id
        return True

    try:
        hop1_min = shadow_min_strength * 0.3
        hop1_candidates: list[tuple[str, float, str]] = []
        for root in top_results[:5]:
            if should_stop():
                break
            for hop1_id in decision_neighbors(root.concept_id):
                if count_edge():
                    break
                if hop1_id in existing_ids:
                    continue
                s1 = edge_strength.get((root.concept_id, hop1_id), 0.0)
                if s1 >= hop1_min:
                    hop1_candidates.append((hop1_id, s1, root.concept_id))

        hop1_candidates.sort(key=lambda item: item[1], reverse=True)
        trace.hop1_candidate_count = len(hop1_candidates)
        for cid, score, root_id in hop1_candidates:
            if should_stop():
                break
            try_add_decision(cid, score, 1, root_id, root_id)

        if len(result.additions) < config.limit and not stop_requested:
            hop2_candidates: dict[str, tuple[float, str, str]] = {}
            for root in top_results[:5]:
                if should_stop():
                    break
                for hop1_id in decision_neighbors(root.concept_id):
                    if count_edge():
                        break
                    if hop1_id in existing_ids:
                        continue
                    s1 = edge_strength.get((root.concept_id, hop1_id), 0.0)
                    for hop2_id in decision_neighbors(hop1_id):
                        if count_edge():
                            break
                        if hop2_id in existing_ids or hop2_id == root.concept_id:
                            continue
                        s2 = edge_strength.get((hop1_id, hop2_id), 0.0)
                        combined = s1 * s2
                        if combined > hop2_candidates.get(hop2_id, (0.0, "", ""))[0]:
                            hop2_candidates[hop2_id] = (combined, root.concept_id, hop1_id)
                        if len(hop2_candidates) >= config.max_frontier:
                            request_stop("frontier_cap_reached")
                            break
                    if stop_requested:
                        break

            trace.hop2_candidate_count = len(hop2_candidates)
            for cid, (score, root_id, parent_id) in candidate_items(hop2_candidates, config.hop2_candidate_limit):
                if should_stop() or len(result.additions) >= config.limit:
                    break
                if score < shadow_min_strength * 0.4:
                    break
                try_add_decision(cid, score, 2, root_id, parent_id)

        if (
            config.hop3_enabled
            and len(result.additions) == 0
            and config.limit > 0
            and not stop_requested
        ):
            if deadline is not None and not deadline.can_start(
                "graph.decision_shadow.hop3",
                min_remaining_ms=config.hop3_min_remaining_ms,
            ):
                request_stop("hop3_budget_unhealthy")
            else:
                hop3_deadline_s = min(
                    child_deadline_s,
                    now_fn() + (config.hop3_child_budget_ms / 1000.0),
                )
                hop3_candidates: dict[str, tuple[float, str, str]] = {}
                hop3_min = shadow_min_strength * 0.15
                for root in top_results[:config.hop3_max_roots]:
                    if should_stop(hop3_deadline_s):
                        break
                    for hop1_id in decision_neighbors(root.concept_id):
                        if count_edge(hop3_deadline_s):
                            break
                        s1 = edge_strength.get((root.concept_id, hop1_id), 0.0)
                        if s1 < hop3_min:
                            continue
                        for hop2_id in decision_neighbors(hop1_id):
                            if count_edge(hop3_deadline_s):
                                break
                            if hop2_id == root.concept_id:
                                continue
                            s2 = edge_strength.get((hop1_id, hop2_id), 0.0)
                            if s1 * s2 < hop3_min:
                                continue
                            for hop3_id in decision_neighbors(hop2_id):
                                if count_edge(hop3_deadline_s):
                                    break
                                if hop3_id in existing_ids or hop3_id == hop1_id or hop3_id == root.concept_id:
                                    continue
                                s3 = edge_strength.get((hop2_id, hop3_id), 0.0)
                                combined = s1 * s2 * s3
                                if combined > hop3_candidates.get(hop3_id, (0.0, "", ""))[0]:
                                    hop3_candidates[hop3_id] = (combined, root.concept_id, hop2_id)
                                if len(hop3_candidates) >= config.hop3_max_frontier:
                                    request_stop("frontier_cap_reached")
                                    break
                            if stop_requested:
                                break
                        if stop_requested:
                            break

                trace.hop3_candidate_count = len(hop3_candidates)
                for cid, (score, root_id, parent_id) in candidate_items(hop3_candidates, config.hop3_candidate_limit)[:10]:
                    if should_stop(hop3_deadline_s) or len(result.additions) >= config.limit:
                        break
                    if score < hop3_min:
                        break
                    try_add_decision(cid, score, 3, root_id, parent_id)
    except Exception:
        return finish("error")

    if trace.stop_reason == "not_run":
        if len(result.additions) >= config.limit:
            trace.stop_reason = "added_limit_reached"
        else:
            trace.stop_reason = "completed"
    return finish()
