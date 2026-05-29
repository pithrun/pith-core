"""REFLECT-030: Server-side L1→L3 concept synthesis via LLM.

Runs as maintenance Phase 7. Clusters L1 observations by knowledge_area,
calls Haiku to synthesize them into L3+ concepts (principles, methods,
heuristics, cognitive strategies). No client dependency.

Follows the established pattern from experiment_llm.py:
- Async Anthropic client with circuit breaker
- JSON response validation
- Skepticism-framed prompts (GAME-1)
"""

import json
import logging
import os
import time
import uuid
from collections import defaultdict
from datetime import datetime

from app.cognitive.auto_reflection import (
    L1_TYPES,
    L3_TYPES,
    _classify_cluster_type,
    _extract_theme,
)
from app.core.datetime_utils import _utc_now, _utc_now_iso
from app.core.metrics_facade import metrics
from app.core.models import Concept, Evidence, SearchQuery
from app.storage import (
    _db,
    _db_immediate,
    list_concepts,
    load_concept,
    save_concept,
)

logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================

MODEL = os.environ.get("PITH_SYNTHESIS_MODEL", "google/gemini-2.0-flash-001")  # COST-001
TIMEOUT_SECONDS = int(os.environ.get("PITH_SYNTHESIS_TIMEOUT_S", "15"))
MAX_TOKENS = 500
SUMMARY_TRUNCATE = 500  # Max chars per concept summary in prompts

MAX_CLUSTERS = int(os.environ.get("PITH_SYNTHESIS_MAX_CLUSTERS", "5"))
MIN_CLUSTER_SIZE = int(os.environ.get("PITH_SYNTHESIS_MIN_CLUSTER_SIZE", "3"))
MAX_CONFIDENCE = float(os.environ.get("PITH_SYNTHESIS_MAX_CONFIDENCE", "0.85"))
DEDUP_COSINE_THRESHOLD = float(os.environ.get("PITH_SYNTHESIS_DEDUP_THRESHOLD", "0.78"))
MAX_CANDIDATES = 200  # Safety cap on L1 concepts to evaluate

# Confidence bracket tiers — empirically set from cluster size distribution
# (n=70 observed clusters: min=13, median=22, max=47)
# Override via env vars to tune without code changes.
CONF_TIER_A_MIN = int(os.environ.get("PITH_SYNTH_TIER_A_MIN", "30"))   # strong  → 0.70–0.85
CONF_TIER_B_MIN = int(os.environ.get("PITH_SYNTH_TIER_B_MIN", "18"))   # moderate → 0.58–0.72
# Below TIER_B_MIN → Tier C: 0.40–0.58


def _confidence_bracket(concept_count: int) -> tuple[float, float, str]:
    """Return (low, high, label) confidence range earned by cluster size."""
    if concept_count >= CONF_TIER_A_MIN:
        return 0.70, 0.85, "strong evidence"
    elif concept_count >= CONF_TIER_B_MIN:
        return 0.58, 0.72, "moderate evidence"
    else:
        return 0.40, 0.58, "thin evidence"

# Circuit breaker: trips on first AuthenticationError, resets on restart
_SYNTHESIS_LLM_AUTH_FAILED: bool = False

# Valid concept types for synthesis output
VALID_SYNTHESIS_TYPES = {"principle", "method", "heuristic", "cognitive_strategy"}

# ============================================================
# Prompt Template (GAME-1: skepticism framing)
# ============================================================

PROMPT_L1_SYNTHESIS = """You are analyzing a knowledge base. Be skeptical — most \
observation clusters are noise or near-duplicate phrasing. Only synthesize if there \
is a genuine non-obvious insight that goes beyond what any single observation says.

These {n} observations in the '{knowledge_area}' domain cluster together:

{numbered_summaries}

{synthesis_hint}

If there is a genuine higher-order insight, synthesize it (50-300 chars). \
The synthesis must be ACTIONABLE — a principle someone can apply, a method \
someone can follow, or a heuristic someone can use for decisions.

If the cluster is just noise, duplicate phrasing, or the observations don't \
actually share a meaningful pattern, say so.

Confidence calibration: this cluster has {n} source concepts ({tier_label}). \
Assign confidence between {conf_low:.2f}–{conf_high:.2f}. Within that range: lean \
toward {conf_high:.2f} if the synthesis is crisp and reusable with no counterexamples; \
lean toward {conf_low:.2f} if the insight is partial or context-dependent.

Respond in JSON only:
{{"meaningful": true/false, "synthesis": "...", "confidence": 0.0-1.0, \
"concept_type": "principle|method|heuristic|cognitive_strategy", \
"knowledge_area": "{knowledge_area}", \
"reason": "brief explanation of why this is/isn't meaningful"}}"""


# ============================================================
# Response Validation (mirrors experiment_llm.py CF-1)
# ============================================================


def _validate_synthesis_response(
    raw: dict,
    bracket: tuple[float, float] | None = None,
) -> dict:
    """Validate and sanitize LLM JSON response.

    If bracket=(low, high) is provided, confidence is clamped to that range
    instead of the global MAX_CONFIDENCE ceiling.
    """
    conf_low, conf_high = bracket if bracket else (0.0, MAX_CONFIDENCE)
    default_conf = (conf_low + conf_high) / 2
    result = {
        "meaningful": bool(raw.get("meaningful", False)),
        "synthesis": str(raw.get("synthesis", ""))[:500],
        "confidence": max(conf_low, min(conf_high, float(raw.get("confidence", default_conf)))),
        "concept_type": raw.get("concept_type", "principle"),
        "knowledge_area": str(raw.get("knowledge_area", "general"))[:50],
        "reason": str(raw.get("reason", ""))[:200],
    }
    if result["concept_type"] not in VALID_SYNTHESIS_TYPES:
        result["concept_type"] = "principle"
    return result


# ============================================================
# LLM API Call (mirrors experiment_llm.py pattern)
# ============================================================


async def _call_anthropic(prompt: str) -> str:
    """Call LLM via OpenRouter (COST-001: switched from Anthropic direct billing). Returns raw text response."""
    global _SYNTHESIS_LLM_AUTH_FAILED
    if _SYNTHESIS_LLM_AUTH_FAILED:
        raise RuntimeError("REFLECT-030: LLM disabled — OPENROUTER_API_KEY invalid this run")

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
            _SYNTHESIS_LLM_AUTH_FAILED = True
            logger.error("REFLECT-030: LLM disabled — API key rejected. Error: %s", e)
            raise
        if "403" in err_str or "permission" in err_str:
            _SYNTHESIS_LLM_AUTH_FAILED = True
            logger.error("REFLECT-030: LLM disabled — API key lacks permissions. Error: %s", e)
            raise
        if "credit" in err_str or "billing" in err_str:
            _SYNTHESIS_LLM_AUTH_FAILED = True
            logger.error("REFLECT-030: LLM disabled — credits depleted. Error: %s", e)
        raise
    return response.choices[0].message.content or ""


# ============================================================
# Candidate Selection (Phase 7.1)
# ============================================================


def _select_synthesis_candidates() -> list[dict]:
    """Select L1 concepts that have never been evaluated for synthesis.

    Returns list of dicts with id, summary, confidence, knowledge_area, concept_type.
    """
    with _db() as conn:
        rows = conn.execute(
            """SELECT c.id, c.summary, c.confidence, c.concept_type,
                      c.knowledge_area as knowledge_area
               FROM concepts c
               WHERE c.is_current = 1
                 AND c.status = 'active'
                 AND c.maturity IN ('ESTABLISHED', 'PROVISIONAL')
                 AND c.concept_type IN (?, ?, ?, ?, ?)
                 AND c.last_synthesis_evaluated_at IS NULL
               ORDER BY c.created_at DESC
               LIMIT ?""",
            (*sorted(L1_TYPES), MAX_CANDIDATES),
        ).fetchall()

    candidates = []
    for row in rows:
        candidates.append({
            "id": row[0],
            "summary": (row[1] or "")[:SUMMARY_TRUNCATE],
            "confidence": row[2] or 0.3,
            "concept_type": row[3] or "observation",
            "knowledge_area": row[4] or "general",
        })
    return candidates


# ============================================================
# Clustering (Phase 7.2)
# ============================================================


def _cluster_candidates(candidates: list[dict]) -> list[dict]:
    """Cluster candidates by knowledge_area. Returns top clusters.

    Each cluster: {knowledge_area, concepts: [...], avg_confidence, synthesis_hint}
    """
    by_area: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        by_area[c["knowledge_area"]].append(c)

    clusters = []
    for area, concepts in by_area.items():
        if len(concepts) < MIN_CLUSTER_SIZE:
            continue

        avg_conf = sum(c["confidence"] for c in concepts) / len(concepts)
        summaries = [c["summary"] for c in concepts]
        target_type, question_template = _classify_cluster_type(summaries)
        theme = _extract_theme(summaries, area)

        clusters.append({
            "knowledge_area": area,
            "concepts": concepts,
            "avg_confidence": avg_conf,
            "target_type": target_type,
            "synthesis_hint": question_template.format(theme=theme),
            "theme": theme,
            "score": len(concepts) * avg_conf,  # Rank metric
        })

    # Rank by score, take top N
    clusters.sort(key=lambda c: c["score"], reverse=True)
    return clusters[:MAX_CLUSTERS]


# ============================================================
# Prompt Builder (Phase 7.3)
# ============================================================


def _build_synthesis_prompt(cluster: dict) -> tuple[str, tuple[float, float]]:
    """Build the synthesis prompt for a cluster.

    Returns (prompt_text, (conf_low, conf_high)) so the caller can clamp
    the LLM's output to the earned confidence bracket.
    """
    summaries = []
    for i, c in enumerate(cluster["concepts"][:10], 1):
        summaries.append(f"{i}. {c['summary']}")

    n = len(cluster["concepts"])
    conf_low, conf_high, tier_label = _confidence_bracket(n)

    prompt = PROMPT_L1_SYNTHESIS.format(
        n=n,
        knowledge_area=cluster["knowledge_area"],
        numbered_summaries="\n".join(summaries),
        synthesis_hint=cluster["synthesis_hint"],
        tier_label=tier_label,
        conf_low=conf_low,
        conf_high=conf_high,
    )
    return prompt, (conf_low, conf_high)


# ============================================================
# Deduplication Guard
# ============================================================


def _check_dedup(synthesis_text: str, knowledge_area: str) -> tuple[bool, float]:
    """Check if synthesis duplicates an existing concept.

    Returns (is_duplicate, max_similarity).
    is_duplicate=True means cosine >= 0.85 — skip creation.
    """
    try:
        from app.retrieval import retrieval_engine
        # Use embedding cosine similarity — blended relevance_score caps at ~0.55
        # for near-duplicates, making the old 0.85 threshold unreachable.
        results = retrieval_engine.search_for_dedup_embedding(synthesis_text, top_k=5)
        for r in results:
            if r.get("knowledge_area", "") != knowledge_area:
                continue
            cosine = r.get("cosine_score", 0.0)
            if cosine >= DEDUP_COSINE_THRESHOLD:
                return True, cosine
        return False, 0.0
    except Exception as e:
        logger.warning("REFLECT-030: Dedup check failed (non-fatal): %s", e)
        return False, 0.0


# ============================================================
# Concept Writer (Phase 7.4)
# ============================================================


def _create_synthesized_concept(
    synthesis: dict,
    source_concept_ids: list[str],
    cluster_knowledge_area: str,
    cycle_id: str,
) -> str | None:
    """Create and save a synthesized L3+ concept. Returns concept_id or None."""
    try:
        concept_id = f"synth_{uuid.uuid4().hex[:12]}"

        # Collect evidence from source L1 concepts
        evidence_list = []
        for src_id in source_concept_ids[:5]:
            src = load_concept(src_id, track_access=False)
            if src and src.evidence:
                for ev in src.evidence[:2]:  # Max 2 per source
                    evidence_list.append(ev)

        # Add synthesis provenance evidence
        evidence_list.append(Evidence(
            source_type="synthesis",
            content=f"Auto-synthesized from {len(source_concept_ids)} L1 concepts by Phase 7 (cycle {cycle_id}). Reason: {synthesis['reason']}",
            reliability_weight=0.5,
            directness=0.4,
            consistency=0.7,
            extraction_source="synthesis_engine",
            timestamp=_utc_now_iso(),
        ))

        concept = Concept(
            id=concept_id,
            version="v1",
            created_at=_utc_now_iso(),
            summary=synthesis["synthesis"],
            concept_type=synthesis["concept_type"],
            confidence=synthesis["confidence"],
            knowledge_area=cluster_knowledge_area,  # DEBT-227: set directly instead of metadata dual-write
            evidence=evidence_list,
            metadata={
                "knowledge_area_source": "canonical",
                "created_by": "synthesis_engine",
                "synthesis_source": source_concept_ids,
                "synthesis_cycle": cycle_id,
                "is_factual": False,
            },
        )

        save_concept(concept)
        logger.info(
            "REFLECT-030: Created synthesized concept %s (%s) in %s: %s",
            concept_id, synthesis["concept_type"], cluster_knowledge_area,
            synthesis["synthesis"][:100],
        )
        return concept_id

    except Exception as e:
        logger.error("REFLECT-030: Failed to create concept: %s", e)
        return None


# ============================================================
# Watermark Writer (Phase 7.45)
# ============================================================


def _mark_evaluated(concept_ids: list[str]) -> int:
    """Set last_synthesis_evaluated_at on evaluated L1 concepts."""
    if not concept_ids:
        return 0
    now = _utc_now_iso()
    try:
        with _db_immediate() as conn:
            placeholders = ",".join("?" for _ in concept_ids)
            conn.execute(
                f"""UPDATE concepts
                    SET last_synthesis_evaluated_at = ?
                    WHERE id IN ({placeholders})
                      AND is_current = 1""",
                [now, *concept_ids],
            )
        return len(concept_ids)
    except Exception as e:
        logger.error("REFLECT-030: Watermark write failed: %s", e)
        return 0


# ============================================================
# Association Linker (Phase 7.5)
# ============================================================


def _link_to_sources(synth_concept_id: str, source_ids: list[str]) -> int:
    """Create 'synthesized_from' associations from L3 to source L1s."""
    linked = 0
    try:
        with _db_immediate() as conn:
            now = _utc_now_iso()
            rows = []
            for src_id in source_ids:
                rows.append((
                    synth_concept_id, src_id, "synthesized_from",
                    0.8, now, "synthesis_engine",
                ))
            if rows:
                conn.executemany(
                    """INSERT OR IGNORE INTO associations
                       (source, target, relation,
                        strength, created_at, mechanism)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    rows,
                )
                linked = len(rows)
    except Exception as e:
        logger.error("REFLECT-030: Association linking failed: %s", e)
    return linked


# ============================================================
# Tracking (Phase 7.6)
# ============================================================


def _record_synthesis_cycle(
    cycle_id: str,
    clusters_evaluated: int,
    clusters_meaningful: int,
    concepts_created: int,
    concepts_deduplicated: int,
    llm_calls: int,
    llm_failures: int,
    total_ms: float,
    details: dict | None = None,
) -> None:
    """Log synthesis cycle to tracking table."""
    try:
        with _db_immediate() as conn:
            conn.execute(
                """INSERT INTO synthesis_tracking
                   (cycle_id, created_at, clusters_evaluated, clusters_meaningful,
                    concepts_created, concepts_deduplicated, llm_calls, llm_failures,
                    total_ms, details)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    cycle_id, _utc_now_iso(), clusters_evaluated, clusters_meaningful,
                    concepts_created, concepts_deduplicated, llm_calls, llm_failures,
                    round(total_ms, 1), json.dumps(details) if details else None,
                ),
            )
    except Exception as e:
        logger.error("REFLECT-030: Tracking write failed: %s", e)


# ============================================================
# Main Orchestrator (Phase 7 entry point)
# ============================================================


async def run_synthesis() -> dict:
    """Run the full Phase 7 synthesis cycle.

    Returns dict with cycle stats for maintenance report.
    """
    from app.core.config import FEATURE_FLAGS

    # Gate check
    if not FEATURE_FLAGS.get("SYNTHESIS_LLM_ENABLED", True):
        return {"status": "skipped", "reason": "feature_flag_disabled"}
    if not os.environ.get("OPENROUTER_API_KEY"):
        return {"status": "skipped", "reason": "no_api_key"}
    if _SYNTHESIS_LLM_AUTH_FAILED:
        return {"status": "skipped", "reason": "circuit_breaker_tripped"}

    cycle_id = uuid.uuid4().hex[:12]
    t0 = time.monotonic()

    # 7.1 Select candidates
    candidates = _select_synthesis_candidates()
    if not candidates:
        logger.info("REFLECT-030: No unevaluated L1 concepts — skipping Phase 7")
        return {"status": "skipped", "reason": "no_candidates"}

    logger.info("REFLECT-030: Phase 7 starting — %d candidates, cycle %s", len(candidates), cycle_id)

    # 7.2 Cluster
    clusters = _cluster_candidates(candidates)
    if not clusters:
        # Mark all candidates as evaluated even if no clusters formed
        all_ids = [c["id"] for c in candidates]
        _mark_evaluated(all_ids)
        logger.info("REFLECT-030: No clusters met minimum size (%d) — marking %d evaluated", MIN_CLUSTER_SIZE, len(all_ids))
        return {"status": "completed", "reason": "no_qualifying_clusters", "candidates": len(candidates)}

    # 7.3-7.5 Process each cluster
    stats = {
        "clusters_evaluated": 0,
        "clusters_meaningful": 0,
        "concepts_created": 0,
        "concepts_deduplicated": 0,
        "llm_calls": 0,
        "llm_failures": 0,
        "cluster_details": [],
    }

    for cluster in clusters:
        cluster_detail = {
            "knowledge_area": cluster["knowledge_area"],
            "concept_count": len(cluster["concepts"]),
            "theme": cluster["theme"],
        }
        stats["clusters_evaluated"] += 1

        # Build prompt and call LLM
        prompt, bracket = _build_synthesis_prompt(cluster)
        stats["llm_calls"] += 1

        try:
            raw_text = await _call_anthropic(prompt)
        except Exception as e:
            stats["llm_failures"] += 1
            cluster_detail["status"] = f"llm_error: {e}"
            stats["cluster_details"].append(cluster_detail)
            # Still mark these L1s as evaluated to prevent retry
            _mark_evaluated([c["id"] for c in cluster["concepts"]])
            continue

        # Parse response
        try:
            text = raw_text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
            parsed = json.loads(text)
            synthesis = _validate_synthesis_response(parsed, bracket=bracket)
        except (json.JSONDecodeError, ValueError) as e:
            stats["llm_failures"] += 1
            cluster_detail["status"] = f"parse_error: {e}"
            stats["cluster_details"].append(cluster_detail)
            _mark_evaluated([c["id"] for c in cluster["concepts"]])
            continue

        # Check meaningfulness
        if not synthesis["meaningful"]:
            cluster_detail["status"] = f"not_meaningful: {synthesis['reason']}"
            stats["cluster_details"].append(cluster_detail)
            _mark_evaluated([c["id"] for c in cluster["concepts"]])
            continue

        stats["clusters_meaningful"] += 1

        # Dedup guard
        is_dup, sim_score = _check_dedup(synthesis["synthesis"], cluster["knowledge_area"])
        if is_dup:
            stats["concepts_deduplicated"] += 1
            cluster_detail["status"] = f"deduplicated (sim={sim_score:.3f})"
            stats["cluster_details"].append(cluster_detail)
            _mark_evaluated([c["id"] for c in cluster["concepts"]])
            continue

        # Create concept
        source_ids = [c["id"] for c in cluster["concepts"]]
        concept_id = _create_synthesized_concept(
            synthesis, source_ids, cluster["knowledge_area"], cycle_id,
        )

        if concept_id:
            stats["concepts_created"] += 1
            _link_to_sources(concept_id, source_ids)
            cluster_detail["status"] = f"created: {concept_id}"
            cluster_detail["synthesis"] = synthesis["synthesis"][:100]
        else:
            cluster_detail["status"] = "creation_failed"

        stats["cluster_details"].append(cluster_detail)

        # 7.45 Mark source L1s as evaluated
        _mark_evaluated(source_ids)

    # 7.6 Record tracking
    total_ms = (time.monotonic() - t0) * 1000
    _record_synthesis_cycle(
        cycle_id=cycle_id,
        clusters_evaluated=stats["clusters_evaluated"],
        clusters_meaningful=stats["clusters_meaningful"],
        concepts_created=stats["concepts_created"],
        concepts_deduplicated=stats["concepts_deduplicated"],
        llm_calls=stats["llm_calls"],
        llm_failures=stats["llm_failures"],
        total_ms=total_ms,
        details=stats["cluster_details"],
    )

    # Emit metrics
    metrics.record("synthesis_concepts_created", stats["concepts_created"])
    # DEBT-228: add labels so latency can be attributed to model and call site
    metrics.record(
        "synthesis_llm_latency_ms",
        total_ms,
        labels={"model": MODEL, "call_site": "synthesis"},
    )
    metrics.record("synthesis_meaningful_rate",
                    stats["clusters_meaningful"] / max(1, stats["clusters_evaluated"]))

    logger.info(
        "REFLECT-030: Phase 7 complete in %.1fs — %d clusters, %d meaningful, "
        "%d concepts created, %d deduped, %d LLM failures",
        total_ms / 1000,
        stats["clusters_evaluated"],
        stats["clusters_meaningful"],
        stats["concepts_created"],
        stats["concepts_deduplicated"],
        stats["llm_failures"],
    )

    return {
        "status": "completed",
        "cycle_id": cycle_id,
        "clusters_evaluated": stats["clusters_evaluated"],
        "clusters_meaningful": stats["clusters_meaningful"],
        "concepts_created": stats["concepts_created"],
        "concepts_deduplicated": stats["concepts_deduplicated"],
        "llm_calls": stats["llm_calls"],
        "llm_failures": stats["llm_failures"],
        "total_ms": round(total_ms, 1),
    }
