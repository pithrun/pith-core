"""
evolution.py — Inline Evolution Supersession Canary (RETRIEVAL-020 Phase 2A)

Detects concepts that have naturally evolved into a newer, higher-type concept
in the EMBEDDING ZONE (0.50–0.82 cosine similarity). This is the "soft supersession"
path: similar topic, type upgrade, authority increase.

Phase 2A: Canary mode only (EVOLUTION_CANARY_MODE=True by default).
  - Detects candidate pairs via embedding search.
  - Logs detections to pith.log (EVOLUTION_CANARY prefix).
  - Does NOT create supersession edges until Phase 2B.

Phase 2B (future): Flip EVOLUTION_CANARY_MODE=False in config.py after reviewing
  canary logs. No other code changes needed.

Phase 3 (future): Wire check_evolution_supersession() into session.py
  _create_new_concept() at concept creation time.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from app.core.config import (
    EVOLUTION_CANARY_MODE,
    EVOLUTION_COSINE_MAX,
    EVOLUTION_COSINE_MIN,
    EVOLUTION_REJECT_COMPOSITE,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Result type
# =============================================================================


@dataclass
class EvolutionResult:
    """Outcome of a single evolution supersession check."""

    checked: bool = False
    pair_detected: bool = False

    # Set when pair_detected=True
    older_concept_id: Optional[str] = None
    newer_concept_id: Optional[str] = None
    cosine_score: float = 0.0
    composite_score: float = 0.0
    type_progression: str = ""
    action_taken: str = ""  # "canary_logged" | "supersession_created" | ""

    # Set when pair_detected=False
    reason: str = ""  # e.g. "no_embedding_index", "no_evolution_candidates"

    time_ms: float = 0.0


# =============================================================================
# Main entry point
# =============================================================================


def check_evolution_supersession(
    new_concept_id: str,
    new_concept_type: str,
    new_authority: float,
    retrieval_engine: Any,
    concept_loader: Callable[[str], Any],
    supersede_fn: Optional[Callable[..., None]] = None,
    canary_mode: bool = True,
) -> EvolutionResult:
    """Check whether a newly created concept is an evolution of an existing one.

    Searches the EVOLUTION ZONE (EVOLUTION_COSINE_MIN <= cosine < EVOLUTION_COSINE_MAX)
    for candidates. Scores each candidate using the recalibrated Phase 1 formula:
        composite = 0.40 * cosine_factor + 0.30 * authority_delta + 0.30 * type_factor

    In canary_mode=True (default): logs detections, no writes.
    In canary_mode=False: creates supersession edge via supersede_fn.

    Args:
        new_concept_id: Concept ID just created.
        new_concept_type: Type of the new concept (e.g., 'principle', 'method').
        new_authority: Authority score of the new concept.
        retrieval_engine: RetrievalEngine instance (has .emb with embedding index).
        concept_loader: Callable to load concept by ID → concept object or None.
        supersede_fn: Callable for creating supersession edge (required if canary_mode=False).
        canary_mode: If True, detect only — do not write supersession edges.

    Returns:
        EvolutionResult with detection outcome.
    """
    from app.cognitive.supersession import TYPE_RANK

    t_start = time.time()
    result = EvolutionResult(checked=True)

    # Config flag is the ultimate authority — overrides parameter for safety
    effective_canary = canary_mode or EVOLUTION_CANARY_MODE

    try:
        # Step 1: Get embedding index from retrieval engine
        emb = getattr(retrieval_engine, "emb", None)
        if (
            emb is None
            or not hasattr(emb, "_id_to_pos")
            or not hasattr(emb, "_index_matrix")
            or not hasattr(emb, "_index_ids")
            or emb._index_matrix is None
        ):
            result.reason = "no_embedding_index"
            return result

        new_pos = emb._id_to_pos.get(new_concept_id)
        if new_pos is None:
            result.reason = "new_concept_not_in_index"
            return result

        # Step 2: O(N) dot product search — L2-normalized vectors → cosine scores
        import numpy as np

        query_vec = emb._index_matrix[new_pos]  # shape: (dim,)
        scores = emb._index_matrix @ query_vec  # shape: (N,) — all cosine scores

        # Step 3: Collect candidates in the evolution zone
        evolution_candidates = []
        for idx, score in enumerate(scores):
            if idx == new_pos:
                continue
            if idx >= len(emb._index_ids):
                continue
            if EVOLUTION_COSINE_MIN <= float(score) < EVOLUTION_COSINE_MAX:
                candidate_id = emb._index_ids[idx]
                if candidate_id:
                    evolution_candidates.append((candidate_id, float(score)))

        if not evolution_candidates:
            result.reason = "no_evolution_candidates"
            return result

        # Step 4: Score candidates — pick highest composite type-progression match
        new_rank = TYPE_RANK.get(new_concept_type, 0)
        best_candidate = None
        best_composite = 0.0

        for cand_id, cosine in evolution_candidates:
            cand = concept_loader(cand_id)
            if cand is None:
                continue

            # Guard: skip if candidate is already superseded (prevent transitive chains A→B→C)
            if getattr(cand, "superseded_by", None):
                continue

            # Guard: new concept must be a genuine type upgrade
            cand_rank = TYPE_RANK.get(getattr(cand, "concept_type", "observation"), 0)
            if cand_rank >= new_rank:
                continue

            # Recalibrated composite (RETRIEVAL-020 Phase 1 formula — no temporal)
            auth_a = getattr(cand, "authority_score", 0.0) or 0.0
            cosine_factor = (cosine - EVOLUTION_COSINE_MIN) / (EVOLUTION_COSINE_MAX - EVOLUTION_COSINE_MIN)
            authority_delta = min(1.0, (new_authority - auth_a) / 0.20)
            type_gap = new_rank - cand_rank
            type_factor = min(1.0, type_gap / 3) if type_gap > 0 else 0.0
            composite = (
                0.40 * cosine_factor
                + 0.30 * max(0.0, authority_delta)
                + 0.30 * type_factor
            )

            if composite < EVOLUTION_REJECT_COMPOSITE:
                continue

            if composite > best_composite:
                best_composite = composite
                best_candidate = (cand_id, cosine, composite, cand_rank)

        if best_candidate is None:
            result.reason = "no_qualifying_candidate"
            return result

        cand_id, cosine, composite, cand_rank = best_candidate
        result.pair_detected = True
        result.older_concept_id = cand_id
        result.newer_concept_id = new_concept_id
        result.cosine_score = cosine
        result.composite_score = composite
        result.type_progression = f"rank_{cand_rank} -> rank_{new_rank}"

        if effective_canary:
            # Canary mode: log only, no writes
            logger.info(
                "EVOLUTION_CANARY: detected pair older=%s newer=%s "
                "cosine=%.3f composite=%.3f type_progression=%s",
                cand_id,
                new_concept_id,
                cosine,
                composite,
                result.type_progression,
            )
            result.action_taken = "canary_logged"
        else:
            # Phase 2B: create supersession edge
            if supersede_fn is None:
                logger.error(
                    "EVOLUTION: supersede_fn required when canary_mode=False; "
                    "pair older=%s newer=%s skipped",
                    cand_id,
                    new_concept_id,
                )
                result.reason = "missing_supersede_fn"
                return result
            supersede_fn(older_id=cand_id, newer_id=new_concept_id, reason="evolution_supersession")
            result.action_taken = "supersession_created"
            logger.info(
                "EVOLUTION: superseded older=%s by newer=%s cosine=%.3f composite=%.3f",
                cand_id,
                new_concept_id,
                cosine,
                composite,
            )

    except Exception as e:
        logger.warning(
            "EVOLUTION: check_evolution_supersession failed: %s",
            e,
            exc_info=True,
        )
        result.reason = f"exception:{type(e).__name__}"

    result.time_ms = (time.time() - t_start) * 1000
    return result
