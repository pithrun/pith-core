"""Semantic Drift Detection — TF-IDF cosine distance across version chains.

Phase 3 v1.1, WS2: Tracks semantic distance across a concept's evolution chain
using TF-IDF cosine distance. When cumulative drift exceeds a threshold, the
concept is flagged as DRIFTED and its maturity is downgraded to PROVISIONAL.

Why TF-IDF (not embedding)?
- TF-IDF vectors are already indexed (~2500+ concepts)
- No API cost, no additional latency
- Captures lexical drift well
- Embedding comparison available as feature-flagged upgrade if accuracy insufficient

Feature-gated by DRIFT_DETECTION_ENABLED.
"""

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field

from app.core.config import (
    DRIFT_CUMULATIVE_MAX,
    DRIFT_SINGLE_STEP_MAX,
    DRIFT_VELOCITY_MAX,
)

logger = logging.getLogger(__name__)


@dataclass
class DriftMeasurement:
    """Result of drift measurement on a concept's version chain."""

    concept_id: str
    version_chain: list[str] = field(default_factory=list)
    pairwise_distances: list[float] = field(default_factory=list)
    cumulative_drift: float = 0.0
    max_single_step: float = 0.0
    drift_velocity: float = 0.0
    flagged: bool = False
    flag_reason: str = ""


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for TF-IDF."""
    return re.findall(r"\b\w+\b", text.lower())


def _compute_tf(tokens: list[str]) -> dict[str, float]:
    """Compute term frequency for a token list."""
    counts = Counter(tokens)
    total = len(tokens) if tokens else 1
    return {term: count / total for term, count in counts.items()}


def _cosine_distance(tf_a: dict[str, float], tf_b: dict[str, float]) -> float:
    """Compute cosine distance (1 - cosine_similarity) between two TF vectors.

    Returns 0.0 (identical) to 1.0 (completely different).
    """
    all_terms = set(tf_a.keys()) | set(tf_b.keys())
    if not all_terms:
        return 0.0

    dot_product = sum(tf_a.get(t, 0.0) * tf_b.get(t, 0.0) for t in all_terms)
    norm_a = math.sqrt(sum(v**2 for v in tf_a.values()))
    norm_b = math.sqrt(sum(v**2 for v in tf_b.values()))

    if norm_a == 0 or norm_b == 0:
        return 1.0

    similarity = dot_product / (norm_a * norm_b)
    return max(0.0, min(1.0, 1.0 - similarity))


def measure_drift(concept_id: str) -> DriftMeasurement:
    """Measure semantic drift across a concept's version chain.

    Algorithm:
    1. Load concept version chain from concept_versions_archive + current
    2. For each consecutive pair, compute TF-IDF cosine distance on summaries
    3. Also compute distance from v1 to vN (cumulative)
    4. If any threshold exceeded → flag as DRIFTED

    Returns:
        DriftMeasurement with distances and flag status.
    """
    result = DriftMeasurement(concept_id=concept_id)

    try:
        from app.storage import _get_connection

        conn = _get_connection()

        # Get current version
        current = conn.execute(
            "SELECT id, version, summary FROM concepts WHERE id = ? AND is_current = 1",
            (concept_id,),
        ).fetchone()

        if not current:
            return result

        # Get archived versions (oldest first)
        archived = []
        try:
            archived = conn.execute(
                """SELECT id, version, summary FROM concept_versions_archive
                   WHERE id = ? ORDER BY created_at ASC""",
                (concept_id,),
            ).fetchall()
        except Exception:
            pass  # Archive table may not exist

        # Build version chain: archived versions + current
        chain = []
        for row in archived:
            summary = row[2] if row[2] else ""
            if not summary:
                # Try to extract from data JSON
                try:
                    import json

                    data_row = conn.execute(
                        "SELECT data FROM concept_versions_archive WHERE id = ? AND version = ?",
                        (row[0], row[1]),
                    ).fetchone()
                    if data_row and data_row[0]:
                        data = json.loads(data_row[0])
                        summary = data.get("summary", "")
                except Exception:
                    pass
            chain.append({"version": row[1], "summary": summary})

        # Add current version
        chain.append({"version": current[1], "summary": current[2] or ""})

        result.version_chain = [v["version"] for v in chain]

        if len(chain) < 2:
            return result  # No drift possible with only 1 version

        # Compute TF vectors for all versions
        tf_vectors = []
        for v in chain:
            tokens = _tokenize(v["summary"])
            tf_vectors.append(_compute_tf(tokens))

        # Pairwise distances (consecutive)
        for i in range(len(tf_vectors) - 1):
            dist = _cosine_distance(tf_vectors[i], tf_vectors[i + 1])
            result.pairwise_distances.append(round(dist, 4))

        # Cumulative drift: distance from first to last
        result.cumulative_drift = round(_cosine_distance(tf_vectors[0], tf_vectors[-1]), 4)

        # Max single step
        result.max_single_step = max(result.pairwise_distances) if result.pairwise_distances else 0.0

        # Velocity: average drift per step
        num_steps = len(result.pairwise_distances)
        result.drift_velocity = round(sum(result.pairwise_distances) / num_steps if num_steps > 0 else 0.0, 4)

        # Check thresholds
        if result.cumulative_drift > DRIFT_CUMULATIVE_MAX:
            result.flagged = True
            result.flag_reason = f"Cumulative drift {result.cumulative_drift:.2f} > {DRIFT_CUMULATIVE_MAX}"
        elif result.max_single_step > DRIFT_SINGLE_STEP_MAX:
            result.flagged = True
            result.flag_reason = f"Single step drift {result.max_single_step:.2f} > {DRIFT_SINGLE_STEP_MAX}"
        elif result.drift_velocity > DRIFT_VELOCITY_MAX:
            result.flagged = True
            result.flag_reason = f"Drift velocity {result.drift_velocity:.2f} > {DRIFT_VELOCITY_MAX}"

        if result.flagged:
            logger.warning(
                "Drift detected for %s: %s (chain=%s, cumulative=%.2f, velocity=%.2f)",
                concept_id,
                result.flag_reason,
                result.version_chain,
                result.cumulative_drift,
                result.drift_velocity,
            )
            # A5 fix: Actually downgrade maturity to PROVISIONAL
            # Guard: Don't downgrade QUARANTINED concepts (already in worse state)
            try:
                rows_updated = conn.execute(
                    "UPDATE concepts SET maturity = 'PROVISIONAL', "
                    "data = json_set(data, '$.maturity', 'PROVISIONAL') "
                    "WHERE id = ? AND maturity != 'QUARANTINED'",
                    (concept_id,),
                ).rowcount
                conn.commit()
                if rows_updated:
                    logger.info(
                        "Drift: Downgraded %s to PROVISIONAL (cumulative=%.3f)",
                        concept_id,
                        result.cumulative_drift,
                    )
            except Exception as e:
                logger.warning("Drift: Failed to downgrade %s maturity: %s", concept_id, e)

    except Exception as e:
        logger.error("Drift measurement failed for %s: %s", concept_id, e)

    return result
