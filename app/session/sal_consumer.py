"""SAL V1 Consumer — transforms SAL output into LLM-consumable context.

Pure transformation layer: SAL dict -> formatted string.
No state, no DB access, no LLM calls, no side effects.

Sprint: SAL-V1-SPRINT-A
Items: RETRIEVAL-059, RETRIEVAL-060, RETRIEVAL-061, RETRIEVAL-062
"""

import logging

logger = logging.getLogger(__name__)

# Minimum clusters for structured (non-flat) formatting
SAL_MIN_CLUSTERS_FOR_STRUCTURE = 3
# Maximum summary length before truncation
SAL_MAX_SUMMARY_LENGTH = 500


def format_sal_context(sal_result: dict | None) -> str | None:
    """Transform SAL output dict into LLM-consumable context string.

    Returns:
        Formatted context string, or None if SAL disabled/failed/below threshold.

    RETRIEVAL-059: Selective activation — structured format only when >=3 clusters.
    RETRIEVAL-060: Cluster topic labels from dominant knowledge_area.
    RETRIEVAL-061: Surprise buffer explicit framing.
    RETRIEVAL-062: Attention-weighted ordering within clusters.
    """
    if sal_result is None:
        return None
    if sal_result.get("fallback_used", False):
        return None
    clusters = sal_result.get("clusters", [])
    if len(clusters) < SAL_MIN_CLUSTERS_FOR_STRUCTURE:
        return None

    grouped = _group_clusters_by_topic(clusters)
    sections = []
    sections.append("## Activated Knowledge (structured by topic)\n")

    for i, (topic_label, concepts) in enumerate(grouped, 1):
        sections.append(_format_cluster(i, topic_label, concepts))

    surprise = sal_result.get("surprise_buffer", [])
    if surprise:
        sections.append(_format_surprise_buffer(surprise))

    conf_env = sal_result.get("confidence_envelope", 0.0)
    if conf_env > 0:
        sections.append(f"\n_Overall confidence: {conf_env:.0%}_")

    return "\n".join(sections)


def _group_clusters_by_topic(clusters: list[dict]) -> list[tuple[str, list[dict]]]:
    """Group clusters by knowledge_area, sort by max attention weight."""
    groups: dict[str, list[dict]] = {}
    for c in clusters:
        ka = c.get("knowledge_area", "") or "general"
        if ka not in groups:
            groups[ka] = []
        groups[ka].append(c)

    result = []
    for ka, concepts in groups.items():
        concepts.sort(key=lambda c: c.get("attention_weight", 0), reverse=True)
        topic_label = _derive_topic_label(ka, concepts)
        result.append((topic_label, concepts))

    result.sort(key=lambda t: max(c.get("attention_weight", 0) for c in t[1]), reverse=True)
    return result


def _derive_topic_label(knowledge_area: str, concepts: list[dict]) -> str:
    """Derive human-readable topic label from knowledge_area."""
    if not knowledge_area or knowledge_area == "general":
        return "General"
    return knowledge_area.replace("_", " ").title()


def _format_cluster(index: int, topic_label: str, concepts: list[dict]) -> str:
    """Format a single topic cluster with attention-weighted ordering."""
    lines = [f"\n### Topic {index}: {topic_label}\n"]
    for c in concepts:
        weight = c.get("attention_weight", 0)
        summary = c.get("summary", "")
        if len(summary) > SAL_MAX_SUMMARY_LENGTH:
            summary = summary[:SAL_MAX_SUMMARY_LENGTH - 3] + "..."
        relevance_label = (
            "HIGH" if weight > 0.1
            else "MEDIUM" if weight > 0.05
            else "LOW"
        )
        lines.append(f"- [{relevance_label}] {summary}")
    return "\n".join(lines)


def _format_surprise_buffer(surprise: list[dict]) -> str:
    """Format surprise buffer with explicit peripheral framing."""
    lines = [
        "\n### Peripheral Concepts",
        "_The following are outside the main topics but may offer"
        " unexpected connections or alternative perspectives._\n",
    ]
    for s in surprise:
        summary = s.get("summary", "")
        if len(summary) > SAL_MAX_SUMMARY_LENGTH:
            summary = summary[:SAL_MAX_SUMMARY_LENGTH - 3] + "..."
        lines.append(f"- {summary}")
    return "\n".join(lines)
