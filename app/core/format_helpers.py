"""Format helpers extracted from SessionManager for cross-module use.

Extracted per A9 amendment of PITH_MODULARITY_AUDIT_AND_PLAN_v2.md to break
the storage→session circular dependency. _format_for_compaction_survival was
originally SessionManager._format_for_compaction_survival (@staticmethod at
session.py:8558).
"""


def format_for_compaction_survival(concept_id: str, summary: str, concept_type: str) -> str:
    """Format critical concept for maximum survival through context summarization.

    Uses structured markers that LLM summarizers tend to preserve at
    higher fidelity than prose. Gated behind COMPACTION_SURVIVAL_FORMAT flag.

    Budget: ~10-20 extra tokens per critical concept.
    """
    if concept_type not in ("constraint", "decision", "principle"):
        return summary

    lines = [f'[CRITICAL-CONTEXT id="{concept_id}"]']
    # Preserve the most important 200 chars
    trimmed = summary[:200]
    lines.append(trimmed)
    lines.append("[/CRITICAL-CONTEXT]")
    return "\n".join(lines)
