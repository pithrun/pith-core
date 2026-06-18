"""Diagnostic source-set answer contract helpers.

This module is intentionally pure. It defines the Track C insufficiency/refusal
prompt candidate and response classification used by eval tooling, but it does
not call an LLM, read environment variables, touch storage, emit metrics, or wire
itself into runtime answer generation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

SOURCE_SET_ANSWER_CONTRACT_SCHEMA_VERSION = "source_set_answer_contract.diagnostic.v1"
INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"


@dataclass(frozen=True)
class SourceSetAnswerContractDecision:
    """Classification for an insufficiency/refusal answer-contract response."""

    answer: str
    refused: bool
    reason: str

    def to_dict(self) -> dict[str, str | bool]:
        return asdict(self)


def build_insufficiency_refusal_prompt(query: str, numbered_context: str) -> str:
    """Build the diagnostic-only insufficiency/refusal prompt."""
    return "\n".join(
        [
            "Answer the query using only the numbered context lines.",
            "Return answer-bearing text, not citations alone.",
            "Every answer phrase must be directly supported by the context.",
            f"If the context does not contain the answer, return exactly: {INSUFFICIENT_CONTEXT}.",
            "Do not infer from workflow patterns, row order, expected IDs, or prior answers.",
            "Do not include diagnostic labels, row IDs, or support-credit metadata.",
            "",
            "## Query",
            query,
            "",
            "## Context",
            numbered_context,
            "",
            "## Output",
            "### Answer:",
        ]
    )


def classify_insufficiency_refusal_answer(
    answer: str | None,
) -> SourceSetAnswerContractDecision:
    """Classify whether the diagnostic answer refused for insufficient context."""
    cleaned = (answer or "").strip()
    if not cleaned:
        return SourceSetAnswerContractDecision(answer="", refused=True, reason="empty_answer")
    if cleaned.upper() == INSUFFICIENT_CONTEXT:
        return SourceSetAnswerContractDecision(
            answer=cleaned,
            refused=True,
            reason="insufficient_context",
        )
    return SourceSetAnswerContractDecision(answer=cleaned, refused=False, reason="answer_present")
