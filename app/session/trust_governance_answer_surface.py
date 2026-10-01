"""Conversation-turn answer surface for trust governance."""

from __future__ import annotations

from typing import Any


def build_trust_governance_answer_surface(
    query_text: str,
    *,
    engine_chain_answer: str | None = None,
    engine_chain_answer_diagnostics: dict[str, Any] | None = None,
) -> tuple[str | None, dict[str, Any] | None]:
    """Return a governed answer or trust diagnostics for trust-relevant queries."""
    from app.authority_chain import build_authority_chain_answer
    from app.trust_governance import is_trust_governance_intent_query

    answer, diagnostics = build_authority_chain_answer(query_text)
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    should_surface = bool(answer) or (
        is_trust_governance_intent_query(query_text)
        and diagnostics.get("mode") == "trust_governance_registry"
        and diagnostics.get("intent") in {"no_known_authority", "ambiguous_domain"}
    )
    if not should_surface:
        return None, None

    surfaced = dict(diagnostics)
    if engine_chain_answer_diagnostics:
        surfaced["engine_chain_answer_diagnostics"] = engine_chain_answer_diagnostics
    if engine_chain_answer and not answer:
        surfaced["engine_chain_answer"] = engine_chain_answer
    return answer, surfaced
