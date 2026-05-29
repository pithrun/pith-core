"""Hot-path answer-path admission for optional conversation-turn work."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

SMALL = "small"
STANDARD = "standard"
DEEP = "deep"
FIRST_CALL_RESUMPTION = "first_call_resumption"

_DEEP_ROUTER_SIGNALS = {"counting", "relational", "temporal", "recall"}
_DEEP_MARKERS = re.compile(
    r"\b("
    r"list\s+all|what\s+are\s+all|every\s+[a-z0-9_-]{1,30}|"
    r"count|how many|timeline|history|yesterday|last time|"
    r"since|previous|prior|remember|remind|mentioned|discussed|constraints?|"
    r"contradiction|deadlock|root cause|rca|multi-hop|relationship"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AnswerPathAdmission:
    """Admission decision for optional answer-path enrichment phases."""

    mode: str
    reason: str
    allow_multihop: bool
    allow_entity_chain: bool
    allow_graph: bool
    allow_optional_injection: bool
    max_concepts_cap: int | None = None
    observe_only: bool = True

    def labels(self) -> dict[str, str]:
        """Return safe metric labels without raw query text."""
        return {
            "mode": self.mode,
            "reason": self.reason,
            "observe_only": str(self.observe_only).lower(),
        }

    def allows_optional_phase(self, phase: str, *, enforce_standard_optional: bool = False) -> bool:
        """Return whether an optional phase may run for this answer path."""
        if self.mode == STANDARD and enforce_standard_optional:
            if phase in {"multihop.retrieve", "entity_chain.retrieve", "S4_graph_walk"}:
                return False
            return not (phase.startswith("injection.") or phase.startswith("assembly."))
        if self.mode != SMALL:
            return True
        if phase == "multihop.retrieve":
            return self.allow_multihop
        if phase == "entity_chain.retrieve":
            return self.allow_entity_chain
        if phase == "S4_graph_walk":
            return self.allow_graph
        if phase.startswith("injection.") or phase.startswith("assembly."):
            return self.allow_optional_injection
        return True


def _router_attr(config: Any, name: str, default: Any = None) -> Any:
    if config is None:
        return default
    return getattr(config, name, default)


def classify_answer_path(
    message: str | None,
    *,
    adaptive_config: Any = None,
    effective_max_concepts: int,
    first_call_hint: bool,
    resumption_hint: bool = False,
    observe_only: bool = True,
) -> AnswerPathAdmission:
    """Classify a turn into small, standard, deep, or first-call paths.

    This function is intentionally pure and fast: no storage, LLM, or app
    service imports belong here.
    """
    if first_call_hint:
        return AnswerPathAdmission(
            mode=FIRST_CALL_RESUMPTION,
            reason="first_call",
            allow_multihop=True,
            allow_entity_chain=True,
            allow_graph=True,
            allow_optional_injection=True,
            observe_only=observe_only,
        )
    if resumption_hint:
        return AnswerPathAdmission(
            mode=FIRST_CALL_RESUMPTION,
            reason="resumption",
            allow_multihop=True,
            allow_entity_chain=True,
            allow_graph=True,
            allow_optional_injection=True,
            observe_only=observe_only,
        )

    text = (message or "").strip()
    signals = set(_router_attr(adaptive_config, "signals", []) or [])
    top_k_multiplier = float(_router_attr(adaptive_config, "top_k_multiplier", 1.0) or 1.0)
    deep_router = bool(
        signals & _DEEP_ROUTER_SIGNALS
        or _router_attr(adaptive_config, "use_multihop", False)
        or _router_attr(adaptive_config, "force_entity_chain", False)
        or top_k_multiplier > 1.0
    )
    deep_marker = bool(_DEEP_MARKERS.search(text))

    if deep_router or deep_marker:
        return AnswerPathAdmission(
            mode=DEEP,
            reason="router_signal" if deep_router else "text_marker",
            allow_multihop=True,
            allow_entity_chain=True,
            allow_graph=True,
            allow_optional_injection=True,
            observe_only=observe_only,
        )

    if len(text) <= 80 and "?" not in text:
        return AnswerPathAdmission(
            mode=SMALL,
            reason="short_non_question",
            allow_multihop=False,
            allow_entity_chain=False,
            allow_graph=False,
            allow_optional_injection=False,
            max_concepts_cap=min(max(1, effective_max_concepts), 4),
            observe_only=observe_only,
        )

    return AnswerPathAdmission(
        mode=STANDARD,
        reason="default",
        allow_multihop=True,
        allow_entity_chain=True,
        allow_graph=True,
        allow_optional_injection=True,
        observe_only=observe_only,
    )
