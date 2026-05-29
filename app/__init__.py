"""
Pith - Personal Knowledge Server

A versioned, conceptual memory system that enables AI to learn and evolve
knowledge over time with epistemic humility, self-awareness, and active learning.
"""

from importlib import import_module

__version__ = "1.0.0"
__author__ = "Pith Contributors"
__license__ = "MIT"

__all__ = [
    "app",
    "Concept",
    "ConceptProposal",
    "ConceptEvolution",
    "Hypothesis",
    "SearchQuery",
    "SearchResult",
    "create_concept",
    "evolve_concept",
    "retrieval_engine",
    "load_concept",
    "save_concept",
    "list_concepts",
]


_LAZY_EXPORTS = {
    "create_concept": ("app.cognitive.learning", "create_concept"),
    "evolve_concept": ("app.cognitive.learning", "evolve_concept"),
    "Concept": ("app.core.models", "Concept"),
    "ConceptProposal": ("app.core.models", "ConceptProposal"),
    "ConceptEvolution": ("app.core.models", "ConceptEvolution"),
    "Hypothesis": ("app.core.models", "Hypothesis"),
    "SearchQuery": ("app.core.models", "SearchQuery"),
    "SearchResult": ("app.core.models", "SearchResult"),
    "retrieval_engine": ("app.retrieval", "retrieval_engine"),
    "app": ("app.api.server", "app"),
    "load_concept": ("app.storage", "load_concept"),
    "save_concept": ("app.storage", "save_concept"),
    "list_concepts": ("app.storage", "list_concepts"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
