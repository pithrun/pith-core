"""
Pith - Personal Knowledge Server

A versioned, conceptual memory system that enables AI to learn and evolve
knowledge over time with epistemic humility, self-awareness, and active learning.
"""

__version__ = "1.0.0"
__author__ = "Pith Contributors"
__license__ = "MIT"

# Import main components for easier access
try:
    from app.learning import create_concept, evolve_concept
    from app.models import (
        Concept,
        ConceptEvolution,
        ConceptProposal,
        Hypothesis,
        SearchQuery,
        SearchResult,
    )
    from app.retrieval import retrieval_engine
    from app.server import app
    from app.storage import list_concepts, load_concept, save_concept
except ImportError:
    # Allow package to be imported even if dependencies aren't installed yet
    pass

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
