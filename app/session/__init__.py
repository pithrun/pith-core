"""Session middleware — re-exports for backward compatibility.

All code has been decomposed into mixin modules per ARCH-009:
  session/helpers.py    — free functions, constants, _BudgetSkip
  session/orient.py     — OrientMixin (present-moment orientation)
  session/lifecycle.py  — LifecycleMixin (start/end session)
  session/turn.py       — ConversationTurnMixin (conversation_turn pipeline)
  session/learn.py      — SessionLearnMixin (session_learn pipeline)
  session/manager.py    — SessionManager (thin orchestrator inheriting all mixins)

Existing modules (unchanged):
  session/bridge.py     — Bridge between session and retrieval
  session/bootstrap.py  — Cognitive bootstrap for new sessions
  session/self_model.py — Self-model management
  session/sal_consumer.py — SAL consumer integration
"""

# Re-export SessionManager (the main class)
from app.session.manager import SessionManager  # noqa: F401

# Re-export free functions used by tests and production code
from app.session.helpers import (  # noqa: F401
    _BudgetSkip,
    _get_coverage_client,
    _is_not_resolved,
    _decompose_query_llm,
    _is_strategic,
    _validate_concept_type,
    _compute_freshness,
    _has_named_entities,
    _sk_content_word_count,
    _extract_subject_key,
    _conflict_prefilter,
    _extract_object_value,
    _chain_aware_prune,
    _TEMPORAL_MEMORY_QUERY,
    _CONTRADICTED_S4_MULTIPLIER,
    _SUPERSEDED_S4_MULTIPLIER,
    DEFAULT_WINDOW,
    TIME_WINDOWS,
    RECENCY_WINDOW_HOURS,
    RECENCY_MIN_CONFIDENCE,
    RECENCY_MAX_INJECT,
    RECENCY_RELEVANCE_SCORE,
    QUARANTINE_RECENCY_EXEMPT_HOURS,
    ORIENTATION_EXCLUDE_PATTERNS,
    _RESOLVED_PATTERNS,
)

# Re-export storage functions that tests patch via app.session namespace
from app.storage import (  # noqa: F401
    _get_connection,
    count_sessions,
    load_resume_snapshot,
    update_session,
    save_session,
    save_concept,
    load_concept,
    list_concepts,
    load_recent_concepts,
    load_associations,
    load_session_velocity,
    recover_interrupted_sessions,
    get_related_concepts,
    count_associations,
    cleanup_expired_snapshots,
    save_resume_snapshot,
)

# Re-export models used by some test co-imports
from app.core.models import SessionEndRequest  # noqa: F401

# Re-export learn module-level counter (accessed by tests via app.session namespace)
from app.session.learn import _PRECISION_GUARD_BLOCKS  # noqa: F401

# Singleton instance — server.py:74 imports this (16 usages in server.py)
# GAUNTLET A1: Added per gauntlet finding F0.1.
session_manager = SessionManager()
