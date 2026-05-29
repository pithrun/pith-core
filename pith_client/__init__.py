"""pith-client — Python SDK for the Pith Cognitive Runtime.

Quickstart (sync)::

    from pith_client import PithClient

    pith = PithClient(api_key="your-key")
    pith.session_start()
    resp = pith.conversation_turn("What do I know about X?")
    print(resp.activated_concepts)
    pith.session_end()

Quickstart (async)::

    from pith_client import AsyncPithClient

    async with AsyncPithClient(api_key="your-key") as pith:
        await pith.session_start()
        resp = await pith.conversation_turn("...")
        await pith.session_end()
"""
__version__ = "0.1.0"

from ._client import PithClient
from ._async_client import AsyncPithClient

from .models import (
    # Tier 1
    SessionResponse, ConversationTurnResponse, LearnResponse,
    SearchResult, Concept, StatsResponse, HealthResponse,
    OrientResponse, CheckpointResponse, SessionsListResponse,
    ConceptWriteResponse,
    # Tier 2
    LinkResponse, Question, ValidationResult, BeliefDiffResponse,
    ImportResponse, CKO, CKOListResponse, ThreadsResponse,
    TracesResponse, LearningMetricsResponse,
    # Tier 3
    MetricsDashboard, BackgroundTasksResponse,
    MetricsSummaryResponse, HealthTrendResponse,
    BenchmarkResponse, MigrationResponse,
    # Shared
    VerbatimFragment, Constraint, ConstraintSet,
    SessionInfo,
)
from .exceptions import (
    PithError,
    PithAPIError,
    PithAuthError,
    PithTimeoutError,
    PithConnectionError,
)

__all__ = [
    # Clients
    "PithClient", "AsyncPithClient",
    # Models
    "SessionResponse", "ConversationTurnResponse", "LearnResponse",
    "SearchResult", "Concept", "StatsResponse", "HealthResponse",
    "OrientResponse", "CheckpointResponse", "SessionsListResponse",
    "ConceptWriteResponse",
    "LinkResponse", "Question", "ValidationResult", "BeliefDiffResponse",
    "ImportResponse", "CKO", "CKOListResponse", "ThreadsResponse",
    "TracesResponse", "LearningMetricsResponse",
    "MetricsDashboard", "BackgroundTasksResponse",
    "MetricsSummaryResponse", "HealthTrendResponse",
    "BenchmarkResponse", "MigrationResponse",
    "VerbatimFragment", "Constraint", "ConstraintSet", "SessionInfo",
    # Exceptions
    "PithError", "PithAPIError", "PithAuthError",
    "PithTimeoutError", "PithConnectionError",
]
