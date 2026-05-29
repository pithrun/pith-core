"""Core package (Layer 0) — zero internal dependencies.

Contains configuration, constants, data models, datetime utilities,
logging, profile resolution, batch utilities, and git cache.
"""
from app.core.config import FEATURE_FLAGS, get_feature_flag  # noqa: F401
from app.core.constants import (  # noqa: F401
    TYPE_AUTHORITY_CAPS,
    FRESHNESS_JUST_NOW_MINS,
    MINUTES_PER_HOUR,
)
from app.core.models import Concept, SearchResult, VerbatimFragment, ConceptProposal  # noqa: F401
from app.core.datetime_utils import _utc_now, _utc_now_iso, _ensure_aware  # noqa: F401
from app.core.format_helpers import format_for_compaction_survival  # noqa: F401
