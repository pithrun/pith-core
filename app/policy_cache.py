"""Policy Cache — In-memory cache with write-through invalidation.

Memory Integrity Spec v1.2, §5.8.6 (CM-C5):
User policy queries hit DB per-request. SQLite bottleneck at 1K concurrent agents.
Solution: In-memory cache with TTL-based expiry and write-invalidation.

Also caches violation stats to avoid repeated expensive aggregation queries.

Phase 3: Infrastructure + violation stats caching.
Phase 4: User-configurable policy caching (Amendment 2).
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

from app.config import FEATURE_FLAGS

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

POLICY_CACHE_TTL_SECONDS = 300  # 5 minutes default TTL
POLICY_CACHE_MAX_ENTRIES = 100  # Max org_id entries before LRU eviction
STATS_CACHE_TTL_SECONDS = 60  # 1 minute TTL for violation stats (more volatile)


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class UserPolicy:
    """User-configurable policy rule (Phase 3 complete).

    Loaded from user_policies table via app.user_policies module.
    Used by PolicyEngine to enforce custom retention, privacy,
    and behavior rules.
    """

    rule_id: str = ""
    policy_type: str = ""  # "retention" | "privacy" | "behavior"
    rule_text: str = ""  # Human-readable rule description
    action: dict = None  # What to do when triggered
    condition: dict = None  # Optional filter for when to apply
    severity: str = "WARN"  # Default severity for policy violations
    enabled: bool = True
    priority: int = 50  # Higher = evaluated first


@dataclass
class CacheEntry:
    """Generic cache entry with TTL tracking."""

    data: Any
    cached_at: float  # time.time() when cached
    ttl_seconds: float  # TTL for this entry
    access_count: int = 0  # Track access frequency for metrics

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.cached_at) > self.ttl_seconds


# =============================================================================
# PolicyCache
# =============================================================================


class PolicyCache:
    """In-memory policy cache with write-through invalidation (CM-C5).

    Thread-safe. Supports:
    - Policy caching by org_id (Phase 4 prep)
    - Violation stats caching (immediate value)
    - TTL-based expiry with configurable per-entry TTL
    - Write-invalidation on policy changes
    - LRU eviction when max entries exceeded
    - Cache metrics for monitoring
    """

    def __init__(self, ttl_seconds: float = POLICY_CACHE_TTL_SECONDS):
        self._policy_cache: dict[str, CacheEntry] = {}  # org_id → CacheEntry(List[UserPolicy])
        self._stats_cache: dict[str, CacheEntry] = {}  # stat_key → CacheEntry(dict)
        self._default_ttl = ttl_seconds
        self._lock = threading.Lock()

        # Metrics
        self._hits = 0
        self._misses = 0
        self._invalidations = 0
        self._evictions = 0

    # -------------------------------------------------------------------------
    # Policy caching (Phase 4 prep — currently returns empty lists)
    # -------------------------------------------------------------------------

    def get_policies(self, org_id: str, conn=None) -> list[UserPolicy]:
        """Get policies from cache, refresh from DB on miss or TTL expiry.

        Args:
            org_id: Organization identifier.
            conn: SQLite connection (used on cache miss to load from DB).

        Returns:
            List of UserPolicy objects for this org.
        """
        if not FEATURE_FLAGS.get("POLICY_CACHE_ENABLED", False):
            return self._load_policies_from_db(org_id, conn)

        with self._lock:
            entry = self._policy_cache.get(org_id)
            if entry and not entry.is_expired:
                entry.access_count += 1
                self._hits += 1
                return entry.data

        # Cache miss or expired — load from DB (outside lock)
        self._misses += 1
        policies = self._load_policies_from_db(org_id, conn)

        with self._lock:
            self._policy_cache[org_id] = CacheEntry(
                data=policies,
                cached_at=time.time(),
                ttl_seconds=self._default_ttl,
            )
            self._maybe_evict_policies()

        return policies

    def invalidate(self, org_id: str) -> None:
        """Called on any policy write/update/delete for this org."""
        with self._lock:
            if org_id in self._policy_cache:
                del self._policy_cache[org_id]
                self._invalidations += 1
                logger.debug("PolicyCache: invalidated org_id=%s", org_id)

    def invalidate_all(self) -> None:
        """Nuclear option — clear all cached policies."""
        with self._lock:
            count = len(self._policy_cache)
            self._policy_cache.clear()
            self._invalidations += count
            logger.info("PolicyCache: invalidated all %d entries", count)

    # -------------------------------------------------------------------------
    # Violation stats caching (immediate value)
    # -------------------------------------------------------------------------

    def get_cached_stats(self, stat_key: str) -> dict | None:
        """Get cached violation stats if available and not expired.

        Args:
            stat_key: Cache key (e.g., "violation_stats", "recent_violations:50").

        Returns:
            Cached dict or None on miss/expiry.
        """
        if not FEATURE_FLAGS.get("POLICY_CACHE_ENABLED", False):
            return None

        with self._lock:
            entry = self._stats_cache.get(stat_key)
            if entry and not entry.is_expired:
                entry.access_count += 1
                self._hits += 1
                return entry.data

        self._misses += 1
        return None

    def set_cached_stats(self, stat_key: str, data: dict, ttl_seconds: float | None = None) -> None:
        """Cache violation stats result.

        Args:
            stat_key: Cache key.
            data: Stats dict to cache.
            ttl_seconds: Override TTL (defaults to STATS_CACHE_TTL_SECONDS).
        """
        if not FEATURE_FLAGS.get("POLICY_CACHE_ENABLED", False):
            return

        ttl = ttl_seconds if ttl_seconds is not None else STATS_CACHE_TTL_SECONDS
        with self._lock:
            self._stats_cache[stat_key] = CacheEntry(
                data=data,
                cached_at=time.time(),
                ttl_seconds=ttl,
            )

    def invalidate_stats(self) -> None:
        """Invalidate all cached stats (called when new violations are logged)."""
        with self._lock:
            if self._stats_cache:
                count = len(self._stats_cache)
                self._stats_cache.clear()
                self._invalidations += count
                logger.debug("PolicyCache: invalidated %d stats entries", count)

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    def get_metrics(self) -> dict:
        """Get cache performance metrics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 3) if total > 0 else 0.0,
                "invalidations": self._invalidations,
                "evictions": self._evictions,
                "policy_entries": len(self._policy_cache),
                "stats_entries": len(self._stats_cache),
                "total_requests": total,
            }

    def reset_metrics(self) -> None:
        """Reset cache metrics (for testing)."""
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._invalidations = 0
            self._evictions = 0

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _load_policies_from_db(self, org_id: str, conn=None) -> list[UserPolicy]:
        """Load user policies from DB.

        Phase 3 COMPLETE: Queries user_policies table for enabled policies.
        org_id is currently unused (single-tenant) but preserved for future multi-tenant.
        """
        try:
            from app.user_policies import list_policies as _list_policies

            policies = _list_policies(include_disabled=False)
            # Convert to the UserPolicy format expected by PolicyCache
            return [
                UserPolicy(
                    rule_id=p.id,
                    policy_type=p.policy_type,
                    rule_text=p.rule,
                    action=p.action,
                    condition=p.condition,
                    priority=p.priority,
                )
                for p in policies
            ]
        except Exception as e:
            err_str = str(e)
            if "no such table" in err_str and "user_policies" in err_str:
                if not getattr(self, "_user_policies_warned", False):
                    logger.warning("user_policies table not found (will not warn again)")
                    self._user_policies_warned = True
            else:
                logger.warning(f"Failed to load user policies from DB: {e}")
            return []

    def _maybe_evict_policies(self) -> None:
        """LRU eviction when cache exceeds max entries. Must hold lock."""
        if len(self._policy_cache) <= POLICY_CACHE_MAX_ENTRIES:
            return

        # Evict least-recently-accessed entries
        sorted_entries = sorted(
            self._policy_cache.items(),
            key=lambda x: x[1].cached_at,
        )
        evict_count = len(self._policy_cache) - POLICY_CACHE_MAX_ENTRIES
        for i in range(evict_count):
            org_id = sorted_entries[i][0]
            del self._policy_cache[org_id]
            self._evictions += 1

        logger.debug("PolicyCache: evicted %d entries (LRU)", evict_count)


# =============================================================================
# Module-level singleton
# =============================================================================

_cache: PolicyCache | None = None


def get_policy_cache() -> PolicyCache:
    """Get or create the singleton PolicyCache instance."""
    global _cache
    if _cache is None:
        _cache = PolicyCache()
    return _cache


def reset_policy_cache() -> None:
    """Reset the singleton (for testing)."""
    global _cache
    _cache = None
