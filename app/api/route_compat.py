"""Route naming normalization — backward-compatible alias middleware.

Provides a ROUTE_RENAME_MAP of old→new route paths and a middleware that
transparently rewrites requests from old paths to new canonical paths.
This allows incremental migration: rename routes in server.py one at a time,
and old callers (server.js, tests, external clients) keep working.

Usage in server.py:
    from app.api.route_compat import install_route_compat_middleware
    install_route_compat_middleware(app)

The middleware logs a deprecation warning on each redirected request so
we can track migration progress via log grep.
"""

import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger(__name__)

# ── Canonical rename map: old_path → new_path ──────────────────────
# Routes are migrated by:
#   1. Renaming the @app decorator in server.py to new path
#   2. Adding entry here (old → new) so old callers still work
#   3. Updating server.js callPithAPI() call (when ready)
#   4. Removing the entry once all callers are updated
#
# IMPORTANT: Only add an entry here AFTER the route has been renamed
# in server.py. The middleware rewrites old→new, so the new path must
# exist as an actual route. Adding entries for not-yet-renamed routes
# will cause 404s.
#
# Infrastructure routes (/health, /status, /dashboard) are intentionally
# excluded — they follow industry convention, not pith_ prefix.
ROUTE_RENAME_MAP: dict[str, str] = {
    # ── Entries added as routes are migrated ──
    # Example (uncomment when route is renamed in server.py):
    # "/session_start": "/pith_session_start",
}

# ── Full planned migration map (reference only — NOT active) ──────
# This documents ALL routes that will eventually be renamed.
# Move entries from here to ROUTE_RENAME_MAP as each route is migrated.
PLANNED_RENAMES: dict[str, str] = {
    # Session routes
    "/session_start": "/pith_session_start",
    "/session_end": "/pith_session_end",
    "/conversation_turn": "/pith_conversation_turn",
    "/session_learn": "/pith_session_learn",
    "/sessions_list": "/pith_sessions_list",
    "/checkpoint": "/pith_checkpoint",
    # Directive routes
    "/directives": "/pith_directives",
    "/directive/": "/pith_directive/",
    # Maintenance / ops routes
    "/maintenance": "/pith_maintenance",
    "/maintenance/status": "/pith_maintenance_status",
    "/backfill/run": "/pith_backfill_run",
    "/backfill/status": "/pith_backfill_status",
    "/backfill/rollback": "/pith_backfill_rollback",
    # Metrics routes
    "/learning_metrics": "/pith_learning_metrics",
    "/memory_projection": "/pith_memory_projection",
    "/retrieval_distribution": "/pith_retrieval_distribution",
    "/metrics/dashboard": "/pith_metrics_dashboard",
    "/metrics/bg_tasks": "/pith_metrics_bg_tasks",
    "/metrics/summary": "/pith_metrics_summary",
    "/metrics/health_trend": "/pith_metrics_health_trend",
    "/metrics/compaction_summary": "/pith_metrics_compaction_summary",
    "/metrics/governance_summary": "/pith_metrics_governance_summary",
    "/metrics/session_activity": "/pith_metrics_session_activity",
    "/metrics/graduation_stats": "/pith_metrics_graduation_stats",
    "/health/summary": "/pith_health_summary",
    "/health/maintenance": "/pith_health_maintenance",
    "/health/backup": "/pith_health_backup",
    "/healthz": "/pith_healthz",
    "/readyz": "/pith_readyz",
    # Association routes
    "/auto_associate_batch": "/pith_auto_associate_batch",
    "/auto_associate/": "/pith_auto_associate/",
    # Knowledge area routes
    "/knowledge_areas": "/pith_knowledge_areas",
    "/knowledge_areas/{area_name}": "/pith_knowledge_areas/{area_name}",
    # Governance routes
    "/validate_response": "/pith_validate_response",
    "/belief_diff": "/pith_belief_diff",
    "/migrate_epistemic_networks": "/pith_migrate_epistemic",
    # Agent token routes
    "/agent_tokens": "/pith_agent_tokens",
    "/agent_tokens/resolve": "/pith_agent_tokens_resolve",
    "/agent_tokens/{token}": "/pith_agent_tokens/{token}",
    # Profile routes
    "/profiles": "/pith_profiles",
    "/profile": "/pith_profile",
    # Thread reorg routes
    "/thread_reorg/mine": "/pith_thread_reorg_mine",
    "/thread_reorg/batch/preview": "/pith_thread_reorg_batch_preview",
    "/thread_reorg/batch/preview_residual": "/pith_thread_reorg_batch_preview_residual",
    "/thread_reorg/batch/commit": "/pith_thread_reorg_batch_commit",
    "/thread_reorg/batch/status": "/pith_thread_reorg_batch_status",
    "/thread_reorg/batch/rollback": "/pith_thread_reorg_batch_rollback",
    # Verbatim routes
    "/verbatim/store": "/pith_verbatim_store",
    "/verbatim/{concept_id}": "/pith_verbatim/{concept_id}",
    "/verbatim/{fragment_id}": "/pith_verbatim/{fragment_id}",
    # Normalize /pith/X → /pith_X
    "/pith/quarantine": "/pith_quarantine",
    "/pith/policy/rejections": "/pith_policy_rejections",
    "/pith/benchmark": "/pith_benchmark",
    "/pith/cko": "/pith_cko",
    "/pith/policies": "/pith_policies",
}


# Build reverse map for quick lookup
_REVERSE_MAP: dict[str, str] = {v: k for k, v in ROUTE_RENAME_MAP.items()}

# Deprecation counter for monitoring
_deprecation_hits: dict[str, int] = {}


def _resolve_path(path: str) -> tuple[str, bool]:
    """Resolve an incoming path to the canonical path.

    Returns (canonical_path, was_redirected).
    Handles both exact matches and prefix matches for parameterized routes.
    """
    # Exact match
    if path in ROUTE_RENAME_MAP:
        return ROUTE_RENAME_MAP[path], True

    # Prefix match for parameterized routes (e.g., /directive/abc → /pith_directive/abc)
    for old_prefix, new_prefix in ROUTE_RENAME_MAP.items():
        if old_prefix.endswith("/") and path.startswith(old_prefix):
            suffix = path[len(old_prefix):]
            return new_prefix + suffix, True

    # Path with query string
    if "?" in path:
        base, qs = path.split("?", 1)
        resolved, was_redirect = _resolve_path(base)
        if was_redirect:
            return f"{resolved}?{qs}", True

    return path, False


class RouteCompatMiddleware(BaseHTTPMiddleware):
    """Middleware that rewrites deprecated route paths to canonical pith_ paths.

    Transparent to callers — no 301/302 redirect, just internal rewrite.
    Logs deprecation warnings for monitoring migration progress.
    """

    async def dispatch(self, request: Request, call_next):
        original_path = request.scope["path"]
        canonical, was_redirected = _resolve_path(original_path)

        if was_redirected:
            # Rewrite the path in-place (no HTTP redirect)
            request.scope["path"] = canonical

            # Track deprecation hits
            _deprecation_hits[original_path] = _deprecation_hits.get(original_path, 0) + 1
            hit_count = _deprecation_hits[original_path]

            # Log first hit and every 100th hit (avoid log spam)
            if hit_count == 1 or hit_count % 100 == 0:
                logger.warning(
                    f"ROUTE-COMPAT: Deprecated path '{original_path}' → '{canonical}' "
                    f"(hit #{hit_count}). Update caller to use canonical path."
                )

        return await call_next(request)


def install_route_compat_middleware(app) -> None:
    """Install the backward-compat route rewriting middleware."""
    app.add_middleware(RouteCompatMiddleware)
    logger.info(
        f"ROUTE-COMPAT: Installed with {len(ROUTE_RENAME_MAP)} route aliases. "
        "Old paths will be transparently rewritten."
    )


def get_deprecation_stats() -> dict:
    """Return deprecation hit counts for monitoring."""
    return {
        "total_hits": sum(_deprecation_hits.values()),
        "unique_paths": len(_deprecation_hits),
        "top_paths": sorted(
            _deprecation_hits.items(), key=lambda x: x[1], reverse=True
        )[:10],
    }
