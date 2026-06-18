"""Governance Configuration — all tunable parameters with rationale.

Every magic number in the governance architecture is defined here with
its rationale, source, and tuning guidance. No unexplained constants.

Parameters sourced from spec Section 14.
Memory Integrity additions sourced from MEMORY_INTEGRITY_SPEC v1.2, §5.6/§5.8.
"""

import os
from dataclasses import dataclass

# =============================================================================
# FEATURE FLAGS — Memory Integrity Spec v1.2, §5.8.2
# =============================================================================
# Every defense component has a feature flag for safe rollout and instant rollback.
# Phase 0a: HARDENED_CONSTRAINTS_ENABLED controls the config hardening deployment.
# Deployment protocol (§5.6.2):
#   1. Deploy with flag = False, measure baseline constraints per turn
#   2. Enable flag = True
#   3. Monitor 48h: alert if avg constraints/turn < 2, or correction signals +50%
#   4. If alerts fire: revert flag, investigate, adjust thresholds

FEATURE_FLAGS = {
    # REFLECT-024 — Idempotent (delta-based) confidence decay (default OFF; enable after verify)
    "REFLECT_DELTA_DECAY": True,  # REFLECT-024: delta-based decay enabled (validated live A/B; migration GOV-026 applied)
    # Phase 0a — Config Hardening
    "HARDENED_CONSTRAINTS_ENABLED": True,  # Config hardening (§5.6.1)
    # Phase 1 — Write-Path Defense (ENABLED — Phase 1 implementation complete)
    "INGESTION_VALIDATION_ENABLED": True,  # Write-path validation
    "POLICY_ENGINE_ENABLED": True,  # PolicyEngine wrapper
    "DEDUP_AT_INGESTION_ENABLED": True,  # Dedup before write
    "VERSION_CHAIN_CONCURRENCY_ENABLED": True,  # §5.2.2 optimistic locking
    "EVIDENCE_ANTISPOOFING_ENABLED": True,  # §5.2.3 evidence method derivation
    "GOVERNANCE_EVENT_WIRING_ENABLED": True,  # §5.8.4 H18 typed event logging
    "INGESTION_LATENCY_OPT_ENABLED": True,  # §5.2.9 H13 contradiction cache
    "REJECTION_VISIBILITY_ENABLED": True,  # §5.2.10 H14 audit endpoint
    # Phase 2 — Temporal + Epistemic (ENABLED — Phase 2 implementation complete)
    "EPISTEMIC_CAPS_ENABLED": True,  # Epistemic authority caps
    "T1_RETROACTIVE_REFLECTION_ENABLED": False,  # REFLECT-021: disabled, 0/13 productive cycles
    "T3_SESSION_END_REFLECTION_ENABLED": False,   # REFLECT-021: disabled, 0/21 productive cycles
    "NONLOSSY_EVOLUTION_ENABLED": True,  # Non-lossy invalidation
    "TEMPORAL_CURRENCY_ENABLED": True,  # Temporal decay
    "EPISODES_ENABLED": True,  # Episode recording
    # Phase 3 — Measurement + Hardening (ENABLED — Phase 3 implementation complete)
    "POISONBENCH_CI_ENABLED": True,  # PoisonBench in CI
    "LLM_CONTRADICTION_TIER2_ENABLED": False,  # LLM-powered contradiction (opt-in: requires ANTHROPIC_API_KEY)
    "FACT_SEEKING_BOOST_ENABLED": True,  # INGEST-015: boost is_factual concepts on fact-seeking queries
    "STRUCTURAL_CONCEPT_CLASSIFIER_ENABLED": True,  # INGEST-017: Layer 1 structural concept classifier
    "STRUCTURAL_QUERY_CLASSIFIER_ENABLED": True,  # INGEST-017: Layer 2 structural query classifier
    "TIER3_LLM_EXTRACTION_ENABLED": True,  # EXP-029: Tier 3 LLM concept extraction (enabled: requires ANTHROPIC_API_KEY)
    "LLM_EXPERIMENT_RESOLUTION_ENABLED": True,  # EXP-003a: LLM experiment synthesis
    "LLM_EXPERIMENT_VERBOSE_LOG": False,  # Verbose LLM I/O logging for experiment resolution
    "FIRMWARE_DEPRECATION_ENABLED": True,  # CM-M5: firmware lifecycle management
    "POLICY_CACHE_ENABLED": True,  # CM-C5: in-memory policy cache
    # Amendment 4 gaps (ENABLED — implementation complete)
    "QUARANTINE_ENDPOINTS_ENABLED": True,  # Quarantine management
    "ROUTER_EPISTEMIC_FILTER_ENABLED": True,  # Router integration (§5.8.7)
    # Phase 3 — LLM Tier 2 + Drift + Cascade (NEW)
    "DRIFT_DETECTION_ENABLED": True,  # WS2: TF-IDF drift measurement on evolution
    "CORRECTION_CASCADE_ENABLED": True,  # WS3: Cascade propagation on corrections
    # Context Management Integration — Mid-session context resilience
    "CONTEXT_PRIORITY_HINTS_ENABLED": True,  # CTX Phase 1: Priority metadata on activated concepts
    "COMPACTION_DETECTION_ENABLED": True,  # CTX Phase 2: Heuristic compaction detection + re-injection
    "COMPACTION_SURVIVAL_FORMAT": True,  # CTX Phase 3: Structured tags for compaction survival (enabled Sprint 23E — 12-event baseline sufficient)
    "AUTO_CHECKPOINT_ENABLED": True,   # SESSION-004: Auto-save checkpoint at URGE+ context pressure and compaction
    "WORKING_CONTEXT_ENABLED": True,  # CONTEXT-001: Structured working context every turn
    # Phase 4 — Memory Integrity Phase 4 (P4-PREREQ + P4a/b/c)
    "CONSTRAINT_ASSEMBLY_ENABLED": True,  # P4-PREREQ: Enable constraint_set population
    "POST_RESPONSE_VALIDATION_ENABLED": True,  # P4a: Post-response validation (ENABLED — instrumentation mode, hard threshold at 0.80)
    "BELIEF_DIFF_ENABLED": True,  # P4b: Belief diff (read-only, low-risk)
    "EXTENDED_EPISTEMIC_NETWORKS_ENABLED": True,  # P4c: Extended epistemic networks (ENABLED — validated against 2865 production concepts)
    # Federation L2 — Cross-instance bridge
    "FEDERATION_EVENTS_ENABLED": False,  # L2: Emit federation events (MAINT-015: OFF until consumer exists)
    "SESSION_REGISTRY_ENABLED": False,  # FED-013: Session heartbeat + working context registry
    # Phase 5 — MATURITY-003: Evidence Pipeline Fix
    "EMBEDDING_DEDUP_ENABLED": True,  # Use embedding cosine for dedup (fallback: TF-IDF)
    "QUARANTINE_RELEASE_ENABLED": True,  # Quarantine release sweep in reflection
    "TEMPORAL_PROMOTION_ENABLED": True,  # Part D: Path C temporal promotion in reflection
    "EVIDENCE_BACKFILL_ENABLED": True,  # Phase A5: one-time evidence backfill for stuck concepts
    # Federation Phase 0 — KA-Relative Governance (FEDERATION_ORCHESTRATION_DESIGN v2.1)
    "KA_RELATIVE_GOVERNANCE_ENABLED": True,  # Master kill switch for all Phase 0 components
    # Currency Actuator — Signal→Action gap fix (CURRENCY_STATUS_ACTUATOR_SPEC)
    "CURRENCY_ACTUATOR_ENABLED": True,  # Phase 2.8: currency_status → status actuator sweep
    # Dynamic Knowledge Areas (KA-ARCH-001)
    "DYNAMIC_KA_ENABLED": True,         # Gates provisional KA creation in normalize_knowledge_area
    "KA_AUTO_BOOST_ENABLED": False,     # Gates auto-inference KA boost in search_lightweight + _apply_ka_boost
    "KA_CROSS_SESSION_SUPPLEMENT": True, # RETRIEVAL-032: KA-scoped supplement for cross-session coverage
    "QUERY_INTENT_EXPANSION_ENABLED": True,  # Retrieval-only alias/query expansion
    "QUERY_INTENT_RESCUE_ENABLED": True,  # Sparse-result re-query with expanded variants
    "QUERY_INTENT_TRACE_ENABLED": True,  # Trace matched aliases/variants in retrieval diagnostics
    # COVERAGE-001: 4-signal LLM coverage validator
    "COVERAGE_LLM_ENABLED": False,  # Phase 3 rollout — enable after monitoring
    # PRODUCT-002: Episodic fact granularity guard
    "EPISODIC_GRANULARITY_GUARD_ENABLED": False,  # Safe rollout — enable after monitoring
    # PERF-FORT-2: Background auto-learn — moves session_learn off critical path
    "BACKGROUND_AUTOLEARN_ENABLED": True,  # True = background (default), False = sync (rollback)
    # --- Verbatim Fragment Feature Flags ---
    # INGEST-037 (Tier 2): Selective regex extraction of code/SQL/formula/quotes.
    #   Captures high-information-density patterns only.
    "VERBATIM_AUTO_EXTRACT_ENABLED": True,  # Default OFF — enable via PITH_FF_VERBATIM_AUTO_EXTRACT_ENABLED=true
    # INGEST-038 (Tier 1): Comprehensive raw conversation text capture.
    #   Stores full user+assistant turns as fragments linked to concepts.
    #   Both are independent — Tier 1 captures the wide net, Tier 2 adds precision.
    "VERBATIM_CONVERSATION_CAPTURE_ENABLED": True,  # Default ON — this is the baseline lossless capture
    # FEEDBACK-001: L1 Retrieval Utility Signal — measures concept utilization in responses
    "FEEDBACK_L1_ENABLED": True,  # True = score every turn (default), False = disable
    # SAL V0 — Structured Activation Layer
    "SAL_ENABLED": False,  # Master toggle — OFF by default, zero overhead when disabled
    # CONTRA-018: Phase 1 removal — L1.8 ingestion-time detection replaces keyword negation
    "CONTRA_018_PHASE1_REMOVED": True,  # Phase 1 was 99% FP; L1.8 + Phase 2 handle genuine contradictions
    # RETRIEVAL-080: Feedback loop — utility flows back into retrieval + recalibration
    "FEEDBACK_LOOP_ENABLED": True,  # Master toggle for utility accumulator + retrieval integration
    # SKILL-DEPLOY-001: Concept synthesis — LLM-driven concept consolidation in maintenance
    "SYNTHESIS_LLM_ENABLED": True,
    # RETRIEVAL-095: Verbatim suppression gate — suppress verbatim PATH B for
    # counting/temporal/compositional question types that caused 4 benchmark regressions.
    # Enabled after TEST-180/TEST-182 test coverage confirmed (was OFF pending tests).
    "RETRIEVAL_095_ENABLED": True,
    # SESSION-012: Cross-session awareness — retrieval proximity boost
    "CROSS_SESSION_BOOST_ENABLED": False,  # SESSION-012: Off by default, opt-in via env
    "REPO_HYGIENE_POLICY_ENABLED": True,  # Operational policy: enforce session isolation semantics
    "DB_BOUNDARY_AUDIT_ENABLED": True,  # Shadow instrumentation for request-root DB boundary inventory
    "READ_ONLY_AGGREGATES_ENABLED": False,  # Tier 1 read-pool rollout for verified pure-read endpoints
    "READ_ONLY_AGGREGATES_FALLBACK_ALLOWED": True,  # Tier 1 safety net: fall back to legacy path if read_snapshot_db raises (parent spec §15)
    "WORKSTREAMS_READ_ENABLED": True,  # Workstreams Phase 0/1: explicit read-only classifier/context actions
    "WORKSTREAMS_WRITE_ENABLED": False,  # Workstreams Phase 2: broad explicit curation/binding writes, disabled by default
    "WORKSTREAMS_ACTIVATION_WRITE_ENABLED": False,  # Activation-only bind/create/skip writes through ensure_workstream_activation
    "WORKSTREAMS_LIFECYCLE_WRITE_ENABLED": False,  # Production lifecycle verbs: start/adopt/progress/complete/reopen/archive
    "WORKSTREAMS_TURN_CONTEXT_ENABLED": False,  # Workstreams Phase 3: explicit active Workstream injection into conversation_turn
    "WORKSTREAMS_ACTIVATION_HINT_ENABLED": False,  # Workstreams API parity: compact read-only activation state hint
    "WORKSTREAMS_NON_EXACT_RECOMMENDATIONS_ENABLED": False,  # Default-safe: lexical candidates are advisory, not bind authority
    # STABILITY-048: Operation-class storage boundary Segment 1.
    # Defaults are off/shadow so the sidecar command log cannot alter live memory
    # behavior until append latency, replay, and no-migration gates pass.
    "OPERATION_COMMAND_LOG_ENABLED": False,
    "OPERATION_COMMAND_LOG_SHADOW": True,
    "OBSERVABILITY_SIDECAR_ENABLED": False,
    "COMMAND_WRITER_DRAIN_ENABLED": False,
    "COMMAND_PRODUCER_ADMISSION_ENABLED": False,
    "SYNTHETIC_COMMAND_PRODUCER_ENABLED": False,
    "OBSERVABILITY_COMMAND_PRODUCER_ENABLED": False,
    "COMMAND_PRODUCER_FOREGROUND_APPEND_ENABLED": False,
    "COMMAND_LOG_STRESS_MODE": False,
    # COGGOV-006/007/008: Auto-correction pipeline (DISABLED — E2E test revealed data corruption bugs)
    "COGGOV_006_AI_SELF_CORRECTION": True,   # Layer 5 detection is safe (corroboration only, 0.65 cap)
    "COGGOV_007_CORRECTION_EVOLUTION": True,  # COGGOV-012 D+a: zero-extraction, evidence-append only. Safe.
    "COGGOV_013_CORRECTION_SUPERSESSION": True,  # COGGOV-013: correction-triggered supersession. Enabled Day 1 rollout.
    "COGGOV_008_SESSION_INJECTION": False,    # Disabled pending 007 fix — injection without safe evolution is risk
    # INGEST-057: Cosine-gated summary replacement in evolve path
    # Phase 1 rollout: OFF (canary). Enable after histogram calibration of thresholds.
    "REPLACEMENT_GATE_ENABLED": False,
}

REPO_HYGIENE_RUNTIME_ROOT_MARKERS = tuple(
    marker.strip()
    for marker in os.environ.get(
        "PITH_REPO_HYGIENE_RUNTIME_ROOT_MARKERS",
        "/_release_worktrees/",
    ).split(",")
    if marker.strip()
)

# COVERAGE-001: Hard timeout for coverage LLM call (milliseconds)
COVERAGE_LLM_TIMEOUT_MS = 2000

# =============================================================================
# STRUCTURED ACTIVATION LAYER V0 — All tunable defaults
# =============================================================================
# Config follows existing pith pattern: env var override -> default constant.
# All values are V0 starting points — calibrate empirically.

SAL_MODE = os.environ.get("PITH_SAL_MODE", "multi_probe_then_attention")
# Options: "multi_probe" | "graph_attention" | "multi_probe_then_attention"

# Multi-probe (Mode A / Mode C first stage)
SAL_PROBE_COUNT = int(os.environ.get("PITH_SAL_PROBE_COUNT", "4"))
SAL_PROBE_OVERLAP_THRESHOLD = float(os.environ.get("PITH_SAL_PROBE_OVERLAP", "0.8"))
SAL_PROBE_EMPTY_MAX_RATIO = float(os.environ.get("PITH_SAL_PROBE_EMPTY_MAX", "0.5"))
SAL_PROBE_MIN_QUALITY = float(os.environ.get("PITH_SAL_PROBE_MIN_QUALITY", "0.3"))

# Graph attention (Mode B / Mode C second stage)
SAL_MIN_ACTIVATION_SIZE = int(os.environ.get("PITH_SAL_MIN_ACTIVATION", "5"))
SAL_MAX_ACTIVATION_SIZE = int(os.environ.get("PITH_SAL_MAX_ACTIVATION", "40"))
SAL_CONFIDENCE_FLOOR = float(os.environ.get("PITH_SAL_CONFIDENCE_FLOOR", "0.3"))
SAL_MAX_ASSOC_PER_CONCEPT = int(os.environ.get("PITH_SAL_MAX_ASSOC", "15"))
SAL_SIMILARITY_EXPONENT = float(os.environ.get("PITH_SAL_SIM_EXP", "2.0"))
SAL_ASSOCIATION_EXPONENT = float(os.environ.get("PITH_SAL_ASSOC_EXP", "0.5"))

# Attention cache
SAL_CACHE_ENABLED = os.environ.get("PITH_SAL_CACHE", "true").lower() == "true"
SAL_CACHE_BYPASS_THRESHOLD = float(os.environ.get("PITH_SAL_CACHE_BYPASS", "0.3"))
SAL_CACHE_TTL = int(os.environ.get("PITH_SAL_CACHE_TTL", "3600"))

# Temporal weighting
SAL_TEMPORAL_ENABLED = os.environ.get("PITH_SAL_TEMPORAL", "true").lower() == "true"
SAL_HALFLIFE_OBSERVATION = int(os.environ.get("PITH_SAL_HL_OBS", "24"))
SAL_HALFLIFE_PATTERN = int(os.environ.get("PITH_SAL_HL_PAT", "72"))
SAL_HALFLIFE_HEURISTIC = int(os.environ.get("PITH_SAL_HL_HEUR", "336"))
SAL_HALFLIFE_DECISION = int(os.environ.get("PITH_SAL_HL_DEC", "168"))
SAL_HALFLIFE_PRINCIPLE = int(os.environ.get("PITH_SAL_HL_PRINC", "720"))

# Over-compression defense
SAL_SURPRISE_BUFFER_ENABLED = os.environ.get("PITH_SAL_SURPRISE", "true").lower() == "true"
SAL_SURPRISE_RELEVANCE_FLOOR = float(os.environ.get("PITH_SAL_SURPRISE_REL", "0.8"))
SAL_SURPRISE_CONN_CEILING = float(os.environ.get("PITH_SAL_SURPRISE_CONN", "0.3"))
SAL_EXPLORATION_ENABLED = os.environ.get("PITH_SAL_EXPLORE", "true").lower() == "true"
SAL_EXPLORATION_RATE = float(os.environ.get("PITH_SAL_EXPLORE_RATE", "0.05"))

# Degeneracy detection
SAL_MONOPOLE_THRESHOLD = float(os.environ.get("PITH_SAL_MONOPOLE", "0.8"))
SAL_UNIFORMITY_THRESHOLD = float(os.environ.get("PITH_SAL_UNIFORM", "0.95"))


def get_feature_flag(name: str, default: bool = False) -> bool:
    """Read feature flag with env-var override (INFRA-003).

    Priority: ENV > config.py FEATURE_FLAGS > default parameter.
    Env var format: PITH_FF_{FLAG_NAME} = "true" | "false" | "1" | "0"

    This allows per-deployment flag overrides without modifying config.py,
    which ships to all consumers in pith-beta.zip.
    """
    env_key = f"PITH_FF_{name}"
    env_val = os.environ.get(env_key)
    if env_val is not None:
        return env_val.lower() in ("true", "1", "yes")
    return FEATURE_FLAGS.get(name, default)


def _env_int_clamped(name: str, *, default: int, low: int, high: int) -> int:
    """Parse an integer env knob without letting malformed values break imports."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, value))


def _env_float_clamped(name: str, *, default: float, low: float, high: float) -> float:
    """Parse a float env knob without letting malformed values break imports."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, value))


def get_feedback_db_lock_timeout_s() -> float:
    """Short optional-write DB lock budget for feedback persistence."""
    return _env_int_clamped(
        "PITH_FEEDBACK_DB_LOCK_TIMEOUT_MS",
        default=50,
        low=0,
        high=30000,
    ) / 1000.0


def get_feedback_db_slow_log_ms() -> int:
    """Slow-log threshold for optional feedback DB sections."""
    return _env_int_clamped(
        "PITH_FEEDBACK_DB_SLOW_LOG_MS",
        default=250,
        low=0,
        high=60000,
    )


def get_autolearn_maintenance_enabled() -> bool:
    """Enable deferred autolearn governance/supersession maintenance."""
    return os.environ.get("PITH_AUTOLEARN_MAINTENANCE_ENABLED", "true").lower() in ("true", "1", "yes")


def get_autolearn_maintenance_immediate_drain_enabled() -> bool:
    """Enable best-effort tiny drain after enqueue."""
    return os.environ.get("PITH_AUTOLEARN_MAINTENANCE_IMMEDIATE_DRAIN_ENABLED", "true").lower() in (
        "true",
        "1",
        "yes",
    )


def get_autolearn_maintenance_sync_drain_enabled() -> bool:
    """Enable threadpool fallback when immediate drain is kicked outside an event loop."""
    return os.environ.get("PITH_AUTOLEARN_MAINTENANCE_SYNC_DRAIN_ENABLED", "true").lower() in (
        "true",
        "1",
        "yes",
    )


def get_autolearn_maintenance_enqueue_timeout_s() -> float:
    """Short optional-write DB lock budget for maintenance enqueue."""
    return _env_int_clamped(
        "PITH_AUTOLEARN_MAINTENANCE_ENQUEUE_TIMEOUT_MS",
        default=50,
        low=0,
        high=30000,
    ) / 1000.0


def get_autolearn_subject_key_timeout_s() -> float:
    """Short optional-write DB lock budget for inline subject-key dedup."""
    return _env_int_clamped(
        "PITH_AUTOLEARN_SUBJECT_KEY_TIMEOUT_MS",
        default=50,
        low=0,
        high=30000,
    ) / 1000.0


def get_autolearn_maintenance_batch_size() -> int:
    """Scheduled autolearn maintenance batch size."""
    return _env_int_clamped("PITH_AUTOLEARN_MAINTENANCE_BATCH_SIZE", default=25, low=0, high=500)


def get_autolearn_maintenance_immediate_batch_size() -> int:
    """Immediate autolearn maintenance drain batch size."""
    return _env_int_clamped("PITH_AUTOLEARN_MAINTENANCE_IMMEDIATE_BATCH_SIZE", default=5, low=0, high=50)


def get_autolearn_maintenance_catchup_max_rows() -> int:
    """Default row budget for manual autolearn maintenance catch-up drains."""
    return _env_int_clamped("PITH_AUTOLEARN_MAINTENANCE_CATCHUP_MAX_ROWS", default=100, low=1, high=5000)


def get_autolearn_maintenance_catchup_max_seconds() -> int:
    """Default wall-clock budget for manual autolearn maintenance catch-up drains."""
    return _env_int_clamped("PITH_AUTOLEARN_MAINTENANCE_CATCHUP_MAX_SECONDS", default=30, low=1, high=600)


def get_autolearn_maintenance_max_task_share() -> float:
    """Max share of a mixed autolearn maintenance batch one task type may claim."""
    return _env_float_clamped("PITH_AUTOLEARN_MAINTENANCE_MAX_TASK_SHARE", default=0.60, low=0.10, high=1.0)


def get_autolearn_maintenance_max_attempts() -> int:
    """Max attempts per autolearn maintenance queue row."""
    return _env_int_clamped("PITH_AUTOLEARN_MAINTENANCE_MAX_ATTEMPTS", default=3, low=1, high=20)


def get_autolearn_maintenance_busy_timeout_ms() -> int:
    """SQLite busy timeout for autolearn maintenance owned connections."""
    return _env_int_clamped("PITH_AUTOLEARN_MAINTENANCE_BUSY_TIMEOUT_MS", default=50, low=0, high=30000)


def get_autolearn_maintenance_running_stale_seconds() -> int:
    """Age after which stuck running queue rows are reset."""
    return _env_int_clamped(
        "PITH_AUTOLEARN_MAINTENANCE_RUNNING_STALE_SECONDS",
        default=600,
        low=30,
        high=86400,
    )


def get_autolearn_catchup_enabled() -> bool:
    """Enable starvation-aware autolearn maintenance catch-up under pressure."""
    return os.environ.get("PITH_AUTOLEARN_CATCHUP_ENABLED", "true").lower() in ("true", "1", "yes")


def get_autolearn_maintenance_pressure_starvation_seconds() -> int:
    """Oldest queued age after which autolearn may drain one row under pressure."""
    return _env_int_clamped(
        "PITH_AUTOLEARN_MAINTENANCE_PRESSURE_STARVATION_SECONDS",
        default=600,
        low=0,
        high=86400,
    )


def get_autolearn_maintenance_supervisor_enabled() -> bool:
    """Enable product-owned autonomous autolearn maintenance catch-up."""
    return os.environ.get("PITH_AUTOLEARN_MAINTENANCE_SUPERVISOR_ENABLED", "true").lower() in (
        "true",
        "1",
        "yes",
    )


def get_autolearn_maintenance_supervisor_interval_seconds() -> int:
    """Cadence for the autonomous autolearn maintenance supervisor."""
    return _env_int_clamped(
        "PITH_AUTOLEARN_MAINTENANCE_SUPERVISOR_INTERVAL_SECONDS",
        default=30,
        low=1,
        high=3600,
    )


def get_autolearn_maintenance_supervisor_batch_size() -> int:
    """Row budget per autonomous autolearn maintenance supervisor pass."""
    return _env_int_clamped(
        "PITH_AUTOLEARN_MAINTENANCE_SUPERVISOR_BATCH_SIZE",
        default=25,
        low=1,
        high=500,
    )


def get_autolearn_maintenance_supervisor_max_wall_seconds() -> int:
    """Wall-clock budget per autonomous autolearn maintenance supervisor pass."""
    return _env_int_clamped(
        "PITH_AUTOLEARN_MAINTENANCE_SUPERVISOR_MAX_WALL_SECONDS",
        default=10,
        low=1,
        high=600,
    )


# =============================================================================
# LLM Tier Detection — OPS-080: Consumer Server Foundation
# Tier 0 = No LLM (no API key), Tier 1 = Commodity (Haiku), Tier 2 = Frontier (opt-in)
# =============================================================================

def get_llm_tier() -> int:
    """Detect available LLM tier from environment.

    Returns:
        0: No LLM available (no ANTHROPIC_API_KEY)
        1: Commodity tier (Haiku-class models for batch/maintenance)
        2: Frontier tier (Opus/Sonnet, opt-in via PITH_FF_FRONTIER_LLM)
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return 0
    frontier_opt_in = os.environ.get("PITH_FF_FRONTIER_LLM", "").lower() in ("true", "1", "yes")
    return 2 if frontier_opt_in else 1


def get_tier_feature_overrides(tier: int) -> dict[str, bool]:
    """Return feature flag overrides based on detected LLM tier.

    Tier 0: All LLM features OFF (safe offline operation).
    Tier 1: Commodity LLM features ON (Haiku-class batch processing).
    Tier 2: All LLM features ON including frontier (Opus/Sonnet).
    """
    if tier == 0:
        return {
            "TIER3_LLM_EXTRACTION_ENABLED": False,
            "LLM_EXPERIMENT_RESOLUTION_ENABLED": False,
            "LLM_CONTRADICTION_TIER2_ENABLED": False,
        }
    elif tier == 1:
        return {
            "TIER3_LLM_EXTRACTION_ENABLED": True,
            "LLM_EXPERIMENT_RESOLUTION_ENABLED": True,
            "LLM_CONTRADICTION_TIER2_ENABLED": False,  # Frontier-only
        }
    else:  # tier >= 2
        return {
            "TIER3_LLM_EXTRACTION_ENABLED": True,
            "LLM_EXPERIMENT_RESOLUTION_ENABLED": True,
            "LLM_CONTRADICTION_TIER2_ENABLED": True,
        }


# Apply tier overrides at module load time
LLM_TIER = get_llm_tier()
_tier_overrides = get_tier_feature_overrides(LLM_TIER)
FEATURE_FLAGS.update(_tier_overrides)


# =============================================================================
# Always-Activate Governance
# =============================================================================
MAX_ALWAYS_ACTIVATE = 6  # Max user-defined always-activate concept slots

# =============================================================================
# Evolution Guard Thresholds (used by learning.py and nonlossy.py)
# FIX-3(A4): Centralized in config.py to avoid circular import risk.
# =============================================================================
MIN_CONFIDENCE_CHANGE = 0.05
MIN_EVIDENCE_CHANGE = 1

# MATURITY-003: Embedding dedup thresholds
# These are calibrated for all-MiniLM-L6-v2 (384-dim, L2-normalized).
# Phase A4 calibration may adjust these values.
# Override via env vars for benchmarking (defaults preserve production behaviour).
EMBEDDING_SKIP_THRESHOLD = float(os.environ.get("PITH_EMBEDDING_SKIP_THRESHOLD", "0.85"))  # Near-duplicate: skip
EMBEDDING_EVOLVE_THRESHOLD = float(os.environ.get("PITH_EMBEDDING_EVOLVE_THRESHOLD", "0.55"))  # Related/paraphrase: evolve
CROSS_KA_EVOLVE_THRESHOLD = float(os.environ.get("PITH_CROSS_KA_EVOLVE_THRESHOLD", "0.75"))  # INGEST-007: Cross-KA merge guard

# INGEST-057: Cosine-gated replacement thresholds for evolve-path summary replacement.
# Tier 1 (>= STRONG): full "newest wins" replacement authority.
# Tier 2 (>= MODERATE, < STRONG): replace only if incoming is longer or more specific.
# Tier 3 (>= EVOLVE, < MODERATE): evidence-only evolution, summary preserved.
# These are heuristic placeholders — calibrate from evolve-time histogram before enabling.
REPLACEMENT_GATE_STRONG = float(os.environ.get("PITH_REPLACEMENT_GATE_STRONG", "0.80"))
REPLACEMENT_GATE_MODERATE = float(os.environ.get("PITH_REPLACEMENT_GATE_MODERATE", "0.70"))
REPLACEMENT_GATE_COSINE_EPSILON = float(os.environ.get("PITH_REPLACEMENT_GATE_COSINE_EPSILON", "0.005"))
MAX_EVIDENCE_PER_CONCEPT = int(os.environ.get("PITH_MAX_EVIDENCE_PER_CONCEPT", "100"))

# INGEST-009: KA synonym groups for cross-KA guard comparison.
# The guard compares knowledge_area strings. Synonym drift (hyphen vs underscore,
# slash-prefixed variants, thematic siblings) causes false cross-KA triggers.
# This map groups synonymous KAs so the guard compares groups, not raw strings.
# Unknown KAs default to themselves (no grouping = conservative, safe default).
KA_SYNONYM_GROUPS: dict[str, str] = {
    # Hyphen/underscore normalization
    "pith-engineering": "pith_engineering",
    "pith-infrastructure": "pith_infrastructure",
    # Slash-prefix normalization
    "pith/benchmarks": "benchmarks",
    # Architecture family
    "architecture_gaps": "architecture",
    "pith_codebase": "architecture",
    "pith_api": "architecture",
    "deployment_coupling": "architecture",
    "epistemology + system design": "architecture",
    "knowledge representation and uncertainty managemen": "architecture",
    # Strategy family
    "competitive_analysis": "business_strategy",
    "strategic_recommendation": "business_strategy",
    # Operations family
    "system_reliability_and_failure_modes": "operations",
    "product_operations": "operations",
    # Process family
    "pith-beta/process": "process",
    # Testing family
    "quality assurance / epistemic validation": "testing",
    "system debugging / test design": "debugging",
    # Learning family
    "classification & information retrieval": "learning",
    # CogGov family
    "pith-beta/gov": "coggov",
    # Sprints family
    "pith-beta/sprint-g": "sprints",
    # Pith engineering family
    "pith-bugs": "pith_engineering",
}


def get_ka_group(ka: str) -> str:
    """Return the canonical KA group for a knowledge_area string.

    Unknown KAs return themselves (conservative default — no false grouping).
    """
    if not ka:
        return ""
    return KA_SYNONYM_GROUPS.get(ka, ka)


def ka_groups_match(ka1: str, ka2: str) -> bool:
    """Check if two knowledge_areas belong to the same synonym group.

    Empty KA on either side returns True (safe default — don't block merges
    for concepts missing KA metadata).
    """
    if not ka1 or not ka2:
        return True
    return get_ka_group(ka1) == get_ka_group(ka2)


# Below EVOLVE_THRESHOLD: create new concept

# RETRIEVAL-021: Activation-learning bridge
ACTIVATION_EVOLVE_BIAS = 0.10  # Effective evolve threshold reduction for activated concepts

# Quarantine release settings (MATURITY-003 Part B)
QUARANTINE_RELEASE_AGE_DAYS = 7
QUARANTINE_RELEASE_CAP = 200  # Max releases per reflection cycle (raised from 50, EVIDENCE_QUARANTINE_SPEC Fix 2)

# Temporal promotion settings (MATURITY-003 Part D)
TEMPORAL_MATURITY_AGE_DAYS = 14  # MATURITY-005: Lowered from 30d — was dead gate, only 1/812 provisionals ever reached 30d
TEMPORAL_MATURITY_MIN_EVIDENCE = 1
TEMPORAL_MATURITY_MIN_ACCESS = 3
TEMPORAL_MATURITY_RECENCY_DAYS = 30  # MATURITY-004: Widened from 14d (was dead intersection with 30d age gate)
TEMPORAL_PROMOTION_CAP = 50  # Max temporal promotions per reflection cycle

# Evidence backfill settings (MATURITY-003 Phase A5)
EVIDENCE_BACKFILL_CAP = (
    300  # Max backfill operations per reflection cycle (raised from 100, EVIDENCE_QUARANTINE_SPEC Fix 4)
)
EVIDENCE_BACKFILL_MIN_ACCESS = 5  # Only backfill concepts accessed >= N times
EVIDENCE_BACKFILL_COSINE_THRESHOLD = 0.55  # Embedding cosine threshold for cross-ref

# =============================================================================
# Memory Integrity — Constraint Assembly Constants (§5.6.1)
# =============================================================================
# Moved from prediction_error.py per spec §5.6.1. Previous values preserved
# as _LEGACY_* for feature flag fallback during rollout.

# Hardened values (active when HARDENED_CONSTRAINTS_ENABLED = True)
CONSTRAINT_AUTHORITY_THRESHOLD = 0.55  # P4-PREREQ: lowered from 0.80 (only 9.3% of concepts reached 0.80)
MAX_CONSTRAINTS = 10  # Was 15→8→10; §5.6.3 audit: 80% of pos 6-15 functional

# Legacy values (active when HARDENED_CONSTRAINTS_ENABLED = False)
_LEGACY_CONSTRAINT_AUTHORITY_THRESHOLD = 0.60
_LEGACY_MAX_CONSTRAINTS = 15


# =============================================================================
# Memory Integrity — Quarantine Model Constants (§5.1.1)
# =============================================================================
# Per §5.1.1: Use maturity field on concepts table, not separate quarantine table.
# DISCARDED state added for concepts that fail quarantine review or auto-expire.
QUARANTINE_AUTO_EXPIRY_DAYS = 30  # Days before QUARANTINED → DISCARDED
QUARANTINE_PROMOTION_MIN_EVIDENCE = 2  # Min evidence items to promote QUARANTINED → PROVISIONAL
PROVISIONAL_PROMOTION_MIN_EVIDENCE = (
    1  # MATURITY-004: Lowered from 2 (84.6% have evidence=1, threshold was unreachable)
)
PROVISIONAL_PROMOTION_MIN_ACCESS = 2  # MATURITY-005: Lowered from 5 — access starvation loop: 54% of provisionals had 0 accesses, avg=1.27 vs old threshold 5
REINFORCEMENT_PROMOTION_THRESHOLD = 8  # STABILITY-001 E: Alternative PROVISIONAL→ESTABLISHED via reinforcement


# =============================================================================
# Memory Integrity — LLM Contradiction Tier 2 (SL-B2, SL-E1)
# =============================================================================
# Tier 2 uses an LLM for semantic contradiction detection when Tier 1
# (keyword + embedding) returns ambiguous scores (0.50-0.80).
MAX_TIER2_CHECKS_PER_SESSION = 2  # Max LLM contradiction calls per session
MAX_TIER2_CHECKS_PER_DAY = 10000  # Global daily cap
TIER2_FALLBACK_ON_CAP = "SOFT_REJECT"  # When cap hit, quarantine instead of LLM check
CONTRADICTION_LLM_PROVIDER = "anthropic"  # Phase 3 v1.1: single-vendor (was "openai")
CONTRADICTION_LLM_MODEL = "claude-haiku-4-5-20251001"  # Phase 3 v1.1: Haiku via Anthropic direct (NOT migrated to OpenRouter — uses ANTHROPIC_API_KEY)
CONTRADICTION_LLM_TIMEOUT_MS = 2000  # Phase 3 v1.2: 2000ms for Haiku (measured ~1.2s typical, was 500)
CONTRADICTION_LLM_CACHE_DAYS = 30  # Cache contradiction results to avoid repeats
# Tier 2 activation range: only call LLM when Tier 1 score is in this range
TIER2_AMBIGUOUS_LOW = 0.50  # Below this → PASS
TIER2_AMBIGUOUS_HIGH = 0.80  # Above this → already high-confidence contradiction

# =============================================================================
# Phase 3 — Correction Cascade (WS3)
# =============================================================================
CASCADE_MAX_DEPTH = 1  # Phase 3 v1.1: depth-1 only (configurable for future)
CASCADE_MAX_AFFECTED = 10  # Cap per cascade event
CASCADE_CORRECTION_THRESHOLD = -0.3  # Confidence drop to trigger cascade
CASCADE_SUPERSESSION_MAGNITUDE = 0.5  # Default magnitude for supersession cascades
CASCADE_MAX_DEMOTE = 0.15  # A5: Max per-concept confidence reduction from cascade
CASCADE_ALERT_THRESHOLD = 100  # NITS-001: cascade_alert fires when count exceeds this
CIRCUIT_BREAKER_ALERT_THRESHOLD = 10  # MONITOR-072: cb_alert fires when trip count exceeds this (per dashboard query window)

# CASCADE-001: Positive Reinforcement Thresholds and Limits
REINFORCEMENT_ENABLED = True  # Feature flag for positive cascade
REINFORCEMENT_EVIDENCE_THRESHOLD = 1  # Min new evidence sources to trigger cascade
REINFORCEMENT_BASE_MAGNITUDE = 0.05  # Base boost: 5% per validation
REINFORCEMENT_MAX_CONFIDENCE = 0.85  # Hard ceiling (matches extraction.py clamp)

# STABILITY-026: M3 compliance cap for PSIS-quarantined concepts
PSIS_QUARANTINE_CONFIDENCE_CAP = 0.4  # M3 ceiling: PSIS-unreviewed concepts cannot exceed 0.4
PSIS_QUARANTINE_EVIDENCE_MARKER = "quarantine:psis-unreviewed"  # Evidence tag set by PSIS ingest

REINFORCEMENT_FLOOR_RECOVERY_LIMIT = 0.15  # Max recovery above pre-demotion confidence
REINFORCEMENT_COOLDOWN_HOURS = 24  # Min hours between boosts per concept
REINFORCEMENT_MIN_ASSOC_STRENGTH = 0.15  # Processing floor — skip weaker associations
REINFORCEMENT_EXCLUDED_RELATIONS = frozenset({"contradicts", "supersedes"})  # Semantically incompatible

# Edge reclassification LLM (Tier 2)
EDGE_LLM_RECLASSIFICATION_ENABLED = LLM_TIER >= 1  # OPS-080: Gated by LLM tier (was: hardcoded True)
EDGE_LLM_MODEL = "claude-haiku-4-5-20251001"  # Anthropic direct (NOT migrated to OpenRouter — uses ANTHROPIC_API_KEY)

# PERF-001: Tier 3 LLM extraction
TIER3_LLM_MODEL = "google/gemini-2.0-flash-001"  # COST-001: switched from Anthropic direct to OpenRouter
TIER3_MAX_CONCEPTS_PER_CALL = 3  # Cap LLM output to prevent flooding
TIER3_MIN_CONVERSATION_LENGTH = 200  # Skip short exchanges (< ~50 words)
TIER3_COOLDOWN_SECONDS = 10  # Minimum interval between Tier 3 calls
TIER3_DAILY_BUDGET = 50  # Max Tier 3 calls per day
TIER3_MAX_INPUT_CHARS = 5000  # Cap per input field in prompt
TIER3_MAX_OUTPUT_TOKENS = 1024  # Haiku response token cap
EDGE_LLM_TIMEOUT_MS = 3000  # Generous timeout for batch processing (not hot path)
EDGE_LLM_CONFIDENCE_THRESHOLD = 0.7  # Min confidence to accept LLM classification
EDGE_LLM_BATCH_SIZE = 25  # DEBT-137: Right-sized from 100 (burn-down complete). 25 matches KA steady-state
EDGE_LLM_ALLOWED_RELATIONS = frozenset(
    {"supports", "contradicts", "derived_from", "part_of", "constrains", "supersedes"}
)  # Restrict to core 6 types — no exotic types

# --- Prospective Indexing (RETRIEVAL-057) ---
PROSPECTIVE_INDEXING_ENABLED = os.environ.get("PITH_PROSPECTIVE_INDEXING", "0") == "1"
PI_LLM_MODEL = os.environ.get("PITH_PI_LLM_MODEL", "google/gemini-2.0-flash-001")
PI_MAX_IMPLICATIONS = int(os.environ.get("PITH_PI_MAX_IMPLICATIONS", "5"))
PI_MAX_OUTPUT_TOKENS = int(os.environ.get("PITH_PI_MAX_OUTPUT_TOKENS", "512"))
PI_DAILY_BUDGET = int(os.environ.get("PITH_PI_DAILY_BUDGET", "200"))
PI_COOLDOWN_SECONDS = float(os.environ.get("PITH_PI_COOLDOWN_SECONDS", "2.0"))
PI_MIN_SUMMARY_LENGTH = int(os.environ.get("PITH_PI_MIN_SUMMARY_LENGTH", "50"))

# --- Event Extraction (INGEST-034) ---
EE_ENABLED = os.environ.get("PITH_EVENT_EXTRACTION", "").lower() in ("1", "true", "yes")
EE_LLM_MODEL = TIER3_LLM_MODEL  # google/gemini-2.0-flash-001 — same as Tier 3
EE_MAX_OUTPUT_TOKENS = 768  # Structured JSON output, smaller than Tier 3's 1024
EE_TIMEOUT_SECONDS = 5  # Haiku ~500ms typical, 5s generous timeout
EE_MAX_EVENTS_PER_CALL = 5  # Cap events per session_learn call
EE_MIN_CONVERSATION_LENGTH = 100  # Skip very short exchanges (< ~25 words)
EE_MAX_INPUT_CHARS = 4000  # Cap combined_text sent to LLM

# KA LLM reclassification (Tier 3) — KA-003
KA_LLM_RECLASSIFICATION_ENABLED = LLM_TIER >= 1  # OPS-080: Gated by LLM tier (was: hardcoded True)
KA_LLM_MODEL = "claude-haiku-4-5-20251001"  # Anthropic direct (NOT migrated to OpenRouter — uses ANTHROPIC_API_KEY)
KA_LLM_TIMEOUT_MS = 3000  # Generous for batch processing (not hot path)
KA_LLM_CONFIDENCE_THRESHOLD = 0.65  # Slightly lower than edge (0.7) — KA is simpler (24-way vs 10-way)
# DEBT-113: KA_LLM_BATCH_SIZE removed — unused (TaskConfig uses its own batch_size)
KA_LLM_MAX_PER_RUN = 25  # STABILITY-013: Reduced from 100 (backlog burn-down complete). 25 is sufficient for steady-state (~5-10 new concepts/session)
KA_PROVISIONAL_MAX = 200  # EUNOMIA-007/A5: Circuit breaker — disable DYNAMIC_KA_ENABLED when provisional KA count exceeds this threshold

# =============================================================================
# Phase 3 — Drift Detection (WS2)
# =============================================================================
DRIFT_CUMULATIVE_MAX = 0.70  # Total TF-IDF drift from origin
DRIFT_SINGLE_STEP_MAX = 0.50  # Any one evolution step distance
DRIFT_VELOCITY_MAX = 0.25  # Average drift per step

# =============================================================================
# Memory Integrity — Firmware Deprecation Constants (CM-M5)
# =============================================================================
# Firmware/always_activate concepts consume guaranteed budget slots every turn.
# Deprecation policy prevents stale firmware from wasting context budget.
FIRMWARE_STALE_DAYS = 90  # Days without access before firmware flagged stale
FIRMWARE_MAX_ACTIVE = 10  # Max active firmware concepts (matches PIN_BUDGET)
FIRMWARE_GRACE_PERIOD_DAYS = 7  # Days after deprecation before full removal
FIRMWARE_PROTECTED_PREFIXES = (  # Concept IDs with these prefixes cannot be deprecated
    "firmware:",  # System firmware entries
)

# =============================================================================
# 14.1 — Authority Parameters
# =============================================================================

# Authority tier thresholds
AUTHORITY_TIER_HIGH = 0.7  # Top ~20% by evidence weight (Pareto distribution)
AUTHORITY_TIER_MED_LOW = 0.3  # Bottom boundary of middle ~60%

# Authority score component weights (must sum to 1.0)
AUTHORITY_TYPE_WEIGHT = 0.35  # concept_type determines base authority
AUTHORITY_EVIDENCE_WEIGHT = 0.30  # Evidence depth (count, provenance, strength)
AUTHORITY_EVOLUTION_WEIGHT = 0.20  # Version count x correction presence
AUTHORITY_STABILITY_WEIGHT = 0.15  # Stability field (already tracked)

# Type weight map — concept_type -> base authority weight
AUTHORITY_TYPE_WEIGHTS = {
    "decision": 0.90,  # L2 — explicit choices MUST constrain behavior
    "constraint": 0.90,  # L2 — boundaries MUST be respected
    "principle": 0.80,  # L3 — reusable rules guide behavior
    "method": 0.75,  # L4 — established processes shape approach
    "process": 0.75,  # L4 — legacy compat, equivalent to method
    "heuristic": 0.70,  # L5 — rules of thumb inform judgment
    "cognitive_strategy": 0.70,  # L5 — meta-reasoning patterns
    "goal": 0.65,  # L2 — active objectives direct attention
    "pattern": 0.50,  # L1 — recurring observations inform
    "system_model": 0.50,  # L6 — models of external systems
    "hypothesis": 0.40,  # L2 — unverified, low authority until confirmed
    "observation": 0.30,  # L1 — raw data, informs but doesn't constrain
    "client_extraction": 0.30,  # Legacy compat — treat as observation
    "preference": 0.65,  # Wave 4b: Preferences moderately authoritative
}
AUTHORITY_TYPE_WEIGHT_DEFAULT = 0.30  # Fallback for unknown types

# Evolution investment scoring
AUTHORITY_EVOLUTION_VERSION_DIVISOR = 10  # score = min(1.0, (versions/10)*0.7 + correction_bonus)
AUTHORITY_EVOLUTION_CORRECTION_BONUS = 0.10  # Was 0.3; hardened per §5.6.1 (reduces correction gaming)

# Presentation mode thresholds
PRESENTATION_CONSTRAINT = 0.80  # >= 0.80: [CONSTRAINT] — must be obeyed
PRESENTATION_DIRECTIVE = 0.60  # 0.60-0.79: [DIRECTIVE] — should be followed
PRESENTATION_CONTEXT = 0.40  # 0.40-0.59: [CONTEXT] — informs reasoning
# < 0.40: BACKGROUND — available but not emphasized

# Bootstrap thresholds (§5.2) — distinct from presentation modes
# Decisions loaded at lower threshold than directive presentation because
# bootstrap captures ALL active decisions, not just those to emphasize.
BOOTSTRAP_DECISION_AUTHORITY_MIN = 0.50  # Spec §5.2: decisions at authority >= 0.50

# Recalibration target distribution (prevents authority inflation)
RECALIBRATION_TARGET_HIGH_PCT = 0.20  # Top 20% high authority
RECALIBRATION_TARGET_MED_PCT = 0.60  # Middle 60% medium
RECALIBRATION_TARGET_LOW_PCT = 0.20  # Bottom 20% low
RECALIBRATION_DEVIATION_THRESHOLD = 0.15  # Trigger normalization if deviation > 15%


# =============================================================================
# 14.2 — Currency Parameters
# =============================================================================

# Concept-type-aware half-lives (days)
# NOTE: These values are from §14.2 (Config Parameters) which differ from §2.3
# (Currency Evaluation narrative). §14.2 values are shorter/more aggressive,
# reflecting real-world tuning over theoretical defaults. §14.2 is authoritative.
CURRENCY_HALF_LIVES = {
    "observation": 14,  # Time-sensitive; stale after ~2 weeks
    "pattern": 14,  # Observations-adjacent
    "hypothesis": 21,  # Slightly more stable than observations
    "decision": 30,  # Revisited monthly
    "goal": 30,  # Goals shift on similar cadence
    "constraint": 60,  # More durable
    "heuristic": 45,  # Refined through practice
    "method": 60,  # Methods evolve as practices change
    "process": 60,  # Legacy compat
    "cognitive_strategy": 60,  # Meta-reasoning patterns
    "principle": 90,  # Durable abstractions
    "system_model": 90,  # External system models
    "client_extraction": 14,  # Legacy compat — treat as observation
    "preference": 180,  # Wave 4b: User preferences are very durable
}
CURRENCY_HALF_LIFE_DEFAULT = 30  # Fallback for unknown types

# INGEST-016: Bimodal decay for factual concepts by temporal_category.
# These override CURRENCY_HALF_LIVES when is_factual=true in the data blob.
# Identity/relational facts rarely change; role/activity facts change more often.
FACTUAL_TEMPORAL_HALF_LIVES = {
    "identity": 365,    # Name, nationality, age — very durable
    "relational": 365,  # Family, relationships — durable social facts
    "role": 120,        # Job, employer, title — changes ~annually
    "activity": 45,     # Hobbies, projects, current work — changes often
}

# Currency score component weights (must sum to 1.0)
CURRENCY_ACCESS_RECENCY_WEIGHT = 0.55  # Primary signal: stdev=0.31 (only discriminating component)
CURRENCY_TOPIC_ACTIVITY_WEIGHT = 0.15  # Reduced: saturated at 0.988 mean pre-fix; desaturation fix in currency.py
CURRENCY_EVIDENCE_FRESHNESS_WEIGHT = 0.25  # Reduced: saturated at 0.958 mean pre-fix; fallback fix in currency.py
CURRENCY_CORRECTION_HISTORY_WEIGHT = 0.05  # Future-proofing: 4 corrections today, may grow
TOPIC_ACTIVITY_NORMALIZATION_MAX = 500  # KA needs ~500 recent concepts to score 1.0 (was 50, saturated)

# Validate weights sum to 1.0 at import time
_currency_weight_sum = (
    CURRENCY_ACCESS_RECENCY_WEIGHT
    + CURRENCY_TOPIC_ACTIVITY_WEIGHT
    + CURRENCY_EVIDENCE_FRESHNESS_WEIGHT
    + CURRENCY_CORRECTION_HISTORY_WEIGHT
)
assert abs(_currency_weight_sum - 1.0) < 0.001, f"CURRENCY_*_WEIGHT values must sum to 1.0, got {_currency_weight_sum}"


# =============================================================================
# 14.3 — Budget Parameters
# =============================================================================

PIN_BUDGET = 10  # Max always-activate concepts (10 x ~50 tokens = ~500 tokens)
CONTEXT_BUDGET_MAIN = int(os.environ.get("PITH_CONTEXT_BUDGET_MAIN", 40))  # Primary context allocation (raised from 20 per RETRIEVAL-058)
CONTEXT_BUDGET_SHADOW = 3  # Shadow expansion (graph walk + association)
OVERFLOW_SUMMARY_MAX = 5  # Max concepts summarized in overflow (noise control)

# Coverage confidence threshold (Fix 1a, adversarial F4)
# Concepts with relevance_score above this are "strong matches" for coverage assessment.
# CALIBRATED Feb 26: Tested [0.30, 0.35, 0.40, 0.45]. At 0.30, noise at 0.35 masks
# sparse signal on cold topics. At 0.35: T4 (cold) correctly flags sparse_coverage,
# T1 (architecture) correctly returns null. Zero false positives on T1/T2/T3.
COVERAGE_RELEVANCE_THRESHOLD = 0.35

# Budget tier names
TIER_GUARANTEED = "guaranteed"  # Always-activate / pinned concepts
TIER_PRIORITY = "priority"  # High authority + high currency
TIER_FILL = "fill"  # Medium-scoring concepts
TIER_OVERFLOW = "overflow"  # Summarized, not full-text


# =============================================================================
# 14.3b — Latency Parameters
# =============================================================================

LATENCY_BUDGET_TOTAL_MS = 90.0  # Per-phase watchdog skip threshold (NOT the total session budget)
LATENCY_WATCHDOG_WARN_THRESHOLD = 0.80  # Warn at 80% consumed
LATENCY_WATCHDOG_SKIP_THRESHOLD_MS = 10.0  # Skip optional phases below this

# MAINT-021: Total governance wall-clock budget (distinct from LATENCY_BUDGET_TOTAL_MS above).
# Governs GovernanceContext.LATENCY_BUDGET_TOTAL_MS. Kept separate to avoid name collision
# with the per-phase watchdog constant. Default: 2000ms.
# OPT-1b: Increased from 2000.0 to 3500.0 (was: MAINT-021).
# Rationale: p95 turn latency is 1098ms, max is 2455ms. At 2000ms, normal
# variance in retrieval (p95=66ms, max=1925ms) starves downstream phases
# (graph_walk, contradiction). 3500ms gives headroom while still capping runaway turns.
# Gauntlet A7: Clamped between 500ms (floor) and 10000ms (ceiling).
_raw_gov_budget = float(os.environ.get("PITH_GOVERNANCE_BUDGET_MS", "3500.0"))
GOVERNANCE_TOTAL_LATENCY_BUDGET_MS = max(500.0, min(10000.0, _raw_gov_budget))


# =============================================================================
# 14.4 — Recalibration & Health Parameters
# =============================================================================

RECALIBRATION_INTERVAL_HOURS = 24  # Full authority/currency recompute daily
GOVERNANCE_EVENTS_RETENTION_DAYS = 30  # Rolling window for event history
HEALTH_CHECK_INTERVAL_MINUTES = 5  # Circuit breaker health check frequency
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 2  # Health checks that must fail to trip breaker
CIRCUIT_BREAKER_RECOVERY_INTERVAL_MINUTES = 1  # HEALTH-011: half-open probe interval when tripped
VACUUM_FREELIST_THRESHOLD_PAGES = 5000  # MAINT-039: run full VACUUM when freelist exceeds ~20MB


# =============================================================================
# BENCH-INFRA-002: BenchmarkIngestionMode
# =============================================================================
# Codifies all benchmark-specific ingestion bypasses into a single frozen
# dataclass. Replaces 8 scattered os.environ.get("PITH_BENCHMARK_MODE") checks.
# All defaults are production-safe (False/off). The from_env() factory reads
# env vars for backward compatibility with existing benchmark scripts.

@dataclass(frozen=True)
class BenchmarkIngestionMode:
    """Benchmark-specific ingestion bypass configuration.

    Each field maps to a specific bypass site documented by BENCHMARK-xxx tag.
    Frozen (immutable) — no accidental mutation mid-session.

    SAFETY: All defaults are production-safe (False/off). Construct via
    from_env() to read environment variables, or pass explicit values.
    """
    enabled: bool = False

    # BENCHMARK-001: Skip garbage detection.
    # GarbageDetector caps grounded concepts at max(5, ceil(words/200)).
    # Benchmark conversations are tiny (~35 words) → max_grounded=5 always,
    # silently discarding 15 of every 20 batch-ingested facts.
    skip_garbage_detection: bool = False

    # BENCHMARK-001b: Skip dedup (force all concepts to CREATE zone).
    # FactConsolidation facts share template structure ("X is famous for Y")
    # so TF-IDF similarity exceeds 0.85 skip threshold even when
    # subject/object differ. Pre-deduplicated by pith_agent conflict resolution.
    skip_dedup: bool = False

    # BENCHMARK-003: Skip retrieval-time contradiction detection.
    # Contradictions require LLM calls per pair — expensive and wasted
    # when each Pith instance lives for one question and is destroyed.
    skip_contradictions: bool = False

    # BENCHMARK-003 override: Allow contradictions even in benchmark mode.
    # Needed when auto-association is enabled to properly mark superseded
    # concepts (see Q1 RCA 2026-03-19).
    allow_contradictions: bool = False

    # BENCHMARK-004: Skip analogy detection.
    # Analogy detection is expensive and not meaningful in benchmark context.
    skip_analogies: bool = False

    # BENCHMARK-005: Skip reinforcement cascades.
    # Cascades are expensive and wasted on ephemeral benchmark instances.
    skip_cascades: bool = False

    # Skip write-time contradiction check.
    # PARITY NOTE: In current code (line 8913), this checks PITH_BENCHMARK_DEDUP_BYPASS
    # with default "false" — NOT defaulting to BENCHMARK_MODE like skip_dedup does.
    # This inconsistency is preserved for behavioral parity.
    skip_write_contradictions: bool = False

    # BENCHMARK-CAP-DEBUG: Log PITH_MAX_INSIGHTS_PER_CALL diagnostics.
    cap_debug_logging: bool = False

    # Autolearn budget (ms). Production=3500, benchmark=5000.
    autolearn_budget_ms: int = 3500

    @classmethod
    def from_env(cls) -> "BenchmarkIngestionMode":
        """Initialize from environment variables.

        Backward compatible with existing env vars:
        - PITH_BENCHMARK_MODE: master toggle
        - PITH_BENCHMARK_DEDUP_BYPASS: separate dedup control (defaults to BENCHMARK_MODE)
        - PITH_BENCHMARK_ALLOW_CONTRADICTIONS: override to re-enable contradictions
        """
        enabled = os.environ.get("PITH_BENCHMARK_MODE", "false").lower() == "true"
        if not enabled:
            return cls()  # all defaults = production-safe

        skip_dedup = os.environ.get(
            "PITH_BENCHMARK_DEDUP_BYPASS",
            os.environ.get("PITH_BENCHMARK_MODE", "false")
        ).lower() == "true"

        allow_contradictions = os.environ.get(
            "PITH_BENCHMARK_ALLOW_CONTRADICTIONS", "false"
        ).lower() == "true"

        # PARITY: write-contra bypass defaults to "false", NOT to BENCHMARK_MODE.
        # This matches original line 8913 behavior. In practice, all benchmark
        # .env files explicitly set PITH_BENCHMARK_DEDUP_BYPASS, so this
        # inconsistency with skip_dedup never manifests.
        skip_write_contradictions = os.environ.get(
            "PITH_BENCHMARK_DEDUP_BYPASS", "false"
        ).lower() == "true"
        cap_debug_logging = os.environ.get(
            "PITH_BENCHMARK_CAP_DEBUG", "true"
        ).lower() == "true"

        return cls(
            enabled=True,
            skip_garbage_detection=True,
            skip_dedup=skip_dedup,
            skip_contradictions=True,
            allow_contradictions=allow_contradictions,
            skip_analogies=True,
            skip_cascades=True,
            skip_write_contradictions=skip_write_contradictions,
            cap_debug_logging=cap_debug_logging,
            autolearn_budget_ms=int(os.environ.get("PITH_AUTOLEARN_BUDGET_MS", "5000")),
        )

    @property
    def skip_retrieval_contradictions(self) -> bool:
        """Should retrieval-time contradiction detection be skipped?
        True when contradictions should be skipped AND not explicitly allowed."""
        return self.skip_contradictions and not self.allow_contradictions


# Module-level singleton — evaluated once at import time.
BENCHMARK = BenchmarkIngestionMode.from_env()
BENCHMARK_READONLY = os.environ.get("PITH_BENCHMARK_READONLY", "false").lower() == "true"
BENCHMARK_DISABLE_EVOLVE = os.environ.get("PITH_DISABLE_EVOLVE", "false").lower() == "true"

AUTOLEARN_BUDGET_MS = int(os.environ.get("PITH_AUTOLEARN_BUDGET_MS", str(BENCHMARK.autolearn_budget_ms)))  # ARCH-D03/D06/PERF-036: Base time budget.
AUTOLEARN_PER_INSIGHT_BUDGET_MS = int(os.environ.get("PITH_AUTOLEARN_PER_INSIGHT_MS", "2000"))
AUTOLEARN_MAX_BUDGET_MS = int(os.environ.get("PITH_AUTOLEARN_MAX_BUDGET_MS", "15000"))
AUTOLEARN_WALL_BUDGET_MS = int(os.environ.get("PITH_AUTOLEARN_WALL_BUDGET_MS", str(AUTOLEARN_MAX_BUDGET_MS)))
SESSION_LEARN_SYNC_WAIT_SECONDS = float(os.environ.get("PITH_SESSION_LEARN_SYNC_WAIT_SECONDS", "8.0"))
SESSION_LEARN_PROCESSING_RETRY_AFTER_SECONDS = float(
    os.environ.get("PITH_SESSION_LEARN_PROCESSING_RETRY_AFTER_SECONDS", "2.0")
)
SESSION_LEARN_EXECUTOR_WORKERS = int(os.environ.get("PITH_SESSION_LEARN_EXECUTOR_WORKERS", "1"))
SESSION_LEARN_LIFECYCLE_JOBS_ENABLED = os.environ.get(
    "PITH_SESSION_LEARN_LIFECYCLE_JOBS_ENABLED", "false"
).lower() in {"1", "true", "yes"}

# ARCH-D05: Periodic KA promotion interval (minutes).
# Promotion runs as background task in conversation_turn, not just session_end.
KA_PROMOTION_INTERVAL_MINUTES = int(os.environ.get("PITH_KA_PROMOTION_INTERVAL_MINUTES", "30"))


# =============================================================================
# Retrieval Scoring Weights (governance-enhanced formula)
# =============================================================================

# New formula: must sum to 1.0
# RETRIEVAL-031: Rebalanced for similarity dominance.
# Previous weights gave similarity only 35%, allowing high-governance
# concepts to outrank semantically relevant ones (73% noise in retrieval audit).
# New weights: similarity 55%, governance 45% (tiebreaker role).
# RETRIEVAL-100: Recency weight — creation-time recency signal in scoring formula.
# Budget taken from similarity (was 0.56). Set to 0.0 to disable.
RETRIEVAL_WEIGHT_RECENCY = float(os.environ.get("PITH_RETRIEVAL_WEIGHT_RECENCY", "0.10"))
RETRIEVAL_RECENCY_HALF_LIFE_DAYS = float(os.environ.get("PITH_RETRIEVAL_RECENCY_HALF_LIFE_DAYS", "30"))
AUTHORITY_ARTIFACT_BOOST_ENABLED = os.environ.get(
    "PITH_AUTHORITY_ARTIFACT_BOOST_ENABLED", "true"
).lower() in ("true", "1", "yes")
AUTHORITY_ARTIFACT_BOOST_WEIGHT = float(os.environ.get(
    "PITH_AUTHORITY_ARTIFACT_BOOST_WEIGHT", "0.18"
))
assert 0.0 <= AUTHORITY_ARTIFACT_BOOST_WEIGHT <= 0.3, (
    "AUTHORITY_ARTIFACT_BOOST_WEIGHT must be [0.0, 0.3], "
    f"got {AUTHORITY_ARTIFACT_BOOST_WEIGHT}"
)

RETRIEVAL_WEIGHT_SIMILARITY = max(0.0, (
    0.56 - RETRIEVAL_WEIGHT_RECENCY  # RETRIEVAL-100: Auto-adjusts when recency weight changes
))
RETRIEVAL_WEIGHT_EMBEDDING = RETRIEVAL_WEIGHT_SIMILARITY  # Backward compat alias (deprecated)
RETRIEVAL_WEIGHT_AUTHORITY = 0.08  # RETRIEVAL-033: Down from 0.12 — freed budget to currency
RETRIEVAL_WEIGHT_CURRENCY = 0.12  # RETRIEVAL-033: Up from 0.08 — recency matters for conflict resolution (correct=highest-serial=most-recent)

# RETRIEVAL-034: Stale Recall Transparency
# When enabled, CONTRADICTED/CONTESTED concepts get [as of {freshness_label}] prefix
# and soft ranking penalties (0.70x / 0.85x). Disable to revert to pre-RETRIEVAL-034 behavior.
STALE_TRANSPARENCY_ENABLED = os.environ.get("STALE_TRANSPARENCY_ENABLED", "true").lower() == "true"
STALE_PENALTY_CONTRADICTED = float(os.environ.get("STALE_PENALTY_CONTRADICTED", "0.70"))
STALE_PENALTY_CONTESTED = float(os.environ.get("STALE_PENALTY_CONTESTED", "0.85"))
STALE_RISK_DETECTOR_ENABLED = os.environ.get("PITH_STALE_RISK_DETECTOR_ENABLED", "false").lower() == "true"
STALE_RISK_AGING_PENALTY_ENABLED = os.environ.get("PITH_STALE_RISK_AGING_PENALTY_ENABLED", "false").lower() == "true"
STALE_RISK_REVIEW_PENALTY_ENABLED = os.environ.get("PITH_STALE_RISK_REVIEW_PENALTY_ENABLED", "false").lower() == "true"
STALE_RISK_ALLOW_CONFIRMED_STALE_PROMOTION = os.environ.get(
    "PITH_STALE_RISK_ALLOW_CONFIRMED_STALE_PROMOTION", "false"
).lower() == "true"
STALE_RISK_MAX_PROMOTIONS_PER_RUN = int(os.environ.get("PITH_STALE_RISK_MAX_PROMOTIONS_PER_RUN", "100"))
STALE_RISK_TYPE_WINDOWS = {"observation": 21, "decision": 30}
STALE_RISK_THRESHOLD_AGING = float(os.environ.get("PITH_STALE_RISK_THRESHOLD_AGING", "0.65"))
STALE_RISK_THRESHOLD_REVIEW = float(os.environ.get("PITH_STALE_RISK_THRESHOLD_REVIEW", "0.85"))
STALE_RISK_PENALTY_AGING = float(os.environ.get("PITH_STALE_RISK_PENALTY_AGING", "0.92"))
STALE_RISK_PENALTY_REVIEW = float(os.environ.get("PITH_STALE_RISK_PENALTY_REVIEW", "0.85"))
STALE_RISK_CONSECUTIVE_REVIEW_HITS = int(os.environ.get("PITH_STALE_RISK_CONSECUTIVE_REVIEW_HITS", "2"))
STALE_RISK_HOT_ACCESS_DAYS = int(os.environ.get("PITH_STALE_RISK_HOT_ACCESS_DAYS", "30"))
STALE_RISK_MIN_ACCESS_COUNT = int(os.environ.get("PITH_STALE_RISK_MIN_ACCESS_COUNT", "20"))
STALE_RISK_DETECTOR_VERSION = os.environ.get("PITH_STALE_RISK_DETECTOR_VERSION", "coggov014_v3_v1")
RETRIEVAL_WEIGHT_CONFIDENCE = 0.0  # RETRIEVAL-038: Zeroed — 300pt ablation proved confidence weight harms boundary retrieval (death spiral: recalibration pulls toward E(c)=0.29, reinforcement inversely correlated)
RETRIEVAL_WEIGHT_STABILITY = 0.03  # RETRIEVAL-031: Down from 0.05
RETRIEVAL_WEIGHT_CONTEXT = 0.08  # Context activation (unchanged)
RETRIEVAL_WEIGHT_GOAL = 0.08  # RETRIEVAL-031: Up from 0.07 (rounding)
RETRIEVAL_WEIGHT_UTILITY = float(os.environ.get("PITH_RETRIEVAL_WEIGHT_UTILITY", "0.05"))  # RETRIEVAL-080: Learned from feedback

# SESSION-012: Cross-session awareness — post-scoring additive boost (not a formula weight)
RETRIEVAL_WEIGHT_SESSION_PROXIMITY = float(os.environ.get(
    "PITH_RETRIEVAL_WEIGHT_SESSION_PROXIMITY", "0.04"
))
CROSS_SESSION_WINDOW_HOURS = float(os.environ.get(
    "PITH_CROSS_SESSION_WINDOW_HOURS", "2.0"
))

# =============================================================================
# RETRIEVAL-080: Feedback Loop — Utility Accumulator Configuration
# =============================================================================
# EMA alphas — asymmetric: learn fast from USED (clear signal), slow from UNUSED (ambiguous)
UTILITY_EMA_ALPHA_USED = 0.15     # Strong positive signal, fast update
UTILITY_EMA_ALPHA_PARTIAL = 0.08  # Moderate signal
UTILITY_EMA_ALPHA_UNUSED = 0.03   # Weak negative signal, slow decay
# Classification-mapped targets (gauntlet G4 fix: raw scores too compressed for EMA)
UTILITY_TARGET_USED = 1.0
UTILITY_TARGET_PARTIAL = 0.5
UTILITY_TARGET_UNUSED = 0.0
# Safety caps
UTILITY_SCORE_MIN = 0.1    # No concept reaches 0.0 (permanent death) via feedback
UTILITY_SCORE_MAX = 0.9    # No concept reaches 1.0 (invulnerable) via feedback
UTILITY_COLD_START = 0.5   # New concepts start here
MIN_UTILITY_SAMPLES = 5    # Min samples before blending into recalibration
# Recalibration blend weights (gauntlet G3 fix: 0.3 utility made L3 cap unreachable)
RECALIBRATION_EVIDENCE_WEIGHT = 0.6   # Evidence factor in blended target
RECALIBRATION_UTILITY_WEIGHT = 0.4    # Utility factor in blended target
# Baseline date — ignore feedback before this (historic contradiction inflation)
FEEDBACK_BASELINE_DATE = "2026-03-27"
# Type-floor override: firmware/constraints/always-activate get minimum utility
UTILITY_STRUCTURAL_FLOOR = 0.7
# L3 concept types eligible for earned authority bonus
L3_CONCEPT_TYPES = frozenset({"principle", "method", "heuristic", "cognitive_strategy"})

# RETRIEVAL-031: Minimum embedding/TF-IDF similarity for retrieval candidacy.
# Concepts below this threshold are excluded BEFORE governance scoring.
# Prevents high-governance concepts from surfacing with near-zero relevance.
# Override via env var for benchmark comparison (e.g. PITH_MIN_RETRIEVAL_SIMILARITY=0.15
# restores pre-RETRIEVAL-031 floor to measure regression against benchmark baselines).
MIN_RETRIEVAL_SIMILARITY = float(os.environ.get("PITH_MIN_RETRIEVAL_SIMILARITY", "0.25"))

# =============================================================================
# Retrieval Freshness Decay (FRESHNESS_UNIFIED_REDESIGN)
# =============================================================================
RETRIEVAL_FRESHNESS_HALF_LIFE_DAYS = 7    # Days for bonus to halve. 7d balances
                                           # discrimination (CV=0.1255) vs stability.
                                           # Set to 99999 to effectively disable decay.
RETRIEVAL_FRESHNESS_MAX_BONUS = 0.08       # Maximum additive freshness bonus (at age=0).
RETRIEVAL_FRESHNESS_EVOLUTION_BONUS = 0.02 # Flat bonus for evolved concepts (version != v1).

# =============================================================================
# Health Freshness Decay (FRESHNESS_UNIFIED_REDESIGN)
# =============================================================================
HEALTH_FRESHNESS_HALF_LIFE_DAYS = 7  # Matches retrieval half-life for consistency.

# =============================================================================
# Circuit Breaker Health Indicators
# =============================================================================

# =============================================================================
# Wave 3c — Compounding Correction Loop (CCL)
# =============================================================================

# =============================================================================
# Wave 4a — Salience Weights
# =============================================================================

SALIENCE_W_ACCESS = 0.25  # Normalized access frequency
SALIENCE_W_GOAL = 0.30  # Goal alignment (weighted by priority)
SALIENCE_W_RECENCY = 0.20  # Exponential decay from last access (7-day half-life)
SALIENCE_W_DEPENDENCY = 0.15  # In-degree (how many concepts link to this one)
SALIENCE_W_THREAD = 0.10  # STUB — Wave 5 Narrative Threads


# =============================================================================
# Wave 3c — Compounding Correction Loop (CCL)
# =============================================================================

CCL_AUTHORITY_BOOST_PER_VIOLATION = 0.05  # Small enough to not distort, compounds over time
CCL_MAX_VIOLATIONS_PER_TURN = 5  # [FIX PF-2] Bound DB writes per turn
CCL_RESPONSE_CAP_CHARS = 2000  # [FIX EC-4] First 2000 chars representative
CCL_TOPIC_OVERLAP_THRESHOLD = 0  # >0 required terms in common to validate


# =============================================================================
# Circuit Breaker Health Indicators
# =============================================================================

# =============================================================================
# Wave 4b — Preference Type Registration
# =============================================================================

PREFERENCE_SALIENCE_FLOOR = 0.4  # Preferences never drop below 0.4 salience
PREFERENCE_SUPERSESSION_COOLDOWN_HOURS = 48  # Don't supersede prefs < 48h old

# =============================================================================
# Wave 4b — Provenance-Weighted Trust (PWT)
# =============================================================================

PWT_VELOCITY_CONFIDENCE_THRESHOLD = 0.7  # Flag concepts created > this with single evidence
PWT_VELOCITY_CLAMP = 0.6  # Clamp velocity anomalies to this
PWT_CORROBORATION_CONFIDENCE_THRESHOLD = 0.8  # Concepts above this need 2+ source types
PWT_UNCORROBORATED_DECAY_PER_REFLECTION = 0.05  # Decay per reflection cycle if uncorroborated

# PWT source type weights
PWT_SOURCE_WEIGHTS = {
    "cross_corroborated": 0.95,
    "observed_behavior": 0.90,
    "user_repeated": 0.85,
    "document_extracted": 0.80,
    "user_explicit": 0.70,
    "unclassified": 0.50,
    "self_generated": 0.40,
}

# Substantive source types — only these count toward corroboration [FIX G2]
PWT_SUBSTANTIVE_SOURCE_TYPES = {"user_explicit", "user_repeated", "document_extracted", "observed_behavior"}

# =============================================================================
# Wave 4b — Calibration & Prediction Tracking
# =============================================================================

CALIBRATION_MIN_PREDICTIONS_FOR_ECE = 50  # ECE not computed until >50 predictions
CALIBRATION_UNTESTED_RATIO_THRESHOLD = 0.7  # untested > 0.7 → "developing" maturity
PREDICTION_RETENTION_DAYS = 90  # Aggregate + delete raw records after 90 days
STALE_PREDICTION_TIMEOUT_DAYS = 30  # Pending predictions > 30 days → "stale"
BLIND_SPOT_MIN_CONCEPTS = 20  # Blind spot detection guard

# =============================================================================
# Wave 5 — Narrative Threads
# =============================================================================

STALENESS_TIERS = {
    "low": {"warning": 7, "auto_pause": 14, "auto_abandon": 30},
    "normal": {"warning": 14, "auto_pause": 30, "auto_abandon": 60},
    "high": {"warning": 30, "auto_pause": 60, "auto_abandon": 120},
}
THREAD_MAX_ORIENTATION_DISPLAY = 10
THREAD_LIST_LIMIT_DEFAULT = 20
THREAD_LIST_LIMIT_MAX = 100
THREAD_MEMBERSHIP_SALIENCE_DIVISOR = 3
TRACE_RETRIEVAL_MIN_SIMILARITY = 0.4
TRACE_RETRIEVAL_SCAN_WINDOW_DAYS = 90
TRACE_RETRIEVAL_SCAN_LIMIT = 500
TRACE_RETRIEVAL_MESSAGE_MIN_LENGTH = 50
TRACE_RETRIEVAL_CIRCUIT_BREAKER_SECONDS = 60
AUTO_LINK_TFIDF_THRESHOLD = 0.25
AUTO_LINK_TITLE_SIMILARITY_THRESHOLD = 0.30  # THREAD-002: Gate 1b bypass threshold

# THREAD-004 — Thread reorganization / sink containment
THREAD_REORG_GUARDRAILS_ENABLED = False
THREAD_REORG_BATCH_WRITE_ENABLED = os.getenv("THREAD_REORG_BATCH_WRITE_ENABLED", "false").lower() == "true"

THREAD_REORG_THREAD_SOFT_CAP = 300
THREAD_REORG_MAX_LINKS_PER_CONCEPT = 2
THREAD_REORG_ASSOC_FLOOR = 0.18
THREAD_REORG_SEMANTIC_FLOOR = 0.70
THREAD_REORG_KA_PURITY_FLOOR = 0.70
THREAD_REORG_MAX_BATCH_SIZE = 50
THREAD_REORG_PCB_ROLLBACK_THRESHOLD = 0.14

THREAD_REORG_EVAL_PRIMARY_SIZE = 80
THREAD_REORG_EVAL_CONTROL_SIZE = 20
THREAD_REORG_CONTROL_REGRESSION_CAUTION = 0.14

# Wave 5 / A.10 — pith_traces config
TRACES_SEARCH_LIMIT_MAX = 50
TRACES_SEARCH_LIMIT_DEFAULT = 10
TRACES_INCLUDE_DATA_DEFAULT = True

# =============================================================================
# Circuit Breaker Health Indicators
# =============================================================================

# =============================================================================
# Wave 6 — Experiment Engine
# =============================================================================

experiment_config = {
    "synthesis": {
        "similarity_floor": 0.15,
        "similarity_ceiling": 0.60,
        "max_candidates": 20,
        "max_concept_age_days": None,  # [T1 fix] null = no filter
        "max_pairwise_per_ka": 5,  # PERF-030: Stratified corpus cap (same as PERF-025 analogy)
    },
    "hypothesis": {
        "cluster_threshold": 0.25,
        "min_cluster_size": 3,
        "max_candidates": 15,
        "max_pairwise_per_ka": 20,  # PERF-030: Higher cap — clustering needs density for min_size=3
    },
    "counterfactual": {
        "max_depth": 3,
        "min_confidence": 0.4,
        "max_candidates": 15,
        "default_direction": "forward",
        "max_seeds": 200,  # PERF-030: Cap graph walk seeds by confidence
    },
    "analogy": {
        "term_overlap_penalty_threshold": 0.5,
        "term_overlap_penalty_multiplier": 0.5,
        "max_candidates": 15,
        "max_pairwise_per_ka": 5,  # PERF-025: Stratified corpus cap for O(N²) loop
        "min_embedding_sim": 0.30,  # EXP-022: Narrowed from 0.25 (eliminates edge noise)
        "max_embedding_sim": 0.50,  # EXP-022: Narrowed from 0.55 (tighter synonym exclusion)
        "type_bonus_same_abstract": 0.10,  # EXP-022: Bonus when both concepts are same abstract type
        "type_bonus_cross_abstract": 0.05,  # EXP-022: Bonus when both abstract but different types
    },
    "general": {
        "min_concepts_required": 10,  # [CS1 fix]
        "min_knowledge_areas_required": 2,  # [CS1 fix]
        "max_stored_candidates": 50,  # [S1 fix]
        "archive_days": 30,
        "cko_min_confidence": 0.7,
        "cko_min_concepts": 3,
        "cko_authority_discount": 0.8,  # [C1 fix]
    },
}

EXPERIMENT_VALID_TYPES = {
    "cross_domain_synthesis",
    "hypothesis_generation",
    "counterfactual",
    "analogy_detection",
}

# =============================================================================
# Circuit Breaker Health Indicators
# =============================================================================

# =============================================================================
# Context Management — Compaction Detection (CTX Phase 2)
# =============================================================================

COMPACTION_TEMPORAL_GAP_SECONDS = 300  # 5 min gap after active session = suspicious
COMPACTION_MIN_TURNS_FOR_DETECTION = 3  # Don't detect compaction on early turns
COMPACTION_CONTEXT_AMNESIA_MIN_TURNS = 5  # previous_response absent after this many turns
COMPACTION_AMNESIA_MIN_LENGTH = 100  # previous_response shorter than this = amnesia signal
COMPACTION_EMPTY_EXTRACTIONS_THRESHOLD = 2  # Consecutive empty extractions before signal fires
COMPACTION_COOLDOWN_SECONDS = 600  # Max 1 detection per 10 min (CTX-2 gauntlet)
COMPACTION_FALSE_POSITIVE_LIMIT = 3  # Disable detection after this many false positives per session (CTX-2)
COMPACTION_SIGNALS_REQUIRED = 2  # Two-signal rule (same pattern as CCL correction detection)

# --- SESSION-010: Cross-session compaction detection (Amendment A1) ---
COMPACTION_PROXIMITY_SECONDS = 120  # Max gap between predecessor end and current session start
COMPACTION_MIN_PREDECESSOR_EVENTS = 3  # Predecessor must have this many learning events to qualify

# --- CONTEXT-001: Working context configuration ---
WORKING_CONTEXT_MAX_TOKENS = 400  # Max tokens for working_context response field

# --- CTX-003: Context pressure monitoring ---
CTX_PRESSURE_WEIGHT_TURNS = 0.30  # Turn count weight
CTX_PRESSURE_WEIGHT_TIME = 0.20  # Elapsed time weight
CTX_PRESSURE_WEIGHT_BYTES = 0.35  # Cumulative bytes weight (strongest proxy)
CTX_PRESSURE_WEIGHT_LEARNS = 0.15  # Learning events weight
CTX_PRESSURE_TURNS_MAX = 40  # 40 turns ≈ full context for heavy tool use
CTX_PRESSURE_TIME_MAX = 90  # 90 min ≈ compaction zone
CTX_PRESSURE_BYTES_MAX = 200_000  # ~200KB previous_response ≈ heavy context
CTX_PRESSURE_LEARNS_MAX = 30  # 30 learning events ≈ substantial session
CTX_PRESSURE_THRESHOLD_SUGGEST = 0.25  # Gentle nudge — p75 of observed distribution
CTX_PRESSURE_THRESHOLD_URGE = 0.35  # Strong warning + payload — p95 of observed distribution
CTX_PRESSURE_THRESHOLD_CRITICAL = 0.45  # Emergency signal — above observed MAX (0.428)
CTX_TELEMETRY_MERGE_ENABLED = os.environ.get("CTX_TELEMETRY_MERGE_ENABLED", "1") == "1"  # CTX-TELEMETRY-001: rollout gate for structured telemetry merge

# RETRIEVAL-013: Temporal evolution check
EVOLUTION_SUPPRESSION_WEIGHT = 0.50  # Max suppression factor (A's score halved at maximum)
EVOLUTION_COSINE_MIN = 0.50  # Lower bound of evolution zone
EVOLUTION_COSINE_MAX = 0.82  # Upper bound (above = supersession territory)

# RETRIEVAL-020: Inline evolution supersession — canary constants
EVOLUTION_CANARY_MODE = False             # Phase 2B LIVE — flipped 2026-03-16 after 3-day canary (0 detections, 279 pairs evaluated, 0 errors)
EVOLUTION_CANARY_DURATION_DAYS = 7       # Documentation-only: Phase 2B transition is manual, not automated by this value
EVOLUTION_CANARY_START_DATE = "2026-03-13"  # MONITOR-041: used to compute elapsed days and surface window-passed alert
EVOLUTION_REJECT_COMPOSITE = 0.50        # Below this → skip candidate (mirrors backfill AUTO_REJECT_COMPOSITE)

# --- HEALTH-009: Auto-association in reflection ---
ASSOC_REFLECTION_MAX_PER_CYCLE = 100  # Max concepts to auto-associate per reflection

# Context Priority Hints — TTL defaults (seconds)
CTX_TTL_FIRMWARE = None  # No expiry — operational rules must persist
CTX_TTL_CONSTRAINT = None  # No expiry — behavioral boundaries
CTX_TTL_CHECKPOINT = 3600  # 1 hour — current work state
CTX_TTL_DECISION = 1800  # 30 min — session decisions
CTX_TTL_ACTIVATED = 600  # 10 min — retrieval results replaceable
CTX_TTL_ORIENTATION = 900  # 15 min — re-servable on demand
CTX_TTL_EXTRACTION_REQ = 120  # 2 min — one-shot nudge
CTX_TTL_CORRECTION = 60  # 1 min — already processed


# PRICING-002: Daily conversation turn budget per tier (concept-producing turns per day)
DAILY_TURN_BUDGET_FREE = 25
DAILY_TURN_BUDGET_PRO = 250
DAILY_TURN_BUDGET_ENTERPRISE = 999999  # effectively unlimited
DAILY_TURN_BUDGET_DEV = 999999
DAILY_TURN_BUDGET_DEFAULT = 75  # current default, backwards compat

# PRICING-006: Budget-aware quality escalation thresholds
# Confidence floors by budget zone [client_floor, heuristic_floor]
BUDGET_ZONE_THRESHOLDS = {
    "normal": {"client": 0.35, "heuristic": 0.45},  # Current defaults
    "conservation": {"client": 0.50, "heuristic": 0.55},  # Elevated
    "critical": {"client": 0.70, "heuristic": 0.75},  # High-value only
    "exhausted": {"client": 1.01, "heuristic": 1.01},  # Block all (unreachable)
}

# =============================================================================
# Federation Phase 0 — KA-Relative Governance Constants
# Source: FEDERATION_ORCHESTRATION_DESIGN v2.1, Components 0.1-0.3
# =============================================================================

# Component 0.1: KA-Relative Authority Scoring
KA_MIN_POPULATION_THRESHOLD = 30

# Component 0.2: Cross-KA Contradiction Dampening
CROSS_KA_DAMPENING = 0.5

# Component 0.3: KA-Aware Query Routing
KA_BOOST_WEIGHT = 0.2

# Startup validation (A6)
assert 10 <= KA_MIN_POPULATION_THRESHOLD <= 100, (
    f"KA_MIN_POPULATION_THRESHOLD must be 10-100, got {KA_MIN_POPULATION_THRESHOLD}"
)
assert 0.0 < CROSS_KA_DAMPENING <= 1.0, f"CROSS_KA_DAMPENING must be (0.0, 1.0], got {CROSS_KA_DAMPENING}"
assert 0.0 < KA_BOOST_WEIGHT <= 0.5, f"KA_BOOST_WEIGHT must be (0.0, 0.5], got {KA_BOOST_WEIGHT}"

# =============================================================================
# RAGAS-DIAG-001: Keyword Supplement (Phase 1.5b)
# When embedding top score < threshold, supplement with TF-IDF keyword matches.
# Default OFF — enable via PITH_KEYWORD_SUPPLEMENT=true.
# =============================================================================
KEYWORD_SUPPLEMENT_ENABLED = os.environ.get("PITH_KEYWORD_SUPPLEMENT", "false").lower() in ("true", "1", "yes")
KEYWORD_SUPPLEMENT_THRESHOLD = float(os.environ.get("PITH_KEYWORD_SUPPLEMENT_THRESHOLD", "0.30"))
KEYWORD_SUPPLEMENT_MAX = int(os.environ.get("PITH_KEYWORD_SUPPLEMENT_MAX", "5"))

# =============================================================================
# RETRIEVAL-101: Supersession chain expansion
# Walk superseded_by chains to inject current-head concepts into retrieval results.
# When a retrieved concept has superseded_by set, walk the chain to the
# current head and add the head to the result set (if not already present).
# =============================================================================
SUPERSESSION_CHAIN_ENABLED = os.environ.get(
    "PITH_SUPERSESSION_CHAIN", "1"
).lower() in ("true", "1")
SUPERSESSION_CHAIN_BUDGET_MS = int(os.environ.get(
    "PITH_SUPERSESSION_CHAIN_BUDGET_MS", "50"
))
SUPERSESSION_CHAIN_MAX_DEPTH = int(os.environ.get(
    "PITH_SUPERSESSION_CHAIN_MAX_DEPTH", "8"
))
# Max concepts to chain-expand per retrieval pass. Prevents runaway expansion
# when many retrieved concepts have chains.
SUPERSESSION_CHAIN_MAX_EXPANSIONS = int(os.environ.get(
    "PITH_SUPERSESSION_CHAIN_MAX_EXPANSIONS", "10"
))

HEALTH_AUTHORITY_ZERO_THRESHOLD = 0.50  # Trip if > 50% concepts have authority = 0
HEALTH_CURRENCY_TIMEOUT_MS = 5000  # Trip if currency scan exceeds 5s
HEALTH_CONTRADICTION_FP_RATE = 0.30  # Trip if false positive rate > 30%
HEALTH_GOVERNANCE_EVENT_OVERFLOW = 100000  # Trip if events table > 100K rows (30-day window)
HEALTH_RECALIBRATION_STALE_HOURS = 72  # Trip if last recalibration > 72h ago

# =============================================================================
# RETRIEVAL-061: KA Exclusion Filter
# Exclude specific knowledge areas from interactive retrieval.
# Comma-separated list. Bypassed in benchmark mode.
# Use case: isolate benchmark-only data from polluting interactive sessions.
# =============================================================================
RETRIEVAL_KA_EXCLUDE = [
    ka.strip()
    for ka in os.environ.get("PITH_RETRIEVAL_KA_EXCLUDE", "pith_benchmarks").split(",")
    if ka.strip()
]

# =============================================================================
# DEBT-249: Write-Durability Configuration
# Stale-processing timeout — requests stuck in "processing" longer than this
# are reclaimed. Override via PITH_WRITE_STALE_MINUTES env var.
# =============================================================================
WRITE_STALE_MINUTES = int(os.environ.get("PITH_WRITE_STALE_MINUTES", "5"))

# =============================================================================
# STABILITY-048 Stage 3B: Lifecycle job queue configuration
# Disabled by default. When enabled, conversation_turn autolearn is durably
# enqueued and drained through a bounded lifecycle runner instead of using the
# direct post-response autolearn executor.
# =============================================================================
LIFECYCLE_JOBS_ENABLED = os.environ.get("PITH_LIFECYCLE_JOBS_ENABLED", "false").lower() in {
    "1",
    "true",
    "yes",
}
LIFECYCLE_JOBS_FALLBACK_DIRECT = os.environ.get(
    "PITH_LIFECYCLE_JOBS_FALLBACK_DIRECT", "true"
).lower() in {"1", "true", "yes"}
LIFECYCLE_WORKERS = max(1, int(os.environ.get("PITH_LIFECYCLE_WORKERS", "1")))
LIFECYCLE_JOB_LEASE_SECONDS = int(os.environ.get("PITH_LIFECYCLE_JOB_LEASE_SECONDS", "300"))
LIFECYCLE_DRAIN_STUCK_SECONDS = int(
    os.environ.get("PITH_LIFECYCLE_DRAIN_STUCK_SECONDS", str(LIFECYCLE_JOB_LEASE_SECONDS))
)
LIFECYCLE_DRAIN_WALL_BUDGET_SECONDS = float(os.environ.get("PITH_LIFECYCLE_DRAIN_WALL_BUDGET_SECONDS", "30"))
LIFECYCLE_JOB_MAX_ATTEMPTS = int(os.environ.get("PITH_LIFECYCLE_JOB_MAX_ATTEMPTS", "3"))
LIFECYCLE_JOB_RETRY_SECONDS = int(os.environ.get("PITH_LIFECYCLE_JOB_RETRY_SECONDS", "60"))
LIFECYCLE_JOB_CLEANUP_DAYS = int(os.environ.get("PITH_LIFECYCLE_JOB_CLEANUP_DAYS", "7"))
LIFECYCLE_SUPERVISOR_ENABLED = os.environ.get(
    "PITH_LIFECYCLE_SUPERVISOR_ENABLED",
    str(LIFECYCLE_JOBS_ENABLED).lower(),
).lower() in {"1", "true", "yes"}
LIFECYCLE_SUPERVISOR_INTERVAL_SECONDS = float(os.environ.get("PITH_LIFECYCLE_SUPERVISOR_INTERVAL_SECONDS", "30"))
LIFECYCLE_SUPERVISOR_BATCH_SIZE = int(os.environ.get("PITH_LIFECYCLE_SUPERVISOR_BATCH_SIZE", "5"))
LIFECYCLE_SUPERVISOR_MAX_WALL_SECONDS = float(os.environ.get("PITH_LIFECYCLE_SUPERVISOR_MAX_WALL_SECONDS", "10"))
LIFECYCLE_SUPERVISOR_STARVATION_SECONDS = float(os.environ.get("PITH_LIFECYCLE_SUPERVISOR_STARVATION_SECONDS", "600"))
REFLECTION_DURABLE_JOBS_ENABLED = os.environ.get(
    "PITH_REFLECTION_DURABLE_JOBS_ENABLED",
    str(LIFECYCLE_JOBS_ENABLED).lower(),
).lower() in {"1", "true", "yes"}
REFLECTION_COMPLETION_MONITOR_ENABLED = os.environ.get(
    "PITH_REFLECTION_COMPLETION_MONITOR_ENABLED",
    str(REFLECTION_DURABLE_JOBS_ENABLED).lower(),
).lower() in {"1", "true", "yes"}
