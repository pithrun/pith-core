"""Shared constants used across multiple modules.

Constants that are referenced by more than one module belong here
to maintain a single source of truth.
"""

# AUTHORITY-001: Type-based authority caps.
# Types not listed here are intentionally uncapped (e.g., principle, decision, method).
# effective_authority = min(authority_score, type_cap)
TYPE_AUTHORITY_CAPS = {
    "observation": 0.65,
    "client_extraction": 0.65,
    "pattern": 0.75,
    "hypothesis": 0.60,
}

# A8: Authorship trust tiers. The ONLY values a concept's provenance may take.
# Unknown values are coerced to 'human' at the request boundary (fail-safe: never
# mint a new uncapped tier from malformed input). Keep in sync with the
# ConversationTurnRequest.provenance validator, Concept.provenance docstring, AND
# the pith_mcp.py:_VALID_PROVENANCE literal (the thin stdio bridge cannot import
# app internals, so it intentionally duplicates this 2-value set — keep both aligned).
PROVENANCE_VALUES: frozenset[str] = frozenset({"human", "agent_loop"})

# A8: Provenance-based authority caps (machine/agent authorship trust-tier).
# effective_authority = min(authority_score, type_cap, provenance_cap).
# RUNG0 Component C: ARMED for 'agent_loop' at 0.35 — below every TYPE_AUTHORITY_CAPS
# entry (weakest human tier = hypothesis 0.60), so loop-authored concepts can never
# outrank human knowledge. 'human' is intentionally absent (None → uncapped).
PROVENANCE_AUTHORITY_CAPS: dict[str, float] = {"agent_loop": 0.35}

# DEBT-051: Freshness label bucket thresholds (minutes).
# Used by _compute_freshness() in session.py.
FRESHNESS_JUST_NOW_MINS = 5
FRESHNESS_MINUTES_AGO_UPPER = 60
FRESHNESS_ONE_HOUR_UPPER = 120
FRESHNESS_HOURS_AGO_UPPER = 360       # ~6 hours
FRESHNESS_EARLIER_TODAY_UPPER = 1440   # 24 hours
FRESHNESS_YESTERDAY_UPPER = 2880      # 48 hours

# DEBT-052: Minutes-per-hour for freshness label computation.
MINUTES_PER_HOUR = 60

# DEBT-053: Governance event type strings.
# Used in INSERT INTO governance_events across cascade, supersession,
# maintenance, session, dedup_scan, firmware_deprecation, async_tasks.
# Prevents typo-class bugs and enables grep-ability.
GOV_EVENT_AUTHORITY_DEMOTION = "authority_demotion"
GOV_EVENT_AUTHORITY_REINFORCEMENT = "authority_reinforcement"
GOV_EVENT_AUTHORITY_REVIEW_FLAGGED = "authority_review_flagged"
GOV_EVENT_CONTRADICTION_REVIEW = "contradiction_review_needed"
GOV_EVENT_DECISION_SUPERSESSION = "decision_supersession"
GOV_EVENT_DEDUP_SCAN_DUPLICATE = "dedup_scan_duplicate"
GOV_EVENT_FIRMWARE_DEPRECATED = "firmware_deprecated"
GOV_EVENT_STALENESS_ALERT = "staleness_alert"
GOV_EVENT_SUPERSESSION_QUALITY_DEGRADATION = "supersession_quality_degradation"
GOV_EVENT_SUPERSESSION_REVIEW = "supersession_review_needed"

# DEBT-057: Tier 2 governance event type strings (log_event callers).
# Used in gov_ctx.log_event() across session, contradiction, correction,
# governance_context, bootstrap, budget, skills.
GOV_EVENT_BOOTSTRAP_COMPLETE = "bootstrap_complete"
GOV_EVENT_BUDGET_ALLOCATED = "budget_allocated"
GOV_EVENT_CCL_VIOLATIONS_DETECTED = "CCL_VIOLATIONS_DETECTED"
GOV_EVENT_CIRCUIT_BREAKER_TRIPPED = "CIRCUIT_BREAKER_TRIPPED"
GOV_EVENT_COMPACTION_REINJECTION = "compaction_reinjection"
GOV_EVENT_CONFIDENCE_RECALIBRATION = "CONFIDENCE_RECALIBRATION"
GOV_EVENT_CONFIDENCE_RECALIBRATION_SUMMARY = "CONFIDENCE_RECALIBRATION_SUMMARY"
GOV_EVENT_CONTRADICTION_DETECTED = "CONTRADICTION_DETECTED"
GOV_EVENT_CONTRADICTION_PHASE_2_COMPLETED = "CONTRADICTION_PHASE_2_COMPLETED"
GOV_EVENT_CONVERSATION_TURN_COMPLETE = "conversation_turn_complete"
GOV_EVENT_CORRECTION_RECORDED = "CORRECTION_RECORDED"
GOV_EVENT_EPISTEMIC_CLASSIFICATION = "EPISTEMIC_CLASSIFICATION"
GOV_EVENT_GOVERNANCE_CONTEXT_CREATED = "governance_context_created"
GOV_EVENT_GRAPH_CONTRADICTION_SIGNAL = "GRAPH_CONTRADICTION_SIGNAL"
GOV_EVENT_LATENCY_DEGRADATION = "LATENCY_DEGRADATION"
GOV_EVENT_LATENCY_WARNING = "LATENCY_WARNING"
GOV_EVENT_RESUME_CONTEXT_INJECTION = "resume_context_injection"
GOV_EVENT_SKILL_EXTRACTED = "SKILL_EXTRACTED"
GOV_EVENT_STALE_RISK_CLEARED = "stale_risk_cleared"
GOV_EVENT_STALE_RISK_RUN_COMPLETE = "stale_risk_run_complete"
GOV_EVENT_STALE_RISK_STATE_CHANGED = "stale_risk_state_changed"
GOV_EVENT_TIER2_LLM_COMPLETED = "TIER2_LLM_COMPLETED"
GOV_EVENT_TURN_DEADLINE_DEGRADED = "TURN_DEADLINE_DEGRADED"
GOV_EVENT_WRITE_CONTEXT_CREATED = "write_context_created"

# DEBT-121: Evidence strength computation constants.
# Used by reflection.py (_compute_evidence_strength, _recalibrate_confidence)
# and scripts/log_evidence_cv.py (offline CV computation).
RECENCY_LAMBDA = 0.01               # Slow recency decay: exp(-λ * age_days)
RECENCY_FLOOR = 0.50                # MAINT-032: Minimum recency factor — prevents infinite confidence decay

# EUNOMIA-038: Evidence-dependent recalibration amplifier.
# K(n) = K_BASE + K_SLOPE * ln(evidence_count)
# K_BASE controls absolute floor (GF4 coverage signal).
# K_SLOPE controls evidence-dependent spread (GF1 discrimination).
RECALIBRATION_K_BASE = 2.0           # Corrects structural multiplicative bias in E(c)
RECALIBRATION_K_SLOPE = 0.3          # Breaks corroboration ceiling for discrimination

LEGACY_EVIDENCE_STRENGTH = 0.187    # EUNOMIA-038: Matched to structured 1-ev mean E(c) (was 0.448 — inverted discrimination)

# LIFECYCLE-001: Status transition marker pairs for lifecycle detection.
# Used by write-time (session.py:_detect_contradiction).
# Each tuple: (before_markers, after_markers, reason_template)
STATUS_TRANSITIONS = [
    (
        ["plan to", "will ", "going to", "intend to", "proposed"],
        ["implemented", "deployed", "built", "completed", "shipped", "launched", "done"],
        "Plan superseded by implementation",
    ),
    (
        ["investigating", "exploring", "looking into", "researching", "analyzing"],
        ["found that", "root cause", "discovered", "turns out", "the issue was", "resolved"],
        "Investigation superseded by finding",
    ),
    (
        ["trying", "attempting", "experimenting with", "testing"],
        ["decided to", "going with", "chose", "opted for", "switched to"],
        "Experiment superseded by decision",
    ),
    (
        ["broken", "failing", "bug", "error", "doesn't work", "not working"],
        ["fixed", "resolved", "patched", "working now", "the fix"],
        "Bug report superseded by fix",
    ),
    (
        ["v1", "initial", "first version", "prototype", "draft"],
        ["v2", "rewrite", "redesign", "refactored", "upgraded", "replaced"],
        "Earlier version superseded by newer version",
    ),
]
