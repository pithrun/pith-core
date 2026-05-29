"""Epistemic Classification — provenance-based trust network assignment.

Memory Integrity Spec v1.2, §5.5.1-§5.5.3:
Phase 2 launches with 3 networks (§5.5.3):
  1. world_fact — independently verifiable facts (cap: 1.0 verified, 0.55 unverified)
  2. preference — user-stated preferences and directives (cap: 0.85)
  3. assessment — everything else / AI analysis (cap: 0.55 verified, 0.40 unverified)

CRITICAL RULE (§5.5.1 Anti-Circular Classification):
  The LLM cannot classify itself into a high-trust tier.
  Classification is derived from PROVENANCE (source_type + evidence_method),
  never from LLM self-report. Only external verification promotes beyond 'assessment'.

Phase 3 adds 4 more networks if data warrants: directive, experience, procedure, hypothesis.
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Any

from app.core.datetime_utils import _ensure_aware, _utc_now

logger = logging.getLogger(__name__)


# =============================================================================
# Epistemic Authority Caps — hard ceilings per network + verification status
# =============================================================================
# These caps are applied at RETRIEVAL time (§5.1.3 CM-C6), not at storage.
# The stored authority_score is the raw score; effective_authority = min(raw, cap).

EPISTEMIC_AUTHORITY_CAPS = {
    "world_fact": {
        "verified": 1.0,  # Verified facts can reach max authority
        "unverified": 0.55,  # Unverified facts capped below DIRECTIVE
        "stale": 0.35,  # Stale facts capped at CONTEXT
        "contradicted": 0.0,  # Contradicted facts get zero authority
    },
    "preference": {
        "verified": 0.85,  # User-stated preferences get high trust
        "unverified": 0.85,  # Preferences are verified by default (user said it)
        "stale": 0.50,  # Old preferences may have changed
        "contradicted": 0.0,
    },
    "assessment": {
        "verified": 0.55,  # AI assessments NEVER reach CONSTRAINT status
        "unverified": 0.40,  # Unverified assessments are CONTEXT at best
        "stale": 0.20,
        "contradicted": 0.0,
    },
    # --- P4c: Extended Epistemic Networks (Phase 3) ---
    # Active only when EXTENDED_EPISTEMIC_NETWORKS_ENABLED=True
    "directive": {
        "verified": 0.95,  # User instructions/preferences — highest trust
        "unverified": 0.85,  # User said it, trust it even without external verification
        "stale": 0.60,
        "contradicted": 0.0,
    },
    "experience": {
        "verified": 0.85,  # Repeated observation across sessions — earned trust
        "unverified": 0.75,  # Earned through repetition
        "stale": 0.45,
        "contradicted": 0.0,
    },
    "procedure": {
        "verified": 0.80,  # Workflows/methods — functional trust
        "unverified": 0.70,  # May become outdated
        "stale": 0.40,
        "contradicted": 0.0,
    },
    "hypothesis": {
        "verified": 0.50,  # Explicitly uncertain — low trust, high value
        "unverified": 0.40,  # Same as assessment unverified
        "stale": 0.20,
        "contradicted": 0.0,
    },
}

# Phase 2 core networks (always valid)
_PHASE2_NETWORKS = {"world_fact", "preference", "assessment"}
# Phase 3 extended networks (valid only when EXTENDED_EPISTEMIC_NETWORKS_ENABLED)
_PHASE3_NETWORKS = {"directive", "experience", "procedure", "hypothesis"}
# All valid networks (used for validation)
VALID_NETWORKS = set(EPISTEMIC_AUTHORITY_CAPS.keys())

# Valid verification statuses
VALID_VERIFICATION_STATUSES = {"verified", "unverified", "stale", "contradicted"}


# =============================================================================
# Epistemic Classification — provenance-based, NOT LLM-based (§5.5.1)
# =============================================================================


def classify_epistemic_network(concept: dict) -> str:
    """Classify epistemic network from provenance, NOT from LLM self-report.

    RULE: The LLM cannot classify itself into a high-trust tier.
    Only external verification can promote beyond 'assessment'.

    Classification logic (§5.5.1):
      1. evidence_method == "llm_assertion" → always 'assessment'
      2. evidence_method == "user_stated" → 'preference'
      3. evidence_method in {"code_verified", "search_verified"} → 'world_fact'
      4. Default → 'assessment' (safest — lowest authority cap)

    Args:
        concept: Dict with at least source_type or evidence_method fields.

    Returns:
        One of: "world_fact", "preference", "assessment"
    """
    from app.core.config import FEATURE_FLAGS

    if not FEATURE_FLAGS.get("EPISTEMIC_CAPS_ENABLED", False):
        return "assessment"  # Default when feature is off

    from app.governance.evidence_method import derive_evidence_method

    # Use evidence_method if available, otherwise derive from source_type
    evidence_method = concept.get("evidence_method")
    if not evidence_method:
        source_type = concept.get("source_type", "auto_learn")
        evidence_method = derive_evidence_method(source_type)

    # Hard rule: LLM assertions are ALWAYS 'assessment'
    if evidence_method == "llm_assertion":
        return "assessment"

    # User-stated → preference
    if evidence_method == "user_stated":
        return "preference"

    # Code-verified or search-verified → world_fact
    if evidence_method in ("code_verified", "search_verified"):
        return "world_fact"

    # --- P4c: Extended classification (Phase 3) ---
    if FEATURE_FLAGS.get("EXTENDED_EPISTEMIC_NETWORKS_ENABLED", False):
        extended = _classify_extended_network(concept)
        if extended:
            return extended

    # Fallback — safest classification
    return "assessment"


def get_verification_status(concept: dict) -> str:
    """Determine verification status from concept data.

    Returns one of: "verified", "unverified", "stale", "contradicted"
    """
    from app.core.config import FEATURE_FLAGS

    if not FEATURE_FLAGS.get("EPISTEMIC_CAPS_ENABLED", False):
        return "unverified"

    # Check for explicit contradictions first
    if concept.get("has_active_contradiction", False):
        return "contradicted"
    if concept.get("verification_status"):
        status = concept["verification_status"]
        if status in VALID_VERIFICATION_STATUSES:
            return status

    # Derive from evidence_method
    from app.governance.evidence_method import derive_evidence_method

    evidence_method = concept.get("evidence_method")
    if not evidence_method:
        source_type = concept.get("source_type", "auto_learn")
        evidence_method = derive_evidence_method(source_type)

    if evidence_method in ("code_verified", "search_verified", "user_stated"):
        return "verified"

    return "unverified"


# =============================================================================
# Effective Authority Cap Computation (§5.5.2 Verification Fraction)
# =============================================================================


def compute_effective_cap(
    network: str,
    verification_status: str,
    verification_fraction: float = 0.0,
) -> float:
    """Compute effective authority cap with continuous verification scoring.

    §5.5.2: Instead of binary verified/unverified, use verification_fraction
    to interpolate between the caps for a smoother transition.

    Args:
        network: Epistemic network ("world_fact", "preference", "assessment")
        verification_status: One of "verified", "unverified", "stale", "contradicted"
        verification_fraction: 0.0 (fully unverified) to 1.0 (fully verified)

    Returns:
        Effective authority cap [0.0, 1.0]
    """
    if network not in EPISTEMIC_AUTHORITY_CAPS:
        network = "assessment"  # Safest fallback

    caps = EPISTEMIC_AUTHORITY_CAPS[network]

    # Contradicted always gets 0.0 regardless of fraction
    if verification_status == "contradicted":
        return 0.0

    # Stale uses the stale cap directly (no interpolation)
    if verification_status == "stale":
        return caps["stale"]

    # For verified/unverified, use verification_fraction to interpolate
    verified_cap = caps["verified"]
    unverified_cap = caps["unverified"]

    # Clamp fraction to [0, 1]
    fraction = max(0.0, min(1.0, verification_fraction))

    return round(unverified_cap + (verified_cap - unverified_cap) * fraction, 4)


def apply_epistemic_cap(
    raw_authority: float,
    concept: dict,
) -> float:
    """Apply epistemic authority cap to a raw authority score.

    This is the main entry point for retrieval-time authority capping.
    Called in assemble_constraint_set and retrieval scoring.

    Args:
        raw_authority: The stored authority_score from compute_authority_score()
        concept: Dict with epistemic_network, verification_status, verification_fraction

    Returns:
        Effective authority = min(raw_authority, epistemic_cap)
    """
    from app.core.config import FEATURE_FLAGS

    if not FEATURE_FLAGS.get("EPISTEMIC_CAPS_ENABLED", False):
        return raw_authority

    # Legacy concepts without epistemic classification are NOT capped.
    # Only concepts that have been classified through classify_epistemic_network()
    # get capped.  This prevents GOV-011 migration defaults (assessment/unverified)
    # from suppressing all pre-existing high-authority concepts.
    network = concept.get("epistemic_network")
    if network is None:
        return raw_authority

    verification_status = concept.get("verification_status", "unverified")
    verification_fraction = concept.get("verification_fraction", 0.0)

    cap = compute_effective_cap(network, verification_status, verification_fraction)
    effective = min(raw_authority, cap)

    if effective < raw_authority:
        logger.debug(
            "Epistemic cap applied: %s/%s (fraction=%.2f) cap=%.4f, raw=%.4f → effective=%.4f",
            network,
            verification_status,
            verification_fraction,
            cap,
            raw_authority,
            effective,
        )

    return round(effective, 4)


# =============================================================================
# Shared Utility — classify_and_annotate_concept (Retrieval Defense §3.1)
# =============================================================================


def classify_and_annotate_concept(concept) -> bool:
    """Classify epistemic network and verification status on a concept, in-place.

    This is the single entry point for storage-time epistemic classification.
    Called from learning.py:create_concept (W1) and session.py:_create_new_concept (W2).

    Works with both Concept model instances (attribute access) and dicts (key access).

    Returns True if classification was applied, False if skipped (feature off).
    """
    from app.core.config import FEATURE_FLAGS

    if not FEATURE_FLAGS.get("EPISTEMIC_CAPS_ENABLED", False):
        return False

    try:
        # Build a dict for classification functions (they expect dicts)
        if hasattr(concept, "model_dump"):
            # Pydantic model — extract relevant fields
            concept_dict = {
                "evidence_method": getattr(concept, "evidence_method", None),
                "source_type": (concept.metadata or {}).get("extraction_source", "auto_learn")
                if hasattr(concept, "metadata") and concept.metadata
                else "auto_learn",
                "has_active_contradiction": getattr(concept, "has_active_contradiction", False),
                "verification_status": getattr(concept, "verification_status", None),
                "evidence": getattr(concept, "evidence", []),
            }
            # Derive evidence_method from evidence list if not set
            if not concept_dict["evidence_method"] and concept.evidence:
                first_ev = concept.evidence[0] if concept.evidence else {}
                if isinstance(first_ev, dict):
                    concept_dict["evidence_method"] = first_ev.get("evidence_method")
                    if not concept_dict["evidence_method"]:
                        concept_dict["source_type"] = first_ev.get("source_type", concept_dict["source_type"])

            network = classify_epistemic_network(concept_dict)
            verification = get_verification_status(concept_dict)

            # Compute verification_fraction from evidence
            verified_count = 0
            total_evidence = len(concept.evidence) if concept.evidence else 0
            if total_evidence > 0:
                for ev in concept.evidence:
                    if isinstance(ev, dict):
                        em = ev.get("evidence_method", "")
                        if em in ("code_verified", "search_verified", "user_stated"):
                            verified_count += 1
            verification_fraction = verified_count / total_evidence if total_evidence > 0 else 0.0

            # Annotate the model
            concept.epistemic_network = network
            concept.verification_status = verification
            concept.verification_fraction = round(verification_fraction, 4)

        elif isinstance(concept, dict):
            # Derive evidence_method from evidence items if not on top-level
            if not concept.get("evidence_method"):
                evidence = concept.get("evidence", [])
                if evidence and isinstance(evidence[0], dict):
                    concept["evidence_method"] = evidence[0].get("evidence_method")
                    if not concept.get("evidence_method") and not concept.get("source_type"):
                        concept["source_type"] = evidence[0].get("source_type", "auto_learn")

            network = classify_epistemic_network(concept)
            verification = get_verification_status(concept)

            # Compute verification_fraction
            evidence = concept.get("evidence", [])
            verified_count = 0
            total = len(evidence) if evidence else 0
            if total > 0:
                for ev in evidence:
                    if isinstance(ev, dict):
                        em = ev.get("evidence_method", "")
                        if em in ("code_verified", "search_verified", "user_stated"):
                            verified_count += 1
            verification_fraction = verified_count / total if total > 0 else 0.0

            concept["epistemic_network"] = network
            concept["verification_status"] = verification
            concept["verification_fraction"] = round(verification_fraction, 4)
        else:
            logger.warning("classify_and_annotate_concept: unsupported concept type: %s", type(concept))
            return False

        logger.debug(
            "Epistemic classification: network=%s, verification=%s, fraction=%.2f",
            network,
            verification,
            verification_fraction,
        )
        return True

    except Exception as e:
        logger.error("classify_and_annotate_concept failed (non-fatal): %s", e)
        return False


# =============================================================================
# Hypothesis Lifecycle — promote_hypothesis (§5.5.4, A4-H2, CM-L4)
# =============================================================================

# Evidence methods that count for graduation (NOT llm_assertion)
VALID_GRADUATION_EVIDENCE = {"code_verified", "user_stated", "search_verified"}

# Maturity levels (ordered)
MATURITY_LEVELS = ["DISCARDED", "QUARANTINED", "PROVISIONAL", "ESTABLISHED"]


def promote_hypothesis(
    concept_id: str,
    hypothesis_index: int,
    new_evidence: list[dict[str, Any]],
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    """Promote hypothesis to concept with evidence quality gates (§5.5.4).

    Graduation criteria (all must be met):
    1. len(evidence_for) >= 3
    2. ALL evidence_for must have source in VALID_GRADUATION_EVIDENCE
       (LLM assertion does NOT count)
    3. len(evidence_against) == 0 (ANY credible counter-evidence blocks)
    4. Parent concept maturity >= PROVISIONAL
    5. Hypothesis age >= 24 hours (prevents same-session self-promotion)

    Args:
        concept_id: ID of the parent concept containing the hypothesis
        hypothesis_index: Index of the hypothesis in the concept's hypotheses list
        new_evidence: Additional evidence items, each with at least
                      {"source": str, "evidence_method": str}
        conn: Optional SQLite connection for loading concept data

    Returns:
        Dict with keys: promoted (bool), reason (str), concept_id (str|None)
    """
    from app.core.config import FEATURE_FLAGS

    if not FEATURE_FLAGS.get("EPISTEMIC_CAPS_ENABLED", False):
        return {
            "promoted": False,
            "reason": "EPISTEMIC_CAPS_ENABLED=False — hypothesis lifecycle is feature-gated",
            "concept_id": None,
        }

    # Load parent concept
    try:
        if conn:
            row = conn.execute(
                "SELECT id, data, maturity, created_at FROM concepts WHERE id = ?",
                (concept_id,),
            ).fetchone()
        else:
            from app.storage import _db

            with _db() as c:
                row = c.execute(
                    "SELECT id, data, maturity, created_at FROM concepts WHERE id = ?",
                    (concept_id,),
                ).fetchone()

        if not row:
            return {"promoted": False, "reason": f"Concept {concept_id} not found", "concept_id": None}
    except Exception as e:
        return {"promoted": False, "reason": f"Error loading concept: {e}", "concept_id": None}

    import json

    concept_data = json.loads(row[1]) if row[1] else {}
    maturity = row[2] or "PROVISIONAL"
    created_at = row[3]

    # Get hypotheses from concept data
    hypotheses = concept_data.get("hypotheses", [])
    if hypothesis_index < 0 or hypothesis_index >= len(hypotheses):
        return {
            "promoted": False,
            "reason": f"Hypothesis index {hypothesis_index} out of range (0-{len(hypotheses) - 1})",
            "concept_id": None,
        }

    hypothesis = hypotheses[hypothesis_index]

    # Collect all evidence (existing + new)
    existing_evidence = hypothesis.get("evidence", [])
    all_evidence = list(existing_evidence) + list(new_evidence)

    # --- Criterion 1: len(evidence_for) >= 3 ---
    evidence_for = [e for e in all_evidence if isinstance(e, dict) and not e.get("against", False)]
    if len(evidence_for) < 3:
        return {
            "promoted": False,
            "reason": f"Insufficient evidence: {len(evidence_for)}/3 required",
            "concept_id": None,
        }

    # --- Criterion 2: ALL evidence must be external ---
    for ev in evidence_for:
        method = ev.get("evidence_method", "llm_assertion")
        if method not in VALID_GRADUATION_EVIDENCE:
            return {
                "promoted": False,
                "reason": f"Evidence method '{method}' is not valid for graduation. "
                f"Required: {VALID_GRADUATION_EVIDENCE}",
                "concept_id": None,
            }

    # --- Criterion 3: No counter-evidence ---
    evidence_against = [e for e in all_evidence if isinstance(e, dict) and e.get("against", False)]
    if evidence_against:
        return {
            "promoted": False,
            "reason": f"Counter-evidence exists ({len(evidence_against)} items) — blocks promotion",
            "concept_id": None,
        }

    # --- Criterion 4: Parent maturity >= PROVISIONAL ---
    maturity_idx = MATURITY_LEVELS.index(maturity) if maturity in MATURITY_LEVELS else -1
    provisional_idx = MATURITY_LEVELS.index("PROVISIONAL")
    if maturity_idx < provisional_idx:
        return {
            "promoted": False,
            "reason": f"Parent concept maturity '{maturity}' is below PROVISIONAL",
            "concept_id": None,
        }

    # --- Criterion 5: Hypothesis age >= 24 hours ---
    hyp_created = hypothesis.get("created_at")
    if hyp_created:
        try:
            hyp_dt = _ensure_aware(datetime.fromisoformat(hyp_created.replace("Z", "+00:00")))
            age = _utc_now() - hyp_dt
            if age < timedelta(hours=24):
                hours = age.total_seconds() / 3600
                return {
                    "promoted": False,
                    "reason": f"Hypothesis too young: {hours:.1f}h < 24h minimum",
                    "concept_id": None,
                }
        except (ValueError, TypeError):
            pass  # If we can't parse the timestamp, skip this check

    # All criteria met — promote
    logger.info(
        "Hypothesis %d in concept %s promoted: %d evidence items, maturity=%s",
        hypothesis_index,
        concept_id,
        len(evidence_for),
        maturity,
    )

    return {
        "promoted": True,
        "reason": f"All 5 graduation criteria met: {len(evidence_for)} external evidence, "
        f"0 counter-evidence, maturity={maturity}",
        "concept_id": concept_id,
        "hypothesis_name": hypothesis.get("name", hypothesis.get("description", "unknown")),
        "evidence_count": len(evidence_for),
    }


# =============================================================================
# P4c: Extended Epistemic Networks — Classification + Migration
# =============================================================================
# Spec: MEMORY_INTEGRITY_PHASE4_SPEC v1.1, Section 3.3
# Feature flag: EXTENDED_EPISTEMIC_NETWORKS_ENABLED (default: False)

# Reinforcement count threshold for experience network promotion
_EXPERIENCE_REINFORCEMENT_THRESHOLD = 3


def _classify_extended_network(concept: dict) -> str | None:
    """Extended classification rules for Phase 3 networks.

    Called only when EXTENDED_EPISTEMIC_NETWORKS_ENABLED=True.
    Returns None if no extended network matches (falls through to assessment).

    Classification priority:
    1. directive — always_activate=True or source is user_preferences
    2. procedure — concept_type in {method, heuristic}
    3. hypothesis — concept_type == "hypothesis" and confidence < 0.4
    4. experience — reinforcement_count >= 3
    """
    # 1. Directive: user instructions / always-on concepts
    always_activate = concept.get("always_activate", False)
    if always_activate:
        return "directive"

    source_type = concept.get("source_type", "")
    if source_type == "user_preferences":
        return "directive"

    # Check if summary starts with [ALWAYS] — strong signal of directive
    summary = concept.get("summary", "")
    if summary.startswith("[ALWAYS]") or summary.startswith("[FIRMWARE]"):
        return "directive"

    # 2. Procedure: methods and heuristics
    concept_type = concept.get("concept_type", "observation")
    if concept_type in ("method", "heuristic", "cognitive_strategy"):
        return "procedure"

    # 3. Hypothesis: explicitly uncertain
    confidence = concept.get("confidence", 0.5)
    if concept_type == "hypothesis" and confidence < 0.4:
        return "hypothesis"

    # 4. Experience: earned through repetition
    reinforcement_count = concept.get("reinforcement_count", 0)
    if reinforcement_count >= _EXPERIENCE_REINFORCEMENT_THRESHOLD:
        return "experience"

    return None


def migrate_epistemic_networks(dry_run: bool = True) -> dict[str, Any]:
    """Migrate existing concepts to extended epistemic networks.

    Scans all concepts and reclassifies based on extended rules:
    - always_activate=True → directive
    - concept_type in (method, heuristic, cognitive_strategy) → procedure
    - concept_type == hypothesis and confidence < 0.4 → hypothesis
    - reinforcement_count >= 3 → experience

    Args:
        dry_run: If True, report what WOULD change without changing it

    Returns:
        Dict with migration statistics and proposed changes
    """
    from app.core.config import FEATURE_FLAGS
    from app.storage import _db

    if not FEATURE_FLAGS.get("EXTENDED_EPISTEMIC_NETWORKS_ENABLED", False):
        return {
            "error": "EXTENDED_EPISTEMIC_NETWORKS_ENABLED=False",
            "dry_run": dry_run,
            "changes": [],
        }

    changes = []
    stats = {
        "scanned": 0,
        "directive": 0,
        "procedure": 0,
        "hypothesis": 0,
        "experience": 0,
        "unchanged": 0,
    }

    with _db() as conn:
        cursor = conn.execute("""
            SELECT id, summary, confidence, concept_type, always_activate,
                   reinforcement_count, epistemic_network, data
            FROM concepts
            WHERE is_current = 1
        """)

        updates = []
        for row in cursor.fetchall():
            stats["scanned"] += 1
            concept_id = row[0]
            current_network = row[6] or "assessment"

            # Build concept dict for classification
            concept_dict = {
                "summary": row[1] or "",
                "confidence": row[2] or 0.5,
                "concept_type": row[3] or "observation",
                "always_activate": bool(row[4]),
                "reinforcement_count": row[5] or 0,
                "source_type": "",  # Not stored in top-level columns
            }

            new_network = _classify_extended_network(concept_dict)
            if new_network and new_network != current_network:
                changes.append(
                    {
                        "concept_id": concept_id,
                        "summary": (row[1] or "")[:100],
                        "from": current_network,
                        "to": new_network,
                        "reason": _migration_reason(concept_dict, new_network),
                    }
                )
                stats[new_network] = stats.get(new_network, 0) + 1
                updates.append((new_network, concept_id))
            else:
                stats["unchanged"] += 1

        if not dry_run and updates:
            for new_net, cid in updates:
                conn.execute(
                    "UPDATE concepts SET epistemic_network = ?, "
                    "data = json_set(data, '$.epistemic_network', ?) "
                    "WHERE id = ?",
                    (new_net, new_net, cid),
                )
            conn.commit()
            logger.info("P4c migration: %d concepts reclassified", len(updates))

    return {
        "dry_run": dry_run,
        "changes": changes[:100],  # Cap output
        "total_changes": len(changes),
        "stats": stats,
    }


def _migration_reason(concept: dict, new_network: str) -> str:
    """Generate human-readable reason for migration."""
    if new_network == "directive":
        if concept.get("always_activate"):
            return "always_activate=True → directive"
        summary = concept.get("summary", "")
        if summary.startswith("[ALWAYS]") or summary.startswith("[FIRMWARE]"):
            return "summary starts with [ALWAYS]/[FIRMWARE] → directive"
        return "source is user_preferences → directive"
    elif new_network == "procedure":
        return f"concept_type={concept.get('concept_type')} → procedure"
    elif new_network == "hypothesis":
        return f"concept_type=hypothesis, confidence={concept.get('confidence', 0):.2f} → hypothesis"
    elif new_network == "experience":
        return f"reinforcement_count={concept.get('reinforcement_count', 0)} → experience"
    return "unknown"
