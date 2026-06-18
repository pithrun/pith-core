"""Learning engine for concept evolution."""

import logging

from app.cognitive.ka_admission import resolve_ka_admission
# FIX-3(A4): Import from config.py (centralized) to avoid circular import risk
from app.core.config import MIN_CONFIDENCE_CHANGE, MIN_EVIDENCE_CHANGE
from app.core.datetime_utils import _utc_now_iso
from app.core.models import Concept, ConceptEvolution, ConceptProposal, Hypothesis
from app.storage import (
    db_immediate,
    get_next_version,
    get_next_version_conn,
    load_concept,
    load_concept_conn,
    save_concept,
    save_concept_conn,
)
from app.cognitive.taxonomy import classify_knowledge_area  # KA-001/DEBT-108 (DEBT-112: removed unused imports)

logger = logging.getLogger(__name__)


def _enqueue_autolearn_maintenance(concept_id: str, concept_version: str = "v1", *, source: str) -> None:
    """Queue secondary maintenance without extending the caller's write lock."""
    try:
        from app.cognitive.autolearn_maintenance_queue import (
            enqueue_autolearn_maintenance,
            kick_autolearn_maintenance_drain,
        )

        enqueue_autolearn_maintenance(
            concept_id,
            concept_version or "v1",
            source=source,
            include_similarity=True,
        )
        kick_autolearn_maintenance_drain()
    except Exception as exc:
        logger.warning(
            "STABILITY-045: Autolearn maintenance enqueue failed for %s (non-fatal): %s",
            concept_id,
            exc,
        )


# DATA-041/DATA-053: Source-anchoring regex — extract file paths from evidence content.
# DATA-053: Broadened from 4 patterns to also capture .yml/.yaml/.json/.md/.toml/.env files.
import re as _re
_FILE_PATH_RE = _re.compile(
    r"(app/\S+\.py"                           # app/ Python files
    r"|server\.js"                             # server.js
    r"|tests?/\S+\.py"                        # test Python files
    r"|migrations?/\S+\.py"                   # migration Python files
    r"|\S+/\S+\.(?:py|yml|yaml|json|md|toml)" # nested paths with dir/ prefix
    r"|[\w][\w.\-]*\.(?:json|ya?ml|toml|md)"  # root-level config files
    r"|\.env(?:\.\w+)?)"                      # .env dotfiles
)


def validate_proposal(proposal: ConceptProposal) -> tuple[bool, str]:
    """Validate new concept proposal."""
    # Check if concept already exists
    existing = load_concept(proposal.concept_id, track_access=False)
    if existing:
        return False, f"Concept {proposal.concept_id} already exists. Use evolve instead."

    # Validate confidence
    if not (0.0 <= proposal.confidence <= 1.0):
        return False, "Confidence must be between 0.0 and 1.0"

    # Require minimum evidence
    if not proposal.evidence:
        return False, "At least one evidence source required"

    # Require summary
    if not proposal.summary or len(proposal.summary) < 10:
        return False, "Summary must be at least 10 characters"

    return True, "Valid proposal"


def create_concept(proposal: ConceptProposal) -> Concept:
    """Create new concept from proposal."""
    # DEBT-108/KA-003: Shared multi-tier classification (keyword → embedding)
    normalized_area, ka_source, ka_confidence = classify_knowledge_area(
        summary=proposal.summary or "",
        raw_area=proposal.knowledge_area or "general",
        strict=False,
    )
    normalized_area, ka_admission_metadata = resolve_ka_admission(
        summary=proposal.summary or "",
        knowledge_area=normalized_area,
        ka_source=ka_source,
        ka_confidence=ka_confidence,
        raw_area=proposal.knowledge_area or "general",
        extraction_source="propose",
        trusted_intentional_general=False,
        now=_utc_now_iso(),
    )
    # Memory Integrity §5.2.3: Evidence method anti-spoofing
    sanitized_evidence = proposal.evidence
    try:
        from app.governance.evidence_method import sanitize_evidence

        if isinstance(proposal.evidence, list):
            sanitized_evidence = sanitize_evidence(list(proposal.evidence), source_type="propose")
    except Exception:
        pass  # Non-fatal

    # DATA-041/DATA-053: Source-anchoring — use module-level _FILE_PATH_RE
    for ev in sanitized_evidence:
        if isinstance(ev, dict) and not ev.get("file_path"):
            content = ev.get("content", "")
            # DATA-058: Strip URLs before matching to prevent false positives
            # (lookbehind at pattern level fails for sub-segments after ://)
            _url_stripped = _re.sub(r'https?://\S+', '', content)
            match = _FILE_PATH_RE.search(_url_stripped)
            if match:
                ev["file_path"] = match.group(1)
    # MONITOR-059: Track file_path extraction hit rate post DATA-053 broadening
    try:
        from app.core.metrics_facade import metrics as _fp_metrics
        _ev_dicts = [ev for ev in sanitized_evidence if isinstance(ev, dict)]
        if _ev_dicts:
            _fp_hits = sum(1 for ev in _ev_dicts if ev.get("file_path"))
            _fp_metrics.record("file_path_extraction_hit_rate", _fp_hits / len(_ev_dicts))
    except Exception:
        pass

    concept = Concept(
        id=proposal.concept_id,
        version="v1",
        created_at=_utc_now_iso(),
        summary=proposal.summary,
        evidence=sanitized_evidence,
        signals=proposal.signals,
        associations=proposal.associations,
        hypotheses=proposal.hypotheses,
        confidence=proposal.confidence,
        concept_type=getattr(proposal, "concept_type", "observation") or "observation",
        stability=0.5,  # New concepts start at medium stability
        access_count=0,
        knowledge_area=normalized_area,  # KA-001: Set directly so save_concept writes it
        original_date=getattr(proposal, "original_date", None),  # TEMPORAL-003
        metadata={
            "knowledge_area": normalized_area,
            "knowledge_area_source": ka_source,
            **ka_admission_metadata,
            "created_by": "learning_engine",
            "agent_id": proposal.agent_id,
        },
    )

    # Retrieval Defense W1: Epistemic classification before storage
    try:
        from app.governance.epistemic import classify_and_annotate_concept

        classified = classify_and_annotate_concept(concept)
        if classified:
            logger.info(
                "W1: Epistemic classification applied to %s: network=%s, verification=%s",
                concept.id,
                concept.epistemic_network,
                concept.verification_status,
            )
    except Exception as e:
        logger.warning("W1: Epistemic classification failed for %s: %s", concept.id, e)

    # Retrieval Defense W6: Content policy check before storage
    try:
        from app.core.config import FEATURE_FLAGS

        if FEATURE_FLAGS.get("INGESTION_VALIDATION_ENABLED", False):
            from app.policy import check_content_policy

            if check_content_policy(concept.summary):
                logger.warning(
                    "W6: Content policy violation for %s — quarantining (authority claim detected)",
                    concept.id,
                )
                concept.maturity = "QUARANTINED"
    except Exception as e:
        logger.warning("W6: Content policy check failed for %s: %s", concept.id, e)

    save_concept(concept)

    # INGEST-037: Save verbatim fragments from proposal
    try:
        if hasattr(proposal, "verbatim_fragments") and proposal.verbatim_fragments:
            from app.storage import save_verbatim_fragment

            for frag in proposal.verbatim_fragments:
                save_verbatim_fragment(
                    concept_id=concept.id,
                    fragment_type=getattr(frag, "fragment_type", "text"),
                    content=getattr(frag, "content", None),
                    pointer_uri=getattr(frag, "pointer_uri", None),
                    pointer_meta=getattr(frag, "pointer_meta", None),
                    concept_version=concept.version,
                )
    except Exception as _vf_err:
        logger.warning("INGEST-037: Verbatim fragment save failed for %s (non-fatal): %s", concept.id, _vf_err)

    # STABILITY-045: Run governance recompute and similarity supersession from
    # the maintenance queue so create_concept does not hold a write lock across
    # expensive secondary work.
    _enqueue_autolearn_maintenance(concept.id, concept.version, source="learning_create")

    # RETRIEVAL-041: Auto-associate DECISION concepts in strategic KAs to
    # recently-active downstream domain concepts. Creates 'governs' edges so
    # S4 graph walk can reach strategic decisions from task-domain queries.
    try:
        from app.cognitive.association import auto_associate_decision_concept

        auto_associate_decision_concept(concept.id, concept)
    except Exception as _ra_err:
        logger.warning(
            "RETRIEVAL-041: decision bridge failed for %s (non-fatal): %s",
            concept.id,
            _ra_err,
        )

    return concept


def _evidence_id(item) -> str:
    """Extract comparable identifier from an evidence item.

    Handles both structured Evidence dicts (which have a guaranteed 'id' field)
    and legacy string evidence.
    """
    if isinstance(item, dict):
        return item.get("id", str(item))
    return str(item)


def _deduplicate_evidence(items: list) -> list:
    """Deduplicate evidence list preserving order. Handles both structured and legacy formats."""
    seen = set()
    result = []
    for item in items:
        eid = _evidence_id(item)
        if eid not in seen:
            seen.add(eid)
            result.append(item)
    return result


def should_evolve(old_concept: Concept, evolution: ConceptEvolution) -> tuple[bool, str]:
    """Determine if evolution is warranted."""
    reasons = []

    # Check confidence change
    if abs(evolution.confidence_change) >= MIN_CONFIDENCE_CHANGE:
        reasons.append(f"Confidence change: {evolution.confidence_change:+.2f}")

    # Check for new evidence — ID-based comparison handles both structured dicts and strings
    existing_ids = {_evidence_id(e) for e in old_concept.evidence}
    new_evidence = [e for e in evolution.new_evidence if _evidence_id(e) not in existing_ids]
    if len(new_evidence) >= MIN_EVIDENCE_CHANGE:
        reasons.append(f"New evidence: {len(new_evidence)} sources")

    # Check summary change
    if evolution.new_summary and evolution.new_summary != old_concept.summary:
        reasons.append("Summary updated")

    # Check for new hypotheses
    if evolution.new_hypotheses:
        reasons.append(f"New hypotheses: {len(evolution.new_hypotheses)}")

    # Evolution warranted if any reason exists
    if reasons:
        return True, "; ".join(reasons)

    return False, "Insufficient changes for evolution"


class VersionConflictError(Exception):
    """Raised when optimistic locking detects a concurrent modification."""

    pass


def _build_evolved_concept(old_concept: Concept, evolution: ConceptEvolution, new_version: str) -> Concept:
    """Build evolved Concept from old concept + evolution data.

    Pure computation — no I/O. Extracted for reuse by both legacy and atomic paths.
    """
    now = _utc_now_iso()
    data = old_concept.model_dump()

    # --- Fields that ALWAYS change on evolution ---
    data["version"] = new_version
    data["supersedes"] = old_concept.version
    data["updated_at"] = now
    data["change_type"] = "refinement"

    # --- Reclassification (if requested) ---
    if evolution.new_concept_type and evolution.new_concept_type != old_concept.concept_type:
        data["concept_type"] = evolution.new_concept_type
        data["change_type"] = "reclassification"

    # --- Fields that merge with new data ---
    data["summary"] = evolution.new_summary or old_concept.summary
    data["evidence"] = _deduplicate_evidence(old_concept.evidence + evolution.new_evidence)
    data["signals"] = list(set(old_concept.signals + evolution.new_signals))
    data["associations"] = list(set(old_concept.associations + evolution.new_associations))
    data["hypotheses"] = merge_hypotheses(old_concept.hypotheses, evolution.new_hypotheses)
    if evolution.new_metadata:
        existing_metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        data["metadata"] = {**existing_metadata, **evolution.new_metadata}

    # --- Fields that update incrementally ---
    data["confidence"] = max(0.0, min(1.0, old_concept.confidence + evolution.confidence_change))
    # STABILITY-026: M3 compliance — cap confidence for PSIS-quarantined concepts
    from app.core.config import PSIS_QUARANTINE_CONFIDENCE_CAP, PSIS_QUARANTINE_EVIDENCE_MARKER
    if PSIS_QUARANTINE_EVIDENCE_MARKER in data.get("evidence", []):
        data["confidence"] = min(data["confidence"], PSIS_QUARANTINE_CONFIDENCE_CAP)
    data["stability"] = min(1.0, old_concept.stability + 0.1)
    # CURRENCY-003: Do NOT refresh last_accessed on evolution.
    # Evolution means the concept was intellectually updated, not that it was
    # recently accessed/retrieved. Refreshing here inflated access_recency_score
    # (0.55 weight) causing 97% currency saturation. last_accessed should only
    # update on actual retrieval activation (pith_conversation_turn).
    # data['last_accessed'] = now  # REMOVED — see CURRENCY-003
    data["access_count"] = old_concept.access_count

    # --- Cryptographic lineage: set parent_hash for version chain ---
    data["parent_hash"] = getattr(old_concept, "content_hash", None) or ""

    return Concept(**data)


def _evolve_concept_atomic(evolution: ConceptEvolution) -> Concept | None:
    """Atomic evolve: single BEGIN IMMEDIATE transaction with optimistic locking.

    Memory Integrity Spec v1.2, §5.2.2. Prevents version chain forking by:
    1. BEGIN IMMEDIATE — acquires SQLite reserved lock at transaction start
    2. Read + verify content_hash — optimistic lock against concurrent modification
    3. Compute new version + save — all within the same transaction
    """
    with db_immediate() as conn:
        # Step 1: Read current version inside transaction
        old_concept = load_concept_conn(conn, evolution.concept_id)
        if not old_concept:
            return None

        # Step 2: Check if evolution warranted
        should_update, reason = should_evolve(old_concept, evolution)
        if not should_update:
            return None

        # Step 3: Optimistic lock — verify content_hash hasn't changed
        expected_hash = getattr(evolution, "expected_parent_hash", None)
        current_hash = getattr(old_concept, "content_hash", None) or ""
        if expected_hash and current_hash != expected_hash:
            raise VersionConflictError(
                f"Concept {evolution.concept_id} was modified concurrently. "
                f"Expected hash {expected_hash}, got {current_hash}"
            )

        # Step 4: Compute new version number inside same transaction
        new_version = get_next_version_conn(conn, evolution.concept_id)

        # Step 5: Build evolved concept
        new_concept = _build_evolved_concept(old_concept, evolution, new_version)

        # Step 6: Save inside same transaction
        save_concept_conn(conn, new_concept)
        logger.debug(
            "Atomic evolve %s: %s → %s (hash: %s)",
            evolution.concept_id,
            old_concept.version,
            new_version,
            current_hash[:8],
        )

    # Governance/similarity maintenance outside transaction via bounded queue.
    _enqueue_autolearn_maintenance(new_concept.id, new_concept.version, source="learning_evolve_atomic")
    return new_concept


def evolve_concept(evolution: ConceptEvolution) -> Concept | None:
    """Evolve existing concept.

    DEPRECATION NOTICE (IC-M1, §5.9.3): This function is deprecated in favor
    of evolve_concept_nonlossy(). When NONLOSSY_EVOLUTION_ENABLED is True,
    this wrapper delegates to the nonlossy path. Will be removed in Phase 4.

    When VERSION_CHAIN_CONCURRENCY_ENABLED, uses atomic BEGIN IMMEDIATE
    transaction with optimistic locking (§5.2.2). Otherwise, legacy path.
    """
    import warnings

    from app.core.config import FEATURE_FLAGS

    # IC-M1: Nonlossy deprecation wrapper — delegates to evolve_concept_nonlossy
    if FEATURE_FLAGS.get("NONLOSSY_EVOLUTION_ENABLED", False):
        logger.warning(
            "evolve_concept() is deprecated (IC-M1). "
            "Use evolve_concept_nonlossy() directly. Will be removed in Phase 4."
        )
        warnings.warn(
            "evolve_concept() is deprecated. Use evolve_concept_nonlossy().",
            DeprecationWarning,
            stacklevel=2,
        )
        # Bridge ConceptEvolution → nonlossy signature
        from app.cognitive.nonlossy import evolve_concept_nonlossy

        new_data = {
            "summary": evolution.new_summary if evolution.new_summary else None,
            "confidence_change": evolution.confidence_change,
            "new_evidence": [e for e in (evolution.new_evidence or [])],
            "new_signals": [s for s in (evolution.new_signals or [])],
            "new_hypotheses": [
                h.model_dump() if hasattr(h, "model_dump") else h for h in (evolution.new_hypotheses or [])
            ],
            "new_concept_type": evolution.new_concept_type
            if hasattr(evolution, "new_concept_type") and evolution.new_concept_type
            else None,
        }
        # Clean out None values
        new_data = {k: v for k, v in new_data.items() if v is not None}
        result_id = None
        try:
            with db_immediate() as conn:
                result_id = evolve_concept_nonlossy(evolution.concept_id, new_data, conn)
                if result_id:
                    conn.commit()
        except Exception:
            logger.exception("Nonlossy evolution failed, falling through to legacy")

        if result_id:
            # Reload the evolved concept to return it
            evolved = load_concept(result_id, track_access=False)
            if evolved:
                _enqueue_autolearn_maintenance(evolved.id, evolved.version, source="learning_evolve_nonlossy")
                # P1 EMBED-001: Re-index evolved concept in retrieval engine
                try:
                    from app.retrieval import retrieval_engine

                    retrieval_engine.add_concept(evolved.id)
                except Exception as _reindex_err:
                    logger.warning(f"EMBED-001: Failed to re-index {evolved.id} (non-fatal): {_reindex_err}")
                # Phase 3 v1.1 WS3-2: Trigger cascade on significant corrections
                _maybe_trigger_cascade(evolution, evolved)
                return evolved
        # If nonlossy returned None (feature off or concept not found), fall through
        # to legacy path for backward compatibility
        logger.debug("Nonlossy path returned None, using legacy evolve for %s", evolution.concept_id)

    if FEATURE_FLAGS.get("VERSION_CHAIN_CONCURRENCY_ENABLED", False):
        return _evolve_concept_atomic(evolution)

    # --- Legacy path (no concurrency protection) ---
    old_concept = load_concept(evolution.concept_id, track_access=False)
    if not old_concept:
        return None

    should_update, reason = should_evolve(old_concept, evolution)
    if not should_update:
        return None

    new_version = get_next_version(evolution.concept_id)
    new_concept = _build_evolved_concept(old_concept, evolution, new_version)

    save_concept(new_concept)
    _enqueue_autolearn_maintenance(new_concept.id, new_concept.version, source="learning_evolve_legacy")
    # P1 EMBED-001: Re-index evolved concept in retrieval engine (legacy path)
    try:
        from app.retrieval import retrieval_engine

        retrieval_engine.add_concept(new_concept.id)
    except Exception as _reindex_err:
        logger.warning(f"EMBED-001: Failed to re-index {new_concept.id} (non-fatal): {_reindex_err}")
    return new_concept


def _maybe_trigger_cascade(evolution: ConceptEvolution, evolved_concept) -> None:
    """Phase 3 v1.1 WS3-2 + CASCADE-001: Trigger correction/reinforcement cascades.

    Fires negative cascade when:
    - CORRECTION_CASCADE_ENABLED feature flag is True
    - confidence_change <= CASCADE_CORRECTION_THRESHOLD (default -0.3)

    Fires positive cascade (CASCADE-001) when:
    - REINFORCEMENT_ENABLED feature flag is True
    - confidence_change >= 0 (not a correction)
    - new_evidence has >= REINFORCEMENT_EVIDENCE_THRESHOLD items
    - session_id is present (independence requirement)
    """
    # BENCHMARK-005: Skip cascades in benchmark mode (expensive, wasted on ephemeral instances)
    from app.core.config import BENCHMARK as _bm_cascade
    if _bm_cascade.skip_cascades:
        return
    # --- Negative cascade (existing WS3-2) ---
    try:
        from app.core.config import CASCADE_CORRECTION_THRESHOLD, FEATURE_FLAGS

        if (
            FEATURE_FLAGS.get("CORRECTION_CASCADE_ENABLED", False)
            and evolution.confidence_change
            and evolution.confidence_change <= CASCADE_CORRECTION_THRESHOLD
        ):
            from app.cognitive.cascade import propagate_correction

            cascade_result = propagate_correction(
                corrected_concept_id=evolution.concept_id,
                correction_magnitude=abs(evolution.confidence_change),
                trigger="correction",
            )
            logger.info(
                "WS3-2: Correction cascade for %s: reviewed=%d, demoted=%d, flagged=%d",
                evolution.concept_id,
                cascade_result.concepts_reviewed,
                cascade_result.concepts_demoted,
                cascade_result.concepts_flagged,
            )
    except Exception as e:
        logger.warning(
            "WS3-2: Correction cascade failed for %s (non-fatal): %s",
            evolution.concept_id,
            e,
        )

    # --- Positive cascade (CASCADE-001) ---
    try:
        from app.core.config import REINFORCEMENT_ENABLED, REINFORCEMENT_EVIDENCE_THRESHOLD

        if not REINFORCEMENT_ENABLED:
            return

        # Only fire on non-negative confidence changes
        if evolution.confidence_change and evolution.confidence_change < 0:
            return

        # Need evidence to trigger reinforcement
        # A1.5: Use raw_evidence_count (Layer 1 insight sources) not len(new_evidence)
        # (Layer 2 structured entries, always 1 from dedup path). Fallback for non-dedup callers.
        new_evidence_count = evolution.raw_evidence_count or (
            len(evolution.new_evidence) if evolution.new_evidence else 0
        )
        if new_evidence_count < REINFORCEMENT_EVIDENCE_THRESHOLD:
            return

        # Need session_id for independence check (A1.2)
        if not evolution.session_id:
            logger.debug(
                "CASCADE-001: %s skipped — no session_id",
                evolution.concept_id,
            )
            return

        from app.cognitive.cascade import propagate_reinforcement

        reinf_result = propagate_reinforcement(
            reinforced_concept_id=evolution.concept_id,
            new_evidence_count=new_evidence_count,
            triggering_session_id=evolution.session_id,
        )
        if reinf_result.concepts_reinforced > 0:
            logger.info(
                "CASCADE-001: Reinforcement for %s: reinforced=%d, ceiling=%d (%.1fms)",
                evolution.concept_id,
                reinf_result.concepts_reinforced,
                reinf_result.concepts_ceiling_hit,
                reinf_result.time_ms,
            )
    except Exception as e:
        logger.warning(
            "CASCADE-001: Reinforcement cascade failed for %s (non-fatal): %s",
            evolution.concept_id,
            e,
        )


def _recompute_governance_scores(concept_id: str) -> None:
    """Recompute and cache authority + currency scores after concept mutation.

    Called after create_concept and evolve_concept to keep cached DB columns
    in sync. Non-fatal — governance scoring failure should never block learning.
    """
    try:
        from app.authority import batch_compute_authority
        from app.governance.currency import batch_compute_currency
        from app.storage import get_db_connection

        conn = get_db_connection()
        batch_compute_authority(conn, concept_ids=[concept_id])
        batch_compute_currency(conn, concept_ids=[concept_id])
    except Exception as e:
        logger.warning("Governance score recompute failed for %s (non-fatal): %s", concept_id, e)


def merge_hypotheses(old: list[Hypothesis], new: list[Hypothesis]) -> list[Hypothesis]:
    """Merge hypothesis lists."""
    # Create dict by name for easy lookup
    merged = {h.name: h for h in old}

    # Add or update with new hypotheses
    for hyp in new:
        if hyp.name in merged:
            # Update existing hypothesis
            existing = merged[hyp.name]
            existing.confidence = max(existing.confidence, hyp.confidence)
            existing.evidence = list(set(existing.evidence + hyp.evidence))
        else:
            # Add new hypothesis
            merged[hyp.name] = hyp

    return list(merged.values())


def detect_contradiction(concept: Concept, new_evidence: str, new_claim: str) -> str | None:
    """Detect if new information contradicts existing concept."""
    # Simple heuristic: check if new claim significantly differs from summary
    # In production, this could use semantic similarity

    # Check for competing hypotheses
    if len(concept.hypotheses) >= 2:
        confidences = [h.confidence for h in concept.hypotheses]
        if max(confidences) - min(confidences) < 0.2:
            return "High conflict: competing hypotheses with similar confidence"

    return None


def create_competing_hypothesis(
    concept_id: str, hypothesis_name: str, description: str, evidence: list[str], confidence: float
) -> Concept | None:
    """Add competing hypothesis to concept."""
    concept = load_concept(concept_id, track_access=False)
    if not concept:
        return None

    # Create new hypothesis
    new_hyp = Hypothesis(name=hypothesis_name, description=description, confidence=confidence, evidence=evidence)

    # Evolve concept with new hypothesis
    evolution = ConceptEvolution(
        concept_id=concept_id,
        new_hypotheses=[new_hyp],
        confidence_change=0.0,  # Competing hypotheses may reduce overall confidence
    )

    return evolve_concept(evolution)
