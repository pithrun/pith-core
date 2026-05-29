"""Knowledge area taxonomy enforcement.

Normalizes knowledge_area values to canonical taxonomy using:
1. Exact match to canonical list
2. Explicit alias map (human-reviewed, correct by construction)
3. Fuzzy match fallback (difflib, threshold 0.6)
4. Default: "general" (strict) or pass-through (permissive)

KA-001: Also provides infer_knowledge_area() for summary-based KA inference
when primary resolution produces "general".
"""

import json
import logging
import os
import re
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from difflib import get_close_matches
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Module-level cache
_canonical_areas: set = set()
_alias_map: dict = {}
_ACTIVE_KA_DESCRIPTIONS: dict[str, str] = {}
_loaded: bool = False


def _load_taxonomy():
    """Load taxonomy: sync seed data from JSON into knowledge_areas table, then
    load all ESTABLISHED+ KAs into the module-level cache.

    First call per process: reads taxonomy.json, upserts seed KAs into DB,
    loads all active KAs from DB into _canonical_areas.
    Subsequent calls: no-op (cached).

    KA-ARCH-001: DB becomes source of truth. taxonomy.json demoted to seed data.
    """
    global _canonical_areas, _alias_map, _ACTIVE_KA_DESCRIPTIONS, _loaded
    if _loaded:
        return

    # KA-SEED-001 Fix 2: Profile-aware taxonomy selection
    taxonomy_profile = os.environ.get("TAXONOMY_PROFILE", "developer")
    config_dir = Path(__file__).parent.parent / "config"
    config_path = config_dir / f"taxonomy_{taxonomy_profile}.json"

    # Fallback chain: profile-specific → taxonomy.json → empty
    if not config_path.exists():
        config_path = config_dir / "taxonomy.json"
        logger.warning(f"Taxonomy profile '{taxonomy_profile}' not found, falling back to taxonomy.json")

    seed_data = {}
    try:
        with open(config_path) as f:
            seed_data = json.load(f)
        _alias_map = seed_data.get("alias_map", {})
        _ACTIVE_KA_DESCRIPTIONS = seed_data.get("descriptions") or _CANONICAL_KA_DESCRIPTIONS
    except Exception as e:
        logger.error(f"Failed to load taxonomy seed config: {e}")
        _alias_map = {}
        _ACTIVE_KA_DESCRIPTIONS = _CANONICAL_KA_DESCRIPTIONS

    # Sync seed KAs into DB (idempotent — INSERT OR IGNORE)
    db = None
    try:
        from app.storage import get_db_connection as get_db
        db = get_db()
        seed_areas = seed_data.get("canonical_areas", [])
        for area in seed_areas:
            db.execute(
                """INSERT OR IGNORE INTO knowledge_areas (name, status, source, description)
                   VALUES (?, 'seed', 'seed', ?)""",
                (area, _ACTIVE_KA_DESCRIPTIONS.get(area, ""))
            )
        db.commit()
        logger.info(f"Taxonomy seed sync: {len(seed_areas)} seed KAs")
    except Exception as e:
        logger.warning(f"Taxonomy DB sync failed (using seed-only mode): {e}")

    # Load all active KAs (seed + established + mature) from DB
    try:
        if db is not None:
            rows = db.execute(
                "SELECT name FROM knowledge_areas WHERE status IN ('seed', 'established', 'mature')"
            ).fetchall()
            _canonical_areas = {r[0] for r in rows}
        else:
            raise RuntimeError("No DB connection")
    except Exception:
        # Fallback: use seed data directly
        _canonical_areas = set(seed_data.get("canonical_areas", []))

    _loaded = True
    logger.info(f"Taxonomy loaded: {len(_canonical_areas)} active KAs, {len(_alias_map)} aliases")

    # DEBT-109 / KA-008: Validate active profile descriptions match seed data
    desc_keys = set(_ACTIVE_KA_DESCRIPTIONS.keys())
    seed_set = set(seed_data.get("canonical_areas", []))
    in_json_not_desc = seed_set - desc_keys
    in_desc_not_json = desc_keys - seed_set
    if in_json_not_desc or in_desc_not_json:
        logger.warning(
            "Taxonomy drift: json_only=%s, desc_only=%s", in_json_not_desc or "none", in_desc_not_json or "none"
        )


def get_canonical_areas() -> frozenset:
    """DEBT-128: Public accessor for canonical knowledge areas.

    Returns a frozenset of all valid knowledge area strings from taxonomy.json.
    Scripts and batch tools should use this instead of manually loading JSON.
    """
    _load_taxonomy()
    return frozenset(_canonical_areas)


# Module-level constant — raised from 0.6 to prevent false positives (KA-ARCH-001)
# Evidence: "relationships"→"operations" at 0.6087, "emotions"→"operations" at 0.667
_FUZZY_CUTOFF = 0.75


def _ensure_provisional(name: str, source: str = "novel"):
    """Create a provisional KA if it doesn't already exist.

    Per A1: concept_count starts at 0 — actual counts are derived during
    evolve_knowledge_areas() via _recount_knowledge_areas(). This eliminates
    double-counting from multiple normalize paths (learning.py + storage.py).

    KA-ARCH-001 Fix 4.
    """
    from app.core.config import KA_PROVISIONAL_MAX, get_feature_flag
    if not get_feature_flag("DYNAMIC_KA_ENABLED", True):
        return

    # EUNOMIA-007/A5: Circuit breaker — halt provisional KA creation when count exceeds threshold
    try:
        from app.storage import get_db_connection as _get_db
        _cb_db = _get_db()
        _prov_count = _cb_db.execute(
            "SELECT COUNT(*) FROM knowledge_areas WHERE status = 'provisional'"
        ).fetchone()[0]
        if _prov_count >= KA_PROVISIONAL_MAX:
            logger.warning(
                f"KA circuit breaker tripped: {_prov_count} provisional KAs >= {KA_PROVISIONAL_MAX} threshold. "
                f"Rejecting new provisional KA '{name}'. Run KA normalization or increase KA_PROVISIONAL_MAX."
            )
            return
    except Exception as e:
        logger.debug(f"KA circuit breaker check failed (proceeding): {e}")

    # Validate KA name: lowercase alphanumeric + underscore only
    import re as _re
    if not _re.match(r'^[a-z0-9_]+$', name) or len(name) > 100:
        logger.debug(f"KA name rejected (invalid format): '{name[:30]}'")
        return

    try:
        from app.storage import get_db_connection as get_db
        db = get_db()
        db.execute(
            """INSERT OR IGNORE INTO knowledge_areas
               (name, status, source, concept_count)
               VALUES (?, 'provisional', ?, 0)""",
            (name, source)
        )
        db.commit()

        # Add to module-level cache if already promoted (race-safe)
        row = db.execute(
            "SELECT status FROM knowledge_areas WHERE name = ?", (name,)
        ).fetchone()
        if row and row[0] in ('established', 'mature'):
            _canonical_areas.add(name)
    except Exception as e:
        logger.debug(f"KA ensure_provisional failed for '{name}': {e}")


def normalize_knowledge_area(area: str, strict: bool = False) -> tuple[str, str]:
    """Normalize KA against dynamic vocabulary. Creates provisional KAs for novel areas.

    KA-ARCH-001 Fix 3: Dynamic normalization with raised fuzzy cutoff.

    Steps:
    1. Exact match against active KAs (seed + established + mature)
    2. Explicit alias match (from taxonomy.json alias_map)
    3. Fuzzy match at 0.75 threshold (raised from 0.6 to prevent false positives)
    4. No match → create provisional KA in DB (permissive) or provisional (strict)

    Returns:
        (normalized_area, source) where source is:
        - "canonical": exact match to active KAs (includes dynamic ones)
        - "alias": mapped via explicit alias
        - "fuzzy": mapped via fuzzy match fallback
        - "provisional": created new provisional KA (strict mode)
        - "novel": created new provisional KA (permissive mode)
        - "default": mapped to "general" (empty input)
    """
    _load_taxonomy()

    area_lower = area.lower().strip() if area else ""
    if not area_lower:
        return ("general", "default")

    # 0. Prefix-based normalization (EUNOMIA-003: catch episode-specific benchmark KAs)
    _PREFIX_MAP = {
        "ama_bench_ep_": "pith_benchmarks",
        "ama-bench-ep-": "pith_benchmarks",
        "benchmark_ep_": "pith_benchmarks",
    }
    for prefix, target in _PREFIX_MAP.items():
        if area_lower.startswith(prefix):
            logger.info(f"Taxonomy prefix match: '{area}' -> '{target}' (prefix: {prefix})")
            return (target, "alias")

    # 1. Exact match against all active KAs (includes dynamic ones)
    if area_lower in _canonical_areas:
        return (area_lower, "canonical")

    # 2. Explicit alias
    if area_lower in _alias_map:
        canonical = _alias_map[area_lower]
        logger.debug(f"Taxonomy alias: '{area}' -> '{canonical}'")
        return (canonical, "alias")

    # 3. Fuzzy match at raised threshold
    if _canonical_areas:
        matches = get_close_matches(area_lower, list(_canonical_areas), n=1, cutoff=_FUZZY_CUTOFF)
        if matches:
            canonical = matches[0]
            logger.info(f"Taxonomy fuzzy match: '{area}' -> '{canonical}'")
            return (canonical, "fuzzy")

    # 3.5 REFLECT-025: Embedding-based classification before creating novel KAs
    # Prevents KA re-fragmentation by mapping unknown KAs to nearest canonical.
    try:
        emb_ka, emb_score, emb_gap = classify_ka_by_embedding(area_lower)
        if emb_ka:
            logger.info(
                f"REFLECT-025: Embedding reclassification '{area}' -> '{emb_ka}' "
                f"(score={emb_score:.3f}, gap={emb_gap:.3f})"
            )
            return (emb_ka, "embedding")
    except Exception as _emb_err:
        logger.debug(f"REFLECT-025: Embedding fallback failed (non-fatal): {_emb_err}")

    # 4. No match — map to 'general' instead of creating novel KA (REFLECT-025)
    # Previously created provisional KAs, causing re-fragmentation (3rd cleanup).
    logger.warning(
        f"REFLECT-025: No canonical match for '{area}' — mapping to 'general' "
        f"(was: create novel provisional)"
    )
    return ("general", "default")


@dataclass(frozen=True)
class KnowledgeAreaBoundaryResult:
    canonical_knowledge_area: str
    source: str
    raw_knowledge_area: str | None
    label_kind: str
    facet: str | None = None
    confidence: float | None = None


_SOURCE_CONTEXT_LABELS = {"conversation"}
_DOMAIN_ALIAS_LABELS = {
    "jobs": "professional",
}
_FACET_TO_CONSUMER_DOMAIN = {
    "preferences": "personal",
    "possessions": "personal",
}
_KNOWN_FACET_LABELS = {
    "events",
    "locations",
    "possessions",
    "preferences",
    "quantitative",
    "temporal",
}
_KNOWN_CONCEPT_TYPE_LABELS = {
    "cognitive_strategy",
    "constraint",
    "decision",
    "heuristic",
    "method",
    "observation",
    "pattern",
    "preference",
    "principle",
}


def _probe_canonical_or_alias_no_warning(raw: str) -> tuple[str | None, str | None]:
    """Return exact/alias KA normalization without warning-emitting fallback."""
    _load_taxonomy()
    if raw in _canonical_areas:
        return raw, "canonical"
    if raw in _alias_map:
        return _alias_map[raw], "alias"
    return None, None


def normalize_knowledge_area_boundary(
    raw_area: str | None,
    *,
    summary: str = "",
    concept_type: str | None = None,
    strict: bool = False,
) -> KnowledgeAreaBoundaryResult:
    """Classify mixed KA/facet/source/type labels before generic KA fallback.

    KA-008: LME client extraction can send facets or concept types in the
    knowledge_area field. Known non-domain labels must not reach the generic
    REFLECT-025 fallback before we preserve and route their semantics.
    """
    raw = str(raw_area).strip().lower() if raw_area is not None else ""
    if not raw:
        return KnowledgeAreaBoundaryResult("general", "default", None, "unknown")

    canonical, source = _probe_canonical_or_alias_no_warning(raw)
    if canonical and canonical != "general":
        return KnowledgeAreaBoundaryResult(canonical, source or "canonical", raw, "canonical_domain")

    if raw in _DOMAIN_ALIAS_LABELS:
        return KnowledgeAreaBoundaryResult(_DOMAIN_ALIAS_LABELS[raw], "domain_alias", raw, "domain_alias")

    if raw in _KNOWN_CONCEPT_TYPE_LABELS:
        inferred = infer_knowledge_area(summary) if summary else None
        return KnowledgeAreaBoundaryResult(
            inferred or "general",
            "concept_type_inferred" if inferred else "concept_type_default",
            raw,
            "concept_type",
        )

    if raw in _SOURCE_CONTEXT_LABELS:
        inferred = infer_knowledge_area(summary) if summary else None
        return KnowledgeAreaBoundaryResult(
            inferred or "general",
            "source_context_inferred" if inferred else "source_context_default",
            raw,
            "source_context",
        )

    if raw in _KNOWN_FACET_LABELS:
        direct = _FACET_TO_CONSUMER_DOMAIN.get(raw)
        if direct:
            return KnowledgeAreaBoundaryResult(direct, "facet_direct", raw, "facet", raw)
        inferred = infer_knowledge_area(summary) if summary else None
        return KnowledgeAreaBoundaryResult(
            inferred or "general",
            "facet_inferred" if inferred else "facet_default",
            raw,
            "facet",
            raw,
        )

    normalized, source = normalize_knowledge_area(raw, strict=strict)
    return KnowledgeAreaBoundaryResult(normalized, source, raw, "unknown")


def infer_knowledge_area(summary: str) -> str | None:
    """Infer knowledge_area from concept summary text using taxonomy keywords.

    Returns canonical area if confident match found, else None.
    Used as fallback when primary KA resolution produces "general".
    KA-001: Write-time inference. KA-002: Batch reclassification.
    """
    _load_taxonomy()
    if not summary or len(summary) < 20:
        return None

    summary_lower = summary.lower()
    words = set(summary_lower.split())

    # Score each canonical area
    scores: defaultdict[str, int] = defaultdict(int)

    # Direct canonical area name matches
    # Gauntlet F1: Use word-boundary regex to avoid "process" matching "processing"
    # E2E fix: Single-word canonical names (testing, security, process) are common
    # English words — score 1pt to avoid false positives from incidental mentions.
    # Multi-word names (product_strategy, competitive_analysis) score 2pt.
    for area in _canonical_areas:
        if area == "general":
            continue  # Never infer "general"
        # Handle multi-word area names (e.g., "product_strategy" → "product strategy")
        area_pattern = area.replace("_", r"[\s_]")
        if re.search(rf"\b{area_pattern}\b", summary_lower):
            points = 2 if "_" in area else 1
            scores[area] += points

    # Alias keyword matches (1 point each, mapped to canonical)
    for alias, canonical in _alias_map.items():
        if canonical == "general":
            continue
        alias_words = alias.replace("_", " ").split()
        # Gauntlet F2: For multi-word aliases, check if ALL constituent words appear
        if len(alias_words) > 1:
            if all(w in words for w in alias_words):
                scores[canonical] += 1
        else:
            # Single-word alias: word-boundary match
            if re.search(rf"\b{re.escape(alias_words[0])}\b", summary_lower):
                scores[canonical] += 1

    if not scores:
        return None

    # Require minimum score of 2 to avoid false positives
    # Gauntlet F3: KA-002 dry run yielded 6.7% at threshold=2; kept threshold=2 (accepted low yield for precision)
    best_area = max(scores, key=scores.get)
    if scores[best_area] >= 2:
        return best_area

    return None


# --- Embedding-based KA classification (KA-001 enhancement) ---

# Lazy-loaded canonical embeddings
_canonical_ka_embeddings: np.ndarray | None = None
_canonical_ka_keys: list | None = None

# Richer descriptions for embedding comparison (24 canonical areas)
_CANONICAL_KA_DESCRIPTIONS = {
    "architecture": "system architecture design patterns component structure data flow technical decisions cognitive architecture",
    "implementation": "code implementation engineering building features retrieval systems tooling pipelines ingestion coding programming",
    "process": "engineering process workflow methodology development practices sprint planning retrospective project management workflow coordination",
    "testing": "tests verification quality assurance test coverage unit tests integration tests",
    "debugging": "bug investigation error diagnosis root cause analysis troubleshooting failure analysis fix repair",
    "operations": "deployment infrastructure monitoring reliability devops system operations maintenance",
    "security": "security safety data protection adversarial defense cognitive safety governance",
    "performance": "performance optimization latency throughput scalability benchmarking speed",
    "product_strategy": "product direction feature prioritization user value product design onboarding roadmap",
    "business_strategy": "business model market position revenue growth competitive advantage pricing",
    "design_principles": "design patterns principles abstractions best practices heuristics rules",
    "specification": "specs requirements protocol definitions API design technical specification system spec",
    "review_methodology": "review process gauntlet adversarial analysis code review spec review audit critique",
    "learning": "knowledge acquisition learning patterns concept extraction memory retention cognition",
    "system_quality": "system health technical debt integrity quality metrics audit code quality",
    "documentation": "docs guides README knowledge base documentation writing",
    "competitive_analysis": "competitive landscape competitor features market analysis benchmarking",
    "project_status": "milestones sprint status progress tracking project planning completion delivery",
    "integration": "system integration API wiring cross-component connections interface",
    "data_models": "data structures schemas database models data representation storage format",
    "problem_solving": "problem solving reasoning analysis root cause investigation diagnosis",
    "user_behavior": "user behavior usage patterns interaction design UX feedback",
    "communication": "communication messaging collaboration team coordination stakeholder",
    "unknown": "unknown uncategorized miscellaneous",
    # KA-006: pith_benchmarks added as canonical — tracks Pith benchmark runs, scores, failure analysis
    "pith_benchmarks": "benchmark evaluation scoring memory retrieval accuracy pith system benchmarks longmemeval ama-bench performance testing runs",
}

# Thresholds (conservative — prefer "unclassified" over misclassification)
KA_EMBEDDING_SCORE_THRESHOLD = 0.40
KA_EMBEDDING_GAP_THRESHOLD = 0.08


def _ensure_canonical_ka_embeddings():
    """Lazy-load canonical area embeddings. ~500ms first call, 0ms thereafter."""
    global _canonical_ka_embeddings, _canonical_ka_keys
    if _canonical_ka_embeddings is not None:
        return

    try:
        _load_taxonomy()
        from app.storage.embedding import embedding_engine

        if not embedding_engine.is_available:
            logger.info("Embeddings unavailable — KA embedding classification disabled")
            _canonical_ka_embeddings = np.array([])  # Empty sentinel
            _canonical_ka_keys = []
            return

        descriptions = _ACTIVE_KA_DESCRIPTIONS or _CANONICAL_KA_DESCRIPTIONS
        _canonical_ka_keys = list(descriptions.keys())
        texts = list(descriptions.values())
        _canonical_ka_embeddings = embedding_engine.embed_batch(texts)
        logger.info(f"KA canonical embeddings loaded: {len(_canonical_ka_keys)} areas")
    except Exception as e:
        logger.warning(f"Failed to load KA canonical embeddings: {e}")
        _canonical_ka_embeddings = np.array([])
        _canonical_ka_keys = []


def classify_ka_by_embedding(
    summary: str,
    embedding: np.ndarray | None = None,
) -> tuple[str | None, float, float]:
    """Classify knowledge_area using embedding similarity.

    Args:
        summary: Concept summary text
        embedding: Pre-computed embedding vector (384-dim, L2-normalized).
                   If None, computes fresh embedding from summary.

    Returns:
        (knowledge_area, confidence_score, gap) or (None, 0.0, 0.0) if:
        - Embeddings unavailable
        - Score below KA_EMBEDDING_SCORE_THRESHOLD
        - Gap below KA_EMBEDDING_GAP_THRESHOLD
    """
    _ensure_canonical_ka_embeddings()

    if _canonical_ka_embeddings is None or len(_canonical_ka_embeddings) == 0:
        return None, 0.0, 0.0

    if embedding is None:
        try:
            from app.storage.embedding import embedding_engine

            embedding = embedding_engine.embed_text(summary)
        except Exception:
            return None, 0.0, 0.0

    # 24-way cosine similarity (dot product on L2-normalized vectors)
    similarities = np.dot(_canonical_ka_embeddings, embedding)
    sorted_indices = np.argsort(similarities)[::-1]
    top_score = float(similarities[sorted_indices[0]])
    second_score = float(similarities[sorted_indices[1]])
    gap = top_score - second_score

    if top_score < KA_EMBEDDING_SCORE_THRESHOLD or gap < KA_EMBEDDING_GAP_THRESHOLD:
        return None, top_score, gap

    top_ka = _canonical_ka_keys[sorted_indices[0]]

    # Never auto-classify as meta-categories — defeats the purpose (gauntlet A1: includes "unclassified")
    if top_ka in ("general", "unknown", "unclassified"):
        return None, top_score, gap

    return top_ka, top_score, gap


# ======================================================================
# KA-ARCH-001: Lifecycle Engine (Fix 5, Fix 6)
# ======================================================================

# Promotion thresholds
KA_PROMOTION_MIN_CONCEPTS = 5        # Min concepts to promote provisional → established
KA_PROMOTION_MIN_SESSIONS = 2        # Must appear in ≥2 distinct sessions
KA_MATURITY_MIN_CONCEPTS = 25        # Promote established → mature
KA_MATURITY_MIN_AGE_DAYS = 7         # Must exist for ≥7 days to mature
KA_DECAY_INACTIVE_DAYS = 90          # Archive after 90 days of no new concepts
KA_MERGE_OVERLAP_THRESHOLD = 0.70    # Concept overlap ratio to trigger merge consideration
KA_NAME_SIMILARITY_THRESHOLD = 0.80  # KA-SEED-001: Normalized name similarity for merge candidate detection
KA_PROMOTION_LEASE_KEY = "ka_promotion_lease"
KA_PROMOTION_LEASE_TTL_SECONDS = 10 * 60


@dataclass
class KaPromotionPlan:
    counts: dict[str, tuple[int, str | None]] = field(default_factory=dict)
    promote: list[tuple[str, int, str | None]] = field(default_factory=list)
    mature: list[tuple[str, int]] = field(default_factory=list)
    archive: list[tuple[str, str]] = field(default_factory=list)
    descriptions: dict[str, str] = field(default_factory=dict)


def _recount_knowledge_areas(db):
    """Recount concept_count for all active KAs from actual concept table.

    Per A1: concept_count is derived, not incremented per-write.
    Also updates last_seen from MAX(created_at) of assigned concepts.
    """
    rows = db.execute(
        """SELECT knowledge_area, COUNT(*) AS concept_count, MAX(created_at) AS last_seen
           FROM concepts
           WHERE is_current = 1 AND knowledge_area IS NOT NULL
           GROUP BY knowledge_area"""
    ).fetchall()
    db.execute(
        "UPDATE knowledge_areas SET concept_count = 0, updated_at = datetime('now') WHERE status != 'archived'"
    )
    for name, count, last_seen in rows:
        db.execute(
            """UPDATE knowledge_areas
               SET concept_count = ?, last_seen = COALESCE(?, last_seen), updated_at = datetime('now')
               WHERE name = ? AND status != 'archived'""",
            (count, last_seen, name),
        )
    db.commit()


def _description_from_summaries(summaries: list[str]) -> str | None:
    if not summaries:
        return None
    from collections import Counter
    words = Counter()
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                  'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                  'would', 'could', 'should', 'may', 'might', 'can', 'shall',
                  'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                  'as', 'into', 'through', 'during', 'before', 'after', 'that',
                  'this', 'it', 'its', 'and', 'or', 'but', 'not', 'no', 'if'}
    for summary in summaries:
        for word in summary.lower().split():
            clean = word.strip('.,!?;:()[]"\'\u2019\u201c\u201d')
            if len(clean) > 3 and clean not in stop_words:
                words[clean] += 1
    top_words = [word for word, _ in words.most_common(12)]
    return " ".join(top_words)[:500] if top_words else None


def _days_since_sqlite_timestamp(db, timestamp: str | None) -> float | None:
    if not timestamp:
        return None
    return db.execute("SELECT julianday('now') - julianday(?)", (timestamp,)).fetchone()[0]


def _build_ka_promotion_plan(db) -> KaPromotionPlan:
    rows = db.execute(
        """SELECT knowledge_area, COUNT(*) AS concept_count, MAX(created_at) AS last_seen
           FROM concepts
           WHERE is_current = 1 AND knowledge_area IS NOT NULL
           GROUP BY knowledge_area"""
    ).fetchall()
    counts = {name: (count, last_seen) for name, count, last_seen in rows}
    plan = KaPromotionPlan(counts=counts)

    ka_rows = db.execute(
        """SELECT name, status, first_seen, last_seen
           FROM knowledge_areas
           WHERE status != 'archived'"""
    ).fetchall()
    for name, status, first_seen, last_seen in ka_rows:
        count, computed_last_seen = counts.get(name, (0, None))
        if status == "provisional" and count >= KA_PROMOTION_MIN_CONCEPTS:
            session_count = db.execute(
                """SELECT COUNT(DISTINCT session_id)
                   FROM concepts
                   WHERE knowledge_area = ? AND is_current = 1""",
                (name,),
            ).fetchone()[0]
            if session_count >= KA_PROMOTION_MIN_SESSIONS:
                plan.promote.append((name, count, computed_last_seen))
                summary_rows = db.execute(
                    """SELECT summary FROM concepts
                       WHERE knowledge_area = ? AND is_current = 1
                       ORDER BY created_at DESC LIMIT 10""",
                    (name,),
                ).fetchall()
                desc = _description_from_summaries([r[0] for r in summary_rows if r[0]])
                if desc:
                    plan.descriptions[name] = desc
        first_seen_age_days = _days_since_sqlite_timestamp(db, first_seen)
        if status == "established" and count >= KA_MATURITY_MIN_CONCEPTS and (
            first_seen_age_days is not None and first_seen_age_days >= KA_MATURITY_MIN_AGE_DAYS
        ):
            plan.mature.append((name, count))
        effective_last_seen = computed_last_seen or last_seen
        last_seen_age_days = _days_since_sqlite_timestamp(db, effective_last_seen)
        if status not in ("seed", "archived") and (
            last_seen_age_days is not None and last_seen_age_days >= KA_DECAY_INACTIVE_DAYS
        ):
            plan.archive.append((name, status))
    return plan


def _apply_ka_promotion_plan(db, plan: KaPromotionPlan) -> dict[str, str]:
    transitions: dict[str, str] = {}
    db.execute(
        "UPDATE knowledge_areas SET concept_count = 0, updated_at = datetime('now') WHERE status != 'archived'"
    )
    for name, (count, last_seen) in plan.counts.items():
        db.execute(
            """UPDATE knowledge_areas
               SET concept_count = ?, last_seen = COALESCE(?, last_seen), updated_at = datetime('now')
               WHERE name = ? AND status != 'archived'""",
            (count, last_seen, name),
        )
    for name, count, _last_seen in plan.promote:
        cursor = db.execute(
            "UPDATE knowledge_areas SET status = 'established', updated_at = datetime('now') WHERE name = ? AND status = 'provisional'",
            (name,),
        )
        if cursor.rowcount:
            transitions[name] = "provisional → established"
            _canonical_areas.add(name)
            logger.info("KA promoted: '%s' -> established (concepts=%s)", name, count)
            desc = plan.descriptions.get(name)
            if desc:
                db.execute(
                    "UPDATE knowledge_areas SET description = ?, updated_at = datetime('now') WHERE name = ?",
                    (desc, name),
                )
    for name, count in plan.mature:
        cursor = db.execute(
            "UPDATE knowledge_areas SET status = 'mature', updated_at = datetime('now') WHERE name = ? AND status = 'established'",
            (name,),
        )
        if cursor.rowcount:
            transitions[name] = "established → mature"
            logger.info("KA matured: '%s' -> mature (concepts=%s)", name, count)
    for name, old_status in plan.archive:
        cursor = db.execute(
            "UPDATE knowledge_areas SET status = 'archived', updated_at = datetime('now') WHERE name = ? AND status NOT IN ('seed', 'archived')",
            (name,),
        )
        if cursor.rowcount:
            transitions[name] = f"{old_status} → archived"
            _canonical_areas.discard(name)
            logger.info("KA archived: '%s' (inactive %sd)", name, KA_DECAY_INACTIVE_DAYS)
    return transitions


def promote_knowledge_areas(*, raise_on_contention: bool = False) -> dict[str, str]:
    """Promote KAs based on lifecycle thresholds. Run as background task.

    Lifecycle transitions:
      provisional → established: concept_count ≥ 5 AND seen in ≥ 2 sessions
      established → mature: concept_count ≥ 25 AND age ≥ 7 days
      any → archived: no new concepts for 90 days (except seed KAs)

    Returns dict of {ka_name: old_status → new_status} transitions.
    KA-ARCH-001 Fix 5.
    """
    try:
        from app.storage import managed_write_db, read_snapshot_db

        total_start = time.perf_counter()
        pre_start = time.perf_counter()
        with read_snapshot_db("ka_promotion_precompute") as db:
            plan = _build_ka_promotion_plan(db)
        logger.info("ka_promotion_precompute_ms=%.1f", (time.perf_counter() - pre_start) * 1000)

        apply_start = time.perf_counter()
        with managed_write_db(timeout_s=0.05, operation="ka_promotion_apply") as db:
            transitions = _apply_ka_promotion_plan(db, plan)
        logger.info(
            "ka_promotion_apply_ms=%.1f ka_promotion_total_ms=%.1f transitions=%s",
            (time.perf_counter() - apply_start) * 1000,
            (time.perf_counter() - total_start) * 1000,
            len(transitions),
        )
        return transitions
    except (RuntimeError, sqlite3.OperationalError) as exc:
        logger.warning("KA promotion skipped under DB contention: %s", exc)
        if raise_on_contention:
            raise
    return {}


# ── ARCH-D05: Periodic promotion support ─────────────────────────────────────

def _should_run_promotion(interval_minutes: int = 30) -> bool:
    """Check if KA promotion should run (not run recently).

    Uses the metadata table to track last promotion timestamp.
    Returns True if promotion hasn't run in the last interval_minutes.
    """
    from datetime import datetime

    from app.storage import get_metadata

    last_run = get_metadata("last_ka_promotion_at")
    if last_run is None:
        return True
    try:
        last_dt = datetime.fromisoformat(last_run)
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=UTC)
        age_minutes = (datetime.now(UTC) - last_dt).total_seconds() / 60
        return age_minutes >= interval_minutes
    except (ValueError, TypeError):
        return True  # Corrupt timestamp → re-run


def _try_acquire_ka_promotion_lease(owner: str, *, timeout_s: float = 0.05) -> bool:
    """Acquire a short-lived promotion lease, returning False under contention."""
    from app.core.datetime_utils import _utc_now_iso
    from app.storage import managed_write_db

    now = datetime.now(UTC)
    try:
        with managed_write_db(timeout_s=timeout_s, operation="ka_promotion_lease_acquire") as db:
            row = db.execute("SELECT value FROM metadata WHERE key = ?", (KA_PROMOTION_LEASE_KEY,)).fetchone()
            if row:
                try:
                    lease = json.loads(row[0])
                    started_at = datetime.fromisoformat(lease.get("started_at", ""))
                    if started_at.tzinfo is None:
                        started_at = started_at.replace(tzinfo=UTC)
                    age_s = (now - started_at).total_seconds()
                    if age_s < KA_PROMOTION_LEASE_TTL_SECONDS:
                        logger.info("ka_promotion_lease_active owner=%s age_s=%.1f", lease.get("owner"), age_s)
                        return False
                    logger.warning("ka_promotion_lease_stale owner=%s age_s=%.1f", lease.get("owner"), age_s)
                except Exception:
                    logger.warning("ka_promotion_lease_stale reason=malformed")
            db.execute(
                """INSERT OR REPLACE INTO metadata (key, value, updated_at)
                   VALUES (?, ?, ?)""",
                (KA_PROMOTION_LEASE_KEY, json.dumps({"owner": owner, "started_at": _utc_now_iso()}), _utc_now_iso()),
            )
            return True
    except (RuntimeError, sqlite3.OperationalError) as exc:
        logger.info("ka_promotion_skipped_contention phase=lease_acquire error=%s", exc)
        return False


def _release_ka_promotion_lease(owner: str, *, status: str, timeout_s: float = 0.05) -> None:
    from app.core.datetime_utils import _utc_now_iso
    from app.storage import managed_write_db

    try:
        with managed_write_db(timeout_s=timeout_s, operation="ka_promotion_lease_release") as db:
            row = db.execute("SELECT value FROM metadata WHERE key = ?", (KA_PROMOTION_LEASE_KEY,)).fetchone()
            owns_lease = row is None
            if row:
                try:
                    lease = json.loads(row[0])
                except Exception:
                    lease = {}
                if lease.get("owner") in (None, owner):
                    owns_lease = True
                    db.execute("DELETE FROM metadata WHERE key = ?", (KA_PROMOTION_LEASE_KEY,))
            if status == "success" and owns_lease:
                db.execute(
                    """INSERT OR REPLACE INTO metadata (key, value, updated_at)
                       VALUES ('last_ka_promotion_at', ?, ?)""",
                    (_utc_now_iso(), _utc_now_iso()),
                )
    except (RuntimeError, sqlite3.OperationalError) as exc:
        logger.info("ka_promotion_skipped_contention phase=lease_release error=%s", exc)


def _run_lease_guarded_ka_promotion(owner: str = "ka_promotion") -> dict[str, str]:
    if not _try_acquire_ka_promotion_lease(owner):
        return {}
    status = "success"
    try:
        return promote_knowledge_areas(raise_on_contention=True)
    except (RuntimeError, sqlite3.OperationalError):
        status = "skipped_contention"
        return {}
    except Exception:
        status = "error"
        raise
    finally:
        _release_ka_promotion_lease(owner, status=status)


def _record_promotion_run() -> None:
    """Record that KA promotion just ran."""
    _release_ka_promotion_lease("manual_record", status="success")


def _generate_ka_description(db, ka_name: str):
    """Auto-generate a KA description from its concept cluster summaries.

    Takes the 10 most recent concept summaries, extracts common themes via
    simple word frequency, and builds a description string.
    """
    rows = db.execute(
        """SELECT summary FROM concepts
           WHERE knowledge_area = ? AND is_current = 1
           ORDER BY created_at DESC LIMIT 10""",
        (ka_name,)
    ).fetchall()

    if not rows:
        return

    summaries = [r[0] for r in rows if r[0]]
    from collections import Counter
    words = Counter()
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                  'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                  'would', 'could', 'should', 'may', 'might', 'can', 'shall',
                  'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                  'as', 'into', 'through', 'during', 'before', 'after', 'that',
                  'this', 'it', 'its', 'and', 'or', 'but', 'not', 'no', 'if'}
    for s in summaries:
        for word in s.lower().split():
            clean = word.strip('.,!?;:()[]"\'\u2019\u201c\u201d')
            if len(clean) > 3 and clean not in stop_words:
                words[clean] += 1

    top_words = [w for w, _ in words.most_common(12)]
    description = " ".join(top_words)

    db.execute(
        "UPDATE knowledge_areas SET description = ?, updated_at = datetime('now') WHERE name = ?",
        (description[:500], ka_name)
    )


def _normalize_ka_name(name: str) -> str:
    """Normalize KA name for comparison: lowercase, strip hyphens/underscores/spaces.

    KA-SEED-001 Fix 5: Name normalization for merge detection.
    """
    return name.lower().replace("-", "_").replace(" ", "_").strip("_")


def detect_ka_merges() -> list[tuple[str, str, float]]:
    """Detect KA pairs that should be merged based on concept overlap OR name similarity.

    Two detection modes (KA-SEED-001 Fix 5):
    1. Name normalization — catches pith-retrieval vs pith_retrieval (exact after normalize)
    2. Name similarity via SequenceMatcher — catches pith_bench vs pith_benchmarks
    3. Concept overlap (Jaccard >= 0.70) — existing behavior preserved

    Returns list of (ka_a, ka_b, score) sorted by score descending.
    Does NOT auto-merge — returns candidates for review.
    KA-ARCH-001 Fix 6 + KA-SEED-001 Fix 5.
    """
    from difflib import SequenceMatcher

    from app.storage import get_db_connection as get_db
    db = get_db()

    # A1: Include all active statuses for comprehensive detection
    active_kas = db.execute(
        "SELECT name FROM knowledge_areas WHERE status IN ('seed', 'provisional', 'established', 'mature')"
    ).fetchall()
    ka_names = [r[0] for r in active_kas]

    if len(ka_names) < 2:
        return []

    candidates = []
    seen_pairs = set()

    # Pass 1: Name normalization — O(n²) string comparison, sub-1ms for n<100
    normalized = {name: _normalize_ka_name(name) for name in ka_names}
    for i, ka_a in enumerate(ka_names):
        for ka_b in ka_names[i+1:]:
            norm_a, norm_b = normalized[ka_a], normalized[ka_b]
            # Exact normalized match (pith-retrieval == pith_retrieval)
            if norm_a == norm_b:
                candidates.append((ka_a, ka_b, 1.0))
                seen_pairs.add((ka_a, ka_b))
                continue
            # Fuzzy name similarity (e.g., pith_bench vs pith_benchmarks)
            ratio = SequenceMatcher(None, norm_a, norm_b).ratio()
            if ratio >= KA_NAME_SIMILARITY_THRESHOLD:
                candidates.append((ka_a, ka_b, ratio))
                seen_pairs.add((ka_a, ka_b))

    # Pass 2: Concept overlap (existing logic) — skip pairs already found by name
    ka_concepts: dict[str, set] = {}
    for ka in ka_names:
        rows = db.execute(
            "SELECT id FROM concepts WHERE knowledge_area = ? AND is_current = 1",
            (ka,)
        ).fetchall()
        ka_concepts[ka] = {r[0] for r in rows}

    for i, ka_a in enumerate(ka_names):
        for ka_b in ka_names[i+1:]:
            if (ka_a, ka_b) in seen_pairs:
                continue
            set_a, set_b = ka_concepts[ka_a], ka_concepts[ka_b]
            if not set_a or not set_b:
                continue
            intersection = len(set_a & set_b)
            union = len(set_a | set_b)
            overlap = intersection / union if union > 0 else 0.0
            if overlap >= KA_MERGE_OVERLAP_THRESHOLD:
                candidates.append((ka_a, ka_b, overlap))

    return sorted(candidates, key=lambda x: x[2], reverse=True)


def merge_kas(absorber: str, absorbed: str):
    """Merge absorbed KA into absorber. Updates all concept references.

    KA-ARCH-001 Fix 6: Merge execution.
    """
    from app.storage import get_db_connection as get_db
    db = get_db()

    db.execute(
        "UPDATE concepts SET knowledge_area = ? WHERE knowledge_area = ?",
        (absorber, absorbed)
    )
    db.execute(
        """UPDATE concepts SET data = json_set(data, '$.metadata.knowledge_area', ?)
           WHERE json_extract(data, '$.metadata.knowledge_area') = ?""",
        (absorber, absorbed)
    )

    new_count = db.execute(
        "SELECT COUNT(*) FROM concepts WHERE knowledge_area = ? AND is_current = 1",
        (absorber,)
    ).fetchone()[0]
    db.execute(
        "UPDATE knowledge_areas SET concept_count = ?, updated_at = datetime('now') WHERE name = ?",
        (new_count, absorber)
    )

    db.execute(
        """UPDATE knowledge_areas
           SET status = 'archived', parent_ka = ?, updated_at = datetime('now')
           WHERE name = ?""",
        (absorber, absorbed)
    )

    db.commit()
    _canonical_areas.discard(absorbed)
    logger.info(f"KA merged: '{absorbed}' → '{absorber}' (new count={new_count})")


def get_ka_hints(max_hints: int = 12) -> list[str]:
    """Get the user's most active KA names for extraction prompt hints.

    Returns up to max_hints KA names, prioritizing established/mature KAs
    by concept_count. Falls back to universal KAs for new users.

    HARDENING-KA-001: Dynamic max_hints — users with many KAs get more hints
    so extraction can target the right area. Capped at 20 to avoid prompt bloat.
    KA-ARCH-001 Fix 9: Adaptive extraction primitive.
    """
    _load_taxonomy()

    try:
        from app.storage import get_db_connection as get_db
        db = get_db()
        # HARDENING-KA-001: Count eligible KAs to scale hints dynamically
        ka_count_row = db.execute(
            """SELECT COUNT(*) FROM knowledge_areas
               WHERE status IN ('seed', 'established', 'mature')
                 AND name NOT IN ('general', 'unclassified', 'unknown')"""
        ).fetchone()
        ka_count = ka_count_row[0] if ka_count_row else 0
        # Scale: ≤15 KAs → use max_hints as-is; 16-30 → 16; 30+ → 20
        if ka_count > 30:
            effective_max = min(max(max_hints, 20), 20)
        elif ka_count > 15:
            effective_max = min(max(max_hints, 16), 20)
        else:
            effective_max = max_hints
        rows = db.execute(
            """SELECT name FROM knowledge_areas
               WHERE status IN ('seed', 'established', 'mature')
                 AND name NOT IN ('general', 'unclassified', 'unknown')
               ORDER BY
                 CASE status WHEN 'mature' THEN 0 WHEN 'established' THEN 1 ELSE 2 END,
                 concept_count DESC
               LIMIT ?""",
            (effective_max,)
        ).fetchall()
        if rows and len(rows) >= 5:
            return [r[0] for r in rows]
    except Exception:
        pass

    # Fallback: universal domain-agnostic KAs
    return ["knowledge", "workflow", "relationships", "context", "goals", "observations"]


def classify_knowledge_area(
    summary: str,
    raw_area: str,
    strict: bool = True,
    embedding: np.ndarray | None = None,
) -> tuple[str, str, float | None]:
    """Multi-tier KA classification: normalize → keyword → embedding.

    Consolidates the duplicated cascade from session.py and learning.py (DEBT-108).

    Args:
        summary: Concept summary text.
        raw_area: Raw knowledge_area string from client or resolver.
        strict: If True (write-time), unknowns → "unclassified". If False (propose), unknowns pass through.
        embedding: Optional pre-computed embedding vector for Tier 2.

    Returns:
        (knowledge_area, ka_source, ka_confidence):
        - knowledge_area: Classified KA string
        - ka_source: Classification method ("canonical", "inferred", "embedding", "default", etc.)
        - ka_confidence: Float confidence score (from embedding tier) or None
    """
    knowledge_area, ka_source = normalize_knowledge_area(raw_area, strict=strict)
    ka_confidence = None

    if knowledge_area in ("general", "unclassified") and summary:
        # Tier 1: Keyword inference (fast, 0ms)
        inferred = infer_knowledge_area(summary)
        if inferred:
            logger.info(f"KA classify (keyword): '{summary[:60]}' → {inferred}")
            knowledge_area = inferred
            ka_source = "inferred"
        else:
            # Tier 2: Embedding classification
            emb_ka, emb_score, emb_gap = classify_ka_by_embedding(summary, embedding=embedding)
            if emb_ka:
                logger.info(
                    f"KA classify (embedding): '{summary[:60]}' → {emb_ka} (score={emb_score:.3f}, gap={emb_gap:.3f})"
                )
                knowledge_area = emb_ka
                ka_source = "embedding"
                ka_confidence = emb_score
            # If both fail, knowledge_area stays as-is ("unclassified" or "general")

    return knowledge_area, ka_source, ka_confidence
