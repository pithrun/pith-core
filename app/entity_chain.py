"""RETRIEVAL-047: Entity-Chain Keyword Retriever for production.

Ported from benchmarks/adapter/entity_chain.py (RETRIEVAL-045).
Extracts named entities from user query, does SQL keyword search per entity,
chains extracted values for multi-hop lookups. Results are unioned with
embedding retrieval in conversation_turn (session.py S4.6).

Key differences from benchmark version:
- Returns SearchResult (not RetrievedConcept)
- Uses production DB path from config
- No benchmark preamble stripping
- Feature-gated via PITH_ENTITY_CHAIN env var
- Time-budgeted (default 150ms) to avoid blocking conversation_turn

RETRIEVAL-073: Hop-priority scoring — hop 1 results score 0.85 (entity-specific
  gold), hop 2 scores 0.78, hop 3+ scores 0.68. Previously flat 0.75 for all
  hops caused entity-specific concepts to be indistinguishable from hop-3+ noise
  in budget governance, losing 17/24 SH 32k failures where gold existed in brain.

RETRIEVAL-075: Total entity chain cap. Entity chain sprawl (2 hop-1 concepts
  expanding to 71 total) dilutes entity-specific gold in budget governance.
  Total cap env-gated via PITH_ENTITY_CHAIN_TOTAL_CAP (default 30).
"""

import re
import os
import time
import sqlite3
import logging
from typing import Optional

from app.models import SearchResult

logger = logging.getLogger(__name__)

# Feature flag
ENTITY_CHAIN_ENABLED = os.environ.get("PITH_ENTITY_CHAIN", "").lower() in ("true", "1")
ENTITY_CHAIN_BUDGET_MS = int(os.environ.get("PITH_ENTITY_CHAIN_BUDGET_MS", "150"))

# RETRIEVAL-073: Hop-priority relevance scores.
# Hop 1 = entity-specific gold (highest priority to survive budget trim).
# Hop 2 = one-step chain facts (still valuable for multi-hop).
# Hop 3+ = distant chain noise (lowest priority).
# Previously all hops scored flat 0.75, making entity-specific gold
# indistinguishable from noise in budget governance tiering.
_HOP_SCORES = {
    1: float(os.environ.get("PITH_EC_HOP1_SCORE", "0.85")),
    2: float(os.environ.get("PITH_EC_HOP2_SCORE", "0.78")),
}
_HOP_DEFAULT_SCORE = float(os.environ.get("PITH_EC_HOP_DEFAULT_SCORE", "0.68"))

# RETRIEVAL-075: Total entity chain result cap.
# Prevents entity chain sprawl (2 hop-1 → 71 total) from overwhelming
# budget governance with noise. Default 30 = ~7% of a 400-concept brain.
_TOTAL_CAP = int(os.environ.get("PITH_ENTITY_CHAIN_TOTAL_CAP", "30"))

# Common words to exclude from entity extraction
_STOPWORDS = {
    'what', 'when', 'where', 'who', 'which', 'how', 'why', 'does', 'did',
    'is', 'are', 'was', 'were', 'has', 'have', 'had', 'the', 'a', 'an',
    'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'and',
    'or', 'not', 'be', 'been', 'being', 'that', 'this', 'based',
    'answer', 'question', 'now', 'you', 'need', 'find', 'tell', 'me',
    'about', 'know', 'can', 'could', 'would', 'should', 'do', 'my',
    'his', 'her', 'its', 'their', 'your', 'our', 'current', 'please',
    'also', 'just', 'like', 'think', 'say', 'said', 'get', 'got',
    'make', 'made', 'take', 'see', 'come', 'want', 'look', 'use',
    'give', 'most', 'some', 'any', 'all', 'each', 'every', 'both',
    'few', 'more', 'other', 'new', 'old', 'first', 'last', 'long',
    'great', 'little', 'own', 'same', 'big', 'high', 'different',
    'small', 'large', 'next', 'early', 'young', 'important', 'public',
    'good', 'right', 'being', 'still', 'here', 'there', 'then', 'than',
    'will', 'shall', 'may', 'might', 'must', 'very', 'after', 'before',
    'remember', 'recall', 'mentioned', 'talked', 'discussed',
}

# Property words from queries that indicate what relationship to find
_PROPERTY_WORDS = {
    'country', 'language', 'capital', 'continent', 'city',
    'religion', 'sport', 'university', 'institution', 'origin',
    'music', 'genre', 'position', 'citizen', 'citizenship',
    'president', 'head', 'government', 'founder', 'creator',
    'author', 'spouse', 'location', 'birthday', 'born',
    'educated', 'headquarters', 'favorite', 'preference',
    'job', 'work', 'company', 'team', 'school', 'home',
    'address', 'email', 'phone', 'name', 'age', 'pet',
}

# Copula verbs for value extraction
_COPULA_VERBS = [
    ' is ', ' was ', ' are ', ' were ', ' has ', ' had ',
    ' plays ', ' speaks ', ' works ', ' lives ', ' died ',
    ' created ', ' founded ', ' employed ', ' married ',
    ' located ', ' received ', ' established ', ' holds ',
    ' associated ', ' originated ', ' comes ', ' prefers ',
    ' enjoys ', ' likes ', ' uses ', ' drives ', ' owns ',
    ' attends ', ' visited ', ' studies ', ' teaches ',
]

# Compound copulas (more specific, checked first)
_COMPOUND_COPULAS = [
    ' is associated with ', ' is located in ', ' is a citizen of ',
    ' is a member of ', ' is affiliated with ', ' is employed by ',
    ' is headquartered in ', ' was educated at ', ' was born in ',
    ' is married to ', ' was founded by ', ' lives in ',
    ' works at ', ' works for ', ' goes to ', ' prefers ',
]


class EntityChainRetriever:
    """Entity-chain retrieval via direct keyword search in the brain DB.

    For multi-hop questions like:
      "What's the capital of the country where Andrew lives?"

    Pipeline:
      1. Extract named entities from question (proper nouns)
      2. SQL keyword search for leaf entity -> hop 1 facts
      3. Extract value entities from hop 1 facts (object after copula)
      4. SQL keyword search for hop 2 entities -> hop 2 facts
      5. Return all found concepts (caller unions with embedding results)
    """

    def __init__(self, db_path: str, max_hops: int = 4, max_per_hop: int = 8):
        self.db_path = db_path
        self.max_hops = max_hops
        self.max_per_hop = max_per_hop
        self._question_keywords: list[str] = []
        self.last_searched_entities: set[str] = set()

    def retrieve(self, question: str, budget_ms: int = 150) -> list[SearchResult]:
        """Main entry: extract entities, chain keyword lookups, return results.

        Args:
            question: The user's message/query
            budget_ms: Time budget in ms. Returns partial results if exceeded.

        Returns:
            List of SearchResult objects to union with embedding results.
        """
        t0 = time.perf_counter()

        # Step 1: Extract entities
        entities = self._extract_entities(question)
        if not entities:
            return []

        logger.info(f"ENTITY-CHAIN: Extracted entities: {entities}")

        # Extract question keywords for SQL boosting
        q_words = set(question.lower().split()) - _STOPWORDS
        self._question_keywords = list(q_words & _PROPERTY_WORDS)

        all_concepts: dict[str, SearchResult] = {}
        hop_queue = list(entities)
        searched: set[str] = set()
        self.last_searched_entities = set()
        hop = 0

        while hop_queue and hop < self.max_hops:
            hop += 1
            current_entities = list(hop_queue)
            hop_queue = []

            for entity in current_entities:
                # Budget check
                elapsed_ms = (time.perf_counter() - t0) * 1000
                if elapsed_ms > budget_ms:
                    logger.info(
                        f"ENTITY-CHAIN: Budget exhausted ({elapsed_ms:.0f}ms > {budget_ms}ms) "
                        f"at hop {hop}, returning {len(all_concepts)} partial results"
                    )
                    self.last_searched_entities = searched
                    return list(all_concepts.values())

                entity_lower = entity.lower().strip()
                if entity_lower in searched or len(entity_lower) < 2:
                    continue
                searched.add(entity_lower)

                # Keyword search
                facts = self._keyword_search(entity_lower, hop)
                for f in facts:
                    if f.concept_id not in all_concepts:
                        all_concepts[f.concept_id] = f

                # Extract values for next hop
                if hop < self.max_hops:
                    for f in facts:
                        values = self._extract_fact_values(f.summary, entity_lower)
                        for v in values:
                            if v.lower() not in searched and len(v) > 1:
                                hop_queue.append(v)

            logger.info(
                f"ENTITY-CHAIN: Hop {hop}: {len(current_entities)} entities -> "
                f"{len(all_concepts)} total, next queue: {len(hop_queue)}"
            )

            # RETRIEVAL-075: Early termination when total cap reached
            if len(all_concepts) >= _TOTAL_CAP:
                logger.info(
                    f"RETRIEVAL-075: Total cap ({_TOTAL_CAP}) reached at hop {hop}, "
                    f"stopping chain traversal"
                )
                break

        self.last_searched_entities = searched
        elapsed_ms = (time.perf_counter() - t0) * 1000

        results = list(all_concepts.values())

        # RETRIEVAL-075: Total cap — prevent entity chain sprawl from overwhelming
        # budget governance. Sort by relevance_score (hop priority) so hop-1
        # entity-specific gold survives the cap over hop-3+ noise.
        if len(results) > _TOTAL_CAP:
            # Secondary sort by question keyword overlap (RETRIEVAL-058 compat)
            def _cap_sort_key(r):
                _kw_hit = 0
                if self._question_keywords:
                    _s = (r.summary or "").lower()
                    _kw_hit = 1 if any(kw in _s for kw in self._question_keywords) else 0
                return (_kw_hit, r.relevance_score)
            results.sort(key=_cap_sort_key, reverse=True)
            logger.info(
                f"RETRIEVAL-075: Entity chain capped {len(results)} -> {_TOTAL_CAP} "
                f"(hop scores preserved for budget governance)"
            )
            results = results[:_TOTAL_CAP]

        logger.info(
            f"ENTITY-CHAIN: Complete: {len(results)} concepts from "
            f"{hop} hops in {elapsed_ms:.0f}ms"
        )
        return results

    def _extract_entities(self, question: str) -> list[str]:
        """Extract named entities (proper nouns, multi-word names) from question."""
        entities = []

        # 1. Quoted strings
        quoted = re.findall(r'"([^"]+)"', question)
        entities.extend(quoted)

        # 2. Proper noun runs
        q_clean = question.replace('"', ' ').replace("'", " ")
        words = q_clean.split()
        current_entity: list[str] = []

        for w in words:
            clean = w.strip('?,!.;:()')
            if not clean:
                if current_entity:
                    ent = ' '.join(current_entity)
                    if ent.lower() not in _STOPWORDS and len(ent) > 1:
                        entities.append(ent)
                    current_entity = []
                continue

            is_proper = (
                clean[0].isupper()
                and clean.lower() not in _STOPWORDS
                and len(clean) > 1
            )
            is_number_after = clean[0].isdigit() and current_entity

            if is_proper or is_number_after:
                current_entity.append(clean)
            else:
                if current_entity:
                    ent = ' '.join(current_entity)
                    if ent.lower() not in _STOPWORDS and len(ent) > 1:
                        entities.append(ent)
                    current_entity = []

        if current_entity:
            ent = ' '.join(current_entity)
            if ent.lower() not in _STOPWORDS and len(ent) > 1:
                entities.append(ent)

        # RETRIEVAL-058B: Extract hyphenated compounds (e.g. "split-finger fastball").
        # These are often domain-specific terms (sports techniques, proper nouns)
        # that the proper-noun extractor misses because they're lowercase.
        hyphenated = re.findall(r'\b(\w+-\w+(?:\s+\w+)?)\b', q_clean)
        for h in hyphenated:
            h_clean = h.strip('?,!.;:()')
            if len(h_clean) > 3 and h_clean.lower() not in _STOPWORDS:
                entities.append(h_clean)

        # Deduplicate: prefer longer entities that subsume shorter ones
        entities = list(dict.fromkeys(entities))
        filtered = []
        for e in sorted(entities, key=len, reverse=True):
            if any(e.lower() in kept.lower() for kept in filtered):
                continue
            filtered.append(e)

        # Leaf entity first (usually last in question for multi-hop)
        filtered.reverse()
        return filtered

    def _keyword_search(self, entity: str, hop: int) -> list[SearchResult]:
        """Search brain DB for concepts mentioning this entity."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row

            words = entity.lower().split()
            if not words:
                conn.close()
                return []

            where_parts = []
            params = []
            for w in words:
                where_parts.append("LOWER(summary) LIKE ?")
                params.append(f"%{w}%")

            # Subject-position boost: entity before copula verb ranks higher
            entity_esc = entity.replace("'", "''")
            subject_boost = (
                f"CASE "
                f"WHEN INSTR(LOWER(summary), ' is ') > 0 "
                f"AND INSTR(LOWER(summary), '{entity_esc}') > 0 "
                f"AND INSTR(LOWER(summary), '{entity_esc}') < INSTR(LOWER(summary), ' is ') "
                f"THEN 2 "
                f"WHEN INSTR(LOWER(summary), ' was ') > 0 "
                f"AND INSTR(LOWER(summary), '{entity_esc}') > 0 "
                f"AND INSTR(LOWER(summary), '{entity_esc}') < INSTR(LOWER(summary), ' was ') "
                f"THEN 2 "
                f"ELSE 0 END"
            )

            # Question keyword boost
            kw_boost = ""
            if self._question_keywords:
                kw_parts = [
                    f"CASE WHEN LOWER(summary) LIKE '%{w}%' THEN 1 ELSE 0 END"
                    for w in self._question_keywords[:5]
                ]
                kw_boost = " + " + " + ".join(kw_parts)

            # Taper for deeper hops (3-tier from adapter)
            # hop 1-2: full budget, hop 3: half, hop 4+: third
            effective_limit = self.max_per_hop
            if hop >= 4:
                effective_limit = max(3, self.max_per_hop // 3)
            elif hop >= 3:
                effective_limit = max(5, self.max_per_hop // 2)

            query = f"""
                SELECT id, summary, confidence, knowledge_area, created_at
                FROM concepts
                WHERE status = 'active'
                  AND ({' AND '.join(where_parts)})
                ORDER BY ({subject_boost}{kw_boost}) DESC, confidence DESC
                LIMIT ?
            """
            params.append(effective_limit)
            rows = conn.execute(query, params).fetchall()
            conn.close()

            # RETRIEVAL-073: Hop-priority scoring
            hop_score = _HOP_SCORES.get(hop, _HOP_DEFAULT_SCORE)

            results = []
            for row in rows:
                results.append(SearchResult(
                    concept_id=row["id"],
                    version="v1",
                    summary=row["summary"] or "",
                    confidence=row["confidence"] or 0.5,
                    relevance_score=hop_score,
                    knowledge_area=row["knowledge_area"],
                    created_at=row["created_at"],  # RETRIEVAL-053
                ))

            if results:
                logger.info(
                    f"ENTITY-CHAIN: '{entity}' -> {len(results)} concepts (hop {hop})"
                )
            return results

        except Exception as e:
            logger.error(f"ENTITY-CHAIN: Keyword search failed for '{entity}': {e}")
            return []

    def _extract_fact_values(self, summary: str, search_entity: str) -> list[str]:
        """Extract the value/object from a fact for next-hop chaining.

        Given: "Andrew lives in San Francisco" + search="andrew"
        Returns: ["San Francisco"]
        """
        values = []
        summary_lower = summary.lower()

        # Find best copula position (compound first, then simple)
        best_idx = -1
        best_len = 0

        for cop in _COMPOUND_COPULAS:
            idx = summary_lower.find(cop)
            if idx > 0 and (best_idx == -1 or idx < best_idx):
                best_idx = idx
                best_len = len(cop)

        if best_idx == -1:
            for cop in _COPULA_VERBS:
                idx = summary_lower.find(cop)
                if idx > 0 and (best_idx == -1 or idx < best_idx):
                    best_idx = idx
                    best_len = len(cop)

        if best_idx == -1:
            return values

        value_part = summary[best_idx + best_len:].strip()

        # Strip filler: "a citizen of", "the position of", leading copulas
        # Leading copulas happen when compound verbs like "plays is X" split
        # on the first verb, leaving "is X" as the value.
        value_part = re.sub(
            r'^(?:is |are |was |were |has |had )?'
            r'(?:a |an |the )?(?:citizen |member |position |'
            r'sport |country |city |language |religion |'
            r'genre |type |capital |university |'
            r'location |place )?(?:of |in |at |for )?',
            '', value_part, flags=re.I
        ).strip()

        if not value_part:
            return values

        # Extract proper nouns from value
        words = value_part.split()
        current: list[str] = []
        connectors = {'of', 'the', 'and', 'de', 'la', 'le', 'del', 'von', 'van'}

        for w in words:
            clean = w.strip('?,!.;:()')
            if not clean:
                continue
            is_proper = clean[0].isupper() and len(clean) > 1
            is_number = clean[0].isdigit() and current
            is_conn = clean.lower() in connectors and current

            if is_proper or is_number or is_conn:
                current.append(clean)
            else:
                if current:
                    while current and current[-1].lower() in connectors:
                        current.pop()
                    ent = ' '.join(current)
                    if len(ent) > 1:
                        values.append(ent)
                    current = []

        if current:
            while current and current[-1].lower() in connectors:
                current.pop()
            ent = ' '.join(current)
            if len(ent) > 1:
                values.append(ent)

        # Fallback: use whole value if no proper nouns found
        if not values and len(value_part) > 2:
            clean_val = re.sub(r'\s+(?:of|in|at|for|to|with|by|from)\s*$', '', value_part).strip()
            if clean_val and len(clean_val) > 2:
                values.append(clean_val)

        return values


# Module-level singleton (lazy init)
_retriever: Optional[EntityChainRetriever] = None


def get_entity_chain_retriever() -> Optional[EntityChainRetriever]:
    """Get or create the entity chain retriever singleton.

    Returns None if PITH_ENTITY_CHAIN is not enabled.
    """
    global _retriever
    if not ENTITY_CHAIN_ENABLED:
        return None
    if _retriever is None:
        profile = os.environ.get("PITH_PROFILE", "rose")
        data_dir = os.environ.get(
            "PITH_DATA_DIR",
            os.path.expanduser(f"~/pith-data/{profile}")
        )
        db_path = os.path.join(data_dir, "pith.db")
        if not os.path.exists(db_path):
            logger.warning(f"ENTITY-CHAIN: DB not found at {db_path}")
            return None
        _max_hops = int(os.environ.get("PITH_ENTITY_CHAIN_MAX_HOPS", "4"))
        _retriever = EntityChainRetriever(db_path=db_path, max_hops=_max_hops)
        logger.info(f"ENTITY-CHAIN: Initialized with db={db_path}, max_hops={_max_hops}")
    return _retriever
