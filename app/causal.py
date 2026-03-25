"""
Causal chain features for Pith server.

Implements causal chain traversal, path finding, and edge management per
TEMPORAL_RETRIEVAL_SPEC.md §10.1 and §10.4.

Provides:
- pith_trace_cause: DAG traversal with cycle detection (recursive CTE)
- pith_find_path: Typed shortest path with confidence decay (BFS via CTE)
- create_causal_edge: Directed edge creation with mechanism quality gate
"""

import logging
from datetime import UTC, datetime
from typing import Any

from app.storage import _get_connection

logger = logging.getLogger(__name__)


# Universal error response pattern
def _error_response(code: str, message: str) -> dict[str, Any]:
    """Return universal error response."""
    return {"success": False, "error": {"code": code, "message": message}}


def _success_response(data: Any) -> dict[str, Any]:
    """Return universal success response."""
    return {"success": True, "data": data}


def pith_trace_cause(
    concept_id: str, direction: str = "root_cause", max_depth: int = 5, chain_id: str | None = None
) -> dict[str, Any]:
    """
    Traverse causal DAG structure to find root causes or consequences.

    TEMPORAL_RETRIEVAL_SPEC.md §10.1: Causal chain traversal with recursive CTE
    and visited-set pattern for cycle detection [M7].

    Args:
        concept_id: Starting concept ID
        direction: "root_cause" (traverse backward) or "consequences" (forward)
        max_depth: Hard cap on traversal depth (default 5)
        chain_id: Optional chain identifier for filtering

    Returns:
        Universal response dict with structure:
        {
            "success": true,
            "data": {
                "chain_id": "...",
                "root_concept": "...",
                "nodes": [
                    {"concept_id": "...", "depth": 0, "relation": "..."}
                ],
                "edges": [
                    {
                        "source": "...",
                        "target": "...",
                        "relation": "...",
                        "strength": 0.8,
                        "mechanism": "...",
                        "direction": "..."
                    }
                ],
                "root_causes": [...],
                "max_depth": 5,
                "traversal_direction": "root_cause"
            }
        }

        On error: Universal error response with code and message.
    """
    try:
        conn = _get_connection()
        cursor = conn.cursor()

        # Validate concept exists
        cursor.execute("SELECT id FROM concepts WHERE id = ?", (concept_id,))
        if not cursor.fetchone():
            logger.warning(f"Concept not found: {concept_id}")
            return _error_response("CONCEPT_NOT_FOUND", f"Concept {concept_id} does not exist")

        # Determine relation and direction for CTE
        if direction == "root_cause":
            # Traverse backward: find what causes this concept
            relation_match = "causes"
            cte_direction = "backward"
            source_col, target_col = "target", "source"
        elif direction == "consequences":
            # Traverse forward: find what this concept causes
            relation_match = "causes"
            cte_direction = "forward"
            source_col, target_col = "source", "target"
        else:
            return _error_response(
                "INVALID_DIRECTION", f"direction must be 'root_cause' or 'consequences', got {direction}"
            )

        # Build chain_id filter
        chain_filter = ""
        params = [concept_id, concept_id, max_depth]
        if chain_id:
            chain_filter = "AND a.chain_id = ?"
            params.append(chain_id)

        # Recursive CTE with visited-set pattern [M7]
        # Visits concept_id column (source in backward, target in forward traversal)
        cte_query = f"""
        WITH RECURSIVE causal_chain(
            current_id,
            depth,
            visited,
            relation,
            strength,
            mechanism,
            direction_val,
            source_id,
            target_id
        ) AS (
            -- Base case: start from concept
            SELECT
                ?,
                0,
                '|' || ? || '|',
                NULL,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL
            UNION ALL
            -- Recursive case: follow causal edges
            SELECT
                a.{source_col},
                cc.depth + 1,
                cc.visited || a.{source_col} || '|',
                a.relation,
                a.strength,
                a.mechanism,
                a.direction,
                a.source,
                a.target
            FROM associations a
            JOIN causal_chain cc ON a.{target_col} = cc.current_id
            WHERE a.relation = ?
                AND a.direction = ?
                AND cc.visited NOT LIKE '%|' || a.{source_col} || '|%'
                AND cc.depth < ?
                {chain_filter}
        )
        SELECT
            current_id,
            depth,
            relation,
            strength,
            mechanism,
            direction_val,
            source_id,
            target_id
        FROM causal_chain
        ORDER BY depth ASC
        """

        params_cte = [concept_id, concept_id, relation_match, cte_direction, max_depth]
        if chain_id:
            params_cte.append(chain_id)

        cursor.execute(cte_query, params_cte)
        rows = cursor.fetchall()

        if not rows:
            logger.info(f"No causal chain found for {concept_id} in direction {direction}")
            return _success_response(
                {
                    "chain_id": chain_id,
                    "root_concept": concept_id,
                    "nodes": [{"concept_id": concept_id, "depth": 0}],
                    "edges": [],
                    "root_causes": [],
                    "max_depth": max_depth,
                    "traversal_direction": direction,
                }
            )

        # Build nodes and edges from results
        nodes_dict = {concept_id: {"concept_id": concept_id, "depth": 0}}
        edges = []
        root_causes = []

        for row in rows:
            current_id, depth, relation, strength, mechanism, direction_val, source_id, target_id = row

            # Add node
            if current_id not in nodes_dict:
                nodes_dict[current_id] = {"concept_id": current_id, "depth": depth, "relation": relation}

            # Add edge if we have full edge data
            if source_id and target_id:
                edges.append(
                    {
                        "source": source_id,
                        "target": target_id,
                        "relation": relation or "causes",
                        "strength": strength,
                        "mechanism": mechanism,
                        "direction": direction_val or cte_direction,
                    }
                )

            # Track leaf nodes as root causes
            if direction == "root_cause" and depth > 0:
                # For backward traversal, current_id at max depth is a root cause
                if depth == max_depth or not any(e["source"] == current_id for e in edges):
                    if current_id not in root_causes:
                        root_causes.append(current_id)

        nodes = list(nodes_dict.values())

        logger.info(
            f"Traced causal chain for {concept_id}: {len(nodes)} nodes, {len(edges)} edges, direction={direction}"
        )

        return _success_response(
            {
                "chain_id": chain_id,
                "root_concept": concept_id,
                "nodes": nodes,
                "edges": edges,
                "root_causes": root_causes,
                "max_depth": max_depth,
                "traversal_direction": direction,
            }
        )

    except Exception as e:
        logger.error(f"Error tracing causal chain: {e}", exc_info=True)
        return _error_response("TRACE_FAILED", str(e))


def pith_find_path(
    from_concept: str, to_concept: str, max_depth: int = 5, relation_types: list[str] | None = None
) -> dict[str, Any]:
    """
    Find typed shortest path between two concepts using BFS.

    TEMPORAL_RETRIEVAL_SPEC.md §10.4: Path confidence calculation [M6]:
    confidence = min(edge_strengths) × decay^(path_length-1)
    where decay = 0.9 by default.

    Does not infer new knowledge; operates on existing edges only.

    Args:
        from_concept: Starting concept ID
        to_concept: Target concept ID
        max_depth: Maximum path length to explore (default 5)
        relation_types: Optional list of relation types to follow
                       (e.g., ["causes", "related_to"])

    Returns:
        Universal response dict with structure:
        {
            "success": true,
            "data": {
                "from": "...",
                "to": "...",
                "path_found": true,
                "path": [
                    {"concept_id": "...", "step": 0},
                    {"concept_id": "...", "step": 1, "relation": "...", "strength": 0.8}
                ],
                "path_length": 3,
                "confidence": 0.73,
                "edges": [...]
            }
        }

        On error: Universal error response.
    """
    try:
        conn = _get_connection()
        cursor = conn.cursor()

        # Validate concepts exist
        cursor.execute("SELECT id FROM concepts WHERE id IN (?, ?)", (from_concept, to_concept))
        found = {row[0] for row in cursor.fetchall()}

        if from_concept not in found:
            return _error_response("FROM_NOT_FOUND", f"Concept {from_concept} does not exist")
        if to_concept not in found:
            return _error_response("TO_NOT_FOUND", f"Concept {to_concept} does not exist")

        # Same-concept short-circuit: trivial path of length 0
        if from_concept == to_concept:
            return _success_response(
                {
                    "from": from_concept,
                    "to": to_concept,
                    "path_found": True,
                    "path": [{"concept_id": from_concept, "step": 0}],
                    "path_length": 0,
                    "confidence": 1.0,
                    "edges": [],
                }
            )

        # Build relation filter
        relation_filter = ""
        if relation_types:
            placeholders = ",".join("?" * len(relation_types))
            relation_filter = f"AND a.relation IN ({placeholders})"

        # BFS via recursive CTE with beam width (visited-set pattern)
        bfs_query = f"""
        WITH RECURSIVE path_search(
            current_id,
            target_id,
            depth,
            visited,
            path_nodes,
            path_edges,
            min_strength
        ) AS (
            -- Base case: start from from_concept
            SELECT
                ?,
                ?,
                0,
                '|' || ? || '|',
                json_array(?),
                json_array(),
                1.0
            UNION ALL
            -- Recursive case: expand to neighbors
            SELECT
                a.target,
                ps.target_id,
                ps.depth + 1,
                ps.visited || a.target || '|',
                json_insert(ps.path_nodes, '$[#]', a.target),
                json_insert(
                    ps.path_edges,
                    '$[#]',
                    json_object(
                        'source', a.source,
                        'target', a.target,
                        'relation', a.relation,
                        'strength', a.strength
                    )
                ),
                MIN(ps.min_strength, COALESCE(a.strength, 1.0))
            FROM associations a
            JOIN path_search ps ON a.source = ps.current_id
            WHERE ps.visited NOT LIKE '%|' || a.target || '|%'
                AND ps.depth < ?
                {relation_filter}
                AND a.target != ps.target_id  -- Avoid immediate cycles
        )
        SELECT
            current_id,
            depth,
            path_nodes,
            path_edges,
            min_strength
        FROM path_search
        WHERE current_id = ?
        ORDER BY depth ASC
        LIMIT 1
        """

        params_bfs = [from_concept, to_concept, from_concept, from_concept, max_depth]
        if relation_types:
            params_bfs.extend(relation_types)
        params_bfs.append(to_concept)

        cursor.execute(bfs_query, params_bfs)
        result = cursor.fetchone()

        if not result:
            logger.info(f"No path found from {from_concept} to {to_concept}")
            return _success_response(
                {
                    "from": from_concept,
                    "to": to_concept,
                    "path_found": False,
                    "path": [{"concept_id": from_concept, "step": 0}],
                    "path_length": 0,
                    "confidence": 0.0,
                    "edges": [],
                }
            )

        current_id, depth, path_nodes_json, path_edges_json, min_strength = result

        # Parse JSON arrays (SQLite json_array returns strings)
        import json

        path_node_ids = json.loads(path_nodes_json) if isinstance(path_nodes_json, str) else path_nodes_json
        path_edges_list = json.loads(path_edges_json) if isinstance(path_edges_json, str) else path_edges_json

        # Calculate confidence: min(edge_strengths) × decay^(path_length-1) [M6]
        decay = 0.9
        if path_edges_list:
            min_strength = min(e.get("strength", 1.0) for e in path_edges_list)
            confidence = min_strength * (decay ** (depth - 1))
        else:
            confidence = 1.0 if depth == 0 else 0.0

        # Build path with steps
        path = [{"concept_id": from_concept, "step": 0}]
        for i, node_id in enumerate(path_node_ids[1:], 1):
            step_data = {"concept_id": node_id, "step": i}
            if i - 1 < len(path_edges_list):
                edge = path_edges_list[i - 1]
                step_data["relation"] = edge.get("relation")
                step_data["strength"] = edge.get("strength")
            path.append(step_data)

        logger.info(f"Found path from {from_concept} to {to_concept}: length={depth}, confidence={confidence:.3f}")

        return _success_response(
            {
                "from": from_concept,
                "to": to_concept,
                "path_found": True,
                "path": path,
                "path_length": depth,
                "confidence": confidence,
                "edges": path_edges_list,
            }
        )

    except Exception as e:
        logger.error(f"Error finding path: {e}", exc_info=True)
        return _error_response("PATH_SEARCH_FAILED", str(e))


def create_causal_edge(
    source: str,
    target: str,
    relation: str = "causes",
    mechanism: str | None = None,
    direction: str = "forward",
    chain_id: str | None = None,
    strength: float = 1.0,
) -> dict[str, Any]:
    """
    Create a directed causal edge in the associations table.

    TEMPORAL_RETRIEVAL_SPEC.md §10.1: Mechanism quality gate [M16]:
    Auto-detected mechanisms must be min 20 chars and confidence >0.5.
    Below threshold: mechanism=NULL and needs_review flag set.

    Args:
        source: Source concept ID
        target: Target concept ID
        relation: Relation type (default "causes")
        mechanism: Optional explanation of causation
        direction: "forward" (source → target) or "backward" (target → source)
        chain_id: Optional identifier linking to a causal chain
        strength: Edge weight 0.0-1.0 (default 1.0)

    Returns:
        Universal response dict with:
        {
            "success": true,
            "data": {
                "edge_id": "(source, target, relation)",
                "source": "...",
                "target": "...",
                "relation": "...",
                "strength": 0.8,
                "mechanism": "...",
                "mechanism_needs_review": false,
                "created_at": "2026-02-25T..."
            }
        }

        On error: Universal error response.
    """
    try:
        conn = _get_connection()
        cursor = conn.cursor()

        # Validate concepts exist
        cursor.execute("SELECT id FROM concepts WHERE id IN (?, ?)", (source, target))
        found = {row[0] for row in cursor.fetchall()}

        if source not in found:
            return _error_response("SOURCE_NOT_FOUND", f"Source concept {source} does not exist")
        if target not in found:
            return _error_response("TARGET_NOT_FOUND", f"Target concept {target} does not exist")

        if source == target:
            return _error_response("SELF_EDGE", "Source and target cannot be the same concept")

        # Validate strength
        if not (0.0 <= strength <= 1.0):
            return _error_response("INVALID_STRENGTH", f"strength must be between 0.0 and 1.0, got {strength}")

        # Validate direction
        if direction not in ("forward", "backward"):
            return _error_response("INVALID_DIRECTION", f"direction must be 'forward' or 'backward', got {direction}")

        # Mechanism quality gate [M16]
        needs_review = False
        mechanism_final = mechanism

        if mechanism:
            if len(mechanism) < 20:
                logger.warning(f"Mechanism too short ({len(mechanism)} chars): {mechanism[:30]}...")
                mechanism_final = None
                needs_review = True

        # Current timestamp in ISO format
        created_at = datetime.now(UTC).isoformat()

        # Insert or replace edge
        cursor.execute(
            """
            INSERT OR REPLACE INTO associations
            (source, target, relation, strength, mechanism, direction, chain_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (source, target, relation, strength, mechanism_final, direction, chain_id, created_at),
        )
        conn.commit()

        logger.info(
            f"Created causal edge {source} -> {target} (relation={relation}, "
            f"strength={strength}, direction={direction})"
        )

        return _success_response(
            {
                "edge_id": f"({source}, {target}, {relation})",
                "source": source,
                "target": target,
                "relation": relation,
                "strength": strength,
                "mechanism": mechanism_final,
                "mechanism_needs_review": needs_review,
                "created_at": created_at,
            }
        )

    except Exception as e:
        logger.error(f"Error creating causal edge: {e}", exc_info=True)
        return _error_response("EDGE_CREATE_FAILED", str(e))
