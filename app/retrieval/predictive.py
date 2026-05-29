"""Predictive activation for context-aware concept pre-loading."""

from collections import defaultdict

from app.core.datetime_utils import _ensure_aware, _utc_now, _utc_now_iso
from app.storage import get_related_concepts, load_concept


class ActivationNode:
    """Represents an activated concept with spreading activation."""

    def __init__(self, concept_id: str, activation: float, source: str = "direct"):
        self.concept_id = concept_id
        self.activation = activation
        self.source = source  # "direct", "spread", "context", "goal"
        self.timestamp = _utc_now()


class PredictiveActivation:
    """
    Manages spreading activation across concept graph.

    Similar to how human memory works - when you think of one concept,
    related concepts become "primed" for faster retrieval.
    """

    def __init__(self):
        self.active_concepts: dict[str, ActivationNode] = {}
        self.activation_history: list[dict] = []
        self.decay_rate = 0.1  # Activation decay per minute
        self.spread_factor = 0.7  # How much activation spreads to neighbors
        self.max_activation = 1.0
        self.min_activation = 0.01  # Below this, remove from active set

    def activate_concept(self, concept_id: str, activation: float = 1.0, source: str = "direct"):
        """
        Directly activate a concept.

        Args:
            concept_id: Concept to activate
            activation: Initial activation level (0.0 to 1.0)
            source: Source of activation (direct, spread, context, goal)
        """
        # Cap activation at max
        activation = min(activation, self.max_activation)

        # Add or update activation
        if concept_id in self.active_concepts:
            # Boost existing activation
            current = self.active_concepts[concept_id]
            new_activation = min(self.max_activation, current.activation + activation)
            self.active_concepts[concept_id].activation = new_activation
            self.active_concepts[concept_id].timestamp = _utc_now()
        else:
            # New activation
            self.active_concepts[concept_id] = ActivationNode(concept_id, activation, source)

        # Record in history
        self.activation_history.append(
            {"concept_id": concept_id, "activation": activation, "source": source, "timestamp": _utc_now_iso()}
        )

        # Spread activation to related concepts
        self._spread_activation(concept_id, activation)

    def _spread_activation(self, concept_id: str, activation: float):
        """Spread activation to related concepts (PERF-023).

        Uses get_related_concepts() which now hits the cached adjacency graph
        (<0.01ms) instead of doing a full DB scan (37ms) per call.

        Removed dead load_associations() call: graph["concept_id"] lookup always
        failed (graph keys are "associations"/"metadata", not concept IDs), so
        strengths always defaulted to 0.5 anyway. Now uses 0.5 directly.
        """
        related = get_related_concepts(concept_id, max_depth=1)

        if not related:
            return

        spread_amount = activation * self.spread_factor

        for related_id in related:
            # Uniform strength 0.5 (was the effective default due to dead code above)
            spread_value = spread_amount * 0.5

            if spread_value >= self.min_activation:
                self.activate_concept(related_id, spread_value, source="spread")

    def activate_from_context(self, context: str, boost: float = 0.5):
        """
        Activate concepts based on context text.

        Uses embedding similarity search (O(K)) instead of brute-force
        word overlap + signal scan across all concepts (O(N)). For the
        top-K results, also checks signal overlap to preserve that feature.

        Args:
            context: Current context (e.g., recent conversation)
            boost: Activation level for context matches
        """
        from app.storage.embedding import embedding_engine

        if embedding_engine.is_available and embedding_engine.index_size > 0:
            # O(K) path: embedding search + signal check on top results only
            context_lower = context.lower()
            raw_results = embedding_engine.search(context, top_k=30)
            for concept_id, emb_score in raw_results:
                if emb_score > 0.20:
                    activation = min(boost, emb_score * 0.5)
                    self.activate_concept(concept_id, activation, source="context")

                    # Check signals on top-K results (preserves signal matching
                    # without the O(N) scan)
                    concept = load_concept(concept_id, track_access=False)
                    if concept:
                        for signal in concept.signals:
                            if signal.lower() in context_lower:
                                self.activate_concept(concept_id, boost * 0.5, source="context")
                                break  # One signal match is enough
        # If embeddings unavailable, skip context activation rather than O(N) scan

    def decay_activation(self):
        """
        Apply time-based decay to all activations.

        Called periodically to simulate memory fade.
        """
        to_remove = []

        for concept_id, node in self.active_concepts.items():
            # Calculate time since last activation
            elapsed = (_utc_now() - _ensure_aware(node.timestamp)).total_seconds() / 60.0

            # Apply exponential decay
            decay = self.decay_rate * elapsed
            new_activation = node.activation * (1 - decay)

            if new_activation < self.min_activation:
                # Activation too low, remove
                to_remove.append(concept_id)
            else:
                # Update activation
                node.activation = new_activation

        # Remove concepts with minimal activation
        for concept_id in to_remove:
            del self.active_concepts[concept_id]

    def get_active_concepts(self, min_activation: float = 0.1, max_results: int = 20) -> list[tuple]:
        """
        Get currently active concepts.

        Returns:
            List of (concept_id, activation) tuples, sorted by activation
        """
        # Apply decay first
        self.decay_activation()

        # Filter by minimum activation
        active = [
            (concept_id, node.activation)
            for concept_id, node in self.active_concepts.items()
            if node.activation >= min_activation
        ]

        # Sort by activation (descending)
        active.sort(key=lambda x: x[1], reverse=True)

        return active[:max_results]

    def boost_retrieval_scores(self, concept_scores: list[tuple], boost_weight: float = 0.3) -> list[tuple]:
        """
        Boost retrieval scores for active concepts.

        Args:
            concept_scores: List of (concept_id, score) tuples
            boost_weight: Weight of activation boost (0.0 to 1.0)

        Returns:
            List of (concept_id, boosted_score) tuples
        """
        # Apply decay first
        self.decay_activation()

        boosted = []

        for concept_id, base_score in concept_scores:
            # Check if concept is active
            activation = 0.0
            if concept_id in self.active_concepts:
                activation = self.active_concepts[concept_id].activation

            # Boost score by activation
            boost = activation * boost_weight
            final_score = base_score + boost

            boosted.append((concept_id, min(1.0, final_score)))

        # Re-sort by boosted score
        boosted.sort(key=lambda x: x[1], reverse=True)

        return boosted

    def predict_next_concepts(self, current_concepts: list[str], num_predictions: int = 5) -> list[str]:
        """
        Predict which concepts are likely to be needed next.

        Based on:
        1. What's currently active
        2. Association patterns
        3. Access history

        Args:
            current_concepts: Currently accessed/displayed concepts
            num_predictions: How many predictions to return

        Returns:
            List of predicted concept IDs
        """
        predictions = defaultdict(float)

        # Score based on current activations
        for concept_id in current_concepts:
            # Get strongly related concepts
            related = get_related_concepts(concept_id, max_depth=1)

            for related_id in related:
                # Check if already active (higher score)
                if related_id in self.active_concepts:
                    predictions[related_id] += self.active_concepts[related_id].activation
                else:
                    predictions[related_id] += 0.3

        # Add highly active concepts not yet accessed
        for concept_id, node in self.active_concepts.items():
            if concept_id not in current_concepts:
                predictions[concept_id] += node.activation * 0.5

        # Sort by prediction score
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

        return [cid for cid, score in sorted_predictions[:num_predictions]]

    def preload_for_query(self, query: str, context: str | None = None):
        """Pre-activate concepts based on upcoming query (PERF-023).

        Direct cache priming without recursive _spread_activation.
        Uses embedding similarity search to find top-K relevant concepts,
        then inserts them directly into the activation cache. This avoids
        the O(K × degree^depth) cascade that was generating ~20-50 wasted
        get_related_concepts calls (each loading all 25K+ edges).

        Spreading is intentionally skipped here because:
        1. preload is a priming hint, not a full activation
        2. search() phase1 (embedding search) already finds the same
           neighbors that spreading would surface
        3. boost_retrieval_scores() still uses the primed cache for scoring

        Args:
            query: The search query
            context: Optional context (recent conversation, etc.)
        """
        from app.storage.embedding import embedding_engine

        if not (embedding_engine.is_available and embedding_engine.index_size > 0):
            return  # No embeddings — skip preload. search works without it.

        # Phase 1: Prime cache from query embedding hits (no spreading)
        raw_results = embedding_engine.search(query, top_k=20)
        for concept_id, emb_score in raw_results:
            if emb_score > 0.25:
                activation = min(0.8, emb_score)
                # Direct cache insertion — bypasses _spread_activation entirely
                if concept_id in self.active_concepts:
                    node = self.active_concepts[concept_id]
                    node.activation = min(self.max_activation, node.activation + activation)
                    node.timestamp = _utc_now()
                else:
                    self.active_concepts[concept_id] = ActivationNode(
                        concept_id, activation, source="query"
                    )

        # Phase 2: Prime cache from context embedding hits (if provided)
        # Also skips spreading — same rationale as Phase 1.
        context_results = []  # F10: init before guard to prevent NameError in history append
        if context:
            context_results = embedding_engine.search(context, top_k=15)
            for concept_id, emb_score in context_results:
                if emb_score > 0.20:
                    activation = min(0.4, emb_score * 0.5)
                    if concept_id in self.active_concepts:
                        node = self.active_concepts[concept_id]
                        node.activation = min(self.max_activation, node.activation + activation)
                        node.timestamp = _utc_now()
                    else:
                        self.active_concepts[concept_id] = ActivationNode(
                            concept_id, activation, source="context"
                        )

        # Record history (batch, not per-concept — avoids O(K) history bloat)
        self.activation_history.append({
            "source": "preload_for_query",
            "query_hits": len([1 for _, s in raw_results if s > 0.25]),
            "context_hits": len([1 for _, s in context_results if s > 0.20]),
            "timestamp": _utc_now_iso(),
        })

    def get_activation_state(self) -> dict:
        """Get current activation state for debugging/monitoring."""
        self.decay_activation()

        return {
            "active_count": len(self.active_concepts),
            "active_concepts": [
                {
                    "concept_id": concept_id,
                    "activation": node.activation,
                    "source": node.source,
                    "age_seconds": (_utc_now() - _ensure_aware(node.timestamp)).total_seconds(),
                }
                for concept_id, node in self.active_concepts.items()
            ],
            "history_size": len(self.activation_history),
            "avg_activation": sum(node.activation for node in self.active_concepts.values()) / len(self.active_concepts)
            if self.active_concepts
            else 0.0,
        }

    def reset(self):
        """Clear all activations (useful for new context/session)."""
        self.active_concepts.clear()
        # Keep history for analysis


# Global instance
predictive_activation = PredictiveActivation()
