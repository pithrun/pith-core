"""Embedding engine for semantic search using sentence-transformers.

Provides L2-normalized 384-dim embeddings via all-MiniLM-L6-v2 with lazy
model loading. In-memory index uses dot-product search (equivalent to cosine
on L2-normalized vectors).

Used by retrieval.py for semantic search. TF-IDF remains for dedup/auto-association.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Current embedding version — bump when changing model or preprocessing
EMBEDDING_VERSION = 1
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class EmbeddingUnavailableError(RuntimeError):
    """Raised when embedding model cannot be loaded (sentence_transformers missing)."""

    pass


# --- Module-level availability check (runs once at import) ---
_EMBEDDING_AVAILABLE: bool = False
_EMBEDDING_UNAVAILABLE_LOGGED: bool = False


def _check_embedding_availability() -> bool:
    """Check if sentence_transformers can be imported. Runs once."""
    try:
        import sentence_transformers  # noqa: F401

        return True
    except ImportError:
        return False


_EMBEDDING_AVAILABLE = _check_embedding_availability()


class EmbeddingEngine:
    """Manages sentence embeddings and in-memory search index.

    Lazy-loads the model on first use to avoid startup cost when embeddings
    aren't needed (e.g., dedup-only paths).
    """

    def __init__(self):
        self._model = None
        self._index_ids: list[str] = []  # concept_id at each position
        self._index_matrix: np.ndarray | None = None  # (N, 384) L2-normalized
        self._id_to_pos: dict = {}  # concept_id -> row position

    @property
    def is_model_loaded(self) -> bool:
        return self._model is not None

    @property
    def is_available(self) -> bool:
        """Whether embedding features are available (sentence_transformers installed)."""
        return _EMBEDDING_AVAILABLE

    @property
    def index_size(self) -> int:
        return len(self._index_ids)

    def load_model(self):
        """Load the sentence-transformer model. Called lazily on first embed."""
        global _EMBEDDING_UNAVAILABLE_LOGGED
        if self._model is not None:
            return
        if not _EMBEDDING_AVAILABLE:
            if not _EMBEDDING_UNAVAILABLE_LOGGED:
                logger.warning(
                    "sentence_transformers not available — embedding features disabled. "
                    "TF-IDF search will be used as fallback."
                )
                _EMBEDDING_UNAVAILABLE_LOGGED = True
            raise EmbeddingUnavailableError("sentence_transformers not installed")
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model: {MODEL_NAME}")
        self._model = SentenceTransformer(MODEL_NAME)
        logger.info(f"Embedding model loaded: {MODEL_NAME} ({EMBEDDING_DIM}-dim)")

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string. Returns L2-normalized 384-dim vector."""
        self.load_model()
        vec = self._model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return vec.astype(np.float32)

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed multiple texts. Returns L2-normalized (N, 384) matrix."""
        self.load_model()
        vecs = self._model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
        return vecs.astype(np.float32)

    def build_index(self, concept_ids: list[str], embeddings: np.ndarray):
        """Populate the in-memory search index from pre-computed embeddings.

        Args:
            concept_ids: List of concept IDs matching embedding rows.
            embeddings: (N, 384) L2-normalized matrix.
        """
        if len(concept_ids) != embeddings.shape[0]:
            raise ValueError(f"Mismatch: {len(concept_ids)} IDs vs {embeddings.shape[0]} embeddings")
        self._index_ids = list(concept_ids)
        self._index_matrix = embeddings.astype(np.float32)
        self._id_to_pos = {cid: i for i, cid in enumerate(self._index_ids)}
        logger.info(f"Embedding index built: {len(self._index_ids)} concepts")

    def search(self, query_text: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Search for most similar concepts by cosine similarity.

        Uses dot product on L2-normalized vectors (= cosine similarity).

        Returns:
            List of (concept_id, similarity_score) tuples, descending by score.
        """
        if self._index_matrix is None or len(self._index_ids) == 0:
            return []

        query_vec = self.embed_text(query_text)  # (384,)
        # Dot product = cosine similarity for L2-normalized vectors
        scores = self._index_matrix @ query_vec  # (N,)

        # Get top-k indices
        if len(scores) <= top_k:
            top_indices = np.argsort(scores)[::-1]
        else:
            # Partial sort for efficiency on large indices
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = []
        for idx in top_indices:
            results.append((self._index_ids[idx], float(scores[idx])))

        # RETRIEVAL-037b v4.2: Deterministic tiebreaker.
        # When multiple concepts share identical cosine similarity (common in
        # benchmark brains where many facts have similar embeddings), sort by
        # concept_id as secondary key. Without this, argpartition returns ties
        # in undefined order that depends on index matrix row positions, causing
        # ±4% benchmark noise between server restarts.
        results.sort(key=lambda r: (-r[1], r[0]))

        return results

    def add_embedding(self, concept_id: str, embedding: np.ndarray):
        """Add a new embedding to the index. If concept already exists, updates it."""
        if concept_id in self._id_to_pos:
            self.update_embedding(concept_id, embedding)
            return

        embedding = embedding.astype(np.float32).reshape(1, -1)
        if self._index_matrix is None:
            self._index_matrix = embedding
        else:
            self._index_matrix = np.vstack([self._index_matrix, embedding])

        self._index_ids.append(concept_id)
        self._id_to_pos[concept_id] = len(self._index_ids) - 1

    def update_embedding(self, concept_id: str, embedding: np.ndarray):
        """Replace the embedding for an existing concept. Adds if not present."""
        if concept_id not in self._id_to_pos:
            self.add_embedding(concept_id, embedding)
            return

        pos = self._id_to_pos[concept_id]
        self._index_matrix[pos] = embedding.astype(np.float32)

    def remove_embedding(self, concept_id: str):
        """Remove a concept from the index using O(1) swap-and-pop."""
        if concept_id not in self._id_to_pos:
            return

        pos = self._id_to_pos[concept_id]
        last_pos = len(self._index_ids) - 1

        if pos != last_pos:
            # Swap with last element
            last_id = self._index_ids[last_pos]
            self._index_matrix[pos] = self._index_matrix[last_pos]
            self._index_ids[pos] = last_id
            self._id_to_pos[last_id] = pos

        # Pop the last element
        self._index_ids.pop()
        del self._id_to_pos[concept_id]

        if len(self._index_ids) == 0:
            self._index_matrix = None
        else:
            self._index_matrix = self._index_matrix[: len(self._index_ids)]


# Module-level singleton
embedding_engine = EmbeddingEngine()
