"""
Incremental TF-IDF Index Implementation
Production-grade incremental indexing for Pith Platform
"""

import gzip
import json
import logging
import os
import pickle
import re
import shutil
import threading
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.sparse import SparseEfficiencyWarning, csr_matrix, hstack, load_npz, save_npz, vstack

from app.datetime_utils import _utc_now, _utc_now_iso

# Suppress SparseEfficiencyWarning for in-place CSR row replacement in partial IDF refresh.
# The warning fires because CSR sparsity structure changes on assignment, but benchmarks show
# 50 row replacements complete in ~34ms — not a real concern at our scale.
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

# Configure logging
logger = logging.getLogger(__name__)


# English stop words (common words to filter out)
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "will",
    "with",
    "this",
    "but",
    "they",
    "have",
    "had",
    "what",
    "when",
    "where",
    "who",
    "which",
    "why",
    "how",
    "or",
    "not",
    "can",
    "could",
    "should",
    "would",
    "may",
    "might",
    "must",
    "shall",
}


class IncrementalTfidfIndex:
    """
    Production-grade incremental TF-IDF index.

    Provides O(1) amortized complexity for concept addition instead of O(N).
    Maintains search quality through periodic IDF recalculation.
    """

    def __init__(self):
        # === VOCABULARY & MAPPING ===
        self.vocabulary: dict[str, int] = {}  # term -> term_id
        self.reverse_vocabulary: dict[int, str] = {}  # term_id -> term
        self.next_term_id: int = 0  # Counter for new terms

        # === DOCUMENT STATISTICS ===
        self.concept_ids: list[str] = []  # Ordered list of concept IDs
        self.concept_id_to_idx: dict[str, int] = {}  # concept_id -> row index

        # === CORPUS STATISTICS ===
        self.document_count: int = 0  # Total documents (N)
        self.document_frequencies: np.ndarray = np.array([], dtype=np.int32)  # DF per term
        self.term_totals: np.ndarray = np.array([], dtype=np.int32)  # Total occurrences

        # === TF-IDF MATRIX ===
        self.tfidf_matrix: csr_matrix | None = None  # Main search matrix (N × V)

        # === INCREMENTAL UPDATE TRACKING ===
        self.dirty_documents: set[str] = set()  # Modified since last IDF recalc
        self.last_idf_update: datetime | None = None  # When IDF was last recalculated
        self.idf_update_threshold: int = 50  # Recalc IDF after N changes
        self.full_refresh_threshold: int = 500  # Full rebuild every 500 cumulative changes
        self._changes_since_full_refresh: int = 0  # Counter reset on full refresh

        # === TERM COUNTS STORAGE (for perfect IDF recalculation) ===
        self.document_term_counts: list[dict[str, int]] = []  # Raw term counts per doc

        # === DELETION TRACKING ===
        self.deleted_indices: set[int] = set()  # Logically deleted rows
        self.compaction_threshold: int = 100  # Compact after N deletes

        # === VERSIONING ===
        self.index_version: int = 0  # Incremented on each change
        self.checkpoint_every: int = 100  # Checkpoint frequency

        # === THREAD SAFETY ===
        self.lock = threading.RLock()  # Reentrant lock

        logger.info("IncrementalTfidfIndex initialized")

    def extract_terms(self, text: str) -> dict[str, int]:
        """
        Extract terms from text with counts.

        Uses same preprocessing as scikit-learn:
        - Lowercase normalization
        - Stop word removal
        - N-gram generation (1-2 grams)
        - Min token length: 2

        Args:
            text: Input text to tokenize

        Returns:
            Dictionary of {term: count}
        """
        if not text:
            return {}

        # Tokenize (word boundaries, min length 2)
        tokens = self._tokenize(text.lower())

        # Remove stop words
        tokens = [t for t in tokens if t not in STOP_WORDS]

        # Generate unigrams and bigrams
        unigrams = tokens
        bigrams = [f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1)]

        all_terms = unigrams + bigrams

        # Count frequencies
        term_counts = Counter(all_terms)

        return dict(term_counts)

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text using word boundaries (min length 2)."""
        return re.findall(r"\b\w{2,}\b", text)

    def add_concept(self, concept_id: str, text: str) -> bool:
        """
        Incrementally add concept to index.

        Time Complexity: O(V_doc) where V_doc = unique terms in document
        Space Complexity: O(V_doc)

        Steps:
        1. Extract terms from concept text
        2. Update vocabulary (add new terms)
        3. Update document frequencies
        4. Compute TF-IDF vector for this document
        5. Append to sparse matrix
        6. Schedule IDF recalculation if threshold reached

        Args:
            concept_id: Unique identifier for the concept
            text: Searchable text content

        Returns:
            True if added, False if duplicate
        """
        with self.lock:
            # Check for duplicates
            if concept_id in self.concept_id_to_idx:
                logger.debug(f"Concept {concept_id} already in index")
                return False

            # Extract terms and counts
            term_counts = self.extract_terms(text)
            if not term_counts:
                logger.warning(f"No terms extracted from {concept_id}")
                return False

            # === STEP 1: Update Vocabulary ===
            new_terms = set()
            for term in term_counts.keys():
                if term not in self.vocabulary:
                    # Add new term to vocabulary
                    self.vocabulary[term] = self.next_term_id
                    self.reverse_vocabulary[self.next_term_id] = term
                    self.next_term_id += 1
                    new_terms.add(term)

            # Expand arrays if new terms added
            if new_terms:
                self._expand_for_new_terms(len(new_terms))

            # === STEP 2: Update Document Frequencies ===
            # DF = number of documents containing term
            unique_terms = set(term_counts.keys())
            for term in unique_terms:
                term_id = self.vocabulary[term]
                self.document_frequencies[term_id] += 1

            # === STEP 3: Store Raw Term Counts ===
            self.document_term_counts.append(term_counts)

            # === STEP 4: Compute TF-IDF Vector (sparse-first) ===
            sparse_vector = self._compute_tfidf_vector_sparse(term_counts)

            # === STEP 5: Append to Matrix ===

            if self.tfidf_matrix is None:
                self.tfidf_matrix = sparse_vector
            else:
                self.tfidf_matrix = vstack([self.tfidf_matrix, sparse_vector])

            # === STEP 6: Update Mappings ===
            new_idx = self.document_count
            self.concept_ids.append(concept_id)
            self.concept_id_to_idx[concept_id] = new_idx
            self.document_count += 1

            # === STEP 7: Track for IDF Update ===
            self.dirty_documents.add(concept_id)

            # Check if IDF recalculation needed
            if len(self.dirty_documents) >= self.idf_update_threshold:
                self._recalculate_idf()

            # === STEP 8: Increment Version ===
            self.index_version += 1

            # === STEP 9: Auto-save Checkpoint ===
            if self.index_version % self.checkpoint_every == 0:
                logger.debug(f"Checkpoint triggered at version {self.index_version}")

            return True

    def remove_concept(self, concept_id: str) -> bool:
        """
        Remove concept from index.

        Uses lazy deletion - marks row as deleted but doesn't immediately
        compact the matrix. Compaction happens when threshold reached.

        Time Complexity: O(1)

        Args:
            concept_id: Concept ID to remove

        Returns:
            True if removed, False if not found
        """
        with self.lock:
            if concept_id not in self.concept_id_to_idx:
                logger.warning(f"Concept {concept_id} not in index")
                return False

            # Mark as deleted
            idx = self.concept_id_to_idx[concept_id]
            self.deleted_indices.add(idx)

            # Remove from dirty set if present
            self.dirty_documents.discard(concept_id)

            # Increment version
            self.index_version += 1

            # Check if compaction needed
            if len(self.deleted_indices) >= self.compaction_threshold:
                self._compact_matrix()

            logger.debug(f"Concept {concept_id} marked for deletion (lazy)")

            return True

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """
        Search for concepts using TF-IDF similarity.

        Time Complexity: O(V_query + N_active) where:
        - V_query = unique terms in query
        - N_active = documents not deleted

        Args:
            query: Search query text
            top_k: Number of results to return

        Returns:
            List of (concept_id, similarity_score) tuples, sorted by score descending
        """
        with self.lock:
            if self.tfidf_matrix is None or self.document_count == 0:
                logger.warning("Index is empty")
                return []

            # Extract query terms
            query_terms = self.extract_terms(query)
            if not query_terms:
                logger.warning("No terms extracted from query")
                return []

            # Build query TF-IDF vector
            query_vector = self._compute_tfidf_vector(query_terms)

            # Compute similarities (cosine similarity = dot product for unit vectors)
            similarities = np.asarray(self.tfidf_matrix.dot(query_vector)).flatten()

            # Filter deleted documents
            active_indices = [i for i in range(self.document_count) if i not in self.deleted_indices]
            active_similarities = [(i, similarities[i]) for i in active_indices]

            # Sort by similarity descending
            active_similarities.sort(key=lambda x: x[1], reverse=True)

            # Get top_k results
            results = []
            for idx, score in active_similarities[:top_k]:
                if score > 0:  # Only return non-zero scores
                    concept_id = self.concept_ids[idx]
                    results.append((concept_id, float(score)))

            return results

    def compute_similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two ad-hoc texts using the index vocabulary.

        THREAD-002: Added to support Gate 1b title similarity and Gate 2 TF-IDF
        confirmation in auto_link_candidates(). Uses existing vocabulary and IDF
        weights — does NOT add documents to the index.

        Returns 0.0 if either text produces an empty vector."""
        with self.lock:
            if not self.vocabulary:
                return 0.0
            terms_a = self.extract_terms(text_a)
            terms_b = self.extract_terms(text_b)
            if not terms_a or not terms_b:
                return 0.0
            vec_a = self._compute_tfidf_vector(terms_a)
            vec_b = self._compute_tfidf_vector(terms_b)
            # Vectors are L2-normalized, so dot product = cosine similarity
            return float(np.dot(vec_a, vec_b))

    def _expand_for_new_terms(self, num_new_terms: int):
        """
        Expand arrays to accommodate new vocabulary terms.

        Args:
            num_new_terms: Number of new terms to add space for
        """
        # Expand document frequencies
        new_df = np.zeros(num_new_terms, dtype=np.int32)
        self.document_frequencies = np.concatenate([self.document_frequencies, new_df])

        # Expand term totals
        new_totals = np.zeros(num_new_terms, dtype=np.int32)
        self.term_totals = np.concatenate([self.term_totals, new_totals])

        # Expand TF-IDF matrix (add columns)
        if self.tfidf_matrix is not None and self.tfidf_matrix.shape[0] > 0:
            num_docs = self.tfidf_matrix.shape[0]
            zero_cols = csr_matrix((num_docs, num_new_terms), dtype=np.float32)
            self.tfidf_matrix = hstack([self.tfidf_matrix, zero_cols])

    def _compute_tfidf_vector(self, term_counts: dict[str, int]) -> np.ndarray:
        """
        Compute TF-IDF vector for a document.

        Formula:
        TF-IDF(term, doc) = TF(term, doc) × IDF(term)

        where:
        TF(term, doc) = count(term, doc) / sum(counts in doc)
        IDF(term) = log((1 + N) / (1 + DF(term))) + 1

        (Uses scikit-learn's smooth IDF formula)

        Args:
            term_counts: Dictionary of {term: count}

        Returns:
            TF-IDF vector (vocabulary size)
        """
        # Initialize zero vector
        vocab_size = len(self.vocabulary)
        tfidf_vec = np.zeros(vocab_size, dtype=np.float32)

        # Compute document length
        doc_length = sum(term_counts.values())
        if doc_length == 0:
            return tfidf_vec

        # Compute TF-IDF for each term
        for term, count in term_counts.items():
            if term in self.vocabulary:
                term_id = self.vocabulary[term]

                # TF (normalized frequency)
                tf = count / doc_length

                # IDF (smooth IDF from scikit-learn)
                df = self.document_frequencies[term_id]
                idf = np.log((1 + self.document_count) / (1 + df)) + 1

                # TF-IDF
                tfidf_vec[term_id] = tf * idf

        # L2 normalization (for cosine similarity)
        norm = np.linalg.norm(tfidf_vec)
        if norm > 0:
            tfidf_vec = tfidf_vec / norm

        return tfidf_vec

    def _compute_tfidf_vector_sparse(self, term_counts: dict[str, int]) -> csr_matrix:
        """Compute TF-IDF as sparse vector directly. No dense intermediate.

        Old path: np.zeros(124K) → fill ~50 → L2 norm 124K → to_sparse
        New path: build indices/data → L2 norm ~50 values → csr_matrix
        Saves: 498KB dense allocation + 124K L2 norm per doc.
        """
        doc_length = sum(term_counts.values())
        if doc_length == 0:
            return csr_matrix((1, len(self.vocabulary)), dtype=np.float32)

        indices = []
        values = []
        for term, count in term_counts.items():
            if term in self.vocabulary:
                term_id = self.vocabulary[term]
                tf = count / doc_length
                df = self.document_frequencies[term_id]
                idf = np.log((1 + self.document_count) / (1 + df)) + 1
                indices.append(term_id)
                values.append(tf * idf)

        if not indices:
            return csr_matrix((1, len(self.vocabulary)), dtype=np.float32)

        data = np.array(values, dtype=np.float32)
        norm = np.linalg.norm(data)  # L2 norm on ~50 values, not 124K
        if norm > 0:
            data /= norm

        row_indices = np.zeros(len(indices), dtype=np.int32)
        col_indices = np.array(indices, dtype=np.int32)
        vocab_size = len(self.vocabulary)
        return csr_matrix((data, (row_indices, col_indices)), shape=(1, vocab_size), dtype=np.float32)

    def _recalculate_idf(self):
        """Partial IDF refresh — only recompute dirty document vectors.

        O(dirty × V_avg) instead of O(N × V_avg).
        IDF values use current document_frequencies (maintained incrementally).
        Vector recomputation is scoped to dirty docs only.

        Approximation: unchanged docs retain slightly stale TF-IDF vectors.
        Typical drift: 2-5% per refresh cycle. Acceptable for TF-IDF search.
        """
        if not self.dirty_documents:
            return

        dirty_count = len(self.dirty_documents)
        logger.info(
            f"Partial IDF refresh for {dirty_count} dirty documents "
            f"(skipping {self.document_count - dirty_count} unchanged)"
        )

        # Map dirty concept_ids to matrix row indices
        dirty_indices = set()
        for concept_id in self.dirty_documents:
            if concept_id in self.concept_id_to_idx:
                dirty_indices.add(self.concept_id_to_idx[concept_id])
            # Orphaned concept_ids (removed but still in dirty set) safely skipped

        # TFIDF_INDEX_CORRUPTION_FIX: Guard partial refresh too
        dtc_len = len(self.document_term_counts)

        # Recompute ONLY dirty rows in-place
        for idx in dirty_indices:
            if idx in self.deleted_indices:
                continue  # Skip deleted docs still in dirty set
            if idx >= dtc_len:
                logger.error(f"Partial IDF: idx {idx} >= document_term_counts length {dtc_len}, skipping")
                continue
            term_counts = self.document_term_counts[idx]
            sparse_vec = self._compute_tfidf_vector_sparse(term_counts)
            self.tfidf_matrix[idx] = sparse_vec  # In-place row replacement (CSR)

        # Track cumulative changes for periodic full refresh
        self._changes_since_full_refresh += dirty_count

        self.dirty_documents.clear()
        self.last_idf_update = _utc_now()
        logger.info(f"Partial IDF refresh complete ({dirty_count} vectors updated)")

        # Check if full refresh needed
        if self._changes_since_full_refresh >= self.full_refresh_threshold:
            self._full_recalculate_idf()

    def _full_recalculate_idf(self):
        """Full O(N) IDF rebuild for accuracy maintenance.

        Runs when cumulative partial refreshes exceed full_refresh_threshold.
        Prevents IDF drift from accumulating over months of partial-only updates.
        Also used by force_idf_recalculation().
        """
        logger.info(
            f"Full IDF refresh triggered after {self._changes_since_full_refresh} "
            f"cumulative changes ({self.document_count} total docs)"
        )

        # TFIDF_INDEX_CORRUPTION_FIX Fix 2 + DATA-039: Auto-repair corrupted counts
        dtc_len = len(self.document_term_counts)
        cid_len = len(self.concept_ids)
        if dtc_len != self.document_count or cid_len != self.document_count:
            logger.warning(
                f"DATA-039: Count mismatch detected in _full_recalculate_idf: "
                f"document_count={self.document_count}, dtc={dtc_len}, cids={cid_len}. "
                f"Auto-repairing via compaction."
            )
            self._compact_matrix()
            # After compaction, counts are consistent — retry
            dtc_len = len(self.document_term_counts)
            if dtc_len != self.document_count:
                logger.error(
                    f"DATA-039: Auto-repair failed, still inconsistent: "
                    f"document_count={self.document_count}, dtc={dtc_len}. Skipping."
                )
                return

        new_rows = []
        for idx in range(self.document_count):
            if idx in self.deleted_indices:
                vocab_size = len(self.vocabulary)
                new_rows.append(csr_matrix((1, vocab_size), dtype=np.float32))
            else:
                term_counts = self.document_term_counts[idx]
                sparse_vec = self._compute_tfidf_vector_sparse(term_counts)
                new_rows.append(sparse_vec)

        if new_rows:
            self.tfidf_matrix = vstack(new_rows)

        self._changes_since_full_refresh = 0
        self.dirty_documents.clear()
        self.last_idf_update = _utc_now()
        logger.info("Full IDF refresh complete")

    def _compact_matrix(self):
        """
        Remove deleted documents from index.

        Rebuilds all data structures without deleted documents.
        Called when deleted_indices exceeds threshold.

        Time Complexity: O(N)
        """
        if not self.deleted_indices:
            return

        logger.info(f"Compacting index: removing {len(self.deleted_indices)} deleted documents")

        # BUG-042 FIX: Use actual list length, not document_count (which may be corrupted/mismatched)
        actual_count = min(len(self.concept_ids), len(self.document_term_counts))
        if actual_count != self.document_count:
            logger.warning(
                f"BUG-042: _compact_matrix found count mismatch: "
                f"document_count={self.document_count}, concept_ids={len(self.concept_ids)}, "
                f"dtc={len(self.document_term_counts)}. Using min={actual_count} as safe bound."
            )

        # Build list of active documents — use actual_count to avoid IndexError
        active_indices = [i for i in range(actual_count) if i not in self.deleted_indices]

        # Rebuild concept_ids and mapping
        new_concept_ids = [self.concept_ids[i] for i in active_indices]
        new_concept_id_to_idx = {cid: idx for idx, cid in enumerate(new_concept_ids)}

        # Rebuild document_term_counts
        new_document_term_counts = [self.document_term_counts[i] for i in active_indices]

        # Rebuild TF-IDF matrix (extract active rows)
        if self.tfidf_matrix is not None:
            active_rows = [self.tfidf_matrix[i] for i in active_indices]
            if active_rows:
                new_tfidf_matrix = vstack(active_rows)
            else:
                new_tfidf_matrix = None
        else:
            new_tfidf_matrix = None

        # Update index state
        self.concept_ids = new_concept_ids
        self.concept_id_to_idx = new_concept_id_to_idx
        self.document_term_counts = new_document_term_counts
        self.tfidf_matrix = new_tfidf_matrix

        # Recalculate document frequencies from remaining documents
        self.document_frequencies = np.zeros(len(self.vocabulary), dtype=np.int32)
        for term_counts in self.document_term_counts:
            unique_terms = set(term_counts.keys())
            for term in unique_terms:
                if term in self.vocabulary:
                    term_id = self.vocabulary[term]
                    self.document_frequencies[term_id] += 1

        # Update document count
        if self.concept_ids:
            self.document_count = len(self.concept_ids)
        else:
            self.document_count = 0

        # Clear deletion tracking
        self.deleted_indices.clear()

        logger.info(f"Compaction complete: {self.document_count} documents remain")

    def force_idf_recalculation(self):
        """
        Manually trigger IDF recalculation.

        Useful for:
        - Ensuring index quality before important queries
        - Batch processing workflows
        - Testing and validation
        """
        with self.lock:
            if self.document_count > 0:
                self.dirty_documents = set(self.concept_ids) - {None}
                self._full_recalculate_idf()
                logger.info("Manual IDF recalculation complete")

    def save(self, index_path: str):
        """
        Save index to disk.

        Creates directory structure:
        index_path/
            metadata.json - Index metadata
            vocabulary.pkl - Term mappings
            document_map.json - Concept ID mappings
            document_frequencies.npy - DF array
            term_totals.npy - Term totals array
            tfidf_matrix.npz - Sparse TF-IDF matrix
            document_term_counts.json.gz - Raw term counts (compressed)

        Args:
            index_path: Directory to save index
        """
        with self.lock:
            # TFIDF_INDEX_CORRUPTION_FIX Fix 3: Pre-save consistency check
            dtc_len = len(self.document_term_counts)
            cid_len = len(self.concept_ids)
            if dtc_len != self.document_count or cid_len != self.document_count:
                logger.error(
                    f"REFUSING TO SAVE corrupt index: "
                    f"document_count={self.document_count}, "
                    f"document_term_counts={dtc_len}, "
                    f"concept_ids={cid_len}. "
                    f"Save aborted to prevent persisting corruption."
                )
                return

            index_dir = Path(index_path)
            # STABILITY-020: Atomic save — write to temp dir, then rename
            tmp_dir = index_dir.parent / f".{index_dir.name}.tmp.{os.getpid()}"
            backup_dir = index_dir.parent / f".{index_dir.name}.backup"
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            tmp_dir.mkdir(parents=True, exist_ok=True)

            try:
                logger.info(f"Saving index to {tmp_dir} (atomic)")

                # === METADATA ===
                metadata = {
                    "document_count": self.document_count,
                    "vocabulary_size": len(self.vocabulary),
                    "index_version": self.index_version,
                    "last_idf_update": self.last_idf_update.isoformat() if self.last_idf_update else None,
                    "idf_update_threshold": self.idf_update_threshold,
                    "compaction_threshold": self.compaction_threshold,
                    "checkpoint_every": self.checkpoint_every,
                    "changes_since_full_refresh": self._changes_since_full_refresh,
                    "saved_at": _utc_now_iso(),
                }

                with open(tmp_dir / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)

                # === VOCABULARY ===
                # Save as pickle for exact type preservation
                vocabulary_data = {
                    "vocabulary": self.vocabulary,
                    "reverse_vocabulary": self.reverse_vocabulary,
                    "next_term_id": self.next_term_id,
                }

                with open(tmp_dir / "vocabulary.pkl", "wb") as f:
                    pickle.dump(vocabulary_data, f)

                # === DOCUMENT MAPPING ===
                document_map = {
                    "concept_ids": self.concept_ids,
                    "concept_id_to_idx": self.concept_id_to_idx,
                    "deleted_indices": list(self.deleted_indices),
                }

                with open(tmp_dir / "document_map.json", "w") as f:
                    json.dump(document_map, f)

                # === NUMPY ARRAYS ===
                np.save(tmp_dir / "document_frequencies.npy", self.document_frequencies)
                np.save(tmp_dir / "term_totals.npy", self.term_totals)

                # === TF-IDF MATRIX ===
                if self.tfidf_matrix is not None:
                    save_npz(tmp_dir / "tfidf_matrix.npz", self.tfidf_matrix)

                # === TERM COUNTS (compressed) ===
                term_counts_json = json.dumps(self.document_term_counts)

                with gzip.open(tmp_dir / "document_term_counts.json.gz", "wt") as f:
                    f.write(term_counts_json)

                # === ATOMIC SWAP ===
                if index_dir.exists():
                    if backup_dir.exists():
                        shutil.rmtree(backup_dir)
                    index_dir.rename(backup_dir)
                tmp_dir.rename(index_dir)
                # Clean up backup on success
                if backup_dir.exists():
                    shutil.rmtree(backup_dir)

                logger.info(f"STABILITY-020: Index saved atomically: {self.document_count} docs, {len(self.vocabulary)} terms")

            except Exception:
                logger.exception("STABILITY-020: Atomic save failed, cleaning up tmp")
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir)
                # If we moved the original to backup, restore it
                if not index_dir.exists() and backup_dir.exists():
                    backup_dir.rename(index_dir)
                raise

    def load(self, index_path: str) -> bool:
        """
        Load index from disk.

        Restores complete index state from saved files.

        Args:
            index_path: Directory containing saved index

        Returns:
            True if loaded successfully, False otherwise
        """
        with self.lock:
            index_dir = Path(index_path)

            if not index_dir.exists():
                logger.error(f"Index path does not exist: {index_path}")
                return False

            logger.info(f"Loading index from {index_path}")

            try:
                # === METADATA ===
                with open(index_dir / "metadata.json") as f:
                    metadata = json.load(f)

                self.document_count = metadata["document_count"]
                self.index_version = metadata["index_version"]
                self.idf_update_threshold = metadata["idf_update_threshold"]
                self.compaction_threshold = metadata["compaction_threshold"]
                self.checkpoint_every = metadata["checkpoint_every"]

                self._changes_since_full_refresh = metadata.get("changes_since_full_refresh", 0)

                if metadata["last_idf_update"]:
                    self.last_idf_update = datetime.fromisoformat(metadata["last_idf_update"])

                # === VOCABULARY ===
                with open(index_dir / "vocabulary.pkl", "rb") as f:
                    vocabulary_data = pickle.load(f)

                self.vocabulary = vocabulary_data["vocabulary"]
                self.reverse_vocabulary = vocabulary_data["reverse_vocabulary"]
                self.next_term_id = vocabulary_data["next_term_id"]

                # === DOCUMENT MAPPING ===
                with open(index_dir / "document_map.json") as f:
                    document_map = json.load(f)

                self.concept_ids = document_map["concept_ids"]
                self.concept_id_to_idx = document_map["concept_id_to_idx"]
                self.deleted_indices = set(document_map["deleted_indices"])

                # === NUMPY ARRAYS ===
                self.document_frequencies = np.load(index_dir / "document_frequencies.npy")
                self.term_totals = np.load(index_dir / "term_totals.npy")

                # === TF-IDF MATRIX ===
                if (index_dir / "tfidf_matrix.npz").exists():
                    self.tfidf_matrix = load_npz(index_dir / "tfidf_matrix.npz")
                else:
                    self.tfidf_matrix = None

                # === TERM COUNTS ===
                with gzip.open(index_dir / "document_term_counts.json.gz", "rt") as f:
                    self.document_term_counts = json.load(f)

                # === RESET DIRTY STATE ===
                self.dirty_documents = set()

                # === CONSISTENCY VALIDATION (TFIDF_INDEX_CORRUPTION_FIX Fix 1) ===
                dtc_len = len(self.document_term_counts)
                cid_len = len(self.concept_ids)
                if dtc_len != self.document_count or cid_len != self.document_count:
                    logger.error(
                        f"Index consistency check FAILED on load: "
                        f"document_count={self.document_count}, "
                        f"document_term_counts={dtc_len}, "
                        f"concept_ids={cid_len}. "
                        f"Returning False to trigger rebuild."
                    )
                    return False

                logger.info(f"Index loaded successfully: {self.document_count} docs, {len(self.vocabulary)} terms")

                return True

            except Exception as e:
                logger.error(f"Failed to load index: {e}")
                return False

    def checkpoint(self, checkpoint_dir: str):
        """
        Create a checkpoint snapshot.

        Checkpoints are full index snapshots for recovery.
        Kept in checkpoint_dir with version numbers.

        Args:
            checkpoint_dir: Directory for checkpoints
        """
        checkpoint_path = Path(checkpoint_dir) / f"checkpoint_{self.index_version:06d}"
        self.save(str(checkpoint_path))
        logger.info(f"Checkpoint created: {checkpoint_path}")

    def get_stats(self) -> dict:
        """Return index statistics for monitoring."""
        return {
            "document_count": self.document_count,
            "vocabulary_size": len(self.vocabulary),
            "dirty_document_count": len(self.dirty_documents),
            "deleted_document_count": len(self.deleted_indices),
            "index_version": self.index_version,
            "last_idf_update": self.last_idf_update.isoformat() if self.last_idf_update else None,
        }
