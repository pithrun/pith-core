"""Retrospective learning - ingest and learn from historical conversations."""

import json
from pathlib import Path

from app.datetime_utils import _utc_now_iso
from app.learning import create_concept, validate_proposal
from app.models import ConceptProposal
from app.storage import load_concept

# Configuration
MIN_CHUNK_SIZE = 200  # Words per chunk
MAX_CHUNK_SIZE = 1000  # Maximum words per chunk
MIN_INSIGHT_LENGTH = 50  # Minimum characters for valid insight
BATCH_SIZE = 10  # Process chunks in batches


class ConversationProcessor:
    """Process historical conversations into learnable concepts."""

    def __init__(self):
        self.processed_count = 0
        self.concepts_created = 0
        self.concepts_evolved = 0
        self.insights_extracted = 0

    def chunk_text(self, text: str, size: int = MIN_CHUNK_SIZE) -> list[str]:
        """
        Split text into meaningful chunks.

        Improvements over original:
        - Respect sentence boundaries
        - Avoid breaking mid-sentence
        - Variable chunk sizes based on content
        """
        # Split into sentences (simple approach)
        sentences = text.replace("!", ".").replace("?", ".").split(".")
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = []
        current_word_count = 0

        for sentence in sentences:
            words = sentence.split()
            word_count = len(words)

            # If adding this sentence exceeds max, start new chunk
            if current_word_count + word_count > MAX_CHUNK_SIZE and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_word_count = word_count
            # If we have enough words for a chunk, complete it
            elif current_word_count >= size and word_count > 20:
                current_chunk.append(sentence)
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_word_count = 0
            # Otherwise keep building chunk
            else:
                current_chunk.append(sentence)
                current_word_count += word_count

        # Add remaining
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def extract_insight_heuristic(self, text: str) -> dict | None:
        """
        Heuristic insight extraction (fallback method).

        Improvements over original:
        - Pattern detection for common themes
        - Multiple signal types
        - Quality filtering
        """
        if len(text) < MIN_INSIGHT_LENGTH:
            return None

        text_lower = text.lower()

        # Detect insight patterns
        patterns = {
            "problem": ["issue", "problem", "error", "bug", "broken", "failing"],
            "inefficiency": ["slow", "inefficient", "bottleneck", "delay", "waste"],
            "decision": ["should", "choose", "decide", "option", "alternative"],
            "learning": ["learned", "discovered", "realized", "found that"],
            "process": ["workflow", "process", "procedure", "steps", "method"],
            "technical": ["code", "function", "api", "database", "server"],
        }

        detected_signals = []
        detected_type = None

        for pattern_type, keywords in patterns.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches >= 2:  # At least 2 keyword matches
                detected_signals.append(pattern_type)
                if not detected_type:
                    detected_type = pattern_type

        if not detected_signals:
            return None

        # Extract key phrases (simple heuristic)
        sentences = text.split(".")
        key_sentence = sentences[0] if sentences else text

        # Limit summary length
        summary = key_sentence[:200] + "..." if len(key_sentence) > 200 else key_sentence

        return {
            "summary": summary.strip(),
            "signals": detected_signals,
            "type": detected_type,
            "confidence": min(0.7, len(detected_signals) * 0.15),
        }

    def extract_insight_structured(self, text: str) -> dict | None:
        """
        Extract structured insights using pattern matching.

        This is more sophisticated than the heuristic version.
        Looks for specific conversation patterns.
        """
        if len(text) < MIN_INSIGHT_LENGTH:
            return None

        insights = []

        # Pattern 1: Problem statements
        if any(phrase in text.lower() for phrase in ["the problem is", "issue is", "challenge is"]):
            # Extract the problem
            for marker in ["the problem is", "issue is", "challenge is"]:
                if marker in text.lower():
                    idx = text.lower().index(marker)
                    problem = text[idx : idx + 200]
                    insights.append(
                        {
                            "type": "problem_identification",
                            "content": problem,
                            "signals": ["problem", "issue", "challenge"],
                        }
                    )

        # Pattern 2: Decisions made
        if any(phrase in text.lower() for phrase in ["we decided", "chose to", "going with"]):
            insights.append({"type": "decision", "content": text[:200], "signals": ["decision", "choice", "selected"]})

        # Pattern 3: Learnings
        if any(phrase in text.lower() for phrase in ["learned that", "discovered", "realized"]):
            insights.append(
                {"type": "learning", "content": text[:200], "signals": ["learning", "insight", "discovery"]}
            )

        # Pattern 4: Process descriptions
        if any(phrase in text.lower() for phrase in ["the process", "workflow", "steps are"]):
            insights.append({"type": "process", "content": text[:200], "signals": ["process", "workflow", "procedure"]})

        # Return best insight
        if insights:
            best = insights[0]
            return {
                "summary": best["content"].strip(),
                "signals": best["signals"],
                "type": best["type"],
                "confidence": 0.65,
            }

        # Fall back to heuristic
        return self.extract_insight_heuristic(text)

    def deduplicate_insight(self, insight: dict, existing_concepts: list[str]) -> bool:
        """
        Check if this insight is substantially different from existing concepts.

        Returns True if it's a duplicate (should skip).
        """
        summary = insight["summary"].lower()

        for concept_id in existing_concepts:
            concept = load_concept(concept_id, track_access=False)
            if not concept:
                continue

            # Simple similarity check
            concept_summary = concept.summary.lower()

            # Check for substring match
            if summary in concept_summary or concept_summary in summary:
                return True

            # Check for high word overlap
            summary_words = set(summary.split())
            concept_words = set(concept_summary.split())

            overlap = len(summary_words & concept_words)
            min_length = min(len(summary_words), len(concept_words))

            if min_length > 0 and overlap / min_length > 0.7:
                return True

        return False

    def process_chunk(
        self, chunk: str, source_id: str, chunk_index: int, knowledge_area: str = "imported"
    ) -> dict | None:
        """
        Process a single chunk and create/evolve concepts.

        Returns processing result.
        """
        # Extract insight
        insight = self.extract_insight_structured(chunk)

        if not insight:
            return None

        self.insights_extracted += 1

        # Create concept proposal
        concept_id = f"imported_{source_id}_{insight['type']}_{chunk_index}"

        # Memory Integrity A4-H1: Dedup at ingestion using TF-IDF
        try:
            from app.config import FEATURE_FLAGS

            if FEATURE_FLAGS.get("DEDUP_AT_INGESTION_ENABLED", False):
                from app.retrieval import retrieval_engine

                dedup_results = retrieval_engine.search_for_dedup_tfidf(insight["summary"], top_k=3)
                if dedup_results and dedup_results[0]["cosine_score"] >= 0.85:
                    return {
                        "status": "duplicate",
                        "concept_id": None,
                        "reason": f"TF-IDF duplicate of {dedup_results[0]['concept_id']} (cosine={dedup_results[0]['cosine_score']:.3f})",
                    }
            else:
                # Fallback: legacy word-overlap dedup
                from app.storage import list_concepts

                existing = list_concepts()
                if self.deduplicate_insight(insight, existing):
                    return {"status": "duplicate", "concept_id": None, "reason": "Similar concept already exists"}
        except Exception as e:
            logger.warning(f"import_conversation: dedup check failed (non-fatal): {e}")
            # Fallback to legacy dedup on error
            from app.storage import list_concepts

            existing = list_concepts()
            if self.deduplicate_insight(insight, existing):
                return {"status": "duplicate", "concept_id": None, "reason": "Similar concept already exists"}

        # Create proposal
        proposal = ConceptProposal(
            concept_id=concept_id,
            summary=insight["summary"],
            knowledge_area=knowledge_area,
            evidence=[f"{source_id}_chunk_{chunk_index}"],
            signals=insight["signals"],
            confidence=insight.get("confidence", 0.5),
        )

        # Validate and create
        valid, message = validate_proposal(proposal)

        if not valid:
            return {"status": "rejected", "concept_id": None, "reason": message}

        # Create concept
        concept = create_concept(proposal)
        self.concepts_created += 1

        # Retrieval Defense W8: Import path quarantine
        # Imported concepts start QUARANTINED — lower provenance than deliberate propose
        try:
            from app.config import FEATURE_FLAGS

            if FEATURE_FLAGS.get("INGESTION_VALIDATION_ENABLED", False):
                from app.storage import save_concept as _save

                concept.maturity = "QUARANTINED"
                concept.quarantine_entered = _utc_now_iso()
                _save(concept)
                import logging

                logging.getLogger(__name__).info(
                    "W8: Import quarantine applied to %s (source=%s)", concept.id, source_id
                )
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning("W8: Import quarantine failed for %s: %s", concept.id, e)

        return {
            "status": "created",
            "concept_id": concept.id,
            "summary": concept.summary,
            "confidence": concept.confidence,
        }

    def process_conversation(
        self, conversation_text: str, source_id: str, knowledge_area: str = "imported", chunk_size: int = MIN_CHUNK_SIZE
    ) -> dict:
        """
        Process entire conversation and extract learnings.

        Returns summary of processing.
        """
        # Split into chunks
        chunks = self.chunk_text(conversation_text, chunk_size)

        results = {
            "source_id": source_id,
            "chunks_processed": len(chunks),
            "concepts_created": 0,
            "concepts_evolved": 0,
            "duplicates_skipped": 0,
            "rejected": 0,
            "insights": [],
        }

        # Process each chunk
        for i, chunk in enumerate(chunks):
            result = self.process_chunk(chunk, source_id, i, knowledge_area)

            if result:
                results["insights"].append(result)

                if result["status"] == "created":
                    results["concepts_created"] += 1
                elif result["status"] == "duplicate":
                    results["duplicates_skipped"] += 1
                elif result["status"] == "rejected":
                    results["rejected"] += 1

        self.processed_count += 1

        return results

    def process_batch(self, conversations: list[dict], knowledge_area: str = "imported") -> dict:
        """
        Process multiple conversations in batch.

        Args:
            conversations: List of {"text": str, "source_id": str} dicts
            knowledge_area: Knowledge area for all concepts

        Returns:
            Batch processing summary
        """
        batch_results = {
            "total_conversations": len(conversations),
            "total_concepts_created": 0,
            "total_duplicates": 0,
            "total_rejected": 0,
            "conversations": [],
        }

        for conv in conversations:
            result = self.process_conversation(
                conversation_text=conv["text"],
                source_id=conv.get("source_id", "unknown"),
                knowledge_area=knowledge_area,
            )

            batch_results["total_concepts_created"] += result["concepts_created"]
            batch_results["total_duplicates"] += result["duplicates_skipped"]
            batch_results["total_rejected"] += result["rejected"]
            batch_results["conversations"].append(result)

        return batch_results

    def get_stats(self) -> dict:
        """Get processing statistics."""
        return {
            "conversations_processed": self.processed_count,
            "concepts_created": self.concepts_created,
            "concepts_evolved": self.concepts_evolved,
            "insights_extracted": self.insights_extracted,
        }


def parse_claude_export(export_path: Path) -> list[dict]:
    """
    Parse Claude chat export JSON.

    Claude exports are in JSON format with conversation structure.
    """
    if not export_path.exists():
        return []

    with open(export_path) as f:
        data = json.load(f)

    conversations = []

    # Claude export format (example - adjust based on actual format)
    if isinstance(data, list):
        for conv in data:
            if "messages" in conv:
                # Combine messages into conversation text
                text = "\n\n".join(f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in conv["messages"])
                conversations.append({"text": text, "source_id": conv.get("id", "unknown")})

    return conversations


def parse_text_file(file_path: Path) -> list[dict]:
    """
    Parse plain text file.

    Simple text files treated as single conversation.
    """
    if not file_path.exists():
        return []

    with open(file_path, encoding="utf-8") as f:
        text = f.read()

    return [{"text": text, "source_id": file_path.stem}]


# Global instance
conversation_processor = ConversationProcessor()
