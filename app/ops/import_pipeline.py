"""
Conversation Import Pipeline — ChatGPT/Claude/Gemini history import.

Spec: Conversation Import Pipeline v2 internal design notes.
Gauntlet: v2 scored 8.5/10 PASS

Pipeline stages:
  [Export File] → [Parser] → [Normalizer] → [Extended Bulk Importer] → [Report Generator]

Architecture:
  - Parsers produce raw conversation dicts per source format
  - Normalizer converts to common intermediate format + chronological sort
  - Extended bulk importer wraps existing process_batch with checkpointing/progress/limits
  - Report generator produces hot path (contradictions) or cold path (belief evolution)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================
# Constants
# ============================================================

MAX_CONVERSATIONS_PER_IMPORT = 5000
WARN_CONVERSATIONS_THRESHOLD = 2000
CHECKPOINT_INTERVAL = 50  # Save progress every N conversations
PROGRESS_INTERVAL = 10  # Emit progress every N conversations
MAX_MESSAGE_TOKENS = 8000  # Truncate individual messages beyond this
TRUNCATION_MARKER = "\n[...truncated...]\n"
MAX_PREVIOUS_MESSAGE_LENGTH = 15000  # Matches session.py MAX_PREVIOUS_RESPONSE


def _restore_env(key: str, original_value: str | None) -> None:
    """Restore an environment variable to its original value."""
    import os

    if original_value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = original_value


# ============================================================
# Data Models
# ============================================================


@dataclass
class NormalizedMessage:
    """Single message in normalized format."""

    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: str | None = None


@dataclass
class NormalizedConversation:
    """Common intermediate format for all source parsers."""

    source: str  # "chatgpt" | "claude" | "gemini"
    source_id: str  # Original conversation ID
    title: str
    created_at: str  # ISO 8601
    messages: list[NormalizedMessage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Flatten messages into conversation text for process_conversation."""
        return "\n\n".join(
            f"{msg.role}: {msg.content}" for msg in self.messages if msg.role != "system" and msg.content.strip()
        )

    @property
    def dedup_hash(self) -> str:
        """Hash for deduplication: source + source_id."""
        return hashlib.sha256(f"{self.source}:{self.source_id}".encode()).hexdigest()[:16]


@dataclass
class ImportProgress:
    """Progress tracking for bulk import."""

    total: int = 0
    processed: int = 0
    concepts_extracted: int = 0
    duplicates_skipped: int = 0
    errors: int = 0
    skipped_dedup: int = 0
    elapsed_sec: float = 0.0
    aborted: bool = False

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "processed": self.processed,
            "concepts_extracted": self.concepts_extracted,
            "duplicates_skipped": self.duplicates_skipped,
            "errors": self.errors,
            "skipped_dedup": self.skipped_dedup,
            "elapsed_sec": round(self.elapsed_sec, 2),
            "aborted": self.aborted,
        }


# ============================================================
# Stage 1: Format Parsers
# ============================================================


class ParseError(Exception):
    """Raised when a parser encounters an unrecoverable format issue."""

    pass


def _truncate_message(content: str, max_tokens: int = MAX_MESSAGE_TOKENS) -> str:
    """Truncate message content preserving first + last halves.

    Uses word count as token proxy (roughly 0.75 tokens per word).
    """
    words = content.split()
    estimated_tokens = len(words)
    if estimated_tokens <= max_tokens:
        return content

    half = max_tokens // 2
    first_half = " ".join(words[:half])
    last_half = " ".join(words[-half:])
    return first_half + TRUNCATION_MARKER + last_half


def parse_chatgpt_export(file_path: Path) -> list[dict]:
    """Parse ChatGPT conversations.json export.

    Handles:
      - ZIP file containing conversations.json
      - Direct conversations.json file
      - Tree-based mapping format (current, post-2024)
      - Legacy flat message array format (pre-2024)

    Returns list of raw conversation dicts with keys:
      title, create_time, messages: [{role, content, timestamp}]
    """
    path = Path(file_path)

    if not path.exists():
        raise ParseError(f"File not found: {path}")

    # Validate file type
    suffix = path.suffix.lower()
    if suffix not in (".zip", ".json"):
        raise ParseError(f"Expected .zip or .json file, got: {path.name}")

    # Extract conversations.json from ZIP if needed
    raw_data = _load_chatgpt_data(path)

    if not isinstance(raw_data, list):
        raise ParseError("Expected top-level JSON array of conversations")

    conversations = []
    for i, conv in enumerate(raw_data):
        try:
            parsed = _parse_single_chatgpt_conversation(conv)
            if parsed and parsed.get("messages"):
                conversations.append(parsed)
        except Exception as e:
            logger.warning(f"Skipping conversation {i}: {e}")

    logger.info(f"ChatGPT parser: {len(conversations)}/{len(raw_data)} conversations parsed")
    return conversations


def _load_chatgpt_data(path: Path) -> list:
    """Load conversations.json from ZIP or direct JSON file."""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            # Look for conversations.json in the ZIP
            json_files = [n for n in zf.namelist() if n.endswith("conversations.json")]
            if not json_files:
                raise ParseError("ZIP does not contain conversations.json")
            with zf.open(json_files[0]) as f:
                return json.load(f)
    else:
        with open(path, encoding="utf-8") as f:
            return json.load(f)


def _parse_single_chatgpt_conversation(conv: dict) -> dict | None:
    """Parse a single ChatGPT conversation from export data.

    Handles both tree-based (current) and flat array (legacy) formats.
    """
    title = conv.get("title", "Untitled")
    create_time = conv.get("create_time")
    conv_id = conv.get("id", conv.get("conversation_id", ""))

    # Determine format: tree-based (has "mapping") vs legacy (has flat "messages")
    if "mapping" in conv and isinstance(conv["mapping"], dict):
        messages = _walk_chatgpt_tree(conv["mapping"])
    elif "messages" in conv and isinstance(conv["messages"], list):
        messages = _parse_chatgpt_flat_messages(conv["messages"])
    else:
        return None  # Unrecognized structure

    # Convert create_time (Unix epoch float) to ISO 8601
    created_at = None
    if create_time is not None:
        try:
            created_at = datetime.fromtimestamp(float(create_time), tz=UTC).isoformat()
        except (ValueError, TypeError, OSError):
            created_at = None

    return {
        "source": "chatgpt",
        "source_id": str(conv_id),
        "title": title,
        "created_at": created_at,
        "messages": messages,
        "metadata": {
            "model": conv.get("default_model_slug", "unknown"),
            "message_count": len(messages),
        },
    }


def _walk_chatgpt_tree(mapping: dict) -> list[dict]:
    """Walk ChatGPT tree structure to reconstruct canonical conversation path.

    ChatGPT uses a tree where edits/regenerations create branches.
    Strategy: find root → walk to most recent leaf (last child at each fork).
    """
    if not mapping:
        return []

    # Find root node (no parent or parent not in mapping)
    root_id = None
    for node_id, node in mapping.items():
        parent = node.get("parent")
        if parent is None or parent not in mapping:
            root_id = node_id
            break

    if root_id is None:
        # Fallback: use first node
        root_id = next(iter(mapping))

    # Walk from root to leaf, always taking the last child (most recent)
    messages = []
    current_id = root_id
    visited = set()

    while current_id and current_id not in visited:
        visited.add(current_id)
        node = mapping.get(current_id)
        if node is None:
            break

        # Extract message if present
        msg_data = node.get("message")
        if msg_data and isinstance(msg_data, dict):
            author = msg_data.get("author", {})
            role = author.get("role", "unknown") if isinstance(author, dict) else "unknown"

            # Only keep user and assistant messages
            if role in ("user", "assistant"):
                content_obj = msg_data.get("content", {})
                if isinstance(content_obj, dict):
                    parts = content_obj.get("parts", [])
                    content = "\n".join(str(p) for p in parts if isinstance(p, str))
                elif isinstance(content_obj, str):
                    content = content_obj
                else:
                    content = ""

                if content.strip():
                    msg_ts = msg_data.get("create_time")
                    timestamp = None
                    if msg_ts is not None:
                        try:
                            timestamp = datetime.fromtimestamp(float(msg_ts), tz=UTC).isoformat()
                        except (ValueError, TypeError, OSError):
                            pass
                    messages.append(
                        {
                            "role": role,
                            "content": content.strip(),
                            "timestamp": timestamp,
                        }
                    )

        # Move to next node: take last child (most recent branch)
        children = node.get("children", [])
        if children:
            current_id = children[-1]  # Last child = most recent
        else:
            break  # Leaf node reached

    return messages


def _parse_chatgpt_flat_messages(messages: list) -> list[dict]:
    """Parse legacy ChatGPT flat message array format (pre-2024)."""
    parsed = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", msg.get("author", "unknown"))
        content = msg.get("content", "")
        if isinstance(content, dict):
            content = content.get("parts", [""])[0] if "parts" in content else str(content)
        if role in ("user", "assistant") and str(content).strip():
            parsed.append(
                {
                    "role": role,
                    "content": str(content).strip(),
                    "timestamp": None,
                }
            )
    return parsed


# ============================================================
# Stage 2: Normalizer
# ============================================================


def normalize_conversations(
    raw_conversations: list[dict],
    source: str,
) -> list[NormalizedConversation]:
    """Convert raw parser output to normalized format, sorted chronologically.

    Normalization rules (from spec):
      - Strip system prompts
      - Deduplicate consecutive identical messages (regenerations)
      - Filter empty/error messages
      - Truncate individual messages at MAX_MESSAGE_TOKENS
      - Sort by created_at (oldest first) for temporal ordering
    """
    normalized = []

    for conv in raw_conversations:
        messages = _normalize_messages(conv.get("messages", []))
        if not messages:
            continue  # Skip conversations with no usable messages

        # Build metadata
        word_count = sum(len(m.content.split()) for m in messages)
        metadata = conv.get("metadata", {})
        metadata["message_count"] = len(messages)
        metadata["word_count"] = word_count

        nc = NormalizedConversation(
            source=source,
            source_id=conv.get("source_id", "unknown"),
            title=conv.get("title", "Untitled"),
            created_at=conv.get("created_at") or datetime.now(UTC).isoformat(),
            messages=messages,
            metadata=metadata,
        )
        normalized.append(nc)

    # Sort chronologically (oldest first) — critical for temporal ordering
    # per benchmark ingestion protocol
    normalized.sort(key=lambda c: c.created_at)

    logger.info(
        f"Normalizer: {len(normalized)} conversations normalized from {source}, "
        f"sorted chronologically [{normalized[0].created_at if normalized else 'N/A'} → "
        f"{normalized[-1].created_at if normalized else 'N/A'}]"
    )
    return normalized


def _normalize_messages(raw_messages: list[dict]) -> list[NormalizedMessage]:
    """Apply normalization rules to message list."""
    messages = []
    prev_content = None

    for msg in raw_messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        # Skip system prompts
        if role == "system":
            continue

        # Skip empty messages
        if not content or not content.strip():
            continue

        # Deduplicate consecutive identical messages (regenerations)
        if content == prev_content:
            continue
        prev_content = content

        # Truncate oversized messages
        content = _truncate_message(content)

        messages.append(
            NormalizedMessage(
                role=role,
                content=content,
                timestamp=msg.get("timestamp"),
            )
        )

    return messages


# ============================================================
# Stage 3: Extended Bulk Importer
# ============================================================


@dataclass
class ImportCheckpoint:
    """Checkpoint state for resumable imports."""

    source: str
    total_conversations: int
    processed_source_ids: list[str] = field(default_factory=list)
    progress: ImportProgress = field(default_factory=ImportProgress)
    started_at: str = ""
    last_updated: str = ""

    def save(self, path: Path) -> None:
        """Persist checkpoint to disk."""
        self.last_updated = datetime.now(UTC).isoformat()
        data = {
            "source": self.source,
            "total_conversations": self.total_conversations,
            "processed_source_ids": self.processed_source_ids,
            "progress": self.progress.as_dict(),
            "started_at": self.started_at,
            "last_updated": self.last_updated,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> ImportCheckpoint | None:
        """Load checkpoint from disk. Returns None if not found."""
        if not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            cp = cls(
                source=data["source"],
                total_conversations=data["total_conversations"],
                processed_source_ids=data.get("processed_source_ids", []),
                started_at=data.get("started_at", ""),
                last_updated=data.get("last_updated", ""),
            )
            prog = data.get("progress", {})
            cp.progress = ImportProgress(
                total=prog.get("total", 0),
                processed=prog.get("processed", 0),
                concepts_extracted=prog.get("concepts_extracted", 0),
                duplicates_skipped=prog.get("duplicates_skipped", 0),
                errors=prog.get("errors", 0),
                skipped_dedup=prog.get("skipped_dedup", 0),
                elapsed_sec=prog.get("elapsed_sec", 0.0),
            )
            return cp
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None


# Abort flag for cancel support
_import_abort_flag = False


def cancel_import() -> None:
    """Set the abort flag to cancel an in-progress import."""
    global _import_abort_flag
    _import_abort_flag = True


def _check_abort() -> bool:
    """Check and return the current abort state."""
    return _import_abort_flag


def _extract_concepts_from_conversation(
    conversation_text: str,
    knowledge_area: str,
) -> list[dict]:
    """Extract concepts from conversation text for SessionManager ingestion.

    Uses ConversationProcessor.extract_insight_structured() for extraction,
    then formats results as concept dicts compatible with extracted_concepts_json
    (the Tier 2 format expected by conversation_turn).

    Returns list of concept dicts with keys:
        summary, confidence, knowledge_area, evidence, concept_type
    """
    from app.cognitive.retrospective import MIN_CHUNK_SIZE, ConversationProcessor

    processor = ConversationProcessor()
    chunks = processor.chunk_text(conversation_text, MIN_CHUNK_SIZE)
    concepts = []

    for chunk_idx, chunk in enumerate(chunks):
        insight = processor.extract_insight_structured(chunk)
        if not insight:
            continue

        summary = insight.get("summary", "").strip()
        if len(summary) < 30:
            summary = f"Imported insight: {summary}"

        concepts.append(
            {
                "summary": summary,
                "confidence": insight.get("confidence", 0.65),
                "knowledge_area": knowledge_area,
                "evidence": [f"Imported from conversation chunk {chunk_idx}"],
                "concept_type": insight.get("type", "observation"),
            }
        )

    return concepts


def extended_import_batch(
    conversations: list[NormalizedConversation],
    session_manager=None,
    knowledge_area_prefix: str = "imported",
    progress_callback: Callable[[ImportProgress], None] | None = None,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
) -> dict:
    """Extended batch import via SessionManager.conversation_turn().

    Routes all imported conversations through the proven ingestion path
    (SessionManager.conversation_turn + extracted_concepts_json) to get
    full pipeline benefits: verbatim, associations, FTS, embeddings,
    session tracking, and governance.

    Args:
        conversations: Normalized, chronologically sorted conversations
        session_manager: Initialized SessionManager instance
        knowledge_area_prefix: Base KA prefix (concepts get "imported:{source}")
        progress_callback: Called every PROGRESS_INTERVAL with ImportProgress
        checkpoint_dir: Directory for checkpoint files. None = no checkpointing.
        resume: If True, attempt to resume from checkpoint

    Returns:
        Import result dict with progress, per-conversation results, and report data
    """
    import os as _os

    from app.core.models import ConversationTurnRequest

    # INGEST-041: Validate session_manager before proceeding
    if session_manager is None:
        raise ValueError(
            "extended_import_batch() requires an initialized SessionManager instance. "
            "Pass session_manager=SessionManager() or use import_conversations() which "
            "creates one automatically."
        )

    global _import_abort_flag
    _import_abort_flag = False

    # Batch size — match PITH_MAX_INSIGHTS_PER_CALL to prevent silent truncation
    max_insights_cap = int(_os.environ.get("PITH_MAX_INSIGHTS_PER_CALL", "30"))
    BATCH_SIZE = min(max_insights_cap, 20)

    # Size limit enforcement
    if len(conversations) > MAX_CONVERSATIONS_PER_IMPORT:
        raise ValueError(
            f"Import exceeds maximum of {MAX_CONVERSATIONS_PER_IMPORT} conversations "
            f"({len(conversations)} provided). Split into smaller batches."
        )
    if len(conversations) > WARN_CONVERSATIONS_THRESHOLD:
        logger.warning(
            f"Large import: {len(conversations)} conversations (warning threshold: {WARN_CONVERSATIONS_THRESHOLD})"
        )

    # Initialize or resume checkpoint
    checkpoint_path = checkpoint_dir / "import_checkpoint.json" if checkpoint_dir else None
    checkpoint = None
    already_processed: set[str] = set()

    if resume and checkpoint_path:
        checkpoint = ImportCheckpoint.load(checkpoint_path)
        if checkpoint:
            already_processed = set(checkpoint.processed_source_ids)
            logger.info(f"Resuming import: {len(already_processed)} already processed")

    if checkpoint is None:
        checkpoint = ImportCheckpoint(
            source=conversations[0].source if conversations else "unknown",
            total_conversations=len(conversations),
            started_at=datetime.now(UTC).isoformat(),
        )

    progress = checkpoint.progress
    progress.total = len(conversations)
    start_time = time.perf_counter() - progress.elapsed_sec  # Account for resumed time

    # Track results per conversation
    conversation_results = []
    imported_source_ids = set(already_processed)

    # Sequential processing — preserves temporal ordering per benchmark protocol
    for i, conv in enumerate(conversations):
        # Check abort flag every PROGRESS_INTERVAL
        if i % PROGRESS_INTERVAL == 0 and _check_abort():
            logger.info(f"Import aborted at conversation {i}/{len(conversations)}")
            progress.aborted = True
            break

        # Dedup: skip if already imported (resume or re-import)
        dedup_key = conv.dedup_hash
        if dedup_key in imported_source_ids:
            progress.skipped_dedup += 1
            progress.processed += 1
            continue

        # Route through SessionManager.conversation_turn() (proven path)
        ka = f"{knowledge_area_prefix}:{conv.source}"
        conv_concepts_created = 0

        try:
            # Step A: Extract concepts from conversation text
            concepts = _extract_concepts_from_conversation(conv.text, ka)

            if not concepts:
                logger.info(f"No concepts extracted from conversation {conv.source_id}")
                progress.processed += 1
                imported_source_ids.add(dedup_key)
                continue

            # Step B: Process in batches through conversation_turn
            # (matches pith_agent.py _memorize_fact_list pattern)
            session_manager.start_session(
                context_hint=f"import_{conv.source}_{conv.source_id}",
                agent_id="import-pipeline",
            )
            try:
                for batch_offset in range(0, len(concepts), BATCH_SIZE):
                    batch = concepts[batch_offset : batch_offset + BATCH_SIZE]
                    tier2_json = json.dumps(batch)

                    # Build synthetic conversation context for auto-learn
                    sample_summaries = "; ".join(c["summary"][:80] for c in batch[:3])

                    req = ConversationTurnRequest(
                        message=f"Import from {conv.source}: {conv.title or conv.source_id}",
                        previous_response=f"Key insights from this conversation: {sample_summaries}",
                        previous_message=conv.text[:MAX_PREVIOUS_MESSAGE_LENGTH],
                        extracted_concepts_json=tier2_json,
                        max_concepts=3,
                        model_id="import-pipeline",
                        agent_id="import-pipeline",
                    )
                    result = session_manager.conversation_turn(req)

                    # Check for rate limiting (from pith_agent.py pattern)
                    bw = getattr(result, "budget_warnings", []) or []
                    rate_limited = any("rate_limit_exceeded" in str(w) for w in bw)
                    if rate_limited:
                        logger.warning(
                            f"Import batch at offset {batch_offset} for conv "
                            f"{conv.source_id} was RATE LIMITED — "
                            f"{len(batch)} concepts may be dropped. "
                            f"Raise PITH_SESSION_LEARN_RATE_LIMIT for bulk import."
                        )
                    else:
                        conv_concepts_created += len(batch)
            finally:
                session_manager.end_session()

            conversation_results.append(
                {
                    "source_id": f"{conv.source}:{conv.source_id}",
                    "concepts_created": conv_concepts_created,
                }
            )
            progress.concepts_extracted += conv_concepts_created
            imported_source_ids.add(dedup_key)
        except Exception as e:
            logger.warning(f"Error importing conversation {conv.source_id}: {e}")
            progress.errors += 1

        progress.processed += 1
        progress.elapsed_sec = time.perf_counter() - start_time

        # Progress callback
        if progress_callback and progress.processed % PROGRESS_INTERVAL == 0:
            progress_callback(progress)

        # Checkpoint
        if checkpoint_path and progress.processed % CHECKPOINT_INTERVAL == 0:
            checkpoint.processed_source_ids = list(imported_source_ids)
            checkpoint.progress = progress
            checkpoint.save(checkpoint_path)
            logger.info(f"Checkpoint saved: {progress.processed}/{progress.total}")

    # Final progress update
    progress.elapsed_sec = time.perf_counter() - start_time

    # Save final checkpoint
    if checkpoint_path:
        checkpoint.processed_source_ids = list(imported_source_ids)
        checkpoint.progress = progress
        checkpoint.save(checkpoint_path)

    logger.info(
        f"Import complete: {progress.processed}/{progress.total} processed, "
        f"{progress.concepts_extracted} concepts, {progress.errors} errors, "
        f"{progress.skipped_dedup} dedup skipped, {progress.elapsed_sec:.1f}s"
    )

    return {
        "progress": progress.as_dict(),
        "conversation_results": conversation_results,
        "source": conversations[0].source if conversations else "unknown",
        "knowledge_area": f"{knowledge_area_prefix}:{conversations[0].source}"
        if conversations
        else knowledge_area_prefix,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
    }


# ============================================================
# Stage 4: Report Generator (Hot + Cold Paths)
# ============================================================

CONTRADICTION_THRESHOLD = 3  # ≥3 contradictions → hot path


def generate_import_report(
    import_result: dict,
    t1: str | None = None,
    t2: str | None = None,
) -> dict:
    """Generate post-import report using hot or cold path.

    Hot path (≥3 contradictions): contradiction-focused report
    Cold path (<3 contradictions): belief evolution report

    Args:
        import_result: Output from extended_import_batch
        t1: Start timestamp for belief_diff (defaults to import start)
        t2: End timestamp for belief_diff (defaults to now)

    Returns:
        Report dict with type, summary, and detailed sections
    """
    from app.cognitive.belief_diff import belief_diff

    progress = import_result.get("progress", {})
    ka = import_result.get("knowledge_area")

    # Default timestamps: use import window
    if t2 is None:
        t2 = datetime.now(UTC).isoformat()
    if t1 is None:
        # Estimate import start: now minus elapsed_sec
        elapsed = progress.get("elapsed_sec", 60)
        from datetime import timedelta

        t1 = (datetime.now(UTC) - timedelta(seconds=elapsed + 10)).isoformat()

    # Run belief_diff to find contradictions in the import window
    diff_result = {}
    try:
        diff_result = belief_diff(t1=t1, t2=t2, knowledge_area=ka)
    except Exception as e:
        logger.warning(f"belief_diff failed during report generation: {e}")

    contradictions = diff_result.get("changed", [])
    num_contradictions = len(contradictions)

    # Route to hot or cold path
    if num_contradictions >= CONTRADICTION_THRESHOLD:
        return _generate_hot_report(progress, contradictions, diff_result, ka)
    else:
        return _generate_cold_report(progress, diff_result, ka)


def _generate_hot_report(
    progress: dict,
    contradictions: list,
    diff_result: dict,
    knowledge_area: str | None,
) -> dict:
    """Hot path: contradiction-focused report (≥3 contradictions found)."""
    # Rank contradictions by confidence delta (most significant first)
    ranked = sorted(
        contradictions,
        key=lambda c: abs(c.get("confidence_delta", 0) if isinstance(c, dict) else 0),
        reverse=True,
    )
    top_5 = ranked[:5]

    # Group by knowledge area for clustering
    ka_clusters: dict[str, list] = {}
    for c in contradictions:
        c_ka = c.get("knowledge_area", "unknown") if isinstance(c, dict) else "unknown"
        ka_clusters.setdefault(c_ka, []).append(c)

    total_concepts = progress.get("concepts_extracted", 0)
    total_convos = progress.get("processed", 0)

    return {
        "type": "contradiction",
        "summary": (
            f"Imported {total_convos} conversations, extracted {total_concepts} concepts. "
            f"Found {len(contradictions)} contradictions across your AI conversation history."
        ),
        "stats": {
            "conversations_processed": total_convos,
            "concepts_extracted": total_concepts,
            "contradictions_found": len(contradictions),
            "knowledge_area_clusters": {k: len(v) for k, v in ka_clusters.items()},
        },
        "top_contradictions": [
            {
                "summary": c.get("summary", "Unknown") if isinstance(c, dict) else str(c),
                "knowledge_area": c.get("knowledge_area", "unknown") if isinstance(c, dict) else "unknown",
                "confidence_delta": c.get("confidence_delta", 0) if isinstance(c, dict) else 0,
            }
            for c in top_5
        ],
        "all_contradictions": contradictions,
        "diff_result": diff_result,
    }


def _generate_cold_report(
    progress: dict,
    diff_result: dict,
    knowledge_area: str | None,
) -> dict:
    """Cold path: belief evolution report (<3 contradictions).

    Shows: concept stats, KA distribution, timeline, cross-platform overlap.
    Always produces something interesting even with 0 contradictions.
    """
    total_concepts = progress.get("concepts_extracted", 0)
    total_convos = progress.get("processed", 0)

    # Gather concept stats from diff_result
    added = diff_result.get("added", [])
    stats = diff_result.get("stats", {})

    # Compute KA distribution from added concepts
    ka_distribution: dict[str, int] = {}
    for concept in added:
        c_ka = concept.get("knowledge_area", "unknown") if isinstance(concept, dict) else "unknown"
        ka_distribution[c_ka] = ka_distribution.get(c_ka, 0) + 1

    # Sort by frequency
    ka_sorted = sorted(ka_distribution.items(), key=lambda x: x[1], reverse=True)

    return {
        "type": "belief_evolution",
        "summary": (
            f"Imported {total_convos} conversations, extracted {total_concepts} concepts. "
            f"Your thinking is remarkably consistent!"
        ),
        "stats": {
            "conversations_processed": total_convos,
            "concepts_extracted": total_concepts,
            "contradictions_found": len(diff_result.get("changed", [])),
        },
        "knowledge_area_distribution": dict(ka_sorted[:10]),
        "top_domains": [ka for ka, _ in ka_sorted[:5]],
        "concepts_added": len(added),
        "diff_result": diff_result,
    }


# ============================================================
# Stage 5: Top-Level Pipeline Orchestrator
# ============================================================

SUPPORTED_SOURCES = ("chatgpt", "claude", "sharegpt")


def parse_sharegpt_export(file_path: Path) -> list[dict]:
    """Parse ShareGPT-format JSON export (HuggingFace public datasets).

    TEST-177: Enables scale testing with 52K-142K public HuggingFace ShareGPT
    datasets. ShareGPT format is a JSON array of conversation objects, each with
    a ``conversations`` list of ``{from, value}`` turns.

    Supported ``from`` roles: ``human`` → user, ``gpt`` → assistant,
    ``system`` → system.  Unknown roles are silently dropped.

    Args:
        file_path: Path to a .json file (array of ShareGPT conversation objects).

    Returns:
        List of normalised conversation dicts compatible with the pith import
        pipeline's ``raw_conversations`` format (same shape as ChatGPT output):
        ``{"title": str, "messages": list[{"role": str, "content": str}], "source": "sharegpt"}``.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"ShareGPT export not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw: list = json.load(fh)

    if not isinstance(raw, list):
        raise ValueError(f"Expected JSON array at top level, got {type(raw).__name__}")

    _ROLE_MAP = {"human": "user", "gpt": "assistant", "system": "system"}
    conversations: list[dict] = []

    for i, conv in enumerate(raw):
        if not isinstance(conv, dict):
            continue
        turns = conv.get("conversations") or conv.get("conversation") or []
        if not isinstance(turns, list):
            continue

        messages: list[dict] = []
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            role = _ROLE_MAP.get(str(turn.get("from", "")).lower())
            value = str(turn.get("value", "")).strip()
            if role and value:
                messages.append({"role": role, "content": value})

        if not messages:
            continue

        title = (
            conv.get("id")
            or conv.get("conversation_id")
            or f"sharegpt_{i}"
        )
        conversations.append({
            "title": str(title),
            "messages": messages,
            "source": "sharegpt",
        })

    logger.info(f"ShareGPT parser: {len(conversations)}/{len(raw)} conversations parsed")
    return conversations


def run_import_pipeline(
    file_path: str | Path,
    source: str,
    skip_report: bool = False,
    resume: bool = False,
    checkpoint_dir: Path | None = None,
    progress_callback: Callable[[ImportProgress], None] | None = None,
) -> dict:
    """Run the full import pipeline: parse → normalize → import → report.

    This is the main entry point for the import feature.

    Args:
        file_path: Path to export file (.zip or .json)
        source: Source platform ("chatgpt", "claude")
        skip_report: If True, skip the report generation step
        resume: If True, attempt to resume from checkpoint
        checkpoint_dir: Directory for checkpoint files
        progress_callback: Called with ImportProgress during import

    Returns:
        Full pipeline result with import progress and optional report
    """
    path = Path(file_path)
    source = source.lower().strip()

    if source not in SUPPORTED_SOURCES:
        raise ValueError(f"Unsupported source: {source}. Supported: {', '.join(SUPPORTED_SOURCES)}")

    # Validate file
    if not path.exists():
        raise FileNotFoundError(f"Export file not found: {path}")
    suffix = path.suffix.lower()
    if suffix not in (".zip", ".json"):
        raise ValueError(f"Expected .zip or .json file, got: {path.name}")

    logger.info(f"Starting import pipeline: source={source}, file={path.name}")
    t1 = datetime.now(UTC).isoformat()

    # Stage 1: Parse
    if source == "chatgpt":
        raw_conversations = parse_chatgpt_export(path)
    elif source == "claude":
        # Use extended Claude parser (future: enhance parse_claude_export)
        from app.cognitive.retrospective import parse_claude_export

        raw_conversations_legacy = parse_claude_export(path)
        # Convert to normalizer-compatible format
        raw_conversations = [
            {
                "source": "claude",
                "source_id": c.get("source_id", "unknown"),
                "title": "Claude Conversation",
                "created_at": None,
                "messages": _legacy_text_to_messages(c.get("text", "")),
                "metadata": {},
            }
            for c in raw_conversations_legacy
        ]
    elif source == "sharegpt":
        raw_conversations = parse_sharegpt_export(path)
    else:
        raise ValueError(f"Parser not implemented for source: {source}")

    if not raw_conversations:
        return {"error": "No conversations found in export file", "progress": ImportProgress().as_dict()}

    # Stage 2: Normalize
    normalized = normalize_conversations(raw_conversations, source)

    if not normalized:
        return {"error": "No conversations remained after normalization", "progress": ImportProgress().as_dict()}

    # Stage 3: Extended bulk import (via SessionManager for full pipeline)
    import os as _os

    # Save original env vars for restoration after import
    _orig_rate_limit = _os.environ.get("PITH_SESSION_LEARN_RATE_LIMIT")
    _orig_max_insights = _os.environ.get("PITH_MAX_INSIGHTS_PER_CALL")
    _orig_bg_autolearn = _os.environ.get("PITH_FF_BACKGROUND_AUTOLEARN_ENABLED")

    try:
        # Set bulk-import-friendly env vars
        _os.environ["PITH_SESSION_LEARN_RATE_LIMIT"] = "200"
        _os.environ["PITH_MAX_INSIGHTS_PER_CALL"] = "30"
        _os.environ["PITH_FF_BACKGROUND_AUTOLEARN_ENABLED"] = "false"

        from app.session import SessionManager

        sm = SessionManager()

        import_result = extended_import_batch(
            conversations=normalized,
            session_manager=sm,
            progress_callback=progress_callback,
            checkpoint_dir=checkpoint_dir,
            resume=resume,
        )
    finally:
        # Restore original env vars
        _restore_env("PITH_SESSION_LEARN_RATE_LIMIT", _orig_rate_limit)
        _restore_env("PITH_MAX_INSIGHTS_PER_CALL", _orig_max_insights)
        _restore_env("PITH_FF_BACKGROUND_AUTOLEARN_ENABLED", _orig_bg_autolearn)

    # Stage 4: Report generation (unless skipped)
    report = None
    if not skip_report and not import_result.get("progress", {}).get("aborted"):
        t2 = datetime.now(UTC).isoformat()
        try:
            report = generate_import_report(import_result, t1=t1, t2=t2)
        except Exception as e:
            logger.warning(f"Report generation failed: {e}")
            report = {"type": "error", "summary": f"Report generation failed: {e}"}

    # MONITOR-113: Zero-concept extraction health check.
    # Alert when non-empty input produced zero concepts — indicates silent pipeline failure.
    _prog = import_result.get("progress", {})
    _prog_extracted = _prog.get("concepts_extracted", 0)
    _prog_aborted = _prog.get("aborted", False)
    if not _prog_aborted and _prog_extracted == 0 and normalized:
        logger.warning(
            "MONITOR-113: Zero concepts extracted from non-empty import "
            f"(source={source}, conversations={len(normalized)}, file={path.name}). "
            "Possible silent pipeline failure — check extraction logs."
        )

    return {
        "status": "completed" if not import_result.get("progress", {}).get("aborted") else "aborted",
        "progress": import_result.get("progress", {}),
        "report": report,
        "source": source,
        "file": str(path),
    }


def _legacy_text_to_messages(text: str) -> list[dict]:
    """Convert legacy flat text format to message list.

    Used for backward compatibility with existing parse_claude_export output.
    """
    messages = []
    current_role = None
    current_content = []

    for line in text.split("\n"):
        if line.startswith("user: "):
            if current_role and current_content:
                messages.append({"role": current_role, "content": "\n".join(current_content)})
            current_role = "user"
            current_content = [line[6:]]
        elif line.startswith("assistant: "):
            if current_role and current_content:
                messages.append({"role": current_role, "content": "\n".join(current_content)})
            current_role = "assistant"
            current_content = [line[11:]]
        elif current_role:
            current_content.append(line)

    if current_role and current_content:
        messages.append({"role": current_role, "content": "\n".join(current_content)})

    return messages
