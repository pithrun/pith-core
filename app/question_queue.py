"""Question queue management."""

import json
import logging
import tempfile
from pathlib import Path

from app.models import Question
from app.profile import resolve_data_dir

logger = logging.getLogger(__name__)

QUEUE_PATH = resolve_data_dir() / "questions.json"
MAX_QUEUE_SIZE = 500


def load_queue() -> list[dict]:
    """Load question queue from disk."""
    if not QUEUE_PATH.exists():
        return []

    try:
        with open(QUEUE_PATH) as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []


def save_queue(queue: list[dict]) -> None:
    """Save question queue to disk with cap enforcement + atomic write."""
    QUEUE_PATH.parent.mkdir(exist_ok=True, parents=True)

    # Cap enforcement: keep highest-priority questions
    if len(queue) > MAX_QUEUE_SIZE:
        queue.sort(key=lambda q: q.get("priority", 0), reverse=True)
        trimmed = len(queue) - MAX_QUEUE_SIZE
        queue = queue[:MAX_QUEUE_SIZE]
        logger.info("Question queue capped: trimmed %d low-priority entries", trimmed)

    # Atomic write: write to temp file, then rename
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(dir=str(QUEUE_PATH.parent), suffix=".tmp", prefix="questions_")
        with open(fd, "w") as f:
            json.dump(queue, f, indent=2)
        Path(tmp_path).replace(QUEUE_PATH)
    except Exception:
        # Clean up temp file on failure
        if tmp_path is not None:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass
        raise


def add_question(question: Question) -> None:
    """Add question to queue."""
    queue = load_queue()

    # Avoid duplicates
    for existing in queue:
        if existing.get("concept_id") == question.concept_id:
            # Update existing question
            existing.update(question.model_dump())
            save_queue(queue)
            return

    # Add new question
    queue.append(question.model_dump())
    save_queue(queue)


def add_questions(questions: list[Question]) -> None:
    """Add multiple questions to queue (batch — single load/save cycle)."""
    queue = load_queue()
    existing_ids = {q.get("concept_id") for q in queue}

    for question in questions:
        if question.concept_id in existing_ids:
            # Update existing
            for i, existing in enumerate(queue):
                if existing.get("concept_id") == question.concept_id:
                    queue[i] = question.model_dump()
                    break
        else:
            queue.append(question.model_dump())
            existing_ids.add(question.concept_id)

    save_queue(queue)


def get_questions(limit: int = 10) -> list[dict]:
    """Get top questions from queue."""
    queue = load_queue()

    # Sort by priority
    queue.sort(key=lambda q: q.get("priority", 0), reverse=True)

    return queue[:limit]


def remove_question(concept_id: str) -> None:
    """Remove question for concept from queue."""
    queue = load_queue()
    queue = [q for q in queue if q.get("concept_id") != concept_id]
    save_queue(queue)
