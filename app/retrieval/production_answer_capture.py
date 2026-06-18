"""Cold-path production answer export for retrieval-policy replay evidence."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "retrieval_policy_production_answers.v1"
MAP_SCHEMA_VERSION = "retrieval_policy_answer_capture_map.v1"
VALID_SOURCES = {"conversation_turn", "session_end"}
REQUIRED_RAW_COLUMNS = {
    "session_id",
    "turn_id",
    "source",
    "assistant_response",
    "response_len",
    "content_hash",
    "captured_at",
    "purged_at",
}


class ProductionAnswerCaptureError(ValueError):
    """Raised when raw production answer evidence cannot be exported safely."""


@dataclass(frozen=True)
class ReplayAnswerMapRow:
    pair_id: str
    session_id: str
    turn_id: str
    source: str = "conversation_turn"
    trace_id: str | None = None
    latency_ms: float | None = None


def load_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""

    with path.open() as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ProductionAnswerCaptureError(f"{path}: expected a JSON object")
    return payload


def build_production_answers_payload(
    *,
    gold_payload: dict[str, Any],
    mapping_payload: dict[str, Any],
    db_path: Path,
) -> dict[str, Any]:
    """Export raw captured assistant responses as replay-compatible answers."""

    gold_pair_ids = _gold_pair_ids(gold_payload)
    rows = _load_mapping_rows(mapping_payload, gold_pair_ids=gold_pair_ids)
    answers = _load_answers_from_db(rows, db_path)
    return {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "map_schema_version": MAP_SCHEMA_VERSION,
            "assembler": "app.retrieval.production_answer_capture",
            "pair_count": len(gold_pair_ids),
            "answer_count": len(answers),
            "source_table": "raw_turn_payloads",
            "promotion_authority": "diagnostic_only_until_replay_capture_passes",
        },
        "answers": answers,
    }


def write_answers_ndjson(path: Path, payload: dict[str, Any]) -> None:
    """Write answer rows as newline-delimited JSON without rewriting a big array."""

    raw_answers = payload.get("answers")
    if not isinstance(raw_answers, list):
        raise ProductionAnswerCaptureError("answers payload must contain answers list")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for answer in raw_answers:
            fh.write(json.dumps(answer, sort_keys=True) + "\n")


def _gold_pair_ids(gold_payload: dict[str, Any]) -> set[str]:
    raw_pairs = gold_payload.get("pairs")
    if not isinstance(raw_pairs, list):
        raise ProductionAnswerCaptureError("gold payload must contain pairs list")
    pair_ids: set[str] = set()
    for idx, raw in enumerate(raw_pairs, start=1):
        if not isinstance(raw, dict):
            raise ProductionAnswerCaptureError(f"gold row {idx}: must be an object")
        pair_id = str(raw.get("id") or "").strip()
        if not pair_id:
            raise ProductionAnswerCaptureError(f"gold row {idx}: id is required")
        if pair_id in pair_ids:
            raise ProductionAnswerCaptureError(f"{pair_id}: duplicate gold pair id")
        pair_ids.add(pair_id)
    return pair_ids


def _load_mapping_rows(
    mapping_payload: dict[str, Any],
    *,
    gold_pair_ids: set[str],
) -> list[ReplayAnswerMapRow]:
    raw_rows = mapping_payload.get("rows") or mapping_payload.get("mappings")
    if not isinstance(raw_rows, list):
        raise ProductionAnswerCaptureError("mapping payload must contain rows list")
    seen: set[str] = set()
    rows: list[ReplayAnswerMapRow] = []
    for idx, raw in enumerate(raw_rows, start=1):
        if not isinstance(raw, dict):
            raise ProductionAnswerCaptureError(f"mapping row {idx}: must be an object")
        pair_id = str(raw.get("pair_id") or raw.get("id") or "").strip()
        if not pair_id:
            raise ProductionAnswerCaptureError(f"mapping row {idx}: pair_id is required")
        if pair_id not in gold_pair_ids:
            raise ProductionAnswerCaptureError(f"{pair_id}: mapping pair id not present in gold")
        if pair_id in seen:
            raise ProductionAnswerCaptureError(f"{pair_id}: duplicate mapping pair id")
        seen.add(pair_id)
        session_id = str(raw.get("session_id") or "").strip()
        turn_id = str(raw.get("turn_id") or "").strip()
        if not session_id:
            raise ProductionAnswerCaptureError(f"{pair_id}: session_id is required")
        if not turn_id:
            raise ProductionAnswerCaptureError(f"{pair_id}: turn_id is required")
        source = str(raw.get("source") or "conversation_turn").strip()
        if source not in VALID_SOURCES:
            raise ProductionAnswerCaptureError(f"{pair_id}: invalid source {source!r}")
        rows.append(
            ReplayAnswerMapRow(
                pair_id=pair_id,
                session_id=session_id,
                turn_id=turn_id,
                source=source,
                trace_id=str(raw.get("trace_id") or "").strip() or None,
                latency_ms=_optional_float(raw.get("latency_ms"), pair_id=pair_id),
            )
        )
    return rows


def _load_answers_from_db(rows: list[ReplayAnswerMapRow], db_path: Path) -> list[dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    try:
        _validate_raw_turn_schema(conn, db_path)
        return [_answer_row_from_db(conn, row, db_path) for row in rows]
    except sqlite3.Error as exc:
        raise ProductionAnswerCaptureError(f"{db_path}: unable to export raw answers: {exc}") from exc
    finally:
        conn.close()


def _validate_raw_turn_schema(conn: sqlite3.Connection, db_path: Path) -> None:
    table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='raw_turn_payloads'"
    ).fetchone()
    if table is None:
        raise ProductionAnswerCaptureError(f"{db_path}: raw_turn_payloads table is missing")
    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(raw_turn_payloads)").fetchall()}
    missing = sorted(REQUIRED_RAW_COLUMNS.difference(columns))
    if missing:
        raise ProductionAnswerCaptureError(f"{db_path}: raw_turn_payloads missing columns: {missing}")


def _answer_row_from_db(
    conn: sqlite3.Connection,
    row: ReplayAnswerMapRow,
    db_path: Path,
) -> dict[str, Any]:
    raw = conn.execute(
        """SELECT assistant_response, response_len, content_hash, captured_at, purged_at
           FROM raw_turn_payloads
           WHERE session_id=? AND turn_id=? AND source=?
           ORDER BY id DESC LIMIT 1""",
        (row.session_id, row.turn_id, row.source),
    ).fetchone()
    if raw is None:
        raise ProductionAnswerCaptureError(f"{row.pair_id}: mapped raw turn was not found in {db_path}")
    assistant_response, response_len, content_hash, captured_at, purged_at = raw
    if purged_at:
        raise ProductionAnswerCaptureError(f"{row.pair_id}: mapped raw turn has been purged")
    answer = str(assistant_response or "").strip()
    if not answer:
        raise ProductionAnswerCaptureError(f"{row.pair_id}: mapped raw turn has empty assistant response")
    answer_row: dict[str, Any] = {
        "pair_id": row.pair_id,
        "answer": answer,
        "trace_id": row.trace_id or row.turn_id,
        "session_id": row.session_id,
        "turn_id": row.turn_id,
        "source": row.source,
        "captured_at": captured_at,
        "response_len": int(response_len or len(answer)),
        "content_hash": str(content_hash or ""),
    }
    if row.latency_ms is not None:
        answer_row["latency_ms"] = row.latency_ms
    return answer_row


def _optional_float(value: Any, *, pair_id: str) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ProductionAnswerCaptureError(f"{pair_id}: latency_ms must be numeric") from exc
