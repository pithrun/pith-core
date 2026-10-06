"""Terminable process boundary for deadline-bound foreground query encoding."""

from __future__ import annotations

import hashlib
import logging
import multiprocessing
import os
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.storage.embedding import EMBEDDING_DIM, MODEL_NAME

logger = logging.getLogger(__name__)

MAX_QUERY_BYTES = 65_536
STARTUP_TIMEOUT_S = 15.0
JOIN_TIMEOUT_S = 0.1
RESTART_BACKOFF_MIN_S = 0.25
RESTART_BACKOFF_MAX_S = 5.0


@dataclass(frozen=True)
class ForegroundEmbeddingResult:
    outcome: str
    vector: np.ndarray | None
    elapsed_ms: float
    encode_ms: float | None = None
    request_hash: str | None = None


def _encoder_process_main(connection, model_name: str, device: str | None) -> None:
    """Child entrypoint. It owns only the sentence-transformer query encoder."""
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, device=device) if device else SentenceTransformer(model_name)
        connection.send({"type": "ready"})
        while True:
            message = connection.recv()
            if not isinstance(message, dict):
                connection.send({"type": "error", "error_type": "invalid_message"})
                continue
            if message.get("type") == "shutdown":
                return
            if message.get("type") != "encode":
                connection.send({"type": "error", "error_type": "invalid_message"})
                continue
            request_id = str(message.get("request_id") or "")[:64]
            text = message.get("text")
            if not request_id or not isinstance(text, str):
                connection.send(
                    {
                        "type": "error",
                        "request_id": request_id,
                        "error_type": "invalid_request",
                    }
                )
                continue
            started = time.perf_counter()
            vector = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
            vector = np.asarray(vector, dtype=np.float32)
            connection.send(
                {
                    "type": "result",
                    "request_id": request_id,
                    "dtype": "float32",
                    "shape": tuple(vector.shape),
                    "vector": vector.tobytes(),
                    "encode_ms": round((time.perf_counter() - started) * 1000.0, 2),
                }
            )
    except EOFError:
        return
    except BaseException as exc:
        try:
            connection.send({"type": "error", "error_type": type(exc).__name__})
        except Exception:
            pass
    finally:
        try:
            connection.close()
        except Exception:
            pass


class ForegroundEmbeddingProcess:
    """Capacity-one spawned encoder with bounded wait and process recycling."""

    def __init__(
        self,
        *,
        context: Any | None = None,
        process_target: Callable[..., None] = _encoder_process_main,
    ) -> None:
        self._context = context or multiprocessing.get_context("spawn")
        self._process_target = process_target
        self._lock = threading.Lock()
        self._capacity = threading.Lock()
        self._process = None
        self._connection = None
        self._starting_process = None
        self._starting_connection = None
        self._state = "cold"
        self._starter: threading.Thread | None = None
        self._restart_timer: threading.Timer | None = None
        self._shutdown = False
        self._consecutive_start_failures = 0
        self._next_start_monotonic = 0.0
        self._counters: dict[str, int] = {
            "starts": 0,
            "restarts": 0,
            "terminations": 0,
            "timeouts": 0,
            "busy": 0,
            "errors": 0,
            "invalid_vectors": 0,
            "completed": 0,
        }

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            process = self._process or self._starting_process
            return {
                "state": self._state,
                "pid": getattr(process, "pid", None),
                "alive": bool(process is not None and process.is_alive()),
                "counters": dict(self._counters),
            }

    def start_async(self) -> bool:
        with self._lock:
            if self._shutdown:
                return False
            if self._state in {"starting", "ready", "busy"}:
                return True
            if time.monotonic() < self._next_start_monotonic:
                self._state = "cold" if self._shutdown else "backoff"
                return False
            self._state = "starting"
            self._starter = threading.Thread(
                target=self._start_worker,
                name="pith-foreground-embedding-starter",
                daemon=True,
            )
            self._starter.start()
            return True

    def _start_worker(self) -> None:
        parent_connection = None
        child_connection = None
        process = None
        try:
            parent_connection, child_connection = self._context.Pipe(duplex=True)
            device = os.environ.get("PITH_EMBEDDING_DEVICE", "").strip() or None
            process = self._context.Process(
                target=self._process_target,
                args=(child_connection, MODEL_NAME, device),
                daemon=True,
                name="pith-foreground-embedding",
            )
            process.start()
            child_connection.close()
            child_connection = None
            with self._lock:
                if self._shutdown:
                    raise RuntimeError("encoder shutdown during startup")
                self._starting_process = process
                self._starting_connection = parent_connection
            if not parent_connection.poll(STARTUP_TIMEOUT_S):
                raise TimeoutError("encoder startup timeout")
            message = parent_connection.recv()
            if not isinstance(message, dict) or message.get("type") != "ready":
                raise RuntimeError("encoder startup failed")
            with self._lock:
                if self._shutdown:
                    raise RuntimeError("encoder shutdown during startup")
                self._process = process
                self._connection = parent_connection
                self._starting_process = None
                self._starting_connection = None
                self._state = "ready"
                self._consecutive_start_failures = 0
                self._counters["starts"] += 1
            return
        except Exception as exc:
            logger.warning("foreground embedding worker startup failed: %s", type(exc).__name__)
            if child_connection is not None:
                try:
                    child_connection.close()
                except Exception:
                    pass
            self._terminate_handles(process, parent_connection)
            with self._lock:
                if self._starting_process is process:
                    self._starting_process = None
                    self._starting_connection = None
                self._consecutive_start_failures += 1
                self._counters["errors"] += 1
                self._state = "backoff"
                delay = min(
                    RESTART_BACKOFF_MAX_S,
                    RESTART_BACKOFF_MIN_S * (2 ** (self._consecutive_start_failures - 1)),
                )
                self._next_start_monotonic = time.monotonic() + delay
                should_restart = not self._shutdown
            if should_restart:
                self._schedule_restart(delay)

    def _schedule_restart(self, delay_s: float) -> None:
        with self._lock:
            if self._shutdown or (self._restart_timer and self._restart_timer.is_alive()):
                return
            timer = threading.Timer(delay_s, self.start_async)
            timer.daemon = True
            self._restart_timer = timer
            self._counters["restarts"] += 1
            timer.start()

    @staticmethod
    def _terminate_handles(process, connection) -> None:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass
        if process is None:
            return
        try:
            if process.is_alive():
                process.terminate()
            process.join(JOIN_TIMEOUT_S)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(JOIN_TIMEOUT_S)
        except Exception:
            pass

    def _recycle_worker(self, outcome: str) -> None:
        with self._lock:
            process, connection = self._process, self._connection
            self._process = None
            self._connection = None
            self._state = "cold" if self._shutdown else "backoff"
            self._next_start_monotonic = time.monotonic() + RESTART_BACKOFF_MIN_S
            self._counters["terminations"] += 1
            if outcome == "timeout":
                self._counters["timeouts"] += 1
            elif outcome == "invalid_vector":
                self._counters["invalid_vectors"] += 1
            else:
                self._counters["errors"] += 1
            should_restart = not self._shutdown
        self._terminate_handles(process, connection)
        if should_restart:
            self._schedule_restart(RESTART_BACKOFF_MIN_S)

    @staticmethod
    def _send_bounded(connection, message: dict[str, Any], deadline: float) -> bool:
        completed = threading.Event()
        failed: list[BaseException] = []

        def _send() -> None:
            try:
                connection.send(message)
            except BaseException as exc:
                failed.append(exc)
            finally:
                completed.set()

        sender = threading.Thread(
            target=_send,
            name="pith-foreground-embedding-send",
            daemon=True,
        )
        sender.start()
        completed.wait(max(0.0, deadline - time.monotonic()))
        return completed.is_set() and not failed

    def encode(self, text: str, *, timeout_ms: float) -> ForegroundEmbeddingResult:
        started = time.perf_counter()
        if not isinstance(text, str):
            return ForegroundEmbeddingResult("error", None, 0.0)
        encoded = text.encode("utf-8")
        request_hash = hashlib.sha256(encoded).hexdigest()[:16]
        if len(encoded) > MAX_QUERY_BYTES or timeout_ms <= 0:
            return ForegroundEmbeddingResult("error", None, 0.0, request_hash=request_hash)
        if not self._capacity.acquire(blocking=False):
            with self._lock:
                self._counters["busy"] += 1
            return ForegroundEmbeddingResult("busy", None, 0.0, request_hash=request_hash)

        try:
            with self._lock:
                state = self._state
                process = self._process
                connection = self._connection
            if state != "ready" or process is None or connection is None or not process.is_alive():
                if state == "ready":
                    with self._lock:
                        self._state = "dead"
                        self._process = None
                        self._connection = None
                    self._terminate_handles(process, connection)
                    state = "cold"
                self.start_async()
                outcome = state if state in {"starting", "backoff"} else "cold"
                return ForegroundEmbeddingResult(outcome, None, 0.0, request_hash=request_hash)

            request_id = uuid.uuid4().hex
            deadline = time.monotonic() + (timeout_ms / 1000.0)
            with self._lock:
                if self._process is process and self._state == "ready":
                    self._state = "busy"
            try:
                sent = self._send_bounded(
                    connection,
                    {"type": "encode", "request_id": request_id, "text": text},
                    deadline,
                )
                if not sent:
                    self._recycle_worker("timeout")
                    return ForegroundEmbeddingResult(
                        "timeout",
                        None,
                        round((time.perf_counter() - started) * 1000.0, 2),
                        request_hash=request_hash,
                    )
                remaining = max(0.0, deadline - time.monotonic())
                if not connection.poll(remaining):
                    self._recycle_worker("timeout")
                    return ForegroundEmbeddingResult(
                        "timeout",
                        None,
                        round((time.perf_counter() - started) * 1000.0, 2),
                        request_hash=request_hash,
                    )
                message = connection.recv()
            except Exception:
                self._recycle_worker("error")
                return ForegroundEmbeddingResult(
                    "error",
                    None,
                    round((time.perf_counter() - started) * 1000.0, 2),
                    request_hash=request_hash,
                )

            if isinstance(message, dict) and message.get("type") == "error":
                self._recycle_worker("error")
                return ForegroundEmbeddingResult(
                    "error",
                    None,
                    round((time.perf_counter() - started) * 1000.0, 2),
                    request_hash=request_hash,
                )
            vector = self._validated_vector(message, request_id)
            if vector is None:
                self._recycle_worker("invalid_vector")
                return ForegroundEmbeddingResult(
                    "invalid_vector",
                    None,
                    round((time.perf_counter() - started) * 1000.0, 2),
                    request_hash=request_hash,
                )
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
            with self._lock:
                self._counters["completed"] += 1
            return ForegroundEmbeddingResult(
                "completed",
                vector,
                elapsed_ms,
                encode_ms=float(message.get("encode_ms")) if message.get("encode_ms") is not None else None,
                request_hash=request_hash,
            )
        finally:
            with self._lock:
                if self._process is process and self._state == "busy":
                    self._state = "ready"
            self._capacity.release()

    @staticmethod
    def _validated_vector(message: Any, request_id: str) -> np.ndarray | None:
        if not isinstance(message, dict) or message.get("type") != "result":
            return None
        if message.get("request_id") != request_id:
            return None
        if message.get("dtype") != "float32" or tuple(message.get("shape") or ()) != (EMBEDDING_DIM,):
            return None
        payload = message.get("vector")
        if not isinstance(payload, bytes) or len(payload) != EMBEDDING_DIM * 4:
            return None
        vector = np.frombuffer(payload, dtype=np.float32).copy()
        if not np.isfinite(vector).all():
            return None
        norm = float(np.linalg.norm(vector))
        if norm < 0.5 or norm > 1.5:
            return None
        return vector

    def shutdown(self) -> None:
        with self._lock:
            self._shutdown = True
            timer = self._restart_timer
            process, connection = self._process, self._connection
            starting_process, starting_connection = self._starting_process, self._starting_connection
            self._process = None
            self._connection = None
            self._starting_process = None
            self._starting_connection = None
            self._state = "cold"
        if timer is not None:
            timer.cancel()
        if connection is not None:
            self._send_bounded(
                connection,
                {"type": "shutdown"},
                time.monotonic() + JOIN_TIMEOUT_S,
            )
        self._terminate_handles(process, connection)
        if starting_process is not process:
            self._terminate_handles(starting_process, starting_connection)


foreground_embedding_process = ForegroundEmbeddingProcess()
