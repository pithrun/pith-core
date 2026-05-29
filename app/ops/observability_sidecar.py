"""In-memory observability sidecar counters for STABILITY-048 Segment 1."""

from __future__ import annotations

import threading
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

from app.storage.operation_classes import utc_now_iso


@dataclass
class ObservabilitySidecar:
    max_events: int = 200
    counters: Counter[str] = field(default_factory=Counter)
    _events: deque[dict[str, Any]] = field(init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self) -> None:
        self._events = deque(maxlen=max(1, int(self.max_events)))

    def record(self, name: str, value: int = 1, labels: dict[str, Any] | None = None) -> None:
        with self._lock:
            self.counters[name] += int(value)
            self._events.appendleft(
                {
                    "timestamp": utc_now_iso(),
                    "name": name,
                    "value": int(value),
                    "labels": dict(labels or {}),
                }
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "available": True,
                "counters": dict(self.counters),
                "recent": list(self._events),
            }

    def reset(self) -> None:
        with self._lock:
            self.counters.clear()
            self._events.clear()


observability_sidecar = ObservabilitySidecar()
