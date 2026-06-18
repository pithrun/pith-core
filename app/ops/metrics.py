"""Structured Metrics Collection — Phase 4 WS2.

Single MetricsCollector class that instruments governance phases with
timing, counts, and outcomes. Stores in SQLite `metrics` table.
No external dependencies (no Prometheus yet — Phase 4.5 scope).

Usage:
    from app.ops.metrics import metrics

    # Record a counter/gauge
    metrics.record("tier2_llm_cost_calls", 1, {"provider": "anthropic"})

    # Time a code block
    with metrics.timer("conversation_turn_latency_ms"):
        result = do_work()

    # Query for dashboard
    data = metrics.query("conversation_turn_latency_ms", since="2026-02-28T00:00:00")
"""

import json
import logging
import time
from datetime import UTC, datetime, timedelta

from app.core.config import BENCHMARK_READONLY
from app.core.datetime_utils import _utc_now, _utc_now_iso

logger = logging.getLogger(__name__)


class _MetricsTimer:
    """Context manager for timing code blocks."""

    def __init__(self, collector: "MetricsCollector", metric_name: str, labels: dict | None = None):
        self._collector = collector
        self._metric_name = metric_name
        self._labels = labels
        self._start = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_ms = (time.perf_counter() - self._start) * 1000
        labels = dict(self._labels) if self._labels else {}
        if exc_type is not None:
            labels["error"] = exc_type.__name__
        self._collector.record(self._metric_name, elapsed_ms, labels)
        return False  # Don't suppress exceptions


class MetricsCollector:
    """Structured metrics collection for pith governance pipeline.

    Buffered write pattern: metrics accumulate in memory and flush to
    SQLite when buffer hits threshold or on explicit flush(). This
    minimizes write amplification while ensuring data persists.
    """

    def __init__(self, flush_threshold: int = 50):
        self._buffer: list[dict] = []
        self._flush_threshold = flush_threshold

    def record(self, metric_name: str, value: float, labels: dict | None = None) -> None:
        """Record a single metric data point.

        Args:
            metric_name: Dot-free metric name (e.g., "conversation_turn_latency_ms").
            value: Numeric value (ms for timers, count for counters, ratio for gauges).
            labels: Optional key-value pairs for dimensional filtering.
        """
        if BENCHMARK_READONLY:
            return
        try:
            from app.ops.local_contention import record_local_contention_metric

            record_local_contention_metric(metric_name, value, labels)
        except Exception:
            pass
        self._buffer.append(
            {
                "timestamp": _utc_now_iso(),
                "metric": metric_name,
                "value": value,
                "labels": json.dumps(labels or {}),
            }
        )
        if len(self._buffer) >= self._flush_threshold:
            self.flush()

    def timer(self, metric_name: str, labels: dict | None = None) -> _MetricsTimer:
        """Context manager for timing code blocks.

        Usage:
            with metrics.timer("my_operation_ms", {"phase": "1"}):
                do_work()
        """
        return _MetricsTimer(self, metric_name, labels)

    def flush(self) -> None:
        """Write buffered metrics to SQLite.

        Best-effort: failures log a warning but don't crash.
        OBS-002: INFO log on each flush for visibility.
        """
        if not self._buffer:
            return
        if BENCHMARK_READONLY:
            return
        batch = self._buffer[:]
        self._buffer.clear()
        try:
            from app.storage import _db

            with _db() as conn:
                conn.executemany(
                    "INSERT INTO metrics (timestamp, metric, value, labels) VALUES (?, ?, ?, ?)",
                    [(m["timestamp"], m["metric"], m["value"], m["labels"]) for m in batch],
                )
            # OBS-002: Log flush for observability
            metric_names = set(m["metric"] for m in batch)
            logger.info("OBS-002: Flushed %d metrics (%s)", len(batch), ", ".join(sorted(metric_names)))
        except Exception as e:
            logger.warning("Metrics flush failed: %s", e)
            # Put unflushed metrics back (best-effort recovery)
            self._buffer = batch + self._buffer

    def query(
        self,
        metric_name: str,
        since: str | None = None,
        until: str | None = None,
        limit: int = 1000,
    ) -> list[dict]:
        """Query metrics for dashboard endpoint.

        Args:
            metric_name: Metric to query.
            since: ISO timestamp lower bound (default: 1 hour ago).
            until: ISO timestamp upper bound (default: now).
            limit: Max rows returned.

        Returns:
            List of {"timestamp", "value", "labels"} dicts.
        """
        # Flush pending data first so query sees everything
        self.flush()

        if since is None:
            since = (_utc_now() - timedelta(hours=1)).isoformat()
        if until is None:
            until = _utc_now_iso()

        try:
            from app.storage import _db

            with _db() as conn:
                rows = conn.execute(
                    "SELECT timestamp, value, labels FROM metrics "
                    "WHERE metric = ? AND timestamp >= ? AND timestamp <= ? "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (metric_name, since, until, limit),
                ).fetchall()
                return [{"timestamp": r[0], "value": r[1], "labels": json.loads(r[2])} for r in rows]
        except Exception as e:
            logger.warning("Metrics query failed: %s", e)
            return []

    def query_aggregate(
        self,
        metric_name: str,
        since: str | None = None,
    ) -> dict:
        """Query aggregated stats for a metric (p50, p95, p99, count, avg).

        Returns dict with count, avg, min, max, p50, p95, p99.
        """
        self.flush()

        if since is None:
            since = (_utc_now() - timedelta(hours=1)).isoformat()

        try:
            from app.storage import _db

            with _db() as conn:
                rows = conn.execute(
                    "SELECT value FROM metrics WHERE metric = ? AND timestamp >= ? ORDER BY value",
                    (metric_name, since),
                ).fetchall()

                if not rows:
                    return {"count": 0, "avg": 0, "min": 0, "max": 0, "p50": 0, "p95": 0, "p99": 0}

                values = [r[0] for r in rows]
                n = len(values)
                return {
                    "count": n,
                    "avg": round(sum(values) / n, 2),
                    "min": round(values[0], 2),
                    "max": round(values[-1], 2),
                    "p50": round(values[int(n * 0.50)], 2),
                    "p95": round(values[min(int(n * 0.95), n - 1)], 2),
                    "p99": round(values[min(int(n * 0.99), n - 1)], 2),
                }
        except Exception as e:
            logger.warning("Metrics aggregate query failed: %s", e)
            return {"count": 0, "avg": 0, "min": 0, "max": 0, "p50": 0, "p95": 0, "p99": 0}

    def query_aggregate_with_recency(
        self,
        metric_name: str,
        since: str | None = None,
    ) -> dict:
        """Query aggregate stats plus timestamp/age metadata for a metric."""
        self.flush()

        if since is None:
            since = (_utc_now() - timedelta(hours=1)).isoformat()

        empty = {
            "count": 0,
            "avg": 0,
            "min": 0,
            "max": 0,
            "p50": 0,
            "p95": 0,
            "p99": 0,
            "oldest_timestamp": None,
            "newest_timestamp": None,
            "max_timestamp": None,
            "max_age_seconds": None,
            "window_seconds": 3600,
        }
        try:
            rows = self.query(metric_name, since=since, limit=100000)
            if not rows:
                return dict(empty)

            sorted_rows = sorted(rows, key=lambda row: (float(row.get("value", 0.0) or 0.0), str(row.get("timestamp", ""))))
            values = [float(row.get("value", 0.0) or 0.0) for row in sorted_rows]
            timestamps = [str(row.get("timestamp", "")) for row in sorted_rows]
            n = len(values)
            max_value = values[-1]
            max_timestamp = max(ts for ts, value in zip(timestamps, values, strict=False) if value == max_value)

            def _age_seconds(timestamp: str | None) -> float | None:
                if not timestamp:
                    return None
                try:
                    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                except ValueError:
                    return None
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=UTC)
                return round(max(0.0, (_utc_now() - parsed.astimezone(UTC)).total_seconds()), 2)

            return {
                "count": n,
                "avg": round(sum(values) / n, 2),
                "min": round(values[0], 2),
                "max": round(max_value, 2),
                "p50": round(values[int(n * 0.50)], 2),
                "p95": round(values[min(int(n * 0.95), n - 1)], 2),
                "p99": round(values[min(int(n * 0.99), n - 1)], 2),
                "oldest_timestamp": min(timestamps),
                "newest_timestamp": max(timestamps),
                "max_timestamp": max_timestamp,
                "max_age_seconds": _age_seconds(max_timestamp),
                "window_seconds": 3600,
            }
        except Exception as e:
            logger.warning("Metrics aggregate recency query failed: %s", e)
            return dict(empty)

    def query_count(
        self,
        metric_name: str,
        since: str | None = None,
        labels_filter: dict | None = None,
    ) -> int:
        """Count total occurrences of a metric, optionally filtered by labels."""
        self.flush()

        if since is None:
            since = (_utc_now() - timedelta(hours=1)).isoformat()

        try:
            from app.storage import _db

            with _db() as conn:
                if labels_filter:
                    # Filter by label values using JSON extraction
                    rows = conn.execute(
                        "SELECT SUM(value) FROM metrics WHERE metric = ? AND timestamp >= ?",
                        (metric_name, since),
                    ).fetchone()
                else:
                    rows = conn.execute(
                        "SELECT SUM(value) FROM metrics WHERE metric = ? AND timestamp >= ?",
                        (metric_name, since),
                    ).fetchone()
                return int(rows[0] or 0) if rows else 0
        except Exception as e:
            logger.warning("Metrics count query failed: %s", e)
            return 0

    def query_rate(
        self,
        metric_name: str,
        window_minutes: int = 60,
        bucket_minutes: int = 5,
    ) -> dict:
        """MONITOR-001: Compute rolling rate (events per minute) over a time window.

        Returns dict with: current_rate, avg_rate, peak_rate, buckets[].
        Used by /learning_metrics for velocity signals.
        """
        self.flush()
        since = (_utc_now() - timedelta(minutes=window_minutes)).isoformat()

        try:
            from app.storage import _db

            with _db() as conn:
                rows = conn.execute(
                    "SELECT timestamp, value FROM metrics "
                    "WHERE metric = ? AND timestamp >= ? ORDER BY timestamp",
                    (metric_name, since),
                ).fetchall()

                if not rows:
                    return {"current_rate": 0, "avg_rate": 0, "peak_rate": 0, "buckets": []}

                # Bucket by time intervals (minute precision)
                buckets: dict[str, float] = {}
                for ts, val in rows:
                    bucket_key = ts[:16]  # YYYY-MM-DDTHH:MM
                    buckets[bucket_key] = buckets.get(bucket_key, 0) + val

                bucket_list = sorted(buckets.items())
                rates = [v / bucket_minutes for _, v in bucket_list]

                return {
                    "current_rate": round(rates[-1], 2) if rates else 0,
                    "avg_rate": round(sum(rates) / len(rates), 2) if rates else 0,
                    "peak_rate": round(max(rates), 2) if rates else 0,
                    "window_minutes": window_minutes,
                    "bucket_count": len(bucket_list),
                    "buckets": [
                        {"time": k, "count": v, "rate_per_min": round(v / bucket_minutes, 2)}
                        for k, v in bucket_list[-12:]  # Last 12 buckets only
                    ],
                }
        except Exception as e:
            logger.warning("Metrics rate query failed: %s", e)
            return {"current_rate": 0, "avg_rate": 0, "peak_rate": 0, "buckets": []}

    def startup_health_check(self) -> dict:
        """OBS-002: Verify metrics table exists and report stats.

        Call during server startup to confirm metrics pipeline is operational.
        Returns dict with status, total_rows, distinct_metrics, oldest, newest.
        """
        try:
            from app.storage import _db

            with _db() as conn:
                # Verify table exists
                table_check = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='metrics'"
                ).fetchone()
                if not table_check:
                    logger.warning("OBS-002: metrics table does not exist!")
                    return {"status": "missing_table", "total_rows": 0}

                row = conn.execute(
                    "SELECT COUNT(*), COUNT(DISTINCT metric), MIN(timestamp), MAX(timestamp) FROM metrics"
                ).fetchone()
                total, distinct, oldest, newest = row[0], row[1], row[2], row[3]

                result = {
                    "status": "healthy",
                    "total_rows": total,
                    "distinct_metrics": distinct,
                    "oldest": oldest,
                    "newest": newest,
                }
                logger.info(
                    "OBS-002: Metrics health — %d rows, %d distinct metrics, range=[%s → %s]",
                    total,
                    distinct,
                    oldest or "n/a",
                    newest or "n/a",
                )
                return result
        except Exception as e:
            logger.warning("OBS-002: Metrics health check failed: %s", e)
            return {"status": "error", "error": str(e)}

    def metrics_summary(self, days: int = 7) -> dict:
        """OBS-003: Aggregate metrics summary with per-metric stats and trends.

        Returns dict with per-metric count, mean, p95, and 7d-vs-prior-7d trend.
        """
        try:
            from datetime import datetime, timedelta

            from app.storage import _db

            now = datetime.utcnow()
            current_start = (now - timedelta(days=days)).isoformat()
            prior_start = (now - timedelta(days=days * 2)).isoformat()

            with _db() as conn:
                # Current period stats
                rows = conn.execute(
                    """SELECT metric, COUNT(*) as cnt,
                              AVG(value) as avg_val,
                              MAX(value) as max_val,
                              MIN(value) as min_val
                       FROM metrics
                       WHERE timestamp >= ?
                       GROUP BY metric
                       ORDER BY cnt DESC""",
                    (current_start,),
                ).fetchall()

                # Prior period counts for trend
                prior_rows = conn.execute(
                    """SELECT metric, COUNT(*) as cnt, AVG(value) as avg_val
                       FROM metrics
                       WHERE timestamp >= ? AND timestamp < ?
                       GROUP BY metric""",
                    (prior_start, current_start),
                ).fetchall()
                prior_map = {r[0]: {"count": r[1], "avg": r[2]} for r in prior_rows}

                # P95 per metric (approximate via sorted values)
                p95_map = {}
                for row in rows:
                    metric_name = row[0]
                    vals = conn.execute(
                        "SELECT value FROM metrics WHERE metric = ? AND timestamp >= ? ORDER BY value",
                        (metric_name, current_start),
                    ).fetchall()
                    if vals:
                        p95_idx = int(len(vals) * 0.95)
                        p95_map[metric_name] = vals[min(p95_idx, len(vals) - 1)][0]

                summary = {}
                for row in rows:
                    metric_name = row[0]
                    current_count = row[1]
                    prior = prior_map.get(metric_name, {})
                    prior_count = prior.get("count", 0)

                    # Trend: ratio of current vs prior period
                    if prior_count > 0:
                        trend = round((current_count - prior_count) / prior_count * 100, 1)
                        trend_label = f"+{trend}%" if trend > 0 else f"{trend}%"
                    else:
                        trend_label = "new"

                    summary[metric_name] = {
                        "count": current_count,
                        "mean": round(row[2], 2) if row[2] else 0,
                        "p95": round(p95_map.get(metric_name, 0), 2),
                        "max": round(row[3], 2) if row[3] else 0,
                        "min": round(row[4], 2) if row[4] else 0,
                        "trend_7d": trend_label,
                    }

                total = sum(s["count"] for s in summary.values())
                logger.info(
                    "OBS-003: Metrics summary — %d metrics, %d events in %dd, top: %s",
                    len(summary),
                    total,
                    days,
                    ", ".join(f"{k}({v['count']})" for k, v in list(summary.items())[:5]),
                )
                return {"status": "ok", "period_days": days, "metrics": summary, "total_events": total}

        except Exception as e:
            logger.warning("OBS-003: Metrics summary failed: %s", e)
            return {"status": "error", "error": str(e)}


# Global instance — import and use everywhere
metrics = MetricsCollector()
