"""Built-in maintenance scheduler — MAINT-033: Consumer Server Foundation.

Runs periodic maintenance without requiring external schedulers (launchd/cron).
Integrated with server lifecycle via startup/shutdown hooks.

Amendments incorporated:
- A2: asyncio.Lock mutex prevents scheduler + /maintenance endpoint race condition
- A4: Interval clamping (60s-86400s) + circuit breaker (5 consecutive failures → pause)
- A5: PITH_DISABLE_BUILTIN_SCHEDULER env var for operators
- A6: Module-level imports (no lazy imports)
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# --- Configuration ---
# A4: Clamp interval to [60, 86400] seconds
_raw_interval = int(os.environ.get("PITH_MAINTENANCE_INTERVAL_SECONDS", str(6 * 3600)))
MAINTENANCE_INTERVAL_SECONDS = max(60, min(86400, _raw_interval))
if _raw_interval != MAINTENANCE_INTERVAL_SECONDS:
    logger.warning(
        f"PITH_MAINTENANCE_INTERVAL_SECONDS={_raw_interval} clamped to {MAINTENANCE_INTERVAL_SECONDS} "
        f"(valid range: 60-86400)"
    )

# ARGUS-C1-F1: Clamp all scheduler config values to valid ranges
_raw_initial_delay = int(os.environ.get("PITH_MAINTENANCE_INITIAL_DELAY", "300"))
INITIAL_DELAY_SECONDS = max(0, min(3600, _raw_initial_delay))
if _raw_initial_delay != INITIAL_DELAY_SECONDS:
    logger.warning(
        "PITH_MAINTENANCE_INITIAL_DELAY=%d clamped to %d (valid range: 0-3600)",
        _raw_initial_delay, INITIAL_DELAY_SECONDS,
    )

_raw_min_concepts = int(os.environ.get("PITH_MIN_CONCEPTS_FOR_MAINTENANCE", "10"))
MIN_CONCEPTS_FOR_MAINTENANCE = max(0, min(10000, _raw_min_concepts))
if _raw_min_concepts != MIN_CONCEPTS_FOR_MAINTENANCE:
    logger.warning(
        "PITH_MIN_CONCEPTS_FOR_MAINTENANCE=%d clamped to %d (valid range: 0-10000)",
        _raw_min_concepts, MIN_CONCEPTS_FOR_MAINTENANCE,
    )

# A4: Circuit breaker — pause after N consecutive failures
MAX_CONSECUTIVE_FAILURES = 5
DEFAULT_PHASES = [1, 2, 3, 4, 5, 6]  # ARCH-D05: phase 6 = standalone promotion sweep

# A2: Shared lock prevents concurrent maintenance runs (scheduler + /maintenance endpoint)
maintenance_lock = asyncio.Lock()

# Module-level state
_scheduler_task: asyncio.Task | None = None
_consecutive_failures: int = 0
_circuit_open: bool = False


def get_scheduler_status() -> dict:
    """Return current scheduler state for /status endpoint."""
    return {
        "circuit_open": _circuit_open,
        "consecutive_failures": _consecutive_failures,
        "scheduler_running": _scheduler_task is not None and not _scheduler_task.done(),
    }


def _write_heartbeat(status: str, details: dict | None = None) -> None:
    """Write maintenance heartbeat JSON for /health/maintenance compatibility."""
    try:
        from app.profile import resolve_data_dir
        data_dir = Path(resolve_data_dir())
        heartbeat_path = data_dir / "maintenance_heartbeat.json"
        heartbeat = {
            "scheduler": "builtin",
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "interval_seconds": MAINTENANCE_INTERVAL_SECONDS,
            "circuit_open": _circuit_open,
            "consecutive_failures": _consecutive_failures,
        }
        if details:
            heartbeat.update(details)
        heartbeat_path.write_text(json.dumps(heartbeat, indent=2))
    except Exception as e:
        logger.debug(f"Heartbeat write failed (non-fatal): {e}")


async def _maintenance_loop() -> None:
    """Core maintenance loop — runs periodically until cancelled."""
    global _consecutive_failures, _circuit_open

    logger.info(
        f"Maintenance scheduler: waiting {INITIAL_DELAY_SECONDS}s initial delay "
        f"(interval={MAINTENANCE_INTERVAL_SECONDS}s, min_concepts={MIN_CONCEPTS_FOR_MAINTENANCE})"
    )
    await asyncio.sleep(INITIAL_DELAY_SECONDS)

    while True:
        try:
            # A4: Circuit breaker check
            if _circuit_open:
                logger.warning(
                    f"Maintenance scheduler: circuit OPEN after {MAX_CONSECUTIVE_FAILURES} "
                    f"consecutive failures. Paused. Set PITH_MAINTENANCE_CIRCUIT_RESET=1 to reset."
                )
                _write_heartbeat("circuit_open")
                # Check every interval if circuit should be manually reset
                if os.environ.get("PITH_MAINTENANCE_CIRCUIT_RESET", "").lower() in ("true", "1"):
                    _consecutive_failures = 0
                    _circuit_open = False
                    os.environ.pop("PITH_MAINTENANCE_CIRCUIT_RESET", None)
                    logger.info("Maintenance scheduler: circuit RESET by operator")
                else:
                    await asyncio.sleep(MAINTENANCE_INTERVAL_SECONDS)
                    continue

            # Check concept count before running
            try:
                from app.storage import _db
                with _db() as conn:
                    count = conn.execute("SELECT COUNT(*) FROM concepts WHERE is_current = 1").fetchone()[0]
                if count < MIN_CONCEPTS_FOR_MAINTENANCE:
                    logger.info(
                        f"Maintenance scheduler: skipping — {count} concepts < {MIN_CONCEPTS_FOR_MAINTENANCE} minimum"
                    )
                    _write_heartbeat("skipped_low_concepts", {"concept_count": count})
                    await asyncio.sleep(MAINTENANCE_INTERVAL_SECONDS)
                    continue
            except Exception as e:
                logger.warning(f"Maintenance scheduler: concept count check failed: {e}")
                # Proceed anyway — maintenance has its own guards

            # A2: Acquire lock to prevent concurrent runs with /maintenance endpoint
            async with maintenance_lock:
                logger.info("Maintenance scheduler: starting scheduled run")
                _write_heartbeat("running")

                from app.maintenance import run_maintenance
                report = await run_maintenance(phases=DEFAULT_PHASES)

                _consecutive_failures = 0  # Reset on success
                logger.info(
                    f"Maintenance scheduler: completed — "
                    f"phases={report.phases_completed if hasattr(report, 'phases_completed') else 'unknown'}"
                )
                _write_heartbeat("completed", {
                    "last_run": datetime.now(timezone.utc).isoformat(),
                })

        except asyncio.CancelledError:
            logger.info("Maintenance scheduler: cancelled (shutdown)")
            _write_heartbeat("cancelled")
            raise
        except Exception as e:
            _consecutive_failures += 1
            logger.error(
                f"Maintenance scheduler: run failed ({_consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}): {e}",
                exc_info=True,
            )
            _write_heartbeat("error", {"error": str(e)})
            # A4: Trip circuit breaker
            if _consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                _circuit_open = True
                logger.error(
                    f"Maintenance scheduler: circuit breaker TRIPPED after "
                    f"{MAX_CONSECUTIVE_FAILURES} consecutive failures"
                )

        await asyncio.sleep(MAINTENANCE_INTERVAL_SECONDS)


async def start_maintenance_scheduler() -> asyncio.Task | None:
    """Start the built-in maintenance scheduler as a background task.

    Returns None if scheduler is disabled via PITH_DISABLE_BUILTIN_SCHEDULER.
    """
    global _scheduler_task

    # A5: Allow operators to disable built-in scheduler
    if os.environ.get("PITH_DISABLE_BUILTIN_SCHEDULER", "").lower() in ("true", "1", "yes"):
        logger.info("Maintenance scheduler: DISABLED via PITH_DISABLE_BUILTIN_SCHEDULER")
        return None

    _scheduler_task = asyncio.create_task(
        _maintenance_loop(),
        name="pith-maintenance-scheduler",
    )
    logger.info("Maintenance scheduler: started")
    _write_heartbeat("started")
    return _scheduler_task


async def stop_maintenance_scheduler() -> None:
    """Stop the maintenance scheduler gracefully."""
    global _scheduler_task
    if _scheduler_task is not None and not _scheduler_task.done():
        _scheduler_task.cancel()
        try:
            await _scheduler_task
        except asyncio.CancelledError:
            pass
        logger.info("Maintenance scheduler: stopped")
    _scheduler_task = None
