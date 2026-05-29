"""Logging configuration for Pith."""

import logging
import logging.handlers
import os
from pathlib import Path

from app.core.profile import resolve_data_dir
from app.core.runtime_identity import RuntimeIdentityLogFilter

# Log level from environment
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def setup_logging():
    """Configure logging for the application."""
    # Resolve log path profile-awaredly at call time (not import time).
    # LOG_DIR env var allows explicit override; default is profile data dir.
    # NOTE: LOG_FILE (module-level constant) is replaced by log_file (local var)
    # throughout this function — affects RotatingFileHandler arg and startup log.
    _log_dir_override = os.getenv("LOG_DIR")
    if _log_dir_override:
        logs_dir = Path(_log_dir_override)
    else:
        logs_dir = resolve_data_dir() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / os.getenv("LOG_FILE", "pith.log")

    # Create formatters
    runtime_filter = RuntimeIdentityLogFilter()
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - "
        "[%(filename)s:%(lineno)d] - "
        "pid=%(pid)s role=%(runtime_role)s profile=%(profile)s "
        "data_dir=%(data_dir)s commit=%(git_commit)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    simple_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler (INFO and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    console_handler.addFilter(runtime_filter)
    root_logger.addHandler(console_handler)

    # File handler (all levels)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    file_handler.addFilter(runtime_filter)
    root_logger.addHandler(file_handler)

    # Set levels for noisy libraries
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Log startup
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - Level: {LOG_LEVEL}, File: {log_file}")

    return root_logger
