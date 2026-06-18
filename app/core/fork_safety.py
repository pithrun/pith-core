"""Fork-safety guard helpers for long-lived API processes."""

from __future__ import annotations

import os
import sys

_DEFAULT_FORK_SENSITIVE_PREFIXES = (
    "numpy",
    "scipy",
    "sklearn",
    "torch",
    "sentence_transformers",
)
_FALSE_VALUES = {"0", "false", "no", "off"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in _FALSE_VALUES


def fork_safety_guard_enabled() -> bool:
    return _env_bool("PITH_FORK_SAFETY_OPTIONAL_SUBPROCESS_GUARD_ENABLED", True)


def fork_sensitive_module_prefixes() -> tuple[str, ...]:
    raw = os.environ.get("PITH_FORK_SAFETY_MODULE_PREFIXES", "")
    if not raw.strip():
        return _DEFAULT_FORK_SENSITIVE_PREFIXES
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def loaded_fork_sensitive_modules() -> tuple[str, ...]:
    prefixes = fork_sensitive_module_prefixes()
    loaded: set[str] = set()
    for module_name in sys.modules:
        for prefix in prefixes:
            if module_name == prefix or module_name.startswith(f"{prefix}."):
                loaded.add(module_name)
                break
    return tuple(sorted(loaded))


def should_suppress_optional_subprocess(reason: str = "") -> bool:
    _ = reason
    return fork_safety_guard_enabled() and bool(loaded_fork_sensitive_modules())
