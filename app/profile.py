"""Compatibility shim for legacy imports.

The canonical profile resolver lives in app.core.profile. Older modules still
import app.profile, especially startup migrations that run very early in boot.
"""

from app.core.profile import *  # noqa: F401,F403
