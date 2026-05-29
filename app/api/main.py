"""Entry point shim — uvicorn expects app.main:app."""
from app.api.server import app  # noqa: F401
