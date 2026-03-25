"""Entry point shim — uvicorn expects app.main:app."""
from app.server import app  # noqa: F401
