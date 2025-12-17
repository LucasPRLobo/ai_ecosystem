"""
Web UI for Prompted Town.

Browser-based interface with real-time updates.

Run with:
    python -m prompted_town.web
"""

from .server import create_app, run_server

__all__ = ["create_app", "run_server"]
