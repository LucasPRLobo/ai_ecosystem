"""
Entry point for running the web UI.

Run with:
    python -m prompted_town.web
"""

from .server import run_server

if __name__ == "__main__":
    run_server()
