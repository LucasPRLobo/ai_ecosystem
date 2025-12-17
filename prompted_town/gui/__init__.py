"""
Desktop GUI for Prompted Town.

This module provides a tkinter-based graphical interface for:
- Viewing the simulation state
- Watching agent interactions
- Running training sessions
- Interactive play mode
"""

from .app import PromptedTownApp, run_app

__all__ = [
    "PromptedTownApp",
    "run_app",
]
