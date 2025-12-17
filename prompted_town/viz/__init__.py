"""
Visualization module for Prompted Town.

This module provides visualization tools for:
- Training progress (rewards, episode lengths)
- Entity interaction graphs (conversations, recruitments)
- Agent locations and movements
- Q-value analysis
"""

from .training_plots import (
    plot_training_rewards,
    plot_episode_lengths,
    plot_recruitment_progress,
    plot_training_summary,
    create_training_dashboard,
)

from .interaction_graph import (
    InteractionTracker,
    create_interaction_graph,
    plot_interaction_graph,
    plot_trust_heatmap,
)

from .live_viz import (
    LiveTrainingVisualizer,
    create_episode_callback,
)

__all__ = [
    # Training plots
    "plot_training_rewards",
    "plot_episode_lengths",
    "plot_recruitment_progress",
    "plot_training_summary",
    "create_training_dashboard",
    # Interaction graph
    "InteractionTracker",
    "create_interaction_graph",
    "plot_interaction_graph",
    "plot_trust_heatmap",
    # Live visualization
    "LiveTrainingVisualizer",
    "create_episode_callback",
]
