"""
Training visualization for Prompted Town RL.

Provides plotting functions for training metrics:
- Reward curves
- Episode lengths
- Recruitment progress
- Multi-agent comparisons
"""

from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from ..rl import TrainingMetrics, EpisodeMetrics


# =============================================================================
# COLOR SCHEMES
# =============================================================================

ROLE_COLORS = {
    "farmer": "#4CAF50",  # Green
    "guard": "#2196F3",   # Blue
    "rebel": "#F44336",   # Red
}

def get_agent_color(agent_id: str) -> str:
    """Get color based on agent role."""
    if "farmer" in agent_id:
        return ROLE_COLORS["farmer"]
    elif "guard" in agent_id:
        return ROLE_COLORS["guard"]
    elif "rebel" in agent_id:
        return ROLE_COLORS["rebel"]
    return "#9E9E9E"  # Gray default


# =============================================================================
# REWARD PLOTS
# =============================================================================

def plot_training_rewards(
    metrics: TrainingMetrics,
    smoothing_window: int = 20,
    figsize: tuple = (12, 6),
    show_individual: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training rewards over episodes.

    Args:
        metrics: Training metrics with episode rewards
        smoothing_window: Window size for moving average
        figsize: Figure size
        show_individual: Show individual agent curves
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    episode_rewards = metrics.episode_rewards
    episodes = range(1, len(episode_rewards) + 1)

    # Get agent IDs from first episode
    if not episode_rewards:
        return fig

    agent_ids = list(episode_rewards[0].keys())

    # --- Left plot: Individual agent rewards ---
    ax1 = axes[0]

    if show_individual:
        for agent_id in agent_ids:
            rewards = [ep.get(agent_id, 0) for ep in episode_rewards]
            smoothed = _smooth(rewards, smoothing_window)

            color = get_agent_color(agent_id)
            ax1.plot(episodes, smoothed, label=agent_id, color=color, alpha=0.8)

    # Total reward per episode
    total_rewards = [sum(ep.values()) for ep in episode_rewards]
    smoothed_total = _smooth(total_rewards, smoothing_window)
    ax1.plot(episodes, smoothed_total, 'k-', linewidth=2, label='Total', alpha=0.9)

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Right plot: Cumulative rewards ---
    ax2 = axes[1]

    for agent_id in agent_ids:
        rewards = [ep.get(agent_id, 0) for ep in episode_rewards]
        cumulative = np.cumsum(rewards)

        color = get_agent_color(agent_id)
        ax2.plot(episodes, cumulative, label=agent_id, color=color, alpha=0.8)

    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Cumulative Reward')
    ax2.set_title('Cumulative Rewards')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_episode_lengths(
    metrics: TrainingMetrics,
    smoothing_window: int = 20,
    figsize: tuple = (10, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot episode lengths over training.

    Args:
        metrics: Training metrics
        smoothing_window: Window for moving average
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    lengths = metrics.episode_lengths
    episodes = range(1, len(lengths) + 1)

    # Raw lengths
    ax.plot(episodes, lengths, alpha=0.3, color='blue', label='Raw')

    # Smoothed
    smoothed = _smooth(lengths, smoothing_window)
    ax.plot(episodes, smoothed, color='blue', linewidth=2, label=f'Smoothed (w={smoothing_window})')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Length (steps)')
    ax.set_title('Episode Lengths Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_recruitment_progress(
    metrics: TrainingMetrics,
    smoothing_window: int = 20,
    figsize: tuple = (10, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot recruitment progress over training.

    Args:
        metrics: Training metrics
        smoothing_window: Window for moving average
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    recruited = metrics.rebels_recruited_history
    episodes = range(1, len(recruited) + 1)

    # Raw values
    ax.plot(episodes, recruited, alpha=0.3, color='red', label='Raw')

    # Smoothed
    smoothed = _smooth(recruited, smoothing_window)
    ax.plot(episodes, smoothed, color='red', linewidth=2, label=f'Smoothed (w={smoothing_window})')

    # Cumulative recruitment
    ax2 = ax.twinx()
    cumulative = np.cumsum(recruited)
    ax2.plot(episodes, cumulative, '--', color='darkred', alpha=0.6, label='Cumulative')
    ax2.set_ylabel('Cumulative Recruits', color='darkred')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Rebels Recruited (per episode)')
    ax.set_title('Recruitment Progress')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_training_summary(
    metrics: TrainingMetrics,
    figsize: tuple = (14, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a summary plot with multiple metrics.

    Args:
        metrics: Training metrics
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    episode_rewards = metrics.episode_rewards
    if not episode_rewards:
        return fig

    agent_ids = list(episode_rewards[0].keys())
    episodes = range(1, len(episode_rewards) + 1)

    # --- Top Left: Rewards ---
    ax = axes[0, 0]
    total_rewards = [sum(ep.values()) for ep in episode_rewards]
    smoothed = _smooth(total_rewards, 20)
    ax.plot(episodes, smoothed, 'b-', linewidth=2)
    ax.fill_between(episodes, smoothed, alpha=0.2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Total Rewards')
    ax.grid(True, alpha=0.3)

    # --- Top Right: Episode Length ---
    ax = axes[0, 1]
    smoothed_len = _smooth(metrics.episode_lengths, 20)
    ax.plot(episodes, smoothed_len, 'g-', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.set_title('Episode Length')
    ax.grid(True, alpha=0.3)

    # --- Bottom Left: Recruitment ---
    ax = axes[1, 0]
    smoothed_recruit = _smooth(metrics.rebels_recruited_history, 20)
    ax.plot(episodes, smoothed_recruit, 'r-', linewidth=2)
    ax.fill_between(episodes, smoothed_recruit, alpha=0.2, color='red')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Rebels Recruited')
    ax.set_title('Recruitment per Episode')
    ax.grid(True, alpha=0.3)

    # --- Bottom Right: Per-agent rewards bar chart ---
    ax = axes[1, 1]
    final_rewards = {aid: metrics.cumulative_rewards.get(aid, 0) for aid in agent_ids}
    colors = [get_agent_color(aid) for aid in agent_ids]
    bars = ax.bar(agent_ids, final_rewards.values(), color=colors, alpha=0.8)
    ax.set_xlabel('Agent')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('Final Cumulative Rewards')
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def create_training_dashboard(
    metrics: TrainingMetrics,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a comprehensive training dashboard.

    Args:
        metrics: Training metrics
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    return plot_training_summary(metrics, figsize=(16, 10), save_path=save_path)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _smooth(values: list, window: int) -> np.ndarray:
    """Apply moving average smoothing."""
    if len(values) < window:
        return np.array(values)

    values_arr = np.array(values, dtype=float)
    kernel = np.ones(window) / window
    smoothed = np.convolve(values_arr, kernel, mode='same')

    # Fix edge effects
    for i in range(window // 2):
        smoothed[i] = np.mean(values_arr[:i + window // 2 + 1])
        smoothed[-(i + 1)] = np.mean(values_arr[-(i + window // 2 + 1):])

    return smoothed
