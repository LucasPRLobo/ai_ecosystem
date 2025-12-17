"""
Live visualization during training for Prompted Town.

Provides real-time visualization of training progress
and agent interactions.
"""

from typing import Optional, Callable
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from ..rl import EpisodeMetrics, TrainingMetrics
from .interaction_graph import InteractionTracker


# =============================================================================
# LIVE TRAINING VISUALIZER
# =============================================================================

class LiveTrainingVisualizer:
    """
    Live visualization of training progress.

    Updates plots in real-time during training.
    """

    def __init__(
        self,
        agent_ids: list[str],
        figsize: tuple = (14, 8),
        update_frequency: int = 10,
    ):
        """
        Initialize the live visualizer.

        Args:
            agent_ids: List of agent IDs being trained
            figsize: Figure size
            update_frequency: Update plots every N episodes
        """
        self.agent_ids = agent_ids
        self.update_frequency = update_frequency

        # Data storage
        self.episode_rewards: list[dict[str, float]] = []
        self.episode_lengths: list[int] = []
        self.rebels_recruited: list[int] = []
        self.interaction_tracker = InteractionTracker()

        # Create figure with subplots
        plt.ion()  # Interactive mode
        self.fig, self.axes = plt.subplots(2, 2, figsize=figsize)
        self.fig.suptitle('Prompted Town Training Progress', fontsize=14)

        # Initialize empty plots
        self._setup_plots()

        plt.tight_layout()
        plt.show(block=False)

    def _setup_plots(self):
        """Setup initial empty plots."""
        # Top left: Rewards
        ax = self.axes[0, 0]
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Episode Rewards')
        ax.grid(True, alpha=0.3)
        self.reward_lines = {}

        # Top right: Episode length
        ax = self.axes[0, 1]
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.set_title('Episode Length')
        ax.grid(True, alpha=0.3)
        self.length_line, = ax.plot([], [], 'b-', linewidth=2)

        # Bottom left: Recruitment
        ax = self.axes[1, 0]
        ax.set_xlabel('Episode')
        ax.set_ylabel('Rebels Recruited')
        ax.set_title('Recruitment Progress')
        ax.grid(True, alpha=0.3)
        self.recruit_line, = ax.plot([], [], 'r-', linewidth=2)
        self.recruit_cumulative_line, = ax.plot([], [], 'r--', alpha=0.5)

        # Bottom right: Stats text
        ax = self.axes[1, 1]
        ax.axis('off')
        self.stats_text = ax.text(
            0.1, 0.5, '', fontsize=10, family='monospace',
            verticalalignment='center',
            transform=ax.transAxes,
        )

    def update(self, episode: int, metrics: EpisodeMetrics):
        """
        Update visualization with new episode data.

        Args:
            episode: Episode number
            metrics: Episode metrics
        """
        # Store data
        self.episode_rewards.append(metrics.total_rewards)
        self.episode_lengths.append(metrics.total_steps)
        self.rebels_recruited.append(metrics.rebels_recruited)

        # Only update plots every N episodes
        if (episode + 1) % self.update_frequency != 0:
            return

        episodes = range(1, len(self.episode_rewards) + 1)

        # Update reward plot
        ax = self.axes[0, 0]
        ax.clear()
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Episode Rewards')
        ax.grid(True, alpha=0.3)

        # Total rewards with smoothing
        total_rewards = [sum(ep.values()) for ep in self.episode_rewards]
        smoothed = self._smooth(total_rewards, 20)
        ax.plot(episodes, smoothed, 'k-', linewidth=2, label='Total')
        ax.fill_between(episodes, smoothed, alpha=0.2)
        ax.legend(loc='upper left', fontsize=8)

        # Update length plot
        ax = self.axes[0, 1]
        ax.clear()
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.set_title('Episode Length')
        ax.grid(True, alpha=0.3)
        smoothed_len = self._smooth(self.episode_lengths, 20)
        ax.plot(episodes, smoothed_len, 'b-', linewidth=2)

        # Update recruitment plot
        ax = self.axes[1, 0]
        ax.clear()
        ax.set_xlabel('Episode')
        ax.set_ylabel('Rebels Recruited')
        ax.set_title('Recruitment Progress')
        ax.grid(True, alpha=0.3)
        smoothed_recruit = self._smooth(self.rebels_recruited, 20)
        ax.plot(episodes, smoothed_recruit, 'r-', linewidth=2, label='Per Episode')
        cumulative = np.cumsum(self.rebels_recruited)
        ax2 = ax.twinx()
        ax2.plot(episodes, cumulative, 'r--', alpha=0.5, label='Cumulative')
        ax2.set_ylabel('Cumulative', color='darkred')
        ax.legend(loc='upper left', fontsize=8)

        # Update stats text
        ax = self.axes[1, 1]
        ax.clear()
        ax.axis('off')

        recent_reward = np.mean(total_rewards[-50:]) if len(total_rewards) >= 50 else np.mean(total_rewards)
        recent_length = np.mean(self.episode_lengths[-50:]) if len(self.episode_lengths) >= 50 else np.mean(self.episode_lengths)
        total_recruited = sum(self.rebels_recruited)

        stats_str = f"""
Training Statistics
{'='*30}

Episodes:        {episode + 1}
Total Steps:     {sum(self.episode_lengths):,}

Recent Avg Reward: {recent_reward:.2f}
Recent Avg Length: {recent_length:.1f}

Total Recruited: {total_recruited}
Success Rate:    {total_recruited / max(episode + 1, 1) * 100:.1f}%

Interactions:
  Conversations: {self.interaction_tracker.get_summary()['total_conversations']}
  Recruit Tries: {self.interaction_tracker.get_summary()['total_recruitment_attempts']}
  Arrests:       {self.interaction_tracker.get_summary()['total_arrests']}
"""
        self.stats_text = ax.text(
            0.05, 0.95, stats_str, fontsize=10, family='monospace',
            verticalalignment='top',
            transform=ax.transAxes,
        )

        # Redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def record_interaction(
        self,
        interaction_type: str,
        initiator: str,
        target: str,
        success: bool = True,
        tick: int = 0,
        day: int = 0,
    ):
        """Record an interaction for tracking."""
        if interaction_type == "conversation":
            self.interaction_tracker.record_conversation(initiator, target, tick, day)
        elif interaction_type == "recruitment":
            self.interaction_tracker.record_recruitment(initiator, target, success, tick, day)
        elif interaction_type == "arrest":
            self.interaction_tracker.record_arrest(initiator, target, tick, day)

    def close(self):
        """Close the visualization."""
        plt.ioff()
        plt.close(self.fig)

    def save(self, path: str):
        """Save the current figure."""
        self.fig.savefig(path, dpi=150, bbox_inches='tight')

    @staticmethod
    def _smooth(values: list, window: int) -> np.ndarray:
        """Apply moving average smoothing."""
        if len(values) < window:
            return np.array(values)

        values_arr = np.array(values, dtype=float)
        kernel = np.ones(window) / window
        return np.convolve(values_arr, kernel, mode='same')


# =============================================================================
# TRAINING CALLBACK
# =============================================================================

def create_episode_callback(
    visualizer: LiveTrainingVisualizer,
    track_interactions: bool = True,
) -> Callable[[int, EpisodeMetrics], None]:
    """
    Create a callback function for training that updates the visualizer.

    Args:
        visualizer: The live visualizer to update
        track_interactions: Whether to track interactions from step info

    Returns:
        Callback function compatible with train()
    """
    def callback(episode: int, metrics: EpisodeMetrics):
        visualizer.update(episode, metrics)

    return callback


# =============================================================================
# SIMPLE PROGRESS BAR
# =============================================================================

class ProgressBar:
    """Simple text-based progress bar for training."""

    def __init__(self, total: int, width: int = 50):
        self.total = total
        self.width = width
        self.current = 0

    def update(self, current: int, metrics: Optional[EpisodeMetrics] = None):
        """Update the progress bar."""
        self.current = current
        progress = current / self.total
        filled = int(self.width * progress)
        bar = '=' * filled + '-' * (self.width - filled)

        stats = ""
        if metrics:
            total_reward = sum(metrics.total_rewards.values())
            stats = f" R:{total_reward:.1f} L:{metrics.total_steps}"

        print(f'\r[{bar}] {current}/{self.total} ({progress*100:.1f}%){stats}', end='', flush=True)

        if current >= self.total:
            print()  # New line at end


# =============================================================================
# DEMO FUNCTION
# =============================================================================

def demo_live_viz():
    """Demo the live visualization with random data."""
    import random

    agent_ids = ["farmer_00", "farmer_01", "guard_02", "rebel_03"]
    viz = LiveTrainingVisualizer(agent_ids, update_frequency=5)

    print("Simulating training progress...")
    print("Close the plot window to stop.")

    try:
        for episode in range(200):
            # Simulate metrics
            metrics = EpisodeMetrics(
                episode=episode,
                total_steps=random.randint(20, 50),
                total_rewards={
                    "farmer_00": random.gauss(0.5, 1.0),
                    "farmer_01": random.gauss(0.5, 1.0),
                    "guard_02": random.gauss(1.0, 0.5),
                    "rebel_03": random.gauss(0.0, 2.0) + (episode / 100),
                },
                final_day=random.randint(1, 7),
                rebels_recruited=1 if random.random() < 0.1 + episode / 500 else 0,
                arrests_made=1 if random.random() < 0.05 else 0,
                quota_completions=random.randint(0, 2),
            )

            # Record some random interactions
            if random.random() < 0.3:
                viz.record_interaction(
                    "conversation",
                    random.choice(agent_ids),
                    random.choice(agent_ids),
                )
            if random.random() < 0.1:
                viz.record_interaction(
                    "recruitment",
                    "rebel_03",
                    random.choice(["farmer_00", "farmer_01"]),
                    success=random.random() < 0.3,
                )

            viz.update(episode, metrics)

            plt.pause(0.05)

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        viz.close()


if __name__ == "__main__":
    demo_live_viz()
