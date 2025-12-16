"""
Training loop for Prompted Town RL agents.

This module provides the training infrastructure for running
experiments with Q-learning agents.

Design principles:
- Flexible training configuration
- Metrics tracking and logging
- Support for evaluation episodes
- Progress visualization
"""

from dataclasses import dataclass, field
from typing import Optional, Callable
import time
from pathlib import Path

from .environment import PromptedTownEnv, EnvConfig
from .q_learning import QLearningAgent, MultiAgentQLearning, QLearningConfig


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Episodes
    num_episodes: int = 1000
    max_steps_per_episode: int = 100

    # Evaluation
    eval_frequency: int = 100
    eval_episodes: int = 10

    # Logging
    log_frequency: int = 50
    save_frequency: int = 200
    save_dir: str = "checkpoints"

    # Environment
    env_config: Optional[EnvConfig] = None


# =============================================================================
# METRICS TRACKING
# =============================================================================

@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""
    episode: int
    total_steps: int
    total_rewards: dict[str, float]
    final_day: int
    rebels_recruited: int
    arrests_made: int
    quota_completions: int


@dataclass
class TrainingMetrics:
    """Aggregated training metrics."""
    episodes_completed: int = 0
    total_steps: int = 0

    # Per-agent cumulative rewards
    cumulative_rewards: dict[str, float] = field(default_factory=dict)

    # Episode history
    episode_rewards: list[dict[str, float]] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    rebels_recruited_history: list[int] = field(default_factory=list)

    # Evaluation metrics
    eval_rewards: list[dict[str, float]] = field(default_factory=list)

    def get_recent_avg_reward(self, agent_id: str, window: int = 50) -> float:
        """Get average reward for recent episodes."""
        recent = [ep.get(agent_id, 0) for ep in self.episode_rewards[-window:]]
        return sum(recent) / max(len(recent), 1)

    def get_summary(self) -> dict:
        """Get summary of training metrics."""
        return {
            "episodes": self.episodes_completed,
            "total_steps": self.total_steps,
            "avg_episode_length": sum(self.episode_lengths) / max(len(self.episode_lengths), 1),
            "avg_rebels_recruited": sum(self.rebels_recruited_history) / max(len(self.rebels_recruited_history), 1),
        }


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train(
    env: PromptedTownEnv,
    agents: MultiAgentQLearning,
    config: TrainingConfig,
    callback: Optional[Callable[[int, EpisodeMetrics], None]] = None,
) -> TrainingMetrics:
    """
    Train agents in the environment.

    Args:
        env: The environment
        agents: Multi-agent Q-learning manager
        config: Training configuration
        callback: Optional callback called after each episode

    Returns:
        TrainingMetrics with results
    """
    metrics = TrainingMetrics()
    metrics.cumulative_rewards = {aid: 0.0 for aid in env.agent_ids}

    start_time = time.time()

    for episode in range(config.num_episodes):
        episode_metrics = run_episode(env, agents, config.max_steps_per_episode)

        # Update metrics
        metrics.episodes_completed += 1
        metrics.total_steps += episode_metrics.total_steps
        metrics.episode_rewards.append(episode_metrics.total_rewards)
        metrics.episode_lengths.append(episode_metrics.total_steps)
        metrics.rebels_recruited_history.append(episode_metrics.rebels_recruited)

        for agent_id, reward in episode_metrics.total_rewards.items():
            metrics.cumulative_rewards[agent_id] += reward

        # Decay exploration
        agents.decay_epsilon_all()

        # Logging
        if (episode + 1) % config.log_frequency == 0:
            elapsed = time.time() - start_time
            avg_reward = sum(episode_metrics.total_rewards.values()) / len(episode_metrics.total_rewards)
            print(f"Episode {episode + 1}/{config.num_episodes} | "
                  f"Steps: {episode_metrics.total_steps} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Rebels: {episode_metrics.rebels_recruited} | "
                  f"Epsilon: {list(agents.agents.values())[0].epsilon:.3f} | "
                  f"Time: {elapsed:.1f}s")

        # Evaluation
        if (episode + 1) % config.eval_frequency == 0:
            eval_rewards = evaluate(env, agents, config.eval_episodes)
            metrics.eval_rewards.append(eval_rewards)

            avg_eval = sum(sum(r.values()) for r in eval_rewards) / len(eval_rewards)
            print(f"  Eval avg reward: {avg_eval / len(env.agent_ids):.2f}")

        # Save checkpoint
        if config.save_dir and (episode + 1) % config.save_frequency == 0:
            save_path = Path(config.save_dir) / f"checkpoint_{episode + 1}"
            agents.save_all(str(save_path))

        # Callback
        if callback:
            callback(episode, episode_metrics)

    return metrics


def run_episode(
    env: PromptedTownEnv,
    agents: MultiAgentQLearning,
    max_steps: int,
    explore: bool = True,
) -> EpisodeMetrics:
    """
    Run a single episode.

    Args:
        env: The environment
        agents: Multi-agent Q-learning manager
        max_steps: Maximum steps per episode
        explore: Whether to use exploration

    Returns:
        EpisodeMetrics for the episode
    """
    observations = env.reset()
    total_rewards = {aid: 0.0 for aid in env.agent_ids}
    arrests_made = 0
    quota_completions = 0

    for step in range(max_steps):
        # Select actions
        action_indices = agents.select_actions(observations, explore=explore)

        # Convert to Action objects
        actions = {
            agent_id: env._index_to_action(agent_id, idx)
            for agent_id, idx in action_indices.items()
        }

        # Step environment
        next_observations, rewards, done, info = env.step(actions)

        # Update agents
        agents.update_all(
            observations,
            action_indices,
            rewards,
            next_observations,
            done,
        )

        # Track rewards
        for agent_id, reward in rewards.items():
            total_rewards[agent_id] += reward

        # Track other metrics
        arrests_made += len(info.arrests)

        observations = next_observations

        if done:
            break

    return EpisodeMetrics(
        episode=0,
        total_steps=step + 1,
        total_rewards=total_rewards,
        final_day=env.state.day,
        rebels_recruited=env.state.total_rebels_recruited,
        arrests_made=arrests_made,
        quota_completions=quota_completions,
    )


def evaluate(
    env: PromptedTownEnv,
    agents: MultiAgentQLearning,
    num_episodes: int,
) -> list[dict[str, float]]:
    """
    Evaluate agents without exploration.

    Args:
        env: The environment
        agents: Multi-agent Q-learning manager
        num_episodes: Number of evaluation episodes

    Returns:
        List of reward dictionaries per episode
    """
    episode_rewards = []

    for _ in range(num_episodes):
        metrics = run_episode(env, agents, max_steps=100, explore=False)
        episode_rewards.append(metrics.total_rewards)

    return episode_rewards


# =============================================================================
# DEMO / QUICK START
# =============================================================================

def quick_train(
    num_episodes: int = 500,
    num_farmers: int = 2,
    num_guards: int = 1,
    num_rebels: int = 1,
    verbose: bool = True,
) -> tuple[PromptedTownEnv, MultiAgentQLearning, TrainingMetrics]:
    """
    Quick training function for testing.

    Args:
        num_episodes: Number of training episodes
        num_farmers: Number of farmer agents
        num_guards: Number of guard agents
        num_rebels: Number of rebel agents
        verbose: Whether to print progress

    Returns:
        Tuple of (env, agents, metrics)
    """
    # Create environment
    env_config = EnvConfig(
        num_farmers=num_farmers,
        num_guards=num_guards,
        num_rebels=num_rebels,
        max_ticks=50,
        max_days=7,
        use_ai_conversations=False,  # Fast training
    )
    env = PromptedTownEnv(env_config)

    # Create agents
    q_config = QLearningConfig(
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.995,
        num_actions=env.get_action_space_size(),
    )
    agents = MultiAgentQLearning(env.agent_ids, q_config)

    # Training config
    train_config = TrainingConfig(
        num_episodes=num_episodes,
        max_steps_per_episode=50,
        log_frequency=50 if verbose else num_episodes + 1,
        eval_frequency=100,
        eval_episodes=5,
        save_frequency=num_episodes + 1,  # Don't save by default
    )

    if verbose:
        print(f"Training {len(env.agent_ids)} agents for {num_episodes} episodes...")
        print(f"Agents: {env.agent_ids}")

    # Train
    metrics = train(env, agents, train_config)

    if verbose:
        print(f"\nTraining complete!")
        print(f"Total steps: {metrics.total_steps}")
        print(f"Average episode length: {sum(metrics.episode_lengths) / len(metrics.episode_lengths):.1f}")

    return env, agents, metrics


def demo_trained_agent(
    env: PromptedTownEnv,
    agents: MultiAgentQLearning,
    num_steps: int = 30,
):
    """
    Demo a trained agent with rendering.

    Args:
        env: The environment
        agents: Trained agents
        num_steps: Number of steps to run
    """
    print("\n" + "=" * 50)
    print("DEMO: Trained Agents in Action")
    print("=" * 50)

    observations = env.reset()
    print(env.render())

    total_rewards = {aid: 0.0 for aid in env.agent_ids}

    for step in range(num_steps):
        # Select actions (no exploration)
        action_indices = agents.select_actions(observations, explore=False)

        actions = {
            agent_id: env._index_to_action(agent_id, idx)
            for agent_id, idx in action_indices.items()
        }

        # Print actions
        print(f"\n--- Step {step + 1} ---")
        for agent_id, action in actions.items():
            print(f"  {agent_id}: {action}")

        # Step
        observations, rewards, done, info = env.step(actions)

        for aid, r in rewards.items():
            total_rewards[aid] += r

        # Print events
        if info.conversations:
            for conv in info.conversations:
                print(f"  CONVERSATION: {conv['initiator']} -> {conv['target']}: {conv['outcome_summary']}")

        if info.arrests:
            print(f"  ARRESTS: {info.arrests}")

        # Render every few steps
        if (step + 1) % 5 == 0:
            print(env.render())

        if done:
            print("\n*** Episode ended ***")
            break

    print(f"\nFinal rewards: {total_rewards}")
    print(f"Rebels recruited: {env.state.total_rebels_recruited}")


# =============================================================================
# MAIN (for direct execution)
# =============================================================================

if __name__ == "__main__":
    print("Prompted Town RL Training")
    print("=" * 50)

    env, agents, metrics = quick_train(
        num_episodes=300,
        num_farmers=2,
        num_guards=1,
        num_rebels=1,
    )

    # Show agent stats
    print("\nAgent Statistics:")
    for agent_id, stats in agents.get_stats().items():
        print(f"  {agent_id}: {stats['states_visited']} states, "
              f"{stats['total_updates']} updates, "
              f"ε={stats['epsilon']:.3f}")

    # Demo
    demo_trained_agent(env, agents, num_steps=20)
