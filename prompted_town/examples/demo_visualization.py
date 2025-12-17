"""
Demo script for Prompted Town visualization.

This demonstrates the visualization capabilities:
- Training curves
- Entity interaction graphs
- Trust heatmaps

Run with:
    python -m prompted_town.examples.demo_visualization [--live] [--save]

Options:
    --live  Show live visualization during training
    --save  Save plots to files
"""

import argparse
import sys
from pathlib import Path

# Ensure matplotlib works in different environments
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plots
import matplotlib.pyplot as plt


def run_training_with_tracking(num_episodes: int = 200, verbose: bool = True):
    """Run training and track interactions for visualization."""
    from prompted_town.rl import (
        EnvConfig,
        PromptedTownEnv,
        QLearningConfig,
        MultiAgentQLearning,
        TrainingConfig,
        TrainingMetrics,
        EpisodeMetrics,
        run_episode,
    )
    from prompted_town.viz import InteractionTracker

    # Create environment
    env_config = EnvConfig(
        num_farmers=2,
        num_guards=1,
        num_rebels=1,
        max_ticks=50,
        max_days=7,
        use_ai_conversations=False,
        random_seed=42,
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

    # Metrics tracking
    metrics = TrainingMetrics()
    metrics.cumulative_rewards = {aid: 0.0 for aid in env.agent_ids}

    # Interaction tracking
    tracker = InteractionTracker()

    if verbose:
        print(f"Training {len(env.agent_ids)} agents for {num_episodes} episodes...")
        print(f"Agents: {env.agent_ids}")

    for episode in range(num_episodes):
        # Run episode with tracking
        observations = env.reset()
        total_rewards = {aid: 0.0 for aid in env.agent_ids}

        for step in range(50):
            action_indices = agents.select_actions(observations, explore=True)
            actions = {
                agent_id: env._index_to_action(agent_id, idx)
                for agent_id, idx in action_indices.items()
            }

            next_observations, rewards, done, info = env.step(actions)

            # Track interactions
            for conv in info.conversations:
                tracker.record_conversation(
                    conv['initiator'],
                    conv['target'],
                    tick=info.tick,
                    day=info.day,
                )
                if conv['intent'] == 'recruit':
                    success = 'succeeded' in conv['outcome_summary']
                    tracker.record_recruitment(
                        conv['initiator'],
                        conv['target'],
                        success=success,
                        tick=info.tick,
                        day=info.day,
                    )

            for arrested in info.arrests:
                # Find the guard who made the arrest
                for aid in env.agent_ids:
                    if 'guard' in aid:
                        tracker.record_arrest(aid, arrested, info.tick, info.day)
                        break

            # Update agents
            agents.update_all(
                observations,
                action_indices,
                rewards,
                next_observations,
                done,
            )

            for aid, r in rewards.items():
                total_rewards[aid] += r

            observations = next_observations
            if done:
                break

        # Decay exploration
        agents.decay_epsilon_all()

        # Store metrics
        metrics.episodes_completed += 1
        metrics.episode_rewards.append(total_rewards)
        metrics.episode_lengths.append(step + 1)
        metrics.rebels_recruited_history.append(env.state.total_rebels_recruited)

        for aid, r in total_rewards.items():
            metrics.cumulative_rewards[aid] += r

        # Progress
        if verbose and (episode + 1) % 50 == 0:
            avg_reward = sum(total_rewards.values()) / len(total_rewards)
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Rebels: {env.state.total_rebels_recruited}")

    return env, agents, metrics, tracker


def demo_static_plots(save_dir: str = None):
    """Generate static plots after training."""
    from prompted_town.viz import (
        plot_training_rewards,
        plot_training_summary,
        create_interaction_graph,
        plot_interaction_graph,
        plot_trust_heatmap,
    )

    print("\n" + "=" * 60)
    print("PROMPTED TOWN VISUALIZATION DEMO")
    print("=" * 60)

    # Run training
    env, agents, metrics, tracker = run_training_with_tracking(num_episodes=300)

    print("\nTraining complete!")
    print(f"Interaction summary: {tracker.get_summary()}")

    # Create save directory if needed
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Plot 1: Training rewards
    print("\nGenerating training rewards plot...")
    fig1 = plot_training_rewards(
        metrics,
        smoothing_window=20,
        save_path=f"{save_dir}/training_rewards.png" if save_dir else None,
    )

    # Plot 2: Training summary dashboard
    print("Generating training summary...")
    fig2 = plot_training_summary(
        metrics,
        save_path=f"{save_dir}/training_summary.png" if save_dir else None,
    )

    # Plot 3: Interaction graph
    print("Generating interaction graph...")
    try:
        graph = create_interaction_graph(
            tracker,
            env.agent_specs,
            min_interactions=1,
        )
        fig3 = plot_interaction_graph(
            graph,
            title="Agent Interactions During Training",
            save_path=f"{save_dir}/interaction_graph.png" if save_dir else None,
        )
    except ImportError as e:
        print(f"  Skipping interaction graph: {e}")
        fig3 = None

    # Plot 4: Trust heatmap
    print("Generating trust heatmap...")
    fig4 = plot_trust_heatmap(
        env.state,
        env.agent_ids,
        title="Final Trust Relationships",
        save_path=f"{save_dir}/trust_heatmap.png" if save_dir else None,
    )

    print("\nAll plots generated!")
    if save_dir:
        print(f"Plots saved to: {save_dir}/")

    plt.show()


def demo_live_training():
    """Demo live training visualization."""
    from prompted_town.rl import (
        EnvConfig,
        PromptedTownEnv,
        QLearningConfig,
        MultiAgentQLearning,
        EpisodeMetrics,
    )
    from prompted_town.viz import LiveTrainingVisualizer

    print("\n" + "=" * 60)
    print("LIVE TRAINING VISUALIZATION")
    print("=" * 60)
    print("Close the plot window to stop training.")

    # Create environment
    env_config = EnvConfig(
        num_farmers=2,
        num_guards=1,
        num_rebels=1,
        max_ticks=50,
        use_ai_conversations=False,
        random_seed=42,
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

    # Create live visualizer
    viz = LiveTrainingVisualizer(env.agent_ids, update_frequency=5)

    try:
        for episode in range(500):
            observations = env.reset()
            total_rewards = {aid: 0.0 for aid in env.agent_ids}

            for step in range(50):
                action_indices = agents.select_actions(observations, explore=True)
                actions = {
                    agent_id: env._index_to_action(agent_id, idx)
                    for agent_id, idx in action_indices.items()
                }

                next_observations, rewards, done, info = env.step(actions)

                # Track interactions
                for conv in info.conversations:
                    viz.record_interaction(
                        "conversation",
                        conv['initiator'],
                        conv['target'],
                    )
                    if conv['intent'] == 'recruit':
                        success = 'succeeded' in conv['outcome_summary']
                        viz.record_interaction(
                            "recruitment",
                            conv['initiator'],
                            conv['target'],
                            success=success,
                        )

                for arrested in info.arrests:
                    for aid in env.agent_ids:
                        if 'guard' in aid:
                            viz.record_interaction("arrest", aid, arrested)
                            break

                agents.update_all(
                    observations, action_indices, rewards, next_observations, done
                )

                for aid, r in rewards.items():
                    total_rewards[aid] += r

                observations = next_observations
                if done:
                    break

            agents.decay_epsilon_all()

            # Update visualization
            metrics = EpisodeMetrics(
                episode=episode,
                total_steps=step + 1,
                total_rewards=total_rewards,
                final_day=env.state.day,
                rebels_recruited=env.state.total_rebels_recruited,
                arrests_made=len(info.arrests),
                quota_completions=0,
            )
            viz.update(episode, metrics)

            plt.pause(0.01)

    except KeyboardInterrupt:
        print("\nTraining stopped by user.")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        print("\nSaving final visualization...")
        viz.save("training_progress.png")
        viz.close()
        print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Prompted Town Visualization Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m prompted_town.examples.demo_visualization
    python -m prompted_town.examples.demo_visualization --live
    python -m prompted_town.examples.demo_visualization --save ./plots
        """
    )
    parser.add_argument(
        '--live',
        action='store_true',
        help='Show live visualization during training',
    )
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Directory to save plots',
    )

    args = parser.parse_args()

    if args.live:
        demo_live_training()
    else:
        demo_static_plots(save_dir=args.save)


if __name__ == "__main__":
    main()
