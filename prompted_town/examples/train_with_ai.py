"""
Training with AI-powered conversations.

This script runs RL training with actual LLM-powered conversations
between agents, using either Anthropic Claude or OpenAI GPT.

Run with:
    python -m prompted_town.examples.train_with_ai --backend anthropic

Environment variables:
    ANTHROPIC_API_KEY: API key for Anthropic Claude
    OPENAI_API_KEY: API key for OpenAI

Options:
    --backend {anthropic,openai,mock}  LLM backend to use
    --episodes N                        Number of training episodes
    --visualize                         Show live visualization
    --save-plots DIR                    Save plots to directory
"""

import argparse
import os
import sys
from pathlib import Path

# Set matplotlib backend before imports
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from prompted_town.rl import (
    EnvConfig,
    PromptedTownEnv,
    QLearningConfig,
    MultiAgentQLearning,
    TrainingMetrics,
    EpisodeMetrics,
)
from prompted_town.ai import create_backend
from prompted_town.viz import (
    InteractionTracker,
    LiveTrainingVisualizer,
    plot_training_summary,
    create_interaction_graph,
    plot_interaction_graph,
)


def train_with_ai(
    backend: str = "mock",
    num_episodes: int = 50,
    visualize: bool = False,
    save_dir: str = None,
    verbose: bool = True,
):
    """
    Run training with AI-powered conversations.

    Args:
        backend: LLM backend ("anthropic", "openai", or "mock")
        num_episodes: Number of training episodes
        visualize: Whether to show live visualization
        save_dir: Directory to save plots
        verbose: Print progress
    """
    # Create AI backend
    if verbose:
        print(f"Creating {backend} backend...")

    if backend == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: ANTHROPIC_API_KEY environment variable not set")
            sys.exit(1)
        ai_backend = create_backend(
            "anthropic",
            api_key=api_key,
            model="claude-3-5-haiku-latest",
        )
    elif backend == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY environment variable not set")
            sys.exit(1)
        ai_backend = create_backend(
            "openai",
            api_key=api_key,
            model="gpt-4o-mini",
        )
    else:
        ai_backend = create_backend("mock")

    # Create environment with AI conversations enabled
    env_config = EnvConfig(
        num_farmers=2,
        num_guards=1,
        num_rebels=1,
        max_ticks=30,  # Shorter episodes with AI (more expensive)
        max_days=5,
        use_ai_conversations=True,
        ai_backend=ai_backend,
        max_conversation_turns=4,  # Shorter conversations
        random_seed=42,
    )
    env = PromptedTownEnv(env_config)

    if verbose:
        print(f"Created environment with {len(env.agent_ids)} agents")
        print(f"Agents: {env.agent_ids}")

    # Create Q-learning agents
    q_config = QLearningConfig(
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon_start=1.0,
        epsilon_end=0.15,
        epsilon_decay=0.98,  # Slower decay for fewer episodes
        num_actions=env.get_action_space_size(),
    )
    agents = MultiAgentQLearning(env.agent_ids, q_config)

    # Tracking
    metrics = TrainingMetrics()
    metrics.cumulative_rewards = {aid: 0.0 for aid in env.agent_ids}
    tracker = InteractionTracker()

    # Live visualization
    viz = None
    if visualize:
        viz = LiveTrainingVisualizer(env.agent_ids, update_frequency=2)

    if verbose:
        print(f"\nStarting training for {num_episodes} episodes...")
        print("=" * 60)

    try:
        for episode in range(num_episodes):
            observations = env.reset()
            total_rewards = {aid: 0.0 for aid in env.agent_ids}
            episode_conversations = []

            for step in range(env_config.max_ticks):
                # Select actions
                action_indices = agents.select_actions(observations, explore=True)
                actions = {
                    agent_id: env._index_to_action(agent_id, idx)
                    for agent_id, idx in action_indices.items()
                }

                # Step environment
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
                        success = 'succeeded' in conv['outcome_summary'].lower()
                        tracker.record_recruitment(
                            conv['initiator'],
                            conv['target'],
                            success=success,
                            tick=info.tick,
                            day=info.day,
                        )

                    episode_conversations.append(conv)

                    if viz:
                        viz.record_interaction(
                            "conversation",
                            conv['initiator'],
                            conv['target'],
                        )
                        if conv['intent'] == 'recruit':
                            viz.record_interaction(
                                "recruitment",
                                conv['initiator'],
                                conv['target'],
                                success='succeeded' in conv['outcome_summary'].lower(),
                            )

                for arrested in info.arrests:
                    for aid in env.agent_ids:
                        if 'guard' in aid:
                            tracker.record_arrest(aid, arrested, info.tick, info.day)
                            if viz:
                                viz.record_interaction("arrest", aid, arrested)
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
            metrics.total_steps += step + 1
            metrics.episode_rewards.append(total_rewards)
            metrics.episode_lengths.append(step + 1)
            metrics.rebels_recruited_history.append(env.state.total_rebels_recruited)

            for aid, r in total_rewards.items():
                metrics.cumulative_rewards[aid] += r

            # Update visualization
            if viz:
                ep_metrics = EpisodeMetrics(
                    episode=episode,
                    total_steps=step + 1,
                    total_rewards=total_rewards,
                    final_day=env.state.day,
                    rebels_recruited=env.state.total_rebels_recruited,
                    arrests_made=len(info.arrests),
                    quota_completions=0,
                )
                viz.update(episode, ep_metrics)
                plt.pause(0.01)

            # Progress output
            if verbose:
                avg_reward = sum(total_rewards.values()) / len(total_rewards)
                conv_count = len(episode_conversations)
                print(f"Episode {episode + 1:3d}/{num_episodes} | "
                      f"Steps: {step + 1:2d} | "
                      f"Reward: {avg_reward:6.2f} | "
                      f"Recruited: {env.state.total_rebels_recruited} | "
                      f"Convs: {conv_count} | "
                      f"ε: {agents.agents[env.agent_ids[0]].epsilon:.3f}")

                # Show conversation logs
                for conv in episode_conversations[:2]:  # First 2 conversations
                    print(f"\n  --- {conv['initiator']} -> {conv['target']} ({conv['intent']}) ---")
                    print(f"  Outcome: {conv['outcome_summary']}")
                    transcript = conv.get('transcript', [])
                    if transcript:
                        for turn in transcript[:6]:  # First 6 turns
                            speaker = turn['speaker']
                            text = turn['text'][:100]  # Truncate long lines
                            if len(turn['text']) > 100:
                                text += "..."
                            print(f"    [{speaker}]: {text}")
                        if len(transcript) > 6:
                            print(f"    ... ({len(transcript) - 6} more turns)")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        if viz:
            viz.close()

    # Print summary
    if verbose:
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Episodes: {metrics.episodes_completed}")
        print(f"Total steps: {metrics.total_steps}")
        print(f"\nInteraction summary:")
        summary = tracker.get_summary()
        print(f"  Conversations: {summary['total_conversations']}")
        print(f"  Recruitment attempts: {summary['total_recruitment_attempts']}")
        print(f"  Successful recruitments: {summary['successful_recruitments']}")
        print(f"  Arrests: {summary['total_arrests']}")

        print(f"\nFinal cumulative rewards:")
        for aid, reward in metrics.cumulative_rewards.items():
            print(f"  {aid}: {reward:.2f}")

    # Save plots
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        print(f"\nSaving plots to {save_dir}/...")

        fig1 = plot_training_summary(metrics)
        fig1.savefig(f"{save_dir}/training_summary.png", dpi=150, bbox_inches='tight')
        print("  Saved training_summary.png")

        try:
            graph = create_interaction_graph(tracker, env.agent_specs)
            fig2 = plot_interaction_graph(graph, title="Agent Interactions (AI Conversations)")
            fig2.savefig(f"{save_dir}/interaction_graph.png", dpi=150, bbox_inches='tight')
            print("  Saved interaction_graph.png")
        except Exception as e:
            print(f"  Skipped interaction_graph: {e}")

        plt.close('all')

    return env, agents, metrics, tracker


def main():
    parser = argparse.ArgumentParser(
        description="Train Prompted Town with AI conversations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--backend',
        choices=['anthropic', 'openai', 'mock'],
        default='mock',
        help='LLM backend to use (default: mock)',
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=30,
        help='Number of training episodes (default: 30)',
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Show live visualization during training',
    )
    parser.add_argument(
        '--save-plots',
        type=str,
        default=None,
        help='Directory to save plots',
    )

    args = parser.parse_args()

    train_with_ai(
        backend=args.backend,
        num_episodes=args.episodes,
        visualize=args.visualize,
        save_dir=args.save_plots,
    )


if __name__ == "__main__":
    main()
