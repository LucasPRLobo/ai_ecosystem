"""
RL module: Reinforcement learning for Prompted Town.

This module provides the learning infrastructure:
- Observation encoding
- Reward computation
- Gym-like environment
- Q-learning agents
- Training utilities
"""

# Observations
from .observations import (
    ObservationConfig,
    get_observation_size,
    encode_observation,
    get_simple_observation,
    get_simple_observation_size,
    discretize_observation,
    hash_observation,
)

# Rewards
from .rewards import (
    RewardBreakdown,
    compute_reward,
    get_farmer_reward,
    get_guard_reward,
    get_rebel_reward,
    shape_reward,
)

# Environment
from .environment import (
    EnvConfig,
    StepInfo,
    PromptedTownEnv,
)

# Q-Learning
from .q_learning import (
    QLearningConfig,
    QLearningAgent,
    MultiAgentQLearning,
    print_q_table_summary,
)

# Training
from .training import (
    TrainingConfig,
    EpisodeMetrics,
    TrainingMetrics,
    train,
    run_episode,
    evaluate,
    quick_train,
    demo_trained_agent,
)

__all__ = [
    # Observations
    "ObservationConfig",
    "get_observation_size",
    "encode_observation",
    "get_simple_observation",
    "get_simple_observation_size",
    "discretize_observation",
    "hash_observation",
    # Rewards
    "RewardBreakdown",
    "compute_reward",
    "get_farmer_reward",
    "get_guard_reward",
    "get_rebel_reward",
    "shape_reward",
    # Environment
    "EnvConfig",
    "StepInfo",
    "PromptedTownEnv",
    # Q-Learning
    "QLearningConfig",
    "QLearningAgent",
    "MultiAgentQLearning",
    "print_q_table_summary",
    # Training
    "TrainingConfig",
    "EpisodeMetrics",
    "TrainingMetrics",
    "train",
    "run_episode",
    "evaluate",
    "quick_train",
    "demo_trained_agent",
]
