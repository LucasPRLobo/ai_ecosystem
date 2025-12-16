"""
Observation encoding for Prompted Town RL agents.

This module converts world state into numeric observations that
agents can learn from. Each agent gets a partial observation
based on what they can perceive.

Design principles:
- Observations are numpy arrays for RL compatibility
- Each agent sees only what they could realistically know
- Features are normalized to [0, 1] or [-1, 1] ranges
- Observation space is fixed-size for tabular methods
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass

from ..core.types import (
    Location,
    TimeOfDay,
    AgentRole,
    Faction,
    TIME_CYCLE,
)
from ..core.world_state import WorldState, AgentState
from ..core.agent_spec import AgentSpec


# =============================================================================
# OBSERVATION SPACE CONFIGURATION
# =============================================================================

# Number of features in the observation vector
# Adjust these based on what information agents should have

NUM_LOCATIONS = len(Location)
NUM_TIME_PERIODS = len(TimeOfDay)
MAX_NEARBY_AGENTS = 4  # Max other agents we encode (for fixed-size obs)


@dataclass
class ObservationConfig:
    """Configuration for observation encoding."""
    include_location: bool = True
    include_time: bool = True
    include_energy: bool = True
    include_gold: bool = True
    include_suspicion: bool = True
    include_quota_progress: bool = True
    include_nearby_agents: bool = True
    include_relationships: bool = True
    include_recruitment_status: bool = True
    max_nearby_agents: int = MAX_NEARBY_AGENTS


DEFAULT_CONFIG = ObservationConfig()


# =============================================================================
# OBSERVATION ENCODING
# =============================================================================

def get_observation_size(config: ObservationConfig = DEFAULT_CONFIG) -> int:
    """Calculate the size of the observation vector."""
    size = 0

    if config.include_location:
        size += NUM_LOCATIONS  # One-hot encoding

    if config.include_time:
        size += NUM_TIME_PERIODS  # One-hot encoding

    if config.include_energy:
        size += 1  # Normalized energy

    if config.include_gold:
        size += 1  # Normalized gold

    if config.include_suspicion:
        size += 1  # Suspicion level

    if config.include_quota_progress:
        size += 2  # Progress and days remaining

    if config.include_nearby_agents:
        # For each nearby slot: is_present, is_guard, is_farmer, relationship
        size += config.max_nearby_agents * 4

    if config.include_recruitment_status:
        size += 2  # Am I recruited, number of known rebels

    return size


def encode_observation(
    state: WorldState,
    agent_id: str,
    agent_specs: dict[str, AgentSpec],
    config: ObservationConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """
    Encode the world state from an agent's perspective.

    Args:
        state: Current world state
        agent_id: The agent observing
        agent_specs: All agent specifications
        config: Observation configuration

    Returns:
        Numpy array of observation features
    """
    agent_state = state.get_agent(agent_id)
    if agent_state is None:
        # Return zeros if agent doesn't exist
        return np.zeros(get_observation_size(config), dtype=np.float32)

    features = []

    # Location (one-hot)
    if config.include_location:
        loc_onehot = np.zeros(NUM_LOCATIONS, dtype=np.float32)
        loc_idx = list(Location).index(agent_state.location)
        loc_onehot[loc_idx] = 1.0
        features.append(loc_onehot)

    # Time of day (one-hot)
    if config.include_time:
        time_onehot = np.zeros(NUM_TIME_PERIODS, dtype=np.float32)
        time_idx = TIME_CYCLE.index(state.time_of_day)
        time_onehot[time_idx] = 1.0
        features.append(time_onehot)

    # Energy (normalized 0-1)
    if config.include_energy:
        energy_norm = agent_state.energy / 100.0
        features.append(np.array([energy_norm], dtype=np.float32))

    # Gold (normalized, capped at 100)
    if config.include_gold:
        gold_norm = min(agent_state.gold / 100.0, 1.0)
        features.append(np.array([gold_norm], dtype=np.float32))

    # Suspicion level (already 0-1)
    if config.include_suspicion:
        features.append(np.array([agent_state.suspicion_level], dtype=np.float32))

    # Quota progress
    if config.include_quota_progress:
        quota_progress = min(
            agent_state.quota_delivered_this_period / max(state.laws.quota_amount, 1),
            1.0
        )
        # Days until deadline (normalized by week)
        days_until = (state.laws.quota_deadline_day - (state.day % 7)) % 7
        days_norm = days_until / 7.0
        features.append(np.array([quota_progress, days_norm], dtype=np.float32))

    # Nearby agents
    if config.include_nearby_agents:
        nearby_features = _encode_nearby_agents(
            state, agent_id, agent_specs, config.max_nearby_agents
        )
        features.append(nearby_features)

    # Recruitment status
    if config.include_recruitment_status:
        is_recruited = 1.0 if agent_state.is_recruited else 0.0
        num_known_rebels = min(len(agent_state.known_rebels) / 5.0, 1.0)
        features.append(np.array([is_recruited, num_known_rebels], dtype=np.float32))

    return np.concatenate(features)


def _encode_nearby_agents(
    state: WorldState,
    agent_id: str,
    agent_specs: dict[str, AgentSpec],
    max_nearby: int,
) -> np.ndarray:
    """Encode information about nearby agents."""
    nearby_ids = state.get_nearby_agents(agent_id)

    # Features per nearby agent: is_present, is_guard, is_farmer, relationship
    features = np.zeros(max_nearby * 4, dtype=np.float32)

    for i, nearby_id in enumerate(nearby_ids[:max_nearby]):
        base_idx = i * 4

        # Is present
        features[base_idx] = 1.0

        # Role indicators
        spec = agent_specs.get(nearby_id)
        if spec:
            features[base_idx + 1] = 1.0 if spec.public_role == AgentRole.GUARD else 0.0
            features[base_idx + 2] = 1.0 if spec.public_role == AgentRole.FARMER else 0.0

        # Relationship (trust level, normalized from [-1, 1] to [0, 1])
        trust = state.get_trust(agent_id, nearby_id)
        features[base_idx + 3] = (trust + 1.0) / 2.0

    return features


# =============================================================================
# OBSERVATION HASHING (for tabular methods)
# =============================================================================

def discretize_observation(
    obs: np.ndarray,
    bins: int = 10,
) -> tuple:
    """
    Discretize a continuous observation into bins.

    Useful for tabular Q-learning where we need hashable states.

    Args:
        obs: Continuous observation vector
        bins: Number of bins per dimension

    Returns:
        Tuple of discretized values (hashable)
    """
    # Clip to [0, 1] range and discretize
    clipped = np.clip(obs, 0.0, 1.0)
    discretized = (clipped * (bins - 1)).astype(int)
    return tuple(discretized.tolist())


def hash_observation(obs: np.ndarray, bins: int = 10) -> int:
    """
    Hash an observation to a single integer.

    For very large state spaces, consider using function approximation instead.
    """
    discrete = discretize_observation(obs, bins)
    return hash(discrete)


# =============================================================================
# SIMPLIFIED OBSERVATION (for faster learning)
# =============================================================================

def get_simple_observation(
    state: WorldState,
    agent_id: str,
    agent_specs: dict[str, AgentSpec],
) -> tuple:
    """
    Get a simplified, discrete observation for tabular methods.

    Returns a tuple that can be used directly as a dictionary key.
    Features:
    - Location index (0-5)
    - Time index (0-3)
    - Energy level (0-4, binned)
    - Has quota met (0-1)
    - Suspicion level (0-2, binned)
    - Nearby guard present (0-1)
    - Nearby potential recruit present (0-1)
    """
    agent_state = state.get_agent(agent_id)
    if agent_state is None:
        return (0, 0, 0, 0, 0, 0, 0)

    # Location index
    loc_idx = list(Location).index(agent_state.location)

    # Time index
    time_idx = TIME_CYCLE.index(state.time_of_day)

    # Energy level (0-4)
    energy_bin = min(4, agent_state.energy // 25)

    # Quota met
    quota_met = 1 if agent_state.quota_delivered_this_period >= state.laws.quota_amount else 0

    # Suspicion level (0-2)
    if agent_state.suspicion_level < 0.3:
        susp_bin = 0
    elif agent_state.suspicion_level < 0.6:
        susp_bin = 1
    else:
        susp_bin = 2

    # Check nearby agents
    nearby_ids = state.get_nearby_agents(agent_id)
    guard_nearby = 0
    recruit_target_nearby = 0

    for nearby_id in nearby_ids:
        spec = agent_specs.get(nearby_id)
        if spec:
            if spec.public_role == AgentRole.GUARD:
                guard_nearby = 1
            elif spec.public_role == AgentRole.FARMER:
                nearby_state = state.get_agent(nearby_id)
                if nearby_state and not nearby_state.is_recruited:
                    recruit_target_nearby = 1

    return (loc_idx, time_idx, energy_bin, quota_met, susp_bin, guard_nearby, recruit_target_nearby)


def get_simple_observation_size() -> int:
    """Get the number of possible simple observations (for Q-table sizing)."""
    # This is an upper bound: 6 * 4 * 5 * 2 * 3 * 2 * 2 = 2880
    return 6 * 4 * 5 * 2 * 3 * 2 * 2
