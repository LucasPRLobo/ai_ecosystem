"""
Q-Learning agents for Prompted Town.

This module implements tabular Q-learning agents that learn
to act in the simulation environment.

Design principles:
- Simple, interpretable tabular method
- Independent learning (no coordination)
- Epsilon-greedy exploration
- Easy to inspect learned values
"""

from dataclasses import dataclass, field
from typing import Optional, Any
from collections import defaultdict
import random
import json
import pickle
from pathlib import Path


# =============================================================================
# Q-LEARNING CONFIGURATION
# =============================================================================

@dataclass
class QLearningConfig:
    """Configuration for Q-learning agents."""
    # Learning parameters
    learning_rate: float = 0.1  # Alpha
    discount_factor: float = 0.95  # Gamma

    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995

    # Action space
    num_actions: int = 14  # Based on environment

    # Q-table initialization
    default_q_value: float = 0.0


# =============================================================================
# Q-LEARNING AGENT
# =============================================================================

class QLearningAgent:
    """
    Tabular Q-learning agent.

    Maintains a Q-table mapping (state, action) -> value.
    Uses epsilon-greedy exploration for action selection.
    """

    def __init__(
        self,
        agent_id: str,
        config: Optional[QLearningConfig] = None,
    ):
        self.agent_id = agent_id
        self.config = config or QLearningConfig()

        # Q-table: state -> action -> value
        # Using defaultdict for automatic initialization
        self.q_table: dict[tuple, dict[int, float]] = defaultdict(
            lambda: defaultdict(lambda: self.config.default_q_value)
        )

        # Exploration rate
        self.epsilon = self.config.epsilon_start

        # Statistics
        self.total_updates = 0
        self.states_visited = set()

    def select_action(
        self,
        state: tuple,
        valid_actions: Optional[list[int]] = None,
        explore: bool = True,
    ) -> int:
        """
        Select an action using epsilon-greedy policy.

        Args:
            state: Current state (tuple for hashing)
            valid_actions: List of valid action indices (optional)
            explore: Whether to use exploration

        Returns:
            Selected action index
        """
        if valid_actions is None:
            valid_actions = list(range(self.config.num_actions))

        if not valid_actions:
            return 0  # Default action if none valid

        # Epsilon-greedy exploration
        if explore and random.random() < self.epsilon:
            return random.choice(valid_actions)

        # Greedy action selection
        q_values = self.q_table[state]

        best_action = valid_actions[0]
        best_value = q_values[best_action]

        for action in valid_actions[1:]:
            value = q_values[action]
            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def update(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
        done: bool,
    ) -> float:
        """
        Update Q-value using the Q-learning update rule.

        Q(s,a) = Q(s,a) + α * (r + γ * max(Q(s',a')) - Q(s,a))

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether episode ended

        Returns:
            TD error for logging
        """
        current_q = self.q_table[state][action]

        if done:
            target = reward
        else:
            # Max Q-value for next state
            next_q_values = self.q_table[next_state]
            if next_q_values:
                max_next_q = max(next_q_values.values())
            else:
                max_next_q = self.config.default_q_value
            target = reward + self.config.discount_factor * max_next_q

        # TD error
        td_error = target - current_q

        # Update Q-value
        new_q = current_q + self.config.learning_rate * td_error
        self.q_table[state][action] = new_q

        # Statistics
        self.total_updates += 1
        self.states_visited.add(state)

        return td_error

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay
        )

    def get_q_values(self, state: tuple) -> dict[int, float]:
        """Get Q-values for a state."""
        return dict(self.q_table[state])

    def get_policy(self, state: tuple) -> int:
        """Get the greedy action for a state."""
        return self.select_action(state, explore=False)

    def get_stats(self) -> dict:
        """Get learning statistics."""
        return {
            "agent_id": self.agent_id,
            "epsilon": self.epsilon,
            "total_updates": self.total_updates,
            "states_visited": len(self.states_visited),
            "q_table_size": sum(len(v) for v in self.q_table.values()),
        }

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self, path: str):
        """Save agent to file."""
        data = {
            "agent_id": self.agent_id,
            "config": {
                "learning_rate": self.config.learning_rate,
                "discount_factor": self.config.discount_factor,
                "epsilon_start": self.config.epsilon_start,
                "epsilon_end": self.config.epsilon_end,
                "epsilon_decay": self.config.epsilon_decay,
                "num_actions": self.config.num_actions,
                "default_q_value": self.config.default_q_value,
            },
            "epsilon": self.epsilon,
            "total_updates": self.total_updates,
            "q_table": {
                str(state): dict(actions)
                for state, actions in self.q_table.items()
            },
        }

        filepath = Path(path)
        if filepath.suffix == ".json":
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
        else:
            with open(filepath, "wb") as f:
                pickle.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "QLearningAgent":
        """Load agent from file."""
        filepath = Path(path)

        if filepath.suffix == ".json":
            with open(filepath) as f:
                data = json.load(f)
        else:
            with open(filepath, "rb") as f:
                data = pickle.load(f)

        config = QLearningConfig(**data["config"])
        agent = cls(data["agent_id"], config)
        agent.epsilon = data["epsilon"]
        agent.total_updates = data["total_updates"]

        # Restore Q-table
        for state_str, actions in data["q_table"].items():
            # Convert string back to tuple
            state = eval(state_str)  # Safe for our tuple format
            for action_str, value in actions.items():
                agent.q_table[state][int(action_str)] = value

        return agent


# =============================================================================
# MULTI-AGENT Q-LEARNING
# =============================================================================

class MultiAgentQLearning:
    """
    Manager for multiple independent Q-learning agents.

    Each agent learns independently, but this class
    provides a unified interface for training.
    """

    def __init__(
        self,
        agent_ids: list[str],
        config: Optional[QLearningConfig] = None,
    ):
        self.config = config or QLearningConfig()
        self.agents: dict[str, QLearningAgent] = {
            agent_id: QLearningAgent(agent_id, self.config)
            for agent_id in agent_ids
        }

    def select_actions(
        self,
        observations: dict[str, tuple],
        valid_actions: Optional[dict[str, list[int]]] = None,
        explore: bool = True,
    ) -> dict[str, int]:
        """Select actions for all agents."""
        actions = {}
        for agent_id, obs in observations.items():
            agent = self.agents.get(agent_id)
            if agent:
                valid = valid_actions.get(agent_id) if valid_actions else None
                actions[agent_id] = agent.select_action(obs, valid, explore)
            else:
                actions[agent_id] = 0  # Default
        return actions

    def update_all(
        self,
        states: dict[str, tuple],
        actions: dict[str, int],
        rewards: dict[str, float],
        next_states: dict[str, tuple],
        done: bool,
    ) -> dict[str, float]:
        """Update all agents with their transitions."""
        td_errors = {}
        for agent_id, agent in self.agents.items():
            if agent_id in states and agent_id in actions:
                td_errors[agent_id] = agent.update(
                    states[agent_id],
                    actions[agent_id],
                    rewards.get(agent_id, 0.0),
                    next_states.get(agent_id, states[agent_id]),
                    done,
                )
        return td_errors

    def decay_epsilon_all(self):
        """Decay exploration for all agents."""
        for agent in self.agents.values():
            agent.decay_epsilon()

    def get_stats(self) -> dict[str, dict]:
        """Get stats for all agents."""
        return {
            agent_id: agent.get_stats()
            for agent_id, agent in self.agents.items()
        }

    def save_all(self, directory: str):
        """Save all agents to directory."""
        dirpath = Path(directory)
        dirpath.mkdir(parents=True, exist_ok=True)

        for agent_id, agent in self.agents.items():
            agent.save(str(dirpath / f"{agent_id}.pkl"))

    @classmethod
    def load_all(cls, directory: str) -> "MultiAgentQLearning":
        """Load all agents from directory."""
        dirpath = Path(directory)
        agents = {}

        for filepath in dirpath.glob("*.pkl"):
            agent = QLearningAgent.load(str(filepath))
            agents[agent.agent_id] = agent

        instance = cls([])
        instance.agents = agents
        return instance


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_q_table_summary(agent: QLearningAgent, top_n: int = 10):
    """Print a summary of the Q-table."""
    print(f"\n{'='*50}")
    print(f"Q-Table Summary for {agent.agent_id}")
    print(f"{'='*50}")
    print(f"States visited: {len(agent.states_visited)}")
    print(f"Total updates: {agent.total_updates}")
    print(f"Current epsilon: {agent.epsilon:.3f}")

    # Find most visited states
    state_counts = {}
    for state in agent.states_visited:
        q_vals = agent.q_table[state]
        max_q = max(q_vals.values()) if q_vals else 0
        state_counts[state] = (len(q_vals), max_q)

    # Sort by number of actions tried
    sorted_states = sorted(
        state_counts.items(),
        key=lambda x: x[1][0],
        reverse=True
    )[:top_n]

    print(f"\nTop {top_n} most explored states:")
    for state, (num_actions, max_q) in sorted_states:
        print(f"  State {state}: {num_actions} actions, max_q={max_q:.2f}")
