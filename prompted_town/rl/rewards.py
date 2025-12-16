"""
Reward computation for Prompted Town RL agents.

This module computes rewards based on agent goals and state changes.
Different agents receive different rewards for the same outcomes
based on their personality and objectives.

Design principles:
- Rewards derived from agent specs (goal weights)
- Shaped rewards to guide learning
- Both immediate and delayed rewards
- Normalized to reasonable ranges
"""

from dataclasses import dataclass, field
from typing import Optional

from ..core.types import GoalType, AgentRole, Faction
from ..core.world_state import WorldState, AgentState
from ..core.agent_spec import AgentSpec, RLProfile
from ..simulation.social import ConversationOutcome


# =============================================================================
# REWARD COMPONENTS
# =============================================================================

@dataclass
class RewardBreakdown:
    """Detailed breakdown of reward components."""
    survival: float = 0.0
    energy: float = 0.0
    wealth: float = 0.0
    quota: float = 0.0
    suspicion: float = 0.0
    recruitment: float = 0.0
    social: float = 0.0
    arrest: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.survival + self.energy + self.wealth +
            self.quota + self.suspicion + self.recruitment +
            self.social + self.arrest
        )

    def to_dict(self) -> dict:
        return {
            "survival": self.survival,
            "energy": self.energy,
            "wealth": self.wealth,
            "quota": self.quota,
            "suspicion": self.suspicion,
            "recruitment": self.recruitment,
            "social": self.social,
            "arrest": self.arrest,
            "total": self.total,
        }


# =============================================================================
# REWARD COMPUTATION
# =============================================================================

def compute_reward(
    prev_state: WorldState,
    new_state: WorldState,
    agent_id: str,
    agent_spec: AgentSpec,
    conversation_outcome: Optional[ConversationOutcome] = None,
) -> RewardBreakdown:
    """
    Compute the reward for an agent based on state transition.

    Args:
        prev_state: State before action
        new_state: State after action
        agent_id: The agent to compute reward for
        agent_spec: Agent's specification (for goal weights)
        conversation_outcome: Result of any conversation this tick

    Returns:
        RewardBreakdown with all components
    """
    reward = RewardBreakdown()

    prev_agent = prev_state.get_agent(agent_id)
    new_agent = new_state.get_agent(agent_id)

    if prev_agent is None or new_agent is None:
        return reward

    profile = agent_spec.rl_profile

    # Get weights (with defaults)
    w_survival = profile.get_reward_weight("survival", 1.0)
    w_wealth = profile.get_reward_weight("wealth", 0.5)
    w_quota = profile.get_reward_weight("quota", 0.5)
    w_suspicion = profile.get_reward_weight("suspicion", -0.5)
    w_social = profile.get_reward_weight("social", 0.3)
    w_recruitment = profile.get_reward_weight("recruitment", 0.0)

    # Survival reward
    reward.survival = _compute_survival_reward(prev_agent, new_agent, w_survival)

    # Energy management reward
    reward.energy = _compute_energy_reward(prev_agent, new_agent)

    # Wealth reward
    reward.wealth = _compute_wealth_reward(prev_agent, new_agent, w_wealth)

    # Quota reward
    reward.quota = _compute_quota_reward(
        prev_state, new_state, prev_agent, new_agent, w_quota
    )

    # Suspicion reward (usually negative)
    reward.suspicion = _compute_suspicion_reward(prev_agent, new_agent, w_suspicion)

    # Recruitment reward (for rebels)
    if agent_spec.true_faction == Faction.REBEL:
        reward.recruitment = _compute_recruitment_reward(
            prev_state, new_state, conversation_outcome, w_recruitment
        )

    # Social reward (from conversations)
    if conversation_outcome:
        reward.social = _compute_social_reward(conversation_outcome, w_social)

    # Arrest penalty
    reward.arrest = _compute_arrest_penalty(prev_agent, new_agent)

    return reward


def _compute_survival_reward(
    prev: AgentState,
    new: AgentState,
    weight: float,
) -> float:
    """Reward for staying alive and functional."""
    if not new.is_alive:
        return -10.0 * weight  # Death is very bad

    if new.is_arrested and not prev.is_arrested:
        return -5.0 * weight  # Getting arrested is bad

    # Small positive for being alive and free
    if new.is_alive and not new.is_arrested:
        return 0.1 * weight

    return 0.0


def _compute_energy_reward(
    prev: AgentState,
    new: AgentState,
) -> float:
    """Reward for energy management."""
    # Penalize very low energy
    if new.energy < 20:
        return -0.2

    # Small reward for maintaining good energy
    if new.energy >= 50:
        return 0.05

    return 0.0


def _compute_wealth_reward(
    prev: AgentState,
    new: AgentState,
    weight: float,
) -> float:
    """Reward for accumulating wealth."""
    gold_diff = new.gold - prev.gold

    if gold_diff > 0:
        return 0.1 * gold_diff * weight
    elif gold_diff < 0:
        return 0.05 * gold_diff * weight  # Less penalty for spending

    return 0.0


def _compute_quota_reward(
    prev_state: WorldState,
    new_state: WorldState,
    prev_agent: AgentState,
    new_agent: AgentState,
    weight: float,
) -> float:
    """Reward for quota progress and completion."""
    reward = 0.0

    # Progress toward quota
    quota_diff = (
        new_agent.quota_delivered_this_period -
        prev_agent.quota_delivered_this_period
    )
    if quota_diff > 0:
        reward += 0.2 * quota_diff * weight

    # Bonus for meeting quota
    quota_amount = new_state.laws.quota_amount
    if (new_agent.quota_delivered_this_period >= quota_amount and
        prev_agent.quota_delivered_this_period < quota_amount):
        reward += 1.0 * weight  # Quota completion bonus

    # Penalty for missing quota (checked at deadline)
    # This is handled by the law enforcement system, but we add urgency
    days_until = (new_state.laws.quota_deadline_day - (new_state.day % 7)) % 7
    if days_until <= 1 and new_agent.quota_delivered_this_period < quota_amount:
        shortfall = quota_amount - new_agent.quota_delivered_this_period
        reward -= 0.1 * shortfall * weight

    return reward


def _compute_suspicion_reward(
    prev: AgentState,
    new: AgentState,
    weight: float,  # Usually negative
) -> float:
    """Reward (penalty) for suspicion changes."""
    susp_diff = new.suspicion_level - prev.suspicion_level

    # Weight is typically negative, so increased suspicion = negative reward
    return susp_diff * weight


def _compute_recruitment_reward(
    prev_state: WorldState,
    new_state: WorldState,
    conversation_outcome: Optional[ConversationOutcome],
    weight: float,
) -> float:
    """Reward for successful recruitment (rebels only)."""
    reward = 0.0

    # Big reward for successful recruitment
    if conversation_outcome and conversation_outcome.recruitment_successful:
        reward += 2.0 * weight

    # Smaller reward for recruitment attempts (learning to try)
    if conversation_outcome and conversation_outcome.recruitment_attempted:
        if not conversation_outcome.recruitment_successful:
            reward += 0.1 * weight  # At least they tried

    # Reward for total rebels recruited
    rebel_diff = new_state.total_rebels_recruited - prev_state.total_rebels_recruited
    if rebel_diff > 0:
        reward += 1.0 * rebel_diff * weight

    return reward


def _compute_social_reward(
    outcome: ConversationOutcome,
    weight: float,
) -> float:
    """Reward for social interactions."""
    reward = 0.0

    # Positive conversations build relationships
    if outcome.was_positive:
        reward += 0.2 * weight

    # Trust increases are good
    avg_trust_delta = (outcome.initiator_trust_delta + outcome.target_trust_delta) / 2
    reward += avg_trust_delta * weight

    return reward


def _compute_arrest_penalty(
    prev: AgentState,
    new: AgentState,
) -> float:
    """Penalty for getting arrested."""
    if new.is_arrested and not prev.is_arrested:
        return -5.0
    return 0.0


# =============================================================================
# ROLE-SPECIFIC REWARD FUNCTIONS
# =============================================================================

def get_farmer_reward(
    prev_state: WorldState,
    new_state: WorldState,
    agent_id: str,
    agent_spec: AgentSpec,
    conversation_outcome: Optional[ConversationOutcome] = None,
) -> float:
    """Simplified reward for farmer agents."""
    breakdown = compute_reward(
        prev_state, new_state, agent_id, agent_spec, conversation_outcome
    )
    return breakdown.total


def get_guard_reward(
    prev_state: WorldState,
    new_state: WorldState,
    agent_id: str,
    agent_spec: AgentSpec,
    arrests_made: list[str],
) -> float:
    """Reward for guard agents."""
    reward = 0.0

    # Guards get reward for arrests
    reward += len(arrests_made) * 1.0

    # Reward for maintaining order (low total suspicion in town)
    total_suspicion = sum(
        a.suspicion_level for a in new_state.agents.values()
        if a.can_act()
    )
    prev_total = sum(
        a.suspicion_level for a in prev_state.agents.values()
        if a.can_act()
    )

    if total_suspicion < prev_total:
        reward += 0.2  # Order improved

    # Reward for catching curfew violations
    # (This would be tracked separately in law enforcement)

    return reward


def get_rebel_reward(
    prev_state: WorldState,
    new_state: WorldState,
    agent_id: str,
    agent_spec: AgentSpec,
    conversation_outcome: Optional[ConversationOutcome] = None,
) -> float:
    """Simplified reward for rebel agents."""
    breakdown = compute_reward(
        prev_state, new_state, agent_id, agent_spec, conversation_outcome
    )

    # Additional rebel-specific adjustments
    extra = 0.0

    # Bonus for growing rebellion without getting caught
    agent = new_state.get_agent(agent_id)
    if agent and not agent.is_arrested:
        if new_state.total_rebels_recruited > prev_state.total_rebels_recruited:
            extra += 1.0

    return breakdown.total + extra


# =============================================================================
# REWARD SHAPING
# =============================================================================

def shape_reward(
    raw_reward: float,
    scale: float = 1.0,
    clip_min: float = -10.0,
    clip_max: float = 10.0,
) -> float:
    """
    Apply reward shaping for stable learning.

    Args:
        raw_reward: Unscaled reward
        scale: Multiplier for reward magnitude
        clip_min: Minimum reward value
        clip_max: Maximum reward value

    Returns:
        Shaped reward
    """
    scaled = raw_reward * scale
    return max(clip_min, min(clip_max, scaled))
