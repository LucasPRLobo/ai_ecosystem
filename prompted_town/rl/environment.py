"""
Multi-agent RL environment for Prompted Town.

This module provides a Gym-like interface for the simulation.
Multiple agents act simultaneously, and the environment handles
state transitions, rewards, and episode management.

Design principles:
- Gym-compatible API (reset, step)
- Multi-agent support (dict observations/actions/rewards)
- Conversation handling (pluggable AI backend)
- Episode tracking and termination
"""

from dataclasses import dataclass, field
from typing import Optional, Any
import random

from ..core.types import (
    Location,
    TimeOfDay,
    ActionType,
    ConversationIntent,
    AgentRole,
)
from ..core.world_state import WorldState, create_default_world
from ..core.agent_spec import (
    AgentSpec,
    create_farmer_spec,
    create_guard_spec,
    create_rebel_spec,
)

from ..simulation import (
    Action,
    step_world,
    enforce_laws,
    get_valid_actions,
    move_action,
    work_action,
    rest_action,
    wait_action,
    deliver_quota_action,
    conversation_action,
    ConversationOutcome,
    apply_conversation_outcome,
)

from ..ai import (
    LLMBackend,
    MockLLMBackend,
    ConversationEngine,
    create_context_from_world,
    parse_conversation_outcome,
)

from .observations import (
    get_simple_observation,
    encode_observation,
    ObservationConfig,
)
from .rewards import compute_reward, RewardBreakdown


# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================

@dataclass
class EnvConfig:
    """Configuration for the environment."""
    # Episode settings
    max_ticks: int = 100
    max_days: int = 14

    # Agents
    num_farmers: int = 3
    num_guards: int = 1
    num_rebels: int = 1

    # AI conversations
    use_ai_conversations: bool = False  # If False, use simple resolution
    ai_backend: Optional[LLMBackend] = None
    max_conversation_turns: int = 6

    # Randomization
    random_seed: Optional[int] = None

    # Observation mode
    use_simple_observations: bool = True  # For tabular methods


# =============================================================================
# STEP INFO
# =============================================================================

@dataclass
class StepInfo:
    """Information returned from a step."""
    tick: int
    day: int
    time_of_day: TimeOfDay
    events: list[dict] = field(default_factory=list)
    conversations: list[dict] = field(default_factory=list)
    violations: list[dict] = field(default_factory=list)
    arrests: list[str] = field(default_factory=list)
    reward_breakdowns: dict[str, RewardBreakdown] = field(default_factory=dict)


# =============================================================================
# MULTI-AGENT ENVIRONMENT
# =============================================================================

class PromptedTownEnv:
    """
    Multi-agent environment for Prompted Town.

    Gym-like interface:
    - reset() -> observations
    - step(actions) -> observations, rewards, done, info
    """

    def __init__(self, config: Optional[EnvConfig] = None):
        self.config = config or EnvConfig()

        # Initialize random
        self.rng = random.Random(self.config.random_seed)

        # Create agent specs
        self.agent_specs: dict[str, AgentSpec] = {}
        self._create_agents()

        # State
        self.state: Optional[WorldState] = None
        self.prev_state: Optional[WorldState] = None

        # Conversation handling
        if self.config.use_ai_conversations:
            self.ai_backend = self.config.ai_backend or MockLLMBackend()
            self.conversation_engine = ConversationEngine(self.ai_backend)
        else:
            self.ai_backend = None
            self.conversation_engine = None

        # Episode tracking
        self.current_tick = 0
        self.episode_rewards: dict[str, float] = {}

    def _create_agents(self):
        """Create agent specifications."""
        agent_id = 0

        # Create farmers
        farmer_names = ["Tom", "Beth", "Giles", "Martha", "Edwin"]
        for i in range(self.config.num_farmers):
            aid = f"farmer_{agent_id:02d}"
            name = farmer_names[i % len(farmer_names)]
            self.agent_specs[aid] = create_farmer_spec(aid, name)
            agent_id += 1

        # Create guards
        guard_names = ["Stern", "Hawk", "Stone"]
        for i in range(self.config.num_guards):
            aid = f"guard_{agent_id:02d}"
            name = guard_names[i % len(guard_names)]
            self.agent_specs[aid] = create_guard_spec(aid, name)
            agent_id += 1

        # Create rebels
        rebel_names = ["Mara", "Shade", "Whisper"]
        for i in range(self.config.num_rebels):
            aid = f"rebel_{agent_id:02d}"
            name = rebel_names[i % len(rebel_names)]
            self.agent_specs[aid] = create_rebel_spec(aid, name)
            agent_id += 1

    @property
    def agent_ids(self) -> list[str]:
        """Get list of all agent IDs."""
        return list(self.agent_specs.keys())

    def reset(
        self,
        seed: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Reset the environment to initial state.

        Returns:
            Dictionary of observations for each agent
        """
        if seed is not None:
            self.rng = random.Random(seed)

        # Create world state
        self.state = create_default_world(
            agent_ids=self.agent_ids,
            random_seed=self.rng.randint(0, 2**31),
        )

        # Distribute agents to starting locations
        self._initialize_agent_positions()

        # Reset tracking
        self.current_tick = 0
        self.prev_state = None
        self.episode_rewards = {aid: 0.0 for aid in self.agent_ids}

        return self._get_observations()

    def _initialize_agent_positions(self):
        """Set initial agent positions."""
        for agent_id, spec in self.agent_specs.items():
            agent = self.state.get_agent(agent_id)
            if agent:
                # Start at home or role-appropriate location
                if spec.public_role == AgentRole.GUARD:
                    agent.location = Location.GATE
                elif spec.public_role == AgentRole.FARMER:
                    agent.location = Location.HOME
                else:
                    agent.location = Location.HOME

    def step(
        self,
        actions: dict[str, Action],
    ) -> tuple[dict, dict, bool, StepInfo]:
        """
        Step the environment forward.

        Args:
            actions: Dictionary mapping agent_id to Action

        Returns:
            observations: Dict of observations per agent
            rewards: Dict of rewards per agent
            done: Whether episode is finished
            info: StepInfo with additional data
        """
        if self.state is None:
            raise RuntimeError("Must call reset() before step()")

        # Store previous state for reward computation
        self.prev_state = self.state.clone()

        # Convert action indices to Action objects if needed
        resolved_actions = self._resolve_actions(actions)

        # Step the simulation
        step_result = step_world(
            self.state,
            resolved_actions,
            random_seed=self.rng.randint(0, 2**31),
        )
        self.state = step_result.new_state

        # Handle conversations
        conversation_outcomes: dict[str, ConversationOutcome] = {}
        conversation_info = []

        for pending_conv in step_result.pending_conversations:
            outcome = self._resolve_conversation(pending_conv)
            if outcome:
                conversation_outcomes[pending_conv.initiator_id] = outcome
                conversation_outcomes[pending_conv.target_id] = outcome

                # Apply to world state
                apply_conversation_outcome(self.state, outcome, self.agent_specs)

                conversation_info.append({
                    "initiator": pending_conv.initiator_id,
                    "target": pending_conv.target_id,
                    "intent": pending_conv.intent.value,
                    "outcome_summary": outcome.get_summary(),
                })

        # Enforce laws
        law_result = enforce_laws(self.state, self.agent_specs)

        # Compute rewards
        rewards = {}
        reward_breakdowns = {}
        for agent_id in self.agent_ids:
            conv_outcome = conversation_outcomes.get(agent_id)
            breakdown = compute_reward(
                self.prev_state,
                self.state,
                agent_id,
                self.agent_specs[agent_id],
                conv_outcome,
            )
            rewards[agent_id] = breakdown.total
            reward_breakdowns[agent_id] = breakdown
            self.episode_rewards[agent_id] += breakdown.total

        # Get observations
        observations = self._get_observations()

        # Check termination
        done = self._check_done()

        # Build info
        info = StepInfo(
            tick=self.state.tick,
            day=self.state.day,
            time_of_day=self.state.time_of_day,
            events=[e.to_dict() for e in step_result.events],
            conversations=conversation_info,
            violations=[{
                "agent": v.agent_id,
                "type": v.violation_type
            } for v in law_result.violations],
            arrests=law_result.arrests_made,
            reward_breakdowns=reward_breakdowns,
        )

        self.current_tick += 1

        return observations, rewards, done, info

    def _resolve_actions(self, actions: dict) -> dict[str, Action]:
        """Convert action inputs to Action objects."""
        resolved = {}

        for agent_id in self.agent_ids:
            action_input = actions.get(agent_id)

            if action_input is None:
                resolved[agent_id] = wait_action(agent_id)
            elif isinstance(action_input, Action):
                resolved[agent_id] = action_input
            elif isinstance(action_input, int):
                # Action index - convert to action
                resolved[agent_id] = self._index_to_action(agent_id, action_input)
            else:
                resolved[agent_id] = wait_action(agent_id)

        return resolved

    def _index_to_action(self, agent_id: str, action_idx: int) -> Action:
        """Convert action index to Action object."""
        # Simple action space:
        # 0: WAIT
        # 1: REST
        # 2: WORK
        # 3-8: MOVE to location (6 locations)
        # 9: DELIVER_QUOTA
        # 10+: CONVERSATION (with nearby agents)

        if action_idx == 0:
            return wait_action(agent_id)
        elif action_idx == 1:
            return rest_action(agent_id)
        elif action_idx == 2:
            return work_action(agent_id)
        elif 3 <= action_idx <= 8:
            locations = list(Location)
            loc_idx = action_idx - 3
            if loc_idx < len(locations):
                return move_action(agent_id, locations[loc_idx])
            return wait_action(agent_id)
        elif action_idx == 9:
            return deliver_quota_action(agent_id)
        else:
            # Conversation actions
            nearby = self.state.get_nearby_agents(agent_id)
            conv_idx = action_idx - 10
            if conv_idx < len(nearby):
                target = nearby[conv_idx]
                # Determine intent based on roles
                agent_spec = self.agent_specs.get(agent_id)
                if agent_spec and agent_spec.true_faction.value == "rebel":
                    intent = ConversationIntent.RECRUIT
                else:
                    intent = ConversationIntent.CASUAL
                return conversation_action(agent_id, target, intent)

            return wait_action(agent_id)

    def get_action_space_size(self) -> int:
        """Get the size of the action space."""
        # WAIT, REST, WORK, 6 MOVE, DELIVER, 4 CONV targets
        return 1 + 1 + 1 + 6 + 1 + 4

    def _resolve_conversation(self, pending) -> Optional[ConversationOutcome]:
        """Resolve a pending conversation."""
        if self.config.use_ai_conversations and self.conversation_engine:
            # Use AI to generate conversation
            try:
                context = create_context_from_world(
                    self.state,
                    self.agent_specs[pending.initiator_id],
                    self.agent_specs[pending.target_id],
                    pending.intent,
                )
                conv_state = self.conversation_engine.run_conversation(
                    context,
                    max_turns=self.config.max_conversation_turns,
                )
                return parse_conversation_outcome(conv_state)
            except Exception as e:
                print(f"Conversation error: {e}")
                return self._simple_conversation_resolution(pending)
        else:
            return self._simple_conversation_resolution(pending)

    def _simple_conversation_resolution(self, pending) -> ConversationOutcome:
        """Simple rule-based conversation resolution (no AI)."""
        initiator_spec = self.agent_specs.get(pending.initiator_id)
        target_spec = self.agent_specs.get(pending.target_id)
        target_state = self.state.get_agent(pending.target_id)

        # Base trust change
        trust_delta = 0.1

        # Recruitment logic
        recruitment_attempted = pending.intent == ConversationIntent.RECRUIT
        recruitment_successful = False

        if recruitment_attempted and target_state:
            # Check if recruitment could succeed
            current_trust = self.state.get_trust(pending.target_id, pending.initiator_id)

            # Success probability based on trust and location
            success_prob = 0.1 + current_trust * 0.3

            if pending.location == Location.TAVERN:
                success_prob += 0.1
            if self.state.time_of_day == TimeOfDay.NIGHT:
                success_prob += 0.1

            # Check for guards nearby (reduces success)
            for nearby_id in pending.nearby_agents:
                if self.agent_specs.get(nearby_id, {}).public_role == AgentRole.GUARD:
                    success_prob -= 0.3

            if self.rng.random() < success_prob and not target_state.is_recruited:
                recruitment_successful = True
                trust_delta = 0.3

        # Suspicion if recruitment attempted in public
        suspicion_raised = False
        suspicion_amount = 0.0
        if recruitment_attempted:
            if pending.location in {Location.MARKET, Location.TOWN_SQUARE}:
                suspicion_raised = True
                suspicion_amount = 0.1
            if pending.nearby_agents:  # Others present
                suspicion_raised = True
                suspicion_amount += 0.05

        return ConversationOutcome(
            initiator_id=pending.initiator_id,
            target_id=pending.target_id,
            turn_count=4,  # Simulated
            initiator_trust_delta=trust_delta,
            target_trust_delta=trust_delta * 0.8,
            recruitment_attempted=recruitment_attempted,
            recruitment_successful=recruitment_successful,
            suspicion_raised=suspicion_raised,
            suspicion_amount=suspicion_amount,
            was_positive=True,
        )

    def _get_observations(self) -> dict[str, Any]:
        """Get observations for all agents."""
        observations = {}

        for agent_id in self.agent_ids:
            if self.config.use_simple_observations:
                observations[agent_id] = get_simple_observation(
                    self.state, agent_id, self.agent_specs
                )
            else:
                observations[agent_id] = encode_observation(
                    self.state, agent_id, self.agent_specs
                )

        return observations

    def _check_done(self) -> bool:
        """Check if episode should end."""
        # Time limit
        if self.state.tick >= self.config.max_ticks:
            return True

        if self.state.day >= self.config.max_days:
            return True

        # All rebels arrested
        rebels_active = False
        for agent_id, spec in self.agent_specs.items():
            if spec.true_faction.value == "rebel":
                agent = self.state.get_agent(agent_id)
                if agent and agent.can_act():
                    rebels_active = True
                    break

        if not rebels_active and self.config.num_rebels > 0:
            return True

        # Rebellion exposed (optional end condition)
        if self.state.rebellion_exposed:
            return True

        return False

    def render(self) -> str:
        """Render the current state as text."""
        if self.state is None:
            return "Environment not initialized. Call reset() first."

        lines = []
        lines.append(f"\n{'='*50}")
        lines.append(f"Day {self.state.day}, {self.state.time_of_day.value.upper()} (Tick {self.state.tick})")
        lines.append(f"Rebels recruited: {self.state.total_rebels_recruited}")
        lines.append(f"{'='*50}")

        # Group agents by location
        by_location: dict[Location, list[str]] = {loc: [] for loc in Location}
        for agent_id, agent in self.state.agents.items():
            if agent.can_act():
                spec = self.agent_specs.get(agent_id)
                role = spec.public_role.value if spec else "?"
                status = f"{agent_id} ({role}) E:{agent.energy} G:{agent.gold}"
                if agent.suspicion_level > 0.3:
                    status += f" S:{agent.suspicion_level:.1f}"
                if agent.is_recruited:
                    status += " [R]"
                by_location[agent.location].append(status)

        for loc, agents in by_location.items():
            if agents:
                lines.append(f"\n{loc.value.upper()}:")
                for a in agents:
                    lines.append(f"  {a}")

        return "\n".join(lines)
