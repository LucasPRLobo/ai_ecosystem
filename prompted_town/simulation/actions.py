"""
Action definitions for Prompted Town.

Actions represent what an agent WANTS to do. The rules engine
determines what actually happens.

Design principles:
- Actions are intentions, not outcomes
- Invalid actions are caught early and converted to WAIT
- Actions are immutable after creation
- All parameters needed to resolve the action are included
"""

from dataclasses import dataclass, field
from typing import Optional

from ..core.types import (
    ActionType,
    ConversationIntent,
    Location,
    ResourceType,
)
from ..core.world_state import WorldState, AgentState


# =============================================================================
# ACTION DATACLASS
# =============================================================================

@dataclass(frozen=True)
class Action:
    """
    A single action taken by an agent.

    Actions are immutable (frozen) - once created, they don't change.
    The rules engine reads actions and produces state changes.

    Different action types require different parameters:
    - MOVE: requires target_location
    - WORK: no extra params (location determines work type)
    - REST: no extra params
    - DELIVER_QUOTA: optional amount (defaults to all available)
    - INITIATE_CONVERSATION: requires target_agent, optional intent
    - WAIT: no extra params
    """
    agent_id: str
    action_type: ActionType

    # For MOVE actions
    target_location: Optional[Location] = None

    # For INITIATE_CONVERSATION actions
    target_agent: Optional[str] = None
    conversation_intent: Optional[ConversationIntent] = None

    # For DELIVER_QUOTA actions
    delivery_amount: Optional[int] = None  # None means "deliver all"

    # For WORK actions that produce specific resources
    work_target: Optional[ResourceType] = None

    def __post_init__(self):
        # Validate required parameters based on action type
        if self.action_type == ActionType.MOVE and self.target_location is None:
            raise ValueError("MOVE action requires target_location")
        if self.action_type == ActionType.INITIATE_CONVERSATION and self.target_agent is None:
            raise ValueError("INITIATE_CONVERSATION action requires target_agent")

    def __str__(self) -> str:
        """Human-readable action description."""
        if self.action_type == ActionType.MOVE:
            return f"{self.agent_id}: MOVE to {self.target_location.value}"
        elif self.action_type == ActionType.WORK:
            return f"{self.agent_id}: WORK"
        elif self.action_type == ActionType.REST:
            return f"{self.agent_id}: REST"
        elif self.action_type == ActionType.DELIVER_QUOTA:
            amt = self.delivery_amount or "all"
            return f"{self.agent_id}: DELIVER_QUOTA ({amt})"
        elif self.action_type == ActionType.INITIATE_CONVERSATION:
            intent = self.conversation_intent.value if self.conversation_intent else "casual"
            return f"{self.agent_id}: TALK to {self.target_agent} ({intent})"
        elif self.action_type == ActionType.WAIT:
            return f"{self.agent_id}: WAIT"
        return f"{self.agent_id}: {self.action_type.value}"


# =============================================================================
# ACTION FACTORIES (convenience functions)
# =============================================================================

def move_action(agent_id: str, destination: Location) -> Action:
    """Create a MOVE action."""
    return Action(
        agent_id=agent_id,
        action_type=ActionType.MOVE,
        target_location=destination,
    )


def work_action(agent_id: str, target: Optional[ResourceType] = None) -> Action:
    """Create a WORK action."""
    return Action(
        agent_id=agent_id,
        action_type=ActionType.WORK,
        work_target=target,
    )


def rest_action(agent_id: str) -> Action:
    """Create a REST action."""
    return Action(
        agent_id=agent_id,
        action_type=ActionType.REST,
    )


def deliver_quota_action(agent_id: str, amount: Optional[int] = None) -> Action:
    """Create a DELIVER_QUOTA action."""
    return Action(
        agent_id=agent_id,
        action_type=ActionType.DELIVER_QUOTA,
        delivery_amount=amount,
    )


def conversation_action(
    agent_id: str,
    target_agent: str,
    intent: ConversationIntent = ConversationIntent.CASUAL,
) -> Action:
    """Create an INITIATE_CONVERSATION action."""
    return Action(
        agent_id=agent_id,
        action_type=ActionType.INITIATE_CONVERSATION,
        target_agent=target_agent,
        conversation_intent=intent,
    )


def wait_action(agent_id: str) -> Action:
    """Create a WAIT action."""
    return Action(
        agent_id=agent_id,
        action_type=ActionType.WAIT,
    )


# =============================================================================
# ACTION VALIDATION
# =============================================================================

@dataclass
class ActionValidation:
    """Result of validating an action."""
    is_valid: bool
    reason: str = ""
    sanitized_action: Optional[Action] = None  # Alternative if invalid


def validate_action(state: WorldState, action: Action) -> ActionValidation:
    """
    Validate an action against the current world state.

    Returns ActionValidation with:
    - is_valid: True if action can be executed
    - reason: Explanation if invalid
    - sanitized_action: Replacement action if invalid (usually WAIT)
    """
    agent_state = state.get_agent(action.agent_id)

    # Check agent exists
    if agent_state is None:
        return ActionValidation(
            is_valid=False,
            reason=f"Agent {action.agent_id} does not exist",
            sanitized_action=None,
        )

    # Check agent can act
    if not agent_state.can_act():
        return ActionValidation(
            is_valid=False,
            reason=f"Agent {action.agent_id} cannot act (dead or arrested)",
            sanitized_action=wait_action(action.agent_id),
        )

    # Validate based on action type
    if action.action_type == ActionType.MOVE:
        return _validate_move(state, agent_state, action)
    elif action.action_type == ActionType.WORK:
        return _validate_work(state, agent_state, action)
    elif action.action_type == ActionType.REST:
        return _validate_rest(state, agent_state, action)
    elif action.action_type == ActionType.DELIVER_QUOTA:
        return _validate_deliver_quota(state, agent_state, action)
    elif action.action_type == ActionType.INITIATE_CONVERSATION:
        return _validate_conversation(state, agent_state, action)
    elif action.action_type == ActionType.WAIT:
        return ActionValidation(is_valid=True)

    return ActionValidation(
        is_valid=False,
        reason=f"Unknown action type: {action.action_type}",
        sanitized_action=wait_action(action.agent_id),
    )


def _validate_move(
    state: WorldState,
    agent_state: AgentState,
    action: Action
) -> ActionValidation:
    """Validate a MOVE action."""
    # Can't move to current location (waste of action)
    if action.target_location == agent_state.location:
        return ActionValidation(
            is_valid=False,
            reason=f"Already at {agent_state.location.value}",
            sanitized_action=wait_action(action.agent_id),
        )

    # Check energy - moving costs energy
    if agent_state.energy < 5:
        return ActionValidation(
            is_valid=False,
            reason="Too exhausted to move",
            sanitized_action=rest_action(action.agent_id),
        )

    return ActionValidation(is_valid=True)


def _validate_work(
    state: WorldState,
    agent_state: AgentState,
    action: Action
) -> ActionValidation:
    """Validate a WORK action."""
    # Check energy
    if agent_state.is_exhausted():
        return ActionValidation(
            is_valid=False,
            reason="Too exhausted to work",
            sanitized_action=rest_action(action.agent_id),
        )

    # Check location allows work
    workable_locations = {Location.FARM, Location.MARKET, Location.TAVERN}
    if agent_state.location not in workable_locations:
        return ActionValidation(
            is_valid=False,
            reason=f"Cannot work at {agent_state.location.value}",
            sanitized_action=wait_action(action.agent_id),
        )

    return ActionValidation(is_valid=True)


def _validate_rest(
    state: WorldState,
    agent_state: AgentState,
    action: Action
) -> ActionValidation:
    """Validate a REST action."""
    # Can always rest, but more effective at home
    return ActionValidation(is_valid=True)


def _validate_deliver_quota(
    state: WorldState,
    agent_state: AgentState,
    action: Action
) -> ActionValidation:
    """Validate a DELIVER_QUOTA action."""
    # Must be at market or town square to deliver
    if agent_state.location not in {Location.MARKET, Location.TOWN_SQUARE}:
        return ActionValidation(
            is_valid=False,
            reason="Must be at market or town square to deliver quota",
            sanitized_action=wait_action(action.agent_id),
        )

    # Must have something to deliver
    grain = agent_state.get_resource(ResourceType.GRAIN)
    if grain <= 0:
        return ActionValidation(
            is_valid=False,
            reason="No grain to deliver",
            sanitized_action=wait_action(action.agent_id),
        )

    return ActionValidation(is_valid=True)


def _validate_conversation(
    state: WorldState,
    agent_state: AgentState,
    action: Action
) -> ActionValidation:
    """Validate an INITIATE_CONVERSATION action."""
    # Check target exists
    target_state = state.get_agent(action.target_agent)
    if target_state is None:
        return ActionValidation(
            is_valid=False,
            reason=f"Target agent {action.target_agent} does not exist",
            sanitized_action=wait_action(action.agent_id),
        )

    # Check target can participate
    if not target_state.can_act():
        return ActionValidation(
            is_valid=False,
            reason=f"Target {action.target_agent} cannot converse",
            sanitized_action=wait_action(action.agent_id),
        )

    # Must be at same location
    if agent_state.location != target_state.location:
        return ActionValidation(
            is_valid=False,
            reason=f"Target {action.target_agent} is not at same location",
            sanitized_action=wait_action(action.agent_id),
        )

    # Can't talk to yourself
    if action.target_agent == action.agent_id:
        return ActionValidation(
            is_valid=False,
            reason="Cannot converse with yourself",
            sanitized_action=wait_action(action.agent_id),
        )

    return ActionValidation(is_valid=True)


# =============================================================================
# ACTION SPACE GENERATION
# =============================================================================

def get_valid_actions(
    state: WorldState,
    agent_id: str,
    include_conversation_targets: bool = True,
) -> list[Action]:
    """
    Get all valid actions for an agent in the current state.

    This is useful for:
    - RL action space enumeration
    - UI showing available options
    - Testing

    Args:
        state: Current world state
        agent_id: The agent to get actions for
        include_conversation_targets: If True, enumerate all possible
            conversation targets. If False, just include one generic
            conversation action.

    Returns:
        List of valid Action objects
    """
    agent_state = state.get_agent(agent_id)
    if agent_state is None or not agent_state.can_act():
        return []

    valid_actions: list[Action] = []

    # WAIT is always valid
    valid_actions.append(wait_action(agent_id))

    # REST is always valid
    valid_actions.append(rest_action(agent_id))

    # MOVE to each other location
    for location in Location:
        if location != agent_state.location:
            action = move_action(agent_id, location)
            validation = validate_action(state, action)
            if validation.is_valid:
                valid_actions.append(action)

    # WORK if at workable location
    work = work_action(agent_id)
    if validate_action(state, work).is_valid:
        valid_actions.append(work)

    # DELIVER_QUOTA if possible
    deliver = deliver_quota_action(agent_id)
    if validate_action(state, deliver).is_valid:
        valid_actions.append(deliver)

    # INITIATE_CONVERSATION with nearby agents
    if include_conversation_targets:
        nearby = state.get_nearby_agents(agent_id)
        for target_id in nearby:
            for intent in ConversationIntent:
                action = conversation_action(agent_id, target_id, intent)
                if validate_action(state, action).is_valid:
                    valid_actions.append(action)

    return valid_actions


def get_action_space_size(include_all_conversations: bool = True) -> int:
    """
    Get the maximum size of the action space.

    Useful for RL agent initialization.

    Note: Actual valid actions at any state will be fewer.
    """
    base_actions = 3  # WAIT, REST, WORK
    move_actions = len(Location) - 1  # Can move to any other location
    deliver_actions = 1

    if include_all_conversations:
        # Worst case: all other agents at same location
        max_agents = 10  # Reasonable upper bound
        conversation_intents = len(ConversationIntent)
        conversation_actions = max_agents * conversation_intents
    else:
        conversation_actions = 1

    return base_actions + move_actions + deliver_actions + conversation_actions
