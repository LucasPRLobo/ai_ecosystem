"""
World transition rules for Prompted Town.

This module contains the core simulation logic: the step_world function
that takes a state and actions and produces a new state.

Design principles:
- Pure functions: same inputs → same outputs
- No side effects: state is cloned, not mutated
- Explicit ordering: resolution order is documented and consistent
- Event generation: all changes are logged as events
"""

from dataclasses import dataclass, field
from typing import Optional
import random

from ..core.types import (
    ActionType,
    Location,
    TimeOfDay,
    ResourceType,
    EventType,
    ConversationIntent,
)
from ..core.world_state import WorldState, AgentState, Event

from .actions import (
    Action,
    validate_action,
    wait_action,
)


# =============================================================================
# STEP RESULT
# =============================================================================

@dataclass
class PendingConversation:
    """
    A conversation that needs to be resolved by the AI engine.

    The rules engine identifies WHEN conversations happen;
    the AI engine determines WHAT is said and the outcome.
    """
    initiator_id: str
    target_id: str
    intent: ConversationIntent
    location: Location
    tick: int
    nearby_agents: list[str] = field(default_factory=list)


@dataclass
class StepResult:
    """
    Result of stepping the world forward.

    Contains the new state plus metadata about what happened.
    """
    new_state: WorldState
    events: list[Event]
    pending_conversations: list[PendingConversation]
    sanitized_actions: dict[str, Action]  # After validation


# =============================================================================
# GAME CONSTANTS (tunable parameters)
# =============================================================================

# Energy costs
ENERGY_COST_MOVE: int = 5
ENERGY_COST_WORK: int = 15
ENERGY_COST_CONVERSATION: int = 5

# Energy recovery
ENERGY_RECOVERY_REST_HOME: int = 40
ENERGY_RECOVERY_REST_OTHER: int = 20
ENERGY_RECOVERY_WAIT: int = 5

# Work production (per work action)
WORK_PRODUCTION_FARM: int = 3  # Grain produced
WORK_PRODUCTION_MARKET: int = 2  # Gold earned (trading)
WORK_PRODUCTION_TAVERN: int = 1  # Gold earned (serving)

# Work gold costs/earnings
WORK_GOLD_MARKET: int = 2  # Gold from market work
WORK_GOLD_TAVERN: int = 1  # Gold from tavern work


# =============================================================================
# MAIN STEP FUNCTION
# =============================================================================

def step_world(
    state: WorldState,
    actions: dict[str, Action],
    random_seed: Optional[int] = None,
) -> StepResult:
    """
    Advance the world by one tick.

    This is a PURE FUNCTION: given the same inputs, it always
    produces the same outputs. The original state is not modified.

    Resolution order:
    1. Validate and sanitize actions
    2. Resolve movement
    3. Resolve work/resource generation
    4. Resolve rest/energy recovery
    5. Resolve quota deliveries
    6. Identify pending conversations
    7. Advance time
    8. Apply passive effects (energy decay, etc.)

    Law enforcement is handled separately in laws.py and called
    after this function if needed.

    Args:
        state: Current world state (not modified)
        actions: Map of agent_id → Action
        random_seed: Optional seed for deterministic randomness

    Returns:
        StepResult with new state, events, and pending conversations
    """
    # Clone state to avoid mutation
    new_state = state.clone()
    new_state.clear_tick_events()

    # Initialize RNG if needed
    rng = random.Random(random_seed) if random_seed else random.Random()

    events: list[Event] = []
    pending_conversations: list[PendingConversation] = []
    sanitized_actions: dict[str, Action] = {}

    # Step 1: Validate and sanitize all actions
    for agent_id in new_state.agents.keys():
        action = actions.get(agent_id)

        if action is None:
            # No action provided, default to WAIT
            action = wait_action(agent_id)

        validation = validate_action(new_state, action)

        if validation.is_valid:
            sanitized_actions[agent_id] = action
        else:
            # Use sanitized alternative or WAIT
            sanitized = validation.sanitized_action or wait_action(agent_id)
            sanitized_actions[agent_id] = sanitized

            # Log the invalid action
            events.append(Event(
                tick=new_state.tick,
                event_type=EventType.AGENT_MOVED,  # Generic event
                agent_id=agent_id,
                details={
                    "original_action": str(action),
                    "reason": validation.reason,
                    "sanitized_to": str(sanitized),
                },
                narrative=f"{agent_id} couldn't {action.action_type.value}: {validation.reason}",
            ))

    # Step 2: Resolve movement
    move_events = _resolve_movement(new_state, sanitized_actions)
    events.extend(move_events)

    # Step 3: Resolve work
    work_events = _resolve_work(new_state, sanitized_actions, rng)
    events.extend(work_events)

    # Step 4: Resolve rest
    rest_events = _resolve_rest(new_state, sanitized_actions)
    events.extend(rest_events)

    # Step 5: Resolve quota deliveries
    quota_events = _resolve_quota_delivery(new_state, sanitized_actions)
    events.extend(quota_events)

    # Step 6: Identify pending conversations
    pending_conversations = _identify_conversations(new_state, sanitized_actions)
    for conv in pending_conversations:
        events.append(Event(
            tick=new_state.tick,
            event_type=EventType.CONVERSATION_STARTED,
            agent_id=conv.initiator_id,
            target_id=conv.target_id,
            location=conv.location,
            details={"intent": conv.intent.value},
            narrative=f"{conv.initiator_id} begins talking to {conv.target_id}",
        ))

    # Step 7: Advance time
    time_events = new_state.advance_time()
    events.extend(time_events)

    # Step 8: Apply passive effects
    passive_events = _apply_passive_effects(new_state, sanitized_actions)
    events.extend(passive_events)

    # Store all events
    for event in events:
        new_state.log_event(event)

    return StepResult(
        new_state=new_state,
        events=events,
        pending_conversations=pending_conversations,
        sanitized_actions=sanitized_actions,
    )


# =============================================================================
# RESOLUTION FUNCTIONS
# =============================================================================

def _resolve_movement(
    state: WorldState,
    actions: dict[str, Action],
) -> list[Event]:
    """Resolve all MOVE actions."""
    events = []

    for agent_id, action in actions.items():
        if action.action_type != ActionType.MOVE:
            continue

        agent = state.get_agent(agent_id)
        if agent is None:
            continue

        old_location = agent.location
        new_location = action.target_location

        # Apply movement
        agent.location = new_location
        agent.adjust_energy(-ENERGY_COST_MOVE)

        events.append(Event(
            tick=state.tick,
            event_type=EventType.AGENT_MOVED,
            agent_id=agent_id,
            location=new_location,
            details={
                "from": old_location.value,
                "to": new_location.value,
                "energy_cost": ENERGY_COST_MOVE,
            },
            narrative=f"{agent_id} travels from {old_location.value} to {new_location.value}",
        ))

    return events


def _resolve_work(
    state: WorldState,
    actions: dict[str, Action],
    rng: random.Random,
) -> list[Event]:
    """Resolve all WORK actions."""
    events = []

    for agent_id, action in actions.items():
        if action.action_type != ActionType.WORK:
            continue

        agent = state.get_agent(agent_id)
        if agent is None:
            continue

        location = agent.location
        produced = {}

        # Work output depends on location
        if location == Location.FARM:
            # Farming produces grain
            amount = WORK_PRODUCTION_FARM
            # Small random variation (±1)
            amount += rng.randint(-1, 1)
            amount = max(1, amount)
            agent.add_resource(ResourceType.GRAIN, amount)
            produced["grain"] = amount

        elif location == Location.MARKET:
            # Market work earns gold (trading, helping vendors)
            gold = WORK_GOLD_MARKET
            agent.gold += gold
            produced["gold"] = gold

        elif location == Location.TAVERN:
            # Tavern work earns less gold but good for social
            gold = WORK_GOLD_TAVERN
            agent.gold += gold
            produced["gold"] = gold

        # Deduct energy
        agent.adjust_energy(-ENERGY_COST_WORK)

        events.append(Event(
            tick=state.tick,
            event_type=EventType.WORK_COMPLETED,
            agent_id=agent_id,
            location=location,
            details={
                "produced": produced,
                "energy_cost": ENERGY_COST_WORK,
            },
            narrative=f"{agent_id} works at {location.value}, producing {produced}",
        ))

    return events


def _resolve_rest(
    state: WorldState,
    actions: dict[str, Action],
) -> list[Event]:
    """Resolve all REST actions."""
    events = []

    for agent_id, action in actions.items():
        if action.action_type != ActionType.REST:
            continue

        agent = state.get_agent(agent_id)
        if agent is None:
            continue

        # More recovery at home
        if agent.location == Location.HOME:
            recovery = ENERGY_RECOVERY_REST_HOME
        else:
            recovery = ENERGY_RECOVERY_REST_OTHER

        old_energy = agent.energy
        agent.adjust_energy(recovery)
        agent.consecutive_rest_ticks += 1

        events.append(Event(
            tick=state.tick,
            event_type=EventType.WORK_COMPLETED,  # Using generic event
            agent_id=agent_id,
            location=agent.location,
            details={
                "action": "rest",
                "energy_recovered": agent.energy - old_energy,
                "at_home": agent.location == Location.HOME,
            },
            narrative=f"{agent_id} rests at {agent.location.value}, recovering energy",
        ))

    return events


def _resolve_quota_delivery(
    state: WorldState,
    actions: dict[str, Action],
) -> list[Event]:
    """Resolve all DELIVER_QUOTA actions."""
    events = []

    for agent_id, action in actions.items():
        if action.action_type != ActionType.DELIVER_QUOTA:
            continue

        agent = state.get_agent(agent_id)
        if agent is None:
            continue

        grain_available = agent.get_resource(ResourceType.GRAIN)

        # Determine amount to deliver
        if action.delivery_amount is not None:
            amount = min(action.delivery_amount, grain_available)
        else:
            amount = grain_available

        if amount <= 0:
            continue

        # Remove grain from inventory
        agent.add_resource(ResourceType.GRAIN, -amount)

        # Track quota progress
        agent.quota_delivered_this_period += amount

        events.append(Event(
            tick=state.tick,
            event_type=EventType.QUOTA_DELIVERED,
            agent_id=agent_id,
            location=agent.location,
            details={
                "amount": amount,
                "total_this_period": agent.quota_delivered_this_period,
                "quota_required": state.laws.quota_amount,
            },
            narrative=f"{agent_id} delivers {amount} grain toward quota "
                      f"({agent.quota_delivered_this_period}/{state.laws.quota_amount})",
        ))

    return events


def _identify_conversations(
    state: WorldState,
    actions: dict[str, Action],
) -> list[PendingConversation]:
    """
    Identify conversations that need AI resolution.

    Note: This doesn't resolve conversations - that's done by the AI engine.
    We just identify which conversations should happen.
    """
    conversations = []
    already_conversing: set[str] = set()

    for agent_id, action in actions.items():
        if action.action_type != ActionType.INITIATE_CONVERSATION:
            continue

        # Skip if either party is already in a conversation this tick
        if agent_id in already_conversing or action.target_agent in already_conversing:
            continue

        agent = state.get_agent(agent_id)
        if agent is None:
            continue

        # Mark both as conversing
        already_conversing.add(agent_id)
        already_conversing.add(action.target_agent)

        # Deduct energy cost
        agent.adjust_energy(-ENERGY_COST_CONVERSATION)
        target = state.get_agent(action.target_agent)
        if target:
            target.adjust_energy(-ENERGY_COST_CONVERSATION)

        # Get nearby agents (potential witnesses)
        nearby = [
            aid for aid in state.get_agents_at_location(agent.location)
            if aid not in {agent_id, action.target_agent}
        ]

        conversations.append(PendingConversation(
            initiator_id=agent_id,
            target_id=action.target_agent,
            intent=action.conversation_intent or ConversationIntent.CASUAL,
            location=agent.location,
            tick=state.tick,
            nearby_agents=nearby,
        ))

    return conversations


def _apply_passive_effects(
    state: WorldState,
    actions: dict[str, Action],
) -> list[Event]:
    """Apply passive effects that happen regardless of actions."""
    events = []

    for agent_id, agent in state.agents.items():
        if not agent.can_act():
            continue

        action = actions.get(agent_id)

        # Reset consecutive rest counter if not resting
        if action and action.action_type != ActionType.REST:
            agent.consecutive_rest_ticks = 0

        # Small energy recovery for waiting
        if action and action.action_type == ActionType.WAIT:
            agent.adjust_energy(ENERGY_RECOVERY_WAIT)

        # Update last action tick
        agent.last_action_tick = state.tick

    return events


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_agents_who_can_see(
    state: WorldState,
    target_agent_id: str,
) -> list[str]:
    """Get list of agents who can observe the target agent."""
    target = state.get_agent(target_agent_id)
    if target is None:
        return []

    return state.get_agents_at_location(target.location)


def check_conversation_witnesses(
    state: WorldState,
    agent_a: str,
    agent_b: str,
) -> list[str]:
    """Get agents who could witness a conversation between two agents."""
    agent_a_state = state.get_agent(agent_a)
    if agent_a_state is None:
        return []

    return [
        aid for aid in state.get_agents_at_location(agent_a_state.location)
        if aid not in {agent_a, agent_b}
    ]
