"""
Law enforcement system for Prompted Town.

This module handles:
- Curfew violation detection and consequences
- Quota compliance checking
- Suspicion accumulation and decay
- Guard patrol and detection logic

Design principles:
- Enforcement is separate from main simulation step
- Violations produce events and state changes
- Guards must be present to enforce (no omniscient detection)
"""

from dataclasses import dataclass, field
from typing import Optional

from ..core.types import (
    Location,
    TimeOfDay,
    EventType,
    AgentRole,
)
from ..core.world_state import WorldState, Event, LawConfig
from ..core.agent_spec import AgentSpec


# =============================================================================
# VIOLATION TYPES
# =============================================================================

@dataclass
class Violation:
    """Record of a law violation."""
    agent_id: str
    violation_type: str  # "curfew", "quota_missed", etc.
    tick: int
    location: Optional[Location] = None
    details: dict = field(default_factory=dict)
    witnessed_by: list[str] = field(default_factory=list)


@dataclass
class EnforcementResult:
    """Result of running law enforcement checks."""
    violations: list[Violation]
    events: list[Event]
    arrests_made: list[str]
    interrogations_started: list[tuple[str, str]]  # (guard_id, suspect_id)


# =============================================================================
# CURFEW ENFORCEMENT
# =============================================================================

def check_curfew_violations(
    state: WorldState,
    agent_specs: dict[str, AgentSpec],
) -> list[Violation]:
    """
    Check for curfew violations.

    A violation occurs when:
    1. Curfew is active (based on time of day)
    2. Agent is in a non-allowed location
    3. A guard is present at that location (to witness it)

    Guards are exempt from curfew.
    """
    violations = []

    if not state.laws.is_curfew_active(state.time_of_day):
        return violations

    for agent_id, agent_state in state.agents.items():
        if not agent_state.can_act():
            continue

        # Get agent spec to check role
        spec = agent_specs.get(agent_id)
        if spec and spec.public_role == AgentRole.GUARD:
            continue  # Guards are exempt

        # Check if in allowed location
        if agent_state.location in state.laws.curfew_allowed_locations:
            continue

        # Check if a guard is present to witness
        guards_present = [
            aid for aid in state.get_agents_at_location(agent_state.location)
            if agent_specs.get(aid) and agent_specs[aid].public_role == AgentRole.GUARD
        ]

        if guards_present:
            violations.append(Violation(
                agent_id=agent_id,
                violation_type="curfew",
                tick=state.tick,
                location=agent_state.location,
                details={
                    "time_of_day": state.time_of_day.value,
                    "allowed_locations": [loc.value for loc in state.laws.curfew_allowed_locations],
                },
                witnessed_by=guards_present,
            ))

    return violations


def apply_curfew_consequences(
    state: WorldState,
    violations: list[Violation],
) -> list[Event]:
    """Apply consequences for curfew violations."""
    events = []

    for violation in violations:
        if violation.violation_type != "curfew":
            continue

        agent = state.get_agent(violation.agent_id)
        if agent is None:
            continue

        # Increase suspicion
        old_suspicion = agent.suspicion_level
        agent.adjust_suspicion(state.laws.curfew_violation_suspicion)

        events.append(Event(
            tick=state.tick,
            event_type=EventType.CURFEW_VIOLATION,
            agent_id=violation.agent_id,
            location=violation.location,
            details={
                "witnessed_by": violation.witnessed_by,
                "suspicion_increase": state.laws.curfew_violation_suspicion,
                "new_suspicion": agent.suspicion_level,
            },
            narrative=f"{violation.agent_id} caught violating curfew at {violation.location.value} "
                      f"by {', '.join(violation.witnessed_by)}. Suspicion: {old_suspicion:.2f} → {agent.suspicion_level:.2f}",
        ))

    return events


# =============================================================================
# QUOTA ENFORCEMENT
# =============================================================================

def check_quota_compliance(
    state: WorldState,
    agent_specs: dict[str, AgentSpec],
) -> list[Violation]:
    """
    Check for quota violations.

    Called at the quota deadline (when day % 7 == quota_deadline_day).
    Only farmers are required to meet quota.
    """
    violations = []

    # Only check on quota deadline day
    if state.day % 7 != state.laws.quota_deadline_day:
        return violations

    # Only check at dawn (start of deadline day)
    if state.time_of_day != TimeOfDay.DAWN:
        return violations

    for agent_id, agent_state in state.agents.items():
        if not agent_state.can_act():
            continue

        # Only farmers have quota
        spec = agent_specs.get(agent_id)
        if not spec or spec.public_role != AgentRole.FARMER:
            continue

        # Check if quota met
        if agent_state.quota_delivered_this_period < state.laws.quota_amount:
            violations.append(Violation(
                agent_id=agent_id,
                violation_type="quota_missed",
                tick=state.tick,
                details={
                    "delivered": agent_state.quota_delivered_this_period,
                    "required": state.laws.quota_amount,
                    "shortfall": state.laws.quota_amount - agent_state.quota_delivered_this_period,
                },
            ))

    return violations


def apply_quota_consequences(
    state: WorldState,
    violations: list[Violation],
) -> list[Event]:
    """Apply consequences for missing quota."""
    events = []

    for violation in violations:
        if violation.violation_type != "quota_missed":
            continue

        agent = state.get_agent(violation.agent_id)
        if agent is None:
            continue

        # Deduct fine from gold
        fine = state.laws.quota_penalty_gold
        old_gold = agent.gold
        agent.gold = max(0, agent.gold - fine)
        actual_fine = old_gold - agent.gold

        # Increase suspicion
        old_suspicion = agent.suspicion_level
        agent.adjust_suspicion(state.laws.quota_penalty_suspicion)

        events.append(Event(
            tick=state.tick,
            event_type=EventType.QUOTA_MISSED,
            agent_id=violation.agent_id,
            details={
                "shortfall": violation.details["shortfall"],
                "fine": actual_fine,
                "suspicion_increase": state.laws.quota_penalty_suspicion,
            },
            narrative=f"{violation.agent_id} missed quota by {violation.details['shortfall']}. "
                      f"Fined {actual_fine} gold. Suspicion: {old_suspicion:.2f} → {agent.suspicion_level:.2f}",
        ))

    return violations_to_events(violations, state.tick)


def reset_quota_tracking(state: WorldState) -> None:
    """Reset quota tracking at start of new period."""
    for agent in state.agents.values():
        agent.quota_delivered_this_period = 0


def violations_to_events(violations: list[Violation], tick: int) -> list[Event]:
    """Convert violations to events for logging."""
    events = []
    for v in violations:
        if v.violation_type == "quota_missed":
            events.append(Event(
                tick=tick,
                event_type=EventType.QUOTA_MISSED,
                agent_id=v.agent_id,
                details=v.details,
                narrative=f"{v.agent_id} failed to meet quota",
            ))
    return events


# =============================================================================
# SUSPICION SYSTEM
# =============================================================================

def decay_suspicion(
    state: WorldState,
    decay_rate: float = 0.02,
) -> list[Event]:
    """
    Gradually reduce suspicion over time.

    Called each tick. Suspicion decays slowly if agent
    isn't doing anything suspicious.
    """
    events = []

    for agent_id, agent in state.agents.items():
        if not agent.can_act():
            continue

        if agent.suspicion_level > 0:
            old = agent.suspicion_level
            agent.adjust_suspicion(-decay_rate)

            # Only log significant decay
            if old - agent.suspicion_level > 0.05:
                events.append(Event(
                    tick=state.tick,
                    event_type=EventType.SUSPICION_RAISED,  # Using for decay too
                    agent_id=agent_id,
                    details={
                        "change": -decay_rate,
                        "new_level": agent.suspicion_level,
                        "reason": "time_decay",
                    },
                    narrative=f"{agent_id}'s suspicion decays slightly",
                ))

    return events


def check_arrest_threshold(
    state: WorldState,
    agent_specs: dict[str, AgentSpec],
) -> list[tuple[str, str]]:
    """
    Check if any agents should be arrested.

    Returns list of (guard_id, arrested_id) pairs.

    Arrest happens when:
    1. Agent suspicion >= arrest threshold
    2. A guard is at the same location
    """
    arrests = []

    for agent_id, agent in state.agents.items():
        if not agent.can_act():
            continue

        if agent.suspicion_level < state.laws.suspicion_arrest_threshold:
            continue

        # Find a guard at same location
        guards_present = [
            aid for aid in state.get_agents_at_location(agent.location)
            if agent_specs.get(aid) and agent_specs[aid].public_role == AgentRole.GUARD
        ]

        if guards_present:
            arrests.append((guards_present[0], agent_id))

    return arrests


def apply_arrest(
    state: WorldState,
    guard_id: str,
    arrested_id: str,
) -> Event:
    """Apply arrest to an agent."""
    agent = state.get_agent(arrested_id)
    if agent:
        agent.is_arrested = True
        agent.location = Location.GATE  # Taken to gate/jail

    return Event(
        tick=state.tick,
        event_type=EventType.ARREST,
        agent_id=arrested_id,
        target_id=guard_id,
        location=Location.GATE,
        details={
            "guard": guard_id,
            "suspicion_level": agent.suspicion_level if agent else 0,
        },
        narrative=f"{arrested_id} has been arrested by {guard_id}!",
    )


def check_interrogation_threshold(
    state: WorldState,
    agent_specs: dict[str, AgentSpec],
) -> list[tuple[str, str]]:
    """
    Check if any agents should be interrogated.

    Returns list of (guard_id, suspect_id) pairs.

    Interrogation happens when:
    1. Agent suspicion >= interrogation threshold but < arrest threshold
    2. A guard is at the same location
    3. Random chance based on suspicion level
    """
    interrogations = []

    for agent_id, agent in state.agents.items():
        if not agent.can_act():
            continue

        # In interrogation range?
        if not (state.laws.suspicion_interrogation_threshold
                <= agent.suspicion_level
                < state.laws.suspicion_arrest_threshold):
            continue

        # Find a guard at same location
        guards_present = [
            aid for aid in state.get_agents_at_location(agent.location)
            if agent_specs.get(aid) and agent_specs[aid].public_role == AgentRole.GUARD
        ]

        if guards_present:
            interrogations.append((guards_present[0], agent_id))

    return interrogations


# =============================================================================
# MAIN ENFORCEMENT FUNCTION
# =============================================================================

def enforce_laws(
    state: WorldState,
    agent_specs: dict[str, AgentSpec],
    apply_consequences: bool = True,
) -> EnforcementResult:
    """
    Run all law enforcement checks.

    This is called after step_world to handle law-related state changes.

    Args:
        state: Current world state (will be modified if apply_consequences=True)
        agent_specs: Agent specifications (needed to identify guards, farmers)
        apply_consequences: If True, apply state changes. If False, just detect.

    Returns:
        EnforcementResult with all violations and events.
    """
    all_violations = []
    all_events = []
    arrests_made = []
    interrogations = []

    # Check curfew
    curfew_violations = check_curfew_violations(state, agent_specs)
    all_violations.extend(curfew_violations)
    if apply_consequences and curfew_violations:
        events = apply_curfew_consequences(state, curfew_violations)
        all_events.extend(events)

    # Check quota (only on deadline)
    quota_violations = check_quota_compliance(state, agent_specs)
    all_violations.extend(quota_violations)
    if apply_consequences and quota_violations:
        events = apply_quota_consequences(state, quota_violations)
        all_events.extend(events)

        # Reset quota tracking for new period
        reset_quota_tracking(state)

    # Decay suspicion
    if apply_consequences:
        decay_events = decay_suspicion(state)
        all_events.extend(decay_events)

    # Check for arrests
    arrest_pairs = check_arrest_threshold(state, agent_specs)
    if apply_consequences:
        for guard_id, arrested_id in arrest_pairs:
            event = apply_arrest(state, guard_id, arrested_id)
            all_events.append(event)
            arrests_made.append(arrested_id)

    # Check for interrogations
    interrogation_pairs = check_interrogation_threshold(state, agent_specs)
    interrogations = interrogation_pairs  # These become pending conversations

    # Log events
    for event in all_events:
        state.log_event(event)

    return EnforcementResult(
        violations=all_violations,
        events=all_events,
        arrests_made=arrests_made,
        interrogations_started=interrogations,
    )
