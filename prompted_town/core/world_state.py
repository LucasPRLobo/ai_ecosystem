"""
World state dataclasses for Prompted Town.

This module defines all mutable state in the simulation:
- AgentState: Current condition of a single agent
- RelationshipState: State of relationship between two agents
- WorldState: Complete simulation snapshot

Design principles:
- All state is serializable (dataclasses with simple types)
- State is separate from specification (AgentSpec is immutable)
- WorldState is a complete snapshot - given a WorldState, you can
  resume simulation without any other context
"""

from dataclasses import dataclass, field
from typing import Optional
import json
from copy import deepcopy

from .types import (
    Location,
    TimeOfDay,
    ResourceType,
    EmotionalState,
    EventType,
    RelationshipLevel,
    trust_to_relationship_level,
    next_time_of_day,
    TIME_CYCLE,
)


# =============================================================================
# CONFIGURATION (affects rules but doesn't change during simulation)
# =============================================================================

@dataclass
class LawConfig:
    """
    Configuration for the town's laws.

    This defines the rules agents must follow and the consequences
    of breaking them. Set once at world creation.
    """
    # Quota settings
    quota_amount: int = 10  # How much each farmer must deliver
    quota_deadline_day: int = 7  # Day of the week quota is due (0-6)
    quota_penalty_gold: int = 5  # Fine for missing quota
    quota_penalty_suspicion: float = 0.2  # Suspicion increase for missing

    # Curfew settings
    curfew_enabled: bool = True
    curfew_start: TimeOfDay = TimeOfDay.NIGHT
    curfew_end: TimeOfDay = TimeOfDay.DAWN
    curfew_allowed_locations: set[Location] = field(
        default_factory=lambda: {Location.HOME, Location.TAVERN}
    )
    curfew_violation_suspicion: float = 0.3  # Suspicion for being caught

    # Suspicion thresholds
    suspicion_interrogation_threshold: float = 0.5  # Guards may interrogate
    suspicion_arrest_threshold: float = 0.8  # Guards will arrest

    def is_curfew_active(self, time_of_day: TimeOfDay) -> bool:
        """Check if curfew is currently in effect."""
        if not self.curfew_enabled:
            return False
        # Curfew is active from start until end (handles overnight)
        start_idx = TIME_CYCLE.index(self.curfew_start)
        end_idx = TIME_CYCLE.index(self.curfew_end)
        current_idx = TIME_CYCLE.index(time_of_day)

        if start_idx <= end_idx:
            return start_idx <= current_idx < end_idx
        else:  # Wraps around midnight
            return current_idx >= start_idx or current_idx < end_idx


# =============================================================================
# AGENT STATE (mutable, changes each tick)
# =============================================================================

@dataclass
class AgentState:
    """
    The current state of a single agent.

    This is separate from AgentSpec (which is the agent's unchanging
    personality and goals). AgentState changes every tick.
    """
    # Reference to spec (by ID, not the object itself for serialization)
    agent_id: str

    # Physical state
    location: Location = Location.HOME
    energy: int = 100  # 0-100, depletes with actions, restored by rest
    is_alive: bool = True
    is_arrested: bool = False

    # Economic state
    gold: int = 10
    inventory: dict[ResourceType, int] = field(default_factory=dict)

    # Quota tracking (per quota period)
    quota_delivered_this_period: int = 0

    # Social/faction state
    is_recruited: bool = False  # Has joined the rebellion
    known_rebels: set[str] = field(default_factory=set)  # Agent IDs they know are rebels

    # Mental/emotional state
    emotional_state: EmotionalState = EmotionalState.CALM
    suspicion_level: float = 0.0  # 0.0 to 1.0, how much authorities suspect them

    # Information state (things they've learned)
    known_information: set[str] = field(default_factory=set)

    # Activity tracking
    last_action_tick: int = 0
    consecutive_rest_ticks: int = 0

    def __post_init__(self):
        # Initialize empty inventory with zeros for all resource types
        for resource in ResourceType:
            if resource not in self.inventory:
                self.inventory[resource] = 0

    def get_resource(self, resource: ResourceType) -> int:
        """Get amount of a resource in inventory."""
        return self.inventory.get(resource, 0)

    def add_resource(self, resource: ResourceType, amount: int) -> None:
        """Add resource to inventory (can be negative to remove)."""
        current = self.inventory.get(resource, 0)
        self.inventory[resource] = max(0, current + amount)

    def adjust_energy(self, delta: int) -> None:
        """Adjust energy, clamping to 0-100."""
        self.energy = max(0, min(100, self.energy + delta))

    def adjust_suspicion(self, delta: float) -> None:
        """Adjust suspicion, clamping to 0.0-1.0."""
        self.suspicion_level = max(0.0, min(1.0, self.suspicion_level + delta))

    def is_exhausted(self) -> bool:
        """Check if agent is too tired to work."""
        return self.energy < 20

    def can_act(self) -> bool:
        """Check if agent can take actions."""
        return self.is_alive and not self.is_arrested


# =============================================================================
# RELATIONSHIP STATE
# =============================================================================

@dataclass
class RelationshipState:
    """
    The state of a relationship between two agents.

    Relationships are bidirectional but may be asymmetric
    (A trusts B more than B trusts A).
    """
    # The two agents in the relationship (ordered tuple for consistency)
    agent_a_id: str
    agent_b_id: str

    # Trust levels (separate for each direction)
    trust_a_to_b: float = 0.0  # -1.0 (hostile) to 1.0 (complete trust)
    trust_b_to_a: float = 0.0

    # Interaction history
    total_interactions: int = 0
    positive_interactions: int = 0
    negative_interactions: int = 0
    last_interaction_tick: int = -1

    # Special flags
    a_knows_b_is_rebel: bool = False
    b_knows_a_is_rebel: bool = False

    def get_trust(self, from_id: str, to_id: str) -> float:
        """Get trust level from one agent to another."""
        if from_id == self.agent_a_id and to_id == self.agent_b_id:
            return self.trust_a_to_b
        elif from_id == self.agent_b_id and to_id == self.agent_a_id:
            return self.trust_b_to_a
        else:
            raise ValueError(f"Agents {from_id}, {to_id} not in this relationship")

    def set_trust(self, from_id: str, to_id: str, value: float) -> None:
        """Set trust level, clamped to [-1, 1]."""
        value = max(-1.0, min(1.0, value))
        if from_id == self.agent_a_id and to_id == self.agent_b_id:
            self.trust_a_to_b = value
        elif from_id == self.agent_b_id and to_id == self.agent_a_id:
            self.trust_b_to_a = value
        else:
            raise ValueError(f"Agents {from_id}, {to_id} not in this relationship")

    def adjust_trust(self, from_id: str, to_id: str, delta: float) -> None:
        """Adjust trust level by delta."""
        current = self.get_trust(from_id, to_id)
        self.set_trust(from_id, to_id, current + delta)

    def get_relationship_level(self, from_id: str, to_id: str) -> RelationshipLevel:
        """Get qualitative relationship level."""
        trust = self.get_trust(from_id, to_id)
        return trust_to_relationship_level(trust)

    def record_interaction(self, tick: int, positive: bool) -> None:
        """Record that an interaction occurred."""
        self.total_interactions += 1
        self.last_interaction_tick = tick
        if positive:
            self.positive_interactions += 1
        else:
            self.negative_interactions += 1


# =============================================================================
# EVENT LOG
# =============================================================================

@dataclass
class Event:
    """
    A single event that occurred in the simulation.

    Events are logged for:
    - Human-readable narration
    - Reward computation
    - Analysis and debugging
    """
    tick: int
    event_type: EventType
    agent_id: Optional[str] = None  # Primary agent involved
    target_id: Optional[str] = None  # Secondary agent (if any)
    location: Optional[Location] = None
    details: dict = field(default_factory=dict)
    narrative: str = ""  # Human-readable description

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "tick": self.tick,
            "event_type": self.event_type.value,
            "agent_id": self.agent_id,
            "target_id": self.target_id,
            "location": self.location.value if self.location else None,
            "details": self.details,
            "narrative": self.narrative,
        }


# =============================================================================
# WORLD STATE (complete simulation snapshot)
# =============================================================================

@dataclass
class WorldState:
    """
    Complete state of the simulation at a point in time.

    This is everything needed to:
    - Resume simulation from this point
    - Compute observations for agents
    - Render the current state

    Fully serializable to JSON.
    """
    # Time tracking
    tick: int = 0
    day: int = 0
    time_of_day: TimeOfDay = TimeOfDay.DAWN

    # Agent states (keyed by agent_id)
    agents: dict[str, AgentState] = field(default_factory=dict)

    # Relationships (keyed by tuple of sorted agent IDs)
    # We use a string key for JSON serialization: "agent_a|agent_b"
    relationships: dict[str, RelationshipState] = field(default_factory=dict)

    # Law configuration
    laws: LawConfig = field(default_factory=LawConfig)

    # Global state
    total_rebels_recruited: int = 0
    rebellion_exposed: bool = False

    # Event log for this tick (cleared each tick, for current tick only)
    current_tick_events: list[Event] = field(default_factory=list)

    # Random seed used (for determinism verification)
    random_seed: Optional[int] = None

    # =========================================================================
    # Agent management
    # =========================================================================

    def get_agent(self, agent_id: str) -> Optional[AgentState]:
        """Get agent state by ID."""
        return self.agents.get(agent_id)

    def get_agents_at_location(self, location: Location) -> list[str]:
        """Get IDs of all agents at a location."""
        return [
            agent_id
            for agent_id, state in self.agents.items()
            if state.location == location and state.can_act()
        ]

    def get_nearby_agents(self, agent_id: str) -> list[str]:
        """Get IDs of other agents at the same location."""
        agent = self.get_agent(agent_id)
        if not agent:
            return []
        return [
            aid for aid in self.get_agents_at_location(agent.location)
            if aid != agent_id
        ]

    # =========================================================================
    # Relationship management
    # =========================================================================

    @staticmethod
    def _relationship_key(agent_a: str, agent_b: str) -> str:
        """Generate consistent key for relationship lookup."""
        return "|".join(sorted([agent_a, agent_b]))

    def get_relationship(self, agent_a: str, agent_b: str) -> RelationshipState:
        """Get or create relationship between two agents."""
        key = self._relationship_key(agent_a, agent_b)
        if key not in self.relationships:
            # Create new relationship with consistent ordering
            sorted_ids = sorted([agent_a, agent_b])
            self.relationships[key] = RelationshipState(
                agent_a_id=sorted_ids[0],
                agent_b_id=sorted_ids[1],
            )
        return self.relationships[key]

    def get_trust(self, from_agent: str, to_agent: str) -> float:
        """Get trust level from one agent to another."""
        rel = self.get_relationship(from_agent, to_agent)
        return rel.get_trust(from_agent, to_agent)

    # =========================================================================
    # Time management
    # =========================================================================

    def advance_time(self) -> list[Event]:
        """
        Advance time by one tick.

        Returns events generated by time advancement (e.g., day change).
        """
        events = []
        self.tick += 1

        old_time = self.time_of_day
        self.time_of_day = next_time_of_day(self.time_of_day)

        # Check for day rollover
        if self.time_of_day == TimeOfDay.DAWN and old_time == TimeOfDay.NIGHT:
            self.day += 1
            events.append(Event(
                tick=self.tick,
                event_type=EventType.DAY_ENDED,
                details={"new_day": self.day},
                narrative=f"Day {self.day} begins.",
            ))

        events.append(Event(
            tick=self.tick,
            event_type=EventType.TIME_ADVANCED,
            details={"time_of_day": self.time_of_day.value},
            narrative=f"It is now {self.time_of_day.value}.",
        ))

        return events

    # =========================================================================
    # Event logging
    # =========================================================================

    def log_event(self, event: Event) -> None:
        """Add an event to the current tick's log."""
        self.current_tick_events.append(event)

    def clear_tick_events(self) -> list[Event]:
        """Clear and return current tick's events."""
        events = self.current_tick_events
        self.current_tick_events = []
        return events

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> dict:
        """Convert entire world state to a dictionary for serialization."""
        return {
            "tick": self.tick,
            "day": self.day,
            "time_of_day": self.time_of_day.value,
            "agents": {
                agent_id: {
                    "agent_id": state.agent_id,
                    "location": state.location.value,
                    "energy": state.energy,
                    "is_alive": state.is_alive,
                    "is_arrested": state.is_arrested,
                    "gold": state.gold,
                    "inventory": {k.value: v for k, v in state.inventory.items()},
                    "quota_delivered_this_period": state.quota_delivered_this_period,
                    "is_recruited": state.is_recruited,
                    "known_rebels": list(state.known_rebels),
                    "emotional_state": state.emotional_state.value,
                    "suspicion_level": state.suspicion_level,
                    "known_information": list(state.known_information),
                    "last_action_tick": state.last_action_tick,
                    "consecutive_rest_ticks": state.consecutive_rest_ticks,
                }
                for agent_id, state in self.agents.items()
            },
            "relationships": {
                key: {
                    "agent_a_id": rel.agent_a_id,
                    "agent_b_id": rel.agent_b_id,
                    "trust_a_to_b": rel.trust_a_to_b,
                    "trust_b_to_a": rel.trust_b_to_a,
                    "total_interactions": rel.total_interactions,
                    "positive_interactions": rel.positive_interactions,
                    "negative_interactions": rel.negative_interactions,
                    "last_interaction_tick": rel.last_interaction_tick,
                    "a_knows_b_is_rebel": rel.a_knows_b_is_rebel,
                    "b_knows_a_is_rebel": rel.b_knows_a_is_rebel,
                }
                for key, rel in self.relationships.items()
            },
            "laws": {
                "quota_amount": self.laws.quota_amount,
                "quota_deadline_day": self.laws.quota_deadline_day,
                "quota_penalty_gold": self.laws.quota_penalty_gold,
                "quota_penalty_suspicion": self.laws.quota_penalty_suspicion,
                "curfew_enabled": self.laws.curfew_enabled,
                "curfew_start": self.laws.curfew_start.value,
                "curfew_end": self.laws.curfew_end.value,
                "curfew_allowed_locations": [loc.value for loc in self.laws.curfew_allowed_locations],
                "curfew_violation_suspicion": self.laws.curfew_violation_suspicion,
                "suspicion_interrogation_threshold": self.laws.suspicion_interrogation_threshold,
                "suspicion_arrest_threshold": self.laws.suspicion_arrest_threshold,
            },
            "total_rebels_recruited": self.total_rebels_recruited,
            "rebellion_exposed": self.rebellion_exposed,
            "random_seed": self.random_seed,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "WorldState":
        """Reconstruct WorldState from dictionary."""
        # Reconstruct law config
        law_data = data.get("laws", {})
        laws = LawConfig(
            quota_amount=law_data.get("quota_amount", 10),
            quota_deadline_day=law_data.get("quota_deadline_day", 7),
            quota_penalty_gold=law_data.get("quota_penalty_gold", 5),
            quota_penalty_suspicion=law_data.get("quota_penalty_suspicion", 0.2),
            curfew_enabled=law_data.get("curfew_enabled", True),
            curfew_start=TimeOfDay(law_data.get("curfew_start", "night")),
            curfew_end=TimeOfDay(law_data.get("curfew_end", "dawn")),
            curfew_allowed_locations={
                Location(loc) for loc in law_data.get("curfew_allowed_locations", ["home", "tavern"])
            },
            curfew_violation_suspicion=law_data.get("curfew_violation_suspicion", 0.3),
            suspicion_interrogation_threshold=law_data.get("suspicion_interrogation_threshold", 0.5),
            suspicion_arrest_threshold=law_data.get("suspicion_arrest_threshold", 0.8),
        )

        # Reconstruct agent states
        agents = {}
        for agent_id, agent_data in data.get("agents", {}).items():
            agents[agent_id] = AgentState(
                agent_id=agent_data["agent_id"],
                location=Location(agent_data["location"]),
                energy=agent_data["energy"],
                is_alive=agent_data["is_alive"],
                is_arrested=agent_data["is_arrested"],
                gold=agent_data["gold"],
                inventory={
                    ResourceType(k): v
                    for k, v in agent_data.get("inventory", {}).items()
                },
                quota_delivered_this_period=agent_data.get("quota_delivered_this_period", 0),
                is_recruited=agent_data.get("is_recruited", False),
                known_rebels=set(agent_data.get("known_rebels", [])),
                emotional_state=EmotionalState(agent_data.get("emotional_state", "calm")),
                suspicion_level=agent_data.get("suspicion_level", 0.0),
                known_information=set(agent_data.get("known_information", [])),
                last_action_tick=agent_data.get("last_action_tick", 0),
                consecutive_rest_ticks=agent_data.get("consecutive_rest_ticks", 0),
            )

        # Reconstruct relationships
        relationships = {}
        for key, rel_data in data.get("relationships", {}).items():
            relationships[key] = RelationshipState(
                agent_a_id=rel_data["agent_a_id"],
                agent_b_id=rel_data["agent_b_id"],
                trust_a_to_b=rel_data["trust_a_to_b"],
                trust_b_to_a=rel_data["trust_b_to_a"],
                total_interactions=rel_data["total_interactions"],
                positive_interactions=rel_data["positive_interactions"],
                negative_interactions=rel_data["negative_interactions"],
                last_interaction_tick=rel_data["last_interaction_tick"],
                a_knows_b_is_rebel=rel_data.get("a_knows_b_is_rebel", False),
                b_knows_a_is_rebel=rel_data.get("b_knows_a_is_rebel", False),
            )

        return cls(
            tick=data.get("tick", 0),
            day=data.get("day", 0),
            time_of_day=TimeOfDay(data.get("time_of_day", "dawn")),
            agents=agents,
            relationships=relationships,
            laws=laws,
            total_rebels_recruited=data.get("total_rebels_recruited", 0),
            rebellion_exposed=data.get("rebellion_exposed", False),
            random_seed=data.get("random_seed"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "WorldState":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def clone(self) -> "WorldState":
        """Create a deep copy of this state."""
        return deepcopy(self)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_default_world(
    agent_ids: Optional[list[str]] = None,
    random_seed: Optional[int] = None,
) -> WorldState:
    """
    Create a default world state with basic setup.

    This creates the world state only - agent specs are created separately
    and linked by agent_id.

    Args:
        agent_ids: List of agent IDs to create states for.
                   If None, creates a default set.
        random_seed: Seed for deterministic simulation.

    Returns:
        A new WorldState ready for simulation.
    """
    if agent_ids is None:
        agent_ids = [
            "farmer_01", "farmer_02", "farmer_03",
            "guard_01", "guard_02",
            "rebel_01",
            "merchant_01",
        ]

    world = WorldState(
        tick=0,
        day=0,
        time_of_day=TimeOfDay.DAWN,
        random_seed=random_seed,
    )

    # Create agent states (all start at home with default values)
    for agent_id in agent_ids:
        world.agents[agent_id] = AgentState(
            agent_id=agent_id,
            location=Location.HOME,
        )

    return world
