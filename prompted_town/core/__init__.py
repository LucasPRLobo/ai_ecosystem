"""
Core module: Types, specifications, and state definitions.

This module provides the foundational data structures used throughout
the Prompted Town simulation.

Exports:
- Enums: Location, TimeOfDay, ActionType, AgentRole, etc.
- Agent specs: AgentSpec, AIPersonality, RLProfile
- World state: WorldState, AgentState, RelationshipState
- Factories: create_farmer_spec, create_guard_spec, create_rebel_spec
"""

# Types and enums
from .types import (
    Location,
    TimeOfDay,
    TIME_CYCLE,
    next_time_of_day,
    AgentRole,
    Faction,
    ActionType,
    ConversationIntent,
    RelationshipLevel,
    trust_to_relationship_level,
    EmotionalState,
    ResourceType,
    EventType,
    TraitName,
    GoalType,
)

# Agent specifications
from .agent_spec import (
    Goal,
    RLProfile,
    SpeakingStyle,
    AIPersonality,
    AgentSpec,
    create_farmer_spec,
    create_guard_spec,
    create_rebel_spec,
)

# World state
from .world_state import (
    LawConfig,
    AgentState,
    RelationshipState,
    Event,
    WorldState,
    create_default_world,
)

__all__ = [
    # Enums
    "Location",
    "TimeOfDay",
    "TIME_CYCLE",
    "next_time_of_day",
    "AgentRole",
    "Faction",
    "ActionType",
    "ConversationIntent",
    "RelationshipLevel",
    "trust_to_relationship_level",
    "EmotionalState",
    "ResourceType",
    "EventType",
    "TraitName",
    "GoalType",
    # Agent specs
    "Goal",
    "RLProfile",
    "SpeakingStyle",
    "AIPersonality",
    "AgentSpec",
    "create_farmer_spec",
    "create_guard_spec",
    "create_rebel_spec",
    # World state
    "LawConfig",
    "AgentState",
    "RelationshipState",
    "Event",
    "WorldState",
    "create_default_world",
]
