"""
Core type definitions for Prompted Town.

This module defines all the fundamental enums used throughout the simulation.
These are intentionally simple and map directly to discrete categories that
can be easily ported to other languages (C++/C# enums).

Design decisions:
- All enums use explicit string values for readable serialization
- No behavior attached to enums - they are pure data
- Grouped by domain (spatial, temporal, actions, social)
"""

from enum import Enum, auto
from typing import Final


# =============================================================================
# SPATIAL TYPES
# =============================================================================

class Location(str, Enum):
    """
    Physical locations in the town where agents can be.

    Each location has different affordances:
    - FARM: Work to produce goods, isolated (good for secret meetings)
    - MARKET: Trade goods, public (high visibility)
    - TAVERN: Social hub, semi-private (good for recruitment at night)
    - HOME: Rest and recover energy, private
    - GATE: Town entrance/exit, always guarded
    - TOWN_SQUARE: Central public area, high foot traffic
    """
    FARM = "farm"
    MARKET = "market"
    TAVERN = "tavern"
    HOME = "home"
    GATE = "gate"
    TOWN_SQUARE = "town_square"


# =============================================================================
# TEMPORAL TYPES
# =============================================================================

class TimeOfDay(str, Enum):
    """
    Discrete time periods that affect behavior and rules.

    - DAWN: Transition period, workers heading to fields
    - DAY: Peak activity, market open, guards active
    - DUSK: Transition period, workers returning
    - NIGHT: Curfew may apply, tavern busy, shadows useful
    """
    DAWN = "dawn"
    DAY = "day"
    DUSK = "dusk"
    NIGHT = "night"


# Time progression order (for cycling)
TIME_CYCLE: Final[list[TimeOfDay]] = [
    TimeOfDay.DAWN,
    TimeOfDay.DAY,
    TimeOfDay.DUSK,
    TimeOfDay.NIGHT,
]


def next_time_of_day(current: TimeOfDay) -> TimeOfDay:
    """Get the next time of day in the cycle."""
    current_idx = TIME_CYCLE.index(current)
    next_idx = (current_idx + 1) % len(TIME_CYCLE)
    return TIME_CYCLE[next_idx]


# =============================================================================
# AGENT ROLE TYPES
# =============================================================================

class AgentRole(str, Enum):
    """
    The public-facing role of an agent in society.

    Note: REBEL is a hidden role - a rebel's public role might be FARMER.
    We track both public_role and true_role separately in AgentSpec.

    - FARMER: Works the land, must meet quota
    - GUARD: Enforces laws, patrols, interrogates
    - MERCHANT: Buys/sells at market
    - INNKEEPER: Runs the tavern, hears gossip
    - LORD: Authority figure (typically NPC, not learning)
    """
    FARMER = "farmer"
    GUARD = "guard"
    MERCHANT = "merchant"
    INNKEEPER = "innkeeper"
    LORD = "lord"


class Faction(str, Enum):
    """
    Hidden allegiance that may differ from public role.

    - LOYALIST: Supports the current order
    - REBEL: Secretly working to undermine authority
    - NEUTRAL: No strong allegiance, self-interested
    """
    LOYALIST = "loyalist"
    REBEL = "rebel"
    NEUTRAL = "neutral"


# =============================================================================
# ACTION TYPES
# =============================================================================

class ActionType(str, Enum):
    """
    High-level discrete actions an agent can take.

    Design principle: Actions are intentions, not animations.
    The simulation resolves what actually happens.

    - MOVE: Travel to a different location
    - WORK: Perform role-appropriate labor (context-dependent)
    - REST: Recover energy (usually at HOME)
    - DELIVER_QUOTA: Pay required taxes/tribute
    - INITIATE_CONVERSATION: Start talking to another agent
    - WAIT: Do nothing, observe
    """
    MOVE = "move"
    WORK = "work"
    REST = "rest"
    DELIVER_QUOTA = "deliver_quota"
    INITIATE_CONVERSATION = "initiate_conversation"
    WAIT = "wait"


class ConversationIntent(str, Enum):
    """
    The hidden goal behind initiating a conversation.

    This is what the initiator wants to achieve - the AI personality
    will pursue this goal through natural dialogue.

    - CASUAL: Build familiarity, no specific goal
    - RECRUIT: Attempt to recruit for rebellion
    - INTERROGATE: Extract information (guards)
    - TRADE: Negotiate a transaction
    - CONSPIRE: Plan actions with known allies
    - INTIMIDATE: Threaten or pressure
    - GATHER_INFO: Subtly extract information
    """
    CASUAL = "casual"
    RECRUIT = "recruit"
    INTERROGATE = "interrogate"
    TRADE = "trade"
    CONSPIRE = "conspire"
    INTIMIDATE = "intimidate"
    GATHER_INFO = "gather_info"


# =============================================================================
# SOCIAL/RELATIONSHIP TYPES
# =============================================================================

class RelationshipLevel(str, Enum):
    """
    Qualitative description of relationship strength.

    Maps to numeric trust values:
    - HOSTILE: trust < -0.6
    - SUSPICIOUS: -0.6 <= trust < -0.2
    - STRANGER: -0.2 <= trust < 0.2
    - ACQUAINTANCE: 0.2 <= trust < 0.5
    - FRIENDLY: 0.5 <= trust < 0.8
    - TRUSTED: trust >= 0.8
    """
    HOSTILE = "hostile"
    SUSPICIOUS = "suspicious"
    STRANGER = "stranger"
    ACQUAINTANCE = "acquaintance"
    FRIENDLY = "friendly"
    TRUSTED = "trusted"


def trust_to_relationship_level(trust: float) -> RelationshipLevel:
    """Convert numeric trust value to qualitative level."""
    if trust < -0.6:
        return RelationshipLevel.HOSTILE
    elif trust < -0.2:
        return RelationshipLevel.SUSPICIOUS
    elif trust < 0.2:
        return RelationshipLevel.STRANGER
    elif trust < 0.5:
        return RelationshipLevel.ACQUAINTANCE
    elif trust < 0.8:
        return RelationshipLevel.FRIENDLY
    else:
        return RelationshipLevel.TRUSTED


class EmotionalState(str, Enum):
    """
    Current emotional state of an agent, affects conversation tone.

    This is transient (changes frequently) vs. personality (stable).
    """
    CALM = "calm"
    ANXIOUS = "anxious"
    ANGRY = "angry"
    FEARFUL = "fearful"
    HOPEFUL = "hopeful"
    DESPERATE = "desperate"
    SUSPICIOUS = "suspicious"
    CONTENT = "content"


# =============================================================================
# RESOURCE/ITEM TYPES
# =============================================================================

class ResourceType(str, Enum):
    """
    Types of resources that can be produced, traded, or taxed.
    """
    GRAIN = "grain"
    GOLD = "gold"
    GOODS = "goods"  # Generic trade goods
    FOOD = "food"    # Prepared/consumable


# =============================================================================
# EVENT TYPES (for logging and triggers)
# =============================================================================

class EventType(str, Enum):
    """
    Types of events that can occur in the simulation.
    Used for logging and potential trigger systems.
    """
    # Movement
    AGENT_MOVED = "agent_moved"

    # Work & Economy
    WORK_COMPLETED = "work_completed"
    QUOTA_DELIVERED = "quota_delivered"
    QUOTA_MISSED = "quota_missed"
    TRADE_COMPLETED = "trade_completed"

    # Social
    CONVERSATION_STARTED = "conversation_started"
    CONVERSATION_ENDED = "conversation_ended"
    RECRUITMENT_ATTEMPT = "recruitment_attempt"
    RECRUITMENT_SUCCESS = "recruitment_success"
    RECRUITMENT_FAILED = "recruitment_failed"

    # Law & Order
    CURFEW_VIOLATION = "curfew_violation"
    SUSPICION_RAISED = "suspicion_raised"
    INTERROGATION = "interrogation"
    ARREST = "arrest"

    # Time
    TIME_ADVANCED = "time_advanced"
    DAY_ENDED = "day_ended"


# =============================================================================
# TRAIT KEYS (standardized trait names)
# =============================================================================

class TraitName(str, Enum):
    """
    Standardized trait names for agent personalities.

    Values are floats from 0.0 to 1.0.
    These influence both RL reward weights and AI conversation style.
    """
    # Core personality
    COURAGE = "courage"           # Willingness to take risks
    LOYALTY = "loyalty"           # Commitment to faction/allies
    GREED = "greed"               # Prioritizes personal wealth
    HONESTY = "honesty"           # Tendency to tell truth
    EMPATHY = "empathy"           # Concern for others' welfare

    # Social traits
    CHARISMA = "charisma"         # Persuasive ability
    SUSPICION = "suspicion"       # Distrust of others
    AGGRESSION = "aggression"     # Tendency toward confrontation

    # Practical traits
    DILIGENCE = "diligence"       # Work ethic
    CUNNING = "cunning"           # Strategic deception ability
    CAUTION = "caution"           # Risk aversion


# =============================================================================
# GOAL TYPES
# =============================================================================

class GoalType(str, Enum):
    """
    Types of goals an agent can pursue.

    Goals influence reward shaping - agents with different goals
    receive different rewards for the same outcomes.
    """
    SURVIVE = "survive"                    # Stay alive, maintain energy
    ACCUMULATE_WEALTH = "accumulate_wealth"  # Maximize gold
    MEET_QUOTA = "meet_quota"              # Satisfy tax obligations
    RECRUIT_REBELS = "recruit_rebels"      # Grow the rebellion
    MAINTAIN_ORDER = "maintain_order"      # Enforce laws (guards)
    PROTECT_FAMILY = "protect_family"      # Keep loved ones safe
    AVOID_SUSPICION = "avoid_suspicion"    # Stay under the radar
    BUILD_RELATIONSHIPS = "build_relationships"  # Make allies
