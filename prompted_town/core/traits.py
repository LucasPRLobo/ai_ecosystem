"""
Abstracted trait system for Prompted Town.

Defines personality traits for agents and atmosphere traits for places.
These traits influence AI behavior, conversation dynamics, and game mechanics.

Design principles:
- Traits are normalized 0.0 to 1.0 for consistency
- Each trait has clear behavioral implications
- Templates provide quick setup for common archetypes
- Traits are composable and can be mixed
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


# =============================================================================
# AGENT TRAITS
# =============================================================================

class AgentTrait(str, Enum):
    """Core personality traits for agents."""

    # How they relate to risk and action
    BOLDNESS = "boldness"           # 0=cautious, 1=reckless
    DISCRETION = "discretion"       # 0=loose-lipped, 1=secretive

    # How they relate to others
    WARMTH = "warmth"               # 0=cold/distant, 1=friendly/open
    TRUST = "trust"                 # 0=suspicious, 1=trusting

    # How they relate to authority
    COMPLIANCE = "compliance"       # 0=rebellious, 1=obedient
    AMBITION = "ambition"           # 0=content, 1=driven

    # Internal state
    RESILIENCE = "resilience"       # 0=fragile, 1=unbreakable
    GRIEVANCE = "grievance"         # 0=satisfied, 1=bitter


@dataclass
class AgentTraits:
    """Collection of personality traits for an agent."""

    boldness: float = 0.5
    discretion: float = 0.5
    warmth: float = 0.5
    trust: float = 0.5
    compliance: float = 0.5
    ambition: float = 0.5
    resilience: float = 0.5
    grievance: float = 0.5

    def get(self, trait: AgentTrait) -> float:
        """Get trait value by enum."""
        return getattr(self, trait.value, 0.5)

    def set(self, trait: AgentTrait, value: float):
        """Set trait value (clamped to 0-1)."""
        setattr(self, trait.value, max(0.0, min(1.0, value)))

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "boldness": self.boldness,
            "discretion": self.discretion,
            "warmth": self.warmth,
            "trust": self.trust,
            "compliance": self.compliance,
            "ambition": self.ambition,
            "resilience": self.resilience,
            "grievance": self.grievance,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentTraits":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})

    def describe(self) -> str:
        """Generate natural language description of personality."""
        parts = []

        # Boldness
        if self.boldness > 0.7:
            parts.append("daring and willing to take risks")
        elif self.boldness < 0.3:
            parts.append("cautious and risk-averse")

        # Discretion
        if self.discretion > 0.7:
            parts.append("keeps secrets well")
        elif self.discretion < 0.3:
            parts.append("tends to speak freely")

        # Warmth
        if self.warmth > 0.7:
            parts.append("warm and approachable")
        elif self.warmth < 0.3:
            parts.append("cold and distant")

        # Trust
        if self.trust > 0.7:
            parts.append("trusting of others")
        elif self.trust < 0.3:
            parts.append("suspicious and guarded")

        # Compliance
        if self.compliance > 0.7:
            parts.append("respects authority")
        elif self.compliance < 0.3:
            parts.append("questions authority")

        # Ambition
        if self.ambition > 0.7:
            parts.append("ambitious and driven")
        elif self.ambition < 0.3:
            parts.append("content with their lot")

        # Resilience
        if self.resilience > 0.7:
            parts.append("emotionally resilient")
        elif self.resilience < 0.3:
            parts.append("emotionally vulnerable")

        # Grievance
        if self.grievance > 0.7:
            parts.append("deeply dissatisfied with the system")
        elif self.grievance < 0.3:
            parts.append("generally satisfied")

        return ", ".join(parts) if parts else "balanced personality"


# =============================================================================
# AGENT TEMPLATES
# =============================================================================

AGENT_TEMPLATES = {
    "the_idealist": AgentTraits(
        boldness=0.8,
        discretion=0.4,
        warmth=0.7,
        trust=0.6,
        compliance=0.2,
        ambition=0.7,
        resilience=0.6,
        grievance=0.8,
    ),
    "the_survivor": AgentTraits(
        boldness=0.3,
        discretion=0.8,
        warmth=0.4,
        trust=0.3,
        compliance=0.6,
        ambition=0.4,
        resilience=0.8,
        grievance=0.5,
    ),
    "the_opportunist": AgentTraits(
        boldness=0.7,
        discretion=0.5,
        warmth=0.5,
        trust=0.3,
        compliance=0.4,
        ambition=0.9,
        resilience=0.6,
        grievance=0.4,
    ),
    "the_loyalist": AgentTraits(
        boldness=0.5,
        discretion=0.6,
        warmth=0.5,
        trust=0.4,
        compliance=0.9,
        ambition=0.3,
        resilience=0.7,
        grievance=0.2,
    ),
    "the_shepherd": AgentTraits(
        boldness=0.6,
        discretion=0.5,
        warmth=0.9,
        trust=0.7,
        compliance=0.5,
        ambition=0.5,
        resilience=0.7,
        grievance=0.5,
    ),
    "the_firebrand": AgentTraits(
        boldness=0.9,
        discretion=0.2,
        warmth=0.6,
        trust=0.5,
        compliance=0.1,
        ambition=0.8,
        resilience=0.7,
        grievance=0.9,
    ),
    "the_cynic": AgentTraits(
        boldness=0.4,
        discretion=0.7,
        warmth=0.3,
        trust=0.2,
        compliance=0.5,
        ambition=0.3,
        resilience=0.8,
        grievance=0.7,
    ),
    "the_innocent": AgentTraits(
        boldness=0.4,
        discretion=0.3,
        warmth=0.8,
        trust=0.8,
        compliance=0.7,
        ambition=0.4,
        resilience=0.4,
        grievance=0.3,
    ),
}


def get_agent_template(name: str) -> AgentTraits:
    """Get a copy of an agent template by name."""
    template = AGENT_TEMPLATES.get(name.lower().replace(" ", "_"))
    if template:
        return AgentTraits(**template.to_dict())
    return AgentTraits()


# =============================================================================
# PLACE TRAITS
# =============================================================================

class PlaceTrait(str, Enum):
    """Atmosphere traits for places."""

    # Physical characteristics
    VISIBILITY = "visibility"       # 0=hidden/private, 1=exposed/public
    COMFORT = "comfort"             # 0=harsh/unwelcoming, 1=cozy/inviting

    # Social characteristics
    FORMALITY = "formality"         # 0=casual, 1=formal/structured
    MIXING = "mixing"               # 0=segregated, 1=diverse mixing

    # Authority/Safety
    SURVEILLANCE = "surveillance"   # 0=unwatched, 1=heavily monitored
    SANCTUARY = "sanctuary"         # 0=dangerous, 1=protected/safe

    # Atmosphere
    TENSION = "tension"             # 0=relaxed, 1=tense/volatile
    ACTIVITY = "activity"           # 0=quiet/still, 1=busy/bustling


@dataclass
class PlaceTraits:
    """Collection of atmosphere traits for a place."""

    visibility: float = 0.5
    comfort: float = 0.5
    formality: float = 0.5
    mixing: float = 0.5
    surveillance: float = 0.5
    sanctuary: float = 0.5
    tension: float = 0.5
    activity: float = 0.5

    def get(self, trait: PlaceTrait) -> float:
        """Get trait value by enum."""
        return getattr(self, trait.value, 0.5)

    def set(self, trait: PlaceTrait, value: float):
        """Set trait value (clamped to 0-1)."""
        setattr(self, trait.value, max(0.0, min(1.0, value)))

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "visibility": self.visibility,
            "comfort": self.comfort,
            "formality": self.formality,
            "mixing": self.mixing,
            "surveillance": self.surveillance,
            "sanctuary": self.sanctuary,
            "tension": self.tension,
            "activity": self.activity,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PlaceTraits":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})

    def describe(self) -> str:
        """Generate natural language description of place."""
        parts = []

        if self.visibility > 0.7:
            parts.append("very public and exposed")
        elif self.visibility < 0.3:
            parts.append("private and secluded")

        if self.comfort > 0.7:
            parts.append("warm and inviting")
        elif self.comfort < 0.3:
            parts.append("cold and unwelcoming")

        if self.surveillance > 0.7:
            parts.append("heavily watched by guards")
        elif self.surveillance < 0.3:
            parts.append("rarely patrolled")

        if self.sanctuary > 0.7:
            parts.append("feels safe and protected")
        elif self.sanctuary < 0.3:
            parts.append("feels dangerous")

        if self.tension > 0.7:
            parts.append("tense atmosphere")
        elif self.tension < 0.3:
            parts.append("relaxed atmosphere")

        if self.activity > 0.7:
            parts.append("busy and bustling")
        elif self.activity < 0.3:
            parts.append("quiet and still")

        return ", ".join(parts) if parts else "unremarkable atmosphere"

    def recruitment_modifier(self) -> float:
        """Calculate how this place affects recruitment success."""
        # Low visibility, low surveillance = better for recruitment
        # High comfort, low tension = people more open
        positive = (1 - self.visibility) + (1 - self.surveillance) + self.comfort + (1 - self.tension)
        return (positive / 4) - 0.5  # Returns -0.5 to +0.5

    def suspicion_modifier(self) -> float:
        """Calculate how this place affects suspicion from activities."""
        # High visibility, high surveillance = more likely to be caught
        return (self.visibility + self.surveillance) / 2


# =============================================================================
# PLACE TEMPLATES
# =============================================================================

PLACE_TEMPLATES = {
    "safe_haven": PlaceTraits(
        visibility=0.2,
        comfort=0.8,
        formality=0.2,
        mixing=0.6,
        surveillance=0.1,
        sanctuary=0.9,
        tension=0.2,
        activity=0.4,
    ),
    "public_square": PlaceTraits(
        visibility=0.9,
        comfort=0.5,
        formality=0.5,
        mixing=0.9,
        surveillance=0.7,
        sanctuary=0.5,
        tension=0.4,
        activity=0.8,
    ),
    "authority_hub": PlaceTraits(
        visibility=0.8,
        comfort=0.4,
        formality=0.9,
        mixing=0.3,
        surveillance=0.9,
        sanctuary=0.7,
        tension=0.6,
        activity=0.6,
    ),
    "seedy_corner": PlaceTraits(
        visibility=0.3,
        comfort=0.3,
        formality=0.1,
        mixing=0.7,
        surveillance=0.2,
        sanctuary=0.2,
        tension=0.7,
        activity=0.5,
    ),
    "workers_rest": PlaceTraits(
        visibility=0.4,
        comfort=0.7,
        formality=0.2,
        mixing=0.5,
        surveillance=0.3,
        sanctuary=0.6,
        tension=0.3,
        activity=0.5,
    ),
    "open_market": PlaceTraits(
        visibility=0.8,
        comfort=0.5,
        formality=0.3,
        mixing=0.9,
        surveillance=0.5,
        sanctuary=0.4,
        tension=0.4,
        activity=0.9,
    ),
    "isolated_outpost": PlaceTraits(
        visibility=0.2,
        comfort=0.3,
        formality=0.4,
        mixing=0.2,
        surveillance=0.1,
        sanctuary=0.3,
        tension=0.5,
        activity=0.2,
    ),
    "neutral_ground": PlaceTraits(
        visibility=0.5,
        comfort=0.5,
        formality=0.5,
        mixing=0.7,
        surveillance=0.4,
        sanctuary=0.5,
        tension=0.4,
        activity=0.5,
    ),
}


def get_place_template(name: str) -> PlaceTraits:
    """Get a copy of a place template by name."""
    template = PLACE_TEMPLATES.get(name.lower().replace(" ", "_"))
    if template:
        return PlaceTraits(**template.to_dict())
    return PlaceTraits()


# =============================================================================
# DEFAULT PLACE TRAITS BY LOCATION
# =============================================================================

from .types import Location

DEFAULT_LOCATION_TRAITS = {
    Location.HOME: PlaceTraits(
        visibility=0.1, comfort=0.9, formality=0.1, mixing=0.1,
        surveillance=0.0, sanctuary=0.9, tension=0.1, activity=0.3,
    ),
    Location.FARM: PlaceTraits(
        visibility=0.3, comfort=0.4, formality=0.2, mixing=0.3,
        surveillance=0.2, sanctuary=0.4, tension=0.3, activity=0.6,
    ),
    Location.MARKET: PlaceTraits(
        visibility=0.8, comfort=0.5, formality=0.3, mixing=0.9,
        surveillance=0.5, sanctuary=0.4, tension=0.4, activity=0.9,
    ),
    Location.TAVERN: PlaceTraits(
        visibility=0.4, comfort=0.8, formality=0.2, mixing=0.7,
        surveillance=0.3, sanctuary=0.6, tension=0.3, activity=0.6,
    ),
    Location.TOWN_SQUARE: PlaceTraits(
        visibility=0.9, comfort=0.4, formality=0.6, mixing=0.8,
        surveillance=0.7, sanctuary=0.5, tension=0.5, activity=0.7,
    ),
    Location.GATE: PlaceTraits(
        visibility=0.8, comfort=0.3, formality=0.8, mixing=0.4,
        surveillance=0.9, sanctuary=0.6, tension=0.6, activity=0.5,
    ),
}


def get_default_location_traits(location: Location) -> PlaceTraits:
    """Get default traits for a location."""
    traits = DEFAULT_LOCATION_TRAITS.get(location)
    if traits:
        return PlaceTraits(**traits.to_dict())
    return PlaceTraits()


# =============================================================================
# RECRUITMENT AGGRESSIVENESS
# =============================================================================

class RecruitmentStyle(str, Enum):
    """How aggressively to pursue recruitment."""
    CAUTIOUS = "cautious"       # Build trust over many conversations
    MODERATE = "moderate"       # Balance between caution and directness
    AGGRESSIVE = "aggressive"   # Push hard, take risks
    DESPERATE = "desperate"     # All or nothing approach


@dataclass
class RecruitmentConfig:
    """Configuration for recruitment behavior."""
    style: RecruitmentStyle = RecruitmentStyle.MODERATE

    # How many positive interactions before attempting recruitment
    trust_threshold: float = 0.3

    # Minimum grievance in target to consider them recruitable
    grievance_threshold: float = 0.4

    # How much to reveal in each conversation
    revelation_rate: float = 0.5

    @classmethod
    def from_style(cls, style: RecruitmentStyle) -> "RecruitmentConfig":
        """Create config from style preset."""
        configs = {
            RecruitmentStyle.CAUTIOUS: cls(
                style=style,
                trust_threshold=0.5,
                grievance_threshold=0.6,
                revelation_rate=0.2,
            ),
            RecruitmentStyle.MODERATE: cls(
                style=style,
                trust_threshold=0.3,
                grievance_threshold=0.4,
                revelation_rate=0.5,
            ),
            RecruitmentStyle.AGGRESSIVE: cls(
                style=style,
                trust_threshold=0.1,
                grievance_threshold=0.3,
                revelation_rate=0.8,
            ),
            RecruitmentStyle.DESPERATE: cls(
                style=style,
                trust_threshold=0.0,
                grievance_threshold=0.2,
                revelation_rate=1.0,
            ),
        }
        return configs.get(style, cls())

    def get_prompt_modifier(self) -> str:
        """Get prompt modification based on style."""
        modifiers = {
            RecruitmentStyle.CAUTIOUS: (
                "Be very careful. Only hint at dissatisfaction, never reveal your true purpose. "
                "If they seem unreceptive, back off immediately. Safety first."
            ),
            RecruitmentStyle.MODERATE: (
                "Gauge their feelings carefully. Share your grievances and see if they reciprocate. "
                "If they seem sympathetic, you may hint at 'others who feel the same way'. "
                "Be ready to back off if they seem uncomfortable."
            ),
            RecruitmentStyle.AGGRESSIVE: (
                "Push the conversation toward recruitment. Share your grievances openly. "
                "If they show any sympathy, make it clear you're part of something bigger. "
                "Ask them directly if they want to be part of the change."
            ),
            RecruitmentStyle.DESPERATE: (
                "Time is running out. You need allies now. Be direct about the resistance. "
                "Appeal to their suffering under the current system. "
                "Make an explicit offer to join - the risk of exposure is worth it."
            ),
        }
        return modifiers.get(self.style, "")
