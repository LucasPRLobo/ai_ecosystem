"""
Agent specification dataclasses for Prompted Town.

This module defines the complete specification of an agent, split into:
- AgentSpec: The full agent definition (combines RL and AI aspects)
- RLProfile: Numeric traits and reward weights for learning
- AIPersonality: Text prompts and style info for conversation

Design principle: These are COMPILED from natural language prompts.
At runtime, they are structured data, not free-form text.

The separation allows:
- RL system to work with pure numbers
- AI system to have rich personality context
- Clear serialization boundaries
"""

from dataclasses import dataclass, field
from typing import Optional

from .types import (
    AgentRole,
    Faction,
    GoalType,
    TraitName,
    ConversationIntent,
    Location,
)


# =============================================================================
# RL-FACING SPECIFICATIONS
# =============================================================================

@dataclass
class Goal:
    """
    A single goal with its priority weight.

    The weight determines how much this goal contributes to reward shaping.
    Weights are relative within an agent's goal set.
    """
    goal_type: GoalType
    weight: float = 1.0  # Relative importance (0.0 to 1.0+)
    description: str = ""  # Human-readable explanation

    def __post_init__(self):
        if self.weight < 0:
            raise ValueError(f"Goal weight must be non-negative, got {self.weight}")


@dataclass
class RLProfile:
    """
    The RL-facing profile of an agent.

    Contains all numeric data needed for:
    - Observation encoding (traits influence what agent notices)
    - Reward computation (weights determine what outcomes matter)
    - Action validity (some actions may be role-restricted)

    All values are normalized to [0.0, 1.0] unless otherwise noted.
    """
    # Core personality traits (influence behavior tendencies)
    traits: dict[TraitName, float] = field(default_factory=dict)

    # Goals with relative weights (influence reward shaping)
    goals: list[Goal] = field(default_factory=list)

    # Reward component weights (how much each outcome type matters)
    # Keys are strings to allow custom reward components
    reward_weights: dict[str, float] = field(default_factory=lambda: {
        "survival": 1.0,       # Staying alive / maintaining energy
        "wealth": 0.5,         # Accumulating gold
        "quota": 0.8,          # Meeting obligations
        "social": 0.3,         # Building relationships
        "faction": 0.5,        # Advancing faction goals
        "suspicion": -0.7,     # Penalty for being suspected
    })

    # Action masking - which actions this agent can take
    # If None, all actions are available
    forbidden_actions: Optional[set[str]] = None

    def get_trait(self, trait: TraitName, default: float = 0.5) -> float:
        """Get trait value with default for undefined traits."""
        return self.traits.get(trait, default)

    def get_reward_weight(self, component: str, default: float = 0.0) -> float:
        """Get reward weight for a component."""
        return self.reward_weights.get(component, default)


# =============================================================================
# AI-FACING SPECIFICATIONS
# =============================================================================

@dataclass
class SpeakingStyle:
    """
    How an agent speaks in a specific context.

    Different contexts call for different communication styles.
    The AI uses these to modulate its language generation.
    """
    context: str  # e.g., "to_guards", "to_farmers", "when_lying"
    style_prompt: str  # Description of how to speak
    vocabulary_hints: list[str] = field(default_factory=list)  # Words/phrases to use
    forbidden_topics: list[str] = field(default_factory=list)  # What to avoid mentioning


@dataclass
class AIPersonality:
    """
    The AI-facing personality specification.

    This is used by the conversation engine to generate dialogue
    that is consistent with the agent's character.

    All fields are text prompts that guide the LLM.
    """
    # Core identity (who they really are, including secrets)
    core_identity: str  # "A farmer secretly organizing resistance against the lord"

    # Public persona (how they present themselves)
    public_persona: str  # "A hardworking farmer struggling to meet quota"

    # Background and history
    background: str  # "Lost family farm to taxes three years ago..."

    # Speaking styles for different contexts
    speaking_styles: dict[str, SpeakingStyle] = field(default_factory=dict)

    # What they know and believe
    beliefs: list[str] = field(default_factory=list)  # "The lord is corrupt"
    secrets: list[str] = field(default_factory=list)  # "I lead the resistance"
    knowledge: list[str] = field(default_factory=list)  # "The guard captain drinks heavily"

    # Emotional triggers and responses
    emotional_triggers: dict[str, str] = field(default_factory=dict)
    # e.g., {"mention_family": "becomes defensive and sad"}

    # Behavioral quirks and tells
    verbal_tics: list[str] = field(default_factory=list)  # "Often pauses mid-sentence"
    tells_when_lying: list[str] = field(default_factory=list)  # "Avoids eye contact"

    # Relationships attitudes (general, specific ones in WorldState)
    attitude_toward_authority: str = "neutral"
    attitude_toward_strangers: str = "cautious"

    def get_speaking_style(self, context: str) -> Optional[SpeakingStyle]:
        """Get speaking style for a context, or None if not defined."""
        return self.speaking_styles.get(context)

    def get_style_prompt_for_context(self, context: str) -> str:
        """Get the style prompt string, with fallback to default."""
        style = self.speaking_styles.get(context)
        if style:
            return style.style_prompt
        default = self.speaking_styles.get("default")
        if default:
            return default.style_prompt
        return "Speaks plainly and directly."


# =============================================================================
# COMBINED AGENT SPECIFICATION
# =============================================================================

@dataclass
class AgentSpec:
    """
    Complete specification of an agent.

    Combines identity, RL profile, and AI personality into one structure.
    This is what gets compiled from natural language prompts.

    Immutable after creation - changes to agent state go in AgentState,
    not here. This is the agent's "DNA", not their current condition.
    """
    # Identity
    agent_id: str  # Unique identifier, e.g., "farmer_01"
    name: str  # Display name, e.g., "Old Tom"
    public_role: AgentRole  # What they appear to be
    true_faction: Faction  # What they actually are (may be hidden)

    # Home location (where they rest, store things)
    home_location: Location = Location.HOME

    # RL-facing specification
    rl_profile: RLProfile = field(default_factory=RLProfile)

    # AI-facing specification
    ai_personality: AIPersonality = field(default_factory=lambda: AIPersonality(
        core_identity="An unnamed townsperson",
        public_persona="A townsperson",
        background="Lives in the town.",
    ))

    # Meta information
    is_player_controlled: bool = False  # If True, RL doesn't control this agent
    is_essential: bool = False  # If True, simulation fails if they die

    def __post_init__(self):
        # Ensure agent_id is valid
        if not self.agent_id or not self.agent_id.strip():
            raise ValueError("agent_id cannot be empty")

    def __hash__(self):
        return hash(self.agent_id)

    def __eq__(self, other):
        if not isinstance(other, AgentSpec):
            return False
        return self.agent_id == other.agent_id


# =============================================================================
# FACTORY FUNCTIONS (for common agent archetypes)
# =============================================================================

def create_farmer_spec(
    agent_id: str,
    name: str,
    faction: Faction = Faction.NEUTRAL,
    personality_override: Optional[AIPersonality] = None,
) -> AgentSpec:
    """Create a basic farmer agent specification."""
    rl_profile = RLProfile(
        traits={
            TraitName.DILIGENCE: 0.7,
            TraitName.COURAGE: 0.4,
            TraitName.LOYALTY: 0.5,
            TraitName.GREED: 0.3,
            TraitName.CAUTION: 0.6,
        },
        goals=[
            Goal(GoalType.SURVIVE, weight=1.0),
            Goal(GoalType.MEET_QUOTA, weight=0.8),
            Goal(GoalType.ACCUMULATE_WEALTH, weight=0.4),
            Goal(GoalType.PROTECT_FAMILY, weight=0.7),
        ],
        reward_weights={
            "survival": 1.0,
            "wealth": 0.4,
            "quota": 0.9,  # Farmers care a lot about quota
            "social": 0.3,
            "faction": 0.2,
            "suspicion": -0.8,
        },
    )

    default_personality = AIPersonality(
        core_identity=f"A hardworking farmer named {name}",
        public_persona="A simple farmer trying to make ends meet",
        background="Has worked the land for many years. Knows the struggle of meeting quota.",
        speaking_styles={
            "default": SpeakingStyle(
                context="default",
                style_prompt="Speaks simply, using farming metaphors. Practical and down-to-earth.",
                vocabulary_hints=["harvest", "soil", "weather", "quota"],
            ),
            "to_guards": SpeakingStyle(
                context="to_guards",
                style_prompt="Deferential and nervous. Uses 'sir' frequently. Avoids direct eye contact.",
                vocabulary_hints=["yes sir", "of course", "right away"],
            ),
            "to_farmers": SpeakingStyle(
                context="to_farmers",
                style_prompt="Relaxed, uses 'we' often. Shares complaints about weather and taxes.",
                vocabulary_hints=["we", "us folk", "the land", "another season"],
            ),
        },
        beliefs=["Hard work should be rewarded", "The quota is too high"],
        attitude_toward_authority="resigned acceptance",
        attitude_toward_strangers="cautious but polite",
    )

    return AgentSpec(
        agent_id=agent_id,
        name=name,
        public_role=AgentRole.FARMER,
        true_faction=faction,
        home_location=Location.HOME,
        rl_profile=rl_profile,
        ai_personality=personality_override or default_personality,
    )


def create_guard_spec(
    agent_id: str,
    name: str,
    personality_override: Optional[AIPersonality] = None,
) -> AgentSpec:
    """Create a basic guard agent specification."""
    rl_profile = RLProfile(
        traits={
            TraitName.COURAGE: 0.7,
            TraitName.LOYALTY: 0.8,  # Loyal to authority
            TraitName.AGGRESSION: 0.5,
            TraitName.SUSPICION: 0.7,
            TraitName.DILIGENCE: 0.6,
        },
        goals=[
            Goal(GoalType.MAINTAIN_ORDER, weight=1.0),
            Goal(GoalType.SURVIVE, weight=0.8),
            Goal(GoalType.ACCUMULATE_WEALTH, weight=0.3),
        ],
        reward_weights={
            "survival": 0.8,
            "wealth": 0.3,
            "quota": 0.2,
            "social": 0.2,
            "faction": 0.8,  # Guards care about faction (loyalist)
            "suspicion": 0.0,  # Guards don't fear suspicion
            "order": 1.0,  # Custom: maintaining order
            "arrests": 0.5,  # Custom: making arrests
        },
    )

    default_personality = AIPersonality(
        core_identity=f"A town guard named {name}, enforcer of the lord's law",
        public_persona="A dutiful guard keeping the peace",
        background="Joined the guard for steady pay. Takes the job seriously.",
        speaking_styles={
            "default": SpeakingStyle(
                context="default",
                style_prompt="Authoritative and clipped. Uses official language.",
                vocabulary_hints=["citizen", "the law", "by order of", "move along"],
            ),
            "to_farmers": SpeakingStyle(
                context="to_farmers",
                style_prompt="Condescending but not cruel. Expects compliance.",
                vocabulary_hints=["you lot", "your quota", "no trouble now"],
            ),
            "interrogating": SpeakingStyle(
                context="interrogating",
                style_prompt="Cold, methodical. Asks pointed questions. Watches for reactions.",
                vocabulary_hints=["where were you", "who did you speak with", "don't lie to me"],
            ),
        },
        beliefs=["Order must be maintained", "The lord's law is just"],
        attitude_toward_authority="loyal and respectful",
        attitude_toward_strangers="suspicious",
    )

    return AgentSpec(
        agent_id=agent_id,
        name=name,
        public_role=AgentRole.GUARD,
        true_faction=Faction.LOYALIST,
        home_location=Location.GATE,  # Guards often stationed at gate
        rl_profile=rl_profile,
        ai_personality=personality_override or default_personality,
    )


def create_rebel_spec(
    agent_id: str,
    name: str,
    cover_role: AgentRole = AgentRole.FARMER,
    personality_override: Optional[AIPersonality] = None,
) -> AgentSpec:
    """
    Create a rebel agent specification.

    Note: Rebels have a cover role (what they appear to be) and their
    true faction (REBEL). The AI personality includes both personas.
    """
    rl_profile = RLProfile(
        traits={
            TraitName.COURAGE: 0.8,
            TraitName.LOYALTY: 0.9,  # Loyal to the cause
            TraitName.CUNNING: 0.7,
            TraitName.CHARISMA: 0.7,
            TraitName.CAUTION: 0.6,
            TraitName.EMPATHY: 0.7,
        },
        goals=[
            Goal(GoalType.RECRUIT_REBELS, weight=1.0),
            Goal(GoalType.SURVIVE, weight=0.9),
            Goal(GoalType.AVOID_SUSPICION, weight=0.8),
            Goal(GoalType.BUILD_RELATIONSHIPS, weight=0.7),
        ],
        reward_weights={
            "survival": 0.9,
            "wealth": 0.2,
            "quota": 0.4,  # Must maintain cover
            "social": 0.8,  # Relationships matter for recruiting
            "faction": 1.0,  # Growing the rebellion is paramount
            "suspicion": -1.0,  # Very bad to be suspected
            "recruitment": 1.0,  # Custom: successful recruitment
        },
    )

    default_personality = AIPersonality(
        core_identity=f"A secret rebel named {name}, working to overthrow the lord while posing as a {cover_role.value}",
        public_persona=f"A {cover_role.value} like any other, nothing suspicious",
        background="Radicalized by injustice. Now works in secret to build resistance.",
        speaking_styles={
            "default": SpeakingStyle(
                context="default",
                style_prompt=f"Speaks like a typical {cover_role.value}. Careful not to stand out.",
                vocabulary_hints=["just like everyone else", "hard times"],
            ),
            "to_guards": SpeakingStyle(
                context="to_guards",
                style_prompt="Excessively normal. Deferential. Never gives them a reason to look closer.",
                vocabulary_hints=["yes sir", "of course", "just on my way to"],
            ),
            "to_potential_recruits": SpeakingStyle(
                context="to_potential_recruits",
                style_prompt="Empathetic, uses shared grievances. Tests the waters carefully before revealing anything.",
                vocabulary_hints=["we all struggle", "have you ever wondered", "what if"],
            ),
            "to_fellow_rebels": SpeakingStyle(
                context="to_fellow_rebels",
                style_prompt="Direct, uses code words. Trusting but still cautious about being overheard.",
                vocabulary_hints=["the harvest", "our friends", "when the time comes"],
            ),
        },
        beliefs=[
            "The current order is unjust",
            "Change requires sacrifice",
            "The people will rise when ready",
        ],
        secrets=[
            "Is a member of the resistance",
            "Knows the identities of other rebels",
        ],
        emotional_triggers={
            "injustice_mentioned": "becomes quietly intense, listens carefully",
            "rebellion_accused": "deflects calmly but heart races",
        },
        tells_when_lying=["speaks slightly faster", "over-explains"],
        attitude_toward_authority="hidden contempt beneath false respect",
        attitude_toward_strangers="evaluating potential",
    )

    return AgentSpec(
        agent_id=agent_id,
        name=name,
        public_role=cover_role,
        true_faction=Faction.REBEL,
        home_location=Location.HOME,
        rl_profile=rl_profile,
        ai_personality=personality_override or default_personality,
    )
