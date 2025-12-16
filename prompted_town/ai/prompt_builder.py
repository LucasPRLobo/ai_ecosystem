"""
Prompt builder for Prompted Town conversations.

This module constructs prompts that guide AI-powered agent conversations.
It combines:
- Agent personality specifications
- Current world context (location, time, relationships)
- Conversation intent
- Speaking style selection

Design principles:
- Prompts are deterministic given the same inputs
- Context-aware style selection
- Clear separation of identity vs. situation
"""

from dataclasses import dataclass, field
from typing import Optional

from ..core.types import (
    Location,
    TimeOfDay,
    ConversationIntent,
    AgentRole,
    Faction,
    RelationshipLevel,
    EmotionalState,
)
from ..core.agent_spec import AgentSpec, AIPersonality
from ..core.world_state import WorldState, RelationshipState


# =============================================================================
# CONVERSATION CONTEXT
# =============================================================================

@dataclass
class ConversationContext:
    """
    Full context for a conversation.

    Contains everything needed to build appropriate prompts.
    """
    # Participants
    speaker_id: str
    listener_id: str

    # Specifications (for personality info)
    speaker_spec: AgentSpec
    listener_spec: AgentSpec

    # World state context
    location: Location
    time_of_day: TimeOfDay
    nearby_agents: list[str] = field(default_factory=list)

    # Relationship context
    trust_speaker_to_listener: float = 0.0
    trust_listener_to_speaker: float = 0.0
    relationship_level: RelationshipLevel = RelationshipLevel.STRANGER
    previous_interactions: int = 0

    # Conversation parameters
    intent: ConversationIntent = ConversationIntent.CASUAL
    is_initiator: bool = True  # Is speaker the one who started the conversation?

    # Knowledge state
    speaker_knows_listener_is_rebel: bool = False
    listener_knows_speaker_is_rebel: bool = False

    # Current emotional states
    speaker_emotional_state: EmotionalState = EmotionalState.CALM
    listener_emotional_state: EmotionalState = EmotionalState.CALM


# =============================================================================
# SPEAKING STYLE SELECTION
# =============================================================================

def select_speaking_style(
    speaker_spec: AgentSpec,
    listener_spec: AgentSpec,
    context: ConversationContext,
) -> str:
    """
    Select the appropriate speaking style based on context.

    Returns the style prompt string.
    """
    personality = speaker_spec.ai_personality

    # Priority order for style selection:
    # 1. Intent-specific styles (if recruiting, use recruitment style)
    # 2. Listener-role-specific styles (if talking to guard, use guard style)
    # 3. Relationship-specific styles (if talking to known rebel, use rebel style)
    # 4. Default style

    style_key = None

    # Check intent-specific styles
    if context.intent == ConversationIntent.RECRUIT:
        style_key = "to_potential_recruits"
    elif context.intent == ConversationIntent.INTERROGATE:
        style_key = "interrogating"
    elif context.intent == ConversationIntent.CONSPIRE:
        style_key = "to_fellow_rebels"

    # Check listener role if no intent-specific style
    if not style_key or style_key not in personality.speaking_styles:
        listener_role = listener_spec.public_role
        role_style_map = {
            AgentRole.GUARD: "to_guards",
            AgentRole.FARMER: "to_farmers",
            AgentRole.MERCHANT: "to_merchants",
            AgentRole.INNKEEPER: "to_innkeeper",
        }
        style_key = role_style_map.get(listener_role)

    # Check if we know they're a rebel
    if context.speaker_knows_listener_is_rebel and style_key not in personality.speaking_styles:
        style_key = "to_fellow_rebels"

    # Use the selected style or fall back to default
    return personality.get_style_prompt_for_context(style_key or "default")


# =============================================================================
# SYSTEM PROMPT (IDENTITY)
# =============================================================================

def build_system_prompt(
    agent_spec: AgentSpec,
    context: ConversationContext,
    is_speaker: bool = True,
) -> str:
    """
    Build the system prompt that defines the agent's identity.

    This establishes WHO the agent is and HOW they should behave.
    """
    personality = agent_spec.ai_personality

    # Select appropriate speaking style
    if is_speaker:
        style = select_speaking_style(agent_spec, context.listener_spec, context)
    else:
        # Create a reverse context for listener
        reverse_context = ConversationContext(
            speaker_id=context.listener_id,
            listener_id=context.speaker_id,
            speaker_spec=context.listener_spec,
            listener_spec=context.speaker_spec,
            location=context.location,
            time_of_day=context.time_of_day,
            nearby_agents=context.nearby_agents,
            trust_speaker_to_listener=context.trust_listener_to_speaker,
            trust_listener_to_speaker=context.trust_speaker_to_listener,
            relationship_level=context.relationship_level,
            previous_interactions=context.previous_interactions,
            intent=ConversationIntent.CASUAL,  # Listener doesn't know intent
            is_initiator=False,
        )
        style = select_speaking_style(agent_spec, context.speaker_spec, reverse_context)

    # Build the system prompt
    parts = []

    # Core identity
    parts.append(f"You are {agent_spec.name}, {personality.core_identity}.")

    # Public persona (how you present yourself)
    parts.append(f"\nPublic persona: {personality.public_persona}")

    # Background
    if personality.background:
        parts.append(f"\nBackground: {personality.background}")

    # Speaking style
    parts.append(f"\nSpeaking style: {style}")

    # Beliefs that might come up
    if personality.beliefs:
        parts.append(f"\nYour beliefs: {'; '.join(personality.beliefs)}")

    # Secrets to protect (only if relevant)
    if personality.secrets and agent_spec.true_faction == Faction.REBEL:
        parts.append(f"\n⚠️ SECRETS (never reveal directly): {'; '.join(personality.secrets)}")

    # Current emotional state
    other_id = context.listener_id if is_speaker else context.speaker_id
    emotional_state = context.speaker_emotional_state if is_speaker else context.listener_emotional_state
    if emotional_state != EmotionalState.CALM:
        parts.append(f"\nCurrent mood: {emotional_state.value}")

    # Verbal tics and tells
    if personality.verbal_tics:
        parts.append(f"\nSpeech patterns: {'; '.join(personality.verbal_tics)}")

    # Conversation guidelines
    parts.append("\n\n--- CONVERSATION GUIDELINES ---")
    parts.append("- Stay in character at all times")
    parts.append("- Respond naturally as this character would")
    parts.append("- Keep responses brief (1-3 sentences typically)")
    parts.append("- Show personality through word choice and tone")
    parts.append("- React to what the other person says")

    return "\n".join(parts)


# =============================================================================
# SITUATION PROMPT (CONTEXT)
# =============================================================================

def build_situation_prompt(context: ConversationContext) -> str:
    """
    Build a prompt describing the current situation.

    This establishes WHERE, WHEN, and WHO is present.
    """
    parts = []

    # Location and time
    parts.append(f"Location: {_describe_location(context.location)}")
    parts.append(f"Time: {_describe_time(context.time_of_day)}")

    # Who is present
    if context.nearby_agents:
        parts.append(f"Also present: {', '.join(context.nearby_agents)}")
    else:
        parts.append("You are alone together.")

    # Relationship context
    parts.append(f"\nRelationship with {context.listener_spec.name}: {context.relationship_level.value}")
    if context.previous_interactions > 0:
        parts.append(f"You've spoken {context.previous_interactions} times before.")
    else:
        parts.append("This is your first real conversation.")

    # Trust levels (phrased naturally)
    if context.trust_speaker_to_listener > 0.5:
        parts.append(f"You feel you can trust {context.listener_spec.name}.")
    elif context.trust_speaker_to_listener < -0.3:
        parts.append(f"You are wary of {context.listener_spec.name}.")

    return "\n".join(parts)


def _describe_location(location: Location) -> str:
    """Get a narrative description of a location."""
    descriptions = {
        Location.FARM: "The fields outside town. Quiet, isolated, the smell of earth and sweat.",
        Location.MARKET: "The busy town market. Vendors calling, people bustling, eyes everywhere.",
        Location.TAVERN: "The local tavern. Dim light, murmured conversations, relative privacy.",
        Location.HOME: "A modest home. Private, comfortable, away from prying eyes.",
        Location.GATE: "The town gate. Guards nearby, travelers passing through.",
        Location.TOWN_SQUARE: "The town square. The heart of public life, always watched.",
    }
    return descriptions.get(location, f"The {location.value}")


def _describe_time(time: TimeOfDay) -> str:
    """Get a narrative description of time."""
    descriptions = {
        TimeOfDay.DAWN: "Early morning. The town is waking up, few people about.",
        TimeOfDay.DAY: "Midday. The town is busy with daily activity.",
        TimeOfDay.DUSK: "Evening. Workers returning home, the day winding down.",
        TimeOfDay.NIGHT: "Night. Dark streets, most honest folk at home. Curfew is in effect.",
    }
    return descriptions.get(time, f"{time.value}")


# =============================================================================
# INTENT-SPECIFIC PROMPTS
# =============================================================================

def build_intent_prompt(
    intent: ConversationIntent,
    context: ConversationContext,
) -> str:
    """
    Build a prompt guiding the agent toward their conversation goal.

    This is added to the speaker's prompt to guide their objective.
    """
    if intent == ConversationIntent.CASUAL:
        return "Your goal: Have a friendly conversation. Build rapport naturally."

    elif intent == ConversationIntent.RECRUIT:
        # Recruitment is delicate
        return (
            "Your goal: Gauge if this person might be sympathetic to... alternative viewpoints. "
            "Do NOT reveal your true allegiance unless you're very confident they're receptive. "
            "Use shared grievances to test the waters. Be subtle. Be patient. "
            "If they seem unreceptive or suspicious, back off gracefully."
        )

    elif intent == ConversationIntent.INTERROGATE:
        return (
            "Your goal: Extract information. Ask probing questions. "
            "Watch for signs of deception. Press if they seem evasive. "
            "You have authority here—use it."
        )

    elif intent == ConversationIntent.TRADE:
        return "Your goal: Negotiate a fair trade. Know your worth."

    elif intent == ConversationIntent.CONSPIRE:
        return (
            "Your goal: Plan together with your ally. Speak in code if others might overhear. "
            "Discuss next steps, share information, coordinate actions."
        )

    elif intent == ConversationIntent.INTIMIDATE:
        return (
            "Your goal: Make them afraid. Assert dominance. "
            "Make clear there are consequences for defiance."
        )

    elif intent == ConversationIntent.GATHER_INFO:
        return (
            "Your goal: Learn what they know without revealing why you want to know. "
            "Ask innocent-seeming questions. Listen more than you speak."
        )

    return ""


# =============================================================================
# TURN PROMPT (for continuing conversations)
# =============================================================================

def build_turn_prompt(
    context: ConversationContext,
    conversation_so_far: list[dict],
    is_speaker_turn: bool,
) -> str:
    """
    Build a prompt for the next turn in an ongoing conversation.

    Args:
        context: The conversation context
        conversation_so_far: List of {"speaker": name, "text": content} dicts
        is_speaker_turn: True if it's the initiator's turn

    Returns:
        User prompt for the next turn
    """
    if is_speaker_turn:
        agent_name = context.speaker_spec.name
        other_name = context.listener_spec.name
    else:
        agent_name = context.listener_spec.name
        other_name = context.speaker_spec.name

    parts = []

    # Show what was just said
    if conversation_so_far:
        last_turn = conversation_so_far[-1]
        parts.append(f'{last_turn["speaker"]} said: "{last_turn["text"]}"')

    parts.append(f"\nAs {agent_name}, respond naturally. Keep it brief and in character.")

    return "\n".join(parts)


# =============================================================================
# COMPLETE PROMPT BUILDING
# =============================================================================

def build_conversation_prompts(
    context: ConversationContext,
    turn_history: Optional[list[dict]] = None,
    for_speaker: bool = True,
) -> tuple[str, str]:
    """
    Build complete prompts for a conversation turn.

    Returns:
        (system_prompt, user_prompt) tuple
    """
    if for_speaker:
        agent_spec = context.speaker_spec
    else:
        agent_spec = context.listener_spec

    # Build system prompt (identity)
    system_prompt = build_system_prompt(agent_spec, context, is_speaker=for_speaker)

    # Build user prompt (situation + turn)
    user_parts = []

    # Situation
    user_parts.append(build_situation_prompt(context))

    # Intent guidance (only for initiator)
    if for_speaker and context.is_initiator:
        intent_prompt = build_intent_prompt(context.intent, context)
        if intent_prompt:
            user_parts.append(f"\n{intent_prompt}")

    # Turn prompt if conversation has started
    if turn_history:
        user_parts.append(f"\n{build_turn_prompt(context, turn_history, for_speaker)}")
    else:
        # First turn - opener
        if for_speaker:
            user_parts.append(f"\nYou approach {context.listener_spec.name}. What do you say to start the conversation?")
        else:
            user_parts.append(f"\n{context.speaker_spec.name} has approached you. Wait for them to speak.")

    user_prompt = "\n".join(user_parts)

    return system_prompt, user_prompt


# =============================================================================
# CONTEXT FROM WORLD STATE
# =============================================================================

def create_context_from_world(
    world: WorldState,
    speaker_spec: AgentSpec,
    listener_spec: AgentSpec,
    intent: ConversationIntent = ConversationIntent.CASUAL,
) -> ConversationContext:
    """
    Create a ConversationContext from world state.

    Convenience function to extract all relevant context.
    """
    speaker_state = world.get_agent(speaker_spec.agent_id)
    listener_state = world.get_agent(listener_spec.agent_id)

    if not speaker_state or not listener_state:
        raise ValueError("Both agents must exist in world state")

    # Get relationship
    rel = world.get_relationship(speaker_spec.agent_id, listener_spec.agent_id)

    # Get nearby agents (excluding participants)
    nearby = [
        aid for aid in world.get_nearby_agents(speaker_spec.agent_id)
        if aid != listener_spec.agent_id
    ]

    return ConversationContext(
        speaker_id=speaker_spec.agent_id,
        listener_id=listener_spec.agent_id,
        speaker_spec=speaker_spec,
        listener_spec=listener_spec,
        location=speaker_state.location,
        time_of_day=world.time_of_day,
        nearby_agents=nearby,
        trust_speaker_to_listener=rel.get_trust(speaker_spec.agent_id, listener_spec.agent_id),
        trust_listener_to_speaker=rel.get_trust(listener_spec.agent_id, speaker_spec.agent_id),
        relationship_level=rel.get_relationship_level(speaker_spec.agent_id, listener_spec.agent_id),
        previous_interactions=rel.total_interactions,
        intent=intent,
        is_initiator=True,
        speaker_knows_listener_is_rebel=rel.a_knows_b_is_rebel if speaker_spec.agent_id == rel.agent_a_id else rel.b_knows_a_is_rebel,
        listener_knows_speaker_is_rebel=rel.b_knows_a_is_rebel if speaker_spec.agent_id == rel.agent_a_id else rel.a_knows_b_is_rebel,
        speaker_emotional_state=speaker_state.emotional_state,
        listener_emotional_state=listener_state.emotional_state,
    )
