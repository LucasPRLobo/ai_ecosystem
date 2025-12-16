"""
Conversation engine for Prompted Town.

This module orchestrates multi-turn AI conversations between agents.
It coordinates:
- Turn-taking between participants
- Prompt building for each turn
- Response generation via LLM backend
- Conversation flow control

Design principles:
- Stateless engine (all state in ConversationState)
- Pluggable LLM backend
- Natural conversation flow with termination detection
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from ..core.types import ConversationIntent, Location, TimeOfDay
from ..core.agent_spec import AgentSpec

from .llm_backend import LLMBackend, Message, LLMResponse
from .prompt_builder import (
    ConversationContext,
    build_conversation_prompts,
    build_system_prompt,
    build_turn_prompt,
)


# =============================================================================
# CONVERSATION STATE
# =============================================================================

class ConversationPhase(str, Enum):
    """Phases of a conversation."""
    NOT_STARTED = "not_started"
    OPENING = "opening"
    MIDDLE = "middle"
    CLOSING = "closing"
    ENDED = "ended"


@dataclass
class Turn:
    """A single turn in a conversation."""
    speaker_id: str
    speaker_name: str
    text: str
    turn_number: int
    raw_response: Optional[LLMResponse] = None


@dataclass
class ConversationState:
    """
    Complete state of an ongoing conversation.

    Tracks everything about the conversation as it progresses.
    """
    context: ConversationContext
    turns: list[Turn] = field(default_factory=list)
    phase: ConversationPhase = ConversationPhase.NOT_STARTED

    # Whose turn is it? True = initiator (speaker), False = responder (listener)
    is_initiator_turn: bool = True

    # Termination tracking
    ended_naturally: bool = False
    end_reason: str = ""

    # Token tracking (for cost estimation)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def current_speaker_spec(self) -> AgentSpec:
        if self.is_initiator_turn:
            return self.context.speaker_spec
        return self.context.listener_spec

    @property
    def current_listener_spec(self) -> AgentSpec:
        if self.is_initiator_turn:
            return self.context.listener_spec
        return self.context.speaker_spec

    def get_transcript_for_prompt(self) -> list[dict]:
        """Get transcript in format suitable for prompt building."""
        return [
            {"speaker": turn.speaker_name, "text": turn.text}
            for turn in self.turns
        ]

    def get_full_transcript(self) -> str:
        """Get human-readable transcript."""
        lines = []
        for turn in self.turns:
            lines.append(f"[{turn.speaker_name}]: {turn.text}")
        return "\n".join(lines)


# =============================================================================
# TERMINATION DETECTION
# =============================================================================

# Keywords/phrases that suggest conversation should end
ENDING_PHRASES = [
    "goodbye", "farewell", "take care", "see you", "must go",
    "have to go", "need to go", "leaving now", "walk away",
    "turns away", "walks away", "ends the conversation",
    "good day", "good night", "until next time",
]

HOSTILE_PHRASES = [
    "get away", "leave me alone", "guards!", "help!",
    "don't trust", "stay away", "reporting you",
]


def should_end_conversation(
    state: ConversationState,
    latest_response: str,
    max_turns: int = 8,
) -> tuple[bool, str]:
    """
    Determine if the conversation should end.

    Returns (should_end, reason).
    """
    # Max turns reached
    if state.turn_count >= max_turns:
        return True, "max_turns"

    response_lower = latest_response.lower()

    # Check for ending phrases
    for phrase in ENDING_PHRASES:
        if phrase in response_lower:
            return True, "natural_ending"

    # Check for hostile ending
    for phrase in HOSTILE_PHRASES:
        if phrase in response_lower:
            return True, "hostile_ending"

    # Check for very short responses (might indicate disengagement)
    if len(latest_response.split()) < 3 and state.turn_count > 2:
        # Short response after a few turns might mean they want to end
        if any(word in response_lower for word in ["fine", "okay", "sure", "whatever"]):
            return True, "disengaged"

    return False, ""


# =============================================================================
# CONVERSATION ENGINE
# =============================================================================

class ConversationEngine:
    """
    Engine for running AI-powered conversations.

    Orchestrates multi-turn dialogues between agent personalities.
    """

    def __init__(
        self,
        backend: LLMBackend,
        temperature: float = 0.8,
        max_tokens_per_turn: int = 150,
    ):
        self.backend = backend
        self.temperature = temperature
        self.max_tokens_per_turn = max_tokens_per_turn

    def start_conversation(
        self,
        context: ConversationContext,
    ) -> ConversationState:
        """
        Start a new conversation.

        Returns initial state (no turns yet).
        """
        return ConversationState(
            context=context,
            phase=ConversationPhase.NOT_STARTED,
            is_initiator_turn=True,
        )

    def run_turn(self, state: ConversationState) -> ConversationState:
        """
        Run a single turn of the conversation.

        Generates a response for the current speaker and updates state.
        """
        if state.phase == ConversationPhase.ENDED:
            return state

        # Update phase
        if state.phase == ConversationPhase.NOT_STARTED:
            state.phase = ConversationPhase.OPENING
        elif state.turn_count >= 2:
            state.phase = ConversationPhase.MIDDLE

        # Build prompts
        system_prompt, user_prompt = build_conversation_prompts(
            context=state.context,
            turn_history=state.get_transcript_for_prompt(),
            for_speaker=state.is_initiator_turn,
        )

        # Build message history for context
        messages = self._build_messages(state, system_prompt, user_prompt)

        # Generate response
        response = self.backend.generate_with_history(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens_per_turn,
        )

        # Clean up response
        response_text = self._clean_response(response.content, state)

        # Create turn
        speaker_spec = state.current_speaker_spec
        turn = Turn(
            speaker_id=speaker_spec.agent_id,
            speaker_name=speaker_spec.name,
            text=response_text,
            turn_number=state.turn_count,
            raw_response=response,
        )

        # Update state
        state.turns.append(turn)

        # Track tokens
        if response.usage:
            state.total_prompt_tokens += response.usage.get("prompt_tokens", 0)
            state.total_completion_tokens += response.usage.get("completion_tokens", 0)

        # Check if conversation should end
        should_end, reason = should_end_conversation(state, response_text)
        if should_end:
            state.phase = ConversationPhase.ENDED
            state.ended_naturally = True
            state.end_reason = reason
        else:
            # Switch turns
            state.is_initiator_turn = not state.is_initiator_turn

        return state

    def run_conversation(
        self,
        context: ConversationContext,
        max_turns: int = 8,
    ) -> ConversationState:
        """
        Run a complete conversation.

        Continues until natural end or max turns reached.
        """
        state = self.start_conversation(context)

        while state.phase != ConversationPhase.ENDED and state.turn_count < max_turns:
            state = self.run_turn(state)

        # Ensure ended state
        if state.phase != ConversationPhase.ENDED:
            state.phase = ConversationPhase.ENDED
            state.end_reason = "max_turns"

        return state

    def _build_messages(
        self,
        state: ConversationState,
        system_prompt: str,
        user_prompt: str,
    ) -> list[Message]:
        """Build message list for LLM."""
        messages = [Message(role="system", content=system_prompt)]

        # Add conversation history as alternating messages
        for turn in state.turns:
            # Determine if this was the current speaker's turn or the other person's
            is_current_speakers_turn = (turn.speaker_id == state.current_speaker_spec.agent_id)

            if is_current_speakers_turn:
                # This was something "I" said - so it's an assistant message
                messages.append(Message(
                    role="assistant",
                    content=turn.text,
                ))
            else:
                # This was something the other person said - so it's a user message
                messages.append(Message(
                    role="user",
                    content=f"[{turn.speaker_name}]: {turn.text}",
                ))

        # Add the current prompt
        messages.append(Message(role="user", content=user_prompt))

        return messages

    def _clean_response(self, response: str, state: ConversationState) -> str:
        """Clean up LLM response."""
        text = response.strip()

        # Remove quotes if the response is wrapped in them
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]

        # Remove character name prefix if present
        speaker_name = state.current_speaker_spec.name
        prefixes_to_remove = [
            f"{speaker_name}:",
            f"[{speaker_name}]:",
            f"{speaker_name} says:",
            f"As {speaker_name},",
        ]
        for prefix in prefixes_to_remove:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()

        # Remove action markers sometimes generated
        if text.startswith("*") and "*" in text[1:]:
            # Keep the action, it's flavor
            pass

        return text.strip()


# =============================================================================
# CONVERSATION RUNNER (high-level interface)
# =============================================================================

def run_conversation(
    backend: LLMBackend,
    initiator_spec: AgentSpec,
    target_spec: AgentSpec,
    location: Location,
    time_of_day: TimeOfDay,
    intent: ConversationIntent = ConversationIntent.CASUAL,
    nearby_agents: Optional[list[str]] = None,
    trust_initiator_to_target: float = 0.0,
    trust_target_to_initiator: float = 0.0,
    max_turns: int = 6,
) -> ConversationState:
    """
    High-level function to run a conversation.

    Simplified interface for common use cases.
    """
    from ..core.types import RelationshipLevel, trust_to_relationship_level

    context = ConversationContext(
        speaker_id=initiator_spec.agent_id,
        listener_id=target_spec.agent_id,
        speaker_spec=initiator_spec,
        listener_spec=target_spec,
        location=location,
        time_of_day=time_of_day,
        nearby_agents=nearby_agents or [],
        trust_speaker_to_listener=trust_initiator_to_target,
        trust_listener_to_speaker=trust_target_to_initiator,
        relationship_level=trust_to_relationship_level(trust_initiator_to_target),
        previous_interactions=0,
        intent=intent,
        is_initiator=True,
    )

    engine = ConversationEngine(backend)
    return engine.run_conversation(context, max_turns=max_turns)


# =============================================================================
# UTILITIES
# =============================================================================

def print_conversation(state: ConversationState) -> None:
    """Print a conversation transcript in a readable format."""
    print(f"\n{'='*60}")
    print(f"CONVERSATION: {state.context.speaker_spec.name} → {state.context.listener_spec.name}")
    print(f"Location: {state.context.location.value}, Time: {state.context.time_of_day.value}")
    print(f"Intent: {state.context.intent.value}")
    print(f"{'='*60}\n")

    for turn in state.turns:
        print(f"[{turn.speaker_name}]: {turn.text}\n")

    print(f"{'='*60}")
    print(f"Turns: {state.turn_count}, Ended: {state.end_reason}")
    print(f"Tokens: {state.total_prompt_tokens} prompt, {state.total_completion_tokens} completion")
    print(f"{'='*60}\n")
