"""
Outcome parser for Prompted Town conversations.

This module extracts structured outcomes from conversation transcripts.
Supports two modes:
- Rule-based: Fast, deterministic, good for testing
- LLM-based: More accurate, slower, better for production

Design principles:
- Produce simulation-compatible ConversationOutcome objects
- Support both parsing modes
- Be conservative (don't over-interpret)
"""

from dataclasses import dataclass
from typing import Optional
import re

from ..core.types import (
    ConversationIntent,
    EmotionalState,
)
from ..simulation.social import ConversationOutcome, DialogueTurn

from .llm_backend import LLMBackend, Message
from .conversation import ConversationState


# =============================================================================
# KEYWORD LISTS FOR RULE-BASED PARSING
# =============================================================================

# Positive sentiment indicators
POSITIVE_KEYWORDS = [
    "friend", "agree", "yes", "right", "understand", "together",
    "thank", "appreciate", "help", "trust", "hope", "smile",
]

# Negative sentiment indicators
NEGATIVE_KEYWORDS = [
    "no", "never", "wrong", "leave", "away", "suspicious",
    "guard", "report", "angry", "hate", "distrust", "careful",
]

# Recruitment-related keywords
RECRUITMENT_POSITIVE = [
    "join", "with us", "together", "change", "fight", "resist",
    "rebellion", "freedom", "enough", "stand up", "counting on",
]

RECRUITMENT_NEGATIVE = [
    "loyal", "dangerous talk", "treason", "report", "guards",
    "no part", "leave me", "won't", "refuse", "madness",
]

# Suspicious behavior indicators
SUSPICIOUS_KEYWORDS = [
    "rebellion", "overthrow", "resist", "fight back", "uprising",
    "secret", "don't tell", "between us", "lord", "corrupt",
]

# Information revelation indicators
INFO_REVEALED_PATTERNS = [
    r"i('m| am) (a |the )?rebel",
    r"we('re| are) planning",
    r"there('s| is) a group",
    r"meet (at|in) the",
    r"the signal (is|will be)",
]


# =============================================================================
# RULE-BASED PARSER
# =============================================================================

class RuleBasedOutcomeParser:
    """
    Parse conversation outcomes using keyword heuristics.

    Fast and deterministic, good for testing and quick estimation.
    """

    def parse(
        self,
        state: ConversationState,
    ) -> ConversationOutcome:
        """
        Parse a completed conversation into structured outcome.
        """
        context = state.context
        transcript = [
            DialogueTurn(speaker_id=t.speaker_id, text=t.text)
            for t in state.turns
        ]

        # Analyze overall sentiment
        positive_score, negative_score = self._analyze_sentiment(state)
        was_positive = positive_score > negative_score

        # Calculate trust deltas
        initiator_trust_delta, target_trust_delta = self._calculate_trust_deltas(
            state, positive_score, negative_score
        )

        # Check for recruitment attempt/success
        recruitment_attempted = context.intent == ConversationIntent.RECRUIT
        recruitment_successful = False
        if recruitment_attempted:
            recruitment_successful = self._check_recruitment_success(state)

        # Check for interrogation
        interrogation_attempted = context.intent == ConversationIntent.INTERROGATE
        information_revealed = []
        if interrogation_attempted:
            information_revealed = self._extract_revealed_info(state)

        # Check for raised suspicion
        suspicion_raised, suspicion_amount = self._check_suspicion(state)

        # Determine emotional outcomes
        initiator_emotion = self._infer_emotional_state(state, is_initiator=True)
        target_emotion = self._infer_emotional_state(state, is_initiator=False)

        return ConversationOutcome(
            initiator_id=context.speaker_id,
            target_id=context.listener_id,
            transcript=transcript,
            turn_count=state.turn_count,
            initiator_trust_delta=initiator_trust_delta,
            target_trust_delta=target_trust_delta,
            recruitment_attempted=recruitment_attempted,
            recruitment_successful=recruitment_successful,
            interrogation_attempted=interrogation_attempted,
            information_revealed=information_revealed,
            suspicion_raised=suspicion_raised,
            suspicion_amount=suspicion_amount,
            was_positive=was_positive,
            reason_ended=state.end_reason,
            initiator_emotional_state=initiator_emotion,
            target_emotional_state=target_emotion,
        )

    def _analyze_sentiment(self, state: ConversationState) -> tuple[float, float]:
        """Analyze overall conversation sentiment."""
        positive = 0
        negative = 0

        all_text = " ".join(t.text.lower() for t in state.turns)

        for keyword in POSITIVE_KEYWORDS:
            positive += all_text.count(keyword)

        for keyword in NEGATIVE_KEYWORDS:
            negative += all_text.count(keyword)

        # Normalize by conversation length
        word_count = max(1, len(all_text.split()))
        positive = positive / word_count * 100
        negative = negative / word_count * 100

        return positive, negative

    def _calculate_trust_deltas(
        self,
        state: ConversationState,
        positive_score: float,
        negative_score: float,
    ) -> tuple[float, float]:
        """Calculate trust changes from conversation."""
        base_delta = 0.05  # Base trust gain from any interaction

        # Modify based on sentiment
        sentiment_modifier = (positive_score - negative_score) / 10
        sentiment_modifier = max(-0.2, min(0.2, sentiment_modifier))

        # Longer conversations build more trust (if positive)
        length_modifier = min(0.1, state.turn_count * 0.02)
        if negative_score > positive_score:
            length_modifier = -length_modifier

        # Both parties adjust similarly (slightly asymmetric)
        initiator_delta = base_delta + sentiment_modifier + length_modifier
        target_delta = base_delta + sentiment_modifier * 0.8 + length_modifier * 0.8

        return initiator_delta, target_delta

    def _check_recruitment_success(self, state: ConversationState) -> bool:
        """Check if a recruitment attempt succeeded."""
        # Look at target's responses for positive signals
        target_id = state.context.listener_id
        target_turns = [t for t in state.turns if t.speaker_id == target_id]

        if not target_turns:
            return False

        # Check last few responses from target
        recent_text = " ".join(t.text.lower() for t in target_turns[-3:])

        positive_signals = sum(
            1 for phrase in RECRUITMENT_POSITIVE
            if phrase in recent_text
        )

        negative_signals = sum(
            1 for phrase in RECRUITMENT_NEGATIVE
            if phrase in recent_text
        )

        # Success if positive signals outweigh negative
        # and conversation didn't end badly
        if state.end_reason == "hostile_ending":
            return False

        return positive_signals > negative_signals and positive_signals >= 1

    def _extract_revealed_info(self, state: ConversationState) -> list[str]:
        """Extract information revealed during interrogation."""
        revealed = []
        target_id = state.context.listener_id

        target_text = " ".join(
            t.text.lower() for t in state.turns
            if t.speaker_id == target_id
        )

        for pattern in INFO_REVEALED_PATTERNS:
            matches = re.findall(pattern, target_text)
            if matches:
                revealed.append(f"Pattern matched: {pattern}")

        return revealed

    def _check_suspicion(self, state: ConversationState) -> tuple[bool, float]:
        """Check if suspicious content was discussed."""
        all_text = " ".join(t.text.lower() for t in state.turns)

        suspicion_count = sum(
            1 for keyword in SUSPICIOUS_KEYWORDS
            if keyword in all_text
        )

        if suspicion_count == 0:
            return False, 0.0

        # Base suspicion amount
        amount = min(0.3, suspicion_count * 0.05)

        # Higher if guards are nearby
        if state.context.nearby_agents:
            # Check if any nearby agent might be a guard
            # (In real implementation, we'd check agent specs)
            amount *= 1.5

        # Higher in public locations
        from ..core.types import Location
        public_locations = {Location.MARKET, Location.TOWN_SQUARE}
        if state.context.location in public_locations:
            amount *= 1.3

        return True, min(0.5, amount)

    def _infer_emotional_state(
        self,
        state: ConversationState,
        is_initiator: bool,
    ) -> EmotionalState:
        """Infer emotional state from conversation."""
        agent_id = state.context.speaker_id if is_initiator else state.context.listener_id
        agent_turns = [t for t in state.turns if t.speaker_id == agent_id]

        if not agent_turns:
            return EmotionalState.CALM

        recent_text = " ".join(t.text.lower() for t in agent_turns[-2:])

        # Check for specific emotions
        if any(w in recent_text for w in ["angry", "furious", "rage"]):
            return EmotionalState.ANGRY
        if any(w in recent_text for w in ["afraid", "scared", "fear"]):
            return EmotionalState.FEARFUL
        if any(w in recent_text for w in ["hope", "maybe", "could be"]):
            return EmotionalState.HOPEFUL
        if any(w in recent_text for w in ["worried", "nervous", "uneasy"]):
            return EmotionalState.ANXIOUS
        if any(w in recent_text for w in ["suspicious", "don't trust"]):
            return EmotionalState.SUSPICIOUS

        return EmotionalState.CALM


# =============================================================================
# LLM-BASED PARSER
# =============================================================================

class LLMOutcomeParser:
    """
    Parse conversation outcomes using LLM analysis.

    More accurate but slower and costs tokens.
    """

    ANALYSIS_PROMPT = """Analyze this conversation and provide a structured assessment.

CONVERSATION:
{transcript}

CONTEXT:
- Initiator: {initiator_name} (Intent: {intent})
- Target: {target_name}
- Location: {location}

Provide your analysis in EXACTLY this format (fill in values):

SENTIMENT: [positive/negative/neutral]
TRUST_CHANGE_INITIATOR: [number from -0.3 to 0.3]
TRUST_CHANGE_TARGET: [number from -0.3 to 0.3]
RECRUITMENT_SUCCESS: [true/false/not_attempted]
INFORMATION_REVEALED: [none/list items]
SUSPICION_RAISED: [true/false]
SUSPICION_AMOUNT: [number from 0 to 0.5]
INITIATOR_EMOTION: [calm/anxious/angry/fearful/hopeful/suspicious]
TARGET_EMOTION: [calm/anxious/angry/fearful/hopeful/suspicious]

Be conservative. Only mark recruitment as successful if there are clear verbal agreements.
Only mark suspicion if truly dangerous topics were discussed."""

    def __init__(self, backend: LLMBackend):
        self.backend = backend

    def parse(self, state: ConversationState) -> ConversationOutcome:
        """Parse using LLM analysis."""
        context = state.context

        # Build transcript for prompt
        transcript_lines = []
        for turn in state.turns:
            transcript_lines.append(f"[{turn.speaker_name}]: {turn.text}")
        transcript = "\n".join(transcript_lines)

        # Build analysis prompt
        prompt = self.ANALYSIS_PROMPT.format(
            transcript=transcript,
            initiator_name=context.speaker_spec.name,
            target_name=context.listener_spec.name,
            intent=context.intent.value,
            location=context.location.value,
        )

        # Get LLM analysis
        response = self.backend.generate(
            prompt=prompt,
            system_prompt="You are a precise conversation analyst. Follow the output format exactly.",
            temperature=0.3,  # Low temperature for consistency
            max_tokens=300,
        )

        # Parse response
        return self._parse_response(response.content, state)

    def _parse_response(
        self,
        response: str,
        state: ConversationState,
    ) -> ConversationOutcome:
        """Parse LLM response into ConversationOutcome."""
        context = state.context

        # Default values
        was_positive = True
        initiator_trust_delta = 0.05
        target_trust_delta = 0.05
        recruitment_attempted = context.intent == ConversationIntent.RECRUIT
        recruitment_successful = False
        information_revealed = []
        suspicion_raised = False
        suspicion_amount = 0.0
        initiator_emotion = EmotionalState.CALM
        target_emotion = EmotionalState.CALM

        # Parse each line
        for line in response.upper().split("\n"):
            line = line.strip()

            if line.startswith("SENTIMENT:"):
                value = line.split(":", 1)[1].strip().lower()
                was_positive = value == "positive"

            elif line.startswith("TRUST_CHANGE_INITIATOR:"):
                try:
                    initiator_trust_delta = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass

            elif line.startswith("TRUST_CHANGE_TARGET:"):
                try:
                    target_trust_delta = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass

            elif line.startswith("RECRUITMENT_SUCCESS:"):
                value = line.split(":", 1)[1].strip().lower()
                recruitment_successful = value == "true"

            elif line.startswith("INFORMATION_REVEALED:"):
                value = line.split(":", 1)[1].strip().lower()
                if value != "none":
                    information_revealed = [item.strip() for item in value.split(",")]

            elif line.startswith("SUSPICION_RAISED:"):
                value = line.split(":", 1)[1].strip().lower()
                suspicion_raised = value == "true"

            elif line.startswith("SUSPICION_AMOUNT:"):
                try:
                    suspicion_amount = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass

            elif line.startswith("INITIATOR_EMOTION:"):
                value = line.split(":", 1)[1].strip().lower()
                initiator_emotion = self._parse_emotion(value)

            elif line.startswith("TARGET_EMOTION:"):
                value = line.split(":", 1)[1].strip().lower()
                target_emotion = self._parse_emotion(value)

        # Build transcript
        transcript = [
            DialogueTurn(speaker_id=t.speaker_id, text=t.text)
            for t in state.turns
        ]

        return ConversationOutcome(
            initiator_id=context.speaker_id,
            target_id=context.listener_id,
            transcript=transcript,
            turn_count=state.turn_count,
            initiator_trust_delta=initiator_trust_delta,
            target_trust_delta=target_trust_delta,
            recruitment_attempted=recruitment_attempted,
            recruitment_successful=recruitment_successful,
            interrogation_attempted=context.intent == ConversationIntent.INTERROGATE,
            information_revealed=information_revealed,
            suspicion_raised=suspicion_raised,
            suspicion_amount=suspicion_amount,
            was_positive=was_positive,
            reason_ended=state.end_reason,
            initiator_emotional_state=initiator_emotion,
            target_emotional_state=target_emotion,
        )

    def _parse_emotion(self, value: str) -> EmotionalState:
        """Parse emotion string to enum."""
        emotion_map = {
            "calm": EmotionalState.CALM,
            "anxious": EmotionalState.ANXIOUS,
            "angry": EmotionalState.ANGRY,
            "fearful": EmotionalState.FEARFUL,
            "hopeful": EmotionalState.HOPEFUL,
            "suspicious": EmotionalState.SUSPICIOUS,
            "desperate": EmotionalState.DESPERATE,
            "content": EmotionalState.CONTENT,
        }
        return emotion_map.get(value, EmotionalState.CALM)


# =============================================================================
# UNIFIED PARSER INTERFACE
# =============================================================================

def parse_conversation_outcome(
    state: ConversationState,
    backend: Optional[LLMBackend] = None,
    use_llm: bool = False,
) -> ConversationOutcome:
    """
    Parse a conversation into a structured outcome.

    Args:
        state: Completed conversation state
        backend: LLM backend (required if use_llm=True)
        use_llm: Whether to use LLM analysis

    Returns:
        ConversationOutcome for simulation integration
    """
    if use_llm:
        if backend is None:
            raise ValueError("Backend required for LLM parsing")
        parser = LLMOutcomeParser(backend)
    else:
        parser = RuleBasedOutcomeParser()

    return parser.parse(state)
