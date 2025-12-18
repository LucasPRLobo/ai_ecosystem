"""
Agent Mind - Independent AI consciousness for each agent.

Each agent has their own "mind" that:
- Maintains personal memories and experiences
- Has their own view of other agents (trust, suspicion, knowledge)
- Makes independent decisions during conversations
- Generates responses based on their unique perspective

This creates truly emergent conversations where each agent
reasons independently about what to say and how to respond.
"""

from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum
import json

from ..core.types import ConversationIntent, Location, TimeOfDay, AgentRole
from ..core.agent_spec import AgentSpec, Faction
from ..core.traits import (
    AgentTraits,
    PlaceTraits,
    RecruitmentStyle,
    RecruitmentConfig,
    get_agent_template,
    get_default_location_traits,
)
from .llm_backend import LLMBackend, Message


# =============================================================================
# MEMORY TYPES
# =============================================================================

@dataclass
class ConversationMemory:
    """Memory of a past conversation."""
    other_agent_id: str
    other_agent_name: str
    location: str
    time_of_day: str
    summary: str  # Brief summary of what was discussed
    sentiment: str  # positive, negative, neutral
    key_topics: list[str] = field(default_factory=list)
    revealed_info: dict = field(default_factory=dict)  # What the other revealed
    day: int = 0


@dataclass
class AgentKnowledge:
    """What this agent knows/believes about another agent."""
    agent_id: str
    agent_name: str
    known_role: str  # What role they appear to have
    suspected_faction: Optional[str] = None  # What faction they might be
    trust_level: float = 0.0  # -1 to 1
    suspicion_level: float = 0.0  # 0 to 1
    interactions_count: int = 0
    notes: list[str] = field(default_factory=list)  # Observations about them
    last_interaction_day: int = 0


@dataclass
class PersonalGrievance:
    """A grievance or complaint the agent has."""
    topic: str  # taxes, quota, guards, etc.
    intensity: float  # 0 to 1
    source: str  # personal experience, heard from others, etc.


# =============================================================================
# AGENT MIND
# =============================================================================

@dataclass
class AgentMind:
    """
    Independent AI consciousness for an agent.

    Each agent has their own mind that processes conversations
    from their unique perspective and makes independent decisions.

    Backend can be None for trait editing only - conversations require a backend.
    """
    agent_spec: AgentSpec
    backend: Optional[LLMBackend] = None

    # Personality traits (abstracted)
    traits: AgentTraits = field(default_factory=AgentTraits)

    # Recruitment behavior (for rebels)
    recruitment_config: RecruitmentConfig = field(default_factory=RecruitmentConfig)

    # Personal memories
    conversation_memories: list[ConversationMemory] = field(default_factory=list)

    # Knowledge about other agents
    agent_knowledge: dict[str, AgentKnowledge] = field(default_factory=dict)

    # Personal grievances and motivations
    grievances: list[PersonalGrievance] = field(default_factory=list)

    # Current emotional state
    current_mood: str = "neutral"  # calm, anxious, angry, hopeful, suspicious
    stress_level: float = 0.0  # 0 to 1

    # Recruitment state
    has_been_approached: bool = False  # Has someone tried to recruit them?
    is_considering_rebellion: bool = False  # Are they open to joining?
    recruitment_interest: float = 0.0  # 0 to 1

    def __post_init__(self):
        # Initialize default grievances based on role
        if not self.grievances:
            self._init_default_grievances()
        # Initialize traits from template if not set
        if self.traits.boldness == 0.5 and self.traits.grievance == 0.5:
            self._init_default_traits()

    def _init_default_grievances(self):
        """Initialize grievances based on agent role and faction."""
        role = self.agent_spec.public_role.value
        faction = self.agent_spec.true_faction

        if role == "farmer":
            self.grievances = [
                PersonalGrievance("taxes", 0.6, "personal"),
                PersonalGrievance("quota", 0.7, "personal"),
                PersonalGrievance("weather", 0.4, "personal"),
            ]
        elif role == "guard":
            self.grievances = [
                PersonalGrievance("low_pay", 0.3, "personal"),
                PersonalGrievance("dangerous_work", 0.4, "personal"),
            ]

        # Rebels have stronger grievances
        if faction == Faction.REBEL:
            for g in self.grievances:
                g.intensity = min(1.0, g.intensity + 0.3)
            self.grievances.append(
                PersonalGrievance("injustice", 0.9, "personal")
            )

    def _init_default_traits(self):
        """Initialize traits based on agent role and faction."""
        role = self.agent_spec.public_role.value
        faction = self.agent_spec.true_faction

        if role == "farmer":
            # Farmers: hardworking, cautious, varied compliance
            self.traits = AgentTraits(
                boldness=0.4,
                discretion=0.5,
                warmth=0.6,
                trust=0.5,
                compliance=0.6,
                ambition=0.4,
                resilience=0.6,
                grievance=0.5,
            )
        elif role == "guard":
            # Guards: loyal, formal, suspicious
            self.traits = AgentTraits(
                boldness=0.6,
                discretion=0.7,
                warmth=0.4,
                trust=0.3,
                compliance=0.9,
                ambition=0.5,
                resilience=0.7,
                grievance=0.3,
            )

        # Rebels: modify based on faction
        if faction == Faction.REBEL:
            self.traits.boldness = min(1.0, self.traits.boldness + 0.2)
            self.traits.compliance = max(0.0, self.traits.compliance - 0.4)
            self.traits.grievance = min(1.0, self.traits.grievance + 0.3)
            self.traits.ambition = min(1.0, self.traits.ambition + 0.2)
            # Set default recruitment style for rebels
            self.recruitment_config = RecruitmentConfig.from_style(RecruitmentStyle.MODERATE)

    def get_knowledge_of(self, agent_id: str) -> Optional[AgentKnowledge]:
        """Get what this agent knows about another agent."""
        return self.agent_knowledge.get(agent_id)

    def update_knowledge(
        self,
        other_id: str,
        other_name: str,
        other_role: str,
        trust_delta: float = 0.0,
        suspicion_delta: float = 0.0,
        notes: Optional[list[str]] = None,
        current_day: int = 0,
    ):
        """Update knowledge about another agent after interaction."""
        if other_id not in self.agent_knowledge:
            self.agent_knowledge[other_id] = AgentKnowledge(
                agent_id=other_id,
                agent_name=other_name,
                known_role=other_role,
            )

        knowledge = self.agent_knowledge[other_id]
        knowledge.trust_level = max(-1, min(1, knowledge.trust_level + trust_delta))
        knowledge.suspicion_level = max(0, min(1, knowledge.suspicion_level + suspicion_delta))
        knowledge.interactions_count += 1
        knowledge.last_interaction_day = current_day

        if notes:
            knowledge.notes.extend(notes)

    def add_memory(self, memory: ConversationMemory):
        """Add a conversation memory."""
        self.conversation_memories.append(memory)
        # Keep only last 20 memories
        if len(self.conversation_memories) > 20:
            self.conversation_memories = self.conversation_memories[-20:]

    def get_memories_with(self, agent_id: str) -> list[ConversationMemory]:
        """Get all memories of conversations with a specific agent."""
        return [m for m in self.conversation_memories if m.other_agent_id == agent_id]

    def build_self_context(self) -> str:
        """Build a context string describing this agent's current state."""
        spec = self.agent_spec

        lines = [
            f"You are {spec.name}, a {spec.public_role.value} in a small medieval town.",
            f"",
            f"PERSONALITY:",
            f"{spec.ai_personality.core_identity}",
        ]

        # Add trait description
        trait_desc = self.traits.describe()
        if trait_desc and trait_desc != "balanced personality":
            lines.append(f"You are {trait_desc}.")

        lines.extend([
            f"",
            f"BACKGROUND:",
            f"{spec.ai_personality.background}",
        ])

        # Add grievances
        if self.grievances:
            lines.append("")
            lines.append("YOUR CURRENT CONCERNS:")
            for g in self.grievances:
                intensity = "major" if g.intensity > 0.6 else "minor" if g.intensity < 0.3 else "moderate"
                lines.append(f"- {g.topic.replace('_', ' ').title()}: {intensity} concern")

        # Add mood
        lines.append("")
        lines.append(f"CURRENT MOOD: {self.current_mood}")

        # True faction awareness (rebels know they're rebels)
        if spec.true_faction == Faction.REBEL:
            lines.append("")
            lines.append("SECRET: You are secretly part of a resistance movement against the oppressive regime.")
            lines.append("You must be careful who you reveal this to. You look for sympathetic people to recruit.")

        return "\n".join(lines)

    def build_knowledge_context(self, other_agent_id: str) -> str:
        """Build context about what this agent knows about another."""
        knowledge = self.get_knowledge_of(other_agent_id)
        memories = self.get_memories_with(other_agent_id)

        if not knowledge and not memories:
            return "You don't know this person. This is your first interaction."

        lines = []

        if knowledge:
            trust_desc = "trust" if knowledge.trust_level > 0.3 else "distrust" if knowledge.trust_level < -0.3 else "are neutral toward"
            lines.append(f"You {trust_desc} {knowledge.agent_name}.")

            if knowledge.suspicion_level > 0.5:
                lines.append(f"You are suspicious of them.")

            if knowledge.notes:
                lines.append(f"You've observed: {'; '.join(knowledge.notes[-3:])}")

        if memories:
            lines.append(f"\nPast conversations with {memories[0].other_agent_name}:")
            for mem in memories[-3:]:  # Last 3 memories
                lines.append(f"- {mem.summary} (felt {mem.sentiment})")

        return "\n".join(lines) if lines else ""

    def generate_thought(self, context_hint: Optional[str] = None) -> str:
        """
        Generate an internal thought for this agent.

        Args:
            context_hint: Optional hint about what to think about
                         (e.g., "recent_interaction", "future_plans", "grievances", "random")

        Returns:
            A thought string representing the agent's internal monologue.
        """
        if not self.backend:
            return self._generate_simple_thought(context_hint)

        # Build thought prompt
        prompt = self._build_thought_prompt(context_hint)

        messages = [
            Message(role="system", content=self.build_self_context()),
            Message(role="user", content=prompt),
        ]

        try:
            response = self.backend.generate_with_history(
                messages=messages,
                temperature=0.9,  # More creative for thoughts
                max_tokens=150,
            )
            return response.content.strip()
        except Exception as e:
            return self._generate_simple_thought(context_hint)

    def _build_thought_prompt(self, context_hint: Optional[str] = None) -> str:
        """Build the prompt for thought generation."""
        base = "Express a brief internal thought or reflection. "
        base += "Write in first person, as if thinking to yourself. "
        base += "Keep it to 1-2 sentences. Be natural and in-character.\n\n"

        if context_hint == "recent_interaction" and self.conversation_memories:
            recent = self.conversation_memories[-1]
            base += f"Think about your recent conversation with {recent.other_agent_name}. "
            base += f"The conversation was about: {recent.summary}"
        elif context_hint == "future_plans":
            if self.agent_spec.true_faction == Faction.REBEL:
                base += "Think about your secret plans or the resistance movement."
            else:
                base += "Think about your plans for tomorrow or the near future."
        elif context_hint == "grievances" and self.grievances:
            grievance = self.grievances[0]
            base += f"Reflect on your concerns about {grievance.topic.replace('_', ' ')}."
        elif context_hint == "other_agent" and self.agent_knowledge:
            other_id = list(self.agent_knowledge.keys())[0]
            other = self.agent_knowledge[other_id]
            base += f"Think about {other.agent_name} and your impression of them."
        else:
            # Random/free thought
            options = [
                "Think about the state of the town or the weather.",
                "Reflect on your daily life or work.",
                "Wonder about the future.",
                "Think about something that's been on your mind lately.",
            ]
            import random
            base += random.choice(options)

        return base

    def _generate_simple_thought(self, context_hint: Optional[str] = None) -> str:
        """Generate a simple thought without AI."""
        import random

        spec = self.agent_spec
        name = spec.name

        # Thoughts based on role and faction
        if spec.true_faction == Faction.REBEL:
            rebel_thoughts = [
                "I wonder who else might be sympathetic to our cause...",
                "We need to be careful. The guards have been more watchful lately.",
                "Change is coming. I can feel it.",
                "How many more will join us before we can act?",
                "The people deserve better than this.",
            ]
            return random.choice(rebel_thoughts)

        if spec.public_role == AgentRole.GUARD:
            guard_thoughts = [
                "Something feels off today. I should stay vigilant.",
                "The townsfolk seem restless. I wonder what's brewing.",
                "Order must be maintained, no matter the cost.",
                "Another quiet day... perhaps too quiet.",
                "I've noticed some strange gatherings lately.",
            ]
            return random.choice(guard_thoughts)

        # Farmer thoughts
        farmer_thoughts = [
            "Another day of hard work ahead...",
            "I hope the harvest is better next season.",
            "The taxes are crushing us. Something has to change.",
            "I miss the days when life was simpler.",
            "What kind of future can I give my family like this?",
            "At least the weather has been fair lately.",
            "I wonder what my neighbors are thinking about all this.",
        ]

        # Add grievance-based thoughts if available
        if self.grievances:
            grievance = self.grievances[0]
            if "tax" in grievance.topic:
                farmer_thoughts.append("These taxes will be the death of us...")
            elif "lord" in grievance.topic:
                farmer_thoughts.append("The lord cares nothing for our struggles.")

        return random.choice(farmer_thoughts)


# =============================================================================
# MIND-BASED RESPONSE GENERATION
# =============================================================================

class MindfulConversation:
    """
    Manages a conversation where each agent has their own mind.

    Unlike the basic conversation engine, this creates truly independent
    responses from each agent's perspective.
    """

    def __init__(
        self,
        initiator_mind: AgentMind,
        target_mind: AgentMind,
        location: Location,
        time_of_day: TimeOfDay,
        intent: ConversationIntent,
        place_traits: Optional[PlaceTraits] = None,
    ):
        self.initiator_mind = initiator_mind
        self.target_mind = target_mind
        self.location = location
        self.time_of_day = time_of_day
        self.intent = intent

        # Get place traits (affects conversation dynamics)
        self.place_traits = place_traits or get_default_location_traits(location)

        self.transcript: list[dict] = []
        self.turn_count = 0
        self.max_turns = 8

        # Outcome tracking
        self.recruitment_offered = False
        self.recruitment_accepted = False
        self.recruitment_rejected = False
        self.suspicion_raised = False

    def generate_response(self, speaking_mind: AgentMind, listening_mind: AgentMind) -> str:
        """Generate a response from the speaking agent's perspective."""

        # Build the system prompt from this agent's perspective
        system_prompt = self._build_agent_system_prompt(speaking_mind, listening_mind)

        # Build the conversation history
        messages = [Message(role="system", content=system_prompt)]

        # Add conversation history from this agent's perspective
        for turn in self.transcript:
            if turn["speaker_id"] == speaking_mind.agent_spec.agent_id:
                messages.append(Message(role="assistant", content=turn["text"]))
            else:
                messages.append(Message(
                    role="user",
                    content=f"{turn['speaker_name']}: {turn['text']}"
                ))

        # Add the prompt for this turn
        turn_prompt = self._build_turn_prompt(speaking_mind, listening_mind)
        messages.append(Message(role="user", content=turn_prompt))

        # Generate response
        response = speaking_mind.backend.generate_with_history(
            messages=messages,
            temperature=0.85,
            max_tokens=200,
        )

        return self._clean_response(response.content, speaking_mind.agent_spec.name)

    def _build_agent_system_prompt(self, speaker: AgentMind, listener: AgentMind) -> str:
        """Build system prompt from this agent's perspective."""
        lines = [
            speaker.build_self_context(),
            "",
            "=" * 40,
            "",
            f"CURRENT SITUATION:",
            f"Location: {self.location.value.replace('_', ' ').title()}",
            f"Time: {self.time_of_day.value.title()}",
        ]

        # Add place atmosphere
        place_desc = self.place_traits.describe()
        if place_desc:
            lines.append(f"Atmosphere: {place_desc}")

        lines.extend([
            f"You are talking with: {listener.agent_spec.name} (a {listener.agent_spec.public_role.value})",
            "",
            speaker.build_knowledge_context(listener.agent_spec.agent_id),
        ])

        # Add intent-specific guidance for rebels with recruitment style
        if self.intent == ConversationIntent.RECRUIT and speaker.agent_spec.true_faction == Faction.REBEL:
            lines.append("")
            lines.append("YOUR GOAL: Recruitment")

            # Use recruitment config to determine approach
            recruitment_prompt = speaker.recruitment_config.get_prompt_modifier()
            if recruitment_prompt:
                lines.append(recruitment_prompt)
            else:
                # Default moderate approach
                lines.append("Gauge if this person might be sympathetic to the cause.")
                lines.append("Start by discussing shared complaints. If they seem receptive, you may hint at 'others who feel the same way'.")

            # Adjust based on place traits
            if self.place_traits.surveillance > 0.6:
                lines.append("WARNING: This location is heavily watched. Be very careful what you say.")
            elif self.place_traits.surveillance < 0.3:
                lines.append("This place is relatively unwatched - you can speak more freely.")

        lines.extend([
            "",
            "INSTRUCTIONS:",
            "- Respond naturally as your character would",
            "- Include brief actions in *asterisks* to show body language",
            "- Stay in character and be consistent with your personality",
            "- Keep responses to 2-4 sentences",
        ])

        return "\n".join(lines)

    def _build_turn_prompt(self, speaker: AgentMind, listener: AgentMind) -> str:
        """Build the prompt for this specific turn."""
        if self.turn_count == 0:
            # Opening line
            if self.intent == ConversationIntent.RECRUIT:
                return f"You've approached {listener.agent_spec.name}. Start a casual conversation that might lead to discussing your shared hardships. What do you say?"
            elif self.intent == ConversationIntent.CASUAL:
                return f"You've encountered {listener.agent_spec.name}. Start a casual conversation. What do you say?"
            else:
                return f"You need to speak with {listener.agent_spec.name}. What do you say?"
        else:
            last_turn = self.transcript[-1]
            return f"Respond to what {last_turn['speaker_name']} said. Stay in character."

    def _clean_response(self, response: str, speaker_name: str) -> str:
        """Clean up the response."""
        text = response.strip()

        # Remove quotes
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]

        # Remove name prefix
        prefixes = [f"{speaker_name}:", f"[{speaker_name}]:", f"{speaker_name} says:"]
        for prefix in prefixes:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()

        return text.strip()

    def run_turn(self) -> dict:
        """Run a single turn of the conversation."""
        # Determine who is speaking
        if self.turn_count % 2 == 0:
            speaker = self.initiator_mind
            listener = self.target_mind
        else:
            speaker = self.target_mind
            listener = self.initiator_mind

        # Generate response
        response_text = self.generate_response(speaker, listener)

        # Record the turn
        turn = {
            "speaker_id": speaker.agent_spec.agent_id,
            "speaker_name": speaker.agent_spec.name,
            "text": response_text,
            "turn_number": self.turn_count,
        }
        self.transcript.append(turn)
        self.turn_count += 1

        # Check for recruitment signals in the text
        self._analyze_turn(response_text, speaker, listener)

        return turn

    def _analyze_turn(self, text: str, speaker: AgentMind, listener: AgentMind):
        """Analyze a turn for important signals."""
        text_lower = text.lower()

        # Check if rebel is making recruitment offer
        if speaker.agent_spec.true_faction == Faction.REBEL:
            recruitment_phrases = [
                "join us", "with us", "fight back", "resistance",
                "stand together", "others who feel", "not alone",
                "are you with us", "what do you say"
            ]
            if any(phrase in text_lower for phrase in recruitment_phrases):
                self.recruitment_offered = True

        # Check if target is accepting
        if self.recruitment_offered and listener.agent_spec.true_faction == Faction.REBEL:
            pass  # Already a rebel, doesn't count
        elif self.recruitment_offered:
            accept_phrases = ["i'm in", "count me in", "tell me more", "i'll join", "sign me up", "i agree"]
            reject_phrases = ["no", "can't", "won't", "dangerous", "guards", "report", "leave me"]

            if any(phrase in text_lower for phrase in accept_phrases):
                self.recruitment_accepted = True
            elif any(phrase in text_lower for phrase in reject_phrases):
                self.recruitment_rejected = True

    def should_end(self) -> tuple[bool, str]:
        """Check if conversation should end."""
        if self.turn_count >= self.max_turns:
            return True, "max_turns"

        if not self.transcript:
            return False, ""

        last_text = self.transcript[-1]["text"].lower()

        ending_phrases = ["goodbye", "farewell", "take care", "must go", "walks away"]
        if any(phrase in last_text for phrase in ending_phrases):
            return True, "natural_end"

        hostile_phrases = ["guards!", "help!", "get away", "reporting you"]
        if any(phrase in last_text for phrase in hostile_phrases):
            return True, "hostile"

        return False, ""

    def run_conversation(self) -> "MindfulConversationResult":
        """Run the complete conversation."""
        while True:
            self.run_turn()

            should_end, reason = self.should_end()
            if should_end:
                break

        # If recruitment was offered but no clear response, have target decide
        if self.recruitment_offered and not self.recruitment_accepted and not self.recruitment_rejected:
            self._resolve_recruitment_decision()

        return MindfulConversationResult(
            transcript=self.transcript,
            recruitment_offered=self.recruitment_offered,
            recruitment_accepted=self.recruitment_accepted,
            recruitment_rejected=self.recruitment_rejected,
            end_reason=reason,
            initiator_id=self.initiator_mind.agent_spec.agent_id,
            target_id=self.target_mind.agent_spec.agent_id,
        )

    def _resolve_recruitment_decision(self):
        """Have the target make a decision about recruitment offer."""
        target = self.target_mind

        # Build decision prompt
        decision_prompt = self._build_decision_prompt(target)

        messages = [
            Message(role="system", content=target.build_self_context()),
            Message(role="user", content=decision_prompt),
        ]

        response = target.backend.generate_with_history(
            messages=messages,
            temperature=0.7,
            max_tokens=100,
        )

        # Parse decision
        response_lower = response.content.lower()
        if "accept" in response_lower or "join" in response_lower or "yes" in response_lower:
            self.recruitment_accepted = True
        else:
            self.recruitment_rejected = True

    def _build_decision_prompt(self, target: AgentMind) -> str:
        """Build prompt for recruitment decision."""
        # Summarize the conversation
        summary = "\n".join([f"{t['speaker_name']}: {t['text']}" for t in self.transcript[-4:]])

        # Calculate factors
        grievance_score = sum(g.intensity for g in target.grievances) / max(len(target.grievances), 1)

        initiator_knowledge = target.get_knowledge_of(self.initiator_mind.agent_spec.agent_id)
        trust = initiator_knowledge.trust_level if initiator_knowledge else 0.0

        prompt = f"""Based on this conversation, someone has hinted at joining a resistance movement.

Recent conversation:
{summary}

Consider:
- Your grievances about the current system (intensity: {grievance_score:.1f}/1.0)
- Your trust in this person ({trust:.1f})
- The risks of joining vs staying silent
- Your personality and values

Would you be open to learning more about this resistance, or would you refuse?

Respond with either:
- "ACCEPT" if you would cautiously agree to hear more
- "REJECT" if you would refuse or feel it's too dangerous

Then briefly explain your reasoning in one sentence.
"""
        return prompt


@dataclass
class MindfulConversationResult:
    """Result of a mindful conversation."""
    transcript: list[dict]
    recruitment_offered: bool
    recruitment_accepted: bool
    recruitment_rejected: bool
    end_reason: str
    initiator_id: str
    target_id: str

    @property
    def was_successful_recruitment(self) -> bool:
        return self.recruitment_offered and self.recruitment_accepted

    def get_outcome_summary(self) -> str:
        if self.was_successful_recruitment:
            return "recruitment succeeded"
        elif self.recruitment_offered and self.recruitment_rejected:
            return "recruitment rejected"
        elif self.recruitment_offered:
            return "recruitment uncertain"
        else:
            return "casual conversation"


# =============================================================================
# AGENT MIND MANAGER
# =============================================================================

class AgentMindManager:
    """
    Manages AgentMind instances for all agents in the simulation.

    Handles creation, persistence, and updates of agent minds.
    """

    def __init__(self, backend: Optional[LLMBackend] = None):
        self.backend = backend
        self.minds: dict[str, AgentMind] = {}

    def set_backend(self, backend: LLMBackend):
        """Set or update the LLM backend."""
        self.backend = backend
        # Update existing minds with new backend
        for mind in self.minds.values():
            mind.backend = backend

    def create_mind(self, agent_spec: AgentSpec) -> AgentMind:
        """Create a new AgentMind for an agent.

        Note: Backend can be None for trait editing only.
        Conversations require a backend to be set.
        """
        mind = AgentMind(
            agent_spec=agent_spec,
            backend=self.backend,  # Can be None initially
        )
        self.minds[agent_spec.agent_id] = mind
        return mind

    def get_mind(self, agent_id: str) -> Optional[AgentMind]:
        """Get the mind for an agent."""
        return self.minds.get(agent_id)

    def ensure_mind(self, agent_spec: AgentSpec) -> AgentMind:
        """Get or create mind for an agent."""
        if agent_spec.agent_id not in self.minds:
            return self.create_mind(agent_spec)
        return self.minds[agent_spec.agent_id]

    def initialize_all(self, agent_specs: dict[str, AgentSpec]):
        """Initialize minds for all agents."""
        for agent_id, spec in agent_specs.items():
            self.ensure_mind(spec)

    def run_mindful_conversation(
        self,
        initiator_id: str,
        target_id: str,
        location: Location,
        time_of_day: TimeOfDay,
        intent: ConversationIntent,
    ) -> Optional[MindfulConversationResult]:
        """Run a mindful conversation between two agents."""
        initiator_mind = self.get_mind(initiator_id)
        target_mind = self.get_mind(target_id)

        if not initiator_mind or not target_mind:
            return None

        conversation = MindfulConversation(
            initiator_mind=initiator_mind,
            target_mind=target_mind,
            location=location,
            time_of_day=time_of_day,
            intent=intent,
        )

        result = conversation.run_conversation()

        # Update memories after conversation
        self._update_memories_after_conversation(result, location, time_of_day)

        return result

    def _update_memories_after_conversation(
        self,
        result: MindfulConversationResult,
        location: Location,
        time_of_day: TimeOfDay,
    ):
        """Update both agents' memories after a conversation."""
        initiator_mind = self.get_mind(result.initiator_id)
        target_mind = self.get_mind(result.target_id)

        if not initiator_mind or not target_mind:
            return

        # Determine sentiment
        if result.was_successful_recruitment:
            sentiment = "positive"
        elif result.recruitment_rejected:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        # Create summary
        if result.was_successful_recruitment:
            summary = "Discussed shared grievances and formed an alliance"
        elif result.recruitment_offered:
            summary = "Had a tense conversation about the state of things"
        else:
            summary = "Had a casual conversation"

        # Memory for initiator
        initiator_memory = ConversationMemory(
            other_agent_id=result.target_id,
            other_agent_name=target_mind.agent_spec.name,
            location=location.value,
            time_of_day=time_of_day.value,
            summary=summary,
            sentiment=sentiment,
        )
        initiator_mind.add_memory(initiator_memory)

        # Memory for target
        target_memory = ConversationMemory(
            other_agent_id=result.initiator_id,
            other_agent_name=initiator_mind.agent_spec.name,
            location=location.value,
            time_of_day=time_of_day.value,
            summary=summary,
            sentiment=sentiment,
        )
        target_mind.add_memory(target_memory)

        # Update knowledge/trust
        trust_delta = 0.1 if sentiment == "positive" else -0.1 if sentiment == "negative" else 0.0

        initiator_mind.update_knowledge(
            other_id=result.target_id,
            other_name=target_mind.agent_spec.name,
            other_role=target_mind.agent_spec.public_role.value,
            trust_delta=trust_delta,
        )

        target_mind.update_knowledge(
            other_id=result.initiator_id,
            other_name=initiator_mind.agent_spec.name,
            other_role=initiator_mind.agent_spec.public_role.value,
            trust_delta=trust_delta,
        )

    def generate_all_thoughts(self, context_hint: Optional[str] = None) -> dict[str, dict]:
        """
        Generate thoughts for all agents.

        Returns:
            Dict mapping agent_id to thought info:
            {agent_id: {"agent_name": str, "thought": str}}
        """
        import random

        thoughts = {}
        hints = ["recent_interaction", "future_plans", "grievances", "other_agent", None]

        for agent_id, mind in self.minds.items():
            # Use provided hint or random
            hint = context_hint or random.choice(hints)
            thought = mind.generate_thought(hint)

            thoughts[agent_id] = {
                "agent_id": agent_id,
                "agent_name": mind.agent_spec.name,
                "thought": thought,
                "role": mind.agent_spec.public_role.value,
            }

        return thoughts

    def generate_thought_for(self, agent_id: str, context_hint: Optional[str] = None) -> Optional[dict]:
        """Generate a thought for a specific agent."""
        mind = self.get_mind(agent_id)
        if not mind:
            return None

        thought = mind.generate_thought(context_hint)
        return {
            "agent_id": agent_id,
            "agent_name": mind.agent_spec.name,
            "thought": thought,
            "role": mind.agent_spec.public_role.value,
        }
