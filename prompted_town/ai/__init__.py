"""
AI module: LLM backends, prompt building, and conversation engine.

This module provides AI-powered conversation capabilities for Prompted Town.
Agents can have genuine conversations driven by their personalities.

Key components:
- LLM Backend: Provider-agnostic interface (OpenAI, Anthropic, Mock)
- Prompt Builder: Context-aware prompt construction
- Conversation Engine: Multi-turn dialogue orchestration
- Outcome Parser: Extract structured outcomes from conversations
"""

# LLM Backend
from .llm_backend import (
    Message,
    LLMResponse,
    LLMBackend,
    MockLLMBackend,
    OpenAIBackend,
    AnthropicBackend,
    CachedBackend,
    create_backend,
)

# Prompt Builder
from .prompt_builder import (
    ConversationContext,
    build_system_prompt,
    build_situation_prompt,
    build_intent_prompt,
    build_conversation_prompts,
    select_speaking_style,
    create_context_from_world,
)

# Conversation Engine
from .conversation import (
    ConversationPhase,
    Turn,
    ConversationState,
    ConversationEngine,
    run_conversation,
    print_conversation,
    should_end_conversation,
)

# Outcome Parser
from .outcome_parser import (
    RuleBasedOutcomeParser,
    LLMOutcomeParser,
    parse_conversation_outcome,
)

__all__ = [
    # LLM Backend
    "Message",
    "LLMResponse",
    "LLMBackend",
    "MockLLMBackend",
    "OpenAIBackend",
    "AnthropicBackend",
    "CachedBackend",
    "create_backend",
    # Prompt Builder
    "ConversationContext",
    "build_system_prompt",
    "build_situation_prompt",
    "build_intent_prompt",
    "build_conversation_prompts",
    "select_speaking_style",
    "create_context_from_world",
    # Conversation Engine
    "ConversationPhase",
    "Turn",
    "ConversationState",
    "ConversationEngine",
    "run_conversation",
    "print_conversation",
    "should_end_conversation",
    # Outcome Parser
    "RuleBasedOutcomeParser",
    "LLMOutcomeParser",
    "parse_conversation_outcome",
]
