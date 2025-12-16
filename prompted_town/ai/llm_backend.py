"""
LLM Backend abstraction for Prompted Town.

This module provides a clean interface for different LLM providers.
The conversation engine uses this interface; the actual provider
can be swapped without changing conversation logic.

Design principles:
- Protocol-based interface (duck typing)
- Async-ready but sync-first for simplicity
- Built-in caching for development
- Mock backend for testing
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable
import hashlib
import json
import os
from pathlib import Path


# =============================================================================
# MESSAGE TYPES
# =============================================================================

@dataclass
class Message:
    """A single message in a conversation."""
    role: str  # "system", "user", "assistant"
    content: str
    name: Optional[str] = None  # For multi-agent conversations

    def to_dict(self) -> dict:
        d = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        return d


@dataclass
class LLMResponse:
    """Response from an LLM."""
    content: str
    model: str
    usage: Optional[dict] = None  # Token counts if available
    raw_response: Optional[dict] = None  # Full API response


# =============================================================================
# BACKEND PROTOCOL
# =============================================================================

@runtime_checkable
class LLMBackend(Protocol):
    """
    Protocol for LLM backends.

    Any class implementing generate() and generate_with_history()
    can be used as a backend.
    """

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> LLMResponse:
        """Generate a response from a single prompt."""
        ...

    def generate_with_history(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> LLMResponse:
        """Generate a response given conversation history."""
        ...


# =============================================================================
# MOCK BACKEND (for testing)
# =============================================================================

@dataclass
class MockResponse:
    """A canned response for the mock backend."""
    trigger: str  # Substring that triggers this response
    response: str
    is_regex: bool = False


class MockLLMBackend:
    """
    Mock LLM backend for testing.

    Returns canned responses based on input patterns.
    Useful for:
    - Unit tests
    - Offline development
    - Deterministic behavior verification
    """

    def __init__(self, default_response: str = "I understand."):
        self.default_response = default_response
        self.canned_responses: list[MockResponse] = []
        self.call_history: list[dict] = []
        self._turn_counter: int = 0  # For varying responses

        # Add some default responses for common scenarios
        self._setup_default_responses()

    def _setup_default_responses(self):
        """Set up default canned responses for testing."""
        # Specific phrase triggers (checked first due to list order)
        self.add_response("what do you say", "Well met, friend. Another long day, isn't it?")
        self.add_response("start the conversation", "Well met, friend. How goes it?")
        self.add_response("wait for them to speak", "*nods in greeting*")

        # Topic triggers (only match in actual dialogue, not location descriptions)
        self.add_response("hello", "Hello there. What brings you here today?")
        self.add_response("weather been", "The weather has been harsh on the crops this season.")
        self.add_response("quota", "The quota weighs heavy on all of us, friend.")
        self.add_response("guards", "Best to keep your head down when they're around.")
        self.add_response("get a drink", "A drink sounds good after a long day's work.")
        self.add_response("rebel", "These are dangerous words, friend. But... I'm listening.")
        self.add_response("the lord", "The lord takes more than his share, that's certain.")
        self.add_response("trust you", "Trust is earned slowly in times like these.")
        self.add_response("where were you", "I don't know anything. I'm just a simple farmer.")

        # Conversation flow responses (for varied dialogue when no match)
        self._conversation_responses = [
            "Aye, that's the truth of it.",
            "Times are hard for everyone these days.",
            "I've been thinking the same thing, friend.",
            "The burden grows heavier each season.",
            "Something must change, but what can we do?",
            "Perhaps there's hope yet. Perhaps we can change things.",
            "We must look out for each other.",
            "The old ways are crumbling. New paths must be found.",
        ]

    def add_response(self, trigger: str, response: str, is_regex: bool = False):
        """Add a canned response."""
        self.canned_responses.append(MockResponse(trigger, response, is_regex))

    def clear_responses(self):
        """Clear all canned responses."""
        self.canned_responses = []

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> LLMResponse:
        """Generate a mock response."""
        self.call_history.append({
            "type": "generate",
            "prompt": prompt,
            "system_prompt": system_prompt,
        })

        response_text = self._find_response(prompt)

        return LLMResponse(
            content=response_text,
            model="mock",
            usage={"prompt_tokens": len(prompt.split()), "completion_tokens": len(response_text.split())},
        )

    def generate_with_history(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> LLMResponse:
        """Generate a mock response based on conversation history."""
        self.call_history.append({
            "type": "generate_with_history",
            "messages": [m.to_dict() for m in messages],
        })

        # Use the last user message to find response
        last_content = ""
        for msg in reversed(messages):
            if msg.role in ("user", "assistant"):
                last_content = msg.content
                break

        response_text = self._find_response(last_content)

        return LLMResponse(
            content=response_text,
            model="mock",
            usage={"prompt_tokens": sum(len(m.content.split()) for m in messages),
                   "completion_tokens": len(response_text.split())},
        )

    def _find_response(self, text: str, use_varied: bool = True) -> str:
        """Find a matching canned response."""
        text_lower = text.lower()
        for resp in self.canned_responses:
            if resp.is_regex:
                import re
                if re.search(resp.trigger, text_lower):
                    return resp.response
            else:
                if resp.trigger.lower() in text_lower:
                    return resp.response

        # No match found - use varied responses for natural conversation flow
        if use_varied and hasattr(self, '_conversation_responses'):
            response = self._conversation_responses[self._turn_counter % len(self._conversation_responses)]
            self._turn_counter += 1
            return response

        return self.default_response

    def reset_turn_counter(self):
        """Reset turn counter for new conversation."""
        self._turn_counter = 0


# =============================================================================
# OPENAI BACKEND
# =============================================================================

class OpenAIBackend:
    """
    OpenAI API backend.

    Requires OPENAI_API_KEY environment variable.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var.")

        # Lazy import
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> LLMResponse:
        """Generate a response using OpenAI."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return LLMResponse(
            content=response.choices[0].message.content,
            model=self.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
            raw_response=response.model_dump(),
        )

    def generate_with_history(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> LLMResponse:
        """Generate a response with conversation history."""
        api_messages = [m.to_dict() for m in messages]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=api_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return LLMResponse(
            content=response.choices[0].message.content,
            model=self.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
            raw_response=response.model_dump(),
        )


# =============================================================================
# ANTHROPIC BACKEND
# =============================================================================

class AnthropicBackend:
    """
    Anthropic Claude API backend.

    Requires ANTHROPIC_API_KEY environment variable.
    """

    def __init__(
        self,
        model: str = "claude-3-5-haiku-latest",
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

        if not self.api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY env var.")

        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> LLMResponse:
        """Generate a response using Anthropic Claude."""
        messages = [{"role": "user", "content": prompt}]

        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        response = self.client.messages.create(**kwargs)

        return LLMResponse(
            content=response.content[0].text,
            model=self.model,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
            },
            raw_response=response.model_dump(),
        )

    def generate_with_history(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> LLMResponse:
        """Generate a response with conversation history."""
        # Separate system message from conversation
        system_prompt = None
        api_messages = []

        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            else:
                api_messages.append({"role": msg.role, "content": msg.content})

        kwargs = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        response = self.client.messages.create(**kwargs)

        return LLMResponse(
            content=response.content[0].text,
            model=self.model,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
            },
            raw_response=response.model_dump(),
        )


# =============================================================================
# CACHING WRAPPER
# =============================================================================

class CachedBackend:
    """
    Caching wrapper for any LLM backend.

    Caches responses to disk to avoid repeated API calls during development.
    """

    def __init__(
        self,
        backend: LLMBackend,
        cache_dir: str = ".llm_cache",
        enabled: bool = True,
    ):
        self.backend = backend
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled

        if enabled:
            self.cache_dir.mkdir(exist_ok=True)

    def _cache_key(self, **kwargs) -> str:
        """Generate a cache key from arguments."""
        content = json.dumps(kwargs, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cached(self, key: str) -> Optional[LLMResponse]:
        """Get a cached response if it exists."""
        if not self.enabled:
            return None

        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                data = json.load(f)
                return LLMResponse(**data)
        return None

    def _save_cache(self, key: str, response: LLMResponse):
        """Save a response to cache."""
        if not self.enabled:
            return

        cache_file = self.cache_dir / f"{key}.json"
        with open(cache_file, "w") as f:
            json.dump({
                "content": response.content,
                "model": response.model,
                "usage": response.usage,
            }, f)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> LLMResponse:
        """Generate with caching."""
        key = self._cache_key(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        cached = self._get_cached(key)
        if cached:
            return cached

        response = self.backend.generate(prompt, system_prompt, temperature, max_tokens)
        self._save_cache(key, response)
        return response

    def generate_with_history(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> LLMResponse:
        """Generate with caching."""
        key = self._cache_key(
            messages=[m.to_dict() for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        cached = self._get_cached(key)
        if cached:
            return cached

        response = self.backend.generate_with_history(messages, temperature, max_tokens)
        self._save_cache(key, response)
        return response


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_backend(
    provider: str = "mock",
    model: Optional[str] = None,
    cache: bool = False,
    **kwargs,
) -> LLMBackend:
    """
    Create an LLM backend.

    Args:
        provider: "mock", "openai", or "anthropic"
        model: Model name (provider-specific)
        cache: Whether to enable response caching
        **kwargs: Additional provider-specific arguments

    Returns:
        An LLM backend instance
    """
    if provider == "mock":
        backend = MockLLMBackend(**kwargs)
    elif provider == "openai":
        backend = OpenAIBackend(model=model or "gpt-4o-mini", **kwargs)
    elif provider == "anthropic":
        backend = AnthropicBackend(model=model or "claude-3-5-haiku-latest", **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    if cache and provider != "mock":
        backend = CachedBackend(backend)

    return backend
