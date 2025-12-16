"""
Tests for the AI conversation module.

Run with: python -m pytest prompted_town/tests/test_ai.py -v
Or directly: python -m prompted_town.tests.test_ai
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from prompted_town.core import (
    Location,
    TimeOfDay,
    ConversationIntent,
    Faction,
    RelationshipLevel,
    WorldState,
    create_default_world,
    create_farmer_spec,
    create_guard_spec,
    create_rebel_spec,
)

from prompted_town.ai import (
    # Backend
    MockLLMBackend,
    Message,
    create_backend,
    # Prompt Builder
    ConversationContext,
    build_conversation_prompts,
    select_speaking_style,
    create_context_from_world,
    # Conversation
    ConversationEngine,
    ConversationState,
    run_conversation,
    print_conversation,
    # Outcome Parser
    RuleBasedOutcomeParser,
    parse_conversation_outcome,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

def create_test_agents():
    """Create test agent specifications."""
    return {
        "farmer_01": create_farmer_spec("farmer_01", "Old Tom"),
        "farmer_02": create_farmer_spec("farmer_02", "Young Beth"),
        "guard_01": create_guard_spec("guard_01", "Captain Stern"),
        "rebel_01": create_rebel_spec("rebel_01", "Silent Mara"),
    }


def create_mock_backend_with_responses():
    """Create a mock backend with scenario-specific responses."""
    backend = MockLLMBackend()

    # Clear defaults and add test-specific responses
    backend.clear_responses()

    # Casual conversation responses (avoid ending phrases like "good day")
    backend.add_response("start the conversation", "Well met, friend. The weather's been fair, hasn't it?")
    backend.add_response("weather", "Aye, though my back aches from the work regardless.")
    backend.add_response("work", "The quota weighs heavy this season. Same for you?")
    backend.add_response("quota", "Same as always. The lord takes more than his share.")
    backend.add_response("lord", "Careful talk, friend. But you're not wrong.")

    # Recruitment scenario responses
    backend.add_response("alternative viewpoints", "What do you mean by that?")
    backend.add_response("what if", "I'm listening... but we should be careful who hears.")
    backend.add_response("together", "Perhaps you're right. Perhaps it is time for change.")
    backend.add_response("change", "Count me in. What do we do next?")

    # Guard interaction responses
    backend.add_response("halt", "Yes sir, just heading home from the fields.")
    backend.add_response("papers", "Here they are, sir. Everything is in order.")
    backend.add_response("suspicious", "I don't know anything, sir. I'm just a farmer.")

    # Ending responses
    backend.add_response("goodbye", "Take care. Until next time.")
    backend.add_response("must go", "Of course. Farewell, friend.")

    # Set a default that doesn't trigger endings
    backend.default_response = "Hmm, that's an interesting thought. Tell me more."

    return backend


# =============================================================================
# BACKEND TESTS
# =============================================================================

def test_mock_backend_basic():
    """Test basic mock backend functionality."""
    backend = MockLLMBackend()

    response = backend.generate("Hello, how are you?")

    assert response.content is not None
    assert response.model == "mock"
    assert len(backend.call_history) == 1
    print(f"✓ Mock backend response: '{response.content[:50]}...'")


def test_mock_backend_canned_response():
    """Test that mock backend returns canned responses."""
    backend = MockLLMBackend()

    # "hello" should trigger a specific response
    response = backend.generate("hello there friend")

    assert "Hello" in response.content or "hello" in response.content.lower()
    print(f"✓ Canned response triggered: '{response.content}'")


def test_mock_backend_with_history():
    """Test mock backend with conversation history."""
    backend = MockLLMBackend()

    messages = [
        Message(role="system", content="You are a farmer."),
        Message(role="user", content="How has the weather been lately?"),
    ]

    response = backend.generate_with_history(messages)

    assert response.content is not None
    assert "weather" in response.content.lower() or "harsh" in response.content.lower()
    print(f"✓ History-based response: '{response.content}'")


def test_create_backend_factory():
    """Test backend factory function."""
    mock_backend = create_backend("mock")
    assert isinstance(mock_backend, MockLLMBackend)
    print("✓ Backend factory creates mock backend")


# =============================================================================
# PROMPT BUILDER TESTS
# =============================================================================

def test_build_conversation_prompts():
    """Test prompt building for conversations."""
    specs = create_test_agents()

    context = ConversationContext(
        speaker_id="rebel_01",
        listener_id="farmer_01",
        speaker_spec=specs["rebel_01"],
        listener_spec=specs["farmer_01"],
        location=Location.TAVERN,
        time_of_day=TimeOfDay.NIGHT,
        intent=ConversationIntent.RECRUIT,
    )

    system_prompt, user_prompt = build_conversation_prompts(context, for_speaker=True)

    # System prompt should include personality info
    assert "Mara" in system_prompt  # Rebel's name
    assert "rebel" in system_prompt.lower() or "resistance" in system_prompt.lower()

    # User prompt should include situation
    assert "tavern" in user_prompt.lower()
    assert "night" in user_prompt.lower()

    print("✓ Prompts built correctly")
    print(f"  System prompt length: {len(system_prompt)} chars")
    print(f"  User prompt length: {len(user_prompt)} chars")


def test_speaking_style_selection():
    """Test that speaking styles change based on context."""
    specs = create_test_agents()

    # Rebel talking to farmer
    context_farmer = ConversationContext(
        speaker_id="rebel_01",
        listener_id="farmer_01",
        speaker_spec=specs["rebel_01"],
        listener_spec=specs["farmer_01"],
        location=Location.TAVERN,
        time_of_day=TimeOfDay.NIGHT,
        intent=ConversationIntent.RECRUIT,
    )

    style_farmer = select_speaking_style(
        specs["rebel_01"], specs["farmer_01"], context_farmer
    )

    # Rebel talking to guard
    context_guard = ConversationContext(
        speaker_id="rebel_01",
        listener_id="guard_01",
        speaker_spec=specs["rebel_01"],
        listener_spec=specs["guard_01"],
        location=Location.GATE,
        time_of_day=TimeOfDay.DAY,
        intent=ConversationIntent.CASUAL,
    )

    style_guard = select_speaking_style(
        specs["rebel_01"], specs["guard_01"], context_guard
    )

    # Styles should be different
    assert style_farmer != style_guard
    print(f"✓ Style to farmer: {style_farmer[:50]}...")
    print(f"✓ Style to guard: {style_guard[:50]}...")


def test_create_context_from_world():
    """Test creating context from world state."""
    specs = create_test_agents()
    world = create_default_world(agent_ids=list(specs.keys()))

    # Set up positions
    world.agents["rebel_01"].location = Location.TAVERN
    world.agents["farmer_01"].location = Location.TAVERN

    context = create_context_from_world(
        world,
        specs["rebel_01"],
        specs["farmer_01"],
        intent=ConversationIntent.RECRUIT,
    )

    assert context.location == Location.TAVERN
    assert context.speaker_id == "rebel_01"
    assert context.listener_id == "farmer_01"
    print("✓ Context created from world state")


# =============================================================================
# CONVERSATION ENGINE TESTS
# =============================================================================

def test_conversation_start():
    """Test starting a conversation."""
    specs = create_test_agents()
    backend = create_mock_backend_with_responses()

    context = ConversationContext(
        speaker_id="farmer_01",
        listener_id="farmer_02",
        speaker_spec=specs["farmer_01"],
        listener_spec=specs["farmer_02"],
        location=Location.FARM,
        time_of_day=TimeOfDay.DAY,
        intent=ConversationIntent.CASUAL,
    )

    engine = ConversationEngine(backend)
    state = engine.start_conversation(context)

    assert state.turn_count == 0
    assert state.is_initiator_turn == True
    print("✓ Conversation started correctly")


def test_conversation_single_turn():
    """Test running a single turn."""
    specs = create_test_agents()
    backend = create_mock_backend_with_responses()

    context = ConversationContext(
        speaker_id="farmer_01",
        listener_id="farmer_02",
        speaker_spec=specs["farmer_01"],
        listener_spec=specs["farmer_02"],
        location=Location.FARM,
        time_of_day=TimeOfDay.DAY,
        intent=ConversationIntent.CASUAL,
    )

    engine = ConversationEngine(backend)
    state = engine.start_conversation(context)
    state = engine.run_turn(state)

    assert state.turn_count == 1
    assert state.turns[0].speaker_id == "farmer_01"
    assert len(state.turns[0].text) > 0
    print(f"✓ First turn: '{state.turns[0].text}'")


def test_conversation_full():
    """Test running a complete conversation."""
    specs = create_test_agents()
    backend = create_mock_backend_with_responses()

    state = run_conversation(
        backend=backend,
        initiator_spec=specs["farmer_01"],
        target_spec=specs["farmer_02"],
        location=Location.TAVERN,
        time_of_day=TimeOfDay.DUSK,
        intent=ConversationIntent.CASUAL,
        max_turns=4,
    )

    assert state.turn_count >= 2  # At least 2 turns
    assert state.turn_count <= 4  # No more than max
    print(f"✓ Full conversation: {state.turn_count} turns")

    # Print transcript
    print("\n--- Transcript ---")
    for turn in state.turns:
        print(f"[{turn.speaker_name}]: {turn.text}")


def test_conversation_recruitment():
    """Test a recruitment conversation."""
    specs = create_test_agents()
    backend = create_mock_backend_with_responses()

    # Add recruitment-specific response
    backend.add_response("grievances", "The taxes are crushing us. Something must change.")

    state = run_conversation(
        backend=backend,
        initiator_spec=specs["rebel_01"],
        target_spec=specs["farmer_01"],
        location=Location.TAVERN,
        time_of_day=TimeOfDay.NIGHT,
        intent=ConversationIntent.RECRUIT,
        max_turns=6,
    )

    print(f"\n✓ Recruitment conversation: {state.turn_count} turns")
    print(f"  Intent: {state.context.intent.value}")
    print(f"  End reason: {state.end_reason}")


# =============================================================================
# OUTCOME PARSER TESTS
# =============================================================================

def test_rule_based_parser():
    """Test the rule-based outcome parser."""
    specs = create_test_agents()
    backend = create_mock_backend_with_responses()

    state = run_conversation(
        backend=backend,
        initiator_spec=specs["farmer_01"],
        target_spec=specs["farmer_02"],
        location=Location.FARM,
        time_of_day=TimeOfDay.DAY,
        intent=ConversationIntent.CASUAL,
        max_turns=4,
    )

    outcome = parse_conversation_outcome(state, use_llm=False)

    assert outcome.initiator_id == "farmer_01"
    assert outcome.target_id == "farmer_02"
    assert outcome.turn_count == state.turn_count
    print(f"✓ Parsed outcome: {outcome.get_summary()}")


def test_parser_trust_calculation():
    """Test that parser calculates trust changes."""
    specs = create_test_agents()
    backend = MockLLMBackend()

    # Force positive conversation
    backend.clear_responses()
    backend.add_response("", "I agree with you, friend. This is wonderful.")

    state = run_conversation(
        backend=backend,
        initiator_spec=specs["farmer_01"],
        target_spec=specs["farmer_02"],
        location=Location.TAVERN,
        time_of_day=TimeOfDay.DUSK,
        intent=ConversationIntent.CASUAL,
        max_turns=4,
    )

    outcome = parse_conversation_outcome(state)

    # Positive conversation should increase trust
    assert outcome.initiator_trust_delta >= 0
    assert outcome.target_trust_delta >= 0
    print(f"✓ Trust deltas: initiator={outcome.initiator_trust_delta:.3f}, target={outcome.target_trust_delta:.3f}")


def test_parser_recruitment_detection():
    """Test that parser detects recruitment outcomes."""
    specs = create_test_agents()
    backend = MockLLMBackend()

    # Set up for successful recruitment
    backend.clear_responses()
    backend.add_response("", "Count me in. I'll join you. Together we can change things.")

    context = ConversationContext(
        speaker_id="rebel_01",
        listener_id="farmer_01",
        speaker_spec=specs["rebel_01"],
        listener_spec=specs["farmer_01"],
        location=Location.TAVERN,
        time_of_day=TimeOfDay.NIGHT,
        intent=ConversationIntent.RECRUIT,
    )

    engine = ConversationEngine(backend)
    state = engine.run_conversation(context, max_turns=4)

    outcome = parse_conversation_outcome(state)

    assert outcome.recruitment_attempted == True
    print(f"✓ Recruitment detected: attempted={outcome.recruitment_attempted}, success={outcome.recruitment_successful}")


def test_parser_suspicion_detection():
    """Test that parser detects suspicious content."""
    specs = create_test_agents()
    backend = MockLLMBackend()

    # Set up suspicious conversation
    backend.clear_responses()
    backend.add_response("", "We must rebel against the lord. The uprising is coming.")

    state = run_conversation(
        backend=backend,
        initiator_spec=specs["rebel_01"],
        target_spec=specs["farmer_01"],
        location=Location.MARKET,  # Public place
        time_of_day=TimeOfDay.DAY,
        intent=ConversationIntent.RECRUIT,
        max_turns=4,
    )

    outcome = parse_conversation_outcome(state)

    assert outcome.suspicion_raised == True
    assert outcome.suspicion_amount > 0
    print(f"✓ Suspicion detected: raised={outcome.suspicion_raised}, amount={outcome.suspicion_amount:.3f}")


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

def test_full_conversation_to_outcome():
    """Test complete flow: context → conversation → outcome → world state."""
    specs = create_test_agents()
    world = create_default_world(agent_ids=list(specs.keys()))
    backend = create_mock_backend_with_responses()

    # Set up positions
    world.agents["rebel_01"].location = Location.TAVERN
    world.agents["farmer_01"].location = Location.TAVERN
    world.time_of_day = TimeOfDay.NIGHT

    # Create context from world
    context = create_context_from_world(
        world,
        specs["rebel_01"],
        specs["farmer_01"],
        intent=ConversationIntent.RECRUIT,
    )

    # Run conversation
    engine = ConversationEngine(backend)
    state = engine.run_conversation(context, max_turns=6)

    # Parse outcome
    outcome = parse_conversation_outcome(state)

    # Apply to world (using simulation module)
    from prompted_town.simulation import apply_conversation_outcome
    events = apply_conversation_outcome(world, outcome, specs)

    print("\n✓ Full integration test passed")
    print(f"  Turns: {state.turn_count}")
    print(f"  Outcome: {outcome.get_summary()}")
    print(f"  Events generated: {len(events)}")


def demo_conversation():
    """Run a demonstration conversation."""
    specs = create_test_agents()
    backend = create_mock_backend_with_responses()

    print("\n" + "=" * 60)
    print("DEMO: Rebel recruitment attempt")
    print("=" * 60)

    state = run_conversation(
        backend=backend,
        initiator_spec=specs["rebel_01"],
        target_spec=specs["farmer_01"],
        location=Location.TAVERN,
        time_of_day=TimeOfDay.NIGHT,
        intent=ConversationIntent.RECRUIT,
        trust_initiator_to_target=0.3,
        trust_target_to_initiator=0.2,
        max_turns=6,
    )

    print_conversation(state)

    outcome = parse_conversation_outcome(state)
    print(f"Outcome Summary: {outcome.get_summary()}")
    print(f"Trust Changes: initiator={outcome.initiator_trust_delta:+.2f}, target={outcome.target_trust_delta:+.2f}")
    print(f"Recruitment: attempted={outcome.recruitment_attempted}, success={outcome.recruitment_successful}")
    print(f"Suspicion: raised={outcome.suspicion_raised}, amount={outcome.suspicion_amount:.2f}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("AI MODULE TESTS")
    print("=" * 50 + "\n")

    # Backend tests
    print("--- Backend Tests ---")
    test_mock_backend_basic()
    test_mock_backend_canned_response()
    test_mock_backend_with_history()
    test_create_backend_factory()

    # Prompt builder tests
    print("\n--- Prompt Builder Tests ---")
    test_build_conversation_prompts()
    test_speaking_style_selection()
    test_create_context_from_world()

    # Conversation engine tests
    print("\n--- Conversation Engine Tests ---")
    test_conversation_start()
    test_conversation_single_turn()
    test_conversation_full()
    test_conversation_recruitment()

    # Outcome parser tests
    print("\n--- Outcome Parser Tests ---")
    test_rule_based_parser()
    test_parser_trust_calculation()
    test_parser_recruitment_detection()
    test_parser_suspicion_detection()

    # Integration tests
    print("\n--- Integration Tests ---")
    test_full_conversation_to_outcome()

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED!")
    print("=" * 50)

    # Run demo
    demo_conversation()
