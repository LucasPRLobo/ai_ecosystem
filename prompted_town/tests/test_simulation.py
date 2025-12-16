"""
Tests for the simulation module.

Run with: python -m pytest prompted_town/tests/test_simulation.py -v
Or directly: python -m prompted_town.tests.test_simulation
"""

import sys
from pathlib import Path

# Add parent to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from prompted_town.core import (
    Location,
    TimeOfDay,
    ActionType,
    AgentRole,
    Faction,
    ConversationIntent,
    ResourceType,
    WorldState,
    AgentState,
    create_default_world,
    create_farmer_spec,
    create_guard_spec,
    create_rebel_spec,
)

from prompted_town.simulation import (
    Action,
    move_action,
    work_action,
    rest_action,
    deliver_quota_action,
    conversation_action,
    wait_action,
    validate_action,
    get_valid_actions,
    step_world,
    enforce_laws,
    ConversationOutcome,
    apply_conversation_outcome,
    ENERGY_COST_MOVE,
    ENERGY_COST_WORK,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

def create_test_world() -> tuple[WorldState, dict]:
    """Create a test world with known agents."""
    agent_ids = ["farmer_01", "farmer_02", "guard_01", "rebel_01"]
    world = create_default_world(agent_ids=agent_ids, random_seed=42)

    specs = {
        "farmer_01": create_farmer_spec("farmer_01", "Tom"),
        "farmer_02": create_farmer_spec("farmer_02", "Beth"),
        "guard_01": create_guard_spec("guard_01", "Stern"),
        "rebel_01": create_rebel_spec("rebel_01", "Mara"),
    }

    return world, specs


# =============================================================================
# ACTION TESTS
# =============================================================================

def test_action_creation():
    """Test action factory functions."""
    move = move_action("agent_01", Location.MARKET)
    assert move.action_type == ActionType.MOVE
    assert move.target_location == Location.MARKET
    print("✓ Move action creation works")

    work = work_action("agent_01")
    assert work.action_type == ActionType.WORK
    print("✓ Work action creation works")

    rest = rest_action("agent_01")
    assert rest.action_type == ActionType.REST
    print("✓ Rest action creation works")

    conv = conversation_action("agent_01", "agent_02", ConversationIntent.RECRUIT)
    assert conv.action_type == ActionType.INITIATE_CONVERSATION
    assert conv.target_agent == "agent_02"
    assert conv.conversation_intent == ConversationIntent.RECRUIT
    print("✓ Conversation action creation works")


def test_action_validation():
    """Test action validation logic."""
    world, specs = create_test_world()

    # Valid move
    move = move_action("farmer_01", Location.FARM)
    validation = validate_action(world, move)
    assert validation.is_valid, f"Move should be valid: {validation.reason}"
    print("✓ Valid move action passes validation")

    # Invalid move - already at location
    world.agents["farmer_01"].location = Location.FARM
    move_same = move_action("farmer_01", Location.FARM)
    validation = validate_action(world, move_same)
    assert not validation.is_valid
    assert "Already at" in validation.reason
    print("✓ Move to same location correctly rejected")

    # Invalid conversation - different location
    world.agents["farmer_01"].location = Location.HOME
    world.agents["farmer_02"].location = Location.MARKET
    conv = conversation_action("farmer_01", "farmer_02")
    validation = validate_action(world, conv)
    assert not validation.is_valid
    assert "not at same location" in validation.reason
    print("✓ Conversation at different location correctly rejected")

    # Valid conversation - same location
    world.agents["farmer_02"].location = Location.HOME
    conv = conversation_action("farmer_01", "farmer_02")
    validation = validate_action(world, conv)
    assert validation.is_valid
    print("✓ Conversation at same location passes validation")


def test_get_valid_actions():
    """Test action space generation."""
    world, specs = create_test_world()

    # Put farmer at home
    world.agents["farmer_01"].location = Location.HOME

    actions = get_valid_actions(world, "farmer_01", include_conversation_targets=False)

    # Should have WAIT, REST, and MOVE to all other locations
    action_types = {a.action_type for a in actions}
    assert ActionType.WAIT in action_types
    assert ActionType.REST in action_types
    assert ActionType.MOVE in action_types

    # Should NOT have WORK (can't work at home)
    assert ActionType.WORK not in action_types

    print(f"✓ Got {len(actions)} valid actions at home")

    # Now at farm - should be able to work
    world.agents["farmer_01"].location = Location.FARM
    actions = get_valid_actions(world, "farmer_01", include_conversation_targets=False)
    action_types = {a.action_type for a in actions}
    assert ActionType.WORK in action_types
    print("✓ Work action available at farm")


# =============================================================================
# SIMULATION STEP TESTS
# =============================================================================

def test_step_movement():
    """Test that movement works correctly."""
    world, specs = create_test_world()
    world.agents["farmer_01"].location = Location.HOME
    initial_energy = world.agents["farmer_01"].energy

    actions = {"farmer_01": move_action("farmer_01", Location.FARM)}

    result = step_world(world, actions, random_seed=42)

    assert result.new_state.agents["farmer_01"].location == Location.FARM
    assert result.new_state.agents["farmer_01"].energy == initial_energy - ENERGY_COST_MOVE
    print("✓ Movement updates location and costs energy")


def test_step_work():
    """Test that work produces resources."""
    world, specs = create_test_world()
    world.agents["farmer_01"].location = Location.FARM
    initial_grain = world.agents["farmer_01"].get_resource(ResourceType.GRAIN)

    actions = {"farmer_01": work_action("farmer_01")}

    result = step_world(world, actions, random_seed=42)

    new_grain = result.new_state.agents["farmer_01"].get_resource(ResourceType.GRAIN)
    assert new_grain > initial_grain, "Should have produced grain"
    print(f"✓ Work produced {new_grain - initial_grain} grain")


def test_step_rest():
    """Test that rest recovers energy."""
    world, specs = create_test_world()
    world.agents["farmer_01"].location = Location.HOME
    world.agents["farmer_01"].energy = 50

    actions = {"farmer_01": rest_action("farmer_01")}

    result = step_world(world, actions, random_seed=42)

    assert result.new_state.agents["farmer_01"].energy > 50
    print(f"✓ Rest recovered energy to {result.new_state.agents['farmer_01'].energy}")


def test_step_time_advances():
    """Test that time advances each step."""
    world, specs = create_test_world()
    initial_tick = world.tick
    initial_time = world.time_of_day

    result = step_world(world, {}, random_seed=42)

    assert result.new_state.tick == initial_tick + 1
    assert result.new_state.time_of_day != initial_time
    print(f"✓ Time advanced from {initial_time.value} to {result.new_state.time_of_day.value}")


def test_step_determinism():
    """Test that simulation is deterministic with same seed."""
    world1, specs = create_test_world()
    world2 = world1.clone()

    # Complex set of actions
    actions = {
        "farmer_01": work_action("farmer_01"),
        "farmer_02": move_action("farmer_02", Location.TAVERN),
        "guard_01": move_action("guard_01", Location.GATE),
        "rebel_01": rest_action("rebel_01"),
    }
    # Set locations for work to be valid
    world1.agents["farmer_01"].location = Location.FARM
    world2.agents["farmer_01"].location = Location.FARM

    result1 = step_world(world1, actions, random_seed=42)
    result2 = step_world(world2, actions, random_seed=42)

    # Compare states
    for agent_id in world1.agents.keys():
        state1 = result1.new_state.agents[agent_id]
        state2 = result2.new_state.agents[agent_id]
        assert state1.location == state2.location, f"Location mismatch for {agent_id}"
        assert state1.energy == state2.energy, f"Energy mismatch for {agent_id}"
        assert state1.gold == state2.gold, f"Gold mismatch for {agent_id}"

    print("✓ Same seed produces identical results (deterministic)")


def test_step_conversation_flagged():
    """Test that conversations are identified for AI resolution."""
    world, specs = create_test_world()

    # Put agents at same location
    world.agents["rebel_01"].location = Location.TAVERN
    world.agents["farmer_01"].location = Location.TAVERN

    actions = {
        "rebel_01": conversation_action("rebel_01", "farmer_01", ConversationIntent.RECRUIT)
    }

    result = step_world(world, actions, random_seed=42)

    assert len(result.pending_conversations) == 1
    conv = result.pending_conversations[0]
    assert conv.initiator_id == "rebel_01"
    assert conv.target_id == "farmer_01"
    assert conv.intent == ConversationIntent.RECRUIT
    print("✓ Conversation correctly flagged for AI resolution")


# =============================================================================
# LAW ENFORCEMENT TESTS
# =============================================================================

def test_curfew_detection():
    """Test that curfew violations are detected."""
    world, specs = create_test_world()

    # Set to night (curfew active)
    world.time_of_day = TimeOfDay.NIGHT

    # Put farmer in market (not allowed during curfew)
    world.agents["farmer_01"].location = Location.MARKET

    # Put guard at market (to witness)
    world.agents["guard_01"].location = Location.MARKET

    result = enforce_laws(world, specs)

    # Should have a curfew violation
    curfew_violations = [v for v in result.violations if v.violation_type == "curfew"]
    assert len(curfew_violations) == 1
    assert curfew_violations[0].agent_id == "farmer_01"
    assert "guard_01" in curfew_violations[0].witnessed_by
    print("✓ Curfew violation detected when guard present")


def test_curfew_no_guard():
    """Test that curfew violations need a guard witness."""
    world, specs = create_test_world()

    # Set to night
    world.time_of_day = TimeOfDay.NIGHT

    # Put farmer in market
    world.agents["farmer_01"].location = Location.MARKET

    # Guard is elsewhere
    world.agents["guard_01"].location = Location.GATE

    result = enforce_laws(world, specs)

    curfew_violations = [v for v in result.violations if v.violation_type == "curfew"]
    assert len(curfew_violations) == 0
    print("✓ No curfew violation without guard witness")


def test_suspicion_increases():
    """Test that violations increase suspicion."""
    world, specs = create_test_world()
    initial_suspicion = world.agents["farmer_01"].suspicion_level

    # Set to night
    world.time_of_day = TimeOfDay.NIGHT
    world.agents["farmer_01"].location = Location.MARKET
    world.agents["guard_01"].location = Location.MARKET

    result = enforce_laws(world, specs)

    assert world.agents["farmer_01"].suspicion_level > initial_suspicion
    print(f"✓ Suspicion increased from {initial_suspicion} to {world.agents['farmer_01'].suspicion_level}")


# =============================================================================
# SOCIAL OUTCOME TESTS
# =============================================================================

def test_conversation_outcome_application():
    """Test that conversation outcomes affect world state."""
    world, specs = create_test_world()

    # Create a recruitment outcome
    outcome = ConversationOutcome(
        initiator_id="rebel_01",
        target_id="farmer_01",
        turn_count=5,
        initiator_trust_delta=0.1,
        target_trust_delta=0.2,
        recruitment_attempted=True,
        recruitment_successful=True,
        was_positive=True,
    )

    events = apply_conversation_outcome(world, outcome, specs)

    # Check farmer is now recruited
    assert world.agents["farmer_01"].is_recruited
    print("✓ Recruitment outcome applied correctly")

    # Check trust changed
    trust = world.get_trust("farmer_01", "rebel_01")
    assert trust == 0.2  # Started at 0, added 0.2
    print("✓ Trust changes applied correctly")

    # Check they know each other are rebels
    assert "rebel_01" in world.agents["farmer_01"].known_rebels
    assert "farmer_01" in world.agents["rebel_01"].known_rebels
    print("✓ Rebel knowledge shared correctly")


def test_conversation_outcome_suspicion():
    """Test that suspicious conversations raise suspicion."""
    world, specs = create_test_world()
    initial_suspicion = world.agents["rebel_01"].suspicion_level

    outcome = ConversationOutcome(
        initiator_id="rebel_01",
        target_id="farmer_01",
        suspicion_raised=True,
        suspicion_amount=0.3,
        was_positive=False,
    )

    apply_conversation_outcome(world, outcome, specs)

    assert world.agents["rebel_01"].suspicion_level > initial_suspicion
    print(f"✓ Suspicion raised from {initial_suspicion} to {world.agents['rebel_01'].suspicion_level}")


# =============================================================================
# INTEGRATION TEST
# =============================================================================

def test_full_simulation_loop():
    """Test a complete simulation loop with multiple ticks."""
    world, specs = create_test_world()

    # Put agents in starting positions
    world.agents["farmer_01"].location = Location.FARM
    world.agents["farmer_02"].location = Location.HOME
    world.agents["guard_01"].location = Location.GATE
    world.agents["rebel_01"].location = Location.TAVERN

    print("\n--- Full Simulation Loop Test ---")
    print(f"Initial state: Day {world.day}, {world.time_of_day.value}")

    for tick in range(8):  # Two full days
        # Decide actions (simple policy)
        actions = {}

        for agent_id, agent_state in world.agents.items():
            if agent_state.is_exhausted():
                actions[agent_id] = rest_action(agent_id)
            elif agent_state.location == Location.FARM:
                actions[agent_id] = work_action(agent_id)
            elif agent_state.location == Location.HOME:
                actions[agent_id] = move_action(agent_id, Location.FARM)
            else:
                actions[agent_id] = wait_action(agent_id)

        # Step simulation
        result = step_world(world, actions, random_seed=42 + tick)
        world = result.new_state

        # Enforce laws
        law_result = enforce_laws(world, specs)

        if tick % 2 == 1:  # Every other tick
            print(f"  Tick {world.tick}: {world.time_of_day.value}, "
                  f"farmer_01 energy={world.agents['farmer_01'].energy}, "
                  f"grain={world.agents['farmer_01'].get_resource(ResourceType.GRAIN)}")

    print(f"Final state: Day {world.day}, {world.time_of_day.value}")
    print(f"  farmer_01: grain={world.agents['farmer_01'].get_resource(ResourceType.GRAIN)}, "
          f"gold={world.agents['farmer_01'].gold}")
    print("✓ Full simulation loop completed successfully")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("SIMULATION MODULE TESTS")
    print("=" * 50 + "\n")

    # Action tests
    print("--- Action Tests ---")
    test_action_creation()
    test_action_validation()
    test_get_valid_actions()

    # Simulation step tests
    print("\n--- Simulation Step Tests ---")
    test_step_movement()
    test_step_work()
    test_step_rest()
    test_step_time_advances()
    test_step_determinism()
    test_step_conversation_flagged()

    # Law enforcement tests
    print("\n--- Law Enforcement Tests ---")
    test_curfew_detection()
    test_curfew_no_guard()
    test_suspicion_increases()

    # Social outcome tests
    print("\n--- Social Outcome Tests ---")
    test_conversation_outcome_application()
    test_conversation_outcome_suspicion()

    # Integration test
    test_full_simulation_loop()

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED!")
    print("=" * 50)
