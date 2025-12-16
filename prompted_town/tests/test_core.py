"""
Tests for the core module.

Run with: python -m pytest prompted_town/tests/test_core.py -v
Or directly: python prompted_town/tests/test_core.py
"""

import json
import sys
from pathlib import Path

# Add parent to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from prompted_town.core import (
    # Types
    Location,
    TimeOfDay,
    ActionType,
    AgentRole,
    Faction,
    TraitName,
    GoalType,
    next_time_of_day,
    trust_to_relationship_level,
    RelationshipLevel,
    # Agent specs
    AgentSpec,
    AIPersonality,
    RLProfile,
    Goal,
    create_farmer_spec,
    create_guard_spec,
    create_rebel_spec,
    # World state
    WorldState,
    AgentState,
    RelationshipState,
    LawConfig,
    create_default_world,
)


def test_time_cycle():
    """Test that time advances correctly."""
    assert next_time_of_day(TimeOfDay.DAWN) == TimeOfDay.DAY
    assert next_time_of_day(TimeOfDay.DAY) == TimeOfDay.DUSK
    assert next_time_of_day(TimeOfDay.DUSK) == TimeOfDay.NIGHT
    assert next_time_of_day(TimeOfDay.NIGHT) == TimeOfDay.DAWN
    print("✓ Time cycle works correctly")


def test_trust_to_relationship():
    """Test trust value to relationship level conversion."""
    assert trust_to_relationship_level(-0.8) == RelationshipLevel.HOSTILE
    assert trust_to_relationship_level(-0.4) == RelationshipLevel.SUSPICIOUS
    assert trust_to_relationship_level(0.0) == RelationshipLevel.STRANGER
    assert trust_to_relationship_level(0.3) == RelationshipLevel.ACQUAINTANCE
    assert trust_to_relationship_level(0.6) == RelationshipLevel.FRIENDLY
    assert trust_to_relationship_level(0.9) == RelationshipLevel.TRUSTED
    print("✓ Trust to relationship level conversion works")


def test_agent_spec_factories():
    """Test agent specification factory functions."""
    farmer = create_farmer_spec("farmer_01", "Old Tom")
    assert farmer.agent_id == "farmer_01"
    assert farmer.name == "Old Tom"
    assert farmer.public_role == AgentRole.FARMER
    assert farmer.true_faction == Faction.NEUTRAL
    assert TraitName.DILIGENCE in farmer.rl_profile.traits
    print(f"✓ Created farmer: {farmer.name} ({farmer.agent_id})")

    guard = create_guard_spec("guard_01", "Captain Stern")
    assert guard.public_role == AgentRole.GUARD
    assert guard.true_faction == Faction.LOYALIST
    assert guard.rl_profile.get_trait(TraitName.SUSPICION) == 0.7
    print(f"✓ Created guard: {guard.name} ({guard.agent_id})")

    rebel = create_rebel_spec("rebel_01", "Silent Mara", cover_role=AgentRole.FARMER)
    assert rebel.public_role == AgentRole.FARMER  # Cover
    assert rebel.true_faction == Faction.REBEL  # True allegiance
    assert GoalType.RECRUIT_REBELS in [g.goal_type for g in rebel.rl_profile.goals]
    print(f"✓ Created rebel: {rebel.name} ({rebel.agent_id})")


def test_world_state_creation():
    """Test world state creation and basic operations."""
    world = create_default_world(random_seed=42)

    assert world.tick == 0
    assert world.day == 0
    assert world.time_of_day == TimeOfDay.DAWN
    assert len(world.agents) == 7  # Default agent count
    assert "farmer_01" in world.agents
    print(f"✓ Created world with {len(world.agents)} agents")


def test_agent_state_operations():
    """Test agent state modifications."""
    agent = AgentState(agent_id="test_agent")

    # Test energy adjustment
    agent.adjust_energy(-30)
    assert agent.energy == 70
    agent.adjust_energy(-100)  # Should clamp to 0
    assert agent.energy == 0
    agent.adjust_energy(50)
    assert agent.energy == 50
    print("✓ Energy adjustment works correctly")

    # Test suspicion adjustment
    agent.adjust_suspicion(0.3)
    assert agent.suspicion_level == 0.3
    agent.adjust_suspicion(0.9)  # Should clamp to 1.0
    assert agent.suspicion_level == 1.0
    print("✓ Suspicion adjustment works correctly")

    # Test resource management
    from prompted_town.core import ResourceType
    agent.add_resource(ResourceType.GRAIN, 10)
    assert agent.get_resource(ResourceType.GRAIN) == 10
    agent.add_resource(ResourceType.GRAIN, -3)
    assert agent.get_resource(ResourceType.GRAIN) == 7
    print("✓ Resource management works correctly")


def test_relationship_state():
    """Test relationship tracking."""
    world = create_default_world()

    # Get/create relationship
    rel = world.get_relationship("farmer_01", "farmer_02")
    assert rel.agent_a_id == "farmer_01"  # Sorted order
    assert rel.agent_b_id == "farmer_02"
    assert rel.trust_a_to_b == 0.0

    # Modify trust
    rel.adjust_trust("farmer_01", "farmer_02", 0.3)
    assert rel.get_trust("farmer_01", "farmer_02") == 0.3
    assert rel.get_trust("farmer_02", "farmer_01") == 0.0  # Asymmetric

    # Check relationship level
    assert rel.get_relationship_level("farmer_01", "farmer_02") == RelationshipLevel.ACQUAINTANCE
    print("✓ Relationship tracking works correctly")


def test_time_advancement():
    """Test world time progression."""
    world = create_default_world()

    assert world.tick == 0
    assert world.time_of_day == TimeOfDay.DAWN

    events = world.advance_time()
    assert world.tick == 1
    assert world.time_of_day == TimeOfDay.DAY
    assert len(events) == 1  # TIME_ADVANCED

    # Advance through a full day
    for _ in range(3):  # DAY -> DUSK -> NIGHT -> DAWN
        events = world.advance_time()

    assert world.tick == 4
    assert world.time_of_day == TimeOfDay.DAWN
    assert world.day == 1  # Day rolled over
    print("✓ Time advancement works correctly")


def test_law_config():
    """Test law configuration."""
    laws = LawConfig()

    # Curfew should be active at night
    assert laws.is_curfew_active(TimeOfDay.NIGHT) == True
    assert laws.is_curfew_active(TimeOfDay.DAY) == False
    assert laws.is_curfew_active(TimeOfDay.DAWN) == False

    # Check allowed locations during curfew
    assert Location.HOME in laws.curfew_allowed_locations
    assert Location.TAVERN in laws.curfew_allowed_locations
    assert Location.MARKET not in laws.curfew_allowed_locations
    print("✓ Law configuration works correctly")


def test_serialization_roundtrip():
    """Test that world state survives JSON serialization."""
    # Create world with some modifications
    world = create_default_world(random_seed=42)
    world.agents["farmer_01"].gold = 50
    world.agents["farmer_01"].location = Location.MARKET
    world.agents["farmer_01"].suspicion_level = 0.3

    # Modify a relationship
    rel = world.get_relationship("farmer_01", "rebel_01")
    rel.adjust_trust("farmer_01", "rebel_01", 0.5)
    rel.a_knows_b_is_rebel = True

    # Advance time a bit
    for _ in range(5):
        world.advance_time()

    # Serialize
    json_str = world.to_json()
    print(f"✓ Serialized world state ({len(json_str)} bytes)")

    # Deserialize
    world2 = WorldState.from_json(json_str)

    # Verify
    assert world2.tick == world.tick
    assert world2.day == world.day
    assert world2.time_of_day == world.time_of_day
    assert world2.agents["farmer_01"].gold == 50
    assert world2.agents["farmer_01"].location == Location.MARKET
    assert world2.agents["farmer_01"].suspicion_level == 0.3

    rel2 = world2.get_relationship("farmer_01", "rebel_01")
    assert rel2.get_trust("farmer_01", "rebel_01") == 0.5
    assert rel2.a_knows_b_is_rebel == True

    print("✓ Serialization roundtrip successful")


def test_world_clone():
    """Test that world state cloning creates independent copy."""
    world1 = create_default_world()
    world1.agents["farmer_01"].gold = 100

    world2 = world1.clone()
    world2.agents["farmer_01"].gold = 200

    assert world1.agents["farmer_01"].gold == 100
    assert world2.agents["farmer_01"].gold == 200
    print("✓ World cloning creates independent copy")


def print_world_summary(world: WorldState):
    """Print a human-readable summary of the world state."""
    print("\n" + "=" * 60)
    print(f"WORLD STATE - Day {world.day}, {world.time_of_day.value.upper()}")
    print(f"Tick: {world.tick}")
    print("=" * 60)

    print("\nAGENTS:")
    for agent_id, state in world.agents.items():
        status = []
        if not state.is_alive:
            status.append("DEAD")
        if state.is_arrested:
            status.append("ARRESTED")
        if state.is_recruited:
            status.append("REBEL")
        status_str = f" [{', '.join(status)}]" if status else ""

        print(f"  {agent_id}: {state.location.value}, "
              f"gold={state.gold}, energy={state.energy}, "
              f"suspicion={state.suspicion_level:.2f}{status_str}")

    print("\nRELATIONSHIPS:")
    for key, rel in world.relationships.items():
        print(f"  {rel.agent_a_id} <-> {rel.agent_b_id}: "
              f"trust {rel.trust_a_to_b:.2f} / {rel.trust_b_to_a:.2f}, "
              f"interactions={rel.total_interactions}")

    print("=" * 60 + "\n")


def demo():
    """Run a demonstration of the core module."""
    print("\n" + "=" * 60)
    print("PROMPTED TOWN - Core Module Demo")
    print("=" * 60)

    # Create agent specs
    print("\nCreating agent specifications...")
    specs = {
        "farmer_01": create_farmer_spec("farmer_01", "Old Tom"),
        "farmer_02": create_farmer_spec("farmer_02", "Young Beth", faction=Faction.NEUTRAL),
        "guard_01": create_guard_spec("guard_01", "Captain Stern"),
        "rebel_01": create_rebel_spec("rebel_01", "Silent Mara"),
    }

    for spec in specs.values():
        print(f"  Created {spec.name} ({spec.public_role.value})")
        if spec.true_faction == Faction.REBEL:
            print(f"    → Secret: Actually a {spec.true_faction.value}!")

    # Create world state
    print("\nCreating world state...")
    world = create_default_world(
        agent_ids=list(specs.keys()),
        random_seed=42
    )

    # Show initial state
    print_world_summary(world)

    # Simulate some changes
    print("Simulating some activity...")

    # Farmer goes to market
    world.agents["farmer_01"].location = Location.MARKET
    world.agents["farmer_01"].gold += 5
    print("  farmer_01 went to market and earned 5 gold")

    # Rebel builds trust with farmer
    rel = world.get_relationship("rebel_01", "farmer_02")
    rel.adjust_trust("rebel_01", "farmer_02", 0.4)
    rel.adjust_trust("farmer_02", "rebel_01", 0.2)
    rel.record_interaction(world.tick, positive=True)
    print("  rebel_01 and farmer_02 had a positive interaction")

    # Guard gets suspicious of rebel
    world.agents["rebel_01"].suspicion_level = 0.25
    print("  guard_01 noticed something odd about rebel_01")

    # Advance time
    for _ in range(2):
        world.advance_time()
    print(f"  Time advanced to {world.time_of_day.value}")

    # Show final state
    print_world_summary(world)

    print("Demo complete!")


if __name__ == "__main__":
    # Run tests
    print("\nRunning tests...\n")
    test_time_cycle()
    test_trust_to_relationship()
    test_agent_spec_factories()
    test_world_state_creation()
    test_agent_state_operations()
    test_relationship_state()
    test_time_advancement()
    test_law_config()
    test_serialization_roundtrip()
    test_world_clone()

    print("\n" + "=" * 40)
    print("All tests passed!")
    print("=" * 40)

    # Run demo
    demo()
