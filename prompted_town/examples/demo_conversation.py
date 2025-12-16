#!/usr/bin/env python3
"""
Interactive demo for Prompted Town AI conversations.

Run with:
    python -m prompted_town.examples.demo_conversation

Options:
    --backend mock     Use mock backend (default, no API needed)
    --backend openai   Use OpenAI (requires OPENAI_API_KEY)
    --backend anthropic Use Anthropic (requires ANTHROPIC_API_KEY)
"""

import sys
import argparse
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from prompted_town.core import (
    Location,
    TimeOfDay,
    ConversationIntent,
    create_farmer_spec,
    create_guard_spec,
    create_rebel_spec,
)

from prompted_town.ai import (
    create_backend,
    ConversationEngine,
    ConversationContext,
    run_conversation,
    print_conversation,
    parse_conversation_outcome,
)


def create_agents():
    """Create the cast of characters."""
    return {
        "tom": create_farmer_spec("farmer_tom", "Old Tom"),
        "beth": create_farmer_spec("farmer_beth", "Young Beth"),
        "stern": create_guard_spec("guard_stern", "Captain Stern"),
        "mara": create_rebel_spec("rebel_mara", "Silent Mara"),
    }


def run_scenario(backend, scenario: str):
    """Run a predefined scenario."""
    agents = create_agents()

    scenarios = {
        "casual": {
            "initiator": agents["tom"],
            "target": agents["beth"],
            "location": Location.FARM,
            "time": TimeOfDay.DAY,
            "intent": ConversationIntent.CASUAL,
            "description": "Two farmers chat during work",
        },
        "recruit": {
            "initiator": agents["mara"],
            "target": agents["tom"],
            "location": Location.TAVERN,
            "time": TimeOfDay.NIGHT,
            "intent": ConversationIntent.RECRUIT,
            "description": "Rebel tries to recruit a farmer",
        },
        "interrogate": {
            "initiator": agents["stern"],
            "target": agents["mara"],
            "location": Location.GATE,
            "time": TimeOfDay.DUSK,
            "intent": ConversationIntent.INTERROGATE,
            "description": "Guard interrogates a suspicious person",
        },
        "tavern": {
            "initiator": agents["mara"],
            "target": agents["beth"],
            "location": Location.TAVERN,
            "time": TimeOfDay.NIGHT,
            "intent": ConversationIntent.GATHER_INFO,
            "description": "Rebel gathers information at the tavern",
        },
    }

    if scenario not in scenarios:
        print(f"Unknown scenario: {scenario}")
        print(f"Available: {', '.join(scenarios.keys())}")
        return

    s = scenarios[scenario]
    print(f"\n{'='*60}")
    print(f"SCENARIO: {scenario.upper()}")
    print(f"{s['description']}")
    print(f"{'='*60}\n")

    state = run_conversation(
        backend=backend,
        initiator_spec=s["initiator"],
        target_spec=s["target"],
        location=s["location"],
        time_of_day=s["time"],
        intent=s["intent"],
        max_turns=8,
    )

    print_conversation(state)

    # Parse and show outcome
    outcome = parse_conversation_outcome(state)
    print("\n--- OUTCOME ANALYSIS ---")
    print(f"Summary: {outcome.get_summary()}")
    print(f"Was positive: {outcome.was_positive}")
    print(f"Trust changes: initiator={outcome.initiator_trust_delta:+.2f}, target={outcome.target_trust_delta:+.2f}")

    if outcome.recruitment_attempted:
        print(f"Recruitment: {'SUCCESS' if outcome.recruitment_successful else 'FAILED'}")

    if outcome.suspicion_raised:
        print(f"Suspicion raised: {outcome.suspicion_amount:.2f}")

    return state, outcome


def interactive_mode(backend):
    """Run interactive conversation mode."""
    agents = create_agents()

    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("\nAvailable agents:")
    for key, spec in agents.items():
        print(f"  {key}: {spec.name} ({spec.public_role.value})")

    print("\nAvailable locations: farm, market, tavern, home, gate, town_square")
    print("Available times: dawn, day, dusk, night")
    print("Available intents: casual, recruit, interrogate, trade, conspire, gather_info")

    while True:
        print("\n" + "-"*40)
        try:
            initiator_key = input("Initiator (or 'quit'): ").strip().lower()
            if initiator_key == 'quit':
                break
            if initiator_key not in agents:
                print(f"Unknown agent. Choose from: {', '.join(agents.keys())}")
                continue

            target_key = input("Target: ").strip().lower()
            if target_key not in agents or target_key == initiator_key:
                print(f"Invalid target. Choose different agent from: {', '.join(agents.keys())}")
                continue

            location_str = input("Location [tavern]: ").strip().lower() or "tavern"
            time_str = input("Time [night]: ").strip().lower() or "night"
            intent_str = input("Intent [casual]: ").strip().lower() or "casual"

            # Map strings to enums
            location_map = {loc.value: loc for loc in Location}
            time_map = {t.value: t for t in TimeOfDay}
            intent_map = {i.value: i for i in ConversationIntent}

            location = location_map.get(location_str, Location.TAVERN)
            time_of_day = time_map.get(time_str, TimeOfDay.NIGHT)
            intent = intent_map.get(intent_str, ConversationIntent.CASUAL)

            print(f"\nRunning conversation: {agents[initiator_key].name} → {agents[target_key].name}")
            print(f"Location: {location.value}, Time: {time_of_day.value}, Intent: {intent.value}")

            state = run_conversation(
                backend=backend,
                initiator_spec=agents[initiator_key],
                target_spec=agents[target_key],
                location=location,
                time_of_day=time_of_day,
                intent=intent,
                max_turns=8,
            )

            print_conversation(state)

            outcome = parse_conversation_outcome(state)
            print(f"\nOutcome: {outcome.get_summary()}")

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Prompted Town Conversation Demo")
    parser.add_argument(
        "--backend",
        choices=["mock", "openai", "anthropic"],
        default="mock",
        help="LLM backend to use"
    )
    parser.add_argument(
        "--scenario",
        choices=["casual", "recruit", "interrogate", "tavern", "all"],
        help="Run a specific scenario"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--model",
        help="Model name (e.g., gpt-4o-mini, claude-3-5-haiku-latest)"
    )

    args = parser.parse_args()

    # Create backend
    print(f"Using backend: {args.backend}")
    try:
        backend = create_backend(
            provider=args.backend,
            model=args.model,
            cache=True,  # Cache responses to save API costs
        )
    except ValueError as e:
        print(f"Error creating backend: {e}")
        print("For OpenAI: export OPENAI_API_KEY=your-key")
        print("For Anthropic: export ANTHROPIC_API_KEY=your-key")
        return

    if args.interactive:
        interactive_mode(backend)
    elif args.scenario:
        if args.scenario == "all":
            for scenario in ["casual", "recruit", "interrogate", "tavern"]:
                run_scenario(backend, scenario)
                print("\n" + "="*60 + "\n")
        else:
            run_scenario(backend, args.scenario)
    else:
        # Default: run all scenarios
        print("Running all demo scenarios...")
        print("(Use --interactive for custom conversations)")
        print("(Use --backend openai/anthropic for real AI)")
        for scenario in ["casual", "recruit"]:
            run_scenario(backend, scenario)


if __name__ == "__main__":
    main()
