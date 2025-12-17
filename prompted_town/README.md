# Prompted Town

A multi-agent reinforcement learning simulation with AI-powered personalities. NPCs learn behavior through Q-learning while their conversations and social interactions are driven by LLM-generated dialogue.

## Overview

Prompted Town simulates a medieval town where agents (farmers, guards, rebels) interact, form relationships, and pursue goals. The simulation combines:

- **Reinforcement Learning**: Agents learn optimal behaviors through Q-learning
- **AI Conversations**: Natural language dialogue powered by Claude/GPT
- **Social Dynamics**: Trust, suspicion, recruitment, and law enforcement
- **Real-time Visualization**: Web-based UI with interactive graph display

### The Setting

A town under authoritarian rule where:
- **Farmers** work the land, pay taxes, and try to survive
- **Guards** enforce laws, patrol locations, and arrest suspicious individuals
- **Rebels** secretly recruit sympathizers while avoiding detection

## Features

- Multi-agent Q-learning with customizable reward functions
- AI-powered conversations with personality-driven dialogue
- Social mechanics: trust levels, suspicion, recruitment
- Law enforcement: curfews, violations, arrests
- Multiple interfaces: CLI training, desktop GUI, web UI
- Real-time visualization with D3.js graph
- Conversation logging and interaction tracking

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd prompted_town

# Install dependencies
pip install -r requirements.txt
```

### Requirements

**Core (always needed):**
- Python 3.10+
- numpy

**AI Conversations (optional):**
- anthropic (for Claude)
- openai (for GPT)

**Visualization:**
- matplotlib
- networkx

**Web UI:**
- flask
- flask-socketio

**Testing:**
- pytest

## Quick Start

### 1. Run Training (CLI)

```bash
# Basic training without AI conversations
python -m prompted_town.rl.training

# Training with AI conversations (requires API key)
export ANTHROPIC_API_KEY="your-key-here"
python -m prompted_town.examples.train_with_ai
```

### 2. Web Interface

```bash
# Start the web server
python -m prompted_town.web

# Open in browser
# http://127.0.0.1:5000
```

### 3. Desktop GUI

```bash
python -m prompted_town.gui
```

## Project Structure

```
prompted_town/
├── core/               # Core types and data structures
│   ├── types.py        # Enums: Location, TimeOfDay, Action, etc.
│   ├── agent_spec.py   # Agent specifications and personalities
│   └── world_state.py  # World state and agent state management
│
├── simulation/         # Game rules and mechanics
│   ├── actions.py      # Action definitions and validation
│   ├── rules.py        # World rules and action resolution
│   ├── social.py       # Social interactions and trust
│   └── laws.py         # Law enforcement and violations
│
├── ai/                 # AI conversation system
│   ├── llm_backend.py  # LLM provider abstraction
│   ├── conversation.py # Conversation engine
│   ├── prompt_builder.py # Context-aware prompt generation
│   └── outcome_parser.py # Parse conversation outcomes
│
├── rl/                 # Reinforcement learning
│   ├── environment.py  # Gym-like environment wrapper
│   ├── observations.py # State observation encoding
│   ├── rewards.py      # Reward computation
│   ├── q_learning.py   # Q-learning implementation
│   └── training.py     # Training loop and utilities
│
├── viz/                # Visualization tools
│   ├── training_plots.py    # Training curves and metrics
│   ├── interaction_graph.py # Social network visualization
│   └── live_viz.py          # Real-time training display
│
├── gui/                # Desktop GUI (tkinter)
│   └── app.py          # Main application
│
├── web/                # Web interface (Flask)
│   ├── server.py       # Flask + SocketIO backend
│   ├── templates/      # HTML templates
│   └── static/         # CSS, JavaScript, D3.js graph
│
├── tests/              # Test suites
│   ├── test_core.py
│   ├── test_simulation.py
│   ├── test_ai.py
│   └── test_rl.py
│
└── examples/           # Example scripts
    ├── demo_conversation.py
    ├── demo_visualization.py
    └── train_with_ai.py
```

## Architecture

### Agent Specification

Each agent has:
- **Public Role**: What they appear to be (farmer, guard)
- **True Faction**: Their actual allegiance (neutral, rebel, authority)
- **Personality**: AI-facing traits for conversation generation
- **RL Profile**: Goals, traits, and reward weights for learning

```python
from prompted_town.core import AgentSpec, AgentRole, Faction

agent = AgentSpec(
    agent_id="farmer_01",
    name="Old Tom",
    public_role=AgentRole.FARMER,
    true_faction=Faction.NEUTRAL,
    # ... personality and RL configuration
)
```

### World State

The simulation tracks:
- Agent locations, energy, gold, suspicion
- Time of day (dawn, day, dusk, night)
- Trust relationships between agents
- Active laws and curfews

### Actions

Agents can:
- **Move**: Travel between locations
- **Work**: Earn gold (location-dependent)
- **Rest**: Recover energy
- **Converse**: Talk with nearby agents
- **Trade**: Exchange goods
- **Patrol/Arrest**: Guard-specific actions

### Conversations

When agents converse, the AI generates contextual dialogue based on:
- Agent personalities and backgrounds
- Current location and time
- Relationship history
- Conversation intent (casual, recruit, interrogate)

## Configuration

### Environment Configuration

```python
from prompted_town.rl import EnvConfig, PromptedTownEnv

config = EnvConfig(
    num_farmers=3,
    num_guards=1,
    num_rebels=1,
    max_ticks=100,
    max_days=7,
    use_ai_conversations=True,
    random_seed=42,
)

env = PromptedTownEnv(config)
```

### Q-Learning Configuration

```python
from prompted_town.rl import QLearningConfig, MultiAgentQLearning

q_config = QLearningConfig(
    learning_rate=0.1,
    discount_factor=0.95,
    epsilon_start=1.0,
    epsilon_end=0.1,
    epsilon_decay=0.995,
)

agents = MultiAgentQLearning(env.agent_ids, q_config)
```

## Web Interface

The web UI provides:

- **Town Map**: Interactive D3.js graph showing locations and agents
- **Controls**: Start/Stop/Step simulation, force conversations
- **Statistics**: Day, tick, time, recruitments, arrests
- **Conversation Log**: Real-time dialogue display
- **AI Toggle**: Enable/disable AI-powered conversations

### Locations

| Location | Description |
|----------|-------------|
| Town Square | Central hub, high visibility |
| Market | Trading, public area |
| Tavern | Social gathering, recruitment spot |
| Farm | Work location, isolated |
| Gate | Town entrance, guarded |
| Home | Private rest area (per agent) |
| Jail | Where arrested agents are held |

## Running Tests

```bash
# Run all tests
pytest prompted_town/tests/

# Run specific test file
pytest prompted_town/tests/test_core.py -v

# Run with coverage
pytest --cov=prompted_town prompted_town/tests/
```

## Examples

### Basic Training Loop

```python
from prompted_town.rl import EnvConfig, PromptedTownEnv, QLearningConfig, MultiAgentQLearning

# Setup
config = EnvConfig(num_farmers=2, num_guards=1, num_rebels=1)
env = PromptedTownEnv(config)
q_config = QLearningConfig(num_actions=env.get_action_space_size())
agents = MultiAgentQLearning(env.agent_ids, q_config)

# Training loop
for episode in range(100):
    obs = env.reset()
    done = False

    while not done:
        actions = agents.select_actions(obs, explore=True)
        action_dict = {aid: env._index_to_action(aid, idx)
                       for aid, idx in actions.items()}

        next_obs, rewards, done, info = env.step(action_dict)
        agents.update_all(obs, actions, rewards, next_obs, done)
        obs = next_obs
```

### AI Conversation

```python
from prompted_town.ai import create_backend, ConversationEngine

# Setup AI backend
backend = create_backend("anthropic", api_key="your-key", model="claude-3-5-haiku-latest")
engine = ConversationEngine(backend)

# Generate conversation
outcome = engine.generate_conversation(
    initiator_spec=rebel_agent,
    target_spec=farmer_agent,
    intent=ConversationIntent.RECRUIT,
    world_context=world_state,
)

print(outcome.transcript)
```

## API Keys

For AI-powered conversations, set your API key:

```bash
# Anthropic Claude
export ANTHROPIC_API_KEY="your-key-here"

# OpenAI GPT
export OPENAI_API_KEY="your-key-here"
```

## License

MIT License

## Version

0.1.0 - Initial release

## Roadmap

Future improvements planned:
- [ ] Independent agent AI consciousness (turn-based conversations)
- [ ] Persistent agent memory across conversations
- [ ] More sophisticated recruitment mechanics
- [ ] Extended social dynamics (factions, alliances)
- [ ] Save/load simulation state
- [ ] Multi-town simulation
