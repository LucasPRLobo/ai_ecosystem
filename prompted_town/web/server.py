"""
Flask server for Prompted Town web UI.

Provides REST API and WebSocket for real-time simulation updates.
"""

import os
import json
import threading
import time
from typing import Optional
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit

from ..core.types import Location, TimeOfDay, ConversationIntent
from ..core.traits import (
    AGENT_TEMPLATES,
    PLACE_TEMPLATES,
    AgentTraits,
    PlaceTraits,
    RecruitmentStyle,
    RecruitmentConfig,
    get_agent_template,
    get_place_template,
)
from ..rl import EnvConfig, PromptedTownEnv, QLearningConfig, MultiAgentQLearning
from ..simulation import conversation_action
from ..ai import create_backend, ConversationEngine


# =============================================================================
# APP SETUP
# =============================================================================

def create_app():
    """Create and configure Flask app."""
    app = Flask(__name__,
                template_folder='templates',
                static_folder='static')
    app.config['SECRET_KEY'] = 'prompted-town-secret'
    return app


app = create_app()
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')


# =============================================================================
# HELPERS
# =============================================================================

def serialize_conversation(conv) -> dict:
    """Convert a ConversationOutcome to a JSON-serializable dict for the frontend."""
    # If already a dict (from environment's info.conversations), fix the format
    if isinstance(conv, dict):
        # The environment builds dicts with initiator/target as IDs and separate name fields
        result = {
            'initiator': conv.get('initiator_name', conv.get('initiator', 'Unknown')),
            'target': conv.get('target_name', conv.get('target', 'Unknown')),
            'intent': conv.get('intent', 'casual'),
            'transcript': conv.get('transcript', []),
            'outcome_summary': conv.get('outcome_summary', ''),
            'recruitment_successful': conv.get('recruitment_successful', False),
            'suspicion_raised': conv.get('suspicion_raised', False),
        }
        return result

    # Get agent names from specs if available (uses global sim)
    initiator_name = conv.initiator_id
    target_name = conv.target_id

    # Try to get names from the simulation's agent specs
    try:
        if sim.env and sim.env.agent_specs:
            init_spec = sim.env.agent_specs.get(conv.initiator_id)
            target_spec = sim.env.agent_specs.get(conv.target_id)
            if init_spec:
                initiator_name = init_spec.name
            if target_spec:
                target_name = target_spec.name
    except Exception:
        pass  # Fall back to IDs

    # Build transcript with speaker names
    transcript = []
    for turn in conv.transcript:
        speaker = turn.speaker_id
        # Map speaker_id to name
        if turn.speaker_id == conv.initiator_id:
            speaker = initiator_name
        elif turn.speaker_id == conv.target_id:
            speaker = target_name
        else:
            # Try to look up other speakers
            try:
                if sim.env and sim.env.agent_specs:
                    spec = sim.env.agent_specs.get(turn.speaker_id)
                    if spec:
                        speaker = spec.name
            except Exception:
                pass

        transcript.append({
            'speaker': speaker,
            'text': turn.text,
        })

    # Determine intent string
    intent = 'casual'
    if conv.recruitment_attempted:
        intent = 'recruit'
    elif conv.interrogation_attempted:
        intent = 'interrogate'
    elif conv.trade_completed:
        intent = 'trade'

    return {
        'initiator': initiator_name,
        'target': target_name,
        'intent': intent,
        'transcript': transcript,
        'outcome_summary': conv.get_summary() if hasattr(conv, 'get_summary') else '',
        'recruitment_successful': conv.recruitment_successful,
        'suspicion_raised': conv.suspicion_raised,
    }


# =============================================================================
# SIMULATION STATE
# =============================================================================

class SimulationManager:
    """Manages the simulation state."""

    def __init__(self):
        self.env: Optional[PromptedTownEnv] = None
        self.agents: Optional[MultiAgentQLearning] = None
        self.is_running = False
        self.use_ai = False
        self.ai_initialized = False
        self.step_delay = 0.8  # seconds
        self.total_arrests = 0
        self.total_conversations = 0
        self.conversation_history = []
        self._run_thread = None
        self.thought_interval = 10  # Generate thoughts every N ticks
        self.ticks_since_thoughts = 0

    def init_simulation(self):
        """Initialize the simulation."""
        from ..ai import AgentMindManager

        env_config = EnvConfig(
            num_farmers=2,
            num_guards=1,
            num_rebels=1,
            max_ticks=200,
            max_days=14,
            use_ai_conversations=False,
            random_seed=None,
        )
        self.env = PromptedTownEnv(env_config)

        q_config = QLearningConfig(
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon_start=0.5,
            epsilon_end=0.1,
            epsilon_decay=0.995,
            num_actions=self.env.get_action_space_size(),
        )
        self.agents = MultiAgentQLearning(self.env.agent_ids, q_config)

        self.env.reset()

        # Initialize mind manager for trait editing (works without AI backend)
        self.env.mind_manager = AgentMindManager(backend=None)
        self.env.mind_manager.initialize_all(self.env.agent_specs)

        self.total_arrests = 0
        self.total_conversations = 0
        self.conversation_history = []

    def reset(self):
        """Reset the simulation."""
        from ..ai import AgentMindManager

        self.stop()
        if self.env:
            # Preserve AI backend reference
            backend = getattr(self.env, 'ai_backend', None)

            self.env.reset()

            # Reinitialize mind manager (preserves backend if AI was enabled)
            self.env.mind_manager = AgentMindManager(backend=backend)
            self.env.mind_manager.initialize_all(self.env.agent_specs)

            self.total_arrests = 0
            self.total_conversations = 0
            self.conversation_history = []

    def step(self):
        """Execute one simulation step."""
        if not self.env or not self.agents:
            return None

        observations = self.env._get_observations()
        action_indices = self.agents.select_actions(observations, explore=True)
        actions = {
            aid: self.env._index_to_action(aid, idx)
            for aid, idx in action_indices.items()
        }

        next_obs, rewards, done, info = self.env.step(actions)

        self.agents.update_all(
            observations, action_indices, rewards, next_obs, done
        )

        self.total_arrests += len(info.arrests)
        self.total_conversations += len(info.conversations)

        # Store conversations
        for conv in info.conversations:
            self.conversation_history.append(conv)

        # Track ticks for thought generation
        self.ticks_since_thoughts += 1

        return {
            'done': done,
            'conversations': info.conversations,
            'arrests': info.arrests,
        }

    def generate_thoughts(self) -> list[dict]:
        """Generate thoughts for all agents."""
        if not self.env or not hasattr(self.env, 'mind_manager') or not self.env.mind_manager:
            return []

        thoughts_dict = self.env.mind_manager.generate_all_thoughts()
        return list(thoughts_dict.values())

    def should_generate_thoughts(self) -> bool:
        """Check if it's time to generate thoughts."""
        return self.ticks_since_thoughts >= self.thought_interval

    def reset_thought_counter(self):
        """Reset the tick counter for thoughts."""
        self.ticks_since_thoughts = 0

    def check_guard_suspicion_thoughts(self) -> list[dict]:
        """
        Check if any guards see suspicious people (50-70% suspicion)
        and generate 'considering arrest' thoughts for them.
        """
        if not self.env or not self.env.state:
            return []

        thoughts = []
        state = self.env.state

        # Find all guards
        for guard_id, spec in self.env.agent_specs.items():
            if spec.public_role.value != "guard":
                continue

            guard_state = state.get_agent(guard_id)
            if not guard_state or not guard_state.can_act():
                continue

            # Check other agents at same location
            agents_at_loc = state.get_agents_at_location(guard_state.location)
            for other_id in agents_at_loc:
                if other_id == guard_id:
                    continue

                other_state = state.get_agent(other_id)
                other_spec = self.env.agent_specs.get(other_id)
                if not other_state or not other_spec:
                    continue

                # Check if suspicion is in consideration range (50-70%)
                if 0.5 <= other_state.suspicion_level <= 0.7:
                    # Generate a considering arrest thought
                    thought_text = self._generate_considering_arrest_thought(
                        spec.name, other_spec.name, other_state.suspicion_level
                    )
                    thoughts.append({
                        "agent_id": guard_id,
                        "agent_name": spec.name,
                        "thought": thought_text,
                        "role": "guard",
                        "type": "considering_arrest",
                    })

        return thoughts

    def _generate_considering_arrest_thought(self, guard_name: str, suspect_name: str, suspicion: float) -> str:
        """Generate a thought about considering an arrest."""
        import random

        suspicion_pct = int(suspicion * 100)

        templates = [
            f"There's something off about {suspect_name}... I should keep a close eye on them.",
            f"That {suspect_name}... their behavior has been suspicious lately. Maybe I should question them.",
            f"{suspect_name} has been acting strange. If this continues, I may need to take action.",
            f"I've got my eye on {suspect_name}. One more slip and they'll have some explaining to do.",
            f"Something tells me {suspect_name} is hiding something. The evidence is mounting...",
            f"Should I arrest {suspect_name} now? No... I need more evidence. But they're on thin ice.",
        ]

        return random.choice(templates)

    def generate_arrest_witness_thoughts(self, arrests: list[dict]) -> list[dict]:
        """Generate thoughts for witnesses of an arrest."""
        if not self.env or not arrests:
            return []

        thoughts = []

        for arrest in arrests:
            arrested_name = arrest.get("arrested_name", "someone")
            guard_name = arrest.get("guard_name", "a guard")
            location = arrest.get("location", "the town")

            # Generate thoughts for all agents at the location
            if self.env.state and self.env.mind_manager:
                for agent_id, mind in self.env.mind_manager.minds.items():
                    # Skip the guard and arrested person
                    if agent_id == arrest.get("guard_id") or agent_id == arrest.get("arrested_id"):
                        continue

                    agent_state = self.env.state.get_agent(agent_id)
                    if not agent_state:
                        continue

                    # Only agents at the location witness
                    if agent_state.location.value != location:
                        continue

                    thought_text = self._generate_witness_arrest_thought(
                        mind, arrested_name, guard_name
                    )
                    thoughts.append({
                        "agent_id": agent_id,
                        "agent_name": mind.agent_spec.name,
                        "thought": thought_text,
                        "role": mind.agent_spec.public_role.value,
                        "type": "witness_arrest",
                    })

        return thoughts

    def _generate_witness_arrest_thought(self, mind, arrested_name: str, guard_name: str) -> str:
        """Generate a witness thought about an arrest."""
        import random
        from ..core.agent_spec import Faction

        # Different thoughts based on faction
        if mind.agent_spec.true_faction == Faction.REBEL:
            templates = [
                f"No... {arrested_name}! We must be more careful. The guards are closing in.",
                f"This is a warning to us all. {guard_name} won't stop until we're all in chains.",
                f"{arrested_name} taken... We need to act soon, or we'll all share their fate.",
                f"The oppression grows bolder. {arrested_name}'s arrest only strengthens my resolve.",
            ]
        else:
            templates = [
                f"*watches nervously as {guard_name} drags {arrested_name} away* That could be any of us...",
                f"Poor {arrested_name}... I hope they're innocent. These are dark times.",
                f"Another arrest... Is anyone truly safe anymore?",
                f"*shivers* I must keep my head down. I don't want to end up like {arrested_name}.",
                f"What did {arrested_name} do? The guards seem more aggressive lately...",
            ]

        return random.choice(templates)

    def force_conversation(self):
        """Force a conversation between two agents."""
        if not self.env:
            return None

        state = self.env.state

        # Find rebel and target
        rebel_id = None
        target_id = None

        for aid, spec in self.env.agent_specs.items():
            if spec.true_faction.value == "rebel":
                rebel_id = aid
                break

        if rebel_id:
            nearby = state.get_nearby_agents(rebel_id)
            for nid in nearby:
                spec = self.env.agent_specs.get(nid)
                if spec and spec.public_role.value == "farmer":
                    target_id = nid
                    break

        if not target_id:
            # Find any two agents at same location
            agents_by_loc = {}
            for aid, agent in state.agents.items():
                if agent.can_act():
                    loc = agent.location
                    if loc not in agents_by_loc:
                        agents_by_loc[loc] = []
                    agents_by_loc[loc].append(aid)

            for loc, agents_list in agents_by_loc.items():
                if len(agents_list) >= 2:
                    rebel_id = agents_list[0]
                    target_id = agents_list[1]
                    break

        if rebel_id and target_id:
            intent = ConversationIntent.RECRUIT if "rebel" in rebel_id else ConversationIntent.CASUAL
            action = conversation_action(rebel_id, target_id, intent)

            actions = {aid: self.env._index_to_action(aid, 0) for aid in self.env.agent_ids}
            actions[rebel_id] = action

            next_obs, rewards, done, info = self.env.step(actions)

            self.total_conversations += len(info.conversations)

            for conv in info.conversations:
                self.conversation_history.append(conv)

            return info.conversations

        return []

    def toggle_ai(self, enable: bool, use_mindful: bool = True):
        """Toggle AI conversations.

        Args:
            enable: Whether to enable AI conversations
            use_mindful: If True, use mindful conversations (independent agent minds)
        """
        self.use_ai = enable

        if enable and not self.ai_initialized:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                print("Initializing AI backend...")
                try:
                    self.env.ai_backend = create_backend(
                        "anthropic",
                        api_key=api_key,
                        model="claude-3-5-haiku-latest"
                    )
                    self.env.conversation_engine = ConversationEngine(self.env.ai_backend)
                    self.env.config.use_ai_conversations = True

                    # Enable mindful conversations - update existing mind_manager with backend
                    if use_mindful and self.env.mind_manager:
                        self.env.mind_manager.set_backend(self.env.ai_backend)
                        self.env.config.use_mindful_conversations = True
                        print("Mindful conversations enabled (independent agent minds)")

                    self.ai_initialized = True
                    print("AI backend initialized successfully")
                    return True
                except Exception as e:
                    print(f"Failed to initialize AI: {e}")
                    import traceback
                    traceback.print_exc()
                    return False
            return False
        elif enable and self.ai_initialized:
            self.env.config.use_ai_conversations = True
            if use_mindful:
                self.env.config.use_mindful_conversations = True
            return True
        else:
            if self.env:
                self.env.config.use_ai_conversations = False
                self.env.config.use_mindful_conversations = False
            return True

    def start(self):
        """Start continuous running."""
        if self.is_running:
            return
        self.is_running = True
        self._run_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._run_thread.start()

    def stop(self):
        """Stop continuous running."""
        self.is_running = False
        if self._run_thread:
            self._run_thread.join(timeout=1)

    def _run_loop(self):
        """Background run loop."""
        while self.is_running:
            try:
                result = self.step()
                if result:
                    state = self.get_state()
                    # Serialize conversations for the frontend
                    convs = result.get('conversations', [])
                    state['step_conversations'] = [serialize_conversation(c) for c in convs]
                    arrests = result.get('arrests', [])
                    state['step_arrests'] = arrests
                    socketio.emit('step_complete', state)

                    # Check for guard "considering arrest" thoughts
                    guard_thoughts = self.check_guard_suspicion_thoughts()
                    if guard_thoughts:
                        socketio.emit('agent_thoughts', {'thoughts': guard_thoughts})

                    # Generate witness thoughts if arrests happened
                    if arrests:
                        witness_thoughts = self.generate_arrest_witness_thoughts(arrests)
                        if witness_thoughts:
                            socketio.emit('agent_thoughts', {'thoughts': witness_thoughts})

                    # Check if it's time to generate regular thoughts
                    if self.should_generate_thoughts():
                        thoughts = self.generate_thoughts()
                        if thoughts:
                            socketio.emit('agent_thoughts', {'thoughts': thoughts})
                        self.reset_thought_counter()

                    if result['done']:
                        self.is_running = False
                        socketio.emit('episode_ended', {})
            except Exception as e:
                print(f"Error in simulation step: {e}")
                socketio.emit('error', {'message': str(e)})
            time.sleep(self.step_delay)

    def get_state(self):
        """Get current simulation state as JSON-serializable dict."""
        if not self.env or not self.env.state:
            return {}

        state = self.env.state

        agents_data = []
        for aid in self.env.agent_ids:
            agent = state.get_agent(aid)
            spec = self.env.agent_specs.get(aid)
            if agent and spec:
                agents_data.append({
                    'id': aid,
                    'name': spec.name,
                    'role': spec.public_role.value,
                    'true_faction': spec.true_faction.value,
                    'location': agent.location.value,
                    'energy': agent.energy,
                    'gold': agent.gold,
                    'suspicion': round(agent.suspicion_level, 2),
                    'is_recruited': agent.is_recruited,
                    'is_arrested': agent.is_arrested,
                    'can_act': agent.can_act(),
                })

        return {
            'day': state.day,
            'tick': state.tick,
            'time_of_day': state.time_of_day.value,
            'total_recruited': state.total_rebels_recruited,
            'total_arrests': self.total_arrests,
            'total_conversations': self.total_conversations,
            'agents': agents_data,
            'is_running': self.is_running,
            'use_ai': self.use_ai,
            'ai_available': bool(os.environ.get("ANTHROPIC_API_KEY")),
        }

    def get_conversations(self, limit: int = 50):
        """Get recent conversations serialized for the frontend."""
        convs = self.conversation_history[-limit:]
        return [serialize_conversation(c) for c in convs]


# Global simulation manager
sim = SimulationManager()


# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/api/state')
def get_state():
    """Get current simulation state."""
    return jsonify(sim.get_state())


@app.route('/api/conversations')
def get_conversations():
    """Get conversation history."""
    return jsonify(sim.get_conversations())


@app.route('/api/init', methods=['POST'])
def init_simulation():
    """Initialize simulation."""
    sim.init_simulation()
    return jsonify({'status': 'ok'})


@app.route('/api/reset', methods=['POST'])
def reset_simulation():
    """Reset simulation."""
    sim.reset()
    return jsonify({'status': 'ok', 'state': sim.get_state()})


@app.route('/api/step', methods=['POST'])
def step_simulation():
    """Step simulation once."""
    result = sim.step()
    return jsonify({
        'status': 'ok',
        'state': sim.get_state(),
        'result': result,
    })


@app.route('/api/force_conversation', methods=['POST'])
def force_conversation():
    """Force a conversation."""
    conversations = sim.force_conversation()
    serialized = [serialize_conversation(c) for c in conversations] if conversations else []
    return jsonify({
        'status': 'ok',
        'state': sim.get_state(),
        'conversations': serialized,
    })


@app.route('/api/toggle_ai', methods=['POST'])
def toggle_ai():
    """Toggle AI conversations."""
    data = request.json or {}
    enable = data.get('enable', False)
    use_mindful = data.get('use_mindful', True)  # Default to mindful mode
    success = sim.toggle_ai(enable, use_mindful=use_mindful)
    return jsonify({
        'status': 'ok' if success else 'error',
        'ai_enabled': sim.use_ai,
        'use_mindful': use_mindful if enable else False,
        'message': 'AI enabled (mindful mode)' if success and enable and use_mindful else 'AI enabled' if success and enable else 'AI disabled' if success else 'No API key found'
    })


@app.route('/api/set_speed', methods=['POST'])
def set_speed():
    """Set simulation speed."""
    data = request.json or {}
    speed = data.get('speed', 0.8)
    sim.step_delay = max(0.1, min(3.0, speed))
    return jsonify({'status': 'ok', 'speed': sim.step_delay})


@app.route('/api/thoughts', methods=['POST'])
def generate_thoughts():
    """Generate thoughts for all agents."""
    thoughts = sim.generate_thoughts()
    return jsonify({
        'status': 'ok',
        'thoughts': thoughts,
    })


@app.route('/api/thought/<agent_id>', methods=['POST'])
def generate_agent_thought(agent_id):
    """Generate a thought for a specific agent."""
    mind_manager = getattr(sim.env, 'mind_manager', None) if sim.env else None
    if not mind_manager:
        return jsonify({'error': 'Mind manager not initialized'}), 400

    thought = mind_manager.generate_thought_for(agent_id)
    if not thought:
        return jsonify({'error': 'Agent not found'}), 404

    return jsonify({
        'status': 'ok',
        'thought': thought,
    })


# =============================================================================
# TRAIT EDITING API
# =============================================================================

@app.route('/api/templates')
def get_templates():
    """Get all available templates."""
    agent_templates = {}
    for name, template in AGENT_TEMPLATES.items():
        agent_templates[name] = {
            'traits': template.to_dict(),
            'description': template.describe(),
        }

    place_templates = {}
    for name, template in PLACE_TEMPLATES.items():
        place_templates[name] = {
            'traits': template.to_dict(),
            'description': template.describe(),
        }

    recruitment_styles = [style.value for style in RecruitmentStyle]

    return jsonify({
        'agent_templates': agent_templates,
        'place_templates': place_templates,
        'recruitment_styles': recruitment_styles,
    })


@app.route('/api/agent/<agent_id>')
def get_agent_details(agent_id):
    """Get detailed info about an agent including traits."""
    if not sim.env:
        return jsonify({'error': 'Simulation not initialized'}), 400

    spec = sim.env.agent_specs.get(agent_id)
    mind_manager = getattr(sim.env, 'mind_manager', None)
    mind = mind_manager.get_mind(agent_id) if mind_manager else None
    agent_state = sim.env.state.get_agent(agent_id) if sim.env.state else None

    if not spec:
        return jsonify({'error': 'Agent not found'}), 404

    data = {
        'id': agent_id,
        'name': spec.name,
        'role': spec.public_role.value,
        'faction': spec.true_faction.value,
        'personality': {
            'core_identity': spec.ai_personality.core_identity,
            'background': spec.ai_personality.background,
        },
    }

    if agent_state:
        data['state'] = {
            'location': agent_state.location.value,
            'energy': agent_state.energy,
            'gold': agent_state.gold,
            'suspicion': agent_state.suspicion_level,
            'is_recruited': agent_state.is_recruited,
            'is_arrested': agent_state.is_arrested,
        }

    if mind:
        data['traits'] = mind.traits.to_dict()
        data['traits_description'] = mind.traits.describe()
        data['mood'] = mind.current_mood
        data['grievances'] = [
            {'topic': g.topic, 'intensity': g.intensity}
            for g in mind.grievances
        ]
        data['recruitment_style'] = mind.recruitment_config.style.value

    return jsonify(data)


@app.route('/api/agent/<agent_id>/traits', methods=['POST'])
def update_agent_traits(agent_id):
    """Update an agent's traits."""
    mind_manager = getattr(sim.env, 'mind_manager', None) if sim.env else None
    if not mind_manager:
        return jsonify({'error': 'Mind manager not initialized'}), 400

    mind = mind_manager.get_mind(agent_id)
    if not mind:
        return jsonify({'error': 'Agent mind not found'}), 404

    data = request.json or {}

    # Update individual traits if provided
    traits_data = data.get('traits', {})
    for trait_name, value in traits_data.items():
        if hasattr(mind.traits, trait_name):
            setattr(mind.traits, trait_name, max(0.0, min(1.0, float(value))))

    return jsonify({
        'status': 'ok',
        'traits': mind.traits.to_dict(),
        'description': mind.traits.describe(),
    })


@app.route('/api/agent/<agent_id>/template', methods=['POST'])
def apply_agent_template(agent_id):
    """Apply a template to an agent."""
    mind_manager = getattr(sim.env, 'mind_manager', None) if sim.env else None
    if not mind_manager:
        return jsonify({'error': 'Mind manager not initialized'}), 400

    mind = mind_manager.get_mind(agent_id)
    if not mind:
        return jsonify({'error': 'Agent mind not found'}), 404

    data = request.json or {}
    template_name = data.get('template')

    if not template_name or template_name not in AGENT_TEMPLATES:
        return jsonify({'error': f'Invalid template: {template_name}'}), 400

    # Apply template
    mind.traits = get_agent_template(template_name)

    return jsonify({
        'status': 'ok',
        'template': template_name,
        'traits': mind.traits.to_dict(),
        'description': mind.traits.describe(),
    })


@app.route('/api/agent/<agent_id>/recruitment_style', methods=['POST'])
def set_recruitment_style(agent_id):
    """Set the recruitment style for an agent (rebels only)."""
    mind_manager = getattr(sim.env, 'mind_manager', None) if sim.env else None
    if not mind_manager:
        return jsonify({'error': 'Mind manager not initialized'}), 400

    mind = mind_manager.get_mind(agent_id)
    if not mind:
        return jsonify({'error': 'Agent mind not found'}), 404

    data = request.json or {}
    style_name = data.get('style', 'moderate')

    try:
        style = RecruitmentStyle(style_name)
        mind.recruitment_config = RecruitmentConfig.from_style(style)
        return jsonify({
            'status': 'ok',
            'style': style.value,
            'config': {
                'trust_threshold': mind.recruitment_config.trust_threshold,
                'grievance_threshold': mind.recruitment_config.grievance_threshold,
                'revelation_rate': mind.recruitment_config.revelation_rate,
            }
        })
    except ValueError:
        return jsonify({'error': f'Invalid style: {style_name}'}), 400


# =============================================================================
# WEBSOCKET EVENTS
# =============================================================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    if not sim.env:
        sim.init_simulation()
    emit('state_update', sim.get_state())


@socketio.on('start')
def handle_start():
    """Start continuous simulation."""
    sim.start()
    emit('started', {})


@socketio.on('stop')
def handle_stop():
    """Stop continuous simulation."""
    sim.stop()
    emit('stopped', {})


@socketio.on('step')
def handle_step():
    """Step simulation once."""
    result = sim.step()
    state = sim.get_state()
    if result:
        convs = result.get('conversations', [])
        state['step_conversations'] = [serialize_conversation(c) for c in convs]
        state['step_arrests'] = result.get('arrests', [])
    emit('step_complete', state)


@socketio.on('reset')
def handle_reset():
    """Reset simulation."""
    sim.reset()
    emit('state_update', sim.get_state())


@socketio.on('force_conversation')
def handle_force_conversation():
    """Force a conversation."""
    conversations = sim.force_conversation()
    serialized = [serialize_conversation(c) for c in conversations] if conversations else []
    emit('conversation', {
        'conversations': serialized,
        'state': sim.get_state(),
    })


@socketio.on('generate_thoughts')
def handle_generate_thoughts():
    """Generate thoughts for all agents."""
    thoughts = sim.generate_thoughts()
    emit('agent_thoughts', {'thoughts': thoughts})


# =============================================================================
# RUN
# =============================================================================

def run_server(host='127.0.0.1', port=5000, debug=False):
    """Run the web server."""
    print(f"\n{'='*50}")
    print("  Prompted Town Web UI")
    print(f"{'='*50}")
    print(f"\n  Open http://{host}:{port} in your browser\n")

    if os.environ.get("ANTHROPIC_API_KEY"):
        print("  ✓ Anthropic API key detected")
    else:
        print("  ✗ No API key - using rule-based conversations")

    print(f"\n{'='*50}\n")

    sim.init_simulation()
    socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    run_server(debug=True)
