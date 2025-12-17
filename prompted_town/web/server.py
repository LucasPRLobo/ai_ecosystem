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

    def init_simulation(self):
        """Initialize the simulation."""
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
        self.total_arrests = 0
        self.total_conversations = 0
        self.conversation_history = []

    def reset(self):
        """Reset the simulation."""
        self.stop()
        if self.env:
            self.env.reset()
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

        return {
            'done': done,
            'conversations': info.conversations,
            'arrests': info.arrests,
        }

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

    def toggle_ai(self, enable: bool):
        """Toggle AI conversations."""
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
                    self.ai_initialized = True
                    print("AI backend initialized successfully")
                    return True
                except Exception as e:
                    print(f"Failed to initialize AI: {e}")
                    return False
            return False
        elif enable and self.ai_initialized:
            self.env.config.use_ai_conversations = True
            return True
        else:
            if self.env:
                self.env.config.use_ai_conversations = False
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
                    state['step_conversations'] = result.get('conversations', [])
                    state['step_arrests'] = result.get('arrests', [])
                    socketio.emit('step_complete', state)
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
        """Get recent conversations."""
        return self.conversation_history[-limit:]


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
    return jsonify({
        'status': 'ok',
        'state': sim.get_state(),
        'conversations': conversations,
    })


@app.route('/api/toggle_ai', methods=['POST'])
def toggle_ai():
    """Toggle AI conversations."""
    data = request.json or {}
    enable = data.get('enable', False)
    success = sim.toggle_ai(enable)
    return jsonify({
        'status': 'ok' if success else 'error',
        'ai_enabled': sim.use_ai,
        'message': 'AI enabled' if success and enable else 'AI disabled' if success else 'No API key found'
    })


@app.route('/api/set_speed', methods=['POST'])
def set_speed():
    """Set simulation speed."""
    data = request.json or {}
    speed = data.get('speed', 0.8)
    sim.step_delay = max(0.1, min(3.0, speed))
    return jsonify({'status': 'ok', 'speed': sim.step_delay})


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
        state['step_conversations'] = result.get('conversations', [])
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
    emit('conversation', {
        'conversations': conversations,
        'state': sim.get_state(),
    })


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
