"""
Tests for the RL module.

Tests cover:
- Observation encoding
- Reward computation
- Environment API
- Q-learning agents
- Training utilities
"""

import pytest
from prompted_town.core.types import (
    Location,
    TimeOfDay,
    AgentRole,
    Faction,
    ConversationIntent,
)
from prompted_town.core.world_state import WorldState, create_default_world
from prompted_town.core.agent_spec import (
    create_farmer_spec,
    create_guard_spec,
    create_rebel_spec,
)
from prompted_town.simulation import (
    Action,
    move_action,
    work_action,
    rest_action,
    wait_action,
    conversation_action,
)
from prompted_town.rl import (
    # Observations
    ObservationConfig,
    get_observation_size,
    encode_observation,
    get_simple_observation,
    get_simple_observation_size,
    discretize_observation,
    hash_observation,
    # Rewards
    RewardBreakdown,
    compute_reward,
    get_farmer_reward,
    get_guard_reward,
    get_rebel_reward,
    shape_reward,
    # Environment
    EnvConfig,
    StepInfo,
    PromptedTownEnv,
    # Q-Learning
    QLearningConfig,
    QLearningAgent,
    MultiAgentQLearning,
    # Training
    TrainingConfig,
    EpisodeMetrics,
    TrainingMetrics,
    train,
    run_episode,
    evaluate,
    quick_train,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def agent_specs():
    """Create test agent specifications."""
    return {
        "farmer_00": create_farmer_spec("farmer_00", "Tom"),
        "guard_01": create_guard_spec("guard_01", "Stern"),
        "rebel_02": create_rebel_spec("rebel_02", "Mara"),
    }


@pytest.fixture
def world_state(agent_specs):
    """Create test world state."""
    state = create_default_world(
        agent_ids=list(agent_specs.keys()),
        random_seed=42,
    )
    return state


@pytest.fixture
def env_config():
    """Create test environment config."""
    return EnvConfig(
        num_farmers=2,
        num_guards=1,
        num_rebels=1,
        max_ticks=50,
        max_days=7,
        use_ai_conversations=False,
        random_seed=42,
    )


@pytest.fixture
def env(env_config):
    """Create test environment."""
    return PromptedTownEnv(env_config)


# =============================================================================
# OBSERVATION TESTS
# =============================================================================

class TestObservations:
    """Tests for observation encoding."""

    def test_simple_observation_returns_tuple(self, world_state, agent_specs):
        """Simple observation should return a hashable tuple."""
        obs = get_simple_observation(world_state, "farmer_00", agent_specs)

        assert isinstance(obs, tuple)
        assert len(obs) == 7  # 7 features in simple observation
        # Should be hashable for Q-table
        assert hash(obs) is not None

    def test_simple_observation_components(self, world_state, agent_specs):
        """Check observation components are reasonable."""
        obs = get_simple_observation(world_state, "farmer_00", agent_specs)

        # Unpack: (loc, time, energy_bin, quota_met, susp_bin, guard_nearby, recruit_target)
        loc_idx, time_idx, energy_bin, quota_met, susp_bin, guard_nearby, recruit_nearby = obs

        assert 0 <= loc_idx < len(Location)
        assert 0 <= time_idx < len(TimeOfDay)
        assert 0 <= energy_bin <= 4  # 5 bins
        assert quota_met in (0, 1)
        assert 0 <= susp_bin <= 2  # 3 bins
        assert guard_nearby in (0, 1)
        assert recruit_nearby in (0, 1)

    def test_full_observation_encoding(self, world_state, agent_specs):
        """Full observation should return numpy array."""
        obs = encode_observation(world_state, "farmer_00", agent_specs)

        assert hasattr(obs, 'shape')  # numpy array
        expected_size = get_observation_size()
        assert obs.shape == (expected_size,)

    def test_discretize_observation(self, world_state, agent_specs):
        """Discretization should produce integer indices."""
        obs = encode_observation(world_state, "farmer_00", agent_specs)
        discrete = discretize_observation(obs, bins=10)

        assert isinstance(discrete, tuple)
        for val in discrete:
            assert isinstance(val, int)

    def test_hash_observation(self, world_state, agent_specs):
        """Observation hash should be consistent."""
        obs1 = encode_observation(world_state, "farmer_00", agent_specs)
        obs2 = encode_observation(world_state, "farmer_00", agent_specs)

        hash1 = hash_observation(obs1)
        hash2 = hash_observation(obs2)

        assert hash1 == hash2

    def test_different_agents_different_observations(self, world_state, agent_specs):
        """Different agents should have different observations."""
        obs_farmer = get_simple_observation(world_state, "farmer_00", agent_specs)
        obs_guard = get_simple_observation(world_state, "guard_01", agent_specs)

        # They start at different locations
        assert obs_farmer != obs_guard


# =============================================================================
# REWARD TESTS
# =============================================================================

class TestRewards:
    """Tests for reward computation."""

    def test_reward_breakdown_total(self):
        """RewardBreakdown total should sum components."""
        breakdown = RewardBreakdown(
            survival=1.0,
            energy=0.5,
            wealth=0.3,
            quota=0.2,
        )

        assert breakdown.total == pytest.approx(2.0)

    def test_reward_breakdown_to_dict(self):
        """RewardBreakdown should convert to dict."""
        breakdown = RewardBreakdown(survival=1.0, arrest=-5.0)
        d = breakdown.to_dict()

        assert "survival" in d
        assert "arrest" in d
        assert "total" in d
        assert d["total"] == pytest.approx(-4.0)

    def test_compute_reward_returns_breakdown(self, world_state, agent_specs):
        """compute_reward should return RewardBreakdown."""
        prev_state = world_state
        new_state = world_state.clone()

        breakdown = compute_reward(
            prev_state,
            new_state,
            "farmer_00",
            agent_specs["farmer_00"],
        )

        assert isinstance(breakdown, RewardBreakdown)

    def test_survival_reward_positive_when_alive(self, world_state, agent_specs):
        """Living agents should get positive survival reward."""
        prev_state = world_state
        new_state = world_state.clone()

        breakdown = compute_reward(
            prev_state,
            new_state,
            "farmer_00",
            agent_specs["farmer_00"],
        )

        assert breakdown.survival > 0

    def test_arrest_penalty(self, world_state, agent_specs):
        """Getting arrested should give negative reward."""
        prev_state = world_state
        new_state = world_state.clone()

        # Arrest the agent
        agent = new_state.get_agent("farmer_00")
        agent.is_arrested = True

        breakdown = compute_reward(
            prev_state,
            new_state,
            "farmer_00",
            agent_specs["farmer_00"],
        )

        assert breakdown.arrest < 0
        assert breakdown.survival < 0

    def test_wealth_reward_for_gold_gain(self, world_state, agent_specs):
        """Gaining gold should give positive wealth reward."""
        prev_state = world_state
        new_state = world_state.clone()

        # Give agent gold
        agent = new_state.get_agent("farmer_00")
        agent.gold += 10

        breakdown = compute_reward(
            prev_state,
            new_state,
            "farmer_00",
            agent_specs["farmer_00"],
        )

        assert breakdown.wealth > 0

    def test_shape_reward_clips_values(self):
        """shape_reward should clip to range."""
        assert shape_reward(100.0, clip_max=10.0) == 10.0
        assert shape_reward(-100.0, clip_min=-10.0) == -10.0

    def test_role_specific_reward_functions(self, world_state, agent_specs):
        """Role-specific reward functions should work."""
        prev_state = world_state
        new_state = world_state.clone()

        farmer_reward = get_farmer_reward(
            prev_state, new_state, "farmer_00", agent_specs["farmer_00"]
        )
        assert isinstance(farmer_reward, float)

        guard_reward = get_guard_reward(
            prev_state, new_state, "guard_01", agent_specs["guard_01"], arrests_made=[]
        )
        assert isinstance(guard_reward, float)

        rebel_reward = get_rebel_reward(
            prev_state, new_state, "rebel_02", agent_specs["rebel_02"]
        )
        assert isinstance(rebel_reward, float)


# =============================================================================
# ENVIRONMENT TESTS
# =============================================================================

class TestEnvironment:
    """Tests for the RL environment."""

    def test_env_creation(self, env):
        """Environment should be created successfully."""
        assert env is not None
        assert len(env.agent_ids) == 4  # 2 farmers + 1 guard + 1 rebel

    def test_env_reset_returns_observations(self, env):
        """Reset should return observations for all agents."""
        observations = env.reset()

        assert isinstance(observations, dict)
        assert len(observations) == len(env.agent_ids)

        for agent_id in env.agent_ids:
            assert agent_id in observations

    def test_env_reset_with_seed_deterministic(self, env_config):
        """Reset with seed should be deterministic."""
        env1 = PromptedTownEnv(env_config)
        env2 = PromptedTownEnv(env_config)

        obs1 = env1.reset(seed=123)
        obs2 = env2.reset(seed=123)

        assert obs1 == obs2

    def test_env_step_returns_correct_types(self, env):
        """Step should return correct types."""
        env.reset()

        # Create actions for all agents
        actions = {aid: wait_action(aid) for aid in env.agent_ids}

        observations, rewards, done, info = env.step(actions)

        assert isinstance(observations, dict)
        assert isinstance(rewards, dict)
        assert isinstance(done, bool)
        assert isinstance(info, StepInfo)

    def test_env_step_with_action_indices(self, env):
        """Step should accept action indices."""
        env.reset()

        # Use integer action indices
        actions = {aid: 0 for aid in env.agent_ids}  # 0 = WAIT

        observations, rewards, done, info = env.step(actions)

        assert isinstance(observations, dict)

    def test_env_action_space_size(self, env):
        """Action space size should be consistent."""
        size = env.get_action_space_size()

        assert size > 0
        assert size == 14  # WAIT + REST + WORK + 6 MOVE + DELIVER + 4 CONV

    def test_env_index_to_action(self, env):
        """Action index conversion should work."""
        env.reset()

        # Test various action indices
        wait = env._index_to_action("farmer_00", 0)
        assert wait.action_type.value == "wait"

        rest = env._index_to_action("farmer_00", 1)
        assert rest.action_type.value == "rest"

        work = env._index_to_action("farmer_00", 2)
        assert work.action_type.value == "work"

        move = env._index_to_action("farmer_00", 3)
        assert move.action_type.value == "move"

    def test_env_done_at_max_ticks(self, env_config):
        """Environment should end at max ticks."""
        env_config.max_ticks = 10
        env = PromptedTownEnv(env_config)
        env.reset()

        done = False
        for _ in range(15):
            if done:
                break
            actions = {aid: 0 for aid in env.agent_ids}
            _, _, done, _ = env.step(actions)

        assert done

    def test_env_render(self, env):
        """Render should return string."""
        env.reset()

        output = env.render()

        assert isinstance(output, str)
        assert "Day" in output

    def test_step_info_contains_data(self, env):
        """StepInfo should contain relevant data."""
        env.reset()
        actions = {aid: 0 for aid in env.agent_ids}
        _, _, _, info = env.step(actions)

        assert hasattr(info, 'tick')
        assert hasattr(info, 'day')
        assert hasattr(info, 'time_of_day')
        assert hasattr(info, 'events')
        assert hasattr(info, 'conversations')
        assert hasattr(info, 'arrests')


# =============================================================================
# Q-LEARNING TESTS
# =============================================================================

class TestQLearning:
    """Tests for Q-learning agents."""

    def test_agent_creation(self):
        """Q-learning agent should be created."""
        agent = QLearningAgent("test_agent")

        assert agent.agent_id == "test_agent"
        assert agent.epsilon == 1.0  # Default start

    def test_agent_select_action_explore(self):
        """Agent should explore with high epsilon."""
        agent = QLearningAgent("test")
        agent.epsilon = 1.0  # Always explore

        # Should return valid action
        state = (0, 0, 0)
        action = agent.select_action(state)

        assert 0 <= action < agent.config.num_actions

    def test_agent_select_action_exploit(self):
        """Agent should exploit with low epsilon."""
        config = QLearningConfig(num_actions=4)
        agent = QLearningAgent("test", config)
        agent.epsilon = 0.0  # Always exploit

        state = (0, 0, 0)
        # Set Q-values so action 2 is best
        agent.q_table[state][0] = 0.0
        agent.q_table[state][1] = 0.0
        agent.q_table[state][2] = 10.0
        agent.q_table[state][3] = 0.0

        action = agent.select_action(state, explore=False)

        assert action == 2

    def test_agent_update_q_value(self):
        """Q-value update should follow Bellman equation."""
        config = QLearningConfig(
            learning_rate=0.1,
            discount_factor=0.9,
            num_actions=4,
        )
        agent = QLearningAgent("test", config)

        state = (0, 0)
        next_state = (1, 0)
        action = 1
        reward = 1.0

        # Initial Q-value is 0
        initial_q = agent.q_table[state][action]
        assert initial_q == 0.0

        # Update
        td_error = agent.update(state, action, reward, next_state, done=False)

        # Q should have increased
        new_q = agent.q_table[state][action]
        assert new_q > initial_q

    def test_agent_epsilon_decay(self):
        """Epsilon should decay correctly."""
        config = QLearningConfig(
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay=0.9,
        )
        agent = QLearningAgent("test", config)

        initial_epsilon = agent.epsilon
        agent.decay_epsilon()

        assert agent.epsilon < initial_epsilon
        assert agent.epsilon == pytest.approx(0.9)

    def test_agent_epsilon_floor(self):
        """Epsilon should not go below minimum."""
        config = QLearningConfig(
            epsilon_start=0.1,
            epsilon_end=0.1,
            epsilon_decay=0.5,
        )
        agent = QLearningAgent("test", config)

        for _ in range(10):
            agent.decay_epsilon()

        assert agent.epsilon >= config.epsilon_end

    def test_agent_stats(self):
        """Agent stats should track learning."""
        agent = QLearningAgent("test")

        # Do some updates
        for i in range(5):
            state = (i, 0)
            agent.update(state, 0, 1.0, (i+1, 0), done=False)

        stats = agent.get_stats()

        assert stats["agent_id"] == "test"
        assert stats["total_updates"] == 5
        assert stats["states_visited"] == 5

    def test_multi_agent_creation(self):
        """Multi-agent manager should create agents."""
        agent_ids = ["agent_1", "agent_2", "agent_3"]
        multi = MultiAgentQLearning(agent_ids)

        assert len(multi.agents) == 3
        assert "agent_1" in multi.agents
        assert "agent_2" in multi.agents
        assert "agent_3" in multi.agents

    def test_multi_agent_select_actions(self):
        """Multi-agent should select actions for all."""
        agent_ids = ["a1", "a2"]
        multi = MultiAgentQLearning(agent_ids)

        observations = {
            "a1": (0, 0, 0),
            "a2": (1, 1, 1),
        }

        actions = multi.select_actions(observations)

        assert "a1" in actions
        assert "a2" in actions

    def test_multi_agent_update_all(self):
        """Multi-agent should update all agents."""
        agent_ids = ["a1", "a2"]
        multi = MultiAgentQLearning(agent_ids)

        states = {"a1": (0,), "a2": (0,)}
        actions = {"a1": 0, "a2": 1}
        rewards = {"a1": 1.0, "a2": -1.0}
        next_states = {"a1": (1,), "a2": (1,)}

        td_errors = multi.update_all(states, actions, rewards, next_states, done=False)

        assert "a1" in td_errors
        assert "a2" in td_errors

    def test_multi_agent_decay_all(self):
        """Multi-agent should decay all epsilons."""
        agent_ids = ["a1", "a2"]
        multi = MultiAgentQLearning(agent_ids)

        initial = multi.agents["a1"].epsilon
        multi.decay_epsilon_all()

        assert multi.agents["a1"].epsilon < initial
        assert multi.agents["a2"].epsilon < initial


# =============================================================================
# TRAINING TESTS
# =============================================================================

class TestTraining:
    """Tests for training utilities."""

    def test_training_config_defaults(self):
        """TrainingConfig should have sensible defaults."""
        config = TrainingConfig()

        assert config.num_episodes > 0
        assert config.max_steps_per_episode > 0
        assert config.eval_frequency > 0

    def test_episode_metrics(self):
        """EpisodeMetrics should store episode data."""
        metrics = EpisodeMetrics(
            episode=1,
            total_steps=50,
            total_rewards={"a1": 10.0, "a2": 5.0},
            final_day=3,
            rebels_recruited=2,
            arrests_made=1,
            quota_completions=3,
        )

        assert metrics.episode == 1
        assert metrics.total_steps == 50
        assert metrics.rebels_recruited == 2

    def test_training_metrics_summary(self):
        """TrainingMetrics should compute summaries."""
        metrics = TrainingMetrics()
        metrics.episode_lengths = [10, 20, 30]
        metrics.rebels_recruited_history = [0, 1, 2]

        summary = metrics.get_summary()

        assert summary["avg_episode_length"] == pytest.approx(20.0)
        assert summary["avg_rebels_recruited"] == pytest.approx(1.0)

    def test_run_episode(self, env):
        """run_episode should complete an episode."""
        env.reset()

        config = QLearningConfig(num_actions=env.get_action_space_size())
        agents = MultiAgentQLearning(env.agent_ids, config)

        metrics = run_episode(env, agents, max_steps=10)

        assert isinstance(metrics, EpisodeMetrics)
        assert metrics.total_steps <= 10
        assert len(metrics.total_rewards) == len(env.agent_ids)

    def test_evaluate_no_exploration(self, env):
        """evaluate should run without exploration."""
        env.reset()

        config = QLearningConfig(num_actions=env.get_action_space_size())
        agents = MultiAgentQLearning(env.agent_ids, config)

        results = evaluate(env, agents, num_episodes=3)

        assert len(results) == 3
        for episode_rewards in results:
            assert isinstance(episode_rewards, dict)

    def test_quick_train_runs(self):
        """quick_train should complete training."""
        env, agents, metrics = quick_train(
            num_episodes=10,
            num_farmers=1,
            num_guards=1,
            num_rebels=1,
            verbose=False,
        )

        assert env is not None
        assert agents is not None
        assert metrics.episodes_completed == 10

    def test_train_with_callback(self, env):
        """train should call callback after each episode."""
        env.reset()

        config = QLearningConfig(num_actions=env.get_action_space_size())
        agents = MultiAgentQLearning(env.agent_ids, config)

        callback_calls = []
        def callback(episode, metrics):
            callback_calls.append(episode)

        train_config = TrainingConfig(
            num_episodes=5,
            max_steps_per_episode=10,
            log_frequency=100,  # Suppress logging
            eval_frequency=100,
            save_frequency=100,
        )

        train(env, agents, train_config, callback=callback)

        assert len(callback_calls) == 5


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for the full RL pipeline."""

    def test_full_training_loop(self):
        """Test complete training loop."""
        env_config = EnvConfig(
            num_farmers=2,
            num_guards=1,
            num_rebels=1,
            max_ticks=20,
            use_ai_conversations=False,
            random_seed=42,
        )
        env = PromptedTownEnv(env_config)

        q_config = QLearningConfig(
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay=0.9,
            num_actions=env.get_action_space_size(),
        )
        agents = MultiAgentQLearning(env.agent_ids, q_config)

        train_config = TrainingConfig(
            num_episodes=20,
            max_steps_per_episode=20,
            log_frequency=100,
            eval_frequency=100,
            save_frequency=100,
        )

        metrics = train(env, agents, train_config)

        assert metrics.episodes_completed == 20
        assert metrics.total_steps > 0

        # Check that learning happened
        for agent_id, agent in agents.agents.items():
            assert agent.total_updates > 0
            assert len(agent.states_visited) > 0

    def test_observation_reward_consistency(self):
        """Observations and rewards should be consistent across steps."""
        env = PromptedTownEnv(EnvConfig(
            num_farmers=1,
            num_guards=1,
            num_rebels=1,
            random_seed=42,
        ))

        obs1 = env.reset(seed=42)

        # Take same actions
        actions = {aid: 0 for aid in env.agent_ids}
        obs2, rewards1, _, _ = env.step(actions)

        # Reset and repeat
        env.reset(seed=42)
        obs3, rewards2, _, _ = env.step(actions)

        # Should be deterministic
        assert obs2 == obs3
        assert rewards1 == rewards2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
