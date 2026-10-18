# ============================================================
# Environment Configuration
# ============================================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ============================================================
# Imports
# ============================================================

import random
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from base_agent import BaseAgent
from chip_game_env import ChipGameEnv

# Value-based agents
from dqn_agent import DQNAgent
from ddqn_agent import DDQNAgent
from dueling_dqn_agent import DuelingDQNAgent

# Random baseline
from random_agent import RandomAgent

# Actor-critic agents
from a2c_agent import A2CAgent
from a3c_agent import A3CAgent
from ppo_agent import PPOAgent

# ============================================================
# Shared Agent Player
# ============================================================

class SharedAgentPlayer:
    """Represents a game player using a shared learning agent."""

    def __init__(self, name: str, shared_agent: BaseAgent):
        self.name = name
        self.shared_agent = shared_agent

    def act(self, state: tf.Tensor, env_state):
        """Delegate action selection to the shared agent."""
        return self.shared_agent.act(state, env_state)

# ============================================================
# Shared Helper Functions
# ============================================================

def create_reward_history(player_names):
    """Initialize reward histories for all players."""
    return {name: [] for name in player_names}

def create_episode_rewards(player_names):
    """Initialize cumulative rewards for a single episode."""
    return {name: 0.0 for name in player_names}

def record_episode_rewards(agent_rewards, total_rewards, players):
    """Append each player's episode reward to its history."""
    for player in players:
        agent_rewards[player.name].append(total_rewards[player.name])

def update_episode_rewards(total_rewards, step_rewards, players):
    """Accumulate the step rewards of all learning players."""
    for player in players:
        total_rewards[player.name] += step_rewards[player.name]

def print_training_summary(agent_type, num_episodes, total_steps, training_start):
    """Display overall training performance statistics."""

    total_time = time.perf_counter() - training_start

    print(f"\n{agent_type} Training Summary")
    print("-" * 60)
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Episodes: {num_episodes}")
    print(f"Total steps: {total_steps}")
    print(
        f"Episodes/sec: "
        f"{num_episodes / max(total_time, 1e-9):.2f}"
    )
    print(
        f"Steps/sec: "
        f"{total_steps / max(total_time, 1e-9):.2f}"
    )

def update_training_progress(
    progress_bar,
    episode,
    episode_start,
    training_start,
    total_steps,
):
    """Update the progress bar with training speed and timing."""

    elapsed = time.perf_counter() - training_start
    episode_time = time.perf_counter() - episode_start

    episodes_per_sec = (episode + 1) / max(elapsed, 1e-9)
    steps_per_sec = total_steps / max(elapsed, 1e-9)

    progress_bar.set_postfix({
        "ep/s": f"{episodes_per_sec:.2f}",
        "steps/s": f"{steps_per_sec:.1f}",
        "ep_time": f"{episode_time:.2f}s",
        "elapsed": f"{elapsed / 60:.1f}m",
    })

def set_random_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        tf.keras.utils.set_random_seed(seed)

# ============================================================
# Agent Initialization Helpers
# ============================================================

def get_agent_class(agent_type):
    """Return the agent class corresponding to its name."""

    if agent_type == "DQN":
        AgentClass = DQNAgent
    elif agent_type == "DDQN":
        AgentClass = DDQNAgent
    elif agent_type == "DuelingDQN":
        AgentClass = DuelingDQNAgent
    elif agent_type == "A2C":
        AgentClass = A2CAgent
    elif agent_type == "A3C":
        AgentClass = A3CAgent
    elif agent_type == "PPO":
        AgentClass = PPOAgent
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    return AgentClass

def create_learning_players(env, agent_type, player_names,**agent_params):
    """
    Create one shared learning agent and four player wrappers.
    All players use the same underlying model and replay memory.
    """
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    AgentClass = get_agent_class(agent_type)

    shared_agent = AgentClass(f"shared_{agent_type}", state_size, action_size, **agent_params)
    
    players = [
        SharedAgentPlayer(name, shared_agent)
        for name in player_names
    ]

    return shared_agent, players, state_size

# ============================================================
# Replay Memory Helpers
# ============================================================

def prepare_replay_batch(minibatch, state_size):
    """
    Convert sampled experiences into TensorFlow tensors.
    Preserve the original NumPy array construction, reshaping,
    data types, and transition ordering.
    """

    states = tf.convert_to_tensor(
        np.array([t[0] for t in minibatch]).reshape(-1, state_size),
        dtype=tf.float32,
    )

    actions = tf.convert_to_tensor(
        np.array([t[1] for t in minibatch]),
        dtype=tf.int64,
    )

    rewards = tf.convert_to_tensor(
        np.array([t[2] for t in minibatch]),
        dtype=tf.float32,
    )

    next_states = tf.convert_to_tensor(
        np.array([t[3] for t in minibatch]).reshape(-1, state_size),
        dtype=tf.float32,
    )

    dones = tf.convert_to_tensor(
        np.array([t[4] for t in minibatch]),
        dtype=tf.float32,
    )

    return states, actions, rewards, next_states, dones

def prepare_actor_critic_batch(rollout, state_size):
    """Convert fresh actor-critic transitions into training tensors."""
    states = tf.convert_to_tensor(np.array([t[0] for t in rollout]).reshape(-1, state_size), dtype=tf.float32)
    actions = tf.convert_to_tensor([t[1] for t in rollout], dtype=tf.int64)
    rewards = tf.convert_to_tensor([t[2] for t in rollout], dtype=tf.float32)
    next_states = tf.convert_to_tensor(np.array([t[3] for t in rollout]).reshape(-1, state_size), dtype=tf.float32)
    dones = tf.convert_to_tensor([t[4] for t in rollout], dtype=tf.float32)

    return states, actions, rewards, next_states, dones

def train_replay_batch(shared_agent, batch_size, state_size):
    """
    Sample experiences uniformly from replay memory and
    perform one training update.
    """

    minibatch = random.sample(
        shared_agent.memory,
        batch_size,
    )

    states, actions, rewards, next_states, dones = (
        prepare_replay_batch(minibatch, state_size)
    )

    shared_agent.replay(
        states,
        actions,
        rewards,
        next_states,
        dones,
    )

def update_agent_after_episode(shared_agent, agent_type, episode):
    """
    Apply epsilon decay and periodically save weights
    and synchronize the target network.

    Preserve the original update order and frequency.
    """

    shared_agent.epsilon = max(
        shared_agent.epsilon * shared_agent.epsilon_decay,
        shared_agent.epsilon_min,
    )

    if episode % 500 == 0:
        shared_agent.save(f"shared_{agent_type}.weights.h5")
        shared_agent.update_target_model()

# ============================================================
# Random Agent Simulation
# ============================================================

def play_random_games(env, num_episodes = 1000, max_steps = 500, seed = None):
    """Run games using four random agents and record results."""

    set_random_seed(seed)

    # Create independent random players.
    player_names = ["A", "B", "C", "D"]

    agents = [
        RandomAgent(name)
        for name in player_names
    ]

    # Initialize experiment results.
    agent_rewards = (
        create_reward_history(player_names)
    )

    steps_per_episode = []

    # Track training performance.
    training_start = time.perf_counter()
    total_steps = 0

    progress_bar = tqdm(
        range(num_episodes),
        desc="Random Agent",
        unit="episode",
    )

    for episode in progress_bar:
        episode_start = time.perf_counter()

        # Start a new game.
        obs, _ = env.reset(seed=seed + episode if seed is not None else None)

        total_rewards = (create_episode_rewards(player_names))

        # Play until termination or the step limit.
        for step in range(max_steps):
            current_agent = agents[env.current_player_index]

            action = current_agent.choose_action(env)

            obs, step_rewards, done, _, info = env.step(action)

            # Preserve original random-baseline accounting:
            # only accumulate the acting agent's reward.
            agent_name = current_agent.name
            total_rewards[agent_name] += step_rewards[agent_name]

            if done:
                break

        # Identify the remaining non-eliminated player.
        # Retained even though the result is not currently used.
        winner: str | None = next(
            (
                player.letter
                for player in env.players
                if not player.eliminated
            ),
            None,
        )

        # Store cumulative rewards for all random agents.
        record_episode_rewards(
            agent_rewards,
            total_rewards,
            agents,
        )

        # Preserve original behavior: record every episode's
        # length, including episodes that reach max_steps.
        steps_per_episode.append(step + 1)

    env.close()

    return agent_rewards, steps_per_episode

# ============================================================
# Reinforcement Learning Training
# ============================================================

def train_value_based_agents(
    env,
    agent_type = "DQN",
    num_episodes = 1000,
    max_steps = 200,
    batch_size = 32,
    seed = None,
    **agent_params,
):
    """
    Train four game players using a single shared agent.

    The original training procedure is retained for all
    supported agent types, including replay memory, epsilon
    decay, and periodic target-network updates.
    """

    set_random_seed(seed)

    # --------------------------------------------------------
    # 1. Initialize the shared agent and players
    # --------------------------------------------------------

    player_names: list[str] = ["A", "B", "C", "D"]

    shared_agent, players, state_size = create_learning_players(
        env,
        agent_type,
        player_names,
        **agent_params,
    )

    # Initialize experiment results.
    agent_rewards = (
        create_reward_history(player_names)
    )

    steps_per_episode = []

    # Track training performance.
    training_start = time.perf_counter()
    total_steps = 0

    progress_bar = tqdm(
        range(num_episodes),
        desc=f"{agent_type} Training",
        unit="episode",
    )

    # --------------------------------------------------------
    # 2. Run training episodes
    # --------------------------------------------------------

    for episode in progress_bar:
        episode_start = time.perf_counter()

        # Reset and reshape the initial observation.
        obs, _ = env.reset(seed=seed + episode if seed is not None else None)
        obs = np.reshape(obs, (1, state_size))

        # Reset episode rewards.
        total_rewards = (create_episode_rewards(player_names))

        # ----------------------------------------------------
        # 3. Interact with the environment
        # ----------------------------------------------------

        for step in range(max_steps):
            total_steps += 1

            # Identify the current player.
            current_player = players[env.current_player_index]

            # Select an action using the shared agent.
            action = current_player.act(
                tf.convert_to_tensor(obs, dtype=tf.float32),
                env.state,
            )

            # Execute the action and observe the transition.
            next_obs, step_rewards, done, _, info = env.step(action)
            next_obs = np.reshape(next_obs, (1, state_size))

            # Preserve the original reward used for learning:
            # the reward associated with the acting player.
            reward = step_rewards[current_player.name]

            # Store the experience in shared replay memory.
            shared_agent.remember(
                obs,
                action,
                reward,
                next_obs,
                done,
                env.steps_num,
            )

            # Move to the next observation.
            obs = next_obs

            # Accumulate each player's reported reward.
            update_episode_rewards(
                total_rewards,
                step_rewards,
                players,
            )

            # ------------------------------------------------
            # 4. Train from replay memory
            # ------------------------------------------------

            # Preserve the original replay condition:
            # - Memory must contain more than batch_size samples.
            # - Training occurs at steps 0, 10, 20, ...
            if (
                len(shared_agent.memory) > batch_size
                and step % 10 == 0
            ):
                train_replay_batch(
                    shared_agent,
                    batch_size,
                    state_size,
                )

            # Preserve the original episode-length behavior:
            # record lengths only when the game terminates.
            if done:
                steps_per_episode.append(step + 1)
                break

        # ----------------------------------------------------
        # 5. Complete the episode
        # ----------------------------------------------------

        # Save each player's cumulative episode reward.
        record_episode_rewards(
            agent_rewards,
            total_rewards,
            players,
        )

        # Decay epsilon, save weights, and update target network
        # according to the original schedule.
        update_agent_after_episode(
            shared_agent,
            agent_type,
            episode,
        )

        # Update progress-bar statistics.
        update_training_progress(
            progress_bar,
            episode,
            episode_start,
            training_start,
            total_steps,
        )

    # --------------------------------------------------------
    # 6. Print results and close the environment
    # --------------------------------------------------------

    print_training_summary(
        agent_type,
        num_episodes,
        total_steps,
        training_start,
    )

    env.close()

    return agent_rewards, steps_per_episode


def train_actor_critic_agents(env, agent_type = "A2C", num_episodes = 1000, max_steps = 200, batch_size = 32, seed = None, **agent_params):
    """Train four players using a shared on-policy actor-critic agent."""

    set_random_seed(seed)

    # Initialize the shared agent and players.
    player_names = ["A", "B", "C", "D"]
    shared_agent, players, state_size = create_learning_players(env, agent_type, player_names, **agent_params)

    agent_rewards = create_reward_history(player_names)
    steps_per_episode = []
    training_start = time.perf_counter()
    total_steps = 0

    progress_bar = tqdm(range(num_episodes), desc=f"{agent_type} Training", unit="episode")

    # Run training episodes.
    for episode in progress_bar:
        episode_start = time.perf_counter()
        obs, _ = env.reset(seed=seed + episode if seed is not None else None)
        obs = np.reshape(obs, (1, state_size))
        total_rewards = create_episode_rewards(player_names)

        # Store fresh transitions for the current policy.
        rollout = []

        for step in range(max_steps):
            total_steps += 1
            current_player = players[env.current_player_index]

            # Select an action from the actor's policy.
            action = current_player.act(tf.convert_to_tensor(obs, dtype=tf.float32), env.state)

            # Execute the action.
            next_obs, step_rewards, done, _, info = env.step(int(action))
            if episode >= num_episodes - 5:
                print(f"Step {step}: action={action}, rewards={step_rewards}, done={done}")

            next_obs = np.reshape(next_obs, (1, state_size))
            reward = step_rewards[current_player.name]

            # Collect an on-policy transition.
            rollout.append((obs, int(action), reward, next_obs, done))
            obs = next_obs

            # Accumulate rewards for all players.
            update_episode_rewards(total_rewards, step_rewards, players)

            # Train on fresh experiences after collecting a rollout.
            if len(rollout) >= batch_size or done:
                states, actions, rewards, next_states, dones = prepare_actor_critic_batch(rollout, state_size)
                shared_agent.replay(states, actions, rewards, next_states, dones)
                rollout.clear()

            if done:
                steps_per_episode.append(step + 1)
                break

        # Train on any remaining transitions.
        if rollout:
            states, actions, rewards, next_states, dones = prepare_actor_critic_batch(rollout, state_size)
            shared_agent.replay(states, actions, rewards, next_states, dones)

        # Record episode results.
        record_episode_rewards(agent_rewards, total_rewards, players)

        # Update progress-bar statistics.
        update_training_progress(progress_bar, episode, episode_start, training_start, total_steps)

    # Print results and close the environment.
    print_training_summary(agent_type, num_episodes, total_steps, training_start)
    env.close()

    return agent_rewards, steps_per_episode
