# ============================================================
# Environment Configuration
# ============================================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ============================================================
# Imports
# ============================================================

import pickle
import random
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from base_agent import BaseAgent
from chip_game_env import ChipGameEnv
from analysis import print_agent_statistics, plot_agent_statistics

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
# Experiment Helper Functions
# ============================================================

def configure_gpu(use_gpu):
    """Disable GPU visibility when requested."""
    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
    """Set the random seed for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        tf.keras.utils.set_random_seed(seed)

def save_agent_results(agent_type, results, directory="data/results"):
    """Save an agent's trial results to disk."""
    os.makedirs(directory, exist_ok=True)
    
    file_path = os.path.join(directory, f"{agent_type}_results.pkl")

    with open(file_path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_agent_results(agent_type, directory="data/results"):
    """Load previously saved trial results, if available."""
    file_path = os.path.join(directory, f"{agent_type}_results.pkl")

    if not os.path.exists(file_path):
        return None

    with open(file_path, "rb") as f:
        results = pickle.load(f)

    return results

def add_agent_results(all_results, agent_type, results):
    """Append completed trial results and save them."""
    if agent_type not in all_results:
        all_results[agent_type] = []

    start_trial = len(all_results[agent_type])

    for i, result in enumerate(results):
        result["trial"] = start_trial + i + 1

    all_results[agent_type].extend(results)

    save_agent_results(agent_type, all_results[agent_type])

    return all_results

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
            update_episode_rewards(total_rewards, step_rewards, agents)

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
                break

            steps_per_episode.append(step + 1)

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
            # if episode >= num_episodes - 5:
            #     print(f"Step {step}: action={action}, rewards={step_rewards}, done={done}")

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
                break

            steps_per_episode.append(step + 1)

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


def train_a3c_agents(
    agent_type="A3C",
    num_episodes=1000,
    max_steps=200,
    batch_size=32,
    num_workers=4,
    seed=None,
    **agent_params
):
    """Train A3C with independent workers and shared global parameters."""
    
    set_random_seed(seed)

    player_names = ["A", "B", "C", "D"]

    sample_env = ChipGameEnv()
    state_size = sample_env.observation_space.shape[0]
    action_size = sample_env.action_space.n
    sample_env.close()

    global_agent = A3CAgent(
        "global_A3C", state_size, action_size, **agent_params
    )

    global_agent.optimizer._create_all_weights(
        global_agent.model.trainable_variables
    )
    
    update_lock = threading.Lock()
    results_lock = threading.Lock()

    agent_rewards = create_reward_history(player_names)
    steps_per_episode = [0] * num_episodes
    episode_rewards = [None] * num_episodes

    training_start = time.perf_counter()

    def worker(worker_id, episode_ids):
        env = ChipGameEnv()

        local_agent = A3CAgent(
            f"worker_{worker_id}", state_size, action_size, **agent_params
        )

        with update_lock:
            local_agent.model.set_weights(global_agent.model.get_weights())

        players = [
            SharedAgentPlayer(name, local_agent)
            for name in player_names
        ]

        for episode in episode_ids:
            obs, _ = env.reset(
                seed=seed + episode if seed is not None else None
            )
            obs = np.reshape(obs, (1, state_size))

            total_rewards = create_episode_rewards(player_names)
            rollout = []

            for step in range(max_steps):
                current_player = players[env.current_player_index]

                action = int(current_player.act(
                    tf.convert_to_tensor(obs, dtype=tf.float32),
                    env.state
                ))

                next_obs, step_rewards, done, _, info = env.step(action)
                next_obs = np.reshape(next_obs, (1, state_size))

                reward = step_rewards[current_player.name]

                rollout.append((obs, action, reward, next_obs, done))

                obs = next_obs

                update_episode_rewards(
                    total_rewards, step_rewards, players
                )

                if len(rollout) >= batch_size or done:
                    states, actions, rewards, next_states, dones = (
                        prepare_actor_critic_batch(rollout, state_size)
                    )

                    gradients = local_agent.compute_gradients(
                        states, actions, rewards, next_states, dones
                    )

                    with update_lock:
                        global_agent.optimizer.apply_gradients(
                            zip(
                                gradients,
                                global_agent.model.trainable_variables
                            )
                        )
                        local_agent.model.set_weights(
                            global_agent.model.get_weights()
                        )

                    rollout.clear()

                if done:
                    break

            if rollout:
                states, actions, rewards, next_states, dones = (
                    prepare_actor_critic_batch(rollout, state_size)
                )

                gradients = local_agent.compute_gradients(
                    states, actions, rewards, next_states, dones
                )

                with update_lock:
                    global_agent.optimizer.apply_gradients(
                        zip(
                            gradients,
                            global_agent.model.trainable_variables
                        )
                    )
                    local_agent.model.set_weights(
                        global_agent.model.get_weights()
                    )

            with results_lock:
                episode_rewards[episode] = total_rewards
                steps_per_episode[episode] = step + 1

        env.close()

    episode_groups = [
        list(range(worker_id, num_episodes, num_workers))
        for worker_id in range(num_workers)
    ]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(worker, worker_id, episode_ids)
            for worker_id, episode_ids in enumerate(episode_groups)
            if episode_ids
        ]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="A3C Workers",
            unit="worker"
        ):
            future.result()

    for rewards in episode_rewards:
        for name in player_names:
            agent_rewards[name].append(rewards[name])

    print_training_summary(
        agent_type,
        num_episodes,
        sum(steps_per_episode),
        training_start
    )

    return agent_rewards, steps_per_episode

# ============================================================
# Main Simulation Functions
# ============================================================

def run_simulation(env, agent_type, num_episodes, max_steps, batch_size, seed = None, **agent_params):
    """Run random games or train the selected agent."""
    if agent_type == "Random":
        agent_rewards, steps_per_episode = play_random_games(env, num_episodes, max_steps, seed=seed)
    elif agent_type in ["DQN", "DDQN", "DuelingDQN"]:
        agent_rewards, steps_per_episode = train_value_based_agents(env, agent_type, num_episodes, max_steps, batch_size, seed=seed, **agent_params)
    elif agent_type in ["A2C"]:
        agent_rewards, steps_per_episode = train_actor_critic_agents(env, agent_type, num_episodes, max_steps, batch_size, seed=seed, **agent_params)
    elif agent_type in ["A3C"]:
        env.close()
        agent_rewards, steps_per_episode = train_a3c_agents(
            agent_type=agent_type,
            num_episodes=num_episodes,
            max_steps=max_steps,
            batch_size=batch_size,
            num_workers=4,
            seed=seed
        )
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")

    return agent_rewards, steps_per_episode


def run_agent_experiment(
    agent_type, num_episodes, max_steps: int, batch_size: int,
    plot = True, print_stats = True, use_gpu = False, seed = None
):
    """Configure, run, and report results for one agent experiment."""

    # Configure hardware and initialize environment.
    configure_gpu(use_gpu)
    env: ChipGameEnv = ChipGameEnv()

    # Run the experiment.
    agent_rewards, steps_per_episode = run_simulation(
        env=env, agent_type=agent_type, num_episodes=num_episodes,
        max_steps=max_steps, batch_size=batch_size, seed=seed
    )

    return agent_rewards, steps_per_episode

def run_multiple_trials(
    agent_type, num_trials, num_episodes, max_steps, batch_size,
    plot=True, print_trial_stats=False, print_overall_stats=True,
    use_gpu=False, base_seed=42
):
    """Run independent trials with fresh agents and environments."""
    results = []

    for trial in range(num_trials):
        seed = base_seed + trial
        set_random_seed(seed)

        tf.keras.backend.clear_session()

        if print_trial_stats:
            print(f"\n{'=' * 60}")
            print(f"{agent_type} - Trial {trial + 1}/{num_trials}")
            print(f"Seed: {seed}")
            print(f"{'=' * 60}")

        agent_rewards, steps_per_episode = run_agent_experiment(
            agent_type=agent_type,
            num_episodes=num_episodes,
            max_steps=max_steps,
            batch_size=batch_size,
            plot=plot,
            use_gpu=use_gpu,
            seed=seed
        )

        total_rewards = np.sum(list(agent_rewards.values()), axis=0)

        if print_trial_stats:
            print_agent_statistics(f"{agent_type} - Trial {trial + 1}", agent_rewards, steps_per_episode)

        if plot:
            figure_dir = "data/figures"
            os.makedirs(figure_dir, exist_ok=True)

            plot_agent_statistics(
                agent_rewards, steps_per_episode, num_episodes,
                agent_type,
                f"{figure_dir}/{agent_type}_seed_{seed}_performance.png"            )

        results.append({
            "trial": trial + 1,
            "seed": seed,
            "agent_rewards": agent_rewards,
            "steps_per_episode": steps_per_episode,
            "mean_reward": np.mean(total_rewards),
            "mean_steps": np.mean(steps_per_episode) if steps_per_episode else np.nan
        })

    if print_overall_stats:
        print(f"\n{'=' * 60}")
        print(f"{agent_type} - Summary Across {num_trials} Trials")
        print(f"{'=' * 60}")

        rewards = np.array([result["mean_reward"] for result in results])
        steps = np.array([result["mean_steps"] for result in results])

        print(f"Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"Steps:  {np.nanmean(steps):.2f} ± {np.nanstd(steps):.2f}")

    return results
