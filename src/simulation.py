import random

from pathlib import Path
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from base_agent import BaseAgent
from chip_game_env import ChipGameEnv
from dqn_agent import DQNAgent
from ddqn_agent import DDQNAgent
from random_agent import RandomAgent


class SharedAgentPlayer:
    """Represents a game player using a shared learning agent."""

    def __init__(self, name: str, shared_agent: BaseAgent) -> None:
        self.name = name
        self.shared_agent = shared_agent

    def act(self, state: tf.Tensor, env_state) -> tf.Tensor:
        return self.shared_agent.act(state, env_state)


def play_random_games(
    env: ChipGameEnv,
    num_episodes: int = 1000,
    max_steps: int = 500,
    log_file_name: str = "data/random_game_log.txt",
):
    """Run games using four random agents and record their results."""

    Path(log_file_name).parent.mkdir(parents=True, exist_ok=True)

    player_names: list[str] = ["A", "B", "C", "D"]
    agents: list[RandomAgent] = [RandomAgent(name) for name in player_names]

    agent_rewards: dict[str, list[float]] = {name: [] for name in player_names}
    steps_per_episode: list[int] = []

    with open(log_file_name, "w") as logfile:
        for episode in tqdm(range(num_episodes), desc="Random Agent", unit="episode"):
            obs, _ = env.reset()

            # Track each agent's total reward during the current episode.
            total_rewards: dict[str, float] = {name: 0.0 for name in player_names}

            for step in range(max_steps):
                current_agent: RandomAgent = agents[env.current_player_index]

                action: int = current_agent.choose_action(env)

                obs, step_rewards, done, _, info = env.step(action)

                # Record the reward received by the acting agent.
                agent_name: str = current_agent.name
                total_rewards[agent_name] += step_rewards[agent_name]

                # Save environment messages to the log.
                for log in info["log"]:
                    logfile.write(log)
                logfile.write(f"Cumulative Rewards: {total_rewards}\n")

                if done:
                    break

            # Find the remaining non-eliminated player.
            winner: str | None = next(
                (player.letter for player in env.players if not player.eliminated),
                None,
            )

            # Store final results for the episode.
            for agent in agents:
                agent_rewards[agent.name].append(total_rewards[agent.name])
    
            steps_per_episode.append(step + 1)

            # print(f"Episode {episode + 1}/{num_episodes}:\nWinner = {winner},\nSteps = {step + 1}")

    env.close()

    return agent_rewards, steps_per_episode


def train_dqn_agents(
    env: ChipGameEnv,
    num_episodes: int = 1000,
    max_steps: int = 200,
    batch_size: int = 32,
    **agent_params,
) -> tuple[dict[str, list[float]], list[int]]:
    """Train four game players using a single shared DQN agent."""
    
    state_size: int = env.observation_space.shape[0]
    action_size: int = env.action_space.n

    shared_agent = DDQNAgent(
        "Shared",
        state_size,
        action_size,
        **agent_params,
    )

    player_names: list[str] = ["A", "B", "C", "D"]

    players: list[SharedAgentPlayer] = [SharedAgentPlayer(name, shared_agent) for name in player_names]

    steps_per_episode: list[int] = []
    agent_rewards: dict[str, list[float]] = {name: [] for name in player_names}

    for episode in tqdm(range(num_episodes), desc="DQN Training", unit="episode"):
        obs, _ = env.reset()
        obs = np.reshape(obs, (1, state_size))
        total_rewards: dict[str, float] = {name: 0.0 for name in player_names}
        
        for step in range(max_steps):
            current_player = players[env.current_player_index]
            action = current_player.act(tf.convert_to_tensor(obs, dtype=tf.float32), env.state)
            next_obs, step_rewards, done, _, info = env.step(action)
            next_obs = np.reshape(next_obs, (1, state_size))

            reward = step_rewards[current_player.name]
            shared_agent.remember(obs, action, reward, next_obs, done, env.steps_num)
            obs = next_obs

            for player in players:
                total_rewards[player.name] += step_rewards[player.name]

            if len(shared_agent.memory) > batch_size and step % 10 == 0:
                minibatch = random.sample(shared_agent.memory, batch_size)
                states = tf.convert_to_tensor(np.array([t[0] for t in minibatch]).reshape(-1, state_size), dtype=tf.float32)
                actions = tf.convert_to_tensor(np.array([t[1] for t in minibatch]), dtype=tf.int64)
                rewards = tf.convert_to_tensor(np.array([t[2] for t in minibatch]), dtype=tf.float32)
                next_states = tf.convert_to_tensor(np.array([t[3] for t in minibatch]).reshape(-1, state_size), dtype=tf.float32)
                dones = tf.convert_to_tensor(np.array([t[4] for t in minibatch]), dtype=tf.float32)

                shared_agent.replay(states, actions, rewards, next_states, dones)
            
            if done:
                steps_per_episode.append(step + 1)
                break
        
        for player in players:
            agent_rewards[player.name].append(total_rewards[player.name])

        shared_agent.epsilon = max(shared_agent.epsilon * shared_agent.epsilon_decay, shared_agent.epsilon_min)

        if episode % 500 == 0:
            shared_agent.save("shared_DQN.weights.h5")
            shared_agent.update_target_model()

    env.close()

    return agent_rewards, steps_per_episode