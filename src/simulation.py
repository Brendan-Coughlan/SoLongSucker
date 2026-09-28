from pathlib import Path
import random
from base_agent import BaseAgent
from chip_game_env import ChipGameEnv
from dqn_agent import DQNAgent
import numpy as np
import tensorflow as tf

class Agent:
    def __init__(self, name, shared_agent):
        self.name = name
        self.shared_agent = shared_agent

    def act(self, state, env_state):
        return self.shared_agent.act(state, env_state)


def play_game_with_multiple_agents(
    env: ChipGameEnv,
    agents: list[BaseAgent],
    agent_rewards: dict[str, list[float]],
    steps_per_episode: list[int],
    num_episodes: int = 10000,
    max_steps: int = 500,
    render: bool = True,
    log_file_name: str = "data/game_log.txt",
) -> None:
    # Create the data directory if it does not already exist.
    Path("data").mkdir(parents=True, exist_ok=True)

    with open(log_file_name, "w") as logfile:
        for episode in range(num_episodes):
            obs, _ = env.reset()

            # Track each agent's total reward during the current episode.
            total_rewards: dict[str, float] = {agent.name: 0.0 for agent in agents}

            for step in range(max_steps):
                if render:
                    env.render()

                # Select the agent whose turn it currently is.
                current_agent: BaseAgent = agents[env.current_player_index]

                action: int = current_agent.choose_action(env)

                obs, step_rewards, done, _, info = env.step(action)

                # Save the environment's description of the step to the log file.
                step_log: list[str] = info["log"]

                for log in step_log:
                    logfile.write(log)

                agent_name: str = current_agent.name

                # Add this step's reward to the agent's episode total.
                total_rewards[agent_name] += step_rewards[agent_name]

                # Log cumulative rewards after each step.
                cumulative_rewards_log: str = f"Cumulative Rewards: {total_rewards}\n"
                logfile.write(cumulative_rewards_log)

                if done:
                    break

            # The remaining non-eliminated player is the winner.
            winner: str | None = next(
                (player.letter for player in env.players if not player.eliminated),
                None,
            )

            print(f"Episode {episode + 1}: " f"Winner: {winner} " f"Steps = {step + 1}")

            # Store each agent's final reward for this episode.
            for agent in agents:
                print(
                    f"  Agent {agent.name}: "
                    f"Total Reward = {total_rewards[agent.name]}"
                )

                agent_rewards[agent.name].append(total_rewards[agent.name])

            steps_per_episode.append(step + 1)

    env.close()


def play_game_with_agents(
    env, agent_type, num_episodes=1000, max_steps=200, batch_size=32, **agent_params
):
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    if agent_type == "DQN":
        AgentClass = DQNAgent
    else:
        raise ValueError("Invalid agent type")

    # Create a single shared agent
    shared_agent = AgentClass("Shared", state_size, action_size, **agent_params)

    # Create individual agents that use the shared network
    agents = [Agent(name, shared_agent) for name in ["A", "B", "C", "D"]]

    steps_per_episode = []
    agent_rewards = {name: [] for name in ["A", "B", "C", "D"]}

    for episode in range(num_episodes):
        obs, _ = env.reset()
        obs = np.reshape(obs, [1, state_size])
        total_rewards = {agent.name: 0 for agent in agents}

        for step in range(max_steps):
            current_agent = agents[env.current_player_index]
            action = current_agent.act(
                tf.convert_to_tensor(obs, dtype=tf.float32), env.state
            )
            next_obs, step_rewards, done, _, info = env.step(action)
            next_obs = np.reshape(next_obs, [1, state_size])

            reward = step_rewards[current_agent.name]
            shared_agent.remember(obs, action, reward, next_obs, done, env.steps_num)
            obs = next_obs

            for agent in agents:
                total_rewards[agent.name] += step_rewards[agent.name]

            if len(shared_agent.memory) > batch_size and step % 10 == 0:
                minibatch = random.sample(shared_agent.memory, batch_size)
                states = tf.convert_to_tensor(
                    np.array([t[0] for t in minibatch]).reshape(-1, state_size),
                    dtype=tf.float32,
                )
                actions = tf.convert_to_tensor(
                    np.array([t[1] for t in minibatch]), dtype=tf.int64
                )
                rewards = tf.convert_to_tensor(
                    np.array([t[2] for t in minibatch]), dtype=tf.float32
                )
                next_states = tf.convert_to_tensor(
                    np.array([t[3] for t in minibatch]).reshape(-1, state_size),
                    dtype=tf.float32,
                )
                dones = tf.convert_to_tensor(
                    np.array([t[4] for t in minibatch]), dtype=tf.float32
                )

                shared_agent.replay(states, actions, rewards, next_states, dones)

            if step % 10 == 0:
                print(
                    f"{agent_type} Agents -> Episode: {episode}/{num_episodes}, progress: {episode/num_episodes}"
                )

            if done:
                steps_per_episode.append(step + 1)
                break

        for agent in agents:
            agent_rewards[agent.name].append(total_rewards[agent.name])

        shared_agent.epsilon = max(
            shared_agent.epsilon * shared_agent.epsilon_decay, shared_agent.epsilon_min
        )

        if episode % 500 == 0:
            shared_agent.save(f"shared_{agent_type}.weights.h5")
            shared_agent.update_target_model()

    env.close()
