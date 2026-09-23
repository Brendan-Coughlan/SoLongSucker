import random
from chip_game_env import ChipGameEnv, GameState
from pathlib import Path
import numpy as np

def play_game_with_multiple_agents(
    env: ChipGameEnv,
    agent_rewards: dict[str, list[float]],
    steps_per_episode: list[int],
    num_episodes: int = 10000,
    max_steps: int = 500,
    render: bool = True,
    log_file_name: str = "data/game_log.txt",
) -> None:
    agents: list[RandomAgent] = [RandomAgent(name) for name in ["A", "B", "C", "D"]]

    Path("data").mkdir(parents=True, exist_ok=True)
    with open(log_file_name, "w") as logfile:
        for episode in range(num_episodes):
            obs, _ = env.reset()

            total_rewards: dict[str, float] = {agent.name: 0.0 for agent in agents}

            for step in range(max_steps):
                if render:
                    env.render()

                current_agent: RandomAgent = agents[env.current_player_index]

                action: int = current_agent.choose_action(env)

                obs, step_rewards, done, _, info = env.step(action)

                step_log: list[str] = info["log"]

                for log in step_log:
                    logfile.write(log)

                agent_name: str = current_agent.name
                total_rewards[agent_name] += step_rewards[agent_name]

                # Log cumulative rewards after each step
                cumulative_rewards_log: str = f"Cumulative Rewards: {total_rewards}\n"
                logfile.write(cumulative_rewards_log)

                if done:
                    break

            winner: str | None = next(
                (player.letter for player in env.players if not player.eliminated),
                None,
            )

            print(f"Episode {episode + 1}: " f"Winner: {winner} " f"Steps = {step + 1}")

            for agent in agents:
                print(
                    f"  Agent {agent.name}: "
                    f"Total Reward = {total_rewards[agent.name]}"
                )
                agent_rewards[agent.name].append(total_rewards[agent.name])

            steps_per_episode.append(step + 1)

    env.close()


class RandomAgent:
    def __init__(self, name: str) -> None:
        self.name: str = name

    def choose_action(self, env: ChipGameEnv) -> int:
        if env.state == GameState.CHOOSE_PILE:
            return random.randint(0, env.NUM_PILES - 1)

        return random.randint(
            env.NUM_PILES,
            env.action_space.n - 1,
        )

def print_agent_statistics(
    agent_rewards: dict[str, list[float]],
    steps_per_episode: list[int],
) -> None:
    rewards = np.array(
        list(agent_rewards.values()),
        dtype=float,
    )

    # Total reward across all four agents for each episode
    episode_rewards = rewards.sum(axis=0)

    steps = np.array(
        steps_per_episode,
        dtype=float,
    )

    print("\nRandom Agent Performance")
    print("-" * 60)

    print(
        f"Reward: "
        f"{np.mean(episode_rewards):.2f} ± "
        f"{np.std(episode_rewards):.2f} "
        f"[{np.min(episode_rewards):.2f}, "
        f"{np.max(episode_rewards):.2f}]"
    )

    print(
        f"Steps:  "
        f"{np.mean(steps):.2f} ± "
        f"{np.std(steps):.2f} "
        f"[{np.min(steps):.0f}, "
        f"{np.max(steps):.0f}]"
    )