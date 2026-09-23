import random
from chip_game_env import ChipGameEnv, GameState


def play_game_with_multiple_agents(
    env: ChipGameEnv,
    num_episodes: int = 1,
    max_steps: int = 100,
    render: bool = True,
) -> None:
    agents: list[RandomAgent] = [RandomAgent(name) for name in ["A", "B", "C", "D"]]

    with open("game_log.txt", "w") as logfile:
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

                # Update total rewards for all agents
                for agent in agents:
                    total_rewards[agent.name] += step_rewards[agent.name]

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
