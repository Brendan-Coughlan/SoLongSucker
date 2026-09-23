from chip_game_env import ChipGameEnv
from random_agent import play_game_with_multiple_agents, print_agent_statistics
from plotter import plot_agent_progress

if __name__ == "__main__":
    NUM_EPISODES: int = 10000
    MAX_STEPS: int = 500

    env: ChipGameEnv = ChipGameEnv()

    agent_rewards: dict[str, list[float]] = {
        "A": [],
        "B": [],
        "C": [],
        "D": [],
    }

    steps_per_episode: list[int] = []

    play_game_with_multiple_agents(
        env=env,
        num_episodes=NUM_EPISODES,
        max_steps=MAX_STEPS,
        render=True,
        agent_rewards=agent_rewards,
        steps_per_episode=steps_per_episode,
    )

    print_agent_statistics(
        agent_rewards,
        steps_per_episode,
    )

    plot_agent_progress(
        agent_rewards=agent_rewards,
        steps_per_episode=steps_per_episode,
        num_episodes=NUM_EPISODES,
    )
