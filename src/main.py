from chip_game_env import ChipGameEnv
from random_agent import RandomAgent
from simulation import play_game_with_multiple_agents
from analysis import plot_agent_progress, print_agent_statistics

if __name__ == "__main__":
    # Set the number of games to run and the maximum length of each game.
    NUM_EPISODES: int = 10000
    MAX_STEPS: int = 500

    # Create the chip game environment shared by the agents.
    env: ChipGameEnv = ChipGameEnv()

    # Store the reward history for each agent across all episodes.
    agent_rewards: dict[str, list[float]] = {
        "A": [],
        "B": [],
        "C": [],
        "D": [],
    }

    # Create the random agents
    agents: list[RandomAgent] = [RandomAgent(name) for name in agent_rewards.keys()]

    # Store how many steps each episode takes to finish.
    steps_per_episode: list[int] = []

    # Run the simulation and collect agent rewards and episode lengths.
    play_game_with_multiple_agents(
        env=env,
        agents=agents,
        num_episodes=NUM_EPISODES,
        max_steps=MAX_STEPS,
        render=True,
        agent_rewards=agent_rewards,
        steps_per_episode=steps_per_episode,
    )

    # Print summary statistics from the completed simulation.
    print_agent_statistics(
        agent_rewards,
        steps_per_episode,
    )

    # Plot agent rewards and episode lengths over the experiment.
    plot_agent_progress(
        agent_rewards=agent_rewards,
        steps_per_episode=steps_per_episode,
        num_episodes=NUM_EPISODES,
        experiment_name="Random Agent Performance",
        plot_file_name="data/random_agent_baseline.png"
    )
