from chip_game_env import ChipGameEnv
from simulation import play_random_games, train_agents
from analysis import print_agent_statistics, plot_agent_statistics


if __name__ == "__main__":
    # Experimental settings.
    NUM_EPISODES: int = 100
    MAX_STEPS: int = 200
    BATCH_SIZE: int = 32

    # Create the game environment.
    env: ChipGameEnv = ChipGameEnv()

    # Train the shared agent.
    agent_rewards, steps_per_episode = train_agents(
        env=env,
        agent_type="A2C",
        num_episodes=NUM_EPISODES,
        max_steps=MAX_STEPS,
        batch_size=BATCH_SIZE
    )

    # Print summary statistics from the completed experiment.
    print_agent_statistics(
        f"A2C Performance",
        agent_rewards,
        steps_per_episode
    )

    # Plot agent stats for random
    plot_agent_statistics(agent_rewards, steps_per_episode, NUM_EPISODES, "A2C", f"data/A2C_performance.png")