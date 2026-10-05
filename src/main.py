from chip_game_env import ChipGameEnv
from simulation import play_random_games, train_agents
from analysis import print_agent_statistics, plot_agent_statistics


if __name__ == "__main__":
    # Experimental settings.
    NUM_EPISODES: int = 10000
    MAX_STEPS: int = 200
    BATCH_SIZE: int = 32

    # Create the game environment.
    env: ChipGameEnv = ChipGameEnv()

    # Train the random agent baseline
    agent_rewards, steps_per_episode = play_random_games(
        env=env,
        num_episodes=NUM_EPISODES,
        max_steps=MAX_STEPS
    )

    # Print summary statistics from the completed experiment.
    print_agent_statistics(
        "Random Performance",
        agent_rewards,
        steps_per_episode
    )

    # Plot agent stats for random
    plot_agent_statistics(agent_rewards, steps_per_episode, NUM_EPISODES, "Random", "data/random_baseline.png")

    for agent_type in ["DQN", "DDQN", "DuelingDQN"]:
        # Train the shared agent.
        agent_rewards, steps_per_episode = train_agents(
            env=env,
            agent_type=agent_type,
            num_episodes=NUM_EPISODES,
            max_steps=MAX_STEPS,
            batch_size=BATCH_SIZE
        )

        # Print summary statistics from the completed experiment.
        print_agent_statistics(
            f"{agent_type} Performance",
            agent_rewards,
            steps_per_episode
        )
    
        # Plot agent stats for random
        plot_agent_statistics(agent_rewards, steps_per_episode, NUM_EPISODES, str(agent_type), f"data/{agent_type}_performance.png")