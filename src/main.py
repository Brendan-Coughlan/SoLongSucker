from chip_game_env import ChipGameEnv
from simulation import play_random_games, train_agents
from analysis import print_agent_statistics, plot_agent_progress, plot_agent_statistics


if __name__ == "__main__":
    # Experimental settings.
    NUM_EPISODES: int = 10000
    MAX_STEPS: int = 200
    BATCH_SIZE: int = 32

    # Create the game environment.
    env: ChipGameEnv = ChipGameEnv()

    # Train the random agent baseline
    # agent_rewards, steps_per_episode = play_random_games(
    #     env=env,
    #     num_episodes=NUM_EPISODES,
    #     max_steps=MAX_STEPS
    # )

    # Train the shared DQN agent.
    agent_rewards, steps_per_episode = train_agents(
        env=env,
        agent_type="DQN",
        num_episodes=NUM_EPISODES,
        max_steps=MAX_STEPS,
        batch_size=BATCH_SIZE
    )

    # Print summary statistics from the completed experiment.
    print_agent_statistics(
        "DQN Performance",
        agent_rewards,
        steps_per_episode
    )

    # plot_agent_progress(agent_rewards, steps_per_episode, NUM_EPISODES, "DQN", "data/dqn_agent.png")
    plot_agent_statistics(agent_rewards, steps_per_episode, NUM_EPISODES, "DQN", "data/dqn_agent.png")