from chip_game_env import ChipGameEnv
from simulation import play_random_games, train_agents
from analysis import print_agent_statistics, plot_agent_statistics
import os

def run_agent_experiment(agent_type: str, num_episodes: int, max_steps: int, batch_size: int, plot: bool = True, render_plots: bool = True, print_stats: bool = True, use_gpu: bool = False):    
    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU usage.   

    # Create the game environment.
    env: ChipGameEnv = ChipGameEnv()

    # Train the specified agent type.
    agent_rewards, steps_per_episode = train_agents(
        env=env,
        agent_type=agent_type,
        num_episodes=num_episodes,
        max_steps=max_steps,
        batch_size=batch_size
    )

    # Print summary statistics from the completed experiment.
    if print_stats:
        print_agent_statistics(
            f"{agent_type} Performance",
            agent_rewards,
            steps_per_episode
        )

    # Plot agent stats for the specified agent type.
    if plot:
        plot_agent_statistics(agent_rewards, steps_per_episode, num_episodes, agent_type, f"data/{agent_type}_performance.png", render_plots=render_plots)

if __name__ == "__main__":
    # Experimental settings.
    NUM_EPISODES: int = 10000
    MAX_STEPS: int = 200
    BATCH_SIZE: int = 32

    # Run experiments for different agent types.
    for agent in ["DQN", "DDQN", "DuelingDQN", "A2C", "A3C", "PPO"]:
        run_agent_experiment(agent, NUM_EPISODES, MAX_STEPS, BATCH_SIZE, use_gpu=False, plot=True, render_plots=False, print_stats=True)