import os

from chip_game_env import ChipGameEnv
from simulation import play_random_games, train_agents
from analysis import print_agent_statistics, plot_agent_statistics


# ============================================================
# Experiment Helpers
# ============================================================

def configure_gpu(use_gpu: bool) -> None:
    """Disable GPU visibility when requested."""
    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def run_simulation(env: ChipGameEnv, agent_type: str, num_episodes: int, max_steps: int, batch_size: int):
    """Run random games or train the selected agent."""
    if agent_type == "Random":
        agent_rewards, steps_per_episode = play_random_games(env, num_episodes, max_steps)
    else:
        agent_rewards, steps_per_episode = train_agents(
            env=env, agent_type=agent_type, num_episodes=num_episodes,
            max_steps=max_steps, batch_size=batch_size
        )

    return agent_rewards, steps_per_episode


def display_experiment_results(
    agent_type: str, agent_rewards, steps_per_episode,
    num_episodes: int, plot: bool, render_plots: bool, print_stats: bool
) -> None:
    """Print experiment statistics and generate performance plots."""
    if print_stats:
        print_agent_statistics(f"{agent_type} Performance", agent_rewards, steps_per_episode)

    if plot:
        plot_agent_statistics(
            agent_rewards, steps_per_episode, num_episodes, agent_type,
            f"data/{agent_type}_performance.png", render_plots=render_plots
        )


# ============================================================
# Run Agent Experiment
# ============================================================

def run_agent_experiment(
    agent_type: str, num_episodes: int, max_steps: int, batch_size: int,
    plot: bool = True, render_plots: bool = True,
    print_stats: bool = True, use_gpu: bool = False
):
    """Configure, run, and report results for one agent experiment."""

    # Configure hardware and initialize environment.
    configure_gpu(use_gpu)
    env: ChipGameEnv = ChipGameEnv()

    # Run the experiment.
    agent_rewards, steps_per_episode = run_simulation(
        env=env, agent_type=agent_type, num_episodes=num_episodes,
        max_steps=max_steps, batch_size=batch_size
    )

    # Print statistics and generate plots.
    display_experiment_results(
        agent_type=agent_type, agent_rewards=agent_rewards,
        steps_per_episode=steps_per_episode, num_episodes=num_episodes,
        plot=plot, render_plots=render_plots, print_stats=print_stats
    )


# ============================================================
# Main Experiment Configuration
# ============================================================

if __name__ == "__main__":
    # Experimental settings
    NUM_EPISODES: int = 2500
    MAX_STEPS: int = 200
    BATCH_SIZE: int = 32

    # Agents to evaluate
    AGENT_TYPES = ["DQN"]

    for agent in AGENT_TYPES:
        run_agent_experiment(
            agent_type=agent, num_episodes=NUM_EPISODES,
            max_steps=MAX_STEPS, batch_size=BATCH_SIZE,
            use_gpu=False, plot=True, render_plots=False, print_stats=True
        )
