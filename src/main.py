
# ============================================================
# Imports
# ============================================================

import os

from chip_game_env import ChipGameEnv
from simulation import play_random_games, train_agents
from analysis import print_agent_statistics, plot_agent_statistics


# ============================================================
# Experiment Helper Functions
# ============================================================

def configure_gpu(use_gpu: bool) -> None:
    """
    Configure GPU visibility for the experiment.

    Preserve the original behavior:
    disable GPU visibility when use_gpu is False.
    """
    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def run_simulation(
    env: ChipGameEnv,
    agent_type: str,
    num_episodes: int,
    max_steps: int,
    batch_size: int,
):
    """
    Run the selected experiment.

    Random agents play games without training.
    All other agent types use the shared-agent training
    procedure defined in simulation.py.
    """

    if agent_type == "Random":

        # Run the random baseline.
        agent_rewards, steps_per_episode = play_random_games(
            env,
            num_episodes,
            max_steps,
        )

    else:

        # Train the selected reinforcement learning agent.
        agent_rewards, steps_per_episode = train_agents(
            env=env,
            agent_type=agent_type,
            num_episodes=num_episodes,
            max_steps=max_steps,
            batch_size=batch_size,
        )

    return agent_rewards, steps_per_episode


def display_experiment_results(
    agent_type: str,
    agent_rewards,
    steps_per_episode,
    num_episodes: int,
    plot: bool,
    render_plots: bool,
    print_stats: bool,
) -> None:
    """
    Display experiment statistics and generate plots
    according to the selected options.
    """

    # Print the experiment's summary statistics.
    if print_stats:
        print_agent_statistics(
            f"{agent_type} Performance",
            agent_rewards,
            steps_per_episode,
        )

    # Generate the performance plot.
    if plot:
        plot_agent_statistics(
            agent_rewards,
            steps_per_episode,
            num_episodes,
            agent_type,
            f"data/{agent_type}_performance.png",
            render_plots=render_plots,
        )


# ============================================================
# Run a Single Agent Experiment
# ============================================================

def run_agent_experiment(
    agent_type: str,
    num_episodes: int,
    max_steps: int,
    batch_size: int,
    plot: bool = True,
    render_plots: bool = True,
    print_stats: bool = True,
    use_gpu: bool = False,
):
    """
    Execute a complete experiment for one agent type.

    Steps:
        1. Configure GPU visibility.
        2. Create the game environment.
        3. Run the selected agent simulation.
        4. Print statistics and/or generate plots.
    """

    # --------------------------------------------------------
    # 1. Configure hardware
    # --------------------------------------------------------

    configure_gpu(use_gpu)

    # --------------------------------------------------------
    # 2. Create the game environment
    # --------------------------------------------------------

    env: ChipGameEnv = ChipGameEnv()

    # --------------------------------------------------------
    # 3. Run the experiment
    # --------------------------------------------------------

    agent_rewards, steps_per_episode = run_simulation(
        env=env,
        agent_type=agent_type,
        num_episodes=num_episodes,
        max_steps=max_steps,
        batch_size=batch_size,
    )

    # --------------------------------------------------------
    # 4. Display and save results
    # --------------------------------------------------------

    display_experiment_results(
        agent_type=agent_type,
        agent_rewards=agent_rewards,
        steps_per_episode=steps_per_episode,
        num_episodes=num_episodes,
        plot=plot,
        render_plots=render_plots,
        print_stats=print_stats,
    )


# ============================================================
# Main Experiment Configuration
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # 1. Experimental settings
    # --------------------------------------------------------

    NUM_EPISODES: int = 2500
    MAX_STEPS: int = 200
    BATCH_SIZE: int = 32

    # --------------------------------------------------------
    # 2. Select agents to evaluate
    # --------------------------------------------------------

    AGENT_TYPES = ["DQN"]

    # --------------------------------------------------------
    # 3. Run the experiments
    # --------------------------------------------------------

    for agent in AGENT_TYPES:
        run_agent_experiment(
            agent_type=agent,
            num_episodes=NUM_EPISODES,
            max_steps=MAX_STEPS,
            batch_size=BATCH_SIZE,
            use_gpu=False,
            plot=True,
            render_plots=False,
            print_stats=True,
        )
