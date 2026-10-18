import os
import random
import numpy as np
import tensorflow as tf

from chip_game_env import ChipGameEnv
from simulation import play_random_games, train_value_based_agents, train_actor_critic_agents
from analysis import print_agent_statistics, plot_agent_statistics


# ============================================================
# Experiment Helpers
# ============================================================

def configure_gpu(use_gpu):
    """Disable GPU visibility when requested."""
    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def run_simulation(env, agent_type, num_episodes, max_steps, batch_size, seed = None, **agent_params):
    """Run random games or train the selected agent."""
    if agent_type == "Random":
        agent_rewards, steps_per_episode = play_random_games(env, num_episodes, max_steps, seed=seed)
    elif agent_type in ["DQN", "DDQN", "DuelingDQN"]:
        agent_rewards, steps_per_episode = train_value_based_agents(env, agent_type, num_episodes, max_steps, batch_size, seed=seed, **agent_params)
    elif agent_type in ["A2C"]:
        agent_rewards, steps_per_episode = train_actor_critic_agents(env, agent_type, num_episodes, max_steps, batch_size, seed=seed, **agent_params)

    return agent_rewards, steps_per_episode


def display_experiment_results(
    agent_type, agent_rewards, steps_per_episode,
    num_episodes, plot, render_plots, print_stats
):
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

    return agent_rewards, steps_per_episode

def run_multiple_trials(
    agent_type, num_trials, num_episodes, max_steps, batch_size,
    plot=True, render_plots=False, print_stats=True,
    use_gpu=False, base_seed=42
):
    """Run independent trials with fresh agents and environments."""
    results = []

    for trial in range(num_trials):
        seed = base_seed + trial if base_seed is not None else None

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            tf.keras.utils.set_random_seed(seed)

        tf.keras.backend.clear_session()

        print(f"\n{'=' * 60}")
        print(f"{agent_type} - Trial {trial + 1}/{num_trials}")
        print(f"Seed: {seed}")
        print(f"{'=' * 60}")

        agent_rewards, steps_per_episode = run_agent_experiment(
            agent_type=agent_type,
            num_episodes=num_episodes,
            max_steps=max_steps,
            batch_size=batch_size,
            plot=False,
            print_stats=False,
            use_gpu=use_gpu
        )

        total_rewards = np.sum(list(agent_rewards.values()), axis=0)

        if print_stats:
            print_agent_statistics(
                f"{agent_type} - Trial {trial + 1}",
                agent_rewards, steps_per_episode
            )

        if plot:
            os.makedirs("data", exist_ok=True)
            plot_agent_statistics(
                agent_rewards, steps_per_episode, num_episodes,
                agent_type,
                f"data/{agent_type}_trial_{trial + 1}_performance.png",
                render_plots=render_plots
            )

        results.append({
            "trial": trial + 1,
            "seed": seed,
            "agent_rewards": agent_rewards,
            "steps_per_episode": steps_per_episode,
            "mean_reward": np.mean(total_rewards),
            "mean_steps": np.mean(steps_per_episode) if steps_per_episode else np.nan
        })

    print(f"\n{'=' * 60}")
    print(f"{agent_type} - Summary Across {num_trials} Trials")
    print(f"{'=' * 60}")

    rewards = np.array([result["mean_reward"] for result in results])
    steps = np.array([result["mean_steps"] for result in results])

    print(f"Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Steps:  {np.nanmean(steps):.2f} ± {np.nanstd(steps):.2f}")

    return results

# ============================================================
# Main Experiment Configuration
# ============================================================
if __name__ == "__main__":
    NUM_TRIALS = 5
    NUM_EPISODES = 10000
    MAX_STEPS = 200
    BATCH_SIZE = 32
    SEED = 42

    AGENT_TYPES = ["DQN"]

    all_results = {}

    for agent in AGENT_TYPES:
        all_results[agent] = run_multiple_trials(
            agent_type=agent,
            num_trials=NUM_TRIALS,
            num_episodes=NUM_EPISODES,
            max_steps=MAX_STEPS,
            batch_size=BATCH_SIZE,
            use_gpu=False,
            plot=True,
            render_plots=False,
            print_stats=True,
            base_seed=SEED
        )