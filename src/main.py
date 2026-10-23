import os
import random
import numpy as np
import tensorflow as tf
import pickle

from chip_game_env import ChipGameEnv
from simulation import play_random_games, train_a3c_agents, train_value_based_agents, train_actor_critic_agents
from analysis import print_agent_statistics, plot_agent_statistics, plot_algorithm_comparison


# ============================================================
# Experiment Helpers
# ============================================================

def configure_gpu(use_gpu):
    """Disable GPU visibility when requested."""
    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def save_agent_results(agent_type, results, directory="data/results"):
    """Save an agent's trial results to disk."""
    os.makedirs(directory, exist_ok=True)

    file_path = os.path.join(directory, f"{agent_type}_results.pkl")

    with open(file_path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved {agent_type} results to {file_path}")


def load_agent_results(agent_type, directory="data/results"):
    """Load previously saved trial results, if available."""
    file_path = os.path.join(directory, f"{agent_type}_results.pkl")

    if not os.path.exists(file_path):
        return None

    with open(file_path, "rb") as f:
        results = pickle.load(f)

    print(f"Loaded {agent_type} results from {file_path}")
    return results

def add_agent_results(all_results, agent_type, results):
    """Append completed trial results and save them."""

    if agent_type not in all_results:
        all_results[agent_type] = []

    start_trial = len(all_results[agent_type])

    for i, result in enumerate(results):
        result["trial"] = start_trial + i + 1

    all_results[agent_type].extend(results)

    save_agent_results(agent_type, all_results[agent_type])

    return all_results

def run_simulation(env, agent_type, num_episodes, max_steps, batch_size, seed = None, **agent_params):
    """Run random games or train the selected agent."""
    if agent_type == "Random":
        agent_rewards, steps_per_episode = play_random_games(env, num_episodes, max_steps, seed=seed)
    elif agent_type in ["DQN", "DDQN", "DuelingDQN"]:
        agent_rewards, steps_per_episode = train_value_based_agents(env, agent_type, num_episodes, max_steps, batch_size, seed=seed, **agent_params)
    elif agent_type in ["A2C"]:
        agent_rewards, steps_per_episode = train_actor_critic_agents(env, agent_type, num_episodes, max_steps, batch_size, seed=seed, **agent_params)
    elif agent_type in ["A3C"]:
        env.close()
        agent_rewards, steps_per_episode = train_a3c_agents(
            agent_type=agent_type,
            num_episodes=num_episodes,
            max_steps=max_steps,
            batch_size=batch_size,
            num_workers=4,
            seed=seed
        )

    return agent_rewards, steps_per_episode


def display_experiment_results(
    agent_type, agent_rewards, steps_per_episode, seed,
    num_episodes, plot, render_plots, print_stats
):
    """Print experiment statistics and generate performance plots."""
    if print_stats:
        print_agent_statistics(f"{agent_type} Performance", agent_rewards, steps_per_episode)

    if plot:
        plot_agent_statistics(
            agent_rewards, steps_per_episode, num_episodes, agent_type,
            f"data/{agent_type}_seed_{seed}_performance.png", render_plots=render_plots
        )


# ============================================================
# Run Agent Experiment
# ============================================================

def run_agent_experiment(
    agent_type: str, num_episodes: int, max_steps: int, batch_size: int,
    plot: bool = True, render_plots: bool = True,
    print_stats: bool = True, use_gpu: bool = False, seed = None
):
    """Configure, run, and report results for one agent experiment."""

    # Configure hardware and initialize environment.
    configure_gpu(use_gpu)
    env: ChipGameEnv = ChipGameEnv()

    # Run the experiment.
    agent_rewards, steps_per_episode = run_simulation(
        env=env, agent_type=agent_type, num_episodes=num_episodes,
        max_steps=max_steps, batch_size=batch_size, seed=seed
    )

    # Print statistics and generate plots.
    display_experiment_results(
        agent_type=agent_type, agent_rewards=agent_rewards,
        steps_per_episode=steps_per_episode, num_episodes=num_episodes, seed=seed,
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
        seed = base_seed + trial

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
            use_gpu=use_gpu,
            seed=seed
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
# if __name__ == "__main__":
#     NUM_TRIALS = 2
#     NUM_EPISODES = 10000
#     MAX_STEPS = 200
#     BATCH_SIZE = 32
#     SEED = 42

#     AGENT_TYPES = ["A2C"]

#     all_results = {}

#     for agent in AGENT_TYPES:
#         # Load previously completed experiments.
#         saved_results = load_agent_results(agent)

#         if saved_results is not None:
#             all_results[agent] = saved_results
#             continue

#         # Run experiments only when saved results don't exist.
#         results = run_multiple_trials(
#             agent_type=agent,
#             num_trials=NUM_TRIALS,
#             num_episodes=NUM_EPISODES,
#             max_steps=MAX_STEPS,
#             batch_size=BATCH_SIZE,
#             base_seed=SEED,
#             plot=True,
#             render_plots=False,
#             print_stats=True
#         )

#         # all_results[agent] = results

#         all_results = add_agent_results(all_results, "A2C", results)

#         # # Save immediately after this algorithm finishes.
#         # save_agent_results(agent, results)

if __name__ == "__main__":
    NUM_TRIALS = 1
    NUM_EPISODES = 10000
    MAX_STEPS = 200
    BATCH_SIZE = 32
    SEED = 49
    AGENT_TYPES = ["A2C"]

    all_results = {}

    for agent in AGENT_TYPES:

        # Load existing results.
        saved_results = load_agent_results(agent)
        all_results[agent] = saved_results if saved_results is not None else []

        # Continue with new seeds to avoid repeating trials.
        existing_seeds = [
            result["seed"]
            for result in all_results[agent]
            if result.get("seed") is not None
        ]

        next_seed = max(max(existing_seeds, default=SEED - 1) + 1, SEED)

        # Run additional trials.
        results = run_multiple_trials(
            agent_type=agent,
            num_trials=NUM_TRIALS,
            num_episodes=NUM_EPISODES,
            max_steps=MAX_STEPS,
            batch_size=BATCH_SIZE,
            base_seed=next_seed,
            plot=True,
            render_plots=False,
            print_stats=True
        )

        # Append new results and save.
        all_results = add_agent_results(all_results, agent, results)