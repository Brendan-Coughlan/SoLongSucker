import matplotlib.pyplot as plt
import numpy as np


def print_agent_statistics(title, agent_rewards, steps_per_episode):
    """Print summary statistics for total rewards and episode lengths."""

    # Calculate total rewards and episode lengths.
    rewards = np.array(list(agent_rewards.values()), dtype=float)
    episode_rewards = rewards.sum(axis=0)
    steps = np.array(steps_per_episode, dtype=float)

    print(f"\n{title}")
    print("-" * 60)
    print(f"Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f} [{np.min(episode_rewards):.2f}, {np.max(episode_rewards):.2f}]")
    print(f"Steps:  {np.mean(steps):.2f} ± {np.std(steps):.2f} [{np.min(steps):.0f}, {np.max(steps):.0f}]")

    total_rewards = np.sum(list(agent_rewards.values()), axis=0)

    print(f"First 20 episodes: {np.mean(total_rewards[:20]):.2f}")
    print(f"Last 20 episodes:  {np.mean(total_rewards[-20:]):.2f}")


def plot_agent_statistics(
    agent_rewards, steps_per_episode,
    num_episodes, experiment_name, plot_file_name,
    render_plots=True
):
    """Plot total rewards and steps per episode."""

    # Prepare episode data.
    rewards = np.array(list(agent_rewards.values()), dtype=float)
    episode_rewards = rewards.sum(axis=0)
    steps = np.array(steps_per_episode, dtype=float)

    max_episodes: int = min(num_episodes, len(episode_rewards), len(steps))
    episode_rewards = episode_rewards[:max_episodes]
    steps = steps[:max_episodes]
    episodes = np.arange(max_episodes)

    # Calculate 20-episode moving averages.
    window_size = min(20, max_episodes)
    smoothed_rewards = np.convolve(episode_rewards, np.ones(window_size) / window_size, mode="valid")
    smoothed_steps = np.convolve(steps, np.ones(window_size) / window_size, mode="valid")
    smoothed_episodes = np.arange(window_size - 1, max_episodes)

    # Create reward and step plots.
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Total rewards
    axs[0].plot(episodes, episode_rewards, label="Total Rewards", linewidth=0.8, color="blue", alpha=0.5)
    axs[0].plot(smoothed_episodes, smoothed_rewards, label="Smoothed Total Rewards", linewidth=2, color="blue")
    axs[0].set_title(f"{experiment_name} Total Rewards")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Total Reward")
    axs[0].legend()
    axs[0].grid(alpha=0.3)

    # Steps per episode
    axs[1].plot(episodes, steps, label="Steps Per Episode", linewidth=0.8, color="green", alpha=0.5)
    axs[1].plot(smoothed_episodes, smoothed_steps, label="Smoothed Steps", linewidth=2, color="green")
    axs[1].set_title(f"{experiment_name} Steps Per Episode")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Steps")
    axs[1].legend()
    axs[1].grid(alpha=0.3)

    # Format and save the figure.
    fig.suptitle(f"{experiment_name} Performance", fontsize=16, fontweight="bold")

    for ax in axs:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(plot_file_name, dpi=300, bbox_inches="tight")

    if render_plots:
        plt.show()