import matplotlib.pyplot as plt
import numpy as np


def print_agent_statistics(
    title: str,
    agent_rewards: dict[str, list[float]],
    steps_per_episode: list[int],
) -> None:
    """Print summary statistics for agent rewards and episode lengths."""

    # Combine rewards from all players across all episodes.
    rewards = np.array(
        list(agent_rewards.values()),
        dtype=float,
    )

    # Combine all four agents' rewards into one total for each episode.
    episode_rewards = rewards.sum(axis=0)

    steps = np.array(
        steps_per_episode,
        dtype=float,
    )

    print(f"\n{title}")
    print("-" * 60)

    print(
        f"Reward: "
        f"{np.mean(episode_rewards):.2f} ± "
        f"{np.std(episode_rewards):.2f} "
        f"[{np.min(episode_rewards):.2f}, "
        f"{np.max(episode_rewards):.2f}]"
    )

    print(
        f"Steps:  "
        f"{np.mean(steps):.2f} ± "
        f"{np.std(steps):.2f} "
        f"[{np.min(steps):.0f}, "
        f"{np.max(steps):.0f}]"
    )


def plot_agent_progress(
    agent_rewards: dict[str, list[float]],
    steps_per_episode: list[int],
    num_episodes: int,
    experiment_name: str,
    plot_file_name: str,
) -> None:
    # Create a 2x3 grid for the combined, individual, and step plots.
    fig, axs = plt.subplots(
        2,
        3,
        figsize=(18, 10),
        sharex=False,
    )

    fig.suptitle(
        "Random Agent Performance",
        fontsize=18,
        fontweight="bold",
    )

    # Limit plotting to the number of episodes available for every agent.
    max_episodes: int = min(
        num_episodes,
        min(len(rewards) for rewards in agent_rewards.values()),
    )

    # Sample every 20th episode to reduce the number of plotted points.
    x_axis: np.ndarray = np.arange(0, max_episodes, 20)

    # Combined Agent Rewards
    for agent_name, rewards in agent_rewards.items():
        y_axis: list[float] = rewards[:max_episodes:20]

        axs[0, 0].plot(
            x_axis[: len(y_axis)],
            y_axis,
            label=f"Agent {agent_name}",
            linewidth=1.5,
            alpha=0.8,
        )

    axs[0, 0].set_title(
        "Combined Agent Rewards",
        fontsize=12,
        fontweight="bold",
    )
    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].set_ylabel("Total Reward")
    axs[0, 0].legend(fontsize=9)
    axs[0, 0].grid(alpha=0.3)

    # Individual Agent Rewards
    for i, (agent_name, rewards) in enumerate(agent_rewards.items()):
        # Assign each agent to one of the four remaining reward plots.
        row: int = (i + 1) // 3
        col: int = (i + 1) % 3

        y_axis: list[float] = rewards[:max_episodes:20]

        axs[row, col].plot(
            x_axis[: len(y_axis)],
            y_axis,
            linewidth=1.5,
            alpha=0.85,
        )

        axs[row, col].set_title(
            f"Agent {agent_name}",
            fontsize=12,
            fontweight="bold",
        )
        axs[row, col].set_xlabel("Episode")
        axs[row, col].set_ylabel("Total Reward")
        axs[row, col].grid(alpha=0.3)

    # Smoothed Steps Per Episode

    steps_ax = axs[1, 2]

    if len(steps_per_episode) > 1:
        # Use up to 50 episodes to calculate the moving average.
        window_size: int = min(
            50,
            len(steps_per_episode) // 2,
        )

        # Smooth episode lengths using a moving average.
        smoothed_steps: np.ndarray = np.convolve(
            steps_per_episode,
            np.ones(window_size) / window_size,
            mode="valid",
        )

        steps_ax.plot(
            range(window_size - 1, len(steps_per_episode)),
            smoothed_steps,
            linewidth=2,
        )

        steps_ax.set_xlabel("Episode")
        steps_ax.set_ylabel("Steps")
        steps_ax.set_title(
            f"Smoothed Steps (Window = {window_size})",
            fontsize=12,
            fontweight="bold",
        )
        steps_ax.grid(alpha=0.3)

    else:
        steps_ax.text(
            0.5,
            0.5,
            "Insufficient data",
            ha="center",
            va="center",
            transform=steps_ax.transAxes,
        )
        steps_ax.set_title("Smoothed Steps")

    # Plot Formatting

    for ax in axs.flat:
        ax.tick_params(labelsize=9)

        # Hide the top and right plot borders for a cleaner appearance.
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(
        plot_file_name,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

def plot_agent_statistics(
    agent_rewards: dict[str, list[float]],
    steps_per_episode: list[int],
    num_episodes: int,
    experiment_name: str,
    plot_file_name: str,
) -> None:
    """Plot total rewards and steps per episode."""

    # Combine rewards from all players across all episodes.
    rewards = np.array(
        list(agent_rewards.values()),
        dtype=float,
    )

    # Combine all four agents' rewards into one total for each episode.
    episode_rewards = rewards.sum(axis=0)

    steps = np.array(
        steps_per_episode,
        dtype=float,
    )

    # Limit data to the requested number of episodes.
    max_episodes: int = min(
        num_episodes,
        len(episode_rewards),
        len(steps),
    )

    episode_rewards = episode_rewards[:max_episodes]
    steps = steps[:max_episodes]

    episodes = np.arange(max_episodes)

    # Use a 20-episode moving average.
    window_size: int = 20

    smoothed_rewards = np.convolve(
        episode_rewards,
        np.ones(window_size) / window_size,
        mode="valid",
    )

    smoothed_steps = np.convolve(
        steps,
        np.ones(window_size) / window_size,
        mode="valid",
    )

    # The smoothed values begin after the first complete window.
    smoothed_episodes = np.arange(
        window_size - 1,
        max_episodes,
    )

    # Create reward and step plots.
    fig, axs = plt.subplots(
        1,
        2,
        figsize=(14, 5),
    )

    # Total Rewards
    axs[0].plot(
        episodes,
        episode_rewards,
        label="Total Rewards",
        linewidth=0.8,
        color="blue",
        alpha=0.5
    )

    axs[0].plot(
        smoothed_episodes,
        smoothed_rewards,
        label="Smoothed Total Rewards",
        linewidth=2,
        color="blue"
    )

    axs[0].set_title(
        f"{experiment_name} Total Rewards"
    )
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Total Reward")
    axs[0].legend()
    axs[0].grid(alpha=0.3)

    # Steps Per Episode
    axs[1].plot(
        episodes,
        steps,
        label="Steps Per Episode",
        linewidth=0.8,
        alpha=0.15,
    )

    axs[1].plot(
        smoothed_episodes,
        smoothed_steps,
        label="Smoothed Steps",
        linewidth=2,
    )

    axs[1].set_title(
        f"{experiment_name} Steps Per Episode"
    )
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Steps")
    axs[1].legend()
    axs[1].grid(alpha=0.3)

    # Overall formatting.
    fig.suptitle(
        f"{experiment_name} Performance",
        fontsize=16,
        fontweight="bold",
    )

    for ax in axs:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()

    plt.savefig(
        plot_file_name,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()