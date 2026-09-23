import matplotlib.pyplot as plt
import numpy as np


def plot_agent_progress(
    agent_rewards: dict[str, list[float]],
    steps_per_episode: list[int],
    num_episodes: int,
    plot_file_name: str = "data/random_agent_baseline.png",
) -> None:
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

    # Determine maximum number of episodes actually recorded
    max_episodes: int = min(
        num_episodes,
        min(len(rewards) for rewards in agent_rewards.values()),
    )

    # Plot every 20 episodes
    x_axis: np.ndarray = np.arange(0, max_episodes, 20)

    # ---------------------------------------------------------
    # Combined agent rewards
    # ---------------------------------------------------------
    for agent_name, rewards in agent_rewards.items():
        y_axis: list[float] = rewards[:max_episodes:20]

        axs[0, 0].plot(
            x_axis[:len(y_axis)],
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

    # ---------------------------------------------------------
    # Individual agent rewards
    # ---------------------------------------------------------
    for i, (agent_name, rewards) in enumerate(agent_rewards.items()):
        row: int = (i + 1) // 3
        col: int = (i + 1) % 3

        y_axis: list[float] = rewards[:max_episodes:20]

        axs[row, col].plot(
            x_axis[:len(y_axis)],
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

    # ---------------------------------------------------------
    # Smoothed steps per episode
    # ---------------------------------------------------------
    steps_ax = axs[1, 2]

    if len(steps_per_episode) > 1:
        window_size: int = min(
            50,
            len(steps_per_episode) // 2,
        )

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

    # ---------------------------------------------------------
    # Apply consistent formatting
    # ---------------------------------------------------------
    for ax in axs.flat:
        ax.tick_params(labelsize=9)

        # Remove unnecessary borders
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(
        plot_file_name,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()