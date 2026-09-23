import matplotlib.pyplot as plt
import numpy as np

def plot_agent_progress(
    agent_rewards: dict[str, list[float]],
    steps_per_episode: list[int],
    num_episodes: int,
    plot_file_name: str = "data/random_agent_baseline.png"
) -> None:
    fig, axs = plt.subplots(3, 2, figsize=(15, 20))

    # Determine maximum number of episodes actually recorded
    max_episodes: int = min(
        num_episodes,
        min(len(rewards) for rewards in agent_rewards.values()),
    )

    # Plot every 20 episodes
    x_axis: np.ndarray = np.arange(0, max_episodes, 20)

    # Combined agent rewards
    for agent_name, rewards in agent_rewards.items():
        y_axis: list[float] = rewards[:max_episodes:20]

        axs[0, 0].plot(
            x_axis[: len(y_axis)],
            y_axis,
            label=f"Agent {agent_name}",
        )

    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].set_ylabel("Total Reward")
    axs[0, 0].set_title("Random Agents - Combined Rewards (Every 20 Episodes)")
    axs[0, 0].legend()

    # Individual agent rewards
    for i, (agent_name, rewards) in enumerate(agent_rewards.items()):
        row: int = (i + 1) // 2
        col: int = (i + 1) % 2

        y_axis: list[float] = rewards[:max_episodes:20]

        axs[row, col].plot(
            x_axis[: len(y_axis)],
            y_axis,
        )

        axs[row, col].set_xlabel("Episode")
        axs[row, col].set_ylabel("Total Reward")
        axs[row, col].set_title(f"Agent {agent_name} - Rewards (Every 20 Episodes)")

    # Smoothed steps per episode
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

        axs[2, 1].plot(
            range(window_size - 1, len(steps_per_episode)),
            smoothed_steps,
        )

        axs[2, 1].set_xlabel("Episode")
        axs[2, 1].set_ylabel("Steps")
        axs[2, 1].set_title(
            "Random Agents - Smoothed Steps per Episode "
            f"(Window Size: {window_size})"
        )

    else:
        axs[2, 1].text(
            0.5,
            0.5,
            "Insufficient data for steps plot",
            ha="center",
            va="center",
        )

    plt.tight_layout()
    plt.savefig(plot_file_name)
    plt.show()
