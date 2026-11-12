
import csv
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from chip_game_env import ChipGameEnv
from simulation import run_multiple_trials

# ============================================================
# Experiment Configuration
# ============================================================

AGENT_TYPE = "DQN"
MASKING_MODES = ["none", "phase", "full"]

NUM_TRIALS = 1
NUM_EPISODES = 5000
MAX_STEPS = 200
BATCH_SIZE = 32
BASE_SEED = 42

OUTPUT_DIR = Path("data/masking_experiments")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Environment Validation
# ============================================================

def test_action_masks(num_games=20):
    """Check masks during randomly generated legal games."""
    env = ChipGameEnv()
    rng = np.random.default_rng(BASE_SEED)

    for episode in range(num_games):
        env.reset(seed=BASE_SEED + episode)

        for step in range(MAX_STEPS):
            mask = env.get_action_mask()

            assert mask.shape == (env.action_space.n,)
            assert np.all(np.isin(mask, [0, 1]))

            if env.done:
                assert not np.any(mask)
                break

            valid_actions = np.flatnonzero(mask)
            assert len(valid_actions) > 0, (
                f"Empty mask: episode={episode}, "
                f"step={step}, state={env.state}"
            )

            action = int(rng.choice(valid_actions))
            _, rewards, done, _, _ = env.step(action)

            assert isinstance(rewards, dict)
            assert set(rewards) == set(env.PLAYER_LETTERS)

            if done:
                assert not np.any(env.get_action_mask())
                break

    env.close()
    print("Action-mask validation passed.")

# ============================================================
# Experiment Helpers
# ============================================================

def get_episode_rewards(trial):
    rewards = np.asarray(
        list(trial["agent_rewards"].values()),
        dtype=float
    )
    return rewards.sum(axis=0)


def summarize_trials(mode, trials):
    rewards = []
    steps = []

    for trial in trials:
        episode_rewards = get_episode_rewards(trial)
        episode_steps = np.asarray(trial["steps_per_episode"])

        assert len(episode_rewards) == NUM_EPISODES
        assert len(episode_steps) == NUM_EPISODES
        assert np.all((episode_steps >= 1) & (episode_steps <= MAX_STEPS))

        rewards.append(np.mean(episode_rewards))
        steps.append(np.mean(episode_steps))

    return {
        "mode": mode,
        "trials": len(trials),
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "steps_mean": float(np.mean(steps)),
        "steps_std": float(np.std(steps))
    }


def plot_comparison(all_results):
    plt.figure(figsize=(10, 6))

    for mode, trials in all_results.items():
        curves = np.asarray([
            get_episode_rewards(trial) for trial in trials
        ])

        window = min(100, NUM_EPISODES)
        kernel = np.ones(window) / window

        smoothed = np.asarray([
            np.convolve(curve, kernel, mode="valid")
            for curve in curves
        ])

        mean = smoothed.mean(axis=0)
        std = smoothed.std(axis=0)
        episodes = np.arange(window, NUM_EPISODES + 1)

        plt.plot(episodes, mean, label=mode.capitalize())

        if len(trials) > 1:
            plt.fill_between(
                episodes, mean - std, mean + std, alpha=0.15
            )

    plt.title(f"{AGENT_TYPE} Action-Masking Comparison")
    plt.xlabel("Training Episode")
    plt.ylabel("Mean Total Reward")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "masking_comparison.png", dpi=300)
    plt.close()

# ============================================================
# Main Experiment
# ============================================================

def main():
    print("\nValidating environment action masks...")
    test_action_masks()

    all_results = {}
    summaries = []

    for mode in MASKING_MODES:
        print(f"\n{'=' * 60}")
        print(f"{AGENT_TYPE} | Masking: {mode}")
        print(f"{'=' * 60}")

        trials = run_multiple_trials(
            agent_type=AGENT_TYPE,
            num_trials=NUM_TRIALS,
            num_episodes=NUM_EPISODES,
            max_steps=MAX_STEPS,
            batch_size=BATCH_SIZE,
            base_seed=BASE_SEED,
            action_masking=mode,
            plot=False,
            print_trial_stats=False,
            print_overall_stats=True,
            use_gpu=False
        )

        for trial in trials:
            trial["action_masking"] = mode

        all_results[mode] = trials
        summaries.append(summarize_trials(mode, trials))

        file_path = OUTPUT_DIR / f"{AGENT_TYPE}_{mode}_results.pkl"
        with open(file_path, "wb") as file:
            pickle.dump(trials, file)

    plot_comparison(all_results)

    summary_path = OUTPUT_DIR / "summary.csv"
    with open(summary_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)

    print(f"\n{'=' * 70}")
    print("FINAL MASKING COMPARISON")
    print(f"{'=' * 70}")
    print(f"{'Masking':<12}{'Reward':>24}{'Steps':>24}")

    for result in summaries:
        reward = f"{result['reward_mean']:.2f} ± {result['reward_std']:.2f}"
        steps = f"{result['steps_mean']:.2f} ± {result['steps_std']:.2f}"
        print(f"{result['mode']:<12}{reward:>24}{steps:>24}")

    print(f"\nResults saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
