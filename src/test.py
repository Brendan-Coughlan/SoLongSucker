from simulation import run_multiple_trials

# ============================================================
# Experiment Configuration
# ============================================================

AGENT_TYPE = "DQN"
MASKING_MODE = "phase"

NUM_TRIALS = 2
NUM_EPISODES = 3000
MAX_STEPS = 200
BATCH_SIZE = 32
BASE_SEED = 42


# ============================================================
# Main Experiment
# ============================================================

def main():
    results, metrics = run_multiple_trials(
        agent_type=AGENT_TYPE,
        num_trials=NUM_TRIALS,
        num_episodes=NUM_EPISODES,
        max_steps=MAX_STEPS,
        batch_size=BATCH_SIZE,
        base_seed=BASE_SEED,
        action_masking=MASKING_MODE,
        plot=False,
        print_trial_stats=True,
        print_overall_stats=True,
        print_resource_stats=True,
        use_gpu=False
    )

    # print(metrics)

if __name__ == "__main__":
    main()
