from simulation import run_multiple_trials

if __name__ == "__main__":
    results = run_multiple_trials(
        agent_type="DQN",
        num_trials=3,
        num_episodes=10000,
        max_steps=200,
        batch_size=32,
        plot=True,
        print_trial_stats=True,
        print_overall_stats=True, 
        use_gpu=False,
        base_seed=42,
        use_action_masking=True
    )