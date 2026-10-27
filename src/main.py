import os
from simulation import (
    load_agent_results,
    save_agent_results,
    add_agent_results,
    run_multiple_trials,
)
from analysis import plot_algorithm_comparison, print_algorithm_comparison_statistics

# ============================================================
# Main Experiment Configuration
# ============================================================
def main():
    print("\nWelcome to So Long Sucker Reinforcement Learning Environment!\n")

    # ============================================================
    # Experiment Configuration
    # ============================================================

    available_agents = [
        "Random", "DQN", "DDQN", "DuelingDQN", "A2C", "A3C"
    ]

    agent_lookup = {
        agent.lower(): agent for agent in available_agents
    }

    print("Available agents:", ", ".join(available_agents))

    agents_input = input(
        "\nWhat agents should be trained? "
        "(Comma-separated, Default: Random): "
    ).strip()

    if agents_input:
        requested_agents = [
            agent.strip() for agent in agents_input.split(",")
        ]
    else:
        requested_agents = ["Random"]

    invalid_agents = [
        agent for agent in requested_agents
        if agent.lower() not in agent_lookup
    ]

    if invalid_agents:
        print(f"Invalid agents: {', '.join(invalid_agents)}")
        return

    agent_types = list(dict.fromkeys(
        agent_lookup[agent.lower()]
        for agent in requested_agents
    ))

    # ------------------------------------------------------------
    # Training Settings
    # ------------------------------------------------------------

    num_trials = int(input("How many NEW trials? (Default: 1, 0 to skip training): ") or 1)

    num_episodes = int(input("How many episodes per trial? (Default: 1000): ") or 1000)

    max_steps = int(input("Maximum steps per episode? (Default: 200): ") or 200)

    batch_size = int(input("Batch size? (Default: 32): ") or 32)

    base_seed = int(input("Base random seed? (Default: 42): ") or 42)

    plot = input("Plot individual trial results? (Y/N, Default: N): ").strip().lower() == "y"

    print_trial_stats = input("Print trial statistics? (Y/N, Default: N): ").strip().lower() == "y"
    print_overall_stats = input("Print overall statistics? (Y/N, Default: Y): ").strip().lower() != "n"

    use_gpu = input("Use GPU? (Y/N, Default: N): ").strip().lower() == "y"

    # ------------------------------------------------------------
    # Saved Results Settings
    # ------------------------------------------------------------

    truncate_results = input("\nReset previously saved trials for selected agents? (Y/N, Default: N): ").strip().lower() == "y"

    # ------------------------------------------------------------
    # Algorithm Comparison Settings
    # ------------------------------------------------------------

    compare = input("Generate algorithm comparison plot? (Y/N, Default: N): ").strip().lower() == "y"

    # ============================================================
    # Configuration Summary
    # ============================================================

    print(f"\n{'=' * 60}")
    print("EXPERIMENT CONFIGURATION")
    print(f"{'=' * 60}")

    print(f"Agents:                    {', '.join(agent_types)}")
    print(f"New trials per agent:      {num_trials}")
    print(f"Episodes per trial:        {num_episodes}")
    print(f"Maximum steps:             {max_steps}")
    print(f"Batch size:                {batch_size}")
    print(f"Base seed:                 {base_seed}")
    print(f"Plot individual trials:    {plot}")
    print(f"Print trial statistics:    {print_trial_stats}")
    print(f"Print overall statistics:  {print_overall_stats}")
    print(f"Use GPU:                   {use_gpu}")
    print(f"Reset saved trials:        {truncate_results}")
    print(f"Algorithm comparison:      {compare}")

    print(f"{'=' * 60}")

    if truncate_results:
        print("\nWARNING: Saved results for the selected agents will be replaced.")

    confirm = input("\nProceed? (Y/N, Default: N): ").strip().lower()

    if confirm != "y":
        print("Operation cancelled.")
        return

    # --------------------------------------------------------
    # Run Experiments
    # --------------------------------------------------------

    all_results = {}

    for agent in agent_types:
        next_seed = base_seed

        print(f"\n{'=' * 60}")
        print(f"Processing {agent}")
        print(f"{'=' * 60}\n")

        if truncate_results:
            # Reset saved results for the current agent
            all_results[agent] = []
            print(f"Resetting saved results for {agent}")

            if num_trials == 0:
                save_agent_results(agent, [])
        else:
            # Load existing results if available
            saved_results = load_agent_results(agent)
            if saved_results is not None:
                all_results[agent] = saved_results
            else:
                all_results[agent] = []

            existing_seeds = [result["seed"] for result in all_results[agent]]
            if existing_seeds:
                next_seed = max(max(existing_seeds) + 1, base_seed)

        print(f"Existing trials for {agent}: {len(all_results[agent])}")
        print(f"New trials will start with seed: {next_seed}")

        # --------------------------------------------------------
        # Run new trials for the current agent
        # --------------------------------------------------------

        if num_trials > 0:
            results = run_multiple_trials(
                agent_type=agent,
                num_trials=num_trials,
                num_episodes=num_episodes,
                max_steps=max_steps,
                batch_size=batch_size,
                plot=plot,
                print_trial_stats=print_trial_stats,
                print_overall_stats=print_overall_stats,
                use_gpu=use_gpu,
                base_seed=next_seed
            )

            all_results = add_agent_results(all_results=all_results, agent_type=agent, results=results)
            print(f"Completed {num_trials} new trials for {agent}")

    # --------------------------------------------------------
    # Algorithm Comparison
    # --------------------------------------------------------

    if compare:
        for agent in available_agents:
            if agent not in all_results:
                saved_results = load_agent_results(agent)
                if saved_results:
                    all_results[agent] = saved_results

        comparison_results = {
            agent: trials
            for agent, trials in all_results.items()
            if trials
        }

        if comparison_results:
            os.makedirs("data/figures", exist_ok=True)

            plot_algorithm_comparison(
                comparison_results,
                plot_file_name="data/figures/algorithm_comparison.png"
                )

            print_algorithm_comparison_statistics(comparison_results)
        else:
            print("No saved results available for comparison.")

    print("\nAll operations completed successfully!")


if __name__ == "__main__":
    main()