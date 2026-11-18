# Filename: main.py
# Author: Brendan Coughlan
# Created/Last Modified: 2026-11-17
# Description: Main script for configuring and running experiments in the So Long Sucker reinforcement learning environment.

import argparse
import os
from simulation import (
    load_agent_results,
    save_agent_results,
    add_agent_results,
    run_multiple_trials,
    print_section,
)
from analysis import plot_algorithm_comparison, print_algorithm_comparison


AGENT_REGISTRY = [
    "Random",
    "DQN",
    "DDQN",
    "DuelingDQN",
    "A2C",
]


def parse_arguments():
    """Parse command-line arguments for experiment configuration."""

    parser = argparse.ArgumentParser(description="Configure and run So Long Sucker experiments.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument(
        "--agents",
        nargs="+",
        choices=AGENT_REGISTRY,
        default=["Random"],
        help="Agents to train or evaluate.",
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of new trials per agent. Use 0 to skip training.",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of episodes per trial.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum steps per episode.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed.",
    )

    parser.add_argument(
        "--action-masking",
        choices=["none", "phase", "full"],
        default="phase",
        help="Action masking strategy.",
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate individual trial performance plots.",
    )

    parser.add_argument(
        "--print-trial-stats",
        action="store_true",
        help="Print statistics for individual trials.",
    )

    parser.add_argument(
        "--no-overall-stats",
        dest="print_overall_stats",
        action="store_false",
        help="Disable aggregate trial statistics.",
    )

    parser.add_argument(
        "--print-resource-stats",
        action="store_true",
        help="Print hardware resource statistics.",
    )

    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Enable GPU usage.",
    )

    parser.add_argument(
        "--reset-results",
        action="store_true",
        help="Replace previously saved results for selected agents.",
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="Generate an algorithm comparison plot.",
    )

    # Overall statistics are enabled by default.
    parser.set_defaults(print_overall_stats=True)

    args = parser.parse_args()

    if args.trials < 0:
        parser.error("--trials cannot be negative.")

    if args.episodes <= 0:
        parser.error("--episodes must be greater than zero.")

    if args.max_steps <= 0:
        parser.error("--max-steps must be greater than zero.")

    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero.")

    return args

def main():
    """Configure and execute reinforcement learning experiments."""

    args = parse_arguments()

    agent_types = list(dict.fromkeys(args.agents))

    print("\n" + "=" * 60)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 60)

    configuration = {
        "Agents": ", ".join(agent_types),
        "New trials per agent": args.trials,
        "Episodes per trial": args.episodes,
        "Maximum steps": args.max_steps,
        "Batch size": args.batch_size,
        "Base seed": args.seed,
        "Action masking": args.action_masking,
        "Plot individual trials": args.plot,
        "Print trial statistics": args.print_trial_stats,
        "Print overall statistics": args.print_overall_stats,
        "Print resource statistics": args.print_resource_stats,
        "Use GPU": args.use_gpu,
        "Reset saved trials": args.reset_results,
        "Algorithm comparison": args.compare,
    }

    for label, value in configuration.items():
        print(f"{label:<30} {value}")

    print("=" * 60)

    all_results = {}

    for agent_type in agent_types:
        print_section(f"PROCESSING {agent_type}", width=60)

        if args.reset_results:
            all_results[agent_type] = []

            print(f"Resetting results for {agent_type}...")

            if args.trials == 0:
                save_agent_results(agent_type, all_results[agent_type])
        else:
            saved_results = load_agent_results(agent_type)

            all_results[agent_type] = saved_results if saved_results is not None else []

        existing_results = all_results[agent_type]

        existing_seeds = [result["seed"] for result in existing_results]
        next_seed = max(max(existing_seeds) + 1, args.seed) if existing_seeds else args.seed

        print(f"Existing trials: {len(existing_results)}")
        print(f"Starting seed: {next_seed}")

        if args.trials == 0:
            print("Skipping trial execution as --trials is set to 0.")
            continue

        results, metrics = run_multiple_trials(
            agent_type=agent_type,
            num_trials=args.trials,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            plot=args.plot,
            print_trial_stats=args.print_trial_stats,
            print_overall_stats=args.print_overall_stats,
            print_resource_stats=args.print_resource_stats,
            use_gpu=args.use_gpu,
            base_seed=next_seed,
            action_masking=args.action_masking
        )

        all_results = add_agent_results(all_results, agent_type, results)

        print(f"Completed {len(results)} new trials for {agent_type}.")

    if args.compare:
        print_section("ALGORITHM COMPARISON", width=60)

        for agent in AGENT_REGISTRY:
            if agent not in all_results:
                saved_results = load_agent_results(agent)

                all_results[agent] = saved_results if saved_results is not None else []

        comparison_results = {agent: trials for agent, trials in all_results.items() if trials}

        if comparison_results:
            plot_algorithm_comparison(comparison_results)
            print_algorithm_comparison(comparison_results)
        else:
            print("No saved results available for comparison.")

    print("\nAll operations completed successfully!")

if __name__ == "__main__":
    main()