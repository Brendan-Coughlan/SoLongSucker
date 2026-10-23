from main import load_agent_results
from analysis import plot_algorithm_comparison, print_algorithm_comparison_statistics

if __name__ == "__main__":
    AGENT_TYPES = ["Random", "DQN", "DDQN", "DuelingDQN", "A2C"]

    all_results = {}

    for agent in AGENT_TYPES:
        results = load_agent_results(agent)

        if results is not None:
            all_results[agent] = results

    plot_algorithm_comparison(
        all_results,
        plot_file_name="data/algorithm_comparison.png",
        window_size=100,
        render_plots=False
    )

    print_algorithm_comparison_statistics(all_results)