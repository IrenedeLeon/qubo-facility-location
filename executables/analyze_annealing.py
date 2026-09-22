"""Analyze simulated annealing performance for the facility location QUBO."""

from pathlib import Path
from typing import Any

import pandas as pd

from src.annealing_analysis import analyze_simulated_annealing_runs
from src.build_facility_location import (
    build_assignment_penalty,
    build_cost,
    build_open_facility_penalty,
    build_variables,
)
from src.configuration import load_configuration
from src.constraint_validation import is_feasible
from src.solvers import solve_exact


CONFIG_PATH = Path("configs/problem_configuration.yml")

NUM_READS_VALUES = [10, 50, 100, 500, 1000]
NUM_RUNS = 20

RESULTS_DIR = Path("results")
RESULTS_PATH = RESULTS_DIR / "annealing_analysis.csv"


def print_analysis_header(
    optimal_energy: float,
    num_runs: int,
) -> None:
    """Print the simulated annealing analysis header.

    Args:
        optimal_energy: Exact ground-state energy used as reference.
        num_runs: Number of independent runs per configuration.
    """
    print()
    print("=" * 135)
    print("SIMULATED ANNEALING ANALYSIS")
    print("=" * 135)
    print(f"Exact optimal energy: {optimal_energy:.2f}")
    print(f"Independent runs per configuration: {num_runs}")

    print()
    print(
        f"{'num reads':>10} "
        f"{'best energy':>18} "
        f"{'mean energy':>20} "
        f"{'optimal rate':>22} "
        f"{'feasibility rate':>22}"
        f"{'run success':>16}"
    )
    print("-" * 135)


def print_analysis_row(
    num_reads: int,
    results: dict[str, float],
) -> None:
    """Print one row of simulated annealing results.

    Args:
        num_reads: Number of reads per simulated annealing run.
        results: Summary statistics across repeated runs.
    """
    best_energy = (
        f"{results['mean_best_energy']:.2f} "
        f"± {results['std_best_energy']:.2f}"
    )
    mean_energy = (
        f"{results['mean_energy']:.2f} "
        f"± {results['std_mean_energy']:.2f}"
    )
    optimal_rate = (
        f"{results['mean_optimal_rate']:.1%} "
        f"± {results['std_optimal_rate']:.1%}"
    )
    feasibility_rate = (
        f"{results['mean_feasibility_rate']:.1%} "
        f"± {results['std_feasibility_rate']:.1%}"
    )
    run_success_rate = f"{results['run_success_rate']:.1%}"

    print(
        f"{num_reads:>10} "
        f"{best_energy:>18} "
        f"{mean_energy:>20} "
        f"{optimal_rate:>22} "
        f"{feasibility_rate:>22}"
        f"{run_success_rate:>16}"
    )


def main() -> None:
    """Run the simulated annealing performance analysis."""
    config = load_configuration(CONFIG_PATH)

    opening_costs = config["opening_costs"]
    assignment_costs = config["assignment_costs"]
    assignment_penalty_strength = config["penalties"]["assignment"]
    open_penalty_strength = config["penalties"]["open"]

    num_clients = len(assignment_costs)
    num_facilities = len(opening_costs)

    assignments, facilities = build_variables(
        num_clients=num_clients,
        num_facilities=num_facilities,
    )

    cost = build_cost(
        assignments=assignments,
        facilities=facilities,
        opening_costs=opening_costs,
        assignment_costs=assignment_costs,
    )

    assignment_penalty = build_assignment_penalty(
        assignments=assignments,
        penalty_strength=assignment_penalty_strength,
    )

    open_facility_penalty = build_open_facility_penalty(
        assignments=assignments,
        facilities=facilities,
        penalty_strength=open_penalty_strength,
    )

    hamiltonian = (
        cost
        + assignment_penalty
        + open_facility_penalty
    )

    model = hamiltonian.compile()
    bqm = model.to_bqm()

    exact_sampleset = solve_exact(bqm)
    optimal_energy = exact_sampleset.first.energy

    def feasibility_check(sample: Any) -> bool:
        return is_feasible(
            sample=sample,
            num_clients=num_clients,
            num_facilities=num_facilities,
        )

    print_analysis_header(
        optimal_energy=optimal_energy,
        num_runs=NUM_RUNS,
    )

    experiment_results = []

    for num_reads in NUM_READS_VALUES:
        results = analyze_simulated_annealing_runs(
            bqm=bqm,
            optimal_energy=optimal_energy,
            num_reads=num_reads,
            num_runs=NUM_RUNS,
            feasibility_check=feasibility_check,
        )

        print_analysis_row(
            num_reads=num_reads,
            results=results,
        )

        experiment_results.append(
            {
                "num_reads": num_reads,
                "num_runs": NUM_RUNS,
                "optimal_energy": optimal_energy,
                **results,
            }
        )

    # Save results to a CSV file
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(experiment_results)
    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    print()
    print(f"Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
