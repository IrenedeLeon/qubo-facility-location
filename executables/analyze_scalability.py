"""Analyze the scalability of the facility location benchmark."""

from functools import partial
from pathlib import Path
from time import perf_counter

import pandas as pd

from src.annealing_analysis import analyze_simulated_annealing_runs
from src.benchmark import (
    OPENING_COSTS,
    compute_feasible_optimal_cost,
    generate_assignment_costs,
)
from src.build_facility_location import build_facility_location_bqm
from src.constraint_validation import is_feasible
from src.solvers import solve_exact


NUM_CLIENTS_VALUES = [
    3,
    5,
    7,
    9,
    10,
    15,
    20,
    30,
    40,
    50,
]

NUM_FACILITIES = len(OPENING_COSTS)

ASSIGNMENT_PENALTY_STRENGTH = 10.0
OPEN_PENALTY_STRENGTH = 10.0

SA_NUM_READS = 100
SA_NUM_RUNS = 20

MAX_EXACT_VARIABLES = 22

RESULTS_DIR = Path("results")
RESULTS_PATH = RESULTS_DIR / "scalability_analysis.csv"

TABLE_WIDTH = 145
ENERGY_TOLERANCE = 1e-9


def main() -> None:
    """Analyze exact and simulated annealing scalability."""
    print()
    print("=" * TABLE_WIDTH)
    print("FACILITY LOCATION SCALABILITY ANALYSIS")
    print("=" * TABLE_WIDTH)

    print(f"Number of facilities (fixed): {NUM_FACILITIES}")
    print(f"Assignment penalty: {ASSIGNMENT_PENALTY_STRENGTH}")
    print(f"Open-facility penalty: {OPEN_PENALTY_STRENGTH}")
    print(f"SA reads per run: {SA_NUM_READS}")
    print(f"SA runs per instance: {SA_NUM_RUNS}")
    print(
        "ExactSolver maximum number of variables: "
        f"{MAX_EXACT_VARIABLES}"
    )

    print()
    print(
        f"{'clients':>7} "
        f"{'vars':>6} "
        f"{'C*':>8} "
        f"{'exact (s)':>11} "
        f"{'SA (s)':>10} "
        f"{'optimal rate':>18} "
        f"{'feasibility':>18} "
        f"{'run success':>14} "
        f"{'best feasible':>18} "
        f"{'relative gap':>18}"
    )
    print("-" * TABLE_WIDTH)

    benchmark_results = []

    for num_clients in NUM_CLIENTS_VALUES:
        assignment_costs = generate_assignment_costs(
            num_clients=num_clients,
        )

        num_variables = (
            NUM_FACILITIES
            + num_clients * NUM_FACILITIES
        )

        search_space_size = 2 ** num_variables

        feasible_optimal_cost = compute_feasible_optimal_cost(
            opening_costs=OPENING_COSTS,
            assignment_costs=assignment_costs,
        )

        bqm = build_facility_location_bqm(
            opening_costs=OPENING_COSTS,
            assignment_costs=assignment_costs,
            assignment_penalty_strength=ASSIGNMENT_PENALTY_STRENGTH,
            open_penalty_strength=OPEN_PENALTY_STRENGTH,
        )

        exact_runtime = None
        exact_energy = None
        exact_feasible = None

        if num_variables <= MAX_EXACT_VARIABLES:
            exact_start_time = perf_counter()

            exact_sampleset = solve_exact(bqm)

            exact_runtime = (
                perf_counter()
                - exact_start_time
            )

            best_exact_sample = exact_sampleset.first
            exact_energy = best_exact_sample.energy

            exact_feasible = is_feasible(
                sample=best_exact_sample.sample,
                num_clients=num_clients,
                num_facilities=NUM_FACILITIES,
            )

            if not exact_feasible:
                raise RuntimeError(
                    "Exact ground state is infeasible for "
                    f"{num_clients} clients."
                )

            if (
                abs(
                    exact_energy
                    - feasible_optimal_cost
                )
                > ENERGY_TOLERANCE
            ):
                raise RuntimeError(
                    f"Exact energy {exact_energy} does not match "
                    "known feasible optimum "
                    f"{feasible_optimal_cost} for "
                    f"{num_clients} clients."
                )

        feasibility_check = partial(
            is_feasible,
            num_clients=num_clients,
            num_facilities=NUM_FACILITIES,
        )

        sa_start_time = perf_counter()

        sa_results = analyze_simulated_annealing_runs(
            bqm=bqm,
            optimal_energy=feasible_optimal_cost,
            num_reads=SA_NUM_READS,
            num_runs=SA_NUM_RUNS,
            feasibility_check=feasibility_check,
        )

        sa_runtime = (
            perf_counter()
            - sa_start_time
        )

        optimal_rate = (
            f"{sa_results['mean_optimal_rate']:.1%} "
            f"± {sa_results['std_optimal_rate']:.1%}"
        )

        feasibility_rate = (
            f"{sa_results['mean_feasibility_rate']:.1%} "
            f"± {sa_results['std_feasibility_rate']:.1%}"
        )

        run_success_rate = (
            f"{sa_results['run_success_rate']:.1%}"
        )

        best_feasible_energy = (
            f"{sa_results['mean_best_feasible_energy']:.2f} "
            f"± {sa_results['std_best_feasible_energy']:.2f}"
        )

        relative_gap = (
            f"{sa_results['mean_relative_gap']:.2%} "
            f"± {sa_results['std_relative_gap']:.2%}"
        )

        if exact_runtime is None:
            exact_runtime_display = "skipped"
        else:
            exact_runtime_display = (
                f"{exact_runtime:.6f}"
            )

        print(
            f"{num_clients:>7} "
            f"{num_variables:>6} "
            f"{feasible_optimal_cost:>8.2f} "
            f"{exact_runtime_display:>11} "
            f"{sa_runtime:>10.4f} "
            f"{optimal_rate:>18} "
            f"{feasibility_rate:>18} "
            f"{run_success_rate:>14} "
            f"{best_feasible_energy:>18} "
            f"{relative_gap:>18}"
        )

        benchmark_results.append(
            {
                "num_clients": num_clients,
                "num_facilities": NUM_FACILITIES,
                "num_variables": num_variables,
                "search_space_size": search_space_size,
                "feasible_optimal_cost": (
                    feasible_optimal_cost
                ),
                "assignment_penalty_strength": (
                    ASSIGNMENT_PENALTY_STRENGTH
                ),
                "open_penalty_strength": (
                    OPEN_PENALTY_STRENGTH
                ),
                "exact_solver_executed": (
                    num_variables
                    <= MAX_EXACT_VARIABLES
                ),
                "exact_energy": exact_energy,
                "exact_feasible": exact_feasible,
                "exact_runtime_seconds": (
                    exact_runtime
                ),
                "sa_num_reads": SA_NUM_READS,
                "sa_num_runs": SA_NUM_RUNS,
                "sa_runtime_seconds": sa_runtime,
                **sa_results,
            }
        )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df = pd.DataFrame(
        benchmark_results,
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    print()
    print(
        f"Results saved to: {RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()
