"""Simulated annealing analysis utilities."""

from collections.abc import Callable
from typing import Any

import dimod
import numpy as np

from src.analysis import (
    compute_feasibility_rate,
    compute_mean_energy,
    compute_optimal_rate,
)
from src.solvers import solve_simulated_annealing


def analyze_simulated_annealing_run(
    bqm: dimod.BinaryQuadraticModel,
    optimal_energy: float,
    num_reads: int,
    feasibility_check: Callable[[Any], bool],
) -> dict[str, float]:
    """Analyze one simulated annealing run.

    Args:
        bqm: Binary quadratic model to sample.
        optimal_energy: Known optimal feasible energy.
        num_reads: Number of reads in the annealing run.
        feasibility_check: Function that checks sample feasibility.

    Returns:
        Dictionary containing sampling and solution-quality metrics.

    Raises:
        RuntimeError: If the run contains no feasible samples.
    """
    sampleset = solve_simulated_annealing(
        bqm=bqm,
        num_reads=num_reads,
    )

    best_energy = sampleset.first.energy

    mean_energy = compute_mean_energy(sampleset)

    _, optimal_rate = compute_optimal_rate(
        sampleset=sampleset,
        optimal_energy=optimal_energy,
    )

    _, feasibility_rate = compute_feasibility_rate(
        sampleset=sampleset,
        feasibility_check=feasibility_check,
    )

    feasible_energies = [
        sample_record.energy
        for sample_record in sampleset.data(
            fields=["sample", "energy"],
        )
        if feasibility_check(sample_record.sample)
    ]

    if not feasible_energies:
        raise RuntimeError(
            "Simulated annealing run contains no feasible samples."
        )

    best_feasible_energy = min(feasible_energies)

    absolute_gap = (
        best_feasible_energy
        - optimal_energy
    )

    relative_gap = (
        absolute_gap / optimal_energy
    )

    return {
        "best_energy": float(best_energy),
        "mean_energy": float(mean_energy),
        "optimal_rate": float(optimal_rate),
        "feasibility_rate": float(feasibility_rate),
        "best_feasible_energy": float(best_feasible_energy),
        "absolute_gap": float(absolute_gap),
        "relative_gap": float(relative_gap),
    }


def analyze_simulated_annealing_runs(
    bqm: dimod.BinaryQuadraticModel,
    optimal_energy: float,
    num_reads: int,
    num_runs: int,
    feasibility_check: Callable[[Any], bool],
) -> dict[str, float]:
    """Analyze repeated simulated annealing runs.

    Args:
        bqm: Binary quadratic model to sample.
        optimal_energy: Known optimal feasible energy.
        num_reads: Number of reads per run.
        num_runs: Number of independent annealing runs.
        feasibility_check: Function that checks sample feasibility.

    Returns:
        Dictionary containing aggregated sampling and
        solution-quality metrics.

    Raises:
        ValueError: If fewer than two runs are requested.
    """
    if num_runs < 2:
        raise ValueError(
            "num_runs must be at least 2."
        )

    run_results = [
        analyze_simulated_annealing_run(
            bqm=bqm,
            optimal_energy=optimal_energy,
            num_reads=num_reads,
            feasibility_check=feasibility_check,
        )
        for _ in range(num_runs)
    ]

    best_energies = np.array(
        [
            result["best_energy"]
            for result in run_results
        ]
    )

    mean_energies = np.array(
        [
            result["mean_energy"]
            for result in run_results
        ]
    )

    optimal_rates = np.array(
        [
            result["optimal_rate"]
            for result in run_results
        ]
    )

    feasibility_rates = np.array(
        [
            result["feasibility_rate"]
            for result in run_results
        ]
    )

    best_feasible_energies = np.array(
        [
            result["best_feasible_energy"]
            for result in run_results
        ]
    )

    absolute_gaps = np.array(
        [
            result["absolute_gap"]
            for result in run_results
        ]
    )

    relative_gaps = np.array(
        [
            result["relative_gap"]
            for result in run_results
        ]
    )

    successful_runs = np.sum(
        np.isclose(
            best_feasible_energies,
            optimal_energy,
        )
    )

    run_success_rate = (
        successful_runs / num_runs
    )

    return {
        "mean_best_energy": float(
            np.mean(best_energies)
        ),
        "std_best_energy": float(
            np.std(best_energies, ddof=1)
        ),
        "mean_energy": float(
            np.mean(mean_energies)
        ),
        "std_mean_energy": float(
            np.std(mean_energies, ddof=1)
        ),
        "mean_optimal_rate": float(
            np.mean(optimal_rates)
        ),
        "std_optimal_rate": float(
            np.std(optimal_rates, ddof=1)
        ),
        "mean_feasibility_rate": float(
            np.mean(feasibility_rates)
        ),
        "std_feasibility_rate": float(
            np.std(feasibility_rates, ddof=1)
        ),
        "mean_best_feasible_energy": float(
            np.mean(best_feasible_energies)
        ),
        "std_best_feasible_energy": float(
            np.std(best_feasible_energies, ddof=1)
        ),
        "mean_absolute_gap": float(
            np.mean(absolute_gaps)
        ),
        "std_absolute_gap": float(
            np.std(absolute_gaps, ddof=1)
        ),
        "mean_relative_gap": float(
            np.mean(relative_gaps)
        ),
        "std_relative_gap": float(
            np.std(relative_gaps, ddof=1)
        ),
        "run_success_rate": float(
            run_success_rate
        ),
    }
