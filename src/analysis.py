"""Analysis utilities for facility location solver results."""

from collections.abc import Callable
from typing import Any

import dimod


def compute_mean_energy(sampleset: dimod.SampleSet) -> float:
    """Compute the mean energy across all sampled reads.

    Args:
        sampleset: SampleSet containing sampled solutions and energies.

    Returns:
        Mean energy weighted by the number of occurrences.
    """
    total_samples = len(sampleset)

    total_energy = sum(
        record.energy * record.num_occurrences
        for record in sampleset.data(
            fields=["energy", "num_occurrences"]
        )
    )

    return total_energy / total_samples

def compute_optimal_rate(
    sampleset: dimod.SampleSet,
    optimal_energy: float,
    tolerance: float = 1e-9,
) -> tuple[int, float]:
    """Compute how often the sampler reaches the optimal energy.

    Args:
        sampleset: SampleSet containing sampled solutions and energies.
        optimal_energy: Reference optimal energy.
        tolerance: Numerical tolerance for energy comparison.

    Returns:
        Number of optimal samples and fraction of samples that are optimal.
    """
    num_optimal = sum(
        record.num_occurrences
        for record in sampleset.data(
            fields=["energy", "num_occurrences"]
        )
        if abs(record.energy - optimal_energy) < tolerance
    )

    optimal_rate = num_optimal / len(sampleset)

    return num_optimal, optimal_rate


def compute_feasibility_rate(
    sampleset: dimod.SampleSet,
    feasibility_check: Callable[[Any], bool],
) -> tuple[int, float]:
    """Compute the fraction of sampled solutions that are feasible.

    Args:
        sampleset: SampleSet containing sampled solutions.
        feasibility_check: Function that returns whether a sample is feasible.

    Returns:
        Number of feasible samples and fraction of samples that are feasible.
    """
    num_feasible = sum(
        record.num_occurrences
        for record in sampleset.data(
            fields=["sample", "num_occurrences"]
        )
        if feasibility_check(record.sample)
    )

    feasibility_rate = num_feasible / len(sampleset)

    return num_feasible, feasibility_rate
