"""Solver interfaces for binary quadratic models."""

import dimod
import neal


def solve_exact(bqm: dimod.BinaryQuadraticModel) -> dimod.SampleSet:
    """Solve a binary quadratic model by exhaustive enumeration.

    Args:
        bqm: Binary quadratic model to solve.

    Returns:
        SampleSet containing all enumerated solutions and their energies.
    """
    sampler = dimod.ExactSolver()
    return sampler.sample(bqm)

def solve_simulated_annealing(
    bqm: dimod.BinaryQuadraticModel,
    num_reads: int = 100,
) -> dimod.SampleSet:
    """Solve a binary quadratic model using simulated annealing.

    Args:
        bqm: Binary quadratic model to solve.
        num_reads: Number of annealing reads to perform.

    Returns:
        SampleSet containing the sampled solutions and their energies.
    """
    sampler = neal.SimulatedAnnealingSampler()

    return sampler.sample(
        bqm,
        num_reads=num_reads,
    )
