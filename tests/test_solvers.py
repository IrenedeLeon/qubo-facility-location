"""Tests for BQM solver utilities."""

import unittest

import dimod

from src.solvers import solve_exact, solve_simulated_annealing


class TestSolvers(unittest.TestCase):
    """Tests for QUBO solver utilities."""

    def setUp(self) -> None:
        """Create a simple BQM with a known optimum."""
        self.bqm = dimod.BinaryQuadraticModel(
            {"x": -1.0},
            {},
            0.0,
            dimod.BINARY,
        )

    def test_exact_solver_finds_known_optimum(self) -> None:
        """ExactSolver should find x=1 with energy -1."""
        sampleset = solve_exact(self.bqm)

        self.assertEqual(sampleset.first.sample["x"], 1)
        self.assertAlmostEqual(sampleset.first.energy, -1.0)

    def test_simulated_annealing_returns_requested_reads(self) -> None:
        """Simulated annealing should return the requested sample count."""
        num_reads = 20

        sampleset = solve_simulated_annealing(
            bqm=self.bqm,
            num_reads=num_reads,
        )

        self.assertEqual(len(sampleset), num_reads)
