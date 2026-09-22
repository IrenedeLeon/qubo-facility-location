"""Tests for analysis utilities."""

import unittest

import dimod

from src.analysis import (
    compute_feasibility_rate,
    compute_mean_energy,
    compute_optimal_rate,
)


class TestAnalysis(unittest.TestCase):
    """Tests for sample analysis utilities."""

    def setUp(self) -> None:
        """Create a deterministic sample set."""
        self.sampleset = dimod.SampleSet.from_samples(
            [
                {"x": 0},
                {"x": 1},
                {"x": 1},
                {"x": 1},
            ],
            vartype=dimod.BINARY,
            energy=[2.0, 1.0, 1.0, 1.0],
        )

    def test_compute_optimal_rate(self) -> None:
        """Compute the fraction of samples at the optimal energy."""
        num_optimal, optimal_rate = compute_optimal_rate(
            sampleset=self.sampleset,
            optimal_energy=1.0,
        )

        self.assertEqual(num_optimal, 3)
        self.assertAlmostEqual(optimal_rate, 0.75)

    def test_compute_mean_energy(self) -> None:
        """Compute the mean sampled energy."""
        mean_energy = compute_mean_energy(self.sampleset)

        self.assertAlmostEqual(mean_energy, 1.25)

    def test_compute_feasibility_rate(self) -> None:
        """Compute the fraction of samples satisfying a feasibility check."""

        def feasibility_check(sample):
            return sample["x"] == 1

        num_feasible, feasibility_rate = compute_feasibility_rate(
            sampleset=self.sampleset,
            feasibility_check=feasibility_check,
        )

        self.assertEqual(num_feasible, 3)
        self.assertAlmostEqual(feasibility_rate, 0.75)
