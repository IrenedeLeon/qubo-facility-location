"""Tests for simulated annealing analysis utilities."""

import unittest
from unittest.mock import patch

import dimod

from src.annealing_analysis import (
    analyze_simulated_annealing_run,
    analyze_simulated_annealing_runs,
)


class TestAnnealingAnalysis(unittest.TestCase):
    """Test simulated annealing analysis utilities."""

    def test_single_run_returns_expected_metrics(self) -> None:
        """Check that a single run returns all expected metrics."""
        bqm = dimod.BinaryQuadraticModel(
            {"x": -1.0},
            {},
            0.0,
            dimod.BINARY,
        )

        def feasibility_check(sample) -> bool:
            return sample["x"] == 1

        result = analyze_simulated_annealing_run(
            bqm=bqm,
            optimal_energy=-1.0,
            num_reads=20,
            feasibility_check=feasibility_check,
        )

        self.assertEqual(
            set(result),
            {
                "best_energy",
                "mean_energy",
                "optimal_rate",
                "feasibility_rate",
                "best_feasible_energy",
                "absolute_gap",
                "relative_gap",
            },
        )

        self.assertLessEqual(
            result["best_energy"],
            0.0,
        )
        self.assertGreaterEqual(
            result["optimal_rate"],
            0.0,
        )
        self.assertLessEqual(
            result["optimal_rate"],
            1.0,
        )
        self.assertGreaterEqual(
            result["feasibility_rate"],
            0.0,
        )
        self.assertLessEqual(
            result["feasibility_rate"],
            1.0,
        )

        self.assertEqual(
            result["best_feasible_energy"],
            -1.0,
        )
        self.assertEqual(
            result["absolute_gap"],
            0.0,
        )
        self.assertEqual(
            result["relative_gap"],
            0.0,
        )

    @patch(
        "src.annealing_analysis.analyze_simulated_annealing_run"
    )
    def test_repeated_runs_compute_summary_statistics(
        self,
        mock_run,
    ) -> None:
        """Check summary statistics across repeated runs."""
        mock_run.side_effect = [
            {
                "best_energy": 16.0,
                "mean_energy": 16.2,
                "optimal_rate": 0.5,
                "feasibility_rate": 1.0,
                "best_feasible_energy": 16.0,
                "absolute_gap": 0.0,
                "relative_gap": 0.0,
            },
            {
                "best_energy": 16.0,
                "mean_energy": 16.4,
                "optimal_rate": 0.7,
                "feasibility_rate": 0.9,
                "best_feasible_energy": 16.0,
                "absolute_gap": 0.0,
                "relative_gap": 0.0,
            },
            {
                "best_energy": 17.0,
                "mean_energy": 16.6,
                "optimal_rate": 0.6,
                "feasibility_rate": 0.8,
                "best_feasible_energy": 17.0,
                "absolute_gap": 1.0,
                "relative_gap": 1.0 / 16.0,
            },
        ]

        bqm = dimod.BinaryQuadraticModel(
            {"x": -1.0},
            {},
            0.0,
            dimod.BINARY,
        )

        result = analyze_simulated_annealing_runs(
            bqm=bqm,
            optimal_energy=16.0,
            num_reads=100,
            num_runs=3,
            feasibility_check=lambda sample: True,
        )

        self.assertAlmostEqual(
            result["mean_best_energy"],
            49.0 / 3.0,
        )
        self.assertAlmostEqual(
            result["mean_energy"],
            16.4,
        )
        self.assertAlmostEqual(
            result["mean_optimal_rate"],
            0.6,
        )
        self.assertAlmostEqual(
            result["mean_feasibility_rate"],
            0.9,
        )

        self.assertAlmostEqual(
            result["mean_best_feasible_energy"],
            49.0 / 3.0,
        )
        self.assertAlmostEqual(
            result["mean_absolute_gap"],
            1.0 / 3.0,
        )
        self.assertAlmostEqual(
            result["mean_relative_gap"],
            1.0 / 48.0,
        )

        self.assertAlmostEqual(
            result["run_success_rate"],
            2.0 / 3.0,
        )

        self.assertGreater(
            result["std_best_energy"],
            0.0,
        )
        self.assertGreater(
            result["std_mean_energy"],
            0.0,
        )
        self.assertGreater(
            result["std_optimal_rate"],
            0.0,
        )
        self.assertGreater(
            result["std_feasibility_rate"],
            0.0,
        )
        self.assertGreater(
            result["std_best_feasible_energy"],
            0.0,
        )
        self.assertGreater(
            result["std_absolute_gap"],
            0.0,
        )
        self.assertGreater(
            result["std_relative_gap"],
            0.0,
        )

        self.assertEqual(
            mock_run.call_count,
            3,
        )

    def test_single_run_raises_when_no_feasible_sample_exists(
        self,
    ) -> None:
        """Check that a run fails if no feasible sample exists."""
        bqm = dimod.BinaryQuadraticModel(
            {"x": -1.0},
            {},
            0.0,
            dimod.BINARY,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "contains no feasible samples",
        ):
            analyze_simulated_annealing_run(
                bqm=bqm,
                optimal_energy=-1.0,
                num_reads=20,
                feasibility_check=lambda sample: False,
            )

    def test_repeated_runs_requires_at_least_two_runs(
        self,
    ) -> None:
        """Check that repeated analysis rejects fewer than two runs."""
        bqm = dimod.BinaryQuadraticModel(
            {"x": -1.0},
            {},
            0.0,
            dimod.BINARY,
        )

        with self.assertRaisesRegex(
            ValueError,
            "num_runs must be at least 2",
        ):
            analyze_simulated_annealing_runs(
                bqm=bqm,
                optimal_energy=-1.0,
                num_reads=100,
                num_runs=1,
                feasibility_check=lambda sample: True,
            )


if __name__ == "__main__":
    unittest.main()
