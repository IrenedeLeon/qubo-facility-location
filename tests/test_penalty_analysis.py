"""Tests for facility location penalty analysis."""

import unittest

from src.penalty_analysis import analyze_penalty_configuration


class TestPenaltyAnalysis(unittest.TestCase):
    """Test penalty-strength behavior for the reference problem."""

    def setUp(self) -> None:
        """Define the reference facility location problem."""
        self.opening_costs = [8.0, 6.0]
        self.assignment_costs = [
            [2.0, 5.0],
            [3.0, 2.0],
            [4.0, 3.0],
        ]

    def test_open_penalty_below_threshold_has_infeasible_ground_state(
        self,
    ) -> None:
        """Check that a weak open-facility penalty allows infeasibility."""
        _, _, num_feasible, feasibility_rate = analyze_penalty_configuration(
            opening_costs=self.opening_costs,
            assignment_costs=self.assignment_costs,
            assignment_penalty_strength=10.0,
            open_penalty_strength=0.0,
        )

        self.assertEqual(num_feasible, 0)
        self.assertEqual(feasibility_rate, 0.0)

    def test_open_penalty_at_threshold_has_degenerate_ground_states(
        self,
    ) -> None:
        """Check degeneracy at the open-facility penalty threshold."""
        ground_energy, num_ground, num_feasible, feasibility_rate = (
            analyze_penalty_configuration(
                opening_costs=self.opening_costs,
                assignment_costs=self.assignment_costs,
                assignment_penalty_strength=10.0,
                open_penalty_strength=3.0,
            )
        )

        self.assertAlmostEqual(ground_energy, 16.0)
        self.assertEqual(num_ground, 3)
        self.assertEqual(num_feasible, 1)
        self.assertAlmostEqual(feasibility_rate, 1.0 / 3.0)

    def test_open_penalty_above_threshold_has_only_feasible_ground_states(
        self,
    ) -> None:
        """Check feasibility above the open-facility penalty threshold."""
        ground_energy, num_ground, num_feasible, feasibility_rate = (
            analyze_penalty_configuration(
                opening_costs=self.opening_costs,
                assignment_costs=self.assignment_costs,
                assignment_penalty_strength=10.0,
                open_penalty_strength=3.01,
            )
        )

        self.assertAlmostEqual(ground_energy, 16.0)
        self.assertEqual(num_ground, 1)
        self.assertEqual(num_feasible, 1)
        self.assertEqual(feasibility_rate, 1.0)

    def test_assignment_penalty_below_threshold_has_infeasible_ground_state(
        self,
    ) -> None:
        """Check that a weak assignment penalty allows infeasibility."""
        _, _, num_feasible, feasibility_rate = analyze_penalty_configuration(
            opening_costs=self.opening_costs,
            assignment_costs=self.assignment_costs,
            assignment_penalty_strength=0.0,
            open_penalty_strength=10.0,
        )

        self.assertEqual(num_feasible, 0)
        self.assertEqual(feasibility_rate, 0.0)

    def test_assignment_penalty_at_threshold_has_degenerate_ground_states(
        self,
    ) -> None:
        """Check degeneracy at the assignment penalty threshold."""
        ground_energy, num_ground, num_feasible, feasibility_rate = (
            analyze_penalty_configuration(
                opening_costs=self.opening_costs,
                assignment_costs=self.assignment_costs,
                assignment_penalty_strength=16.0 / 3.0,
                open_penalty_strength=10.0,
            )
        )

        self.assertAlmostEqual(ground_energy, 16.0)
        self.assertEqual(num_ground, 2)
        self.assertEqual(num_feasible, 1)
        self.assertEqual(feasibility_rate, 0.5)

    def test_assignment_penalty_above_threshold_has_only_feasible_ground_states(
        self,
    ) -> None:
        """Check feasibility above the assignment penalty threshold."""
        ground_energy, num_ground, num_feasible, feasibility_rate = (
            analyze_penalty_configuration(
                opening_costs=self.opening_costs,
                assignment_costs=self.assignment_costs,
                assignment_penalty_strength=5.34,
                open_penalty_strength=10.0,
            )
        )

        self.assertAlmostEqual(ground_energy, 16.0)
        self.assertEqual(num_ground, 1)
        self.assertEqual(num_feasible, 1)
        self.assertEqual(feasibility_rate, 1.0)


if __name__ == "__main__":
    unittest.main()
