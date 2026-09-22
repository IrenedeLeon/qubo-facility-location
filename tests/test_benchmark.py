"""Tests for scalable facility location benchmark utilities."""

import unittest

from src.benchmark import (
    compute_feasible_optimal_cost,
    generate_assignment_costs,
)


class TestBenchmark(unittest.TestCase):
    """Test benchmark instance generation."""

    def test_generate_base_instance(self) -> None:
        """Check the assignment costs for three clients."""
        assignment_costs = generate_assignment_costs(
            num_clients=3,
        )

        expected_costs = [
            [2.0, 5.0],
            [3.0, 2.0],
            [4.0, 3.0],
        ]

        self.assertEqual(
            assignment_costs,
            expected_costs,
        )

    def test_generate_larger_instance_repeats_pattern(self) -> None:
        """Check that larger instances repeat the base cost pattern."""
        assignment_costs = generate_assignment_costs(
            num_clients=7,
        )

        expected_costs = [
            [2.0, 5.0],
            [3.0, 2.0],
            [4.0, 3.0],
            [2.0, 5.0],
            [3.0, 2.0],
            [4.0, 3.0],
            [2.0, 5.0],
        ]

        self.assertEqual(
            assignment_costs,
            expected_costs,
        )

    def test_generate_single_client_instance(self) -> None:
        """Check the smallest valid benchmark instance."""
        assignment_costs = generate_assignment_costs(
            num_clients=1,
        )

        self.assertEqual(
            assignment_costs,
            [[2.0, 5.0]],
        )

    def test_generated_rows_are_independent(self) -> None:
        """Check that repeated cost rows are independent copies."""
        assignment_costs = generate_assignment_costs(
            num_clients=4,
        )

        assignment_costs[0][0] = 999.0

        self.assertEqual(
            assignment_costs[3],
            [2.0, 5.0],
        )

    def test_zero_clients_raises_value_error(self) -> None:
        """Check that zero clients is rejected."""
        with self.assertRaisesRegex(
            ValueError,
            "num_clients must be positive",
        ):
            generate_assignment_costs(
                num_clients=0,
            )

    def test_negative_clients_raises_value_error(self) -> None:
        """Check that negative client counts are rejected."""
        with self.assertRaisesRegex(
            ValueError,
            "num_clients must be positive",
        ):
            generate_assignment_costs(
                num_clients=-1,
            )

    def test_compute_feasible_optimal_cost_reference_instance(
        self,
    ) -> None:
        """Check the known optimum of the reference instance."""
        optimal_cost = compute_feasible_optimal_cost(
            opening_costs=[8.0, 6.0],
            assignment_costs=[
                [2.0, 5.0],
                [3.0, 2.0],
                [4.0, 3.0],
            ],
        )

        self.assertEqual(
            optimal_cost,
            16.0,
        )

    def test_compute_feasible_optimal_cost_prefers_facility_a(
        self,
    ) -> None:
        """Check an instance where opening only facility A is optimal."""
        optimal_cost = compute_feasible_optimal_cost(
            opening_costs=[2.0, 10.0],
            assignment_costs=[
                [1.0, 5.0],
                [1.0, 5.0],
            ],
        )

        self.assertEqual(
            optimal_cost,
            4.0,
        )

    def test_compute_feasible_optimal_cost_prefers_facility_b(
        self,
    ) -> None:
        """Check an instance where opening only facility B is optimal."""
        optimal_cost = compute_feasible_optimal_cost(
            opening_costs=[10.0, 2.0],
            assignment_costs=[
                [5.0, 1.0],
                [5.0, 1.0],
            ],
        )

        self.assertEqual(
            optimal_cost,
            4.0,
        )

    def test_compute_feasible_optimal_cost_prefers_both_facilities(
        self,
    ) -> None:
        """Check an instance where opening both facilities is optimal."""
        optimal_cost = compute_feasible_optimal_cost(
            opening_costs=[1.0, 1.0],
            assignment_costs=[
                [1.0, 10.0],
                [10.0, 1.0],
            ],
        )

        self.assertEqual(
            optimal_cost,
            4.0,
        )

    def test_compute_feasible_optimal_cost_requires_two_facilities(
        self,
    ) -> None:
        """Check that instances with other facility counts are rejected."""
        with self.assertRaisesRegex(
            ValueError,
            "Benchmark requires exactly two facilities",
        ):
            compute_feasible_optimal_cost(
                opening_costs=[8.0, 6.0, 4.0],
                assignment_costs=[
                    [2.0, 5.0, 3.0],
                ],
            )


if __name__ == "__main__":
    unittest.main()
