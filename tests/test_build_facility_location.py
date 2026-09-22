"""Tests for the facility location QUBO formulation."""

import unittest
import dimod

from src.build_facility_location import (
    build_assignment_penalty,
    build_cost,
    build_open_facility_penalty,
    build_variables,
    build_facility_location_bqm,
)


class TestBuildFacilityLocation(unittest.TestCase):
    """Tests for the facility location QUBO formulation."""

    def setUp(self) -> None:
        """Create the small reference problem used across tests."""
        self.num_clients = 3
        self.num_facilities = 2

        self.opening_costs = [8, 6]
        self.assignment_costs = [
            [2, 5],
            [3, 2],
            [4, 3],
        ]

        self.assignments, self.facilities = build_variables(
            num_clients=self.num_clients,
            num_facilities=self.num_facilities,
        )

    def test_build_variables(self) -> None:
        """Build the expected number of assignment and facility variables."""
        self.assertEqual(len(self.assignments), 3)
        self.assertTrue(
            all(len(row) == 2 for row in self.assignments)
        )
        self.assertEqual(len(self.facilities), 2)

    @staticmethod
    def evaluate_expression(expression, sample) -> float:
        """Compile and evaluate a PyQUBO expression for a binary sample."""
        bqm = expression.compile().to_bqm()
        return bqm.energy(sample)

    def test_cost_for_known_solution(self) -> None:
        """Compute the expected economic cost for a known solution."""
        cost = build_cost(
            assignments=self.assignments,
            facilities=self.facilities,
            opening_costs=self.opening_costs,
            assignment_costs=self.assignment_costs,
        )

        sample = {
            "assignment_0_0": 0,
            "assignment_0_1": 1,
            "assignment_1_0": 0,
            "assignment_1_1": 1,
            "assignment_2_0": 0,
            "assignment_2_1": 1,
            "facility_0": 0,
            "facility_1": 1,
        }

        energy = self.evaluate_expression(cost, sample)

        self.assertAlmostEqual(energy, 16.0)

    def test_assignment_penalty_is_zero_for_valid_assignment(self) -> None:
        """A valid one-hot assignment should have zero penalty."""
        penalty = build_assignment_penalty(
            assignments=self.assignments,
            penalty_strength=10.0,
        )

        sample = {
            "assignment_0_0": 0,
            "assignment_0_1": 1,
            "assignment_1_0": 1,
            "assignment_1_1": 0,
            "assignment_2_0": 0,
            "assignment_2_1": 1,
        }

        energy = self.evaluate_expression(penalty, sample)

        self.assertAlmostEqual(energy, 0.0)

    def test_assignment_penalty_for_unassigned_client(self) -> None:
        """One unassigned client should incur one assignment penalty."""
        penalty = build_assignment_penalty(
            assignments=self.assignments,
            penalty_strength=10.0,
        )

        sample = {
            "assignment_0_0": 0,
            "assignment_0_1": 0,
            "assignment_1_0": 1,
            "assignment_1_1": 0,
            "assignment_2_0": 0,
            "assignment_2_1": 1,
        }

        energy = self.evaluate_expression(penalty, sample)

        self.assertAlmostEqual(energy, 10.0)

    def test_assignment_penalty_for_double_assignment(self) -> None:
        """One doubly assigned client should incur one assignment penalty."""
        penalty = build_assignment_penalty(
            assignments=self.assignments,
            penalty_strength=10.0,
        )

        sample = {
            "assignment_0_0": 1,
            "assignment_0_1": 1,
            "assignment_1_0": 1,
            "assignment_1_1": 0,
            "assignment_2_0": 0,
            "assignment_2_1": 1,
        }

        energy = self.evaluate_expression(penalty, sample)

        self.assertAlmostEqual(energy, 10.0)

    def test_open_facility_penalty_for_closed_facility(self) -> None:
        """Assignments to closed facilities should be penalized."""
        penalty = build_open_facility_penalty(
            assignments=self.assignments,
            facilities=self.facilities,
            penalty_strength=10.0,
        )

        sample = {
            "assignment_0_0": 0,
            "assignment_0_1": 1,
            "assignment_1_0": 0,
            "assignment_1_1": 1,
            "assignment_2_0": 0,
            "assignment_2_1": 1,
            "facility_0": 0,
            "facility_1": 0,
        }

        energy = self.evaluate_expression(penalty, sample)

        self.assertAlmostEqual(energy, 30.0)

    def test_build_facility_location_bqm_has_expected_size(self) -> None:
        """Check the number of variables in the complete BQM."""
        bqm = build_facility_location_bqm(
            opening_costs=[8.0, 6.0],
            assignment_costs=[
                [2.0, 5.0],
                [3.0, 2.0],
                [4.0, 3.0],
            ],
            assignment_penalty_strength=10.0,
            open_penalty_strength=10.0,
        )

        self.assertEqual(
            bqm.num_variables,
            8,
        )


    def test_build_facility_location_bqm_preserves_known_optimum(
        self,
    ) -> None:
        """Check the exact optimum of the reference BQM."""
        bqm = build_facility_location_bqm(
            opening_costs=[8.0, 6.0],
            assignment_costs=[
                [2.0, 5.0],
                [3.0, 2.0],
                [4.0, 3.0],
            ],
            assignment_penalty_strength=10.0,
            open_penalty_strength=10.0,
        )

        sampleset = dimod.ExactSolver().sample(bqm)

        self.assertAlmostEqual(
            sampleset.first.energy,
            16.0,
        )
