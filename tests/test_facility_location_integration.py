"""Integration tests for the facility location workflow."""

import unittest

from src.build_facility_location import (
    build_assignment_penalty,
    build_cost,
    build_open_facility_penalty,
    build_variables,
)
from src.constraint_validation import is_feasible
from src.solvers import solve_exact


class TestFacilityLocationIntegration(unittest.TestCase):
    """Integration tests for the complete facility location formulation."""

    def test_exact_solver_finds_known_feasible_optimum(self) -> None:
        """The reference problem should have optimal energy 16."""
        opening_costs = [8, 6]
        assignment_costs = [
            [2, 5],
            [3, 2],
            [4, 3],
        ]

        num_clients = 3
        num_facilities = 2

        assignments, facilities = build_variables(
            num_clients=num_clients,
            num_facilities=num_facilities,
        )

        cost = build_cost(
            assignments=assignments,
            facilities=facilities,
            opening_costs=opening_costs,
            assignment_costs=assignment_costs,
        )

        assignment_penalty = build_assignment_penalty(
            assignments=assignments,
            penalty_strength=10.0,
        )

        open_penalty = build_open_facility_penalty(
            assignments=assignments,
            facilities=facilities,
            penalty_strength=10.0,
        )

        hamiltonian = (
            cost
            + assignment_penalty
            + open_penalty
        )

        bqm = hamiltonian.compile().to_bqm()

        sampleset = solve_exact(bqm)
        best = sampleset.first

        self.assertAlmostEqual(best.energy, 16.0)

        self.assertTrue(
            is_feasible(
                sample=best.sample,
                num_clients=num_clients,
                num_facilities=num_facilities,
            )
        )

        self.assertEqual(best.sample["facility_0"], 0)
        self.assertEqual(best.sample["facility_1"], 1)

        for client_index in range(num_clients):
            self.assertEqual(
                best.sample[f"assignment_{client_index}_0"],
                0,
            )
            self.assertEqual(
                best.sample[f"assignment_{client_index}_1"],
                1,
            )
