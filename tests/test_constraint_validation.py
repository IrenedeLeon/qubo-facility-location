"""Tests for facility location constraint validation."""

import unittest

from src.constraint_validation import is_feasible


NUM_CLIENTS = 3
NUM_FACILITIES = 2


class TestConstraintValidation(unittest.TestCase):
    """Tests for facility location constraint validation."""

    def test_feasible_solution(self) -> None:
        """A valid assignment to an open facility should be feasible."""
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

        self.assertTrue(
            is_feasible(
                sample=sample,
                num_clients=NUM_CLIENTS,
                num_facilities=NUM_FACILITIES,
            )
        )

    def test_multiple_assignments_are_infeasible(self) -> None:
        """A client assigned to multiple facilities should be infeasible."""
        sample = {
            "assignment_0_0": 1,
            "assignment_0_1": 1,
            "assignment_1_0": 0,
            "assignment_1_1": 1,
            "assignment_2_0": 0,
            "assignment_2_1": 1,
            "facility_0": 1,
            "facility_1": 1,
        }

        self.assertFalse(
            is_feasible(
                sample=sample,
                num_clients=NUM_CLIENTS,
                num_facilities=NUM_FACILITIES,
            )
        )

    def test_unassigned_client_is_infeasible(self) -> None:
        """A client with no facility assignment should be infeasible."""
        sample = {
            "assignment_0_0": 0,
            "assignment_0_1": 0,
            "assignment_1_0": 0,
            "assignment_1_1": 1,
            "assignment_2_0": 0,
            "assignment_2_1": 1,
            "facility_0": 0,
            "facility_1": 1,
        }

        self.assertFalse(
            is_feasible(
                sample=sample,
                num_clients=NUM_CLIENTS,
                num_facilities=NUM_FACILITIES,
            )
        )

    def test_assignment_to_closed_facility_is_infeasible(self) -> None:
        """An assignment to a closed facility should be infeasible."""
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

        self.assertFalse(
            is_feasible(
                sample=sample,
                num_clients=NUM_CLIENTS,
                num_facilities=NUM_FACILITIES,
            )
        )


if __name__ == "__main__":
    unittest.main()
