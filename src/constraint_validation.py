"""Constraint validation utilities for the facility location problem."""

from collections.abc import Mapping


def is_feasible(
    sample: Mapping[str, int],
    num_clients: int,
    num_facilities: int,
) -> bool:
    """Check whether a facility location sample is feasible.

    Args:
        sample: Mapping from variable names to binary values.
        num_clients: Number of clients in the problem.
        num_facilities: Number of candidate facilities.

    Returns:
        True if all problem constraints are satisfied, otherwise False.
    """
    for client_index in range(num_clients):
        total_assignments = sum(
            sample[f"assignment_{client_index}_{facility_index}"]
            for facility_index in range(num_facilities)
        )

        if total_assignments != 1:
            return False

    for client_index in range(num_clients):
        for facility_index in range(num_facilities):
            assignment = sample[
                f"assignment_{client_index}_{facility_index}"
            ]
            facility = sample[f"facility_{facility_index}"]

            if assignment > facility:
                return False

    return True
