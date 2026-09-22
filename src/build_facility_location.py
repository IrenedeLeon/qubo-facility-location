"""QUBO formulation utilities for the facility location problem."""

from pyqubo import Binary

import dimod

def build_variables(num_clients: int, num_facilities: int):
    """Build binary decision variables for the facility location problem.

    Creates binary variables representing facility opening decisions and
    client-to-facility assignments.

    Args:
        num_clients: Number of clients that must be assigned to a facility.
        num_facilities: Number of candidate facilities that can be opened.

    Returns:
        A tuple containing:
            - assignments: Two-dimensional list where assignments[i][j] is 1
              if client i is assigned to facility j, and 0 otherwise.
            - facilities: List where facilities[j] is 1 if facility j is open,
              and 0 otherwise.
    """
    facilities = [
        Binary(f"facility_{j}")
        for j in range(num_facilities)
    ]

    assignments = [
        [
            Binary(f"assignment_{i}_{j}")
            for j in range(num_facilities)
        ]
        for i in range(num_clients)
    ]

    return assignments, facilities


def build_assignment_penalty(
    assignments,
    penalty_strength: float = 10.0,
):
    """Build the penalty enforcing one facility assignment per client.

    Penalizes solutions in which a client is assigned to either zero or
    multiple facilities. The penalty is zero when each client is assigned
    to exactly one facility.

    Args:
        assignments: Two-dimensional list of binary assignment variables,
            indexed by client and facility.
        penalty_strength: Weight applied to violations of the assignment
            constraint.

    Returns:
        Symbolic PyQUBO expression representing the assignment penalty.
    """
    penalty = 0

    for client_assignments in assignments:
        penalty += (sum(client_assignments) - 1) ** 2

    return penalty_strength * penalty


def build_open_facility_penalty(
    assignments,
    facilities,
    penalty_strength: float = 10.0,
):
    """Build the penalty preventing assignments to closed facilities.

    A penalty is applied whenever a client is assigned to a facility whose
    opening variable is zero.

    Args:
        assignments: Two-dimensional list of binary assignment variables,
            indexed by client and facility.
        facilities: List of binary facility-opening variables.
        penalty_strength: Weight applied to assignments made to closed
            facilities.

    Returns:
        Symbolic PyQUBO expression representing the facility-opening penalty.
    """
    penalty = 0

    for client_assignments in assignments:
        for facility_index, assignment in enumerate(client_assignments):
            penalty += assignment * (1 - facilities[facility_index])

    return penalty_strength * penalty


def build_cost(
    assignments,
    facilities,
    opening_costs,
    assignment_costs,
):
    """Build the objective cost for the facility location problem.

    The objective combines the cost of opening facilities with the cost of
    assigning each client to a facility.

    Args:
        assignments: Two-dimensional list of binary assignment variables,
            indexed by client and facility.
        facilities: List of binary facility-opening variables.
        opening_costs: Opening cost associated with each facility.
        assignment_costs: Two-dimensional cost matrix where element [i][j]
            is the cost of assigning client i to facility j.

    Returns:
        Symbolic PyQUBO expression representing the total facility location
        cost.
    """
    opening_cost = sum(
        cost * facility
        for cost, facility in zip(opening_costs, facilities)
    )

    assignment_cost = sum(
        assignment_costs[client_index][facility_index]
        * assignments[client_index][facility_index]
        for client_index in range(len(assignments))
        for facility_index in range(len(facilities))
    )

    return opening_cost + assignment_cost

def build_facility_location_bqm(
        opening_costs: list[float],
        assignment_costs: list[list[float]],
        assignment_penalty_strength: float,
        open_penalty_strength: float,
    ):
        """Build the binary quadratic model for a facility location instance.

        Args:
            opening_costs: Cost of opening each candidate facility.
            assignment_costs: Cost of assigning each client to each facility.
            assignment_penalty_strength: Penalty strength enforcing exactly one
                facility assignment per client.
            open_penalty_strength: Penalty strength preventing assignments to
                closed facilities.

        Returns:
            Binary quadratic model representing the facility location problem.
        """
        num_clients = len(assignment_costs)
        num_facilities = len(opening_costs)

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
            penalty_strength=assignment_penalty_strength,
        )

        open_facility_penalty = build_open_facility_penalty(
            assignments=assignments,
            facilities=facilities,
            penalty_strength=open_penalty_strength,
        )

        hamiltonian = (
            cost
            + assignment_penalty
            + open_facility_penalty
        )

        model = hamiltonian.compile()

        return model.to_bqm()
