"""Penalty analysis utilities for the facility location QUBO."""

from src.build_facility_location import (
    build_assignment_penalty,
    build_cost,
    build_open_facility_penalty,
    build_variables,
)
from src.constraint_validation import is_feasible
from src.solvers import solve_exact


ENERGY_TOLERANCE = 1e-9


def analyze_penalty_configuration(
    opening_costs: list[float],
    assignment_costs: list[list[float]],
    assignment_penalty_strength: float,
    open_penalty_strength: float,
) -> tuple[float, int, int, float]:
    """Analyze ground states for a pair of penalty strengths.

    Builds the facility location QUBO using the specified assignment and
    open-facility penalty strengths. The resulting BQM is solved exactly,
    and the feasibility of every ground state is evaluated.

    Args:
        opening_costs: Cost of opening each candidate facility.
        assignment_costs: Cost of assigning each client to each facility.
        assignment_penalty_strength: Penalty strength enforcing exactly one
            facility assignment per client.
        open_penalty_strength: Penalty strength preventing assignments to
            closed facilities.

    Returns:
        A tuple containing:
            - Ground-state energy.
            - Number of ground states.
            - Number of feasible ground states.
            - Fraction of ground states that are feasible.
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
    bqm = model.to_bqm()

    sampleset = solve_exact(bqm)
    ground_energy = sampleset.first.energy

    ground_states = [
        record
        for record in sampleset.data(fields=["sample", "energy"])
        if abs(record.energy - ground_energy) < ENERGY_TOLERANCE
    ]

    num_ground_states = len(ground_states)

    num_feasible_ground_states = sum(
        is_feasible(
            sample=record.sample,
            num_clients=num_clients,
            num_facilities=num_facilities,
        )
        for record in ground_states
    )

    ground_feasibility_rate = (
        num_feasible_ground_states / num_ground_states
    )

    return (
        ground_energy,
        num_ground_states,
        num_feasible_ground_states,
        ground_feasibility_rate,
    )
