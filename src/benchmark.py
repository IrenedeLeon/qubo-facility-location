"""Benchmark utilities for scalable facility location instances."""


BASE_ASSIGNMENT_COSTS = [
    [2.0, 5.0],
    [3.0, 2.0],
    [4.0, 3.0],
]

OPENING_COSTS = [8.0, 6.0]


def generate_assignment_costs(
    num_clients: int,
) -> list[list[float]]:
    """Generate deterministic assignment costs for a benchmark instance.

    The base assignment-cost pattern is repeated until the requested
    number of clients is reached.

    Args:
        num_clients: Number of clients in the benchmark instance.

    Returns:
        Assignment-cost matrix with one row per client.

    Raises:
        ValueError: If num_clients is not positive.
    """
    if num_clients < 1:
        raise ValueError("num_clients must be positive.")

    return [
        BASE_ASSIGNMENT_COSTS[
            client_index % len(BASE_ASSIGNMENT_COSTS)
        ].copy()
        for client_index in range(num_clients)
    ]

def compute_feasible_optimal_cost(
    opening_costs: list[float],
    assignment_costs: list[list[float]],
) -> float:
    """Compute the optimal feasible cost for a two-facility instance.

    Args:
        opening_costs: Cost of opening the two candidate facilities.
        assignment_costs: Assignment costs for each client.

    Returns:
        Minimum feasible economic cost.

    Raises:
        ValueError: If the instance does not contain exactly two facilities.
    """
    if len(opening_costs) != 2:
        raise ValueError(
            "Benchmark requires exactly two facilities."
        )

    cost_a = (
        opening_costs[0]
        + sum(costs[0] for costs in assignment_costs)
    )

    cost_b = (
        opening_costs[1]
        + sum(costs[1] for costs in assignment_costs)
    )

    cost_both = (
        sum(opening_costs)
        + sum(min(costs) for costs in assignment_costs)
    )

    return min(
        cost_a,
        cost_b,
        cost_both,
    )
