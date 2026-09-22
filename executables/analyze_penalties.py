"""Analyze penalty strengths for the facility location QUBO."""

from pathlib import Path

from src.configuration import load_configuration
from src.penalty_analysis import analyze_penalty_configuration


CONFIG_PATH = Path("configs/problem_configuration.yml")


def print_analysis_header(
    title: str,
    fixed_penalty_name: str,
    fixed_penalty_value: float,
) -> None:
    """Print the header for a penalty analysis.

    Args:
        title: Title describing the penalty analysis.
        fixed_penalty_name: Name of the penalty kept fixed.
        fixed_penalty_value: Value of the penalty kept fixed.
    """
    print()
    print("=" * 152)
    print(title)
    print("=" * 152)
    print(f"{fixed_penalty_name}: {fixed_penalty_value}")

    print()
    print(
        f"{'penalty strength':>16} "
        f"{'energy':>10} "
        f"{'different ground states':>24} "
        f"{'feasible ground states':>12} "
        f"{'rate':>10} "
        f"{'all feasible':>14}"
    )
    print("-" * 112)


def print_analysis_row(
    penalty_strength: float,
    ground_energy: float,
    num_ground_states: int,
    num_feasible_ground_states: int,
    ground_feasibility_rate: float,
) -> None:
    """Print one row of penalty-analysis results.

    Args:
        penalty_strength: Penalty strength being evaluated.
        ground_energy: Minimum energy found by exact enumeration.
        num_ground_states: Number of states with minimum energy.
        num_feasible_ground_states: Number of feasible minimum-energy states.
        ground_feasibility_rate: Fraction of ground states that are feasible.
    """
    all_ground_states_feasible = (
        num_feasible_ground_states == num_ground_states
    )

    print(
        f"{penalty_strength:>16.2f} "
        f"{ground_energy:>10.2f} "
        f"{num_ground_states:>24} "
        f"{num_feasible_ground_states:>22} "
        f"{ground_feasibility_rate:>9.1%} "
        f"{str(all_ground_states_feasible):>14}"
    )


def main() -> None:
    """Run open-facility and assignment penalty analyses."""
    config = load_configuration(CONFIG_PATH)

    opening_costs = config["opening_costs"]
    assignment_costs = config["assignment_costs"]

    default_assignment_penalty = config["penalties"]["assignment"]
    default_open_penalty = config["penalties"]["open"]

    open_penalty_values = [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        3.01,
        3.1,
        3.5,
        4.0,
        5.0,
        10.0,
    ]

    print_analysis_header(
        title="OPEN-FACILITY PENALTY ANALYSIS",
        fixed_penalty_name="Fixed assignment penalty",
        fixed_penalty_value=default_assignment_penalty,
    )

    for open_penalty_strength in open_penalty_values:
        results = analyze_penalty_configuration(
            opening_costs=opening_costs,
            assignment_costs=assignment_costs,
            assignment_penalty_strength=default_assignment_penalty,
            open_penalty_strength=open_penalty_strength,
        )

        print_analysis_row(
            open_penalty_strength,
            *results,
        )

    assignment_penalty_values = [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        5.2,
        5.3,
        5.33,
        5.3333333333,
        5.34,
        5.5,
        6.0,
        8.0,
        10.0,
    ]

    print_analysis_header(
        title="ASSIGNMENT PENALTY ANALYSIS",
        fixed_penalty_name="Fixed open-facility penalty",
        fixed_penalty_value=default_open_penalty,
    )

    for assignment_penalty_strength in assignment_penalty_values:
        results = analyze_penalty_configuration(
            opening_costs=opening_costs,
            assignment_costs=assignment_costs,
            assignment_penalty_strength=assignment_penalty_strength,
            open_penalty_strength=default_open_penalty,
        )

        print_analysis_row(
            assignment_penalty_strength,
            *results,
        )


if __name__ == "__main__":
    main()
