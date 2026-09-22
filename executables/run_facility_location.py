"""Run the end-to-end facility location QUBO workflow."""

from pathlib import Path

from src.configuration import load_configuration
from src.constraint_validation import is_feasible
from src.solvers import (solve_exact, solve_simulated_annealing)

from src.build_facility_location import (
    build_assignment_penalty,
    build_cost,
    build_open_facility_penalty,
    build_variables,
)

from src.analysis import (
    compute_feasibility_rate,
    compute_mean_energy,
    compute_optimal_rate,
)

CONFIG_PATH = Path("configs/problem_configuration.yml")


def main() -> None:
    """Run the facility location QUBO workflow."""

    print("=" * 60)
    print("FACILITY LOCATION QUBO")
    print("=" * 60)

    # Load problem configuration
    config = load_configuration(CONFIG_PATH)

    problem_name = config["problem"]["name"]
    opening_costs = config["opening_costs"]
    assignment_costs = config["assignment_costs"]

    assignment_penalty_strength = config["penalties"]["assignment"]
    open_penalty_strength = config["penalties"]["open"]

    num_clients = len(assignment_costs)
    num_facilities = len(opening_costs)

    num_assignment_variables = num_clients * num_facilities
    num_facility_variables = num_facilities
    num_binary_variables = (
        num_assignment_variables + num_facility_variables
    )

    print(f"\nProblem: {problem_name}")
    print(f"Clients: {num_clients}")
    print(f"Candidate facilities: {num_facilities}")

    print("\nBinary variables:")
    print(f"  Assignment variables: {num_assignment_variables}")
    print(f"  Facility variables:   {num_facility_variables}")
    print(f"  Total:                {num_binary_variables}")

    print("\nPenalty strengths:")
    print(f"  Assignment constraint: {assignment_penalty_strength}")
    print(f"  Open facility constraint: {open_penalty_strength}")

    # Build binary variables
    print("\nBuilding binary variables...")

    assignments, facilities = build_variables(
        num_clients=num_clients,
        num_facilities=num_facilities,
    )

    # Build objective function
    print("Building objective function...")

    cost = build_cost(
        assignments=assignments,
        facilities=facilities,
        opening_costs=opening_costs,
        assignment_costs=assignment_costs,
    )

    # Build constraint penalties
    print("Building constraint penalties...")

    assignment_penalty = build_assignment_penalty(
        assignments=assignments,
        penalty_strength=assignment_penalty_strength,
    )

    open_facility_penalty = build_open_facility_penalty(
        assignments=assignments,
        facilities=facilities,
        penalty_strength=open_penalty_strength,
    )

    # Build Hamiltonian
    print("Building QUBO Hamiltonian...")

    hamiltonian = (
        cost
        + assignment_penalty
        + open_facility_penalty
    )

    # Compile PyQUBO model
    print("Compiling PyQUBO model...")

    model = hamiltonian.compile()

    print("\nQUBO model compiled successfully.")
    print("=" * 60)

    qubo, offset = model.to_qubo()

    print("\nCompiled QUBO:")
    print(f"  Number of QUBO terms: {len(qubo)}")
    print(f"  Offset: {offset}")

    print("\nQUBO coefficients:")
    for variables, coefficient in sorted(qubo.items()):
        print(f"  {variables}: {coefficient}")


    bqm = model.to_bqm()

    print("\nBinary Quadratic Model:")
    print(f"  Variable type: {bqm.vartype}")
    print(f"  Number of variables: {bqm.num_variables}")
    print(f"  Number of interactions: {bqm.num_interactions}")
    print(f"  Offset: {bqm.offset}")

    print("\nLinear biases:")
    for variable, bias in bqm.linear.items():
        print(f"  {variable}: {bias}")

    print("\nQuadratic biases:")
    for variables, bias in bqm.quadratic.items():
        print(f"  {variables}: {bias}")

    print("\nSolving BQM with ExactSolver...")

    exact_sampleset = solve_exact(bqm)
    best_exact_sample = exact_sampleset.first

    print("\nBest solution:")
    print(f"  Energy: {best_exact_sample.energy}")

    for variable, value in sorted(best_exact_sample.sample.items()):
        print(f"  {variable}: {value}")

    print("\nSolving BQM with SimulatedAnnealingSampler...")

    num_reads = 100

    sa_sampleset = solve_simulated_annealing(
        bqm=bqm,
        num_reads=num_reads,
    )

    best_sa_sample = sa_sampleset.first

    print("\nBest simulated annealing solution:")
    print(f"  Energy: {best_sa_sample.energy}")

    for variable, value in sorted(best_sa_sample.sample.items()):
        print(f"  {variable}: {value}")

    print("\nSample summary:")
    print(sa_sampleset.aggregate())

    optimal_energy = best_exact_sample.energy

    num_optimal, optimal_rate = compute_optimal_rate(
        sampleset=sa_sampleset,
        optimal_energy=optimal_energy,
    )


    def feasibility_check(sample):
        return is_feasible(
            sample=sample,
            num_clients=num_clients,
            num_facilities=num_facilities,
        )

    num_feasible, feasibility_rate = compute_feasibility_rate(
        sampleset=sa_sampleset,
        feasibility_check=feasibility_check,
    )

    mean_energy = compute_mean_energy(sa_sampleset)

    print(f"\nOptimal energy from ExactSolver: {optimal_energy}")
    print(f"Optimal samples from SA: {num_optimal}/{num_reads}")
    print(f"Optimal rate: {optimal_rate:.1%}")


    print(f"Feasible samples from SA: {num_feasible}/{num_reads}")
    print(f"Feasibility rate: {feasibility_rate:.1%}")

    print(f"Mean energy: {mean_energy:.2f}")


if __name__ == "__main__":
    main()
