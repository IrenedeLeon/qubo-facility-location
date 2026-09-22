"""Plot facility location scalability results."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_PATH = Path("results/scalability_analysis.csv")
FIGURES_DIR = Path("results/figures")

OPTIMAL_RECOVERY_PATH = (
    FIGURES_DIR / "scalability_optimal_recovery"
)
OPTIMALITY_GAP_PATH = (
    FIGURES_DIR / "scalability_optimality_gap"
)
RUNTIME_PATH = (
    FIGURES_DIR / "scalability_runtime"
)

FIGURE_DPI = 300


def load_results() -> pd.DataFrame:
    """Load scalability benchmark results.

    Returns:
        DataFrame containing scalability experiment results.

    Raises:
        FileNotFoundError: If the results CSV does not exist.
    """
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Results file not found: {RESULTS_PATH}"
        )

    return pd.read_csv(RESULTS_PATH)


def save_figure(
    figure: plt.Figure,
    output_path: Path,
) -> None:
    """Save a figure in PNG and SVG formats.

    Args:
        figure: Matplotlib figure to save.
        output_path: Output path without a file extension.
    """
    figure.savefig(
        output_path.with_suffix(".png"),
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    figure.savefig(
        output_path.with_suffix(".svg"),
        bbox_inches="tight",
    )


def plot_optimal_recovery(
    results: pd.DataFrame,
) -> None:
    """Plot optimal-sample and run-success rates.

    Args:
        results: Scalability benchmark results.
    """
    figure, axis = plt.subplots(
        figsize=(8, 5),
    )

    variables = results["num_variables"]

    axis.plot(
        variables,
        100 * results["mean_optimal_rate"],
        marker="o",
        label="Optimal sample rate",
    )

    axis.plot(
        variables,
        100 * results["run_success_rate"],
        marker="s",
        label="Run success rate",
    )

    axis.set_xlabel(
        "Number of binary variables"
    )
    axis.set_ylabel(
        "Rate (%)"
    )
    axis.set_title(
        "Exact-optimum recovery under a fixed sampling budget"
    )

    axis.set_ylim(
        bottom=0,
        top=105,
    )

    axis.grid(
        alpha=0.3,
    )

    axis.legend()

    figure.tight_layout()

    save_figure(
        figure=figure,
        output_path=OPTIMAL_RECOVERY_PATH,
    )

    plt.close(figure)


def plot_optimality_gap(
    results: pd.DataFrame,
) -> None:
    """Plot the relative optimality gap.

    Args:
        results: Scalability benchmark results.
    """
    figure, axis = plt.subplots(
        figsize=(8, 5),
    )

    variables = results["num_variables"]

    mean_gap = (
        100
        * results["mean_relative_gap"]
    )

    std_gap = (
        100
        * results["std_relative_gap"]
    )

    axis.errorbar(
        variables,
        mean_gap,
        yerr=std_gap,
        marker="o",
        capsize=4,
    )

    axis.set_xlabel(
        "Number of binary variables"
    )
    axis.set_ylabel(
        "Relative optimality gap (%)"
    )
    axis.set_title(
        "Best feasible solution quality"
    )

    axis.set_ylim(
        bottom=0,
    )

    axis.grid(
        alpha=0.3,
    )

    figure.tight_layout()

    save_figure(
        figure=figure,
        output_path=OPTIMALITY_GAP_PATH,
    )

    plt.close(figure)


def plot_runtime(
    results: pd.DataFrame,
) -> None:
    """Plot exact and simulated annealing runtimes.

    Args:
        results: Scalability benchmark results.
    """
    figure, axis = plt.subplots(
        figsize=(8, 5),
    )

    variables = results["num_variables"]

    exact_results = results[
        results["exact_solver_executed"]
    ]

    axis.plot(
        exact_results["num_variables"],
        exact_results["exact_runtime_seconds"],
        marker="o",
        label="ExactSolver",
    )

    axis.plot(
        variables,
        results["sa_runtime_seconds"],
        marker="s",
        label="Simulated annealing",
    )

    axis.set_xlabel(
        "Number of binary variables"
    )
    axis.set_ylabel(
        "Runtime (s)"
    )
    axis.set_title(
        "Observed solver runtime"
    )

    axis.set_yscale(
        "log"
    )

    axis.grid(
        alpha=0.3,
    )

    axis.legend()

    figure.tight_layout()

    save_figure(
        figure=figure,
        output_path=RUNTIME_PATH,
    )

    plt.close(figure)


def main() -> None:
    """Generate scalability benchmark figures."""
    print()
    print(
        "FACILITY LOCATION SCALABILITY PLOTS"
    )
    print("=" * 45)

    results = load_results()

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_optimal_recovery(
        results=results,
    )
    print(
        f"Saved: {OPTIMAL_RECOVERY_PATH}.png/.svg"
    )

    plot_optimality_gap(
        results=results,
    )
    print(
        f"Saved: {OPTIMALITY_GAP_PATH}.png/.svg"
    )

    plot_runtime(
        results=results,
    )
    print(
        f"Saved: {RUNTIME_PATH}.png/.svg"
    )


if __name__ == "__main__":
    main()
