"""Plot simulated annealing sampling-budget results."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_PATH = Path("results/annealing_analysis.csv")
FIGURES_DIR = Path("results/figures")

OPTIMAL_RATE_PATH = (
    FIGURES_DIR / "annealing_optimal_rate"
)
MEAN_ENERGY_PATH = (
    FIGURES_DIR / "annealing_mean_energy"
)

FIGURE_DPI = 300


def load_results() -> pd.DataFrame:
    """Load simulated annealing analysis results.

    Returns:
        DataFrame containing simulated annealing results.

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


def plot_optimal_rate(
    results: pd.DataFrame,
) -> None:
    """Plot optimal sample rate against number of reads.

    Args:
        results: Simulated annealing analysis results.
    """
    figure, axis = plt.subplots(
        figsize=(8, 5),
    )

    mean_optimal_rate = (
        100
        * results["mean_optimal_rate"]
    )

    std_optimal_rate = (
        100
        * results["std_optimal_rate"]
    )

    axis.errorbar(
        results["num_reads"],
        mean_optimal_rate,
        yerr=std_optimal_rate,
        marker="o",
        capsize=4,
    )

    axis.set_xlabel(
        "Number of reads per run"
    )
    axis.set_ylabel(
        "Optimal sample rate (%)"
    )
    axis.set_title(
        "Optimal-sample rate vs sampling budget"
    )

    axis.set_ylim(
        bottom=0,
        top=100,
    )

    axis.set_xscale(
        "log"
    )

    axis.grid(
        alpha=0.3,
    )

    figure.tight_layout()

    save_figure(
        figure=figure,
        output_path=OPTIMAL_RATE_PATH,
    )

    plt.close(figure)


def plot_mean_energy(
    results: pd.DataFrame,
) -> None:
    """Plot mean sampled energy against number of reads.

    Args:
        results: Simulated annealing analysis results.
    """
    figure, axis = plt.subplots(
        figsize=(8, 5),
    )

    axis.errorbar(
        results["num_reads"],
        results["mean_energy"],
        yerr=results["std_mean_energy"],
        marker="o",
        capsize=4,
        label="Mean sampled energy",
    )

    if "optimal_energy" in results.columns:
        optimal_energy = results[
            "optimal_energy"
        ].iloc[0]

        axis.axhline(
            optimal_energy,
            linestyle="--",
            label="Exact optimum",
        )

    axis.set_xlabel(
        "Number of reads per run"
    )
    axis.set_ylabel(
        "Energy"
    )
    axis.set_title(
        "Mean sampled energy vs sampling budget"
    )

    axis.set_xscale(
        "log"
    )

    axis.grid(
        alpha=0.3,
    )

    axis.legend()

    figure.tight_layout()

    save_figure(
        figure=figure,
        output_path=MEAN_ENERGY_PATH,
    )

    plt.close(figure)


def main() -> None:
    """Generate simulated annealing analysis figures."""
    print()
    print(
        "SIMULATED ANNEALING ANALYSIS PLOTS"
    )
    print("=" * 45)

    results = load_results()

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_optimal_rate(
        results=results,
    )
    print(
        f"Saved: {OPTIMAL_RATE_PATH}.png/.svg"
    )

    plot_mean_energy(
        results=results,
    )
    print(
        f"Saved: {MEAN_ENERGY_PATH}.png/.svg"
    )


if __name__ == "__main__":
    main()
