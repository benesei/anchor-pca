"""Plot source-period scree curves from the saved temporal-experiment CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def plot_source_scree(scree: pd.DataFrame, figures_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for index, (_, group) in enumerate(scree.groupby("source_period", sort=True)):
        group = group.sort_values("component")
        periods = group[["start_year", "end_year"]].drop_duplicates()
        if len(periods) != 1 or group.component.duplicated().any():
            raise ValueError("Each source needs one year range and unique components.")
        start, end = periods.iloc[0]
        cumulative = group.explained_variance_fraction.cumsum()
        mask = group.component <= 10
        style = dict(marker="o", markersize=3, linewidth=1.4, color=colors(index))
        axes[0].plot(
            group.loc[mask, "component"],
            100 * group.loc[mask, "explained_variance_fraction"],
            label=f"{int(start)}-{int(end)}", **style,
        )
        axes[1].plot(group.loc[mask, "component"], 100 * cumulative[mask], **style)

    for axis in axes:
        axis.axvline(2, color="black", linestyle="--", linewidth=1.1)
        axis.set_xticks(range(1, 11))
        axis.grid(alpha=0.2)
        axis.set_xlabel("Principal-component rank")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Variance explained by component (%)")
    axes[0].set_title("Source-domain scree curves")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].set_ylabel("Cumulative explained variance (%)")
    axes[1].set_title("Source-domain cumulative variance")
    axes[1].annotate(
        "$k=2$", xy=(2, axes[1].get_ylim()[1]), xytext=(5, -5),
        textcoords="offset points", ha="left", va="top", fontsize=8,
    )
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "fama_french_49_source_scree.pdf", dpi=300)
    plt.close(fig)


def main() -> None:
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=folder / "results/source_scree.csv")
    parser.add_argument("--figures-dir", type=Path, default=folder / "figures")
    args = parser.parse_args()
    plot_source_scree(pd.read_csv(args.input_csv), args.figures_dir)


if __name__ == "__main__":
    main()
