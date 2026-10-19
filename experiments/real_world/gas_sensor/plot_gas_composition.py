"""Plot the gas composition of the ten batches from saved counts."""
import argparse
from pathlib import Path
import pandas as pd
from compute_rolling_split_explained_variance import plot_gas_composition, validate_reported_gas_composition

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
    args = parser.parse_args()
    counts = pd.read_csv(args.input_csv).pivot(index="batch", columns="gas", values="n_obs")
    validate_reported_gas_composition(counts)
    plot_gas_composition(counts, args.figures_dir)
