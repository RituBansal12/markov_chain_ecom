"""
Visualize Markov model results: transition matrix heatmap, absorption bars, expected steps bar.

Inputs:
- data_clean/transition_matrix.csv
- reports/absorption_probabilities.csv
- reports/expected_steps_to_absorption.csv

Outputs:
- visualizations/transition_heatmap.png
- visualizations/absorption_from_START.png (if START row present)
- visualizations/expected_steps_by_state.png

CLI:
- --matrix, --absorption, --expected-steps

Usage:
  python -m modelling_scripts.05_visualize_results
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from cleaning_scripts.common import get_paths, configure_logging


def save_heatmap(tm: pd.DataFrame, out_path: Path, title: str = "Transition Matrix (Probabilities)"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(tm, annot=False, cmap="Blues", fmt=".2f")
    plt.title(title)
    plt.xlabel("To State")
    plt.ylabel("From State")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_bar(series: pd.Series, out_path: Path, title: str, xlabel: str, ylabel: str):
    plt.figure(figsize=(6, 4))
    series.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize Markov model results")
    parser.add_argument("--matrix", type=str, default=None, help="Path to transition matrix CSV")
    parser.add_argument("--absorption", type=str, default=None, help="Path to absorption probabilities CSV")
    parser.add_argument("--expected-steps", type=str, default=None, help="Path to expected steps CSV")
    args = parser.parse_args()

    log = configure_logging("visualize")

    root, _, clean, reports = get_paths()
    matrix_path = Path(args.matrix) if args.matrix else (clean / "transition_matrix.csv")
    absorb_path = Path(args.absorption) if args.absorption else (reports / "absorption_probabilities.csv")
    steps_path = Path(args.expected_steps) if args.expected_steps else (reports / "expected_steps_to_absorption.csv")

    tm = pd.read_csv(matrix_path, index_col=0)
    vis_dir = root / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    save_heatmap(tm, vis_dir / "transition_heatmap.png")

    # Absorption bars from START if present
    if absorb_path.exists():
        B = pd.read_csv(absorb_path, index_col=0)
        if "START" in B.index:
            save_bar(B.loc["START"], vis_dir / "absorption_from_START.png",
                     title="Absorption Probabilities from START", xlabel="Absorbing State", ylabel="Probability")

    # Expected steps bar
    if steps_path.exists():
        t_df = pd.read_csv(steps_path, index_col=0)
        t = t_df.iloc[:, 0] if t_df.shape[1] >= 1 else pd.Series(dtype=float)
        save_bar(t.sort_values(ascending=False), vis_dir / "expected_steps_by_state.png",
                 title="Expected Steps to Absorption by Transient State", xlabel="State", ylabel="Steps")

    log.info(f"Saved visualizations to {vis_dir}")


if __name__ == "__main__":
    main()
