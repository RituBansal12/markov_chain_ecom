"""
Analyze an absorbing Markov chain: fundamental matrix, absorption probabilities, expected steps.

Inputs:
- data_clean/transition_matrix.csv (default) or --matrix path

Outputs:
- reports/absorption_probabilities.csv
- reports/expected_steps_to_absorption.csv
- reports/markov_model_meta.json

CLI:
- --matrix, --absorbing

Usage:
  python -m modelling_scripts.02_markov_model \
    --absorbing PURCHASE,DROP
"""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from typing import List
from cleaning_scripts.common import get_paths, ensure_dirs, configure_logging

ABSORBING_STATES_DEFAULT = ["PURCHASE", "DROP"]


def analyze_absorbing_chain(transition_matrix: pd.DataFrame, absorbing_states: List[str]):
    states = list(transition_matrix.index)
    absorbing = [s for s in states if s in set(absorbing_states)]
    transient = [s for s in states if s not in set(absorbing_states)]
    if len(absorbing) == 0:
        raise ValueError("No absorbing states found in matrix")

    # Reorder matrix as transient + absorbing
    tm = transition_matrix.loc[transient + absorbing, transient + absorbing]

    t = len(transient)
    a = len(absorbing)

    Q = tm.iloc[:t, :t].to_numpy(dtype=float)
    R = tm.iloc[:t, t:].to_numpy(dtype=float)

    I = np.eye(t)
    # (I - Q) inverse
    try:
        N = np.linalg.inv(I - Q)
    except np.linalg.LinAlgError as e:
        raise ValueError("(I - Q) is singular; model invalid for absorbing analysis") from e

    B = N @ R
    ones = np.ones((t, 1))
    expected_steps = (N @ ones).reshape(-1)

    # Return as DataFrames/Series with labels
    N_df = pd.DataFrame(N, index=transient, columns=transient)
    B_df = pd.DataFrame(B, index=transient, columns=absorbing)
    t_ser = pd.Series(expected_steps, index=transient, name="expected_steps")
    return {
        "states_order": transient + absorbing,
        "transient": transient,
        "absorbing": absorbing,
        "Q": Q,
        "R": R,
        "N": N_df,
        "B": B_df,
        "t": t_ser,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze absorbing Markov chain from transition matrix")
    parser.add_argument("--matrix", type=str, default=None, help="Path to transition matrix CSV")
    parser.add_argument("--absorbing", type=str, default="PURCHASE,DROP", help="Comma-separated absorbing states")
    args = parser.parse_args()

    log = configure_logging("markov_model")
    ensure_dirs()

    root, _, clean, reports = get_paths()
    matrix_path = Path(args.matrix) if args.matrix else (clean / "transition_matrix.csv")
    abs_states = [s.strip() for s in args.absorbing.split(",") if s.strip()]

    log.info(f"Reading transition matrix from {matrix_path}")
    tm = pd.read_csv(matrix_path, index_col=0)

    results = analyze_absorbing_chain(tm, abs_states)

    # Persist outputs
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "absorption_probabilities.csv").write_text("")
    results["B"].to_csv(reports / "absorption_probabilities.csv")
    results["t"].to_csv(reports / "expected_steps_to_absorption.csv")

    meta = {
        "states_order": results["states_order"],
        "transient": results["transient"],
        "absorbing": results["absorbing"],
    }
    with open(reports / "markov_model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    log.info("Wrote absorption probabilities and expected steps to reports/")


if __name__ == "__main__":
    main()
