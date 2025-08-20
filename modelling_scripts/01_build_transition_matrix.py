"""
Build the transition probability matrix from session transitions.

Inputs:
- data_clean/journeys_markov_ready.csv (default) or --input path

Outputs:
- data_clean/transition_matrix.csv
- data_clean/transition_counts.csv

CLI:
- --input, --output, --counts-output, --absorbing

Usage:
  python -m modelling_scripts.01_build_transition_matrix \
    --absorbing PURCHASE,DROP

Notes: Absorbing states configurable via --absorbing. Creates output directories if missing.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Optional
from cleaning_scripts.common import get_paths, ensure_dirs, configure_logging

ABSORBING_STATES_DEFAULT = ["PURCHASE", "DROP"]


def build_transition_counts(transitions: pd.DataFrame) -> pd.DataFrame:
    counts = (
        transitions.groupby(["from_state", "to_state"]).size().rename("count").reset_index()
    )
    return counts


def build_transition_matrix_from_counts(counts: pd.DataFrame, all_states: Optional[List[str]] = None,
                                        absorbing_states: Optional[List[str]] = None) -> pd.DataFrame:
    absorbing_states = absorbing_states or ABSORBING_STATES_DEFAULT
    # Pivot to matrix
    mat = counts.pivot(index="from_state", columns="to_state", values="count").fillna(0.0)
    # Ensure all states are present as rows and columns
    if all_states is None:
        all_states = sorted(set(mat.index).union(mat.columns))
    mat = mat.reindex(index=all_states, columns=all_states, fill_value=0.0)

    # Normalize rows
    row_sums = mat.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        prob = mat.div(row_sums.replace({0: np.nan}), axis=0)
    prob = prob.fillna(0.0)

    # Enforce absorbing rows
    for s in absorbing_states:
        if s in prob.index:
            prob.loc[s, :] = 0.0
            prob.loc[s, s] = 1.0

    # Validate stochastic
    tol = 1e-8
    rowsum = prob.sum(axis=1).astype(float)
    assert np.allclose(rowsum.values, np.ones_like(rowsum.values), atol=tol), "Row sums not 1 within tolerance"
    assert (prob.values >= -1e-12).all(), "Negative probabilities found"
    return prob


def build_transition_matrix_from_transitions(transitions: pd.DataFrame,
                                             absorbing_states: Optional[List[str]] = None,
                                             include_states: Optional[List[str]] = None) -> pd.DataFrame:
    absorbing_states = absorbing_states or ABSORBING_STATES_DEFAULT
    if include_states is None:
        include_states = sorted(set(transitions["from_state"]).union(transitions["to_state"]))
    counts = build_transition_counts(transitions)
    return build_transition_matrix_from_counts(counts, all_states=include_states, absorbing_states=absorbing_states)


def main():
    parser = argparse.ArgumentParser(description="Build transition matrix from journeys_markov_ready.csv")
    parser.add_argument("--input", type=str, default=None, help="Path to transitions CSV (journeys_markov_ready.csv)")
    parser.add_argument("--output", type=str, default=None, help="Path to save transition matrix CSV")
    parser.add_argument("--counts-output", type=str, default=None, help="Optional path to save transition counts CSV")
    parser.add_argument("--absorbing", type=str, default="PURCHASE,DROP", help="Comma-separated absorbing states")
    args = parser.parse_args()

    log = configure_logging("build_transition_matrix")
    ensure_dirs()

    root, _, clean, reports = get_paths()
    input_path = Path(args.input) if args.input else (clean / "journeys_markov_ready.csv")
    output_path = Path(args.output) if args.output else (clean / "transition_matrix.csv")
    counts_out = Path(args.counts_output) if args.counts_output else (clean / "transition_counts.csv")

    abs_states = [s.strip() for s in args.absorbing.split(",") if s.strip()]

    log.info(f"Reading transitions from {input_path}")
    df = pd.read_csv(input_path)

    # Build matrix
    all_states = sorted(set(df["from_state"]).union(df["to_state"]))
    mat = build_transition_matrix_from_transitions(df, absorbing_states=abs_states, include_states=all_states)

    # Persist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts_out.parent.mkdir(parents=True, exist_ok=True)
    mat.to_csv(output_path)
    build_transition_counts(df).to_csv(counts_out, index=False)
    log.info(f"Wrote transition matrix to {output_path} with shape {mat.shape}")
    log.info(f"Wrote transition counts to {counts_out}")


if __name__ == "__main__":
    main()
