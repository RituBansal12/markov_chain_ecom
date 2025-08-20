"""
 Shared Markov modeling utilities.

 Functions:
 - build_transition_counts
 - build_transition_matrix_from_counts
 - build_transition_matrix_from_transitions
 - analyze_absorbing_chain

 Notes: This module performs no filesystem I/O; callers are responsible for reading/writing.
 """
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Optional

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
