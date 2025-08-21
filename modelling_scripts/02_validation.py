"""
Validate the transition matrix and model soundness (row sums, non-negativity, absorbing rows, N existence).

Inputs:
- data_clean/transition_matrix.csv (default) or --matrix path

Outputs:
- reports/validation_report.json

CLI:
- --matrix, --absorbing

Usage:
  python -m modelling_scripts.04_validation \
    --absorbing PURCHASE,DROP
"""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from typing import List
from cleaning_scripts.common import get_paths, ensure_dirs, configure_logging, write_json

ABSORBING_STATES_DEFAULT = ["PURCHASE", "DROP"]


def validate_transition_matrix(tm: pd.DataFrame, absorbing_states: List[str]):
    tol = 1e-8
    issues = []
    # Square
    if tm.shape[0] != tm.shape[1]:
        issues.append(f"Matrix not square: {tm.shape}")
    # Row sums
    rowsum = tm.sum(axis=1).astype(float)
    if not np.allclose(rowsum.values, np.ones_like(rowsum.values), atol=tol):
        issues.append("Row sums not equal to 1 within tolerance")
    # Non-negative
    if (tm.values < -1e-12).any():
        issues.append("Negative probabilities present")
    # Absorbing rows
    for s in absorbing_states:
        if s in tm.index:
            row = tm.loc[s]
            if not np.isclose(row.get(s, 0.0), 1.0, atol=tol):
                issues.append(f"Absorbing state {s} does not have self-loop 1.0")
            if not np.isclose(row.drop(labels=[s], errors='ignore').sum(), 0.0, atol=tol):
                issues.append(f"Absorbing state {s} has outgoing probability != 0 to others")
    # Absorbing existence
    has_abs = any(s in tm.index for s in absorbing_states)
    if not has_abs:
        issues.append("No absorbing states present in matrix")

    # Try N existence
    try:
        transient = [s for s in tm.index if s not in set(absorbing_states)]
        Q = tm.loc[transient, transient].to_numpy(dtype=float)
        I = np.eye(len(transient))
        _ = np.linalg.inv(I - Q)
    except np.linalg.LinAlgError:
        issues.append("(I - Q) is singular; fundamental matrix undefined")

    return issues


def main():
    parser = argparse.ArgumentParser(description="Validate transition matrix and Markov model setup")
    parser.add_argument("--matrix", type=str, default=None, help="Path to transition matrix CSV")
    parser.add_argument("--absorbing", type=str, default="PURCHASE,DROP")
    args = parser.parse_args()

    log = configure_logging("validation")
    ensure_dirs()

    root, _, clean, reports = get_paths()
    matrix_path = Path(args.matrix) if args.matrix else (clean / "transition_matrix.csv")
    abs_states = [s.strip() for s in args.absorbing.split(",") if s.strip()]

    tm = pd.read_csv(matrix_path, index_col=0)
    issues = validate_transition_matrix(tm, abs_states)

    report = {
        "matrix_path": str(matrix_path),
        "absorbing_states": abs_states,
        "valid": len(issues) == 0,
        "issues": issues,
    }
    write_json(reports / "validation_report.json", report)
    if issues:
        log.warning(f"Validation found issues: {issues}")
    else:
        log.info("Validation passed with no issues")


if __name__ == "__main__":
    main()
