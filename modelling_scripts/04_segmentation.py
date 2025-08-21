"""
Segment-specific Markov analysis per plan.

Inputs:
- data_clean/journeys_markov_ready.csv (default) or --input path

Outputs:
- reports/segments/new_repeat/absorption_probabilities.csv
- reports/segments/new_repeat/expected_time_to_absorption.csv
- reports/segments/new_repeat/expected_steps_to_absorption.csv
- reports/segments/category/absorption_probabilities.csv
- reports/segments/category/expected_time_to_absorption.csv
- reports/segments/category/expected_steps_to_absorption.csv
- reports/segments/purchase_rate/absorption_probabilities.csv
- reports/segments/purchase_rate/expected_time_to_absorption.csv
- reports/segments/purchase_rate/expected_steps_to_absorption.csv
- reports/segments/single_vs_multiple/absorption_probabilities.csv
- reports/segments/single_vs_multiple/expected_time_to_absorption.csv
- reports/segments/single_vs_multiple/expected_steps_to_absorption.csv

CLI: --input

Usage:
  python -m modelling_scripts.04_segmentation

Notes:
- Run Modelling scripts 01-03 before running this script.
"""
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd

from cleaning_scripts.common import get_paths, ensure_dirs, configure_logging
from .lib import (
    build_transition_matrix_from_transitions,
    analyze_absorbing_chain,
    compute_avg_holding_times,
    expected_time_to_absorption,
)

ABSORBING_STATES_DEFAULT = ["PURCHASE", "DROP"]


def build_user_metrics(transitions: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-user metrics required for segmentation.
    Columns: visitorid,total_sessions,purchase_sessions,distinct_purchase_categories,purchase_rate,is_repeat_user,category_span,purchase_rate_bucket
    """
    # Session-level table
    sess = (
        transitions.groupby(["visitorid", "session_id"]).agg(
            has_purchase=("has_purchase_in_session", "max"),
            dominant_category=("dominant_session_categoryid", lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else pd.NA),
        )
        .reset_index()
    )

    # User-level aggregation
    per_user = (
        sess.groupby("visitorid").agg(
            total_sessions=("session_id", "nunique"),
            purchase_sessions=("has_purchase", "sum"),
        )
        .reset_index()
    )

    # Distinct categories among purchase sessions
    purchase_sessions = sess[sess["has_purchase"] == True]
    distinct_cats = (
        purchase_sessions.groupby("visitorid")["dominant_category"].nunique().rename("distinct_purchase_categories").reset_index()
    )
    per_user = per_user.merge(distinct_cats, on="visitorid", how="left")
    per_user["distinct_purchase_categories"] = per_user["distinct_purchase_categories"].fillna(0).astype(int)

    # Derived fields
    per_user["purchase_rate"] = (per_user["purchase_sessions"] / per_user["total_sessions"]).astype(float)
    per_user["is_repeat_user"] = per_user["purchase_sessions"] > 0
    per_user["category_span"] = np.where(per_user["distinct_purchase_categories"] <= 1, "single", "multiple")
    per_user["purchase_rate_bucket"] = pd.cut(
        per_user["purchase_rate"], bins=[-0.001, 0.2, 0.5, 1.0], labels=["low", "medium", "high"], include_lowest=True
    )
    return per_user[[
        "visitorid",
        "total_sessions",
        "purchase_sessions",
        "distinct_purchase_categories",
        "purchase_rate",
        "is_repeat_user",
        "category_span",
        "purchase_rate_bucket",
    ]]


def segment_new_vs_repeat(transitions: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {
        "new": transitions[transitions["is_repeat"] == False],
        "repeat": transitions[transitions["is_repeat"] == True],
    }


def top_k_categories_by_sessions(transitions: pd.DataFrame, k: int = 5) -> List[int]:
    # Count sessions per category using dominant_session_categoryid
    sess_cats = (
        transitions[["visitorid", "session_id", "dominant_session_categoryid"]]
        .drop_duplicates(["visitorid", "session_id"])
    )
    vc = sess_cats["dominant_session_categoryid"].value_counts(dropna=True)
    cats = [int(c) for c in vc.head(k).index.tolist() if pd.notna(c)]
    return cats


def segment_by_top_categories(transitions: pd.DataFrame, k: int = 5) -> Dict[str, pd.DataFrame]:
    cats = top_k_categories_by_sessions(transitions, k=k)
    return {f"category_{c}": transitions[transitions["dominant_session_categoryid"] == c] for c in cats}


def segment_by_purchase_rate(transitions: pd.DataFrame, user_metrics: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for bucket in ["low", "medium", "high"]:
        users = user_metrics[user_metrics["purchase_rate_bucket"].astype(str) == bucket]["visitorid"]
        if not users.empty:
            out[bucket] = transitions[transitions["visitorid"].isin(users)]
    return out


def segment_single_vs_multiple(transitions: pd.DataFrame, user_metrics: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for span in ["single", "multiple"]:
        users = user_metrics[user_metrics["category_span"] == span]["visitorid"]
        if not users.empty:
            out[span] = transitions[transitions["visitorid"].isin(users)]
    return out


def compute_absorption_probabilities(seg_df: pd.DataFrame) -> pd.DataFrame:
    if seg_df.empty:
        return pd.DataFrame(columns=["state", "PURCHASE", "DROP"]).set_index("state")
    mat = build_transition_matrix_from_transitions(seg_df, absorbing_states=ABSORBING_STATES_DEFAULT)
    res = analyze_absorbing_chain(mat, ABSORBING_STATES_DEFAULT)
    return res["B"]  # index: transient states, columns: PURCHASE,DROP


def compute_time_and_steps(seg_df: pd.DataFrame, target_absorbing: str):
    """Compute expected time and expected steps to a specific absorbing state using only sessions relevant to that state.
    target_absorbing in {"PURCHASE", "DROP"}
    Returns (time_series, steps_series) indexed by transient states.
    """
    assert target_absorbing in {"PURCHASE", "DROP"}
    if target_absorbing == "PURCHASE":
        sdf = seg_df[seg_df["has_purchase_in_session"] == True]
    else:
        sdf = seg_df[seg_df["has_purchase_in_session"] == False]

    if sdf.empty:
        empty_time = pd.Series(dtype=float, name="expected_time_seconds")
        empty_steps = pd.Series(dtype=float, name="expected_steps")
        return empty_time, empty_steps

    mat = build_transition_matrix_from_transitions(sdf, absorbing_states=[target_absorbing])
    ht = compute_avg_holding_times(sdf, states=list(mat.index), state_col="from_state", delta_col="delta_seconds")
    time_ser = expected_time_to_absorption(mat, [target_absorbing], ht)
    steps_ser = analyze_absorbing_chain(mat, [target_absorbing])["t"]
    return time_ser, steps_ser


def aggregate_and_write(seg_name: str, seg_map: Dict[str, pd.DataFrame], out_dir: Path, log) -> None:
    """Aggregate metrics for a segmentation and write 3 CSVs under out_dir/seg_name.
    Files: absorption_probabilities.csv, expected_time_to_absorption.csv, expected_steps_to_absorption.csv
    """
    seg_out = out_dir / seg_name
    seg_out.mkdir(parents=True, exist_ok=True)

    # Collect absorption probabilities for each segment
    abs_rows = []
    time_rows = []
    step_rows = []

    for name, sdf in seg_map.items():
        if sdf.empty:
            log.warning(f"Segment {seg_name}/{name} is empty; skipping")
            continue
        # Absorption probabilities from full segment matrix (PURCHASE,DROP)
        B = compute_absorption_probabilities(sdf)
        if not B.empty:
            tmp = B.reset_index().rename(columns={"index": "state"})
            tmp.insert(0, "segment", name)
            abs_rows.append(tmp)

        # Expected time and steps to PURCHASE and DROP separately
        t_pur, s_pur = compute_time_and_steps(sdf, "PURCHASE")
        t_drp, s_drp = compute_time_and_steps(sdf, "DROP")

        # Align by state name
        all_states = sorted(set(t_pur.index).union(t_drp.index))
        t_df = pd.DataFrame({
            "state": all_states,
            "expected_time_seconds_PURCHASE": [float(t_pur.get(st, np.nan)) for st in all_states],
            "expected_time_seconds_DROP": [float(t_drp.get(st, np.nan)) for st in all_states],
        })
        t_df.insert(0, "segment", name)
        time_rows.append(t_df)

        s_all_states = sorted(set(s_pur.index).union(s_drp.index))
        s_df = pd.DataFrame({
            "state": s_all_states,
            "expected_steps_PURCHASE": [float(s_pur.get(st, np.nan)) for st in s_all_states],
            "expected_steps_DROP": [float(s_drp.get(st, np.nan)) for st in s_all_states],
        })
        s_df.insert(0, "segment", name)
        step_rows.append(s_df)

    # Write outputs
    if abs_rows:
        abs_df = pd.concat(abs_rows, ignore_index=True)
        abs_df.to_csv(seg_out / "absorption_probabilities.csv", index=False)
    else:
        (seg_out / "absorption_probabilities.csv").write_text("")

    if time_rows:
        time_df = pd.concat(time_rows, ignore_index=True)
        time_df.to_csv(seg_out / "expected_time_to_absorption.csv", index=False)
    else:
        (seg_out / "expected_time_to_absorption.csv").write_text("")

    if step_rows:
        steps_df = pd.concat(step_rows, ignore_index=True)
        steps_df.to_csv(seg_out / "expected_steps_to_absorption.csv", index=False)
    else:
        (seg_out / "expected_steps_to_absorption.csv").write_text("")


def main():
    parser = argparse.ArgumentParser(description="Segmentation analysis per plan")
    parser.add_argument("--input", type=str, default=None, help="Path to transitions CSV (journeys_markov_ready.csv)")
    # Back-compat args (parsed but unused)
    parser.add_argument("--absorbing", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--top-k-categories", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--min-category-count", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    log = configure_logging("segmentation")
    ensure_dirs()

    root, _, clean, reports = get_paths()
    input_path = Path(args.input) if args.input else (clean / "journeys_markov_ready.csv")

    log.info(f"Reading transitions from {input_path}")
    df = pd.read_csv(input_path)

    out_dir = reports / "segments"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 5. user_metrics.csv
    user_metrics = build_user_metrics(df)

    # 1. New vs Repeat Users
    nr_segments = segment_new_vs_repeat(df)
    aggregate_and_write("new_repeat", nr_segments, out_dir, log)

    # 2. Category Segmentation (Top 5 by sessions)
    cat_segments = segment_by_top_categories(df, k=5)
    aggregate_and_write("category", cat_segments, out_dir, log)

    # 3. Purchase Rate Segmentation
    pr_segments = segment_by_purchase_rate(df, user_metrics)
    aggregate_and_write("purchase_rate", pr_segments, out_dir, log)

    # 4. Single vs Multiple Category Buyers
    sm_segments = segment_single_vs_multiple(df, user_metrics)
    aggregate_and_write("single_vs_multiple", sm_segments, out_dir, log)

    log.info("Segmentation analysis complete")


if __name__ == "__main__":
    main()
