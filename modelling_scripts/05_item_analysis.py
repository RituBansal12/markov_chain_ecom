"""
Item-level absorption analysis using session transitions and an absorbing Markov chain.

Inputs:
- data_clean/journeys_markov_ready.csv (default) or --input

Outputs:
- reports/items/absorption_probabilities.csv
- reports/items/expected_time_to_absorption.csv
- reports/items/expected_steps_to_absorption.csv

CLI:
- --input: Path to transitions CSV
- --top-k: Number of top items to save by purchase/drop probability (default: 10)
- --min-sessions: Minimum distinct sessions an item must appear in to be analyzed (default: 50)

Usage:
- python -m modelling_scripts.05_item_analysis \
    --input data_clean/journeys_markov_ready.csv \
    --top-k 10
    --min-sessions 50

Notes:
- Metrics are computed per item by selecting all sessions where the item appears (either as from_itemid or to_itemid).
- Absorption probabilities are taken from the START state of those sessions for absorbing states PURCHASE and DROP.
- Expected time and expected steps to absorption are computed separately for PURCHASE-only sessions and DROP-only sessions.
"""
import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from cleaning_scripts.common import get_paths, ensure_dirs, configure_logging
from .lib import (
    build_transition_matrix_from_transitions,
    analyze_absorbing_chain,
    compute_avg_holding_times,
    expected_time_to_absorption,
)


def build_item_to_sessions(df: pd.DataFrame) -> Dict[int, List[Tuple[int, int]]]:
    """Map each itemid to the list of (visitorid, session_id) pairs where it appears in the session.

    Considers both from_itemid and to_itemid. Returns a dict[itemid] -> list of session keys.
    """
    items_per_sess = (
        df.assign(items=lambda x: x[["from_itemid", "to_itemid"]].values.tolist())
          .assign(items=lambda x: x["items"].apply(lambda p: [i for i in p if pd.notna(i)]))
          .groupby(["visitorid", "session_id"])['items']
          .sum()
          .apply(lambda arr: sorted(set(int(i) for i in arr)))
          .rename("session_items")
          .reset_index()
    )
    exploded = items_per_sess.explode("session_items").dropna(subset=["session_items"]).rename(columns={"session_items": "itemid"})
    exploded["itemid"] = exploded["itemid"].astype(int)
    grouped = exploded.groupby("itemid").apply(lambda g: list(zip(g["visitorid"].astype(int), g["session_id"].astype(int)))).to_dict()
    return grouped


def compute_absorption_probs_for_item(transitions: pd.DataFrame) -> Tuple[float, float]:
    """Compute absorption probabilities to PURCHASE and DROP from START for the provided transitions subset.

    Returns (p_purchase, p_drop).
    """
    if transitions.empty:
        return (np.nan, np.nan)
    mat = build_transition_matrix_from_transitions(transitions, absorbing_states=["PURCHASE", "DROP"])
    res = analyze_absorbing_chain(mat, ["PURCHASE", "DROP"])
    B = res["B"]
    p_purchase = float(B.loc["START"].get("PURCHASE", np.nan)) if "START" in B.index else np.nan
    p_drop = float(B.loc["START"].get("DROP", np.nan)) if "START" in B.index else np.nan
    return (p_purchase, p_drop)


def compute_time_and_steps_for_item(transitions: pd.DataFrame, target: str) -> Tuple[float, float]:
    """Compute expected time (seconds) and expected steps to a specific absorbing state from START.

    target in {"PURCHASE", "DROP"}. Uses only sessions that actually absorb into the target state.
    Returns (expected_time_seconds, expected_steps).
    """
    assert target in {"PURCHASE", "DROP"}
    if target == "PURCHASE":
        sdf = transitions[transitions["has_purchase_in_session"] == True]
    else:
        sdf = transitions[transitions["has_purchase_in_session"] == False]

    if sdf.empty:
        return (np.nan, np.nan)

    mat = build_transition_matrix_from_transitions(sdf, absorbing_states=[target])
    ht = compute_avg_holding_times(sdf, states=list(mat.index), state_col="from_state", delta_col="delta_seconds")
    t_ser = expected_time_to_absorption(mat, [target], ht)
    s_ser = analyze_absorbing_chain(mat, [target])["t"]
    t_val = float(t_ser.get("START", np.nan))
    s_val = float(s_ser.get("START", np.nan))
    return (t_val, s_val)


def main():
    parser = argparse.ArgumentParser(description="Item-level absorption analysis")
    parser.add_argument("--input", type=str, default=None, help="Path to transitions CSV (journeys_markov_ready.csv)")
    parser.add_argument("--top-k", type=int, default=10, help="For logging: top-K items by purchase/drop probability")
    parser.add_argument("--min-sessions", type=int, default=50, help="Minimum distinct sessions an item must appear in to be analyzed")
    # Back-compat: accepted but unused; keeps run_pipeline working if it passes --absorbing
    parser.add_argument("--absorbing", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    log = configure_logging("item_analysis")
    ensure_dirs()

    root, _, clean, reports = get_paths()
    input_path = Path(args.input) if args.input else (clean / "journeys_markov_ready.csv")

    log.info(f"Reading transitions from {input_path}")
    df = pd.read_csv(input_path)

    # Build mapping from item -> list of sessions
    item_sessions = build_item_to_sessions(df)
    log.info(f"Found {len(item_sessions):,} items with at least one session")

    # Pre-index for efficient filtering by (visitorid, session_id)
    df_keyed = df.set_index(["visitorid", "session_id"]).sort_index()

    # Collect metrics per item
    abs_rows = []  # itemid, purchase_prob, drop_prob, sessions
    time_rows = []  # itemid, expected_time_seconds_PURCHASE, expected_time_seconds_DROP
    step_rows = []  # itemid, expected_steps_PURCHASE, expected_steps_DROP
    skipped_few_sessions = 0

    for itemid, sess_list in item_sessions.items():
        sess_keys = list({(int(v), int(s)) for v, s in sess_list})
        if not sess_keys:
            continue
        session_count = len(set(sess_keys))
        if session_count < args.min_sessions:
            skipped_few_sessions += 1
            continue
        sdf = df_keyed.loc[pd.MultiIndex.from_tuples(sess_keys)].reset_index()

        p_pur, p_drop = compute_absorption_probs_for_item(sdf)
        t_pur, s_pur = compute_time_and_steps_for_item(sdf, "PURCHASE")
        t_drop, s_drop = compute_time_and_steps_for_item(sdf, "DROP")

        abs_rows.append({
            "itemid": int(itemid),
            "sessions": int(session_count),
            "purchase_prob": p_pur,
            "drop_prob": p_drop,
        })
        time_rows.append({
            "itemid": int(itemid),
            "expected_time_seconds_PURCHASE": t_pur,
            "expected_time_seconds_DROP": t_drop,
        })
        step_rows.append({
            "itemid": int(itemid),
            "expected_steps_PURCHASE": s_pur,
            "expected_steps_DROP": s_drop,
        })

    # Build dataframes
    abs_df = pd.DataFrame(abs_rows)
    time_df = pd.DataFrame(time_rows)
    steps_df = pd.DataFrame(step_rows)

    log.info(f"Skipped {skipped_few_sessions:,} items with < {args.min_sessions} sessions")

    # Select only the union of top-K items by purchase and drop probabilities
    if not abs_df.empty:
        top_by_purchase = abs_df.sort_values(["purchase_prob", "sessions"], ascending=[False, False]).head(args.top_k)
        top_by_drop = abs_df.sort_values(["drop_prob", "sessions"], ascending=[False, False]).head(args.top_k)
        top_ids = pd.Index(top_by_purchase["itemid"]).union(top_by_drop["itemid"])
        abs_df = abs_df[abs_df["itemid"].isin(top_ids)].copy()
        time_df = time_df[time_df["itemid"].isin(top_ids)].copy()
        steps_df = steps_df[steps_df["itemid"].isin(top_ids)].copy()
        log.info("Top items by purchase probability:\n" + top_by_purchase.to_string(index=False))
        log.info("Top items by drop probability:\n" + top_by_drop.to_string(index=False))

    # Write outputs
    items_dir = reports / "items"
    items_dir.mkdir(parents=True, exist_ok=True)
    abs_out = items_dir / "absorption_probabilities.csv"
    time_out = items_dir / "expected_time_to_absorption.csv"
    steps_out = items_dir / "expected_steps_to_absorption.csv"

    abs_cols = ["itemid", "sessions", "purchase_prob", "drop_prob"]
    if not abs_df.empty:
        abs_df[abs_cols].sort_values("itemid").to_csv(abs_out, index=False)
    else:
        abs_out.write_text("")

    time_cols = ["itemid", "expected_time_seconds_PURCHASE", "expected_time_seconds_DROP"]
    if not time_df.empty:
        time_df[time_cols].sort_values("itemid").to_csv(time_out, index=False)
    else:
        time_out.write_text("")

    steps_cols = ["itemid", "expected_steps_PURCHASE", "expected_steps_DROP"]
    if not steps_df.empty:
        steps_df[steps_cols].sort_values("itemid").to_csv(steps_out, index=False)
    else:
        steps_out.write_text("")

    log.info("Item-level absorption analysis complete: wrote CSVs to reports/items/")


if __name__ == "__main__":
    main()
