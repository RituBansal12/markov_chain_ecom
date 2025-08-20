import argparse
import json
import pandas as pd
from cleaning_scripts.common import get_paths, ensure_dirs, configure_logging, write_json


def main():
    parser = argparse.ArgumentParser(description="QC checks and summary report")
    parser.add_argument("--sample", action="store_true", help="Use sample outputs")
    args = parser.parse_args()

    log = configure_logging("qc_finalize")
    ensure_dirs()

    _, _, clean, reports = get_paths()
    trans_csv = clean / ("journeys_markov_ready_sample.csv" if args.sample else "journeys_markov_ready.csv")
    map_csv = clean / "item_to_category.csv"

    trans = pd.read_csv(trans_csv, dtype={"visitorid": "string"})
    mapping = pd.read_csv(map_csv, dtype={"itemid": "string"})

    expected_cols = [
        "visitorid", "session_id", "session_start_ts", "session_end_ts",
        "from_state", "to_state", "from_ts", "to_ts", "delta_seconds",
        "from_categoryid", "to_categoryid", "from_itemid", "to_itemid",
        "is_repeat", "dominant_session_categoryid", "event_count_in_session", "has_purchase_in_session",
    ]

    # Session-level validations
    sess = trans.groupby(["visitorid", "session_id"], as_index=False)
    first_rows = sess.nth(0)
    last_rows = sess.nth(-1)
    has_purchase = sess["has_purchase_in_session"].max()["has_purchase_in_session"].astype(bool)
    last_state = last_rows["to_state"].reset_index(drop=True)
    purchase_sessions_ok = bool(((has_purchase & (last_state == "PURCHASE")) | (~has_purchase & (last_state == "DROP"))).all())
    first_from_start_ok = bool((first_rows["from_state"] == "START").all())

    # No negative durations
    non_negative_deltas_ok = bool((trans["delta_seconds"] >= 0).all())

    # Columns check
    columns_ok = set(expected_cols).issubset(set(trans.columns))

    summary = {
        "transitions_rows": int(len(trans)),
        "unique_visitors": int(trans["visitorid"].nunique()),
        "unique_sessions": int(trans[["visitorid", "session_id"]].drop_duplicates().shape[0]),
        "has_purchase_sessions_pct": float(100 * trans.groupby(["visitorid", "session_id"])['has_purchase_in_session'].max().mean()),
        "mapping_rows": int(len(mapping)),
        "columns_present": list(trans.columns),
        "qc": {
            "columns_ok": bool(columns_ok),
            "non_negative_deltas_ok": non_negative_deltas_ok,
            "first_transition_starts_ok": first_from_start_ok,
            "purchase_session_terminal_ok": purchase_sessions_ok,
        },
    }

    write_json(reports / ("prep_summary_sample.json" if args.sample else "prep_summary.json"), summary)
    log.info(f"Wrote summary report with keys: {list(summary.keys())}")


if __name__ == "__main__":
    main()
