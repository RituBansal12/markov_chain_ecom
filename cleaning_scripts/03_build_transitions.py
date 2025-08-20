"""
Build sessionized customer-journey transitions with START/DROP and segment labels.

Inputs:
- data_clean/events_enriched.parquet (from 02_enrich_events)

Outputs:
- data_clean/journeys_markov_ready.csv

CLI:
- --session-gap-minutes: inactivity gap in minutes to split sessions (default: 30)

Usage:
  python -m cleaning_scripts.03_build_transitions \
    --session-gap-minutes 30

Notes:
- Sessionizes by inactivity gap, collapses consecutive duplicate states.
- Appends START at session begin and DROP at end if no PURCHASE occurs.
- Adds segment features: is_repeat, dominant_session_categoryid, event_count, has_purchase_in_session.
- Ensures output directories exist if missing.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from cleaning_scripts.common import get_paths, ensure_dirs, configure_logging

STATE_START = "START"
STATE_VIEW = "VIEW"
STATE_ADD = "ADD_TO_CART"
STATE_PURCHASE = "PURCHASE"
STATE_DROP = "DROP"

PRIORITY = {STATE_VIEW: 0, STATE_ADD: 1, STATE_PURCHASE: 2}


def assign_state(row):
    return row["event_norm"]


def build_sessions(df: pd.DataFrame, log, gap_minutes: int = 30) -> pd.DataFrame:
    df = df.sort_values(["visitorid", "ts", "event_norm"], kind="mergesort")
    # ties broken by priority ordering
    df["event_priority"] = df["event_norm"].map(PRIORITY).astype("int8")
    df = df.sort_values(["visitorid", "ts", "event_priority"], kind="mergesort")

    # Sessionize per visitor by inactivity gap
    gap = pd.Timedelta(minutes=gap_minutes)
    df["prev_ts"] = df.groupby("visitorid")["ts"].shift(1)
    df["gap"] = (df["ts"] - df["prev_ts"]).fillna(pd.Timedelta(seconds=0))
    new_session = (df["gap"] > gap) | df["prev_ts"].isna()
    df["session_id"] = new_session.groupby(df["visitorid"]).cumsum().astype("int64")

    # Keep sessions with at least two events
    counts = df.groupby(["visitorid", "session_id"]).size().rename("event_count").reset_index()
    df = df.merge(counts, on=["visitorid", "session_id"], how="left")
    df = df[df["event_count"] >= 2]

    # Session start/end timestamps and has_purchase
    has_purchase = (
        df.assign(is_purchase=(df["event_norm"] == STATE_PURCHASE))
        .groupby(["visitorid", "session_id"])['is_purchase']
        .any()
        .rename("has_purchase")
        .reset_index()
    )
    sess_bounds = df.groupby(["visitorid", "session_id"]).agg(session_start_ts=("ts", "min"), session_end_ts=("ts", "max")).reset_index()
    df = df.merge(sess_bounds, on=["visitorid", "session_id"], how="left")
    df = df.merge(has_purchase, on=["visitorid", "session_id"], how="left")

    return df


def collapse_repeats(states: pd.DataFrame) -> pd.DataFrame:
    # Collapse repeated consecutive states within session
    states = states.copy()
    states["state"] = states["event_norm"]
    # Identify where state changes within each session
    states["prev_state"] = states.groupby(["visitorid", "session_id"])['state'].shift(1)
    mask = (states["state"] != states["prev_state"]) | (states["prev_state"].isna())
    states = states[mask]
    return states


def build_transitions(states: pd.DataFrame) -> pd.DataFrame:
    # Prepend START and append DROP if no purchase
    def session_transitions(g):
        rows = []
        g = g.sort_values("ts")
        # Stop at first PURCHASE (absorbing)
        if (g["event_norm"] == STATE_PURCHASE).any():
            mask = (g["event_norm"] == STATE_PURCHASE).to_numpy()
            first_p_pos = int(np.argmax(mask))  # safe because any() ensured at least one True
            # include the purchase row but drop anything after
            g = g.iloc[: first_p_pos + 1]
        # START -> first
        first = g.iloc[0]
        rows.append({
            "visitorid": first["visitorid"],
            "session_id": first["session_id"],
            "session_start_ts": first["session_start_ts"],
            "session_end_ts": first["session_end_ts"],
            "from_state": STATE_START,
            "to_state": first["state"],
            "from_ts": first["ts"],
            "to_ts": first["ts"],
            "from_categoryid": pd.NA,
            "to_categoryid": first["categoryid"],
            "from_itemid": pd.NA,
            "to_itemid": first["itemid"],
        })
        # Middle transitions
        for prev, curr in zip(g.iloc[:-1].itertuples(index=False), g.iloc[1:].itertuples(index=False)):
            rows.append({
                "visitorid": curr.visitorid,
                "session_id": curr.session_id,
                "session_start_ts": curr.session_start_ts,
                "session_end_ts": curr.session_end_ts,
                "from_state": prev.state,
                "to_state": curr.state,
                "from_ts": prev.ts,
                "to_ts": curr.ts,
                "from_categoryid": prev.categoryid,
                "to_categoryid": curr.categoryid,
                "from_itemid": prev.itemid,
                "to_itemid": curr.itemid,
            })
        # Append DROP if no purchase
        if not g["event_norm"].eq(STATE_PURCHASE).any():
            last = g.iloc[-1]
            rows.append({
                "visitorid": last["visitorid"],
                "session_id": last["session_id"],
                "session_start_ts": last["session_start_ts"],
                "session_end_ts": last["session_end_ts"],
                "from_state": last["state"],
                "to_state": STATE_DROP,
                "from_ts": last["ts"],
                "to_ts": last["ts"],
                "from_categoryid": last["categoryid"],
                "to_categoryid": pd.NA,
                "from_itemid": last["itemid"],
                "to_itemid": pd.NA,
            })
        return pd.DataFrame.from_records(rows)

    transitions = (
        states.groupby(["visitorid", "session_id"], group_keys=False).apply(session_transitions)
    )
    # Compute delta_seconds
    transitions["delta_seconds"] = (pd.to_datetime(transitions["to_ts"]) - pd.to_datetime(transitions["from_ts"])).dt.total_seconds().astype("float32")
    return transitions


def add_segments(transitions: pd.DataFrame) -> pd.DataFrame:
    # New vs Repeat by first purchase across all sessions per visitor
    purchase_ts = (
        transitions[transitions["to_state"] == STATE_PURCHASE]
        .groupby("visitorid")["to_ts"].min()
        .rename("first_purchase_ts")
        .reset_index()
    )
    df = transitions.merge(purchase_ts, on="visitorid", how="left")
    df["is_repeat"] = (pd.notna(df["first_purchase_ts"]) & (pd.to_datetime(df["session_start_ts"]) > pd.to_datetime(df["first_purchase_ts"])) )
    # Dominant session category: mode of to_categoryid within session
    mode_cat = (
        df.groupby(["visitorid", "session_id"])['to_categoryid']
        .agg(lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else pd.NA)
        .rename("dominant_session_categoryid")
        .reset_index()
    )
    df = df.merge(mode_cat, on=["visitorid", "session_id"], how="left")
    return df


def main():
    parser = argparse.ArgumentParser(description="Build session transitions and segments from enriched events")
    parser.add_argument("--session-gap-minutes", type=int, default=30, help="Inactivity gap to start a new session")
    args = parser.parse_args()

    log = configure_logging("build_transitions")
    ensure_dirs()

    _, _, clean, _ = get_paths()
    enriched = clean / "events_enriched.parquet"
    out_csv = clean / "journeys_markov_ready.csv"

    log.info(f"Reading enriched events from {enriched}")
    ev = pd.read_parquet(enriched)

    # Build sessions
    ev = build_sessions(ev, log, gap_minutes=args.session_gap_minutes)

    # State assignment and collapse repeats
    ev["state"] = ev["event_norm"]
    states = collapse_repeats(ev)

    # Build transitions with START/DROP
    trans = build_transitions(states)

    # QC helpers
    trans["has_purchase_in_session"] = trans.groupby(["visitorid", "session_id"])['to_state'].transform(lambda s: (s == STATE_PURCHASE).any())
    trans["event_count_in_session"] = trans.groupby(["visitorid", "session_id"])['to_state'].transform('size').astype("int32")

    # Segments
    trans = add_segments(trans)

    # Final ordering of columns per plan
    cols = [
        "visitorid", "session_id", "session_start_ts", "session_end_ts",
        "from_state", "to_state", "from_ts", "to_ts", "delta_seconds",
        "from_categoryid", "to_categoryid", "from_itemid", "to_itemid",
        "is_repeat", "dominant_session_categoryid", "event_count_in_session", "has_purchase_in_session",
    ]
    trans = trans[cols]

    # Assertions (QA checks)
    assert not ((trans["from_state"].isin([STATE_DROP, STATE_PURCHASE]) & (trans["to_state"] != trans["from_state"])) ).any(), "Impossible transitions from absorbing states"
    assert (trans["delta_seconds"] >= 0).all(), "Negative delta_seconds found"

    trans.to_csv(out_csv, index=False)
    log.info(f"Wrote transitions to {out_csv} with {len(trans):,} rows")


if __name__ == "__main__":
    main()
