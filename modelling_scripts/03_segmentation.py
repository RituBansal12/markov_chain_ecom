"""
Segment-specific Markov analysis (new vs repeat users, top product categories).

Inputs:
- data_clean/journeys_markov_ready.csv (default) or --input path

Outputs (per segment):
- reports/segments/<segment>/transition_matrix.csv
- reports/segments/<segment>/absorption_probabilities.csv
- reports/segments/<segment>/expected_steps_to_absorption.csv

CLI: --top-k-categories, --min-category-count, --absorbing

Usage:
  python -m modelling_scripts.03_segmentation \
    --absorbing PURCHASE,DROP \
    --top-k-categories 10 \
    --min-category-count 1000
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List
from cleaning_scripts.common import get_paths, ensure_dirs, configure_logging
from .lib import build_transition_matrix_from_transitions, analyze_absorbing_chain

ABSORBING_STATES_DEFAULT = ["PURCHASE", "DROP"]


def segment_new_vs_repeat(df: pd.DataFrame):
    return {
        "new": df[df["is_repeat"] == False],
        "repeat": df[df["is_repeat"] == True],
    }


def segment_by_category(df: pd.DataFrame, top_k: int = 10, min_count: int = 1000):
    vc = df["dominant_session_categoryid"].value_counts(dropna=True)
    cats = [int(c) for c in vc[vc >= min_count].head(top_k).index.tolist()]
    out = {f"category_{c}": df[df["dominant_session_categoryid"] == c] for c in cats}
    return out


def compute_and_save_for_segment(name: str, seg_df: pd.DataFrame, abs_states: List[str], out_dir: Path, log):
    if seg_df.empty:
        log.warning(f"Segment {name} is empty; skipping")
        return
    # Build matrix
    mat = build_transition_matrix_from_transitions(seg_df, absorbing_states=abs_states)
    seg_dir = out_dir / name
    seg_dir.mkdir(parents=True, exist_ok=True)
    mat.to_csv(seg_dir / "transition_matrix.csv")
    # Absorbing analysis
    res = analyze_absorbing_chain(mat, abs_states)
    res["B"].to_csv(seg_dir / "absorption_probabilities.csv")
    res["t"].to_csv(seg_dir / "expected_steps_to_absorption.csv")


def main():
    parser = argparse.ArgumentParser(description="Segmentation analysis for Markov model")
    parser.add_argument("--input", type=str, default=None, help="Path to transitions CSV (journeys_markov_ready.csv)")
    parser.add_argument("--absorbing", type=str, default="PURCHASE,DROP", help="Comma-separated absorbing states")
    parser.add_argument("--top-k-categories", type=int, default=10)
    parser.add_argument("--min-category-count", type=int, default=1000)
    args = parser.parse_args()

    log = configure_logging("segmentation")
    ensure_dirs()

    root, _, clean, reports = get_paths()
    input_path = Path(args.input) if args.input else (clean / "journeys_markov_ready.csv")
    abs_states = [s.strip() for s in args.absorbing.split(",") if s.strip()]

    log.info(f"Reading transitions from {input_path}")
    df = pd.read_csv(input_path)

    out_dir = reports / "segments"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Segments: new vs repeat
    nv = segment_new_vs_repeat(df)
    for name, seg_df in nv.items():
        compute_and_save_for_segment(f"new_repeat/{name}", seg_df, abs_states, out_dir, log)

    # Segments: top categories
    cat_segs = segment_by_category(df, top_k=args.top_k_categories, min_count=args.min_category_count)
    for name, seg_df in cat_segs.items():
        compute_and_save_for_segment(f"category/{name}", seg_df, abs_states, out_dir, log)

    log.info("Segmentation analysis complete")


if __name__ == "__main__":
    main()
