"""
Orchestrate the full pipeline for customer-journey Markov modeling.

Cleaning steps:
1) cleaning_scripts.01_build_item_category
   - Inputs: data/item_properties_part1.csv, data/item_properties_part2.csv
   - Output: data_clean/item_to_category.csv
2) cleaning_scripts.02_enrich_events
   - Inputs: data/events.csv, data_clean/item_to_category.csv
   - Output: data_clean/events_enriched.parquet
3) cleaning_scripts.03_build_transitions
   - Input: data_clean/events_enriched.parquet
   - Output: data_clean/journeys_markov_ready.csv
4) cleaning_scripts.04_qc_and_finalize
   - Inputs: data_clean/journeys_markov_ready.csv, data_clean/item_to_category.csv
   - Output: reports/prep_summary.json

Modelling steps:
5) modelling_scripts.01_build_transition_matrix
   - Input: data_clean/journeys_markov_ready.csv
   - Outputs: data_clean/transition_matrix.csv, data_clean/transition_counts.csv
6) modelling_scripts.02_validation
   - Input: data_clean/transition_matrix.csv
   - Output: reports/validation_report.json
7) modelling_scripts.03_markov_model
   - Input: data_clean/transition_matrix.csv
   - Outputs: reports/absorption_probabilities.csv, reports/expected_steps_to_absorption.csv, reports/markov_model_meta.json
8) modelling_scripts.04_segmentation
   - Input: data_clean/journeys_markov_ready.csv
   - Outputs: reports/segments/*/*
9) modelling_scripts.05_item_analysis
   - Input: data_clean/journeys_markov_ready.csv
   - Outputs: reports/items/*.csv

Usage:
  python -m run_pipeline \
    --chunk-rows 1000000 \
    --session-gap-minutes 30 \
    --absorbing PURCHASE,DROP \
    --segments-top-k 10 \
    --segments-min-count 1000 \
    --items-top-k 10 \
    --items-min-sessions 50


CLI:
- --chunk-rows: Rows per chunk for streaming CSVs
- --session-gap-minutes: Session inactivity gap in minutes to split sessions
- --absorbing: Comma-separated absorbing states
- --segments-top-k: Top-K categories for segmentation
- --segments-min-count: Minimum category count for segmentation
- --items-top-k: Top-K items for item analysis
- --items-min-sessions: Minimum distinct sessions an item must appear in to be analyzed

Notes:
- Each script is responsible for creating its output directories if missing.
- This orchestrator passes common parameters along and logs completion of each step.
"""
import argparse
import subprocess
import sys
from typing import List, Optional
from cleaning_scripts.common import configure_logging


def run_module(module: str, args: Optional[List[str]] = None):
    args = args or []
    cmd = [sys.executable, "-m", module] + args
    print("$", " ".join(cmd))
    res = subprocess.run(cmd, check=True)
    return res.returncode


def main():
    parser = argparse.ArgumentParser(description="Run full Markov pipeline (cleaning + modelling)")
    parser.add_argument("--chunk-rows", type=int, default=None, help="Rows per chunk for streaming CSVs")
    parser.add_argument("--session-gap-minutes", type=int, default=None, help="Session inactivity gap in minutes")
    parser.add_argument("--absorbing", type=str, default="PURCHASE,DROP", help="Comma-separated absorbing states")
    parser.add_argument("--segments-top-k", type=int, default=None, help="Top-K categories for segmentation")
    parser.add_argument("--segments-min-count", type=int, default=None, help="Minimum category count for segmentation")
    parser.add_argument("--items-top-k", type=int, default=None, help="Top-K items for item analysis")
    parser.add_argument("--items-min-sessions", type=int, default=None, help="Minimum distinct sessions an item must appear in to be analyzed")
    args = parser.parse_args()

    log = configure_logging("pipeline")

    # 1) Item-category mapping
    args1 = []
    if args.chunk_rows is not None:
        args1 += ["--chunk-rows", str(args.chunk_rows)]
    run_module("cleaning_scripts.01_build_item_category", args1)
    log.info("Step 1 complete: item_to_category.csv ready")

    # 2) Enrich events
    args2 = []
    if args.chunk_rows is not None:
        args2 += ["--chunk-rows", str(args.chunk_rows)]
    run_module("cleaning_scripts.02_enrich_events", args2)
    log.info("Step 2 complete: events_enriched.parquet ready")

    # 3) Build transitions
    args3 = []
    if args.session_gap_minutes is not None:
        args3 += ["--session-gap-minutes", str(args.session_gap_minutes)]
    run_module("cleaning_scripts.03_build_transitions", args3)
    log.info("Step 3 complete: journeys_markov_ready.csv ready")

    # 4) QC + report
    run_module("cleaning_scripts.04_qc_and_finalize")
    log.info("Step 4 complete: prep_summary.json written")

    # 5) Build transition matrix
    args5 = ["--absorbing", args.absorbing]
    run_module("modelling_scripts.01_build_transition_matrix", args5)
    log.info("Step 5 complete: transition_matrix.csv and transition_counts.csv written")

    # 6) Validate
    args6 = ["--absorbing", args.absorbing]
    run_module("modelling_scripts.02_validation", args6)
    log.info("Step 6 complete: validation_report.json written")

    # 7) Markov model analysis
    args7 = ["--absorbing", args.absorbing]
    run_module("modelling_scripts.03_markov_model", args7)
    log.info("Step 7 complete: absorption probabilities and expected steps written")

    # 8) Segmentation analysis
    args9 = ["--absorbing", args.absorbing]
    if args.segments_top_k is not None:
        args9 += ["--top-k-categories", str(args.segments_top_k)]
    if args.segments_min_count is not None:
        args9 += ["--min-category-count", str(args.segments_min_count)]
    run_module("modelling_scripts.04_segmentation", args9)
    log.info("Step 8 complete: segmentation outputs under reports/segments/")

    # 9) Item-level analysis
    args10 = ["--absorbing", args.absorbing]
    if args.items_top_k is not None:
        args10 += ["--top-k", str(args.items_top_k)]
    if args.items_min_sessions is not None:
        args10 += ["--min-sessions", str(args.items_min_sessions)]
    run_module("modelling_scripts.05_item_analysis", args10)
    log.info("Step 9 complete: item analysis written to reports/items/")


if __name__ == "__main__":
    main()
