import argparse
import subprocess
import sys
from cleaning_scripts.common import configure_logging


def run_module(module: str, args: list[str] | None = None):
    args = args or []
    cmd = [sys.executable, "-m", module] + args
    print("$", " ".join(cmd))
    res = subprocess.run(cmd, check=True)
    return res.returncode


def main():
    parser = argparse.ArgumentParser(description="Run data prep pipeline")
    parser.add_argument("--sample", action="store_true", help="Run a quick sample pipeline")
    parser.add_argument("--chunk-rows", type=int, default=None, help="Override chunk size")
    args = parser.parse_args()

    log = configure_logging("pipeline")

    # 0) Ensure dirs
    run_module("cleaning_scripts.00_prep_dirs")

    # 1) Item-category mapping
    args1 = []
    if args.chunk_rows:
        args1 += ["--chunk-rows", str(args.chunk_rows)]
    if args.sample:
        args1 += ["--sample", "--sample-items", "50000"]
    run_module("cleaning_scripts.01_build_item_category", args1)

    # 2) Enrich events
    args2 = []
    if args.chunk_rows:
        args2 += ["--chunk-rows", str(args.chunk_rows)]
    if args.sample:
        args2 += ["--sample", "--sample-rows", "200000"]
    run_module("cleaning_scripts.02_enrich_events", args2)

    # 3) Build transitions
    args3 = []
    if args.sample:
        args3 += ["--sample"]
    run_module("cleaning_scripts.03_build_transitions", args3)

    # 4) QC + report
    args4 = []
    if args.sample:
        args4 += ["--sample"]
    run_module("cleaning_scripts.04_qc_and_finalize", args4)


if __name__ == "__main__":
    main()
