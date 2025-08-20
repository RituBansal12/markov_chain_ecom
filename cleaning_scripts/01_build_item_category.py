"""
Build a stable item->category mapping by streaming item_properties files.

Inputs:
- data/item_properties_part1.csv
- data/item_properties_part2.csv

Outputs:
- data_clean/item_to_category.csv

CLI:
- --chunk-rows: rows per chunk when streaming item_properties

Usage:
  python -m cleaning_scripts.01_build_item_category \
    --chunk-rows 1000000

Notes: Creates output directories if missing via ensure_dirs(); also ensures parent
directory for the output before saving.
"""
import argparse
import pandas as pd
from pathlib import Path
from typing import Optional
from cleaning_scripts.common import get_paths, ensure_dirs, DEFAULT_CHUNK_ROWS, configure_logging

CATEGORY_PROPERTY_KEY = "categoryid"


def latest_category_per_item(properties_paths, chunk_rows: int, log, max_items: Optional[int] = None):
    """Stream item_properties files, keep latest category per itemid by timestamp.
    If max_items is set, stop early once that many unique items have been collected.
    """
    latest = {}
    stop = False
    for p in properties_paths:
        if stop:
            break
        log.info(f"Processing properties file: {p}")
        for chunk in pd.read_csv(
            p,
            usecols=["timestamp", "itemid", "property", "value"],
            dtype={
                "timestamp": "int64",
                "itemid": "string",
                "property": "string",
                "value": "string",
            },
            chunksize=chunk_rows,
        ):
            # Drop nulls in any column
            chunk = chunk.dropna(subset=["timestamp", "itemid", "property", "value"])  # type: ignore
            # Filter to category rows only
            cat = chunk[chunk["property"].str.lower() == CATEGORY_PROPERTY_KEY].copy()
            if cat.empty:
                continue
            # Convert value to Int64 (nullable), then dropna
            cat["categoryid"] = pd.to_numeric(cat["value"], errors="coerce").astype("Int64")
            cat = cat.dropna(subset=["categoryid"])  # rows where conversion failed
            # Keep latest per itemid within this chunk by timestamp
            cat = cat.sort_values(["itemid", "timestamp"]).drop_duplicates("itemid", keep="last")
            # Merge into global latest
            for row in cat[["itemid", "timestamp", "categoryid"]].itertuples(index=False):
                iid = str(row.itemid)
                ts = int(row.timestamp)
                cid = int(row.categoryid)
                prev = latest.get(iid)
                if prev is None or ts >= prev[0]:
                    latest[iid] = (ts, cid)
            if max_items is not None and len(latest) >= max_items:
                log.info(f"Early stopping after reaching {max_items} unique items")
                stop = True
                break
    # Convert to DataFrame
    out = (
        pd.DataFrame(
            ((iid, ts, cid) for iid, (ts, cid) in latest.items()),
            columns=["itemid", "timestamp", "categoryid"],
        )
        .astype({"itemid": "string", "timestamp": "int64", "categoryid": "int64"})
        .sort_values("itemid")
        .reset_index(drop=True)
    )
    return out


def main():
    parser = argparse.ArgumentParser(description="Build stable item->category mapping (latest by timestamp)")
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS, help="Rows per chunk when streaming item_properties")
    args = parser.parse_args()

    log = configure_logging("item_category")
    ensure_dirs()

    root, data, clean, _ = get_paths()
    part1 = data / "item_properties_part1.csv"
    part2 = data / "item_properties_part2.csv"
    out_path = clean / "item_to_category.csv"

    df = latest_category_per_item([part1, part2], args.chunk_rows, log)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info(f"Wrote mapping to {out_path} with {len(df):,} rows")


if __name__ == "__main__":
    main()
