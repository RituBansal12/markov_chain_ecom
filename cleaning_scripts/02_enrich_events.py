"""
Enrich raw events with normalized fields and item categories; write tidy parquet.

Inputs:
- data/events.csv
- data_clean/item_to_category.csv (from 01_build_item_category)

Outputs:
- data_clean/events_enriched.parquet

CLI:
- --chunk-rows: rows per chunk when streaming events

Usage:
  python -m cleaning_scripts.02_enrich_events \
    --chunk-rows 1000000

Notes: Ensures required output directories exist and writes Parquet incrementally for full run.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from cleaning_scripts.common import get_paths, ensure_dirs, DEFAULT_CHUNK_ROWS, configure_logging

ALLOWED_EVENTS = {"view": "VIEW", "addtocart": "ADD_TO_CART", "transaction": "PURCHASE"}


def load_item_category(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"itemid": "string", "timestamp": "int64", "categoryid": "int64"})
    # Keep only itemid->categoryid for join
    return df[["itemid", "categoryid"]]


def normalize_and_filter_events(chunk: pd.DataFrame) -> pd.DataFrame:
    # Drop obvious bad rows
    chunk = chunk.dropna(subset=["timestamp", "visitorid", "event", "itemid"])  # type: ignore
    # Normalize event names
    chunk["event_norm"] = chunk["event"].str.lower().map(ALLOWED_EVENTS)
    chunk = chunk[chunk["event_norm"].notna()]
    # Transaction consistency
    # For purchases, transactionid must be non-null; for others, set null
    is_purchase = chunk["event_norm"] == "PURCHASE"
    chunk.loc[~is_purchase, "transactionid"] = pd.NA
    chunk = chunk[~(is_purchase & chunk["transactionid"].isna())]
    # Convert timestamp to datetime and add date/hour/dow
    ts = pd.to_datetime(chunk["timestamp"], unit="ms", utc=True)
    chunk["ts"] = ts
    chunk["date"] = ts.dt.date.astype("string")
    chunk["hour"] = ts.dt.hour.astype("int16")
    chunk["dow"] = ts.dt.dayofweek.astype("int8")
    return chunk


def main():
    parser = argparse.ArgumentParser(description="Enrich events with item category and tidy fields")
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    args = parser.parse_args()

    log = configure_logging("enrich_events")
    ensure_dirs()

    root, data, clean, _ = get_paths()
    events_path = data / "events.csv"
    map_path = clean / "item_to_category.csv"
    enriched_path = clean / "events_enriched.parquet"

    # Load mapping into memory
    item_cat = load_item_category(map_path)
    log.info(f"Loaded item_to_category mapping: {len(item_cat):,} rows")

    rows_written = 0

    # Stream to parquet using pyarrow ParquetWriter
    writer = None
    for chunk in pd.read_csv(
        events_path,
        usecols=["timestamp", "visitorid", "event", "itemid", "transactionid"],
        dtype={
            "timestamp": "int64",
            "visitorid": "string",
            "event": "string",
            "itemid": "string",
            "transactionid": "string",
        },
        chunksize=args.chunk_rows,
    ):
        chunk = normalize_and_filter_events(chunk)
        chunk = chunk.merge(item_cat, on="itemid", how="left")
        chunk["has_category"] = chunk["categoryid"].notna()
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(str(enriched_path), table.schema)
        writer.write_table(table)
        rows_written += len(chunk)
        log.info(f"Enriched rows written: {rows_written:,}")
    if writer is not None:
        writer.close()

    log.info(f"Wrote enriched events to {enriched_path}")


if __name__ == "__main__":
    main()
