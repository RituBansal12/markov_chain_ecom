import argparse
from pathlib import Path
import pandas as pd
import numpy as np
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
    parser.add_argument("--sample", action="store_true", help="Run in sample mode for quick validation")
    parser.add_argument("--sample-rows", type=int, default=200_000, help="Limit number of events in sample mode")
    args = parser.parse_args()

    log = configure_logging("enrich_events")
    ensure_dirs()

    root, data, clean, _ = get_paths()
    events_path = data / "events.csv"
    map_path = clean / "item_to_category.csv"
    enriched_path = clean / ("events_enriched_sample.parquet" if args.sample else "events_enriched.parquet")

    # Load mapping into memory
    item_cat = load_item_category(map_path)
    log.info(f"Loaded item_to_category mapping: {len(item_cat):,} rows")

    rows_written = 0

    if args.sample:
        # Accumulate into memory (bounded by sample_rows)
        acc = []
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
            # Left join category
            chunk = chunk.merge(item_cat, on="itemid", how="left")
            # Coverage flag
            chunk["has_category"] = chunk["categoryid"].notna()
            remaining = args.sample_rows - rows_written
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining]
            acc.append(chunk)
            rows_written += len(chunk)
            log.info(f"Accumulated rows: {rows_written:,}")
        if acc:
            out = pd.concat(acc, ignore_index=True)
            out.to_parquet(enriched_path, index=False)
    else:
        # Stream to parquet using pyarrow ParquetWriter
        import pyarrow as pa
        import pyarrow.parquet as pq
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
