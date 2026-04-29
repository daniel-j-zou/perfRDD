#!/usr/bin/env python
"""Download NYC TLC 2009 yellow taxi data and build the Haggag-Paci sample.

The Haggag & Paci (2014) "Default Tips" paper uses 2009 credit-card
transactions from the Vendor system (vendor_name == 'VTS'), which had a
$15 fare threshold separating fixed-amount tip suggestions ($2/$3/$4)
from percentage suggestions (20%/25%/30%).

Usage (from the perfrdd repo root):
    python -m experiments.datasets.taxi.download                  # 2009-01 only
    python -m experiments.datasets.taxi.download --months 1 2 3   # selected months
    python -m experiments.datasets.taxi.download --months $(seq 1 12)   # full year

Each month is ~450 MB on disk. The script filters to VTS + credit-card and
appends to a single processed parquet at data/processed/vts_credit.parquet.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
PROCESSED = HERE / "data" / "processed"
PROCESSED_FILE = PROCESSED / "vts_credit.parquet"

URL_TEMPLATE = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2009-{month:02d}.parquet"


def fetch_month(month: int) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    dest = RAW / f"yellow_tripdata_2009-{month:02d}.parquet"
    if dest.exists():
        print(f"  [skip] {dest.name} already present ({dest.stat().st_size/1e6:.0f} MB)")
        return dest
    url = URL_TEMPLATE.format(month=month)
    print(f"  [fetch] {url}")
    urlretrieve(url, dest)
    print(f"  [done]  {dest.stat().st_size/1e6:.0f} MB")
    return dest


def filter_month(parquet_path: Path) -> pd.DataFrame:
    cols = [
        "vendor_name", "Trip_Pickup_DateTime", "Payment_Type",
        "Fare_Amt", "Tip_Amt", "Tolls_Amt", "surcharge", "mta_tax",
        "Trip_Distance", "Passenger_Count",
    ]
    df = pd.read_parquet(parquet_path, columns=cols)
    df = df[df["vendor_name"] == "VTS"]
    df = df[df["Payment_Type"].str.upper() == "CREDIT"]
    df = df[(df["Fare_Amt"] >= 2.5) & (df["Fare_Amt"] <= 200)]
    df = df[df["Tip_Amt"] >= 0]
    return df


def build(months) -> Path:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    frames = []
    for m in months:
        path = fetch_month(m)
        print(f"  [filter] month {m:02d}")
        frames.append(filter_month(path))
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(PROCESSED_FILE, index=False)
    print(f"\n[ok] wrote {PROCESSED_FILE} ({len(out):,} rows, "
          f"{PROCESSED_FILE.stat().st_size/1e6:.0f} MB)")
    return PROCESSED_FILE


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--months", type=int, nargs="+", default=[1])
    args = p.parse_args(argv)
    build(args.months)


if __name__ == "__main__":
    main()
