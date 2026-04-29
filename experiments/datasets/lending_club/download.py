#!/usr/bin/env python
"""Download Lending Club public loan-stats archives (no auth needed).

The original Lending Club resources URL is still live and serves the
historical accepted-loans CSV bundles (LoanStats3a..LoanStats3d, plus
quarterly files). FICO scores were stripped from the public archive
years ago, so the adapter uses `dti` as the running variable instead.
For a FICO-based RDD, use the Kaggle `wordsforthewise/lending-club`
dataset (auth required) and place the result at the same path.

Usage (from the repo root):
    python -m experiments.datasets.lending_club.download           # 3a only
    python -m experiments.datasets.lending_club.download --all     # 3a..3d
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
LOANS_CSV = RAW / "loans.csv"
BASE = "https://resources.lendingclub.com"
ARCHIVES = ["LoanStats3a", "LoanStats3b", "LoanStats3c", "LoanStats3d"]


def fetch_and_unzip(name: str) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    zip_path = RAW / f"{name}.csv.zip"
    csv_path = RAW / f"{name}.csv"
    if not zip_path.exists():
        url = f"{BASE}/{name}.csv.zip"
        print(f"  [fetch] {url}")
        urlretrieve(url, zip_path)
        print(f"  [done]  {zip_path.stat().st_size/1e6:.0f} MB")
    if not csv_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(RAW)
    return csv_path


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true",
                   help="download all four LoanStats archives (~150 MB)")
    args = p.parse_args(argv)

    archives = ARCHIVES if args.all else ARCHIVES[:1]
    csvs = [fetch_and_unzip(a) for a in archives]

    # Concatenate into the canonical loans.csv that the adapter reads.
    # The public files have a banner first line and a trailing rows-of-summary
    # block; we skip the banner via skiprows=1 and trim trailing junk by
    # requiring the `id` field to be numeric.
    print(f"\n[concat] writing {LOANS_CSV}")
    import pandas as pd
    frames = []
    for c in csvs:
        df = pd.read_csv(c, skiprows=1, low_memory=False)
        df = df[pd.to_numeric(df["loan_amnt"], errors="coerce").notna()]
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(LOANS_CSV, index=False)
    print(f"[ok] {LOANS_CSV} ({len(out):,} rows, "
          f"{LOANS_CSV.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
