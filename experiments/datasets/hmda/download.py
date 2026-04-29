#!/usr/bin/env python
"""Build the HMDA 2024 processed CSV used by the dual-threshold adapter.

Two stages:

  1. Fetch nationwide originated loans from CFPB's data-browser API,
     state by state (the API rejects nationwide-uncoded requests).
     One CSV per state is saved under data/raw/by_state/.
  2. Concatenate them and apply the cleaning logic from
     `data_process.ipynb` (drop high-NA columns, dropna, build dti_num,
     restrict to CLTV<=100 + total_units==1) to produce
     data/raw/dftest_export.csv.

Idempotent: per-state files that already exist are skipped, and the
processed CSV is rebuilt from cached state files without re-downloading.
"""
from __future__ import annotations

import os
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
BY_STATE = RAW / "by_state"
PROCESSED_CSV = RAW / "dftest_export.csv"

STATES = (
    "AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS "
    "MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV "
    "WI WY DC"
).split()

API = "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv?years=2024&actions_taken=1&states={state}"


def fetch_states() -> None:
    BY_STATE.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(STATES, 1):
        dest = BY_STATE / f"{s}.csv"
        if dest.exists() and dest.stat().st_size > 1000:
            print(f"  [skip] {s} ({dest.stat().st_size/1e6:.1f} MB)")
            continue
        url = API.format(state=s)
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"  [{i}/{len(STATES)}] {s}: {dest.stat().st_size/1e6:.1f} MB")
        except Exception as e:
            print(f"  [{i}/{len(STATES)}] {s}: FAIL {e}")


def _dti_to_numeric(x) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in ("nan", "na", "n/a", "exempt"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        pass
    s = s.replace(" ", "")
    m = re.match(r"^<(\d+(?:\.\d+)?)%$", s)
    if m:
        return float(m.group(1)) / 2
    m = re.match(r"^>(\d+(?:\.\d+)?)%$", s)
    if m:
        return float(m.group(1)) + 5
    m = re.match(r"^(\d+(?:\.\d+)?)%-<(\d+(?:\.\d+)?)%$", s)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2
    m = re.match(r"^(\d+(?:\.\d+)?)%-(\d+(?:\.\d+)?)%$", s)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2
    return np.nan


NUM_COLS = [
    "interest_rate", "combined_loan_to_value_ratio", "property_value", "income",
    "loan_term", "loan_amount", "tract_population", "tract_minority_population_percent",
    "ffiec_msa_md_median_family_income", "tract_to_msa_income_percentage",
    "tract_owner_occupied_units", "tract_one_to_four_family_homes",
    "tract_median_age_of_housing_units", "dti_num",
]


def process() -> Path:
    files = sorted(BY_STATE.glob("*.csv"))
    if not files:
        sys.exit(f"no per-state CSVs in {BY_STATE}; run fetch_states first")
    print(f"  [load] {len(files)} state files")
    df = pd.concat(
        (pd.read_csv(f, low_memory=False) for f in files),
        ignore_index=True,
    )
    print(f"  [load] {len(df):,} rows, {df.shape[1]} cols")

    na_rate = df.isna().mean()
    keep = na_rate[na_rate <= 0.35].index.tolist()
    df = df[keep]
    print(f"  [filter] dropped high-NA cols, {df.shape[1]} cols remain")

    df = df.dropna().reset_index(drop=True)
    print(f"  [filter] dropna -> {len(df):,} rows")

    if "debt_to_income_ratio" in df.columns:
        df["dti_num"] = df["debt_to_income_ratio"].apply(_dti_to_numeric)
        df = df.drop(columns=["debt_to_income_ratio"])

    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "total_units" in df.columns:
        df["total_units"] = pd.to_numeric(df["total_units"], errors="coerce")

    if "combined_loan_to_value_ratio" in df.columns:
        df = df[df["combined_loan_to_value_ratio"] <= 100]
        print(f"  [filter] CLTV <= 100 -> {len(df):,} rows")

    if "total_units" in df.columns:
        df = df[df["total_units"] == 1]
        print(f"  [filter] total_units == 1 -> {len(df):,} rows")

    df = df.dropna(subset=[c for c in NUM_COLS if c in df.columns]).reset_index(drop=True)
    print(f"  [filter] complete-case on numeric cols -> {len(df):,} rows")

    df.to_csv(PROCESSED_CSV, index=False)
    print(f"\n[ok] wrote {PROCESSED_CSV} ({len(df):,} rows, "
          f"{PROCESSED_CSV.stat().st_size/1e6:.0f} MB)")
    return PROCESSED_CSV


def main() -> None:
    print("[stage 1] fetch per-state originations")
    fetch_states()
    print("\n[stage 2] process and export")
    process()


if __name__ == "__main__":
    main()
