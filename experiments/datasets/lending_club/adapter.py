"""Lending Club loan-level data.

Score Q = FICO score (composite of credit history). LC enforced an
eligibility floor at FICO 660 (early years) and 600 (later). The natural
RDD outcome is the realized loan return / interest rate among approved
loans.

This adapter loads the *accepted* loans archive. Columns vary slightly
by year; missing column failures are common — see README for the
fallback aliases the adapter tries.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample

DATA_PATH = Path(__file__).parent / "data" / "raw" / "loans.csv"

# Y candidates in order of preference: realized internal rate of return on
# the loan, the originated interest rate, or the loan amount.
Y_CANDIDATES = ["int_rate", "interest_rate"]

X_NUMERIC_CANDIDATES = [
    "annual_inc", "dti", "loan_amnt", "term", "emp_length_num",
    "delinq_2yrs", "open_acc", "pub_rec", "revol_util",
    "total_acc", "inq_last_6mths",
]


def _coerce_pct(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        return pd.to_numeric(s.astype(str).str.rstrip("%").str.strip(), errors="coerce")
    return pd.to_numeric(s, errors="coerce")


def load() -> RDDSample:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} missing. Run "
            "`python -m experiments.datasets.lending_club.download` "
            "(requires kaggle CLI auth)."
        )

    df = pd.read_csv(DATA_PATH, low_memory=False)

    # Q: FICO low end of the reported range; fall back to the midpoint if
    # only fico_range_high is present.
    if "fico_range_low" in df.columns:
        df["fico"] = pd.to_numeric(df["fico_range_low"], errors="coerce")
    elif "last_fico_range_low" in df.columns:
        df["fico"] = pd.to_numeric(df["last_fico_range_low"], errors="coerce")
    else:
        raise KeyError("no FICO column found in lending_club loans CSV")

    # Y: interest rate (commonly stored as '13.49%').
    y_col = next((c for c in Y_CANDIDATES if c in df.columns), None)
    if y_col is None:
        raise KeyError(f"none of {Y_CANDIDATES} present in lending_club CSV")
    df["y"] = _coerce_pct(df[y_col])

    x_cols = [c for c in X_NUMERIC_CANDIDATES if c in df.columns]
    for c in x_cols:
        df[c] = _coerce_pct(df[c])

    keep = df[["fico", "y"] + x_cols].notna().all(axis=1)
    df = df[keep]

    return RDDSample(
        Q=df["fico"].to_numpy(dtype=float),
        X=df[x_cols].to_numpy(dtype=float),
        Y=df["y"].to_numpy(dtype=float),
        threshold=660.0,
        name="lending_club",
        feature_names=x_cols,
        description=(
            "Lending Club approved-loans archive. Q = FICO low; "
            "treatment = 1{Q >= 660} (eligibility floor in early years); "
            "Y = originated interest rate."
        ),
        citation="Lending Club historical loan archive (Kaggle: wordsforthewise/lending-club)",
    )
