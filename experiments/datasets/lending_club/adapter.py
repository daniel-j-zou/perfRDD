"""Lending Club public loan-stats archive.

FICO scores were stripped from the public Lending Club archive
(resources.lendingclub.com) years ago, so this adapter uses
`dti` (debt-to-income ratio) as Q instead — Lending Club's
underwriting policy historically capped DTI at ~35% / 40%, so the
threshold has policy bite.

For a FICO-based version, use the Kaggle `wordsforthewise/lending-club`
dataset (auth required) and override the FICO_COL / threshold below.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample

DATA_PATH = Path(__file__).parent / "data" / "raw" / "loans.csv"

Q_COL = "dti"
Y_COL = "int_rate"
THRESHOLD = 30.0  # a common LC underwriting trigger

X_NUMERIC = [
    "loan_amnt", "annual_inc", "delinq_2yrs", "open_acc",
    "pub_rec", "total_acc", "inq_last_6mths",
]
X_CATEGORICAL = ["term", "home_ownership", "purpose", "verification_status"]


def _coerce_pct(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        return pd.to_numeric(s.astype(str).str.rstrip("%").str.strip(), errors="coerce")
    return pd.to_numeric(s, errors="coerce")


def _ordinal(s: pd.Series) -> np.ndarray:
    return s.astype("category").cat.codes.to_numpy(dtype=float)


def load() -> RDDSample:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} missing. Run "
            "`python -m experiments.datasets.lending_club.download` first."
        )

    df = pd.read_csv(DATA_PATH, low_memory=False)
    df[Q_COL] = pd.to_numeric(df[Q_COL], errors="coerce")
    df[Y_COL] = _coerce_pct(df[Y_COL])

    for c in X_NUMERIC:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    keep_numeric = [c for c in X_NUMERIC if c in df.columns]
    keep_categorical = [c for c in X_CATEGORICAL if c in df.columns]
    df = df.dropna(subset=[Q_COL, Y_COL] + keep_numeric).copy()

    X_parts = [df[c].to_numpy(dtype=float).reshape(-1, 1) for c in keep_numeric]
    cat_names = []
    for c in keep_categorical:
        X_parts.append(_ordinal(df[c].fillna("MISSING")).reshape(-1, 1))
        cat_names.append(f"{c}_code")
    X = np.hstack(X_parts) if X_parts else np.zeros((len(df), 0))

    return RDDSample(
        Q=df[Q_COL].to_numpy(dtype=float),
        X=X,
        Y=df[Y_COL].to_numpy(dtype=float),
        threshold=THRESHOLD,
        name="lending_club",
        feature_names=keep_numeric + cat_names,
        description=(
            "Lending Club public loan-stats archive. Q = DTI; "
            f"treatment = 1{{Q >= {THRESHOLD}}} (LC underwriting trigger); "
            "Y = originated interest rate. FICO is not in the public "
            "archive — substitute when using the Kaggle mirror."
        ),
        citation="Lending Club historical loan-stats archive (resources.lendingclub.com)",
    )
