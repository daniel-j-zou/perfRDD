"""Lending Club — DTI threshold, continuous lender-ROI outcome.

Continuous-score / continuous-outcome reframing of the Lending Club archive:

    Q = DTI (debt-to-income, continuous)
    Y = (total_pymnt - funded_amnt) / funded_amnt   (realized lender ROI, continuous)
    D = 1{DTI >= 30}                                 (underwriting trigger)

Only completed loans (Fully Paid / Charged Off / Default) are kept, since ROI is
undefined for loans still current. This is the continuous-outcome counterpart of
``lending_default`` (which used a binary repayment indicator). Screened result:
**boundary** at full n (the 250k interior does not survive) — see RESEARCH_LOG.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample

RAW = Path(__file__).resolve().parent.parent / "lending_club" / "data" / "raw" / "loans.csv"
THRESHOLD = 30.0
FEATURES = [
    "loan_amnt", "annual_inc", "delinq_2yrs",
    "open_acc", "pub_rec", "total_acc", "inq_last_6mths",
]
_COMPLETED = {"Fully Paid", "Charged Off", "Default"}


def load() -> RDDSample:
    cols = ["dti", "loan_status", "total_pymnt", "funded_amnt"] + FEATURES
    df = pd.read_csv(RAW, usecols=cols, low_memory=False)
    df = df[df["loan_status"].isin(_COMPLETED)].copy()
    for c in ["dti", "total_pymnt", "funded_amnt"] + FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["funded_amnt"] > 0]
    df["Y"] = (df["total_pymnt"] - df["funded_amnt"]) / df["funded_amnt"]
    df = df[np.isfinite(df[["dti", "Y"] + FEATURES].to_numpy(float)).all(axis=1)]
    df = df[(df["dti"] >= 0.0) & (df["dti"] <= 60.0) & (df["Y"] > -1.5) & (df["Y"] < 1.5)]
    return RDDSample(
        Q=df["dti"].to_numpy(float),
        X=df[FEATURES].to_numpy(float),
        Y=df["Y"].to_numpy(float),
        threshold=THRESHOLD,
        name="lending_roi",
        feature_names=list(FEATURES),
    )
