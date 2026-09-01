"""Lending Club — DTI threshold, loan-repayment outcome.

A performative-RDD-friendly reframing of the Lending Club archive. Unlike the
``lending_club`` adapter (whose outcome is the originated interest rate, a price
the lender sets mechanically from the score), this adapter uses **loan
repayment** as the welfare outcome:

    Y = 1{loan_status == "Fully Paid"}     (0 for Charged Off / Default)

so the policy question is where to set the DTI underwriting cutoff to maximize
realized repayment. The treatment ``D = 1{DTI >= 30}`` is Lending Club's
published underwriting trigger, at which pricing/screening tightens. The effect
of crossing it on repayment is heterogeneous in latent risk, so ``alpha(eta)``
changes sign across the overlap window and the welfare criterion has an interior
optimum — see ``screen_candidate``.

Data: the raw ``loans.csv`` shipped with the ``lending_club`` dataset (public
LoanStats archive). Only completed loans (Fully Paid / Charged Off / Default)
are kept, since repayment is undefined for loans still current.
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
_COMPLETED = {"Fully Paid": 1.0, "Charged Off": 0.0, "Default": 0.0}


def load() -> RDDSample:
    cols = ["dti", "loan_status"] + FEATURES
    df = pd.read_csv(RAW, usecols=cols, low_memory=False)
    df = df[df["loan_status"].isin(_COMPLETED)].copy()
    df["Y"] = df["loan_status"].map(_COMPLETED).astype(float)
    for c in ["dti"] + FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[np.isfinite(df[["dti", "Y"] + FEATURES].to_numpy(float)).all(axis=1)]
    df = df[(df["dti"] >= 0.0) & (df["dti"] <= 60.0)]   # drop out-of-range DTI
    return RDDSample(
        Q=df["dti"].to_numpy(float),
        X=df[FEATURES].to_numpy(float),
        Y=df["Y"].to_numpy(float),
        threshold=THRESHOLD,
        name="lending_default",
        feature_names=list(FEATURES),
    )
