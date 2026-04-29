"""NYC TLC 2009 yellow-taxi data — Haggag & Paci (2014) "Default Tips".

Vendor (VTS) credit-card transactions only: at fare = $15 the suggested-tip
system flips from fixed amounts ($2/$3/$4) to percentages (20%/25%/30%).

Q = fare amount, threshold = $15, treatment = above the threshold = percentage
regime, Y = tip amount.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample

DATA_PATH = Path(__file__).parent / "data" / "processed" / "vts_credit.parquet"

X_COLS_NUMERIC = [
    "Trip_Distance",
    "Passenger_Count",
    "Tolls_Amt",
    "surcharge",
    "hour_of_day",
    "day_of_week",
]


def load() -> RDDSample:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} missing. Run "
            "`python -m experiments.datasets.taxi.download` first."
        )
    df = pd.read_parquet(DATA_PATH)

    # Derive time-of-day and day-of-week features from pickup datetime.
    pickup = pd.to_datetime(df["Trip_Pickup_DateTime"])
    df = df.assign(
        hour_of_day=pickup.dt.hour.astype(float),
        day_of_week=pickup.dt.dayofweek.astype(float),
    )

    # Drop rows with any NaN in the X columns or in Q/Y.
    keep = df[X_COLS_NUMERIC + ["Fare_Amt", "Tip_Amt"]].notna().all(axis=1)
    df = df[keep]

    return RDDSample(
        Q=df["Fare_Amt"].to_numpy(dtype=float),
        X=df[X_COLS_NUMERIC].to_numpy(dtype=float),
        Y=df["Tip_Amt"].to_numpy(dtype=float),
        threshold=15.0,
        name="taxi",
        feature_names=list(X_COLS_NUMERIC),
        description=(
            "NYC TLC 2009 yellow-taxi credit-card transactions, Vendor "
            "(VTS) only. Q = fare amount; treatment = 1{Q >= 15}; "
            "Y = tip amount in dollars."
        ),
        citation="Haggag & Paci (2014), AEJ:Applied",
    )
