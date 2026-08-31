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

PAPER_X_COLS = [
    "Trip_Distance",
    "Passenger_Count",
    "hour_of_day",
    "day_of_week",
]


def _percentage_regime_at_or_above_15(q: np.ndarray) -> np.ndarray:
    """Historical Vendor assignment rule used in Haggag--Paci's RDD."""
    return (np.asarray(q) >= 15.0).astype(int)


def prepare_haggag_paci_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the published main-RDD restrictions to public TLC records.

    Haggag and Paci restrict the main discontinuity sample to Vendor credit-card
    rides before November 2009 with no toll, tax, or surcharge; daytime hours;
    standard-meter fare increments; and fares between $5 and $25.  The local
    processed file is already restricted to Vendor credit-card rides, but the
    conditions are repeated where their source columns are available so this
    helper remains safe on a less processed input frame.
    """
    df = frame.copy()
    if "vendor_name" in df:
        df = df[df["vendor_name"] == "VTS"]
    if "Payment_Type" in df:
        df = df[df["Payment_Type"].astype(str).str.upper() == "CREDIT"]

    pickup = pd.to_datetime(df["Trip_Pickup_DateTime"])
    hour = (
        pickup.dt.hour.astype(float)
        + pickup.dt.minute.astype(float) / 60.0
        + pickup.dt.second.astype(float) / 3600.0
    )
    day = pickup.dt.dayofweek.astype(float)
    daytime = (
        ((day < 5) & (hour >= 6.0) & (hour < 16.0))
        | ((day >= 5) & (hour >= 6.0) & (hour < 20.0))
    )
    before_november = pickup < pd.Timestamp("2009-11-01")
    no_tolls = df["Tolls_Amt"].fillna(0.0).eq(0.0)
    no_surcharge = df["surcharge"].fillna(0.0).eq(0.0)
    no_tax = df["mta_tax"].fillna(0.0).eq(0.0)
    fare_in_range = df["Fare_Amt"].between(5.0, 25.0)
    fare_units = np.rint((df["Fare_Amt"] - 2.5) / 0.4)
    standard_meter_grid = np.isclose(
        df["Fare_Amt"], 2.5 + 0.4 * fare_units, atol=1e-6
    )
    keep = (
        before_november
        & daytime
        & no_tolls
        & no_surcharge
        & no_tax
        & fare_in_range
        & standard_meter_grid
    )
    result = df.loc[keep].copy()
    result["hour_of_day"] = hour.loc[keep]
    result["day_of_week"] = day.loc[keep]
    return result


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


def load_haggag_paci() -> RDDSample:
    """Load the public-data analogue of the paper's main $15 RDD sample."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} missing. Run "
            "`python -m experiments.datasets.taxi.download` first."
        )
    frame = prepare_haggag_paci_frame(pd.read_parquet(DATA_PATH))
    complete = frame[PAPER_X_COLS + ["Fare_Amt", "Tip_Amt"]].notna().all(axis=1)
    frame = frame.loc[complete].copy()
    X = frame[PAPER_X_COLS].to_numpy(dtype=float)
    scale = X.std(axis=0)
    if np.any(scale <= 0.0):
        raise ValueError("Haggag--Paci controls contain a constant column")
    X = (X - X.mean(axis=0)) / scale
    return RDDSample(
        Q=frame["Fare_Amt"].to_numpy(dtype=float),
        X=X,
        Y=frame["Tip_Amt"].to_numpy(dtype=float),
        threshold=15.0,
        treatment_rule=_percentage_regime_at_or_above_15,
        name="taxi_haggag_paci",
        feature_names=[f"standardized_{name}" for name in PAPER_X_COLS],
        description=(
            "Public-data analogue of Haggag and Paci's main 2009 Vendor RDD: "
            "credit-card rides without tolls, taxes, or surcharges; published "
            "daytime restrictions; standard-meter fare grid; fares $5--$25."
        ),
        citation="Haggag & Paci (2014), AEJ:Applied",
        extras={
            "source_rows_after_paper_restrictions": int(len(frame)),
            "paper_sample_restrictions_applied": True,
            "control_standardization": "full restricted source sample",
        },
    )
