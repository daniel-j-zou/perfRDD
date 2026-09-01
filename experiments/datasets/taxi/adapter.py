"""NYC TLC 2009 yellow-taxi data — Haggag & Paci (2014) "Default Tips".

Vendor (VTS) credit-card transactions are the main application: at fare = $15
the suggested-tip system flips from fixed amounts ($2/$3/$4) to percentages
(20%/25%/30%).  The same paper-restricted adapter can also retain Competitor
(CMT) records for placebo and overlap diagnostics.

Q = fare amount, threshold = $15, treatment = above the threshold = percentage
regime, Y = tip amount.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample

DATA_PATH = Path(__file__).parent / "data" / "processed" / "vts_credit.parquet"
RAW_DATA_PATH = (
    Path(__file__).parent / "data" / "raw" / "yellow_tripdata_2009-01.parquet"
)

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


def prepare_haggag_paci_frame(
    frame: pd.DataFrame, *, vendor: str | None = "VTS"
) -> pd.DataFrame:
    """Apply the published main-RDD restrictions to public TLC records.

    Haggag and Paci restrict the main discontinuity sample to credit-card rides
    from the selected vendor before November 2009 with no toll, tax, or
    surcharge; daytime hours; standard-meter fare increments; and fares between
    $5 and $25.  ``vendor='VTS'`` preserves the original application behavior;
    ``vendor='CMT'`` is used by the competitor placebo diagnostic.  Passing
    ``vendor=None`` skips vendor filtering and is useful for constructing a
    pooled standardization reference.
    """
    df = frame.copy()
    if vendor is not None and "vendor_name" in df:
        requested_vendor = str(vendor).strip().upper()
        df = df[
            df["vendor_name"].astype(str).str.strip().str.upper()
            == requested_vendor
        ]
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


def load_haggag_paci_vendor(
    vendor: str = "VTS",
    *,
    standardization_means: np.ndarray | None = None,
    standardization_scales: np.ndarray | None = None,
) -> RDDSample:
    """Load a paper-restricted 2009 January sample for one TLC vendor.

    The processed file contains the VTS application sample.  Competitor (CMT)
    observations are read from the local raw January parquet using only the
    columns needed by the published restrictions and outcome model.  Optional
    standardization moments allow a CMT placebo to use exactly the VTS control
    scale, so the two fitted outcome curves are comparable in the same index
    units.  The returned moments are recorded in ``sample.extras`` for
    reproducibility.
    """
    requested_vendor = str(vendor).strip().upper()
    if requested_vendor == "VTS":
        source_path = DATA_PATH
        if not source_path.exists():
            raise FileNotFoundError(
                f"{source_path} missing. Run "
                "`python -m experiments.datasets.taxi.download` first."
            )
        frame = pd.read_parquet(source_path)
    else:
        source_path = RAW_DATA_PATH
        if not source_path.exists():
            raise FileNotFoundError(
                f"{source_path} missing; the local January raw parquet is needed "
                "for the competitor diagnostic."
            )
        raw_columns = [
            "vendor_name", "Trip_Pickup_DateTime", "Payment_Type",
            "Fare_Amt", "Tip_Amt", "Tolls_Amt", "surcharge", "mta_tax",
            "Trip_Distance", "Passenger_Count",
        ]
        frame = pd.read_parquet(source_path, columns=raw_columns)

    frame = prepare_haggag_paci_frame(frame, vendor=requested_vendor)
    complete = frame[PAPER_X_COLS + ["Fare_Amt", "Tip_Amt"]].notna().all(axis=1)
    frame = frame.loc[complete].copy()
    X = frame[PAPER_X_COLS].to_numpy(dtype=float)
    if standardization_means is None and standardization_scales is None:
        mean = X.mean(axis=0)
        scale = X.std(axis=0)
    elif standardization_means is not None and standardization_scales is not None:
        mean = np.asarray(standardization_means, dtype=float)
        scale = np.asarray(standardization_scales, dtype=float)
        if mean.shape != (len(PAPER_X_COLS),) or scale.shape != mean.shape:
            raise ValueError(
                "standardization moments must have one entry per paper control"
            )
    else:
        raise ValueError(
            "standardization_means and standardization_scales must be supplied "
            "together"
        )
    if np.any(scale <= 0.0):
        raise ValueError("Haggag--Paci controls contain a constant column")
    X = (X - mean) / scale
    return RDDSample(
        Q=frame["Fare_Amt"].to_numpy(dtype=float),
        X=X,
        Y=frame["Tip_Amt"].to_numpy(dtype=float),
        threshold=15.0,
        treatment_rule=_percentage_regime_at_or_above_15,
        name=f"taxi_haggag_paci_{requested_vendor.lower()}",
        feature_names=[f"standardized_{name}" for name in PAPER_X_COLS],
        description=(
            "Public-data analogue of Haggag and Paci's main 2009 "
            f"{requested_vendor} sample: credit-card rides without tolls, taxes, "
            "or surcharges; published daytime restrictions; standard-meter fare "
            "grid; fares $5--$25."
        ),
        citation="Haggag & Paci (2014), AEJ:Applied",
        extras={
            "source_rows_after_paper_restrictions": int(len(frame)),
            "paper_sample_restrictions_applied": True,
            "control_standardization": (
                "full restricted source sample"
                if standardization_means is None
                else "VTS full restricted source sample"
            ),
            "control_standardization_means": [float(value) for value in mean],
            "control_standardization_scales": [float(value) for value in scale],
            "vendor": requested_vendor,
            "source_path": str(source_path),
        },
    )


def load_haggag_paci() -> RDDSample:
    """Load the public-data analogue of the paper's main $15 VTS RDD sample."""
    sample = load_haggag_paci_vendor("VTS")
    # Keep the established public loader name stable for downstream scripts.
    sample.name = "taxi_haggag_paci"
    return sample
