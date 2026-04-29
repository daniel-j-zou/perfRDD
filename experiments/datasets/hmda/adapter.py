"""HMDA 2024 mortgage data — dual-threshold RDD.

Two running variables (CLTV ratio and loan amount) with separate cutoffs;
treatment is the AND of both ("above both"). Outcome is interest rate
adjusted by a PMI proxy when CLTV > 80.

The processed dataset `dftest_export.csv` is built by `data_process.ipynb`
from the public HMDA 2024 LAR; see this folder's README.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample

# Preferred location for new datasets:
PRIMARY = Path(__file__).parent / "data" / "raw" / "dftest_export.csv"
# Fallback: legacy location used by the existing notebooks.
FALLBACK = Path(__file__).parent / "dftest_export.csv"

SCORE1_COL = "loan_to_value_ratio"
SCORE2_COL = "loan_amount"
C1, C2 = 80.1, 766550.0
PMI_RATE = 0.008

X_COL = [
    "income", "dti_num",
    "property_value", "loan_term",
    "derived_ethnicity", "derived_race", "derived_sex",
    "derived_loan_product_type", "occupancy_type", "construction_method",
    "loan_type", "loan_purpose", "lien_status",
    "business_or_commercial_purpose", "open_end_line_of_credit",
    "hoepa_status", "reverse_mortgage",
    "activity_year", "county_code",
    "tract_population", "tract_minority_population_percent",
    "ffiec_msa_md_median_family_income", "tract_to_msa_income_percentage",
    "tract_owner_occupied_units", "tract_one_to_four_family_homes",
    "tract_median_age_of_housing_units",
]


def _resolve_path() -> Path:
    if PRIMARY.exists():
        return PRIMARY
    if FALLBACK.exists():
        return FALLBACK
    raise FileNotFoundError(
        f"HMDA processed CSV missing. Expected at one of:\n"
        f"  {PRIMARY}\n"
        f"  {FALLBACK}\n"
        "See experiments/datasets/hmda/README.md for build instructions."
    )


def load() -> RDDSample:
    path = _resolve_path()
    df = pd.read_csv(path, low_memory=False)

    # Build the PMI-adjusted outcome.
    cltv_gt_80 = (df[SCORE1_COL] > 80).astype(int)
    y = df["interest_rate"].to_numpy(dtype=float) + 100.0 * PMI_RATE * cltv_gt_80.to_numpy(dtype=float)

    # Numeric design matrix: drop categoricals so a method receives a clean
    # ndarray. Per-method preprocessing (one-hot encoding) lives in methods/.
    X_numeric = [c for c in X_COL if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    X = df[X_numeric].to_numpy(dtype=float)

    Q = df[[SCORE1_COL, SCORE2_COL]].to_numpy(dtype=float)
    return RDDSample(
        Q=Q,
        X=X,
        Y=y,
        threshold=(C1, C2),
        name="hmda",
        feature_names=list(X_numeric),
        description=(
            "HMDA 2024 mortgage RDD. Q = (CLTV, loan amount); "
            "treatment = both above (80.1, conforming limit); "
            "Y = interest rate + 100 * PMI_RATE * 1{CLTV > 80}."
        ),
        citation="HMDA 2024 LAR (CFPB)",
        extras={"raw_categorical_cols": [c for c in X_COL if c not in X_numeric]},
    )
