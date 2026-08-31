"""GPA / academic probation (Lindo, Sanders, Oreopoulos 2010).

Score Q = first-year GPA distance from the probation cutoff.
Treatment is *below* zero (placed on academic probation).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample

DATA_PATH = Path(__file__).parent / "Dep_Data" / "final_processed_data.csv"

X_COLS = [
    "hsgrade_pct",
    "totcredits_year1",
    "loc_campus1",
    "loc_campus2",
    "male",
    "bpl_north_america",
    "age_at_entry",
    "english",
]
Q_COL = "dist_from_cut"
Y_COL = "nextGPA"


def _below_zero(Q: np.ndarray) -> np.ndarray:
    return (Q < 0).astype(int)


def load() -> RDDSample:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} missing. See "
            "experiments/datasets/gpa/README.md for download instructions."
        )
    df = pd.read_csv(DATA_PATH)
    df = df[df[Y_COL].notna()].copy()
    return RDDSample(
        Q=df[Q_COL].to_numpy(dtype=float),
        X=df[X_COLS].to_numpy(dtype=float),
        Y=df[Y_COL].to_numpy(dtype=float),
        threshold=0.0,
        name="gpa",
        feature_names=list(X_COLS),
        description=(
            "Academic probation RDD. Q = first-year GPA distance from cutoff; "
            "treatment = 1{Q < 0} (placed on probation); Y = GPA at the next "
            "recorded evaluation minus the applicable probation cutoff. This "
            "complete-case outcome is post-treatment selected; use redesign.py "
            "for policy-oriented full-population outcomes."
        ),
        citation="Lindo, Sanders, Oreopoulos (2010), AEJ:Applied",
        treatment_rule=_below_zero,
    )
