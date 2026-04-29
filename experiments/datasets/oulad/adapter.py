"""Open University Learning Analytics Dataset (OULAD).

RDD setup: for each student-module-presentation, take the *first* TMA
(tutor-marked assignment) score as the running variable, and the mean of
*subsequent* TMAs as the outcome. The natural threshold is the UK pass
mark of 40 — failing the first TMA is a discouragement / withdrawal
signal.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample

RAW = Path(__file__).parent / "data" / "raw"

X_NUMERIC = ["num_of_prev_attempts", "studied_credits"]
X_CATEGORICAL = ["gender", "highest_education", "imd_band", "age_band", "disability"]


def _ordinal_encode(s: pd.Series) -> np.ndarray:
    return s.astype("category").cat.codes.to_numpy(dtype=float)


def load() -> RDDSample:
    sa_path = RAW / "studentAssessment.csv"
    asm_path = RAW / "assessments.csv"
    si_path = RAW / "studentInfo.csv"
    if not (sa_path.exists() and asm_path.exists() and si_path.exists()):
        raise FileNotFoundError(
            f"OULAD CSVs missing under {RAW}. Run "
            "`python -m experiments.datasets.oulad.download` first."
        )

    sa = pd.read_csv(sa_path)
    asm = pd.read_csv(asm_path)
    si = pd.read_csv(si_path)
    sa["score"] = pd.to_numeric(sa["score"], errors="coerce")

    # Restrict to TMAs and rank within (student, module, presentation) by date.
    tma = sa.merge(asm, on="id_assessment")
    tma = tma[tma["assessment_type"] == "TMA"].copy()
    keys = ["id_student", "code_module", "code_presentation"]
    tma["rank"] = tma.groupby(keys)["date"].rank(method="first")

    first = tma[tma["rank"] == 1][keys + ["score"]].rename(columns={"score": "first_score"})
    later = (
        tma[tma["rank"] > 1]
        .groupby(keys)["score"]
        .mean()
        .reset_index()
        .rename(columns={"score": "later_mean"})
    )
    df = first.merge(later, on=keys).merge(si, on=keys)
    df = df.dropna(subset=["first_score", "later_mean"])

    # Build numeric X: numeric columns + ordinal-encoded categoricals.
    X_parts = [df[c].to_numpy(dtype=float).reshape(-1, 1) for c in X_NUMERIC]
    cat_names = []
    for c in X_CATEGORICAL:
        X_parts.append(_ordinal_encode(df[c]).reshape(-1, 1))
        cat_names.append(f"{c}_code")
    X = np.hstack(X_parts)

    return RDDSample(
        Q=df["first_score"].to_numpy(dtype=float),
        X=X,
        Y=df["later_mean"].to_numpy(dtype=float),
        threshold=40.0,
        name="oulad",
        feature_names=list(X_NUMERIC) + cat_names,
        description=(
            "OULAD first-TMA RDD. Q = first TMA score; "
            "treatment = 1{Q >= 40} (passed UK threshold); "
            "Y = mean score on subsequent TMAs in the same module-presentation."
        ),
        citation="Kuzilek, Hlosta & Zdrahal (2017), Scientific Data",
    )
