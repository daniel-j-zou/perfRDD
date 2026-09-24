"""Open University Learning Analytics Dataset (OULAD).

RDD setup: for each student-module-presentation, take the *first* TMA
(tutor-marked assignment) score as the running variable, and the mean of
*subsequent* TMAs as the outcome. The natural threshold is the UK pass
mark of 40 — failing the first TMA is a discouragement / withdrawal
signal.

Assessment due dates are read as numbers.  ``assessments.csv`` codes a
missing date as ``?``, so pandas reads the column as text; ranking text
dates ordered them lexically ("117" < "19" < "54") and, before 2026-09-24,
picked the wrong "first" TMA for about three quarters of the rows.

``load()`` keeps the original covariates (ordinal codes).  ``load_rich()``
uses predetermined covariates chosen for a stronger first stage: one-hot
demographics, module-presentation fixed effects, registration date, VLE
engagement before the first TMA's due date, and the mean CMA score due before
that date (with a missing indicator).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample

RAW = Path(__file__).parent / "data" / "raw"

X_NUMERIC = ["num_of_prev_attempts", "studied_credits"]
X_CATEGORICAL = ["gender", "highest_education", "imd_band", "age_band", "disability"]
KEYS = ["id_student", "code_module", "code_presentation"]


def _ordinal_encode(s: pd.Series) -> np.ndarray:
    return s.astype("category").cat.codes.to_numpy(dtype=float)


def _require(*names: str) -> None:
    missing = [n for n in names if not (RAW / n).exists()]
    if missing:
        raise FileNotFoundError(
            f"OULAD CSVs missing under {RAW}: {missing}. Run "
            "`python -m experiments.datasets.oulad.download` first."
        )


def _frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    """First-TMA score, later-TMA mean, and student info per enrolment."""
    _require("studentAssessment.csv", "assessments.csv", "studentInfo.csv")
    sa = pd.read_csv(RAW / "studentAssessment.csv")
    asm = pd.read_csv(RAW / "assessments.csv")
    si = pd.read_csv(RAW / "studentInfo.csv")
    sa["score"] = pd.to_numeric(sa["score"], errors="coerce")
    asm["date"] = pd.to_numeric(asm["date"], errors="coerce")   # "?" -> NaN

    assessed = sa.merge(asm, on="id_assessment")
    tma = assessed[assessed["assessment_type"] == "TMA"].copy()
    tma["rank"] = tma.groupby(KEYS)["date"].rank(method="first")

    first = tma[tma["rank"] == 1][KEYS + ["score", "date"]].rename(
        columns={"score": "first_score", "date": "first_due"}
    )
    later = (
        tma[tma["rank"] > 1]
        .groupby(KEYS)["score"]
        .mean()
        .reset_index()
        .rename(columns={"score": "later_mean"})
    )
    df = first.merge(later, on=KEYS).merge(si, on=KEYS)
    df = df.dropna(subset=["first_score", "later_mean"]).reset_index(drop=True)
    return df, assessed


def _sample(df: pd.DataFrame, X: np.ndarray, names: list, label: str) -> RDDSample:
    return RDDSample(
        Q=df["first_score"].to_numpy(dtype=float),
        X=X,
        Y=df["later_mean"].to_numpy(dtype=float),
        threshold=40.0,
        name=label,
        feature_names=names,
        description=(
            "OULAD first-TMA RDD. Q = first TMA score; "
            "treatment = 1{Q >= 40} (passed UK threshold); "
            "Y = mean score on subsequent TMAs in the same module-presentation."
        ),
        citation="Kuzilek, Hlosta & Zdrahal (2017), Scientific Data",
    )


def load() -> RDDSample:
    df, _ = _frame()
    X_parts = [df[c].to_numpy(dtype=float).reshape(-1, 1) for c in X_NUMERIC]
    cat_names = []
    for c in X_CATEGORICAL:
        X_parts.append(_ordinal_encode(df[c]).reshape(-1, 1))
        cat_names.append(f"{c}_code")
    X = np.hstack(X_parts)
    return _sample(df, X, list(X_NUMERIC) + cat_names, "oulad")


def load_rich() -> RDDSample:
    """Same sample with predetermined covariates that predict the first TMA."""
    _require("studentRegistration.csv", "studentVle.csv")
    df, assessed = _frame()

    reg = pd.read_csv(RAW / "studentRegistration.csv")
    reg["date_registration"] = pd.to_numeric(reg["date_registration"], errors="coerce")
    df = df.merge(reg[KEYS + ["date_registration"]], on=KEYS, how="left")

    # VLE engagement strictly before the first TMA's due date.
    vle = pd.read_csv(RAW / "studentVle.csv")
    vle = vle.merge(df[KEYS + ["first_due"]], on=KEYS, how="inner")
    vle = vle[vle["date"] < vle["first_due"]]
    engagement = vle.groupby(KEYS).agg(
        clicks=("sum_click", "sum"), active_days=("date", "nunique")
    ).reset_index()
    df = df.merge(engagement, on=KEYS, how="left").fillna({"clicks": 0, "active_days": 0})

    # Computer-marked assessments due before the first TMA.
    cma = assessed[assessed["assessment_type"] == "CMA"].merge(
        df[KEYS + ["first_due"]], on=KEYS
    )
    cma = cma[cma["date"] < cma["first_due"]]
    cma = cma.groupby(KEYS)["score"].mean().rename("cma_before").reset_index()
    df = df.merge(cma, on=KEYS, how="left")

    columns, names = [], []

    def add(values, name):
        columns.append(np.asarray(values, dtype=float).reshape(-1, 1))
        names.append(name)

    for c in X_NUMERIC:
        add(df[c], c)
    dummies = pd.get_dummies(df[X_CATEGORICAL].astype(str), drop_first=True)
    for c in dummies.columns:
        add(dummies[c], c)
    run = df["code_module"] + "_" + df["code_presentation"]
    run_dummies = pd.get_dummies(run, prefix="run", drop_first=True)
    for c in run_dummies.columns:
        add(run_dummies[c], c)
    registration = df["date_registration"]
    add(registration.fillna(registration.median()), "date_registration")
    add(registration.isna(), "date_registration_missing")
    add(np.log1p(df["clicks"]), "log1p_clicks_before_first_due")
    add(df["active_days"] / df["first_due"].clip(lower=1), "active_day_share_before_first_due")
    cma_missing = df["cma_before"].isna()
    add(df["cma_before"].fillna(df["cma_before"].mean()), "cma_mean_before_first_due")
    add(cma_missing, "cma_before_missing")
    X = np.hstack(columns)
    return _sample(df, X, names, "oulad_rich")
