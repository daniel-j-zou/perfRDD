"""NHANES — HbA1c diabetic threshold RDD.

Q = HbA1c (LBXGH), threshold = 6.5% (ADA diabetic diagnosis cutoff),
Y = systolic blood pressure (BPXSY1), X = age, sex, race, BMI, poverty
index ratio.

The 6.5% threshold doesn't directly *cause* anything in cross-section,
but among people near the cutoff, being above it correlates with being
told you're diabetic and being put on treatment, which has downstream
effects on comorbidities like blood pressure. This is a fuzzier RDD
than the academic ones; treat with care.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample

RAW = Path(__file__).parent / "data" / "raw"

X_COLS = ["RIDAGEYR", "RIAGENDR", "RIDRETH3", "BMXBMI", "INDFMPIR"]
X_NAMES = ["age", "sex", "race_eth", "bmi", "poverty_index"]


def load() -> RDDSample:
    files = ["GHB_J.XPT", "DEMO_J.XPT", "BMX_J.XPT", "BPX_J.XPT"]
    missing = [f for f in files if not (RAW / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"missing NHANES files in {RAW}: {missing}. Run "
            "`python -m experiments.datasets.nhanes.download` first."
        )

    ghb = pd.read_sas(RAW / "GHB_J.XPT")[["SEQN", "LBXGH"]]
    demo = pd.read_sas(RAW / "DEMO_J.XPT")[["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH3", "INDFMPIR"]]
    bmx = pd.read_sas(RAW / "BMX_J.XPT")[["SEQN", "BMXBMI"]]
    bpx = pd.read_sas(RAW / "BPX_J.XPT")[["SEQN", "BPXSY1"]]

    df = ghb.merge(demo, on="SEQN").merge(bmx, on="SEQN").merge(bpx, on="SEQN")
    df = df.dropna(subset=["LBXGH", "BPXSY1"] + X_COLS)

    return RDDSample(
        Q=df["LBXGH"].to_numpy(dtype=float),
        X=df[X_COLS].to_numpy(dtype=float),
        Y=df["BPXSY1"].to_numpy(dtype=float),
        threshold=6.5,
        name="nhanes",
        feature_names=list(X_NAMES),
        description=(
            "NHANES 2017-2018 HbA1c RDD. Q = HbA1c (%); "
            "treatment = 1{Q >= 6.5} (ADA diabetic threshold); "
            "Y = systolic blood pressure (mmHg)."
        ),
        citation="CDC NCHS NHANES 2017-2018 cycle (public data)",
    )
