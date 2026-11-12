"""Workers' compensation weekly benefits, Kentucky and Michigan (Meyer, Viscusi and Durbin 1995).

Dose design. Temporary total disability benefits are two-thirds of the pre-injury weekly wage,
up to a state maximum. Both states raised the maximum (dates in MVD 1995; not checked here),
so each state has a before and an after schedule:

    A = min(2/3 x prewage, cap(state, period)),   cap in 1982 dollars.

The data give only real (1982-dollar) wages and benefits and no injury date, so the real cap
varies a little across claimants; ``cap`` here is the median benefit of the high earners in
each state-period (KY 149.9 / 246.8, MI 218.3 / 324.0).

The published sample keeps only two groups: low earners, whose benefit is 2/3 of the wage and
did not change, and high earners, above the new maximum, who were at the cap before and after.
Middle earners are excluded, so the score has a gap (KY: no wages between $250 and $350; MI:
between $325 and $625). Low and high earners therefore overlap little in eta, and the dose
effect is identified mainly by the cap increase for high earners against low earners (MVD's
difference in differences), not by transport along the score.

Data: the ``injury`` data frame of the CRAN package ``wooldridge`` (1.4-7, GPL-3), "Professor
Meyer kindly provided the data". See README.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample

HERE = Path(__file__).parent
CSV = HERE / "data" / "processed" / "injury.csv"
RDATA = HERE / "data" / "raw" / "cran_wooldridge" / "wooldridge" / "data" / "injury.RData"
Z_NAMES = ["age", "male"]


def frame() -> pd.DataFrame:
    if not CSV.exists():
        raise FileNotFoundError(f"{CSV} missing; see README.md (Rscript converts {RDATA})")
    d = pd.read_csv(CSV)
    d["state"] = np.where(d.ky == 1, "KY", "MI")
    d["group"] = d.state + np.where(d.afchnge == 1, "_after", "_before").astype(object)
    caps = d[d.highearn == 1].groupby("group").benefit.median()
    d["cap"] = d.group.map(caps)
    d["A"] = np.minimum(2.0 / 3.0 * d.prewage, d.cap)
    return d


def rule_check(d: pd.DataFrame | None = None) -> dict:
    d = frame() if d is None else d
    lo = d.highearn == 0
    return {
        "low_exact_1usd": float((abs(d.benefit - 2 / 3 * d.prewage) < 1)[lo].mean()),
        "low_within_5pct": float((abs(d.benefit / (2 / 3 * d.prewage) - 1) < 0.05)[lo].mean()),
        "high_within_5pct_of_cap": float((abs(d.benefit / d.cap - 1) < 0.05)[~lo].mean()),
        "high_within_10pct_of_cap": float((abs(d.benefit / d.cap - 1) < 0.10)[~lo].mean()),
        "caps": d.groupby("group").cap.first().round(1).to_dict(),
    }


def design(d: pd.DataFrame) -> pd.DataFrame:
    """Predetermined covariates (totmed is after the injury and is left out)."""
    cols = ["age", "male", "married", "hosp", "head", "neck", "upextr", "trunk", "lowback", "lowextr", "occdis",
            "mi", "afchnge"]
    X = pd.concat([d[cols], pd.get_dummies(d.indust.fillna(-1), prefix="ind", drop_first=True),
                   pd.get_dummies(d.injtype.fillna(-1), prefix="inj", drop_first=True)], axis=1).astype(float)
    for c in list(X.columns):
        if X[c].isna().any():
            X[c + "_na"] = X[c].isna().astype(float)
            X[c] = X[c].fillna(X[c].median())
    X["age2"] = X.age ** 2 / 100.0
    return X.loc[:, X.std() > 0]


def load_dose(outcome: str = "ldurat") -> RDDSample:
    """Q = log pre-injury weekly wage (1982 $); A = formula benefit (1982 $); Y = log weeks of benefits."""
    d = frame()
    X = design(d)
    return RDDSample(
        Q=np.log(d.prewage.to_numpy(float)), X=X.to_numpy(float), Y=d[outcome].to_numpy(float),
        threshold=float("nan"), name="wc_injury", feature_names=list(X.columns),
        description="Workers' compensation, KY and MI, before/after cap increases; dose = min(2/3 wage, cap).",
        citation="Meyer, Viscusi and Durbin (1995), AER 85(3); data via R package wooldridge (injury).",
        extras={"A": d.A.to_numpy(float), "a_ref": None, "dose_levels": None, "group": d.group.to_numpy(),
                "highearn": d.highearn.to_numpy(int), "benefit_published": d.benefit.to_numpy(float),
                "Z_names": [z for z in Z_NAMES if z in X.columns], "durat": d.durat.to_numpy(float)},
    )
