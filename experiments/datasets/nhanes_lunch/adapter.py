"""NHANES school-lunch income eligibility (Schanzenbach 2009 design), cycles 2005-2016.

Children aged 5-18 who attend kindergarten-12th grade (DBQ360 = 1), pooled over
NHANES cycles D-I (2005-06 to 2015-16). Cycle J is excluded because its household
education and age variables were recoded.

Q = INDFMPIR, family income-to-poverty ratio (0-5, top-coded at 5).
D = 1{Q <= 1.85}: income-eligible for free or reduced-price school meals (free at
    1.30). This is intent-to-treat: eligibility is set on application or direct-
    certification income, not survey income, and the Community Eligibility Provision
    (from 2011) makes some schools universal, so the design is fuzzy.
X = age, age^2, female, race/ethnicity (4 dummies), household size, household
    reference person's education (3 dummies incl. missing), age and married status,
    cycle dummies.
Y = load_lunch_days(): school lunches per week (DBD381, 0-5);
    load_subsidized(): receives free or reduced-price lunch (DBQ390 in {1, 2}), among
      children whose school serves lunch (DBQ370 = 1); non-participants coded 0;
    load_bmi(): measured BMI (BMXBMI).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
CYCLES = "DEFGHI"
THRESHOLD = 1.85
X_COLS = ["age", "age2", "female", "mexican", "other_hisp", "black", "other_race",
          "hh_size", "ref_hs_or_less", "ref_some_college", "ref_edu_missing",
          "ref_age", "ref_married"] + [f"cycle_{c}" for c in CYCLES[1:]]


def build() -> pd.DataFrame:
    frames = []
    for c in CYCLES:
        demo = pd.read_sas(RAW / f"DEMO_{c}.xpt", format="xport")
        dbq = pd.read_sas(RAW / f"DBQ_{c}.xpt", format="xport")
        bmx = pd.read_sas(RAW / f"BMX_{c}.xpt", format="xport")
        d = demo.merge(dbq[["SEQN", "DBQ360", "DBQ370", "DBD381", "DBQ390"]], on="SEQN") \
                .merge(bmx[["SEQN", "BMXBMI"]], on="SEQN", how="left")
        d["cycle"] = c
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d = d[(d.DBQ360 == 1) & d.RIDAGEYR.between(5, 18) & d.INDFMPIR.notna()]
    edu = d.DMDHREDU
    days = d.DBD381.where(d.DBD381.between(0, 5))
    days = days.where(d.DBQ370 != 2, 0.0)          # school does not serve lunch -> 0
    serves = d.DBQ370 == 1
    subsidized = np.where(serves, np.where(d.DBQ390.isin([1, 2]), 1.0, 0.0), np.nan)
    subsidized = np.where(serves & d.DBQ390.isna() & (days > 0), np.nan, subsidized)
    out = pd.DataFrame({
        "Q": d.INDFMPIR.to_numpy(),
        "age": d.RIDAGEYR.to_numpy(), "age2": d.RIDAGEYR.to_numpy() ** 2,
        "female": (d.RIAGENDR == 2).astype(float).to_numpy(),
        "mexican": (d.RIDRETH1 == 1).astype(float).to_numpy(),
        "other_hisp": (d.RIDRETH1 == 2).astype(float).to_numpy(),
        "black": (d.RIDRETH1 == 4).astype(float).to_numpy(),
        "other_race": (d.RIDRETH1 == 5).astype(float).to_numpy(),
        "hh_size": d.DMDHHSIZ.to_numpy(),
        "ref_hs_or_less": edu.isin([1, 2, 3]).astype(float).to_numpy(),
        "ref_some_college": (edu == 4).astype(float).to_numpy(),
        "ref_edu_missing": (~edu.isin([1, 2, 3, 4, 5])).astype(float).to_numpy(),
        "ref_age": d.DMDHRAGE.fillna(d.DMDHRAGE.median()).to_numpy(),
        "ref_married": (d.DMDHRMAR == 1).astype(float).to_numpy(),
        "lunch_days": days.to_numpy(),
        "subsidized": subsidized,
        "bmi": d.BMXBMI.to_numpy(),
    })
    for c in CYCLES[1:]:
        out[f"cycle_{c}"] = (d.cycle == c).astype(float).to_numpy()
    return out


def _sample(y_col: str) -> RDDSample:
    df = build()
    df = df[df[y_col].notna()]
    return RDDSample(
        Q=df.Q.to_numpy(float), X=df[X_COLS].to_numpy(float), Y=df[y_col].to_numpy(float),
        threshold=THRESHOLD, name=f"nhanes_lunch_{y_col}", feature_names=list(X_COLS),
        description=("NHANES 2005-2016 school-age children. Q = income-to-poverty ratio; "
                     f"treatment = 1{{Q <= 1.85}} (free/reduced-price eligibility, ITT); Y = {y_col}."),
        citation="CDC NCHS NHANES public files; design as in Schanzenbach (2009, JHR)",
        treatment_rule=lambda Q: (Q <= THRESHOLD).astype(int),
    )


def load() -> RDDSample:
    return _sample("lunch_days")


def load_lunch_days() -> RDDSample:
    return _sample("lunch_days")


def load_subsidized() -> RDDSample:
    return _sample("subsidized")


def load_bmi() -> RDDSample:
    return _sample("bmi")
