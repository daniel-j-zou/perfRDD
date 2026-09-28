"""Chile grade retention (MINEDUC public student-performance files).

One cohort: the decision year t (default 2017, before the post-2018 promotion rules),
covariates from t-1 and outcomes from t+1, linked by the masked student ID MRUN.

Q  = annual grade average in year t (1.0-7.0, 0.1 grid).
D  = 1{Q <= 4.4}, eligibility for retention under the 1-failed-subject rule (average
     below 4.5). This is intent-to-treat: the public files have no subject grades, so
     retention itself is fuzzy in Q (see README). The screen mirrors below-cutoff
     designs; the threshold 4.45 makes the treated set exactly {Q <= 4.4}.
X  = prior-year average, attendance and retention; over-age for grade; male; grade
     level; secondary indicator; municipal and private school; rural.
Y  = load(): the year-(t+1) grade average (complete cases; retained students repeat
     the grade, so this outcome favors retention mechanically).
     load_enrolled(): 1 if the student completes year t+1 (promoted or retained).

Sample: students promoted or retained in year t (average > 0) in regular primary
grades 2-8 (COD_ENSE 110) and regular academic secondary grades 1-3 (COD_ENSE 310),
with a year-(t-1) record. Secondary grade 4 is excluded because promoted students
graduate.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
PROCESSED = HERE / "data" / "processed"
THRESHOLD = 4.45
COLS = {"MRUN", "PROM_GRAL", "ASISTENCIA", "SIT_FIN", "COD_ENSE", "COD_GRADO",
        "GEN_ALU", "EDAD_ALU", "COD_DEPE2", "RURAL_RBD"}
X_COLS = ["prior_gpa", "prior_att", "prior_retained", "overage", "male",
          "grade_level", "secondary", "municipal", "private", "rural"]


def _norm(col: str) -> str:
    return col.replace("﻿", "").replace("ï»¿", "").strip().upper()


def _read_year(year: int) -> pd.DataFrame:
    """Students with a final status of promoted or retained, one row per MRUN."""
    files = list((RAW / str(year)).glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSV for {year} under {RAW}. Run "
            "`python -m experiments.datasets.chile_retention.download`.")
    df = pd.read_csv(files[0], sep=";", encoding="latin-1", dtype=str,
                     usecols=lambda c: _norm(c) in COLS)
    df.columns = [_norm(c) for c in df.columns]
    df["SIT_FIN"] = df.SIT_FIN.fillna("").str.strip()
    df = df[df.SIT_FIN.isin(["P", "R"])].copy()
    for c in ("PROM_GRAL", "ASISTENCIA"):
        df[c] = pd.to_numeric(df[c].str.replace(",", ".", regex=False), errors="coerce")
    for c in ("MRUN", "COD_ENSE", "COD_GRADO", "GEN_ALU", "EDAD_ALU", "COD_DEPE2", "RURAL_RBD"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # A student with two P/R records in one year is ambiguous; drop all of them.
    return df[~df.MRUN.duplicated(keep=False)].set_index("MRUN")


def build_cohort(year: int = 2017) -> pd.DataFrame:
    cache = PROCESSED / f"cohort_{year}.csv.gz"
    if cache.exists():
        return pd.read_csv(cache)
    now, prior, nxt = _read_year(year), _read_year(year - 1), _read_year(year + 1)
    regular = (((now.COD_ENSE == 110) & now.COD_GRADO.between(2, 8))
               | ((now.COD_ENSE == 310) & now.COD_GRADO.between(1, 3)))
    now = now[regular & (now.PROM_GRAL > 0)]
    now = now.join(prior[["PROM_GRAL", "ASISTENCIA", "SIT_FIN"]].add_prefix("P_"), how="inner")
    now = now[now.P_PROM_GRAL > 0]
    now = now.join(nxt[["PROM_GRAL"]].add_prefix("N_"), how="left")
    level = np.where(now.COD_ENSE == 310, now.COD_GRADO + 8, now.COD_GRADO)
    out = pd.DataFrame({
        "Q": now.PROM_GRAL.round(1).to_numpy(),
        "prior_gpa": now.P_PROM_GRAL.to_numpy(),
        "prior_att": now.P_ASISTENCIA.to_numpy(),
        "prior_retained": (now.P_SIT_FIN == "R").astype(float).to_numpy(),
        "age": now.EDAD_ALU.to_numpy(),
        "male": (now.GEN_ALU == 1).astype(float).to_numpy(),
        "grade_level": level.astype(float),
        "secondary": (now.COD_ENSE == 310).astype(float).to_numpy(),
        "municipal": (now.COD_DEPE2 == 1).astype(float).to_numpy(),
        "private": (now.COD_DEPE2 == 3).astype(float).to_numpy(),
        "rural": (now.RURAL_RBD == 1).astype(float).to_numpy(),
        "retained": (now.SIT_FIN == "R").astype(float).to_numpy(),
        "next_gpa": now.N_PROM_GRAL.where(now.N_PROM_GRAL > 0).to_numpy(),
    })
    out["enrolled_next"] = out.next_gpa.notna().astype(float)
    # Over-age relative to the median age of the grade level; missing ages -> 0.
    out["overage"] = (out.age - out.groupby("grade_level").age.transform("median")).fillna(0.0)
    out = out.dropna(subset=["prior_att"])
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    return out


def _sample(y_col: str, name: str, year: int) -> RDDSample:
    df = build_cohort(year)
    df = df[df[y_col].notna()]
    return RDDSample(
        Q=df.Q.to_numpy(float),
        X=df[X_COLS].to_numpy(float),
        Y=df[y_col].to_numpy(float),
        threshold=THRESHOLD,
        name=name,
        feature_names=list(X_COLS),
        description=(f"Chile grade retention, decision year {year}. Q = annual average; "
                     f"treatment = 1{{Q <= 4.4}} (retention-eligible, ITT); Y = {y_col}."),
        citation="MINEDUC Centro de Estudios, Rendimiento por estudiante (datos abiertos)",
        treatment_rule=lambda Q: (Q < THRESHOLD).astype(int),
        extras={"retained": df.retained.to_numpy(float)},
    )


def load(year: int = 2017) -> RDDSample:
    return _sample("next_gpa", "chile_retention", year)


def load_enrolled(year: int = 2017) -> RDDSample:
    return _sample("enrolled_next", "chile_retention_enrolled", year)
