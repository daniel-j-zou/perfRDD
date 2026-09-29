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
        "GEN_ALU", "EDAD_ALU", "FEC_NAC_ALU", "COD_DEPE", "COD_DEPE2", "RURAL_RBD"}
YOUTH_SECONDARY = (310, 410, 510, 610, 710, 810, 910)
ADULT_SECONDARY = (363, 463, 563, 663, 763, 863, 963)
LAST_YEAR = 2025
X_COLS = ["prior_gpa", "prior_att", "prior_retained", "overage", "male",
          "grade_level", "secondary", "municipal", "private", "rural"]


def _norm(col: str) -> str:
    return col.replace("\ufeff", "").replace("\u00ef\u00bb\u00bf", "").strip().upper()


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
    df = df.reindex(columns=sorted(COLS))   # older years lack EDAD_ALU and COD_DEPE2
    for c in COLS - {"PROM_GRAL", "ASISTENCIA", "SIT_FIN"}:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Age at 30 June from the birth date where EDAD_ALU is absent (before 2016). The
    # birth date is YYYYMM in recent files and YYYYMMDD in older ones.
    born = df.FEC_NAC_ALU.where(df.FEC_NAC_ALU < 1e7, df.FEC_NAC_ALU // 100)
    df["EDAD_ALU"] = df.EDAD_ALU.fillna(year - born // 100 - (born % 100 > 6))
    df.loc[~df.EDAD_ALU.between(4, 30), "EDAD_ALU"] = np.nan
    # Grouped dependency (1 municipal, 2 subsidized, 3 private) from COD_DEPE before 2014.
    df["COD_DEPE2"] = df.COD_DEPE2.fillna(df.COD_DEPE.map({1: 1, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}))
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


def _completers(first: int, last: int = LAST_YEAR) -> set:
    """MRUNs promoted from the final secondary grade in any year first..last: regular
    4th year (youth codes) or the final adult level (adult codes, grade >= 3)."""
    done = set()
    for year in range(first, last + 1):
        f = next((RAW / str(year)).glob("*.csv"))
        d = pd.read_csv(f, sep=";", encoding="latin-1", dtype=str,
                        usecols=lambda c: _norm(c) in {"MRUN", "SIT_FIN", "COD_ENSE", "COD_GRADO"})
        d.columns = [_norm(c) for c in d.columns]
        ense, grado = pd.to_numeric(d.COD_ENSE), pd.to_numeric(d.COD_GRADO)
        fin = ((ense.isin(YOUTH_SECONDARY) & (grado == 4))
               | (ense.isin(ADULT_SECONDARY) & (grado >= 3)))
        ok = fin & (d.SIT_FIN.fillna("").str.strip() == "P")
        done.update(pd.to_numeric(d.MRUN[ok], errors="coerce").dropna().astype(np.int64))
    return done


def build_long_run(year: int = 2012) -> pd.DataFrame:
    """Decision-year cohort with a long-run outcome: finished secondary by LAST_YEAR.

    Grades: primary 6-8 and youth secondary 1-3, including technical-vocational tracks
    (unlike build_cohort). On-time completion is at most 6 years after `year`.
    """
    cache = PROCESSED / f"longrun_{year}.csv.gz"
    if cache.exists():
        return pd.read_csv(cache)
    now, prior, nxt = _read_year(year), _read_year(year - 1), _read_year(year + 1)
    youth = now.COD_ENSE.isin(YOUTH_SECONDARY)
    keep = (((now.COD_ENSE == 110) & now.COD_GRADO.between(6, 8))
            | (youth & now.COD_GRADO.between(1, 3)))
    now = now[keep & (now.PROM_GRAL > 0)]
    now = now.join(prior[["PROM_GRAL", "ASISTENCIA", "SIT_FIN"]].add_prefix("P_"), how="inner")
    now = now[(now.P_PROM_GRAL > 0) & now.P_ASISTENCIA.notna()]
    now = now.join(nxt[["PROM_GRAL"]].add_prefix("N_"), how="left")
    done = _completers(year + 1)
    level = np.where(now.COD_ENSE == 110, now.COD_GRADO, now.COD_GRADO + 8)
    out = pd.DataFrame({
        "Q": now.PROM_GRAL.round(1).to_numpy(),
        "prior_gpa": now.P_PROM_GRAL.to_numpy(),
        "prior_att": now.P_ASISTENCIA.to_numpy(),
        "prior_retained": (now.P_SIT_FIN == "R").astype(float).to_numpy(),
        "age": now.EDAD_ALU.to_numpy(),
        "male": (now.GEN_ALU == 1).astype(float).to_numpy(),
        "grade_level": level.astype(float),
        "secondary": (now.COD_ENSE != 110).astype(float).to_numpy(),
        "municipal": (now.COD_DEPE2 == 1).astype(float).to_numpy(),
        "private": (now.COD_DEPE2 == 3).astype(float).to_numpy(),
        "rural": (now.RURAL_RBD == 1).astype(float).to_numpy(),
        "retained": (now.SIT_FIN == "R").astype(float).to_numpy(),
        "next_gpa": now.N_PROM_GRAL.where(now.N_PROM_GRAL > 0).to_numpy(),
        "completed_secondary": np.isin(now.index.to_numpy(np.int64), list(done)).astype(float),
    })
    out["enrolled_next"] = out.next_gpa.notna().astype(float)
    out["overage"] = (out.age - out.groupby("grade_level").age.transform("median")).fillna(0.0)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    return out


def _sample(y_col: str, name: str, year: int, long_run: bool = False,
            donut: bool = False) -> RDDSample:
    df = build_long_run(year) if long_run else build_cohort(year)
    if donut:
        df = df[df.Q.round(1) != 4.5]
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


def load_completed(year: int = 2012) -> RDDSample:
    """Long-run outcome: finished secondary (regular or adult track) by 2025."""
    return _sample("completed_secondary", "chile_retention_completed", year, long_run=True)


def load_completed_donut(year: int = 2012) -> RDDSample:
    """As load_completed, dropping the heaped value Q = 4.5."""
    return _sample("completed_secondary", "chile_retention_completed_donut", year,
                   long_run=True, donut=True)
