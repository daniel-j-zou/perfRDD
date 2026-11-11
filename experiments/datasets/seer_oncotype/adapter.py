"""SEER breast cancer with the Oncotype DX Recurrence Score (chemotherapy line at 25).

Restricted data: the author exports a case listing from SEER*Stat (README.md) to
``data/raw/oncotype_export.csv``; the file never enters git. Column names in SEER*Stat
exports are the dictionary labels; ``COLUMN_ALIASES`` maps the ones used here. ``status()``
reports which were found.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
EXPORT = RAW / "oncotype_export.csv"
THRESHOLD = 25.5          # chemotherapy advised above 25 (TAILORx)

COLUMN_ALIASES = {
    "rs": ["Recurrence Score", "Oncotype DX Recurrence Score", "RS", "ONCOTYPE_RS",
           "Recurrence Score (2004+)"],
    "age": ["Age at diagnosis", "Age recode with <1 year olds", "AGE_DX", "Age"],
    "year": ["Year of diagnosis", "YEAR_DX"],
    "grade": ["Grade", "Grade (thru 2017)", "Grade Recode (thru 2017)"],
    "size_mm": ["CS tumor size (2004-2015)", "Tumor Size Summary (2016+)", "Tumor size"],
    "er": ["ER Status Recode Breast Cancer (1990+)", "ER Status"],
    "pr": ["PR Status Recode Breast Cancer (1990+)", "PR Status"],
    "her2": ["Derived HER2 Recode (2010+)", "HER2 Status"],
    "nodes": ["Regional nodes positive (1988+)", "Nodes positive"],
    "chemo": ["Chemotherapy recode (yes, no/unk)", "Chemotherapy"],
    "survival_months": ["Survival months"],
    "vital": ["Vital status recode (study cutoff used)", "Vital status"],
    "cod": ["SEER cause-specific death classification", "Cause of death"],
    "registry": ["SEER registry", "Registry"],
    "histology": ["Histologic Type ICD-O-3", "Histology"],
}


def _canon(df: pd.DataFrame) -> dict:
    cols = {c.strip().lower(): c for c in df.columns}
    found = {}
    for canon, variants in COLUMN_ALIASES.items():
        for v in variants:
            if v.lower() in cols:
                found[canon] = cols[v.lower()]
                break
    return found


def status() -> dict:
    if not EXPORT.exists():
        return {"export": "missing", "path": str(EXPORT)}
    head = pd.read_csv(EXPORT, nrows=5)
    c = _canon(head)
    return {"export": "present", "columns": list(head.columns), "recognized": sorted(c),
            "missing": sorted(set(COLUMN_ALIASES) - set(c))}


def build(min_year: int = 2010, max_year: int = 2015, horizon_months: int = 60) -> pd.DataFrame:
    if not EXPORT.exists():
        raise FileNotFoundError(f"{EXPORT} missing: export the case listing from SEER*Stat (README.md).")
    df = pd.read_csv(EXPORT, low_memory=False)
    c = _canon(df)
    need = ["rs", "age", "year", "survival_months", "vital"]
    missing = [k for k in need if k not in c]
    if missing:
        raise ValueError(f"export lacks {missing}; status()={status()}")
    num = lambda k: pd.to_numeric(df[c[k]], errors="coerce") if k in c else pd.Series(np.nan, index=df.index)
    d = pd.DataFrame({"rs": num("rs"), "age": num("age"), "year": num("year"),
                      "months": num("survival_months"), "size_mm": num("size_mm"),
                      "nodes": num("nodes")})
    dead = df[c["vital"]].astype(str).str.lower().str.contains("dead")
    d["death5"] = ((d.months < horizon_months) & dead).astype(float)
    d["censored"] = (d.months < horizon_months) & (~dead)
    for k in ("grade", "er", "pr", "her2", "chemo", "registry", "histology"):
        if k in c:
            d[k] = df[c[k]].astype(str)
    d = d[d.rs.between(0, 100) & d.year.between(min_year, max_year) & d.age.between(18, 90)
          & (~d.censored)]
    if "chemo" in d:
        d["chemo_yes"] = d.chemo.str.lower().str.startswith("yes").astype(float)
    return d.reset_index(drop=True)


def load(**kw) -> RDDSample:
    d = build(**kw)
    X = pd.DataFrame({"age": d.age, "age2": (d.age - 60) ** 2, "year": d.year})
    if d.size_mm.notna().any():
        X["size_mm"] = d.size_mm.fillna(d.size_mm.median())
        X["size_missing"] = d.size_mm.isna().astype(float)
    for k in ("grade", "pr", "registry", "histology"):
        if k in d:
            X = pd.concat([X, pd.get_dummies(d[k], prefix=k, drop_first=True, dtype=float).iloc[:, :25]], axis=1)
    X = X.loc[:, X.std() > 0]
    return RDDSample(
        Q=d.rs.to_numpy(float), X=X.to_numpy(float), Y=(1.0 - d.death5).to_numpy(float),
        threshold=THRESHOLD, name="seer_oncotype", feature_names=list(X.columns),
        description="SEER Oncotype DX: Q = Recurrence Score; treatment = 1{Q > 25} (chemotherapy "
                    "advised); Y = survival to 5 years.",
        citation="SEER Research Plus specialized database (restricted); Sparano et al. NEJM 2018",
        extras={"chemo": d.chemo_yes.to_numpy(float) if "chemo_yes" in d else None},
    )
