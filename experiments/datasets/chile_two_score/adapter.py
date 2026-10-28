"""Chile university-application eligibility as a two-score rule (PAES 2024).

Same files and outcomes as ``chile_admission`` (MINEDUC open data, linked by MRUN), but
with the two compulsory tests as separate scores:

Q = (CLEC_MAX, MATE1_MAX), the best reading and math-1 scores over the valid sittings
    (100-1000). The published PROMEDIO_CM_MAX equals their average exactly.
D = 1{(Q1 + Q2) / 2 >= 458}: eligible to apply to universities in the centralized
    system. The full DEMRE rule is
        eligible = elective test taken  and  (average >= 458  or  top 10% of class),
    and it reproduces the published HABILITACION_POST flag for all 267,341 registrants
    with a valid score and NEM (checked 2026-10-27). The sample keeps registrants who
    took an elective test and are not in the top 10%, so D is sharp in Q. The top-10%
    group is eligible at any score, so it carries no variation in D.
X = chile_admission's covariates without the top-10% flag (constant here).
Y = load(): any undergraduate enrollment 2025; load_univ_2024(), load_univ_2025().

The deployed rule is a fixed-weight compensatory rule. Counterfactual rules on Q:
other weights, a cutoff on each test (both / either), or a band.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample
from experiments.datasets.chile_admission.adapter import SCORES, SOCIO, _enrolled, _num

HERE = Path(__file__).parent
PROCESSED = HERE / "data" / "processed"
THRESHOLD = 458.0
X_COLS = ["nem", "ranking", "public", "private", "technical", "nonregular", "female",
          "years_since_grad", "income_group", "income_missing", "metro"]


def build() -> pd.DataFrame:
    """All registrants with a valid compulsory average and NEM (chile_admission's
    sample), with the two test scores, the rule's parts and the published flag."""
    cache = PROCESSED / "two_score_2024.csv.gz"
    if cache.exists():
        return pd.read_csv(cache)
    if not SCORES.exists():
        raise FileNotFoundError(f"{SCORES} missing; see experiments/datasets/chile_admission/README.md")
    s = pd.read_csv(SCORES, sep=";", dtype=str, encoding="utf-8-sig",
                    usecols=["MRUN", "PROMEDIO_CM_MAX", "CLEC_MAX", "MATE1_MAX", "HCSOC_MAX",
                             "CIEN_MAX", "HABILITACION_POST", "PTJE_NEM", "PTJE_RANKING",
                             "PORC_SUP_NOTAS", "DEPENDENCIA", "RAMA_EDUCACIONAL", "COD_SEXO",
                             "ANYO_DE_EGRESO", "CODIGO_REGION_EGRESO"])
    e = pd.read_csv(SOCIO, sep=";", dtype=str, encoding="utf-8-sig",
                    usecols=["MRUN", "INGRESO_PERCAPITA_GRUPO_FA"])
    s = s.merge(e, on="MRUN", how="left")
    s["MRUN"] = _num(s.MRUN)
    s = s[s.MRUN.notna() & ~s.MRUN.duplicated(keep=False)]
    s = s[(_num(s.PROMEDIO_CM_MAX) > 0) & (_num(s.PTJE_NEM) > 0)]
    dep = _num(s.DEPENDENCIA)
    rama = s.RAMA_EDUCACIONAL.fillna("").str.strip()
    inc = _num(s.INGRESO_PERCAPITA_GRUPO_FA)
    mrun = s.MRUN.astype(np.int64).to_numpy()
    any24, univ24 = _enrolled(2024)
    any25, univ25 = _enrolled(2025)
    out = pd.DataFrame({
        "cl": _num(s.CLEC_MAX).to_numpy(),
        "m1": _num(s.MATE1_MAX).to_numpy(),
        "cm_avg": _num(s.PROMEDIO_CM_MAX).to_numpy(),
        "elective": ((_num(s.HCSOC_MAX) > 0) | (_num(s.CIEN_MAX) > 0)).astype(float).to_numpy(),
        "top10": (_num(s.PORC_SUP_NOTAS) == 10).astype(float).to_numpy(),
        "eligible_published": _num(s.HABILITACION_POST).isin([1, 2]).astype(float).to_numpy(),
        "nem": _num(s.PTJE_NEM).to_numpy(),
        "ranking": _num(s.PTJE_RANKING).to_numpy(),
        "public": dep.isin([1, 2, 6]).astype(float).to_numpy(),
        "private": (dep == 4).astype(float).to_numpy(),
        "technical": rama.str.startswith("T").astype(float).to_numpy(),
        "nonregular": rama.isin(["H2", "H3", "H4"]).astype(float).to_numpy(),
        "female": (_num(s.COD_SEXO) == 2).astype(float).to_numpy(),
        "years_since_grad": (2023 - _num(s.ANYO_DE_EGRESO)).clip(0, 40).to_numpy(),
        "income_group": inc.where(inc.between(1, 10)).to_numpy(),
        "metro": (_num(s.CODIGO_REGION_EGRESO) == 13).astype(float).to_numpy(),
        "univ_2024": np.isin(mrun, list(univ24)).astype(float),
        "any_2025": np.isin(mrun, list(any25)).astype(float),
        "univ_2025": np.isin(mrun, list(univ25)).astype(float),
    })
    out["income_missing"] = out.income_group.isna().astype(float)
    out["income_group"] = out.income_group.fillna(out.income_group.median())
    out["years_since_grad"] = out.years_since_grad.fillna(0.0)
    out = out.dropna(subset=["cl", "m1", "nem", "ranking"])
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    return out


def rule(df: pd.DataFrame) -> np.ndarray:
    """DEMRE's 2024 eligibility rule, from the scores and the two flags."""
    avg_ok = (df.cl + df.m1) / 2 >= THRESHOLD
    return ((df.elective == 1) & (avg_ok | (df.top10 == 1))).to_numpy(int)


def _average_rule(Q: np.ndarray) -> np.ndarray:
    return (Q.mean(axis=1) >= THRESHOLD).astype(int)


def _sample(y_col: str) -> RDDSample:
    df = build()
    df = df[(df.elective == 1) & (df.top10 == 0) & df[y_col].notna()]
    return RDDSample(
        Q=df[["cl", "m1"]].to_numpy(float), X=df[X_COLS].to_numpy(float),
        Y=df[y_col].to_numpy(float), threshold=(THRESHOLD, THRESHOLD),
        name=f"chile_two_score_{y_col}", feature_names=list(X_COLS),
        description=("Chile PAES 2024, elective-test takers outside the top 10%. "
                     "Q = (best reading, best math-1); treatment = 1{(Q1 + Q2)/2 >= 458} "
                     f"(university-application eligibility); Y = {y_col}."),
        citation="MINEDUC Centro de Estudios / DEMRE, PAES 2024 and Matricula Ed. Superior",
        treatment_rule=_average_rule,
        extras={"weights": (0.5, 0.5), "cutoff": THRESHOLD,
                "eligible_published": df.eligible_published.to_numpy(float)},
    )


def load() -> RDDSample:
    return _sample("any_2025")


def load_univ_2024() -> RDDSample:
    return _sample("univ_2024")


def load_univ_2025() -> RDDSample:
    return _sample("univ_2025")
