"""Chile university-application eligibility (PAES admission process 2024).

MINEDUC open data, linked by the masked student ID MRUN:
  * PAES 2024 registrants and scores (DEMRE), with high-school grades and school;
  * PAES 2024 family-income group;
  * higher-education enrollment 2024 and 2025 (undergraduate).

Q = PROMEDIO_CM_MAX, the best average of the two compulsory tests (reading and
    math 1), on the 100-1000 PAES scale.
D = 1{Q >= 458}: eligible to apply to universities in the centralized admission
    system. Students in the top 10% of their class can also apply, so the design is
    fuzzy; top10 is included in X. Other score rules nearby: the university solidarity
    loan (FSCU) and the state-guaranteed loan (CAE) start at 485, and the Bicentenario
    and Juan Gomez Millas scholarships start around 500-510.
X = high-school score (NEM), ranking score, top-10% indicator, public / private school,
    technical track, non-regular program (night, validation), female, years since
    graduation, family-income group (1-10) with a missing indicator, Santiago region.
Y = load_univ_2024(): enrolled at a university in 2024;
    load_any_2025(): enrolled in any undergraduate program in 2025;
    load_univ_2025(): enrolled at a university in 2025.

Registrants with no valid compulsory score (Q = 0) or no NEM score are excluded.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
PROCESSED = HERE / "data" / "processed"
THRESHOLD = 458.0
X_COLS = ["nem", "ranking", "top10", "public", "private", "technical", "nonregular",
          "female", "years_since_grad", "income_group", "income_missing", "metro"]

# Flexible specification: cubic terms and an interaction in the two grade scores, and
# NEM interacted with the technical track (X may contain any fixed transformations).
FLEX_COLS = X_COLS + ["nem2", "nem3", "ranking2", "ranking3", "nem_x_ranking", "nem_x_technical"]

SCORES = RAW / "PAES-2024-Inscritos-Puntajes" / "A_INSCRITOS_PUNTAJES_PAES_2024_PUB_MRUN.csv"
SOCIO = RAW / "PAES-2024-Socioeconomicos" / "B_SOCIOECONOMICO_DOMICILIO_PAES_2024_PUB_MRUN.csv"


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False).str.strip(),
                         errors="coerce")


def _enrolled(year: int) -> tuple:
    """Undergraduate enrollment in `year`: any program, and university programs."""
    f = next((RAW / f"Matricula-Ed-Superior-{year}").glob("*.csv"))
    m = pd.read_csv(f, sep=";", dtype=str, usecols=["mrun", "tipo_inst_1", "nivel_global"])
    m = m[m.nivel_global == "Pregrado"]
    mrun = _num(m.mrun)
    univ = set(mrun[m.tipo_inst_1 == "Universidades"].dropna().astype(np.int64))
    return set(mrun.dropna().astype(np.int64)), univ


def build() -> pd.DataFrame:
    cache = PROCESSED / "admission_2024.csv.gz"
    if cache.exists():
        return pd.read_csv(cache)
    if not SCORES.exists():
        raise FileNotFoundError(f"{SCORES} missing; see experiments/datasets/chile_admission/README.md")
    s = pd.read_csv(SCORES, sep=";", dtype=str, encoding="utf-8-sig",
                    usecols=["MRUN", "PROMEDIO_CM_MAX", "PTJE_NEM", "PTJE_RANKING",
                             "PORC_SUP_NOTAS", "DEPENDENCIA", "RAMA_EDUCACIONAL", "COD_SEXO",
                             "ANYO_DE_EGRESO", "CODIGO_REGION_EGRESO"])
    e = pd.read_csv(SOCIO, sep=";", dtype=str, encoding="utf-8-sig",
                    usecols=["MRUN", "INGRESO_PERCAPITA_GRUPO_FA"])
    s = s.merge(e, on="MRUN", how="left")
    s["MRUN"] = _num(s.MRUN)
    s = s[s.MRUN.notna() & ~s.MRUN.duplicated(keep=False)]
    q, nem = _num(s.PROMEDIO_CM_MAX), _num(s.PTJE_NEM)
    s = s[(q > 0) & (nem > 0)]
    dep = _num(s.DEPENDENCIA)
    rama = s.RAMA_EDUCACIONAL.fillna("").str.strip()
    inc = _num(s.INGRESO_PERCAPITA_GRUPO_FA)
    mrun = s.MRUN.astype(np.int64).to_numpy()
    any24, univ24 = _enrolled(2024)
    any25, univ25 = _enrolled(2025)
    out = pd.DataFrame({
        "Q": _num(s.PROMEDIO_CM_MAX).to_numpy(),
        "nem": _num(s.PTJE_NEM).to_numpy(),
        "ranking": _num(s.PTJE_RANKING).to_numpy(),
        "top10": (_num(s.PORC_SUP_NOTAS) == 10).astype(float).to_numpy(),
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
        "any_2024": np.isin(mrun, list(any24)).astype(float),
    })
    out["income_missing"] = out.income_group.isna().astype(float)
    out["income_group"] = out.income_group.fillna(out.income_group.median())
    out["years_since_grad"] = out.years_since_grad.fillna(0.0)
    out = out.dropna(subset=["Q", "nem", "ranking"])
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    return out


def _flex(df: pd.DataFrame) -> pd.DataFrame:
    z = lambda v: (v - v.mean()) / v.std()
    n, r = z(df.nem), z(df.ranking)
    return df.assign(nem2=n ** 2, nem3=n ** 3, ranking2=r ** 2, ranking3=r ** 3,
                     nem_x_ranking=n * r, nem_x_technical=n * df.technical)


def _sample(y_col: str, flex: bool = False) -> RDDSample:
    df = build()
    cols = FLEX_COLS if flex else X_COLS
    if flex:
        df = _flex(df)
    return RDDSample(
        Q=df.Q.to_numpy(float), X=df[cols].to_numpy(float), Y=df[y_col].to_numpy(float),
        threshold=THRESHOLD, name=f"chile_admission_{y_col}{'_flex' if flex else ''}",
        feature_names=list(cols),
        description=("Chile PAES 2024. Q = best reading/math-1 average; treatment = "
                     f"1{{Q >= 458}} (university-application eligibility, ITT); Y = {y_col}."),
        citation="MINEDUC Centro de Estudios / DEMRE, PAES 2024 and Matricula Ed. Superior",
        extras={"univ_2024": df.univ_2024.to_numpy(float)},
    )


def load() -> RDDSample:
    return _sample("any_2025")


def load_univ_2024() -> RDDSample:
    return _sample("univ_2024")


def load_any_2025() -> RDDSample:
    return _sample("any_2025")


def load_univ_2025() -> RDDSample:
    return _sample("univ_2025")


def load_univ_2024_flex() -> RDDSample:
    return _sample("univ_2024", flex=True)


def load_any_2025_flex() -> RDDSample:
    return _sample("any_2025", flex=True)


def load_univ_2025_flex() -> RDDSample:
    return _sample("univ_2025", flex=True)
