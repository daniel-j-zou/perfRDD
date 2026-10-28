"""Chile teacher scholarship (Beca Vocacion de Profesor, BVP), admission 2024.

The same MINEDUC open files as ``chile_admission`` (linked by MRUN), plus the 2024
scholarship-award file.

Q = PROMEDIO_CM_MAX, the best reading/math-1 average (100-1000), centred at the
    student's own cutoff.
Cutoffs (2024): 625 in general; 595 for students in the top 10% of their class who
    graduated in 2023 from a non-private school (municipal, subsidized, delegated
    administration or local public service). Group 1 = that route, group 0 = everyone
    else. Among 2024 award holders, 97% of group 0 have Q >= 625 and 91% of the top-10%
    holders have Q >= 595. The rest come in through other routes, such as PACE and
    earlier scores.
D = 1{Q >= own cutoff}: academically eligible for the scholarship, which pays tuition
    in an accredited teaching degree. It is intent-to-treat. Receiving the award also
    requires enrolling in such a degree and applying for benefits.
X = chile_admission's covariates (top-10% flag included), plus the group indicator.
Y = load(): first-year enrollment in a university education program in 2024
    (teaching degrees; psychopedagogy excluded); load_ped_2025(): enrolled in one in 2025.
    extras["bvp_2024"] is the award (first stage).

Sample: registrants with a valid average and NEM who took an elective test (needed to
apply through the centralized system).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample
from experiments.datasets.chile_admission.adapter import RAW, SCORES, SOCIO, _num

HERE = Path(__file__).parent
PROCESSED = HERE / "data" / "processed"
AWARDS = RAW / "Asignaciones-de-Becas-y-Creditos-2024" / "Asignacion 2024_PA_PUBL.csv"
GROUP_CUTOFFS = {0: 625.0, 1: 595.0}
NON_PRIVATE = {1, 2, 3, 5, 6}   # DEPENDENCIA codes other than 4 (private, unsubsidized)
X_COLS = ["nem", "ranking", "top10", "public", "private", "technical", "nonregular",
          "female", "years_since_grad", "income_group", "income_missing", "metro", "group"]


def _pedagogy(year: int) -> tuple:
    """MRUNs enrolled in a university education program in `year`: all, and first-year."""
    f = next((RAW / f"Matricula-Ed-Superior-{year}").glob("*.csv"))
    m = pd.read_csv(f, sep=";", dtype=str,
                    usecols=["mrun", "nivel_global", "tipo_inst_1", "area_conocimiento",
                             "nomb_carrera", "anio_ing_carr_ori"])
    m = m[(m.nivel_global == "Pregrado") & (m.tipo_inst_1 == "Universidades")
          & (m.area_conocimiento == "Educación")
          & ~m.nomb_carrera.str.upper().str.contains("PSICOPEDAGOG", na=False)]
    mrun = _num(m.mrun)
    first = mrun[m.anio_ing_carr_ori == str(year)]
    return set(mrun.dropna().astype(np.int64)), set(first.dropna().astype(np.int64))


def build() -> pd.DataFrame:
    cache = PROCESSED / "teacher_scholarship_2024.csv.gz"
    if cache.exists():
        return pd.read_csv(cache)
    if not SCORES.exists():
        raise FileNotFoundError(f"{SCORES} missing; see experiments/datasets/chile_admission/README.md")
    s = pd.read_csv(SCORES, sep=";", dtype=str, encoding="utf-8-sig",
                    usecols=["MRUN", "PROMEDIO_CM_MAX", "HCSOC_MAX", "CIEN_MAX", "PTJE_NEM",
                             "PTJE_RANKING", "PORC_SUP_NOTAS", "DEPENDENCIA", "RAMA_EDUCACIONAL",
                             "COD_SEXO", "ANYO_DE_EGRESO", "CODIGO_REGION_EGRESO"])
    e = pd.read_csv(SOCIO, sep=";", dtype=str, encoding="utf-8-sig",
                    usecols=["MRUN", "INGRESO_PERCAPITA_GRUPO_FA"])
    s = s.merge(e, on="MRUN", how="left")
    s["MRUN"] = _num(s.MRUN)
    s = s[s.MRUN.notna() & ~s.MRUN.duplicated(keep=False)]
    s = s[(_num(s.PROMEDIO_CM_MAX) > 0) & (_num(s.PTJE_NEM) > 0)
          & ((_num(s.HCSOC_MAX) > 0) | (_num(s.CIEN_MAX) > 0))]
    dep = _num(s.DEPENDENCIA)
    rama = s.RAMA_EDUCACIONAL.fillna("").str.strip()
    inc = _num(s.INGRESO_PERCAPITA_GRUPO_FA)
    egreso = _num(s.ANYO_DE_EGRESO)
    top10 = _num(s.PORC_SUP_NOTAS) == 10
    group = (top10 & dep.isin(NON_PRIVATE) & (egreso == 2023)).astype(int)
    mrun = s.MRUN.astype(np.int64).to_numpy()
    _, first24 = _pedagogy(2024)
    any25, _ = _pedagogy(2025)
    a = pd.read_csv(AWARDS, sep=";", dtype=str, encoding="latin-1")
    bvp = set(_num(a.MRUN[a.BENEFICIO_BECA_FSCU.str.startswith("BVP", na=False)]).dropna().astype(np.int64))
    q = _num(s.PROMEDIO_CM_MAX).to_numpy()
    cut = np.where(group.to_numpy() == 1, GROUP_CUTOFFS[1], GROUP_CUTOFFS[0])
    out = pd.DataFrame({
        "q_raw": q, "cutoff": cut, "Q": q - cut, "group": group.to_numpy(),
        "nem": _num(s.PTJE_NEM).to_numpy(), "ranking": _num(s.PTJE_RANKING).to_numpy(),
        "top10": top10.astype(float).to_numpy(),
        "public": dep.isin([1, 2, 6]).astype(float).to_numpy(),
        "private": (dep == 4).astype(float).to_numpy(),
        "technical": rama.str.startswith("T").astype(float).to_numpy(),
        "nonregular": rama.isin(["H2", "H3", "H4"]).astype(float).to_numpy(),
        "female": (_num(s.COD_SEXO) == 2).astype(float).to_numpy(),
        "years_since_grad": (2023 - egreso).clip(0, 40).to_numpy(),
        "income_group": inc.where(inc.between(1, 10)).to_numpy(),
        "metro": (_num(s.CODIGO_REGION_EGRESO) == 13).astype(float).to_numpy(),
        "ped_2024": np.isin(mrun, list(first24)).astype(float),
        "ped_2025": np.isin(mrun, list(any25)).astype(float),
        "bvp_2024": np.isin(mrun, list(bvp)).astype(float),
    })
    out["income_missing"] = out.income_group.isna().astype(float)
    out["income_group"] = out.income_group.fillna(out.income_group.median())
    out["years_since_grad"] = out.years_since_grad.fillna(0.0)
    out = out.dropna(subset=["Q", "nem", "ranking"])
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    return out


def _sample(y_col: str) -> RDDSample:
    df = build()
    return RDDSample(
        Q=df.Q.to_numpy(float), X=df[X_COLS].to_numpy(float), Y=df[y_col].to_numpy(float),
        threshold=0.0, name=f"chile_teacher_scholarship_{y_col}", feature_names=list(X_COLS),
        description=("Chile PAES 2024, elective-test takers. Q = best reading/math-1 average "
                     "minus the own cutoff (625; 595 for recent top-10% graduates of non-private "
                     f"schools); treatment = 1{{Q >= 0}} (teacher-scholarship eligibility, ITT); "
                     f"Y = {y_col}."),
        citation="MINEDUC Centro de Estudios / DEMRE, PAES 2024, Matricula Ed. Superior, "
                 "Asignacion de Becas y Creditos 2024",
        treatment_rule=lambda Q: (Q >= 0).astype(int),
        extras={"group": df.group.to_numpy(int), "group_cutoffs": GROUP_CUTOFFS,
                "q_raw": df.q_raw.to_numpy(float), "bvp_2024": df.bvp_2024.to_numpy(float)},
    )


def load() -> RDDSample:
    return _sample("ped_2024")


def load_ped_2025() -> RDDSample:
    return _sample("ped_2025")
