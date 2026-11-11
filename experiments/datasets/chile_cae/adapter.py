"""Chile: the PSU 475 line for the state-guaranteed loan (CAE) and scholarships.

See README.md for the download steps (author). This adapter expects, per admission year
``Y`` under ``data/raw/<Y>/``:
  * a scores file with the reading/mathematics average and the covariates (DEMRE
    "Inscritos y puntajes" or the MINEDUC PSU/PDT/PAES file), carrying ``mrun``;
  * optionally a socioeconomic file (DEMRE "Socioeconomico") joined on ``mrun`` or ``ID_aux``;
  * the MINEDUC higher-education enrollment file for year ``Y + 1``
    (``Matricula Ed Superior``), carrying ``mrun``.
Column names differ across years; ``COLUMN_ALIASES`` maps the known variants. ``status()``
lists what is present and which columns were recognized, so the author can see what is
missing before anything is built.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
PROCESSED = HERE / "data" / "processed"
THRESHOLD = 475.0

# canonical name -> known variants (upper-cased, stripped) in the DEMRE / MINEDUC files
COLUMN_ALIASES = {
    "mrun": ["MRUN"],
    "id_aux": ["ID_AUX", "ID"],
    "avg_lm": ["PROM_LM", "PROMLM", "PROMEDIO_LM", "PROMEDIO_CM", "PROMEDIO_CM_MAX",
               "PROM_CL_MAT", "PTJE_PROM_LM", "PROMEDIO_LYC_MATE"],
    "nem": ["PTJE_NEM", "NEM", "PUNTAJE_NEM"],
    "ranking": ["PTJE_RANKING", "RANKING", "PUNTAJE_RANKING"],
    "sex": ["SEXO", "COD_SEXO"],
    "school_dep": ["DEPENDENCIA", "COD_DEPENDENCIA", "GRUPO_DEPENDENCIA"],
    "school_branch": ["RAMA_EDUCACIONAL", "RAMA", "COD_RAMA"],
    "grad_year": ["ANYO_DE_EGRESO", "ANO_EGRESO", "AGNO_EGRESO", "ANIO_EGRESO"],
    "region": ["CODIGO_REGION", "REGION", "COD_REGION", "REGION_DOMICILIO"],
    "income_group": ["INGRESO_BRUTO_FAM", "INGRESO_PERCAPITA_GRUPO_FA", "GRUPO_INGRESO",
                     "INGRESO_FAMILIAR"],
    "educ_father": ["EDUCACION_PADRE", "EDU_PADRE"],
    "educ_mother": ["EDUCACION_MADRE", "EDU_MADRE"],
}


def _read_any(path: Path) -> pd.DataFrame:
    for sep in (";", ",", "\t", "|"):
        try:
            df = pd.read_csv(path, sep=sep, encoding="latin-1", low_memory=False, nrows=5)
            if df.shape[1] > 3:
                return pd.read_csv(path, sep=sep, encoding="latin-1", low_memory=False)
        except Exception:
            continue
    raise ValueError(f"could not parse {path}")


def _canon(df: pd.DataFrame) -> dict:
    cols = {c.strip().upper(): c for c in df.columns}
    found = {}
    for canon, variants in COLUMN_ALIASES.items():
        for v in variants:
            if v in cols:
                found[canon] = cols[v]
                break
    return found


def status() -> dict:
    """What is in data/raw and which columns are recognized, per year and file."""
    out = {}
    for ydir in sorted(RAW.glob("*")):
        if not ydir.is_dir():
            continue
        files = sorted(glob.glob(str(ydir / "*.csv")) + glob.glob(str(ydir / "*.CSV")))
        out[ydir.name] = {}
        for f in files:
            try:
                head = pd.read_csv(f, sep=None, engine="python", encoding="latin-1", nrows=5)
                out[ydir.name][Path(f).name] = {"columns": len(head.columns),
                                                "recognized": sorted(_canon(head))}
            except Exception as e:
                out[ydir.name][Path(f).name] = {"error": str(e)[:80]}
    return out


def build(year: int, force: bool = False) -> pd.DataFrame:
    """Scores and covariates of admission year ``year`` linked by mrun to enrollment in
    ``year + 1``. Raises with a list of what is missing."""
    cache = PROCESSED / f"cae_{year}.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)
    ydir = RAW / str(year)
    if not ydir.exists():
        raise FileNotFoundError(f"{ydir} missing. See README.md; status() lists what is present.")
    scores = soc = enrol = None
    for f in sorted(ydir.glob("*.csv")) + sorted(ydir.glob("*.CSV")):
        df = _read_any(f)
        c = _canon(df)
        if "avg_lm" in c and scores is None:
            scores = (df, c)
        elif ("income_group" in c or "educ_mother" in c) and soc is None:
            soc = (df, c)
    for f in sorted((RAW / str(year + 1)).glob("*.csv")) if (RAW / str(year + 1)).exists() else []:
        df = _read_any(f)
        up = {x.strip().upper() for x in df.columns}
        if "MRUN" in up and ({"NIVEL_GLOBAL", "TIPO_INST_1", "NOMB_INST"} & up):
            enrol = df
            break
    missing = [k for k, v in (("scores file with the LM average", scores),
                              (f"MINEDUC enrollment file for {year + 1} (with mrun)", enrol)) if v is None]
    if missing:
        raise FileNotFoundError(f"year {year}: missing {missing}; status()={status().get(str(year))}")
    df, c = scores
    if "mrun" not in c:
        raise ValueError(f"year {year}: the scores file has no mrun column; the identified DEMRE "
                         "files or the MINEDUC test files are needed to link enrollment (README).")
    d = pd.DataFrame({"mrun": pd.to_numeric(df[c["mrun"]], errors="coerce"),
                      "Q": pd.to_numeric(df[c["avg_lm"]].astype(str).str.replace(",", "."), errors="coerce")})
    for k in ("nem", "ranking", "sex", "school_dep", "school_branch", "grad_year", "region",
              "income_group", "educ_father", "educ_mother"):
        if k in c:
            d[k] = df[c[k]]
    if soc is not None and "mrun" in soc[1]:
        s, cs = soc
        add = {k: s[cs[k]] for k in ("income_group", "educ_father", "educ_mother", "region") if k in cs and k not in d}
        if add:
            d = d.merge(pd.DataFrame({"mrun": pd.to_numeric(s[cs["mrun"]], errors="coerce"), **add}),
                        on="mrun", how="left")
    ecols = {x.strip().upper(): x for x in enrol.columns}
    e = pd.DataFrame({"mrun": pd.to_numeric(enrol[ecols["MRUN"]], errors="coerce")})
    tipo = ecols.get("TIPO_INST_1")
    e["univ"] = enrol[tipo].astype(str).str.upper().str.contains("UNIVERSIDAD").astype(float) if tipo else 1.0
    e = e.groupby("mrun", as_index=False).agg(univ=("univ", "max"))
    d = d.merge(e.assign(any_enroll=1.0), on="mrun", how="left")
    d["any_enroll"] = d.any_enroll.fillna(0.0)
    d["univ"] = d.univ.fillna(0.0)
    d = d[d.Q.notna() & d.mrun.notna()].drop_duplicates("mrun")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    d.to_parquet(cache, index=False)
    return d


def _design(d: pd.DataFrame):
    X = pd.DataFrame(index=d.index)
    for k in ("nem", "ranking"):
        if k in d:
            X[k] = pd.to_numeric(d[k].astype(str).str.replace(",", "."), errors="coerce")
    if "sex" in d:
        X["female"] = (d.sex.astype(str).str.upper().isin(["2", "F", "MUJER"])).astype(float)
    for k in ("school_dep", "school_branch", "region", "income_group", "educ_father", "educ_mother"):
        if k in d:
            dummies = pd.get_dummies(d[k].astype(str), prefix=k, drop_first=True, dtype=float)
            X = pd.concat([X, dummies.iloc[:, :30]], axis=1)
    if "grad_year" in d:
        gy = pd.to_numeric(d.grad_year, errors="coerce")
        X["recent_grad"] = (gy >= gy.max() - 1).astype(float)
    X = X.loc[:, X.std() > 0].fillna(X.median())
    return X


def load(year: int = 2022, outcome: str = "univ") -> RDDSample:
    d = build(year)
    X = _design(d)
    return RDDSample(
        Q=d.Q.to_numpy(float), X=X.to_numpy(float), Y=d[outcome].to_numpy(float),
        threshold=THRESHOLD, name=f"chile_cae_{year}_{outcome}", feature_names=list(X.columns),
        description=(f"Chile admission {year}: Q = PSU/PAES reading-mathematics average; "
                     f"treatment = 1{{Q >= 475}} (CAE loan and scholarship eligibility); Y = {outcome} "
                     f"enrollment in {year + 1}."),
        citation="DEMRE / MINEDUC open data; Solis (JPE 2017)",
    )


# ----------------------------------------------------------------------- PAES-era loan line
# On the PAES scale (admission 2023 onward) the loan requirement for universities is an
# average of 485 in reading and mathematics-1 (beneficiosestudiantiles.cl, CAE; the FSCU
# loan starts at the same score). The PAES files and the enrollment files already in
# ``../chile_admission/data/raw`` carry mrun, so this needs no download.
THRESHOLD_PAES = 485.0


def load_paes(outcome: str = "univ_2024") -> RDDSample:
    """PAES 2024 registrants (the ``chile_admission`` build) with the 485 loan line.
    ``outcome`` in univ_2024, any_2024, univ_2025, any_2025."""
    from experiments.datasets.chile_admission import adapter as A
    df = A.build()
    df = df[df[outcome].notna()]
    return RDDSample(
        Q=df.Q.to_numpy(float), X=df[A.X_COLS].to_numpy(float), Y=df[outcome].to_numpy(float),
        threshold=THRESHOLD_PAES, name=f"chile_cae_paes2024_{outcome}", feature_names=list(A.X_COLS),
        description=("Chile PAES 2024: Q = best reading/mathematics-1 average; treatment = "
                     "1{Q >= 485} (state-guaranteed loan CAE and FSCU eligibility, ITT); "
                     f"Y = {outcome}. The application line 458 sits 27 points below."),
        citation="MINEDUC Centro de Estudios / DEMRE, PAES 2024 and Matricula Ed. Superior; "
                 "beneficiosestudiantiles.cl (CAE requirements)",
    )


def load_paes_univ_2025() -> RDDSample:
    return load_paes("univ_2025")


def load_paes_any_2025() -> RDDSample:
    return load_paes("any_2025")
