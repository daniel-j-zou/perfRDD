"""Very low birth weight: NCHS 1995-2002 birth-cohort linked birth/infant-death files.

Q = birth weight in grams (reported, not imputed); the clinical threshold is 1500 g
(very low birth weight), below which infants receive more intensive care (Almond,
Doyle, Kowalski and Williams 2010, QJE). Treatment is 1{Q < 1500}: the rule treats
below, so the estimator scripts mirror the score (see ``mirrored``). Y = infant death
within the first year (the linked death record), binary. X = gestation, plurality, sex,
mother's age, race and Hispanic origin, education, marital status, prenatal care,
birth order, tobacco, weight gain and the father's age, with missing-value flags.

Files (public use; see README.md): ``data/raw/LinkCO<yy>USDen.dat`` (denominator-plus
file, all US births of the year, about 4 million, fixed width 230) and the user guides
``LinkCO<yy>Guide.pdf``. ``build(year)`` parses one fixed-width file and caches
``data/processed/vlbw_<year>.parquet``; the default sample is 2002 alone, ``years=YEARS``
pools 1995-2002.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
PROCESSED = HERE / "data" / "processed"
YEARS = tuple(range(1995, 2003))   # cohort files sharing the 2002 layout (checked 2026-11-11)


def _den(year: int) -> Path:
    m = list(RAW.glob(f"LinkCO{year % 100:02d}US[Dd][Ee][Nn].dat"))
    return m[0] if m else RAW / f"LinkCO{year % 100:02d}USDen.dat"


def _cache(year: int) -> Path:
    return PROCESSED / f"vlbw_{year}.parquet"
THRESHOLD = 1500.0

# 1-based inclusive positions from LinkCO02Guide.pdf (denominator-plus record, length 230)
LAYOUT = {
    "matchs": (1, 1), "resstat": (11, 11), "state": (12, 13),
    "mage": (30, 31), "ormoth": (33, 33), "mrace": (36, 37), "mrace3": (38, 38),
    "meduc": (39, 40), "dmar": (43, 43), "mplbir": (44, 45),
    "totord": (47, 48), "livord": (49, 50), "monpre": (51, 52), "nprevist": (54, 55),
    "fage": (61, 62), "gestimp": (73, 73), "gestat": (74, 75), "csex": (79, 79),
    "bwif": (80, 80), "dbirwt": (81, 84), "plurimp": (88, 88), "dplural": (89, 89),
    "fmaps": (90, 91), "tobacco": (118, 118), "cigar": (119, 120), "wtgain": (126, 127),
    "venl30": (160, 160), "ven30m": (161, 161), "aged": (211, 213), "recwt": (223, 230),
}

X_COLS = ["gestat", "gestat2", "twin", "triplet_plus", "male", "mage", "mage2", "hispanic",
          "black", "other_race", "educ_years", "educ_missing", "unmarried", "no_prenatal",
          "prenatal_month", "prenatal_missing", "first_birth", "livord", "smoker",
          "tobacco_missing", "cigar", "wtgain", "wtgain_missing", "fage", "fage_missing"]


def _require(year: int) -> None:
    if not _den(year).exists():
        raise FileNotFoundError(f"{_den(year)} missing. See README.md for the download steps.")


def build(year: int = 2002, force: bool = False) -> pd.DataFrame:
    """Parse one year's denominator-plus file into a typed frame of US births (cached).
    The 1995-2002 files share the layout (record length 232 with CRLF; checked on the
    year, birth weight, gestation, sex and age-at-death fields of each file)."""
    if _cache(year).exists() and not force:
        return pd.read_parquet(_cache(year))
    _require(year)
    den = _den(year)
    # fixed width 230 plus the line terminator (CRLF)
    size = den.stat().st_size
    reclen = next(r for r in (232, 231, 230) if size % r == 0)
    arr = np.fromfile(den, dtype=np.uint8).reshape(-1, reclen)

    def col(a, b):
        return pd.Series(arr[:, a - 1:b].copy().view(f"S{b - a + 1}").ravel()).str.decode("ascii")

    raw = {k: col(a, b) for k, (a, b) in LAYOUT.items()}
    num = lambda c: pd.to_numeric(raw[c].str.strip(), errors="coerce")
    df = pd.DataFrame({
        "death": (raw["matchs"].str.strip() == "1").astype(float),
        "aged": num("aged"),
        "weight": num("dbirwt"), "weight_imputed": (raw["bwif"].str.strip() == "1"),
        "gestat": num("gestat"), "gest_imputed": (raw["gestimp"].str.strip() != ""),
        "plural": num("dplural"), "csex": num("csex"), "mage": num("mage"),
        "ormoth": num("ormoth"), "mrace3": num("mrace3"), "meduc": num("meduc"),
        "dmar": num("dmar"), "mplbir": num("mplbir"), "livord": num("livord"),
        "totord": num("totord"), "monpre": num("monpre"), "nprevist": num("nprevist"),
        "fage": num("fage"), "fmaps": num("fmaps"), "tobacco": num("tobacco"),
        "cigar": num("cigar"), "wtgain": num("wtgain"),
        "venl30": num("venl30"), "ven30m": num("ven30m"),
        "state": raw["state"].str.strip(), "resstat": num("resstat"),
        "recwt": num("recwt"),
    })
    df["year"] = year
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_cache(year), index=False)
    return df


def sample_frame(weights: tuple = (500.0, 5000.0), years: tuple = (2002,)) -> pd.DataFrame:
    """Births of ``years`` with a reported birth weight in ``weights``, a known gestation,
    singletons and multiples alike, with the covariate columns of ``X_COLS`` built."""
    df = pd.concat([build(y) for y in years], ignore_index=True)
    keep = ((~df.weight_imputed) & df.weight.between(*weights) & (df.weight != 9999)
            & df.gestat.between(17, 47) & (~df.gest_imputed) & df.mage.between(10, 54)
            & df.csex.isin([1, 2]) & df.plural.between(1, 5) & (df.resstat != 4))
    d = df.loc[keep].copy()
    d["gestat2"] = (d.gestat - 38) ** 2
    d["twin"] = (d.plural == 2).astype(float)
    d["triplet_plus"] = (d.plural >= 3).astype(float)
    d["male"] = (d.csex == 1).astype(float)
    d["mage2"] = (d.mage - 27) ** 2
    d["hispanic"] = d.ormoth.between(1, 5).astype(float)
    d["black"] = (d.mrace3 == 3).astype(float)
    d["other_race"] = (d.mrace3 == 2).astype(float)
    educ_known = d.meduc.between(0, 17)
    d["educ_years"] = np.where(educ_known, d.meduc, 12.0)
    d["educ_missing"] = (~educ_known).astype(float)
    d["unmarried"] = (d.dmar == 2).astype(float)
    pre_known = d.monpre.between(0, 9)
    d["no_prenatal"] = ((d.monpre == 0) & pre_known).astype(float)
    d["prenatal_month"] = np.where(pre_known & (d.monpre > 0), d.monpre, 2.0)
    d["prenatal_missing"] = (~pre_known).astype(float)
    ord_known = d.livord.between(0, 31)
    d["first_birth"] = ((d.livord == 1) & ord_known).astype(float)
    d["livord"] = np.where(ord_known, d.livord, 2.0)
    tob_known = d.tobacco.isin([1, 2])
    d["smoker"] = ((d.tobacco == 1) & tob_known).astype(float)
    d["tobacco_missing"] = (~tob_known).astype(float)
    d["cigar"] = np.where(d.cigar.between(0, 98) & tob_known, d.cigar, 0.0)
    wt_known = d.wtgain.between(0, 98)
    d["wtgain"] = np.where(wt_known, d.wtgain, 30.0)
    d["wtgain_missing"] = (~wt_known).astype(float)
    fage_known = d.fage.between(10, 98)
    d["fage"] = np.where(fage_known, d.fage, 30.0)
    d["fage_missing"] = (~fage_known).astype(float)
    return d


def load(weights: tuple = (500.0, 5000.0), years: tuple = (2002,)) -> RDDSample:
    d = sample_frame(weights, years)
    span = f"{min(years)}" if len(years) == 1 else f"{min(years)}-{max(years)}"
    return RDDSample(
        Q=d.weight.to_numpy(float), X=d[X_COLS].to_numpy(float), Y=d.death.to_numpy(float),
        threshold=THRESHOLD, name=f"vlbw_{span}", feature_names=list(X_COLS),
        description=(f"NCHS {span} birth cohort linked file(s). Q = birth weight in grams; "
                     "treatment = 1{Q < 1500} (very low birth weight care); "
                     "Y = infant death within one year."),
        citation=f"NCHS, {span} Birth Cohort Linked Birth/Infant Death Data Sets (public use)",
        treatment_rule=lambda Q: (np.asarray(Q) < THRESHOLD).astype(int),
        extras={"ventilation": (d.venl30.eq(1) | d.ven30m.eq(1)).to_numpy(float),
                "gestat": d.gestat.to_numpy(float), "plural": d.plural.to_numpy(float),
                "year": d.year.to_numpy(float)},
    )


def mirrored(weights: tuple = (500.0, 5000.0), years: tuple = (2002,)):
    """(Q, X, Y, phi0, frame) with the score mirrored so that treatment is Q >= phi0:
    Q = -weight, phi0 = -1499.5 (weights are integers, so this is weight < 1500)."""
    d = sample_frame(weights, years)
    return -d.weight.to_numpy(float), d[X_COLS].to_numpy(float), d.death.to_numpy(float), -1499.5, d
