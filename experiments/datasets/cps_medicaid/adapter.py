"""Medicaid eligibility at 138 percent of the poverty line: CPS ASEC 2016-2020.

Q = family income as a percent of the family's poverty threshold (100 * FTOTVAL /
FPOVCUT, the ratio behind the ASEC poverty recodes PERLIS / POVLL). In ACA expansion
states, non-elderly adults with income up to 138 percent of the poverty line are
income-eligible for Medicaid; above it they get marketplace subsidies instead. Treatment
is 1{Q <= 138} (treated = income-eligible), so the rule treats below the line and the
estimator scripts should mirror the score (see ``mirrored``). Y = 1 if covered by
Medicaid at any time in the income year (``CAID``), or, with ``load_uninsured``, 1 if
uninsured all year. X = age (and squared), sex, race/ethnicity, education, marital
status, own children under 18 in the family, family size, employment status, state and
income-year indicators.

Sample: civilian adults 19-64 in the poverty universe, in states where the expansion was
in force for the whole income year (``EXPANSION_DATES``, KFF), not disabled-eligible
(no SSI receipt, no Medicare, no work-preventing disability). Pregnancy is not asked in
the ASEC, so that exclusion cannot be made (see README.md).

Files (public use, Census Bureau; see download.py): survey years 2019 and 2020 as CSV
zips (hhpub / ffpub / pppub members), survey years 2016-2018 as the hierarchical
fixed-width ``asec<year>_pubuse.dat.gz`` (record type in column 1; the layout below is
identical in the three data dictionaries). ``build()`` parses all five years once and
caches ``data/processed/cps_asec_2016_2020.parquet``.
"""
from __future__ import annotations

import datetime as dt
import gzip
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
PROCESSED = HERE / "data" / "processed"
CACHE = PROCESSED / "cps_asec_2016_2020.parquet"
THRESHOLD = 138.0
SURVEY_YEARS = (2016, 2017, 2018, 2019, 2020)  # income year = survey year - 1

FW_FILES = {2016: "asec2016_pubuse_v3.dat.gz", 2017: "asec2017_pubuse.dat.gz",
            2018: "asec2018_pubuse.dat.gz"}
CSV_FILES = {2019: "asecpub19csv.zip", 2020: "asecpub20csv.zip"}
RECLEN = 1076  # plus the newline

# 1-based start and width, from 08ASEC2018_Data_Dict_Full.txt (same in 2016 and 2017)
LAYOUT_H = {"H_SEQ": (2, 5), "H_HHTYPE": (20, 1), "GESTFIPS": (42, 2)}
LAYOUT_F = {"FH_SEQ": (2, 5), "FFPOS": (7, 2), "FKIND": (9, 1), "FTYPE": (10, 1),
            "FPERSONS": (11, 2), "FOWNU18": (27, 1), "FRELU18": (29, 1),
            "FPOVCUT": (32, 5), "POVLL": (38, 2), "FTOTVAL": (205, 8)}
LAYOUT_P = {"PH_SEQ": (2, 5), "PPPOS": (7, 2), "A_AGE": (19, 2), "A_MARITL": (21, 1),
            "A_SEX": (24, 1), "A_HGA": (25, 2), "PRDTRACE": (27, 2), "PRPERTYP": (30, 1),
            "PEHSPNON": (31, 1), "A_FAMTYP": (35, 1), "A_FAMREL": (36, 1),
            "PF_SEQ": (48, 2), "PRDISFLG": (82, 2), "PRCITSHP": (95, 1),
            "MARSUPWT": (155, 8), "A_LFSR": (200, 1), "A_WKSTAT": (202, 1),
            "PEMLR": (218, 1), "WKSWORK": (258, 2), "HRSWK": (268, 2),
            "SSI_YN": (432, 1), "DIS_HP": (483, 1), "PTOTVAL": (580, 8),
            "PEARNVAL": (588, 8), "PERLIS": (606, 1), "POV_UNIV": (607, 1),
            "WICYN": (608, 1), "MCARE": (629, 1), "MCAID": (635, 1), "CHAMP": (641, 1),
            "OUT": (667, 1), "CARE": (668, 1), "CAID": (669, 1), "MON": (670, 2),
            "OTH": (672, 1), "OTYP_1": (673, 1), "OTYP_2": (674, 1), "OTYP_3": (675, 1),
            "OTYP_4": (676, 1), "OTYP_5": (677, 1), "HEA": (691, 1), "IHSFLG": (692, 1),
            "AHIPER": (693, 1), "PCHIP": (706, 1), "COV_GH": (707, 1), "COV_HI": (708, 1),
            "CH_MC": (709, 1), "CH_HI": (710, 1)}

CSV_H = ["H_SEQ", "H_HHTYPE", "GESTFIPS"]
CSV_F = ["FH_SEQ", "FFPOS", "FKIND", "FTYPE", "FPERSONS", "FOWNU18", "FRELU18", "FPOVCUT",
         "POVLL", "FTOTVAL"]
CSV_P = ["PH_SEQ", "PPPOS", "PF_SEQ", "A_AGE", "A_MARITL", "A_SEX", "A_HGA", "PRDTRACE",
         "PRPERTYP", "PEHSPNON", "A_FAMTYP", "A_FAMREL", "PRDISFLG", "PRCITSHP",
         "MARSUPWT", "A_LFSR", "A_WKSTAT", "PEMLR", "WKSWORK", "HRSWK", "SSI_YN", "DIS_HP",
         "PTOTVAL", "PEARNVAL", "PERLIS", "POV_UNIV", "WICYN", "COV", "PUB", "PRIV", "MRK",
         "CAID", "MCAID", "MCARE", "HEA"]

# KFF, "Status of State Action on the Medicaid Expansion Decision" (as of November
# 2026; fetched 2026-11-11): implementation dates of the ACA adult expansion. States
# absent here had not expanded by 2019 (AL, FL, GA, KS, MS, NC, OK, SC, SD, TN, TX, WI,
# WY, and MO, NE's 10/2020 start is after the last income year).
EXPANSION_DATES = {
    "AK": dt.date(2015, 9, 1), "AZ": dt.date(2014, 1, 1), "AR": dt.date(2014, 1, 1),
    "CA": dt.date(2014, 1, 1), "CO": dt.date(2014, 1, 1), "CT": dt.date(2014, 1, 1),
    "DE": dt.date(2014, 1, 1), "DC": dt.date(2014, 1, 1), "HI": dt.date(2014, 1, 1),
    "ID": dt.date(2020, 1, 1), "IL": dt.date(2014, 1, 1), "IN": dt.date(2015, 2, 1),
    "IA": dt.date(2014, 1, 1), "KY": dt.date(2014, 1, 1), "LA": dt.date(2016, 7, 1),
    "ME": dt.date(2019, 1, 10), "MD": dt.date(2014, 1, 1), "MA": dt.date(2014, 1, 1),
    "MI": dt.date(2014, 4, 1), "MN": dt.date(2014, 1, 1), "MO": dt.date(2021, 10, 1),
    "MT": dt.date(2016, 1, 1), "NE": dt.date(2020, 10, 1), "NV": dt.date(2014, 1, 1),
    "NH": dt.date(2014, 8, 15), "NJ": dt.date(2014, 1, 1), "NM": dt.date(2014, 1, 1),
    "NY": dt.date(2014, 1, 1), "NC": dt.date(2023, 12, 1), "ND": dt.date(2014, 1, 1),
    "OH": dt.date(2014, 1, 1), "OK": dt.date(2021, 7, 1), "OR": dt.date(2014, 1, 1),
    "PA": dt.date(2015, 1, 1), "RI": dt.date(2014, 1, 1), "SD": dt.date(2023, 7, 1),
    "UT": dt.date(2020, 1, 1), "VT": dt.date(2014, 1, 1), "VA": dt.date(2019, 1, 1),
    "WA": dt.date(2014, 1, 1), "WV": dt.date(2014, 1, 1),
}
# a state counts as "expansion" for an income year when the expansion was in force by
# January 31 of that year (the only case this grace period decides is Maine, 1/10/2019);
# states that expanded later in the year are "partial" and dropped for that year.
FULL_YEAR_BY = (1, 31)

FIPS = {1: "AL", 2: "AK", 4: "AZ", 5: "AR", 6: "CA", 8: "CO", 9: "CT", 10: "DE", 11: "DC",
        12: "FL", 13: "GA", 15: "HI", 16: "ID", 17: "IL", 18: "IN", 19: "IA", 20: "KS",
        21: "KY", 22: "LA", 23: "ME", 24: "MD", 25: "MA", 26: "MI", 27: "MN", 28: "MS",
        29: "MO", 30: "MT", 31: "NE", 32: "NV", 33: "NH", 34: "NJ", 35: "NM", 36: "NY",
        37: "NC", 38: "ND", 39: "OH", 40: "OK", 41: "OR", 42: "PA", 44: "RI", 45: "SC",
        46: "SD", 47: "TN", 48: "TX", 49: "UT", 50: "VT", 51: "VA", 53: "WA", 54: "WV",
        55: "WI", 56: "WY"}

BASE_COLS = ["age", "age2", "female", "hispanic", "black", "asian", "other_race",
             "lt_hs", "some_college", "ba_plus", "married", "n_children", "fam_size",
             "employed", "unemployed"]


def expansion_status(state: str, income_year: int) -> str:
    """'expansion', 'partial' (expanded during the year) or 'none'."""
    d = EXPANSION_DATES.get(state)
    if d is None or d.year > income_year:
        return "none"
    return "expansion" if d <= dt.date(income_year, *FULL_YEAR_BY) else "partial"


# ----------------------------------------------------------------------------- parsing

def _require(year: int) -> Path:
    name = FW_FILES.get(year) or CSV_FILES[year]
    p = RAW / str(year) / name
    if not p.exists():
        raise FileNotFoundError(f"{p} missing. Run download.py (see README.md).")
    return p


def _fw_frame(arr: np.ndarray, layout: dict) -> pd.DataFrame:
    out = {}
    for k, (a, w) in layout.items():
        s = pd.Series(arr[:, a - 1:a - 1 + w].copy().view(f"S{w}").ravel()).str.decode("ascii")
        out[k] = pd.to_numeric(s.str.strip(), errors="coerce")
    return pd.DataFrame(out)


def _read_fixed_width(year: int):
    with gzip.open(_require(year), "rb") as f:
        raw = f.read()
    reclen = next(r for r in (RECLEN + 1, RECLEN + 2, RECLEN) if len(raw) % r == 0)
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(-1, reclen)
    rtype = arr[:, 0]
    hh = _fw_frame(arr[rtype == ord("1")], LAYOUT_H)
    fam = _fw_frame(arr[rtype == ord("2")], LAYOUT_F)
    per = _fw_frame(arr[rtype == ord("3")], LAYOUT_P)
    # legacy (2014-2018) files have no "any coverage" recode: build it from the
    # component recodes and items. Indian Health Service alone does not count as
    # coverage (Census convention). Checked against the published uninsured rates
    # (all ages, weighted): 2015 9.1, 2016 8.8, 2017 8.8 percent (README.md).
    oth_ins = ((per.OTYP_1 == 1) | (per.OTYP_2 == 1) | (per.OTYP_3 == 1) | (per.OTYP_5 == 1)
               | ((per.OTH == 1) & (per.IHSFLG != 1)))
    medicaid = (per.CAID == 1) | (per.MCAID == 1) | (per.CH_MC == 1)
    medicare = (per.CARE == 1) | (per.MCARE == 1)
    private = (per.COV_GH == 1) | (per.COV_HI == 1) | (per.OUT == 1) | per.CH_HI.isin([1, 2])
    public = medicaid | medicare | (per.CHAMP == 1) | (per.PCHIP == 1) | oth_ins
    per["COV"] = np.where(private | public | (per.AHIPER == 1), 1, 2)
    per["PUB"] = np.where(public, 1, 2)
    per["PRIV"] = np.where(private, 1, 2)
    per["MRK"] = np.nan  # marketplace flag only from 2019
    per["MEDICAID"] = np.where(medicaid, 1, 2)
    per["MCARE"] = np.where(medicare, 1, 2)
    return hh, fam, per


def _read_csv(year: int):
    z = zipfile.ZipFile(_require(year))
    names = {m.filename.split("/")[-1].lower(): m.filename for m in z.infolist()}
    yy = year % 100

    def read(member, cols):
        with z.open(names[f"{member}{yy}.csv"]) as f:
            return pd.read_csv(f, usecols=cols)

    hh, fam, per = read("hhpub", CSV_H), read("ffpub", CSV_F), read("pppub", CSV_P)
    # 2019+ files: CAID is "Medicaid coverage last year"; MCAID there bundles CHIP and
    # other means-tested coverage, so it is kept only as a raw field
    per["MEDICAID"] = per.CAID
    return hh, fam, per


def _assemble(year: int, hh, fam, per) -> pd.DataFrame:
    per = per.merge(hh, left_on="PH_SEQ", right_on="H_SEQ", how="left")
    per = per.merge(fam, left_on=["PH_SEQ", "PF_SEQ"], right_on=["FH_SEQ", "FFPOS"], how="left")
    d = pd.DataFrame({
        "survey_year": year, "income_year": year - 1,
        "hh_seq": per.PH_SEQ, "pppos": per.PPPOS, "fam_pos": per.PF_SEQ,
        "state_fips": per.GESTFIPS, "weight": per.MARSUPWT / 100.0,  # 2 implied decimals
        "age": per.A_AGE, "sex": per.A_SEX, "race": per.PRDTRACE, "hispanic_raw": per.PEHSPNON,
        "educ": per.A_HGA, "marital": per.A_MARITL, "pemlr": per.PEMLR, "wkstat": per.A_WKSTAT,
        "wkswork": per.WKSWORK, "hrswk": per.HRSWK, "citizenship": per.PRCITSHP,
        "prpertyp": per.PRPERTYP, "famtyp": per.A_FAMTYP, "famrel": per.A_FAMREL,
        "prdisflg": per.PRDISFLG, "dis_hp": per.DIS_HP, "ssi_yn": per.SSI_YN,
        "ptotval": per.PTOTVAL, "pearnval": per.PEARNVAL, "perlis": per.PERLIS,
        "pov_univ": per.POV_UNIV, "wic": per.WICYN, "health": per.HEA,
        "any_cov": per.COV, "pub": per.PUB, "priv": per.PRIV, "mrk": per.MRK,
        "medicaid_yr": per.MEDICAID, "caid": per.CAID, "mcaid": per.MCAID, "mcare": per.MCARE,
        "fkind": per.FKIND, "ftype": per.FTYPE, "fpersons": per.FPERSONS,
        "fownu18": per.FOWNU18, "frelu18": per.FRELU18, "fpovcut": per.FPOVCUT,
        "povll": per.POVLL, "ftotval": per.FTOTVAL,
    })
    return d


def build(force: bool = False) -> pd.DataFrame:
    """All persons of the five ASEC files with the fields above (cached parquet)."""
    if CACHE.exists() and not force:
        return pd.read_parquet(CACHE)
    frames = []
    for y in SURVEY_YEARS:
        hh, fam, per = _read_fixed_width(y) if y in FW_FILES else _read_csv(y)
        frames.append(_assemble(y, hh, fam, per))
    df = pd.concat(frames, ignore_index=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE, index=False)
    return df


# ------------------------------------------------------------------------------ sample

def sample_frame(ages=(19, 64), drop_disabled: bool = True, citizens_only: bool = False,
                 states: str = "expansion") -> pd.DataFrame:
    """Adults in the poverty universe with the covariates built.

    states: 'expansion' (default) keeps state-years with the expansion in force all year;
    'all' keeps every state-year and adds the ``expansion`` column for the caller."""
    df = build()
    df["state"] = df.state_fips.map(FIPS)
    df["expansion"] = [expansion_status(s, y) for s, y in zip(df.state, df.income_year)]
    keep = (df.age.between(*ages) & (df.prpertyp == 2) & (df.pov_univ == 1)
            & (df.fpovcut > 0) & df.ftotval.notna())
    if states == "expansion":
        keep &= df.expansion == "expansion"
    if drop_disabled:
        keep &= ~((df.ssi_yn == 1) | (df.mcare == 1) | (df.dis_hp == 1))
    if citizens_only:
        keep &= df.citizenship.between(1, 4)
    d = df.loc[keep].copy()
    d["Q"] = 100.0 * d.ftotval / d.fpovcut
    d["medicaid"] = (d.medicaid_yr == 1).astype(float)
    d["uninsured"] = (d.any_cov == 2).astype(float)
    d["any_public"] = (d.pub == 1).astype(float)
    d["private"] = (d.priv == 1).astype(float)
    d["age2"] = (d.age - 40.0) ** 2
    d["female"] = (d.sex == 2).astype(float)
    d["hispanic"] = (d.hispanic_raw == 1).astype(float)
    d["black"] = (d.race == 2).astype(float)
    d["asian"] = (d.race == 4).astype(float)
    d["other_race"] = (~d.race.isin([1, 2, 4])).astype(float)
    d["lt_hs"] = (d.educ < 39).astype(float)
    d["some_college"] = d.educ.between(40, 42).astype(float)
    d["ba_plus"] = (d.educ >= 43).astype(float)
    d["married"] = d.marital.between(1, 3).astype(float)
    d["n_children"] = d.fownu18.astype(float)
    d["fam_size"] = d.fpersons.astype(float)
    d["employed"] = d.pemlr.isin([1, 2]).astype(float)
    d["unemployed"] = d.pemlr.isin([3, 4]).astype(float)
    d["noncitizen"] = (d.citizenship == 5).astype(float)
    d["any_difficulty"] = (d.prdisflg == 1).astype(float)
    return d


def design_columns(d: pd.DataFrame, base_state: str = "CA", base_year: int = 2015):
    """Append state and income-year indicators (one base each); returns the column list."""
    cols = list(BASE_COLS)
    for s in sorted(d.state.unique()):
        if s != base_state:
            d[f"state_{s}"] = (d.state == s).astype(float)
            cols.append(f"state_{s}")
    for y in sorted(d.income_year.unique()):
        if y != base_year:
            d[f"year_{y}"] = (d.income_year == y).astype(float)
            cols.append(f"year_{y}")
    return cols


def _make(d: pd.DataFrame, outcome: str, name: str, desc: str) -> RDDSample:
    cols = design_columns(d)
    return RDDSample(
        Q=d.Q.to_numpy(float), X=d[cols].to_numpy(float), Y=d[outcome].to_numpy(float),
        threshold=THRESHOLD, name=name, feature_names=cols,
        description=desc,
        citation=("U.S. Census Bureau, Current Population Survey, Annual Social and Economic "
                  "Supplement, 2016-2020 public-use files; expansion dates from KFF"),
        treatment_rule=lambda Q: (np.asarray(Q) <= THRESHOLD).astype(int),
        extras={"weight": d.weight.to_numpy(float), "state": d.state.to_numpy(),
                "income_year": d.income_year.to_numpy(int),
                "medicaid": d.medicaid.to_numpy(float), "uninsured": d.uninsured.to_numpy(float),
                "any_public": d.any_public.to_numpy(float), "private": d.private.to_numpy(float),
                "noncitizen": d.noncitizen.to_numpy(float),
                "any_difficulty": d.any_difficulty.to_numpy(float),
                "fair_poor_health": d.health.isin([4, 5]).to_numpy(float),
                "hh_seq": d.hh_seq.to_numpy(int)},
    )


def load(ages=(19, 64), drop_disabled: bool = True, citizens_only: bool = False) -> RDDSample:
    """Y = Medicaid coverage at any time in the income year."""
    d = sample_frame(ages, drop_disabled, citizens_only)
    return _make(d, "medicaid", "cps_medicaid",
                 "CPS ASEC 2016-2020, adults 19-64 in full-year ACA expansion states. "
                 "Q = family income as % of the poverty threshold; treatment = 1{Q <= 138} "
                 "(income-eligible for Medicaid); Y = Medicaid coverage in the income year.")


def load_uninsured(ages=(19, 64), drop_disabled: bool = True, citizens_only: bool = False) -> RDDSample:
    """Y = uninsured for the whole income year."""
    d = sample_frame(ages, drop_disabled, citizens_only)
    return _make(d, "uninsured", "cps_uninsured",
                 "CPS ASEC 2016-2020, adults 19-64 in full-year ACA expansion states. "
                 "Q = family income as % of the poverty threshold; treatment = 1{Q <= 138}; "
                 "Y = uninsured all year.")


def mirrored(outcome: str = "medicaid", **kw):
    """(Q, X, Y, phi0, frame) with the score mirrored so that treatment is Q >= phi0:
    Q = -percent of poverty, phi0 = -138."""
    d = sample_frame(**kw)
    cols = design_columns(d)
    return -d.Q.to_numpy(float), d[cols].to_numpy(float), d[outcome].to_numpy(float), -THRESHOLD, d
