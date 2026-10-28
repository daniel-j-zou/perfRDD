"""US census tracts: the CDFI Investment Area test as an either-cutoff rule on three scores.

The CDFI Fund's Investment Area test (12 CFR 1805.201(b)(3)(ii)) makes a census tract
eligible if any one of these holds:

    poverty rate >= 20%,
    MFI ratio <= 80%, where the benchmark is the greater of the metro and national metro
        MFI for metro tracts, the greater of the statewide and national non-metro MFI
        otherwise,
    unemployment rate >= 1.5 x the national rate (5.4% in the 2016-2020 ACS),

or it lies wholly in an Empowerment Zone / Enterprise Community, or in a county that lost
at least 10% (metro) or 5% (non-metro) of its population. The last two routes are
exceptions to the score rule.

Q = (poverty rate %, MFI ratio %, unemployment ratio to the national rate), the CDFI
    Fund's published 2016-2020 ACS values on 2020 tracts, natural units. CDFI leaves a
    blank where a rate is 0 or the median is top-coded; those blanks are filled from the
    ACS summary file. The ratio is computed as rate / 5.4 in double precision, which
    reproduces CDFI's flag exactly (see ``rule_check()``).
D = 1{Q1 >= 20 or Q2 <= 80 or Q3 >= 1.5}.
X = 2020 decennial tract counts (Census Planning Database 2023, ``*_CEN_2020`` columns
    and the 2020 census self-response rate) and the OMB metro flag, plus state
    indicators. No ACS estimate enters X.
Y = load(): change in log(1 + LODES8 workplace jobs), 2022 to 2023. CDFI published the
    2016-2020 Investment Areas on January 5, 2023.
    load_placebo(): the same change 2021 to 2022 (before publication).

load() drops tracts in an Empowerment Zone / Enterprise Community or in a
population-loss county, so that the published flag equals the score rule for every
sample tract; load_all() keeps them, with the flags in extras. load_two_score() keeps
the tracts outside the exceptions that fail the income test (MFI ratio > 80 or no MFI
estimate), where the rule is 1{poverty >= 20 or unemployment ratio >= 1.5}.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample
from experiments.datasets.us_tracts_ciia.download import LODES_YEARS
from experiments.datasets.us_tracts_lic._excel import read_xlsb
from experiments.datasets.us_tracts_lic.download import STATES

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
PROCESSED = HERE / "data" / "processed"
CACHE = PROCESSED / "us_tracts_ciia.csv.gz"
CDFI_IA = RAW / "CDFI_Investment_Areas_ACS_2016_2020.xlsb"
ACS_DIR = RAW / "acs2020"
PDB = RAW / "pdb2023tr.csv"
LODES_DIR = RAW / "lodes8"

POV_CUT, MFI_CUT, UNEMP_CUT = 20.0, 80.0, 1.5
NATIONAL_UNEMP = 5.4          # 2016-2020 ACS national unemployment rate (CDFI NOTES sheet)
THRESHOLD = (POV_CUT, MFI_CUT, UNEMP_CUT)
BASE_X = ["log_pop", "log_density", "urban", "age_under5", "age_5_17", "age_18_24",
          "age_45_64", "age_65plus", "hispanic", "nh_black", "nh_asian", "nh_other",
          "inst_gq", "noninst_gq", "family_hh", "married_hh", "female_head_hh",
          "single_person_hh", "hh_with_kids", "hh_size", "renter", "vacant",
          "self_response_rate", "self_response_missing", "metro"]
STATE_FIPS = [f for f, st, _ in STATES if st != "pr"]
CITATION = ("CDFI Fund, CDFI Program Investment Areas, 2016-2020 ACS (January 2023); Census "
            "Planning Database 2023; LEHD LODES8")


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _require(*paths: Path) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{missing[0]} missing. Run `python -m experiments.datasets.us_tracts_ciia.download`.")


# --------------------------------------------------------------------------- CDFI file
def _cdfi() -> pd.DataFrame:
    """CDFI's 2016-2020 Investment Area file: one row per 2020 tract (50 states, DC and
    Puerto Rico). Blank cells are scores the ACS could not estimate."""
    rows = read_xlsb(CDFI_IA, "DATA")
    cols = ["geoid", "ia", "state_name", "county_name", "cbsa_ind", "pop_acs", "pov",
            "pov_ind", "mfi_tract", "mfi_cbsa", "mfi_metro_nat", "mfi_nonmetro_st",
            "mfi_nonmetro_nat", "mfi_ratio", "mfi_ind", "unemp", "unemp_ratio_published",
            "unemp_ind", "ez_ind", "county_pop_change", "pop_loss_ind"]
    if [str(c).strip() for c in rows[0]][:2] != ["ct2020", "ia2020"] or len(rows[0]) != len(cols):
        raise ValueError(f"unexpected header in {CDFI_IA.name}: {rows[0]}")
    df = pd.DataFrame([r + [None] * (len(cols) - len(r)) for r in rows[1:]], columns=cols)
    yes = lambda s: s.astype(str).str.strip().str.upper().str.startswith("YES")  # noqa: E731
    out = pd.DataFrame({
        "geoid": df.geoid.astype(str).str.strip().str.zfill(11),
        "state_name": df.state_name.astype(str).str.strip(),
        "county_name": df.county_name.astype(str).str.strip(),
        "metro": (df.cbsa_ind.astype(str).str.strip() == "Metro").astype(float),
        "ia_published": yes(df.ia).astype(float),
        "pov_ok_published": yes(df.pov_ind).astype(float),
        "mfi_ok_published": yes(df.mfi_ind).astype(float),
        "unemp_ok_published": yes(df.unemp_ind).astype(float),
        "empowerment_zone": yes(df.ez_ind).astype(float),
        "pop_loss": yes(df.pop_loss_ind).astype(float),
        "pop_loss_type": df.pop_loss_ind.astype(str).str.strip().where(yes(df.pop_loss_ind), ""),
    })
    for c in ["pop_acs", "pov", "mfi_tract", "mfi_cbsa", "mfi_metro_nat", "mfi_nonmetro_st",
              "mfi_nonmetro_nat", "mfi_ratio", "unemp", "unemp_ratio_published",
              "county_pop_change"]:
        out[c] = _num(df[c])
    out["state"] = out.geoid.str[:2]
    # Q3 as CDFI computed it: rate / 5.4 in double precision (8.1 / 5.4 < 1.5 in floats)
    out["unemp_ratio"] = out.unemp / NATIONAL_UNEMP
    m = out.metro == 1
    metro_b = np.fmax(out.mfi_cbsa, out.mfi_metro_nat)
    nonmetro_b = np.fmax(out.mfi_nonmetro_st, out.mfi_nonmetro_nat)
    out["benchmark_mfi"] = np.where(m, metro_b, nonmetro_b)
    out["benchmark_type"] = np.where(
        m, np.where(out.mfi_cbsa > out.mfi_metro_nat, "metro", "national_metro"),
        np.where(out.mfi_nonmetro_st > out.mfi_nonmetro_nat, "state_nonmetro",
                 "national_nonmetro"))
    out["mfi_ratio_recomputed"] = 100 * out.mfi_tract / out.benchmark_mfi
    out["exception"] = ((out.empowerment_zone == 1) | (out.pop_loss == 1)).astype(float)
    return out


# --------------------------------------------------------------------------- ACS 2016-2020
def _acs_table(table: str, cols: list) -> pd.DataFrame:
    x = pd.read_csv(ACS_DIR / f"acsdt5y2020-{table}.dat", sep="|", usecols=["GEO_ID"] + cols,
                    dtype={"GEO_ID": str})
    x = x[x.GEO_ID.str.startswith("1400000US")]                 # census tracts
    x = x.assign(geoid=x.GEO_ID.str[-11:]).drop(columns="GEO_ID")
    for c in cols:
        x[c] = _num(x[c])
    return x


def _acs() -> pd.DataFrame:
    """Tract scores recomputed from the ACS 2016-2020 table-based summary file: poverty
    100 x B17001_002 / B17001_001 and unemployment 100 x B23025_005 / B23025_003, both
    rounded half up to 0.1 (as in S1701 and S2301), and B19113 median family income.
    Negative B19113 values are the Census Bureau's missing-value codes."""
    t = _acs_table("b17001", ["B17001_E001", "B17001_E002"])
    t = t.merge(_acs_table("b23025", ["B23025_E003", "B23025_E005"]), on="geoid", how="outer")
    t = t.merge(_acs_table("b19113", ["B19113_E001"]), on="geoid", how="outer")
    rate = lambda num, den: np.floor(1000 * num / den.where(den > 0) + 0.5 + 1e-9) / 10  # noqa
    return pd.DataFrame({
        "geoid": t.geoid,
        "pov_acs": rate(t.B17001_E002, t.B17001_E001),
        "unemp_acs": rate(t.B23025_E005, t.B23025_E003),
        "mfi_acs": t.B19113_E001.where(t.B19113_E001 > 0),
    })


def _fill_from_acs(df: pd.DataFrame) -> pd.DataFrame:
    """CDFI shows a blank where the ACS value is 0, 100 or a top- or bottom-coded median.
    Fill the blanks that cannot change the rule: a 0% poverty or unemployment rate, and a
    median family income top-coded at $250,001 (its ratio, >= 250,001 / benchmark, is a
    lower bound far above 80). Blanks at 100% and at the $2,499 bottom code stay blank,
    as CDFI treats them as not qualifying."""
    for c in ["pov", "unemp", "mfi_ratio"]:
        df[f"{c}_published"] = df[c]
    df["pov_filled"] = (df.pov.isna() & (df.pov_acs == 0)).astype(float)
    df["unemp_filled"] = (df.unemp.isna() & (df.unemp_acs == 0)).astype(float)
    df["mfi_topcoded"] = (df.mfi_ratio.isna() & (df.mfi_acs >= 250001)).astype(float)
    df.loc[df.pov_filled == 1, "pov"] = 0.0
    df.loc[df.unemp_filled == 1, "unemp"] = 0.0
    top = df.mfi_topcoded == 1
    df.loc[top, "mfi_tract"] = df.loc[top, "mfi_acs"]
    df.loc[top, "mfi_ratio"] = 100 * df.loc[top, "mfi_acs"] / df.loc[top, "benchmark_mfi"]
    df["unemp_ratio"] = df.unemp / NATIONAL_UNEMP
    return df


# --------------------------------------------------------------------------- covariates
def _pdb() -> pd.DataFrame:
    """2020 decennial counts from the Census Planning Database 2023 tract file."""
    cols = ["GIDTR", "LAND_AREA", "Tot_Population_CEN_2020", "URBAN_POP_CEN_2020",
            "Pop_under_5_CEN_2020", "Pop_5_17_CEN_2020", "Pop_18_24_CEN_2020",
            "Pop_45_64_CEN_2020", "Pop_65plus_CEN_2020", "Inst_GQ_CEN_2020",
            "Non_Inst_GQ_CEN_2020", "Hispanic_CEN_2020", "NH_White_alone_CEN_2020",
            "NH_Blk_alone_CEN_2020", "NH_Asian_alone_CEN_2020", "Rel_Family_HHD_CEN_2020",
            "MrdCple_Fmly_HHD_CEN_2020", "Female_No_SP_CEN_2020", "Sngl_Prns_HHD_CEN_2020",
            "HHD_PPL_Und_18_CEN_2020", "Tot_Prns_in_HHD_CEN_2020", "Tot_Housing_Units_CEN_2020",
            "Tot_Occp_Units_CEN_2020", "Tot_Vacant_Units_CEN_2020", "Renter_Occp_HU_CEN_2020",
            "self_response_rate_cen_2020"]
    p = pd.read_csv(PDB, usecols=cols, dtype={"GIDTR": str}, encoding="latin-1")
    c = {k: _num(p[k]) for k in cols[1:]}
    pop = c["Tot_Population_CEN_2020"]
    hh = c["Tot_Occp_Units_CEN_2020"]
    pp, ph = pop.where(pop > 0), hh.where(hh > 0)
    hu = c["Tot_Housing_Units_CEN_2020"].where(c["Tot_Housing_Units_CEN_2020"] > 0)
    out = pd.DataFrame({
        "geoid": p.GIDTR.str.zfill(11), "pop2020": pop,
        "log_pop": np.log1p(pop),
        "log_density": np.log1p(pop / c["LAND_AREA"].where(c["LAND_AREA"] > 0)),
        "urban": c["URBAN_POP_CEN_2020"] / pp,
        "age_under5": c["Pop_under_5_CEN_2020"] / pp, "age_5_17": c["Pop_5_17_CEN_2020"] / pp,
        "age_18_24": c["Pop_18_24_CEN_2020"] / pp, "age_45_64": c["Pop_45_64_CEN_2020"] / pp,
        "age_65plus": c["Pop_65plus_CEN_2020"] / pp,
        "hispanic": c["Hispanic_CEN_2020"] / pp, "nh_black": c["NH_Blk_alone_CEN_2020"] / pp,
        "nh_asian": c["NH_Asian_alone_CEN_2020"] / pp,
        "nh_other": 1 - (c["Hispanic_CEN_2020"] + c["NH_White_alone_CEN_2020"]
                         + c["NH_Blk_alone_CEN_2020"] + c["NH_Asian_alone_CEN_2020"]) / pp,
        "inst_gq": c["Inst_GQ_CEN_2020"] / pp, "noninst_gq": c["Non_Inst_GQ_CEN_2020"] / pp,
        "family_hh": c["Rel_Family_HHD_CEN_2020"] / ph,
        "married_hh": c["MrdCple_Fmly_HHD_CEN_2020"] / ph,
        "female_head_hh": c["Female_No_SP_CEN_2020"] / ph,
        "single_person_hh": c["Sngl_Prns_HHD_CEN_2020"] / ph,
        "hh_with_kids": c["HHD_PPL_Und_18_CEN_2020"] / ph,
        "hh_size": c["Tot_Prns_in_HHD_CEN_2020"] / ph,
        "renter": c["Renter_Occp_HU_CEN_2020"] / ph,
        "vacant": c["Tot_Vacant_Units_CEN_2020"] / hu,
        "self_response_rate": c["self_response_rate_cen_2020"],
    })
    # no self-response rate where the 2020 census used update/leave or enumeration only
    out["self_response_missing"] = out.self_response_rate.isna().astype(float)
    out["self_response_rate"] = out.self_response_rate.fillna(out.self_response_rate.median())
    return out.drop_duplicates("geoid")


# --------------------------------------------------------------------------- outcomes
def _lodes_states() -> dict:
    return {y: {st for _, st, _ in STATES
                if (LODES_DIR / f"{st}_wac_S000_JT00_{y}.csv.gz").exists()} for y in LODES_YEARS}


def _lodes() -> pd.DataFrame:
    """Workplace jobs (C000) per 2020 tract from LODES8 (2020 blocks; Connecticut keeps
    its legacy counties, as in CDFI's file)."""
    frames = {}
    for _, st, _ in STATES:
        for y in LODES_YEARS:
            f = LODES_DIR / f"{st}_wac_S000_JT00_{y}.csv.gz"
            if f.exists():
                w = pd.read_csv(f, usecols=["w_geocode", "C000"], dtype={"w_geocode": str})
                frames.setdefault(y, []).append(w.groupby(w.w_geocode.str[:11]).C000.sum())
    out = pd.concat({y: pd.concat(v) for y, v in frames.items()}, axis=1)
    out.columns = [f"jobs_{y}" for y in out.columns]
    out.index.name = "geoid"
    return out.reset_index()


# --------------------------------------------------------------------------- build
def build() -> pd.DataFrame:
    """One row per 2020 tract in CDFI's file, with the published scores and flags, the
    benchmark, the 2020 covariates and the LODES8 outcomes."""
    if CACHE.exists():
        df = pd.read_csv(CACHE, dtype={"geoid": str, "state": str, "pop_loss_type": str,
                                       "benchmark_type": str}, float_precision="round_trip")
        df["pop_loss_type"] = df.pop_loss_type.fillna("")
        df["unemp_ratio"] = df.unemp / NATIONAL_UNEMP      # recomputed, not parsed
        return df
    _require(CDFI_IA, PDB, LODES_DIR, *[ACS_DIR / f"acsdt5y2020-{t}.dat"
                                        for t in ("b17001", "b19113", "b23025")])
    df = _cdfi().merge(_acs(), on="geoid", how="left")
    df = _fill_from_acs(df)
    df = df.merge(_pdb(), on="geoid", how="left")
    df = df.merge(_lodes(), on="geoid", how="left")
    have = _lodes_states()
    abbr = {f: st for f, st, _ in STATES}
    for y in LODES_YEARS:
        in_file = df.state.map(abbr).isin(have[y])
        df[f"jobs_{y}"] = np.where(in_file, df[f"jobs_{y}"].fillna(0.0), np.nan)
    df["dlog_jobs_22_23"] = np.log1p(df.jobs_2023) - np.log1p(df.jobs_2022)
    df["dlog_jobs_21_22"] = np.log1p(df.jobs_2022) - np.log1p(df.jobs_2021)
    df["D"] = rule(df.pov.to_numpy(), df.mfi_ratio.to_numpy(), df.unemp_ratio.to_numpy())
    PROCESSED.mkdir(parents=True, exist_ok=True)
    # atomic write: other sessions may load this dataset at the same time
    tmp = CACHE.with_name(f"{CACHE.name}.{os.getpid()}.part")
    df.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, CACHE)
    return df


def rule(pov: np.ndarray, mfi_ratio: np.ndarray, unemp_ratio: np.ndarray) -> np.ndarray:
    """The three score criteria; a missing score never qualifies (CDFI's convention)."""
    pov, mfi_ratio, ur = (np.asarray(a, float) for a in (pov, mfi_ratio, unemp_ratio))
    with np.errstate(invalid="ignore"):
        return ((pov >= POV_CUT) | (mfi_ratio <= MFI_CUT) | (ur >= UNEMP_CUT)).astype(int)


def rule_two(pov: np.ndarray, unemp_ratio: np.ndarray) -> np.ndarray:
    pov, ur = np.asarray(pov, float), np.asarray(unemp_ratio, float)
    with np.errstate(invalid="ignore"):
        return ((pov >= POV_CUT) | (ur >= UNEMP_CUT)).astype(int)


def _either_rule(Q: np.ndarray) -> np.ndarray:
    return rule(Q[:, 0], Q[:, 1], Q[:, 2])


def _either_rule_two(Q: np.ndarray) -> np.ndarray:
    return rule_two(Q[:, 0], Q[:, 1])


def rule_check() -> dict:
    """Agreement of the rule with CDFI's flags over all tracts in CDFI's file, and of
    CDFI's scores with the ACS summary file. The scores are the published ones with the
    blanks filled as in ``_fill_from_acs``."""
    df = build()
    d = rule(df.pov, df.mfi_ratio, df.unemp_ratio)
    with np.errstate(invalid="ignore"):
        exact_unemp = (df.unemp >= UNEMP_CUT * NATIONAL_UNEMP - 1e-9).astype(float)
    exc = df.exception == 1
    pub = df.ia_published
    bp, bu, bm = (df.pov_published.notna() & df.pov_acs.notna(),
                  df.unemp_published.notna() & df.unemp_acs.notna(),
                  df.mfi_ratio_published.notna())
    blank = lambda c, acs: df[c].isna() & df[acs].notna()                     # noqa: E731
    return {
        "tracts": int(len(df)),
        "rule_or_exceptions_vs_cdfi_flag": float(np.mean((d | exc) == pub)),
        "rule_vs_cdfi_flag": float(np.mean(d == pub)),
        "cdfi_flag_only_through_exception": int(((d == 0) & (pub == 1)).sum()),
        "rule_eligible_not_flagged": int(((d == 1) & (pub == 0)).sum()),
        "exception_tracts": int(exc.sum()),
        "empowerment_zone": int(df.empowerment_zone.sum()),
        "pop_loss": df.pop_loss_type[df.pop_loss == 1].value_counts().to_dict(),
        "rule_vs_cdfi_flag_outside_exceptions": float(np.mean(d[~exc] == pub[~exc])),
        "pov_flag_vs_pov_ge_20": float(np.mean((df.pov >= POV_CUT) == df.pov_ok_published)),
        "mfi_flag_vs_ratio_le_80": float(np.mean((df.mfi_ratio <= MFI_CUT) == df.mfi_ok_published)),
        "unemp_flag_vs_ratio_ge_1.5_double": float(np.mean(
            (df.unemp_ratio >= UNEMP_CUT) == df.unemp_ok_published)),
        "unemp_flag_vs_rate_ge_8.1_exact": float(np.mean(exact_unemp == df.unemp_ok_published)),
        "tracts_at_8.1_flagged_no": int(((df.unemp.round(1) == 8.1)
                                         & (df.unemp_ok_published == 0)).sum()),
        "unemp_ratio_published_max_abs_diff": float(np.nanmax(np.abs(
            df.unemp_ratio_published - df.unemp_ratio))),
        "mfi_ratio_recomputed_max_abs_diff": float(np.nanmax(np.abs(
            df.mfi_ratio_recomputed - df.mfi_ratio))),
        "benchmark_type_counts": df.loc[df.mfi_ratio.notna(), "benchmark_type"]
                                   .value_counts().to_dict(),
        "acs_pov_equal_to_cdfi": float(np.mean(np.abs(df.pov_published - df.pov_acs)[bp] < 0.05)),
        "acs_unemp_equal_to_cdfi": float(np.mean(
            np.abs(df.unemp_published - df.unemp_acs)[bu] < 0.05)),
        "acs_mfi_equal_to_cdfi": float(np.mean((df.mfi_tract == df.mfi_acs)[bm])),
        "blank_filled": {"pov_0": int(df.pov_filled.sum()), "unemp_0": int(df.unemp_filled.sum()),
                         "mfi_topcoded_250001": int(df.mfi_topcoded.sum())},
        "blank_kept": {"pov_100": int(blank("pov", "pov_acs").sum()),
                       "unemp_100": int(blank("unemp", "unemp_acs").sum()),
                       "mfi_bottomcoded_2499": int((df.mfi_ratio.isna()
                                                    & (df.mfi_acs <= 2499)).sum())},
        "missing_scores_after_fill": {c: int(df[c].isna().sum())
                                      for c in ["pov", "mfi_ratio", "unemp"]},
    }


# --------------------------------------------------------------------------- samples
def _frame(y_col: str, keep_exceptions: bool,
           scores=("pov", "mfi_ratio", "unemp_ratio")) -> pd.DataFrame:
    df = build()
    df = df[df.state.isin(STATE_FIPS)]
    if not keep_exceptions:
        df = df[df.exception == 0]
    keep = df[list(scores)].notna().all(axis=1) & df[BASE_X].notna().all(axis=1) \
        & df[y_col].notna()
    return df[keep]


def _design(df: pd.DataFrame):
    """Covariates that vary in the sample, plus indicators for the states present (the
    first is the base). The self-response flag is constant once Alaska drops out."""
    states = sorted(df.state.unique())
    dummies = {f"st_{f}": (df.state == f).astype(float) for f in states[1:]}
    df = df.assign(**dummies)
    x_cols = [c for c in BASE_X if df[c].nunique() > 1] + list(dummies)
    return df, x_cols


def _extras(df: pd.DataFrame) -> dict:
    return {
        "geoid": df.geoid.to_numpy(), "state": df.state.to_numpy(),
        "eligible_published": df.ia_published.to_numpy(float),
        "pov_ok_published": df.pov_ok_published.to_numpy(float),
        "mfi_ok_published": df.mfi_ok_published.to_numpy(float),
        "unemp_ok_published": df.unemp_ok_published.to_numpy(float),
        "empowerment_zone": df.empowerment_zone.to_numpy(float),
        "pop_loss": df.pop_loss.to_numpy(float),
        "pop_loss_type": df.pop_loss_type.to_numpy(),
        "county_pop_change": df.county_pop_change.to_numpy(float),
        "exception": df.exception.to_numpy(float),
        "metro": df.metro.to_numpy(float),
        "benchmark_type": df.benchmark_type.to_numpy(),
        "benchmark_mfi": df.benchmark_mfi.to_numpy(float),
        "mfi_tract": df.mfi_tract.to_numpy(float),
        "unemp_rate": df.unemp.to_numpy(float),
        "mfi_ratio": df.mfi_ratio.to_numpy(float),
        "mfi_topcoded": df.mfi_topcoded.to_numpy(float),
        "pov_filled": df.pov_filled.to_numpy(float),
        "unemp_filled": df.unemp_filled.to_numpy(float),
        "pop_acs": df.pop_acs.to_numpy(float),
        "jobs": df[[f"jobs_{y}" for y in LODES_YEARS]].to_numpy(float),
        "jobs_years": LODES_YEARS,
    }


def _sample(y_col: str, label: str, keep_exceptions: bool = False,
            suffix: str = "") -> RDDSample:
    df, x_cols = _design(_frame(y_col, keep_exceptions))
    scope = ("all tracts, including Empowerment Zone and population-loss tracts"
             if keep_exceptions else
             "excluding Empowerment Zone / Enterprise Community and population-loss tracts")
    return RDDSample(
        Q=df[["pov", "mfi_ratio", "unemp_ratio"]].to_numpy(float),
        X=df[x_cols].to_numpy(float), Y=df[y_col].to_numpy(float), threshold=THRESHOLD,
        name=f"us_tracts_ciia{suffix}_{y_col}", feature_names=x_cols,
        description=("US 2020 census tracts (50 states + DC), " + scope + ". Q = (poverty "
                     "rate %, MFI ratio %, unemployment ratio to the 5.4% national rate), ACS "
                     "2016-2020 as published by the CDFI Fund; treatment = 1{Q1 >= 20 or "
                     "Q2 <= 80 or Q3 >= 1.5} (CDFI Investment Area); X = 2020 decennial; "
                     f"Y = {label}."),
        citation=CITATION, treatment_rule=_either_rule, extras=_extras(df),
    )


def load() -> RDDSample:
    return _sample("dlog_jobs_22_23", "change in log(1 + workplace jobs), LODES8 2022 to 2023")


def load_placebo() -> RDDSample:
    return _sample("dlog_jobs_21_22", "change in log(1 + workplace jobs), 2021 to 2022 (placebo)")


def load_all() -> RDDSample:
    """As load(), keeping the Empowerment Zone and population-loss tracts. D is the score
    rule; extras['eligible_published'] is CDFI's flag, which also counts the exceptions."""
    return _sample("dlog_jobs_22_23", "change in log(1 + workplace jobs), LODES8 2022 to 2023",
                   keep_exceptions=True, suffix="_all")


def load_two_score(y_col: str = "dlog_jobs_22_23") -> RDDSample:
    """Tracts outside the exceptions that fail the income test: MFI ratio > 80, or no
    MFI estimate (CDFI's convention: a blank never qualifies). There the rule is
    1{poverty >= 20 or unemployment ratio >= 1.5}. The restriction conditions on the MFI
    ratio, a noisy score; see README."""
    df = _frame(y_col, keep_exceptions=False, scores=("pov", "unemp_ratio"))
    df, x_cols = _design(df[~(df.mfi_ratio <= MFI_CUT)])
    label = {"dlog_jobs_22_23": "change in log(1 + workplace jobs), LODES8 2022 to 2023",
             "dlog_jobs_21_22": "change in log(1 + workplace jobs), 2021 to 2022 (placebo)"
             }.get(y_col, y_col)
    return RDDSample(
        Q=df[["pov", "unemp_ratio"]].to_numpy(float),
        X=df[x_cols].to_numpy(float), Y=df[y_col].to_numpy(float),
        threshold=(POV_CUT, UNEMP_CUT),
        name=f"us_tracts_ciia_two_score_{y_col}", feature_names=x_cols,
        description=("US 2020 census tracts (50 states + DC) that fail the CDFI income test "
                     "(MFI ratio > 80 or not estimated), outside Empowerment Zones and population-loss "
                     "counties. Q = (poverty rate %, unemployment ratio to the 5.4% national "
                     "rate), ACS 2016-2020 as published by the CDFI Fund; treatment = "
                     "1{Q1 >= 20 or Q2 >= 1.5} (CDFI Investment Area); X = 2020 decennial; "
                     f"Y = {label}."),
        citation=CITATION, treatment_rule=_either_rule_two, extras=_extras(df),
    )
