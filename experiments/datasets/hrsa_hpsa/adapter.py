"""HRSA Health Professional Shortage Areas (HPSAs): an integer points score, NHSC funding cutoff by year.

HRSA scores every designated HPSA. A primary care score runs from 0 to 25 and is a fixed-weight
sum of four component scales (FR 68:32531, 30 May 2003, and the reconstruction below):

    score = 2 x ratio scale (0-5)          population per FTE primary care physician
          + poverty points (0-5)           percent of the population below 100% of poverty
          + infant health points (0-5)     infant mortality or low birth weight
          + travel points (0-5)            time or distance to the nearest source of care

The National Health Service Corps (NHSC) funds loan repayors (LRP) and places scholars (SP) at
sites in HPSAs, in descending order of HPSA score. Each year this produces a lowest funded score.

Q = the primary care HPSA score of the current snapshot (integer, 0-25). Most designated HPSAs
    were rescored in HRSA's September 2025 national update (``extras["score_date"]``).
D = 1{Q >= c}, with c = the effective NHSC LRP cutoff of the chosen fiscal year, derived from
    HRSA's "NHSC LRP Program Applicant Metrics" (award rate by site HPSA score); see
    ``LRP_CUTOFFS`` and ``lrp_cutoff_check()``. The default is FY 2025, c = 13. Because Q is an
    integer, the threshold is stored as c - 0.5.
X = HPSA characteristics that are not score inputs (designation and population type, metro and
    rural status, component geography, designation population and age, US-Mexico border) and
    2023-24 county characteristics from the Area Health Resources File (AHRF 2024-2025),
    aggregated over the HPSA's counties, plus state indicators.
Y = load(): a PLACEHOLDER, not an outcome: NHSC primary care FTE clinicians per 10,000
    residents in the HPSA's counties in 2024 (AHRF), which predates the 2025 scores. No public
    HRSA file reports NHSC clinicians by HPSA, and no county file yet covers a year after the
    September 2025 rescoring. ``load_sites_2026()`` uses NHSC-approved sites per 10,000
    residents in October 2026 (Health Sites file), which is later but county-level and counts
    approved sites rather than clinicians.

Sample: Designated or Proposed-for-Withdrawal primary care HPSAs of the geographic,
high-needs geographic, population, FQHC, FQHC look-alike, rural health clinic and Indian
Health Service/tribal/urban Indian types, with an AHRF county match. Excluded: withdrawn HPSAs
(their score is no longer in force), correctional facilities (scored from degree-of-shortage
groups as 21/15/9, not from the components) and "Other Facility" (11 HPSAs).
"""
from __future__ import annotations

import os
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample
from experiments.datasets.fl_school_grades._spreadsheet import read, sheet_names
from experiments.datasets.hrsa_hpsa.download import FR_NOTICES, HPSA_FILES, LRP_YEARS, RAW

HERE = Path(__file__).parent
PROCESSED = HERE / "data" / "processed"
CACHE_VERSION = 1
SNAPSHOT = pd.Timestamp("2026-10-26")      # "Data Warehouse Record Create Date" of the HPSA files
AHRF_ZIP = RAW / "AHRF_2024-2025_CSV.zip"
AHRF_PREFIX = "NCHWA-2024-2025+AHRF+COUNTY+CSV/AHRF2025"

# Effective NHSC LRP cutoff by fiscal year: the lowest site HPSA score from which every score
# with at least 20 traditional-LRP applicants has an award rate of at least 25%, provided the
# pooled award rate below it is under 25% (at least 50 applicants). Derived from the applicant
# metrics by lrp_cutoff_check(). The tables pool primary care, dental and mental health sites.
# FY 2021 and FY 2022 (American Rescue Plan funding) have no binding cutoff.
LRP_CUTOFFS = {2020: 16, 2021: None, 2022: None, 2023: 16, 2024: 17, 2025: 13}
MAIN_FY = 2025

# HRSA notices in the Federal Register that set the HPSA scores of greatest shortage. "sp_pc" is
# the primary care physician threshold for NHSC scholar placement, "sp_mh" psychiatrists, "sp_dh"
# dentists, "lrp" the loan-repayment priority score (all disciplines). None = not in the notice.
FR_THRESHOLDS = [
    # doc number, FR citation, period, sp_pc, sp_mh, sp_dh, lrp
    ("03-30820", "68 FR 69709 (2003-12-15)", "2003-04 placements", 14, 18, 18, 14),
    ("04-18409", "69 FR 49906 (2004-08-12)", "2004-05 placements", 14, 20, 20, 14),
    ("05-17180", "70 FR 51356 (2005-08-30)", "2005-06 placements", 14, 20, 20, 14),
    ("E6-9974", "71 FR 36348 (2006-06-26)", "2006-07 placements", 14, 19, 19, 17),
    ("E7-11423", "72 FR 32853 (2007-06-14)", "2007-08 placements", 14, 19, 18, 17),
    ("E8-13454", "73 FR 34025 (2008-06-16)", "2008-09 placements", 14, 19, 17, 17),
    ("E9-12531", "74 FR 25568 (2009-05-28)", "FY 2010 LRP awards", None, None, None, 10),
    ("E9-14741", "74 FR 29708 (2009-06-23)", "2009-10 placements", 17, 17, 17, None),
    ("2010-15356", "75 FR 36102 (2010-06-24)", "2010-11 placements", 17, 17, 17, None),
    ("2010-28083", "75 FR 68368 (2010-11-05)", "FY 2011 LRP awards", None, None, None, 10),
    ("2011-19505", "76 FR 46305 (2011-08-02)", "2011-12 placements", 16, 16, 16, None),
    ("2012-1844", "77 FR 4332 (2012-01-27)", "FY 2012 LRP awards", None, None, None, 14),
    ("2012-8928", "77 FR 22329 (2012-04-13)", "2012-13 placements", 16, 16, 16, None),
    ("2013-15112", "78 FR 38061 (2013-06-25)", "2013-14 placements", 16, 16, 16, None),
    ("2014-08264", "79 FR 20890 (2014-04-14)", "2014-15 placements", 14, 14, 14, None),
    ("2015-11049", "80 FR 26569 (2015-05-08)", "2015-16 placements", 16, 16, 16, None),
    ("2016-10527", "81 FR 27143 (2016-05-05)", "2016-17 placements", 17, 17, 17, None),
    ("2017-09024", "82 FR 20897 (2017-05-04)", "2017-18 placements", 17, 17, 17, None),
    ("2018-10699", "83 FR 23470 (2018-05-21)", "2018-19 placements", 18, 18, 18, None),
]

AREA_TYPES = {"Geographic HPSA": "geographic", "High Needs Geographic HPSA": "high_needs",
              "HPSA Population": "population"}
FACILITY_TYPES = {"Federally Qualified Health Center": "fqhc",
                  "Federally Qualified Health Center Look A Like": "fqhc_lal",
                  "Rural Health Clinic": "rhc",
                  "Indian Health Service, Tribal Health, and Urban Indian Health Organizations": "itu"}
TYPES = {**AREA_TYPES, **FACILITY_TYPES}
NHSC_FTE = {"PC": "nhsc_fte_prim_care_provdrs", "DH": "nhsc_fte_dent_provdrs",
            "MH": "nhsc_fte_mentl_hlth_provdr"}
Y_PLACEHOLDER = "y_placeholder_pre_nhsc_fte_per10k_2024"
Y_SITES = "y_nhsc_sites_per10k_2026"

_COLS = {
    "HPSA ID": "hpsa_id", "HPSA Name": "name", "Designation Type": "desig_type",
    "HPSA Score": "score", "PC MCTA Score": "mcta", "HPSA Status": "status",
    "HPSA Designation Date": "designated", "HPSA Designation Last Update Date": "score_date",
    "Withdrawn Date": "withdrawn", "Metropolitan Indicator": "metro",
    "HPSA Degree of Shortage": "dos", "HPSA FTE": "fte",
    "HPSA Designation Population": "desig_pop",
    "% of Population Below 100% Poverty": "pct_poverty", "HPSA Formal Ratio": "formal_ratio",
    "HPSA Population Type": "pop_type", "Primary State Abbreviation": "state",
    "HPSA Estimated Underserved Population": "underserved_pop",
    "HPSA Provider Ratio Goal": "ratio_goal", "HPSA Shortage": "shortage",
}


# ------------------------------------------------------------------------ raw files

def _need(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run python -m experiments.datasets.hrsa_hpsa.download")
    return path


def _hpsa_rows(disc: str) -> pd.DataFrame:
    """One row per HPSA component, as published."""
    df = pd.read_csv(_need(RAW / HPSA_FILES[disc]), dtype=str, keep_default_na=False,
                     encoding_errors="replace")
    return df.loc[:, ~df.columns.str.startswith("Unnamed")]


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.strip().replace({"": None, "NA": None}),
                         errors="coerce")


def _hpsa_frame(disc: str) -> pd.DataFrame:
    """One row per HPSA (all statuses): the HPSA-level fields, component geography and the
    HPSA's counties."""
    rows = _hpsa_rows(disc)
    rural = rows["Rural Status"].map({"Rural": 1.0, "Partially Rural": 0.5, "Non-Rural": 0.0})
    g = rows.assign(_rural=rural).groupby("HPSA ID", sort=False)
    comp = pd.DataFrame({
        "n_components": g.size(),
        "rural_share": g["_rural"].mean(),
        "border": g["U.S. - Mexico Border County Indicator"].agg(lambda s: float((s == "Y").any())),
        "comp_type": g["HPSA Component Type Description"].agg(lambda s: s.mode().iat[0]),
        "counties": g["State and County Federal Information Processing Standard Code"].agg(
            lambda s: ";".join(sorted({v for v in s if re.fullmatch(r"\d{5}", v)}))),
        "score_conflict": g["HPSA Score"].agg(lambda s: float(s.nunique() > 1)),
    })
    cols = {k: v for k, v in _COLS.items() if k in rows}
    h = rows.drop_duplicates("HPSA ID")[list(cols)].rename(columns=cols).set_index("hpsa_id")
    h = h.join(comp).reset_index()
    for c in ["score", "fte", "desig_pop", "pct_poverty", "underserved_pop", "mcta"]:
        if c in h:
            h[c] = _num(h[c])
    for c in ["designated", "score_date", "withdrawn"]:
        h[c] = pd.to_datetime(h[c].replace("", None), format="%m/%d/%Y", errors="coerce")
    r = h.formal_ratio.str.extract(r"^\s*(\d+):(\d+)\s*$").astype(float)
    h["ratio"] = r[0] / r[1]
    h["discipline"] = disc
    if disc == "PC":
        h = h.join(_pc_components(h))
    return h


def _pc_components(h: pd.DataFrame) -> pd.DataFrame:
    """Primary care ratio and poverty points from the published inputs (FR 68:32531).

    Ratio scale (population per FTE primary care physician): >= 10,000 -> 5, >= 5,000 -> 4,
    >= 4,000 -> 3, >= 3,500 -> 2, >= 3,000 -> 1; with no physicians, population >= 2,500 / 2,000
    / 1,500 / 1,000 / 500 -> 5 / 4 / 3 / 2 / 1. The ratio scale is double-weighted. Poverty
    points: >= 50% -> 5, >= 40 -> 4, >= 30 -> 3, >= 20 -> 2, >= 15 -> 1. The rest of the score
    (infant health and travel, 0-10) is ``other_pts``. Only for geographic and population HPSAs.
    """
    area = h.desig_type.isin(list(AREA_TYPES))
    fte, pop, pov = h.fte.to_numpy(float), h.desig_pop.to_numpy(float), h.pct_poverty.to_numpy(float)
    ratio = np.where(fte > 0, pop / np.where(fte > 0, fte, 1.0), np.nan)
    by_ratio = np.select([ratio >= 10000, ratio >= 5000, ratio >= 4000, ratio >= 3500,
                          ratio >= 3000], [5, 4, 3, 2, 1], 0)
    by_pop = np.select([pop >= 2500, pop >= 2000, pop >= 1500, pop >= 1000, pop >= 500],
                       [5, 4, 3, 2, 1], 0)
    rscale = np.where(fte > 0, by_ratio, by_pop).astype(float)
    ppts = np.select([pov >= 50, pov >= 40, pov >= 30, pov >= 20, pov >= 15],
                     [5, 4, 3, 2, 1], 0).astype(float)
    ok = area & np.isfinite(fte) & np.isfinite(pop) & np.isfinite(pov)
    out = pd.DataFrame({"ratio_pts": np.where(ok, 2 * rscale, np.nan),
                        "poverty_pts": np.where(ok, ppts, np.nan)}, index=h.index)
    out["other_pts"] = h.score - out.ratio_pts - out.poverty_pts
    return out


def _ahrf() -> pd.DataFrame:
    """County file: the AHRF 2024-2025 columns used for X and the placeholder outcome."""
    want = {
        "geo": ["rural_urban_contnm_23"],
        "env": ["land_area_mi2_20"],
        "pop": ["popn_est_23", "popn_est_ge65_23", "pers_povty_23", "pers_povty_detrmnd_23",
                "medn_famly_incom_23", "noins_incpovlt400_lt65_pct_22"],
        "hp": ["phys_nf_prim_care_pc_exc_rsdt_23", "np_npi_24", "pa_npi_24"],
        "hf": [f"{v}_{y}" for v in NHSC_FTE.values() for y in (23, 24)] + ["nhsc_sites_24"],
    }
    z = zipfile.ZipFile(_need(AHRF_ZIP))
    out = None
    for part, cols in want.items():
        d = pd.read_csv(z.open(f"{AHRF_PREFIX}{part}.csv"), usecols=["fips_st_cnty"] + cols,
                        dtype=str, encoding="latin-1", keep_default_na=False)
        d = d.set_index("fips_st_cnty").apply(_num)
        out = d if out is None else out.join(d)
    return out


def _nhsc_sites() -> pd.Series:
    """NHSC-approved sites by county in the Health Sites file (October 2026 snapshot)."""
    s = pd.read_csv(_need(RAW / "SITE_DASHBOARD.csv"), dtype=str, keep_default_na=False,
                    encoding="latin-1", usecols=["Site Category", "State County FIPS Code"])
    s = s[s["Site Category"] == "National Health Service Corps"]
    return s.groupby("State County FIPS Code").size().rename("nhsc_sites_2026")


def _county_aggregates(h: pd.DataFrame, disc: str) -> pd.DataFrame:
    """AHRF covariates and NHSC measures over each HPSA's counties: counts are summed, rates
    and codes population-weighted."""
    ahrf = _ahrf().join(_nhsc_sites(), how="left")
    ahrf["nhsc_sites_2026"] = ahrf.nhsc_sites_2026.fillna(0.0)
    long = (h[["hpsa_id", "counties"]].assign(fips=h.counties.str.split(";"))
            .explode("fips").drop(columns="counties"))
    long = long[long.fips.notna() & (long.fips != "")]
    long = long.join(ahrf, on="fips", how="inner")
    w = long.popn_est_23.fillna(0.0)
    for c in ["rural_urban_contnm_23", "medn_famly_incom_23", "noins_incpovlt400_lt65_pct_22"]:
        long[f"w_{c}"] = long[c] * w
        long[f"wt_{c}"] = w.where(long[c].notna(), 0.0)
    g = long.groupby("hpsa_id")
    s = g.sum(min_count=1, numeric_only=True)
    fte = NHSC_FTE[disc]
    a = pd.DataFrame(index=s.index)
    a["n_counties"] = g.size()
    a["cty_pop"] = s.popn_est_23
    a["cty_log_pop"] = np.log(s.popn_est_23)
    a["cty_log_density"] = np.log(s.popn_est_23 / s.land_area_mi2_20)
    a["cty_pov_pct"] = 100 * s.pers_povty_23 / s.pers_povty_detrmnd_23
    a["cty_pct65"] = 100 * s.popn_est_ge65_23 / s.popn_est_23
    a["cty_log_pcp_per10k"] = np.log1p(1e4 * s.phys_nf_prim_care_pc_exc_rsdt_23 / s.popn_est_23)
    a["cty_log_np_per10k"] = np.log1p(1e4 * s.np_npi_24 / s.popn_est_23)
    a["cty_log_pa_per10k"] = np.log1p(1e4 * s.pa_npi_24 / s.popn_est_23)
    a["cty_rucc"] = s.w_rural_urban_contnm_23 / s.wt_rural_urban_contnm_23
    a["cty_log_income"] = np.log(s.w_medn_famly_incom_23 / s.wt_medn_famly_incom_23)
    a["cty_uninsured_pct"] = s.w_noins_incpovlt400_lt65_pct_22 / s.wt_noins_incpovlt400_lt65_pct_22
    a["y_pre_nhsc_fte_per10k_2023"] = 1e4 * s[f"{fte}_23"] / s.popn_est_23
    a[Y_PLACEHOLDER] = 1e4 * s[f"{fte}_24"] / s.popn_est_23
    a["nhsc_sites_per10k_2024"] = 1e4 * s.nhsc_sites_24 / s.popn_est_23
    a[Y_SITES] = 1e4 * s.nhsc_sites_2026 / s.popn_est_23
    return a.replace([np.inf, -np.inf], np.nan)


def build(disc: str = "PC") -> pd.DataFrame:
    """Every HPSA of the discipline (all statuses and types) with county covariates and the
    NHSC measures. Cached in data/processed/; bump CACHE_VERSION after changing the build (other
    sessions may run the registry with an older copy of this module), or delete the cache."""
    cache = PROCESSED / f"hrsa_hpsa_{disc}_v{CACHE_VERSION}.csv.gz"
    if not cache.exists():
        h = _hpsa_frame(disc)
        h = h.join(_county_aggregates(h, disc), on="hpsa_id")
        PROCESSED.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_name(f"{cache.name}.{os.getpid()}.part")
        h.to_csv(tmp, index=False, compression="gzip")
        os.replace(tmp, cache)
    h = pd.read_csv(cache, dtype={"hpsa_id": str, "counties": str}, keep_default_na=False,
                    na_values=[""])
    for c in ["designated", "score_date", "withdrawn"]:
        h[c] = pd.to_datetime(h[c])
    for c in ["name", "desig_type", "status", "metro", "pop_type", "state", "comp_type", "counties"]:
        h[c] = h[c].fillna("").astype(str)
    return h


# --------------------------------------------------------------------------- cutoffs

def lrp_award_table(fy: int) -> pd.DataFrame:
    """Traditional NHSC LRP applicants and awards by site HPSA score for one fiscal year
    ("Applicant Site Information" sheet of the FY applicant metrics)."""
    path = _need(RAW / "nhsc" / f"FY{fy}_NHSC_LRP_Program_Applicant_Metrics.xlsx")
    names = sheet_names(path)
    rows = read(path, names.index("Applicant Site Information"))
    k = next(i for i, r in enumerate(rows) if r and r[0] == "By Site HPSA Score")
    out = []
    for r in rows[k + 1:]:
        if not r or not isinstance(r[0], float):
            break
        out.append({"score": int(r[0]), "submitted": float(r[1]), "awarded": float(r[2])})
    t = pd.DataFrame(out)
    t["rate"] = t.awarded / t.submitted.where(t.submitted > 0)
    t.insert(0, "fy", fy)
    return t


def effective_cutoff(t: pd.DataFrame, min_rate: float = 0.25, min_n: int = 20,
                     min_below: int = 50) -> int | None:
    """Lowest score from which every score with >= min_n applicants has an award rate of at
    least min_rate; None when the pooled rate below it is not under min_rate (no binding cutoff)."""
    big = t[t.submitted >= min_n].sort_values("score", ascending=False)
    c = None
    for s, rate in zip(big.score, big.rate):
        if rate < min_rate:
            break
        c = s
    below = t[t.score < c] if c is not None else t.iloc[:0]
    if c is None or below.submitted.sum() < min_below:
        return None
    return int(c) if below.awarded.sum() / below.submitted.sum() < min_rate else None


def lrp_cutoff_check() -> pd.DataFrame:
    """Award rates around the derived cutoff, by fiscal year, and agreement with LRP_CUTOFFS."""
    rows = []
    for fy in LRP_YEARS:
        t = lrp_award_table(fy)
        c = effective_cutoff(t)
        r = dict(zip(t.score, t.rate))
        above = t[t.score >= c] if c else t
        below = t[t.score < c] if c else t.iloc[:0]
        rows.append({
            "fy": fy, "cutoff": c, "matches_LRP_CUTOFFS": c == LRP_CUTOFFS[fy],
            "submitted": int(t.submitted.sum()), "awarded": int(t.awarded.sum()),
            "rate_at_c": r.get(c), "rate_at_c_minus_1": r.get(c - 1) if c else None,
            "rate_at_or_above": above.awarded.sum() / above.submitted.sum(),
            "rate_below": (below.awarded.sum() / below.submitted.sum()) if len(below) else None,
        })
    return pd.DataFrame(rows)


def _fr_text(doc: str) -> str:
    t = _need(RAW / "fr" / f"{doc}.txt").read_text(encoding="utf-8", errors="replace")
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", t))


def fr_check() -> pd.DataFrame:
    """Re-read each threshold in FR_THRESHOLDS from the notice text."""
    pat = {
        "sp_pc": r"\(1\) Primary (?:medical )?care HPSAs with (?:a )?scores? of (\d+) and above",
        "sp_mh": r"[Mm]ental health HPSAs with (?:a )?scores? of (\d+) and above",
        "sp_dh": r"[Dd]ental HPSAs with (?:a )?scores? of (\d+) and above",
        "lrp": r"HPSAs \(appropriate to each discipline\) with scores of (\d+) and above",
    }
    rows = []
    for doc, cite, period, *vals in FR_THRESHOLDS:
        t = _fr_text(doc)
        found = {k: (int(m.group(1)) if (m := re.search(p, t)) else None) for k, p in pat.items()}
        rows.append({"doc": doc, "cite": cite, "period": period,
                     **{k: v for k, v in zip(pat, vals)},
                     "agree": all(found[k] == v for k, v in zip(pat, vals))})
    return pd.DataFrame(rows)


def component_check() -> pd.DataFrame:
    """For non-withdrawn geographic and population HPSAs: share whose score minus the
    reconstructed ratio and poverty points lies in [0, 10] (the range of infant health plus
    travel points), under double and single weights."""
    h = build("PC")
    h = h[(h.status != "Withdrawn") & h.desig_type.isin(list(AREA_TYPES)) & h.ratio_pts.notna()]
    rows = []
    for wr, wp in [(2, 1), (2, 2), (1, 1)]:
        res = h.score - wr / 2 * h.ratio_pts - wp * h.poverty_pts
        rows.append({"ratio_weight": wr, "poverty_weight": wp, "n": len(h),
                     "share_in_0_10": float(((res >= 0) & (res <= 10)).mean()),
                     "min": float(res.min()), "max": float(res.max())})
    return pd.DataFrame(rows)


# -------------------------------------------------------------------------- samples

def _frame(disc: str, types: dict, y_col: str) -> pd.DataFrame:
    h = build(disc)
    m = (h.status.isin(["Designated", "Proposed For Withdrawal"]) & h.desig_type.isin(list(types))
         & h.score.notna() & h.cty_log_pop.notna() & h[y_col].notna())
    return h[m].reset_index(drop=True)


def _design(df: pd.DataFrame, with_mcta: bool = False) -> tuple[np.ndarray, list]:
    X = pd.DataFrame(index=df.index)
    kind = df.desig_type.map(TYPES)
    for k in ["high_needs", "population", "fqhc", "fqhc_lal", "rhc", "itu"]:
        X[f"type_{k}"] = (kind == k).astype(float)          # base: geographic HPSA
    pt = df.pop_type
    X["pop_medicaid"] = pt.str.contains("Medicaid").astype(float)
    X["pop_migrant"] = pt.str.contains("Migrant|Seasonal").astype(float)
    X["pop_homeless"] = pt.str.contains("Homeless").astype(float)
    X["pop_native_american"] = pt.str.contains("Native American").astype(float)
    X["pop_other"] = pt.str.contains("Other Population").astype(float)
    X["nonmetro"] = (df.metro == "Non-Metropolitan").astype(float)
    X["frontier"] = (df.metro == "Frontier").astype(float)
    X["metro_unknown"] = (df.metro == "Unknown").astype(float)
    X["rural_share"] = df.rural_share.fillna(df.rural_share.median())
    X["rural_missing"] = df.rural_share.isna().astype(float)
    X["comp_csd"] = (df.comp_type == "County Subdivision").astype(float)
    X["comp_county"] = (df.comp_type == "Single County").astype(float)
    X["log_n_components"] = np.log(df.n_components)
    X["log_desig_pop"] = np.log1p(df.desig_pop.fillna(0.0))
    X["years_designated"] = (SNAPSHOT - df.designated).dt.days / 365.25
    X["border"] = df.border
    X["log_n_counties"] = np.log(df.n_counties)
    for c in ["cty_log_pop", "cty_log_density", "cty_pov_pct", "cty_pct65", "cty_log_pcp_per10k",
              "cty_log_np_per10k", "cty_log_pa_per10k", "cty_rucc", "cty_log_income",
              "cty_uninsured_pct"]:
        X[c] = df[c].fillna(df[c].median())
    if with_mcta:
        X["mcta"] = df.mcta.fillna(df.mcta.median())
        X["mcta_missing"] = df.mcta.isna().astype(float)
    for s in sorted(df.state.unique()):
        if s != "CA":                                       # California is the base
            X[f"state_{s}"] = (df.state == s).astype(float)
    X = X.loc[:, X.std() > 0]
    return X.to_numpy(float), list(X.columns)


def _extras(df: pd.DataFrame, fy: int, c: int, y_col: str) -> dict:
    return {
        "fy": fy, "cutoff": c - 0.5, "cutoff_score": c, "lrp_cutoffs": LRP_CUTOFFS,
        "fr_thresholds": FR_THRESHOLDS, "y_name": y_col,
        "group": df.desig_type.map(TYPES).to_numpy(str),
        "hpsa_id": df.hpsa_id.to_numpy(str), "state": df.state.to_numpy(str),
        "status": df.status.to_numpy(str), "score_date": df.score_date.to_numpy(),
        "rescored_2025_09": ((df.score_date >= "2025-09-01")
                             & (df.score_date < "2025-10-01")).to_numpy(float),
        "proposed_withdrawal": (df.status == "Proposed For Withdrawal").to_numpy(float),
        "counties": df.counties.to_numpy(str), "mcta": df.mcta.to_numpy(float)
        if "mcta" in df else None,
        "y_pre_2023": df.y_pre_nhsc_fte_per10k_2023.to_numpy(float),
        "y_pre_2024": df[Y_PLACEHOLDER].to_numpy(float),
        "y_sites_2026": df[Y_SITES].to_numpy(float),
    }


def _sample(disc: str = "PC", fy: int = MAIN_FY, y_col: str = Y_PLACEHOLDER,
            types: dict = TYPES, with_mcta: bool = False) -> RDDSample:
    c = LRP_CUTOFFS[fy]
    if c is None:
        raise ValueError(f"FY {fy} has no binding LRP cutoff (award rates flat in the score)")
    df = _frame(disc, types, y_col)
    X, names = _design(df, with_mcta)
    label = {"PC": "primary care", "DH": "dental", "MH": "mental health"}[disc]
    return RDDSample(
        Q=df.score.to_numpy(float), X=X, Y=df[y_col].to_numpy(float), threshold=c - 0.5,
        name=f"hrsa_hpsa_{disc}_fy{fy}_{y_col}", feature_names=names,
        description=(f"HRSA {label} HPSAs (snapshot {SNAPSHOT.date()}), designated or proposed "
                     f"for withdrawal. Q = HPSA score (integer); treatment = 1{{Q >= {c}}}, the "
                     f"effective NHSC LRP cutoff in FY {fy}; Y = {y_col}"
                     + (" (PLACEHOLDER: measured before the 2025 scores)"
                        if y_col == Y_PLACEHOLDER else "") + "."),
        citation="HRSA Data Warehouse, HPSA detail files and Health Sites (data.hrsa.gov); "
                 "HRSA NHSC LRP Program Applicant Metrics FY 2020-2025; HRSA Area Health "
                 "Resources File 2024-2025; HRSA notices in the Federal Register 2003-2018",
        treatment_rule=lambda Q, c=c: (np.asarray(Q) >= c).astype(int),
        extras=_extras(df, fy, c, y_col),
    )


def load() -> RDDSample:
    """Primary care; FY 2025 LRP cutoff (13); Y = placeholder (2024 county NHSC FTE per 10k)."""
    return _sample()


def load_fy(fy: int) -> RDDSample:
    """Primary care with another fiscal year's cutoff: 2020 (16), 2023 (16), 2024 (17), 2025 (13)."""
    return _sample(fy=fy)


def load_sites_2026(fy: int = MAIN_FY) -> RDDSample:
    """Y = NHSC-approved sites per 10,000 residents in the HPSA's counties, October 2026."""
    return _sample(fy=fy, y_col=Y_SITES)


def load_area(fy: int = MAIN_FY) -> RDDSample:
    """Geographic, high-needs geographic and population HPSAs only."""
    return _sample(fy=fy, types=AREA_TYPES)


def load_with_mcta(fy: int = MAIN_FY) -> RDDSample:
    """Adds the maternity care target area (MCTA) score to X. MCTA is a separate supplementary
    score, not a component of the HPSA score, but it shares inputs (poverty, travel) with it."""
    return _sample(fy=fy, with_mcta=True)


def load_dental(fy: int = MAIN_FY) -> RDDSample:
    """Dental HPSAs (score 0-26); the LRP cutoff is the pooled one. Y = NHSC dental FTE."""
    return _sample("DH", fy=fy)


def load_mental(fy: int = MAIN_FY) -> RDDSample:
    """Mental health HPSAs (score 0-25); the LRP cutoff is the pooled one. Y = NHSC MH FTE."""
    return _sample("MH", fy=fy)


def load_components(fy: int = MAIN_FY) -> RDDSample:
    """Three-score view for primary care geographic and population HPSAs:
    Q = (ratio points 0-10, poverty points 0-5, infant health + travel points 0-10). The first
    two are rebuilt from the published inputs; the third is the score minus the two. The rule
    is 1{Q1 + Q2 + Q3 >= c}."""
    c = LRP_CUTOFFS[fy]
    df = _frame("PC", AREA_TYPES, Y_PLACEHOLDER)
    df = df[df.ratio_pts.notna()].reset_index(drop=True)
    X, names = _design(df)
    w = (1.0, 1.0, 1.0)
    return RDDSample(
        Q=df[["ratio_pts", "poverty_pts", "other_pts"]].to_numpy(float), X=X,
        Y=df[Y_PLACEHOLDER].to_numpy(float), threshold=(c - 0.5,) * 3,
        name=f"hrsa_hpsa_PC_components_fy{fy}", feature_names=names,
        description=("Primary care geographic and population HPSAs. Q = (ratio points, poverty "
                     "points, infant health + travel points); treatment = 1{sum >= "
                     f"{c}}}; Y = {Y_PLACEHOLDER} (PLACEHOLDER)."),
        citation="HRSA HPSA detail file (data.hrsa.gov); scoring criteria FR 68:32531 (2003)",
        treatment_rule=lambda Q, c=c: (np.asarray(Q) @ np.array(w) >= c).astype(int),
        extras={**_extras(df, fy, c, Y_PLACEHOLDER), "weights": w},
    )
