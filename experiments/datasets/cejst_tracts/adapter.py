"""CEJST v2.0 census tracts: "disadvantaged" as a both-cutoffs rule (low income AND burden).

The Climate and Economic Justice Screening Tool (CEQ; v2.0 released 19-20 December 2024,
2010 census tracts) identifies a tract as disadvantaged when, in its core rule, it is at
or above the 65th percentile of the adjusted share of people below 200% of the federal
poverty line (low income) AND at or above the 90th percentile of at least one burden
indicator in seven categories (climate, energy, transportation, housing, legacy
pollution, water, health). The scoring code is ``score_narwhal.py`` (see README).

Q = (Q1, Q2) on a 0-100 percentile scale, as published in 2.0-communities.csv:
    Q1 = low-income percentile (adjusted % below 200% FPL, college students excluded);
    Q2 = max over the 23 percentile burden indicators paired with low income. The lead
         paint percentile enters only where the median home value percentile is < 90,
         as the rule requires.
D = 1{Q1 >= 65 and Q2 >= 90}: the core both-cutoffs rule.
    extras["eligible_published"] is the published final flag ("Identified as
    disadvantaged"). It adds tracts through rules outside Q: yes/no burdens (historic
    underinvestment, abandoned mines, formerly used defense sites) with low income;
    the workforce category (paired with >= 10% of adults without a high-school
    diploma instead of low income); tribal overlap; tracts surrounded by disadvantaged
    tracts with low income >= 50th percentile; tracts disadvantaged in v1.0
    (grandfathered). ``rule_check()`` decomposes the disagreement.
X = race/ethnicity and age shares, log population, share not enrolled in higher
    education, state indicators (all from the same file; ACS 2015-2019).
Y = PLACEHOLDER: all zeros. The file has no outcome measured after designation.

Sample: tracts in the 50 states, DC and Puerto Rico with both scores and X observed. The
four island territories are excluded (no burden data; separate territory rule).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
CSV = RAW / "2.0-communities.csv"

LI_CUT, BURDEN_CUT, HOUSE_VALUE_CUT, LOW_HS_CUT = 65.0, 90.0, 90.0, 10.0
THRESHOLD = (LI_CUT, BURDEN_CUT)
ISLANDS = ("American Samoa", "Guam", "Northern Mariana Islands", "Virgin Islands")

GEOID = "Census tract 2010 ID"
STATE = "State/Territory"
LI = "Adjusted percent of individuals below 200% Federal Poverty Line (percentile)"
LI_FLAG = "Is low income?"
LEAD = "Percent pre-1960s housing (lead paint indicator) (percentile)"
HOUSE_VALUE = "Median value ($) of owner-occupied housing units (percentile)"
# Percentile burden indicators that are paired with low income (category in brackets).
BURDENS = {
    "Expected agricultural loss rate (Natural Hazards Risk Index) (percentile)": "climate",
    "Expected building loss rate (Natural Hazards Risk Index) (percentile)": "climate",
    "Expected population loss rate (Natural Hazards Risk Index) (percentile)": "climate",
    "Share of properties at risk of flood in 30 years (percentile)": "climate",
    "Share of properties at risk of fire in 30 years (percentile)": "climate",
    "Energy burden (percentile)": "energy",
    "PM2.5 in the air (percentile)": "energy",
    "Diesel particulate matter exposure (percentile)": "transportation",
    "Traffic proximity and volume (percentile)": "transportation",
    "DOT Travel Barriers Score (percentile)": "transportation",
    "Housing burden (percent) (percentile)": "housing",
    LEAD: "housing",
    "Share of the tract's land area that is covered by impervious surface or cropland as a "
    "percent (percentile)": "housing",
    "Share of homes with no kitchen or indoor plumbing (percentile)": "housing",
    "Proximity to hazardous waste sites (percentile)": "pollution",
    "Proximity to NPL (Superfund) sites (percentile)": "pollution",
    "Proximity to Risk Management Plan (RMP) facilities (percentile)": "pollution",
    "Wastewater discharge (percentile)": "water",
    "Leaky underground storage tanks (percentile)": "water",
    "Current asthma among adults aged greater than or equal to 18 years (percentile)": "health",
    "Diagnosed diabetes among adults aged greater than or equal to 18 years (percentile)":
        "health",
    "Coronary heart disease among adults aged greater than or equal to 18 years (percentile)":
        "health",
    "Low life expectancy (percentile)": "health",
}
# Yes/no burdens paired with low income (housing, pollution).
BOOL_BURDENS = [
    "Tract experienced historic underinvestment",
    "Is there at least one abandoned mine in this census tract, where missing data is "
    "treated as False?",
    "Is there at least one Formerly Used Defense Site (FUDS) in the tract, where missing "
    "data is treated as False?",
]
# Workforce category: any of these >= 90th percentile AND >= 10% adults without HS.
WORKFORCE = [
    "Low median household income as a percent of area median income (percentile)",
    "Linguistic isolation (percent) (percentile)",          # not used in Puerto Rico
    "Unemployment (percent) (percentile)",
    "Percent of individuals < 100% Federal Poverty Line (percentile)",
]
LOW_HS = "Percent individuals age 25 or over with less than high school degree"
FLAGS = {
    "final": "Identified as disadvantaged",
    "own": "Identified as disadvantaged without considering neighbors",
    "neighbors": "Identified as disadvantaged based on neighbors and relaxed low income "
                 "threshold only",
    "tribal": "Identified as disadvantaged due to tribal overlap",
    "v1": "Identified as disadvantaged in v1.0",
    "grandfathered": "Identified as disadvantaged solely due to status in v1.0 "
                     "(grandfathered)",
}
X_COLS = {
    "Percent Black or African American alone": "pct_black",
    "Percent American Indian / Alaska Native": "pct_aian",
    "Percent Asian": "pct_asian",
    "Percent Native Hawaiian or Pacific": "pct_nhpi",
    "Percent two or more races": "pct_two_plus",
    "Percent Hispanic or Latino": "pct_hispanic",
    "Percent other races": "pct_other",
    "Percent age under 10": "pct_age_under10",
    "Percent age over 64": "pct_age_over64",
    "Percent of residents who are not currently enrolled in higher ed": "pct_not_in_college",
}


def _bool(s: pd.Series) -> np.ndarray:
    if s.dtype == bool:
        return s.to_numpy()
    m = {True: True, False: False, "True": True, "False": False}
    return s.map(m).fillna(False).astype(bool).to_numpy()


def _pct(s: pd.Series) -> np.ndarray:
    """Percentile on a 0-100 scale (two columns are published on 0-1)."""
    v = s.astype(float).to_numpy()
    return v * 100 if np.nanmax(v) <= 1.0001 else v


@lru_cache(maxsize=1)
def tracts() -> pd.DataFrame:
    """All 74,134 tracts with Q, the rule's pieces and the published flags."""
    if not CSV.exists():
        raise FileNotFoundError(
            f"{CSV} missing. Run `python -m experiments.datasets.cejst_tracts.download` "
            "(see README).")
    d = pd.read_csv(CSV, dtype={GEOID: str}, low_memory=False)
    st = d[STATE].to_numpy()
    out = pd.DataFrame({"geoid": d[GEOID], "state": st})
    out["q1"] = _pct(d[LI])
    B = np.column_stack([_pct(d[c]) for c in BURDENS])
    hv = _pct(d[HOUSE_VALUE])
    lead = list(BURDENS).index(LEAD)
    B[:, lead] = np.where(hv < HOUSE_VALUE_CUT, B[:, lead], np.nan)
    with np.errstate(all="ignore"):
        out["q2"] = np.where(np.isnan(B).all(1), np.nan, np.nanmax(np.nan_to_num(B, nan=-1), 1))
    out["argmax_burden"] = [list(BURDENS)[i] for i in np.argmax(np.nan_to_num(B, nan=-1), 1)]
    li_pub = _bool(d[LI_FLAG])
    out["low_income_published"] = li_pub.astype(int)
    out["bool_burden"] = np.column_stack([_bool(d[c]) for c in BOOL_BURDENS]).any(1)
    pr = st == "Puerto Rico"
    W = np.column_stack([np.nan_to_num(_pct(d[c]), nan=-1) >= BURDEN_CUT for c in WORKFORCE])
    W[:, 1] &= ~pr
    out["workforce"] = W.any(1) & (np.nan_to_num(d[LOW_HS].astype(float), nan=-1) >= LOW_HS_CUT)
    for k, c in FLAGS.items():
        out[f"pub_{k}"] = _bool(d[c]).astype(int)
    out["island"] = np.isin(st, ISLANDS)
    # Rebuild of "without considering neighbors" from the published columns.
    core_pub = li_pub & (np.nan_to_num(out["q2"], nan=-1) >= BURDEN_CUT)
    out["own_rebuilt"] = (core_pub | (li_pub & out["bool_burden"]) | out["workforce"]
                          | out["pub_tribal"].astype(bool)
                          | (out["island"] & li_pub)).astype(int)
    for c, n in X_COLS.items():
        out[n] = d[c].astype(float)
    out["log_pop"] = np.log1p(d["Total population"].astype(float))
    out["income_imputed"] = _bool(d["Income data has been estimated based on geographic "
                                    "neighbor income"]).astype(int)
    out["categories_exceeded"] = d["Total categories exceeded"]
    return out


def _core(Q: np.ndarray) -> np.ndarray:
    Q = np.asarray(Q, float)
    return ((Q[:, 0] >= LI_CUT) & (Q[:, 1] >= BURDEN_CUT)).astype(int)


def rule_check() -> pd.DataFrame:
    """Core rule on Q vs the published final flag, with the reason for each disagreement.

    Reasons are assigned in order: the yes/no burdens, workforce, tribal overlap,
    neighbours (surrounded by disadvantaged tracts), grandfathered from v1.0. The sample
    is load()'s.
    """
    t = tracts()
    s = t[_keep(t)].copy()
    D = _core(s[["q1", "q2"]].fillna(-1).to_numpy())
    pub = s["pub_final"].to_numpy()
    li = s["q1"].to_numpy() >= LI_CUT
    reason = np.full(len(s), "", dtype=object)
    miss = (D == 0) & (pub == 1)
    for name, m in [("yes/no burden with low income", li & s["bool_burden"].to_numpy()),
                    ("workforce (education-paired)", s["workforce"].to_numpy()),
                    ("tribal overlap", s["pub_tribal"].to_numpy() == 1),
                    ("neighbours", s["pub_neighbors"].to_numpy() == 1),
                    ("grandfathered v1.0", s["pub_grandfathered"].to_numpy() == 1)]:
        reason[miss & (reason == "") & m] = name
    reason[miss & (reason == "")] = "other"
    reason[(D == 1) & (pub == 0)] = "core=1, published=0"
    rows = [("tracts", len(s)), ("core D = 1", int(D.sum())), ("published = 1", int(pub.sum())),
            ("agreement", float(np.mean(D == pub)))]
    rows += [(f"disagree: {r}", int((reason == r).sum())) for r in pd.unique(reason[reason != ""])]
    return pd.DataFrame(rows, columns=["quantity", "value"])


def _keep(t: pd.DataFrame) -> pd.Series:
    """States, DC and PR with both scores and every X column observed."""
    keep = ~t["island"] & t["q1"].notna() & t["q2"].notna()
    return keep & t[list(X_COLS.values()) + ["log_pop"]].notna().all(1)


def load() -> RDDSample:
    t = tracts()
    df = t[_keep(t)].reset_index(drop=True)
    states = sorted(df["state"].unique())
    st = pd.get_dummies(df["state"]).reindex(columns=states[1:], fill_value=0)
    st.columns = ["state_" + c.replace(" ", "_") for c in st.columns]
    Xdf = pd.concat([df[list(X_COLS.values()) + ["log_pop"]], st], axis=1).astype(float)
    Q = df[["q1", "q2"]].to_numpy(float)
    return RDDSample(
        Q=Q,
        X=Xdf.to_numpy(),
        Y=np.zeros(len(df)),
        threshold=THRESHOLD,
        name="cejst_tracts",
        feature_names=list(Xdf.columns),
        description=(
            "CEJST v2.0 (Dec 2024) census tracts (2010 boundaries), 50 states + DC + PR. "
            "Q1 = low-income percentile, Q2 = max burden-indicator percentile; core rule "
            "Q1 >= 65 and Q2 >= 90. Y IS A PLACEHOLDER (all zeros): no outcome after "
            "designation is in the file."),
        citation=("Council on Environmental Quality, Climate and Economic Justice Screening "
                  "Tool v2.0 (2024), 2.0-communities.csv via Wayback Machine capture of "
                  "static-data-screeningtool.geoplatform.gov, 22 Jan 2025."),
        treatment_rule=_core,
        extras={
            "geoid": df["geoid"].to_numpy(),
            "state": df["state"].to_numpy(),
            "eligible_published": df["pub_final"].to_numpy(),
            "eligible_own_published": df["pub_own"].to_numpy(),
            "low_income_published": df["low_income_published"].to_numpy(),
            "bool_burden": df["bool_burden"].to_numpy().astype(int),
            "workforce": df["workforce"].to_numpy().astype(int),
            "tribal": df["pub_tribal"].to_numpy(),
            "neighbors": df["pub_neighbors"].to_numpy(),
            "grandfathered": df["pub_grandfathered"].to_numpy(),
            "income_imputed": df["income_imputed"].to_numpy(),
            "argmax_burden": df["argmax_burden"].to_numpy(),
            # readiness near-boundary share: distance to the low-income cutoff only
            "weights": (1.0, 0.0),
            "cutoff": LI_CUT,
            "y_placeholder": True,
        },
    )
