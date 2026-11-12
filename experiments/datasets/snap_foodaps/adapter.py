"""SNAP benefit formula in the FoodAPS-1 public-use files (USDA ERS; interviews April 2012 - January 2013).

Dose design. A SNAP household's monthly allotment is a published function of its net income:

    benefit = max_allot(size) - ceil(0.30 x net income),   at least $16 for 1-2 persons,

    net income = max(0, AI - excess shelter),
    AI         = gross - 0.20 x earnings - standard deduction(size) - dependent care
                 - child support paid - (medical costs of elderly/disabled members - $35)+,
    excess shelter = (shelter costs - 0.5 x AI)+, capped unless a member is elderly or disabled.

Parameters for the 48 States and DC (FNS "FY 2012 / FY 2013 Allotments and Deduction Information"
and "FY 2012 / FY 2013 Minimum SNAP Allotment"; ARRA maximum allotments, in force April 2009 to
31 October 2013, the same in both fiscal years):

    max allotment   1: 200, 2: 367, 3: 526, 4: 668, 5: 793, 6: 952, 7: 1,052, 8: 1,202, +150 each
    standard ded.   FY 2012: 147 (1-3), 155 (4), 181 (5), 208 (6+);  FY 2013: 149, 160, 187, 214
    shelter cap     FY 2012: 459;  FY 2013: 469
    minimum benefit $16 (1-2 persons) in both years; medical threshold $35; earnings deduction 20%

Interview month -> fiscal year: April-September 2012 -> FY 2012; October-December 2012 and January
2013 -> FY 2013 (the PUF has the month only; January interviews are January 2013). No interview
falls after October 2013, so every household is under the ARRA tables.

What the public-use file lacks (User's Guide, appendix C, table C1): the State (so neither the
State standard utility allowances nor the BBCE rules can be applied), individual income amounts
(only indicators of each income type; household income INCHHAVG_R is given, so the earnings share
is approximated), the elderly/disabled share of medical costs (EXPOPMEDICAL60), and the
administrative SNAP amounts (ADMINAMT1-4). The recorded benefit in the PUF is SNAPLASTAMT_R, the
self-reported last SNAP amount in $50 bins (codes 1-18). ERS also provides BENEST1_HH-BENEST4_HH,
Mathematica's simulated formula benefits computed on the restricted file (four model runs).

Hence two benefit "rules" are compared with the recorded bin in ``rule_check``: this module's
reconstruction from PUF variables, and the BENEST runs.

Sample: households with administratively confirmed current SNAP receipt (SNAPNOWADMIN = 1;
1,316 households). Unit = the household without guests (HHSIZE); Mathematica's runs 3-4 split
273 households into several SNAP units, which the PUF cannot reproduce.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw" / "csv"
PROC = HERE / "data" / "processed"

# --------------------------------------------------------------------------- rule parameters
MAX_ALLOT = {1: 200, 2: 367, 3: 526, 4: 668, 5: 793, 6: 952, 7: 1052, 8: 1202}
MAX_ADD = 150
STD_DED = {2012: {1: 147, 2: 147, 3: 147, 4: 155, 5: 181, 6: 208},
           2013: {1: 149, 2: 149, 3: 149, 4: 160, 5: 187, 6: 214}}
SHELTER_CAP = {2012: 459, 2013: 469}
MIN_BEN = 16
MED_THRESHOLD = 35.0
EARN_DED = 0.20
RATE = 0.30
# Income standards (48 States + DC). FY 2012 from the FNS table; FY 2013 = 2012 HHS poverty
# guideline / 12 rounded up (the FNS FY 2013 table could not be fetched; the guideline is in the
# PUF as POVGUIDE_HH and matches 11,170 + 3,960 per extra person).
NET_STD = {2012: ([908, 1226, 1545, 1863, 2181, 2500, 2818, 3136], 319),
           2013: ([931, 1261, 1591, 1921, 2251, 2581, 2911, 3241], 330)}
GROSS_STD = {2012: ([1180, 1594, 2008, 2422, 2836, 3249, 3663, 4077], 414),
             2013: ([1211, 1640, 2069, 2498, 2927, 3356, 3785, 4214], 430)}

# SNAPLASTAMT_R bins (codebook: 1: 0 < amount <= 16, 2: 16 < a < 50, 3: 50 <= a < 100, ...,
# 17: 750 <= a < 800, 18: 800 <= a < 1200)
BIN_LO = np.array([np.nan, 0.0, 16.0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800])
BIN_HI = np.array([np.nan, 16.0, 50.0, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 1200])

INC_TYPES = ["incearnindavg_r", "incunempindavg_r", "inctransferindavg_r", "incretdisindavg_r",
             "incinvestindavg_r", "incotherindavg_r"]
INC_SHORT = {c: c[3:-len("indavg_r")] for c in INC_TYPES}       # earn, unemp, transfer, retdis, invest, other
EXP = ["exprentmrtg_r", "exphomeins_r", "expproptax_r", "expelectric_r", "expheatfuel_r", "expwastedisp_r",
       "exphealthins_r", "expcopay_r", "expdoctor_r", "exprx_r", "expchildcare_r", "expchildsupport_r",
       "expadultcare_r"]
Z_NAMES = ["n_children", "pr_age", "n_elderly"]


def max_allot(size) -> np.ndarray:
    s = np.asarray(size, int)
    base = np.array([MAX_ALLOT[min(k, 8)] for k in s.ravel()], float).reshape(s.shape)
    return base + MAX_ADD * np.maximum(s - 8, 0)


def std_ded(size, fy) -> np.ndarray:
    s = np.minimum(np.asarray(size, int), 6)
    fy = np.asarray(fy, int)
    return np.array([STD_DED[f][k] for k, f in zip(s.ravel(), fy.ravel())], float)


def _std(table, size, fy) -> np.ndarray:
    out = []
    for k, f in zip(np.asarray(size, int), np.asarray(fy, int)):
        base, add = table[f]
        out.append(base[min(k, 8) - 1] + add * max(k - 8, 0))
    return np.array(out, float)


def schedule(net, size, fy=None, floor: bool = True) -> np.ndarray:
    """Benefit in dollars from net income (the dose schedule, one per household size)."""
    net = np.maximum(np.asarray(net, float), 0.0)
    size = np.asarray(size, int)
    b = max_allot(size) - np.ceil(RATE * net - 1e-9)
    lo = np.where(size <= 2, MIN_BEN, 0.0) if floor else 0.0
    return np.maximum(b, lo)


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run experiments.datasets.snap_foodaps.download (README.md)")


def _num(s: pd.Series) -> pd.Series:
    """Individual-file codes: V (valid skip), D (don't know), R (refused) -> NaN."""
    return pd.to_numeric(s, errors="coerce")


# --------------------------------------------------------------------------- household frame
def persons() -> pd.DataFrame:
    _require(RAW / "faps_individual_puf.csv")
    d = pd.read_csv(RAW / "faps_individual_puf.csv", dtype=str)
    d.columns = d.columns.str.lower()
    for c in ["hhnum", "pnum", "guest", "sex", "relation_r", "age_r", "hispanic", "racecat_r", "educcat", "marital",
              "employment", "reasonnowork", "schlevel_r", "schlunch", "schbrkfstrecvd", "wicind", "healthstatus",
              "usborn", "uscitizen"] + INC_TYPES:
        d[c] = _num(d[c])
    return d


def spending() -> pd.DataFrame:
    """7-day food spending per household: FAH food items (net of savings and coupons, imputed when
    missing), FAH event totals (may include nonfood items and tax), FAFH event totals (incl. tip)."""
    for f in ("faps_fahitem_puf.csv", "faps_fahevent_puf.csv", "faps_fafhevent_puf.csv"):
        _require(RAW / f)
    it = pd.read_csv(RAW / "faps_fahitem_puf.csv", usecols=["hhnum", "totitemexp", "imputedexp"], encoding="latin-1")
    exp = it.totitemexp.where(it.totitemexp > -990)
    exp = exp.fillna(it.imputedexp.where(it.imputedexp > -990))
    fah_food = exp.fillna(0.0).groupby(it.hhnum).sum()
    ev = pd.read_csv(RAW / "faps_fahevent_puf.csv", usecols=["hhnum", "totalpaid"])
    fah_paid = ev.totalpaid.where(ev.totalpaid >= 0).fillna(0.0).groupby(ev.hhnum).sum()
    fah_n = ev.groupby("hhnum").size()
    fa = pd.read_csv(RAW / "faps_fafhevent_puf.csv", usecols=["hhnum", "totalpaid"])
    fafh_paid = fa.totalpaid.where(fa.totalpaid >= 0).fillna(0.0).groupby(fa.hhnum).sum()
    fafh_n = fa.groupby("hhnum").size()
    out = pd.DataFrame({"fah_food": fah_food, "fah_paid": fah_paid, "fafh_paid": fafh_paid,
                        "n_fah": fah_n, "n_fafh": fafh_n})
    return out.fillna(0.0)


def frame(rebuild: bool = False) -> pd.DataFrame:
    """All 4,826 households with the reconstructed net income and formula benefit; cached."""
    PROC.mkdir(parents=True, exist_ok=True)
    cache = PROC / "households.parquet"
    if cache.exists() and not rebuild:
        return pd.read_parquet(cache)
    _require(RAW / "faps_household_puf.csv")
    h = pd.read_csv(RAW / "faps_household_puf.csv")
    p = persons()
    p = p[p.guest == 0].copy()
    age = p.age_r.where(p.age_r >= 0)
    p["elderly"] = (age >= 60).astype(float)
    p["child"] = (age < 18).astype(float)
    p["disabled"] = ((age < 60) & (p.reasonnowork == 5) & (p.incretdisindavg_r == 1)).astype(float)
    p["employed"] = p.employment.isin([1, 2]).astype(float)
    p["looking"] = p.employment.isin([3, 4]).astype(float)
    p["earner"] = (p.incearnindavg_r == 1).astype(float)
    p["unearned_k"] = (p[INC_TYPES[1:]] == 1).sum(axis=1).astype(float)
    p["school_lunch"] = (p.schlunch == 1).astype(float)
    p["school_bkfst"] = (p.schbrkfstrecvd == 1).astype(float)
    p["wic"] = (p.wicind == 1).astype(float)
    for c in INC_TYPES:
        p["any_" + INC_SHORT[c]] = (p[c] == 1).astype(float)
    g = p.groupby("hhnum")
    agg = g[["elderly", "child", "disabled", "employed", "looking", "earner", "unearned_k", "school_lunch",
             "school_bkfst", "wic"] + ["any_" + INC_SHORT[c] for c in INC_TYPES]].sum()
    agg.columns = ["n_elderly", "n_children", "n_disabled", "n_employed", "n_looking", "n_earners", "n_unearned",
                   "n_school_lunch", "n_school_bkfst", "n_wic"] + ["n_" + INC_SHORT[c] for c in INC_TYPES]
    adults = p[(age >= 18) | age.isna()]
    agg["max_educ"] = adults.groupby("hhnum").educcat.max()
    pr = p[p.relation_r == 0].set_index("hhnum")
    for c in ["age_r", "sex", "racecat_r", "hispanic", "educcat", "marital", "healthstatus", "usborn", "uscitizen",
              "employment"]:
        agg["pr_" + c] = pr[c]
    d = h.merge(agg, left_on="hhnum", right_index=True, how="left")
    d = d.merge(spending(), left_on="hhnum", right_index=True, how="left")
    for c in ["fah_food", "fah_paid", "fafh_paid", "n_fah", "n_fafh"]:
        d[c] = d[c].fillna(0.0)
    d["fy"] = np.where(d.initintrvmon.isin([10, 11, 12, 1]), 2013, 2012)
    d = reconstruct(d)
    d.to_parquet(cache)
    return d


def reconstruct(d: pd.DataFrame, earn: str = "share", utilities: str = "actual") -> pd.DataFrame:
    """Net income and formula benefit from PUF variables (see the module docstring).

    earn: "share" (earnings = gross x earners / (earners + unearned income sources)), "all" (gross
    is all earnings if anyone earns), "none" (no earnings deduction).
    """
    d = d.copy()
    e = {c: d[c].where(d[c] >= 0).fillna(0.0) for c in EXP}
    d["n_exp_missing"] = sum((d[c] < 0).astype(int) for c in EXP)
    G = d.inchhavg_r.astype(float)
    k_e, k_u = d.n_earners.fillna(0), d.n_unearned.fillna(0)
    if earn == "share":
        E = G * np.where(k_e > 0, k_e / np.maximum(k_e + k_u, 1), 0.0)
    elif earn == "all":
        E = G * (k_e > 0)
    else:
        E = 0.0 * G
    size = d.hhsize.astype(int)
    ed = (d.n_elderly.fillna(0) + d.n_disabled.fillna(0)) > 0
    med = e["exphealthins_r"] + e["expcopay_r"] + e["expdoctor_r"] + e["exprx_r"] + e["expadultcare_r"]
    med_ed = med * (d.n_elderly.fillna(0) + d.n_disabled.fillna(0)).clip(upper=size) / size
    med_ded = np.maximum(med_ed - MED_THRESHOLD, 0.0)
    dep = e["expchildcare_r"]
    cs = e["expchildsupport_r"]
    sd = std_ded(size, d.fy)
    ai = G - EARN_DED * E - sd - dep - cs - med_ded            # not floored: the continuous score
    util = e["expelectric_r"] + e["expheatfuel_r"] + e["expwastedisp_r"]
    shelter = e["exprentmrtg_r"] + e["exphomeins_r"] + e["expproptax_r"] + util
    excess = np.maximum(shelter - 0.5 * np.maximum(ai, 0.0), 0.0)
    cap = d.fy.map(SHELTER_CAP).astype(float)
    excess = np.where(ed, excess, np.minimum(excess, cap))
    d["gross"] = G
    d["earn_hat"] = E
    d["elderly_disabled"] = ed.astype(float)
    d["std_ded"] = sd
    d["med_ded"] = med_ded
    d["shelter"] = shelter
    d["excess_shelter"] = excess
    d["adj_income"] = ai
    d["net_raw"] = ai - excess                                   # score Q (dollars, may be < 0)
    d["net"] = np.maximum(d.net_raw, 0.0)
    d["max_allot"] = max_allot(size)
    d["ben_formula"] = schedule(d.net, size)
    d["pass_gross"] = (G <= _std(GROSS_STD, size, d.fy)).astype(float)
    d["pass_net"] = (d.net <= _std(NET_STD, size, d.fy)).astype(float)
    return d


def participants(rebuild: bool = False, **kw) -> pd.DataFrame:
    """Households with administratively confirmed current SNAP receipt (SNAPNOWADMIN = 1)."""
    d = frame(rebuild)
    if kw:
        d = reconstruct(d, **kw)
    return d[d.snapnowadmin == 1].reset_index(drop=True)


# --------------------------------------------------------------------------- rule check
def _bin_distance(v, code) -> np.ndarray:
    """Distance (dollars) from v to the recorded SNAPLASTAMT_R bin; 0 inside the bin."""
    v = np.asarray(v, float)
    c = np.asarray(code, int)
    lo, hi = BIN_LO[np.clip(c, 0, 18)], BIN_HI[np.clip(c, 0, 18)]
    below = np.where(c == 1, v <= lo, v < lo)                    # bin 1 is (0, 16]
    above = np.where(c == 1, v > hi, (v >= hi) & (c != 18)) | ((c == 18) & (v > hi))
    return np.where(below, lo - v, np.where(above, v - hi, 0.0))


def rule_check(d: pd.DataFrame | None = None) -> pd.DataFrame:
    """Agreement of formula benefits with the recorded (self-reported, $50-binned) SNAP amount.

    Rows: the reconstruction from PUF variables (three earnings treatments), Mathematica's simulated
    benefits BENEST1-4 (restricted data, State rules), and the maximum allotment. Columns: share in
    the recorded bin, within $10 of it, within 10% of it, and (for continuous benefits) agreement with
    BENEST3: exact, within $10, within 10%. Units: participants with a recorded amount (code 1-18).
    """
    base = participants() if d is None else d
    code = base.snaplastamt_r.to_numpy(int)
    ok = (code >= 1) & (code <= 18)
    b3 = base.benest3_hh.to_numpy(float)
    rows = []
    cands: Dict[str, np.ndarray] = {}
    for earn in ("share", "all", "none"):
        cands[f"formula (earn={earn})"] = reconstruct(base, earn=earn).ben_formula.to_numpy(float)
    for k in (1, 2, 3, 4):
        cands[f"BENEST{k}_HH"] = base[f"benest{k}_hh"].to_numpy(float)
    cands["max allotment"] = base.max_allot.to_numpy(float)
    for name, v in cands.items():
        dist = _bin_distance(v, code)
        r = {"rule": name, "n": int(ok.sum()),
             "in_bin": float(np.mean(dist[ok] == 0)),
             "within_10usd": float(np.mean(dist[ok] <= 10)),
             "within_10pct": float(np.mean(dist[ok] <= 0.1 * np.maximum(v[ok], 1.0)))}
        if not name.startswith("BENEST3"):
            m = b3 >= 0
            r["vs_benest3_exact"] = float(np.mean(np.abs(v[m] - b3[m]) < 0.5))
            r["vs_benest3_10usd"] = float(np.mean(np.abs(v[m] - b3[m]) <= 10))
            r["vs_benest3_10pct"] = float(np.mean(np.abs(v[m] - b3[m]) <= 0.1 * np.maximum(b3[m], 1.0)))
        rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- dose sample
OUTCOMES = {
    "fah_food": "7-day food-at-home spending on food items ($, net of savings and coupons; FAH item file)",
    "fah_paid": "7-day food-at-home spending, event totals ($, may include nonfood items and tax)",
    "total_food": "7-day food spending: FAH food items + food away from home (event totals incl. tip)",
    "fafh_paid": "7-day food-away-from-home spending ($)",
}


def design(d: pd.DataFrame) -> pd.DataFrame:
    """Predetermined covariates: size and composition, the respondent's demographics, education,
    employment, income-type indicators, housing tenure, other assistance, region, urbanicity, month."""
    X = pd.DataFrame(index=d.index)
    for k in range(2, 7):
        X[f"size_{k}"] = (d.hhsize == k).astype(float)
    X["size_7p"] = (d.hhsize >= 7).astype(float)
    for c in ["n_children", "n_elderly", "n_disabled", "n_employed", "n_looking", "n_earners", "n_unemp",
              "n_transfer", "n_retdis", "n_invest", "n_other", "n_wic", "n_school_lunch", "n_school_bkfst"]:
        X[c] = d[c].fillna(0.0).astype(float)
    age = d.pr_age_r.where(d.pr_age_r >= 0)
    X["pr_age"] = age.fillna(age.median())
    X["pr_age2"] = X.pr_age ** 2 / 100
    X["pr_female"] = (d.pr_sex == 2).astype(float)
    X["pr_black"] = (d.pr_racecat_r == 2).astype(float)
    X["pr_other_race"] = d.pr_racecat_r.isin([3, 4, 5, 6]).astype(float)
    X["pr_hispanic"] = (d.pr_hispanic == 1).astype(float)
    X["pr_married"] = (d.pr_marital == 1).astype(float)
    X["pr_usborn"] = (d.pr_usborn == 1).astype(float)
    X["pr_health_fairpoor"] = d.pr_healthstatus.isin([4, 5]).astype(float)
    X["pr_employed"] = d.pr_employment.isin([1, 2]).astype(float)
    ed = d.max_educ
    X["educ_hs"] = (ed == 3).astype(float)
    X["educ_somecoll"] = (ed == 4).astype(float)
    X["educ_ba"] = (ed >= 5).astype(float)
    X["educ_na"] = ed.isna().astype(float)
    X["own_home"] = (d.housingown == 1).astype(float)
    X["rent_free"] = (d.housingown == 3).astype(float)
    X["public_housing"] = (d.housingpub == 1).astype(float)
    X["subsidized"] = (d.housingsub == 1).astype(float)
    X["any_vehicle"] = (d.anyvehicle == 1).astype(float)
    X["wic_hh"] = (d.wichh == 1).astype(float)
    X["rural"] = d.rural.astype(float)
    X["nonmetro"] = d.nonmetro.astype(float)
    for r in (2, 3, 4):
        X[f"region_{r}"] = (d.region == r).astype(float)
    for m in sorted(d.initintrvmon.unique())[1:]:
        X[f"month_{m}"] = (d.initintrvmon == m).astype(float)
    return X.loc[:, X.std(axis=0) > 0]


def load_dose(outcome: str = "fah_food", dose: str = "relative", earn: str = "share", rebuild: bool = False,
              min_events: int = 0) -> RDDSample:
    """Participant-level dose sample.

    dose="relative": Q = 0.3 x net income / max allotment (share of the break-even net income; may be
    < 0), A = formula benefit / max allotment, a_ref = 1 (net income <= 0: the maximum allotment).
    dose="dollars": Q = net income ($, may be < 0), A = formula benefit ($); the schedules differ by
    household size (intercept = max allotment); a_ref is left to the caller.
    """
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {sorted(OUTCOMES)}")
    d = participants(rebuild, earn=earn) if earn != "share" else participants(rebuild)
    d["total_food"] = d.fah_food + d.fafh_paid
    if min_events:
        d = d[(d.n_fah + d.n_fafh) >= min_events].reset_index(drop=True)
    X = design(d)
    if dose == "relative":
        Q = (RATE * d.net_raw / d.max_allot).to_numpy(float)
        A = (d.ben_formula / d.max_allot).to_numpy(float)
        a_ref = 1.0
    elif dose == "dollars":
        Q = d.net_raw.to_numpy(float)
        A = d.ben_formula.to_numpy(float)
        a_ref = None
    else:
        raise ValueError("dose must be 'relative' or 'dollars'")
    size = d.hhsize.to_numpy(int)
    sched = lambda q, s=None: schedule(q if dose == "dollars" else q * max_allot(s) / RATE, s) / (  # noqa: E731
        1.0 if dose == "dollars" else max_allot(s))
    return RDDSample(
        Q=Q,
        X=X.to_numpy(float),
        Y=d[outcome].to_numpy(float),
        threshold=0.0,
        name="snap_foodaps",
        feature_names=list(X.columns),
        description=(f"FoodAPS-1 SNAP households (administratively confirmed), 2012-13. Q = reconstructed net "
                     f"income ({dose}); A = formula SNAP benefit; Y = {OUTCOMES[outcome]}."),
        citation=("USDA ERS, National Household Food Acquisition and Purchase Survey (FoodAPS), public-use files "
                  "(2016-2021 releases); SNAP parameters from USDA FNS FY 2012 and FY 2013 COLA tables."),
        treatment_rule=lambda q: (np.asarray(q) > 0).astype(int),
        extras={
            "A": A,
            "a_ref": a_ref,
            "dose_levels": None,
            "schedule": sched,
            "group": size,
            "Z_names": [z for z in Z_NAMES if z in X.columns],
            "outcomes": {k: (d[k] if k in d else d.fah_food + d.fafh_paid).to_numpy(float) for k in OUTCOMES},
            "hhnum": d.hhnum.to_numpy(),
            "psu": (d.tsstrata.astype(str) + "_" + d.tspsu.astype(str)).to_numpy(),
            "hhwgt": d.hhwgt.to_numpy(float),
            "net": d.net.to_numpy(float),
            "net_raw": d.net_raw.to_numpy(float),
            "ben_formula": d.ben_formula.to_numpy(float),
            "max_allot": d.max_allot.to_numpy(float),
            "snaplastamt_code": d.snaplastamt_r.to_numpy(int),
            "benest3": d.benest3_hh.to_numpy(float),
            "fy": d.fy.to_numpy(int),
            "rule_check": rule_check,
        },
    )
