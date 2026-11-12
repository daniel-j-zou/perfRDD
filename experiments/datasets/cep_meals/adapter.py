"""Texas Community Eligibility Provision (CEP): free-meal claiming share on the identified
student percentage, school sites x program years 2015-16 .. 2025-26.

Under CEP (7 CFR 245.9(f)) a school, a group of schools or a whole LEA whose identified
student percentage (ISP: students certified for free meals without an application, over
enrolled students, as of April 1 of the prior school year) is at least the threshold may
serve every breakfast and lunch free. The share of meals reimbursed at the federal free
rate is the free claiming percentage

    s = min(1, 1.6 x ISP),            the rest at the paid rate,

where in years 2-4 of a 4-year cycle the ISP may be the higher of the current one and the
one at the start of the cycle, and a group's ISP is its pooled ISP. The threshold was 40%
until the rule of 26 Sept. 2023 (88 FR 65778, effective 26 Oct. 2023) lowered it to 25%;
Texas applied 25% from program year 2024-25 (ISP of April 1, 2024).

Adoption is a choice. The DOSE used here is the claiming share OFFERED to a school on its
own ISP (an intention-to-treat dose):

    A = s(Q, py) = 1{Q >= c_py} x min(1, 1.6 Q),   c_py = 0.40 (py <= 2024), 0.25 (py >= 2025),

with Q the site's ISP (0-1). a_ref = 0 (not offered). The received dose (CEP x the claiming
share implied by the site's lunch reimbursement) is in ``extras["A_received"]``.

``load_dose()`` returns an ``RDDSample`` (see README.md):

    Q          site ISP (0-1), "reported April of the previous program year" (TDA)
    A          extras["A"]: offered free claiming share (above); group = program year
    Y          lunch participation: lunches served / (lunch days x enrollment) in program
               year t (``outcome="lunch"``; "breakfast", "meals" also available)
    X          log enrollment, grade composition, charter, alternative-instruction campus,
               FSMC, ESC region, CE size, leave-one-out CE ISP, program-year indicators

Program years are labelled by the calendar year in which they end (2024 = 2023-24).
"""
from __future__ import annotations

import glob
import os
import tempfile
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
PROC = HERE / "data" / "processed"
CACHE = PROC / "cep_meals_panel.csv.gz"

MULT = 1.6
PROGRAM_YEARS = tuple(range(2016, 2027))                 # claims 2015-16 .. 2025-26
COVID_YEARS = (2020, 2021, 2022)                         # school closures, nationwide free-meal waivers
STALE_ISP_YEARS = (2018,)                                # TDA's 2017-18 file repeats the 2016-17 SiteISP for every site
DEFAULT_YEARS = (2016, 2017, 2019, 2023, 2024, 2025, 2026)
Z_NAMES = ["elem", "log_enr", "charter"]


def threshold(py) -> np.ndarray:
    """CEP minimum ISP (0-1) for elections in program year py: 0.40 through 2023-24, 0.25 after."""
    return np.where(np.asarray(py) >= 2025, 0.25, 0.40)


def schedule(q, group, unit: str = "share") -> np.ndarray:
    """Offered dose on the site's own ISP.

    q      ISP on a 0-1 scale; group: program year (2025 = 2024-25), scalar or array;
    unit   "share": free claiming share 1{q >= c} x min(1, 1.6 q) (``extras["A"]``);
           "isp": the ISP that would be multiplied (q above the threshold, else 0).
    """
    q = np.asarray(q, float)
    g = np.broadcast_to(np.asarray(group), q.shape)
    on = q >= threshold(g) - 1e-12
    if unit == "share":
        return np.where(on, np.minimum(1.0, MULT * q), 0.0)
    if unit == "isp":
        return np.where(on, q, 0.0)
    raise ValueError(unit)


# ----------------------------------------------------------------------------- readers
def _need(pattern: str) -> list:
    files = sorted(glob.glob(str(RAW / pattern)))
    if not files:
        raise FileNotFoundError(f"{RAW / pattern} missing. Run experiments/datasets/cep_meals/download.py "
                                "(see README).")
    return files


def _norm_ids(d: pd.DataFrame) -> pd.DataFrame:
    d["ceid"] = d.ceid.astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
    d["siteid"] = d.siteid.astype(str).str.replace(r"\D", "", regex=True).str.zfill(4)
    return d


def read_participation() -> pd.DataFrame:
    out = []
    for f in _need("tda_participation/part_*.csv"):
        d = pd.read_csv(f, dtype=str).rename(columns={
            "site_isp": "siteisp", "typeofsnporg": "typeoforg", "severneedbreakfast": "severeneedbreakfast",
            "geocoded_column": "geo", "geoloc_data": "geo", "geolocation": "geo"})
        d["py"] = int(Path(f).name.split("_")[1])
        out.append(_norm_ids(d))
    d = pd.concat(out, ignore_index=True)
    d["isp"] = pd.to_numeric(d.siteisp, errors="coerce") / 100.0
    d["cdc"] = d.countydistrictcode.astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)
    d["campus_id"] = d.cdc + d.siteid.str[-3:]
    return d


def read_claims() -> pd.DataFrame:
    out = []
    for f in _need("tda_claims/claims_*.csv"):
        d = pd.read_csv(f, dtype={"ceid": str, "siteid": str})
        d["py"] = int(Path(f).name.split("_")[1])
        out.append(_norm_ids(d))
    return pd.concat(out, ignore_index=True)


def read_askted() -> pd.DataFrame:
    a = pd.read_csv(_need("tea_askted/askted_*.csv")[0], dtype=str)
    a["campus_id"] = a.school_number.str.replace(r"\D", "", regex=True).str.zfill(9)
    return a[["campus_id", "instruction_type", "charter_type", "nces_school_id", "grade_range"]] \
        .drop_duplicates("campus_id")


# ----------------------------------------------------------------------------- rates, implied share
def lunch_rates(d: pd.DataFrame) -> pd.DataFrame:
    """Federal lunch reimbursement per free / reduced / paid lunch by program year, recovered by OLS
    of the site's lunch reimbursement on its free, reduced and paid lunch counts (non-CEP sites).

    The coefficients average over the 2-cent higher rate of SFAs with >= 60% free/reduced lunches
    and include the performance-based add-on (6-9 cents); 2022-23 includes the Keep Kids Fed Act
    supplement (+40 cents)."""
    rows = []
    m = (d.cep == "N") & (d.lunch > 0) & d.lunch_reimb.notna()
    for py, g in d[m].groupby("py"):
        Xr = g[["lunch_free", "lunch_redc", "lunch_paid"]].to_numpy(float)
        b, *_ = np.linalg.lstsq(Xr, g.lunch_reimb.to_numpy(float), rcond=None)
        e = (g.lunch_reimb - Xr @ b) / g.lunch
        rows.append({"py": py, "free": b[0], "reduced": b[1], "paid": b[2], "n": len(g),
                     "share_within_2c": float((e.abs() <= 0.02).mean())})
    return pd.DataFrame(rows)


def _group_isp(d: pd.DataFrame) -> pd.DataFrame:
    """Per CE-year: enrollment-weighted ISP of its CEP sites (the pooled group ISP when the CE claims
    one percentage for all its CEP sites), the max of that over t-3..t, the CE-wide ISP, and the
    median implied claiming share of its CEP sites."""
    w = d.enr_avg.fillna(0).clip(lower=1)
    d = d.assign(_w=w, _wi=w * d.isp)
    ce = d[d.isp.notna()].groupby(["py", "ceid"]).agg(ce_w=("_w", "sum"), ce_wi=("_wi", "sum"),
                                                      ce_n=("isp", "size")).reset_index()
    ce["isp_ce"] = ce.ce_wi / ce.ce_w
    c = d[(d.cep == "Y") & d.isp.notna()]
    grp = c.groupby(["py", "ceid"]).agg(g_w=("_w", "sum"), g_wi=("_wi", "sum"), n_cep=("isp", "size")).reset_index()
    grp["isp_grp"] = grp.g_wi / grp.g_w
    ok = c[(c.lunch > 100) & c.s_impl.notna()]
    grp = grp.merge(ok.groupby(["py", "ceid"]).s_impl.median().rename("s_ce").reset_index(), how="left")
    piv = grp.pivot(index="ceid", columns="py", values="isp_grp")
    mx = pd.DataFrame({py: piv[[c for c in piv.columns if py - 3 <= c <= py]].max(axis=1) for py in piv.columns})
    mx = mx.stack().rename("isp_grp_max4").reset_index()
    mx.columns = ["ceid", "py", "isp_grp_max4"]
    grp = grp.merge(mx, on=["ceid", "py"], how="left")
    return ce[["py", "ceid", "isp_ce", "ce_n", "ce_w"]].merge(
        grp[["py", "ceid", "isp_grp", "isp_grp_max4", "n_cep", "s_ce"]], how="left")


# ----------------------------------------------------------------------------- panel
def build_panel() -> pd.DataFrame:
    p, c = read_participation(), read_claims()
    d = p.merge(c, on=["py", "ceid", "siteid"], how="inner")
    d = d.merge(read_askted(), on="campus_id", how="left")
    r = lunch_rates(d)
    d = d.merge(r[["py", "free", "paid"]].rename(columns={"free": "rate_free", "paid": "rate_paid"}), on="py",
                how="left")
    d["reimb_per_lunch"] = d.lunch_reimb / d.lunch
    d["s_impl"] = np.where(d.cep == "Y", (d.reimb_per_lunch - d.rate_paid) / (d.rate_free - d.rate_paid), np.nan)
    d = d.merge(_group_isp(d), on=["py", "ceid"], how="left")
    d["A"] = schedule(d.isp.fillna(0), d.py)
    d.loc[d.isp.isna(), "A"] = np.nan
    d["A_received"] = np.where(d.cep == "Y", np.clip(d.s_impl, 0, 1), 0.0)
    d["rule_site"] = np.minimum(1, MULT * d.isp)
    d["rule_grp"] = np.minimum(1, MULT * d.isp_grp_max4)
    d["y_lunch"] = d.lunch / d.lunch_den
    d["y_breakfast"] = d.bfast / d.bfast_den
    d["y_meals"] = (d.lunch + d.bfast) / d.lunch_den
    return d


def panel(rebuild: bool = False) -> pd.DataFrame:
    if CACHE.exists() and not rebuild:
        return pd.read_csv(CACHE, dtype={"ceid": str, "siteid": str, "cdc": str, "campus_id": str,
                                         "esc": str, "nces_school_id": str}, low_memory=False)
    d = build_panel()
    PROC.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=PROC, suffix=".csv.gz")
    os.close(fd)
    d.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, CACHE)
    return d


# ----------------------------------------------------------------------------- design
GRADES = ["gradeprek", "gradekinder"] + [f"grade{i}" for i in range(1, 13)]


def sample(d: pd.DataFrame, years: Iterable[int] = DEFAULT_YEARS, outcome: str = "lunch") -> pd.DataFrame:
    """Public and charter school sites with an ISP, at least 6 claim months, enrollment >= 20 and an
    outcome in (0, 1.5]."""
    y = f"y_{outcome}"
    m = (d.py.isin(list(years)) & d.typeoforg.isin(["Public", "Charter"]) & d.isp.notna()
         & (d.n_months >= 6) & (d.enr_avg >= 20) & (d[y] > 0) & (d[y] <= 1.5))
    return d[m].reset_index(drop=True)


def design(d: pd.DataFrame) -> pd.DataFrame:
    G = (d[GRADES].astype(str) == "Y").astype(float)
    X = pd.DataFrame(index=d.index)
    X["log_enr"] = np.log(d.enr_avg.astype(float))
    X["elem"] = G[["gradekinder", "grade1", "grade2", "grade3", "grade4", "grade5"]].mean(1)
    X["mid"] = G[["grade6", "grade7", "grade8"]].mean(1)
    X["high"] = G[["grade9", "grade10", "grade11", "grade12"]].mean(1)
    X["prek"] = G["gradeprek"]
    X["n_grades"] = G.sum(1)
    X["charter"] = (d.typeoforg == "Charter").astype(float)
    it = d.instruction_type.fillna("")
    X["alt_instr"] = it.str.contains("ALTERNATIVE|DAEP|JJAEP").astype(float)
    X["askted_missing"] = (it == "").astype(float)
    X["fsmc"] = (d.managedbyfsmc.astype(str).str.upper().str[0] == "Y").astype(float)
    # CE (district) context: size and the ISP of the CE's other sites (leave one out)
    w = d.enr_avg.astype(float)
    key = [d.py, d.ceid]
    ce_w = w.groupby(key).transform("sum")
    ce_wi = (w * d.isp).groupby(key).transform("sum")
    ce_n = w.groupby(key).transform("size")
    loo = (ce_wi - w * d.isp) / (ce_w - w).clip(lower=1)
    X["log_ce_enr"] = np.log(ce_w)
    X["log_ce_sites"] = np.log(ce_n)
    X["loo_isp_missing"] = (ce_n <= 1).astype(float)
    X["loo_isp"] = np.where(ce_n > 1, loo, np.nan)
    X["loo_isp"] = X.loo_isp.fillna(X.loo_isp.mean())
    for name, col in (("esc", d.esc.astype(str).str.replace(r"\D", "", regex=True).str.lstrip("0")),
                      ("py", d.py.astype(str))):
        X = pd.concat([X, pd.get_dummies(col, prefix=name, drop_first=True).astype(float)], axis=1)
    return X.loc[:, X.std() > 0]


def load_dose(outcome: str = "lunch", years: Iterable[int] = DEFAULT_YEARS, dose: str = "offered",
              rebuild: bool = False) -> RDDSample:
    """Site x program-year dose sample.

    outcome  "lunch" (lunches / (lunch days x enrollment)), "breakfast", "meals" (lunches +
             breakfasts per enrolled student per lunch day);
    dose     "offered" (default; A = s(Q, py), deterministic) or "received" (CEP x implied
             claiming share; NOT a function of Q: adoption is chosen and the share uses group /
             cycle ISPs).
    """
    years = tuple(years)
    d = sample(panel(rebuild), years, outcome)
    X = design(d)
    A = d.A.to_numpy(float) if dose == "offered" else d.A_received.to_numpy(float)
    ex = {
        "A": A,
        "a_ref": 0.0,
        "dose_levels": None,
        "schedule": schedule,
        "group": d.py.to_numpy(),
        "regime": threshold(d.py.to_numpy()),
        "Z_names": [z for z in Z_NAMES if z in X.columns],
        "ce_id": d.ceid.to_numpy(),
        "site_id": (d.ceid + "-" + d.siteid).to_numpy(),
        "campus_id": d.campus_id.to_numpy(),
        "py": d.py.to_numpy(),
        "cep": (d.cep == "Y").to_numpy(),
        "A_offered": d.A.to_numpy(float),
        "A_received": d.A_received.to_numpy(float),
        "s_impl": d.s_impl.to_numpy(float),
        "isp_grp_max4": d.isp_grp_max4.to_numpy(float),
        "isp_ce": d.isp_ce.to_numpy(float),
        "enrollment": d.enr_avg.to_numpy(float),
        "y_lunch": d.y_lunch.to_numpy(float),
        "y_breakfast": d.y_breakfast.to_numpy(float),
        "y_meals": d.y_meals.to_numpy(float),
        "df": d,
    }
    return RDDSample(
        Q=d.isp.to_numpy(float), X=X.to_numpy(float), Y=d[f"y_{outcome}"].to_numpy(float),
        threshold=0.40, name="cep_meals", feature_names=list(X.columns),
        treatment_rule=lambda q, g=d.py.to_numpy(): (np.asarray(q) >= threshold(g)).astype(int),
        description=(f"Texas CEP, school sites x program years {min(years)}-{max(years)}; Q = site ISP; "
                     f"A = {dose} free claiming share 1{{Q >= c_t}} min(1, 1.6 Q), c_t = 0.40 / 0.25 "
                     f"(from 2024-25); Y = {outcome} participation"),
        citation=("Texas Department of Agriculture, School Nutrition Programs site participation and meal "
                  "reimbursement datasets (data.texas.gov), PY 2015-16..2026-27; TEA AskTED; 7 CFR 245.9(f)."),
        extras=ex,
    )


def load() -> RDDSample:
    """Registry entry point: the default dose sample (D = 1{ISP >= threshold of the year})."""
    return load_dose()


# ----------------------------------------------------------------------------- checks
def rule_check(tol: float = 0.01) -> pd.DataFrame:
    """CEP sites: claiming share implied by the lunch reimbursement vs the formula.

    site:  min(1, 1.6 x site ISP);
    group: min(1, 1.6 x max over t-3..t of the enrollment-weighted ISP of the CE's CEP sites)
           (the pooled group ISP and the cycle's "higher of" rule, approximated; CE-level, using
           the median implied share of the CE's CEP sites).
    """
    d = panel()
    rows = []
    c = d[(d.cep == "Y") & d.isp.notna() & (d.lunch > 100)]
    for py, g in c.groupby("py"):
        e = g.s_impl - g.rule_site
        ce = g.drop_duplicates("ceid")
        ce = ce[ce.s_ce.notna()]
        eg = ce.s_ce - ce.rule_grp
        rows.append({
            "py": py, "cep_sites": len(g), "site_within_tol": float((e.abs() <= tol).mean()),
            "site_within_3pt": float((e.abs() <= 0.03).mean()), "site_implied_above": float((e > 0.03).mean()),
            "site_implied_below": float((e < -0.03).mean()), "ce_years": len(ce),
            "ce_one_share": float(g.groupby("ceid").s_impl.agg(lambda s: s.max() - s.min() <= 0.005).mean()),
            "group_within_tol": float((eg.abs() <= tol).mean()), "group_within_3pt": float((eg.abs() <= 0.03).mean()),
            "implied_ge_current_group": float((ce.s_ce >= np.minimum(1, MULT * ce.isp_grp) - tol).mean()),
            "implied_eq_1": float((g.s_impl >= 0.995).mean()),
        })
    return pd.DataFrame(rows)


def adoption(edges=(0, .15, .20, .25, .30, .35, .40, .45, .50, .55, .625, .70, .80, 1.01), level: str = "site"
             ) -> pd.DataFrame:
    """CEP share by ISP bin (rows) and program year (columns). level="site": share of sites with
    CEP; level="ce": share of CEs electing CEP for all sites, by the CE's enrollment-weighted ISP."""
    p = read_participation()
    p = p[p.typeoforg.isin(["Public", "Charter"]) & p.isp.notna()]
    if level == "ce":
        p = p.assign(cepY=(p.cep == "Y").astype(float))
        p = p.groupby(["py", "ceid"]).agg(isp=("isp", "mean"), cepY=("cepY", "min")).reset_index()
        p["cep"] = np.where(p.cepY == 1, "Y", "N")
    b = pd.cut(p.isp, list(edges), right=False)
    return p.assign(bin=b, y=(p.cep == "Y").astype(float)).pivot_table(index="bin", columns="py", values="y",
                                                                         aggfunc="mean")


def readiness(s: Optional[RDDSample] = None, n_dec: int = 10) -> dict:
    """R^2 of Q on X, sd(eta), share at dose 0 and at the cap, eta quantiles by dose decile, bunching."""
    s = s or load_dose()
    Q, A = s.Q, s.extras["A"]
    Xc = np.column_stack([np.ones(len(Q)), s.X])
    beta, *_ = np.linalg.lstsq(Xc, Q, rcond=None)
    T = Xc @ beta
    eta = Q - T
    pos = A > 0
    cuts = np.unique(np.quantile(A[pos], np.linspace(0, 1, n_dec + 1)))
    by = [("dose 0", int((~pos).sum()), *np.quantile(eta[~pos], [0.1, 0.5, 0.9]))]
    for j in range(len(cuts) - 1):
        m = pos & (A >= cuts[j]) & ((A < cuts[j + 1]) if j < len(cuts) - 2 else (A <= cuts[j + 1]))
        if m.sum():
            by.append((f"{cuts[j]:.3f}-{cuts[j + 1]:.3f}", int(m.sum()), *np.quantile(eta[m], [0.1, 0.5, 0.9])))
    g = s.extras["group"]
    return {"n": len(Q), "p": s.p, "R2": float(1 - eta.var() / Q.var()), "sd_Q": float(Q.std()),
            "sd_eta": float(eta.std()), "sd_T": float(T.std()), "share_dose0": float((~pos).mean()),
            "share_cap": float((A >= 1 - 1e-12).mean()), "by_dose": by,
            "bunching_pre_40": bunching(Q[g <= 2024], (0.30, 0.35, 0.40, 0.45, 0.50)),
            "bunching_post_25": bunching(Q[g >= 2025], (0.20, 0.25, 0.30, 0.35, 0.40))}


def bunching(q: np.ndarray, points, width: float = 0.01) -> dict:
    """Count in [x, x + w) over count in [x - w, x) at the threshold and at placebo points."""
    q = np.asarray(q, float)
    out = {}
    for x in points:
        hi = ((q >= x) & (q < x + width)).sum()
        lo = ((q >= x - width) & (q < x)).sum()
        out[x] = (int(lo), int(hi), float(hi / lo) if lo else np.nan)
    return out
