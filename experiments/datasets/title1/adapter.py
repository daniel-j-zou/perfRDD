"""ESEA Title I-A Targeted Grants: the formula weight on the SAIPE child-poverty share.

Dose design. A school district's Targeted Grant is its weighted count of formula children
times the state expenditure factor, ratably reduced to the appropriation (ESEA sec. 1125;
CRS R45141, Table 2). Formula children are the Census SAIPE estimate of children aged
5-17 in poverty (plus small counts of neglected/delinquent, foster and TANF children that
are not in the public files). With q = poverty share of the 5-17 population and N = the
5-17 population, the two weighted counts are

    percentage scale   N * g(q),  g piecewise linear with slopes 1, 1.75, 2.5, 3.25, 4
                       on [0, 15.58%], [15.58, 22.11], [22.11, 30.16], [30.16, 38.24], > 38.24%
    number scale       h(qN),     slopes 1, 1.5, 2, 2.5, 3 on [0, 691], [691, 2262],
                       [2262, 7851], [7851, 35514], > 35514 formula children

and the district gets the larger. It is eligible only if q >= 5% and qN >= 10.

``A`` (``dose="weight"``, the default) is the percentage-scale weighted count per
school-age child, ``g(q) 1{q >= 0.05}``: zero below 5%, a jump to 0.05 at 5%, then convex
and piecewise linear in q. It is a function of the score alone. The default sample keeps
districts with N >= 200, so the 10-child rule never binds above 5%. ``dose="max"`` uses
the larger of the two scales (the number scale binds for large districts); ``dose=
"dollars"`` multiplies the weight by the state expenditure factor, 0.4 x the state's
current spending per pupil bounded to [0.8, 1.2] x the national figure (F-33, third
preceding year), i.e. authorized dollars per school-age child before the ratable
reduction, which is common to all districts in a year.

Timing. Title I allocations for ED fiscal year t-1 (school year t-1 to t; Census F-33
fiscal year t) use SAIPE income-year t-3 estimates. The panel year ``year`` is the F-33
fiscal year t in which the money is received. Covariates are from F-33 year t-1 and CCD
fall 2014; the outcome is F-33 spending in year t (``lead=0``) or t+1.

Actual allocations are fuzzy: F-33 item C14 (Title I revenue through the state) pools the
four Part A grants (Basic, Concentration, Targeted, Education Finance Incentive), the
85-95% hold-harmless guarantees, state re-allocations (small-LEA alternative poverty data,
charter LEAs, boundary changes) and carryover. ``rule_check`` reports how well the formula
dose predicts it.
"""
from __future__ import annotations

import io
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample

RAW = Path(__file__).parent / "data" / "raw"

# --------------------------------------------------------------------------- statute
ELIG_SHARE = 0.05                      # Targeted Grant eligibility: q >= 5% ...
ELIG_COUNT = 10                        # ... and at least 10 formula children
PCT_CUTS = np.array([0.1558, 0.2211, 0.3016, 0.3824])
PCT_W = np.array([1.0, 1.75, 2.5, 3.25, 4.0])
NUM_CUTS = np.array([691.0, 2262.0, 7851.0, 35514.0])
NUM_W = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
APPE_RATE, APPE_LO, APPE_HI = 0.40, 0.80, 1.20
KINKS = (ELIG_SHARE,) + tuple(PCT_CUTS)
BASIC_SHARE = 0.02                     # Basic Grant: q > 2% and >= 10 children
CONC_SHARE, CONC_COUNT = 0.15, 6500    # Concentration Grant: q > 15% or > 6,500 children

DEFAULT_YEARS = (2016, 2017, 2018, 2019)   # F-33 fiscal years (pre-COVID, pre-ESSER)
SAIPE_LAG = 3                              # SAIPE income year = year - 3
STATES = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT",
    "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL",
    "18": "IN", "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME", "24": "MD",
    "25": "MA", "26": "MI", "27": "MN", "28": "MS", "29": "MO", "30": "MT", "31": "NE",
    "32": "NV", "33": "NH", "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
    "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
    "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA", "54": "WV",
    "55": "WI", "56": "WY",
}
LOCALES = ("11", "12", "13", "21", "22", "23", "31", "32", "33", "41", "42", "43")
EDGE_FILES = {   # school year -> (zip, member); pipe-delimited, no header; LEAID col 0, LOCALE col 23
    2017: ("EDGE_GEOCODE_PUBLICLEA_1617.zip", "EDGE_GEOCODE_PUBLICLEA_1617/EDGE_GEOCODE_PUBLICLEA_1617.TXT"),
    2018: ("EDGE_GEOCODE_PUBLICLEA_1718.zip", "EDGE_GEOCODE_PUBLICLEA_1718/EDGE_GEOCODE_PUBLICLEA_1718.TXT"),
    2019: ("EDGE_GEOCODE_PUBLICLEA_1819.zip", "EDGE_GEOCODE_PUBLICLEA_1819.TXT"),
}
CCD_MEMBERSHIP = {  # fall of the school year -> (zip, member, separator)
    2014: ("ccd_lea_052_1415_w_0216161a_txt.zip", "ccd_lea_052_1415_w_0216161a.txt", "\t"),
    2015: ("ccd_lea_052_1516_w_1a_011717_csv.zip", "ccd_lea_052_1516_w_1a_011717.csv", ","),
}


# --------------------------------------------------------------------------- schedule
def _stepwise(x, cuts, w) -> np.ndarray:
    """Integral of a step weight: sum_k w_k * |[0, x] intersected with segment k|."""
    edges = np.concatenate([[0.0], cuts, [np.inf]])
    x = np.asarray(x, float)[..., None]
    seg = np.clip(x - edges[:-1], 0.0, edges[1:] - edges[:-1])
    return seg @ w


def pct_weight(q) -> np.ndarray:
    """Percentage-scale weighted formula children per school-age child, g(q)."""
    return _stepwise(q, PCT_CUTS, PCT_W)


def num_weighted(c) -> np.ndarray:
    """Number-scale weighted count h(c) for c formula children."""
    return _stepwise(c, NUM_CUTS, NUM_W)


def targeted_weight(q, pop517=None, scale: str = "pct") -> np.ndarray:
    """Weighted formula children per school-age child, 0 if not eligible.

    ``scale="pct"``: g(q). ``scale="max"``: max(g(q), h(q N) / N), needs ``pop517`` = N.
    With ``pop517`` the 10-child rule is applied as well.
    """
    q = np.asarray(q, float)
    w = pct_weight(q)
    elig = q >= ELIG_SHARE - 1e-12
    if pop517 is not None:
        n = np.asarray(pop517, float)
        c = q * n
        elig = elig & (c >= ELIG_COUNT - 1e-9)
        if scale == "max":
            w = np.maximum(w, num_weighted(c) / np.maximum(n, 1.0))
    elif scale == "max":
        raise ValueError("scale='max' needs pop517")
    return np.where(elig, w, 0.0)


def make_schedule(dose: str, factors: Optional[Dict[str, float]] = None):
    """``schedule(q, group, pop517=None) -> dose`` in the units of ``load_dose``.

    ``group`` is "<ST>_<year>" (e.g. "MI_2017"); it matters only for ``dose="dollars"``
    (the state expenditure factor). ``dose="max"`` needs ``pop517``.
    """
    def schedule(q, group, pop517=None):
        q = np.asarray(q, float)
        if dose == "weight":
            return targeted_weight(q)
        if dose == "max":
            return targeted_weight(q, pop517, scale="max")
        if dose == "dollars":
            f = np.array([factors[g] for g in np.atleast_1d(group)], float)
            return targeted_weight(q) * (f if q.ndim else f[0])
        raise ValueError(dose)
    return schedule


# --------------------------------------------------------------------------- readers
def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run: python -m experiments.datasets.title1.download")
    return path


@lru_cache(maxsize=None)
def saipe(y: int) -> pd.DataFrame:
    """SAIPE school-district estimates, income year y (fixed-width; layout in data/raw/saipe)."""
    p = _require(RAW / "saipe" / f"ussd{y % 100:02d}.txt")
    d = pd.read_fwf(p, colspecs=[(0, 2), (3, 8), (9, 81), (82, 90), (91, 99), (100, 108)],
                    names=["fips", "id", "saipe_name", "pop", "pop517", "pov517"],
                    dtype={"fips": str, "id": str}, encoding="latin1")
    d["leaid"] = d["fips"] + d["id"]
    d = d[d["fips"].isin(STATES)]
    return d[["leaid", "fips", "saipe_name", "pop", "pop517", "pov517"]].reset_index(drop=True)


F33_ITEMS = {"V33": "enroll", "C14": "title1", "TFEDREV": "fedrev", "TSTREV": "strev",
             "TLOCREV": "locrev", "TOTALREV": "totrev", "TCURELSC": "cur", "TCURINST": "inst",
             "TOTALEXP": "totexp"}


@lru_cache(maxsize=None)
def f33(y: int) -> pd.DataFrame:
    """F-33 school-system finances, Census fiscal year y; dollars (file is in $1,000)."""
    p = _require(RAW / "f33" / f"elsec{y % 100:02d}.txt")
    d = pd.read_csv(p, dtype=str, encoding="latin1", index_col=False)
    # STATE is the Census Bureau's own state code (alphabetical), not FIPS; the first two
    # digits of the NCES LEA id are the FIPS code.
    out = pd.DataFrame({"leaid": d["NCESID"].str.strip(), "census_state": d["STATE"],
                        "schlev": d["SCHLEV"], "f33_name": d["NAME"]})
    out["fips"] = out["leaid"].str[:2]
    for k, v in F33_ITEMS.items():
        x = pd.to_numeric(d[k], errors="coerce")
        out[v] = x if k == "V33" else 1000.0 * x
    return out


@lru_cache(maxsize=None)
def appe_factor(y: int) -> pd.Series:
    """0.4 x state current spending per pupil, bounded to [0.8, 1.2] x national (F-33 year y).

    Proxy for ED's state average per-pupil expenditure (current expenditure per pupil in
    average daily attendance, third preceding year); here per pupil in fall membership.
    """
    d = f33(y)
    d = d[d["fips"].isin(STATES) & (d["enroll"] > 0)]
    s = d.groupby("fips")[["cur", "enroll"]].sum()
    appe = s["cur"] / s["enroll"]
    nat = s["cur"].sum() / s["enroll"].sum()
    return APPE_RATE * appe.clip(APPE_LO * nat, APPE_HI * nat)


@lru_cache(maxsize=None)
def race_shares() -> pd.DataFrame:
    """CCD fall-2014 membership shares by race (fall 2015 where 2014 is missing)."""
    frames = []
    for yr in (2014, 2015):
        zname, member, sep = CCD_MEMBERSHIP[yr]
        with zipfile.ZipFile(_require(RAW / "ccd" / zname)) as z:
            d = pd.read_csv(io.BytesIO(z.read(member)), sep=sep, dtype={"LEAID": str},
                            usecols=["LEAID", "MEMBER", "AM", "AS", "HI", "BL", "WH", "HP", "TR"],
                            encoding="latin1")
        num = d[["MEMBER", "AM", "AS", "HI", "BL", "WH", "HP", "TR"]].apply(pd.to_numeric, errors="coerce")
        races = num[["AM", "AS", "HI", "BL", "WH", "HP", "TR"]]
        tot = races.sum(axis=1)
        ok = (races >= 0).all(axis=1) & (tot > 0)
        r = pd.DataFrame({
            "leaid": d["LEAID"].str.zfill(7),
            "sh_black": num["BL"] / tot, "sh_hisp": num["HI"] / tot, "sh_asian": num["AS"] / tot,
            "sh_other": (num["AM"] + num["HP"] + num["TR"]) / tot,
            "ccd_member": num["MEMBER"], "race_year": yr})
        frames.append(r[ok.values])
    r = pd.concat(frames).drop_duplicates("leaid", keep="first")
    return r.set_index("leaid")


@lru_cache(maxsize=None)
def locale(sy: int) -> pd.Series:
    """NCES locale code (11-43) by LEAID, EDGE geocode file for school year ending ``sy``."""
    sy = min(max(sy, min(EDGE_FILES)), max(EDGE_FILES))
    zname, member = EDGE_FILES[sy]
    with zipfile.ZipFile(_require(RAW / "edge" / zname)) as z:
        d = pd.read_csv(io.BytesIO(z.read(member)), sep="|", header=None, dtype=str,
                        usecols=[0, 23], encoding="latin1")
    d.columns = ["leaid", "locale"]
    return d.drop_duplicates("leaid").set_index("leaid")["locale"].str.strip()


# --------------------------------------------------------------------------- panel
def year_frame(t: int, lead: int = 0) -> pd.DataFrame:
    """One row per SAIPE district matched to F-33, for receipt year t (F-33 fiscal year)."""
    s = saipe(t - SAIPE_LAG).copy()
    s = s[s["pop517"] > 0]
    s["q"] = s["pov517"] / s["pop517"]
    for k in (1, 2):                                    # earlier SAIPE estimates
        p = saipe(t - SAIPE_LAG - k).set_index("leaid")
        s[f"q_lag{k}"] = s["leaid"].map(p["pov517"] / p["pop517"].where(p["pop517"] > 0))
    cur, lag, out = f33(t), f33(t - 1).set_index("leaid"), f33(t + lead).set_index("leaid")
    d = s.merge(cur.drop(columns="fips"), on="leaid", how="inner")
    d["year"] = t
    d["saipe_year"] = t - SAIPE_LAG
    d["state"] = d["fips"].map(STATES)
    d["group"] = d["state"] + "_" + str(t)
    for v in ("enroll", "cur", "inst", "title1", "fedrev"):
        d[f"{v}_lag"] = d["leaid"].map(lag[v])
    for v in ("enroll", "cur", "inst", "title1", "fedrev", "totrev"):
        d[f"{v}_out"] = d["leaid"].map(out[v])
    d["appe_factor"] = d["fips"].map(appe_factor(t - SAIPE_LAG))
    d["locale"] = d["leaid"].map(locale(t - 1))
    d = d.join(race_shares(), on="leaid")
    return d


@lru_cache(maxsize=None)
def _panel(years: tuple, lead: int) -> pd.DataFrame:
    d = pd.concat([year_frame(t, lead) for t in years], ignore_index=True)
    d["A_weight"] = targeted_weight(d["q"])
    d["A_max"] = targeted_weight(d["q"], d["pop517"], scale="max")
    d["A_dollars"] = d["A_weight"] * d["appe_factor"]
    d["num_scale_binds"] = d["A_max"] > d["A_weight"] + 1e-12
    pos = d["enroll"] > 0
    d["title1_pp"] = np.where(pos, d["title1"] / d["enroll"].where(pos), np.nan)
    d["title1_per_child"] = d["title1"] / d["pop517"]
    for v in ("cur", "inst", "title1", "fedrev"):
        d[f"{v}_pp_lag"] = d[f"{v}_lag"] / d["enroll_lag"].where(d["enroll_lag"] > 0)
    for v in ("cur", "inst", "title1", "fedrev", "totrev"):
        d[f"{v}_pp_out"] = d[f"{v}_out"] / d["enroll_out"].where(d["enroll_out"] > 0)
    return d


def panel(years: Sequence[int] = DEFAULT_YEARS, lead: int = 0) -> pd.DataFrame:
    """District x year frame with every variable (no sample restrictions beyond the merge)."""
    return _panel(tuple(years), lead).copy()


def analysis_frame(years: Sequence[int] = DEFAULT_YEARS, lead: int = 0, min_pop517: int = 200,
                   min_enroll: int = 100, schlev: Sequence[str] = ("01", "02", "03"),
                   pp_range=(3000.0, 60000.0)) -> pd.DataFrame:
    """Sample used by ``load_dose``: regular districts, N >= 200, enrolment >= 100 in t-1
    and t, per-pupil current spending in ``pp_range`` in t-1 and t+lead, X non-missing."""
    d = panel(years, lead)
    keep = (d["schlev"].isin(schlev) & (d["pop517"] >= min_pop517)
            & (d["enroll_lag"] >= min_enroll) & (d["enroll"] >= min_enroll)
            & (d["enroll_out"] >= min_enroll)
            & d["cur_pp_lag"].between(*pp_range) & d["cur_pp_out"].between(*pp_range)
            & d["sh_black"].notna() & d["q_lag1"].notna())
    return d[keep].reset_index(drop=True)


X_CONT = ["log_enroll_lag", "log_cur_pp_lag", "sh_black", "sh_hisp", "sh_asian", "sh_other"]
Z_NAMES = ["log_enroll_lag", "log_cur_pp_lag", "sh_black", "sh_hisp"]
OUTCOMES = {
    "cur_pp": "current spending per pupil, F-33 year t+lead (dollars)",
    "inst_pp": "instructional current spending per pupil, F-33 year t+lead",
    "log_cur_pp": "log current spending per pupil",
    "d_cur_pp": "current spending per pupil, year t+lead minus year t-1",
    "title1_pp": "Title I revenue per pupil (F-33 C14), year t+lead: the first stage",
    "fedrev_pp": "federal revenue per pupil, year t+lead",
}


def design(d: pd.DataFrame, lag_q: bool = False):
    """X matrix and names: continuous covariates, locale, school level, state, year dummies."""
    cols = {
        "log_enroll_lag": np.log(d["enroll_lag"]),
        "log_cur_pp_lag": np.log(d["cur_pp_lag"]),
        "sh_black": d["sh_black"], "sh_hisp": d["sh_hisp"],
        "sh_asian": d["sh_asian"], "sh_other": d["sh_other"],
    }
    if lag_q:
        cols["q_lag1"] = d["q_lag1"]
    X = pd.DataFrame(cols)
    loc = d["locale"].where(d["locale"].isin(LOCALES), "na")
    for c in LOCALES[1:] + ("na",):
        if (loc == c).any():
            X[f"locale_{c}"] = (loc == c).astype(float)
    X["schlev_elem"] = (d["schlev"] == "01").astype(float)
    X["schlev_sec"] = (d["schlev"] == "02").astype(float)
    states = sorted(d["state"].unique())
    for s in states[1:]:
        X[f"st_{s}"] = (d["state"] == s).astype(float)
    for t in sorted(d["year"].unique())[1:]:
        X[f"yr_{t}"] = (d["year"] == t).astype(float)
    X = X.loc[:, X.std() > 0] if len(X) > 1 else X
    return X.to_numpy(float), list(X.columns)


def _outcome(d: pd.DataFrame, outcome: str) -> np.ndarray:
    if outcome == "cur_pp":
        return d["cur_pp_out"].to_numpy(float)
    if outcome == "inst_pp":
        return d["inst_pp_out"].to_numpy(float)
    if outcome == "log_cur_pp":
        return np.log(d["cur_pp_out"].to_numpy(float))
    if outcome == "d_cur_pp":
        return (d["cur_pp_out"] - d["cur_pp_lag"]).to_numpy(float)
    if outcome == "title1_pp":
        return d["title1_pp_out"].to_numpy(float)
    if outcome == "fedrev_pp":
        return d["fedrev_pp_out"].to_numpy(float)
    raise ValueError(f"outcome must be one of {sorted(OUTCOMES)}")


def load_dose(years: Sequence[int] = DEFAULT_YEARS, outcome: str = "cur_pp", dose: str = "weight",
              lead: int = 0, lag_q: bool = False, **sample_kw) -> RDDSample:
    """Dose sample: Q = SAIPE poverty share (income year t-3), A = formula Targeted weight.

    ``dose``: "weight" (default; g(q) 1{q >= 5%}, a function of q only), "max" (larger of the
    percentage and number scales; needs pop517), "dollars" (weight x state APPE factor).
    ``lag_q=True`` adds the SAIPE share one year earlier to X.
    """
    d = analysis_frame(years, lead, **sample_kw)
    X, names = design(d, lag_q=lag_q)
    factors = None
    if dose == "dollars":
        factors = dict(zip(d["group"], d["appe_factor"]))
    schedule = make_schedule(dose, factors)
    Q = d["q"].to_numpy(float)
    grp = d["group"].to_numpy()
    pop = d["pop517"].to_numpy(float)
    A = schedule(Q, grp, pop) if dose == "max" else schedule(Q, grp)
    return RDDSample(
        Q=Q, X=X, Y=_outcome(d, outcome), threshold=ELIG_SHARE, name="title1",
        feature_names=names,
        description=(
            "ESEA Title I-A Targeted Grants, school districts, F-33 years "
            f"{min(years)}-{max(years)}. Q = SAIPE share of children 5-17 in poverty (income "
            "year t-3); dose = formula weighted children per school-age child, zero below 5%, "
            f"convex piecewise linear with kinks at {', '.join(f'{100 * k:.2f}%' for k in PCT_CUTS)}. "
            f"Y = {OUTCOMES[outcome]}."),
        citation=("U.S. Census Bureau, SAIPE school district estimates and Annual Survey of "
                  "School System Finances (F-33); NCES Common Core of Data and EDGE; ESEA "
                  "sec. 1125; CRS R45141 (Title I-A formulas)."),
        treatment_rule=lambda q: (np.asarray(q) >= ELIG_SHARE).astype(int),
        extras={
            "A": A, "a_ref": 0.0, "dose_levels": None, "schedule": schedule, "dose": dose,
            "group": grp, "state": d["state"].to_numpy(), "year": d["year"].to_numpy(),
            "Z_names": [z for z in Z_NAMES if z in names], "leaid": d["leaid"].to_numpy(),
            "published_dose": None,   # ED LEA allocations by grant: see README, AUTHOR TO-DO
            "title1_pp": d["title1_pp"].to_numpy(float),
            "title1_per_child": d["title1_per_child"].to_numpy(float),
            "A_weight": d["A_weight"].to_numpy(float), "A_max": d["A_max"].to_numpy(float),
            "A_dollars": d["A_dollars"].to_numpy(float),
            "num_scale_binds": d["num_scale_binds"].to_numpy(bool),
            "pop517": pop, "pov517": d["pov517"].to_numpy(float),
            "appe_factor": d["appe_factor"].to_numpy(float),
            "q_lag1": d["q_lag1"].to_numpy(float),
            "kinks": KINKS, "outcome": outcome, "lead": lead, "frame": d,
        },
    )


def load() -> RDDSample:
    """Registry default: the dose sample (D = 1{q >= 5%}, Targeted Grant eligibility)."""
    return load_dose()


# --------------------------------------------------------------------------- rule check
def _r2(y, cols, fe=None) -> float:
    """R^2 of OLS of y on an intercept, ``cols`` and dummies for ``fe``."""
    blocks = [np.ones((len(y), 1))] + [np.asarray(c, float).reshape(len(y), -1) for c in cols]
    if fe is not None:
        codes = pd.factorize(pd.Series(fe))[0]
        blocks.append(np.eye(codes.max() + 1)[codes][:, 1:])
    D = np.column_stack(blocks)
    b = np.linalg.lstsq(D, y, rcond=None)[0]
    return float(1 - np.var(y - D @ b) / np.var(y))


def rule_check(years: Sequence[int] = DEFAULT_YEARS) -> pd.DataFrame:
    """How well the formula dose predicts actual Title I revenue per school-age child.

    Rows: R^2 of F-33 Title I revenue (year t) per school-age child on (a) the poverty
    share q (linear), (b) the Targeted weight A, (c) A with the number scale, (d) a proxy
    for all four Part A formulas (Basic q 1{q > 2%}, Concentration q 1{q > 15% or > 6,500},
    Targeted/EFIG weight), each times the state factor, (e) last year's Title I revenue
    (hold harmless), all with year dummies; and the correlation of A with the outcome.
    """
    d = analysis_frame(years, lead=0)
    y = d["title1_per_child"].to_numpy(float)
    q = d["q"].to_numpy(float)
    n = d["pop517"].to_numpy(float)
    f = d["appe_factor"].to_numpy(float) / d["appe_factor"].mean()
    c = q * n
    basic = q * ((q > BASIC_SHARE) & (c >= ELIG_COUNT))
    conc = q * ((q > CONC_SHARE) | (c > CONC_COUNT))
    lag = (d["title1_lag"] / d["pop517"]).to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(lag)
    rows = []

    def add(label, sub, cols, fe):
        rows.append({"model": label, "n": int(sub.sum()),
                     "R2": round(_r2(y[sub], [x[sub] for x in cols], fe[sub] if fe is not None else None), 3)})

    yr = d["year"].to_numpy()
    sy = d["group"].to_numpy()
    A, Am = d["A_weight"].to_numpy(float), d["A_max"].to_numpy(float)
    add("q (linear) + year", ok, [q], yr)
    add("A = g(q) 1{q>=5%} + year", ok, [A], yr)
    add("A + q + year", ok, [A, q], yr)
    add("A_max (number scale) + q + year", ok, [Am, q], yr)
    add("A x state factor + year", ok, [A * f], yr)
    add("Part A proxy (Basic, Conc, Targeted) x factor + year", ok, [basic * f, conc * f, Am * f], yr)
    add("Part A proxy + state-year", ok, [basic, conc, Am], sy)
    add("lagged Title I (hold harmless) + year", ok, [lag], yr)
    add("lagged Title I + Part A proxy + state-year", ok, [lag, basic, conc, Am], sy)
    out = pd.DataFrame(rows)
    out.attrs["corr_A_actual"] = float(np.corrcoef(A[ok], y[ok])[0, 1])
    out.attrs["corr_A_actual_pp"] = float(np.corrcoef(A[ok], d["title1_pp"].to_numpy(float)[ok])[0, 1])
    return out


def timing_check(years: Sequence[int] = DEFAULT_YEARS) -> pd.DataFrame:
    """Correlation of F-33 Title I per school-age child (year t) with the formula weight
    built from SAIPE income year t-k, k = 1..4 (same districts; the formula uses k = 3)."""
    d = analysis_frame(years, lead=0)
    rows = []
    for k in (1, 2, 3, 4):
        ser = []
        for t in sorted(d["year"].unique()):
            y = t - k
            if not (RAW / "saipe" / f"ussd{y % 100:02d}.txt").exists():
                continue
            s = saipe(y).set_index("leaid")
            m = d["year"] == t
            share = d.loc[m, "leaid"].map(s["pov517"] / s["pop517"].where(s["pop517"] > 0))
            ser.append(share)
        if not ser:
            continue
        share = pd.concat(ser).reindex(d.index)
        A = targeted_weight(share.to_numpy(float))
        y = d["title1_per_child"].to_numpy(float)
        ok = np.isfinite(A) & np.isfinite(y) & share.notna().to_numpy()
        rows.append({"saipe_year": f"t-{k}", "n": int(ok.sum()),
                     "corr": round(float(np.corrcoef(A[ok], y[ok])[0, 1]), 3),
                     "R2_with_year": round(_r2(y[ok], [A[ok], share.to_numpy(float)[ok]],
                                               d["year"].to_numpy()[ok]), 3)})
    return pd.DataFrame(rows)
