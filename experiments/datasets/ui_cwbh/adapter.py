"""UI weekly benefit schedules in the CWBH data (5 states, 1976-1984), from Landais (2015).

Dose design. The weekly benefit amount (WBA) is a published function of the claimant's
earnings in the base period, linear between a floor and a cap:

    WBA = clip(round(s / frac), min_b(t), max_b(t)),

with s the highest-quarter earnings (LA, ID, MO, NM) or the average of the two highest
quarters (WA), and frac, min_b, max_b set by state law and changed over time. The values
below are those in Landais's state programs (``Programs/prog_<state>.do``), keyed on the
date the benefit year began:

    LA  frac 25;        min 10;            max 141, 149 (2 Sep 79), 164 (7 Sep 80), 183 (6 Sep 81), 205 (5 Sep 82)
    ID  frac 26;        min 17, 36 (Jul 80); max 90, 99 (4 Jul 76), 110, 116, 121, 132, 145 (1 Jul 77..81), 159 (20 Jun 82)
    MO  frac 20, 22.22 (2 Dec 79); min 15;  max 85, 105 (2 Dec 79)
    NM  frac 26;        min 18..29 (each Jan); max 90..145 (each Jan)
    WA  frac 25 on the top-two-quarter average; min 17..51, max 126..185 (each Jul)

WA reports only base-period wages (bpw) and the highest quarter (hqw); as Landais does, the
second quarter is proxied by (bpw - hqw) / 3, so the WA score is measured with error.
Rounding was read from the data (see ``rule_check``).

Sample (Landais's ``cleaning_cwbh_v1.do`` and the state programs): monetarily eligible,
regular state UI, new claims, no denial, no separation disqualification (not recorded in
LA), benefit year started before 1 Jan 1984 and after the first certified week in the file,
claims without an interruption of more than three weeks. One row per claim (id x benefit
year). Landais's ``kink.do`` also drops claims at the cap below the kink and claims off
the cap above it (``on_schedule``).

Reference dose: the cap, as in ``ui_upjohn``. Q = s / s_cap(state, period), so the cap
kink is at 1 in every schedule; ``dose="real"`` instead measures the score and the dose in
1984 dollars, so that the 25 state-period schedules differ (several schedules).

Data use: the readme asks users to contact Bruce Meyer or Patricia Anderson for a formal
agreement to use the CWBH files. The openICPSR deposit (Landais 2015, project 114581)
carries CC BY 4.0 for data; confirm before publishing results.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw" / "Prog_AEJ_RKD" / "Data"
PROC = HERE / "data" / "processed"
STATES = ("LA", "ID", "MO", "NM", "WA")

COLS = ["id", "yr", "rectype", "r2v18", "r2v20", "r2v21", "r3v7", "r4v9", "r3v17", "r2v31", "r4v16",
        "r2v51", "r2v52", "r2v23", "r2v24", "r2v25", "r2v27", "r2v28", "r2v30", "r2v36", "r2v10", "r2v11",
        "r2v12", "r2v13", "r2v14", "r3v13", "r3v25", "r3v26", "r3v27", "r3v29", "r3v32", "r3v33", "r3v39",
        "r3v10"]

# (start date of the benefit year, value). Sorted; the first value applies before the first date.
Steps = List[Tuple[str, float]]
SCHED: Dict[str, Dict[str, Steps]] = {
    "LA": {"frac": [("1900-01-01", 25.0)],
           "min": [("1900-01-01", 10.0)],
           "max": [("1900-01-01", 141.0), ("1979-09-02", 149.0), ("1980-09-07", 164.0),
                   ("1981-09-06", 183.0), ("1982-09-05", 205.0)],
           "start": [("1979-01-01", 1.0)],
           "round": [("1900-01-01", 0.0), ("1983-04-03", 2.0)]},
    "ID": {"frac": [("1900-01-01", 26.0)],
           "min": [("1900-01-01", 17.0), ("1980-07-01", 36.0), ("1984-01-01", 45.0)],
           "max": [("1900-01-01", 90.0), ("1976-07-04", 99.0), ("1977-07-01", 110.0), ("1978-07-01", 116.0),
                   ("1979-07-01", 121.0), ("1980-07-01", 132.0), ("1981-07-01", 145.0), ("1982-06-20", 159.0)],
           "start": [("1976-01-17", 1.0)],
           "round": [("1900-01-01", 0.0), ("1983-07-01", 1.0)]},
    "MO": {"frac": [("1900-01-01", 20.0), ("1979-12-02", 1 / 0.045)],
           "min": [("1900-01-01", 15.0)],
           "max": [("1900-01-01", 85.0), ("1979-12-02", 105.0)],
           "start": [("1978-01-01", 1.0)],
           "round": [("1900-01-01", 0.0)]},
    "NM": {"frac": [("1900-01-01", 26.0)],
           "min": [("1900-01-01", 18.0), ("1980-01-01", 22.0), ("1981-01-01", 24.0), ("1982-01-01", 26.0),
                   ("1983-01-01", 29.0)],
           "max": [("1900-01-01", 90.0), ("1979-01-01", 98.0), ("1980-01-01", 106.0), ("1981-01-01", 117.0),
                   ("1982-01-01", 130.0), ("1983-01-01", 142.0), ("1984-01-01", 145.0)],
           "start": [("1980-01-01", 1.0)],
           "round": [("1900-01-01", 0.0)]},
    "WA": {"frac": [("1900-01-01", 25.0)],
           "min": [("1900-01-01", 17.0), ("1980-07-06", 41.0), ("1981-07-01", 45.0), ("1982-07-01", 49.0),
                   ("1983-07-01", 51.0)],
           "max": [("1900-01-01", 126.0), ("1978-01-04", 128.0), ("1979-06-25", 137.0), ("1980-07-06", 150.0),
                   ("1981-07-01", 163.0), ("1982-07-01", 178.0), ("1983-07-01", 185.0)],
           "start": [("1979-06-25", 1.0)],
           "round": [("1900-01-01", 0.0)]},
}
MAXDUR = {"LA": 28.0, "ID": 26.0, "MO": 26.0, "NM": 26.0, "WA": 30.0}  # LA 26 from 3 Apr 1983
FRAC_BPW = {"LA": 2 / 5, "ID": np.nan, "MO": 1 / 3, "NM": 3 / 5, "WA": 1 / 3}

Z_NAMES = ["age", "male", "educ", "log_bpw_ratio"]


def _yymmdd(x) -> pd.Series:
    s = pd.Series(np.asarray(x)).astype("int64").astype(str).str.zfill(6)
    return pd.to_datetime("19" + s, format="%Y%m%d", errors="coerce")


def _step(state: str, key: str, dates: pd.Series) -> np.ndarray:
    steps = SCHED[state][key]
    out = np.full(len(dates), steps[0][1], dtype=float)
    d = dates.to_numpy()
    for start, v in steps[1:]:
        out[d >= np.datetime64(start)] = v
    return out


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Unzip Data/rawdata/*.dta from openICPSR 114581 (Landais 2015) into "
            f"{RAW.parent}; see README.md")


# --------------------------------------------------------------------------- claims
def build_claims(state: str) -> pd.DataFrame:
    """One row per claim (id x benefit year) after Landais's filters, from the week records."""
    path = RAW / "rawdata" / f"cwbh_{state}.dta"
    _require(path)
    parts = []
    reader = pd.read_stata(path, columns=COLS, iterator=True, convert_categoricals=False)
    while True:
        try:
            ch = reader.read(500_000)
        except StopIteration:
            break
        if ch is None or len(ch) == 0:
            break
        keep = ((ch.rectype == 4) & (ch.r2v18 == 1) & (ch.r2v20 == 1) & (ch.r2v21 == 0) & (ch.r3v7 == 1)
                & (ch.r4v9 == 0))
        if state != "LA":
            keep &= ch.r3v17.isin([0, 9])
        parts.append(ch[keep])
    w = pd.concat(parts, ignore_index=True)
    w["week"] = _yymmdd(w.r4v16)
    w = w.sort_values(["id", "r2v31", "week"])
    same = (w.id.to_numpy()[1:] == w.id.to_numpy()[:-1]) & (w.r2v31.to_numpy()[1:] == w.r2v31.to_numpy()[:-1])
    lag = np.r_[np.nan, np.where(same, (w.week.to_numpy()[1:] - w.week.to_numpy()[:-1]) / np.timedelta64(1, "D"),
                                 np.nan)]
    w["interrup"] = lag > 21
    g = w.groupby(["id", "r2v31"], sort=False)
    c = g.first().reset_index()
    c["start_claim"] = g.week.min().to_numpy()
    c["end_claim"] = g.week.max().to_numpy()
    c["interrupted"] = g.interrup.max().to_numpy()
    c["n_weeks_rec"] = g.size().to_numpy()
    c["state"] = state
    c["first_week_in_file"] = w.week.min()
    return c


def claims(states: Sequence[str] = STATES, rebuild: bool = False) -> pd.DataFrame:
    """Claim-level frame for the states, cached in data/processed/claims_<state>.parquet."""
    PROC.mkdir(parents=True, exist_ok=True)
    out = []
    for st in states:
        f = PROC / f"claims_{st}.parquet"
        if rebuild or not f.exists():
            build_claims(st).to_parquet(f)
        out.append(pd.read_parquet(f))
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------- frame
def frame(states: Sequence[str] = STATES, rebuild: bool = False) -> pd.DataFrame:
    """Landais's analysis sample with the score, the schedule parameters and outcomes."""
    c = claims(states, rebuild)
    c["bby"] = _yymmdd(c.r2v31)
    c = c[c.bby.notna() & (c.bby < pd.Timestamp("1984-01-01")) & (c.bby >= c.first_week_in_file)
          & ~c.interrupted.astype(bool)].copy()
    parts = []
    for st, d in c.groupby("state"):
        d = d.copy()
        for k in ("frac", "min", "max", "start", "round"):
            d[k + "_b"] = _step(st, k, d.bby)
        d = d[d.bby >= pd.Timestamp(SCHED[st]["start"][0][0])]
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    hqw = d.r2v24.astype(float)
    bpw = d.r2v23.astype(float)
    d["hqw"] = hqw
    d["bpw"] = bpw
    wa = (d.state == "WA").to_numpy()
    d["s"] = np.where(wa, np.maximum(0.0, 0.5 * (hqw + (bpw - hqw) / 3.0)), hqw)
    d["wba"] = d.r2v28.astype(float)
    d["wba_aug"] = d.r2v27.astype(float)
    d["s_cap"] = d.max_b * d.frac_b
    d["s_floor"] = d.min_b * d.frac_b
    d["schedule"] = d.state + "_" + d.max_b.astype(int).astype(str) + "_" + d.min_b.astype(int).astype(str) \
        + "_" + d.frac_b.round(2).astype(str)
    d["period"] = d.groupby("state").max_b.rank(method="dense").astype(int)
    d["weeks_paid"] = d.r2v52.astype(float)
    d["weeks_claimed"] = d.r2v51.astype(float)
    d["duration"] = (d.end_claim - d.start_claim).dt.days / 7.0
    d["potdur"] = d.r2v30.astype(float)
    d["benefits_t1"] = d.r2v36.astype(float)
    age = (d.yr.astype(float) - d.r2v11.astype(float)).where(d.r2v11 != 99)
    d["age"] = age
    d["male"] = (d.r2v10 == 1).astype(float).where(~d.r2v10.isin([0, 9]))
    d["black"] = (d.r2v13 == 2).astype(float)
    d["educ"] = d.r2v12.astype(float).where(d.r2v12 != 99)
    d["dependents"] = d.r3v33.astype(float).where(d.r3v33 != 9)
    d["hh_size"] = d.r3v32.astype(float).where(d.r3v32 != 9)
    d["recall"] = (d.r3v26 == 1).astype(float)
    d["industry"] = (d.r3v13.astype(int) // 1000).clip(0, 9).astype(str)
    d["log_bpw_ratio"] = np.log(np.maximum(bpw, 1.0) / np.maximum(hqw, 1.0))
    d["weeks_worked"] = d.r2v25.astype(float)
    d["office"] = d.state + "_" + d.r2v14.astype(str)
    return d


ROUND = {0: "ceil", 1: "floor", 2: "round"}


def schedule_dollars(s, frac, lo, hi, rounding="ceil") -> np.ndarray:
    """WBA in dollars. ``rounding`` is "ceil", "floor", "round" or a per-unit array of codes (ROUND)."""
    raw = np.asarray(s, float) / np.asarray(frac, float)
    r = {"round": np.floor(raw + 0.5 - 1e-9), "floor": np.floor(raw + 1e-9), "ceil": np.ceil(raw - 1e-9)}
    if isinstance(rounding, str):
        out = r[rounding]
    else:
        code = np.asarray(rounding, int)
        out = np.select([code == 0, code == 1, code == 2], [r["ceil"], r["floor"], r["round"]])
    return np.clip(out, lo, hi)


def rule_check(d: pd.DataFrame | None = None) -> pd.DataFrame:
    """Share of claims whose reported WBA equals the formula, by state and rounding rule.

    "dated" uses the rule switch found in the data (ceil; LA rounds from 3 Apr 1983, ID
    floors from 1 Jul 1983). WA uses the proxy score, so its interior match is low.
    """
    d = frame() if d is None else d
    rows = []
    for st, g in d.groupby("state"):
        row = {"state": st, "n": len(g)}
        inner = ((g.s > g.s_floor) & (g.s < g.s_cap)).to_numpy()
        for r in ("ceil", "floor", "round", "dated"):
            a = schedule_dollars(g.s, g.frac_b, g.min_b, g.max_b, g.round_b.to_numpy() if r == "dated" else r)
            row[r] = float(np.mean(a == g.wba))
            row[r + "_interior"] = float(np.mean(a[inner] == g.wba.to_numpy()[inner]))
        row["at_cap_above_kink"] = float(np.mean(g.wba[g.s >= g.s_cap] == g.max_b[g.s >= g.s_cap]))
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- dose sample
OUTCOMES = {
    "weeks_paid": "weeks of UI paid in the benefit year (CWBH r2v52, all tiers)",
    "duration": "claimed duration: weeks from the first to the last certified week (Landais)",
    "weeks_claimed": "weeks claimed in the benefit year (r2v51)",
}


def _cpi() -> pd.DataFrame:
    c = pd.read_stata(RAW / "cpi2.dta")
    c["ym"] = pd.to_datetime(dict(year=c.year.astype(int), month=c.month2.astype(int), day=1)).dt.to_period("M")
    base = float(c.loc[c.year == 1984, "cpi"].mean())
    return pd.Series(base / c.cpi.to_numpy(float), index=c.ym)


def dose_frame(states: Sequence[str] = ("LA", "ID", "MO", "NM"), on_schedule: bool = True,
               rebuild: bool = False) -> pd.DataFrame:
    """Analysis sample with the dose. WA (if asked for) gets Landais's score imputation:
    s = frac x WBA for claims strictly between the floor and the cap, the proxy at the cap,
    and claims at the cap with a proxy below the kink are dropped."""
    d = frame(states, rebuild)
    d = d[d.s > 0].copy()
    wa = (d.state == "WA").to_numpy()
    interior = (d.wba > d.min_b) & (d.wba < d.max_b)
    d["score_imputed"] = wa & interior.to_numpy()
    d.loc[d.score_imputed, "s"] = d.loc[d.score_imputed, "frac_b"] * d.loc[d.score_imputed, "wba"]
    d["wba_formula"] = schedule_dollars(d.s, d.frac_b, d.min_b, d.max_b, d.round_b.to_numpy())
    d["on_schedule"] = d.wba_formula == d.wba
    if on_schedule:
        d = d[d.on_schedule]
    defl = _cpi()
    d["defl"] = defl.reindex(d.bby.dt.to_period("M")).to_numpy()
    d["s_real"] = d.s * d.defl
    d["wba_real"] = d.wba_formula * d.defl
    d["max_real"] = d.max_b * d.defl
    return d.reset_index(drop=True)


def design(d: pd.DataFrame) -> pd.DataFrame:
    """Covariates: demographics, base-period ratios, industry, and schedule x quarter indicators."""
    X = pd.DataFrame(index=d.index)
    for c in ("age", "educ", "dependents", "hh_size"):
        v = d[c].astype(float)
        X[c] = v.fillna(v.median())
        X[c + "_na"] = v.isna().astype(float)
    X["age2"] = X["age"] ** 2 / 100.0
    X["male"] = d["male"].fillna(0.0)
    X["black"] = d["black"]
    X["recall"] = d["recall"]
    X["log_bpw_ratio"] = d["log_bpw_ratio"]
    X["log_bpw_ratio2"] = d["log_bpw_ratio"] ** 2
    X["weeks_worked"] = d["weeks_worked"].where(d["weeks_worked"] < 99, np.nan).fillna(0.0) / 52.0
    for v in sorted(set(d.industry) - {"5"}):
        X[f"ind_{v}"] = (d.industry == v).astype(float)
    cell = d.state + "_" + d.bby.dt.to_period("Q").astype(str)
    cells = pd.DataFrame({f"cell_{v}": (cell == v).astype(float) for v in sorted(cell.unique())[1:]}, index=d.index)
    X = pd.concat([X, cells], axis=1)
    return X.loc[:, X.std(axis=0) > 0]


def load_dose(states: Sequence[str] = ("LA", "ID", "MO", "NM"), outcome: str = "weeks_paid",
              dose: str = "relative", on_schedule: bool = True, rebuild: bool = False) -> RDDSample:
    """Claim-level dose sample.

    dose="relative": Q = s / s_cap(schedule), A = WBA / max_b, a_ref = 1 (the cap).
    dose="real": Q = s in 1984 dollars, A = WBA in 1984 dollars (the 25 schedules differ).
    """
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {sorted(OUTCOMES)}")
    d = dose_frame(states, on_schedule, rebuild)
    d = d[np.isfinite(d[outcome])].reset_index(drop=True)
    X = design(d)
    if dose == "relative":
        Q = (d.s / d.s_cap).to_numpy(float)
        A = (d.wba_formula / d.max_b).to_numpy(float)
        a_ref = 1.0
    elif dose == "real":
        Q = d.s_real.to_numpy(float)
        A = d.wba_real.to_numpy(float)
        a_ref = None
    else:
        raise ValueError("dose must be 'relative' or 'real'")
    return RDDSample(
        Q=Q,
        X=X.to_numpy(float),
        Y=d[outcome].to_numpy(float),
        threshold=1.0,
        name="ui_cwbh",
        feature_names=list(X.columns),
        description=(f"CWBH UI claims ({', '.join(states)}), benefit years 1976-83. Q = base-period "
                     f"score ({dose}); A = weekly benefit by the formula; Y = {OUTCOMES[outcome]}."),
        citation=("Landais (2015), AEJ: Economic Policy 7(4), openICPSR 114581; Continuous Wage and "
                  "Benefit History data (US DOL)."),
        treatment_rule=lambda q: (np.asarray(q) < 1.0).astype(int),
        extras={
            "A": A,
            "a_ref": a_ref,
            "dose_levels": None,
            "schedule": d.schedule.to_numpy(),
            "state": d.state.to_numpy(),
            "Z_names": [z for z in Z_NAMES if z in X.columns],
            "outcomes": {k: d[k].to_numpy(float) for k in OUTCOMES},
            "wba": d.wba_formula.to_numpy(float),
            "max_b": d.max_b.to_numpy(float),
            "min_b": d.min_b.to_numpy(float),
            "frac": d.frac_b.to_numpy(float),
            "round_code": d.round_b.to_numpy(int),
            "s": d.s.to_numpy(float),
            "s_cap": d.s_cap.to_numpy(float),
            "defl": d.defl.to_numpy(float),
            "bpw": d.bpw.to_numpy(float),
            "potdur": d.potdur.to_numpy(float),
            "claim_id": (d.state + "_" + d.id.astype(str)).to_numpy(),
            "bby": d.bby.to_numpy(),
        },
    )
