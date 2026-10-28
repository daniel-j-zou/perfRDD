"""CMS ESRD Quality Incentive Program (QIP): a payment-reduction dose on the TPS.

Each payment year (PY), CMS scores every Medicare dialysis facility on a Total
Performance Score (TPS, an integer 0-100; 0-30 in PY 2012-2013) computed from its
performance two years earlier (PY t uses calendar year t-2). A facility whose TPS falls
below the year's minimum TPS (mTPS) has all its Medicare ESRD payments for CY t cut by
0.5% for each 10 points (5 points in PY 2012-2013) below the minimum, up to 2%
(42 CFR 413.177; CMS "Program Details" fact sheets, see README):

    PY 2014-2026:  TPS >= m: 0;  m-10..m-1: 0.5;  m-20..m-11: 1.0;  m-30..m-21: 1.5;
                   below m-30: 2.0   (percent)
    PY 2012 (m=26): 26-30: 0; 21-25: 0.5; 16-20: 1.0; 11-15: 1.5; 0-10: 2.0
    PY 2013 (m=30): 30: 0; 26-29: 1.0; 21-25: 1.5; 0-20: 2.0
    PY 2022: no reductions (COVID-19 special rule; the PSSR reports N/A for all)

Q = TPS - mTPS of the payment year (integer). One scale for PY 2014-2026, where the
    schedule in Q is the same every year; effective cutoffs -0.5, -10.5, -20.5, -30.5.
A = the formula payment reduction (%), schedule(Q, PY). `rule_check()` compares it with
    the published reduction.
D = 1{A > 0} = 1{Q < -0.5}.
X = lagged QIP results (TPS percentile in PY t-1 and t-2, PY t-1 reduction and measure
    scores), corporate chain, facility age, hospital-based CCN, log eligible patients,
    state and payment-year indicators; with ``dfc=True`` (PY 2020+) also profit status,
    stations, late shift and home modalities from Dialysis Facility Compare.
Y = ``load_dose()``: TPS in PY t+2, whose performance year is CY t, the year the PY t
    reduction is paid. Alternatives: TPS in PY t+1 (performance year t-1, mostly before
    the PY t result is announced), standardized hospitalization / mortality rates.

Units are facility (CCN) x payment year.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample
from experiments.datasets.cms_esrd_qip import pssr

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
PROCESSED = HERE / "data" / "processed"
CACHE = PROCESSED / "esrd_qip_panel.csv.gz"
YEARS = tuple(range(2012, 2027))

# Minimum TPS by payment year. Sources: CMS fact sheets in data/raw/docs (PY 2012-2019,
# 2024-2026; PY 2025-2028 also in the 2025-2028 QIP summary). PY 2020, 2021 and 2023
# have no fact sheet on cms.gov; their minimum is the one the PSSR bands imply (every
# band edge brackets exactly, see rule_check()).
MIN_TPS: Dict[int, Optional[int]] = {
    2012: 26, 2013: 30, 2014: 53, 2015: 60, 2016: 54, 2017: 60, 2018: 49, 2019: 60,
    2020: 59, 2021: 56, 2022: None, 2023: 83, 2024: 57, 2025: 55, 2026: 53,
}
MIN_TPS_SOURCE = {py: "fact sheet" for py in MIN_TPS}
MIN_TPS_SOURCE.update({2020: "PSSR bands", 2021: "PSSR bands", 2023: "PSSR bands",
                       2022: "no reductions"})
DOSE_LEVELS = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
# Lower TPS bound of each band, highest band first, and the reduction of each band
# (the last dose applies below the last bound).
BANDS = {2012: ([26, 21, 16, 11], [0.0, 0.5, 1.0, 1.5, 2.0]),
         2013: ([30, 26, 21], [0.0, 1.0, 1.5, 2.0])}
for _py, _m in MIN_TPS.items():
    if _py not in BANDS and _m is not None:
        BANDS[_py] = ([_m, _m - 10, _m - 20, _m - 30], [0.0, 0.5, 1.0, 1.5, 2.0])
CUTOFFS_Q = (-0.5, -10.5, -20.5, -30.5)   # PY 2014-2026, in units of Q = TPS - mTPS

MEASURES_X = ["ktv", "vat", "hypercal", "nhsn", "srr", "strr", "shr", "ichcahps"]
CHAINS = {"davita": ("DAVITA", "GAMBRO"),
          "fresenius": ("FRESENIUS", "RENAL CARE GROUP", "LIBERTY DIALYSIS"),
          "dci": ("DIALYSIS CLINIC, INC",),
          "usrc": ("US RENAL CARE",),
          "ara": ("AMERICAN RENAL ASSOC",)}
INDEPENDENT = ("INDEPENDENT", "NOT A CHAIN", "STATE OWNED")

# Dialysis Facility Compare file in each January snapshot (data/raw/archive/).
DFC_FILES = {2020: "snap_2020-01-31/DFC_SOCRATA_FAC_DATA.csv",
             2021: "snap_2021-01-31/23ew-n7w9.csv",
             2023: "snap_2023-01-18/DFC_FACILITY.csv",
             2024: "snap_2024-01-18/DFC_FACILITY.csv",
             2025: "snap_2025-01-15/DFC_FACILITY.csv",
             2026: "snap_2026-01-21/DFC_FACILITY.csv"}

OUTCOMES = {
    "tps_t2": "TPS in PY t+2 (performance year CY t, the year the PY t reduction is paid)",
    "tps_t1": "TPS in PY t+1 (performance year CY t-1, mostly before PY t is announced)",
    "shr": "standardized hospitalization rate (DFC, per 100 patient-years), CY t",
    "smr": "standardized mortality rate (DFC, per 100 patient-years), CY t-3..t",
}


# --------------------------------------------------------------------------- rule
def dose_from_tps(tps, py) -> np.ndarray:
    """Payment reduction (%) from integer TPS and payment year (NaN for PY 2022)."""
    tps = np.asarray(tps, float)
    py = np.broadcast_to(np.asarray(py), tps.shape)
    out = np.full(tps.shape, np.nan)
    for y in np.unique(py):
        if int(y) not in BANDS:
            continue
        lows, doses = BANDS[int(y)]
        m = (py == y) & ~np.isnan(tps)
        # number of band bounds the score falls below
        k = (tps[m][:, None] < np.asarray(lows)[None, :]).sum(1)
        out[m] = np.asarray(doses)[k]
    return out


def schedule(q, group) -> np.ndarray:
    """Dose (%) from Q = TPS - mTPS and the payment year."""
    q = np.asarray(q, float)
    g = np.broadcast_to(np.asarray(group), q.shape)
    m = np.array([MIN_TPS.get(int(y)) or np.nan for y in g.ravel()], float).reshape(q.shape)
    return dose_from_tps(np.round(q + m), g)


# --------------------------------------------------------------------------- data
def _chain(owner: pd.Series) -> pd.Series:
    o = owner.fillna("").astype(str).str.upper().str.strip()
    out = pd.Series("other_chain", index=o.index)
    out[o == ""] = "unknown"
    out[o.isin(INDEPENDENT)] = "independent"
    for k, keys in CHAINS.items():
        out[o.apply(lambda s: any(s.startswith(x) or x in s for x in keys))] = k
    return out


def _hospital_based(ccn: pd.Series) -> pd.Series:
    """CCN serial 2300-2499 or 3500-3799: hospital-based or hospital satellite unit."""
    s = pd.to_numeric(ccn.astype(str).str[-4:], errors="coerce")
    return (s.between(2300, 2499) | s.between(3500, 3799)).astype(int)


def _dfc(year: int) -> pd.DataFrame:
    f = RAW / "archive" / DFC_FILES[year]
    d = pd.read_csv(f, dtype=str, encoding="latin-1")
    cc = next(c for c in d.columns if c in ("Provider Number", "CMS Certification Number (CCN)"))
    chain_owned = next(c for c in d.columns if c.lower() == "chain owned")
    yes = lambda c: (d[c].astype(str).str.strip().str.lower() == "yes").astype(float)
    return pd.DataFrame({
        "ccn": d[cc].astype(str).str.strip().str.zfill(6),
        "dfc_forprofit": (d["Profit or Non-Profit"].str.lower() == "profit").astype(float),
        "dfc_chain": yes(chain_owned),
        "dfc_stations": pd.to_numeric(d["# of Dialysis Stations"], errors="coerce"),
        "dfc_late_shift": yes("Late Shift"),
        "dfc_pd": yes("Offers peritoneal dialysis"),
        "dfc_home_hd": yes("Offers home hemodialysis training"),
        "dfc_ichd": yes("Offers in-center hemodialysis"),
        "shr_rate": pd.to_numeric(d["Hospitalization Rate (Facility)"], errors="coerce"),
        "smr_rate": pd.to_numeric(d["Mortality Rate (Facility)"], errors="coerce"),
    }).drop_duplicates("ccn")


def _build() -> pd.DataFrame:
    q = pssr.read_all(YEARS)
    q = q.assign(_miss=q["tps"].isna()).sort_values(["py", "ccn", "_miss"])
    q = q.drop_duplicates(["py", "ccn"]).drop(columns="_miss")
    pats = pd.concat([pssr.patients(py) for py in YEARS], ignore_index=True)
    q = q.merge(pats, on=["py", "ccn"], how="left")
    q["mtps"] = q["py"].map(MIN_TPS).astype(float)
    q["q"] = q["tps"] - q["mtps"]
    q["dose"] = dose_from_tps(q["tps"], q["py"])
    q["tps_pct"] = q.groupby("py")["tps"].rank(pct=True)
    q["chain"] = _chain(q["owner"])
    q["hospital_based"] = _hospital_based(q["ccn"])
    start = pd.to_datetime((q["py"] - 2).astype(str) + "-01-01")   # performance year
    q["age_years"] = ((start - q["cert_date"]).dt.days / 365.25).clip(lower=0)

    key = ["ccn", "py"]
    mcols = [f"m_{m}" for m in MEASURES_X]
    lag1 = q[key + ["tps", "tps_pct", "red_pub"] + mcols].copy()
    lag1["py"] += 1
    lag1 = lag1.rename(columns={c: f"lag1_{c}" for c in lag1.columns if c not in key})
    lag2 = q[key + ["tps_pct"]].assign(py=q["py"] + 2).rename(columns={"tps_pct": "lag2_tps_pct"})
    lead = {}
    for h in (1, 2):
        ld = q[key + ["tps", "q", "tps_pct", "red_pub"]].assign(py=q["py"] - h)
        lead[h] = ld.rename(columns={c: f"{c}_t{h}" for c in ld.columns if c not in key})
    q = q.merge(lag1, on=key, how="left").merge(lag2, on=key, how="left")
    q = q.merge(lead[1], on=key, how="left").merge(lead[2], on=key, how="left")

    # DFC: characteristics from the January snapshot of PY t; outcomes for CY t from
    # the snapshot of January t+2 (SHR of CY t, SMR of CY t-3..t).
    dfc = {y: _dfc(y) for y in DFC_FILES}
    chars = pd.concat([d.drop(columns=["shr_rate", "smr_rate"]).assign(py=y)
                       for y, d in dfc.items()], ignore_index=True)
    outc = pd.concat([d[["ccn", "shr_rate", "smr_rate"]].assign(py=y - 2)
                      for y, d in dfc.items()], ignore_index=True)
    q = q.merge(chars, on=key, how="left").merge(outc, on=key, how="left")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    q.to_csv(CACHE, index=False)
    return q


def panel(rebuild: bool = False) -> pd.DataFrame:
    """Facility x payment-year panel, PY 2012-2026 (cached in data/processed/)."""
    if CACHE.exists() and not rebuild:
        return pd.read_csv(CACHE, dtype={"ccn": str, "zip": str},
                           parse_dates=["cert_date"], low_memory=False)
    return _build()


def rule_check() -> pd.DataFrame:
    """Formula reduction vs published reduction, every scored facility, each PY."""
    df = panel()
    df = df[df["tps"].notna() & df["red_pub"].notna()]
    df = df.assign(ok=np.isclose(df["dose"], df["red_pub"]))

    def edges(z):
        pos, zero = z[z["red_pub"] > 0]["tps"], z[z["red_pub"] == 0]["tps"]
        return pd.Series({"max_tps_reduced": pos.max(), "min_tps_unreduced": zero.min()})

    out = df.groupby("py").agg(scored=("ok", "size"), agree=("ok", "mean"),
                               mismatches=("ok", lambda s: int((~s).sum())),
                               reduced_share=("red_pub", lambda s: float((s > 0).mean())))
    out = out.join(df.groupby("py").apply(edges))
    out["mtps"] = pd.Series(MIN_TPS)
    out["mtps_source"] = pd.Series(MIN_TPS_SOURCE)
    return out


# --------------------------------------------------------------------------- sample
def _x(df: pd.DataFrame, dfc: bool):
    base = {}
    for c in ("lag1_tps_pct", "lag2_tps_pct"):
        base[c] = df[c].fillna(0.0)
        base[c.replace("_tps_pct", "_missing")] = df[c].isna().astype(float)
    base["lag1_reduction"] = df["lag1_red_pub"].fillna(0.0)
    for m in MEASURES_X:
        c = f"lag1_m_{m}"
        base[c] = df[c].fillna(0.0)
        base[f"{c}_missing"] = df[c].isna().astype(float)
    for k in ("davita", "fresenius", "dci", "usrc", "ara", "other_chain", "unknown"):
        base[f"chain_{k}"] = (df["chain"] == k).astype(float)
    base["hospital_based"] = df["hospital_based"].astype(float)
    base["age_years"] = df["age_years"].fillna(df["age_years"].median()).clip(upper=50)
    base["log_patients"] = np.log1p(df["n_patients"].fillna(5.0))
    base["patients_missing"] = df["n_patients"].isna().astype(float)
    if dfc:
        base["dfc_forprofit"] = df["dfc_forprofit"]
        base["log_stations"] = np.log1p(df["dfc_stations"].fillna(df["dfc_stations"].median()))
        for c in ("dfc_late_shift", "dfc_pd", "dfc_home_hd"):
            base[c] = df[c]
    X = pd.DataFrame(base, index=df.index)
    states = sorted(df["state"].unique())
    st = pd.get_dummies(df["state"]).reindex(columns=states[1:], fill_value=0)
    st.columns = [f"state_{s}" for s in st.columns]
    years = sorted(df["py"].unique())
    yr = pd.get_dummies(df["py"]).reindex(columns=years[1:], fill_value=0)
    yr.columns = [f"py_{y}" for y in yr.columns]
    X = pd.concat([X, st, yr], axis=1).astype(float)
    return X.to_numpy(), list(X.columns)


def load_dose(outcome: str = "tps_t2", years: Optional[Sequence[int]] = None,
              dfc: bool = False) -> RDDSample:
    """Dose design: Q = TPS - mTPS, A = payment reduction (%), group = payment year.

    outcome: one of OUTCOMES. years: payment years to keep (default: every PY 2014-2026
    except 2022 with the outcome observed). dfc: add Dialysis Facility Compare
    characteristics to X (only PY 2020, 2021, 2023-2026 have them).
    """
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {list(OUTCOMES)}")
    df = panel()
    ycol = {"tps_t2": "tps_t2", "tps_t1": "tps_t1", "shr": "shr_rate", "smr": "smr_rate"}[outcome]
    keep = df["py"].between(2014, 2026) & (df["py"] != 2022) & df["q"].notna()
    keep &= df[ycol].notna() & df["dose"].notna()
    if years is not None:
        keep &= df["py"].isin(list(years))
    if dfc:
        keep &= df["dfc_forprofit"].notna()
    df = df[keep].reset_index(drop=True)
    X, names = _x(df, dfc)
    Q = df["q"].to_numpy(float)
    g = df["py"].to_numpy(int)
    A = schedule(Q, g)
    z = ["chain_davita", "chain_fresenius", "hospital_based", "age_years", "log_patients"]
    if dfc:
        z += ["dfc_forprofit", "log_stations"]
    yrs = sorted(set(g.tolist()))
    return RDDSample(
        Q=Q, X=X, Y=df[ycol].to_numpy(float), threshold=CUTOFFS_Q[0],
        name="cms_esrd_qip" + ("" if outcome == "tps_t2" else f"_{outcome}")
        + ("_dfc" if dfc else ""),
        feature_names=names,
        description=(
            f"CMS ESRD QIP, dialysis facilities, PY {yrs[0]}-{yrs[-1]} (PY {yrs}). "
            "Q = TPS - minimum TPS of the payment year; A = Medicare payment reduction "
            "(%): 0.5 per 10 points below the minimum, max 2. "
            f"Y = {OUTCOMES[outcome]}."),
        citation=("CMS ESRD QIP Performance Score Summary Reports PY 2012-2026 (cms.gov, "
                  "ESRD QIP Public Reporting & Certificates); Dialysis Facility Compare "
                  "archived snapshots (data.cms.gov Provider Data Catalog); 42 CFR 413.177."),
        treatment_rule=lambda q: (np.asarray(q) < CUTOFFS_Q[0]).astype(int),
        extras={
            "A": A,
            "a_ref": 0.0,
            "dose_levels": DOSE_LEVELS.copy(),
            "schedule": schedule,
            "group": g,
            "Z_names": z,
            "published_dose": df["red_pub"].to_numpy(float),
            "cutoffs": CUTOFFS_Q,
            "min_tps": dict(MIN_TPS),
            "ccn": df["ccn"].to_numpy(),
            "tps": df["tps"].to_numpy(float),
            "Y_alt": {k: df[c].to_numpy(float) for k, c in
                      {"tps_t1": "tps_t1", "tps_t2": "tps_t2", "q_t1": "q_t1",
                       "q_t2": "q_t2", "pct_t1": "tps_pct_t1", "pct_t2": "tps_pct_t2",
                       "reduction_t1": "red_pub_t1", "reduction_t2": "red_pub_t2",
                       "shr": "shr_rate", "smr": "smr_rate"}.items()},
            "raw": df,
        },
    )


def load() -> RDDSample:
    """Binary view of the dose design: D = 1{any reduction}, Y = TPS in PY t+2."""
    return load_dose()


# --------------------------------------------------------------------------- readiness
def readiness(s: Optional[RDDSample] = None) -> dict:
    """README readiness numbers for a dose sample."""
    s = s or load_dose()
    Z = np.column_stack([np.ones(s.n), s.X])
    T = Z @ np.linalg.lstsq(Z, s.Q, rcond=None)[0]
    eta = s.Q - T
    A = s.extras["A"]
    out = {"n": int(s.n), "facilities": int(len(set(s.extras["ccn"]))),
           "p": int(s.p), "R2": float(1 - eta.var() / s.Q.var()),
           "sd_Q": float(s.Q.std()), "sd_T": float(T.std()), "sd_eta": float(eta.std()),
           "share_dose0": float((A == 0).mean()),
           "dose_share": {float(a): float((A == a).mean()) for a in DOSE_LEVELS},
           "eta_quantiles_by_dose": {
               float(a): np.quantile(eta[A == a], [0.1, 0.5, 0.9]).round(2).tolist()
               for a in DOSE_LEVELS if (A == a).sum() > 0},
           "T_quantiles_by_dose": {
               float(a): np.quantile(T[A == a], [0.1, 0.5, 0.9]).round(2).tolist()
               for a in DOSE_LEVELS if (A == a).sum() > 0}}
    q = np.round(s.Q).astype(int)

    def ratio(c):   # c is a half-integer edge: units at c+0.5, c+1.5 vs c-0.5, c-1.5
        up = np.isin(q, [int(c + 0.5), int(c + 1.5)]).sum()
        dn = np.isin(q, [int(c - 0.5), int(c - 1.5)]).sum()
        return [int(up), int(dn), round(up / dn, 3) if dn else None]
    out["bunching_cutoffs"] = {c: ratio(c) for c in CUTOFFS_Q}
    out["bunching_placebo"] = {c: ratio(c) for c in
                               (14.5, 9.5, 4.5, -5.5, -15.5, -25.5)}
    out["agree_published"] = float(np.isclose(A, s.extras["published_dose"]).mean())
    return out
