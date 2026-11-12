"""Medicare operating IME adjustment as a dose on the resident-to-bed ratio.

Rule (section 1886(d)(5)(B)(ii) of the Social Security Act; 42 CFR 412.105(d)(3); Pub. L.
108-173 sec. 502(a), as described in the FY 2005 IPPS final rule, 69 FR 49088, and the FY 2023
rule, 87 FR 49065):

    IME adjustment factor = c x [(1 + r)^0.405 - 1],

r = residents (FTE) per available bed, c = 1.42 (FY 2005), 1.37 (FY 2006), 1.32 (FY 2007) and
1.35 (FY 2008 and after). The Impact File lists r (``resbed``) and the factor (``tchop``)
that CMS models for FY t.

Panel: IPPS hospital (CCN) x FY t, 50 states and DC (no Maryland, no Puerto Rico).
  Q   100 r (residents per 100 beds), as in the Impact File for FY t. 0 for non-teaching
      hospitals (57-72% of hospitals a year).
  A   100 x the IME adjustment factor from the formula (percent added to the DRG payment).
  Y   ``outcome="log_cases"`` (default): log Medicare cases in FY t+1 (MedPAR count printed in
      the FY t+3 Impact File); ``"cmi_lead"``: transfer-adjusted case-mix index in the FY t+3
      file.

    from experiments.datasets.medicare_ime import adapter
    s = adapter.load_dose()          # FY 2005-2024
    adapter.rule_check()
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample

HERE = Path(__file__).parent
PROCESSED = HERE / "data" / "processed"
PANEL = PROCESSED / "ime_panel.csv.gz"
PANEL_VERSION = 1
YEARS = range(2005, 2028)
MULT = {2005: 1.42, 2006: 1.37, 2007: 1.32}          # 1.35 from FY 2008 on
RRC = {7, 15, 17, 22}
MDH = {14, 15}
SCH = {16, 17}
CONT = ["log_beds", "occupancy", "cmi", "dpp", "mcr_pct", "log_cases_lag", "wageindex", "opccr"]
Z_NAMES = ["log_beds", "cmi", "rural"]


def multiplier(year) -> np.ndarray:
    return np.array([MULT.get(int(y), 1.35) for y in np.atleast_1d(year)])


def formula(r, c):
    """IME adjustment factor (fraction) from the resident-to-bed ratio r and the multiplier c."""
    r = np.asarray(r, float)
    return np.asarray(c, float) * ((1.0 + r) ** 0.405 - 1.0)


def schedule(q, group):
    """A = 100 c(FY) [(1 + q/100)^0.405 - 1]; q = residents per 100 beds; group = "<FY>"."""
    yr = np.array([int(str(s).split("|")[0]) for s in np.asarray(group)])
    return 100 * formula(np.asarray(q, float) / 100, multiplier(yr))


def _read_tchop(year: int) -> pd.DataFrame:
    from experiments.datasets.medicare_dsh.adapter import RAW, _extract
    if year <= 2024:
        d = pd.read_csv(RAW / "impact" / f"impact{year}.csv", dtype={"provider": str}, low_memory=False)
        d.columns = [c.lower().strip() for c in d.columns]
        prov, tch = d["provider"], d["tchop"]
    else:
        txt = [p for p in _extract(RAW / "impact" / f"fy{year}_impact.zip").glob("*.txt")
               if re.search(r"final rule", p.name, re.I) and "descr" not in p.name.lower()]
        d = pd.read_csv(txt[0], sep="\t", skiprows=1, dtype={"Provider Number": str}, encoding="latin-1",
                        low_memory=False)
        d.columns = [c.strip() for c in d.columns]
        prov, tch = d["Provider Number"], d["TCHOP"]
    out = pd.DataFrame({"provider": prov.astype(str).str.strip().str.zfill(6), "year": year,
                        "tchop": pd.to_numeric(tch, errors="coerce")})
    return out.drop_duplicates("provider")


def build_panel(force: bool = False) -> pd.DataFrame:
    if PANEL.exists() and not force:
        p = pd.read_csv(PANEL, dtype={"provider": str})
        if p.panel_version.iloc[0] == PANEL_VERSION:
            return p
    from experiments.datasets.medicare_dsh.adapter import read_impact
    imp = pd.concat([read_impact(y).merge(_read_tchop(y), on=["provider", "year"], how="left") for y in YEARS],
                    ignore_index=True)
    key = imp.set_index(["provider", "year"])
    df = imp.copy()
    for lag, tag, cols in [(3, "_lead3", ["bills", "cmi", "resbed"]), (-1, "_lag1", ["resbed"])]:
        idx = pd.MultiIndex.from_arrays([df.provider, df.year + lag])
        for c in cols:
            df[c + tag] = key[c].reindex(idx).to_numpy()
    present = set(zip(imp.provider, imp.year))
    df["exit2"] = [np.nan if y + 2 > max(YEARS) else float((p, y + 2) not in present)
                   for p, y in zip(df.provider, df.year)]
    df["c"] = multiplier(df.year)
    df["A_formula"] = 100 * formula(df.resbed, df.c)
    df["A_pub"] = 100 * df.tchop
    df["panel_version"] = PANEL_VERSION
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(PANEL, index=False)
    return df


def _frame(years):
    df = build_panel()
    st = df.provider.str[:2]
    df = df[df.year.between(*years) & ~st.isin(["21", "40"]) & df.resbed.notna() & (df.resbed >= 0)]
    return df.copy()


def design(df: pd.DataFrame, geo: str = "region") -> pd.DataFrame:
    d = pd.DataFrame(index=df.index)
    d["log_beds"] = np.log(df.beds.where(df.beds > 0))
    d["occupancy"] = (df.adc / df.beds.where(df.beds > 0)).clip(upper=1.5)
    d["cmi"] = df.cmi
    d["dpp"] = (100 * df.dshpct).where((100 * df.dshpct).between(0, 100))
    d["mcr_pct"] = df.mcr_pct.where(df.mcr_pct <= 1.5)
    d["log_cases_lag"] = np.log(df.bills.where(df.bills > 0))
    d["wageindex"] = df.wageindex
    d["opccr"] = df.opccr.where(df.opccr.between(0, 3))
    d["rural"] = (~df.urspa.astype(str).str.contains("URBAN")).astype(float)
    d["lurban"] = df.urspa.astype(str).eq("LURBAN").astype(float)
    d["rrc"] = df.ptype.isin(RRC).astype(float)
    d["sch"] = df.ptype.isin(SCH).astype(float)
    d["mdh"] = df.ptype.isin(MDH).astype(float)
    yrs = sorted(df.year.unique())
    yd = pd.get_dummies(df.year, prefix="fy").astype(float).reindex(columns=[f"fy_{y}" for y in yrs[1:]], fill_value=0)
    if geo == "state":
        sts = sorted(df.provider.str[:2].unique())
        gd = pd.get_dummies(df.provider.str[:2], prefix="st").astype(float).reindex(columns=[f"st_{s}" for s in sts[1:]], fill_value=0)
    else:
        regs = sorted(df.region.dropna().astype(int).unique())
        gd = pd.get_dummies(df.region.astype("Int64"), prefix="reg").astype(float).reindex(
            columns=[f"reg_{r}" for r in regs[1:]], fill_value=0)
    return pd.concat([d, yd, gd], axis=1)


OUTCOMES = {"log_cases": "log Medicare cases in FY t+1 (Impact File FY t+3)",
            "cmi_lead": "transfer-adjusted case-mix index in the FY t+3 Impact File"}


def load_dose(years=(2005, 2024), outcome: str = "log_cases", geo: str = "region",
              teaching_only: bool = False) -> RDDSample:
    """Hospital x FY sample for the dose design (see the module docstring)."""
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {sorted(OUTCOMES)}")
    df = _frame(years)
    if teaching_only:
        df = df[df.resbed > 0]
    df["log_cases"] = np.log(df.bills_lead3.where(df.bills_lead3 > 0))
    df["cmi_lead"] = df.cmi_lead3
    X = design(df, geo)
    ok = X.notna().all(axis=1) & df[outcome].notna()
    df, X = df[ok].reset_index(drop=True), X[ok].reset_index(drop=True)
    X = X.loc[:, X.std(axis=0) > 0]
    group = df.year.astype(str).to_numpy()
    Q = 100 * df.resbed.to_numpy(float)
    A = schedule(Q, group)
    extras = {
        "A": A, "a_ref": 0.0, "dose_levels": None, "schedule": schedule, "group": group,
        "Z_names": [z for z in Z_NAMES if z in X.columns], "unit_id": df.provider.to_numpy(),
        "year": df.year.to_numpy(int), "published_dose": df.A_pub.to_numpy(float), "c": df.c.to_numpy(float),
        "teaching": (df.resbed > 0).to_numpy(int), "resbed_lag1": df.resbed_lag1.to_numpy(float),
        "outcomes": {k: df[k].to_numpy(float) for k in OUTCOMES}, "outcome": outcome,
        "units": "Q = residents per 100 beds; A = IME adjustment in percent",
    }
    return RDDSample(
        Q=Q, X=X.to_numpy(float), Y=df[outcome].to_numpy(float), threshold=0.0, name="medicare_ime",
        feature_names=list(X.columns),
        description="Medicare operating IME adjustment c[(1+r)^0.405 - 1] on the resident-to-bed ratio r",
        citation="CMS IPPS Impact Files (NBER copies FY 2005-2024; CMS FY 2025-2027); 42 CFR 412.105(d)(3)",
        treatment_rule=lambda q: (np.asarray(q) > 0).astype(int),
        extras=extras,
    )


def load() -> RDDSample:
    return load_dose()


def rule_check(tol: float = 1e-4) -> pd.DataFrame:
    """Formula vs the published operating IME factor (tchop), by FY (50 states + DC, no Maryland)."""
    df = build_panel()
    df = df[~df.provider.str[:2].isin(["21", "40"]) & df.resbed.notna() & df.tchop.notna()]
    rows = []
    for y, g in df.groupby("year"):
        ok = (g.tchop - formula(g.resbed, g.c)).abs() < tol
        t = g.resbed > 0
        rows.append({"fy": y, "c": float(g.c.iloc[0]), "n": len(g), "match": float(ok.mean()), "mismatch": int((~ok).sum()),
                     "match_teaching": float(ok[t].mean()), "share_r0": float((g.resbed == 0).mean()),
                     "r_median_teaching": float(g.resbed[t].median()), "r_max": float(g.resbed.max()),
                     "n_r_at_1.9999": int((g.resbed.round(4) == 1.9999).sum())})
    return pd.DataFrame(rows)


def readiness(s: RDDSample | None = None) -> dict:
    s = s or load_dose()
    Xt = np.column_stack([np.ones(s.n), s.X])
    T = Xt @ np.linalg.lstsq(Xt, s.Q, rcond=None)[0]
    eta = s.Q - T
    A = s.extras["A"]
    ref = A <= 1e-12
    out = {"n": s.n, "units": len(np.unique(s.extras["unit_id"])), "p": s.p,
           "R2": float(np.var(T) / np.var(s.Q)), "sd_Q": float(s.Q.std()), "sd_eta": float(eta.std()),
           "share_ref": float(ref.mean())}
    if (~ref).sum() > 50:
        t = ~ref
        Xt1 = Xt[t]
        T1 = Xt1 @ np.linalg.lstsq(Xt1, s.Q[t], rcond=None)[0]
        out["R2_teaching_only_fit"] = float(np.var(T1) / np.var(s.Q[t]))
        out["eta_ref_q10_50_90"] = np.round(np.quantile(eta[ref], [0.1, 0.5, 0.9]), 2).tolist() if ref.any() else None
        out["eta_teach_q10_50_90"] = np.round(np.quantile(eta[t], [0.1, 0.5, 0.9]), 2).tolist()
        out["corr_eta_T_ref"] = float(np.corrcoef(eta[ref], T[ref])[0, 1]) if ref.sum() > 2 else None
    return out
