"""Medicare low wage index hospital policy (FY 2020-2024) as a dose on the hospital's wage index.

Rule (FY 2020 IPPS final rule, 84 FR 42326-42328; restated in footnote 6 of each year's
Table 2): for a hospital whose otherwise applicable wage index Q is below the 25th
percentile wage index value across all hospitals, p25(t), the wage index is increased by

    A = 0.5 * max(p25(t) - Q, 0),

budget neutral through a cut to the standardized amounts of all hospitals. p25(t) from the
Table 2 footnotes: FY 2020 0.8457, FY 2021 0.8469 (0.8465 in the final rule, 0.8469 in the
correction notice), FY 2022 0.8437, FY 2023 0.8427, FY 2024 0.8667, FY 2025 0.9009 (final
rule and correction notice). After the D.C. Circuit held the policy unlawful (Bridgeport
Hosp. v. Becerra, 108 F.4th 882, 23 July 2024), the interim final action CMS-1808-IFC
(89 FR 80405, effective 30 Sep 2024) recomputed FY 2025 without it; FY 2026-2027 have no
policy (with transitional exceptions for formerly low wage index hospitals). A 5% cap on
decreases from the prior year's wage index applied after the policy in FY 2020 (all
hospitals), FY 2021-2022 (hospitals hit by the new OMB delineations) and from FY 2023 on
(all hospitals, 42 CFR 412.64(h)(7)).

Panel: IPPS hospital (CCN) x FY t, 50 states and DC (no Maryland, no Puerto Rico by default).
  Q   the FY t wage index before the low wage index adjustment and the cap (Table 2 column
      "Wage Index Prior to Quartile and ..."); in FY 2016-2019 (no policy) the FY t wage index.
  A   the increase from the policy by the formula; 0 in FY 2016-2019 and from FY 2025.
  ``load_dose`` gives Q and A in wage index points (x100): Q = 84.57 is a wage index of 0.8457.
  Y   ``outcome="log_ahw"`` (default): log average hourly wage (Table 2) from the hospital's
      cost report for FY t (cost-reporting periods beginning 1 Oct t-1 to 30 Sep t), published
      as "Average Hourly Wage FY t+4" (the FY s wage index uses FY s-4 cost reports,
      84 FR 42301-42302). ``"dlog_ahw"``: the same minus log AHW from the FY t-4 cost report
      (the data behind Q). ``"log_cases"``: log Medicare cases in FY t+1 (Impact File FY t+3).

    from experiments.datasets.medicare_wage_index import adapter
    s = adapter.load_dose()            # FY 2016-2023 (outcome needs the FY t+4 Table 2)
    adapter.rule_check()
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
EXTRACTED = RAW / "extracted"
PROCESSED = HERE / "data" / "processed"
PANEL = PROCESSED / "wage_index_panel.csv.gz"
PANEL_VERSION = 1

# Table 2 used per FY: the last published version (correction notice / correcting amendment).
TABLE2 = {
    2016: r"CMS-1632-CN Table 2\.txt",
    2017: r"CMS 1655-CN2 FY 2017 Correction Notice Table 2\.txt",
    2018: r"CMS 1677 FY 2018 CN Table 2\.txt",
    2019: r"CMS_1694_CN_Table_2\.txt",
    2020: r"CMS_1716_CN_Table_2\.txt",
    2021: r"CMS-1735-CN Table 2\.txt",
    2022: r"CMS 1752-F2 \(Correcting Amendment\) Table 2\.txt",
    2023: r"CMS-1771-F Table 2\.txt",
    2024: r"CMS-1785-CN Table 2\.txt",
    2025: r"CMS-1808-IFC Tables 2\.txt",           # what was paid: no low wage index policy
    2026: r"CMS-1833-F Tables 2\.txt",
    2027: r"CMS-1849-CN Table 2\.txt",
}
TABLE2_FY2025_FR = r"CMS-1808-CN Tables 2\.txt"   # FY 2025 as first finalised (policy on)
YEARS = range(2016, 2028)
P25 = {2020: 0.8457, 2021: 0.8469, 2022: 0.8437, 2023: 0.8427, 2024: 0.8667}
P25_FY2025_REMOVED = 0.9009
RRC = {7, 15, 17, 22}
MDH = {14, 15}
SCH = {16, 17}
CONT = ["log_ahw_lag", "dlog_ahw_lag", "log_beds", "occupancy", "resbed", "cmi", "mcr_pct", "dpp",
        "log_cases_lag", "opccr"]
Z_NAMES = ["log_beds", "rural", "log_ahw_lag"]


# ---------------------------------------------------------------- reading
def _extract(zpath: Path) -> Path:
    """Extract a zip (and zips inside it) into extracted/<stem>; refuses unsafe member paths."""
    dest = EXTRACTED / zpath.stem
    if dest.exists() and any(dest.rglob("*.txt")):
        return dest

    def ext(zf: zipfile.ZipFile, d: Path) -> None:
        root = d.resolve()
        for m in zf.infolist():
            if m.is_dir():
                continue
            target = (d / m.filename).resolve()
            if root not in target.parents:
                raise ValueError(f"unsafe member {m.filename!r} in {zpath}")
            data = zf.read(m)
            if m.filename.lower().endswith(".zip"):
                sub = d / Path(m.filename).stem
                sub.mkdir(parents=True, exist_ok=True)
                ext(zipfile.ZipFile(io.BytesIO(data)), sub)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)

    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        ext(z, dest)
    return dest


def _table2_path(year: int, pattern: str | None = None) -> Path:
    z = RAW / "tables" / f"fy{year}_tables_2_3_4.zip"
    if not z.exists():
        raise FileNotFoundError(f"{z} missing. Run: python -m experiments.datasets.medicare_wage_index.download")
    pat = re.compile(pattern or TABLE2[year])
    hits = [p for p in _extract(z).rglob("*.txt") if pat.fullmatch(p.name)]
    if len(hits) != 1:
        raise FileNotFoundError(f"Table 2 for FY {year}: {len(hits)} matches for {pat.pattern}")
    return hits[0]


def _col(cols, regex, required=True):
    hit = [c for c in cols if re.search(regex, c)]
    if not hit and required:
        raise KeyError(f"no column matching {regex!r} in {cols}")
    return hit[0] if hit else None


def read_table2(year: int, pattern: str | None = None) -> pd.DataFrame:
    """One FY's Table 2: wage indexes before / with the quartile adjustment / final, AHW, CBSAs."""
    d = pd.read_csv(_table2_path(year, pattern), sep="\t", skiprows=1, dtype=str, encoding="latin-1")
    d.columns = [re.sub(r"\s+", " ", c).strip() for c in d.columns]
    cols = list(d.columns)
    d = d.rename(columns={cols[0]: "ccn"})
    d["ccn"] = d.ccn.astype(str).str.strip()
    d = d[d.ccn.str.fullmatch(r"[0-9A-Z]{6}")].copy()
    num = lambda c: pd.to_numeric(d[c], errors="coerce") if c else np.nan   # noqa: E731
    out = pd.DataFrame({"ccn": d.ccn, "year": year})
    out["cmi_t2"] = num(_col(cols, r"^2Case-Mix"))
    final = _col(cols, rf"(With Quartile and Cap|With Cap)$", required=False) or _col(cols, rf"^3 ?FY ?{year} Wage Index$")
    prior = _col(cols, r"Prior to (Quartile|Cap)", required=False)
    quart = _col(cols, r"Wage Index With Quartile$", required=False)
    trans = _col(cols, r"^9 ", required=False)
    out["wi_final"] = num(final)
    out["wi_prior"] = num(prior) if prior else out.wi_final
    out["wi_quart"] = num(quart) if quart else out.wi_prior
    out["wi_trans"] = num(trans) if trans else np.nan
    out["wi_prev"] = num(_col(cols, rf"^FY {year - 1} Wage Index$", required=False))
    for k, lab in enumerate((year - 2, year - 1, year)):
        out[f"ahw_{k}"] = num(_col(cols, rf"^4 ?Average Hourly Wage FY ?{lab}$"))
    out["geo_cbsa"] = d[_col(cols, r"^Geographic CBSA")].astype(str).str.strip()
    pay = _col(cols, r"Reclassified/ ?Redesignated CBSA|Wage Index Payment CBSA")
    out["pay_cbsa"] = d[pay].fillna("").astype(str).str.strip()
    mg = _col(cols, r"^MGCRB Reclass$", required=False)
    lu = _col(cols, r"^Lugar", required=False)
    r103 = _col(cols, r"1886\(d\)\(8\)\(E\)", required=False)
    flag = lambda c: d[c].fillna("").astype(str).str.strip().ne("") if c else False   # noqa: E731
    out["mgcrb"] = flag(mg).astype(int)
    out["lugar"] = flag(lu).astype(int)
    out["rural103"] = flag(r103).astype(int)
    out["reclass"] = ((out.mgcrb + out.lugar + out.rural103) > 0).astype(int) | out.pay_cbsa.ne("").astype(int)
    return out.drop_duplicates("ccn")


def read_impact(year: int) -> pd.DataFrame:
    from experiments.datasets.medicare_dsh.adapter import read_impact as _ri
    d = _ri(year).rename(columns={"provider": "ccn"})
    d["dpp"] = 100 * d.dshpct
    return d[["ccn", "year", "beds", "adc", "bills", "resbed", "mcr_pct", "dpp", "opccr", "ptype", "region",
              "cmi", "urspa", "urgeo", "wageindex"]]


# ---------------------------------------------------------------- rule
def formula(q, p25):
    """Low wage index increase (wage index points) from the score q and the year's 25th percentile."""
    q = np.asarray(q, float)
    p = np.asarray(p25, float)
    return 0.5 * np.maximum(p - q, 0.0)


def p25_of(year) -> np.ndarray:
    y = np.asarray(year).astype(int)
    return np.array([P25.get(int(v), -np.inf) for v in np.atleast_1d(y)])


def schedule(q, group):
    """Score and dose in wage index points (x100): A = 0.5 max(100 p25(FY) - q, 0) in FY 2020-2024,
    0 otherwise; group = "<FY>", or "<FY>|placebo=<p25>" for a pseudo-policy year."""
    g = np.asarray(group).astype(str)
    yr = np.array([int(s.split("|")[0]) for s in g])
    p = p25_of(yr)
    pl = np.array([float(s.split("placebo=")[1]) if "placebo=" in s else np.nan for s in g])
    p = np.where(np.isfinite(pl), pl, p)
    q = np.asarray(q, float)
    return np.where(np.isfinite(p), 100 * formula(q / 100, np.where(np.isfinite(p), p, 0.0)), 0.0)


# ---------------------------------------------------------------- panel
def build_panel(force: bool = False) -> pd.DataFrame:
    if PANEL.exists() and not force:
        p = pd.read_csv(PANEL, dtype={"ccn": str})
        if p.panel_version.iloc[0] == PANEL_VERSION:
            return p
    t2 = pd.concat([read_table2(y) for y in YEARS], ignore_index=True)
    # AHW by cost-report year (label L = cost reports for FY L-4); values are identical across vintages
    long = []
    for y in YEARS:
        g = t2[t2.year == y]
        for k, lab in enumerate((y - 2, y - 1, y)):
            long.append(pd.DataFrame({"ccn": g.ccn, "cr_year": lab - 4, "ahw": g[f"ahw_{k}"], "vintage": y}))
    long = pd.concat(long).dropna(subset=["ahw"]).sort_values("vintage").drop_duplicates(["ccn", "cr_year"], keep="last")
    ahw = long.set_index(["ccn", "cr_year"]).ahw
    df = t2.copy()
    for lag, name in [(0, "ahw_cr_t"), (-4, "ahw_cr_tm4"), (-5, "ahw_cr_tm5"), (-6, "ahw_cr_tm6"), (1, "ahw_cr_tp1")]:
        df[name] = ahw.reindex(pd.MultiIndex.from_arrays([df.ccn, df.year + lag])).to_numpy()
    imp = pd.concat([read_impact(y) for y in range(2016, 2028)], ignore_index=True)
    df = df.merge(imp, on=["ccn", "year"], how="left")
    key = imp.set_index(["ccn", "year"])
    df["bills_lead3"] = key.bills.reindex(pd.MultiIndex.from_arrays([df.ccn, df.year + 3])).to_numpy()
    fr25 = read_table2(2025, TABLE2_FY2025_FR).set_index("ccn")
    df["wi_quart_fy2025_fr"] = np.where(df.year == 2025, fr25.wi_quart.reindex(df.ccn).to_numpy(), np.nan)
    df["p25"] = p25_of(df.year)
    df["A_formula"] = np.where(np.isfinite(df.p25), formula(df.wi_prior, np.where(np.isfinite(df.p25), df.p25, 0)), 0.0)
    df["A_pub"] = df.wi_quart - df.wi_prior
    df["cap_binds"] = ((df.wi_final - df.wi_quart).abs() > 1.5e-4).astype(int)
    df["panel_version"] = PANEL_VERSION
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(PANEL, index=False)
    return df


def _frame(years, keep_pr=False, drop_cap=True):
    df = build_panel()
    st = df.ccn.str[:2]
    df = df[df.year.between(*years) & (st != "21") & df.ccn.str[2:].str.fullmatch(r"\d{4}")]   # no MD, no B campuses
    if not keep_pr:
        df = df[st.loc[df.index] != "40"]
    if drop_cap:
        df = df[df.cap_binds == 0]
    return df.copy()


def design(df: pd.DataFrame, x: str = "full") -> pd.DataFrame:
    """Covariates. x="full": hospital traits + lagged own wage + region + year; "state": + state;
    "nowage": without the lagged own wage."""
    d = pd.DataFrame(index=df.index)
    d["log_ahw_lag"] = np.log(df.ahw_cr_tm4)
    d["dlog_ahw_lag"] = (np.log(df.ahw_cr_tm4) - np.log(df.ahw_cr_tm6)).fillna(0.0)
    d["dlog_ahw_lag_na"] = df.ahw_cr_tm6.isna().astype(float)
    d["log_beds"] = np.log(df.beds.where(df.beds > 0))
    d["occupancy"] = (df.adc / df.beds.where(df.beds > 0)).clip(upper=1.5)
    d["resbed"] = df.resbed
    d["cmi"] = df.cmi.fillna(df.cmi_t2)
    d["mcr_pct"] = df.mcr_pct.where(df.mcr_pct <= 1.5)
    d["dpp"] = df.dpp.where(df.dpp.between(0, 100))
    d["log_cases_lag"] = np.log(df.bills.where(df.bills > 0))
    d["opccr"] = df.opccr.where(df.opccr.between(0, 3))
    d["rural"] = (~df.urgeo.astype(str).str.contains("URBAN")).astype(float)
    d["lurban"] = df.urgeo.astype(str).eq("LURBAN").astype(float)
    d["reclass"] = df.reclass.astype(float)
    d["rural103"] = df.rural103.astype(float)
    d["rrc"] = df.ptype.isin(RRC).astype(float)
    d["sch"] = df.ptype.isin(SCH).astype(float)
    d["mdh"] = df.ptype.isin(MDH).astype(float)
    d["teaching"] = (df.resbed > 0).astype(float)
    if x == "nowage":
        d = d.drop(columns=["log_ahw_lag", "dlog_ahw_lag", "dlog_ahw_lag_na"])
    yrs = sorted(df.year.unique())
    yd = pd.get_dummies(df.year, prefix="fy").astype(float).reindex(columns=[f"fy_{y}" for y in yrs[1:]], fill_value=0)
    if x == "state":
        sts = sorted(df.ccn.str[:2].unique())
        geo = pd.get_dummies(df.ccn.str[:2], prefix="st").astype(float).reindex(columns=[f"st_{s}" for s in sts[1:]], fill_value=0)
    else:
        regs = sorted(df.region.dropna().astype(int).unique())
        geo = pd.get_dummies(df.region.astype("Int64"), prefix="reg").astype(float).reindex(
            columns=[f"reg_{r}" for r in regs[1:]], fill_value=0)
    return pd.concat([d, yd, geo], axis=1)


OUTCOMES = {"log_ahw": "log average hourly wage, cost report FY t (Table 2 of FY t+4)",
            "dlog_ahw": "log AHW, cost report FY t, minus log AHW, cost report FY t-4",
            "log_cases": "log Medicare cases in FY t+1 (Impact File FY t+3)"}


def load_dose(years=(2016, 2023), outcome: str = "log_ahw", x: str = "full", q_max: float | None = 100.0,
              trim_eta: bool = True, placebo=None, keep_pr: bool = False, drop_cap: bool = True) -> RDDSample:
    """Hospital x FY sample for the dose design (see the module docstring).

    q_max: keep Q < q_max (wage index points); the default 100 drops high-wage areas, whose
    covariates lie far from those of the dosed hospitals (None keeps all).
    trim_eta: keep units whose eta (full-sample OLS of Q on X) lies inside the 0.5-99.5% range of
    the dosed units' eta, so the eta spline box has dosed units throughout.
    placebo: years without the policy (e.g. (2018, 2019)) given a pseudo-dose by the same rule, with
    p25 = the 25th percentile of Q across all Table 2 hospitals that year; years must end before 2020.
    """
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {sorted(OUTCOMES)}")
    pl_p25 = {}
    if placebo:
        if max(years) >= min(P25):
            raise ValueError("placebo runs must end before FY 2020")
        allp = build_panel()
        pl_p25 = {int(y): float(allp[allp.year == y].wi_prior.quantile(0.25)) for y in placebo}
    df = _frame(years, keep_pr, drop_cap)
    df["log_ahw"] = np.log(df.ahw_cr_t)
    df["dlog_ahw"] = np.log(df.ahw_cr_t) - np.log(df.ahw_cr_tm4)
    df["log_cases"] = np.log(df.bills_lead3.where(df.bills_lead3 > 0))
    X = design(df, x)
    ok = X.notna().all(axis=1) & df[outcome].notna() & df.wi_prior.notna()
    df, X = df[ok].reset_index(drop=True), X[ok].reset_index(drop=True)
    X = X.loc[:, X.std(axis=0) > 0]
    group = np.array([f"{y}|placebo={pl_p25[y]:.4f}" if y in pl_p25 else str(y) for y in df.year])
    Q = 100 * df.wi_prior.to_numpy(float)                    # wage index points
    A = schedule(Q, group)
    keep = np.ones(len(df), bool) if q_max is None else Q < q_max
    if trim_eta:
        Xt = np.column_stack([np.ones(len(df)), X.to_numpy(float)])
        eta = Q - Xt @ np.linalg.lstsq(Xt[keep], Q[keep], rcond=None)[0]
        lo, hi = np.quantile(eta[keep & (A > 0)], [0.005, 0.995])
        keep &= (eta >= lo) & (eta <= hi)
    df, X, Q, A, group = df[keep].reset_index(drop=True), X[keep].reset_index(drop=True), Q[keep], A[keep], group[keep]
    X = X.loc[:, X.std(axis=0) > 0]
    extras = {
        "A": A, "a_ref": 0.0, "dose_levels": None, "schedule": schedule, "group": group,
        "Z_names": [z for z in Z_NAMES if z in X.columns], "unit_id": df.ccn.to_numpy(),
        "year": df.year.to_numpy(int), "published_dose": 100 * df.A_pub.to_numpy(float), "p25": df.p25.to_numpy(float),
        "wi_final": df.wi_final.to_numpy(float), "cbsa": np.where(df.pay_cbsa.astype(str).str.len() > 0,
                                                                   df.pay_cbsa.astype(str), df.geo_cbsa.astype(str)),
        "outcomes": {k: df[k].to_numpy(float) for k in OUTCOMES},
        "outcome": outcome, "cutoffs": {y: 100 * v for y, v in P25.items()}, "units": "wage index points (x100)",
    }
    return RDDSample(
        Q=Q, X=X.to_numpy(float), Y=df[outcome].to_numpy(float),
        threshold=100 * float(np.median(list(P25.values()))), name="medicare_wage_index", feature_names=list(X.columns),
        description=("Medicare low wage index hospital policy (FY 2020-2024): wage index raised by half the gap "
                     "to the 25th percentile; Q = wage index before the policy"),
        citation="CMS IPPS final rules, Table 2 (FY 2016-2027); 84 FR 42326-42328; 89 FR 80405",
        treatment_rule=lambda q: np.zeros(len(np.atleast_1d(q)), int),
        extras=extras,
    )


def load() -> RDDSample:
    return load_dose()


def rule_check() -> pd.DataFrame:
    """Published increase (with quartile - prior) against the formula, by FY; cap and percentile checks."""
    df = build_panel()
    rows = []
    for y, g_all in df.groupby("year"):
        g = g_all[g_all.ccn.str[:2] != "21"]                 # Maryland: in Table 2, not paid under IPPS
        pub = g.wi_quart - g.wi_prior
        ok = (pub - g.A_formula).abs() < 1.5e-4
        rows.append({"fy": y, "n": len(g), "p25_published": P25.get(y, np.nan),
                     "p25_of_Q_all_table2": float(g_all.wi_prior.quantile(0.25)), "match": float(ok.mean()),
                     "mismatch": int((~ok).sum()), "share_dose_pos": float((g.A_formula > 0).mean()),
                     "cap_binds": float(g.cap_binds.mean())})
    g = df[(df.year == 2025) & (df.ccn.str[:2] != "21")]
    pub = g.wi_quart_fy2025_fr - g.wi_prior
    ok = (pub - formula(g.wi_prior, P25_FY2025_REMOVED)).abs() < 1.5e-4
    rows.append({"fy": "2025 FR/CN (removed by the IFC)", "n": int(pub.notna().sum()), "p25_published": P25_FY2025_REMOVED,
                 "p25_of_Q_all_table2": float(df[df.year == 2025].wi_prior.quantile(0.25)), "match": float(ok[pub.notna()].mean()),
                 "mismatch": int((~ok[pub.notna()]).sum()), "share_dose_pos": float((pub > 1e-9).mean()), "cap_binds": np.nan})
    return pd.DataFrame(rows)


def readiness(s: RDDSample | None = None) -> dict:
    s = s or load_dose()
    Xt = np.column_stack([np.ones(s.n), s.X])
    T = Xt @ np.linalg.lstsq(Xt, s.Q, rcond=None)[0]
    eta = s.Q - T
    A = s.extras["A"]
    ref = A <= 1e-12
    pos = ~ref
    cuts = np.quantile(A[pos], np.linspace(0, 1, 6))
    by = [("ref", int(ref.sum()), *np.round(np.quantile(eta[ref], [0.1, 0.5, 0.9]), 2))]
    for j in range(5):
        m = pos & (A >= cuts[j]) & (A <= cuts[j + 1])
        by.append((f"q{j + 1}:{cuts[j]:.2f}-{cuts[j + 1]:.2f}", int(m.sum()), *np.round(np.quantile(eta[m], [0.1, 0.5, 0.9]), 2)))
    yr = s.extras["year"]
    pol = np.isin(yr, list(P25))
    r2_pol = float(np.var(T[pol]) / np.var(s.Q[pol])) if pol.any() else np.nan
    return {"n": s.n, "units": len(np.unique(s.extras["unit_id"])), "p": s.p,
            "R2": float(np.var(T) / np.var(s.Q)), "R2_policy_years": r2_pol, "sd_Q": float(s.Q.std()),
            "sd_T": float(T.std()), "sd_eta": float(eta.std()), "share_ref": float(ref.mean()),
            "share_ref_policy_years": float(ref[pol].mean()) if pol.any() else np.nan, "by_dose": by}
