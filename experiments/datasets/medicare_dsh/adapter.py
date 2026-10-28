"""Medicare DSH: the operating DSH adjustment as a dose on the disproportionate patient percentage.

42 CFR 412.106(d), as applied in CMS's IPPS Impact Files (FY 2007 on):
  DPP < 15%:            0
  15% <= DPP < 20.2%:   2.5% + 0.65  (DPP - 15%)
  DPP >= 20.2%:         5.88% + 0.825 (DPP - 20.2%)
  capped at 12% for urban hospitals with < 100 beds (missing beds count as < 100) and for
  rural hospitals with < 500 beds that are neither rural referral centers nor (from FY
  2007, except FY 2013) Medicare-dependent hospitals; Pickle hospitals get 35%; from FY
  2014 the result is multiplied by 0.25 (section 1886(r)), and the other 75% is paid as
  uncompensated-care payments to the same eligible hospitals.

Panel: IPPS hospital (CCN) x federal fiscal year t, 50 states and DC (no Maryland).
  Q   DPP in percent, as in the Impact File for FY t (CMS's projection basis).
  A   operating DSH adjustment in percent of the DRG payment, from the formula.
  Y   ``outcome="log_cases"`` (default): log Medicare cases in FY t+1 (MedPAR, from the FY
      t+3 Impact File); ``"exit"``: 1 if the hospital is absent from the FY t+2 file;
      ``"dpp_lead"``: DPP in the FY t+3 file.

    from experiments.datasets.medicare_dsh import adapter
    s = adapter.load_dose()               # FY 2008-2013, before the ACA split
    adapter.rule_check()
"""
from __future__ import annotations

import re
import zipfile
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
EXTRACTED = RAW / "extracted"
PROCESSED = HERE / "data" / "processed"
PANEL = PROCESSED / "dsh_panel.csv.gz"
PANEL_VERSION = 2
YEARS = range(2005, 2028)
RRC = {7, 15, 17, 22}
MDH = {14, 15}
CUT, KINK, CAP = 15.0, 20.2, 12.0
CONT = ["log_beds", "occupancy", "resbed", "cmi", "mcr_pct", "log_cases_lag", "wageindex", "opccr"]
Z_NAMES = ["log_beds", "resbed", "cmi", "rural"]


def _extract(zpath: Path) -> Path:
    dest = EXTRACTED / zpath.stem
    if dest.exists() and any(dest.rglob("*")):
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    with zipfile.ZipFile(zpath) as z:
        for m in z.infolist():
            target = (dest / m.filename).resolve()
            if root not in target.parents:
                raise ValueError(f"unsafe member {m.filename!r} in {zpath}")
            if m.is_dir():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(m))
    return dest


def read_impact(year: int) -> pd.DataFrame:
    """One FY's IPPS Impact File, harmonised columns."""
    if year <= 2024:
        f = RAW / "impact" / f"impact{year}.csv"
        if not f.exists():
            raise FileNotFoundError(f"{f} missing. Run: python -m experiments.datasets.medicare_dsh.download")
        d = pd.read_csv(f, dtype={"provider": str}, low_memory=False)
        d.columns = [c.lower().strip() for c in d.columns]
        d["dsh_pub"] = d["dshopg"] if "dshopg" in d else d["dshopp"]
    else:
        z = RAW / "impact" / f"fy{year}_impact.zip"
        if not z.exists():
            raise FileNotFoundError(f"{z} missing. Run download.py")
        txt = [p for p in _extract(z).glob("*.txt")
               if re.search(r"final rule", p.name, re.I) and "descr" not in p.name.lower()]
        d = pd.read_csv(txt[0], sep="\t", skiprows=1, dtype={"Provider Number": str},
                        encoding="latin-1", low_memory=False)
        d.columns = [c.strip() for c in d.columns]
        ren = {"Provider Number": "provider", "Name": "provname", "Beds": "beds",
               "Average Daily Census": "adc", "DSHPCT": "dshpct", "DSHOPP": "dsh_pub",
               "URSPA": "urspa", "URGEO": "urgeo", "Provider Type": "ptype", "Bills": "bills",
               "Resident to Bed Ratio": "resbed", "Region": "region", "Medicare Percentage": "mcr_pct",
               "UCP_ADJ": "ucp_adj", "Operating CCR": "opccr", f"FY {year} Wage Index": "wageindex"}
        d = d.rename(columns=ren)
    cmi = [c for c in d.columns if re.fullmatch(r"tacmiv\d*", c, re.I)]
    cmi = [c for c in cmi if c.lower() == "tacmiv"] or sorted(cmi)[-1:]
    out = pd.DataFrame({"provider": d.provider.astype(str).str.strip().str.zfill(6), "year": year})
    for c in ("dshpct", "dsh_pub", "beds", "adc", "bills", "resbed", "mcr_pct", "wageindex", "opccr", "ptype", "region"):
        out[c] = pd.to_numeric(d[c], errors="coerce") if c in d else np.nan
    uc = "ucp_adj" if "ucp_adj" in d else ("ucp" if "ucp" in d else None)
    out["ucp_adj"] = pd.to_numeric(d[uc], errors="coerce") if uc else np.nan
    out["cmi"] = pd.to_numeric(d[cmi[0]], errors="coerce") if cmi else np.nan
    ur = d["urspa"] if "urspa" in d else d["urgeo"]
    out["urspa"] = ur.astype(str).str.strip().str.upper()
    out["urgeo"] = d["urgeo"].astype(str).str.strip().str.upper() if "urgeo" in d else out.urspa
    out["provname"] = d.get("provname", pd.Series([""] * len(d))).astype(str).str.strip()
    out = out[out.provider.str.fullmatch(r"\d{6}")]
    return out.drop_duplicates("provider")


def formula(dpp_pct, capped, factor):
    """Operating DSH adjustment (percent) from the DPP (percent)."""
    d = np.asarray(dpp_pct, float)
    raw = np.where(d < CUT, 0.0, np.where(d < KINK, 2.5 + 0.65 * (d - CUT), 5.88 + 0.825 * (d - KINK)))
    return np.where(np.asarray(capped, bool), np.minimum(raw, CAP), raw) * np.asarray(factor, float)


def classify(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rural"] = (~df.urspa.str.contains("URBAN")).astype(int)
    rrc = df.ptype.isin(RRC)
    mdh = (df.ptype.isin(MDH) & (df.year >= 2007) & (df.year != 2013)) | (df.ptype.isna() & (df.year == 2015))
    small_urban = (df.rural == 0) & ~(df.beds >= 100)
    small_rural = (df.rural == 1) & ~(df.beds >= 500) & ~rrc & ~mdh
    df["capped_class"] = (small_urban | small_rural).astype(int)
    df["factor"] = np.where(df.year >= 2014, 0.25, 1.0)
    df["pickle"] = (np.abs(df.dsh_pub - 0.35 * df.factor) < 1e-6).astype(int)
    df["dpp"] = 100 * df.dshpct
    df["A_formula"] = formula(df.dpp, df.capped_class, df.factor)
    df["A_pub"] = 100 * df.dsh_pub
    df["group"] = [f"{y}|{'capped' if c else 'uncapped'}" for y, c in zip(df.year, df.capped_class)]
    return df


def schedule(q, group):
    """Operating DSH adjustment (percent) from the DPP (percent); group = "<FY>|capped|uncapped"."""
    g = np.asarray(group).astype(str)
    yr = np.array([int(s.split("|")[0]) for s in g])
    capped = np.array([s.split("|")[1] == "capped" for s in g])
    return formula(q, capped, np.where(yr >= 2014, 0.25, 1.0))


def build_panel(force: bool = False) -> pd.DataFrame:
    if PANEL.exists() and not force:
        p = pd.read_csv(PANEL, dtype={"provider": str})
        if p.panel_version.iloc[0] == PANEL_VERSION:
            return p
    imp = classify(pd.concat([read_impact(y) for y in YEARS], ignore_index=True))
    key = imp.set_index(["provider", "year"])
    df = imp.copy()
    for lag, tag, cols in [(3, "_lead3", ["bills", "dpp", "beds"]), (-1, "_lag1", ["bills", "dpp"])]:
        idx = pd.MultiIndex.from_arrays([df.provider, df.year + lag])
        for c in cols:
            df[c + tag] = key[c].reindex(idx).to_numpy()
    present = set(zip(imp.provider, imp.year))
    df["exit2"] = [np.nan if y + 2 > max(YEARS) else float((p, y + 2) not in present)
                   for p, y in zip(df.provider, df.year)]
    df["panel_version"] = PANEL_VERSION
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(PANEL, index=False)
    return df


def _sample_frame(years):
    df = build_panel()
    st = df.provider.str[:2]
    df = df[df.year.between(*years) & ~st.isin(["21", "40"])]       # no Maryland, Puerto Rico
    df = df[(df.pickle == 0) & df.dpp.between(0, 100)]
    return df


def load_dose(years=(2008, 2013), outcome: str = "log_cases") -> RDDSample:
    """Hospital x FY sample for the dose design (see module docstring)."""
    df = _sample_frame(years).copy()
    df["log_cases"] = np.log(df.bills_lead3)
    df["exit"] = df.exit2
    df["dpp_lead"] = df.dpp_lead3
    df["log_beds"] = np.log(df.beds.where(df.beds > 0))
    df["occupancy"] = df.adc / df.beds.where(df.beds > 0)
    df["log_cases_lag"] = np.log(df.bills.where(df.bills > 0))
    df["mcr_pct"] = df.mcr_pct.where(df.mcr_pct <= 1.5)
    ycol = {"log_cases": "log_cases", "exit": "exit", "dpp_lead": "dpp_lead"}[outcome]
    df = df.dropna(subset=CONT + [ycol, "dpp"]).reset_index(drop=True)
    df["lurban"] = df.urspa.eq("LURBAN").astype(int)
    df["rrc"] = df.ptype.isin(RRC).astype(int)
    df["sch"] = df.ptype.isin({16, 17}).astype(int)
    df["mdh"] = df.ptype.isin(MDH).astype(int)
    df["teaching"] = (df.resbed > 0).astype(int)
    years_ = sorted(df.year.unique())
    yd = pd.get_dummies(df.year, prefix="fy").astype(float).reindex(columns=[f"fy_{y}" for y in years_[1:]], fill_value=0)
    states = sorted(df.provider.str[:2].unique())
    sd = pd.get_dummies(df.provider.str[:2], prefix="ssa_st").astype(float).reindex(
        columns=[f"ssa_st_{s}" for s in states[1:]], fill_value=0)
    X = pd.concat([df[CONT + ["rural", "lurban", "rrc", "sch", "mdh", "teaching"]].astype(float), yd, sd], axis=1)
    A = df.A_formula.to_numpy(float)
    extras = {
        "A": A, "a_ref": 0.0, "dose_levels": None, "schedule": schedule,
        "group": df.group.to_numpy(), "Z_names": list(Z_NAMES), "unit_id": df.provider.to_numpy(),
        "year": df.year.to_numpy(), "published_dose": df.A_pub.to_numpy(float),
        "capped_class": df.capped_class.to_numpy(int), "factor": df.factor.to_numpy(float),
        "ucp_adj": df.ucp_adj.to_numpy(float), "beds": df.beds.to_numpy(float),
        "log_cases": df.log_cases.to_numpy(float), "exit": df.exit.to_numpy(float),
        "dpp_lead": df.dpp_lead.to_numpy(float), "dpp_lag1": df.dpp_lag1.to_numpy(float),
        "cutoffs": {"eligibility": CUT, "kink": KINK, "cap_reached_at": KINK + (CAP - 5.88) / 0.825},
        "outcome": outcome,
    }
    return RDDSample(
        Q=df.dpp.to_numpy(float), X=X.to_numpy(float), Y=df[ycol].to_numpy(float), threshold=CUT,
        name="medicare_dsh", feature_names=list(X.columns),
        description=("Medicare operating DSH adjustment (42 CFR 412.106) on the disproportionate "
                     "patient percentage; jump at 15%, kink at 20.2%, 12% cap for small hospitals"),
        citation="CMS IPPS Impact Files (NBER mirror, FY 2005-2024; CMS FY 2025-2027); 42 CFR 412.106",
        treatment_rule=lambda q: (np.asarray(q) >= CUT).astype(int),
        extras=extras,
    )


def load() -> RDDSample:
    """Registry entry: the dose sample; D = 1{DPP >= 15%}."""
    return load_dose()


def rule_check() -> pd.DataFrame:
    """Formula vs published operating DSH adjustment, by FY (50 states + DC, no Maryland)."""
    df = classify(pd.concat([read_impact(y) for y in YEARS], ignore_index=True))
    st = df.provider.str[:2]
    df = df[~st.isin(["21", "40"]) & df.dpp.notna() & df.A_pub.notna()]
    rows = []
    for y, g in df.groupby("year"):
        np_ = g[g.pickle == 0]
        ok = np.abs(np_.A_formula - np_.A_pub) < 0.05
        rows.append({"fy": y, "n": len(g), "pickle": int(g.pickle.sum()), "match": float(ok.mean()),
                     "mismatch": int((~ok).sum()), "eligible_share": float((np_.dpp >= CUT).mean()),
                     "capped_class_share": float(np_.capped_class.mean()),
                     "at_cap_share": float((np_.A_formula >= CAP * np_.factor - 1e-9).mean())})
    return pd.DataFrame(rows)


def readiness(s: RDDSample | None = None) -> dict:
    s = s or load_dose()
    Xt = np.column_stack([np.ones(s.n), s.X])
    T = Xt @ np.linalg.lstsq(Xt, s.Q, rcond=None)[0]
    eta = s.Q - T
    A = s.extras["A"]
    ref = A <= 1e-12
    pos = ~ref
    cuts = np.quantile(A[pos], np.linspace(0, 1, 11))
    by = [("ref", int(ref.sum()), *np.round(np.quantile(eta[ref], [0.1, 0.5, 0.9]), 2))]
    for j in range(10):
        m = pos & (A >= cuts[j]) & (A <= cuts[j + 1])
        by.append((f"d{j + 1}:{cuts[j]:.2f}-{cuts[j + 1]:.2f}", int(m.sum()),
                   *np.round(np.quantile(eta[m], [0.1, 0.5, 0.9]), 2)))

    def ratio(points, h=0.5):
        out = {}
        for c in points:
            up = int(((s.Q >= c) & (s.Q < c + h)).sum())
            dn = int(((s.Q >= c - h) & (s.Q < c)).sum())
            out[c] = (up, dn, round(up / max(dn, 1), 2))
        return out
    return {"n": s.n, "units": len(np.unique(s.extras["unit_id"])), "p": s.p,
            "R2": float(np.var(T) / np.var(s.Q)), "sd_Q": float(s.Q.std()), "sd_T": float(T.std()),
            "sd_eta": float(eta.std()), "share_ref": float(ref.mean()), "by_dose": by,
            "bunching": ratio([CUT, KINK]), "bunching_cap": ratio([s.extras["cutoffs"]["cap_reached_at"]]),
            "placebo": ratio([10.0, 12.5, 17.5, 25.0, 30.0, 35.0])}
