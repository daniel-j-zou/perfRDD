"""CMS Hospital Readmissions Reduction Program (HRRP): a kinked dose on readmission ratios.

From FY 2019 (42 CFR 412.154, peer-group method), a subsection (d) hospital's base
operating DRG payments in fiscal year t are multiplied by

    factor = 1 - min(0.03, NM * sum_c 1{n_c >= 25, ERR_c > med_c} * w_c * (ERR_c - med_c))

over six conditions c (AMI, COPD, HF, pneumonia, CABG, THA/TKA). ERR_c is the hospital's
excess readmission ratio over a three-year window ending June 30 of t-2, med_c the
median ERR of its dual-proportion peer group (quintile), w_c the condition's share of
the hospital's base DRG payments, n_c its eligible discharges, NM the year's neutrality
modifier. The payment reduction is 100 * (1 - factor) percent, capped at 3.

The score is six-dimensional and the dose is continuous. ``load_dose(cond)`` takes one
condition's excess as the running variable:

Q = ERR_c - med_c (default c = HF, which nearly every hospital has); threshold 0.
A = published payment reduction (%). Holding the other conditions fixed, A is
    min(3, a_other + slope * max(Q, 0)): flat below 0, a kink at 0 with a
    hospital-specific slope 100 * NM * w_c (0 when n_c < 25).
X = dual proportion, peer-group, state and FY indicators, log eligible discharges and
    DRG payment ratio for each condition.
Y = ERR_c in the FY t+3 file, whose window (July t-2 to June t+1) does not overlap FY t's.

Units are hospital (CCN) x fiscal year, FY 2019-2027 except FY 2023 (its cms.gov page
links the FY 2024 file).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw" / "supp"
PROCESSED = HERE / "data" / "processed"
CACHE = PROCESSED / "hrrp_panel.csv.gz"
CONDS = ("ami", "copd", "hf", "pneumonia", "cabg", "thatka")
FILES = {
    2019: "hrrp_fy2019/Section 508 - FR FY 19 - tab - fy_2019_ipps_final_rule_hrrp_supplemental_file.txt",
    2020: "hrrp_fy2020/FY2020_IPPS_Final_Rule_HRRP_Supplemental_File tab2.txt",
    2021: "hrrp_fy2021/FY2021_Final_Rule_Supplemental_File_FR FY 2021.txt",
    2022: "hrrp_fy2022/508_Compliant_Version_of_FR FY 2022.csv",
    2024: "hrrp_fy2024/508 version of FY2024_Final_Rule_Supplemental_File Variations FR FY 2024 tab .txt",
    2025: "hrrp_fy2025/508-Version-FR-FY-2025.csv",
    2026: "hrrp_fy2026/FR FY 2026.txt",
    2027: "hrrp_fy2027/FY2027_HRRP_Supplemental_File - FR FY 2027 Tab.txt",
}
CAP = 3.0


def _read(fy: int) -> pd.DataFrame:
    f = RAW / FILES[fy]
    if not f.exists():
        raise FileNotFoundError(f"{f} missing; run the download script (see README).")
    sep = "," if f.suffix == ".csv" else "\t"
    lines = f.read_text(encoding="latin-1").splitlines()
    k = next(i for i, l in enumerate(lines) if l.lstrip("﻿ï»¿").startswith("Hospital CCN"))
    df = pd.read_csv(f, sep=sep, skiprows=k, dtype=str, encoding="latin-1")
    df.columns = [" ".join(str(c).split()).lower() for c in df.columns]
    num = lambda c: pd.to_numeric(df[c].astype(str).str.replace(",", "").str.replace("%", ""),
                                  errors="coerce")
    out = pd.DataFrame({"ccn": df["hospital ccn"].astype(str).str.strip().str.zfill(6),
                        "fy": fy,
                        "factor": num("payment adjustment factor"),
                        "dual": num("dual proportion"),
                        "peer": num("peer group assignment"),
                        "nm": num("neutrality modifier")})
    names = {"ami": "ami", "copd": "copd", "hf": "hf", "pneumonia": "pneumonia",
             "cabg": "cabg", "thatka": "tha/tka"}
    for c, lab in names.items():
        out[f"n_{c}"] = num(f"number of eligible discharges for {lab}")
        out[f"err_{c}"] = num(f"err for {lab}")
        out[f"med_{c}"] = num(f"peer group median err for {lab}")
        out[f"w_{c}"] = num(f"drg payment ratio for {lab}")
        out[f"pen_{c}"] = (df[f"penalty indicator for {lab}"].astype(str).str.strip() == "Y").astype(int)
    out["red_pub"] = (100 * (1 - out["factor"])).round(2)
    return out[out["ccn"].str.fullmatch(r"\d{6}")].reset_index(drop=True)


def contributions(df: pd.DataFrame) -> pd.DataFrame:
    """Each condition's term NM * w_c * (ERR_c - med_c), in percent (0 if not penalized)."""
    out = {}
    for c in CONDS:
        elig = (df[f"n_{c}"] >= 25) & (df[f"err_{c}"] > df[f"med_{c}"])
        term = df["nm"] * df[f"w_{c}"].fillna(0) * (df[f"err_{c}"] - df[f"med_{c}"])
        out[c] = 100 * term.where(elig, 0.0).fillna(0.0)
    return pd.DataFrame(out, index=df.index)


def _build() -> pd.DataFrame:
    df = pd.concat([_read(fy) for fy in FILES], ignore_index=True)
    df = df.drop_duplicates(["fy", "ccn"])
    con = contributions(df)
    df["index_pct"] = con.sum(1)                          # uncapped penalty index
    df["red_formula"] = np.minimum(CAP, df["index_pct"])
    for c in CONDS:
        df[f"q_{c}"] = df[f"err_{c}"] - df[f"med_{c}"]
        df[f"other_{c}"] = df["index_pct"] - con[c]
        df[f"slope_{c}"] = 100 * df["nm"] * df[f"w_{c}"].fillna(0) * (df[f"n_{c}"] >= 25)
    lead = df[["ccn", "fy"] + [f"err_{c}" for c in CONDS] + [f"q_{c}" for c in CONDS]
              + ["red_pub"]].copy()
    for h in (3, 4):
        ld = lead.assign(fy=lead["fy"] - h)
        ld = ld.rename(columns={c: f"{c}_t{h}" for c in ld.columns if c not in ("ccn", "fy")})
        df = df.merge(ld, on=["ccn", "fy"], how="left")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE, index=False)
    return df


def panel(rebuild: bool = False) -> pd.DataFrame:
    if CACHE.exists() and not rebuild:
        return pd.read_csv(CACHE, dtype={"ccn": str})
    return _build()


def rule_check() -> pd.DataFrame:
    """Published reduction vs the formula, and penalty indicators vs the eligibility rule."""
    df = panel()
    df = df[df["factor"].notna()]
    f4 = 1 - df["red_formula"] / 100
    pen_ok = np.all([((df[f"n_{c}"] >= 25) & (df[f"err_{c}"] > df[f"med_{c}"])).astype(int)
                     == df[f"pen_{c}"] for c in CONDS], axis=0)
    out = df.assign(abs_err=(df["red_pub"] - df["red_formula"]).abs(),
                    ok_round=np.isclose(f4.round(4), df["factor"], atol=1e-9),
                    ok_trunc=np.isclose(np.floor(f4 * 1e4 + 1e-9) / 1e4, df["factor"], atol=1e-9),
                    ok_ceil=np.isclose(np.ceil(f4 * 1e4 - 1e-9) / 1e4, df["factor"], atol=1e-9),
                    ok_pen=pen_ok)
    return out.groupby("fy").agg(
        hospitals=("ccn", "size"), reduced=("red_pub", lambda s: float((s > 0).mean())),
        capped=("red_pub", lambda s: float((s >= CAP - 1e-9).mean())),
        agree_round4=("ok_round", "mean"), agree_trunc4=("ok_trunc", "mean"),
        agree_ceil4=("ok_ceil", "mean"), max_abs_diff_pct=("abs_err", "max"),
        penalty_flags_agree=("ok_pen", "mean"))


def _x(df: pd.DataFrame):
    base = {"dual": df["dual"]}
    for c in CONDS:
        base[f"log_n_{c}"] = np.log1p(df[f"n_{c}"].fillna(0))
        base[f"w_{c}"] = df[f"w_{c}"].fillna(0)
    X = pd.DataFrame(base, index=df.index)
    for name, col in (("peer", "peer"), ("fy", "fy")):
        lv = sorted(df[col].dropna().unique())
        d = pd.get_dummies(df[col]).reindex(columns=lv[1:], fill_value=0)
        d.columns = [f"{name}_{int(v)}" for v in d.columns]
        X = pd.concat([X, d], axis=1)
    st = df["ccn"].str[:2]
    lv = sorted(st.unique())
    d = pd.get_dummies(st).reindex(columns=lv[1:], fill_value=0)
    d.columns = [f"state_{v}" for v in d.columns]
    X = pd.concat([X, d], axis=1).astype(float)
    return X.to_numpy(), list(X.columns)


def load_dose(cond: str = "hf", horizon: int = 3) -> RDDSample:
    """Q = ERR_cond - peer median; A = payment reduction (%); Y = ERR_cond at FY t+horizon."""
    df = panel()
    y = f"err_{cond}_t{horizon}"
    keep = df[f"q_{cond}"].notna() & df[y].notna() & (df[f"n_{cond}"] >= 25) & df["red_pub"].notna()
    df = df[keep].reset_index(drop=True)
    X, names = _x(df)
    Q = df[f"q_{cond}"].to_numpy(float)
    other = df[f"other_{cond}"].to_numpy(float)
    slope = df[f"slope_{cond}"].to_numpy(float)

    def schedule(q, group):
        """Dose (%) of row(s) `group` of this sample at excess q (other conditions fixed)."""
        g = np.asarray(group, int)
        return np.minimum(CAP, other[g] + slope[g] * np.maximum(np.asarray(q, float), 0.0))

    rows = np.arange(len(df))
    return RDDSample(
        Q=Q, X=X, Y=df[y].to_numpy(float), threshold=0.0,
        name=f"cms_hrrp_{cond}", feature_names=names,
        description=(f"CMS HRRP, FY {sorted(df['fy'].unique().tolist())}, hospitals with >= 25 "
                     f"{cond.upper()} discharges. Q = ERR - peer-group median ERR for {cond}; "
                     f"A = payment reduction (%), kinked at Q = 0; Y = ERR for {cond} in FY t+{horizon}."),
        citation=("CMS IPPS final rules FY 2019-2027, HRRP supplemental data files (cms.gov); "
                  "42 CFR 412.154."),
        extras={
            "A": schedule(Q, rows), "a_ref": other, "dose_levels": None,
            "schedule": schedule, "group": rows, "fy": df["fy"].to_numpy(int),
            "slope": slope, "cap": CAP,
            "Z_names": ["dual", f"log_n_{cond}", f"w_{cond}"],
            "published_dose": df["red_pub"].to_numpy(float),
            "ccn": df["ccn"].to_numpy(), "raw": df,
        },
    )


def load() -> RDDSample:
    return load_dose()


def readiness(s: Optional[RDDSample] = None) -> dict:
    s = s or load_dose()
    Z = np.column_stack([np.ones(s.n), s.X])
    T = Z @ np.linalg.lstsq(Z, s.Q, rcond=None)[0]
    eta = s.Q - T
    A = s.extras["A"]
    fy = s.extras["fy"]
    qd = s.Q - pd.Series(s.Q).groupby(fy).transform("mean").to_numpy()
    above = s.Q > 0
    out = {"n": int(s.n), "hospitals": int(len(set(s.extras["ccn"]))), "p": int(s.p),
           "R2": float(1 - eta.var() / s.Q.var()), "R2_within_fy": float(1 - eta.var() / qd.var()),
           "sd_Q": float(s.Q.std()), "sd_T": float(T.std()), "sd_eta": float(eta.std()),
           "share_dose0": float((A == 0).mean()), "share_capped": float((A >= CAP).mean()),
           "share_Q_above0": float(above.mean()),
           "A_quantiles_10_50_90": np.quantile(A, [.1, .5, .9]).round(3).tolist(),
           "eta_q_by_side": {"Q<=0": np.quantile(eta[~above], [.1, .5, .9]).round(4).tolist(),
                             "Q>0": np.quantile(eta[above], [.1, .5, .9]).round(4).tolist()},
           "agree_published": float(np.mean(np.abs(A - s.extras["published_dose"]) < 0.011)),
           "corr_YQ": float(np.corrcoef(s.Y, s.Q)[0, 1])}
    h = 0.25 * eta.std()
    out["near_kink_share"] = float((np.abs(s.Q) < h).mean())
    # density around the kink: counts in [0, h) vs [-h, 0), and the same at placebo points
    def ratio(c, w=0.01):
        up = ((s.Q >= c) & (s.Q < c + w)).sum(); dn = ((s.Q < c) & (s.Q >= c - w)).sum()
        return [int(up), int(dn), round(up / dn, 3) if dn else None]
    out["bunching_kink"] = ratio(0.0)
    out["bunching_placebo"] = {c: ratio(c) for c in (-0.06, -0.03, 0.03, 0.06)}
    return out
