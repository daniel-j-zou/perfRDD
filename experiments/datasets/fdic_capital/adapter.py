"""US bank prompt-corrective-action (PCA) capital categories from FDIC call-report data:
"well capitalized" is a both rule on three (to 2014) or four (2015 on) capital ratios.

Under section 38 of the FDI Act, an insured bank is well capitalized if every capital ratio
is at or above its cutoff (and no written agreement, order or directive requires a specific
capital level), and it is undercapitalized if any ratio is below its "adequately capitalized"
cutoff. The cutoffs (%) are

    well:       total RBC >= 10, tier 1 RBC >= 6, leverage >= 5               (to 2014)
                total RBC >= 10, tier 1 RBC >= 8, CET1 >= 6.5, leverage >= 5  (2015 on)
    adequately: 8 / 4 / 4 (to 2014),  8 / 6 / 4.5 / 4 (2015 on)

(12 CFR 325.103 and the identical OCC 6.4, Fed 208.43 and OTS 565.4 rules to 2014-12-31;
12 CFR 324.403 and its OCC and Fed twins from 2015-01-01, 12 CFR 324.401(c)). From 2020 a
qualifying community bank may elect the community bank leverage ratio (CBLR) framework and is
then deemed well capitalized with a leverage ratio above 9% (8% in 2020 Q2-Q4, 8.5% in 2021;
12 CFR 324.12, 324.303); electors report no risk-based ratios and are dropped here.

Ratios are recomputed at full precision from the PCA components in the FDIC BankFind
financials (the API's own ratio fields are rounded or equal to these, see ``rule_check``):
total = RBC / RWAJT, tier 1 = RBCT1J / RWAJ, CET1 = RBCT1C / RWAJ, leverage = RBCT1 / AVASSETJ.

Q = load(): the binding margin min_j (ratio_j - well cutoff_j) in percentage points, so
    D = 1{Q >= 0} = well capitalized by ratios. load_two_score(): (total RBC, leverage),
    D = 1{total >= 10 and leverage >= 5}, dropping the 3 bank-years where tier 1 RBC or CET1
    alone fails, so D = well capitalized for every row. load_adequate*(): the binding margin to the
    adequately-capitalized cutoffs, D = 1{adequately capitalized or better}. Q is capped at
    10 pp below and 30 pp above each cutoff (D unchanged); extras keep the uncapped values.
X = log total assets, loans / assets, deposits / assets, real-estate and C&I loans / assets,
    securities / assets, brokered deposits / assets, noncurrent loans / loans (all at the
    previous quarter-end), log(1 + age), charter class, holding-company status,
    specialization group, state, mutual, subchapter S, metro, community-bank flag and year
    indicators. No capital ratio and no same-quarter balance-sheet item.
Y = load(), load_two_score(), load_adequate(), load_quarterly(): log(gross loans t+4
    quarters) - log(gross loans t), clipped to [-1, 1] (0.9% of rows). The ``*_fail`` loaders: 1{the bank fails within the next 8 quarters} (FDIC
    failures list, resolution type FAILURE).

Main sample: fourth quarters 1992-2024 (2023 for failure), insured commercial banks and
savings institutions (BKCLASS N, NM, SM, SB, SI, SL), not CBLR electors, with complete ratios,
a call report at the previous quarter-end, and Y observed. Capital ratios are very persistent,
so the main loaders use one quarter per year; ``load_quarterly()`` keeps every quarter.
"""
from __future__ import annotations

import glob
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample
from experiments.datasets.fdic_capital.download import LAST, RAW

HERE = Path(__file__).parent
PROCESSED = HERE / "data" / "processed"
CACHE = PROCESSED / "fdic_capital_panel.parquet"

PCA_START = pd.Timestamp("1992-12-31")      # PCA rules effective 1992-12-19
BASEL3 = pd.Timestamp("2015-01-01")         # 12 CFR 324.401(c)
LAST_Q = pd.Timestamp(LAST)                 # last call report in the data, 2026-06-30
CLASSES = ("N", "NM", "SM", "SB", "SI", "SL")   # insured banks and thrifts; drops NC, OI
RATIOS = ("tot", "t1", "cet1", "lev")
# Cutoffs (%) by regime: 12 CFR 325.103(b) to 2014, 12 CFR 324.403(b) from 2015.
WELL = {"pre2015": {"tot": 10.0, "t1": 6.0, "lev": 5.0},
        "basel3": {"tot": 10.0, "t1": 8.0, "cet1": 6.5, "lev": 5.0}}
ADEQ = {"pre2015": {"tot": 8.0, "t1": 4.0, "lev": 4.0},
        "basel3": {"tot": 8.0, "t1": 6.0, "cet1": 4.5, "lev": 4.0}}
SIGNIF = {"pre2015": {"tot": 6.0, "t1": 3.0, "lev": 3.0},
          "basel3": {"tot": 6.0, "t1": 4.0, "cet1": 3.0, "lev": 3.0}}
CRITICAL = 2.0                              # tangible equity / assets <= 2 (proxied by leverage)
CATEGORY = {4: "well", 3: "adequate", 2: "under", 1: "significantly_under", 0: "critically_under"}
# Acquisitions by the bank (history change codes; the bank is ACQ_CERT).
ACQ_CODES = {211, 215, 216, 217, 221, 222, 223, 224, 225}
Y_CLIP = 1.0        # y_dloan is clipped to [-1, 1] (loans x0.37 to x2.72); y_dloan_raw is not
# Q is capped to cutoff - 10 .. cutoff + 30 pp (margins in [-10, 30]); D is unchanged for any
# cutoff inside that range. Uncapped values are in extras.
Q_CAP = (-10.0, 30.0)

_STR = ["NAME", "BKCLASS", "CHRTAGNT", "REGAGNT", "STALP"]
_NUM = ["CERT", "REPDTE", "SPECGRP", "HCTMULT", "HCTNONE", "ESTYMD", "CB", "MUTUAL", "SUBCHAPS",
        "CALLFORM", "FORM31", "METRO", "OFFDOM", "CERTCONS", "DENOVO",
        "RBCRWAJ", "RBC1RWAJ", "IDT1RWAJR", "RBC1AAJ", "IDT1CER", "RBCT1CER", "CBLRIND",
        "RBC", "RBCT1", "RBCT1J", "RBCT1C", "RWAJ", "RWAJT", "AVASSETJ", "EQ",
        "ASSET", "LNLSGR", "LNLSNET", "LNRE", "LNCI", "SC", "NCLNLS", "DEP", "DEPDOM", "BRO",
        "NETINC", "ROA"]


def _require() -> None:
    if not glob.glob(str(RAW / "financials" / "*_p0.csv.gz")):
        raise FileNotFoundError(f"{RAW / 'financials'} is empty. Run "
                                "python -m experiments.datasets.fdic_capital.download")


# --------------------------------------------------------------------------------- rule

def regime(date) -> np.ndarray:
    return np.where(pd.to_datetime(date) < BASEL3, "pre2015", "basel3")


def cblr_cutoff(date) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CBLR leverage cutoff (%), whether equality qualifies, and the grace-period floor, by
    report date: > 9 in 2020 Q1 and from 2022 (grace if > 8); >= 8 from 2020-04-23 to
    2020-12-31 (grace if >= 7); > 8.5 in 2021 (grace if > 7.5). 12 CFR 324.12(a)(1), (c),
    324.303(a), (b), (d). The 8% rule effective 2026-07-01 (91 FR 22973) is after the data."""
    d = pd.to_datetime(pd.Series(date)).to_numpy()
    cut = np.full(len(d), 9.0)
    eq = np.zeros(len(d), bool)
    q2_20 = (d >= np.datetime64("2020-06-30")) & (d <= np.datetime64("2020-12-31"))
    y21 = (d >= np.datetime64("2021-01-01")) & (d <= np.datetime64("2021-12-31"))
    cut[q2_20], eq[q2_20] = 8.0, True
    cut[y21] = 8.5
    return cut, eq, cut - 1.0


def _margins(df: pd.DataFrame, cuts: dict) -> pd.DataFrame:
    """ratio - cutoff for each ratio under the row's regime; +inf where a ratio is not part
    of the rule (CET1 before 2015)."""
    pre = (df["date"] < BASEL3).to_numpy()
    out = pd.DataFrame(index=df.index)
    for r in RATIOS:
        c_pre = cuts["pre2015"].get(r, np.nan)
        c_post = cuts["basel3"][r]
        v = df[r].to_numpy(float)
        m = v - np.where(pre, c_pre, c_post)
        if r not in cuts["pre2015"]:
            m = np.where(pre, np.inf, m)
        out[r] = m
    return out


def binding(df: pd.DataFrame, cuts: dict) -> tuple[np.ndarray, np.ndarray]:
    """(Q, binding ratio): Q = min_j (ratio_j - cutoff_j), NaN if any rule ratio is missing."""
    M = _margins(df, cuts)
    arr = M.to_numpy(float)
    bad = np.isnan(arr).any(axis=1)
    arr2 = np.where(np.isnan(arr), np.inf, arr)
    q = arr2.min(axis=1)
    b = np.asarray(RATIOS)[arr2.argmin(axis=1)]
    return np.where(bad, np.nan, q), np.where(bad, "", b)


def category(df: pd.DataFrame) -> np.ndarray:
    """PCA category from the ratios alone (4 well ... 0 critically undercapitalized); -1 if
    a ratio is missing. Critically undercapitalized uses leverage <= 2 as a proxy for
    tangible equity / total assets <= 2. CBLR electors: well if leverage beats the CBLR
    cutoff or its grace-period floor (an elector still in the framework below the cutoff is
    in its grace period; the two-quarter limit is not checked), else adequately capitalized
    if leverage >= 4."""
    qw, _ = binding(df, WELL)
    qa, _ = binding(df, ADEQ)
    qs, _ = binding(df, SIGNIF)
    cat = np.where(qw >= 0, 4, np.where(qa >= 0, 3, np.where(qs >= 0, 2, 1)))
    cat = np.where(df["lev"].to_numpy(float) <= CRITICAL, 0, cat)
    cat = np.where(np.isnan(qw), -1, cat)
    cb = (df["CBLRIND"] == 1).to_numpy()
    if cb.any():
        cut, eq, floor = cblr_cutoff(df.loc[cb, "date"])
        lev = df.loc[cb, "lev"].to_numpy(float)
        well = (lev > floor) | (eq & (lev >= floor))
        c = np.where(well, 4, np.where(lev >= 4, 3, np.where(lev >= 3, 2, 1)))
        c = np.where(lev <= CRITICAL, 0, c)
        cat[cb] = np.where(np.isnan(lev), -1, c)
    return cat


def well_rule(Q: np.ndarray) -> np.ndarray:
    """D = 1{Q >= 0}: every ratio at or above its well-capitalized cutoff."""
    return (np.asarray(Q) >= 0).astype(int)


def two_score_rule(Q: np.ndarray) -> np.ndarray:
    """D = 1{total RBC >= 10 and leverage >= 5}."""
    return ((Q[:, 0] >= 10.0) & (Q[:, 1] >= 5.0)).astype(int)


# -------------------------------------------------------------------------------- build

def _read_financials() -> pd.DataFrame:
    files = sorted(glob.glob(str(RAW / "financials" / "*_p*.csv.gz")))
    dtype = {**{c: str for c in _STR}, **{c: float for c in _NUM}}
    parts = [pd.read_csv(f, dtype=dtype, usecols=_STR + _NUM, keep_default_na=False,
                         na_values=[""]) for f in files]
    f = pd.concat(parts, ignore_index=True)
    for c in _STR:
        f[c] = f[c].fillna("").astype("category")
    f["CERT"] = f["CERT"].astype(int)
    f["REPDTE"] = f["REPDTE"].astype(int)
    return f


def _read_failures() -> pd.DataFrame:
    fl = pd.read_csv(RAW / "failures" / "failures_1991_2026.csv.gz", dtype=str,
                     keep_default_na=False)
    fl["fdate"] = pd.to_datetime(fl.FAILDATE, format="%m/%d/%Y")
    fl["CERT"] = pd.to_numeric(fl.CERT, errors="coerce")
    return fl.dropna(subset=["CERT"]).astype({"CERT": int})


def _read_history() -> pd.DataFrame:
    h = pd.concat([pd.read_csv(p, dtype=str, keep_default_na=False)
                   for p in sorted(glob.glob(str(RAW / "history" / "history_*.csv.gz")))])
    h["code"] = pd.to_numeric(h.CHANGECODE, errors="coerce")
    h["acq"] = pd.to_numeric(h.ACQ_CERT, errors="coerce")
    h["out"] = pd.to_numeric(h.OUT_CERT, errors="coerce")
    h["eff"] = pd.to_datetime(h.EFFDATE.str[:10])
    return h


def _qidx(d: pd.Series) -> pd.Series:
    d = pd.to_datetime(d)
    return d.dt.year * 4 + d.dt.quarter - 1


def _shift(f: pd.DataFrame, cols: list[str], k: int, suffix: str) -> pd.DataFrame:
    """Values of ``cols`` at quarter qidx + k for the same CERT (NaN if no report then)."""
    s = f[["CERT", "qidx"] + cols].copy()
    s["qidx"] = s["qidx"] - k
    return f[["CERT", "qidx"]].merge(s.rename(columns={c: c + suffix for c in cols}),
                                     on=["CERT", "qidx"], how="left").drop(
        columns=["CERT", "qidx"]).set_index(f.index)


def _build() -> pd.DataFrame:
    _require()
    f = _read_financials()
    f["date"] = pd.to_datetime(f.REPDTE.astype(str), format="%Y%m%d")
    f["qidx"] = _qidx(f.date)
    f["year"] = f.date.dt.year
    f["quarter"] = f.date.dt.quarter
    with np.errstate(divide="ignore", invalid="ignore"):
        f["tot"] = 100 * f.RBC / f.RWAJT.where(f.RWAJT > 0)
        f["t1"] = 100 * f.RBCT1J / f.RWAJ.where(f.RWAJ > 0)
        f["cet1"] = 100 * f.RBCT1C / f.RWAJ.where(f.RWAJ > 0)
        f["lev"] = 100 * f.RBCT1 / f.AVASSETJ.where(f.AVASSETJ > 0)
        a = f.ASSET.where(f.ASSET > 0)
        f["sh_loans"] = f.LNLSGR / a
        f["sh_dep"] = f.DEP / a
        f["sh_re"] = f.LNRE / a
        f["sh_ci"] = f.LNCI / a
        f["sh_sc"] = f.SC / a
        f["sh_bro"] = f.BRO.fillna(0) / a
        f["sh_ncl"] = (f.NCLNLS / f.LNLSGR.where(f.LNLSGR > 0)).fillna(0)
        f["log_asset"] = np.log(a)
        f["log_loans"] = np.log(f.LNLSGR.where(f.LNLSGR > 0))
    for name, cuts in [("q_well", WELL), ("q_adeq", ADEQ)]:
        q, b = binding(f, cuts)
        f[name], f[name.replace("q_", "bind_")] = q, b
    f["category"] = category(f)
    # previous quarter-end (X), four quarters ahead (Y) and back (placebo)
    lag_cols = ["log_asset", "sh_loans", "sh_dep", "sh_re", "sh_ci", "sh_sc", "sh_bro",
                "sh_ncl", "tot", "t1", "lev", "q_well"]
    f = f.join(_shift(f, lag_cols, -1, "_l1"))
    f = f.join(_shift(f, ["log_loans", "q_well", "category", "sh_bro"], 4, "_f4"))
    f = f.join(_shift(f, ["log_loans", "q_well"], -4, "_l4"))
    f["y_dloan_raw"] = f.log_loans_f4 - f.log_loans
    f["y_dloan"] = f.y_dloan_raw.clip(-Y_CLIP, Y_CLIP)
    f["y_dbro"] = f.sh_bro_f4 - f.sh_bro
    f["placebo_dloan"] = f.log_loans - f.log_loans_l4
    last = int(_qidx(pd.Series([LAST_Q])).iloc[0])
    f["exit_4q"] = np.where(f.qidx + 4 <= last, f.log_loans_f4.isna(), np.nan)
    # failures within the next 8 quarters
    fl = _read_failures()
    fl["fq"] = _qidx(fl.fdate)
    for col, kind in [("y_fail8", "FAILURE"), ("y_assist8", "ASSISTANCE")]:
        first = fl[fl.RESTYPE == kind].groupby("CERT").fq.min()
        fq = f.CERT.map(first)
        hit = (fq > f.qidx) & (fq <= f.qidx + 8)
        f[col] = np.where(f.qidx + 8 <= last, hit.astype(float), np.nan)
    fail = fl[fl.RESTYPE == "FAILURE"].groupby("CERT").fdate.min()
    f["fail_date"] = f.CERT.map(fail)
    # acquisitions of other institutions by this bank in the next 4 quarters
    h = _read_history()
    h = h[h.code.isin(ACQ_CODES) & h.acq.notna() & (h.acq != h.out)]
    acq = set(zip(h.acq.astype(int), _qidx(h.eff)))
    nxt = np.zeros(len(f), bool)
    cert, qi = f.CERT.to_numpy(), f.qidx.to_numpy()
    for k in range(1, 5):
        nxt |= np.fromiter(((c, q + k) in acq for c, q in zip(cert, qi)), bool, len(f))
    f["acq_next4"] = nxt.astype(float)
    # age
    est = pd.to_datetime(f.ESTYMD.astype("Int64").astype(str), format="%Y%m%d",
                         errors="coerce")
    f["age"] = ((f.date - est).dt.days / 365.25).clip(lower=0)
    drop = ["CHRTAGNT", "CALLFORM", "FORM31", "OFFDOM", "CERTCONS", "DENOVO", "RBC", "RBCT1",
            "RBCT1J", "RBCT1C", "RWAJ", "RWAJT", "AVASSETJ", "EQ", "LNLSNET", "LNRE", "LNCI",
            "SC", "NCLNLS", "DEP", "DEPDOM", "NETINC", "ROA", "sh_loans", "sh_dep", "sh_re",
            "sh_ci", "sh_sc", "sh_ncl", "log_loans_f4", "sh_bro_f4", "log_loans_l4",
            "q_well_l4", "ESTYMD"]
    return f.drop(columns=drop)


def build() -> pd.DataFrame:
    """The bank-quarter panel 1991Q4-2026Q2 with ratios, rule, lags, outcomes (cached)."""
    if not CACHE.exists():
        df = _build()
        PROCESSED.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=PROCESSED, prefix=".tmp_", suffix=".parquet")
        os.close(fd)
        try:
            df.to_parquet(tmp, index=False, compression="zstd")
            os.chmod(tmp, 0o644)
            os.replace(tmp, CACHE)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    return pd.read_parquet(CACHE)


# ------------------------------------------------------------------------------- checks

# FDIC Quarterly Banking Profile graph books, "Capital Category Distribution", December 31 of
# each year (data/raw/qbp/qbpgr_<year>dec.pdf; read with pdftotext on 2026-10-28), and the
# June 30, 2020 web table (qbp/capital_category_2020jun.html). DIF-member institutions
# excluding US branches of foreign banks; categories by ratios only, CBLR electors (2020) as
# well capitalized. (well, adequately, under, significantly under, critically under).
QBP_COUNTS = {
    "2006-12-31": (8621, 57, 3, 0, 0), "2007-12-31": (8444, 79, 6, 2, 2),
    "2008-12-31": (8098, 147, 31, 13, 16), "2009-12-31": (7652, 172, 91, 50, 47),
    "2010-12-31": (7338, 140, 80, 66, 33), "2011-12-31": (7082, 101, 78, 72, 24),
    "2012-12-31": (6882, 91, 55, 41, 14), "2013-12-31": (6667, 72, 39, 26, 8),
    "2014-12-31": (6415, 52, 24, 14, 4), "2015-12-31": (6112, 35, 16, 17, 2),
    "2016-12-31": (5865, 22, 9, 12, 5), "2017-12-31": (5637, 22, 5, 6, 0),
    "2018-12-31": (5384, 11, 5, 5, 1), "2019-12-31": (5163, 6, 3, 3, 2),
    "2020-06-30": (5057, 2, 3, 2, 2),
}


def rule_check() -> dict:
    """The recomputed ratios against the API's ratio fields, the rule's category counts
    against the QBP's published counts, and coverage."""
    p = build()
    ins = p.BKCLASS.isin(CLASSES)
    out: dict = {}
    agree = {}
    for mine, api, rnd in [("tot", "RBCRWAJ", None), ("t1", "IDT1RWAJR", None),
                           ("t1", "RBC1RWAJ", 2), ("lev", "RBC1AAJ", None),
                           ("cet1", "IDT1CER", None), ("cet1", "RBCT1CER", 2)]:
        m = ins & p[mine].notna() & p[api].notna() & (p[api] != 0)
        v = p.loc[m, mine].round(rnd) if rnd else p.loc[m, mine]
        tol = 0.0051 if rnd else 1e-6
        agree[f"{mine} vs {api}"] = (int(m.sum()), float((np.abs(v - p.loc[m, api]) < tol).mean()))
    out["ratio_agreement (n, share)"] = agree
    rows = []
    for d, pub in QBP_COUNTS.items():
        s = p[ins & (p.date == pd.Timestamp(d))]
        mine = [int((s.category == k).sum()) for k in (4, 3, 2, 1, 0)]
        rows.append({"date": d, "qbp": pub, "rule": tuple(mine),
                     "missing_ratios": int((s.category == -1).sum()),
                     "not_well_qbp": sum(pub[1:]), "not_well_rule": sum(mine[1:])})
    out["qbp_vs_rule"] = rows
    base = ins & (p.date >= PCA_START)
    out["bank_quarters_from_1992Q4"] = int(base.sum())
    out["missing_ratios_share"] = float((p.q_well.isna() & base & (p.CBLRIND != 1)).sum()
                                        / base.sum())
    out["cblr_electors"] = int((base & (p.CBLRIND == 1)).sum())
    return out


def density_check(df: pd.DataFrame | None = None, widths=(0.1, 0.25, 0.5)) -> dict:
    """Counts just below and just above each cutoff. For ratio j, only rows whose other rule
    ratios meet their well-capitalized cutoffs (so ratio j alone decides D)."""
    df = _frame("y_dloan") if df is None else df
    M = _margins(df, WELL)
    out = {}
    for r in RATIOS:
        others = [o for o in RATIOS if o != r]
        ok = (M[others] >= 0).all(axis=1) & np.isfinite(M[r])
        m = M.loc[ok, r]
        out[r] = {w: (int(((m >= -w) & (m < 0)).sum()), int(((m >= 0) & (m < w)).sum()))
                  for w in widths}
    q = df.q_well
    out["Q"] = {w: (int(((q >= -w) & (q < 0)).sum()), int(((q >= 0) & (q < w)).sum()))
                for w in widths}
    return out


# ------------------------------------------------------------------------------ samples

LAG_X = ["log_asset_l1", "sh_loans_l1", "sh_dep_l1", "sh_re_l1", "sh_ci_l1", "sh_sc_l1",
         "sh_bro_l1", "sh_ncl_l1"]


def _frame(y: str, quarters=(4,)) -> pd.DataFrame:
    p = build()
    m = ((p.date >= PCA_START) & p.BKCLASS.isin(CLASSES) & (p.CBLRIND.fillna(0) != 1)
         & p.q_well.notna() & p.quarter.isin(quarters) & p[y].notna()
         & p[LAG_X].notna().all(axis=1))
    return p[m].reset_index(drop=True)


def _design(df: pd.DataFrame, quarters: bool = False) -> tuple:
    X: dict[str, pd.Series] = {"log_asset_l1": df.log_asset_l1}
    for c in LAG_X[1:]:
        X[c] = df[c].clip(0, 1)
    X["log1p_age"] = np.log1p(df.age.fillna(df.age.median()))
    X["age_missing"] = df.age.isna()
    for k in ("N", "SM", "SB", "SI", "SL"):                     # state nonmember base
        X[f"class_{k}"] = df.BKCLASS == k
    X["hc_multibank"] = df.HCTMULT == 1                          # one-bank HC base
    X["hc_none"] = df.HCTNONE == 1
    for v in sorted(df.SPECGRP.dropna().unique()):
        if v != 4:                                               # commercial lending base
            X[f"spec_{int(v)}"] = df.SPECGRP == v
    X["mutual"] = df.MUTUAL == 1
    X["subchapter_s"] = df.SUBCHAPS == 1
    X["metro"] = df.METRO == 1
    X["metro_missing"] = df.METRO.isna()
    X["community_bank"] = df.CB == 1
    st = df.STALP.astype(str)
    sv = st.value_counts()
    states = sorted(sv[sv >= 100].index)
    for s in states:
        if s != "TX":                                            # Texas base
            X[f"state_{s}"] = st == s
    X["state_other"] = ~st.isin(states)
    for yr in sorted(df.year.unique())[1:]:
        X[f"year_{yr}"] = df.year == yr
    if quarters:
        for q in (1, 2, 3):
            X[f"quarter_{q}"] = df.quarter == q
    Xd = pd.DataFrame(X).astype(float)
    Xd = Xd.loc[:, Xd.std() > 0]
    return Xd.to_numpy(float), list(Xd.columns)


def _extras(df: pd.DataFrame) -> dict:
    return {
        "cert": df.CERT.to_numpy(int), "date": df.date.dt.strftime("%Y-%m-%d").to_numpy(str),
        "year": df.year.to_numpy(int), "name": df.NAME.astype(str).to_numpy(str),
        "bkclass": df.BKCLASS.astype(str).to_numpy(str),
        "state": df.STALP.astype(str).to_numpy(str),
        "group": regime(df.date), "group_cutoffs": WELL,
        "total_rbc": df.tot.to_numpy(float), "tier1_rbc": df.t1.to_numpy(float),
        "cet1": df.cet1.to_numpy(float), "leverage": df.lev.to_numpy(float),
        "q_well": df.q_well.to_numpy(float), "q_adequate": df.q_adeq.to_numpy(float),
        "binding_ratio": df.bind_well.to_numpy(str),
        "category": df.category.map(CATEGORY).to_numpy(str),
        "q_well_lag1": df.q_well_l1.to_numpy(float), "q_well_lead4": df.q_well_f4.to_numpy(float),
        "y_dloan": df.y_dloan.to_numpy(float), "y_dloan_raw": df.y_dloan_raw.to_numpy(float),
        "y_dbrokered": df.y_dbro.to_numpy(float), "y_fail8": df.y_fail8.to_numpy(float),
        "y_assist8": df.y_assist8.to_numpy(float), "exit_4q": df.exit_4q.to_numpy(float),
        "acq_next4": df.acq_next4.to_numpy(float),
        "placebo_dloan": df.placebo_dloan.to_numpy(float),
        "fail_date": df.fail_date.dt.strftime("%Y-%m-%d").fillna("").to_numpy(str),
        "published_designation": "none: no bank-level PCA category is published; the QBP "
                                 "publishes aggregate counts only (see rule_check)",
        "regulator": df.REGAGNT.astype(str).to_numpy(str),
        "rule_text": "well capitalized = 1{every ratio >= its cutoff}; cutoffs by regime in "
                     "extras['group_cutoffs'] (12 CFR 325.103 to 2014, 324.403 from 2015)",
    }


_CITATION = ("FDIC BankFind Suite API (api.fdic.gov/banks): financials (call reports), "
             "failures and history, index of 2026-08; 12 CFR 325.103 (to 2014) and 12 CFR "
             "324.403 (2015 on); FDIC Quarterly Banking Profile graph books 2006-2019")
_Y_LABEL = {"y_dloan": "log(gross loans t+4q) - log(gross loans t), clipped to [-1, 1]",
            "y_fail8": "1{bank fails within the next 8 quarters}"}


def _single(y: str, quarters=(4,), adequate: bool = False) -> RDDSample:
    df = _frame(y, quarters)
    X, names = _design(df, quarters=len(quarters) > 1)
    col, lab = (("q_adeq", "adequately capitalized or better") if adequate else
                ("q_well", "well capitalized"))
    ex = _extras(df)
    ex.update({"cutoff": 0.0, "treated_side": f"{lab}: Q >= 0, every ratio at or above its "
                                              "cutoff; the untreated side is the either rule "
                                              "(some ratio below its cutoff)"})
    if adequate:
        ex["group_cutoffs"] = ADEQ
        ex["binding_ratio_well"] = ex["binding_ratio"]
        ex["binding_ratio"] = df.bind_adeq.to_numpy(str)
    tag = ("_adequate" if adequate else "") + ("_quarterly" if len(quarters) > 1 else "")
    return RDDSample(
        Q=df[col].clip(*Q_CAP).to_numpy(float), X=X, Y=df[y].to_numpy(float), threshold=0.0,
        name=f"fdic_capital{tag}_{y}", feature_names=names,
        description=(f"US insured banks, {'all quarters' if len(quarters) > 1 else 'Q4'} "
                     f"{df.year.min()}-{df.year.max()}. Q = min over the PCA capital ratios of "
                     f"(ratio - {'adequately' if adequate else 'well'}-capitalized cutoff), pp; "
                     f"treatment = 1{{Q >= 0}}, {lab}; Y = {_Y_LABEL[y]}."),
        citation=_CITATION, treatment_rule=well_rule, extras=ex,
    )


def _two_score(y: str) -> RDDSample:
    """(total RBC, leverage). Drops the bank-years that meet both cutoffs but fail on tier 1
    RBC or CET1 alone, so that D = 1{Q1 >= 10 and Q2 >= 5} equals well capitalized."""
    df = _frame(y)
    pre = df.date < BASEL3
    other_ok = (df.t1 >= np.where(pre, 6.0, 8.0)) & (pre | (df.cet1 >= 6.5))
    ok = other_ok | (df.tot < 10.0) | (df.lev < 5.0)
    dropped = int((~ok).sum())
    df = df[ok].reset_index(drop=True)
    X, names = _design(df)
    ex = _extras(df)
    ex.update({"cutoffs": (10.0, 5.0), "rule": "both",
               "dropped_other_ratio_alone_fails": dropped,
               "treated_side": "well capitalized: Q1 >= 10 and Q2 >= 5; the untreated side "
                               "is the either rule Q1 < 10 or Q2 < 5"})
    return RDDSample(
        Q=np.column_stack([df.tot.clip(10 + Q_CAP[0], 10 + Q_CAP[1]),
                           df.lev.clip(5 + Q_CAP[0], 5 + Q_CAP[1])]),
        X=X, Y=df[y].to_numpy(float),
        threshold=(10.0, 5.0), name=f"fdic_capital_two_score_{y}", feature_names=names,
        description=("US insured banks, Q4 (minus 3 bank-years where tier 1 RBC or CET1 "
                     "alone fails). Q = (total risk-based capital ratio %, tier 1 "
                     "leverage ratio %); treatment = 1{Q1 >= 10 and Q2 >= 5}, well "
                     f"capitalized; Y = {_Y_LABEL[y]}."),
        citation=_CITATION, treatment_rule=two_score_rule, extras=ex,
    )


def load() -> RDDSample:
    """Q4 bank-years: Q = binding margin to the well-capitalized cutoffs; D = 1{Q >= 0};
    Y = log loan growth over the next four quarters."""
    return _single("y_dloan")


def load_fail() -> RDDSample:
    """As load(), with Y = failure within the next 8 quarters (Q4 1992-2023)."""
    return _single("y_fail8")


def load_quarterly() -> RDDSample:
    """Every quarter; Q, D, Y as load(). Large (about 0.9 million rows)."""
    return _single("y_dloan", quarters=(1, 2, 3, 4))


def load_two_score() -> RDDSample:
    """Q = (total RBC ratio, leverage ratio); D = 1{Q1 >= 10 and Q2 >= 5}; Y = loan growth."""
    return _two_score("y_dloan")


def load_two_score_fail() -> RDDSample:
    return _two_score("y_fail8")


def load_adequate() -> RDDSample:
    """Q = binding margin to the adequately-capitalized cutoffs; D = 1{Q >= 0}, not
    undercapitalized; Y = loan growth."""
    return _single("y_dloan", adequate=True)


def load_adequate_fail() -> RDDSample:
    return _single("y_fail8", adequate=True)
