"""Massachusetts Chapter 70 school aid: the yearly aid increase per pupil as a dose.

Each fiscal year t the Chapter 70 formula (M.G.L. c. 70; DESE "Complete formula
spreadsheet", sheet ``Aid436``) gives every operating school district

    Chapter 70_t = base_t + max(FA_t - base_t, m_t x enr_t)                       (1)

    base_t   base aid: the district's Chapter 70 aid of t-1 (plus a rare adjustment)
    FA_t     foundation aid = foundation budget_t - required local contribution_t
    enr_t    foundation enrollment of year t
    m_t      the year's minimum aid increase per pupil (in the budget act):
             $25 (FY2016), $55 (FY2017), $30 (FY2018-20, FY2022), none (FY2021),
             $60 (FY2023-24), $104 (FY2025), $150 (FY2026), $160 (FY2027)

Dividing (1) by enrollment gives a one-dimensional schedule. The SCORE is the
foundation gap per pupil and the DOSE is the aid increase per pupil:

    Q = (FA_t - base_t) / enr_t                 ($ per pupil; < 0 for hold-harmless districts)
    A = (Chapter 70_t - base_t) / enr_t = max(Q, m_t)

so the districts whose foundation aid does not exceed last year's aid by m_t (the
"minimum aid" districts) all get the same dose m_t, and the others get A = Q. Each fiscal
year is a separate schedule (its floor m_t). Overlays that break (1) for a few districts
(hold-harmless to an earlier budget run, transitional low-income relief, the SOA minimum
aid adjustment, the non-operating reduction) are measured by ``rule_check()``.

``load_dose()`` returns a district x fiscal-year ``RDDSample`` with Y = next spring's MCAS
(grades 3-8, ELA by default) and X = enrollment, demographics, location, lagged outcome
and fiscal-year indicators. Fiscal years are labelled by the year in which they end
(2024 = July 2023 - June 2024; its MCAS is spring 2024 = school year "sy" 2024).
"""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample
from experiments.datasets.ma_chapter70 import xlsx

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
PROC = HERE / "data" / "processed"
CACHE = PROC / "ma_chapter70_panel.csv.gz"

FISCAL_YEARS = tuple(range(2016, 2028))
# Minimum aid increase per pupil, $ (DESE Chapter 70 pages for each FY; FY2021 had none).
MIN_AID = {2016: 25, 2017: 55, 2018: 30, 2019: 30, 2020: 30, 2021: 0, 2022: 30,
           2023: 60, 2024: 60, 2025: 104, 2026: 150, 2027: 160}
# Effort reduction (share of the gap above target removed): FY16 50%, FY17-18 85%, FY19+ 100%.
EFFORT_REDUCTION = {2016: .50, 2017: .85, 2018: .85, **{y: 1.0 for y in range(2019, 2028)}}

# Column positions in each year's Aid436 sheet (0-based) and a keyword that the combined
# header of that column must contain (checked on reading).
_C16 = dict(base=4, fb=5, rc=6, fa=7, tap=9, enr_prev=16, enr=17, minadd=31, nonop=33, c70=34)
_C17 = dict(_C16, c70=35, other=34)
_C20 = dict(base=6, fb=7, rc=8, fa=9, tap=11, enr_prev=18, enr=19, minadd=33, nonop=35, other=54, c70=37)
_C21 = dict(base=6, fb=7, rc=8, fa=9, tap=11, enr_prev=12, enr=13, minadd=23, nonop=36, other=37, c70=38)
_C22 = dict(base=7, fb=8, rc=9, fa=10, tap=12, enr_prev=13, enr=14, minadd=17, minadj=20, nonop=22,
            other=23, c70=24)
COLS = {2016: _C16, 2017: _C17, 2018: _C17, 2019: dict(_C17, other2=27), 2020: _C20, 2021: _C21,
        **{y: _C22 for y in range(2022, 2028)}}
MIN_COL = {y: (30 if y < 2020 else 32 if y == 2020 else 22 if y == 2021 else 16) for y in FISCAL_YEARS}
KEYS = dict(base="base", fb="foundation budget", rc="required contribution", fa="foundation aid",
            tap="target aid percentage", enr="foundation enrollment", enr_prev="foundation enrollment",
            minadd="to reach min", nonop="non-op reduction", c70="chapter 70")


def formula_file(fy: int) -> Path:
    if fy >= 2023:
        return RAW / f"ch70_fy{fy}" / f"chapter-{fy}.xlsm"
    name = f"chapter-{fy - 2000}.xlsm" if fy <= 2020 else f"chapter-{fy}.xlsm"
    return RAW / f"ch70_fy{fy}_wayback" / name


def _need(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run experiments/datasets/ma_chapter70/download.py "
                                "(see README).")
    return path


# ----------------------------------------------------------------------------- schedule
def min_aid(group) -> np.ndarray:
    return np.vectorize(lambda g: float(MIN_AID[int(g)]), otypes=[float])(group)


def schedule(q, group) -> np.ndarray:
    """Dose ($ per foundation pupil) set by the formula: max(q, m_t), group = fiscal year."""
    q = np.asarray(q, float)
    return np.maximum(q, min_aid(np.broadcast_to(np.asarray(group), q.shape)))


# ----------------------------------------------------------------------------- formula
def _header(rows, hi, j) -> str:
    parts = [str(rows[k][j]) for k in range(hi - 4, hi + 1) if j < len(rows[k]) and rows[k][j] not in (None, "")]
    return " ".join(" ".join(parts).split()).lower()


def read_aid(fy: int) -> pd.DataFrame:
    """One row per LEA (operating and non-operating) from the year's Aid436 sheet."""
    f = _need(formula_file(fy))
    sheet = next(s for s in xlsx.sheet_names(f) if s.lower() == "aid436")
    rows = xlsx.read_sheet(f, sheet)
    hi = next(i for i, r in enumerate(rows) if r and r[0] == "LEA")
    cols = COLS[fy]
    for k, j in cols.items():
        key = KEYS.get(k)
        if key and key not in _header(rows, hi, j):
            raise ValueError(f"FY{fy} Aid436 column {j} ({k}) header {_header(rows, hi, j)!r} lacks {key!r}")
    h = _header(rows, hi, MIN_COL[fy])
    if "min aid per pupil" not in h:
        raise ValueError(f"FY{fy}: no minimum aid header in column {MIN_COL[fy]}: {h!r}")
    m_sheet = float(re.search(r"([\d.]+) min aid per pupil", h).group(1))
    if m_sheet != MIN_AID[fy]:
        raise ValueError(f"FY{fy}: minimum aid in sheet {m_sheet} != {MIN_AID[fy]}")
    recs = []
    for r in rows[hi + 1:]:
        if not r or not isinstance(r[0], float) or r[0] >= 999 or len(r) < 2 or not isinstance(r[1], str):
            continue
        rec = {"lea": int(r[0]), "name": " ".join(str(r[1]).split()), "op": r[2]}
        for k, j in cols.items():
            rec[k] = r[j] if j < len(r) else None
        recs.append(rec)
    d = pd.DataFrame(recs)
    for c in d.columns.drop(["lea", "name"]):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["fy"], d["m"] = fy, float(MIN_AID[fy])
    return d


def formula_panel() -> pd.DataFrame:
    """All LEAs x FY2016-27 with the formula dose, the published dose and the rule check."""
    d = pd.concat([read_aid(fy) for fy in FISCAL_YEARS], ignore_index=True)
    for c in ("minadj", "other", "other2", "nonop", "minadd"):
        d[c] = d[c].fillna(0.0) if c in d else 0.0
    d["operating"] = (d.op == 1) & (d.enr > 0)
    d["gap"] = d.fa - d.base
    d["Q"] = d.gap / d.enr
    d["c70_formula"] = d.base + np.maximum(d.gap, d.m * d.enr)
    d["A"] = np.maximum(d.Q, d.m)                       # = schedule(Q, fy)
    d["A_published"] = (d.c70 - d.base) / d.enr
    d["err"] = d.c70 - d.c70_formula
    d["exact"] = d.err.abs() <= 1.0                     # within $1 (rounding)
    d["at_min"] = d.gap < d.m * d.enr
    d["level_exact"] = (d.c70 - d.fa).abs() <= 1.0      # aid = foundation budget - required contribution
    d["dist_code"] = d.lea.map(lambda x: f"{x:04d}0000")
    return d


# ----------------------------------------------------------------------------- covariates
def _enroll() -> pd.DataFrame:
    e = pd.read_csv(_need(RAW / "e2c_enroll" / "enrollment_district.csv"), dtype={"dist_code": str})
    e = e[e.dist_code.str.endswith("0000")]
    e["lowinc"] = e.li_pct.fillna(e.ecd_pct)            # low income (to 2014, 2022 on) or econ. disadv.
    e["lowinc_def"] = np.where(e.li_pct.notna(), "li", "ecd")
    keep = {"sy": "fy", "dist_code": "dist_code", "total_cnt": "enr_e2c", "lowinc": "share_lowinc",
            "el_pct": "share_el", "swd_pct": "share_swd", "hl_pct": "share_hispanic",
            "baa_pct": "share_black", "as_pct": "share_asian", "mnhl_pct": "share_multi",
            "wh_pct": "share_white", "hn_pct": "share_highneeds", "flne_pct": "share_flne",
            "lowinc_def": "lowinc_def"}
    return e[list(keep)].rename(columns=keep)


def _grades() -> pd.DataFrame:
    e = pd.read_csv(_need(RAW / "e2c_enroll" / "enrollment_district.csv"), dtype={"dist_code": str})
    e = e[e.dist_code.str.endswith("0000")]
    tot = e.total_cnt.where(e.total_cnt > 0)
    out = pd.DataFrame({"fy": e.sy, "dist_code": e.dist_code})
    out["share_k5"] = e[["pk_cnt", "k_cnt", "g1_cnt", "g2_cnt", "g3_cnt", "g4_cnt", "g5_cnt"]].sum(1) / tot
    out["share_912"] = e[["g9_cnt", "g10_cnt", "g11_cnt", "g12_cnt"]].sum(1) / tot
    out["has_38"] = (e[["g3_cnt", "g4_cnt", "g5_cnt", "g6_cnt", "g7_cnt", "g8_cnt"]].sum(1) > 0).astype(float)
    return out


def _mcas() -> pd.DataFrame:
    m = pd.read_csv(_need(RAW / "e2c_mcas" / "mcas_district.csv"), dtype={"dist_code": str})
    m = m[m.test_grade == "ALL (03-08)"]
    m["pct_me"] = 100 * m.m_plus_e_cnt / m.stu_cnt
    w = m.pivot_table(index=["sy", "dist_code"], columns="subject_code",
                      values=["pct_me", "avg_scaled_score", "stu_cnt"], aggfunc="first")
    w.columns = [f"{a}_{b.lower()}".replace("avg_scaled_score", "ss") for a, b in w.columns]
    return w.reset_index()


def _ppx() -> pd.DataFrame:
    p = pd.read_csv(_need(RAW / "e2c_ppx" / "expenditures_per_pupil.csv"), dtype={"dist_code": str})
    return p.rename(columns={"sy": "fy"})[["fy", "dist_code", "in_dist_exp_per_pupil", "tot_exp_per_pupil",
                                           "in_dist_fte"]]


def _nss() -> pd.DataFrame:
    n = pd.read_csv(_need(RAW / "e2c_nss" / "nss.csv"), dtype={"dist_code": str})
    return n.rename(columns={"sy": "fy"})[["fy", "dist_code", "actl_nss_amt", "req_nss_amt"]]


_TOWN_FIX = {"manchester": "manchester-by-the-sea"}


def _towns() -> pd.DataFrame:
    """Census 2023 Gazetteer county subdivisions of MA: town name, county FIPS, lat/lon."""
    g = pd.read_csv(_need(RAW / "census_gaz" / "2023_gaz_cousubs_25.txt"), sep="\t", dtype={"GEOID": str})
    g.columns = [c.strip() for c in g.columns]
    g = g[g.ALAND > 0]
    g["town"] = (g.NAME.str.replace(r"\s+(town|city|Town city)$", "", regex=True).str.lower().str.strip())
    g["county"] = g.GEOID.str[2:5]
    return g[["town", "county", "INTPTLAT", "INTPTLONG"]].rename(columns={"INTPTLAT": "lat", "INTPTLONG": "lon"})


def _members() -> pd.DataFrame:
    """Regional district -> member towns (LEA numbers), from each year's ``leas`` sheet
    (FY2018 on; the FY2016-17 workbooks do not list members)."""
    out = []
    for fy in FISCAL_YEARS:
        f = _need(formula_file(fy))
        rows = xlsx.read_sheet(f, next(s for s in xlsx.sheet_names(f) if s.lower() == "leas"))
        hi = next((i for i, r in enumerate(rows[:12]) if r and "fraclea" in [str(x) for x in r]), None)
        if hi is None:
            continue
        j = [str(x) for x in rows[hi]].index("fraclea")
        for r in rows[hi + 1:]:
            if len(r) > j + 3 and isinstance(r[j], float) and isinstance(r[j + 2], float):
                out.append({"fy": fy, "lea": int(r[j]), "member": int(r[j + 2])})
    return pd.DataFrame(out).drop_duplicates()


def _geo(d: pd.DataFrame) -> pd.DataFrame:
    """County (of the first-listed member town) and mean lat/lon of the member towns, per LEA.
    Static: a regional district's members are taken from the first year that lists them."""
    t = _towns()
    towns = d[d.lea <= 351].drop_duplicates("lea")
    tn = pd.DataFrame({"member": towns.lea.to_numpy(), "town": towns.name.str.lower().replace(_TOWN_FIX).to_numpy()})
    tn = tn.merge(t, on="town", how="left")
    mem = _members()
    first = mem.groupby("lea").fy.transform("min")
    mem = mem[mem.fy == first][["lea", "member"]]
    own = pd.DataFrame({"lea": towns.lea.to_numpy(), "member": towns.lea.to_numpy()})
    mem = pd.concat([mem[~mem.lea.isin(own.lea)], own], ignore_index=True)
    mem = mem.merge(tn[["member", "county", "lat", "lon"]], on="member", how="left")
    geo = mem.groupby("lea").agg(lat=("lat", "mean"), lon=("lon", "mean"), county=("county", "first"),
                                 n_members=("member", "nunique"))
    return geo.reset_index()


def keyfactors() -> pd.DataFrame:
    """DESE "Chapter 70 Trends" (keyfactors.xlsx, sheet dataAid): every LEA, FY2007-FY2026:
    foundation enrollment and budget, required contribution, Chapter 70 aid and its
    components. Used for the history before FY2016 (the lagged score of FY2016) and as a
    cross-check of the formula spreadsheets."""
    rows = xlsx.read_sheet(_need(RAW / "keyfactors" / "keyfactors.xlsx"), "dataAid")
    hi = next(i for i, r in enumerate(rows[:10]) if r and r[0] == "Org8Codefy")
    hdr = rows[hi][:20]
    k = pd.DataFrame([r[:20] for r in rows[hi + 1:] if r and r[0]], columns=hdr)
    for c in hdr[4:]:
        k[c] = pd.to_numeric(k[c], errors="coerce")
    k = k[k.LEANumCode != 999].rename(columns={"LEANumCode": "lea", "distfoundenro": "enr",
                                               "distfoundbudget": "fb", "distrlc": "rc", "c70aid": "c70"})
    k["lea"], k["fy"] = k.lea.astype(int), k.fy.astype(int)
    k = k.sort_values(["lea", "fy"])
    prev = k.groupby("lea").c70.shift(1).where(k.groupby("lea").fy.shift(1) == k.fy - 1)
    k["Q_kf"] = (k.fb - k.rc - prev) / k.enr
    return k[["lea", "fy", "enr", "fb", "rc", "c70", "Q_kf", "disttargaidpct"]]


# ----------------------------------------------------------------------------- panel
def build_panel() -> pd.DataFrame:
    d = formula_panel()
    d = d.merge(_geo(d), on="lea", how="left")
    d = d.merge(_enroll(), on=["fy", "dist_code"], how="left")
    d = d.merge(_grades(), on=["fy", "dist_code"], how="left")
    d = d.merge(_ppx(), on=["fy", "dist_code"], how="left")
    d = d.merge(_nss(), on=["fy", "dist_code"], how="left")
    mc = _mcas().set_index(["sy", "dist_code"])
    cols = list(mc.columns)

    def attach(year_of, suffix):
        got = mc.reindex(list(zip(year_of, d.dist_code)))[cols].to_numpy()
        for k, c in enumerate(cols):
            d[f"{c}{suffix}"] = got[:, k]

    attach(d.fy + 1, "_t1")                     # spring of fiscal year t+1
    attach(d.fy, "_t0")                         # spring of fiscal year t
    d["lag_year"] = np.where(d.fy - 1 == 2020, 2019, d.fy - 1)   # no MCAS in 2020
    attach(d.lag_year, "_lag")
    # lagged spending (FY t-1) and lagged score (Q of t-1, same LEA)
    d = d.sort_values(["lea", "fy"]).reset_index(drop=True)
    g = d.groupby("lea")
    d["ppx_lag"] = g.in_dist_exp_per_pupil.shift(1).where(g.fy.shift(1) == d.fy - 1)
    d["Q_lag"] = g.Q.shift(1).where(g.fy.shift(1) == d.fy - 1)
    kf = keyfactors().set_index(["lea", "fy"])
    q15 = kf.Q_kf.reindex(list(zip(d.lea, d.fy - 1))).to_numpy()   # FY2015 score for FY2016
    d["Q_lag"] = d.Q_lag.where(d.Q_lag.notna() | (d.fy != FISCAL_YEARS[0]), q15)
    d["base_pp"] = d.base / d.enr
    d["fb_pp"] = d.fb / d.enr
    d["rc_share"] = d.rc / d.fb
    d["enr_growth"] = np.log(d.enr.where(d.enr > 0) / d.enr_prev.where(d.enr_prev > 0))
    return d


def panel(rebuild: bool = False) -> pd.DataFrame:
    if CACHE.exists() and not rebuild:
        return pd.read_csv(CACHE, low_memory=False, dtype={"dist_code": str, "county": str})
    d = build_panel()
    PROC.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=PROC, suffix=".csv.gz")
    os.close(fd)
    d.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, CACHE)
    return pd.read_csv(CACHE, low_memory=False, dtype={"dist_code": str, "county": str})


# ----------------------------------------------------------------------------- loaders
OUTCOMES = {   # name -> (panel column of Y, panel column of its lag or None)
    "ela": ("pct_me_ela_t1", "pct_me_ela_lag"), "math": ("pct_me_math_t1", "pct_me_math_lag"),
    "ela_ss": ("ss_ela_t1", "ss_ela_lag"), "math_ss": ("ss_math_t1", "ss_math_lag"),
    "ela_t0": ("pct_me_ela_t0", "pct_me_ela_lag"), "math_t0": ("pct_me_math_t0", "pct_me_math_lag"),
    "ppx": ("in_dist_exp_per_pupil", "ppx_lag"),
}
CONT_X = ["log_enroll", "share_lowinc", "share_el", "share_swd", "share_hispanic", "share_black",
          "share_asian", "share_multi", "share_k5", "share_912", "lat", "lon", "y_lag", "Q_lag"]
Z_NAMES = ["log_enroll", "share_lowinc", "share_el", "y_lag"]


def design(d: pd.DataFrame, lag_col: Optional[str], score_lag: bool = True) -> pd.DataFrame:
    """Covariate matrix X for panel rows d (fiscal-year and county indicators included)."""
    X = pd.DataFrame(index=d.index)
    X["log_enroll"] = np.log(d.enr)
    for c in ("share_lowinc", "share_el", "share_swd", "share_hispanic", "share_black", "share_asian",
              "share_multi", "share_k5", "share_912"):
        X[c] = d[c]
    X["regional"] = (d.lea >= 400).astype(float)
    X["lat"], X["lon"] = d.lat, d.lon
    if lag_col is not None:
        ylag = d[lag_col]
        X["y_lag_missing"] = ylag.isna().astype(float)
        X["y_lag"] = ylag.fillna(ylag.mean())
    if score_lag:
        X["Q_lag"] = d.Q_lag
    X = X.replace([np.inf, -np.inf], np.nan)
    for c in X.columns:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].mean())
    for y in sorted(d.fy.unique())[1:]:
        X[f"fy_{y}"] = (d.fy == y).astype(float)
    for c in sorted(d.county.dropna().unique())[1:]:
        X[f"county_{c}"] = (d.county == c).astype(float)
    return X.loc[:, X.std() > 0]


def load_dose(outcome: str = "ela", years: Iterable[int] = (2016, 2017, 2018, 2020, 2021, 2022, 2023, 2024, 2025),
              on_rule: bool = True, score_lag: bool = True, rebuild: bool = False) -> RDDSample:
    """District x fiscal-year dose sample (operating districts with the outcome).

    outcome: "ela" / "math" (percent meeting or exceeding expectations, grades 3-8, spring of
             t+1), "ela_ss" / "math_ss" (average scaled score, t+1), "ela_t0" / "math_t0"
             (spring of t), "ppx" (in-district expenditure per pupil in FY t).
    years:   fiscal years (FY2019's t+1 test, spring 2020, was cancelled).
    on_rule: keep only district-years whose published aid equals formula (1) within $1.
    score_lag: put last year's score Q_{t-1} in X.
    """
    d = panel(rebuild)
    ycol, lcol = OUTCOMES[outcome]
    d = d[d.operating & d.fy.isin(list(years)) & d[ycol].notna()]
    if on_rule:
        d = d[d.exact]
    d = d.reset_index(drop=True)
    X = design(d, lcol, score_lag)
    ex = {
        "A": d.A.to_numpy(float),
        "a_ref": 30.0,
        "dose_levels": None,
        "schedule": schedule,
        "group": d.fy.to_numpy(),
        "m": d.m.to_numpy(float),
        "at_min": d.at_min.to_numpy(bool),
        "Z_names": [z for z in Z_NAMES if z in X.columns],
        "lea_id": d.lea.to_numpy(),
        "dist_code": d.dist_code.to_numpy(),
        "fy": d.fy.to_numpy(),
        "enr": d.enr.to_numpy(float),
        "A_published": d.A_published.to_numpy(float),
        "exact": d.exact.to_numpy(bool),
        "base_pp": d.base_pp.to_numpy(float),
        "fb_pp": d.fb_pp.to_numpy(float),
        "target_aid_pct": d.tap.to_numpy(float),
        "ppx": d.in_dist_exp_per_pupil.to_numpy(float),
        "df": d,
    }
    return RDDSample(
        Q=d.Q.to_numpy(float), X=X.to_numpy(float), Y=d[ycol].to_numpy(float), threshold=30.0,
        name="ma_chapter70", feature_names=list(X.columns),
        treatment_rule=lambda q: (np.asarray(q) > 30.0),
        description=(f"Massachusetts Chapter 70, district x FY {min(years)}-{max(years)}; Q = (foundation aid - "
                     f"base aid) / foundation enrollment; A = max(Q, m_t) $ per pupil; Y = {ycol}"),
        citation=("Massachusetts DESE, Chapter 70 formula spreadsheets FY2016-FY2027 (FY2016-22 via the "
                  "Internet Archive), Chapter 70 Trends; E2C hub MCAS, enrollment and expenditure files."),
        extras=ex,
    )


def load() -> RDDSample:
    """Registry entry point: the default dose sample (D = 1{Q > $30})."""
    return load_dose()


# ----------------------------------------------------------------------------- checks
def _explained(g: pd.DataFrame) -> pd.Series:
    """Misses of (1) accounted for by a published overlay column of the same sheet."""
    e = g.err
    inc = [g.other, g.other2, g.other + g.other2, g.minadj, g.minadj + g.other, -g.nonop]
    hit = np.zeros(len(g), bool)
    for v in inc:
        hit |= (e - v).abs().to_numpy() <= 1.0
    hit |= ((g.c70 - g.other).abs() <= 1.0).to_numpy()            # overlay given as a level (FY17, FY20)
    return pd.Series(hit, index=g.index)


def rule_check(tol: float = 1.0) -> pd.DataFrame:
    """Published Chapter 70 aid vs formula (1), operating districts, by fiscal year."""
    d = formula_panel()
    o = d[d.operating]
    rows = []
    for fy, g in o.groupby("fy"):
        miss = g.err.abs() > tol
        rows.append({
            "fy": fy, "m": MIN_AID[fy], "n": len(g),
            "exact": float((~miss).mean()),
            "within_1_per_pupil": float((g.err.abs() / g.enr <= 1.0).mean()),
            "misses": int(miss.sum()),
            "misses_explained_by_overlay": int((miss & _explained(g)).sum()),
            "share_at_min": float(g.at_min.mean()),
            "share_aid_eq_FB_minus_RC": float(g.level_exact.mean()),
        })
    return pd.DataFrame(rows)


def keyfactors_check() -> pd.DataFrame:
    """Formula spreadsheet vs Chapter 70 Trends: share of operating district-years that agree."""
    d = formula_panel()
    k = keyfactors()
    m = d[d.operating].merge(k, on=["lea", "fy"], suffixes=("", "_kf"))
    out = []
    for fy, g in m.groupby("fy"):
        out.append({"fy": fy, "n": len(g), **{c: float(((g[c] - g[c + "_kf"]).abs() <= 1).mean())
                                              for c in ("enr", "fb", "rc", "c70")}})
    return pd.DataFrame(out)


def readiness(s: Optional[RDDSample] = None, n_dec: int = 10) -> dict:
    """R^2 of Q on X, sd(eta), share at the floor, eta quantiles by dose decile."""
    s = s or load_dose()
    Q, A, m = s.Q, s.extras["A"], s.extras["m"]
    Xc = np.column_stack([np.ones(len(Q)), s.X])
    beta, *_ = np.linalg.lstsq(Xc, Q, rcond=None)
    T = Xc @ beta
    eta = Q - T
    pos = A > m + 1e-9
    cuts = np.quantile(A[pos], np.linspace(0, 1, n_dec + 1))
    by = [("at floor m_t", int((~pos).sum()), *np.quantile(eta[~pos], [0.1, 0.5, 0.9]))]
    for j in range(n_dec):
        sel = pos & (A >= cuts[j]) & (A <= cuts[j + 1])
        by.append((f"d{j + 1}: {cuts[j]:.0f}-{cuts[j + 1]:.0f}", int(sel.sum()), *np.quantile(eta[sel], [0.1, 0.5, 0.9])))
    return {"n": len(Q), "p": s.p, "districts": int(len(np.unique(s.extras["lea_id"]))),
            "R2": float(1 - eta.var() / Q.var()), "sd_Q": float(Q.std()), "sd_T": float(T.std()),
            "sd_eta": float(eta.std()), "share_at_floor": float((~pos).mean()),
            "share_at_30": float((np.abs(A - 30) < 1e-9).mean()), "by_dose": by}


def bunching(width: float = 25.0) -> pd.DataFrame:
    """Operating district-years with Q in [m - w, m) and [m, m + w), by year (kink at Q = m_t)."""
    d = formula_panel()
    o = d[d.operating]
    rows = []
    for fy, g in o.groupby("fy"):
        m = MIN_AID[fy]
        lo, hi = int(((g.Q >= m - width) & (g.Q < m)).sum()), int(((g.Q >= m) & (g.Q < m + width)).sum())
        rows.append({"fy": fy, "below": lo, "above": hi})
    return pd.DataFrame(rows)
