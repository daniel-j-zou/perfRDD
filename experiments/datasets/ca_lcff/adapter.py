"""California LCFF concentration grant: a continuous dose that starts at a 55% kink.

Per ADA, the Local Control Funding Formula pays a school district or charter school

    base_g x [1 + 0.20 x UPP + c_t x max(UPP - 0.55, 0)]        (grade span g)

where UPP is the unduplicated pupil percentage (low-income, English learner and foster
students as a share of enrollment, summed over the current and two prior Census Days)
and c_t is the concentration-grant rate: 0.50 through 2020-21 and 0.65 from 2021-22
(AB 130). The DOSE is the concentration grant, zero for UPP <= 0.55 and linear above it,
with a slope that differs by fiscal year (rate regime x base grant). A charter school's
UPP in the concentration term is capped at the UPP of its authorizing school district
(or of a district where it has a site), so a charter's schedule also depends on its cap.

``load_dose()`` returns an LEA x fiscal-year panel as an ``RDDSample`` (see README.md):

    Q          UPP (0-1) of fiscal year t
    threshold  0.55
    A          extras["A"]: formula concentration grant per ADA in dollars at the year's
               grades 4-6 base rate, c_t x r46_t x max(min(Q, cap) - 0.55, 0)
    group      extras["group"]: fiscal year (2021 = 2020-21); extras["regime"] = c_t
    Y          CAASPP percent meeting or exceeding the standard (all grades), ELA or
               math, in spring of fiscal year t+1 (``outcome_lead=1``)
    X          charter flag, log enrollment, race/ethnicity and disability shares,
               grade-span ADA shares, district type, virtual charter, location, lagged
               outcome, county and year indicators

Fiscal years are labelled by the calendar year in which they end (2021 = 2020-21).
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample
from experiments.datasets.ca_lcff import xlsx

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
PROC = HERE / "data" / "processed"
CACHE = PROC / "ca_lcff_panel.csv.gz"

THRESHOLD = 0.55
FISCAL_YEARS = (2020, 2021, 2022, 2023, 2024, 2025, 2026)      # 2020 = 2019-20
C_RATE = {2020: 0.50, 2021: 0.50, 2022: 0.65, 2023: 0.65, 2024: 0.65, 2025: 0.65, 2026: 0.65}
SUPP_RATE = 0.20
# Adjusted base grant per ADA by grade span (TK/K-3 incl. the 10.4% class-size add-on,
# 4-6, 7-8, 9-12 incl. the 2.6% add-on). Recovered from the LCFF Summary Data (base grant
# = rates x grade-span ADA for every LEA without a necessary-small-school allowance, to
# the dollar) and consistent with the statutory COLAs: 0% (2020-21), 5.07% (2021-22),
# 6.56% + 6.28% augmentation (2022-23), 8.22%, 1.07%, 2.30%. 2019-20 equals 2020-21.
BASE_RATES = {
    2020: (8503, 7818, 8050, 9572), 2021: (8503, 7818, 8050, 9572),
    2022: (8935, 8215, 8458, 10057), 2023: (10119, 9304, 9580, 11391),
    2024: (10951, 10069, 10367, 12327), 2025: (11068, 10177, 10478, 12460),
    2026: (11323, 10411, 10719, 12746),
}
SPANS = ("ada_k3", "ada_46", "ada_78", "ada_912")
CERT_ORDER = ("anr1", "an", "p2", "p1")      # latest certification first
CAASPP_YEARS = (2019, 2021, 2022, 2023, 2024, 2025, 2026)
LAG_YEARS = (2019, 2022, 2023, 2024, 2025, 2026)   # 2021 had partial, optional testing
RACE = {"hispanic": "H", "black": "B", "asian": "A", "filipino": "F", "amind": "I",
        "pacisl": "P", "multi": "T", "white": "W"}


# ----------------------------------------------------------------------------- schedule
def conc_rate(group) -> np.ndarray:
    return np.vectorize(lambda g: C_RATE[int(g)], otypes=[float])(group)


def base46(group) -> np.ndarray:
    return np.vectorize(lambda g: float(BASE_RATES[int(g)][1]), otypes=[float])(group)


def schedule(q, group, cap=None, unit: str = "dollars") -> np.ndarray:
    """Concentration-grant dose set by the formula.

    q      UPP on a 0-1 scale; group: fiscal year (2021 = 2020-21), scalar or array;
    cap    charter cap on the UPP (None or inf = no cap);
    unit   "dollars": per ADA at the year's grades 4-6 base rate (``extras["A"]``);
           "share": as a fraction of the base grant, c_t x max(min(q, cap) - 0.55, 0).
    """
    q = np.asarray(q, float)
    g = np.broadcast_to(np.asarray(group), q.shape)
    qc = q if cap is None else np.minimum(q, np.broadcast_to(np.asarray(cap, float), q.shape))
    share = conc_rate(g) * np.maximum(qc - THRESHOLD, 0.0)
    if unit == "share":
        return share
    if unit == "dollars":
        return share * base46(g)
    raise ValueError(unit)


def formula_grant(fy, upp, cap, ada_spans: np.ndarray, base_adj_fallback=None) -> np.ndarray:
    """LEA concentration grant in dollars as CDE computes it.

    sum over grade spans of round(c_t x rate_g,t x max(min(UPP, cap) - 0.55, 0) x ADA_g).
    Where grade-span ADA is missing (2019-20), c_t x base_adj x excess with base_adj
    given by ``base_adj_fallback`` (the adjusted base grant implied by the published
    supplemental grant, supp / (0.20 x UPP)).
    """
    fy = np.asarray(fy)
    ex = np.maximum(np.minimum(np.asarray(upp, float), np.asarray(cap, float)) - THRESHOLD, 0.0)
    c = conc_rate(fy)
    R = np.array([BASE_RATES[int(y)] for y in fy], float)
    A = np.asarray(ada_spans, float)
    by_span = np.round(c[:, None] * R * ex[:, None] * A).sum(1)
    if base_adj_fallback is not None:
        miss = np.isnan(A).any(1)
        by_span = np.where(miss, c * np.asarray(base_adj_fallback, float) * ex, by_span)
    return by_span


# ----------------------------------------------------------------------------- readers
def _need(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run experiments/datasets/ca_lcff/download.py "
                                "(see README).")
    return path


def _cert(sheet: str) -> Optional[str]:
    s = sheet.upper()
    if "R1" in s:
        return "anr1"
    if " AN" in s or "ANNUAL" in s:
        return "an"
    if "P-2" in s:
        return "p2"
    if "P-1" in s:
        return "p1"
    return None


def _col(h: str) -> Optional[str]:
    h = " ".join(str(h).split()).lower().rstrip("12").strip()
    rules = [("county code", "county"), ("district code", "district"), ("school code", "school"),
             ("local educational", "name"), ("charter number", "charter_num"),
             ("k-3", "ada_k3"), ("4 - 6", "ada_46"), ("4-6", "ada_46"), ("7 - 8", "ada_78"),
             ("7-8", "ada_78"), ("9 - 12", "ada_912"), ("9-12", "ada_912"),
             ("total funded ada", "ada"), ("unduplicated", "upp"), ("lcff base", "base"),
             ("supplemental", "supp"), ("concentration", "conc"),
             ("county operations", "cog"), ("total lcff entitlement", "lcff_total"),
             ("total local revenue", "local_rev")]
    for key, name in rules:
        if key in h:
            return name
    return None


def _code(v, width: int):
    """CDS code part as a zero-padded string (some sheets store codes as numbers)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, float):
        return str(int(v)).zfill(width)
    v = str(v).strip()
    return v.zfill(width) if v.isdigit() else v


def read_lcff_summary() -> pd.DataFrame:
    """Every certification of every LCFF Summary workbook (districts, COEs, charters)."""
    out = []
    for fy in FISCAL_YEARS:
        f = _need(RAW / "pa" / f"lcffsummary{fy - 2001:02d}{fy - 2000:02d}.xlsx")
        for sheet in xlsx.sheet_names(f):
            cert = _cert(sheet)
            if cert is None or "DESC" in sheet.upper():
                continue
            rows = xlsx.read_sheet(f, sheet)
            hi = next(i for i, r in enumerate(rows[:25]) if r and r[0] == "County Code")
            names = [_col(h) if h else None for h in rows[hi]]
            recs = []
            for r in rows[hi + 1:]:
                if not r or not isinstance(r[0], str) or not r[0].strip().isdigit():
                    continue
                recs.append({n: (r[j] if j < len(r) else None) for j, n in enumerate(names) if n})
            df = pd.DataFrame(recs)
            for c, w in (("county", 2), ("district", 5), ("school", 7), ("charter_num", 4)):
                df[c] = df[c].map(lambda v, w=w: _code(v, w))
            for c in ("ada", "upp", "base", "supp", "conc", "lcff_total", "local_rev") + SPANS:
                if c in df:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                else:
                    df[c] = np.nan
            df["is_charter"] = df.charter_num.notna() & ~df.charter_num.astype(str).isin(["N/A", ""])
            df["is_coe"] = (~df.is_charter) & df.district.str.startswith("10") & (df.school == "0000000")
            if df.upp.max() > 1.5:          # 2019-20 and 2020-21 P-1 report percent (COEs: 0-1)
                df.loc[~df.is_coe, "upp"] = df.loc[~df.is_coe, "upp"] / 100.0
            df["fy"], df["cert"], df["sheet"] = fy, cert, sheet
            out.append(df)
    d = pd.concat(out, ignore_index=True)
    d["cds"] = d.county + d.district + d.school
    return d


def _lcff_panel() -> pd.DataFrame:
    """Latest certification per LEA-year, districts and charters, with the charter cap."""
    allc = read_lcff_summary()
    rank = {c: k for k, c in enumerate(CERT_ORDER)}
    allc["rank"] = allc.cert.map(rank)
    best = allc.groupby("fy")["rank"].transform("min")
    d = allc[(allc["rank"] == best) & ~allc.is_coe].copy()
    # authorizer UPP for the charter cap: the authorizing district's P-2 UPP
    p2 = allc[(allc.cert == "p2") & ~allc.is_charter & ~allc.is_coe]
    p2 = p2.set_index(["fy", "county", "district"]).upp
    d["upp_auth"] = [p2.get(k, np.nan) for k in zip(d.fy, d.county, d.district)]
    d.loc[~d.is_charter, "upp_auth"] = np.nan
    # 2019-20 has no grade-span ADA: carry 2020-21's grade-span shares (ADA was held
    # harmless at 2019-20 P-2 levels in 2020-21) -- used for covariates only.
    nxt = d[d.fy == 2021].set_index("cds")
    m = d.fy == 2020
    tot = nxt[list(SPANS)].sum(1)
    for s in SPANS:
        d.loc[m, s + "_share"] = d.loc[m, "cds"].map(nxt[s] / tot).values
    for s in SPANS:
        d.loc[~m, s + "_share"] = d.loc[~m, s] / d.loc[~m, list(SPANS)].sum(1)
    d["lea_type"] = np.where(d.is_charter, "charter", "district")
    keep = ["fy", "cert", "cds", "county", "district", "school", "name", "charter_num", "lea_type",
            "ada", *SPANS, *(s + "_share" for s in SPANS), "upp", "upp_auth", "base", "supp",
            "conc", "lcff_total", "local_rev"]
    # Basic-aid ("community funded") districts: local property taxes cover the whole LCFF
    # entitlement, so a higher concentration grant does not raise their revenue.
    d["basic_aid"] = (d.lea_type == "district") & (d.local_rev >= d.lcff_total)
    keep.append("basic_aid")
    d = d[(d.ada > 0) & (d.base > 0) & d.upp.notna()]
    return d[keep].reset_index(drop=True)


def _add_formula(d: pd.DataFrame) -> pd.DataFrame:
    """Formula grant, the cap actually used, and the doses."""
    fy = d.fy.to_numpy()
    upp = d.upp.to_numpy(float)
    spans = d[list(SPANS)].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        badj_supp = np.where(upp > 0, d.supp / (SUPP_RATE * upp), np.nan)
    d["base_adj"] = np.where(np.isnan(spans).any(1), badj_supp,
                             (spans * np.array([BASE_RATES[int(y)] for y in fy], float)).sum(1))
    charter = (d.lea_type == "charter").to_numpy()
    auth = d.upp_auth.to_numpy(float)
    cap0 = np.where(charter & ~np.isnan(auth), auth, np.inf)
    d["cap_source"] = np.where(~charter, "none", np.where(np.isnan(auth), "unknown", "authorizer"))
    d["conc_formula"] = formula_grant(fy, upp, cap0, spans, badj_supp)
    tol = np.maximum(2.0, 1e-4 * d.conc.to_numpy())
    off = np.abs(d.conc - d.conc_formula).to_numpy() > tol
    # Charters whose grant the authorizer cap does not reproduce: the cap comes from a
    # district where the charter has a site, or (county / state authorized) is unknown.
    # Take the cap implied by the published grant.
    c = conc_rate(fy)
    with np.errstate(divide="ignore", invalid="ignore"):
        implied = THRESHOLD + d.conc.to_numpy() / (c * d.base_adj.to_numpy())
    fix = charter & off & (upp > THRESHOLD)
    cap = np.where(fix, np.where(d.conc.to_numpy() > 0, np.minimum(implied, upp), THRESHOLD), cap0)
    d.loc[fix, "cap_source"] = "published"
    d["cap"] = cap
    d["cap_binds"] = upp > np.minimum(cap, np.inf) + 1e-9
    d["conc_formula_used"] = formula_grant(fy, upp, cap, spans, badj_supp)
    d["regime"] = c
    d["A"] = schedule(upp, fy, cap, unit="dollars")
    d["A_share"] = schedule(upp, fy, cap, unit="share")
    d["A_lea"] = d.conc_formula_used / d.ada
    d["published_dose"] = d.conc / d.ada
    return d


def _upc() -> pd.DataFrame:
    out = []
    for fy in sorted(set(FISCAL_YEARS) | {2018, 2019}):
        f = _need(RAW / "upc" / f"cupc{fy - 2001:02d}{fy - 2000:02d}-k12.xlsx")
        rows = xlsx.read_sheet(f, "LEA-Level CALPADS UPC Data")
        hdr = [" ".join(str(h).split()).lower() if h else "" for h in rows[1]]

        def j(key):
            return next(k for k, h in enumerate(hdr) if h.startswith(key))
        ix = {"county": j("county code"), "district": j("district code"), "school": j("school code"),
              "dist_type": j("district type"), "enr": j("total enrollment"),
              "frpm": j("unduplicated frpm"), "el": j("english learner"),
              "upc": j("calpads unduplicated"), "irc": j("irc")}
        nslp = [k for k, h in enumerate(hdr) if h.startswith("nslp")]
        recs = []
        for r in rows[2:]:
            if not r or len(r) <= ix["upc"] or not isinstance(r[ix["county"]], str):
                continue
            rec = {k: r[v] for k, v in ix.items()}
            rec["nslp"] = r[nslp[0]] if nslp else None
            recs.append(rec)
        df = pd.DataFrame(recs)
        df["fy"] = fy
        out.append(df)
    d = pd.concat(out, ignore_index=True)
    d["cds"] = d.county + d.district + d.school
    for c in ("enr", "frpm", "el", "upc"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.drop_duplicates(["fy", "cds"])
    # three-year rolling UPP as the formula defines it, from the raw CALPADS counts
    w = d.pivot(index="cds", columns="fy", values=["upc", "enr"])
    roll = {}
    for fy in FISCAL_YEARS:
        yrs = [fy - 2, fy - 1, fy]
        num = w["upc"].reindex(columns=yrs).sum(1, min_count=3)
        den = w["enr"].reindex(columns=yrs).sum(1, min_count=3)
        roll[fy] = num / den
    r = pd.DataFrame(roll).stack().rename("upp_calpads").reset_index()
    r.columns = ["cds", "fy", "upp_calpads"]
    d = d.merge(r, on=["cds", "fy"], how="left")
    d["upc_share_cur"] = d.upc / d.enr
    d["cep"] = d.nslp.astype(str).str.contains("CEP", case=False)
    return d[["fy", "cds", "dist_type", "enr", "upc", "upc_share_cur", "upp_calpads", "cep", "irc"]]


def _enroll() -> pd.DataFrame:
    files = {2020: "cenroll1920.txt", 2021: "cenroll2021.txt", 2022: "cenroll2122.txt",
             2023: "cenroll2223.txt", 2024: "cdenroll2324-v2.txt", 2025: "cdenroll2425.txt",
             2026: "cdenroll2526.txt"}
    out = []
    for fy, name in files.items():
        df = pd.read_csv(_need(RAW / "enroll" / name), sep="\t", dtype=str,
                         encoding="latin-1", keep_default_na=False)
        cnt = "CumulativeEnrollment" if "CumulativeEnrollment" in df else "TOTAL_ENR"
        df["n"] = pd.to_numeric(df[cnt].replace("*", "0"), errors="coerce")
        df["Charter"] = df.Charter.str.strip().str.upper().str[0]          # A / N / Y
        cat = df.ReportingCategory.str.replace("RE_", "R", regex=False)
        cat = cat.replace({"SG_DS": "SWD", "SD": "SWD"}) if cnt == "TOTAL_ENR" else cat.replace({"SD": "SWD"})
        df["cat"] = cat
        dist = df[(df.AggregateLevel == "D") & (df.Charter == "N")]
        sch = df[(df.AggregateLevel == "S") & (df.Charter == "Y")]
        both = pd.concat([dist.assign(SchoolCode="0000000"), sch])
        both["cds"] = both.CountyCode + both.DistrictCode + both.SchoolCode
        wide = both.pivot_table(index="cds", columns="cat", values="n", aggfunc="sum")
        tot = wide["TA"]
        rec = pd.DataFrame({"fy": fy, "enr_total": tot}, index=wide.index)
        for k, code in RACE.items():      # a missing category row means no students
            rec[f"share_{k}"] = wide.get("R" + code, pd.Series(0.0, index=wide.index)).fillna(0) / tot
        rec["share_swd"] = wide.get("SWD", pd.Series(0.0, index=wide.index)).fillna(0) / tot
        rec["enroll_measure"] = "cumulative" if cnt != "TOTAL_ENR" else "census_day"
        out.append(rec.reset_index())
    return pd.concat(out, ignore_index=True)


def _read_caaspp(year: int) -> pd.DataFrame:
    folder = _need(RAW / "caaspp" / str(year))
    f = next(p for p in folder.iterdir() if "entities" not in p.name and p.suffix == ".txt")
    df = pd.read_csv(f, sep="," if year == 2019 else "^", dtype=str, encoding="latin-1")
    df.columns = [c.strip() for c in df.columns]
    ren = {"Subgroup ID": "group", "Student Group ID": "group", "Test Id": "test", "Test ID": "test",
           "Students Tested": "tested", "Total Students Tested": "tested",
           "Mean Scale Score": "mss", "Percentage Standard Met and Above": "pct_met",
           "Percentage Proficient and Above": "pct_met", "Grade": "grade",
           "County Code": "county", "District Code": "district", "School Code": "school",
           "Test Type": "ttype"}
    df = df.rename(columns=ren)
    df = df[(df.group == "1") & (df.ttype == "B")].copy()
    for c in ("tested", "mss", "pct_met", "grade"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["cds"] = df.county + df.district + df.school
    df["subject"] = df.test.map({"1": "ela", "2": "math"})
    # all-grades percent met / proficient
    allg = df[df.grade == 13][["cds", "subject", "pct_met", "tested"]]
    # tested-weighted mean over grades of (LEA - state) mean scale score
    st = df[df.cds == "00000000000000"].set_index(["subject", "grade"]).mss
    g = df[(df.grade < 13) & df.mss.notna() & (df.tested > 0)].copy()
    g["gap"] = g.mss - st.reindex(list(zip(g.subject, g.grade))).to_numpy()
    g["w"] = g.tested * g.gap
    gap = g.groupby(["cds", "subject"]).agg(w=("w", "sum"), t=("tested", "sum"))
    gap["mss_gap"] = gap.w / gap.t
    out = allg.merge(gap[["mss_gap"]].reset_index(), on=["cds", "subject"], how="outer")
    out["year"] = year
    return out


def _caaspp() -> pd.DataFrame:
    d = pd.concat([_read_caaspp(y) for y in CAASPP_YEARS], ignore_index=True)
    w = d.pivot_table(index=["cds", "year"], columns="subject", values=["pct_met", "mss_gap", "tested"])
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    return w.reset_index()


def _directory() -> pd.DataFrame:
    df = pd.read_csv(_need(RAW / "directory" / "pubschls.txt"), sep="\t", dtype=str,
                     encoding="utf-8", keep_default_na=False)
    df = df.rename(columns={"CDSCode": "cds"})
    for c in ("Latitude", "Longitude"):
        df[c] = pd.to_numeric(df[c].replace("No Data", ""), errors="coerce")
    return df[["cds", "County", "DOC", "DOCType", "FundingType", "Virtual", "SOC", "EdOpsCode",
               "Latitude", "Longitude", "StatusType"]].drop_duplicates("cds")


# ----------------------------------------------------------------------------- panel
def build_panel() -> pd.DataFrame:
    d = _add_formula(_lcff_panel())
    d = d.merge(_upc().drop(columns=[]), on=["fy", "cds"], how="left")
    d = d.merge(_enroll(), on=["fy", "cds"], how="left")
    d = d.merge(_directory(), on="cds", how="left")
    cs = _caaspp().set_index(["cds", "year"])
    cols = ["pct_met_ela", "pct_met_math", "mss_gap_ela", "mss_gap_math", "tested_ela", "tested_math"]

    def attach(year_of, suffix):
        idx = list(zip(d.cds, year_of))
        got = cs.reindex(idx)[cols].to_numpy()
        for k, c in enumerate(cols):
            d[f"{c}{suffix}"] = got[:, k]

    d["y_year"] = d.fy + 1
    attach(d.y_year, "_t1")                     # spring of fiscal year t+1
    attach(d.fy, "_t0")                         # spring of fiscal year t
    d["lag_year"] = [max([y for y in LAG_YEARS if y <= fy - 1]) for fy in d.fy]
    attach(d.lag_year, "_lag")
    return d


def panel(rebuild: bool = False) -> pd.DataFrame:
    if CACHE.exists() and not rebuild:
        return pd.read_csv(CACHE, low_memory=False,
                           dtype={"cds": str, "county": str, "district": str, "school": str,
                                  "charter_num": str, "DOC": str, "SOC": str, "irc": str,
                                  "EdOpsCode": str})
    d = build_panel()
    PROC.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=PROC, suffix=".csv.gz")
    os.close(fd)
    d.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, CACHE)
    return d


# ----------------------------------------------------------------------------- loaders
CONT_X = ["log_enroll", "share_hispanic", "share_black", "share_asian", "share_filipino",
          "share_amind", "share_pacisl", "share_multi", "share_swd", "ada_k3_share",
          "ada_46_share", "ada_78_share", "y_lag", "lat", "lon"]
Z_NAMES = ["charter", "log_enroll", "share_hispanic", "share_swd", "y_lag"]


def design(d: pd.DataFrame, subject: str = "ela", measure: str = "pct_met"):
    """Covariate matrix X (DataFrame) for panel rows d."""
    X = pd.DataFrame(index=d.index)
    X["charter"] = (d.lea_type == "charter").astype(float)
    enr = d.enr.where(d.enr > 0, d.enr_total)
    X["log_enroll"] = np.log(enr.where(enr > 0, d.ada))
    for k in RACE:
        if k != "white":
            X[f"share_{k}"] = d[f"share_{k}"]
    X["share_swd"] = d.share_swd
    for s in ("ada_k3", "ada_46", "ada_78"):
        X[f"{s}_share"] = d[f"{s}_share"]
    dt = d.DOCType.fillna("")
    X["elem_district"] = ((d.lea_type == "district") & dt.str.contains("Elementary")).astype(float)
    X["high_district"] = ((d.lea_type == "district") & dt.str.contains("High")).astype(float)
    X["virtual_charter"] = ((d.lea_type == "charter") & d.Virtual.isin(["F", "V"])).astype(float)
    X["county_authorized"] = ((d.lea_type == "charter") & d.district.str.startswith("10")).astype(float)
    ylag = d[f"{measure}_{subject}_lag"]
    X["y_lag_missing"] = ylag.isna().astype(float)
    X["y_lag"] = ylag.fillna(ylag.mean())
    X["lag_is_2019"] = (d.lag_year == 2019).astype(float)
    ll = d.Latitude.notna() & d.Longitude.notna()
    X["latlon_missing"] = (~ll).astype(float)
    X["lat"] = d.Latitude.where(ll, d.Latitude.mean())
    X["lon"] = d.Longitude.where(ll, d.Longitude.mean())
    for c in X.columns:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].mean())
    for y in sorted(d.fy.unique())[1:]:
        X[f"fy_{y}"] = (d.fy == y).astype(float)
    for c in sorted(d.county.unique())[1:]:
        X[f"county_{c}"] = (d.county == c).astype(float)
    X = X.loc[:, X.std() > 0]
    return X


def load_dose(subject: str = "ela", measure: str = "pct_met", lea: str = "all",
              years: Iterable[int] = (2021, 2022, 2023, 2024, 2025), outcome_lead: int = 1,
              rebuild: bool = False) -> RDDSample:
    """LEA x fiscal-year dose sample. lea: "all" (districts + charters), "district", "charter".

    outcome_lead=1: Y is measured in spring of fiscal year t+1 (2021 -> spring 2022);
    outcome_lead=0: spring of fiscal year t. 2020 (2019-20) is excluded by default because
    its t+1 outcome is the partial, optional spring-2021 test.
    """
    d = panel(rebuild)
    d = d[d.fy.isin(list(years))]
    if lea != "all":
        d = d[d.lea_type == lea]
    ycol = f"{measure}_{subject}_t{outcome_lead}"
    d = d[d[ycol].notna() & d.upp.notna()].reset_index(drop=True)
    X = design(d, subject, measure)
    ex = {
        "A": d.A.to_numpy(float),
        "a_ref": 0.0,
        "dose_levels": None,
        "schedule": schedule,
        "group": d.fy.to_numpy(),
        "regime": d.regime.to_numpy(float),
        "cap": d.cap.to_numpy(float),
        "cap_source": d.cap_source.to_numpy(),
        "cap_binds": d.cap_binds.to_numpy(bool),
        "Z_names": [z for z in Z_NAMES if z in X.columns],
        "lea_id": d.cds.to_numpy(),
        "lea_type": d.lea_type.to_numpy(),
        "fy": d.fy.to_numpy(),
        "ada": d.ada.to_numpy(float),
        "A_share": d.A_share.to_numpy(float),
        "A_lea": d.A_lea.to_numpy(float),
        "published_dose": d.published_dose.to_numpy(float),
        "conc_published": d.conc.to_numpy(float),
        "conc_formula": d.conc_formula_used.to_numpy(float),
        "supp_per_ada": (d.supp / d.ada).to_numpy(float),
        "basic_aid": d.basic_aid.astype(bool).to_numpy(),
        "upp_calpads": d.upp_calpads.to_numpy(float),
        "upc_share_cur": d.upc_share_cur.to_numpy(float),
        "Y_same": d[f"{measure}_{subject}_t0"].to_numpy(float),
        "tested": d[f"tested_{subject}_t{outcome_lead}"].to_numpy(float),
        "df": d,
    }
    return RDDSample(
        Q=d.upp.to_numpy(float), X=X.to_numpy(float), Y=d[ycol].to_numpy(float),
        threshold=THRESHOLD, name="ca_lcff", feature_names=list(X.columns),
        description=(f"California LCFF concentration grant, LEA x fiscal year {min(years)}-"
                     f"{max(years)} ({lea}); Q = UPP, A = c_t x r46_t x max(min(UPP, cap) - 0.55, 0) "
                     f"per ADA; Y = CAASPP {measure} {subject}, spring of t+{outcome_lead}"),
        citation=("California Department of Education, LCFF Summary Data 2019-20..2025-26; "
                  "CALPADS UPC; Census Day / cumulative enrollment; CAASPP research files (ETS)."),
        extras=ex,
    )


def load() -> RDDSample:
    """Registry entry point: the default dose sample (D = 1{UPP > 0.55})."""
    return load_dose()


# ----------------------------------------------------------------------------- checks
def rule_check(tol: float = 0.5) -> pd.DataFrame:
    """Published concentration grant vs the formula, by fiscal year and LEA type.

    formula: sum_g round(c_t x rate_g,t x max(min(UPP, cap) - 0.55, 0) x ADA_g); for
    charters cap = the authorizing district's P-2 UPP (if the authorizer is a school
    district), else no cap. 2019-20 has no grade-span ADA and uses base_adj = supp /
    (0.20 x UPP), so its check is "given the published supplemental grant".
    """
    d = panel()
    rows = []
    for (fy, t), g in d.groupby(["fy", "lea_type"]):
        e = (g.conc - g.conc_formula).abs()
        tol_g = tol if fy >= 2021 else np.maximum(2.0, 1e-4 * g.conc)
        rows.append({
            "fy": fy, "lea_type": t, "cert": g.cert.iloc[0], "n": len(g),
            "conc>0": float((g.conc > 0).mean()),
            "exact": float((e <= tol_g).mean()),
            "within_1pct": float((e <= 0.01 * g.conc.abs() + 1).mean()),
            "cap_authorizer": int((g.cap_source == "authorizer").sum()),
            "cap_published": int((g.cap_source == "published").sum()),
            "cap_unknown": int((g.cap_source == "unknown").sum()),
            "exact_with_cap_used": float(((g.conc - g.conc_formula_used).abs() <= np.maximum(tol_g, 2)).mean()),
            "upp_calpads_match": float((np.abs(g.upp - g.upp_calpads) < 5e-5).mean()),
            "upp_calpads_within_1pt": float((np.abs(g.upp - g.upp_calpads) < 0.01).mean()),
        })
    return pd.DataFrame(rows)


def readiness(s: Optional[RDDSample] = None, n_dec: int = 10) -> dict:
    """R^2 of Q on X, sd(eta), share at dose 0, eta quantiles by dose decile, bunching."""
    s = s or load_dose()
    Q, A = s.Q, s.extras["A"]
    Xc = np.column_stack([np.ones(len(Q)), s.X])
    beta, *_ = np.linalg.lstsq(Xc, Q, rcond=None)
    T = Xc @ beta
    eta = Q - T
    pos = A > 0
    cuts = np.quantile(A[pos], np.linspace(0, 1, n_dec + 1))
    by = [("dose 0", int((~pos).sum()), *np.quantile(eta[~pos], [0.1, 0.5, 0.9]))]
    for j in range(n_dec):
        m = pos & (A >= cuts[j]) & (A <= cuts[j + 1])
        by.append((f"d{j + 1}: {cuts[j]:.0f}-{cuts[j + 1]:.0f}", int(m.sum()),
                   *np.quantile(eta[m], [0.1, 0.5, 0.9])))
    return {"n": len(Q), "p": s.p, "R2": float(1 - eta.var() / Q.var()), "sd_Q": float(Q.std()),
            "sd_eta": float(eta.std()), "sd_T": float(T.std()), "share_dose0": float((~pos).mean()),
            "by_dose": by, "bunching": bunching(Q)}


def bunching(q: np.ndarray, width: float = 0.01,
             points=(0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75)) -> dict:
    """Count in [x, x + w) over count in [x - w, x) at the kink and at placebo points."""
    q = np.asarray(q, float)
    out = {}
    for x in points:
        hi = ((q >= x) & (q < x + width)).sum()
        lo = ((q >= x - width) & (q < x)).sum()
        out[x] = (int(lo), int(hi), float(hi / lo) if lo else np.nan)
    return out
