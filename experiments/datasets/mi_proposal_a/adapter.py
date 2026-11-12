"""Michigan school foundation allowance under Proposal A: a dose set by a published formula.

Each fiscal year t (2008 = school year 2007-08) every local school district gets a
per-pupil foundation allowance F_t from section 20 of the State School Aid Act (MCL
388.1620) as amended by that year's School Aid act. The formula takes the district's
prior-year allowance F_{t-1} (the score Q) and the year's parameters: the basic (from
2019-20 "target") allowance B_t, its increase x_t = B_t - B_{t-1}, the prior-year minimum,
and a slope constant c_t. In "2x" years a district at the minimum gets 2 x_t, a district
at or above the basic gets x_t, and districts between get a linear taper; in other years
everybody gets x_t, or the floor is raised (2012-13, 2021-22).

    Q          extras / sample.Q: F_{t-1}, the district's published prior-year allowance ($/pupil)
    A          extras["A"]: the year's increase beyond the basic increase,
               A = F_t - F_{t-1} - x_t  ($/pupil); reproduced by ``schedule``
    a_ref      0: the increase of a district at or above the basic, and of every district in
               years with a uniform increase
    group      fiscal year t (each year's parameters are a separate schedule)

Sources (README): MDE State Aid Financial Status Reports (final payment of each year,
mdoe.state.mi.us/samspublic), which list each district's "FY t Foundation", "FY 1995
Foundation", membership and taxable value; the School Aid public acts (legislature.mi.gov)
for the rule; Census F-33 for spending; NCES CCD for covariates.
"""
from __future__ import annotations

import math
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
PROC = HERE / "data" / "processed"

STATUS_FILES = {2000: "ALL0013", 2001: "all0114", 2002: "all0213", 2003: "all0313", 2004: "ALL0413",
                2005: "ALL0513", 2006: "ALL0613", 2007: "ALL0713", 2008: "ALL0813", 2009: "ALL0912",
                2010: "ALL1012", 2011: "ALL1112", 2012: "ALL1212", 2013: "ALL1312", 2014: "ALL1412",
                2015: "ALL1512", 2016: "ALL1611", 2017: "ALL1711", 2018: "ALL1811", 2019: "ALL1911",
                2020: "ALL2011", 2021: "ALL2111", 2022: "ALL2211", 2023: "ALL2311", 2024: "ALL2411",
                2025: "ALL2511", 2026: "ALL2611"}

# ---------------------------------------------------------------------------------------------
# 1. MDE State Aid Financial Status Reports (final payment of the year): one record per district
# ---------------------------------------------------------------------------------------------
_NUM = r"\(?-?[\d,]+\.\d{2}\)?"
_HDR = re.compile(r"District:\s+(\w{5})\s+ISD:\s*(\w+)")
_FOOT = re.compile(r"^\s*District:\s+(\w{5})\s+(\S.*\S)\s*$")
_PAT = {
    "fdn": re.compile(r"FY (\d{4}) Foundation\s+(" + _NUM + ")"),
    "memb": re.compile(r"State Aid Membership\s+(" + _NUM + ")"),
    "hstv": re.compile(r"(?<!Non-)Homestead Tax Value\s+(" + _NUM + ")"),
    "nhtv": re.compile(r"Non-Homestead Tax Value\s+(" + _NUM + ")"),
    "nhtv_pp": re.compile(r"(?:NHS T\.V\. Per Membership|NHS Tax Value Per Pupil)\s+(" + _NUM + ")"),
    "nonpre_tv": re.compile(r"Non-Pre TV\s+([\d,]+)\s"),
    "loc_rev_pp": re.compile(r"Local Revenue Per Membership\s+(" + _NUM + ")"),
    "eq22c": re.compile(r"^\s*22c\s+(?:FOUNDATION\s+)?EQUITY PAYMENT\s+(" + _NUM + ")"),
    "hh20m_pp": re.compile(r"20j/m\s+HOLD HARMLESS PAYMENT 20m \$([\d,]+\.\d{2}) PP"),
}


def _num(s: str) -> float:
    neg = s.startswith("(") or s.startswith("-")
    v = float(s.strip("()-").replace(",", ""))
    return -v if neg else v


def status_text(fy: int) -> Path:
    """pdftotext -layout copy of the year's status report (cached in data/processed/status_txt)."""
    name = STATUS_FILES[fy]
    out = PROC / "status_txt" / f"{name}.txt"
    if not out.exists():
        pdf = RAW / "sams_status" / f"{name}.pdf"
        if not pdf.exists():
            raise FileNotFoundError(f"{pdf} missing: run download.py --only status")
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["pdftotext", "-layout", str(pdf), str(out)], check=True)
    return out


def read_status(fy: int) -> pd.DataFrame:
    """One row per district code in the fiscal year's final State Aid status report."""
    rows, cur, rec = [], None, {}
    with open(status_text(fy), errors="replace") as fh:
        for line in fh:
            m = _HDR.search(line)
            if m and m.group(1) != cur:
                cur = m.group(1)
                rec = {"fy": fy, "code": cur, "isd": m.group(2)}
                rows.append(rec)
            if cur is None:
                continue
            m = _FOOT.match(line)
            if m and m.group(1) == cur and "ISD:" not in line and "name" not in rec:
                rec["name"] = m.group(2)
            for k, p in _PAT.items():
                m = p.search(line)
                if not m:
                    continue
                if k == "fdn":
                    y = int(m.group(1))
                    key = "F95" if y == 1995 else ("F" if y == fy else f"F{y}")
                    rec.setdefault(key, _num(m.group(2)))
                elif k == "nonpre_tv":
                    rec.setdefault(k, float(m.group(1).replace(",", "")))
                elif k not in rec:
                    rec[k] = _num(m.group(1))
    d = pd.DataFrame(rows)
    for c in ["F", "F95", "memb", "hstv", "nhtv", "nhtv_pp", "nonpre_tv", "loc_rev_pp", "eq22c", "hh20m_pp"]:
        if c not in d:
            d[c] = np.nan
    return d


def status_panel(rebuild: bool = False) -> pd.DataFrame:
    """District x fiscal year 2000-2026 from the status reports (cache data/processed/status_panel.csv.gz)."""
    path = PROC / "status_panel.csv.gz"
    if path.exists() and not rebuild:
        return pd.read_csv(path, dtype={"code": str, "isd": str})
    d = pd.concat([read_status(fy) for fy in sorted(STATUS_FILES)], ignore_index=True)
    d = d[d.F.notna()].copy()
    last3 = d.code.str[2:].astype(int)
    # local districts have codes CC000-CC599; public school academies (charters) CC600-CC999
    d["lea"] = (last3 < 600).astype(int)
    d["eq22c_pp"] = d.eq22c / d.memb
    tmp = path.with_suffix(".tmp")
    d.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, path)
    return d


# ---------------------------------------------------------------------------------------------
# 2. The rule (section 20 of the State School Aid Act, as amended by each year's School Aid act)
# ---------------------------------------------------------------------------------------------
# x: the year's increase in the basic (target) allowance as applied to districts at or above it
# (FY 2003: $200 under 2002 PA 521 although the basic rose by $400; FY 2008: $48, the increase
# of the 1994-95 hold-harmless districts, since the basic itself jumped to the state maximum).
# kind "u": uniform increase x; "2x": 2x formula
#     F_t = min(B, F' + 2x - (x - c)(F' - M)/(B - M))  for M <= F' < Bp,
#     F_t = B  for F' = Bp,  F_t = F' + x  for F' > Bp;
# "floor": F_t = max(F', floor); "catch": F_t = B for F' <= Bp; "plus": F_t = min(B, F' + x) for F' < Bp.
# M = the prior-year minimum used by the act ("considered to be" amounts), B = this year's basic,
# Bp = last year's basic. ACT = the public act that sets section 20 for the fiscal year.
RULE: Dict[int, dict] = {
    2001: dict(kind="u", x=300, act="2000 PA 297"),
    2002: dict(kind="u", x=300, act="2001 PA 121", eq=(6500.0, np.inf)),
    2003: dict(kind="u", x=200, act="2002 PA 521"),
    2004: dict(kind="u", x=0, act="2003 PA 158"),
    2005: dict(kind="u", x=0, act="2004 PA 351"),
    2006: dict(kind="u", x=175, act="2005 PA 155"),
    2007: dict(kind="u", x=210, act="2006 PA 342", eq=(7360.0, 23.0)),
    2008: dict(kind="2x", x=48, two=96, c=20, M=7108, B=8433, Bp=8385, den=1325, rnd="nearest",
               act="2007 PA 137"),
    2009: dict(kind="2x", x=56, c=20, M=7204, B=8489, Bp=8433, act="2008 PA 268"),
    2010: dict(kind="u", x=0, act="2009 PA 121"),
    2011: dict(kind="u", x=0, act="2010 PA 110"),
    2012: dict(kind="u", x=-470, act="2011 PA 62"),
    2013: dict(kind="floor", x=0, floor=6966, act="2012 PA 201"),
    2014: dict(kind="2x", x=30, c=10, M=6966, B=8049, Bp=8019, floor=6966, act="2013 PA 60",
               eq=(7076.0, 50.0)),
    2015: dict(kind="plus", x=50, B=8099, Bp=8049, act="2014 PA 196", eq=(7251.0, 125.0)),
    2016: dict(kind="2x", x=70, c=23, M=7251, B=8169, Bp=8099, act="2015 PA 85"),
    2017: dict(kind="2x", x=60, c=20, M=7391, B=8229, Bp=8169, act="2016 PA 249"),
    2018: dict(kind="2x", x=60, c=20, M=7511, B=8289, Bp=8229, act="2017 PA 108"),
    2019: dict(kind="2x", x=120, c=40, M=7631, B=8409, Bp=8289, act="2018 PA 265"),
    2020: dict(kind="2x", x=120, c=40, M=7871, B=8529, Bp=8409, act="2019 PA 58"),
    2021: dict(kind="u", x=0, act="2020 PA 165"),
    2022: dict(kind="catch", x=171, B=8700, Bp=8529, act="2021 PA 48"),
    2023: dict(kind="u", x=450, act="2022 PA 144"),
    2024: dict(kind="u", x=458, act="2023 PA 103"),
    2025: dict(kind="u", x=0, act="2024 PA 120"),
    2026: dict(kind="u", x=442, act="2025 PA 15"),
}
YEARS_2X = (2008, 2009, 2014, 2016, 2017, 2018, 2019, 2020)


def _round(v, how: str = "up"):
    v = np.round(np.asarray(v, float), 6)
    if how == "nearest":
        return np.floor(v + 0.5)
    return np.ceil(v)


def statutory(q, fy: int) -> np.ndarray:
    """F_t from F' = q (the prior-year allowance including any per-pupil equity payment of that
    year, which the act adds back) for a district that is not hold-harmless."""
    p = RULE[int(fy)]
    q = np.asarray(q, float)
    k, x = p["kind"], p["x"]
    if k == "u":
        f = q + x
    elif k == "floor":
        f = np.maximum(q, p["floor"])
    elif k == "2x":
        q1 = np.maximum(q, p.get("floor", -np.inf))
        two, den = p.get("two", 2 * x), p.get("den", p["B"] - p["M"])
        lin = np.minimum(_round(q1 + two - (x - p["c"]) * (q1 - p["M"]) / den, p.get("rnd", "up")), p["B"])
        f = np.where(q < p["Bp"], lin, np.where(q == p["Bp"], p["B"], q + x))
    elif k == "plus":
        f = np.where(q < p["Bp"], np.minimum(q + x, p["B"]), np.where(q == p["Bp"], p["B"], q + x))
    elif k == "catch":
        f = np.where(q <= p["Bp"], p["B"], q + x)
    else:
        raise ValueError(k)
    return _round(f, p.get("rnd", "up"))


def equity(f, fy: int) -> np.ndarray:
    """Section 22c equity payment per pupil of year fy given that year's statutory allowance f."""
    p = RULE[int(fy)]
    f = np.asarray(f, float)
    if "eq" not in p:
        return np.zeros_like(f)
    level, cap = p["eq"]
    return np.clip(level - f, 0.0, cap)


def schedule(q, group, unit: str = "extra") -> np.ndarray:
    """The dose by the formula for non-hold-harmless districts.

    q      score: prior-year effective allowance F_{t-1} + eq_{t-1} ($/pupil)
    group  fiscal year t (scalar or array)
    unit   "extra": Fe_t - q - x_t (the dose A; 0 = the basic increase);
           "increase": Fe_t - q;  "level": Fe_t
    """
    q = np.atleast_1d(np.asarray(q, float))
    g = np.broadcast_to(np.asarray(group), q.shape)
    out = np.empty_like(q)
    for fy in np.unique(g):
        m = g == fy
        f = statutory(q[m], int(fy))
        fe = f + equity(f, int(fy))
        out[m] = fe - (q[m] + RULE[int(fy)]["x"] if unit == "extra" else q[m] if unit == "increase" else 0.0)
    return out

# Hold-harmless districts (1994-95 allowance > $6,500, MCL 388.1620(3)(c)): F_{t-1} + min(x_t, cpi_t
# F_{t-1}), rounded up; cpi_t = the U.S. CPI change "in the calendar year ending in the immediately
# preceding fiscal year" (May revenue estimating conference). The rates below are where the
# published allowances put them (feasible intervals in the README); they agree with the BLS CPI-U
# calendar-year changes of 1999, 2000, 2015, 2020, 2021 and 2024. In other years x_t binds.
CPI = {2001: 0.022, 2002: 0.034, 2017: 0.001, 2022: 0.0123, 2023: 0.047, 2026: 0.0295}


def statutory_hh(q, fy: int) -> np.ndarray:
    """F_t of a hold-harmless district from F_{t-1} = q, without the section 20m add-backs
    (FY 2018, 2019, 2023, 2024), which are not reproduced."""
    p = RULE[int(fy)]
    q = np.asarray(q, float)
    x = p["x"]
    if p["kind"] == "floor":
        return q
    inc = np.minimum(x, CPI.get(int(fy), np.inf) * q) if x > 0 else np.full_like(q, x)
    return _round(q + inc, p.get("rnd", "up"))


# ---------------------------------------------------------------------------------------------
# 3. Covariates and outcomes: Census F-33, SAIPE, NCES CCD / EDGE
# ---------------------------------------------------------------------------------------------
def ccd_map() -> pd.DataFrame:
    """Michigan state district code (5 digits) -> NCES LEAID, CCD LEA directory 2015-16."""
    import io
    import zipfile
    with zipfile.ZipFile(RAW / "ccd" / "ccd_lea_029_1516_w_1a_011717_csv.zip") as z:
        d = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), dtype=str, encoding="latin-1")
    d = d[d.STABR == "MI"]
    return d[["ST_LEAID", "LEAID", "LEA_NAME", "LEA_TYPE"]].rename(columns={"ST_LEAID": "code", "LEAID": "leaid"})


def ccd_race() -> pd.DataFrame:
    """Shares of fall-2015 membership by race / ethnicity (CCD 052, 2015-16), by LEAID."""
    import io
    import zipfile
    with zipfile.ZipFile(RAW / "ccd" / "ccd_lea_052_1516_w_1a_011717_csv.zip") as z:
        d = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), dtype={"LEAID": str}, encoding="latin-1",
                        usecols=["STABR", "LEAID", "MEMBER", "AM", "AS", "HI", "BL", "WH", "HP", "TR"])
    d = d[d.STABR == "MI"].copy()
    num = d[["MEMBER", "AM", "AS", "HI", "BL", "WH", "HP", "TR"]].apply(pd.to_numeric, errors="coerce")
    m = num.MEMBER.where(num.MEMBER > 0)
    return pd.DataFrame({"leaid": d.LEAID.str.zfill(7), "share_black": num.BL / m, "share_hisp": num.HI / m,
                         "share_asian": (num.AS + num.HP) / m, "share_other": (num.AM + num.TR) / m})


def edge_locale() -> pd.DataFrame:
    """NCES locale (11-43), county FIPS and latitude / longitude by LEAID, EDGE 2016-17."""
    import io
    import zipfile
    member = "EDGE_GEOCODE_PUBLICLEA_1617/EDGE_GEOCODE_PUBLICLEA_1617.TXT"
    with zipfile.ZipFile(RAW / "ccd" / "EDGE_GEOCODE_PUBLICLEA_1617.zip") as z:
        d = pd.read_csv(io.BytesIO(z.read(member)), sep="|", header=None, dtype=str, encoding="latin-1")
    d = d[d[5] == "MI"]
    return pd.DataFrame({"leaid": d[0].str.zfill(7), "county": d[8], "lat": pd.to_numeric(d[10]),
                         "lon": pd.to_numeric(d[11]), "locale": d[23].str.strip()})


def saipe() -> pd.DataFrame:
    """SAIPE school-district estimates, income years 1999-2024: population 5-17 and in poverty."""
    out = []
    for y in range(1999, 2025):
        ext = "dat" if y <= 2002 else "txt"
        p = RAW / "saipe" / f"sd{y % 100:02d}-mi.{ext}"
        if not p.exists():
            continue
        for line in open(p, encoding="latin-1"):
            tok = line.split()
            if len(tok) < 7 or tok[0] != "26":
                continue
            # last two tokens: file name and date; before them total pop, pop 5-17, poor 5-17
            nums = [t for t in tok[-5:-2]]
            try:
                pop, p517, v517 = (float(t) for t in nums)
            except ValueError:
                continue
            out.append((y, "26" + tok[1], pop, p517, v517))
    return pd.DataFrame(out, columns=["inc_year", "leaid", "pop", "pop517", "pov517"])


def f33() -> pd.DataFrame:
    """Census F-33 fiscal years 2000-2024, Michigan rows: enrollment, current spending, revenue."""
    out = []
    for y in range(2000, 2025):
        p = RAW / "f33" / f"elsec{y % 100:02d}_mi.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p, dtype={"NCESID": str}, index_col=False, low_memory=False)
        g = lambda c: pd.to_numeric(d[c], errors="coerce") if c in d else np.nan
        out.append(pd.DataFrame({
            "leaid": d.NCESID.str.strip().str.zfill(7), "fy": y, "enr": g("V33"),
            "cur": g("TCURELSC"), "inst": g("TCURINST"), "totrev": g("TOTALREV"), "fedrev": g("TFEDREV"),
            "strev": g("TSTREV"), "locrev": g("TLOCREV"), "title1": g("C14"), "capout": g("TCAPOUT"),
            "salwage": g("Z33"), "schlev": d.get("SCHLEV")}))
    d = pd.concat(out, ignore_index=True)
    d = d[d.enr > 0].drop_duplicates(["leaid", "fy"])
    for c in ("cur", "inst", "totrev", "fedrev", "strev", "locrev", "title1", "capout", "salwage"):
        d[c + "_pp"] = 1000.0 * d[c] / d.enr          # F-33 amounts are in $1,000
    return d


# ---------------------------------------------------------------------------------------------
# 4. District x fiscal-year panel
# ---------------------------------------------------------------------------------------------
LOCALE_GROUP = {"1": "city", "2": "suburb", "3": "town", "4": "rural"}


def build_panel() -> pd.DataFrame:
    s = status_panel()
    s = s[s.lea == 1].copy()
    has_eq = s.fy.map(lambda t: "eq" in RULE.get(int(t), {})).to_numpy(bool)
    s["eq"] = np.where(has_eq, s.eq22c_pp.round(0).fillna(0.0), 0.0)
    s["Fe"] = s["F"] + s["eq"]
    # local revenue per membership pupil (18 mills on non-homestead value): published from FY 2009,
    # 0.018 x non-homestead taxable value per membership before
    s["lr_pp"] = s.loc_rev_pp.where(s.loc_rev_pp.notna(), 0.018 * s.nhtv_pp)
    s = s.sort_values(["code", "fy"]).reset_index(drop=True)
    g = s.groupby("code")
    consecutive = g.fy.shift(1) == s.fy - 1
    for c in ("F", "Fe", "memb", "lr_pp"):
        s[c + "_lag"] = g[c].shift(1).where(consecutive)
    s = s[(s.fy >= 2001) & s.Fe_lag.notna()].reset_index(drop=True)
    s["Q"] = s.Fe_lag
    s["x"] = s.fy.map(lambda t: RULE[int(t)]["x"]).astype(float)
    s["hh"] = (s.F95 > 6500).astype(int)
    s["A_pub"] = s.Fe - s.Q - s.x
    # the formula: schedule() for non-hold-harmless districts, statutory_hh() (no 20m add-backs) for
    # hold-harmless ones; the formula's statutory allowance and equity payment are kept too
    fy, q = s.fy.to_numpy(), s.Q.to_numpy()
    f_nh, f_hh, eq_f = np.empty(len(s)), np.empty(len(s)), np.empty(len(s))
    for t in np.unique(fy):
        m = fy == t
        f_nh[m] = statutory(q[m], int(t))
        eq_f[m] = equity(f_nh[m], int(t))
        f_hh[m] = statutory_hh(q[m], int(t))
    hh = s.hh.to_numpy() == 1
    s["F_formula"] = np.where(hh, f_hh, f_nh)
    s["eq_formula"] = np.where(hh, 0.0, eq_f)
    s["A_formula"] = s.F_formula + s.eq_formula - s.Q - s.x
    # published allowance and equity payment both equal the formula (equity: within $1, since the
    # per-pupil payment is recovered as amount / membership)
    s["rule_ok"] = (((s.F - s.F_formula).abs() < 0.5) & ((s["eq"] - s.eq_formula).abs() < 1.01)).astype(int)
    s["kind"] = s.fy.map(lambda t: RULE[int(t)]["kind"])
    # links
    s = s.merge(ccd_map()[["code", "leaid"]], on="code", how="left")
    s = s.merge(ccd_race(), on="leaid", how="left").merge(edge_locale(), on="leaid", how="left")
    sp = saipe()
    sp["pov_rate"] = sp.pov517 / sp.pop517.where(sp.pop517 > 0)
    sp["fy"] = sp.inc_year + 2                          # income year t-2 for fiscal year t
    s = s.merge(sp[["leaid", "fy", "pov_rate", "pop517"]], on=["leaid", "fy"], how="left")
    f = f33()
    keep = ["cur_pp", "inst_pp", "totrev_pp", "fedrev_pp", "strev_pp", "locrev_pp", "title1_pp", "capout_pp", "enr"]
    for lag, suf in ((0, ""), (1, "_lag"), (-1, "_lead"), (2, "_lag2")):
        ff = f[["leaid", "fy"] + keep].copy()
        ff["fy"] = ff.fy + lag
        s = s.merge(ff.rename(columns={c: c + "_f33" + suf for c in keep}), on=["leaid", "fy"], how="left")
    s["dcur"] = s.cur_pp_f33 - s.cur_pp_f33_lag
    s["dinst"] = s.inst_pp_f33 - s.inst_pp_f33_lag
    s["dcur_lead"] = s.cur_pp_f33_lead - s.cur_pp_f33_lag
    s["dstrev"] = s.strev_pp_f33 - s.strev_pp_f33_lag
    s["dtotrev"] = s.totrev_pp_f33 - s.totrev_pp_f33_lag
    return s.reset_index(drop=True)


def panel(rebuild: bool = False) -> pd.DataFrame:
    path = PROC / "mi_panel.csv.gz"
    if path.exists() and not rebuild:
        return pd.read_csv(path, dtype={"code": str, "isd": str, "leaid": str, "county": str, "locale": str})
    d = build_panel()
    tmp = path.with_suffix(".tmp")
    d.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, path)
    return d


# ---------------------------------------------------------------------------------------------
# 5. Dose design
# ---------------------------------------------------------------------------------------------
THRESHOLD = 0.0          # the prior-year minimum (score "gap" = 0)
CONT = ["log_memb", "log_lr", "pov_rate", "fed_share", "title1_pp", "share_black", "share_hisp",
        "share_asian", "share_other", "lat", "lon"]
Z_NAMES = ["log_memb", "pov_rate", "log_lr"]
OUTCOMES = {
    "dcur": "change in current spending per pupil, F-33 FY t minus FY t-1 ($)",
    "dinst": "change in instructional current spending per pupil, t minus t-1 ($)",
    "dcur_lead": "change in current spending per pupil, FY t+1 minus FY t-1 ($)",
    "dstrev": "change in state revenue per pupil, t minus t-1 ($) (first stage)",
    "dtotrev": "change in total revenue per pupil, t minus t-1 ($)",
}


# HFA "History of Foundation Allowance Growth" (Fiscal Brief, Feb. 25, 2026, Table 2): the minimum
# allowance of each fiscal year, including that year's equity payment (target from FY 2022-23).
HFA_MINIMUM = {2000: 5700, 2001: 6000, 2002: 6500, 2003: 6700, 2004: 6700, 2005: 6700, 2006: 6875,
               2007: 7108, 2008: 7204, 2009: 7316, 2010: 7316, 2011: 7316, 2012: 6846, 2013: 6966,
               2014: 7076, 2015: 7251, 2016: 7391, 2017: 7511, 2018: 7631, 2019: 7871, 2020: 8111,
               2021: 8111, 2022: 8700, 2023: 9150, 2024: 9608, 2025: 9608, 2026: 10050}


def prior_minimum(fy) -> np.ndarray:
    """The prior-year minimum allowance for fiscal year fy (HFA table); it equals the lowest Q among
    the formula districts of every year 2001-2026 (checked in ``rule_check``)."""
    return np.asarray([HFA_MINIMUM[int(t) - 1] for t in np.ravel(fy)], float).reshape(np.shape(fy))


def design(d: pd.DataFrame, county: bool = False) -> pd.DataFrame:
    X = pd.DataFrame(index=d.index)
    X["log_memb"] = np.log(d.memb_lag.clip(lower=1))
    X["log_lr"] = np.log(d.lr_pp_lag.clip(lower=1))
    X["pov_rate"] = d.pov_rate
    X["fed_share"] = d.fedrev_pp_f33_lag / d.totrev_pp_f33_lag
    X["title1_pp"] = d.title1_pp_f33_lag / 1000.0
    for c in ("share_black", "share_hisp", "share_asian", "share_other", "lat", "lon"):
        X[c] = d[c]
    for c in CONT:
        miss = X[c].isna() | ~np.isfinite(X[c])
        if miss.any():
            X[c] = X[c].where(~miss, X.loc[~miss, c].median())
            X[c + "_miss"] = miss.astype(float)
    lg = d.locale.astype(str).str[0].map(LOCALE_GROUP).fillna("na")
    for v in ("suburb", "town", "rural", "na"):
        if (lg == v).any():
            X[f"locale_{v}"] = (lg == v).astype(float)
    for t in sorted(d.fy.unique())[1:]:
        X[f"fy{t}"] = (d.fy == t).astype(float)
    if county:
        for c in sorted(d.county.dropna().unique())[1:]:
            X[f"cty{c}"] = (d.county == c).astype(float)
    return X


def load_dose(outcome: str = "dcur", years: Iterable[int] = range(2001, 2025), score: str = "gap",
              include_hh: bool = False, rule_only: bool = True, county: bool = False,
              trim: float = 0.01, rebuild: bool = False):
    """District x fiscal-year dose sample (local districts; public school academies are excluded).

    score   "gap": Q = F_{t-1} + eq_{t-1} minus the prior-year minimum ($); "dollars": Q = F_{t-1} + eq_{t-1}
    A       F_t + eq_t - (F_{t-1} + eq_{t-1}) - x_t: the increase beyond the basic increase ($/pupil)
    include_hh  add the 1994-95 hold-harmless districts (their rule needs the CPI; 20m add-backs)
    rule_only   keep district-years whose published allowance (and equity payment) equals the formula
    trim    drop the outcome's top and bottom ``trim`` share (F-33 spending jumps from reorganizations)
    """
    from experiments._core.sample import RDDSample
    d = panel(rebuild)
    d = d[d.fy.isin(list(years)) & d[outcome].notna() & d.leaid.notna()]
    if not include_hh:
        d = d[d.hh == 0]
    if rule_only:
        d = d[d.rule_ok == 1]
    if trim:
        lo, hi = d[outcome].quantile([trim, 1 - trim])
        d = d[d[outcome].between(lo, hi)]
    d = d.reset_index(drop=True)
    X = design(d, county=county)
    qmin = prior_minimum(d.fy)
    Q = d.Q.to_numpy(float) - (qmin if score == "gap" else 0.0)
    ex = {
        "A": d.A_pub.to_numpy(float),
        "a_ref": 0.0,
        "dose_levels": None,
        "schedule": schedule if score == "dollars" else (lambda q, g, unit="extra": schedule(
            np.asarray(q, float) + prior_minimum(np.broadcast_to(np.asarray(g), np.shape(q))), g, unit)),
        "group": d.fy.to_numpy(),
        "Z_names": [z for z in Z_NAMES if z in X.columns],
        "district_id": d.code.to_numpy(),
        "leaid": d.leaid.to_numpy(),
        "Q_dollars": d.Q.to_numpy(float),
        "prior_minimum": qmin,
        "x_basic": d.x.to_numpy(float),
        "A_formula": d.A_formula.to_numpy(float),
        "F": d.F.to_numpy(float), "eq": d["eq"].to_numpy(float),
        "hh": d.hh.to_numpy(int), "rule_ok": d.rule_ok.to_numpy(int),
        "kind": d.kind.to_numpy(),
        "membership": d.memb.to_numpy(float),
        "df": d,
    }
    return RDDSample(
        Q=Q, X=X.to_numpy(float), Y=d[outcome].to_numpy(float), threshold=THRESHOLD, name="mi_proposal_a",
        feature_names=list(X.columns),
        description=(f"Michigan foundation allowance (Proposal A), local districts x FY {min(years)}-{max(years)}; "
                     f"Q = prior-year allowance ({score}); A = increase beyond the basic increase; Y = {OUTCOMES[outcome]}"),
        citation=("MDE State Aid Financial Status Reports FY 2000-2026; State School Aid Act sec. 20 and 22c "
                  "(public acts 2000-2025); Census F-33 FY 2000-2024; SAIPE; NCES CCD 2015-16 / EDGE 2016-17."),
        treatment_rule=lambda q: (np.asarray(q) > 0).astype(int),
        extras=ex,
    )


def load():
    return load_dose()


def rule_check(include_hh: bool = True) -> pd.DataFrame:
    """Share of district-years whose published allowance (and equity payment) equals the formula,
    by fiscal year: non-hold-harmless districts, hold-harmless districts (CPI rule, no 20m add-backs)."""
    d = panel()
    rows = []
    for t, z in d.groupby("fy"):
        nh, hh = z[z.hh == 0], z[z.hh == 1]
        rows.append({"fy": int(t), "act": RULE[int(t)]["act"], "kind": RULE[int(t)]["kind"],
                     "x": RULE[int(t)]["x"], "n_formula": len(nh), "match": round(nh.rule_ok.mean(), 4),
                     "n_hh": len(hh), "match_hh": round(hh.rule_ok.mean(), 4) if include_hh else np.nan,
                     "share_A0": round(float((nh.A_pub.abs() < 1e-9).mean()), 3),
                     "min_Q_equals_HFA": bool(nh.Q[nh.rule_ok == 1].min() == HFA_MINIMUM[int(t) - 1]),
                     "A_max": float(nh.A_pub[nh.rule_ok == 1].max())})
    return pd.DataFrame(rows)


def readiness(s=None) -> dict:
    s = load_dose() if s is None else s
    Xt = np.column_stack([np.ones(s.n), s.X])
    gam = np.linalg.lstsq(Xt, s.Q, rcond=None)[0]
    eta = s.Q - Xt @ gam
    g = s.extras["group"]
    qw = s.Q - pd.Series(s.Q).groupby(g).transform("mean").to_numpy()
    return {"n": s.n, "districts": int(len(np.unique(s.extras["district_id"]))), "p": s.p,
            "R2": float(1 - eta.var() / s.Q.var()), "R2_within_year": float(1 - eta.var() / qw.var()),
            "sd_Q": float(s.Q.std()), "sd_eta": float(eta.std()),
            "share_ref": float((np.abs(s.extras["A"]) < 1e-9).mean()),
            "share_Q0": float((np.abs(s.Q) < 1e-9).mean())}


# ---------------------------------------------------------------------------------------------
# 6. Statute text (for reading the rule; data/processed/acts_txt is this function's cache)
# ---------------------------------------------------------------------------------------------
def act_section(fy: int, sec: str = "20") -> str:
    """Plain text of section ``sec`` of the School Aid act that set the fiscal year's allowance
    (e.g. act_section(2016) is sec. 20 of 2015 PA 85; act_section(2015, "22c") the equity payment)."""
    import html as _html
    year, num = RULE[int(fy)]["act"].split(" PA ")
    src = RAW / "acts" / f"{year}-PA-{int(num):04d}.htm"
    out = PROC / "acts_txt" / f"{year}-PA-{int(num):04d}.txt"
    if out.exists():
        t = out.read_text()
    else:
        s = src.read_bytes().decode("cp1252", errors="replace")
        t = _html.unescape(re.sub(r"<[^>]+>", " ", s))
        t = re.sub(r"[ \t\r\f\v\xa0]+", " ", t)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(t)
    t = re.sub(r"\s+", " ", t)
    m = re.search(r"Sec\.\s*%s\.\s" % re.escape(sec), t)
    if not m:
        raise KeyError(f"sec. {sec} not found in {src.name}")
    nxt = re.search(r"Sec\.\s*\d+[a-z]*\.\s", t[m.end():])
    return t[m.start(): m.end() + (nxt.start() if nxt else 20000)]
