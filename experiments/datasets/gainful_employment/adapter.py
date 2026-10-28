"""US gainful-employment (GE) programs: the 2015 debt-to-earnings (D/E) rates, an either
rule for passing and a both rule for failing.

Under the 2014 GE rule (34 CFR 668.403, 79 FR 64890), the Department of Education computes
two rates for each GE program (6-digit OPEID x credential level x 6-digit CIP): the annual
earnings rate, a = annual loan payment / annual earnings, and the discretionary income
rate, d = annual loan payment / (annual earnings - 1.5 x poverty guideline), both in %.
Annual earnings are the higher of the mean and median SSA earnings of the completer
cohort. Then

    pass = 1{a <= 8  or  d <= 20},
    fail = 1{a > 12 (or earnings 0)  and  d > 30 (or discretionary earnings <= 0)},
    zone = neither.

The only official rates under this rule, the 2015 rates (earnings of calendar 2014), were
released on 2017-01-09; the rule was rescinded in 2019 (84 FR 31392) before any program
lost aid. ED publishes each rate truncated to two decimals, 100.00 for a discretionary
rate with debt but non-positive discretionary earnings, and at most 999.99. Programs in
their transition period also got transitional rates (same earnings, the debt of the
2014-15 completers) when they did not pass and had 10+ transitional completers; the final
rates are the lower of the two (668.404(g)(3)), so the published result is the best of up
to four rate results.

Q = (a, d), the final rates as of the release: the lower of the standard and transitional
    rate on each dimension, recomputed from the published numerators and denominators
    (equal to the published rates except where an alternate-earnings appeal later
    replaced them). d is capped at 100, ED's own code for non-positive discretionary
    earnings; the cap changes no rule with a discretionary cutoff below 100.
    ``load()``: Q = d among programs with a > 8. ``load_annual()``: Q = a among programs
    with d > 20.
D = load_two_score(): 1{a > 8 and d > 20}, not passing (zone or fail), a both rule
    (the complement of the either rule for passing). load_two_score_fail():
    1{a > 12 and d > 30}. load(): 1{d > 20}. load_annual(): 1{a > 8}.
X = predetermined program and institution characteristics: credential level, 2-digit CIP
    family, institution type (control x length), state, IPEDS locale, log size of the
    D/E completer cohort (SSA verified count), log(1 + IPEDS completions) of the program
    in 2014-15 and 2015-16, log(1 + institution completions 2015-16), log number of GE
    programs with rates and of IPEDS campuses at the institution. No debt or earnings.
Y = load*(): 1{zero IPEDS completions in 2019-20}, "closed". The ``*_dlog`` loaders:
    log(1 + completions 2019-20) - log(1 + completions 2015-16). IPEDS completions are
    summed over every UNITID of the 6-digit OPEID, the program's CIP (CIP 2010 -> 2020 via
    NCES's crosswalk from 2019-20 on) and the IPEDS award levels of its credential level.

Sample: US programs whose institution is in IPEDS HD2016 and whose program had completions
in 2015-16, except programs whose campuses had all closed before the release (CLOSEDAT in
HD2016-HD2021). Programs with an approved alternate-earnings appeal stay in, with D as of
the release; ``extras`` flags them.
"""
from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample
# A standard-library .xls (BIFF8) reader; the environment has no xlrd.
from experiments.datasets.fl_school_grades._spreadsheet import read
from experiments.datasets.gainful_employment.download import C_YEARS, GE_FILE, RAW

HERE = Path(__file__).parent
PROCESSED = HERE / "data" / "processed"
CACHE = PROCESSED / "gainful_employment_programs.csv.gz"
SSA_FILE = RAW / "other" / "GE_SSA_Earnings_2014.xls"
IPEDS = RAW / "ipeds"
CROSSWALK = IPEDS / "Crosswalk2010to2020.csv"

RELEASE = pd.Timestamp("2017-01-09")
POVERTY_150 = 1.5 * 11770          # 150% of the 2015 HHS poverty guideline, one person
# Cut points (%) of 34 CFR 668.403(c): pass a <= 8 or d <= 20; fail a > 12 and d > 30.
PASS_CUT = (8.0, 20.0)
FAIL_CUT = (12.0, 30.0)
D_CAP = 100.0                       # Q2 = min(d, 100); 100 = ED's code for disc. earnings <= 0
EPS = 1e-6                          # the rates lie on a 0.01 grid
STATUS = {"PASS": 2, "ZONE": 1, "FAIL": 0}
STATUS_NAME = {2: "PASS", 1: "ZONE", 0: "FAIL"}
CRED = {1: "undergrad_cert", 2: "associate", 3: "bachelor", 4: "postbacc_cert",
        5: "master", 6: "doctoral", 7: "first_professional", 8: "grad_cert"}
# GE credential level -> IPEDS award levels (AWLEVEL; 20 and 21 split level 1 from 2019-20).
AWLEVEL = {1: (1, 2, 4, 20, 21), 2: (3,), 3: (5,), 4: (6,), 5: (7,), 6: (17, 18, 19),
           7: (18,), 8: (6, 8)}
ITYPE = {"PUBLIC LESS THAN 2 YEARS": "public_lt2", "PUBLIC 2 TO 3 YEARS": "public_2to3",
         "PUBLIC 4 YEARS OR MORE": "public_4plus",
         "PRIVATE, NOT-FOR-PROFIT LESS THAN 2 YEARS": "nonprofit_lt2",
         "PRIVATE, NOT-FOR-PROFIT 2 TO 3 YEARS": "nonprofit_2to3",
         "PRIVATE, NOT-FOR-PROFIT 4 YEARS OR MORE": "nonprofit_4plus",
         "PROPRIETARY LESS THAN 2 YEARS": "forprofit_lt2",
         "PROPRIETARY 2 TO 3 YEARS": "forprofit_2to3",
         "PROPRIETARY 4 YEARS OR MORE": "forprofit_4plus", "FOREIGN SCHOOLS": "foreign"}

_GE_COLS = {
    "Institution Code (six-digit OPEID)": "opeid6", "Institution Name": "name", "City": "city",
    "State": "state", "Zip": "zip", "Institution Type": "itype", "CIP Code": "cip",
    "CIP Name": "cip_name", "Credential Level": "cred_label",
    "Official Program Pass/Zone/Fail": "result_published",
    "Official Appeal Decision": "appeal_decision", "Appeal Status": "appeal_status",
    "Debt-to-Earnings Annual Rate": "a_pub", "Debt-to-Earnings Annual Rate Numerator": "payment",
    "Debt-to-Earnings Annual Rate Denominator": "earnings",
    "Debt-to-Earnings Annual Rate Pass/Fail/Zone": "a_result_pub",
    "Debt-to-Earnings Discretionary Income Rate": "d_pub",
    "Debt-to-Earnings Discretionary Income Rate Numerator": "d_payment",
    "Debt-to-Earnings Discretionary Income Rate Denominator": "disc_earnings",
    "Debt-to-Earnings Discretionary Income Rate Pass/Fail/Zone": "d_result_pub",
    "Debt-to-Earnings Transitional Rate": "ta_pub",
    "Debt-to-Earnings Transitional Rate Numerator": "t_payment",
    "Debt-to-Earnings Transitional Rate Denominator": "t_earnings",
    "Debt-to-Earnings Transitional Rate Pass/Fail/Zone": "ta_result_pub",
    "Debt-to-Earnings Transitional Discretionary Income Rate": "td_pub",
    "Debt-to-Earnings Transitional Discretionary Income Rate Numerator": "td_payment",
    "Debt-to-Earnings Transitional Discretionary Income Rate Denominator": "t_disc_earnings",
    "Debt-to-Earnings Transitional Discretionary Income Rate Pass/Fail/Zone": "td_result_pub",
    "Mean  Annual Earnings From SSA": "mean_earnings",
    "Median Annual Earnings from SSA": "median_earnings",
}


def _require(*paths: Path) -> None:
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(
                f"{p} missing. Run python -m experiments.datasets.gainful_employment.download")


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("*", "", regex=False).str.strip(),
                         errors="coerce")


def _sheet(path: Path) -> pd.DataFrame:
    rows = read(path, 0)
    hdr = [str(h).strip() if h is not None else f"col{j}" for j, h in enumerate(rows[0])]
    return pd.DataFrame([list(r) + [None] * (len(hdr) - len(r)) for r in rows[1:]], columns=hdr)


# --------------------------------------------------------------------------- GE files

def _read_ge() -> pd.DataFrame:
    raw = _sheet(RAW / GE_FILE)
    raw.columns = [" ".join(c.split()) if c not in _GE_COLS else c for c in raw.columns]
    df = raw.rename(columns=_GE_COLS)[list(_GE_COLS.values())]
    for c in ["opeid6", "name", "city", "state", "zip", "itype", "cip", "cip_name",
              "cred_label", "result_published", "appeal_decision", "appeal_status",
              "a_result_pub", "d_result_pub", "ta_result_pub", "td_result_pub"]:
        df[c] = df[c].fillna("").astype(str).str.strip()
    for c in ["a_pub", "d_pub", "ta_pub", "td_pub"]:
        df[f"{c}_starred"] = df[c].astype(str).str.contains(r"\*").astype(float)
    for c in ["a_pub", "payment", "earnings", "d_pub", "d_payment", "disc_earnings", "ta_pub",
              "t_payment", "t_earnings", "td_pub", "td_payment", "t_disc_earnings",
              "mean_earnings", "median_earnings"]:
        df[c] = _num(df[c])
    df["opeid6"] = df.opeid6.str.zfill(6)
    df["cip"] = df.cip.str.zfill(6)
    df["cred"] = df.cred_label.str[:2].astype(int)
    df["itype"] = df.itype.map(ITYPE)
    # "*": rates replaced after an approved alternate-earnings appeal; "**": the same,
    # a second mark that the glossary does not explain (15 programs, all approved appeals).
    df["result_mark"] = df.result_published.str.count(r"\*").astype(float)
    df["result_published"] = df.result_published.str.replace("*", "", regex=False)
    return df


def _read_ssa_counts() -> pd.DataFrame:
    """Size of the D/E completer cohort ("Verified Student Count") from FSA's 2014 SSA
    earnings file; the earnings columns are not used."""
    s = _sheet(SSA_FILE)
    s.columns = [c.strip() for c in s.columns]
    out = pd.DataFrame({
        "opeid6": s["OPEID"].astype(str).str.strip().str.zfill(6),
        "cip": s["CIP Code"].astype(str).str.strip().str.zfill(6),
        "cred": pd.to_numeric(s["Credential Level Code"], errors="coerce"),
        "cohort_n": _num(s["Verified Student Count"]),
    })
    return out.dropna(subset=["cred"]).astype({"cred": int}).drop_duplicates(
        ["opeid6", "cip", "cred"])


# --------------------------------------------------------------------------------- rule

def _trunc(x) -> np.ndarray:
    """ED's published precision: truncated (not rounded) to 0.01, at most 999.99."""
    return np.minimum(np.floor(np.asarray(x, float) * 100 + 1e-7) / 100, 999.99)


def _disc_rate(payment, disc_earnings) -> np.ndarray:
    payment, den = np.asarray(payment, float), np.asarray(disc_earnings, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = _trunc(100 * payment / np.where(den > 0, den, np.nan))
    # Discretionary earnings <= 0 (published as 0): 100.00 if there is debt, else 0.00.
    r = np.where(den > 0, r, np.where(payment > 0, 100.0, 0.0))
    return np.where(np.isnan(payment), np.nan, r)


def rates_at_release(df: pd.DataFrame) -> pd.DataFrame:
    """The four rates recomputed from the published numerators and denominators (the
    values before any alternate-earnings appeal) and the final rates a, d (the lower of
    standard and transitional on each dimension)."""
    out = pd.DataFrame(index=df.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["a_std"] = np.where(df.earnings > 0, _trunc(100 * df.payment / df.earnings), 999.99)
        out["ta"] = _trunc(100 * df.t_payment / df.t_earnings.where(df.t_earnings > 0))
    out["d_std"] = _disc_rate(df.payment, df.disc_earnings)
    out["td"] = _disc_rate(df.td_payment, df.t_disc_earnings)
    out["a"] = np.fmin(out.a_std, out.ta)
    out["d"] = np.fmin(out.d_std, out.td)
    return out


def status(a, d) -> np.ndarray:
    """2 = pass, 1 = zone, 0 = fail (34 CFR 668.403(c)); a, d in %."""
    a, d = np.asarray(a, float), np.asarray(d, float)
    passing = (a <= PASS_CUT[0] + EPS) | (d <= PASS_CUT[1] + EPS)
    failing = (a > FAIL_CUT[0] + EPS) & (d > FAIL_CUT[1] + EPS)
    return np.where(passing, 2, np.where(failing, 0, 1))


def not_passing_rule(Q: np.ndarray) -> np.ndarray:
    """D = 1{a > 8 and d > 20}: zone or fail."""
    return ((Q[:, 0] > PASS_CUT[0] + EPS) & (Q[:, 1] > PASS_CUT[1] + EPS)).astype(int)


def fail_rule(Q: np.ndarray) -> np.ndarray:
    """D = 1{a > 12 and d > 30}."""
    return ((Q[:, 0] > FAIL_CUT[0] + EPS) & (Q[:, 1] > FAIL_CUT[1] + EPS)).astype(int)


# -------------------------------------------------------------------------------- IPEDS

def _ipeds_csv(zpath: Path, usecols=None) -> pd.DataFrame:
    """The revised (``_rv``) member when the zip has one, else the original."""
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        name = next((n for n in names if n.lower().endswith("_rv.csv")), names[0])
        with z.open(name) as fh:
            d = pd.read_csv(fh, dtype=str, encoding="latin-1", keep_default_na=False)
    d.columns = [c.strip().upper() for c in d.columns]
    return d[usecols] if usecols else d


def _hd(year: int) -> pd.DataFrame:
    d = _ipeds_csv(IPEDS / f"HD{year}.zip")
    d["opeid6"] = d.OPEID.str.strip().str.zfill(8).str[:6]
    d.loc[~d.OPEID.str.strip().str.isdigit(), "opeid6"] = ""
    d["closed_at"] = pd.to_datetime(d.CLOSEDAT.where(d.CLOSEDAT.str.match(r"\d\d/\d\d/\d{4}$")),
                                    format="%m/%d/%Y", errors="coerce")
    return d


def _completions(year: int) -> pd.DataFrame:
    c = _ipeds_csv(IPEDS / f"C{year}_A.zip", ["UNITID", "CIPCODE", "MAJORNUM", "AWLEVEL",
                                              "CTOTALT"])
    c = c[c.MAJORNUM.str.strip() == "1"]                    # first majors
    c = c.assign(awlevel=pd.to_numeric(c.AWLEVEL, errors="coerce"),
                 n=pd.to_numeric(c.CTOTALT, errors="coerce").fillna(0),
                 cip=c.CIPCODE.str.strip())
    return c[["UNITID", "cip", "awlevel", "n"]]


def _cip_map() -> pd.DataFrame:
    """CIP 2010 -> CIP 2020 (NCES crosswalk), 6-digit codes, one row per pair."""
    cw = pd.read_csv(CROSSWALK, dtype=str, encoding="latin-1", keep_default_na=False)
    for c in ["CIPCode2010", "CIPCode2020"]:
        cw[c] = cw[c].str.replace('="', "", regex=False).str.replace('"', "", regex=False)
    cw = cw[(cw.CIPCode2010.str.len() == 7) & (cw.CIPCode2020.str.len() == 7)]
    return cw[["CIPCode2010", "CIPCode2020", "Action"]].rename(
        columns={"CIPCode2010": "cip", "CIPCode2020": "cip2020", "Action": "cip_action"})


def _program_completions(ge: pd.DataFrame) -> pd.DataFrame:
    """IPEDS completions of each GE program, C2015_A (2014-15) to C2021_A (2020-21).

    A UNITID belongs to a 6-digit OPEID if HD2016 or that year's HD file says so, so a
    later change of OPEID does not end a program. Completions are first majors, summed
    over the UNITIDs, the IPEDS award levels of the credential level and, from 2019-20
    (CIP 2020), the CIP 2020 codes the program's CIP 2010 code maps to.

    ``campuses_closed_at``: the closing date (CLOSEDAT, HD2016-HD2021) of the last of the
    campuses (UNITIDs) with completions in the program in 2015-16, if all of them
    closed."""
    keys = ge[["opeid6", "cip", "cred"]].copy()
    keys["cip_dot"] = keys.cip.str[:2] + "." + keys.cip.str[2:]
    lv = pd.DataFrame([(c, a) for c, levels in AWLEVEL.items() for a in levels],
                      columns=["cred", "awlevel"])
    cmap = _cip_map()
    hd = {y: _hd(y) for y in C_YEARS}
    closed = pd.concat([hd[y][["UNITID", "closed_at"]] for y in C_YEARS if y >= 2016])
    closed = closed.dropna().groupby("UNITID").closed_at.min()
    out = keys[["opeid6", "cip", "cred"]].copy()
    for y in C_YEARS:
        units = pd.concat([hd[2016][["UNITID", "opeid6"]], hd[y][["UNITID", "opeid6"]]])
        units = units[units.opeid6 != ""].drop_duplicates()
        c = _completions(y).merge(units, on="UNITID")
        k = keys.merge(lv, on="cred")
        if y >= 2020:                                       # CIP 2020 in C2020_A on
            k = k.merge(cmap, left_on="cip_dot", right_on="cip", how="left",
                        suffixes=("", "_cw"))
            k["cip_y"] = k.cip2020.fillna(k.cip_dot)
        else:
            k["cip_y"] = k.cip_dot
        k = k[["opeid6", "cip", "cred", "cip_y", "awlevel"]].drop_duplicates()
        k = k.merge(c.rename(columns={"cip": "cip_y"}), on=["opeid6", "cip_y", "awlevel"])
        s = k.groupby(["opeid6", "cip", "cred"], as_index=False).n.sum()
        out = out.merge(s.rename(columns={"n": f"c{y}"}), on=["opeid6", "cip", "cred"],
                        how="left")
        out[f"c{y}"] = out[f"c{y}"].fillna(0.0)
        if y == 2016:
            u = k[k.n > 0].assign(closed_at=lambda t: t.UNITID.map(closed))
            last = u.groupby(["opeid6", "cip", "cred"]).closed_at.agg(
                lambda s: s.max() if s.notna().all() else pd.NaT)
            out = out.merge(last.rename("campuses_closed_at").reset_index(),
                            on=["opeid6", "cip", "cred"], how="left")
    act = keys.merge(cmap, left_on="cip_dot", right_on="cip", how="left", suffixes=("", "_cw"))
    act["cip_changed"] = (act.cip_action.fillna("") != "No substantive changes") | (
        act.cip2020.fillna("") != act.cip_dot)
    chg = act.groupby(["opeid6", "cip", "cred"], as_index=False).cip_changed.max()
    return out.merge(chg, on=["opeid6", "cip", "cred"], how="left")


def _institutions() -> pd.DataFrame:
    """Per 6-digit OPEID: HD2016 campuses, main-campus locale and total completions
    2015-16 (first majors, all CIPs and levels)."""
    hd = _hd(2016)
    hd = hd[hd.opeid6 != ""]
    main = hd.sort_values("OPEID").drop_duplicates("opeid6")      # branch 00 first
    inst = hd.groupby("opeid6").UNITID.nunique().rename("n_campuses").to_frame()
    inst["locale"] = main.set_index("opeid6").LOCALE.astype(str).str[:1].reindex(inst.index)
    c = _ipeds_csv(IPEDS / "C2016_A.zip", ["UNITID", "CIPCODE", "MAJORNUM", "AWLEVEL",
                                           "CTOTALT"])
    c = c[(c.CIPCODE.str.strip() == "99") & (c.MAJORNUM.str.strip() == "1")]
    c = c.assign(n=pd.to_numeric(c.CTOTALT, errors="coerce").fillna(0))
    c = c.merge(hd[["UNITID", "opeid6"]], on="UNITID")
    inst["inst_c2016"] = c.groupby("opeid6").n.sum().reindex(inst.index).fillna(0.0)
    return inst.reset_index()


# -------------------------------------------------------------------------------- build

def build() -> pd.DataFrame:
    """Every program in the 2015 final-rates file, with the rates at release, the rule,
    IPEDS completions 2014-15 to 2020-21, institution facts and the cohort size."""
    if not CACHE.exists():
        df = _build()
        PROCESSED.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=PROCESSED, prefix=".tmp_", suffix=".csv.gz")
        os.close(fd)
        try:
            df.to_csv(tmp, index=False, compression="gzip")
            os.chmod(tmp, 0o644)
            os.replace(tmp, CACHE)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    p = pd.read_csv(CACHE, keep_default_na=False, na_values=[""], low_memory=False,
                    dtype={"opeid6": str, "cip": str, "zip": str, "locale": str})
    for c in ["name", "state", "itype", "cip_name", "result_published", "appeal_decision",
              "appeal_status", "result_rule", "locale"]:
        p[c] = p[c].fillna("").astype(str)
    p["campuses_closed_at"] = pd.to_datetime(p.campuses_closed_at)
    return p


def _build() -> pd.DataFrame:
    _require(RAW / GE_FILE, SSA_FILE, CROSSWALK,
             *[IPEDS / f"C{y}_A.zip" for y in C_YEARS],
             *[IPEDS / f"HD{y}.zip" for y in C_YEARS])
    ge = _read_ge()
    ge = pd.concat([ge, rates_at_release(ge)], axis=1)
    ge["status_rule"] = status(ge.a, ge.d)
    ge["status_std_rule"] = status(ge.a_std, ge.d_std)
    ge["result_rule"] = ge.status_rule.map(STATUS_NAME)
    ge["status_published"] = ge.result_published.map(STATUS)
    ge["appeal_approved"] = ge.appeal_decision.eq("Approve").astype(float)
    ge = ge.merge(_read_ssa_counts(), on=["opeid6", "cip", "cred"], how="left")
    ge = ge.merge(_program_completions(ge), on=["opeid6", "cip", "cred"], how="left")
    inst = _institutions()
    ge = ge.merge(inst, on="opeid6", how="left")
    ge["in_hd2016"] = ge.n_campuses.notna().astype(float)
    ge["n_programs_inst"] = ge.groupby("opeid6").cip.transform("size").astype(float)
    ge["closed_before_release"] = (ge.campuses_closed_at < RELEASE).astype(float)
    by2020 = ge.campuses_closed_at < pd.Timestamp("2020-07-01")
    ge["campuses_closed_by_2020"] = by2020.astype(float)
    ge["y_closed"] = (ge.c2020 == 0).astype(float)
    ge["y_closed_2021"] = (ge.c2021 == 0).astype(float)
    ge["y_dlog"] = np.log1p(ge.c2020) - np.log1p(ge.c2016)
    ge["y_dlog_2021"] = np.log1p(ge.c2021) - np.log1p(ge.c2016)
    ge["placebo_dlog"] = np.log1p(ge.c2016) - np.log1p(ge.c2015)
    return ge.sort_values(["opeid6", "cred", "cip"]).reset_index(drop=True)


# ------------------------------------------------------------------------------- checks

def rule_check() -> dict:
    """The rule against the published results, the recomputed rates against the
    published rates, and the IPEDS match."""
    p = build()
    us = p.itype != "foreign"
    appr = p.appeal_approved == 1
    pub_np = p.status_published < 2
    rule_np = p.status_rule < 2
    pub_f = p.status_published == 0
    rule_f = p.status_rule == 0
    std_np = p.status_std_rule < 2
    same = {}
    for c, r in [("a_pub", "a_std"), ("d_pub", "d_std"), ("ta_pub", "ta"), ("td_pub", "td")]:
        m = p[c].notna() & (p[f"{c}_starred"] == 0) & ~appr
        same[c] = float((np.abs(p[c] - p[r]) < 0.005)[m].mean())
    return {
        "programs": int(len(p)),
        "published_counts": p.result_published.value_counts().to_dict(),
        "rule_counts_at_release": p.result_rule.value_counts().to_dict(),
        "recomputed_rate_equals_published (no appeal)": same,
        "result_agree_no_approved_appeal": float((p.status_rule == p.status_published)[~appr].mean()),
        "result_mismatch_no_approved_appeal": int((p.status_rule != p.status_published)[~appr].sum()),
        "approved_appeals": int(appr.sum()),
        "approved_appeals_rule_at_release": p.result_rule[appr].value_counts().to_dict(),
        "approved_appeals_published": p.result_published[appr].value_counts().to_dict(),
        "not_passing_agree_all": float((rule_np == pub_np).mean()),
        "not_passing_mismatch_all": int((rule_np != pub_np).sum()),
        "fail_agree_all": float((rule_f == pub_f).mean()),
        "fail_mismatch_all": int((rule_f != pub_f).sum()),
        "standard_rates_only_not_passing_agree": float((std_np == pub_np)[~appr].mean()),
        "passed_on_transitional_rates": int(((p.status_std_rule < 2) & (p.status_rule == 2)).sum()),
        "zone_from_fail_on_transitional": int(((p.status_std_rule == 0)
                                               & (p.status_rule == 1)).sum()),
        "us_programs": int(us.sum()),
        "institution_in_hd2016": float(p.in_hd2016[us].mean()),
        "program_completions_2014_15_or_2015_16": float(((p.c2015 + p.c2016) > 0)[us].mean()),
        "program_completions_2015_16": float((p.c2016 > 0)[us].mean()),
        "closed_before_release_programs": int(p.closed_before_release[us].sum()),
        "cohort_n_matched": float(p.cohort_n.notna().mean()),
    }


# ------------------------------------------------------------------------------ samples

def _frame() -> pd.DataFrame:
    p = build()
    m = ((p.itype != "foreign") & (p.in_hd2016 == 1) & (p.c2016 > 0)
         & (p.closed_before_release == 0))
    return p[m].reset_index(drop=True)


def _design(df: pd.DataFrame) -> tuple:
    X = pd.DataFrame(index=df.index)
    X["log_cohort_n"] = np.log(df.cohort_n.fillna(df.cohort_n.median()))
    X["cohort_n_missing"] = df.cohort_n.isna().astype(float)
    X["log1p_c2015"] = np.log1p(df.c2015)
    X["log1p_c2016"] = np.log1p(df.c2016)
    X["log1p_inst_c2016"] = np.log1p(df.inst_c2016)
    X["log_programs_inst"] = np.log(df.n_programs_inst)
    X["log_campuses"] = np.log(df.n_campuses)
    for c in sorted(CRED):
        if c != 1:                                           # undergraduate certificate base
            X[f"cred_{CRED[c]}"] = (df.cred == c).astype(float)
    for t in sorted(set(ITYPE.values()) - {"foreign", "forprofit_lt2"}):
        X[f"itype_{t}"] = (df.itype == t).astype(float)      # for-profit < 2 years base
    fam = df.cip.str[:2]
    big = fam.value_counts()
    big = sorted(big[big >= 30].index)
    for f in big[1:]:
        X[f"cip2_{f}"] = (fam == f).astype(float)
    X["cip2_other"] = (~fam.isin(big)).astype(float)
    sv = df.state.value_counts()
    states = sorted(sv[sv >= 20].index)
    for s in states:
        if s != "CA":                                        # California base
            X[f"state_{s}"] = (df.state == s).astype(float)
    X["state_other"] = (~df.state.isin(states)).astype(float)
    for k, v in {"2": "suburb", "3": "town", "4": "rural"}.items():   # city base
        X[f"locale_{v}"] = (df.locale == k).astype(float)
    X = X.loc[:, X.std() > 0]
    return X.to_numpy(float), list(X.columns)


def _extras(df: pd.DataFrame) -> dict:
    return {
        "opeid6": df.opeid6.to_numpy(str), "cip": df.cip.to_numpy(str),
        "cred": df.cred.to_numpy(int), "itype": df.itype.to_numpy(str),
        "state": df.state.to_numpy(str), "name": df.name.to_numpy(str),
        "credential": df.cred.map(CRED).to_numpy(str),
        "a_std": df.a_std.to_numpy(float), "d_std": df.d_std.to_numpy(float),
        "d_uncapped": df.d.to_numpy(float),
        "a_transitional": df.ta.to_numpy(float), "d_transitional": df.td.to_numpy(float),
        "a_published": df.a_pub.to_numpy(float), "d_published": df.d_pub.to_numpy(float),
        "result_rule_at_release": df.result_rule.to_numpy(str),
        "result_published": df.result_published.to_numpy(str),
        "appeal_approved": df.appeal_approved.to_numpy(float),
        "appeal_decision": df.appeal_decision.to_numpy(str),
        "disc_earnings_nonpositive": (df.disc_earnings <= 0).to_numpy(float),
        "transitional_computed": df.ta.notna().to_numpy(float),
        "cip_changed_2020": df.cip_changed.astype(float).to_numpy(),
        "completions": df[[f"c{y}" for y in C_YEARS]].to_numpy(float),
        "completion_years": tuple(f"{y - 1}-{str(y)[2:]}" for y in C_YEARS),
        "campuses_closed_by_2020": df.campuses_closed_by_2020.to_numpy(float),
        "y_closed_2021": df.y_closed_2021.to_numpy(float),
        "y_dlog": df.y_dlog.to_numpy(float), "y_closed": df.y_closed.to_numpy(float),
        "placebo_dlog_2015_16": df.placebo_dlog.to_numpy(float),
        "pass_rule": "pass = 1{a <= 8 or d <= 20}; fail = 1{a > 12 and d > 30}; zone otherwise",
        "pass_cutoffs": PASS_CUT, "fail_cutoffs": FAIL_CUT,
    }


_CITATION = ("US Department of Education, 2015 GE debt-to-earnings rates (final, released "
             "2017-01-09), studentaid.gov; 34 CFR 668.403-668.404 (79 FR 64890); NCES IPEDS "
             "completions C2015_A-C2021_A and HD2015-HD2021")
_Y_LABEL = {"y_closed": "1{zero IPEDS completions in 2019-20}",
            "y_dlog": "log(1 + completions 2019-20) - log(1 + completions 2015-16)"}


def _two_score(boundary: str, y_col: str) -> RDDSample:
    df = _frame()
    X, names = _design(df)
    cut, rule, lab = ((PASS_CUT, not_passing_rule, "not passing (zone or fail)")
                      if boundary == "pass" else (FAIL_CUT, fail_rule, "fail"))
    Q = np.column_stack([df.a, np.minimum(df.d, D_CAP)])
    pub = (df.status_published < 2) if boundary == "pass" else (df.status_published == 0)
    ex = _extras(df)
    ex.update({"boundary": boundary, "cutoffs": cut, "eligible_published": pub.to_numpy(float),
               "treated_side": f"{lab}: above both cut points (Q1 > {cut[0]:g} and "
                               f"Q2 > {cut[1]:g}); the untreated side is the either rule "
                               f"Q1 <= {cut[0]:g} or Q2 <= {cut[1]:g}"})
    return RDDSample(
        Q=Q, X=X, Y=df[y_col].to_numpy(float), threshold=cut,
        name=f"gainful_employment_two_score_{boundary}_{y_col}", feature_names=names,
        description=("US gainful-employment programs, 2015 D/E rates (released 2017-01-09). "
                     "Q = (annual earnings rate %, discretionary income rate % capped at 100), "
                     f"final rates at release; treatment = 1{{Q1 > {cut[0]:g} and Q2 > "
                     f"{cut[1]:g}}}, {lab}; Y = {_Y_LABEL[y_col]}."),
        citation=_CITATION, treatment_rule=rule, extras=ex,
    )


def _single(y_col: str, score: str = "d") -> RDDSample:
    """One score, the other held on the side where it does not pass: d among a > 8
    (score "d"), or a among d > 20 (score "a"). D = 1{score above its cut}, not passing."""
    df = _frame()
    other, j = ("a", 0) if score == "d" else ("d", 1)
    df = df[df[other] > PASS_CUT[j] + EPS].reset_index(drop=True)
    X, names = _design(df)
    cut = PASS_CUT[1 - j]
    q = np.minimum(df.d, D_CAP) if score == "d" else df.a
    ex = _extras(df)
    ex.update({other: df[other].to_numpy(float), "cutoff": cut,
               "eligible_published": (df.status_published < 2).to_numpy(float)})
    lab = {"d": ("discretionary income rate (%, capped at 100)", "an annual D/E rate above 8%"),
           "a": ("annual earnings rate (%)", "a discretionary D/E rate above 20%")}[score]
    return RDDSample(
        Q=q.to_numpy(float), X=X, Y=df[y_col].to_numpy(float), threshold=cut,
        name=f"gainful_employment_{score}_{y_col}", feature_names=names,
        description=(f"US gainful-employment programs with {lab[1]}, 2015 rates. Q = "
                     f"{lab[0]}; treatment = 1{{Q > {cut:g}}}, not passing; "
                     f"Y = {_Y_LABEL[y_col]}."),
        citation=_CITATION, treatment_rule=lambda Q: (Q > cut + EPS).astype(int),
        extras=ex,
    )


def load() -> RDDSample:
    """Single score: discretionary rate among programs with annual rate > 8%;
    D = 1{d > 20} (not passing); Y = closed (zero completions in 2019-20)."""
    return _single("y_closed")


def load_dlog() -> RDDSample:
    return _single("y_dlog")


def load_annual() -> RDDSample:
    """Single score: annual rate among programs with discretionary rate > 20%;
    D = 1{a > 8} (not passing); Y = closed. Far more mass at its cutoff than ``load()``."""
    return _single("y_closed", "a")


def load_annual_dlog() -> RDDSample:
    return _single("y_dlog", "a")


def load_two_score() -> RDDSample:
    """Q = (a, d); D = 1{a > 8 and d > 20} (not passing); Y = closed."""
    return _two_score("pass", "y_closed")


def load_two_score_dlog() -> RDDSample:
    return _two_score("pass", "y_dlog")


def load_two_score_fail() -> RDDSample:
    """Q = (a, d); D = 1{a > 12 and d > 30} (fail); Y = closed."""
    return _two_score("fail", "y_closed")


def load_two_score_fail_dlog() -> RDDSample:
    return _two_score("fail", "y_dlog")
