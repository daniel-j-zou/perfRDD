"""North Carolina No Child Left Behind: Adequate Yearly Progress (AYP), 2003-04 to 2009-10.

Under NCLB a school makes AYP when every accountable student group meets the annual
measurable objective (AMO) in reading and in mathematics, tests at least 95% of its
students, and the school shows progress on the other academic indicator (OAI). A Title I
school that misses AYP in the same subject two years running enters Title I School
Improvement and moves up one stage with each further miss: public school choice, then
supplemental educational services (SES), corrective action, a restructuring plan and
restructuring (NC Accountability Workbook 2010, sec. 3.2; DPI "Adequate Yearly Progress in
North Carolina", 2011).

The rule, per cell (school x year x grade span x subject x student group):

* accountable cells: student groups with at least 40 full-academic-year scores, plus the
  school as a whole whatever its size. The database reports % proficient only for these
  cells (no group cell below 40 valid scores has one). 1 + proficiency cells +
  participation cells equals the published number of AYP targets for 91-99% of
  school-years in 2004-2007 and 2009-2011.
* a cell meets its target if p >= CI or p >= SH, where p is the % proficient,
  CI = AMO - 1.645 sqrt(AMO (1 - AMO) / n) (the 95% confidence interval, one-sided, with n
  the group's valid scores) and SH = 1 - 0.9 (1 - p_prev) (safe harbor: 10% fewer
  non-proficient students than in the same cell last year; only when that cell was
  accountable last year). Ahn & Vigdor's "effective threshold" is min(CI, SH).
* margin of a cell = p - min(CI, SH), in percentage points; Q_R, Q_M = the smallest margin
  over all reading / math cells of the school (both grade spans); Q = min(Q_R, Q_M)
  (Ahn & Vigdor 2014's assignment variable). The school meets every proficiency target
  iff Q >= 0, i.e. Q_R >= 0 and Q_M >= 0.

The AMOs are the workbook's (3-8 reading/math: 68.9/74.6 to 2004, 76.7/81.0 in 2005,
76.7/65.8 in 2006-07 after the new math test, 43.2/77.2 in 2008-10 after the new reading
test, 71.6/88.6 in 2011; grade 10: 52.0/54.9, 35.4/70.8, 38.5/68.4, 69.3/84.2); the
database carries them in every cell. The workbook says "95% confidence interval" without a
formula; z = 1.645 on the AMO's standard error fits the published status best, or within
0.5 points of the best, in each clean year (vs 1.28, 1.96, 2.33 and an observed-p error).

Q = Q (pp) for load(); (Q_R, Q_M) for load_two_score().
D = 1{Q < 0}: the school missed AYP; the treated side is below the cutoff. In the default
    sample, Title I schools already in School Improvement (improvement year >= 1 during
    year t), a miss moves the school up one stage in t+1: choice -> SES -> corrective
    action -> restructuring plan -> restructuring. This is where the effect is: Ahn &
    Vigdor find their largest effect for restructuring, and here the jump in next year's
    proficiency is +2 to +4 points in these schools and not positive in the others
    (``local_jump``; README).
X = school characteristics not computed from test scores: % free/reduced-price lunch,
    log enrollment, test-taker shares of Black, Hispanic, Asian, American Indian,
    multiracial, LEP and disabled students (counts only), teacher licensure, experience,
    degrees, turnover and National Board shares, charter, magnet/alternative/special,
    year-round calendar, Title I (known from 2006-07), grade-span category, NCES locale,
    district and year indicators.
Y = next year's % proficient (all students, both grade spans pooled by valid scores), math
    for load() and reading for load_reading().

Sample: spring years with both subjects' cells, every participation rate >= 95% (so AYP
is decided by proficiency alone) and the next year's proficiency. AYP years in the data
are 2003-2011; 2003 serves only as the safe-harbor base (its published AYP does not follow
the database's cells: 66% agreement) and 2011 has no outcome. MAIN_YEARS = (2004, 2005,
2010) are the years in which the rule reproduces the published AYP status for 96-98% of
schools; ALL_YEARS = 2004-2010 adds 2006-2009 (80-90%: new math test with a
concordance-based safe harbor in 2006, new reading test in 2008, retests in 2009,
unreported participation cells in 2008). See ``rule_check()`` and the README.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample
# Standard-library .xls reader (no xlrd in the environment), shared with the Florida dataset.
from experiments.datasets.fl_school_grades._spreadsheet import read as read_xls
from experiments.datasets.nc_ayp._mdb import MDB
from experiments.datasets.nc_ayp.download import ABC_FILES, SRC_MEMBER, SRC_ZIP

HERE = Path(__file__).parent
PROCESSED = HERE / "data" / "processed"
CELLS = PROCESSED / "nc_ayp_cells.csv.gz"
CACHE = PROCESSED / "nc_ayp_panel.csv.gz"

Z = 1.645                 # one-sided 95% confidence interval
SH_FACTOR = 0.9           # safe harbor: 10% fewer non-proficient than last year
PART_MIN = 0.95           # participation target
EPS = 1e-6                # pp; margins are rounded to 1e-6 pp
AYP_YEARS = tuple(range(2003, 2012))
ALL_YEARS = tuple(range(2004, 2011))     # years with a safe-harbor base and an outcome
MAIN_YEARS = (2004, 2005, 2010)          # rule agrees with published AYP for >= 96%
# database subgroup code -> short name
GROUPS = {"ALL": "all", "INDIAN": "amind", "ASIAN": "asian", "BLACK": "black",
          "HISPANIC": "hispanic", "MRACIAL": "multi", "WHITE": "white", "FREELNCH": "frl",
          "LMTENG": "lep", "DISABL": "swd", "PACI_ISLANDER": "pacisl"}
STAGES = {0: "none", 1: "choice", 2: "ses", 3: "corrective", 4: "restr_plan",
          5: "restructuring"}
_TXT = ["lea", "sch", "school_id", "span", "subject", "group", "category", "locale", "locale_raw", "name",
        "school_type", "site_type", "calendar", "title1", "si_phase", "abc_code",
        "binding_group", "binding_subject"]
_CITATION = ("NC Department of Public Instruction, School Report Card research database "
             "2010-11 (SCH_ADEQUATE_PROGRESS, SCHOOL_SUMMARY, SITE), via the Internet "
             "Archive; Ahn & Vigdor (2014), NBER WP 20511")


# ------------------------------------------------------------------------ raw files

def _need(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run python -m experiments.datasets.nc_ayp.download")
    return path


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _str(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip()


def _read_tables() -> dict:
    """The five tables the adapter uses, read from the Access database. The 675 MB
    database is extracted from the zip to a temporary file next to the cache and deleted
    afterwards."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(_need(SRC_ZIP)) as z:
        fd, tmp = tempfile.mkstemp(suffix=".mdb", dir=PROCESSED)
        try:
            with os.fdopen(fd, "wb") as out, z.open(SRC_MEMBER) as src:
                shutil.copyfileobj(src, out, 1 << 22)
            with MDB(tmp) as m:
                return {t: pd.DataFrame(m.table(t)) for t in
                        ("SCH_ADEQUATE_PROGRESS", "SCHOOL_SUMMARY", "SITE",
                         "SCH_AYP_ATTENDANCE", "HIGH_SCH_GRADUATION")}
        finally:
            os.remove(tmp)


def _year(s: pd.Series) -> pd.Series:
    """'2004-2005' -> 2005 (the spring of the school year)."""
    return _str(s).str[5:].astype(int)


def _abc_codes() -> pd.DataFrame:
    """Published AYP status and AYP code from the archived ABCs lists (cross-check)."""
    out = []
    for y, path in ABC_FILES.items():
        if not path.exists():
            continue
        rows = read_xls(path, 0)
        i = next(k for k, r in enumerate(rows[:12])
                 if r and "LEA Code" in [str(x).strip() for x in r])
        h = [str(x).strip() for x in rows[i]]
        d = pd.DataFrame([r[:len(h)] + [None] * (len(h) - len(r)) for r in rows[i + 1:]],
                         columns=h)
        d = d[d["LEA Code"].notna()]
        out.append(pd.DataFrame({
            "lea": _str(d["LEA Code"]), "sch": _str(d["Sch Code"]), "year": y,
            "abc_ayp": _str(d["AYP"]), "abc_code": _str(d["AYP Code"])}))
    return (pd.concat(out, ignore_index=True).drop_duplicates(["lea", "sch", "year"])
            if out else pd.DataFrame(columns=["lea", "sch", "year", "abc_ayp", "abc_code"]))


# -------------------------------------------------------------------------- cells

def _cells(sap: pd.DataFrame) -> pd.DataFrame:
    """One row per school x year x span x subject x group with a reported proficiency
    or participation rate, with the rule's CI and SH bounds and the cell margin (pp)."""
    base = pd.DataFrame({
        "lea": _str(sap.SAP_LEA_CD), "sch": _str(sap.SAP_SCH_CD), "year": _year(sap.SAP_YEAR_CD),
        "span": np.where(_str(sap.SAP_GRADE_CD) == "10", "10", "3-8"),
        "subject": _str(sap.SAP_TEST_CD)})
    long = []
    for code, g in GROUPS.items():
        d = base.copy()
        d["group"] = g
        for col, src in (("n", "VALSCORE"), ("elig", "ELIG")):
            c = f"SAP_{src}_{code}_NUM"
            d[col] = _num(sap[c]) if c in sap else np.nan
        for col, src in (("tested", "TESTED"), ("target", "TARGET"), ("p", "MET")):
            c = f"SAP_{src}_{code}_PCT"
            d[col] = _num(sap[c]) if c in sap else np.nan
        long.append(d[d.p.notna() | d.tested.notna()])
    c = pd.concat(long, ignore_index=True)
    c = c.drop_duplicates(["lea", "sch", "year", "span", "subject", "group"])
    key = ["lea", "sch", "span", "subject", "group"]
    prev = c.loc[c.p.notna(), key + ["year", "p"]].rename(columns={"p": "p_prev"})
    prev["year"] += 1
    c = c.merge(prev, on=key + ["year"], how="left")
    t, n = c.target, c.n.clip(lower=1)
    c["ci_bound"] = t - Z * np.sqrt(t * (1 - t) / n)
    c["sh_bound"] = 1 - SH_FACTOR * (1 - c.p_prev)
    c["eff_threshold"] = np.fmin(c.ci_bound, c.sh_bound)          # NaN-safe min
    c["margin"] = np.round(100 * (c.p - c.eff_threshold), 6)
    c["part_miss"] = (c.tested < PART_MIN - 1e-9).astype(float).where(c.tested.notna())
    return c.sort_values(["lea", "sch", "year", "span", "subject", "group"]).reset_index(drop=True)


def cells() -> pd.DataFrame:
    """The cell table (cached with the panel)."""
    build()
    return pd.read_csv(CELLS, dtype={c: str for c in ("lea", "sch", "span", "subject", "group")},
                       keep_default_na=False, na_values=[""])


# -------------------------------------------------------------------------- panel

def build() -> pd.DataFrame:
    """One row per school-year 2002-03 to 2010-11 with the margins, published AYP and
    school-improvement status, covariates, history and next-year outcomes."""
    if not CACHE.exists() or not CELLS.exists():
        tabs = _read_tables()
        c = _cells(tabs["SCH_ADEQUATE_PROGRESS"])
        p = _panel(c, tabs)
        PROCESSED.mkdir(parents=True, exist_ok=True)
        for df, path in ((c, CELLS), (p, CACHE)):
            fd, tmp = tempfile.mkstemp(suffix=".csv.gz", dir=PROCESSED)
            os.close(fd)
            df.to_csv(tmp, index=False, compression="gzip")
            os.chmod(tmp, 0o644)
            os.replace(tmp, path)
    p = pd.read_csv(CACHE, dtype={c: str for c in _TXT}, keep_default_na=False,
                    na_values=[""])
    for col in _TXT:
        if col in p:
            p[col] = p[col].fillna("")
    return p


def _panel(c: pd.DataFrame, tabs: dict) -> pd.DataFrame:
    key = ["lea", "sch", "year"]
    prof = c[c.p.notna()]
    # margins by subject and overall, and the binding cell
    q = prof.pivot_table(index=key, columns="subject", values="margin", aggfunc="min")
    q = q.rename(columns={"R": "q_read", "M": "q_math"}).reset_index()
    nb = prof.pivot_table(index=key, columns="subject", values="margin", aggfunc="size")
    nb = nb.rename(columns={"R": "n_cells_read", "M": "n_cells_math"}).reset_index()
    b = prof.loc[prof.groupby(key).margin.idxmin(), key + ["group", "subject"]]
    b = b.rename(columns={"group": "binding_group", "subject": "binding_subject"})
    part = c[c.tested.notna()].groupby(key).agg(
        part_min=("tested", "min"), part_misses=("part_miss", "sum"),
        n_part_cells=("tested", "size")).reset_index()
    # all-students % proficient, both spans pooled by valid scores
    al = prof[prof.group == "all"].assign(w=lambda d: d.n.fillna(0).clip(lower=0))
    al = al.assign(pw=al.p * al.w)
    pa = al.groupby(key + ["subject"]).agg(pw=("pw", "sum"), w=("w", "sum")).reset_index()
    pa["prof"] = 100 * pa.pw / pa.w.where(pa.w > 0)
    pa = pa.pivot_table(index=key, columns="subject", values="prof").rename(
        columns={"R": "prof_read", "M": "prof_math"}).reset_index()
    span = c.groupby(key).span.agg(lambda s: "+".join(sorted(set(s)))).rename("spans").reset_index()
    # test-taker composition (counts only): reading cells, both spans
    rd = c[c.subject == "R"].pivot_table(index=key, columns="group", values="n", aggfunc="sum")
    comp = pd.DataFrame(index=rd.index)
    comp["n_tested"] = rd["all"]
    for g in ("black", "hispanic", "asian", "amind", "multi", "lep", "swd", "frl"):
        comp[f"share_{g}"] = 100 * rd.get(g, np.nan) / rd["all"].where(rd["all"] > 0)
    comp = comp.reset_index()

    s = tabs["SCHOOL_SUMMARY"]
    ssm = pd.DataFrame({
        "lea": _str(s.SSM_LEA_CD), "sch": _str(s.SSM_SCH_CD), "year": _year(s.SSM_YEAR_CD),
        "ayp_pub": _num(s.SSM_AYP_MET_IND), "n_targets": _num(s.SSM_AYP_TARGET_NUM),
        "n_targets_met": _num(s.SSM_AYP_TARGETMET_NUM),
        "si_ind": _num(s.SSM_SCH_IMPROVMNT_IND), "si_years_next": _num(s.SSM_IMPROVMNT_YRS_NUM),
        "si_phase": _str(s.SSM_SCH_IMPV_PHASE_TXT), "si_reason": _num(s.SSM_IMPROVMNT_REASON_CD),
        "pct_frl": 100 * _num(s.SSM_POVERTY_PCT), "enrollment": _num(s.SSM_STUDENT_NUM),
        "pct_licensed": 100 * _num(s.SSM_FLICENSED_TEACH_PCT),
        "pct_teach_0_3": 100 * _num(s.SSM_TCHYRS_0THRU3_PCT),
        "pct_teach_advdeg": 100 * _num(s.SSM_ADVANCE_DGR_PCT),
        "pct_teach_turnover": 100 * _num(s.SSM_1YR_TCHR_TRNOVR_PCT),
        "n_teachers": _num(s.SSM_CLASS_TEACHER_NUM), "n_nbpts": _num(s.SSM_NBPTS_NUM),
        "attendance": 100 * _num(s.SSM_AVG_DAILY_ATTEND_PCT),
        "category": _str(s.SSM_CATEGORY_CD), "locale_raw": _str(s.SSM_NCES_LOCALE_CD),
    }).drop_duplicates(key)
    loc = {"1": "city", "2": "city", "11": "city", "12": "city", "13": "city",
           "3": "suburb", "4": "suburb", "21": "suburb", "22": "suburb", "23": "suburb",
           "5": "town", "6": "town", "31": "town", "32": "town", "33": "town",
           "7": "rural", "8": "rural", "41": "rural", "42": "rural", "43": "rural"}
    ssm["locale"] = ssm.locale_raw.map(loc).fillna("")
    t = tabs["SITE"]
    site = pd.DataFrame({
        "lea": _str(t.STE_UNIT_CD), "sch": _str(t.STE_SITE_CD), "year": _year(t.STE_YEAR_CD),
        "name": _str(t.STE_SITE_NM), "site_type": _str(t.STE_TYPE_CD),
        "school_type": _str(t.STE_SCHOOL_TYPE_TXT), "calendar": _str(t.STE_CALENDAR_ONLY_TXT),
        "title1": _str(t.STE_TITLE1_TYPE_CD), "grade_lo": _num(t.STE_BEGIN_GRADE_NUM),
        "grade_hi": _num(t.STE_END_GRADE_NUM)}).drop_duplicates(key)

    p = ssm.merge(q, on=key, how="outer")
    for df in (nb, b, part, pa, span, comp, site):
        p = p.merge(df, on=key, how="left")
    p["q"] = np.fmin(p.q_read, p.q_math)
    p.loc[p.q_read.isna() | p.q_math.isna(), "q"] = np.nan
    p["school_id"] = p.lea + p.sch
    p["d_rule"] = (p.q < -EPS).astype(float).where(p.q.notna())   # 1 = misses AYP

    # history: published AYP one and two years back; improvement year during t (set by
    # AYP through t-1) and for t+1 (set by AYP through t)
    hist = p[key + ["ayp_pub", "si_years_next"]]
    for k, names in ((1, ("ayp_lag1", "si_years")), (2, ("ayp_lag2", None))):
        h = hist.assign(year=hist.year + k).rename(
            columns={"ayp_pub": names[0], "si_years_next": names[1] or "_drop"})
        p = p.merge(h, on=key, how="left").drop(columns="_drop", errors="ignore")
    p["si_years"] = p.si_years.fillna(0)
    p["si_years_next"] = p.si_years_next.fillna(0)
    # sanction a miss in t triggers in t+1 (Title I schools): next improvement stage when in
    # improvement; choice when the school also missed in t-1; the watch list otherwise
    stage = np.where(p.si_years >= 1, np.minimum(p.si_years + 1, 5),
                     np.where(p.ayp_lag1 == 0, 1, 0))
    p["stage_at_risk"] = [STAGES[int(v)] for v in stage]
    p["title1_known"] = p.title1.eq("Y") | (p.si_years >= 1) | (p.si_years_next >= 1)

    # next year's outcomes
    nxt = p[key + ["prof_read", "prof_math", "ayp_pub", "si_years_next"]].rename(columns={
        "prof_read": "y_read", "prof_math": "y_math", "ayp_pub": "y_ayp",
        "si_years_next": "y_si_years_next"})
    nxt["year"] -= 1
    p = p.merge(nxt, on=key, how="left")
    p["y_year"] = np.where(p.y_math.notna() | p.y_read.notna(), p.year + 1, np.nan)

    abc = _abc_codes()
    if len(abc):
        p = p.merge(abc, on=key, how="left")
    p["abc_code"] = p.get("abc_code", pd.Series("", index=p.index)).fillna("")
    return p.sort_values(key).reset_index(drop=True)


# --------------------------------------------------------------------------- checks

def rule_check(years=AYP_YEARS) -> pd.DataFrame:
    """By year: published AYP against the rule.

    * ``targets_eq``: share of schools whose published number of targets equals
      1 + proficiency cells + participation cells (the OAI is the 1);
    * ``agree_all``: rule (proficiency margins and participation >= 95%) vs published
      AYP, all schools with cells in both subjects;
    * ``agree_part_ok``: the proficiency rule vs published AYP among schools whose every
      participation rate is >= 95% (the loaders' sample);
    * ``made_pub``, ``made_rule``: AYP rates; ``gr``: share with growth-model code (ABCs).
    """
    p = build()
    p = p[p.year.isin(years) & p.q.notna() & p.ayp_pub.notna()]
    rows = []
    for y, g in p.groupby("year"):
        pred_all = ((g.q >= -EPS) & (g.part_misses.fillna(0) == 0)).astype(int)
        ok = g[g.part_misses.fillna(0) == 0]
        tgt = 1 + g.n_cells_read.fillna(0) + g.n_cells_math.fillna(0) + g.n_part_cells.fillna(0)
        rows.append({
            "year": y, "schools": len(g),
            "targets_eq": float((tgt == g.n_targets).mean()),
            "made_pub": float(g.ayp_pub.mean()), "made_rule": float(pred_all.mean()),
            "agree_all": float((pred_all == g.ayp_pub).mean()),
            "part_ok": len(ok),
            "agree_part_ok": float(((ok.q >= -EPS).astype(int) == ok.ayp_pub).mean()),
            "gr": float(g.abc_code.str.contains("GR").mean()) if g.abc_code.ne("").any()
            else np.nan,
        })
    return pd.DataFrame(rows)


# -------------------------------------------------------------------------- samples

def _design(df: pd.DataFrame) -> tuple:
    X = pd.DataFrame(index=df.index)
    cont = {"pct_frl": df.pct_frl, "log_enrollment": np.log(df.enrollment.where(df.enrollment > 0)),
            "pct_licensed": df.pct_licensed, "pct_teach_0_3": df.pct_teach_0_3,
            "pct_teach_advdeg": df.pct_teach_advdeg,
            "pct_teach_turnover": df.pct_teach_turnover,
            "pct_nbpts": 100 * df.n_nbpts / df.n_teachers.where(df.n_teachers > 0)}
    for g in ("black", "hispanic", "asian", "amind", "multi", "lep", "swd"):
        cont[f"share_{g}"] = df[f"share_{g}"]
    for name, v in cont.items():
        v = pd.Series(v, index=df.index).astype(float)
        if name.startswith("pct_"):
            v = v.clip(0, 100)
        X[name] = v.fillna(v.median())
        if v.isna().any():
            X[f"{name}_missing"] = v.isna().astype(float)
    X["charter"] = (df.site_type == "C").astype(float)
    st = df.school_type.str.lower()
    X["magnet"] = st.str.contains("magnet").astype(float)
    X["alternative"] = st.str.contains("alternative").astype(float)
    X["special_school"] = st.str.contains("special|hospital|vocational").astype(float)
    X["year_round"] = df.calendar.str.contains("Year-Round").astype(float)
    X["title1"] = (df.title1 == "Y").astype(float)
    X["title1_unknown"] = (df.year <= 2006).astype(float)    # the flag starts in 2006-07
    X["has_grade10"] = df.spans.str.contains("10").astype(float)
    X["has_grades3_8"] = df.spans.str.contains("3-8").astype(float)
    for cat, name in (("M", "middle"), ("H", "high"), ("I", "elem_middle"),
                      ("T", "middle_high"), ("A", "k12")):
        X[name] = (df.category == cat).astype(float)         # elementary is the base
    for loc in ("suburb", "town", "rural"):
        X[f"locale_{loc}"] = (df.locale == loc).astype(float)  # city is the base
    dummies = {f"year_{y}": (df.year == y).astype(float) for y in sorted(df.year.unique())[1:]}
    # district indicators; charters (LEA codes with a letter) share one category
    lea = np.where(df.site_type == "C", "charter", df.lea)
    base = pd.Series(lea).value_counts().index[0]
    dummies.update({f"lea_{l}": (lea == l).astype(float) for l in sorted(set(lea)) if l != base})
    X = pd.concat([X, pd.DataFrame(dummies, index=df.index)], axis=1)
    X = X.loc[:, X.std() > 0]
    return X.to_numpy(float), list(X.columns)


def _frame(years, y_col: str, sample: str) -> pd.DataFrame:
    p = build()
    m = (p.year.isin(years) & p.q.notna() & p.ayp_pub.notna() & p[y_col].notna()
         & (p.part_misses.fillna(0) == 0))
    if sample not in SAMPLES:
        raise ValueError(f"sample must be one of {list(SAMPLES)}")
    if sample == "in_improvement":
        # Title I schools in School Improvement during t: a miss advances one stage
        m &= p.si_years >= 1
    elif sample == "restructuring":
        # corrective action or later: a miss = restructuring plan or restructuring
        m &= p.si_years >= 3
    elif sample == "entering":
        # Title I (flag known from 2006-07), not in improvement, missed AYP last year:
        # a miss = School Improvement year 1 (choice)
        m &= (p.title1 == "Y") & (p.si_years == 0) & (p.ayp_lag1 == 0)
    elif sample == "first_threat":
        # not in improvement and made AYP last year (Ahn & Vigdor Table 4; Title I status
        # is unknown before 2006-07, so all schools): a miss = watch list
        m &= (p.si_years == 0) & (p.ayp_lag1 == 1)
    return p[m].reset_index(drop=True)


SAMPLES = ("in_improvement", "restructuring", "entering", "first_threat", "all")


_Y_LABEL = {"y_math": "next year's % proficient in math (all students)",
            "y_read": "next year's % proficient in reading (all students)"}


def _extras(df: pd.DataFrame, y_col: str) -> dict:
    return {
        "year": df.year.to_numpy(int), "y_year": df.y_year.to_numpy(int),
        "school_id": df.school_id.to_numpy(str), "lea": df.lea.to_numpy(str),
        "eligible_published": (1 - df.ayp_pub).to_numpy(float),   # 1 = missed AYP (published)
        "ayp_published": df.ayp_pub.to_numpy(float),
        "q_read": df.q_read.to_numpy(float), "q_math": df.q_math.to_numpy(float),
        "binding_group": df.binding_group.to_numpy(str),
        "binding_subject": df.binding_subject.to_numpy(str),
        "stage_at_risk": df.stage_at_risk.to_numpy(str),
        "si_years": df.si_years.to_numpy(float), "si_years_next": df.si_years_next.to_numpy(float),
        "ayp_lag1": df.ayp_lag1.to_numpy(float), "title1": (df.title1 == "Y").to_numpy(float),
        "abc_code": df.abc_code.to_numpy(str),
        "prof_read": df.prof_read.to_numpy(float), "prof_math": df.prof_math.to_numpy(float),
        "y_definition": _Y_LABEL[y_col], "side": "below",
        "treated_side": "missed AYP: Q < 0 (some reading or math cell below its effective "
                        "threshold min(CI, SH))",
    }


def _sample(years, y_col: str, sample: str, two: bool) -> RDDSample:
    df = _frame(years, y_col, sample)
    X, names = _design(df)
    ex = _extras(df, y_col)
    yrs = ", ".join(map(str, years))
    if two:
        Q = df[["q_read", "q_math"]].to_numpy(float)
        thr = (0.0, 0.0)
        rule = lambda Q: ((Q[:, 0] < -EPS) | (Q[:, 1] < -EPS)).astype(int)  # noqa: E731
        ex.update({"cutoffs": thr, "rule": "either", "q_columns": ("q_read", "q_math"),
                   "rule_note": "D = 1{Q1 < 0 or Q2 < 0} (missed AYP); making AYP is the "
                                "'both' rule Q1 >= 0 and Q2 >= 0"})
        qdesc, name = "Q = (reading margin, math margin) in pp", f"nc_ayp_two_score_{sample}_{y_col}"
    else:
        Q, thr = df.q.to_numpy(float), 0.0
        rule = lambda Q: (Q < -EPS).astype(int)  # noqa: E731
        ex.update({"cutoff": 0.0})
        qdesc, name = "Q = min(reading, math margin) in pp", f"nc_ayp_{sample}_{y_col}"
    return RDDSample(
        Q=Q, X=X, Y=df[y_col].to_numpy(float), threshold=thr, name=name,
        feature_names=names,
        description=(f"North Carolina NCLB AYP, spring years {yrs}, sample '{sample}'. "
                     f"{qdesc}, margin = % proficient - min(CI bound, safe-harbor bound) "
                     f"over accountable groups; treatment = missed AYP (Q < 0); "
                     f"Y = {_Y_LABEL[y_col]}."),
        citation=_CITATION, treatment_rule=rule, extras=ex)


def load(years=MAIN_YEARS, y_col: str = "y_math") -> RDDSample:
    """Title I schools in School Improvement. Q = min margin (pp); D = 1{Q < 0} = missed
    AYP, which moves the school up one sanction stage next year (treated below the
    cutoff); Y = next year's % proficient in math. ``years=ALL_YEARS`` for 2004-2010."""
    return _sample(years, y_col, "in_improvement", two=False)


def load_two_score(years=MAIN_YEARS, y_col: str = "y_math") -> RDDSample:
    """As load(), with Q = (reading margin, math margin) and D = 1{Q1 < 0 or Q2 < 0}
    (missed AYP); the untreated region (made AYP) is the 'both' rule."""
    return _sample(years, y_col, "in_improvement", two=True)


def load_reading(years=MAIN_YEARS) -> RDDSample:
    """As load(), Y = next year's % proficient in reading."""
    return _sample(years, "y_read", "in_improvement", two=False)


def load_restructuring(years=ALL_YEARS, y_col: str = "y_math") -> RDDSample:
    """Schools in improvement year 3+ (corrective action or later): a miss means a
    restructuring plan or restructuring (Ahn & Vigdor Table 5). Small: all years."""
    return _sample(years, y_col, "restructuring", two=False)


def load_entering(years=ALL_YEARS, y_col: str = "y_math") -> RDDSample:
    """Title I schools (2006-07 on) that missed AYP last year and are not in improvement:
    a miss starts School Improvement (public school choice)."""
    return _sample(years, y_col, "entering", two=False)


def load_first_threat(years=MAIN_YEARS, y_col: str = "y_math") -> RDDSample:
    """Schools not in improvement that made AYP last year (Ahn & Vigdor Table 4): a miss
    puts them on the watch list."""
    return _sample(years, y_col, "first_threat", two=False)


def load_all(years=MAIN_YEARS, y_col: str = "y_math") -> RDDSample:
    """Every school (Ahn & Vigdor Table 3); the consequence of a miss depends on the
    school's history and Title I status."""
    return _sample(years, y_col, "all", two=False)


def load_all_two_score(years=MAIN_YEARS, y_col: str = "y_math") -> RDDSample:
    return _sample(years, y_col, "all", two=True)


# ------------------------------------------------------------------------ effect check

def local_jump(sample: RDDSample, h: float, col: int | None = None,
               covariates: bool = False, cluster: bool = True,
               y: np.ndarray | None = None) -> dict:
    """Local-linear jump in Y at the cutoff, treated (missed AYP, Q < 0) minus untreated:
    triangular kernel of half-width h (pp), separate slopes, school-clustered standard
    error. For a two-score sample, ``col`` picks one margin (default: the min of both).
    With ``covariates``, sample.X enters linearly (centred in the window)."""
    Q = np.asarray(sample.Q, float)
    q = Q if Q.ndim == 1 else (Q.min(axis=1) if col is None else Q[:, col])
    yv = np.asarray(sample.Y if y is None else y, float)
    w = np.clip(1 - np.abs(q) / h, 0, None)
    k = (w > 0) & np.isfinite(yv)
    t = (q < -EPS).astype(float)
    Z = np.column_stack([np.ones(k.sum()), t[k], q[k], t[k] * q[k]])
    if covariates:
        Xk = sample.X[k]
        Xk = Xk[:, Xk.std(axis=0) > 0]
        Z = np.column_stack([Z, Xk - Xk.mean(axis=0)])
    W = w[k]
    A = Z.T @ (Z * W[:, None])
    beta = np.linalg.lstsq(A, Z.T @ (W * yv[k]), rcond=None)[0]
    e = yv[k] - Z @ beta
    s = Z * (W * e)[:, None]
    n, m = Z.shape
    if cluster:
        g = np.asarray(sample.extras["school_id"])[k]
        s = pd.DataFrame(s).groupby(g).sum().to_numpy()
        adj = len(s) / (len(s) - 1) * (n - 1) / (n - m)
    else:
        adj = n / (n - m)
    Ainv = np.linalg.pinv(A)
    V = Ainv @ (s.T @ s) @ Ainv * adj
    return {"h": h, "n_in_window": int(n), "treated_in_window": int(t[k].sum()),
            "jump": float(beta[1]), "se": float(np.sqrt(V[1, 1])), "covariates": covariates}
