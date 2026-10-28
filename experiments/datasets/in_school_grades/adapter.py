"""Indiana state A-F school grades (IDOE, 511 IAC 6.2-10): fixed-weight sums of domain
points with grade bands at 90 / 80 / 70 / 60.

Under 511 IAC 6.2-10 (effective 2016-03-01, used for the 2015-16 to 2018-19 grades), each
school's overall points are a weighted sum of up to five domain scores, each on a 0-100
scale except growth, which can reach 150 (section 3; IDOE's presentation of the model):

* grades 3-8:  Performance (3-8) 50%, Growth (4-8) 50%;
* grades 9-12: Performance (10) 20%, Growth (10-12) 20%, Multiple Measures (12) 60%
  (graduation 30% and college and career readiness 30%);
* a school serving both spans weights the two sets by its share of enrolled students in
  grades 3-8 and 9-12. A domain without enough students drops out and its weight moves to
  the others (for example 100% Performance (3-8) without growth data; Performance (10) 40%
  and Multiple Measures 60% without growth); a school open three years or less may elect
  to be graded on growth alone (section 10).

The published overall points are round-half-up of the weighted sum to one decimal, and
the letter grade is A >= 90.0, B >= 80.0, C >= 70.0, D >= 60.0, F below (section 3(b)).
The State Board places schools once a year (2016-17 grades on 2017-10-04, 2017-18 grades
on 2018-11-14, both before the next spring's tests). An F counts toward state intervention
(IC 20-31-9; consecutive years in the lowest category) and puts the school's attendance
area in the Choice Scholarship (voucher) "F" public school track for the next school year.

Points are published only for 2016-17 and 2017-18 ("School Domain Scores" sheets). The
2015-16 file has letter grades only. The 2018-19 grades are "hold harmless" grades
(Public Law 2-2020, signed 2020-02-12: the better of the 2017-18 grade and the computed
2018-19 grade), published without points; the 2019-20 grades (no spring 2020 tests) are
hold-harmless grades too. Rating year y below means school year (y-1)-y.

Q = overall points minus (c - 0.05), with c the cutoff of the chosen boundary. The
    published overall is on a 0.1 grid, so D = 1{Q >= 0} is exactly "overall >= c", the
    higher grade. ``load_two_score()``: Q = (Performance 3-8, Growth 4-8) for schools graded
    on those two domains alone, with weights (0.5, 0.5); since the overall is
    round-half-up(0.5 Q1 + 0.5 Q2), "overall >= c" is "0.5 Q1 + 0.5 Q2 >= c - 0.05".
X = school characteristics from the same school year's enrollment files (IDOE's annual
    count, not computed from test scores): % free/reduced-price meals, % Black, Hispanic,
    Asian and multiracial, % English learners, % special education, log enrollment, lowest
    and highest grade enrolled, the corporation's (district's) log enrollment and %
    free/reduced meals, charter, NCES locale (city, suburb, town; rural is the base; from
    the current directory), and the domain set (high school, combined; grades 3-8 is the
    base; other).
Y = ``load()``: the 2017-18 overall points, matched by the four-character IDOE school
    number. The pooled loaders (grades 3-8 schools, rating years 2016-17 and 2017-18): the
    next spring's mean of the E/LA and math pass rates in grades 3-8 (ISTEP+ 2018 for the
    2016-17 grades, ILEARN 2019 for the 2017-18 grades), with a year indicator.

Sample: public schools (traditional and charter) with a published 2016-17 letter grade
and overall points, and Y observed. Non-public schools are graded too but have no
demographic covariates in the enrollment files; they stay in the cache with
``nonpublic = 1``.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample
# Standard-library .xlsx reader (the environment has no openpyxl), shared with Florida.
from experiments.datasets.fl_school_grades._spreadsheet import read, sheet_names
from experiments.datasets.in_school_grades import download as dl

HERE = Path(__file__).parent
RAW = dl.RAW
PROCESSED = HERE / "data" / "processed"
CACHE = PROCESSED / "in_school_grades_panel.csv.gz"

POINT_YEARS = (2017, 2018)              # rating years with published domain points
MAIN_YEAR = 2017                        # the only year with next-year overall points
POOLED_YEARS = (2017, 2018)
CUTOFFS = (90.0, 80.0, 70.0, 60.0)      # A, B, C, D minimum overall points (6.2-10-3(b))
BOUNDARIES = {"AB": 0, "BC": 1, "CD": 2, "DF": 3}
MAIN_BOUNDARY = "DF"                    # the boundary with consequences (F)
WEIGHTS = (0.5, 0.5)                    # Performance (3-8), Growth (4-8)
HALF = 0.05                             # half the 0.1 reporting grid
EPS = 1e-9
GRADE_POINTS = {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0, "F": 0.0}
DOMAINS = ["perf38", "gr48", "perf10", "gr1012", "mm12"]
_DOMAIN_COLS = ["corp", "corp_name", "school", "name",
                "perf38", "w_perf38", "gr48", "w_gr48", "perf10", "w_perf10",
                "gr1012", "w_gr1012", "mm12", "w_mm12", "overall", "grade"]
_TXT = ["corp", "corp_name", "school", "name", "grade", "rule_type", "locale", "county",
        "corp_type", "grade_prev", "grade_2016", "grade_2017", "grade_2018", "grade_2019",
        "grade_2020", "grade_final", "y_grade"]


# ------------------------------------------------------------------------ raw files

def _need(name: str) -> Path:
    path = RAW / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run python -m experiments.datasets.in_school_grades.download")
    return path


def _sheet(name: str, sheet: str | int) -> list:
    path = _need(name)
    i = sheet if isinstance(sheet, int) else sheet_names(path).index(sheet)
    return read(path, i)


def _id(s: pd.Series) -> pd.Series:
    """Four-character IDOE number; some sheets store public numbers as floats."""
    s = s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    return s.where(~s.str.fullmatch(r"\d{1,3}"), s.str.zfill(4))


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _frame(rows: list, header_row: int = 0, columns: list | None = None) -> pd.DataFrame:
    hdr = columns or [" ".join(str(c).split()) if c is not None else f"col{j}"
                      for j, c in enumerate(rows[header_row])]
    body = [list(r[:len(hdr)]) + [None] * (len(hdr) - len(r)) for r in rows[header_row + 1:]]
    return pd.DataFrame(body, columns=hdr)


def _domain(year: int) -> pd.DataFrame:
    """The "School Domain Scores" sheet: points and weight of each domain, overall points
    and grade (the column layout is the same in both years)."""
    name = dl.GRADES_2017 if year == 2017 else dl.GRADES_2018
    df = _frame(_sheet(name, "School Domain Scores"), columns=_DOMAIN_COLS)
    df["school"] = _id(df.school)
    df["corp"] = _id(df.corp)
    for c in _DOMAIN_COLS[4:15]:
        df[c] = _num(df[c])
    df["grade"] = df.grade.fillna("").astype(str).str.strip()
    df["year"] = year
    return df[df.school != "None"].reset_index(drop=True)


def _letters() -> pd.DataFrame:
    """Published letter grades by school, 2015-16 to 2019-20 (2018-19 and 2019-20 are the
    hold-harmless grades)."""
    def one(name, sheet, col, out, ncol):
        rows = _sheet(name, sheet)
        df = _frame(rows, columns=[f"c{j}" for j in range(len(rows[0]))])
        return pd.DataFrame({"school": _id(df[f"c{ncol}"]),
                             out: df[f"c{col}"].fillna("").astype(str).str.strip()})

    parts = [
        pd.concat([one(dl.GRADES_2016, "Public School", 4, "grade_2016", 2),
                   one(dl.GRADES_2016, "Nonpublic", 4, "grade_2016", 2)]),
        one(dl.GRADES_2017, "School Grades", 4, "grade_2017", 2),
        one(dl.GRADES_2018, "School Grades", 4, "grade_2018", 2),
        one(dl.GRADES_2019, "School Grades", 4, "grade_2019", 2),
        one(dl.GRADES_2020, "K-12 Grades", 4, "grade_2020", 2),
    ]
    out = parts[0].drop_duplicates("school")
    for p in parts[1:]:
        out = out.merge(p.drop_duplicates("school"), on="school", how="outer")
    return out


def _enrollment(year: int) -> pd.DataFrame:
    """School enrollment, demographics and grade span for the school year ending in
    ``year`` (sheet ``str(year)``, IDOE's annual count). Blank cells are zero counts."""
    eth = _frame(_sheet(dl.ENR_ETHNIC, str(year)))
    low = {c.lower(): c for c in eth.columns}

    def col(df, key):
        return _num(df[low[key]]).fillna(0.0)

    total_col = next(c for c in eth.columns if c.lower() in ("grand total", "total enrollment"))
    total = _num(eth[total_col])
    out = pd.DataFrame({"school": _id(eth["Schl ID"]), "corp": _id(eth["Corp ID"]),
                        "enrollment": total})
    for key, name in [("black", "black"), ("hispanic", "hispanic"), ("asian", "asian"),
                      ("multiracial", "multiracial"), ("white", "white")]:
        out[f"n_{name}"] = col(eth, key)
    out["n_frl"] = col(eth, "free/reduced price meals")
    sp = _frame(_sheet(dl.ENR_SPECIAL, str(year)))
    sp = pd.DataFrame({"school": _id(sp["Schl ID"]), "n_ell": _num(sp["ELL N"]).fillna(0.0),
                       "n_sped": _num(sp["Special Education N"]).fillna(0.0)})
    gr = _frame(_sheet(dl.ENR_GRADE, str(year)))
    levels = [("Pre-K", -1), ("Kindergarten", 0)] + [(f"Grade {k}", k) for k in range(1, 13)]
    g = pd.DataFrame({"school": _id(gr["Schl ID"])})
    counts = np.column_stack([_num(gr[c]).fillna(0.0) for c, _ in levels])
    adult = _num(gr["Grade 12+/Adult"]).fillna(0.0).to_numpy()
    vals = np.array([v for _, v in levels], float)
    has = counts > 0
    g["low_grade"] = np.where(has.any(1), vals[np.argmax(has, 1)], np.nan)
    g["high_grade"] = np.where(has.any(1), vals[len(vals) - 1 - np.argmax(has[:, ::-1], 1)],
                               np.nan)
    g.loc[(adult > 0) & g.high_grade.isna(), "high_grade"] = 12.0
    e38, e912 = counts[:, 4:10].sum(1), counts[:, 10:14].sum(1)
    both = e38 + e912
    g["share38"] = np.where(both > 0, e38 / np.where(both > 0, both, 1), np.nan)
    out = out.merge(sp, on="school", how="left").merge(g, on="school", how="left")
    # Corporation context from the same file (public schools only).
    corp = out.groupby("corp")[["enrollment", "n_frl"]].sum()
    out["corp_enrollment"] = out.corp.map(corp.enrollment)
    out["corp_frl"] = out.corp.map(corp.n_frl / corp.enrollment.where(corp.enrollment > 0))
    return out.drop(columns="corp").drop_duplicates("school")


def _directory() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Locale and county by school, corporation type by corporation, from the 2026-27
    directory (only these columns are read)."""
    sch = _frame(_sheet(dl.DIRECTORY, "SCHL"))
    sch = pd.DataFrame({"school": _id(sch.IDOE_SCHOOL_ID),
                        "locale": sch.LOCALE.fillna("").astype(str),
                        "county": sch.COUNTY_NAME.fillna("").astype(str)})
    corp = _frame(_sheet(dl.DIRECTORY, "CORP"))
    corp = pd.DataFrame({"corp": _id(corp.IDOE_CORPORATION_ID),
                         "corp_type": corp.CORPORATION_TYPE.fillna("").astype(str),
                         "corp_county": corp.COUNTY_NAME.fillna("").astype(str)})
    return sch.drop_duplicates("school"), corp.drop_duplicates("corp")


def _pass_rates() -> pd.DataFrame:
    """Grades 3-8 school totals: mean of the E/LA and math pass (proficient) rates, in
    percent. ISTEP+ spring 2018 and ILEARN spring 2019; suppressed cells (fewer than 10
    students) are missing."""
    rows = _sheet(dl.ISTEP_2018, 0)
    df = _frame(rows, header_row=1)
    tot = df.iloc[:, -9:]
    ist = pd.DataFrame({"school": _id(df.iloc[:, 2]),
                        "ela": _num(tot.iloc[:, 2]), "math": _num(tot.iloc[:, 5]),
                        "tested": _num(tot.iloc[:, 1])})
    ist["pass_rate"] = 100 * (ist.ela + ist.math) / 2
    ist["test_year"] = 2018
    out = [ist]
    parts = {}
    for subj in ("ELA", "Math"):
        rows = _sheet(dl.ILEARN_2019, subj)
        h = next(k for k, r in enumerate(rows) if r and r[0] == "Corp ID")
        df = _frame(rows, header_row=h)
        parts[subj] = pd.DataFrame({"school": _id(df.iloc[:, 2]), subj: _num(df.iloc[:, -1]),
                                    f"{subj}_tested": _num(df.iloc[:, -2])})
    il = parts["ELA"].merge(parts["Math"], on="school", how="outer")
    out.append(pd.DataFrame({"school": il.school, "ela": il.ELA, "math": il.Math,
                             "tested": il.ELA_tested,
                             "pass_rate": 100 * (il.ELA + il.Math) / 2, "test_year": 2019}))
    out = pd.concat(out, ignore_index=True)
    out = out[out.school.str.fullmatch(r"[0-9A-Z]{4}")]
    return out.drop_duplicates(["school", "test_year"])


# ---------------------------------------------------------------------------- rule

def _round1(x) -> np.ndarray:
    """Round half up to one decimal (the 1e-9 absorbs binary error at .x5)."""
    return np.floor(np.asarray(x, float) * 10 + 0.5 + 1e-9) / 10


def _rule_type(w: pd.DataFrame) -> pd.Series:
    """Which domains carry weight: ems (3-8: 0.5/0.5), hs (0.2/0.2/0.6), combined (both
    spans), growth_only, mm_only, hs_partial (high school with a domain missing),
    ems_partial (3-8 with growth missing), none (no points)."""
    w = w.fillna(0.0)
    ems = (w.w_perf38 > 0) | (w.w_gr48 > 0)
    hs = (w.w_perf10 > 0) | (w.w_gr1012 > 0) | (w.w_mm12 > 0)
    tot = w.sum(axis=1)
    return pd.Series(np.select(
        [tot == 0,
         (w.w_perf38 == 0.5) & (w.w_gr48 == 0.5) & ~hs,
         (w.w_perf10 == 0.2) & (w.w_gr1012 == 0.2) & (w.w_mm12 == 0.6) & ~ems,
         (w.w_gr48 == 1) & (tot == 1),
         (w.w_mm12 == 1) & (tot == 1),
         ems & hs, hs & ~ems, ems & ~hs],
        ["none", "ems", "hs", "growth_only", "mm_only", "combined", "hs_partial",
         "ems_partial"], "other"), index=w.index)


def state_rule(df: pd.DataFrame) -> pd.DataFrame:
    """The overall points from the published domain points and weights: x (the weighted
    sum), overall_computed (x rounded half up to 0.1) and the computed grade."""
    out = pd.DataFrame(index=df.index)
    w = df[[f"w_{d}" for d in DOMAINS]].fillna(0.0).to_numpy()
    p = df[DOMAINS].fillna(0.0).to_numpy()
    out["x"] = (w * p).sum(1)
    out["weight_sum"] = w.sum(1)
    out["overall_computed"] = _round1(out.x)
    out["grade_computed"] = _grade(out.overall_computed)
    return out


def _grade(score) -> np.ndarray:
    s = np.asarray(score, float)
    return np.select([s >= 90, s >= 80, s >= 70, s >= 60, s >= 0], list("ABCDF"), "")


# --------------------------------------------------------------------------- panel

def build() -> pd.DataFrame:
    """Every school with a row in the 2016-17 or 2017-18 domain sheet: published points
    and grade, the computed rule, covariates, the published grades of 2015-16 to 2019-20
    and the next-year outcomes. Cached in ``data/processed`` (atomic write)."""
    if not CACHE.exists():
        df = _build()
        PROCESSED.mkdir(parents=True, exist_ok=True)
        tmp = CACHE.with_name(f"{CACHE.name}.{os.getpid()}.part")
        df.to_csv(tmp, index=False, compression="gzip")
        os.replace(tmp, CACHE)
    p = pd.read_csv(CACHE, keep_default_na=False, na_values=[""], low_memory=False,
                    dtype={c: str for c in _TXT})
    for c in _TXT:
        p[c] = p[c].fillna("").astype(str)
    return p


def _build() -> pd.DataFrame:
    letters = _letters()
    sch_dir, corp_dir = _directory()
    tests = _pass_rates()
    frames = []
    for y in POINT_YEARS:
        d = _domain(y)
        d = pd.concat([d, state_rule(d)], axis=1)
        d["rule_type"] = _rule_type(d[[f"w_{k}" for k in DOMAINS]])
        d["graded"] = d.grade.isin(list("ABCDF")).astype(float)
        # "No Grade" rows carry overall = 0.0 in 2017-18; their points are not a grade.
        d.loc[d.graded == 0, "overall"] = np.nan
        d["nonpublic"] = (~d.school.str.fullmatch(r"\d{4}")).astype(float)
        d = d.merge(_enrollment(y), on="school", how="left")
        frames.append(d)
    p = pd.concat(frames, ignore_index=True)
    p = p.merge(letters, on="school", how="left")
    for c in ["grade_2016", "grade_2017", "grade_2018", "grade_2019", "grade_2020"]:
        p[c] = p[c].fillna("")
    p["grade_prev"] = np.where(p.year == 2017, p.grade_2016, p.grade_2017)
    # The domain sheet holds the State Board's placement (2017-10-04, 2018-11-14); the
    # "School Grades" sheet of the same file holds the grade after appeals.
    p["grade_final"] = np.where(p.year == 2017, p.grade_2017, p.grade_2018)
    p["appeal_changed"] = (p.grade_final.isin(list("ABCDF")) & p.grade.isin(list("ABCDF"))
                           & (p.grade_final != p.grade)).astype(float)
    p = p.merge(sch_dir, on="school", how="left").merge(corp_dir, on="corp", how="left")
    for c in ["locale", "county", "corp_type", "corp_county"]:
        p[c] = p[c].fillna("")
    p["county"] = p.county.where(p.county != "", p.corp_county)
    cid = pd.to_numeric(p.corp, errors="coerce")
    # Charter: the corporation is a charter in the current directory, or no longer exists
    # and has a charter-range number (9100+, outside the state, ESC and university blocks).
    gone_charter = (p.corp_type == "") & (cid >= 9100) & ~cid.between(9500, 9625)
    p["charter"] = ((p.corp_type == "Charter") | gone_charter).astype(float)
    # Turnaround school operators (state takeover after six Fs): 8810-8825.
    p["turnaround_operator"] = cid.between(8800, 8830).astype(float)

    # Next year's outcomes.
    nxt = p[p.year == 2018][["school", "overall", "grade", "perf38", "gr48", "x"]].rename(
        columns={"overall": "y_overall", "grade": "y_grade_sheet", "perf38": "y_perf38",
                 "gr48": "y_gr48", "x": "y_x"})
    nxt["year"] = 2017
    p = p.merge(nxt, on=["school", "year"], how="left")
    p["y_grade"] = np.where(p.year == 2017, p.grade_2018, p.grade_2019)
    p["y_grade_points"] = p.y_grade.map(GRADE_POINTS)
    p["y_closed"] = np.where(p.year == 2017, p.y_grade_sheet.isna().astype(float), np.nan)
    p = p.drop(columns="y_grade_sheet")
    t = tests.rename(columns={"pass_rate": "y_pass_rate", "ela": "y_ela", "math": "y_math",
                              "tested": "y_tested"})
    t["year"] = t.test_year - 1
    p = p.merge(t.drop(columns="test_year"), on=["school", "year"], how="left")
    return p.sort_values(["year", "school"]).reset_index(drop=True)


# -------------------------------------------------------------------------- checks

def rule_check() -> pd.DataFrame:
    """Per rating year and domain set: does the weighted sum of the published domain
    points, rounded half up, reproduce the published overall points, and does the
    published grade equal the band of the published overall points?"""
    p = build()
    rows = []
    for (y, t), g in p[p.graded == 1].groupby(["year", "rule_type"]):
        rows.append({
            "year": y, "rule_type": t, "graded": len(g),
            "public": int((g.nonpublic == 0).sum()),
            "overall_reproduced": float((g.overall_computed == g.overall).mean()),
            "within_0.1": float(((g.overall_computed - g.overall).abs() <= 0.1 + EPS).mean()),
            "grade_from_published_overall": float((_grade(g.overall) == g.grade).mean()),
            "grade_from_computed": float((g.grade_computed == g.grade).mean()),
            "changed_on_appeal": int(g.appeal_changed.sum()),
        })
    return pd.DataFrame(rows)


def hold_harmless_check() -> pd.DataFrame:
    """2018-19 hold-harmless grade against the 2017-18 grade (no 2018-19 points exist):
    the share of schools whose 2018-19 grade is at least the 2017-18 grade."""
    p = build()
    g = p[(p.year == 2018) & p.grade.isin(list("ABCDF")) & p.grade_2019.isin(list("ABCDF"))]
    up = g.grade_2019.map(GRADE_POINTS) >= g.grade.map(GRADE_POINTS)
    return pd.DataFrame([{"schools": len(g), "not_lower": float(up.mean()),
                          "same": float((g.grade_2019 == g.grade).mean()),
                          "higher": float((g.grade_2019.map(GRADE_POINTS)
                                           > g.grade.map(GRADE_POINTS)).mean())}])


def boundary_mass(width: float = 2.0, years=(MAIN_YEAR,)) -> dict:
    """Graded public schools with overall points within ``width`` of each cutoff."""
    p = build()
    m = (p.graded == 1) & (p.nonpublic == 0) & p.year.isin(years)
    return {b: int(((p.overall[m] - CUTOFFS[j]).abs() < width).sum())
            for b, j in BOUNDARIES.items()}


def bunching(width: float = 1.0, years=POINT_YEARS) -> pd.DataFrame:
    """Graded public schools with overall points in [c - width, c) and [c, c + width)."""
    p = build()
    m = (p.graded == 1) & (p.nonpublic == 0) & p.year.isin(years)
    o = p.overall[m]
    return pd.DataFrame([{"cutoff": c, "below": int(((o >= c - width) & (o < c)).sum()),
                          "above": int(((o >= c) & (o < c + width)).sum())} for c in CUTOFFS])


# ------------------------------------------------------------------------- samples

def _design(df: pd.DataFrame) -> tuple[np.ndarray, list]:
    X = pd.DataFrame(index=df.index)

    def fill(name, v):
        v = v.astype(float)
        X[name] = v.fillna(v.median())
        if v.isna().any():
            X[f"{name}_missing"] = v.isna().astype(float)

    enr = df.enrollment.where(df.enrollment > 0)
    for c in ["frl", "black", "hispanic", "asian", "multiracial", "ell", "sped"]:
        fill(f"pct_{c}", 100 * df[f"n_{c}"] / enr)
    fill("log_enrollment", np.log(enr))
    fill("low_grade", df.low_grade)
    fill("high_grade", df.high_grade)
    fill("corp_log_enrollment", np.log(df.corp_enrollment.where(df.corp_enrollment > 0)))
    fill("corp_pct_frl", 100 * df.corp_frl)
    X["charter"] = df.charter.astype(float)
    loc = df.locale.str.split(":").str[0].str.strip().replace({"Suburban": "Suburb"})
    for k in ["City", "Suburb", "Town"]:
        X[f"locale_{k.lower()}"] = (loc == k).astype(float)
    X["locale_missing"] = (~loc.isin(["City", "Suburb", "Town", "Rural"])).astype(float)
    X["high_school"] = df.rule_type.isin(["hs", "hs_partial"]).astype(float)
    X["combined"] = (df.rule_type == "combined").astype(float)
    X["other_domains"] = df.rule_type.isin(["growth_only", "mm_only", "ems_partial",
                                            "other"]).astype(float)
    for y in sorted(df.year.unique())[1:]:
        X[f"year_{y}"] = (df.year == y).astype(float)
    X = X.loc[:, X.std() > 0]
    return X.to_numpy(float), list(X.columns)


def _rows(years, y_col: str, ems_only: bool) -> pd.DataFrame:
    p = build()
    m = (p.year.isin(years) & (p.graded == 1) & (p.nonpublic == 0) & p.overall.notna()
         & p[y_col].notna())
    if ems_only:
        m &= p.rule_type == "ems"
    return p[m].reset_index(drop=True)


def _extras(df: pd.DataFrame, j: int, cut: float) -> dict:
    higher = set("ABCD"[:j + 1])
    group = (df.year.astype(str) + "-" + df.rule_type).to_numpy()
    return {
        "group": group, "group_cutoffs": {g: cut for g in sorted(set(group))},
        "year": df.year.to_numpy(int), "school": df.school.to_numpy(str),
        "corp": df.corp.to_numpy(str), "county": df.county.to_numpy(str),
        "rule_type": df.rule_type.to_numpy(str),
        **{d: df[d].to_numpy(float) for d in DOMAINS},
        "domain_weights": df[[f"w_{d}" for d in DOMAINS]].fillna(0.0).to_numpy(float),
        "x": df.x.to_numpy(float), "overall_published": df.overall.to_numpy(float),
        "grade_published": df.grade.to_numpy(str),
        "grade_final": df.grade_final.to_numpy(str),
        "eligible_published": df.grade_final.isin(higher).to_numpy(float),
        "grade_prev": df.grade_prev.to_numpy(str),
        "grade_2019_hold_harmless": df.grade_2019.to_numpy(str),
        "y_grade": df.y_grade.to_numpy(str), "y_overall": df.y_overall.to_numpy(float),
        "y_pass_rate": df.y_pass_rate.to_numpy(float),
        "turnaround_operator": df.turnaround_operator.to_numpy(float),
        "cutoff_table": dict(zip("ABCD", CUTOFFS)),
    }


def _single(boundary: str, years, y_col: str, ems_only: bool, tag: str) -> RDDSample:
    j = BOUNDARIES[boundary]
    c = CUTOFFS[j]
    df = _rows(years, y_col, ems_only)
    X, names = _design(df)
    hi, lo = "ABCD"[j], "BCDF"[j]
    q = (df.overall - (c - HALF)).to_numpy(float)
    ex = _extras(df, j, c - HALF)
    ex["boundary"] = boundary
    return RDDSample(
        Q=q, X=X, Y=df[y_col].to_numpy(float), threshold=0.0,
        name=f"in_school_grades{tag}_{boundary}_{y_col}", feature_names=names,
        description=(f"Indiana state A-F school grades, rating year(s) "
                     f"{', '.join(f'{y - 1}-{y % 100:02d}' for y in years)}. Q = overall points "
                     f"- ({c} - 0.05); treatment = 1{{Q >= 0}} ({hi} or better rather than "
                     f"{lo}); Y = {y_col} (next year)."),
        citation="Indiana Department of Education, state A-F school grade results "
                 "2016-17 and 2017-18 (School Domain Scores); 511 IAC 6.2-10",
        treatment_rule=lambda Q: (Q >= 0).astype(int),
        extras=ex,
    )


def _two(boundary: str, years, y_col: str, tag: str) -> RDDSample:
    j = BOUNDARIES[boundary]
    cut = CUTOFFS[j] - HALF
    df = _rows(years, y_col, ems_only=True)
    X, names = _design(df)
    w = np.array(WEIGHTS)
    ex = _extras(df, j, cut)
    ex.update({"weights": WEIGHTS, "cutoff": cut, "boundary": boundary})
    return RDDSample(
        Q=df[["perf38", "gr48"]].to_numpy(float), X=X, Y=df[y_col].to_numpy(float),
        threshold=(cut, cut), name=f"in_school_grades_two_score{tag}_{boundary}",
        feature_names=names,
        description=("Indiana state A-F school grades, grades 3-8 schools. Q = (Performance "
                     f"3-8, Growth 4-8); treatment = 1{{0.5 Q1 + 0.5 Q2 >= {cut:.2f}}}; "
                     f"Y = {y_col} (next year)."),
        citation="Indiana Department of Education, state A-F school grade results "
                 "2016-17 and 2017-18 (School Domain Scores); 511 IAC 6.2-10",
        # 0.5 Q1 + 0.5 Q2 is on a 0.05 grid and can sit exactly on the cutoff.
        treatment_rule=lambda Q: (Q @ w >= cut - EPS).astype(int),
        extras=ex,
    )


def load() -> RDDSample:
    """D/F boundary (60): D = 1 for a D or better, 0 for an F; Y = 2017-18 overall points."""
    return _single(MAIN_BOUNDARY, (MAIN_YEAR,), "y_overall", False, "")


def load_ab() -> RDDSample:
    return _single("AB", (MAIN_YEAR,), "y_overall", False, "")


def load_bc() -> RDDSample:
    return _single("BC", (MAIN_YEAR,), "y_overall", False, "")


def load_cd() -> RDDSample:
    return _single("CD", (MAIN_YEAR,), "y_overall", False, "")


def load_df() -> RDDSample:
    return _single("DF", (MAIN_YEAR,), "y_overall", False, "")


def load_two_score(boundary: str = MAIN_BOUNDARY) -> RDDSample:
    """Grades 3-8 schools: Q = (Performance 3-8, Growth 4-8), weights (0.5, 0.5),
    treatment = 1{0.5 Q1 + 0.5 Q2 >= c - 0.05}; Y = 2017-18 overall points."""
    return _two(boundary, (MAIN_YEAR,), "y_overall", "")


def load_pooled(boundary: str = MAIN_BOUNDARY) -> RDDSample:
    """Grades 3-8 schools, rating years 2016-17 and 2017-18 pooled; Q = overall points -
    (c - 0.05); Y = next spring's mean E/LA and math pass rate (ISTEP+ 2018 / ILEARN 2019)."""
    return _single(boundary, POOLED_YEARS, "y_pass_rate", True, "_pooled")


def load_two_score_pooled(boundary: str = MAIN_BOUNDARY) -> RDDSample:
    """As ``load_pooled`` with Q = (Performance 3-8, Growth 4-8), weights (0.5, 0.5)."""
    return _two(boundary, POOLED_YEARS, "y_pass_rate", "_pooled")
