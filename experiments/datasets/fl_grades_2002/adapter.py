"""Florida A+ school grades, 2002-2011, at the D/F boundary (FLDOE grade points + NCES CCD).

From 2001-02 every graded Florida public school earns one point per percent of eligible
students meeting a criterion, and the letter grade comes from the total (FLDOE technical
assistance papers ``SchoolGradesCalcTAP03.pdf`` to ``TAP13.pdf``, the 2001-02 guide sheet
and ``SchoolGradesCalcGuide14.pdf``):

* status components: percent at FCAT level 3 or above in reading and in math, and the
  writing measure (mean of percent >= 3.0 and percent >= 3.5); science from 2006-07;
* growth components: percent making learning gains in reading and in math, and the
  percent of the lowest 25% making learning gains in reading; in math too from 2006-07;
* from 2006-07 a 10-point retake bonus for some high schools (half of the grade 11-12
  retakers pass).

Spring 2002-2006: 6 components (600 points), A >= 410, B >= 380, C >= 320, D >= 280.
Spring 2007-2011: 8 components (800 points), A >= 525, B >= 495, C >= 435, D >= 395.
Extra conditions (A needs 95% tested and adequate progress of the lowest 25%; B and C
lose a letter without adequate progress in the current or previous year; schools testing
under 90% get an "I" or a lower grade) never move a school across the D/F line, except
the last one. So the deployed D/F rule is

    F  <=>  points < D minimum (280 or 395),   or a D on points with under 90% tested.

From 2009-10 schools with grade 12 are graded on 1,600 points (half from graduation and
acceleration components); from 2011-12 middle schools on 900 points and 6-12/K-12 schools
on 1,700 points, grades fall at most one letter a year (points are topped up), and from
2012-13 a school with fewer than 25% of students reading at grade level drops a letter
(D to F included). Those school-years are in ``build()`` and ``rule_check()`` but not in
the loaders.

Q = load(): grade points minus the year's D minimum (280 or 395), integer valued.
    D = 1{Q < 0}: the school is graded F, the treated side (F carried the consequences:
    Opportunity Scholarship vouchers through 2005, state intervention, pressure). Note
    the sign: treatment is BELOW the cutoff.
    load_two_score(): Q = (status points + retake bonus, growth points), each centred at
    half of the year's D minimum, so D = 1{Q1 + Q2 < 0} with weights (1, 1).
X = NCES CCD for the same school year: % free or reduced-price lunch, % non-white, log
    membership, pupil/teacher ratio, Title I, magnet, charter, locale (city, suburb,
    town, rural); FLDOE school type; year and district indicators.
Y = load(): the next year's percent of possible points, i.e. the mean of all its status
    and growth components (0-100, comparable across the 600- and 800-point years),
    matched by district and school number. Other loaders: next year's mean reading and
    math proficiency, next status and growth means, next grade (A=4 ... F=0), not graded
    next year.

Sample: school-years in spring 2002-2011 graded A-F on the FCAT-only points scale (600
or 800 points), with points and a next-year outcome. Excluded: 1,600-point high schools
in 2010-2011 (a third, non-FCAT score), grades I, P (pending) and N (new school).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample
from experiments.datasets.fl_grades_2002._ccd import read_ccd
from experiments.datasets.fl_grades_2002._grades import read_year
from experiments.datasets.fl_grades_2002.download import CCD, SPREADSHEETS

HERE = Path(__file__).parent
PROCESSED = HERE / "data" / "processed"
CACHE = PROCESSED / "fl_grades_2002_panel.csv.gz"

# Minimum total points for A, B, C, D by points scale (F below D). Sources: TAP03-TAP09
# Table 1 (600 and 800), TAP10 Table 7 (1,600), TAP12 Tables 6-9 (900, 1,700), Guide14
# Table 8 (2014 type-03 high schools).
SCALES = {
    600: (410, 380, 320, 280),
    800: (525, 495, 435, 395),
    900: (590, 560, 490, 445),
    1600: (1050, 990, 870, 790),
    1700: (1115, 1050, 925, 840),
}
HIGH_2014 = {1600: (1120, 1040, 880, 800), 800: (560, 520, 440, 400)}
YEARS = tuple(range(2002, 2015))
MAIN_YEARS = tuple(range(2002, 2012))
STATUS = ["read_prof", "math_prof", "write_prof", "sci_prof"]
GROWTH = ["read_gain", "math_gain", "read_l25", "math_l25"]
GRADES = "ABCDF"
GRADE_POINTS = {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0, "F": 0.0}
TYPES = {1: "elementary", 2: "middle", 3: "high", 4: "combination"}
_TXT = ["file", "name", "grade", "grade_prev", "grade_prev2", "weighting", "ap_read_cur",
        "ap_read_prev", "ap_read_impr", "ap_math_cur", "ap_math_prev", "ap_math_impr",
        "read25_met", "protected", "ms_accel", "ap_at_risk", "charter", "title1", "alt_ese",
        "ccd_type", "ccd_level", "ccd_title1", "ccd_magnet", "ccd_charter", "ccd_locale",
        "ncessch", "grade_computed", "grade_band", "y_grade", "reduction"]


# -------------------------------------------------------------------------- the rule

def d_minimum(year: int) -> int:
    """Minimum points for a D on the FCAT-only scale (600 points to 2006, 800 after)."""
    return SCALES[600 if year <= 2006 else 800][3]


def _scale(p: pd.DataFrame) -> np.ndarray:
    """Points basis of each row's grading scale."""
    y = p.year.to_numpy()
    sc = np.where(y <= 2006, 600.0, 800.0)
    high16 = (p.file == "high") & p.points_total.notna() & (p.year.between(2010, 2011))
    sc = np.where(high16, 1600.0, sc)
    s12 = np.where(p.file == "high", p.scale, np.where(p.ms_accel == "Y", 900.0, 800.0))
    sc = np.where(y == 2012, s12, sc)
    sc = np.where(y >= 2013, p.scale, sc)
    return sc


def _cutoffs(p: pd.DataFrame) -> np.ndarray:
    out = np.full((len(p), 4), np.nan)
    for k, (sc, y, t) in enumerate(zip(p.scale_used, p.year, p.school_type)):
        if sc in SCALES:
            out[k] = HIGH_2014[sc] if (y == 2014 and t == 3 and sc in HIGH_2014
                                       and p.file.iat[k] == "high") else SCALES[sc]
    return out


def _band(points: np.ndarray, cut: np.ndarray) -> np.ndarray:
    g = np.select([points >= cut[:, k] for k in range(4)], list("ABCD"), "F").astype(object)
    g[np.isnan(points) | np.isnan(cut[:, 3])] = ""
    return g


def _lower(g: str) -> str:
    return GRADES[min(GRADES.index(g) + 1, 4)] if g in tuple(GRADES) else g


def _met(*flags: str) -> bool:
    return any(f == "Y" for f in flags)


def _grade_row(r) -> tuple[str, str]:
    """Grade from points and the year's extra conditions; returns (grade, reason)."""
    g = r.grade_band
    if g == "":
        return "", ""
    y = r.year
    reason = ""
    # Adequate progress of the lowest 25% (none in 2012). A: current year; B, C: current
    # or previous year (2002: no previous year, so B and C are not reduced).
    if y != 2012 and g in "ABC":
        if y <= 2006:
            cur = r.read_l25 >= 50
            prev = (r.read_l25_prev >= 50) if y >= 2003 else True
            ok_a, ok_bc = cur, cur or prev
        else:
            rc = _met(r.ap_read_cur, r.ap_read_impr)
            mc = _met(r.ap_math_cur, r.ap_math_impr)
            ok_a = rc and mc
            ok_bc = _met(r.ap_read_cur, r.ap_read_impr, r.ap_read_prev) and _met(
                r.ap_math_cur, r.ap_math_impr, r.ap_math_prev)
        if (g == "A" and not ok_a) or (g in "BC" and not ok_bc):
            g, reason = _lower(g), "adequate_progress"
    # 2002-2003: an A also needs lowest-25% reading gains within 10 points of all gains.
    if not reason and g == "A" and y <= 2003 and r.read_l25 < r.read_gain - 10:
        g, reason = "B", "similar_gains"
    # 2010-2014 high-school scales: an A also needs the at-risk graduation-rate target.
    if not reason and g == "A" and r.scale_used >= 1600 and r.ap_at_risk == "N":
        g, reason = "B", "at_risk_graduation"
    # 2013-2014 reading performance requirement (can move a D to an F).
    if not reason and y >= 2013 and r.read25_met == "N":
        g, reason = _lower(g), "reading_25pct"
    # Participation: an A needs 95% tested; under 90% the grade is lowered (or "I").
    if not reason and g == "A" and r.pct_tested < 95:
        g, reason = "B", "tested_under_95"
    if not reason and r.pct_tested < 90:
        g, reason = _lower(g), "tested_under_90"
    # 2012-2014: no grade more than one letter below the previous year's.
    if y >= 2012 and r.grade_prev in tuple(GRADES):
        floor = GRADES[min(GRADES.index(r.grade_prev) + 1, 4)]
        if GRADES.index(g) > GRADES.index(floor):
            g, reason = floor, "drop_protection"
    return g, reason


# --------------------------------------------------------------------------- panel

def build() -> pd.DataFrame:
    """Every school-year in the 2001-02 to 2013-14 spreadsheets with the computed grade,
    CCD characteristics, lagged points and next-year outcomes (cached)."""
    if not CACHE.exists():
        df = _build()
        PROCESSED.mkdir(parents=True, exist_ok=True)
        tmp = CACHE.with_name(CACHE.name + ".tmp")
        df.to_csv(tmp, index=False, compression="gzip")
        tmp.replace(CACHE)
    p = pd.read_csv(CACHE, keep_default_na=False, na_values=[""], low_memory=False)
    for c in _TXT:
        if c in p:
            p[c] = p[c].fillna("").astype(str)
    return p


def _build() -> pd.DataFrame:
    p = pd.concat([read_year(y) for y in sorted(SPREADSHEETS)], ignore_index=True)
    # A school listed in both 2010-2014 files keeps its high-school row.
    p = p.sort_values("file").drop_duplicates(["year", "district", "school"], keep="first")
    key = ["district", "school"]
    p = p.sort_values(key + ["year"]).reset_index(drop=True)

    p["status_pts"] = p[STATUS].sum(axis=1, min_count=1)
    p["growth_pts"] = p[GROWTH].sum(axis=1, min_count=1)
    p["bonus_pts"] = p.bonus.fillna(0.0)
    p["fcat_points"] = p.status_pts + p.growth_pts + p.bonus_pts
    p["scale_used"] = _scale(p)
    p["points_final"] = np.where(p.year >= 2012, p.points_adjusted,
                                 np.where(p.scale_used == 1600, p.points_total, p.points_pub))
    cut = _cutoffs(p)
    for j, g in enumerate("ABCD"):
        p[f"cut_{g}"] = cut[:, j]
    p["grade_band"] = _band(p.points_final.to_numpy(float), cut)
    p.loc[p.year < 2002, "grade_band"] = ""

    prev = p[key + ["year", "read_l25"]].rename(columns={"read_l25": "read_l25_prev"})
    prev["year"] += 1
    p = p.merge(prev, on=key + ["year"], how="left")
    res = [_grade_row(r) for r in p.itertuples(index=False)]
    p["grade_computed"] = [g for g, _ in res]
    p["reduction"] = [w for _, w in res]

    # CCD school characteristics, same school year.
    ccd = pd.concat([read_ccd(y) for y in sorted(CCD)], ignore_index=True)
    p = p.merge(ccd, on=key + ["year"], how="left")

    # Lagged points (previous year, same scale family) and next-year outcomes.
    p["q_df"] = p.fcat_points - np.array([d_minimum(y) if y >= 2002 else np.nan
                                          for y in p.year])
    lag = p[key + ["year", "q_df"]].rename(columns={"q_df": "q_lag"})
    lag["year"] += 1
    p = p.merge(lag, on=key + ["year"], how="left")
    nxt = pd.DataFrame({
        "district": p.district, "school": p.school, "year": p.year - 1,
        "y_rm": (p.read_prof + p.math_prof) / 2,
        # Means of the components present (2011-12 high schools have no science score).
        "y_points_pct": p[STATUS + GROWTH].mean(axis=1),
        "y_status": p[STATUS].mean(axis=1), "y_growth": p[GROWTH].mean(axis=1),
        "y_grade": p.grade, "y_present": 1.0,
    })
    p = p.merge(nxt, on=key + ["year"], how="left")
    p["y_grade_points"] = p.y_grade.map(GRADE_POINTS)
    p["y_closed"] = np.where(p.year < max(SPREADSHEETS), 1.0 - p.y_present.fillna(0.0), np.nan)
    return p.drop(columns="y_present")


# --------------------------------------------------------------------------- checks

def rule_check() -> pd.DataFrame:
    """By year: published A-F grade against the computed grade (points band plus the
    year's extra conditions), overall and at the D/F line; and the published points
    against the sum of the published components."""
    p = build()
    p = p[p.year >= 2002]
    rows = []
    for y, g in p.groupby("year"):
        m = g.grade.isin(list(GRADES)) & (g.grade_computed != "")
        fcat = g[(g.scale_used <= 800) & g.points_pub.notna() & (g.year <= 2011)]
        df_pub = g.grade[m] == "F"
        df_cmp = g.grade_computed[m] == "F"
        rows.append({
            "year": y, "rows": len(g), "graded": int(g.grade.isin(list(GRADES)).sum()),
            "not_graded": int((~g.grade.isin(list(GRADES))).sum()),
            "points_is_sum": (float((fcat.fcat_points == fcat.points_pub).mean())
                              if len(fcat) else None),
            "agree": float((g.grade[m] == g.grade_computed[m]).mean()),
            "F_published": int(df_pub.sum()), "F_computed": int(df_cmp.sum()),
            "agree_DF": float((df_pub == df_cmp).mean()),
            # F on points alone, ignoring every extra condition.
            "agree_DF_points_only": float(((g.grade_band[m] == "F") == df_pub).mean()),
        })
    return pd.DataFrame(rows)


def df_exceptions() -> pd.DataFrame:
    """Graded school-years whose published F status differs from the computed one."""
    p = build()
    m = p.grade.isin(list(GRADES)) & (p.grade_computed != "")
    bad = m & ((p.grade == "F") != (p.grade_computed == "F"))
    return p.loc[bad, ["year", "district", "school", "name", "school_type", "file", "grade",
                       "grade_computed", "grade_band", "reduction", "grade_prev",
                       "points_final", "scale_used", "pct_tested", "read25_met",
                       "protected"]]


def boundary_mass(width: float = 10, years=MAIN_YEARS) -> pd.DataFrame:
    """School-years in the loaders' sample within `width` points of the D minimum."""
    df = _frame("y_points_pct", years)
    near = df.q_df.abs() <= width                 # |points - D minimum| <= width
    g = df.assign(F=(df.q_df < 0).astype(int), near=near.astype(int),
                  F_near=((df.q_df < 0) & near).astype(int)).groupby("year")
    out = g[["F", "near", "F_near"]].sum()
    out.insert(0, "n", g.size())
    out.loc["total"] = out.sum()
    return out.astype(int)


# -------------------------------------------------------------------------- samples

def _design(df: pd.DataFrame) -> tuple[np.ndarray, list]:
    X = pd.DataFrame(index=df.index)
    mem = df.member.where(df.member > 0)
    frl = (100 * df.frl_count / mem).clip(0, 100)
    nonwhite = (100 * (1 - df.white / df.race_total.where(df.race_total > 0))).clip(0, 100)
    X["pct_frl"] = frl.fillna(frl.median())
    X["frl_missing"] = frl.isna().astype(float)
    X["pct_nonwhite"] = nonwhite.fillna(nonwhite.median())
    X["pct_black"] = (100 * df.black / df.race_total.where(df.race_total > 0)).clip(0, 100)
    X["pct_hispanic"] = (100 * df.hisp / df.race_total.where(df.race_total > 0)).clip(0, 100)
    for c in ("pct_black", "pct_hispanic"):
        X[c] = X[c].fillna(X[c].median())
    X["race_missing"] = nonwhite.isna().astype(float)
    lm = np.log(mem)
    X["log_membership"] = lm.fillna(lm.median())
    X["membership_missing"] = lm.isna().astype(float)
    ptr = (mem / df.fte.where(df.fte > 0)).clip(upper=60)
    X["pupil_teacher"] = ptr.fillna(ptr.median())
    X["pupil_teacher_missing"] = ptr.isna().astype(float)
    X["title1"] = (df.ccd_title1 == "1").astype(float)
    X["magnet"] = (df.ccd_magnet == "1").astype(float)
    X["charter"] = ((df.ccd_charter == "1") | (df.charter == "Y")).astype(float)
    loc = pd.to_numeric(df.ccd_locale, errors="coerce")
    # NCES locale: 1-8 (to 2005-06), 11-43 urban-centric (2006-07 on). City is the base.
    old = {1: "city", 2: "city", 3: "suburb", 4: "suburb", 5: "town", 6: "town",
           7: "rural", 8: "rural"}
    cat = np.where(loc >= 11, loc // 10, np.nan)
    cat = pd.Series(np.where(loc >= 11, pd.Series(cat).map({1: "city", 2: "suburb",
                                                            3: "town", 4: "rural"}),
                             loc.map(old)), index=df.index)
    for c in ("suburb", "town", "rural"):
        X[f"locale_{c}"] = (cat == c).astype(float)
    X["locale_missing"] = cat.isna().astype(float)
    for t in (2, 3, 4):
        X[TYPES[t]] = (df.school_type == t).astype(float)
    for y in sorted(df.year.unique())[1:]:
        X[f"year_{y}"] = (df.year == y).astype(float)
    dist = df.district.where(df.district <= 67, 99)
    for d in sorted(dist.unique()):
        if d != 13:                                   # Miami-Dade is the base
            X[f"district_{d:02d}"] = (dist == d).astype(float)
    X = X.loc[:, X.std() > 0]
    return X.to_numpy(float), list(X.columns)


def _frame(y_col: str, years=MAIN_YEARS) -> pd.DataFrame:
    p = build()
    df = p[p.year.isin(years) & p.grade.isin(list(GRADES)) & (p.scale_used <= 800)
           & p.points_pub.notna() & p.fcat_points.notna() & p[y_col].notna()]
    return df.reset_index(drop=True)


def _extras(df: pd.DataFrame) -> dict:
    return {
        "year": df.year.to_numpy(int), "district": df.district.to_numpy(int),
        "school": df.school.to_numpy(int),
        "school_id": (df.district * 10000 + df.school).to_numpy(int),
        "school_type": df.school_type.to_numpy(float),
        "cutoff_points": np.array([d_minimum(y) for y in df.year], float),
        "points": df.fcat_points.to_numpy(float),
        "status_points": (df.status_pts + df.bonus_pts).to_numpy(float),
        "growth_points": df.growth_pts.to_numpy(float),
        "grade_published": df.grade.to_numpy(str),
        "grade_computed": df.grade_computed.to_numpy(str),
        "grade_prev": df.grade_prev.to_numpy(str),
        "eligible_published": (df.grade == "F").to_numpy(float),
        "pct_tested": df.pct_tested.to_numpy(float),
        "q_lag": df.q_lag.to_numpy(float),
        "voucher_era": (df.year <= 2005).to_numpy(float),
    }


def _citation() -> str:
    return ("Florida Department of Education, School Grades 2001-02 to 2011-12 and Guides "
            "to Calculating School Grades (technical assistance papers); NCES Common Core "
            "of Data, Public School Universe 2001-02 to 2010-11")


def load(y_col: str = "y_points_pct", years=MAIN_YEARS) -> RDDSample:
    """D/F boundary: Q = grade points - D minimum (280 to 2006, 395 from 2007);
    D = 1{Q < 0} = graded F (treated side below the cutoff); Y = next year's percent of
    possible points."""
    df = _frame(y_col, years)
    X, names = _design(df)
    return RDDSample(
        Q=df.q_df.to_numpy(float), X=X, Y=df[y_col].to_numpy(float), threshold=0.0,
        name=f"fl_grades_2002_DF_{y_col}", feature_names=names,
        description=(f"Florida A+ school grades, spring {min(years)}-{max(years)}, D/F "
                     "boundary. Q = grade points minus the D minimum (280 on the 600-point "
                     "scale to 2006, 395 on the 800-point scale from 2007); treatment = "
                     f"1{{Q < 0}} (graded F); Y = {y_col} (next year)."),
        citation=_citation(),
        treatment_rule=lambda Q: (np.asarray(Q) < 0).astype(int),
        extras={**_extras(df), "cutoff": 0.0, "treated_side": "below"},
    )


def load_next_proficiency() -> RDDSample:
    """Y = next year's mean of the reading and math proficiency components (percent at
    FCAT level 3 or above)."""
    return load("y_rm")


def load_next_status() -> RDDSample:
    return load("y_status")


def load_next_growth() -> RDDSample:
    return load("y_growth")


def load_next_grade() -> RDDSample:
    """Y = next year's grade (A=4 ... F=0); ordinal, not a continuous outcome."""
    return load("y_grade_points")


def load_closed() -> RDDSample:
    """Y = 1 if the school has no row in the next year's file (binary)."""
    return load("y_closed")


def load_two_score(era: str = "all", y_col: str = "y_points_pct") -> RDDSample:
    """Status points against growth points at the D/F line.

    Q1 = status points (reading, math, writing and, from 2007, science proficiency, plus
    the high-school retake bonus); Q2 = growth points (learning gains in reading and math
    and of the lowest 25% in reading and, from 2007, math). Grade points = Q1 + Q2, so
    F <=> Q1 + Q2 < c with weights (1, 1). era = "2002-2006" (c = 280) or "2007-2011"
    (c = 395) keeps raw points; era = "all" centres each score at c / 2 so that the rule
    is Q1 + Q2 < 0 in every year (year indicators in X absorb the shift)."""
    years = {"all": MAIN_YEARS, "2002-2006": tuple(range(2002, 2007)),
             "2007-2011": tuple(range(2007, 2012))}[era]
    df = _frame(y_col, years)
    c = np.array([d_minimum(y) for y in df.year], float)
    q1 = (df.status_pts + df.bonus_pts).to_numpy(float)
    q2 = df.growth_pts.to_numpy(float)
    if era == "all":
        q1, q2, cut = q1 - c / 2, q2 - c / 2, 0.0
    else:
        cut = float(c[0])
    X, names = _design(df)
    w = np.array([1.0, 1.0])
    return RDDSample(
        Q=np.column_stack([q1, q2]), X=X, Y=df[y_col].to_numpy(float),
        threshold=(cut / 2, cut / 2), name=f"fl_grades_2002_two_score_{era}_{y_col}",
        feature_names=names,
        description=(f"Florida school grades {min(years)}-{max(years)}, D/F boundary. "
                     "Q = (status points + retake bonus, growth points)"
                     + (" each minus half the D minimum" if era == "all" else "")
                     + f"; treatment = 1{{Q1 + Q2 < {cut:g}}} (graded F); Y = {y_col}."),
        citation=_citation(),
        treatment_rule=lambda Q: (np.asarray(Q) @ w < cut).astype(int),
        extras={**_extras(df), "weights": (1.0, 1.0), "cutoff": cut,
                "treated_side": "below", "era": era},
    )
