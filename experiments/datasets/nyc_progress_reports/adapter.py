"""NYC DOE School Progress Reports, 2006-07 to 2012-13: an A-F grade from the sum of three
category scores, with cut scores by school type and year.

Each elementary, middle and K-8 school ("EMS" report) gets three category scores in
points: School Environment (15), Student Performance (25; 30 in 2006-07) and Student
Progress (60; 55 in 2006-07). Each is a weighted sum of metrics scored by their "percent
of range" against a peer group (75%) and the city (25%; 67/33 in 2006-07). The overall
score is

    overall = Environment + Performance + Progress + additional credit,

and the grade is A/B/C/D/F by cut scores on the overall score that are fixed for each
school type (elementary, K-8, middle) and year. They are set from the score distribution
(percentiles in 2006-07, a set distribution of 25/35/30/7/3% from 2009-10). From 2009-10
"floors" can raise the grade, never lower it: in 2009-10 a school graded A (B) in 2008-09
gets at least C (D), and so does one whose average ELA and math result is in the top 25%
of its type; from 2010-11 a school in the top third gets at least C; in 2011-12 and
2012-13 a school graded A the year before gets at least D. Floors never reach B, so the
A/B and B/C boundaries are sharp in every year.

D and F schools faced "school improvement measures and target setting" and, without
progress, possible leadership change, restructuring or closure (also C for three years
running); A schools got financial rewards (DOE, Consequences page, 2007-10-13).

Q = overall - c (``load()`` and the other single-score loaders), with c the published
    minimum overall score of the higher grade for the school's type and year; Q is on a
    0.01 grid (0.1 from 2007-08) and D = 1{Q >= 0} = the higher grade.
    ``load_two_score()``: Q = (Q1, Q2) with Q1 = 100 (overall - Progress - 0.4 c) / 40 and
    Q2 = 100 (Progress - 0.6 c) / 60, each in percent of its category maximum net of its
    share of the cutoff; 0.4 Q1 + 0.6 Q2 = overall - c exactly.
X = school characteristics not computed from that year's tests: October-register
    demographics (free-lunch, ELL, special-education, Black, Hispanic, Asian and male
    shares, log enrollment, share in tested grades 3-8), the DOE peer index (demographic
    for elementary/K-8; the students' 4th-grade proficiency for middle schools), the
    previous year's overall-score percentile, school type, charter, community school
    district and year indicators.
Y = the next year's overall score, matched by DBN (``load_next_progress``: the next
    year's Progress score; ``load_no_grade_next``: no graded report next year).

Sample: EMS school-years graded A-F with an overall score in 2006-07 to 2011-12 (spring
years 2007-2012), with Y observed. 2012-13 is used only as the outcome of 2011-12: the
Progress Report ended after 2012-13. For the C/D and D/F boundaries, school-years whose
floor could lift the grade to (at least) the higher grade are excluded (``floor``).
"""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample
# Standard-library spreadsheet reader (the environment has no openpyxl).
from experiments.datasets.fl_school_grades._spreadsheet import read, sheet_names
from experiments.datasets.nyc_progress_reports.download import DEMOG, DOE, pr_file

HERE = Path(__file__).parent
PROCESSED = HERE / "data" / "processed"
CACHE = PROCESSED / "nyc_progress_reports_panel.csv.gz"

YEARS = (2007, 2008, 2009, 2010, 2011, 2012, 2013)     # spring year of the school year
MAIN_YEARS = (2007, 2008, 2009, 2010, 2011, 2012)      # rating years with an outcome
TWO_SCORE_YEARS = (2008, 2009, 2010, 2011, 2012)       # category maxima 15 / 25 / 60
LEVELS = ("Elementary", "K-8", "Middle")
GRADES = "ABCDF"
BOUNDARIES = {"AB": 0, "BC": 1, "CD": 2, "DF": 3}
MAIN_BOUNDARY = "CD"     # C or better vs D/F: the grades with consequences
CATEGORY_MAX = {y: ((15, 30, 55) if y == 2007 else (15, 25, 60)) for y in YEARS}
TWO_SCORE_WEIGHTS = (0.4, 0.6)    # (Environment + Performance, Progress) of 100 points
EPS = 1e-9

# Minimum overall score for A, B, C, D by school type (F below D). Sources (README):
# 2007-08 and 2008-09 Educator Guides, Appendix A; 2009-10 Educator Guide, Appendix A;
# 2010-11 to 2012-13: the cut-score table on page 1 of each school's Progress Report.
# 2006-07 (no archived guide): the lowest published score with that grade or better; the
# DOE cut lies in (highest score with a lower grade, this value], see CUTOFF_GAPS_2007.
CUTOFFS = {
    2007: {"Elementary": (64.11, 49.91, 38.79, 31.06), "K-8": (64.33, 50.36, 38.59, 30.08),
           "Middle": (65.22, 50.50, 38.86, 31.27)},
    2008: {"Elementary": (59.6, 45.8, 32.6, 28.4), "K-8": (66.7, 49.7, 38.4, 23.4),
           "Middle": (65.9, 49.7, 39.0, 29.4)},
    2009: {lv: (68.0, 54.0, 43.0, 33.0) for lv in LEVELS},
    2010: {"Elementary": (58.5, 40.7, 29.1, 19.1), "K-8": (57.1, 41.2, 29.5, 18.0),
           "Middle": (62.1, 42.6, 30.6, 18.5)},
    2011: {"Elementary": (56.7, 40.6, 25.7, 18.3), "K-8": (57.1, 41.3, 25.3, 14.5),
           "Middle": (58.2, 41.5, 23.8, 14.0)},
    2012: {"Elementary": (64.7, 49.8, 35.8, 25.6), "K-8": (64.1, 51.8, 37.6, 30.0),
           "Middle": (65.5, 51.7, 36.3, 23.2)},
    2013: {"Elementary": (60.1, 48.8, 36.5, 30.0), "K-8": (63.1, 53.2, 40.6, 32.0),
           "Middle": (67.7, 53.0, 37.3, 29.1)},
}
CUTOFF_GAPS_2007 = {   # (highest score with the lower grade, lowest with the higher)
    "Elementary": ((63.99, 64.11), (49.72, 49.91), (38.64, 38.79), (30.90, 31.06)),
    "K-8": ((63.10, 64.33), (49.76, 50.36), (38.04, 38.59), (29.35, 30.08)),
    "Middle": ((65.21, 65.22), (50.24, 50.50), (38.81, 38.86), (29.27, 31.27)),
}
# Share of each school type whose average ELA and math result earns the C floor.
FLOOR_TOP_SHARE = {2010: 0.25, 2011: 1 / 3, 2012: 1 / 3, 2013: 1 / 3}
# The DOE result-file columns behind "average ELA and math": median proficiency in
# 2009-10 (the only proficiency level in that file), average proficiency from 2010-11.
_PROF_COLS = {y: ("ELA Median Proficiency", "Math Median Proficiency") if y == 2010
              else ("ELA Average Proficiency", "Math Average Proficiency")
              for y in FLOOR_TOP_SHARE}

_COLS = {
    "DBN": "dbn", "DISTRICT": "district", "SCHOOL": "name",
    "PROGRESS REPORT TYPE": "type", "SCHOOL LEVEL*": "level", "PEER INDEX*": "peer_index",
    "GRADE": "grade", "OVERALL GRADE": "grade", "OVERALL SCORE": "overall",
    "ENVIRONMENT CATEGORY SCORE": "env", "PERFORMANCE CATEGORY SCORE": "perf",
    "PROGRESS CATEGORY SCORE": "prog", "ADDITIONAL CREDIT": "ac",
}
_NUM = ["peer_index", "overall", "env", "perf", "prog", "ac"]
_TXT = ["dbn", "district", "name", "type", "level", "grade", "prior_grade", "implied",
        "floor", "grade_rule", "y_grade", "y_level"]


def _label(year: int) -> str:
    return f"{year - 1}-{str(year)[2:]}"


# ----------------------------------------------------------------------- raw files

def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.strip(), errors="coerce")


def _read_year(year: int) -> pd.DataFrame:
    """One school year's Open Data summary: EMS reports of elementary, K-8 and middle
    schools, category scores in points."""
    path = pr_file(year)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run python -m experiments.datasets.nyc_progress_reports.download")
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    raw.columns = [" ".join(c.split()) for c in raw.columns]
    prior = f"{_label(year - 1)} PROGRESS REPORT GRADE"
    out = pd.DataFrame({"year": year}, index=raw.index)
    out["prior_grade"] = raw[prior].str.strip() if prior in raw else ""
    names = {c: re.sub(r"^\d{4}-\d{2,4}\s*", "", c) for c in raw.columns if c != prior}
    for c, short in names.items():
        if short in _COLS:
            out[_COLS[short]] = raw[c].str.strip()
    for c in _NUM:
        out[c] = _num(out[c]) if c in out else np.nan
    out["type"] = out["type"].replace({"ESMS": "EMS"})
    out["level"] = out["level"].str.replace(" School", "", regex=False).str.strip()
    out = out[(out["type"] == "EMS") & out["level"].isin(LEVELS)].copy()
    if year == 2007:         # 2006-07 category scores are fractions of the category
        for c, m in zip(["env", "perf", "prog"], CATEGORY_MAX[2007]):
            out[c] = out[c] * m
    out = out.drop_duplicates()
    # 2012-13 lists 04M206 twice (Elementary, with prior grades; K-8, without) with the
    # same scores; keep the first row of a DBN when the scores and grade agree.
    same = out.duplicated(["dbn", "grade", "overall", "env", "perf", "prog", "ac"])
    out = out[~same]
    if out.dbn.duplicated().any():
        raise ValueError(f"{path}: duplicate EMS rows for "
                         f"{sorted(out.dbn[out.dbn.duplicated()])}")
    return out.reset_index(drop=True)


def _demographics() -> pd.DataFrame:
    d = pd.read_csv(DEMOG, dtype=str, keep_default_na=False)
    out = pd.DataFrame({"dbn": d.DBN.str.strip(),
                        "year": d.schoolyear.str[4:].astype(int)})
    for c in ["total_enrollment", "fl_percent", "frl_percent", "ell_percent",
              "sped_percent", "black_per", "hispanic_per", "asian_per", "white_per",
              "male_per"] + [f"grade{g}" for g in range(3, 9)]:
        out[c] = _num(d[c])
    # Free lunch is reported through 2008-09, free or reduced-price lunch from 2009-10.
    out["poverty"] = out.fl_percent.fillna(out.frl_percent)
    out["tested_share"] = out[[f"grade{g}" for g in range(3, 9)]].sum(axis=1, min_count=1) \
        / out.total_enrollment.where(out.total_enrollment > 0)
    return out.drop(columns=[f"grade{g}" for g in range(3, 9)] + ["fl_percent",
                                                                  "frl_percent"])


def _proficiency(year: int) -> pd.DataFrame:
    """DBN and the ELA and math proficiency behind the 'top of its type' floor."""
    path = DOE[year]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run python -m experiments.datasets.nyc_progress_reports.download")
    names = sheet_names(path)
    rows = read(path, names.index("Student Performance") if "Student Performance" in names
                else 0)
    h = next(k for k, r in enumerate(rows) if r and str(r[0] or "").strip() == "DBN")
    hdr = [str(x).strip() if x is not None else "" for x in rows[h]]
    je, jm = (hdr.index(c) for c in _PROF_COLS[year])
    recs = [(str(r[0]).strip(), r[je] if je < len(r) else None, r[jm] if jm < len(r) else None)
            for r in rows[h + 1:] if r and r[0]]
    out = pd.DataFrame(recs, columns=["dbn", "prof_ela", "prof_math"])
    out["prof_ela"], out["prof_math"] = _num(out.prof_ela), _num(out.prof_math)
    out["year"] = year
    return out.drop_duplicates("dbn")


# ---------------------------------------------------------------------------- rule

def _letter(score: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    """Grade from the overall score and per-row cut scores (n, 4)."""
    k = (score[:, None] < cuts - EPS).sum(axis=1)      # number of cutoffs not reached
    out = np.array(list(GRADES))[k]
    return np.where(np.isnan(score), "", out)


def _cut_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.array([CUTOFFS[y][lv] for y, lv in zip(df.year, df.level)], dtype=float)


def state_rule(df: pd.DataFrame) -> pd.DataFrame:
    """The grade from the overall score (``implied``), the floor that applies
    (``floor``: "C", "D" or ""), and the grade after the floor (``grade_rule``)."""
    out = pd.DataFrame(index=df.index)
    out["implied"] = _letter(df.overall.to_numpy(float), _cut_matrix(df))
    graded = df.grade.isin(list(GRADES)) & df.overall.notna()
    avg = (df.prof_ela + df.prof_math) / 2
    top = pd.Series(False, index=df.index)
    for (y, lv), g in df[graded & avg.notna()].groupby(["year", "level"]):
        if y in FLOOR_TOP_SHARE:
            a = avg[g.index]
            top[g.index] = a >= a.quantile(1 - FLOOR_TOP_SHARE[y])
    out["prof_top"] = np.where(df.year.isin(list(FLOOR_TOP_SHARE)), top.astype(float), np.nan)
    floor_c = top | ((df.year == 2010) & (df.prior_grade == "A"))
    floor_d = (((df.year == 2010) & (df.prior_grade == "B"))
               | (df.year.isin([2012, 2013]) & (df.prior_grade == "A")))
    out["floor"] = np.where(floor_c, "C", np.where(floor_d, "D", ""))
    rank = {g: i for i, g in enumerate(GRADES)}
    out["grade_rule"] = [GRADES[min(rank[a], rank[f] if f else 9)] if a else ""
                         for a, f in zip(out.implied, out.floor)]
    return out


# --------------------------------------------------------------------------- panel

def build() -> pd.DataFrame:
    """Every EMS school-year 2006-07 to 2012-13 with scores, the rule, demographics, the
    previous year's percentile and the next year's outcomes. Cached in
    ``data/processed`` (written to a temporary file, then renamed)."""
    if not CACHE.exists():
        df = _build()
        PROCESSED.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=PROCESSED, prefix=CACHE.name, suffix=".tmp")
        os.close(fd)
        try:
            df.to_csv(tmp, index=False, compression="gzip")
            os.chmod(tmp, 0o644)
            os.replace(tmp, CACHE)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    p = pd.read_csv(CACHE, keep_default_na=False, na_values=[""], low_memory=False,
                    dtype={c: str for c in _TXT})
    for c in _TXT:
        p[c] = p[c].fillna("").astype(str)
    return p


def _build() -> pd.DataFrame:
    p = pd.concat([_read_year(y) for y in YEARS], ignore_index=True)
    p = p.merge(_demographics(), on=["dbn", "year"], how="left")
    prof = pd.concat([_proficiency(y) for y in FLOOR_TOP_SHARE], ignore_index=True)
    p = p.merge(prof, on=["dbn", "year"], how="left")
    cuts = _cut_matrix(p)
    for j, g in enumerate("ABCD"):
        p[f"cut_{g}"] = cuts[:, j]
    p = pd.concat([p, state_rule(p)], axis=1)
    p["graded"] = (p.grade.isin(list(GRADES)) & p.overall.notna()).astype(float)
    p["charter"] = p.dbn.str.startswith("84").astype(float)
    p["csd"] = p.dbn.str[:2].astype(int)
    # Percentile of the overall score within year and school type (graded schools).
    g = p.graded == 1
    p["pct"] = np.nan
    p.loc[g, "pct"] = p[g].groupby(["year", "level"]).overall.rank(pct=True) * 100
    # Next year's report (EMS, elementary / K-8 / middle), matched by DBN.
    nxt = p[g][["dbn", "year", "overall", "prog", "perf", "env", "grade", "level", "pct"]]
    nxt = nxt.rename(columns={"overall": "y_overall", "prog": "y_prog", "perf": "y_perf",
                              "env": "y_env", "grade": "y_grade", "level": "y_level",
                              "pct": "y_pct"})
    nxt["year"] -= 1
    p = p.merge(nxt, on=["dbn", "year"], how="left")
    p["y_grade"] = p.y_grade.fillna("")
    p["y_level"] = p.y_level.fillna("")
    p["y_year"] = np.where(p.year < max(YEARS), p.year + 1, np.nan)
    # No graded report next year: closed, phasing out (no grade), too few students, or
    # changed to another report type.
    p["y_no_grade"] = np.where(p.year < max(YEARS), p.y_overall.isna().astype(float), np.nan)
    # Previous year's percentile and score (same DBN).
    lag = p[g][["dbn", "year", "pct", "overall"]].rename(
        columns={"pct": "pct_lag", "overall": "q_lag"})
    lag["year"] += 1
    p = p.merge(lag, on=["dbn", "year"], how="left")
    return p.sort_values(["year", "dbn"]).reset_index(drop=True)


# -------------------------------------------------------------------------- checks

def rule_check() -> pd.DataFrame:
    """Per year (graded EMS schools): agreement of the published grade with (a) the
    grade from the overall score alone and (b) that grade after the floors; agreement
    of 1{overall >= c} with the published grade at each boundary, all schools and
    after dropping the school-years whose floor reaches the higher grade; and the
    overall score against the published category scores."""
    p = build()
    p = p[p.graded == 1]
    rows = []
    for y, g in p.groupby("year"):
        r = {"year": y, "graded": len(g),
             "sum_within_0.2": float(((g.env + g.perf + g.prog + g.ac.fillna(0)
                                       - g.overall).abs() <= 0.2 + EPS).mean()),
             "agree_score_only": float((g.implied == g.grade).mean()),
             "agree_with_floors": float((g.grade_rule == g.grade).mean()),
             "raised_by_floor": int((g.grade_rule != g.implied).sum()),
             "published_above_score": int((g.grade < g.implied).sum())}
        for b, j in BOUNDARIES.items():
            hi = list(GRADES[:j + 1])
            d = g.overall.to_numpy() >= g[f"cut_{GRADES[j]}"].to_numpy() - EPS
            pub = g.grade.isin(hi).to_numpy()
            keep = ~_floor_excluded(g, j)
            r[f"{b}_all"] = float((d == pub).mean())
            r[f"{b}_kept"] = float((d == pub)[keep].mean())
            r[f"{b}_dropped"] = int((~keep).sum())
        rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------- samples

def _floor_excluded(df: pd.DataFrame, j: int) -> np.ndarray:
    """True where a floor guarantees at least the higher grade of boundary j."""
    fl = df.floor.to_numpy(str)
    hi = set(GRADES[:j + 1])
    return np.isin(fl, list(hi))


def _design(df: pd.DataFrame) -> tuple[np.ndarray, list]:
    X = pd.DataFrame(index=df.index)

    def fill(name, v):
        X[name] = v.fillna(v.median())
        if v.isna().any():
            X[f"{name}_missing"] = v.isna().astype(float)

    fill("pct_poverty", df.poverty)
    fill("pct_ell", df.ell_percent)
    fill("pct_sped", df.sped_percent)
    fill("pct_black", df.black_per)
    fill("pct_hispanic", df.hispanic_per)
    fill("pct_asian", df.asian_per)
    fill("pct_male", df.male_per)
    fill("log_enrollment", np.log(df.total_enrollment.where(df.total_enrollment > 0)))
    fill("tested_share", df.tested_share)
    em = df.level != "Middle"
    fill("peer_index_em", df.peer_index.where(em))           # 0-100, demographic
    X.loc[~em, "peer_index_em"] = 0.0
    fill("peer_index_middle", df.peer_index.where(~em))      # 1-4.5, 4th-grade results
    X.loc[em, "peer_index_middle"] = 0.0
    fill("pct_lag", df.pct_lag)
    X["k8"] = (df.level == "K-8").astype(float)
    X["middle"] = (df.level == "Middle").astype(float)
    X["charter"] = df.charter.astype(float)
    for d in sorted(df.csd.unique()):
        if d not in (2, 84):                       # district 2 is the base; 84 = charter
            X[f"csd_{d:02d}"] = (df.csd == d).astype(float)
    for y in sorted(df.year.unique())[1:]:
        X[f"year_{y}"] = (df.year == y).astype(float)
    X = X.loc[:, X.std() > 0]
    return X.to_numpy(float), list(X.columns)


def _frame(j: int, y_col: str, years) -> pd.DataFrame:
    p = build()
    m = p.year.isin(years) & (p.graded == 1) & p[y_col].notna()
    p = p[m]
    return p[~_floor_excluded(p, j)].reset_index(drop=True)


def _extras(df: pd.DataFrame, j: int) -> dict:
    c = df[f"cut_{GRADES[j]}"].to_numpy(float)
    group = (df.year.astype(str) + "-" + df.level).to_numpy()
    return {
        "group": group,
        "group_cutoffs": {g: float(v) for g, v in sorted(dict(zip(group, c)).items())},
        "boundary": "".join(GRADES[j:j + 2]), "cutoff_overall": c,
        "year": df.year.to_numpy(int), "level": df.level.to_numpy(str),
        "dbn": df.dbn.to_numpy(str), "csd": df.csd.to_numpy(int),
        "overall": df.overall.to_numpy(float), "environment": df.env.to_numpy(float),
        "performance": df.perf.to_numpy(float), "progress": df.prog.to_numpy(float),
        "additional_credit": df.ac.to_numpy(float),
        "grade_published": df.grade.to_numpy(str),
        "eligible_published": df.grade.isin(list(GRADES[:j + 1])).to_numpy(float),
        "floor": df.floor.to_numpy(str), "prior_grade": df.prior_grade.to_numpy(str),
        "y_year": df.y_year.to_numpy(float), "y_level_changed":
            (df.y_level.ne("") & df.y_level.ne(df.level)).to_numpy(float),
        "q_lag": df.q_lag.to_numpy(float), "cutoff_table": CUTOFFS,
    }


def _sample(boundary: str = MAIN_BOUNDARY, y_col: str = "y_overall") -> RDDSample:
    j = BOUNDARIES[boundary]
    df = _frame(j, y_col, MAIN_YEARS)
    X, names = _design(df)
    hi, lo = GRADES[j], GRADES[j + 1]
    q = np.round(df.overall.to_numpy(float) - df[f"cut_{hi}"].to_numpy(float), 2)
    return RDDSample(
        Q=q, X=X, Y=df[y_col].to_numpy(float), threshold=0.0,
        name=f"nyc_progress_reports_{boundary}_{y_col}", feature_names=names,
        description=(f"NYC DOE School Progress Reports (elementary, K-8, middle), "
                     f"{_label(MAIN_YEARS[0])} to {_label(MAIN_YEARS[-1])}. Q = overall "
                     f"score - the {hi} cut score of the school's type and year; treatment "
                     f"= 1{{Q >= 0}} (grade {hi} or better rather than {lo}); Y = {y_col} "
                     "(next year)."),
        citation="NYC DOE School Progress Reports (NYC Open Data) and Educator Guides "
                 "2007-08 to 2012-13; Rockoff and Turner (2010, AEJ: Economic Policy)",
        treatment_rule=lambda Q: (np.asarray(Q) >= -EPS).astype(int),
        extras=_extras(df, j),
    )


def load() -> RDDSample:
    """C or better vs D/F (the grades with consequences); Y = next year's overall score."""
    return _sample(MAIN_BOUNDARY)


def load_cd() -> RDDSample:
    return _sample("CD")


def load_ab() -> RDDSample:
    """A vs B (no floor reaches B, so every graded school-year is kept)."""
    return _sample("AB")


def load_bc() -> RDDSample:
    return _sample("BC")


def load_df() -> RDDSample:
    return _sample("DF")


def load_next_progress() -> RDDSample:
    """C or better vs D/F; Y = next year's Student Progress category score (points)."""
    return _sample(MAIN_BOUNDARY, "y_prog")


def load_no_grade_next() -> RDDSample:
    """C or better vs D/F; Y = 1 if the school has no graded EMS report the next year
    (closure, phase-out, too few students, or another report type). D/F schools often
    have no next-year grade, so ``load()``'s outcome is missing selectively."""
    return _sample(MAIN_BOUNDARY, "y_no_grade")


def load_two_score(boundary: str = MAIN_BOUNDARY) -> RDDSample:
    """Q = (Q1, Q2), Q1 = 100 (overall - Progress - 0.4 c) / 40 (Environment + Performance
    + additional credit), Q2 = 100 (Progress - 0.6 c) / 60, both in percent of the
    category maximum net of the category's share of the cut score c; treatment =
    1{0.4 Q1 + 0.6 Q2 >= 0} = 1{overall >= c}. Spring years 2008-2012 (category maxima
    15 / 25 / 60); same X and Y as ``load()``."""
    j = BOUNDARIES[boundary]
    df = _frame(j, "y_overall", TWO_SCORE_YEARS)
    df = df[df.prog.notna()].reset_index(drop=True)    # one school-year has no Progress
    X, names = _design(df)
    c = df[f"cut_{GRADES[j]}"].to_numpy(float)
    o, pr = df.overall.to_numpy(float), df.prog.to_numpy(float)
    Q = np.column_stack([100 * (o - pr - 0.4 * c) / 40, 100 * (pr - 0.6 * c) / 60])
    w = np.array(TWO_SCORE_WEIGHTS)
    ex = _extras(df, j)
    ex.update({"weights": TWO_SCORE_WEIGHTS, "cutoff": 0.0,
               "q_units": "percent of category maximum, net of the category's share of "
                          "the cut score (0.4 c, 0.6 c)"})
    return RDDSample(
        Q=Q, X=X, Y=df.y_overall.to_numpy(float), threshold=(0.0, 0.0),
        name=f"nyc_progress_reports_two_score_{boundary}", feature_names=names,
        description=("NYC DOE School Progress Reports, 2007-08 to 2011-12. Q1 = Environment"
                     " + Performance + additional credit, Q2 = Progress, each in percent of"
                     " its maximum (40, 60) net of its share of the cut score; treatment = "
                     "1{0.4 Q1 + 0.6 Q2 >= 0}; Y = next year's overall score."),
        citation="NYC DOE School Progress Reports (NYC Open Data) and Educator Guides",
        # Tolerance: 0.4 Q1 + 0.6 Q2 equals overall - c up to floating point.
        treatment_rule=lambda Q: (np.asarray(Q) @ w >= -1e-6).astype(int),
        extras=ex,
    )
